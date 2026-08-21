/*
 * KVM-backed user-mode execution (x86-64 guest on x86-64 host).
 *
 * Instead of TCG-translating the guest, we run it natively inside a KVM VM at
 * ring 0 (long mode, CPL0).  The guest address space is identity-mapped:
 * guest-virtual == guest-physical == host-virtual (guest_base == 0).  KVM
 * memslots mirror linux-user's actual guest mappings, so a guest access to
 * mapped memory GUPs the identical host page (it just works), while an access to
 * a host-unmapped or PROT_NONE page inside a slot fails GUP and returns a bare
 * -EFAULT from KVM_RUN with the guest RIP left at the faulting instruction.
 *
 * (Blanket slots covering the whole address space are NOT viable: KVM eagerly
 * allocates per-page rmap/lpage_info metadata proportional to slot size, so a
 * 128 TiB blanket would need terabytes of host RAM.  Mirroring keeps metadata
 * proportional to memory the guest actually uses.)
 *
 * A tiny host-generated ring-0 "nanokernel" (GDT/IDT/TSS + one-instruction
 * out-stubs) turns every syscall and CPU exception into a KVM_EXIT_IO: MSR_LSTAR
 * points at `out %al,$SYSCALL_PORT`, and each IDT gate is `out %al,$vector`.
 * `out` clobbers nothing, so on the exit we read pristine guest state straight
 * from the KVM_CAP_SYNC_REGS mmap page and, after handling, write the resume RIP
 * back the same way (the kernel's complete_fast_pio_out kvm_is_linear_rip guard
 * preserves our RIP).  The nanokernel + guest page tables live in one "control"
 * memslot at a high GPA, above the [0, 2^47) user range, so guest mappings can
 * never collide with them.
 *
 * M1: single vCPU, mirror the mappings present at load time, syscall + basic
 * exception decode.  Dynamic mmap tracking, threads, fork, signals and precise
 * fault addresses come in later milestones.
 */

#include "qemu/osdep.h"
#include "qemu.h"
#include "user-internals.h"
#include "signal-common.h"
#include "user/guest-base.h"
#include "user/page-protection.h"
#include "kvm-user.h"

#include <linux/kvm.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

bool kvm_user_enabled;

static int kvm_debug;
#define DBG(...) do { if (kvm_debug) { fprintf(stderr, "qemu-kvm: " __VA_ARGS__); } } while (0)

/* ------------------------------------------------------------------ */
/* Control region: guest page tables + nanokernel, at a high GPA.      */
/* ------------------------------------------------------------------ */

#define CTRL_GPA     0x800000000000ULL     /* 2^47, above the user range     */
#define CTRL_SIZE    0x00800000ULL         /* 8 MiB                          */
#define PT_OFF       0x00000000ULL         /* page tables at ctrl + 0        */
#define NANO_OFF     0x00400000ULL         /* nanokernel at ctrl + 4 MiB     */
#define NANO_GPA     (CTRL_GPA + NANO_OFF)
#define NANO_SIZE    0x00200000ULL         /* one 2 MiB page                 */

#define NANO_VA_BASE 0xffffffff80000000ULL /* kernel-half VA of the nanokernel */

/* nanokernel sub-layout (offsets within the nanokernel area) */
#define OFF_GDT          0x0000
#define OFF_IDT          0x1000            /* 256 * 16                        */
#define OFF_TSS          0x2000
#define OFF_STUBS        0x3000            /* 256 * 4                         */
#define OFF_SYSCALL_STUB 0x3400
#define OFF_IST1_TOP     0x8000            /* fault stack top                 */

#define SEL_CODE 0x08
#define SEL_DATA 0x10
#define SEL_TSS  0x18

#define SYSCALL_PORT 0x40

/* MSRs */
#define MSR_EFER       0xc0000080
#define MSR_STAR       0xc0000081
#define MSR_LSTAR      0xc0000082
#define MSR_SFMASK     0xc0000084
#define MSR_FS_BASE    0xc0000100
#define MSR_GS_BASE    0xc0000101

/* CR/EFER bits */
#define CR0_PE 0x1
#define CR0_MP 0x2
#define CR0_ET 0x10
#define CR0_NE 0x20
#define CR0_WP 0x10000
#define CR0_PG 0x80000000ULL
#define CR4_PAE        0x20
#define CR4_OSFXSR     0x200
#define CR4_OSXMMEXCPT 0x400
#define CR4_FSGSBASE   0x10000
#define EFER_SCE 0x1
#define EFER_LME 0x100
#define EFER_LMA 0x400
#define EFER_NXE 0x800

#define PTE_P  0x1
#define PTE_RW 0x2
#define PTE_PS 0x80

/* x86-64 user address ceiling (TASK_SIZE); regions at/above this (e.g. the
 * vsyscall page) are not real host mappings and are handled specially. */
#define USER_VA_END  0x00007ffffffff000ULL

/* ------------------------------------------------------------------ */
/* Global VM state (M1: one VM, one vCPU)                             */
/* ------------------------------------------------------------------ */

static int kvm_fd = -1;
static int vm_fd = -1;
static uint8_t *ctrl;                 /* host mapping of the control region */
static uint8_t *nano;                 /* = ctrl + NANO_OFF                  */
static uint64_t cr3_gpa;              /* guest-physical of PML4             */
static uint32_t next_slot;

/* fs/gs base last pushed to the vCPU, to avoid needless MSR writes. */
static uint64_t cached_fs_base = ~0ULL, cached_gs_base = ~0ULL;

static int kvm_ioctl(int fd, int req, void *arg, const char *what)
{
    int r = ioctl(fd, req, arg);
    if (r < 0) {
        fprintf(stderr, "qemu-kvm: %s failed: %s\n", what, strerror(errno));
        _exit(1);
    }
    return r;
}

/* ------------------------------------------------------------------ */
/* Memory: control slot + per-mapping mirror slots                    */
/* ------------------------------------------------------------------ */

static void add_memslot(uint32_t slot, uint64_t gpa, uint64_t hva, uint64_t size)
{
    struct kvm_userspace_memory_region region = {
        .slot = slot,
        .flags = 0,
        .guest_phys_addr = gpa,
        .memory_size = size,
        .userspace_addr = hva,
    };
    kvm_ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &region,
              "KVM_SET_USER_MEMORY_REGION");
}

/* walk_memory_regions callback: mirror each guest page-flags region as a slot.
 * guest_base == 0, so guest-physical == userspace address (identity). */
static int mirror_region_cb(void *priv, vaddr start, vaddr end, int flags)
{
    /* Skip kernel-half / non-backed regions (e.g. the vsyscall page): their
     * identity userspace_addr is not a valid host mapping. */
    if (start >= USER_VA_END) {
        DBG("  skip: [0x%llx, 0x%llx) flags=0x%x\n",
            (unsigned long long)start, (unsigned long long)end, flags);
        return 0;
    }
    DBG("  slot: [0x%llx, 0x%llx) flags=0x%x\n",
        (unsigned long long)start, (unsigned long long)end, flags);
    add_memslot(next_slot++, start, start, end - start);
    return 0;
}

static void setup_memory(void)
{
    /* Control region (page tables + nanokernel) at its high GPA. */
    ctrl = mmap(NULL, CTRL_SIZE, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ctrl == MAP_FAILED) {
        perror("qemu-kvm: mmap control region");
        _exit(1);
    }
    nano = ctrl + NANO_OFF;
    add_memslot(next_slot++, CTRL_GPA, (uint64_t)(uintptr_t)ctrl, CTRL_SIZE);

    /* Mirror the guest mappings that exist at load time. */
    walk_memory_regions(NULL, mirror_region_cb);
}

/* ------------------------------------------------------------------ */
/* Static identity guest page tables (in the control region)          */
/* ------------------------------------------------------------------ */

static void setup_page_tables(void)
{
    /*
     * PML4[0..255] -> 256 PDPTs of 512 * 1GiB identity pages, covering all of
     * [0, 2^47).  PML4[511] -> a chain mapping the nanokernel's kernel-half VA
     * to NANO_GPA.  Tables live in the control region; a table page at control
     * offset O has guest-physical address CTRL_GPA + O.
     */
    const int NUSER = 256;
    uint8_t *base = ctrl + PT_OFF;
    memset(base, 0, (2 + NUSER + 1) * 4096);

    uint64_t *pml4  = (uint64_t *)(base + 0);
    uint64_t *pdptk = (uint64_t *)(base + (1 + NUSER) * 4096);
    uint64_t *pdk   = (uint64_t *)(base + (2 + NUSER) * 4096);
    uint64_t gpa_base = CTRL_GPA + PT_OFF;

    for (int i = 0; i < NUSER; i++) {
        uint64_t *pdpt = (uint64_t *)(base + (1 + i) * 4096);
        pml4[i] = (gpa_base + (uint64_t)(1 + i) * 4096) | PTE_P | PTE_RW;
        for (int j = 0; j < 512; j++) {
            uint64_t va = ((uint64_t)i * 512 + j) << 30;   /* 1GiB stride */
            pdpt[j] = va | PTE_P | PTE_RW | PTE_PS;
        }
    }

    /* Kernel-half chain for the nanokernel VA: PML4[511]->PDPTk[510]->PDk[0]. */
    pml4[511]  = (gpa_base + (uint64_t)(1 + NUSER) * 4096) | PTE_P | PTE_RW;
    pdptk[510] = (gpa_base + (uint64_t)(2 + NUSER) * 4096) | PTE_P | PTE_RW;
    pdk[0]     = NANO_GPA | PTE_P | PTE_RW | PTE_PS;         /* 2MiB page */

    cr3_gpa = gpa_base;   /* PML4 is at control offset 0 */
}

/* ------------------------------------------------------------------ */
/* Nanokernel (GDT / IDT / TSS / out-stubs)                           */
/* ------------------------------------------------------------------ */

static void set_gdt_entry(uint64_t off, uint64_t val)
{
    *(uint64_t *)(nano + OFF_GDT + off) = val;
}

static void set_tss_desc(uint64_t off, uint64_t base, uint32_t limit)
{
    uint8_t *d = nano + OFF_GDT + off;
    memset(d, 0, 16);
    d[0] = limit & 0xff;
    d[1] = (limit >> 8) & 0xff;
    d[2] = base & 0xff;
    d[3] = (base >> 8) & 0xff;
    d[4] = (base >> 16) & 0xff;
    d[5] = 0x89;                    /* present | available 64-bit TSS (type 9) */
    d[6] = (limit >> 16) & 0x0f;
    d[7] = (base >> 24) & 0xff;
    *(uint32_t *)(d + 8) = (base >> 32) & 0xffffffff;
}

static void set_idt_gate(int vec, uint64_t handler_va)
{
    uint8_t *e = nano + OFF_IDT + vec * 16;
    memset(e, 0, 16);
    *(uint16_t *)(e + 0) = handler_va & 0xffff;
    *(uint16_t *)(e + 2) = SEL_CODE;
    e[4] = 1;                       /* IST1 */
    e[5] = 0x8e;                    /* present | DPL0 | 64-bit interrupt gate */
    *(uint16_t *)(e + 6) = (handler_va >> 16) & 0xffff;
    *(uint32_t *)(e + 8) = (handler_va >> 32) & 0xffffffff;
}

static void setup_nanokernel(void)
{
    memset(nano, 0, NANO_SIZE);

    /* GDT: null, code64, data, TSS. */
    set_gdt_entry(0x00, 0);
    set_gdt_entry(SEL_CODE, 0x00af9a000000ffffULL);  /* P DPL0 code L=1 G=1 */
    set_gdt_entry(SEL_DATA, 0x00cf92000000ffffULL);  /* P DPL0 data D/B=1 G=1 */
    set_tss_desc(SEL_TSS, NANO_VA_BASE + OFF_TSS, 0x67);

    /* TSS with IST1 -> fault stack top. */
    *(uint64_t *)(nano + OFF_TSS + 36) = NANO_VA_BASE + OFF_IST1_TOP;
    *(uint16_t *)(nano + OFF_TSS + 102) = 104;        /* IOPB offset (none) */

    /* Per-vector stubs: `out %al,$vec` (E6 vec), stride 4, + hlt guard. */
    for (int v = 0; v < 256; v++) {
        uint8_t *s = nano + OFF_STUBS + v * 4;
        s[0] = 0xe6; s[1] = (uint8_t)v; s[2] = 0xf4; s[3] = 0x90;
        set_idt_gate(v, NANO_VA_BASE + OFF_STUBS + v * 4);
    }

    /* Syscall stub at LSTAR: `out %al,$SYSCALL_PORT` + hlt guard. */
    {
        uint8_t *s = nano + OFF_SYSCALL_STUB;
        s[0] = 0xe6; s[1] = SYSCALL_PORT; s[2] = 0xf4;
    }
}

/* ------------------------------------------------------------------ */
/* vCPU register / CPUID / MSR setup                                  */
/* ------------------------------------------------------------------ */

static void kvm_seg(struct kvm_segment *s, uint16_t sel, uint8_t type,
                    uint8_t s_bit, uint8_t l, uint8_t db)
{
    memset(s, 0, sizeof(*s));
    s->base = 0;
    s->limit = 0xffffffff;
    s->selector = sel;
    s->type = type;
    s->present = 1;
    s->dpl = 0;
    s->s = s_bit;
    s->l = l;
    s->db = db;
    s->g = 1;
}

static void setup_sregs(CPUState *cs, CPUX86State *env)
{
    struct kvm_sregs sregs;
    kvm_ioctl(cs->kvm_fd, KVM_GET_SREGS, &sregs, "KVM_GET_SREGS");

    kvm_seg(&sregs.cs, SEL_CODE, 11, 1, 1, 0);
    kvm_seg(&sregs.ds, SEL_DATA, 3, 1, 0, 1);
    sregs.es = sregs.fs = sregs.gs = sregs.ss = sregs.ds;
    /* Preserve any FS/GS base already established (e.g. by ld.so TLS). */
    sregs.fs.base = env->segs[R_FS].base;
    sregs.gs.base = env->segs[R_GS].base;

    memset(&sregs.tr, 0, sizeof(sregs.tr));
    sregs.tr.base = NANO_VA_BASE + OFF_TSS;
    sregs.tr.limit = 0x67;
    sregs.tr.selector = SEL_TSS;
    sregs.tr.type = 11;             /* busy 64-bit TSS */
    sregs.tr.present = 1;

    sregs.gdt.base = NANO_VA_BASE + OFF_GDT;
    sregs.gdt.limit = 0x3f;
    sregs.idt.base = NANO_VA_BASE + OFF_IDT;
    sregs.idt.limit = 0xfff;

    sregs.cr0 = CR0_PE | CR0_MP | CR0_ET | CR0_NE | CR0_WP | CR0_PG;
    sregs.cr3 = cr3_gpa;
    sregs.cr4 = CR4_PAE | CR4_OSFXSR | CR4_OSXMMEXCPT | CR4_FSGSBASE;
    sregs.efer = EFER_SCE | EFER_LME | EFER_LMA | EFER_NXE;
    sregs.cr8 = 0;

    kvm_ioctl(cs->kvm_fd, KVM_SET_SREGS, &sregs, "KVM_SET_SREGS");
    cached_fs_base = env->segs[R_FS].base;
    cached_gs_base = env->segs[R_GS].base;
}

static void set_msr(CPUState *cs, uint32_t index, uint64_t data)
{
    struct {
        struct kvm_msrs h;
        struct kvm_msr_entry e[1];
    } m = { 0 };
    m.h.nmsrs = 1;
    m.e[0].index = index;
    m.e[0].data = data;
    kvm_ioctl(cs->kvm_fd, KVM_SET_MSRS, &m, "KVM_SET_MSRS");
}

static void setup_msrs(CPUState *cs)
{
    set_msr(cs, MSR_EFER, EFER_SCE | EFER_LME | EFER_LMA | EFER_NXE);
    set_msr(cs, MSR_STAR,
            ((uint64_t)SEL_DATA << 48) | ((uint64_t)SEL_CODE << 32));
    set_msr(cs, MSR_LSTAR, NANO_VA_BASE + OFF_SYSCALL_STUB);
    set_msr(cs, MSR_SFMASK, 0x257fd5);   /* mask like the Linux kernel entry */
}

static void setup_cpuid(CPUState *cs)
{
    int nent = 256;
    struct kvm_cpuid2 *cpuid =
        g_malloc0(sizeof(*cpuid) + nent * sizeof(struct kvm_cpuid_entry2));
    cpuid->nent = nent;
    kvm_ioctl(kvm_fd, KVM_GET_SUPPORTED_CPUID, cpuid, "KVM_GET_SUPPORTED_CPUID");
    kvm_ioctl(cs->kvm_fd, KVM_SET_CPUID2, cpuid, "KVM_SET_CPUID2");
    g_free(cpuid);
}

/* ------------------------------------------------------------------ */
/* env <-> vCPU register sync (GPRs via KVM_CAP_SYNC_REGS)            */
/* ------------------------------------------------------------------ */

static void sync_to_vcpu(CPUState *cs, CPUX86State *env)
{
    struct kvm_run *run = cs->kvm_run;
    struct kvm_regs *r = &run->s.regs.regs;

    r->rax = env->regs[R_EAX];
    r->rbx = env->regs[R_EBX];
    r->rcx = env->regs[R_ECX];
    r->rdx = env->regs[R_EDX];
    r->rsi = env->regs[R_ESI];
    r->rdi = env->regs[R_EDI];
    r->rsp = env->regs[R_ESP];
    r->rbp = env->regs[R_EBP];
    r->r8  = env->regs[8];
    r->r9  = env->regs[9];
    r->r10 = env->regs[10];
    r->r11 = env->regs[11];
    r->r12 = env->regs[12];
    r->r13 = env->regs[13];
    r->r14 = env->regs[14];
    r->r15 = env->regs[15];
    r->rip = env->eip;
    r->rflags = env->eflags | 0x2;
    run->kvm_dirty_regs = KVM_SYNC_X86_REGS;

    /* Push FS/GS base if arch_prctl (etc.) changed it on the host side. */
    if (env->segs[R_FS].base != cached_fs_base) {
        set_msr(cs, MSR_FS_BASE, env->segs[R_FS].base);
        cached_fs_base = env->segs[R_FS].base;
    }
    if (env->segs[R_GS].base != cached_gs_base) {
        set_msr(cs, MSR_GS_BASE, env->segs[R_GS].base);
        cached_gs_base = env->segs[R_GS].base;
    }
}

static void sync_from_vcpu(CPUState *cs, CPUX86State *env)
{
    struct kvm_regs *r = &cs->kvm_run->s.regs.regs;

    env->regs[R_EAX] = r->rax;
    env->regs[R_EBX] = r->rbx;
    env->regs[R_ECX] = r->rcx;
    env->regs[R_EDX] = r->rdx;
    env->regs[R_ESI] = r->rsi;
    env->regs[R_EDI] = r->rdi;
    env->regs[R_ESP] = r->rsp;
    env->regs[R_EBP] = r->rbp;
    env->regs[8]  = r->r8;
    env->regs[9]  = r->r9;
    env->regs[10] = r->r10;
    env->regs[11] = r->r11;
    env->regs[12] = r->r12;
    env->regs[13] = r->r13;
    env->regs[14] = r->r14;
    env->regs[15] = r->r15;
    env->eip = r->rip;
    env->eflags = r->rflags;
    env->cc_op = CC_OP_EFLAGS;
    env->cc_src = 0;
    env->df = (r->rflags & 0x400) ? -1 : 1;
}

/* ------------------------------------------------------------------ */
/* Setup entry point                                                  */
/* ------------------------------------------------------------------ */

void kvm_user_setup(CPUState *cs)
{
    CPUX86State *env = cpu_env(cs);

    kvm_debug = getenv("QEMU_KVM_DEBUG") != NULL;

    if (guest_base != 0) {
        fprintf(stderr, "qemu-kvm: requires identity mapping (guest_base==0), "
                        "got guest_base=0x%lx\n", (unsigned long)guest_base);
        _exit(1);
    }

    kvm_fd = open("/dev/kvm", O_RDWR | O_CLOEXEC);
    if (kvm_fd < 0) {
        perror("qemu-kvm: open /dev/kvm");
        _exit(1);
    }
    vm_fd = kvm_ioctl(kvm_fd, KVM_CREATE_VM, (void *)0, "KVM_CREATE_VM");
    DBG("vm created (fd=%d)\n", vm_fd);

    setup_memory();
    DBG("memory slots done (%u slots)\n", next_slot);
    setup_page_tables();
    DBG("page tables done (cr3=0x%llx)\n", (unsigned long long)cr3_gpa);
    setup_nanokernel();
    DBG("nanokernel done\n");

    cs->kvm_fd = kvm_ioctl(vm_fd, KVM_CREATE_VCPU, (void *)0, "KVM_CREATE_VCPU");
    DBG("vcpu created (fd=%d)\n", cs->kvm_fd);

    int mmap_size = kvm_ioctl(kvm_fd, KVM_GET_VCPU_MMAP_SIZE, (void *)0,
                              "KVM_GET_VCPU_MMAP_SIZE");
    cs->kvm_run = mmap(NULL, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                       cs->kvm_fd, 0);
    if (cs->kvm_run == MAP_FAILED) {
        perror("qemu-kvm: mmap kvm_run");
        _exit(1);
    }
    DBG("kvm_run mapped\n");

    /* Order matters: CPUID before sregs/msrs (KVM validates CR4/EFER/XCR0). */
    setup_cpuid(cs);
    DBG("cpuid done\n");

    {
        struct kvm_mp_state mp = { .mp_state = KVM_MP_STATE_RUNNABLE };
        kvm_ioctl(cs->kvm_fd, KVM_SET_MP_STATE, &mp, "KVM_SET_MP_STATE");
    }

    setup_sregs(cs, env);
    DBG("sregs done\n");
    setup_msrs(cs);
    DBG("msrs done\n");

    /* env->eflags is authoritative; keep the CC lazy-flags machinery inert. */
    env->cc_op = CC_OP_EFLAGS;
    env->cc_src = 0;
    env->df = 1;

    /*
     * start_exclusive() (fork, exit->plugin, core dump, ...) dereferences
     * current_cpu, which normally is only set inside TCG's cpu_exec().  Set it
     * for this thread so those paths work on the KVM path too.
     */
    current_cpu = cs;

    cs->kvm_run->kvm_valid_regs = KVM_SYNC_X86_REGS;

    DBG("setup done: entry rip=0x%llx rsp=0x%llx\n",
        (unsigned long long)env->eip, (unsigned long long)env->regs[R_ESP]);
}

/* ------------------------------------------------------------------ */
/* Run loop                                                           */
/* ------------------------------------------------------------------ */

/* CPU-pushed 64-bit interrupt frame. */
struct ist_frame {
    uint64_t rip, cs, rflags, rsp, ss;
};

static void read_ist_frame(uint64_t rsp_va, bool has_errcode,
                           struct ist_frame *f, uint64_t *errcode)
{
    /* rsp_va is a kernel-half guest VA inside the nanokernel's IST stack; its
     * host backing is nano + (rsp_va - NANO_VA_BASE). */
    uint8_t *p = nano + (rsp_va - NANO_VA_BASE);

    if (has_errcode) {
        *errcode = *(uint64_t *)p;
        p += 8;
    } else {
        *errcode = 0;
    }
    f->rip    = *(uint64_t *)(p + 0);
    f->cs     = *(uint64_t *)(p + 8);
    f->rflags = *(uint64_t *)(p + 16);
    f->rsp    = *(uint64_t *)(p + 24);
    f->ss     = *(uint64_t *)(p + 32);
}

static bool vector_has_errcode(int v)
{
    switch (v) {
    case 8: case 10: case 11: case 12: case 13: case 14: case 17:
        return true;
    default:
        return false;
    }
}

int kvm_cpu_exec_user(CPUState *cs)
{
    CPUX86State *env = cpu_env(cs);
    struct kvm_run *run = cs->kvm_run;

    current_cpu = cs;

    for (;;) {
        sync_to_vcpu(cs, env);

        DBG("KVM_RUN enter rip=0x%llx\n", (unsigned long long)env->eip);
        int r = ioctl(cs->kvm_fd, KVM_RUN, 0);
        int err = errno;

        sync_from_vcpu(cs, env);
        DBG("KVM_RUN exit r=%d errno=%d reason=%u rip=0x%llx\n",
            r, r < 0 ? err : 0, run->exit_reason, (unsigned long long)env->eip);

        if (r < 0) {
            if (err == EINTR || err == EAGAIN) {
                return EXCP_INTERRUPT;
            }
            if (err == EFAULT) {
                /* GUP failure: unmapped/protected guest memory.  M1: approximate
                 * the fault address; M2 adds instruction decode for si_addr. */
                env->cr[2] = env->eip;
                env->error_code = 0;
                return EXCP0E_PAGE;
            }
            fprintf(stderr, "qemu-kvm: KVM_RUN failed: %s\n", strerror(err));
            _exit(1);
        }

        switch (run->exit_reason) {
        case KVM_EXIT_IO: {
            uint16_t port = run->io.port;

            /* Only single-byte OUT from the nanokernel stubs are legitimate. */
            if (run->io.direction != KVM_EXIT_IO_OUT ||
                run->io.size != 1 || run->io.count != 1) {
                force_sig(TARGET_SIGILL);
                return EXCP_INTERRUPT;
            }

            if (port == SYSCALL_PORT) {
                /* syscall: RCX = return addr, R11 = saved rflags. */
                env->eip = env->regs[R_ECX];
                env->eflags = env->regs[11];
                return EXCP_SYSCALL;
            }
            if (port == 0x80) {
                struct ist_frame f;
                uint64_t ec;
                read_ist_frame(env->regs[R_ESP], false, &f, &ec);
                env->eip = f.rip;
                env->regs[R_ESP] = f.rsp;
                env->eflags = f.rflags;
                return 0x80;
            }
            if (port < 32) {
                int vec = port;
                struct ist_frame f;
                uint64_t ec;
                read_ist_frame(env->regs[R_ESP], vector_has_errcode(vec), &f, &ec);
                env->eip = f.rip;
                env->regs[R_ESP] = f.rsp;
                env->eflags = f.rflags;
                env->error_code = ec;
                cs->exception_index = vec;

                switch (vec) {
                case 0:  return EXCP00_DIVZ;
                case 1:  return EXCP01_DB;
                case 3:  return EXCP03_INT3;
                case 4:  return EXCP04_INTO;
                case 5:  return EXCP05_BOUND;
                case 6:  return EXCP06_ILLOP;
                case 13: return EXCP0D_GPF;
                case 14:
                    /* CR2 for #PF: needs sregs; M1 approximates with faulting VA. */
                    env->cr[2] = f.rip;
                    return EXCP0E_PAGE;
                default:
                    force_sig(TARGET_SIGILL);
                    return EXCP_INTERRUPT;
                }
            }
            force_sig(TARGET_SIGILL);
            return EXCP_INTERRUPT;
        }

        case KVM_EXIT_HLT:
            fprintf(stderr, "qemu-kvm: unexpected KVM_EXIT_HLT at rip=0x%llx\n",
                    (unsigned long long)env->eip);
            force_sig(TARGET_SIGSEGV);
            return EXCP_INTERRUPT;

        case KVM_EXIT_MMIO:
            /* Access to a GPA with no memslot: an unmapped (never-reserved)
             * guest address -> SIGSEGV.  M2 will decode si_addr precisely. */
            env->cr[2] = run->mmio.phys_addr;
            env->error_code = run->mmio.is_write ? 2 : 0;
            return EXCP0E_PAGE;

        case KVM_EXIT_INTR:
            return EXCP_INTERRUPT;

        case KVM_EXIT_FAIL_ENTRY:
            fprintf(stderr, "qemu-kvm: KVM_EXIT_FAIL_ENTRY reason=0x%llx\n",
                    (unsigned long long)run->fail_entry.hardware_entry_failure_reason);
            _exit(1);

        case KVM_EXIT_INTERNAL_ERROR:
            fprintf(stderr, "qemu-kvm: KVM_EXIT_INTERNAL_ERROR suberror=%u rip=0x%llx\n",
                    run->internal.suberror, (unsigned long long)env->eip);
            _exit(1);

        case KVM_EXIT_SHUTDOWN:
            fprintf(stderr, "qemu-kvm: KVM_EXIT_SHUTDOWN (triple fault) rip=0x%llx\n",
                    (unsigned long long)env->eip);
            force_sig(TARGET_SIGSEGV);
            return EXCP_INTERRUPT;

        default:
            fprintf(stderr, "qemu-kvm: unhandled exit_reason=%u\n",
                    run->exit_reason);
            _exit(1);
        }
    }
}
