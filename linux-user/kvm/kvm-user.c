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
#include <sys/syscall.h>
#include "user/signal.h"

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
#define CR4_OSXSAVE    0x40000

/* XCR0 feature bits enabled at start (SIMD on): x87 | SSE | AVX. */
#define XCR0_INIT      0x7
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

/*
 * Memslots are created at 1 GiB "chunk" granularity, on demand, and never
 * removed.  A chunk slot is identity (userspace_addr == guest_phys_addr), so
 * whichever host pages are actually mapped within it just work, and holes
 * (unmapped / munmap'd / PROT_NONE) fail GUP -> -EFAULT.  This keeps per-page
 * KVM metadata proportional to the chunks the guest touches, and means
 * munmap/mprotect need no slot bookkeeping at all (the host mapping and its
 * protection are the source of truth via GUP).
 */
#define CHUNK_BITS  30
#define CHUNK_SIZE  (1ULL << CHUNK_BITS)
#define NUM_CHUNKS  (USER_VA_END >> CHUNK_BITS)

/* ------------------------------------------------------------------ */
/* Global VM state (M1: one VM, one vCPU)                             */
/* ------------------------------------------------------------------ */

static int kvm_fd = -1;
static int vm_fd = -1;
static uint8_t *ctrl;                 /* host mapping of the control region */
static uint8_t *nano;                 /* = ctrl + NANO_OFF                  */
static uint64_t cr3_gpa;              /* guest-physical of PML4             */
static uint32_t next_slot;
static uint8_t *chunk_present;        /* bitmap: which 1GiB chunks have slots */

/* Per-vCPU (per guest thread) state, hung off CPUState::accel. */
typedef struct KVMUserVCPU {
    int vcpu_id;
    uint64_t cached_fs_base;          /* fs/gs base last pushed, to skip MSR writes */
    uint64_t cached_gs_base;
    struct kvm_xsave xsave_scratch;   /* scratch for FP get/put around signals */
} KVMUserVCPU;

static inline KVMUserVCPU *vcpu_of(CPUState *cs)
{
    return (KVMUserVCPU *)cs->accel;
}

/* vCPU id allocation + parking (KVM can't destroy vCPUs, so recycle fds). */
static int next_vcpu_id;
typedef struct ParkedVCPU {
    int vcpu_id;
    int fd;
    struct kvm_run *run;
    struct ParkedVCPU *next;
} ParkedVCPU;
static ParkedVCPU *parked_list;
/* clone_lock (held around thread create/exit) also serializes these. */

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

/* Ensure a memslot exists for the 1GiB chunk containing gpa. */
static void ensure_chunk(uint64_t gpa)
{
    uint64_t c = gpa >> CHUNK_BITS;

    if (c >= NUM_CHUNKS) {
        return;                                 /* kernel-half / vsyscall etc. */
    }
    if (chunk_present[c >> 3] & (1u << (c & 7))) {
        return;
    }
    chunk_present[c >> 3] |= (1u << (c & 7));

    uint64_t base = c << CHUNK_BITS;
    uint64_t size = MIN(CHUNK_SIZE, USER_VA_END - base);
    add_memslot(next_slot++, base, base, size);
    DBG("  chunk slot %u: [0x%llx, +0x%llx)\n", next_slot - 1,
        (unsigned long long)base, (unsigned long long)size);
}

/*
 * Ensure memslots cover [start, start+len).  Called from the mmap layer for
 * every new/moved guest mapping, and at setup for the load-time mappings.
 * No-op until the VM exists (early target_mmap during ELF load is covered by
 * the setup-time walk below).
 */
void kvm_user_track_range(uint64_t start, uint64_t len)
{
    if (!kvm_user_enabled || vm_fd < 0 || len == 0) {
        return;
    }
    uint64_t end = start + len;
    for (uint64_t g = start & ~(CHUNK_SIZE - 1); g < end; g += CHUNK_SIZE) {
        ensure_chunk(g);
    }
}

static int mirror_region_cb(void *priv, vaddr start, vaddr end, int flags)
{
    /*
     * Only slot readable memory.  PROT_NONE reservations (which can span many
     * TiB, e.g. a PIE/ld.so address-space reservation) get no slot: a guest
     * access to them correctly faults (KVM_EXIT_MMIO -> SIGSEGV), and when the
     * guest later mprotect()s a sub-range readable, the mprotect hook slots it.
     */
    if (!(flags & PAGE_READ)) {
        DBG("  skip non-readable [0x%llx, 0x%llx) flags=0x%x\n",
            (unsigned long long)start, (unsigned long long)end, flags);
        return 0;
    }
    kvm_user_track_range(start, end - start);
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

    chunk_present = g_malloc0((NUM_CHUNKS + 7) / 8);

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
    sregs.cr4 = CR4_PAE | CR4_OSFXSR | CR4_OSXMMEXCPT | CR4_FSGSBASE |
                CR4_OSXSAVE;
    sregs.efer = EFER_SCE | EFER_LME | EFER_LMA | EFER_NXE;
    sregs.cr8 = 0;

    kvm_ioctl(cs->kvm_fd, KVM_SET_SREGS, &sregs, "KVM_SET_SREGS");
    vcpu_of(cs)->cached_fs_base = env->segs[R_FS].base;
    vcpu_of(cs)->cached_gs_base = env->segs[R_GS].base;
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

static void setup_xcrs(CPUState *cs)
{
    /* Enable x87|SSE|AVX in XCR0 so the guest can use AVX/AVX2 (SIMD on).
     * Must follow KVM_SET_CPUID2 (KVM validates XCR0 against CPUID leaf 0xD). */
    struct kvm_xcrs xcrs = { 0 };
    xcrs.nr_xcrs = 1;
    xcrs.xcrs[0].xcr = 0;
    xcrs.xcrs[0].value = XCR0_INIT;
    kvm_ioctl(cs->kvm_fd, KVM_SET_XCRS, &xcrs, "KVM_SET_XCRS");
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
    KVMUserVCPU *v = vcpu_of(cs);
    if (env->segs[R_FS].base != v->cached_fs_base) {
        set_msr(cs, MSR_FS_BASE, env->segs[R_FS].base);
        v->cached_fs_base = env->segs[R_FS].base;
    }
    if (env->segs[R_GS].base != v->cached_gs_base) {
        set_msr(cs, MSR_GS_BASE, env->segs[R_GS].base);
        v->cached_gs_base = env->segs[R_GS].base;
    }
}

/*
 * FP/SIMD state normally lives in the vCPU; env's decomposed FP fields are
 * stale.  The signal machinery reads/writes env's FP (to save into / restore
 * from the signal frame), so sync it lazily exactly around those points.
 * (M3: single scratch buffer; per-vCPU in M4.)
 */
void kvm_user_get_fpu(CPUState *cs)
{
    struct kvm_xsave *buf = &vcpu_of(cs)->xsave_scratch;
    kvm_ioctl(cs->kvm_fd, KVM_GET_XSAVE, buf, "KVM_GET_XSAVE");
    x86_cpu_xrstor_all_areas(X86_CPU(cs), buf, sizeof(buf->region));
}

void kvm_user_put_fpu(CPUState *cs)
{
    struct kvm_xsave *buf = &vcpu_of(cs)->xsave_scratch;
    memset(buf, 0, sizeof(*buf));
    x86_cpu_xsave_all_areas(X86_CPU(cs), buf, sizeof(buf->region));
    kvm_ioctl(cs->kvm_fd, KVM_SET_XSAVE, buf, "KVM_SET_XSAVE");
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

/*
 * Create (or recycle a parked) vCPU for this guest thread and prime all of its
 * control state from env.  Runs on the owning thread.  Called for the main
 * thread from kvm_user_setup() and for each new guest thread from clone_func().
 * Must be serialized (the callers hold clone_lock) w.r.t. vcpu-id/parking.
 */
void kvm_user_init_vcpu(CPUState *cs)
{
    CPUX86State *env = cpu_env(cs);
    KVMUserVCPU *v = g_new0(KVMUserVCPU, 1);
    cs->accel = (AccelCPUState *)v;
    v->cached_fs_base = ~0ULL;
    v->cached_gs_base = ~0ULL;

    /* Recycle a parked vCPU if any, else create a fresh one. */
    ParkedVCPU *p = parked_list;
    if (p) {
        parked_list = p->next;
        v->vcpu_id = p->vcpu_id;
        cs->kvm_fd = p->fd;
        cs->kvm_run = p->run;
        g_free(p);
        DBG("vcpu recycled id=%d fd=%d\n", v->vcpu_id, cs->kvm_fd);
    } else {
        v->vcpu_id = next_vcpu_id++;
        cs->kvm_fd = kvm_ioctl(vm_fd, KVM_CREATE_VCPU,
                               (void *)(intptr_t)v->vcpu_id, "KVM_CREATE_VCPU");
        int mmap_size = kvm_ioctl(kvm_fd, KVM_GET_VCPU_MMAP_SIZE, (void *)0,
                                  "KVM_GET_VCPU_MMAP_SIZE");
        cs->kvm_run = mmap(NULL, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                           cs->kvm_fd, 0);
        if (cs->kvm_run == MAP_FAILED) {
            perror("qemu-kvm: mmap kvm_run");
            _exit(1);
        }
        /* CPUID is immutable after first KVM_RUN, so set it once at creation. */
        setup_cpuid(cs);
        DBG("vcpu created id=%d fd=%d\n", v->vcpu_id, cs->kvm_fd);
    }

    {
        struct kvm_mp_state mp = { .mp_state = KVM_MP_STATE_RUNNABLE };
        kvm_ioctl(cs->kvm_fd, KVM_SET_MP_STATE, &mp, "KVM_SET_MP_STATE");
    }

    setup_sregs(cs, env);
    setup_xcrs(cs);
    setup_msrs(cs);

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

    DBG("vcpu %d ready: rip=0x%llx rsp=0x%llx\n", v->vcpu_id,
        (unsigned long long)env->eip, (unsigned long long)env->regs[R_ESP]);
}

/*
 * Force another vCPU thread out of KVM_RUN (for start_exclusive: fork, core
 * dump).  Sets immediate_exit so an about-to-enter KVM_RUN returns at once, and
 * sends host_interrupt_signal so a thread already inside KVM_RUN returns -EINTR.
 */
void kvm_user_kick(CPUState *cs)
{
    /*
     * start_exclusive() calls qemu_cpu_kick() directly (not via cpu_exit), so
     * set exit_request here too; the run loop acquire-loads it after clearing
     * immediate_exit.
     */
    qatomic_set(&cs->exit_request, true);
    if (cs->kvm_run) {
        qatomic_set(&cs->kvm_run->immediate_exit, 1);
    }
    /*
     * Only signal OTHER threads.  Kicking ourselves (e.g. cpu_exit() from our
     * own host signal handler) must not tgkill self -- that would re-enter the
     * handler forever; exit_request/immediate_exit already handle the self case.
     */
    if (cs != current_cpu) {
        TaskState *ts = get_task_state(cs);
        if (ts && host_interrupt_signal) {
            syscall(__NR_tgkill, getpid(), ts->ts_tid, host_interrupt_signal);
        }
    }
}

/* Installed as qemu_cpu_kick_hook so start_exclusive can evict KVM vCPUs. */
extern void (*qemu_cpu_kick_hook)(CPUState *cpu);

/* Park an exiting thread's vCPU for reuse (KVM has no destroy-vcpu). */
void kvm_user_park_vcpu(CPUState *cs)
{
    KVMUserVCPU *v = vcpu_of(cs);
    if (!v) {
        return;
    }
    ParkedVCPU *p = g_new0(ParkedVCPU, 1);
    p->vcpu_id = v->vcpu_id;
    p->fd = cs->kvm_fd;
    p->run = cs->kvm_run;
    p->next = parked_list;
    parked_list = p;
    DBG("vcpu parked id=%d\n", v->vcpu_id);
    g_free(v);
    cs->accel = NULL;
}

/*
 * fork() support.  KVM fds are bound to the parent's mm and are useless in the
 * child, so the child rebuilds its VM/slots/vCPU from scratch.  The control
 * region (page tables + nanokernel) and all guest memory are COW-preserved at
 * the same host addresses, so only the KVM objects need recreating.
 */

/* Parent side, before fork(): snapshot live FP into env so the child (COW)
 * inherits it and can push it to its rebuilt vCPU. */
void kvm_user_fork_start(void)
{
    if (current_cpu) {
        kvm_user_get_fpu(current_cpu);
    }
}

/* Child side, after fork(): the only surviving thread rebuilds the VM. */
void kvm_user_fork_child(CPUState *cs)
{
    /* Inherited fds refer to the parent's VM; drop this thread's and the VM's. */
    close(cs->kvm_fd);
    close(vm_fd);
    close(kvm_fd);
    g_free(vcpu_of(cs));            /* COW copy of parent's per-vCPU struct */
    cs->accel = NULL;
    cs->kvm_run = NULL;

    /* Reset VM-global bookkeeping (other threads are gone in the child). */
    next_slot = 0;
    next_vcpu_id = 0;
    parked_list = NULL;
    memset(chunk_present, 0, (NUM_CHUNKS + 7) / 8);

    kvm_fd = open("/dev/kvm", O_RDWR | O_CLOEXEC);
    if (kvm_fd < 0) {
        perror("qemu-kvm: fork child open /dev/kvm");
        _exit(1);
    }
    vm_fd = kvm_ioctl(kvm_fd, KVM_CREATE_VM, (void *)0, "KVM_CREATE_VM");

    /* Control region content is COW-preserved; just re-slot it and the guest
     * mappings (page tables / nanokernel need no rebuild). */
    add_memslot(next_slot++, CTRL_GPA, (uint64_t)(uintptr_t)ctrl, CTRL_SIZE);
    walk_memory_regions(NULL, mirror_region_cb);

    kvm_user_init_vcpu(cs);
    kvm_user_put_fpu(cs);          /* restore the inherited FP snapshot */
    DBG("fork child rebuilt VM (vm_fd=%d)\n", vm_fd);
}

void kvm_user_setup(CPUState *cs)
{
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

    /* Route qemu_cpu_kick() to the KVM eviction path (start_exclusive). */
    qemu_cpu_kick_hook = kvm_user_kick;

    kvm_user_init_vcpu(cs);
    DBG("setup done\n");
}

/* ------------------------------------------------------------------ */
/* Fault address decoding                                             */
/* ------------------------------------------------------------------ */

/*
 * When KVM returns a bare -EFAULT (GUP failure for a page inside a memslot,
 * e.g. a PROT_NONE guard page or a munmap'd hole) it carries no fault address.
 * Recover it by decoding the memory operand of the faulting instruction at RIP
 * using the (already synced) guest register values.  Handles the common
 * ModRM/SIB load/store encodings including REX and VEX prefixes, plus string
 * ops; falls back to RIP for anything it can't decode.  guest_base == 0, so a
 * guest VA is also the host VA of the same byte.
 */
static uint64_t decode_fault_addr(CPUX86State *env)
{
    uint64_t rip = env->eip;

    if (!(page_get_flags(rip) & PAGE_READ)) {
        return rip;                         /* instruction fetch fault */
    }
    const uint8_t *ip = (const uint8_t *)(uintptr_t)rip;

    int i = 0, seg = -1;
    int rex_x = 0, rex_b = 0;

    /* Legacy prefixes. */
    for (;;) {
        uint8_t b = ip[i];
        if (b == 0x66 || b == 0x67 || b == 0xf0 || b == 0xf2 || b == 0xf3 ||
            b == 0x2e || b == 0x36 || b == 0x3e || b == 0x26) {
            i++;
            continue;
        }
        if (b == 0x64) { seg = R_FS; i++; continue; }
        if (b == 0x65) { seg = R_GS; i++; continue; }
        break;
    }

    /* REX / VEX, then step over the opcode so ip[i] lands on ModRM. */
    uint8_t b = ip[i];
    if (b == 0xc5) {                         /* 2-byte VEX */
        i += 2 + 1;                          /* VEX + opcode */
    } else if (b == 0xc4) {                  /* 3-byte VEX */
        rex_x = !((ip[i + 1] >> 6) & 1);
        rex_b = !((ip[i + 1] >> 5) & 1);
        i += 3 + 1;
    } else {
        if ((b & 0xf0) == 0x40) {            /* REX */
            rex_x = (b >> 1) & 1;
            rex_b = b & 1;
            i++;
        }
        uint8_t op = ip[i];
        /* String ops have no ModRM: operand is RSI (load) and/or RDI (store). */
        switch (op) {
        case 0xa4: case 0xa5:                /* movs */
        case 0xa6: case 0xa7:                /* cmps */
            if (!(page_get_flags(env->regs[R_EDI]) & PAGE_VALID)) {
                return env->regs[R_EDI];
            }
            return env->regs[R_ESI];
        case 0xaa: case 0xab:                /* stos */
        case 0xae: case 0xaf:                /* scas */
            return env->regs[R_EDI];
        case 0xac: case 0xad:                /* lods */
            return env->regs[R_ESI];
        }
        if (op == 0x0f) {                    /* 2/3-byte opcode */
            i++;
            if (ip[i] == 0x38 || ip[i] == 0x3a) {
                i++;
            }
        }
        i++;                                 /* opcode byte */
    }

    uint8_t modrm = ip[i++];
    int mod = modrm >> 6, rm = modrm & 7;

    if (mod == 3) {
        return rip;                          /* register operand: no memory */
    }

    uint64_t base = 0, index = 0;
    int scale = 0, have_base = 1, have_index = 0;

    if (rm == 4) {                           /* SIB */
        uint8_t sib = ip[i++];
        scale = sib >> 6;
        int idx = ((sib >> 3) & 7) | (rex_x << 3);
        int bas = (sib & 7) | (rex_b << 3);
        if (((sib >> 3) & 7) != 4 || rex_x) {
            have_index = 1;
            index = env->regs[idx];
        }
        if ((sib & 7) == 5 && mod == 0) {
            have_base = 0;                   /* disp32, no base */
        } else {
            base = env->regs[bas];
        }
    } else if (rm == 5 && mod == 0) {        /* RIP-relative */
        int32_t disp;
        memcpy(&disp, &ip[i], 4);
        i += 4;
        uint64_t addr = rip + i + disp;      /* best-effort (ignores immediate) */
        if (seg >= 0) {
            addr += env->segs[seg].base;
        }
        return addr;
    } else {
        base = env->regs[rm | (rex_b << 3)];
    }

    int64_t disp = 0;
    if (mod == 1) {
        disp = (int8_t)ip[i++];
    } else if (mod == 2 || (mod == 0 && !have_base)) {
        int32_t d;
        memcpy(&d, &ip[i], 4);
        disp = d;
        i += 4;
    }

    uint64_t addr = (have_base ? base : 0) +
                    (have_index ? (index << scale) : 0) + disp;
    if (seg >= 0) {
        addr += env->segs[seg].base;
    }
    return addr;
}

/* Build a #PF-style error_code for a fault at addr from the guest page flags. */
static uint32_t fault_error_code(uint64_t addr, bool is_write)
{
    int flags = page_get_flags(addr);
    uint32_t ec = 0;

    if (flags & PAGE_VALID) {
        ec |= PG_ERROR_P_MASK;               /* present -> SEGV_ACCERR */
    }
    /* A write to a present, non-writable page is a write-permission fault. */
    if (is_write || ((flags & PAGE_VALID) && !(flags & PAGE_WRITE))) {
        ec |= PG_ERROR_W_MASK;
    }
    return ec;
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

#define VSYSCALL_PAGE 0xffffffffff600000ULL

static uint64_t read_cr2(CPUState *cs)
{
    struct kvm_sregs sregs;
    kvm_ioctl(cs->kvm_fd, KVM_GET_SREGS, &sregs, "KVM_GET_SREGS");
    return sregs.cr2;
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

        /*
         * Kick-race discipline (mirrors accel/kvm): clear immediate_exit, THEN
         * acquire-load exit_request/signal_pending.  A kicker sets exit_request
         * before immediate_exit, so either we observe exit_request here, or the
         * kicker's immediate_exit=1 makes the KVM_RUN below return -EINTR.
         */
        qatomic_set(&cs->kvm_run->immediate_exit, 0);
        if (qatomic_read(&cs->exit_request) ||
            qatomic_read(&get_task_state(cs)->signal_pending)) {
            return EXCP_INTERRUPT;
        }

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
                /* GUP failure inside a memslot (PROT_NONE/hole/munmap'd): KVM
                 * gives no address, so decode the faulting instruction. */
                uint64_t addr = decode_fault_addr(env);
                env->cr[2] = addr;
                env->error_code = fault_error_code(addr, false);
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
                case 14: {
                    /*
                     * Guest #PF.  With static all-present identity user PTs this
                     * only fires for kernel-half VAs the guest can't map -- in
                     * practice the legacy vsyscall page.  CR2 holds the address.
                     */
                    uint64_t cr2 = read_cr2(cs);
                    if ((cr2 & ~0xfffULL) == VSYSCALL_PAGE) {
                        env->eip = cr2;          /* emulate_vsyscall reads offset */
                        return EXCP_VSYSCALL;
                    }
                    env->cr[2] = cr2;
                    return EXCP0E_PAGE;
                }
                case 16:                          /* #MF: x87 FP exception */
                case 19:                          /* #XM: SIMD FP exception */
                    force_sig_fault(TARGET_SIGFPE, TARGET_FPE_FLTINV, f.rip);
                    return EXCP_INTERRUPT;
                case 17:                          /* #AC: alignment check */
                    force_sig(TARGET_SIGBUS);
                    return EXCP_INTERRUPT;
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
            /* Access to a GPA with no memslot (unmapped guest address).  The
             * faulting GPA is exact; derive the error code from page flags. */
            env->cr[2] = run->mmio.phys_addr;
            env->error_code = fault_error_code(run->mmio.phys_addr,
                                               run->mmio.is_write);
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
