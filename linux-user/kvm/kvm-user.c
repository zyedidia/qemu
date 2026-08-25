/*
 * KVM-backed user-mode execution (x86-64 guest on x86-64 host).
 *
 * Instead of TCG-translating the guest, we run it natively inside a KVM VM at
 * ring 0 (long mode, CPL0).  Guest-virtual == host-virtual (guest_base == 0),
 * so linux-user's host mappings back the guest's directly.  Guest-PHYSICAL
 * space is separate and private to us: guest VA space is carved into 16 MiB
 * chunks, and each chunk that contains a readable guest page owns a compact
 * GPA chunk from a pool, wired up by one KVM memslot (GPA chunk -> the chunk's
 * host VA) plus eight 2 MiB leaf PDEs in the guest page tables (guest VA ->
 * GPA chunk).  A guest access to mapped memory thus GUPs the identical host
 * page (it just works); an access to a host-unmapped or PROT_NONE page inside
 * a live chunk fails GUP and returns a bare -EFAULT from KVM_RUN with the
 * guest RIP left at the faulting instruction; an access to a VA with no chunk
 * takes a guest #PF with an exact CR2.  When munmap/mremap/shmdt/mprotect
 * leaves a chunk with no readable page, its slot is deleted and its GPA chunk
 * recycled (kvm_user_untrack_range).  See the CHUNK_BITS comment for why the
 * unit is 16 MiB.
 *
 * (Identity GVA == GPA is NOT viable: usable guest-physical width is the
 * host's MAXPHYADDR, commonly just 39 bits = 512 GiB, while host mmap
 * scatters guest mappings across ~128 TiB of VA space.  Blanket slots are out
 * for the same reason, and per-mapping translation also keeps KVM's per-slot
 * metadata proportional to memory the guest actually uses.)
 *
 * A tiny host-generated ring-0 "nanokernel" (GDT/IDT/TSS + one-instruction
 * out-stubs) turns every syscall and CPU exception into a KVM_EXIT_IO: MSR_LSTAR
 * points at `out %al,$SYSCALL_PORT`, and each IDT gate is `out %al,$vector`.
 * `out` clobbers nothing, so on the exit we read pristine guest state straight
 * from the KVM_CAP_SYNC_REGS mmap page and, after handling, write the resume RIP
 * back the same way (the kernel's complete_fast_pio_out kvm_is_linear_rip guard
 * preserves our RIP).  The nanokernel + guest page tables live in one "control"
 * memslot in GPA chunk 0, which is never handed to a guest chunk, so guest
 * mappings can never collide with them.
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
#include "exec/mmap-lock.h"
#include "kvm-user.h"

#include <linux/kvm.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include "user/signal.h"

bool kvm_user_enabled;

static int kvm_debug;
#define DBG(...) do { if (kvm_debug) { fprintf(stderr, "qemu-kvm: " __VA_ARGS__); } } while (0)

/* VM-exit accounting (QEMU_KVM_STATS=1 to print a per-process summary). */
static int kvm_stats;
static uint64_t st_syscall, st_fault, st_eintr, st_exc, st_other, st_total;
#define SC_HIST_N 1024
static uint64_t sc_hist[SC_HIST_N];   /* per-syscall-number exit histogram */

/* ------------------------------------------------------------------ */
/* Control region: guest page tables + nanokernel, in GPA chunk 0.     */
/* ------------------------------------------------------------------ */

#define CTRL_GPA     0x00000000ULL         /* control region base GPA        */
#define PT_OFF       0x00000000ULL         /* base page tables at ctrl + 0   */

/* Base page-table page indices within the control region. */
#define PT_PML4_PAGE   0
#define PT_PDPT0_PAGE  1                   /* 256 user PDPTs: pages 1..256   */
#define PT_PDPTK_PAGE  257                 /* kernel-half PDPT               */
#define PT_PDK_PAGE    258                 /* kernel-half PD                 */
#define PT_TOP_PT_PAGE 259                 /* PT for the partial 2 MiB at USER_VA_END */
#define PT_BASE_PAGES  260                 /* base tables occupy [0, 260)    */

#define NANO_OFF     0x00400000ULL         /* nanokernel at ctrl + 4 MiB     */
#define NANO_GPA     (CTRL_GPA + NANO_OFF)
#define NANO_SIZE    0x00200000ULL         /* one 2 MiB page                 */

/*
 * Per-region page directories: one PD page per 1 GiB user region, at a fixed
 * control-region offset (region r -> PD_POOL_OFF + r*4096).  PDPTEs point at
 * these statically (set once at setup); only the leaf 2 MiB PDEs are toggled
 * on the fly by ensure_chunk()/free_chunk().  The pool spans the whole user
 * range but is lazily backed, so untouched regions cost no memory.  It sits
 * above the nanokernel [4,6) MiB.
 */
#define PD_POOL_OFF  0x00800000ULL         /* 8 MiB */

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
 * GVA -> GPA translation is at 16 MiB "chunk" granularity, on demand.  A GVA
 * chunk that contains a readable guest page owns a 16 MiB-aligned GPA chunk
 * from a compact pool (bump-allocated + recycled), wired up by one memslot
 * (slot id == GPA chunk index; userspace_addr == the chunk's GVA) and eight
 * 2 MiB leaf PDEs in its region's page directory.  Within a live chunk the
 * host mapping and its protection remain the source of truth via GUP: holes
 * (unmapped / munmap'd / PROT_NONE) fail GUP -> -EFAULT, so a partial munmap
 * or a mprotect that leaves some readable page needs no bookkeeping here at
 * all.  Only when a whole chunk no longer holds any readable page -- every
 * page unmapped or decommitted to PROT_NONE -- is it torn down and its GPA
 * chunk recycled (kvm_user_untrack_range).
 *
 * Why 16 MiB and 2 MiB PDEs rather than 1 GiB PDPTEs: guest-physical space is
 * scarce (39-bit MAXPHYADDR = 512 GiB here) while a real process (e.g. a
 * browser content process) scatters ~1 GiB of live memory as a few MiB in
 * each of hundreds of distinct 1 GiB windows.  At 1 GiB granularity each such
 * window burned a whole GPA chunk and the 512-chunk pool exhausted with the
 * process barely started.  16 MiB chunks cut that waste ~64x while a chunk
 * still amortises one memslot over enough GPA that the memslot cap
 * (KVM_CAP_NR_MEMSLOTS) and the 512 GiB GPA cap coincide -- so total coverage
 * is unchanged.  CHUNK_BITS is the single knob: raise it toward 30 to favour
 * huge dense mappings, lower it toward 21 to favour sparse ones.
 */
#define CHUNK_BITS  24                     /* 16 MiB translation unit */
#define CHUNK_SIZE  (1ULL << CHUNK_BITS)
#define NUM_CHUNKS  ((USER_VA_END + CHUNK_SIZE - 1) >> CHUNK_BITS)

#define PDE_BITS        21                 /* 2 MiB leaf pages */
#define PDE_SIZE        (1ULL << PDE_BITS)
#define REGION_BITS     30                 /* 1 GiB: one page directory's span */
#define REGION_SIZE     (1ULL << REGION_BITS)
#define NUM_REGIONS     ((USER_VA_END + REGION_SIZE - 1) >> REGION_BITS)
#define PDES_PER_CHUNK  (CHUNK_SIZE / PDE_SIZE)               /* 8   */
#define CHUNKS_PER_REGION (REGION_SIZE / CHUNK_SIZE)          /* 64  */

/* Control region: base tables + nanokernel + the per-region PD pool, rounded
 * up to a chunk.  Sits in the low GPA chunks, which are reserved from the
 * allocator (CTRL_CHUNKS of them). */
#define CTRL_SIZE ((PD_POOL_OFF + NUM_REGIONS * 4096ULL + CHUNK_SIZE - 1) \
                   & ~(CHUNK_SIZE - 1))
#define CTRL_CHUNKS (CTRL_SIZE >> CHUNK_BITS)

#define GPA_FREE    UINT32_MAX

/* ------------------------------------------------------------------ */
/* Global VM state (M1: one VM, one vCPU)                             */
/* ------------------------------------------------------------------ */

static int kvm_fd = -1;
static int vm_fd = -1;
static uint8_t *ctrl;                 /* host mapping of the control region */
static uint8_t *nano;                 /* = ctrl + NANO_OFF                  */
static uint64_t cr3_gpa;              /* guest-physical of PML4             */

/*
 * GVA<->GPA chunk state.  All mutation happens under mmap_lock (taken inside
 * kvm_user_track_range/kvm_user_untrack_range; setup and fork are
 * single-threaded); gpa_owner is additionally read locklessly on the
 * KVM_EXIT_MMIO path, hence the qatomic accessors there.
 */
static uint32_t *chunk_map;           /* 16MiB GVA chunk -> GPA chunk (0=none) */
static uint32_t *gpa_owner;           /* GPA chunk -> GVA chunk, or GPA_FREE   */
static uint32_t *gpa_free_stack;      /* recycled GPA chunks (LIFO)            */
static uint32_t n_gpa_free;
static uint32_t next_gpa_chunk;       /* bump allocator; [0,CTRL_CHUNKS)=ctrl  */
static uint32_t pool_chunks;          /* GPA chunks total: usable are          */
                                      /* [CTRL_CHUNKS, pool_chunks)            */

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

/*
 * Guest-invisible host fds (/dev/kvm, VM, vCPUs) must survive the guest's own
 * close()/close_range() (Firefox's content process closes "superfluous" fds for
 * sandbox hygiene, which would otherwise tear down our VM).  Track them so the
 * syscall layer can refuse guest closes.
 */
static bool *prot_fds;
static int prot_cap;

static void kvm_mark_fd(int fd, bool v)
{
    if (fd < 0) {
        return;
    }
    if (fd >= prot_cap) {
        int nc = fd + 16;
        prot_fds = g_realloc(prot_fds, nc);
        memset(prot_fds + prot_cap, 0, nc - prot_cap);
        prot_cap = nc;
    }
    prot_fds[fd] = v;
}

bool kvm_user_is_internal_fd(int fd)
{
    return fd >= 0 && fd < prot_cap && prot_fds[fd];
}

/* Smallest internal fd >= from, or -1 (for close_range splitting). */
int kvm_user_next_internal_fd(int from)
{
    for (int fd = from < 0 ? 0 : from; fd < prot_cap; fd++) {
        if (prot_fds[fd]) {
            return fd;
        }
    }
    return -1;
}

/* Dump KVM's own per-vCPU counters (includes in-kernel exits not seen by the
 * userspace run loop) via KVM_GET_STATS_FD -- no root needed. */
static void kvm_dump_kvm_stats(int vcpu_fd)
{
    int sfd = ioctl(vcpu_fd, KVM_GET_STATS_FD, 0);
    if (sfd < 0) {
        return;
    }
    struct kvm_stats_header hdr;
    if (pread(sfd, &hdr, sizeof(hdr), 0) != sizeof(hdr)) {
        close(sfd);
        return;
    }
    size_t dsz = sizeof(struct kvm_stats_desc) + hdr.name_size;
    char *descs = g_malloc0(dsz * hdr.num_desc);
    if (pread(sfd, descs, dsz * hdr.num_desc, hdr.desc_offset) > 0) {
        for (unsigned i = 0; i < hdr.num_desc; i++) {
            struct kvm_stats_desc *d = (struct kvm_stats_desc *)(descs + i * dsz);
            uint64_t v = 0;
            if (pread(sfd, &v, 8, hdr.data_offset + d->offset) == 8 && v) {
                fprintf(stderr, "  kvmstat %-24s = %llu\n",
                        d->name, (unsigned long long)v);
            }
        }
    }
    g_free(descs);
    close(sfd);
}

/* Print a per-process VM-exit summary (QEMU_KVM_STATS=1); called at exit. */
void kvm_user_dump_stats(void)
{
    if (!kvm_stats || st_total == 0) {
        return;
    }
    uint64_t known = st_syscall + st_fault + st_exc + st_eintr;
    st_other = st_total > known ? st_total - known : 0;
    fprintf(stderr,
        "[qemu-kvm-stats] pid %d exits=%llu  syscall=%llu  fault=%llu  "
        "exc=%llu  eintr=%llu  other=%llu\n",
        getpid(),
        (unsigned long long)st_total, (unsigned long long)st_syscall,
        (unsigned long long)st_fault, (unsigned long long)st_exc,
        (unsigned long long)st_eintr, (unsigned long long)st_other);

    /* Top syscall numbers by VM-exit count (x86-64 NRs). */
    static const struct { int nr; const char *name; } names[] = {
        {0,"read"},{1,"write"},{3,"close"},{7,"poll"},{9,"mmap"},{10,"mprotect"},
        {11,"munmap"},{12,"brk"},{13,"rt_sigaction"},{14,"rt_sigprocmask"},
        {15,"rt_sigreturn"},{24,"sched_yield"},{25,"mremap"},{28,"madvise"},
        {35,"nanosleep"},{39,"getpid"},{56,"clone"},{60,"exit"},{62,"kill"},
        {96,"gettimeofday"},{98,"getrusage"},{131,"sigaltstack"},{158,"arch_prctl"},
        {186,"gettid"},{200,"tkill"},{202,"futex"},{204,"sched_getaffinity"},
        {218,"set_tid_address"},{228,"clock_gettime"},{230,"clock_nanosleep"},
        {234,"tgkill"},{257,"openat"},{273,"set_robust_list"},{302,"prlimit64"},
        {318,"getrandom"},{334,"rseq"},{435,"clone3"},{-1,NULL}
    };
    fprintf(stderr, "[qemu-kvm-stats] top syscall-exit NRs:\n");
    for (int rank = 0; rank < 15; rank++) {
        int best = -1;
        uint64_t bestv = 0;
        for (int i = 0; i < SC_HIST_N; i++) {
            if (sc_hist[i] > bestv) { bestv = sc_hist[i]; best = i; }
        }
        if (best < 0 || bestv == 0) {
            break;
        }
        const char *nm = "?";
        for (int j = 0; names[j].nr >= 0; j++) {
            if (names[j].nr == best) { nm = names[j].name; break; }
        }
        fprintf(stderr, "    nr=%-4d %-18s %llu\n", best, nm,
                (unsigned long long)bestv);
        sc_hist[best] = 0;   /* consume so next rank finds the next */
    }

    if (current_cpu) {
        kvm_dump_kvm_stats(current_cpu->kvm_fd);
    }
}

/* ------------------------------------------------------------------ */
/* Memory: GVA<->GPA chunk translation + slot bookkeeping             */
/* ------------------------------------------------------------------ */

/*
 * Chunk recycling deletes memslots, and by default x86 KVM reacts to a slot
 * deletion by invalidating the ENTIRE VM's TDP mappings -- a legacy
 * workaround for a never-diagnosed VFIO passthrough regression -- so every
 * recycle forces all vCPUs to refault their whole working set (measured at
 * ~40 ms per 128 MiB resident).  Kernels >= 6.12 expose that behavior as
 * KVM_X86_QUIRK_SLOT_ZAP_ALL; disabling it makes a deletion zap only the
 * deleted slot's GPA range.  Correctness is unaffected either way: the
 * remote TLB flush on deletion (which GPA recycling relies on) happens in
 * both modes.  Call once per VM, right after KVM_CREATE_VM.
 */
static void disable_slot_zap_quirk(void)
{
#ifdef KVM_X86_QUIRK_SLOT_ZAP_ALL
    int mask = ioctl(kvm_fd, KVM_CHECK_EXTENSION, KVM_CAP_DISABLE_QUIRKS2);
    if (mask > 0 && (mask & KVM_X86_QUIRK_SLOT_ZAP_ALL)) {
        struct kvm_enable_cap cap = {
            .cap = KVM_CAP_DISABLE_QUIRKS2,
            .args = { KVM_X86_QUIRK_SLOT_ZAP_ALL },
        };
        kvm_ioctl(vm_fd, KVM_ENABLE_CAP, &cap,
                  "KVM_ENABLE_CAP (disable slot-zap-all quirk)");
        DBG("slot-zap-all quirk disabled: deletes zap only the dead slot\n");
    } else {
        DBG("slot-zap-all quirk unsupported (pre-6.12 kernel): each chunk "
            "recycle invalidates the whole VM's TDP mappings\n");
    }
#endif
}

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

static void del_memslot(uint32_t slot)
{
    struct kvm_userspace_memory_region region = { .slot = slot };

    kvm_ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &region,
              "KVM_SET_USER_MEMORY_REGION (delete)");
}

/*
 * Usable guest-physical space is bounded by the host's MAXPHYADDR, possibly
 * reduced (e.g. by AMD memory encryption) -- commonly just 39 bits (512 GiB).
 * Ask KVM itself: find the highest power-of-two limit under which it accepts
 * a one-page memslot.
 */
static uint64_t probe_gpa_limit(void)
{
    void *page = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        perror("qemu-kvm: mmap gpa probe page");
        _exit(1);
    }
    for (int bits = 52; bits > CHUNK_BITS; bits--) {
        struct kvm_userspace_memory_region region = {
            .slot = 0,
            .guest_phys_addr = (1ULL << bits) - 0x1000,
            .memory_size = 0x1000,
            .userspace_addr = (uint64_t)(uintptr_t)page,
        };
        if (ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &region) == 0) {
            region.memory_size = 0;
            kvm_ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &region,
                      "KVM_SET_USER_MEMORY_REGION (delete)");
            munmap(page, 0x1000);
            DBG("gpa limit: 2^%d\n", bits);
            return 1ULL << bits;
        }
    }
    fprintf(stderr, "qemu-kvm: could not size guest-physical space\n");
    _exit(1);
}

/* First of the PDES_PER_CHUNK (8) leaf 2 MiB PDEs backing GVA chunk c, in its
 * region's statically-placed page directory (layout fixed by setup_page_tables). */
static uint64_t *chunk_pdes(uint64_t c)
{
    uint64_t region = c / CHUNKS_PER_REGION;
    uint64_t sub    = c % CHUNKS_PER_REGION;
    uint64_t *pd = (uint64_t *)(ctrl + PD_POOL_OFF + region * 4096);
    return &pd[sub * PDES_PER_CHUNK];
}

/* Memslot span of a GVA chunk (the top chunk stops at USER_VA_END). */
static uint64_t chunk_span(uint64_t gva_chunk)
{
    return MIN(CHUNK_SIZE, USER_VA_END - (gva_chunk << CHUNK_BITS));
}

/*
 * On exhaustion, characterize the live chunk set so the cause is diagnosable:
 * sum the guest's readable bytes and bucket each live chunk by how much of its
 * CHUNK_SIZE is readable.  A set dominated by nearly-empty chunks
 * (readable_total << live * CHUNK_SIZE) means fragmentation or a leaked
 * decommit path -- finer granularity or a missing untrack.  Chunks that are
 * mostly full mean a genuine working set that needs more guest-physical space.
 * Runs once, under mmap_lock (held by the track path), just before we fault.
 */
static uint64_t *stat_readable_per_gpa;
static uint64_t stat_readable_total;

static int exhaust_stat_cb(void *priv, vaddr start, vaddr end, int flags)
{
    if (!(flags & PAGE_READ)) {
        return 0;
    }
    stat_readable_total += end - start;
    for (uint64_t c = start >> CHUNK_BITS; c <= (end - 1) >> CHUNK_BITS; c++) {
        if (c < NUM_CHUNKS && chunk_map[c] && chunk_map[c] < pool_chunks) {
            uint64_t cs = c << CHUNK_BITS, ce = cs + CHUNK_SIZE;
            uint64_t ov = MIN(end, ce) - MAX(start, cs);
            stat_readable_per_gpa[chunk_map[c]] += ov;
        }
    }
    return 0;
}

static void dump_exhaustion_stats(void)
{
    unsigned live = 0, b_tiny = 0, b_small = 0, b_mid = 0, b_full = 0;

    stat_readable_per_gpa = g_new0(uint64_t, pool_chunks);
    stat_readable_total = 0;
    walk_memory_regions(NULL, exhaust_stat_cb);

    for (uint32_t g = CTRL_CHUNKS; g < pool_chunks; g++) {
        if (qatomic_read(&gpa_owner[g]) == GPA_FREE) {
            continue;
        }
        live++;
        uint64_t r = stat_readable_per_gpa[g];   /* readable of this 16 MiB */
        if (r < (1u << 20)) {
            b_tiny++;                            /* < 1 MiB: near-empty */
        } else if (r < (CHUNK_SIZE / 4)) {
            b_small++;                           /* < quarter full */
        } else if (r < CHUNK_SIZE) {
            b_mid++;                             /* partial */
        } else {
            b_full++;                            /* full 16 MiB */
        }
    }
    fprintf(stderr,
        "qemu-kvm: [exhaustion] live chunks=%u  readable total=%llu MiB "
        "(%.1f%% of chunk span)\n"
        "qemu-kvm: [exhaustion] per-chunk readable: <1MiB=%u  <4MiB=%u  "
        "<16MiB=%u  full=%u\n"
        "qemu-kvm: [exhaustion] %s\n",
        live, (unsigned long long)(stat_readable_total >> 20),
        live ? 100.0 * stat_readable_total / ((uint64_t)live << CHUNK_BITS) : 0.0,
        b_tiny, b_small, b_mid, b_full,
        (b_tiny + b_small) > live / 2
        ? "mostly-empty chunks -> fragmentation or a leaked decommit path"
        : "chunks mostly full -> genuine working set exceeds guest-physical space");
    g_free(stat_readable_per_gpa);
    stat_readable_per_gpa = NULL;
}

static uint32_t alloc_gpa_chunk(void)
{
    if (n_gpa_free > 0) {
        return gpa_free_stack[--n_gpa_free];
    }
    if (next_gpa_chunk < pool_chunks) {
        return next_gpa_chunk++;
    }
    /*
     * Out of guest-physical space: all usable GPA chunks hold live mappings
     * simultaneously.  Leave this chunk untranslated; a guest access to it
     * will take #PF -> SIGSEGV.
     */
    static bool warned;
    if (!warned) {
        warned = true;
        fprintf(stderr, "qemu-kvm: out of guest-physical space (%u %llu MiB "
                "chunks in simultaneous use); guest accesses to further "
                "mappings will fault\n",
                pool_chunks - (unsigned)CTRL_CHUNKS,
                (unsigned long long)(CHUNK_SIZE >> 20));
        dump_exhaustion_stats();
    }
    return 0;
}

/*
 * Install the guest translation for chunk c -> GPA chunk g: the chunk's eight
 * 2 MiB leaf PDEs.  A full chunk is eight 2 MiB PS pages.  The one partial
 * chunk at the top of the user range instead drops its last 2 MiB PDE to a
 * 4 KiB PT that stops exactly at USER_VA_END: a PS page there would let
 * accesses in [USER_VA_END, region end) reach GPAs beyond the chunk's
 * memslot, and a no-slot GPA access puts the vCPU into KVM's in-kernel MMIO
 * emulation, from which no clean #PF can be delivered (see KVM_EXIT_MMIO in
 * the run loop).  With an exact tail those accesses take a guest #PF like any
 * other unmapped VA.
 */
static void install_chunk_pte(uint64_t c, uint32_t g)
{
    uint64_t gpa = (uint64_t)g << CHUNK_BITS;
    uint64_t span = chunk_span(c);
    uint64_t *pde = chunk_pdes(c);

    for (uint64_t j = 0; j < PDES_PER_CHUNK; j++) {
        uint64_t off = j << PDE_BITS;
        if (off + PDE_SIZE <= span) {
            pde[j] = (gpa + off) | PTE_P | PTE_RW | PTE_PS;   /* full 2 MiB */
        } else if (off < span) {
            /* Partial 2 MiB (only the top chunk): exact 4 KiB PT. */
            uint64_t *pt = (uint64_t *)(ctrl + PT_OFF + PT_TOP_PT_PAGE * 4096);
            for (uint64_t k = 0; k < 512; k++) {
                uint64_t poff = off + (k << 12);
                pt[k] = poff < span ? (gpa + poff) | PTE_P | PTE_RW : 0;
            }
            pde[j] = (CTRL_GPA + PT_OFF + PT_TOP_PT_PAGE * 4096) |
                     PTE_P | PTE_RW;
        } else {
            pde[j] = 0;                          /* beyond span (top chunk) */
        }
    }
}

/* Give the 16 MiB GVA chunk containing va a GPA chunk, memslot and leaf PDEs. */
static void ensure_chunk(uint64_t va)
{
    uint64_t c = va >> CHUNK_BITS;

    if (c >= NUM_CHUNKS) {
        return;                                 /* kernel-half / vsyscall etc. */
    }
    if (chunk_map[c]) {
        return;
    }
    uint32_t g = alloc_gpa_chunk();
    if (!g) {
        return;
    }
    chunk_map[c] = g;
    qatomic_set(&gpa_owner[g], (uint32_t)c);
    /*
     * Slot before PDEs: a concurrent vCPU's page walk must never find a
     * translation to a slotless GPA (that would surface as spurious MMIO).
     * Not-present entries are never cached, so the install itself needs no
     * TLB invalidation.  (The PDPTE above these PDEs is static and always
     * present -- see setup_page_tables -- so only the leaves toggle.)
     */
    add_memslot(g, (uint64_t)g << CHUNK_BITS, c << CHUNK_BITS, chunk_span(c));
    install_chunk_pte(c, g);
    DBG("  chunk va=0x%llx -> gpa chunk %u (+0x%llx)\n",
        (unsigned long long)(c << CHUNK_BITS), g,
        (unsigned long long)chunk_span(c));
}

/* Tear down an empty chunk's translation and recycle its GPA chunk. */
static void free_chunk(uint64_t c)
{
    uint32_t g = chunk_map[c];
    uint64_t *pde = chunk_pdes(c);

    /*
     * Leaf PDEs first, so a page walk after the flush below faults instead of
     * re-creating a translation; then the slot DELETE, which zaps the chunk's
     * EPT entries and flushes every vCPU's TLB (evicting any stale
     * GVA->GPA->HVA translation) before returning.  Only after that may the
     * GPA chunk be handed to a different GVA chunk.  The region's PD and
     * PDPTE stay in place (static); an all-zero PD simply faults.
     */
    for (uint64_t j = 0; j < PDES_PER_CHUNK; j++) {
        pde[j] = 0;
    }
    del_memslot(g);
    chunk_map[c] = 0;
    qatomic_set(&gpa_owner[g], GPA_FREE);
    gpa_free_stack[n_gpa_free++] = g;
    DBG("  chunk va=0x%llx freed (gpa chunk %u recycled)\n",
        (unsigned long long)(c << CHUNK_BITS), g);
}

/*
 * Ensure chunk translations cover [start, start+len).  Called from the mmap
 * layer for every new/moved readable guest mapping, and at setup for the
 * load-time mappings.  No-op until the VM exists (early target_mmap during
 * ELF load is covered by the setup-time walk).
 */
void kvm_user_track_range(uint64_t start, uint64_t len)
{
    if (!kvm_user_enabled || vm_fd < 0 || len == 0) {
        return;
    }
    mmap_lock();                   /* serializes all chunk/PDE bookkeeping */
    uint64_t end = start + len;
    for (uint64_t g = start & ~(CHUNK_SIZE - 1); g < end; g += CHUNK_SIZE) {
        ensure_chunk(g);
    }
    mmap_unlock();
}

/*
 * Recycle any chunk in [start, start+len) that no longer contains a readable
 * guest page.  Called from the mmap layer after munmap/mremap/shmdt removed a
 * mapping or mprotect made a range non-readable (decommit); the readability
 * test and the teardown run under mmap_lock so they are atomic against
 * concurrent mmap.
 */
void kvm_user_untrack_range(uint64_t start, uint64_t len)
{
    if (!kvm_user_enabled || vm_fd < 0 || len == 0) {
        return;
    }
    mmap_lock();
    uint64_t last_c = MIN((start + len - 1) >> CHUNK_BITS,
                          (uint64_t)NUM_CHUNKS - 1);
    for (uint64_t c = start >> CHUNK_BITS; c <= last_c; c++) {
        /*
         * A chunk's slot exists only to let KVM GUP readable host pages; a
         * page with no PAGE_READ (PROT_NONE reservation or decommit) fails
         * GUP anyway.  So the chunk is live iff some page in it is still
         * readable -- NOT merely still mapped.  Testing PAGE_READ rather than
         * PAGE_VALID is what lets an allocator's mprotect(PROT_NONE) decommit
         * recycle the chunk instead of pinning it for the reservation's life.
         */
        if (chunk_map[c] &&
            !page_range_any_flags(c << CHUNK_BITS,
                                  (c << CHUNK_BITS) + chunk_span(c) - 1,
                                  PAGE_READ)) {
            free_chunk(c);
        }
    }
    mmap_unlock();
}

static int mirror_region_cb(void *priv, vaddr start, vaddr end, int flags)
{
    /*
     * Only readable memory gets a chunk.  PROT_NONE reservations (which can
     * span many TiB, e.g. a PIE/ld.so address-space reservation) get none: a
     * guest access to them correctly faults (#PF -> SIGSEGV), and when the
     * guest later mprotect()s a sub-range readable, the mprotect hook maps it.
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
    /*
     * Size the GPA chunk pool.  The low CTRL_CHUNKS chunks are the control
     * region (one big memslot, slot 0); the rest are handed out to GVA chunks,
     * each with its own memslot whose id is the GPA chunk index.  The pool is
     * capped by both guest-physical space and the memslot count; at 16 MiB
     * chunks on a 39-bit host these coincide near 512 GiB.
     */
    uint64_t gpa_limit = probe_gpa_limit();
    int nr_slots = ioctl(kvm_fd, KVM_CHECK_EXTENSION, KVM_CAP_NR_MEMSLOTS);
    if (nr_slots <= 0) {
        nr_slots = 32;                          /* KVM's historical default */
    }
    pool_chunks = MIN(gpa_limit >> CHUNK_BITS, (uint64_t)nr_slots);
    if (pool_chunks <= CTRL_CHUNKS + 4) {
        fprintf(stderr, "qemu-kvm: guest-physical space too small\n");
        _exit(1);
    }
    DBG("gpa pool: %u usable %llu MiB chunks (control uses %u)\n",
        pool_chunks - (unsigned)CTRL_CHUNKS,
        (unsigned long long)(CHUNK_SIZE >> 20), (unsigned)CTRL_CHUNKS);

    chunk_map = g_malloc0(NUM_CHUNKS * sizeof(*chunk_map));
    gpa_owner = g_new(uint32_t, pool_chunks);
    for (uint32_t i = 0; i < pool_chunks; i++) {
        gpa_owner[i] = GPA_FREE;
    }
    gpa_free_stack = g_new(uint32_t, pool_chunks);
    n_gpa_free = 0;
    next_gpa_chunk = CTRL_CHUNKS;                /* [0,CTRL_CHUNKS) reserved */

    /* Control region: base page tables + nanokernel + per-region PD pool.
     * Large but lazily backed -- only touched PDs consume memory. */
    ctrl = mmap(NULL, CTRL_SIZE, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (ctrl == MAP_FAILED) {
        perror("qemu-kvm: mmap control region");
        _exit(1);
    }
    nano = ctrl + NANO_OFF;
    add_memslot(0, CTRL_GPA, (uint64_t)(uintptr_t)ctrl, CTRL_SIZE);
}

/* ------------------------------------------------------------------ */
/* Guest page tables (in the control region)                          */
/* ------------------------------------------------------------------ */

static void setup_page_tables(void)
{
    /*
     * PML4[0..255] -> 256 PDPTs covering user VAs [0, 2^47).  Every user PDPTE
     * is set once here to point at its 1 GiB region's page directory in the PD
     * pool (region r's PD at PD_POOL_OFF + r*4096); those PDs start all-zero
     * (lazily backed) and only their leaf 2 MiB PDEs are demand-installed by
     * ensure_chunk()/free_chunk().  Keeping the PDPTE->PD level static means it
     * is never cached-then-stale, so no paging-structure-cache flush is ever
     * needed there.  PML4[511] -> a chain mapping the nanokernel's kernel-half
     * VA to NANO_GPA.  A table page at control offset O has GPA CTRL_GPA + O.
     */
    const int NUSER = 256;
    uint8_t *base = ctrl + PT_OFF;
    memset(base, 0, PT_BASE_PAGES * 4096);

    uint64_t *pml4  = (uint64_t *)(base + PT_PML4_PAGE * 4096);
    uint64_t *pdptk = (uint64_t *)(base + PT_PDPTK_PAGE * 4096);
    uint64_t *pdk   = (uint64_t *)(base + PT_PDK_PAGE * 4096);
    uint64_t gpa_base = CTRL_GPA + PT_OFF;

    for (int i = 0; i < NUSER; i++) {
        pml4[i] = (gpa_base + (uint64_t)(PT_PDPT0_PAGE + i) * 4096) |
                  PTE_P | PTE_RW;
    }

    /* Point every user region's PDPTE at its static PD page. */
    for (uint64_t r = 0; r < NUM_REGIONS; r++) {
        uint64_t *pdpt = (uint64_t *)(base + (PT_PDPT0_PAGE + (r >> 9)) * 4096);
        pdpt[r & 511] = (CTRL_GPA + PD_POOL_OFF + r * 4096) | PTE_P | PTE_RW;
    }

    /* Kernel-half chain for the nanokernel VA: PML4[511]->PDPTk[510]->PDk[0]. */
    pml4[511]  = (gpa_base + (uint64_t)PT_PDPTK_PAGE * 4096) | PTE_P | PTE_RW;
    pdptk[510] = (gpa_base + (uint64_t)PT_PDK_PAGE * 4096) | PTE_P | PTE_RW;
    pdk[0]     = NANO_GPA | PTE_P | PTE_RW | PTE_PS;         /* 2MiB page */

    cr3_gpa = gpa_base + PT_PML4_PAGE * 4096;
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
        kvm_mark_fd(cs->kvm_fd, true);
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
    kvm_mark_fd(cs->kvm_fd, false);
    kvm_mark_fd(vm_fd, false);
    kvm_mark_fd(kvm_fd, false);
    close(cs->kvm_fd);
    close(vm_fd);
    close(kvm_fd);
    g_free(vcpu_of(cs));            /* COW copy of parent's per-vCPU struct */
    cs->accel = NULL;
    cs->kvm_run = NULL;

    /* Reset VM-global bookkeeping (other threads are gone in the child). */
    next_vcpu_id = 0;
    parked_list = NULL;
    /* Inherited parked-vCPU fds are gone too; the chunk allocator state
     * (chunk_map/gpa_owner) is COW-preserved and rebuilt from below. */
    memset(prot_fds, 0, prot_cap);

    kvm_fd = open("/dev/kvm", O_RDWR | O_CLOEXEC);
    if (kvm_fd < 0) {
        perror("qemu-kvm: fork child open /dev/kvm");
        _exit(1);
    }
    kvm_mark_fd(kvm_fd, true);
    vm_fd = kvm_ioctl(kvm_fd, KVM_CREATE_VM, (void *)0, "KVM_CREATE_VM");
    disable_slot_zap_quirk();
    kvm_mark_fd(vm_fd, true);

    /*
     * The control region (base tables, per-region PDs and their leaf PDEs)
     * and the chunk allocator state are plain COW memory and survive fork
     * intact, so the inherited page tables already encode the parent's
     * GVA -> GPA chunk assignment.  Rebuild the memslots from chunk_map[]
     * verbatim -- NOT by re-walking the mappings, which could assign GPA
     * chunks differently from what the PDEs say.
     */
    add_memslot(0, CTRL_GPA, (uint64_t)(uintptr_t)ctrl, CTRL_SIZE);
    for (uint64_t c = 0; c < NUM_CHUNKS; c++) {
        if (chunk_map[c]) {
            add_memslot(chunk_map[c], (uint64_t)chunk_map[c] << CHUNK_BITS,
                        c << CHUNK_BITS, chunk_span(c));
        }
    }

    kvm_user_init_vcpu(cs);
    kvm_user_put_fpu(cs);          /* restore the inherited FP snapshot */
    DBG("fork child rebuilt VM (vm_fd=%d)\n", vm_fd);
}

void kvm_user_setup(CPUState *cs)
{
    kvm_debug = getenv("QEMU_KVM_DEBUG") != NULL;
    kvm_stats = getenv("QEMU_KVM_STATS") != NULL;

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
    kvm_mark_fd(kvm_fd, true);
    vm_fd = kvm_ioctl(kvm_fd, KVM_CREATE_VM, (void *)0, "KVM_CREATE_VM");
    kvm_mark_fd(vm_fd, true);
    DBG("vm created (fd=%d)\n", vm_fd);
    disable_slot_zap_quirk();

    setup_memory();
    setup_page_tables();
    DBG("page tables done (cr3=0x%llx)\n", (unsigned long long)cr3_gpa);
    setup_nanokernel();
    DBG("nanokernel done\n");

    /* Map the guest mappings that exist at load time.  Must follow
     * setup_page_tables(): the walk installs leaf PDEs. */
    walk_memory_regions(NULL, mirror_region_cb);
    DBG("load-time mappings mapped (%u gpa chunks)\n", next_gpa_chunk - 1);

    /* Route qemu_cpu_kick() to the KVM eviction path (start_exclusive). */
    qemu_cpu_kick_hook = kvm_user_kick;

    kvm_user_init_vcpu(cs);
    DBG("setup done\n");

    /* Visible proof this process is executing inside the KVM VM.  Suppress with
     * QEMU_KVM_QUIET=1. */
    if (!getenv("QEMU_KVM_QUIET")) {
        fprintf(stderr,
            "\033[1;32m[qemu-kvm]\033[0m pid %d \342\226\266 running "
            "\033[1m%s\033[0m natively at ring 0 in a KVM VM "
            "(long mode, SIMD)\n", getpid(), exec_path);
    }
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

/*
 * True if the guest page flags now permit the access that just faulted (only
 * ec's W and I/D bits are consulted).  A fault on a page that is valid by
 * handling time raced another thread's mmap/mprotect -- chunk slots and
 * PDPTEs trail the page-flag update -- so, like the kernel on a spurious
 * fault, the access should be retried rather than raised as a signal.
 */
static bool fault_resolved(uint64_t addr, uint32_t ec)
{
    int flags = page_get_flags(addr);

    if (ec & PG_ERROR_I_D_MASK) {
        return flags & PAGE_EXEC;
    }
    if (ec & PG_ERROR_W_MASK) {
        return (flags & PAGE_READ) && (flags & PAGE_WRITE);
    }
    return flags & PAGE_READ;
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
    uint64_t retry_addr = 0;
    int retry_count = 0;

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
        st_total++;
        DBG("KVM_RUN exit r=%d errno=%d reason=%u rip=0x%llx\n",
            r, r < 0 ? err : 0, run->exit_reason, (unsigned long long)env->eip);

        if (r < 0) {
            if (err == EINTR || err == EAGAIN) {
                st_eintr++;
                return EXCP_INTERRUPT;
            }
            if (err == EFAULT) {
                st_fault++;
                /* GUP failure inside a memslot (PROT_NONE/hole/munmap'd): KVM
                 * gives no address, so decode the faulting instruction. */
                uint64_t addr = decode_fault_addr(env);
                /*
                 * Spurious (raced a concurrent mmap/mprotect): retry.  The
                 * access direction is unknown here, so demand write
                 * permission too; the cap guards against a page GUP
                 * persistently refuses (e.g. OOM).
                 */
                if (fault_resolved(addr, PG_ERROR_W_MASK)) {
                    if (addr != retry_addr) {
                        retry_addr = addr;
                        retry_count = 0;
                    }
                    if (retry_count++ < 64) {
                        continue;
                    }
                }
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
                st_syscall++;
                sc_hist[env->regs[R_EAX] & (SC_HIST_N - 1)]++;
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
                st_exc++;
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
                     * Guest #PF, CR2 exact.  Fires for the legacy vsyscall
                     * page, for other kernel-half VAs, and for any user VA
                     * whose chunk has no translation: unmapped memory, a
                     * mapping decommitted to PROT_NONE (still PAGE_VALID, but
                     * its chunk was recycled), or a mapping whose PDPTE was
                     * racing in (then the retry below resumes it).
                     */
                    uint64_t cr2 = read_cr2(cs);
                    if ((cr2 & ~0xfffULL) == VSYSCALL_PAGE) {
                        env->eip = cr2;          /* emulate_vsyscall reads offset */
                        return EXCP_VSYSCALL;
                    }
                    if (fault_resolved(cr2, ec)) {
                        if (cr2 != retry_addr) {
                            retry_addr = cr2;
                            retry_count = 0;
                        }
                        if (retry_count++ < 64) {
                            continue;
                        }
                    }
                    /*
                     * The hardware error code's P bit only reflects our coarse
                     * chunk tables, where "no chunk" covers both truly
                     * unmapped memory and a mapped-but-PROT_NONE page.  Derive
                     * the guest-visible code from linux-user's page flags
                     * instead so si_code is SEGV_ACCERR vs SEGV_MAPERR
                     * correctly (keep only the hardware W bit for direction).
                     */
                    env->cr[2] = cr2;
                    env->error_code =
                        fault_error_code(cr2, ec & PG_ERROR_W_MASK);
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

        case KVM_EXIT_MMIO: {
            /*
             * EPT access to a GPA with no memslot.  Cannot legitimately
             * happen: every GPA a guest page-table walk can produce lies
             * within a live memslot (full chunks by construction; the top
             * partial chunk via exact sub-tables), and chunk teardown clears
             * the PTEs and flushes every vCPU's TLB before the slot dies.
             * Nor is recovery possible: KVM's in-kernel emulator has already
             * advanced past the instruction and will complete the access
             * with run->mmio data on the next KVM_RUN, clobbering any signal
             * delivery we attempt.  So: report and die.
             */
            st_fault++;
            uint64_t gpa = run->mmio.phys_addr;
            uint64_t g = gpa >> CHUNK_BITS;
            uint32_t owner = (g >= 1 && g < pool_chunks)
                             ? qatomic_read(&gpa_owner[g]) : GPA_FREE;
            fprintf(stderr, "qemu-kvm: unexpected MMIO exit: %s gpa=0x%llx "
                    "(gpa chunk %llu: %s va chunk 0x%llx) size=%u rip=0x%llx\n",
                    run->mmio.is_write ? "write" : "read",
                    (unsigned long long)gpa, (unsigned long long)g,
                    owner == GPA_FREE ? "unowned; nominal" : "owned by",
                    (unsigned long long)(owner == GPA_FREE ? 0 : owner),
                    run->mmio.len, (unsigned long long)env->eip);
            _exit(1);
        }

        case KVM_EXIT_INTR:
            st_eintr++;
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
