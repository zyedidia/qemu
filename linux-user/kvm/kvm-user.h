/*
 * KVM-backed user-mode execution (x86-64 guest on x86-64 host).
 *
 * Runs the guest binary natively inside a KVM VM at ring 0 instead of
 * TCG-translating it; syscalls, faults and signals become VM exits that the
 * existing linux-user layer handles.  See docs in linux-user/kvm/kvm-user.c.
 */
#ifndef LINUX_USER_KVM_USER_H
#define LINUX_USER_KVM_USER_H

/*
 * True when this process should execute the guest via KVM rather than TCG.
 * Set from argv[0] == "qemu-kvm", the -kvm flag, or QEMU_KVM in the env.
 * Defined unconditionally (referenced from target-independent main.c) but only
 * ever true for an x86-64 guest on an x86-64 host.
 */
extern bool kvm_user_enabled;

#if defined(TARGET_X86_64)
/*
 * Build the VM, blanket memslots and ring-0 nanokernel, then create and prime
 * the initial vCPU from the (already loaded) initial CPU state in env.
 * Called from main() after init_main_thread(), before cpu_loop().
 */
void kvm_user_setup(CPUState *cs);

/*
 * Run the vCPU until a VM exit that maps to a guest-visible event, syncing env
 * from the vCPU, and return the corresponding EXCP_* code.  Drop-in replacement
 * for cpu_exec() on the KVM path.
 */
int kvm_cpu_exec_user(CPUState *cs);

/*
 * Ensure KVM chunk translations (memslot + guest PDPTE) cover
 * [start, start+len).  Called from the mmap layer whenever a new/moved guest
 * mapping appears.  No-op unless KVM mode is active and the VM has been
 * created.
 */
void kvm_user_track_range(uint64_t start, uint64_t len);

/*
 * Tear down and recycle the translation of any 1 GiB chunk in
 * [start, start+len) that no longer contains any guest mapping.  Called from
 * the mmap layer after munmap/mremap/shmdt have cleared the range's page
 * flags.  No-op unless KVM mode is active and the VM has been created.
 */
void kvm_user_untrack_range(uint64_t start, uint64_t len);

/*
 * Sync FP/SIMD state between the vCPU and env, called by the signal-frame code:
 * get_fpu before saving env's FP into a signal frame, put_fpu after restoring
 * env's FP from a frame on sigreturn.
 */
void kvm_user_get_fpu(CPUState *cs);
void kvm_user_put_fpu(CPUState *cs);

/*
 * Per-thread vCPU lifecycle.  init_vcpu creates (or recycles a parked) vCPU for
 * a guest thread and primes it from env; call on the owning thread under
 * clone_lock.  park_vcpu recycles an exiting thread's vCPU.
 */
void kvm_user_init_vcpu(CPUState *cs);
void kvm_user_park_vcpu(CPUState *cs);

/* Force another vCPU thread out of KVM_RUN (start_exclusive). */
void kvm_user_kick(CPUState *cs);

/* fork() support: snapshot FP in the parent before fork; rebuild the VM in the
 * child (KVM fds don't survive fork). */
void kvm_user_fork_start(void);
void kvm_user_fork_child(CPUState *cs);

/* True if fd is a guest-invisible internal KVM fd that a guest close()/dup2()
 * must not touch (would tear down the VM). */
bool kvm_user_is_internal_fd(int fd);
/* Smallest internal fd >= from, or -1 (for close_range splitting). */
int kvm_user_next_internal_fd(int from);

/* Print a per-process VM-exit summary if QEMU_KVM_STATS is set (called at exit). */
void kvm_user_dump_stats(void);
#endif /* TARGET_X86_64 */

#endif /* LINUX_USER_KVM_USER_H */
