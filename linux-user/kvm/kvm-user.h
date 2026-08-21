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
#endif /* TARGET_X86_64 */

#endif /* LINUX_USER_KVM_USER_H */
