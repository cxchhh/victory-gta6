//! Copies of title memory that a fault does not end: `try_copy` copies
//! `len` bytes and says whether all of them were copied. It is the plain
//! copy instruction (`rep movsb`), and a vectored exception handler that,
//! for an access violation at that one instruction which no other handler
//! takes (the guards' handler opens a guarded block and the copy goes on),
//! resumes the routine at its failure return.
//!
//! Why: title memory was read and written through the kernel
//! (`ReadProcessMemory`, `NtWriteVirtualMemory`) so that a page the title
//! unmapped meanwhile fails the call instead of faulting. Under Wine the
//! writes cost 22 us for 4 KiB, 57 us for 64 KiB and 750 us for 1 MiB
//! against 0.4 / 6.7 / 105 us for a copy (RTX 4090 host, Wine 11.22,
//! `host::tests::kernel_copy_cost`), and the copy back writes each run of
//! changed bytes with its own call. `VICTORY_FAST_COPY=0`: the kernel
//! calls, as before.

use std::sync::OnceLock;

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_POINTERS,
};

std::arch::global_asm!(
    ".globl victory_try_copy",
    ".globl victory_try_copy_insn",
    ".globl victory_try_copy_fault",
    "victory_try_copy:",
    "    push rdi",
    "    push rsi",
    "    mov rdi, rcx",
    "    mov rsi, rdx",
    "    mov rcx, r8",
    "victory_try_copy_insn:",
    "    rep movsb",
    "    xor eax, eax",
    "    pop rsi",
    "    pop rdi",
    "    ret",
    "victory_try_copy_fault:",
    "    mov eax, 1",
    "    pop rsi",
    "    pop rdi",
    "    ret",
);

unsafe extern "C" {
    fn victory_try_copy(dst: *mut u8, src: *const u8, len: usize) -> u32;
    fn victory_try_copy_insn();
    fn victory_try_copy_fault();
}

const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the system hands a valid record and context.
    unsafe {
        let record = &*(*info).ExceptionRecord;
        let context = &mut *(*info).ContextRecord;
        if record.ExceptionCode as NTSTATUS == EXCEPTION_ACCESS_VIOLATION
            && context.Rip == victory_try_copy_insn as *const () as usize as u64
        {
            context.Rip = victory_try_copy_fault as *const () as usize as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    EXCEPTION_CONTINUE_SEARCH
}

/// Whether the fast copy is on (`VICTORY_FAST_COPY`, default on); the
/// handler is installed on the first ask, last of all handlers.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_FAST_COPY").is_none_or(|v| v != "0");
        if on {
            // SAFETY: a handler of static code; never removed.
            unsafe { AddVectoredExceptionHandler(0, Some(handler)) };
        }
        on
    })
}

/// Copies `len` bytes from `src` to `dst`; false if a fault stopped it
/// (some bytes may have been copied).
///
/// # Safety
/// The ranges may be unmapped or protected (that is what the false is
/// for), but must not be memory whose fault means something else to its
/// owner beyond the guards' handler (which runs first).
pub(crate) unsafe fn try_copy(dst: *mut u8, src: *const u8, len: usize) -> bool {
    debug_assert!(on());
    // SAFETY: forwarded; the routine only copies.
    unsafe { victory_try_copy(dst, src, len) == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory as mem;

    #[test]
    fn a_copy_stops_at_an_unmapped_page_and_says_so() {
        assert!(on());
        // SAFETY: an allocation of the test's own.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 3 << 16, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE).cast::<u8>();
            let src = vec![9u8; 3 << 16];
            assert!(try_copy(base, src.as_ptr(), 2 << 16));
            assert_eq!(*base.add((2 << 16) - 1), 9);
            assert_ne!(mem::VirtualFree(base.add(2 << 16).cast(), 1 << 16, mem::MEM_DECOMMIT), 0);
            assert!(!try_copy(base.add(1 << 16), src.as_ptr(), 2 << 16), "a write into a decommitted page");
            let mut back = vec![0u8; 2 << 16];
            assert!(!try_copy(back.as_mut_ptr(), base.add(1 << 16), 2 << 16), "a read from a decommitted page");
            assert!(try_copy(back.as_mut_ptr(), base, 1 << 16));
            let mut old = 0;
            mem::VirtualProtect(base.cast(), 1 << 16, mem::PAGE_READONLY, &mut old);
            assert!(!try_copy(base, src.as_ptr(), 16), "a write into a read-only page");
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }
}
