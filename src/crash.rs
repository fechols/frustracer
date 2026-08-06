//! Crash handler — a symbolized call stack (Rust AND C++, in one list) plus a
//! minidump, printed by the process itself the moment it faults.
//!
//! WHY THIS EXISTS. Until this module landed, a crash produced exactly one
//! datum: cargo's `exit code: 0xc0000005, STATUS_ACCESS_VIOLATION`. No faulting
//! module, no stack, nothing to act on — so every crash cost a manual debugger
//! attach just to learn *which DLL* faulted. That is a bad trade in a process
//! that hands its device, queue and swapchain to four vendor runtimes (NGX,
//! ffx, XeSS/XeLL, ORT/DirectML) and whose teardown order is load-bearing. The
//! tree's own scars: `gpu/mod.rs`'s FI-swapchain context "faults outright
//! (0xC0000005 ~3.7 s in)" and is fixed only by a deliberate `mem::forget`;
//! `xess_fg.rs` records a shutdown AV in `Interface::assume_vtable`; and
//! CLAUDE.md records a panic that unwound through teardown and **took a second
//! access violation on the way out, so the crash the user saw had nothing to do
//! with the cause**. A top-level filter that reports the FIRST exception is the
//! structural answer to that last one.
//!
//! WHY ONE STACK COVERS BOTH LANGUAGES. On Windows/MSVC there is one unwinder
//! (`RtlVirtualUnwind`) and one symbol format (PDB). The Rust code and the
//! cc-built `shim/*.cpp` compile into static libs linked into the ONE exe, so
//! they share ONE `frustracer.pdb` and a single native walk interleaves them:
//!
//!   Rust            demangled name + file:line   (release: line-tables-only)
//!   shim/*.cpp      name + file:line             (cc forces /Z7 on MSVC —
//!                                                 full CodeView, regardless
//!                                                 of the debug LEVEL string)
//!   vendor DLLs     module.dll+0xOFFSET          (no public PDBs — and this
//!                                                 IS the attribution wanted)
//!   MS DLLs         module+offset now; real names later in WinDbg against
//!                   the .dmp + the public symbol server
//!
//! Note `.cargo/config.toml` sets `-C target-cpu=native` with no
//! `force-frame-pointers`, so the walk MUST use unwind data — never RBP
//! chaining. `backtrace` and dbghelp do exactly that on x64.
//!
//! FOOTPRINT. Nothing here imports dbghelp: `backtrace` LoadLibrary's it
//! itself, and `MiniDumpWriteDump` is resolved with GetProcAddress (the
//! `gpu/pix.rs` pattern). So dbghelp loads only on an actual crash and every
//! `--check*` stays DLL-free. The handler is an OS-level filter with zero
//! render-path contact: no rng draws, no globals a kernel reads, so it cannot
//! move a gate or a pixel.
//!
//! LEVERS (dev, default-off, loud when armed):
//!   --no-crash-handler   don't install (a pre-parse argv scan, the
//!                        `--no-settings` precedent — the flag must be
//!                        readable before `cli::parse_from` runs)
//!   FR_NO_CRASH=1        same, by environment
//!   FR_CRASH_FULLDUMP=1  MiniDumpWithFullMemory instead of stacks+modules
//!                        (~10 GB on THE WORLD — opt in per bug, never by
//!                        default)
//!   FR_CRASH_VERIFY=1    report, from an atexit handler AFTER main returns,
//!                        whether the installed filter is still ours — the
//!                        observable answer to "did something displace it?"
//!   FR_CRASH_TEST=deref|cpp|panic|overflow|atexit
//!                        deliberately fault at install time. `cpp` faults
//!                        INSIDE a shim translation unit, which is what makes
//!                        "it prints C++ frames too" provable rather than
//!                        assumed — the anti-vacuity gate for this module.
//!                        `atexit` is the other one: it faults AFTER main has
//!                        returned, proving the handler outlives it.
//!
//! Both post-main levers (`FR_CRASH_VERIFY`, `FR_CRASH_TEST=atexit`) fire ONLY
//! on a path where `main` RETURNS — i.e. the interactive session. Rust's
//! `std::process::exit` is `ExitProcess`, which never walks the CRT's atexit
//! list, so every headless path here bypasses them; they say so at arm time
//! rather than doing nothing quietly. See `EXIT_ROUTE_NOTE`.
//!
//! KNOWN LIMIT: when a debugger is attached, Windows hands the exception to it
//! and `UnhandledExceptionFilter` never calls a top-level filter — so under
//! cdb/WinDbg the debugger's view is the authoritative one, not this.

use std::sync::atomic::AtomicBool;

/// Frames printed before the walk is cut off. Deep enough for a rayon +
/// driver stack, short enough that a runaway walk in a corrupt process ends.
const MAX_FRAMES: usize = 64;

/// Set once `install` has run; makes a second call a no-op.
static INSTALLED: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Pure formatting — all platforms, so `self_test` runs everywhere `--check`
// does (the gates' DLL-free bar) and the tables can be pinned without faulting
// anything.
// ---------------------------------------------------------------------------

/// Human name for an NTSTATUS-shaped exception code. The list is the set worth
/// recognizing on sight in this process; anything else prints as hex, which is
/// still actionable.
pub fn exception_name(code: u32) -> &'static str {
    match code {
        0xC0000005 => "EXCEPTION_ACCESS_VIOLATION",
        0xC000001D => "EXCEPTION_ILLEGAL_INSTRUCTION",
        0xC0000006 => "EXCEPTION_IN_PAGE_ERROR",
        0xC000008C => "EXCEPTION_ARRAY_BOUNDS_EXCEEDED",
        0xC000008E => "EXCEPTION_FLT_DIVIDE_BY_ZERO",
        0xC0000094 => "EXCEPTION_INT_DIVIDE_BY_ZERO",
        0xC0000095 => "EXCEPTION_INT_OVERFLOW",
        0xC00000FD => "STATUS_STACK_OVERFLOW",
        0xC0000374 => "STATUS_HEAP_CORRUPTION",
        0xC0000409 => "STATUS_STACK_BUFFER_OVERRUN (__fastfail)",
        0xC0000602 => "STATUS_FAIL_FAST_EXCEPTION",
        0x80000003 => "EXCEPTION_BREAKPOINT",
        0x40010006 => "DBG_PRINTEXCEPTION_C",
        0xE06D7363 => "C++ exception (MSVC throw)",
        0x406D1388 => "MS_VC_EXCEPTION (thread name)",
        _ => "unknown exception",
    }
}

/// Access-violation flavor, from `ExceptionInformation[0]`. The distinction
/// matters: a READ of a freed vtable and a WRITE through a stale pointer are
/// different bugs, and this is the cheapest place to tell them apart.
pub fn av_flavor(kind: usize) -> &'static str {
    match kind {
        0 => "reading",
        1 => "writing",
        8 => "executing (DEP)",
        _ => "accessing",
    }
}

/// `module+0xoffset`, the form that makes a vendor-DLL frame greppable against
/// a driver release. A zero base (module unknown) degrades to the raw address
/// rather than printing a nonsense offset.
pub fn module_offset(module: &str, base: usize, addr: usize) -> String {
    if base == 0 || addr < base {
        format!("{module}+?")
    } else {
        format!("{module}+0x{:x}", addr - base)
    }
}

/// True when this invocation should skip installing. Pure so `self_test` can
/// pin it, and scanned from raw argv because the flag has to be readable
/// BEFORE `cli::parse_from` runs — the `settings::headless_args` shape.
pub fn disabled_by_args<I: IntoIterator<Item = S>, S: AsRef<str>>(args: I) -> bool {
    args.into_iter().any(|a| a.as_ref() == "--no-crash-handler")
}

// ---------------------------------------------------------------------------
// Windows implementation
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod imp {
    use super::{av_flavor, exception_name, module_offset, INSTALLED, MAX_FRAMES};
    use std::ffi::{c_void, CStr};
    use std::os::windows::io::AsRawHandle;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::OnceLock;
    use windows::core::{PCSTR, PCWSTR};
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::Diagnostics::Debug::{
        RtlLookupFunctionEntry, RtlVirtualUnwind, SetUnhandledExceptionFilter, CONTEXT,
        EXCEPTION_POINTERS, IMAGEHLP_LINEW64, MINIDUMP_EXCEPTION_INFORMATION,
        RTL_VIRTUAL_UNWIND_HANDLER_TYPE, SYMBOL_INFOW,
    };
    use windows::Win32::System::LibraryLoader::{
        GetModuleFileNameW, GetModuleHandleExW, GetModuleHandleW, GetProcAddress, LoadLibraryExW,
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        LOAD_LIBRARY_SEARCH_SYSTEM32,
    };
    use windows::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentProcessId, GetCurrentThreadId,
        GetCurrentThreadStackLimits, SetThreadStackGuarantee,
    };

    /// The MINIDUMP_TYPE bits we use, spelled numerically straight from
    /// dbghelp.h so this does not depend on windows-rs constant naming.
    const MINIDUMP_NORMAL: u32 = 0x0000;
    const MINIDUMP_WITH_FULL_MEMORY: u32 = 0x0002;
    const MINIDUMP_WITH_UNLOADED_MODULES: u32 = 0x0020;
    const MINIDUMP_WITH_INDIRECTLY_REFERENCED_MEMORY: u32 = 0x0040;
    const MINIDUMP_WITH_PROCESS_THREAD_DATA: u32 = 0x0100;
    const MINIDUMP_WITH_THREAD_INFO: u32 = 0x1000;

    /// Default: every thread's stack, the module list with versions, and the
    /// exception CONTEXT — a few MB, enough to open in WinDbg and walk any
    /// thread. Full memory is opt-in because this process's resident set is
    /// measured in gigabytes.
    const DUMP_DEFAULT: u32 = MINIDUMP_NORMAL
        | MINIDUMP_WITH_THREAD_INFO
        | MINIDUMP_WITH_UNLOADED_MODULES
        | MINIDUMP_WITH_INDIRECTLY_REFERENCED_MEMORY
        | MINIDUMP_WITH_PROCESS_THREAD_DATA;

    /// Hand-declared so nothing imports dbghelp (the pix.rs policy). Handles
    /// and the type flags ride as raw words; only the struct types come from
    /// windows-rs, because dbghelp.h packs some of them and getting a layout
    /// wrong by hand is a silent corruption.
    type MiniDumpWriteDumpFn = unsafe extern "system" fn(
        *mut c_void,
        u32,
        *mut c_void,
        u32,
        *const MINIDUMP_EXCEPTION_INFORMATION,
        *const c_void,
        *const c_void,
    ) -> i32;
    type SymFromAddrWFn =
        unsafe extern "system" fn(*mut c_void, u64, *mut u64, *mut SYMBOL_INFOW) -> i32;
    type SymGetLineFromAddrW64Fn =
        unsafe extern "system" fn(*mut c_void, u64, *mut u32, *mut IMAGEHLP_LINEW64) -> i32;

    /// dbghelp entry points, resolved once and only on an actual crash.
    struct DbgHelp {
        minidump: Option<MiniDumpWriteDumpFn>,
        sym_from_addr: Option<SymFromAddrWFn>,
        sym_get_line: Option<SymGetLineFromAddrW64Fn>,
    }

    static DBGHELP: OnceLock<Option<DbgHelp>> = OnceLock::new();

    fn dbghelp() -> Option<&'static DbgHelp> {
        DBGHELP
            .get_or_init(|| unsafe {
                // SYSTEM32, not the default search order: a bare name searches
                // the EXE DIRECTORY first, and this loads inside an already
                // faulting process — the one moment a stray dbghelp.dll next
                // to the exe should not be able to win. (The tree's usual
                // LOAD_WITH_ALTERED_SEARCH_PATH is a no-op for a bare name;
                // it exists for the absolute SDK paths dxc/oidn/pix load.)
                let module = LoadLibraryExW(
                    windows::core::w!("dbghelp.dll"),
                    None,
                    LOAD_LIBRARY_SEARCH_SYSTEM32,
                )
                .ok()?;
                let sym = |n: &CStr| GetProcAddress(module, PCSTR(n.as_ptr() as _));
                Some(DbgHelp {
                    minidump: sym(c"MiniDumpWriteDump").map(|f| std::mem::transmute(f)),
                    sym_from_addr: sym(c"SymFromAddrW").map(|f| std::mem::transmute(f)),
                    sym_get_line: sym(c"SymGetLineFromAddrW64").map(|f| std::mem::transmute(f)),
                })
            })
            .as_ref()
    }

    /// `SYMBOL_INFOW` with room for the name that trails it — dbghelp writes
    /// `MaxNameLen` wide chars starting at the struct's 1-element `Name`.
    #[repr(C)]
    struct SymBuf {
        si: SYMBOL_INFOW,
        _name_tail: [u16; MAX_SYM_NAME],
    }
    const MAX_SYM_NAME: usize = 1024;

    /// Symbolize through dbghelp the way cdb does.
    ///
    /// WHY NOT JUST `backtrace::resolve`: measured on this tree, backtrace
    /// resolves Rust addresses perfectly (name + file:line, inline frames
    /// expanded) but returns a NEARBY PUBLIC symbol with no line for addresses
    /// in the cc-built C++ — the gate's `cpp` arm reported
    /// `std::operator+<wchar_t,...>` and NO LINE for an address that cdb
    /// resolves, from the very same PDB, as `frshim_crash_test_inner` at its
    /// real `shim/ffx_shim.cpp` line. (Deliberately not quoting the line
    /// NUMBER: it moves with every edit to that file, and the fact worth
    /// recording is that one source had a line and the other did not.) The
    /// debug info is all there; only backtrace's inline-context path
    /// mis-handles it. So dbghelp is asked directly and its answer wins, with
    /// backtrace kept for the inline chain it IS good at.
    ///
    /// Deliberately does NOT call `SymInitialize`: backtrace already owns the
    /// process-wide dbghelp session, and a second initialize would fail — or
    /// worse, win the race and leave backtrace unable to symbolize anything.
    /// Every caller resolves through backtrace first, so the session exists.
    fn sym_dbghelp(addr: usize) -> Option<(String, Option<(String, u32)>)> {
        let dh = dbghelp()?;
        let from_addr = dh.sym_from_addr?;
        unsafe {
            let proc = GetCurrentProcess().0;
            let mut buf: SymBuf = std::mem::zeroed();
            buf.si.SizeOfStruct = std::mem::size_of::<SYMBOL_INFOW>() as u32;
            buf.si.MaxNameLen = MAX_SYM_NAME as u32;
            let mut disp: u64 = 0;
            if from_addr(proc, addr as u64, &mut disp, &mut buf.si) == 0 {
                return None;
            }
            let len = (buf.si.NameLen as usize).min(MAX_SYM_NAME);
            let name = String::from_utf16_lossy(std::slice::from_raw_parts(
                buf.si.Name.as_ptr(),
                len,
            ));
            if name.is_empty() {
                return None;
            }
            // Line info is best-effort: a public-only module (every vendor
            // DLL) legitimately has none.
            let line = dh.sym_get_line.and_then(|get_line| {
                let mut l: IMAGEHLP_LINEW64 = std::mem::zeroed();
                l.SizeOfStruct = std::mem::size_of::<IMAGEHLP_LINEW64>() as u32;
                let mut ldisp: u32 = 0;
                if get_line(proc, addr as u64, &mut ldisp, &mut l) == 0 {
                    return None;
                }
                let file = l.FileName.to_string().ok()?;
                Some((file, l.LineNumber))
            });
            Some((name, line))
        }
    }

    /// Vendor + system DLLs whose PRESENCE and PATH are worth recording: a
    /// driver-store path carries the driver version, which is the first thing
    /// that differs when "it worked last week".
    const WATCH_DLLS: &[&str] = &[
        "nvngx_dlssg.dll",
        "nvngx_dlssd.dll",
        "_nvngx.dll",
        "nvngx.dll",
        "libxess.dll",
        "libxess_fg.dll",
        "libxell.dll",
        "amd_fidelityfx_loader_dx12.dll",
        "amd_fidelityfx_dx12.dll",
        "dxcompiler.dll",
        "D3D12Core.dll",
        "d3d12.dll",
        "dxgi.dll",
        "OpenImageDenoise.dll",
        "onnxruntime.dll",
        "DirectML.dll",
        "WinPixEventRuntime.dll",
    ];

    /// Reentrancy guard: a fault *inside* the handler must die quietly rather
    /// than recurse. Also makes the first faulting thread the only reporter —
    /// two threads formatting into stderr at once produces an unreadable
    /// interleave exactly when readability matters most.
    static REPORTING: AtomicBool = AtomicBool::new(false);

    /// `rayon-7 (tid 1234)` — the Rust thread NAME as well as the OS id. The
    /// default panic hook printed the name and this one replaces it, so
    /// dropping it would be a regression: in a 32-worker rayon fan-out
    /// "rayon-7" is the identifier that says WHICH worker died, and a bare OS
    /// id says nothing on its own. A foreign (driver-created) thread has no
    /// Rust name and prints the id alone.
    fn thread_label() -> String {
        let tid = unsafe { GetCurrentThreadId() };
        match std::thread::current().name() {
            Some(n) => format!("{n} (tid {tid})"),
            None => format!("(tid {tid})"),
        }
    }

    /// Resolve an address to (module file name, module base).
    fn module_of(addr: usize) -> Option<(String, usize)> {
        if addr == 0 {
            return None;
        }
        unsafe {
            let mut h = HMODULE::default();
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                PCWSTR(addr as *const u16),
                &mut h,
            )
            .ok()?;
            let mut buf = [0u16; 512];
            let n = GetModuleFileNameW(Some(h), &mut buf);
            if n == 0 {
                return None;
            }
            let full = String::from_utf16_lossy(&buf[..n as usize]);
            let name = full.rsplit(['\\', '/']).next().unwrap_or(&full).to_string();
            Some((name, h.0 as usize))
        }
    }

    /// Full path of a loaded module by name, or None when it is not loaded.
    fn module_path(name: &str) -> Option<String> {
        unsafe {
            let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            let h = GetModuleHandleW(PCWSTR(wide.as_ptr())).ok()?;
            let mut buf = [0u16; 512];
            let n = GetModuleFileNameW(Some(h), &mut buf);
            if n == 0 {
                return None;
            }
            Some(String::from_utf16_lossy(&buf[..n as usize]))
        }
    }

    /// The faulting thread's own stack extent, `(low, high)` — the handler
    /// runs ON that thread, so the "current thread" API is the right one, and
    /// it is the documented way to ask (rather than poking the TEB by offset).
    ///
    /// Used to sanity-check an RSP before dereferencing it: an out-of-bounds
    /// read inside the exception filter is not a truncated report, it is NO
    /// report, because the reentrancy guard swallows the second fault. A
    /// degenerate answer widens to "accept anything", i.e. the unguarded
    /// behavior — never narrows to "reject everything", which would silently
    /// disable the leaf fallback.
    fn stack_bounds() -> (usize, usize) {
        unsafe {
            let (mut lo, mut hi) = (0usize, 0usize);
            GetCurrentThreadStackLimits(&mut lo, &mut hi);
            if lo >= hi {
                return (0, usize::MAX);
            }
            (lo, hi)
        }
    }

    /// Unwind from the EXCEPTION's OWN CONTEXT rather than from the handler's
    /// current stack.
    ///
    /// Walking the current stack (what `backtrace::trace` does) has to cross
    /// `KiUserExceptionDispatcher` to reach the fault, and MEASURED here that
    /// works for an ordinary frame and FAILS for a LEAF one: a leaf function
    /// modifies no stack and so gets no RUNTIME_FUNCTION entry, the lookup
    /// returns null, and the walk simply ends — the whole C++ arm of the gate
    /// reported eight frames of dispatcher noise and nothing else. That is not
    /// a test artifact: a vendor DLL faulting in a leaf would lose its stack
    /// the same way, which is precisely the case this module exists for.
    ///
    /// Seeding from the CONTEXT starts the walk AT the faulting instruction —
    /// no dispatcher frames to skip — and the null-entry case becomes a leaf
    /// FALLBACK (return address at [RSP], pop 8) instead of a dead end. This
    /// is the textbook x64 loop, i.e. what StackWalk64 does internally.
    fn stack_from_context(ctx: &CONTEXT) -> Vec<usize> {
        let mut pcs = Vec::with_capacity(MAX_FRAMES);
        // RtlVirtualUnwind MUTATES the context it walks, so take a copy — the
        // original belongs to the exception record the minidump still needs.
        let mut c = *ctx;
        let (stack_lo, stack_hi) = stack_bounds();
        for depth in 0..MAX_FRAMES {
            if c.Rip == 0 {
                break;
            }
            pcs.push(c.Rip as usize);
            let mut image_base: u64 = 0;
            let entry = unsafe { RtlLookupFunctionEntry(c.Rip, &mut image_base, None) };
            if entry.is_null() {
                // LEAF FALLBACK, SCOPED TO DEPTH 0 ON PURPOSE. A leaf can only
                // ever be the INNERMOST frame: every frame above it holds a
                // RETURN address, so its function made a call and therefore
                // carries unwind data. A null entry deeper in the stack means
                // code with no unwind info at all (JIT, a hand-written thunk),
                // where guessing [RSP] fabricates frames as readily as it
                // recovers them — and a fabricated stack in a crash report is
                // worse than a short one, because it is believed. Stopping
                // here loses nothing real; depth 0 is the case the C++ gate
                // arm actually established.
                let rsp = c.Rsp as usize;
                // The read must also be SAFE: a fault inside the filter is
                // caught by the REPORTING guard, which means NO report at all.
                // The faulting thread's own TEB gives its stack extent, so a
                // wild RSP (the stack-overflow shape) is rejected rather than
                // dereferenced.
                if depth > 0 || rsp < stack_lo || rsp.saturating_add(8) > stack_hi {
                    break;
                }
                let ret = unsafe { (rsp as *const u64).read_volatile() };
                if ret == 0 {
                    break;
                }
                c.Rip = ret;
                c.Rsp += 8;
            } else {
                let mut handler_data: *mut c_void = std::ptr::null_mut();
                let mut establisher: u64 = 0;
                unsafe {
                    RtlVirtualUnwind(
                        // UNW_FLAG_NHANDLER: unwind only, run no handlers.
                        RTL_VIRTUAL_UNWIND_HANDLER_TYPE(0),
                        image_base,
                        c.Rip,
                        entry,
                        &mut c,
                        &mut handler_data,
                        &mut establisher,
                        None,
                    );
                }
            }
        }
        pcs
    }

    /// Format a list of program counters. `first_is_exact` says whether pc[0]
    /// is a faulting address (symbolize it as-is) or a return address (a
    /// return address points AFTER the call, so symbolizing it can land in the
    /// next line — or the next function entirely — hence the -1).
    fn format_pcs(pcs: &[usize], first_is_exact: bool) -> String {
        let mut out = String::new();
        if pcs.is_empty() {
            out.push_str("   <stack walk produced no frames>\n");
            return out;
        }
        for (i, &pc) in pcs.iter().enumerate() {
            let (m, base) = module_of(pc).unwrap_or_else(|| ("<unknown>".to_string(), 0));
            let mark = if i == 0 && first_is_exact { ">>" } else { "  " };
            out.push_str(&format!(
                "{mark} #{i:<3} 0x{pc:016x}  {}\n",
                module_offset(&m, base, pc)
            ));
            let sym_at = if i == 0 && first_is_exact { pc } else { pc.saturating_sub(1) };

            // backtrace FIRST: it owns the process-wide dbghelp session (so
            // this call is also what initializes it), and where it works it is
            // the BETTER answer — it expands the inline chain innermost-first,
            // which plain SymFromAddrW cannot.
            let mut inline: Vec<(String, Option<(String, u32)>)> = Vec::new();
            backtrace::resolve(sym_at as *mut c_void, |s| {
                if let Some(name) = s.name().map(|n| n.to_string()) {
                    let line = match (s.filename(), s.lineno()) {
                        (Some(f), Some(l)) => Some((f.display().to_string(), l)),
                        _ => None,
                    };
                    inline.push((name, line));
                }
            });

            // LINE INFO IS THE ARBITER, and it is what tells the two failure
            // modes apart. A real symbol record carries file:line; backtrace's
            // bad answer for C++ addresses is a nearby PUBLIC symbol, and
            // publics have no lines. So: trust backtrace when any entry has a
            // line, else fall back to dbghelp — which is exactly the case that
            // recovers `frshim_crash_test_inner  shim/ffx_shim.cpp:596` where
            // backtrace offered `std::operator+<wchar_t,...>` and no line.
            // A vendor DLL has no lines from either, and then they agree.
            if inline.iter().any(|(_, l)| l.is_some()) {
                for (name, line) in &inline {
                    out.push_str(&format!("            {name}\n"));
                    if let Some((f, l)) = line {
                        out.push_str(&format!("            {f}:{l}\n"));
                    }
                }
            } else if let Some((name, line)) = sym_dbghelp(sym_at) {
                out.push_str(&format!("            {name}\n"));
                if let Some((f, l)) = line {
                    out.push_str(&format!("            {f}:{l}\n"));
                }
            } else if let Some((name, _)) = inline.first() {
                out.push_str(&format!("            {name}\n"));
            } else {
                out.push_str("            <no symbol>\n");
            }
        }
        out
    }

    /// Frames belonging to the reporting machinery itself rather than to the
    /// bug — on the panic path these sit on top of every stack and push the
    /// interesting frames out of view.
    fn is_reporting_noise(name: &str) -> bool {
        name.contains("frustracer::crash::")
            || name.starts_with("backtrace::")
            || name.starts_with("std::panicking::")
            || name.starts_with("core::panicking::")
            || name.starts_with("std::panic::")
            || name.starts_with("std::sys::backtrace::")
    }

    /// Walk the CURRENT stack — the panic path, which has no CONTEXT to seed
    /// from. Every pc here is a return address.
    ///
    /// Leading frames from the hook and the panic runtime are dropped: the
    /// header already says it is a panic and where, so those frames are pure
    /// noise ahead of the code that actually panicked.
    fn format_current_stack() -> String {
        let mut pcs = Vec::with_capacity(MAX_FRAMES);
        backtrace::trace(|frame| {
            pcs.push(frame.ip() as usize);
            pcs.len() < MAX_FRAMES
        });
        let skip = pcs
            .iter()
            .position(|&pc| {
                let mut keep = true;
                backtrace::resolve(pc.saturating_sub(1) as *mut c_void, |s| {
                    if let Some(n) = s.name().map(|n| n.to_string()) {
                        if is_reporting_noise(&n) {
                            keep = false;
                        }
                    }
                });
                keep
            })
            // All noise (or nothing resolved) — print it all rather than
            // print nothing.
            .unwrap_or(0);
        format_pcs(&pcs[skip..], false)
    }

    fn format_modules() -> String {
        let mut out = String::new();
        for name in WATCH_DLLS {
            if let Some(p) = module_path(name) {
                out.push_str(&format!("   {p}\n"));
            }
        }
        if out.is_empty() {
            out.push_str("   <none of the watched runtimes are loaded>\n");
        }
        out
    }

    /// `<exe dir>/frustracer-crash-<pid>.<ext>` — the settings-file rule
    /// (`settings::dir()`), which is this project's one side-channel-file
    /// convention. The pid keeps repeated runs from overwriting each other.
    fn artifact(ext: &str) -> PathBuf {
        let pid = unsafe { GetCurrentProcessId() };
        crate::settings::dir().join(format!("frustracer-crash-{pid}.{ext}"))
    }

    fn write_minidump(info: *const EXCEPTION_POINTERS, path: &std::path::Path) -> Result<(), String> {
        unsafe {
            let write = dbghelp()
                .and_then(|d| d.minidump)
                .ok_or_else(|| "dbghelp.dll has no MiniDumpWriteDump".to_string())?;

            let file = std::fs::File::create(path).map_err(|e| format!("{e}"))?;
            let mut mei = MINIDUMP_EXCEPTION_INFORMATION {
                ThreadId: GetCurrentThreadId(),
                ExceptionPointers: info as *mut EXCEPTION_POINTERS,
                // FALSE: the pointers are in OUR address space, which is the
                // whole point of dumping ourselves.
                ClientPointers: Default::default(),
            };
            let kind = if std::env::var_os("FR_CRASH_FULLDUMP").is_some() {
                DUMP_DEFAULT | MINIDUMP_WITH_FULL_MEMORY
            } else {
                DUMP_DEFAULT
            };
            let ok = write(
                GetCurrentProcess().0,
                GetCurrentProcessId(),
                file.as_raw_handle(),
                kind,
                &mut mei as *const MINIDUMP_EXCEPTION_INFORMATION,
                std::ptr::null(),
                std::ptr::null(),
            );
            if ok == 0 {
                return Err("MiniDumpWriteDump returned FALSE".to_string());
            }
            Ok(())
        }
    }

    /// Emit the report to stderr and to the crash log next to the exe.
    fn emit(report: &str) {
        eprint!("{report}");
        let txt = artifact("txt");
        match std::fs::write(&txt, report) {
            Ok(()) => eprintln!("crash: report written to {}", txt.display()),
            Err(e) => eprintln!("crash: could not write {} ({e})", txt.display()),
        }
    }

    unsafe extern "system" fn handler(info: *const EXCEPTION_POINTERS) -> i32 {
        // EXCEPTION_EXECUTE_HANDLER = 1: report, then let the process die
        // without the Windows Error Reporting dialog.
        const EXECUTE_HANDLER: i32 = 1;

        if REPORTING.swap(true, Ordering::SeqCst) {
            // Either the handler itself faulted, or a second thread arrived.
            // Say nothing and die — a recursive report is worse than none.
            return EXECUTE_HANDLER;
        }

        let (code, fault_ip, av, ctx) = unsafe {
            let Some(p) = info.as_ref() else {
                eprintln!("crash: exception with no EXCEPTION_POINTERS — nothing to report");
                return EXECUTE_HANDLER;
            };
            let Some(rec) = p.ExceptionRecord.as_ref() else {
                eprintln!("crash: exception with no EXCEPTION_RECORD — nothing to report");
                return EXECUTE_HANDLER;
            };
            let code = rec.ExceptionCode.0 as u32;
            let ip = rec.ExceptionAddress as usize;
            let av = if code == 0xC000_0005 && rec.NumberParameters >= 2 {
                Some((rec.ExceptionInformation[0], rec.ExceptionInformation[1]))
            } else {
                None
            };
            (code, ip, av, p.ContextRecord.as_ref())
        };

        // The single most valuable line, printed FIRST and with the least
        // machinery behind it: which module faulted. Everything after this can
        // fail without costing the answer to "was it us or the driver".
        let (mname, mbase) = module_of(fault_ip).unwrap_or_else(|| ("<unknown>".to_string(), 0));
        eprintln!();
        eprintln!("=== FRUSTRACER CRASH ===");
        eprintln!("crash: {} (0x{code:08X})", exception_name(code));
        if let Some((kind, addr)) = av {
            eprintln!("crash:   {} address 0x{addr:016x}", av_flavor(kind));
        }
        eprintln!(
            "crash:   at 0x{fault_ip:016x}  {}",
            module_offset(&mname, mbase, fault_ip)
        );

        let mut report = String::new();
        report.push_str("\n=== FRUSTRACER CRASH ===\n");
        report.push_str(&format!("{} (0x{code:08X})\n", exception_name(code)));
        if let Some((kind, addr)) = av {
            report.push_str(&format!("  {} address 0x{addr:016x}\n", av_flavor(kind)));
        }
        report.push_str(&format!(
            "  at 0x{fault_ip:016x}  {}\n",
            module_offset(&mname, mbase, fault_ip)
        ));
        report.push_str(&format!("  thread {}\n", thread_label()));
        report.push_str("\nfaulting thread stack (>> = faulting frame):\n");
        // Prefer the exception's own CONTEXT: it starts the walk AT the fault
        // and survives a leaf faulting frame. Fall back to the current stack
        // only if there is no context at all.
        match ctx {
            Some(c) => report.push_str(&format_pcs(&stack_from_context(c), true)),
            None => {
                report.push_str("   <no CONTEXT — walking the handler's own stack instead>\n");
                report.push_str(&format_current_stack());
            }
        }
        report.push_str("\nloaded runtimes:\n");
        report.push_str(&format_modules());

        // stderr already carried the header; print the rest, then the file.
        eprint!("{}", &report[report.find("\nfaulting thread stack").unwrap_or(0)..]);
        let txt = artifact("txt");
        match std::fs::write(&txt, &report) {
            Ok(()) => eprintln!("crash: report written to {}", txt.display()),
            Err(e) => eprintln!("crash: could not write {} ({e})", txt.display()),
        }

        let dmp = artifact("dmp");
        match write_minidump(info, &dmp) {
            Ok(()) => eprintln!(
                "crash: minidump written to {} (open in WinDbg for every thread's stack)",
                dmp.display()
            ),
            Err(e) => eprintln!("crash: no minidump — {e}"),
        }
        eprintln!("=== END CRASH ===");

        EXECUTE_HANDLER
    }

    fn panic_hook(info: &std::panic::PanicHookInfo<'_>) {
        // A Rust panic is not an SEH exception, so it needs its own entry
        // point — and release does NOT set panic = "abort", so unwinding is
        // live and a panic during teardown can provoke a SECOND fault whose
        // stack says nothing about the cause (the documented incident this
        // module's header cites). Reporting AT the panic point is the fix.
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic payload>".to_string());
        let loc = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown location>".to_string());
        let who = thread_label();

        // The message is built BEFORE the guard so that losing the race still
        // says something. Installing this hook REPLACES the default one, so a
        // bare `return` here would make a second, DIFFERENT panic vanish
        // entirely — not even "thread panicked" — which is a real loss under a
        // rayon fan-out where several workers can fail at once for unrelated
        // reasons. One line each: the full report belongs to the first, the
        // rest are recorded rather than dropped.
        if REPORTING.swap(true, Ordering::SeqCst) {
            eprintln!("crash: panic on thread {who} at {loc} (report suppressed — another is in flight): {msg}");
            return;
        }

        let mut report = String::new();
        report.push_str("\n=== FRUSTRACER PANIC ===\n");
        report.push_str(&format!("panicked at {loc}\n  {msg}\n"));
        report.push_str(&format!("  thread {who}\n"));
        report.push_str("\nstack:\n");
        report.push_str(&format_current_stack());
        emit(&report);
        eprintln!("=== END PANIC ===");
        // Let unwinding proceed: a caught panic must stay caught.
        REPORTING.store(false, Ordering::SeqCst);
    }

    /// Is OUR filter still the installed one? `SetUnhandledExceptionFilter`
    /// returns the previous filter, so setting ours and immediately restoring
    /// what was there is a read with no net change.
    ///
    /// This exists because there IS one real way the handler can be bypassed:
    /// nothing stops a vendor DLL (NGX, a driver, ffx) from installing its own
    /// top-level filter and displacing ours. Nothing "destructs" the handler —
    /// the filter is process state and our code stays mapped — but it can be
    /// OVERWRITTEN, and that is worth being able to observe rather than assume.
    fn filter_is_ours() -> bool {
        unsafe {
            let cur = SetUnhandledExceptionFilter(Some(handler));
            SetUnhandledExceptionFilter(cur);
            cur.map(|f| f as usize) == Some(handler as *const () as usize)
        }
    }

    /// THE PRECONDITION BOTH POST-MAIN LEVERS DEPEND ON, stated at arm time
    /// because a probe that silently does nothing is worse than no probe:
    /// someone running `FR_CRASH_VERIFY=1 --check`, seeing no line, would
    /// reasonably conclude the filter had been displaced.
    ///
    /// MEASURED, both directions, same binary: the interactive path (which
    /// RETURNS from `main`, so the CRT's `mainCRTStartup` calls `exit()`)
    /// fires the hook and faults with `execute_onexit_table` in the stack;
    /// `--check`, which leaves through `std::process::exit`, exits 0 and never
    /// runs it. Rust's `process::exit` is `ExitProcess` on Windows, which
    /// terminates without walking the CRT's atexit list — so EVERY headless
    /// path in this program (`--check*`, `--spin`, `--cinematic`, every
    /// `exit(2)` argument error) bypasses these two levers by construction.
    const EXIT_ROUTE_NOTE: &str =
        "[late] hooks fire only when main RETURNS (the interactive path) — \
         std::process::exit is ExitProcess, which skips the CRT atexit list, \
         so --check*/--spin never reach them";

    /// Runs from a CRT `atexit` handler, i.e. AFTER `main` has returned —
    /// registered first so it runs last (atexit is LIFO). This is the probe
    /// for "does the handler still cover the post-main phase", which is where
    /// static destructors and DLL_PROCESS_DETACH live.
    extern "C" fn late_hook() {
        if std::env::var_os("FR_CRASH_VERIFY").is_some() {
            if filter_is_ours() {
                eprintln!("crash: [late] filter still installed after main returned");
            } else {
                eprintln!(
                    "crash: [late] ANOTHER MODULE REPLACED the unhandled-exception filter — \
                     crashes from here on will not be reported by us"
                );
            }
        }
        if matches!(fault_arm().as_deref(), Some("atexit")) {
            eprintln!("crash: [late] faulting from an atexit handler (post-main)");
            unsafe { crate::crash::frshim_crash_test() }
        }
    }

    /// The `FR_CRASH_TEST` value, lowercased — read in two places.
    fn fault_arm() -> Option<String> {
        std::env::var_os("FR_CRASH_TEST").map(|v| v.to_string_lossy().to_ascii_lowercase())
    }

    /// Deliberate faults, so "it symbolizes both languages" is testable rather
    /// than asserted. The `cpp` arm is the load-bearing one: it faults inside
    /// a shim translation unit, so a report naming `shim/*.cpp:LINE` proves
    /// the one-PDB claim end to end. The `atexit` arm is the other: it faults
    /// after `main` has returned, proving the handler outlives it.
    fn fault_injection() {
        let Some(v) = fault_arm() else {
            return;
        };
        eprintln!("crash: FR_CRASH_TEST={v} — faulting deliberately (dev lever)");
        match v.as_str() {
            // Armed here, fired from late_hook after main returns.
            "atexit" => {
                eprintln!("crash: armed for the post-main atexit phase");
                eprintln!("crash: {EXIT_ROUTE_NOTE}");
            }
            "deref" => unsafe {
                std::ptr::write_volatile(std::ptr::null_mut::<u8>(), 1);
            },
            "cpp" => unsafe { crate::crash::frshim_crash_test() },
            "panic" => panic!("FR_CRASH_TEST=panic — deliberate panic"),
            "overflow" => {
                // Unbounded on purpose — this arm exists to blow the stack.
                #[allow(unconditional_recursion)]
                fn recurse(n: u64) -> u64 {
                    // +1 each level so the optimizer cannot turn this into a
                    // loop; black_box keeps the result live.
                    std::hint::black_box(recurse(n + 1) + 1)
                }
                std::hint::black_box(recurse(1));
            }
            other => eprintln!(
                "crash: FR_CRASH_TEST={other} is not one of \
                 deref|cpp|panic|overflow|atexit — ignored"
            ),
        }
    }

    pub fn install() {
        if std::env::var_os("FR_NO_CRASH").is_some() {
            eprintln!("crash: FR_NO_CRASH set — handler not installed");
            return;
        }
        if INSTALLED.swap(true, Ordering::SeqCst) {
            return;
        }
        unsafe {
            SetUnhandledExceptionFilter(Some(handler));
            // Reserve a slice of stack so a STATUS_STACK_OVERFLOW still leaves
            // room to run the handler at all. Per-thread and set on main only;
            // a worker overflow degrades to the bare header line.
            let mut bytes: u32 = 64 * 1024;
            let _ = SetThreadStackGuarantee(&mut bytes);
        }
        std::panic::set_hook(Box::new(panic_hook));
        // Register the post-main probe FIRST so it runs LAST (atexit is LIFO).
        // Only when a lever asks for it: a default session registers nothing.
        let verify = std::env::var_os("FR_CRASH_VERIFY").is_some();
        if verify || matches!(fault_arm().as_deref(), Some("atexit")) {
            unsafe { crate::crash::frshim_register_atexit(late_hook) };
            // The atexit ARM prints this itself (below, next to its own line);
            // FR_CRASH_VERIFY has no other announcement, so it says it here.
            if verify {
                eprintln!("crash: FR_CRASH_VERIFY=1 — {EXIT_ROUTE_NOTE}");
            }
        }
        fault_injection();
    }

    /// Windows half of the module's gate: address resolution has to actually
    /// work, and has to FAIL on an address in no module (otherwise "resolves"
    /// is vacuous).
    pub fn self_test_win() -> Result<(), String> {
        let probe = self_test_win as *const () as usize;
        let (name, base) = module_of(probe)
            .ok_or_else(|| "module_of failed on an address inside our own image".to_string())?;
        if !name.to_ascii_lowercase().ends_with(".exe") {
            return Err(format!("own address resolved to '{name}', expected the .exe"));
        }
        if base == 0 || probe <= base {
            return Err(format!("bad module base 0x{base:x} for probe 0x{probe:x}"));
        }
        if module_of(0).is_some() {
            return Err("module_of(0) resolved to a module — the lookup is vacuous".to_string());
        }
        // A crafted CStr is the only thing the dbghelp path needs at gate time;
        // dbghelp itself is deliberately NOT loaded here (--check stays
        // DLL-free — it loads on a real crash only).
        let _ = CStr::from_bytes_with_nul(b"MiniDumpWriteDump\0")
            .map_err(|e| format!("bad symbol literal: {e}"))?;
        Ok(())
    }
}

#[cfg(windows)]
unsafe extern "C" {
    /// Deliberate fault inside a C++ translation unit — see `FR_CRASH_TEST`.
    /// Lives in `shim/ffx_shim.cpp`, which is the shim built unconditionally
    /// on Windows (the DLSS shims are gated on the SDK being present).
    fn frshim_crash_test();
    /// Register a callback on the CRT's `atexit` list — Rust has no way to do
    /// this, and the post-main phase is what the probe is for.
    fn frshim_register_atexit(cb: extern "C" fn());
}

#[cfg(windows)]
pub use imp::install;

/// Non-Windows builds have no SEH, no dbghelp and no PDB — the headless gates
/// still build and run, they simply have no handler to install.
#[cfg(not(windows))]
pub fn install() {}

/// Gate: the pure tables and (on Windows) address resolution. Deliberately
/// does NOT fault, does not load dbghelp, and does not perturb the installed
/// filter — the `progress::self_test` discipline of leaving globals as it
/// found them.
pub fn self_test() -> Result<(), String> {
    // Exception naming, including the ones this project has actually hit.
    if exception_name(0xC0000005) != "EXCEPTION_ACCESS_VIOLATION" {
        return Err("0xC0000005 must name the access violation".into());
    }
    if exception_name(0xC00000FD) != "STATUS_STACK_OVERFLOW" {
        return Err("0xC00000FD must name the stack overflow".into());
    }
    if exception_name(0x1234) != "unknown exception" {
        return Err("an unlisted code must fall through to 'unknown exception'".into());
    }

    // AV flavor: read vs write is the difference between a freed-vtable read
    // and a stale-pointer write, so the mapping is pinned rather than trusted.
    for (kind, want) in [(0usize, "reading"), (1, "writing"), (8, "executing (DEP)")] {
        if av_flavor(kind) != want {
            return Err(format!("av_flavor({kind}) should be '{want}'"));
        }
    }
    if av_flavor(99) != "accessing" {
        return Err("an unknown AV kind must fall through to 'accessing'".into());
    }

    // module+offset arithmetic, including both degenerate arms.
    if module_offset("a.dll", 0x1000, 0x1234) != "a.dll+0x234" {
        return Err("module_offset arithmetic is wrong".into());
    }
    if module_offset("a.dll", 0, 0x1234) != "a.dll+?" {
        return Err("a zero base must not produce a fabricated offset".into());
    }
    if module_offset("a.dll", 0x2000, 0x1000) != "a.dll+?" {
        return Err("an address below the base must not underflow".into());
    }

    // The kill lever is a raw-argv scan (it has to run before the parser), so
    // pin that it matches only the exact flag.
    if !disabled_by_args(["--gpu", "--no-crash-handler"]) {
        return Err("--no-crash-handler must disable the handler".into());
    }
    if disabled_by_args(["--gpu", "--no-crash"]) {
        return Err("a prefix of the flag must not disable the handler".into());
    }

    #[cfg(windows)]
    imp::self_test_win()?;

    Ok(())
}
