//! Runtime PIX event loader — named Begin/End brackets on D3D12 command
//! lists so PIX GPU captures (and RenderDoc) show the wavefront levels,
//! feeds, and upscaler evaluates by name. Same footprint policy as
//! Streamline/OIDN/XeSS/DXC: nothing links — `WinPixEventRuntime.dll` (from
//! a PIX install or the WinPixEventRuntime NuGet, dropped under the
//! gitignored `SDKs/pix/bin/x64`) is `LoadLibraryExW`'d at runtime and three
//! exports are resolved with GetProcAddress. Markers are opt-in
//! (`--pix-markers`) and every call is an inert no-op when the DLL is absent
//! or the flag is off, so default sessions and every `--check*` stay
//! byte-identical and DLL-free.

use std::ffi::{c_void, CStr, CString};
use std::sync::OnceLock;
use windows::core::{Interface, PCSTR, PCWSTR};
use windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList;
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LoadLibraryExW, LOAD_WITH_ALTERED_SEARCH_PATH,
};

type BeginFn = unsafe extern "system" fn(*mut c_void, u64, PCSTR);
type EndFn = unsafe extern "system" fn(*mut c_void);
type MarkerFn = unsafe extern "system" fn(*mut c_void, u64, PCSTR);

struct Pix {
    begin: BeginFn,
    end: EndFn,
    marker: MarkerFn,
}

/// None until `init`; stays None (all calls no-op) when markers are off or
/// the DLL is missing.
static PIX: OnceLock<Option<Pix>> = OnceLock::new();

/// PIX default event color (0 lets PIX pick from the name hash).
const COLOR: u64 = 0;

/// Load the event runtime. Call once at startup, only when the user asked
/// for markers; a missing DLL prints one loud line and leaves everything
/// inert (never an error — captures without markers still work).
pub fn init(dir: &str, enabled: bool) {
    let _ = PIX.set(if enabled { load(dir) } else { None });
}

fn load(dir: &str) -> Option<Pix> {
    let abs = std::fs::canonicalize(dir)
        .map(|p| p.to_string_lossy().trim_start_matches(r"\\?\").to_string())
        .unwrap_or_else(|_| dir.to_string());
    let path = format!("{abs}\\WinPixEventRuntime.dll");
    let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
    let module =
        match unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, LOAD_WITH_ALTERED_SEARCH_PATH) }
        {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "pix: --pix-markers requested but {path} failed to load ({e}); markers off. \
                     Drop WinPixEventRuntime.dll there (PIX install or the WinPixEventRuntime \
                     NuGet) or point --pix-path / FRUSTRACER_PIX_PATH at it."
                );
                return None;
            }
        };
    let sym = |name: &CStr| unsafe { GetProcAddress(module, PCSTR(name.as_ptr() as _)) };
    let (Some(begin), Some(end), Some(marker)) = (
        sym(c"PIXBeginEventOnCommandList"),
        sym(c"PIXEndEventOnCommandList"),
        sym(c"PIXSetMarkerOnCommandList"),
    ) else {
        eprintln!("pix: {path} is missing the OnCommandList exports; markers off.");
        return None;
    };
    eprintln!("pix: command-list markers on ({path})");
    unsafe {
        Some(Pix {
            begin: std::mem::transmute::<_, BeginFn>(begin),
            end: std::mem::transmute::<_, EndFn>(end),
            marker: std::mem::transmute::<_, MarkerFn>(marker),
        })
    }
}

fn table() -> Option<&'static Pix> {
    PIX.get().and_then(|o| o.as_ref())
}

/// RAII event bracket: `let _e = pix::scope(list, c"wavefront");` ends the
/// event when dropped. Holds a COM clone (a refcount bump), not a borrow, so
/// the guard never conflicts with `&mut` access to the surrounding frame
/// state. Zero work when markers are off.
///
/// The bracket drives TWO instruments: the PIX event and (under
/// `--gpu-timing`) a `gputime` timestamp region of the same name. Keeping
/// them on one bracket is deliberate — PIX cannot analyze a capture on an
/// Intel adapter at all, so the timestamps are the only numbers available
/// there, and a marker that wasn't timed would be a silent hole in the table.
/// Both are independently opt-in and both are inert when off.
pub struct PixScope(Option<ID3D12GraphicsCommandList>, super::gputime::TimeScope);

impl Drop for PixScope {
    fn drop(&mut self) {
        if let (Some(list), Some(p)) = (&self.0, table()) {
            unsafe { (p.end)(list.as_raw()) };
        }
        // .1 (the TimeScope) closes its timestamp in its own Drop, after this.
    }
}

pub fn scope(list: &ID3D12GraphicsCommandList, name: &CStr) -> PixScope {
    let t = super::gputime::scope(list, || name.to_string_lossy().into_owned());
    match table() {
        Some(p) => {
            unsafe { (p.begin)(list.as_raw(), COLOR, PCSTR(name.as_ptr() as _)) };
            PixScope(Some(list.clone()), t)
        }
        None => PixScope(None, t),
    }
}

/// Formatted variant for dynamic names (`level {d}`); allocates only when
/// markers (or timing) are actually on.
pub fn scope_fmt(list: &ID3D12GraphicsCommandList, args: std::fmt::Arguments) -> PixScope {
    let t = super::gputime::scope(list, || args.to_string());
    match table() {
        Some(p) => {
            let name = CString::new(args.to_string()).unwrap_or_default();
            unsafe { (p.begin)(list.as_raw(), COLOR, PCSTR(name.as_ptr() as _)) };
            PixScope(Some(list.clone()), t)
        }
        None => PixScope(None, t),
    }
}

/// One-shot marker (no bracket).
#[allow(dead_code)]
pub fn marker(list: &ID3D12GraphicsCommandList, name: &CStr) {
    if let Some(p) = table() {
        unsafe { (p.marker)(list.as_raw(), COLOR, PCSTR(name.as_ptr() as _)) };
    }
}
