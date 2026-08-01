//! DXGI adapter selection. This machine has three GPUs (Intel Arc, NVIDIA
//! RTX, AMD iGPU); DLSS requires the NVIDIA adapter and FSR Ray Regeneration
//! an AMD (RDNA4) one, so we enumerate explicitly with a vendor preference
//! instead of trusting adapter 0.

use windows::core::{Interface, Result};
use windows::Win32::Graphics::Dxgi::*;

const VENDOR_NVIDIA: u32 = 0x10DE;
const VENDOR_AMD: u32 = 0x1002;
const VENDOR_INTEL: u32 = 0x8086;

/// The vendor of the adapter we actually PICKED — the input to every
/// vendor-aware default (`main::vendor_defaults`, `trace::leaf_group`).
///
/// Deliberately not the same type as `Prefer`: a preference is what the user
/// asked for and may not be honored (a box without that vendor falls back to
/// the first hardware adapter), while this is what the device is. Defaults must
/// key off the fact, never the request.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    /// Anything else, including a software/virtual adapter that slipped the
    /// SOFTWARE flag. Always takes the cross-vendor default — an unknown GPU
    /// is exactly the case where a tuned constant is least likely to hold.
    Other,
}

impl Vendor {
    fn of(id: u32) -> Self {
        match id {
            VENDOR_NVIDIA => Vendor::Nvidia,
            VENDOR_AMD => Vendor::Amd,
            VENDOR_INTEL => Vendor::Intel,
            _ => Vendor::Other,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Vendor::Nvidia => "NVIDIA",
            Vendor::Amd => "AMD",
            Vendor::Intel => "Intel",
            Vendor::Other => "unknown-vendor",
        }
    }
}

/// The last `pick()`'s vendor, as a process-global so consumers that never see
/// an `AdapterPick` can read it — specifically the KERNEL-ASSEMBLY constants in
/// trace.rs, which are chosen long after the device exists and are threaded
/// through nothing.
///
/// Recording it inside `pick()` is what makes it unforgettable: every path that
/// obtains a device (GpuContext, HeadlessGpu, every `--check*` suite) goes
/// through that one function, so a new device path cannot silently inherit a
/// stale vendor. One device per process in practice; last-writer-wins is the
/// correct rule regardless, since the newest pick IS the device in use.
static PICKED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

/// The picked adapter's vendor, or `Other` before any pick (the conservative
/// answer — cross-vendor defaults).
pub fn picked_vendor() -> Vendor {
    match PICKED.load(std::sync::atomic::Ordering::Relaxed) {
        0 => Vendor::Nvidia,
        1 => Vendor::Amd,
        2 => Vendor::Intel,
        _ => Vendor::Other,
    }
}

fn record_picked(v: Vendor) {
    let code = match v {
        Vendor::Nvidia => 0,
        Vendor::Amd => 1,
        Vendor::Intel => 2,
        Vendor::Other => 3,
    };
    PICKED.store(code, std::sync::atomic::Ordering::Relaxed);
}

/// Which vendor's best-VRAM adapter to prefer (never a hard requirement —
/// the caller's feature-support probe is the real gate; a wrong-vendor pick
/// just reports unsupported and falls back to the plain path).
#[derive(Clone, Copy, PartialEq)]
pub enum Prefer {
    Nvidia,
    Amd,
    Intel,
}

pub struct AdapterPick {
    pub adapter: IDXGIAdapter4,
    pub name: String,
    pub vendor: Vendor,
}

fn desc_name(desc: &DXGI_ADAPTER_DESC3) -> String {
    let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(128);
    String::from_utf16_lossy(&desc.Description[..len])
}

/// Pick the preferred vendor's adapter with the most VRAM; fall back to the
/// first hardware adapter (high-performance order).
pub fn pick(factory: &IDXGIFactory6, prefer: Prefer) -> std::result::Result<AdapterPick, String> {
    let want = match prefer {
        Prefer::Nvidia => VENDOR_NVIDIA,
        Prefer::Amd => VENDOR_AMD,
        Prefer::Intel => VENDOR_INTEL,
    };
    let mut best: Option<(IDXGIAdapter4, DXGI_ADAPTER_DESC3)> = None;
    let mut fallback: Option<(IDXGIAdapter4, DXGI_ADAPTER_DESC3)> = None;
    for i in 0.. {
        let adapter: IDXGIAdapter4 = match unsafe {
            factory.EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
        } {
            Ok(a) => a,
            Err(_) => break,
        };
        let desc = unsafe { adapter.GetDesc3() }.map_err(|e| e.to_string())?;
        if (desc.Flags & DXGI_ADAPTER_FLAG3_SOFTWARE) != DXGI_ADAPTER_FLAG3_NONE {
            continue;
        }
        if fallback.is_none() {
            fallback = Some((adapter.clone(), desc));
        }
        if desc.VendorId == want
            && best
                .as_ref()
                .is_none_or(|(_, b)| desc.DedicatedVideoMemory > b.DedicatedVideoMemory)
        {
            best = Some((adapter, desc));
        }
    }
    let picked = match (best, fallback) {
        (Some(p), _) => p,
        (None, Some(p)) => p,
        (None, None) => return Err("no hardware DXGI adapter found".into()),
    };
    let name = desc_name(&picked.1);
    let vendor = Vendor::of(picked.1.VendorId);
    record_picked(vendor);
    Ok(AdapterPick { adapter: picked.0, name, vendor })
}

pub fn create_factory(debug: bool) -> Result<IDXGIFactory6> {
    let flags = if debug { DXGI_CREATE_FACTORY_DEBUG } else { DXGI_CREATE_FACTORY_FLAGS(0) };
    unsafe { CreateDXGIFactory2(flags) }
}


/// (current usage, budget) of the device's adapter's LOCAL memory segment —
/// the scene-upload diagnostic: WDDM demotes over-budget commits silently
/// (10-100× slowdown, no error), so init prints where it landed. Best-effort:
/// None on any failure.
pub fn vram_info(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
) -> Option<(u64, u64)> {
    let factory: IDXGIFactory4 = create_factory(false).ok()?.cast().ok()?;
    let luid = unsafe { device.GetAdapterLuid() };
    let adapter: IDXGIAdapter3 = unsafe { factory.EnumAdapterByLuid(luid) }.ok()?;
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }
        .ok()?;
    Some((info.CurrentUsage, info.Budget))
}
