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
}

/// The last `pick()`'s vendor, as a process-global for consumers that never see
/// an `AdapterPick` and for which "the session's adapter" is genuinely the right
/// question — SESSION POLICY: `main::vendor_defaults` (which render mode a
/// flagless session starts in) and the `--spin` warm-up count. Those describe
/// the session, not a device, so a global is correct for them.
///
/// **It is NOT the input to per-device decisions.** Kernel-assembly constants
/// and driver-defect refusals must use `vendor_of_device` instead: with two
/// devices live (`--dual-gpu`) this global is last-writer-wins and would compile
/// one device's kernels against the other's vendor. The AMD candidate-TMin
/// workaround (`trace::cand_defs`) is the sharp case — arming it on the wrong
/// device silently restores a `tmin-overshoot` bug on every leaf primary.
///
/// Recording it inside `pick()` is what makes it unforgettable: every path that
/// obtains a device (GpuContext, HeadlessGpu, every `--check*` suite) goes
/// through that one function, so a new device path cannot silently inherit a
/// stale vendor.
static PICKED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

/// The session adapter's vendor, or `Other` before any pick (the conservative
/// answer — cross-vendor defaults). Session policy only; see the caveat above.
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

#[derive(Clone)]
pub struct AdapterPick {
    pub adapter: IDXGIAdapter4,
    pub name: String,
    pub vendor: Vendor,
    /// Dedicated video memory, the tie-break within a vendor.
    pub vram: u64,
    /// Packed adapter LUID — the stable identity of this physical adapter,
    /// matching `gputime`'s key and `ID3D12Device::GetAdapterLuid`. What a
    /// secondary search excludes the primary by; comparing `IDXGIAdapter4`
    /// pointers would not, since DXGI may hand back distinct interface
    /// pointers for one adapter.
    // Consumed by the `--dual-gpu` secondary search (stage 2); recorded here
    // because it is the identity the whole feature keys on and enumeration is
    // the only place it is cheaply available.
    #[allow(dead_code)]
    pub luid: u64,
}

fn desc_name(desc: &DXGI_ADAPTER_DESC3) -> String {
    let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(128);
    String::from_utf16_lossy(&desc.Description[..len])
}

/// Every HARDWARE adapter on the box, in DXGI high-performance order.
///
/// Deliberately does NOT record `PICKED`: **enumeration is not selection.** The
/// dual-GPU secondary search needs to look at every adapter, and if looking
/// moved the session vendor, it would silently retune (or mis-refuse) kernels
/// already built for the primary — the failure `vendor_of_device` exists to
/// prevent, reintroduced one level up. `pick` is the only writer of `PICKED`,
/// and it is the only function that chooses.
pub fn enumerate(factory: &IDXGIFactory6) -> Vec<AdapterPick> {
    let mut out = Vec::new();
    for i in 0.. {
        let adapter: IDXGIAdapter4 = match unsafe {
            factory.EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
        } {
            Ok(a) => a,
            Err(_) => break,
        };
        let Ok(desc) = (unsafe { adapter.GetDesc3() }) else { continue };
        if (desc.Flags & DXGI_ADAPTER_FLAG3_SOFTWARE) != DXGI_ADAPTER_FLAG3_NONE {
            continue;
        }
        out.push(AdapterPick {
            adapter,
            name: desc_name(&desc),
            vendor: Vendor::of(desc.VendorId),
            vram: desc.DedicatedVideoMemory as u64,
            luid: ((desc.AdapterLuid.HighPart as u32 as u64) << 32)
                | desc.AdapterLuid.LowPart as u64,
        });
    }
    out
}

/// Pick the preferred vendor's adapter with the most VRAM; fall back to the
/// first hardware adapter (high-performance order).
pub fn pick(factory: &IDXGIFactory6, prefer: Prefer) -> std::result::Result<AdapterPick, String> {
    let want = match prefer {
        Prefer::Nvidia => VENDOR_NVIDIA,
        Prefer::Amd => VENDOR_AMD,
        Prefer::Intel => VENDOR_INTEL,
    };
    let all = enumerate(factory);
    let want = Vendor::of(want);
    // `reduce` with a strict `>`, NOT `max_by_key`: on equal VRAM the original
    // loop kept the FIRST adapter (it only replaced on strictly greater) while
    // `max_by_key` keeps the last. That differs exactly when two adapters have
    // identical VRAM — two identical cards, which is the configuration this
    // whole feature targets — so the tie-break is preserved deliberately.
    let best = all
        .iter()
        .filter(|a| a.vendor == want)
        .reduce(|a, b| if b.vram > a.vram { b } else { a });
    let picked = match best.or_else(|| all.first()) {
        Some(p) => p.clone(),
        None => return Err("no hardware DXGI adapter found".into()),
    };
    record_picked(picked.vendor);
    Ok(picked)
}

pub fn create_factory(debug: bool) -> Result<IDXGIFactory6> {
    let flags = if debug { DXGI_CREATE_FACTORY_DEBUG } else { DXGI_CREATE_FACTORY_FLAGS(0) };
    unsafe { CreateDXGIFactory2(flags) }
}


/// The adapter a device was created on, found by its LUID. The one route from
/// an `ID3D12Device` back to DXGI, shared by `vram_info` and `vendor_of_device`.
fn adapter_of_device<T: Interface>(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
) -> Option<T> {
    let factory: IDXGIFactory4 = create_factory(false).ok()?.cast().ok()?;
    let luid = unsafe { device.GetAdapterLuid() };
    unsafe { factory.EnumAdapterByLuid(luid) }.ok()
}

/// The vendor of the adapter THIS device was created on.
///
/// The per-device answer, and the one every kernel-assembly constant and
/// driver-defect refusal must use — `picked_vendor()` is a process-global that
/// says nothing about which device you are holding, and under `--dual-gpu` two
/// devices of different vendors are live at once. Deriving it from the device
/// makes passing the wrong vendor unrepresentable: there is no argument to get
/// backwards, the device IS the authority (the `vram_info` discipline).
///
/// Best-effort like `vram_info`: `Other` on any failure, which is the
/// conservative answer — every vendor-keyed arm is an opt-IN workaround or
/// tuning, so an unknown vendor takes the cross-vendor path.
pub fn vendor_of_device(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
) -> Vendor {
    let Some(adapter) = adapter_of_device::<IDXGIAdapter4>(device) else {
        return Vendor::Other;
    };
    match unsafe { adapter.GetDesc3() } {
        Ok(d) => Vendor::of(d.VendorId),
        Err(_) => Vendor::Other,
    }
}

/// (current usage, budget) of the device's adapter's LOCAL memory segment —
/// the scene-upload diagnostic: WDDM demotes over-budget commits silently
/// (10-100× slowdown, no error), so init prints where it landed. Best-effort:
/// None on any failure.
pub fn vram_info(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
) -> Option<(u64, u64)> {
    let adapter: IDXGIAdapter3 = adapter_of_device(device)?;
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }
        .ok()?;
    Some((info.CurrentUsage, info.Budget))
}
