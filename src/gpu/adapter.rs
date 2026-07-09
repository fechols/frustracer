//! DXGI adapter selection. This machine has three GPUs (Intel Arc, NVIDIA
//! RTX, AMD iGPU); DLSS requires the NVIDIA adapter specifically, so we
//! enumerate explicitly instead of trusting adapter 0.

use windows::core::Result;
use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Dxgi::*;

const VENDOR_NVIDIA: u32 = 0x10DE;

pub struct AdapterPick {
    pub adapter: IDXGIAdapter4,
    pub luid: LUID,
    pub name: String,
    pub is_nvidia: bool,
}

fn desc_name(desc: &DXGI_ADAPTER_DESC3) -> String {
    let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(128);
    String::from_utf16_lossy(&desc.Description[..len])
}

/// Pick the NVIDIA adapter with the most VRAM; fall back to the first
/// hardware adapter (high-performance order) when `require_nvidia` is off.
pub fn pick(factory: &IDXGIFactory6, require_nvidia: bool) -> std::result::Result<AdapterPick, String> {
    let mut best: Option<(IDXGIAdapter4, DXGI_ADAPTER_DESC3)> = None;
    let mut fallback: Option<(IDXGIAdapter4, DXGI_ADAPTER_DESC3)> = None;
    let mut listing = String::new();
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
        listing.push_str(&format!(
            "  adapter {}: {} (vendor {:#06x}, {} MB)\n",
            i,
            desc_name(&desc),
            desc.VendorId,
            desc.DedicatedVideoMemory / (1024 * 1024)
        ));
        if fallback.is_none() {
            fallback = Some((adapter.clone(), desc));
        }
        if desc.VendorId == VENDOR_NVIDIA
            && best
                .as_ref()
                .is_none_or(|(_, b)| desc.DedicatedVideoMemory > b.DedicatedVideoMemory)
        {
            best = Some((adapter, desc));
        }
    }
    let picked = match (best, fallback, require_nvidia) {
        (Some(p), _, _) => p,
        (None, _, true) => {
            return Err(format!("no NVIDIA adapter found (required for DLSS); adapters:\n{listing}"))
        }
        (None, Some(p), false) => p,
        (None, None, _) => return Err("no hardware DXGI adapter found".into()),
    };
    let name = desc_name(&picked.1);
    let is_nvidia = picked.1.VendorId == VENDOR_NVIDIA;
    Ok(AdapterPick { adapter: picked.0, luid: picked.1.AdapterLuid, name, is_nvidia })
}

pub fn create_factory(debug: bool) -> Result<IDXGIFactory6> {
    let flags = if debug { DXGI_CREATE_FACTORY_DEBUG } else { DXGI_CREATE_FACTORY_FLAGS(0) };
    unsafe { CreateDXGIFactory2(flags) }
}
