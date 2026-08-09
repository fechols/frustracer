//! Safe wrapper over the FidelityFX shim (shim/ffx_shim.h). Owns the ffx-api
//! lifecycle for the FSR backend: loader DLL load, provider version
//! enumeration (the support probe — RDNA4-gated for Ray Regeneration),
//! denoiser + upscaler context creation against the NATIVE D3D12 device (ffx
//! uses no interposer/proxies, unlike Streamline), and the per-frame
//! denoise/upscale dispatches recorded into our command list.

use super::ffx_sys as sys;
use std::ffi::c_void;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandQueue, ID3D12Device};
use windows::Win32::Graphics::Dxgi::{IDXGISwapChain3, IDXGISwapChain4};

pub use sys::{FfxShimDenoiseDesc, FfxShimFgConfig, FfxShimFgPrepare, FfxShimUpscaleDesc};

// ffx_api.h effect ids, used only for debug-message routing.
const EFFECT_ID_UPSCALE: u64 = 0x0001_0000;
const EFFECT_ID_FRAMEGEN: u64 = 0x0002_0000;
const EFFECT_ID_DENOISER: u64 = 0x0005_0000;
const DEBUG_LEVEL_WARNINGS: u32 = 0x2;

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// FFX runtime messages -> stderr. type: 0 error, 1 warning (FfxApiMsgType).
extern "C" fn log_cb(ty: u32, msg: *const u16) {
    if msg.is_null() {
        return;
    }
    let mut len = 0usize;
    while unsafe { *msg.add(len) } != 0 {
        len += 1;
    }
    let text = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(msg, len) });
    match ty {
        0 => eprintln!("[ffx ERROR] {}", text.trim_end()),
        _ => eprintln!("[ffx WARN] {}", text.trim_end()),
    }
}

pub fn result_str(r: i32) -> String {
    let name = match r {
        sys::FFX_OK => "OK",
        1 => "ERROR",
        sys::FFX_ERR_UNKNOWN_DESCTYPE => "ERROR_UNKNOWN_DESCTYPE",
        sys::FFX_ERR_RUNTIME_ERROR => "ERROR_RUNTIME_ERROR",
        sys::FFX_ERR_NO_PROVIDER => "NO_PROVIDER",
        sys::FFX_ERR_MEMORY => "ERROR_MEMORY",
        sys::FFX_ERR_PARAMETER => "ERROR_PARAMETER",
        7 => "PROVIDER_NO_SUPPORT_NEW_DESCTYPE",
        sys::FFXSHIM_ERR_LOAD_LIBRARY => "shim: LoadLibrary failed",
        sys::FFXSHIM_ERR_GET_PROC => "shim: ffx* export missing",
        sys::FFXSHIM_ERR_NOT_LOADED => "shim: not loaded",
        sys::FFXSHIM_ERR_BAD_ARG => "shim: bad argument",
        _ => return format!("ffx result {r}"),
    };
    format!("{name} ({r})")
}

/// One provider version from the enumeration: (override id, display name).
pub type Version = (u64, String);

pub struct FfxContext {
    denoiser: *mut c_void,
    upscaler: *mut c_void,
}

impl FfxContext {
    /// LoadLibrary the ffx loader DLL from `dir` and route runtime messages
    /// for both effects to stderr. Creates no contexts yet.
    pub fn load(dir: &str) -> Result<Self, String> {
        let loader = format!("{dir}\\amd_fidelityfx_loader_dx12.dll");
        if !std::path::Path::new(&loader).exists() {
            return Err(format!("ffx loader not found at {loader}"));
        }
        let loader_w = wide(&loader);
        let r = unsafe { sys::ffxshim_load(loader_w.as_ptr()) };
        if r != sys::FFX_OK {
            return Err(format!("ffx load failed: {}", result_str(r)));
        }
        for id in [EFFECT_ID_DENOISER, EFFECT_ID_UPSCALE] {
            // Best effort; an old loader rejecting the debug desc is fine.
            unsafe { sys::ffxshim_set_debug(id, Some(log_cb), DEBUG_LEVEL_WARNINGS) };
        }
        Ok(Self { denoiser: std::ptr::null_mut(), upscaler: std::ptr::null_mut() })
    }

    /// Enumerate provider versions for one effect on `device`. An empty list
    /// means the effect is unsupported there (Ray Regeneration: no RDNA4).
    pub fn versions(&self, upscaler: bool, device: &ID3D12Device) -> Result<Vec<Version>, String> {
        self.versions_raw(upscaler, device.as_raw())
    }

    /// The same enumeration with a NULL device: the loader then lists every
    /// provider it could load from disk, with no device-support filter — the
    /// failure-path diagnostic that separates "provider DLLs missing or
    /// unloadable" from "this adapter rejected by every provider".
    pub fn versions_any(&self, upscaler: bool) -> Result<Vec<Version>, String> {
        self.versions_raw(upscaler, std::ptr::null_mut())
    }

    fn versions_raw(&self, upscaler: bool, dev: *mut std::ffi::c_void) -> Result<Vec<Version>, String> {
        let mut count: u64 = 0;
        let r = unsafe {
            sys::ffxshim_query_versions(upscaler as i32, dev, &mut count, std::ptr::null_mut(), std::ptr::null_mut())
        };
        if r != sys::FFX_OK {
            return Err(format!("ffx version query failed: {}", result_str(r)));
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut ids = vec![0u64; count as usize];
        let mut names = vec![std::ptr::null::<i8>(); count as usize];
        let mut inout = count;
        let r = unsafe {
            sys::ffxshim_query_versions(upscaler as i32, dev, &mut inout, ids.as_mut_ptr(), names.as_mut_ptr())
        };
        if r != sys::FFX_OK {
            return Err(format!("ffx version query failed: {}", result_str(r)));
        }
        Ok(ids
            .into_iter()
            .zip(names)
            .take(inout as usize)
            .map(|(id, name)| {
                let name = if name.is_null() {
                    String::new()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().into_owned()
                };
                (id, name)
            })
            .collect())
    }

    /// Create the Ray Regeneration context for the session's signal set
    /// (`sys::SIGNALS` — direct diffuse/specular, ambient occlusion, indirect
    /// specular; every stochastic term the 1-spp shade produces). `max` is the
    /// dynamic-resolution range max (window size); every dispatch names its
    /// own renderSize <= max and re-states the SAME signal flags.
    pub fn create_denoiser(&mut self, device: &ID3D12Device, max: (u32, u32)) -> Result<(), String> {
        debug_assert!(self.denoiser.is_null());
        let mut ctx = std::ptr::null_mut();
        let r = unsafe {
            sys::ffxshim_create_denoiser(device.as_raw(), max.0, max.1, sys::SIGNALS, 0, 0, &mut ctx)
        };
        if r != sys::FFX_OK {
            return Err(format!("ffx denoiser create failed: {}", result_str(r)));
        }
        self.denoiser = ctx;
        Ok(())
    }

    /// Apply the `--fsr-*` denoiser tuning overrides (ffxConfigureDescDenoiser-
    /// KeyValue). Each is a single f32 the provider's default stands in for
    /// when the flag is absent, so a flagless session configures nothing and
    /// behaves exactly as before these levers existed. A rejected key is a
    /// loud line, never a session failure — these are A/B knobs.
    pub fn tune_denoiser(&self, t: &crate::fsr::DenoiseTuning) {
        debug_assert!(!self.denoiser.is_null());
        for (key, name, v) in t.entries() {
            let r = unsafe {
                sys::ffxshim_denoiser_kv(self.denoiser, key, 1, &v as *const f32 as *const c_void)
            };
            if r != sys::FFX_OK {
                eprintln!("fsr: denoiser key {name} = {v} rejected ({})", result_str(r));
            } else {
                eprintln!("fsr: denoiser {name} = {v}");
            }
        }
    }

    /// Create the FSR upscaler context: HDR linear input, dynamic input
    /// resolution, fixed presentation size `out`. DEPTH_INVERTED because the
    /// depth plane reuses `xess::view_z_to_clip_depth` (reversed-Z: near→1,
    /// far→0 — the same single-source encoder --check-xess gates).
    /// `version_id` selects the provider (`fsr::pick_version`): 0 = the
    /// loader default with the compile-time API pin (FSR4, today's path
    /// bit-for-bit), nonzero = ffxOverrideVersion to that enumerated id
    /// (the FSR 3.1 flavor).
    pub fn create_upscaler(
        &mut self,
        device: &ID3D12Device,
        max_render: (u32, u32),
        out: (u32, u32),
        debug_checking: bool,
        version_id: u64,
    ) -> Result<(), String> {
        debug_assert!(self.upscaler.is_null());
        let mut flags = sys::UPSCALE_HDR | sys::UPSCALE_DYNAMIC_RESOLUTION | sys::UPSCALE_DEPTH_INVERTED;
        if debug_checking {
            flags |= sys::UPSCALE_DEBUG_CHECKING;
        }
        let mut ctx = std::ptr::null_mut();
        let r = unsafe {
            sys::ffxshim_create_upscaler(
                device.as_raw(),
                max_render.0,
                max_render.1,
                out.0,
                out.1,
                flags,
                version_id,
                &mut ctx,
            )
        };
        if r != sys::FFX_OK {
            return Err(format!("ffx upscaler create failed: {}", result_str(r)));
        }
        self.upscaler = ctx;
        Ok(())
    }

    /// Render resolution the upscaler recommends for a quality mode at
    /// `display` — used only to derive the dynamic range seed/min; the range
    /// max is the window itself.
    pub fn upscaler_render_res(&self, display: (u32, u32), quality: u32) -> Result<(u32, u32), String> {
        if self.upscaler.is_null() {
            return Err("no upscaler context".into());
        }
        let (mut rw, mut rh) = (0u32, 0u32);
        let r = unsafe {
            sys::ffxshim_upscaler_render_res(self.upscaler, display.0, display.1, quality, &mut rw, &mut rh)
        };
        if r != sys::FFX_OK || rw == 0 || rh == 0 {
            return Err(format!("ffx render-res query failed: {}", result_str(r)));
        }
        Ok((rw, rh))
    }

    pub fn denoise(&self, d: &FfxShimDenoiseDesc) -> Result<(), String> {
        if self.denoiser.is_null() {
            return Err("no denoiser context".into());
        }
        let r = unsafe { sys::ffxshim_denoise(self.denoiser, d) };
        if r != sys::FFX_OK {
            return Err(format!("ffx denoise failed: {}", result_str(r)));
        }
        Ok(())
    }

    pub fn upscale(&self, d: &FfxShimUpscaleDesc) -> Result<(), String> {
        if self.upscaler.is_null() {
            return Err("no upscaler context".into());
        }
        let r = unsafe { sys::ffxshim_upscale(self.upscaler, d) };
        if r != sys::FFX_OK {
            return Err(format!("ffx upscale failed: {}", result_str(r)));
        }
        Ok(())
    }
}

impl Drop for FfxContext {
    /// Contexts must be destroyed with the GPU idle and the device alive —
    /// GpuContext::drop drains the queue and drops the FSR state before the
    /// device, same discipline as the XeSS context. The DLL itself stays
    /// loaded until ffxshim_unload; we keep it for process lifetime (matching
    /// the XeSS loader) to avoid unload-order hazards with driver threads.
    fn drop(&mut self) {
        for ctx in [&mut self.denoiser, &mut self.upscaler] {
            if !ctx.is_null() {
                let r = unsafe { sys::ffxshim_destroy(ctx) };
                if r != sys::FFX_OK {
                    eprintln!("[ffx] context destroy failed: {}", result_str(r));
                }
            }
        }
    }
}

// ---- frame generation ----
//
// FG has a different lifetime shape from the upscaler/denoiser pair: the
// frame-interpolation SWAPCHAIN must exist before the first GetBuffer (it
// owns the backbuffers the session renders into — see d3d12::SwapWrap), while
// the FG EFFECT context is display-size-bound and rebuilds on resize. And FG
// can run in sessions where FSR is NOT the wired upscaler (an XeSS session on
// a non-Intel adapter takes ffx-FG), so loading is a free function, not an
// FfxContext method — the shim's loader state is process-global and
// idempotent either way.

/// Ensure the ffx loader is live and the framegeneration provider directory
/// is preloaded. `loader_dir` is the session's --ffx-path; `fg_dir` is where
/// amd_fidelityfx_framegeneration_dx12.dll lives (the prebuilt drop ships it
/// in the FSR sample directory, NOT next to the loader — --fg-path). Safe to
/// call when FSR already loaded the shim; safe to call twice.
pub fn fg_load(loader_dir: &str, fg_dir: &str) -> Result<(), String> {
    let loader = format!("{loader_dir}\\amd_fidelityfx_loader_dx12.dll");
    if !std::path::Path::new(&loader).exists() {
        return Err(format!("ffx loader not found at {loader}"));
    }
    let loader_w = wide(&loader);
    let r = unsafe { sys::ffxshim_load(loader_w.as_ptr()) };
    if r != sys::FFX_OK {
        return Err(format!("ffx load failed: {}", result_str(r)));
    }
    let fg_dir_w = wide(fg_dir);
    let r = unsafe { sys::ffxshim_preload_dir(fg_dir_w.as_ptr()) };
    if r != sys::FFX_OK {
        return Err(format!("fg provider preload failed: {}", result_str(r)));
    }
    unsafe { sys::ffxshim_set_debug(EFFECT_ID_FRAMEGEN, Some(log_cb), DEBUG_LEVEL_WARNINGS) };
    Ok(())
}

/// Enumerate framegeneration provider versions on `device`. Empty = FG
/// unsupported here (provider DLL missing from the preload dirs, or every
/// provider rejected this adapter).
pub fn fg_versions(device: &ID3D12Device) -> Result<Vec<Version>, String> {
    let mut count: u64 = 0;
    let r = unsafe {
        sys::ffxshim_query_versions_fg(device.as_raw(), &mut count, std::ptr::null_mut(), std::ptr::null_mut())
    };
    if r != sys::FFX_OK {
        return Err(format!("fg version query failed: {}", result_str(r)));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut ids = vec![0u64; count as usize];
    let mut names = vec![std::ptr::null::<i8>(); count as usize];
    let mut inout = count;
    let r = unsafe {
        sys::ffxshim_query_versions_fg(device.as_raw(), &mut inout, ids.as_mut_ptr(), names.as_mut_ptr())
    };
    if r != sys::FFX_OK {
        return Err(format!("fg version query failed: {}", result_str(r)));
    }
    Ok(ids
        .into_iter()
        .zip(names)
        .take(inout as usize)
        .map(|(id, name)| {
            let name = if name.is_null() {
                String::new()
            } else {
                unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().into_owned()
            };
            (id, name)
        })
        .collect())
}

/// The frame-interpolation swapchain context. Owns the FI proxy: every ref
/// the session holds on the proxy (D3d::swapchain, backbuffers) must be
/// released BEFORE this drops — GpuContext's field order guarantees it (d3d
/// declared before the FG state).
pub struct FgSwapchain {
    ctx: *mut c_void,
}

/// Wrap `sc` with the frame-interpolation swapchain on `queue`. Ok returns
/// (proxy, context); Err hands the ORIGINAL swapchain back untouched so the
/// session degrades to a normal present path.
pub fn fg_wrap_swapchain(
    queue: &ID3D12CommandQueue,
    sc: IDXGISwapChain3,
) -> Result<(IDXGISwapChain3, FgSwapchain), (IDXGISwapChain3, String)> {
    // The wrap desc speaks IDXGISwapChain4; flip-model chains on any OS new
    // enough to run this renderer implement it.
    let sc4: IDXGISwapChain4 = match sc.cast() {
        Ok(s) => s,
        Err(e) => return Err((sc, format!("IDXGISwapChain4 cast: {e}"))),
    };
    drop(sc);
    let mut raw = sc4.into_raw();
    let mut ctx = std::ptr::null_mut();
    let r = unsafe { sys::ffxshim_fg_swapchain_wrap(queue.as_raw(), &mut raw, &mut ctx) };
    if r != sys::FFX_OK {
        // Create failed: the desc's in/out pointer is untouched — reclaim the
        // original reference we handed over.
        let orig = unsafe { IDXGISwapChain3::from_raw(raw) };
        return Err((orig, format!("wrap create: {}", result_str(r))));
    }
    // The out pointer is the FI proxy carrying one reference for us; every
    // IDXGISwapChain4 is prefix-compatible with 3.
    let proxy = unsafe { IDXGISwapChain3::from_raw(raw) };
    Ok((proxy, FgSwapchain { ctx }))
}

impl FgSwapchain {
    /// Block until every pending (paced/generated) present retired.
    pub fn wait_for_presents(&self) {
        let r = unsafe { sys::ffxshim_fg_swapchain_wait(self.ctx) };
        if r != sys::FFX_OK {
            eprintln!("[ffx] fg wait-for-presents failed: {}", result_str(r));
        }
    }

    /// Register (Some) or unregister (None) the premultiplied display-space
    /// UI texture the FI proxy composites onto BOTH real and generated
    /// frames — the post-interpolation HUD, which is what stops the proxy
    /// warping baked HUD pixels by scene motion on every generated frame.
    /// The resource rests in PIXEL_SHADER_RESOURCE (the HudFi contract);
    /// INTERNAL_UI_DOUBLE_BUFFER makes the proxy snapshot it at Present so
    /// the pacing thread never reads our mid-write texture.
    pub fn register_ui(&self, resource: Option<*mut c_void>) -> Result<(), String> {
        let (ui, flags) = match resource {
            Some(p) => (
                sys::FfxShimRes { resource: p, state: sys::RES_STATE_PIXEL_READ },
                sys::UI_COMP_PREMUL_ALPHA | sys::UI_COMP_INTERNAL_DOUBLE_BUFFER,
            ),
            // "May be empty" per the header — the unregister spelling.
            None => (sys::FfxShimRes::NULL, 0),
        };
        let r = unsafe { sys::ffxshim_fg_swapchain_ui(self.ctx, &ui, flags) };
        if r != sys::FFX_OK {
            return Err(format!("fg ui register failed: {}", result_str(r)));
        }
        Ok(())
    }
}

impl Drop for FgSwapchain {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            self.wait_for_presents();
            let r = unsafe { sys::ffxshim_destroy(&mut self.ctx) };
            if r != sys::FFX_OK {
                eprintln!("[ffx] fg swapchain destroy failed: {}", result_str(r));
            }
        }
    }
}

/// The FG effect context — display-size-bound (recreated on resize by the
/// owner; the swapchain context above survives a ResizeBuffers).
pub struct FgContext {
    ctx: *mut c_void,
}

impl FgContext {
    /// `backbuffer_format`: sys::SURFACE_FORMAT_* ordinal of the swapchain.
    /// `hdr` arms FG_HDR — Hdr10/PQ only (Sdr10 is a gamma SDR backbuffer; the
    /// FI swapchain reads the transfer function off the chain's own declared
    /// colour space). `version_id` 0 = provider default + API pin; nonzero =
    /// the enumerated override (`fsr::pick_fg_version`).
    pub fn create(
        device: &ID3D12Device,
        display: (u32, u32),
        max_render: (u32, u32),
        backbuffer_format: u32,
        hdr: bool,
        debug_checking: bool,
        version_id: u64,
    ) -> Result<Self, String> {
        let mut flags = sys::FG_DEPTH_INVERTED;
        if hdr {
            flags |= sys::FG_HDR;
        }
        if debug_checking {
            flags |= sys::FG_DEBUG_CHECKING;
        }
        let mut ctx = std::ptr::null_mut();
        let r = unsafe {
            sys::ffxshim_create_fg(
                device.as_raw(),
                display.0,
                display.1,
                max_render.0,
                max_render.1,
                backbuffer_format,
                flags,
                version_id,
                &mut ctx,
            )
        };
        if r != sys::FFX_OK {
            return Err(format!("fg context create failed: {}", result_str(r)));
        }
        Ok(Self { ctx })
    }

    pub fn configure(&self, c: &FfxShimFgConfig) -> Result<(), String> {
        let r = unsafe { sys::ffxshim_fg_configure(self.ctx, c) };
        if r != sys::FFX_OK {
            return Err(format!("fg configure failed: {}", result_str(r)));
        }
        Ok(())
    }

    pub fn prepare(&self, p: &FfxShimFgPrepare) -> Result<(), String> {
        let r = unsafe { sys::ffxshim_fg_prepare(self.ctx, p) };
        if r != sys::FFX_OK {
            return Err(format!("fg prepare failed: {}", result_str(r)));
        }
        Ok(())
    }
}

impl Drop for FgContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            let r = unsafe { sys::ffxshim_destroy(&mut self.ctx) };
            if r != sys::FFX_OK {
                eprintln!("[ffx] fg context destroy failed: {}", result_str(r));
            }
        }
    }
}
