//! Raw FFI surface of the FidelityFX shim (shim/ffx_shim.h). These structs
//! mirror the SHIM's flat C structs only — never the real ffx-api structs,
//! which exist solely inside shim/ffx_shim.cpp (the SL-shim doctrine).

#![allow(dead_code)]

use std::ffi::c_void;

pub const FFX_OK: i32 = 0;

// FfxApiReturnCodes (ffx_api.h) — decoded by ffx::result_str.
pub const FFX_ERR_UNKNOWN_DESCTYPE: i32 = 2;
pub const FFX_ERR_RUNTIME_ERROR: i32 = 3;
pub const FFX_ERR_NO_PROVIDER: i32 = 4;
pub const FFX_ERR_MEMORY: i32 = 5;
pub const FFX_ERR_PARAMETER: i32 = 6;

// Shim-private (ffx_shim.h).
pub const FFXSHIM_ERR_LOAD_LIBRARY: i32 = -1000;
pub const FFXSHIM_ERR_GET_PROC: i32 = -1001;
pub const FFXSHIM_ERR_NOT_LOADED: i32 = -1002;
pub const FFXSHIM_ERR_BAD_ARG: i32 = -1003;

// FfxApiDenoiserSignalFlags (re-exported by the shim header). The four the
// shim builds descs for are the four this renderer has a source for; the other
// three (dominant-light visibility, indirect diffuse, specular occlusion) are
// rejected by the shim rather than silently dispatching a set that disagrees
// with the context's creation flags.
pub const SIGNAL_AMBIENT_OCCLUSION: u32 = 1 << 0;
pub const SIGNAL_DIRECT_DIFFUSE: u32 = 1 << 1;
pub const SIGNAL_DIRECT_SPECULAR: u32 = 1 << 2;
pub const SIGNAL_INDIRECT_SPECULAR: u32 = 1 << 5;

/// The signal set an FSR4-RR session subscribes to. ONE constant: the create
/// and every dispatch must name the identical set (the ffx header's
/// if-and-only-if rule), so they read it from here.
pub const SIGNALS: u32 =
    SIGNAL_DIRECT_DIFFUSE | SIGNAL_DIRECT_SPECULAR | SIGNAL_AMBIENT_OCCLUSION | SIGNAL_INDIRECT_SPECULAR;

// FfxApiCreateContextUpscaleFlags subset.
pub const UPSCALE_HDR: u32 = 1 << 0;
pub const UPSCALE_DEPTH_INVERTED: u32 = 1 << 3;
pub const UPSCALE_DYNAMIC_RESOLUTION: u32 = 1 << 6;
pub const UPSCALE_DEBUG_CHECKING: u32 = 1 << 7;

// FfxApiCreateContextFramegenerationFlags subset (ffx_framegeneration.h). The
// absent bits are deliberate: display-res MVs (ours are render-res), jitter
// cancellation (our MV fill sites project unjittered hit points), infinite
// depth (our reversed-Z clip encode has a finite far with sky at exactly 0.0).
pub const FG_ASYNC_SUPPORT: u32 = 1 << 0;
pub const FG_DEPTH_INVERTED: u32 = 1 << 3;
pub const FG_HDR: u32 = 1 << 5;
pub const FG_DEBUG_CHECKING: u32 = 1 << 6;

// FfxApiDispatchFramegenerationFlags subset — debug overlays for the FG
// dispatch/configure path.
pub const FG_DISPATCH_DEBUG_TEAR_LINES: u32 = 1 << 0;
pub const FG_DISPATCH_DEBUG_VIEW: u32 = 1 << 2;
pub const FG_DISPATCH_DEBUG_PACING_LINES: u32 = 1 << 4;

// FfxApiSurfaceFormat ordinals for the two swapchain formats this renderer
// creates (ffx_api_types.h — the enum is ordinal, no explicit values).
pub const SURFACE_FORMAT_R16G16B16A16_FLOAT: u32 = 4;
pub const SURFACE_FORMAT_B8G8R8A8_UNORM: u32 = 14;
pub const SURFACE_FORMAT_R10G10B10A2_UNORM: u32 = 17;

// FfxApiUpscaleQualityMode (ffx_upscale.h) — used for the render-res range
// derivation queries.
pub const QUALITY_MODE_NATIVEAA: u32 = 0;
pub const QUALITY_MODE_QUALITY: u32 = 1;
pub const QUALITY_MODE_ULTRA_PERFORMANCE: u32 = 4;

// FfxApiResourceState subset — must match the D3D12 state the resource is
// actually in when the FFX dispatch executes.
pub const RES_STATE_UNORDERED_ACCESS: u32 = 1 << 1;
pub const RES_STATE_COMPUTE_READ: u32 = 1 << 2;

pub type FfxShimLogCb = Option<extern "C" fn(ty: u32, msg: *const u16)>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfxShimRes {
    pub resource: *mut c_void,
    pub state: u32,
}

impl FfxShimRes {
    pub const NULL: FfxShimRes = FfxShimRes { resource: std::ptr::null_mut(), state: 0 };
}

#[repr(C)]
pub struct FfxShimDenoiseDesc {
    pub cmdlist: *mut c_void,
    /// Must equal the context's creation signalFlags (`SIGNALS`).
    pub signal_flags: u32,
    pub linear_depth: FfxShimRes,
    pub motion_vectors: FfxShimRes,
    pub normals: FfxShimRes,
    pub specular_albedo: FfxShimRes,
    pub diffuse_albedo: FfxShimRes,
    pub dd_in: FfxShimRes,
    pub dd_out: FfxShimRes,
    pub ds_in: FfxShimRes,
    pub ds_out: FfxShimRes,
    pub ao_in: FfxShimRes,
    pub ao_out: FfxShimRes,
    pub is_in: FfxShimRes,
    pub is_out: FfxShimRes,
    pub mv_scale: [f32; 3],
    pub jitter: [f32; 2],
    pub cam_pos_delta: [f32; 3],
    pub view: [f32; 16],
    pub projection: [f32; 16],
    pub depth_bounds_min: f32,
    pub depth_bounds_max: f32,
    pub render_w: u32,
    pub render_h: u32,
    pub frame_index: u32,
    pub reset: i32,
    pub non_gamma_albedo: i32,
}

// The C++ twin (ffx_shim.cpp) asserts the IDENTICAL two literals. `signal_flags`
// is a u32 wedged between 8-aligned members, so it opens a padding hole both
// languages must lay out the same way; if they ever disagree, one of the two
// builds fails here instead of every resource pointer silently shifting by 4
// bytes at the dispatch.
const _: () = assert!(std::mem::offset_of!(FfxShimDenoiseDesc, linear_depth) == 16);
const _: () = assert!(std::mem::size_of::<FfxShimDenoiseDesc>() == 416);

#[repr(C)]
pub struct FfxShimUpscaleDesc {
    pub cmdlist: *mut c_void,
    pub color: FfxShimRes,
    pub depth: FfxShimRes,
    pub motion_vectors: FfxShimRes,
    pub output: FfxShimRes,
    pub jitter: [f32; 2],
    pub mv_scale: [f32; 2],
    pub render_w: u32,
    pub render_h: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub enable_sharpening: i32,
    pub sharpness: f32,
    pub frame_time_delta_ms: f32,
    pub pre_exposure: f32,
    pub reset: i32,
    pub cam_near: f32,
    pub cam_far: f32,
    pub cam_fovy: f32,
    pub view_space_to_meters: f32,
    pub flags: u32,
}

#[repr(C)]
pub struct FfxShimFgConfig {
    pub swapchain: *mut c_void,
    pub enabled: i32,
    pub allow_async: i32,
    pub hudless: FfxShimRes,
    pub flags: u32,
    pub only_present_generated: i32,
    pub rect_left: u32,
    pub rect_top: u32,
    pub rect_w: u32,
    pub rect_h: u32,
    pub min_max_luminance: [f32; 2],
    pub frame_id: u64,
}

#[repr(C)]
pub struct FfxShimFgPrepare {
    pub cmdlist: *mut c_void,
    pub frame_id: u64,
    pub flags: u32,
    pub render_w: u32,
    pub render_h: u32,
    pub jitter: [f32; 2],
    pub mv_scale: [f32; 2],
    pub frame_time_delta_ms: f32,
    pub reset: i32,
    pub cam_near: f32,
    pub cam_far: f32,
    pub cam_fovy: f32,
    pub view_space_to_meters: f32,
    pub depth: FfxShimRes,
    pub motion_vectors: FfxShimRes,
    pub cam_pos: [f32; 3],
    pub cam_up: [f32; 3],
    pub cam_right: [f32; 3],
    pub cam_fwd: [f32; 3],
}

// The FG twins of the denoise-desc layout pins: `hudless` sits after two i32s
// (an 8-aligned member after a 4-hole boundary), and `depth` lands after a
// float run that ends mid-slot — the exact shapes a C/Rust packing divergence
// would silently shift. The C++ TU asserts the identical literals.
const _: () = assert!(std::mem::offset_of!(FfxShimFgConfig, hudless) == 16);
const _: () = assert!(std::mem::size_of::<FfxShimFgConfig>() == 72);
const _: () = assert!(std::mem::offset_of!(FfxShimFgPrepare, depth) == 72);
const _: () = assert!(std::mem::size_of::<FfxShimFgPrepare>() == 152);

unsafe extern "C" {
    pub fn ffxshim_load(loader_dll_path: *const u16) -> i32;
    pub fn ffxshim_unload();
    pub fn ffxshim_set_debug(effect_id: u64, cb: FfxShimLogCb, level: u32) -> i32;
    pub fn ffxshim_query_versions(
        is_upscaler: i32,
        device: *mut c_void,
        inout_count: *mut u64,
        ids: *mut u64,
        names: *mut *const i8,
    ) -> i32;
    pub fn ffxshim_create_denoiser(
        device: *mut c_void,
        max_w: u32,
        max_h: u32,
        signal_flags: u32,
        flags: u32,
        version_id: u64,
        ctx_out: *mut *mut c_void,
    ) -> i32;
    pub fn ffxshim_create_upscaler(
        device: *mut c_void,
        max_render_w: u32,
        max_render_h: u32,
        out_w: u32,
        out_h: u32,
        flags: u32,
        version_id: u64,
        ctx_out: *mut *mut c_void,
    ) -> i32;
    pub fn ffxshim_destroy(ctx: *mut *mut c_void) -> i32;
    pub fn ffxshim_upscaler_render_res(
        upscaler_ctx: *mut c_void,
        display_w: u32,
        display_h: u32,
        quality_mode: u32,
        out_rw: *mut u32,
        out_rh: *mut u32,
    ) -> i32;
    pub fn ffxshim_denoiser_kv(denoiser_ctx: *mut c_void, key: u64, count: u64, data: *const c_void) -> i32;
    pub fn ffxshim_denoise(denoiser_ctx: *mut c_void, d: *const FfxShimDenoiseDesc) -> i32;
    pub fn ffxshim_upscale(upscaler_ctx: *mut c_void, d: *const FfxShimUpscaleDesc) -> i32;
    pub fn ffxshim_preload_dir(dir: *const u16) -> i32;
    pub fn ffxshim_query_versions_fg(
        device: *mut c_void,
        inout_count: *mut u64,
        ids: *mut u64,
        names: *mut *const i8,
    ) -> i32;
    pub fn ffxshim_fg_swapchain_wrap(
        game_queue: *mut c_void,
        inout_swapchain: *mut *mut c_void,
        out_sc_ctx: *mut *mut c_void,
    ) -> i32;
    pub fn ffxshim_fg_swapchain_wait(sc_ctx: *mut c_void) -> i32;
    pub fn ffxshim_fg_swapchain_ui(sc_ctx: *mut c_void, ui: *const FfxShimRes, premul: i32) -> i32;
    pub fn ffxshim_create_fg(
        device: *mut c_void,
        display_w: u32,
        display_h: u32,
        max_render_w: u32,
        max_render_h: u32,
        backbuffer_format: u32,
        flags: u32,
        version_id: u64,
        out_ctx: *mut *mut c_void,
    ) -> i32;
    pub fn ffxshim_fg_configure(fg_ctx: *mut c_void, c: *const FfxShimFgConfig) -> i32;
    pub fn ffxshim_fg_prepare(fg_ctx: *mut c_void, p: *const FfxShimFgPrepare) -> i32;
}
