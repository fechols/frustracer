//! Raw FFI declarations mirroring shim/sl_shim.h exactly. Everything typed
//! here is a plain-C struct owned by us; all real Streamline structs are
//! constructed inside the C++ shim from the SDK headers.

#![allow(dead_code)]

use std::ffi::c_void;

pub type SlShimLogCb = extern "C" fn(ty: u32, msg: *const i8);

#[repr(C)]
pub struct SlShimInitDesc {
    pub interposer_path: *const u16,
    pub plugins_path: *const u16,
    pub app_id: u32,
    pub show_console: i32,
    pub log_level: u32,
    pub log_cb: Option<SlShimLogCb>,
    pub features: *const u32,
    pub num_features: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlShimDlssdOptions {
    pub mode: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub preset: u32,
    pub normal_roughness_packed: i32,
    pub use_camera_matrices: i32,
    pub world_to_view: [f32; 16],
    pub view_to_world: [f32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct SlShimDlssdOptimal {
    pub render_width: u32,
    pub render_height: u32,
    pub sharpness: f32,
    pub render_width_min: u32,
    pub render_height_min: u32,
    pub render_width_max: u32,
    pub render_height_max: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlShimConstants {
    pub view_to_clip: [f32; 16],
    pub clip_to_view: [f32; 16],
    pub clip_to_prev_clip: [f32; 16],
    pub prev_clip_to_clip: [f32; 16],
    pub jitter: [f32; 2],
    pub mvec_scale: [f32; 2],
    pub cam_pos: [f32; 3],
    pub cam_up: [f32; 3],
    pub cam_right: [f32; 3],
    pub cam_fwd: [f32; 3],
    pub cam_near: f32,
    pub cam_far: f32,
    pub cam_fov: f32,
    pub cam_aspect: f32,
    pub depth_inverted: i32,
    pub camera_motion_included: i32,
    pub mvec_3d: i32,
    pub reset: i32,
    pub ortho: i32,
    pub mvec_jittered: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlShimResourceTag {
    pub resource: *mut c_void,
    pub state: u32,
    pub buffer_type: u32,
    pub lifecycle: u32,
    /// Active sub-rect (sl::Extent) for dynamic resolution. All-zero means
    /// "whole resource" — SL's own null-extent convention, so pre-DRS call
    /// sites need no flag.
    pub extent_top: u32,
    pub extent_left: u32,
    pub extent_width: u32,
    pub extent_height: u32,
}

// sl::Feature ids (sl_core_types.h)
pub const FEATURE_DLSS: u32 = 0;
pub const FEATURE_DLSS_G: u32 = 1000;
pub const FEATURE_DLSS_RR: u32 = 1001;

// sl::DLSSMode (sl_dlss.h)
pub const DLSS_MODE_OFF: u32 = 0;
pub const DLSS_MODE_MAX_PERFORMANCE: u32 = 1;
pub const DLSS_MODE_BALANCED: u32 = 2;
pub const DLSS_MODE_MAX_QUALITY: u32 = 3;
pub const DLSS_MODE_ULTRA_PERFORMANCE: u32 = 4;
pub const DLSS_MODE_ULTRA_QUALITY: u32 = 5;
pub const DLSS_MODE_DLAA: u32 = 6;

// sl::DLSSDPreset (sl_dlss_d.h): E = latest transformer model.
pub const DLSSD_PRESET_DEFAULT: u32 = 0;
pub const DLSSD_PRESET_D: u32 = 4;
pub const DLSSD_PRESET_E: u32 = 5;

// sl::BufferType (sl_core_types.h)
pub const BUFFER_DEPTH: u32 = 0;
pub const BUFFER_MOTION_VECTORS: u32 = 1;
pub const BUFFER_SCALING_INPUT_COLOR: u32 = 3;
pub const BUFFER_SCALING_OUTPUT_COLOR: u32 = 4;
pub const BUFFER_NORMALS: u32 = 5;
pub const BUFFER_ROUGHNESS: u32 = 6;
pub const BUFFER_ALBEDO: u32 = 7;
pub const BUFFER_SPECULAR_ALBEDO: u32 = 8;
pub const BUFFER_NORMAL_ROUGHNESS: u32 = 14;
pub const BUFFER_SPECULAR_HIT_DISTANCE: u32 = 42;
pub const BUFFER_LINEAR_DEPTH: u32 = 49;

// sl::ResourceLifecycle
pub const LIFECYCLE_ONLY_VALID_NOW: u32 = 0;
pub const LIFECYCLE_VALID_UNTIL_PRESENT: u32 = 1;
pub const LIFECYCLE_VALID_UNTIL_EVALUATE: u32 = 2;

// sl::LogLevel
pub const LOG_LEVEL_OFF: u32 = 0;
pub const LOG_LEVEL_DEFAULT: u32 = 1;
pub const LOG_LEVEL_VERBOSE: u32 = 2;

// Selected sl::Result codes for diagnostics (sl_result.h).
pub const SL_OK: i32 = 0;

unsafe extern "C" {
    pub fn slshim_load_and_init(desc: *const SlShimInitDesc) -> i32;
    pub fn slshim_shutdown() -> i32;
    pub fn slshim_is_feature_supported(feature: u32, luid: *const c_void, luid_size: u32) -> i32;
    pub fn slshim_set_d3d_device(native_d3d12_device: *mut c_void) -> i32;
    pub fn slshim_upgrade_interface(iface_inout: *mut *mut c_void) -> i32;
    pub fn slshim_get_native_interface(proxy: *mut c_void, native_out: *mut *mut c_void) -> i32;
    pub fn slshim_dlssd_get_optimal_settings(
        o: *const SlShimDlssdOptions,
        out: *mut SlShimDlssdOptimal,
    ) -> i32;
    pub fn slshim_dlssd_set_options(viewport: u32, o: *const SlShimDlssdOptions) -> i32;
    pub fn slshim_dlssd_get_state(viewport: u32, vram_bytes_out: *mut u64) -> i32;
    pub fn slshim_allocate_resources(cmdlist: *mut c_void, feature: u32, viewport: u32) -> i32;
    pub fn slshim_free_resources(feature: u32, viewport: u32) -> i32;
    pub fn slshim_new_frame_token(frame_index: *const u32, token_out: *mut *mut c_void) -> i32;
    pub fn slshim_set_constants(token: *mut c_void, viewport: u32, c: *const SlShimConstants) -> i32;
    pub fn slshim_tag_resources(
        token: *mut c_void,
        viewport: u32,
        tags: *const SlShimResourceTag,
        n: u32,
        cmdlist: *mut c_void,
    ) -> i32;
    pub fn slshim_evaluate(feature: u32, token: *mut c_void, viewport: u32, cmdlist: *mut c_void) -> i32;
}
