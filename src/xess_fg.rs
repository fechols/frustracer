//! XeSS Frame Generation (libxess_fg.dll) + XeLL low-latency (libxell.dll) —
//! the Intel FG family of `--fg` (W4 leg 3), for XeSS sessions on Arc.
//!
//! Footprint policy is xess.rs's, verbatim: nothing links the SDKs, both DLLs
//! are `LoadLibraryExW`'d at runtime from the same directory libxess.dll
//! lives in (the xess_path default already points there), the `#[repr(C)]`
//! structs are hand-transcribed from `xefg_swapchain[_d3d12].h` /
//! `xell[_d3d12].h` (pack(8) == natural x64 layout for these field mixes),
//! and headless runs never touch any of it.
//!
//! Architecture: XeSS-FG is a SWAPCHAIN wrapper, like the ffx FI swapchain —
//! `xefgSwapChainD3D12InitFromSwapChain` wraps the app's chain and
//! `GetSwapChainPtr` hands back the proxy every present must go through from
//! then on (the same d3d12::SwapWrap hook point the ffx family uses). Unlike
//! ffx there is no separate display-size effect context: the swapchain
//! context IS the whole feature. XeLL is a HARD requirement — the FG context
//! must be linked to a XeLL context (`SetLatencyReduction`) and all SIX
//! latency markers must fire per generated frame, or interpolation misreports
//! and can decline.

#![allow(dead_code)]

use std::ffi::c_void;

// ---------------------------------------------------------------------------
// Convention constants — the xess.rs discipline: every undocumented polarity
// in one place. XeFG shares the SR plane set (same mvec/depth textures), so
// the conventions deliberately MIRROR xess.rs's and must move in lockstep.
// ---------------------------------------------------------------------------

/// Jitter reported to XeFG = sign * the renderer's sample offset — the same
/// unnegated convention XeSS-SR settled on (`xess::JITTER_SIGN`).
pub const JITTER_SIGN: f32 = 1.0;

/// Frame-constant motionVectorScale for our MV plane (pixels, y-down,
/// current -> previous — the exact plane XeSS-SR consumes at
/// `xess::VELOCITY_SCALE` (1,1)).
pub const MV_SCALE: (f32, f32) = (1.0, 1.0);

// xefg_swapchain_init_flags_t.
pub const INIT_FLAG_INVERTED_DEPTH: u32 = 1 << 0;
pub const INIT_FLAG_EXTERNAL_DESCRIPTOR_HEAP: u32 = 1 << 1;
pub const INIT_FLAG_USE_NDC_VELOCITY: u32 = 1 << 3;
pub const INIT_FLAG_JITTERED_MV: u32 = 1 << 4;

/// Our init flags: reversed-Z clip depth (the shared `view_z_to_clip_depth`
/// encode, near = 1, sky = exactly 0), pixel-space UNjittered MVs (analytic
/// hit-point reprojection, the SL mvec_jittered = 0 contract), internal
/// descriptor heap (no EXTERNAL flag — the library manages its own), UI mode
/// AUTO with nothing tagged = interpolate the whole backbuffer (HUD baked in,
/// the leg-1 known-accept).
pub const INIT_FLAGS: u32 = INIT_FLAG_INVERTED_DEPTH;

// xefg_swapchain_resource_type_t.
pub const RES_HUDLESS_COLOR: u32 = 0;
pub const RES_DEPTH: u32 = 1;
pub const RES_MOTION_VECTOR: u32 = 2;
pub const RES_UI: u32 = 3;
pub const RES_BACKBUFFER: u32 = 4;

// xefg_swapchain_resource_validity_t.
pub const RV_UNTIL_NEXT_PRESENT: u32 = 0;

// xefg_swapchain_ui_mode_t.
pub const UI_MODE_AUTO: u32 = 0;

// xell_latency_marker_type_t — all six are REQUIRED per frame.
pub const XELL_SIMULATION_START: u32 = 0;
pub const XELL_SIMULATION_END: u32 = 1;
pub const XELL_RENDERSUBMIT_START: u32 = 2;
pub const XELL_RENDERSUBMIT_END: u32 = 3;
pub const XELL_PRESENT_START: u32 = 4;
pub const XELL_PRESENT_END: u32 = 5;

pub const XEFG_SUCCESS: i32 = 0;

/// The subset of xefg_swapchain_result_t worth naming in diagnostics.
pub fn result_name(r: i32) -> &'static str {
    match r {
        0 => "SUCCESS",
        2 => "WARNING_OLD_DRIVER",
        3 => "WARNING_TOO_FEW_FRAMES",
        4 => "WARNING_FRAMES_ID_MISMATCH",
        5 => "WARNING_MISSING_PRESENT_STATUS",
        6 => "WARNING_RESOURCE_SIZES_MISMATCH",
        -1 => "ERROR_UNSUPPORTED_DEVICE",
        -2 => "ERROR_UNSUPPORTED_DRIVER",
        -3 => "ERROR_UNINITIALIZED",
        -4 => "ERROR_INVALID_ARGUMENT",
        -5 => "ERROR_DEVICE_OUT_OF_MEMORY",
        -6 => "ERROR_DEVICE",
        -10 => "ERROR_UNSUPPORTED",
        -11 => "ERROR_CANT_LOAD_LIBRARY",
        -12 => "ERROR_MISMATCH_INPUT_RESOURCES",
        -13 => "ERROR_INCORRECT_OUTPUT_RESOURCES",
        -14 => "ERROR_INCORRECT_INPUT_RESOURCES",
        -15 => "ERROR_LATENCY_REDUCTION_UNSUPPORTED",
        -16 => "ERROR_LATENCY_REDUCTION_FUNCTION_MISSING",
        -17 => "ERROR_HRESULT_FAILURE",
        -18 => "ERROR_DXGI_INVALID_CALL",
        -19 => "ERROR_POINTER_STILL_IN_USE",
        -20 => "ERROR_INVALID_DESCRIPTOR_HEAP",
        -21 => "ERROR_WRONG_CALL_ORDER",
        -1000 => "ERROR_UNKNOWN",
        _ => "result?",
    }
}

// ---------------------------------------------------------------------------
// FFI mirrors (pack(8) == natural x64 for these mixes).
// ---------------------------------------------------------------------------

type Handle = *mut c_void;

#[repr(C)]
#[derive(Default)]
pub struct Version {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub reserved: u16,
}

#[repr(C)]
struct InitParams {
    p_application_swap_chain: *mut c_void,
    init_flags: u32,
    max_interpolated_frames: u32, // must be 1
    creation_node_mask: u32,
    visible_node_mask: u32,
    p_temp_buffer_heap: *mut c_void,
    buffer_heap_offset: u64,
    p_temp_texture_heap: *mut c_void,
    texture_heap_offset: u64,
    p_pipeline_library: *mut c_void,
    ui_mode: u32,
}

#[repr(C)]
struct ResourceData {
    ty: u32,
    validity: u32,
    resource_base: [u32; 2],
    resource_size: [u32; 2],
    p_resource: *mut c_void,
    incoming_state: u32, // D3D12_RESOURCE_STATES
}

#[repr(C)]
struct FrameConstants {
    view: [f32; 16], // row-major (glam transposes at this boundary — the SL
    proj: [f32; 16], // row_major() convention, NOT the ffx memcpy)
    jitter_x: f32,
    jitter_y: f32,
    mv_scale_x: f32,
    mv_scale_y: f32,
    reset_history: u32,
    frame_render_time: f32,
}

#[repr(C)]
#[derive(Default)]
struct Properties {
    required_descriptor_count: u32,
    temp_buffer_heap_size: u64,
    temp_texture_heap_size: u64,
    constant_buffer_size: u64,
    max_supported_interpolations: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct PresentStatus {
    pub frames_presented: u32,
    pub frame_gen_result: i32,
    pub is_frame_gen_enabled: u32,
}

#[repr(C)]
struct XellSleepParams {
    minimum_interval_us: u32,
    /// bit 0 = bLowLatencyMode, bit 1 = bLowLatencyBoost (no-op today).
    flags: u32,
}

#[cfg(windows)]
mod loader {
    use super::*;
    use windows::core::{Interface, PCSTR, PCWSTR};
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::Graphics::Dxgi::IDXGISwapChain3;
    use windows::Win32::System::LibraryLoader::{
        GetProcAddress, LoadLibraryExW, LOAD_WITH_ALTERED_SEARCH_PATH,
    };

    struct FgApi {
        get_version: unsafe extern "C" fn(*mut Version) -> i32,
        create_context: unsafe extern "C" fn(*mut c_void, *mut Handle) -> i32,
        init_from_swapchain: unsafe extern "C" fn(Handle, *mut c_void, *const InitParams) -> i32,
        get_swapchain_ptr:
            unsafe extern "C" fn(Handle, *const windows::core::GUID, *mut *mut c_void) -> i32,
        tag_frame_resource:
            unsafe extern "C" fn(Handle, *mut c_void, u32, *const ResourceData) -> i32,
        tag_frame_constants: unsafe extern "C" fn(Handle, u32, *const FrameConstants) -> i32,
        set_present_id: unsafe extern "C" fn(Handle, u32) -> i32,
        set_enabled: unsafe extern "C" fn(Handle, u32) -> i32,
        set_latency_reduction: unsafe extern "C" fn(Handle, *mut c_void) -> i32,
        get_last_present_status: unsafe extern "C" fn(Handle, *mut PresentStatus) -> i32,
        get_properties: unsafe extern "C" fn(Handle, *mut Properties) -> i32,
        destroy: unsafe extern "C" fn(Handle) -> i32,
    }

    struct XellApi {
        get_version: unsafe extern "C" fn(*mut Version) -> i32,
        create_context: unsafe extern "C" fn(*mut c_void, *mut Handle) -> i32,
        destroy_context: unsafe extern "C" fn(Handle) -> i32,
        set_sleep_mode: unsafe extern "C" fn(Handle, *const XellSleepParams) -> i32,
        sleep: unsafe extern "C" fn(Handle, u32) -> i32,
        add_marker: unsafe extern "C" fn(Handle, u32, u32) -> i32,
    }

    fn load_dll(dir: &str, name: &str) -> Result<HMODULE, String> {
        let path = format!("{dir}\\{name}");
        let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
        unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, LOAD_WITH_ALTERED_SEARCH_PATH) }
            .map_err(|e| format!("failed to load {path}: {e}"))
    }

    macro_rules! resolve {
        ($h:expr, $dll:literal, $name:literal) => {{
            let sym = unsafe { GetProcAddress($h, PCSTR(concat!($name, "\0").as_ptr())) }
                .ok_or_else(|| format!("{}: missing export {}", $dll, $name))?;
            #[allow(clippy::missing_transmute_annotations)]
            let f = unsafe { std::mem::transmute(sym) };
            f
        }};
    }

    /// A live XeSS-FG swapchain context + its linked XeLL context. Owns the
    /// FI proxy: every ref the session holds on the proxy must release
    /// BEFORE this drops (the ffx FgSwapchain field-order discipline —
    /// destroy tears the proxy down). HMODULEs are never freed (SL/OIDN
    /// policy).
    pub struct XefgSwapchain {
        api: FgApi,
        xell: XellApi,
        fg: Handle,
        xell_ctx: Handle,
        /// The ORIGINAL app swapchain, kept alive for the context's whole
        /// life: unlike the ffx wrap (which consumes and internally replaces
        /// the input chain), the xefg proxy DELEGATES to the app's chain —
        /// releasing the last app-side ref kills the real swapchain under
        /// the proxy (measured: a silent native crash right after init).
        app: *mut c_void,
    }

    impl XefgSwapchain {
        /// Wrap `app_swapchain` (a raw IDXGISwapChain* whose ONE caller ref
        /// is transferred in) on `queue`, with XeLL created + linked. Ok
        /// returns the context and the FI proxy (carrying one ref for the
        /// caller); the library holds its own ref on the original, so the
        /// caller's transferred ref is released here either way. Err hands
        /// nothing back — the caller still owns `app_swapchain` (it is NOT
        /// released on failure).
        pub fn wrap(
            dll_dir: &str,
            device: *mut c_void,
            queue: *mut c_void,
            app_swapchain: *mut c_void,
        ) -> Result<(Self, *mut c_void), String> {
            let dir = std::fs::canonicalize(dll_dir)
                .map(|p| p.to_string_lossy().trim_start_matches(r"\\?\").to_string())
                .unwrap_or_else(|_| dll_dir.to_string());
            let h_fg = load_dll(&dir, "libxess_fg.dll")?;
            let h_xl = load_dll(&dir, "libxell.dll")?;
            let api = FgApi {
                get_version: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainGetVersion"),
                create_context: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainD3D12CreateContext"),
                init_from_swapchain: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainD3D12InitFromSwapChain"),
                get_swapchain_ptr: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainD3D12GetSwapChainPtr"),
                tag_frame_resource: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainD3D12TagFrameResource"),
                tag_frame_constants: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainTagFrameConstants"),
                set_present_id: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainSetPresentId"),
                set_enabled: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainSetEnabled"),
                set_latency_reduction: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainSetLatencyReduction"),
                get_last_present_status: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainGetLastPresentStatus"),
                get_properties: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainGetProperties"),
                destroy: resolve!(h_fg, "libxess_fg.dll", "xefgSwapChainDestroy"),
            };
            let xell = XellApi {
                get_version: resolve!(h_xl, "libxell.dll", "xellGetVersion"),
                create_context: resolve!(h_xl, "libxell.dll", "xellD3D12CreateContext"),
                destroy_context: resolve!(h_xl, "libxell.dll", "xellDestroyContext"),
                set_sleep_mode: resolve!(h_xl, "libxell.dll", "xellSetSleepMode"),
                sleep: resolve!(h_xl, "libxell.dll", "xellSleep"),
                add_marker: resolve!(h_xl, "libxell.dll", "xellAddMarkerData"),
            };

            let mut ver = Version::default();
            if unsafe { (api.get_version)(&mut ver) } == XEFG_SUCCESS {
                eprintln!("fg: XeSS-FG SDK {}.{}.{}", ver.major, ver.minor, ver.patch);
            }
            let mut xlver = Version::default();
            if unsafe { (xell.get_version)(&mut xlver) } == XEFG_SUCCESS {
                eprintln!("fg: XeLL SDK {}.{}.{}", xlver.major, xlver.minor, xlver.patch);
            }

            // XeLL first: the FG init checks the link's viability.
            let mut xell_ctx: Handle = std::ptr::null_mut();
            let r = unsafe { (xell.create_context)(device, &mut xell_ctx) };
            if r != XEFG_SUCCESS || xell_ctx.is_null() {
                return Err(format!("xellD3D12CreateContext: {} ({r})", result_name(r)));
            }
            let sleep_params =
                XellSleepParams { minimum_interval_us: 0, flags: 1 /* low-latency mode */ };
            let r = unsafe { (xell.set_sleep_mode)(xell_ctx, &sleep_params) };
            if r != XEFG_SUCCESS {
                eprintln!("fg: xellSetSleepMode: {} ({r}) — continuing", result_name(r));
            }

            let mut fg: Handle = std::ptr::null_mut();
            let r = unsafe { (api.create_context)(device, &mut fg) };
            if r < XEFG_SUCCESS || fg.is_null() {
                unsafe { (xell.destroy_context)(xell_ctx) };
                return Err(format!("xefgSwapChainD3D12CreateContext: {} ({r})", result_name(r)));
            }
            if r > XEFG_SUCCESS {
                eprintln!("fg: xefg create: {}", result_name(r));
            }

            let r = unsafe { (api.set_latency_reduction)(fg, xell_ctx) };
            if r != XEFG_SUCCESS {
                eprintln!("fg: SetLatencyReduction: {} ({r}) — continuing", result_name(r));
            }

            let params = InitParams {
                p_application_swap_chain: app_swapchain,
                init_flags: INIT_FLAGS,
                max_interpolated_frames: 1, // the header's hard requirement
                creation_node_mask: 0,
                visible_node_mask: 0,
                p_temp_buffer_heap: std::ptr::null_mut(),
                buffer_heap_offset: 0,
                p_temp_texture_heap: std::ptr::null_mut(),
                texture_heap_offset: 0,
                p_pipeline_library: std::ptr::null_mut(),
                ui_mode: UI_MODE_AUTO,
            };
            let r = unsafe { (api.init_from_swapchain)(fg, queue, &params) };
            if r < XEFG_SUCCESS {
                let e = format!("xefgSwapChainD3D12InitFromSwapChain: {} ({r})", result_name(r));
                unsafe { (api.destroy)(fg) };
                unsafe { (xell.destroy_context)(xell_ctx) };
                return Err(e);
            }
            if r > XEFG_SUCCESS {
                eprintln!("fg: xefg init: {}", result_name(r));
            }

            let mut proxy: *mut c_void = std::ptr::null_mut();
            let riid = IDXGISwapChain3::IID;
            let r = unsafe { (api.get_swapchain_ptr)(fg, &riid, &mut proxy) };
            if r != XEFG_SUCCESS || proxy.is_null() {
                let e = format!("xefgSwapChainD3D12GetSwapChainPtr: {} ({r})", result_name(r));
                unsafe { (api.destroy)(fg) };
                unsafe { (xell.destroy_context)(xell_ctx) };
                return Err(e);
            }

            // The caller's transferred ref on the app swapchain lives in
            // `app` until Drop — see the field's comment.
            Ok((Self { api, xell, fg, xell_ctx, app: app_swapchain }, proxy))
        }

        pub fn set_enabled(&self, on: bool) {
            let r = unsafe { (self.api.set_enabled)(self.fg, on as u32) };
            if r != XEFG_SUCCESS {
                eprintln!("fg: xefg SetEnabled({on}): {}", result_name(r));
            }
        }

        pub fn set_present_id(&self, id: u32) {
            let r = unsafe { (self.api.set_present_id)(self.fg, id) };
            if r != XEFG_SUCCESS {
                eprintln!("fg: xefg SetPresentId: {}", result_name(r));
            }
        }

        /// Tag a render-res input plane for this present id. `state` is the
        /// D3D12 state the plane rests in when the FG work executes.
        pub fn tag_resource(
            &self,
            present_id: u32,
            ty: u32,
            resource: *mut c_void,
            state: u32,
            w: u32,
            h: u32,
        ) -> Result<(), String> {
            let d = ResourceData {
                ty,
                validity: RV_UNTIL_NEXT_PRESENT,
                resource_base: [0, 0],
                resource_size: [w, h],
                p_resource: resource,
                incoming_state: state,
            };
            let r = unsafe {
                (self.api.tag_frame_resource)(self.fg, std::ptr::null_mut(), present_id, &d)
            };
            if r < XEFG_SUCCESS {
                return Err(format!("tag_resource(ty {ty}): {} ({r})", result_name(r)));
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn tag_constants(
            &self,
            present_id: u32,
            view_row_major: [f32; 16],
            proj_row_major: [f32; 16],
            jitter: (f32, f32),
            reset: bool,
            frame_ms: f32,
        ) -> Result<(), String> {
            let c = FrameConstants {
                view: view_row_major,
                proj: proj_row_major,
                jitter_x: JITTER_SIGN * jitter.0,
                jitter_y: JITTER_SIGN * jitter.1,
                mv_scale_x: MV_SCALE.0,
                mv_scale_y: MV_SCALE.1,
                reset_history: reset as u32,
                frame_render_time: frame_ms.clamp(0.1, 200.0),
            };
            let r = unsafe { (self.api.tag_frame_constants)(self.fg, present_id, &c) };
            if r < XEFG_SUCCESS {
                return Err(format!("tag_constants: {} ({r})", result_name(r)));
            }
            Ok(())
        }

        pub fn last_status(&self) -> Result<PresentStatus, String> {
            let mut s = PresentStatus::default();
            let r = unsafe { (self.api.get_last_present_status)(self.fg, &mut s) };
            if r < XEFG_SUCCESS {
                return Err(format!("GetLastPresentStatus: {} ({r})", result_name(r)));
            }
            Ok(s)
        }

        pub fn sleep(&self, frame_id: u32) {
            let r = unsafe { (self.xell.sleep)(self.xell_ctx, frame_id) };
            if r != XEFG_SUCCESS && r != 2 {
                eprintln!("fg: xellSleep: {}", result_name(r));
            }
        }

        pub fn marker(&self, frame_id: u32, marker: u32) {
            unsafe { (self.xell.add_marker)(self.xell_ctx, frame_id, marker) };
        }
    }

    impl Drop for XefgSwapchain {
        fn drop(&mut self) {
            // Destroy tears the FI proxy down; the owner released every proxy
            // ref first (GpuContext field order) and drained the queue. The
            // app chain's ref is released LAST — the proxy delegates to it
            // until destroy returns.
            let r = unsafe { (self.api.destroy)(self.fg) };
            if r != XEFG_SUCCESS {
                eprintln!("fg: xefg destroy: {}", result_name(r));
            }
            let r = unsafe { (self.xell.destroy_context)(self.xell_ctx) };
            if r != XEFG_SUCCESS {
                eprintln!("fg: xell destroy: {}", result_name(r));
            }
            unsafe {
                let _ = IDXGISwapChain3::from_raw(self.app);
            }
        }
    }
}

#[cfg(windows)]
pub use loader::XefgSwapchain;
