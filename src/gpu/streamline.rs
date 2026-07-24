//! Safe wrapper over the Streamline shim (see shim/sl_shim.h). Owns the SL
//! lifecycle: init (manual hooking + frame-based tagging), feature support
//! checks, interface upgrades (device/factory proxies), DLSS-RR options and
//! the per-frame token/constants/tags/evaluate sequence.

use super::streamline_sys as sys;
use std::ffi::c_void;
use windows::core::Interface;

pub use sys::{SlShimConstants, SlShimDlssdOptimal, SlShimDlssdOptions, SlShimResourceTag};

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Streamline log messages -> stderr. type: 0 info, 1 warn, 2 error.
extern "C" fn log_cb(ty: u32, msg: *const i8) {
    let text = unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy();
    let text = text.trim_end();
    match ty {
        0 => eprintln!("[sl] {text}"),
        1 => eprintln!("[sl WARN] {text}"),
        _ => eprintln!("[sl ERROR] {text}"),
    }
}

/// Development placeholder (the Streamline sample's id). A real NGX
/// application id is only needed for shipping builds.
const DEV_APP_ID: u32 = 231313132;

pub struct SlContext {
    _priv: (),
}

impl SlContext {
    /// LoadLibrary the interposer from `sl_dir` and slInit with manual
    /// hooking + frame-based resource tagging. Must run before any DXGI
    /// factory/swapchain creation.
    pub fn init(sl_dir: &str, features: &[u32], verbose: bool) -> Result<Self, String> {
        let interposer = format!("{sl_dir}\\sl.interposer.dll");
        if !std::path::Path::new(&interposer).exists() {
            return Err(format!("Streamline interposer not found at {interposer}"));
        }
        let interposer_w = wide(&interposer);
        let plugins_w = wide(sl_dir);
        let desc = sys::SlShimInitDesc {
            interposer_path: interposer_w.as_ptr(),
            plugins_path: plugins_w.as_ptr(),
            app_id: DEV_APP_ID,
            show_console: 0,
            log_level: if verbose { sys::LOG_LEVEL_VERBOSE } else { sys::LOG_LEVEL_DEFAULT },
            log_cb: Some(log_cb),
            features: features.as_ptr(),
            num_features: features.len() as u32,
        };
        let r = unsafe { sys::slshim_load_and_init(&desc) };
        if r != sys::SL_OK {
            return Err(format!("slInit failed: {}", result_str(r)));
        }
        Ok(Self { _priv: () })
    }

    pub fn is_feature_supported(&self, feature: u32, luid: windows::Win32::Foundation::LUID) -> Result<(), String> {
        // LUID is 8 bytes {LowPart: u32, HighPart: i32} — SL wants raw bytes.
        let bytes: [u8; 8] = unsafe { std::mem::transmute(luid) };
        let r = unsafe {
            sys::slshim_is_feature_supported(feature, bytes.as_ptr() as *const c_void, 8)
        };
        if r != sys::SL_OK {
            return Err(result_str(r));
        }
        Ok(())
    }

    pub fn set_d3d_device(&self, device: &windows::Win32::Graphics::Direct3D12::ID3D12Device) -> Result<(), String> {
        let r = unsafe { sys::slshim_set_d3d_device(device.as_raw()) };
        if r != sys::SL_OK {
            return Err(format!("slSetD3DDevice failed: {}", result_str(r)));
        }
        Ok(())
    }

    /// Upgrade a COM interface to its Streamline proxy. The passed-in
    /// reference is untouched (we AddRef via clone); the returned wrapper
    /// owns the proxy.
    pub fn upgrade<T: Interface>(&self, iface: &T) -> Result<T, String> {
        let mut raw = iface.clone().into_raw();
        let r = unsafe { sys::slshim_upgrade_interface(&mut raw) };
        if r != sys::SL_OK {
            // Ownership of the original ref was not consumed on failure —
            // reclaim it so the AddRef doesn't leak.
            let _ = unsafe { T::from_raw(raw) };
            return Err(format!("slUpgradeInterface failed: {}", result_str(r)));
        }
        Ok(unsafe { T::from_raw(raw) })
    }

    /// Returns the native interface behind an SL proxy, or an error if the
    /// pointer is not a proxy — used to assert the swapchain really is
    /// SL-hooked (presentCommon must fire every frame under manual hooking).
    pub fn native_of_raw(&self, proxy: *mut c_void) -> Result<*mut c_void, String> {
        let mut native = std::ptr::null_mut();
        let r = unsafe { sys::slshim_get_native_interface(proxy, &mut native) };
        if r != sys::SL_OK || native.is_null() {
            return Err(format!("slGetNativeInterface failed: {}", result_str(r)));
        }
        Ok(native)
    }

    pub fn dlssd_optimal_settings(&self, o: &SlShimDlssdOptions) -> Result<SlShimDlssdOptimal, String> {
        let mut out = SlShimDlssdOptimal::default();
        let r = unsafe { sys::slshim_dlssd_get_optimal_settings(o, &mut out) };
        if r != sys::SL_OK {
            return Err(format!("slDLSSDGetOptimalSettings failed: {}", result_str(r)));
        }
        Ok(out)
    }

    pub fn dlssd_set_options(&self, viewport: u32, o: &SlShimDlssdOptions) -> Result<(), String> {
        let r = unsafe { sys::slshim_dlssd_set_options(viewport, o) };
        if r != sys::SL_OK {
            return Err(format!("slDLSSDSetOptions failed: {}", result_str(r)));
        }
        Ok(())
    }

    pub fn dlssd_vram(&self, viewport: u32) -> Result<u64, String> {
        let mut vram = 0u64;
        let r = unsafe { sys::slshim_dlssd_get_state(viewport, &mut vram) };
        if r != sys::SL_OK {
            return Err(format!("slDLSSDGetState failed: {}", result_str(r)));
        }
        Ok(vram)
    }

    pub fn new_frame_token(&self, frame_index: u32) -> Result<*mut c_void, String> {
        let mut token = std::ptr::null_mut();
        let r = unsafe { sys::slshim_new_frame_token(&frame_index, &mut token) };
        if r != sys::SL_OK || token.is_null() {
            return Err(format!("slGetNewFrameToken failed: {}", result_str(r)));
        }
        Ok(token)
    }

    pub fn set_constants(&self, token: *mut c_void, viewport: u32, c: &SlShimConstants) -> Result<(), String> {
        let r = unsafe { sys::slshim_set_constants(token, viewport, c) };
        if r != sys::SL_OK {
            return Err(format!("slSetConstants failed: {}", result_str(r)));
        }
        Ok(())
    }

    pub fn tag_resources(
        &self,
        token: *mut c_void,
        viewport: u32,
        tags: &[SlShimResourceTag],
        cmdlist: *mut c_void,
    ) -> Result<(), String> {
        let r = unsafe {
            sys::slshim_tag_resources(token, viewport, tags.as_ptr(), tags.len() as u32, cmdlist)
        };
        if r != sys::SL_OK {
            return Err(format!("slSetTagForFrame failed: {}", result_str(r)));
        }
        Ok(())
    }

    pub fn evaluate(&self, feature: u32, token: *mut c_void, viewport: u32, cmdlist: *mut c_void) -> Result<(), String> {
        let r = unsafe { sys::slshim_evaluate(feature, token, viewport, cmdlist) };
        if r != sys::SL_OK {
            return Err(format!("slEvaluateFeature failed: {}", result_str(r)));
        }
        Ok(())
    }

    // ---- DLSS-G + Reflex/PCL (frame generation) ----

    pub fn dlssg_set_options(&self, viewport: u32, o: &sys::SlShimDlssgOptions) -> Result<(), String> {
        let r = unsafe { sys::slshim_dlssg_set_options(viewport, o) };
        if r != sys::SL_OK {
            return Err(format!("slDLSSGSetOptions failed: {}", result_str(r)));
        }
        Ok(())
    }

    pub fn dlssg_state(&self, viewport: u32) -> Result<sys::SlShimDlssgState, String> {
        let mut out = sys::SlShimDlssgState::default();
        let r = unsafe { sys::slshim_dlssg_get_state(viewport, &mut out) };
        if r != sys::SL_OK {
            return Err(format!("slDLSSGGetState failed: {}", result_str(r)));
        }
        Ok(out)
    }

    /// Must run at least once for DLSS-G (even at mode 0); the sleep is the
    /// per-frame half.
    pub fn reflex_set_options(&self, mode: u32) -> Result<(), String> {
        let r = unsafe { sys::slshim_reflex_set_options(mode) };
        if r != sys::SL_OK {
            return Err(format!("slReflexSetOptions failed: {}", result_str(r)));
        }
        Ok(())
    }

    pub fn reflex_sleep(&self, token: *mut c_void) -> Result<(), String> {
        let r = unsafe { sys::slshim_reflex_sleep(token) };
        if r != sys::SL_OK {
            return Err(format!("slReflexSleep failed: {}", result_str(r)));
        }
        Ok(())
    }

    pub fn pcl_marker(&self, marker: u32, token: *mut c_void) -> Result<(), String> {
        let r = unsafe { sys::slshim_pcl_set_marker(marker, token) };
        if r != sys::SL_OK {
            return Err(format!("slPCLSetMarker failed: {}", result_str(r)));
        }
        Ok(())
    }
}

impl Drop for SlContext {
    fn drop(&mut self) {
        unsafe { sys::slshim_shutdown() };
    }
}

/// Human-readable subset of sl::Result (sl_result.h) for diagnostics.
fn result_str(r: i32) -> String {
    let name = match r {
        0 => "eOk",
        1 => "eErrorIO",
        2 => "eErrorDriverOutOfDate",
        3 => "eErrorOSOutOfDate",
        4 => "eErrorOSDisabledHWS",
        5 => "eErrorDeviceNotCreated",
        6 => "eErrorNoSupportedAdapterFound",
        7 => "eErrorAdapterNotSupported",
        8 => "eErrorNoPlugins",
        9 => "eErrorVulkanAPI",
        10 => "eErrorDXGIAPI",
        11 => "eErrorD3DAPI",
        12 => "eErrorNRDAPI",
        13 => "eErrorNVAPI",
        14 => "eErrorReflexAPI",
        15 => "eErrorNGXFailed",
        16 => "eErrorJSONParsing",
        17 => "eErrorMissingProxy",
        18 => "eErrorMissingResourceState",
        19 => "eErrorInvalidIntegration",
        20 => "eErrorMissingInputParameter",
        21 => "eErrorNotInitialized",
        22 => "eErrorComputeFailed",
        23 => "eErrorInitNotCalled",
        24 => "eErrorExceptionHandler",
        25 => "eErrorInvalidParameter",
        26 => "eErrorMissingConstants",
        27 => "eErrorDuplicatedConstants",
        28 => "eErrorMissingOrInvalidAPI",
        29 => "eErrorCommonConstantsMissing",
        30 => "eErrorUnsupportedInterface",
        31 => "eErrorFeatureMissing",
        32 => "eErrorFeatureNotSupported",
        33 => "eErrorFeatureMissingHooks",
        34 => "eErrorFeatureFailedToLoad",
        35 => "eErrorFeatureWrongPriority",
        36 => "eErrorFeatureMissingDependency",
        37 => "eErrorFeatureManagerInvalidState",
        38 => "eErrorInvalidState",
        39 => "eWarnOutOfVRAM",
        -1000 => "shim: LoadLibrary failed",
        -1001 => "shim: GetProcAddress failed",
        -1002 => "shim: too many tags",
        -1003 => "shim: feature function fetch before init",
        _ => "unknown",
    };
    format!("{r} ({name})")
}
