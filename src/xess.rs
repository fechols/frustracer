//! Intel XeSS Super Resolution — the third upscaler path next to DLSS-RR and
//! OIDN. Same footprint policy as OIDN: nothing links the SDK; `libxess.dll`
//! is loaded at runtime from the SDK bin directory (default
//! `SDKs/XeSS-SDK/bin`), every entry point is resolved into a fn-pointer
//! table, and headless builds/`--check-xess` stay DLL-free. XeSS is a plain C
//! API (`xess.h` / `xess_d3d12.h`), safe to mirror directly in Rust — the
//! structs below are hand-transcribed from those headers (pack(8), which on
//! x64 with only ptr/u32/u64 fields is the natural `#[repr(C)]` layout).
//!
//! XeSS mode is the 100%-"a pixel is a sample" mode: every frame is a fresh
//! jittered 1-spp full-depth hybrid trace at whatever render resolution the
//! CPU can afford (`ScaleCtl` + `quantize_res`), and XeSS's temporal
//! accumulation of the jittered sample stream is the ONLY spatial
//! reconstruction — no block filling, no `sparse_fill`, no CPU accumulation.
//! Dynamic resolution is first-class in the SDK: input planes are allocated
//! once at the queried max, and each execute names its own
//! `input_width/height` sub-rect.

use std::ffi::c_void;

// ---------------------------------------------------------------------------
// Convention constants — the single place XeSS's undocumented sign/flag
// polarities live (the `gpu/mod.rs::shim_constants` precedent for SL).
// ---------------------------------------------------------------------------

/// Jitter reported to XeSS = JITTER_SIGN * the renderer's sample offset
/// (dlss::jitter_for). SL wants the negated offset; XeSS samples add the
/// offset to the projection matrix directly, so start unnegated. Empirical
/// gate: the wrong sign wobbles a static image by 2x the jitter.
pub const JITTER_SIGN: f32 = 1.0;

/// Multiplier applied to our motion vectors (pixels, y-down, current ->
/// previous — dlss::GPixel::mv) via xessSetVelocityScale. XeSS's default
/// convention matches "sample at (x,y) + mv = where it was last frame" in
/// input-res pixels; flip both signs here if motion smears directionally.
pub const VELOCITY_SCALE: (f32, f32) = (1.0, 1.0);

/// xess_init_flags_t::XESS_INIT_FLAG_INVERTED_DEPTH (xess.h): the depth
/// buffer is reversed-Z (near = 1, far = 0).
pub const XESS_INIT_FLAG_INVERTED_DEPTH: u32 = 1 << 1;

/// xess_init_flags_t::XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE (xess.h): XeSS
/// computes exposure internally; the per-frame `exposure_scale` scalar and
/// exposure texture are ignored.
pub const XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE: u32 = 1 << 8;

/// Init flags (xess_init_flags_t bitmask): HDR color, low-res pixel MVs
/// without jitter, reversed-Z depth — matching what we feed: our MVs are
/// analytic reprojections of the exact hit point (no raster jitter baked in,
/// the same contract as SL's mvec_jittered = 0), depth is reversed-Z [0,1]
/// clip depth (near = 1, sky/far = exactly 0 — reversed-Z is the encoding
/// fp32 keeps the most precision in, which is why the SDK flags it), and the
/// color is linear HDR scaled by the per-frame `exposure_scale` scalar (1.0).
/// `--xess-autoexposure` ORs in ENABLE_AUTOEXPOSURE at init (A/B lever).
pub const INIT_FLAGS: u32 = XESS_INIT_FLAG_INVERTED_DEPTH;

/// xess_quality_settings_t::XESS_QUALITY_SETTING_ULTRA_PERFORMANCE. The
/// init-time setting fixes the LOWER end of the dynamic input range —
/// measured on SDK 2.0.2: QUALITY yields min == its own 1.7x scale (no
/// downward headroom for heavy scenes), ULTRA_PERFORMANCE opens the full
/// [~1/3 scale .. native] span. The controller starts where the default lock
/// would sit regardless (main.rs), so this only widens the relief valve.
pub const QUALITY_SETTING: i32 = 100;

// ---------------------------------------------------------------------------
// FFI mirror of xess.h / xess_d3d12.h
// ---------------------------------------------------------------------------

pub type XessContext = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Xess2D {
    pub x: u32,
    pub y: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct XessVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub reserved: u16,
}

/// xess_d3d12_init_params_t — field order is the ABI.
#[repr(C)]
pub struct XessD3d12InitParams {
    pub output_resolution: Xess2D,
    pub quality_setting: i32,
    pub init_flags: u32,
    pub creation_node_mask: u32,
    pub visible_node_mask: u32,
    /// Optional external heaps; NULL = XeSS allocates internally.
    pub temp_buffer_heap: *mut c_void,
    pub buffer_heap_offset: u64,
    pub temp_texture_heap: *mut c_void,
    pub texture_heap_offset: u64,
    pub pipeline_library: *mut c_void,
}

/// xess_d3d12_execute_params_t — field order is the ABI. All resource
/// pointers are raw ID3D12Resource*; inputs must be in
/// NON_PIXEL_SHADER_RESOURCE state, the output in UNORDERED_ACCESS.
#[repr(C)]
pub struct XessD3d12ExecuteParams {
    pub color_texture: *mut c_void,
    pub velocity_texture: *mut c_void,
    pub depth_texture: *mut c_void,
    pub exposure_scale_texture: *mut c_void,
    pub responsive_pixel_mask_texture: *mut c_void,
    pub output_texture: *mut c_void,
    /// Jitter in [-0.5, 0.5] (input-res pixels).
    pub jitter_offset_x: f32,
    pub jitter_offset_y: f32,
    /// Input color scale (scalar path; no exposure texture bound).
    pub exposure_scale: f32,
    /// Nonzero resets the temporal history this frame.
    pub reset_history: u32,
    /// The dynamic input resolution this frame — the whole point.
    pub input_width: u32,
    pub input_height: u32,
    pub input_color_base: Xess2D,
    pub input_motion_vector_base: Xess2D,
    pub input_depth_base: Xess2D,
    pub input_responsive_mask_base: Xess2D,
    pub reserved0: Xess2D,
    pub output_color_base: Xess2D,
    pub descriptor_heap: *mut c_void,
    pub descriptor_heap_offset: u32,
}

pub const XESS_RESULT_SUCCESS: i32 = 0;

/// xess_result_t -> readable name (warnings are positive, errors negative).
pub fn result_name(r: i32) -> &'static str {
    match r {
        0 => "SUCCESS",
        1 => "WARNING_NONEXISTING_FOLDER",
        2 => "WARNING_OLD_DRIVER",
        -1 => "ERROR_UNSUPPORTED_DEVICE",
        -2 => "ERROR_UNSUPPORTED_DRIVER",
        -3 => "ERROR_UNINITIALIZED",
        -4 => "ERROR_INVALID_ARGUMENT",
        -5 => "ERROR_DEVICE_OUT_OF_MEMORY",
        -6 => "ERROR_DEVICE",
        -7 => "ERROR_NOT_IMPLEMENTED",
        -8 => "ERROR_INVALID_CONTEXT",
        -9 => "ERROR_OPERATION_IN_PROGRESS",
        -10 => "ERROR_UNSUPPORTED",
        -11 => "ERROR_CANT_LOAD_LIBRARY",
        -12 => "ERROR_WRONG_CALL_ORDER",
        -1000 => "ERROR_UNKNOWN",
        _ => "ERROR_(unrecognized)",
    }
}

// ---------------------------------------------------------------------------
// Depth encoding: the tracer stores linear view-Z (t * dot(dir, forward),
// sky = far — dlss::GPixel::view_z); XeSS expects a [0,1] depth buffer. This
// is reversed-Z perspective depth (INIT_FLAGS carries INVERTED_DEPTH) for
// the same near/far the matrices use (dlss::near_far):
// d = near*(far - z) / (z*(far - near)). Monotone decreasing in z,
// d(near) = 1, d(far) = 0 — the zero numerator puts sky exactly on 0, so
// the sentinel cannot drift.
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn view_z_to_clip_depth(view_z: f32, near: f32, far: f32) -> f32 {
    let z = view_z.max(near);
    ((near * (far - z)) / (z * (far - near))).clamp(0.0, 1.0)
}

/// Inverse of `view_z_to_clip_depth` (self-test roundtrip only).
pub fn clip_depth_to_view_z(d: f32, near: f32, far: f32) -> f32 {
    (near * far) / (near + d * (far - near))
}

// ---------------------------------------------------------------------------
// Dynamic-resolution controller — pure math, exercised by --check-xess.
// ---------------------------------------------------------------------------

/// Render-res quantum for the height axis. Coarse on purpose: every step
/// change drops the temporal cache, reallocates the render-res G-buffers,
/// and (in the OIDN sub-mode) recommits the filter — hysteresis by
/// quantization. 36 divides 1080 and is a multiple of 9, so 16:9 windows get
/// exact-integer widths on every step.
pub const RES_STEP: usize = 36;

/// Round-half-up width from the exact window aspect (XeSS requires the input
/// aspect to match the output), clamped into the queried range. The single
/// width source shared by `quantize_res` and `StepLimiter`'s ramp
/// intermediates — one formula, so a ramp's final frame reproduces the
/// quantized endpoint width bit-exactly.
pub fn width_for_height(rh: usize, out: (usize, usize), min_w: usize, max_w: usize) -> usize {
    let (ow, oh) = out;
    ((rh * ow + oh / 2) / oh).clamp(min_w.max(1), max_w.max(1))
}

/// Map a linear scale estimate to a concrete (rw, rh): quantize the height
/// to RES_STEP, derive the width from the window aspect, clamp both into the
/// queried [min, max] range. Clamps override the quantum — the range bounds
/// are hard.
pub fn quantize_res(
    scale: f32,
    out: (usize, usize),
    min: (usize, usize),
    max: (usize, usize),
) -> (usize, usize) {
    let (_, oh) = out;
    let rh_raw = (scale * oh as f32).round() as usize;
    let rh = (rh_raw / RES_STEP * RES_STEP).clamp(min.1.max(RES_STEP), max.1.max(RES_STEP));
    (width_for_height(rh, out, min.0, max.0), rh)
}

/// The render scale every mode locks at by default — native 100%, the
/// DLAA-shaped arm (the wired upscaler runs as a pure antialiaser/denoiser at
/// window res). ONE value for the CPU tracer, `--gpu` and `--dxr` alike:
/// the upscalers are the reconstruction stage in all three, so the arm that
/// happens to be live is not a reason to change how many pixels get traced
/// (F/SPACE cycling between arms therefore no longer moves the render res).
/// HISTORY, fourth move: quality 2/3 from 2026-07-26, native 100% from
/// 2026-07-31, quality 2/3 again briefly on 2026-08-08, native again the same
/// day (the user's call) — so perf numbers recorded at "the flagless default"
/// carry their era's scale: 0.444x-pixels in the two quality windows,
/// full-res in the native ones. `--lock-res quality` spells the 2/3 arm.
/// NOTE this re-closes the vendor_defaults res-basis caveat the quality
/// window had re-opened (see main::vendor_defaults).
pub const DEFAULT_LOCK_SCALE: f32 = 1.0;

/// --lock-res argument -> fixed render scale (both upscaler paths consume it
/// through `quantize_res`, which range-clamps). Named presets are the
/// standard DLSS ratios — "quality" is deliberately the literal 2/3, NOT
/// `DEFAULT_LOCK_SCALE` (they coincide today, but the default has moved three
/// times; the preset vocabulary must not move with it); a bare number is
/// accepted as a ratio in (0, 1] — the filter also rejects NaN.
pub fn lock_scale(arg: &str) -> Option<f32> {
    match arg {
        "quality" => Some(2.0 / 3.0),
        "balanced" => Some(0.58),
        "performance" => Some(0.5),
        "ultra-performance" => Some(1.0 / 3.0),
        "native" => Some(1.0),
        s => s.parse::<f32>().ok().filter(|r| *r > 0.0 && *r <= 1.0),
    }
}

/// Log2-scale proportional controller, the `depth_est` design transplanted
/// from quadtree-depth units to resolution-scale units: cost ~ area ~
/// scale^2, so levels-of-headroom in scale space = 0.5*log2(budget/elapsed).
/// Slow-up (creep toward super-resolution when frames are cheap — e.g. the
/// camera stopped and the temporal cache kicked in), fast-down (a blown
/// frame sheds pixels immediately), deadband above 60% budget so the scale
/// parks instead of flapping across a quantization boundary. No wall clock
/// is read here; the caller feeds it the previous frame's measured time.
pub struct ScaleCtl {
    /// log2 of the linear scale estimate (0 = window height).
    est: f32,
    /// log2 bounds derived from the queried min/max input heights.
    lo: f32,
    hi: f32,
}

const SCALE_GAIN: f32 = 0.6;
const SCALE_STEP_UP_MAX: f32 = 0.03; // ~2%/frame — jitter accumulation likes stability
const SCALE_STEP_DOWN_MAX: f32 = 0.5; // ~30% fewer pixels per axis in one step

impl ScaleCtl {
    pub fn new(opt_h: usize, min_h: usize, max_h: usize, out_h: usize) -> Self {
        let l = |h: usize| (h.max(1) as f32 / out_h as f32).log2();
        Self { est: l(opt_h), lo: l(min_h), hi: l(max_h) }
    }

    /// Current linear scale estimate.
    pub fn scale(&self) -> f32 {
        self.est.exp2()
    }

    pub fn update(&mut self, last_ms: f32, budget_ms: f32) {
        let err = (budget_ms / last_ms.max(0.1)).log2() * 0.5; // headroom in scale space
        let step = (SCALE_GAIN * err).clamp(-SCALE_STEP_DOWN_MAX, SCALE_STEP_UP_MAX);
        if step < 0.0 || last_ms < 0.6 * budget_ms {
            self.est = (self.est + step).clamp(self.lo, self.hi);
        }
    }
}

/// Minimum frames between APPLIED resolution steps (~1.5 s at 60 fps).
pub const STEP_DWELL: u32 = 90;

/// Frames over which an adopted step is ramped from the previous endpoint
/// (~0.4 s at 60 fps); 0 = snap. Intermediates are exact-aspect,
/// range-clamped, and need no RES_STEP quantum; rounded heights repeat, so
/// the temporal cache is dropped on at most |Δh| distinct-res frames per
/// ramp, not on every ramp frame. Must stay below STEP_DWELL so a ramp
/// always completes before the next adoption can fire — `from` is then
/// always a completed, quantized endpoint.
pub const RAMP_FRAMES: u32 = 24;
const _: () = assert!(RAMP_FRAMES < STEP_DWELL);

/// Rate-limits applied resolution changes for both DRS paths. Every applied
/// step costs upscaler history quality (RR re-initializes its denoiser on an
/// input-res change; XeSS rescales its accumulation), and the controller's
/// slow climb after a shed would otherwise cross each RES_STEP quantum as
/// its own step — one shed-then-recover excursion firing several history
/// hits in quick succession, which presents as patchy re-convergence noise
/// ("dancing"). A new target is adopted only after STEP_DWELL frames at the
/// current res; when the dwell expires the CURRENT target is adopted in one
/// multi-quantum decision (one history hit instead of five). The only bypass
/// is an emergency shed on a badly blown frame — growing never bypasses.
///
/// An adoption is a decision, not a jump: with a nonzero `ramp` the output
/// resolution lerps from the previous endpoint to the adopted one over
/// `ramp` frames instead of snapping (the sharpness pop of a multi-quantum
/// jump, spread thin). Only the height lerps; the width is derived from the
/// window aspect each frame via `width_for_height`. An emergency shed still
/// snaps — it exists to relieve a blown frame NOW — and cancels any ramp in
/// flight, including fast-forwarding a down-ramp whose endpoint is already
/// the target.
pub struct StepLimiter {
    /// Adopted endpoint — always a `quantize_res` result.
    applied: (usize, usize),
    /// Ramp start: the previously adopted endpoint (== `applied` when idle).
    from: (usize, usize),
    /// Frames since the last adoption — the dwell counter AND the ramp phase.
    since: u32,
    /// Configured ramp length; 0 = snap.
    ramp: u32,
}

impl StepLimiter {
    pub fn new(ramp: u32) -> Self {
        Self { applied: (0, 0), from: (0, 0), since: 0, ramp }
    }

    /// The adopted endpoint (where any in-flight ramp is headed).
    pub fn endpoint(&self) -> (usize, usize) {
        self.applied
    }

    /// True while a ramp is in flight (this frame's output != the endpoint).
    pub fn ramping(&self) -> bool {
        self.ramp != 0 && self.from != self.applied && self.since < self.ramp
    }

    pub fn apply(
        &mut self,
        target: (usize, usize),
        emergency_shed: bool,
        out: (usize, usize),
        min: (usize, usize),
        max: (usize, usize),
    ) -> (usize, usize) {
        if self.applied.0 == 0 {
            // First frame adopts unconditionally, without a ramp.
            self.applied = target;
            self.from = target;
            self.since = 0;
            return self.applied;
        }
        self.since = self.since.saturating_add(1);
        // The shed gate must see the ramp's CURRENT output, not the endpoint:
        // mid up-ramp, a shed target below the endpoint but above the current
        // output would otherwise GROW resolution on a blown frame. It is
        // deliberately NOT gated on `target != applied`: mid down-ramp with
        // the target already the endpoint, a blown frame must fast-forward
        // the ramp (snap to the endpoint), not keep descending slowly.
        let cur = self.output(out, min, max);
        if emergency_shed && target.1 < cur.1 {
            self.applied = target; // snap, cancelling any ramp in flight
            self.from = target;
            self.since = 0;
        } else if target != self.applied && self.since >= STEP_DWELL {
            self.from = self.applied; // ramp departs the completed endpoint
            self.applied = target;
            self.since = 0;
        }
        self.output(out, min, max)
    }

    /// The resolution THIS frame renders at: the applied endpoint, or a
    /// height-lerped intermediate while a ramp is in flight. Linear in
    /// height (monotone under rounding, endpoint exact at t = 1); the width
    /// is derived per frame, never lerped — an independently lerped width
    /// can drift off the exact window aspect, which XeSS rejects.
    fn output(&self, out: (usize, usize), min: (usize, usize), max: (usize, usize)) -> (usize, usize) {
        if self.ramp == 0 || self.from == self.applied || self.since >= self.ramp {
            return self.applied;
        }
        let t = self.since as f32 / self.ramp as f32;
        let rh = (self.from.1 as f32 + (self.applied.1 as f32 - self.from.1 as f32) * t).round()
            as usize;
        // Endpoints are already range-clamped; the height clamp is defensive.
        let rh = rh.clamp(min.1.max(1), max.1.max(1));
        (width_for_height(rh, out, min.0, max.0), rh)
    }
}

// ---------------------------------------------------------------------------
// Runtime loader + safe-ish wrapper (Windows-only, interactive path only).
// ---------------------------------------------------------------------------

#[cfg(windows)]
pub use loader::Xess;

#[cfg(windows)]
mod loader {
    use super::*;
    use std::ffi::c_void;
    use windows::core::{PCSTR, PCWSTR};
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{
        GetProcAddress, LoadLibraryExW, LOAD_WITH_ALTERED_SEARCH_PATH,
    };

    /// The resolved XeSS C entry points (undecorated x64 exports).
    struct Api {
        get_version: unsafe extern "C" fn(*mut XessVersion) -> i32,
        create_context: unsafe extern "C" fn(*mut c_void, *mut XessContext) -> i32,
        get_optimal_input_resolution: unsafe extern "C" fn(
            XessContext,
            *const Xess2D,
            i32, // xess_quality_settings_t
            *mut Xess2D,
            *mut Xess2D,
            *mut Xess2D,
        ) -> i32,
        d3d12_init: unsafe extern "C" fn(XessContext, *const XessD3d12InitParams) -> i32,
        d3d12_execute:
            unsafe extern "C" fn(XessContext, *mut c_void, *const XessD3d12ExecuteParams) -> i32,
        set_velocity_scale: unsafe extern "C" fn(XessContext, f32, f32) -> i32,
        is_optimal_driver: unsafe extern "C" fn(XessContext) -> i32,
        destroy_context: unsafe extern "C" fn(XessContext) -> i32,
    }

    fn load_dll(dir: &str, name: &str) -> Result<HMODULE, String> {
        let path = format!("{dir}\\{name}");
        let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
        unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, LOAD_WITH_ALTERED_SEARCH_PATH) }
            .map_err(|e| format!("failed to load {path}: {e}"))
    }

    macro_rules! resolve {
        ($h:expr, $name:literal) => {{
            let sym = unsafe { GetProcAddress($h, PCSTR(concat!($name, "\0").as_ptr())) }
                .ok_or_else(|| format!("libxess.dll: missing export {}", $name))?;
            #[allow(clippy::missing_transmute_annotations)]
            let f = unsafe { std::mem::transmute(sym) };
            f
        }};
    }

    fn ok(r: i32, what: &str) -> Result<(), String> {
        if r >= XESS_RESULT_SUCCESS {
            if r > 0 {
                eprintln!("xess: {what}: {}", result_name(r));
            }
            Ok(())
        } else {
            Err(format!("xess: {what} failed: {} ({r})", result_name(r)))
        }
    }

    /// A live XeSS D3D12 context. Owns the fn-pointer table and the context
    /// handle; the HMODULE is deliberately never freed (the SL/OIDN policy).
    /// The caller guarantees the GPU is idle before drop (GpuContext's Drop
    /// waits the queue) — xessDestroyContext requires completed command lists.
    pub struct Xess {
        api: Api,
        ctx: XessContext,
    }

    impl Xess {
        /// Load the DLL from `dll_dir` and create + initialize a context on
        /// `device` (a raw ID3D12Device*) for `out` output resolution at
        /// QUALITY_SETTING. Returns the context plus the (optimal, min, max)
        /// input resolutions — the dynamic range every frame must stay in.
        /// `autoexposure` ORs XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE into
        /// INIT_FLAGS (the flag values/polarity story stays single-sourced
        /// at the top of this file; only the runtime bool travels in).
        pub fn new(
            dll_dir: &str,
            device: *mut c_void,
            out: (u32, u32),
            autoexposure: bool,
        ) -> Result<(Self, Xess2D, Xess2D, Xess2D), String> {
            // Absolute path: ALTERED_SEARCH_PATH only helps absolute paths.
            let dir = std::fs::canonicalize(dll_dir)
                .map(|p| p.to_string_lossy().trim_start_matches(r"\\?\").to_string())
                .unwrap_or_else(|_| dll_dir.to_string());
            let h_dll = load_dll(&dir, "libxess.dll")?;
            let api = Api {
                get_version: resolve!(h_dll, "xessGetVersion"),
                create_context: resolve!(h_dll, "xessD3D12CreateContext"),
                get_optimal_input_resolution: resolve!(h_dll, "xessGetOptimalInputResolution"),
                d3d12_init: resolve!(h_dll, "xessD3D12Init"),
                d3d12_execute: resolve!(h_dll, "xessD3D12Execute"),
                set_velocity_scale: resolve!(h_dll, "xessSetVelocityScale"),
                is_optimal_driver: resolve!(h_dll, "xessIsOptimalDriver"),
                destroy_context: resolve!(h_dll, "xessDestroyContext"),
            };

            let mut ver = XessVersion::default();
            if unsafe { (api.get_version)(&mut ver) } == XESS_RESULT_SUCCESS {
                eprintln!("xess: SDK {}.{}.{}", ver.major, ver.minor, ver.patch);
            }

            let mut ctx: XessContext = std::ptr::null_mut();
            ok(unsafe { (api.create_context)(device, &mut ctx) }, "D3D12CreateContext")?;
            let this = Self { api, ctx }; // owns ctx from here — destroyed on early error

            if unsafe { (this.api.is_optimal_driver)(this.ctx) } == 2 {
                eprintln!("xess: WARNING_OLD_DRIVER — expect degraded performance/quality");
            }

            let out2d = Xess2D { x: out.0, y: out.1 };
            let (mut opt, mut min, mut max) =
                (Xess2D::default(), Xess2D::default(), Xess2D::default());
            ok(
                unsafe {
                    (this.api.get_optimal_input_resolution)(
                        this.ctx,
                        &out2d,
                        QUALITY_SETTING,
                        &mut opt,
                        &mut min,
                        &mut max,
                    )
                },
                "GetOptimalInputResolution",
            )?;
            if opt.x == 0 || opt.y == 0 || max.x == 0 || max.y == 0 {
                return Err("xess: degenerate input-resolution query".into());
            }

            let params = XessD3d12InitParams {
                output_resolution: out2d,
                quality_setting: QUALITY_SETTING,
                init_flags: INIT_FLAGS
                    | if autoexposure { XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE } else { 0 },
                creation_node_mask: 0,
                visible_node_mask: 0,
                temp_buffer_heap: std::ptr::null_mut(),
                buffer_heap_offset: 0,
                temp_texture_heap: std::ptr::null_mut(),
                texture_heap_offset: 0,
                pipeline_library: std::ptr::null_mut(),
            };
            ok(unsafe { (this.api.d3d12_init)(this.ctx, &params) }, "D3D12Init")?;
            ok(
                unsafe {
                    (this.api.set_velocity_scale)(this.ctx, VELOCITY_SCALE.0, VELOCITY_SCALE.1)
                },
                "SetVelocityScale",
            )?;
            Ok((this, opt, min, max))
        }

        /// Record upscaling commands into `list` (raw ID3D12GraphicsCommandList*).
        pub fn execute(
            &self,
            list: *mut c_void,
            params: &XessD3d12ExecuteParams,
        ) -> Result<(), String> {
            ok(unsafe { (self.api.d3d12_execute)(self.ctx, list, params) }, "D3D12Execute")
        }
    }

    impl Drop for Xess {
        fn drop(&mut self) {
            if !self.ctx.is_null() {
                let r = unsafe { (self.api.destroy_context)(self.ctx) };
                if r < XESS_RESULT_SUCCESS {
                    eprintln!("xess: DestroyContext: {}", result_name(r));
                }
            }
        }
    }
}
