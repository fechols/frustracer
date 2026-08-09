//! GPU presentation layer: SDL3 hands us an HWND, we own a D3D12 device on
//! the NVIDIA adapter, a DXGI swapchain, and the upload/fullscreen-pass
//! machinery. Everything here consumes finished CPU frames after
//! `render_frame`/`resolve` return — no tracer state is touched.
//!
//! Milestones: M1 `present_cpu` (blit of the CPU-tonemapped frame),
//! M2 `present_hdr` (GPU tonemap of the raw accumulation), M3 Streamline
//! proxy plumbing, M4 `present_rr` (DLSS Ray Reconstruction).

pub mod adapter;
pub mod autoexp;
pub mod d3d12;
/// What the monitor under the window can actually display (HDR on/off, peak
/// luminance) — re-probed on every move, not a startup fact.
pub mod display;
pub mod dual;
pub mod dxc;
pub mod dxr;
pub mod ffx;
pub mod ffx_rr;
pub mod ffx_sys;
pub mod ffx_up;
pub mod gputime;
/// The HUD/menu overlay's GPU half: dirty-rect texture uploads + the
/// premultiplied composite draw inside `fullscreen_to_backbuffer`.
pub mod hud;
/// Raw-NGX DLSS-G's guide conversion (`--fg` DLSS sessions): clip depth +
/// reflection-aware motion vectors.
pub mod ngxfg_guides;
pub mod ngxrr;
pub mod pix;
/// `--quinlight`: the registered-consensus fuse of every wired upscaler.
pub mod quin;
pub mod rr;
pub mod bc7gpu;
pub mod bloom;
pub mod frd_gpu;
pub mod nrd_gpu;
pub mod tonemap;
pub mod trace;
pub mod upload;
pub mod xr;

use crate::dlss;
use d3d12::{transition, D3d, Result};
use std::sync::atomic::AtomicU32;
use windows::core::Interface;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;

pub struct GpuOptions {
    /// The temporal-upscaler fallback chain: which levels of
    /// DLSS-RR → FSR4-RR → XeSS → FSR3 to probe, in that fixed order — the
    /// first level whose support probe passes is wired for the session
    /// (exactly one upscaler per session: DLSS decides the SL-proxy-vs-native
    /// device split up front, and the native levels are first-hit-wins).
    /// Every level exhausted = plain presentation with a loud line.
    pub chain: crate::upchain::UpChain,
    /// Directory holding libxess.dll.
    pub xess_dir: String,
    /// Directory holding amd_fidelityfx_loader_dx12.dll + provider DLLs.
    pub ffx_dir: String,
    /// Explicit adapter vendor preference (--prefer-nvidia / --prefer-intel /
    /// --prefer-amd). None = NVIDIA (main.rs flips the default to AMD when
    /// FSR was explicitly forced). A preference, not a requirement — the
    /// per-level support probes still gate, so a pick without DLSS/FSR
    /// support just falls through the chain.
    pub prefer: Option<adapter::Prefer>,
    /// OR XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE into the XeSS init flags
    /// (--xess-autoexposure; A/B lever, default off).
    pub xess_autoexposure: bool,
    /// Ray Regeneration tuning overrides applied at denoiser creation
    /// (`--fsr-max-radiance` &c). All-None = configure nothing = the
    /// provider's own defaults.
    pub fsr_tune: crate::fsr::DenoiseTuning,
    /// D3D12 debug layer + DXGI debug factory + verbose SL logging.
    pub debug: bool,
    /// Present at sync interval 1 (default). `--no-vsync` clears it for
    /// uncapped benchmark presentation (tearing swapchain when DXGI
    /// supports it).
    pub vsync: bool,
    /// `--hdr`: ask for the 10-bit R10G10B10A2 swapchain (PQ or gamma-2.2 by
    /// the display probe) instead of the 8-bit SDR one. A *request* — the
    /// G2084 declare may be refused, so read `GpuContext::encoding()` for
    /// what actually happened, never this.
    pub hdr: bool,
    /// `--hdr10`: force the 10-bit PQ (R10G10B10A2 + G2084) swapchain in ANY
    /// session — the A/B lever, and override-wins like `--hdr-peak` (it fires
    /// even where the display probe says HDR is off; the probe can be wrong,
    /// and a lever that no-ops exactly then is no escape hatch). Read
    /// `GpuContext::encoding()` for what actually happened.
    pub hdr10: bool,
    /// `--no-hdr10`: force the 10-bit gamma-2.2 (Sdr10) arm — "10-bit, but
    /// NOT PQ" — even on an HDR-on display. The A/B lever; without it the
    /// Sdr10 arm would be unreachable from the command line on an HDR box.
    pub sdr10: bool,
    /// Where linear 1.0 lands, in nits (`--hdr-paper-white`, default 200). The
    /// scene is authored so 1.0 ≈ diffuse white (see `scene::default_light`).
    pub paper_white: f32,
    /// Override the display's reported peak luminance (`--hdr-peak`). None =
    /// use whatever `gpu::hdr` probes from the monitor.
    pub peak_nits: Option<f32>,
    /// `--quinlight`: wire EVERY chain level the box supports instead of the
    /// first, run them all over the same traced frame, and present the
    /// registered-consensus fuse of their outputs (gpu/quin.rs). The one
    /// session shape where the "exactly one upscaler" rule is deliberately
    /// suspended.
    pub quin: bool,
    /// `--quin-anchor N`: which engine defines the fuse's spatial frame (it is
    /// never warped). None = the first wired engine, which is the highest chain
    /// level present — i.e. a DENOISING one wherever the box has one, which is
    /// what you want as the anchor (see the quin docs).
    pub quin_anchor: Option<u32>,
    /// Frame generation for the session — ON BY DEFAULT (`--no-fg` clears
    /// it). Family follows the wired upscaler — native sessions (FSR4-RR /
    /// FSR3 / XeSS) take the FidelityFX frame-interpolation swapchain built
    /// here; DLSS sessions take raw-NGX DLSS-G when the shim is built in.
    /// Unsupported combinations fall through with a loud line, never an
    /// error; --quinlight sessions compose (the fuse's own arms carry the
    /// per-family per-frame contract).
    pub fg: bool,
    /// Directory holding amd_fidelityfx_framegeneration_dx12.dll (`--fg-path`;
    /// the prebuilt drop ships it in the FSR sample dir, NOT next to the
    /// loader the default --ffx-path points at).
    pub fg_dir: String,
    /// `--dual-gpu N`: hand the second adapter N of the level-`dual_depth` tile
    /// rows. None = single-GPU, and every dual code path is then structurally
    /// unreachable. Armed lazily by `init_trace` (it needs the scene), and a
    /// share the balancer drives to zero is the pre-feature path exactly.
    pub dual_gpu: Option<u32>,
    /// `--dual-gpu-depth K`: the quadtree level the split happens at, so the
    /// share granularity is 1/2^K. Trades balance resolution against duplicated
    /// ladder work (levels 0..K run on BOTH devices).
    pub dual_depth: u32,
    /// `--dual-gpu-auto`: let the balancer move the share rather than pinning
    /// it where the flag put it.
    pub dual_auto: bool,
    /// `--dual-gpu-arm`: force the SECONDARY's pipeline. None = the vendor
    /// policy (`dual::arm_for`), which is the shipping default.
    pub dual_arm: Option<dual::Arm>,
}

/// Which chain level a session actually wired — derived from the live state
/// (the Options can never disagree with reality). `Plain` = nothing wired
/// (--no-upscale or chain exhausted).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WiredUpscaler {
    /// --quinlight: several levels wired at once, presented through the fuse.
    Quin,
    Rr,
    Fsr4,
    Xess,
    Fsr3,
    Plain,
}

/// Which pre-upscale denoiser a session asks for (the ONE slot both GPU arms
/// share). Computed once in main from the post-exclusivity Opts and passed to
/// init_trace/init_dxr — the FsrRes variant-IS-the-fact discipline.
#[derive(Clone, Copy)]
pub enum DnKind<'a> {
    /// NRD (ReBLUR) — the directory holding NRD.dll.
    Nrd(&'a str),
    /// FRD — the from-scratch engine (src/frd.rs); no external artifact,
    /// kernels compile through the session DXC like every other unit.
    Frd,
}

/// The live engine in that slot. One enum rather than two Options so
/// one-denoiser-per-session is structural, and so nrd_frame_step / the shed
/// machinery / the presenters' `is_some()` predicates work engine-blind.
enum DnGpu {
    Nrd(nrd_gpu::NrdGpu),
    Frd(frd_gpu::FrdGpu),
}

impl DnGpu {
    fn size(&self) -> (u32, u32) {
        match self {
            DnGpu::Nrd(g) => (g.rw, g.rh),
            DnGpu::Frd(g) => (g.rw, g.rh),
        }
    }

    fn tag(&self) -> &'static str {
        match self {
            DnGpu::Nrd(_) => "nrd",
            DnGpu::Frd(_) => "frd",
        }
    }

    fn matches(&self, kind: &DnKind) -> bool {
        matches!(
            (self, kind),
            (DnGpu::Nrd(_), DnKind::Nrd(_)) | (DnGpu::Frd(_), DnKind::Frd)
        )
    }

    // The wire-plane accessors arm_denoiser_for's register block reads —
    // identical names/formats on both engines (the FrdGpu plane contract).
    fn plane_in_mv(&self) -> &ID3D12Resource {
        match self {
            DnGpu::Nrd(g) => g.plane_in_mv(),
            DnGpu::Frd(g) => g.plane_in_mv(),
        }
    }
    fn plane_in_nr(&self) -> &ID3D12Resource {
        match self {
            DnGpu::Nrd(g) => g.plane_in_nr(),
            DnGpu::Frd(g) => g.plane_in_nr(),
        }
    }
    fn plane_in_viewz(&self) -> &ID3D12Resource {
        match self {
            DnGpu::Nrd(g) => g.plane_in_viewz(),
            DnGpu::Frd(g) => g.plane_in_viewz(),
        }
    }
    fn plane_in_diff(&self) -> &ID3D12Resource {
        match self {
            DnGpu::Nrd(g) => g.plane_in_diff(),
            DnGpu::Frd(g) => g.plane_in_diff(),
        }
    }
    fn plane_in_spec(&self) -> &ID3D12Resource {
        match self {
            DnGpu::Nrd(g) => g.plane_in_spec(),
            DnGpu::Frd(g) => g.plane_in_spec(),
        }
    }
    fn plane_out_diff(&self) -> &ID3D12Resource {
        match self {
            DnGpu::Nrd(g) => g.plane_out_diff(),
            DnGpu::Frd(g) => g.plane_out_diff(),
        }
    }
    fn plane_out_spec(&self) -> &ID3D12Resource {
        match self {
            DnGpu::Nrd(g) => g.plane_out_spec(),
            DnGpu::Frd(g) => g.plane_out_spec(),
        }
    }
}

/// A live XeSS-SR context + its resource set and the queried input-resolution
/// range (optimal, min, max) every dynamic frame must stay inside.
struct XessState {
    ctx: crate::xess::Xess,
    res: xr::XessResources,
    opt: (u32, u32),
    min: (u32, u32),
    max: (u32, u32),
}

/// The flavor's resource set: the full Ray Regeneration plane/composite
/// machinery, or the three-plane upscale-only set — a 3.1 session must not
/// commit RR's nine planes + signal UAVs.
enum FsrRes {
    Rr(ffx_rr::FsrResources),
    Up(ffx_up::Fsr3Resources),
}

impl FsrRes {
    fn upscaled(&self) -> &ID3D12Resource {
        match self {
            FsrRes::Rr(r) => &r.upscaled,
            FsrRes::Up(r) => &r.upscaled,
        }
    }

    /// The variant IS the flavor — derived, so the two can never disagree.
    fn flavor(&self) -> crate::fsr::Flavor {
        match self {
            FsrRes::Rr(_) => crate::fsr::Flavor::Fsr4Rr,
            FsrRes::Up(_) => crate::fsr::Flavor::Fsr3,
        }
    }

    /// (depth, mvec, mv_scale) for the FG prepare — each flavor's plane pair
    /// with the SAME mv_scale its own upscale dispatch uses (the trio carries
    /// pixels, the RR plane UV deltas), so FG and upscaler read one MV
    /// convention per session by construction.
    fn fg_inputs(&self, rw: u32, rh: u32) -> (&ID3D12Resource, &ID3D12Resource, [f32; 2]) {
        match self {
            FsrRes::Up(r) => {
                let (d, m) = r.fg_inputs();
                (d, m, [crate::fsr::UPSCALE_MV_SIGN.0, crate::fsr::UPSCALE_MV_SIGN.1])
            }
            FsrRes::Rr(r) => {
                let (d, m) = r.fg_inputs();
                (d, m, [crate::fsr::UPSCALE_MV_SIGN.0 * rw as f32, crate::fsr::UPSCALE_MV_SIGN.1 * rh as f32])
            }
        }
    }
}

/// Live FSR contexts on the native device + the flavor's resources and the
/// dynamic input-resolution range: seed from the Quality-mode query, floor
/// from UltraPerformance, max = the window (every ffx context takes a
/// per-dispatch renderSize, so no reallocation ever happens on a res step).
/// `Fsr4Rr` = Ray Regeneration denoiser + FSR4 upscaler; `Fsr3` = the 3.1
/// upscaler alone (`fsr::pick_version` chose the provider at init).
struct FsrState {
    ctx: ffx::FfxContext,
    res: FsrRes,
    opt: (u32, u32),
    min: (u32, u32),
    max: (u32, u32),
}

/// Frame generation over the FidelityFX frame-interpolation swapchain (the
/// native-session `--fg` family — DLSS sessions use DLSS-G instead). The FI
/// swapchain proxy replaced `D3d::swapchain` at creation (d3d12::SwapWrap);
/// `sc` is the context that owns it, `ctx` the display-size-bound FG effect
/// (None = creation failed or resize is mid-rebuild — the proxy then presents
/// passthrough, never wrongly).
///
/// The per-frame protocol ("prepared" handshake): an arm that can feed FG
/// calls `GpuContext::fg_prepare` before `fullscreen_to_backbuffer` — that
/// advances `frame_id` by EXACTLY 1 (the ffx contract; any other delta resets
/// interpolation history), configures the swapchain live, and records the
/// PrepareV2 dispatch (depth + MVs) into the frame's list. The funnel then
/// consumes `prepared`; a frame presented WITHOUT a prepare (plain arms, mode
/// switches, the pause-menu hold) finds it false and configures the FI
/// swapchain DISABLED first, so pacing never runs against stale motion.
/// Cells because the funnel and the hold path run on `&self`.
struct FgState {
    sc: ffx::FgSwapchain,
    ctx: Option<ffx::FgContext>,
    /// The user-facing toggle (`--fg` starts true; set_fg_enabled flips).
    enabled: bool,
    /// What the FI swapchain was last CONFIGURED to (avoid re-issuing
    /// disable configures every held frame).
    live: std::cell::Cell<bool>,
    /// Set by fg_prepare, consumed by fullscreen_to_backbuffer.
    prepared: std::cell::Cell<bool>,
    frame_id: std::cell::Cell<u64>,
    /// Last frame's wall time, stashed once per loop by `fg_set_frame_ms`
    /// (main.rs owns the clock — the renderer never reads one), read by every
    /// arm's prepare. Pacing quality follows this number.
    frame_ms: std::cell::Cell<f32>,
    /// (min, max) nits handed to the HDR dispatch patch; (0,0) = SDR, no
    /// patch.
    luminance: [f32; 2],
    /// The picked provider (override id + display name) — the boot line and
    /// the title bar read the name.
    version: (u64, String),
    /// FR_FG_TRACE=1: per-prepare diagnostics (reset prepares, depth/MV
    /// resource-set swaps) — the AMD mode-cycle-slowdown investigation lever.
    trace: bool,
    /// Last prepare's (depth, mvec) resource pointers — a change is the
    /// render-mode-switch signature the trace reports (each arm feeds FG its
    /// own planes: CPU-upload vs wavefront-pack vs DXR-pack).
    last_res: std::cell::Cell<(usize, usize)>,
    /// Count of live→disabled transitions (the pause/resume lines).
    pauses: std::cell::Cell<u32>,
    /// The mode-switch straddle (set by `fg_mode_switch`, consumed by
    /// `fg_prepare`): skip exactly ONE prepare so the funnel presents one
    /// frame with the FI proxy configured disabled — the K-toggle sequence
    /// compressed to a single frame, which is MEASURED (R9700, 2026-07-31) to
    /// clear the AMD provider's wedged pacing state instead of carrying it
    /// across the switch's reset+resource+cadence discontinuity.
    skip_prepare: std::cell::Cell<bool>,
    /// What the FI swapchain's UI registration currently holds: true = the
    /// HudFi display-space texture, false = null. Dedup for
    /// `fg_register_ui` — the proxy must never be re-configured per steady
    /// frame, and every disable path drives this to null.
    ui_reg: std::cell::Cell<bool>,
    /// A UI registration configure failed once — latch and stop trying (the
    /// baked backbuffer draw covers every frame after; loud once).
    ui_shed: std::cell::Cell<bool>,
}

/// Raw-NGX DLSS-G FFI (shim/dlssg_shim.cpp — the quinlight-player blueprint).
/// The dispatch struct exists in BOTH cfg arms so the call sites typecheck;
/// without the DLSS SDK at build time the fns stub to UNSUPPORTED and the
/// session runs without frame generation (a non-SDK build has no DLSS at
/// all — RR rides the same gate).
mod ngxfg {
    use std::ffi::c_void;

    pub const ERR_UNSUPPORTED: i32 = -3;

    #[repr(C)]
    pub struct FrDlssgDispatch {
        pub cmdlist: *mut c_void,
        pub color: *mut c_void,
        pub motion: *mut c_void,
        pub depth: *mut c_void,
        pub output: *mut c_void,
        pub frame_id: u64,
        pub reset: i32,
        pub view_to_clip: [f32; 16],
        pub clip_to_view: [f32; 16],
        pub clip_to_prev_clip: [f32; 16],
        pub prev_clip_to_clip: [f32; 16],
        pub jitter: [f32; 2],
        pub mv_scale: [f32; 2],
        pub cam_pos: [f32; 3],
        pub cam_up: [f32; 3],
        pub cam_right: [f32; 3],
        pub cam_fwd: [f32; 3],
        pub cam_near: f32,
        pub cam_far: f32,
        pub cam_fov: f32,
        pub cam_aspect: f32,
        pub rend_w: u32,
        pub rend_h: u32,
        pub depth_inverted: i32,
    }
    // The dlssg_shim.cpp twin asserts the identical literals (the ffx FG
    // desc discipline: pin the padding-hole shapes on both sides).
    const _: () = assert!(std::mem::offset_of!(FrDlssgDispatch, view_to_clip) == 52);
    const _: () = assert!(std::mem::size_of::<FrDlssgDispatch>() == 400);

    #[cfg(dlss_ngx)]
    pub const BUILT: bool = true;
    #[cfg(dlss_ngx)]
    unsafe extern "C" {
        pub fn frdlssg_create(
            device: *mut c_void,
            disp_w: u32,
            disp_h: u32,
            rend_w: u32,
            rend_h: u32,
            color_hdr: i32,
            out_handle: *mut *mut c_void,
        ) -> i32;
        pub fn frdlssg_dispatch(handle: *mut c_void, d: *const FrDlssgDispatch) -> i32;
        /// FEATURE-scoped res move: ReleaseFeature + CreateFeature at the new
        /// display AND render res — params/init untouched (destroy's
        /// DestroyParameters killed the SHARED SL NGX parameter map
        /// mid-session and every subsequent sl.dlss_d evaluate failed
        /// 0xBAD00004 FeatureNotFound). Both sizes travel because a WINDOW
        /// resize routes through here too: rend-only left the feature at the
        /// old display with a larger render res, which NGX rejects at
        /// evaluate with 0xBAD00005 FAIL_InvalidParameter.
        pub fn frdlssg_recreate(
            handle: *mut c_void,
            disp_w: u32,
            disp_h: u32,
            rend_w: u32,
            rend_h: u32,
        ) -> i32;
        pub fn frdlssg_destroy(handle: *mut c_void);
    }

    #[cfg(not(dlss_ngx))]
    pub const BUILT: bool = false;
    #[cfg(not(dlss_ngx))]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn frdlssg_create(
        _device: *mut c_void,
        _dw: u32,
        _dh: u32,
        _rw: u32,
        _rh: u32,
        _hdr: i32,
        _out: *mut *mut c_void,
    ) -> i32 {
        ERR_UNSUPPORTED
    }
    #[cfg(not(dlss_ngx))]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn frdlssg_dispatch(_h: *mut c_void, _d: *const FrDlssgDispatch) -> i32 {
        ERR_UNSUPPORTED
    }
    #[cfg(not(dlss_ngx))]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn frdlssg_recreate(
        _h: *mut c_void,
        _dw: u32,
        _dh: u32,
        _rw: u32,
        _rh: u32,
    ) -> i32 {
        ERR_UNSUPPORTED
    }
    #[cfg(not(dlss_ngx))]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn frdlssg_destroy(_h: *mut c_void) {}
}

/// How many consecutive FG dispatches a MOVED render res must hold before
/// the NGX feature recreates at it. The feature is fixed-res by creation, so
/// following a res move costs a feature release + re-create (queue drain
/// included) — fine once per SPACE/F mode switch, ruinous per frame under a
/// `--lock-res dynamic` ramp, whose per-frame res changes must never
/// qualify. 8 ≈ under a tenth of a second at interactive rates.
const FG_RECREATE_STABLE: u32 = 8;

/// Raw-NGX DLSS-G state (DLSS sessions when the shim is compiled in — see
/// `ngxfg`). No swapchain wrap, no pacer, no handshake: WE evaluate the
/// interpolated frame into `out` and present it BEFORE the real frame
/// (pair-present; vsync spaces the two). The NGX feature is created LAZILY at
/// the first dispatch — the session's locked render res is main.rs's
/// decision, unknown at GpuContext::new.
struct NgxFgState {
    /// The interpolated-frame target: window-res RGBA16F, rests
    /// PIXEL_SHADER_RESOURCE (tonemap source at SRV_SLOT_NGXFG), UAV during
    /// the evaluate.
    out: ID3D12Resource,
    handle: std::cell::Cell<*mut std::ffi::c_void>,
    /// The render res the feature was created at; (0,0) = not yet created.
    /// The NGX feature is fixed-res by creation
    /// (DynamicResolutionScaling=false) — a frame at a different res skips
    /// FG until the moved res HOLDS `FG_RECREATE_STABLE` dispatches, then
    /// the feature FEATURE-SCOPE-recreates at it in place
    /// (`frdlssg_recreate` — never a destroy; see `res_pend` and TRAP 8).
    dims: std::cell::Cell<(u32, u32)>,
    enabled: bool,
    failed: std::cell::Cell<bool>,
    /// A previous frame seeded the feature's internal history — the
    /// interpolated output is only presentable when the PAIR exists.
    primed: std::cell::Cell<bool>,
    frame_id: std::cell::Cell<u64>,
    /// The recreate-storm guard: (pending res, consecutive dispatches seen
    /// at it). A SPACE/F mode switch moves the render res ONCE (e.g. a
    /// dynamic-DRS CPU arm's res vs the GPU arms' locked default; with the
    /// one locked scale every arm shares, most switches move nothing) and
    /// then holds —
    /// it recreates a fraction of a second in; a `--lock-res dynamic` RAMP
    /// changes res every frame, never qualifies, and keeps skipping (the
    /// pre-recreate behavior); a completed DRS step holds >= the 90-frame
    /// dwell and recreates once per adoption.
    res_pend: std::cell::Cell<((u32, u32), u32)>,
    /// Whether the last present was a PAIR (interpolated + real) — exact by
    /// construction, since this path calls Present itself. The title bar's
    /// presented-per-rendered multiplier.
    pair: std::cell::Cell<bool>,
    /// The guide-conversion pass (see `ngxfg_guides.rs`): clip depth (the FG
    /// snippet's depth contract is DLSS-SR's [0,1] buffer, not RR's linear
    /// tag) + reflection-aware MVs (the virtual-image blend — the
    /// DamagedHelmet reflection-swim fix). `None` = BOTH levers disabled it
    /// or a loud creation failure — the evaluate then gets the raw RR
    /// planes.
    guides: Option<std::cell::RefCell<ngxfg_guides::GuidePass>>,
    /// A guide-pass `ensure` failed at feature creation — loud once, then
    /// the raw-plane arms (never a session failure).
    guides_failed: std::cell::Cell<bool>,
    /// FR_NGXFG_DEPTH=linear — hand the evaluate the raw linear view-Z plane
    /// (the round-1 A/B arm).
    depth_linear: bool,
    /// FR_NGXFG_RMV=off — hand the evaluate the raw SURFACE-MV plane (the
    /// A/B arm that brings the reflection swimming back on demand).
    rmv_off: bool,
    /// FR_NGXFG_JITTER: 0 = RAW (un-negated) — THE DEFAULT, settled by
    /// measurement; 1 = zero; 2 = "neg", the SL-negated convention this used
    /// to default to.
    ///
    /// THE NEGATION WAS WRONG, and it shipped because it was reasoned by
    /// analogy instead of measured: Streamline's RR wants a NEGATED sample
    /// offset (settled empirically, and documented as such), so "same NGX
    /// family, one sign" looked safe. It is not — raw NGX wants the offset
    /// AS IS. quinlight could never have caught it (its jitter was 0,0, so
    /// every sign is identical), which is the same blind spot that hid traps
    /// 4-6.
    ///
    /// A sign error misplaces content by TWICE the jitter (~1 px), which is
    /// invisible on diffuse geometry and blatant on a small, extremely bright
    /// specular highlight — the sun reflecting off DamagedHelmet's metal,
    /// where the disc's ~44,000 radiance against a ~1.0 scene turns a 1 px
    /// warp into a strobing smear. Diagnosed by elimination on 2026-07-26:
    /// resolution, frame rate, the resize path, the virtual-image MVs,
    /// PAIR_BACKBUFFERS and the scene were each ruled out by measurement,
    /// and the FidelityFX interpolator was CLEAN on the identical frame —
    /// which localized it to what we hand NGX. Both `raw` and `0` look clean
    /// (they differ by only half a pixel); `raw` is the one that is true.
    jitter_mode: u8,
    /// FR_NGXFG_MV: 0 = pixel scale {1,1} — THE DEFAULT, settled from
    /// dlssg-to-fsr3, which passes DLSSG.MvecScale STRAIGHT into FSR3's
    /// motionVectorScale (unit: pixels) and works across shipped SL titles;
    /// the SDK header's "[-1,1]" comment is stale, and the quinlight-era
    /// {1/rw,1/rh} starved the snippet of geometry motion ~2000× (zero MVs
    /// in quinlight meant any scale "worked"). 1 = "norm" {1/rw,1/rh} (the
    /// round-1 arm), 2 = "neg" {-1,-1}, 3 = "normneg" {-1/rw,-1/rh}.
    mv_mode: u8,
    /// FR_NGXFG_SHOW: 0 = normal pair (interp, then real). 1 = "interp" —
    /// the INTERPOLATED frame for both halves, so its artifacts are
    /// inspectable at full rate instead of strobing against the real frame.
    /// 2 = "real" — the REAL frame for both halves (the evaluate still
    /// runs): NOTHING NGX-made reaches the screen, so any artifact that
    /// survives is in the PRESENT PATH, not the generated frames — the
    /// process-of-elimination null test.
    show_mode: u8,
    /// FR_NGXFG_CAM=identity — quinlight's exact camera block (identity
    /// matrices, axis-aligned basis, near .1/far 10000/fov 60°): the
    /// configuration the snippet was PROVEN with on this box. Calmer
    /// artifacts under it = our camera-matrix plumbing is the poison.
    cam_identity: bool,
    /// FR_NGXFG_MAT=col — pass glam's column-major matrices raw instead of
    /// `row_major`. quinlight's identity matrices are transpose-invariant,
    /// so the raw-NGX matrix majority was never validated (SL may transpose
    /// internally before setting the DLSSG params).
    mat_col: bool,
    /// FR_NGXFG_FFMV=off — skip the round-3 firefly-MV bake (the guide
    /// pass's FF table uploads `ffc = 0`, the provably-identical A/B): the
    /// FG MV plane keeps surface/virtual MVs at glow pixels, and the night
    /// swarm strobes on generated frames again on demand.
    ffmv_off: bool,
    /// LAST SUCCESSFULLY EVALUATED frame's baked swarm — the prev half of
    /// the round-3 MV pair, updated beside `primed` so it stays aligned with
    /// the frame the NGX feature actually retained across skipped/failed
    /// dispatches. Starts `Fireflies::off()`: the bake's count-mismatch
    /// fallback then reprojects the CURRENT pose (camera-motion-only MV),
    /// and the first evaluate presents real-only anyway (`primed`).
    prev_ff: std::cell::Cell<crate::fireflies::Fireflies>,
    /// FR_NGXFG_RIPPLEMV=off — skip the round-4 ripple-MV reconstruction: the
    /// guide kernel takes `n_p = n_c` and reduces to the round-2/3 unfold
    /// bit-for-bit, so water reflections strobe on generated frames again on
    /// demand.
    ripplemv_off: bool,
    /// FR_NGXFG_RIPPLEDT=off — disarm round 4's large-dt confidence fade
    /// (`ngxfg_guides::RIPPLE_DT_LO/HI`): the reconstruction runs unfaded at
    /// any clock delta. The reconstruction itself stays near-exact at those
    /// deltas (the `ripple_probe` test); what the fade withholds is the
    /// resulting 200-550 px/frame MV at 8K density, which measured as severe
    /// water glitching when handed to NGX. NOTE the glitchy unfaded arm was
    /// only ever observed WITH the pre-`wire_cam_far` f16 sky-compare bug
    /// live — this lever on the fixed build is the pending A/B that decides
    /// whether the fade can narrow.
    ripdt_off: bool,
    /// FR_NGXFG_TONEMAP — the range-compression probe (see
    /// `ngxfg_guides::TonePass` for the full diagnosis this exists to settle).
    /// 0 = off (the shipped stream, byte-identical), 1 = `scale`, 2 =
    /// `reinhard`. `tone` is None whenever the mode is 0 or the pass failed to
    /// build, so the dispatch site needs only the one Option check.
    tone_mode: u8,
    tone: Option<std::cell::RefCell<ngxfg_guides::TonePass>>,
    /// LAST SUCCESSFULLY EVALUATED frame's ripple clock — the prev half of
    /// the round-4 pair, updated beside `primed` for exactly the reason
    /// `prev_ff` is: it must name the frame the NGX feature actually
    /// retained, not the last one we recorded.
    ///
    /// `None` (no history yet) yields `t_prev = t_cur`, i.e. a zero gradient
    /// delta and today's kernel. It deliberately does NOT default to 0.0: a
    /// session whose clock is already minutes in would inject a huge bogus
    /// delta on the first armed frame — a confident wrong MV, which is worse
    /// than the missing one this round exists to fix.
    prev_clock: std::cell::Cell<Option<f32>>,
}

/// XeSS-FG frame generation (Intel XeSS sessions — the third `--fg` family).
/// The xefg swapchain proxy does interpolation + pacing at Present (the ffx
/// FI shape); XeLL is created and linked inside the wrap (a hard xefg
/// requirement). Same funnel handshake as DLSS-G: the funnel READS
/// `prepared` (an unprepared present disables generation), `xefg_end_frame`
/// consumes it after the XeLL present markers.
struct XefgState {
    sc: crate::xess_fg::XefgSwapchain,
    enabled: bool,
    on: std::cell::Cell<bool>,
    prepared: std::cell::Cell<bool>,
    failed: std::cell::Cell<bool>,
    /// The xefg presentId — u32, +1 per prepared frame (tags, constants,
    /// XeLL sleep/markers all key on it).
    frame_id: std::cell::Cell<u32>,
    poll: std::cell::Cell<u32>,
    logged: std::cell::Cell<bool>,
    /// Last healthy poll's frames-presented-per-present (0 = not yet
    /// measured). The title bar's presented-per-rendered multiplier.
    mult: std::cell::Cell<u32>,
    /// A RES_UI tag failed once — stop tagging (loud once; untagged =
    /// today's baked-HUD interpolation, never a session/FG failure).
    ui_shed: std::cell::Cell<bool>,
}

/// The trace res must sit inside this context's SDK range. The caller already
/// quantize_res-clamped it, but the range is the SDK's contract, so a drift
/// fails loudly at init rather than quietly at execute. Split out of
/// `wire_fsr_feed` because a `--quinlight` FSR 3.1 that SHARES the XeSS feed
/// still has to satisfy its own range — one frame is traced for every engine.
fn fsr_range_check(fs: &FsrState, rw: u32, rh: u32) -> Result<()> {
    if rw < fs.min.0 || rh < fs.min.1 || rw > fs.max.0 || rh > fs.max.1 {
        return Err(format!(
            "trace res {}x{} outside FSR render range {}x{}..{}x{}",
            rw, rh, fs.min.0, fs.min.1, fs.max.0, fs.max.1
        ));
    }
    Ok(())
}

/// Field order is drop order (Rust drops in declaration order). The old
/// Streamline teardown ordering (proxies before slShutdown) died with SL —
/// the raw-NGX session holds its own device clone, so it may drop anywhere;
/// the surviving constraints are the FG swapchain-wrapper ones on `fg`/`fg_x`
/// (declared after `d3d`).

/// The --waveviz funnel draw's source arm (see `GpuContext::waveviz_src`).
#[derive(Clone, Copy, PartialEq)]
enum WvSrc {
    None,
    Trace,
    Dxr,
}

/// `--dual-gpu`, interactive: the second adapter, its own scene upload and
/// tracer, the band transfer, and the balancer.
///
/// DECLARED BEFORE `d3d` IN `GpuContext` AND THAT IS LOAD-BEARING. Fields drop
/// in declaration order, and this holds two things that must die before the
/// primary device does: the transfer's UPLOAD buffer, which is a primary-device
/// resource, and the secondary's `HeadlessGpu`, whose own Drop drains its queue
/// (so no separate wait-idle is needed here — unlike the xess/fsr/fg contexts,
/// which is why this sits outside that guard rather than inside it).
struct DualState {
    hg: trace::HeadlessGpu,
    /// The secondary's OWN scene core. `ensure_scene_gpu`'s Rc is device-bound
    /// and cannot cross, so the scene is uploaded twice — the dominant arming
    /// cost, and the known accept.
    core: std::rc::Rc<trace::SceneGpu>,
    /// The secondary's tracer, which is NOT necessarily the primary's kind:
    /// its arm follows its OWN adapter's vendor (`dual::arm_for`), so a 4090
    /// primary on DXR pairs with an Arc secondary on the wavefront. Both are
    /// driven through `dual::Tracer`, which is what lets one schedule serve
    /// every combination.
    sec: dual::Secondary,
    xf: dual::BandTransfer,
    /// The share, the controllers and the ONE per-frame decision — shared with
    /// `--spin` and `--cinematic`, which is the point (see `dual::Balancer`).
    bal: dual::Balancer,
    /// What the band must carry, both properties of the PRIMARY and both
    /// resolved per frame (`wire_feed_add` can run after construction):
    /// whether it has a full-size pack, and whether its feed reads the guide
    /// half. `pack` false is a plain session, whose pack buffers are dummies.
    pack: bool,
    ext: bool,
    /// The previous frame's secondary + transfer wall time, fed to the balancer
    /// on the NEXT frame: a presenter records the merge and only then knows
    /// them, and the frame's own total is not available until it presents.
    last_sec_ms: f32,
    /// Whether the settle-at-zero verdict has been said. A dual session that
    /// silently stops using its secondary looks exactly like one that never
    /// armed, and those are very different answers — but it only needs saying
    /// once.
    said: bool,
    /// Same, for the mixed-arm stand-down: an fb session would otherwise print
    /// it every frame H is held on.
    said_mixed: bool,
    /// Same, for the pack-disagreement stand-down. A SEPARATE latch rather than
    /// one shared with `said_mixed`: the two denies have independent causes and
    /// can hold at once, and a shared latch would silently swallow whichever
    /// arrived second — which is the one you would need to see, since it is the
    /// rarer of the two.
    said_pack: bool,
}

/// What a resize keeps of `DualState`: everything that is not window-bound.
///
/// Declared ahead of `d3d` in `GpuContext` for the same Drop-order reason
/// `DualState` is.
struct DualKeep {
    hg: trace::HeadlessGpu,
    core: std::rc::Rc<trace::SceneGpu>,
    bal: dual::Balancer,
    /// THE SECONDARY'S ARM, carried across rather than re-decided. It is a
    /// property of the ADAPTER and is settled once, at open: re-running the
    /// policy on re-adoption would let an explicit `--dual-gpu-arm` lapse at
    /// the first F11 — the secondary silently switching pipelines mid-session,
    /// which reads as a frame-time discontinuity at the resize with no line
    /// explaining it — and would re-probe `require_caps` against a device that
    /// has already answered.
    arm: dual::Arm,
}

pub struct GpuContext {
    /// `--dual-gpu`. FIRST FIELD deliberately — see `DualState`.
    dual: Option<DualState>,
    /// The requested share, kept because `dual` is built lazily at the first
    /// `init_trace` (which has the scene). Cleared on a build failure so a
    /// SPACE re-entry does not retry a whole second scene upload per keypress —
    /// the `trace_failed` latch, owned here rather than in main.rs because this
    /// is where the failure is seen.
    dual_want: Option<u32>,
    dual_depth: u32,
    dual_auto: bool,
    /// `--dual-gpu-arm`: the forced secondary pipeline, or None for the vendor
    /// policy. Read once, at `build_dual` — see `DualKeep::arm` for why a
    /// resize must not re-decide it.
    dual_arm: Option<dual::Arm>,
    /// What survives a resize: the secondary's device and its SCENE core (both
    /// window-independent, and the scene upload is the expensive one), plus the
    /// balancer's converged state — a window resize is not a reason to re-learn
    /// which adapter pays. Also declared ahead of `d3d`, for the same reason
    /// `dual` is.
    dual_keep: Option<DualKeep>,
    d3d: D3d,
    passes: tonemap::Passes,
    /// Glare. Always built (never Option): the tonemap PS declares its halo SRV
    /// unconditionally, so the descriptor must be valid even under `--no-bloom`,
    /// where the pass simply isn't recorded and strength is 0.
    bloom: bloom::BloomGpu,
    /// Auto-exposure's luminance meter (gpu/autoexp.rs). Always built, like
    /// bloom — recording is gated per frame on `autoexp::enabled()`, and the
    /// buffers are constant-sized (8K tile budget), so there is no resize path.
    autoexp: autoexp::AutoExpGpu,
    /// The newest collected meter value (mean log2-luminance of a presented
    /// frame's tonemap source, FRAMES_IN_FLIGHT frames old) — stashed by
    /// `fullscreen_to_backbuffer`, consumed by main's controller via
    /// `take_meter`. Cell: the present recorder is `&self`.
    meter: std::cell::Cell<Option<f32>>,
    blit: upload::BlitUpload,
    hdr: upload::HdrUpload,
    /// The HUD/menu overlay (gpu/hud.rs): window-sized premultiplied RGBA8,
    /// dirty-rect-uploaded, composited by `fullscreen_to_backbuffer` in every
    /// present arm. Always built (cheap); `visible` gates the draw.
    hud: hud::HudGpu,
    /// The frame-generation UI target (see hud::HudFi): a display-space
    /// premultiplied render of the HUD the wrapper-FG proxies composite
    /// AFTER interpolation (ffx: registered UI resource; XeSS-FG: RES_UI
    /// tag) — what stops the baked HUD warping/jumping on generated frames.
    /// None = no wrapper FG armed, or creation failed (loud; the baked
    /// backbuffer draw covers every frame then).
    fg_ui: Option<hud::HudFi>,
    /// What `fullscreen_to_backbuffer` last presented `(use_tonemap, srv_slot,
    /// inv_samples)` — `present_again`'s re-present source (the pause menu
    /// holds the frame without tracing). Cell: the recorder is `&self`.
    last_present: std::cell::Cell<Option<(bool, u32, f32)>>,
    /// Which tracer produced the frame being presented — the --waveviz funnel
    /// draw reads exactly that arm's ticket buffer (None = CPU-fed: no GPU
    /// tickets exist, the overlay stands down). Stamped at the top of every
    /// presenter; a Cell so `present_again`'s replay inherits the last real
    /// present's source, the last_present pattern.
    waveviz_src: std::cell::Cell<WvSrc>,
    /// The SHARED scene core both GPU tracers hold an Rc of: uploaded once
    /// per session by whichever of init_trace/init_dxr runs first, so the
    /// second tracer (and every resize re-entry — the device survives
    /// resize_output) skips the scene upload + BLAS build entirely. Cleared
    /// by drop_scene_tracers (the scene was edited) and by the init-failure
    /// eviction arm (a session with no live tracer must not strand ~9 GB
    /// under the CPU renderer).
    scene_gpu: Option<std::rc::Rc<trace::SceneGpu>>,
    /// The GPU-resident tracer (--gpu): quadtree + shading in compute with
    /// RayQuery rays. Lives on whatever device the queue runs on (v1 forces
    /// the native pipeline — main.rs disables DLSS/XeSS/OIDN under --gpu).
    trace: Option<trace::TraceGpu>,
    /// The DXR DispatchRays pipeline (the F key): lazily built on first
    /// enable, window-res, plain presentation via SRV_SLOT_DXR.
    dxr: Option<dxr::DxrGpu>,
    rr: Option<rr::RrResources>,
    /// XeSS-SR (native pipeline only; never coexists with `sl`). Explicitly
    /// torn down by GpuContext::drop after a queue drain — xessDestroyContext
    /// requires completed command lists and a live device.
    xess: Option<XessState>,
    /// FSR. Normally the session's ONE wired flavor (FSR4-RR or FSR 3.1); under
    /// --quinlight it is the FSR4-RR flavor and `fsr3` holds 3.1 alongside it.
    /// Teardown discipline: ffxDestroyContext needs completed lists and a live
    /// device, so GpuContext::drop drains the queue and drops this explicitly.
    fsr: Option<FsrState>,
    /// --quinlight only: the FSR 3.1 upscale-only context, live ALONGSIDE
    /// `fsr` (FSR4-RR). They are two contexts of the same ffx effect at
    /// different provider versions — independent, so both can be created.
    fsr3: Option<FsrState>,
    /// --quinlight: the fuse pass over every wired engine's output. It AddRefs
    /// the engine output textures it reads, so it may drop in any order
    /// relative to them; GpuContext::drop still tears it down explicitly after
    /// the queue drain, like every other GPU-resource field.
    quin: Option<quin::Quin>,
    /// The --quinlight session config, kept because the fuse is built lazily
    /// from init_trace/init_dxr (which hold the DXC its PSO needs) and those do
    /// not take a GpuOptions. `(anchor, debug)`; None = not a quinlight session.
    quin_cfg: Option<(Option<u32>, bool)>,
    /// GPU-resident NPPD (`--gpu --nppd`, XeSS sessions only): ONNX Runtime
    /// executing on `d3d.queue` with the tracer's NppdRes buffers bound as
    /// tensors. Dropped before `trace`'s resources it wraps is fine — the
    /// wrap AddRefs, and onnxruntime.dll is never unloaded.
    nppd_gpu: Option<crate::nppd::NppdGpu>,
    /// Whether the recurrent state carries history (false forces the next
    /// NPPD frame to zero the warped-state input — a reset).
    nppd_state_valid: bool,
    /// The pre-upscale denoiser slot (`--nrd` | `--frd`, XeSS/FSR3
    /// sessions): NRD's DLL-served passes or FRD's own kernels between the
    /// SAME bridge kernels (nrd_bridge.hlsl). One enum, one slot — the CLI
    /// exclusivity is structural here. Field/latch names keep the nrd_
    /// prefix during coexistence (the plan's Phase-E rename).
    nrd_gpu: Option<DnGpu>,
    /// The nppd_state_valid pattern: false → the next NRD frame passes
    /// AccumulationMode::RESTART (set on gpu_reset, never on motion).
    nrd_hist_valid: bool,
    /// Previous frame's matrices + jitter for CommonSettings (NRD wants the
    /// prev pair explicitly; its own contract, like dlss_prev/xess_prev).
    nrd_prev: Option<(crate::dlss::CamMatrices, (f32, f32))>,
    /// NRD's consecutively-growing frame index (its contract: +1 per FRAME,
    /// restartable after a non-CONTINUE mode).
    nrd_frame_idx: u32,
    /// The NRD shed, split in two because D3D12 command lists don't refcount:
    /// a failing frame only FLAGS the shed here (NrdGpu's heaps/PSOs/pools
    /// are still referenced by up to FRAMES_IN_FLIGHT submitted lists — and,
    /// on a mid-record failure, by the very list about to Present — so an
    /// immediate drop executes GPU work against destroyed objects: device
    /// removal). `nrd_shed_cleanup` at the next presenter entry drains and
    /// frees. Stays true afterwards as the session tombstone — NRD is never
    /// rebuilt this session (the nppd-gpu shape; `arm_nrd_for` refuses).
    nrd_shed: bool,
    /// Frame generation (native sessions; see FgState). Declared AFTER `d3d`
    /// deliberately: d3d's swapchain/backbuffer refs on the FI proxy must
    /// release before FgState's drop destroys the swapchain context.
    fg: Option<FgState>,
    /// XeSS-FG frame generation (Intel XeSS sessions; see XefgState).
    /// Declared AFTER `d3d` like `fg`: destroying the xefg context tears the
    /// FI proxy down, so d3d's proxy refs must release first.
    fg_x: Option<XefgState>,
    /// Raw-NGX DLSS-G (DLSS sessions with the NDA SDK built in; see
    /// NgxFgState). The NGX handle is destroyed explicitly in Drop /
    /// resize_output after a queue drain — no COM field-order constraint.
    fg_n: Option<NgxFgState>,
    /// The created DLSSD feature (allocation dims = the DRS range max;
    /// per-frame subrects carry the real render res). Destroyed explicitly
    /// in Drop / recreated across resize with the queue drained. Declared
    /// BEFORE `ngxrr` so plain field-order drop is a correct backstop: the
    /// feature must release before the session's refcounted NGX shutdown.
    rr_feature: Option<ngxrr::RrFeature>,
    /// The raw-NGX DLSS-RR session (Some = the chain's DLSS level is live —
    /// the availability probe passed on this adapter). Replaces the retired
    /// Streamline context; holds its own device clone. The refcounted NGX
    /// shutdown runs on drop — see `rr_feature`'s ordering note above.
    ngxrr: Option<ngxrr::NgxRr>,
    pub adapter_name: String,
    /// What the picked adapter IS (not what `--prefer-*` asked for) — the input
    /// to `main::vendor_defaults`.
    pub adapter_vendor: adapter::Vendor,
    /// The presentation curve every present arm reads (see
    /// `fullscreen_to_backbuffer`). One field, so a display change is a retune.
    tone: crate::tone::ToneParams,
    /// The adapter and window, kept so the display can be RE-probed when the
    /// window moves to another monitor — the probe is not a one-time startup
    /// fact. `None` display = never probed (the SDR path never does).
    adapter: windows::Win32::Graphics::Dxgi::IDXGIAdapter4,
    hwnd: windows::Win32::Foundation::HWND,
    display: Option<display::DisplayHdr>,
}

/// The per-frame raw-NGX RR evaluate shared by the CPU-fed (`present_rr`) and
/// GPU-fed (`present_trace_rr`/`present_dxr_rr`/quin) paths — the retired
/// `rr_sl_sequence`'s successor, between the same output PSR->UAV->PSR
/// barriers the callers keep. The world<->view matrices ride EVERY evaluate
/// (they feed the SpecularHitDistance path — the reason the SL path re-sent
/// options per frame); jitter polarity and mv_scale come lever-resolved off
/// the session (`ngxrr::NgxRr` — the ONE place those conventions live);
/// `{fc.rw, fc.rh}` is the DRS subrect (the sl::Extent replacement).
fn rr_ngx_sequence(
    nx: &ngxrr::NgxRr,
    feat: &ngxrr::RrFeature,
    rr: &rr::RrResources,
    list: &ID3D12GraphicsCommandList,
    fc: &dlss::FrameConstants,
) -> Result<()> {
    let _ev = pix::scope(list, c"rr-eval");
    let p = rr.plane_resources();
    let d = ngxrr::FrDlssdDispatch {
        cmdlist: list.as_raw(),
        color: p[0].0.as_raw(),
        output: rr.output.as_raw(),
        depth: p[2].0.as_raw(),
        motion: p[3].0.as_raw(),
        diff_albedo: p[4].0.as_raw(),
        spec_albedo: p[5].0.as_raw(),
        normal_rough: p[1].0.as_raw(),
        spec_hit: p[6].0.as_raw(),
        world_to_view: row_major(&fc.world_to_view),
        view_to_clip: row_major(&fc.view_to_clip),
        jitter: [nx.jitter_mul * fc.jitter.0, nx.jitter_mul * fc.jitter.1],
        mv_scale: nx.mv_scale(fc.rw as u32, fc.rh as u32),
        rend_w: fc.rw as u32,
        rend_h: fc.rh as u32,
        reset: fc.reset as i32,
        frame_time_ms: 0.0, // unset — the helper defaults it
    };
    feat.evaluate(nx, &d)
}

/// Register a texture the tonemap can present FROM: its SRV goes into the
/// tonemap's heap (for the draw) and into the glare pyramid's heap (for the
/// compute read), in the matching slot of each.
///
/// The two are paired here, in one call, deliberately. Only ONE CBV_SRV_UAV heap
/// may be bound at a time, so bloom cannot reach into the tonemap's heap — and it
/// cannot copy the descriptor across at present time either, because
/// `CopyDescriptors` may not READ from a shader-visible heap. Each source
/// therefore needs its descriptor created in BOTH heaps, and a source that got
/// one but not the other would present fine and simply lose its glare — a silent
/// failure. Route every tonemap source through here and it can't happen.
///
/// (`SRV_SLOT_BLOOM` is bloom's OUTPUT into the tonemap heap and is not a source;
/// it stays a plain `Passes::create_srv`.)
fn wire_tonemap_src(
    device: &ID3D12Device,
    passes: &tonemap::Passes,
    bloom: &bloom::BloomGpu,
    aexp: &autoexp::AutoExpGpu,
    res: &ID3D12Resource,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    slot: u32,
) {
    passes.create_srv(device, res, format, slot);
    bloom.create_source_srv(device, res, format, slot);
    // The auto-exposure meter reads the same sources from ITS heap — the same
    // one-heap-at-a-time argument as bloom's, and the same silent-failure mode
    // if a source got the draw SRV but not this one (it would present fine and
    // simply never meter, i.e. exposure would freeze at 1.0 in that arm).
    aexp.create_source_srv(device, res, format, slot);
}

/// Build the frame-generation UI target (`hud::HudFi`) for a wrapper-FG
/// session — the display-space premultiplied HUD render the FI/xefg proxies
/// composite AFTER interpolation. ONE builder for boot AND resize (the two
/// open-coded copies drifted apart once already). Format per present space:
/// RGBA16F under HDR10 (8-bit PQ bands, and the backbuffer's R10G10B10A2 has
/// 2-bit alpha — unusable as a UI blend source), RGBA8 under SDR/Sdr10
/// (hud.hlsl mode 1 is a passthrough bit-copy of the Slint buffer). Creation
/// failure is one loud line + the baked-HUD fallback, never a session
/// failure.
fn make_fg_ui(
    device: &ID3D12Device,
    passes: &tonemap::Passes,
    hud: &hud::HudGpu,
    space: d3d12::PresentSpace,
    w: u32,
    h: u32,
) -> Option<hud::HudFi> {
    use windows::Win32::Graphics::Dxgi::Common::*;
    let fmt = match space {
        d3d12::PresentSpace::Hdr10 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => DXGI_FORMAT_R8G8B8A8_UNORM,
    };
    match hud::HudFi::new(device, passes, hud, fmt, w, h) {
        Ok(f) => Some(f),
        Err(e) => {
            eprintln!("fg: ui pre-pass target creation failed ({e}) — HUD stays baked pre-present");
            None
        }
    }
}

impl GpuContext {
    pub fn new(hwnd: HWND, w: u32, h: u32, opts: &GpuOptions) -> Result<Self> {
        let factory =
            adapter::create_factory(opts.debug).map_err(|e| format!("CreateDXGIFactory2: {e}"))?;
        let prefer = opts.prefer.unwrap_or(adapter::Prefer::Nvidia);
        let pick = adapter::pick(&factory, prefer)?;
        eprintln!("gpu: using adapter \"{}\"", pick.name);
        let device = d3d12::create_device(&pick.adapter, opts.debug)?;

        // Chain level 1 (DLSS-RR), via raw NGX. DLSS is the top of the chain
        // by POLICY only now — the retired Streamline interposer's
        // init-before-any-DXGI-factory constraint died with it, so the probe
        // runs on the ordinary device like every native level's. Both DLSS
        // features (RR + frame generation) ride one build gate: without the
        // DLSS SDK there is NO DLSS at all and the chain falls through.
        let mut ngxrr = if !opts.chain.dlss {
            None
        } else if pick.vendor != adapter::Vendor::Nvidia {
            eprintln!("dlss: level unavailable (no NVIDIA adapter) — falling through the chain");
            None
        } else if !ngxrr::BUILT {
            eprintln!(
                "dlss: built without the DLSS SDK — set FRUSTRACER_DLSS_SDK and rebuild \
                 (chain falls to FSR4/XeSS/FSR3)"
            );
            None
        } else {
            match ngxrr::NgxRr::open(&device) {
                Ok(nx) => {
                    eprintln!("dlss: raw-NGX Ray Reconstruction available");
                    Some(nx)
                }
                Err(e) => {
                    eprintln!("dlss: level unavailable ({e}) — falling through the chain");
                    None
                }
            }
        };
        // Resolve the native upscaler chain before choosing a frame-generation
        // swapchain. In particular, XeSS-FG must not force HDR10/SDR merely
        // because XeSS was requested: the XeSS level itself has to initialize
        // and win the chain first. Descriptor wiring waits until the
        // format-dependent tonemap/bloom heaps exist below.
        let (xess_state, fsr_state, fsr3_state) =
            Self::probe_native(&device, opts, w, h, ngxrr.is_some());

        // Frame generation, Intel family (the XeSS-FG swapchain, XeLL
        // linked): taken when the chain is headed for the XeSS level on an
        // Intel adapter — the family follows the upscaler. The wrap happens
        // inside D3d::with_queue like the ffx one; a failed wrap/init is a
        // loud line + a normal session.
        let try_xefg = opts.fg
            && ngxrr.is_none()
            && pick.vendor == adapter::Vendor::Intel
            && xess_state.is_some();

        // ONE 10-bit format (R10G10B10A2, 4 B/px), two curves: PQ on an
        // HDR-ON display, gamma-2.2 (Sdr10) everywhere else — the probe
        // decides, `--hdr10`/`--no-hdr10` force an arm, `--no-hdr` keeps the
        // legacy 8-bit chain. The old scRGB fp16 chain (8 B/px) is GONE: the
        // present is the whole frame budget whenever the display hangs off a
        // different GPU than the renderer — DWM must then COPY every frame
        // across, which at 7680x3969 is 244 MB at fp16 vs 122 MB at 10-bit.
        // Measured on this box (world, 8K, display on an Intel B70 while a
        // 4090 renders): 6.1 -> 10.0 rendered fps, ~80 -> ~51 ms per present,
        // while the GPU itself was only doing 14.7 ms of work per frame.
        // 10-bit gamma keeps the deep-colour quality f16 bought on SDR panels
        // (the 8-bit backbuffer banded), so nothing is traded away there.
        //
        // The wrapper-FG families need no special-casing any more: XeSS-FG
        // rejected scRGB fp16 outright (measured: InitFromSwapChain returns
        // INVALID_ARGUMENT on R16G16B16A16_FLOAT), which used to force those
        // sessions onto PQ-or-8-bit; both defaults are now R10G10B10A2, the
        // format it verifiably wraps. If a wrap still refuses,
        // D3d::with_queue rebuilds at 8-bit and wraps again — FG is why the
        // session exists.
        //
        // The probe runs ONCE: it needs the HWND (which exists — the swapchain
        // is created for it).
        let hdr_on =
            opts.hdr && !opts.hdr10 && !opts.sdr10 && display::probe(&pick.adapter, hwnd).enabled;
        let want = if !opts.hdr {
            d3d12::PresentSpace::Sdr
        } else if opts.hdr10 {
            d3d12::PresentSpace::Hdr10
        } else if opts.sdr10 {
            d3d12::PresentSpace::Sdr10
        } else if hdr_on {
            d3d12::PresentSpace::Hdr10
        } else {
            d3d12::PresentSpace::Sdr10
        };
        if want == d3d12::PresentSpace::Hdr10 && !opts.hdr10 {
            eprintln!(
                "present: HDR display — defaulting to 10-bit PQ (--no-hdr10 forces 10-bit \
                 gamma-2.2 SDR, --no-hdr forces 8-bit SDR)"
            );
        }

        // Frame generation, native family (the ffx frame-interpolation
        // swapchain). Decided BEFORE the swapchain exists because the FI
        // proxy owns the backbuffers the session renders into — the wrap
        // happens inside D3d::with_queue, between colour-space negotiation
        // and RTV creation. Arm only when a device-filtered FG provider
        // version actually enumerates; every failure is a loud line + a
        // normal session (the chain-fallback shape).
        let mut fg_versions: Vec<ffx::Version> = Vec::new();
        if opts.fg && ngxrr.is_none() && !try_xefg {
            match ffx::fg_load(&opts.ffx_dir, &opts.fg_dir) {
                Ok(()) => match ffx::fg_versions(&device) {
                    Ok(v) if !v.is_empty() => {
                        for (id, name) in &v {
                            eprintln!("fg: provider version 0x{id:x} \"{name}\"");
                        }
                        fg_versions = v;
                    }
                    Ok(_) => eprintln!(
                        "fg: no framegeneration provider supports this adapter — frame generation off"
                    ),
                    Err(e) => eprintln!("fg: provider enumeration failed ({e}) — frame generation off"),
                },
                Err(e) => eprintln!("fg: {e} — frame generation off"),
            }
        }

        // RefCells, not `let mut`: the wrap closure writes the context slot
        // while the surrounding scope still holds the binding. `into_inner`
        // right after the d3d block.
        let fg_sc: std::cell::RefCell<Option<ffx::FgSwapchain>> = std::cell::RefCell::new(None);
        let fg_xefg: std::cell::RefCell<Option<crate::xess_fg::XefgSwapchain>> =
            std::cell::RefCell::new(None);
        // Pair-present arms only in raw-NGX FG sessions — extra backbuffers
        // so a buffer never comes back around while still queued for scanout
        // (see d3d12::PAIR_BACKBUFFERS). Only the plain arm below can carry
        // it: an NGX-FG session is a DLSS session, and a DLSS session never
        // takes a swapchain wrapper (try_xefg and the ffx enumeration both
        // require ngxrr.is_none()).
        let pair = opts.fg && ngxfg::BUILT && ngxrr.is_some();
        let d3d = {
            let queue = d3d12::create_queue(&device)?;
            if try_xefg {
                // The XeFG wrap needs the DEVICE at hook time (context
                // creation); `device` is alive inside with_queue while the
                // hook runs, so the raw pointer captured here stays valid.
                let dev_raw = device.as_raw();
                let xess_dir = opts.xess_dir.clone();
                let mut wrap = |q: &ID3D12CommandQueue,
                                sc: windows::Win32::Graphics::Dxgi::IDXGISwapChain3|
                 -> std::result::Result<
                    windows::Win32::Graphics::Dxgi::IDXGISwapChain3,
                    (windows::Win32::Graphics::Dxgi::IDXGISwapChain3, String),
                > {
                    let raw = sc.into_raw();
                    match crate::xess_fg::XefgSwapchain::wrap(&xess_dir, dev_raw, q.as_raw(), raw)
                    {
                        Ok((ctx, proxy)) => {
                            eprintln!("fg: swapchain wrapped by the XeSS-FG proxy (XeLL linked)");
                            *fg_xefg.borrow_mut() = Some(ctx);
                            Ok(unsafe {
                                windows::Win32::Graphics::Dxgi::IDXGISwapChain3::from_raw(proxy)
                            })
                        }
                        Err(e) => Err((
                            unsafe {
                                windows::Win32::Graphics::Dxgi::IDXGISwapChain3::from_raw(raw)
                            },
                            e,
                        )),
                    }
                };
                D3d::with_queue(
                    &factory, device, queue, hwnd, w, h, opts.vsync, want,
                    Some(d3d12::FgHook { wrap: &mut wrap }),
                    false,
                )?
            } else if fg_versions.is_empty() {
                D3d::with_queue(&factory, device, queue, hwnd, w, h, opts.vsync, want, None, pair)?
            } else {
                let mut wrap = |q: &ID3D12CommandQueue,
                                sc: windows::Win32::Graphics::Dxgi::IDXGISwapChain3|
                 -> std::result::Result<
                    windows::Win32::Graphics::Dxgi::IDXGISwapChain3,
                    (windows::Win32::Graphics::Dxgi::IDXGISwapChain3, String),
                > {
                    let (proxy, sc_ctx) = ffx::fg_wrap_swapchain(q, sc)?;
                    eprintln!("fg: swapchain wrapped by the FidelityFX frame-interpolation proxy");
                    *fg_sc.borrow_mut() = Some(sc_ctx);
                    Ok(proxy)
                };
                D3d::with_queue(
                    &factory, device, queue, hwnd, w, h, opts.vsync, want,
                    Some(d3d12::FgHook { wrap: &mut wrap }),
                    false,
                )?
            }
        };
        let fg_sc = fg_sc.into_inner();
        let fg_xefg = fg_xefg.into_inner();

        // The display probe and the curve it implies. Only the PQ arm has a
        // curve to aim (peak/paper-white) — the Sdr and Sdr10 arms are
        // display-encoded gamma with one static curve (ToneParams::SDR) and
        // nothing to ask the monitor.
        let disp = if d3d.space == d3d12::PresentSpace::Hdr10 {
            Some(display::probe(&pick.adapter, hwnd))
        } else {
            None
        };
        let tone = match disp {
            Some(d) => {
                let t = d.tone_pq(opts.paper_white, opts.peak_nits);
                // Report the curve we ACTUALLY installed, not the probe's opinion:
                // --hdr-peak overrides the probe (including an "HDR off" verdict),
                // so keying the message off `d.enabled` could announce SDR levels
                // while running an HDR rolloff.
                if t.headroom > 1.0 {
                    eprintln!(
                        "hdr: display peak {:.0} nits (full-frame {:.0}){}, paper white {:.0} \
                         -> headroom {:.1}x",
                        t.peak_nits(),
                        d.max_full_frame_nits,
                        if opts.peak_nits.is_some() { " [--hdr-peak override]" } else { "" },
                        opts.paper_white,
                        t.headroom
                    );
                } else {
                    // Not a failure — a --hdr10 session on an HDR-off output:
                    // the degenerate SDR rolloff, PQ-encoded at paper white.
                    eprintln!(
                        "hdr: display reports HDR off — SDR levels through the PQ encode \
                         (enable Windows HDR on this monitor for highlights)"
                    );
                }
                t
            }
            None => crate::tone::ToneParams::SDR,
        };

        let (rr_opt, rr_min, rr_max) = Self::query_rr_res(ngxrr.as_ref(), w, h);

        let passes = tonemap::Passes::new(&d3d.device, d3d.format)?;
        let bloom = bloom::BloomGpu::new(&d3d.device, w as u32, h as u32)?;
        let aexp_gpu = autoexp::AutoExpGpu::new(&d3d.device)?;
        passes.create_srv(
            &d3d.device,
            bloom.glare_srv_source(),
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            tonemap::SRV_SLOT_BLOOM,
        );
        Self::wire_native_outputs(
            &d3d.device,
            &passes,
            &bloom,
            &aexp_gpu,
            xess_state.as_ref(),
            fsr_state.as_ref(),
        );
        // The blit texture matches the SWAPCHAIN, not the CLI flag: under the
        // 10-bit spaces the CPU arms hand over packed 10-bit u32, under SDR
        // u32 0RGB. Keyed off `d3d.space` so a refused colour space (or an FG
        // rebuild) can't leave them mismatched.
        let blit = match d3d.space {
            d3d12::PresentSpace::Sdr10 | d3d12::PresentSpace::Hdr10 => {
                upload::BlitUpload::new_10bit(&d3d, w, h)?
            }
            d3d12::PresentSpace::Sdr => upload::BlitUpload::new(&d3d, w, h)?,
        };
        let hdr = upload::HdrUpload::new(&d3d, w, h)?;
        // The HUD overlay (SRV slot 9): not a tonemap source — bloom never
        // reads it — so a plain create_srv inside HudGpu::new, never
        // wire_tonemap_src.
        let hud = hud::HudGpu::new(&d3d.device, &passes, d3d.format, w, h)?;
        // The DLSSD feature is created EAGERLY here (the SL path created
        // lazily at first evaluate) — a create failure surfaces at session
        // start, the better place; the transient-queue warm-up is a one-time
        // ~100 ms. Allocation dims = the DRS range max; DLAA when the
        // optimal-settings query degenerated (opt == min == max == output).
        let mut rr_feature: Option<ngxrr::RrFeature> = None;
        let rr_res = if let Some(nx) = &ngxrr {
            let dlaa = rr_opt == (w, h) && rr_min == rr_opt && rr_max == rr_opt;
            match nx.create_feature(rr_max, (w, h), dlaa) {
                Ok(f) => {
                    rr_feature = Some(f);
                    let r = rr::RrResources::new(&d3d.device, rr_opt, rr_min, rr_max, w, h)?;
                    wire_tonemap_src(
                        &d3d.device,
                        &passes,
                        &bloom,
                        &aexp_gpu,
                        &r.output,
                        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                        tonemap::SRV_SLOT_RR,
                    );
                    Some(r)
                }
                Err(e) => {
                    eprintln!("dlss: {e} — level dropped (plain presentation unless a native level wired)");
                    None
                }
            }
        } else {
            None
        };
        if rr_res.is_none() {
            // A dead feature must not leave dlss_ready() half-true anywhere.
            ngxrr = None;
        }
        // The blit arm takes the blit PSO, which `fullscreen_to_backbuffer`
        // never blooms (its source is an already-encoded, CPU-tonemapped image —
        // the CPU applied glare in `render::resolve`). So it gets a plain
        // tonemap SRV and no bloom source: nothing would ever read one.
        passes.create_srv(&d3d.device, &blit.texture, blit.format, tonemap::SRV_SLOT_BLIT);
        wire_tonemap_src(
            &d3d.device,
            &passes,
            &bloom,
            &aexp_gpu,
            &hdr.texture,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            tonemap::SRV_SLOT_HDR,
        );

        // Raw-NGX DLSS-G (preferred when built): the interpolated-frame
        // target + lazy-created feature. The swapchain format is irrelevant
        // on this path — there is no swapchain policing because there is no
        // swapchain hook (NGX sees only internal fp16 textures).
        let fg_n = if opts.fg && ngxfg::BUILT && rr_res.is_some() {
            match d3d12::committed_tex(
                &d3d.device,
                w,
                h,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ) {
                Ok(out) => {
                    wire_tonemap_src(
                        &d3d.device,
                        &passes,
                        &bloom,
                        &aexp_gpu,
                        &out,
                        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                        tonemap::SRV_SLOT_NGXFG,
                    );
                    eprintln!(
                        "fg: raw-NGX DLSS-G armed (no Streamline pacer — pair-present; feature \
                         created on the first frame)"
                    );
                    // Empirical-settling levers (env vars, the FR_ABL
                    // idiom — quinlight settled the snippet's conventions
                    // against zero MVs / zero jitter / [0,1] luma-depth, so
                    // the motion-dependent ones are settled HERE instead).
                    // Only a departure from the default prints a line — but
                    // an UNRECOGNIZED value is loud too and takes the
                    // default: a silently no-op'd A/B walk is the exact
                    // failure mode these levers exist to prevent. Values are
                    // matched case-insensitively. Returns 0 for unset /
                    // unrecognized, i+1 for legal[i].
                    let lever = |k: &str, legal: &[&str]| -> u8 {
                        let Ok(s) = std::env::var(k) else { return 0 };
                        let s = s.to_ascii_lowercase();
                        match legal.iter().position(|v| *v == s) {
                            Some(i) => i as u8 + 1,
                            None => {
                                eprintln!(
                                    "fg: {k}={s} unrecognized (legal: {}) — using the default",
                                    legal.join("|")
                                );
                                0
                            }
                        }
                    };
                    let depth_linear = lever("FR_NGXFG_DEPTH", &["linear"]) == 1;
                    if depth_linear {
                        eprintln!(
                            "fg: FR_NGXFG_DEPTH=linear — raw linear view-Z to the NGX \
                             evaluate (the round-1 A/B arm)"
                        );
                    }
                    let rmv_off = lever("FR_NGXFG_RMV", &["off"]) == 1;
                    if rmv_off {
                        eprintln!(
                            "fg: FR_NGXFG_RMV=off — raw surface MVs to the NGX evaluate \
                             (expect reflections to swim under motion)"
                        );
                    }
                    let guides = if depth_linear && rmv_off {
                        None
                    } else {
                        match ngxfg_guides::GuidePass::new(&d3d.device) {
                            Ok(p) => Some(std::cell::RefCell::new(p)),
                            Err(e) => {
                                eprintln!(
                                    "fg: guide-conversion pass creation failed ({e}) — \
                                     falling back to the raw RR planes (expect reflection \
                                     shimmer)"
                                );
                                None
                            }
                        }
                    };
                    let jitter_mode = lever("FR_NGXFG_JITTER", &["0", "neg"]);
                    match jitter_mode {
                        1 => eprintln!("fg: FR_NGXFG_JITTER=0 — zero jitter to the evaluate"),
                        2 => eprintln!(
                            "fg: FR_NGXFG_JITTER=neg — SL-negated jitter (the pre-2026-07-26 \
                             default; expect specular highlights to strobe)"
                        ),
                        _ => {}
                    }
                    let mv_mode = lever("FR_NGXFG_MV", &["norm", "neg", "normneg"]);
                    match mv_mode {
                        1 => eprintln!(
                            "fg: FR_NGXFG_MV=norm — {{1/rw,1/rh}} mvecScale (round-1 arm)"
                        ),
                        2 => eprintln!("fg: FR_NGXFG_MV=neg — negated pixel mvecScale"),
                        3 => eprintln!("fg: FR_NGXFG_MV=normneg — negated {{1/rw,1/rh}} mvecScale"),
                        _ => {}
                    }
                    let show_mode = lever("FR_NGXFG_SHOW", &["interp", "real"]);
                    match show_mode {
                        1 => eprintln!(
                            "fg: FR_NGXFG_SHOW=interp — presenting the interpolated \
                             frame for BOTH halves of each pair (non-generating frames \
                             fall back to the real frame)"
                        ),
                        2 => eprintln!(
                            "fg: FR_NGXFG_SHOW=real — presenting the REAL frame for \
                             BOTH halves (evaluate still runs; artifacts that survive \
                             this are in the present path, not the generated frames)"
                        ),
                        _ => {}
                    }
                    let cam_identity = lever("FR_NGXFG_CAM", &["identity"]) == 1;
                    if cam_identity {
                        eprintln!(
                            "fg: FR_NGXFG_CAM=identity — quinlight's identity camera block \
                             to the evaluate (combine with FR_NGXFG_DEPTH=linear for the \
                             faithful quinlight reproduction — its depth was never \
                             matrix-consistent either)"
                        );
                    }
                    let mat_col = lever("FR_NGXFG_MAT", &["col"]) == 1;
                    if mat_col {
                        eprintln!("fg: FR_NGXFG_MAT=col — column-major matrices to the evaluate");
                    }
                    let ffmv_off = lever("FR_NGXFG_FFMV", &["off"]) == 1;
                    if ffmv_off {
                        eprintln!(
                            "fg: FR_NGXFG_FFMV=off — firefly glow keeps surface MVs on \
                             generated frames (expect the night swarm to strobe under motion)"
                        );
                    }
                    let ripplemv_off = lever("FR_NGXFG_RIPPLEMV", &["off"]) == 1;
                    if ripplemv_off {
                        eprintln!(
                            "fg: FR_NGXFG_RIPPLEMV=off — water keeps a still-mirror unfold on \
                             generated frames (expect rippling reflections to strobe)"
                        );
                    }
                    let ripdt_off = lever("FR_NGXFG_RIPPLEDT", &["off"]) == 1;
                    if ripdt_off {
                        eprintln!(
                            "fg: FR_NGXFG_RIPPLEDT=off — round 4's large-dt confidence fade \
                             disarmed (unfaded reconstruction at any clock delta; expect the \
                             low-framerate water glitch back on demand)"
                        );
                    }
                    // FR_NGXFG_TONEMAP — which curve shapes the color handed
                    // to NGX. DEFAULT reinhard (2026-07-31): the flow
                    // estimator needs a display-curve-shaped input, and
                    // reinhard is the measured-correct arm parked AND rotating
                    // (see ngxfg_guides::TonePass for the elimination record).
                    // `off` is the kill arm — raw linear rr.output, the
                    // sun-disc strobe returns on demand; `scale`/`log` stay as
                    // the diagnostic arms. Parsed by hand rather than through
                    // `lever` above: that closure's 0 means
                    // unset-or-unrecognized, which cannot express a nonzero
                    // default. The numeric modes are pinned to the HLSL
                    // (1 = scale, 3 = log, else reinhard; 0 never reaches the
                    // shader — `tone` is None).
                    let tone_mode: u8 = match std::env::var("FR_NGXFG_TONEMAP") {
                        Err(_) => 2,
                        Ok(s) => match s.to_ascii_lowercase().as_str() {
                            "off" => 0,
                            "scale" => 1,
                            "reinhard" => 2,
                            "log" => 3,
                            other => {
                                eprintln!(
                                    "fg: FR_NGXFG_TONEMAP={other} unrecognized (legal: \
                                     off|scale|reinhard|log) — using the default (reinhard)"
                                );
                                2
                            }
                        },
                    };
                    let tone = if tone_mode == 0 {
                        eprintln!(
                            "fg: FR_NGXFG_TONEMAP=off — raw linear radiance to the NGX \
                             evaluate (expect the sun disc to strobe on generated frames)"
                        );
                        None
                    } else {
                        match tone_mode {
                            1 => eprintln!(
                                "fg: FR_NGXFG_TONEMAP=scale — linear range scale to the \
                                 evaluate (expect the disc to double-ghost under rotation)"
                            ),
                            3 => eprintln!(
                                "fg: FR_NGXFG_TONEMAP=log — log compression to the \
                                 evaluate (expect ghosting + banding)"
                            ),
                            _ => {} // reinhard — the default, silent
                        }
                        match ngxfg_guides::TonePass::new(&d3d.device) {
                            Ok(t) => Some(std::cell::RefCell::new(t)),
                            Err(e) => {
                                eprintln!(
                                    "fg: tone pass creation failed ({e}) — raw linear \
                                     radiance to the evaluate (expect the sun disc to \
                                     strobe on generated frames)"
                                );
                                None
                            }
                        }
                    };
                    Some(NgxFgState {
                        out,
                        handle: std::cell::Cell::new(std::ptr::null_mut()),
                        dims: std::cell::Cell::new((0, 0)),
                        enabled: true,
                        failed: std::cell::Cell::new(false),
                        primed: std::cell::Cell::new(false),
                        frame_id: std::cell::Cell::new(0),
                        res_pend: std::cell::Cell::new(((0, 0), 0)),
                        pair: std::cell::Cell::new(false),
                        guides,
                        guides_failed: std::cell::Cell::new(false),
                        depth_linear,
                        rmv_off,
                        jitter_mode,
                        mv_mode,
                        show_mode,
                        cam_identity,
                        mat_col,
                        ffmv_off,
                        prev_ff: std::cell::Cell::new(crate::fireflies::Fireflies::off()),
                        ripplemv_off,
                        ripdt_off,
                        prev_clock: std::cell::Cell::new(None),
                        tone_mode,
                        tone,
                    })
                }
                Err(e) => {
                    eprintln!("fg: interpolated-frame target creation failed ({e}) — FG off");
                    None
                }
            }
        } else {
            None
        };

        // Frame generation, part 2: the display-size FG effect context. The
        // provider pick keys on the session's resolved upscaler family (an
        // FSR4-RR session prefers the 4.x ML frame generation; everything
        // else the 3.1 interpolation), which is only known after probe_native
        // — the swapchain wrap above deliberately did not need it. A create
        // failure leaves the wrapped proxy presenting passthrough: correct,
        // just generation-free.
        let fg_state = fg_sc.map(|sc| {
            let fsr4_session =
                fsr_state.as_ref().is_some_and(|f| f.res.flavor() == crate::fsr::Flavor::Fsr4Rr);
            let version = if fsr_state.is_some() {
                crate::fsr::pick_fg_version(&fg_versions, fsr4_session)
            } else {
                // The wrap happened before the chain resolved; only the FSR
                // arms carry the prepare today. A session that wired XeSS (or
                // nothing) presents through the proxy passthrough — harmless,
                // one line.
                eprintln!(
                    "fg: wired upscaler has no ffx frame-generation pairing yet — \
                     use --fsr3/--fsr for frame generation today"
                );
                None
            };
            let ctx = version.as_ref().and_then(|(id, name)| {
                match ffx::FgContext::create(
                    &d3d.device,
                    (w, h),
                    (w, h),
                    match d3d.space {
                        d3d12::PresentSpace::Sdr10 | d3d12::PresentSpace::Hdr10 => {
                            ffx_sys::SURFACE_FORMAT_R10G10B10A2_UNORM
                        }
                        d3d12::PresentSpace::Sdr => ffx_sys::SURFACE_FORMAT_B8G8R8A8_UNORM,
                    },
                    // FG_HDR = "the backbuffer is high dynamic range" — Hdr10
                    // ONLY: Sdr10 is a gamma-encoded SDR buffer that happens
                    // to be 10 bits wide. The FI swapchain reads the transfer
                    // function off the chain's own declared colour space, so
                    // PQ needs no extra plumbing (undeclared = sRGB/G22).
                    d3d.space == d3d12::PresentSpace::Hdr10,
                    opts.debug,
                    *id,
                ) {
                    Ok(ctx) => {
                        eprintln!("fg: frame generation live — provider \"{name}\" (1 generated frame per rendered frame)");
                        Some(ctx)
                    }
                    Err(e) => {
                        eprintln!("fg: {e} — swapchain proxy presents passthrough");
                        None
                    }
                }
            });
            if version.is_none() {
                eprintln!("fg: no provider version parseable — swapchain proxy presents passthrough");
            }
            // Only the PQ arm has real nits to report — Sdr10 is gamma SDR
            // (and its ToneParams has no meaningful peak).
            let luminance = if d3d.space == d3d12::PresentSpace::Hdr10 {
                [0.0, tone.peak_nits()]
            } else {
                [0.0, 0.0]
            };
            FgState {
                sc,
                ctx,
                enabled: true,
                live: std::cell::Cell::new(false),
                prepared: std::cell::Cell::new(false),
                frame_id: std::cell::Cell::new(0),
                frame_ms: std::cell::Cell::new(16.6),
                luminance,
                version: version.unwrap_or((0, String::new())),
                trace: std::env::var("FR_FG_TRACE").ok().as_deref() == Some("1"),
                last_res: std::cell::Cell::new((0, 0)),
                pauses: std::cell::Cell::new(0),
                skip_prepare: std::cell::Cell::new(false),
                ui_reg: std::cell::Cell::new(false),
                ui_shed: std::cell::Cell::new(false),
            }
        });

        // XeSS-FG, part 2: only the XeSS arms carry the prepare, so the state
        // arms iff the XeSS level actually wired (else the proxy presents
        // passthrough — the leg-1 shape). Init-time enable, the DLSS-G
        // lesson: the funnel disables it on any unprepared present.
        let fg_x = fg_xefg.map(|sc| {
            let wired_xess = xess_state.is_some();
            if wired_xess {
                sc.set_enabled(true);
                eprintln!(
                    "fg: XeSS frame generation live — 1 generated frame per rendered frame"
                );
            } else {
                sc.set_enabled(false);
                eprintln!(
                    "fg: XeSS level did not wire — XeSS-FG proxy presents passthrough"
                );
            }
            XefgState {
                sc,
                enabled: wired_xess,
                on: std::cell::Cell::new(wired_xess),
                prepared: std::cell::Cell::new(false),
                failed: std::cell::Cell::new(!wired_xess),
                frame_id: std::cell::Cell::new(0),
                poll: std::cell::Cell::new(0),
                logged: std::cell::Cell::new(false),
                mult: std::cell::Cell::new(0),
                ui_shed: std::cell::Cell::new(false),
            }
        });

        // The frame-generation UI target: built for either wrapper-FG family
        // (the proxy composites it post-interpolation; see make_fg_ui). The
        // ffx key is the WRAPPER, not the effect ctx — resize rebuilds the
        // ctx AFTER the target, so a ctx key would leave every resized
        // session bare and the two sites drifted on exactly that. The xefg
        // key is `enabled`: a passthrough proxy (XeSS never wired) can never
        // consume the target (`xefg_hud` requires `on`).
        let fg_ui = if fg_state.is_some() || fg_x.as_ref().is_some_and(|x| x.enabled) {
            make_fg_ui(&d3d.device, &passes, &hud, d3d.space, w, h)
        } else {
            None
        };

        Ok(Self {
            // Armed by `init_trace` — it needs the scene, and a session that
            // never enters the wavefront tracer must not pay a second upload.
            dual: None,
            dual_want: opts.dual_gpu,
            dual_depth: opts.dual_depth,
            dual_auto: opts.dual_auto,
            dual_arm: opts.dual_arm,
            dual_keep: None,
            ngxrr,
            rr_feature,
            d3d,
            passes,
            bloom,
            autoexp: aexp_gpu,
            meter: std::cell::Cell::new(None),
            blit,
            hdr,
            hud,
            fg_ui,
            last_present: std::cell::Cell::new(None),
            waveviz_src: std::cell::Cell::new(WvSrc::None),
            scene_gpu: None,
            trace: None,
            dxr: None,
            rr: rr_res,
            xess: xess_state,
            fsr: fsr_state,
            fsr3: fsr3_state,
            quin: None,
            quin_cfg: opts.quin.then_some((opts.quin_anchor, opts.debug)),
            nppd_gpu: None,
            nppd_state_valid: false,
            nrd_gpu: None,
            nrd_hist_valid: false,
            nrd_prev: None,
            nrd_frame_idx: 0,
            nrd_shed: false,
            fg: fg_state,
            fg_x,
            fg_n,
            adapter_name: pick.name,
            adapter_vendor: pick.vendor,
            tone,
            adapter: pick.adapter,
            hwnd,
            display: disp,
        })
    }

    /// Bring up the native chain levels (2-4) for an output size, and create
    /// their presentation SRVs. Shared by `new` and `resize_output` — a resize
    /// re-runs exactly the probe that wired the session, so it can never end up
    /// with a different engine set than it started with.
    ///
    /// FSR4-RR and FSR 3.1 are the SAME ffx-api effect at two provider
    /// versions, probed as two chain levels; XeSS and both ffx flavors are
    /// created on the NATIVE device.
    ///
    /// Normally this is FIRST-HIT-WINS and only runs when DLSS didn't take the
    /// session. Under `--quinlight` that rule is deliberately suspended and
    /// EVERY supported level is wired, because the fuse's inputs ARE the engines
    /// (gpu/quin.rs). Two things make the coexistence legal, and neither is new:
    ///   * every context lives on the one native device — DLSS-RR is a raw-NGX
    ///     evaluate on the session queue (no interposer since the SL
    ///     retirement), so the XeSS/ffx contexts record into the same native
    ///     command list beside it.
    ///   * FSR4-RR and FSR 3.1 are independent ffx CONTEXTS, so one session can
    ///     hold both.
    /// A level that fails to come up is just not an engine: the fuse is
    /// N-generic, so --quinlight degrades to whatever actually wired.
    #[allow(clippy::type_complexity)]
    fn probe_native(
        device: &ID3D12Device,
        opts: &GpuOptions,
        w: u32,
        h: u32,
        dlss_live: bool,
    ) -> (Option<XessState>, Option<FsrState>, Option<FsrState>) {
        let (mut xess, mut fsr, mut fsr3) = (None, None, None);
        let all = opts.quin;
        if dlss_live && !all {
            return (xess, fsr, fsr3);
        }
        if opts.chain.fsr4 {
            match Self::init_fsr(
                &opts.ffx_dir,
                device,
                w,
                h,
                opts.debug,
                crate::fsr::Flavor::Fsr4Rr,
                &opts.fsr_tune,
            ) {
                Ok(s) => fsr = Some(s),
                Err(e) => eprintln!("fsr4: level unavailable ({e}) — falling through the chain"),
            }
        }
        if (all || fsr.is_none()) && opts.chain.xess {
            // Input planes are allocated once at the range MAX; every frame
            // uploads and names its own sub-rect (dynamic res).
            match Self::init_xess(&opts.xess_dir, device, w, h, opts.xess_autoexposure) {
                Ok(s) => xess = Some(s),
                Err(e) => eprintln!("xess: level unavailable ({e}) — falling through the chain"),
            }
        }
        if (all || (fsr.is_none() && xess.is_none())) && opts.chain.fsr3 {
            match Self::init_fsr(
                &opts.ffx_dir,
                device,
                w,
                h,
                opts.debug,
                crate::fsr::Flavor::Fsr3,
                &opts.fsr_tune,
            ) {
                Ok(s) => {
                    if fsr.is_none() {
                        fsr = Some(s);
                    } else {
                        // --quinlight with BOTH ffx flavors: FSR4-RR already owns
                        // SRV_SLOT_FSR (there is one standalone-FSR present slot),
                        // and 3.1 is here purely as a fuse engine — the fuse reads
                        // its engines from its OWN descriptor heap, never these
                        // slots.
                        fsr3 = Some(s);
                    }
                }
                Err(e) => eprintln!("fsr3: level unavailable ({e})"),
            }
        }
        if !dlss_live
            && fsr.is_none()
            && xess.is_none()
            && opts.chain != crate::upchain::UpChain::NONE
        {
            eprintln!(
                "upscale: NO temporal upscaler available — chain exhausted \
                 (dlss -> fsr4 -> xess -> fsr3); PLAIN presentation"
            );
        }
        (xess, fsr, fsr3)
    }

    /// Install the presentation descriptors for the native chain selected by
    /// `probe_native`. Kept separate because engine initialization must happen
    /// before the FG swapchain decision, while these heaps exist only after
    /// that swapchain has fixed the presentation format.
    fn wire_native_outputs(
        device: &ID3D12Device,
        passes: &tonemap::Passes,
        bloom: &bloom::BloomGpu,
        aexp: &autoexp::AutoExpGpu,
        xess: Option<&XessState>,
        fsr: Option<&FsrState>,
    ) {
        if let Some(s) = xess {
            wire_tonemap_src(
                device,
                passes,
                bloom,
                aexp,
                &s.res.output,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                tonemap::SRV_SLOT_XESS,
            );
        }
        if let Some(s) = fsr {
            wire_tonemap_src(
                device,
                passes,
                bloom,
                aexp,
                s.res.upscaled(),
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                tonemap::SRV_SLOT_FSR,
            );
        }
    }

    /// Query DLSS-RR's optimal Quality-mode render resolution and its
    /// dynamic range for an output size — the CPU renders inside [min, max]
    /// and RR upscales + denoises to the window size (step-wise DRS via the
    /// per-evaluate InRenderSubrectDimensions). A failed or degenerate query
    /// falls back to DLAA (opt == min == max == output), which main.rs reads
    /// as "DRS off, fixed res". Re-callable — a window resize re-queries at
    /// the new output size.
    #[allow(clippy::type_complexity)]
    fn query_rr_res(
        ngxrr: Option<&ngxrr::NgxRr>,
        w: u32,
        h: u32,
    ) -> ((u32, u32), (u32, u32), (u32, u32)) {
        let Some(nx) = ngxrr else {
            return ((w, h), (w, h), (w, h));
        };
        match nx.optimal(w, h) {
            Ok((opt, min, max)) if opt.0 > 0 && opt.1 > 0 => {
                eprintln!(
                    "dlss: RR Quality {}x{} -> render {}x{} (range {}x{}..{}x{})",
                    w, h, opt.0, opt.1, min.0, min.1, max.0, max.1,
                );
                // Malformed halves of the range collapse to the optimal
                // size — never invent a range the driver didn't report.
                let pmin = if min.0 > 0 && min.1 > 0 && min.0 <= opt.0 && min.1 <= opt.1 {
                    min
                } else {
                    opt
                };
                let pmax = if max.0 >= opt.0 && max.1 >= opt.1 { max } else { opt };
                (opt, pmin, pmax)
            }
            Ok(_) => {
                eprintln!("dlss: optimal-settings degenerate — falling back to DLAA at native res");
                ((w, h), (w, h), (w, h))
            }
            Err(e) => {
                eprintln!("dlss: optimal-settings query failed ({e}) — falling back to DLAA at native res");
                ((w, h), (w, h), (w, h))
            }
        }
    }

    /// Bring up the XeSS context + resource set for an output size. Split
    /// out of `new` (the init_fsr shape) so every failure funnels into one
    /// "xess: disabled" fallback and a window resize can rebuild it.
    fn init_xess(
        xess_dir: &str,
        device: &ID3D12Device,
        w: u32,
        h: u32,
        autoexposure: bool,
    ) -> Result<XessState> {
        let (ctx, opt, min, max) =
            crate::xess::Xess::new(xess_dir, device.as_raw(), (w, h), autoexposure)?;
        eprintln!(
            "xess: {}x{} -> optimal {}x{} (range {}x{}..{}x{})",
            w, h, opt.x, opt.y, min.x, min.y, max.x, max.y,
        );
        let r = xr::XessResources::new(device, max.x, max.y, w, h)
            .map_err(|e| format!("resource allocation failed: {e}"))?;
        Ok(XessState {
            ctx,
            res: r,
            opt: (opt.x, opt.y),
            min: (min.x, min.y),
            max: (max.x, max.y),
        })
    }

    /// Rebuild every window-size-dependent field at a new client size — the
    /// GPU half of a live window resize. The context itself (device, queue,
    /// PSOs, SL interposer + proxies) survives: ResizeBuffers on the SL
    /// proxy swapchain is the documented resize pattern, whereas a full SL
    /// shutdown/re-init in-process is unvalidated. The tracer pipelines and
    /// upscaler contexts are torn down here and rebuilt by the caller's
    /// session re-entry (init_trace / init_dxr at the re-derived locked
    /// render res), which is also why the upscaler planes are recreated
    /// FIRST — wire_session_feed needs them live.
    pub fn resize_output(&mut self, w: u32, h: u32, opts: &GpuOptions) -> Result<()> {
        // Rebuild WHAT WAS WIRED, never re-probe the chain — a resize must
        // not switch upscalers mid-session (captured before the teardown
        // below clears the state it derives from).
        let wired = self.wired();
        // Frame generation straddles the resize: pending paced presents must
        // retire before ResizeBuffers reaches the FI proxy, and the FG effect
        // context is display-size-bound — drop it here, recreate at the new
        // size at the bottom. The swapchain CONTEXT survives (it owns the
        // proxy the session keeps presenting through; the proxy forwards
        // ResizeBuffers to its internal chain).
        if let Some(fg) = &mut self.fg {
            fg.sc.wait_for_presents();
            // The UI registration points at the window-sized HudFi texture,
            // which is about to drop — drive the proxy to null while it is
            // still ALIVE (the swapchain context survives the resize and
            // must not hold a dangling pointer).
            if fg.ui_reg.get() {
                if let Err(e) = fg.sc.register_ui(None) {
                    // The proxy may still hold the pointer — LEAK the old
                    // target rather than let `fg_ui = None` below dangle it
                    // (window-sized, once per FAILED unregister — a rare
                    // path; the rebuild below still creates a fresh one).
                    eprintln!(
                        "fg: ui unregister on resize failed ({e}) — leaking the old UI target"
                    );
                    if let Some(old) = self.fg_ui.take() {
                        std::mem::forget(old);
                    }
                }
                fg.ui_reg.set(false);
            }
            // A resize is a structural change — let a shed registration
            // retry at the new size.
            fg.ui_shed.set(false);
            fg.live.set(false);
            fg.prepared.set(false);
            fg.ctx = None;
        }
        self.fg_ui = None;
        // XeSS-FG: disable so no interpolation is pending across the
        // ResizeBuffers (which the xefg proxy forwards); re-enabled below if
        // the XeSS level comes back up.
        if let Some(x) = &self.fg_x {
            if x.on.get() {
                x.sc.set_enabled(false);
                x.on.set(false);
            }
            x.prepared.set(false);
            // The RES_UI tag is window-sized, so a resize is exactly the
            // structural change that can cure a shed tag — let it retry at
            // the new size (the ffx ui_shed rule above).
            x.ui_shed.set(false);
        }
        // Everything below drops live GPU resources; drain first (the
        // GpuContext::drop discipline — xess/ffx destroy-context require
        // completed command lists, ResizeBuffers requires zero outstanding
        // backbuffer refs).
        self.d3d.wait_idle()?;
        // Raw-NGX DLSS-G: the feature is display-size-bound (created at the
        // old window dims) — release it now (queue just drained) and let the
        // first frame at the new size lazy-recreate; the interpolated-frame
        // target rebuilds below with the other window-sized textures.
        //
        // THE FEATURE SURVIVES THE RESIZE, and that is load-bearing rather
        // than an optimization. Destroying it here CRASHED the process, every
        // time, and the mechanism is the one frdlssg_recreate already documents
        // one level up (SL-era — the sharer was the in-process Streamline;
        // structurally impossible since the retirement, the cheap recreate
        // shape kept): frdlssg_destroy tore at NGX state Streamline
        // SHARED, so the RR rebuild immediately below it could no
        // longer even query NGX (slDLSSDGetOptimalSettings -> eErrorNGXFailed),
        // and SL's Present hook then indexed NGX's feature table with a garbage
        // id (rcx = fffffffa12121206, bit-identical across processes) and took
        // an access violation inside _nvngx — swallowed by its SEH and surfacing
        // to us as Present returning E_ABORT, after which the session shed RR,
        // shed DXR, and panicked at the plain present. MEASURED: reproduces with
        // 418 MB of scene VRAM as readily as with 5.8 GB (so it is not memory
        // pressure), and disappears entirely when the feature is kept — the RR
        // query then succeeds and 8K comes up clean. quinlight, the raw-NGX
        // blueprint, cannot exhibit any of this: it has exactly ONE NGX client
        // and no Streamline at all.
        //
        // dims is deliberately NOT cleared: the res-follow recreate above wants
        // to see the OLD size so it recognizes the move and rebuilds the feature
        // at the new display AND render res (a rend-only rebuild is 0xBAD00005).
        // FR_FG_RESIZE_DESTROY=1 restores the destroying path for A/B only.
        let fg_destroy = std::env::var("FR_FG_RESIZE_DESTROY").ok().as_deref() == Some("1");
        if let Some(n) = &self.fg_n {
            if fg_destroy {
                eprintln!(
                    "fg: FR_FG_RESIZE_DESTROY=1 — destroying the NGX feature across the \
                     resize (the known-crashing path; A/B only)"
                );
                let h = n.handle.replace(std::ptr::null_mut());
                if !h.is_null() {
                    unsafe { ngxfg::frdlssg_destroy(h) };
                }
                n.dims.set((0, 0));
            }
            n.primed.set(false);
            n.res_pend.set(((0, 0), 0));
        }
        self.trace = None;
        self.dxr = None;
        // NRD is RENDER-res-bound (NrdGpu at rw,rh), and at --lock-res the
        // render res follows the window — so the instance must die with the
        // tracers or the re-entry's arm_nrd_for hits its size-mismatch
        // refusal on every maximize ("NRD armed at 1920x1080, this arm
        // traces 3435x1332" — the user-repro that found this). Safe by the
        // tracer-drop argument: the queue is drained above, so none of the
        // mid-session shed's two-phase discipline applies; the re-entry
        // re-arms at the new res (the None arm resets history/prev/idx and
        // prints the armed line again). `nrd_shed` deliberately survives —
        // it marks a runtime FAILURE, not a size, and a resize must not
        // un-tombstone a genuinely failing NRD.
        self.nrd_gpu = None;
        self.nrd_hist_valid = false;
        // --dual-gpu: the secondary's TRACER and its staging are window-bound
        // and go with the primary's; its DEVICE and its SCENE CORE survive in
        // `dual_keep`, so the re-entry re-pays kernel compiles and planes but
        // NOT the second scene upload — the same bargain `scene_gpu` strikes
        // one line down, and by far the more expensive of the two to re-pay.
        self.dual_keep = self.dual.take().map(|d| DualKeep {
            arm: d.sec.arm(),
            hg: d.hg,
            core: d.core,
            bal: d.bal,
        });
        // scene_gpu deliberately SURVIVES: the shared core is scene-bound,
        // not window-bound (device+queue live across d3d.resize), so the
        // re-entry's tracer rebuilds skip the scene upload + BLAS build —
        // resize/F11 no longer re-pays them (only kernel compiles + the
        // window-sized planes).
        self.nppd_gpu = None;
        self.nppd_state_valid = false;
        self.xess = None;
        self.fsr = None;
        self.fsr3 = None;
        self.quin = None;
        self.rr = None;
        self.d3d.resize(w, h)?;
        self.blit = match self.d3d.space {
            d3d12::PresentSpace::Sdr10 | d3d12::PresentSpace::Hdr10 => {
                upload::BlitUpload::new_10bit(&self.d3d, w, h)?
            }
            d3d12::PresentSpace::Sdr => upload::BlitUpload::new(&self.d3d, w, h)?,
        };
        self.hdr = upload::HdrUpload::new(&self.d3d, w, h)?;
        // The HUD overlay is window-sized: new texture (fresh SRV in slot 9),
        // new ring, content undefined until crate::hud's forced full-window
        // frame lands. Carry the visibility flag across.
        let hud_visible = self.hud.visible.get();
        self.hud = hud::HudGpu::new(&self.d3d.device, &self.passes, self.d3d.format, w, h)?;
        self.hud.visible.set(hud_visible);
        // The frame-generation UI target follows the window (the straddle
        // above already nulled the proxy's registration and dropped the old
        // texture). Same predicate + loud-fallback rule as boot — ONE
        // builder (make_fg_ui), so the two sites can't drift again.
        self.fg_ui = if self.fg.is_some() || self.fg_x.as_ref().is_some_and(|x| x.enabled) {
            make_fg_ui(&self.d3d.device, &self.passes, &self.hud, self.d3d.space, w, h)
        } else {
            None
        };
        // A resize invalidates the last-present record: every source it could
        // name is being rebuilt at the new size.
        self.last_present.set(None);
        // The glare pyramid is window-sized; rebuild it and re-point the tonemap's
        // halo SRV at the new level 0 (the old resource is gone).
        self.bloom.set_res(&self.d3d.device, w, h)?;
        self.passes.create_srv(
            &self.d3d.device,
            self.bloom.glare_srv_source(),
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            tonemap::SRV_SLOT_BLOOM,
        );
        // The blit arm never blooms (blit PSO — see `new`), so a plain SRV.
        self.passes.create_srv(
            &self.d3d.device,
            &self.blit.texture,
            self.blit.format,
            tonemap::SRV_SLOT_BLIT,
        );
        // A resize can also be a monitor change (drag to another display, then
        // let go). Re-probe here rather than trusting the poll to catch up.
        self.refresh_display(opts.paper_white, opts.peak_nits);
        wire_tonemap_src(
            &self.d3d.device,
            &self.passes,
            &self.bloom,
            &self.autoexp,
            &self.hdr.texture,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            tonemap::SRV_SLOT_HDR,
        );
        if self.ngxrr.is_some() {
            let (opt, min, max) = Self::query_rr_res(self.ngxrr.as_ref(), w, h);
            let r = rr::RrResources::new(&self.d3d.device, opt, min, max, w, h)?;
            wire_tonemap_src(
                &self.d3d.device,
                &self.passes,
                &self.bloom,
                &self.autoexp,
                &r.output,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                tonemap::SRV_SLOT_RR,
            );
            self.rr = Some(r);
            // The DLSSD feature follows: ReleaseFeature + CreateFeature at
            // the new dims (the queue was drained at the top of this
            // function — the recreate contract). BOTH dim pairs move: a
            // window resize changes render and target together, and the
            // old-target x new-render rebuild is the measured
            // FAIL_InvalidParameter class. A failed recreate sheds RR loud —
            // dlss_ready() must not stay half-true on a dead feature.
            let nx = self.ngxrr.as_ref().unwrap();
            let dlaa = opt == (w, h) && min == opt && max == opt;
            let ok = match &mut self.rr_feature {
                Some(f) => f.recreate(nx, max, (w, h), dlaa),
                None => nx.create_feature(max, (w, h), dlaa).map(|f| {
                    self.rr_feature = Some(f);
                }),
            };
            if let Err(e) = ok {
                eprintln!("dlss: {e} after resize — RR dropped for this session");
                self.rr_feature = None;
                self.rr = None;
                self.ngxrr = None;
            }
        }
        match wired {
            // --quinlight: the session's engines ARE "what was wired", so the
            // rebuild is the same probe `new` ran. It cannot switch upscalers
            // mid-session (the invariant this match exists to protect): support
            // is a property of the box, not of the window size. The fuse itself
            // is rebuilt lazily by the session's init_trace/init_dxr re-entry,
            // which a resize already forces.
            WiredUpscaler::Quin => {
                let (x, f, f3) =
                    Self::probe_native(&self.d3d.device, opts, w, h, self.ngxrr.is_some());
                Self::wire_native_outputs(
                    &self.d3d.device,
                    &self.passes,
                    &self.bloom,
                    &self.autoexp,
                    x.as_ref(),
                    f.as_ref(),
                );
                self.xess = x;
                self.fsr = f;
                self.fsr3 = f3;
            }
            WiredUpscaler::Xess => {
                match Self::init_xess(&opts.xess_dir, &self.d3d.device, w, h, opts.xess_autoexposure) {
                    Ok(s) => {
                        wire_tonemap_src(
                            &self.d3d.device,
                            &self.passes,
                            &self.bloom,
                            &self.autoexp,
                            &s.res.output,
                            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                            tonemap::SRV_SLOT_XESS,
                        );
                        self.xess = Some(s);
                    }
                    Err(e) => eprintln!("xess: disabled after resize — {e}"),
                }
            }
            WiredUpscaler::Fsr4 | WiredUpscaler::Fsr3 => {
                let flavor = if wired == WiredUpscaler::Fsr4 {
                    crate::fsr::Flavor::Fsr4Rr
                } else {
                    crate::fsr::Flavor::Fsr3
                };
                match Self::init_fsr(&opts.ffx_dir, &self.d3d.device, w, h, opts.debug, flavor, &opts.fsr_tune) {
                    Ok(s) => {
                        wire_tonemap_src(
                            &self.d3d.device,
                            &self.passes,
                            &self.bloom,
                            &self.autoexp,
                            s.res.upscaled(),
                            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                            tonemap::SRV_SLOT_FSR,
                        );
                        self.fsr = Some(s);
                    }
                    Err(e) => eprintln!("fsr: disabled after resize — {e}"),
                }
            }
            // Rr rebuilds through the sl block above; Plain has nothing.
            WiredUpscaler::Rr | WiredUpscaler::Plain => {}
        }
        // Raw-NGX DLSS-G: rebuild the window-sized interpolated-frame target
        // + its tonemap SRV (the feature itself lazy-recreates on the first
        // frame).
        if let Some(n) = &mut self.fg_n {
            n.out = d3d12::committed_tex(
                &self.d3d.device,
                w,
                h,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )?;
            wire_tonemap_src(
                &self.d3d.device,
                &self.passes,
                &self.bloom,
                &self.autoexp,
                &n.out,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                tonemap::SRV_SLOT_NGXFG,
            );
        }
        // XeSS-FG re-enable when the XeSS level came back up (the swapchain
        // context itself survived the ResizeBuffers).
        if let Some(x) = &self.fg_x {
            if self.xess.is_some() && x.enabled && !x.failed.get() {
                x.sc.set_enabled(true);
                x.on.set(true);
            }
        }
        // Frame generation, rebuilt at the new display size (the provider
        // pick was made at boot and does not move — `version` survives on the
        // state). Only when the FSR level actually came back up: its arms
        // carry the prepare.
        if let Some(fg) = &mut self.fg {
            if self.fsr.is_some() && fg.version.0 != 0 {
                match ffx::FgContext::create(
                    &self.d3d.device,
                    (w, h),
                    (w, h),
                    match self.d3d.space {
                        d3d12::PresentSpace::Sdr10 | d3d12::PresentSpace::Hdr10 => {
                            ffx_sys::SURFACE_FORMAT_R10G10B10A2_UNORM
                        }
                        d3d12::PresentSpace::Sdr => ffx_sys::SURFACE_FORMAT_B8G8R8A8_UNORM,
                    },
                    self.d3d.space == d3d12::PresentSpace::Hdr10,
                    opts.debug,
                    fg.version.0,
                ) {
                    Ok(ctx) => fg.ctx = Some(ctx),
                    Err(e) => eprintln!("fg: disabled after resize — {e}"),
                }
            }
        }
        Ok(())
    }

    /// Bring up the ffx context(s) + the requested flavor's resources. Split
    /// out of `new` so every failure funnels into one fall-through line.
    /// `flavor` is the chain level being probed: `Fsr4Rr` requires the Ray
    /// Regeneration provider (RDNA4) and errs without it; `Fsr3` never
    /// consults the denoiser enumeration (--fsr3 and the chain's last level).
    fn init_fsr(
        ffx_dir: &str,
        device: &ID3D12Device,
        w: u32,
        h: u32,
        debug: bool,
        flavor: crate::fsr::Flavor,
        tune: &crate::fsr::DenoiseTuning,
    ) -> Result<FsrState> {
        let mut ctx = ffx::FfxContext::load(ffx_dir)?;
        // Ray Regeneration probe. A probe *error* degrades exactly like an
        // empty enumeration — on non-RDNA4 adapters the denoiser query may
        // fail outright rather than report zero providers, and either way
        // the answer is "no RR here", not "no FSR here".
        let den = if flavor == crate::fsr::Flavor::Fsr3 {
            Vec::new()
        } else {
            match ctx.versions(false, device) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("fsr: denoiser probe failed ({e}); treating as no provider");
                    Vec::new()
                }
            }
        };
        for (id, name) in &den {
            eprintln!("fsr: denoiser provider {name} (id {id:#x})");
        }
        if flavor == crate::fsr::Flavor::Fsr4Rr && den.is_empty() {
            return Err(
                "no Ray Regeneration provider on this adapter (RDNA4 required)".into(),
            );
        }
        let ups = match ctx.versions(true, device) {
            Ok(v) if !v.is_empty() => v,
            bad => {
                // Diagnose before failing: a null-device enumeration lists
                // what the loader could load from disk, separating "provider
                // DLLs missing/unloadable" from "adapter rejected by every
                // provider".
                match ctx.versions_any(true) {
                    Ok(all) if !all.is_empty() => {
                        for (id, name) in &all {
                            eprintln!("fsr: provider on disk (device-independent): {name} (id {id:#x})");
                        }
                        eprintln!("fsr: providers loaded but none supports this adapter/device");
                    }
                    Ok(_) => eprintln!("fsr: loader found NO provider DLLs next to itself (check --ffx-path)"),
                    Err(e) => eprintln!("fsr: device-independent enumeration also failed ({e})"),
                }
                return Err(match bad {
                    Ok(_) => "FSR upscaler reports no provider on this adapter".into(),
                    Err(e) => e,
                });
            }
        };
        for (id, name) in &ups {
            eprintln!("fsr: upscaler provider {name} (id {id:#x})");
        }
        let fsr3 = flavor == crate::fsr::Flavor::Fsr3;
        let Some((vid, picked)) = crate::fsr::pick_version(&ups, !den.is_empty(), fsr3) else {
            return Err("no FSR 3.x upscaler provider enumerated (see the provider list above)".into());
        };
        // The pick can only disagree with the request through a logic bug —
        // Fsr4Rr requires the RR enumeration (gated above) and fsr3 forces.
        debug_assert_eq!(picked, flavor);
        if fsr3 {
            eprintln!("fsr: FSR 3.1 upscale-only provider (id {vid:#x})");
        }
        // Dynamic-resolution range: max = the window itself (the controller
        // creeps to native while still); every context takes maxRenderSize =
        // window and a per-dispatch renderSize.
        if flavor == crate::fsr::Flavor::Fsr4Rr {
            ctx.create_denoiser(device, (w, h))?;
            if tune.any() {
                ctx.tune_denoiser(tune);
            }
        }
        ctx.create_upscaler(device, (w, h), (w, h), debug, vid)?;
        let q = |mode: u32, ratio: f32| -> (u32, u32) {
            ctx.upscaler_render_res((w, h), mode).unwrap_or_else(|e| {
                let f = crate::fsr::fallback_render_res((w as usize, h as usize), ratio);
                eprintln!("fsr: render-res query failed ({e}); using ratio fallback");
                (f.0 as u32, f.1 as u32)
            })
        };
        let opt = q(ffx_sys::QUALITY_MODE_QUALITY, crate::fsr::RATIO_QUALITY);
        let min = q(ffx_sys::QUALITY_MODE_ULTRA_PERFORMANCE, crate::fsr::RATIO_ULTRA_PERFORMANCE);
        eprintln!(
            "fsr: {}x{} -> seed {}x{} (range {}x{}..{}x{})",
            w, h, opt.0, opt.1, min.0, min.1, w, h,
        );
        let res = match flavor {
            crate::fsr::Flavor::Fsr4Rr => FsrRes::Rr(ffx_rr::FsrResources::new(device, w, h, w, h)?),
            crate::fsr::Flavor::Fsr3 => FsrRes::Up(ffx_up::Fsr3Resources::new(device, w, h, w, h)?),
        };
        Ok(FsrState { ctx, res, opt, min, max: (w, h) })
    }

    /// Wire the session's live upscaler input planes as feed targets on a GPU
    /// tracer (`wire` = TraceGpu's or DxrGpu's `wire_feed_add` — it APPENDS, and
    /// is called once per engine, which is what --quinlight needs; the tracer is
    /// freshly built here, so there is nothing to clear).
    ///
    /// The trace res was quantize_res-clamped by the caller, but the range is
    /// the SDK's contract: re-check here so a drift fails loudly at init. A gbuf
    /// session with no live upscaler is a wiring bug, not a fallback.
    fn wire_session_feed(
        &self,
        rw: u32,
        rh: u32,
        mut wire: impl FnMut(
            trace::FeedKind,
            &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
        ) -> Result<()>,
    ) -> Result<()> {
        // --quinlight: EVERY wired engine gets fed, each into its own descriptor
        // set (wire_feed appends). The engines' register sets overlap — DLSS-RR
        // and FSR4-RR both claim u16..u22 — which is exactly what the sets are
        // for. XeSS and FSR 3.1 are the one exception that needs no second feed:
        // their plane sets are byte-identical, so FSR 3.1 upscales straight from
        // the XeSS trio (ffx_up::upscale_res_shared) and only pays a feed of its
        // own when XeSS is absent.
        // The predicate is STRUCTURAL, not `self.quin.is_some()`: the fuse is
        // built after the feed is wired (it needs the DXC that init_trace holds),
        // so asking for it here would always say no. More than one live engine IS
        // a quinlight session.
        if self.quin_engines().0.len() > 1 {
            if let Some(rr) = &self.rr {
                if self.ngxrr.is_some() {
                    Self::wire_rr_feed(rr, rw, rh, &mut wire)?;
                }
            }
            // The sharing keys on the FLAVOR, not on the field. probe_native
            // parks FSR 3.1 in `fsr` whenever FSR4-RR is absent — which is the
            // NVIDIA set (dlss-rr + fsr3 + xess) — and only in `fsr3` when both
            // ffx flavors came up. A field-keyed rule fed the `fsr`-resident 3.1
            // a whole descriptor set per frame that record_quin_engines then
            // never read (it hands every Up-flavor context the XeSS trio): a full
            // wasted feed dispatch on the primary dev config, and it burned the
            // third and last FEED_SET. FSR4-RR always needs its own eleven-plane
            // feed; a shared 3.1 still has its SDK range checked, since the one
            // traced res must be legal for it too.
            for fs in [self.fsr.as_ref(), self.fsr3.as_ref()].into_iter().flatten() {
                if self.xess.is_some() && matches!(fs.res, FsrRes::Up(_)) {
                    fsr_range_check(fs, rw, rh)?;
                    continue;
                }
                Self::wire_fsr_feed(fs, rw, rh, &mut wire)?;
            }
            if let Some(x) = &self.xess {
                Self::wire_xess_feed(x, rw, rh, &mut wire)?;
            }
            return Ok(());
        }
        if let Some(rr) = &self.rr {
            Self::wire_rr_feed(rr, rw, rh, &mut wire)
        } else if let Some(x) = &self.xess {
            Self::wire_xess_feed(x, rw, rh, &mut wire)
        } else if let Some(fs) = &self.fsr {
            Self::wire_fsr_feed(fs, rw, rh, &mut wire)
        } else {
            Err("gbuf session with no live upscaler".into())
        }
    }

    /// Every engine's feed wiring is one of these three. Each re-checks the
    /// trace res against ITS SDK's range: the caller clamped it, but the range
    /// is the SDK's contract, so a drift fails loudly at init rather than
    /// quietly at execute. (Under --quinlight the res must satisfy every wired
    /// engine at once — main.rs intersects the ranges, and these are what
    /// enforce it.)
    fn wire_rr_feed(
        rr: &rr::RrResources,
        rw: u32,
        rh: u32,
        wire: &mut impl FnMut(
            trace::FeedKind,
            &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
        ) -> Result<()>,
    ) -> Result<()> {
        if rw < rr.min.0 || rh < rr.min.1 || rw > rr.max.0 || rh > rr.max.1 {
            return Err(format!(
                "trace res {}x{} outside DLSS-RR render range {}x{}..{}x{}",
                rw, rh, rr.min.0, rr.min.1, rr.max.0, rr.max.1
            ));
        }
        let pl = rr.plane_resources();
        wire(
            trace::FeedKind::Rr,
            &[
                (trace::FEED_COLOR, pl[0].0, pl[0].1),
                (trace::FEED_NR, pl[1].0, pl[1].1),
                (trace::FEED_DEPTH, pl[2].0, pl[2].1),
                (trace::FEED_MVEC, pl[3].0, pl[3].1),
                (trace::FEED_ALB, pl[4].0, pl[4].1),
                (trace::FEED_SPEC, pl[5].0, pl[5].1),
                (trace::FEED_SPECHIT, pl[6].0, pl[6].1),
                // The FG guide pass's ripple plane rides FEED_FSR_AO (u26):
                // same R16F/RWTexture2D<float>, and an RR session never runs
                // the FSR-RR kernel that owns it. Not an RR input.
                (trace::FEED_FSR_AO, pl[7].0, pl[7].1),
            ],
        )
    }

    fn wire_xess_feed(
        x: &XessState,
        rw: u32,
        rh: u32,
        wire: &mut impl FnMut(
            trace::FeedKind,
            &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
        ) -> Result<()>,
    ) -> Result<()> {
        if rw < x.min.0 || rh < x.min.1 || rw > x.max.0 || rh > x.max.1 {
            return Err(format!(
                "trace res {}x{} outside XeSS input range {}x{}..{}x{}",
                rw, rh, x.min.0, x.min.1, x.max.0, x.max.1
            ));
        }
        let pl = x.res.plane_resources();
        wire(
            trace::FeedKind::Xess,
            &[
                (trace::FEED_COLOR, pl[0].0, pl[0].1),
                (trace::FEED_MVEC, pl[1].0, pl[1].1),
                (trace::FEED_DEPTH, pl[2].0, pl[2].1),
            ],
        )
    }

    fn wire_fsr_feed(
        fs: &FsrState,
        rw: u32,
        rh: u32,
        wire: &mut impl FnMut(
            trace::FeedKind,
            &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
        ) -> Result<()>,
    ) -> Result<()> {
        fsr_range_check(fs, rw, rh)?;
        match &fs.res {
            FsrRes::Up(res) => {
                let pl = res.plane_resources();
                wire(
                    trace::FeedKind::Fsr3,
                    &[
                        (trace::FEED_COLOR, pl[0].0, pl[0].1),
                        (trace::FEED_MVEC, pl[1].0, pl[1].1),
                        (trace::FEED_DEPTH, pl[2].0, pl[2].1),
                    ],
                )
            }
            FsrRes::Rr(res) => {
                // cs_feed_fsr_rr's register/plane mapping (feed.hlsl — keep in
                // lockstep; plane_resources returns upload order: depth_lin,
                // depth_clip, mvec, normals, diff_alb, spec_alb, dd_in, ds_in,
                // residual, ao_in, is_in).
                let pl = res.plane_resources();
                wire(
                    trace::FeedKind::FsrRr,
                    &[
                        (trace::FEED_SPECHIT, pl[0].0, pl[0].1), // R32F linear depth
                        (trace::FEED_DEPTH, pl[1].0, pl[1].1),   // R32F clip depth
                        (trace::FEED_FSR_MVEC, pl[2].0, pl[2].1),
                        (trace::FEED_NR, pl[3].0, pl[3].1), // RGB10A2 oct-normals
                        (trace::FEED_ALB, pl[4].0, pl[4].1),
                        (trace::FEED_SPEC, pl[5].0, pl[5].1),
                        (trace::FEED_FSR_DD, pl[6].0, pl[6].1),
                        (trace::FEED_FSR_DS, pl[7].0, pl[7].1),
                        (trace::FEED_COLOR, pl[8].0, pl[8].1), // RGBA16F residual
                        (trace::FEED_FSR_AO, pl[9].0, pl[9].1), // R16F AO
                        (trace::FEED_FSR_IS, pl[10].0, pl[10].1), // RGBA16F indirect spec
                    ],
                )
            }
        }
    }

    /// Bring up the GPU-resident tracer: capability gates, kernel compiles,
    /// scene upload + BLAS/TLAS build (synchronous, one-time), HDR SRV wire-up.
    /// `(rw, rh)` is the session's fixed trace resolution (the locked render
    /// res in upscaler sessions, the window size otherwise); `gbuf` sizes the
    /// G-buffer pack (full-size iff an upscaler will consume it).
    /// `nppd` (XeSS sessions only): `(dll_dir, model_path)` for the
    /// GPU-resident NPPD pre-denoise stage — its init failure is a loud line
    /// + plain GPU-XeSS, never a session failure.
    #[allow(clippy::too_many_arguments)]
    pub fn init_trace(
        &mut self,
        dxc: &dxc::Dxc,
        scene: &crate::scene::Scene,
        bvh: &crate::bvh::Bvh,
        rw: u32,
        rh: u32,
        gbuf: bool,
        nppd: Option<(&str, &str)>,
        dn: Option<DnKind>,
        debug: bool,
        bc7_mode: crate::bc7::Bc7Mode,
    ) -> Result<()> {
        let dev = self.d3d.device.clone();
        let core = self.ensure_scene_gpu(scene, bvh, bc7_mode)?;
        let tg = trace::TraceGpu::new(
            &dev,
            dxc,
            scene,
            bvh,
            core,
            rw,
            rh,
            gbuf,
            nppd.is_some(),
            dn.is_some(),
            debug,
            &mut self.d3d,
        );
        let mut tg = match tg {
            Ok(t) => t,
            Err(e) => {
                self.evict_unused_scene_gpu();
                return Err(e);
            }
        };
        // Upscaler sessions: wire the live upscaler's input planes as feed
        // targets — the feed kernel writes them directly, no CPU upload.
        // (Failures before `self.trace = Some(..)` evict the cached core —
        // the tracer never went live, so nobody holds it.)
        if gbuf {
            let wired = self
                .wire_session_feed(rw, rh, |kind, targets| tg.wire_feed_add(&dev, kind, targets));
            if let Err(e) = wired {
                self.evict_unused_scene_gpu();
                return Err(e);
            }
        }
        // GPU-resident NPPD: ORT session on OUR device/queue, tensors bound
        // over the NppdRes buffers. XeSS-only (RR is itself a denoiser — the
        // same exclusion as the CPU paths); a failure keeps the session
        // running plain and frees the ~340 MB staging.
        if let Some((dir, model)) = nppd {
            if self.xess.is_none() {
                self.evict_unused_scene_gpu();
                return Err("NPPD composition requires the XeSS session".into());
            }
            let built = tg.nppd.as_ref().ok_or("NPPD staging missing".to_string()).and_then(
                |n| {
                    crate::nppd::NppdGpu::new(
                        dir,
                        model,
                        self.d3d.device.as_raw(),
                        self.d3d.queue.as_raw(),
                        rw as usize,
                        rh as usize,
                        n.frame.as_raw(),
                        n.warped.as_raw(),
                        n.out.as_raw(),
                        n.state.as_raw(),
                    )
                },
            );
            match built {
                Ok(g) => {
                    self.nppd_gpu = Some(g);
                    self.nppd_state_valid = false;
                }
                Err(e) => {
                    eprintln!("nppd-gpu: unavailable ({e}); running plain GPU-XeSS");
                    tg.nppd = None;
                }
            }
        }
        // Pre-upscale denoising (NRD or FRD): the engine's passes between the
        // bridge kernels, all on the one list (no split — pure compute). The
        // arming/wiring itself is `arm_denoiser_for` — ONE block shared with
        // init_dxr, and ONE engine instance shared by both arms. A failure
        // keeps the session running plain and sheds loudly (the nppd-gpu
        // shape). The denoiser + --nppd conflict is refused at the
        // CLI/session boundary (both claim the pre-upscale color slot),
        // asserted again here.
        if let Some(dn) = dn {
            if nppd.is_some() {
                self.evict_unused_scene_gpu();
                return Err("the denoiser and --nppd both claim the pre-upscale color slot".into());
            }
            if let Err(e) = self.arm_denoiser_for(&dev, dxc, dn, rw, rh, debug, &mut |t| {
                tg.wire_nrd_feed(&dev, t)
            })
            {
                let hint = match dn {
                    DnKind::Nrd(_) => {
                        " (install-prerequisites.bat nrd builds the SDK; --no-nrd silences this)"
                    }
                    DnKind::Frd => "",
                };
                let tag = match dn {
                    DnKind::Nrd(_) => "nrd",
                    DnKind::Frd => "frd",
                };
                eprintln!("{tag}: unavailable ({e}); running plain GPU upscaling{hint}");
                tg.nrd = None;
            }
        }
        wire_tonemap_src(
            &self.d3d.device,
            &self.passes,
            &self.bloom,
            &self.autoexp,
            &tg.hdr,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            tonemap::SRV_SLOT_GPU,
        );
        self.trace = Some(tg);
        // The fuse, over whatever engines the chain wired (--quinlight only).
        self.build_quin(dxc)?;
        // --dual-gpu. Built LAST and never fatal: the session is a working
        // single-GPU one at this point, and a second adapter that cannot be
        // opened is a missing optimisation, not a failed session. One loud
        // line, the `nppd-gpu: unavailable (…)` shape.
        if let Some(share) = self.dual_want {
            if self.dual.is_none() {
                match self.build_dual(dxc, scene, bvh, rw, rh, share, gbuf, debug, bc7_mode) {
                    Ok(()) => {}
                    Err(e) => {
                        eprintln!("dual-gpu: unavailable ({e}) — running single-GPU");
                        // Latch it off: without this every SPACE re-entry
                        // retries a full second scene upload + BLAS build.
                        self.dual_want = None;
                    }
                }
            }
        }
        Ok(())
    }

    /// Open the second adapter, upload the scene to it, and build its tracer.
    ///
    /// Everything here is device-parameterised already — `from_pick`,
    /// `SceneGpu::new_uploaded`, `TraceGpu::new` — so this is assembly, not new
    /// machinery. The staging cap is sized from the FULL screen at the WORST
    /// stride set, so neither a rebalance nor a wiring change reallocates.
    #[allow(clippy::too_many_arguments)]
    fn build_dual(
        &mut self,
        dxc: &dxc::Dxc,
        scene: &crate::scene::Scene,
        bvh: &crate::bvh::Bvh,
        rw: u32,
        rh: u32,
        share: u32,
        gbuf: bool,
        debug: bool,
        bc7_mode: crate::bc7::Bc7Mode,
    ) -> Result<()> {
        // The split depth must stay strictly under the leaf frontier — see
        // `TraceGpu::set_split`, where that is a soundness guard rather than a
        // sanity check. Clamping HERE rather than letting `set_split` refuse
        // is what keeps a small render resolution splitting at all: at 240x135
        // `depth_full` is 3, so the requested depth 3 would be declined every
        // frame while depth 2 works perfectly well. A frame too small for even
        // depth 1 has nothing to balance, and says so once.
        let full = trace::depth_full(rw, rh);
        if full <= 1 {
            return Err(format!(
                "a {rw}x{rh} frame is one quadtree level deep — there are no tiles to split"
            ));
        }
        let depth = self.dual_depth.clamp(1, trace::MAX_SPLIT_DEPTH.min(full - 1));
        if depth != self.dual_depth.clamp(1, trace::MAX_SPLIT_DEPTH) {
            eprintln!(
                "dual-gpu: split depth {} is at or past the leaf frontier at {rw}x{rh} \
                 (depth_full={full}) — splitting at depth {depth} instead",
                self.dual_depth
            );
        }
        let side = 1u32 << depth;
        let share = share.min(side - 1);
        // A resize kept the device, the scene core and the balancer's converged
        // state; only the window-bound half is rebuilt.
        let kept = self.dual_keep.take();
        let resumed = kept.is_some();
        let (mut hg, core, bal, arm) = match kept {
            Some(k) => (k.hg, k.core, k.bal, k.arm),
            None => {
                let f = adapter::create_factory(debug).map_err(|e| format!("factory: {e}"))?;
                let prim_luid = adapter::luid_of_device(&self.d3d.device);
                let pick = adapter::enumerate(&f)
                    .into_iter()
                    .filter(|a| a.luid != prim_luid)
                    .reduce(|a, b| if b.vram > a.vram { b } else { a })
                    .ok_or("no second hardware adapter")?;
                let mut hg = trace::HeadlessGpu::from_pick(&pick, debug)?;
                let dev = hg.device.clone();
                // WHICH PIPELINE THE SECONDARY RUNS, decided here and once.
                // The caps probe comes BEFORE the scene upload — the cheapest
                // place to change our mind, since the core is arm-independent
                // and both arms take the same Rc. A device that cannot host a
                // DxrGpu degrades to the wavefront LOUDLY rather than failing
                // the session: on such a box the wavefront is exactly what the
                // shipping policy would have picked anyway.
                let caps = dxr::require_caps(&dev);
                let arm = dual::arm_for(pick.vendor, self.dual_arm, caps.is_ok());
                if let Err(e) = &caps {
                    if self.dual_arm == Some(dual::Arm::Dxr) {
                        eprintln!(
                            "dual-gpu: --dual-gpu-arm dxr cannot be honoured on secondary \
                             \"{}\" ({e}) — its share runs the compute wavefront instead",
                            pick.name
                        );
                    } else if pick.vendor != adapter::Vendor::Intel
                        && pick.vendor != adapter::Vendor::Other
                    {
                        eprintln!(
                            "dual-gpu: secondary \"{}\" cannot run the DXR pipeline ({e}) — \
                             its share runs the compute wavefront instead",
                            pick.name
                        );
                    }
                }
                let core = std::rc::Rc::new(trace::SceneGpu::new_uploaded(
                    &dev, scene, bvh, &mut hg, bc7_mode,
                )?);
                (
                    hg,
                    core,
                    dual::Balancer::new(share, depth, self.dual_auto, dual::SHARE_DWELL),
                    arm,
                )
            }
        };
        // A resize can move `depth_full` under the depth the kept balancer
        // converged at, and its depth is baked in (it indexes the row grid).
        // Rebuilding it at the legal depth costs the converged share, which is
        // strictly better than every frame's `set_split` declining.
        let mut bal = bal;
        if bal.depth() != depth {
            bal = dual::Balancer::new(share, depth, self.dual_auto, dual::SHARE_DWELL);
        }
        let dev = hg.device.clone();
        // The pack matters here in a way it never did for a capture: an
        // interactive frame is FED, so `record_feed` reads the secondary's rows
        // of it. The secondary therefore stores one whenever the PRIMARY does —
        // `gbuf` mirrored, and both forcing hooks mirrored per frame in
        // `record_trace`, since its own feed list is empty by construction. A
        // plain session mirrors the dummy instead and only `accum` crosses.
        let sec = match arm {
            dual::Arm::Wave => dual::Secondary::Wave(trace::TraceGpu::new(
                &dev, dxc, scene, bvh, core.clone(), rw, rh, gbuf, false, false, debug, &mut hg,
            )?),
            // DxrGpu::new needs no upload harness — there is no queue or
            // submit anywhere in it, unlike TraceGpu::new's software trees.
            dual::Arm::Dxr => dual::Secondary::Dxr(dxr::DxrGpu::new(
                &dev,
                dxc,
                scene,
                core.clone(),
                rw,
                rh,
                gbuf,
                false,
                debug,
            )?),
        };
        let xf = dual::BandTransfer::new(
            &dev,
            &self.d3d.device,
            dual::payload_bytes(&dual::MAX_FED_STRIDES, rw, rh),
        )?;
        if !resumed {
            eprintln!(
                "dual-gpu: secondary \"{}\" running {} at {}/{} rows{} | primary \"{}\"",
                hg.adapter_name,
                arm.name(),
                bal.rows(),
                side,
                if self.dual_auto { ", balancer driving" } else { ", pinned" },
                self.adapter_name
            );
        }
        self.dual = Some(DualState {
            hg,
            core,
            sec,
            xf,
            bal,
            pack: false,
            ext: false,
            last_sec_ms: 0.0,
            said: false,
            said_mixed: false,
            said_pack: false,
        });
        Ok(())
    }

    /// Record the wavefront trace for this frame — THE ONE SITE every
    /// wavefront presenter goes through, so `--dual-gpu` reaches all six of
    /// them with a one-line change each and cannot be half-wired.
    ///
    /// Single-GPU (the overwhelmingly common case, and the case a balancer
    /// that reached zero produces) is the early return: `write_cb` +
    /// `record_frame`, byte for byte what the presenters used to inline.
    ///
    /// THE DUAL SCHEDULE, and why it cannot live in `session()` the way
    /// `--spin`'s does: a presenter records AND presents in one call, so the
    /// only place both devices can be in flight together is inside it.
    ///
    ///   secondary submit  ->  primary record  ->  d3d.split_frame
    ///     -> wait(secondary) -> band out -> hop -> record the band IN
    ///
    /// `split_frame` is what makes it concurrent rather than serial: it
    /// executes the primary's trace immediately, so the CPU's wait on the
    /// secondary overlaps it. Without it the primary would not start until the
    /// frame's single ExecuteCommandLists at present time, and the two devices
    /// would run one after the other — a working feature reported as a large
    /// regression. The band copy then rides the FRESH list, which the same
    /// queue executes after the trace by FIFO order, so it needs no fence of
    /// its own; only `hop` does, because it is a CPU memcpy out of the
    /// secondary's readback.
    fn record_trace(
        &mut self,
        p: &trace::FrameParams,
        hybrid: bool,
        slot: usize,
    ) -> Result<()> {
        let Self { dual, trace, d3d, .. } = self;
        let tg = trace.as_ref().ok_or("no wavefront tracer")?;
        Self::record_split(dual.as_mut(), d3d, dual::Tracer::Wave(tg), p, hybrid, slot)
    }

    /// The DXR twin of `record_trace` — THE ONE SITE every DXR presenter goes
    /// through, for the same reason.
    ///
    /// `hybrid` is the wavefront's R-key A/B and has no meaning here (this
    /// pipeline has no reference kernel), so it is not a parameter; the shared
    /// schedule takes `true` and the DXR arm ignores it.
    fn record_dxr_trace(&mut self, p: &trace::FrameParams, slot: usize) -> Result<()> {
        let Self { dual, dxr, d3d, .. } = self;
        let dg = dxr.as_ref().ok_or("DXR pipeline not initialized")?;
        Self::record_split(dual.as_mut(), d3d, dual::Tracer::Dxr(dg), p, true, slot)
    }

    /// The frame schedule itself, for EITHER primary pipeline.
    ///
    /// An associated fn rather than a method so its two callers can each
    /// destructure `Self` against their own tracer field — `record_trace`
    /// borrows `self.trace`, `record_dxr_trace` `self.dxr`, and neither can be
    /// reborrowed out of `self` while `dual` and `d3d` are also live.
    ///
    /// ONE BODY, NOT TWO TWINS. Nearly every line below is an ordering rule,
    /// and `dual::Balancer`'s own header records what three hand-copies of a
    /// fifteen-line tick cost when one of them was fixed and the others were
    /// not. `dual::Tracer` exists precisely so this does not have to happen a
    /// second time at a larger scale.
    fn record_split(
        dual: Option<&mut DualState>,
        d3d: &mut D3d,
        prim: dual::Tracer<'_>,
        p: &trace::FrameParams,
        hybrid: bool,
        slot: usize,
    ) -> Result<()> {
        let (rw, rh) = match prim {
            dual::Tracer::Wave(t) => (t.rw, t.rh),
            dual::Tracer::Dxr(d) => (d.rw, d.rh),
        };
        let tg = prim;
        let Some(d) = dual else {
            tg.write_cb(slot, p);
            return tg.record(&d3d.list, slot, p, hybrid);
        };
        // The secondary must store exactly the pack half the PRIMARY's feed
        // will read — resolved per frame, because `wire_feed_add` can run after
        // the secondary was built.
        // What the band must carry is a property of the PRIMARY: whether it
        // has a full-size pack at all, and whether its feed reads the guide
        // half. Both are resolved per frame, because `wire_feed_add` can run
        // after the secondary was built.
        //
        // IT IS THE **AND** OF BOTH SIDES, not the primary alone. A SPACE
        // cycle can leave them disagreeing (the secondary was built at
        // whichever session came first, and `init_dxr`'s `gbuf` need not match
        // `init_trace`'s), and copying a band into a stride-sized dummy is an
        // out-of-bounds `CopyBufferRegion` — which does NOT fault. The whole
        // list fails to Close, and the allocator is then permanently broken:
        // `list Close: The parameter is incorrect` followed by `list Reset`
        // forever. Degrading the frame is cheap; that is not.
        let sec = d.sec.as_ref();
        d.pack = tg.pack_full() && sec.pack_full();
        d.ext = d.pack && tg.gbuf_ext_needed();
        sec.force_gbuf_ext(d.ext);
        sec.force_fsr_sig(d.pack && tg.fsr_sig());
        // The NRD RTGI fold must agree across the band too — an unmirrored
        // secondary packs direct-only dd for its rows, a per-band ReBLUR
        // denoise-semantics seam (no out-of-bounds hazard, just wrong input).
        sec.force_nrd_sig(Some(d.pack && tg.nrd_sig()));
        // ...AND THE ONE-SIDED CASE MUST DENY THE FRAME, not just narrow the
        // payload. The AND above stops the out-of-bounds copy, but a primary
        // that HAS a pack paired with a secondary that has none would still
        // SPLIT: the primary then renders only its own rows, never writing
        // `gbuf`/`gbuf_ext` for the band, while `record_feed` dispatches FULL
        // SCREEN and reads them anyway — stale MVs/depth/albedo from whatever
        // frame last ran unbanded, handed to the upscaler as if current. That
        // is a quieter version of the hole the AND exists to prevent, so it
        // takes the same answer the mixed-arm stand-down does. The reverse
        // pairing is fine: a primary with no pack has no feed to read one.
        let pack_denies = (tg.pack_full() && !sec.pack_full()).then_some(
            "the secondary has no G-buffer pack while the primary's feed reads one",
        );

        // FREEZE THE SPLIT WHILE THE FRAME IS ACCUMULATING, and this is a
        // CORRECTNESS rule rather than the "don't disturb replay" performance
        // one it started as. `accum` is per-device: if a row moves from the
        // primary to the secondary at accumulated frame 10, the secondary's
        // accum for that row holds only frames 10..n while `record_resolve`
        // still divides by n — a dark band, growing darker the later the move.
        // Upscaler sub-modes never accumulate (every frame stores at frame 0),
        // so they rebalance freely; the plain sub-mode's still frames do not.
        // Freezing on `p.replay` instead — the obvious reading — would have
        // frozen a PARKED upscaler session forever, which is the state a
        // user leaves a window in.
        let frozen = p.accumulate && p.frame > 0;
        let rows = d.bal.tick(frozen);
        // MIXED ARMS CANNOT RENDER EVERY FRAME. The DXR pipeline has no
        // hemisphere stage at all, so a DXR partner on an fb frame would draw
        // its band with the bounce tier silently absent — a visibly flatter-lit
        // band appearing the instant H is pressed, with no error anywhere.
        // Degrading the FRAME rather than refusing the session is right: fb is
        // a live toggle, and every non-bounce frame is still splittable.
        let mixed = (prim.arm() != d.sec.arm()).then(|| dual::mixed_denies(p)).flatten();
        if let Some(why) = mixed {
            if !d.said_mixed {
                d.said_mixed = true;
                eprintln!("dual-gpu: {why} — these frames render single-GPU");
            }
        }
        if let Some(why) = pack_denies {
            if !d.said_pack {
                d.said_pack = true;
                eprintln!("dual-gpu: {why} — these frames render single-GPU");
            }
        }
        let denied = mixed.or(pack_denies);
        let rows = if denied.is_some() { 0 } else { rows };
        let (prim_split, sec_split) = trace::TileSplit::for_share(rows, d.bal.depth());
        // ARM BOTH DEVICES OR NEITHER. `set_split` can decline (a depth past
        // the leaf frontier once a small render resolution has moved
        // `depth_full` under it, `--waveviz`), and a half-armed frame is the
        // hole class: the primary skipping rows nothing else renders. So the
        // secondary's half is settled FIRST and the primary is only narrowed
        // once it is; anything short of that puts both back on the whole
        // screen, which is the pre-feature path exactly.
        let band = sec_split.and_then(|sp| {
            let b = sp.row_range(rw, rh)?;
            if !sec.set_region(sp, rw, rh) {
                return None;
            }
            Some(b)
        });
        let armed = band.filter(|_| tg.set_region(prim_split, rw, rh));
        let Some((y0, y1)) = armed else {
            // `TileSplit::ALL` maps to the whole screen on both arms, and that
            // is the one region neither can refuse — so the restore always
            // takes, which is what makes "both or neither" total.
            let _ = tg.set_region(trace::TileSplit::ALL, rw, rh);
            let _ = sec.set_region(trace::TileSplit::ALL, rw, rh);
            // THE ONE EXIT EVERY SINGLE-GPU FRAME TAKES — a deny above, a
            // declined region here, or a zero share from the balancer itself —
            // so demoting the tick once, here, covers all three. Without it
            // `dual_frame_cost` feeds this frame to `observe` as a SPLIT whose
            // secondary cost nothing, which steps the auto-balancer's share up
            // on a frame the secondary sat out entirely. (A zero-share tick is
            // already `Tick::Idle`; marking it again only counts one more idle
            // frame, which is what it is.)
            d.bal.mark_unsplit();
            tg.write_cb(slot, p);
            return tg.record(&d3d.list, slot, p, hybrid);
        };
        d.bal.mark_ran();
        let strides = dual::fed_strides(d.pack, d.ext);
        tg.write_cb(slot, p);
        {
            let hg2 = &mut d.hg;
            sec.write_cb(0, p);
            // A DXR secondary's `record_frame` can fail where the wavefront's
            // cannot, so its error joins the DEFERRED set below rather than
            // being `?`-ed here — the fence is already in flight.
            let mut rec2 = Ok(());
            let v2 = hg2.submit(|l| rec2 = sec.record(l, 0, p, hybrid))?;
            let rec1 = tg.record(&d3d.list, slot, p, hybrid);
            // The primary is now EXECUTING; the wait below overlaps it.
            //
            // THE SPLIT'S ERROR IS DEFERRED PAST THE WAIT, never `?`-ed here.
            // The secondary's fence is in flight from the `submit` above, and
            // `HeadlessGpu::submit` puts the ordering on its caller: returning
            // without waiting leaves the NEXT frame's submit resetting a
            // command allocator whose list may still be executing, which is
            // UB and in practice removes the secondary device. One recoverable
            // frame error (the presenter aborts the frame and main.rs sheds
            // the upscaler) would become a wedged adapter.
            let split = d3d.split_frame(slot);
            let t0 = std::time::Instant::now();
            let waited = hg2.wait(v2);
            // EVERY error above is reported only now, past the wait — see the
            // comment on `split`. `rec1`/`rec2` join it for the same reason.
            rec2?;
            rec1?;
            split?;
            waited?;
            let src = sec.fed_planes();
            let mut rec = Ok(());
            hg2.run(|l| rec = d.xf.record_out(l, &src[..strides.len()], rw, y0, y1))?;
            rec?;
            d.xf.hop(slot, dual::payload_bytes(strides, rw, y1 - y0))?;
            // Everything the secondary cost us: its trace past the launch point
            // plus getting its band across. The balancer sees it next frame,
            // when this frame's own total is finally known.
            d.last_sec_ms = t0.elapsed().as_secs_f32() * 1e3;
        }
        let dst = tg.fed_planes();
        d.xf.record_in(&d3d.list, &dst[..strides.len()], rw, y0, y1, slot)?;
        Ok(())
    }

    /// The balancer's tick and the resulting split on both tracers. `None` =
    /// the secondary sits this frame out, which sets `TileSplit::ALL` on the
    /// primary: the pre-feature path.
    ///
    /// Close the balancer's loop with the frame's measured total. Called by
    /// main.rs once the frame has actually presented, because that total is not
    /// knowable inside a presenter.
    ///
    /// A SOLO frame's cost is the no-split baseline; a dual frame's is what the
    /// split achieved. Both are the same quantity off the same clock, which is
    /// the whole requirement — see `ShareCtl::solo`.
    pub fn dual_frame_cost(&mut self, frame_ms: f32) {
        let Some(d) = self.dual.as_mut() else { return };
        // prim = the frame minus what the secondary held it up for; sec = that
        // hold plus the transfer, which `last_sec_ms` already spans. On a solo
        // or idle tick neither is meaningful and `observe` ignores them.
        let prim = (frame_ms - d.last_sec_ms).max(0.0);
        d.bal.observe(prim, d.last_sec_ms, frame_ms);
        d.last_sec_ms = 0.0;
        self.dual_report();
    }

    /// The dual verdict for the title bar: `(rows, side)`, or None when the
    /// session is not dual at all. A `0` there means the balancer measured the
    /// secondary as not paying and the frame ran the pre-feature path —
    /// correct, and the one state a silent feature is indistinguishable from.
    pub fn dual_status(&self) -> Option<(u32, u32)> {
        self.dual.as_ref().map(|d| (d.bal.rows(), d.bal.side()))
    }

    /// Say the verdict ONCE, the first time the balancer settles at zero after
    /// having actually rendered a split frame. A dual session that silently
    /// stops using its secondary looks exactly like one that never armed, and
    /// those are very different answers.
    pub fn dual_report(&mut self) {
        let Some(d) = self.dual.as_mut() else { return };
        if d.bal.rows() == 0 && d.bal.ran() && !d.said {
            d.said = true;
            let (adopts, _, _, solos) = d.bal.counts();
            eprintln!(
                "dual-gpu: the balancer settled at 0 of {} rows after {adopts} adoptions \
                 and {solos} solo baselines — the secondary does not pay for itself here, \
                 and the session is running the pre-feature single-GPU path",
                d.bal.side()
            );
        }
    }

    pub fn trace_ready(&self) -> bool {
        self.trace.is_some()
    }

    /// The DXR twin of `trace_ready` — the SPACE/F entry guard, so a re-entry
    /// into an already-built pipeline skips the per-press `Dxc::load`
    /// (`init_dxr` is idempotent, but the DXC load at its call site was not:
    /// each one is a LoadLibrary pair that nothing ever frees).
    pub fn dxr_ready(&self) -> bool {
        self.dxr.is_some()
    }

    /// (usage, budget) of the render adapter's LOCAL segment — the mode-cycle
    /// diagnostic. A session whose usage sits at/over budget after the second
    /// tracer lands is in WDDM's silent-demotion regime (the 10-100×-no-error
    /// slowdown class — `adapter::vram_info`'s note).
    pub fn vram_now(&self) -> Option<(u64, u64)> {
        adapter::vram_info(&self.d3d.device)
    }

    /// Build-or-fetch the SHARED scene core — the upload + BLAS/TLAS build
    /// runs once per session (or per scene edit), whichever tracer asks
    /// first; the other gets the cached Rc for free.
    fn ensure_scene_gpu(
        &mut self,
        scene: &crate::scene::Scene,
        bvh: &crate::bvh::Bvh,
        bc7_mode: crate::bc7::Bc7Mode,
    ) -> Result<std::rc::Rc<trace::SceneGpu>> {
        if self.scene_gpu.is_none() {
            let dev = self.d3d.device.clone();
            let core = trace::SceneGpu::new_uploaded(&dev, scene, bvh, &mut self.d3d, bc7_mode)?;
            self.scene_gpu = Some(std::rc::Rc::new(core));
        }
        Ok(self.scene_gpu.clone().expect("just ensured"))
    }

    /// The init-failure eviction arm: when tracer construction fails and no
    /// tracer is live, a cached core is ~gigabytes serving nobody under the
    /// CPU renderer — drop it (the failure latch in main.rs means nothing
    /// will ask again this session).
    fn evict_unused_scene_gpu(&mut self) {
        if self.trace.is_none() && self.dxr.is_none() {
            self.scene_gpu = None;
        }
    }

    /// Drop the resident GPU tracers (--gpu wavefront + DXR) and their
    /// dependents (the quinlight fuse, GPU-resident NPPD) so the next mode
    /// entry rebuilds their SceneGpu/BLAS/TLAS from the current scene — the
    /// runtime frustum-snapshot capture path, which edits the scene live.
    /// Drains the queue first so no in-flight frame references the freed
    /// resources; `init_trace`/`init_dxr` re-read the scene at call time and
    /// re-run `build_quin`. The upscaler contexts (rr/xess/fsr) are kept —
    /// they are resolution-, not scene-, bound.
    pub fn drop_scene_tracers(&mut self) {
        let _ = self.d3d.wait_idle();
        self.nppd_gpu = None;
        self.quin = None;
        self.trace = None;
        self.dxr = None;
        // --dual-gpu: the secondary's core is scene-bound like the primary's,
        // so BOTH halves go — device included, since keeping it would strand a
        // `dual_keep` whose scene core is stale by construction. Its own Drop
        // drains its queue.
        self.dual = None;
        self.dual_keep = None;
        // The shared core is scene-bound too — the next mode entry must
        // re-upload from the edited scene, never serve the stale cache.
        self.scene_gpu = None;
    }

    /// Push a TOD change (`scene::apply_tod`) into the GPU pipelines' cached
    /// base constants — sun rows + SH sky + sky_scale/night. A pipeline built
    /// lazily AFTER the change needs nothing: `init_trace`/`init_dxr` read the
    /// scene at call time.
    pub fn refresh_sky(&mut self, scene: &crate::scene::Scene) {
        if let Some(t) = &mut self.trace {
            t.refresh_sky(scene);
        }
        if let Some(d) = &mut self.dxr {
            d.refresh_sky(scene);
        }
        // The secondary carries its OWN constant buffer, so a sky pushed to the
        // primary alone would seam the frame exactly along the split.
        if let Some(d) = &mut self.dual {
            d.sec.refresh_sky(scene);
        }
    }

    /// Drop the wavefront tracer's structure-replay key. main.rs calls this on
    /// every GPU-present error arm: a present chain that recorded a producing
    /// frame and then aborted it (the list never executed) would leave the key
    /// claiming a structure the GPU never built. Invalidating on ANY present
    /// error covers that from one place — harmless when the frame replayed or
    /// aborted before recording (it only forces the next frame to re-produce).
    pub fn invalidate_replay(&self) {
        if let Some(t) = &self.trace {
            t.invalidate_replay();
        }
        // The secondary keeps its own replay key over its own band, and a
        // present error abandons ITS recorded frame too.
        if let Some(d) = &self.dual {
            d.sec.as_ref().invalidate_replay();
        }
    }

    /// The foliage-sway twin of `invalidate_replay`: forget every animated-
    /// TLAS slot's baked clock. main.rs calls this on every GPU present-error
    /// arm (wavefront AND DXR since v0.2) — a recorded-but-aborted frame
    /// marked its slot baked, but the TLAS build never executed, and the
    /// skip fast-path would bind a TLAS that was never written. Routed
    /// through the SHARED scene core, so one site covers both pipelines
    /// (via `self.dxr` alone, a wavefront-only session was a silent no-op).
    /// Harmless when sway is off/absent.
    pub fn invalidate_sway(&self) {
        if let Some(s) = &self.scene_gpu {
            if let Some(sw) = s.sway.as_ref() {
                sw.invalidate();
            }
        }
        // The secondary's core is a SEPARATE upload with its own animated-TLAS
        // ring, so it needs the same forget — a wavefront-only session already
        // taught this lesson once (routing through `self.dxr` alone was a
        // silent no-op).
        if let Some(sw) = self.dual.as_ref().and_then(|d| d.core.sway.as_ref()) {
            sw.invalidate();
        }
    }

    /// Build the DXR DispatchRays pipeline (the F key / --dxr). Idempotent —
    /// a live pipeline is kept. `(rw, rh)` is the session's fixed DXR trace
    /// resolution (the locked render res when `gbuf` composes with the wired
    /// upscaler, the window size otherwise); scene buffers + BLAS/TLAS come
    /// from the SHARED core (`ensure_scene_gpu` — uploaded once per session,
    /// whichever tracer asks first).
    pub fn init_dxr(
        &mut self,
        dxc: &dxc::Dxc,
        scene: &crate::scene::Scene,
        // Not uploaded (the DXR pipeline never binds the software tree), but
        // read on the CPU as the --blas-split chunking source.
        bvh: &crate::bvh::Bvh,
        rw: u32,
        rh: u32,
        gbuf: bool,
        dn: Option<DnKind>,
        debug: bool,
        bc7_mode: crate::bc7::Bc7Mode,
    ) -> Result<()> {
        if self.dxr.is_some() {
            return Ok(());
        }
        let dev = self.d3d.device.clone();
        let core = self.ensure_scene_gpu(scene, bvh, bc7_mode)?;
        let mut d = match dxr::DxrGpu::new(&dev, dxc, scene, core, rw, rh, gbuf, dn.is_some(), debug)
        {
            Ok(d) => d,
            Err(e) => {
                self.evict_unused_scene_gpu();
                return Err(e);
            }
        };
        if gbuf {
            // Pre-store failure: evict the cached core (the init_trace shape).
            let wired = self
                .wire_session_feed(rw, rh, |kind, targets| d.wire_feed_add(&dev, kind, targets));
            if let Err(e) = wired {
                self.evict_unused_scene_gpu();
                return Err(e);
            }
        }
        // Pre-upscale denoising — arm_denoiser_for, DXR flavor (one shared
        // block, one shared engine instance — see init_trace's call site).
        if let Some(dn) = dn {
            if let Err(e) = self.arm_denoiser_for(&dev, dxc, dn, rw, rh, debug, &mut |t| {
                d.wire_nrd_feed(&dev, t)
            })
            {
                let (tag, hint) = match dn {
                    DnKind::Nrd(_) => (
                        "nrd",
                        " (install-prerequisites.bat nrd builds the SDK; --no-nrd silences this)",
                    ),
                    DnKind::Frd => ("frd", ""),
                };
                eprintln!("{tag}: unavailable ({e}); running plain GPU upscaling{hint}");
                d.nrd = None;
            }
        }
        wire_tonemap_src(
            &self.d3d.device,
            &self.passes,
            &self.bloom,
            &self.autoexp,
            &d.hdr,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            tonemap::SRV_SLOT_DXR,
        );
        self.dxr = Some(d);
        // The fuse, over whatever engines the chain wired (--quinlight only).
        self.build_quin(dxc)?;
        // --dual-gpu, the `init_trace` block verbatim: a DXR session splits too
        // (the band is `DxrGpu::set_band`, and `TileSplit::row_range` speaks
        // its units by construction). Built LAST and never fatal — the session
        // is a working single-GPU one at this point, and a second adapter that
        // cannot be opened is a missing optimisation, not a failed session.
        if let Some(share) = self.dual_want {
            if self.dual.is_none() {
                match self.build_dual(dxc, scene, bvh, rw, rh, share, gbuf, debug, bc7_mode) {
                    Ok(()) => {}
                    Err(e) => {
                        eprintln!("dual-gpu: unavailable ({e}) — running single-GPU");
                        // Latch it off: without this every SPACE re-entry
                        // retries a full second scene upload + BLAS build.
                        self.dual_want = None;
                    }
                }
            }
        }
        Ok(())
    }

    /// One DXR frame: constants -> DispatchRays -> resolve -> tonemap ->
    /// present. `samples` divides the accumulation (the present_trace shape).
    pub fn present_dxr(&mut self, p: &trace::FrameParams, samples: u32) -> Result<()> {
        self.waveviz_src.set(WvSrc::Dxr);
        crate::zone!("present-dxr");
        if self.dxr.is_none() {
            return Err("DXR pipeline not initialized".into());
        }
        let slot = self.d3d.begin_frame()?;
        // --dual-gpu goes through the ONE site; the band copy it may record
        // lands on the frame list BEFORE the full-screen resolve below, which
        // is what that resolve depends on.
        if let Err(e) = self.record_dxr_trace(p, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        let d = self.dxr.as_ref().expect("checked above");
        d.record_resolve(&self.d3d.list, slot, samples);
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_DXR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// Re-present the last resolved DXR frame without tracing — the
    /// converged-idle path (present_hold's contract: record_resolve left hdr
    /// in PIXEL_SHADER_RESOURCE).
    pub fn present_dxr_hold(&mut self) -> Result<()> {
        self.waveviz_src.set(WvSrc::Dxr);
        if self.dxr.is_none() {
            return Err("DXR pipeline not initialized".into());
        }
        let slot = self.d3d.begin_frame()?;
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_DXR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// Screenshot path for DXR mode (the image exists only on the GPU).
    pub fn read_dxr_output(&mut self) -> Result<(Vec<u32>, usize, usize)> {
        let Some(d) = &self.dxr else {
            return Err("DXR pipeline not initialized".into());
        };
        let output = d.hdr.clone();
        self.read_hdr_output(output)
    }

    /// DLSS-RR fed by the DXR pipeline (the `--dxr` default in a DLSS
    /// session): one command list = DispatchRays -> feed (pack -> the 7 SL
    /// input planes) -> the SL sequence -> tonemap(SRV_SLOT_RR) -> present.
    /// The whole list executes on the session queue, which IS the SL proxy
    /// queue whenever RR is live — the present_trace_rr contract verbatim;
    /// record_frame's trailing global UAV barrier fences the pack + accum
    /// for the feed.
    pub fn present_dxr_rr(
        &mut self,
        p: &trace::FrameParams,
        fc: &dlss::FrameConstants,
        frame_idx: u32,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::Dxr);
        crate::zone!("present-dxr-rr");
        let _ = frame_idx; // was the SL frame token's index; the raw evaluate needs none
        if self.dxr.is_none() || self.ngxrr.is_none() || self.rr.is_none() {
            return Err("DXR pipeline + DLSS-RR not both initialized".into());
        }
        let slot = self.d3d.begin_frame()?;
        {
            let d3d = &mut self.d3d;
            let d = self.dxr.as_ref().unwrap();
            // The session res is fixed at init; fc must agree (it drives the
            // extent tags SL derives the ratio from).
            if (fc.rw as u32, fc.rh as u32) != (d.rw, d.rh) {
                d3d.abort_frame();
                return Err(format!(
                    "frame constants {}x{} != DXR res {}x{}",
                    fc.rw, fc.rh, d.rw, d.rh
                ));
            }
        }
        // --dual-gpu goes through the ONE site, and it must precede the feed:
        // `record_feed` dispatches FULL SCREEN and reads the secondary's rows
        // of the pack, so the band has to have landed on this list first.
        if let Err(e) = self.record_dxr_trace(p, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        {
            let d3d = &mut self.d3d;
            let d = self.dxr.as_ref().unwrap();
            if let Err(e) = d.record_feed(&d3d.list, slot) {
                d3d.abort_frame();
                return Err(e);
            }
        }
        {
            let d3d = &mut self.d3d;
            let nx = self.ngxrr.as_ref().unwrap();
            let Some(feat) = self.rr_feature.as_ref() else {
                d3d.abort_frame();
                return Err("DLSSD feature not created".into());
            };
            let rr = self.rr.as_ref().unwrap();
            unsafe {
                d3d.list.ResourceBarrier(&[transition(
                    &rr.output,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            if let Err(e) = rr_ngx_sequence(nx, feat, rr, &d3d.list, fc) {
                d3d.abort_frame();
                return Err(e);
            }
            unsafe {
                d3d.list.ResourceBarrier(&[transition(
                    &rr.output,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                )]);
            }
        }
        // Raw-NGX frame generation owns the tail when armed (pair-present).
        if self.fg_n.is_some() {
            return self.ngxfg_tail(slot, fc, &p.fireflies, &p.clouds);
        }
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_RR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// XeSS-SR fed by the DXR pipeline (`--dxr --xess`): DispatchRays ->
    /// feed -> XeSS upscale -> tonemap(SRV_SLOT_XESS) -> present — the
    /// present_trace_xess chain minus the NPPD split (NPPD stays a
    /// wavefront-session composition).
    pub fn present_dxr_xess(
        &mut self,
        p: &trace::FrameParams,
        jitter: (f32, f32),
        reset: bool,
        fc: &dlss::FrameConstants,
        frame_ms: f32,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::Dxr);
        crate::zone!("present-dxr-xess");
        if self.dxr.is_none() || self.xess.is_none() {
            return Err("DXR pipeline + XeSS not both initialized".into());
        }
        self.nrd_shed_cleanup()?;
        let slot = self.d3d.begin_frame()?;
        // --dual-gpu goes through the ONE site, and it must precede the feed:
        // `record_feed` dispatches FULL SCREEN and reads the secondary's rows
        // of the pack, so the band has to have landed on this list first.
        if let Err(e) = self.record_dxr_trace(p, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        let nrd_armed = self.nrd_gpu.is_some() && self.dxr.as_ref().is_some_and(|d| d.nrd_armed());
        let mut nrd_ok = nrd_armed;
        {
            let d3d = &mut self.d3d;
            let d = self.dxr.as_ref().unwrap();
            if nrd_ok {
                // The shared NRD step, DXR flavor.
                let dev = d3d.device.clone();
                nrd_ok = Self::nrd_frame_step(
                    &mut self.nrd_gpu,
                    &mut self.nrd_prev,
                    &mut self.nrd_frame_idx,
                    &mut self.nrd_hist_valid,
                    &dev,
                    &d3d.list,
                    slot,
                    d.rw,
                    d.rh,
                    fc,
                    jitter,
                    reset,
                    &|| d.record_nrd_pack(&d3d.list, slot),
                    &|| d.record_nrd_out(&d3d.list, slot),
                );
            }
            // The fold: an NRD frame needs NO engine feed dispatch — the
            // guides were written by the folded cs_nrd_pack, the color by
            // cs_nrd_out. Only the shed/plain arm feeds.
            let feed = if nrd_ok { Ok(()) } else { d.record_feed(&d3d.list, slot) };
            if let Err(e) = feed {
                d3d.abort_frame();
                return Err(e);
            }
        }
        if nrd_armed && !nrd_ok {
            // The shed, FLAG-only (see present_trace_xess's twin comment):
            // in-flight lists still reference NrdGpu — nrd_shed_cleanup
            // drains and frees at the next presenter entry.
            self.nrd_shed = true;
            self.nrd_hist_valid = false;
        }
        {
            let d3d = &mut self.d3d;
            let d = self.dxr.as_ref().unwrap();
            let x = self.xess.as_ref().unwrap();
            unsafe {
                d3d.list.ResourceBarrier(&[transition(
                    &x.res.output,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            let (c, m, dep) = x.res.input_ptrs();
            let params = crate::xess::XessD3d12ExecuteParams {
                color_texture: c,
                velocity_texture: m,
                depth_texture: dep,
                exposure_scale_texture: std::ptr::null_mut(),
                responsive_pixel_mask_texture: std::ptr::null_mut(),
                output_texture: x.res.output.as_raw(),
                jitter_offset_x: crate::xess::JITTER_SIGN * jitter.0,
                jitter_offset_y: crate::xess::JITTER_SIGN * jitter.1,
                exposure_scale: 1.0,
                reset_history: reset as u32,
                input_width: d.rw,
                input_height: d.rh,
                input_color_base: Default::default(),
                input_motion_vector_base: Default::default(),
                input_depth_base: Default::default(),
                input_responsive_mask_base: Default::default(),
                reserved0: Default::default(),
                output_color_base: Default::default(),
                descriptor_heap: std::ptr::null_mut(),
                descriptor_heap_offset: 0,
            };
            {
                let _ev = pix::scope(&d3d.list, c"xess-eval");
                if let Err(e) = x.ctx.execute(d3d.list.as_raw(), &params) {
                    d3d.abort_frame();
                    return Err(e);
                }
            }
            unsafe {
                d3d.list.ResourceBarrier(&[transition(
                    &x.res.output,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                )]);
            }
        }
        self.xefg_prepare(fc, frame_ms);
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch,
        // restoring whatever list state the XeSS dispatch left behind.
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_XESS, 1.0);
        self.xefg_end_frame(slot)
    }

    /// FSR 3.1 fed by the DXR pipeline: DispatchRays -> feed (pack -> the 3
    /// upscaler input planes, on-GPU) -> one FSR 3.1 upscale dispatch ->
    /// tonemap(SRV_SLOT_FSR) -> present. The XeSS-chain shape — FSR3's
    /// input set IS the XeSS trio — with the ffx dispatch in the middle;
    /// never an SL session (the chain wires FSR only on the native device).
    pub fn present_dxr_fsr3(
        &mut self,
        p: &trace::FrameParams,
        fc: &dlss::FrameConstants,
        frame_ms: f32,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::Dxr);
        crate::zone!("present-dxr-fsr3");
        if self.dxr.is_none() || self.fsr.is_none() {
            return Err("DXR pipeline + FSR not both initialized".into());
        }
        debug_assert!(self.ngxrr.is_none());
        self.nrd_shed_cleanup()?;
        let slot = self.d3d.begin_frame()?;
        {
            let d3d = &mut self.d3d;
            let d = self.dxr.as_ref().unwrap();
            // The session res is fixed at init; fc must agree (it names the
            // upscale dispatch's renderSize sub-rect).
            if (fc.rw as u32, fc.rh as u32) != (d.rw, d.rh) {
                d3d.abort_frame();
                return Err(format!(
                    "frame constants {}x{} != DXR res {}x{}",
                    fc.rw, fc.rh, d.rw, d.rh
                ));
            }
        }
        // --dual-gpu goes through the ONE site, and it must precede the feed:
        // `record_feed` dispatches FULL SCREEN and reads the secondary's rows
        // of the pack, so the band has to have landed on this list first.
        if let Err(e) = self.record_dxr_trace(p, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        let nrd_armed = self.nrd_gpu.is_some() && self.dxr.as_ref().is_some_and(|d| d.nrd_armed());
        let mut nrd_ok = nrd_armed;
        {
            let d3d = &mut self.d3d;
            let d = self.dxr.as_ref().unwrap();
            if nrd_ok {
                // The shared NRD step (fc carries this presenter's
                // jitter/reset).
                let dev = d3d.device.clone();
                nrd_ok = Self::nrd_frame_step(
                    &mut self.nrd_gpu,
                    &mut self.nrd_prev,
                    &mut self.nrd_frame_idx,
                    &mut self.nrd_hist_valid,
                    &dev,
                    &d3d.list,
                    slot,
                    d.rw,
                    d.rh,
                    fc,
                    fc.jitter,
                    fc.reset,
                    &|| d.record_nrd_pack(&d3d.list, slot),
                    &|| d.record_nrd_out(&d3d.list, slot),
                );
            }
            // The fold: an NRD frame needs NO engine feed dispatch — the
            // guides were written by the folded cs_nrd_pack, the color by
            // cs_nrd_out. Only the shed/plain arm feeds.
            let feed = if nrd_ok { Ok(()) } else { d.record_feed(&d3d.list, slot) };
            if let Err(e) = feed {
                d3d.abort_frame();
                return Err(e);
            }
        }
        if nrd_armed && !nrd_ok {
            // The shed, FLAG-only (see present_trace_xess's twin comment):
            // in-flight lists still reference NrdGpu — nrd_shed_cleanup
            // drains and frees at the next presenter entry.
            self.nrd_shed = true;
            self.nrd_hist_valid = false;
        }
        {
            let fs = self.fsr.as_ref().unwrap();
            if let Err(e) = Self::record_fsr3_upscale(fs, &self.d3d.list, fc, frame_ms, None) {
                self.d3d.abort_frame();
                return Err(e);
            }
        }
        self.fg_set_frame_ms(frame_ms);
        if let Some(fs) = &self.fsr {
            let (dep, mv, scale) = fs.res.fg_inputs(fc.rw as u32, fc.rh as u32);
            self.fg_prepare(&self.d3d.list, fc, dep, mv, scale);
        }
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch,
        // restoring whatever list state the ffx dispatch left behind.
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_FSR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// Whether the GPU-resident NPPD stage came up (`--gpu --nppd`; the J
    /// toggle is only honored when this is true — wiring is init-time).
    pub fn nppd_gpu_ready(&self) -> bool {
        self.nppd_gpu.is_some()
    }

    /// One fully GPU-resident frame: constants -> trace (wavefront quadtree,
    /// or the vanilla reference when `hybrid` is false — the R-key A/B) ->
    /// resolve -> tonemap -> present. `samples` divides the accumulation.
    pub fn present_trace(&mut self, p: &trace::FrameParams, samples: u32, hybrid: bool) -> Result<()> {
        self.waveviz_src.set(WvSrc::Trace);
        crate::zone!("present-trace");
        if self.trace.is_none() {
            return Err("GPU tracer not initialized".into());
        }
        let slot = self.d3d.begin_frame()?;
        if let Err(e) = self.record_trace(p, hybrid, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        let tg = self.trace.as_ref().unwrap();
        tg.record_resolve(&self.d3d.list, slot, samples);
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_GPU, 1.0);
        self.d3d.end_frame(slot)
    }

    /// Re-present the last resolved frame without tracing or accumulating —
    /// the converged-idle path (re-adding a pinned-seed sample while resolve
    /// divides by a pinned count would brighten the image without bound).
    /// `record_resolve` leaves hdr in PIXEL_SHADER_RESOURCE, so the tonemap
    /// blit is all that's needed.
    pub fn present_hold(&mut self) -> Result<()> {
        self.waveviz_src.set(WvSrc::Trace);
        if self.trace.is_none() {
            return Err("GPU tracer not initialized".into());
        }
        let slot = self.d3d.begin_frame()?;
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_GPU, 1.0);
        self.d3d.end_frame(slot)
    }

    /// The one NRD frame step all four NRD-capable presenters share
    /// (wavefront/DXR × XeSS/FSR3): bridge pack → NRD's own passes →
    /// delta-form recompose, plus the prev-matrices/frame-index/latch
    /// bookkeeping. Returns whether NRD produced this frame's color (false ⇒
    /// the caller runs the normal color-writing feed, and — if a denoiser
    /// exists — sheds it for the session AFTER its borrows end). Free fn
    /// over explicit fields: the presenters hold field-split borrows this
    /// must thread through, not own.
    #[allow(clippy::too_many_arguments)]
    fn nrd_frame_step(
        nrd_gpu: &mut Option<DnGpu>,
        nrd_prev: &mut Option<(crate::dlss::CamMatrices, (f32, f32))>,
        nrd_frame_idx: &mut u32,
        nrd_hist_valid: &mut bool,
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        slot: usize,
        rw: u32,
        rh: u32,
        fc: &dlss::FrameConstants,
        jitter: (f32, f32),
        reset: bool,
        pack: &dyn Fn() -> Result<()>,
        out: &dyn Fn() -> Result<()>,
    ) -> bool {
        let Some(dn) = nrd_gpu.as_mut() else { return false };
        let mats = crate::dlss::CamMatrices {
            world_to_view: fc.world_to_view,
            view_to_clip: fc.view_to_clip,
        };
        let (pm, pj) = nrd_prev.unwrap_or((mats, jitter));
        let hist_reset = reset || !*nrd_hist_valid;
        let tag = dn.tag();
        let step = pack()
            .and_then(|()| match dn {
                DnGpu::Nrd(ng) => {
                    let cs = nrd_gpu::common_settings(
                        &mats,
                        &pm,
                        jitter,
                        pj,
                        rw,
                        rh,
                        fc.far,
                        *nrd_frame_idx,
                        hist_reset,
                    );
                    let rs = nrd_gpu::reblur_settings();
                    ng.record(device, list, slot, &cs, &rs)
                }
                DnGpu::Frd(fg) => {
                    // The kernel's camera facts, derived once per frame: the
                    // translation step (the specular parallax numerator),
                    // the forward vector (the n·v proxy), and the
                    // world→pixel projection scale (m11 · rh/2 — the blur
                    // radius converter). O = the view matrix's inverse
                    // translation; forward = its z row — glam col-major,
                    // rigid, so all are exact.
                    let o_cur = mats.world_to_view.inverse().w_axis.truncate();
                    let o_prev = pm.world_to_view.inverse().w_axis.truncate();
                    let step = (o_cur - o_prev).length();
                    let fwd = mats.world_to_view.row(2).truncate().normalize_or_zero();
                    let proj = mats.view_to_clip.y_axis.y * rh as f32 * 0.5;
                    fg.record(list, slot, hist_reset, fc.far, proj, step, fwd.to_array())
                }
            })
            .and_then(|()| out());
        match step {
            Ok(()) => {
                *nrd_prev = Some((mats, jitter));
                *nrd_frame_idx = nrd_frame_idx.wrapping_add(1);
                *nrd_hist_valid = true;
                true
            }
            Err(e) => {
                eprintln!("{tag}: frame failed ({e}); shedding — plain upscaling continues");
                false
            }
        }
    }

    /// Arm the pre-upscale denoiser (NRD or FRD — `dn` picks) for one
    /// tracer — the ONE arming block both GPU arms share (init_trace and
    /// init_dxr used to carry verbatim copies). Reuses the session's single
    /// engine instance, building it on first arming: both arms trace at the
    /// session-locked res, and sharing the instance is what keeps a SPACE/F
    /// cycle's MEMOIZED arm wired at live planes — a second instance would
    /// strand the first arm's NRD_FEED_SET descriptors on dropped pools
    /// (packing into ghosts, denoising never-written inputs). `wire` points
    /// the caller's tracer at the planes — engine-blind, since both engines
    /// carry the same plane contract. An Err leaves that tracer unarmed
    /// (`nrd_armed()` false — it runs plain); the caller sheds loudly.
    #[allow(clippy::too_many_arguments)]
    fn arm_denoiser_for(
        &mut self,
        dev: &ID3D12Device,
        dxc: &dxc::Dxc,
        dn: DnKind,
        rw: u32,
        rh: u32,
        debug: bool,
        wire: &mut dyn FnMut(
            &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
        ) -> Result<()>,
    ) -> Result<()> {
        if self.nrd_shed {
            return Err("the denoiser was shed earlier this session".into());
        }
        // The engine whose planes the bridge owns: XeSS or the FSR 3.1
        // upscaler (the byte-identical-trio pair; `fsr` is the SESSION state,
        // either flavor — `fsr3` is quinlight's extra engine). RR/FSR-RR
        // already denoise and never reach here. cs_nrd_out owns COLOR; since
        // the fold, cs_nrd_pack owns the MVEC/DEPTH guides too (the retired
        // record_feed_nrd's job), so all three wire into the NRD set. The
        // resources are CLONED (COM AddRef) so no borrow of self survives
        // into the build below.
        let [(color_res, color_fmt), (mvec_res, mvec_fmt), (depth_res, depth_fmt)] =
            if let Some(x) = &self.xess {
                x.res.plane_resources().map(|(r, f)| (r.clone(), f))
            } else if let Some(FsrRes::Up(res)) = self.fsr.as_ref().map(|f| &f.res) {
                res.plane_resources().map(|(r, f)| (r.clone(), f))
            } else {
                return Err("the pre-upscale denoiser composes with an XeSS or FSR3 session".into());
            };
        match &self.nrd_gpu {
            Some(g) if (g.size()) == (rw, rh) && g.matches(&dn) => {}
            Some(g) if !g.matches(&dn) => {
                // Kind is session-constant (one Opts), so a mismatch here is
                // a lifecycle hole, not a path — refuse like the size arm.
                return Err(format!("{} armed, this arm asks for the other engine", g.tag()));
            }
            Some(g) => {
                // The desync BACKSTOP, not a normal path: both arms share
                // --lock-res, and resize_output drops the instance with the
                // tracers (it fired on every maximize until it did — the
                // window moving moves the locked render res, which the old
                // "unreachable" reasoning missed). A mismatch reaching here
                // now means a NEW lifecycle hole; refuse rather than rebuild
                // (the rebuild is the stranding bug).
                let (gw, gh) = g.size();
                return Err(format!(
                    "denoiser armed at {gw}x{gh}, this arm traces {rw}x{rh}"
                ));
            }
            None => {
                // The one success line per engine — every other nrd:/frd:
                // line is a failure path, and an armed session used to be
                // indistinguishable from an unarmed one (the user-report
                // class this fixes). Once per session: both arms share the
                // instance, so the SPACE/F re-arms land in the reuse arm
                // above.
                let built = match dn {
                    DnKind::Nrd(dir) => {
                        let g = nrd_gpu::NrdGpu::new(dev, dir, rw, rh)?;
                        // The dir is printed because the standard and
                        // --nrd-perf DLLs are version-indistinguishable
                        // (LibraryDesc has no perf bit) — this line is the
                        // only record of which one loaded.
                        eprintln!(
                            "nrd: armed — ReBLUR pre-upscale denoising at {rw}x{rh} ({dir})"
                        );
                        DnGpu::Nrd(g)
                    }
                    DnKind::Frd => {
                        let g = frd_gpu::FrdGpu::new(dev, dxc, rw, rh, debug)?;
                        // Names the COMPILE arm honestly: phases B/C build
                        // fp32 kernels regardless of the OPTIONS4 probe —
                        // "capable" is the phase-D promise, not the running
                        // precision.
                        eprintln!(
                            "frd: armed — recurrent pre-upscale denoising at {rw}x{rh} \
                             (fp32 kernels; fp16 {})",
                            if g.fp16 { "capable" } else { "unavailable" }
                        );
                        DnGpu::Frd(g)
                    }
                };
                self.nrd_gpu = Some(built);
                self.nrd_hist_valid = false;
                self.nrd_prev = None;
                self.nrd_frame_idx = 0;
            }
        }
        let ng = self.nrd_gpu.as_ref().unwrap();
        use windows::Win32::Graphics::Dxgi::Common::*;
        // Set NRD_FEED_SET: the bridge's registers — u16/u18/u19 are the
        // ENGINE's color/depth/mvec planes (cs_nrd_out owns color, the folded
        // cs_nrd_pack owns the two guides; no separate engine feed runs in an
        // NRD frame), u26 is NRD's linear view-Z (moved off u18 for the fold
        // — nrd_bridge.hlsl's register map is the lockstep twin).
        wire(&[
            (16, &color_res, color_fmt),
            (17, ng.plane_in_nr(), DXGI_FORMAT_R10G10B10A2_UNORM),
            (18, &depth_res, depth_fmt),
            (19, &mvec_res, mvec_fmt),
            (20, ng.plane_out_spec(), DXGI_FORMAT_R16G16B16A16_FLOAT),
            (23, ng.plane_in_mv(), DXGI_FORMAT_R16G16B16A16_FLOAT),
            (24, ng.plane_in_diff(), DXGI_FORMAT_R16G16B16A16_FLOAT),
            (25, ng.plane_in_spec(), DXGI_FORMAT_R16G16B16A16_FLOAT),
            (26, ng.plane_in_viewz(), DXGI_FORMAT_R32_FLOAT),
            (27, ng.plane_out_diff(), DXGI_FORMAT_R16G16B16A16_FLOAT),
        ])
    }

    /// The deferred half of the NRD shed (see the `nrd_shed` field comment):
    /// at a presenter entry nothing new references NrdGpu's objects, one
    /// wait_idle covers every submitted list that still does, and the drop +
    /// wiring clear (which releases the bridge planes AND disarms the
    /// tracers' fsr_sig/gbuf_ext_needed nrd terms, so the pack stops storing
    /// for a consumer that no longer exists) become safe.
    fn nrd_shed_cleanup(&mut self) -> Result<()> {
        if !self.nrd_shed || self.nrd_gpu.is_none() {
            return Ok(());
        }
        self.d3d.wait_idle()?;
        self.nrd_gpu = None;
        self.nrd_hist_valid = false;
        if let Some(tg) = self.trace.as_mut() {
            tg.clear_nrd_wired();
        }
        if let Some(d) = self.dxr.as_mut() {
            d.clear_nrd_wired();
        }
        Ok(())
    }

    /// XeSS-SR fed by the GPU-resident tracer (`--gpu --xess`): trace
    /// (wavefront or reference) -> feed (pack -> input planes, on-GPU — the
    /// CPU upload of the xr.rs path does not exist here) -> XeSS upscale ->
    /// tonemap(SRV_SLOT_XESS) -> present. `jitter` is the renderer's sample
    /// offset; xess::JITTER_SIGN settles the reported sign, nowhere else.
    /// With `nppd` (and the stage built), the frame SPLITS around the
    /// inference: list A = trace + NPPD staging (pack + warp), submitted
    /// without a Present; ORT's DML work lands on the same queue behind it;
    /// list B = feed(guides) + denoised-color crop + XeSS + tonemap +
    /// the one Present. Queue order is the only synchronization.
    pub fn present_trace_xess(
        &mut self,
        p: &trace::FrameParams,
        hybrid: bool,
        jitter: (f32, f32),
        reset: bool,
        nppd: bool,
        fc: &dlss::FrameConstants,
        frame_ms: f32,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::Trace);
        crate::zone!("present-trace-xess");
        if self.trace.is_none() || self.xess.is_none() {
            return Err("GPU tracer + XeSS not both initialized".into());
        }
        self.nrd_shed_cleanup()?;
        let nppd_on = nppd && self.nppd_gpu.is_some();
        // Armed = this ARM's tracer is wired too — a session whose other arm
        // armed NRD runs this one plain instead of tripping the shed.
        let nrd_armed =
            self.nrd_gpu.is_some() && self.trace.as_ref().is_some_and(|t| t.nrd_armed());
        let mut nrd_ok = nrd_armed;
        let slot = self.d3d.begin_frame()?;
        // The trace goes through the ONE site, so --dual-gpu reaches this arm
        // without the presenter knowing anything about it.
        if let Err(e) = self.record_trace(p, hybrid, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        {
            // Field-split borrows: the recorder reads the tracer, abort needs
            // d3d mutably.
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            if nppd_on {
                // A reset frame zeroes the warped-state input instead of
                // warping (the graph rewrites the state buffer either way).
                let state_valid = self.nppd_state_valid && !reset;
                if let Err(e) = tg.record_nppd_pre(&d3d.list, slot, state_valid) {
                    d3d.abort_frame();
                    return Err(e);
                }
                d3d.split_frame(slot)?;
                if let Err(e) = self.nppd_gpu.as_mut().unwrap().run() {
                    d3d.abort_frame();
                    return Err(e);
                }
                self.nppd_state_valid = true;
            }
            if nrd_ok {
                // The NRD (ReBLUR) pre-upscale denoise: pack → NRD's own
                // passes → delta-form recompose into the XeSS color plane,
                // all on the one list (pure compute — no split_frame; the
                // NPPD slot minus the ORT run). A failure sheds NRD for the
                // session and the frame continues plain (the normal feed
                // below overwrites the color plane from accum).
                let dev = d3d.device.clone();
                nrd_ok = Self::nrd_frame_step(
                    &mut self.nrd_gpu,
                    &mut self.nrd_prev,
                    &mut self.nrd_frame_idx,
                    &mut self.nrd_hist_valid,
                    &dev,
                    &d3d.list,
                    slot,
                    tg.rw,
                    tg.rh,
                    fc,
                    jitter,
                    reset,
                    &|| tg.record_nrd_pack(&d3d.list, slot),
                    &|| tg.record_nrd_out(&d3d.list, slot),
                );
            }
            // The fold: an NRD frame needs NO engine feed dispatch — the
            // guides were written by the folded cs_nrd_pack, the color by
            // cs_nrd_out. Only the shed/plain arm feeds.
            let feed =
                if nrd_ok { Ok(()) } else { tg.record_feed(&d3d.list, slot, nppd_on) };
            if let Err(e) = feed {
                d3d.abort_frame();
                return Err(e);
            }
        }
        if nrd_armed && !nrd_ok {
            // The shed, FLAG-only: in-flight lists (and, on a mid-record
            // failure, THIS frame's) still reference NrdGpu's heaps/PSOs/
            // pools — nrd_shed_cleanup drains and frees at the next presenter
            // entry. Never rebuilt this session (the nppd-gpu shape).
            self.nrd_shed = true;
            self.nrd_hist_valid = false;
        }
        if !nppd_on {
            // J-off (or a run failure upstream): the next NPPD frame starts
            // from a reset state.
            self.nppd_state_valid = false;
        }
        {
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            let x = self.xess.as_ref().unwrap();
            if let Err(e) = Self::record_xess_eval(x, &d3d.list, tg.rw, tg.rh, jitter, reset) {
                d3d.abort_frame();
                return Err(e);
            }
        }
        self.xefg_prepare(fc, frame_ms);
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch,
        // restoring whatever list state the XeSS dispatch left behind.
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_XESS, 1.0);
        self.xefg_end_frame(slot)
    }

    /// Record EVERY wired engine's upscale into its own output texture, then the
    /// registered-consensus fuse over them — the shared middle of the two
    /// --quinlight present arms (wavefront and DXR). The caller has already
    /// recorded the trace and ONE feed dispatch per engine.
    ///
    /// All of it rides one command list, in FIFO order: each engine reads the
    /// same G-buffer-derived planes (they are read-only consumers, so no barrier
    /// between them) and writes its own window-res output; the fuse then reads
    /// those N outputs as SRVs and writes the one image the tonemap presents.
    /// Everything — the raw-NGX RR evaluate included — executes on the one
    /// native queue, exactly as present_trace_rr's list does.
    ///
    /// An engine that errors here fails the frame — by this point it is wired,
    /// fed, and counted in the fuse's N, so silently skipping it would fuse a
    /// stale output.
    /// `sky_sh` rides through to an FSR4-RR engine's composite (the AO signal's
    /// remodulation factor — the one sky made it directional).
    #[allow(clippy::too_many_arguments)]
    fn record_quin_engines(
        &mut self,
        rw: u32,
        rh: u32,
        jitter: (f32, f32),
        reset: bool,
        fc: &dlss::FrameConstants,
        prev_pos: Option<glam::Vec3A>,
        frame_idx: u32,
        frame_ms: f32,
        sky_sh: &crate::sh::Sh9,
    ) -> Result<()> {
        // DLSS-RR (engine 0 when the raw-NGX session is live).
        if let (Some(nx), Some(feat), Some(rr)) =
            (self.ngxrr.as_ref(), self.rr_feature.as_ref(), self.rr.as_ref())
        {
            let d3d = &self.d3d;
            unsafe {
                d3d.list.ResourceBarrier(&[transition(
                    &rr.output,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            rr_ngx_sequence(nx, feat, rr, &d3d.list, fc)?;
            unsafe {
                d3d.list.ResourceBarrier(&[transition(
                    &rr.output,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                )]);
            }
        }
        // The ffx flavors. FSR4-RR denoises + upscales; FSR 3.1 upscales only —
        // and does it from the XeSS trio when XeSS is also wired (one feed, two
        // readers), else from its own planes.
        let shared = self.xess.as_ref().map(|x| x.res.plane_resources());
        for fs in [self.fsr.as_ref(), self.fsr3.as_ref()].into_iter().flatten() {
            match &fs.res {
                FsrRes::Rr(_) => Self::record_fsr_rr_sequence(
                    fs, &self.d3d.list, fc, prev_pos, frame_idx, frame_ms, sky_sh,
                )?,
                FsrRes::Up(_) => {
                    Self::record_fsr3_upscale(fs, &self.d3d.list, fc, frame_ms, shared.as_ref())?
                }
            }
        }
        // XeSS.
        if let Some(x) = self.xess.as_ref() {
            Self::record_xess_eval(x, &self.d3d.list, rw, rh, jitter, reset)?;
        }
        // The fuse: N engine outputs -> one image at SRV_SLOT_QUIN.
        self.quin.as_ref().ok_or("quinlight fuse not built")?.record(&self.d3d.list);
        Ok(())
    }

    /// The --quinlight ffx prepare inputs: the depth/MV planes actually
    /// WRITTEN this frame. An FSR4-RR flavor always has its own fed
    /// eleven-plane set (reversed-Z clip depth + UV-delta MVs, mv_scale =
    /// SIGN*(rw,rh) — the standalone present_trace_fsr_rr pick, and what the
    /// 4.x ML provider pairs with). A SHARED FSR 3.1 upscales from the XeSS
    /// trio and its OWN planes are never fed (wire_session_feed skips them)
    /// — handing fg_prepare FsrRes::Up's planes there would be STALE data;
    /// the XeSS trio is byte-identical to the Up trio (R32F reversed-Z clip
    /// depth, RG16F pixel MVs, NPSR rest), so mv_scale = UPSCALE_MV_SIGN
    /// pixels. No XeSS wired = the Up planes WERE fed (the else arm).
    fn quin_ffx_fg_inputs(
        &self,
        rw: u32,
        rh: u32,
    ) -> Option<(&ID3D12Resource, &ID3D12Resource, [f32; 2])> {
        for fs in [self.fsr.as_ref(), self.fsr3.as_ref()].into_iter().flatten() {
            if matches!(fs.res, FsrRes::Rr(_)) {
                return Some(fs.res.fg_inputs(rw, rh));
            }
        }
        if let Some(x) = &self.xess {
            let p = x.res.plane_resources();
            return Some((
                p[2].0,
                p[1].0,
                [crate::fsr::UPSCALE_MV_SIGN.0, crate::fsr::UPSCALE_MV_SIGN.1],
            ));
        }
        [self.fsr.as_ref(), self.fsr3.as_ref()]
            .into_iter()
            .flatten()
            .next()
            .map(|fs| fs.res.fg_inputs(rw, rh))
    }

    /// The quin arms' presentation tail — exactly one FG family can be armed
    /// (fg_n / fg / fg_x are mutually exclusive by construction in `new`),
    /// and each gets its per-frame contract over the FUSED present: raw NGX
    /// pair-presents (interpolating quin.output via `ngxfg_target`), ffx FI
    /// gets its PrepareV2 from the planes actually fed, XeSS-FG its
    /// tag+marker prepare. The fuse<->plain toggles and the pause-menu hold
    /// keep working through the same funnel handshake / reset latches every
    /// other arm uses (a plain frame simply never prepares).
    fn quin_fg_tail(
        &mut self,
        slot: usize,
        fc: &dlss::FrameConstants,
        frame_ms: f32,
        ff: &crate::fireflies::Fireflies,
        cl: &crate::clouds::Clouds,
    ) -> Result<()> {
        // Raw NGX owns the tail when armed (pair-present; no handshake).
        if self.fg_n.is_some() {
            return self.ngxfg_tail(slot, fc, ff, cl);
        }
        // ffx FI: frame_ms + PrepareV2 with the planes actually fed.
        self.fg_set_frame_ms(frame_ms);
        if self.fg.is_some() {
            if let Some((dep, mv, scale)) = self.quin_ffx_fg_inputs(fc.rw as u32, fc.rh as u32)
            {
                self.fg_prepare(&self.d3d.list, fc, dep, mv, scale);
            }
        }
        // XeSS-FG: prepare BEFORE the funnel (no-op when fg_x is None);
        // xefg_end_frame consumes `prepared` after the XeLL present markers
        // and degenerates to a plain end_frame otherwise — the universal
        // tail for the ffx/none cases too.
        self.xefg_prepare(fc, frame_ms);
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_QUIN, 1.0);
        self.xefg_end_frame(slot)
    }

    /// The registered-consensus fuse fed by the GPU-resident tracer
    /// (`--gpu --quinlight`): trace -> one feed dispatch per engine -> every
    /// wired upscaler -> the LK-registered winsorized fuse -> tonemap
    /// (SRV_SLOT_QUIN) -> present (with the session's FG family's per-frame
    /// contract in the tail — see `quin_fg_tail`). One command list; one
    /// Present, or two under raw-NGX pair-present.
    #[allow(clippy::too_many_arguments)]
    pub fn present_trace_quin(
        &mut self,
        p: &trace::FrameParams,
        hybrid: bool,
        jitter: (f32, f32),
        reset: bool,
        fc: &dlss::FrameConstants,
        prev_pos: Option<glam::Vec3A>,
        frame_idx: u32,
        frame_ms: f32,
        sky_sh: &crate::sh::Sh9,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::Trace);
        crate::zone!("present-trace-quin");
        if self.trace.is_none() || self.quin.is_none() {
            return Err("GPU tracer + quinlight fuse not both initialized".into());
        }
        let slot = self.d3d.begin_frame()?;
        if let Err(e) = self.record_trace(p, hybrid, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        let (rw, rh) = {
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            if let Err(e) = tg.record_feed(&d3d.list, slot, false) {
                d3d.abort_frame();
                return Err(e);
            }
            (tg.rw, tg.rh)
        };
        if let Err(e) =
            self.record_quin_engines(rw, rh, jitter, reset, fc, prev_pos, frame_idx, frame_ms, sky_sh)
        {
            self.d3d.abort_frame();
            return Err(e);
        }
        self.quin_fg_tail(slot, fc, frame_ms, &p.fireflies, &p.clouds)
    }

    /// The DXR twin of `present_trace_quin` (the `--dxr` default + --quinlight):
    /// DispatchRays instead of the wavefront, same engines, same fuse.
    #[allow(clippy::too_many_arguments)]
    pub fn present_dxr_quin(
        &mut self,
        p: &trace::FrameParams,
        jitter: (f32, f32),
        reset: bool,
        fc: &dlss::FrameConstants,
        prev_pos: Option<glam::Vec3A>,
        frame_idx: u32,
        frame_ms: f32,
        sky_sh: &crate::sh::Sh9,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::Dxr);
        crate::zone!("present-dxr-quin");
        if self.dxr.is_none() || self.quin.is_none() {
            return Err("DXR pipeline + quinlight fuse not both initialized".into());
        }
        let slot = self.d3d.begin_frame()?;
        // --dual-gpu goes through the ONE site, and it must precede the feed:
        // `record_feed` dispatches FULL SCREEN and reads the secondary's rows
        // of the pack, so the band has to have landed on this list first.
        if let Err(e) = self.record_dxr_trace(p, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        let (rw, rh) = {
            let d3d = &mut self.d3d;
            let d = self.dxr.as_ref().unwrap();
            if let Err(e) = d.record_feed(&d3d.list, slot) {
                d3d.abort_frame();
                return Err(e);
            }
            (d.rw, d.rh)
        };
        if let Err(e) =
            self.record_quin_engines(rw, rh, jitter, reset, fc, prev_pos, frame_idx, frame_ms, sky_sh)
        {
            self.d3d.abort_frame();
            return Err(e);
        }
        self.quin_fg_tail(slot, fc, frame_ms, &p.fireflies, &p.clouds)
    }

    /// Screenshot path for --quinlight: the fused image exists only on the GPU.
    pub fn read_quin_output(&mut self) -> Result<Vec<u32>> {
        let Some(q) = &self.quin else {
            return Err("quinlight fuse not initialized".into());
        };
        let output = q.output.clone();
        Ok(self.read_hdr_output(output)?.0)
    }

    /// FSR 3.1 fed by the GPU-resident tracer (`--gpu` with FSR3 wired):
    /// trace (wavefront or reference) -> feed (pack -> the 3 input planes,
    /// on-GPU) -> one FSR 3.1 upscale dispatch -> tonemap(SRV_SLOT_FSR) ->
    /// present. present_trace_xess minus the NPPD composition (XeSS-only).
    pub fn present_trace_fsr3(
        &mut self,
        p: &trace::FrameParams,
        hybrid: bool,
        fc: &dlss::FrameConstants,
        frame_ms: f32,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::Trace);
        crate::zone!("present-trace-fsr3");
        if self.trace.is_none() || self.fsr.is_none() {
            return Err("GPU tracer + FSR not both initialized".into());
        }
        debug_assert!(self.ngxrr.is_none());
        self.nrd_shed_cleanup()?;
        let slot = self.d3d.begin_frame()?;
        {
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            if (fc.rw as u32, fc.rh as u32) != (tg.rw, tg.rh) {
                d3d.abort_frame();
                return Err(format!(
                    "frame constants {}x{} != trace res {}x{}",
                    fc.rw, fc.rh, tg.rw, tg.rh
                ));
            }
        }
        // The trace goes through the ONE site (`record_trace`), which is what
        // puts --dual-gpu in every wavefront presenter at once.
        if let Err(e) = self.record_trace(p, hybrid, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        let nrd_armed =
            self.nrd_gpu.is_some() && self.trace.as_ref().is_some_and(|t| t.nrd_armed());
        let mut nrd_ok = nrd_armed;
        {
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            if nrd_ok {
                // The NRD pre-upscale denoise — the shared step (fc carries
                // this presenter's jitter/reset).
                let dev = d3d.device.clone();
                nrd_ok = Self::nrd_frame_step(
                    &mut self.nrd_gpu,
                    &mut self.nrd_prev,
                    &mut self.nrd_frame_idx,
                    &mut self.nrd_hist_valid,
                    &dev,
                    &d3d.list,
                    slot,
                    tg.rw,
                    tg.rh,
                    fc,
                    fc.jitter,
                    fc.reset,
                    &|| tg.record_nrd_pack(&d3d.list, slot),
                    &|| tg.record_nrd_out(&d3d.list, slot),
                );
            }
            // The fold: an NRD frame needs NO engine feed dispatch — the
            // guides were written by the folded cs_nrd_pack, the color by
            // cs_nrd_out. Only the shed/plain arm feeds.
            let feed = if nrd_ok { Ok(()) } else { tg.record_feed(&d3d.list, slot, false) };
            if let Err(e) = feed {
                d3d.abort_frame();
                return Err(e);
            }
        }
        if nrd_armed && !nrd_ok {
            // The shed, FLAG-only (see present_trace_xess's twin comment):
            // in-flight lists still reference NrdGpu — nrd_shed_cleanup
            // drains and frees at the next presenter entry.
            self.nrd_shed = true;
            self.nrd_hist_valid = false;
        }
        {
            let fs = self.fsr.as_ref().unwrap();
            if let Err(e) = Self::record_fsr3_upscale(fs, &self.d3d.list, fc, frame_ms, None) {
                self.d3d.abort_frame();
                return Err(e);
            }
        }
        self.fg_set_frame_ms(frame_ms);
        if let Some(fs) = &self.fsr {
            let (dep, mv, scale) = fs.res.fg_inputs(fc.rw as u32, fc.rh as u32);
            self.fg_prepare(&self.d3d.list, fc, dep, mv, scale);
        }
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch,
        // restoring whatever list state the ffx dispatch left behind.
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_FSR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// Screenshot path for GPU-trace mode (the image exists only on the GPU).
    /// Returns (pixels, w, h) — the tracer's hdr is render-res in upscaler
    /// sessions.
    pub fn read_trace_output(&mut self) -> Result<(Vec<u32>, usize, usize)> {
        let Some(tg) = &self.trace else {
            return Err("GPU tracer not initialized".into());
        };
        let output = tg.hdr.clone();
        self.read_hdr_output(output)
    }

    fn read_trace_buffer(&mut self, res: &ID3D12Resource, size: usize) -> Result<Vec<u8>> {
        let rb = d3d12::ReadbackBuffer::new(&self.d3d.device, size)?;
        let src = res.clone();
        self.d3d.run_once(|list| unsafe {
            let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            list.ResourceBarrier(&[transition(&src, ua, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyBufferRegion(&rb.resource, 0, &src, 0, size as u64);
            list.ResourceBarrier(&[transition(&src, D3D12_RESOURCE_STATE_COPY_SOURCE, ua)]);
        })?;
        let mut ptr = std::ptr::null_mut();
        unsafe { rb.resource.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("Map: {e}"))?;
        let out = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) }.to_vec();
        unsafe { rb.resource.Unmap(0, None) };
        Ok(out)
    }

    /// C key in GPU mode: the on-GPU analog of render::verify — trace the
    /// current view unjittered through both the wavefront quadtree and the
    /// vanilla reference, compare per pixel (same intersector both sides —
    /// the exact-zero gates), and report. Clobbers accum/tbuf/info; the
    /// caller resets the accumulation.
    pub fn verify_trace(
        &mut self,
        cam: &crate::camera::CamBasis,
        q: crate::shade::Quality,
        clouds: crate::clouds::Clouds,
        fireflies: crate::fireflies::Fireflies,
        sway_time: Option<f32>,
    ) -> Result<String> {
        let (tbuf, info, counters, px) = {
            let Some(tg) = &self.trace else {
                return Err("GPU tracer not initialized".into());
            };
            (tg.tbuf.clone(), tg.info.clone(), tg.counters.clone(), (tg.rw * tg.rh) as usize)
        };
        let p = trace::FrameParams {
            sway_prev_time: None,
            cam: *cam,
            frame: 0,
            accumulate: true,
            jitter: false,
            frame_jitter: None,
            prev_cam: None,
            q: crate::shade::Quality { fb: crate::shade::FrustumBounce::OFF, ..q },
            verify: false,
            spp: 1,
            probe_sample: 0,
            // The live session state: both kernels read the same CB, so the
            // same-seed comparison holds whatever the sky is doing — and the
            // C verify then exercises the cloud code the session actually runs.
            clouds,
            fireflies,
            // The session's clock: BOTH lists bind the same (possibly
            // animated) TLAS — record_wavefront rebuilds slot 0's ring TLAS
            // and stashes the clock; record_reference reuses the stash, and
            // its own record_rebuild would be a free bit-equal skip. The
            // comparison stays same-intersector, same-TLAS.
            sway_time,
            // verify_trace calls record_wavefront directly (not record_frame),
            // so this is dead — but the field must be set.
            replay: false,
        };
        {
            // Field-split borrow: run_once needs d3d mutably, the recorder
            // reads the tracer.
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            // Drain the queue BEFORE touching CB slot 0: a presented frame
            // still in flight may be reading it (run_once waits idle too,
            // but only after this write would have raced it).
            d3d.wait_idle()?;
            tg.write_cb(0, &p);
            d3d.run_once(|l| tg.record_wavefront(l, 0, &p, true))?;
        }
        let wave_t = self.read_trace_buffer(&tbuf, px * 4)?;
        let wave_info = self.read_trace_buffer(&info, px * 4)?;
        // CTR_TOTAL: the tail holds the FR_WIDTH slots (unread unless armed).
        let ctrs = self.read_trace_buffer(&counters, trace::CTR_TOTAL as usize * 4)?;
        {
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            d3d.run_once(|l| tg.record_reference(l, 0))?;
        }
        let ref_t = self.read_trace_buffer(&tbuf, px * 4)?;

        let f = |b: &[u8], i: usize| f32::from_le_bytes(b[i * 4..][..4].try_into().unwrap());
        let u = |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..][..4].try_into().unwrap());
        let (mut false_sky, mut overshoot, mut extra, mut sentinel) = (0u64, 0u64, 0u64, 0u64);
        let mut max_rel = 0.0f32;
        for i in 0..px {
            if u(&wave_info, i) == 0xffff_ffff {
                sentinel += 1;
            }
            let (rt, wt) = (f(&ref_t, i), f(&wave_t, i));
            match (rt.is_finite(), wt.is_finite()) {
                (true, true) => {
                    let rel = (wt - rt) / rt.max(1e-6);
                    max_rel = max_rel.max(rel.abs());
                    if rel > 1e-4 {
                        overshoot += 1;
                    }
                }
                (true, false) => false_sky += 1,
                (false, true) => extra += 1,
                _ => {}
            }
        }
        let ok = false_sky == 0 && overshoot == 0 && extra == 0 && sentinel == 0;
        // sky-px: the empty-space proof's product (pixels resolved with ZERO
        // rays) as a fraction of the frame — the C key is the only way to
        // read it for a scene --spin can't load (THE WORLD).
        let sky_px = u(&ctrs, trace::CTR_SKY_PX as usize) as u64;
        // FR_WIDTH: append the per-kernel compiled-width line — the C key is
        // the interactive read of the report (the sky-px precedent).
        let width_line = if trace::width_probe_on() {
            let c: Vec<u32> = (0..trace::CTR_TOTAL as usize).map(|i| u(&ctrs, i)).collect();
            format!(" | {}", trace::format_width_report(&c))
        } else {
            String::new()
        };
        Ok(format!(
            "gpu verify ({px} px): false-sky {false_sky} | tmin-overshoot {overshoot} | hybrid-extra {extra} | unwritten {sentinel} | max rel t err {max_rel:.2e} | tiles: {} splits, {} sky, {} leaves, {} blocked | sky-px {} ({:.1}%){width_line} -> {}",
            u(&ctrs, trace::CTR_SPLIT as usize),
            u(&ctrs, trace::CTR_SKY as usize),
            u(&ctrs, trace::CTR_LEAF as usize),
            u(&ctrs, trace::CTR_BLOCKED as usize),
            sky_px,
            sky_px as f64 * 100.0 / px as f64,
            if ok { "OK" } else { "FAILED" },
        ))
    }

    /// Raw-NGX RR session + feature + planes live => the evaluate is
    /// available (M4).
    pub fn dlss_ready(&self) -> bool {
        self.ngxrr.is_some() && self.rr_feature.is_some() && self.rr.is_some()
    }

    /// Which chain level this session wired — derived from the live state
    /// (at most one upscaler exists per session by the probe's construction,
    /// EXCEPT under --quinlight, where several do and the fuse presents), so it
    /// can never disagree with the contexts actually held.
    pub fn wired(&self) -> WiredUpscaler {
        // Tested first: a quinlight session holds several engines, and any of
        // the branches below would also match one of them.
        if self.quin.is_some() {
            WiredUpscaler::Quin
        } else if self.dlss_ready() {
            WiredUpscaler::Rr
        } else if let Some(f) = &self.fsr {
            match f.res.flavor() {
                crate::fsr::Flavor::Fsr4Rr => WiredUpscaler::Fsr4,
                crate::fsr::Flavor::Fsr3 => WiredUpscaler::Fsr3,
            }
        } else if self.xess.is_some() {
            WiredUpscaler::Xess
        } else {
            WiredUpscaler::Plain
        }
    }

    /// The engines a --quinlight session fuses, in CHAIN order (which is also
    /// fuse order, so engine 0 is the highest level present — a DENOISING one
    /// wherever the box has one, which is what the default anchor wants).
    ///
    /// Their window-res RGBA16F outputs are the fuse's SRVs. Ordinary sessions
    /// return at most one, which is why `quin_engines().len() < 2` is the
    /// "nothing to fuse" test.
    fn quin_engines(&self) -> (Vec<&ID3D12Resource>, quin::Engines) {
        let mut res: Vec<&ID3D12Resource> = Vec::new();
        let mut names: Vec<&'static str> = Vec::new();
        if let Some(r) = &self.rr {
            if self.ngxrr.is_some() {
                res.push(&r.output);
                names.push("dlss-rr");
            }
        }
        for f in [self.fsr.as_ref(), self.fsr3.as_ref()].into_iter().flatten() {
            res.push(f.res.upscaled());
            names.push(match f.res.flavor() {
                crate::fsr::Flavor::Fsr4Rr => "fsr4-rr",
                crate::fsr::Flavor::Fsr3 => "fsr3",
            });
        }
        if let Some(x) = &self.xess {
            res.push(&x.res.output);
            names.push("xess");
        }
        (res, quin::Engines(names))
    }

    /// Build the fuse over whatever engines came up — called from
    /// init_trace/init_dxr, which already hold the DXC the PSO needs. A single
    /// engine is not a consensus: the session then presents that engine
    /// directly (its own chain level), and `wired()` reports it.
    fn build_quin(&mut self, dxc: &dxc::Dxc) -> Result<()> {
        let Some((anchor_opt, debug)) = self.quin_cfg else {
            return Ok(());
        };
        // Idempotent (the init_dxr precedent): the fuse reads the UPSCALER
        // outputs, which are session objects independent of which tracer
        // feeds them, so one fuse serves both. With the SPACE mode cycle the
        // second tracer's LAZY init lands here mid-session, and replacing a
        // fuse whose PSO/heap prior frames may still reference is exactly
        // the in-flight-release class D3D12 forbids.
        if self.quin.is_some() {
            return Ok(());
        }
        let (w, h) = (self.d3d.width, self.d3d.height);
        let (engines, names) = self.quin_engines();
        if engines.len() < 2 {
            eprintln!(
                "quinlight: only {} engine(s) came up ({}) — nothing to fuse; \
                 presenting that level directly",
                engines.len(),
                if names.0.is_empty() { "none".into() } else { names.names() }
            );
            return Ok(());
        }
        // Default: anchor on a DENOISING engine (DLSS-RR, else FSR4-RR) when the
        // box wired one — the anchor is never warped, so it is the engine whose
        // spatial frame survives, and a ray-reconstruction image is both the
        // cleanest reference for the LK solve and the one worth keeping. An
        // explicit --quin-anchor always wins.
        // An out-of-range --quin-anchor is a typo, and silently clamping it would
        // report someone else's engine as the anchor they asked for — an A/B run
        // against a bad index would read as a legitimate result. Say so, then
        // clamp (the shader clamps too; this is about the user, not soundness).
        let last = engines.len() as u32 - 1;
        if let Some(a) = anchor_opt.filter(|a| *a > last) {
            eprintln!(
                "quinlight: --quin-anchor {a} is past the last engine ({last}: {}) — \
                 clamping to {last}",
                names.names()
            );
        }
        let anchor = anchor_opt.unwrap_or_else(|| names.default_anchor()).min(last);
        let q =
            quin::Quin::new(&self.d3d.device, dxc, &engines, names.clone(), anchor, w, h, debug)?;
        // A tonemap SOURCE, so it goes through wire_tonemap_src: the fused image
        // is what the glare pyramid must read (bloom's source SRV per slot), or
        // a --quinlight session would present without highlights.
        wire_tonemap_src(
            &self.d3d.device,
            &self.passes,
            &self.bloom,
            &self.autoexp,
            &q.output,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            tonemap::SRV_SLOT_QUIN,
        );
        eprintln!(
            "quinlight: fusing {} engines [{}], anchor {} ({}) — registered consensus \
             (LK + warp + winsorized mean){}",
            engines.len(),
            names.names(),
            names.0[anchor as usize],
            match anchor_opt {
                Some(_) => "--quin-anchor",
                None if quin::DENOISING.contains(&names.0[anchor as usize]) =>
                    "default: the denoising engine",
                None => "default: engine 0",
            },
            if engines.len() == 2 {
                "; NOTE 2 engines => the winsorized mean IS a plain mean (see gpu/quin.rs)"
            } else {
                ""
            }
        );
        // The measured caveat, said out loud: the reduce can only pull the clean
        // anchor toward the noisy ones, so a mixed stack is quieter than the TAA
        // engines but NOISIER than the denoiser alone. Not a failure — a property
        // of the mean — but a user who wired this expecting a free win should be
        // told, not left to discover it.
        if names.mixed_noise(anchor) {
            eprintln!(
                "quinlight: NOTE the anchor denoises but [{}] does not — the winsorized mean \
                 pulls the anchor toward the noisier engines, so expect MORE temporal noise \
                 than {} alone (measured; see CLAUDE.md). --quinlight --no-dlss fuses peers.",
                names
                    .0
                    .iter()
                    .filter(|n| !quin::DENOISING.contains(n))
                    .copied()
                    .collect::<Vec<_>>()
                    .join(" + "),
                names.0[anchor as usize],
            );
        }
        self.quin = Some(q);
        Ok(())
    }

    /// The optimal render resolution for DLSS mode (SL's Quality-mode
    /// optimal size; == window size when the query fell back to DLAA) —
    /// the fixed fallback when the DRS range is degenerate.
    pub fn rr_render_res(&self) -> Option<(u32, u32)> {
        self.rr.as_ref().map(|r| r.opt)
    }

    /// DLSS-RR input-resolution range: (optimal, min, max) — the same shape
    /// as `xess_res_range`, feeding the shared DRS controller. min == max
    /// means the driver reported no dynamic range (DRS off).
    pub fn rr_res_range(&self) -> Option<((u32, u32), (u32, u32), (u32, u32))> {
        self.rr.as_ref().map(|r| (r.opt, r.min, r.max))
    }

    /// XeSS live => the dynamic-res upscale path is available.
    pub fn xess_ready(&self) -> bool {
        self.xess.is_some()
    }

    /// The fused engines, for the HUD ("dlss-rr + xess + fsr3").
    pub fn quin_names(&self) -> Option<String> {
        self.quin.as_ref().map(|q| q.names.names())
    }

    /// Will this session present through the fuse? True iff `--quinlight` AND at
    /// least two engines actually came up. The fuse itself is built later (in
    /// init_trace/init_dxr, which hold the DXC), but the ANSWER is fixed at
    /// probe time — main.rs needs it before then, to pick the render res that
    /// satisfies every engine at once.
    pub fn quin_planned(&self) -> bool {
        self.quin_cfg.is_some() && self.quin_engines().0.len() > 1
    }

    /// The render-res range every wired engine accepts: the INTERSECTION of
    /// their SDK ranges (max of the mins, min of the maxes). A --quinlight frame
    /// is traced ONCE and fed to all of them, so the one res must be legal for
    /// each — and `wire_session_feed`'s per-engine range checks enforce it.
    /// None = no engine wired.
    pub fn quin_res_range(&self) -> Option<((u32, u32), (u32, u32), (u32, u32))> {
        let mut it = [
            self.rr.as_ref().filter(|_| self.ngxrr.is_some()).map(|r| (r.opt, r.min, r.max)),
            self.fsr.as_ref().map(|f| (f.opt, f.min, f.max)),
            self.fsr3.as_ref().map(|f| (f.opt, f.min, f.max)),
            self.xess.as_ref().map(|x| (x.opt, x.min, x.max)),
        ]
        .into_iter()
        .flatten();
        let first = it.next()?;
        Some(it.fold(first, |(o, lo, hi), (o2, lo2, hi2)| {
            (
                (o.0.min(o2.0), o.1.min(o2.1)),
                (lo.0.max(lo2.0), lo.1.max(lo2.1)),
                (hi.0.min(hi2.0), hi.1.min(hi2.1)),
            )
        }))
    }

    /// XeSS input-resolution range: (optimal, min, max). Every dynamic frame
    /// must trace inside [min, max]; the controller starts at optimal.
    pub fn xess_res_range(&self) -> Option<((u32, u32), (u32, u32), (u32, u32))> {
        self.xess.as_ref().map(|x| (x.opt, x.min, x.max))
    }

    /// FSR live => a working upscale chain exists (which flavor is
    /// `fsr_flavor`'s answer).
    pub fn fsr_ready(&self) -> bool {
        self.fsr.is_some()
    }

    /// Which FSR pipeline this session initialized: `Fsr4Rr` (Ray
    /// Regeneration + FSR4) or `Fsr3` (3.1 upscale-only). None when FSR
    /// never came up.
    pub fn fsr_flavor(&self) -> Option<crate::fsr::Flavor> {
        self.fsr.as_ref().map(|f| f.res.flavor())
    }

    /// FSR input-resolution range: (seed, min, max) — the same shape as
    /// `xess_res_range`, feeding the shared DRS controller.
    pub fn fsr_res_range(&self) -> Option<((u32, u32), (u32, u32), (u32, u32))> {
        self.fsr.as_ref().map(|f| (f.opt, f.min, f.max))
    }

    /// FSR mode: upload the G-buffer + signal sub-rects, Ray Regeneration
    /// denoise (two chained signal dispatches), remodulation composite, FSR4
    /// upscale to the window-res output, tonemap, present. Everything both
    /// ffx dispatches see is in render-res space (their `renderSize` names
    /// this frame's sub-rect); only the upscaled output is window-sized.
    /// `prev_pos` is the previous frame's camera position (the denoiser's
    /// cameraPositionDelta); `frame_ms` feeds the upscaler's frameTimeDelta.
    #[allow(clippy::too_many_arguments)]
    pub fn present_fsr(
        &mut self,
        g: &dlss::GBufs,
        f: &crate::fsr::FsrBufs,
        fc: &dlss::FrameConstants,
        prev_pos: Option<glam::Vec3A>,
        frame_idx: u32,
        frame_ms: f32,
        sky_sh: &crate::sh::Sh9,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::None);
        crate::zone!("present-fsr");
        let Some(fs) = &self.fsr else {
            return Err("FSR not initialized".into());
        };
        let FsrRes::Rr(res) = &fs.res else {
            return Err("present_fsr on an FSR 3.1 session (present_fsr3 owns that frame)".into());
        };
        // Release-build cover for record_upload's dimension debug_asserts.
        let (rw, rh) = (fc.rw as u32, fc.rh as u32);
        if (g.rw, g.rh) != (fc.rw, fc.rh) || (f.rw, f.rh) != (fc.rw, fc.rh) {
            return Err(format!(
                "FSR frame is {}x{}, buffers {}x{} / {}x{}",
                fc.rw, fc.rh, g.rw, g.rh, f.rw, f.rh
            ));
        }
        if rw < fs.min.0 || rh < fs.min.1 || rw > fs.max.0 || rh > fs.max.1 {
            return Err(format!(
                "FSR frame {}x{} outside render range {}x{}..{}x{}",
                rw, rh, fs.min.0, fs.min.1, fs.max.0, fs.max.1
            ));
        }
        let (near, far) = (fc.near, fc.far);
        let slot = self.d3d.begin_frame()?;
        if let Err(e) = res.record_upload(&self.d3d, slot, g, f, fc.rw, fc.rh, near, far) {
            self.d3d.abort_frame();
            return Err(e);
        }

        if let Err(e) =
            Self::record_fsr_rr_sequence(fs, &self.d3d.list, fc, prev_pos, frame_idx, frame_ms, sky_sh)
        {
            self.d3d.abort_frame();
            return Err(e);
        }

        self.fg_set_frame_ms(frame_ms);
        if let Some(fs) = &self.fsr {
            let (dep, mv, scale) = fs.res.fg_inputs(fc.rw as u32, fc.rh as u32);
            self.fg_prepare(&self.d3d.list, fc, dep, mv, scale);
        }
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch,
        // restoring whatever list state the ffx dispatches left behind.
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_FSR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// The Ray Regeneration + FSR4 middle shared by the CPU-fed
    /// (`present_fsr`) and GPU-fed (`present_trace_fsr_rr`/
    /// `present_dxr_fsr_rr`) chains: denoise (one chained ffxDispatch) ->
    /// remodulation composite -> FSR4 upscale, with all barriers. Inputs are
    /// whatever filled the nine planes (CPU sub-rect upload or the
    /// cs_feed_fsr_rr kernel — both leave them resting in
    /// NON_PIXEL_SHADER_RESOURCE). The caller owns the open frame (and
    /// aborts it on Err).
    /// `sky_sh` is the scene's sky in order-2 SH — the AO signal's remodulation
    /// factor, which the composite pass evaluates per pixel against the normals
    /// plane. It used to be a compile-time constant (`shade::AMBIENT`); the one
    /// sky makes it directional, so it has to travel with the frame.
    #[allow(clippy::too_many_arguments)]
    fn record_fsr_rr_sequence(
        fs: &FsrState,
        list: &ID3D12GraphicsCommandList,
        fc: &dlss::FrameConstants,
        prev_pos: Option<glam::Vec3A>,
        frame_idx: u32,
        frame_ms: f32,
        sky_sh: &crate::sh::Sh9,
    ) -> Result<()> {
        let FsrRes::Rr(res) = &fs.res else {
            return Err("FSR4-RR sequence on an FSR 3.1 session".into());
        };
        let (rw, rh) = (fc.rw as u32, fc.rh as u32);
        if rw < fs.min.0 || rh < fs.min.1 || rw > fs.max.0 || rh > fs.max.1 {
            return Err(format!(
                "FSR frame {}x{} outside render range {}x{}..{}x{}",
                rw, rh, fs.min.0, fs.min.1, fs.max.0, fs.max.1
            ));
        }
        // Ray Regeneration: signals in UAV state, one ffxDispatch with the
        // common desc + both signal descs chained (built in the shim).
        res.barrier_denoise_begin(list);
        let r = res.denoise_res();
        let dd_desc = ffx::FfxShimDenoiseDesc {
            cmdlist: list.as_raw(),
            // The dispatch's signal set must equal the context's creation set
            // (the ffx header's if-and-only-if rule) — one constant, both.
            signal_flags: ffx_sys::SIGNALS,
            linear_depth: r.depth_lin,
            motion_vectors: r.mvec,
            normals: r.normals,
            specular_albedo: r.spec_alb,
            diffuse_albedo: r.diff_alb,
            dd_in: r.dd_in,
            dd_out: r.dd_out,
            ds_in: r.ds_in,
            ds_out: r.ds_out,
            ao_in: r.ao_in,
            ao_out: r.ao_out,
            is_in: r.is_in,
            is_out: r.is_out,
            // Our MV plane is already PreviousUV - CurrentUV with the depth
            // delta in B (converted at the fill site) — the header's unit
            // scale.
            mv_scale: [1.0, 1.0, 1.0],
            // fc.jitter is the renderer's sample offset in pixels; the ffx
            // polarity knob is fsr::JITTER_SIGN, nowhere else.
            jitter: [crate::fsr::JITTER_SIGN * fc.jitter.0, crate::fsr::JITTER_SIGN * fc.jitter.1],
            cam_pos_delta: match prev_pos {
                Some(p) => v3(p - fc.pos),
                None => [0.0; 3],
            },
            // FfxApiMatrix4x4 is row-major storage with ROW-vector
            // convention; per its own compatibility table, glam's
            // column-major/column-vector matrices memcpy DIRECTLY — the
            // deliberate contrast with SL's row_major() transpose.
            view: fc.world_to_view.to_cols_array(),
            projection: fc.view_to_clip.to_cols_array(),
            depth_bounds_min: fc.near,
            depth_bounds_max: fc.far,
            render_w: rw,
            render_h: rh,
            frame_index: frame_idx,
            reset: fc.reset as i32,
            non_gamma_albedo: 0, // albedos are sqrt-encoded (fsr.rs wire)
        };
        {
            let _ev = pix::scope(list, c"fsr-denoise");
            fs.ctx.denoise(&dd_desc)?;
        }
        res.barrier_denoise_end(list);

        // Remodulate (binds from scratch — the post-ffx state restore).
        res.record_composite(list, rw, rh, sky_sh);

        // FSR4 upscale: composite -> window-res output. The shared MV plane
        // holds UV-deltas here, so the scale multiplies the render dims back
        // in to hand FSR pixel-space MVs (polarity knob: fsr::UPSCALE_MV_SIGN).
        res.barrier_upscale_begin(list);
        let up_desc = Self::fsr_upscale_desc(
            list.as_raw(),
            res.upscale_res(),
            fc,
            fs.max,
            frame_ms,
            [
                crate::fsr::UPSCALE_MV_SIGN.0 * rw as f32,
                crate::fsr::UPSCALE_MV_SIGN.1 * rh as f32,
            ],
        );
        {
            let _ev = pix::scope(list, c"fsr-upscale");
            fs.ctx.upscale(&up_desc)?;
        }
        res.barrier_upscale_end(list);
        Ok(())
    }

    /// Ray Regeneration + FSR4 fed by the GPU-resident tracer: trace -> feed
    /// (pack + sig -> the nine FSR planes, on-GPU) -> denoise -> composite ->
    /// upscale -> tonemap(SRV_SLOT_FSR) -> present. Never an SL session.
    #[allow(clippy::too_many_arguments)]
    pub fn present_trace_fsr_rr(
        &mut self,
        p: &trace::FrameParams,
        hybrid: bool,
        fc: &dlss::FrameConstants,
        prev_pos: Option<glam::Vec3A>,
        frame_idx: u32,
        frame_ms: f32,
        sky_sh: &crate::sh::Sh9,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::Trace);
        crate::zone!("present-trace-fsr-rr");
        if self.trace.is_none() || self.fsr.is_none() {
            return Err("GPU tracer + FSR not both initialized".into());
        }
        debug_assert!(self.ngxrr.is_none());
        let slot = self.d3d.begin_frame()?;
        {
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            if (fc.rw as u32, fc.rh as u32) != (tg.rw, tg.rh) {
                d3d.abort_frame();
                return Err(format!(
                    "frame constants {}x{} != trace res {}x{}",
                    fc.rw, fc.rh, tg.rw, tg.rh
                ));
            }
        }
        // The trace goes through the ONE site (`record_trace`), which is what
        // puts --dual-gpu in every wavefront presenter at once.
        if let Err(e) = self.record_trace(p, hybrid, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        {
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            if let Err(e) = tg.record_feed(&d3d.list, slot, false) {
                d3d.abort_frame();
                return Err(e);
            }
        }
        {
            let fs = self.fsr.as_ref().unwrap();
            if let Err(e) =
                Self::record_fsr_rr_sequence(fs, &self.d3d.list, fc, prev_pos, frame_idx, frame_ms, sky_sh)
            {
                self.d3d.abort_frame();
                return Err(e);
            }
        }
        self.fg_set_frame_ms(frame_ms);
        if let Some(fs) = &self.fsr {
            let (dep, mv, scale) = fs.res.fg_inputs(fc.rw as u32, fc.rh as u32);
            self.fg_prepare(&self.d3d.list, fc, dep, mv, scale);
        }
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_FSR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// Ray Regeneration + FSR4 fed by the DXR pipeline — the
    /// `present_dxr_fsr3` shape with the full denoise sequence in the middle.
    #[allow(clippy::too_many_arguments)]
    pub fn present_dxr_fsr_rr(
        &mut self,
        p: &trace::FrameParams,
        fc: &dlss::FrameConstants,
        prev_pos: Option<glam::Vec3A>,
        frame_idx: u32,
        frame_ms: f32,
        sky_sh: &crate::sh::Sh9,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::Dxr);
        crate::zone!("present-dxr-fsr-rr");
        if self.dxr.is_none() || self.fsr.is_none() {
            return Err("DXR pipeline + FSR not both initialized".into());
        }
        debug_assert!(self.ngxrr.is_none());
        let slot = self.d3d.begin_frame()?;
        {
            let d3d = &mut self.d3d;
            let d = self.dxr.as_ref().unwrap();
            if (fc.rw as u32, fc.rh as u32) != (d.rw, d.rh) {
                d3d.abort_frame();
                return Err(format!(
                    "frame constants {}x{} != DXR res {}x{}",
                    fc.rw, fc.rh, d.rw, d.rh
                ));
            }
        }
        // --dual-gpu goes through the ONE site, and it must precede the feed:
        // `record_feed` dispatches FULL SCREEN and reads the secondary's rows
        // of the pack, so the band has to have landed on this list first.
        if let Err(e) = self.record_dxr_trace(p, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        {
            let d3d = &mut self.d3d;
            let d = self.dxr.as_ref().unwrap();
            if let Err(e) = d.record_feed(&d3d.list, slot) {
                d3d.abort_frame();
                return Err(e);
            }
        }
        {
            let fs = self.fsr.as_ref().unwrap();
            if let Err(e) =
                Self::record_fsr_rr_sequence(fs, &self.d3d.list, fc, prev_pos, frame_idx, frame_ms, sky_sh)
            {
                self.d3d.abort_frame();
                return Err(e);
            }
        }
        self.fg_set_frame_ms(frame_ms);
        if let Some(fs) = &self.fsr {
            let (dep, mv, scale) = fs.res.fg_inputs(fc.rw as u32, fc.rh as u32);
            self.fg_prepare(&self.d3d.list, fc, dep, mv, scale);
        }
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_FSR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// The FSR upscale dispatch desc, shared by both flavors — every field
    /// is flavor-identical except `mv_scale` (the RR flavor's plane holds
    /// UV-deltas and multiplies the render dims back in; the FSR3 plane
    /// already holds pixel MVs and passes the bare signs).
    fn fsr_upscale_desc(
        cmdlist: *mut std::ffi::c_void,
        res: (ffx_sys::FfxShimRes, ffx_sys::FfxShimRes, ffx_sys::FfxShimRes, ffx_sys::FfxShimRes),
        fc: &dlss::FrameConstants,
        out: (u32, u32),
        frame_ms: f32,
        mv_scale: [f32; 2],
    ) -> ffx::FfxShimUpscaleDesc {
        let (color, depth_clip, mvec, output) = res;
        ffx::FfxShimUpscaleDesc {
            cmdlist,
            color,
            depth: depth_clip,
            motion_vectors: mvec,
            output,
            // fc.jitter is the renderer's sample offset in pixels; the ffx
            // polarity knob is fsr::JITTER_SIGN, nowhere else.
            jitter: [crate::fsr::JITTER_SIGN * fc.jitter.0, crate::fsr::JITTER_SIGN * fc.jitter.1],
            mv_scale,
            render_w: fc.rw as u32,
            render_h: fc.rh as u32,
            out_w: out.0,
            out_h: out.1,
            enable_sharpening: 0,
            sharpness: 0.0,
            frame_time_delta_ms: frame_ms.clamp(0.1, 200.0),
            pre_exposure: 1.0,
            reset: fc.reset as i32,
            cam_near: fc.near,
            cam_far: fc.far,
            cam_fovy: fc.fov_y,
            view_space_to_meters: 1.0,
            flags: 0,
        }
    }

    /// FSR 3.1 mode: upload the three standard temporal-upscaler inputs
    /// (this frame's 1-spp HDR shade from `accum`, pixel-space MVs, clip
    /// depth), one FSR 3.1 upscale dispatch to the window-res output,
    /// tonemap, present. No denoiser anywhere — XeSS-shaped, with the full
    /// temporal input set (jitter, frameTimeDelta, camera params, reset)
    /// on the dispatch. Everything the dispatch sees is in render-res
    /// space; only the upscaled output is window-sized.
    pub fn present_fsr3(
        &mut self,
        accum: &[AtomicU32],
        g: &dlss::GBufs,
        fc: &dlss::FrameConstants,
        frame_ms: f32,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::None);
        crate::zone!("present-fsr");
        let Some(fs) = &self.fsr else {
            return Err("FSR not initialized".into());
        };
        let FsrRes::Up(res) = &fs.res else {
            return Err("present_fsr3 on an FSR4+RR session (present_fsr owns that frame)".into());
        };
        // Release-build cover for record_upload's dimension debug_asserts.
        let (rw, rh) = (fc.rw as u32, fc.rh as u32);
        if (g.rw, g.rh) != (fc.rw, fc.rh) {
            return Err(format!(
                "FSR3 frame is {}x{}, G-buffers {}x{}",
                fc.rw, fc.rh, g.rw, g.rh
            ));
        }
        if accum.len() < fc.rw * fc.rh * 3 {
            return Err(format!(
                "FSR3 frame {}x{} exceeds the accum prefix ({} px)",
                fc.rw,
                fc.rh,
                accum.len() / 3
            ));
        }
        if rw < fs.min.0 || rh < fs.min.1 || rw > fs.max.0 || rh > fs.max.1 {
            return Err(format!(
                "FSR3 frame {}x{} outside render range {}x{}..{}x{}",
                rw, rh, fs.min.0, fs.min.1, fs.max.0, fs.max.1
            ));
        }
        let slot = self.d3d.begin_frame()?;
        if let Err(e) = res.record_upload(&self.d3d, slot, accum, g, fc.rw, fc.rh, fc.near, fc.far) {
            self.d3d.abort_frame();
            return Err(e);
        }

        if let Err(e) = Self::record_fsr3_upscale(fs, &self.d3d.list, fc, frame_ms, None) {
            self.d3d.abort_frame();
            return Err(e);
        }

        self.fg_set_frame_ms(frame_ms);
        if let Some(fs) = &self.fsr {
            let (dep, mv, scale) = fs.res.fg_inputs(fc.rw as u32, fc.rh as u32);
            self.fg_prepare(&self.d3d.list, fc, dep, mv, scale);
        }
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch,
        // restoring whatever list state the ffx dispatch left behind.
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_FSR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// The FSR 3.1 upscale middle shared by the CPU-fed (`present_fsr3`) and
    /// GPU-fed (`present_trace_fsr3`/`present_dxr_fsr3`) chains: barriers +
    /// one upscale dispatch of the render-res sub-rect to the window-res
    /// output. The plane already holds pixel-space MVs, so the scale is the
    /// bare polarity signs. The caller owns the open frame (and aborts it on
    /// Err).
    /// `src` (--quinlight): upscale from FOREIGN input planes — the XeSS trio,
    /// which is byte-for-byte FSR 3.1's own plane set, so a quinlight session
    /// feeds one trio and both SDKs read it (see `upscale_res_shared`). None =
    /// this context's own planes, the single-engine path.
    fn record_fsr3_upscale(
        fs: &FsrState,
        list: &ID3D12GraphicsCommandList,
        fc: &dlss::FrameConstants,
        frame_ms: f32,
        src: Option<&[(&ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT); 3]>,
    ) -> Result<()> {
        let FsrRes::Up(res) = &fs.res else {
            return Err("FSR3 upscale on an FSR4+RR session".into());
        };
        let (rw, rh) = (fc.rw as u32, fc.rh as u32);
        if rw < fs.min.0 || rh < fs.min.1 || rw > fs.max.0 || rh > fs.max.1 {
            return Err(format!(
                "FSR3 frame {}x{} outside render range {}x{}..{}x{}",
                rw, rh, fs.min.0, fs.min.1, fs.max.0, fs.max.1
            ));
        }
        res.barrier_upscale_begin(list);
        let up_desc = Self::fsr_upscale_desc(
            list.as_raw(),
            match src {
                Some(p) => res.upscale_res_shared(p),
                None => res.upscale_res(),
            },
            fc,
            fs.max,
            frame_ms,
            [crate::fsr::UPSCALE_MV_SIGN.0, crate::fsr::UPSCALE_MV_SIGN.1],
        );
        {
            let _ev = pix::scope(list, c"fsr-upscale");
            fs.ctx.upscale(&up_desc)?;
        }
        res.barrier_upscale_end(list);
        Ok(())
    }

    /// One XeSS upscale into `x.res.output`, recorded on `d3d.list`: the block
    /// every GPU-fed XeSS path shares (present_trace_xess, present_dxr_xess, and
    /// the --quinlight fuse, which runs XeSS as one engine among several).
    /// `(rw, rh)` is the trace res — the sub-rect XeSS reads.
    ///
    /// Leaves `output` back in PIXEL_SHADER_RESOURCE (where the tonemap and the
    /// fuse both expect their inputs).
    fn record_xess_eval(
        x: &XessState,
        list: &ID3D12GraphicsCommandList,
        rw: u32,
        rh: u32,
        jitter: (f32, f32),
        reset: bool,
    ) -> Result<()> {
        unsafe {
            list.ResourceBarrier(&[transition(
                &x.res.output,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
        let (c, m, d) = x.res.input_ptrs();
        let params = crate::xess::XessD3d12ExecuteParams {
            color_texture: c,
            velocity_texture: m,
            depth_texture: d,
            exposure_scale_texture: std::ptr::null_mut(),
            responsive_pixel_mask_texture: std::ptr::null_mut(),
            output_texture: x.res.output.as_raw(),
            jitter_offset_x: crate::xess::JITTER_SIGN * jitter.0,
            jitter_offset_y: crate::xess::JITTER_SIGN * jitter.1,
            exposure_scale: 1.0,
            reset_history: reset as u32,
            input_width: rw,
            input_height: rh,
            input_color_base: Default::default(),
            input_motion_vector_base: Default::default(),
            input_depth_base: Default::default(),
            input_responsive_mask_base: Default::default(),
            reserved0: Default::default(),
            output_color_base: Default::default(),
            descriptor_heap: std::ptr::null_mut(),
            descriptor_heap_offset: 0,
        };
        {
            let _ev = pix::scope(list, c"xess-eval");
            x.ctx.execute(list.as_raw(), &params)?;
        }
        unsafe {
            list.ResourceBarrier(&[transition(
                &x.res.output,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
        Ok(())
    }

    /// FSR twin of `read_rr_output`: the denoised+upscaled image exists only
    /// on the GPU; screenshots in FSR mode read it back through the same
    /// path.
    pub fn read_fsr_output(&mut self) -> Result<Vec<u32>> {
        let Some(fs) = &self.fsr else {
            return Err("FSR not initialized".into());
        };
        let output = fs.res.upscaled().clone();
        Ok(self.read_hdr_output(output)?.0)
    }

    /// Shared front half of the XeSS paths: validate, begin the frame,
    /// upload the sub-rect inputs, record the upscale dispatch. On Ok the
    /// output texture is in UNORDERED_ACCESS and the frame is still open —
    /// the caller finishes it (tonemap + present, or readback). On Err the
    /// recorded frame was aborted; nothing reached the GPU.
    #[allow(clippy::too_many_arguments)]
    fn record_xess_dispatch(
        &mut self,
        color: &xr::ColorSrc,
        g: &dlss::GBufs,
        rw: usize,
        rh: usize,
        jitter: (f32, f32),
        reset: bool,
        near: f32,
        far: f32,
    ) -> Result<usize> {
        let Some(x) = &self.xess else {
            return Err("XeSS not initialized".into());
        };
        // Release-build cover for record_upload's debug_asserts: a frame
        // outside the queried range (or mis-sized G-buffers) must not reach
        // the SDK.
        if (g.rw, g.rh) != (rw, rh) {
            return Err(format!("XeSS frame is {}x{}, G-buffers {}x{}", rw, rh, g.rw, g.rh));
        }
        if rw < x.min.0 as usize
            || rh < x.min.1 as usize
            || rw > x.max.0 as usize
            || rh > x.max.1 as usize
        {
            return Err(format!(
                "XeSS frame {}x{} outside input range {}x{}..{}x{}",
                rw, rh, x.min.0, x.min.1, x.max.0, x.max.1
            ));
        }
        let slot = self.d3d.begin_frame()?;
        if let Err(e) = x.res.record_upload(&self.d3d, slot, color, g, rw, rh, near, far) {
            self.d3d.abort_frame();
            return Err(e);
        }
        unsafe {
            self.d3d.list.ResourceBarrier(&[transition(
                &x.res.output,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }

        let (c, m, d) = x.res.input_ptrs();
        let params = crate::xess::XessD3d12ExecuteParams {
            color_texture: c,
            velocity_texture: m,
            depth_texture: d,
            exposure_scale_texture: std::ptr::null_mut(),
            responsive_pixel_mask_texture: std::ptr::null_mut(),
            output_texture: x.res.output.as_raw(),
            jitter_offset_x: crate::xess::JITTER_SIGN * jitter.0,
            jitter_offset_y: crate::xess::JITTER_SIGN * jitter.1,
            // With ENABLE_AUTOEXPOSURE the SDK computes exposure internally
            // and ignores both the scalar and the (null) texture.
            exposure_scale: 1.0,
            reset_history: reset as u32,
            input_width: rw as u32,
            input_height: rh as u32,
            input_color_base: Default::default(),
            input_motion_vector_base: Default::default(),
            input_depth_base: Default::default(),
            input_responsive_mask_base: Default::default(),
            reserved0: Default::default(),
            output_color_base: Default::default(),
            descriptor_heap: std::ptr::null_mut(),
            descriptor_heap_offset: 0,
        };
        {
            let _ev = pix::scope(&self.d3d.list, c"xess-eval");
            if let Err(e) = x.ctx.execute(self.d3d.list.as_raw(), &params) {
                // Nothing executed on the GPU yet — abandon the recorded
                // frame so the caller can fall back to the CPU present path.
                self.d3d.abort_frame();
                return Err(e);
            }
        }
        Ok(slot)
    }

    /// XeSS-SR: upload this frame's `rw×rh` color (raw 1-spp or
    /// OIDN-denoised) + MV + depth sub-rects, record the upscale into the
    /// window-res output, tonemap, present. Everything XeSS sees — jitter,
    /// MVs, depth — is in input-res pixel space; only the output is
    /// window-sized. `jitter` is the renderer's sample offset; the sign
    /// reported to XeSS is settled in xess::JITTER_SIGN, nowhere else.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn present_xess(
        &mut self,
        color: &xr::ColorSrc,
        g: &dlss::GBufs,
        rw: usize,
        rh: usize,
        jitter: (f32, f32),
        reset: bool,
        near: f32,
        far: f32,
        fc: &dlss::FrameConstants,
        frame_ms: f32,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::None);
        crate::zone!("present-xess");
        let slot = self.record_xess_dispatch(color, g, rw, rh, jitter, reset, near, far)?;
        let x = self.xess.as_ref().unwrap();
        unsafe {
            self.d3d.list.ResourceBarrier(&[transition(
                &x.res.output,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
        self.xefg_prepare(fc, frame_ms);
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch,
        // restoring whatever list state the XeSS dispatch left behind.
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_XESS, 1.0);
        self.xefg_end_frame(slot)
    }

    /// XeSS-SR, post-upscale-denoise flavor: same dispatch, but instead of
    /// the tonemap/present tail the window-res HDR output is copied to the
    /// persistent readback buffer, the submission is executed WITHOUT a
    /// Present and waited on, and the result lands in `out` (linear RGB f32,
    /// window-res). The caller denoises it and presents via `present_cpu` —
    /// the frame's single Present. Synchronous by design; this is the
    /// experiment path, not the fast one.
    #[allow(clippy::too_many_arguments)]
    pub fn upscale_xess_to_cpu(
        &mut self,
        color: &xr::ColorSrc,
        g: &dlss::GBufs,
        rw: usize,
        rh: usize,
        jitter: (f32, f32),
        reset: bool,
        near: f32,
        far: f32,
        out: &mut [f32],
    ) -> Result<()> {
        crate::zone!("xess-readback");
        let slot = self.record_xess_dispatch(color, g, rw, rh, jitter, reset, near, far)?;
        let x = self.xess.as_ref().unwrap();
        if let Err(e) = x.res.record_readback(&self.d3d) {
            self.d3d.abort_frame();
            return Err(e);
        }
        self.d3d.submit_and_wait(slot)?;
        x.res.read_back(out)
    }

    /// M4: DLSS Ray Reconstruction — upload the 1-spp radiance + G-buffers,
    /// run slEvaluateFeature, tonemap the denoised HDR output, present.
    pub fn present_rr(
        &mut self,
        accum: &[AtomicU32],
        g: &dlss::GBufs,
        fc: &dlss::FrameConstants,
        frame_idx: u32,
        // The frame's baked swarm — the raw-NGX FG tail's round-3 firefly
        // MVs (the GPU-fed arms read it off FrameParams; this CPU-fed arm
        // is handed the same `Fireflies::live` value the RenderCtx traced
        // with). Unused when FG is not armed.
        ff: &crate::fireflies::Fireflies,
        // Same shape, for the round-4 ripple MVs: the clock the RenderCtx
        // traced this frame with. Unused when FG is not armed.
        cl: &crate::clouds::Clouds,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::None);
        crate::zone!("present-rr");
        let _ = frame_idx; // was the SL frame token's index; the raw evaluate needs none
        let (Some(nx), Some(feat), Some(rr)) =
            (&self.ngxrr, &self.rr_feature, &self.rr)
        else {
            return Err("DLSS-RR not initialized".into());
        };
        // Release-build cover for record_upload's dimension debug_asserts:
        // a frame outside the queried DRS range (or beyond plane capacity)
        // must not reach SL.
        let (rw, rh) = (fc.rw as u32, fc.rh as u32);
        if rw < rr.min.0 || rh < rr.min.1 || rw > rr.max.0 || rh > rr.max.1 {
            return Err(format!(
                "DLSS-RR frame {}x{} outside render range {}x{}..{}x{}",
                rw, rh, rr.min.0, rr.min.1, rr.max.0, rr.max.1
            ));
        }
        let slot = self.d3d.begin_frame()?;
        if let Err(e) = rr.record_upload(&self.d3d, slot, accum, g, fc.rw, fc.rh) {
            self.d3d.abort_frame();
            return Err(e);
        }
        unsafe {
            self.d3d.list.ResourceBarrier(&[transition(
                &rr.output,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }

        if let Err(e) = rr_ngx_sequence(nx, feat, rr, &self.d3d.list, fc) {
            // Abandon the recorded-but-unexecuted frame: nothing reached the
            // GPU, so tracked resource states are unchanged, and closing the
            // list lets the next present's begin_frame Reset it. The caller
            // can fall back to the non-DLSS present path.
            self.d3d.abort_frame();
            return Err(e);
        }

        unsafe {
            self.d3d.list.ResourceBarrier(&[transition(
                &rr.output,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
        if self.fg_n.is_some() {
            return self.ngxfg_tail(slot, fc, ff, cl);
        }
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_RR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// DLSS-RR fed by the GPU-resident tracer (`--gpu` default): one command
    /// list = trace -> feed (pack -> the 7 SL input planes, on-GPU — the CPU
    /// upload of present_rr does not exist here) -> the SL sequence (token,
    /// constants, options, tags, evaluate) -> tonemap(SRV_SLOT_RR) ->
    /// present. The whole list executes on the SL PROXY queue (validated in
    /// M7); the feed's back-transitions leave the planes in
    /// NON_PIXEL_SHADER_RESOURCE, exactly what rr.tags declares at evaluate.
    pub fn present_trace_rr(
        &mut self,
        p: &trace::FrameParams,
        hybrid: bool,
        fc: &dlss::FrameConstants,
        frame_idx: u32,
    ) -> Result<()> {
        self.waveviz_src.set(WvSrc::Trace);
        crate::zone!("present-trace-rr");
        let _ = frame_idx; // was the SL frame token's index; the raw evaluate needs none
        if self.trace.is_none() || self.ngxrr.is_none() || self.rr.is_none() {
            return Err("GPU tracer + DLSS-RR not both initialized".into());
        }
        let slot = self.d3d.begin_frame()?;
        {
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            // The session res is fixed and range-checked at init; fc must
            // agree (it drives the extent tags SL derives the ratio from).
            if (fc.rw as u32, fc.rh as u32) != (tg.rw, tg.rh) {
                d3d.abort_frame();
                return Err(format!(
                    "frame constants {}x{} != trace res {}x{}",
                    fc.rw, fc.rh, tg.rw, tg.rh
                ));
            }
        }
        // The trace goes through the ONE site (`record_trace`), which is what
        // puts --dual-gpu in every wavefront presenter at once.
        if let Err(e) = self.record_trace(p, hybrid, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        {
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            if let Err(e) = tg.record_feed(&d3d.list, slot, false) {
                d3d.abort_frame();
                return Err(e);
            }
        }
        {
            let d3d = &mut self.d3d;
            let nx = self.ngxrr.as_ref().unwrap();
            let Some(feat) = self.rr_feature.as_ref() else {
                d3d.abort_frame();
                return Err("DLSSD feature not created".into());
            };
            let rr = self.rr.as_ref().unwrap();
            unsafe {
                d3d.list.ResourceBarrier(&[transition(
                    &rr.output,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            if let Err(e) = rr_ngx_sequence(nx, feat, rr, &d3d.list, fc) {
                d3d.abort_frame();
                return Err(e);
            }
            unsafe {
                d3d.list.ResourceBarrier(&[transition(
                    &rr.output,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                )]);
            }
        }
        if self.fg_n.is_some() {
            return self.ngxfg_tail(slot, fc, &p.fireflies, &p.clouds);
        }
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_RR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// Read back the denoised RR output and tonemap it on the CPU with the
    /// same curve as `render::resolve` at 1 spp. Screenshots in DLSS mode
    /// need this — the denoised image exists only in GPU memory. Synchronous
    /// and allocation-heavy by design; it runs on a keypress, not per frame.
    pub fn read_rr_output(&mut self) -> Result<Vec<u32>> {
        let Some(rr) = &self.rr else {
            return Err("DLSS-RR not initialized".into());
        };
        let output = rr.output.clone();
        Ok(self.read_hdr_output(output)?.0)
    }

    /// XeSS twin of `read_rr_output`: the upscaled image exists only on the
    /// GPU; screenshots in XeSS mode read it back through the same path.
    pub fn read_xess_output(&mut self) -> Result<Vec<u32>> {
        let Some(x) = &self.xess else {
            return Err("XeSS not initialized".into());
        };
        let output = x.res.output.clone();
        Ok(self.read_hdr_output(output)?.0)
    }

    /// Dims come from the texture itself — the tracer's hdr is RENDER-res in
    /// upscaler sessions, window-res otherwise; rr/xess outputs are always
    /// window-res.
    fn read_hdr_output(&mut self, output: ID3D12Resource) -> Result<(Vec<u32>, usize, usize)> {
        let desc = unsafe { output.GetDesc() };
        let (w, h) = (desc.Width as usize, desc.Height as usize);
        let pitch = d3d12::aligned_pitch(w * 8);
        let rb = d3d12::ReadbackBuffer::new(&self.d3d.device, pitch * h)?;
        let fp = d3d12::footprint(
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            w as u32,
            h as u32,
            8,
            0,
        );
        self.d3d.run_once(|list| unsafe {
            list.ResourceBarrier(&[transition(
                &output,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyTextureRegion(
                &d3d12::loc_footprint(&rb.resource, fp),
                0,
                0,
                0,
                &d3d12::loc_subresource(&output),
                None,
            );
            list.ResourceBarrier(&[transition(
                &output,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        })?;
        let mut ptr = std::ptr::null_mut();
        unsafe { rb.resource.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| format!("readback Map: {e}"))?;
        let mut out = vec![0u32; w * h];
        for y in 0..h {
            let row: &[[half::f16; 4]] = unsafe {
                std::slice::from_raw_parts((ptr as *const u8).add(y * pitch) as *const _, w)
            };
            for (x, px) in row.iter().enumerate() {
                // Screenshots and --check PNGs stay SDR 8-bit regardless of the
                // session's swapchain: a PNG has nowhere to put a nit. This used
                // to open-code the curve a third time; it now shares tone::map,
                // so the file can't drift from the screen.
                let c = glam::Vec3A::new(px[0].into(), px[1].into(), px[2].into());
                // ... but they DO carry the session's live exposure — P must
                // capture what the screen shows (the "two sessions must agree
                // about what P captures" rule), and 1.0 is bit-inert.
                let tp = crate::tone::ToneParams {
                    exposure: self.tone.exposure,
                    ..crate::tone::ToneParams::SDR
                };
                let m = crate::tone::map(c, tp)
                    .clamp(glam::Vec3A::ZERO, glam::Vec3A::ONE)
                    * 255.0;
                let q = |v: f32| (v + 0.5) as u32;
                out[y * w + x] = (q(m.x) << 16) | (q(m.y) << 8) | q(m.z);
            }
        }
        unsafe { rb.resource.Unmap(0, None) };
        Ok((out, w, h))
    }

    /// M1: present the CPU-tonemapped u32 0RGB frame.
    pub fn present_cpu(&mut self, pixels: &[u32]) -> Result<()> {
        self.waveviz_src.set(WvSrc::None);
        crate::zone!("present-cpu");
        let slot = self.d3d.begin_frame()?;
        self.blit.record(&self.d3d, slot, pixels);
        self.fullscreen_to_backbuffer(false, tonemap::SRV_SLOT_BLIT, 1.0);
        self.d3d.end_frame(slot)
    }

    /// `present_cpu` for the 10-bit swapchain (Sdr10 or Hdr10): the CPU
    /// produced the final packed 10-bit u32 (`render::present_px_sdr10` /
    /// `present_px_pq` applied the curve, the overlay, the encode, and the
    /// R10G10B10A2 pack), so this is still a straight blit — the blit PS
    /// stays a passthrough and never learns about colour spaces.
    pub fn present_cpu_10bit(&mut self, pixels: &[u32]) -> Result<()> {
        self.waveviz_src.set(WvSrc::None);
        crate::zone!("present-cpu-10bit");
        let slot = self.d3d.begin_frame()?;
        self.blit.record_10bit(&self.d3d, slot, pixels);
        self.fullscreen_to_backbuffer(false, tonemap::SRV_SLOT_BLIT, 1.0);
        self.d3d.end_frame(slot)
    }

    /// Which colour space presentation actually negotiated — NOT merely what
    /// the flags asked for (the G2084 declare can be refused and the FG wrap
    /// can force a rebuild; see `D3d::with_queue`). The CPU arms pick their
    /// encode on this.
    pub fn encoding(&self) -> d3d12::PresentSpace {
        self.d3d.space
    }

    /// Re-probe the monitor the window now sits on and re-derive the curve.
    /// Returns the new state when it actually changed, so the caller can log
    /// once rather than every poll.
    ///
    /// Cheap by construction: this is a `GetDesc1` and a field write. It is NOT
    /// a swapchain rebuild, and it deliberately does not reset the upscaler
    /// history — a display change is a change of output device, not of scene.
    /// Only the Hdr10 arm has anything to retune (peak/paper aim the PQ
    /// rolloff); the gamma arms' curve is the static `ToneParams::SDR`.
    pub fn refresh_display(&mut self, paper_white: f32, peak: Option<f32>) -> Option<display::DisplayHdr> {
        if self.d3d.space != d3d12::PresentSpace::Hdr10 {
            return None; // one static curve, no display to ask
        }
        let d = display::probe(&self.adapter, self.hwnd);
        if Some(d) == self.display {
            return None;
        }
        self.display = Some(d);
        // A display move is a retune of the CURVE, never of the aperture — the
        // session's live exposure survives it.
        self.tone = crate::tone::ToneParams {
            exposure: self.tone.exposure,
            ..d.tone_pq(paper_white, peak)
        };
        Some(d)
    }

    /// The presentation curve every present arm is reading right now. The CPU
    /// arms pull it each frame (`CpuPresent::tone`), which is how a display
    /// change reaches them.
    pub fn tone(&self) -> crate::tone::ToneParams {
        self.tone
    }

    /// Auto-exposure: the linear scale the presentation curve applies next
    /// frame. UNCONDITIONAL by design — `refresh_display` early-outs on the
    /// gamma wires, so exposure must not route through it (a setter gated on
    /// Hdr10 would be dead in every SDR/Sdr10 session).
    pub fn set_exposure(&mut self, e: f32) {
        self.tone.exposure = e;
    }

    /// Auto-exposure: the newest collected meter value (mean log2-luminance of
    /// a presented frame's tonemap source, FRAMES_IN_FLIGHT frames old),
    /// stashed by `fullscreen_to_backbuffer`. Take-semantics so the controller
    /// never folds one measurement twice.
    pub fn take_meter(&self) -> Option<f32> {
        self.meter.take()
    }


    /// M2: present the raw linear-HDR accumulation with the GPU tonemap.
    pub fn present_hdr(&mut self, accum: &[AtomicU32], samples: u32) -> Result<()> {
        self.waveviz_src.set(WvSrc::None);
        crate::zone!("present-hdr");
        let slot = self.d3d.begin_frame()?;
        self.hdr.record(&self.d3d, slot, accum);
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_HDR, 1.0 / samples.max(1) as f32);
        self.d3d.end_frame(slot)
    }

    /// The resource behind a tonemap SRV slot. Bloom needs the RESOURCE, not just
    /// its descriptor, because its downsample is a compute dispatch and D3D12
    /// state is per-resource (see `fullscreen_to_backbuffer`). Keep in lockstep
    /// with the `passes.create_srv` calls in `new` / `resize_output` /
    /// `ensure_trace` / `ensure_dxr` — an omission here silently costs that mode
    /// its glare rather than corrupting anything.
    fn tonemap_source(&self, srv_slot: u32) -> Option<&ID3D12Resource> {
        match srv_slot {
            tonemap::SRV_SLOT_HDR => Some(&self.hdr.texture),
            tonemap::SRV_SLOT_RR => self.rr.as_ref().map(|r| &r.output),
            tonemap::SRV_SLOT_XESS => self.xess.as_ref().map(|s| &s.res.output),
            tonemap::SRV_SLOT_FSR => self.fsr.as_ref().map(|s| s.res.upscaled()),
            tonemap::SRV_SLOT_GPU => self.trace.as_ref().map(|t| &t.hdr),
            tonemap::SRV_SLOT_DXR => self.dxr.as_ref().map(|d| &d.hdr),
            tonemap::SRV_SLOT_QUIN => self.quin.as_ref().map(|q| &q.output),
            // The raw-NGX interpolated frame: without this arm the pair's
            // first present skipped BLOOM (tonemap_source → None → strength
            // 0), so every bright highlight strobed glare-on/glare-off at
            // half the present rate — shipped in the first --fg cut, and on
            // a sun-reflecting helmet it reads as the reflections dancing.
            // (Its bloom SRV was already registered by wire_tonemap_src.)
            tonemap::SRV_SLOT_NGXFG => self.fg_n.as_ref().map(|n| &n.out),
            _ => None,
        }
    }

    // ---- frame generation (both families' per-frame protocols) ----
    //
    // The four dead_code-allowed accessors below are the LIVE-toggle API a
    // runtime FG switch would consume. Nothing consumes it today — the
    // settings menu's frame-generation row shipped restart-tier (the file
    // drives the DEFAULT arm; no key toggles FG live) — but the disable
    // choreography in `set_fg_enabled` (funnel handshake / passthrough
    // configure / SetEnabled, per family) is exactly what a live row needs
    // and is not obvious to rederive, so the group stays wired-shaped.

    /// True when a frame-generation family is wired: the ffx FI proxy with a
    /// live FG effect context, raw-NGX DLSS-G, or the XeSS-FG proxy.
    #[allow(dead_code)]
    pub fn fg_wired(&self) -> bool {
        self.fg.as_ref().is_some_and(|f| f.ctx.is_some())
            || self.fg_x.as_ref().is_some_and(|x| !x.failed.get())
            || self.fg_n.as_ref().is_some_and(|n| !n.failed.get())
    }

    /// Whether generated frames are currently being inserted (the toggle
    /// state, not the per-frame handshake).
    #[allow(dead_code)]
    pub fn fg_enabled(&self) -> bool {
        self.fg.as_ref().is_some_and(|f| f.ctx.is_some() && f.enabled)
            || self.fg_x.as_ref().is_some_and(|x| x.enabled && !x.failed.get())
            || self.fg_n.as_ref().is_some_and(|n| n.enabled && !n.failed.get())
    }

    /// The wired FG family's display name (the boot line / title bar).
    #[allow(dead_code)]
    pub fn fg_label(&self) -> Option<&str> {
        if self.fg_n.as_ref().is_some_and(|n| !n.failed.get()) {
            return Some("DLSS-G (NGX)");
        }
        if self.fg_x.as_ref().is_some_and(|x| !x.failed.get()) {
            return Some("XeSS-FG");
        }
        self.fg.as_ref().filter(|f| f.ctx.is_some()).map(|f| f.version.1.as_str())
    }

    /// The presented-per-rendered multiplier for the title bar: None when no
    /// FG family is armed (or the toggle is off); Some(m) when armed. The
    /// frame loop counts RENDERED frames, so an FG session's displayed fps
    /// under-reports what the monitor receives by this factor. m is measured
    /// wherever the family can be asked — raw NGX pair-presents itself (exact
    /// by construction), XeSS-FG reads its status poll's frames-presented
    /// (assumes 2 until the first poll lands), ffx FI has no query and
    /// reports 2 by configuration when live. m == 1 is meaningful: armed but
    /// not inserting (holds, unprimed reset frames).
    pub fn fg_display_mult(&self) -> Option<u32> {
        if let Some(n) = &self.fg_n {
            if n.enabled && !n.failed.get() {
                return Some(if n.pair.get() { 2 } else { 1 });
            }
        }
        if let Some(x) = &self.fg_x {
            if x.enabled && !x.failed.get() {
                let m = x.mult.get();
                return Some(if m == 0 {
                    if x.on.get() { 2 } else { 1 }
                } else {
                    m
                });
            }
        }
        if let Some(f) = &self.fg {
            if f.ctx.is_some() && f.enabled {
                return Some(if f.live.get() { 2 } else { 1 });
            }
        }
        None
    }

    /// The live toggle. Turning off disables generation immediately (the ffx
    /// proxy configures passthrough; DLSS-G's mode-off edge lands via the
    /// funnel handshake on the very next present; XeSS-FG flips SetEnabled).
    #[allow(dead_code)]
    pub fn set_fg_enabled(&mut self, on: bool) {
        if let Some(n) = &mut self.fg_n {
            n.enabled = on;
            if !on {
                n.primed.set(false);
                n.pair.set(false);
            }
        }
        if let Some(x) = &mut self.fg_x {
            x.enabled = on;
            if !on && x.on.get() {
                x.sc.set_enabled(false);
                x.on.set(false);
            }
        }
        if let Some(fg) = &mut self.fg {
            fg.enabled = on;
            if !on {
                self.fg_disable_now();
            }
        }
    }

    /// The render-mode-switch FG straddle (the AMD mode-cycle-slowdown fix,
    /// diagnosed 2026-07-31 on the R9700): carrying the ffx FG prepare stream
    /// seamlessly across a SPACE/F mode switch — a reset=1 prepare + a
    /// depth/MV resource-set swap + a frame-time cadence jump, all while
    /// generation stays enabled — wedges the AMD provider's pacing into a
    /// massive persistent slowdown. Both a window resize (context rebuild)
    /// and a K plain-toggle round trip (configure disabled → passthrough
    /// presents → configure enabled, NO rebuild) were measured to clear it,
    /// so the DEFAULT here is the cheapest cure as prevention: skip the next
    /// prepare, giving the FI proxy exactly one disabled passthrough present
    /// at the switch seam. FR_FG_CYCLE=off restores the old
    /// carry-straight-across behavior (the repro arm); FR_FG_CYCLE=recreate
    /// is the heavy A/B arm — rebuild the display-size effect context (the
    /// resize path's straddle; the FI SWAPCHAIN context survives throughout).
    pub fn fg_mode_switch(&mut self, debug: bool) {
        let recreate = match std::env::var("FR_FG_CYCLE").ok().as_deref() {
            Some("recreate") => true,
            Some("off") => return,
            Some(other) => {
                // A silent no-op A/B walk is the failure mode levers exist to
                // prevent — loud, then the default.
                eprintln!(
                    "fg: FR_FG_CYCLE={other} unrecognized (expected off|recreate) — \
                     taking the default (pause straddle)"
                );
                false
            }
            None => false,
        };
        if !recreate {
            // The default: one-frame pause straddle. Nothing to do when no
            // effect context exists (passthrough proxy or FG-less session).
            if let Some(fg) = &self.fg {
                if fg.ctx.is_some() {
                    fg.skip_prepare.set(true);
                }
            }
            return;
        }
        {
            let Some(fg) = &self.fg else { return };
            if fg.ctx.is_none() || fg.version.0 == 0 {
                return;
            }
        }
        // The teardown discipline, in its order: UNREGISTER first (the
        // disable configure, while the effect context the proxy's callback
        // points into is still alive — idempotent via `live`), THEN retire
        // pending paced presents (they can actually retire with generation
        // off), and only then drop the context. The resize straddle skips
        // the unregister and survives, but hygiene is free here.
        self.fg_disable_now();
        let Some(fg) = &mut self.fg else { return };
        fg.sc.wait_for_presents();
        fg.prepared.set(false);
        fg.ctx = None;
        let (w, h) = (self.d3d.width, self.d3d.height);
        match ffx::FgContext::create(
            &self.d3d.device,
            (w, h),
            (w, h),
            match self.d3d.space {
                d3d12::PresentSpace::Sdr10 | d3d12::PresentSpace::Hdr10 => {
                    ffx_sys::SURFACE_FORMAT_R10G10B10A2_UNORM
                }
                d3d12::PresentSpace::Sdr => ffx_sys::SURFACE_FORMAT_B8G8R8A8_UNORM,
            },
            self.d3d.space == d3d12::PresentSpace::Hdr10,
            debug,
            fg.version.0,
        ) {
            Ok(ctx) => {
                fg.ctx = Some(ctx);
                eprintln!("fg: FR_FG_CYCLE=recreate — effect context rebuilt on mode switch");
            }
            Err(e) => eprintln!(
                "fg: FR_FG_CYCLE=recreate rebuild failed ({e}) — generation off until resize"
            ),
        }
    }

    /// XeSS-FG per-frame work — call in the XeSS arms AFTER the feed/upload
    /// (planes final, resting NON_PIXEL_SHADER_RESOURCE), BEFORE the funnel:
    /// XeLL sleep + first markers, the depth/MV tags and frame constants for
    /// this presentId, the presentId itself, and the lazy re-enable.
    fn xefg_prepare(&self, fc: &dlss::FrameConstants, frame_ms: f32) {
        let (Some(x), Some(xs)) = (&self.fg_x, &self.xess) else { return };
        if !x.enabled || x.failed.get() {
            return;
        }
        let id = x.frame_id.get().wrapping_add(1);
        x.frame_id.set(id);
        x.sc.sleep(id);
        x.sc.marker(id, crate::xess_fg::XELL_SIMULATION_START);
        x.sc.marker(id, crate::xess_fg::XELL_SIMULATION_END);
        x.sc.marker(id, crate::xess_fg::XELL_RENDERSUBMIT_START);
        let npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 as u32;
        let (_c, m, d) = xs.res.input_ptrs();
        let (rw, rh) = (fc.rw as u32, fc.rh as u32);
        let r = x
            .sc
            .tag_resource(id, crate::xess_fg::RES_DEPTH, d, npsr, rw, rh)
            .and_then(|()| {
                x.sc.tag_resource(id, crate::xess_fg::RES_MOTION_VECTOR, m, npsr, rw, rh)
            })
            .and_then(|()| {
                // Row-major matrices — the SL transpose convention, NOT the
                // ffx memcpy (the header says row-major explicitly).
                x.sc.tag_constants(
                    id,
                    row_major(&fc.world_to_view),
                    row_major(&fc.view_to_clip),
                    fc.jitter,
                    fc.reset,
                    frame_ms,
                )
            });
        if let Err(e) = r {
            eprintln!("fg: XeSS-FG {e} — frame generation off");
            x.sc.set_enabled(false);
            x.on.set(false);
            x.failed.set(true);
            return;
        }
        // The UI texture tag (RES_UI, window-sized, premultiplied — xefg's
        // DEFAULT alpha convention; the NOT_PREMUL init flag is the opt-out
        // we don't take). Under UI_MODE_AUTO this resolves the proxy to
        // BACKBUFFER_UITEXTURE: interpolate the backbuffer as before, REFINE
        // the UI region on generated frames from the tagged texture — the
        // baked HUD draw stays, so the tag is strictly additive and OPTIONAL:
        // a failure sheds the tag alone (loud once), never frame generation.
        // The funnel records the HudFi pre-pass on prepared XeSS-FG frames.
        if !x.ui_shed.get() && self.hud.drawable() {
            if let Some(ui) = &self.fg_ui {
                let psr = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 as u32;
                if let Err(e) = x.sc.tag_resource(
                    id,
                    crate::xess_fg::RES_UI,
                    ui.resource().as_raw(),
                    psr,
                    self.d3d.width,
                    self.d3d.height,
                ) {
                    eprintln!("fg: XeSS-FG ui tag failed ({e}) — HUD stays baked pre-present");
                    x.ui_shed.set(true);
                }
            }
        }
        x.sc.set_present_id(id);
        if !x.on.get() {
            x.sc.set_enabled(true);
            x.on.set(true);
        }
        x.prepared.set(true);
    }

    /// The XeSS arms' end_frame: XeLL render-submit-end + present markers
    /// around Execute+Present, then the periodic status poll (a negative
    /// frameGenResult = sticky off; the first clean second-interval poll logs
    /// framesPresented — 2 per present = generating).
    fn xefg_end_frame(&mut self, slot: usize) -> Result<()> {
        let bracket = self.fg_x.as_ref().is_some_and(|x| x.prepared.replace(false));
        if !bracket {
            return self.d3d.end_frame(slot);
        }
        let x = self.fg_x.as_ref().unwrap();
        let id = x.frame_id.get();
        x.sc.marker(id, crate::xess_fg::XELL_RENDERSUBMIT_END);
        x.sc.marker(id, crate::xess_fg::XELL_PRESENT_START);
        let r = self.d3d.end_frame(slot);
        let x = self.fg_x.as_ref().unwrap();
        x.sc.marker(id, crate::xess_fg::XELL_PRESENT_END);
        let n = x.poll.get() + 1;
        x.poll.set(n);
        if n % 120 == 0 {
            match x.sc.last_status() {
                Ok(s) if s.frame_gen_result < 0 => {
                    eprintln!(
                        "fg: XeSS-FG present status {} — frame generation off",
                        crate::xess_fg::result_name(s.frame_gen_result)
                    );
                    x.sc.set_enabled(false);
                    x.on.set(false);
                    x.failed.set(true);
                }
                Ok(s) => {
                    x.mult.set(s.frames_presented.max(1));
                    if !x.logged.get() && n >= 240 {
                        x.logged.set(true);
                        eprintln!(
                            "fg: XeSS-FG last present: {} frame(s) presented, gen result {}, enabled {}",
                            s.frames_presented,
                            crate::xess_fg::result_name(s.frame_gen_result),
                            s.is_frame_gen_enabled != 0
                        );
                    }
                }
                Err(_) => {}
            }
        }
        r
    }

    /// What raw-NGX FG interpolates and which tonemap slot presents the REAL
    /// half: a quinlight session hands NGX the FUSED image (`quin.output` —
    /// same window-res RGBA16F, same PIXEL_SHADER_RESOURCE rest state as
    /// `rr.output`, written by `Quin::record` earlier on this very list),
    /// every other session the RR output. Resolved per frame — the clone is
    /// an AddRef, which dodges the `&mut self` borrows inside the dispatch —
    /// and per frame is also what makes a resize safe: the rebuilt
    /// `quin.output` is picked up on the next dispatch.
    fn ngxfg_target(&self) -> Option<(ID3D12Resource, u32)> {
        if let Some(q) = &self.quin {
            return Some((q.output.clone(), tonemap::SRV_SLOT_QUIN));
        }
        self.rr.as_ref().map(|rr| (rr.output.clone(), tonemap::SRV_SLOT_RR))
    }

    /// Raw-NGX DLSS-G: record this frame's interpolation evaluate into the
    /// open list. `ngxfg_target()`'s color is the CURRENT frame (the RR
    /// output, or the fused image under --quinlight); the NGX feature
    /// retains the previous one internally, so the evaluate produces the
    /// frame BETWEEN them into `fg_n.out`. Returns true when that output is
    /// presentable (a prior frame primed the pair and this one is no reset).
    /// The feature is created lazily at the frame's locked render res.
    /// `slot` = the frame's in-flight slot (the guide pass's firefly CB ring
    /// writes into it); `ff` = the frame's baked swarm (`Fireflies::off()`
    /// shape on day frames — count 0 skips the whole round-3 bake).
    fn ngxfg_dispatch(
        &mut self,
        slot: usize,
        fc: &dlss::FrameConstants,
        ff: &crate::fireflies::Fireflies,
        cl: &crate::clouds::Clouds,
    ) -> bool {
        let (Some(n), Some(rr)) = (&self.fg_n, &self.rr) else { return false };
        if !n.enabled || n.failed.get() {
            return false;
        }
        // The interpolation color source (the one field the quin session
        // swaps); guide planes/dims stay on `rr` — its planes are fed in
        // quin sessions too, and quin.output's dims ARE rr.ow x rr.oh.
        let Some((tgt_color, _)) = self.ngxfg_target() else { return false };
        let (rw, rh) = (fc.rw as u32, fc.rh as u32);
        // A render-res MOVE recreates the feature instead of skipping
        // forever: the CPU renderer fills the same MV/depth/guide planes the
        // GPU arms do, so FG follows SPACE/F mode cycles across their
        // different locked resolutions (CPU quality-2/3 vs GPU/DXR native).
        // Gated on the moved res HOLDING FG_RECREATE_STABLE dispatches —
        // `--lock-res dynamic` ramps change res per frame and must keep the
        // old skip behavior, never a per-frame recreate storm.
        if !n.handle.get().is_null() && n.dims.get() != (rw, rh) {
            let (pres, prev_cnt) = n.res_pend.get();
            let cnt = if pres == (rw, rh) { prev_cnt + 1 } else { 1 };
            n.res_pend.set(((rw, rh), cnt));
            if cnt < FG_RECREATE_STABLE {
                // One line per move EPISODE (pres == settled), not per new
                // res — a dynamic ramp changes res every frame, and cnt == 1
                // alone would print on each of its ~24 intermediates.
                if cnt == 1 && pres == (0, 0) {
                    let (ow, oh) = n.dims.get();
                    eprintln!(
                        "fg: render res moved {}x{} -> {}x{} — frame generation \
                         recreates once the new res holds {} frames",
                        ow, oh, rw, rh, FG_RECREATE_STABLE
                    );
                }
                n.primed.set(false);
                return false;
            }
            let (ow, oh) = n.dims.get();
            eprintln!(
                "fg: recreating the NGX frame-generation feature at {}x{} (was {}x{})",
                rw, rh, ow, oh
            );
            // The resize-path discipline (resize_output): drain the queue
            // before releasing — in-flight frames may still reference the
            // feature's internals and the guide pass's heap, whose
            // descriptors the ensure below rewrites.
            if let Err(e) = self.d3d.wait_idle() {
                eprintln!("fg: queue drain before FG recreate failed ({e}) — frame generation off");
                n.failed.set(true);
                n.primed.set(false);
                return false;
            }
            // FEATURE-scoped recreate — deliberately NOT destroy + lazy
            // create (an SL-era lesson kept for its cheapness: destroy tore
            // at the NGX parameter map the in-process Streamline SHARED and
            // every subsequent RR evaluate failed 0xBAD00004 FeatureNotFound;
            // today the DLSSD session shares the same map, so the discipline
            // still holds).
            let r = unsafe { ngxfg::frdlssg_recreate(n.handle.get(), rr.ow, rr.oh, rw, rh) };
            if r != 0 {
                eprintln!("fg: NGX feature recreate failed ({r}) — frame generation off");
                n.failed.set(true);
                n.primed.set(false);
                return false;
            }
            n.dims.set((rw, rh));
            n.primed.set(false);
            n.res_pend.set(((0, 0), 0));
            // The guide planes follow the render res (their descriptors
            // rewrite safely — the queue just drained).
            if let Some(gp) = &n.guides {
                match gp.borrow_mut().ensure(&self.d3d.device, rw, rh, rr.guide_inputs()) {
                    Ok(()) => n.guides_failed.set(false),
                    Err(e) => {
                        eprintln!(
                            "fg: guide-plane recreate failed ({e}) — falling back to the \
                             raw RR planes (expect reflection shimmer)"
                        );
                        n.guides_failed.set(true);
                    }
                }
            }
        }
        if n.handle.get().is_null() {
            let mut h = std::ptr::null_mut();
            let r = unsafe {
                ngxfg::frdlssg_create(self.d3d.device.as_raw(), rr.ow, rr.oh, rw, rh, 1, &mut h)
            };
            if r != 0 || h.is_null() {
                if r == ngxfg::ERR_UNSUPPORTED {
                    // The shim's fall-through code (FrameGeneration.Available=0
                    // — old driver / unsupported hardware; details already on
                    // stderr), distinct from a real create error.
                    eprintln!("fg: DLSS-G unsupported on this adapter/driver — frame generation off");
                } else {
                    eprintln!("fg: raw-NGX DLSS-G create failed ({r}) — frame generation off");
                }
                n.failed.set(true);
                return false;
            }
            n.handle.set(h);
            n.dims.set((rw, rh));
            // The guide planes are created (and every descriptor rewritten —
            // a resize rebuilt the RR planes) HERE, with the feature: the
            // one point where the render res is known and no in-flight frame
            // references the pass's heap (first dispatch ever, or the first
            // after resize_output's wait_idle).
            if let Some(gp) = &n.guides {
                match gp.borrow_mut().ensure(&self.d3d.device, rw, rh, rr.guide_inputs()) {
                    Ok(()) => n.guides_failed.set(false),
                    Err(e) => {
                        eprintln!(
                            "fg: guide-plane creation failed ({e}) — falling back to the \
                             raw RR planes (expect reflection shimmer)"
                        );
                        n.guides_failed.set(true);
                    }
                }
            }
            eprintln!(
                "fg: raw-NGX DLSS-G live — 1 generated frame per rendered frame (pair-present)"
            );
        }
        // dims == (rw, rh) from here (a stable move recreated in place
        // above; an unstable one returned above). A res that bounced BACK
        // before the threshold clears its pending count.
        n.res_pend.set(((0, 0), 0));
        let id = n.frame_id.get() + 1;
        n.frame_id.set(id);
        let (dep, mv) = rr.fg_inputs();
        let (psr, npsr, uav) = (
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let list = &self.d3d.list;
        // The guide pass converts the RR planes into what the FG snippet's
        // contracts actually want (see ngxfg_guides.rs): depth = the [0,1]
        // clip mapping of the matrices below (depth_inverted stays 0 — both
        // encodings grow with distance), MVs = the virtual-image blend that
        // stops mirror reflections being dragged with the surface. Each
        // lever falls back to the raw plane independently.
        let guide_pass = match (&n.guides, n.guides_failed.get()) {
            (Some(gp), false) => Some(gp.borrow()),
            _ => None,
        };
        if let Some(gp) = &guide_pass {
            // world -> PREVIOUS clip: column-vector composition, right to
            // left (world -> view -> clip -> prev clip).
            let m_prev = fc.clip_to_prev_clip * fc.view_to_clip * fc.world_to_view;
            let m = m_prev.to_cols_array_2d();
            let (near, far) = (fc.near, fc.far);
            let tanh = (fc.fov_y * 0.5).tan();
            let v4 = |v: glam::Vec3A| [v.x, v.y, v.z, 0.0];
            // Round 3: bake this frame's firefly splat rows (screen-space —
            // see ngxfg_guides.rs). ffc = 0 is the structural off: day
            // frames, --no-fireflies, and the FR_NGXFG_FFMV=off A/B all
            // execute the pre-round-3 kernel stream bit-identically.
            let ff_table = if n.ffmv_off || ff.count == 0 {
                ngxfg_guides::FfTable::empty()
            } else {
                ngxfg_guides::ff_guide_rows(
                    ff,
                    &n.prev_ff.get(),
                    rw as f32,
                    rh as f32,
                    fc.fov_y,
                    fc.pos,
                    fc.forward,
                    &(fc.view_to_clip * fc.world_to_view),
                    &m_prev,
                )
            };
            let p = ngxfg_guides::GuideParams {
                w: rw,
                h: rh,
                a: far / (far - near),
                b: -near * far / (far - near),
                m,
                org: v4(fc.pos),
                fwd: v4(fc.forward),
                // The CamBasis pre-scaling (camera.rs): right by
                // tan(fov/2)*aspect, up by tan(fov/2).
                rgt: v4(fc.right * (tanh * fc.aspect)),
                upv: v4(fc.up * tanh),
                rmv: (!n.rmv_off) as u32,
                // The SAME far the pack clamps a missed reflection to
                // (`spec_hit_t`), so the kernel can recognize "reflected sky"
                // and reproject it as a direction instead of a point 2*diag
                // away. Both come from dlss::near_far — keep them one source —
                // but the plane is R16F, so the compare threshold is far's f16
                // FLOOR (`wire_cam_far`): the exact f32 never fires when f16
                // rounds far down (THE WORLD's 138.56 -> 138.5), which
                // silently re-opened the round-2 sky parallax on world water.
                cam_far: ngxfg_guides::wire_cam_far(far),
                // Round 4. `t_prev == t_cur` whenever there is no retained
                // frame yet, when the clock is pinned (--check*'s
                // CLOUD_CHECK_TIME), or when the lever is off — all three
                // give a zero gradient delta, i.e. the round-2/3 kernel.
                t_cur: cl.time,
                t_prev: n.prev_clock.get().unwrap_or(cl.time),
                diag: cl.diag,
                ripplemv: (!n.ripplemv_off) as u32,
                ripdt: (!n.ripdt_off) as u32,
                _pad2: 0.0,
            };
            gp.record(list, &p, slot, &ff_table);
        }
        let depth_res = match &guide_pass {
            Some(gp) if !n.depth_linear => gp.clip().as_raw(),
            _ => dep.as_raw(),
        };
        let motion_res = match &guide_pass {
            Some(gp) if !n.rmv_off => gp.mv().as_raw(),
            _ => mv.as_raw(),
        };
        unsafe {
            list.ResourceBarrier(&[
                transition(&tgt_color, psr, npsr),
                transition(&n.out, psr, uav),
            ]);
        }
        // FR_NGXFG_TONEMAP probe: compress the color source into the scratch and
        // hand NGX that instead. Deliberately AFTER the barrier above — that is
        // what puts it into NON_PIXEL_SHADER_RESOURCE, the state a COMPUTE
        // SRV read requires (leaving it PSR is the exact debug-layer error the
        // bloom pyramid documents, and GBV-only, so it would go unnoticed).
        let tone_color = match &n.tone {
            Some(tp) => {
                let mut tpm = tp.borrow_mut();
                match tpm.ensure(&self.d3d.device, rr.ow, rr.oh, &tgt_color, &n.out) {
                    Ok(()) => {
                        tpm.record_compress(list, n.tone_mode);
                        unsafe {
                            list.ResourceBarrier(&[transition(tpm.scratch(), uav, npsr)]);
                        }
                        Some(tpm.scratch().as_raw())
                    }
                    Err(e) => {
                        eprintln!("fg: FR_NGXFG_TONEMAP ensure failed ({e}) — probe off this frame");
                        None
                    }
                }
            }
            None => None,
        };
        // Matrix majority: `row_major` matches what shim_constants hands SL,
        // but SL's closed plugin may itself transpose before setting the
        // DLSSG params — quinlight's identity matrices were transpose-
        // invariant, so this was never validated. FR_NGXFG_MAT=col walks it;
        // FR_NGXFG_CAM=identity substitutes quinlight's whole proven camera
        // block (matrices AND basis) to isolate camera plumbing entirely.
        const IDENT: [f32; 16] =
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mat = |m: &glam::Mat4| -> [f32; 16] {
            if n.cam_identity {
                IDENT
            } else if n.mat_col {
                m.to_cols_array()
            } else {
                row_major(m)
            }
        };
        let d = ngxfg::FrDlssgDispatch {
            cmdlist: list.as_raw(),
            // The probe swaps ONLY this: the compressed scratch in place of the
            // linear color source. Every other field is byte-identical, which
            // is what makes it an A/B of one variable.
            color: tone_color.unwrap_or_else(|| tgt_color.as_raw()),
            motion: motion_res,
            depth: depth_res,
            output: n.out.as_raw(),
            frame_id: id,
            reset: fc.reset as i32,
            view_to_clip: mat(&fc.view_to_clip),
            clip_to_view: mat(&fc.clip_to_view),
            clip_to_prev_clip: mat(&fc.clip_to_prev_clip),
            prev_clip_to_clip: mat(&fc.prev_clip_to_clip),
            // Default: the RAW sample offset. Raw NGX does NOT want
            // Streamline's negation — see `jitter_mode` for how that
            // assumption shipped and how it was caught.
            jitter: match n.jitter_mode {
                1 => [0.0, 0.0],
                2 => [-fc.jitter.0, -fc.jitter.1], // "neg": the old, wrong default
                _ => [fc.jitter.0, fc.jitter.1],
            },
            // PIXEL scale — settled from dlssg-to-fsr3, which hands
            // DLSSG.MvecScale straight to FSR3's motionVectorScale (unit:
            // pixels) and works across shipped SL titles; the SDK header's
            // "[-1,1]" comment is stale, and the quinlight-era {1/rw,1/rh}
            // starved the snippet of geometry motion ~2000× (quinlight's
            // MVs were zero, so any scale "worked" there). FR_NGXFG_MV
            // walks the alternatives.
            mv_scale: match n.mv_mode {
                1 => [1.0 / rw as f32, 1.0 / rh as f32],
                2 => [-1.0, -1.0],
                3 => [-1.0 / rw as f32, -1.0 / rh as f32],
                _ => [1.0, 1.0],
            },
            // The identity arm is quinlight's literal camera block (its
            // dlssg_shim_d3d12.cpp lines 355-372).
            cam_pos: if n.cam_identity { [0.0; 3] } else { v3(fc.pos) },
            cam_up: if n.cam_identity { [0.0, 1.0, 0.0] } else { v3(fc.up) },
            cam_right: if n.cam_identity { [1.0, 0.0, 0.0] } else { v3(fc.right) },
            cam_fwd: if n.cam_identity { [0.0, 0.0, 1.0] } else { v3(fc.forward) },
            cam_near: if n.cam_identity { 0.1 } else { fc.near },
            cam_far: if n.cam_identity { 10_000.0 } else { fc.far },
            cam_fov: if n.cam_identity { 1.047_197_5 } else { fc.fov_y },
            cam_aspect: if n.cam_identity {
                rr.ow as f32 / rr.oh as f32
            } else {
                fc.aspect
            },
            rend_w: rw,
            rend_h: rh,
            // Grows with distance in both depth arms — not inverted.
            depth_inverted: 0,
        };
        let r = unsafe { ngxfg::frdlssg_dispatch(n.handle.get(), &d) };
        // Undo the compression IN PLACE on the NGX output, while it is still
        // UNORDERED_ACCESS. Presentation therefore needs no second path: the
        // tonemap reads n.out and sees linear values exactly as it does with
        // the probe off. Also hands the scratch back to UAV for next frame.
        if tone_color.is_some() {
            if let Some(tp) = &n.tone {
                let tpm = tp.borrow();
                tpm.record_expand(list, &n.out, n.tone_mode);
                unsafe {
                    list.ResourceBarrier(&[transition(tpm.scratch(), npsr, uav)]);
                }
            }
        }
        unsafe {
            list.ResourceBarrier(&[
                transition(&tgt_color, npsr, psr),
                transition(&n.out, uav, psr),
            ]);
        }
        if let Some(gp) = &guide_pass {
            gp.restore(list);
        }
        if r != 0 {
            eprintln!("fg: raw-NGX DLSS-G evaluate failed ({r}) — frame generation off");
            n.failed.set(true);
            n.primed.set(false);
            return false;
        }
        // A reset evaluate re-seeds the history but its OUTPUT pairs against
        // a stale frame — present real-only this frame, pairs from the next.
        let show = n.primed.get() && !fc.reset;
        n.primed.set(true);
        // The swarm NGX now retains — next frame's prev half of the round-3
        // MV pair. Set only on a SUCCESSFUL evaluate (beside `primed`), so a
        // skipped/failed dispatch keeps the pairing aligned with the frame
        // the feature actually holds.
        n.prev_ff.set(*ff);
        // Same rule, same reason, for the round-4 ripple clock: it must name
        // the frame the feature retained, not the last one we recorded.
        n.prev_clock.set(Some(cl.time));
        show
    }

    /// The pair-present tail for the RR arms under raw-NGX frame generation:
    /// evaluate, present the INTERPOLATED frame first (it sits between prev
    /// and current in time), then the real frame. Under vsync the two
    /// presents land a vblank apart — that IS the pacing; no handshake is
    /// needed because nothing generates behind our back.
    fn ngxfg_tail(
        &mut self,
        slot: usize,
        fc: &dlss::FrameConstants,
        ff: &crate::fireflies::Fireflies,
        cl: &crate::clouds::Clouds,
    ) -> Result<()> {
        let show = self.ngxfg_dispatch(slot, fc, ff, cl);
        if let Some(n) = &self.fg_n {
            n.pair.set(show);
        }
        // FR_NGXFG_SHOW walks WHAT the two presents show, never the pacing
        // (same two Presents either way): 1 = interp twice (inspect the
        // generated frames at full rate), 2 = real twice (nothing NGX-made
        // on screen — artifacts that survive are the present path's).
        let show_mode = self.fg_n.as_ref().map_or(0, |n| n.show_mode);
        // The REAL half's tonemap slot follows the session's interpolation
        // source (SRV_SLOT_QUIN under --quinlight, SRV_SLOT_RR otherwise).
        let real_slot =
            self.ngxfg_target().map_or(tonemap::SRV_SLOT_RR, |(_, s)| s);
        let (mid_slot, mut end_slot) = match show_mode {
            1 => (tonemap::SRV_SLOT_NGXFG, tonemap::SRV_SLOT_NGXFG),
            2 => (real_slot, real_slot),
            _ => (tonemap::SRV_SLOT_NGXFG, real_slot),
        };
        // A non-generating frame (unprimed reset, res-moved skip — which
        // returns BEFORE the evaluate — or a failed evaluate) has nothing
        // fresh in fg_n.out: under SHOW=interp the single present must fall
        // back to the real frame, or a failed session re-presents a stale /
        // never-written texture indefinitely.
        if !show {
            end_slot = real_slot;
        }
        // FR_NGXFG_PACE=1: per-frame pacing probe (kept diagnostic lever, the
        // FR_DLSSG_NO_RR class) — logs the backbuffer indices both halves
        // record into and the swapchain's own frame statistics, so a pacing
        // anomaly (skipped/duplicated vblanks, present-queue stalls, a
        // buffer-rotation break) is diffable between arms from a log instead
        // of argued about. A 2026-07 flicker report was triaged with it:
        // wavefront and DXR arms measured byte-identical rotation
        // (1/2 -> 3/4 -> 5/0 mod PAIR_BACKBUFFERS, one pair per frame).
        // Caveat: run the window FOREGROUND — DWM retires an occluded
        // window's presents unthrottled, which voids the vsync half of the
        // statistics.
        static PACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let pace = *PACE.get_or_init(|| std::env::var("FR_NGXFG_PACE").is_ok());
        let t0 = pace.then(std::time::Instant::now);
        let bb0 = pace.then(|| unsafe { self.d3d.swapchain.GetCurrentBackBufferIndex() });
        if show {
            self.fullscreen_to_backbuffer(true, mid_slot, 1.0);
            self.d3d.present_mid(slot)?;
        }
        let bb1 = pace.then(|| unsafe { self.d3d.swapchain.GetCurrentBackBufferIndex() });
        self.fullscreen_to_backbuffer(true, end_slot, 1.0);
        let r = self.d3d.end_frame(slot);
        if pace {
            let mut st = windows::Win32::Graphics::Dxgi::DXGI_FRAME_STATISTICS::default();
            let stats = unsafe { self.d3d.swapchain.GetFrameStatistics(&mut st) };
            let id = self.fg_n.as_ref().map_or(0, |n| n.frame_id.get());
            eprintln!(
                "fg-pace: id={} show={} bb={}/{} ms={:.2} pc={} spc={} src={} {}",
                id,
                show as u8,
                bb0.unwrap_or(9),
                bb1.unwrap_or(9),
                t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0),
                st.PresentCount,
                st.SyncQPCTime / 10_000,
                st.SyncRefreshCount,
                if stats.is_err() { "(stats-err)" } else { "" },
            );
        }
        r
    }

    /// Stash last frame's wall-clock ms for this loop's prepare (main.rs owns
    /// the clock; pacing quality follows this number).
    pub fn fg_set_frame_ms(&self, ms: f32) {
        if let Some(fg) = &self.fg {
            fg.frame_ms.set(ms);
        }
    }

    /// Drive the FI swapchain's UI registration to `active` (the HudFi
    /// display-space texture) or null. Dedup'd via `ui_reg` — the proxy is
    /// re-configured only on TRANSITIONS, never per steady frame. A failed
    /// configure sheds loudly ONCE (`ui_shed`) and the funnel's baked
    /// backbuffer draw covers every frame after — never a black HUD, never a
    /// session failure. (Known edge: a failed UNREGISTER after a successful
    /// register can leave the proxy compositing a stale UI onto generated
    /// frames beside the baked one — accepted; the alternative is retrying a
    /// failing configure per frame.)
    fn fg_register_ui(&self, active: bool) {
        let Some(fg) = &self.fg else { return };
        if fg.ui_shed.get() || fg.ui_reg.get() == active {
            return;
        }
        let res = if active {
            let Some(ui) = &self.fg_ui else { return };
            fg.sc.register_ui(Some(ui.resource().as_raw()))
        } else {
            fg.sc.register_ui(None)
        };
        match res {
            Ok(()) => {
                fg.ui_reg.set(active);
                if fg.trace {
                    eprintln!(
                        "fg-trace: ui resource {} (frame_id {})",
                        if active { "registered" } else { "unregistered" },
                        fg.frame_id.get()
                    );
                }
            }
            Err(e) => {
                fg.ui_shed.set(true);
                eprintln!("fg: ui registration failed ({e}) — HUD stays baked pre-present");
            }
        }
    }

    /// Configure the FI swapchain DISABLED at the current frame id, once
    /// (idempotent via `live`). Used by the toggle, the pause gate, and the
    /// funnel's not-prepared fallback.
    fn fg_disable_now(&self) {
        let Some(fg) = &self.fg else { return };
        // Null the UI registration with the disable (idempotent via ui_reg,
        // BEFORE the ctx/live early-returns — teardown routes through here,
        // and the registration lives on the SWAPCHAIN context (fg.sc), which
        // outlives the effect ctx: the proxy must not hold the HudFi pointer
        // past this call even when the effect ctx is already gone).
        self.fg_register_ui(false);
        let Some(ctx) = &fg.ctx else { return };
        if !fg.live.get() {
            return;
        }
        let cfg = ffx::FfxShimFgConfig {
            swapchain: self.d3d.swapchain.as_raw(),
            enabled: 0,
            allow_async: 0,
            hudless: ffx_sys::FfxShimRes::NULL,
            flags: 0,
            only_present_generated: 0,
            rect_left: 0,
            rect_top: 0,
            rect_w: self.d3d.width,
            rect_h: self.d3d.height,
            min_max_luminance: fg.luminance,
            frame_id: fg.frame_id.get(),
        };
        if let Err(e) = ctx.configure(&cfg) {
            eprintln!("fg: disable configure failed ({e})");
        }
        fg.live.set(false);
        // Transition-only (the early-return above latches on `live`), so this
        // is keypress-rate at most — the pause/resume pair is what makes the
        // FI layer's on/off cadence visible in a session log.
        let n = fg.pauses.get() + 1;
        fg.pauses.set(n);
        eprintln!(
            "fg: interpolation paused (mode-switch straddle / plain toggle / hold / teardown; pause #{n}, frame_id {})",
            fg.frame_id.get()
        );
    }

    /// Record this frame's FG work: advance the frame id by exactly one (the
    /// ffx contract), configure the FI swapchain live, and record the
    /// PrepareV2 dispatch (depth + motion-vector dilation) into the frame's
    /// list. Call AFTER the depth/mvec planes are final for the frame (post
    /// feed/upload — both stay in NON_PIXEL_SHADER_RESOURCE, the compute-read
    /// state the dispatch declares), BEFORE `fullscreen_to_backbuffer`.
    /// `mv_scale` converts the plane's MV convention to the provider's UV
    /// space: (1,1) for the pixel-space RG16F trio planes (the FSR3-upscale
    /// precedent), (rw,rh) for the FSR4-RR UV-delta plane.
    fn fg_prepare(
        &self,
        list: &ID3D12GraphicsCommandList,
        fc: &dlss::FrameConstants,
        depth: &ID3D12Resource,
        mvec: &ID3D12Resource,
        mv_scale: [f32; 2],
    ) {
        let Some(fg) = &self.fg else { return };
        let Some(ctx) = &fg.ctx else { return };
        if !fg.enabled {
            return;
        }
        // The mode-switch straddle: consume the flag and prepare NOTHING this
        // frame — the funnel then finds `prepared` unset and configures the
        // FI proxy disabled for exactly one passthrough present, after which
        // the next frame's prepare resumes generation. frame_id deliberately
        // does not advance (the disable configure reuses the last id, the
        // K-toggle sequence bit-for-bit).
        if fg.skip_prepare.replace(false) {
            return;
        }
        let frame_ms = fg.frame_ms.get();
        let id = fg.frame_id.get() + 1;
        fg.frame_id.set(id);
        let cfg = ffx::FfxShimFgConfig {
            swapchain: self.d3d.swapchain.as_raw(),
            enabled: 1,
            allow_async: 0,
            hudless: ffx_sys::FfxShimRes::NULL,
            flags: 0,
            only_present_generated: 0,
            rect_left: 0,
            rect_top: 0,
            rect_w: self.d3d.width,
            rect_h: self.d3d.height,
            min_max_luminance: fg.luminance,
            frame_id: id,
        };
        if let Err(e) = ctx.configure(&cfg) {
            eprintln!("fg: configure failed ({e})");
            return;
        }
        if !fg.live.get() {
            eprintln!("fg: interpolation resumed (frame_id {id})");
        }
        fg.live.set(true);
        if fg.trace {
            let res = (depth.as_raw() as usize, mvec.as_raw() as usize);
            if fg.last_res.replace(res) != res {
                eprintln!(
                    "fg-trace: prepare resource set changed (depth {:#x}, mvec {:#x}) at frame_id {id}",
                    res.0, res.1
                );
            }
            if fc.reset {
                eprintln!(
                    "fg-trace: prepare reset=1 at frame_id {id} (render {}x{}, mv_scale {:?}, dt {:.1} ms)",
                    fc.rw, fc.rh, mv_scale, frame_ms
                );
            }
        }
        let prep = ffx::FfxShimFgPrepare {
            cmdlist: list.as_raw(),
            frame_id: id,
            flags: 0,
            render_w: fc.rw as u32,
            render_h: fc.rh as u32,
            // The ffx-family jitter polarity settled for the upscaler applies
            // here too (one constant, one family).
            jitter: [crate::fsr::JITTER_SIGN * fc.jitter.0, crate::fsr::JITTER_SIGN * fc.jitter.1],
            mv_scale,
            frame_time_delta_ms: frame_ms.clamp(0.1, 200.0),
            reset: fc.reset as i32,
            cam_near: fc.near,
            cam_far: fc.far,
            cam_fovy: fc.fov_y,
            view_space_to_meters: 1.0,
            depth: ffx_sys::FfxShimRes {
                resource: depth.as_raw(),
                state: ffx_sys::RES_STATE_COMPUTE_READ,
            },
            motion_vectors: ffx_sys::FfxShimRes {
                resource: mvec.as_raw(),
                state: ffx_sys::RES_STATE_COMPUTE_READ,
            },
            cam_pos: v3(fc.pos),
            cam_up: v3(fc.up),
            cam_right: v3(fc.right),
            cam_fwd: v3(fc.forward),
        };
        if let Err(e) = ctx.prepare(&prep) {
            eprintln!("fg: prepare failed ({e})");
            return;
        }
        fg.prepared.set(true);
    }

    /// The single backbuffer bind point — all 16 present arms funnel here.
    ///
    /// The presentation curve is read from `self.tone` rather than threaded
    /// through every arm: that is what makes a display change (a new monitor, or
    /// Windows HDR toggled underneath us) a one-field retune instead of a
    /// signature change at 16 call sites. Glare is likewise applied here and
    /// nowhere else, on whatever this frame's tonemap source turns out to be.
    fn fullscreen_to_backbuffer(&self, use_tonemap: bool, srv_slot: u32, inv_samples: f32) {
        // Frame generation handshake: a frame that reaches presentation
        // WITHOUT an fg_prepare (plain arms, mode switches, holds) must not
        // let the FI swapchain interpolate against stale motion — configure
        // it disabled first (idempotent). A prepared frame consumes its flag
        // — and `fi_live` remembers that THIS present actually interpolates,
        // which is what gates the post-interpolation HUD below.
        let mut fi_live = false;
        if let Some(fg) = &self.fg {
            if !fg.prepared.replace(false) {
                self.fg_disable_now();
            } else {
                fi_live = true;
            }
        }
        // XeSS-FG's half — READ prepared without consuming (xefg_end_frame
        // consumes after the XeLL present markers, which bracket the Present
        // — AFTER this funnel).
        if let Some(x) = &self.fg_x {
            if x.on.get() && !x.prepared.get() {
                x.sc.set_enabled(false);
                x.on.set(false);
            }
        }
        // HUD overlay dirty rects: consume whatever the frame staged. Recorded
        // BEFORE the backbuffer transition (any point in the list before the
        // draws is fine — begin_frame's fence wait already made this slot's
        // ring memory safe). Runs even while the HUD is hidden, so the texture
        // stays current and re-showing needs no special case.
        let hud_slot = self.d3d.frame_index % d3d12::FRAMES_IN_FLIGHT;
        self.hud.record_upload(&self.d3d.list, hud_slot);

        // Frame-generation UI: on interpolating presents, render the HUD into
        // the display-space HudFi target and hand THAT to the proxy so the UI
        // is composited AFTER interpolation — the baked backbuffer HUD gets
        // warped by scene motion on every generated frame (the jumping-HUD
        // defect). ffx: registered UI resource, composited onto BOTH pair
        // halves, so the baked draw is SKIPPED once the registration is live.
        // XeSS-FG: RES_UI tag (xefg_prepare) with UI_MODE_AUTO resolving to
        // BACKBUFFER_UITEXTURE — the proxy REFINES the UI region on generated
        // frames, so the baked draw stays. Non-FG arms (DLSS-G pair-present,
        // plain, holds, CPU presenters) are untouched by construction.
        // `!ui_shed`: after a tag failure nothing consumes the target, so
        // the pre-pass would be a wasted fullscreen draw on every prepared
        // frame for the rest of the session (or until a resize re-arms it).
        let xefg_hud = self
            .fg_x
            .as_ref()
            .is_some_and(|x| x.on.get() && x.prepared.get() && !x.ui_shed.get())
            && self.fg_ui.is_some()
            && self.hud.drawable();
        let fi_hud = fi_live
            && self.hud.drawable()
            && self.fg.as_ref().is_some_and(|fg| !fg.ui_shed.get())
            && self.fg_ui.is_some();
        if fi_hud || xefg_hud {
            self.fg_ui.as_ref().unwrap().record(&self.d3d.list, &self.passes, self.tone);
        }
        if fi_hud {
            self.fg_register_ui(true);
        } else {
            // Dedup'd null: hidden HUD, disabled/passthrough presents, holds.
            self.fg_register_ui(false);
        }

        // Glare (src/bloom.rs): a display-stage pass on whatever the tonemap is
        // about to read, so EVERY GPU chain — RR, XeSS, FSR, plain, DXR — gets it
        // from this one place, and nothing upstream (accum, the temporal cache,
        // the upscaler guides) is touched. `--no-bloom` records nothing and
        // passes strength 0, which is what makes it bit-identical to the
        // pre-bloom presentation.
        let bloom_src = if use_tonemap && crate::bloom::enabled() {
            self.tonemap_source(srv_slot)
        } else {
            None
        };
        let bloom = if let Some(src) = bloom_src {
            // Every tonemap source RESTS in PIXEL_SHADER_RESOURCE — that is the
            // state the draw below wants. The pyramid's downsample, though, is a
            // COMPUTE dispatch, and D3D12 requires NON_PIXEL_SHADER_RESOURCE for
            // a shader-resource read outside the pixel stage: leaving it in PSR
            // is a debug-layer error on every bloomed frame (both are read states,
            // so drivers wave it through, which is exactly why it went unnoticed —
            // the layer only arms under --gpu-debug). Borrow it for the pyramid
            // and hand it back before the draw.
            let (psr, npsr) = (
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
            unsafe { self.d3d.list.ResourceBarrier(&[transition(src, psr, npsr)]) };
            self.bloom.record(&self.d3d.list, srv_slot);
            unsafe { self.d3d.list.ResourceBarrier(&[transition(src, npsr, psr)]) };
            let (gw, gh) = self.bloom.glare_dims();
            (crate::bloom::strength(), 1.0 / gw as f32, 1.0 / gh as f32)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Auto-exposure's luminance meter (gpu/autoexp.rs): collect this
        // slot's 2-frame-old mean first (begin_frame's fence wait already
        // retired that frame — the gputime contract; `hud_slot` above IS the
        // fence-waited slot), then record this frame's reduction on the same
        // source, with bloom's own PSR<->NPSR borrow. Gated on the lever so
        // `--no-auto-exposure` records nothing (the --no-bloom discipline).
        // The blit arms (use_tonemap false) are CPU-tonemapped and meter
        // CPU-side instead (CpuPresent).
        if use_tonemap && crate::autoexp::enabled() {
            if let Some(src) = self.tonemap_source(srv_slot) {
                if let Some(v) = self.autoexp.collect(hud_slot) {
                    self.meter.set(Some(v));
                }
                let (sw, sh) = {
                    let d = unsafe { src.GetDesc() };
                    (d.Width as u32, d.Height)
                };
                let (psr, npsr) = (
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                );
                unsafe { self.d3d.list.ResourceBarrier(&[transition(src, psr, npsr)]) };
                self.autoexp.record(&self.d3d.list, srv_slot, sw, sh, inv_samples, hud_slot);
                unsafe { self.d3d.list.ResourceBarrier(&[transition(src, npsr, psr)]) };
            }
        }

        let bb = unsafe { self.d3d.swapchain.GetCurrentBackBufferIndex() };
        let backbuffer = &self.d3d.backbuffers[bb as usize];
        unsafe {
            self.d3d.list.ResourceBarrier(&[transition(
                backbuffer,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )]);
        }
        let pso = if use_tonemap { &self.passes.tonemap_pso } else { &self.passes.blit_pso };
        self.passes.record(
            &self.d3d.list,
            pso,
            srv_slot,
            inv_samples,
            bloom,
            self.tone,
            self.d3d.rtv_handle(bb),
            self.d3d.width,
            self.d3d.height,
        );
        // The --waveviz overlay composite: a fullscreen draw over the
        // PRESENTED image — the HUD's shape and insertion point, which is
        // what makes it work under every upscaler (the overlay is blended
        // AFTER reconstruction; feeding hash colors INTO a temporal model
        // was rejected — it would smear per-frame tickets). Reads the live
        // arm's render-res tbuf (tickets) as the t2 root SRV, bracketed
        // UNORDERED_ACCESS <-> PIXEL_SHADER_RESOURCE around the draw (the
        // bloom bracket's pattern). Below the HUD so the menu stays
        // readable over it.
        if trace::waveviz_on() && trace::waveviz_live() {
            let src = match self.waveviz_src.get() {
                WvSrc::Trace => self.trace.as_ref().map(|t| (&t.tbuf, t.rw, t.rh)),
                WvSrc::Dxr => self.dxr.as_ref().map(|d| (&d.tbuf, d.rw, d.rh)),
                WvSrc::None => None,
            };
            if let (Some((tbuf, rw, rh)), Some(pso)) =
                (src, self.passes.waveviz_pso.as_ref())
            {
                unsafe {
                    self.d3d.list.ResourceBarrier(&[transition(
                        tbuf,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    )]);
                }
                self.passes.record_waveviz(
                    &self.d3d.list,
                    pso,
                    unsafe { tbuf.GetGPUVirtualAddress() },
                    rw,
                    rh,
                    self.d3d.width,
                    self.d3d.height,
                    self.tone,
                    self.d3d.rtv_handle(bb),
                );
                unsafe {
                    self.d3d.list.ResourceBarrier(&[transition(
                        tbuf,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                }
            }
        }
        // The HUD/menu composite: a second fullscreen draw over the tonemapped
        // frame while the backbuffer is still a render target — this ONE
        // insertion covers every present arm. `Passes::record` rebinds
        // everything, so the extra draw needs no state bookkeeping; the hud
        // PSO blends premultiplied-over and its PS reads only the
        // scale/mode lanes of the shared root-constant layout.
        //
        // SKIPPED only when the ffx FI proxy holds a LIVE UI registration for
        // this interpolating present (the proxy composites the registered
        // texture onto both pair halves — baking it too would double-HUD the
        // real frames). The `ui_reg` check is what guarantees a registration
        // FAILURE still bakes the HUD this very frame — zero HUD-less frames
        // by construction. XeSS-FG deliberately keeps the baked draw (its
        // proxy REFINES the backbuffer's UI region from the tag).
        if self.hud.drawable()
            && !(fi_hud && self.fg.as_ref().is_some_and(|fg| fg.ui_reg.get()))
        {
            self.passes.record(
                &self.d3d.list,
                &self.hud.pso,
                tonemap::SRV_SLOT_OVERLAY,
                1.0,
                (0.0, 0.0, 0.0),
                self.tone,
                self.d3d.rtv_handle(bb),
                self.d3d.width,
                self.d3d.height,
            );
        }
        unsafe {
            self.d3d.list.ResourceBarrier(&[transition(
                backbuffer,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            )]);
        }
        if bloom.0 > 0.0 {
            self.bloom.restore(&self.d3d.list);
        }
        // Remember what was presented so `present_again` (the pause menu's
        // trace-free hold) can re-run this exact call.
        self.last_present.set(Some((use_tonemap, srv_slot, inv_samples)));
    }

    /// Stage a HUD frame's dirty rects for the next recorded present (main
    /// loop, once per frame, before the present arm runs).
    pub fn hud_stage(&self, frame: crate::hud::HudFrame) {
        self.hud.stage(frame);
    }

    /// Whether the HUD composite draw runs (fed per frame from the session's
    /// HUD state; staging continues regardless).
    pub fn set_hud_visible(&self, on: bool) {
        self.hud.visible.set(on);
    }

    /// Re-present the last presented image + the current HUD overlay without
    /// tracing, evaluating any upscaler, or advancing any history — the
    /// pause-menu hold, generalizing `present_hold`/`present_dxr_hold` to
    /// every arm (every tonemap source rests in PIXEL_SHADER_RESOURCE between
    /// frames). Errs when nothing was ever presented or the recorded source
    /// no longer exists (a mode switch dropped its textures) — the caller
    /// falls back to rendering a normal frame.
    pub fn present_again(&mut self) -> Result<()> {
        let Some((use_tonemap, srv_slot, inv_samples)) = self.last_present.get() else {
            return Err("nothing presented yet".into());
        };
        // The blit/hdr uploads always exist; every other slot's source is
        // owned by an Option field that a mode switch may have dropped.
        let src_alive = matches!(srv_slot, tonemap::SRV_SLOT_BLIT | tonemap::SRV_SLOT_HDR)
            || self.tonemap_source(srv_slot).is_some();
        if !src_alive {
            return Err("last-present source was dropped".into());
        }
        let slot = self.d3d.begin_frame()?;
        self.fullscreen_to_backbuffer(use_tonemap, srv_slot, inv_samples);
        self.d3d.end_frame(slot)
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        // xessDestroyContext / ffxDestroyContext require all pending command
        // lists complete and a live device: drain the queue, then drop the
        // contexts here — before the field-order teardown releases the
        // swapchain/device.
        if self.xess.is_some()
            || self.fsr.is_some()
            || self.fg.is_some()
            || self.fg_x.is_some()
            || self.fg_n.is_some()
            || self.rr_feature.is_some()
        {
            let _ = self.d3d.wait_idle();
            self.xess = None;
            self.fsr = None;
            // Raw-NGX DLSS-G: release the feature with the queue drained and
            // the device alive (its texture drops by field order).
            if let Some(n) = &self.fg_n {
                let h = n.handle.replace(std::ptr::null_mut());
                if !h.is_null() {
                    unsafe { ngxfg::frdlssg_destroy(h) };
                }
            }
            // Raw-NGX DLSS-RR: same discipline — feature released with the
            // queue drained; the session (ngxrr) drops by field order and
            // releases its refcounted NGX init.
            if let Some(f) = self.rr_feature.take() {
                f.destroy();
            }
            // FG, in three ordered steps.
            //
            // First unregister: a generating session leaves the FI swapchain
            // CONFIGURED LIVE, which is a registration holding the proxy's
            // frame-generation callback into the effect context dropped below,
            // and destroying a context the proxy still points into is not
            // defensible whatever else is true. `fg_disable_now` is the
            // existing unregister — the same `enabled: 0` configure the K
            // toggle, the pause gate and the funnel's not-prepared fallback
            // use, idempotent via `live`, so a session that was not generating
            // pays nothing. Teardown is simply the fourth caller.
            //
            // Honest about what this buys: it is API hygiene, NOT the fix.
            // Adding it did not move the teardown crash at all — measured, on
            // this exact stack. The leak below is what fixes that.
            self.fg_disable_now();
            // Then retire pending paced presents while the queue is live (they
            // can actually retire now that generation is off), and only then
            // drop the display-size effect context. The SWAPCHAIN context
            // stays for field order — it must destroy AFTER d3d releases its
            // proxy/backbuffer refs (fg is declared after d3d exactly for
            // this).
            if let Some(mut fg) = self.fg.take() {
                fg.sc.wait_for_presents();
                fg.ctx = None;
                // ...and then DELIBERATELY LEAK the FI swapchain context.
                //
                // `ffxDestroyContext` on it does not reliably return on this
                // stack: measured over repeated runs it either spins forever
                // inside the provider (a QPC busy-wait, main thread, with the
                // proxy's presenter thread already gone -- 90-thread dump has
                // nobody left to satisfy it) or faults outright (0xC0000005
                // ~3.7 s in). Neither depends on anything we can see: the wrap
                // is configured disabled, presents are retired, the effect
                // context is gone, and every documented ordering constraint
                // (d3d12.rs:419 -- proxy refs released first) is met by field
                // order. Attaching the FG version desc, which the provider
                // does demand, changed nothing here either.
                //
                // This is the LAST act of the process, so the leak costs
                // exactly nothing the OS is not about to reclaim, and it turns
                // a crash-or-hang on every FG session into a clean exit. It is
                // scoped to process teardown ON PURPOSE: `FgSwapchain::drop`
                // still destroys normally wherever a context must go before a
                // new chain can exist on the HWND. (The old colour-space
                // unwind path is gone — a refused G2084 re-declare now
                // relabels the session Sdr10 and keeps the proxy.)
                //
                // NOT the real fix, which is architectural: quinlight-player
                // drives frame interpolation itself
                // (ffxFrameInterpolationContextDestroy + its own pair-present)
                // and never wraps the swapchain, so it cannot reach this path
                // at all. See the FG section in CLAUDE.md.
                std::mem::forget(fg.sc);
            }
        }
    }
}

/// glam Mat4 (column-major) -> the NGX-family row-major float[16]. This is
/// THE transpose boundary — nothing else in the codebase reorders matrices.
fn row_major(m: &glam::Mat4) -> [f32; 16] {
    m.transpose().to_cols_array()
}

fn v3(v: glam::Vec3A) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// A headless upscaler session for the cinematic capture mode: ONE chain
/// level's SDK context + resource set at 100% render scale (render res ==
/// output res — DLAA-grade reconstruction), hosted on a `HeadlessGpu` instead
/// of a swapchain. The engine states and eval middles are the interactive
/// session's own (`probe_native`, `record_xess_eval`, `record_fsr3_upscale`,
/// `record_fsr_rr_sequence`, `rr_ngx_sequence`), so the two paths cannot
/// drift; what this type adds is only the narrow driver API — probe, wire,
/// per-sub-frame eval, and a linear-f32 readback of the reconstructed output.
///
/// Drop discipline: drop this BEFORE the `HeadlessGpu` (locals in reverse
/// declaration order do it naturally) — the XeSS/ffx context destructors need
/// a live device, and every harness submit already blocked, so "completed
/// command lists" holds by construction (the DLSSD feature release rides the
/// same argument).
pub struct CineUp {
    rr: Option<rr::RrResources>,
    /// The DLSSD feature paired with `rr` (created per shot resolution —
    /// the caller drops the whole CineUp on a res change, harness idle).
    rr_feat: Option<ngxrr::RrFeature>,
    xess: Option<XessState>,
    fsr: Option<FsrState>,
    /// Which chain level won: "dlss-rr" | "fsr4-rr" | "xess" | "fsr3".
    pub name: &'static str,
    ow: u32,
    oh: u32,
    readback: Option<d3d12::ReadbackBuffer>,
}

impl CineUp {
    /// Probe the chain (DLSS-RR -> FSR4-RR -> XeSS -> FSR3, honoring
    /// `opts.chain`) for an output size, at 100% render scale. `ngxrr` is the
    /// harness's raw-NGX RR session — Some means the DLSS level already
    /// passed its adapter + availability probes, so it wins the chain. None =
    /// chain exhausted (the caller falls back to accumulation, loudly).
    pub fn probe(
        device: &ID3D12Device,
        ngxrr: Option<&ngxrr::NgxRr>,
        opts: &GpuOptions,
        w: u32,
        h: u32,
    ) -> Option<CineUp> {
        let up = |rr, rr_feat, xess, fsr, name| CineUp {
            rr,
            rr_feat,
            xess,
            fsr,
            name,
            ow: w,
            oh: h,
            readback: None,
        };
        if let Some(nx) = ngxrr {
            // DLAA by construction: planes at the output size, feature
            // created with dlaa = true (the retired SL path expressed this
            // as a degenerate opt == min == max range; the raw create says
            // it honestly). No optimal-settings query: native is the point.
            match rr::RrResources::new(device, (w, h), (w, h), (w, h), w, h)
                .and_then(|r| Ok((nx.create_feature((w, h), (w, h), true)?, r)))
            {
                Ok((f, r)) => {
                    return Some(up(Some(r), Some(f), None, None, "dlss-rr"));
                }
                Err(e) => eprintln!(
                    "dlss: RR resource/feature creation failed ({e}) — falling through the chain"
                ),
            }
        }
        // First-hit-wins over the native levels, exactly the interactive
        // probe: `fsr` holds FSR4-RR where the RDNA4 provider exists, else
        // XeSS came up, else `fsr` holds the FSR 3.1 flavor (quin is off, so
        // the states are mutually exclusive and `fsr3` stays None).
        let (xess, fsr, _fsr3) = GpuContext::probe_native(device, opts, w, h, false);
        if let Some(fs) = fsr {
            let name = match fs.res {
                FsrRes::Rr(_) => "fsr4-rr",
                FsrRes::Up(_) => "fsr3",
            };
            return Some(up(None, None, None, Some(fs), name));
        }
        if let Some(x) = xess {
            return Some(up(None, None, Some(x), None, "xess"));
        }
        None
    }

    /// Wire this engine's input planes as a tracer's feed targets (`wire` =
    /// `TraceGpu::wire_feed` / `DxrGpu::wire_feed`). The per-engine helpers
    /// re-check the trace res against the SDK range, so a 100% res an engine
    /// cannot host fails loudly here — the caller falls back to accumulation.
    pub fn wire(
        &self,
        rw: u32,
        rh: u32,
        mut wire: impl FnMut(
            trace::FeedKind,
            &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
        ) -> Result<()>,
    ) -> Result<()> {
        if let Some(rr) = &self.rr {
            GpuContext::wire_rr_feed(rr, rw, rh, &mut wire)
        } else if let Some(x) = &self.xess {
            GpuContext::wire_xess_feed(x, rw, rh, &mut wire)
        } else if let Some(fs) = &self.fsr {
            GpuContext::wire_fsr_feed(fs, rw, rh, &mut wire)
        } else {
            Err("cinematic upscaler with no engine".into())
        }
    }

    /// Record this engine's evaluate on the harness's list — the exact middle
    /// the present arms record, minus the swapchain around it. Call AFTER the
    /// frame's `record_feed` on the same list.
    #[allow(clippy::too_many_arguments)]
    pub fn record_eval(
        &self,
        ngxrr: Option<&ngxrr::NgxRr>,
        list: &ID3D12GraphicsCommandList,
        fc: &dlss::FrameConstants,
        jitter: (f32, f32),
        reset: bool,
        frame_idx: u32,
        frame_ms: f32,
        prev_pos: Option<glam::Vec3A>,
        sky_sh: &crate::sh::Sh9,
    ) -> Result<()> {
        if let Some(rr) = &self.rr {
            let nx = ngxrr.ok_or("DLSS-RR engine without a raw-NGX session")?;
            let feat = self.rr_feat.as_ref().ok_or("DLSS-RR engine without a feature")?;
            unsafe {
                list.ResourceBarrier(&[transition(
                    &rr.output,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            rr_ngx_sequence(nx, feat, rr, list, fc)?;
            unsafe {
                list.ResourceBarrier(&[transition(
                    &rr.output,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                )]);
            }
            Ok(())
        } else if let Some(x) = &self.xess {
            GpuContext::record_xess_eval(x, list, fc.rw as u32, fc.rh as u32, jitter, reset)
        } else if let Some(fs) = &self.fsr {
            match &fs.res {
                FsrRes::Rr(_) => GpuContext::record_fsr_rr_sequence(
                    fs, list, fc, prev_pos, frame_idx, frame_ms, sky_sh,
                ),
                FsrRes::Up(_) => GpuContext::record_fsr3_upscale(fs, list, fc, frame_ms, None),
            }
        } else {
            Err("cinematic upscaler with no engine".into())
        }
    }

    /// Read the reconstructed output back as linear-RGB f32 triples — the
    /// cinematic writer's contract (`cine_write_frame` owns the ONE tone
    /// curve, so this must hand it linear light; `read_hdr_output` would
    /// SDR-tonemap on the way out). Runs once per OUTPUT frame; the readback
    /// buffer is persistent, allocated on first use.
    pub fn read_output(&mut self, hg: &mut trace::HeadlessGpu, out: &mut [f32]) -> Result<()> {
        let (w, h) = (self.ow as usize, self.oh as usize);
        assert_eq!(out.len(), w * h * 3);
        let pitch = d3d12::aligned_pitch(w * 8);
        if self.readback.is_none() {
            self.readback = Some(d3d12::ReadbackBuffer::new(&hg.device, pitch * h)?);
        }
        let rb = self.readback.as_ref().unwrap();
        let output = if let Some(rr) = &self.rr {
            &rr.output
        } else if let Some(x) = &self.xess {
            &x.res.output
        } else if let Some(fs) = &self.fsr {
            fs.res.upscaled()
        } else {
            return Err("cinematic upscaler with no engine".into());
        };
        let fp = d3d12::footprint(
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            self.ow,
            self.oh,
            8,
            0,
        );
        hg.run(|list| unsafe {
            list.ResourceBarrier(&[transition(
                output,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyTextureRegion(
                &d3d12::loc_footprint(&rb.resource, fp),
                0,
                0,
                0,
                &d3d12::loc_subresource(output),
                None,
            );
            list.ResourceBarrier(&[transition(
                output,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        })?;
        let mut ptr = std::ptr::null_mut();
        unsafe { rb.resource.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| format!("readback Map: {e}"))?;
        let base = ptr as usize; // usize crosses the rayon closure; rows are disjoint
        use rayon::prelude::*;
        out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            let src: &[[half::f16; 4]] = unsafe {
                std::slice::from_raw_parts((base as *const u8).add(y * pitch) as *const _, w)
            };
            for (x, px) in src.iter().enumerate() {
                row[x * 3] = f32::from(px[0]);
                row[x * 3 + 1] = f32::from(px[1]);
                row[x * 3 + 2] = f32::from(px[2]);
            }
        });
        unsafe { rb.resource.Unmap(0, None) };
        Ok(())
    }
}
