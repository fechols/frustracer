//! GPU presentation layer: SDL2 hands us an HWND, we own a D3D12 device on
//! the NVIDIA adapter, a DXGI swapchain, and the upload/fullscreen-pass
//! machinery. Everything here consumes finished CPU frames after
//! `render_frame`/`resolve` return — no tracer state is touched.
//!
//! Milestones: M1 `present_cpu` (blit of the CPU-tonemapped frame),
//! M2 `present_hdr` (GPU tonemap of the raw accumulation), M3 Streamline
//! proxy plumbing, M4 `present_rr` (DLSS Ray Reconstruction).

pub mod adapter;
pub mod d3d12;
pub mod dxc;
pub mod dxr;
pub mod ffx;
pub mod ffx_rr;
pub mod ffx_sys;
pub mod ffx_up;
pub mod pix;
pub mod rr;
pub mod streamline;
pub mod streamline_sys;
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
use windows::Win32::Graphics::Dxgi::IDXGIFactory6;

pub struct GpuOptions {
    /// The temporal-upscaler fallback chain: which levels of
    /// DLSS-RR → FSR4-RR → XeSS → FSR3 to probe, in that fixed order — the
    /// first level whose support probe passes is wired for the session
    /// (exactly one upscaler per session: DLSS decides the SL-proxy-vs-native
    /// device split up front, and the native levels are first-hit-wins).
    /// Every level exhausted = plain presentation with a loud line.
    pub chain: crate::upchain::UpChain,
    /// Directory holding sl.interposer.dll and the SL plugins.
    pub sl_dir: String,
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
}

/// Which chain level a session actually wired — derived from the live state
/// (the Options can never disagree with reality). `Plain` = nothing wired
/// (--no-upscale or chain exhausted).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WiredUpscaler {
    Rr,
    Fsr4,
    Xess,
    Fsr3,
    Plain,
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

/// Field order is drop order (Rust drops in declaration order), and teardown
/// must mirror Streamline's documented shutdown sequence: release the proxied
/// queue/swapchain (`d3d`, which also waits the GPU idle) and both proxy
/// wrappers BEFORE `sl` runs slShutdown + FreeLibrary — the proxies' vtables
/// live in sl.interposer.dll, so releasing them after unload is UB.
pub struct GpuContext {
    d3d: D3d,
    passes: tonemap::Passes,
    blit: upload::BlitUpload,
    hdr: upload::HdrUpload,
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
    /// FSR (native pipeline only; never coexists with `sl` or `xess`). Same
    /// teardown discipline: ffxDestroyContext needs completed lists and a
    /// live device, so GpuContext::drop drains the queue and drops this
    /// explicitly.
    fsr: Option<FsrState>,
    /// GPU-resident NPPD (`--gpu --nppd`, XeSS sessions only): ONNX Runtime
    /// executing on `d3d.queue` with the tracer's NppdRes buffers bound as
    /// tensors. Dropped before `trace`'s resources it wraps is fine — the
    /// wrap AddRefs, and onnxruntime.dll is never unloaded.
    nppd_gpu: Option<crate::nppd::NppdGpu>,
    /// Whether the recurrent state carries history (false forces the next
    /// NPPD frame to zero the warped-state input — a reset).
    nppd_state_valid: bool,
    /// Kept alive for the app lifetime: the hooked CreateCommandQueue /
    /// CreateSwapChain entry points live on these (Frame Gen will need them
    /// again), and dropping them to refcount 0 destroys the SL wrappers.
    _proxy_device: Option<ID3D12Device>,
    _proxy_factory: Option<IDXGIFactory6>,
    pub adapter_name: String,
    pub adapter_is_nvidia: bool,
    /// Some(_) when Streamline is live; the queue and swapchain are then SL
    /// proxies and every present goes through Streamline (a manual-hooking
    /// requirement, and what makes Frame Generation possible later).
    /// Declared LAST so slShutdown runs after every proxy is released.
    sl: Option<streamline::SlContext>,
}

/// The per-frame Streamline sequence shared by the CPU-fed (`present_rr`) and
/// GPU-fed (`present_trace_rr`) RR paths: token -> constants -> options (the
/// world<->view matrices feed the SpecularHitDistance path, so they refresh
/// every frame) -> tags -> evaluate. Same token + viewport 0 throughout; the
/// jitter sign reported to SL is settled inside shim_constants, nowhere else.
fn rr_sl_sequence(
    sl: &streamline::SlContext,
    rr: &rr::RrResources,
    list: &ID3D12GraphicsCommandList,
    fc: &dlss::FrameConstants,
    frame_idx: u32,
) -> Result<()> {
    let _ev = pix::scope(list, c"rr-eval");
    let list_raw = list.as_raw();
    let token = sl.new_frame_token(frame_idx)?;
    sl.set_constants(token, 0, &shim_constants(fc))?;
    sl.dlssd_set_options(0, &shim_options(fc, rr.ow, rr.oh))?;
    sl.tag_resources(token, 0, &rr.tags(fc.rw, fc.rh), list_raw)?;
    sl.evaluate(streamline_sys::FEATURE_DLSS_RR, token, 0, list_raw)?;
    Ok(())
}

impl GpuContext {
    pub fn new(hwnd: HWND, w: u32, h: u32, opts: &GpuOptions) -> Result<Self> {
        // Chain level 1 (DLSS-RR). Streamline must initialize before any
        // DXGI factory exists — which is why DLSS is structurally the top of
        // the chain: it decides the SL-proxy-vs-native device split before
        // anything else can be probed.
        let mut sl = if opts.chain.dlss {
            match streamline::SlContext::init(
                &opts.sl_dir,
                &[streamline_sys::FEATURE_DLSS_RR],
                opts.debug,
            ) {
                Ok(s) => {
                    eprintln!("dlss: Streamline initialized ({})", opts.sl_dir);
                    Some(s)
                }
                Err(e) => {
                    eprintln!("dlss: level unavailable ({e}) — falling through the chain");
                    None
                }
            }
        } else {
            None
        };

        let factory =
            adapter::create_factory(opts.debug).map_err(|e| format!("CreateDXGIFactory2: {e}"))?;
        let prefer = opts.prefer.unwrap_or(adapter::Prefer::Nvidia);
        let pick = adapter::pick(&factory, prefer)?;
        eprintln!("gpu: using adapter \"{}\"", pick.name);
        if sl.is_some() && !pick.is_nvidia {
            eprintln!("dlss: level unavailable (no NVIDIA adapter) — falling through the chain");
            sl = None;
        }
        if let Some(s) = &sl {
            if let Err(e) = s.is_feature_supported(streamline_sys::FEATURE_DLSS_RR, pick.luid) {
                eprintln!("dlss: level unavailable (Ray Reconstruction unsupported: {e}) — falling through the chain");
                sl = None;
            }
        }

        let device = d3d12::create_device(&pick.adapter, opts.debug)?;

        // Manual-hooking proxy plumbing: queue from the proxy DEVICE,
        // swapchain from the proxy FACTORY — both required so present and
        // queue submissions are visible to SL (presentCommon fires every
        // frame from day one; DLSS-G later needs exactly these hooks).
        let mut proxy_device: Option<ID3D12Device> = None;
        let mut proxy_factory: Option<IDXGIFactory6> = None;
        let d3d = if let Some(s) = &sl {
            s.set_d3d_device(&device)?;
            let pdev: ID3D12Device = s.upgrade(&device)?;
            let queue = d3d12::create_queue(&pdev)?;
            let pfac: IDXGIFactory6 = s.upgrade(&factory)?;
            let d3d = D3d::with_queue(&pfac, device, queue, hwnd, w, h, opts.vsync)?;
            // The swapchain we hold MUST be the SL proxy — every present has
            // to route through presentCommon under manual hooking. A proxy
            // resolves to a distinct native interface; verify loudly.
            match s.native_of_raw(d3d.swapchain.as_raw()) {
                Ok(native) if native != d3d.swapchain.as_raw() => {
                    eprintln!("dlss: swapchain is SL-proxied (present hooked)");
                    // native_of_raw hands back a borrowed pointer; no release needed
                }
                Ok(_) => eprintln!("dlss: WARNING — swapchain proxy check inconclusive (same pointer)"),
                Err(e) => eprintln!("dlss: WARNING — swapchain is NOT an SL proxy ({e}); Frame Gen and RR may misbehave"),
            }
            proxy_device = Some(pdev);
            proxy_factory = Some(pfac);
            d3d
        } else {
            let queue = d3d12::create_queue(&device)?;
            D3d::with_queue(&factory, device, queue, hwnd, w, h, opts.vsync)?
        };

        let (rr_opt, rr_min, rr_max) = Self::query_rr_res(sl.as_ref(), w, h);

        let passes = tonemap::Passes::new(&d3d.device)?;
        let blit = upload::BlitUpload::new(&d3d, w, h)?;
        let hdr = upload::HdrUpload::new(&d3d, w, h)?;
        let rr_res = if sl.is_some() {
            let r = rr::RrResources::new(&d3d.device, rr_opt, rr_min, rr_max, w, h)?;
            passes.create_srv(
                &d3d.device,
                &r.output,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                tonemap::SRV_SLOT_RR,
            );
            Some(r)
        } else {
            None
        };
        passes.create_srv(
            &d3d.device,
            &blit.texture,
            d3d12::SWAPCHAIN_FORMAT,
            tonemap::SRV_SLOT_BLIT,
        );
        passes.create_srv(
            &d3d.device,
            &hdr.texture,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            tonemap::SRV_SLOT_HDR,
        );

        // Native chain levels (2-4), probed only when DLSS didn't win —
        // XeSS and the ffx contexts are created on the real device and
        // never coexist with the SL proxies. First hit wins; every failure
        // is one loud fall-through line. FSR4-RR and FSR3 are the SAME
        // ffx-api effect probed as two chain levels: level 2 requires the
        // Ray Regeneration provider (RDNA4), level 4 is the cross-vendor
        // FSR 3.1 upscale-only flavor (provider via ffxOverrideVersion).
        let mut xess_state: Option<XessState> = None;
        let mut fsr_state: Option<FsrState> = None;
        if sl.is_none() {
            let wire_fsr = |s: FsrState| {
                passes.create_srv(
                    &d3d.device,
                    s.res.upscaled(),
                    windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                    tonemap::SRV_SLOT_FSR,
                );
                s
            };
            if opts.chain.fsr4 {
                match Self::init_fsr(&opts.ffx_dir, &d3d.device, w, h, opts.debug, crate::fsr::Flavor::Fsr4Rr, &opts.fsr_tune) {
                    Ok(s) => fsr_state = Some(wire_fsr(s)),
                    Err(e) => eprintln!("fsr4: level unavailable ({e}) — falling through the chain"),
                }
            }
            if fsr_state.is_none() && opts.chain.xess {
                // Input planes are allocated once at the range MAX; every
                // frame uploads and names its own sub-rect (dynamic res).
                match Self::init_xess(&opts.xess_dir, &d3d.device, w, h, opts.xess_autoexposure) {
                    Ok(s) => {
                        passes.create_srv(
                            &d3d.device,
                            &s.res.output,
                            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                            tonemap::SRV_SLOT_XESS,
                        );
                        xess_state = Some(s);
                    }
                    Err(e) => eprintln!("xess: level unavailable ({e}) — falling through the chain"),
                }
            }
            if fsr_state.is_none() && xess_state.is_none() && opts.chain.fsr3 {
                match Self::init_fsr(&opts.ffx_dir, &d3d.device, w, h, opts.debug, crate::fsr::Flavor::Fsr3, &opts.fsr_tune) {
                    Ok(s) => fsr_state = Some(wire_fsr(s)),
                    Err(e) => eprintln!("fsr3: level unavailable ({e})"),
                }
            }
            if fsr_state.is_none()
                && xess_state.is_none()
                && opts.chain != crate::upchain::UpChain::NONE
            {
                eprintln!(
                    "upscale: NO temporal upscaler available — chain exhausted \
                     (dlss -> fsr4 -> xess -> fsr3); PLAIN presentation"
                );
            }
        }

        Ok(Self {
            sl,
            _proxy_device: proxy_device,
            _proxy_factory: proxy_factory,
            d3d,
            passes,
            blit,
            hdr,
            trace: None,
            dxr: None,
            rr: rr_res,
            xess: xess_state,
            fsr: fsr_state,
            nppd_gpu: None,
            nppd_state_valid: false,
            adapter_name: pick.name,
            adapter_is_nvidia: pick.is_nvidia,
        })
    }

    /// Query DLSS-RR's optimal Quality-mode render resolution and its
    /// dynamic range for an output size — the CPU renders inside [min, max]
    /// and RR upscales + denoises to the window size (step-wise DRS via
    /// sl::Extent tags). A failed or degenerate query falls back to DLAA
    /// (opt == min == max == output), which main.rs reads as "DRS off,
    /// fixed res". Re-callable — a window resize re-queries at the new
    /// output size.
    #[allow(clippy::type_complexity)]
    fn query_rr_res(
        sl: Option<&streamline::SlContext>,
        w: u32,
        h: u32,
    ) -> ((u32, u32), (u32, u32), (u32, u32)) {
        let Some(s) = sl else {
            return ((w, h), (w, h), (w, h));
        };
        let opt = streamline::SlShimDlssdOptions {
            mode: streamline_sys::DLSS_MODE_MAX_QUALITY,
            output_width: w,
            output_height: h,
            preset: streamline_sys::DLSSD_PRESET_E,
            normal_roughness_packed: 1,
            use_camera_matrices: 0,
            world_to_view: [0.0; 16],
            view_to_world: [0.0; 16],
        };
        match s.dlssd_optimal_settings(&opt) {
            Ok(o) if o.render_width > 0 && o.render_height > 0 => {
                eprintln!(
                    "dlss: RR Quality {}x{} -> render {}x{} (range {}x{}..{}x{})",
                    w, h, o.render_width, o.render_height,
                    o.render_width_min, o.render_height_min,
                    o.render_width_max, o.render_height_max,
                );
                // Malformed halves of the range collapse to the optimal
                // size — never invent a range the driver didn't report.
                let po = (o.render_width, o.render_height);
                let pmin = if o.render_width_min > 0
                    && o.render_height_min > 0
                    && o.render_width_min <= o.render_width
                    && o.render_height_min <= o.render_height
                {
                    (o.render_width_min, o.render_height_min)
                } else {
                    po
                };
                let pmax = if o.render_width_max >= o.render_width
                    && o.render_height_max >= o.render_height
                {
                    (o.render_width_max, o.render_height_max)
                } else {
                    po
                };
                (po, pmin, pmax)
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
        // Everything below drops live GPU resources; drain first (the
        // GpuContext::drop discipline — xess/ffx destroy-context require
        // completed command lists, ResizeBuffers requires zero outstanding
        // backbuffer refs).
        self.d3d.wait_idle()?;
        self.trace = None;
        self.dxr = None;
        self.nppd_gpu = None;
        self.nppd_state_valid = false;
        self.xess = None;
        self.fsr = None;
        self.rr = None;
        self.d3d.resize(w, h)?;
        self.blit = upload::BlitUpload::new(&self.d3d, w, h)?;
        self.hdr = upload::HdrUpload::new(&self.d3d, w, h)?;
        self.passes.create_srv(
            &self.d3d.device,
            &self.blit.texture,
            d3d12::SWAPCHAIN_FORMAT,
            tonemap::SRV_SLOT_BLIT,
        );
        self.passes.create_srv(
            &self.d3d.device,
            &self.hdr.texture,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            tonemap::SRV_SLOT_HDR,
        );
        if self.sl.is_some() {
            let (opt, min, max) = Self::query_rr_res(self.sl.as_ref(), w, h);
            let r = rr::RrResources::new(&self.d3d.device, opt, min, max, w, h)?;
            self.passes.create_srv(
                &self.d3d.device,
                &r.output,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                tonemap::SRV_SLOT_RR,
            );
            self.rr = Some(r);
        }
        match wired {
            WiredUpscaler::Xess => {
                match Self::init_xess(&opts.xess_dir, &self.d3d.device, w, h, opts.xess_autoexposure) {
                    Ok(s) => {
                        self.passes.create_srv(
                            &self.d3d.device,
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
                        self.passes.create_srv(
                            &self.d3d.device,
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

    /// Wire the session's live upscaler input planes as feed targets on a
    /// GPU tracer (`wire` = TraceGpu's or DxrGpu's `wire_feed`). The trace
    /// res was quantize_res-clamped by the caller, but the range is the
    /// SDK's contract: re-check here so a drift fails loudly at init. A gbuf
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
        if let Some(rr) = &self.rr {
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
                ],
            )
        } else if let Some(x) = &self.xess {
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
        } else if let Some(fs) = &self.fsr {
            if rw < fs.min.0 || rh < fs.min.1 || rw > fs.max.0 || rh > fs.max.1 {
                return Err(format!(
                    "trace res {}x{} outside FSR render range {}x{}..{}x{}",
                    rw, rh, fs.min.0, fs.min.1, fs.max.0, fs.max.1
                ));
            }
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
                    // cs_feed_fsr_rr's register/plane mapping (feed.hlsl —
                    // keep in lockstep; plane_resources returns upload order:
                    // depth_lin, depth_clip, mvec, normals, diff_alb,
                    // spec_alb, dd_in, ds_in, residual, ao_in, is_in).
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
        } else {
            Err("gbuf session with no live upscaler".into())
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
        debug: bool,
        bc7_q: Option<crate::bc7::Quality>,
    ) -> Result<()> {
        let dev = self.d3d.device.clone();
        let mut tg = trace::TraceGpu::new(
            &dev,
            dxc,
            scene,
            bvh,
            rw,
            rh,
            gbuf,
            nppd.is_some(),
            debug,
            bc7_q,
            &mut self.d3d,
        )?;
        // Upscaler sessions: wire the live upscaler's input planes as feed
        // targets — the feed kernel writes them directly, no CPU upload.
        if gbuf {
            self.wire_session_feed(rw, rh, |kind, targets| tg.wire_feed(&dev, kind, targets))?;
        }
        // GPU-resident NPPD: ORT session on OUR device/queue, tensors bound
        // over the NppdRes buffers. XeSS-only (RR is itself a denoiser — the
        // same exclusion as the CPU paths); a failure keeps the session
        // running plain and frees the ~340 MB staging.
        if let Some((dir, model)) = nppd {
            if self.xess.is_none() {
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
        self.passes.create_srv(
            &self.d3d.device,
            &tg.hdr,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            tonemap::SRV_SLOT_GPU,
        );
        self.trace = Some(tg);
        Ok(())
    }

    pub fn trace_ready(&self) -> bool {
        self.trace.is_some()
    }

    /// Build the DXR DispatchRays pipeline (the F key / --dxr). Idempotent —
    /// a live pipeline is kept. `(rw, rh)` is the session's fixed DXR trace
    /// resolution (the locked render res when `gbuf` composes with the wired
    /// upscaler, the window size otherwise); scene buffers + BLAS/TLAS build
    /// once here (the scene is static, --stress included).
    pub fn init_dxr(
        &mut self,
        dxc: &dxc::Dxc,
        scene: &crate::scene::Scene,
        // Unused since the DXR pipeline stopped uploading the software BVH
        // (SceneGpu sw_bvh: None); kept so the call sites stay uniform.
        _bvh: &crate::bvh::Bvh,
        rw: u32,
        rh: u32,
        gbuf: bool,
        debug: bool,
        bc7_q: Option<crate::bc7::Quality>,
    ) -> Result<()> {
        if self.dxr.is_some() {
            return Ok(());
        }
        let dev = self.d3d.device.clone();
        let mut d = dxr::DxrGpu::new(&dev, dxc, scene, rw, rh, gbuf, debug, bc7_q, &mut self.d3d)?;
        if gbuf {
            self.wire_session_feed(rw, rh, |kind, targets| d.wire_feed(&dev, kind, targets))?;
        }
        self.passes.create_srv(
            &self.d3d.device,
            &d.hdr,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            tonemap::SRV_SLOT_DXR,
        );
        self.dxr = Some(d);
        Ok(())
    }

    /// One DXR frame: constants -> DispatchRays -> resolve -> tonemap ->
    /// present. `samples` divides the accumulation (the present_trace shape).
    pub fn present_dxr(&mut self, p: &trace::FrameParams, samples: u32) -> Result<()> {
        crate::zone!("present-dxr");
        let Some(d) = &self.dxr else {
            return Err("DXR pipeline not initialized".into());
        };
        let slot = self.d3d.begin_frame()?;
        d.write_cb(slot, p);
        if let Err(e) = d.record_frame(&self.d3d.list, slot) {
            self.d3d.abort_frame();
            return Err(e);
        }
        d.record_resolve(&self.d3d.list, slot, samples);
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_DXR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// Re-present the last resolved DXR frame without tracing — the
    /// converged-idle path (present_hold's contract: record_resolve left hdr
    /// in PIXEL_SHADER_RESOURCE).
    pub fn present_dxr_hold(&mut self) -> Result<()> {
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
        crate::zone!("present-dxr-rr");
        if self.dxr.is_none() || self.sl.is_none() || self.rr.is_none() {
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
            d.write_cb(slot, p);
            if let Err(e) = d.record_frame(&d3d.list, slot) {
                d3d.abort_frame();
                return Err(e);
            }
            if let Err(e) = d.record_feed(&d3d.list, slot) {
                d3d.abort_frame();
                return Err(e);
            }
        }
        {
            let d3d = &mut self.d3d;
            let sl = self.sl.as_ref().unwrap();
            let rr = self.rr.as_ref().unwrap();
            unsafe {
                d3d.list.ResourceBarrier(&[transition(
                    &rr.output,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            if let Err(e) = rr_sl_sequence(sl, rr, &d3d.list, fc, frame_idx) {
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
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch —
        // the post-evaluate state restore eDisableCLStateTracking makes the
        // host's responsibility.
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
    ) -> Result<()> {
        crate::zone!("present-dxr-xess");
        if self.dxr.is_none() || self.xess.is_none() {
            return Err("DXR pipeline + XeSS not both initialized".into());
        }
        let slot = self.d3d.begin_frame()?;
        {
            let d3d = &mut self.d3d;
            let d = self.dxr.as_ref().unwrap();
            d.write_cb(slot, p);
            if let Err(e) = d.record_frame(&d3d.list, slot) {
                d3d.abort_frame();
                return Err(e);
            }
            if let Err(e) = d.record_feed(&d3d.list, slot) {
                d3d.abort_frame();
                return Err(e);
            }
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
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch,
        // restoring whatever list state the XeSS dispatch left behind.
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_XESS, 1.0);
        self.d3d.end_frame(slot)
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
        crate::zone!("present-dxr-fsr3");
        if self.dxr.is_none() || self.fsr.is_none() {
            return Err("DXR pipeline + FSR not both initialized".into());
        }
        debug_assert!(self.sl.is_none());
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
            d.write_cb(slot, p);
            if let Err(e) = d.record_frame(&d3d.list, slot) {
                d3d.abort_frame();
                return Err(e);
            }
            if let Err(e) = d.record_feed(&d3d.list, slot) {
                d3d.abort_frame();
                return Err(e);
            }
        }
        {
            let fs = self.fsr.as_ref().unwrap();
            if let Err(e) = Self::record_fsr3_upscale(fs, &self.d3d, fc, frame_ms) {
                self.d3d.abort_frame();
                return Err(e);
            }
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
        crate::zone!("present-trace");
        let Some(tg) = &self.trace else {
            return Err("GPU tracer not initialized".into());
        };
        let slot = self.d3d.begin_frame()?;
        tg.write_cb(slot, p);
        tg.record_frame(&self.d3d.list, slot, p, hybrid);
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
        if self.trace.is_none() {
            return Err("GPU tracer not initialized".into());
        }
        let slot = self.d3d.begin_frame()?;
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_GPU, 1.0);
        self.d3d.end_frame(slot)
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
    ) -> Result<()> {
        crate::zone!("present-trace-xess");
        if self.trace.is_none() || self.xess.is_none() {
            return Err("GPU tracer + XeSS not both initialized".into());
        }
        let nppd_on = nppd && self.nppd_gpu.is_some();
        let slot = self.d3d.begin_frame()?;
        {
            // Field-split borrows: the recorder reads the tracer, abort needs
            // d3d mutably.
            let d3d = &mut self.d3d;
            let tg = self.trace.as_ref().unwrap();
            tg.write_cb(slot, p);
            tg.record_frame(&d3d.list, slot, p, hybrid);
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
            if let Err(e) = tg.record_feed(&d3d.list, slot, nppd_on) {
                d3d.abort_frame();
                return Err(e);
            }
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
            unsafe {
                d3d.list.ResourceBarrier(&[transition(
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
                input_width: tg.rw,
                input_height: tg.rh,
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
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch,
        // restoring whatever list state the XeSS dispatch left behind.
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_XESS, 1.0);
        self.d3d.end_frame(slot)
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
        crate::zone!("present-trace-fsr3");
        if self.trace.is_none() || self.fsr.is_none() {
            return Err("GPU tracer + FSR not both initialized".into());
        }
        debug_assert!(self.sl.is_none());
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
            tg.write_cb(slot, p);
            tg.record_frame(&d3d.list, slot, p, hybrid);
            if let Err(e) = tg.record_feed(&d3d.list, slot, false) {
                d3d.abort_frame();
                return Err(e);
            }
        }
        {
            let fs = self.fsr.as_ref().unwrap();
            if let Err(e) = Self::record_fsr3_upscale(fs, &self.d3d, fc, frame_ms) {
                self.d3d.abort_frame();
                return Err(e);
            }
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
    pub fn verify_trace(&mut self, cam: &crate::camera::CamBasis, q: crate::shade::Quality) -> Result<String> {
        let (tbuf, info, counters, px) = {
            let Some(tg) = &self.trace else {
                return Err("GPU tracer not initialized".into());
            };
            (tg.tbuf.clone(), tg.info.clone(), tg.counters.clone(), (tg.rw * tg.rh) as usize)
        };
        let p = trace::FrameParams {
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
        let ctrs = self.read_trace_buffer(&counters, trace::CTR_COUNT as usize * 4)?;
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
        Ok(format!(
            "gpu verify ({px} px): false-sky {false_sky} | tmin-overshoot {overshoot} | hybrid-extra {extra} | unwritten {sentinel} | max rel t err {max_rel:.2e} | tiles: {} splits, {} sky, {} leaves, {} blocked -> {}",
            u(&ctrs, trace::CTR_SPLIT as usize),
            u(&ctrs, trace::CTR_SKY as usize),
            u(&ctrs, trace::CTR_LEAF as usize),
            u(&ctrs, trace::CTR_BLOCKED as usize),
            if ok { "OK" } else { "FAILED" },
        ))
    }

    /// Streamline live => RR evaluate is available (M4).
    pub fn dlss_ready(&self) -> bool {
        self.sl.is_some() && self.rr.is_some()
    }

    /// Which chain level this session wired — derived from the live state
    /// (at most one upscaler exists per session by the probe's construction),
    /// so it can never disagree with the contexts actually held.
    pub fn wired(&self) -> WiredUpscaler {
        if self.dlss_ready() {
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
    pub fn present_fsr(
        &mut self,
        g: &dlss::GBufs,
        f: &crate::fsr::FsrBufs,
        fc: &dlss::FrameConstants,
        prev_pos: Option<glam::Vec3A>,
        frame_idx: u32,
        frame_ms: f32,
    ) -> Result<()> {
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

        if let Err(e) = Self::record_fsr_rr_sequence(fs, &self.d3d, fc, prev_pos, frame_idx, frame_ms) {
            self.d3d.abort_frame();
            return Err(e);
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
    fn record_fsr_rr_sequence(
        fs: &FsrState,
        d3d: &D3d,
        fc: &dlss::FrameConstants,
        prev_pos: Option<glam::Vec3A>,
        frame_idx: u32,
        frame_ms: f32,
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
        res.barrier_denoise_begin(&d3d.list);
        let r = res.denoise_res();
        let dd_desc = ffx::FfxShimDenoiseDesc {
            cmdlist: d3d.list.as_raw(),
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
            let _ev = pix::scope(&d3d.list, c"fsr-denoise");
            fs.ctx.denoise(&dd_desc)?;
        }
        res.barrier_denoise_end(&d3d.list);

        // Remodulate (binds from scratch — the post-ffx state restore).
        res.record_composite(&d3d.list, rw, rh);

        // FSR4 upscale: composite -> window-res output. The shared MV plane
        // holds UV-deltas here, so the scale multiplies the render dims back
        // in to hand FSR pixel-space MVs (polarity knob: fsr::UPSCALE_MV_SIGN).
        res.barrier_upscale_begin(&d3d.list);
        let up_desc = Self::fsr_upscale_desc(
            d3d.list.as_raw(),
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
            let _ev = pix::scope(&d3d.list, c"fsr-upscale");
            fs.ctx.upscale(&up_desc)?;
        }
        res.barrier_upscale_end(&d3d.list);
        Ok(())
    }

    /// Ray Regeneration + FSR4 fed by the GPU-resident tracer: trace -> feed
    /// (pack + sig -> the nine FSR planes, on-GPU) -> denoise -> composite ->
    /// upscale -> tonemap(SRV_SLOT_FSR) -> present. Never an SL session.
    pub fn present_trace_fsr_rr(
        &mut self,
        p: &trace::FrameParams,
        hybrid: bool,
        fc: &dlss::FrameConstants,
        prev_pos: Option<glam::Vec3A>,
        frame_idx: u32,
        frame_ms: f32,
    ) -> Result<()> {
        crate::zone!("present-trace-fsr-rr");
        if self.trace.is_none() || self.fsr.is_none() {
            return Err("GPU tracer + FSR not both initialized".into());
        }
        debug_assert!(self.sl.is_none());
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
            tg.write_cb(slot, p);
            tg.record_frame(&d3d.list, slot, p, hybrid);
            if let Err(e) = tg.record_feed(&d3d.list, slot, false) {
                d3d.abort_frame();
                return Err(e);
            }
        }
        {
            let fs = self.fsr.as_ref().unwrap();
            if let Err(e) =
                Self::record_fsr_rr_sequence(fs, &self.d3d, fc, prev_pos, frame_idx, frame_ms)
            {
                self.d3d.abort_frame();
                return Err(e);
            }
        }
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_FSR, 1.0);
        self.d3d.end_frame(slot)
    }

    /// Ray Regeneration + FSR4 fed by the DXR pipeline — the
    /// `present_dxr_fsr3` shape with the full denoise sequence in the middle.
    pub fn present_dxr_fsr_rr(
        &mut self,
        p: &trace::FrameParams,
        fc: &dlss::FrameConstants,
        prev_pos: Option<glam::Vec3A>,
        frame_idx: u32,
        frame_ms: f32,
    ) -> Result<()> {
        crate::zone!("present-dxr-fsr-rr");
        if self.dxr.is_none() || self.fsr.is_none() {
            return Err("DXR pipeline + FSR not both initialized".into());
        }
        debug_assert!(self.sl.is_none());
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
            d.write_cb(slot, p);
            if let Err(e) = d.record_frame(&d3d.list, slot) {
                d3d.abort_frame();
                return Err(e);
            }
            if let Err(e) = d.record_feed(&d3d.list, slot) {
                d3d.abort_frame();
                return Err(e);
            }
        }
        {
            let fs = self.fsr.as_ref().unwrap();
            if let Err(e) =
                Self::record_fsr_rr_sequence(fs, &self.d3d, fc, prev_pos, frame_idx, frame_ms)
            {
                self.d3d.abort_frame();
                return Err(e);
            }
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

        if let Err(e) = Self::record_fsr3_upscale(fs, &self.d3d, fc, frame_ms) {
            self.d3d.abort_frame();
            return Err(e);
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
    fn record_fsr3_upscale(
        fs: &FsrState,
        d3d: &D3d,
        fc: &dlss::FrameConstants,
        frame_ms: f32,
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
        res.barrier_upscale_begin(&d3d.list);
        let up_desc = Self::fsr_upscale_desc(
            d3d.list.as_raw(),
            res.upscale_res(),
            fc,
            fs.max,
            frame_ms,
            [crate::fsr::UPSCALE_MV_SIGN.0, crate::fsr::UPSCALE_MV_SIGN.1],
        );
        {
            let _ev = pix::scope(&d3d.list, c"fsr-upscale");
            fs.ctx.upscale(&up_desc)?;
        }
        res.barrier_upscale_end(&d3d.list);
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
    ) -> Result<()> {
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
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch,
        // restoring whatever list state the XeSS dispatch left behind.
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_XESS, 1.0);
        self.d3d.end_frame(slot)
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
    ) -> Result<()> {
        crate::zone!("present-rr");
        let (Some(sl), Some(rr)) = (&self.sl, &self.rr) else {
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

        if let Err(e) = rr_sl_sequence(sl, rr, &self.d3d.list, fc, frame_idx) {
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
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch,
        // which is exactly the post-evaluate state restore that
        // eDisableCLStateTracking makes the host's responsibility.
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
        crate::zone!("present-trace-rr");
        if self.trace.is_none() || self.sl.is_none() || self.rr.is_none() {
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
            tg.write_cb(slot, p);
            tg.record_frame(&d3d.list, slot, p, hybrid);
            if let Err(e) = tg.record_feed(&d3d.list, slot, false) {
                d3d.abort_frame();
                return Err(e);
            }
        }
        {
            let d3d = &mut self.d3d;
            let sl = self.sl.as_ref().unwrap();
            let rr = self.rr.as_ref().unwrap();
            unsafe {
                d3d.list.ResourceBarrier(&[transition(
                    &rr.output,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            if let Err(e) = rr_sl_sequence(sl, rr, &d3d.list, fc, frame_idx) {
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
        // The tonemap pass re-binds root sig/PSO/heaps/viewport from scratch —
        // the post-evaluate state restore eDisableCLStateTracking makes the
        // host's responsibility.
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
                let tm = |v: half::f16| -> u32 {
                    let c = f32::from(v).max(0.0);
                    let m = (1.0 - (-c).exp()).clamp(0.0, 1.0).powf(1.0 / 2.2);
                    (m * 255.0 + 0.5) as u32
                };
                out[y * w + x] = (tm(px[0]) << 16) | (tm(px[1]) << 8) | tm(px[2]);
            }
        }
        unsafe { rb.resource.Unmap(0, None) };
        Ok((out, w, h))
    }

    /// M1: present the CPU-tonemapped u32 0RGB frame.
    pub fn present_cpu(&mut self, pixels: &[u32]) -> Result<()> {
        crate::zone!("present-cpu");
        let slot = self.d3d.begin_frame()?;
        self.blit.record(&self.d3d, slot, pixels);
        self.fullscreen_to_backbuffer(false, tonemap::SRV_SLOT_BLIT, 1.0);
        self.d3d.end_frame(slot)
    }

    /// M2: present the raw linear-HDR accumulation with the GPU tonemap.
    pub fn present_hdr(&mut self, accum: &[AtomicU32], samples: u32) -> Result<()> {
        crate::zone!("present-hdr");
        let slot = self.d3d.begin_frame()?;
        self.hdr.record(&self.d3d, slot, accum);
        self.fullscreen_to_backbuffer(true, tonemap::SRV_SLOT_HDR, 1.0 / samples.max(1) as f32);
        self.d3d.end_frame(slot)
    }

    fn fullscreen_to_backbuffer(&self, use_tonemap: bool, srv_slot: u32, inv_samples: f32) {
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
            self.d3d.rtv_handle(bb),
            self.d3d.width,
            self.d3d.height,
        );
        unsafe {
            self.d3d.list.ResourceBarrier(&[transition(
                backbuffer,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            )]);
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        // xessDestroyContext / ffxDestroyContext require all pending command
        // lists complete and a live device: drain the queue, then drop the
        // contexts here — before the field-order teardown releases the
        // swapchain/device.
        if self.xess.is_some() || self.fsr.is_some() {
            let _ = self.d3d.wait_idle();
            self.xess = None;
            self.fsr = None;
        }
    }
}

/// glam Mat4 (column-major) -> Streamline row-major float[16]. This is THE
/// transpose boundary — nothing else in the codebase reorders matrices.
fn row_major(m: &glam::Mat4) -> [f32; 16] {
    m.transpose().to_cols_array()
}

fn v3(v: glam::Vec3A) -> [f32; 3] {
    [v.x, v.y, v.z]
}

fn shim_constants(fc: &dlss::FrameConstants) -> streamline::SlShimConstants {
    streamline::SlShimConstants {
        view_to_clip: row_major(&fc.view_to_clip),
        clip_to_view: row_major(&fc.clip_to_view),
        clip_to_prev_clip: row_major(&fc.clip_to_prev_clip),
        prev_clip_to_clip: row_major(&fc.prev_clip_to_clip),
        // fc.jitter is the sample-position offset the renderer used, in
        // pixels; SL's (undocumented) polarity is the NEGATED offset —
        // settled empirically 2026-07: this sign gives a rock-stable static
        // image, the others wobble by 2x the jitter. The flip lives HERE,
        // nowhere else.
        jitter: [-fc.jitter.0, -fc.jitter.1],
        // MV values are pixels; this scale normalizes them for SL.
        mvec_scale: [1.0 / fc.rw as f32, 1.0 / fc.rh as f32],
        cam_pos: v3(fc.pos),
        cam_up: v3(fc.up),
        cam_right: v3(fc.right),
        cam_fwd: v3(fc.forward),
        cam_near: fc.near,
        cam_far: fc.far,
        cam_fov: fc.fov_y,
        cam_aspect: fc.aspect,
        depth_inverted: 0,
        camera_motion_included: 1,
        mvec_3d: 0,
        reset: fc.reset as i32,
        ortho: 0,
        mvec_jittered: 0,
    }
}

fn shim_options(fc: &dlss::FrameConstants, ow: u32, oh: u32) -> streamline::SlShimDlssdOptions {
    streamline::SlShimDlssdOptions {
        // Render res == output res only when the optimal-settings query fell
        // back to native; otherwise the frame was traced at the Quality-mode
        // render size and RR upscales.
        mode: if (fc.rw as u32, fc.rh as u32) == (ow, oh) {
            streamline_sys::DLSS_MODE_DLAA
        } else {
            streamline_sys::DLSS_MODE_MAX_QUALITY
        },
        output_width: ow,
        output_height: oh,
        // Preset E = the latest transformer model (sl_dlss_d.h:38); fall
        // back to DLSSD_PRESET_D / _DEFAULT if E misbehaves.
        preset: streamline_sys::DLSSD_PRESET_E,
        normal_roughness_packed: 1,
        use_camera_matrices: 1,
        world_to_view: row_major(&fc.world_to_view),
        view_to_world: row_major(&fc.view_to_world),
    }
}
