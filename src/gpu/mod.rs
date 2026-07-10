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
pub mod rr;
pub mod streamline;
pub mod streamline_sys;
pub mod tonemap;
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
    /// Attempt DLSS via Streamline (falls back to a native pipeline with a
    /// log line when the SDK/adapter/driver doesn't support it).
    pub dlss: bool,
    /// Directory holding sl.interposer.dll and the SL plugins.
    pub sl_dir: String,
    /// Attempt XeSS-SR (requires the native, non-Streamline pipeline —
    /// main.rs forces dlss off when this is set).
    pub xess: bool,
    /// Directory holding libxess.dll.
    pub xess_dir: String,
    /// D3D12 debug layer + DXGI debug factory + verbose SL logging.
    pub debug: bool,
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
    rr: Option<rr::RrResources>,
    /// XeSS-SR (native pipeline only; never coexists with `sl`). Explicitly
    /// torn down by GpuContext::drop after a queue drain — xessDestroyContext
    /// requires completed command lists and a live device.
    xess: Option<XessState>,
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

impl GpuContext {
    pub fn new(hwnd: HWND, w: u32, h: u32, opts: &GpuOptions) -> Result<Self> {
        // Streamline must initialize before any DXGI factory exists.
        let mut sl = if opts.dlss {
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
                    eprintln!("dlss: disabled — {e}");
                    None
                }
            }
        } else {
            None
        };

        let factory =
            adapter::create_factory(opts.debug).map_err(|e| format!("CreateDXGIFactory2: {e}"))?;
        let pick = adapter::pick(&factory, false)?;
        eprintln!("gpu: using adapter \"{}\"", pick.name);
        if sl.is_some() && !pick.is_nvidia {
            eprintln!("dlss: disabled — no NVIDIA adapter");
            sl = None;
        }
        if let Some(s) = &sl {
            if let Err(e) = s.is_feature_supported(streamline_sys::FEATURE_DLSS_RR, pick.luid) {
                eprintln!("dlss: Ray Reconstruction not supported on this adapter — {e}");
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
            let d3d = D3d::with_queue(&pfac, device, queue, hwnd, w, h)?;
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
            D3d::with_queue(&factory, device, queue, hwnd, w, h)?
        };

        // Query DLSS-RR's optimal Quality-mode render resolution and its
        // dynamic range — the CPU renders inside [min, max] and RR upscales
        // + denoises to the window size (step-wise DRS via sl::Extent tags).
        // A failed or degenerate query falls back to DLAA (opt == min ==
        // max == output), which main.rs reads as "DRS off, fixed res".
        let (rr_opt, rr_min, rr_max) = if let Some(s) = &sl {
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
        } else {
            ((w, h), (w, h), (w, h))
        };

        let passes = tonemap::Passes::new(&d3d.device)?;
        let blit = upload::BlitUpload::new(&d3d, w, h)?;
        let hdr = upload::HdrUpload::new(&d3d, w, h)?;
        let rr_res = if sl.is_some() {
            let r = rr::RrResources::new(&d3d, rr_opt, rr_min, rr_max, w, h)?;
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

        // XeSS-SR lives on the native pipeline only — its context is created
        // on the real device, and it never coexists with the SL proxies
        // (main.rs forces dlss off for --xess sessions). Input planes are
        // allocated once at the range MAX; every frame uploads and names its
        // own sub-rect (dynamic resolution).
        let xess_state = if opts.xess && sl.is_none() {
            match crate::xess::Xess::new(&opts.xess_dir, d3d.device.as_raw(), (w, h)) {
                Ok((ctx, opt, min, max)) => {
                    eprintln!(
                        "xess: {}x{} -> optimal {}x{} (range {}x{}..{}x{})",
                        w, h, opt.x, opt.y, min.x, min.y, max.x, max.y,
                    );
                    match xr::XessResources::new(&d3d, max.x, max.y, w, h) {
                        Ok(r) => {
                            passes.create_srv(
                                &d3d.device,
                                &r.output,
                                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                                tonemap::SRV_SLOT_XESS,
                            );
                            Some(XessState {
                                ctx,
                                res: r,
                                opt: (opt.x, opt.y),
                                min: (min.x, min.y),
                                max: (max.x, max.y),
                            })
                        }
                        Err(e) => {
                            eprintln!("xess: disabled — resource allocation failed: {e}");
                            None
                        }
                    }
                }
                Err(e) => {
                    eprintln!("xess: disabled — {e}");
                    None
                }
            }
        } else {
            if opts.xess {
                eprintln!("xess: disabled — cannot coexist with Streamline (use --xess, which implies --no-dlss)");
            }
            None
        };

        Ok(Self {
            sl,
            _proxy_device: proxy_device,
            _proxy_factory: proxy_factory,
            d3d,
            passes,
            blit,
            hdr,
            rr: rr_res,
            xess: xess_state,
            adapter_name: pick.name,
            adapter_is_nvidia: pick.is_nvidia,
        })
    }

    /// Streamline live => RR evaluate is available (M4).
    pub fn dlss_ready(&self) -> bool {
        self.sl.is_some() && self.rr.is_some()
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
        x.res.record_upload(&self.d3d, slot, color, g, rw, rh, near, far);
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
        if let Err(e) = x.ctx.execute(self.d3d.list.as_raw(), &params) {
            // Nothing executed on the GPU yet — abandon the recorded frame so
            // the caller can fall back to the CPU present path.
            self.d3d.abort_frame();
            return Err(e);
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
        let slot = self.record_xess_dispatch(color, g, rw, rh, jitter, reset, near, far)?;
        let x = self.xess.as_ref().unwrap();
        x.res.record_readback(&self.d3d);
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
        rr.record_upload(&self.d3d, slot, accum, g, fc.rw, fc.rh);
        unsafe {
            self.d3d.list.ResourceBarrier(&[transition(
                &rr.output,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }

        // Per-frame Streamline sequence: token -> constants -> options (the
        // world<->view matrices feed the SpecularHitDistance path, so they
        // refresh every frame) -> tags -> evaluate. Same token + viewport 0
        // throughout.
        let list_raw = self.d3d.list.as_raw();
        let sl_seq = || -> Result<()> {
            let token = sl.new_frame_token(frame_idx)?;
            sl.set_constants(token, 0, &shim_constants(fc))?;
            sl.dlssd_set_options(0, &shim_options(fc, rr.ow, rr.oh))?;
            sl.tag_resources(token, 0, &rr.tags(fc.rw, fc.rh), list_raw)?;
            sl.evaluate(streamline_sys::FEATURE_DLSS_RR, token, 0, list_raw)?;
            Ok(())
        };
        if let Err(e) = sl_seq() {
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

    /// Read back the denoised RR output and tonemap it on the CPU with the
    /// same curve as `render::resolve` at 1 spp. Screenshots in DLSS mode
    /// need this — the denoised image exists only in GPU memory. Synchronous
    /// and allocation-heavy by design; it runs on a keypress, not per frame.
    pub fn read_rr_output(&mut self) -> Result<Vec<u32>> {
        let Some(rr) = &self.rr else {
            return Err("DLSS-RR not initialized".into());
        };
        let output = rr.output.clone();
        self.read_hdr_output(output)
    }

    /// XeSS twin of `read_rr_output`: the upscaled image exists only on the
    /// GPU; screenshots in XeSS mode read it back through the same path.
    pub fn read_xess_output(&mut self) -> Result<Vec<u32>> {
        let Some(x) = &self.xess else {
            return Err("XeSS not initialized".into());
        };
        let output = x.res.output.clone();
        self.read_hdr_output(output)
    }

    fn read_hdr_output(&mut self, output: ID3D12Resource) -> Result<Vec<u32>> {
        let (w, h) = (self.d3d.width as usize, self.d3d.height as usize);
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
        Ok(out)
    }

    /// M1: present the CPU-tonemapped u32 0RGB frame.
    pub fn present_cpu(&mut self, pixels: &[u32]) -> Result<()> {
        let slot = self.d3d.begin_frame()?;
        self.blit.record(&self.d3d, slot, pixels);
        self.fullscreen_to_backbuffer(false, tonemap::SRV_SLOT_BLIT, 1.0);
        self.d3d.end_frame(slot)
    }

    /// M2: present the raw linear-HDR accumulation with the GPU tonemap.
    pub fn present_hdr(&mut self, accum: &[AtomicU32], samples: u32) -> Result<()> {
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
        // xessDestroyContext requires all pending command lists complete and
        // a live device: drain the queue, then drop the XeSS context here —
        // before the field-order teardown releases the swapchain/device.
        if self.xess.is_some() {
            let _ = self.d3d.wait_idle();
            self.xess = None;
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
