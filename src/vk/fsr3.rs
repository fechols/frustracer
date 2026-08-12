//! FSR3 upscaling over the stock FidelityFX `ffx_vk` backend.
//!
//! The Vulkan peer of the D3D12 `--fsr3` arm (`gpu/ffx_up.rs`), and the first
//! vendor upscaler this backend has. It is **CPU-fed**: colour comes from
//! `accum` and the two guides from `dlss::GBufs`, which the CPU renderer
//! produces on every platform — so this needs no G-buffer pack from the Vulkan
//! tracer and no presentation stage, exactly as `gpu/ffx_up.rs::record_upload`
//! serves the CPU renderer on Windows.
//!
//! FFX ITSELF OWNS ITS PIPELINES, DESCRIPTORS AND INTERNAL RESOURCES. Nothing
//! here goes through `vk::layout`'s derived register map: the SDK creates its
//! own descriptor pools from the SPIR-V permutations committed under
//! `SDKs/FidelityFX-SDK-prebuilt/shaders/vk`, so this module is images, a
//! staging path, barriers, and one call into `shim/ffx_fsr3_vk.cpp`.
//!
//! See that shim for the three facts the public FFX headers do not state (the
//! link-time frame-generation stub, the zero-initialised scratch, and the three
//! mandatory shared resources), and `vk::device` for why `shaderFloat16` and
//! `VK_KHR_get_memory_requirements2` are enabled on a device whose own corpus
//! uses neither.

use ash::vk::{self, Handle};
use std::ffi::c_void;

use super::device::{Buffer, Vk};
use super::headless::VkHeadless;

// The C ABI from shim/ffx_fsr3_vk.h. Declared only where the SDK was actually
// compiled in — `built()` is the Rust-visible half of the same `cfg`.
#[cfg(ffx_fsr3_vk)]
unsafe extern "C" {
    fn frshim_fsr3vk_create(
        physical_device: *mut c_void,
        device: *mut c_void,
        get_device_proc_addr: *mut c_void,
        max_render_w: u32,
        max_render_h: u32,
        upscale_w: u32,
        upscale_h: u32,
        flags: u32,
        out_handle: *mut *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn frshim_fsr3vk_dispatch(
        handle: *mut c_void,
        cmd_buf: *mut c_void,
        color: u64,
        depth: u64,
        motion: u64,
        output: u64,
        shared_dilated_depth: u64,
        shared_dilated_motion: u64,
        shared_recon_prev_depth: u64,
        render_w: u32,
        render_h: u32,
        upscale_w: u32,
        upscale_h: u32,
        jitter_x: f32,
        jitter_y: f32,
        mv_scale_x: f32,
        mv_scale_y: f32,
        frame_time_delta_ms: f32,
        camera_near: f32,
        camera_far: f32,
        camera_fov_y: f32,
        reset: i32,
    ) -> i32;
    fn frshim_fsr3vk_destroy(handle: *mut c_void);
}

/// Mirrors `shim/ffx_fsr3_vk.h`'s flag enum. Ours, not FFX's — the SDK
/// constants stay on the C++ side.
pub const HDR: u32 = 1 << 0;
pub const DEPTH_INVERTED: u32 = 1 << 1;
#[allow(dead_code)]
pub const DEPTH_INFINITE: u32 = 1 << 2;
#[allow(dead_code)]
pub const AUTO_EXPOSURE: u32 = 1 << 3;

/// Was the SDK's Vulkan backend compiled into this binary? A false here is an
/// environment fact (`./install-prerequisites.sh fsr3src`, plus the distro's
/// vulkan-headers), never a failure — the gate SKIPs on it, the way every
/// absent-SDK path in this tree does.
///
/// KEYED ON `ffx_fsr3_vk`, NOT `ffx_fsr3_src`, and the difference is a link
/// error rather than a nicety: build.rs compiles `ffx_vk.cpp` + our shim only
/// when `want_vk` (Linux AND <vulkan/vulkan.h> present), while `ffx_fsr3_src`
/// says merely that the backend-NEUTRAL units built. On macOS the two diverge
/// by construction — that platform takes the Metal `FfxInterface` instead — so
/// declaring the `frshim_fsr3vk_*` externs under the broader cfg left them
/// undefined at link time on every Mac carrying the SDK source. One cfg per
/// artifact, the same rule `ffx_fsr3_metal` follows for the transpiled
/// metallibs.
pub const fn built() -> bool {
    cfg!(ffx_fsr3_vk)
}

/// Everything one FFX dispatch needs that is not a resource.
///
/// A struct because there are two callers now (`frame` CPU-fed and
/// `frame_fed` GPU-fed) and the whole point of factoring them is that they
/// cannot disagree about this block: a difference here would present as a
/// quality difference between the two feed routes, which is exactly the
/// comparison V13 makes, so it would corrupt its own instrument.
#[derive(Clone, Copy)]
// Only the `ffx_fsr3_vk` build calls into this; on macOS the whole arm
// compiles out (the Metal `FfxInterface` stands in), so the allow is
// scoped to exactly that platform rather than blanketed — a dead-code
// warning is signal in this tree and must keep firing on Linux.
#[cfg_attr(not(ffx_fsr3_vk), allow(dead_code))]
pub struct Dispatch {
    /// The renderer's own sample offset, through `fsr::JITTER_SIGN` (or the
    /// `FR_VK_FSR3_JITTER` lever) — the one convention the two FidelityFX
    /// generations disagree about.
    pub jitter: (f32, f32),
    /// `fsr::UPSCALE_MV_SIGN`, not a resolution: the mvec plane is already
    /// pixel-space, y-down, current -> previous.
    pub mv_scale: (f32, f32),
    pub near_far: (f32, f32),
    pub fov_y: f32,
    /// Milliseconds. A FIXED clock in every gate — the `nrd_gpu::NOMINAL_DT_MS`
    /// precedent: a deterministic run must not put a wall timer into a vendor
    /// library's internal curves.
    pub dt_ms: f32,
    pub reset: bool,
}

/// One image plus what it takes to upload to and read from it. Local rather
/// than shared with `tracer.rs`'s: that one is fixed at RGBA16F/STORAGE for the
/// resolve target, and generalising it is a refactor this stage does not need
/// to drag along.
struct Img {
    img: vk::Image,
    /// A `STORAGE_IMAGE` view, for a compute kernel of OURS to write this
    /// plane instead of a host upload (`Fsr3::feed_views`, B3).
    ///
    /// Additive and structurally unable to perturb the CPU-fed path: FFX
    /// IGNORES caller views entirely and rebuilds its own from the format in
    /// the `FfxResourceDescription` it is handed (see the header note beside
    /// `rdesc_tex2d`), so nothing downstream reads this.
    view: vk::ImageView,
    mem: vk::DeviceMemory,
    w: u32,
    h: u32,
    /// Bytes per texel of `fmt`, for the staging buffer and the readback. Kept
    /// beside the format so the two can never disagree.
    bpp: u64,
}

// Only the `ffx_fsr3_vk` build calls into this; on macOS the whole arm
// compiles out (the Metal `FfxInterface` stands in), so the allow is
// scoped to exactly that platform rather than blanketed — a dead-code
// warning is signal in this tree and must keep firing on Linux.
#[cfg_attr(not(ffx_fsr3_vk), allow(dead_code))]
impl Img {
    fn new(vkd: &Vk, w: u32, h: u32, fmt: vk::Format, bpp: u64) -> Result<Img, String> {
        let d = &vkd.device;
        // Every image here is both an FFX resource and a copy endpoint: the
        // three inputs are TRANSFER_DST, the output is TRANSFER_SRC, and the
        // three shared ones are neither in practice — but FFX asks for the full
        // set in its own allocations, and a uniform usage keeps the create call
        // one line instead of a table.
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(fmt)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let img =
            unsafe { d.create_image(&ci, None) }.map_err(|e| format!("vkCreateImage(fsr3): {e}"))?;
        let req = unsafe { d.get_image_memory_requirements(img) };
        let idx = super::device::mem_type_index(
            &vkd.mem,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| "no device-local memory type for an FSR3 image".to_string())?;
        let mem = unsafe {
            d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(idx),
                None,
            )
        }
        .map_err(|e| format!("vkAllocateMemory(fsr3 image): {e}"))?;
        unsafe { d.bind_image_memory(img, mem, 0) }
            .map_err(|e| format!("vkBindImageMemory(fsr3): {e}"))?;
        let view = unsafe {
            d.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(img)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(fmt)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )
        }
        .map_err(|e| format!("vkCreateImageView(fsr3): {e}"))?;
        Ok(Img { img, view, mem, w, h, bpp })
    }

    fn bytes(&self) -> u64 {
        self.w as u64 * self.h as u64 * self.bpp
    }

    fn destroy(&self, vkd: &Vk) {
        unsafe {
            vkd.device.destroy_image_view(self.view, None);
            vkd.device.destroy_image(self.img, None);
            vkd.device.free_memory(self.mem, None);
        }
    }

    /// FFX takes the raw handle; `ash` keeps it as a u64 already.
    fn raw(&self) -> u64 {
        self.img.as_raw()
    }
}

/// A layout transition on one image. Deliberately the coarse
/// `ALL_COMMANDS`/`MEMORY_READ|WRITE` form: this path runs a handful of
/// barriers per frame around a dispatch that is itself milliseconds, so
/// narrowing them would buy nothing and cost a table of stage masks to keep
/// in step with FFX's internal ones.
fn transition(
    d: &ash::Device,
    cmd: vk::CommandBuffer,
    img: &Img,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    let b = vk::ImageMemoryBarrier::default()
        .old_layout(from)
        .new_layout(to)
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(img.img)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );
    unsafe {
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[b],
        );
    }
}

/// The FSR3 upscaler over a Vulkan device.
// Only the `ffx_fsr3_vk` build calls into this; on macOS the whole arm
// compiles out (the Metal `FfxInterface` stands in), so the allow is
// scoped to exactly that platform rather than blanketed — a dead-code
// warning is signal in this tree and must keep firing on Linux.
#[cfg_attr(not(ffx_fsr3_vk), allow(dead_code))]
pub struct Fsr3 {
    handle: *mut c_void,
    render: (u32, u32),
    upscale: (u32, u32),
    color: Img,
    depth: Img,
    motion: Img,
    output: Img,
    // FFX's cross-frame temporal state. Mandatory despite reading optional in
    // the header — see shim/ffx_fsr3_vk.h.
    dilated_depth: Img,
    dilated_motion: Img,
    recon_prev_depth: Img,
    /// Host-visible staging, one per input plane plus one for the readback.
    /// Persistent because this is a per-frame path and a fresh allocation per
    /// frame would measure the allocator rather than the upscaler.
    stage_color: Buffer,
    stage_depth: Buffer,
    stage_motion: Buffer,
    readback: Buffer,
}

impl Fsr3 {
    /// `render` is the traced resolution, `upscale` the output. Both are fixed
    /// for the context's life: `ENABLE_DYNAMIC_RESOLUTION` is deliberately not
    /// set, so a resolution change is a destroy and recreate.
    pub fn new(
        hg: &VkHeadless,
        render: (u32, u32),
        upscale: (u32, u32),
        flags: u32,
    ) -> Result<Fsr3, String> {
        #[cfg(not(ffx_fsr3_vk))]
        {
            let _ = (hg, render, upscale, flags);
            return Err("FSR3 was not compiled into this binary".into());
        }
        #[cfg(ffx_fsr3_vk)]
        {
            let vkd = &hg.vk;
            let (rw, rh) = render;
            let (ow, oh) = upscale;

            // The formats are a contract with the shim, which names each one to
            // FFX and whose FfxSurfaceFormat must match what is allocated here —
            // FFX rebuilds its own view from the format it was told, so a
            // disagreement is garbage rather than an error.
            let color = Img::new(vkd, rw, rh, vk::Format::R16G16B16A16_SFLOAT, 8)?;
            let depth = Img::new(vkd, rw, rh, vk::Format::R32_SFLOAT, 4)?;
            let motion = Img::new(vkd, rw, rh, vk::Format::R16G16_SFLOAT, 4)?;
            let output = Img::new(vkd, ow, oh, vk::Format::R16G16B16A16_SFLOAT, 8)?;
            // The SDK's own formats for its shared resources, at RENDER res.
            let dilated_depth = Img::new(vkd, rw, rh, vk::Format::R32_SFLOAT, 4)?;
            let dilated_motion = Img::new(vkd, rw, rh, vk::Format::R16G16_SFLOAT, 4)?;
            let recon_prev_depth = Img::new(vkd, rw, rh, vk::Format::R32_UINT, 4)?;

            let stage_color = vkd.buffer(color.bytes(), vk::BufferUsageFlags::TRANSFER_SRC, true)?;
            let stage_depth = vkd.buffer(depth.bytes(), vk::BufferUsageFlags::TRANSFER_SRC, true)?;
            let stage_motion =
                vkd.buffer(motion.bytes(), vk::BufferUsageFlags::TRANSFER_SRC, true)?;
            let readback = vkd.buffer(output.bytes(), vk::BufferUsageFlags::TRANSFER_DST, true)?;

            let mut handle: *mut c_void = std::ptr::null_mut();
            // `vkGetDeviceProcAddr` comes from ash's loader rather than from a
            // linked symbol, which is what keeps the shim free of any direct
            // Vulkan call of its own.
            let gdpa = vkd.instance.fp_v1_0().get_device_proc_addr;
            let rc = unsafe {
                frshim_fsr3vk_create(
                    vkd.phys.as_raw() as *mut c_void,
                    vkd.device.handle().as_raw() as *mut c_void,
                    gdpa as *mut c_void,
                    rw,
                    rh,
                    ow,
                    oh,
                    flags,
                    &mut handle,
                )
            };
            if rc != 0 || handle.is_null() {
                return Err(format!(
                    "frshim_fsr3vk_create failed ({rc}) — the [fsr3-vk] lines above carry FFX's own diagnosis"
                ));
            }

            let f = Fsr3 {
                handle,
                render,
                upscale,
                color,
                depth,
                motion,
                output,
                dilated_depth,
                dilated_motion,
                recon_prev_depth,
                stage_color,
                stage_depth,
                stage_motion,
                readback,
            };

            // Every image starts UNDEFINED; put them where the dispatch expects
            // them once, so the per-frame path carries only the upload's own
            // transitions. Inputs rest in SHADER_READ_ONLY_OPTIMAL (FFX's
            // COMPUTE_READ), everything FFX writes rests in GENERAL (its
            // UNORDERED_ACCESS).
            hg.run(|d, cmd| {
                for i in [&f.color, &f.depth, &f.motion] {
                    transition(
                        d,
                        cmd,
                        i,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    );
                }
                for i in [&f.output, &f.dilated_depth, &f.dilated_motion, &f.recon_prev_depth] {
                    transition(d, cmd, i, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
                }
            })?;

            Ok(f)
        }
    }

    /// One upscale: upload the three input planes, then dispatch, in ONE submit.
    ///
    /// `color` is linear RGB f32 (three per pixel, i.e. `accum` after the
    /// sample divide); `depth` is `xess::view_z_to_clip_depth`'s reversed-Z clip
    /// depth; `motion` is `GBufs::mvec`'s f16 bit patterns copied verbatim —
    /// pixel-space, y-down, current -> previous, which is why `Dispatch::
    /// mv_scale` is `fsr::UPSCALE_MV_SIGN` rather than a resolution.
    ///
    /// The CPU-fed arm. `frame_fed` is the GPU-fed one; both end in the same
    /// `record_ffx`, which is what keeps the two from drifting in the dispatch
    /// desc — the one place a divergence would read as a quality difference
    /// rather than as a bug.
    pub fn frame(
        &self,
        hg: &VkHeadless,
        color: &[f32],
        depth: &[f32],
        motion: &[u16],
        dp: Dispatch,
    ) -> Result<(), String> {
        #[cfg(not(ffx_fsr3_vk))]
        {
            let _ = (hg, color, depth, motion, dp);
            Err("FSR3 was not compiled into this binary".into())
        }
        #[cfg(ffx_fsr3_vk)]
        {
            let vkd = &hg.vk;
            let (rw, rh) = self.render;
            let px = rw as usize * rh as usize;
            if color.len() != px * 3 || depth.len() != px || motion.len() != px * 2 {
                return Err(format!(
                    "fsr3 frame: plane sizes {} / {} / {} do not match {rw}x{rh}",
                    color.len(),
                    depth.len(),
                    motion.len()
                ));
            }

            // Colour narrows through the SAME saturating converter the D3D12
            // arm uses, so an extreme HDR value becomes f16::MAX and never
            // +inf — the wire discipline every f16 colour plane here follows.
            let mut cbuf = vec![0u16; px * 4];
            for i in 0..px {
                cbuf[i * 4] = crate::fsr::f16_sat(color[i * 3]).to_bits();
                cbuf[i * 4 + 1] = crate::fsr::f16_sat(color[i * 3 + 1]).to_bits();
                cbuf[i * 4 + 2] = crate::fsr::f16_sat(color[i * 3 + 2]).to_bits();
                cbuf[i * 4 + 3] = 0;
            }
            vkd.write(&self.stage_color, bytemuck_u16(&cbuf))?;
            vkd.write(&self.stage_depth, bytemuck_f32(depth))?;
            vkd.write(&self.stage_motion, bytemuck_u16(motion))?;

            let rc = std::cell::Cell::new(0i32);
            hg.run(|d, cmd| {
                for (img, buf) in [
                    (&self.color, &self.stage_color),
                    (&self.depth, &self.stage_depth),
                    (&self.motion, &self.stage_motion),
                ] {
                    transition(
                        d,
                        cmd,
                        img,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    );
                    let r = vk::BufferImageCopy::default()
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D { width: img.w, height: img.h, depth: 1 });
                    unsafe {
                        d.cmd_copy_buffer_to_image(
                            cmd,
                            buf.buf,
                            img.img,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[r],
                        );
                    }
                    transition(
                        d,
                        cmd,
                        img,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    );
                }

                rc.set(self.record_ffx(cmd, &dp));
            })?;
            if rc.get() != 0 {
                return Err(format!("frshim_fsr3vk_dispatch failed ({})", rc.get()));
            }
            Ok(())
        }
    }

    /// The GPU-FED arm: instead of three host uploads, a compute kernel of ours
    /// writes the three input images in place, then FFX consumes them — all in
    /// ONE command buffer, which is D3D12's own "trace -> feed -> upscale on
    /// one list" shape.
    ///
    /// `feed` records that kernel (`VkTracer::record_feed`). It is handed the
    /// device and the command buffer with the three inputs already in
    /// `GENERAL` — the layout a `RWTexture2D` store requires — and this puts
    /// them back in `SHADER_READ_ONLY_OPTIMAL` afterwards, which is where the
    /// CPU-fed path also leaves them and what FFX's `COMPUTE_READ` state
    /// declaration means. So the two arms hand FFX images in the same layout
    /// by construction; only the writer differs.
    pub fn frame_fed<F>(&self, hg: &VkHeadless, feed: F, dp: Dispatch) -> Result<(), String>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer) -> Result<(), String>,
    {
        #[cfg(not(ffx_fsr3_vk))]
        {
            let _ = (hg, feed, dp);
            Err("FSR3 was not compiled into this binary".into())
        }
        #[cfg(ffx_fsr3_vk)]
        {
            let rc = std::cell::Cell::new(0i32);
            let ferr: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
            hg.run(|d, cmd| {
                for i in [&self.color, &self.depth, &self.motion] {
                    transition(
                        d,
                        cmd,
                        i,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::ImageLayout::GENERAL,
                    );
                }
                if let Err(e) = feed(d, cmd) {
                    *ferr.borrow_mut() = Some(e);
                    return;
                }
                for i in [&self.color, &self.depth, &self.motion] {
                    transition(
                        d,
                        cmd,
                        i,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    );
                }
                rc.set(self.record_ffx(cmd, &dp));
            })?;
            if let Some(e) = ferr.borrow_mut().take() {
                return Err(e);
            }
            if rc.get() != 0 {
                return Err(format!("frshim_fsr3vk_dispatch failed ({})", rc.get()));
            }
            Ok(())
        }
    }

    /// The FFX dispatch itself — the tail both arms end in, so neither can
    /// drift from the other in the descriptor a quality comparison rests on.
    #[cfg(ffx_fsr3_vk)]
    fn record_ffx(&self, cmd: vk::CommandBuffer, dp: &Dispatch) -> i32 {
        let (rw, rh) = self.render;
        let (near, far) = dp.near_far;
        unsafe {
            frshim_fsr3vk_dispatch(
                self.handle,
                cmd.as_raw() as *mut c_void,
                self.color.raw(),
                self.depth.raw(),
                self.motion.raw(),
                self.output.raw(),
                self.dilated_depth.raw(),
                self.dilated_motion.raw(),
                self.recon_prev_depth.raw(),
                rw,
                rh,
                self.upscale.0,
                self.upscale.1,
                dp.jitter.0,
                dp.jitter.1,
                dp.mv_scale.0,
                dp.mv_scale.1,
                dp.dt_ms,
                near,
                far,
                dp.fov_y,
                i32::from(dp.reset),
            )
        }
    }

    /// The three input images as `STORAGE_IMAGE` views, in the order
    /// `cs_feed_xess` writes them: colour (u16, RGBA16F), depth (u18, R32F),
    /// mvec (u19, RG16F).
    ///
    /// Returned as an array whose ORDER is the contract, and stated here
    /// because the formats differ while the descriptor type does not — a
    /// swapped pair is legal to Vulkan, legal to the derived layout, and wrong
    /// only in the values, which is why V13 gates the values.
    pub fn feed_views(&self) -> [vk::ImageView; 3] {
        [self.color.view, self.depth.view, self.motion.view]
    }

    /// Read one of the three INPUT planes back, for the feed gate's oracle.
    /// `which` indexes `feed_views`' order.
    pub fn read_input(&self, hg: &VkHeadless, which: usize) -> Result<Vec<u8>, String> {
        let img = match which {
            0 => &self.color,
            1 => &self.depth,
            2 => &self.motion,
            _ => return Err(format!("fsr3 read_input: no plane {which}")),
        };
        // The inputs REST in SHADER_READ_ONLY_OPTIMAL (see `new`), which is
        // NOT a legal `vkCmdCopyImageToBuffer` source — the spec admits only
        // TRANSFER_SRC_OPTIMAL, GENERAL and SHARED_PRESENT_KHR. So hop and hop
        // back, leaving the image where every other path expects to find it.
        // (Written the wrong way once, and the validation layer named the VUID
        // exactly rather than the copy silently returning stale bytes — the
        // good failure mode, and the `--gpu-debug` argument again.)
        hg.run(|d, cmd| {
            transition(
                d,
                cmd,
                img,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            let r = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D { width: img.w, height: img.h, depth: 1 });
            unsafe {
                d.cmd_copy_image_to_buffer(
                    cmd,
                    img.img,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.readback.buf,
                    &[r],
                );
            }
            transition(
                d,
                cmd,
                img,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        })?;
        hg.vk.read(&self.readback, img.bytes() as usize)
    }

    /// Flood the three input images with a byte pattern.
    ///
    /// V13's anti-vacuity: three images that were uploaded once and never
    /// written again still hold plausible contents, so a feed that never ran
    /// compares clean (the M3d lesson, and V3/V9's sentinel for the same
    /// reason). Filling them with a value no real plane can produce is what
    /// separates "the feed wrote this" from "something did, once".
    pub fn poison_inputs(&self, hg: &VkHeadless, byte: u8) -> Result<(), String> {
        let n = self.color.bytes().max(self.depth.bytes()).max(self.motion.bytes()) as usize;
        let stage = hg.vk.buffer(n as u64, vk::BufferUsageFlags::TRANSFER_SRC, true)?;
        hg.vk.write(&stage, &vec![byte; n])?;
        let r = hg.run(|d, cmd| {
            for i in [&self.color, &self.depth, &self.motion] {
                transition(
                    d,
                    cmd,
                    i,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                let reg = vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D { width: i.w, height: i.h, depth: 1 });
                unsafe {
                    d.cmd_copy_buffer_to_image(
                        cmd,
                        stage.buf,
                        i.img,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[reg],
                    );
                }
                transition(
                    d,
                    cmd,
                    i,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
        });
        hg.vk.free_buffer(&stage);
        r
    }

    /// The upscaled frame as linear f32 RGB at output res.
    pub fn read_output(&self, hg: &VkHeadless) -> Result<Vec<f32>, String> {
        let (ow, oh) = self.upscale;
        hg.run(|d, cmd| {
            let r = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D { width: ow, height: oh, depth: 1 });
            unsafe {
                d.cmd_copy_image_to_buffer(
                    cmd,
                    self.output.img,
                    vk::ImageLayout::GENERAL,
                    self.readback.buf,
                    &[r],
                );
            }
        })?;
        let bytes = hg.vk.read(&self.readback, self.output.bytes() as usize)?;
        let px = ow as usize * oh as usize;
        let mut out = vec![0f32; px * 3];
        for i in 0..px {
            for c in 0..3 {
                let o = (i * 4 + c) * 2;
                let h = u16::from_le_bytes([bytes[o], bytes[o + 1]]);
                out[i * 3 + c] = super::tracer::half_from_bits(h);
            }
        }
        Ok(out)
    }

    /// Explicit teardown, the `VkScene`/`VkTracer`/`VkTextures` convention: the
    /// FFX context must die while the device is still alive, and it needs that
    /// device IDLE — which `VkHeadless::run` has already left it, but a
    /// dispatch is the last thing that touched these images, so say it anyway.
    pub fn destroy(self, hg: &VkHeadless) {
        let vkd = &hg.vk;
        unsafe {
            let _ = vkd.device.device_wait_idle();
        }
        #[cfg(ffx_fsr3_vk)]
        unsafe {
            frshim_fsr3vk_destroy(self.handle);
        }
        for i in [
            &self.color,
            &self.depth,
            &self.motion,
            &self.output,
            &self.dilated_depth,
            &self.dilated_motion,
            &self.recon_prev_depth,
        ] {
            i.destroy(vkd);
        }
        for b in [&self.stage_color, &self.stage_depth, &self.stage_motion, &self.readback] {
            vkd.free_buffer(b);
        }
    }
}

// Only the `ffx_fsr3_vk` build calls into this; on macOS the whole arm
// compiles out (the Metal `FfxInterface` stands in), so the allow is
// scoped to exactly that platform rather than blanketed — a dead-code
// warning is signal in this tree and must keep firing on Linux.
#[cfg_attr(not(ffx_fsr3_vk), allow(dead_code))]
fn bytemuck_u16(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

// Only the `ffx_fsr3_vk` build calls into this; on macOS the whole arm
// compiles out (the Metal `FfxInterface` stands in), so the allow is
// scoped to exactly that platform rather than blanketed — a dead-code
// warning is signal in this tree and must keep firing on Linux.
#[cfg_attr(not(ffx_fsr3_vk), allow(dead_code))]
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
