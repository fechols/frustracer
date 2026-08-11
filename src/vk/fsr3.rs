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
#[cfg(ffx_fsr3_src)]
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
/// environment fact (`./install-prerequisites.sh fsr3src`), never a failure —
/// the gate SKIPs on it, the way every absent-SDK path in this tree does.
pub const fn built() -> bool {
    cfg!(ffx_fsr3_src)
}

/// One image plus what it takes to upload to and read from it. Local rather
/// than shared with `tracer.rs`'s: that one is fixed at RGBA16F/STORAGE for the
/// resolve target, and generalising it is a refactor this stage does not need
/// to drag along.
struct Img {
    img: vk::Image,
    mem: vk::DeviceMemory,
    w: u32,
    h: u32,
    /// Bytes per texel of `fmt`, for the staging buffer and the readback. Kept
    /// beside the format so the two can never disagree.
    bpp: u64,
}

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
        Ok(Img { img, mem, w, h, bpp })
    }

    fn bytes(&self) -> u64 {
        self.w as u64 * self.h as u64 * self.bpp
    }

    fn destroy(&self, vkd: &Vk) {
        unsafe {
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
        #[cfg(not(ffx_fsr3_src))]
        {
            let _ = (hg, render, upscale, flags);
            return Err("FSR3 was not compiled into this binary".into());
        }
        #[cfg(ffx_fsr3_src)]
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
    /// pixel-space, y-down, current -> previous, which is why `mv_scale` is
    /// `fsr::UPSCALE_MV_SIGN` rather than a resolution.
    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        &self,
        hg: &VkHeadless,
        color: &[f32],
        depth: &[f32],
        motion: &[u16],
        jitter: (f32, f32),
        mv_scale: (f32, f32),
        near_far: (f32, f32),
        fov_y: f32,
        dt_ms: f32,
        reset: bool,
    ) -> Result<(), String> {
        #[cfg(not(ffx_fsr3_src))]
        {
            let _ = (hg, color, depth, motion, jitter, mv_scale, near_far, fov_y, dt_ms, reset);
            Err("FSR3 was not compiled into this binary".into())
        }
        #[cfg(ffx_fsr3_src)]
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

            let (near, far) = near_far;
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

                let code = unsafe {
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
                        jitter.0,
                        jitter.1,
                        mv_scale.0,
                        mv_scale.1,
                        dt_ms,
                        near,
                        far,
                        fov_y,
                        i32::from(reset),
                    )
                };
                rc.set(code);
            })?;
            if rc.get() != 0 {
                return Err(format!("frshim_fsr3vk_dispatch failed ({})", rc.get()));
            }
            Ok(())
        }
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
        #[cfg(ffx_fsr3_src)]
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

fn bytemuck_u16(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
