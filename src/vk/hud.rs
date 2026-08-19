//! GPU half of the HUD/menu overlay on Vulkan — the peer of `gpu/hud.rs`
//! (the CPU half, Slint software rendering, is `crate::hud`; the wire between
//! them is `gfx::hud_frame`): a window-sized premultiplied RGBA8 image,
//! dirty-rect uploads through a persistently mapped staging ring, and the
//! COMPOSITE DRAW that `vk::display::Passes::record_frame` records after the
//! tonemap — ONE insertion point, so the window, the loading page and any
//! later present arm get the HUD through the same pass. B6b rung 4.
//!
//! TRUE dirty rectangles, the same three layers as D3D12: Slint re-rasterizes
//! only the dirty region, `crate::hud` packs only those rects' bytes, and this
//! module memcpys only those bytes into the ring and records ONE
//! `vkCmdCopyBufferToImage` region per rect. A frame with an unchanged HUD
//! stages nothing, copies nothing and pays only the composite draw; a hidden
//! HUD pays nothing at all. `stats()` reports the copies and bytes rather than
//! asserting them here — V21 is where the zero is asserted, on a synthetic
//! frame; live, `FRUSTRACER_HUD_STATS=1` prints each non-empty upload.
//!
//! WHERE IT DIFFERS FROM THE D3D12 HALF, each deliberately:
//!
//! * **The composite pipeline is not here.** `gpu/hud.rs` owns its PSO because
//!   a D3D12 root signature never rebuilds. `vk::display::Passes` CAN be
//!   rebuilt (a surface format renegotiation on resize), and a pipeline built
//!   against a destroyed layout is the silent class this backend refuses — so
//!   the HUD pipeline and its descriptor set live in `Passes`, keyed on `fmt`
//!   with the tonemap's, and die with it. This module owns the IMAGE and the
//!   UPLOADS; `Passes::bind_overlay` points the set at the image.
//! * **No pitch alignment dance.** D3D12 wants a 256-aligned row pitch and a
//!   512-aligned placed footprint; Vulkan's only constraint on a buffer→image
//!   copy is `bufferOffset % texelSize == 0`, so the ring slice IS the image,
//!   tightly packed at `w*4` per row, and a rect's source is its natural
//!   position in it (`bufferRowLength = w`, the slice's own pitch).
//! * **No interior mutability.** `Presenter::present` is `&mut self`, so
//!   `stage`/`record_upload` take `&mut self` and there is no `RefCell` to
//!   consume at record time. The D3D12 `&self` shape exists because
//!   `fullscreen_to_backbuffer` is `&self` there; nothing here is.
//! * **The image rests in `GENERAL`.** The backend's standing rule
//!   (`display::upload_rgba16f`'s doc): legal for sampling and as a copy
//!   destination alike, so the per-upload barriers are MEMORY barriers and
//!   never a layout transition, and the descriptor `bind_overlay` writes says
//!   `GENERAL` too — the descriptor/layout pairing B5a found broken elsewhere
//!   and fixed structurally. The one transition is `UNDEFINED → GENERAL` at
//!   creation, and its `old_layout` is `UNDEFINED` exactly once: thereafter
//!   every barrier carries `GENERAL → GENERAL`, because an `UNDEFINED` old
//!   layout licenses the driver to DISCARD the texels — on a compressing
//!   implementation (RADV with DCC) that is the rest of the HUD vanishing on
//!   the first partial upload, and llvmpipe would never show it. V21's
//!   partial-rect arm names that as a RADV-proven tooth.
//! * **The image is CLEARED to transparent black at creation**, in the same
//!   submit as its transition. D3D12 never clears its texture and gets away
//!   with it because `crate::hud` forces a full-window FIRST frame, so every
//!   texel is written before `drawable()` can be true; that is a contract
//!   between two modules, and V21's synthetic fixture — three small rects,
//!   no full-window frame — is exactly what breaks it. MEASURED: RADV handed
//!   back zeroed memory and the gate passed; llvmpipe did not, and 1320
//!   background texels composited garbage. Here the guarantee is structural:
//!   an unuploaded texel is `(0,0,0,0)` premultiplied, which the blend leaves
//!   invisible, whatever the first upload covers.
//!
//! RING SLICES follow `headless::FRAMES_IN_FLIGHT`, which is 1: a staged rect
//! is memcpy'd into the slice the previous present's `wait_submit` already
//! retired, which is what makes the write safe with no fence of its own. If a
//! fence ring ever lands, that constant moves and this ring follows.

use ash::vk;

use crate::gfx::hud_frame::{DirtyRect, HudFrame};
use crate::vk::device::{Buffer, Vk};
use crate::vk::display::Image;
use crate::vk::headless::{VkHeadless, FRAMES_IN_FLIGHT};

/// What one `record_upload` did — copies recorded and bytes memcpy'd.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UploadStats {
    pub rects: usize,
    pub bytes: usize,
}

pub struct HudVk {
    /// The overlay image (premultiplied RGBA8), resting in `GENERAL`.
    pub image: Image,
    /// `FRAMES_IN_FLIGHT` window-sized slices, persistently mapped.
    staging: Buffer,
    ptr: *mut u8,
    /// One slice = one full window, tightly packed.
    slice: usize,
    pitch: usize,
    /// Staged, not yet recorded: rects and their tightly packed bytes, in
    /// `gfx::hud_frame`'s layout, appended across stages.
    rects: Vec<DirtyRect>,
    bytes: Vec<u8>,
    /// The image holds valid pixels (the first upload happened) — compositing
    /// before that would blend uninitialized memory.
    uploaded: bool,
    /// Whether the composite draw runs (the HUD/menu is on screen). Fed per
    /// frame by the session; staging keeps happening while hidden so the image
    /// stays current and re-showing needs no special case.
    pub visible: bool,
    /// Cumulative `record_upload` totals since creation — the live probe.
    total: UploadStats,
    /// Which ring slice the next upload memcpys into.
    frame: usize,
}

impl HudVk {
    /// Create the image and the ring at the window's extent, and run the one
    /// `UNDEFINED → GENERAL` transition plus the clear to transparent black
    /// (its own submit, through `hg.run`, which is legal here because nothing
    /// is in flight: callers are the presenter's constructor and its resize,
    /// both after a `wait_submit`).
    pub fn new(hg: &VkHeadless, w: u32, h: u32) -> Result<HudVk, String> {
        let vkd = &hg.vk;
        let image = Image::new(
            vkd,
            w,
            h,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        let pitch = w as usize * 4;
        let slice = pitch * h as usize;
        let staging = match vkd.buffer(
            (slice * FRAMES_IN_FLIGHT) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            true,
        ) {
            Ok(b) => b,
            Err(e) => {
                image.destroy(vkd);
                return Err(e);
            }
        };
        let ptr = match unsafe {
            vkd.device.map_memory(staging.mem, 0, staging.size, vk::MemoryMapFlags::empty())
        } {
            Ok(p) => p as *mut u8,
            Err(e) => {
                vkd.free_buffer(&staging);
                image.destroy(vkd);
                return Err(format!("vkMapMemory(hud staging): {e}"));
            }
        };
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let img = image.img;
        if let Err(e) = hg.run(|d, cmd| unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(img)
                    .subresource_range(range)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
            );
            d.cmd_clear_color_image(
                cmd,
                img,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
                &[range],
            );
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(img)
                    .subresource_range(range)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_READ,
                    )],
            );
        }) {
            unsafe { vkd.device.unmap_memory(staging.mem) };
            vkd.free_buffer(&staging);
            image.destroy(vkd);
            return Err(format!("hud image init transition: {e}"));
        }
        Ok(HudVk {
            image,
            staging,
            ptr,
            slice,
            pitch,
            rects: Vec::new(),
            bytes: Vec::new(),
            uploaded: false,
            visible: false,
            total: UploadStats::default(),
            frame: 0,
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.image.w, self.image.h)
    }

    /// Stage a frame's dirty rects (once per frame, before the present
    /// records). Appends — an arm that errors out after staging leaves the
    /// rects for the next recorded frame rather than dropping them.
    pub fn stage(&mut self, frame: HudFrame) {
        if frame.rects.is_empty() {
            return;
        }
        self.rects.extend_from_slice(&frame.rects);
        self.bytes.extend_from_slice(&frame.bytes);
    }

    /// True when the composite draw should record: something is on screen and
    /// the image holds real pixels.
    pub fn drawable(&self) -> bool {
        self.visible && self.uploaded
    }

    /// Cumulative upload totals — the live dirty-rect probe (`--qa pos`).
    pub fn stats(&self) -> UploadStats {
        self.total
    }

    /// Consume the staged rects: memcpy each into this frame's ring slice at
    /// its natural position and record one copy region per rect, bracketed by
    /// the two memory barriers (fragment read → transfer write → fragment
    /// read). Returns what it did; `(0, 0)` when nothing was staged, in which
    /// case NOTHING is recorded — not even a barrier.
    pub fn record_upload(&mut self, d: &ash::Device, cmd: vk::CommandBuffer) -> UploadStats {
        if self.rects.is_empty() {
            return UploadStats::default();
        }
        let (w, h) = (self.image.w, self.image.h);
        let slot = self.frame % FRAMES_IN_FLIGHT;
        self.frame = self.frame.wrapping_add(1);
        let base = slot * self.slice;
        // SAFETY: the mapping covers `FRAMES_IN_FLIGHT * slice` bytes and
        // `slot < FRAMES_IN_FLIGHT`; the previous present's `wait_submit`
        // retired the copy that last read this slice.
        let dst = unsafe { std::slice::from_raw_parts_mut(self.ptr.add(base), self.slice) };

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.image.img)
                    .subresource_range(range)
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
            );
        }

        let mut regions: Vec<vk::BufferImageCopy> = Vec::with_capacity(self.rects.len());
        let mut src_off = 0usize;
        let mut bytes = 0usize;
        for r in &self.rects {
            // Defensive clamp — a rect from a stale (pre-resize) frame must
            // not index outside the ring (`gpu/hud.rs`'s clamp, same reason).
            let x = r.x.min(w);
            let y = r.y.min(h);
            let rw = r.w.min(w - x);
            let rh = r.h.min(h - y);
            let row_bytes = r.w as usize * 4;
            let copy_bytes = rw as usize * 4;
            for row in 0..rh as usize {
                let o = (y as usize + row) * self.pitch + x as usize * 4;
                dst[o..o + copy_bytes]
                    .copy_from_slice(&self.bytes[src_off + row * row_bytes..][..copy_bytes]);
            }
            src_off += row_bytes * r.h as usize;
            if rw == 0 || rh == 0 {
                continue;
            }
            bytes += copy_bytes * rh as usize;
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset((base + y as usize * self.pitch + x as usize * 4) as u64)
                    .buffer_row_length(w)
                    .buffer_image_height(h)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D { x: x as i32, y: y as i32, z: 0 })
                    .image_extent(vk::Extent3D { width: rw, height: rh, depth: 1 }),
            );
        }
        unsafe {
            if !regions.is_empty() {
                d.cmd_copy_buffer_to_image(
                    cmd,
                    self.staging.buf,
                    self.image.img,
                    vk::ImageLayout::GENERAL,
                    &regions,
                );
            }
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.image.img)
                    .subresource_range(range)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            );
        }
        let did = UploadStats { rects: regions.len(), bytes };
        self.rects.clear();
        self.bytes.clear();
        if did.rects > 0 {
            self.uploaded = true;
        }
        self.total.rects += did.rects;
        self.total.bytes += did.bytes;
        did
    }

    pub fn destroy(&self, vkd: &Vk) {
        unsafe { vkd.device.unmap_memory(self.staging.mem) };
        vkd.free_buffer(&self.staging);
        self.image.destroy(vkd);
    }
}
