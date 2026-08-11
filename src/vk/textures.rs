//! Scene textures as Vulkan images — the `texs[]` table M3a bound and M3b
//! left unwritten.
//!
//! One `VkImage` per `Scene::textures` entry, full mip chain, `_SRGB` vs
//! `_UNORM` by the texture's own role flag — `SceneGpu::new_uploaded`'s
//! contract, expressed in the other API. Nothing about the DATA is re-decided
//! here: `texture.rs` owns the texels, the chain, the slope/variance arms and
//! the sRGB role, and both backends upload what it produced.
//!
//! THE ONE DELIBERATE DIFFERENCE IS BC7, AND IT IS A DEFERRAL RATHER THAN A
//! DIVERGENCE. On D3D12 opaque textures block-compress on the way to the GPU
//! (8 bpp vs 32) through a compute encoder; here every texture uploads as
//! RGBA8, i.e. the `--no-bc7` arm. That is a MEMORY question, not a
//! correctness one, and it moves the CPU-vs-GPU comparison in the safe
//! direction: the CPU samplers keep exact RGBA8 either way, so an uncompressed
//! upload is strictly CLOSER to the reference than the shipping D3D12 default
//! (whose own albedo A/B reads 0.0001-0.0004 against a 0.02 limit precisely
//! because of the encode). Compression lands with the Vulkan session that
//! needs the footprint, and it inherits `bc7::should_compress` verbatim when
//! it does — including the carve-out that alpha-masked and height-carrying
//! textures never compress, which is a VISIBILITY contract.
//!
//! Uploads go through a reusable host-visible staging buffer in batches: one
//! submit per batch rather than one per (texture, mip), because a 313-texture
//! scene is ~4000 subresources and a fence wait each would dominate the load.
//! Staging is host memory on the way out, which is the same "no staging pool
//! yet" boundary the rest of `vk/` sits behind.

use ash::vk;

use crate::scene::Scene;
use crate::vk::device::Vk;
use crate::vk::headless::VkHeadless;

/// One uploaded texture, resting in `SHADER_READ_ONLY_OPTIMAL`.
pub struct Tex {
    pub img: vk::Image,
    pub mem: vk::DeviceMemory,
    pub view: vk::ImageView,
    /// Kept because every barrier here covers the WHOLE chain: a range that
    /// transitioned only level 0 would leave the mips in `UNDEFINED`, which is
    /// a validation error at sample time and a blank minified surface without
    /// the layer to say so.
    pub levels: u32,
}

pub struct VkTextures {
    /// Parallel to `Scene::textures`.
    pub texs: Vec<Tex>,
    /// 1x1 opaque white. Fills the array when the scene has no textures (a
    /// zero-length binding is illegal, so the layout's cap floors at 1) and is
    /// what `FR_VK_DROP_STREAM=texs` substitutes for every entry.
    pub fallback: Tex,
    /// Uploaded bytes, for the summary line — the only number that can tell a
    /// run that uploaded the table from one that uploaded nothing.
    pub bytes: u64,
    /// Subresources uploaded (base + mips, summed). Reported for the same
    /// reason: a chain that silently lost its mips would still show plausible
    /// bytes on a scene of 1x1 textures and nothing else would notice.
    pub levels: u32,
}

/// Batch target. Any single texture whose whole chain exceeds this still gets
/// its own batch — the staging buffer is sized to `max(this, largest chain)`,
/// so a 4K texture is never a failure, just its own submit.
const STAGE_TARGET: u64 = 64 << 20;

impl VkTextures {
    pub fn new(hg: &VkHeadless, scene: &Scene) -> Result<VkTextures, String> {
        let vkd = &hg.vk;

        // Every texture's chain, as (level, w, h, bytes) plus the total. The
        // walk is done once and reused for sizing, allocation and upload so
        // the three cannot disagree about what a chain is.
        let chain_bytes = |t: &crate::texture::Texture| -> u64 {
            let mut n = (t.w as u64) * (t.h as u64) * 4;
            for m in &t.mips {
                n += (m.w as u64) * (m.h as u64) * 4;
            }
            n
        };
        let max_chain = scene.textures.iter().map(chain_bytes).max().unwrap_or(4);
        let stage_size = max_chain.max(STAGE_TARGET);
        let stage = vkd.buffer(stage_size, vk::BufferUsageFlags::TRANSFER_SRC, true)?;

        let mut texs: Vec<Tex> = Vec::with_capacity(scene.textures.len());
        let mut bytes = 0u64;
        let mut levels = 0u32;
        for t in &scene.textures {
            let fmt = if t.srgb {
                vk::Format::R8G8B8A8_SRGB
            } else {
                vk::Format::R8G8B8A8_UNORM
            };
            let n = 1 + t.mips.len() as u32;
            match create_tex(vkd, t.w, t.h, n, fmt) {
                Ok(tex) => texs.push(tex),
                Err(e) => {
                    for x in &texs {
                        x.destroy(vkd);
                    }
                    vkd.free_buffer(&stage);
                    return Err(e);
                }
            }
            bytes += chain_bytes(t);
            levels += n;
        }

        let fallback = match create_tex(vkd, 1, 1, 1, vk::Format::R8G8B8A8_UNORM) {
            Ok(t) => t,
            Err(e) => {
                for x in &texs {
                    x.destroy(vkd);
                }
                vkd.free_buffer(&stage);
                return Err(e);
            }
        };

        let r = (|| -> Result<(), String> {
            // The fallback rides its own tiny batch first, so the loop below
            // deals only with scene textures.
            let white = [255u8; 4];
            upload_batch(hg, &stage, &[(&fallback, vec![(1u32, 1u32, &white[..])])])?;

            let mut batch: Vec<(&Tex, Vec<(u32, u32, &[u8])>)> = Vec::new();
            let mut have = 0u64;
            for (i, t) in scene.textures.iter().enumerate() {
                let sz = chain_bytes(t);
                if have + sz > stage.size && !batch.is_empty() {
                    upload_batch(hg, &stage, &batch)?;
                    batch.clear();
                    have = 0;
                }
                let mut lv: Vec<(u32, u32, &[u8])> = Vec::with_capacity(1 + t.mips.len());
                lv.push((t.w, t.h, texel_bytes(&t.texels)));
                for m in &t.mips {
                    lv.push((m.w, m.h, texel_bytes(&m.texels)));
                }
                batch.push((&texs[i], lv));
                have += sz;
            }
            if !batch.is_empty() {
                upload_batch(hg, &stage, &batch)?;
            }
            Ok(())
        })();
        vkd.free_buffer(&stage);
        if let Err(e) = r {
            for x in &texs {
                x.destroy(vkd);
            }
            fallback.destroy(vkd);
            return Err(e);
        }

        Ok(VkTextures { texs, fallback, bytes, levels })
    }

    pub fn destroy(&self, hg: &VkHeadless) {
        for t in &self.texs {
            t.destroy(&hg.vk);
        }
        self.fallback.destroy(&hg.vk);
    }
}

impl Tex {
    fn destroy(&self, vkd: &Vk) {
        unsafe {
            vkd.device.destroy_image_view(self.view, None);
            vkd.device.destroy_image(self.img, None);
            vkd.device.free_memory(self.mem, None);
        }
    }
}

/// `[u8; 4]` texels as bytes. POD with no padding and alignment 1, so this is
/// a borrow rather than a copy — a per-texel `flat_map` is a real cost at the
/// ~20M texels a big scene carries, and a `Vec` per level would double the
/// batch's host footprint on top of the staging buffer's.
fn texel_bytes(t: &[[u8; 4]]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(t.as_ptr() as *const u8, std::mem::size_of_val(t)) }
}

fn create_tex(vkd: &Vk, w: u32, h: u32, levels: u32, fmt: vk::Format) -> Result<Tex, String> {
    let d = &vkd.device;
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(fmt)
        .extent(vk::Extent3D { width: w.max(1), height: h.max(1), depth: 1 })
        .mip_levels(levels.max(1))
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let img =
        unsafe { d.create_image(&ci, None) }.map_err(|e| format!("vkCreateImage(tex): {e}"))?;
    let req = unsafe { d.get_image_memory_requirements(img) };
    let idx = crate::vk::device::mem_type_index(
        &vkd.mem,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| "no device-local memory type for a scene texture".to_string())?;
    let mem = unsafe {
        d.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(idx),
            None,
        )
    }
    .map_err(|e| format!("vkAllocateMemory(tex): {e}"))?;
    unsafe { d.bind_image_memory(img, mem, 0) }
        .map_err(|e| format!("vkBindImageMemory(tex): {e}"))?;
    let view = unsafe {
        d.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(fmt)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(levels.max(1))
                        .layer_count(1),
                ),
            None,
        )
    }
    .map_err(|e| format!("vkCreateImageView(tex): {e}"))?;
    Ok(Tex { img, mem, view, levels })
}

/// Stage one batch of whole mip chains and land them all in one submit.
fn upload_batch(
    hg: &VkHeadless,
    stage: &crate::vk::device::Buffer,
    batch: &[(&Tex, Vec<(u32, u32, &[u8])>)],
) -> Result<(), String> {
    let mut host: Vec<u8> = Vec::new();
    // (image, level, w, h, offset)
    let mut regions: Vec<(vk::Image, u32, u32, u32, u64)> = Vec::new();
    for (tex, levels) in batch {
        for (lv, (w, h, bytes)) in levels.iter().enumerate() {
            // Buffer offsets for an image copy must be 4-aligned (and a
            // multiple of the texel size, which is 4 here). Every level is a
            // whole number of RGBA8 texels, so the running length already is.
            debug_assert_eq!(host.len() % 4, 0);
            regions.push((tex.img, lv as u32, *w, *h, host.len() as u64));
            host.extend_from_slice(bytes);
        }
    }
    hg.vk.write(stage, &host)?;

    hg.run(|d, cmd| unsafe {
        let to_dst: Vec<vk::ImageMemoryBarrier> = batch
            .iter()
            .map(|(t, _)| {
                vk::ImageMemoryBarrier::default()
                    .image(t.img)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(t.levels)
                            .layer_count(1),
                    )
            })
            .collect();
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_dst,
        );

        for (img, lv, w, h, off) in &regions {
            let r = vk::BufferImageCopy::default()
                .buffer_offset(*off)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(*lv)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D { width: *w, height: *h, depth: 1 });
            d.cmd_copy_buffer_to_image(
                cmd,
                stage.buf,
                *img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[r],
            );
        }

        let to_read: Vec<vk::ImageMemoryBarrier> = batch
            .iter()
            .map(|(t, _)| {
                vk::ImageMemoryBarrier::default()
                    .image(t.img)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(t.levels)
                            .layer_count(1),
                    )
            })
            .collect();
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_read,
        );
    })
}
