//! Scene textures as Vulkan images — the `texs[]` table M3a bound and M3b
//! left unwritten.
//!
//! One `VkImage` per `Scene::textures` entry, full mip chain, `_SRGB` vs
//! `_UNORM` by the texture's own role flag — `SceneGpu::new_uploaded`'s
//! contract, expressed in the other API. Nothing about the DATA is re-decided
//! here: `texture.rs` owns the texels, the chain, the slope/variance arms and
//! the sRGB role, and both backends upload what it produced.
//!
//! BC7 IS REAL HERE (M3j), in both arms, and nothing about the DECISION is
//! re-made: `bc7::should_compress` is called verbatim, carve-out included —
//! alpha-masked cutout masks and height-carrying relief fields stay exact
//! RGBA8 because the intersector `.Load()`s a hard threshold out of them, and
//! that is a VISIBILITY contract rather than a per-backend quality choice.
//! The `Texture::srgb` role picks `BC7_SRGB_BLOCK` vs `BC7_UNORM_BLOCK`
//! exactly as it picks `_SRGB` vs `_UNORM` for RGBA8.
//!
//! Three per-level arms, D3D12's `TexArm` in the other API: RGBA8 straight
//! through, the `--bc7-cpu` ispc arm's pre-encoded blocks straight through,
//! and the default GPU arm, which stages the SOURCE texels and encodes them
//! on the way in (`vk::bc7`) so the blocks never touch the CPU. The kernel is
//! shared with D3D12; see that module for what a second host cost it (one
//! byte-offset window, and nothing else).
//!
//! `textureCompressionBC` is enabled-when-present, so a device without BC
//! support runs the `--no-bc7` arm loudly. That degrade is honest rather than
//! merely safe: an uncompressed upload is strictly CLOSER to the CPU
//! reference than the compressed default, since the CPU samplers keep exact
//! RGBA8 either way — which is why the D3D12 default's own albedo A/B reads
//! 0.0001-0.0004 against a 0.02 limit while `--no-bc7` reads zero.
//!
//! Uploads go through `vk::stage`'s ring in batches: one submit per batch
//! rather than one per (texture, mip), because a 313-texture scene is ~3000
//! subresources and a fence wait each would dominate the load. This module
//! had its own private staging buffer first and gave it up when the scene
//! streams grew one — the sizing rule, the mapping and the accounting are
//! things there should be one of, and the batch loop below (which is genuinely
//! image-specific, since its destinations are subresources rather than a byte
//! range) is what stayed. THE GPU ENCODER ADDS A SECOND BUDGET TO THAT LOOP
//! and it is not the obvious one: a batch is bounded by the ring in SOURCE
//! bytes and by the block buffer in ENCODED bytes, and encoded is not simply
//! a quarter of source — a 1x1 mip is 4 raw bytes and a whole 16-byte block,
//! so a scene of tiny textures can encode to FOUR TIMES what it staged.

use ash::vk;

use crate::bc7;
use crate::scene::Scene;
use crate::vk::bc7::VkBc7;
use crate::vk::device::Vk;
use crate::vk::headless::VkHeadless;
use crate::vk::spirv::Spirv;
use crate::vk::stage::Stage;

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
    /// Bytes staged and submits spent, for the report line.
    pub staged: (u64, u32),
    /// Textures that block-compressed, and what the table costs on the device
    /// either way. Both are reported rather than asserted, for the reason
    /// `bytes` is: an armed run and a silently-degraded one upload the same
    /// COUNT of textures, and only the footprint tells them apart.
    pub n_bc7: usize,
    pub device_bytes: u64,
    /// Which arm ran, in one phrase, and why if it was not the one asked for.
    pub bc7_note: String,
}

/// One level's staging form. The three arms are D3D12's `TexArm` per (texture,
/// mip) rather than per texture, because the ARM is a texture property and the
/// BYTES are a level property, and only the second one the batch loop budgets.
enum Src<'a> {
    /// RGBA8 texels, copied straight to the image.
    Rgba(&'a [u8]),
    /// Pre-encoded blocks (`--bc7-cpu`), copied straight to the image.
    Blk(&'a [u8]),
    /// RGBA8 texels staged for the GPU encoder, whose blocks are copied.
    Enc(&'a [u8]),
}

impl Src<'_> {
    fn bytes(&self) -> &[u8] {
        match self {
            Src::Rgba(b) | Src::Blk(b) | Src::Enc(b) => b,
        }
    }
}

/// Round a staging offset up to BC7's block size.
///
/// `vkCmdCopyBufferToImage` requires `bufferOffset` to be a multiple of the
/// compressed texel block size (16 B) when the destination is a BC format,
/// and the kernel's `Store4` wants the same of `dst_off`. Applied to EVERY
/// level rather than only compressed ones: RGBA8 needs 4, 16 is a superset,
/// and one rule that always holds beats a conditional one that holds where
/// somebody remembered it.
fn align16(n: u64) -> u64 {
    n.next_multiple_of(bc7::BLOCK_BYTES as u64)
}

impl VkTextures {
    pub fn new(
        hg: &VkHeadless,
        sp: &Spirv,
        scene: &Scene,
        mode: bc7::Bc7Mode,
    ) -> Result<VkTextures, String> {
        let vkd = &hg.vk;

        // WHICH TEXTURES COMPRESS — `bc7::should_compress` verbatim, plus the
        // device's own BC support. A device without it takes the `--no-bc7`
        // arm, which is a shipping configuration and not a degradation
        // invented here.
        let bc_ok = vkd.info.texture_compression_bc;
        let mut compress: Vec<bool> = scene
            .textures
            .iter()
            .map(|t| mode.armed() && bc_ok && bc7::should_compress(t))
            .collect();
        let mut note = if !mode.armed() {
            "RGBA8 — the --no-bc7 arm".to_string()
        } else if !bc_ok {
            "RGBA8 — this device has no textureCompressionBC".to_string()
        } else {
            String::new()
        };

        // The `--bc7-cpu` arm pre-encodes here (largest-first, the D3D12
        // upload path's own LPT scheduling: an ispc encode is dominated by its
        // biggest textures and a naive order leaves one straggler running
        // alone). It shares no code with the kernel, which is what makes it an
        // independent cross-check rather than a second spelling.
        let mut cpu_blocks: Vec<Option<Vec<Vec<u8>>>> =
            scene.textures.iter().map(|_| None).collect();
        if let bc7::Bc7Mode::Cpu(q) = mode {
            use rayon::prelude::*;
            let t0 = std::time::Instant::now();
            let mut order: Vec<usize> = (0..scene.textures.len()).filter(|&i| compress[i]).collect();
            order.sort_by_key(|&i| {
                std::cmp::Reverse(scene.textures[i].w as u64 * scene.textures[i].h as u64)
            });
            let done: Vec<(usize, Vec<Vec<u8>>)> = order
                .par_iter()
                .map(|&i| {
                    let t = &scene.textures[i];
                    let mut lv = vec![bc7::encode_opaque(t, q)];
                    lv.extend(t.mips.iter().map(|m| bc7::encode_level(m.w, m.h, &m.texels, q)));
                    (i, lv)
                })
                .collect();
            for (i, b) in done {
                cpu_blocks[i] = Some(b);
            }
            if !order.is_empty() {
                note = format!("BC7 via the ispc CPU arm in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
        }

        // Sizes, per arm. `stage_bytes` is what the RING carries (source
        // texels for the GPU arm, encoded blocks for the CPU arm) and
        // `enc_bytes` is what the BLOCK BUFFER carries; the batch loop budgets
        // against both because they are not proportional (module header).
        let raw_chain = |t: &crate::texture::Texture| -> u64 {
            let mut n = align16((t.w as u64) * (t.h as u64) * 4);
            for m in &t.mips {
                n += align16((m.w as u64) * (m.h as u64) * 4);
            }
            n
        };
        let enc_chain = |t: &crate::texture::Texture| -> u64 {
            let mut n = align16(bc7::encoded_len(t.w, t.h) as u64);
            for m in &t.mips {
                n += align16(bc7::encoded_len(m.w, m.h) as u64);
            }
            n
        };
        let stage_chain = |i: usize| -> u64 {
            let t = &scene.textures[i];
            if cpu_blocks[i].is_some() { enc_chain(t) } else { raw_chain(t) }
        };
        // The whole table is what this ring carries, and one whole mip chain
        // is the piece that must fit UNDIVIDED — the batch loop below splits
        // between textures and never inside one, so a 4K texture is never a
        // failure, just its own submit. That `atom` is the reason `Stage::new`
        // takes one at all.
        let max_chain = (0..scene.textures.len()).map(stage_chain).max().unwrap_or(4);
        let total: u64 = (0..scene.textures.len()).map(stage_chain).sum();
        let mut stage = Stage::new(vkd, total, max_chain)?;

        // The GPU encoder, and D3D12's rule for it verbatim: a construction
        // failure is one LOUD line and an uncompressed upload, never an
        // implicit ~20-second CPU encode. Sized to the largest batch this ring
        // can produce, floored at one whole chain for the same reason `atom`
        // is.
        let mut enc: Option<VkBc7> = None;
        if let bc7::Bc7Mode::Gpu(q) = mode {
            let want: u64 = (0..scene.textures.len())
                .filter(|&i| compress[i])
                .map(|i| enc_chain(&scene.textures[i]))
                .sum();
            let atom = (0..scene.textures.len())
                .filter(|&i| compress[i])
                .map(|i| enc_chain(&scene.textures[i]))
                .max()
                .unwrap_or(0);
            if want > 0 {
                // `.max(atom)` is the SECOND budget's correctness floor, and
                // it is what makes the batch loop's `have_enc + esz >
                // block_cap` test safe rather than merely usual: a chain
                // bigger than the cap would flush an empty batch and then
                // overrun anyway, exactly the `Stage::new` overrun its own
                // header names.
                let cap = want.min(stage.size()).max(atom);
                match VkBc7::new(hg, sp, cap, q) {
                    Ok(e) => {
                        e.bind_src(vkd, stage.buf());
                        enc = Some(e);
                        note = format!("BC7 via the GPU encoder (effort {})", q.effort());
                    }
                    Err(e) => {
                        eprintln!(
                            "bc7: GPU encoder unavailable ({e}) — textures upload UNCOMPRESSED \
                             RGBA8 (--bc7-cpu forces the CPU encode)"
                        );
                        compress.iter_mut().for_each(|c| *c = false);
                        note = "RGBA8 — the GPU encoder failed to construct".to_string();
                    }
                }
            }
        }
        let n_bc7 = compress.iter().filter(|&&c| c).count();
        if n_bc7 == 0 && note.is_empty() {
            note = "RGBA8 — nothing in this scene is compressible".to_string();
        }
        let block_cap = enc.as_ref().map_or(0, |e| e.block_cap);

        let free_enc = |e: &Option<VkBc7>| {
            if let Some(e) = e {
                e.destroy(vkd);
            }
        };

        let mut texs: Vec<Tex> = Vec::with_capacity(scene.textures.len());
        let mut bytes = 0u64;
        let mut device_bytes = 0u64;
        let mut levels = 0u32;
        for (i, t) in scene.textures.iter().enumerate() {
            let fmt = if compress[i] {
                crate::vk::bc7::vk_format(t)
            } else if t.srgb {
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
                    free_enc(&enc);
                    stage.free(vkd);
                    return Err(e);
                }
            }
            bytes += raw_chain(t);
            device_bytes += if compress[i] { enc_chain(t) } else { raw_chain(t) };
            levels += n;
        }

        let fallback = match create_tex(vkd, 1, 1, 1, vk::Format::R8G8B8A8_UNORM) {
            Ok(t) => t,
            Err(e) => {
                for x in &texs {
                    x.destroy(vkd);
                }
                free_enc(&enc);
                stage.free(vkd);
                return Err(e);
            }
        };

        let r = (|| -> Result<(), String> {
            // The fallback rides its own tiny batch first, so the loop below
            // deals only with scene textures.
            let white = [255u8; 4];
            upload_batch(
                hg,
                &mut stage,
                &enc,
                &[(&fallback, vec![(1u32, 1u32, Src::Rgba(&white[..]))])],
            )?;

            let mut batch: Vec<(&Tex, Vec<(u32, u32, Src)>)> = Vec::new();
            let mut have = 0u64;
            let mut have_enc = 0u64;
            for (i, t) in scene.textures.iter().enumerate() {
                let sz = stage_chain(i);
                // TWO budgets, and the second one is why a batch can be
                // ring-sized and still not fit: encoded bytes are ~a quarter
                // of source for a big level and FOUR TIMES it for a 1x1 mip,
                // so a table of small textures overruns the block buffer long
                // before it fills the ring.
                let esz = if compress[i] && enc.is_some() { enc_chain(t) } else { 0 };
                if !batch.is_empty()
                    && (have + sz > stage.size() || have_enc + esz > block_cap)
                {
                    upload_batch(hg, &mut stage, &enc, &batch)?;
                    batch.clear();
                    have = 0;
                    have_enc = 0;
                }
                let mut lv: Vec<(u32, u32, Src)> = Vec::with_capacity(1 + t.mips.len());
                match (&cpu_blocks[i], compress[i]) {
                    (Some(b), true) => {
                        lv.push((t.w, t.h, Src::Blk(&b[0])));
                        for (k, m) in t.mips.iter().enumerate() {
                            lv.push((m.w, m.h, Src::Blk(&b[k + 1])));
                        }
                    }
                    (_, true) => {
                        lv.push((t.w, t.h, Src::Enc(texel_bytes(&t.texels))));
                        for m in &t.mips {
                            lv.push((m.w, m.h, Src::Enc(texel_bytes(&m.texels))));
                        }
                    }
                    _ => {
                        lv.push((t.w, t.h, Src::Rgba(texel_bytes(&t.texels))));
                        for m in &t.mips {
                            lv.push((m.w, m.h, Src::Rgba(texel_bytes(&m.texels))));
                        }
                    }
                }
                batch.push((&texs[i], lv));
                have += sz;
                have_enc += esz;
            }
            if !batch.is_empty() {
                upload_batch(hg, &mut stage, &enc, &batch)?;
            }
            Ok(())
        })();
        let staged = (stage.bytes(), stage.chunks());
        stage.free(vkd);
        free_enc(&enc);
        if let Err(e) = r {
            for x in &texs {
                x.destroy(vkd);
            }
            fallback.destroy(vkd);
            return Err(e);
        }

        Ok(VkTextures {
            texs,
            fallback,
            bytes,
            levels,
            staged,
            n_bc7,
            device_bytes,
            bc7_note: note,
        })
    }

    pub fn destroy(&self, hg: &VkHeadless) {
        for t in &self.texs {
            t.destroy(&hg.vk);
        }
        self.fallback.destroy(&hg.vk);
    }
}

impl Tex {
    pub(crate) fn destroy(&self, vkd: &Vk) {
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

pub(crate) fn create_tex(
    vkd: &Vk,
    w: u32,
    h: u32,
    levels: u32,
    fmt: vk::Format,
) -> Result<Tex, String> {
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

/// Where one level's bytes come from at copy time, and where they go.
struct Region {
    img: vk::Image,
    level: u32,
    w: u32,
    h: u32,
    /// Byte offset into the STAGING RING of this level's staged bytes.
    src: u64,
    /// Byte offset into the BLOCK BUFFER, when the GPU encoder produces this
    /// level. `None` copies straight out of the ring.
    enc: Option<u64>,
    /// Compressed destination, i.e. the copy speaks blocks rather than texels.
    block: bool,
}

/// Stage one batch of whole mip chains and land them all in one submit.
///
/// ONE SUBMIT covers all three arms, encoder included, and that ordering is
/// the whole shape: every `Src::Enc` level dispatches first, then ONE barrier
/// makes every block visible, then every copy runs. Interleaving
/// dispatch-and-copy per level would buy nothing (the constants already
/// serialize the dispatches — see `vk::bc7`) and cost a barrier pair each.
fn upload_batch(
    hg: &VkHeadless,
    stage: &mut Stage,
    enc: &Option<VkBc7>,
    batch: &[(&Tex, Vec<(u32, u32, Src)>)],
) -> Result<(), String> {
    let mut host: Vec<u8> = Vec::new();
    let mut regions: Vec<Region> = Vec::new();
    let mut enc_off = 0u64;
    for (tex, levels) in batch {
        for (lv, (w, h, src)) in levels.iter().enumerate() {
            // 16 rather than the 4 an RGBA8 copy needs: a BC destination's
            // `bufferOffset` must be a multiple of the 16-byte compressed
            // block, and one alignment rule that always holds beats a
            // conditional one (see `align16`).
            let at = align16(host.len() as u64);
            host.resize(at as usize, 0);
            let e = match src {
                Src::Enc(_) => {
                    let off = enc_off;
                    enc_off += align16(crate::vk::bc7::VkBc7::level_bytes(*w, *h));
                    Some(off)
                }
                _ => None,
            };
            regions.push(Region {
                img: tex.img,
                level: lv as u32,
                w: *w,
                h: *h,
                src: at,
                enc: e,
                block: e.is_some() || matches!(src, Src::Blk(_)),
            });
            host.extend_from_slice(src.bytes());
        }
    }
    stage.write(&host)?;

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

        // Encode first, all of it, then one fence. The dispatches read the
        // ring the host just wrote, which the batch's own submit ordering
        // covers (a host write in this closure happens before the submit).
        if let Some(e) = enc {
            let mut any = false;
            for r in &regions {
                if let Some(dst) = r.enc {
                    e.record_encode(d, cmd, r.w, r.h, r.src, dst);
                    any = true;
                }
            }
            if any {
                e.record_flush(d, cmd);
            }
        }

        for r in &regions {
            // `bufferRowLength` is in TEXELS. For a BC destination it must be
            // a multiple of the 4-texel block width, which `blocks(w) * 4` is
            // by construction — and it EXCEEDS `w` whenever the level is not
            // 4-aligned, which is exactly the padding the encoder's edge
            // clamp (and `encode_level`'s replicate) produced. D3D12's own
            // 256-byte block pitch has no counterpart here, so the shear trap
            // the kernel's header names cannot arise on this side.
            //
            // MEASURED EQUIVALENT TO 0, and stated because the natural
            // suspicion is that one of the two is wrong: a zero row length
            // means "tightly packed according to imageExtent", and for a
            // compressed format that already rounds the extent UP to whole
            // blocks — so this is documentation of a layout, not a correction
            // to one (a planted 0 changed no number on any scene). What is
            // NOT free is the OFFSET: `align16` above is load-bearing, and a
            // 4-aligned one is a validation error naming this exact copy.
            let (row_len, img_h) = if r.block {
                (bc7::blocks(r.w) * bc7::BLOCK, bc7::blocks(r.h) * bc7::BLOCK)
            } else {
                (0, 0)
            };
            let (buf, off) = match r.enc {
                Some(dst) => (enc.as_ref().expect("Enc region without an encoder").block_buf().buf, dst),
                None => (stage.buf().buf, r.src),
            };
            let region = vk::BufferImageCopy::default()
                .buffer_offset(off)
                .buffer_row_length(row_len)
                .buffer_image_height(img_h)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(r.level)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D { width: r.w, height: r.h, depth: 1 });
            d.cmd_copy_buffer_to_image(
                cmd,
                buf,
                r.img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
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
