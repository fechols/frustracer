//! The GPU BC7 encoder on Vulkan — `gpu/bc7gpu.rs`'s peer over the same
//! kernel.
//!
//! `shaders/bc7enc.hlsl` is not ported: DXC compiles it to SPIR-V exactly as
//! it compiles it to DXIL, and the ONE thing that had to change for two hosts
//! is the window (`src_off`/`dst_off`), because D3D12 slides root SRV/UAV
//! virtual addresses where Vulkan binds one descriptor per buffer. Zero
//! offsets are exact integer identities, so both hosts encode the same blocks
//! from the same source.
//!
//! # Why a GPU encoder here at all
//!
//! Because the alternative is not "RGBA8", it is "a 20-second stall". The
//! session default is `Gpu(Fast)`, and D3D12's own rule is that an encoder
//! that fails to construct falls back LOUDLY to uncompressed rather than
//! silently to the ispc path (`bc7::encode_opaque` measured ~20 s for Intel
//! Sponza, and there is deliberately no BC7 disk cache). A Vulkan backend
//! that only had the CPU arm would have to pick one of those two behaviours
//! for its default session and would be wrong either way.
//!
//! # Three differences from the D3D12 host, and only three
//!
//! - **No banding.** D3D12 streams each mip through a 256 MB ring in row
//!   bands because an upload heap's rows are 256-byte-pitch-aligned and one
//!   4K mip does not fit; here the texture path's batch loop already
//!   guarantees a whole mip chain fits the ring, so one dispatch covers a
//!   whole level with a TIGHT `mw * 4` source pitch. The kernel's clamp still
//!   edge-replicates the bottom and right edges, which is what makes a
//!   non-4-aligned mip legal.
//! - **The constants are a UNIFORM BUFFER**, not root constants — DXC has no
//!   flag to promote a `cbuffer` to push constants and `[[vk::push_constant]]`
//!   would be an HLSL edit. So they are rewritten between dispatches with
//!   `vkCmdUpdateBuffer`, which is the ladder's `push` shape, and they carry
//!   the ladder's lesson with them: BOTH barriers, because a per-dispatch
//!   constant block needs a WRITE-AFTER-READ edge as well as the obvious
//!   read-after-write one. The consequence is that the dispatches in one batch
//!   SERIALIZE against each other — acceptable because a 4K level is a million
//!   blocks and fills the machine on its own, and because the alternative
//!   (one descriptor set per level) is allocation churn for the tiny mips
//!   that are the only ones that would benefit.
//! - **Tight block rows.** `row_blocks` is `bw`, not D3D12's 256-byte-aligned
//!   `block_pitch`: `vkCmdCopyBufferToImage` takes `bufferRowLength` in
//!   TEXELS with no pitch alignment to satisfy, so the shear trap the kernel's
//!   header names does not exist on this side.
//!
//! # What is NOT re-decided here
//!
//! `bc7::should_compress` — the alpha-masked and height-carrying carve-outs
//! are a VISIBILITY contract (see that module's header), not a per-backend
//! choice, and the predicate is called verbatim. Neither is the format role
//! split: `Texture::srgb` picks `_SRGB` vs `_UNORM` for BC7 exactly as it does
//! for RGBA8.

use ash::vk;

use crate::bc7;
use crate::gfx::shaders::BC7ENC_HLSL;
use crate::vk::device::{Buffer, Vk};
use crate::vk::headless::VkHeadless;
use crate::vk::layout::{self, Layouts};
use crate::vk::reflect::{self, Map};
use crate::vk::spirv::{self, Spirv};

/// The encoder's own descriptor family: `t0` staged texels, `u0` blocks,
/// `b0` constants. Deliberately its OWN map and layout rather than a corner of
/// the tracer's — the register numbers collide with the tracer's `bvh_nodes` /
/// `accum` / `FrameCb`, and `reflect::Map` would report that as a conflict and
/// be right to (see `vk/layout.rs`: the FAMILY is the unit).
pub struct VkBc7 {
    layouts: Layouts,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
    /// `cbuffer C : register(b0)`, rewritten per dispatch.
    cb: Buffer,
    /// Encoded blocks, device-local, read back out by `cmd_copy_buffer_to_image`.
    blocks: Buffer,
    /// Bytes of `blocks` — the batch budget the caller must respect.
    pub block_cap: u64,
    effort: u32,
}

impl VkBc7 {
    /// `block_cap` = the largest encoded byte total one batch may hold.
    pub fn new(
        hg: &VkHeadless,
        sp: &Spirv,
        block_cap: u64,
        q: bc7::Quality,
    ) -> Result<VkBc7, String> {
        let vkd = &hg.vk;
        // cs_6_0 rather than D3D12's fxc cs_5_0: the "no DXC dependency"
        // argument that picked 5_0 there is a D3D12 fact (fxc ships in the OS,
        // so the encoder exists before any tracer kernel does), and here DXC
        // IS the only compiler. The source is the same text either way.
        let words = sp.compile(BC7ENC_HLSL, "cs_bc7_encode", "cs_6_0", "bc7enc", false)?;
        let descs = reflect::reflect(&words)?;
        let mut map = Map::default();
        let conflicts = map.add("bc7enc", &descs);
        if !conflicts.is_empty() {
            return Err(format!("bc7: the encoder's own register map conflicts: {conflicts:?}"));
        }
        // No unbounded arrays in this unit, so the cap is irrelevant; 1 states
        // that rather than inheriting a number from a family it is not in.
        let layouts = Layouts::build(vkd, &map, 1, None)?;

        let d = &vkd.device;
        let mut counts: std::collections::BTreeMap<vk::DescriptorType, u32> = Default::default();
        for e in map.entries.values() {
            *counts.entry(layout::desc_type(e.kind)).or_default() += e.count.max(1);
        }
        let sizes: Vec<vk::DescriptorPoolSize> = counts
            .iter()
            .map(|(&ty, &n)| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(n))
            .collect();
        let pool = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| format!("bc7: vkCreateDescriptorPool: {e}"))?;
        let set = match unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts.sets),
            )
        } {
            Ok(s) => s[0],
            Err(e) => {
                unsafe { d.destroy_descriptor_pool(pool, None) };
                layouts.destroy(vkd);
                return Err(format!("bc7: vkAllocateDescriptorSets: {e}"));
            }
        };

        let pipeline = match layout::compute_pipeline(vkd, &layouts, &words, "cs_bc7_encode") {
            Ok(p) => p,
            Err(e) => {
                unsafe { d.destroy_descriptor_pool(pool, None) };
                layouts.destroy(vkd);
                return Err(e);
            }
        };

        // 36 bytes of constants; 64 keeps the update a whole number of the
        // 4-byte units `vkCmdUpdateBuffer` requires with room to grow.
        let cb = vkd.buffer(64, vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, false)?;
        let blocks = vkd.buffer(
            block_cap.max(bc7::BLOCK_BYTES as u64),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            false,
        )?;

        let e = VkBc7 {
            layouts,
            pool,
            set,
            pipeline,
            cb,
            blocks,
            block_cap: block_cap.max(bc7::BLOCK_BYTES as u64),
            effort: q.effort(),
        };
        e.write_desc(vkd, spirv::Reg::B, 0, &e.cb, vk::DescriptorType::UNIFORM_BUFFER);
        e.write_desc(vkd, spirv::Reg::U, 0, &e.blocks, vk::DescriptorType::STORAGE_BUFFER);
        Ok(e)
    }

    /// Point `t0` at the staging ring a batch will be written through.
    ///
    /// Called once per `Stage`, never per level — the per-level window rides
    /// the constants. The ring is HOST_VISIBLE, which is what D3D12's arm does
    /// too (its `src` is the upload heap read straight as a root SRV): a
    /// shader reading host memory is the right trade for a one-off load-time
    /// pass whose alternative is a second full copy of every texel.
    pub fn bind_src(&self, vkd: &Vk, src: &Buffer) {
        self.write_desc(vkd, spirv::Reg::T, 0, src, vk::DescriptorType::STORAGE_BUFFER);
    }

    fn write_desc(&self, vkd: &Vk, reg: spirv::Reg, n: u32, b: &Buffer, ty: vk::DescriptorType) {
        let info = [vk::DescriptorBufferInfo::default().buffer(b.buf).range(vk::WHOLE_SIZE)];
        let w = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(spirv::binding_of(reg, n))
            .descriptor_type(ty)
            .buffer_info(&info);
        unsafe { vkd.device.update_descriptor_sets(&[w], &[]) };
    }

    /// Encoded bytes of one `mw x mh` level — what the caller budgets against
    /// `block_cap`, and the `dst_off` stride between levels in a batch.
    pub fn level_bytes(mw: u32, mh: u32) -> u64 {
        bc7::encoded_len(mw, mh) as u64
    }

    /// Record one level's encode: `mh` texel rows at `src_off` (tight
    /// `mw * 4` pitch) into `dst_off` of the block buffer.
    ///
    /// # Safety
    /// `cmd` must be recording, and `src_off .. + mh * mw * 4` must be inside
    /// the buffer `bind_src` named.
    pub unsafe fn record_encode(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        mw: u32,
        mh: u32,
        src_off: u64,
        dst_off: u64,
    ) {
        let bw = bc7::blocks(mw);
        let bh = bc7::blocks(mh);
        let consts: [u32; 9] = [
            mw,
            mh,
            mw * 4,
            bw,
            bh,
            bw, // tight block rows — no 256-byte pitch to honour here
            self.effort,
            src_off as u32,
            dst_off as u32,
        ];
        let mut bytes = [0u8; 36];
        for (i, x) in consts.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
        unsafe {
            // WAR: the previous dispatch must be done READING the constants.
            barrier(d, cmd);
            d.cmd_update_buffer(cmd, self.cb.buf, 0, &bytes);
            // RAW: and this one must SEE the write.
            barrier(d, cmd);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layouts.pipeline,
                0,
                &[self.set],
                &[],
            );
            d.cmd_dispatch(cmd, bw.div_ceil(8), bh.div_ceil(8), 1);
        }
    }

    /// Make every encode recorded so far visible to the copies that follow.
    ///
    /// # Safety
    /// `cmd` must be recording.
    pub unsafe fn record_flush(&self, d: &ash::Device, cmd: vk::CommandBuffer) {
        barrier(d, cmd);
    }

    pub fn block_buf(&self) -> &Buffer {
        &self.blocks
    }

    pub fn destroy(&self, vkd: &Vk) {
        unsafe {
            vkd.device.destroy_pipeline(self.pipeline, None);
            vkd.device.destroy_descriptor_pool(self.pool, None);
        }
        self.layouts.destroy(vkd);
        vkd.free_buffer(&self.cb);
        vkd.free_buffer(&self.blocks);
    }
}

/// The dispatch/transfer memory fence this pass needs, in both directions:
/// constants written by transfer and read by compute, blocks written by
/// compute and read by transfer.
fn barrier(d: &ash::Device, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::UNIFORM_READ
                | vk::AccessFlags::TRANSFER_READ
                | vk::AccessFlags::TRANSFER_WRITE,
        );
    unsafe {
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[mb],
            &[],
            &[],
        );
    }
}

/// The Vulkan format a compressed `t` uploads as — `bc7::dxgi_format`'s twin,
/// keyed on the same `Texture::srgb` role for the same reason (color maps
/// decode through the sRGB transfer in hardware; normal / rough-metal maps are
/// LINEAR data and must not).
pub fn vk_format(t: &crate::texture::Texture) -> vk::Format {
    if t.srgb { vk::Format::BC7_SRGB_BLOCK } else { vk::Format::BC7_UNORM_BLOCK }
}

// ---------------------------------------------------------------------------
// The gates. `--check-vk` V10, and the reason it exists at all: every other
// number in that suite passes identically whether the encoder produced BC7
// blocks or garbage of the right length. V6's radiance A/B is a 2% bar on a
// frame where textures are one term, and its texture anti-vacuity probe asks
// whether the table reached the shader, not whether its CONTENT survived —
// so a stride bug, a wrong window or a broken partition table all read green
// there while the picture is visibly wrong.

/// One `cs_bc7_read` pipeline plus its output buffer, reused across probes.
struct Dec {
    layouts: Layouts,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
    cb: Buffer,
    out: Buffer,
}

impl Dec {
    fn new(hg: &VkHeadless, sp: &Spirv, max_texels: u64) -> Result<Dec, String> {
        let vkd = &hg.vk;
        let words = sp.compile(
            crate::gfx::shaders::BC7_READ_HLSL,
            "cs_bc7_read",
            "cs_6_0",
            "bc7read",
            false,
        )?;
        let descs = reflect::reflect(&words)?;
        let mut map = Map::default();
        map.add("bc7read", &descs);
        let layouts = Layouts::build(vkd, &map, 1, None)?;
        let d = &vkd.device;
        let mut counts: std::collections::BTreeMap<vk::DescriptorType, u32> = Default::default();
        for e in map.entries.values() {
            *counts.entry(layout::desc_type(e.kind)).or_default() += e.count.max(1);
        }
        let sizes: Vec<vk::DescriptorPoolSize> = counts
            .iter()
            .map(|(&ty, &n)| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(n))
            .collect();
        let pool = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| format!("bc7 gate: vkCreateDescriptorPool: {e}"))?;
        let set = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts.sets),
            )
        }
        .map_err(|e| format!("bc7 gate: vkAllocateDescriptorSets: {e}"))?[0];
        let pipeline = layout::compute_pipeline(vkd, &layouts, &words, "cs_bc7_read")?;
        let cb = vkd.buffer(
            16,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            false,
        )?;
        let out = vkd.buffer(
            max_texels * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            false,
        )?;
        let dec = Dec { layouts, pool, set, pipeline, cb, out };
        let bi = [vk::DescriptorBufferInfo::default().buffer(dec.cb.buf).range(vk::WHOLE_SIZE)];
        let bo = [vk::DescriptorBufferInfo::default().buffer(dec.out.buf).range(vk::WHOLE_SIZE)];
        unsafe {
            d.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(dec.set)
                        .dst_binding(spirv::binding_of(spirv::Reg::B, 0))
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&bi),
                    vk::WriteDescriptorSet::default()
                        .dst_set(dec.set)
                        .dst_binding(spirv::binding_of(spirv::Reg::U, 0))
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&bo),
                ],
                &[],
            )
        };
        Ok(dec)
    }

    /// Upload `blocks` into a `BC7_UNORM_BLOCK` image and read the hardware
    /// decoder's RGBA8 back.
    ///
    /// UNORM, deliberately never `_SRGB` even for a color texture: the kernel
    /// must see raw code values, and a transfer function applied on the way
    /// out would be charged to the encoder.
    fn roundtrip(
        &self,
        hg: &VkHeadless,
        blocks: &[u8],
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>, String> {
        let vkd = &hg.vk;
        let d = &vkd.device;
        let img = crate::vk::textures::create_tex(vkd, w, h, 1, vk::Format::BC7_UNORM_BLOCK)?;
        let up = vkd.buffer(blocks.len().max(16) as u64, vk::BufferUsageFlags::TRANSFER_SRC, true)?;
        unsafe {
            let p = d
                .map_memory(up.mem, 0, up.size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("bc7 gate: vkMapMemory: {e}"))? as *mut u8;
            std::ptr::copy_nonoverlapping(blocks.as_ptr(), p, blocks.len());
            d.unmap_memory(up.mem);
        }
        let ii = [vk::DescriptorImageInfo::default()
            .image_view(img.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        unsafe {
            d.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(spirv::binding_of(spirv::Reg::T, 0))
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&ii)],
                &[],
            )
        };
        let mut cbb = [0u8; 8];
        cbb[0..4].copy_from_slice(&w.to_le_bytes());
        cbb[4..8].copy_from_slice(&h.to_le_bytes());
        let r = hg.run(|d, cmd| unsafe {
            let to_dst = vk::ImageMemoryBarrier::default()
                .image(img.img)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            let region = vk::BufferImageCopy::default()
                .buffer_row_length(bc7::blocks(w) * bc7::BLOCK)
                .buffer_image_height(bc7::blocks(h) * bc7::BLOCK)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            d.cmd_copy_buffer_to_image(
                cmd,
                up.buf,
                img.img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
            let to_read = vk::ImageMemoryBarrier::default()
                .image(img.img)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            d.cmd_update_buffer(cmd, self.cb.buf, 0, &cbb);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
            barrier(d, cmd);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layouts.pipeline,
                0,
                &[self.set],
                &[],
            );
            d.cmd_dispatch(cmd, w.div_ceil(8), h.div_ceil(8), 1);
            barrier(d, cmd);
        });
        vkd.free_buffer(&up);
        img.destroy(vkd);
        r?;
        hg.read_buffer(&self.out, (w * h) as usize * 4)
    }

    fn destroy(&self, vkd: &Vk) {
        unsafe {
            vkd.device.destroy_pipeline(self.pipeline, None);
            vkd.device.destroy_descriptor_pool(self.pool, None);
        }
        self.layouts.destroy(vkd);
        vkd.free_buffer(&self.cb);
        vkd.free_buffer(&self.out);
    }
}

/// Both gate probes encode at a NON-ZERO, and DIFFERENT, source and
/// destination window.
///
/// A gate that always passed 0 for both would score the kernel's arithmetic
/// and never its windowing — which is the one thing a second host added to it,
/// and therefore the one thing most likely to be wrong here. Different values
/// additionally separate the two: a kernel that read `dst_off` where it meant
/// `src_off` (or applied one offset to both) would sail through equal ones.
/// They are 16-byte multiples because that is what a BC destination's
/// `bufferOffset` and a `Store4` both want.
const GATE_SRC_OFF: u64 = 16;
const GATE_DST_OFF: u64 = 32;

/// Encode `texels` with the GPU kernel and read the raw blocks back.
///
/// The blocks come to the HOST rather than straight into an image so the
/// structural gate can compare them byte-for-byte — a stride bug is a
/// statement about the block STORE, and the decoded image is one inference
/// removed from it.
fn encode_host(
    hg: &VkHeadless,
    e: &VkBc7,
    src_ptr: *mut u8,
    texels: &[[u8; 4]],
    w: u32,
    h: u32,
) -> Result<Vec<u8>, String> {
    let bytes = texels.len() * 4;
    unsafe {
        std::ptr::copy_nonoverlapping(
            texels.as_ptr() as *const u8,
            src_ptr.add(GATE_SRC_OFF as usize),
            bytes,
        )
    };
    hg.run(|d, cmd| unsafe {
        e.record_encode(d, cmd, w, h, GATE_SRC_OFF, GATE_DST_OFF);
        e.record_flush(d, cmd);
    })?;
    let n = bc7::encoded_len(w, h);
    let all = hg.read_buffer(e.block_buf(), GATE_DST_OFF as usize + n)?;
    Ok(all[GATE_DST_OFF as usize..].to_vec())
}

/// The STRUCTURAL gate — synthetic textures, so it fires on every scene,
/// including the untextured procedural default where the fidelity half below
/// has nothing to score. `--check-gpu`'s `bc7-gpu` transplanted, teeth for
/// teeth.
pub fn structural(hg: &VkHeadless, sp: &Spirv) -> Result<String, String> {
    if !hg.vk.info.texture_compression_bc {
        return Ok("SKIP (no textureCompressionBC on this device)".into());
    }
    let vkd = &hg.vk;
    let e = VkBc7::new(
        hg,
        sp,
        GATE_DST_OFF + bc7::encoded_len(64, 64) as u64,
        bc7::Quality::Fast,
    )?;
    let dec = match Dec::new(hg, sp, 64 * 64) {
        Ok(d) => d,
        Err(err) => {
            e.destroy(vkd);
            return Err(err);
        }
    };
    let src = vkd.buffer(
        GATE_SRC_OFF + 64 * 64 * 4,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        true,
    )?;
    let ptr = unsafe { vkd.device.map_memory(src.mem, 0, src.size, vk::MemoryMapFlags::empty()) }
        .map_err(|err| format!("bc7 gate: vkMapMemory: {err}"))? as *mut u8;
    e.bind_src(vkd, &src);

    let r = (|| -> Result<String, String> {
        // Teeth 1 + 2: an all-even flat colour. Mode 6 represents it EXACTLY
        // via e0 == e1 (a P-bit is shared across an endpoint's channels, so
        // only agreeing parity is exact), which makes any loss here a wiring
        // bug rather than quantization; and every block of a flat texture
        // must be byte-identical to block 0, which is what catches a wrong
        // row stride.
        let flat_c = [200u8, 30, 30, 255];
        let flat: Vec<[u8; 4]> = vec![flat_c; 16 * 16];
        let blocks = encode_host(hg, &e, ptr, &flat, 16, 16)?;
        let b0 = &blocks[..bc7::BLOCK_BYTES];
        if b0.iter().all(|&b| b == 0) {
            return Err("flat block encoded to all zeros (kernel did not run?)".into());
        }
        for (i, b) in blocks.chunks_exact(bc7::BLOCK_BYTES).enumerate() {
            if b != b0 {
                return Err(format!(
                    "flat texture block {i} differs from block 0 (stride bug? the store must \
                     honour row_blocks)"
                ));
            }
        }
        let d16 = dec.roundtrip(hg, &blocks, 16, 16)?;
        for (px, d) in d16.chunks_exact(4).enumerate() {
            if d[0] != flat_c[0] || d[1] != flat_c[1] || d[2] != flat_c[2] {
                return Err(format!(
                    "all-even flat colour must round-trip BIT-EXACT; px {px} decoded \
                     ({},{},{}) want ({},{},{})",
                    d[0], d[1], d[2], flat_c[0], flat_c[1], flat_c[2]
                ));
            }
        }

        // Tooth 3: a gradient ramp at every effort tier. The chooser keeps
        // the lower SSE, so a smooth ramp must never get WORSE for trying the
        // two-subset mode.
        let ramp: Vec<[u8; 4]> = (0..64u32)
            .flat_map(|y| {
                (0..64u32)
                    .map(move |x| [(x * 4 + 1) as u8, (y * 4) as u8, (x * 2 + y * 2) as u8, 255])
            })
            .collect();
        let mut worst_psnr = f64::INFINITY;
        for q in [
            bc7::Quality::UltraFast,
            bc7::Quality::Fast,
            bc7::Quality::Basic,
            bc7::Quality::Slow,
        ] {
            let et = VkBc7::new(hg, sp, GATE_DST_OFF + bc7::encoded_len(64, 64) as u64, q)?;
            et.bind_src(vkd, &src);
            let bl = encode_host(hg, &et, ptr, &ramp, 64, 64);
            et.destroy(vkd);
            let bl = bl?;
            let d = dec.roundtrip(hg, &bl, 64, 64)?;
            let mut sq = 0f64;
            for (px, s) in ramp.iter().enumerate() {
                for c in 0..3 {
                    let e = d[px * 4 + c].abs_diff(s[c]) as f64;
                    sq += e * e;
                }
            }
            let mse = sq / (ramp.len() as f64 * 3.0);
            let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { 99.0 };
            worst_psnr = worst_psnr.min(psnr);
            if psnr < 30.0 {
                return Err(format!(
                    "ramp PSNR {psnr:.1} dB < 30 at effort {} (encoder math broken?)",
                    q.effort()
                ));
            }
        }

        // Tooth 4: a two-CLUSTER block — four colours no single line can fit.
        // Mode 6 alone leaves ~20-LSB errors, so a small max error proves the
        // mode-1 arm FIRED and that its partition/anchor tables and packing
        // agree with THIS hardware decoder (a wrong table decodes texels
        // against the wrong subset and the error explodes).
        let cl = |x: usize, y: usize| -> [u8; 4] {
            match (x < 2, y % 2 == 0) {
                (true, true) => [200, 0, 0, 255],
                (true, false) => [220, 20, 20, 255],
                (false, true) => [0, 0, 200, 255],
                (false, false) => [20, 20, 220, 255],
            }
        };
        let pair: Vec<[u8; 4]> =
            (0..4usize).flat_map(|y| (0..4usize).map(move |x| cl(x, y))).collect();
        let bl = encode_host(hg, &e, ptr, &pair, 4, 4)?;
        let d = dec.roundtrip(hg, &bl, 4, 4)?;
        let mut worst = 0u32;
        for (px, s) in pair.iter().enumerate() {
            for c in 0..3 {
                worst = worst.max(d[px * 4 + c].abs_diff(s[c]) as u32);
            }
        }
        if worst > 6 {
            return Err(format!(
                "two-cluster block max err {worst} LSB > 6 — the mode-1 arm did not fire (or its \
                 partition/anchor tables disagree with the decoder)"
            ));
        }
        Ok(format!(
            "flat bit-exact + one block shape, ramp PSNR >= {worst_psnr:.1} dB at all 4 efforts, \
             two-cluster max err {worst} LSB"
        ))
    })();

    unsafe { vkd.device.unmap_memory(src.mem) };
    vkd.free_buffer(&src);
    dec.destroy(vkd);
    e.destroy(vkd);
    r
}

pub struct Fidelity {
    pub textures: usize,
    /// Mean |decoded - source| per RGB channel sample, in 8-bit LSB.
    pub mean_abs: f64,
    pub max_abs: u32,
    /// Worst per-texture RGB PSNR, dB.
    pub worst_psnr: f64,
    pub worst_name: String,
}

/// The FIDELITY gate — M11's twin, on the scene's own textures.
///
/// Encodes every compressible texture's base level with the SESSION'S ARM (the
/// GPU one runs the session's own encoder, so there is no determinism bridge
/// to argue about; `--bc7-cpu` re-encodes through the deterministic ispc path
/// `bc7::self_test` pins), decodes it back through the hardware, and diffs
/// against the CPU RGBA8 the CPU tracer samples.
///
/// RGB only, for M11's reason: nothing ever samples a compressed texture's
/// alpha (the cutout path reads only the alpha-masked RGBA8 set), and "opaque"
/// means every alpha >= 250 — a 252 would quantize and read here as false loss.
///
/// `Ok(None)` = BC7 off, or nothing compressible.
pub fn fidelity(
    hg: &VkHeadless,
    sp: &Spirv,
    scene: &crate::scene::Scene,
    mode: bc7::Bc7Mode,
) -> Result<Option<Fidelity>, String> {
    let Some(q) = mode.quality() else {
        return Ok(None);
    };
    if !hg.vk.info.texture_compression_bc {
        return Ok(None);
    }
    let ids: Vec<usize> = (0..scene.textures.len())
        .filter(|&i| bc7::should_compress(&scene.textures[i]))
        .collect();
    if ids.is_empty() {
        return Ok(None);
    }
    let vkd = &hg.vk;
    let cpu_arm = matches!(mode, bc7::Bc7Mode::Cpu(_));
    let max_texels =
        ids.iter().map(|&i| scene.textures[i].w as u64 * scene.textures[i].h as u64).max().unwrap();
    let max_blocks = ids.iter().map(|&i| {
        let t = &scene.textures[i];
        bc7::encoded_len(t.w, t.h) as u64
    }).max().unwrap();

    let dec = Dec::new(hg, sp, max_texels)?;
    // In a GATE, encoder construction failure is a hard error — the session
    // path degrades loudly to RGBA8, but a gate that silently skips is the
    // shape this suite refuses everywhere else.
    let enc =
        if cpu_arm { None } else { Some(VkBc7::new(hg, sp, GATE_DST_OFF + max_blocks, q)?) };
    let src = if cpu_arm {
        None
    } else {
        Some(vkd.buffer(
            GATE_SRC_OFF + max_texels * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            true,
        )?)
    };
    let ptr = match &src {
        Some(b) => unsafe {
            vkd.device.map_memory(b.mem, 0, b.size, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| format!("bc7 gate: vkMapMemory: {e}"))? as *mut u8,
        None => std::ptr::null_mut(),
    };
    if let (Some(e), Some(b)) = (&enc, &src) {
        e.bind_src(vkd, b);
    }

    let r = (|| -> Result<Fidelity, String> {
        let mut sum = 0f64;
        let mut n = 0u64;
        let mut max_abs = 0u32;
        let mut worst_psnr = f64::INFINITY;
        let mut worst_name = String::new();
        for &i in &ids {
            let t = &scene.textures[i];
            let blocks = match &enc {
                Some(e) => encode_host(hg, e, ptr, &t.texels, t.w, t.h)?,
                None => bc7::encode_opaque(t, q),
            };
            let d = dec.roundtrip(hg, &blocks, t.w, t.h)?;
            let mut sq = 0f64;
            for (px, s) in t.texels.iter().enumerate() {
                for c in 0..3 {
                    let e = d[px * 4 + c].abs_diff(s[c]) as u32;
                    max_abs = max_abs.max(e);
                    sum += e as f64;
                    n += 1;
                    sq += (e as f64) * (e as f64);
                }
            }
            let mse = sq / (t.texels.len() as f64 * 3.0);
            let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { 99.0 };
            if psnr < worst_psnr {
                worst_psnr = psnr;
                worst_name = t.source.clone();
            }
        }
        Ok(Fidelity {
            textures: ids.len(),
            mean_abs: if n > 0 { sum / n as f64 } else { 0.0 },
            max_abs,
            worst_psnr,
            worst_name,
        })
    })();

    if let Some(b) = &src {
        unsafe { vkd.device.unmap_memory(b.mem) };
        vkd.free_buffer(b);
    }
    if let Some(e) = &enc {
        e.destroy(vkd);
    }
    dec.destroy(vkd);
    r.map(Some)
}
