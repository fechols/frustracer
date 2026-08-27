//! The display stage: `tonemap.hlsl` and `blit.hlsl` DRAWN, rather than
//! dispatched.
//!
//! Everything else in this backend is compute. Eighteen gate stages each end in
//! a number and `--cinematic` writes PNGs, but until this module nothing here
//! had ever created a graphics pipeline, and the shared tone curve reached a
//! Vulkan image only by way of `render::resolve_hdr` on the CPU. This is the
//! first rung of the display stage: the same two shaders D3D12 presents
//! through, compiled from the same `src/shaders/` corpus, rendering into an
//! offscreen colour attachment that a gate can read back.
//!
//! THE LAYOUT IS DERIVED, exactly as the tracer's is (`vk::reflect` ->
//! `vk::layout`), which is worth restating because the temptation is stronger
//! here: there are only four bindings and writing them down would be shorter
//! than reflecting them. It would also be a second statement of a register map
//! whose first statement is the `register(...)` declarations both backends
//! already compile, and Vulkan does not fail loudly when the two disagree —
//! M2c's planted binding shift PASSED unvalidated.
//!
//! ONE LAYOUT SERVES BOTH PIPELINES, and that is agreement with D3D12 rather
//! than thrift: `gpu/tonemap.rs` builds one root signature for blit, tonemap
//! and waveviz alike, and says why (`tone` "is passed unconditionally so the
//! two PSOs can share one root signature"). `blit.hlsl` declares only `t0`, so
//! its pipeline is created against a layout carrying three bindings it never
//! touches — legal, and legal for the same reason every tracer kernel is:
//! `descriptorBindingPartiallyBound`, a hard device requirement since M3a.
//!
//! WHAT THIS MODULE DELIBERATELY DOES NOT DO: present. A swapchain is rung 2;
//! this one renders into an image it owns, which is what makes it gateable on
//! a device with no surface support at all — and therefore on llvmpipe, and
//! therefore in CI.
//!
//! THE HUD COMPOSITE JOINED IT (B6b rung 4): `hud.hlsl` is the third pair in
//! the union, drawn with `layout::Blend::Premultiplied` as a second draw INSIDE
//! the same rendering instance, after the tonemap or blit — `gpu/mod.rs`'s
//! `fullscreen_to_backbuffer` shape, the ONE insertion point every present arm
//! passes through. The pipeline lives HERE rather than in `vk/hud.rs` for a
//! reason D3D12 does not have: `Passes` can be rebuilt (a format renegotiation
//! on resize), and a pipeline created against a destroyed layout is exactly
//! the silent class this backend refuses — keyed on `fmt` with tonemap/blit,
//! it dies and is reborn with them. A SECOND descriptor set (`hud_set`) points
//! `t0` at the HUD image while `set`'s `t0` stays the tonemap source; `b0` is
//! the same UBO in both, which is what makes `hud.hlsl`'s "mirrors tonemap's
//! Params" claim structural — one `set_params` serves both draws. `vk/hud.rs`
//! owns the image and the uploads; V21 is the gate.

use ash::vk;

use crate::vk::device::{Buffer, Vk};
use crate::vk::headless::VkHeadless;
use crate::vk::layout::{self, Layouts};
use crate::vk::reflect::Map;
use crate::vk::spirv::{binding_of, Reg, Spirv};

/// The ten DWORDs of `tonemap.hlsl`'s `cbuffer Params`, in the order
/// `gpu/tonemap.rs` writes them.
///
/// D3D12 pushes these as ROOT CONSTANTS; Vulkan has no equivalent for a `b`
/// register (DXC offers no flag to promote a cbuffer to push constants, and
/// `[[vk::push_constant]]` would be an HLSL edit), so they ride a uniform
/// buffer. The bytes are identical either way: `-fvk-use-dx-layout` puts the
/// members at DX offsets, and the one member that could straddle a 16-byte
/// boundary — `float2 bloom_texel` — is deliberately at offset 8, which the
/// shader's own comment records as the `fsr_composite` bug's lesson.
///
/// EVERY FIELD IS AN f32, `mode` included: it is written as the float literal
/// 1.0 or 2.0, never an integer. Naming the fields rather than filling ten
/// anonymous slots is what makes a misordering hard to write; V18 is what
/// catches one written anyway, and WHICH row catches it depends on where the
/// error lands, which is worth knowing before trusting a green run:
///
/// * A slide of the WHOLE block zeroes `inv_samples`, so the frame goes black
///   and all three rows fail (measured: worst 1.00e0 on both SDR wires).
/// * An error confined to the TAIL is the quiet one, and **hdr10 is the only
///   row that sees it** — `ToneParams::SDR` has `scale` and `mode` both at 1.0,
///   so swapping those two is literally undetectable on the SDR wires, while
///   hdr10 carries 0.02 and 2.0 and the swap sends it down the Gamma22 arm.
///   Measured: sdr 2.37e-3 and sdr10 9.41e-4, i.e. their EXACT unplanted
///   values, against hdr10 at 5.83e-1 — 233x over its limit.
///
/// So the hdr10 row is not a third wire for completeness; on this block it is
/// the only detector for half the failure modes. Do not drop it because the
/// SDR rows pass.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub inv_samples: f32,
    pub bloom_strength: f32,
    pub bloom_texel: [f32; 2],
    pub knee: f32,
    pub headroom: f32,
    pub scale: f32,
    pub mode: f32,
    pub exposure: f32,
    pub guard_strength: f32,
}

impl Params {
    /// Build the block for one `ToneParams`, the way `Passes::record` does on
    /// D3D12 — same source values, same order, same `mode` literals.
    ///
    /// `guard_strength` comes from the LEVER (`autoexp::guard_strength_live`)
    /// rather than from `ToneParams`, which is the D3D12 site's own choice and
    /// is copied deliberately: it is a display-stage process lever the funnel
    /// consults, exactly 0.0 when off, which is the shader's structural off arm.
    pub fn new(tone: crate::tone::ToneParams, inv_samples: f32, bloom: (f32, f32, f32)) -> Params {
        Params {
            inv_samples,
            bloom_strength: bloom.0,
            bloom_texel: [bloom.1, bloom.2],
            knee: tone.knee,
            headroom: tone.headroom,
            scale: tone.scale,
            // The mode literals are shared with tonemap.hlsl / hud.hlsl and
            // with `gpu/tonemap.rs`; all of them move together. 0 was
            // scRGB-linear, retired with the f16 swapchain.
            mode: match tone.mode {
                crate::tone::ToneMode::Gamma22 => 1.0,
                crate::tone::ToneMode::Pq => 2.0,
            },
            exposure: tone.exposure,
            guard_strength: crate::autoexp::guard_strength_live(),
        }
    }

    /// The wire form — ten contiguous f32, which is what the UBO holds.
    pub fn words(&self) -> [f32; 10] {
        [
            self.inv_samples,
            self.bloom_strength,
            self.bloom_texel[0],
            self.bloom_texel[1],
            self.knee,
            self.headroom,
            self.scale,
            self.mode,
            self.exposure,
            self.guard_strength,
        ]
    }
}

/// An image this stage owns — either the source it samples or the target it
/// draws into.
///
/// A FIFTH image helper in this backend, and the justification is that it is a
/// different RESOURCE CLASS rather than a fifth copy: the four private ones
/// (`tracer.rs`, `nrd.rs`, `textures.rs`, `fsr3.rs`) all create
/// `STORAGE | SAMPLED | TRANSFER_*` images, and a colour attachment needs
/// `COLOR_ATTACHMENT` usage and spends its life in a layout none of them ever
/// enter. Consolidating all five behind one parameterised helper is a real
/// follow-on; doing it here would be a refactor of four modules riding a slice
/// that is about drawing a triangle.
pub struct Image {
    pub img: vk::Image,
    pub view: vk::ImageView,
    mem: vk::DeviceMemory,
    pub w: u32,
    pub h: u32,
}

impl Image {
    pub fn new(
        vkd: &Vk,
        w: u32,
        h: u32,
        fmt: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Image, String> {
        let d = &vkd.device;
        let ici = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(fmt)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let img = unsafe { d.create_image(&ici, None) }
            .map_err(|e| format!("vkCreateImage: {e}"))?;
        let req = unsafe { d.get_image_memory_requirements(img) };
        let ti = crate::vk::device::mem_type_index(
            &vkd.mem,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| "no DEVICE_LOCAL memory type for a display image".to_string())?;
        let mem = unsafe {
            d.allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ti),
                None,
            )
        }
        .map_err(|e| format!("vkAllocateMemory(image): {e}"))?;
        unsafe { d.bind_image_memory(img, mem, 0) }
            .map_err(|e| format!("vkBindImageMemory: {e}"))?;
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
        .map_err(|e| format!("vkCreateImageView: {e}"))?;
        Ok(Image { img, view, mem, w, h })
    }

    pub fn destroy(&self, vkd: &Vk) {
        unsafe {
            vkd.device.destroy_image_view(self.view, None);
            vkd.device.destroy_image(self.img, None);
            vkd.device.free_memory(self.mem, None);
        }
    }
}

/// Upload `w*h*4` RGBA f16 texels and leave the image in `GENERAL`.
///
/// GENERAL rather than `SHADER_READ_ONLY_OPTIMAL` for this backend's standing
/// reason: everything rests in GENERAL, which is legal for both sampling and
/// storage, so the pass sequence needs memory barriers and never a layout
/// transition — and the descriptor written for it says GENERAL too, which is
/// the pairing B5a found broken elsewhere and fixed structurally.
pub fn upload_rgba16f(hg: &VkHeadless, img: &Image, rgb: &[f32]) -> Result<(), String> {
    let px = (img.w as usize) * (img.h as usize);
    if rgb.len() != px * 3 {
        return Err(format!("upload_rgba16f: got {} floats, want {}", rgb.len(), px * 3));
    }
    let mut bytes = Vec::with_capacity(px * 8);
    for i in 0..px {
        for c in 0..3 {
            bytes.extend_from_slice(&half::f16::from_f32(rgb[i * 3 + c]).to_bits().to_le_bytes());
        }
        bytes.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
    }
    let vkd = &hg.vk;
    let stage = vkd.buffer(bytes.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, true)?;
    let r = (|| -> Result<(), String> {
        vkd.write(&stage, &bytes)?;
        hg.run(|d, cmd| unsafe {
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(img.img)
                    .subresource_range(range)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
            );
            d.cmd_copy_buffer_to_image(
                cmd,
                stage.buf,
                img.img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D { width: img.w, height: img.h, depth: 1 })],
            );
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(img.img)
                    .subresource_range(range)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            );
        })
    })();
    vkd.free_buffer(&stage);
    r
}

/// Read a rendered target back as raw bytes, `bpp` per texel.
///
/// The image is left in `TRANSFER_SRC_OPTIMAL` by `record`, which is where the
/// copy wants it — unlike the tracer's planes, which rest in GENERAL and are
/// copied from there. A colour attachment cannot rest in GENERAL and still be
/// rendered into optimally, so this stage is the one place in the backend with
/// a real layout lifecycle.
pub fn read_target(hg: &VkHeadless, img: &Image, bpp: usize) -> Result<Vec<u8>, String> {
    let n = (img.w as usize) * (img.h as usize) * bpp;
    let vkd = &hg.vk;
    let stage = vkd.buffer(n as u64, vk::BufferUsageFlags::TRANSFER_DST, true)?;
    let r = (|| -> Result<Vec<u8>, String> {
        hg.run(|d, cmd| unsafe {
            d.cmd_copy_image_to_buffer(
                cmd,
                img.img,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                stage.buf,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D { width: img.w, height: img.h, depth: 1 })],
            );
        })?;
        vkd.read(&stage, n)
    })();
    vkd.free_buffer(&stage);
    r
}

/// The four fields a colour target must supply to `Passes::record_to`.
///
/// Private, and it exists for a documentation reason rather than an
/// abstraction one: rebinding `dst` to this inside `record_to` leaves that
/// function's body the TEXTUAL original, so "rung 2 did not change how V18
/// records" is checkable by reading the diff rather than by trusting a
/// re-typed body. An owned `Image` and a presentation-engine-owned swapchain
/// image both project onto it.
#[derive(Clone, Copy)]
struct Target {
    img: vk::Image,
    view: vk::ImageView,
    w: u32,
    h: u32,
}

/// The display pipelines, their one derived layout, and the descriptor sets
/// they draw against — `set` for tonemap/blit (t0 = the source), `hud_set`
/// for the overlay (t0 = the HUD image); same `b0`, same sampler.
pub struct Passes {
    pub map: Map,
    layouts: Layouts,
    pub tonemap: vk::Pipeline,
    pub blit: vk::Pipeline,
    /// The HUD composite — `hud.hlsl`, premultiplied blend. B6b rung 4.
    pub hud: vk::Pipeline,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    hud_set: vk::DescriptorSet,
    ubo: Buffer,
    pub fmt: vk::Format,
}

/// What the first draw of `Passes::record_frame` is — the tonemap, the blit,
/// or nothing at all (the loading page: a clear with the HUD composited over
/// it, before any source exists to tonemap).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Draw {
    Tonemap,
    Blit,
    None,
}

impl Passes {
    /// Compile both fullscreen pairs, union their declarations into one map,
    /// and build a pipeline apiece against the layout that falls out.
    pub fn new(hg: &VkHeadless, sp: &Spirv, fmt: vk::Format) -> Result<Passes, String> {
        let vkd = &hg.vk;
        let d = &vkd.device;

        // The spike guard's calibration constants ride INJECTED DEFINES, and
        // pasting them is not optional: `guard_defs`' own doc names the trap,
        // and `gpu/tonemap.rs` pastes the identical prefix. Skipping it here
        // would leave the `FR_AEXP_GUARD_*` sweep levers reaching one backend
        // and silently not the other — which is the probe-reach class this
        // codebase has shipped more than once.
        //
        // (`--check-spirv` deliberately compiles this file BARE, with the
        // `#ifndef` fallbacks standing in. That difference is correct: that
        // gate is about whether the source compiles, not about which constants
        // a session injects.)
        let tm_src = format!("{}{}", crate::autoexp::guard_defs(), crate::gfx::shaders::TONEMAP_HLSL);

        let mut map = Map::default();
        let mut compile = |src: &str, entry: &str, target: &str, tag: &str| -> Result<Vec<u32>, String> {
            let w = sp.compile(src, entry, target, tag, false)?;
            let descs = crate::vk::reflect::reflect(&w)?;
            let conflicts = map.add(tag, &descs);
            if !conflicts.is_empty() {
                return Err(conflicts.join("; "));
            }
            Ok(w)
        };
        let vs_e = crate::gfx::shaders::GFX_VS;
        let ps_e = crate::gfx::shaders::GFX_PS;
        let tm_vs = compile(&tm_src, vs_e, "vs_6_0", "tonemap-vs")?;
        let tm_ps = compile(&tm_src, ps_e, "ps_6_0", "tonemap-ps")?;
        let bl_vs = compile(crate::gfx::shaders::BLIT_HLSL, vs_e, "vs_6_0", "blit-vs")?;
        let bl_ps = compile(crate::gfx::shaders::BLIT_HLSL, ps_e, "ps_6_0", "blit-ps")?;
        // The HUD pair joins the SAME union: `hud.hlsl` declares t0 and b0 with
        // the names and kinds tonemap.hlsl already put there, so `Map::add`
        // sees no conflict and `Layouts::build` sees nothing new — the shader's
        // own "reuses the tonemap root signature" header, made a fact here.
        let hd_vs = compile(crate::gfx::shaders::HUD_HLSL, vs_e, "vs_6_0", "hud-vs")?;
        let hd_ps = compile(crate::gfx::shaders::HUD_HLSL, ps_e, "ps_6_0", "hud-ps")?;

        // No unbounded array in this corpus, so the cap is inert — passed as 1
        // rather than the device ceiling so a stray unbounded binding would be
        // a loud failure rather than a 4096-descriptor pool nobody asked for.
        let layouts = Layouts::build(vkd, &map, 1, None)?;

        let mk = |vs: &[u32], ps: &[u32]| {
            layout::graphics_pipeline(vkd, &layouts, vs, vs_e, ps, ps_e, fmt)
        };
        let tonemap = match mk(&tm_vs, &tm_ps) {
            Ok(p) => p,
            Err(e) => {
                layouts.destroy(vkd);
                return Err(e);
            }
        };
        let blit = match mk(&bl_vs, &bl_ps) {
            Ok(p) => p,
            Err(e) => {
                unsafe { d.destroy_pipeline(tonemap, None) };
                layouts.destroy(vkd);
                return Err(e);
            }
        };
        let hud = match layout::graphics_pipeline_blend(
            vkd,
            &layouts,
            &hd_vs,
            vs_e,
            &hd_ps,
            ps_e,
            fmt,
            layout::Blend::Premultiplied,
        ) {
            Ok(p) => p,
            Err(e) => {
                unsafe {
                    d.destroy_pipeline(tonemap, None);
                    d.destroy_pipeline(blit, None);
                }
                layouts.destroy(vkd);
                return Err(e);
            }
        };

        // OUR OWN SAMPLER, and the reason is one word in the shader:
        // `tonemap.hlsl:20` says "linear, clamp", while the tracer's static
        // sampler is REPEAT. Borrowing that one would wrap the glare tap at
        // the frame edge — a small, plausible, entirely wrong halo.
        let sampler = unsafe {
            d.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        }
        .map_err(|e| format!("vkCreateSampler: {e}"))?;

        let ubo = vkd.buffer(
            std::mem::size_of::<[f32; 10]>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            true,
        )?;

        // Sized from the map, like the tracer's — TWO sets of the one layout
        // (the tonemap/blit set and the HUD set), so every count doubles.
        const SETS: u32 = 2;
        let mut counts: std::collections::BTreeMap<vk::DescriptorType, u32> = Default::default();
        for e in map.entries.values() {
            let n = if e.count == 0 { 1 } else { e.count };
            *counts.entry(layout::desc_type(e.kind)).or_default() += n * SETS;
        }
        let sizes: Vec<vk::DescriptorPoolSize> = counts
            .iter()
            .map(|(&ty, &n)| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(n))
            .collect();
        let pool = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(SETS).pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| format!("vkCreateDescriptorPool: {e}"))?;
        let set_layouts: Vec<vk::DescriptorSetLayout> =
            std::iter::repeat_n(layouts.sets[0], SETS as usize).collect();
        let sets = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(|e| format!("vkAllocateDescriptorSets: {e}"))?;
        let (set, hud_set) = (sets[0], sets[1]);

        let p = Passes { map, layouts, tonemap, blit, hud, sampler, pool, set, hud_set, ubo, fmt };
        p.wire_static(vkd);
        Ok(p)
    }

    /// The bindings that never change: the sampler and the constant buffer —
    /// into BOTH sets, so one `set_params` reaches the tonemap and the HUD
    /// draws alike (the `b0` both shaders declare is one buffer).
    fn wire_static(&self, vkd: &Vk) {
        let bi = [vk::DescriptorBufferInfo::default()
            .buffer(self.ubo.buf)
            .range(vk::WHOLE_SIZE)];
        let si = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
        let mut writes = Vec::new();
        for set in [self.set, self.hud_set] {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding_of(Reg::B, 0))
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&bi),
            );
            // Guarded on the map, the `--sw-rays`/TLAS rule: a write to a
            // binding the layout does not carry is not a no-op. `blit.hlsl`
            // alone declares no sampler, so this is only unconditional while
            // tonemap is in the union — which it is, and the guard is what
            // keeps that a fact rather than an assumption.
            if self.map.entries.contains_key(&(0, binding_of(Reg::S, 0))) {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding_of(Reg::S, 0))
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&si),
                );
            }
        }
        unsafe { vkd.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Point the HUD set's `t0` at the overlay image (`vk::hud::HudVk`'s).
    ///
    /// Called once per image — at build and again after every resize, since a
    /// resize recreates the image AND may recreate `Passes` (a fresh one has an
    /// unwritten `hud_set`). The image rests in `GENERAL` for the backend's
    /// standing reason (`upload_rgba16f`'s doc), and the descriptor says so.
    pub fn bind_overlay(&self, vkd: &Vk, view: vk::ImageView) {
        let ii = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(self.hud_set)
            .dst_binding(binding_of(Reg::T, 0))
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&ii)];
        unsafe { vkd.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Point `src` (t0) and `glare` (t1) at one view.
    ///
    /// BOTH AT THE SAME IMAGE, which is what `gpu::tonemap::selftest` does
    /// (`:123-128`) rather than inventing a 1x1 dummy. Two reasons, and the
    /// second is the one that matters: `bloom_strength = 0` makes the glare tap
    /// structurally dead so the value is irrelevant, and binding the real
    /// source keeps M12b's pre-glare arm — which distinguishes a shader
    /// sampling post-glare colour by feeding the source as its own glare —
    /// transplantable later without changing this wiring.
    pub fn bind_source(&self, vkd: &Vk, view: vk::ImageView) {
        let ii = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let mut writes = Vec::new();
        for reg in [0u32, 1] {
            let b = binding_of(Reg::T, reg);
            if self.map.entries.contains_key(&(0, b)) {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.set)
                        .dst_binding(b)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&ii),
                );
            }
        }
        unsafe { vkd.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Write the constant block. Host-visible and persistently mapped by
    /// `Vk::write`, so this is a plain memcpy — legal because every caller
    /// writes it before the submit that reads it, and `VkHeadless::run` fences
    /// each submit (the same contract `vk::nrd`'s per-dispatch offsets rest on).
    pub fn set_params(&self, vkd: &Vk, p: Params) -> Result<(), String> {
        let mut bytes = Vec::with_capacity(40);
        for v in p.words() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        vkd.write(&self.ubo, &bytes)
    }

    /// Record: transition the target in, draw three vertices, transition it out
    /// to `TRANSFER_SRC_OPTIMAL` so a readback needs no second barrier.
    ///
    /// `tonemap` picks the pipeline the way `fullscreen_to_backbuffer` does —
    /// one bool, two PSOs, one layout — so the blit arm is reachable with no
    /// second descriptor set and no second recording path.
    pub fn record(&self, d: &ash::Device, cmd: vk::CommandBuffer, dst: &Image, tonemap: bool) {
        self.record_to(d, cmd, dst.img, dst.view, dst.w, dst.h, tonemap);
    }

    /// `record` over the four fields it actually uses, so a target this module
    /// does not OWN can be one.
    ///
    /// The split exists for exactly one caller — the swapchain, whose images
    /// are created and owned by the presentation engine and therefore have no
    /// `VkDeviceMemory` at all. `Image::mem` is private and non-`Option`, so
    /// the alternative was teaching `Image` to be sometimes-unowned: a
    /// nullable field on the type every other display path uses, to serve the
    /// one path that never allocates. Borrowing four fields is smaller and
    /// keeps `Image`'s invariant ("this handle owns that allocation") intact.
    ///
    /// `record` above stays as the thin wrapper so V18's recording is
    /// TEXTUALLY unchanged rather than merely equivalent.
    ///
    /// THE TWO TRANSITIONS ARE ALREADY RIGHT FOR A SWAPCHAIN IMAGE and neither
    /// is parameterised, which is worth stating because it looks like an
    /// oversight: the opening `UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL` is
    /// correct for a freshly acquired image (its contents are undefined by
    /// contract, and we clear), and the closing
    /// `COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL` is exactly where a
    /// copy-before-present wants it. The present path appends ONE further
    /// barrier to `PRESENT_SRC_KHR` after its copy; it does not edit these.
    #[allow(clippy::too_many_arguments)]
    pub fn record_to(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        img: vk::Image,
        view: vk::ImageView,
        w: u32,
        h: u32,
        tonemap: bool,
    ) {
        self.record_frame(d, cmd, img, view, w, h, if tonemap { Draw::Tonemap } else { Draw::Blit }, false);
    }

    /// `record_to` with the first draw CHOSEN and the HUD composite OPTIONAL —
    /// the whole frame's recording, B6b rung 4.
    ///
    /// `record_to` above stays the thin wrapper (`tonemap: bool`, no overlay)
    /// so V18/V19/V20's recording is TEXTUALLY the rung-1 body — the same
    /// trick `record` pulled on `record_to` when the swapchain arrived.
    ///
    /// THE OVERLAY IS A SECOND DRAW INSIDE THE SAME RENDERING INSTANCE, after
    /// the first, with no barrier between: the premultiplied blend reads the
    /// attachment the first draw wrote, and rasterization order within a
    /// subpass is what the spec guarantees for exactly that. It binds `hud_set`
    /// (t0 = the HUD image) against the SAME pipeline layout, viewport and
    /// scissor already set. `Draw::None` is the loading page — clear, then
    /// overlay — and binds nothing the tonemap would need, so the tonemap set's
    /// unwritten `t0` (no source exists yet) is never consulted.
    ///
    /// `overlay` is the caller's `visible && drawable` conjunction: a HUD that
    /// is hidden, or one whose image has never been uploaded, records NOTHING
    /// here — the structural off-state, which V21's hidden/unstaged frames pin
    /// as byte identity with a tonemap-only render.
    #[allow(clippy::too_many_arguments)]
    pub fn record_frame(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        img: vk::Image,
        view: vk::ImageView,
        w: u32,
        h: u32,
        draw: Draw,
        overlay: bool,
    ) {
        let dst = Target { img, view, w, h };
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(dst.img)
                    .subresource_range(range)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)],
            );

            let att = [vk::RenderingAttachmentInfo::default()
                .image_view(dst.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                // CLEAR rather than LOAD: the image arrives UNDEFINED, so
                // loading it would be reading uninitialised memory — and the
                // clear colour is what a gate's "did the draw cover anything"
                // question is asked against.
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
                })];
            let ri = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: dst.w, height: dst.h },
                })
                .layer_count(1)
                .color_attachments(&att);
            d.cmd_begin_rendering(cmd, &ri);
            if draw != Draw::None {
                d.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    if draw == Draw::Tonemap { self.tonemap } else { self.blit },
                );
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layouts.pipeline,
                    0,
                    &[self.set],
                    &[],
                );
            }
            d.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: dst.w as f32,
                    height: dst.h as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            d.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: dst.w, height: dst.h },
                }],
            );
            if draw != Draw::None {
                d.cmd_draw(cmd, 3, 1, 0, 0);
            }
            if overlay {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.hud);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layouts.pipeline,
                    0,
                    &[self.hud_set],
                    &[],
                );
                d.cmd_draw(cmd, 3, 1, 0, 0);
            }
            d.cmd_end_rendering(cmd);

            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(dst.img)
                    .subresource_range(range)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
            );
        }
    }

    pub fn destroy(&self, vkd: &Vk) {
        unsafe {
            vkd.device.destroy_pipeline(self.tonemap, None);
            vkd.device.destroy_pipeline(self.blit, None);
            vkd.device.destroy_pipeline(self.hud, None);
            vkd.device.destroy_sampler(self.sampler, None);
            vkd.device.destroy_descriptor_pool(self.pool, None);
        }
        vkd.free_buffer(&self.ubo);
        self.layouts.destroy(vkd);
    }
}

/// The byte offsets of R, G and B in one texel of an 8-bit RGBA-family format,
/// or `None` for a format whose order this module does not know.
///
/// ONE STATEMENT OF BYTE ORDER, consulted by `decode` AND by the present
/// path's byte identity, and it exists because the alternative shipped a
/// latent defect for as long as rung 1 did. `decode`'s catch-all arm read
/// `px[2], px[1], px[0]` — BGRA — which is right for the ONE 8-bit format
/// rung 1 ever renders and silently wrong for the other. The headless
/// surface's own preferred format, measured on BOTH ICDs on this box, is
/// `R8G8B8A8_UNORM`: it is `formats[0]`, i.e. exactly what a present path that
/// took the driver's first choice would negotiate. Inheriting the catch-all
/// there would have swapped R and B at every texel where they differ —
/// **255 in 256** of a hashed pattern — while every other assertion in the
/// gate stayed green, because a swizzle preserves alpha, preserves coverage,
/// and preserves a byte-for-byte comparison of two images that were BOTH
/// swizzled.
///
/// Returning `Option` rather than guessing is the whole point: a format this
/// table has never seen gets no plausible answer, and the present path's
/// format selection REFUSES it up front rather than decoding it wrongly.
pub fn rgb_offsets(fmt: vk::Format) -> Option<[usize; 3]> {
    match fmt {
        // B,G,R,A in memory — D3D12's `B8G8R8A8_UNORM` byte for byte, which is
        // what lets `gpu::tonemap::selftest`'s oracle transplant unchanged.
        vk::Format::B8G8R8A8_UNORM => Some([2, 1, 0]),
        // R,G,B,A in memory. The name says so and it is still worth pinning
        // here, because this is the one the surface hands us by default.
        vk::Format::R8G8B8A8_UNORM => Some([0, 1, 2]),
        _ => None,
    }
}

/// Decode one texel of a presented image back to linear-ish [0,1] RGB.
///
/// The formats and their unpacking are `gpu::tonemap::selftest`'s
/// (`:194-208`), and the 10-bit one carries over UNCHANGED because the two
/// APIs agree about it: Vulkan's `A2B10G10R10_UNORM_PACK32` puts R in the low
/// ten bits exactly as D3D12's `R10G10B10A2_UNORM` does. Worth stating because
/// the names suggest the opposite and agreement is what a transplanted oracle
/// depends on.
///
/// The 8-bit arm goes through `rgb_offsets` rather than assuming BGRA — see
/// that function for the defect this closes.
pub fn decode(px: &[u8], fmt: vk::Format) -> [f32; 3] {
    match fmt {
        vk::Format::A2B10G10R10_UNORM_PACK32 => {
            let v = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
            [
                (v & 0x3ff) as f32 / 1023.0,
                ((v >> 10) & 0x3ff) as f32 / 1023.0,
                ((v >> 20) & 0x3ff) as f32 / 1023.0,
            ]
        }
        _ => match rgb_offsets(fmt) {
            Some(o) => [
                px[o[0]] as f32 / 255.0,
                px[o[1]] as f32 / 255.0,
                px[o[2]] as f32 / 255.0,
            ],
            // A format the table does not know. Returning a PLAUSIBLE guess is
            // precisely the failure `rgb_offsets` exists to prevent, so return
            // something that cannot be mistaken for a colour and that fails
            // every comparison it reaches: NaN propagates, and `!(a <= b)` is
            // true for it, so a tolerance check reports the failure rather
            // than swallowing it. Unreachable in practice — the present path
            // refuses such a format at negotiation — which is why this is a
            // sentinel and not an error path.
            None => [f32::NAN; 3],
        },
    }
}

/// The alpha channel of one texel, [0,1].
///
/// A FREE COVERAGE WITNESS, and the only one this stage has: every display
/// pixel shader writes alpha 1.0 unconditionally (`blit.hlsl` forces it,
/// `tonemap.hlsl` returns it), while the M12 oracle this gate transplants
/// compares RGB only. So "alpha is at maximum everywhere" answers a question
/// no colour comparison can — did the draw actually cover this pixel — and it
/// costs nothing, the bytes already being read back.
///
/// Deliberately NOT routed through `rgb_offsets`: alpha sits at byte 3 in both
/// 8-bit orders that function distinguishes, so the catch-all here is correct
/// for exactly the formats the present path admits, and a lookup would suggest
/// a variation that does not exist. The RGB triple is where the orders differ
/// and that is where the table is consulted.
pub fn decode_alpha(px: &[u8], fmt: vk::Format) -> f32 {
    match fmt {
        vk::Format::A2B10G10R10_UNORM_PACK32 => {
            let v = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
            ((v >> 30) & 0x3) as f32 / 3.0
        }
        _ => px[3] as f32 / 255.0,
    }
}
