//! The reference tracer on Vulkan: `cs_reference` rendering into `accum`, and
//! `cs_resolve` turning that into an RGBA16F image.
//!
//! This is the smallest thing that can be WRONG in an interesting way. The
//! wavefront ladder (M3c) adds queues and indirect dispatch on top, but every
//! one of those kernels reads the same streams through the same layout and
//! shades through the same `shade.hlsli` — so if a stream is bound at the
//! wrong slot, a material stride is skewed, the TLAS is built from the wrong
//! addresses, or the cbuffer packs differently under `-fvk-use-dx-layout`,
//! this is where it shows, as a picture that disagrees with the CPU.
//!
//! WHY THE CLOUD CACHES ARE BUILT AND DISPATCHED HERE. `cs_reference` reads
//! the amortized sky lattice (`--sky-lod`, default 4) and the slab-space
//! cloud-shadow cache (`--cloud-shadow`, default 16) — both at registers the
//! wavefront otherwise uses for tile queues — so a tracer that skipped their
//! fills would read whatever those buffers happen to contain and shade a black
//! sky. The alternative (forcing both levers off for the gate) would make the
//! gate cover a configuration nobody ships. So the fills run, exactly as
//! `record_cloud_shadow`/`record_sky_lod` run them on D3D12, and the caches
//! are covered for free.

use ash::vk;

use crate::gfx::frame::{FrameCb, FrameParams, CB_STRIDE};
use crate::gfx::shaders as gs;
use crate::scene::Scene;
use crate::vk::device::Buffer;
use crate::vk::headless::VkHeadless;
use crate::vk::layout::{self, Layouts};
use crate::vk::reflect::{DescKind, Map};
use crate::vk::scene::VkScene;
use crate::vk::spirv::{binding_of, Reg, Spirv};

/// One GPU image, plus what it takes to bind and read it.
struct Image {
    img: vk::Image,
    view: vk::ImageView,
    mem: vk::DeviceMemory,
}

pub struct VkTracer {
    pub rw: u32,
    pub rh: u32,
    pub accum: Buffer,
    pub tbuf: Buffer,
    pub info: Buffer,
    counters: Buffer,
    cloud_lod: Buffer,
    cloud_shadow: Buffer,
    frame_cb: Buffer,
    push: Buffer,
    hdr: Image,
    samp_lin: vk::Sampler,
    samp_aniso: vk::Sampler,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    layouts: Layouts,
    pipes: [vk::Pipeline; 4], // reference, resolve, sky_lod, cloud_shadow
    cb_base: FrameCb,
    scene_aabb: ([f32; 3], [f32; 3]),
    sky_lod_k: u32,
    cloud_shadow_n: u32,
    /// The derived register map, kept so `bind()` writes exactly the slots the
    /// modules declared — never a hand-listed set.
    map: Map,
}

const P_REFERENCE: usize = 0;
const P_RESOLVE: usize = 1;
const P_SKY_LOD: usize = 2;
const P_CLOUD_SHADOW: usize = 3;

impl VkTracer {
    pub fn new(
        hg: &VkHeadless,
        sp: &Spirv,
        scene: &Scene,
        vs: &VkScene,
        rw: u32,
        rh: u32,
    ) -> Result<VkTracer, String> {
        let vkd = &hg.vk;
        let d = &vkd.device;

        // ONE assembly for both units, from the shipping entry point — the
        // snapshots come back with it, so the buffers below are SIZED against
        // exactly the constants the kernels were COMPILED against (the
        // `TraceSources` contract; a desync here is the documented
        // device-hang class).
        let srcs = gs::trace_sources(&gs::TraceKeys {
            scene,
            // THIS device's vendor, a fact rather than a preference: on AMD
            // `cand_defs` arms the candidate-loop TMin workaround, and arming
            // it on the wrong device restores the defect it exists to fix.
            vendor: vkd.info.vendor(),
            sway_armed: false,
        });
        let units: [(&str, &str, &str); 4] = [
            (&srcs.reference, "cs_reference", "reference"),
            (&srcs.resolve, "cs_resolve", "resolve"),
            (&srcs.sky, "cs_sky_lod", "sky-lod"),
            (&srcs.sky, "cs_cloud_shadow", "cloud-shadow"),
        ];

        // Compile first, reflect the compiled words, THEN build the layout —
        // the M3a order, and the reason there is no register table in this
        // file. Deduped by source so the two sky entries compile once.
        let mut words: Vec<Vec<u32>> = Vec::new();
        let mut map = Map::default();
        for (src, entry, tag) in units.iter() {
            let w = sp.compile(src, entry, "cs_6_5", tag, false)?;
            let descs = crate::vk::reflect::reflect(&w)?;
            let conflicts = map.add(tag, &descs);
            if !conflicts.is_empty() {
                return Err(conflicts.join("; "));
            }
            words.push(w);
        }

        // `texs[]` is sized to the scene, not to the device ceiling: this is a
        // SESSION layout, and M3a's own finding was that the map is a function
        // of the modules a session compiled. A textureless scene still needs a
        // count of at least 1 — a zero-length binding is illegal.
        let tex_cap = (scene.textures.len() as u32).max(1);
        let layouts = Layouts::build(vkd, &map, tex_cap, None)?;

        let mut pipes = [vk::Pipeline::null(); 4];
        for (i, (_, entry, _)) in units.iter().enumerate() {
            match layout::compute_pipeline(vkd, &layouts, &words[i], entry) {
                Ok(p) => pipes[i] = p,
                Err(e) => {
                    for p in pipes.iter().filter(|p| **p != vk::Pipeline::null()) {
                        unsafe { d.destroy_pipeline(*p, None) };
                    }
                    layouts.destroy(vkd);
                    return Err(format!("{entry}: {e}"));
                }
            }
        }

        let sb = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let ub = vk::BufferUsageFlags::UNIFORM_BUFFER;
        let px = (rw as u64) * (rh as u64);
        let accum = vkd.buffer(px * 12, sb, false)?;
        let tbuf = vkd.buffer(px * 4, sb, false)?;
        let info = vkd.buffer(px * 4, sb, false)?;
        let counters = vkd.buffer(u64::from(gs::CTR_TOTAL) * 4, sb, false)?;
        // The amortized cloud lattice: one float4 per point, one point of
        // border past each far edge — `TraceGpu::new`'s sizing verbatim,
        // against the SNAPSHOT k.
        let shift = srcs.sky_lod.trailing_zeros();
        let lw = (rw >> shift) as u64 + 2;
        let lh = (rh >> shift) as u64 + 2;
        let cloud_lod = vkd.buffer((lw * lh).max(1) * 16, sb, false)?;
        // Sized at the CAP: the live side is derived per frame from the sun's
        // footprint, so the allocation cannot track it.
        let csn = if srcs.cloud_shadow_n > 0 {
            u64::from(crate::clouds::CLOUD_SHADOW_MAX)
        } else {
            1
        };
        let cloud_shadow = vkd.buffer(csn * csn * 4, sb, false)?;
        let frame_cb = vkd.buffer(CB_STRIDE as u64, ub, true)?;
        let push = vkd.buffer(16, ub, true)?;

        let hdr = create_image(vkd, rw, rh)?;
        let samp = |aniso: f32| -> Result<vk::Sampler, String> {
            let ci = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(aniso > 1.0)
                .max_anisotropy(aniso)
                .max_lod(vk::LOD_CLAMP_NONE);
            unsafe { d.create_sampler(&ci, None) }
                .map_err(|e| format!("vkCreateSampler: {e}"))
        };
        let samp_lin = samp(1.0)?;
        let samp_aniso = samp(1.0)?;

        // The pool is sized FROM THE MAP, like everything else here.
        let mut counts: std::collections::BTreeMap<vk::DescriptorType, u32> = Default::default();
        for e in map.entries.values() {
            let n = if e.count == 0 { tex_cap } else { e.count };
            *counts.entry(layout::desc_type(e.kind)).or_default() += n;
        }
        let sizes: Vec<vk::DescriptorPoolSize> = counts
            .iter()
            .map(|(&ty, &n)| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(n))
            .collect();
        let pool = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(layouts.sets.len() as u32)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| format!("vkCreateDescriptorPool: {e}"))?;
        let sets = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts.sets),
            )
        }
        .map_err(|e| format!("vkAllocateDescriptorSets: {e}"))?;

        let t = VkTracer {
            rw,
            rh,
            accum,
            tbuf,
            info,
            counters,
            cloud_lod,
            cloud_shadow,
            frame_cb,
            push,
            hdr,
            samp_lin,
            samp_aniso,
            pool,
            sets,
            layouts,
            pipes,
            cb_base: FrameCb::base(scene, rw, rh),
            scene_aabb: crate::gfx::scene::shadow_aabb(scene),
            sky_lod_k: srcs.sky_lod,
            cloud_shadow_n: srcs.cloud_shadow_n,
            map,
        };
        t.write_descriptors(hg, vs);
        Ok(t)
    }

    /// Write EVERY slot the map contains, from a table keyed by `(set, reg)`.
    ///
    /// Slots with no real resource take the 16-byte dummy rather than going
    /// unwritten: `PARTIALLY_BOUND` makes unwritten legal for descriptors no
    /// dispatch touches, but "no dispatch touches it" is a claim about every
    /// kernel this layout will ever serve, and a bound zero buffer costs
    /// nothing to be sure with. Storage IMAGES are the exception — an image
    /// cannot be stood in for by a buffer, so the feed targets (which the
    /// reference and resolve units do not declare at all) stay unwritten and
    /// ride the flag.
    fn write_descriptors(&self, hg: &VkHeadless, vs: &VkScene) {
        let d = &hg.vk.device;
        let b = |buf: &Buffer| [vk::DescriptorBufferInfo::default().buffer(buf.buf).range(vk::WHOLE_SIZE)];

        // (set, register) -> buffer. Named by REGISTER, which is how the
        // shaders name them, with `binding_of` doing the translation — the
        // never-a-literal rule.
        // TEETH. A layout DERIVED from the shaders cannot be tested by
        // writing a wrong one, and a bound-stream table cannot be tested by
        // reading it — so this omits one stream by NAME and binds the zero
        // dummy in its place, which is exactly the shape of the bug this
        // stage caught on its first run (`blas_tri` on the dummy shaded the
        // whole frame as triangle 0, and the visibility gate saw nothing).
        // The teeth are the radiance A/B's, not a claim about this file.
        let drop = std::env::var("FR_VK_DROP_STREAM").ok();
        if let Some(name) = &drop {
            eprintln!(
                "check-vk: FR_VK_DROP_STREAM={name} — that stream is bound to the ZERO \
                 dummy; this run MUST fail"
            );
        }
        let bufs: Vec<(u32, Reg, u32, &Buffer)> = vec![
            (0, Reg::B, 0, &self.frame_cb),
            (0, Reg::B, 1, &self.push),
            (0, Reg::T, 2, &vs.positions),
            (0, Reg::T, 3, &vs.normals),
            (0, Reg::T, 4, &vs.indices),
            (0, Reg::T, 5, &vs.tri_mat),
            (0, Reg::T, 6, &vs.materials),
            (0, Reg::U, 0, &self.accum),
            (0, Reg::U, 1, &self.tbuf),
            (0, Reg::U, 2, &self.info),
            (0, Reg::U, 3, &self.counters),
            (0, Reg::U, 5, &self.cloud_lod),
            (0, Reg::U, 6, &self.cloud_shadow),
            (1, Reg::T, 0, &vs.uv_buf),
            (1, Reg::T, 1, &vs.indices),
            (1, Reg::T, 2, &vs.tri_mat),
            (1, Reg::T, 3, &vs.mat_cutout),
            (1, Reg::T, 4, &vs.positions),
            (1, Reg::T, 5, &vs.mat_height),
            (1, Reg::T, 6, &vs.mat_shadow),
            // `--blas-split` is the DEFAULT, so these are load-bearing, not
            // spare: `tri_of` indexes every stream through them.
            (1, Reg::T, 7, &vs.blas_tri),
            (1, Reg::T, 8, &vs.chunk_base),
        ];
        let mut infos: Vec<[vk::DescriptorBufferInfo; 1]> = Vec::new();
        let mut plan: Vec<(u32, u32, vk::DescriptorType)> = Vec::new();
        for (&(set, binding), e) in &self.map.entries {
            let ty = layout::desc_type(e.kind);
            match e.kind {
                DescKind::UniformBuffer | DescKind::StorageBuffer => {
                    let dropped = drop.as_deref().is_some_and(|d| e.names.iter().any(|n| n == d));
                    let hit = if dropped {
                        &vs.dummy
                    } else {
                        bufs.iter()
                            .find(|&&(s, r, n, _)| s == set && binding_of(r, n) == binding)
                            .map(|&(_, _, _, buf)| buf)
                            .unwrap_or(&vs.dummy)
                    };
                    if std::env::var_os("FR_VK_MAP").is_some() {
                        let real = !std::ptr::eq(hit, &vs.dummy);
                        eprintln!(
                            "check-vk:   bind set {set} binding {binding} <- {} ({})",
                            if real { "REAL" } else { "dummy" },
                            e.names.iter().cloned().collect::<Vec<_>>().join("/")
                        );
                    }
                    infos.push(b(hit));
                    plan.push((set, binding, ty));
                }
                _ => {}
            }
        }
        let mut writes: Vec<vk::WriteDescriptorSet> = plan
            .iter()
            .zip(infos.iter())
            .map(|(&(set, binding, ty), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.sets[set as usize])
                    .dst_binding(binding)
                    .descriptor_type(ty)
                    .buffer_info(info)
            })
            .collect();

        // The TLAS, and the one storage image the resolve pass writes.
        let accels = [vs.tlas];
        let mut asw = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&accels);
        let mut w_as = vk::WriteDescriptorSet::default()
            .dst_set(self.sets[0])
            .dst_binding(binding_of(Reg::T, 7))
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut asw);
        w_as.descriptor_count = 1;
        writes.push(w_as);

        let ii = [vk::DescriptorImageInfo::default()
            .image_view(self.hdr.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(self.sets[0])
                .dst_binding(binding_of(Reg::U, 14))
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&ii),
        );

        let si = [
            [vk::DescriptorImageInfo::default().sampler(self.samp_lin)],
            [vk::DescriptorImageInfo::default().sampler(self.samp_aniso)],
        ];
        for (i, reg) in [0u32, 1].iter().enumerate() {
            if self.map.entries.contains_key(&(1, binding_of(Reg::S, *reg))) {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.sets[1])
                        .dst_binding(binding_of(Reg::S, *reg))
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&si[i]),
                );
            }
        }
        unsafe { d.update_descriptor_sets(&writes, &[]) };
    }

    /// One frame: the two cache fills, the reference dispatch, then resolve.
    /// The whole thing is one submit — the `HeadlessGpu::run` contract.
    pub fn render(&self, hg: &VkHeadless, p: &FrameParams, samples: u32) -> Result<(), String> {
        let vkd = &hg.vk;
        let mut cb = self.cb_base.with_frame(p, false, false, false, false, false, false);
        cb.cloud_grid = if self.cloud_shadow_n == 0 || !p.clouds.enabled {
            [0.0; 4]
        } else {
            crate::clouds::shadow_grid_row(
                self.cb_base.sun,
                self.scene_aabb,
                p.clouds.diag,
                self.cloud_shadow_n,
            )
        };
        vkd.write(&self.frame_cb, cb.bytes())?;
        // `cbuffer Push : register(b1)` is 4 dwords; only the first is read
        // here (`inv_samples`), but the whole row is written so a slot is
        // never left holding the previous frame's bytes.
        let inv = 1.0f32 / samples.max(1) as f32;
        let mut pb = [0u8; 16];
        pb[..4].copy_from_slice(&inv.to_bits().to_le_bytes());
        vkd.write(&self.push, &pb)?;

        let gx = self.rw.div_ceil(8);
        let gy = self.rh.div_ceil(8);
        let sky_pts = ((self.rw / self.sky_lod_k) + 2) * ((self.rh / self.sky_lod_k) + 2);
        let sky_groups = sky_pts.div_ceil(64);
        let csn_groups =
            (crate::clouds::CLOUD_SHADOW_MAX * crate::clouds::CLOUD_SHADOW_MAX).div_ceil(64);
        let k = self.sky_lod_k;
        let csn = self.cloud_shadow_n;

        hg.run(|d, cmd| unsafe {
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layouts.pipeline,
                0,
                &self.sets,
                &[],
            );
            // The output image starts UNDEFINED and every pass reads/writes it
            // as a storage image, so the one transition it ever needs is
            // UNDEFINED -> GENERAL, once per frame.
            let ib = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(self.hdr.img)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                )
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[ib],
            );

            if csn > 0 {
                d.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipes[P_CLOUD_SHADOW],
                );
                d.cmd_dispatch(cmd, csn_groups.min(32768), csn_groups.div_ceil(32768), 1);
            }
            if k > 1 {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipes[P_SKY_LOD]);
                d.cmd_dispatch(cmd, sky_groups.min(32768), sky_groups.div_ceil(32768), 1);
            }
            barrier(d, cmd);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipes[P_REFERENCE]);
            d.cmd_dispatch(cmd, gx, gy, 1);
            barrier(d, cmd);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipes[P_RESOLVE]);
            d.cmd_dispatch(cmd, gx, gy, 1);
            barrier(d, cmd);
        })
    }

    /// The resolved RGBA16F image, decoded to f32 RGB — the `read_hdr_output`
    /// peer, and the only thing in this file that proves the storage image was
    /// ever written.
    pub fn read_hdr(&self, hg: &VkHeadless) -> Result<Vec<f32>, String> {
        let vkd = &hg.vk;
        let n = (self.rw as u64) * (self.rh as u64) * 8;
        let stage = vkd.buffer(n, vk::BufferUsageFlags::TRANSFER_DST, true)?;
        let r = hg.run(|d, cmd| unsafe {
            let region = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D { width: self.rw, height: self.rh, depth: 1 });
            d.cmd_copy_image_to_buffer(
                cmd,
                self.hdr.img,
                vk::ImageLayout::GENERAL,
                stage.buf,
                &[region],
            );
        });
        let out = r.and_then(|_| vkd.read(&stage, n as usize)).map(|b| {
            b.chunks_exact(2)
                .map(|c| f32::from(half_from_bits(u16::from_le_bytes(c.try_into().unwrap()))))
                .collect::<Vec<f32>>()
        });
        vkd.free_buffer(&stage);
        out
    }

    pub fn destroy(&self, hg: &VkHeadless) {
        let vkd = &hg.vk;
        let d = &vkd.device;
        unsafe {
            let _ = d.device_wait_idle();
            for p in self.pipes {
                d.destroy_pipeline(p, None);
            }
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_sampler(self.samp_lin, None);
            d.destroy_sampler(self.samp_aniso, None);
            d.destroy_image_view(self.hdr.view, None);
            d.destroy_image(self.hdr.img, None);
            d.free_memory(self.hdr.mem, None);
        }
        self.layouts.destroy(vkd);
        for b in [
            &self.accum,
            &self.tbuf,
            &self.info,
            &self.counters,
            &self.cloud_lod,
            &self.cloud_shadow,
            &self.frame_cb,
            &self.push,
        ] {
            vkd.free_buffer(b);
        }
    }
}

fn barrier(d: &ash::Device, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    unsafe {
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[mb],
            &[],
            &[],
        );
    }
}

fn create_image(vkd: &crate::vk::device::Vk, rw: u32, rh: u32) -> Result<Image, String> {
    let d = &vkd.device;
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R16G16B16A16_SFLOAT)
        .extent(vk::Extent3D { width: rw, height: rh, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let img = unsafe { d.create_image(&ci, None) }
        .map_err(|e| format!("vkCreateImage: {e}"))?;
    let req = unsafe { d.get_image_memory_requirements(img) };
    let idx = crate::vk::device::mem_type_index(
        &vkd.mem,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| "no device-local memory type for the output image".to_string())?;
    let mem = unsafe {
        d.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(idx),
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
                .format(vk::Format::R16G16B16A16_SFLOAT)
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
    Ok(Image { img, view, mem })
}

/// IEEE binary16 -> f32. The one decode the readback needs; the tree's other
/// f16 sites (`dlss::ld16`) live on the D3D12 side of a `#[cfg]`.
fn half_from_bits(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // Subnormal: renormalize.
            let mut e2 = -1i32;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | (((127 - 15 + e2 + 1) as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 0x1f {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
