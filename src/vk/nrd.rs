//! VkNrd — the Vulkan host for NRD's own compute passes, twinning
//! `gpu/nrd_gpu.rs`. `src/nrd.rs` is the library binding (portable since
//! B4b-i), `gfx::denoise` is the shared vocabulary and settings, and what lives
//! here is the RECORDING: NRD's texture pools, one pipeline per `PipelineDesc`
//! from the SPIR-V blobs the library serves, the descriptor sets its dispatches
//! want, and the barriers between them.
//!
//! THIS IS THE FIRST DESCRIPTOR LAYOUT IN THIS BACKEND THAT IS DECLARED RATHER
//! THAN DERIVED, and that inverts the rule every other module here follows.
//! `vk::reflect` walks OUR compiled modules and `vk::layout` turns the result
//! into set layouts precisely so no register map is ever transcribed — but
//! NRD's blobs are pre-compiled with NRD's OWN `-fvk-*-shift` values, which it
//! reports in `LibraryDesc::spirv_binding_offsets`, and reflecting them would
//! be reading back a number the library already told us. So `binding_of`'s
//! never-a-literal rule does not reach here: the offsets come from the library
//! at run time and nothing in this file spells one.
//!
//! THE TRAP THAT MAKES THAT WORTH SAYING TWICE: NRD's CMakeLists sets
//! `SPIRV_SREG_OFFSET 0 / BREG 2 / UREG 3 / TREG 20` as plain `set()`s, and
//! `Source/Wrapper.cpp` REORDERS them into the struct as
//! `{sampler, texture, constantBuffer, storageTextureAndBuffer}` = `{0, 20, 2,
//! 3}`. A recorder reading the CMake order binds every resource one window off.
//! `--check-nrd`'s N1 pins the struct order by NAME for exactly this reason,
//! and does so un-`cfg`'d, so the Windows gate protects the value only this
//! file consumes.
//!
//! FOUR DIFFERENCES FROM THE D3D12 TWIN, each because the API differs rather
//! than because the design changed:
//!
//! 1. **Per-pipeline set layouts.** D3D12 sizes ONE table to the per-set
//!    maxima and leaves the slots past a pipeline's actual count holding
//!    whatever the previous frame put there — legal, and legal here only via
//!    `descriptorBindingPartiallyBound`. `PipelineDesc::resource_ranges` gives
//!    the exact counts, so each pipeline gets a layout that describes exactly
//!    what it binds and nothing is stale by construction. The price is that
//!    pipeline layouts differ per pipeline, which breaks set-1 compatibility
//!    (Vulkan's rule: sets 0..N must be identical for set N to survive a
//!    layout change), so both sets are bound per dispatch. That is one call.
//! 2. **No layout transitions and no state tracker.** Every image rests in
//!    `GENERAL` for its whole life — legal for `SAMPLED_IMAGE` and
//!    `STORAGE_IMAGE` alike — so D3D12's `Reg::state` and its
//!    NPSR<->UA bracketing have no counterpart and must not be invented. One
//!    global memory barrier per dispatch replaces the whole thing; the A/B
//!    that would justify narrowing measured a WASH on D3D12 (ReBLUR's ~31
//!    passes are full-screen, so there is no overlap to unlock).
//! 3. **The CB is a DYNAMIC uniform buffer.** D3D12 rebinds a root CBV per
//!    dispatch at a ring offset; the Vulkan spelling of "same buffer, moving
//!    window" is `UNIFORM_BUFFER_DYNAMIC` plus a per-dispatch offset, which
//!    also keeps set 1 a single object rather than a ring of sets. Dynamic vs
//!    non-dynamic is a LAYOUT property and invisible to the shader, so NRD's
//!    modules need no cooperation.
//! 4. **`RING_FRAMES` collapses to 1.** `VkHeadless::run` fences every submit,
//!    so there are no frames in flight to ring against and the descriptor pool
//!    can simply be RESET at the top of each `record()`. THAT IS A CONTRACT,
//!    not an observation: a future presenter with real frames in flight must
//!    either ring the pool or keep the fence, and resetting a pool whose sets
//!    are still executing is undefined behaviour the validation layer will
//!    name.
//!
//! WHAT IS DELIBERATELY NOT PORTED: `FR_NRD_DEBUG`'s OUT_VALIDATION dump (72
//! lines under a lever, and the readback ring it needs does not exist here) and
//! `FR_NRD_BARRIER=narrow` (see difference 2 — it is an arm for a barrier
//! scheme this file does not have). The validation PLANE is still allocated,
//! because NRD names it in `reg_for` whether or not it writes it, and making it
//! conditional would trade one allocation for a failure mode.

use crate::gfx::denoise::Plane;
use crate::nrd;
use crate::vk::device::Vk;
use crate::vk::headless::VkHeadless;
use ash::vk;

type Result<T> = std::result::Result<T, String>;

/// nrd::Format -> Vulkan (the NRDDescs.h enum order — the same ordinals
/// `gpu::nrd_gpu::dxgi_format` switches on, so the two tables read as one
/// statement side by side and a transcription slip in either is visible by
/// eye).
fn vk_format(f: u32) -> Result<vk::Format> {
    Ok(match f {
        0 => vk::Format::R8_UNORM,
        1 => vk::Format::R8_SNORM,
        2 => vk::Format::R8_UINT,
        3 => vk::Format::R8_SINT,
        4 => vk::Format::R8G8_UNORM,
        5 => vk::Format::R8G8_SNORM,
        6 => vk::Format::R8G8_UINT,
        7 => vk::Format::R8G8_SINT,
        8 => vk::Format::R8G8B8A8_UNORM,
        9 => vk::Format::R8G8B8A8_SNORM,
        10 => vk::Format::R8G8B8A8_UINT,
        11 => vk::Format::R8G8B8A8_SINT,
        12 => vk::Format::R8G8B8A8_SRGB,
        13 => vk::Format::R16_UNORM,
        14 => vk::Format::R16_SNORM,
        15 => vk::Format::R16_UINT,
        16 => vk::Format::R16_SINT,
        17 => vk::Format::R16_SFLOAT,
        18 => vk::Format::R16G16_UNORM,
        19 => vk::Format::R16G16_SNORM,
        20 => vk::Format::R16G16_UINT,
        21 => vk::Format::R16G16_SINT,
        22 => vk::Format::R16G16_SFLOAT,
        23 => vk::Format::R16G16B16A16_UNORM,
        24 => vk::Format::R16G16B16A16_SNORM,
        25 => vk::Format::R16G16B16A16_UINT,
        26 => vk::Format::R16G16B16A16_SINT,
        27 => vk::Format::R16G16B16A16_SFLOAT,
        28 => vk::Format::R32_UINT,
        29 => vk::Format::R32_SINT,
        30 => vk::Format::R32_SFLOAT,
        31 => vk::Format::R32G32_UINT,
        32 => vk::Format::R32G32_SINT,
        33 => vk::Format::R32G32_SFLOAT,
        34 => vk::Format::R32G32B32_UINT,
        35 => vk::Format::R32G32B32_SINT,
        36 => vk::Format::R32G32B32_SFLOAT,
        37 => vk::Format::R32G32B32A32_UINT,
        38 => vk::Format::R32G32B32A32_SINT,
        39 => vk::Format::R32G32B32A32_SFLOAT,
        // D3D12 spells this one A2B10G10R10 in Vulkan's channel order: DXGI's
        // R10G10B10A2 puts R in the LOW bits, which is Vulkan's
        // A2B10G10R10_UNORM_PACK32, NOT A2R10G10B10. The bridge writes this
        // plane through `NRD_FrontEnd_PackNormalAndRoughness`'s enc-2 layout,
        // so a swapped pair here is a swapped normal, not a compile error.
        40 => vk::Format::A2B10G10R10_UNORM_PACK32,
        41 => vk::Format::A2B10G10R10_UINT_PACK32,
        42 => vk::Format::B10G11R11_UFLOAT_PACK32,
        43 => vk::Format::E5B9G9R9_UFLOAT_PACK32,
        _ => return Err(format!("nrd: unknown Format {f}")),
    })
}

/// An image this module owns. The seven app planes are BORROWED from the
/// tracer (B4a allocated and wired them, and the bridge kernels write them), so
/// only NRD's pools and the validation plane are owned here.
struct Owned {
    img: vk::Image,
    view: vk::ImageView,
    mem: vk::DeviceMemory,
}

/// One registered resource as the dispatch loop sees it: JUST the view.
///
/// There is no `state` twin of D3D12's `Reg` because there is no state — every
/// image rests in `GENERAL` — and no image handle because the only thing that
/// wants one is the one-time UNDEFINED->GENERAL transition, which runs over
/// `owned` and must NOT touch the app planes (the tracer already laid those out,
/// and a second UNDEFINED-sourced barrier would discard the pack).
#[derive(Copy, Clone)]
struct Reg {
    view: vk::ImageView,
}

/// Headroom over N1's measured 31 dispatches, asserted rather than trusted.
const MAX_DISPATCHES: usize = 48;

pub struct VkNrd {
    pub nrd: nrd::Nrd,
    pipes: Vec<vk::Pipeline>,
    /// Per pipeline: the set-0 layout its `resource_ranges` describe, and the
    /// pipeline layout `[set0, set1]` built from it.
    set0_layouts: Vec<vk::DescriptorSetLayout>,
    pipe_layouts: Vec<vk::PipelineLayout>,
    set1_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    samplers: Vec<vk::Sampler>,
    /// `[permanent pool.., transient pool.., app planes..]` — the same
    /// concatenation `reg_for` indexes on the D3D12 side.
    regs: Vec<Reg>,
    /// Everything in `regs` this module allocated (pools + the validation
    /// plane); the app planes are absent because the tracer owns them.
    owned: Vec<Owned>,
    perm_base: usize,
    trans_base: usize,
    plane_idx: [usize; Plane::COUNT],
    /// NRD's binding windows, read from the library. Never literals — see the
    /// module header's reorder trap.
    b_texture: u32,
    b_cbuffer: u32,
    b_storage: u32,
    cb: crate::vk::device::Buffer,
    /// PERSISTENTLY MAPPED, the `d3d12::UploadBuffer` shape: the whole dispatch
    /// list is known before any of it is recorded, so all ~31 slots are filled
    /// in one host pass and there is nothing to rewrite BETWEEN dispatches.
    /// That is what makes a plain host write legal here where the wavefront
    /// ladder's per-dispatch constants needed `vkCmdUpdateBuffer` and its
    /// easy-to-omit write-after-read edge — the hazard does not arise when
    /// every write happens before the submit.
    cb_ptr: *mut u8,
    cb_slot: u64,
    pub rw: u32,
    pub rh: u32,
    prev_size: (u32, u32),
    /// The pools and the validation plane are created UNDEFINED and must reach
    /// GENERAL exactly once — `UNDEFINED` as an old layout licenses the driver
    /// to discard contents, so re-transitioning per frame would wipe NRD's
    /// permanent pool, which is precisely the history it denoises with.
    laid_out: bool,
    /// The LAST frame's dispatch count, and the MAX over this instance's life.
    /// Both are reported because they legitimately differ and the gap is
    /// informative: NRD's dispatch list is a function of the settings, so a
    /// RESTART frame skips every pass that would consume a history it has just
    /// thrown away. Reporting only the last would make V15's line depend on
    /// which frame it happened to end on; reporting only the max would hide
    /// that the reset latch reached the library at all.
    pub dispatch_count: usize,
    pub dispatch_max: usize,
    pub pool_sets: u32,
}

impl VkNrd {
    /// Open the library, build a REBLUR_DIFFUSE_SPECULAR instance, and stand up
    /// everything its `InstanceDesc` asks for. `app_planes` are the tracer's
    /// bridge planes in `Plane::ALL` order, minus validation — B4a allocated
    /// them and `wire_nrd` already points `cs_nrd_pack`/`cs_nrd_out` at them,
    /// so the engine registers the SAME images rather than a second set that
    /// would then need copying at both ends.
    pub fn new(
        hg: &VkHeadless,
        dll_dir: &str,
        rw: u32,
        rh: u32,
        app_planes: &[(vk::Image, vk::ImageView)],
    ) -> Result<Self> {
        if app_planes.len() != Plane::COUNT - 1 {
            return Err(format!(
                "nrd: {} app planes, expected {} (Plane::ALL minus OutValidation)",
                app_planes.len(),
                Plane::COUNT - 1
            ));
        }
        let denoisers =
            [nrd::DenoiserDesc { identifier: 0, denoiser: nrd::DENOISER_REBLUR_DIFFUSE_SPECULAR }];
        let inst = nrd::Nrd::new(dll_dir, &denoisers)?;
        let d = inst.instance_desc();
        let vkd = &hg.vk;
        let dev = &vkd.device;

        // --- The binding windows, from the library. `spirv_binding_offsets` is
        // captured on the instance (B4b-i) and N1 pins it; this is its only
        // consumer, which is why that pin is un-cfg'd.
        let o = inst.spirv_offsets;
        let b_sampler = o.sampler_offset + d.samplers_base_register_index;
        let b_texture = o.texture_offset + d.resources_base_register_index;
        let b_cbuffer = o.constant_buffer_offset + d.constant_buffer_register_index;
        let b_storage = o.storage_texture_and_buffer_offset + d.resources_base_register_index;

        // --- Samplers. `InstanceDesc::samplers` is [NEAREST_CLAMP,
        // LINEAR_CLAMP] and they are IMMUTABLE in the layout, which is both the
        // idiomatic spelling and what keeps them out of the per-frame writes.
        let samp_kinds =
            unsafe { std::slice::from_raw_parts(d.samplers, d.samplers_num as usize) }.to_vec();
        let mut samplers = Vec::with_capacity(samp_kinds.len());
        for &s in &samp_kinds {
            let filter = if s == nrd::SAMPLER_LINEAR_CLAMP {
                vk::Filter::LINEAR
            } else {
                vk::Filter::NEAREST
            };
            let mip = if s == nrd::SAMPLER_LINEAR_CLAMP {
                vk::SamplerMipmapMode::LINEAR
            } else {
                vk::SamplerMipmapMode::NEAREST
            };
            let ci = vk::SamplerCreateInfo::default()
                .mag_filter(filter)
                .min_filter(filter)
                .mipmap_mode(mip)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(vk::LOD_CLAMP_NONE);
            samplers.push(
                unsafe { dev.create_sampler(&ci, None) }
                    .map_err(|e| format!("nrd: vkCreateSampler: {e}"))?,
            );
        }

        // --- Set 1: the constant buffer (DYNAMIC) + the immutable samplers.
        // NRD puts both in `constantBufferAndSamplersSpaceIndex`, which is a
        // SPACE and therefore a SET under the shift scheme every module in this
        // corpus is compiled with.
        let mut set1_bindings = vec![vk::DescriptorSetLayoutBinding::default()
            .binding(b_cbuffer)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        for (i, s) in samplers.iter().enumerate() {
            set1_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b_sampler + i as u32)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .immutable_samplers(std::slice::from_ref(s)),
            );
        }
        let set1_layout = unsafe {
            dev.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&set1_bindings),
                None,
            )
        }
        .map_err(|e| format!("nrd: set-1 layout: {e}"))?;

        // --- One set-0 layout + pipeline layout + pipeline per PipelineDesc.
        let pipes_desc =
            unsafe { std::slice::from_raw_parts(d.pipelines, d.pipelines_num as usize) };
        let entry = unsafe { std::ffi::CStr::from_ptr(d.shader_entry_point) }.to_owned();
        let mut set0_layouts = Vec::with_capacity(pipes_desc.len());
        let mut pipe_layouts = Vec::with_capacity(pipes_desc.len());
        let mut pipes = Vec::with_capacity(pipes_desc.len());
        for (i, p) in pipes_desc.iter().enumerate() {
            // The layout describes EXACTLY this pipeline's ranges, in the order
            // NRD lists them, each starting at its type's window base.
            let ranges = unsafe {
                std::slice::from_raw_parts(p.resource_ranges, p.resource_ranges_num as usize)
            };
            let mut bindings = Vec::new();
            let (mut ntex, mut nstor) = (0u32, 0u32);
            for r in ranges {
                let (base, ty, n) = match r.descriptor_type {
                    nrd::DESC_TEXTURE => {
                        let b = b_texture + ntex;
                        ntex += r.descriptors_num;
                        (b, vk::DescriptorType::SAMPLED_IMAGE, r.descriptors_num)
                    }
                    nrd::DESC_STORAGE_TEXTURE => {
                        let b = b_storage + nstor;
                        nstor += r.descriptors_num;
                        (b, vk::DescriptorType::STORAGE_IMAGE, r.descriptors_num)
                    }
                    other => {
                        return Err(format!("nrd: pipeline {i} has DescriptorType {other}"))
                    }
                };
                // NRD reports ranges, but each descriptor is its own binding
                // here: DXC emits one decoration per declared resource, not an
                // array, so a range of N is N consecutive bindings.
                for k in 0..n {
                    bindings.push(
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(base + k)
                            .descriptor_type(ty)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    );
                }
            }
            let set0 = unsafe {
                dev.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
            }
            .map_err(|e| format!("nrd: pipeline {i} set-0 layout: {e}"))?;
            let layouts = [set0, set1_layout];
            let pl = unsafe {
                dev.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    None,
                )
            }
            .map_err(|e| format!("nrd: pipeline {i} layout: {e}"))?;

            // The blob. B4b-i MEASURED that 9 of 14 sit at a non-4-byte-aligned
            // address — legal, NRD packs them back to back and promises nothing
            // — so this COPIES into a Vec<u32> and must never cast the pointer:
            // vkCreateShaderModule takes *const u32.
            let cs = p.shader();
            if !cs.is_present() {
                return Err(format!("nrd: pipeline {i} has no SPIR-V (rebuild the library)"));
            }
            let n = cs.size as usize;
            if !n.is_multiple_of(4) {
                return Err(format!("nrd: pipeline {i} SPIR-V is {n} B, not a whole word count"));
            }
            let mut words = vec![0u32; n / 4];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    cs.bytecode as *const u8,
                    words.as_mut_ptr() as *mut u8,
                    n,
                )
            };
            let module = unsafe {
                dev.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&words),
                    None,
                )
            }
            .map_err(|e| format!("nrd: pipeline {i} shader module: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry);
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(pl);
            let pipe = unsafe { dev.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) }
                .map_err(|(_, e)| format!("nrd: pipeline {i}: {e}"))?[0];
            unsafe { dev.destroy_shader_module(module, None) };
            set0_layouts.push(set0);
            pipe_layouts.push(pl);
            pipes.push(pipe);
        }

        // --- The texture pools NRD asks the app to allocate, then our own
        // validation plane, then the tracer's seven.
        let mut owned = Vec::new();
        let mut regs = Vec::new();
        let push = |fmt: u32, w: u32, h: u32, owned: &mut Vec<Owned>, regs: &mut Vec<Reg>| -> Result<()> {
            let im = create_image(vkd, w.max(1), h.max(1), vk_format(fmt)?)?;
            regs.push(Reg { view: im.view });
            owned.push(im);
            Ok(())
        };
        let perm =
            unsafe { std::slice::from_raw_parts(d.permanent_pool, d.permanent_pool_size as usize) };
        let trans =
            unsafe { std::slice::from_raw_parts(d.transient_pool, d.transient_pool_size as usize) };
        let perm_base = 0usize;
        for t in perm {
            let f = t.downsample_factor.max(1) as u32;
            push(t.format, rw.div_ceil(f), rh.div_ceil(f), &mut owned, &mut regs)?;
        }
        let trans_base = regs.len();
        for t in trans {
            let f = t.downsample_factor.max(1) as u32;
            push(t.format, rw.div_ceil(f), rh.div_ceil(f), &mut owned, &mut regs)?;
        }
        let mut plane_idx = [0usize; Plane::COUNT];
        for (k, p) in Plane::ALL.iter().enumerate() {
            plane_idx[k] = regs.len();
            if *p == Plane::OutValidation {
                push(p.nrd_format(), rw, rh, &mut owned, &mut regs)?;
            } else {
                regs.push(Reg { view: app_planes[k].1 });
            }
        }

        // --- The descriptor pool, sized from NRD's OWN per-frame totals rather
        // than from MAX_DISPATCHES * the per-set maxima. `total_textures_num` /
        // `total_storage_textures_num` / `sets_max_num` are exactly the
        // "descriptors one frame of dispatches needs" numbers the library
        // publishes for this purpose (N1 measures 70 / 61 / 36 against 31
        // dispatches), so an over-allocation here would be inventing a number
        // the library already answered. +1 set for set 1, which is allocated
        // per frame alongside them because the pool reset frees everything.
        let dp = &d.descriptor_pool_desc;
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(dp.total_textures_num.max(1)),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(dp.total_storage_textures_num.max(1)),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(samplers.len().max(1) as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1),
        ];
        let pool_sets = dp.sets_max_num.max(1) + 1;
        let pool = unsafe {
            dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(pool_sets).pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| format!("nrd: descriptor pool: {e}"))?;

        // --- The CB ring. One host-visible buffer, one slot per dispatch, at
        // the device's own uniform-buffer offset alignment (D3D12's fixed 256 is
        // a D3D constant; here it is a queryable limit and must be queried).
        let props = unsafe { vkd.instance.get_physical_device_properties(vkd.phys) };
        let align = props.limits.min_uniform_buffer_offset_alignment.max(1);
        let cb_slot = (d.constant_buffer_max_data_size as u64).next_multiple_of(align).max(align);
        let cb =
            vkd.buffer(cb_slot * MAX_DISPATCHES as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, true)?;
        let cb_ptr = unsafe {
            dev.map_memory(cb.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| format!("nrd: map CB ring: {e}"))? as *mut u8;

        eprintln!(
            "vk-nrd: v{}.{}.{} — {} pipelines, pools {}+{}, bindings s{} b{} u{} t{}, cb {} B x {}",
            inst.version.0,
            inst.version.1,
            inst.version.2,
            pipes.len(),
            perm.len(),
            trans.len(),
            b_sampler,
            b_cbuffer,
            b_storage,
            b_texture,
            cb_slot,
            MAX_DISPATCHES,
        );

        Ok(Self {
            nrd: inst,
            pipes,
            set0_layouts,
            pipe_layouts,
            set1_layout,
            pool,
            samplers,
            regs,
            owned,
            perm_base,
            trans_base,
            plane_idx,
            b_texture,
            b_cbuffer,
            b_storage,
            cb,
            cb_ptr,
            cb_slot,
            rw,
            rh,
            prev_size: (rw, rh),
            laid_out: false,
            dispatch_count: 0,
            dispatch_max: 0,
            pool_sets,
        })
    }

    /// The previous frame's size, for `CommonSettings::resourceSizePrev`. Equal
    /// to (rw, rh) for this instance's whole life today — the res is locked at
    /// construction — so it is bookkeeping against a future resize path, kept
    /// because the failure it prevents is silent (NRD would reproject through
    /// the wrong previous rect and denoise a slightly wrong history).
    pub fn prev_size(&self) -> (u32, u32) {
        self.prev_size
    }

    /// How many pipelines the library served — reported by V15 beside the
    /// dispatch count, and the cheapest cross-check that this backend built
    /// the same instance `--check-nrd`'s N1 audits (14 / 31 on v4.17.3).
    pub fn pipeline_count(&self) -> usize {
        self.pipes.len()
    }

    fn reg_for(&self, r: &nrd::ResourceDesc) -> Result<usize> {
        if r.ty == nrd::RES_PERMANENT_POOL {
            return Ok(self.perm_base + r.index_in_pool as usize);
        }
        if r.ty == nrd::RES_TRANSIENT_POOL {
            return Ok(self.trans_base + r.index_in_pool as usize);
        }
        Plane::from_resource_type(r.ty)
            .map(|p| self.plane_idx[p.index()])
            .ok_or_else(|| format!("nrd: unmapped ResourceType {}", r.ty))
    }

    /// Bring every image this module OWNS into `GENERAL`, once. The tracer's
    /// seven are excluded on purpose: `VkTracer::nrd_lay_out` already did them,
    /// and a second UNDEFINED-sourced transition would discard the pack the
    /// bridge just wrote.
    unsafe fn lay_out(&mut self, dev: &ash::Device, cmd: vk::CommandBuffer) {
        if self.laid_out {
            return;
        }
        self.laid_out = true;
        let bs: Vec<vk::ImageMemoryBarrier> = self
            .owned
            .iter()
            .map(|o| {
                vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .image(o.img)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    )
            })
            .collect();
        unsafe {
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &bs,
            )
        };
    }

    /// Record one frame's NRD passes. The caller has already filled the IN
    /// planes (`cs_nrd_pack`) into the same command buffer; the OUT planes are
    /// ready for `cs_nrd_out` when this returns.
    ///
    /// THE POOL RESET IS WHY THIS NEEDS THE FENCE-PER-SUBMIT CONTRACT: it frees
    /// every set the previous call allocated, which is only legal once that
    /// call's work has completed.
    pub fn record(
        &mut self,
        dev: &ash::Device,
        cmd: vk::CommandBuffer,
        cs: &nrd::CommonSettings,
        rs: &nrd::ReblurSettings,
    ) -> Result<()> {
        self.nrd.set_common_settings(cs)?;
        self.nrd.set_reblur_settings(0, rs)?;
        // Snapshot: the slice `compute_dispatches` returns is owned by the
        // instance, borrows it, and is overwritten by the next call — while the
        // loop below needs `&mut self`.
        let dispatches: Vec<nrd::DispatchDesc> =
            self.nrd.compute_dispatches(&[0])?.to_vec();
        if dispatches.len() > MAX_DISPATCHES {
            return Err(format!(
                "nrd: {} dispatches > CB ring capacity {MAX_DISPATCHES}",
                dispatches.len()
            ));
        }
        self.dispatch_count = dispatches.len();
        self.dispatch_max = self.dispatch_max.max(dispatches.len());
        unsafe { self.lay_out(dev, cmd) };

        unsafe { dev.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty()) }
            .map_err(|e| format!("nrd: vkResetDescriptorPool: {e}"))?;

        // One allocation call for the whole frame: set 1 followed by one set-0
        // per dispatch, each with ITS pipeline's layout.
        let mut want = vec![self.set1_layout];
        for dsp in &dispatches {
            let pi = dsp.pipeline_index as usize;
            want.push(
                *self
                    .set0_layouts
                    .get(pi)
                    .ok_or_else(|| format!("nrd: pipeline_index {pi} out of range"))?,
            );
        }
        let sets = unsafe {
            dev.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool)
                    .set_layouts(&want),
            )
        }
        .map_err(|e| format!("nrd: vkAllocateDescriptorSets({}): {e}", want.len()))?;
        let set1 = sets[0];

        // Set 1's one writable member. The samplers are immutable and must NOT
        // be written — the layout carries them.
        let cb_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.cb.buf)
            .offset(0)
            .range(self.cb_slot)];
        unsafe {
            dev.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set1)
                    .dst_binding(self.b_cbuffer)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .buffer_info(&cb_info)],
                &[],
            )
        };

        // Constants: NRD flags identical-CB runs, so a repeat reuses the last
        // slot rather than re-uploading it.
        let mut cb_off_last: Option<u64> = None;
        let mut offs: Vec<u32> = Vec::with_capacity(dispatches.len());
        for (di, dsp) in dispatches.iter().enumerate() {
            let off = if dsp.constant_buffer_data_size == 0 {
                // A dispatch with no constants still binds set 1, and a dynamic
                // offset must name a slot inside the buffer — reuse the last
                // written one (0 on the first, which is in range by
                // construction). Its shader reads nothing from it.
                cb_off_last.unwrap_or(0)
            } else {
                let reuse = dsp.constant_buffer_data_matches_previous_dispatch != 0;
                match (reuse, cb_off_last) {
                    // NRD flags identical-CB runs; reuse the slot rather than
                    // re-uploading the same bytes.
                    (true, Some(o)) => o,
                    _ => {
                        let o = di as u64 * self.cb_slot;
                        let n = dsp.constant_buffer_data_size as usize;
                        if n > self.cb_slot as usize {
                            return Err(format!(
                                "nrd: dispatch {di} CB {n} B > slot {}",
                                self.cb_slot
                            ));
                        }
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                dsp.constant_buffer_data,
                                self.cb_ptr.add(o as usize),
                                n,
                            )
                        };
                        cb_off_last = Some(o);
                        o
                    }
                }
            };
            offs.push(off as u32);
        }

        // The dispatch loop.
        for (di, dsp) in dispatches.iter().enumerate() {
            let resources =
                unsafe { std::slice::from_raw_parts(dsp.resources, dsp.resources_num as usize) };
            let set0 = sets[di + 1];

            // The descriptor writes, in the ranges' concatenated order — the
            // i-th TEXTURE resource lands at `b_texture + i`, the j-th STORAGE
            // at `b_storage + j`. That ordering IS the contract with NRD's
            // shaders and is the same one the D3D12 table build follows.
            let mut infos: Vec<[vk::DescriptorImageInfo; 1]> = Vec::with_capacity(resources.len());
            let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(resources.len());
            let (mut ti, mut si) = (0u32, 0u32);
            let mut kinds: Vec<(u32, vk::DescriptorType)> = Vec::with_capacity(resources.len());
            for r in resources {
                let idx = self.reg_for(r)?;
                let (binding, ty) = if r.descriptor_type == nrd::DESC_TEXTURE {
                    let b = self.b_texture + ti;
                    ti += 1;
                    (b, vk::DescriptorType::SAMPLED_IMAGE)
                } else {
                    let b = self.b_storage + si;
                    si += 1;
                    (b, vk::DescriptorType::STORAGE_IMAGE)
                };
                infos.push([vk::DescriptorImageInfo::default()
                    .image_view(self.regs[idx].view)
                    .image_layout(vk::ImageLayout::GENERAL)]);
                kinds.push((binding, ty));
            }
            for (k, (binding, ty)) in kinds.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set0)
                        .dst_binding(*binding)
                        .descriptor_type(*ty)
                        .image_info(&infos[k]),
                );
            }
            unsafe { dev.update_descriptor_sets(&writes, &[]) };

            // One global memory barrier: the previous dispatch's storage writes
            // must be visible to this one's reads. See the module header for
            // why this is not narrowed.
            let mb = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe {
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[mb],
                    &[],
                    &[],
                );
                let pi = dsp.pipeline_index as usize;
                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipes[pi]);
                dev.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipe_layouts[pi],
                    0,
                    &[set0, set1],
                    &[offs[di]],
                );
                dev.cmd_dispatch(cmd, dsp.grid_width as u32, dsp.grid_height as u32, 1);
            }
        }

        // One trailing barrier so the bridge's `cs_nrd_out` sees the OUT writes.
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        unsafe {
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mb],
                &[],
                &[],
            )
        };
        self.prev_size = (self.rw, self.rh);
        Ok(())
    }

    pub fn destroy(&self, hg: &VkHeadless) {
        let dev = &hg.vk.device;
        unsafe {
            for p in &self.pipes {
                dev.destroy_pipeline(*p, None);
            }
            for l in &self.pipe_layouts {
                dev.destroy_pipeline_layout(*l, None);
            }
            for l in &self.set0_layouts {
                dev.destroy_descriptor_set_layout(*l, None);
            }
            dev.destroy_descriptor_set_layout(self.set1_layout, None);
            dev.destroy_descriptor_pool(self.pool, None);
            for s in &self.samplers {
                dev.destroy_sampler(*s, None);
            }
            for o in &self.owned {
                dev.destroy_image_view(o.view, None);
                dev.destroy_image(o.img, None);
                dev.free_memory(o.mem, None);
            }
            // Unmap before the free — `free_buffer` frees the memory the
            // persistent map points into.
            dev.unmap_memory(self.cb.mem);
        }
        hg.vk.free_buffer(&self.cb);
    }
}

fn create_image(vkd: &Vk, w: u32, h: u32, fmt: vk::Format) -> Result<Owned> {
    let d = &vkd.device;
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(fmt)
        .extent(vk::Extent3D { width: w, height: h, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        // SAMPLED as well as STORAGE: NRD binds the same pool texture as a UAV
        // in the pass that writes it and as an SRV in the pass that reads it,
        // so both usages are real. TRANSFER_SRC is for readback in the gate.
        .usage(
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let img =
        unsafe { d.create_image(&ci, None) }.map_err(|e| format!("nrd: vkCreateImage: {e}"))?;
    let req = unsafe { d.get_image_memory_requirements(img) };
    let idx = crate::vk::device::mem_type_index(
        &vkd.mem,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| "nrd: no device-local memory type for a pool texture".to_string())?;
    let mem = unsafe {
        d.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(idx),
            None,
        )
    }
    .map_err(|e| format!("nrd: vkAllocateMemory(image): {e}"))?;
    unsafe { d.bind_image_memory(img, mem, 0) }
        .map_err(|e| format!("nrd: vkBindImageMemory: {e}"))?;
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
    .map_err(|e| format!("nrd: vkCreateImageView: {e}"))?;
    Ok(Owned { img, view, mem })
}
