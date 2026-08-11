//! The tracer on Vulkan: the reference kernel (`cs_reference` into `accum`,
//! `cs_resolve` into an RGBA16F image), the WAVEFRONT QUADTREE beside it —
//! seed, the indirect level ladder, and the leaf + sky terminal fills — and the
//! HEMISPHERE BOUNCE TIERS the H key cycles (`fb_mode > 0`: the batched hemi
//! wavefront and the one compose splat it feeds).
//!
//! The reference kernel is the smallest thing that can be WRONG in an
//! interesting way, which is why it landed first: every kernel here reads the
//! same streams through the same layout and shades through the same
//! `shade.hlsli`, so if a stream is bound at the wrong slot, a material stride
//! is skewed, the TLAS is built from the wrong addresses, or the cbuffer packs
//! differently under `-fvk-use-dx-layout`, it shows there as a picture that
//! disagrees with the CPU. Having proven that, the ladder can be scored
//! GPU-vs-GPU against it — which is what lets the quadtree's soundness gates
//! demand EXACT agreement rather than a statistical bar.
//!
//! THREE THINGS THE LADDER SPELLS DIFFERENTLY FROM D3D12, and only three:
//!
//! - **The ping-pong is a descriptor SET, not a rebound root UAV.** `qin`/
//!   `qout` (u5/u6) swap every level; D3D12 rewrites two root descriptors,
//!   which Vulkan has no equivalent of. So set 0 is allocated FIVE times off
//!   one layout — LADDER A (u5=qa, u6=qb), LADDER B (the swap), TERMINAL
//!   (u5=cloud_lod, u6=cloud_shadow, the registers' second meaning once the
//!   ladder has drained), and HEMI A/B, which swap the hemi cell queues AND
//!   move three more slots at once: u7 to the hemi leaf queue, u9 to the hemi
//!   cut pool, and t0 back to the BINARY BVH, because the hemi kernels compile
//!   `frustum.hlsli`'s binary `bound_query` deliberately (short queries lose on
//!   the wide tree — measured +35% there against -54% on the tile path). A pass
//!   binds the variant its parity names; one `vkCmdBindDescriptorSets` per
//!   dispatch group, no per-dispatch descriptor traffic.
//!
//!   THE HEMI UNITS RE-DECLARE u5/u6/u7/u9 AS DIFFERENT STRUCTS — `HemiCellRec`
//!   queues where the ladder has `TileRec` ones — and that is NOT a conflict
//!   for the derived map, which keys on descriptor KIND (both are storage
//!   buffers) rather than on the HLSL type. Which is the whole reason the
//!   variants are a per-`(set, register)` OVERRIDE table over one shared base
//!   rather than a second layout: the layout genuinely is the same, and only
//!   the resource behind five of its slots changes.
//! - **Per-dispatch push constants become `vkCmdUpdateBuffer`.** `b1` is a
//!   uniform buffer here (DXC has no flag to promote a cbuffer to push
//!   constants, and `[[vk::push_constant]]` would be an HLSL edit), and the
//!   ladder rewrites it twice per level — which a host write cannot do,
//!   since every host write in a `run()` closure happens before the submit.
//!   An inline transfer update does it at the right point in the stream, and
//!   costs nothing extra: a barrier already sits between every pair of
//!   dispatches. (A dynamic-offset UBO ring is the other shape; it would make
//!   the DERIVED layout special-case one binding, which is worse.)
//! - **There are no resource STATES.** D3D12's `args` transitions
//!   UNORDERED_ACCESS <-> INDIRECT_ARGUMENT around every `ExecuteIndirect`;
//!   Vulkan needs only the execution/memory edge, so one global barrier
//!   covering `COMPUTE|TRANSFER -> COMPUTE|DRAW_INDIRECT` replaces both
//!   transitions and the UAV barrier between them.
//!
//! STRUCTURE REPLAY IS HERE TOO (`render_wavefront_replay`): a bit-equal-basis
//! frame skips the seed and the whole ladder and re-dispatches the persisted
//! terminal queues. The one thing it needs that D3D12 does not is nothing at
//! all — the queues are ordinary buffers with no states to restore — so the
//! only real work was factoring the terminal fills so both paths run the SAME
//! code rather than two similar-looking blocks.
//!
//! `--sw-rays` IS COVERED, and it turned out to want no new set variant at all.
//! That lever swaps every RayQuery body for `rt_sw.hlsli`'s traversal of OUR
//! binary BVH, which reads the tree at t0 and `tri_idx` at t1 — two registers
//! that already MEAN different things per phase here, so the arm is two more
//! OVERRIDES on variants that exist. The ladder keeps the WIDE tree at t0 (a
//! frustum query cannot descend the binary one) and takes `ft_bnode` at t1 for
//! `level_finish`'s leaf-cut translation; the TERMINAL variant — which the
//! leaf, sky AND reference passes all bind — takes the binary tree at t0 and
//! the real `tri_idx` at t1. That is D3D12's own two rebinds, spelled as the
//! difference between two variants instead of as root-descriptor writes.
//!
//! ONE THING THE LEVER DOES NOT BUY HERE, and it is worth saying rather than
//! implying: a device with no ray tracing. The corpus under it declares no
//! acceleration structure — which is exactly why the TLAS write below is
//! guarded on the MAP rather than issued unconditionally — but `VkScene` still
//! builds a BLAS/TLAS nothing reads, so the gate still requires
//! `VK_KHR_ray_query`. Making that conditional is a separate claim, and one
//! neither ICD on this box can test.
//!
//! COMPOSE IS DISPATCHED ONLY UNDER fb, and that is D3D12's rule verbatim
//! rather than an omission: with fb off the leaf and sky passes splat straight
//! into `accum` through `queues.hlsli::accum_splat`, so a compose would be a
//! buffer-to-buffer copy of a full screen.
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

use crate::gfx::frame::{
    fb_mode_of, FrameCb, FrameParams, CB_STRIDE, HEMI_BATCH, HEMI_MAX_DEPTH,
};
use crate::gfx::shaders as gs;
use crate::scene::Scene;
use crate::vk::device::Buffer;
use crate::vk::headless::VkHeadless;
use crate::vk::layout::{self, Layouts};
use crate::vk::reflect::{DescKind, Map};
use crate::vk::scene::VkScene;
use crate::vk::spirv::{binding_of, Reg, Spirv};
use crate::vk::textures::VkTextures;

/// Bytes of a `#[repr(C)]` slice — a reinterpret, not a copy.
fn bytes_of<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// The binary BVH in `gfx::scene`'s wire format — the `--no-ftree` arm's t0
/// stream. Eager because this backend has no staging ring yet; the field
/// mapping itself is shared with D3D12 (`gpu_bvh_node`).
fn nodes_wire(bvh: &crate::bvh::Bvh) -> Vec<crate::gfx::scene::GpuBvhNode> {
    bvh.nodes.iter().map(crate::gfx::scene::gpu_bvh_node).collect()
}

/// A host-visible buffer with `bytes` already in it. The frustum tree is the
/// one stream this file uploads, and it goes the way `vk::scene` sends the
/// rest — mapped, no staging. The staging ring is a throughput question and
/// lands with the rest of them.
fn host_buf(
    vkd: &crate::vk::device::Vk,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<Buffer, String> {
    let b = vkd.buffer(bytes.len().max(4) as u64, usage, true)?;
    vkd.write(&b, bytes)?;
    Ok(b)
}

/// One GPU image, plus what it takes to bind and read it.
struct Image {
    img: vk::Image,
    view: vk::ImageView,
    mem: vk::DeviceMemory,
}

/// The wavefront quadtree's own half: the queues, the indirect args, the
/// frustum structure, the ladder kernels, and the two ping-pong descriptor
/// sets. Kept `Option` so a future device or memory gate is expressible without
/// threading a flag through every method — nothing turns it off today, the last
/// thing that did being `--sw-rays`, which now brings its own streams instead.
struct Wave {
    /// Ping-pong tile queues. Level `d` reads `d % 2 == 0 ? qa : qb`.
    qa: Buffer,
    qb: Buffer,
    qleaf: Buffer,
    qsky: Buffer,
    cut_pool: Buffer,
    /// 16 slots of `VkDispatchIndirectCommand` — same 12-byte layout D3D12's
    /// dispatch command signature consumes, written by `cs_prep`/`cs_prep_mul`.
    args: Buffer,
    /// t0 space0: the 8-wide frustum tree in its QUANTIZED wire format
    /// (`ftree::QFNode`), or the binary BVH under `--no-ftree`. The tracer's
    /// software half — RT cores cannot answer a frustum query.
    tree: Buffer,
    /// t1 space0 — `--sw-rays` + cut consumption + FTREE only: the wide tree's
    /// slot -> binary-node map (the `FNode.bnode` field the quantized wire
    /// format deliberately drops), which `level_finish` reads to translate a
    /// slot-ref leaf cut into the binary node ids the software ray traversal
    /// seeds from. It rides `tri_idx`'s register because that register is DEAD
    /// in every ladder kernel — the same phase-scoped re-meaning u5/u6 take,
    /// and the terminal variant binds the real `tri_idx` there before any ray
    /// fires.
    ft_bnode: Option<Buffer>,
    pipes: [vk::Pipeline; 9],
    /// Set 0 with u5=qa/u6=qb and the swap; index 1 of each is the SAME set-1
    /// handle the terminal variant holds (the scene/texture set never varies).
    sets_a: Vec<vk::DescriptorSet>,
    sets_b: Vec<vk::DescriptorSet>,
    depth_full: u32,
    pub cap_leaf: u32,
    pub cap_sky: u32,
}

/// The hemisphere bounce tiers' own half: the batch-transient cell queues and
/// cut pool, the per-pixel planes the leaf/compose pair passes radiance
/// through, the shading-point queue, the BINARY tree these kernels descend, and
/// two more set-0 variants.
///
/// Separate from `Wave` because it is optional for a different reason: the
/// queues are sized to ONE BATCH of the worst-case cell fan-out
/// (`HEMI_BATCH * 4^(HEMI_MAX_DEPTH-1)` = 1,048,576 records), which is ~290 MB
/// of buffers that a device or a run with no interest in fb should not be
/// asked for. That batch reset IS the memory bound — the whole reason the
/// wavefront is batched at all.
struct Hemi {
    /// Cell ping-pong. The ROOT pass writes `hqout`, so it runs under the ODD
    /// variant (u6 = hq_a) and level 0 then reads hq_a as `hqin`.
    hq_a: Buffer,
    hq_b: Buffer,
    hq_leaf: Buffer,
    cut: Buffer,
    /// u10/u11: the leaf pass's ambient-free color and its `kd` weight — what
    /// makes compose a pure weight x mass multiply (`compose.hlsl`'s header).
    partial: Buffer,
    ambw: Buffer,
    /// u12: the FIXED-POINT accumulator. Integer atomics are order-independent,
    /// which is what makes a queue-driven integrator reproducible run to run.
    hbuf: Buffer,
    /// u13: the shading points. Host-visible because the probe path writes it
    /// from the CPU; the frame path has `cs_leaf` append into it instead.
    pts: Buffer,
    pipes: [vk::Pipeline; 8],
    sets_a: Vec<vk::DescriptorSet>,
    sets_b: Vec<vk::DescriptorSet>,
}

const H_ROOT: usize = 0;
const H_CELL: usize = 1;
const H_LEAF: usize = 2;
const H_COMPOSE: usize = 3;
const H_PREP_BATCH: usize = 4;
const H_SEED_PROBES: usize = 5;
const H_CLEAR_H: usize = 6;
/// `cs_leaf` from `TraceSources::leaf_fb` — the SAME kernel with the hemi arm
/// compiled IN. Two PSOs from one source is a register-pressure decision
/// (`LEAF_NO_FB`), not a feature flag, and it has to be honored here or an fb
/// frame's leaf pass would append no shading points at all.
const H_LEAF_FB: usize = 7;

const ARG_HEMI_ROOT: u32 = 11;
const ARG_HEMI_CELL: u32 = 12;
const ARG_HEMI_LEAF: u32 = 13;

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
    /// The BINARY BVH. TWO consumers, which is why it sits here rather than in
    /// `Hemi` where it started: the hemi kernels descend it in every session
    /// (short queries lose on the wide tree — measured +35% there against -54%
    /// on the tile path), and under `--sw-rays` the software ray loops descend
    /// it too, in the reference, leaf and hemi-leaf units alike. Under
    /// `--no-ftree` it is byte-identical to `Wave::tree`; that duplicate is the
    /// price of `Buffer` being owned rather than shared, and it is a memory
    /// question only.
    binary: Buffer,
    /// `bvh.tri_idx` at t1 space0 — scene triangle ids in leaf slices, the one
    /// stream `rt_sw.hlsli` needs that nothing else in the corpus declares.
    /// `None` without `--sw-rays`, where t1 falls through to the dummy.
    sw_tri: Option<Buffer>,
    hdr: Image,
    samp_lin: vk::Sampler,
    samp_aniso: vk::Sampler,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    layouts: Layouts,
    pipes: [vk::Pipeline; 4], // reference, resolve, sky_lod, cloud_shadow
    wave: Option<Wave>,
    hemi: Option<Hemi>,
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

const W_SEED: usize = 0;
const W_PREP: usize = 1;
const W_PREP_MUL: usize = 2;
const W_CLEAR_INFO: usize = 3;
const W_LEVEL: usize = 4;
const W_LEVEL_WIDE: usize = 5;
const W_LEAF: usize = 6;
const W_SKY: usize = 7;
/// The replay seed: zero every counter EXCEPT the three the terminal fills
/// consume. Not a variant of `cs_seed` — a different kernel with a keep-set,
/// which is what makes "skip the ladder" expressible at all.
const W_SEED_REPLAY: usize = 8;

/// `cs_prep`'s "do not zero any counter" sentinel, and the two args slots the
/// terminal fills use — lockstep with `gpu/trace.rs`'s consts, which are
/// `#[cfg(windows)]`-only.
const NO_RESET: u32 = 0xffff_ffff;
const ARG_LEAF: u32 = 14;
const ARG_SKY: u32 = 15;

impl VkTracer {
    pub fn new(
        hg: &VkHeadless,
        sp: &Spirv,
        scene: &Scene,
        vs: &VkScene,
        vt: &VkTextures,
        bvh: &crate::bvh::Bvh,
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
        // The ladder. Under `--sw-rays` these compile against `rt_sw.hlsli`
        // like everything else and the only difference is which buffer sits
        // behind t0/t1 per phase (module header).
        let wave_units: [(&str, &str, &str); 9] = [
            (&srcs.wavefront, "cs_seed", "wf-seed"),
            (&srcs.wavefront, "cs_prep", "wf-prep"),
            (&srcs.wavefront, "cs_prep_mul", "wf-prep-mul"),
            (&srcs.wavefront, "cs_clear_info", "wf-clear-info"),
            (&srcs.wavefront, "cs_level", "wf-level"),
            (&srcs.wavefront, "cs_level_wide", "wf-level-wide"),
            (&srcs.leaf, "cs_leaf", "wf-leaf"),
            (&srcs.sky, "cs_sky", "wf-sky"),
            (&srcs.wavefront, "cs_seed_replay", "wf-seed-replay"),
        ];
        // The hemisphere tiers. `cs_leaf` appears TWICE across the two tables
        // — once from `leaf` and once from `leaf_fb` — which is the point:
        // they are two PSOs from one source and the fb frame needs the second.
        let hemi_units: [(&str, &str, &str); 8] = [
            (&srcs.hemi_wave, "cs_hemi_root", "hemi-root"),
            (&srcs.hemi_wave, "cs_hemi_cell", "hemi-cell"),
            (&srcs.hemi_leaf, "cs_hemi_leaf", "hemi-leaf"),
            (&srcs.compose, "cs_compose", "compose"),
            (&srcs.wavefront, "cs_prep_batch", "wf-prep-batch"),
            (&srcs.wavefront, "cs_seed_probes", "wf-seed-probes"),
            (&srcs.wavefront, "cs_clear_h", "wf-clear-h"),
            (&srcs.leaf_fb, "cs_leaf", "wf-leaf-fb"),
        ];
        // Nothing turns these off today (see `Wave`'s doc). The flags stay
        // because the compile/allocate/write sites all key off them, so a
        // future gate — a device floor, a memory ceiling — is one expression
        // rather than a rewrite.
        let want_wave = true;
        // fb rides on the ladder: `cs_leaf` is what appends the shading points,
        // and compose reads the planes the leaf pass filled.
        let want_hemi = want_wave;

        // Compile first, reflect the compiled words, THEN build the layout —
        // the M3a order, and the reason there is no register table in this
        // file. The map unions ALL THREE halves, which is what makes the
        // ladder's extra streams (t0's frustum tree, the queues at u5..u9) and
        // the hemi ones (u10..u13) appear in the layout without anything here
        // listing them.
        let mut words: Vec<Vec<u32>> = Vec::new();
        let mut map = Map::default();
        let all: Vec<&(&str, &str, &str)> = units
            .iter()
            .chain(wave_units.iter().take(if want_wave { 9 } else { 0 }))
            .chain(hemi_units.iter().take(if want_hemi { 8 } else { 0 }))
            .collect();
        for (src, entry, tag) in all {
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
        // Said here rather than left to `vkCreateDescriptorSetLayout`: a scene
        // with more textures than the device can bind per stage is a fact
        // about the pair, and the error should name both numbers.
        if tex_cap > vkd.info.max_sampled_images {
            return Err(format!(
                "{} scene texture(s) exceeds this device's maxPerStageDescriptorSampledImages \
                 ({})",
                tex_cap, vkd.info.max_sampled_images
            ));
        }
        let layouts = Layouts::build(vkd, &map, tex_cap, None)?;

        let mut pipes = [vk::Pipeline::null(); 4];
        let mut wpipes = [vk::Pipeline::null(); 9];
        let mut hpipes = [vk::Pipeline::null(); 8];
        {
            let mut built: Vec<vk::Pipeline> = Vec::new();
            let mut make = |i: usize, entry: &str| -> Result<vk::Pipeline, String> {
                let p = layout::compute_pipeline(vkd, &layouts, &words[i], entry)
                    .map_err(|e| format!("{entry}: {e}"))?;
                built.push(p);
                Ok(p)
            };
            let r = (|| -> Result<(), String> {
                for (i, (_, entry, _)) in units.iter().enumerate() {
                    pipes[i] = make(i, entry)?;
                }
                if want_wave {
                    for (j, (_, entry, _)) in wave_units.iter().enumerate() {
                        wpipes[j] = make(units.len() + j, entry)?;
                    }
                }
                if want_hemi {
                    let base = units.len() + wave_units.len();
                    for (j, (_, entry, _)) in hemi_units.iter().enumerate() {
                        hpipes[j] = make(base + j, entry)?;
                    }
                }
                Ok(())
            })();
            if let Err(e) = r {
                for p in &built {
                    unsafe { d.destroy_pipeline(*p, None) };
                }
                layouts.destroy(vkd);
                return Err(e);
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
        // TRANSFER_DST as well as host-visible: the reference path writes this
        // once from the host before its submit, while the ladder rewrites it
        // INSIDE the stream with `vkCmdUpdateBuffer` (module header). Both
        // spellings need to be legal on the one buffer.
        let push = vkd.buffer(16, ub | vk::BufferUsageFlags::TRANSFER_DST, true)?;

        // The ladder's queues, sized to the structural worst case exactly as
        // `TraceGpu::new` sizes them, so `CTR_OVERFLOW` can be gated at 0: at
        // depth d there are at most 4^d rects, internal tiles live at depth
        // < D, every terminal contains at least one depth-D path cell, and a
        // split allocates one cut slot.
        let mut cb_base = FrameCb::base(scene, rw, rh);
        let dd = gs::depth_full(rw, rh);
        // ONE batch's worth of cells and cut slots. `cs_prep_batch` zeroes the
        // hemi counters per batch, so these bound the memory however many
        // shading points a frame produces.
        let cap_hemi_cell = u64::from(HEMI_BATCH) * (1u64 << (2 * (HEMI_MAX_DEPTH - 1)));
        let cap_hemi_cut =
            u64::from(HEMI_BATCH) * (((1u64 << (2 * (HEMI_MAX_DEPTH - 1))) - 1) / 3 + 1);
        // The software half of the intersector, built ahead of both optional
        // halves because both want it: the hemi kernels descend the binary tree
        // in every session, and under `--sw-rays` so do the ray loops in the
        // reference, leaf and hemi-leaf units — which is also the one arm that
        // needs `tri_idx`, since a RayQuery gets its triangles from the driver.
        let binary = host_buf(vkd, bytes_of(&nodes_wire(bvh)), sb)?;
        let sw_tri = if gs::sw_rays() {
            Some(host_buf(vkd, bytes_of(&bvh.tri_idx), sb)?)
        } else {
            None
        };
        let mut wave = if want_wave {
            if dd > 11 {
                return Err(format!("{rw}x{rh} needs quadtree depth {dd} > 11 indirect-arg slots"));
            }
            let cap_tile = if dd >= 1 { 1u64 << (2 * (dd - 1)) } else { 1 };
            let cap_leaf = 1u64 << (2 * dd);
            let cap_cut = ((1u64 << (2 * dd)) - 1) / 3 + 1;
            // `--sw-rays` + FTREE: `level_finish` translates each leaf-emitting
            // split's slot-ref cut into a SECOND fresh slot of binary node ids,
            // so doubling keeps the pool structurally overflow-free. Sizing it
            // is not optional tidiness — the exhaustion arm degrades to ROOT
            // seeding, which is sound but is a different structure, and this
            // backend measured 107 such fallbacks at 800x600 before the
            // doubling landed, against D3D12's 0 on the identical frame.
            let cap_cut = if gs::sw_rays_leaf() && srcs.ftree_on { cap_cut * 2 } else { cap_cut };
            // The kernels read every one of these off the cbuffer, so a
            // buffer sized here and a cap written there must be the same
            // number — hence one place computing both. The hemi three are the
            // SAME arithmetic one structure out: at most one depth-D cell per
            // batched point, and one cut slot per interior split.
            cb_base.set_caps(
                cap_tile as u32,
                cap_leaf as u32,
                cap_leaf as u32,
                cap_cut as u32,
                if want_hemi { cap_hemi_cell as u32 } else { 0 },
                if want_hemi { cap_hemi_cell as u32 } else { 0 },
                if want_hemi { cap_hemi_cut as u32 } else { 0 },
            );
            // t0: the frustum structure the ladder descends. FTREE is the
            // default, and its wire format is the QUANTIZED one — the
            // per-processor split verdict `ftree.rs` documents, not a
            // shortcut: the CPU keeps f32 nodes, the GPU trades decode ALU for
            // -56% tree bandwidth and the decoded boxes still CONTAIN the true
            // ones, so every prune stays conservative.
            let ft = if srcs.ftree_on { Some(crate::ftree::FTree::build(bvh)) } else { None };
            let tree = match &ft {
                Some(f) => host_buf(vkd, bytes_of(&f.quantized()), sb)?,
                None => host_buf(vkd, bytes_of(&nodes_wire(bvh)), sb)?,
            };
            // The leaf-cut translation map. Gated on `sw_rays_leaf` rather than
            // `sw_rays` for the reason the HLSL is: under `--no-cut-rays` the
            // leaf traverses from the root, `level_finish` compiles no
            // translation, and the wavefront unit declares no t1 at all.
            let ft_bnode = match &ft {
                Some(f) if gs::sw_rays_leaf() => {
                    Some(host_buf(vkd, bytes_of(&f.bnode_flat()), sb)?)
                }
                _ => None,
            };
            Some(Wave {
                qa: vkd.buffer(cap_tile * 24, sb, false)?,
                qb: vkd.buffer(cap_tile * 24, sb, false)?,
                qleaf: vkd.buffer(cap_leaf * gs::LEAF_REC_BYTES, sb, false)?,
                qsky: vkd.buffer(cap_leaf * 16, sb, false)?,
                cut_pool: vkd.buffer(cap_cut * 256, sb, false)?,
                args: vkd.buffer(16 * 12, sb | vk::BufferUsageFlags::INDIRECT_BUFFER, false)?,
                tree,
                ft_bnode,
                pipes: wpipes,
                sets_a: Vec::new(),
                sets_b: Vec::new(),
                depth_full: dd,
                cap_leaf: cap_leaf as u32,
                cap_sky: cap_leaf as u32,
            })
        } else {
            None
        };

        let mut hemi = if want_hemi {
            Some(Hemi {
                hq_a: vkd.buffer(cap_hemi_cell * 64, sb, false)?,
                hq_b: vkd.buffer(cap_hemi_cell * 64, sb, false)?,
                hq_leaf: vkd.buffer(cap_hemi_cell * 64, sb, false)?,
                cut: vkd.buffer(cap_hemi_cut * 256, sb, false)?,
                partial: vkd.buffer(px * 12, sb, false)?,
                ambw: vkd.buffer(px * 12, sb, false)?,
                hbuf: vkd.buffer(px * 16, sb, false)?,
                // Host-visible: the probe path writes the shading points from
                // the CPU (`run_hemi_probes`), which is the same no-staging
                // choice `vk::scene` makes for every other stream. A frame's
                // `cs_leaf` appends into it device-side, which host-visible
                // memory serves perfectly well.
                pts: vkd.buffer(px * 32, sb, true)?,
                pipes: hpipes,
                sets_a: Vec::new(),
                sets_b: Vec::new(),
            })
        } else {
            None
        };

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
        // TWO samplers, and the split is `trace_common.hlsli`'s: `samp_lin` is
        // trilinear and takes an explicit ray-cone lod through `SampleLevel`,
        // `samp_aniso` is hardware anisotropic and is fed the elliptical
        // footprint through `SampleGrad`. `--aniso 1` (or a device without the
        // feature) makes the second a copy of the first, which is the
        // isotropic path VERBATIM — the bit-identical off arm, by
        // construction rather than by gate.
        let want = crate::texture::max_aniso();
        let aniso = if vkd.info.sampler_anisotropy {
            want.min(vkd.info.max_anisotropy).max(1.0)
        } else {
            1.0
        };
        let samp_lin = samp(1.0)?;
        let samp_aniso = samp(aniso)?;

        // FIVE allocations of set 0 with everything live — the two ladder
        // parities, the two hemi parities, and the terminal meaning of u5/u6
        // (see the module header). Set 1 is allocated ONCE and shared by all
        // five, because nothing in it varies: every variant is a set-0
        // property.
        let mut want_layouts = layouts.sets.clone();
        for _ in 0..(2 * wave.is_some() as usize + 2 * hemi.is_some() as usize) {
            want_layouts.push(layouts.sets[0]);
        }

        // The pool is sized FROM THE MAP, like everything else here — times
        // the number of set-0 copies, which is the only reason a count here is
        // not simply the map's own.
        let copies = want_layouts.len() as u32;
        let mut counts: std::collections::BTreeMap<vk::DescriptorType, u32> = Default::default();
        for e in map.entries.values() {
            let n = if e.count == 0 { tex_cap } else { e.count };
            *counts.entry(layout::desc_type(e.kind)).or_default() += n * copies;
        }
        let sizes: Vec<vk::DescriptorPoolSize> = counts
            .iter()
            .map(|(&ty, &n)| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(n))
            .collect();
        let pool = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(copies)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| format!("vkCreateDescriptorPool: {e}"))?;
        let all_sets = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&want_layouts),
            )
        }
        .map_err(|e| format!("vkAllocateDescriptorSets: {e}"))?;
        let sets = all_sets[..layouts.sets.len()].to_vec();
        // Each variant is [its own set 0] ++ [the shared tail].
        let mut next = layouts.sets.len();
        let variant = |i: usize| -> Vec<vk::DescriptorSet> {
            std::iter::once(all_sets[i]).chain(sets[1..].iter().copied()).collect()
        };
        if let Some(w) = &mut wave {
            w.sets_a = variant(next);
            w.sets_b = variant(next + 1);
            next += 2;
        }
        if let Some(h) = &mut hemi {
            h.sets_a = variant(next);
            h.sets_b = variant(next + 1);
        }

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
            binary,
            sw_tri,
            hdr,
            samp_lin,
            samp_aniso,
            pool,
            sets,
            layouts,
            pipes,
            wave,
            hemi,
            cb_base,
            scene_aabb: crate::gfx::scene::shadow_aabb(scene),
            sky_lod_k: srcs.sky_lod,
            cloud_shadow_n: srcs.cloud_shadow_n,
            map,
        };
        t.write_descriptors(hg, vs, vt);
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
    /// Every set-0 variant, plus the shared set 1 and the texture table.
    ///
    /// The variants differ only in WHICH RESOURCE sits behind a handful of
    /// registers (see the module header), so this hands `write_variant` an
    /// OVERRIDE list and lets it write the other ~25 streams identically. The
    /// alternative — a "wavefront writer" and a "hemi writer" beside this one —
    /// would be three tables of the same streams to keep in step, which is the
    /// transcription hazard the derived layout exists to remove.
    fn write_descriptors(&self, hg: &VkHeadless, vs: &VkScene, vt: &VkTextures) {
        let drop = std::env::var("FR_VK_DROP_STREAM").ok();
        if let Some(name) = &drop {
            eprintln!(
                "check-vk: FR_VK_DROP_STREAM={name} — that stream is bound to the ZERO \
                 dummy; this run MUST fail"
            );
        }
        // The TERMINAL variant, which is also the only one a reference-kernel
        // frame ever binds: u5/u6 hold the cloud lattice and the slab-space
        // shadow cache, the registers' second meaning.
        //
        // `--sw-rays` adds two more. The reference, leaf and sky passes all
        // bind THIS variant and all three run the software intersector, so t0
        // becomes the BINARY tree (the ladder's wide one is a structure the ray
        // loops cannot descend) and t1 the real `tri_idx`. That pair is D3D12's
        // two rebinds between the ladder and the fills, spelled as a variant
        // difference rather than as root-descriptor writes.
        let mut term: Vec<(Reg, u32, &Buffer)> =
            vec![(Reg::U, 5, &self.cloud_lod), (Reg::U, 6, &self.cloud_shadow)];
        if let Some(t) = &self.sw_tri {
            term.push((Reg::T, 0, &self.binary));
            term.push((Reg::T, 1, t));
        }
        self.write_variant(hg, vs, &self.sets, &term, &drop);
        if let Some(w) = &self.wave {
            // The ladder keeps the WIDE tree at t0 (it comes from the base) and
            // takes the slot -> node map at t1 when the lever arms it.
            for (sets, q0, q1) in [(&w.sets_a, &w.qa, &w.qb), (&w.sets_b, &w.qb, &w.qa)] {
                let mut o: Vec<(Reg, u32, &Buffer)> = vec![(Reg::U, 5, q0), (Reg::U, 6, q1)];
                if let Some(bn) = &w.ft_bnode {
                    o.push((Reg::T, 1, bn));
                }
                self.write_variant(hg, vs, sets, &o, &drop);
            }
        }
        if let Some(h) = &self.hemi {
            // FIVE moved slots, not two: the cell ping-pong, the hemi leaf
            // queue and cut pool at u7/u9, and t0 back to the binary tree —
            // plus `tri_idx` under the lever, since `cs_hemi_leaf` shoots its
            // rays through the same software loops.
            for (sets, q0, q1) in [(&h.sets_a, &h.hq_a, &h.hq_b), (&h.sets_b, &h.hq_b, &h.hq_a)] {
                let mut o: Vec<(Reg, u32, &Buffer)> = vec![
                    (Reg::T, 0, &self.binary),
                    (Reg::U, 5, q0),
                    (Reg::U, 6, q1),
                    (Reg::U, 7, &h.hq_leaf),
                    (Reg::U, 9, &h.cut),
                ];
                if let Some(t) = &self.sw_tri {
                    o.push((Reg::T, 1, t));
                }
                self.write_variant(hg, vs, sets, &o, &drop);
            }
        }
        // `FR_VK_DROP_STREAM=texs` is the whole-run lever; `bind_textures` is
        // also what V6's own anti-vacuity probe calls, so the two share one
        // write path and cannot disagree about what "dropped" means.
        self.bind_textures(hg, vt, drop.as_deref() == Some("texs"));
    }

    fn write_variant(
        &self,
        hg: &VkHeadless,
        vs: &VkScene,
        sets: &[vk::DescriptorSet],
        over: &[(Reg, u32, &Buffer)],
        drop: &Option<String>,
    ) {
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
        //
        // THE OVERRIDES GO FIRST and the lookup takes the FIRST match, which
        // is what makes a variant a diff over one shared base rather than a
        // second table.
        let mut bufs: Vec<(u32, Reg, u32, &Buffer)> =
            over.iter().map(|&(r, n, b)| (0u32, r, n, b)).collect();
        bufs.extend_from_slice(&[
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
        ]);
        if let Some(h) = &self.hemi {
            // The per-pixel planes and the shading-point queue. These do NOT
            // vary by variant — only the queues behind u5..u9 and the tree at
            // t0 do — so they belong in the base beside everything else.
            bufs.extend_from_slice(&[
                (0, Reg::U, 10, &h.partial),
                (0, Reg::U, 11, &h.ambw),
                (0, Reg::U, 12, &h.hbuf),
                (0, Reg::U, 13, &h.pts),
            ]);
        }
        if let Some(w) = &self.wave {
            // The ladder's own streams. t0 is the FRUSTUM structure — the
            // software half RT cores cannot serve — and t1 is `tri_idx`,
            // declared only under `--sw-rays` and so normally absent from the
            // map entirely (it falls through to the dummy if it is not).
            bufs.extend_from_slice(&[
                (0, Reg::T, 0, &w.tree),
                (0, Reg::U, 4, &w.args),
                (0, Reg::U, 7, &w.qleaf),
                (0, Reg::U, 8, &w.qsky),
                (0, Reg::U, 9, &w.cut_pool),
            ]);
        }
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
                    .dst_set(sets[set as usize])
                    .dst_binding(binding)
                    .descriptor_type(ty)
                    .buffer_info(info)
            })
            .collect();

        // The TLAS, and the one storage image the resolve pass writes.
        //
        // GUARDED ON THE MAP, unlike the storage image: `--sw-rays`' corpus
        // declares no acceleration structure at all (`rt_sw.hlsli` traverses
        // our own BVH), so the derived layout has no such binding — and a write
        // to a binding the layout does not have is not a harmless no-op. The
        // sampler writes below have carried this guard since M3a for the same
        // reason; this one was simply never reachable until now.
        let accels = [vs.tlas];
        let mut asw = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&accels);
        let mut w_as = vk::WriteDescriptorSet::default()
            .dst_set(sets[0])
            .dst_binding(binding_of(Reg::T, 7))
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut asw);
        w_as.descriptor_count = 1;
        if self.map.entries.contains_key(&(0, binding_of(Reg::T, 7))) {
            writes.push(w_as);
        }

        let ii = [vk::DescriptorImageInfo::default()
            .image_view(self.hdr.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(sets[0])
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
                        .dst_set(sets[1])
                        .dst_binding(binding_of(Reg::S, *reg))
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&si[i]),
                );
            }
        }
        unsafe { d.update_descriptor_sets(&writes, &[]) };
    }

    /// Write (or rewrite) the `texs[]` array.
    ///
    /// Found by KIND, not by register: it is the corpus's one unbounded
    /// sampled-image array (`Texture2D<float4> texs[]`), and V5's own
    /// anti-vacuity already asserts exactly one exists — so this needs no
    /// `TEX_TABLE_BUFS` literal, which is just as well, since that const lives
    /// in the Windows-only `gpu/trace.rs` and a second copy of it here would
    /// be the transcription M3a exists to avoid.
    ///
    /// `fallback = true` points every entry at the 1x1 WHITE image. That is a
    /// strong perturbation on purpose — albedo goes white everywhere AND alpha
    /// goes opaque, so a cutout scene loses its masks too, i.e. it perturbs
    /// visibility as well as radiance.
    ///
    /// Rebinding mid-session is legal here because `VkHeadless::run` fences
    /// every submit, so nothing is ever pending against these sets.
    pub fn bind_textures(&self, hg: &VkHeadless, vt: &VkTextures, fallback: bool) {
        let Some((set, binding)) = self
            .map
            .entries
            .iter()
            .find(|(_, e)| matches!(e.kind, DescKind::SampledImage) && e.count == 0)
            .map(|(&k, _)| k)
        else {
            return;
        };
        let n = vt.texs.len().max(1);
        let infos: Vec<vk::DescriptorImageInfo> = (0..n)
            .map(|i| {
                let t = if fallback || vt.texs.is_empty() { &vt.fallback } else { &vt.texs[i] };
                vk::DescriptorImageInfo::default()
                    .image_view(t.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
        if std::env::var_os("FR_VK_MAP").is_some() {
            eprintln!(
                "check-vk:   bind set {set} binding {binding} <- {n} sampled image(s){}",
                if fallback { " [the 1x1 fallback]" } else { "" }
            );
        }
        unsafe {
            hg.vk.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.sets[set as usize])
                    .dst_binding(binding)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&infos)],
                &[],
            )
        };
    }

    /// The quadtree depth the ladder runs, and the terminal-queue capacities —
    /// the numbers the accounting gate needs to read the queues back. `None`
    /// under `--sw-rays`, which needs a fourth set variant (module header).
    pub fn wave_shape(&self) -> Option<(u32, u32, u32)> {
        self.wave.as_ref().map(|w| (w.depth_full, w.cap_leaf, w.cap_sky))
    }

    /// The per-frame cbuffer, written host-side before the submit.
    fn write_cb(&self, hg: &VkHeadless, p: &FrameParams) -> Result<(), String> {
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
        hg.vk.write(&self.frame_cb, cb.bytes())
    }

    /// The two per-frame cloud caches, recorded under whichever set has
    /// `cloud_lod`/`cloud_shadow` at u5/u6 — i.e. the TERMINAL variant, never
    /// a ladder one. Shared by the reference frame and the wavefront's
    /// terminal fills, which is what keeps their dispatch shapes identical.
    unsafe fn record_cloud_caches(&self, d: &ash::Device, cmd: vk::CommandBuffer) {
        let sky_pts = ((self.rw / self.sky_lod_k) + 2) * ((self.rh / self.sky_lod_k) + 2);
        let sky_groups = sky_pts.div_ceil(64);
        let csn_groups =
            (crate::clouds::CLOUD_SHADOW_MAX * crate::clouds::CLOUD_SHADOW_MAX).div_ceil(64);
        unsafe {
            if self.cloud_shadow_n > 0 {
                d.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipes[P_CLOUD_SHADOW],
                );
                d.cmd_dispatch(cmd, csn_groups.min(32768), csn_groups.div_ceil(32768), 1);
            }
            if self.sky_lod_k > 1 {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipes[P_SKY_LOD]);
                d.cmd_dispatch(cmd, sky_groups.min(32768), sky_groups.div_ceil(32768), 1);
            }
        }
    }

    /// One frame: the two cache fills, the reference dispatch, then resolve.
    /// The whole thing is one submit — the `HeadlessGpu::run` contract.
    pub fn render(&self, hg: &VkHeadless, p: &FrameParams, samples: u32) -> Result<(), String> {
        let vkd = &hg.vk;
        self.write_cb(hg, p)?;
        // `cbuffer Push : register(b1)` is 4 dwords; only the first is read
        // here (`inv_samples`), but the whole row is written so a slot is
        // never left holding the previous frame's bytes.
        let inv = 1.0f32 / samples.max(1) as f32;
        let mut pb = [0u8; 16];
        pb[..4].copy_from_slice(&inv.to_bits().to_le_bytes());
        vkd.write(&self.push, &pb)?;

        let gx = self.rw.div_ceil(8);
        let gy = self.rh.div_ceil(8);

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

            // Counters are PER FRAME. On D3D12 `cs_seed` zeroes them at the top
            // of the ladder; the reference kernel has no seed, so the fill is
            // explicit — and it must exist at all, because device-local memory
            // starts undefined and a "> 0" must-fire reading uninitialized
            // bytes is a must-fire that passes for free.
            d.cmd_fill_buffer(cmd, self.counters.buf, 0, vk::WHOLE_SIZE, 0);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );

            self.record_cloud_caches(d, cmd);
            barrier(d, cmd);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipes[P_REFERENCE]);
            d.cmd_dispatch(cmd, gx, gy, 1);
            barrier(d, cmd);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipes[P_RESOLVE]);
            d.cmd_dispatch(cmd, gx, gy, 1);
            barrier(d, cmd);
        })
    }

    /// ONE WAVEFRONT QUADTREE FRAME: seed -> depth_full x (prep-args -> level)
    /// -> leaf + sky fills. `record_wavefront`'s peer, minus the arms this
    /// stage does not cover (module header): no hemi, no compose (fb OFF makes
    /// the leaf/sky passes splat straight into `accum`, so compose would be a
    /// buffer-to-buffer copy — D3D12 skips it for the same reason), no replay.
    ///
    /// `clear_sentinel` floods `info` with `0xffffffff` first, which is what
    /// makes the exactly-once coverage gate possible: a pixel no terminal
    /// record covered still reads the sentinel afterwards.
    ///
    /// STATICALLY RECORDED, exactly as on D3D12 — every scheduling decision
    /// after the seed is a GPU-written counter feeding `vkCmdDispatchIndirect`,
    /// so an empty level dispatches zero groups rather than being skipped by
    /// the CPU. That is the property the whole design rests on and the reason
    /// there is no readback anywhere in here.
    pub fn render_wavefront(
        &self,
        hg: &VkHeadless,
        p: &FrameParams,
        clear_sentinel: bool,
    ) -> Result<(), String> {
        let w = self.wave.as_ref().ok_or("this tracer has no wavefront ladder")?;
        self.write_cb(hg, p)?;

        let px = self.rw * self.rh;
        let clear_groups = px.div_ceil(256);
        let wide_on = gs::WIDE_LEVELS_ON.load(std::sync::atomic::Ordering::Relaxed);
        let wide_n = gs::wide_levels();
        // Read from the SAME function that writes the cbuffer's `fb_mode`, so
        // the kernels this records and the constant they branch on cannot
        // disagree about which tier is running.
        let fb_mode = fb_mode_of(&p.q);

        hg.run(|d, cmd| unsafe {
            // Level 0 consumes queue A, so the seed must enqueue its root
            // THERE — bind the A variant before anything runs.
            self.bind(d, cmd, &w.sets_a);
            // push0 = 0: this backend has no work-graph arm, so the seed
            // always enqueues its own root. Written rather than inherited —
            // the buffer holds whatever the last frame left.
            self.push(d, cmd, [0, 0, 0, 0]);
            self.go(d, cmd, w.pipes[W_SEED], 1, 1);
            if clear_sentinel {
                self.go(
                    d,
                    cmd,
                    w.pipes[W_CLEAR_INFO],
                    clear_groups.min(32768),
                    clear_groups.div_ceil(32768),
                );
            }
            self.clear_h(d, cmd, fb_mode);
            barrier(d, cmd);

            for lvl in 0..w.depth_full {
                let (in_ctr, out_ctr) = if lvl % 2 == 0 {
                    (gs::CTR_TILE_A, gs::CTR_TILE_B)
                } else {
                    (gs::CTR_TILE_B, gs::CTR_TILE_A)
                };
                // Shallow levels take the wave-cooperative kernel: ONE GROUP
                // per tile instead of one thread per tile (see WIDE_LEVELS —
                // level 0 is a single tile, so the serial ladder runs one lane
                // over the whole BVH there).
                let wide = wide_on && lvl < wide_n;
                // prep and the level kernel it feeds run under the SAME set:
                // prep touches only `counters` and `args`, which every variant
                // holds identically, so the parity bind covers both.
                self.bind(d, cmd, if lvl % 2 == 0 { &w.sets_a } else { &w.sets_b });
                self.push(d, cmd, [in_ctr, out_ctr, if wide { 1 } else { 32 }, lvl]);
                self.go(d, cmd, w.pipes[W_PREP], 1, 1);
                self.push(d, cmd, [in_ctr, out_ctr, 0, 0]);
                self.go_indirect(
                    d,
                    cmd,
                    w.pipes[if wide { W_LEVEL_WIDE } else { W_LEVEL }],
                    &w.args,
                    lvl,
                );
            }

            self.record_terminal_fills(d, cmd, w, fb_mode);
            self.record_hemi_tail(d, cmd, w, fb_mode, p.q.fb.depth, clear_groups);
        })
    }

    /// Structure replay: a frame whose basis bit-equals the previous producing
    /// frame's re-dispatches the persisted terminal queues and skips the seed
    /// and the WHOLE level ladder — the ladder is the wavefront's fixed cost,
    /// and on a parked camera this deletes it (D3D12 measured -43% of the GPU
    /// frame span there).
    ///
    /// Soundness is entirely in one sentence: the terminal structure is a pure
    /// function of (scene, BVH, basis, rw, rh), while spp/jitter/frame/fb/
    /// quality/clouds all ride the cbuffer — so a replay frame re-shades from a
    /// fresh `FrameParams` against a structure that provably still describes
    /// this view, and the result must be BIT-IDENTICAL to a fresh trace. That
    /// is a gate, not a hope: V9 compares tbuf/info/accum bitwise.
    ///
    /// The queues stay byte-intact between producing frames because only
    /// `cs_seed` and the ladder ever WRITE them — the leaf/sky passes read,
    /// the hemi passes rebind u5..u9 to their own transients, and the
    /// reference/resolve units declare no queues at all.
    ///
    /// THE CALLER PROVES THE BIT-EQUALITY. D3D12 keeps a `last_struct` key and
    /// auto-selects inside `record_frame`; there is no per-frame driver on this
    /// backend yet, so that predicate lands with the presenter rather than
    /// being written here as a field nothing reads.
    pub fn render_wavefront_replay(
        &self,
        hg: &VkHeadless,
        p: &FrameParams,
        clear_sentinel: bool,
    ) -> Result<(), String> {
        let w = self.wave.as_ref().ok_or("this tracer has no wavefront ladder")?;
        self.write_cb(hg, p)?;
        let clear_groups = (self.rw * self.rh).div_ceil(256);
        let fb_mode = fb_mode_of(&p.q);

        hg.run(|d, cmd| unsafe {
            // The terminal variant throughout: nothing here touches the tile
            // queues, so the ladder's A/B binds have no work to do.
            self.bind(d, cmd, &self.sets);
            self.go(d, cmd, w.pipes[W_SEED_REPLAY], 1, 1);
            if clear_sentinel {
                self.go(
                    d,
                    cmd,
                    w.pipes[W_CLEAR_INFO],
                    clear_groups.min(32768),
                    clear_groups.div_ceil(32768),
                );
            }
            self.clear_h(d, cmd, fb_mode);
            barrier(d, cmd);
            self.record_terminal_fills(d, cmd, w, fb_mode);
            self.record_hemi_tail(d, cmd, w, fb_mode, p.q.fb.depth, clear_groups);
        })
    }

    /// The leaf and sky fills, shared by the full path and the replay — which
    /// is what makes "replay re-runs ONLY the terminal fills" a fact about the
    /// code rather than a claim about two similar-looking blocks.
    ///
    /// Runs under the TERMINAL variant: u5/u6 revert to the cloud lattice and
    /// shadow cache, which is exactly what the leaf and sky kernels re-declare
    /// them as.
    unsafe fn record_terminal_fills(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        w: &Wave,
        fb_mode: u32,
    ) {
        unsafe {
            self.bind(d, cmd, &self.sets);
            self.push(d, cmd, [gs::CTR_LEAF, NO_RESET, 1, ARG_LEAF]);
            self.go(d, cmd, w.pipes[W_PREP], 1, 1);
            // Sky takes the MULTIPLYING prep: SKY_SPLIT groups share each
            // record, so one huge proven-empty rect cannot serialize on a
            // single group (that shape was ~70% of the tracer's frame once).
            self.push(d, cmd, [gs::CTR_SKY, NO_RESET, gs::SKY_SPLIT, ARG_SKY]);
            self.go(d, cmd, w.pipes[W_PREP_MUL], 1, 1);
            // Both cloud caches, ahead of BOTH consumers — `cs_sky` on the
            // proven-empty rects and `cs_leaf`'s own miss branch.
            self.record_cloud_caches(d, cmd);
            self.push(d, cmd, [gs::CTR_LEAF, 0, 0, 0]);
            // fb frames take the OTHER leaf PSO — same source, hemi arm
            // compiled IN — because that arm is what appends the shading points
            // the hemisphere passes below consume. Sharing one PSO here would
            // leave the hemi queue empty and every gate below vacuous.
            let leaf_pipe = match (&self.hemi, fb_mode > 0) {
                (Some(h), true) => h.pipes[H_LEAF_FB],
                _ => w.pipes[W_LEAF],
            };
            self.go_indirect(d, cmd, leaf_pipe, &w.args, ARG_LEAF);
            self.push(d, cmd, [gs::CTR_SKY, 0, 0, 0]);
            self.go_indirect(d, cmd, w.pipes[W_SKY], &w.args, ARG_SKY);
            barrier(d, cmd);
        }
    }

    /// Zero the fixed-point H accumulator, once per fb FRAME.
    ///
    /// Mandatory rather than tidy, and it was missing from the frame path until
    /// the replay factoring put the two next to each other: `hbuf` is written
    /// by ATOMIC ADD (that is what makes the integrator order-independent), so
    /// an unzeroed frame integrates on top of the previous one's answer — and
    /// nothing downstream can tell, because compose folds whatever is there
    /// into `accum` and V8's frame half scores accounting, not radiance.
    unsafe fn clear_h(&self, d: &ash::Device, cmd: vk::CommandBuffer, fb_mode: u32) {
        let Some(h) = &self.hemi else { return };
        if fb_mode == 0 {
            return;
        }
        let g = (self.rw * self.rh * 4).div_ceil(256);
        unsafe { self.go(d, cmd, h.pipes[H_CLEAR_H], g.min(32768), g.div_ceil(32768)) };
    }

    /// The hemisphere wavefront plus its one compose splat, shared for the same
    /// reason. No-op with fb off, which is D3D12's rule verbatim: with no
    /// bounce tier there is no ambient term to fold in, and compose would
    /// degenerate to a full-screen buffer-to-buffer copy.
    unsafe fn record_hemi_tail(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        w: &Wave,
        fb_mode: u32,
        fb_depth: u32,
        clear_groups: u32,
    ) {
        let Some(h) = &self.hemi else { return };
        if fb_mode == 0 {
            return;
        }
        unsafe {
            // Every hit pixel appended a point, so batch over the WORST CASE;
            // batches past the GPU-side count dispatch zero groups, which is
            // what lets this be recorded statically with no readback anywhere.
            self.record_hemi(d, cmd, h, w, self.rw * self.rh, fb_depth);
            // partial + ambW * ambient(H) -> accum: the single splat, and the
            // ONE pass in the tracer that is per-PIXEL rather than queue-driven.
            self.bind(d, cmd, &self.sets);
            self.go(
                d,
                cmd,
                h.pipes[H_COMPOSE],
                clear_groups.min(32768),
                clear_groups.div_ceil(32768),
            );
            barrier(d, cmd);
        }
    }

    /// The hemisphere wavefront over the points in `hemi_pts`, in `HEMI_BATCH`
    /// slices — `record_hemi`'s peer. Each batch resets the transient cell
    /// queues and cut pool (`cs_prep_batch`), and THAT reset is what bounds the
    /// memory: the caps size one batch, not one frame.
    ///
    /// The parity dance is one step off the ladder's, deliberately: the ROOT
    /// pass writes `hqout`, so it runs under the ODD variant (u6 = hq_a) and
    /// level 0 then reads hq_a as `hqin` under the EVEN one. Getting that
    /// backwards costs the whole batch silently — the root's output lands in a
    /// queue nothing reads.
    unsafe fn record_hemi(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        h: &Hemi,
        w: &Wave,
        max_points: u32,
        fb_depth: u32,
    ) {
        let n_batches = max_points.div_ceil(HEMI_BATCH);
        let levels = fb_depth.clamp(2, HEMI_MAX_DEPTH) - 1;
        unsafe {
            for b in 0..n_batches {
                let base = b * HEMI_BATCH;
                self.bind(d, cmd, &h.sets_b);
                // Batch prep: the root pass's args PLUS the batch-scoped
                // counter reset — one kernel, because the reset has to happen
                // before anything in the batch enqueues.
                self.push(d, cmd, [gs::CTR_HEMI_PT, base, 32, ARG_HEMI_ROOT]);
                self.go(d, cmd, h.pipes[H_PREP_BATCH], 1, 1);
                self.push(d, cmd, [base, gs::CTR_HEMI_A, 0, 0]);
                self.go_indirect(d, cmd, h.pipes[H_ROOT], &w.args, ARG_HEMI_ROOT);

                for l in 0..levels {
                    let (in_ctr, out_ctr) = if l % 2 == 0 {
                        (gs::CTR_HEMI_A, gs::CTR_HEMI_B)
                    } else {
                        (gs::CTR_HEMI_B, gs::CTR_HEMI_A)
                    };
                    self.bind(d, cmd, if l % 2 == 0 { &h.sets_a } else { &h.sets_b });
                    self.push(d, cmd, [in_ctr, out_ctr, 32, ARG_HEMI_CELL]);
                    self.go(d, cmd, w.pipes[W_PREP], 1, 1);
                    self.push(d, cmd, [in_ctr, out_ctr, 0, 0]);
                    self.go_indirect(d, cmd, h.pipes[H_CELL], &w.args, ARG_HEMI_CELL);
                }

                // Leaf rays: FOUR threads per leaf cell (one stratified Arvo
                // ray per midpoint sub-cell), so 8 records per 32-wide group.
                self.push(d, cmd, [gs::CTR_HEMI_LEAF, NO_RESET, 8, ARG_HEMI_LEAF]);
                self.go(d, cmd, w.pipes[W_PREP], 1, 1);
                self.push(d, cmd, [0, 0, 0, 0]);
                self.go_indirect(d, cmd, h.pipes[H_LEAF], &w.args, ARG_HEMI_LEAF);
            }
        }
    }

    /// `vkCmdBindDescriptorSets` for one set-0 variant plus the shared tail.
    unsafe fn bind(&self, d: &ash::Device, cmd: vk::CommandBuffer, sets: &[vk::DescriptorSet]) {
        unsafe {
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layouts.pipeline,
                0,
                sets,
                &[],
            );
        }
    }

    /// `cbuffer Push : register(b1)`, rewritten IN THE STREAM (module header).
    ///
    /// IT CARRIES ITS OWN BARRIERS, BOTH OF THEM, and that is the whole reason
    /// it is one function rather than three lines at each of its two dozen call
    /// sites: a per-dispatch constant block needs a WRITE-AFTER-READ edge as
    /// well as the obvious read-after-write one, and the WAR edge is the one
    /// that is easy to omit and invisible when you do.
    ///
    /// Omitting it cost the entire ladder past level 0 on the first run,
    /// silently: the transfer is free to execute ahead of the dispatch it
    /// textually FOLLOWS, so `cs_prep` read the NEXT level's `push3`, wrote its
    /// indirect args to the wrong slot, and every level after the first
    /// dispatched zero groups. Nothing faulted, validation was clean, and the
    /// frame simply came back with one split and no terminals.
    unsafe fn push(&self, d: &ash::Device, cmd: vk::CommandBuffer, v: [u32; 4]) {
        unsafe {
            barrier(d, cmd); // WAR: the last dispatch must be done READING b1
            let mut b = [0u8; 16];
            for (i, x) in v.iter().enumerate() {
                b[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
            d.cmd_update_buffer(cmd, self.push.buf, 0, &b);
            barrier(d, cmd); // RAW: and the next one must SEE the write
        }
    }

    unsafe fn go(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        pipe: vk::Pipeline,
        gx: u32,
        gy: u32,
    ) {
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            d.cmd_dispatch(cmd, gx, gy, 1);
        }
    }

    unsafe fn go_indirect(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        pipe: vk::Pipeline,
        args: &Buffer,
        slot: u32,
    ) {
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            d.cmd_dispatch_indirect(cmd, args.buf, u64::from(slot) * 12);
        }
    }

    /// The `--check-vk` probe path: upload a CPU-generated shading-point set and
    /// run ONLY the hemisphere passes over it — `run_hemi_probes`' peer.
    ///
    /// Both sides of the A/B then integrate at the EXACT same `(o, n)`, which is
    /// what makes a statistical comparison against a CPU cosine reference mean
    /// anything. The CB `frame` seeds the Arvo draws, so calling again with
    /// `clear = false` and a different frame ACCUMULATES another independent
    /// estimate into H — and `cs_seed_probes` keeps the verify counters across
    /// those passes deliberately, so the exact-zero gates observe every seed's
    /// rays rather than only the last seed's.
    pub fn run_hemi_probes(
        &self,
        hg: &VkHeadless,
        p: &FrameParams,
        probes: &[(glam::Vec3A, glam::Vec3A)],
        clear: bool,
    ) -> Result<(), String> {
        let h = self.hemi.as_ref().ok_or("this tracer has no hemisphere tiers")?;
        let w = self.wave.as_ref().ok_or("this tracer has no wavefront ladder")?;
        if probes.len() > (self.rw * self.rh) as usize {
            return Err(format!("{} probes exceeds the hbuf/pts capacity", probes.len()));
        }
        self.write_cb(hg, p)?;

        // `HemiPointRec` — o | pixel | n | pad. `pixel` is the probe INDEX, so
        // probe i's estimate lands at hbuf[i].
        let mut bytes = Vec::with_capacity(probes.len() * 32);
        for (i, (o, n)) in probes.iter().enumerate() {
            for v in [o.x, o.y, o.z] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes.extend_from_slice(&(i as u32).to_le_bytes());
            for v in [n.x, n.y, n.z] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }
        hg.vk.write(&h.pts, &bytes)?;

        let n = probes.len() as u32;
        let clear_groups = (self.rw * self.rh * 4).div_ceil(256);
        hg.run(|d, cmd| unsafe {
            // Any variant would serve these two — they touch `counters` and
            // `hbuf`, which every variant holds identically — but binding the
            // one the batch loop starts under keeps the stream readable.
            self.bind(d, cmd, &h.sets_b);
            self.push(d, cmd, [n, u32::from(clear), 0, 0]);
            self.go(d, cmd, h.pipes[H_SEED_PROBES], 1, 1);
            if clear {
                self.go(
                    d,
                    cmd,
                    h.pipes[H_CLEAR_H],
                    clear_groups.min(32768),
                    clear_groups.div_ceil(32768),
                );
            }
            barrier(d, cmd);
            self.record_hemi(d, cmd, h, w, n, p.q.fb.depth);
            barrier(d, cmd);
        })
    }

    /// The fixed-point H accumulator, as raw u32 — 4 per point (`x|y|z|psa`).
    pub fn read_hbuf(&self, hg: &VkHeadless, n_points: usize) -> Result<Vec<u32>, String> {
        let h = self.hemi.as_ref().ok_or("this tracer has no hemisphere tiers")?;
        let b = hg.read_buffer(&h.hbuf, n_points * 4 * 4)?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// Are the hemisphere tiers live? Always, today — see `Wave`'s doc on why
    /// the `Option` stays.
    pub fn has_hemi(&self) -> bool {
        self.hemi.is_some()
    }

    /// The terminal queues, as raw u32 — the accounting gate's input.
    /// `LeafRec` is `LEAF_REC_BYTES`, `SkyRec` is 16.
    pub fn read_queues(
        &self,
        hg: &VkHeadless,
        n_leaf: usize,
        n_sky: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        let w = self.wave.as_ref().ok_or("this tracer has no wavefront ladder")?;
        let u32s = |b: Vec<u8>| -> Vec<u32> {
            b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        // Clamped: on overflow the counters keep incrementing past the record
        // writes, and the point is to reach the CTR_OVERFLOW failure with a
        // diagnostic rather than to die reading out of bounds here.
        let nl = n_leaf.min(w.cap_leaf as usize) * (gs::LEAF_REC_BYTES / 4) as usize;
        let ns = n_sky.min(w.cap_sky as usize) * 4;
        let leaf = if nl == 0 { Vec::new() } else { u32s(hg.read_buffer(&w.qleaf, nl * 4)?) };
        let sky = if ns == 0 { Vec::new() } else { u32s(hg.read_buffer(&w.qsky, ns * 4)?) };
        Ok((leaf, sky))
    }

    /// The resolved RGBA16F image, decoded to f32 RGB — the `read_hdr_output`
    /// peer, and the only thing in this file that proves the storage image was
    /// ever written.
    /// The frame's counter block. Zeroed at the top of every `render`, so this
    /// describes the LAST frame rather than the run — which is what a
    /// "the path fired" must-fire wants.
    pub fn read_counters(&self, hg: &VkHeadless) -> Result<Vec<u32>, String> {
        let n = gs::CTR_TOTAL as usize;
        let b = hg.read_buffer(&self.counters, n * 4)?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

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
            for p in self
                .pipes
                .iter()
                .chain(self.wave.iter().flat_map(|w| w.pipes.iter()))
                .chain(self.hemi.iter().flat_map(|h| h.pipes.iter()))
            {
                d.destroy_pipeline(*p, None);
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
            &self.binary,
        ] {
            vkd.free_buffer(b);
        }
        for b in self.sw_tri.iter() {
            vkd.free_buffer(b);
        }
        if let Some(w) = &self.wave {
            for b in [&w.qa, &w.qb, &w.qleaf, &w.qsky, &w.cut_pool, &w.args, &w.tree]
                .into_iter()
                .chain(w.ft_bnode.iter())
            {
                vkd.free_buffer(b);
            }
        }
        if let Some(h) = &self.hemi {
            for b in [&h.hq_a, &h.hq_b, &h.hq_leaf, &h.cut, &h.partial, &h.ambw, &h.hbuf, &h.pts] {
                vkd.free_buffer(b);
            }
        }
    }
}

/// The one memory edge every pass here needs, in one place.
///
/// This replaces THREE D3D12 constructs at once: the UAV barrier, and the
/// `args` buffer's UNORDERED_ACCESS <-> INDIRECT_ARGUMENT transition pair
/// around every `ExecuteIndirect`. Vulkan has no resource states, so what is
/// left is the execution/memory dependency — and making it global rather than
/// per-buffer is deliberate: the ladder's dependency graph is
/// "everything before this dispatch, then this dispatch", every stage of it,
/// so per-resource barriers would be a longer statement of the same thing with
/// somewhere for a missing edge to hide.
///
/// `TRANSFER` is on BOTH sides because the ladder rewrites `b1` with
/// `vkCmdUpdateBuffer` between dispatches; `DRAW_INDIRECT` is the stage that
/// reads `args`, which is the one place a compute-only pipeline still touches
/// a graphics-sounding stage bit.
fn barrier(d: &ash::Device, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::UNIFORM_READ
                | vk::AccessFlags::INDIRECT_COMMAND_READ
                | vk::AccessFlags::TRANSFER_WRITE,
        );
    unsafe {
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::TRANSFER
                | vk::PipelineStageFlags::DRAW_INDIRECT,
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
