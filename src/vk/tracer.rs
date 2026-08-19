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
    fb_mode_of, FrameCb, FrameParams, CB_STRIDE, GBUF_EXT_STRIDE, GBUF_STRIDE, HEMI_BATCH,
    HEMI_MAX_DEPTH,
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

/// Upload bytes of the software trees this file owns — the sizing input for
/// their staging ring. The wide tree is the big one: ~112 B per wide node
/// against the binary tree's 32, and a scene's wide nodes are roughly a
/// seventh of its binary ones, so at 34.4M triangles this is ~480 MB.
fn tree_bytes(bvh: &crate::bvh::Bvh, ft: Option<&crate::ftree::FTree>, sw: bool) -> u64 {
    let binary = (bvh.nodes.len() * std::mem::size_of::<crate::gfx::scene::GpuBvhNode>()) as u64;
    let tree = match ft {
        Some(f) => f.quantized_bytes() as u64,
        None => binary,
    };
    binary + tree + if sw { (bvh.tri_idx.len() * 4) as u64 } else { 0 }
}

/// One GPU image, plus what it takes to bind and read it.
struct Image {
    img: vk::Image,
    view: vk::ImageView,
    mem: vk::DeviceMemory,
}

// The NRD plane ordering — `NrdGpu`'s own (gpu/nrd_gpu.rs's `P_*`), minus the
// validation plane, which is a debug dump this backend has no lever for yet.
const N_IN_MV: usize = 0;
const N_IN_NR: usize = 1;
const N_IN_VIEWZ: usize = 2;
const N_IN_DIFF: usize = 3;
const N_IN_SPEC: usize = 4;
const N_OUT_DIFF: usize = 5;
const N_OUT_SPEC: usize = 6;
const N_COUNT: usize = 7;

/// `(bridge register, plane)` — the ONE place `nrd_bridge.hlsl`'s register map
/// is transcribed on this backend, and it is a transcription of the SHADER,
/// which is the only thing that can declare it. `u16`/`u18`/`u19` are absent on
/// purpose: those are the engine's colour and guide planes, owned by whoever
/// runs the upscaler, and `wire_feed` already points them at its images.
const NRD_REGS: [(u32, usize); N_COUNT] = [
    (17, N_IN_NR),
    (20, N_OUT_SPEC),
    (23, N_IN_MV),
    (24, N_IN_DIFF),
    (25, N_IN_SPEC),
    (26, N_IN_VIEWZ),
    (27, N_OUT_DIFF),
];

/// The NRD bridge: seven denoiser planes and the two kernels that pack into
/// and recompose from them.
///
/// The planes carry `NrdGpu`'s formats exactly, because the bridge is
/// deliberately ENGINE-BLIND — `wire_nrd_feed`'s own doc on the D3D12 side says
/// the kernels "neither know nor care which engine sits between them" — so a
/// plane whose format drifted here would be a plane a real NRD instance could
/// not be handed.
///
/// ONE DIFFERENCE FROM D3D12, and it deletes code rather than adding it: every
/// plane rests in `GENERAL` for its whole life. That layout is legal for both
/// `SAMPLED_IMAGE` and `STORAGE_IMAGE`, so the pass sequence needs only memory
/// barriers and never a layout transition — D3D12's NPSR<->UA bracketing has no
/// counterpart here and must not be invented. The price is that GENERAL can
/// disable framebuffer compression on some hardware, which is irrelevant to a
/// gate and is the trade every compute-only engine makes.
struct Nrd {
    planes: [Image; N_COUNT],
    /// `[cs_nrd_pack, cs_nrd_out]`.
    pipes: [vk::Pipeline; 2],
    /// The planes are created UNDEFINED, and the transition to GENERAL must
    /// happen EXACTLY ONCE: `UNDEFINED` as an old layout licenses the driver to
    /// discard contents, so re-transitioning per dispatch would wipe the pack
    /// on the way to the recompose that reads it.
    laid_out: std::cell::Cell<bool>,
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
    /// 16 slots of `VkDispatchIndirectCommand` — same `gs::ARG_STRIDE` layout
    /// D3D12's dispatch command signature consumes, written by
    /// `cs_prep`/`cs_prep_mul`.
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
    /// The G-buffer pack, `dlss::GBufs` interleaved on the GPU: CORE at u15
    /// (16 B/px — mv.xy | view_z | prev_z) and the guide/signal half at u32
    /// (72 B/px). Full-size only when `gbuf_full`; otherwise a stride-sized
    /// minimum, exactly as D3D12 sizes them, because `FLAG_GBUF` is the ONLY
    /// thing standing between the stores and an out-of-bounds write — a
    /// storage buffer bound with `WHOLE_SIZE` has no bounds check, and the
    /// alternative stand-in (`VkScene::dummy`) is SIXTEEN BYTES.
    pub gbuf: Buffer,
    pub gbuf_ext: Buffer,
    /// Whether the two above are full-size. Read by `write_cb` — it is what
    /// arms `FLAG_GBUF`, so it cannot drift from the allocation above.
    gbuf_full: bool,
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
    /// `cs_feed_xess`, under `TracerOpts::feed`. `None` is what makes
    /// `record_feed` refuse rather than dispatch a null pipeline.
    feed_pipe: Option<vk::Pipeline>,
    /// The bridge, under `TracerOpts::nrd`. `None` is what makes
    /// `record_nrd_*` refuse rather than dispatch a null pipeline.
    nrd: Option<Nrd>,
    /// Arm the pack's sig lanes on the frames this tracer records.
    ///
    /// A PER-FRAME CELL, not a construction flag, and that is what makes the
    /// invariance gateable: `FLAG_FSR_SIG` and its dependents are cbuffer bits,
    /// `accum` is bit-identical across them (the capture is assignment-only —
    /// D3D12's N6b asserts exactly this), so two traces one bit apart need no
    /// recompile and no second tracer. Defaults to the bridge's presence.
    nrd_sig: std::cell::Cell<bool>,
    wave: Option<Wave>,
    hemi: Option<Hemi>,
    cb_base: FrameCb,
    scene_aabb: ([f32; 3], [f32; 3]),
    sky_lod_k: u32,
    cloud_shadow_n: u32,
    /// The derived register map, kept so `bind()` writes exactly the slots the
    /// modules declared — never a hand-listed set.
    map: Map,
    /// Bytes staged and submits spent uploading the software trees. Reported,
    /// never asserted — a run that staged nothing and one that staged the
    /// whole tree allocate the same buffers, and only a byte total tells them
    /// apart (the `--check-spirv` assembled-bytes lesson).
    pub staged: (u64, u32),
}

/// What this tracer is BUILT to do, as opposed to what a frame asks it for.
///
/// A struct rather than more positionals, and the reason is written into
/// `new`'s own comment history: the argument list reached nine with the pack
/// (B2), that comment warned the pile was the hazard and that the next flag
/// had to be decided at the same site, and the feed (B3) is that next flag.
/// Both fields are CONSTRUCTION properties for the same underlying reason —
/// each sizes or compiles something a per-frame choice could then outrun.
#[derive(Clone, Copy, Default)]
pub struct TracerOpts {
    /// Size the G-buffer pack full and arm `FLAG_GBUF` on every frame this
    /// tracer records — the D3D12 `TraceGpu::new` argument of the same name.
    /// The buffers are sized by it, so it cannot be a per-frame choice without
    /// the stores outrunning their allocation; with `gbuf_full` false the pair
    /// is ONE stride and the flag is the only thing between the ext store and
    /// an out-of-bounds write of the whole screen.
    pub gbuf_full: bool,
    /// Compile `cs_feed_xess` and carry its three storage-image bindings.
    ///
    /// OPTIONAL rather than always-on because the map is derived: compiling
    /// the unit unconditionally would move V5's slot count (46 -> 49) and every
    /// tracer's descriptor-pool sizing, i.e. it would move recorded numbers for
    /// stages that never dispatch a feed. It IMPLIES `gbuf_full` — the kernel
    /// reads the pack — and `new` forces that rather than trusting the caller.
    pub feed: bool,
    /// Compile the NRD BRIDGE (`cs_nrd_pack` / `cs_nrd_out`) and allocate the
    /// seven denoiser planes it packs into and recomposes from.
    ///
    /// The bridge joins the TRACER family rather than needing one of its own,
    /// and that is a property of the source rather than a convenience:
    /// `TraceSources::nrd_bridge()` pastes `trace_common.hlsli`, so it declares
    /// the tracer's registers plus `u17`, `u20`, `u23`..`u27` — all free — and
    /// `u16`/`u18`/`u19`, which are the feed's images at the SAME descriptor
    /// kind. It declares no `t` register at all. Hence one more unit, not a
    /// second layout.
    ///
    /// It IMPLIES `feed`, and not merely because both want `gbuf_full`: the
    /// bridge's whole claim is that its recompose reproduces what the feed
    /// wrote, so the feed's colour plane is the only reference V14 can score it
    /// against. Forced in `new` for the same reason `feed` forces `gbuf_full`.
    pub nrd: bool,
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
        opts: TracerOpts,
    ) -> Result<VkTracer, String> {
        let vkd = &hg.vk;
        let d = &vkd.device;
        // The feed kernel reads `gbuf`, so a feed over a stride-sized pack
        // would read one texel's worth of allocation for every pixel. FORCED
        // rather than asserted: the two flags are not independent, and the
        // caller that gets it wrong is asking for the feed, which is the
        // stronger request. The bridge wants both — `cs_nrd_pack` reads the
        // pack's sig lanes, and its recompose is only checkable against the
        // feed's own colour plane.
        let want_feed = opts.feed || opts.nrd;
        let gbuf_full = opts.gbuf_full || want_feed;

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
        // The feed. ONE entry point of the several `feed.hlsl` declares, which
        // is what keeps its footprint to three bindings: DXC drops every
        // `feed_*` declaration `cs_feed_xess` does not reference, so the unit
        // contributes exactly u16 (RGBA16F colour), u18 (R32F reversed-Z clip
        // depth) and u19 (RG16F mvec). Its other two inputs — `accum` at u0 and
        // `gbuf` at u15, the latter from `trace_common.hlsli` — are the SAME
        // declarations the tracer already binds, so they join no new slot.
        let feed_units: [(&str, &str, &str); 1] = [(&srcs.feed, "cs_feed_xess", "feed-xess")];
        // The NRD bridge. `nrd_bridge()` is a METHOD rather than a keyed
        // `Option` field on this side (the DXR pipeline's `DxrSources` carries
        // the Option), so nothing in the shared core had to change for this —
        // it is assembled on demand and kept alive for the borrows below.
        //
        // TWO entry points from ONE source, the `frd_blur.hlsl` shape: the pack
        // and the recompose are the bridge's front and back half and share
        // every declaration between them.
        let nrd_src = opts.nrd.then(|| srcs.nrd_bridge()).unwrap_or_default();
        let nrd_units: [(&str, &str, &str); 2] =
            [(&nrd_src, "cs_nrd_pack", "nrd-pack"), (&nrd_src, "cs_nrd_out", "nrd-out")];
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
            .chain(feed_units.iter().take(if want_feed { 1 } else { 0 }))
            .chain(nrd_units.iter().take(if opts.nrd { 2 } else { 0 }))
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
        let mut feed_pipe: Option<vk::Pipeline> = None;
        let mut nrd_pipes: Option<[vk::Pipeline; 2]> = None;
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
                let mut i = units.len() + wave_units.len() + hemi_units.len();
                if want_feed {
                    // Its index is the END of `all`, which is where the chain
                    // above put it — the same positional agreement the wave
                    // and hemi blocks rely on.
                    feed_pipe = Some(make(i, feed_units[0].1)?);
                    i += 1;
                }
                if opts.nrd {
                    nrd_pipes =
                        Some([make(i, nrd_units[0].1)?, make(i + 1, nrd_units[1].1)?]);
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
        // The pack. Full-size only under `gbuf_full`; otherwise ONE stride, the
        // D3D12 sizing verbatim (gpu/trace.rs's `committed_buffer` pair). The
        // ext half is allocated whenever the core one is, NOT only when a
        // guide-consuming kind is wired — D3D12's own reason applies here
        // unchanged: the consumer arrives after construction, so the buffer
        // must always be able to receive the stores, and `FLAG_GBUF_EXT` gates
        // the WRITES, which is where the cost is.
        let gbuf = vkd.buffer(if gbuf_full { px * GBUF_STRIDE } else { GBUF_STRIDE }, sb, false)?;
        let gbuf_ext =
            vkd.buffer(if gbuf_full { px * GBUF_EXT_STRIDE } else { GBUF_EXT_STRIDE }, sb, false)?;
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
        //
        // These stream through `vk::stage` exactly as the scene's do, and
        // GENERATED rather than collected for the reason that matters at
        // scale: `nodes_wire`'s `Vec<GpuBvhNode>` was ~960 MB at 34.4M
        // triangles and the wide tree's `quantized()` another ~480, each of
        // them a full second copy on top of the mapped buffer it was written
        // into. Per-node conversion is the shared thing (`gpu_bvh_node`,
        // `FTree::quantize_node`); the iteration is each backend's own.
        let ft = if srcs.ftree_on { Some(crate::ftree::FTree::build(bvh)) } else { None };
        let mut stage = crate::vk::stage::Stage::new(
            vkd,
            tree_bytes(bvh, ft.as_ref(), gs::sw_rays()),
            256,
        )?;
        let trees = (|| -> Result<(Buffer, Option<Buffer>), String> {
            let binary = stage.stream(hg, &bvh.nodes, crate::gfx::scene::gpu_bvh_node, sb)?;
            let sw_tri = if gs::sw_rays() {
                Some(stage.stream(hg, &bvh.tri_idx, |t| *t, sb)?)
            } else {
                None
            };
            Ok((binary, sw_tri))
        })();
        let (binary, sw_tri) = match trees {
            Ok(t) => t,
            Err(e) => {
                stage.free(vkd);
                return Err(e);
            }
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
            let tree = match &ft {
                Some(f) => stage.stream_gen(hg, f.nodes.len(), |i| f.quantize_node(i), sb)?,
                None => stage.stream(hg, &bvh.nodes, crate::gfx::scene::gpu_bvh_node, sb)?,
            };
            // The leaf-cut translation map. Gated on `sw_rays_leaf` rather than
            // `sw_rays` for the reason the HLSL is: under `--no-cut-rays` the
            // leaf traverses from the root, `level_finish` compiles no
            // translation, and the wavefront unit declares no t1 at all.
            let ft_bnode = match &ft {
                Some(f) if gs::sw_rays_leaf() => {
                    Some(stage.stream_gen(hg, f.nodes.len() * 8, |i| f.bnode_at(i), sb)?)
                }
                _ => None,
            };
            Some(Wave {
                qa: vkd.buffer(cap_tile * 24, sb, false)?,
                qb: vkd.buffer(cap_tile * 24, sb, false)?,
                qleaf: vkd.buffer(cap_leaf * gs::LEAF_REC_BYTES, sb, false)?,
                qsky: vkd.buffer(cap_leaf * 16, sb, false)?,
                cut_pool: vkd.buffer(cap_cut * 256, sb, false)?,
                args: vkd.buffer(
                    16 * gs::ARG_STRIDE,
                    sb | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    false,
                )?,
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
        // Every tree that was going to be uploaded has been; the ring is a
        // 64 MB mapped allocation with nothing left to carry.
        let staged = (stage.bytes(), stage.chunks());
        stage.free(vkd);

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
        // The bridge's planes. Formats are `NrdGpu`'s, verbatim — see `Nrd`.
        let nrd = match nrd_pipes {
            None => None,
            Some(pipes) => {
                let f16x4 = vk::Format::R16G16B16A16_SFLOAT;
                // NRD enc-2 packs the normal's L1-octahedral pair plus
                // roughness into 10:10:10:2. Vulkan spells D3D12's
                // R10G10B10A2_UNORM as A2B10G10R10_UNORM_PACK32 — the
                // component ORDER in the name is reversed because one names
                // memory order and the other names the packed word's high bits
                // down, and they describe the same bytes.
                let fmts = [
                    f16x4,                                      // in_mv
                    vk::Format::A2B10G10R10_UNORM_PACK32,       // in_nr
                    vk::Format::R32_SFLOAT,                     // in_viewz
                    f16x4,                                      // in_diff
                    f16x4,                                      // in_spec
                    f16x4,                                      // out_diff
                    f16x4,                                      // out_spec
                ];
                let mut planes: Vec<Image> = Vec::with_capacity(N_COUNT);
                for f in fmts {
                    planes.push(create_image_fmt(vkd, rw, rh, f)?);
                }
                Some(Nrd {
                    planes: planes.try_into().map_err(|_| "nrd plane count")?,
                    pipes,
                    laid_out: std::cell::Cell::new(false),
                })
            }
        };
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
            gbuf,
            gbuf_ext,
            gbuf_full,
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
            feed_pipe,
            nrd_sig: std::cell::Cell::new(nrd.is_some()),
            nrd,
            wave,
            hemi,
            cb_base,
            scene_aabb: crate::gfx::scene::shadow_aabb(scene),
            sky_lod_k: srcs.sky_lod,
            cloud_shadow_n: srcs.cloud_shadow_n,
            map,
            staged,
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
            // The pack. Bound in EVERY variant and whether or not `gbuf_full`
            // is set: an unarmed tracer's pair is stride-sized rather than
            // absent, so binding them costs nothing and is strictly better
            // than letting them fall through to a 16-byte dummy that a
            // mis-set `FLAG_GBUF` would then overrun. The flag is still the
            // safety boundary — this just removes one way to be unlucky.
            (0, Reg::U, 15, &self.gbuf),
            (0, Reg::U, 32, &self.gbuf_ext),
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

    /// Point `cs_feed_xess`'s three output registers at an upscaler's input
    /// images. The Vulkan spelling of D3D12's `trace::wire_feed_targets` — the
    /// shared, owner-independent half both tracers wrap — and deliberately the
    /// same shape: the tracer BINDS views it does not own, so image lifetime
    /// stays beside every other resource of whoever created them
    /// (`vk::fsr3::Fsr3` today).
    ///
    /// `views` is (colour u16, depth u18, mvec u19), named by ROLE at the call
    /// site rather than by index, because the three formats differ (RGBA16F /
    /// R32F / RG16F) and a swapped pair is exactly the class a derived layout
    /// cannot catch — every one of them is a `STORAGE_IMAGE` to Vulkan.
    ///
    /// GUARDED ON THE MAP, the `--sw-rays` TLAS rule: a tracer built without
    /// `TracerOpts::feed` compiled no unit declaring these registers, so the
    /// layout has no such bindings, and a write to a binding the layout lacks
    /// is not a harmless no-op. Calling this on an unfed tracer is therefore
    /// inert rather than undefined.
    ///
    /// Rebinding mid-session is legal for `bind_textures`' reason:
    /// `VkHeadless::run` fences every submit, so nothing is ever pending
    /// against these sets.
    pub fn wire_feed(&self, hg: &VkHeadless, views: [vk::ImageView; 3]) {
        let d = &hg.vk.device;
        // EVERY set-0 variant, not just the terminal one: `record_feed` binds
        // the terminal variant today, but a variant that silently lacked the
        // targets would be a latent trap for whichever pass reaches for them
        // next, and the write is three descriptors.
        let mut all: Vec<&[vk::DescriptorSet]> = vec![&self.sets];
        if let Some(w) = &self.wave {
            all.push(&w.sets_a);
            all.push(&w.sets_b);
        }
        if let Some(h) = &self.hemi {
            all.push(&h.sets_a);
            all.push(&h.sets_b);
        }
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = views
            .iter()
            .map(|v| {
                [vk::DescriptorImageInfo::default()
                    .image_view(*v)
                    .image_layout(vk::ImageLayout::GENERAL)]
            })
            .collect();
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        for sets in &all {
            for (i, reg) in [16u32, 18, 19].iter().enumerate() {
                let b = binding_of(Reg::U, *reg);
                if !self.map.entries.contains_key(&(0, b)) {
                    continue;
                }
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(sets[0])
                        .dst_binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&infos[i]),
                );
            }
        }
        if std::env::var_os("FR_VK_MAP").is_some() {
            eprintln!(
                "check-vk:   wire_feed <- {} write(s) over {} set-0 variant(s)",
                writes.len(),
                all.len()
            );
        }
        if !writes.is_empty() {
            unsafe { d.update_descriptor_sets(&writes, &[]) };
        }
    }

    /// Record `cs_feed_xess`: fan this frame's pack + `accum` out into the
    /// images `wire_feed` bound. One thread per pixel, 8x8 groups — the
    /// kernel's own `[numthreads]`.
    ///
    /// Recorded rather than run, so the caller can put it in the SAME command
    /// buffer as the upscaler dispatch that consumes its output — D3D12's
    /// "trace -> feed -> upscale on one list" shape.
    ///
    /// The feed's registers all live in the shared base of every variant
    /// (`accum` u0, `gbuf` u15, `FrameCb` b0, plus the three targets), so this
    /// binds the TERMINAL variant and needs no variant of its own.
    ///
    /// # Safety
    /// `cmd` must be in the recording state, and `write_cb` must have run for
    /// THIS frame: the kernel reads `rw`/`rh`/`CAM_NEAR`/`CAM_FAR` out of the
    /// frame constants, so a stale block feeds right values for a wrong frame.
    pub unsafe fn record_feed(&self, d: &ash::Device, cmd: vk::CommandBuffer) -> Result<(), String> {
        let pipe = self.feed_pipe.ok_or("this tracer was not built with TracerOpts::feed")?;
        unsafe {
            barrier(d, cmd);
            self.bind(d, cmd, &self.sets);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            d.cmd_dispatch(cmd, self.rw.div_ceil(8), self.rh.div_ceil(8), 1);
            barrier(d, cmd);
        }
        Ok(())
    }

    /// Point the bridge's seven registers at the planes this tracer owns.
    ///
    /// Unlike `wire_feed`, which binds views it does NOT own (the upscaler's),
    /// these planes belong to the tracer — the bridge is our own kernel pair
    /// and the planes exist to sit between its halves. So this takes no
    /// argument; it is "publish what I already have".
    ///
    /// Guarded on the map for `wire_feed`'s reason (a write to a binding the
    /// layout lacks is not a harmless no-op), which also makes calling it on a
    /// tracer built without `TracerOpts::nrd` inert rather than undefined.
    pub fn wire_nrd(&self, hg: &VkHeadless) {
        let Some(n) = &self.nrd else { return };
        let d = &hg.vk.device;
        let mut all: Vec<&[vk::DescriptorSet]> = vec![&self.sets];
        if let Some(w) = &self.wave {
            all.push(&w.sets_a);
            all.push(&w.sets_b);
        }
        if let Some(h) = &self.hemi {
            all.push(&h.sets_a);
            all.push(&h.sets_b);
        }
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = NRD_REGS
            .iter()
            .map(|(_, p)| {
                [vk::DescriptorImageInfo::default()
                    .image_view(n.planes[*p].view)
                    .image_layout(vk::ImageLayout::GENERAL)]
            })
            .collect();
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        for sets in &all {
            for (i, (reg, _)) in NRD_REGS.iter().enumerate() {
                let b = binding_of(Reg::U, *reg);
                if !self.map.entries.contains_key(&(0, b)) {
                    continue;
                }
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(sets[0])
                        .dst_binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&infos[i]),
                );
            }
        }
        if std::env::var_os("FR_VK_MAP").is_some() {
            eprintln!(
                "check-vk:   wire_nrd <- {} write(s) over {} set-0 variant(s)",
                writes.len(),
                all.len()
            );
        }
        if !writes.is_empty() {
            unsafe { d.update_descriptor_sets(&writes, &[]) };
        }
    }

    /// Arm or disarm the pack's sig lanes for the frames recorded after this.
    /// See the `nrd_sig` field: a cbuffer bit, so the two arms of the
    /// capture-invariance gate need no second tracer.
    pub fn set_nrd_sig(&self, on: bool) {
        self.nrd_sig.set(on);
    }

    /// Read one bridge plane back, by the `N_*` index.
    ///
    /// THIS IS WHAT MAKES THE PASSTHROUGH GATE NON-VACUOUS, and the reason is
    /// worth stating because a first draft shipped without it and a planted
    /// tooth PASSED: with the planes unbound the recompose reads zeros, so
    /// `D_out − D_in` is zero, `col` collapses onto `base`, and the byte
    /// identity holds FOR THE WRONG REASON. A passthrough makes the delta zero
    /// by construction, so byte-identity alone provably cannot separate "the
    /// bridge computed a zero delta" from "the bridge read nothing". Asking
    /// whether the pack's data ARRIVED is the missing half — M3d's texture
    /// probe in another currency.
    pub fn read_nrd_plane(&self, hg: &VkHeadless, idx: usize) -> Result<Vec<u8>, String> {
        let n = self.nrd.as_ref().ok_or("this tracer was not built with TracerOpts::nrd")?;
        let p = n.planes.get(idx).ok_or("nrd plane index")?;
        // Bytes per texel, by the same table `new` created these with.
        let bpp: u64 = match idx {
            N_IN_NR => 4,
            N_IN_VIEWZ => 4,
            _ => 8,
        };
        let bytes = bpp * self.rw as u64 * self.rh as u64;
        let dst = hg.vk.buffer(bytes, vk::BufferUsageFlags::TRANSFER_DST, true)?;
        let region = [vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D { width: self.rw, height: self.rh, depth: 1 })];
        // GENERAL is a legal copy SOURCE (unlike SHADER_READ_ONLY_OPTIMAL,
        // which B3 learned the hard way), so the planes need no transition at
        // all to be read — the one place resting everything in GENERAL pays
        // for itself outright.
        let r = hg.run(|d, cmd| unsafe {
            d.cmd_copy_image_to_buffer(
                cmd,
                p.img,
                vk::ImageLayout::GENERAL,
                dst.buf,
                &region,
            );
        });
        let out = r.and_then(|_| hg.vk.read(&dst, bytes as usize));
        hg.vk.free_buffer(&dst);
        out
    }

    /// The bridge's planes as `(image, view)` in `Plane::ALL` order, MINUS
    /// `OutValidation` — which the bridge has no use for and does not allocate.
    ///
    /// This is how a real engine gets registered against the SAME images the
    /// bridge writes and reads, rather than a second set that would then need
    /// copying at both ends. `vk::nrd::VkNrd::new` is the one caller and it
    /// checks the length, so a plane appended to `N_*` without being handed
    /// over is an error at construction rather than a silently short list.
    ///
    /// `None` when the tracer was built without `TracerOpts::nrd` — the same
    /// shape `wire_nrd` and `record_nrd_*` take, so an unarmed tracer cannot be
    /// handed to an engine by accident.
    pub fn nrd_plane_handles(&self) -> Option<Vec<(vk::Image, vk::ImageView)>> {
        let n = self.nrd.as_ref()?;
        Some(n.planes.iter().map(|p| (p.img, p.view)).collect())
    }

    /// Re-derive the sun / sky / emissive constant rows after a lighting
    /// change — `TraceGpu::refresh_sky`'s twin, and the same one-line body over
    /// the same shared `FrameCb` method.
    ///
    /// The capture path is the caller that needs it: `--cinematic`'s time of
    /// day moves per OUTPUT frame (the world's attractors, or an authored
    /// keyframe hour), and `cine_frame_state` mutates the scene's lighting
    /// without touching anything the per-frame `with_frame` rebuilds.
    ///
    /// That caller — `run_cinematic_vk` — is
    /// `cfg(all(unix, not(target_os = "macos")))`, so on macOS this module
    /// compiles (it is unix) with nothing calling this. The allow says so
    /// rather than letting a real dead-code warning accumulate beside it;
    /// `src/vk/fsr3.rs` marks its own cfg-orphaned items the same way.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    pub fn refresh_sky(&mut self, scene: &Scene) {
        self.cb_base.refresh_sky_rows(scene, self.rw, self.rh);
    }

    /// Bring the bridge's planes into `GENERAL`, once. See `Nrd::laid_out` for
    /// why exactly once and not per dispatch.
    unsafe fn nrd_lay_out(&self, d: &ash::Device, cmd: vk::CommandBuffer, n: &Nrd) {
        if n.laid_out.replace(true) {
            return;
        }
        let bs: Vec<vk::ImageMemoryBarrier> = n
            .planes
            .iter()
            .map(|p| {
                vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(p.img)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    )
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            })
            .collect();
        unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &bs,
            );
        }
    }

    /// The bridge's FRONT half: `gbuf`/`gbuf_ext`/`accum` -> the five IN
    /// planes, plus the engine's own depth and mvec guides (THE FOLD — those
    /// stores moved into `cs_nrd_pack` on the D3D12 side, which is why an NRD
    /// frame runs no separate engine feed at all).
    ///
    /// # Safety
    /// `cmd` must be recording and `write_cb` must have run for THIS frame with
    /// the sig lanes armed — a pack over unarmed lanes reads zeros and the
    /// recompose then subtracts them.
    pub unsafe fn record_nrd_pack(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
    ) -> Result<(), String> {
        let n = self.nrd.as_ref().ok_or("this tracer was not built with TracerOpts::nrd")?;
        unsafe {
            self.nrd_lay_out(d, cmd, n);
            barrier(d, cmd);
            self.bind(d, cmd, &self.sets);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, n.pipes[0]);
            d.cmd_dispatch(cmd, self.rw.div_ceil(8), self.rh.div_ceil(8), 1);
            barrier(d, cmd);
        }
        Ok(())
    }

    /// The bridge's BACK half: the delta-form recompose into the engine's
    /// colour plane (`u16` — the same image `wire_feed` bound).
    ///
    /// # Safety
    /// As `record_nrd_pack`, and the OUT planes must carry whatever the engine
    /// produced from the IN ones.
    pub unsafe fn record_nrd_out(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
    ) -> Result<(), String> {
        let n = self.nrd.as_ref().ok_or("this tracer was not built with TracerOpts::nrd")?;
        unsafe {
            self.nrd_lay_out(d, cmd, n);
            barrier(d, cmd);
            self.bind(d, cmd, &self.sets);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, n.pipes[1]);
            d.cmd_dispatch(cmd, self.rw.div_ceil(8), self.rh.div_ceil(8), 1);
            barrier(d, cmd);
        }
        Ok(())
    }

    /// A PASSTHROUGH denoiser: OUT = IN, byte for byte.
    ///
    /// This is N3's control arm — the engine that does nothing — and it is what
    /// makes the bridge's claim checkable without an engine at all: with
    /// `OUT == IN` the delta form's correction is exactly zero, so the
    /// recompose must reproduce `base`, which is exactly what the feed wrote.
    /// Any drift is the bridge's arithmetic or its wiring and nothing else.
    ///
    /// # Safety
    /// `cmd` must be recording, and the IN planes must hold a completed pack.
    pub unsafe fn record_nrd_passthrough(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
    ) -> Result<(), String> {
        let n = self.nrd.as_ref().ok_or("this tracer was not built with TracerOpts::nrd")?;
        let region = [vk::ImageCopy::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .extent(vk::Extent3D { width: self.rw, height: self.rh, depth: 1 })];
        unsafe {
            self.nrd_lay_out(d, cmd, n);
            barrier(d, cmd);
            for (src, dst) in [(N_IN_DIFF, N_OUT_DIFF), (N_IN_SPEC, N_OUT_SPEC)] {
                d.cmd_copy_image(
                    cmd,
                    n.planes[src].img,
                    vk::ImageLayout::GENERAL,
                    n.planes[dst].img,
                    vk::ImageLayout::GENERAL,
                    &region,
                );
            }
            barrier(d, cmd);
        }
        Ok(())
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
    /// only if the ladder was never built; nothing turns it off today.
    pub fn wave_shape(&self) -> Option<(u32, u32, u32)> {
        self.wave.as_ref().map(|w| (w.depth_full, w.cap_leaf, w.cap_sky))
    }

    /// The per-frame cbuffer, written host-side before the submit.
    fn write_cb(&self, hg: &VkHeadless, p: &FrameParams) -> Result<(), String> {
        // `gbuf_full` and `gbuf_ext` follow the ALLOCATION (the field is set by
        // the constructor argument that sized the two buffers), which is what
        // keeps `FLAG_GBUF` from ever arming over a stride-sized pack — a
        // storage buffer has no bounds check, so that flag is memory safety
        // and not an optimization (gfx/frame.rs's own note on it).
        //
        // The ext half arms WITH the core one rather than on demand, matching
        // `--check-gpu` M7's `force_gbuf_ext(true)` and for its reason: the
        // consumer here is a CPU readback, not a feed kernel, and
        // `dlss::mv_selftest`'s sky arm reads the normal lane, which lives in
        // ext. Core-only would fail that arm for a reason unrelated to the
        // pack.
        //
        // THE SIG LANES follow the bridge, through a per-frame cell rather than
        // the allocation: `cs_nrd_pack` reads `sig`/`sig2` out of `gbuf_ext` to
        // build the diffuse and specular IN planes, so a bridge over unarmed
        // lanes packs zeros and the recompose then subtracts them. They are
        // safe to move per frame because the capture is assignment-only — the
        // exported values are already computed — which is what makes `accum`
        // bit-identical across the toggle and the invariance gateable without a
        // second tracer.
        //
        // `remod_exact` rides ARMED, matching the D3D12 shipping default, so
        // the two backends' packs agree about what `sig.w`'s high half carries
        // (`m_d`, on loan from `shadow_t`). `nrd_sig` arms with it because
        // `remod_exact` is derived from the conjunction.
        //
        // The rest stay off, each structurally rather than as a default:
        // `sky_ext_skip` is an optimization whose whole premise is that NRD is
        // the SOLE ext subscriber, which a gate reading the pack back on the
        // CPU is not; `nrd_rejitter` is NVIDIA's Jacobian and is engine-gated
        // on D3D12 for a stated reason (an oracle sharing a post-process with
        // the arm under test is not an oracle), and no engine sits between the
        // bridge halves yet; `rclamp` is default-OFF there too.
        //
        // A POSITIONAL PILE THIS LONG IS A HAZARD, and this call site is the
        // evidence: the NRD exact-remodulation merge added `remod_exact` and
        // `rclamp` on the D3D12 side and left this one a compile error that
        // only a Linux build could see. When the next one lands, decide it
        // HERE too — a wrong `false` would be silent where the missing
        // argument was loud.
        let g = self.gbuf_full;
        let sig = g && self.nrd_sig.get();
        let mut cb =
            self.cb_base.with_frame(p, g, sig, g, sig, false, false, sig, (false, false), false);
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
            d.cmd_dispatch_indirect(cmd, args.buf, u64::from(slot) * gs::ARG_STRIDE);
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
                .chain(self.feed_pipe.iter())
                .chain(self.nrd.iter().flat_map(|n| n.pipes.iter()))
            {
                d.destroy_pipeline(*p, None);
            }
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_sampler(self.samp_lin, None);
            d.destroy_sampler(self.samp_aniso, None);
            for i in std::iter::once(&self.hdr)
                .chain(self.nrd.iter().flat_map(|n| n.planes.iter()))
            {
                d.destroy_image_view(i.view, None);
                d.destroy_image(i.img, None);
                d.free_memory(i.mem, None);
            }
        }
        self.layouts.destroy(vkd);
        for b in [
            &self.accum,
            &self.tbuf,
            &self.info,
            &self.gbuf,
            &self.gbuf_ext,
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
    create_image_fmt(vkd, rw, rh, vk::Format::R16G16B16A16_SFLOAT)
}

/// `create_image` with the format spelled, for the NRD planes — whose wire
/// formats are `NrdGpu`'s and therefore not all RGBA16F (`in_viewz` is R32F and
/// `in_nr` is the packed 10:10:10:2 NRD enc-2 normal+roughness).
///
/// TRANSFER_DST rides along with TRANSFER_SRC unconditionally: the bridge's
/// passthrough control arm copies IN planes over OUT ones, and a usage bit
/// added only under a gate would make the diagnostic's subject a specially
/// built resource set rather than the shipping one — the `FR_SPLIT_AUDIT`
/// lesson, which cost `vk::stage` the same bit one milestone earlier.
fn create_image_fmt(
    vkd: &crate::vk::device::Vk,
    rw: u32,
    rh: u32,
    fmt: vk::Format,
) -> Result<Image, String> {
    let d = &vkd.device;
    let ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(fmt)
        .extent(vk::Extent3D { width: rw, height: rh, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        // SAMPLED rides along with STORAGE for the NRD planes: a real engine
        // between `cs_nrd_pack` and `cs_nrd_out` binds the SAME image as a UAV
        // in the pass that writes it and an SRV in the pass that reads it, so
        // both usages are genuinely used. Unconditional rather than per-plane
        // for the reason TRANSFER_SRC already is — a usage bit added only under
        // one caller makes that caller's resource set a special one.
        .usage(
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
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
    Ok(Image { img, view, mem })
}

/// IEEE binary16 -> f32. The one decode the readback needs; the tree's other
/// f16 sites (`dlss::ld16`) live on the D3D12 side of a `#[cfg]`.
pub(crate) fn half_from_bits(h: u16) -> f32 {
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
