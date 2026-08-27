//! The tracer on WebGPU — the recorder the browser will run, running
//! natively so it can be gated.
//!
//! The REFERENCE KERNEL (`cs_reference` into `accum`, `cs_resolve` into an
//! RGBA16F storage texture) plus the two per-frame cloud caches it reads, the
//! WAVEFRONT QUADTREE beside it — seed, the indirect level ladder, and the
//! leaf + sky terminal fills — the HEMISPHERE BOUNCE TIERS the H key cycles
//! (`fb_mode > 0`), and STRUCTURE REPLAY.
//!
//! The reference kernel landed first, and for the reason `vk/tracer.rs`'s
//! header gives: it is the smallest thing that can be WRONG in an interesting
//! way. Every kernel above it reads the same streams through the same layout
//! and shades through the same `shade.hlsli`, so a stream bound at the wrong
//! slot, a skewed material stride, or a cbuffer that packs differently shows
//! up THERE as a picture that disagrees with the CPU — before a quadtree is in
//! the way to confuse the attribution. Having proven that, the ladder is
//! scored GPU-vs-GPU against it, which is what lets J7 demand EXACT agreement
//! rather than a statistical bar.
//!
//! FIVE THINGS THIS SPELLS DIFFERENTLY FROM BOTH NATIVE BACKENDS.
//!
//! - **One bind group per (entry, variant), not one set for the tracer.**
//!   D3D12 binds one root signature and dispatches a dozen kernels against
//!   it; Vulkan binds one descriptor set per PARITY. WebGPU computes a
//!   dispatch's usage scope from the pipeline LAYOUT rather than from the
//!   shader's static use, so a layout carrying more than the entry declares
//!   makes dispatches conflict with each other (C1 measured exactly this on
//!   the smoke — `headless.rs`'s header). Layouts are therefore per entry
//!   point, and so are bind groups. They are still built ONCE, at
//!   construction, so a dispatch costs a `set_bind_group` and no descriptor
//!   traffic.
//!
//!   THIS IS ALSO WHY THE HEMISPHERE TIER NEEDED NO ARGUMENT HERE. The hemi
//!   units re-declare u5/u6/u7/u9 as `HemiCellRec` queues where the ladder has
//!   `TileRec` ones, and the Vulkan port owes a paragraph explaining why that
//!   is not a conflict for its ONE shared layout (it keys on descriptor kind,
//!   not on the HLSL type). Per-entry derivation makes the question disappear:
//!   a hemi entry's layout is built from a hemi entry's IR.
//! - **The push ring is SIZED FROM THE PROGRAM.** A fixed 64 rows carried the
//!   ladder; the hemisphere tail is ~10 dispatches per `HEMI_BATCH` slice of
//!   the framebuffer, so 720p asks for ~580. Because every row is written
//!   before the submit, a dispatch cannot reuse its predecessor's row and the
//!   ring's length IS the longest program. See [`push_rows_for`].
//! - **`b1` is a DYNAMIC-OFFSET UNIFORM RING.** The ladder rewrites the push
//!   block twice per level. D3D12 has root constants; Vulkan has
//!   `vkCmdUpdateBuffer` — an inline transfer at the right point in the
//!   stream. WebGPU has neither: every host write in a recording happens
//!   before the submit. So all the push rows a frame needs are written up
//!   front into one uniform buffer at the device's
//!   `min_uniform_buffer_offset_alignment` stride, and each dispatch selects
//!   its row with a dynamic offset. This is the one place a DERIVED layout
//!   carries a hand-made decision, which is why it is a parameter of
//!   `layout::build_unit` and named at exactly one call site below.
//! - **There are no barriers and no resource states.** WebGPU's per-dispatch
//!   usage scopes ARE the synchronization. D3D12 transitions `args`
//!   UNORDERED_ACCESS <-> INDIRECT_ARGUMENT around every `ExecuteIndirect`
//!   and Vulkan issues one global `COMPUTE|TRANSFER -> COMPUTE|DRAW_INDIRECT`
//!   edge per dispatch pair; here the implementation tracks hazards, and
//!   that this is sufficient is part of what `--check-wgpu` J3 already
//!   proved on the smoke.
//! - **There is no acceleration structure.** `webgpu/scene.rs`'s header has
//!   this in full: the browser path is `--sw-rays`, every ray traverses our
//!   own binary BVH through `rt_sw.hlsli`, and the tree is a storage buffer
//!   like everything else.
//!
//! THE CORPUS IS THE BROWSER'S, ASSEMBLED BY THE BROWSER'S OWN FUNCTION.
//! `gfx::shaders::web_keys` and the two `web_*_unit` wrappers are shared
//! with `--check-wgsl`'s `web_units`, so the text this compiles is the text
//! that gate validates and the text the W7 golden pins. That sharing is the
//! whole reason the golden means anything for this file: a tracer that
//! assembled its own sources would be a second corpus wearing the first
//! one's audit.

use crate::gfx::frame::{
    CB_STRIDE, FrameCb, FrameParams, GBUF_EXT_STRIDE, GBUF_STRIDE, HEMI_BATCH, HEMI_MAX_DEPTH,
    fb_mode_of,
};
use crate::gfx::shaders as gs;
use crate::scene::Scene;
use crate::spirv::{Reg, Spirv, binding_of};
use crate::wgsl::BindKind;

use super::device::Wgpu;
use super::headless::WgpuHeadless;
use super::layout::{self, Unit};
use super::scene::WgpuScene;

/// How many push rows one frame's program can need.
///
/// SIZED FROM THE PROGRAM rather than fixed at a literal, and that is forced
/// rather than tidy: every row is written before the submit (module header),
/// so a dispatch cannot reuse the row of the one before it and the ring must
/// be at least as long as the longest program a frame can record. The
/// HEMISPHERE TAIL is what makes this a real number — it is ~10 steps per
/// `HEMI_BATCH` slice of the framebuffer, so 1280x720 asks for ~580 rows where
/// the ladder alone asks for ~30. The literal 64 this replaced would have
/// refused every fb frame at every resolution.
///
/// A ring that had to GROW mid-recording would mean a second submit, which is
/// the one thing this design does not do — so the sizing is up front, and
/// `run_steps` still refuses loudly on an overrun. That refusal is what keeps
/// this a sizing decision rather than a soundness one.
fn push_rows_for(depth_full: u32, px: u64, hemi: bool) -> u64 {
    // seed + clear_info + clear_h, the ladder's prep/level pair per level, and
    // the terminal fills (two preps, two cloud caches, leaf, sky).
    let ladder = 3 + 2 * u64::from(depth_full) + 6;
    let tail = if hemi {
        // Per batch: prep_batch, the root, a prep/cell pair per level, then the
        // leaf prep and the leaf rays. Plus the one compose splat per frame.
        let levels = u64::from(HEMI_MAX_DEPTH - 1);
        px.div_ceil(u64::from(HEMI_BATCH)) * (4 + 2 * levels) + 1
    } else {
        0
    };
    // Headroom for the probe path, which plans its own short program. A spare
    // row costs one alignment quantum.
    ladder + tail + 8
}

/// Which resource sits behind a slot. The three things a WebGPU bind group
/// entry can be, and the reason this is an enum rather than three tables:
/// the resolver walks the DECLARED bindings in order and must answer for
/// each one whatever kind it is.
enum Res<'a> {
    Buf(&'a wgpu::Buffer),
    /// A uniform buffer bound as a WINDOW rather than whole — the push ring,
    /// whose binding is one row and whose offset is dynamic.
    Window(&'a wgpu::Buffer, u64),
    Tex(&'a wgpu::TextureView),
    Samp(&'a wgpu::Sampler),
}

/// Which meaning the overloaded registers carry for a dispatch.
///
/// `u5`/`u6` are the ladder's tile-queue ping-pong AND the terminal phase's
/// cloud caches; `t0` is the WIDE frustum tree for the ladder and the BINARY
/// tree for every ray-shooting pass, and `t1` follows it (the ladder's
/// leaf-cut translation map vs `bvh.tri_idx`). The Vulkan backend spells this
/// as five descriptor-set variants over one layout; here it is which override
/// table the bind-group builder was given, and the bind groups for every
/// (entry, variant) pair are built once at construction.
///
/// The hemisphere's two parities move FIVE slots at once rather than two, and
/// the extra three are the reason they are variants at all rather than a
/// second base: `u7` to the hemi leaf queue, `u9` to the hemi cut pool, and
/// `t0` back to the BINARY tree, because the hemi kernels compile
/// `frustum.hlsli`'s binary `bound_query` deliberately (short queries lose on
/// the wide tree — measured +35% there against -54% on the tile path).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variant {
    /// The terminal phase: `u5`/`u6` are the cloud caches, `t0`/`t1` the
    /// binary tree and its triangle ids. The reference, resolve, leaf and sky
    /// passes all bind this.
    Terminal,
    /// Ladder, even levels: `u5` = queue A (in), `u6` = queue B (out).
    LadderA,
    /// Ladder, odd levels: the swap.
    LadderB,
    /// Hemisphere, even levels: `u5` = cell queue A (in), `u6` = B (out).
    HemiA,
    /// Hemisphere, odd levels: the swap. THE ROOT PASS RUNS UNDER THIS ONE —
    /// it WRITES `hqout`, so the parity dance is one step off the ladder's and
    /// level 0 then reads hq_a as `hqin` under the even variant. Getting it
    /// backwards costs the whole batch SILENTLY: the root's output lands in a
    /// queue nothing reads. MEASURED, by planting exactly that — J8 answers
    /// `leaf-rays 0`, `psa-viol 45/56` with max error 3.14 (the whole
    /// hemisphere unaccounted) and an AO error of 0.59 against a 0.02 bar.
    HemiB,
}

/// Every variant, in `Variant as usize` order — the bind-group table's index
/// space, stated once so the builder and the dispatcher cannot disagree about
/// how long it is.
const VARIANTS: [Variant; 5] =
    [Variant::Terminal, Variant::LadderA, Variant::LadderB, Variant::HemiA, Variant::HemiB];

/// One compiled entry point plus its per-variant bind groups.
struct Pass {
    unit: Unit,
    /// Indexed by `Variant as usize`; each is the bind groups for that
    /// variant, indexed by GROUP NUMBER (`None` where the entry declares
    /// nothing in that group).
    groups: Vec<Vec<Option<wgpu::BindGroup>>>,
}

const P_REFERENCE: usize = 0;
const P_RESOLVE: usize = 1;
const P_SKY_LOD: usize = 2;
const P_CLOUD_SHADOW: usize = 3;
// The ladder, appended to the same pass table — one index space, so `go`
// takes a pass number and nothing has to know which half it came from.
const P_SEED: usize = 4;
const P_PREP: usize = 5;
const P_PREP_MUL: usize = 6;
const P_CLEAR_INFO: usize = 7;
const P_LEVEL: usize = 8;
const P_LEVEL_WIDE: usize = 9;
const P_LEAF: usize = 10;
const P_SKY: usize = 11;
// The hemisphere tiers, appended to the same index space. `cs_leaf` appears
// TWICE across the table — once from `srcs.leaf` and once from `srcs.leaf_fb`
// — and that is the point rather than an accident: they are two pipelines from
// ONE source separated by `LEAF_NO_FB`, a register-pressure decision, and an
// fb frame needs the second or its leaf pass appends no shading points at all
// and every hemisphere gate below goes vacuous.
const P_HEMI_ROOT: usize = 12;
const P_HEMI_CELL: usize = 13;
const P_HEMI_LEAF: usize = 14;
const P_COMPOSE: usize = 15;
const P_PREP_BATCH: usize = 16;
const P_SEED_PROBES: usize = 17;
const P_CLEAR_H: usize = 18;
const P_LEAF_FB: usize = 19;
/// Structure replay's seed: re-dispatches the persisted terminal queues
/// instead of running the ladder that filled them.
const P_SEED_REPLAY: usize = 20;

/// `cs_prep`'s "do not zero any counter" sentinel, and the args slots the
/// terminal fills and the hemisphere passes launch from — lockstep with
/// `gpu/trace.rs`'s consts (which are `#[cfg(windows)]`-only) and
/// `vk/tracer.rs`'s copies. The ladder owns slots 0..=10 (one per level), so
/// the hemi three sit above it and the terminal two at the top of the 16.
const NO_RESET: u32 = 0xffff_ffff;
const ARG_HEMI_ROOT: u32 = 11;
const ARG_HEMI_CELL: u32 = 12;
const ARG_HEMI_LEAF: u32 = 13;
const ARG_LEAF: u32 = 14;
const ARG_SKY: u32 = 15;

/// The ladder's own resources.
struct Wave {
    /// The tile-queue ping-pong.
    qa: wgpu::Buffer,
    qb: wgpu::Buffer,
    qleaf: wgpu::Buffer,
    qsky: wgpu::Buffer,
    cut_pool: wgpu::Buffer,
    /// The indirect-argument slots every level and terminal fill launches
    /// from. `INDIRECT | STORAGE` on one buffer is what the whole design
    /// rests on: a GPU-written counter becomes a dispatch shape without the
    /// CPU ever seeing the count.
    args: wgpu::Buffer,
    /// `t0` for the ladder: the FRUSTUM structure (the quantized FTree, or
    /// the binary tree under `--no-ftree`). A frustum query cannot descend
    /// the binary tree, which is why this is a different buffer from
    /// `WgpuTracer::binary` rather than the same one under another name.
    tree: wgpu::Buffer,
    /// `t1` for the ladder under `--sw-rays` + FTREE: `level_finish`'s
    /// slot-ref -> binary-node translation map. `None` leaves t1 on the
    /// dummy, which is what the corpus compiles to when the arm is off.
    ft_bnode: Option<wgpu::Buffer>,
    depth_full: u32,
    cap_leaf: u32,
    cap_sky: u32,
}

/// The hemisphere bounce tiers' own half: the batch-transient cell queues and
/// cut pool, the per-pixel planes the leaf/compose pair passes radiance
/// through, and the shading-point queue.
///
/// Separate from [`Wave`] because it is optional for a different reason. The
/// queues are sized to ONE BATCH of the worst-case cell fan-out
/// (`HEMI_BATCH * 4^(HEMI_MAX_DEPTH-1)` = 1,048,576 records), which is ~280 MB
/// of buffers a run with no interest in fb should not be asked for. That
/// per-batch reset IS the memory bound — the whole reason the hemisphere
/// wavefront is batched at all.
///
/// EVERY ONE OF THESE FITS UNDER WEBGPU'S DEFAULTS, which is worth stating
/// because C2's bistro run did not: the largest is the cut pool at 88 MB and
/// the cell queues are 64 MB each, all under the 128 MB default
/// `max_storage_buffer_binding_size`. So this tier costs device MEMORY without
/// costing a limits row — and `scene::ask_for` carries the rows anyway, so the
/// ask stays derived from the same arithmetic rather than from that reading.
struct Hemi {
    /// Cell ping-pong. See [`Variant::HemiB`] for why the root writes hq_a.
    hq_a: wgpu::Buffer,
    hq_b: wgpu::Buffer,
    hq_leaf: wgpu::Buffer,
    cut: wgpu::Buffer,
    /// u10/u11: the leaf pass's ambient-free colour and its `kd` weight — what
    /// makes compose a pure weight x mass multiply (`compose.hlsl`'s header).
    partial: wgpu::Buffer,
    ambw: wgpu::Buffer,
    /// u12: the FIXED-POINT accumulator. Integer atomics are order-independent,
    /// which is what makes a queue-driven integrator reproducible run to run —
    /// and what makes a sun disc reaching a gather path a saturation bug rather
    /// than a brightness one.
    hbuf: wgpu::Buffer,
    /// u13: the shading points. `cs_leaf` (the fb arm) appends into it on a
    /// frame; the probe path writes it from the host.
    pts: wgpu::Buffer,
    cap_cell: u64,
    cap_cut: u64,
}

/// The ladder's STRUCTURAL queue caps for one resolution.
///
/// One function because it has two callers that run at different times:
/// [`WgpuTracer::new`] allocates from it, and `webgpu::scene::ask_for` sizes
/// the DEVICE ASK from it — and the ask has to be right before a device
/// exists, so the two cannot share a value, only a derivation. Two copies of
/// this arithmetic is exactly the duplicated-constant hazard that gets caught
/// by nobody: the buffers would be one size and `set_caps` would tell the
/// shaders another.
///
/// The bound itself is `TraceGpu::new`'s and `VkTracer::new`'s, verbatim, and
/// is what lets `CTR_OVERFLOW` be gated at exactly 0: at depth d there are at
/// most 4^d rects, internal tiles live at depth < D, every terminal contains
/// at least one depth-D path cell, and a split allocates one cut slot.
pub struct Caps {
    pub depth: u32,
    pub tile: u64,
    pub leaf: u64,
    pub cut: u64,
}

/// `doubled_cut`: under `--sw-rays` + FTREE, `level_finish` translates each
/// leaf-emitting split's slot-ref cut into a SECOND fresh slot of binary node
/// ids, so the pool needs twice the slots. Not optional tidiness — the
/// exhaustion arm degrades to ROOT seeding, which is sound but is a DIFFERENT
/// structure, and the Vulkan port measured 107 such fallbacks at 800x600
/// before the doubling was mirrored.
pub fn caps_for(rw: u32, rh: u32, doubled_cut: bool) -> Caps {
    let d = gs::depth_full(rw, rh);
    // Clamped for the ARITHMETIC only — a depth past the indirect-arg slots
    // is refused by the caller, loudly, rather than silently sized down here.
    let dc = d.min(11);
    let cut = ((1u64 << (2 * dc)) - 1) / 3 + 1;
    Caps {
        depth: d,
        tile: if dc >= 1 { 1u64 << (2 * (dc - 1)) } else { 1 },
        leaf: 1u64 << (2 * dc),
        cut: if doubled_cut { cut * 2 } else { cut },
    }
}

/// The hemisphere tier's per-BATCH caps: cell records and cut slots.
///
/// A peer of [`caps_for`] and public for the same reason — `scene::ask_for`
/// sizes the device ask from it before a device exists, and
/// [`WgpuTracer::new`] allocates from it afterwards. Two copies of this
/// arithmetic is exactly the duplicated-constant hazard nobody catches: the
/// buffers would be one size and `set_caps` would tell the shaders another.
///
/// `HEMI_BATCH * 4^(HEMI_MAX_DEPTH-1)` is the worst-case fan-out of one batch,
/// and the batch reset is what keeps it a bound rather than a frame total.
pub fn hemi_caps() -> (u64, u64) {
    let cell = u64::from(HEMI_BATCH) * (1u64 << (2 * (HEMI_MAX_DEPTH - 1)));
    let cut = u64::from(HEMI_BATCH) * (((1u64 << (2 * (HEMI_MAX_DEPTH - 1))) - 1) / 3 + 1);
    (cell, cut)
}

/// One recorded dispatch: which entry, under which variant, with which push
/// row, launched how.
///
/// THE LADDER IS BUILT AS DATA FIRST AND RECORDED SECOND, and that is a
/// WebGPU necessity rather than a style: every push row a frame needs must be
/// written BEFORE the submit (module header), so the row values have to be
/// known before recording starts. Building a `Vec<Step>` and then walking it
/// twice — once to write the ring, once to record — is how the two stay in
/// step without the control flow being written out twice.
enum Step {
    Direct { pass: usize, variant: Variant, push: [u32; 4], gx: u32, gy: u32 },
    Indirect { pass: usize, variant: Variant, push: [u32; 4], slot: u32 },
}

impl Step {
    fn push(&self) -> [u32; 4] {
        match *self {
            Step::Direct { push, .. } | Step::Indirect { push, .. } => push,
        }
    }
}

pub struct WgpuTracer {
    pub rw: u32,
    pub rh: u32,
    pub accum: wgpu::Buffer,
    pub tbuf: wgpu::Buffer,
    pub info: wgpu::Buffer,
    counters: wgpu::Buffer,
    gbuf: wgpu::Buffer,
    gbuf_ext: wgpu::Buffer,
    cloud_lod: wgpu::Buffer,
    cloud_shadow: wgpu::Buffer,
    frame_cb: wgpu::Buffer,
    /// The push ring — see the module header.
    push: wgpu::Buffer,
    push_stride: u64,
    /// Rows the ring holds, from [`push_rows_for`]. Kept so `run_steps` gates
    /// against what was ALLOCATED rather than against a constant that could
    /// drift from it.
    push_rows: u64,
    /// The BINARY BVH and `bvh.tri_idx`: what `rt_sw.hlsli` descends and the
    /// triangle ids its leaves hold. Under `--sw-rays` — which the browser
    /// corpus REQUIRES — these are the whole intersector.
    binary: wgpu::Buffer,
    sw_tri: wgpu::Buffer,
    wave: Option<Wave>,
    hemi: Option<Hemi>,
    _hdr: wgpu::Texture,
    hdr: wgpu::TextureView,
    passes: Vec<Pass>,
    cb_base: FrameCb,
    scene_aabb: ([f32; 3], [f32; 3]),
    sky_lod_k: u32,
    cloud_shadow_n: u32,
    /// Emitted WGSL bytes across every unit — the gate's "the translator
    /// ran" number.
    pub wgsl_bytes: usize,
    /// Bytes this tracer allocated on the device (its own, not the scene's).
    pub bytes: u64,
}

impl WgpuTracer {
    pub fn new(
        dev: &Wgpu,
        sp: &Spirv,
        scene: &Scene,
        ws: &WgpuScene,
        bvh: &crate::bvh::Bvh,
        rw: u32,
        rh: u32,
    ) -> Result<WgpuTracer, String> {
        // The corpus premise, asserted rather than assumed — `web_units`
        // makes the same check for the same reason: without the lever the
        // trace units declare a RaytracingAccelerationStructure, which
        // WebGPU cannot express.
        if !gs::sw_rays() {
            return Err("the WebGPU tracer needs --sw-rays (WebGPU has no ray tracing; the \
                        browser corpus traverses our own BVH through rt_sw.hlsli)"
                .into());
        }
        // ONE assembly, from the shared browser-corpus entry point — the
        // snapshots come back with it, so the buffers below are SIZED against
        // exactly the constants the kernels were COMPILED against (the
        // `TraceSources` contract; a desync here is the documented
        // device-hang class).
        let srcs = gs::trace_sources(&gs::web_keys(scene));
        let texweb = crate::gfx::texweb::hlsl(&ws.tex.plan);
        // ONE index space, in `P_*` order — the ladder is appended rather
        // than kept in a second table, so `go` takes a pass number and
        // nothing has to know which half it came from.
        let units: [(&str, &str, &str); 21] = [
            (&srcs.reference, "cs_reference", "reference"),
            (&srcs.resolve, "cs_resolve", "resolve"),
            (&srcs.sky, "cs_sky_lod", "sky-lod"),
            (&srcs.sky, "cs_cloud_shadow", "cloud-shadow"),
            (&srcs.wavefront, "cs_seed", "wf-seed"),
            (&srcs.wavefront, "cs_prep", "wf-prep"),
            (&srcs.wavefront, "cs_prep_mul", "wf-prep-mul"),
            (&srcs.wavefront, "cs_clear_info", "wf-clear-info"),
            (&srcs.wavefront, "cs_level", "wf-level"),
            (&srcs.wavefront, "cs_level_wide", "wf-level-wide"),
            (&srcs.leaf, "cs_leaf", "wf-leaf"),
            (&srcs.sky, "cs_sky", "wf-sky"),
            // The hemisphere tiers. `srcs.leaf_fb` is the SAME kernel as
            // `srcs.leaf` with the hemi arm compiled in — see `P_LEAF_FB`.
            (&srcs.hemi_wave, "cs_hemi_root", "hemi-root"),
            (&srcs.hemi_wave, "cs_hemi_cell", "hemi-cell"),
            (&srcs.hemi_leaf, "cs_hemi_leaf", "hemi-leaf"),
            (&srcs.compose, "cs_compose", "compose"),
            (&srcs.wavefront, "cs_prep_batch", "wf-prep-batch"),
            (&srcs.wavefront, "cs_seed_probes", "wf-seed-probes"),
            (&srcs.wavefront, "cs_clear_h", "wf-clear-h"),
            (&srcs.leaf_fb, "cs_leaf", "wf-leaf-fb"),
            (&srcs.wavefront, "cs_seed_replay", "wf-seed-replay"),
        ];

        // The one hand-made entry in a derived layout (module header): the
        // push block takes a dynamic offset. Named by REGISTER through
        // `binding_of`, never as a literal.
        let dynamic = [(0u32, binding_of(Reg::B, 1))];

        let mut passes: Vec<Unit> = Vec::with_capacity(units.len());
        let mut wgsl_bytes = 0usize;
        for (src, entry, tag) in units {
            let text = gs::web_trace_unit(&texweb, src);
            let u = layout::build_unit(dev, sp, &text, entry, tag, &dynamic)?;
            wgsl_bytes += u.wgsl_bytes;
            passes.push(u);
        }

        // THE ASK COVERED THE CORPUS — asserted, not assumed. `needed_
        // bindings()` is derived from the shift rule and the highest register
        // class the corpus uses; this is the check that a corpus which grew a
        // register past it fails HERE, loudly, instead of at some browser's
        // createBindGroupLayout. (It cannot be checked before the device is
        // opened, because the device is what the ask is spent on.)
        let ceiling = super::device::needed_bindings();
        for u in &passes {
            for d in &u.layout.decls {
                if d.binding >= ceiling {
                    return Err(format!(
                        "binding {} ({:?}) is at or above the asked ceiling {ceiling} — \
                         the corpus grew a register the device ask does not cover",
                        d.binding, d.name
                    ));
                }
            }
        }

        let px = u64::from(rw) * u64::from(rh);
        // A `Cell` rather than a `mut` total: the allocator closure below runs
        // interleaved with the streams, and a `&mut` capture would lock the
        // total for the closure's whole lifetime.
        let bytes = std::cell::Cell::new(0u64);
        let sb = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let buf = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            bytes.set(bytes.get() + size);
            dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let accum = buf("accum", px * 12, sb);
        let tbuf = buf("tbuf", px * 4, sb);
        let info = buf("info", px * 4, sb);
        let counters = buf("counters", u64::from(gs::CTR_TOTAL) * 4, sb);
        // The G-buffer pack, at ONE STRIDE — this tracer arms no pack, and
        // `FLAG_GBUF` is what stands between the stores and an out-of-bounds
        // write. WebGPU bounds-clamps a storage access, so the consequence
        // here is a lost store rather than the memory corruption the native
        // backends face; the sizing follows them anyway, because "the flag is
        // the safety boundary" is a property of the SHADER and should not
        // read differently per backend.
        let gbuf = buf("gbuf", GBUF_STRIDE, sb);
        let gbuf_ext = buf("gbuf_ext", GBUF_EXT_STRIDE, sb);

        // The amortized cloud lattice: one float4 per point, one point of
        // border past each far edge — `TraceGpu::new`'s sizing verbatim,
        // against the SNAPSHOT k.
        let shift = srcs.sky_lod.trailing_zeros();
        let lw = u64::from(rw >> shift) + 2;
        let lh = u64::from(rh >> shift) + 2;
        let cloud_lod = buf("cloud_lod", (lw * lh).max(1) * 16, sb);
        // Sized at the CAP: the live side is derived per frame from the sun's
        // footprint, so the allocation cannot track it.
        let csn = if srcs.cloud_shadow_n > 0 {
            u64::from(crate::clouds::CLOUD_SHADOW_MAX)
        } else {
            1
        };
        let cloud_shadow = buf("cloud_shadow", csn * csn * 4, sb);

        // The ladder's queue caps come from `caps_for`, which is also what the
        // DEVICE ASK was sized from before this device existed — one
        // derivation, two callers (see its doc comment). Hoisted ABOVE the push
        // ring because the ring is sized from the PROGRAM and the program's
        // length is a function of the quadtree depth.
        let caps = caps_for(rw, rh, gs::sw_rays_leaf() && srcs.ftree_on);
        let dd = caps.depth;
        if dd > 11 {
            return Err(format!("{rw}x{rh} needs quadtree depth {dd} > 11 indirect-arg slots"));
        }

        let ub = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let frame_cb = buf("frame_cb", CB_STRIDE as u64, ub);
        // The ring stride is the DEVICE's alignment, read back from what was
        // granted rather than assumed to be the 256-byte default — a device
        // that grants a coarser one would silently misalign every row.
        let push_stride = u64::from(dev.limits.min_uniform_buffer_offset_alignment).max(16);
        let push_rows = push_rows_for(dd, px, true);
        let push = buf("push ring", push_stride * push_rows, ub);

        let binary = super::scene::stream(dev, "bvh", &bvh.nodes, crate::gfx::scene::gpu_bvh_node, sb);
        let sw_tri = super::scene::stream(dev, "tri_idx", &bvh.tri_idx, |t| *t, sb);
        bytes.set(bytes.get() + binary.size() + sw_tri.size());

        // ---- the ladder ----
        let ft = if srcs.ftree_on { Some(crate::ftree::FTree::build(bvh)) } else { None };
        let (cap_tile, cap_leaf, cap_cut) = (caps.tile, caps.leaf, caps.cut);
        let (cap_hcell, cap_hcut) = hemi_caps();
        let mut cb_base = FrameCb::base(scene, rw, rh);
        // The kernels read every one of these off the cbuffer, so a buffer
        // sized here and a cap written there must be the same number — hence
        // one place computing both, and why the hemi three come from
        // `hemi_caps()` rather than from a second copy of its arithmetic.
        cb_base.set_caps(
            cap_tile as u32,
            cap_leaf as u32,
            cap_leaf as u32,
            cap_cut as u32,
            cap_hcell as u32,
            cap_hcell as u32,
            cap_hcut as u32,
        );

        let tree = match &ft {
            // The QUANTIZED wire format — the per-processor split `ftree.rs`
            // documents, not a shortcut: the GPU trades decode ALU for -56%
            // tree bandwidth and the decoded boxes still CONTAIN the true
            // ones, so every prune stays conservative.
            Some(f) => super::scene::stream(dev, "ftree", &(0..f.nodes.len()).collect::<Vec<_>>(), |&i| f.quantize_node(i), sb),
            None => super::scene::stream(dev, "ftree(binary)", &bvh.nodes, crate::gfx::scene::gpu_bvh_node, sb),
        };
        // Gated on `sw_rays_leaf` rather than `sw_rays` for the reason the
        // HLSL is: under `--no-cut-rays` the leaf traverses from the root,
        // `level_finish` compiles no translation, and the wavefront unit
        // declares no t1 at all.
        let ft_bnode = match &ft {
            Some(f) if gs::sw_rays_leaf() => Some(super::scene::stream(
                dev,
                "ft_bnode",
                &(0..f.nodes.len() * 8).collect::<Vec<_>>(),
                |&i| f.bnode_at(i),
                sb,
            )),
            _ => None,
        };
        bytes.set(bytes.get() + tree.size() + ft_bnode.as_ref().map_or(0, |b| b.size()));
        let wave = Wave {
            qa: buf("wf qa", cap_tile * 24, sb),
            qb: buf("wf qb", cap_tile * 24, sb),
            qleaf: buf("wf qleaf", cap_leaf * gs::LEAF_REC_BYTES, sb),
            qsky: buf("wf qsky", cap_leaf * 16, sb),
            cut_pool: buf("wf cut pool", cap_cut * 256, sb),
            args: buf("wf args", 16 * gs::ARG_STRIDE, sb | wgpu::BufferUsages::INDIRECT),
            tree,
            ft_bnode,
            depth_full: dd,
            cap_leaf: cap_leaf as u32,
            cap_sky: cap_leaf as u32,
        };

        // ---- the hemisphere tiers ----
        //
        // Built unconditionally, as the Vulkan port builds them: this tracer
        // always has a ladder, and a tier that existed only on some runs would
        // mean the gate covers a configuration and the browser ships another.
        // The cost is ~280 MB, which is what the per-batch reset buys back —
        // see [`Hemi`].
        let hemi = Hemi {
            hq_a: buf("hemi qa", cap_hcell * 64, sb),
            hq_b: buf("hemi qb", cap_hcell * 64, sb),
            hq_leaf: buf("hemi qleaf", cap_hcell * 64, sb),
            cut: buf("hemi cut pool", cap_hcut * 256, sb),
            partial: buf("hemi partial", px * 12, sb),
            ambw: buf("hemi ambw", px * 12, sb),
            hbuf: buf("hemi hbuf", px * 16, sb),
            // 32 bytes a point, not 16 — one `HemiPointRec` carries the
            // position, the normal and the pixel it belongs to. `COPY_DST` is
            // already in `sb`, which is what lets the probe path write points
            // from the host where Vulkan asks for a host-visible allocation.
            pts: buf("hemi pts", px * 32, sb),
            cap_cell: cap_hcell,
            cap_cut: cap_hcut,
        };

        let hdr_tex = dev.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr"),
            size: wgpu::Extent3d { width: rw, height: rh, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        bytes.set(bytes.get() + px * 8);
        let hdr = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut t = WgpuTracer {
            rw,
            rh,
            accum,
            tbuf,
            info,
            counters,
            gbuf,
            gbuf_ext,
            cloud_lod,
            cloud_shadow,
            frame_cb,
            push,
            push_stride,
            push_rows,
            binary,
            sw_tri,
            wave: Some(wave),
            hemi: Some(hemi),
            _hdr: hdr_tex,
            hdr,
            passes: Vec::new(),
            cb_base,
            scene_aabb: crate::gfx::scene::shadow_aabb(scene),
            sky_lod_k: srcs.sky_lod,
            cloud_shadow_n: srcs.cloud_shadow_n,
            wgsl_bytes,
            bytes: bytes.get(),
        };
        // Bind groups last: they borrow every buffer above, so they are built
        // once the tracer owns them.
        let built = passes
            .into_iter()
            .map(|u| {
                let groups = VARIANTS
                    .iter()
                    .map(|&v| t.bind_variant(dev, ws, &u, v))
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(Pass { unit: u, groups })
            })
            .collect::<Result<Vec<Pass>, String>>()?;
        t.passes = built;
        Ok(t)
    }

    /// (group, register) -> resource, and the ONE place the register map is
    /// stated on this backend.
    ///
    /// Named by REGISTER with `binding_of` doing the translation — the
    /// never-a-literal rule — and structured as OVERRIDES over a shared base
    /// so a variant is a diff rather than a second table. The lookup takes
    /// the FIRST match, so overrides go first.
    ///
    /// A slot the corpus declares and this table has nothing for falls
    /// through to `scene.dummy`, exactly as the Vulkan twin's does. That
    /// fall-through is load-bearing and dangerous in equal measure — the
    /// Vulkan suite's own worst bug was a real stream landing on the dummy
    /// and shading a whole frame as triangle 0 — which is why the gate's
    /// radiance A/B against the CPU is what proves the table, and why
    /// `FR_WGPU_MAP` prints REAL-vs-dummy per slot.
    fn resolve<'a>(
        &'a self,
        ws: &'a WgpuScene,
        variant: Variant,
        group: u32,
        binding: u32,
    ) -> Option<Res<'a>> {
        let mut table: Vec<(u32, Reg, u32, Res<'a>)> = Vec::new();
        match variant {
            // The TERMINAL meaning of the overloaded registers: u5/u6 are the
            // cloud lattice and the slab-space shadow cache, and t0/t1 are the
            // BINARY tree and its triangle ids — the software intersector the
            // reference, leaf and sky passes all run.
            Variant::Terminal => table.extend([
                (0, Reg::U, 5, Res::Buf(&self.cloud_lod)),
                (0, Reg::U, 6, Res::Buf(&self.cloud_shadow)),
                (0, Reg::T, 0, Res::Buf(&self.binary)),
                (0, Reg::T, 1, Res::Buf(&self.sw_tri)),
            ]),
            // The ladder keeps the WIDE tree at t0 (a frustum query cannot
            // descend the binary one) and takes the slot -> node map at t1
            // when the lever arms it. Level 0 consumes queue A, so the EVEN
            // parity has A as `qin`.
            Variant::LadderA | Variant::LadderB => {
                if let Some(w) = &self.wave {
                    let (q0, q1) = if variant == Variant::LadderA {
                        (&w.qa, &w.qb)
                    } else {
                        (&w.qb, &w.qa)
                    };
                    table.extend([
                        (0, Reg::T, 0, Res::Buf(&w.tree)),
                        (0, Reg::U, 5, Res::Buf(q0)),
                        (0, Reg::U, 6, Res::Buf(q1)),
                    ]);
                    if let Some(bn) = &w.ft_bnode {
                        table.push((0, Reg::T, 1, Res::Buf(bn)));
                    }
                }
            }
            // FIVE moved slots, not two: the cell ping-pong, the hemi leaf
            // queue and cut pool at u7/u9 — which OVERRIDE the ladder's own
            // queues at those registers, since the hemi units re-declare them
            // as `HemiCellRec` streams — and t0 back to the BINARY tree,
            // because these kernels compile the binary `bound_query`
            // deliberately. `tri_idx` follows t0, since `cs_hemi_leaf` shoots
            // its rays through the same software loops.
            //
            // On this backend that re-declaration needs no argument at all,
            // where the Vulkan twin owes one: layouts here are DERIVED PER
            // ENTRY POINT from each unit's own naga IR, so a hemi entry's
            // layout already describes a `HemiCellRec` queue where the ladder's
            // describes a `TileRec` one. Nothing is shared to conflict.
            Variant::HemiA | Variant::HemiB => {
                if let Some(h) = &self.hemi {
                    let (q0, q1) = if variant == Variant::HemiA {
                        (&h.hq_a, &h.hq_b)
                    } else {
                        (&h.hq_b, &h.hq_a)
                    };
                    table.extend([
                        (0, Reg::T, 0, Res::Buf(&self.binary)),
                        (0, Reg::T, 1, Res::Buf(&self.sw_tri)),
                        (0, Reg::U, 5, Res::Buf(q0)),
                        (0, Reg::U, 6, Res::Buf(q1)),
                        (0, Reg::U, 7, Res::Buf(&h.hq_leaf)),
                        (0, Reg::U, 9, Res::Buf(&h.cut)),
                    ]);
                }
            }
        }
        if let Some(h) = &self.hemi {
            // The per-pixel planes and the shading-point queue. These do NOT
            // vary by variant — only the queues behind u5..u9 and the tree at
            // t0 do — so they belong in the base beside everything else, and
            // the compose pass reads them under TERMINAL.
            table.extend([
                (0, Reg::U, 10, Res::Buf(&h.partial)),
                (0, Reg::U, 11, Res::Buf(&h.ambw)),
                (0, Reg::U, 12, Res::Buf(&h.hbuf)),
                (0, Reg::U, 13, Res::Buf(&h.pts)),
            ]);
        }
        if let Some(w) = &self.wave {
            // The ladder's own streams, in the BASE rather than a variant:
            // every parity holds them identically.
            table.extend([
                (0, Reg::U, 4, Res::Buf(&w.args)),
                (0, Reg::U, 7, Res::Buf(&w.qleaf)),
                (0, Reg::U, 8, Res::Buf(&w.qsky)),
                (0, Reg::U, 9, Res::Buf(&w.cut_pool)),
            ]);
        }
        table.extend([
            (0, Reg::B, 0, Res::Buf(&self.frame_cb)),
            (0, Reg::B, 1, Res::Window(&self.push, 16)),
            (0, Reg::T, 2, Res::Buf(&ws.positions)),
            (0, Reg::T, 3, Res::Buf(&ws.normals)),
            (0, Reg::T, 4, Res::Buf(&ws.indices)),
            (0, Reg::T, 5, Res::Buf(&ws.tri_mat)),
            (0, Reg::T, 6, Res::Buf(&ws.materials)),
            (0, Reg::U, 0, Res::Buf(&self.accum)),
            (0, Reg::U, 1, Res::Buf(&self.tbuf)),
            (0, Reg::U, 2, Res::Buf(&self.info)),
            (0, Reg::U, 3, Res::Buf(&self.counters)),
            (0, Reg::U, 14, Res::Tex(&self.hdr)),
            (0, Reg::U, 15, Res::Buf(&self.gbuf)),
            (0, Reg::U, 32, Res::Buf(&self.gbuf_ext)),
            (1, Reg::T, 0, Res::Buf(&ws.uv_buf)),
            (1, Reg::T, 1, Res::Buf(&ws.indices)),
            (1, Reg::T, 2, Res::Buf(&ws.tri_mat)),
            (1, Reg::T, 3, Res::Buf(&ws.mat_cutout)),
            (1, Reg::T, 4, Res::Buf(&ws.positions)),
            (1, Reg::T, 5, Res::Buf(&ws.mat_height)),
            (1, Reg::T, 6, Res::Buf(&ws.mat_shadow)),
            (1, Reg::T, 7, Res::Buf(&ws.blas_tri)),
            (1, Reg::T, 8, Res::Buf(&ws.chunk_base)),
            // The browser texture plan (gfx::texweb): the meta rows, the
            // mip-0 texel payload, and one Texture2DArray per bucket at
            // consecutive registers from BUCKET_REG0.
            (1, Reg::T, crate::gfx::texweb::META_REG, Res::Buf(&ws.tex.meta)),
            (1, Reg::T, crate::gfx::texweb::TEXELS_REG, Res::Buf(&ws.tex.texels)),
            (1, Reg::S, 0, Res::Samp(&ws.tex.samp_lin)),
            (1, Reg::S, 1, Res::Samp(&ws.tex.samp_aniso)),
        ]);
        for (bi, v) in ws.tex.buckets.iter().enumerate() {
            table.push((1, Reg::T, crate::gfx::texweb::BUCKET_REG0 + bi as u32, Res::Tex(v)));
        }
        table
            .into_iter()
            .find(|&(g, r, n, _)| g == group && binding_of(r, n) == binding)
            .map(|(_, _, _, res)| res)
    }

    /// One entry point's bind groups for one variant, indexed by group.
    fn bind_variant(
        &self,
        dev: &Wgpu,
        ws: &WgpuScene,
        u: &Unit,
        variant: Variant,
    ) -> Result<Vec<Option<wgpu::BindGroup>>, String> {
        let map = std::env::var_os("FR_WGPU_MAP").is_some();
        let mut out = Vec::with_capacity(u.layout.groups.len());
        for (g, bgl) in u.layout.groups.iter().enumerate() {
            let Some(bgl) = bgl else {
                out.push(None);
                continue;
            };
            let g = g as u32;
            let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();
            for d in u.layout.group_decls(g) {
                let hit = self.resolve(ws, variant, g, d.binding);
                if map {
                    eprintln!(
                        "check-wgpu:   bind group {g} binding {} <- {} ({:?})",
                        d.binding,
                        if hit.is_some() { "REAL" } else { "dummy" },
                        d.name
                    );
                }
                let resource = match (hit, d.kind) {
                    (Some(Res::Buf(b)), _) => b.as_entire_binding(),
                    (Some(Res::Window(b, size)), _) => {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: b,
                            offset: 0,
                            size: std::num::NonZeroU64::new(size),
                        })
                    }
                    (Some(Res::Tex(v)), _) => wgpu::BindingResource::TextureView(v),
                    (Some(Res::Samp(s)), _) => wgpu::BindingResource::Sampler(s),
                    // Nothing bound. A BUFFER slot takes the dummy; a texture
                    // or sampler slot cannot, and a corpus that declared one
                    // this table has no resource for is a defect rather than
                    // something to paper over — WebGPU has no null descriptor.
                    (None, BindKind::Uniform | BindKind::Storage { .. }) => {
                        ws.dummy.as_entire_binding()
                    }
                    (None, k) => {
                        return Err(format!(
                            "no resource for group {g} binding {} ({:?}, {k:?}) — a texture or \
                             sampler slot has no dummy",
                            d.binding, d.name
                        ));
                    }
                };
                entries.push(wgpu::BindGroupEntry { binding: d.binding, resource });
            }
            out.push(Some(dev.scoped("bind group", || {
                dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: bgl,
                    entries: &entries,
                })
            })?));
        }
        Ok(out)
    }

    /// The frame cbuffer. The Vulkan twin's `write_cb` with every optional
    /// half OFF: this tracer arms no G-buffer pack, no feed and no NRD
    /// bridge, so the pack flags stay false — which is not a default but the
    /// safety boundary, since the two pack buffers are ONE STRIDE (see their
    /// allocation).
    fn write_cb(&self, dev: &Wgpu, p: &FrameParams) {
        let mut cb =
            self.cb_base.with_frame(p, false, false, false, false, false, false, false, (false, false), false);
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
        dev.queue.write_buffer(&self.frame_cb, 0, cb.bytes());
    }

    /// Write one row of the push ring. Every row a frame needs is written
    /// BEFORE the submit (module header), and a dispatch selects its row with
    /// the dynamic offset `push_at` returns.
    fn write_push(&self, dev: &Wgpu, row: u64, words: [u32; 4]) {
        let b: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        dev.queue.write_buffer(&self.push, row * self.push_stride, &b);
    }

    fn push_at(&self, row: u64) -> u32 {
        (row * self.push_stride) as u32
    }

    /// Bind one entry point's groups for one variant, with the push row's
    /// dynamic offset where that entry declares the push block.
    fn bind(&self, p: &mut wgpu::ComputePass<'_>, pass: usize, variant: Variant, row: u64) {
        let pa = &self.passes[pass];
        p.set_pipeline(&pa.unit.pipeline);
        let offs = [self.push_at(row)];
        for (g, bg) in pa.groups[variant as usize].iter().enumerate() {
            if let Some(bg) = bg {
                // The dynamic-offset slice must have exactly as many entries
                // as the LAYOUT declared dynamic bindings in that group — so
                // it is derived from the same decls the layout was, and a unit
                // that declares no push block gets an empty slice.
                let dyn_here = pa
                    .unit
                    .layout
                    .group_decls(g as u32)
                    .any(|d| d.group == 0 && d.binding == binding_of(Reg::B, 1));
                p.set_bind_group(g as u32, bg, if dyn_here { &offs } else { &[] });
            }
        }
    }

    /// Bind and dispatch one entry point under one variant.
    fn go(&self, p: &mut wgpu::ComputePass<'_>, pass: usize, variant: Variant, row: u64, gx: u32, gy: u32) {
        self.bind(p, pass, variant, row);
        p.dispatch_workgroups(gx, gy, 1);
    }

    /// The two per-frame cloud caches, appended to a step program.
    ///
    /// `cs_reference` and `cs_sky` read the amortized sky lattice
    /// (`--sky-lod`) and the slab-space cloud-shadow cache (`--cloud-shadow`)
    /// at registers the wavefront otherwise uses for tile queues, so a tracer
    /// that skipped these fills would read whatever those buffers happen to
    /// contain and shade a black sky. Forcing both levers off would make the
    /// gate cover a configuration nobody ships.
    ///
    /// Neither kernel reads the push block; they inherit the row the caller
    /// last planned, which is why they carry it forward rather than zeroing
    /// it — a row is a dispatch's own, and copying keeps that true.
    fn plan_cloud_caches(&self, steps: &mut Vec<Step>) {
        let carry = steps.last().map_or([0; 4], |s| s.push());
        let sky_pts = ((self.rw / self.sky_lod_k) + 2) * ((self.rh / self.sky_lod_k) + 2);
        let sky_groups = sky_pts.div_ceil(64);
        let csn_groups =
            (crate::clouds::CLOUD_SHADOW_MAX * crate::clouds::CLOUD_SHADOW_MAX).div_ceil(64);
        if self.cloud_shadow_n > 0 {
            steps.push(Step::Direct {
                pass: P_CLOUD_SHADOW,
                variant: Variant::Terminal,
                push: carry,
                gx: csn_groups.min(32768),
                gy: csn_groups.div_ceil(32768),
            });
        }
        if self.sky_lod_k > 1 {
            steps.push(Step::Direct {
                pass: P_SKY_LOD,
                variant: Variant::Terminal,
                push: carry,
                gx: sky_groups.min(32768),
                gy: sky_groups.div_ceil(32768),
            });
        }
    }

    /// One REFERENCE frame: the two cache fills, the reference dispatch, then
    /// resolve — all in one submit.
    pub fn render(&self, hg: &WgpuHeadless, p: &FrameParams, samples: u32) -> Result<(), String> {
        self.write_cb(&hg.dev, p);
        // `cbuffer Push : register(b1)` is 4 dwords; only the first is read
        // here (`inv_samples`), but the whole row is written so a slot never
        // holds the previous frame's bytes.
        let inv = 1.0f32 / samples.max(1) as f32;
        let gx = self.rw.div_ceil(8);
        let gy = self.rh.div_ceil(8);
        let mut steps: Vec<Step> = Vec::new();
        self.plan_cloud_caches(&mut steps);
        for pass in [P_REFERENCE, P_RESOLVE] {
            steps.push(Step::Direct {
                pass,
                variant: Variant::Terminal,
                push: [inv.to_bits(), 0, 0, 0],
                gx,
                gy,
            });
        }
        // The cloud fills inherit whichever row precedes them, and with
        // nothing before them that is the zero carry — so give them the real
        // one rather than leaving `inv_samples` at zero in a row they never
        // read but a future kernel might.
        for s in &mut steps {
            if let Step::Direct { push, .. } = s {
                *push = [inv.to_bits(), 0, 0, 0];
            }
        }
        self.run_steps(hg, &steps, true)
    }

    /// ONE WAVEFRONT QUADTREE FRAME: seed -> depth_full x (prep-args ->
    /// level) -> the leaf + sky terminal fills.
    ///
    /// STATICALLY RECORDED, exactly as on D3D12 and Vulkan — every scheduling
    /// decision after the seed is a GPU-written counter feeding
    /// `dispatch_workgroups_indirect`, so an empty level dispatches zero
    /// groups rather than being skipped by the CPU. That is the property the
    /// whole design rests on and the reason there is no readback in here.
    ///
    /// `clear_sentinel` floods `info` with `0xffffffff` first, which is what
    /// makes the exactly-once coverage gate possible: a pixel no terminal
    /// record covered still reads the sentinel afterwards.
    ///
    /// COMPOSE IS PLANNED ONLY UNDER fb, and that is D3D12's and Vulkan's rule
    /// verbatim rather than an omission: with fb off the leaf and sky passes
    /// splat straight into `accum` through `queues.hlsli::accum_splat`, so
    /// compose would be a buffer-to-buffer copy of a full screen.
    pub fn render_wavefront(
        &self,
        hg: &WgpuHeadless,
        p: &FrameParams,
        clear_sentinel: bool,
    ) -> Result<(), String> {
        let w = self.wave.as_ref().ok_or("this tracer has no wavefront ladder")?;
        // Read from the SAME function that writes the cbuffer's `fb_mode`, so
        // the kernels this plans and the constant they branch on cannot
        // disagree about which tier is running.
        let fb_mode = fb_mode_of(&p.q);
        let dev = &hg.dev;
        self.write_cb(dev, p);

        let clear_groups = (self.rw * self.rh).div_ceil(256);
        let wide_on = gs::WIDE_LEVELS_ON.load(std::sync::atomic::Ordering::Relaxed);
        let wide_n = gs::wide_levels();

        // ---- the program, as data (see `Step`) ----
        let mut steps: Vec<Step> = Vec::new();
        // push0 = 0: this backend has no work-graph arm, so the seed always
        // enqueues its own root. Written rather than inherited — the ring
        // holds whatever the last frame left.
        steps.push(Step::Direct {
            pass: P_SEED,
            variant: Variant::LadderA,
            push: [0, 0, 0, 0],
            gx: 1,
            gy: 1,
        });
        if clear_sentinel {
            steps.push(Step::Direct {
                pass: P_CLEAR_INFO,
                variant: Variant::LadderA,
                push: [0, 0, 0, 0],
                gx: clear_groups.min(32768),
                gy: clear_groups.div_ceil(32768),
            });
        }
        self.plan_clear_h(&mut steps, fb_mode);
        for lvl in 0..w.depth_full {
            let (in_ctr, out_ctr) = if lvl % 2 == 0 {
                (gs::CTR_TILE_A, gs::CTR_TILE_B)
            } else {
                (gs::CTR_TILE_B, gs::CTR_TILE_A)
            };
            let variant = if lvl % 2 == 0 { Variant::LadderA } else { Variant::LadderB };
            // Shallow levels take the wave-cooperative kernel: ONE GROUP per
            // tile instead of one thread per tile (level 0 is a single tile,
            // so the serial ladder would run one lane over the whole BVH).
            let wide = wide_on && lvl < wide_n;
            // prep and the level kernel it feeds run under the SAME variant:
            // prep touches only `counters` and `args`, which every variant
            // holds identically, so the parity bind covers both.
            steps.push(Step::Direct {
                pass: P_PREP,
                variant,
                push: [in_ctr, out_ctr, if wide { 1 } else { 32 }, lvl],
                gx: 1,
                gy: 1,
            });
            steps.push(Step::Indirect {
                pass: if wide { P_LEVEL_WIDE } else { P_LEVEL },
                variant,
                push: [in_ctr, out_ctr, 0, 0],
                slot: lvl,
            });
        }
        self.plan_terminal_fills(&mut steps, fb_mode);
        self.plan_hemi_tail(&mut steps, fb_mode, p.q.fb.depth, clear_groups);
        self.run_steps(hg, &steps, true)
    }

    /// STRUCTURE REPLAY: a bit-equal-basis frame re-dispatches the persisted
    /// terminal queues and skips the seed and the WHOLE level ladder. The
    /// ladder is the wavefront's fixed cost, and on a parked camera this
    /// deletes it (D3D12 measured -43% of the GPU frame span there).
    ///
    /// Soundness is entirely in one sentence: the terminal structure is a pure
    /// function of (scene, BVH, basis, rw, rh), while spp/jitter/frame/fb/
    /// quality/clouds all ride the cbuffer — so a replay frame re-shades from a
    /// fresh `FrameParams` against a structure that provably still describes
    /// this view, and the result must be BIT-IDENTICAL to a fresh trace. That
    /// is a gate, not a hope: J9 compares tbuf/info/accum bitwise.
    ///
    /// The queues stay byte-intact between producing frames because only
    /// `cs_seed` and the ladder ever WRITE them — the leaf and sky passes read,
    /// the hemi passes rebind u5..u9 to their own transients, and the
    /// reference/resolve units declare no queues at all.
    ///
    /// THE CALLER PROVES THE BIT-EQUALITY. D3D12 keeps a `last_struct` key and
    /// auto-selects inside `record_frame`; there is no per-frame driver on this
    /// backend yet, so that predicate lands with the presenter (Stage D) rather
    /// than being written here as a field nothing reads.
    pub fn render_wavefront_replay(
        &self,
        hg: &WgpuHeadless,
        p: &FrameParams,
        clear_sentinel: bool,
    ) -> Result<(), String> {
        self.wave.as_ref().ok_or("structure replay is the ladder's, and this tracer has none")?;
        let fb_mode = fb_mode_of(&p.q);
        self.write_cb(&hg.dev, p);
        let clear_groups = (self.rw * self.rh).div_ceil(256);

        // The TERMINAL variant throughout the head: nothing here touches the
        // tile queues, so the ladder's A/B parities have no work to do.
        let mut steps: Vec<Step> = vec![Step::Direct {
            pass: P_SEED_REPLAY,
            variant: Variant::Terminal,
            push: [0, 0, 0, 0],
            gx: 1,
            gy: 1,
        }];
        if clear_sentinel {
            steps.push(Step::Direct {
                pass: P_CLEAR_INFO,
                variant: Variant::Terminal,
                push: [0, 0, 0, 0],
                gx: clear_groups.min(32768),
                gy: clear_groups.div_ceil(32768),
            });
        }
        self.plan_clear_h(&mut steps, fb_mode);
        self.plan_terminal_fills(&mut steps, fb_mode);
        self.plan_hemi_tail(&mut steps, fb_mode, p.q.fb.depth, clear_groups);
        // NO COUNTER CLEAR: `cs_seed_replay` keeps the terminal structure it is
        // about to re-dispatch. See `run_steps`.
        self.run_steps(hg, &steps, false)
    }

    /// Zero the fixed-point H accumulator, once per fb FRAME.
    ///
    /// MANDATORY rather than tidy, and the Vulkan port had it missing from the
    /// frame path until the replay factoring put the two next to each other:
    /// `hbuf` is written by ATOMIC ADD (that is what makes the integrator
    /// order-independent), so an unzeroed frame integrates on top of the
    /// previous one's answer — and nothing downstream can tell, because compose
    /// folds whatever is there into `accum` and J8's frame half scores
    /// accounting, not radiance.
    fn plan_clear_h(&self, steps: &mut Vec<Step>, fb_mode: u32) {
        if fb_mode == 0 || self.hemi.is_none() {
            return;
        }
        let g = (self.rw * self.rh * 4).div_ceil(256);
        steps.push(Step::Direct {
            pass: P_CLEAR_H,
            variant: Variant::Terminal,
            push: [0, 0, 0, 0],
            gx: g.min(32768),
            gy: g.div_ceil(32768),
        });
    }

    /// The hemisphere wavefront plus its one compose splat. No-op with fb off,
    /// for the reason [`Self::render_wavefront`]'s header gives.
    fn plan_hemi_tail(
        &self,
        steps: &mut Vec<Step>,
        fb_mode: u32,
        fb_depth: u32,
        clear_groups: u32,
    ) {
        if fb_mode == 0 || self.hemi.is_none() {
            return;
        }
        // Every hit pixel appended a point, so batch over the WORST CASE;
        // batches past the GPU-side count dispatch zero groups, which is what
        // lets this be planned statically with no readback anywhere.
        self.plan_hemi(steps, u64::from(self.rw) * u64::from(self.rh), fb_depth);
        // partial + ambW * ambient(H) -> accum: the single splat, and the ONE
        // pass in the tracer that is per-PIXEL rather than queue-driven.
        steps.push(Step::Direct {
            pass: P_COMPOSE,
            variant: Variant::Terminal,
            push: [0, 0, 0, 0],
            gx: clear_groups.min(32768),
            gy: clear_groups.div_ceil(32768),
        });
    }

    /// The hemisphere wavefront over the points in `pts`, in `HEMI_BATCH`
    /// slices. Each batch resets the transient cell queues and cut pool
    /// (`cs_prep_batch`), and THAT reset is what bounds the memory: the caps
    /// size one batch, not one frame.
    ///
    /// The parity dance is one step off the ladder's, deliberately — see
    /// [`Variant::HemiB`] for what getting it backwards costs, and what
    /// planting it measured.
    fn plan_hemi(&self, steps: &mut Vec<Step>, max_points: u64, fb_depth: u32) {
        let n_batches = max_points.div_ceil(u64::from(HEMI_BATCH));
        let levels = fb_depth.clamp(2, HEMI_MAX_DEPTH) - 1;
        for b in 0..n_batches {
            let base = (b * u64::from(HEMI_BATCH)) as u32;
            // Batch prep: the root pass's args PLUS the batch-scoped counter
            // reset — one kernel, because the reset has to happen before
            // anything in the batch enqueues.
            steps.push(Step::Direct {
                pass: P_PREP_BATCH,
                variant: Variant::HemiB,
                push: [gs::CTR_HEMI_PT, base, 32, ARG_HEMI_ROOT],
                gx: 1,
                gy: 1,
            });
            steps.push(Step::Indirect {
                pass: P_HEMI_ROOT,
                variant: Variant::HemiB,
                push: [base, gs::CTR_HEMI_A, 0, 0],
                slot: ARG_HEMI_ROOT,
            });

            for l in 0..levels {
                let (in_ctr, out_ctr) = if l % 2 == 0 {
                    (gs::CTR_HEMI_A, gs::CTR_HEMI_B)
                } else {
                    (gs::CTR_HEMI_B, gs::CTR_HEMI_A)
                };
                let variant = if l % 2 == 0 { Variant::HemiA } else { Variant::HemiB };
                steps.push(Step::Direct {
                    pass: P_PREP,
                    variant,
                    push: [in_ctr, out_ctr, 32, ARG_HEMI_CELL],
                    gx: 1,
                    gy: 1,
                });
                steps.push(Step::Indirect {
                    pass: P_HEMI_CELL,
                    variant,
                    push: [in_ctr, out_ctr, 0, 0],
                    slot: ARG_HEMI_CELL,
                });
            }

            // Leaf rays: FOUR threads per leaf cell (one stratified Arvo ray
            // per midpoint sub-cell), so 8 records per 32-wide group.
            steps.push(Step::Direct {
                pass: P_PREP,
                variant: Variant::HemiA,
                push: [gs::CTR_HEMI_LEAF, NO_RESET, 8, ARG_HEMI_LEAF],
                gx: 1,
                gy: 1,
            });
            steps.push(Step::Indirect {
                pass: P_HEMI_LEAF,
                variant: Variant::HemiA,
                push: [0, 0, 0, 0],
                slot: ARG_HEMI_LEAF,
            });
        }
    }

    /// The leaf and sky fills, appended to a step program. Shared so the
    /// replay path records the SAME code rather than a second block that looks
    /// like it — that factoring was the only real work in porting replay, and
    /// it is what makes "a replayed frame is bit-identical" a claim about one
    /// piece of code rather than about two that agree today.
    fn plan_terminal_fills(&self, steps: &mut Vec<Step>, fb_mode: u32) {
        steps.push(Step::Direct {
            pass: P_PREP,
            variant: Variant::Terminal,
            push: [gs::CTR_LEAF, NO_RESET, 1, ARG_LEAF],
            gx: 1,
            gy: 1,
        });
        // Sky takes the MULTIPLYING prep: SKY_SPLIT groups share each record,
        // so one huge proven-empty rect cannot serialize on a single group
        // (that shape was ~70% of the tracer's frame once).
        steps.push(Step::Direct {
            pass: P_PREP_MUL,
            variant: Variant::Terminal,
            push: [gs::CTR_SKY, NO_RESET, gs::SKY_SPLIT, ARG_SKY],
            gx: 1,
            gy: 1,
        });
        // Both cloud caches, ahead of BOTH consumers — `cs_sky` on the
        // proven-empty rects and `cs_leaf`'s own miss branch.
        self.plan_cloud_caches(steps);
        // fb frames take the OTHER leaf pipeline — same source, hemi arm
        // compiled IN — because that arm is what appends the shading points the
        // hemisphere passes below consume. Sharing one pipeline here would
        // leave the hemi point queue empty and every gate below vacuous.
        let leaf = if fb_mode > 0 && self.hemi.is_some() { P_LEAF_FB } else { P_LEAF };
        steps.push(Step::Indirect {
            pass: leaf,
            variant: Variant::Terminal,
            push: [gs::CTR_LEAF, 0, 0, 0],
            slot: ARG_LEAF,
        });
        steps.push(Step::Indirect {
            pass: P_SKY,
            variant: Variant::Terminal,
            push: [gs::CTR_SKY, 0, 0, 0],
            slot: ARG_SKY,
        });
    }

    /// Write every push row, then record every dispatch.
    ///
    /// TWO WALKS OVER ONE PROGRAM, and the split is the WebGPU constraint in
    /// its plainest form: `queue.write_buffer` lands before the submit, so a
    /// row cannot be written between two dispatches. Each step gets its own
    /// ring row, which makes a row's lifetime exactly one dispatch and
    /// removes the write-after-read hazard the Vulkan twin has to barrier
    /// for by hand (its `push` carries a WAR edge whose omission silently
    /// cost the whole ladder past level 0 on the first run).
    /// `clear_counters` is the PROGRAM's decision, not this function's, and
    /// making it a parameter is the fix for two bugs one blanket clear caused.
    /// Both paths that keep counters across a submit spell their keep-set in
    /// the KERNEL, and a host-side clear silently overrode it:
    ///
    /// * `cs_seed_replay` keeps CTR_LEAF / CTR_SKY / CTR_CUT / CTR_SKY_PX —
    ///   the whole terminal structure it is about to re-dispatch. Cleared, the
    ///   fills launched zero groups and the replay wrote NOTHING (J9 measured
    ///   120000 poison survivors).
    /// * `cs_seed_probes` keeps the verify and stats counters across
    ///   accumulate seeds deliberately, so the exact-zero gates observe every
    ///   seed's rays. Cleared, J8 scored one seed in eight and still PASSED —
    ///   the quieter of the two by far, and the reason this is a parameter
    ///   rather than a `!replay` special case. (Measured: leaf-rays 344 with
    ///   the clear, 2752 without. Exactly 8x, and the gate never said a word.)
    ///
    /// So the caller passes what its seed kernel expects, and the two
    /// statements of the keep-set cannot drift apart.
    fn run_steps(
        &self,
        hg: &WgpuHeadless,
        steps: &[Step],
        clear_counters: bool,
    ) -> Result<(), String> {
        let dev = &hg.dev;
        if steps.len() as u64 > self.push_rows {
            return Err(format!(
                "{} dispatches in one frame exceeds the {}-row push ring",
                steps.len(),
                self.push_rows
            ));
        }
        for (row, s) in steps.iter().enumerate() {
            self.write_push(dev, row as u64, s.push());
        }
        // The ladder's args buffer is needed only by an INDIRECT step, so a
        // reference-only program records without one — asking for it up front
        // would make the reference path depend on a ladder it never touches.
        let args = self.wave.as_ref().map(|w| &w.args);
        if args.is_none() && steps.iter().any(|s| matches!(s, Step::Indirect { .. })) {
            return Err("an indirect step with no ladder to launch from".into());
        }
        hg.run(|enc| {
            // Counters are PER FRAME, and `cs_seed` zeroes the ones it owns —
            // but `clear_buffer` here is what makes an "> 0" must-fire mean
            // something on the counters no kernel resets. See the doc comment
            // for why it is the caller's decision.
            if clear_counters {
                enc.clear_buffer(&self.counters, 0, None);
            }
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            for (row, s) in steps.iter().enumerate() {
                match *s {
                    Step::Direct { pass, variant, gx, gy, .. } => {
                        self.go(&mut p, pass, variant, row as u64, gx, gy);
                    }
                    Step::Indirect { pass, variant, slot, .. } => {
                        self.bind(&mut p, pass, variant, row as u64);
                        // NO BARRIER, and that this is sufficient is what
                        // `--check-wgpu` J3 already proved on the smoke:
                        // WebGPU's per-dispatch usage scopes are the
                        // synchronization, so the args buffer written by the
                        // preceding `cs_prep` is visible to the indirect read
                        // here without an explicit edge.
                        let args = args.expect("checked above");
                        p.dispatch_workgroups_indirect(args, u64::from(slot) * gs::ARG_STRIDE);
                    }
                }
            }
        })
    }

    /// The `--check-wgpu` probe path: upload a CPU-generated shading-point set
    /// and run ONLY the hemisphere passes over it — `run_hemi_probes`' peer on
    /// the CPU side.
    ///
    /// Both sides of the A/B then integrate at the EXACT same `(o, n)`, which
    /// is what makes a statistical comparison against a CPU cosine reference
    /// mean anything. The CB `frame` seeds the Arvo draws, so calling again
    /// with `clear = false` and a different frame ACCUMULATES another
    /// independent estimate into H — and `cs_seed_probes` keeps the verify
    /// counters across those passes deliberately, so the exact-zero gates
    /// observe every seed's rays rather than only the last seed's.
    pub fn run_hemi_probes(
        &self,
        hg: &WgpuHeadless,
        p: &FrameParams,
        probes: &[(glam::Vec3A, glam::Vec3A)],
        clear: bool,
    ) -> Result<(), String> {
        let h = self.hemi.as_ref().ok_or("this tracer has no hemisphere tiers")?;
        self.wave.as_ref().ok_or("this tracer has no wavefront ladder")?;
        if probes.len() > (self.rw * self.rh) as usize {
            return Err(format!("{} probes exceeds the hbuf/pts capacity", probes.len()));
        }
        let dev = &hg.dev;
        self.write_cb(dev, p);

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
        // Ordered before the submit below, which is what makes this the peer of
        // Vulkan's host-visible write rather than a staging copy needing its
        // own edge.
        dev.queue.write_buffer(&h.pts, 0, &bytes);

        let n = probes.len() as u32;
        let clear_groups = (self.rw * self.rh * 4).div_ceil(256);
        // Any variant would serve these two — they touch `counters` and `hbuf`,
        // which every variant holds identically — but planning them under the
        // one the batch loop starts with keeps the stream readable.
        let mut steps: Vec<Step> = vec![Step::Direct {
            pass: P_SEED_PROBES,
            variant: Variant::HemiB,
            push: [n, u32::from(clear), 0, 0],
            gx: 1,
            gy: 1,
        }];
        if clear {
            steps.push(Step::Direct {
                pass: P_CLEAR_H,
                variant: Variant::HemiB,
                push: [0, 0, 0, 0],
                gx: clear_groups.min(32768),
                gy: clear_groups.div_ceil(32768),
            });
        }
        self.plan_hemi(&mut steps, u64::from(n), p.q.fb.depth);
        // `clear` is the same flag `cs_seed_probes` takes as push1 — one
        // decision, spelled once on each side.
        self.run_steps(hg, &steps, clear)
    }

    /// The fixed-point H accumulator, as raw u32 — 4 per point (`x|y|z|psa`).
    pub fn read_hbuf(&self, hg: &WgpuHeadless, n_points: usize) -> Result<Vec<u32>, String> {
        let h = self.hemi.as_ref().ok_or("this tracer has no hemisphere tiers")?;
        let b = hg.read_buffer(&h.hbuf, n_points * 4 * 4)?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// Are the hemisphere tiers live? Always, today — the `Option` stays for
    /// the same reason `Wave`'s does.
    pub fn has_hemi(&self) -> bool {
        self.hemi.is_some()
    }

    /// The hemisphere tier's per-batch caps, for the gate's accounting line.
    pub fn hemi_caps(&self) -> (u64, u64) {
        self.hemi.as_ref().map_or((0, 0), |h| (h.cap_cell, h.cap_cut))
    }

    /// The ladder's structural queues, for the gate's accounting.
    pub fn read_queues(
        &self,
        hg: &WgpuHeadless,
        n_leaf: usize,
        n_sky: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        let w = self.wave.as_ref().ok_or("this tracer has no wavefront ladder")?;
        let le = |b: Vec<u8>| -> Vec<u32> {
            b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        // Clamped to the ALLOCATION, not to the reported count: a count that
        // overran its queue is exactly the failure the caller is about to
        // gate, and reading past the buffer would take the device out first.
        let leaf_bytes =
            (n_leaf as u64 * gs::LEAF_REC_BYTES).min(w.qleaf.size()) as usize;
        let sky_bytes = (n_sky as u64 * 16).min(w.qsky.size()) as usize;
        Ok((le(hg.read_buffer(&w.qleaf, leaf_bytes)?), le(hg.read_buffer(&w.qsky, sky_bytes)?)))
    }

    /// Entry points compiled — one WGSL module and one layout each.
    pub fn units(&self) -> usize {
        self.passes.len()
    }

    pub fn depth_full(&self) -> u32 {
        self.wave.as_ref().map_or(0, |w| w.depth_full)
    }

    pub fn caps(&self) -> (u32, u32) {
        self.wave.as_ref().map_or((0, 0), |w| (w.cap_leaf, w.cap_sky))
    }

    /// Read `n` f32s out of a device buffer.
    pub fn read_f32(&self, hg: &WgpuHeadless, b: &wgpu::Buffer, n: usize) -> Result<Vec<f32>, String> {
        let v = hg.read_buffer(b, n * 4)?;
        Ok(v.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// Flood `accum` and `tbuf` with a sentinel word.
    ///
    /// THE M3d LESSON, and this backend re-learned it the hard way: a planted
    /// ping-pong bug made the ladder emit ZERO terminal records, and the
    /// wavefront-vs-reference image A/B compared CLEAN — because `accum`
    /// still held the reference frame nothing had overwritten. An operation
    /// that never happened compares clean against its own oracle. So the
    /// gate poisons first, and "how many channels still hold the poison" is
    /// an assertion rather than a diagnostic.
    ///
    /// `write_buffer`, not `clear_buffer`: WebGPU can only clear to zeros,
    /// and zero is a legitimate radiance. The write is ordered before the
    /// next submit, so it needs no barrier of its own.
    pub fn poison(&self, hg: &WgpuHeadless, word: u32) {
        let words = (self.rw as usize) * (self.rh as usize);
        let pat: Vec<u8> =
            std::iter::repeat_n(word, words * 3).flat_map(|w| w.to_le_bytes()).collect();
        hg.dev.queue.write_buffer(&self.accum, 0, &pat);
        hg.dev.queue.write_buffer(&self.tbuf, 0, &pat[..words * 4]);
    }

    /// Read `n` u32s out of a device buffer.
    pub fn read_u32(&self, hg: &WgpuHeadless, b: &wgpu::Buffer, n: usize) -> Result<Vec<u32>, String> {
        let v = hg.read_buffer(b, n * 4)?;
        Ok(v.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    pub fn read_counters(&self, hg: &WgpuHeadless) -> Result<Vec<u32>, String> {
        let v = hg.read_buffer(&self.counters, gs::CTR_TOTAL as usize * 4)?;
        Ok(v.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}
