//! The tracer on WebGPU — the recorder the browser will run, running
//! natively so it can be gated.
//!
//! This rung is the REFERENCE KERNEL (`cs_reference` into `accum`,
//! `cs_resolve` into an RGBA16F storage texture) plus the two per-frame
//! cloud caches it reads. That is deliberately the same thing the Vulkan
//! port landed first, and for the reason `vk/tracer.rs`'s header gives: the
//! reference kernel is the smallest thing that can be WRONG in an
//! interesting way. Every kernel above it reads the same streams through the
//! same layout and shades through the same `shade.hlsli`, so a stream bound
//! at the wrong slot, a skewed material stride, or a cbuffer that packs
//! differently shows up HERE as a picture that disagrees with the CPU —
//! before a quadtree is in the way to confuse the attribution.
//!
//! FOUR THINGS THIS SPELLS DIFFERENTLY FROM BOTH NATIVE BACKENDS.
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

use crate::gfx::frame::{CB_STRIDE, FrameCb, FrameParams, GBUF_EXT_STRIDE, GBUF_STRIDE};
use crate::gfx::shaders as gs;
use crate::scene::Scene;
use crate::spirv::{Reg, Spirv, binding_of};
use crate::wgsl::BindKind;

use super::device::Wgpu;
use super::headless::WgpuHeadless;
use super::layout::{self, Unit};
use super::scene::WgpuScene;

/// How many push rows one frame can need. The reference path uses ONE; the
/// ladder (Stage C2b) writes two per level over at most 11 levels plus the
/// terminal fills, so this is that worst case with headroom. Sized here
/// rather than grown on demand because every row must be written before the
/// submit — a ring that had to grow mid-recording would mean a second
/// submit, which is the one thing the design does not do.
const PUSH_ROWS: u64 = 64;

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
/// The hemisphere's two parities are Stage C2c and are absent rather than
/// stubbed — a variant nothing binds would be a bind group nothing proves.
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
}

/// Every variant, in `Variant as usize` order — the bind-group table's index
/// space, stated once so the builder and the dispatcher cannot disagree about
/// how long it is.
const VARIANTS: [Variant; 3] = [Variant::Terminal, Variant::LadderA, Variant::LadderB];

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

/// `cs_prep`'s "do not zero any counter" sentinel, and the two args slots the
/// terminal fills use — lockstep with `gpu/trace.rs`'s consts (which are
/// `#[cfg(windows)]`-only) and `vk/tracer.rs`'s copies.
const NO_RESET: u32 = 0xffff_ffff;
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
    /// The BINARY BVH and `bvh.tri_idx`: what `rt_sw.hlsli` descends and the
    /// triangle ids its leaves hold. Under `--sw-rays` — which the browser
    /// corpus REQUIRES — these are the whole intersector.
    binary: wgpu::Buffer,
    sw_tri: wgpu::Buffer,
    wave: Option<Wave>,
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
        let units: [(&str, &str, &str); 12] = [
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

        let ub = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let frame_cb = buf("frame_cb", CB_STRIDE as u64, ub);
        // The ring stride is the DEVICE's alignment, read back from what was
        // granted rather than assumed to be the 256-byte default — a device
        // that grants a coarser one would silently misalign every row.
        let push_stride = u64::from(dev.limits.min_uniform_buffer_offset_alignment).max(16);
        let push = buf("push ring", push_stride * PUSH_ROWS, ub);

        let binary = super::scene::stream(dev, "bvh", &bvh.nodes, crate::gfx::scene::gpu_bvh_node, sb);
        let sw_tri = super::scene::stream(dev, "tri_idx", &bvh.tri_idx, |t| *t, sb);
        bytes.set(bytes.get() + binary.size() + sw_tri.size());

        // ---- the ladder ----
        //
        // The queue caps come from `caps_for`, which is also what the DEVICE
        // ASK was sized from before this device existed — one derivation, two
        // callers (see its doc comment).
        let ft = if srcs.ftree_on { Some(crate::ftree::FTree::build(bvh)) } else { None };
        let caps = caps_for(rw, rh, gs::sw_rays_leaf() && srcs.ftree_on);
        let dd = caps.depth;
        if dd > 11 {
            return Err(format!("{rw}x{rh} needs quadtree depth {dd} > 11 indirect-arg slots"));
        }
        let (cap_tile, cap_leaf, cap_cut) = (caps.tile, caps.leaf, caps.cut);
        let mut cb_base = FrameCb::base(scene, rw, rh);
        // The kernels read every one of these off the cbuffer, so a buffer
        // sized here and a cap written there must be the same number — hence
        // one place computing both. The hemi three stay 0: that tier is C2c,
        // and a cap for a queue nothing allocates would be a lie the shader
        // would believe.
        cb_base.set_caps(cap_tile as u32, cap_leaf as u32, cap_leaf as u32, cap_cut as u32, 0, 0, 0);

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
            binary,
            sw_tri,
            wave: Some(wave),
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
        self.run_steps(hg, &steps)
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
    /// NO HEMISPHERE TIER (Stage C2c): with fb off the leaf and sky passes
    /// splat straight into `accum` through `queues.hlsli::accum_splat`, so
    /// compose would be a buffer-to-buffer copy of a full screen — which is
    /// D3D12's and Vulkan's rule verbatim, not an omission. A frame that ASKS
    /// for a bounce tier is refused rather than quietly rendered without one.
    pub fn render_wavefront(
        &self,
        hg: &WgpuHeadless,
        p: &FrameParams,
        clear_sentinel: bool,
    ) -> Result<(), String> {
        let w = self.wave.as_ref().ok_or("this tracer has no wavefront ladder")?;
        let fb_mode = crate::gfx::frame::fb_mode_of(&p.q);
        if fb_mode != 0 {
            return Err(format!(
                "fb_mode {fb_mode}: the WebGPU tracer has no hemisphere tier yet (Stage C2c) — \
                 rendering without one would silently drop the ambient term"
            ));
        }
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
        self.plan_terminal_fills(&mut steps);
        self.run_steps(hg, &steps)
    }

    /// The leaf and sky fills, appended to a step program. Shared so the
    /// replay path (Stage C2c) records the SAME code rather than a second
    /// block that looks like it.
    fn plan_terminal_fills(&self, steps: &mut Vec<Step>) {
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
        steps.push(Step::Indirect {
            pass: P_LEAF,
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
    fn run_steps(&self, hg: &WgpuHeadless, steps: &[Step]) -> Result<(), String> {
        let dev = &hg.dev;
        if steps.len() as u64 > PUSH_ROWS {
            return Err(format!(
                "{} dispatches in one frame exceeds the {PUSH_ROWS}-row push ring",
                steps.len()
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
            // something on the counters no kernel resets.
            enc.clear_buffer(&self.counters, 0, None);
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
