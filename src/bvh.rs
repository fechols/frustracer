use crate::scene::Scene;
use glam::Vec3A;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

/// A/B lever (`--no-cut-rays`): when off, cut-SEEDED rays traverse from the root
/// instead of from the tile's node cut. The inherited `tmin` (the tile's proven
/// `t_start`) is untouched — it is a scalar, not a node reference — so this
/// isolates exactly what the CUT itself is worth to the ray path, separately
/// from what the inherited distance bound is worth.
///
/// That is the question that decides whether the frustum structure and the ray
/// BVH can be *separate trees*: a cut built over one tree cannot index another.
/// On the GPU the cut already seeds nothing (every ray is a driver DXR
/// RayQuery), so the answer there is already zero; this measures the CPU.
///
/// Semantics are identical either way — the root covers every node any cut can
/// contain, and hemi's own verify oracle already uses this exact root fallback
/// as the reference for its `cut_miss` gate. Only node visits differ.
pub static CUT_SEED_RAYS: AtomicBool = AtomicBool::new(true);

#[inline]
fn cut_seed_rays() -> bool {
    CUT_SEED_RAYS.load(Ordering::Relaxed)
}

/// Per-consumer companion to `CUT_SEED_RAYS` (`--cut-hemi` re-enables), because
/// the two consumers disagree and the global lever conflates them.
///
/// The PRIMARY path's cuts are short (mean ~18) and seeding from them is worth
/// ~10% on procedural scenes. The HEMI path's cuts sit pinned at the `HEMI_CUT`
/// capacity of 64 (hence its enormous `cut_overflows`), and seeding a bounce ray
/// from 64 scattered coarse roots measured 3-10% SLOWER than one coherent
/// descent from the root — on BOTH the historical tree and the M2 (3-axis,
/// c_trav=3) tree, so it is the scatter, not the array size. The same economics
/// are already recorded in `hemi.rs` ("an occlusion ray is ~10 node visits; a
/// bound query on a dense cut is more"). DEFAULT OFF since that re-measure.
///
/// Read by hemi.rs at its leaf-ray sites; the cut still drives the bound
/// QUERIES either way, and `--check`'s probe gates force this ON so the
/// cut-miss gate keeps exercising the cut machinery. Sound for the same reason
/// as `CUT_SEED_RAYS`: the root covers every node any cut can hold.
pub static CUT_SEED_HEMI: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn cut_seed_hemi() -> bool {
    CUT_SEED_HEMI.load(Ordering::Relaxed)
}

/// Relief (heightfield) rendering: surfaces whose normal map carries an
/// alpha-channel heightfield (`Material::height_amp` > 0) are ray-marched at
/// the intersector choke point — the hit either moves FARTHER along the ray
/// (inward-only displacement; front-side entries) or is REJECTED outright
/// when the ray exits the prism without touching the field (the alpha-cutout
/// monotonicity precedent — silhouettes come out right because the miss
/// really continues traversal).
///
/// `HEIGHT_ARMED` is the SESSION lever (`--heightfield` arms it): off means
/// no swept AABBs and no march anywhere — structurally the pre-relief
/// renderer (part of the scene-cache lever word, since the sweep changes the
/// stored tree). `HEIGHT_ON` is the live V-key toggle: a pure
/// shading+visibility switch that needs NO rebuild — the swept boxes stay
/// conservative for both modes (they only ever contain the flat triangle),
/// so every frustum claim, temporal entry, and hemi bound remains sound
/// with the toggle in either state. The DEFAULT is DISARMED: the sweep's
/// edge-extension pad (`grow_height_sweep` — ±`HEIGHT_EDGE_EXTEND`·depth on
/// every axis) measurably wrecks BVH quality on scenes where EVERY triangle
/// carries height and triangles are only a few texels wide (DamagedHelmet:
/// 596 → 146 ms/frame close-up, 4×, with relief OFF — the armed-but-off
/// tree pays the whole price for a toggle that may never fire), so an armed
/// session is opt-in: `--heightfield` arms AND starts relief ON, and V then
/// toggles relief ↔ normal-mapping live within the armed session (armed
/// stays true across the toggle, no rebuild). Unarmed sessions get the
/// pre-relief renderer bit-exactly and V prints a note instead.
static HEIGHT_ARMED: AtomicBool = AtomicBool::new(false);
static HEIGHT_ON: AtomicBool = AtomicBool::new(false);

pub fn set_height_armed(on: bool) {
    HEIGHT_ARMED.store(on, Ordering::Relaxed);
}

pub fn height_armed() -> bool {
    HEIGHT_ARMED.load(Ordering::Relaxed)
}

pub fn set_height_on(on: bool) {
    HEIGHT_ON.store(on, Ordering::Relaxed);
}

pub fn height_on() -> bool {
    HEIGHT_ON.load(Ordering::Relaxed) && HEIGHT_ARMED.load(Ordering::Relaxed)
}

/// Coarse linear steps across the ray∩prism interval, then bisections + one
/// secant inside the bracketing pair. Fixed counts (not footprint-scaled):
/// the intersector scope has no cone for shadow/AO rays, and constant counts
/// keep every gate deterministic. Grazing rays undersampling texel-thin
/// spires is the documented known-accept. Mirrored in trace_common.hlsli.
pub(crate) const HEIGHT_COARSE: u32 = 16;
pub(crate) const HEIGHT_REFINE: u32 = 5;

/// How far the march may continue PAST the footprint's exit edge, in units
/// of `height_amp` texels of uv travel. This is the edge-crack fix: a ray
/// descending into a recess near a shared edge drifts laterally into the
/// NEIGHBOR's prism, and the neighbor never surfaces as a candidate — the
/// two triangles are coplanar, so the ray's one plane crossing lies inside
/// THIS triangle, and neither möller-trumbore nor the RT hardware ever
/// tests the neighbor. On real meshes (triangle fans) every edge leaked a
/// dark band about one relief depth wide. Continuing the march with the
/// affine uv extension wrap-samples the SAME chart — which for a continuous
/// atlas IS the neighbor's field — so the crack fills with the surface the
/// neighbor would have produced; the hit reports with bary clamped to the
/// edge. Bounded (crack width ~ amp texels, so 4× covers non-grazing rays)
/// so true silhouettes keep carving beyond a texel-scale fringe; the swept
/// AABBs pad every axis by `EXTEND · depth` and the march CLAMPS its per-ray
/// budget to that same world size, which is the containment contract
/// (extended hits stay inside claimed-occupied boxes): the per-ray budget is
/// derived from the DIRECTIONAL texel rate while the pad rides the
/// geometric-mean texel size, so without the clamp an anisotropic chart's
/// sparse UV axis could out-travel the pad. Chart seams get a wrong-field
/// fringe instead of a crack — accepted.
pub(crate) const HEIGHT_EDGE_EXTEND: f32 = 4.0;

#[derive(Clone, Copy)]
pub struct Aabb {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: Vec3A::INFINITY,
        max: Vec3A::NEG_INFINITY,
    };

    pub(crate) fn grow(&mut self, p: Vec3A) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub(crate) fn grow_aabb(&mut self, b: &Aabb) {
        self.min = self.min.min(b.min);
        self.max = self.max.max(b.max);
    }

    pub(crate) fn area(&self) -> f32 {
        let e = (self.max - self.min).max(Vec3A::ZERO);
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }
}

/// count > 0: leaf over tri_idx[left_first .. left_first+count].
/// count == 0: internal, children at left_first and left_first + 1.
///
/// GATEWAY (foliage sway, SAH gateway mode): `count & GATEWAY_BIT` marks a
/// sway cell's TRUTHFUL FAT LEAF — `left_first .. left_first+leaf_count()`
/// is the cell's whole contiguous tri range, its aabb is the ONLY box swept
/// by the motion bound, and the cell's rest-space-TIGHT subtree sits
/// implicitly at `gateway_index + 1`. Bound/cut consumers read it as a leaf
/// (box distance, emit-don't-descend — conservative for every pose because
/// the box is swept); ONLY the ray loops descend, through a nested call
/// with the ray origin shifted into cell-rest space. `leaf_count() == 0` is
/// the E FILLER (EMPTY box, no subtree — the cherry expansion's second-slot
/// spacer): leaf-shaped everywhere, skipped by the ray arm, a guaranteed
/// slab miss. Every consumer that reads `count` AS A RANGE must go through
/// `leaf_count()` — a raw read of a gateway's count is a 2^31-sized range.
pub const GATEWAY_BIT: u32 = 1 << 31;

#[derive(Clone, Copy)]
pub struct BvhNode {
    pub aabb: Aabb,
    pub left_first: u32,
    pub count: u32,
}

impl BvhNode {
    #[inline(always)]
    pub fn is_gateway(&self) -> bool {
        self.count & GATEWAY_BIT != 0
    }
    /// The triangle count with the gateway bit masked off — the ONLY correct
    /// way to read `count` as a range length.
    #[inline(always)]
    pub fn leaf_count(&self) -> u32 {
        self.count & !GATEWAY_BIT
    }
}

pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    pub tri_idx: Vec<u32>,
    /// Lazily built wide tree for frustum/hemisphere bound queries.
    ///
    /// This cache belongs to the binary tree whose node ids it mirrors. A
    /// scene rebuild therefore drops both structures together; keeping it on
    /// `Bvh` prevents cuts or slot-to-node mappings from surviving into a
    /// replacement hierarchy.
    pub(crate) ftree: OnceLock<crate::ftree::FTree>,
}

pub struct Ray {
    pub o: Vec3A,
    pub d: Vec3A,
    pub inv_d: Vec3A,
}

impl Ray {
    pub fn new(o: Vec3A, d: Vec3A) -> Self {
        Ray { o, d, inv_d: d.recip() }
    }
}

/// Map a world-space ray into a sway cell's REST space — the exact inverse
/// of the cell's affine pose `M(p) = p + u·(a + b·p.y)` (foliage v0.4).
/// `u.y ≡ 0` makes M unipotent (det = 1), so the inverse is closed-form:
/// `M⁻¹(q) = q − u·(a + b·q.y)` (exact — q.y is untouched, so the weight
/// the forward map used is recomputable verbatim). The rest-space direction
/// is DELIBERATELY left unnormalized: `o_r + t·d_r = M⁻¹(o + t·d)`, so the
/// world-space hit parameter t is preserved and `o + t·d` lands on the
/// DISPLACED surface — distance == t keeps holding where the claims live
/// (world space). `inv_d` is recomputed only when the direction actually
/// shears (`b·d.y ≠ 0` — above-the-band cells have b = 0 and horizontal
/// rays d.y = 0, both keep the v0.3 translation cost); one `recip` per
/// gateway entry otherwise, amortized over the whole subtree descent.
#[inline]
pub(crate) fn shear_ray(ray: &Ray, s: &crate::foliage::CellShear) -> Ray {
    // u.y == 0 (the wind_with contract), so `- u * w` leaves o.y/d.y alone.
    let o = ray.o - s.u * (s.a + s.b * ray.o.y);
    let bd = s.b * ray.d.y;
    if bd == 0.0 {
        Ray { o, d: ray.d, inv_d: ray.inv_d }
    } else {
        let d = ray.d - s.u * bd;
        Ray { o, d, inv_d: d.recip() }
    }
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub t: f32,
    pub tri: u32,
    pub u: f32,
    pub v: f32,
}

const BINS: usize = 12;
const MAX_LEAF: usize = 8;

/// SAH traversal cost, as a ratio to the intersection cost (`C_isect` is fixed
/// at 1 — only the ratio is meaningful). Settable once, before the build, by
/// `--bvh-ctrav` / `--bvh-maxleaf`; the build stays deterministic for any fixed
/// setting, which is what `Bvh::identical` requires.
///
/// **This term is what lets the leaf test fire at all.** The SAH comparison is
///
/// ```text
///   split = C_trav*A_P + C_isect*(A_L*N_L + A_R*N_R)      vs
///   leaf  =              C_isect*(A_P*N)
/// ```
///
/// (both sides multiplied through by A_P, cancelling the 1/A_P normalization).
/// With `C_trav = 0` — which is what the original code computed — the split cost
/// is *unconditionally* <= the leaf cost, because `A_L, A_R <= A_P` and
/// `N_L + N_R = N`. Splitting was charged nothing, so the leaf test could never
/// win, `MAX_LEAF` was dead on the main path, and every subtree recursed to the
/// hard `count <= 2` floor. Measured result: ~1.2 nodes per triangle on both the
/// default scene (90,201 nodes / 79,741 tris) and San Miguel low-poly (6,842,553
/// / 5,617,453) — roughly 4x the nodes a properly terminated tree builds, and a
/// 328 MB node array on San Miguel where ~80 MB would do.
static C_TRAV_BITS: AtomicU32 = AtomicU32::new(0x4040_0000); // 3.0f32
static MAX_LEAF_N: AtomicUsize = AtomicUsize::new(MAX_LEAF);

/// How many axes the binned SAH searches (`--bvh-axes`). 1 = the historical
/// build (widest centroid axis only); 3 = all axes, global best. Kept as an A/B
/// knob rather than hardcoded so the axis change and the `C_trav` change can be
/// attributed separately — they landed together and 3-axis alone looked like the
/// larger effect on San Miguel — and it was: 3-axis is a -33% ray-node win,
/// while `C_trav` is speed-neutral and buys memory instead.
static SPLIT_AXES: AtomicUsize = AtomicUsize::new(3);

/// The M7 bake-off lever (`--bvh-builder`): which algorithm builds the tree.
/// All builders produce the same `Bvh` (every consumer, gate, and the .fcache
/// work unchanged — the id is in `build_key`), and all are byte-deterministic
/// (`Bvh::identical` gates them like the SAH build). Score on the measured
/// counters, never the SAH readout — see the module doc's anti-correlation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Builder {
    /// The default: binned SAH (3-axis, C_trav) — the M2 build.
    Sah = 0,
    /// Morton-order LBVH: sort by 63-bit Morton code, split at the highest
    /// differing bit, M2 cost-model leaf test. The bake-off CONTROL.
    Lbvh = 1,
    /// PLOC-style agglomerative: bottom-up mutual-nearest-neighbor merging
    /// under d(A,B) = SA(A ∪ B) — the SOFM instinct with the correct metric.
    Ploc = 2,
    /// Batch SOM on a 3D lattice (fixed seed, fixed epochs): the converged
    /// lattice is a density-adaptive warped grid, i.e. a LEARNED space-
    /// filling curve — BMU lattice Morton replaces raw Morton in the LBVH
    /// path, isolating exactly that one variable vs the control.
    Som = 3,
}

static BUILDER: AtomicUsize = AtomicUsize::new(0);

pub fn set_builder(name: &str) -> Option<Builder> {
    let b = match name {
        "sah" => Builder::Sah,
        "lbvh" => Builder::Lbvh,
        "ploc" => Builder::Ploc,
        "som" => Builder::Som,
        _ => return None,
    };
    BUILDER.store(b as usize, Ordering::Relaxed);
    Some(b)
}

#[inline]
pub fn builder() -> Builder {
    match BUILDER.load(Ordering::Relaxed) {
        1 => Builder::Lbvh,
        2 => Builder::Ploc,
        3 => Builder::Som,
        _ => Builder::Sah,
    }
}

/// Reference C_trav for the `quality()` SAH readout, so trees built at DIFFERENT
/// C_trav are still comparable on one scale. (Scoring each tree at its own build
/// C_trav makes the number rise with C_trav by construction and compare nothing.)
pub const SAH_REF_C_TRAV: f32 = 1.0;

pub fn set_c_trav(v: f32) {
    C_TRAV_BITS.store(v.to_bits(), Ordering::Relaxed);
}

pub fn set_max_leaf(v: usize) {
    MAX_LEAF_N.store(v, Ordering::Relaxed);
}

pub fn set_split_axes(v: usize) {
    SPLIT_AXES.store(v.clamp(1, 3), Ordering::Relaxed);
}

#[inline]
pub(crate) fn c_trav() -> f32 {
    f32::from_bits(C_TRAV_BITS.load(Ordering::Relaxed))
}

#[inline]
pub(crate) fn max_leaf() -> usize {
    MAX_LEAF_N.load(Ordering::Relaxed)
}

#[inline]
fn split_axes() -> usize {
    SPLIT_AXES.load(Ordering::Relaxed)
}

/// Identity of the build PARAMETERS, for the scene-cache key. The sidecar stores
/// a BUILT tree, so a cache written at one (c_trav, max_leaf) must never be served
/// to a session asking for another — otherwise a parameter sweep silently
/// benchmarks the same cached tree every time. Distinct from `CACHE_VERSION`,
/// which versions the on-disk FORMAT.
pub fn build_key() -> u64 {
    ((c_trav().to_bits() as u64) << 32)
        | ((builder() as u64) << 16)
        | ((max_leaf() as u64) << 8)
        | split_axes() as u64
}
/// Traversal stack capacity. Both traversal loops push at most one entry per
/// level, so the required stack is exactly the tree depth; `Bvh::build`
/// hard-asserts `max_depth() <= TRAV_STACK` once, which keeps the hot loops
/// branch-free (a silent overflow in release would be an out-of-bounds write,
/// and dropping the far child instead would be UNSOUND — a missed subtree
/// breaks the frustum/temporal lower-bound contracts). SAH depth is ~45-50 at
/// 10M tris and grows ~2 levels per 4x tris; 96 covers 100M+ with margin.
pub(crate) const TRAV_STACK: usize = 96;

/// Build-quality readout — the A/B currency for any builder change. `leaf_hist[i]`
/// counts leaves holding exactly `i` triangles (the last bucket is saturating);
/// bucket 1-2 dominating is the signature of a missing traversal cost (see
/// `C_TRAV_BITS`).
#[derive(Default, Clone)]
pub struct BvhQuality {
    pub nodes: usize,
    pub internal: usize,
    pub leaves: usize,
    /// Sway gateway nodes (incl. E fillers) — see `quality`'s gateway arm.
    pub gateways: usize,
    pub tri_refs: usize,
    pub mean_leaf: f32,
    pub max_depth: usize,
    pub sah: f32,
    /// SUM_internal A(n)/A(root) — SAH's predicted internal-node visits per
    /// uniform random ray. Compare against the measured `ray_nodes` counter.
    pub node_term: f32,
    /// SUM_leaf N(n)*A(n)/A(root) — SAH's predicted triangle tests per ray.
    pub tri_term: f32,
    pub bytes: usize,
    pub leaf_hist: [usize; 17],
}

impl BvhQuality {
    pub fn line(&self, tris: usize) -> String {
        let h: Vec<String> = (1..=8).map(|i| format!("{}", self.leaf_hist[i])).collect();
        let gw = if self.gateways > 0 { format!(" | gateways {}", self.gateways) } else { String::new() };
        format!(
            "nodes {} ({:.2}/tri, {:.0} MB) | leaves {} mean {:.2} tris | depth {} | SAH@1 {:.1} (nodes {:.2} + tris {:.2}) | leaf-hist[1..8] {}{gw}",
            self.nodes,
            self.nodes as f32 / tris.max(1) as f32,
            self.bytes as f32 / (1024.0 * 1024.0),
            self.leaves,
            self.mean_leaf,
            self.max_depth,
            self.sah,
            self.node_term,
            self.tri_term,
            h.join(",")
        )
    }
}

/// Scratch capacity for `intersect_multi`'s front-to-back root sort. Matches the
/// cut capacity every `refine_cut` caller uses (`frustum::MAX_CUT`, `hemi::HEMI_CUT`,
/// `shaft::SHAFT_CUT` — all 64), so the sorted path takes every root in practice;
/// a longer cut falls through to the unsorted tail loop rather than dropping a root.
const ROOT_SORT: usize = 64;

/// Subtree handed to phase 2 of the build: `nodes[node_i]` still holds its
/// (first, count) leaf-form range over the ITEM array until the stitch
/// replaces it. `tri_first`/`tri_count` locate the range's slice of the
/// FINAL tri buffer — equal to (first, count) when no gateway ctx is live,
/// since every item then has width 1 and the item array IS the tri buffer.
struct Pending {
    node_i: u32,
    first: u32,
    count: u32,
    tri_first: u32,
    tri_count: u32,
}

/// Build-time context for GATEWAY mode (`foliage::gateway_mode()` — sway on
/// the SAH builder): each sway cell rides the top-level build as ONE
/// pseudo-prim (id `n + c`, indexing the extended `tri_aabb`/`centroids`
/// tails with the cell's SWEPT box) and is materialized as a gateway + a
/// rest-space-TIGHT subtree at child emission (`emit_gateway`). This is the
/// design doc's "pad at cell granularity, never per-triangle": profiling
/// measured the per-tri sweep at ~80% of the CPU sway bill (canopy triangle
/// tests +107%), and here the ONLY swept boxes are the ≤ MAX_CELLS
/// gateways'. Everything is a pure function of the partition + levers (CSR
/// in ascending tri order, sequential unions, a fixed second-slot rule), so
/// the build stays byte-deterministic — `Bvh::identical` gates it.
struct GatewayCtx {
    /// CSR cell -> tris, ascending tri id within a cell — the canonical
    /// order the range is EMITTED in (the cell subtree's own SAH build then
    /// permutes it in place, so a built gateway's range is a permutation of
    /// the cell), and the order both rest-box unions accumulate in (the
    /// pseudo-prim's entry and the subtree root's bounds loop both run over
    /// the ascending sequence — the root bounds are computed BEFORE any
    /// partition swap — so the gateway box debug-assert can demand
    /// bit-equality).
    cell_off: Vec<u32>,
    cell_tris: Vec<u32>,
    /// Per-cell displacement pad (`foliage::displacement_bound_with` at
    /// `sweep_mult()` — exactly `foliage::sway_pad`'s value for any tri of
    /// the cell).
    pad: Vec<f32>,
    /// Real-triangle count; pseudo ids are `n .. n + cells`.
    n: u32,
}

impl GatewayCtx {
    #[inline]
    fn is_pseudo(&self, id: u32) -> bool {
        id >= self.n
    }
    #[inline]
    fn cell(&self, id: u32) -> usize {
        (id - self.n) as usize
    }
    #[inline]
    fn cell_range(&self, c: usize) -> &[u32] {
        &self.cell_tris[self.cell_off[c] as usize..self.cell_off[c + 1] as usize]
    }
    /// How many FINAL tri_idx slots this item occupies.
    #[inline]
    fn width(&self, id: u32) -> u32 {
        if self.is_pseudo(id) {
            let c = self.cell(id);
            self.cell_off[c + 1] - self.cell_off[c]
        } else {
            1
        }
    }
}

/// The mutable half of gateway mode threaded through `subdivide_range`:
/// the ctx plus the tri buffer leaf/gateway emission writes into (the item
/// array being subdivided is a PERMUTATION WORK ARRAY in this mode, never
/// the final `tri_idx`).
struct GwPhase<'a> {
    ctx: &'a GatewayCtx,
    tribuf: &'a mut [u32],
}

/// Phase-1 stop size — a pure function of n ONLY (never the thread count):
/// the node order must be byte-identical across machines. ~256 subtrees keeps
/// every core busy without ballooning the sequential top phase.
fn par_threshold(n: usize) -> usize {
    (n / 256).max(2048)
}

impl Bvh {
    /// Assemble a binary tree and its empty per-tree auxiliary cache.
    ///
    /// All builders and cache loaders pass through here so a deserialized or
    /// alternative-builder BVH has the same lifetime coupling as the SAH
    /// builder.
    pub(crate) fn from_parts(nodes: Vec<BvhNode>, tri_idx: Vec<u32>) -> Bvh {
        Bvh { nodes, tri_idx, ftree: OnceLock::new() }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tri_idx.is_empty()
    }

    /// Deterministic two-phase parallel build. Phase 1 splits sequentially
    /// until ranges fall under `par_threshold` (node allocation order is
    /// sequential ⇒ the top of the tree is pinned); phase 2 builds each
    /// pending range — disjoint, ascending slices of `tri_idx` after the
    /// in-place partitions — into a local arena in parallel (collect
    /// preserves pending order and per-subtree fp ops run sequentially ⇒ the
    /// output is byte-identical across runs AND thread counts, which `--check`
    /// gates on loaded scenes); a cheap sequential stitch splices the arenas
    /// in. Phase 1 never creates real leaves (threshold > MAX_LEAF and a
    /// too-big node always splits, via SAH or the median fallback), so the
    /// pending ranges exactly partition 0..n.
    /// Build under the session's `--bvh-builder` (default: the SAH build).
    /// The depth contract is asserted here for EVERY builder — never "fix"
    /// an overflow by dropping stack entries.
    pub fn build(scene: &Scene) -> Bvh {
        let bvh = match builder() {
            Builder::Sah => Self::build_sah(scene),
            b => crate::builders::build_alt(scene, b),
        };
        let depth = bvh.max_depth();
        assert!(
            depth <= TRAV_STACK,
            "BVH depth {depth} exceeds the {TRAV_STACK}-entry traversal stack"
        );
        bvh
    }

    fn build_sah(scene: &Scene) -> Bvh {
        let n = scene.indices.len();
        let (mut tri_aabb, mut centroids): (Vec<Aabb>, Vec<Vec3A>) = scene
            .indices
            .par_iter()
            .enumerate()
            .map(|(i, tri)| {
                let (a, b, c) = (
                    scene.positions[tri[0] as usize],
                    scene.positions[tri[1] as usize],
                    scene.positions[tri[2] as usize],
                );
                let mut bb = Aabb::EMPTY;
                bb.grow(a);
                bb.grow(b);
                bb.grow(c);
                grow_height_sweep(scene, i as u32, a, b, c, &mut bb);
                // NO grow_sway_sweep here (unlike builders.rs): in gateway
                // mode the pad lives on the ≤ MAX_CELLS gateway boxes and
                // per-tri boxes stay rest-space TIGHT — the measured ~80% of
                // the CPU sway bill was exactly this per-tri pad. Without
                // sway the call was a no-op, so its removal is bit-neutral.
                (bb, (a + b + c) / 3.0)
            })
            .unzip();

        // GATEWAY mode: derive the build ctx from the ONE shared partition.
        let gw_ctx: Option<GatewayCtx> = match &scene.sway {
            Some(sw) if crate::foliage::gateway_mode() && !sw.cells.is_empty() => {
                let k = sw.cells.len();
                // Counting sort: cell -> tris, ascending tri id per cell.
                let mut off = vec![0u32; k + 1];
                for &c in sw.tri_cell.iter() {
                    if c != crate::foliage::STATIC_CELL {
                        off[c as usize + 1] += 1;
                    }
                }
                for i in 0..k {
                    off[i + 1] += off[i];
                }
                let total = off[k] as usize;
                if total == 0 {
                    None
                } else {
                    let mut cur = off.clone();
                    let mut cell_tris = vec![0u32; total];
                    for (t, &c) in sw.tri_cell.iter().enumerate() {
                        if c != crate::foliage::STATIC_CELL {
                            cell_tris[cur[c as usize] as usize] = t as u32;
                            cur[c as usize] += 1;
                        }
                    }
                    let pad: Vec<f32> = sw
                        .cells
                        .iter()
                        .map(|cl| {
                            crate::foliage::displacement_bound_with(
                                cl.w_max,
                                cl.scale,
                                crate::foliage::sweep_mult(),
                            )
                        })
                        .collect();
                    Some(GatewayCtx { cell_off: off, cell_tris, pad, n: n as u32 })
                }
            }
            _ => None,
        };

        // Pseudo-prim tails: one entry per cell — the SWEPT box (the rest
        // union in CSR order ± the pad; `emit_gateway` re-derives the same
        // box from the subtree root and debug-asserts bit-equality) and its
        // rest-box center as the binning centroid.
        if let Some(ctx) = &gw_ctx {
            let k = ctx.cell_off.len() - 1;
            tri_aabb.reserve_exact(k);
            centroids.reserve_exact(k);
            for c in 0..k {
                let mut bb = Aabb::EMPTY;
                for &t in ctx.cell_range(c) {
                    bb.grow_aabb(&tri_aabb[t as usize]);
                }
                centroids.push(0.5 * (bb.min + bb.max));
                // x/z only: the v0.4 shear has NO y component (u.y ≡ 0), so
                // the swept box is exact in y for free.
                let p = Vec3A::new(ctx.pad[c], 0.0, ctx.pad[c]);
                tri_aabb.push(Aabb { min: bb.min - p, max: bb.max + p });
            }
        }

        // The item array phase 1/2 subdivide: without ctx it IS the final
        // tri buffer (identity permutation, today's path bit-identically);
        // with ctx it is [static tris ascending | pseudo ids] and the final
        // tri buffer is written separately at leaf/gateway emission.
        let (mut items, mut tribuf): (Vec<u32>, Vec<u32>) = match (&gw_ctx, &scene.sway) {
            (Some(ctx), Some(sw)) => {
                let k = ctx.cell_off.len() - 1;
                let mut v = Vec::with_capacity(n - ctx.cell_tris.len() + k);
                for t in 0..n as u32 {
                    if sw.tri_cell[t as usize] == crate::foliage::STATIC_CELL {
                        v.push(t);
                    }
                }
                for c in 0..k as u32 {
                    v.push(n as u32 + c);
                }
                (v, vec![u32::MAX; n])
            }
            _ => ((0..n as u32).collect(), Vec::new()),
        };

        // No 2n up-front node reserve (8 GB at 100M tris): phase 1 grows
        // organically (small) and the stitch reserves the exact total.
        let mut nodes = Vec::new();
        nodes.push(BvhNode {
            aabb: Aabb::EMPTY,
            left_first: 0,
            count: items.len() as u32,
        });

        if n > 0 {
            let threshold = par_threshold(n);
            let mut pending = Vec::new();
            {
                let mut gphase = gw_ctx
                    .as_ref()
                    .map(|ctx| GwPhase { ctx, tribuf: &mut tribuf });
                subdivide_range(
                    &mut nodes,
                    0,
                    &mut items,
                    &tri_aabb,
                    &centroids,
                    &mut Some((&mut pending, threshold)),
                    &mut gphase,
                    0,
                );
            }

            // Phase-2 slicing: the same disjoint-ascending walk, over the
            // ITEM array and (in gateway mode) the tri buffer too — gaps
            // (phase-1 inline gateway blocks) fall out of the `- consumed`
            // arithmetic exactly like they always did for the item walk.
            let mut slices: Vec<(&mut [u32], u32)> = Vec::with_capacity(pending.len());
            let mut rest: &mut [u32] = &mut items;
            let mut consumed = 0u32;
            for p in &pending {
                let (_, r) = rest.split_at_mut((p.first - consumed) as usize);
                let (s, r2) = r.split_at_mut(p.count as usize);
                slices.push((s, p.first));
                rest = r2;
                consumed = p.first + p.count;
            }
            let arenas: Vec<Vec<BvhNode>> = if let Some(ctx) = &gw_ctx {
                let mut tslices: Vec<(&mut [u32], u32)> = Vec::with_capacity(pending.len());
                let mut trest: &mut [u32] = &mut tribuf;
                let mut tconsumed = 0u32;
                for p in &pending {
                    let (_, r) = trest.split_at_mut((p.tri_first - tconsumed) as usize);
                    let (s, r2) = r.split_at_mut(p.tri_count as usize);
                    tslices.push((s, p.tri_first));
                    trest = r2;
                    tconsumed = p.tri_first + p.tri_count;
                }
                slices
                    .into_par_iter()
                    .zip(tslices)
                    .map(|((islice, _ibase), (tslice, tbase))| {
                        build_subtree(islice, tbase, &tri_aabb, &centroids, Some((ctx, tslice)))
                    })
                    .collect()
            } else {
                slices
                    .into_par_iter()
                    .map(|(slice, base)| build_subtree(slice, base, &tri_aabb, &centroids, None))
                    .collect()
            };

            // Stitch: arena local 0 lands in the pending node's slot and
            // local i>0 at off+i, so every internal link rebases by one
            // uniform +off — child pairs stay adjacent (count==0 ⇒ children
            // at left_first, left_first+1), and a gateway's implicit +1
            // subtree survives because the splice is contiguous and a
            // pending range is never gateway-ROOTED (a singleton pseudo is
            // materialized at child emission, never pended). Leaf AND
            // gateway ranges were lifted to global tri positions inside
            // build_subtree (count > 0 covers both), so the rebase leaves
            // them alone.
            let extra: usize = arenas.iter().map(|a| a.len() - 1).sum();
            nodes.reserve_exact(extra);
            for (p, arena) in pending.iter().zip(&arenas) {
                let off = (nodes.len() - 1) as u32;
                let rebase = |nd: &BvhNode| BvhNode {
                    left_first: if nd.count == 0 { nd.left_first + off } else { nd.left_first },
                    ..*nd
                };
                nodes[p.node_i as usize] = rebase(&arena[0]);
                nodes.extend(arena[1..].iter().map(rebase));
            }
        }

        if gw_ctx.is_some() {
            debug_assert!(
                tribuf.iter().all(|&t| t != u32::MAX),
                "gateway build left tri buffer slots unwritten"
            );
            Bvh::from_parts(nodes, tribuf)
        } else {
            Bvh::from_parts(nodes, items)
        }
    }

    /// Byte-equality of two builds — the determinism gate: `--check` rebuilds
    /// once and asserts this, pinning the two-phase build's "identical across
    /// runs and thread counts" contract. Bit compare (f32::to_bits), so a
    /// -0.0/NaN drift can't hide.
    pub fn identical(&self, other: &Bvh) -> bool {
        self.tri_idx == other.tri_idx
            && self.nodes.len() == other.nodes.len()
            && self.nodes.iter().zip(&other.nodes).all(|(a, b)| {
                a.left_first == b.left_first
                    && a.count == b.count
                    && a.aabb.min.to_array().map(f32::to_bits)
                        == b.aabb.min.to_array().map(f32::to_bits)
                    && a.aabb.max.to_array().map(f32::to_bits)
                        == b.aabb.max.to_array().map(f32::to_bits)
            })
    }

    /// Tree quality, for A/B-ing builders. `sah` is the classic expected-cost
    /// SAH under the same (C_trav, C_isect=1) the builder used:
    ///
    /// ```text
    ///   SAH = C_trav * SUM_internal A(n)/A(root) + SUM_leaf N(n) * A(n)/A(root)
    /// ```
    ///
    /// Note this is the RAY consumer's cost model. It is deliberately not the
    /// frustum bound query's — that one never dereferences a triangle, so its
    /// cost is node count and box tightness, not surface-area-weighted triangle
    /// tests. Do not tune the frustum side against this number.
    pub fn quality(&self, c_trav: f32) -> BvhQuality {
        let mut q = BvhQuality {
            nodes: self.nodes.len(),
            bytes: self.nodes.len() * std::mem::size_of::<BvhNode>()
                + self.tri_idx.len() * 4,
            ..Default::default()
        };
        if self.is_empty() {
            return q;
        }
        let root_area = self.nodes[0].aabb.area().max(1e-20);
        // Split the two SAH terms. `node_term` = SUM_internal A(n)/A(root) is the
        // model's PREDICTED internal-node visits per random ray; it is the number
        // to hold against the measured `ray_nodes` counter when asking whether
        // SAH predicts this renderer at all.
        let mut node_term = 0.0f64;
        let mut tri_term = 0.0f64;
        for n in &self.nodes {
            let p = (n.aabb.area() / root_area) as f64;
            if n.count == 0 {
                q.internal += 1;
                node_term += p;
            } else if n.is_gateway() {
                // A sway gateway is traversal overhead, not a triangle test:
                // a ray pays its (swept) box before the subtree, so its area
                // rides node_term; its span must NOT enter tri_refs/tri_term
                // — the +1 subtree's real leaves already carry those tris,
                // and double-counting would corrupt the builder A/B readout.
                // The E filler (masked count 0) lands here too, carrying a
                // COPY of its sibling gateway's box — so a cherry's box area
                // enters node_term twice, which is honest: a ray really does
                // slab-test both. Counted in `gateways` for the diagnostic
                // line.
                q.gateways += 1;
                node_term += p;
            } else {
                q.leaves += 1;
                q.tri_refs += n.count as usize;
                tri_term += n.count as f64 * p;
                let b = (n.count as usize).min(q.leaf_hist.len() - 1);
                q.leaf_hist[b] += 1;
            }
        }
        q.node_term = node_term as f32;
        q.tri_term = tri_term as f32;
        q.sah = (c_trav as f64 * node_term + tri_term) as f32;
        q.mean_leaf = q.tri_refs as f32 / q.leaves.max(1) as f32;
        q.max_depth = self.max_depth();
        q
    }

    /// Max node depth (root = 1). Iterative — O(nodes), run once at build so
    /// the traversal loops can stay branch-free (see TRAV_STACK).
    pub fn max_depth(&self) -> usize {
        if self.is_empty() || self.nodes.is_empty() {
            return 0;
        }
        let mut max = 0usize;
        let mut stack = vec![(0u32, 1usize)];
        while let Some((i, d)) = stack.pop() {
            max = max.max(d);
            let node = &self.nodes[i as usize];
            if node.count == 0 {
                stack.push((node.left_first, d + 1));
                stack.push((node.left_first + 1, d + 1));
            } else if node.is_gateway() && node.leaf_count() > 0 {
                // Descend the gateway's implicit +1 subtree: the TRAV_STACK
                // assert must cover the WHOLE path (top + cell subtree) —
                // conservative for the nested-call ray arms (which get a
                // fresh stack each) and exactly right for the single-stack
                // HLSL software descent in rt_sw.hlsli.
                stack.push((i + 1, d + 1));
            }
        }
        max
    }

    /// Closest hit in (tmin, tmax). `visits` counts BVH node visits.
    pub fn intersect(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        mut tmax: f32,
        visits: &mut u64,
    ) -> Option<Hit> {
        if self.is_empty() {
            return None;
        }
        let mut best: Option<Hit> = None;
        self.intersect_from(scene, ray, tmin, &mut tmax, 0, &mut best, visits);
        best
    }

    /// Closest hit in (tmin, tmax) with traversal seeded from a tile's node
    /// cut instead of the root — primary rays inside a quadtree leaf tile skip
    /// the top of the tree the tile's ancestors already culled. Secondary rays
    /// and reference rays use `intersect`.
    ///
    /// Roots are traversed FRONT-TO-BACK, by slab entry distance. `intersect_from`
    /// already orders its two children near-first for the same reason: the first
    /// hit sets `tmax`, and every later slab test prunes against it. Walking the
    /// roots in cut-ARRAY order throws that away — a far root can land the first
    /// hit, leaving `tmax` loose so the remaining roots cannot be pruned. Measured
    /// on San Miguel that inverted the cut's whole value (it cost 5.4% instead of
    /// saving); the effect is largest on dense scenes, and alpha cutout amplifies
    /// it because a masked rejection does not shrink `tmax` at all.
    pub fn intersect_multi(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        mut tmax: f32,
        roots: &[u32],
        visits: &mut u64,
    ) -> Option<Hit> {
        if self.is_empty() {
            return None;
        }
        if !cut_seed_rays() {
            return self.intersect(scene, ray, tmin, tmax, visits);
        }
        let mut best: Option<Hit> = None;

        // refine_cut caps every cut at its capacity N (64 for all three callers),
        // so the sorted path takes every root in practice. The tail loop below is
        // the soundness backstop: dropping a root would drop geometry -> false sky.
        let n = roots.len().min(ROOT_SORT);
        let mut ord = [(f32::INFINITY, 0u32); ROOT_SORT];
        for (slot, &r) in ord[..n].iter_mut().zip(&roots[..n]) {
            *slot = (slab_t(&self.nodes[r as usize].aabb, ray, tmin, tmax), r);
        }
        ord[..n].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for &(d, r) in &ord[..n] {
            // Ascending order: once a root's entry is at or beyond the CURRENT
            // tmax, every remaining root is too. (`slab_t` returns +inf on a miss,
            // so misses sort last and end the loop here.)
            if !(d < tmax) {
                break;
            }
            self.intersect_from(scene, ray, tmin, &mut tmax, r, &mut best, visits);
        }
        for &r in &roots[n..] {
            if slab_t(&self.nodes[r as usize].aabb, ray, tmin, tmax).is_finite() {
                self.intersect_from(scene, ray, tmin, &mut tmax, r, &mut best, visits);
            }
        }
        best
    }

    /// A gateway's CURRENT cell pose — `None` (the rest pose) when the scene
    /// carries no sway (a defensive impossibility: gateways only build with
    /// `Scene::sway` attached), the `FR_SWAY_ABL=noshift` cost probe is
    /// armed, or the baked wind is exactly zero (headless paths, dead
    /// cells). The cell id is derived, never stored: the gateway's truthful
    /// range is one cell by construction, so its first tri's `tri_cell`
    /// entry IS the cell.
    #[inline]
    fn gateway_shear(&self, scene: &Scene, node: &BvhNode) -> Option<crate::foliage::CellShear> {
        match &scene.sway {
            Some(sw) if !crate::foliage::sway_abl().noshift => {
                let s =
                    sw.shear(sw.tri_cell[self.tri_idx[node.left_first as usize] as usize]);
                (s.u != Vec3A::ZERO).then_some(s)
            }
            _ => None,
        }
    }

    fn intersect_from(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        tmax: &mut f32,
        start: u32,
        best: &mut Option<Hit>,
        visits: &mut u64,
    ) {
        // The stack carries each deferred far child's ENTRY DISTANCE alongside its
        // index. A node pushed when tmax was loose is often already beaten by the
        // time it is popped — a closer hit has since shrunk tmax — and re-descending
        // it costs its whole subtree. Storing only the index (the original) meant
        // that node's own AABB was never re-tested on pop, so the kill had to be
        // rediscovered one level down, per child, for the entire subtree.
        let mut stack = [(0u32, 0.0f32); TRAV_STACK];
        let mut sp = 0usize;
        let mut node_idx = start;
        loop {
            let node = &self.nodes[node_idx as usize];
            *visits += 1;
            if node.count > 0 {
                if node.is_gateway() {
                    // GATEWAY (foliage sway): descend the cell's rest-space
                    // subtree at +1 with the ray mapped into cell-rest space
                    // — ONE shear per cell entry instead of the old per-test
                    // head (`shear_ray`: the exact unipotent inverse, t
                    // preserved, inv_d recomputed only when b·d.y ≠ 0).
                    // Nested call = fresh stack; gateways never nest, so
                    // recursion depth is 1. Masked count 0 = the E filler:
                    // no subtree, fall straight to the pop tail. tmax/best
                    // merge through the &mut params.
                    if node.leaf_count() > 0 {
                        if tri_probe() {
                            GATEWAY_ENTRIES.fetch_add(1, Ordering::Relaxed);
                        }
                        if let Some(s) = self.gateway_shear(scene, node) {
                            let sray = shear_ray(ray, &s);
                            self.intersect_from(scene, &sray, tmin, tmax, node_idx + 1, best, visits);
                        } else {
                            self.intersect_from(scene, ray, tmin, tmax, node_idx + 1, best, visits);
                        }
                    }
                } else {
                    let first = node.left_first as usize;
                    for &t in &self.tri_idx[first..first + node.count as usize] {
                        if let Some((tt, u, v)) = moller_trumbore(scene, t, ray) {
                            if tt > tmin && tt < *tmax {
                                *tmax = tt;
                                *best = Some(Hit { t: tt, tri: t, u, v });
                            }
                        }
                    }
                }
                // Pop the nearest deferred node that tmax has not already killed.
                loop {
                    if sp == 0 {
                        return;
                    }
                    sp -= 1;
                    let (idx, d) = stack[sp];
                    if d < *tmax {
                        node_idx = idx;
                        break;
                    }
                }
            } else {
                let l = node.left_first;
                let dl = slab_t(&self.nodes[l as usize].aabb, ray, tmin, *tmax);
                let dr = slab_t(&self.nodes[l as usize + 1].aabb, ray, tmin, *tmax);
                let (near, far, dnear, dfar) = if dl <= dr {
                    (l, l + 1, dl, dr)
                } else {
                    (l + 1, l, dr, dl)
                };
                if dnear.is_finite() {
                    node_idx = near;
                    if dfar.is_finite() {
                        // Capacity is guaranteed by build's max_depth assert;
                        // the debug_assert stays as belt-and-braces.
                        debug_assert!(sp < stack.len(), "BVH traversal stack overflow");
                        stack[sp] = (far, dfar);
                        sp += 1;
                    }
                } else {
                    loop {
                        if sp == 0 {
                            return;
                        }
                        sp -= 1;
                        let (idx, d) = stack[sp];
                        if d < *tmax {
                            node_idx = idx;
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Any hit in (tmin, tmax) — early-out occlusion test for shadow/AO rays.
    pub fn occluded(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        tmax: f32,
        visits: &mut u64,
    ) -> bool {
        if self.is_empty() {
            return false;
        }
        if !slab_t(&self.nodes[0].aabb, ray, tmin, tmax).is_finite() {
            return false;
        }
        self.occluded_from(scene, ray, tmin, tmax, 0, visits)
    }

    /// Any hit in (tmin, tmax) with traversal seeded from a node cut — the
    /// occlusion analog of `intersect_multi`, for hemisphere/shaft bounce rays
    /// that own a cut (their OWN apex-relative cut, never a primary tile's).
    pub fn occluded_multi(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        tmax: f32,
        roots: &[u32],
        visits: &mut u64,
    ) -> bool {
        if self.is_empty() {
            return false;
        }
        if !cut_seed_rays() {
            return self.occluded(scene, ray, tmin, tmax, visits);
        }
        for &r in roots {
            if slab_t(&self.nodes[r as usize].aabb, ray, tmin, tmax).is_finite()
                && self.occluded_from(scene, ray, tmin, tmax, r, visits)
            {
                return true;
            }
        }
        false
    }

    fn occluded_from(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        tmax: f32,
        start: u32,
        visits: &mut u64,
    ) -> bool {
        let mut stack = [0u32; TRAV_STACK];
        let mut sp = 0usize;
        let mut node_idx = start;
        loop {
            let node = &self.nodes[node_idx as usize];
            *visits += 1;
            if node.count > 0 {
                if node.is_gateway() {
                    // GATEWAY: one shear per cell entry (see intersect_from).
                    if node.leaf_count() > 0 {
                        if tri_probe() {
                            GATEWAY_ENTRIES.fetch_add(1, Ordering::Relaxed);
                        }
                        let hit = if let Some(s) = self.gateway_shear(scene, node) {
                            let sray = shear_ray(ray, &s);
                            self.occluded_from(scene, &sray, tmin, tmax, node_idx + 1, visits)
                        } else {
                            self.occluded_from(scene, ray, tmin, tmax, node_idx + 1, visits)
                        };
                        if hit {
                            return true;
                        }
                    }
                } else {
                    let first = node.left_first as usize;
                    for &t in &self.tri_idx[first..first + node.count as usize] {
                        if let Some((tt, _, _)) = moller_trumbore(scene, t, ray) {
                            if tt > tmin && tt < tmax {
                                return true;
                            }
                        }
                    }
                }
            } else {
                let l = node.left_first;
                if slab_t(&self.nodes[l as usize].aabb, ray, tmin, tmax).is_finite() {
                    if slab_t(&self.nodes[l as usize + 1].aabb, ray, tmin, tmax).is_finite() {
                        debug_assert!(sp < stack.len(), "BVH traversal stack overflow");
                        stack[sp] = l + 1;
                        sp += 1;
                    }
                    node_idx = l;
                    continue;
                }
                if slab_t(&self.nodes[l as usize + 1].aabb, ray, tmin, tmax).is_finite() {
                    node_idx = l + 1;
                    continue;
                }
            }
            if sp == 0 {
                return false;
            }
            sp -= 1;
            node_idx = stack[sp];
        }
    }

    /// Light-transport occlusion: the RGB throughput of segment (tmin, tmax)
    /// — ONE when clear, ZERO on any opaque hit, and the product of
    /// `Material::shadow_tint` per transmissive interface crossed (the
    /// tinted-shadows feature; `--no-tinted-shadows` kills it). This is the
    /// query every LIGHT ray takes (sun shadow, translucency back ray,
    /// firefly shadow, sampled AO, hemi-AO leaf); `occluded` KEEPS the binary
    /// "any geometry in segment" semantics for geometric oracles (hemi
    /// empty-cell verification, relief self-tests) — a glass-containing cell
    /// must never verify as "empty". With `!scene.any_transmissive` this runs
    /// `occluded` verbatim — bit-identical traversal, the structural
    /// guarantee for procedural/stress scenes and the lever-off A/B.
    pub fn transmittance(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        tmax: f32,
        visits: &mut u64,
    ) -> Vec3A {
        if self.is_empty() {
            return Vec3A::ONE;
        }
        if !scene.any_transmissive {
            return if self.occluded(scene, ray, tmin, tmax, visits) {
                Vec3A::ZERO
            } else {
                Vec3A::ONE
            };
        }
        let mut tp = Vec3A::ONE;
        if slab_t(&self.nodes[0].aabb, ray, tmin, tmax).is_finite()
            && self.transmittance_from(scene, ray, tmin, tmax, 0, &mut tp, visits)
        {
            return Vec3A::ZERO;
        }
        tp
    }

    /// `transmittance` seeded from a node cut — the light-transport analog of
    /// `occluded_multi` (hemisphere bounce rays that own their OWN
    /// apex-relative cut). Cut roots are a frontier of disjoint subtrees, so
    /// the per-root throughputs multiply into the segment's total; any opaque
    /// hit short-circuits to ZERO.
    pub fn transmittance_multi(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        tmax: f32,
        roots: &[u32],
        visits: &mut u64,
    ) -> Vec3A {
        if self.is_empty() {
            return Vec3A::ONE;
        }
        if !scene.any_transmissive {
            return if self.occluded_multi(scene, ray, tmin, tmax, roots, visits) {
                Vec3A::ZERO
            } else {
                Vec3A::ONE
            };
        }
        if !cut_seed_rays() {
            return self.transmittance(scene, ray, tmin, tmax, visits);
        }
        let mut tp = Vec3A::ONE;
        for &r in roots {
            if slab_t(&self.nodes[r as usize].aabb, ray, tmin, tmax).is_finite()
                && self.transmittance_from(scene, ray, tmin, tmax, r, &mut tp, visits)
            {
                return Vec3A::ZERO;
            }
        }
        tp
    }

    /// `occluded_from`'s traversal with the tinted leaf test: a transmissive
    /// in-range hit multiplies `tp` and traversal CONTINUES (the relief march
    /// and alpha cutout inside `moller_trumbore` still apply first — a
    /// cutout-rejected texel passes untinted); an opaque in-range hit returns
    /// true (⇒ ZERO at the caller), as does `tp` decaying under
    /// `SHADOW_TP_MIN` (the early-out that bounds deep glass stacks). Each
    /// tri lives in exactly one leaf slice, so no interface multiplies twice;
    /// the traversal order is deterministic, so the product is bit-stable
    /// per ray (and a two-factor product is order-independent bitwise —
    /// f32 multiplication commutes).
    fn transmittance_from(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        tmax: f32,
        start: u32,
        tp: &mut Vec3A,
        visits: &mut u64,
    ) -> bool {
        let mut stack = [0u32; TRAV_STACK];
        let mut sp = 0usize;
        let mut node_idx = start;
        loop {
            let node = &self.nodes[node_idx as usize];
            *visits += 1;
            if node.count > 0 {
                if node.is_gateway() {
                    // GATEWAY: one shear per cell entry (see intersect_from);
                    // `tp` threads through the nested call, so per-interface
                    // tints inside a cell multiply exactly as before.
                    if node.leaf_count() > 0 {
                        if tri_probe() {
                            GATEWAY_ENTRIES.fetch_add(1, Ordering::Relaxed);
                        }
                        let opaque = if let Some(s) = self.gateway_shear(scene, node) {
                            let sray = shear_ray(ray, &s);
                            self.transmittance_from(scene, &sray, tmin, tmax, node_idx + 1, tp, visits)
                        } else {
                            self.transmittance_from(scene, ray, tmin, tmax, node_idx + 1, tp, visits)
                        };
                        if opaque {
                            return true;
                        }
                    }
                } else {
                    let first = node.left_first as usize;
                    for &t in &self.tri_idx[first..first + node.count as usize] {
                        if let Some((tt, _, _)) = moller_trumbore(scene, t, ray) {
                            if tt > tmin && tt < tmax {
                                let m = &scene.materials[scene.tri_mat[t as usize] as usize];
                                if m.transmission <= 0.0 {
                                    return true;
                                }
                                *tp *= m.shadow_tint();
                                TRANS_PASS.fetch_add(1, Ordering::Relaxed);
                                if tp.max_element() < SHADOW_TP_MIN {
                                    return true;
                                }
                            }
                        }
                    }
                }
            } else {
                let l = node.left_first;
                if slab_t(&self.nodes[l as usize].aabb, ray, tmin, tmax).is_finite() {
                    if slab_t(&self.nodes[l as usize + 1].aabb, ray, tmin, tmax).is_finite() {
                        debug_assert!(sp < stack.len(), "BVH traversal stack overflow");
                        stack[sp] = l + 1;
                        sp += 1;
                    }
                    node_idx = l;
                    continue;
                }
                if slab_t(&self.nodes[l as usize + 1].aabb, ray, tmin, tmax).is_finite() {
                    node_idx = l + 1;
                    continue;
                }
            }
            if sp == 0 {
                return false;
            }
            sp -= 1;
            node_idx = stack[sp];
        }
    }
}

/// Throughput floor for `transmittance`: below this the segment counts as
/// opaque (ZERO) — the deterministic early-out that keeps a deep glass stack
/// from costing unbounded leaf visits. Mirrored in trace_common.hlsli.
pub const SHADOW_TP_MIN: f32 = 1e-3;

/// Diagnostic twin of the GPU's `CTR_TRANS_PASS`: tinted-shadow interface
/// crossings (`shadow_tint` multiplies) since process start. Relaxed adds at
/// transmissive crossings only — rare enough to be free. Read by the check
/// harness's FR_CHECK_AB_DUMP decomposition to compare crossing COUNTS
/// against the GPU counter: a duplicate-candidate divergence multiplies the
/// same tint twice, which is invisible to every value gate but doubles this.
pub static TRANS_PASS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Build one pending subtree into a local arena: root at local 0, internal
/// `left_first` = LOCAL node index (the stitch rebases them by one uniform
/// +off), leaf ranges lifted to GLOBAL tri-buffer positions here so the
/// stitch never touches them. `tri_idx` is this subtree's disjoint ITEM
/// slice; `base` is the slice's global TRI-BUFFER start (== its item start
/// when `gw` is None, since the item array is the tri buffer then). With
/// `gw` Some, `gw.1` is the range's disjoint tri-buffer slice and leaf/
/// gateway emission writes into it at LOCAL offsets; the `count > 0` lift
/// below lands leaves, gateways AND the E filler (harmless — never
/// dereferenced) at their global positions in one pass.
fn build_subtree(
    tri_idx: &mut [u32],
    base: u32,
    tri_aabb: &[Aabb],
    centroids: &[Vec3A],
    gw: Option<(&GatewayCtx, &mut [u32])>,
) -> Vec<BvhNode> {
    let mut nodes = Vec::new();
    nodes.push(BvhNode {
        aabb: Aabb::EMPTY,
        left_first: 0,
        count: tri_idx.len() as u32,
    });
    let mut gphase = gw.map(|(ctx, tribuf)| GwPhase { ctx, tribuf });
    subdivide_range(&mut nodes, 0, tri_idx, tri_aabb, centroids, &mut None, &mut gphase, 0);
    for nd in &mut nodes {
        if nd.count > 0 {
            nd.left_first += base;
        }
    }
    nodes
}

/// Materialize one sway cell: the gateway node at `gw_i` (which MUST be the
/// arena's current tail — the subtree lands implicitly at `gw_i + 1`) over
/// the cell's truthful tri range, written to `tribuf` at `tri_first` in CSR
/// (ascending-tri) order, then the rest-space-TIGHT subtree via the classic
/// ctx-None recursion (inside a cell there are no pseudos, and `tri_aabb`'s
/// first n entries are the unpadded rest boxes). The gateway's box is the
/// subtree root's rest box padded by the cell's displacement bound — the
/// ONE swept box — and must come out bit-equal to the pseudo-prim entry the
/// parent binned with (same union, same order), which the debug_assert pins.
fn emit_gateway(
    nodes: &mut Vec<BvhNode>,
    gw_i: usize,
    pseudo: u32,
    tri_first: u32,
    tri_aabb: &[Aabb],
    centroids: &[Vec3A],
    g: &mut GwPhase<'_>,
) {
    debug_assert_eq!(gw_i + 1, nodes.len(), "gateway subtree must start at gw_i + 1");
    let c = g.ctx.cell(pseudo);
    let range = g.ctx.cell_range(c);
    let cc = range.len();
    debug_assert!(cc > 0, "empty sway cell reached the build");
    g.tribuf[tri_first as usize..tri_first as usize + cc].copy_from_slice(range);
    nodes.push(BvhNode {
        aabb: Aabb::EMPTY,
        left_first: tri_first,
        count: cc as u32,
    });
    let sub_i = gw_i + 1;
    let tb: &mut [u32] = g.tribuf;
    subdivide_range(nodes, sub_i, tb, tri_aabb, centroids, &mut None, &mut None, 0);
    let p = Vec3A::new(g.ctx.pad[c], 0.0, g.ctx.pad[c]);
    let sub_bb = nodes[sub_i].aabb;
    let bb = Aabb { min: sub_bb.min - p, max: sub_bb.max + p };
    debug_assert_eq!(
        bb.min.to_array().map(f32::to_bits),
        tri_aabb[pseudo as usize].min.to_array().map(f32::to_bits),
        "gateway box drifted from its pseudo-prim entry"
    );
    nodes[gw_i] = BvhNode {
        aabb: bb,
        left_first: tri_first,
        count: GATEWAY_BIT | cc as u32,
    };
}

/// Gateway-mode leaf emission: the item range (all static tris here) is
/// copied to its tri-buffer slot and the node's `left_first` moves from
/// ITEM to TRI coordinates. `count` is already the tri count (static items
/// have width 1). Without a ctx the item array IS the tri buffer and no
/// copy exists — the classic path.
fn emit_static_leaf(nodes: &mut [BvhNode], node_i: usize, items: &[u32], tri_first: u32, g: &mut GwPhase<'_>) {
    g.tribuf[tri_first as usize..tri_first as usize + items.len()].copy_from_slice(items);
    nodes[node_i].left_first = tri_first;
}

/// The recursive SAH subdivide, shared by both build phases. `first` in the
/// node being subdivided is relative to `tri_idx` (phase 1 passes the whole
/// array, so relative == global; phase 2 passes its subtree slice). With
/// `pend` set (phase 1), a range at or under the threshold is recorded and
/// left for phase 2 instead of being split.
///
/// GATEWAY mode (`gw` Some): `tri_idx` is the ITEM permutation (static tris
/// + cell pseudo-prims) and `tri_first` is this range's cursor into the
/// final tri buffer (`g.tribuf`), advanced by item WIDTHS. The rules, each
/// load-bearing for the `gateway+1` adjacency: a range containing a pseudo
/// NEVER becomes a leaf (force-split — the median fallback guarantees
/// separation); a singleton-pseudo CHILD is materialized inline at child
/// emission (never recursed, never pended) into the SECOND slot of its pair
/// (child order in a pair is semantically free — traversal reorders by
/// distance); a two-pseudo pair takes the CHERRY expansion `P(Q, G_B)`,
/// `Q(E, G_A)` with E the empty filler leaf. The entry arm below only ever
/// fires for a gateway ROOT (an all-one-cell build). With `gw` None every
/// branch here collapses to the historical code path bit-identically.
fn subdivide_range(
    nodes: &mut Vec<BvhNode>,
    node_i: usize,
    tri_idx: &mut [u32],
    tri_aabb: &[Aabb],
    centroids: &[Vec3A],
    pend: &mut Option<(&mut Vec<Pending>, usize)>,
    gw: &mut Option<GwPhase<'_>>,
    tri_first: u32,
) {
    let (first, count) = {
        let node = &nodes[node_i];
        (node.left_first as usize, node.count as usize)
    };

    // Gateway ROOT: the whole range is one cell. Must precede the pending
    // check (a pended gateway root would tear the +1 adjacency at stitch).
    if let Some(g) = gw.as_mut() {
        if count == 1 && g.ctx.is_pseudo(tri_idx[first]) {
            emit_gateway(nodes, node_i, tri_idx[first], tri_first, tri_aabb, centroids, g);
            return;
        }
    }

    // Width-sum (= tri count) of the range: the pending threshold and the
    // child tri cursors live in TRI space. Equal to `count` without a ctx.
    let (has_pseudo, wsum) = if let Some(g) = gw.as_ref() {
        let mut hp = false;
        let mut w = 0u32;
        for &t in &tri_idx[first..first + count] {
            hp |= g.ctx.is_pseudo(t);
            w += g.ctx.width(t);
        }
        (hp, w as usize)
    } else {
        (false, count)
    };

    if let Some((pending, threshold)) = pend {
        if wsum <= *threshold {
            pending.push(Pending {
                node_i: node_i as u32,
                first: first as u32,
                count: count as u32,
                tri_first,
                tri_count: wsum as u32,
            });
            return;
        }
    }

    let mut bounds = Aabb::EMPTY;
    let mut cbounds = Aabb::EMPTY;
    for &t in &tri_idx[first..first + count] {
        bounds.grow_aabb(&tri_aabb[t as usize]);
        cbounds.grow(centroids[t as usize]);
    }
    nodes[node_i].aabb = bounds;

    if count <= 2 && !has_pseudo {
        if let Some(g) = gw.as_mut() {
            emit_static_leaf(nodes, node_i, &tri_idx[first..first + count], tri_first, g);
        }
        return; // leaf
    }

    let ext = cbounds.max - cbounds.min;
    let parent_area = bounds.area();
    let leaf_cost = parent_area * count as f32;
    let c_trav = c_trav();
    let max_leaf = max_leaf();

    // Binned SAH over ALL THREE axes — the winner is the global (axis, bin)
    // minimum. Binning only the widest centroid axis is cheaper but leaves
    // quality on the table, and the build is a once-per-scene, cached cost.
    //
    // Ties break toward the lowest axis then the lowest bin (strict `<`), which
    // is what keeps the node order a pure function of the geometry — the
    // `Bvh::identical` byte-determinism contract.
    let mut best_cost = f32::INFINITY;
    let mut best_axis = usize::MAX;
    let mut best_bin = 0usize;
    let mut best_k = 0.0f32;
    let mut best_cmin = 0.0f32;

    // 1-axis mode reproduces the historical search: the widest centroid axis only.
    let widest = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };
    let all = [0usize, 1, 2];
    let one = [widest];
    let axes: &[usize] = if split_axes() == 1 { &one } else { &all };

    for &axis in axes {
        let cmin = cbounds.min[axis];
        let cext = ext[axis];
        if cext <= 1e-8 {
            continue; // degenerate on this axis (e.g. identical centroids)
        }
        let mut bin_bounds = [Aabb::EMPTY; BINS];
        let mut bin_count = [0usize; BINS];
        let k = BINS as f32 * (1.0 - 1e-6) / cext;
        for &t in &tri_idx[first..first + count] {
            let b = ((centroids[t as usize][axis] - cmin) * k) as usize;
            bin_count[b] += 1;
            bin_bounds[b].grow_aabb(&tri_aabb[t as usize]);
        }
        // Sweep: cost of splitting after bin i.
        let mut right_area = [0.0f32; BINS];
        let mut right_count = [0usize; BINS];
        let mut acc = Aabb::EMPTY;
        let mut cnt = 0;
        for i in (1..BINS).rev() {
            acc.grow_aabb(&bin_bounds[i]);
            cnt += bin_count[i];
            right_area[i] = acc.area();
            right_count[i] = cnt;
        }
        acc = Aabb::EMPTY;
        cnt = 0;
        for i in 0..BINS - 1 {
            acc.grow_aabb(&bin_bounds[i]);
            cnt += bin_count[i];
            if cnt == 0 || right_count[i + 1] == 0 {
                continue;
            }
            // C_trav*A_P + C_isect*(A_L*N_L + A_R*N_R), C_isect == 1. The
            // C_trav*A_P term is the whole point: without it the leaf test at
            // the bottom can never win (see C_TRAV_BITS).
            let cost = c_trav * parent_area
                + acc.area() * cnt as f32
                + right_area[i + 1] * right_count[i + 1] as f32;
            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_bin = i;
                best_k = k;
                best_cmin = cmin;
            }
        }
    }

    let mut split_at = usize::MAX;
    if best_axis != usize::MAX && (best_cost < leaf_cost || count > max_leaf || has_pseudo) {
        // In-place partition by bin threshold on the winning axis.
        let (axis, k, cmin) = (best_axis, best_k, best_cmin);
        let mut i = first;
        let mut j = first + count - 1;
        while i <= j {
            let b = ((centroids[tri_idx[i] as usize][axis] - cmin) * k) as usize;
            if b <= best_bin {
                i += 1;
            } else {
                tri_idx.swap(i, j);
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        }
        if i > first && i < first + count {
            split_at = i;
        }
    }

    if split_at == usize::MAX {
        if count <= max_leaf && !has_pseudo {
            if let Some(g) = gw.as_mut() {
                emit_static_leaf(nodes, node_i, &tri_idx[first..first + count], tri_first, g);
            }
            return; // leaf
        }
        // Degenerate SAH (e.g., identical centroids) — median split on the
        // widest centroid axis. Phase 1 relies on a too-big node ALWAYS
        // splitting, so this arm must stay unconditional above max_leaf —
        // and a pseudo-carrying range relies on it the same way (count >= 2
        // always yields two non-empty halves here).
        let axis = if ext.x >= ext.y && ext.x >= ext.z {
            0
        } else if ext.y >= ext.z {
            1
        } else {
            2
        };
        tri_idx[first..first + count].sort_unstable_by(|&a, &b| {
            centroids[a as usize][axis]
                .partial_cmp(&centroids[b as usize][axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        split_at = first + count / 2;
    }

    // Child ranges in ITEM space and their TRI-buffer cursors (A's = ours;
    // B's = ours + A's width-sum — cursors follow ITEM order regardless of
    // which SLOT a range lands in below).
    let (a_first, a_count) = (first, split_at - first);
    let (b_first, b_count) = (split_at, first + count - split_at);
    let (a_tri, b_tri) = if let Some(g) = gw.as_ref() {
        let wa: u32 = tri_idx[a_first..a_first + a_count]
            .iter()
            .map(|&t| g.ctx.width(t))
            .sum();
        (tri_first, tri_first + wa)
    } else {
        // Without a ctx the item array IS the tri buffer: cursors unused by
        // the classic arms below, kept in item coords for uniformity.
        (a_first as u32, b_first as u32)
    };

    let l = nodes.len();
    nodes.push(BvhNode {
        aabb: Aabb::EMPTY,
        left_first: a_first as u32,
        count: a_count as u32,
    });
    nodes.push(BvhNode {
        aabb: Aabb::EMPTY,
        left_first: b_first as u32,
        count: b_count as u32,
    });
    nodes[node_i].left_first = l as u32;
    nodes[node_i].count = 0;

    // Singleton-pseudo children are materialized HERE (never recursed, never
    // pended): the gateway must sit in the SECOND slot of its pair so its
    // subtree at +1 doesn't collide with the sibling, and it must be emitted
    // BEFORE the sibling recursion so the subtree really lands at +1.
    let a_sp = gw.as_ref().is_some_and(|g| a_count == 1 && g.ctx.is_pseudo(tri_idx[a_first]));
    let b_sp = gw.as_ref().is_some_and(|g| b_count == 1 && g.ctx.is_pseudo(tri_idx[b_first]));
    match (a_sp, b_sp) {
        (false, false) => {
            subdivide_range(nodes, l, tri_idx, tri_aabb, centroids, pend, gw, a_tri);
            subdivide_range(nodes, l + 1, tri_idx, tri_aabb, centroids, pend, gw, b_tri);
        }
        (false, true) => {
            // B is the gateway — already in the second slot.
            let g = gw.as_mut().unwrap();
            emit_gateway(nodes, l + 1, tri_idx[b_first], b_tri, tri_aabb, centroids, g);
            subdivide_range(nodes, l, tri_idx, tri_aabb, centroids, pend, gw, a_tri);
        }
        (true, false) => {
            // SWAP: slot l takes B (recursed), slot l+1 the gateway A. Child
            // order within a pair is semantically free; the tri cursors keep
            // following ITEM order (A's block precedes B's in the buffer).
            nodes[l] = BvhNode { aabb: Aabb::EMPTY, left_first: b_first as u32, count: b_count as u32 };
            nodes[l + 1] = BvhNode { aabb: Aabb::EMPTY, left_first: a_first as u32, count: 1 };
            let g = gw.as_mut().unwrap();
            emit_gateway(nodes, l + 1, tri_idx[a_first], a_tri, tri_aabb, centroids, g);
            subdivide_range(nodes, l, tri_idx, tri_aabb, centroids, pend, gw, b_tri);
        }
        (true, true) => {
            // CHERRY: two gateways can't be siblings (each needs its subtree
            // at +1). Emit P(Q, G_B), Q(E, G_A) — E is the zero-tri filler
            // leaf (leaf-shaped to every consumer, masked count 0 so the ray
            // arm skips it). E deliberately carries a COPY of its sibling
            // gateway's box, NOT Aabb::EMPTY: an inverted box's quantized
            // ftree slot decodes to inf slack (measured — the self-test
            // rejects it) and inverted-box distance math is a NaN hazard in
            // every other consumer; a duplicate box costs one redundant slab
            // test and can never lower a bound below what G_A itself reports.
            let g = gw.as_mut().unwrap();
            emit_gateway(nodes, l + 1, tri_idx[b_first], b_tri, tri_aabb, centroids, g);
            let m = nodes.len();
            nodes.push(BvhNode { aabb: Aabb::EMPTY, left_first: 0, count: GATEWAY_BIT });
            nodes.push(BvhNode { aabb: Aabb::EMPTY, left_first: a_first as u32, count: 1 });
            emit_gateway(nodes, m + 1, tri_idx[a_first], a_tri, tri_aabb, centroids, g);
            let ga_box = nodes[m + 1].aabb;
            nodes[m].aabb = ga_box;
            nodes[l] = BvhNode { aabb: ga_box, left_first: m as u32, count: 0 };
        }
    }
}

/// Slab test: entry t if the ray hits the box within (tmin, tmax), else +INF.
#[inline(always)]
fn slab_t(aabb: &Aabb, ray: &Ray, tmin: f32, tmax: f32) -> f32 {
    let t1 = (aabb.min - ray.o) * ray.inv_d;
    let t2 = (aabb.max - ray.o) * ray.inv_d;
    let t_enter = t1.min(t2).max_element().max(tmin);
    let t_exit = t1.max(t2).min_element().min(tmax);
    if t_exit >= t_enter { t_enter } else { f32::INFINITY }
}

/// Inward sweep of a height-carrying triangle's AABB: union with the copy
/// translated by `−n̂_g · depth_world(tri)`, so the displaced surface (⊂ the
/// prism) is CONTAINED — strictly-inward displacement pokes below the flat
/// triangle's plane, i.e. outside its bare AABB (exactly zero margin for an
/// axis-aligned floor tri), and a pit-wall hit at `t' < t_plane` from a
/// recessed apex would otherwise fire the exact-zero tmin-overshoot /
/// false-empty gates. Every claim consumer (frustum queries, temporal cache,
/// hemi bounds, the ftree — its slots ARE these AABBs) inherits soundness
/// from this one site. Gated on the SESSION lever, not the V toggle: the
/// swept tree serves both toggle states without a rebuild. Called by every
/// builder's tri-AABB site (bvh.rs + builders.rs).
#[inline]
pub(crate) fn grow_height_sweep(
    scene: &Scene,
    tri: u32,
    a: Vec3A,
    b: Vec3A,
    c: Vec3A,
    bb: &mut Aabb,
) {
    if !scene.any_height || !height_armed() {
        return;
    }
    let d = tri_height_depth(scene, tri);
    if d > 0.0 {
        let n = (b - a).cross(c - a).normalize_or_zero();
        bb.grow(a - n * d);
        bb.grow(b - n * d);
        bb.grow(c - n * d);
        // The edge-extension budget (HEIGHT_EDGE_EXTEND × amp texels =
        // EXTEND × depth in world units): extended hits land laterally
        // BEYOND the footprint, and a hit outside every box would let a
        // frustum claim declare its region empty — pad all axes so the
        // containment argument covers the extension too (conservative:
        // includes the unneeded +n̂ direction).
        let pad = Vec3A::splat(HEIGHT_EDGE_EXTEND * d);
        bb.min -= pad;
        bb.max += pad;
    }
}

/// Sway sweep of a leaf triangle's AABB (`--foliage-sway` v0.2, the swept-box
/// phase): pad BOTH directions by `foliage::sway_pad` — the cell's
/// displacement bound at the sweep multiplier — so the box contains the
/// triangle at EVERY reachable pose (the shear is signed; v0.4 pads x/z
/// only, since `u.y ≡ 0` means no pose ever moves a vertex vertically). Every claim
/// consumer (frustum queries, temporal cache, hemi bounds, structure replay,
/// the ftree, the GPU's uploaded software trees) inherits soundness from
/// this one site, exactly like `grow_height_sweep` above; the containment
/// pin lives in `foliage::self_test` (one `sway_pad` serves the build and
/// the pin — the `tri_height_depth` discipline). Gated on `Scene::sway`,
/// which only exists under `foliage::sweep_armed()` — unarmed sessions and
/// foliage-free scenes build the exact pre-sway tree, and the lever keys the
/// .fcache (`scene_cache::lever_word` bit 5 + the sway word).
#[inline]
pub(crate) fn grow_sway_sweep(scene: &Scene, tri: u32, bb: &mut Aabb) {
    let Some(sw) = &scene.sway else { return };
    let pad = crate::foliage::sway_pad(sw, tri);
    if pad > 0.0 {
        // x/z only — the v0.4 shear has no y component (u.y ≡ 0).
        let p = Vec3A::new(pad, 0.0, pad);
        bb.min -= p;
        bb.max += p;
    }
}

/// World-space relief depth of `tri`: `height_amp` (texel widths) × the
/// triangle's texel size in world units, `sqrt(world_area/(uv_area·w·h))`.
/// ONE function serving BOTH the build-time AABB sweep and the march — their
/// bitwise agreement is the containment proof (displaced surface ⊂ swept
/// AABB), pinned by `height_self_test`. 0.0 = no relief on this triangle
/// (no map / no amp / degenerate UVs or geometry — the march skips and the
/// plane hit stands verbatim).
#[inline]
pub(crate) fn tri_height_depth(scene: &Scene, tri: u32) -> f32 {
    let m = &scene.materials[scene.tri_mat[tri as usize] as usize];
    if m.height_amp <= 0.0 || m.normal_tex == crate::scene::NO_TEX {
        return 0.0;
    }
    let tx = &scene.textures[m.normal_tex as usize];
    let [i0, i1, i2] = scene.indices[tri as usize];
    let v0 = scene.positions[i0 as usize];
    let e1 = scene.positions[i1 as usize] - v0;
    let e2 = scene.positions[i2 as usize] - v0;
    let (uv0, uv1, uv2) = (
        scene.texcoords[i0 as usize],
        scene.texcoords[i1 as usize],
        scene.texcoords[i2 as usize],
    );
    let d1 = uv1 - uv0;
    let d2 = uv2 - uv0;
    let wa = 0.5 * e1.cross(e2).length();
    let ua = 0.5 * (d1.x * d2.y - d1.y * d2.x).abs();
    let denom = ua * (tx.w * tx.h) as f32;
    if !(denom > 1e-20) || !(wa > 0.0) {
        return 0.0;
    }
    let ts = (wa / denom).sqrt();
    if !ts.is_finite() { 0.0 } else { m.height_amp * ts }
}

/// Scene-wide maximum relief depth in world units (`FrameCb::height_max`;
/// 0.0 = no height data, which is also how the CB encodes `any_height` for
/// FLAG_HEIGHT). This is metadata, NOT a conservative ray-t widening:
/// plane-to-relief `dt = depth / abs(dot(ray_dir, normal))` is unbounded at
/// grazing incidence. One parallel pass at GPU-session init.
pub fn height_max_world(scene: &Scene) -> f32 {
    if !scene.any_height || !height_armed() {
        return 0.0;
    }
    (0..scene.indices.len() as u32)
        .into_par_iter()
        .map(|t| tri_height_depth(scene, t))
        .reduce(|| 0.0f32, f32::max)
}

/// The relief march: given the plane hit `(t_p, u, v)` on a height-carrying
/// triangle, march `g(t) = ĥ(t) − field(uv(t))` over the ray∩prism interval
/// (ĥ ∈ [0,1]; 1 = the plane, 0 = full depth below) and return the marched
/// hit `(t', u', v')` — or None when the ray exits the prism untouched (the
/// silhouette/cutout reject: traversal continues). Both ĥ and the
/// barycentrics are exactly AFFINE in t (the bary rates come from Cramer
/// against the plane basis; the normal component cancels identically), so
/// the in-footprint set along the ray is a single interval and refined hits
/// between two in-footprint samples are in-footprint by convexity.
///
/// Entry rules, by orientation (they unify the primary case, the underside
/// case, and the own-triangle secondary-ray case):
/// - PLANE entry (descending, enters at ĥ=1 = exactly `t_p`): hit at the
///   first `g ≤ 0`. A field at exactly 1.0 at entry (255-alpha plateau —
///   `height_bilinear`'s nested lerp is exact there) returns the ORIGINAL
///   hit verbatim — the flat-field bit-identity.
/// - BELOW entry (ascending, enters at ĥ=0, solid): hit at the first
///   `g ≥ 0` — the underside crossing, `t' ≤ t_p`; floors stay opaque from
///   below. Sound against inherited claims because the swept AABBs contain
///   the whole prism.
/// - INTERIOR entry (origin inside the prism — a secondary ray from a
///   recessed shading point, eps-offset along the SHADING-normal side): the
///   two-phase POM shadow rule — skip while solid, then hit on the next
///   `g ≤ 0`. This is what neutralizes eps-offset acne and lets recessed
///   points see out of their own pits; a genuinely-solid interior origin
///   mis-passing is bounded by one depth and documented.
///
/// Pure function of (hit, texels) — zero rng draws; every same-seed /
/// replay / VisCtl-burn contract is structurally untouched.
#[inline]
fn height_march(
    scene: &Scene,
    tri: u32,
    ray: &Ray,
    e1: Vec3A,
    e2: Vec3A,
    t_p: f32,
    u: f32,
    v: f32,
    depth: f32,
) -> Option<(f32, f32, f32)> {
    let m = &scene.materials[scene.tri_mat[tri as usize] as usize];
    let tx = &scene.textures[m.normal_tex as usize];
    let nn = e1.cross(e2);
    let n2 = nn.dot(nn);
    // ĥ slope along the ray, per unit t (ĥ(t) = 1 + (t − t_p)·dh). The MT
    // det guard already rejected plane-parallel rays, so dh ≠ 0.
    let dh = ray.d.dot(nn) / (n2.sqrt() * depth);
    if !dh.is_finite() || dh == 0.0 {
        return Some((t_p, u, v));
    }
    // Barycentric rates along the ray: β1(p) = ((p−v0)×e2)·N/|N|², so
    // β̇1 = (d×e2)·N/|N|² (and the transpose for β2). The n̂_g component of
    // d cancels — (N×e2)·N ≡ 0 — so no explicit projection is needed.
    let bu = ray.d.cross(e2).dot(nn) / n2;
    let bv = e1.cross(ray.d).dot(nn) / n2;
    let field = |b1: f32, b2: f32| -> f32 {
        let uv = scene.tri_uv(tri, b1, b2);
        tx.height_bilinear(uv.x, uv.y)
    };
    // Interval endpoints: ĥ=1 at exactly t_p, ĥ=0 at t_p − 1/dh.
    let t_h0 = t_p - 1.0 / dh;
    let (t_a, t_b, sgn, two_phase) = if dh < 0.0 {
        (t_p, t_h0, 1.0f32, false) // plane entry, marching down
    } else if t_h0 > 0.0 {
        (t_h0, t_p, -1.0f32, false) // below entry, marching up through solid
    } else {
        (0.0, t_p, 1.0f32, true) // interior entry (origin inside the prism)
    };
    // Footprint interval on t — each bary constraint is linear in t and
    // holds AT t_p (MT accepted the plane hit), so ct > 0 clips the start
    // and ct < 0 clips the end, uniformly for both march directions.
    let (mut lo, hi_slab) = (t_a, t_b);
    let mut hi_foot = t_b;
    for (c0, ct) in [(u, bu), (v, bv), (1.0 - u - v, -bu - bv)] {
        if ct != 0.0 {
            let tc = t_p - c0 / ct;
            if ct > 0.0 {
                lo = lo.max(tc);
            } else {
                hi_foot = hi_foot.min(tc);
            }
        }
    }
    // Edge extension past the exit edge (HEIGHT_EDGE_EXTEND — the crack
    // fix; see its header). Bounded in uv travel and by the slab itself.
    let mut hi = hi_foot.min(hi_slab);
    if hi_foot < hi_slab {
        let [i0, i1, i2] = scene.indices[tri as usize];
        let uv0 = scene.texcoords[i0 as usize];
        let duv = (scene.texcoords[i1 as usize] - uv0) * bu
            + (scene.texcoords[i2 as usize] - uv0) * bv;
        let texel_rate =
            glam::Vec2::new(duv.x * tx.w as f32, duv.y * tx.h as f32).length();
        if texel_rate > 0.0 {
            // The uv-travel budget, CLAMPED to the sweep's world-space pad
            // (HEIGHT_EDGE_EXTEND · depth — grow_height_sweep). texel_rate is
            // DIRECTIONAL while `depth` carries the geometric-mean texel size,
            // so on an anisotropic chart the unclamped budget along the
            // sparse UV axis could exceed the pad and land an extension hit
            // outside the swept AABB — the claim-violation class the pad
            // exists to prevent. The clamp restores containment (travel along
            // a unit-dir ray ≤ budget in every axis ≤ the pad); the price is
            // that heavily stretched charts may under-fill their cracks.
            let budget = (HEIGHT_EDGE_EXTEND * m.height_amp / texel_rate)
                .min(HEIGHT_EDGE_EXTEND * depth);
            hi = (hi_foot + budget).min(hi_slab);
        }
    }
    if !(hi > lo) {
        return None;
    }
    if sgn > 0.0 && !two_phase && field(u, v) >= 1.0 {
        return Some((t_p, u, v)); // flat plateau at the plane: the exact hit
    }
    let f_at = |t: f32| -> f32 {
        let b1 = u + (t - t_p) * bu;
        let b2 = v + (t - t_p) * bv;
        sgn * (1.0 + (t - t_p) * dh - field(b1, b2))
    };
    // Extension hits shade with the EDGE's attributes: clamp the reported
    // bary to the footprint (identity — same bits — for interior hits).
    let out = |t_hit: f32| -> Option<(f32, f32, f32)> {
        let mut b1 = (u + (t_hit - t_p) * bu).max(0.0);
        let mut b2 = (v + (t_hit - t_p) * bv).max(0.0);
        let s = b1 + b2;
        if s > 1.0 {
            b1 /= s;
            b2 /= s;
        }
        Some((t_hit, b1, b2))
    };
    let step = (hi - lo) / HEIGHT_COARSE as f32;
    let mut armed = !two_phase;
    let mut prev: Option<(f32, f32)> = None;
    for k in 0..=HEIGHT_COARSE {
        let t_k = if k == HEIGHT_COARSE { hi } else { lo + step * k as f32 };
        let f = f_at(t_k);
        if f <= 0.0 {
            if armed {
                let t_hit = match prev {
                    Some((ta, fa)) => {
                        // Bisect + secant inside the bracket.
                        let (mut ta, mut fa, mut tb, mut fb) = (ta, fa, t_k, f);
                        for _ in 0..HEIGHT_REFINE {
                            let tm = 0.5 * (ta + tb);
                            let fm = f_at(tm);
                            if fm <= 0.0 {
                                (tb, fb) = (tm, fm);
                            } else {
                                (ta, fa) = (tm, fm);
                            }
                        }
                        if fa > fb {
                            (ta + fa * (tb - ta) / (fa - fb)).clamp(ta, tb)
                        } else {
                            tb
                        }
                    }
                    // No air sample before the crossing (entered solid at
                    // the entry face / side): hit here.
                    None => t_k,
                };
                return out(t_hit);
            }
            // interior phase A: still inside the solid — keep skipping.
        } else {
            armed = true;
            prev = Some((t_k, f));
        }
    }
    None
}

/// Empty-scene traversal gates, run by `--check`: construction must not
/// descend through the sentinel root, and every query must return its clear
/// identity without touching either that root or a supplied cut.
pub fn empty_self_test() -> Result<(), String> {
    let mut scene = crate::scene::SceneBuilder::new().finish(crate::scene::default_sun());
    // Exercise the dedicated transmissive arms too. An empty scene has no
    // material to derive this bit naturally, but clear space remains ONE
    // regardless of the scene-level fast-path selector.
    scene.any_transmissive = true;
    let bvh = Bvh::build(&scene);
    if !bvh.is_empty() || bvh.max_depth() != 0 {
        return Err("empty build is not an empty depth-0 hierarchy".into());
    }
    let q = bvh.quality(SAH_REF_C_TRAV);
    if q.max_depth != 0 || q.internal != 0 || q.leaves != 0 || q.tri_refs != 0 {
        return Err("empty quality report contains hierarchy work".into());
    }

    let ray = Ray::new(Vec3A::ZERO, Vec3A::Z);
    // Deliberately invalid roots prove multi-root entry points take the empty
    // identity before dereferencing a caller-supplied node id.
    let roots = [u32::MAX];
    let mut visits = 0;
    if bvh.intersect(&scene, &ray, 0.0, 10.0, &mut visits).is_some()
        || bvh
            .intersect_multi(&scene, &ray, 0.0, 10.0, &roots, &mut visits)
            .is_some()
        || bvh.occluded(&scene, &ray, 0.0, 10.0, &mut visits)
        || bvh.occluded_multi(&scene, &ray, 0.0, 10.0, &roots, &mut visits)
    {
        return Err("empty visibility query reported a hit".into());
    }
    if bvh.transmittance(&scene, &ray, 0.0, 10.0, &mut visits) != Vec3A::ONE
        || bvh.transmittance_multi(&scene, &ray, 0.0, 10.0, &roots, &mut visits)
            != Vec3A::ONE
    {
        return Err("empty transmittance query did not return ONE".into());
    }
    let f = crate::frustum::TileFrustum::half_space(Vec3A::ZERO, Vec3A::Y);
    if crate::frustum::nearest_geometry_distance_within(
        &bvh,
        &f,
        0.0,
        f32::INFINITY,
        &roots,
        &mut visits,
    )
    .is_some()
    {
        return Err("empty frustum bound query reported geometry".into());
    }
    let mut cut = [0u32; crate::frustum::MAX_CUT];
    let mut overflows = 0;
    if crate::frustum::refine_cut(
        &bvh,
        &f,
        0.0,
        f32::INFINITY,
        &roots,
        &mut cut,
        &mut visits,
        &mut overflows,
    ) != 0
        || overflows != 0
    {
        return Err("empty frustum cut was not empty".into());
    }
    if visits != 0 {
        return Err(format!("empty traversal visited {visits} nodes"));
    }
    Ok(())
}

/// Relief-march gates, run by `--check` (the sphcell precedent — analytic
/// single-triangle scenes, closed-form expectations): the flat-field bitwise
/// identity, marched-hit closed forms (vertical + oblique, incl. the bary
/// rates), the silhouette/side-exit reject must-fire, interior-entry escape
/// and pit-wall occlusion, the underside crossing, grazing finite-interval
/// regressions, prism containment, the build-vs-march depth pin, and the
/// toggle-off bitwise identity.
pub fn height_self_test() -> Result<(), String> {
    use crate::scene::{MatKind, Material, NO_TEX, Scene};
    use crate::texture::Texture;
    let mk_scene = |amp: f32, alpha: &dyn Fn(u32, u32) -> u8| -> Scene {
        let mut sc = Scene {
            positions: vec![Vec3A::ZERO, Vec3A::new(4.0, 0.0, 0.0), Vec3A::new(0.0, 4.0, 0.0)],
            normals: vec![Vec3A::Z; 3],
            texcoords: vec![
                glam::Vec2::new(0.0, 0.0),
                glam::Vec2::new(1.0, 0.0),
                glam::Vec2::new(0.0, 1.0),
            ],
            indices: vec![[0, 1, 2]],
            tri_mat: vec![0],
            materials: vec![Material {
                albedo: Vec3A::ONE,
                roughness: 0.8,
                metallic: 0.0,
                anisotropy: 0.0,
                sheen: 0.0,
                translucency: 0.0,
                transmission: 0.0,
                trans_tint: Vec3A::splat(-1.0),
                ior: 1.5,
                ripple_amp: 0.0,
                emissive: Vec3A::ZERO,
                normal_tex: 0,
                normal_scale: 1.0,
                height_amp: amp,
                rough_tex: NO_TEX,
                metal_tex: NO_TEX,
                emissive_tex: NO_TEX,
                class: crate::matclass::IDX_DEFAULT as u8,
                kind: MatKind::Diffuse,
            }],
            textures: vec![Texture {
                w: 8,
                h: 8,
                texels: (0..8)
                    .flat_map(|y| (0..8).map(move |x| [128, 128, 255, alpha(x, y)]))
                    .collect(),
                alpha_masked: false,
                srgb: false,
                source: String::new(),
                h2n: true,
                n2h: false,
                mips: Vec::new(),
            }],
            any_alpha: false,
            any_height: false,
            any_transmissive: false,
            sun: crate::sky::Sun::new(Vec3A::Y),
            sky_sh: crate::sh::Sh9::ZERO,
            sky_scale: 1.0,
            night: 0.0,
            sway: None,
            sway_regions: Vec::new(),
            diag: 1.0,
            eps: 1e-4,
            ao_radius: 0.03,
            content_min: Vec3A::ZERO,
            content_max: Vec3A::ZERO,
        };
        crate::scene::finalize_scalars(&mut sc);
        sc
    };
    let saved_on = HEIGHT_ON.load(Ordering::Relaxed);
    let saved_armed = HEIGHT_ARMED.load(Ordering::Relaxed);
    set_height_on(true);
    set_height_armed(true);
    let restore = |r: Result<(), String>| {
        HEIGHT_ON.store(saved_on, Ordering::Relaxed);
        HEIGHT_ARMED.store(saved_armed, Ordering::Relaxed);
        r
    };
    let run = || -> Result<(), String> {
        let mut vis = 0u64;
        // Depth pin: world area 8, uv area 0.5, 8×8 texels ⇒ texel size
        // sqrt(8/32) = 0.5 world units; amp 2.0 ⇒ depth exactly 1.0. The
        // BUILD's sweep and the MARCH call this same function — this pin is
        // the containment agreement.
        let flat = mk_scene(2.0, &|_, _| 255);
        if tri_height_depth(&flat, 0) != 1.0 {
            return Err(format!("depth pin: {} != 1.0", tri_height_depth(&flat, 0)));
        }
        // (a) Flat 255 field: bitwise the plane hit (the constructed entry).
        let bvh = Bvh::build(&flat);
        let ray = Ray::new(Vec3A::new(1.0, 1.0, 5.0), -Vec3A::Z);
        let h = bvh.intersect(&flat, &ray, 0.0, f32::INFINITY, &mut vis)
            .ok_or("flat: no hit")?;
        let zero = mk_scene(0.0, &|_, _| 255);
        let bvh0 = Bvh::build(&zero);
        let h0 = bvh0.intersect(&zero, &ray, 0.0, f32::INFINITY, &mut vis)
            .ok_or("flat: no plane hit")?;
        if (h.t.to_bits(), h.u.to_bits(), h.v.to_bits()) != (h0.t.to_bits(), h0.u.to_bits(), h0.v.to_bits())
        {
            return Err(format!("flat-field identity: ({},{},{}) vs ({},{},{})", h.t, h.u, h.v, h0.t, h0.u, h0.v));
        }
        // (b) Vertical ray on an x-ramp: uv fixed along the ray, so the
        // crossing is ĥ = field(uv) ⇒ t' = t_p + (1−f)·depth, closed-form.
        let ramp = mk_scene(2.0, &|x, _| (x * 16) as u8);
        let bvhr = Bvh::build(&ramp);
        let h = bvhr.intersect(&ramp, &ray, 0.0, f32::INFINITY, &mut vis)
            .ok_or("ramp: no hit")?;
        let f0 = 24.0 / 255.0; // bilinear at tex-x 1.5: (16+32)/2
        let want = 5.0 + (1.0 - f0) * 1.0;
        if ((h.t - want) / want).abs() > 1e-5 {
            return Err(format!("ramp vertical: t' {} want {want}", h.t));
        }
        if h.t < 5.0 {
            return Err("front-side hit moved CLOSER than the plane".into());
        }
        // (c) Oblique ray on a constant 128 field: closed-form t' AND the
        // marched barycentrics (the lateral tracking).
        let half = mk_scene(2.0, &|_, _| 128);
        let bvhh = Bvh::build(&half);
        let d = Vec3A::new(0.3, 0.1, -1.0).normalize();
        let o = Vec3A::new(1.0, 1.0, 2.0);
        let ray_o = Ray::new(o, d);
        let h = bvhh.intersect(&half, &ray_o, 0.0, f32::INFINITY, &mut vis)
            .ok_or("oblique: no hit")?;
        let t_p = 2.0 / -d.z;
        let f = 128.0 / 255.0;
        let want_t = t_p + (1.0 - f) / (-d.z); // dh = d·ẑ/depth, depth 1
        if ((h.t - want_t) / want_t).abs() > 1e-5 {
            return Err(format!("oblique: t' {} want {want_t}", h.t));
        }
        let p = o + d * h.t;
        let (want_u, want_v) = (p.x / 4.0, p.y / 4.0);
        if (h.u - want_u).abs() > 1e-5 || (h.v - want_v).abs() > 1e-5 {
            return Err(format!("oblique bary ({},{}) want ({want_u},{want_v})", h.u, h.v));
        }
        // Containment: the marched point lies inside the prism (ĥ ∈ [0,1]
        // within slack), hence inside the swept AABB.
        if !(p.z <= 1e-5 && p.z >= -1.0 - 1e-5) {
            return Err(format!("oblique hit z {} outside the prism", p.z));
        }
        // (d) Silhouette / side-exit reject: an empty prism (alpha 0) and a
        // grazing ray that leaves the footprint before reaching the bottom —
        // relief-on must return NO hit where the flat triangle DID hit
        // (anti-vacuity: the reject really rejected something).
        let pit = mk_scene(2.0, &|_, _| 0);
        let bvhp = Bvh::build(&pit);
        let dg = Vec3A::new(1.0, 0.0, -0.15).normalize();
        let rayg = Ray::new(Vec3A::new(2.0, 0.2, 0.2), dg);
        if bvhp.intersect(&pit, &rayg, 0.0, f32::INFINITY, &mut vis).is_some() {
            return Err("side-exit: expected a reject (silhouette)".into());
        }
        if bvh0.intersect(&zero, &rayg, 0.0, f32::INFINITY, &mut vis).is_none() {
            return Err("side-exit: the flat triangle must hit (vacuous reject test)".into());
        }
        // (e) Interior entry: an eps-offset-style origin inside the prism.
        // Escape: uniform low field, ray up — must NOT occlude (the pit sees
        // its sky). Wall: a 255 wall region ahead — must occlude, with
        // t' < t_plane (sound only because of the swept AABBs).
        let low = mk_scene(2.0, &|_, _| 64);
        let bvhl = Bvh::build(&low);
        let up = Ray::new(Vec3A::new(1.0, 1.0, -0.3), Vec3A::Z);
        if bvhl.occluded(&low, &up, 0.0, 10.0, &mut vis) {
            return Err("interior escape: upward ray must not be occluded".into());
        }
        let walled = mk_scene(2.0, &|x, _| if x >= 4 { 255 } else { 64 });
        let bvhw = Bvh::build(&walled);
        // dz/dx = 0.45: the plane hit stays inside the footprint (x ≤ 3 at
        // y = 1) while ĥ is still below 1.0 when the ray reaches the 255
        // wall band (world x ∈ [1.75, 2.25]) — both constraints needed.
        let dw = Vec3A::new(1.0, 0.0, 0.45).normalize();
        let rayw = Ray::new(Vec3A::new(1.0, 1.0, -0.7), dw);
        if !bvhw.occluded(&walled, &rayw, 0.0, 10.0, &mut vis) {
            return Err("pit wall: tilted ray must be occluded".into());
        }
        let hw = bvhw.intersect(&walled, &rayw, 0.0, f32::INFINITY, &mut vis)
            .ok_or("pit wall: no hit")?;
        let t_plane = 0.7 / dw.z;
        if hw.t >= t_plane {
            return Err(format!("pit wall t' {} must be < plane t {t_plane}", hw.t));
        }
        // (f) Underside crossing: from below, the solid's bottom is met at
        // ĥ = field ⇒ t' = t_h0 + f·(t_p − t_h0); floors stay opaque.
        let rayb = Ray::new(Vec3A::new(1.0, 1.0, -2.0), Vec3A::Z);
        let hb = bvhh.intersect(&half, &rayb, 0.0, f32::INFINITY, &mut vis)
            .ok_or("underside: no hit")?;
        let want_b = 1.0 + f; // t_h0 = 1 (z −2→−1), t_p = 2, ĥ spans [0,1]
        if ((hb.t - want_b) / want_b).abs() > 1e-5 {
            return Err(format!("underside: t' {} want {want_b}", hb.t));
        }
        // (f2) A world-depth +/- widening is NOT a ray-t bound. At grazing
        // incidence, dt = depth/|d.n| exceeds the one-unit depth. Pin both
        // interval ends against the CPU reference:
        //   * descending: plane t is BEFORE old (tmin-depth), marched t is in;
        //   * ascending:  plane t is AFTER old (tmax+depth), marched t is in.
        // The GPU must enumerate the full positive base-triangle ray and
        // apply these logical bounds only after candidate_reject marches t.
        {
            let dg_down = Vec3A::new(1.0, 0.0, -0.4).normalize();
            let rg_down = Ray::new(Vec3A::new(0.2, 0.2, 0.8), dg_down);
            let hg_down = bvhh
                .intersect(&half, &rg_down, 0.0, f32::INFINITY, &mut vis)
                .ok_or("grazing tmin: no full-interval hit")?;
            let tp_down = 0.8 / -dg_down.z;
            let logical_tmin = hg_down.t - 0.05;
            if !(tp_down < logical_tmin - 1.0 && hg_down.t > logical_tmin) {
                return Err(format!(
                    "grazing tmin setup invalid: plane {tp_down}, hit {}, logical {logical_tmin}",
                    hg_down.t
                ));
            }
            let bounded = bvhh
                .intersect(&half, &rg_down, logical_tmin, f32::INFINITY, &mut vis)
                .ok_or("grazing tmin: marched in-range hit was culled")?;
            if (bounded.t - hg_down.t).abs() > 1e-5 {
                return Err(format!(
                    "grazing tmin: bounded hit {} != full hit {}",
                    bounded.t, hg_down.t
                ));
            }

            let dg_up = Vec3A::new(1.0, 0.0, 0.4).normalize();
            let rg_up = Ray::new(Vec3A::new(0.2, 0.2, -1.1), dg_up);
            let hg_up = bvhh
                .intersect(&half, &rg_up, 0.0, f32::INFINITY, &mut vis)
                .ok_or("grazing tmax: no full-interval hit")?;
            let tp_up = 1.1 / dg_up.z;
            let logical_tmax = hg_up.t + 0.05;
            if !(tp_up > logical_tmax + 1.0 && hg_up.t < logical_tmax) {
                return Err(format!(
                    "grazing tmax setup invalid: plane {tp_up}, hit {}, logical {logical_tmax}",
                    hg_up.t
                ));
            }
            let bounded = bvhh
                .intersect(&half, &rg_up, 0.0, logical_tmax, &mut vis)
                .ok_or("grazing tmax: marched in-range hit was culled")?;
            if (bounded.t - hg_up.t).abs() > 1e-5 {
                return Err(format!(
                    "grazing tmax: bounded hit {} != full hit {}",
                    bounded.t, hg_up.t
                ));
            }
        }
        // (g) Edge-crack fill (HEIGHT_EDGE_EXTEND): two coplanar triangles
        // sharing an edge on one continuous chart, constant mid field
        // (surface at ĥ = 128/255). A ray entering A near the edge crosses
        // into B's prism IN AIR — B never candidates (one shared plane, the
        // crossing lies inside A), so pre-fix this ray fell through the
        // crack. The extension must hit the continuous surface at the flat
        // level, at a point provably beyond A's footprint.
        {
            let mut quad = mk_scene(2.0, &|_, _| 128);
            quad.positions.push(Vec3A::new(4.0, 4.0, 0.0));
            quad.normals.push(Vec3A::Z);
            quad.texcoords.push(glam::Vec2::new(1.0, 1.0));
            quad.indices.push([1, 3, 2]);
            quad.tri_mat.push(0);
            crate::scene::finalize_scalars(&mut quad);
            let bvhq = Bvh::build(&quad);
            let dq = Vec3A::new(1.0, 0.0, -1.0).normalize();
            let oq = Vec3A::new(2.0, 0.8, 1.0);
            let hq = bvhq
                .intersect(&quad, &Ray::new(oq, dq), 0.0, f32::INFINITY, &mut vis)
                .ok_or("edge crack: the crossing ray must hit (leaked)")?;
            let f = 128.0 / 255.0;
            let want_t = (1.0 + (1.0 - f)) * std::f32::consts::SQRT_2;
            if ((hq.t - want_t) / want_t).abs() > 1e-4 {
                return Err(format!("edge crack: t' {} want {want_t}", hq.t));
            }
            let p = oq + dq * hq.t;
            if p.x + p.y <= 4.0 {
                return Err(format!(
                    "edge crack: hit at x+y {} is inside A — the extension didn't fire",
                    p.x + p.y
                ));
            }
            if hq.u < -1e-6 || hq.v < -1e-6 || hq.u + hq.v > 1.0 + 1e-6 {
                return Err(format!("edge crack: bary ({}, {}) not clamped", hq.u, hq.v));
            }
            // Containment: the extension hit must lie inside the producing
            // triangle's swept+padded AABB — the claim-soundness contract the
            // budget clamp exists for (a hit outside every occupied box would
            // let a frustum claim declare its region empty).
            let mut bb = Aabb::EMPTY;
            let (a, b2, c) =
                (quad.positions[0], quad.positions[1], quad.positions[2]);
            bb.grow(a);
            bb.grow(b2);
            bb.grow(c);
            grow_height_sweep(&quad, 0, a, b2, c, &mut bb);
            let eps = 1e-5;
            if (p.cmplt(bb.min - Vec3A::splat(eps)) | p.cmpgt(bb.max + Vec3A::splat(eps))).any()
            {
                return Err(format!(
                    "edge crack: hit {p:?} outside the swept box [{:?}, {:?}]",
                    bb.min, bb.max
                ));
            }
        }

        // (h) Toggle off: bitwise the plane hit of an amp-0 scene.
        set_height_on(false);
        let hoff = bvhr.intersect(&ramp, &ray, 0.0, f32::INFINITY, &mut vis)
            .ok_or("toggle-off: no hit")?;
        set_height_on(true);
        let hz = bvh0.intersect(&zero, &ray, 0.0, f32::INFINITY, &mut vis).unwrap();
        if (hoff.t.to_bits(), hoff.u.to_bits(), hoff.v.to_bits())
            != (hz.t.to_bits(), hz.u.to_bits(), hz.v.to_bits())
        {
            return Err("toggle-off is not bitwise the plane hit".into());
        }
        Ok(())
    };
    let r = run();
    restore(r)
}

/// Tinted-shadow gates, run by `--check` (the height_self_test pattern —
/// analytic scenes, closed-form expectations): single-interface tint bitwise,
/// two-interface product, opaque termination, the `SHADOW_TP_MIN` cutoff,
/// the primary-visibility pin (glass still HITS), binary `occluded` still
/// seeing glass (the geometric-oracle contract), the cut-seeded twin, and the
/// lever-off binary block.
pub fn tinted_shadow_self_test() -> Result<(), String> {
    use crate::scene::{MatKind, Material, NO_TEX, Scene};
    let mat_tinted = |transmission: f32, albedo: Vec3A, trans_tint: Vec3A| Material {
        albedo,
        roughness: 0.05,
        metallic: 0.0,
        anisotropy: 0.0,
        sheen: 0.0,
        translucency: 0.0,
        transmission,
        trans_tint,
        ior: 1.5,
        ripple_amp: 0.0,
        emissive: Vec3A::ZERO,
        normal_tex: NO_TEX,
        normal_scale: 1.0,
        height_amp: 0.0,
        rough_tex: NO_TEX,
        metal_tex: NO_TEX,
        emissive_tex: NO_TEX,
        class: crate::matclass::IDX_DEFAULT as u8,
        kind: MatKind::Diffuse,
    };
    // Sentinel tint = "use albedo" — the bit-identity path every existing
    // transmissive material takes.
    let mat = |transmission: f32, albedo: Vec3A| mat_tinted(transmission, albedo, Vec3A::splat(-1.0));
    // Parallel triangles across the ray o=(1,1,5), d=-Z: glass1 at z=3
    // (t=2), glass2 at z=2 (t=3), opaque at z=1 (t=4).
    let mk_scene = |mats: Vec<Material>, zs: &[f32]| -> Scene {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for (i, &z) in zs.iter().enumerate() {
            positions.push(Vec3A::new(0.0, 0.0, z));
            positions.push(Vec3A::new(4.0, 0.0, z));
            positions.push(Vec3A::new(0.0, 4.0, z));
            let b = (i * 3) as u32;
            indices.push([b, b + 1, b + 2]);
        }
        let n = positions.len();
        let tri_mat = (0..zs.len() as u32).collect();
        let mut sc = Scene {
            positions,
            normals: vec![Vec3A::Z; n],
            texcoords: vec![glam::Vec2::ZERO; n],
            indices,
            tri_mat,
            materials: mats,
            textures: Vec::new(),
            any_alpha: false,
            any_height: false,
            any_transmissive: false,
            sun: crate::sky::Sun::new(Vec3A::Y),
            sky_sh: crate::sh::Sh9::ZERO,
            sky_scale: 1.0,
            night: 0.0,
            sway: None,
            sway_regions: Vec::new(),
            diag: 1.0,
            eps: 1e-4,
            ao_radius: 0.03,
            content_min: Vec3A::ZERO,
            content_max: Vec3A::ZERO,
        };
        crate::scene::finalize_scalars(&mut sc);
        sc
    };
    let saved = crate::scene::tinted_shadows();
    crate::scene::set_tinted_shadows(true);
    let restore = |r: Result<(), String>| {
        crate::scene::set_tinted_shadows(saved);
        r
    };
    let run = || -> Result<(), String> {
        let mut vis = 0u64;
        let (g1, g2) = (
            mat(0.9, Vec3A::new(0.82, 0.9, 1.0)),
            mat(0.5, Vec3A::new(0.5, 0.6, 0.7)),
        );
        let (tint1, tint2) = (g1.shadow_tint(), g2.shadow_tint());
        let sc = mk_scene(vec![g1, g2, mat(0.0, Vec3A::ONE)], &[3.0, 2.0, 1.0]);
        if !sc.any_transmissive {
            return Err("any_transmissive not armed".into());
        }
        let bvh = Bvh::build(&sc);
        let ray = Ray::new(Vec3A::new(1.0, 1.0, 5.0), -Vec3A::Z);
        // (a) One interface: bitwise the material's tint.
        let tp = bvh.transmittance(&sc, &ray, 0.0, 2.5, &mut vis);
        if tp.to_array().map(f32::to_bits) != tint1.to_array().map(f32::to_bits) {
            return Err(format!("one interface: {tp:?} != {tint1:?}"));
        }
        // (b) Two interfaces: the product (order-independent bitwise — f32
        // multiplication commutes and there are exactly two factors).
        let tp = bvh.transmittance(&sc, &ray, 0.0, 3.5, &mut vis);
        let want = tint1 * tint2;
        if tp.to_array().map(f32::to_bits) != want.to_array().map(f32::to_bits) {
            return Err(format!("two interfaces: {tp:?} != {want:?}"));
        }
        // (c) Opaque termination: exactly ZERO.
        if bvh.transmittance(&sc, &ray, 0.0, 10.0, &mut vis) != Vec3A::ZERO {
            return Err("opaque hit must zero the throughput".into());
        }
        // (d) The primary-visibility pin: glass still HITS (shading-only
        // feature — visibility untouched).
        let h = bvh
            .intersect(&sc, &ray, 0.0, f32::INFINITY, &mut vis)
            .ok_or("primary ray must still hit glass")?;
        if (h.t - 2.0).abs() > 1e-5 || h.tri != 0 {
            return Err(format!("primary hit t {} tri {} — want the z=3 glass", h.t, h.tri));
        }
        // (e) Binary `occluded` still sees glass — the geometric-oracle
        // contract (empty-cell verification must not look through glass).
        if !bvh.occluded(&sc, &ray, 0.0, 2.5, &mut vis) {
            return Err("occluded() must keep binary any-geometry semantics".into());
        }
        // (f) Cut-seeded twin from the root cut is bit-equal.
        let tp_m = bvh.transmittance_multi(&sc, &ray, 0.0, 2.5, &[0], &mut vis);
        if tp_m.to_array().map(f32::to_bits) != tint1.to_array().map(f32::to_bits) {
            return Err(format!("multi[root]: {tp_m:?} != {tint1:?}"));
        }
        // (f2) An explicit trans_tint replaces albedo as the tint source (the
        // water class): shadow_tint = trans_tint · transmission bitwise, even
        // though the albedo (a dark 0.1) differs — and transmittance carries
        // it through.
        let wtint = Vec3A::new(0.75, 0.92, 0.96);
        let wt = mat_tinted(0.9, Vec3A::splat(0.1), wtint);
        let want_wt = wtint * 0.9;
        if wt.shadow_tint().to_array().map(f32::to_bits) != want_wt.to_array().map(f32::to_bits) {
            return Err("trans_tint must override albedo as the shadow tint source".into());
        }
        let sc_w = mk_scene(vec![wt], &[3.0]);
        let bvhw = Bvh::build(&sc_w);
        let tpw = bvhw.transmittance(&sc_w, &ray, 0.0, 2.5, &mut vis);
        if tpw.to_array().map(f32::to_bits) != want_wt.to_array().map(f32::to_bits) {
            return Err(format!("water transmittance {tpw:?} != {want_wt:?}"));
        }
        // (g) SHADOW_TP_MIN cutoff: a tint under the floor counts opaque.
        let faint = mk_scene(vec![mat(5.0e-4, Vec3A::ONE)], &[3.0]);
        let bvhf = Bvh::build(&faint);
        if bvhf.transmittance(&faint, &ray, 0.0, 10.0, &mut vis) != Vec3A::ZERO {
            return Err("sub-floor tint must count as opaque".into());
        }
        // (h) Lever off: finalize never arms, glass binary-blocks (the
        // pre-feature behavior, bit-identically through the occluded arm).
        crate::scene::set_tinted_shadows(false);
        let sc_off = mk_scene(
            vec![mat(0.9, Vec3A::new(0.82, 0.9, 1.0))],
            &[3.0],
        );
        crate::scene::set_tinted_shadows(true);
        if sc_off.any_transmissive {
            return Err("lever off: any_transmissive must not arm".into());
        }
        let bvho = Bvh::build(&sc_off);
        if bvho.transmittance(&sc_off, &ray, 0.0, 10.0, &mut vis) != Vec3A::ZERO {
            return Err("lever off: glass must binary-block".into());
        }
        Ok(())
    };
    let r = run();
    restore(r)
}

/// `FR_SWAY_TRI=1` — a SILENT read-only probe (the `FR_ORACLE` arming shape):
/// count möller-trumbore triangle tests, and the subset that took a nonzero
/// sway shift. Deliberately global CUMULATIVE relaxed atomics rather than the
/// `LocalStats` batching discipline: the traversal loops thread a bare
/// `visits: &mut u64`, so a batched test counter would mean a second
/// out-param through every entry point and dozens of call sites — for a
/// probe. The cost trade is documented instead: an ARMED run pays measurable
/// per-test counting contention, so take COUNTS from an armed run and
/// MILLISECONDS from an unarmed one — sound because `--spin` is
/// deterministic (counters are bit-identical across runs). Unarmed cost is
/// one initialized-OnceLock deref + a predictable branch per test, identical
/// across every A/B arm, so it cancels in every delta.
pub static TRI_TESTS: AtomicU64 = AtomicU64::new(0);
pub static TRI_TESTS_SHIFTED: AtomicU64 = AtomicU64::new(0);
/// Gateway subtree entries (nested descents), same probe discipline.
pub static GATEWAY_ENTRIES: AtomicU64 = AtomicU64::new(0);
pub fn tri_probe() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FR_SWAY_TRI").is_ok_and(|v| v != "0"))
}

#[inline(always)]
fn moller_trumbore(scene: &Scene, tri: u32, ray: &Ray) -> Option<(f32, f32, f32)> {
    if tri_probe() {
        TRI_TESTS.fetch_add(1, Ordering::Relaxed);
    }
    // --foliage-sway: translate the ray into the leaf cell's REST space
    // (equivalent to translating the triangle by +offset; t is preserved, so
    // the caller's `o + t·d` lands on the DISPLACED surface, and a secondary
    // ray re-entering here is shifted again — consistent by construction).
    // Shadows the `ray` binding so the WHOLE tail — the MT test, the relief
    // march, the alpha cutout — sees one space. Sound against every
    // frustum/tmin claim because the build swept leaf AABBs by the
    // displacement bound (`grow_sway_sweep`); every ray type funnels through
    // here (the cutout argument verbatim), so the exact-zero verify gates
    // stay like-for-like. `Scene::sway` is None on unarmed/foliage-free
    // scenes and `offsets` are all-zero at the rest pose (headless paths) —
    // both take the untouched original ray.
    // GATEWAY MODE (the SAH default with sway armed) NEVER runs this head:
    // the shift happened ONCE at the gateway entry (the traversal arms
    // above), the ray is already in cell-rest space, and shifting again
    // here would double-displace — plus this per-test `tri_cell` load is
    // exactly the 20-69 MB random access the gateway restructure evicted
    // from the hot loop. The head stays LIVE for the alt builders
    // (lbvh/ploc/som keep the v0.2 per-tri-sweep + per-test-shift path).
    // `FR_SWAY_ABL=noshift` skips the shift at run time in either mode —
    // the cost probe. Unset = initialized-deref checks only.
    let shifted;
    let ray = match &scene.sway {
        Some(sw)
            if !crate::foliage::gateway_mode()
                && !crate::foliage::sway_abl().noshift
                && sw.tri_cell[tri as usize] != crate::foliage::STATIC_CELL =>
        {
            let s = sw.shear(sw.tri_cell[tri as usize]);
            if s.u != Vec3A::ZERO {
                if tri_probe() {
                    TRI_TESTS_SHIFTED.fetch_add(1, Ordering::Relaxed);
                }
                // The MT test itself never reads inv_d, but the shadowed
                // binding must still carry a consistent one for the tail
                // (relief is a leaf-material known-incompatible; keep it
                // honest anyway).
                shifted = shear_ray(ray, &s);
                &shifted
            } else {
                ray
            }
        }
        _ => ray,
    };
    let [i0, i1, i2] = scene.indices[tri as usize];
    let v0 = scene.positions[i0 as usize];
    let e1 = scene.positions[i1 as usize] - v0;
    let e2 = scene.positions[i2 as usize] - v0;
    let p = ray.d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-10 {
        return None;
    }
    let inv = 1.0 / det;
    let s = ray.o - v0;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = ray.d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t <= 0.0 {
        return None;
    }
    // Relief march (see `height_march`): the hit moves inward along the ray
    // or is rejected outright. Runs BEFORE the alpha cutout so the cutout
    // tests the DISPLACED uv — the surface the ray actually touched. Every
    // ray type funnels through here (the cutout argument verbatim), so the
    // exact-zero verify gates stay like-for-like; a below/interior hit can
    // land EARLIER than the plane t, which is sound only because the BVH's
    // swept AABBs contain the whole prism (see the build-time sweep).
    let (mut t, mut u, mut v) = (t, u, v);
    if scene.any_height && height_on() {
        let depth = tri_height_depth(scene, tri);
        if depth > 0.0 {
            match height_march(scene, tri, ray, e1, e2, t, u, v, depth) {
                Some(h) => (t, u, v) = h,
                None => return None,
            }
        }
    }
    // Alpha cutout: a candidate on an alpha-masked textured triangle is
    // REJECTED (not accepted-and-continued) where the mask says transparent —
    // `intersect_from` keeps tmax unshrunk and walks on to the true nearest
    // opaque hit; `occluded_from` keeps searching. Every ray type (hybrid,
    // verify reference, shadow, AO, hemi, shaft) funnels through here, so the
    // exact-zero gates stay like-for-like. The frustum bound queries still
    // treat masked triangles as solid AABBs — sound, because rejection only
    // removes hits: the true nearest hit moves FARTHER, so a conservative
    // lower bound stays a lower bound (inherited tmin never overshoots, and
    // hemi cells become at most less provably-empty, never falsely empty).
    if scene.any_alpha {
        if let crate::scene::MatKind::Textured { tex } =
            scene.materials[scene.tri_mat[tri as usize] as usize].kind
        {
            let tx = &scene.textures[tex as usize];
            if tx.alpha_masked {
                let uv = scene.tri_uv(tri, u, v);
                if tx.alpha_nearest(uv.x, uv.y) < 128 {
                    return None;
                }
            }
        }
    }
    Some((t, u, v))
}
