use crate::bvh::{Aabb, Bvh};
use glam::Vec3A;

/// A frustum with its apex at `origin` and up to 5 side planes through the
/// apex (so a plane is fully described by its inward-pointing normal; zero
/// normals never cull). Screen tiles use 4 planes; hemisphere cells 1 or 3;
/// light shafts 4 + a tangent-plane clip.
pub struct TileFrustum {
    pub origin: Vec3A,
    normals: [Vec3A; 5],
    /// Per-plane outward relaxation in world units (zero on every non-shared
    /// path). A padded plane admits every point within `pads[k]` of the true
    /// half-space — how a frustum built at one apex stays conservative for a
    /// group of nearby apexes (hemi sharing): erring inclusive costs
    /// efficiency, never correctness.
    pads: [f32; 5],
    /// Planes actually in use — the tail of `normals` is zero and a zero
    /// normal never culls, so skipping it is pure savings in the plane-test
    /// loop (the hottest loop in the tracer).
    n_planes: usize,
}

impl TileFrustum {
    /// `corners` are the (normalized) ray directions through the tile's
    /// continuous image-plane corners, in perimeter order.
    pub fn new(origin: Vec3A, corners: [Vec3A; 4]) -> Self {
        Self::with_clip(origin, corners, Vec3A::ZERO)
    }

    /// A 4-corner cone additionally clipped by the through-apex plane with
    /// inward normal `clip` (the light-shaft case: `clip` = the surface
    /// normal, excluding the own surface exactly — sound because only
    /// samples with `wi·n > 0` ever consult the shaft, and their segments
    /// lie wholly in the clip half-space).
    pub fn with_clip(origin: Vec3A, corners: [Vec3A; 4], clip: Vec3A) -> Self {
        let center = corners[0] + corners[1] + corners[2] + corners[3];
        let mut normals = [Vec3A::ZERO; 5];
        for i in 0..4 {
            let n = corners[i].cross(corners[(i + 1) % 4]);
            let n = if n.dot(center) < 0.0 { -n } else { n };
            // Degenerate (near-parallel corner rays) → zero normal → the plane
            // test below never culls, which is the safe direction.
            normals[i] = n.normalize_or_zero();
        }
        normals[4] = clip;
        let n_planes = if clip == Vec3A::ZERO { 4 } else { 5 };
        TileFrustum { origin, normals, pads: [0.0; 5], n_planes }
    }

    /// Hemisphere "root frustum" for the bounce integrator: one plane through
    /// the apex with inward normal `n` (the surface tangent plane), the rest
    /// zero (which never cull). The hemisphere analog of the whole-screen
    /// root — the apex is a shading point, not the camera.
    pub fn half_space(origin: Vec3A, n: Vec3A) -> Self {
        let mut normals = [Vec3A::ZERO; 5];
        normals[0] = n;
        TileFrustum { origin, normals, pads: [0.0; 5], n_planes: 1 }
    }

    /// `half_space` relaxed outward by `pad` — the shared-hemisphere root.
    /// `pad` must be the group's measured out-of-plane apex spread η (fp hit
    /// points are NOT exactly coplanar), and the caller must keep η ≪ eps or
    /// the own surface (at −eps below the plane) re-enters the claims — the
    /// blocked-everywhere collapse. Enforced by the group qualifier, not here.
    pub fn half_space_padded(origin: Vec3A, n: Vec3A, pad: f32) -> Self {
        let mut f = Self::half_space(origin, n);
        f.pads[0] = pad;
        f
    }

    /// Spherical-triangle cell (a hemisphere-quadtree tile): the cell's edges
    /// are great-circle arcs, so its boundary is exactly three planes through
    /// the apex + zero planes. Orientation is fixed by the vertex sum, the
    /// same trick as `new` — a positive combination of the vertices lies
    /// strictly inside any sub-hemisphere cell, so the sign fix always picks
    /// the inward normal.
    pub fn tri_cell(origin: Vec3A, a: Vec3A, b: Vec3A, c: Vec3A) -> Self {
        let center = a + b + c;
        let mut normals = [Vec3A::ZERO; 5];
        for (i, (u, v)) in [(a, b), (b, c), (c, a)].into_iter().enumerate() {
            let n = u.cross(v);
            let n = if n.dot(center) < 0.0 { -n } else { n };
            normals[i] = n.normalize_or_zero();
        }
        TileFrustum { origin, normals, pads: [0.0; 5], n_planes: 3 }
    }

    /// `tri_cell` for a shared-apex group: every plane k is relaxed by
    /// `pad_k = d_par·|n_k − (n_k·axis)axis| + eta·|n_k·axis|`, where the
    /// group's apexes lie within `d_par` of this apex along the plane through
    /// it with normal `axis` and within `eta` across it (both MEASURED from
    /// the fp apexes — coplanarity is never assumed). Any ray from any group
    /// apex through a direction inside the unpadded cell then stays inside
    /// the padded cone: its points x satisfy n_k·(x − origin) ≥ −pad_k
    /// (|n_k·Δ| ≤ pad_k for every group displacement Δ; n_k·d ≥ −(sample fp),
    /// absorbed by the plane test's distance-growing slack).
    pub fn tri_cell_padded(
        origin: Vec3A,
        a: Vec3A,
        b: Vec3A,
        c: Vec3A,
        d_par: f32,
        eta: f32,
        axis: Vec3A,
    ) -> Self {
        let mut f = Self::tri_cell(origin, a, b, c);
        for i in 0..3 {
            let along = f.normals[i].dot(axis);
            let in_plane = (f.normals[i] - along * axis).length();
            f.pads[i] = d_par * in_plane + eta * along.abs();
        }
        f
    }

    /// Inclusive direction-containment against the active planes (`slack`
    /// in the units of the unit normals). Exposed so `sphcell::self_test`
    /// validates samples against the REAL cell construction the integrator
    /// queries with, not a hand-rebuilt copy of it.
    pub fn contains_dir(&self, d: Vec3A, slack: f32) -> bool {
        self.normals[..self.n_planes].iter().all(|n| d.dot(*n) >= -slack)
    }

    /// Active (normal, pad) pairs, for consumers that test many boxes per
    /// plane instead of many planes per box (ftree's 8-slot lane loop).
    #[inline]
    pub(crate) fn for_planes(&self, mut f: impl FnMut(Vec3A, f32)) {
        for (n, pad) in self.normals[..self.n_planes].iter().zip(&self.pads) {
            f(*n, *pad);
        }
    }

    /// Conservative: true only if the box is fully outside some side plane.
    /// False positives (box outside but not past a single plane) cost
    /// efficiency, never correctness.
    #[inline]
    pub(crate) fn aabb_outside(&self, aabb: &Aabb) -> bool {
        for (n, pad) in self.normals[..self.n_planes].iter().zip(&self.pads) {
            // positive vertex: the box corner farthest along the plane normal
            let pv = Vec3A::select(n.cmpge(Vec3A::ZERO), aabb.max, aabb.min);
            let rel = pv - self.origin;
            // eps pushes the plane outward slightly (fp slack, safe direction);
            // pad relaxes it further for shared-apex frustums (zero elsewhere)
            let eps = 1e-5 * (1.0 + rel.abs().max_element());
            if n.dot(rel) < -eps - pad {
                return true;
            }
        }
        false
    }
}

// THE RANGE IS FRUSTUM-AWARE (`frustum_aabb_dist`), and here is exactly what
// that is and is not worth — measured, so nobody re-derives it either way.
//
// The plane test culls with all four planes; `point_aabb_dist` ranges with none
// of them. That is exact for a small box in front of the apex and VACUOUS for a
// box STRADDLING the cone: it survives the cull (it really does intersect)
// while its nearest point lies outside, so the tile is ranged against geometry
// it cannot see. The standard ground quad is the pathological limit — one flat
// slab intersecting nearly every tile frustum whose nearest point sits directly
// under the camera, so every ground-facing tile was told "camera height" at
// every depth, and no subdivision escapes it (a narrower sub-frustum still
// intersects that slab). The counters agreed: `blocked` was 4032 of 4033 splits
// on the procedural scene and 3099 of 3101 on --stress — ~99.9% of all bound
// queries making NO distance progress, t_start averaging 0.42 of the true hit.
//
// `frustum_aabb_dist` fixes it with math instead of geometry, and it works:
// --stress blocked 99.4% -> 53.9%, t_start 0.55 -> 0.73 of the true hit, one
// ftree probe's bound 34.3 -> 96.5.
//
// WHAT IT DOES NOT BUY — do not expect ray-path wins from a tighter bound:
//     ray nodes (spp=4, --stress) 27,487,294,554 -> 27,487,141,775  (-0.00056%)
// 152 thousand of 27.5 BILLION node visits. An spp 1/4/16 sweep found no trend
// (+-1.7% scatter around zero), which is the test that would have caught a
// per-ray win amortizing against this function's fixed cost. The structural
// reason is scene- and hardware-independent: `slab_t` culls a node only when it
// lies ENTIRELY before tmin, and nodes between the camera and the first hit are
// rare almost by definition — that region is mostly empty, which is why the hit
// is where it is. Equivalently, forcing the GPU leaf TMin to 0 costs 1.1-1.7%
// on Arc and nothing measurable on NVIDIA, and AMD's re-origining note already
// recorded it as free and worth nothing. The quadtree's ray-path payoff is the
// EMPTY-SPACE PROOF (tiles proven empty tracing no rays), not the bound.
//
// THE OTHER REPAIR, MEASURED AND REJECTED: subdividing the ground quad into a
// grid fixes the same problem with geometry. Bound quality soared even higher
// (--stress blocked -> 37.9%, t_start -> 0.95) and it cost time everywhere —
// CPU default 39.5 -> 42.4 ms, GPU B70 default 1.571 -> 1.802, 4090 1.299 ->
// 2.023 — because more, smaller boxes means more frustum-tree nodes. Geometry
// is the wrong lever; the query was.

/// R&D lever (FR_RANGE=0 restores the pre-fix point-to-box ranging).
fn range_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FR_RANGE").map_or(true, |v| v != "0"))
}

#[inline(always)]
fn point_aabb_dist(p: Vec3A, aabb: &Aabb) -> f32 {
    (p.clamp(aabb.min, aabb.max) - p).length()
}

// WHY THE RANGE IS DELIBERATELY FRUSTUM-BLIND (two measured dead ends — do
// not re-derive them).
//
// The plane test culls with all four planes; `point_aabb_dist` above ranges
// with none of them. That looks like an obvious oversight: a box STRADDLING the
// cone survives the cull (it really does intersect) while its nearest point
// lies outside, so the tile is ranged against geometry it cannot see. The
// standard ground quad is the pathological limit — one flat slab intersecting
// nearly every tile frustum whose nearest point is directly under the camera,
// so every ground-facing tile is told "camera height" at every depth, and no
// subdivision escapes it. The counters agreed: `blocked` was 4032/4033 splits
// on the procedural scene and 3099/3101 on --stress (~99.9% of bound queries
// making NO distance progress), with t_start averaging 0.42 of the true hit.
//
// BOTH repairs were built and measured, and BOTH lost:
//   1. Subdivide the ground into a grid (fixes it with geometry). Bound quality
//      soared — stress blocked 99.4% -> 37.9%, t_start 0.55 -> 0.95 of the true
//      hit — and it cost time everywhere: CPU default 39.5 -> 42.4 ms, GPU B70
//      default 1.571 -> 1.802, 4090 default 1.299 -> 2.023. More, smaller boxes
//      means more frustum-tree nodes, and that outweighs the better bound.
//   2. Clip the box against the four apex planes and range the remainder
//      (fixes it with math — no extra triangles, no tree growth). Sound (every
//      zero-gate stayed 0) and it worked: stress blocked 99.4% -> 53.9%,
//      t_start 0.55 -> 0.73, one probe's bound 34.3 -> 96.5. RAY NODES MOVED BY
//      0.0006% (default: bit-identical), and CPU stress went 41.2 -> 43.2 ms.
//
// The second result is the general one, and it explains the first: `slab_t`
// DOES prune on tmin, but a front-to-back traversal with tmax shrinking on the
// first hit visits few nodes before that hit anyway. Raising tmin only culls
// nodes whose slab exit precedes the hit — exactly the ones the walk was
// already disposing of cheaply. **The inherited distance bound removes work
// that is already nearly free**, which is equally why forcing the GPU's leaf
// TMin to 0 costs only 1.1-1.7% on Arc and nothing measurable on NVIDIA, and
// why AMD's re-origining note records it as free and worth nothing. The
// quadtree's payoff is the EMPTY-SPACE PROOF (sky tiles tracing no rays), not
// the distance bound. Tighten the range only if a measurement shows the ray
// path has become tmin-sensitive.

/// Distance from the apex to a conservative SUPERSET of (box ∩ frustum) — the
/// frustum-aware range. Every plane passes through the apex, so in
/// apex-relative coordinates each is `dot(n, v) >= -s`; for axis `a` a point can
/// only satisfy that if `n_a·v_a + (the most the other two axes can contribute)
/// >= -s`, which bounds `v_a` from one side. Clipping to that bound removes only
/// points that provably fail the plane test, so the clipped box still CONTAINS
/// box ∩ frustum and its distance is still a lower bound on the true nearest
/// hit — the `nearest_geometry_distance` soundness contract. Planes apply in
/// sequence (tighter, still sound); an empty result falls back to the unclipped
/// distance rather than culling, so this can never become the false-sky class.
///
/// `pub(crate)` and shared with ftree.rs ON PURPOSE: slot boxes ARE binary node
/// AABBs, and `ftree::self_test` pins the two trees' bounds BIT-IDENTICAL, so
/// they must run the same expression on the same values — not two ports of it.
#[inline]
pub(crate) fn frustum_aabb_dist(f: &TileFrustum, aabb: &Aabb) -> f32 {
    let o = f.origin;
    if !range_on() {
        return point_aabb_dist(o, aabb); // FR_RANGE=0: the pre-fix ranging
    }
    let mut lo = [aabb.min.x - o.x, aabb.min.y - o.y, aabb.min.z - o.z];
    let mut hi = [aabb.max.x - o.x, aabb.max.y - o.y, aabb.max.z - o.z];
    for (nv, pad) in f.normals[..f.n_planes].iter().zip(&f.pads) {
        let n = [nv.x, nv.y, nv.z];
        if n[0] == 0.0 && n[1] == 0.0 && n[2] == 0.0 {
            continue; // degenerate plane: never culls, so it never clips either
        }
        // The same outward slack `aabb_outside` grants, so a point this clip
        // discards is one the plane test would also have called outside.
        let ext = lo.iter().chain(hi.iter()).fold(0.0f32, |m, v| m.max(v.abs()));
        let s = 1e-5 * (1.0 + ext) + *pad;
        let m = [
            (n[0] * lo[0]).max(n[0] * hi[0]),
            (n[1] * lo[1]).max(n[1] * hi[1]),
            (n[2] * lo[2]).max(n[2] * hi[2]),
        ];
        let total = m[0] + m[1] + m[2];
        for a in 0..3 {
            if n[a] == 0.0 {
                continue;
            }
            let bound = (-s - (total - m[a])) / n[a]; // `total - m[a]` = the others' best
            if n[a] > 0.0 {
                if bound > lo[a] {
                    lo[a] = bound;
                }
            } else if bound < hi[a] {
                hi[a] = bound;
            }
            if lo[a] > hi[a] {
                // box ∩ frustum is empty, so nothing in this box can be the
                // nearest visible hit: +INF, i.e. the caller's `d >= best`
                // drops it. This MUST NOT fall back to the unclipped distance
                // instead — that is sound as a lower bound but destroys
                // MONOTONICITY (a child returning a weak small bound where its
                // parent returned a strong large one), and monotonicity is what
                // makes `d >= best` subtree pruning order-independent. The two
                // trees visit in different orders, so `ftree::self_test` caught
                // it immediately: wide 81.55 vs binary 82.58 on probe 456.
                // Emptiness is conservative here because the clip carries the
                // same outward slack `aabb_outside` does, so any point of
                // box ∩ frustum survives every clip step.
                return f32::INFINITY;
            }
        }
    }
    let c = [
        if lo[0] > 0.0 { lo[0] } else if hi[0] < 0.0 { hi[0] } else { 0.0 },
        if lo[1] > 0.0 { lo[1] } else if hi[1] < 0.0 { hi[1] } else { 0.0 },
        if lo[2] > 0.0 { lo[2] } else if hi[2] < 0.0 { hi[2] } else { 0.0 },
    ];
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
}

#[inline(always)]
fn point_aabb_max_dist(p: Vec3A, aabb: &Aabb) -> f32 {
    (p - aabb.min).abs().max((p - aabb.max).abs()).length()
}

/// The conservative nearest-distance query — the heart of the frustracer.
///
/// Returns the smallest Euclidean distance from the camera origin at which any
/// geometry inside the frustum *could* start, beyond the inherited `t_start`,
/// or `None` if nothing intersects the frustum (→ sky tile).
///
/// Correctness invariants:
/// - Distances are Euclidean from the shared ray origin; with normalized ray
///   directions this is exactly the ray parameter t, so the result is a valid
///   `tmin` for every pixel ray inside the frustum.
/// - The region proven empty by an ancestor tile is frustum ∩ ball(origin,
///   t_start) — a *spherical* bound. A node is skipped as "already proven
///   empty" only when it lies entirely inside that ball
///   (`max_dist <= t_start`); candidates are clamped up to `t_start` so the
///   result is monotonic down the quadtree. A planar near clip here would
///   over-cull near frustum corners and let child rays start past geometry.
///
/// `roots` is the tile's inherited node cut — the parent's surviving BVH
/// nodes from `refine_cut` (the whole-screen root passes `[0]`). The primary
/// tile path calls this with `t_limit = INFINITY` (via `ftree::Accel`, which
/// dispatches wide vs binary); the docs elsewhere still say
/// "nearest_geometry_distance" for that unclamped shape.
///
/// The bound is clamped to a max distance of interest: only
/// geometry strictly closer than `t_limit` is reported. `None` means
/// "frustum ∩ (ball(t_limit) \ ball(t_start)) is empty" — it says NOTHING
/// beyond `t_limit`, so a `None` may only be consumed as "open within
/// t_limit" (AO, light shafts), never as sky. A `Some(t)` is still a valid
/// lower bound for ALL frustum geometry beyond t_start (nodes are pruned only
/// at `d >= best >= t`), so it remains a sound child tc. The `d >= best`
/// prune in `visit` does the early-out — seeding `best = t_limit` is the
/// entire mechanism.
pub fn nearest_geometry_distance_within(
    bvh: &Bvh,
    f: &TileFrustum,
    t_start: f32,
    t_limit: f32,
    roots: &[u32],
    visits: &mut u64,
) -> Option<f32> {
    if bvh.is_empty() {
        return None;
    }
    let mut best = t_limit;
    for &r in roots {
        visit(bvh, f, t_start, r, &mut best, visits);
    }
    (best < t_limit).then_some(best)
}

/// Budget for a tile's inherited BVH node cut. Correctness never depends on
/// it: an exhausted budget emits internal nodes coarsely (MAX_CUT = 1 would
/// degenerate to re-descending from the root, exactly the pre-cut behavior).
pub const MAX_CUT: usize = 64;

/// The advance/slack rule shared by every inherited-tmin chain (screen tiles,
/// hemisphere cells, shaft subrects): a bound `t` advances the inherited
/// `t_start` only past fp slack (relative 1e-4, floored by the scene-scaled
/// `eps`), and an advanced claim is shaved by the same relative slack so the
/// strict `t > tmin` hit acceptance keeps room. Monotone — the result never
/// drops below `t_start`. Returns `(advanced, tc)`; blocked (unadvanced)
/// consumers must still subdivide.
#[inline]
pub fn advance_tc(t: f32, t_start: f32, eps: f32) -> (bool, f32) {
    let advanced = t > t_start + (t_start * 1e-4).max(eps);
    let tc = if advanced { (t * (1.0 - 1e-4)).max(t_start) } else { t_start };
    (advanced, tc)
}

/// Refine a parent tile's node cut into this tile's cut.
///
/// Keeps every node that could contain geometry visible to any ray of this
/// tile or its descendants. A node is dropped only when
/// - it is fully outside this tile's frustum (descendant frustums and all
///   jittered leaf rays are contained in it), or
/// - `max_dist(origin, aabb) <= t_ball`: every point of the node is at
///   Euclidean distance <= t_ball, tmin only grows down the quadtree, and ray
///   hit acceptance is strictly `t > tmin` — no descendant ray can hit it, or
/// - `dist(origin, aabb) >= t_far`: every point of the node is at distance
///   >= t_far. Sound ONLY under the clamped-consumer contract: every bound
///   query on this cut is `_within(t_limit <= t_far)` and every ray on it
///   uses `tmax <= t_far` (hit acceptance is strictly `t < tmax`). The
///   primary path passes `t_far = INFINITY`, which never drops; the
///   hemisphere AO path passes its ao_radius.
///
/// NO distance-to-best pruning here, ever. The `d >= best` prune belongs to
/// the bound query only: a far node can still be the nearest thing inside a
/// descendant's smaller frustum, and pruning it from the cut would surface as
/// false sky. The cut handed to children must only ever come from here.
///
/// Iterative work stack with the invariant `out_len + work_len <= N`, so a
/// surviving node always has a slot: internal nodes split into their two
/// children only while the budget allows, otherwise they are emitted coarsely
/// (never dropped). Generic over the cut capacity so each caller owns its
/// budget constant (screen tiles MAX_CUT, hemisphere HEMI_CUT, shafts
/// SHAFT_CUT — all 64 today; larger hemisphere budgets were measured slower,
/// see hemi.rs). Overflow only coarsens a cut, it never affects correctness.
pub fn refine_cut<const N: usize>(
    bvh: &Bvh,
    f: &TileFrustum,
    t_ball: f32,
    t_far: f32,
    parent_cut: &[u32],
    out: &mut [u32; N],
    visits: &mut u64,
    overflows: &mut u64,
) -> usize {
    if bvh.tri_idx.is_empty() {
        return 0; // the n == 0 root has count == 0 and would parse as internal
    }
    let mut work = [0u32; N];
    debug_assert!(parent_cut.len() <= N);
    let mut wlen = parent_cut.len().min(N);
    work[..wlen].copy_from_slice(&parent_cut[..wlen]);
    let mut olen = 0usize;
    // The primary path passes t_far = INFINITY; skip the per-node distance
    // (a sqrt) when the comparison is statically dead.
    let has_far = t_far.is_finite();
    while wlen > 0 {
        wlen -= 1;
        let idx = work[wlen];
        *visits += 1;
        let node = &bvh.nodes[idx as usize];
        if f.aabb_outside(&node.aabb) {
            continue;
        }
        if point_aabb_max_dist(f.origin, &node.aabb) <= t_ball {
            continue; // entirely inside the proven-empty ball
        }
        if has_far && point_aabb_dist(f.origin, &node.aabb) >= t_far {
            continue; // entirely beyond the consumers' tmax clamp
        }
        if node.count == 0 && olen + wlen + 2 <= N {
            work[wlen] = node.left_first;
            work[wlen + 1] = node.left_first + 1;
            wlen += 2;
        } else {
            if node.count == 0 {
                *overflows += 1;
            }
            out[olen] = idx;
            olen += 1;
        }
    }
    olen
}

fn visit(bvh: &Bvh, f: &TileFrustum, t_start: f32, idx: u32, best: &mut f32, visits: &mut u64) {
    *visits += 1;
    let node = &bvh.nodes[idx as usize];
    if f.aabb_outside(&node.aabb) {
        return;
    }
    if point_aabb_max_dist(f.origin, &node.aabb) <= t_start {
        return; // entirely inside the proven-empty ball
    }
    let d = frustum_aabb_dist(f, &node.aabb).max(t_start);
    if d >= *best {
        return; // can't improve (also the early-out once best hits the t_start floor)
    }
    if node.count > 0 {
        // Leaf: the box distance is a conservative lower bound for its triangles.
        *best = d;
        return;
    }
    let l = node.left_first;
    let dl = point_aabb_dist(f.origin, &bvh.nodes[l as usize].aabb);
    let dr = point_aabb_dist(f.origin, &bvh.nodes[l as usize + 1].aabb);
    if dl <= dr {
        visit(bvh, f, t_start, l, best, visits);
        visit(bvh, f, t_start, l + 1, best, visits);
    } else {
        visit(bvh, f, t_start, l + 1, best, visits);
        visit(bvh, f, t_start, l, best, visits);
    }
}
