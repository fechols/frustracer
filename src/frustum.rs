use crate::bvh::{Aabb, Bvh};
use glam::Vec3A;

/// A screen-tile frustum: apex at the camera origin, 4 side planes through the
/// origin (so a plane is fully described by its inward-pointing normal).
pub struct TileFrustum {
    pub origin: Vec3A,
    normals: [Vec3A; 4],
}

impl TileFrustum {
    /// `corners` are the (normalized) ray directions through the tile's
    /// continuous image-plane corners, in perimeter order.
    pub fn new(origin: Vec3A, corners: [Vec3A; 4]) -> Self {
        let center = corners[0] + corners[1] + corners[2] + corners[3];
        let mut normals = [Vec3A::ZERO; 4];
        for i in 0..4 {
            let n = corners[i].cross(corners[(i + 1) % 4]);
            let n = if n.dot(center) < 0.0 { -n } else { n };
            // Degenerate (near-parallel corner rays) → zero normal → the plane
            // test below never culls, which is the safe direction.
            normals[i] = n.normalize_or_zero();
        }
        TileFrustum { origin, normals }
    }

    /// Conservative direction containment: true only if the conic hull of
    /// `dirs` provably lies inside this frustum's cone — equivalently, every
    /// point `origin + (positive combination of dirs)` is inside the frustum
    /// (all planes pass through `origin`). Used by the temporal cache: the
    /// caller reduces "this ray segment is inside the old frustum" to a pure
    /// direction test (see `temporal::lookup` for the decomposition).
    ///
    /// `dirs` must be normalized: the strict 1e-5 margin is an *angular*
    /// margin, and it is load-bearing — a tile shares boundary planes with its
    /// parent and with same-position tiles, where these dots are exact zeros
    /// that must reject; it also absorbs the ~1e-7 error of normal
    /// construction and dir normalization.
    ///
    /// A zero normal REJECTS here — the opposite polarity to `aabb_outside`,
    /// where a zero normal must never cull. Culling errs toward "intersecting";
    /// containment errs toward "not contained". Don't "fix" either direction.
    #[inline]
    pub fn contains_dirs(&self, dirs: &[Vec3A]) -> bool {
        for n in &self.normals {
            if *n == Vec3A::ZERO {
                return false;
            }
            for d in dirs {
                if n.dot(*d) < 1e-5 {
                    return false;
                }
            }
        }
        true
    }

    /// Conservative: true only if the box is fully outside some side plane.
    /// False positives (box outside but not past a single plane) cost
    /// efficiency, never correctness.
    #[inline]
    fn aabb_outside(&self, aabb: &Aabb) -> bool {
        for n in &self.normals {
            // positive vertex: the box corner farthest along the plane normal
            let pv = Vec3A::select(n.cmpge(Vec3A::ZERO), aabb.max, aabb.min);
            let rel = pv - self.origin;
            // eps pushes the plane outward slightly (fp slack, safe direction)
            let eps = 1e-5 * (1.0 + rel.abs().max_element());
            if n.dot(rel) < -eps {
                return true;
            }
        }
        false
    }
}

#[inline(always)]
fn point_aabb_dist(p: Vec3A, aabb: &Aabb) -> f32 {
    (p.clamp(aabb.min, aabb.max) - p).length()
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
/// nodes from `refine_cut` (the whole-screen root passes `[0]`).
pub fn nearest_geometry_distance(
    bvh: &Bvh,
    f: &TileFrustum,
    t_start: f32,
    roots: &[u32],
    visits: &mut u64,
) -> Option<f32> {
    let mut best = f32::INFINITY;
    for &r in roots {
        visit(bvh, f, t_start, r, &mut best, visits);
    }
    best.is_finite().then_some(best)
}

/// Budget for a tile's inherited BVH node cut. Correctness never depends on
/// it: an exhausted budget emits internal nodes coarsely (MAX_CUT = 1 would
/// degenerate to re-descending from the root, exactly the pre-cut behavior).
pub const MAX_CUT: usize = 64;

/// Refine a parent tile's node cut into this tile's cut.
///
/// Keeps every node that could contain geometry visible to any ray of this
/// tile or its descendants. A node is dropped only when
/// - it is fully outside this tile's frustum (descendant frustums and all
///   jittered leaf rays are contained in it), or
/// - `max_dist(origin, aabb) <= t_ball`: every point of the node is at
///   Euclidean distance <= t_ball, tmin only grows down the quadtree, and ray
///   hit acceptance is strictly `t > tmin` — no descendant ray can hit it.
///
/// NO distance-to-best pruning here, ever. The `d >= best` prune belongs to
/// the bound query only: a far node can still be the nearest thing inside a
/// descendant's smaller frustum, and pruning it from the cut would surface as
/// false sky. The cut handed to children must only ever come from here.
///
/// Iterative work stack with the invariant `out_len + work_len <= MAX_CUT`,
/// so a surviving node always has a slot: internal nodes split into their two
/// children only while the budget allows, otherwise they are emitted coarsely
/// (never dropped).
pub fn refine_cut(
    bvh: &Bvh,
    f: &TileFrustum,
    t_ball: f32,
    parent_cut: &[u32],
    out: &mut [u32; MAX_CUT],
    visits: &mut u64,
    overflows: &mut u64,
) -> usize {
    if bvh.tri_idx.is_empty() {
        return 0; // the n == 0 root has count == 0 and would parse as internal
    }
    let mut work = [0u32; MAX_CUT];
    debug_assert!(parent_cut.len() <= MAX_CUT);
    let mut wlen = parent_cut.len().min(MAX_CUT);
    work[..wlen].copy_from_slice(&parent_cut[..wlen]);
    let mut olen = 0usize;
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
        if node.count == 0 && olen + wlen + 2 <= MAX_CUT {
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
    let d = point_aabb_dist(f.origin, &node.aabb).max(t_start);
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
