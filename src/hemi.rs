//! Hemisphere frustum-bounce integrator: the incoming-light integral at a
//! shading point, dispatched by the same depth-first frustum recursion as the
//! screen quadtree. Cells proven empty resolve analytically (exact projected
//! solid angle — for GI, sky radiance × PSA with sun-glow refinement — zero
//! rays, zero variance); unresolved cells reaching the query cutoff distribute
//! one stratified ray per sub-cell, seeded from the inherited cut with the
//! inherited tmin — frustums accelerate, rays resolve, exactly like the
//! primary path.
//!
//! Soundness mirrors the primary path with the apex at the shading point:
//! - The apex is `hit + n·eps` (the standard secondary-ray origin). The root
//!   `t_start` is 0, NOT eps — asserting ball(o, eps) empty is false at
//!   concave corners (the false-sky bug shape). The own surface needs no
//!   epsilon here at all: it lies in the plane n·(x−o) = −eps, strictly below
//!   the tangent half-space, so the root plane excludes it geometrically.
//! - The root cut is the BVH root `[0]`. A primary tile's cut is INVALID here
//!   (it was culled against the camera apex and its ball drops were
//!   ball(camera, tc)). The primary tmin never appears — the invariant
//!   "secondary rays never see the tile's tmin" holds by construction; the
//!   hemisphere has its own apex-relative tmin chain.
//! - Cells are spherical triangles (`sphcell`); midpoint children exactly
//!   partition the parent, so inherited (tc, cut) is sound by the same
//!   containment argument as pixel quadrants. Leaf sample directions are
//!   strictly inside their cell (fp slack ≪ the plane test's inclusive eps).
//! - Blocked cells (no distance progress — an AABB straddling the tangent
//!   plane at the horizon) still subdivide, never stop; the leaf rays eat
//!   the residual.
//! - AO mode clamps every query to `t_limit` (ao_radius): `None` means "open
//!   within t_limit" and is consumed only as that — never as sky. GI mode
//!   uses t_limit = ∞, where `None` genuinely means sky.

use crate::bvh::{Bvh, Ray};
use crate::frustum::{self, TileFrustum};
use crate::scene::Scene;
use crate::shade;
use crate::sphcell;
use crate::stats::LocalStats;
use glam::Vec3A;
use std::f32::consts::PI;

/// Hemisphere node-cut budget. Same as the screen tiles' MAX_CUT: bigger
/// budgets represent dense neighborhoods more precisely but make every query
/// and ray pay per-root costs — measured slower both ways. Correctness never
/// depends on the value (overflow emits coarsely, never drops).
const HEMI_CUT: usize = 64;

/// Query-tree cutoff, the hemisphere analog of LEAF_TILE: cells this many
/// levels above the ray budget stop querying and distribute one stratified
/// ray per sub-cell instead. One bound query then amortizes over 4^LEAF_LEVELS
/// rays — per-ray queries were measured to cost far more than they saved
/// (occlusion rays are ~10 node visits; a bound query on a dense cut is more).
const LEAF_LEVELS: u32 = 1;

/// Shading quality for radiance carried back by GI leaf rays (the depth-1
/// policy): one shadow sample, no AO, no further reflections or hemispheres —
/// same spirit as the specular bounce, and structurally recursion-free.
/// pub(crate): the `--check` GI reference must implement the SAME policy so
/// the A/B isolates integrator error from policy error.
pub(crate) const BOUNCE_Q: shade::Quality = shade::Quality {
    shadow_samples: 1,
    ao_samples: 0,
    reflections: false,
    fb: shade::FrustumBounce::OFF,
};

/// `--check` instrumentation: re-validates every claim the integrator makes
/// with reference rays — the hemisphere analogs of the false-sky and
/// tmin-overshoot gates. Passed as `Some` only by the check harness.
#[derive(Default)]
pub struct VerifyCounters {
    pub points: u64,
    /// Points where |accounted PSA − π| > 1e-3 (a cell escaped accounting:
    /// every cell must be empty, leaf, or subdivided).
    pub psa_violations: u64,
    /// Empty-cell claims contradicted by a reference ray through the cell.
    pub false_empty: u64,
    /// Leaf rays whose tmin skipped geometry a tmin=0 reference ray hits.
    pub tmin_overshoot: u64,
    /// Leaf rays where the cut-seeded traversal disagreed with the full tree.
    pub cut_miss: u64,
    pub max_psa_err: f32,
}

impl VerifyCounters {
    pub fn ok(&self) -> bool {
        self.psa_violations == 0
            && self.false_empty == 0
            && self.tmin_overshoot == 0
            && self.cut_miss == 0
    }

    /// Fold another probe's counters in — the `--check` probe sweeps run
    /// per-probe counters in parallel and reduce them sequentially.
    pub fn merge(&mut self, o: &VerifyCounters) {
        self.points += o.points;
        self.psa_violations += o.psa_violations;
        self.false_empty += o.false_empty;
        self.tmin_overshoot += o.tmin_overshoot;
        self.cut_miss += o.cut_miss;
        self.max_psa_err = self.max_psa_err.max(o.max_psa_err);
    }
}

#[derive(Default)]
struct Acc {
    /// Analytic mass from proven-empty cells. AO: PSA in .x. GI: ∫sky·cos dω.
    open: Vec3A,
    /// Monte Carlo mass from leaf rays. AO: Σ V·cosθ·Ω in .x.
    /// GI: Σ L(d)·cosθ·Ω.
    ray: Vec3A,
    /// Verify-only: PSA of every empty + leaf cell; must total π.
    accounted: f32,
}

/// GI mode state; `None` runs AO (pure visibility).
#[derive(Clone, Copy)]
struct Gi {
    sun: Vec3A,
    /// Shade depth of the point being integrated (bounce rays shade at +1).
    depth: u32,
}

struct Cx<'a> {
    scene: &'a Scene,
    bvh: &'a Bvh,
    o: Vec3A,
    n: Vec3A,
    max_depth: u32,
    t_limit: f32,
    gi: Option<Gi>,
}

/// Cosine-weighted hemisphere visibility within `t_limit` at apex `o` with
/// normal `n` — the frustum-dispatched replacement for the sampled AO loop.
/// `(t1, t2)` is a right-handed tangent frame (t1 × t2 = n). Unbiased for the
/// same integral the cosine-sampled open fraction estimates.
#[allow(clippy::too_many_arguments)]
pub fn ao(
    scene: &Scene,
    bvh: &Bvh,
    o: Vec3A,
    n: Vec3A,
    t1: Vec3A,
    t2: Vec3A,
    max_depth: u32,
    t_limit: f32,
    rng: &mut fastrand::Rng,
    verify: Option<&mut VerifyCounters>,
    ls: &mut LocalStats,
) -> f32 {
    let cx = Cx { scene, bvh, o, n, max_depth: max_depth.max(1), t_limit, gi: None };
    // No upper clamp: a leaf sample's V·cosθ·Ω can overshoot its cell's PSA,
    // so the sum can exceed π; truncating only that high tail would bias the
    // estimator low (the --check signed-mean gate exists to catch bias).
    // Every term is ≥ 0, so no lower clamp is needed either.
    integrate(&cx, t1, t2, rng, verify, ls).x / PI
}

/// Cosine-weighted incoming radiance over the hemisphere, divided by π — the
/// drop-in replacement for `AMBIENT · ao` in the renderer's convention
/// (Lambert's 1/π lives in the light terms, so a uniform sky of radiance L
/// yields exactly L). Empty cells integrate `sky()` analytically (with
/// sun-glow refinement); leaf-ray hits carry one bounce of surface radiance
/// shaded at the depth-1 policy (`BOUNCE_Q`).
#[allow(clippy::too_many_arguments)]
pub fn gi(
    scene: &Scene,
    bvh: &Bvh,
    o: Vec3A,
    n: Vec3A,
    t1: Vec3A,
    t2: Vec3A,
    max_depth: u32,
    sun: Vec3A,
    depth: u32,
    rng: &mut fastrand::Rng,
    verify: Option<&mut VerifyCounters>,
    ls: &mut LocalStats,
) -> Vec3A {
    let cx = Cx {
        scene,
        bvh,
        o,
        n,
        max_depth: max_depth.max(1),
        t_limit: f32::INFINITY,
        gi: Some(Gi { sun, depth }),
    };
    (integrate(&cx, t1, t2, rng, verify, ls) / PI).max(Vec3A::ZERO)
}

fn integrate(
    cx: &Cx,
    t1: Vec3A,
    t2: Vec3A,
    rng: &mut fastrand::Rng,
    mut verify: Option<&mut VerifyCounters>,
    ls: &mut LocalStats,
) -> Vec3A {
    ls.hemi_points += 1;
    let mut acc = Acc::default();
    // Root: the tangent half-space, queried over the whole BVH.
    let root = TileFrustum::half_space(cx.o, cx.n);
    ls.hemi_queries += 1;
    let bound = frustum::nearest_geometry_distance_within(
        cx.bvh,
        &root,
        0.0,
        cx.t_limit,
        &[0],
        &mut ls.hemi_nodes,
    );
    match bound {
        None => {
            // The whole hemisphere is open within t_limit — one query, done.
            ls.hemi_cells_empty += 1;
            if let Some(v) = verify.as_deref_mut() {
                acc.accounted = PI;
                for [a, b, c] in sphcell::octants(cx.n, t1, t2) {
                    check_empty(cx, [a, b, c], v, ls);
                }
            }
            match cx.gi {
                None => acc.open.x = PI,
                Some(g) => {
                    for [a, b, c] in sphcell::octants(cx.n, t1, t2) {
                        acc.open += sky_cell(cx.n, g.sun, a, b, c, 4);
                    }
                }
            }
        }
        Some(t) => {
            // Advance + refine exactly like tile_step, then recurse into the
            // 4 octants (depth 1; leaves land at depth == max_depth).
            let (_, tc) = frustum::advance_tc(t, 0.0, cx.scene.eps);
            let mut cut = [0u32; HEMI_CUT];
            let len = frustum::refine_cut(
                cx.bvh,
                &root,
                tc,
                cx.t_limit,
                &[0],
                &mut cut,
                &mut ls.hemi_nodes,
                &mut ls.cut_overflows,
            );
            debug_assert!(len > 0, "hemisphere root refine emptied a non-open cut");
            let child: &[u32] = if len > 0 { &cut[..len] } else { &[0] };
            for [a, b, c] in sphcell::octants(cx.n, t1, t2) {
                cell(cx, a, b, c, 1, tc, child, rng, &mut verify, ls, &mut acc);
            }
        }
    }
    if let Some(v) = verify {
        v.points += 1;
        let err = (acc.accounted - PI).abs();
        v.max_psa_err = v.max_psa_err.max(err);
        if err > 1e-3 {
            v.psa_violations += 1;
        }
    }
    acc.open + acc.ray
}

/// One spherical-triangle cell: bound query over the inherited cut, then
/// empty → analytic, query cutoff → stratified rays, else refine + split.
#[allow(clippy::too_many_arguments)]
fn cell(
    cx: &Cx,
    a: Vec3A,
    b: Vec3A,
    c: Vec3A,
    depth: u32,
    t_start: f32,
    cut_in: &[u32],
    rng: &mut fastrand::Rng,
    verify: &mut Option<&mut VerifyCounters>,
    ls: &mut LocalStats,
    acc: &mut Acc,
) {
    let f = TileFrustum::tri_cell(cx.o, a, b, c);
    ls.hemi_queries += 1;
    let bound = frustum::nearest_geometry_distance_within(
        cx.bvh,
        &f,
        t_start,
        cx.t_limit,
        cut_in,
        &mut ls.hemi_nodes,
    );
    let Some(t) = bound else {
        ls.hemi_cells_empty += 1;
        match cx.gi {
            None => acc.open.x += sphcell::psa(a, b, c, cx.n),
            // Refinement budget: enough extra levels to reach ~6° cells even
            // for an octant-sized empty cell near the sun glow.
            Some(g) => acc.open += sky_cell(cx.n, g.sun, a, b, c, 5u32.saturating_sub(depth)),
        }
        if let Some(v) = verify.as_deref_mut() {
            acc.accounted += sphcell::psa(a, b, c, cx.n);
            check_empty(cx, [a, b, c], v, ls);
        }
        return;
    };
    // Same advance/slack rule as tile_step; blocked (unadvanced) cells carry
    // t_start through and still subdivide.
    let (_, tc) = frustum::advance_tc(t, t_start, cx.scene.eps);
    if depth + LEAF_LEVELS >= cx.max_depth {
        // Query-leaf: no more bound queries below here — distribute one
        // stratified ray per sub-cell, all seeded from the inherited cut
        // with this cell's tc (the LEAF_TILE amortization, on the sphere).
        leaf_rays(cx, a, b, c, cx.max_depth.saturating_sub(depth), tc, cut_in, rng, verify, ls, acc);
        return;
    }
    let mut cut = [0u32; HEMI_CUT];
    let len = frustum::refine_cut(
        cx.bvh,
        &f,
        tc,
        cx.t_limit,
        cut_in,
        &mut cut,
        &mut ls.hemi_nodes,
        &mut ls.cut_overflows,
    );
    debug_assert!(len > 0, "refine_cut emptied a non-open hemisphere cell");
    let child: &[u32] = if len > 0 { &cut[..len] } else { cut_in };
    let (mab, mbc, mca) = sphcell::midpoints(a, b, c);
    for [ca, cb, cc] in [[a, mab, mca], [mab, b, mbc], [mca, mbc, c], [mab, mbc, mca]] {
        cell(cx, ca, cb, cc, depth + 1, tc, child, rng, verify, ls, acc);
    }
}

/// Distribute stratified rays over the 4^levels sub-cells of a query-leaf —
/// pure geometric subdivision, zero BVH queries. Every sub-cell is contained
/// in the query-leaf, so the inherited (tc, cut) covers every ray.
#[allow(clippy::too_many_arguments)]
fn leaf_rays(
    cx: &Cx,
    a: Vec3A,
    b: Vec3A,
    c: Vec3A,
    levels: u32,
    tc: f32,
    cut: &[u32],
    rng: &mut fastrand::Rng,
    verify: &mut Option<&mut VerifyCounters>,
    ls: &mut LocalStats,
    acc: &mut Acc,
) {
    if levels > 0 {
        let (mab, mbc, mca) = sphcell::midpoints(a, b, c);
        for [ca, cb, cc] in [[a, mab, mca], [mab, b, mbc], [mca, mbc, c], [mab, mbc, mca]] {
            leaf_rays(cx, ca, cb, cc, levels - 1, tc, cut, rng, verify, ls, acc);
        }
        return;
    }
    let d = sphcell::sample_tri(a, b, c, rng.f32(), rng.f32());
    let ray = Ray::new(cx.o, d);
    let weight = d.dot(cx.n).max(0.0) * sphcell::solid_angle(a, b, c);
    ls.hemi_leaf_rays += 1;
    ls.secondary_rays += 1;
    match cx.gi {
        None => {
            let occ =
                cx.bvh.occluded_multi(cx.scene, &ray, tc, cx.t_limit, cut, &mut ls.ray_nodes);
            if !occ {
                acc.ray.x += weight;
            }
            if let Some(v) = verify.as_deref_mut() {
                acc.accounted += sphcell::psa(a, b, c, cx.n);
                verify_leaf_ray(cx, &ray, tc, v, ls);
                if occ != cx.bvh.occluded(cx.scene, &ray, tc, cx.t_limit, &mut ls.ray_nodes) {
                    v.cut_miss += 1;
                }
            }
        }
        Some(g) => {
            let hit =
                cx.bvh
                    .intersect_multi(cx.scene, &ray, tc, f32::INFINITY, cut, &mut ls.ray_nodes);
            if let Some(v) = verify.as_deref_mut() {
                acc.accounted += sphcell::psa(a, b, c, cx.n);
                verify_leaf_ray(cx, &ray, tc, v, ls);
                let full =
                    cx.bvh.intersect(cx.scene, &ray, tc, f32::INFINITY, &mut ls.ray_nodes);
                let miss = match (&hit, &full) {
                    (Some(h), Some(f)) => (h.t - f.t).abs() > 1e-3 * f.t,
                    (None, None) => false,
                    _ => true,
                };
                if miss {
                    v.cut_miss += 1;
                }
            }
            let l = match hit {
                None => shade::sky(d, g.sun),
                Some(h) => shade::shade(
                    cx.scene,
                    cx.bvh,
                    &ray,
                    &h,
                    &BOUNCE_Q,
                    rng,
                    g.sun,
                    g.depth + 1,
                    ls,
                    None,
                ),
            };
            acc.ray += l * weight;
        }
    }
}

/// tmin soundness gate: a tmin=0 reference ray must not hit strictly inside
/// the claimed-empty ball.
fn verify_leaf_ray(cx: &Cx, ray: &Ray, tc: f32, v: &mut VerifyCounters, ls: &mut LocalStats) {
    if let Some(h) = cx.bvh.intersect(cx.scene, ray, 0.0, f32::INFINITY, &mut ls.ray_nodes) {
        if h.t < tc * (1.0 - 1e-3) {
            v.tmin_overshoot += 1;
        }
    }
}

/// Analytic sky over a proven-empty cell: centroid radiance × exact PSA,
/// with pure-math midpoint refinement near the sun-glow lobe (`dot^32`,
/// significant out to ~21°) — an empty parent proves all children empty, so
/// refinement costs sky() evaluations only, no BVH work.
fn sky_cell(n: Vec3A, sun: Vec3A, a: Vec3A, b: Vec3A, c: Vec3A, levels: u32) -> Vec3A {
    let cen = sphcell::centroid(a, b, c);
    if levels > 0 {
        // Angular radius of the cell around its centroid.
        let cos_r = cen.dot(a).min(cen.dot(b)).min(cen.dot(c)).clamp(-1.0, 1.0);
        // Refine while the cell is coarser than ~12° — the horizon→zenith
        // gradient is anti-correlated with the cosine weight, so a coarse
        // centroid systematically over-brightens (measured +1.2% at octant
        // granularity). Near the sun-glow lobe (cos 21.5° ≈ 0.93,
        // angle(cen, sun) < radius + lobe) refine further, to ~6°.
        let coarse = cos_r < 0.978;
        let near_glow = cos_r < 0.995 && {
            let sin_r = (1.0 - cos_r * cos_r).max(0.0).sqrt();
            cen.dot(sun) > cos_r * 0.93 - sin_r * 0.368
        };
        if coarse || near_glow {
            let (mab, mbc, mca) = sphcell::midpoints(a, b, c);
            let mut sum = Vec3A::ZERO;
            for [ca, cb, cc] in [[a, mab, mca], [mab, b, mbc], [mca, mbc, c], [mab, mbc, mca]] {
                sum += sky_cell(n, sun, ca, cb, cc, levels - 1);
            }
            return sum;
        }
    }
    shade::sky(cen, sun) * sphcell::psa(a, b, c, n)
}

/// Reference-ray re-validation of an empty-cell claim: directions strictly
/// inside the cell must all be unoccluded within t_limit (shaved for fp).
fn check_empty(cx: &Cx, [a, b, c]: [Vec3A; 3], v: &mut VerifyCounters, ls: &mut LocalStats) {
    const GRID: [[f32; 3]; 6] =
        [[1., 1., 1.], [6., 1., 1.], [1., 6., 1.], [1., 1., 6.], [3., 3., 1.], [1., 3., 3.]];
    let tmax = if cx.t_limit.is_finite() { cx.t_limit * (1.0 - 1e-3) } else { f32::INFINITY };
    for w in GRID {
        let d = (a * w[0] + b * w[1] + c * w[2]).normalize();
        if cx.bvh.occluded(cx.scene, &Ray::new(cx.o, d), 0.0, tmax, &mut ls.ray_nodes) {
            v.false_empty += 1;
        }
    }
}
