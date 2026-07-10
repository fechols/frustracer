//! Light-shaft shadow culling: a 4-corner frustum from the shading point
//! through the area light's rect proves subrects of the light unoccluded, so
//! their shadow samples skip the occlusion ray entirely — the integrand is
//! evaluated unchanged (identical sampling, identical estimator; only
//! provably-unnecessary ray casts are removed). Ambiguous subrects keep their
//! shadow rays, seeded from the shaft's refined cut with the shaft's tmin —
//! the rays that remain are exactly the penumbra rays.
//!
//! Soundness:
//! - The apex is the shading point `p` (eps-offset, same as every secondary
//!   ray). The shaft has its own tmin chain from t_start = 0; the primary
//!   tile's tmin never appears.
//! - Every shaft frustum carries the surface tangent plane as a 5th clip
//!   plane. This is what excludes the own-surface AABB (which hugs the apex
//!   at distance ~0 and can never be culled by the 4 side planes — without
//!   the clip, every shaft is "blocked at zero" and nothing is ever proven
//!   lit). Sound because `classify` is only consulted for samples with
//!   `wi·n > 0`, whose whole segment lies in the clip half-space.
//! - A subrect's query is clamped to `t_limit` = the max distance from `p`
//!   to the subrect's corners (distance is convex over the rect, so the max
//!   is exact). `None` ⇒ frustum ∩ ball(t_limit) is empty ⇒ no occluder can
//!   sit between `p` and any light sample in the subrect (whose distance is
//!   ≤ t_limit; the shadow ray tmax is dist − eps < t_limit). The light rect
//!   itself has no BVH geometry, so nothing at the boundary confuses this.
//! - Cuts are refined with `t_far` = the same t_limit — sound because every
//!   consumer ray uses tmax < t_limit (the clamped-consumer contract).
//! - Degenerate shafts (p edge-on to the light: near-coplanar corner dirs)
//!   produce zero plane normals that never cull — subrects just stay
//!   ambiguous and fall back to rays. Safe, never wrong.

use crate::bvh::Bvh;
use crate::frustum::{self, TileFrustum};
use crate::scene::Scene;
use crate::stats::LocalStats;
use glam::Vec3A;

const SHAFT_CUT: usize = 64;
/// Root + one adaptive 4-way split of the light rect. Deeper splits cost more
/// bound queries than the ≤ 4 shadow rays they could save.
const MAX_LEAVES: usize = 4;

/// One classified leaf subrect of the light, in the light's (su, sv) param
/// space ([-1, 1]²: `lp = center + su·u + sv·v`).
pub struct Leaf {
    /// [su0, sv0, su1, sv1]
    pub r: [f32; 4],
    /// Proven unoccluded: every shadow sample in this subrect skips its ray.
    pub lit: bool,
    tc: f32,
    cut_off: u16,
    cut_len: u16,
}

pub struct Shaft {
    leaves: [Leaf; MAX_LEAVES],
    len: usize,
    cuts: [u32; MAX_LEAVES * SHAFT_CUT],
}

/// What to do with a shadow sample at light param (su, sv).
pub enum Class<'a> {
    /// Proven unoccluded — skip the ray.
    Lit,
    /// Shoot the ray: traversal seeded from the shaft cut with the shaft tmin.
    Test { tmin: f32, cut: &'a [u32] },
}

impl Shaft {
    pub fn leaves(&self) -> &[Leaf] {
        &self.leaves[..self.len]
    }

    pub fn classify(&self, su: f32, sv: f32) -> Class<'_> {
        // Leaves tile [-1,1]²; boundary directions belong to both closures,
        // so either classification is sound — first match wins.
        for l in self.leaves() {
            if su >= l.r[0] && su <= l.r[2] && sv >= l.r[1] && sv <= l.r[3] {
                return if l.lit {
                    Class::Lit
                } else {
                    Class::Test {
                        tmin: l.tc,
                        cut: &self.cuts[l.cut_off as usize..(l.cut_off + l.cut_len) as usize],
                    }
                };
            }
        }
        // No leaf contains (su, sv): out-of-range or NaN input. Never borrow
        // a neighbor's claim — a lit leaf's proof does not cover this sample,
        // and skipping its ray on it would be false-lit. Full-tree test.
        Class::Test { tmin: 0.0, cut: &[0] }
    }
}

/// Corner directions (normalized, perimeter order) and the exact max distance
/// from `p` over the subrect.
fn corners(scene: &Scene, p: Vec3A, r: [f32; 4]) -> ([Vec3A; 4], f32) {
    let l = &scene.light;
    let lp = |su: f32, sv: f32| l.center + l.u * su + l.v * sv;
    let cs = [lp(r[0], r[1]), lp(r[2], r[1]), lp(r[2], r[3]), lp(r[0], r[3])];
    let mut max_d = 0.0f32;
    let mut dirs = [Vec3A::ZERO; 4];
    for (i, c) in cs.iter().enumerate() {
        let v = *c - p;
        max_d = max_d.max(v.length());
        dirs[i] = v.normalize_or_zero();
    }
    (dirs, max_d * (1.0 + 1e-5))
}

/// Build the shaft for shading point `p` with surface normal `n`: one root
/// query over the whole light rect; if ambiguous, one adaptive 4-way split.
pub fn build(scene: &Scene, bvh: &Bvh, p: Vec3A, n: Vec3A, ls: &mut LocalStats) -> Shaft {
    let mut s = Shaft {
        leaves: std::array::from_fn(|_| Leaf { r: [0.0; 4], lit: true, tc: 0.0, cut_off: 0, cut_len: 0 }),
        len: 0,
        cuts: [0; MAX_LEAVES * SHAFT_CUT],
    };
    let root_r = [-1.0, -1.0, 1.0, 1.0];
    let (dirs, t_limit) = corners(scene, p, root_r);
    let f = TileFrustum::with_clip(p, dirs, n);
    ls.shaft_queries += 1;
    let bound =
        frustum::nearest_geometry_distance_within(bvh, &f, 0.0, t_limit, &[0], &mut ls.shaft_nodes);
    let Some(t) = bound else {
        s.leaves[0] = Leaf { r: root_r, lit: true, tc: 0.0, cut_off: 0, cut_len: 0 };
        s.len = 1;
        return s;
    };
    let (_, tc) = frustum::advance_tc(t, 0.0, scene.eps);
    let mut root_cut = [0u32; SHAFT_CUT];
    let root_len = frustum::refine_cut(
        bvh,
        &f,
        tc,
        t_limit,
        &[0],
        &mut root_cut,
        &mut ls.shaft_nodes,
        &mut ls.cut_overflows,
    );
    debug_assert!(root_len > 0, "shaft root refine emptied a non-lit cut");
    let parent: &[u32] = if root_len > 0 { &root_cut[..root_len] } else { &[0] };
    let (mu, mv) = (0.0f32, 0.0f32);
    for (i, r) in [
        [-1.0, -1.0, mu, mv],
        [mu, -1.0, 1.0, mv],
        [-1.0, mv, mu, 1.0],
        [mu, mv, 1.0, 1.0],
    ]
    .into_iter()
    .enumerate()
    {
        let (dirs, t_lim) = corners(scene, p, r);
        let fs = TileFrustum::with_clip(p, dirs, n);
        ls.shaft_queries += 1;
        let sub = frustum::nearest_geometry_distance_within(
            bvh,
            &fs,
            tc,
            t_lim,
            parent,
            &mut ls.shaft_nodes,
        );
        s.leaves[i] = match sub {
            None => Leaf { r, lit: true, tc: 0.0, cut_off: 0, cut_len: 0 },
            Some(ts) => {
                let (_, tcs) = frustum::advance_tc(ts, tc, scene.eps);
                let off = i * SHAFT_CUT;
                let mut cut = [0u32; SHAFT_CUT];
                let len = frustum::refine_cut(
                    bvh,
                    &fs,
                    tcs,
                    t_lim,
                    parent,
                    &mut cut,
                    &mut ls.shaft_nodes,
                    &mut ls.cut_overflows,
                );
                debug_assert!(len > 0, "shaft sub refine emptied a non-lit cut");
                let (cut_src, len) = if len > 0 { (&cut[..len], len) } else { (parent, parent.len()) };
                s.cuts[off..off + len].copy_from_slice(&cut_src[..len]);
                Leaf { r, lit: false, tc: tcs, cut_off: off as u16, cut_len: len as u16 }
            }
        };
    }
    s.len = 4;
    s
}
