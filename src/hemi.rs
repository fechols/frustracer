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

use crate::bvh::Ray;
use crate::frustum::{self, TileFrustum};
use crate::ftree::Accel;
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
pub const HEMI_CUT: usize = 64;

/// Hemi-share group qualifiers, both applied to spreads MEASURED from the fp
/// apexes (coplanarity is never assumed — hit points carry möller-trumbore
/// fp error off the true triangle plane):
/// - out-of-plane spread η must stay well below eps or the padded root plane
///   re-admits the own surface (which sits at −eps below it) — the
///   blocked-everywhere collapse. eps/4 leaves the exclusion a healthy margin
///   over `aabb_outside`'s slab-scale slack (~eps/10).
/// - total spread δ caps at ao_radius/8: a grazing-angle group on a long thin
///   triangle can span many pixel footprints, and pads that large degrade the
///   capture toward all-blocked — per-pixel is cheaper there.
pub const SHARE_ETA_FRAC: f32 = 0.25;
pub const SHARE_DELTA_FRAC: f32 = 0.125;

/// Shared-hemisphere record capacities. Query-leaves occur at exactly
/// depth = max_depth − 1 (the LEAF_LEVELS cutoff), so ≤ 4^(FB_DEPTH_CAP−1)
/// leaves; cuts are pushed once per refine on the internal spine
/// (1 + 4 + 16 at the cap). Deeper fb.depth poisons the capture — the group
/// falls back per-pixel, coarser never wrong.
pub const FB_DEPTH_CAP: u32 = 4;
const REC_LEAVES: usize = 64;
const REC_CUTS: usize = 24;

/// One recorded query-leaf: the spherical-triangle cell, its inherited empty
/// claim `tc` (from the REP's apex — members consume `tc − delta`), and an
/// index into the record's deduped cut slots (siblings share their parent's
/// cut verbatim).
#[derive(Clone, Copy)]
struct LeafCell {
    a: Vec3A,
    b: Vec3A,
    c: Vec3A,
    tc: f32,
    cut: u16,
}

const LEAF_ZERO: LeafCell =
    LeafCell { a: Vec3A::ZERO, b: Vec3A::ZERO, c: Vec3A::ZERO, tc: 0.0, cut: 0 };

#[derive(Clone, Copy)]
struct CutSlot {
    nodes: [u32; HEMI_CUT],
    len: u16,
}

const CUT_ZERO: CutSlot = CutSlot { nodes: [0; HEMI_CUT], len: 0 };

/// One coherent group's shared hemisphere capture: the rep's whole padded
/// quadtree, reduced to what consumers need. Empty cells are NOT stored per
/// cell — their analytic values (PSA / ∫sky·cos) are pure functions of
/// group-invariant inputs (bit-equal n, shared sun), so capture folds them
/// into `open_mass` once and every member adds it verbatim. Query-leaves
/// store (cell, tc, cut) so each member shoots its OWN stratified rays from
/// its own apex with `tmin = max(0, tc − delta)` — sharing amortizes the
/// bound queries, never the samples.
///
/// Allocate once per tile and re-`capture` per group: `reset` only rewinds
/// the counters (a fresh ~10 KB zeroing per 2×2 group would cost real frame
/// time at ~50k groups/frame).
pub struct HemiShare {
    pub delta: f32,
    /// Whole padded hemisphere proven open within the capture t_limit — the
    /// member fast path (AO: π; GI: analytic sky), no leaves recorded.
    pub open: bool,
    /// Capture bailed (fb.depth over the record cap, or slot overflow —
    /// structurally impossible at preset depths). Consumers must fall back.
    pub poisoned: bool,
    /// True if captured under GI (sky radiance folded); consumers assert the
    /// mode matches — an AO record's `open_mass` is a PSA, not radiance.
    gi_mode: bool,
    open_mass: Vec3A,
    open_psa: f32,
    n_leaves: u16,
    n_cuts: u16,
    leaves: [LeafCell; REC_LEAVES],
    cuts: [CutSlot; REC_CUTS],
    /// Verify-only (the check harness): keep every empty cell so Apply can
    /// re-validate the rep's claims with reference rays from EACH member's
    /// apex. Never set on the render path (Vec::new never allocates).
    pub record_empties: bool,
    pub empties: Vec<[Vec3A; 3]>,
}

impl HemiShare {
    pub fn new() -> Self {
        HemiShare {
            delta: 0.0,
            open: false,
            poisoned: false,
            gi_mode: false,
            open_mass: Vec3A::ZERO,
            open_psa: 0.0,
            n_leaves: 0,
            n_cuts: 0,
            leaves: [LEAF_ZERO; REC_LEAVES],
            cuts: [CUT_ZERO; REC_CUTS],
            record_empties: false,
            empties: Vec::new(),
        }
    }

    fn reset(&mut self, delta: f32, gi_mode: bool) {
        self.delta = delta;
        self.open = false;
        self.poisoned = false;
        self.gi_mode = gi_mode;
        self.open_mass = Vec3A::ZERO;
        self.open_psa = 0.0;
        self.n_leaves = 0;
        self.n_cuts = 0;
        self.empties.clear();
    }

    fn push_cut(&mut self, cut: &[u32]) -> Option<u16> {
        if self.n_cuts as usize == REC_CUTS {
            return None;
        }
        let i = self.n_cuts;
        let slot = &mut self.cuts[i as usize];
        slot.nodes[..cut.len()].copy_from_slice(cut);
        slot.len = cut.len() as u16;
        self.n_cuts += 1;
        Some(i)
    }
}

impl Default for HemiShare {
    fn default() -> Self {
        Self::new()
    }
}

/// Capture the shared hemisphere at the group rep's apex `o` with normal `n`
/// into `rec`. Every frustum is padded for the group: cell planes by
/// `pad_k = delta·|in-plane(n_k)| + eta·|n_k·n|`, the root by `eta` alone —
/// pads are derived from spreads MEASURED on the fp apexes, and the caller's
/// qualifiers keep `eta ≪ eps` (own-surface exclusion) and `delta` small
/// (pad blowup). AO passes `t_limit = ao_radius` and the capture queries at
/// `ao_radius + delta` so a member's claims cover exactly its own radius;
/// `sun` Some = GI mode (t_limit ∞), folding analytic sky per empty cell.
/// Consumes no rng; costs one tree of bound queries + refines per GROUP.
#[allow(clippy::too_many_arguments)]
pub fn share_capture(
    scene: &Scene,
    accel: Accel,
    o: Vec3A,
    n: Vec3A,
    t1: Vec3A,
    t2: Vec3A,
    max_depth: u32,
    t_limit: f32,
    delta: f32,
    eta: f32,
    sun: Option<Vec3A>,
    cl: &crate::clouds::Clouds,
    rec: &mut HemiShare,
    ls: &mut LocalStats,
) {
    rec.reset(delta, sun.is_some());
    let max_depth = max_depth.max(1);
    if max_depth > FB_DEPTH_CAP {
        rec.poisoned = true;
        return;
    }
    let cx = Cx {
        scene,
        accel,
        o,
        n,
        max_depth,
        t_limit: if t_limit.is_finite() { t_limit + delta } else { t_limit },
        // The capture itself never shades (empty-cell folding is dome-only —
        // pure functions of n/sun, bit-identical per member), but the state
        // rides along so the record can never diverge from its consumers'.
        gi: sun.map(|s| Gi { sun: s, depth: 0, cl: *cl }),
    };
    let root = TileFrustum::half_space_padded(o, n, eta);
    ls.hemi_queries += 1;
    let bound =
        accel.nearest_within(&root, 0.0, cx.t_limit, accel.root_cut(), &mut ls.hemi_nodes);
    let Some(t) = bound else {
        rec.open = true;
        return;
    };
    let (_, tc) = frustum::advance_tc(t, 0.0, scene.eps);
    let mut cut = [0u32; HEMI_CUT];
    let len = accel.refine_cut(
        &root,
        tc,
        cx.t_limit,
        accel.root_cut(),
        &mut cut,
        &mut ls.hemi_nodes,
        &mut ls.cut_overflows,
    );
    debug_assert!(len > 0, "padded hemisphere root refine emptied a non-open cut");
    let child: &[u32] = if len > 0 { &cut[..len] } else { accel.root_cut() };
    let Some(cut_idx) = rec.push_cut(child) else {
        rec.poisoned = true;
        return;
    };
    for [a, b, c] in sphcell::octants(n, t1, t2) {
        capture_cell(&cx, a, b, c, 1, tc, cut_idx, delta, eta, rec, ls);
        if rec.poisoned {
            return;
        }
    }
}

/// `cell()`'s recursion, recording terminals instead of consuming them.
/// Frustums are the padded twins; everything else (advance/slack, blocked
/// cells subdivide, LEAF_LEVELS cutoff) is verbatim.
#[allow(clippy::too_many_arguments)]
fn capture_cell(
    cx: &Cx,
    a: Vec3A,
    b: Vec3A,
    c: Vec3A,
    depth: u32,
    t_start: f32,
    cut_idx: u16,
    delta: f32,
    eta: f32,
    rec: &mut HemiShare,
    ls: &mut LocalStats,
) {
    let f = TileFrustum::tri_cell_padded(cx.o, a, b, c, delta, eta, cx.n);
    let cut_in = &rec.cuts[cut_idx as usize];
    ls.hemi_queries += 1;
    let bound = cx.accel.nearest_within(
        &f,
        t_start,
        cx.t_limit,
        &cut_in.nodes[..cut_in.len as usize],
        &mut ls.hemi_nodes,
    );
    let Some(t) = bound else {
        // Empty for the whole group: fold the analytic mass once. The values
        // are bit-identical per member (pure functions of the shared n/sun).
        ls.hemi_cells_empty += 1;
        match cx.gi {
            None => rec.open_mass.x += sphcell::psa(a, b, c, cx.n),
            Some(g) => {
                rec.open_mass += sky_cell(
                    cx.n,
                    g.sun,
                    cx.scene.sky_scale,
                    cx.scene.night,
                    a,
                    b,
                    c,
                    5u32.saturating_sub(depth),
                )
            }
        }
        rec.open_psa += sphcell::psa(a, b, c, cx.n);
        if rec.record_empties {
            rec.empties.push([a, b, c]);
        }
        return;
    };
    let (_, tc) = frustum::advance_tc(t, t_start, cx.scene.eps);
    if depth + LEAF_LEVELS >= cx.max_depth {
        if rec.n_leaves as usize == REC_LEAVES {
            rec.poisoned = true;
            return;
        }
        rec.leaves[rec.n_leaves as usize] = LeafCell { a, b, c, tc, cut: cut_idx };
        rec.n_leaves += 1;
        return;
    }
    let mut cut = [0u32; HEMI_CUT];
    let len = cx.accel.refine_cut(
        &f,
        tc,
        cx.t_limit,
        &rec.cuts[cut_idx as usize].nodes[..rec.cuts[cut_idx as usize].len as usize],
        &mut cut,
        &mut ls.hemi_nodes,
        &mut ls.cut_overflows,
    );
    debug_assert!(len > 0, "padded refine_cut emptied a non-open hemisphere cell");
    let child_idx = if len > 0 {
        match rec.push_cut(&cut[..len]) {
            Some(i) => i,
            None => {
                rec.poisoned = true;
                return;
            }
        }
    } else {
        cut_idx
    };
    let (mab, mbc, mca) = sphcell::midpoints(a, b, c);
    for [ca, cb, cc] in [[a, mab, mca], [mab, b, mbc], [mca, mbc, c], [mab, mbc, mca]] {
        capture_cell(cx, ca, cb, cc, depth + 1, tc, child_idx, delta, eta, rec, ls);
        if rec.poisoned {
            return;
        }
    }
}

/// Query-tree cutoff, the hemisphere analog of LEAF_TILE: cells this many
/// levels above the ray budget stop querying and distribute one stratified
/// ray per sub-cell instead. One bound query then amortizes over 4^LEAF_LEVELS
/// rays — per-ray queries were measured to cost far more than they saved
/// (occlusion rays are ~10 node visits; a bound query on a dense cut is more).
const LEAF_LEVELS: u32 = 1;

/// Shading quality for radiance carried back by GI leaf rays (the depth-1
/// policy): one shadow sample, no further reflections or hemispheres — same
/// spirit as the specular bounce, and structurally recursion-free.
/// pub(crate): the `--check` GI reference must implement the SAME policy so
/// the A/B isolates integrator error from policy error.
///
/// `ao_samples` IS THE WHOLE DIFFERENCE BETWEEN GI AND A FLAT AMBIENT, and it
/// used to be 0. A bounce surface's own ambient is `sky_sh.irradiance(n) * ao`
/// (shade.rs), so at 0 the `ao` factor is 1.0 and every bounce surface is lit
/// as though it stood in an open field under the full sky — including a wall
/// deep under an arcade that can see almost none of it. Each occluded
/// direction then hands the integral a full-sky-lit surface, so `gi()`
/// collapses toward the unoccluded sky value EVERYWHERE: a uniform lift with
/// no structure, which reads on screen as exactly the flat ambient constant
/// this tier exists to replace — brighter than no GI at all, and visibly
/// WORSE, which is why it survived every gate (they bound error against a
/// reference running this same policy, so both sides were flat together).
///
/// MEASURED, San Miguel's patio at 15:30 (1280x720, 96 spp, luminance):
///
/// | `ao_samples` | mean | shadowed | contrast p90/p10 |
/// |---|---|---|---|
/// | 0 (the bug) | 46.30 | 35.68 | 2.34 |
/// | **1** | **26.15** | **14.45** | **4.70** |
/// | 2 | 26.15 | 14.45 | 4.70 |
/// | fb OFF (no GI) | 22.45 | 6.17 | 10.01 |
///
/// One ray is enough: 2 measured 0.19% different — variance the accumulation
/// path launders — for 21% more time, and 1 is symmetric with
/// `shadow_samples`. Shadowed regions stay 2.3x above the fb-OFF tier, which
/// IS the bounce light; what returns is the falloff. Whole-frame cost on that
/// probe 28.4 -> 37.1 s (+31%), far cheaper than a second bounce and it keeps
/// the tier recursion-free.
pub(crate) const BOUNCE_Q: shade::Quality = shade::Quality {
    shadow_samples: 1,
    ao_samples: 1,
    reflections: false,
    fb: shade::FrustumBounce::OFF,
    // Bounce hits never re-bounce (RTGI's own leaf policy included): the
    // SH×AO ambient above IS the tail. Keeps both GI tiers recursion-bounded.
    rtgi: false,
    // The hemi gather DELIVERS emissive transport (fb.gi drops the cluster
    // NEE instead); the RTGI bounce overrides this per frame via struct
    // update when NEE is live (the NEE-keep rule — see Quality's field doc).
    emissive_display: true,
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
    /// Cloud state for the leaf-hit BOUNCE shade only — its direct sun term
    /// is cloud-shadowed like any other `shade()` (T ≤ 1, so the 2^18
    /// fixed-point accumulator only gets QUIETER). The gather geometry —
    /// `sky_cell`, `dome()` leaf misses — never sees clouds (sky.rs's table).
    cl: crate::clouds::Clouds,
}

struct Cx<'a> {
    scene: &'a Scene,
    /// Rays go to `accel.bvh` (always); bound queries dispatch to the wide
    /// frustum tree when one is wired — the two-tree split.
    accel: Accel<'a>,
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
    accel: Accel,
    o: Vec3A,
    n: Vec3A,
    t1: Vec3A,
    t2: Vec3A,
    max_depth: u32,
    t_limit: f32,
    share: Option<&HemiShare>,
    rng: &mut fastrand::Rng,
    verify: Option<&mut VerifyCounters>,
    ls: &mut LocalStats,
) -> f32 {
    let cx = Cx { scene, accel, o, n, max_depth: max_depth.max(1), t_limit, gi: None };
    // No upper clamp: a leaf sample's V·cosθ·Ω can overshoot its cell's PSA,
    // so the sum can exceed π; truncating only that high tail would bias the
    // estimator low (the --check signed-mean gate exists to catch bias).
    // Every term is ≥ 0, so no lower clamp is needed either.
    integrate(&cx, t1, t2, share, rng, verify, ls).x / PI
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
    accel: Accel,
    o: Vec3A,
    n: Vec3A,
    t1: Vec3A,
    t2: Vec3A,
    max_depth: u32,
    sun: Vec3A,
    cl: &crate::clouds::Clouds,
    depth: u32,
    share: Option<&HemiShare>,
    rng: &mut fastrand::Rng,
    verify: Option<&mut VerifyCounters>,
    ls: &mut LocalStats,
) -> Vec3A {
    let cx = Cx {
        scene,
        accel,
        o,
        n,
        max_depth: max_depth.max(1),
        t_limit: f32::INFINITY,
        gi: Some(Gi { sun, depth, cl: *cl }),
    };
    (integrate(&cx, t1, t2, share, rng, verify, ls) / PI).max(Vec3A::ZERO)
}

fn integrate(
    cx: &Cx,
    t1: Vec3A,
    t2: Vec3A,
    share: Option<&HemiShare>,
    rng: &mut fastrand::Rng,
    mut verify: Option<&mut VerifyCounters>,
    ls: &mut LocalStats,
) -> Vec3A {
    ls.hemi_points += 1;
    let mut acc = Acc::default();
    // The shared group capture when one exists (hemi sharing): this point
    // runs ZERO bound queries — analytic mass is folded from the record and
    // every query-leaf shoots this member's own fresh rays with the claim
    // shrunk by delta. Otherwise the tangent half-space is queried over the
    // whole BVH as always.
    if let Some(sh) = share {
        ls.hemi_share_points += 1;
        debug_assert!(!sh.poisoned, "a poisoned hemi share must not be consumed");
        debug_assert!(
            sh.gi_mode == cx.gi.is_some(),
            "hemi share record mode mismatch (AO record consumed by GI or vice versa)"
        );
        if sh.open {
            // The rep's padded claim covers this apex: whole hemisphere open
            // within the (already delta-widened) capture limit.
            ls.hemi_cells_empty += 1;
            open_hemisphere(cx, t1, t2, &mut acc, verify.as_deref_mut(), ls);
        } else {
            acc.open += sh.open_mass;
            if let Some(v) = verify.as_deref_mut() {
                // Re-validate the rep's folded empty claims from THIS apex
                // (the harness captures with record_empties).
                acc.accounted += sh.open_psa;
                for e in &sh.empties {
                    check_empty(cx, *e, v, ls);
                }
            }
            // Every recorded leaf sits at the same depth max(1, max_depth −
            // LEAF_LEVELS) (capture_cell's recursion is uniform), so the
            // sub-cell budget the unshared path would use there is
            // min(LEAF_LEVELS, max_depth − 1) — a bare LEAF_LEVELS would
            // silently 4× the ray count at max_depth = 1 (no preset reaches
            // it, but the estimator shape must match the unshared twin).
            let levels = cx.max_depth.saturating_sub(1).min(LEAF_LEVELS);
            for l in &sh.leaves[..sh.n_leaves as usize] {
                let cut = &sh.cuts[l.cut as usize];
                let t_start = (l.tc - sh.delta).max(0.0);
                leaf_rays(
                    cx,
                    l.a,
                    l.b,
                    l.c,
                    levels,
                    t_start,
                    &cut.nodes[..cut.len as usize],
                    rng,
                    &mut verify,
                    ls,
                    &mut acc,
                );
            }
        }
        return finish(&mut acc, verify);
    }
    let root = TileFrustum::half_space(cx.o, cx.n);
    ls.hemi_queries += 1;
    let bound = cx.accel.nearest_within(
        &root,
        0.0,
        cx.t_limit,
        cx.accel.root_cut(),
        &mut ls.hemi_nodes,
    );
    match bound {
        None => {
            // The whole hemisphere is open within t_limit — one query, done.
            ls.hemi_cells_empty += 1;
            open_hemisphere(cx, t1, t2, &mut acc, verify.as_deref_mut(), ls);
        }
        Some(t) => {
            // Advance + refine exactly like tile_step, then recurse into the
            // 4 octants (depth 1; leaves land at depth == max_depth).
            let (_, tc) = frustum::advance_tc(t, 0.0, cx.scene.eps);
            let mut cut = [0u32; HEMI_CUT];
            let len = cx.accel.refine_cut(
                &root,
                tc,
                cx.t_limit,
                cx.accel.root_cut(),
                &mut cut,
                &mut ls.hemi_nodes,
                &mut ls.cut_overflows,
            );
            debug_assert!(len > 0, "hemisphere root refine emptied a non-open cut");
            let child: &[u32] = if len > 0 { &cut[..len] } else { cx.accel.root_cut() };
            for [a, b, c] in sphcell::octants(cx.n, t1, t2) {
                cell(cx, a, b, c, 1, tc, child, rng, &mut verify, ls, &mut acc);
            }
        }
    }
    finish(&mut acc, verify)
}

/// Whole-hemisphere-open resolution shared by the root query's `None` and a
/// shared `RootSeed::Open`: analytic PSA/sky over the octants, re-verified
/// from THIS apex (a member validates the rep's claim at its own origin).
fn open_hemisphere(
    cx: &Cx,
    t1: Vec3A,
    t2: Vec3A,
    acc: &mut Acc,
    verify: Option<&mut VerifyCounters>,
    ls: &mut LocalStats,
) {
    if let Some(v) = verify {
        acc.accounted = PI;
        for [a, b, c] in sphcell::octants(cx.n, t1, t2) {
            check_empty(cx, [a, b, c], v, ls);
        }
    }
    match cx.gi {
        None => acc.open.x = PI,
        Some(g) => {
            for [a, b, c] in sphcell::octants(cx.n, t1, t2) {
                acc.open += sky_cell(cx.n, g.sun, cx.scene.sky_scale, cx.scene.night, a, b, c, 4);
            }
        }
    }
}

/// PSA accounting + the integrator's return value.
fn finish(acc: &mut Acc, verify: Option<&mut VerifyCounters>) -> Vec3A {
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
    let bound =
        cx.accel.nearest_within(&f, t_start, cx.t_limit, cut_in, &mut ls.hemi_nodes);
    let Some(t) = bound else {
        ls.hemi_cells_empty += 1;
        match cx.gi {
            None => acc.open.x += sphcell::psa(a, b, c, cx.n),
            // Refinement budget: enough extra levels to reach ~6° cells even
            // for an octant-sized empty cell near the sun glow.
            Some(g) => {
                acc.open += sky_cell(
                    cx.n,
                    g.sun,
                    cx.scene.sky_scale,
                    cx.scene.night,
                    a,
                    b,
                    c,
                    5u32.saturating_sub(depth),
                )
            }
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
    let len = cx.accel.refine_cut(
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
            // Cut-seed only when it pays. A hemi cut sits pinned at HEMI_CUT (64),
            // so seeding from it is 64 scattered coarse roots against one coherent
            // root descent — measured 3-10% slower. The cut still drives the bound
            // QUERIES; this is only about how the leaf RAY traverses. Same tmin
            // either way, so the tmin-overshoot / cut-miss gates are unaffected.
            // Under the wide tree the cut is slot-refs and must translate to
            // binary roots first (ray_roots) — rays only ever walk the ray BVH.
            // `transmittance`, not `occluded`: AO is a LIGHT query, so a
            // glass occluder passes its tint (folded to gray — the sampled-AO
            // tier's mean-of-components rule, exact 1.0/0.0 on opaque scenes
            // via the true divide).
            let tp = if crate::bvh::cut_seed_hemi() {
                let mut buf = [0u32; HEMI_CUT];
                let roots = cx.accel.ray_roots(cut, &mut buf);
                cx.accel.bvh.transmittance_multi(
                    cx.scene,
                    &ray,
                    tc,
                    cx.t_limit,
                    roots,
                    &mut ls.ray_nodes,
                )
            } else {
                cx.accel.bvh.transmittance(cx.scene, &ray, tc, cx.t_limit, &mut ls.ray_nodes)
            };
            acc.ray.x += weight * ((tp.x + tp.y + tp.z) / 3.0);
            if let Some(v) = verify.as_deref_mut() {
                acc.accounted += sphcell::psa(a, b, c, cx.n);
                verify_leaf_ray(cx, &ray, tc, v, ls);
                // Cut-vs-root agreement. Bit-equality is the gate on the
                // binary arm (exact ZERO/ONE); with ≥3 tinted interfaces the
                // two traversals may associate the f32 product differently,
                // so tinted throughputs get a 1-ulp-scale relative slack —
                // a real cut miss (a whole interface dropped) moves the
                // product by the interface's tint, orders of magnitude more.
                let root =
                    cx.accel.bvh.transmittance(cx.scene, &ray, tc, cx.t_limit, &mut ls.ray_nodes);
                let agree = tp == root
                    || (tp - root).abs().max_element() <= 1e-6 * root.abs().max_element();
                if !agree {
                    v.cut_miss += 1;
                }
            }
        }
        Some(g) => {
            let hit = if crate::bvh::cut_seed_hemi() {
                let mut buf = [0u32; HEMI_CUT];
                let roots = cx.accel.ray_roots(cut, &mut buf);
                cx.accel
                    .bvh
                    .intersect_multi(cx.scene, &ray, tc, f32::INFINITY, roots, &mut ls.ray_nodes)
            } else {
                cx.accel
                    .bvh
                    .intersect(cx.scene, &ray, tc, f32::INFINITY, &mut ls.ray_nodes)
            };
            if let Some(v) = verify.as_deref_mut() {
                acc.accounted += sphcell::psa(a, b, c, cx.n);
                verify_leaf_ray(cx, &ray, tc, v, ls);
                let full =
                    cx.accel.bvh.intersect(cx.scene, &ray, tc, f32::INFINITY, &mut ls.ray_nodes);
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
                // `gather`, not the full sky. A GI leaf ray landing in the sun
                // disc would (a) double-count light `direct_d` already delivers
                // with its own shadow ray, and (b) push ~1e3 radiance into the
                // 2^18 fixed-point hemi accumulator, which would saturate. The
                // star field is the opposite case (nothing else delivers it and
                // its mean is ~1e-3), so `gather` carries it in — see sky.rs's
                // star row.
                None => crate::sky::gather(d, g.sun, cx.scene.sky_scale, cx.scene.night),
                Some(h) => shade::shade(
                    cx.scene,
                    cx.accel.bvh,
                    &ray,
                    &h,
                    None,
                    &BOUNCE_Q,
                    rng,
                    g.sun,
                    &g.cl,
                    // Bounce hits stay ISOTROPIC (aniso 1) at an octant-scale
                    // spread: the cell footprint is coarse by design —
                    // over-blurred bounce albedo is variance reduction, and 16
                    // taps per GI ray would buy nothing. The GPU mirrors this
                    // (hemi_leaf.hlsl: aniso false).
                    shade::Cone::bounce(),
                    g.depth + 1,
                    ls,
                    None,
                    shade::VisCtl::Off,
                    None,
                    // Fireflies never light bounce surfaces (the gather
                    // exclusion — the stars rule).
                    None,
                    // Emissive cluster lights: None — the GI gather IS the
                    // emissive transport here (the display `color += e` on
                    // this very shade call delivers a hit emitter's
                    // radiance), and the NEE tier is off under fb.gi by the
                    // same rule (src/emissive.rs's inverted once-per-path
                    // argument). A Some here would double-count.
                    None,
                ),
            };
            acc.ray += l * weight;
        }
    }
}

/// tmin soundness gate: a tmin=0 reference ray must not hit strictly inside
/// the claimed-empty ball.
///
/// Only for a sample the integrand actually uses. Arvo sampling of a
/// horizon-adjacent cell can land fp-epsilon BELOW the tangent plane, where
/// `weight = d·n max 0` is exactly 0 — the sample contributes nothing, and
/// nothing was ever claimed about it: the hemi ROOT CUT *is* the tangent
/// half-space, so the bound query proves emptiness over the open hemisphere
/// and says nothing about directions outside it. Traced from tmin=0, such a
/// direction grazes back down onto the apex's OWN surface (at -eps) at
/// t = eps/|d·n|, an eps-offset artifact rather than occlusion — and at a
/// grazing angle that t is large enough to land deep inside a perfectly
/// sound ball (measured on Intel Arc: d·n = -4.31e-4 put the own ground
/// plane at t = 39.36 inside a correct empty claim of 57.26). NVIDIA and AMD
/// round the sample the other way and never trip it; that is luck, not
/// soundness, so the guard — not the platform — is the invariant.
fn verify_leaf_ray(cx: &Cx, ray: &Ray, tc: f32, v: &mut VerifyCounters, ls: &mut LocalStats) {
    if ray.d.dot(cx.n) <= 0.0 {
        return;
    }
    if let Some(h) = cx.accel.bvh.intersect(cx.scene, ray, 0.0, f32::INFINITY, &mut ls.ray_nodes) {
        if h.t < tc * (1.0 - 1e-3) {
            v.tmin_overshoot += 1;
        }
    }
}

/// Analytic sky over a proven-empty cell: centroid radiance × exact PSA, with
/// pure-math midpoint refinement where the dome varies fastest — an empty
/// parent proves all children empty, so refinement costs `dome()` evaluations
/// only, no BVH work.
///
/// This integrates the DOME, never `sky::radiance` — and that is a correctness
/// requirement, not a preference. Centroid point-sampling a cell that is COARSER
/// than the sun disc would either miss the disc entirely (the sun contributes
/// nothing) or land on it and splat the whole cell at ~1e3 radiance. Classic
/// aliasing: fireflies and energy loss, unfixable by adding `levels`. Excluding
/// the disc removes the sharp feature outright — which is precisely why the
/// frequency split is the right architecture and not merely a convenience.
#[allow(clippy::too_many_arguments)]
fn sky_cell(
    n: Vec3A,
    sun: Vec3A,
    scale: f32,
    night: f32,
    a: Vec3A,
    b: Vec3A,
    c: Vec3A,
    levels: u32,
) -> Vec3A {
    let cen = sphcell::centroid(a, b, c);
    if levels > 0 {
        // Angular radius of the cell around its centroid.
        let cos_r = cen.dot(a).min(cen.dot(b)).min(cen.dot(c)).clamp(-1.0, 1.0);
        // Refine while the cell is coarser than ~12° — the horizon→zenith
        // gradient is anti-correlated with the cosine weight, so a coarse
        // centroid systematically over-brightens.
        let coarse = cos_r < 0.978;
        // The dome's sharpest surviving feature is the MIE AUREOLE (the forward
        // Henyey-Greenstein lobe at g = 0.76, which falls ~4x over 20° — a
        // little softer than the `dot^32` glow this test used to chase). Refine
        // to ~6° within a conservative 30° cone of the sun:
        // cos(angle(cen, sun)) > cos(r + 30°), expanded.
        let near_aureole = cos_r < 0.995 && {
            let sin_r = (1.0 - cos_r * cos_r).max(0.0).sqrt();
            cen.dot(sun) > cos_r * 0.866 - sin_r * 0.5
        };
        if coarse || near_aureole {
            let (mab, mbc, mca) = sphcell::midpoints(a, b, c);
            let mut sum = Vec3A::ZERO;
            for [ca, cb, cc] in [[a, mab, mca], [mab, b, mbc], [mca, mbc, c], [mab, mbc, mca]] {
                sum += sky_cell(n, sun, scale, night, ca, cb, cc, levels - 1);
            }
            return sum;
        }
    }
    // `gather`, not `dome`: the star field's smooth mean rides along (see
    // sky.rs's star row). It adds no sharp feature for the refinement above to
    // chase — it is near-constant over the whole upper hemisphere — so the
    // `coarse`/`near_aureole` budget is unaffected.
    crate::sky::gather(cen, sun, scale, night) * sphcell::psa(a, b, c, n)
}

/// Reference-ray re-validation of an empty-cell claim: directions strictly
/// inside the cell must all be unoccluded within t_limit (shaved for fp).
fn check_empty(cx: &Cx, [a, b, c]: [Vec3A; 3], v: &mut VerifyCounters, ls: &mut LocalStats) {
    const GRID: [[f32; 3]; 6] =
        [[1., 1., 1.], [6., 1., 1.], [1., 6., 1.], [1., 1., 6.], [3., 3., 1.], [1., 3., 3.]];
    let tmax = if cx.t_limit.is_finite() { cx.t_limit * (1.0 - 1e-3) } else { f32::INFINITY };
    for w in GRID {
        let d = (a * w[0] + b * w[1] + c * w[2]).normalize();
        if cx.accel.bvh.occluded(cx.scene, &Ray::new(cx.o, d), 0.0, tmax, &mut ls.ray_nodes) {
            v.false_empty += 1;
        }
    }
}
