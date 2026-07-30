//! Step-0 measurement harness for the frustum-query experiments.
//!
//! `FR_ORACLE=1` arms it; **default OFF and behaviour-free**. Every probe is a
//! read-only observation taken after the real work has already happened — no
//! rng draws, no writes to `accum`/`tbuf`/`info`, no effect on the cut, the
//! claim, or the recursion. So every same-seed / replay / VisCtl contract holds
//! trivially, and an armed run and an unarmed run produce the same image.
//!
//! It exists to answer, with numbers rather than argument, which of the
//! candidate frustum-query changes is worth building:
//!
//! * **Q1** — how many cut entries would a Kay-Kajiya slab-interval reject
//!   remove (MLRTA 2005 section 3.3), and — the number that actually matters —
//!   how many tiles would lose their *whole* cut and become **sky**, which is
//!   the 80-93% term in the leaf.hlsl ablation.
//! * **Q3** — how many cut entries are carried past the tile's own occlusion
//!   frontier, i.e. what a Teller-style frustum-advance structure would never
//!   have touched (TR-740 figure 1: we are in the "frustum culling" class).
//! * **Q4** — how often Teller's exact covering test would fire (TR-740
//!   section 4, optimization 2: all four extreme rays hitting one convex
//!   element proves every interior ray hits it, because the rays from a fixed
//!   apex meeting a convex set form a convex cone), and what pixel share it
//!   would cover. Pixels are the payoff, tiles tested are the cost.
//! * **Q5** — what share of `refine_cut`'s plane tests a straddle mask would
//!   eliminate (TR-740 section 4, optimization 3: a box straddling both
//!   opposing planes straddles every analogous child plane).
//!
//! Q2 (the sky pixels an adaptive-termination rule would forfeit) is
//! deliberately absent: it needs an ancestor mask threaded through
//! `trace_tile`, and Q4 may well subsume that experiment outright. Add it only
//! if adaptive termination survives this pass.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;

use glam::Vec3A;

use crate::bvh::{Aabb, Bvh, Ray};
use crate::camera::CamBasis;
use crate::frustum::TileFrustum;
use crate::scene::Scene;

/// Quadtree depths tracked individually; deeper folds into the last row.
const MAX_D: usize = 16;

/// Q4 tests four corner rays per tile. The full cut budget, deliberately: the
/// sound predicate is "only **or minimum-depth** element impinging", so a fat
/// cut can still fire, and gating this at 8 measured the gate rather than the
/// opportunity (mean cut is ~10-19).
const Q4_MAX_CUT: usize = crate::frustum::MAX_CUT;

/// Frames between reports. The counters are cumulative, so a `--check` run
/// (one hybrid frame) prints once and a `--spin` run prints a running table.
const REPORT_EVERY: u64 = 120;

pub fn armed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FR_ORACLE").is_ok_and(|v| v != "0"))
}

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        struct Counters { $($name: [AtomicU64; MAX_D],)* }
        static C: Counters = Counters {
            $($name: [const { AtomicU64::new(0) }; MAX_D],)*
        };
    };
}

counters! {
    tiles,          // split tiles observed
    cut_len,        // sum of their cut lengths
    // Q1 — the slab-interval reject
    q1_entries,     // cut entries tested
    q1_rejects,     // ...that the interval test proves the beam misses
    q1_sky_tiles,   // tiles whose WHOLE cut is rejected => new sky
    q1_sky_px,      // ...and the pixels they cover
    // Q3 — cut entries past this tile's own occlusion frontier
    q3_entries,
    q3_beyond,
    q3_leaves,      // leaf tiles probed (anti-vacuity)
    // Q4 — Teller's exact covering test
    q4_tested,      // tiles tested
    q4_fires,       // ...sound, with today's frustum-blind apex range
    q4_px,
    q4_beam,        // ...sound, with the frustum-aware slab-interval range
    q4_beam_px,
    q4_loose,       // upper bound: four corners agree, interference ignored
    q4_loose_px,
    // Q5 — straddle-mask inheritance
    q5_entries,
    q5_straddle_x,  // straddles both opposing planes on one axis pair
    q5_straddle_y,
}

static FRAMES: AtomicU64 = AtomicU64::new(0);

#[inline]
fn bump(c: &[AtomicU64; MAX_D], d: u32, n: u64) {
    c[(d as usize).min(MAX_D - 1)].fetch_add(n, Relaxed);
}

/// Euclidean point-to-box distance — `frustum::point_aabb_dist` is private and
/// this is a diagnostic, so it is recomputed rather than widening that API.
#[inline]
fn point_box_dist(p: Vec3A, b: &Aabb) -> f32 {
    (p.clamp(b.min, b.max) - p).length()
}

/// Per-axis direction interval over the tile's corner hull, plus its
/// reciprocal. An axis whose interval straddles zero is unusable (the
/// reciprocal spans infinity) and is neutralised rather than clamped: a `0.0`
/// reciprocal, never an infinity, so no `INF * 0.0` NaN can appear.
struct SlabIv {
    rlo: [f32; 3],
    rhi: [f32; 3],
    usable: [bool; 3],
    /// `min |v|` over the corner hull, in (0, 1]. The interval math lives in
    /// the unnormalised convex-combination parameterisation; a Euclidean
    /// distance needs this factor to stay a LOWER bound. 0 = unavailable.
    min_len: f32,
}

impl SlabIv {
    fn from_corners(corners: &[Vec3A; 4]) -> SlabIv {
        let c = corners.iter().fold(Vec3A::ZERO, |a, b| a + *b);
        let min_len = if c.length_squared() > 1e-12 {
            let axis = c.normalize();
            // |v| >= v·axis = sum(alpha_i (d_i·axis)) >= min_i (d_i·axis).
            let m = corners.iter().fold(f32::INFINITY, |m, d| m.min(d.dot(axis)));
            if m > 0.0 { m * (1.0 - 1e-5) } else { 0.0 }
        } else {
            0.0
        };
        let mut iv = SlabIv { rlo: [0.0; 3], rhi: [0.0; 3], usable: [false; 3], min_len };
        for a in 0..3 {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for c in corners {
                let v = c[a];
                lo = lo.min(v);
                hi = hi.max(v);
            }
            // A margin, not a bare sign test: a near-zero component makes the
            // reciprocal enormous and the interval useless anyway.
            const MU: f32 = 1e-6;
            if lo > MU || hi < -MU {
                let (a0, a1) = (1.0 / lo, 1.0 / hi);
                iv.rlo[a] = a0.min(a1);
                iv.rhi[a] = a0.max(a1);
                iv.usable[a] = true;
            }
        }
        iv
    }

    /// Conservative separation: true only if NO ray of the corner hull cone
    /// through `o` meets `b`.
    ///
    /// Interval arithmetic over the axis-independent box `[dmin, dmax]` is a
    /// superset of the hull, so the entry range lies below every ray's own
    /// entry and the exit range above every ray's own exit. The pairwise form
    /// `exists (a,b), a != b : E_a > X_b` reduces exactly to
    /// `max_a E_a > min_b X_b`, because interval arithmetic gives `E_a <= X_a`
    /// pointwise — two horizontal reductions, not a six-pair loop. Each ray's
    /// own values differ from these by one positive scale, which preserves the
    /// strict inequality, so a firing test proves a genuine miss.
    /// The beam's entry range against `b`, in the hull parameterisation.
    /// `(enter, exit)`; `enter > exit` is the separation proof.
    #[inline]
    fn range(&self, o: Vec3A, b: &Aabb) -> (f32, f32) {
        let ext = (b.min - o).abs().max((b.max - o).abs()).max_element();
        let s = 1e-5 * (1.0 + ext); // the outward slack `aabb_outside` grants;
                                    // inflating the box only weakens the reject
        let (mut enter, mut exit) = (f32::NEG_INFINITY, f32::INFINITY);
        for a in 0..3 {
            if !self.usable[a] {
                continue; // neutral in both reductions
            }
            let lo = b.min[a] - o[a] - s;
            let hi = b.max[a] - o[a] + s;
            let (p, q, r, t) =
                (lo * self.rlo[a], lo * self.rhi[a], hi * self.rlo[a], hi * self.rhi[a]);
            enter = enter.max(p.min(q).min(r.min(t)));
            exit = exit.min(p.max(q).max(r.max(t)));
        }
        (enter, exit)
    }

    /// Conservative separation: no ray of the hull cone meets the box.
    #[inline]
    fn rejects(&self, o: Vec3A, b: &Aabb) -> bool {
        let (enter, exit) = self.range(o, b);
        enter > exit
    }

    /// A frustum-aware LOWER bound on the Euclidean distance from the apex to
    /// any point of (box ∩ cone) — the cheap alternative to FR_RANGE's box
    /// clip. Clamp before scaling: `min_len <= 1` preserves a lower bound only
    /// for non-negative values. Returns 0 when unavailable, which is always a
    /// valid (if useless) lower bound.
    #[inline]
    fn entry_dist(&self, o: Vec3A, b: &Aabb) -> f32 {
        if self.min_len <= 0.0 {
            return 0.0;
        }
        let (enter, exit) = self.range(o, b);
        if enter > exit {
            return f32::INFINITY; // provably missed
        }
        self.min_len * enter.max(0.0)
    }
}

fn corners_of(cam: &CamBasis, x0: usize, y0: usize, x1: usize, y1: usize) -> [Vec3A; 4] {
    let (a, b, c, d) = (x0 as f32, y0 as f32, x1 as f32, y1 as f32);
    [cam.ray_dir(a, b), cam.ray_dir(c, b), cam.ray_dir(c, d), cam.ray_dir(a, d)]
}

/// Observe one tile that is about to subdivide. Answers Q1, Q4 and Q5.
///
/// Called after `refine_cut` has produced `cut`, so it sees exactly what the
/// children will inherit.
pub fn probe_split(
    cam: &CamBasis,
    scene: &Scene,
    bvh: &Bvh,
    f: &TileFrustum,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    depth: u32,
    cut: &[u32],
) {
    if !armed() || cut.is_empty() {
        return;
    }
    let px = ((x1 - x0) * (y1 - y0)) as u64;
    bump(&C.tiles, depth, 1);
    bump(&C.cut_len, depth, cut.len() as u64);

    let corners = corners_of(cam, x0, y0, x1, y1);
    let iv = SlabIv::from_corners(&corners);

    // ---- Q1: what the interval reject would remove from this cut.
    let mut rejected = 0u64;
    for &i in cut {
        if iv.rejects(f.origin, &bvh.nodes[i as usize].aabb) {
            rejected += 1;
        }
    }
    bump(&C.q1_entries, depth, cut.len() as u64);
    bump(&C.q1_rejects, depth, rejected);
    if rejected == cut.len() as u64 {
        // Every survivor proved missed => the bound query would have returned
        // None => this tile is sky and traces no rays at all.
        bump(&C.q1_sky_tiles, depth, 1);
        bump(&C.q1_sky_px, depth, px);
    }

    // ---- Q5: how much of the plane work a straddle mask would inherit away.
    // Ternary per plane: a box straddling BOTH opposing planes of an axis pair
    // spans the frustum there, so it straddles the separating plane too and
    // every analogous child plane. Planes arrive in perimeter order, so the
    // opposing pairs are (0,2) and (1,3).
    let mut sides = [0i32; 4];
    for &i in cut {
        let b = &bvh.nodes[i as usize].aabb;
        let mut k = 0usize;
        f.for_planes(|n, pad| {
            if k < 4 {
                // Positive and negative vertices: fully inside, fully outside,
                // or straddling.
                let pv = Vec3A::select(n.cmpge(Vec3A::ZERO), b.max, b.min);
                let nv = Vec3A::select(n.cmpge(Vec3A::ZERO), b.min, b.max);
                sides[k] = if n.dot(nv - f.origin) >= -pad {
                    1 // wholly inside
                } else if n.dot(pv - f.origin) < -pad {
                    -1 // wholly outside
                } else {
                    0 // straddling
                };
                k += 1;
            }
        });
        if k == 4 {
            bump(&C.q5_entries, depth, 1);
            if sides[0] == 0 && sides[2] == 0 {
                bump(&C.q5_straddle_x, depth, 1);
            }
            if sides[1] == 0 && sides[3] == 0 {
                bump(&C.q5_straddle_y, depth, 1);
            }
        }
    }

    // ---- Q4: Teller's exact covering test.
    if cut.len() > Q4_MAX_CUT {
        return;
    }
    bump(&C.q4_tested, depth, 1);
    let mut visits = 0u64;
    let mut tri: Option<u32> = None;
    let mut t_hi = 0.0f32;
    let mut same = true;
    for c in &corners {
        let ray = Ray::new(cam.origin, *c);
        match bvh.intersect(scene, &ray, 0.0, f32::INFINITY, &mut visits) {
            Some(h) => {
                t_hi = t_hi.max(h.t);
                match tri {
                    None => tri = Some(h.tri),
                    Some(t) if t == h.tri => {}
                    Some(_) => {
                        same = false;
                        break;
                    }
                }
            }
            None => {
                same = false;
                break;
            }
        }
    }
    let Some(tri) = tri.filter(|_| same) else { return };

    // The loose arm is an UPPER BOUND only: four corners agreeing does not by
    // itself prove that nothing else reaches the tile first.
    bump(&C.q4_loose, depth, 1);
    bump(&C.q4_loose_px, depth, px);

    // The SOUND arm is Teller's full condition — "the only **or
    // minimum-depth** element impinging". The second clause is what makes it
    // fire: other elements may impinge, they just must not be able to reach
    // any ray of the tile sooner than the covering triangle does.
    //
    // `t_hi` bounds the covering triangle's hit distance over the WHOLE tile,
    // not just the corners: in the unnormalised hull parameterisation `n·d` is
    // linear, so `t = n·(p0-o)/(n·d)` attains its extremes at corners, and a
    // unit interior direction has `|v| <= 1`, so its Euclidean `t` is at most
    // the unnormalised one. Hence every interior ray hits the triangle no
    // farther than `t_hi`, and any entry whose nearest point lies beyond
    // `t_hi` cannot get there first.
    // Measured BOTH ways, because which range is used is the whole question.
    // `apex` is today's frustum-blind `point_aabb_dist`; `beam` additionally
    // takes the slab-interval entry distance, which is frustum-aware. The
    // ground quad is the pathological case for the first and the reason the
    // blocked rate is ~99.9%: one flat slab whose nearest point sits directly
    // under the camera looks near to every tile, so it "interferes" with
    // everything.
    let (mut clear_apex, mut clear_beam) = (true, true);
    for &i in cut {
        let n = &bvh.nodes[i as usize];
        // Near enough to interfere — permitted only if this IS the leaf
        // holding the covering triangle. An overflow-emitted internal node
        // carries no triangle list, so it is conservatively an interferer.
        let holds = n.count > 0
            && bvh.tri_idx[n.left_first as usize..(n.left_first + n.leaf_count()) as usize]
                .contains(&tri);
        if holds {
            continue;
        }
        let apex = point_box_dist(f.origin, &n.aabb);
        if apex <= t_hi {
            clear_apex = false;
        }
        if apex.max(iv.entry_dist(f.origin, &n.aabb)) <= t_hi {
            clear_beam = false;
        }
        if !clear_apex && !clear_beam {
            break;
        }
    }
    if clear_apex {
        bump(&C.q4_fires, depth, 1);
        bump(&C.q4_px, depth, px);
    }
    if clear_beam {
        bump(&C.q4_beam, depth, 1);
        bump(&C.q4_beam_px, depth, px);
    }
}

/// Observe one shaded leaf tile. Answers Q3.
///
/// `t_max` is the farthest primary hit over the tile's own pixels, so a cut
/// entry whose nearest point lies beyond it could not have been hit by any ray
/// of this tile — it was carried down the whole ladder for nothing. That is the
/// operational form of "beyond the occlusion frontier".
pub fn probe_leaf(bvh: &Bvh, f: &TileFrustum, depth: u32, cut: &[u32], t_max: f32) {
    if !armed() || cut.is_empty() || !t_max.is_finite() {
        return;
    }
    bump(&C.q3_leaves, depth, 1);
    let mut beyond = 0u64;
    for &i in cut {
        if point_box_dist(f.origin, &bvh.nodes[i as usize].aabb) > t_max {
            beyond += 1;
        }
    }
    bump(&C.q3_entries, depth, cut.len() as u64);
    bump(&C.q3_beyond, depth, beyond);
}

fn sum(c: &[AtomicU64; MAX_D]) -> u64 {
    c.iter().map(|v| v.load(Relaxed)).sum()
}

fn pct(n: u64, d: u64) -> f32 {
    if d == 0 { 0.0 } else { 100.0 * n as f32 / d as f32 }
}

/// Called once per hybrid frame; prints a cumulative table every
/// `REPORT_EVERY` frames (so a one-frame `--check` prints exactly once).
pub fn frame_end(rw: usize, rh: usize) {
    if !armed() {
        return;
    }
    let n = FRAMES.fetch_add(1, Relaxed) + 1;
    if n % REPORT_EVERY != 0 && n != 1 {
        return;
    }
    let px = (rw * rh) as u64 * n;
    let (tiles, cut_len) = (sum(&C.tiles), sum(&C.cut_len));
    eprintln!("\n=== oracle (FR_ORACLE) after {n} hybrid frame(s), {rw}x{rh} ===");
    eprintln!(
        "  tiles {tiles} | cut mean {:.2} | screen px {px}",
        if tiles == 0 { 0.0 } else { cut_len as f32 / tiles as f32 }
    );
    eprintln!(
        "  Q1 slab reject : entries {}/{} ({:.1}%) | whole-cut => NEW SKY {} tiles, {} px ({:.2}% of screen)",
        sum(&C.q1_rejects),
        sum(&C.q1_entries),
        pct(sum(&C.q1_rejects), sum(&C.q1_entries)),
        sum(&C.q1_sky_tiles),
        sum(&C.q1_sky_px),
        pct(sum(&C.q1_sky_px), px)
    );
    eprintln!(
        "  Q3 frontier    : {}/{} cut entries beyond the tile's own max hit ({:.1}%) over {} leaf tiles",
        sum(&C.q3_beyond),
        sum(&C.q3_entries),
        pct(sum(&C.q3_beyond), sum(&C.q3_entries)),
        sum(&C.q3_leaves)
    );
    eprintln!(
        "  Q4 covering    : tested {} | apex-range {} tiles / {:.2}% px | BEAM-range {} tiles / {:.2}% px | loose upper {} tiles / {:.2}% px",
        sum(&C.q4_tested),
        sum(&C.q4_fires),
        pct(sum(&C.q4_px), px),
        sum(&C.q4_beam),
        pct(sum(&C.q4_beam_px), px),
        sum(&C.q4_loose),
        pct(sum(&C.q4_loose_px), px)
    );
    eprintln!(
        "  Q5 straddle    : {:.1}% of cut entries straddle both x-planes, {:.1}% both y-planes ({} entries)",
        pct(sum(&C.q5_straddle_x), sum(&C.q5_entries)),
        pct(sum(&C.q5_straddle_y), sum(&C.q5_entries)),
        sum(&C.q5_entries)
    );
    // PER DEPTH, and the pixel columns are only meaningful here: tiles at one
    // depth are disjoint, but a parent's pixels CONTAIN its children's, so
    // summing a pixel count across depths double-counts (it read 159% of the
    // screen on San Miguel before this was fixed).
    //
    // Both Q1 and Q4 are monotone downward — a child's cone is a subset of its
    // parent's, so anything proved for the parent still holds — which makes
    // **the deepest split row the screen fraction to quote**: every pixel
    // inside a firing tile higher up is inside one there too.
    eprintln!("  per depth — tiles | cut | Q1 rej% | Q1 new-sky px% | Q4 beam px% | Q5 straddle-x%:");
    for d in 0..MAX_D {
        let t = C.tiles[d].load(Relaxed);
        if t == 0 {
            continue;
        }
        let f = (rw * rh) as u64 * FRAMES.load(Relaxed).max(1);
        eprintln!(
            "    d{d:<2} tiles {t:<8} cut {:<6.2} Q1 {:<5.1} sky {:<6.2} Q4 {:<6.2} ({} tiles) Q5x {:.1}",
            C.cut_len[d].load(Relaxed) as f32 / t as f32,
            pct(C.q1_rejects[d].load(Relaxed), C.q1_entries[d].load(Relaxed)),
            pct(C.q1_sky_px[d].load(Relaxed), f),
            pct(C.q4_beam_px[d].load(Relaxed), f),
            C.q4_beam[d].load(Relaxed),
            pct(C.q5_straddle_x[d].load(Relaxed), C.q5_entries[d].load(Relaxed)),
        );
    }
    eprintln!(
        "  (quote the DEEPEST split row for screen coverage; the whole-run px% above \
         double-counts across depths and is an upper bound only)"
    );
}
