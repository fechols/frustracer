//! Temporal frustum cache: the previous frame's quadtree as an acceleration
//! structure for the current one.
//!
//! Every internal quadtree node proves `frustum ∩ ball(origin, tc)` empty of
//! geometry (sky nodes prove their whole semi-infinite cone empty). With a
//! static scene those proofs outlive the frame, so we cache one distance per
//! node plus the frame's `CamBasis` and, next frame, harvest a `t_start` head
//! start for each tile from an enclosing old frustum — pure cone geometry,
//! no BVH traversal.
//!
//! Every entry is a *standalone* world-space claim: "no geometry on any
//! frustum ray at parameter <= value" ≡ "frustum ∩ ball(origin, value) empty"
//! (a frustum is the union of its apex rays). That holds even for entries that
//! were themselves built on a temporal seed — each cross-frame hop only
//! shrinks the claim (δ subtraction, 1e-4 shrink), never grows it, so error
//! cannot accumulate in the unsafe direction.

use crate::camera::CamBasis;
use crate::render::LEAF_TILE;
use crate::stats::LocalStats;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// How many old ancestor levels the containment walk tries per tile. A tile
/// shares boundary planes with its parent (exact-zero dots, rejected by the
/// strict margin), so translation passes typically land 2–3 levels up; deeper
/// walks are dominated by what the tile already inherited from its parent.
pub const WALK_MAX: u32 = 4;

/// One frame's quadtree bounds: a flat linear quadtree of f32 bit patterns.
/// NaN = node not evaluated this frame, +INF = sky (whole cone empty),
/// finite = the node's `tc` (proven-empty ball radius).
///
/// Written with relaxed atomics — each node is written by exactly one rayon
/// task per frame and read only on the *next* frame, across the thread-pool
/// join (same happens-before argument as the accum buffer).
pub struct TemporalCache {
    nodes: Vec<AtomicU32>,
    /// offsets[d] = (4^d − 1) / 3, the start of level d.
    offsets: Vec<usize>,
    /// Deepest cached level (inclusive) — the deepest depth `tile_step` can
    /// run at for this resolution.
    max_depth: u32,
}

impl TemporalCache {
    pub fn new(rw: usize, rh: usize) -> Self {
        // Internal nodes exist while a tile is bigger than LEAF_TILE; the
        // deepest tile_step depth is ceil(log2(max(rw,rh)/LEAF_TILE)) − 1, but
        // allocate through the leaf depth — store() bounds-checks anyway and
        // odd resolutions can go one level deeper on some branches.
        let max_depth = ((rw.max(rh) as f32) / LEAF_TILE as f32).log2().ceil() as u32;
        let mut offsets = Vec::with_capacity(max_depth as usize + 2);
        let mut total = 0usize;
        for d in 0..=max_depth + 1 {
            offsets.push(total);
            total += 1usize << (2 * d);
        }
        let nodes = (0..total).map(|_| AtomicU32::new(f32::NAN.to_bits())).collect();
        TemporalCache { nodes, offsets, max_depth }
    }

    /// Reset every entry to "not evaluated" for a new frame.
    pub fn clear(&self) {
        for n in &self.nodes {
            n.store(f32::NAN.to_bits(), Relaxed);
        }
    }

    pub fn store(&self, depth: u32, path: u32, t: f32) {
        if depth <= self.max_depth {
            self.nodes[self.offsets[depth as usize] + path as usize].store(t.to_bits(), Relaxed);
        }
    }

    fn load(&self, depth: u32, path: u32) -> f32 {
        if depth <= self.max_depth {
            f32::from_bits(self.nodes[self.offsets[depth as usize] + path as usize].load(Relaxed))
        } else {
            f32::NAN
        }
    }
}

/// The result of a temporal probe for one tile.
pub enum Seed {
    /// Nothing usable — trace exactly as before.
    None,
    /// Proven-empty distance: raise the tile's t_start to this (if larger).
    T(f32),
    /// The tile's whole frustum is proven empty — fill sky, zero BVH work.
    Sky,
}

/// Probe the previous frame's cache for a head start for the tile at
/// (rect, depth, path), whose rays already carry the inherited `t_start`.
///
/// Identical basis (static camera): the same node's entry applies verbatim —
/// same frustum, same origin, no shrink, no geometry. This is what makes
/// accumulation frames cheap; under motion same-depth containment can never
/// pass (shared boundary planes give exact-zero dots).
///
/// Moving camera — the segment decomposition. A new-ray point p(s) = o₁ + s·d
/// must be proven empty for every s in (0, seed]:
/// - s ≤ t_start is covered by the tile's own inherited claim
///   (F_new ∩ ball(o₁, t_start) is empty — a standalone claim of this frame).
/// - s in [t_start, seed] is covered by the OLD node's claim if p is inside
///   the old frustum and ball. Relative to the old origin,
///   p − o₀ = s·(d + λ·t̂) with λ = δ/s ranging over [δ/seed, δ/t_start], so
///   all such points lie in the conic hull of the 8 directions
///   {dⱼ + λ_min·t̂, dⱼ + λ_max·t̂} over the 4 tile corners (bilinear in
///   (λ, corner weights) — extremes at the parameter-box corners). A pure
///   direction test, no apex condition. The ball part is
///   |p − o₀| ≤ s + δ ≤ t₀, giving seed = (t₀ − δ)·(1 − 1e-4).
///
/// Both bounds of λ matter (found empirically by the --check dolly pass):
/// - No apex condition: naive frustum-in-frustum containment requires the
///   translation direction inside the old node's view cone — for a forward
///   dolly that is the exact screen center, which is the quadtree's root
///   split corner and lies on cell boundaries at EVERY depth, so nothing
///   would ever pass.
/// - λ_min > 0, not 0: the raw corner dirs are the s → ∞ limit, which the
///   segment never reaches (it ends at seed). Testing them reintroduces
///   exact-zero dots on every plane the tile shares with an old cell —
///   e.g. the old root genuinely contains the dolly-translated root segment,
///   and only the λ_min-tilted test can prove it. Sky candidates have no ball
///   bound (seed = ∞), so λ_min = 0 falls out naturally for them.
/// - t_start = 0 (the root): the segment starts at the apex, λ_max = ∞ —
///   degenerates to testing t̂ itself alongside the λ_min-tilted corners.
///
/// The walk tests old nodes along the reprojected point's quadtree path from
/// the tile's own depth upward (an old *deeper* cell is angularly smaller and
/// can never contain the tile). First pass wins: tc is monotone nondecreasing
/// with depth along the old chain and containment is upward-closed, so the
/// deepest pass is the best available bound. Finite candidates whose seed
/// cannot beat the inherited t_start are skipped without a test — in
/// blocked-dominated scenes that is most of them.
pub fn lookup(
    prev: &TemporalCache,
    prev_cam: &CamBasis,
    cam: &CamBasis,
    rw: usize,
    rh: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    t_start: f32,
    depth: u32,
    path: u32,
    ls: &mut LocalStats,
) -> Seed {
    if prev_cam == cam {
        let t = prev.load(depth, path);
        if t.is_nan() {
            return Seed::None;
        }
        return if t == f32::INFINITY { Seed::Sky } else { Seed::T(t) };
    }

    // Reproject the tile's center direction into the old camera's screen.
    let dc = cam.ray_dir((x0 + x1) as f32 * 0.5, (y0 + y1) as f32 * 0.5);
    let (fx, fy) = match prev_cam.project(dc) {
        Some(p) => p,
        None => return Seed::None,
    };
    if fx < 0.0 || fx >= rw as f32 || fy < 0.0 || fy >= rh as f32 {
        return Seed::None; // center left the old view — nothing can contain us
    }

    // Old-tree path containing (fx, fy) down to this tile's depth. This
    // replays trace_tile's integer midpoint splits exactly (`xm = x0 + w/2`,
    // quadrants TL=0 TR=1 BL=2 BR=3) and MUST stay in lockstep with them:
    // only then does a cached path map back to the identical rect — and, with
    // the cached basis, the bit-identical frustum — it was traced with.
    let lo = depth.saturating_sub(WALK_MAX);
    let mut paths = [0u32; 16];
    let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
    debug_assert!((depth as usize) < paths.len());
    let (mut ox0, mut oy0, mut ox1, mut oy1) = (0usize, 0usize, rw, rh);
    let mut opath = 0u32;
    rects[0] = (ox0, oy0, ox1, oy1);
    for k in 1..=depth {
        let xm = ox0 + (ox1 - ox0) / 2;
        let ym = oy0 + (oy1 - oy0) / 2;
        let qx = (fx >= xm as f32) as u32;
        let qy = (fy >= ym as f32) as u32;
        if qx == 0 { ox1 = xm } else { ox0 = xm }
        if qy == 0 { oy1 = ym } else { oy0 = ym }
        opath = (opath << 2) | (qy << 1) | qx;
        if k >= lo {
            paths[k as usize] = opath;
            rects[k as usize] = (ox0, oy0, ox1, oy1);
        }
    }

    // Corner dirs of the new tile — same construction tile_frustum uses, so
    // the cone we test is exactly the cone the tile traces. When the camera
    // translated, each candidate tests the tilted copies from the segment
    // decomposition (see the doc comment); pure rotation (δ = 0) needs the
    // corners only. δ is inflated for its own f32 rounding (relative — both
    // origins are the exact values their frames traced with); using the
    // inflated value for λ_max and the plain value for λ_min only widens the
    // tested cone, which is the safe direction.
    let corners = [
        cam.ray_dir(x0 as f32, y0 as f32),
        cam.ray_dir(x1 as f32, y0 as f32),
        cam.ray_dir(x1 as f32, y1 as f32),
        cam.ray_dir(x0 as f32, y1 as f32),
    ];
    let delta = (cam.origin - prev_cam.origin).length();
    let delta_safe = delta + 1e-5 * delta;
    let t_hat = if delta > 0.0 {
        (cam.origin - prev_cam.origin) * (1.0 / delta)
    } else {
        glam::Vec3A::ZERO
    };

    for k in (lo..=depth).rev() {
        let t_old = prev.load(k, paths[k as usize]);
        if t_old.is_nan() {
            continue; // below an old sky/capped node — keep climbing
        }
        let seed = if t_old == f32::INFINITY {
            f32::INFINITY
        } else {
            ((t_old - delta_safe) * (1.0 - 1e-4)).max(0.0)
        };
        if seed <= t_start {
            continue; // can't beat the inherited bound — not worth a test
        }
        let mut dirs = [glam::Vec3A::ZERO; 8];
        let ndirs = if delta > 0.0 {
            let lmin = delta / seed; // 0 for sky — its claim has no far end
            for j in 0..4 {
                dirs[j] = (corners[j] + t_hat * lmin).normalize();
            }
            if t_start > 0.0 {
                let lmax = delta_safe / t_start;
                for j in 0..4 {
                    dirs[4 + j] = (corners[j] + t_hat * lmax).normalize();
                }
                8
            } else {
                // Segment starts at the apex: λ_max = ∞ is t̂ itself.
                dirs[4] = t_hat;
                5
            }
        } else {
            dirs[..4].copy_from_slice(&corners);
            4
        };
        let (rx0, ry0, rx1, ry1) = rects[k as usize];
        let f_old = prev_cam.tile_frustum(rx0, ry0, rx1, ry1);
        ls.temporal_tests += 1;
        if f_old.contains_dirs(&dirs[..ndirs]) {
            if t_old == f32::INFINITY {
                // Old node proved its whole cone empty (query None composed
                // with its inherited ball claim): every tested segment point
                // is inside that cone at *some* distance — all empty. The
                // [0, t_start] prefix is the inherited claim's, as always.
                return Seed::Sky;
            }
            return Seed::T(seed);
        }
    }
    Seed::None
}
