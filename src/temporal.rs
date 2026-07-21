//! Temporal frustum cache: the previous frame's quadtree as an acceleration
//! structure for the current one.
//!
//! Every internal quadtree node proves `frustum ∩ ball(origin, tc)` empty of
//! geometry (sky nodes prove their whole semi-infinite cone empty). With a
//! static scene those proofs outlive the frame, so we cache one distance per
//! node plus the frame's `CamBasis` and, next frame, harvest a `t_start` head
//! start for each tile — pure projective geometry, no BVH traversal.
//!
//! Every entry is a *standalone* world-space claim: "no geometry on any
//! frustum ray at parameter <= value" ≡ "frustum ∩ ball(origin, value) empty"
//! (a frustum is the union of its apex rays). That holds even for entries that
//! were themselves built on a temporal seed — each cross-frame hop only
//! shrinks the claim, never grows it, so error cannot accumulate in the
//! unsafe direction.
//!
//! The old quadtree PARTITIONS the direction sphere (within the old root
//! cone) into frontier cells — the deepest non-NaN entry over each screen
//! region. The consumer query is a *region min*: find every old cell the new
//! tile's (tilt-widened) direction cone overlaps and take the minimum claim.
//! This is what lets rotation reuse the cache: a panned tile straddling four
//! old cells gets the min of four same-depth-quality bounds, where a
//! containment-in-one-cell test (the v1 design) got nothing.
//!
//! The consumer holds a RING of previous caches (newest first, each with its
//! own basis), because claims never go stale in a static scene — a region
//! that pans off the newest screen and back still has claims in an older
//! entry. `lookup` retries older entries only on misses whose cause is
//! pose-specific (off-screen, behind-plane, blown budget); a completed
//! not-useful min stops the scan — it is pinned by real geometry that no
//! older cache can claim past.

use crate::camera::CamBasis;
use crate::frustum::MAX_CUT;
use crate::render::leaf_tile;
use crate::stats::LocalStats;
use glam::Vec3A;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

/// Outward padding (px) on the query bbox before selecting overlapped cells.
/// Errs inclusive — extra cells only lower the min — and covers projection /
/// bbox fp error (~1e-4 px) with two orders of margin.
const PAD_Q: f32 = 0.5;

/// Screen-acceptance tolerance (px): an extreme direction projecting farther
/// than this outside the old screen means part of the tile's tail is covered
/// by NO old claim → no reuse. Polarity is the OPPOSITE of PAD_Q: accepting
/// too much is unsound. Must stay well below the producer's `aabb_outside`
/// outward slack (~6e-3 px at this scale — old claims extend that far past
/// the exact cone) and above projection round-trip noise (~4e-4 px). It is
/// load-bearing: under pure forward dolly the new corner dirs are
/// bit-identical to the old ones and project exactly onto the old screen
/// boundary, so a strict test would reject the root — the biggest seed.
const ACCEPT_PAD: f32 = 1e-3;

/// Cells a single region-min query may visit. Degenerate queries (tiny
/// t_start → huge tilt → near-screen-sized bbox) hit this and give up.
/// Exceeding the budget must surface as "no reuse", NEVER as Sky — an
/// unfinished scan's running min is not a valid claim.
const VISIT_BUDGET: u32 = 128;

/// One frame's quadtree bounds: a flat linear quadtree of f32 bit patterns.
/// NaN = node not evaluated this frame, +INF = sky (whole cone empty),
/// finite = the node's `tc` (proven-empty ball radius).
///
/// Written with relaxed atomics — each node is written by exactly one rayon
/// task per frame and read only on the *next* frame, across the thread-pool
/// join (same happens-before argument as the accum buffer).
///
/// Frontier invariants the region-min query depends on (writer side):
/// entries are stored BEFORE recursing into children, sky stores return
/// without recursing, and the buffer is cleared before every producing frame
/// — so a NaN entry implies its whole subtree is NaN (the parent is the
/// frontier there), and a +INF entry has an all-NaN subtree.
pub struct TemporalCache {
    nodes: Vec<AtomicU32>,
    /// offsets[d] = (4^d − 1) / 3, the start of level d.
    offsets: Vec<usize>,
    /// Deepest cached level (inclusive) — the deepest depth `tile_step` can
    /// run at for this resolution.
    pub max_depth: u32,
}

impl TemporalCache {
    pub fn new(rw: usize, rh: usize) -> Self {
        // Internal nodes exist while a tile is bigger than LEAF_TILE; the
        // deepest tile_step depth is ceil(log2(max(rw,rh)/LEAF_TILE)) − 1, but
        // allocate through the leaf depth — store() bounds-checks anyway and
        // odd resolutions can go one level deeper on some branches.
        let max_depth = ((rw.max(rh) as f32) / leaf_tile() as f32).log2().ceil() as u32;
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

/// Maximum consecutive adoption hops before a node is forced back onto a
/// real bound query (which re-extends the claim and re-anchors the cut at
/// age 0). Claims shrink by δ per hop and adopted cuts inherit the previous
/// pose's culling, so unbounded chains decay quality — never correctness.
/// Must stay < 7: age 7 would collide with CutStore's u64::MAX sentinel.
const MAX_ADOPT_AGE: u32 = 3;

/// Cut-arena budget in u32 per terminal-count unit (see `CutStore::new`).
const CUT_ARENA_PER_TERMINAL: usize = 8;

/// One producing frame's refined node cuts: for each Split quadtree node,
/// the exact `refine_cut` output it handed its children — a per-cone PVS of
/// every BVH subtree with geometry inside that cone (frustum-outside drops
/// are pure geometry, proven-empty-ball drops contain nothing inside the
/// cone, and the primary far-clamp is INFINITY, which never drops). Paired
/// with exactly one TemporalCache generation and consumed only from the
/// ring's NEWEST entry: a consumer whose direction hull is contained in the
/// node's cone may skip its bound query and re-refine this cut instead.
///
/// Same relaxed-atomic discipline as TemporalCache: each slot written by one
/// task per frame, read only on a later frame across the pool join.
pub struct CutStore {
    /// One packed word per linear-quadtree slot (TemporalCache's offsets
    /// layout): off:32 | (len−1):6 | age:3. u64::MAX = nothing stored —
    /// unreachable as a real value because age 7 is never written. The
    /// 32-bit off field covers any arena `top` (an AtomicU32) can address —
    /// resolution can never outgrow the packing (a 23-bit off was tried and
    /// silently truncated past ~16.8M window pixels: adopted tiles refined
    /// OTHER nodes' cuts and false-skied — the 8K corruption bug).
    idx: Vec<AtomicU64>,
    arena: Vec<AtomicU32>,
    top: AtomicU32,
    offsets: Vec<usize>,
    pub max_depth: u32,
}

impl CutStore {
    pub fn new(rw: usize, rh: usize) -> Self {
        // Mirror TemporalCache's depth/offsets exactly — a cut is stored at
        // the same (depth, path) key as its node's claim.
        let max_depth = ((rw.max(rh) as f32) / leaf_tile() as f32).log2().ceil() as u32;
        let mut offsets = Vec::with_capacity(max_depth as usize + 2);
        let mut total = 0usize;
        for d in 0..=max_depth + 1 {
            offsets.push(total);
            total += 1usize << (2 * d);
        }
        // Arena budget: Split nodes number < terminals (branching ≥ 2), and
        // terminals ≤ ceil(rw/4)·ceil(rh/4) (replay.rs's bound); the mean cut
        // is ~10 on the default scene. Overflow stores nothing — the
        // consumer just misses one adoption; counted, never wrong. `top` is
        // an AtomicU32 and the packed off field holds all 32 of its bits, so
        // the only hard cap is the bump cursor's own width.
        let arena_len = rw.div_ceil(4) * rh.div_ceil(4) * CUT_ARENA_PER_TERMINAL;
        assert!(arena_len <= u32::MAX as usize, "cut arena exceeds the u32 bump cursor");
        CutStore {
            idx: (0..total).map(|_| AtomicU64::new(u64::MAX)).collect(),
            arena: (0..arena_len).map(|_| AtomicU32::new(0)).collect(),
            top: AtomicU32::new(0),
            offsets,
            max_depth,
        }
    }

    /// Reset for a new producing frame.
    pub fn clear(&self) {
        for n in &self.idx {
            n.store(u64::MAX, Relaxed);
        }
        self.top.store(0, Relaxed);
    }

    /// Store a node's refine_cut output. Returns false when the arena is
    /// full (nothing stored — the caller counts it).
    pub fn store(&self, depth: u32, path: u32, cut: &[u32], age: u32) -> bool {
        debug_assert!(!cut.is_empty() && cut.len() <= MAX_CUT && age < 7);
        if depth > self.max_depth {
            return true;
        }
        let off = self.top.fetch_add(cut.len() as u32, Relaxed) as usize;
        if off + cut.len() > self.arena.len() {
            return false;
        }
        for (k, &v) in cut.iter().enumerate() {
            self.arena[off + k].store(v, Relaxed);
        }
        let packed = ((off as u64) << 9) | (((cut.len() - 1) as u64) << 3) | age as u64;
        self.idx[self.offsets[depth as usize] + path as usize].store(packed, Relaxed);
        true
    }

    /// (arena offset, cut length, adoption-chain age) for a stored node.
    fn load(&self, depth: u32, path: u32) -> Option<(u32, u32, u32)> {
        if depth > self.max_depth {
            return None;
        }
        let v = self.idx[self.offsets[depth as usize] + path as usize].load(Relaxed);
        if v == u64::MAX {
            return None;
        }
        Some(((v >> 9) as u32, ((v >> 3) & 63) as u32 + 1, (v & 7) as u32))
    }

    /// Copy a stored cut into the caller's stack buffer; returns its length.
    pub fn copy_cut(&self, off: u32, len: u32, out: &mut [u32; MAX_CUT]) -> usize {
        copy_cut_from(&self.arena, off, len, out)
    }
}

/// Copy a cut out of a bump-allocated atomic arena into the caller's stack
/// buffer; returns its length. Shared by `CutStore` and `replay::ReplayCache`
/// — the two cut arenas — so bounds handling lives in one place.
pub fn copy_cut_from(arena: &[AtomicU32], off: u32, len: u32, out: &mut [u32; MAX_CUT]) -> usize {
    let (off, len) = (off as usize, len as usize);
    debug_assert!(len <= MAX_CUT && off + len <= arena.len());
    for k in 0..len {
        out[k] = arena[off + k].load(Relaxed);
    }
    len
}

/// The result of a temporal probe for one tile.
pub enum Seed {
    /// Nothing usable — trace exactly as before.
    None,
    /// Proven-empty distance: raise the tile's t_start to this (if larger).
    T(f32),
    /// The newest cache's region-min COMPLETED at a finite value: real
    /// geometry pins the cone at ~m, so this tile's bound query cannot
    /// meaningfully advance (the "blocked" regime, the most common tile
    /// outcome) — SKIP it. Raise t_start to `t` if larger and refine the
    /// tile's own inherited cut at that distance; no old data is consumed by
    /// the refine, so no containment condition exists — the skip is sound
    /// unconditionally and the min is merely the predictor of when it is
    /// profitable. `age` chains the skips so MAX_ADOPT_AGE forces a real
    /// query before the un-advanced claims decay (each hop shrinks by δ).
    Skip { t: f32, age: u32 },
    /// Identical basis only: the node's OWN old cut is exactly valid (same
    /// cone) — skip the bound query AND refine from the stored cut
    /// (`CutStore::copy_cut(off, len)`), which is already this node's
    /// tightness rather than its parent's.
    TCut { t: f32, off: u32, len: u32, age: u32 },
    /// The tile's whole frustum is proven empty — fill sky, zero BVH work.
    Sky,
}

/// One cache's answer, with the *reason* for a miss — the ring wrapper's
/// retry policy keys on it.
enum Probe {
    Sky,
    T(f32),
    Skip { t: f32, age: u32 },
    TCut { t: f32, off: u32, len: u32, age: u32 },
    Miss(Miss),
}

enum Miss {
    /// An extreme direction projects behind the old image plane.
    BehindPlane,
    /// An extreme lands off the old screen — part of the tail has no claim
    /// in THIS cache; an older cache with a different pose may cover it.
    OffScreen,
    /// Region-min visit budget blown (unfinished scan — no claim). A
    /// different pose shapes a different query region; retrying is cheap
    /// and bounded at ring-len × VISIT_BUDGET.
    Budget,
    /// The min completed but `seed <= t_start`. The min is pinned by real
    /// nearest geometry along these directions — an older cache faces the
    /// same geometry and cannot claim past it. The ring must STOP here.
    NotUseful,
    /// Unreachable for a validly-produced buffer; skip the entry.
    NanRoot,
}

/// Probe a ring of previous-frame caches (newest first, each paired with the
/// exact basis it was traced with) for the tile's head start. First useful
/// answer wins; misses retry older entries per the `Miss` policy. Every
/// entry is a standalone world-space claim (static scene — age cannot stale
/// it), and each probe is the complete, independent segment-decomposition
/// query below, so "None, never Sky" survives the retries untouched.
#[allow(clippy::too_many_arguments)]
pub fn lookup(
    ring: &[(&TemporalCache, CamBasis)],
    cuts: Option<&CutStore>,
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
    for (age, (prev, prev_cam)) in ring.iter().enumerate() {
        let hit = |ls: &mut LocalStats| {
            if age > 0 {
                ls.temporal_ring_hits += 1;
                ls.temporal_ring_age_sum += age as u64;
            }
        };
        // Cut adoption consults only the NEWEST entry: older poses have
        // larger δ (their hulls rarely stay contained) and strictly staler
        // cuts, and the CutStore is paired with the newest cache generation.
        let entry_cuts = if age == 0 { cuts } else { None };
        match probe_one(prev, prev_cam, cam, entry_cuts, rw, rh, x0, y0, x1, y1, t_start, depth, path, ls) {
            Probe::Sky => {
                hit(ls);
                return Seed::Sky;
            }
            Probe::T(t) => {
                hit(ls);
                return Seed::T(t);
            }
            Probe::Skip { t, age } => return Seed::Skip { t, age },
            Probe::TCut { t, off, len, age } => return Seed::TCut { t, off, len, age },
            Probe::Miss(Miss::NotUseful) => return Seed::None,
            Probe::Miss(_) => {}
        }
    }
    Seed::None
}

/// Probe the previous frame's cache for a head start for the tile at
/// (rect, depth, path), whose rays already carry the inherited `t_start`.
///
/// Identical basis (static camera): the same node's entry applies verbatim —
/// same frustum, same origin, no shrink, no geometry. A NaN entry (below an
/// old sky or depth-capped node) falls through to the general query, which
/// finds the covering frontier ancestor.
///
/// Moving camera — the segment decomposition. A new-ray point p(s) = o₁ + s·d
/// must be proven empty for every s in (0, seed]:
/// - s ≤ t_start is covered by the tile's own inherited claim
///   (F_new ∩ ball(o₁, t_start) is empty — a standalone claim of this frame).
/// - s in [t_start, seed]: relative to the old origin the point's direction
///   is ∝ d + λ·t̂ with λ = δ/s ∈ (0, λ_max], λ_max = δ_safe/t_start — inside
///   the conic hull of {corners} ∪ {corners + λ_max·t̂} (t_start == 0 makes
///   λ_max = ∞, i.e. t̂ itself). No apex condition exists (requiring the new
///   apex inside one old cone would demand the translation direction inside
///   it — for a forward dolly that is the screen center, a cell corner at
///   every depth, so nothing would ever pass).
///
/// Coverage query: `CamBasis::project` of a positive combination of
/// directions is an exact CONVEX combination of their projections (weights
/// αᵢ(dᵢ·forward) — a gnomonic-projection identity; also scale-invariant, so
/// the tilted dirs need no normalization). Hence projecting just the extreme
/// dirs and padding their bbox conservatively covers the projection of every
/// segment direction. Every old frontier cell whose rect intersects that box
/// contributes its claim to the min m; each segment point then lies in some
/// contributing cell's cone with |p − o₀| ≤ s + δ ≤ m ≤ its tc — inside its
/// proven cone∩ball. All contributions +INF ⇒ the whole tail is inside
/// fully-empty cones ⇒ Sky.
///
/// seed = (m − δ_safe)·(1 − 1e-4) for δ > 0 (the shrink protects only the δ
/// subtraction); for δ = 0 the origins are bitwise equal and the claim
/// transfers exactly: seed = m, no shrink — same as the verbatim path.
#[allow(clippy::too_many_arguments)]
fn probe_one(
    prev: &TemporalCache,
    prev_cam: &CamBasis,
    cam: &CamBasis,
    cuts: Option<&CutStore>,
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
) -> Probe {
    if prev_cam == cam {
        let t = prev.load(depth, path);
        if !t.is_nan() {
            if t == f32::INFINITY {
                return Probe::Sky;
            }
            // Identical basis ⇒ identical cone: the node's OWN old cut is
            // exactly valid here — the strongest adoption (age-gated so a
            // chain still re-queries).
            if let Some(cs) = cuts {
                match cs.load(depth, path) {
                    Some((off, len, age)) if age < MAX_ADOPT_AGE => {
                        return Probe::TCut { t, off, len, age };
                    }
                    Some(_) => ls.temporal_adopt_requery += 1,
                    None => {}
                }
            }
            return Probe::T(t);
        }
        // NaN: below an old sky/capped node — the general query below finds
        // the covering frontier ancestor (δ = 0 → exact transfer).
    }

    // Extreme directions of the segment's direction hull, projected into the
    // old screen. Corners use the same construction as tile_frustum — that
    // identity is what makes the coverage exact.
    let corners = [
        cam.ray_dir(x0 as f32, y0 as f32),
        cam.ray_dir(x1 as f32, y0 as f32),
        cam.ray_dir(x1 as f32, y1 as f32),
        cam.ray_dir(x0 as f32, y1 as f32),
    ];
    let delta = (cam.origin - prev_cam.origin).length();
    // Inflate δ for its own f32 rounding (relative — both origins are the
    // exact values their frames traced with). Using the inflated value for
    // λ_max only widens the tested hull: the safe direction.
    let delta_safe = delta + 1e-5 * delta;
    let mut dirs = [Vec3A::ZERO; 8];
    dirs[..4].copy_from_slice(&corners);
    let ndirs = if delta > 0.0 {
        let t_hat = (cam.origin - prev_cam.origin) * (1.0 / delta);
        if t_start > 0.0 {
            let lmax = delta_safe / t_start;
            for j in 0..4 {
                dirs[4 + j] = corners[j] + t_hat * lmax; // unnormalized: project is scale-invariant
            }
            8
        } else {
            dirs[4] = t_hat; // λ_max = ∞ — the segment starts at the apex
            5
        }
    } else {
        4
    };

    let (mut bx0, mut by0, mut bx1, mut by1) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for d in &dirs[..ndirs] {
        let (fx, fy) = match prev_cam.project(*d) {
            Some(p) => p,
            None => return Probe::Miss(Miss::BehindPlane), // behind the old image plane
        };
        if fx < -ACCEPT_PAD || fx > rw as f32 + ACCEPT_PAD || fy < -ACCEPT_PAD || fy > rh as f32 + ACCEPT_PAD {
            return Probe::Miss(Miss::OffScreen); // part of the tail has no old claim
        }
        bx0 = bx0.min(fx);
        by0 = by0.min(fy);
        bx1 = bx1.max(fx);
        by1 = by1.max(fy);
    }
    // Pad outward (inclusive-erring, sound), clamp to the screen for
    // traversal only — cells tile exactly [0,rw]×[0,rh].
    let q = (
        (bx0 - PAD_Q).max(0.0),
        (by0 - PAD_Q).max(0.0),
        (bx1 + PAD_Q).min(rw as f32),
        (by1 + PAD_Q).min(rh as f32),
    );

    let root_t = prev.load(0, 0);
    if root_t.is_nan() {
        // Never "contribute nothing": an empty min would read +INF = false
        // sky. (Unreachable for a validly-produced prev buffer.)
        return Probe::Miss(Miss::NanRoot);
    }
    // Exact usefulness threshold: m <= bail ⇔ seed <= t_start. Once the
    // running min crosses it, neither a useful seed nor Sky is possible.
    let bail = if delta > 0.0 { t_start / (1.0 - 1e-4) + delta_safe } else { t_start };
    // Descend at most one level past the tile's own depth: an ancestor's
    // claim is always valid for its whole region (tc is monotone
    // nondecreasing down the old chain), so any cap is sound — deeper is
    // only tighter. Measured: +2 resolved no additional sky in the check
    // scene (the limiter is the real sky-boundary geometry, not frontier
    // resolution) and cost ~3% more cells.
    let cap = (depth + 1).min(prev.max_depth);
    let mut budget = VISIT_BUDGET;
    ls.temporal_tests += 1; // the root load
    let m = match region_min(prev, root_t, 0, 0, (0, 0, rw, rh), q, cap, bail, &mut budget, ls) {
        Some(m) => m,
        None => return Probe::Miss(Miss::Budget), // blown: unfinished scan, no claim
    };
    if m == f32::INFINITY {
        // Every overlapped old cone is fully empty (each a None query
        // composed with its inherited ball claim) — so is our whole tail.
        // The [0, t_start] prefix is the inherited claim's, as always.
        return Probe::Sky;
    }
    let seed = if delta > 0.0 { (m - delta_safe) * (1.0 - 1e-4) } else { m };
    // A COMPLETED finite min predicts the bound query is pinned by real
    // geometry at ~m and cannot meaningfully advance — skip it (Seed::Skip:
    // the tile refines its own inherited cut at max(seed, t_start); nothing
    // old is consumed, so the skip needs no further condition). The age
    // ledger is keyed by the tile's OWN (depth, path) against the previous
    // frame's store — both frames share the same rect layout, so the key
    // tracks this screen region's requery cadence; it governs efficiency
    // decay only, never correctness. Age-capped (or ledger-less) tiles fall
    // back to the plain seed / not-useful outcome and run the real query.
    if let Some(cs) = cuts {
        match cs.load(depth, path) {
            Some((_, _, age)) if age >= MAX_ADOPT_AGE => ls.temporal_adopt_requery += 1,
            found => {
                let age = found.map_or(0, |(_, _, a)| a);
                return Probe::Skip { t: seed.max(t_start), age };
            }
        }
    }
    if seed > t_start { Probe::T(seed) } else { Probe::Miss(Miss::NotUseful) }
}

/// Min claim over the old cells intersecting query box `q`, descending from
/// the cell (depth, path) with (non-NaN) entry `t` and rect `rect`.
/// Returns None only when the visit budget is exhausted — an unfinished
/// scan's running min must never be used (it could surface as false Sky).
/// An early return at `m <= bail` is fine: the true min is even smaller and
/// both map to "no useful seed".
///
/// The child rect arithmetic replays `trace_tile`'s integer midpoint splits
/// exactly (`xm = x0 + w/2`, quadrants TL=0 TR=1 BL=2 BR=3) and MUST stay in
/// lockstep with them: only then does a cached path map back to the identical
/// rect — and, with the cached basis, the bit-identical frustum — it was
/// traced with.
#[allow(clippy::too_many_arguments)]
fn region_min(
    prev: &TemporalCache,
    t: f32,
    depth: u32,
    path: u32,
    rect: (usize, usize, usize, usize),
    q: (f32, f32, f32, f32),
    cap: u32,
    bail: f32,
    budget: &mut u32,
    ls: &mut LocalStats,
) -> Option<f32> {
    if t == f32::INFINITY || depth >= cap {
        // Sky frontier (subtree provably all-NaN) or the descent cap: this
        // cell's own claim stands for its whole region.
        return Some(t);
    }
    let (x0, y0, x1, y1) = rect;
    let xm = x0 + (x1 - x0) / 2;
    let ym = y0 + (y1 - y0) / 2;
    let children = [
        (0u32, (x0, y0, xm, ym)),
        (1u32, (xm, y0, x1, ym)),
        (2u32, (x0, ym, xm, y1)),
        (3u32, (xm, ym, x1, y1)),
    ];
    let mut m = f32::INFINITY;
    let mut contributed = false;
    for (quad, cr) in children {
        if cr.0 >= cr.2 || cr.1 >= cr.3 {
            continue; // empty rect (degenerate 1-px split) covers nothing
        }
        // Closed overlap test: a query point on a shared edge belongs to both
        // closed cell cones, and both must contribute.
        if (cr.0 as f32) > q.2 || (cr.2 as f32) < q.0 || (cr.1 as f32) > q.3 || (cr.3 as f32) < q.1 {
            continue;
        }
        if *budget == 0 {
            return None;
        }
        *budget -= 1;
        ls.temporal_tests += 1;
        let ct = prev.load(depth + 1, (path << 2) | quad);
        // NaN child: never evaluated, so THIS cell is the frontier there and
        // its claim covers the child's region (store-before-recurse — see
        // TemporalCache doc).
        let cm = if ct.is_nan() {
            t
        } else {
            region_min(prev, ct, depth + 1, (path << 2) | quad, cr, q, cap, bail, budget, ls)?
        };
        contributed = true;
        m = m.min(cm);
        if m <= bail {
            return Some(m); // true min only smaller; still "no useful seed"
        }
    }
    // Children exactly partition rect, so a nonempty rect ∩ q always has an
    // overlapping child — an uncontributed min would be a false +INF.
    debug_assert!(contributed, "region_min: no child overlapped a nonempty query");
    if contributed { Some(m) } else { None }
}
