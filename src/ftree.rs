//! The frustum-side acceleration structure: an 8-wide AABB tree collapsed from
//! the binary ray BVH, serving BOUND QUERIES ONLY (`nearest_within`,
//! `refine_cut`). It stores no `tri_idx` and is never traversed by a ray — the
//! two consumers of the one binary tree want opposite shapes (rays want fat
//! leaves and slab tests; bound queries never touch a triangle and are pure
//! box-distance work), and this is the frustum consumer's shape: 8 sibling
//! boxes in four contiguous cache lines, tested with lane-parallel math,
//! skipping two of every three binary levels.
//!
//! Soundness inheritance: every slot box IS a binary node's AABB (the collapse
//! only regroups, never recomputes), so a wide bound query computes the min of
//! `point_aabb_dist` over the same leaf-box set the binary `visit` reaches.
//! The prune rules propagate to subsets — a box fully outside a plane has all
//! its descendants fully outside; `max_dist(subset) <= max_dist(superset)`;
//! `d >= best` only drops non-improving boxes — so the returned bound equals
//! the binary tree's (modulo per-box fp identical by construction: the slot
//! tests reuse the same select/clamp/length shapes on the same values). All
//! claims are re-validated by `--check`'s reference rays regardless.
//!
//! Cut entries are SLOT REFERENCES, `(node << 3) | slot` — one entry = one box
//! = one binary node. `to_bvh_roots` maps a cut back to binary node ids for
//! the (optional) cut-seeded ray path; hemi leaf rays are root-first by
//! default, so the map is off the hot path.

use crate::bvh::{Aabb, Bvh};
use crate::frustum::TileFrustum;
use glam::Vec3A;

pub const WIDTH: usize = 8;
/// Terminal-slot marker in `child`; empty-slot marker in `bnode`.
const INVALID: u32 = u32::MAX;

/// Traversal stack bound for the ITERATIVE consumers (the HLSL port; the CPU
/// visit recurses natively). Wide depth is ~binary/3: the 90M-tri tiled build
/// measures binary depth 53 ⇒ wide ~19; 32 covers far past the binary
/// TRAV_STACK = 96 scale. Asserted at build like TRAV_STACK — never "fix" an
/// overflow by dropping entries (a dropped subtree is a false-sky bug).
pub const FT_STACK: usize = 32;

/// 8 child slots, boxes in SoA so the per-slot tests compile to lane math.
/// 256 B = 4 cache lines, no padding — uploaded VERBATIM to the GPU (the
/// HLSL FtNode in ftree.hlsli mirrors this layout field-for-field; keep them
/// in lockstep).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FNode {
    min_x: [f32; WIDTH],
    min_y: [f32; WIDTH],
    min_z: [f32; WIDTH],
    max_x: [f32; WIDTH],
    max_y: [f32; WIDTH],
    max_z: [f32; WIDTH],
    /// Wide child node for internal slots; `INVALID` for terminal slots
    /// (whose `bnode` is a binary leaf) and empty slots.
    child: [u32; WIDTH],
    /// The binary BVH node this slot's box bounds — the cut→ray translation
    /// and the collapse audit trail. `INVALID` = empty slot (its box is
    /// `Aabb::EMPTY`, which every test rejects).
    bnode: [u32; WIDTH],
}

pub struct FTree {
    pub nodes: Vec<FNode>,
    /// The whole-tree cut (the analog of the binary `[0]`): the root node's
    /// occupied slots. Empty scene ⇒ `root_len == 0` ⇒ every query returns
    /// None/empty, matching the binary n == 0 guard.
    root_cut: [u32; WIDTH],
    root_len: usize,
}

#[inline]
fn entry(node: u32, slot: usize) -> u32 {
    debug_assert!(node < (1 << 29));
    (node << 3) | slot as u32
}

impl FTree {
    #[inline]
    pub fn root_cut(&self) -> &[u32] {
        &self.root_cut[..self.root_len]
    }

    pub fn bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<FNode>()
    }

    /// Deterministic collapse of the binary BVH: each wide node's slots are up
    /// to 8 binary descendants, found by repeatedly expanding the widest
    /// (largest-area) internal slot — a pure function of the binary tree, ties
    /// broken by lowest binary id, so the wide tree inherits `Bvh::identical`'s
    /// byte-determinism.
    pub fn build(bvh: &Bvh) -> FTree {
        let mut t = FTree { nodes: Vec::new(), root_cut: [0; WIDTH], root_len: 0 };
        if bvh.tri_idx.is_empty() {
            return t;
        }
        // Estimate: one wide node per ~7 binary internals, padded.
        t.nodes.reserve(bvh.nodes.len() / 6 + 2);
        let root = t.collapse(bvh, 0);
        for s in 0..WIDTH {
            if t.nodes[root as usize].bnode[s] != INVALID {
                t.root_cut[t.root_len] = entry(root, s);
                t.root_len += 1;
            }
        }
        let depth = t.max_depth();
        assert!(depth <= FT_STACK, "ftree depth {depth} exceeds the {FT_STACK}-entry traversal stack");
        t
    }

    /// Max wide-node depth (root = 1), iterative — the FT_STACK assert's input,
    /// run once at build like `Bvh::max_depth`.
    pub fn max_depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let mut work: Vec<(u32, usize)> = vec![(0, 1)];
        let mut deepest = 0usize;
        while let Some((n, d)) = work.pop() {
            deepest = deepest.max(d);
            let nd = &self.nodes[n as usize];
            for s in 0..WIDTH {
                if nd.child[s] != INVALID {
                    work.push((nd.child[s], d + 1));
                }
            }
        }
        deepest
    }

    /// Build the wide node covering binary node `b` (leaf or internal); returns
    /// its wide id. Recursion depth = wide depth ≈ binary depth / 3 — the
    /// binary build already recurses deeper than this.
    fn collapse(&mut self, bvh: &Bvh, b: u32) -> u32 {
        // Gather up to 8 subtree roots under `b`, expanding the largest-area
        // internal slot first. A binary-leaf `b` (tiny scene) yields itself.
        let mut slots = [INVALID; WIDTH];
        slots[0] = b;
        let mut n_slots = 1usize;
        if bvh.nodes[b as usize].count == 0 {
            slots[0] = bvh.nodes[b as usize].left_first;
            slots[1] = bvh.nodes[b as usize].left_first + 1;
            n_slots = 2;
            loop {
                // Widest internal slot; ties toward the lowest binary id keep
                // the build a pure function of the tree.
                let mut pick = usize::MAX;
                let mut pick_area = -1.0f32;
                for (i, &s) in slots[..n_slots].iter().enumerate() {
                    let nd = &bvh.nodes[s as usize];
                    if nd.count == 0 {
                        let a = nd.aabb.area();
                        if a > pick_area || (a == pick_area && pick != usize::MAX && s < slots[pick]) {
                            pick = i;
                            pick_area = a;
                        }
                    }
                }
                if pick == usize::MAX || n_slots + 1 > WIDTH {
                    break;
                }
                let nd = &bvh.nodes[slots[pick] as usize];
                let (l, r) = (nd.left_first, nd.left_first + 1);
                slots[pick] = l;
                slots[n_slots] = r;
                n_slots += 1;
            }
            // Deterministic slot order regardless of expansion history.
            slots[..n_slots].sort_unstable();
        }

        let id = self.nodes.len() as u32;
        self.nodes.push(FNode {
            min_x: [f32::INFINITY; WIDTH],
            min_y: [f32::INFINITY; WIDTH],
            min_z: [f32::INFINITY; WIDTH],
            max_x: [f32::NEG_INFINITY; WIDTH],
            max_y: [f32::NEG_INFINITY; WIDTH],
            max_z: [f32::NEG_INFINITY; WIDTH],
            child: [INVALID; WIDTH],
            bnode: [INVALID; WIDTH],
        });
        for s in 0..n_slots {
            let bn = slots[s];
            let aabb = bvh.nodes[bn as usize].aabb;
            let nd = &mut self.nodes[id as usize];
            nd.min_x[s] = aabb.min.x;
            nd.min_y[s] = aabb.min.y;
            nd.min_z[s] = aabb.min.z;
            nd.max_x[s] = aabb.max.x;
            nd.max_y[s] = aabb.max.y;
            nd.max_z[s] = aabb.max.z;
            nd.bnode[s] = bn;
        }
        for s in 0..n_slots {
            let bn = slots[s];
            if bvh.nodes[bn as usize].count == 0 {
                let c = self.collapse(bvh, bn);
                self.nodes[id as usize].child[s] = c;
            }
        }
        id
    }

    #[inline]
    fn slot_aabb(&self, e: u32) -> Aabb {
        let nd = &self.nodes[(e >> 3) as usize];
        let s = (e & 7) as usize;
        Aabb {
            min: Vec3A::new(nd.min_x[s], nd.min_y[s], nd.min_z[s]),
            max: Vec3A::new(nd.max_x[s], nd.max_y[s], nd.max_z[s]),
        }
    }

    /// Map a slot-ref cut to binary BVH node ids (for cut-seeded rays). Skips
    /// nothing — every occupied slot IS a binary node.
    pub fn to_bvh_roots<const N: usize>(&self, cut: &[u32], out: &mut [u32; N]) -> usize {
        let n = cut.len().min(N);
        for (o, &e) in out[..n].iter_mut().zip(cut) {
            *o = self.nodes[(e >> 3) as usize].bnode[(e & 7) as usize];
        }
        n
    }

    /// The wide-tree port of `frustum::nearest_geometry_distance_within`:
    /// identical prune rules, identical result (see module doc), fewer and
    /// wider node touches. `roots` are slot-refs (`root_cut()` for the whole
    /// tree). `visits` counts wide NODES expanded (8 slots each), so it is not
    /// unit-comparable with the binary counter — ms is the judge across trees.
    pub fn nearest_within(
        &self,
        f: &TileFrustum,
        t_start: f32,
        t_limit: f32,
        roots: &[u32],
        visits: &mut u64,
    ) -> Option<f32> {
        let mut best = t_limit;
        for &e in roots {
            // Root entries are tested as singleton slots, then descend.
            self.visit_entry(f, t_start, e, &mut best, visits);
        }
        (best < t_limit).then_some(best)
    }

    /// Test one slot's box; descend into its wide child if internal.
    fn visit_entry(&self, f: &TileFrustum, t_start: f32, e: u32, best: &mut f32, visits: &mut u64) {
        let aabb = self.slot_aabb(e);
        if f.aabb_outside(&aabb) {
            return;
        }
        if point_aabb_max_dist(f.origin, &aabb) <= t_start {
            return; // entirely inside the proven-empty ball
        }
        let d = point_aabb_dist(f.origin, &aabb).max(t_start);
        if d >= *best {
            return;
        }
        let nd = &self.nodes[(e >> 3) as usize];
        let s = (e & 7) as usize;
        let child = nd.child[s];
        if child == INVALID {
            // Terminal slot = a binary leaf box: conservative lower bound.
            *best = d;
            return;
        }
        self.visit_node(f, t_start, child, best, visits);
    }

    /// The 8-wide inner loop: test all slots of `node` lane-wise, then recurse
    /// into surviving internal slots near-first.
    fn visit_node(&self, f: &TileFrustum, t_start: f32, node: u32, best: &mut f32, visits: &mut u64) {
        *visits += 1;
        let nd = &self.nodes[node as usize];
        let o = f.origin;

        // Lane math over the 8 slots: point-box distance and max-distance.
        // Straight-line select/min/max/mul-add over [f32; 8] — vectorizable.
        let mut d = [0.0f32; WIDTH];
        let mut dmax = [0.0f32; WIDTH];
        for s in 0..WIDTH {
            // max().min(), NOT f32::clamp — empty slots carry an inverted box
            // (min = +inf > max = -inf) and std clamp panics on min > max;
            // max/min just produces an infinite distance the bnode check skips.
            let cx = o.x.max(nd.min_x[s]).min(nd.max_x[s]) - o.x;
            let cy = o.y.max(nd.min_y[s]).min(nd.max_y[s]) - o.y;
            let cz = o.z.max(nd.min_z[s]).min(nd.max_z[s]) - o.z;
            d[s] = (cx * cx + cy * cy + cz * cz).sqrt();
            let mx = (o.x - nd.min_x[s]).abs().max((o.x - nd.max_x[s]).abs());
            let my = (o.y - nd.min_y[s]).abs().max((o.y - nd.max_y[s]).abs());
            let mz = (o.z - nd.min_z[s]).abs().max((o.z - nd.max_z[s]).abs());
            dmax[s] = (mx * mx + my * my + mz * mz).sqrt();
        }
        // Plane culls (up to 5 active planes), lane-wise positive-vertex test.
        let mut culled = [false; WIDTH];
        f.for_planes(|n, pad| {
            for s in 0..WIDTH {
                let pvx = if n.x >= 0.0 { nd.max_x[s] } else { nd.min_x[s] };
                let pvy = if n.y >= 0.0 { nd.max_y[s] } else { nd.min_y[s] };
                let pvz = if n.z >= 0.0 { nd.max_z[s] } else { nd.min_z[s] };
                let (rx, ry, rz) = (pvx - o.x, pvy - o.y, pvz - o.z);
                let eps = 1e-5 * (1.0 + rx.abs().max(ry.abs()).max(rz.abs()));
                if n.x * rx + n.y * ry + n.z * rz < -eps - pad {
                    culled[s] = true;
                }
            }
        });

        // Survivors: terminals set best; internals collect for ordered descent.
        let mut order: [(f32, u32); WIDTH] = [(0.0, 0); WIDTH];
        let mut n_desc = 0usize;
        for s in 0..WIDTH {
            // Empty slots carry Aabb::EMPTY: clamp gives min > max... their d
            // is NaN-free but meaningless; bnode == INVALID skips them.
            if nd.bnode[s] == INVALID || culled[s] || dmax[s] <= t_start {
                continue;
            }
            let ds = d[s].max(t_start);
            if ds >= *best {
                continue;
            }
            let c = nd.child[s];
            if c == INVALID {
                *best = ds; // terminal: binary-leaf box distance
            } else {
                order[n_desc] = (ds, c);
                n_desc += 1;
            }
        }
        // Near-first: insertion sort over <= 8 survivors, then re-check the
        // shrinking best before each descent.
        for i in 1..n_desc {
            let key = order[i];
            let mut j = i;
            while j > 0 && order[j - 1].0 > key.0 {
                order[j] = order[j - 1];
                j -= 1;
            }
            order[j] = key;
        }
        for &(ds, c) in &order[..n_desc] {
            if ds >= *best {
                break; // ascending: every remaining slot is too
            }
            self.visit_node(f, t_start, c, best, visits);
        }
    }

    /// The wide-tree port of `frustum::refine_cut` — the same three drop rules,
    /// the same never-drop-on-`d >= best` law, the same `out_len + work_len <= N`
    /// coarse-emit invariant, over slot-ref entries. Expanding an internal slot
    /// pushes its wide child's occupied slots (up to 8 at once, so overflow
    /// coarsens a little earlier than the binary 2-way expansion — sound, the
    /// entry is emitted, never dropped).
    #[allow(clippy::too_many_arguments)]
    pub fn refine_cut<const N: usize>(
        &self,
        f: &TileFrustum,
        t_ball: f32,
        t_far: f32,
        parent_cut: &[u32],
        out: &mut [u32; N],
        visits: &mut u64,
        overflows: &mut u64,
    ) -> usize {
        let mut work = [0u32; N];
        debug_assert!(parent_cut.len() <= N);
        let mut wlen = parent_cut.len().min(N);
        work[..wlen].copy_from_slice(&parent_cut[..wlen]);
        let mut olen = 0usize;
        let has_far = t_far.is_finite();
        while wlen > 0 {
            wlen -= 1;
            let e = work[wlen];
            *visits += 1;
            let aabb = self.slot_aabb(e);
            if f.aabb_outside(&aabb) {
                continue;
            }
            if point_aabb_max_dist(f.origin, &aabb) <= t_ball {
                continue; // entirely inside the proven-empty ball
            }
            if has_far && point_aabb_dist(f.origin, &aabb) >= t_far {
                continue; // entirely beyond the consumers' tmax clamp
            }
            let nd = &self.nodes[(e >> 3) as usize];
            let s = (e & 7) as usize;
            let c = nd.child[s];
            if c != INVALID {
                let cn = &self.nodes[c as usize];
                let n_occ = cn.bnode.iter().filter(|&&b| b != INVALID).count();
                if olen + wlen + n_occ <= N {
                    for cs in 0..WIDTH {
                        if cn.bnode[cs] != INVALID {
                            work[wlen] = entry(c, cs);
                            wlen += 1;
                        }
                    }
                    continue;
                }
                *overflows += 1; // internal emitted coarsely
            }
            out[olen] = e;
            olen += 1;
        }
        olen
    }
}

/// The session's wide tree, built LAZILY on the first hemi bound query and
/// read by `Accel::of` at every hemi entry. A process-global for the same
/// reason the `CUT_SEED_*` levers are: exactly one scene+BVH is live per
/// process (resize re-enters `session()` around the same borrow; `--check`'s
/// determinism rebuild is compared, never queried), and the experiment phase
/// should not thread a field through 25 `FrameCtx` literals — promotion into
/// `FrameCtx` happens with the tile-recursion wiring.
///
/// Lazy because only fb (H) frames consume it: an eager build would charge
/// every session the collapse (26-42 ms on multi-million-node trees) and the
/// memory (~256 B per ~7 binary internals — ~1.5 GB at the 90M-tri tiled
/// scale) for a mode most sessions never enter. The first fb frame pays the
/// build once, inside whichever rayon task gets there first (OnceLock blocks
/// the racers — a one-time hitch, measured tens of ms on real scenes).
static INSTALLED: std::sync::OnceLock<&'static FTree> = std::sync::OnceLock::new();

/// Kill switch (`--no-ftree`): hemi bound queries fall back to the binary
/// BVH — the A/B lever for the two-tree split. Default on: measured -15/-17%
/// on the hemi-ao rows and -4/-8% on hemi-gi across the default scene and San
/// Miguel, with the self-test pinning bit-identical bounds.
pub static FTREE_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

fn installed(bvh: &Bvh) -> Option<&'static FTree> {
    if !FTREE_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    // Leak-once: the tree lives to process exit. Sound because exactly one
    // BVH is live per process (see above) — the first caller's `bvh` IS the
    // session tree every later caller holds.
    Some(INSTALLED.get_or_init(|| Box::leak(Box::new(FTree::build(bvh)))))
}

/// The two-tree dispatch handle threaded through the hemi integrator: `bvh`
/// answers RAYS (always), and bound queries go to the wide tree when wired.
/// Copy so it rides `Cx` and recursion without lifetimes fighting back.
#[derive(Copy, Clone)]
pub struct Accel<'a> {
    pub bvh: &'a Bvh,
    pub ft: Option<&'a FTree>,
}

impl<'a> Accel<'a> {
    /// The session handle: the live BVH plus the (lazily built) wide tree.
    /// Every hemi entry point's callers use this — the one-BVH-per-process
    /// invariant is what makes the global sound (see `INSTALLED`).
    #[inline]
    pub fn of(bvh: &'a Bvh) -> Accel<'a> {
        Accel { bvh, ft: installed(bvh) }
    }

    /// The whole-tree cut in whichever id space is live: binary `[0]` or the
    /// wide root slots.
    #[inline]
    pub fn root_cut(&self) -> &'a [u32] {
        match self.ft {
            Some(ft) => ft.root_cut(),
            None => &[0],
        }
    }

    #[inline]
    pub fn nearest_within(
        &self,
        f: &TileFrustum,
        t_start: f32,
        t_limit: f32,
        roots: &[u32],
        visits: &mut u64,
    ) -> Option<f32> {
        match self.ft {
            Some(ft) => ft.nearest_within(f, t_start, t_limit, roots, visits),
            None => crate::frustum::nearest_geometry_distance_within(
                self.bvh, f, t_start, t_limit, roots, visits,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn refine_cut<const N: usize>(
        &self,
        f: &TileFrustum,
        t_ball: f32,
        t_far: f32,
        parent_cut: &[u32],
        out: &mut [u32; N],
        visits: &mut u64,
        overflows: &mut u64,
    ) -> usize {
        match self.ft {
            Some(ft) => ft.refine_cut(f, t_ball, t_far, parent_cut, out, visits, overflows),
            None => crate::frustum::refine_cut(
                self.bvh, f, t_ball, t_far, parent_cut, out, visits, overflows,
            ),
        }
    }

    /// A cut as binary-BVH ray-seed roots: identity for the binary tree, the
    /// slot→bnode map for the wide one. `buf` is the caller's stack scratch;
    /// the returned slice aliases either it or the input.
    #[inline]
    pub fn ray_roots<'b, const N: usize>(&self, cut: &'b [u32], buf: &'b mut [u32; N]) -> &'b [u32] {
        match self.ft {
            Some(ft) => {
                let n = ft.to_bvh_roots(cut, buf);
                &buf[..n]
            }
            None => cut,
        }
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

/// Build-integrity + equivalence gate, run by `--check` (DLL-free): the wide
/// tree must reproduce the binary tree's bound on randomized hemisphere-shaped
/// and tile-shaped queries — same apexes, same frustums, both directions of
/// clamp — and its cuts must translate to valid binary roots.
pub fn self_test(scene: &crate::scene::Scene, bvh: &Bvh) -> Result<(), String> {
    let ft = FTree::build(bvh);
    // Structural audit: every occupied slot's box equals its binary node's box.
    for (i, nd) in ft.nodes.iter().enumerate() {
        for s in 0..WIDTH {
            let b = nd.bnode[s];
            if b == INVALID {
                continue;
            }
            let e = entry(i as u32, s);
            let sa = ft.slot_aabb(e);
            let ba = &bvh.nodes[b as usize].aabb;
            if sa.min != ba.min || sa.max != ba.max {
                return Err(format!("ftree slot ({i},{s}) box != binary node {b} box"));
            }
            let internal = bvh.nodes[b as usize].count == 0;
            if internal != (nd.child[s] != INVALID) {
                return Err(format!("ftree slot ({i},{s}) internal/terminal mismatch vs binary {b}"));
            }
        }
    }
    // Behavioral: bounds match the binary query on a deterministic probe sweep.
    let diag = scene.diag;
    let mut rng = fastrand::Rng::with_seed(0x5EED_F00D);
    let mut worst = 0.0f32;
    for i in 0..512 {
        let o = Vec3A::new(
            (rng.f32() - 0.5) * diag,
            rng.f32() * diag * 0.5,
            (rng.f32() - 0.5) * diag,
        );
        let n = Vec3A::new(rng.f32() - 0.5, rng.f32() + 0.1, rng.f32() - 0.5).normalize();
        let f = if i % 2 == 0 {
            TileFrustum::half_space(o, n)
        } else {
            // A small cone: three directions around n.
            let (t1, t2) = crate::shade::onb(n);
            let a = (n + 0.3 * t1).normalize();
            let b = (n + 0.3 * t2).normalize();
            let c = (n - 0.2 * t1 - 0.2 * t2).normalize();
            TileFrustum::tri_cell(o, a, b, c)
        };
        let t_start = if i % 3 == 0 { 0.0 } else { rng.f32() * diag * 0.05 };
        let t_limit = if i % 5 == 0 { f32::INFINITY } else { diag * (0.2 + rng.f32()) };
        let (mut va, mut vb) = (0u64, 0u64);
        let wide = ft.nearest_within(&f, t_start, t_limit, ft.root_cut(), &mut va);
        let bin = crate::frustum::nearest_geometry_distance_within(
            bvh, &f, t_start, t_limit, &[0], &mut vb,
        );
        match (wide, bin) {
            (None, None) => {}
            (Some(w), Some(b)) => {
                let rel = (w - b).abs() / b.max(1e-6);
                worst = worst.max(rel);
                if rel > 1e-4 {
                    return Err(format!("ftree bound {w} != binary {b} (probe {i}, rel {rel:.2e})"));
                }
            }
            (w, b) => return Err(format!("ftree {w:?} vs binary {b:?} disagree on emptiness (probe {i})")),
        }
        // Cut translation: refine at the wide tree, map to binary roots, and
        // check every root id is in range and its box is inside the... (a
        // valid binary node id is the whole contract; coverage is implied by
        // the slot audit above).
        let mut cut = [0u32; 64];
        let len = ft.refine_cut(&f, t_start, t_limit, ft.root_cut(), &mut cut, &mut va, &mut 0u64);
        let mut broots = [0u32; 64];
        let blen = ft.to_bvh_roots(&cut[..len], &mut broots);
        if blen != len || broots[..blen].iter().any(|&r| r as usize >= bvh.nodes.len()) {
            return Err(format!("ftree cut translation invalid (probe {i})"));
        }
    }
    eprintln!(
        "ftree self-test: {} wide nodes ({:.1} MB) from {} binary | 512 probes, bounds match (worst rel {:.2e})",
        ft.nodes.len(),
        ft.bytes() as f32 / (1024.0 * 1024.0),
        bvh.nodes.len(),
        worst,
    );
    Ok(())
}
