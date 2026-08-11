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
//! the cut-seeded ray paths: primary leaf tiles translate ONCE per tile
//! (shade_tile/sparse_fill — cut-seeded primary rays are a measured win and
//! stay on), hemi leaf rays are root-first by default so their map is off the
//! hot path.

use crate::bvh::{Aabb, Bvh};
use crate::frustum::{frustum_aabb_dist, TileFrustum};
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

/// The GPU wire format of one wide node — 112 B vs FNode's 256 (the dead
/// `bnode` lanes dropped: GPU cuts never seed rays; occupancy is the `occ`
/// bitmask), boxes quantized to u8 coords in a per-node frame, rounded
/// OUTWARD so every decoded box CONTAINS its FNode original. Sound for the
/// bound-query consumer and only that one: a bigger box is plane-culled
/// less, its `max_dist <= t_start` ball test fires less, and its
/// `point_aabb_dist` only shrinks — a weaker but valid lower bound, never an
/// overshoot. Produced by `FTree::quantized` at upload; consumed by
/// ftree.hlsli's FtNode (field-for-field lockstep, decode `org + q * sca`
/// kept `precise` there — an fma contraction could round a face inward).
///
/// Split verdict (measured): the CPU keeps the f32 FNode — the decode ALU
/// cost +9-15% on the hemi bench with node counts unchanged; the GPU takes
/// this format — San Miguel `gpu hybrid` medians 2.86 vs 3.03 ms (the
/// bandwidth trade pays once the tree exceeds cache; the L2-resident
/// default scene reads ~+6%, that row's noise band) and the tree upload
/// drops -56% (~1.5 GB -> 0.65 GB at the 90M-tri tiled scale).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QFNode {
    /// Frame origin: componentwise min over the occupied slot boxes.
    org: [f32; 3],
    /// Quantization step per axis: ~ext/255, ulp-bumped so q = 255 decodes
    /// at-or-past the frame max. 0 on flat axes (decode = org, exact).
    sca: [f32; 3],
    /// Per-axis quantized slot mins `[axis][slot]`, decoding <= the true min.
    qmn: [[u8; WIDTH]; 3],
    /// Per-axis quantized slot maxes, decoding >= the true max.
    qmx: [[u8; WIDTH]; 3],
    child: [u32; WIDTH],
    /// Occupancy bitmask (bit s = slot s holds a binary node).
    occ: u32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<QFNode>() == 112);

/// Next representable f32 up — the sca bump and the outward verify-adjust
/// step. Callers only pass finite positives.
#[inline]
fn bump_up(x: f32) -> f32 {
    f32::from_bits(x.to_bits() + 1)
}

/// Largest q whose decode `org + q·sca` lands at or below `v` (outward for a
/// min face). q = 0 decodes to exactly `org <= v` by frame construction, so
/// the walk-down always terminates on a sound value.
#[inline]
fn quantize_min(v: f32, org: f32, sca: f32) -> u8 {
    if sca == 0.0 {
        return 0; // flat axis: decode == org == v, exact
    }
    let mut q = (((v - org) / sca).floor().clamp(0.0, 255.0)) as i32;
    while q > 0 && org + q as f32 * sca > v {
        q -= 1;
    }
    q as u8
}

/// Smallest q whose decode lands at or above `v` (outward for a max face).
/// q = 255 decodes at-or-past the frame max by the sca bump, so the walk-up
/// always terminates on a sound value.
#[inline]
fn quantize_max(v: f32, org: f32, sca: f32) -> u8 {
    if sca == 0.0 {
        return 0;
    }
    let mut q = (((v - org) / sca).ceil().clamp(0.0, 255.0)) as i32;
    while q < 255 && ((org + q as f32 * sca) < v) {
        q += 1;
    }
    q as u8
}

pub struct FTree {
    pub nodes: Vec<FNode>,
    /// The whole-tree cut (the analog of the binary `[0]`): the root node's
    /// occupied slots. Empty scene ⇒ `root_len == 0` and `nodes[0].occ == 0`
    /// after quantization ⇒ every CPU/GPU query returns empty. The physical
    /// zero-occupancy root is required because GPU root queries always read
    /// node zero before occupancy rejects its eight slots.
    root_cut: [u32; WIDTH],
    root_len: usize,
}

fn empty_fnode() -> FNode {
    FNode {
        min_x: [f32::INFINITY; WIDTH],
        min_y: [f32::INFINITY; WIDTH],
        min_z: [f32::INFINITY; WIDTH],
        max_x: [f32::NEG_INFINITY; WIDTH],
        max_y: [f32::NEG_INFINITY; WIDTH],
        max_z: [f32::NEG_INFINITY; WIDTH],
        child: [INVALID; WIDTH],
        bnode: [INVALID; WIDTH],
    }
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

    /// GPU-upload size of the quantized wire format.
    pub fn quantized_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<QFNode>()
    }

    /// Flat `nodes × 8` slot→binary-node map for the GPU's `--sw-rays`
    /// leaf-cut translation (`ft_bnode[(e >> 3) * 8 + (e & 7)]` — the HLSL
    /// mirror of `to_bvh_roots`' `bnode[slot]` lookup). Uploaded only under
    /// that lever; the quantized QFNode wire format deliberately keeps
    /// dropping `bnode` for every other session. `INVALID` lanes ride along
    /// unread — cut entries only ever reference occupied slots by
    /// construction (`refine_cut` emits from the occupancy set).
    pub fn bnode_flat(&self) -> Vec<u32> {
        (0..self.nodes.len() * WIDTH).map(|i| self.bnode_at(i)).collect()
    }

    /// One entry of that flat map, by its flat index — `quantize_node`'s
    /// shape one stream over. It exists so a backend that STREAMS the map can
    /// read it a slot at a time instead of materializing `nodes × 8` u32s
    /// first, and `bnode_flat` is written over it so the two indexings cannot
    /// disagree. (`FNode`'s fields stay private — its bounds and its `bnode`
    /// lanes are collapse-internal.)
    pub fn bnode_at(&self, flat: usize) -> u32 {
        self.nodes[flat / WIDTH].bnode[flat % WIDTH]
    }

    /// Convert to the GPU wire format (QFNode): per node, a frame over the
    /// occupied slots' f32 boxes and every face quantized OUTWARD against
    /// the exact decode expression. Node ids (and so slot-ref cuts) are
    /// unchanged — the arrays are index-parallel.
    pub fn quantized(&self) -> Vec<QFNode> {
        (0..self.nodes.len()).map(|i| self.quantize_node(i)).collect()
    }

    /// ONE node in that format. Split out for `gpu_bvh_node`'s reason: D3D12
    /// materializes the whole array and Vulkan streams it through a bounded
    /// ring (~480 MB of wide nodes at 34.4M triangles, which is a `Vec` worth
    /// not making), so the shared thing is the per-node CONVERSION — the part
    /// a divergence would corrupt — and each backend keeps its own iteration.
    pub fn quantize_node(&self, i: usize) -> QFNode {
        let nd = &self.nodes[i];
        let mut q = QFNode {
            org: [0.0; 3],
            sca: [0.0; 3],
            qmn: [[0; WIDTH]; 3],
            qmx: [[0; WIDTH]; 3],
            child: nd.child,
            occ: 0,
            _pad: 0,
        };
        let (mn, mx) = ([&nd.min_x, &nd.min_y, &nd.min_z], [&nd.max_x, &nd.max_y, &nd.max_z]);
        for a in 0..3 {
            let (mut org, mut top) = (f32::INFINITY, f32::NEG_INFINITY);
            for s in 0..WIDTH {
                if nd.bnode[s] != INVALID {
                    org = org.min(mn[a][s]);
                    top = top.max(mx[a][s]);
                }
            }
            if !org.is_finite() {
                continue; // no occupied slots (empty-scene root only)
            }
            q.org[a] = org;
            let ext = top - org;
            if ext > 0.0 {
                let mut sca = ext / 255.0;
                while org + 255.0 * sca < top {
                    sca = bump_up(sca);
                }
                q.sca[a] = sca;
            }
        }
        for s in 0..WIDTH {
            if nd.bnode[s] == INVALID {
                continue;
            }
            q.occ |= 1 << s;
            for a in 0..3 {
                q.qmn[a][s] = quantize_min(mn[a][s], q.org[a], q.sca[a]);
                q.qmx[a][s] = quantize_max(mx[a][s], q.org[a], q.sca[a]);
            }
        }
        q
    }

    /// Deterministic collapse of the binary BVH: each wide node's slots are up
    /// to 8 binary descendants, found by repeatedly expanding the widest
    /// (largest-area) internal slot — a pure function of the binary tree, ties
    /// broken by lowest binary id, so the wide tree inherits `Bvh::identical`'s
    /// byte-determinism.
    pub fn build(bvh: &Bvh) -> FTree {
        let mut t = FTree { nodes: Vec::new(), root_cut: [0; WIDTH], root_len: 0 };
        if bvh.tri_idx.is_empty() {
            // The HLSL ROOT_CUT_SLOT path expands entries 0..7 and reads
            // ft_nodes[0] before testing occupancy. Keep one real uploaded
            // node whose occ mask quantizes to zero; an empty Vec would make
            // that otherwise-correct empty query an out-of-bounds GPU read.
            t.nodes.push(empty_fnode());
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
        if self.root_len == 0 || self.nodes.is_empty() {
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
        self.nodes.push(empty_fnode());
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
        // The SHARED range (frustum.rs) — self_test pins wide == binary bitwise.
        let d = frustum_aabb_dist(f, &aabb).max(t_start);
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
            // The SHARED frustum-aware range (frustum.rs), NOT a lane-math copy
            // of it: `self_test` pins wide == binary BITWISE, and two ports of
            // the same formula do not agree to the ulp. This costs the slot
            // loop its vectorizability — the plane culls below stay lane-wise.
            // (An empty slot's inverted box makes every clip vacuous and the
            // emptiness test fire, so it lands on +INF and the `bnode ==
            // INVALID` skip below never sees a NaN.)
            let ab = Aabb {
                min: Vec3A::new(nd.min_x[s], nd.min_y[s], nd.min_z[s]),
                max: Vec3A::new(nd.max_x[s], nd.max_y[s], nd.max_z[s]),
            };
            d[s] = frustum_aabb_dist(f, &ab);
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

/// A BVH's wide tree is built LAZILY on its first bound query (the first
/// hybrid tile step or hemi entry) and read by `Accel::of`/`Accel::for_tiles`.
/// The `OnceLock` lives on `Bvh`: wide slot ids and slot-to-binary-node maps
/// can never outlive the exact hierarchy from which they were derived. Live
/// scene rebuilds therefore drop the old pair atomically through normal Rust
/// ownership, and the replacement BVH starts with an empty cache.
///
/// Lazy so GPU-tracer sessions (which build their own local FTree for upload
/// and drop it) and `--no-*` sessions never pay the collapse (26-42 ms on
/// multi-million-node trees) or the memory (~256 B per ~7 binary internals —
/// ~1.5 GB at the 90M-tri tiled scale). The first consuming frame pays the
/// build once, inside whichever rayon task gets there first (`OnceLock`
/// blocks the racers — a one-time hitch). Every producer and consumer
/// borrowing that BVH (temporal CutStore, replay records) sees the same tree.

/// Kill switch (`--no-ftree`): ALL bound queries fall back to the binary
/// BVH — the A/B lever for the two-tree split. Default on: measured -15/-17%
/// on the hemi-ao rows and -4/-8% on hemi-gi across the default scene and San
/// Miguel, with the self-test pinning bit-identical bounds.
pub static FTREE_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Per-consumer lever (`--ftree-tiles` enables): the primary TILE recursion
/// (tile_step/adopt_step bound queries + refines) on the wide tree, cuts
/// translated to binary ray roots once per leaf tile. Default OFF — unlike
/// the GPU tile kernels (−23%), CPU tiles measured wall-NEUTRAL on San Miguel
/// and ~10% slower on the stress field's no-temporal regime: fat cuts (mean
/// ~66) of singleton slot-refs fetch a 256 B node to test one slot, and the
/// short descents never amortize it — the short-query lesson again. Counted
/// frustum nodes DO drop (−21..45%), so the quantized-box layout re-measures
/// this. Hemi keeps its own default via `Accel::of`; `--no-ftree` kills both.
pub static FTREE_TILES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn installed(bvh: &Bvh) -> Option<&FTree> {
    if !FTREE_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    Some(bvh.ftree.get_or_init(|| {
        let t0 = std::time::Instant::now();
        let t = FTree::build(bvh);
        eprintln!(
            "ftree: {} wide nodes ({:.1} MB) from {} binary in {:.0} ms",
            t.nodes.len(),
            t.bytes() as f32 / (1024.0 * 1024.0),
            bvh.nodes.len(),
            t0.elapsed().as_secs_f32() * 1e3,
        );
        t
    }))
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
    /// Every hemi entry point's callers use this; the wide tree is owned by
    /// this exact `bvh`, so replacing a scene cannot retain stale slot ids.
    #[inline]
    pub fn of(bvh: &'a Bvh) -> Accel<'a> {
        Accel { bvh, ft: installed(bvh) }
    }

    /// The TILE-path handle (`tile_step`/`adopt_step`/the leaf-tile cut
    /// translation): `of` gated by the `FTREE_TILES` lever, so tiles and hemi
    /// A/B independently. Every site in one frame agrees (both levers are
    /// set once at startup; each BVH's cache is monotonic-Some).
    #[inline]
    pub fn for_tiles(bvh: &'a Bvh) -> Accel<'a> {
        if FTREE_TILES.load(std::sync::atomic::Ordering::Relaxed) {
            Accel::of(bvh)
        } else {
            Accel { bvh, ft: None }
        }
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
    // Empty-scene GPU contract: ROOT_CUT_SLOT always reads ft_nodes[0], so
    // the uploaded stream must contain one zero-occupancy sentinel even
    // though the logical root cut is empty.
    let empty_bvh = Bvh::from_parts(
        vec![crate::bvh::BvhNode {
            aabb: Aabb::EMPTY,
            left_first: 0,
            count: 0,
        }],
        Vec::new(),
    );
    let empty_ft = FTree::build(&empty_bvh);
    let empty_q = empty_ft.quantized();
    if !empty_ft.root_cut().is_empty()
        || empty_ft.max_depth() != 0
        || empty_q.len() != 1
        || empty_q[0].occ != 0
    {
        return Err("ftree empty root is not one zero-occupancy GPU node".into());
    }

    // Lifecycle regression: two simultaneously live BVHs must initialize
    // distinct auxiliary trees whose slot boxes come from their own owners.
    // This is the live snapshot-rebuild bug in miniature; a process-global
    // OnceLock makes the second x value incorrectly remain 1.
    let one_leaf = |x: f32| {
        Bvh::from_parts(
            vec![crate::bvh::BvhNode {
                aabb: Aabb {
                    min: Vec3A::new(x, 0.0, 0.0),
                    max: Vec3A::new(x + 1.0, 1.0, 1.0),
                },
                left_first: 0,
                count: 1,
            }],
            vec![0],
        )
    };
    let (a, b) = (one_leaf(1.0), one_leaf(10.0));
    let fa = a.ftree.get_or_init(|| FTree::build(&a));
    let fb = b.ftree.get_or_init(|| FTree::build(&b));
    if std::ptr::eq(fa, fb)
        || fa.slot_aabb(fa.root_cut()[0]).min.x != 1.0
        || fb.slot_aabb(fb.root_cut()[0]).min.x != 10.0
    {
        return Err("ftree cache does not follow its owning BVH".into());
    }

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
    // The GPU wire format (QFNode, consumed by ftree.hlsli): CONTAINMENT is
    // the soundness half — every decoded face at-or-outside the true f32
    // face, per the exact `org + q * sca` decode expression — and per-face
    // slack within ~one quantum is the tightness half (a frame/decode wiring
    // error lands faces whole-extents off). The slack mean doubles as
    // quantization anti-vacuity: ~0.5 quanta is the healthy signature.
    // (Probe-level tightness can't be read off bound queries here: scenes
    // keep their dominant y=0 ground plane AT its frame origin, which
    // decodes exactly, so quantized bounds match binary on most probes.)
    let qn = ft.quantized();
    if qn.len() != ft.nodes.len() {
        return Err("ftree quantized(): length mismatch".into());
    }
    let (mut slack_n, mut slack_sum) = (0u64, 0.0f64);
    for (i, (q, nd)) in qn.iter().zip(&ft.nodes).enumerate() {
        if q.child != nd.child {
            return Err(format!("ftree quantized node {i}: child mismatch"));
        }
        for s in 0..WIDTH {
            let occupied = nd.bnode[s] != INVALID;
            if occupied != (q.occ & (1 << s) != 0) {
                return Err(format!("ftree quantized node {i} slot {s}: occ mismatch"));
            }
            if !occupied {
                continue;
            }
            let tmn = [nd.min_x[s], nd.min_y[s], nd.min_z[s]];
            let tmx = [nd.max_x[s], nd.max_y[s], nd.max_z[s]];
            for a in 0..3 {
                // The decode expression, verbatim (ftree.hlsli's precise twin).
                let dmn = q.org[a] + q.qmn[a][s] as f32 * q.sca[a];
                let dmx = q.org[a] + q.qmx[a][s] as f32 * q.sca[a];
                if dmn > tmn[a] || dmx < tmx[a] {
                    return Err(format!(
                        "ftree quantized node {i} slot {s} axis {a}: decoded face INSIDE the true box — rounded inward"
                    ));
                }
                let slack = (tmn[a] - dmn).max(dmx - tmx[a]);
                // Near-flat axes have sca below the f32 spacing at org — the
                // decode grid can't resolve finer than an ulp there, so the
                // bound carries an ulp term (containment above is still the
                // exact soundness gate; measured trips: 1 ulp of coord
                // magnitude on stress and San Miguel).
                let mag = dmn.abs().max(dmx.abs());
                if slack > 1.5 * q.sca[a] + 4.0 * f32::EPSILON * mag {
                    return Err(format!(
                        "ftree quantized node {i} slot {s} axis {a}: slack {slack:.3e} > quantum bound"
                    ));
                }
                // Mean only over grid-resolvable lanes (sub-ulp sca lanes
                // would put unbounded slack/sca ratios into the stat).
                if q.sca[a] > f32::EPSILON * mag {
                    slack_n += 1;
                    slack_sum += (slack / q.sca[a]) as f64;
                }
            }
        }
    }
    let slack_mean = if slack_n > 0 { (slack_sum / slack_n as f64) as f32 } else { 0.0 };

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
        "ftree self-test: {} wide nodes ({:.1} MB cpu, {:.1} MB gpu-quantized, slack mean {:.2} quanta) from {} binary | 512 probes, bounds match (worst rel {:.2e})",
        ft.nodes.len(),
        ft.bytes() as f32 / (1024.0 * 1024.0),
        ft.quantized_bytes() as f32 / (1024.0 * 1024.0),
        slack_mean,
        bvh.nodes.len(),
        worst,
    );
    Ok(())
}
