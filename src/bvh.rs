use crate::scene::Scene;
use glam::Vec3A;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct Aabb {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: Vec3A::INFINITY,
        max: Vec3A::NEG_INFINITY,
    };

    fn grow(&mut self, p: Vec3A) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    fn grow_aabb(&mut self, b: &Aabb) {
        self.min = self.min.min(b.min);
        self.max = self.max.max(b.max);
    }

    fn area(&self) -> f32 {
        let e = (self.max - self.min).max(Vec3A::ZERO);
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }
}

/// count > 0: leaf over tri_idx[left_first .. left_first+count].
/// count == 0: internal, children at left_first and left_first + 1.
#[derive(Clone, Copy)]
pub struct BvhNode {
    pub aabb: Aabb,
    pub left_first: u32,
    pub count: u32,
}

pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    pub tri_idx: Vec<u32>,
}

pub struct Ray {
    pub o: Vec3A,
    pub d: Vec3A,
    pub inv_d: Vec3A,
}

impl Ray {
    pub fn new(o: Vec3A, d: Vec3A) -> Self {
        Ray { o, d, inv_d: d.recip() }
    }
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub t: f32,
    pub tri: u32,
    pub u: f32,
    pub v: f32,
}

const BINS: usize = 12;
const MAX_LEAF: usize = 8;
/// Traversal stack capacity. Both traversal loops push at most one entry per
/// level, so the required stack is exactly the tree depth; `Bvh::build`
/// hard-asserts `max_depth() <= TRAV_STACK` once, which keeps the hot loops
/// branch-free (a silent overflow in release would be an out-of-bounds write,
/// and dropping the far child instead would be UNSOUND — a missed subtree
/// breaks the frustum/temporal lower-bound contracts). SAH depth is ~45-50 at
/// 10M tris and grows ~2 levels per 4x tris; 96 covers 100M+ with margin.
const TRAV_STACK: usize = 96;

/// Subtree handed to phase 2 of the build: `nodes[node_i]` still holds its
/// (first, count) leaf-form range over `tri_idx` until the stitch replaces it.
struct Pending {
    node_i: u32,
    first: u32,
    count: u32,
}

/// Phase-1 stop size — a pure function of n ONLY (never the thread count):
/// the node order must be byte-identical across machines. ~256 subtrees keeps
/// every core busy without ballooning the sequential top phase.
fn par_threshold(n: usize) -> usize {
    (n / 256).max(2048)
}

impl Bvh {
    /// Deterministic two-phase parallel build. Phase 1 splits sequentially
    /// until ranges fall under `par_threshold` (node allocation order is
    /// sequential ⇒ the top of the tree is pinned); phase 2 builds each
    /// pending range — disjoint, ascending slices of `tri_idx` after the
    /// in-place partitions — into a local arena in parallel (collect
    /// preserves pending order and per-subtree fp ops run sequentially ⇒ the
    /// output is byte-identical across runs AND thread counts, which `--check`
    /// gates on loaded scenes); a cheap sequential stitch splices the arenas
    /// in. Phase 1 never creates real leaves (threshold > MAX_LEAF and a
    /// too-big node always splits, via SAH or the median fallback), so the
    /// pending ranges exactly partition 0..n.
    pub fn build(scene: &Scene) -> Bvh {
        let n = scene.indices.len();
        let (tri_aabb, centroids): (Vec<Aabb>, Vec<Vec3A>) = scene
            .indices
            .par_iter()
            .map(|tri| {
                let (a, b, c) = (
                    scene.positions[tri[0] as usize],
                    scene.positions[tri[1] as usize],
                    scene.positions[tri[2] as usize],
                );
                let mut bb = Aabb::EMPTY;
                bb.grow(a);
                bb.grow(b);
                bb.grow(c);
                (bb, (a + b + c) / 3.0)
            })
            .unzip();

        // No 2n up-front node reserve (8 GB at 100M tris): phase 1 grows
        // organically (small) and the stitch reserves the exact total.
        let mut nodes = Vec::new();
        let mut tri_idx: Vec<u32> = (0..n as u32).collect();
        nodes.push(BvhNode {
            aabb: Aabb::EMPTY,
            left_first: 0,
            count: n as u32,
        });

        if n > 0 {
            let threshold = par_threshold(n);
            let mut pending = Vec::new();
            subdivide_range(
                &mut nodes,
                0,
                &mut tri_idx,
                &tri_aabb,
                &centroids,
                &mut Some((&mut pending, threshold)),
            );

            let mut slices: Vec<(&mut [u32], u32)> = Vec::with_capacity(pending.len());
            let mut rest: &mut [u32] = &mut tri_idx;
            let mut consumed = 0u32;
            for p in &pending {
                let (_, r) = rest.split_at_mut((p.first - consumed) as usize);
                let (s, r2) = r.split_at_mut(p.count as usize);
                slices.push((s, p.first));
                rest = r2;
                consumed = p.first + p.count;
            }
            let arenas: Vec<Vec<BvhNode>> = slices
                .into_par_iter()
                .map(|(slice, base)| build_subtree(slice, base, &tri_aabb, &centroids))
                .collect();

            // Stitch: arena local 0 lands in the pending node's slot and
            // local i>0 at off+i, so every internal link rebases by one
            // uniform +off — child pairs stay adjacent (count==0 ⇒ children
            // at left_first, left_first+1). Leaf ranges were lifted to global
            // tri_idx positions inside build_subtree.
            let extra: usize = arenas.iter().map(|a| a.len() - 1).sum();
            nodes.reserve_exact(extra);
            for (p, arena) in pending.iter().zip(&arenas) {
                let off = (nodes.len() - 1) as u32;
                let rebase = |nd: &BvhNode| BvhNode {
                    left_first: if nd.count == 0 { nd.left_first + off } else { nd.left_first },
                    ..*nd
                };
                nodes[p.node_i as usize] = rebase(&arena[0]);
                nodes.extend(arena[1..].iter().map(rebase));
            }
        }

        let bvh = Bvh { nodes, tri_idx };
        let depth = bvh.max_depth();
        assert!(
            depth <= TRAV_STACK,
            "BVH depth {depth} exceeds the {TRAV_STACK}-entry traversal stack"
        );
        bvh
    }

    /// Byte-equality of two builds — the determinism gate: `--check` rebuilds
    /// once and asserts this, pinning the two-phase build's "identical across
    /// runs and thread counts" contract. Bit compare (f32::to_bits), so a
    /// -0.0/NaN drift can't hide.
    pub fn identical(&self, other: &Bvh) -> bool {
        self.tri_idx == other.tri_idx
            && self.nodes.len() == other.nodes.len()
            && self.nodes.iter().zip(&other.nodes).all(|(a, b)| {
                a.left_first == b.left_first
                    && a.count == b.count
                    && a.aabb.min.to_array().map(f32::to_bits)
                        == b.aabb.min.to_array().map(f32::to_bits)
                    && a.aabb.max.to_array().map(f32::to_bits)
                        == b.aabb.max.to_array().map(f32::to_bits)
            })
    }

    /// Max node depth (root = 1). Iterative — O(nodes), run once at build so
    /// the traversal loops can stay branch-free (see TRAV_STACK).
    pub fn max_depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let mut max = 0usize;
        let mut stack = vec![(0u32, 1usize)];
        while let Some((i, d)) = stack.pop() {
            max = max.max(d);
            let node = &self.nodes[i as usize];
            if node.count == 0 {
                stack.push((node.left_first, d + 1));
                stack.push((node.left_first + 1, d + 1));
            }
        }
        max
    }

    /// Closest hit in (tmin, tmax). `visits` counts BVH node visits.
    pub fn intersect(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        mut tmax: f32,
        visits: &mut u64,
    ) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        self.intersect_from(scene, ray, tmin, &mut tmax, 0, &mut best, visits);
        best
    }

    /// Closest hit in (tmin, tmax) with traversal seeded from a tile's node
    /// cut instead of the root — primary rays inside a quadtree leaf tile skip
    /// the top of the tree the tile's ancestors already culled. Each root is
    /// slab-tested against the shrinking tmax, so an early hit prunes the
    /// remaining roots. Secondary rays and reference rays use `intersect`.
    pub fn intersect_multi(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        mut tmax: f32,
        roots: &[u32],
        visits: &mut u64,
    ) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for &r in roots {
            if slab_t(&self.nodes[r as usize].aabb, ray, tmin, tmax).is_finite() {
                self.intersect_from(scene, ray, tmin, &mut tmax, r, &mut best, visits);
            }
        }
        best
    }

    fn intersect_from(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        tmax: &mut f32,
        start: u32,
        best: &mut Option<Hit>,
        visits: &mut u64,
    ) {
        let mut stack = [0u32; TRAV_STACK];
        let mut sp = 0usize;
        let mut node_idx = start;
        loop {
            let node = &self.nodes[node_idx as usize];
            *visits += 1;
            if node.count > 0 {
                let first = node.left_first as usize;
                for &t in &self.tri_idx[first..first + node.count as usize] {
                    if let Some((tt, u, v)) = moller_trumbore(scene, t, ray) {
                        if tt > tmin && tt < *tmax {
                            *tmax = tt;
                            *best = Some(Hit { t: tt, tri: t, u, v });
                        }
                    }
                }
                if sp == 0 {
                    break;
                }
                sp -= 1;
                node_idx = stack[sp];
            } else {
                let l = node.left_first;
                let dl = slab_t(&self.nodes[l as usize].aabb, ray, tmin, *tmax);
                let dr = slab_t(&self.nodes[l as usize + 1].aabb, ray, tmin, *tmax);
                let (near, far, dnear, dfar) = if dl <= dr {
                    (l, l + 1, dl, dr)
                } else {
                    (l + 1, l, dr, dl)
                };
                if dnear.is_finite() {
                    node_idx = near;
                    if dfar.is_finite() {
                        // Capacity is guaranteed by build's max_depth assert;
                        // the debug_assert stays as belt-and-braces.
                        debug_assert!(sp < stack.len(), "BVH traversal stack overflow");
                        stack[sp] = far;
                        sp += 1;
                    }
                } else if sp > 0 {
                    sp -= 1;
                    node_idx = stack[sp];
                } else {
                    break;
                }
            }
        }
    }

    /// Any hit in (tmin, tmax) — early-out occlusion test for shadow/AO rays.
    pub fn occluded(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        tmax: f32,
        visits: &mut u64,
    ) -> bool {
        if !slab_t(&self.nodes[0].aabb, ray, tmin, tmax).is_finite() {
            return false;
        }
        self.occluded_from(scene, ray, tmin, tmax, 0, visits)
    }

    /// Any hit in (tmin, tmax) with traversal seeded from a node cut — the
    /// occlusion analog of `intersect_multi`, for hemisphere/shaft bounce rays
    /// that own a cut (their OWN apex-relative cut, never a primary tile's).
    pub fn occluded_multi(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        tmax: f32,
        roots: &[u32],
        visits: &mut u64,
    ) -> bool {
        for &r in roots {
            if slab_t(&self.nodes[r as usize].aabb, ray, tmin, tmax).is_finite()
                && self.occluded_from(scene, ray, tmin, tmax, r, visits)
            {
                return true;
            }
        }
        false
    }

    fn occluded_from(
        &self,
        scene: &Scene,
        ray: &Ray,
        tmin: f32,
        tmax: f32,
        start: u32,
        visits: &mut u64,
    ) -> bool {
        let mut stack = [0u32; TRAV_STACK];
        let mut sp = 0usize;
        let mut node_idx = start;
        loop {
            let node = &self.nodes[node_idx as usize];
            *visits += 1;
            if node.count > 0 {
                let first = node.left_first as usize;
                for &t in &self.tri_idx[first..first + node.count as usize] {
                    if let Some((tt, _, _)) = moller_trumbore(scene, t, ray) {
                        if tt > tmin && tt < tmax {
                            return true;
                        }
                    }
                }
            } else {
                let l = node.left_first;
                if slab_t(&self.nodes[l as usize].aabb, ray, tmin, tmax).is_finite() {
                    if slab_t(&self.nodes[l as usize + 1].aabb, ray, tmin, tmax).is_finite() {
                        debug_assert!(sp < stack.len(), "BVH traversal stack overflow");
                        stack[sp] = l + 1;
                        sp += 1;
                    }
                    node_idx = l;
                    continue;
                }
                if slab_t(&self.nodes[l as usize + 1].aabb, ray, tmin, tmax).is_finite() {
                    node_idx = l + 1;
                    continue;
                }
            }
            if sp == 0 {
                return false;
            }
            sp -= 1;
            node_idx = stack[sp];
        }
    }
}

/// Build one pending subtree into a local arena: root at local 0, internal
/// `left_first` = LOCAL node index (the stitch rebases them by one uniform
/// +off), leaf ranges lifted to GLOBAL `tri_idx` positions here so the stitch
/// never touches them. `tri_idx` is this subtree's disjoint slice; `base` is
/// the slice's global start.
fn build_subtree(
    tri_idx: &mut [u32],
    base: u32,
    tri_aabb: &[Aabb],
    centroids: &[Vec3A],
) -> Vec<BvhNode> {
    let mut nodes = Vec::new();
    nodes.push(BvhNode {
        aabb: Aabb::EMPTY,
        left_first: 0,
        count: tri_idx.len() as u32,
    });
    subdivide_range(&mut nodes, 0, tri_idx, tri_aabb, centroids, &mut None);
    for nd in &mut nodes {
        if nd.count > 0 {
            nd.left_first += base;
        }
    }
    nodes
}

/// The recursive SAH subdivide, shared by both build phases. `first` in the
/// node being subdivided is relative to `tri_idx` (phase 1 passes the whole
/// array, so relative == global; phase 2 passes its subtree slice). With
/// `pend` set (phase 1), a range at or under the threshold is recorded and
/// left for phase 2 instead of being split.
fn subdivide_range(
    nodes: &mut Vec<BvhNode>,
    node_i: usize,
    tri_idx: &mut [u32],
    tri_aabb: &[Aabb],
    centroids: &[Vec3A],
    pend: &mut Option<(&mut Vec<Pending>, usize)>,
) {
    let (first, count) = {
        let node = &nodes[node_i];
        (node.left_first as usize, node.count as usize)
    };
    if let Some((pending, threshold)) = pend {
        if count <= *threshold {
            pending.push(Pending {
                node_i: node_i as u32,
                first: first as u32,
                count: count as u32,
            });
            return;
        }
    }

    let mut bounds = Aabb::EMPTY;
    let mut cbounds = Aabb::EMPTY;
    for &t in &tri_idx[first..first + count] {
        bounds.grow_aabb(&tri_aabb[t as usize]);
        cbounds.grow(centroids[t as usize]);
    }
    nodes[node_i].aabb = bounds;

    if count <= 2 {
        return; // leaf
    }

    let ext = cbounds.max - cbounds.min;
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };
    let cmin = cbounds.min[axis];
    let cext = ext[axis];

    let mut split_at = usize::MAX;
    if cext > 1e-8 {
        // Binned SAH.
        let mut bin_bounds = [Aabb::EMPTY; BINS];
        let mut bin_count = [0usize; BINS];
        let k = BINS as f32 * (1.0 - 1e-6) / cext;
        for &t in &tri_idx[first..first + count] {
            let b = ((centroids[t as usize][axis] - cmin) * k) as usize;
            bin_count[b] += 1;
            bin_bounds[b].grow_aabb(&tri_aabb[t as usize]);
        }
        // Sweep: cost of splitting after bin i.
        let mut right_area = [0.0f32; BINS];
        let mut right_count = [0usize; BINS];
        let mut acc = Aabb::EMPTY;
        let mut cnt = 0;
        for i in (1..BINS).rev() {
            acc.grow_aabb(&bin_bounds[i]);
            cnt += bin_count[i];
            right_area[i] = acc.area();
            right_count[i] = cnt;
        }
        let mut best_cost = f32::INFINITY;
        let mut best_bin = 0;
        acc = Aabb::EMPTY;
        cnt = 0;
        for i in 0..BINS - 1 {
            acc.grow_aabb(&bin_bounds[i]);
            cnt += bin_count[i];
            if cnt == 0 || right_count[i + 1] == 0 {
                continue;
            }
            let cost = acc.area() * cnt as f32 + right_area[i + 1] * right_count[i + 1] as f32;
            if cost < best_cost {
                best_cost = cost;
                best_bin = i;
            }
        }
        let leaf_cost = bounds.area() * count as f32;
        if best_cost.is_finite() && (best_cost < leaf_cost || count > MAX_LEAF) {
            // In-place partition by bin threshold.
            let mut i = first;
            let mut j = first + count - 1;
            while i <= j {
                let b = ((centroids[tri_idx[i] as usize][axis] - cmin) * k) as usize;
                if b <= best_bin {
                    i += 1;
                } else {
                    tri_idx.swap(i, j);
                    if j == 0 {
                        break;
                    }
                    j -= 1;
                }
            }
            if i > first && i < first + count {
                split_at = i;
            }
        }
    }

    if split_at == usize::MAX {
        if count <= MAX_LEAF {
            return; // leaf
        }
        // Degenerate SAH (e.g., identical centroids) — median split.
        tri_idx[first..first + count].sort_unstable_by(|&a, &b| {
            centroids[a as usize][axis]
                .partial_cmp(&centroids[b as usize][axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        split_at = first + count / 2;
    }

    let l = nodes.len();
    nodes.push(BvhNode {
        aabb: Aabb::EMPTY,
        left_first: first as u32,
        count: (split_at - first) as u32,
    });
    nodes.push(BvhNode {
        aabb: Aabb::EMPTY,
        left_first: split_at as u32,
        count: (first + count - split_at) as u32,
    });
    nodes[node_i].left_first = l as u32;
    nodes[node_i].count = 0;
    subdivide_range(nodes, l, tri_idx, tri_aabb, centroids, pend);
    subdivide_range(nodes, l + 1, tri_idx, tri_aabb, centroids, pend);
}

/// Slab test: entry t if the ray hits the box within (tmin, tmax), else +INF.
#[inline(always)]
fn slab_t(aabb: &Aabb, ray: &Ray, tmin: f32, tmax: f32) -> f32 {
    let t1 = (aabb.min - ray.o) * ray.inv_d;
    let t2 = (aabb.max - ray.o) * ray.inv_d;
    let t_enter = t1.min(t2).max_element().max(tmin);
    let t_exit = t1.max(t2).min_element().min(tmax);
    if t_exit >= t_enter { t_enter } else { f32::INFINITY }
}

#[inline(always)]
fn moller_trumbore(scene: &Scene, tri: u32, ray: &Ray) -> Option<(f32, f32, f32)> {
    let [i0, i1, i2] = scene.indices[tri as usize];
    let v0 = scene.positions[i0 as usize];
    let e1 = scene.positions[i1 as usize] - v0;
    let e2 = scene.positions[i2 as usize] - v0;
    let p = ray.d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-10 {
        return None;
    }
    let inv = 1.0 / det;
    let s = ray.o - v0;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = ray.d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t <= 0.0 {
        return None;
    }
    // Alpha cutout: a candidate on an alpha-masked textured triangle is
    // REJECTED (not accepted-and-continued) where the mask says transparent —
    // `intersect_from` keeps tmax unshrunk and walks on to the true nearest
    // opaque hit; `occluded_from` keeps searching. Every ray type (hybrid,
    // verify reference, shadow, AO, hemi, shaft) funnels through here, so the
    // exact-zero gates stay like-for-like. The frustum bound queries still
    // treat masked triangles as solid AABBs — sound, because rejection only
    // removes hits: the true nearest hit moves FARTHER, so a conservative
    // lower bound stays a lower bound (inherited tmin never overshoots, and
    // hemi cells become at most less provably-empty, never falsely empty).
    if scene.any_alpha {
        if let crate::scene::MatKind::Textured { tex } =
            scene.materials[scene.tri_mat[tri as usize] as usize].kind
        {
            let tx = &scene.textures[tex as usize];
            if tx.alpha_masked {
                let uv = scene.tri_uv(tri, u, v);
                if tx.alpha_nearest(uv.x, uv.y) < 128 {
                    return None;
                }
            }
        }
    }
    Some((t, u, v))
}
