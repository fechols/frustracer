use crate::scene::Scene;
use glam::Vec3A;

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

impl Bvh {
    pub fn build(scene: &Scene) -> Bvh {
        let n = scene.indices.len();
        let mut tri_aabb = Vec::with_capacity(n);
        let mut centroids = Vec::with_capacity(n);
        for tri in &scene.indices {
            let (a, b, c) = (
                scene.positions[tri[0] as usize],
                scene.positions[tri[1] as usize],
                scene.positions[tri[2] as usize],
            );
            let mut bb = Aabb::EMPTY;
            bb.grow(a);
            bb.grow(b);
            bb.grow(c);
            tri_aabb.push(bb);
            centroids.push((a + b + c) / 3.0);
        }
        let mut bvh = Bvh {
            nodes: Vec::with_capacity(2 * n.max(1)),
            tri_idx: (0..n as u32).collect(),
        };
        bvh.nodes.push(BvhNode {
            aabb: Aabb::EMPTY,
            left_first: 0,
            count: n as u32,
        });
        if n > 0 {
            bvh.subdivide(0, &tri_aabb, &centroids);
        }
        bvh
    }

    fn subdivide(&mut self, node_i: usize, tri_aabb: &[Aabb], centroids: &[Vec3A]) {
        let (first, count) = {
            let node = &self.nodes[node_i];
            (node.left_first as usize, node.count as usize)
        };

        let mut bounds = Aabb::EMPTY;
        let mut cbounds = Aabb::EMPTY;
        for &t in &self.tri_idx[first..first + count] {
            bounds.grow_aabb(&tri_aabb[t as usize]);
            cbounds.grow(centroids[t as usize]);
        }
        self.nodes[node_i].aabb = bounds;

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
            for &t in &self.tri_idx[first..first + count] {
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
                let cost =
                    acc.area() * cnt as f32 + right_area[i + 1] * right_count[i + 1] as f32;
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
                    let b = ((centroids[self.tri_idx[i] as usize][axis] - cmin) * k) as usize;
                    if b <= best_bin {
                        i += 1;
                    } else {
                        self.tri_idx.swap(i, j);
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
            self.tri_idx[first..first + count].sort_unstable_by(|&a, &b| {
                centroids[a as usize][axis]
                    .partial_cmp(&centroids[b as usize][axis])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            split_at = first + count / 2;
        }

        let l = self.nodes.len();
        self.nodes.push(BvhNode {
            aabb: Aabb::EMPTY,
            left_first: first as u32,
            count: (split_at - first) as u32,
        });
        self.nodes.push(BvhNode {
            aabb: Aabb::EMPTY,
            left_first: split_at as u32,
            count: (first + count - split_at) as u32,
        });
        self.nodes[node_i].left_first = l as u32;
        self.nodes[node_i].count = 0;
        self.subdivide(l, tri_aabb, centroids);
        self.subdivide(l + 1, tri_aabb, centroids);
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
        let mut stack = [0u32; 64];
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
                        // SAH depth stays well under 64 even at 10M-tri scenes
                        // (~45-50 expected); assert in debug rather than UB.
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
        let mut stack = [0u32; 64];
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
