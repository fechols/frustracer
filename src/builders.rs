//! M7 builder bake-off: alternative RAY-BVH builders behind `--bvh-builder`
//! (default `sah` = the M2 binned-SAH build in bvh.rs). All three produce the
//! same `Bvh` — every consumer, gate, and the .fcache work unchanged (the
//! builder id rides `bvh::build_key`) — and all are byte-deterministic
//! (sequential or order-independent-parallel; `Bvh::identical` gates them).
//!
//! The experiment design isolates PARTITION ORDER: `lbvh` and `som` run the
//! SAME top-down code-split finisher with the SAME M2 cost-model leaf test —
//! the only difference is the sort key (raw Morton vs the SOM's learned
//! space-filling curve), so their delta is exactly what a converged 3D SOM
//! lattice buys over the raw Z-curve. `ploc` is the strongest known-good
//! agglomerative competitor — the "cluster by proximity" instinct under the
//! CORRECT metric d(A,B) = SA(A ∪ B) instead of the SOM's L2.
//!
//! Score on the measured counters (`bvh quality` line + spin ray_nodes/ms) —
//! NEVER on the SAH readout, which anti-correlates with measured cost here
//! (shared-origin rays, inherited t_start; see CLAUDE.md).
//!
//! Dev caveat: the .fcache keys on build PARAMETERS (`bvh::build_key`), not
//! on this file's code — editing a builder's semantics serves the previous
//! run's stale tree until the scene's sidecar is deleted (or CACHE_VERSION
//! bumped, once the change ships).
//!
//! Verdict (2026-07-13, spin path 250, measured ray nodes): sah best-or-
//! close everywhere — it stays the default. ploc −34% vs sah on San Miguel
//! (real dense-scene clustering merit) but +121% on --stress 5000; lbvh
//! 2.7-4.4× worse (the control); som WORSE than raw Morton on both scenes
//! (+3% / +41%) — the learned curve's BMU cell boundaries tear the
//! bit-prefix locality the code splitter exploits.

use crate::bvh::{self, Aabb, Builder, Bvh, BvhNode};
use crate::scene::Scene;
use glam::Vec3A;
use rayon::prelude::*;

/// Depth at which the code splitter abandons code bits for median splits:
/// 64 code levels + log2 of any remaining run stays under TRAV_STACK = 96.
const BALANCE_DEPTH: u32 = 64;

pub fn build_alt(scene: &Scene, which: Builder) -> Bvh {
    // Foliage sway on a bake-off builder keeps the v0.2 per-tri-sweep +
    // per-test-shift path (`grow_sway_sweep` below + `moller_trumbore`'s
    // head, both gated on `!gateway_mode()`): gateway subtrees are SAH-only,
    // and this arm doubles as the built-in A/B against them. Loud, since the
    // per-tri sweep is the measured ~80%-of-the-bill regime.
    if scene.sway.is_some() {
        eprintln!(
            "foliage sway: per-tri sweep path ({which:?} bake-off builder) — \
             gateway subtrees are SAH-only"
        );
    }
    let n = scene.indices.len();
    let (tri_aabb, centroids): (Vec<Aabb>, Vec<Vec3A>) = scene
        .indices
        .par_iter()
        .enumerate()
        .map(|(i, tri)| {
            let (a, b, c) = (
                scene.positions[tri[0] as usize],
                scene.positions[tri[1] as usize],
                scene.positions[tri[2] as usize],
            );
            let mut bb = Aabb::EMPTY;
            bb.grow(a);
            bb.grow(b);
            bb.grow(c);
            crate::bvh::grow_height_sweep(scene, i as u32, a, b, c, &mut bb);
            crate::bvh::grow_sway_sweep(scene, i as u32, &mut bb);
            (bb, (a + b + c) / 3.0)
        })
        .unzip();
    if n == 0 {
        // Mirror the SAH build's empty-scene shape: one empty leaf root.
        return Bvh::from_parts(
            vec![BvhNode { aabb: Aabb::EMPTY, left_first: 0, count: 0 }],
            Vec::new(),
        );
    }

    match which {
        Builder::Lbvh => code_split_build(&tri_aabb, &morton_codes(&centroids)),
        Builder::Som => code_split_build(&tri_aabb, &som_codes(&centroids)),
        Builder::Ploc => ploc_build(&tri_aabb, &centroids),
        Builder::Sah => unreachable!("sah dispatches in bvh.rs"),
    }
}

// ---------------------------------------------------------------- morton --

/// Spread the low 21 bits of v to every third bit of a u64.
#[inline]
fn spread3(v: u64) -> u64 {
    let mut x = v & 0x1f_ffff; // 21 bits
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8)) & 0x100f00f00f00f00f;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3;
    x = (x | (x << 2)) & 0x1249249249249249;
    x
}

#[inline]
fn morton3(x: u64, y: u64, z: u64) -> u64 {
    (spread3(x) << 2) | (spread3(y) << 1) | spread3(z)
}

/// Centroid bounds of the point set (sequential — deterministic fp).
fn bounds(points: &[Vec3A]) -> Aabb {
    let mut bb = Aabb::EMPTY;
    for p in points {
        bb.grow(*p);
    }
    bb
}

/// 63-bit Morton code per centroid over the centroid bounds, 21 bits/axis.
fn morton_codes(centroids: &[Vec3A]) -> Vec<u64> {
    let bb = bounds(centroids);
    let ext = (bb.max - bb.min).max(Vec3A::splat(1e-30));
    centroids
        .par_iter()
        .map(|c| {
            let q = ((*c - bb.min) / ext * ((1 << 21) as f32 - 1.0))
                .clamp(Vec3A::ZERO, Vec3A::splat((1 << 21) as f32 - 1.0));
            morton3(q.x as u64, q.y as u64, q.z as u64)
        })
        .collect()
}

// ------------------------------------------------- code-split finisher --

/// Top-down builder over any per-tri sort key: sort (code, tri), split each
/// run at the highest differing code bit (median once codes are equal or the
/// depth budget wants balance), M2 cost-model leaf test (leaf `A·N` vs split
/// `C_trav·A + A_L·N_L + A_R·N_R`) with the `max_leaf` hard cap. Sequential
/// and allocation-ordered — byte-deterministic by construction.
fn code_split_build(tri_aabb: &[Aabb], codes_in: &[u64]) -> Bvh {
    let n = codes_in.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&i| (codes_in[i as usize], i));
    let codes: Vec<u64> = order.iter().map(|&i| codes_in[i as usize]).collect();

    let mut nodes: Vec<BvhNode> = Vec::with_capacity(n / 2);
    nodes.push(BvhNode { aabb: Aabb::EMPTY, left_first: 0, count: 0 });
    let c_trav = bvh::c_trav();
    let max_leaf = bvh::max_leaf();

    // Explicit work stack (node_i, lo, hi, depth) — no recursion limits.
    let mut work: Vec<(u32, u32, u32, u32)> = vec![(0, 0, n as u32, 0)];
    while let Some((node_i, lo, hi, depth)) = work.pop() {
        let (lo_u, hi_u) = (lo as usize, hi as usize);
        let count = hi - lo;
        let mut bb = Aabb::EMPTY;
        for &t in &order[lo_u..hi_u] {
            bb.grow_aabb(&tri_aabb[t as usize]);
        }
        if count == 1 {
            nodes[node_i as usize] = BvhNode { aabb: bb, left_first: lo, count };
            continue;
        }
        // Split position: highest differing code bit, else median. Past the
        // balance budget always median (bounds depth under TRAV_STACK).
        let mid = if depth >= BALANCE_DEPTH || codes[lo_u] == codes[hi_u - 1] {
            lo + count / 2
        } else {
            find_split(&codes, lo_u, hi_u) as u32
        };
        if count as usize <= max_leaf {
            // M2 leaf test on this partition's two children.
            let (mut bl, mut br) = (Aabb::EMPTY, Aabb::EMPTY);
            for &t in &order[lo_u..mid as usize] {
                bl.grow_aabb(&tri_aabb[t as usize]);
            }
            for &t in &order[mid as usize..hi_u] {
                br.grow_aabb(&tri_aabb[t as usize]);
            }
            let split_cost = c_trav * bb.area()
                + bl.area() * (mid - lo) as f32
                + br.area() * (hi - mid) as f32;
            if bb.area() * count as f32 <= split_cost {
                nodes[node_i as usize] = BvhNode { aabb: bb, left_first: lo, count };
                continue;
            }
        }
        let l = nodes.len() as u32;
        nodes.push(BvhNode { aabb: Aabb::EMPTY, left_first: 0, count: 0 });
        nodes.push(BvhNode { aabb: Aabb::EMPTY, left_first: 0, count: 0 });
        nodes[node_i as usize] = BvhNode { aabb: bb, left_first: l, count: 0 };
        // Push right first so the LEFT child pops next: children then land in
        // DFS order like the SAH build's allocation (locality, determinism).
        work.push((l + 1, mid, hi, depth + 1));
        work.push((l, lo, mid, depth + 1));
    }
    Bvh::from_parts(nodes, order)
}

/// First index in (lo, hi) where the highest differing bit of
/// codes[lo]..codes[hi-1] flips — the standard LBVH split search.
fn find_split(codes: &[u64], lo: usize, hi: usize) -> usize {
    let first = codes[lo];
    let last = codes[hi - 1];
    let common = (first ^ last).leading_zeros();
    // Binary search: largest index whose code shares > common leading bits
    // with `first`.
    let (mut a, mut b) = (lo, hi - 1);
    while a + 1 < b {
        let m = (a + b) / 2;
        if (first ^ codes[m]).leading_zeros() > common {
            a = m;
        } else {
            b = m;
        }
    }
    b
}

// ------------------------------------------------------------------ som --

/// Batch SOM on a 3D lattice, fixed seed-free init (unit = its grid cell
/// center) and fixed epochs — deterministic: assignments are pure parallel
/// maps, the update accumulates SEQUENTIALLY in point order. The converged
/// lattice warps toward triangle density; sorting by BMU lattice Morton
/// (high bits) is the LEARNED space-filling curve, with raw Morton in the
/// low bits keeping intra-cell order coherent.
fn som_codes(centroids: &[Vec3A]) -> Vec<u64> {
    let n = centroids.len();
    let bb = bounds(centroids);
    let ext = (bb.max - bb.min).max(Vec3A::splat(1e-30));
    // Lattice side: ~4·max_leaf tris per unit, clamped to the 6-bit Morton
    // budget (64³ = 262k units).
    let k = ((n as f64 / (bvh::max_leaf() as f64 * 4.0)).cbrt().round() as usize).clamp(4, 64);
    let kf = k as f32;
    let uidx = |x: usize, y: usize, z: usize| (z * k + y) * k + x;

    // Init: unit = its cell center in space.
    let mut units: Vec<Vec3A> = Vec::with_capacity(k * k * k);
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let c = (Vec3A::new(x as f32, y as f32, z as f32) + 0.5) / kf;
                units.push(bb.min + c * ext);
            }
        }
    }

    // The BMU search assumes units stay near their lattice cells (they are
    // anchored by the neighborhood term), so it checks the 3³ cells around
    // the point's grid cell — deterministic ties by lowest unit index.
    let cell_of = |p: Vec3A| -> (usize, usize, usize) {
        let g = ((p - bb.min) / ext * kf).clamp(Vec3A::ZERO, Vec3A::splat(kf - 1.0));
        (g.x as usize, g.y as usize, g.z as usize)
    };
    let bmu = |units: &[Vec3A], p: Vec3A| -> (usize, usize, usize) {
        let (cx, cy, cz) = cell_of(p);
        let mut best = f32::INFINITY;
        let mut pick = (cx, cy, cz);
        for dz in -1i64..=1 {
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (x, y, z) = (cx as i64 + dx, cy as i64 + dy, cz as i64 + dz);
                    if x < 0 || y < 0 || z < 0 || x >= k as i64 || y >= k as i64 || z >= k as i64 {
                        continue;
                    }
                    let (x, y, z) = (x as usize, y as usize, z as usize);
                    let d = (units[uidx(x, y, z)] - p).length_squared();
                    if d < best {
                        best = d;
                        pick = (x, y, z);
                    }
                }
            }
        }
        pick
    };

    const EPOCHS: usize = 6;
    for e in 0..EPOCHS {
        // sigma decays 1.5 -> 0.5 lattice units; radius-1 neighborhood.
        let sigma = 1.5 * (0.5f32 / 1.5).powf(e as f32 / (EPOCHS - 1) as f32);
        let inv2s2 = 1.0 / (2.0 * sigma * sigma);
        let assign: Vec<(u16, u16, u16)> = centroids
            .par_iter()
            .map(|&p| {
                let (x, y, z) = bmu(&units, p);
                (x as u16, y as u16, z as u16)
            })
            .collect();
        // Sequential neighborhood-weighted accumulation (determinism).
        let mut acc: Vec<Vec3A> = vec![Vec3A::ZERO; units.len()];
        let mut wsum: Vec<f32> = vec![0.0; units.len()];
        for (p, &(bx, by, bz)) in centroids.iter().zip(&assign) {
            for dz in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (x, y, z) = (bx as i64 + dx, by as i64 + dy, bz as i64 + dz);
                        if x < 0 || y < 0 || z < 0 || x >= k as i64 || y >= k as i64 || z >= k as i64
                        {
                            continue;
                        }
                        let w = (-((dx * dx + dy * dy + dz * dz) as f32) * inv2s2).exp();
                        let u = uidx(x as usize, y as usize, z as usize);
                        acc[u] += *p * w;
                        wsum[u] += w;
                    }
                }
            }
        }
        for (u, unit) in units.iter_mut().enumerate() {
            if wsum[u] > 0.0 {
                *unit = acc[u] / wsum[u];
            } // empty neighborhoods keep their position (stay anchored)
        }
    }

    // Codes: BMU lattice Morton (18 bits) high, raw 15-bit/axis Morton low.
    centroids
        .par_iter()
        .map(|&p| {
            let (x, y, z) = bmu(&units, p);
            let lattice = morton3(x as u64, y as u64, z as u64);
            let q = ((p - bb.min) / ext * ((1 << 15) as f32 - 1.0))
                .clamp(Vec3A::ZERO, Vec3A::splat((1 << 15) as f32 - 1.0));
            let fine = morton3(q.x as u64, q.y as u64, q.z as u64);
            (lattice << 45) | fine
        })
        .collect()
}

// ----------------------------------------------------------------- ploc --

/// One node of the bottom-up merge tree (u32::MAX children = leaf slot
/// holding `tri`).
struct PNode {
    aabb: Aabb,
    left: u32,
    right: u32,
    tri: u32,
    /// Subtree triangle count (collapse decision input).
    count: u32,
}

/// PLOC search radius (standard 16): each cluster considers its R sorted
/// neighbors each direction; mutual nearest neighbors merge.
const PLOC_R: usize = 16;

fn ploc_build(tri_aabb: &[Aabb], centroids: &[Vec3A]) -> Bvh {
    let n = tri_aabb.len();
    // Initial clusters: single tris in Morton order (ties by index).
    let codes = morton_codes(centroids);
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&i| (codes[i as usize], i));

    let mut arena: Vec<PNode> = order
        .iter()
        .map(|&t| PNode { aabb: tri_aabb[t as usize], left: u32::MAX, right: u32::MAX, tri: t, count: 1 })
        .collect();
    // Live cluster list: (arena node, aabb copy for locality).
    let mut cl: Vec<u32> = (0..n as u32).collect();
    let mut boxes: Vec<Aabb> = arena.iter().map(|p| p.aabb).collect();

    while cl.len() > 1 {
        let m = cl.len();
        // Nearest neighbor by union area within ±PLOC_R (parallel, pure;
        // ties toward the lower index — deterministic).
        let nn: Vec<u32> = (0..m)
            .into_par_iter()
            .map(|i| {
                let lo = i.saturating_sub(PLOC_R);
                let hi = (i + PLOC_R + 1).min(m);
                let mut best = f32::INFINITY;
                let mut pick = u32::MAX;
                for j in lo..hi {
                    if j == i {
                        continue;
                    }
                    let mut u = boxes[i];
                    u.grow_aabb(&boxes[j]);
                    let a = u.area();
                    if a < best {
                        best = a;
                        pick = j as u32;
                    }
                }
                pick
            })
            .collect();
        // Merge mutual pairs (i < j), compact in order.
        let mut out_cl: Vec<u32> = Vec::with_capacity(m);
        let mut out_boxes: Vec<Aabb> = Vec::with_capacity(m);
        let mut merged = vec![false; m];
        for i in 0..m {
            if merged[i] {
                continue;
            }
            let j = nn[i] as usize;
            if j != usize::MAX as usize && j < m && nn[j] as usize == i && i < j && !merged[j] {
                let mut u = boxes[i];
                u.grow_aabb(&boxes[j]);
                let id = arena.len() as u32;
                let count = arena[cl[i] as usize].count + arena[cl[j] as usize].count;
                arena.push(PNode { aabb: u, left: cl[i], right: cl[j], tri: u32::MAX, count });
                merged[i] = true;
                merged[j] = true;
                out_cl.push(id);
                out_boxes.push(u);
            } else if nn[i] == u32::MAX {
                // No neighbor in radius (m == 1 handled by the loop cond).
                out_cl.push(cl[i]);
                out_boxes.push(boxes[i]);
                merged[i] = true;
            } else if !(nn[j] as usize == i && j < i) {
                // Not merging this pass (its partner merged elsewhere or the
                // pairing wasn't mutual); carried forward as-is. The case
                // j < i with mutual nn was handled when j was visited.
                out_cl.push(cl[i]);
                out_boxes.push(boxes[i]);
                merged[i] = true;
            }
        }
        // Progress guard: mutual-NN always yields >= 1 merge among live
        // clusters, but guard against a pathological stall anyway.
        assert!(out_cl.len() < m, "PLOC made no progress at {m} clusters");
        cl = out_cl;
        boxes = out_boxes;
    }

    // Convert the merge tree to the Bvh layout with the M2 collapse rule:
    // a subtree becomes a leaf when count <= max_leaf and the leaf cost wins
    // against its (recursive, memoized) split cost.
    let root = cl[0];
    let c_trav = bvh::c_trav();
    let max_leaf = bvh::max_leaf() as u32;

    // Bottom-up subtree costs (arena order IS bottom-up: children precede
    // parents by construction).
    let mut cost: Vec<f32> = vec![0.0; arena.len()];
    let mut as_leaf: Vec<bool> = vec![false; arena.len()];
    for i in 0..arena.len() {
        let nd = &arena[i];
        if nd.tri != u32::MAX {
            cost[i] = nd.aabb.area();
            as_leaf[i] = true;
            continue;
        }
        let split = c_trav * nd.aabb.area() + cost[nd.left as usize] + cost[nd.right as usize];
        let leaf = nd.aabb.area() * nd.count as f32;
        if nd.count <= max_leaf && leaf <= split {
            cost[i] = leaf;
            as_leaf[i] = true;
        } else {
            cost[i] = split;
        }
    }

    // Mutual-NN merging can chain (measured: depth 176 on San Miguel — thin
    // sorted geometry merges linearly), so the conversion carries a depth
    // budget: a subtree that cannot fit the remaining TRAV_STACK budget is
    // re-emitted as a balanced median-split treelet over its triangles (its
    // DFS order — locality kept, depth forced to log2). Everything within
    // budget keeps PLOC's own topology.
    let mut sub_depth: Vec<u32> = vec![1; arena.len()];
    for i in 0..arena.len() {
        let nd = &arena[i];
        if nd.tri == u32::MAX {
            sub_depth[i] = 1 + sub_depth[nd.left as usize].max(sub_depth[nd.right as usize]);
        }
    }
    let budget = crate::bvh::TRAV_STACK as u32 - 4;

    // DFS emit (explicit stack): nodes in allocation order, tri_idx in leaf
    // order — the exact layout every consumer expects.
    let mut nodes: Vec<BvhNode> = Vec::with_capacity(arena.len() / 2 + 1);
    let mut tri_idx: Vec<u32> = Vec::with_capacity(n);
    nodes.push(BvhNode { aabb: Aabb::EMPTY, left_first: 0, count: 0 });
    let mut work: Vec<(u32, u32, u32)> = vec![(root, 0, 0)]; // (arena id, node slot, depth)
    let mut rebalanced = 0u64;
    while let Some((a, slot, depth)) = work.pop() {
        let nd = &arena[a as usize];
        if as_leaf[a as usize] {
            let first = tri_idx.len() as u32;
            collect_tris(&arena, a, &mut tri_idx);
            nodes[slot as usize] =
                BvhNode { aabb: nd.aabb, left_first: first, count: tri_idx.len() as u32 - first };
            continue;
        }
        // Rebalance at the point of no return only: while the remaining
        // budget still exceeds what a median treelet over this subtree
        // needs, keep descending PLOC's own topology — the over-deep path
        // flattens as low as possible and every shallow branch survives.
        let remaining = budget.saturating_sub(depth);
        if sub_depth[a as usize] > remaining {
            let need = 33 - nd.count.leading_zeros(); // ~ceil(log2(count)) + 1
            if remaining <= need {
                let first = tri_idx.len();
                collect_tris(&arena, a, &mut tri_idx);
                emit_balanced(&mut nodes, &mut tri_idx, tri_aabb, first, slot, max_leaf as usize);
                rebalanced += 1;
                continue;
            }
        }
        let l = nodes.len() as u32;
        nodes.push(BvhNode { aabb: Aabb::EMPTY, left_first: 0, count: 0 });
        nodes.push(BvhNode { aabb: Aabb::EMPTY, left_first: 0, count: 0 });
        nodes[slot as usize] = BvhNode { aabb: nd.aabb, left_first: l, count: 0 };
        work.push((nd.right, l + 1, depth + 1));
        work.push((nd.left, l, depth + 1));
    }
    if rebalanced > 0 {
        eprintln!("ploc: {rebalanced} over-deep subtrees re-emitted as median treelets");
    }
    Bvh::from_parts(nodes, tri_idx)
}

/// Balanced median-split emit over tri_idx[first..] (a rebalanced PLOC
/// subtree): depth log2(count), leaves at the max_leaf cap. `slot` receives
/// the subtree root.
fn emit_balanced(
    nodes: &mut Vec<BvhNode>,
    tri_idx: &mut [u32],
    tri_aabb: &[Aabb],
    first: usize,
    slot: u32,
    max_leaf: usize,
) {
    let mut work: Vec<(usize, usize, u32)> = vec![(first, tri_idx.len(), slot)];
    while let Some((lo, hi, at)) = work.pop() {
        let mut bb = Aabb::EMPTY;
        for &t in &tri_idx[lo..hi] {
            bb.grow_aabb(&tri_aabb[t as usize]);
        }
        if hi - lo <= max_leaf {
            nodes[at as usize] =
                BvhNode { aabb: bb, left_first: lo as u32, count: (hi - lo) as u32 };
            continue;
        }
        let mid = lo + (hi - lo) / 2;
        let l = nodes.len() as u32;
        nodes.push(BvhNode { aabb: Aabb::EMPTY, left_first: 0, count: 0 });
        nodes.push(BvhNode { aabb: Aabb::EMPTY, left_first: 0, count: 0 });
        nodes[at as usize] = BvhNode { aabb: bb, left_first: l, count: 0 };
        work.push((mid, hi, l + 1));
        work.push((lo, mid, l));
    }
}

/// Append a collapsed subtree's triangles in DFS order (explicit stack —
/// PLOC subtrees can be deep).
fn collect_tris(arena: &[PNode], root: u32, out: &mut Vec<u32>) {
    let mut work = vec![root];
    while let Some(a) = work.pop() {
        let nd = &arena[a as usize];
        if nd.tri != u32::MAX {
            out.push(nd.tri);
        } else {
            work.push(nd.right);
            work.push(nd.left);
        }
    }
}
