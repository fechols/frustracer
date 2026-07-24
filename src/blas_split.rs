//! BLAS chunk planner: cut the ray BVH into maximal subtrees under a primitive
//! cap, one DXR bottom-level acceleration structure per subtree.
//!
//! `SceneGpu` USED to build ONE BLAS over `scene.indices` in order plus a single
//! identity-instance TLAS, which is what made `PrimitiveIndex() == tri` true on
//! both GPU intersectors (see `gpu/shaders/rt.hlsli`'s header). This module is
//! the other end of that trade, and now the DEFAULT: the driver's structure is
//! addressable at BVH-subtree granularity — every chunk IS a BVH node, so a node
//! cut has instances to talk to — at the price of one indirection per hit.
//!
//! It ships on for ROBUSTNESS rather than speed. BLAS scratch is sized by the
//! largest single geometry, and THE WORLD's one 34.4M-triangle BLAS asked
//! Intel's driver for 1891 MB of it and REMOVED THE DEVICE mid-boot (NVIDIA
//! asked 276 MB for the same build and survived). Cut at 64k the scratch is a
//! function of one chunk — 3 MB — and the session runs. `--no-blas-split`
//! restores the single BLAS, bit-identically, and is the A/B arm.
//!
//! **The cut is the same shape `frustum::refine_cut` produces**: descend from
//! the root, emit a chunk the moment a subtree fits under `max_prims`. That
//! makes the chunk set an antichain covering every leaf exactly once, which is
//! precisely "a partition of the triangles into spatially coherent groups" —
//! the property a BLAS wants and that a flat `tris[i*64..]` slicing does not
//! have.
//!
//! **On the cap.** 64k prims per chunk puts real scenes at a few hundred
//! chunks of a few thousand triangles each — the regime RT drivers are tuned
//! for, and few enough builds that compaction stays affordable. A cap in the
//! tens makes every scene a soup of 10^5-10^6 single-use BLASes, each paying a
//! header and an instance transition; that regime is reachable (`--blas-split
//! 64`) precisely so it can be measured rather than argued about.
//!
//! Builder-agnostic on purpose: chunks collect their leaves' `tri_idx` spans
//! rather than assuming a subtree owns a contiguous slice of `tri_idx`, so the
//! plan holds for every `--bvh-builder` (`sah|lbvh|ploc|som`), including the
//! agglomerative ones whose leaf order is not a partition of the array by
//! construction.

use crate::bvh::Bvh;
use std::sync::atomic::{AtomicU32, Ordering};

/// Primitives per BLAS — the DEFAULT, and what `--blas-split` given bare takes.
/// See the module doc: this is the "conventional band" cap, not a hardware
/// limit.
pub const DEFAULT_MAX_PRIMS: u32 = 65536;

/// Session cap, 0 == `--no-blas-split` (one BLAS over the whole scene — the
/// pre-feature build, bit-identical). Set once from the CLI before any GPU
/// tracer is built and read at `SceneGpu` upload, the `texture::set_aniso` /
/// `bvh::set_builder` precedent: both tracers construct too deep in the call
/// graph to thread an `Opts` through, and this is a session constant.
static MAX_PRIMS: AtomicU32 = AtomicU32::new(0);

pub fn set_max_prims(cap: Option<u32>) {
    MAX_PRIMS.store(cap.unwrap_or(0), Ordering::Relaxed);
}

/// The armed cap, or `None` for the single-BLAS build.
pub fn max_prims() -> Option<u32> {
    match MAX_PRIMS.load(Ordering::Relaxed) {
        0 => None,
        n => Some(n),
    }
}

/// A cut of the BVH into per-BLAS triangle groups.
///
/// `packed_tris` is chunk-major and holds ORIGINAL triangle ids: it is both the
/// order the per-chunk BLAS index buffer is built in AND the remap the shaders
/// read to recover a tri id from `(InstanceID, PrimitiveIndex)`. Chunk `i` owns
/// `packed_tris[chunk_base[i] .. chunk_base[i + 1]]` — `chunk_base` carries the
/// trailing sentinel so that slice is a subtraction, and the GPU indexes it
/// directly by `InstanceID`.
pub struct BlasPlan {
    pub packed_tris: Vec<u32>,
    /// len == `chunks() + 1` (trailing sentinel == `packed_tris.len()`).
    pub chunk_base: Vec<u32>,
    /// Chunk -> the BVH node it was cut at. Never read by the tracer; this is
    /// the instance <-> node mapping a future cut-driven TLAS rebuild needs,
    /// and the self-test's proof that the chunk set really is a cut.
    pub chunk_node: Vec<u32>,
}

impl BlasPlan {
    pub fn chunks(&self) -> usize {
        self.chunk_node.len()
    }

    /// Triangles in chunk `i`.
    pub fn prims(&self, i: usize) -> u32 {
        self.chunk_base[i + 1] - self.chunk_base[i]
    }

    /// The original tri ids of chunk `i`, in BLAS primitive order.
    pub fn tris(&self, i: usize) -> &[u32] {
        &self.packed_tris[self.chunk_base[i] as usize..self.chunk_base[i + 1] as usize]
    }

    /// (min, mean, max) primitives per chunk — the `gpu scene:` line's numbers.
    pub fn stats(&self) -> (u32, f32, u32) {
        let n = self.chunks();
        if n == 0 {
            return (0, 0.0, 0);
        }
        let mut lo = u32::MAX;
        let mut hi = 0;
        for i in 0..n {
            let p = self.prims(i);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        (lo, self.packed_tris.len() as f32 / n as f32, hi)
    }

    fn emit(&mut self, bvh: &Bvh, root: u32, stack: &mut Vec<u32>) {
        stack.clear();
        stack.push(root);
        while let Some(n) = stack.pop() {
            let node = &bvh.nodes[n as usize];
            if node.count > 0 {
                let f = node.left_first as usize;
                self.packed_tris
                    .extend_from_slice(&bvh.tri_idx[f..f + node.count as usize]);
            } else {
                // Right first so the left child pops first: DFS order, hence a
                // deterministic plan for a fixed tree (the `Bvh::identical`
                // contract extends to the chunking).
                stack.push(node.left_first + 1);
                stack.push(node.left_first);
            }
        }
        self.chunk_base.push(self.packed_tris.len() as u32);
        self.chunk_node.push(root);
    }
}

/// Per-node subtree triangle counts, iteratively (the tree is up to
/// `TRAV_STACK` = 96 deep and recursion here would be one more stack contract
/// to maintain).
fn subtree_tris(bvh: &Bvh) -> Vec<u32> {
    let mut cnt = vec![0u32; bvh.nodes.len()];
    let mut stack: Vec<(u32, bool)> = vec![(0, false)];
    while let Some((n, up)) = stack.pop() {
        let node = &bvh.nodes[n as usize];
        if node.count > 0 {
            cnt[n as usize] = node.count;
        } else if up {
            cnt[n as usize] =
                cnt[node.left_first as usize] + cnt[node.left_first as usize + 1];
        } else {
            stack.push((n, true));
            stack.push((node.left_first, false));
            stack.push((node.left_first + 1, false));
        }
    }
    cnt
}

/// Cut `bvh` into chunks of at most `max_prims` triangles.
///
/// A scene under the cap yields exactly ONE chunk — the current single-BLAS
/// build reached through this path, which is what keeps the armed/unarmed
/// comparison honest on small scenes.
pub fn plan(bvh: &Bvh, max_prims: u32) -> BlasPlan {
    let max_prims = max_prims.max(1);
    let mut p = BlasPlan {
        packed_tris: Vec::with_capacity(bvh.tri_idx.len()),
        chunk_base: vec![0],
        chunk_node: Vec::new(),
    };
    if bvh.nodes.is_empty() || bvh.tri_idx.is_empty() {
        return p;
    }
    let cnt = subtree_tris(bvh);
    let mut stack: Vec<u32> = vec![0];
    let mut scratch: Vec<u32> = Vec::new();
    while let Some(n) = stack.pop() {
        let node = &bvh.nodes[n as usize];
        if cnt[n as usize] <= max_prims {
            p.emit(bvh, n, &mut scratch);
        } else if node.count > 0 {
            // An oversized LEAF — only reachable with `--bvh-maxleaf` above the
            // cap, since the SAH build's leaves hold single digits. There is no
            // node to cut at, so the span itself splits into <= max_prims runs
            // (all of them tagged with the same node id: the cut hook stays
            // truthful, one node simply owns several instances).
            let f = node.left_first as usize;
            let total = node.count as usize;
            let mut off = 0;
            while off < total {
                let take = (total - off).min(max_prims as usize);
                p.packed_tris
                    .extend_from_slice(&bvh.tri_idx[f + off..f + off + take]);
                p.chunk_base.push(p.packed_tris.len() as u32);
                p.chunk_node.push(n);
                off += take;
            }
        } else {
            stack.push(node.left_first + 1);
            stack.push(node.left_first);
        }
    }
    p
}

/// Above this triangle count the sub-64 caps are SKIPPED (loudly). A plan holds
/// `packed_tris` (4 B/tri) plus a `chunk_base`/`chunk_node` pair that are 4 B
/// per CHUNK — negligible at the shipping cap, but cap 1 means one chunk per
/// triangle and the pair becomes another 8 B/tri. On a `--tile 4x4` San Miguel
/// (89.9M tris) that sweep alone would spike ~1 GB inside a gate whose job is
/// structural, so the fine caps stay on trees where they are free. What they
/// exercise (many chunks, the oversized-leaf split) any ordinary `--check`
/// still covers — the default scene is 80k tris.
const FINE_CAP_TRIS: usize = 4_000_000;

/// `--check` gate (pure, DLL-free, GPU-free): the plan's structural contracts
/// at several caps, on the session's real tree.
///
/// What it proves, in the order the GPU depends on it: every chunk fits the cap
/// (the BLAS build's premise), the chunks PARTITION the triangle set exactly
/// (a duplicate would double-shade a triangle, a gap would drop one — and the
/// remap would silently point at the wrong tri either way), the chunk set is a
/// real antichain cut of the tree (the instance <-> node mapping's premise),
/// the oversized-leaf split MUST FIRE (the one branch the cap sweep cannot
/// reach by construction), and the plan is deterministic (the byte-identical-
/// build contract) — the last proven once at the shipping cap, since it is a
/// property of the walk and not of the cap. Anything skipped for size says so.
pub fn self_test(bvh: &Bvh) -> Result<(), String> {
    let n_tris = bvh.tri_idx.len();
    if n_tris == 0 {
        return Err("empty BVH".into());
    }
    let cnt = subtree_tris(bvh);
    if cnt[0] as usize != n_tris {
        return Err(format!("root subtree {} != {} tris", cnt[0], n_tris));
    }
    // The widest leaf the session's builder produced. `max_leaf - 1` is the
    // largest cap that PROVABLY forces the oversized-leaf split — see the
    // dedicated must-fire below, which is the only thing standing between that
    // branch and a silently vacuous gate.
    let max_leaf = bvh.nodes.iter().filter(|n| n.count > 0).map(|n| n.count).max().unwrap_or(0);

    // Small caps exercise many chunks even on a small probe tree; the last is
    // the whole scene, which must degenerate to one chunk.
    let mut caps: Vec<u32> = vec![4096, DEFAULT_MAX_PRIMS, n_tris as u32];
    if n_tris <= FINE_CAP_TRIS {
        caps.splice(0..0, [1u32, 7, 64, max_leaf.saturating_sub(1).max(1)]);
    } else {
        eprintln!(
            "blas-split self-test: {n_tris} tris > {FINE_CAP_TRIS} — skipping the sub-64 caps \
             (memory); the cut/partition/determinism gates still run at 4096, {DEFAULT_MAX_PRIMS} \
             and whole-scene"
        );
    }
    caps.sort_unstable();
    caps.dedup();
    for cap in caps {
        let p = plan(bvh, cap);
        if p.chunks() == 0 {
            return Err(format!("cap {cap}: no chunks"));
        }
        if p.packed_tris.len() != n_tris {
            return Err(format!(
                "cap {cap}: packed {} tris, expected {n_tris}",
                p.packed_tris.len()
            ));
        }
        if p.chunk_base.len() != p.chunks() + 1 {
            return Err(format!("cap {cap}: chunk_base missing its sentinel"));
        }

        // Exact partition: every original tri id exactly once.
        let mut seen = vec![false; n_tris];
        for i in 0..p.chunks() {
            let prims = p.prims(i);
            if prims == 0 || prims > cap {
                return Err(format!("cap {cap}: chunk {i} holds {prims} prims"));
            }
            for &t in p.tris(i) {
                let t = t as usize;
                if t >= n_tris {
                    return Err(format!("cap {cap}: chunk {i} tri id {t} out of range"));
                }
                if seen[t] {
                    return Err(format!("cap {cap}: tri {t} appears in two chunks"));
                }
                seen[t] = true;
            }
        }
        if let Some(t) = seen.iter().position(|s| !s) {
            return Err(format!("cap {cap}: tri {t} covered by no chunk"));
        }

        // The chunk set is an antichain covering every leaf: descend from the
        // root stopping at marked nodes; every path must terminate on one, and
        // every marked node must be reachable without passing another (a chunk
        // nested under a chunk would go unreached).
        let mut marked = vec![false; bvh.nodes.len()];
        for &n in &p.chunk_node {
            marked[n as usize] = true;
        }
        let mut reached = 0usize;
        let mut covered = 0usize;
        let mut stack: Vec<u32> = vec![0];
        while let Some(n) = stack.pop() {
            let node = &bvh.nodes[n as usize];
            if marked[n as usize] {
                reached += 1;
                covered += cnt[n as usize] as usize;
            } else if node.count > 0 {
                return Err(format!("cap {cap}: leaf {n} under no chunk"));
            } else {
                stack.push(node.left_first + 1);
                stack.push(node.left_first);
            }
        }
        let distinct = marked.iter().filter(|m| **m).count();
        if reached != distinct {
            return Err(format!(
                "cap {cap}: {} chunk nodes, {reached} reachable — a chunk nests under another",
                distinct
            ));
        }
        if covered != n_tris {
            return Err(format!("cap {cap}: cut covers {covered} of {n_tris} tris"));
        }
    }

    // A cap at or above the triangle count is the single-BLAS build.
    let whole = plan(bvh, n_tris as u32);
    if whole.chunks() != 1 {
        return Err(format!(
            "whole-scene cap: {} chunks, expected 1",
            whole.chunks()
        ));
    }

    // The oversized-leaf split, MUST-FIRE. It is the one branch no cap in the
    // sweep reaches by construction: it needs a leaf wider than the cap, and
    // `--bvh-maxleaf 1` (or a builder that never widens) would leave it dead
    // while every gate above still passed. `max_leaf - 1` forces it whenever
    // the tree has any multi-triangle leaf at all, and splitting a leaf is the
    // ONLY way two chunks can share a node id — so a duplicate in chunk_node
    // is the observable proof it ran, with no plumbing through `plan`.
    if max_leaf >= 2 && n_tris <= FINE_CAP_TRIS {
        let cap = max_leaf - 1;
        let p = plan(bvh, cap);
        let mut nodes = p.chunk_node.clone();
        nodes.sort_unstable();
        if !nodes.windows(2).any(|w| w[0] == w[1]) {
            return Err(format!(
                "cap {cap} (widest leaf {max_leaf}): no leaf was split — the oversized-leaf \
                 path is untested"
            ));
        }
    } else if max_leaf < 2 {
        eprintln!(
            "blas-split self-test: widest leaf is {max_leaf} tri — the oversized-leaf split \
             is unreachable on this tree, NOT tested"
        );
    } else {
        // No silent caps: this arm is a big scene, where planning at
        // `max_leaf - 1` would allocate one chunk per few triangles.
        eprintln!(
            "blas-split self-test: {n_tris} tris — oversized-leaf must-fire SKIPPED (memory); \
             it runs on every scene at or under {FINE_CAP_TRIS} tris"
        );
    }

    // Determinism (same tree in, same bytes out) — a property of the walk, not
    // of the cap, so it is proven once at the shipping cap rather than holding
    // two full plans alive at every cap in the sweep.
    let d = plan(bvh, DEFAULT_MAX_PRIMS);
    {
        let q = plan(bvh, DEFAULT_MAX_PRIMS);
        if q.packed_tris != d.packed_tris
            || q.chunk_base != d.chunk_base
            || q.chunk_node != d.chunk_node
        {
            return Err("plan is not deterministic".into());
        }
    }
    let (lo, mean, hi) = d.stats();
    eprintln!(
        "blas-split self-test: OK — {} tris -> {} chunks at cap {} (prims min {lo} mean {mean:.0} max {hi})",
        n_tris,
        d.chunks(),
        DEFAULT_MAX_PRIMS
    );
    Ok(())
}
