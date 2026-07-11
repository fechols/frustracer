//! Static-frame structure replay: record one producing frame's terminal
//! quadtree — every `shade_tile` / `fill_sky` call with its exact inputs — so
//! a later frame with a bit-equal camera basis can re-shade without a single
//! frustum query or cut refinement.
//!
//! Soundness: the quadtree structure, every inherited `t_start`, and every
//! cut are functions of (scene, BVH, basis, rw, rh) only. Jitter stays inside
//! pixel footprints and tile frustums are built through pixel-grid edges, so
//! a bit-equal basis at the same resolution means bit-identical tile frusta —
//! the recorded (t_start, cut) pairs are exactly the claims the leaf-tile
//! inheritance argument needs (CLAUDE.md). Shading parameters (quality, frame
//! index, jitter, G-buffers) are NOT recorded: replay takes them from the
//! fresh `FrameCtx`, so replayed frames advance the RNG / jitter sequence
//! like traced ones. `--check` gates replay-vs-trace bit-identity of tbuf,
//! info, and accum.
//!
//! Written under rayon with the codebase's relaxed-atomic bump pattern: each
//! slot is written by exactly one task (a `fetch_add` reservation), and reads
//! happen on a later frame across the thread-pool join — the same
//! happens-before argument as the accum buffer and TemporalCache.

use crate::frustum::MAX_CUT;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// u32 words per leaf record: packed x0|y0, packed x1|y1, t_start bits,
/// depth, cut offset, cut length.
const LEAF_WORDS: usize = 6;
/// u32 words per sky record: packed x0|y0, packed x1|y1, depth.
const SKY_WORDS: usize = 3;
/// Cut-arena budget in u32 per terminal slot (~1.2× the observed cut mean on
/// the default scene). Overflow poisons the frame — never wrong, one lost
/// replay opportunity, counted by the caller via `valid()`.
const ARENA_PER_TERMINAL: usize = 12;

pub struct LeafRec {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub t_start: f32,
    pub depth: u32,
    pub cut_off: u32,
    pub cut_len: u32,
}

pub struct SkyRec {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub depth: u32,
}

pub struct ReplayCache {
    leaves: Vec<AtomicU32>,
    n_leaves: AtomicU32,
    skies: Vec<AtomicU32>,
    n_skies: AtomicU32,
    cuts: Vec<AtomicU32>,
    cut_top: AtomicU32,
    poisoned: AtomicU32,
    /// Terminal-count capacity (leaves and skies each).
    cap: u32,
}

impl ReplayCache {
    /// Capacity bound: every terminal rect (leaf or sky) spans ≥ 4 px per
    /// axis — the smallest tile that still splits is 9 px (halves 4 + 5), so
    /// nothing smaller than 4 is ever produced (or the root itself is tiny
    /// and there is exactly one interval). Hence ≤ ceil(rw/4)·ceil(rh/4)
    /// terminals of each kind, exact and resolution-only.
    pub fn new(rw: usize, rh: usize) -> Self {
        let cap = rw.div_ceil(4) * rh.div_ceil(4);
        // u16 rect packing: the exclusive corner x1 can equal rw, so the
        // resolution itself must fit — any realistic window does. A larger
        // request would record garbage; refuse to replay instead.
        let packable = rw <= u16::MAX as usize && rh <= u16::MAX as usize;
        ReplayCache {
            leaves: (0..cap * LEAF_WORDS).map(|_| AtomicU32::new(0)).collect(),
            n_leaves: AtomicU32::new(0),
            skies: (0..cap * SKY_WORDS).map(|_| AtomicU32::new(0)).collect(),
            n_skies: AtomicU32::new(0),
            cuts: (0..cap * ARENA_PER_TERMINAL).map(|_| AtomicU32::new(0)).collect(),
            cut_top: AtomicU32::new(0),
            poisoned: AtomicU32::new(if packable { 0 } else { 1 }),
            cap: cap as u32,
        }
    }

    /// Reset for a new recording frame. The poison state is sticky only for
    /// unpackable resolutions (see `new`).
    pub fn begin(&self, rw: usize, rh: usize) {
        self.n_leaves.store(0, Relaxed);
        self.n_skies.store(0, Relaxed);
        self.cut_top.store(0, Relaxed);
        let packable = rw <= u16::MAX as usize && rh <= u16::MAX as usize;
        let fits = rw.div_ceil(4) * rh.div_ceil(4) <= self.cap as usize;
        self.poisoned.store(if packable && fits { 0 } else { 1 }, Relaxed);
    }

    /// Mark the frame unreplayable (capacity overflow, or the capped arm was
    /// reached while recording — structurally unreachable, cheap defense).
    pub fn poison(&self) {
        self.poisoned.store(1, Relaxed);
    }

    pub fn valid(&self) -> bool {
        self.poisoned.load(Relaxed) == 0
    }

    pub fn counts(&self) -> (u32, u32) {
        // Clamp: a poisoned frame's cursors may have run past capacity.
        (self.n_leaves.load(Relaxed).min(self.cap), self.n_skies.load(Relaxed).min(self.cap))
    }

    pub fn push_leaf(
        &self,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        t_start: f32,
        depth: u32,
        cut: &[u32],
    ) {
        debug_assert!(cut.len() <= MAX_CUT && !cut.is_empty());
        let li = self.n_leaves.fetch_add(1, Relaxed) as usize;
        let co = self.cut_top.fetch_add(cut.len() as u32, Relaxed) as usize;
        if li >= self.cap as usize || co + cut.len() > self.cuts.len() {
            self.poison();
            return;
        }
        for (k, &v) in cut.iter().enumerate() {
            self.cuts[co + k].store(v, Relaxed);
        }
        let b = li * LEAF_WORDS;
        self.leaves[b].store(pack_xy(x0, y0), Relaxed);
        self.leaves[b + 1].store(pack_xy(x1, y1), Relaxed);
        self.leaves[b + 2].store(t_start.to_bits(), Relaxed);
        self.leaves[b + 3].store(depth, Relaxed);
        self.leaves[b + 4].store(co as u32, Relaxed);
        self.leaves[b + 5].store(cut.len() as u32, Relaxed);
    }

    pub fn push_sky(&self, x0: usize, y0: usize, x1: usize, y1: usize, depth: u32) {
        let si = self.n_skies.fetch_add(1, Relaxed) as usize;
        if si >= self.cap as usize {
            self.poison();
            return;
        }
        let b = si * SKY_WORDS;
        self.skies[b].store(pack_xy(x0, y0), Relaxed);
        self.skies[b + 1].store(pack_xy(x1, y1), Relaxed);
        self.skies[b + 2].store(depth, Relaxed);
    }

    pub fn leaf(&self, i: usize) -> LeafRec {
        let b = i * LEAF_WORDS;
        let (x0, y0) = unpack_xy(self.leaves[b].load(Relaxed));
        let (x1, y1) = unpack_xy(self.leaves[b + 1].load(Relaxed));
        LeafRec {
            x0,
            y0,
            x1,
            y1,
            t_start: f32::from_bits(self.leaves[b + 2].load(Relaxed)),
            depth: self.leaves[b + 3].load(Relaxed),
            cut_off: self.leaves[b + 4].load(Relaxed),
            cut_len: self.leaves[b + 5].load(Relaxed),
        }
    }

    pub fn sky(&self, i: usize) -> SkyRec {
        let b = i * SKY_WORDS;
        let (x0, y0) = unpack_xy(self.skies[b].load(Relaxed));
        let (x1, y1) = unpack_xy(self.skies[b + 1].load(Relaxed));
        SkyRec { x0, y0, x1, y1, depth: self.skies[b + 2].load(Relaxed) }
    }

    /// Copy a recorded cut into the caller's stack buffer; returns its length.
    pub fn copy_cut(&self, off: u32, len: u32, out: &mut [u32; MAX_CUT]) -> usize {
        crate::temporal::copy_cut_from(&self.cuts, off, len, out)
    }

    /// Exact pixel accounting over the recorded terminals — the check gate
    /// asserts leaf px + sky px == rw·rh (children exactly partition their
    /// parent, so a valid recording tiles the screen).
    pub fn accounting(&self) -> (u64, u64) {
        let (nl, ns) = self.counts();
        let mut leaf_px = 0u64;
        for i in 0..nl as usize {
            let r = self.leaf(i);
            leaf_px += ((r.x1 - r.x0) * (r.y1 - r.y0)) as u64;
        }
        let mut sky_px = 0u64;
        for i in 0..ns as usize {
            let r = self.sky(i);
            sky_px += ((r.x1 - r.x0) * (r.y1 - r.y0)) as u64;
        }
        (leaf_px, sky_px)
    }
}

fn pack_xy(x: usize, y: usize) -> u32 {
    ((x as u32) << 16) | (y as u32 & 0xFFFF)
}

fn unpack_xy(v: u32) -> (usize, usize) {
    ((v >> 16) as usize, (v & 0xFFFF) as usize)
}
