//! Loading-progress sink — the data half of the in-window loading screen.
//!
//! A global, publish-only sink written by the scene loaders (on a worker
//! thread) and read by the ~30 Hz loading-screen loop in `run_window`. It is
//! ZERO-COST WHEN INACTIVE by construction: every publisher early-outs on one
//! relaxed `ACTIVE` load, and `activate()` is called ONLY from the interactive
//! `run_window` path. Every headless suite (`--check*`, `--spin`, `--*-dump`)
//! exits in `main()` before a window exists and never activates the sink, so
//! the gates stay a pure function of the command line and no loud line moves.
//!
//! Two display rows: an outer STAGE (world island i/n — the world loader's
//! per-part loop) and an inner PHASE (the current sub-step: parsing, textures,
//! BVH, …). The publish sites sit at the SAME boundaries as the existing
//! stderr "loud lines" (world.rs's per-island lines, `obj materials:`,
//! `scene: … BVH … ready`), so the screen and the log tell one story.
//!
//! Threading: the counters are relaxed atomics (rayon-safe — texture decode
//! `tick()`s from many workers at once); the two label strings ride a `Mutex`
//! (the reader is a slow UI loop, contention is nil). No render-path,
//! rng-stream, or gate contact — display only, like the HUD it feeds.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::Mutex;

/// The inner sub-step of the load. `label()` is the on-screen text.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Cache,
    Parse,
    Textures,
    Heights,
    Merge,
    Bvh,
    Sidecar,
    GpuUpload,
}

impl Phase {
    fn from_u32(v: u32) -> Phase {
        match v {
            1 => Phase::Cache,
            2 => Phase::Parse,
            3 => Phase::Textures,
            4 => Phase::Heights,
            5 => Phase::Merge,
            6 => Phase::Bvh,
            7 => Phase::Sidecar,
            8 => Phase::GpuUpload,
            _ => Phase::Idle,
        }
    }
    fn as_u32(self) -> u32 {
        match self {
            Phase::Idle => 0,
            Phase::Cache => 1,
            Phase::Parse => 2,
            Phase::Textures => 3,
            Phase::Heights => 4,
            Phase::Merge => 5,
            Phase::Bvh => 6,
            Phase::Sidecar => 7,
            Phase::GpuUpload => 8,
        }
    }
}

/// The on-screen label for a phase. Total over the enum (no `_` arm) so a new
/// variant can't ship without a label — the `self_test` totality check.
pub fn label(p: Phase) -> &'static str {
    match p {
        Phase::Idle => "loading",
        Phase::Cache => "reading cache",
        Phase::Parse => "parsing geometry",
        Phase::Textures => "decoding textures",
        Phase::Heights => "deriving heightfields",
        Phase::Merge => "merging scenes",
        Phase::Bvh => "building BVH",
        Phase::Sidecar => "writing cache",
        Phase::GpuUpload => "uploading to GPU (BC7 + BLAS/TLAS)",
    }
}

static ACTIVE: AtomicBool = AtomicBool::new(false);
static PHASE: AtomicU32 = AtomicU32::new(0);
static DONE: AtomicU32 = AtomicU32::new(0);
static TOTAL: AtomicU32 = AtomicU32::new(0); // 0 => indeterminate (marquee)
static STAGE_DONE: AtomicU32 = AtomicU32::new(0);
static STAGE_TOTAL: AtomicU32 = AtomicU32::new(0);
static STAGE: Mutex<String> = Mutex::new(String::new());
static DETAIL: Mutex<String> = Mutex::new(String::new());

/// Arm the sink. Interactive-only — see the module note. Idempotent.
pub fn activate() {
    ACTIVE.store(true, Relaxed);
}

/// True once `activate()` has run. Publishers gate on this so an inactive
/// (headless) session pays only one relaxed load per call.
#[inline]
pub fn active() -> bool {
    ACTIVE.load(Relaxed)
}

/// Set the outer stage row: "step `i` of `n` — `name`" (1-based `i`). Used by
/// the world loader's per-island loop; a single-scene load leaves it unset.
pub fn stage(i: u32, n: u32, name: &str) {
    if !active() {
        return;
    }
    STAGE_DONE.store(i, Relaxed);
    STAGE_TOTAL.store(n, Relaxed);
    if let Ok(mut s) = STAGE.lock() {
        s.clear();
        s.push_str(name);
    }
}

/// Enter a sub-phase. `total == 0` means indeterminate (the UI shows a marquee
/// instead of a fraction). Resets the `DONE` counter that `tick()` advances.
pub fn phase(p: Phase, detail: &str, total: u32) {
    if !active() {
        return;
    }
    PHASE.store(p.as_u32(), Relaxed);
    DONE.store(0, Relaxed);
    TOTAL.store(total, Relaxed);
    if let Ok(mut d) = DETAIL.lock() {
        d.clear();
        d.push_str(detail);
    }
}

/// Advance the current phase's counter by one (a decoded texture, a solved
/// height map). Relaxed and lock-free — safe to call from every rayon worker.
#[inline]
pub fn tick() {
    if !active() {
        return;
    }
    DONE.fetch_add(1, Relaxed);
}

/// A clamped completion fraction in `[0, 1]`. `total == 0` => 0.0 (the caller
/// treats indeterminate as the marquee case separately via the snapshot).
pub fn fraction(done: u32, total: u32) -> f32 {
    if total == 0 {
        return 0.0;
    }
    (done as f32 / total as f32).clamp(0.0, 1.0)
}

/// What the loading loop reads each UI tick.
#[derive(Clone, Default)]
pub struct Snapshot {
    pub stage_done: u32,
    pub stage_total: u32,
    pub stage_name: String,
    pub phase_label: &'static str,
    pub detail: String,
    pub done: u32,
    pub total: u32,
    /// `-1.0` when indeterminate (marquee); else the clamped `[0,1]` fraction.
    pub frac: f32,
}

/// The current progress, or `None` when the sink is inactive (headless). The
/// UI loop maps `None` to a blank/black frame — a loading screen only ever
/// exists once `activate()` has run.
pub fn snapshot() -> Option<Snapshot> {
    if !active() {
        return None;
    }
    let done = DONE.load(Relaxed);
    let total = TOTAL.load(Relaxed);
    Some(Snapshot {
        stage_done: STAGE_DONE.load(Relaxed),
        stage_total: STAGE_TOTAL.load(Relaxed),
        stage_name: STAGE.lock().map(|s| s.clone()).unwrap_or_default(),
        phase_label: label(Phase::from_u32(PHASE.load(Relaxed))),
        detail: DETAIL.lock().map(|s| s.clone()).unwrap_or_default(),
        done,
        total,
        frac: if total == 0 { -1.0 } else { fraction(done, total) },
    })
}

/// Pure-math + inactive-contract gate, wired into `--check`.
pub fn self_test() -> Result<(), String> {
    // Inactive no-op: with the sink un-armed (the headless state), a publish
    // must change nothing observable and `snapshot()` must stay `None`.
    if active() {
        return Err("progress sink already active in a headless run".into());
    }
    phase(Phase::Bvh, "should be ignored", 10);
    stage(3, 7, "ignored");
    tick();
    if snapshot().is_some() {
        return Err("inactive sink produced a snapshot".into());
    }

    // Fraction clamp + endpoints.
    if fraction(0, 0) != 0.0 {
        return Err("fraction(_, 0) must be 0".into());
    }
    if fraction(5, 10) != 0.5 || fraction(10, 10) != 1.0 {
        return Err("fraction endpoints wrong".into());
    }
    if fraction(99, 10) != 1.0 {
        return Err("fraction must clamp to 1".into());
    }

    // Label totality: every phase (0..=8) maps to a non-empty label.
    for v in 0..=8u32 {
        if label(Phase::from_u32(v)).is_empty() {
            return Err(format!("empty label for phase {v}"));
        }
    }
    // Round-trip the enum discriminants (a mis-numbered `as_u32`/`from_u32`
    // would desync the wire the atomics carry).
    for p in [
        Phase::Idle,
        Phase::Cache,
        Phase::Parse,
        Phase::Textures,
        Phase::Heights,
        Phase::Merge,
        Phase::Bvh,
        Phase::Sidecar,
        Phase::GpuUpload,
    ] {
        if Phase::from_u32(p.as_u32()) != p {
            return Err("phase enum round-trip mismatch".into());
        }
    }

    // Tick monotonicity: briefly OWN the global (headless is single-threaded
    // here), exercise the active path, then restore the inactive state so the
    // rest of the run sees the sink exactly as it found it.
    activate();
    phase(Phase::Textures, "probe", 4);
    tick();
    tick();
    let ok = snapshot().is_some_and(|s| s.done == 2 && s.total == 4 && (s.frac - 0.5).abs() < 1e-6);
    // Restore inactive + clear the counters we dirtied.
    ACTIVE.store(false, Relaxed);
    PHASE.store(0, Relaxed);
    DONE.store(0, Relaxed);
    TOTAL.store(0, Relaxed);
    STAGE_DONE.store(0, Relaxed);
    STAGE_TOTAL.store(0, Relaxed);
    if let Ok(mut s) = STAGE.lock() {
        s.clear();
    }
    if let Ok(mut d) = DETAIL.lock() {
        d.clear();
    }
    if !ok {
        return Err("active tick/fraction did not track".into());
    }
    Ok(())
}
