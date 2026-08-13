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
/// Set while several scenes load AT ONCE (the world loader's parallel
/// per-island block). The inner phase row's counters then come from whichever
/// part published last, so a fraction built from them would MIX PARTS and run
/// backwards on screen. Under this flag `phase()` keeps the LABEL — still
/// honest, it names whatever step some part is in — but forces the row
/// INDETERMINATE and leaves the DETAIL the caller set on entry, so the row
/// reads "decoding textures — 7 islands at once" over a marquee. The one
/// truthful fraction during the fan-out is then the STAGE row, which
/// `stage_done` advances once per FINISHED island.
static MULTI: AtomicBool = AtomicBool::new(false);

/// Arm the sink. Interactive-only — see the module note. Idempotent.
pub fn activate() {
    ACTIVE.store(true, Relaxed);
}

/// Enter/leave concurrent-load mode. PRIVATE on purpose — `multi_guard()` is
/// the only way in, so the leave can never be skipped (see `MultiGuard`).
/// Leaving also clears the counters the concurrent phases left behind, so the
/// next sequential `phase()` starts from a clean row.
fn set_multi(on: bool) {
    MULTI.store(on, Relaxed);
    if !on {
        DONE.store(0, Relaxed);
        TOTAL.store(0, Relaxed);
    }
}

/// Scope handle for concurrent-load mode: hold one across a parallel load
/// block and leaving the scope — by RETURN OR PANIC — restores sequential
/// publishing. A plain set/clear pair was written first and rejected: the
/// clear sat after the `collect()` that PROPAGATES a worker's panic, so a
/// failed island load would have left this process-global armed for whatever
/// published next. Today that is unobservable (the scene-load thread's panic
/// is caught by `worker.join()` in `run_window`, which exits), but the flag's
/// reset should not rest on the happy path plus a caller two modules away.
/// Deliberately NOT re-entrant — one fan-out is live at a time, and a nested
/// guard would clear the flag at the INNER drop.
pub struct MultiGuard(());

impl Drop for MultiGuard {
    fn drop(&mut self) {
        set_multi(false);
    }
}

/// Enter concurrent-load mode until the returned guard drops.
#[must_use = "multi mode ends when the guard drops — bind it to a named local"]
pub fn multi_guard() -> MultiGuard {
    set_multi(true);
    MultiGuard(())
}

/// True once `activate()` has run. Publishers gate on this so an inactive
/// (headless) session pays only one relaxed load per call.
#[inline]
pub fn active() -> bool {
    ACTIVE.load(Relaxed)
}

/// ARM the outer stage row: `i` done out of `n`, labelled `name`. The world
/// loader arms it at `(0, n, "")` and then lets `stage_done` count; a
/// single-scene load leaves the row unset.
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

/// Count one island DONE against the stage row, and name the finisher. The
/// world loader's per-island loads run CONCURRENTLY, so "step i of n, now
/// starting X" has no referent — the honest reading is a completion count plus
/// whatever just landed, which is exactly what the HUD's "{done} / {total}
/// {name}" row then says. Called from rayon workers: the count is a relaxed
/// fetch_add and the name a short `Mutex` write, the `tick()` discipline.
pub fn stage_done(name: &str) {
    if !active() {
        return;
    }
    STAGE_DONE.fetch_add(1, Relaxed);
    if let Ok(mut s) = STAGE.lock() {
        s.clear();
        s.push_str(name);
    }
}

/// Enter a sub-phase. `total == 0` means indeterminate (the UI shows a marquee
/// instead of a fraction). Resets the `DONE` counter that `tick()` advances.
///
/// Under a live `multi_guard` the label still lands, but `total` is forced to
/// 0 and the detail is left alone — see `MULTI` for why a fraction is a lie
/// there.
pub fn phase(p: Phase, detail: &str, total: u32) {
    if !active() {
        return;
    }
    let multi = MULTI.load(Relaxed);
    PHASE.store(p.as_u32(), Relaxed);
    DONE.store(0, Relaxed);
    TOTAL.store(if multi { 0 } else { total }, Relaxed);
    if !multi {
        if let Ok(mut d) = DETAIL.lock() {
            d.clear();
            d.push_str(detail);
        }
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

/// Advance by `n` at once. The GPU upload's unit is MiB of work, not one per
/// item: its steps differ in cost by four orders of magnitude (a 4-byte
/// material stream vs a 64 MB texture band), so a one-per-submit counter would
/// race to 99% through the thousands of small texture submits and then sit
/// still through the acceleration-structure build. Same relaxed, lock-free
/// discipline as `tick()`.
#[inline]
pub fn tick_n(n: u32) {
    if !active() {
        return;
    }
    DONE.fetch_add(n, Relaxed);
}

/// Re-label the current phase's DETAIL row WITHOUT touching its counters —
/// the sub-step name inside one long metered phase ("geometry" -> "textures"
/// -> "acceleration structures" inside `GpuUpload`). `phase()` is the wrong
/// call for that: it resets `DONE`, so a bar built from one unit space would
/// snap back to zero at every sub-step.
///
/// Honours `MULTI` for the same reason `phase()` does — during a fan-out the
/// detail belongs to whoever armed the block, not to whichever part published
/// last.
pub fn detail(s: &str) {
    if !active() || MULTI.load(Relaxed) {
        return;
    }
    if let Ok(mut d) = DETAIL.lock() {
        d.clear();
        d.push_str(s);
    }
}

/// The phase's work is DONE — land the bar on exactly 100% whatever the
/// estimate said. Never lowers `DONE` (an overshoot is already clamped by
/// `fraction`), so this closes an UNDERSHOOT and nothing else.
///
/// It exists because a `total` is a cost MODEL and some arms are invisible to
/// it: the GPU upload's denominator is computed before the session picks a
/// tracer, so a `--no-blas-split` boot never ticks the acceleration-structure
/// share it reserved and a wavefront boot ticks a software-tree upload it
/// never reserved. Both are honest estimates; only the endpoint has to be
/// exact, and asserting it once at the end is cheaper and more robust than an
/// accounting identity every arm has to maintain.
pub fn finish_phase() {
    if !active() {
        return;
    }
    let (done, total) = (DONE.load(Relaxed), TOTAL.load(Relaxed));
    if total > done {
        DONE.fetch_add(total - done, Relaxed);
    }
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
    if MULTI.load(Relaxed) {
        return Err("progress sink left in multi mode".into());
    }
    phase(Phase::Bvh, "should be ignored", 10);
    stage(3, 7, "ignored");
    stage_done("ignored");
    tick();
    tick_n(64);
    detail("ignored");
    finish_phase();
    if snapshot().is_some() {
        return Err("inactive sink produced a snapshot".into());
    }
    // …and nothing landed in the state either. `snapshot()` alone cannot see
    // that: the first ACTIVE `phase()` below resets `DONE`, so a publisher
    // that forgot its `active()` guard would leak silently.
    if DONE.load(Relaxed) != 0 || TOTAL.load(Relaxed) != 0 || STAGE_TOTAL.load(Relaxed) != 0 {
        return Err("inactive publish reached the counters".into());
    }
    if DETAIL.lock().map_or(false, |d| !d.is_empty()) {
        return Err("inactive publish reached the detail row".into());
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

    // `tick_n` accumulates like `tick`, and `detail` re-labels WITHOUT
    // resetting the counter — the property the GpuUpload bar rests on (a
    // sub-step rename mid-phase must not snap the bar back to zero).
    phase(Phase::GpuUpload, "geometry", 100);
    tick_n(30);
    tick();
    detail("textures");
    tick_n(9);
    let ok_n = snapshot().is_some_and(|s| {
        s.done == 40 && s.total == 100 && s.detail == "textures" && (s.frac - 0.4).abs() < 1e-6
    });

    // `finish_phase` closes an UNDERSHOOT exactly (the arm whose cost model
    // reserved units nobody ticked) and leaves an OVERSHOOT alone — it must
    // never walk the counter backwards, since `fraction` already clamps and a
    // caller may keep ticking after it.
    finish_phase();
    let ok_fin = snapshot().is_some_and(|s| s.done == 100 && (s.frac - 1.0).abs() < 1e-6);
    tick_n(25);
    finish_phase();
    let ok_fin_over = snapshot().is_some_and(|s| s.done == 125 && s.frac == 1.0);

    // The stage row COUNTS completions: arm at 0/n, then one bump per island,
    // and the row names whichever finished last (the parallel contract).
    stage(0, 7, "");
    stage_done("alpha");
    stage_done("beta");
    let ok_stage = snapshot()
        .is_some_and(|s| s.stage_done == 2 && s.stage_total == 7 && s.stage_name == "beta");

    // Multi mode: a concurrent part's `phase()` may set the label but must NOT
    // publish a fraction (it would mix parts) nor overwrite the detail the
    // fan-out set on entry. `-1.0` is the HUD's marquee case. The call order
    // mirrors world.rs's: the fan-out sets its detail, THEN arms multi — and
    // it arms it through the GUARD, so the shipped entry/leave path is what
    // gets tested rather than a hand-written pair of stores.
    phase(Phase::Parse, "7 islands at once", 0);
    let ok_multi = {
        let _multi = multi_guard();
        phase(Phase::Textures, "some part's detail", 4);
        detail("a part's rename");
        snapshot().is_some_and(|s| {
            s.total == 0
                && s.frac < 0.0
                && s.detail == "7 islands at once"
                && s.phase_label == label(Phase::Textures)
        })
    };
    // The guard's drop left multi mode and cleared the counters the concurrent
    // phases left behind.
    let ok_multi_off =
        !MULTI.load(Relaxed) && snapshot().is_some_and(|s| s.done == 0 && s.total == 0);

    // Restore inactive + clear the counters we dirtied.
    ACTIVE.store(false, Relaxed);
    MULTI.store(false, Relaxed);
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
    if !ok_n {
        return Err("tick_n did not accumulate, or detail() reset the counter".into());
    }
    if !ok_fin {
        return Err("finish_phase did not close the undershoot to exactly 100%".into());
    }
    if !ok_fin_over {
        return Err("finish_phase walked an overshot counter backwards".into());
    }
    if !ok_stage {
        return Err("stage_done did not count completions".into());
    }
    if !ok_multi {
        return Err("multi-mode phase/detail published a fraction or clobbered the detail".into());
    }
    if !ok_multi_off {
        return Err("leaving multi mode left the phase counters dirty".into());
    }
    Ok(())
}
