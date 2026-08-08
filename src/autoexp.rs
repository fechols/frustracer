//! Automatic exposure — the interactive aperture (`--no-auto-exposure` kills,
//! `--exposure-bias EV` composes a manual offset).
//!
//! The tonemap is anchored at a fixed paper white, so an enclosure whose sun is
//! occluded by construction (San Miguel's patio, Sponza's atrium) renders 2-3
//! stops under a lit exterior — physically correct, and exactly why
//! `--cinematic-exposure` exists for offline captures. This module is the
//! interactive counterpart: a DISPLAY-STAGE controller that eases a clamped EV
//! offset toward what the displayed image's mean log2-luminance asks for, and
//! hands the result to the presentation curve as `tone::ToneParams::exposure`.
//!
//! Structure, in the bloom discipline (display-stage, gate-blind by
//! construction):
//! - The MEASUREMENT is open-loop: meters read the pre-glare, PRE-exposure
//!   tonemap source (linear radiance), and exposure exists only in the
//!   presentation curve downstream — no servo feedback, no oscillation mode.
//!   GPU sessions meter with a small reduction pass (gpu/autoexp.rs); the
//!   CPU-presented arms use `meter_accum`/`meter_hdr` below.
//! - The CONTROLLER (`step`) lives in `session()`'s frame loop only, so every
//!   headless path (`--check*`, `--spin`, `--cinematic`) never adapts and the
//!   exposure they see is exactly 1.0 — no gate, `check.png`, or benchmark
//!   moves (the cinematic `exposure_scale` inertness contract; `tone::shape`
//!   additionally BRANCHES around the multiply at 1.0).
//! - Off is structural: `exposure()` returns exactly 1.0 when disabled and
//!   unbiased, and the GPU meter records nothing when the lever is off.
//!
//! `TARGET_LOG2` is the one calibration constant (the `EL_BOOST` /
//! `MOON_E_OVER_PI` one-knob class): the EV the controller asks for is
//! `TARGET_LOG2 - measured`, so the value IS "what a correctly-exposed frame's
//! mean log2-luminance reads". Calibrate against THE WORLD's sunlit boot pose
//! with `FR_AEXP_TRACE=1` (prints measured/ev/scale ~1/s) so an exterior lands
//! at ~0 EV — today's look — and enclosures open up.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use rayon::prelude::*;

/// Mean log2-luminance (paper-white units) of a correctly-exposed frame — the
/// classic mid-grey key, `log2(0.18)`. See the module header for calibration.
pub const TARGET_LOG2: f32 = -2.4739312;
/// EV clamp: how far the controller may pull BACK from today's fixed exposure.
pub const EV_MIN: f32 = -2.0;
/// EV clamp: how far it may open up. Started at 3.0 (the enclosures'
/// 2-3 stops); raised to 5.0 on the first feel-test — a truly dark scene
/// (night island, deep interior) sits 5+ stops under mid-grey, so at +3 it
/// parked on the clamp and still read dark. If a scene still parks here
/// (`FR_AEXP_TRACE=1` shows `ev +5.000` holding), this constant is the knob —
/// raising TARGET_LOG2 instead brightens EVERY scene, exteriors included.
pub const EV_MAX: f32 = 5.0;
/// Park band: once the eased EV is this close to the ask, hold — a controller
/// that chases the last tenth of a stop breathes visibly on stochastic frames.
pub const DEADBAND_EV: f32 = 0.15;
/// Adaptation time constant, seconds (the EMA's `1 - exp(-dt/tau)` gain).
pub const TAU_S: f32 = 1.0;
/// Luminance floor inside the log — bounds a black frame at ~-19.9 stops
/// instead of -inf. Mirrored as a literal in shaders/autoexp.hlsl (the
/// clouds-wind idiom).
const LUMA_EPS: f32 = 1e-6;

/// `--no-auto-exposure` clears (consumed once by main's lever block, toggled
/// live by the settings menu — the bloom shape).
static ENABLED: AtomicBool = AtomicBool::new(true);
/// `--exposure-bias` in EV, f32 bits (the `scene::set_detail_strength` idiom).
/// 0.0 encodes as bit-pattern 0, so the static's default IS the default.
static BIAS: AtomicU32 = AtomicU32::new(0);

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}
pub fn set_bias(ev: f32) {
    BIAS.store(ev.to_bits(), Ordering::Relaxed);
}
pub fn bias() -> f32 {
    f32::from_bits(BIAS.load(Ordering::Relaxed))
}

/// `FR_AEXP_TRACE=1` — the calibration/diagnostic probe (read once).
pub fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FR_AEXP_TRACE").is_ok_and(|v| v != "0"))
}

/// One controller tick: ease the session's EV toward what the measurement asks
/// for. Pure — the caller owns the state (one f32, persisted across resize in
/// `Persist::autoexp_ev`), which is what makes it trivially deterministic and
/// self-testable.
///
/// Shape notes (the `depth_est` controller family): the ask is clamped BEFORE
/// easing, so a pathological frame (all-black, sun-filling) tugs at a bounded
/// target instead of winding the EMA past the clamp; the deadband compares the
/// eased value to the ask and HOLDS inside it (park, don't breathe); dt is
/// clamped so a hitch or a menu-hold gap eases one bounded step rather than
/// snapping.
pub fn step(ev: f32, measured_log2: f32, dt: f32) -> f32 {
    let desired = (TARGET_LOG2 - measured_log2).clamp(EV_MIN, EV_MAX);
    if !desired.is_finite() {
        return ev;
    }
    if (desired - ev).abs() <= DEADBAND_EV {
        return ev;
    }
    let dt = dt.clamp(0.0, 0.25);
    let gain = 1.0 - (-dt / TAU_S).exp();
    ev + (desired - ev) * gain
}

/// The EV -> linear scale the presentation curve consumes
/// (`tone::ToneParams::exposure`). Exactly 1.0 when auto is off and the bias is
/// 0 — the structural off state every bit-identity claim rests on (`exp2` is
/// not trusted to return 1.0-exactly-by-luck; the zero case short-circuits,
/// the `cinematic::exposure_scale` idiom). Bias composes additively in stops
/// and applies even with auto off — a manual exposure lever for free.
pub fn exposure(ev: f32) -> f32 {
    let e = if enabled() { ev } else { 0.0 } + bias();
    if e == 0.0 {
        1.0
    } else {
        e.exp2()
    }
}

/// Rec.709 luma of one linear RGB pixel, floored for the log.
#[inline]
fn log2_luma(r: f32, g: f32, b: f32) -> f64 {
    let l = 0.2126 * r.max(0.0) + 0.7152 * g.max(0.0) + 0.0722 * b.max(0.0);
    (l.max(LUMA_EPS) as f64).log2()
}

/// Mean log2-luminance of an interleaved linear-RGB slice (the `resolve_hdr`
/// arms' already-averaged input). One streaming rayon pass; called only when
/// the lever is on.
pub fn meter_hdr(hdr: &[f32]) -> f32 {
    let n = hdr.len() / 3;
    if n == 0 {
        return TARGET_LOG2;
    }
    let sum: f64 = hdr.par_chunks_exact(3).map(|c| log2_luma(c[0], c[1], c[2])).sum();
    (sum / n as f64) as f32
}

/// Mean log2-luminance of the accumulation buffer (f32 bit patterns holding
/// SUMS — the `render::resolve` input contract, hence the 1/samples).
pub fn meter_accum(accum: &[std::sync::atomic::AtomicU32], samples: u32) -> f32 {
    let n = accum.len() / 3;
    if n == 0 {
        return TARGET_LOG2;
    }
    let inv = 1.0 / samples.max(1) as f32;
    let sum: f64 = accum
        .par_chunks_exact(3)
        .map(|c| {
            let v = |a: &std::sync::atomic::AtomicU32| f32::from_bits(a.load(Ordering::Relaxed)) * inv;
            log2_luma(v(&c[0]), v(&c[1]), v(&c[2]))
        })
        .sum();
    (sum / n as f64) as f32
}

/// Closed-form gates on the controller and the CPU meters. Hooked by `--check`.
/// Saves and restores the process levers (the wide-tiles save/restore pattern —
/// a `--no-auto-exposure --check` must still prove the on arm, and the run's
/// own lever state must survive this test).
pub fn self_test() -> Result<(), String> {
    struct Restore(bool, f32);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_enabled(self.0);
            set_bias(self.1);
        }
    }
    let _restore = Restore(enabled(), bias());

    // (1) The structural off state: disabled + unbiased is EXACTLY 1.0, for
    // any EV the controller might hold — the bit-identity contract.
    set_enabled(false);
    set_bias(0.0);
    for ev in [-3.0, 0.0, 0.7, 3.0] {
        let e = exposure(ev);
        if e.to_bits() != 1.0f32.to_bits() {
            return Err(format!("disabled exposure({ev}) = {e} != exactly 1.0"));
        }
    }

    // (2) EV -> scale anchors, exact (exp2 of integers is exact in f32).
    set_enabled(true);
    if exposure(0.0) != 1.0 || exposure(1.0) != 2.0 || exposure(-1.0) != 0.5 {
        return Err("EV anchors: 0 -> 1.0, +1 -> 2.0, -1 -> 0.5 must be exact".into());
    }

    // (3) Bias composes, and works with auto OFF (the manual-exposure lever):
    // disabled + bias 1.0 is exactly 2.0 whatever the controller holds.
    set_enabled(false);
    set_bias(1.0);
    if exposure(5.0) != 2.0 {
        return Err(format!("disabled+bias 1.0: exposure(5.0) = {} != 2.0", exposure(5.0)));
    }
    set_bias(0.0);
    set_enabled(true);

    // (4) Convergence: a constant ask is reached to within the deadband in
    // ~5 tau, from both sides, without ever overshooting (the EMA gain is in
    // [0, 1), so each step lands strictly between ev and the ask).
    for (start, ask_ev) in [(0.0f32, 2.0f32), (2.5, -1.0)] {
        let measured = TARGET_LOG2 - ask_ev; // desired == ask_ev by construction
        let mut ev = start;
        for _ in 0..300 {
            let next = step(ev, measured, 1.0 / 60.0);
            let lo = start.min(ask_ev) - 1e-4;
            let hi = start.max(ask_ev) + 1e-4;
            if !(lo..=hi).contains(&next) {
                return Err(format!("step overshot: {start} -> {ask_ev}, ev {next}"));
            }
            ev = next;
        }
        if (ev - ask_ev).abs() > DEADBAND_EV + 1e-3 {
            return Err(format!("no convergence: {start} -> {ask_ev}, parked at {ev}"));
        }
    }

    // (5) Clamps: a pathological ask parks INSIDE the legal band, never past.
    let mut hi = 0.0f32;
    let mut lo = 0.0f32;
    for _ in 0..600 {
        hi = step(hi, TARGET_LOG2 - 30.0, 1.0 / 60.0); // asks +30 EV
        lo = step(lo, TARGET_LOG2 + 30.0, 1.0 / 60.0); // asks -30 EV
    }
    if hi > EV_MAX || (EV_MAX - hi) > DEADBAND_EV + 1e-3 {
        return Err(format!("EV_MAX clamp: parked at {hi}, want ~{EV_MAX}"));
    }
    if lo < EV_MIN || (lo - EV_MIN) > DEADBAND_EV + 1e-3 {
        return Err(format!("EV_MIN clamp: parked at {lo}, want ~{EV_MIN}"));
    }

    // (6) Deadband: an ask within the band returns the state BITWISE — the
    // park-don't-breathe contract.
    let parked = step(1.0, TARGET_LOG2 - 1.1, 1.0 / 60.0);
    if parked.to_bits() != 1.0f32.to_bits() {
        return Err(format!("deadband breach: ask 1.1 from 1.0 moved to {parked}"));
    }

    // (7) dt is clamped: a 100 s gap (menu hold, hitch) eases one bounded step,
    // and a huge dt still cannot overshoot.
    let jumped = step(0.0, TARGET_LOG2 - 3.0, 100.0);
    if !(0.0..=3.0).contains(&jumped) {
        return Err(format!("dt clamp: 100 s step landed at {jumped}"));
    }

    // (8) Determinism: identical feeds, identical bits (two controllers).
    let (mut a, mut b) = (0.3f32, 0.3f32);
    for i in 0..200 {
        let m = TARGET_LOG2 - ((i % 7) as f32) * 0.4;
        a = step(a, m, 1.0 / 144.0);
        b = step(b, m, 1.0 / 144.0);
    }
    if a.to_bits() != b.to_bits() {
        return Err("controller not deterministic over identical feeds".into());
    }

    // (9) The CPU meter: a uniform mid-grey frame reads log2(0.18), so the
    // desired EV there is ~0 — the calibration identity. And the accum flavor
    // must agree with the hdr one on the same image (sums at 4 samples).
    let px = 64 * 64;
    let hdr: Vec<f32> = (0..px * 3).map(|_| 0.18).collect();
    let m = meter_hdr(&hdr);
    if (m - 0.18f32.log2()).abs() > 1e-3 {
        return Err(format!("meter_hdr(uniform 0.18) = {m}, want {}", 0.18f32.log2()));
    }
    let accum: Vec<std::sync::atomic::AtomicU32> =
        (0..px * 3).map(|_| std::sync::atomic::AtomicU32::new((0.18f32 * 4.0).to_bits())).collect();
    let ma = meter_accum(&accum, 4);
    if (ma - m).abs() > 1e-3 {
        return Err(format!("meter_accum {ma} disagrees with meter_hdr {m}"));
    }

    eprintln!("autoexp self-test: OK");
    Ok(())
}
