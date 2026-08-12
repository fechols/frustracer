//! Automatic exposure — the interactive aperture (`--no-auto-exposure` kills,
//! `--exposure-bias EV` composes a manual offset). DEFAULT ON since 2026-08-10
//! (the user's call). THREE MOVES: default-ON for one day, OFF on 2026-08-08
//! (with RTGI on by default enclosures light themselves, so the aperture was
//! holding at ~1.0 and paying for nothing), and ON again now — after the
//! 2026-08-10 `EV_MAX` rein-in to +2.0, which is what makes an armed session
//! a bounded 4x rather than the 32x that washed genuinely dark scenes into
//! dusk. `--no-auto-exposure` is the kill lever and restores exactly 1.0
//! (plus any `--exposure-bias`), i.e. the pre-feature look.
//!
//! The tonemap is anchored at a fixed paper white, so an enclosure whose sun is
//! occluded by construction (San Miguel's patio, Sponza's atrium) renders 2-3
//! stops under a lit exterior — physically correct, and exactly why
//! `--cinematic-exposure` exists for offline captures. This module is the
//! interactive counterpart: a DISPLAY-STAGE controller that eases a clamped EV
//! offset toward what the displayed image's mean log2-luminance asks for, and
//! hands the result to the presentation curve as `tone::ToneParams::exposure`.
//!
//! WHERE the aperture is spent is a lever — `--autoexp-mode tonemap|lights`,
//! see `Mode`. The default `tonemap` arm is the presentation-curve multiply
//! described above and below; the `lights` arm spends the same EV upstream, as
//! a gain on every emitter in the scene (`scene::apply_light_gain`). The two
//! compute the same image because the rendering equation is linear in emitted
//! radiance, which `--check`'s `light-gain` gate proves; they differ in what
//! the gain passes through, and in that the lights arm has no spike guard.
//!
//! Structure, in the bloom discipline (display-stage, gate-blind by
//! construction):
//! - The MEASUREMENT is open-loop: meters read the pre-glare, PRE-exposure
//!   tonemap source (linear radiance), and exposure exists only in the
//!   presentation curve downstream — no servo feedback, no oscillation mode.
//!   GPU sessions meter with a small reduction pass (gpu/autoexp.rs); the
//!   CPU-presented arms use `meter_accum`/`meter_hdr` below. The lights arm
//!   keeps that property by SUBTRACTING the gain the measured frame was
//!   rendered with (`degain`) — without it the loop closes.
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
/// EV clamp: how far it may open up. Four moves — 3.0 (the enclosures' 2-3
/// stops) -> 5.0 on the first feel-test (a truly dark scene sits 5+ stops
/// under mid-grey, so at +3 it parked on the clamp and still read dark) ->
/// 3.0 on 2026-08-10 (the rein-in: +5 is a 32x aperture that washes a
/// genuinely dark scene into dusk) -> 2.0 the same day, on the FIREFLY
/// report below.
///
/// PARKING ON THIS CLAMP IS THE DESIGN, not a failure to converge: a night
/// scene is SUPPOSED to read dark, and the clamp is what stops the controller
/// from arguing with that. `--exposure-bias` is the per-session escape, and
/// raising TARGET_LOG2 is NOT the alternative — that brightens every scene,
/// exteriors included.
///
/// WHY THE LAST MOVE, and what it does and does not buy: XeSS+NRD sessions in
/// dark emissive scenes show fireflies, and the aperture is why they became
/// VISIBLE rather than why they exist (they are 1-spp RTGI-bounce variance on
/// small bright emitters — see `upscaler_defaults`, which deliberately leaves
/// such a session without the deterministic cluster NEE that would remove
/// them at the source). The aperture only decides where the frame sits on the
/// tonemap: at 32x a dark frame sat in the curve's SATURATING region, where an
/// outlier and its surround compress together and the noise is laundered by
/// the rolloff; lower apertures sit in the near-LINEAR region, which preserves
/// contrast — including outlier contrast. So this constant trades one
/// population for the other and cannot fix either: it damps a MODERATE outlier
/// (sub-paper-white, so less gain really is less amplification) while making a
/// SATURATING one more salient (the dot pins white at any of these apertures
/// while the background keeps darkening). Do not read a firefly change here as
/// evidence about the denoiser; the honest fix lives upstream.
pub const EV_MAX: f32 = 2.0;
/// Park band: once the eased EV is this close to the ask, hold — a controller
/// that chases the last tenth of a stop breathes visibly on stochastic frames.
pub const DEADBAND_EV: f32 = 0.15;
/// Adaptation time constant, seconds (the EMA's `1 - exp(-dt/tau)` gain).
pub const TAU_S: f32 = 1.0;
/// Luminance floor inside the log — bounds a black frame at ~-19.9 stops
/// instead of -inf. Mirrored as a literal in shaders/autoexp.hlsl (the
/// clouds-wind idiom).
const LUMA_EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// The SPIKE GUARD — a per-pixel exemption from the aperture's boost.
// ---------------------------------------------------------------------------
//
// The aperture is one scalar, so it has no way to spend its gain selectively:
// it brightens a dark scene and the stochastic dots in it by the same factor.
// The guard withholds the boost from pixels that are outliers against their own
// surround, under ONE invariant that is what makes it safe to arm by default:
//
//     auto-exposure may never make an outlier pixel BRIGHTER than the
//     auto-exposure-off image would have made it.
//
// So the exemption only ever relaxes toward 1.0 and never past it, and the
// worst case of a false positive is that a glint is presented at exactly its
// `--no-auto-exposure` brightness — a look that already ships. Contrast the
// NRD residual cap (`nrd::oracle::rclamp_scale`), whose correction can pull a
// real emissive texel BELOW truth and which is therefore default-off.
//
// SCOPE, stated because it bounds what a feel-test can expect: the tonemap
// saturates, so a spike already deep in the rolloff (radiance >> paper white)
// pins white at 1x and at 4x alike and the aperture never made it worse. The
// population auto-exposure genuinely CREATES is the near-linear one — a dot at
// ~0.3 over a ~0.02 surround, where a 4x aperture roughly doubles the absolute
// display gap. That is what this removes, and nothing else. The fireflies
// themselves are 1-spp bounce variance and the fix for THEM is upstream
// (`--emissive-lights`; see EV_MAX's note).
//
// COST, measured rather than assumed (4090, 1080p, guard on vs off, the aperture
// verified BOOSTING in both arms via FR_AEXP_TRACE — a null result at exposure
// <= 1 would be vacuous, since the block is structurally unreachable there):
//
//   * GPU present (bistro Exterior --tod 22, XeSS+NRD, parked terrace lamp pose,
//     aperture 3.61x, --no-vsync --no-fg, 12 samples/arm): 140.36 vs 140.98 fps
//     median = +0.031 ms/frame, +0.4% — INSIDE the run-to-run band (the arms'
//     ranges overlap almost entirely). 8 extra point loads in one fullscreen PS
//     is not something a 4090 notices.
//   * CPU present (procedural, `--cpu --no-upscale --exposure-bias 2`, two
//     interleaved reps): +0.8 and +2.3 ms on a ~59 ms frame, ~2.5%. This arm
//     pays what the GPU one does not — `guard_plane` materializes a luma plane
//     AND an exposure plane per frame — and it is the reason the guard is a
//     prepass here instead of 8 taps inlined into `resolve`'s atomic read loop.
//     Do NOT re-measure this on a heavy scene: CPU bistro traces at ~1000
//     ms/frame and swamps the present stage by three orders of magnitude (that
//     attempt read "no effect" and was measuring the tracer).
//   * TEMPORAL STABILITY (FRUSTRACER_STAB, the same parked pose, 30 windows per
//     arm): median 0.230/255 in BOTH arms. The concern was a pixel oscillating
//     across the ramp between frames, which is exactly what that instrument
//     sees; the smooth ramp is doing its job.

/// Ring radius in SOURCE texels — the 8 taps sit at the four axial and four
/// diagonal offsets of +/-`R`, a DONUT rather than the adjacent 3x3 ring both
/// sibling clamps use.
///
/// The departure is forced by where this runs: at the present funnel, AFTER the
/// upscaler, so a 1-px firefly at trace res is a several-px blob at window res.
/// An adjacent ring samples the blob's own body, `ring_max` comes back hot, and
/// the K_MAX arm below shields the very thing being hunted.
///
/// MEASURED, and this is why it is 4 rather than the 2 it shipped as for an
/// hour (`FR_AEXP_GUARD_TRACE=1`, bistro Exterior `--tod 22`, XeSS+NRD at
/// 1080p, the terrace and street lamp poses):
///
/// | r | terrace fired | street fired | street max luma/ring_max |
/// |---|---|---|---|
/// | 2 |  84 px | 255 px |  43.6 |
/// | 4 | 455 px | 1287 px | 919.1 |
///
/// The last column is the finding: at radius 2 the brightest spike's ring is
/// still INSIDE the spike, so `ring_max` tracks it and the shield fires; at
/// radius 4 the ring clears the blob and the true contrast — three orders of
/// magnitude — finally shows. The 5x rise in fired pixels is the guard reaching
/// the population it was built for, and it is still only 0.02-0.06% of frame.
pub const GUARD_RADIUS: i32 = 4;
/// The ring-MEAN multiplier. A pixel must beat this many times its surround's
/// mean to be treated as a spike.
///
/// MEASURED, and deliberately NOT
/// `nrd::oracle::RCLAMP_K_MEAN`/`frd::FIREFLY_K` (both 8.0, with a recorded
/// intent that the tree's clamps "speak one value"): those operate on raw 1-spp
/// radiance over an adjacent ring, while this operates on DISPLAY luma over the
/// `GUARD_RADIUS` donut after a temporal upscaler has already integrated the
/// dot toward its surround, so the ratios here are structurally smaller. On the
/// frames this exists for (`FR_AEXP_GUARD_TRACE=1`, bistro Exterior `--tod 22`,
/// XeSS+NRD 1080p, the terrace and street lamp poses) the `luma/ring_mean`
/// p99.9 is only **2.39 / 2.92** — i.e. at 8.0 this would have fired on
/// NOTHING, which is the "it gated nothing" failure `RCLAMP_K_HARD` records,
/// and a guard that cannot fire cannot be gated either.
///
/// `FR_AEXP_GUARD_K=<mean>,<max>` is the sweep lever, and the histogram to pick
/// from is the trace's `luma/ring_mean` column (never `luma/cap`, which is a
/// function of this very constant — see `guard_trace_dump`).
pub const GUARD_K_MEAN: f32 = 4.0;
/// The ring-MAX multiplier, and the term that protects RESOLVED features. The
/// cap is a `max` of the two, so a lone stochastic dot (whose ring has
/// `mean ~= max`) is bound by the mean term, while a real bright feature — a
/// lamp, the sun's limb, one of `fireflies.rs`'s Gaussian glow splats — has at
/// least one bright neighbour, giving `max >> mean`, and is shielded. Both
/// reductions come out of one loop, so the discrimination is free.
/// (`nrd::oracle::rclamp_scale`'s conjunction, same rationale.)
///
/// UNSWEPT, unlike `GUARD_K_MEAN`, and the honest reason is that this term is
/// a SHIELD rather than a threshold: it can only ever raise the cap, so a value
/// that is too low merely lets more pixels through to the mean term (which was
/// measured) while one that is too high protects everything. The `--tod 22`
/// trace records `luma/ring_max` reaching 919 at the radius this ships at, so
/// the two terms are separated by orders of magnitude on real content and the
/// exact value here is not what decides anything. `FR_AEXP_GUARD_K`'s second
/// field sweeps it if a feel-test ever finds a resolved feature dimmed.
pub const GUARD_K_MAX: f32 = 2.0;
/// Ramp width, as a multiple of the cap: the exemption fades in over
/// `luma/cap` in `[1, RAMP]`. Must be > 1 — that is a precondition of the
/// ramp's own divisor, not a taste limit; at exactly 1 the fade divides by
/// zero. A smooth ramp rather than a binary pin so a pixel sitting near the
/// threshold cannot pop between two brightnesses on consecutive frames.
///
/// Not swept: with `K_MEAN` measured, the interesting question was WHICH pixels
/// are exempted rather than how fast the exemption arrives, and 2.0 spans a
/// factor-of-two band that the measured p99.9 sits just below. It has no env
/// lever for the same reason — add one if a feel-test ever objects to the
/// transition rather than to the threshold.
pub const GUARD_RAMP: f32 = 2.0;
/// Fewer in-range taps than this and the pixel is left UNGUARDED — a
/// 3-neighbour estimate is not a neighbourhood (`RCLAMP_MIN_RING`'s rule). Taps
/// off the frame are EXCLUDED rather than replicated (the NRD sentinel
/// convention, not FRD's clamp-to-edge), so the frame border is simply
/// un-guarded instead of guarded against a ring of copies of itself.
pub const GUARD_MIN_RING: u32 = 4;

/// The guard's tunables, resolved once per process. Passed explicitly rather
/// than read from a global inside `guard_scale` so the rule stays a pure
/// function of its inputs and `self_test` can sweep it without touching process
/// state — and so the HLSL twin, which receives the same numbers as injected
/// defines, is scored against exactly what the CPU used.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GuardParams {
    pub k_mean: f32,
    pub k_max: f32,
    pub ramp: f32,
    pub radius: i32,
}

impl Default for GuardParams {
    fn default() -> Self {
        Self { k_mean: GUARD_K_MEAN, k_max: GUARD_K_MAX, ramp: GUARD_RAMP, radius: GUARD_RADIUS }
    }
}

/// `--auto-exposure` spells the default (consumed once by main's lever block,
/// toggled live by the settings menu — the bloom shape). DEFAULT ON — the
/// default is DUPLICATED in cli.rs's `defaults()` `autoexp` field AND
/// settings.rs's menu-row `Toggle { default }`; flip all three in lockstep.
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

/// WHERE the aperture is spent — `--autoexp-mode tonemap|lights`.
///
/// The rendering equation is LINEAR in emitted radiance, so scaling every
/// emitter by `g` scales every path's radiance by `g` exactly: the two arms
/// compute the same image, and `--check`'s `light-gain` gate proves it. What
/// differs is everything the gain passes THROUGH on the way:
///
/// - `Tonemap` (the default, and the shipped behaviour bit-for-bit): the EV
///   becomes `tone::ToneParams::exposure`, a pre-curve multiply at the
///   presentation stage. The renderer never sees it, so the upscalers and
///   denoisers integrate an un-gained scene and their ABSOLUTE clamps
///   (`--fsr-max-radiance`, NRD/FRD's firefly caps) sit at a fixed scene
///   brightness rather than at the presented one.
/// - `Lights`: the EV becomes `Scene`-level light magnitudes (see
///   `scene::apply_light_gain`) and the presentation curve stays at exactly
///   1.0. The denoisers then see the brightened signal — arguably the right
///   place for their clamps to act — at the cost of making every EV move a
///   LIGHTING change the temporal histories must absorb (they do; it is the
///   TOD-scrub class) and of losing the SPIKE GUARD, which is inherently
///   display-stage: it works by relaxing a per-PIXEL display exposure toward
///   1.0, and a global light gain has no per-pixel lever. That last point is
///   why this is an A/B lever and not a replacement.
///
/// Exactly one destination is armed at a time, structurally:
/// `display_exposure(ev) * light_gain(ev) == exposure(ev)` with the other
/// factor exactly 1.0 (self_test arm 15).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// The aperture is a pre-curve multiply at the tonemap (the default).
    Tonemap,
    /// The aperture is a gain on every emitter in the scene.
    Lights,
}

impl Mode {
    /// The CLI/settings vocabulary — one parser, so `cli.rs` and a settings
    /// file cannot disagree about what the words mean.
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "tonemap" => Some(Mode::Tonemap),
            "lights" => Some(Mode::Lights),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Tonemap => "tonemap",
            Mode::Lights => "lights",
        }
    }
}

/// `--autoexp-mode` (consumed once by main's lever block, flipped live by the
/// settings menu — the bloom shape). DEFAULT `Lights` since 2026-08-11 (the
/// user's call, after a feel-test); `--autoexp-mode tonemap` is the opt-out and
/// the A/B arm. Like `ENABLED`, the default is DUPLICATED in cli.rs's
/// `defaults()` and settings.rs's menu-row `Cycle { default_ix }`; flip all
/// three in lockstep, and `cli::self_test` pins that they agree.
///
/// CONSEQUENCE OF THE DEFAULT, stated because nothing else will say it: the
/// SPIKE GUARD is now unreachable in a default session. Its arming test is
/// `exposure > 1.0`, and this arm holds display exposure at exactly 1.0 — so
/// `--autoexp-spike-guard`, itself default-ON, is inert unless the session also
/// passes `--autoexp-mode tonemap`. That is inherent, not an oversight: the
/// guard withholds a per-PIXEL share of a display multiply, and a global light
/// gain has no per-pixel lever to withhold.
static MODE: AtomicU32 = AtomicU32::new(Mode::Lights as u32);

pub fn set_mode(m: Mode) {
    MODE.store(m as u32, Ordering::Relaxed);
}
pub fn mode() -> Mode {
    if MODE.load(Ordering::Relaxed) == Mode::Lights as u32 { Mode::Lights } else { Mode::Tonemap }
}

/// `--no-autoexp-spike-guard` kills, `--autoexp-spike-guard` spells the
/// default. DEFAULT ON, and — like `ENABLED` — the default is DUPLICATED in
/// cli.rs's `defaults()` and settings.rs's menu-row `Toggle { default }`; flip
/// all three in lockstep.
static GUARD: AtomicBool = AtomicBool::new(true);
/// `--autoexp-spike-strength` in [0, 1], f32 bits. 1.0 = a full exemption (an
/// outlier is presented at exactly its unboosted brightness); 0.0 is the
/// structural off state.
static GUARD_STRENGTH: AtomicU32 = AtomicU32::new(1.0f32.to_bits());

pub fn set_guard(on: bool) {
    GUARD.store(on, Ordering::Relaxed);
}
pub fn guard() -> bool {
    GUARD.load(Ordering::Relaxed)
}
pub fn set_guard_strength(k: f32) {
    GUARD_STRENGTH.store(k.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}
pub fn guard_strength() -> f32 {
    f32::from_bits(GUARD_STRENGTH.load(Ordering::Relaxed))
}

/// The strength the presentation curve should actually use: the lever value, or
/// exactly 0.0 when the guard is off. One place decides, so no call site can
/// arm the block while the lever says otherwise.
pub fn guard_strength_live() -> f32 {
    if guard() { guard_strength() } else { 0.0 }
}

/// `FR_AEXP_GUARD_K=<mean>,<max>` and `FR_AEXP_GUARD_R=<n>` — the calibration
/// sweep levers (read once). Loud on departure AND loud-plus-default on an
/// illegal value: a mistyped sweep cell that silently measured the shipping
/// constants is the exact failure the `FR_LEAF` rule exists to prevent.
pub fn guard_params() -> GuardParams {
    static P: std::sync::OnceLock<GuardParams> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        let mut p = GuardParams::default();
        if let Ok(v) = std::env::var("FR_AEXP_GUARD_K") {
            let f: Vec<Option<f32>> = v.split(',').map(|s| s.trim().parse::<f32>().ok()).collect();
            match f.as_slice() {
                [Some(m), Some(x)] if *m > 0.0 && *x > 0.0 => {
                    p.k_mean = *m;
                    p.k_max = *x;
                    eprintln!("FR_AEXP_GUARD_K: k_mean {m}, k_max {x} (default {GUARD_K_MEAN}, {GUARD_K_MAX})");
                }
                _ => eprintln!(
                    "FR_AEXP_GUARD_K: '{v}' unrecognized (legal: <mean>,<max>, both > 0) — \
                     using {GUARD_K_MEAN},{GUARD_K_MAX}"
                ),
            }
        }
        if let Ok(v) = std::env::var("FR_AEXP_GUARD_R") {
            match v.trim().parse::<i32>() {
                Ok(r) if (1..=8).contains(&r) => {
                    p.radius = r;
                    eprintln!("FR_AEXP_GUARD_R: ring radius {r} (default {GUARD_RADIUS})");
                }
                _ => eprintln!("FR_AEXP_GUARD_R: '{v}' unrecognized (legal: 1..=8) — using {GUARD_RADIUS}"),
            }
        }
        p
    })
}

/// `FR_AEXP_GUARD_TRACE=1` — the calibration probe: dumps the `luma/cap`
/// distribution so the K constants are picked from a measured frame rather than
/// inherited (read once).
pub fn guard_trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FR_AEXP_GUARD_TRACE").is_ok_and(|v| v != "0"))
}

/// The guard's compile-time constants as HLSL defines, pasted ahead of
/// tonemap.hlsl (the `spp_defs`/`detail_defs` idiom).
///
/// Emitting them from `guard_params()` — rather than letting the shader's own
/// `#ifndef` fallbacks stand — is what makes the `FR_AEXP_GUARD_*` sweep levers
/// provably REACH the GPU arm. A lever that prints a departure line and then
/// measures the shipping constants is this tree's most expensive recurring bug
/// (the probe-reach trap); `{:?}` rather than `{}` so an integral value still
/// prints as `4.0` and cannot be read as an HLSL int.
pub fn guard_defs() -> String {
    let p = guard_params();
    format!(
        "#define AEXP_GUARD_K_MEAN {:?}\n\
         #define AEXP_GUARD_K_MAX {:?}\n\
         #define AEXP_GUARD_RAMP {:?}\n\
         #define AEXP_GUARD_R {}\n\
         #define AEXP_GUARD_MIN_RING {}\n",
        p.k_mean, p.k_max, p.ramp, p.radius, GUARD_MIN_RING
    )
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

/// The session's total aperture in STOPS: the controller's eased EV when auto
/// is on, plus the manual bias, which applies even with auto off (a free manual
/// exposure lever). Mode-independent — this is the aperture, not where it is
/// spent.
pub fn ev_total(ev: f32) -> f32 {
    (if enabled() { ev } else { 0.0 }) + bias()
}

/// Stops -> linear scale, with the structural off state every bit-identity
/// claim in this feature rests on: exactly 0 stops returns exactly 1.0, because
/// `exp2` is not trusted to return 1.0-by-luck (the `cinematic::exposure_scale`
/// idiom).
#[inline]
fn scale_of(stops: f32) -> f32 {
    if stops == 0.0 { 1.0 } else { stops.exp2() }
}

/// The session's total aperture as a linear scale, whichever arm spends it.
/// `display_exposure(ev) * light_gain(ev) == exposure(ev)` by construction.
pub fn exposure(ev: f32) -> f32 {
    scale_of(ev_total(ev))
}

/// The aperture in stops as the SCENE will carry it — `ev_total` under
/// `Mode::Lights`, and exactly 0.0 otherwise. Kept in stops as well as a scale
/// because the meter de-gain below is an additive correction in log space.
pub fn light_gain_ev(ev: f32) -> f32 {
    match mode() {
        Mode::Lights => ev_total(ev),
        Mode::Tonemap => 0.0,
    }
}

/// What `scene::apply_light_gain` multiplies every emitter by. Exactly 1.0 in
/// the tonemap arm, so that arm's scene is bitwise the pre-feature one.
pub fn light_gain(ev: f32) -> f32 {
    scale_of(light_gain_ev(ev))
}

/// What the presentation curve consumes (`tone::ToneParams::exposure`).
/// Exactly 1.0 in the lights arm, so that arm's tonemap is bitwise the
/// pre-exposure one — and therefore the SPIKE GUARD, whose own arming test is
/// `exposure > 1.0`, is structurally absent there (see `Mode`).
pub fn display_exposure(ev: f32) -> f32 {
    scale_of(ev_total(ev) - light_gain_ev(ev))
}

/// Undo the light gain a measured frame was RENDERED with, so the controller
/// stays open-loop.
///
/// This is the one thing the lights arm genuinely has to add. Today's meter
/// reads a frame the aperture never touched, so `step`'s ask is a statement
/// about the SCENE. Under `Mode::Lights` the gain feeds the renderer, so a
/// frame rendered at `2^e` measures `log2(L) + e` and feeding that back would
/// close a servo loop around a `FRAMES_IN_FLIGHT`-delayed measurement.
/// Subtracting the stops that frame was rendered with recovers `log2(L)`
/// exactly, which makes the controller mathematically identical to today's —
/// so its convergence, deadband and clamp gates transfer unchanged (arm 16).
///
/// `gain_ev` is the value `light_gain_ev` returned when that frame was
/// RECORDED, not the current one — the GPU meter's readback lands
/// `FRAMES_IN_FLIGHT` frames later, so `gpu::autoexp` carries it per ring slot.
/// The 0.0 case returns the measurement BITWISE (a branch, never `- 0.0`),
/// which is what keeps the tonemap arm's controller untouched.
pub fn degain(measured_log2: f32, gain_ev: f32) -> f32 {
    if gain_ev == 0.0 { measured_log2 } else { measured_log2 - gain_ev }
}

/// Rec.709 luma of one linear RGB pixel, clamped non-negative. The ONE weight
/// triple in this module — the meters and the spike guard must agree about what
/// "bright" means, and shaders/autoexp.hlsl + shaders/tonemap.hlsl mirror it as
/// a literal (the clouds-wind idiom).
#[inline]
pub fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r.max(0.0) + 0.7152 * g.max(0.0) + 0.0722 * b.max(0.0)
}

/// Rec.709 luma of one linear RGB pixel, floored for the log.
#[inline]
fn log2_luma(r: f32, g: f32, b: f32) -> f64 {
    (luma(r, g, b).max(LUMA_EPS) as f64).log2()
}

/// THE RULE — the per-pixel exposure a pixel should actually receive, given its
/// own luma and the reductions over its ring. The single source of truth: the
/// HLSL twin in tonemap.hlsl is scored against this by `--check-gpu`'s M12, and
/// the CPU-presented arms reach it through `guard_plane`.
///
/// `nv` is the number of IN-RANGE ring taps that contributed to `ring_mean` /
/// `ring_max`; out-of-range taps are excluded from both, not zeroed (a zeroed
/// tap would deflate the mean and turn the frame border into a false-positive
/// factory, the mirror of the "sky base would inflate the cap into vacuity"
/// note on the NRD cap).
///
/// Every off arm is a BRANCH returning `exposure` BITWISE — never a computed
/// `lerp(e, e, 0)` and never a `* 1.0` (`frd_temporal.hlsl`'s "the skip is a
/// BRANCH, never a computed 1.0"), which is what makes a disarmed session
/// bit-identical to the pre-feature renderer rather than identical by fp luck.
#[inline]
pub fn guard_scale(
    luma: f32,
    ring_mean: f32,
    ring_max: f32,
    nv: u32,
    exposure: f32,
    strength: f32,
    p: GuardParams,
) -> f32 {
    // `!(x > y)` throughout so a NaN in any input takes the off arm.
    // exposure <= 1.0 is an off arm and not merely an optimization: when the
    // aperture is stopping DOWN, exempting an outlier would make it brighter
    // relative to its surround — the exact inverse of the intent.
    if !(exposure > 1.0) || !(strength > 0.0) || nv < GUARD_MIN_RING || !(luma > 0.0) {
        return exposure;
    }
    let cap = (p.k_mean * ring_mean).max(p.k_max * ring_max);
    // A NaN in either reduction reaches here and takes this arm, so `cap` is a
    // real number below.
    if !(luma > cap) {
        return exposure;
    }
    // The ramp's saturated end is tested DIRECTLY rather than reached through
    // `luma/cap`, for a case that is easy to write off as impossible: an
    // exactly-black ring makes `cap` exactly 0, and that is a lit dot on
    // absolute black — the most extreme outlier the guard can ever see. An
    // `if !(cap > 0.0) { return exposure }` guard would hand precisely that
    // pixel the full boost. Testing `luma >= cap * ramp` exempts it, and also
    // means neither side has to agree about the semantics of x/0 (the division
    // is only evaluated where `cap > 0`).
    let t = if luma >= cap * p.ramp {
        1.0
    } else {
        ((luma / cap - 1.0) / (p.ramp - 1.0)).clamp(0.0, 1.0)
    };
    let w = t * t * (3.0 - 2.0 * t) * strength; // smoothstep, scaled
    if !(w > 0.0) {
        return exposure;
    }
    // Written as 1 + (E-1)(1-w), NOT lerp(E, 1, w): both factors are
    // non-negative here, so the result is >= 1.0 by CONSTRUCTION (no clamp to
    // audit) and a full exemption lands on EXACTLY 1.0.
    1.0 + (exposure - 1.0) * (1.0 - w)
}

/// Can the guard fire at all this frame? Checked BEFORE a luma plane is built,
/// so a disarmed frame allocates nothing and costs one comparison.
#[inline]
pub fn guard_armed(exposure: f32) -> bool {
    exposure > 1.0 && guard_strength_live() > 0.0
}

/// The guard plane for an already-averaged linear HDR image — the `resolve_hdr`
/// arms (OIDN / NPPD / XeSS-post, and every P-screenshot re-resolve).
pub fn guard_plane_hdr(hdr: &[f32], rw: usize, rh: usize, exposure: f32) -> Option<Vec<f32>> {
    if !guard_armed(exposure) {
        return None;
    }
    guard_plane(&luma_plane_hdr(hdr), rw, rh, exposure)
}

/// The guard plane for the accumulation buffer — the `resolve` arms.
pub fn guard_plane_accum(
    accum: &[std::sync::atomic::AtomicU32],
    samples: u32,
    rw: usize,
    rh: usize,
    exposure: f32,
) -> Option<Vec<f32>> {
    if !guard_armed(exposure) {
        return None;
    }
    guard_plane(&luma_plane_accum(accum, samples), rw, rh, exposure)
}

/// Linear luma of an interleaved RGB slice — the `resolve_hdr` arms' input.
fn luma_plane_hdr(hdr: &[f32]) -> Vec<f32> {
    hdr.par_chunks_exact(3).map(|c| luma(c[0], c[1], c[2])).collect()
}

/// Linear luma of the accumulation buffer (bit patterns holding SUMS — the
/// `render::resolve` input contract, hence the 1/samples).
fn luma_plane_accum(accum: &[std::sync::atomic::AtomicU32], samples: u32) -> Vec<f32> {
    let inv = 1.0 / samples.max(1) as f32;
    accum
        .par_chunks_exact(3)
        .map(|c| {
            let v = |a: &std::sync::atomic::AtomicU32| f32::from_bits(a.load(Ordering::Relaxed)) * inv;
            luma(v(&c[0]), v(&c[1]), v(&c[2]))
        })
        .collect()
}

/// A per-pixel exposure plane for the CPU-presented arms — one rayon pass over
/// a precomputed luma plane.
///
/// A PLANE rather than per-pixel ring taps at the six resolver call sites,
/// because the no-bloom `render::resolve` fast path reads `accum` atomically
/// inline: 8 taps x 3 relaxed loads per pixel there is real cost, and one
/// prepass also unifies the accum and hdr sources behind one code path.
///
/// Returns `None` when the guard cannot fire at all, so the callers' fast path
/// is the pre-feature one and no memory is committed.
pub fn guard_plane(luma: &[f32], rw: usize, rh: usize, exposure: f32) -> Option<Vec<f32>> {
    let strength = guard_strength_live();
    if !(exposure > 1.0) || !(strength > 0.0) || luma.len() < rw * rh || rw == 0 || rh == 0 {
        return None;
    }
    let p = guard_params();
    let r = p.radius;
    // One row per task, indexed — the `render::resolve` idiom, so ordering is a
    // property of the construction rather than of rayon's collect.
    let mut out = vec![0.0f32; rw * rh];
    out.par_chunks_mut(rw).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let (mut sum, mut mx, mut nv) = (0.0f32, 0.0f32, 0u32);
            for (dx, dy) in RING_OFFSETS {
                let (qx, qy) = (x as i32 + dx * r, y as i32 + dy * r);
                if qx < 0 || qy < 0 || qx >= rw as i32 || qy >= rh as i32 {
                    continue; // EXCLUDED, never replicated
                }
                let v = luma[qy as usize * rw + qx as usize];
                sum += v;
                mx = mx.max(v);
                nv += 1;
            }
            let mean = if nv > 0 { sum / nv as f32 } else { 0.0 };
            *o = guard_scale(luma[y * rw + x], mean, mx, nv, exposure, strength, p);
        }
    });
    if guard_trace_on() {
        guard_trace_dump(luma, &out, rw, rh, exposure, p);
    }
    Some(out)
}

/// The 8 ring directions, scaled by the radius at each use. Shared by
/// `guard_plane` and mirrored by tonemap.hlsl's unrolled loop.
const RING_OFFSETS: [(i32, i32); 8] =
    [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// `FR_AEXP_GUARD_TRACE=1`'s output — the histogram the K constants are meant
/// to be PICKED from.
///
/// It reports the distributions of `luma/ring_mean` and `luma/ring_max`, not of
/// `luma/cap`: the cap is a function of the very constants being chosen, so a
/// luma/cap histogram would be circular and would only ever tell you that the
/// current setting is the current setting. These two ratios are exactly what
/// `K_MEAN` and `K_MAX` are thresholds ON, so a percentile here reads directly
/// as "this K exempts that share of the frame".
///
/// Rate-limited to ~1/s. Reached in a GPU session through the P-screenshot /
/// QA-socket capture path, which is a CPU re-resolve — so the calibration
/// workflow is `FR_AEXP_GUARD_TRACE=1` plus a `frqa screenshot` at each pose.
fn guard_trace_dump(luma: &[f32], out: &[f32], rw: usize, rh: usize, exposure: f32, p: GuardParams) {
    use std::sync::Mutex;
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    {
        let mut g = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_some_and(|t| t.elapsed().as_secs_f32() < 1.0) {
            return;
        }
        *g = Some(std::time::Instant::now());
    }
    let r = p.radius;
    let (mut rm, mut rx) = (Vec::new(), Vec::new());
    // Subsample: a distribution's tail percentiles are what matter here and
    // every 3rd pixel resolves them fine, while a full pass at 4K is a real
    // stall on a probe that runs every second.
    for y in (0..rh).step_by(3) {
        for x in (0..rw).step_by(3) {
            let (mut sum, mut mx, mut nv) = (0.0f32, 0.0f32, 0u32);
            for (dx, dy) in RING_OFFSETS {
                let (qx, qy) = (x as i32 + dx * r, y as i32 + dy * r);
                if qx < 0 || qy < 0 || qx >= rw as i32 || qy >= rh as i32 {
                    continue;
                }
                let v = luma[qy as usize * rw + qx as usize];
                sum += v;
                mx = mx.max(v);
                nv += 1;
            }
            let l = luma[y * rw + x];
            if nv < GUARD_MIN_RING || !(l > 0.0) || !(sum > 0.0) {
                continue;
            }
            rm.push(l / (sum / nv as f32));
            if mx > 0.0 {
                rx.push(l / mx);
            }
        }
    }
    if rm.is_empty() {
        eprintln!("aexp-guard: no eligible pixels to profile");
        return;
    }
    let pct = |v: &mut Vec<f32>, q: f32| -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[(((v.len() - 1) as f32) * q) as usize]
    };
    // `out` already carries the verdict at the CURRENT constants, so the fired
    // share is measured rather than re-derived — it cannot drift from the plane
    // the frame is actually presented with.
    let fired = out.iter().filter(|&&e| e < exposure).count();
    let (m50, m90, m99, m999, mmax) =
        (pct(&mut rm, 0.5), pct(&mut rm, 0.9), pct(&mut rm, 0.99), pct(&mut rm, 0.999), pct(&mut rm, 1.0));
    let (x99, x999, xmax) = (pct(&mut rx, 0.99), pct(&mut rx, 0.999), pct(&mut rx, 1.0));
    eprintln!(
        "aexp-guard: e {exposure:.2} k_mean {} k_max {} r {} | luma/ring_mean p50 {m50:.2} \
         p90 {m90:.2} p99 {m99:.2} p99.9 {m999:.2} max {mmax:.2} | luma/ring_max p99 {x99:.2} \
         p99.9 {x999:.2} max {xmax:.2} | fired {fired} px ({:.3}% of frame)",
        p.k_mean,
        p.k_max,
        p.radius,
        100.0 * fired as f32 / (rw * rh) as f32
    );
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
/// both arms are pinned whichever way the session's lever sits, so a
/// `--no-auto-exposure --check` still proves the on arm and a default `--check`
/// still proves the off arm, and the run's own lever state survives this test).
pub fn self_test() -> Result<(), String> {
    struct Restore(bool, f32, bool, f32, Mode);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_enabled(self.0);
            set_bias(self.1);
            set_guard(self.2);
            set_guard_strength(self.3);
            set_mode(self.4);
        }
    }
    let _restore = Restore(enabled(), bias(), guard(), guard_strength(), mode());
    // Every arm below (1)-(14) is about the aperture's MAGNITUDE, which is
    // mode-independent — pin the default arm so they read the same whichever
    // way the session's lever sits. (15)/(16) drive the lever themselves.
    set_mode(Mode::Tonemap);

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

    // ---- the spike guard ---------------------------------------------------
    // Pinned with the lever save/restore above, so a `--no-autoexp-spike-guard`
    // session still proves the ON arm and vice versa (the wide-tiles pattern).
    let gp = GuardParams::default();
    let spike = |e: f32, s: f32| guard_scale(1.0, 0.01, 0.01, 8, e, s, gp);

    // (10) Every off arm returns the exposure BITWISE — the disarmed-stream
    // contract. A spike over a near-black ring is used throughout, so each
    // failure below means that arm, not a failure to detect.
    for (name, got, want) in [
        ("strength 0", spike(4.0, 0.0), 4.0f32),
        ("exposure == 1", spike(1.0, 1.0), 1.0),
        ("exposure < 1 (stopping down)", spike(0.5, 1.0), 0.5),
        ("nv < MIN_RING", guard_scale(1.0, 0.01, 0.01, GUARD_MIN_RING - 1, 4.0, 1.0, gp), 4.0),
        ("luma 0", guard_scale(0.0, 0.01, 0.01, 8, 4.0, 1.0, gp), 4.0),
        ("dark ring, dark pixel", guard_scale(0.001, 0.01, 0.01, 8, 4.0, 1.0, gp), 4.0),
        ("NaN luma", guard_scale(f32::NAN, 0.01, 0.01, 8, 4.0, 1.0, gp), 4.0),
        ("NaN ring", guard_scale(1.0, f32::NAN, f32::NAN, 8, 4.0, 1.0, gp), 4.0),
    ] {
        if got.to_bits() != want.to_bits() {
            return Err(format!("guard off arm '{name}': got {got}, want exactly {want}"));
        }
    }

    // (11) THE SAFETY INVARIANT, swept: the guard may only ever relax TOWARD
    // 1.0, never past it and never above the frame's own exposure. This is what
    // bounds a false positive at "looks like --no-auto-exposure".
    for &e in &[1.25f32, 2.0, 4.0, 16.0] {
        for i in 0..=64 {
            let l = 0.01 * 1.15f32.powi(i);
            for &s in &[0.25f32, 0.5, 1.0] {
                let g = guard_scale(l, 0.01, 0.01, 8, e, s, gp);
                // Upper bound carries one relative ulp: the form is
                // 1 + (E-1)(1-w), which reproduces E exactly on this range but
                // is not required to for every representable E.
                if !(g >= 1.0) || g > e * (1.0 + f32::EPSILON) {
                    return Err(format!("guard invariant: luma {l}, e {e}, s {s} -> {g} outside [1, {e}]"));
                }
            }
        }
    }

    // (12) The ramp: monotone non-increasing in luma (a brighter outlier is
    // exempted at least as much), exactly `exposure` at the low endpoint, and
    // exactly 1.0 once past the ramp at full strength.
    let (e, ring) = (4.0f32, 0.01f32);
    let cap = (gp.k_mean * ring).max(gp.k_max * ring);
    let mut prev = f32::INFINITY;
    for i in 0..=200 {
        let l = cap * (0.5 + i as f32 * 0.02);
        let g = guard_scale(l, ring, ring, 8, e, 1.0, gp);
        if g > prev + 1e-6 {
            return Err(format!("guard ramp not monotone at luma/cap {}: {g} > {prev}", l / cap));
        }
        prev = g;
    }
    if guard_scale(cap, ring, ring, 8, e, 1.0, gp).to_bits() != e.to_bits() {
        return Err("guard ramp: luma exactly at the cap must be bitwise unguarded".into());
    }
    let full = guard_scale(cap * gp.ramp * 4.0, ring, ring, 8, e, 1.0, gp);
    if full.to_bits() != 1.0f32.to_bits() {
        return Err(format!("guard ramp: a deep outlier at strength 1 must land on exactly 1.0, got {full}"));
    }

    // (13) THE K_MAX SHIELD, with teeth BOTH ways — the assertion that keeps
    // this from being satisfied by a guard that never fires or one that fires
    // everywhere. Same pixel, same mean; only whether it has a bright NEIGHBOUR
    // differs, which is exactly what separates a stochastic dot from a resolved
    // feature (a lamp, the sun's limb, a fireflies.rs glow splat).
    let hot = 1.0f32;
    let lone = guard_scale(hot, hot / 8.0, hot / 8.0, 8, e, 1.0, gp); // ring ~= its own mean
    let feature = guard_scale(hot, hot / 8.0, hot, 8, e, 1.0, gp); // one neighbour just as bright
    if lone >= e {
        return Err(format!("guard toothless: a lone spike over a dark ring was not exempted ({lone} vs {e})"));
    }
    if feature.to_bits() != e.to_bits() {
        return Err(format!(
            "K_MAX shield breached: a spike WITH a bright neighbour must be bitwise unguarded, got {feature}"
        ));
    }
    // (13b) THE BLACK-RING CASE, which the natural `cap > 0` guard silently
    // hands the full boost to: an exactly-zero ring makes the cap exactly 0, and
    // that pixel is a lit dot on absolute black — the most extreme outlier the
    // guard can see, so it must take the FULL exemption rather than the off arm.
    // Also pinned NaN-safe, since the two live one `!(x > y)` apart.
    let black = guard_scale(hot, 0.0, 0.0, 8, e, 1.0, gp);
    if black.to_bits() != 1.0f32.to_bits() {
        return Err(format!(
            "guard: a spike on an exactly-black ring must be fully exempted (exactly 1.0), got {black}"
        ));
    }
    let black_half = guard_scale(hot, 0.0, 0.0, 8, e, 0.5, gp);
    if !(black_half > 1.0 && black_half < e) {
        return Err(format!(
            "guard: the black-ring arm must still honour strength (got {black_half} at 0.5)"
        ));
    }

    // (14) `guard_plane`: the None fast path, determinism, an unguarded border,
    // and a planted outlier actually exempted on a real grid.
    set_guard(true);
    set_guard_strength(1.0);
    let (pw, ph) = (32usize, 24usize);
    let mut lp = vec![0.01f32; pw * ph];
    let spike_i = 12 * pw + 16;
    lp[spike_i] = 5.0;
    if guard_plane(&lp, pw, ph, 1.0).is_some() {
        return Err("guard_plane must return None at exposure 1.0 (no boost to withhold)".into());
    }
    set_guard(false);
    if guard_plane(&lp, pw, ph, 4.0).is_some() {
        return Err("guard_plane must return None when the guard lever is off".into());
    }
    set_guard(true);
    let plane = guard_plane(&lp, pw, ph, 4.0).ok_or("guard_plane returned None when armed")?;
    if plane.len() != pw * ph {
        return Err(format!("guard_plane length {} != {}", plane.len(), pw * ph));
    }
    if plane[spike_i] >= 4.0 {
        return Err(format!("guard_plane: the planted outlier was not exempted ({})", plane[spike_i]));
    }
    let flat_i = 4 * pw + 4;
    if plane[flat_i].to_bits() != 4.0f32.to_bits() {
        return Err(format!("guard_plane: a flat-field pixel must be bitwise unguarded, got {}", plane[flat_i]));
    }
    let again = guard_plane(&lp, pw, ph, 4.0).unwrap();
    if plane.iter().zip(&again).any(|(a, b)| a.to_bits() != b.to_bits()) {
        return Err("guard_plane is not deterministic".into());
    }
    // A corner has fewer than 8 in-range taps at any radius; with MIN_RING = 4
    // it may or may not be guarded, but it must never be NaN or out of band.
    for (i, &g) in plane.iter().enumerate() {
        if !g.is_finite() || !(1.0..=4.0 + f32::EPSILON * 4.0).contains(&g) {
            return Err(format!("guard_plane px {i}: {g} outside [1, 4]"));
        }
    }

    // (15) The MODE split. Exactly one destination is armed, and the split is
    // exact rather than approximate: the product of the two must reproduce the
    // total aperture BITWISE (they partition `ev_total` additively in stops,
    // and one side is always exactly 0 stops -> exactly 1.0).
    set_enabled(true);
    set_bias(0.0);
    for m in [Mode::Tonemap, Mode::Lights] {
        set_mode(m);
        if Mode::parse(m.as_str()) != Some(m) {
            return Err(format!("Mode vocabulary does not round-trip for {m:?}"));
        }
        for ev in [-2.0f32, -0.5, 0.0, 1.0, 2.0] {
            let (d, l, tot) = (display_exposure(ev), light_gain(ev), exposure(ev));
            let idle = if m == Mode::Lights { d } else { l };
            if idle.to_bits() != 1.0f32.to_bits() {
                return Err(format!("{m:?} ev {ev}: the idle destination must be exactly 1.0, got {idle}"));
            }
            if (d * l).to_bits() != tot.to_bits() {
                return Err(format!(
                    "{m:?} ev {ev}: display {d} * light {l} = {} != exposure {tot} bitwise",
                    d * l
                ));
            }
        }
    }
    // The bias reaches the lights arm too — a `--no-auto-exposure
    // --exposure-bias 2 --autoexp-mode lights` session is "make the scene
    // physically 4x brighter", and that is a feature, not an accident.
    set_mode(Mode::Lights);
    set_enabled(false);
    set_bias(2.0);
    if light_gain(9.0) != 4.0 || display_exposure(9.0).to_bits() != 1.0f32.to_bits() {
        return Err(format!(
            "lights arm must carry the bias with auto off: gain {}, display {}",
            light_gain(9.0),
            display_exposure(9.0)
        ));
    }
    set_bias(0.0);
    set_enabled(true);

    // (16) THE DE-GAIN IDENTITY — what keeps the lights arm's controller
    // open-loop. A frame rendered at gain g measures log2(L) + log2(g), so
    // de-gaining must recover a controller ask identical to the one today's
    // arm produces from the same scene. Identical BITWISE at the anchors, and
    // identical in the ask across a sweep — if this drifts, the two arms settle
    // at different EVs and the whole equivalence claim is void.
    let scene_log2 = -5.25f32; // some dark enclosure
    for ev_f in [0.0f32, 0.5, 1.0, 2.0, -1.5] {
        set_mode(Mode::Lights);
        let gain_ev = light_gain_ev(ev_f);
        let measured = scene_log2 + gain_ev; // what the meter would read
        let recovered = degain(measured, gain_ev);
        if (recovered - scene_log2).abs() > 1e-5 {
            return Err(format!(
                "degain: rendered at {gain_ev} stops, measured {measured}, recovered {recovered} != {scene_log2}"
            ));
        }
        // The controller must then take the same step it takes in the tonemap
        // arm, which never de-gains at all.
        set_mode(Mode::Tonemap);
        let plain = step(0.3, degain(scene_log2, light_gain_ev(ev_f)), 1.0 / 60.0);
        set_mode(Mode::Lights);
        let lit = step(0.3, recovered, 1.0 / 60.0);
        if (plain - lit).abs() > 1e-5 {
            return Err(format!("degain: controller diverged at {ev_f} stops ({plain} vs {lit})"));
        }
    }
    // The 0-stop case is BITWISE, never `- 0.0` — that is what leaves the
    // tonemap arm's controller textually and numerically untouched.
    for m in [Mode::Tonemap, Mode::Lights] {
        set_mode(m);
        for v in [-5.25f32, 0.0, 3.0, -0.0] {
            if degain(v, 0.0).to_bits() != v.to_bits() {
                return Err(format!("{m:?}: degain({v}, 0.0) must be bitwise inert"));
            }
        }
    }
    // TEETH: without the subtraction the loop is closed, and it must provably
    // land somewhere else — otherwise arm (16) would pass on a no-op.
    set_mode(Mode::Lights);
    let naive = step(0.3, scene_log2 + 2.0, 1.0 / 60.0);
    let correct = step(0.3, scene_log2, 1.0 / 60.0);
    if (naive - correct).abs() < 1e-3 {
        return Err("degain teeth: a NON-de-gained measurement must move the controller".into());
    }

    eprintln!("autoexp self-test: OK");
    Ok(())
}
