//! Fireflies — N small light-emitting particles drifting on curl-noise paths,
//! each a REAL point light (1/d² falloff + a real shadow ray in `shade()`'s
//! direct tier) plus a depth-tested Gaussian glow splat on the camera paths.
//! Default-on, but they exist only after dusk: brightness scales by
//! `Scene::night` (the stars' own fade scalar), and a `night == 0.0` session —
//! every flagless default-day run — snapshots `count = 0`, so the pre-firefly
//! renderer is reproduced STRUCTURALLY (guarded branches, no unconditional
//! `+0.0`), the `apply_tod`-unreachable precedent. `--no-fireflies` is the
//! kill lever, `--fireflies N` the count (clamped to `MAX_FIREFLIES` — the
//! GPU constant-buffer rows are sized to it).
//!
//! # The contracts everything here is built on
//!
//! - **Zero rng draws, zero wall clock.** Every function is a pure function of
//!   (inputs, `Fireflies`); poses are CLOSED-FORM in (index, time) — no
//!   integration state — with the clock OWNED by main.rs (`cloud_time`: the
//!   clouds' own "upscaler frames advance, plain accumulation only at frame 0"
//!   policy, `idx·CLOUD_SPIN_DT` under --spin, `CLOUD_CHECK_TIME` pinned in
//!   every --check*). Firefly shadow rays are HARD (one deterministic ray, no
//!   area sampling), so every same-seed / replay / VisCtl-burn bit-identity
//!   contract is untouched with no burn-accounting changes.
//! - **The one-sky gather exclusion (the stars rule).** Fireflies live in the
//!   direct-light tier and the display paths ONLY — never `sky::dome()`, the
//!   SH projection, the hemi gathers, or the GI reference estimators (`shade`'s
//!   recursion and the hemi tier pass `ff = None`). They light what their
//!   shadow rays reach and nothing else; like emissive materials, they do not
//!   light bounce surfaces.
//! - **Scale-relative.** Every length is a multiple of `Fireflies::diag`
//!   (`Scene::diag`), so one set of constants serves the diag-10 auto-fit,
//!   `--stress`, and `--tile`.
//! - **CPU/GPU parity by DATA.** Poses are baked once per frame on the CPU and
//!   uploaded as f32 rows in the frame constants — the HLSL twins re-derive
//!   nothing, so there is no cross-language transcendental drift to gate; the
//!   light/glow math is term-for-term mirrored (`shade.hlsli::ff_light`,
//!   `trace_common.hlsli::ff_glow`).
//!
//! Known-accepts (documented, deliberate): no motion vectors (drift reads as
//! shading change to the upscalers — the clouds accept); the glow is a
//! primary-camera-path feature (no glow in reflections or through glass — a
//! per-continuation N-scan buys a near-invisible payoff at glossy roughness);
//! fireflies cast no light through translucency; a converging still frame
//! freezes them mid-flight (the cloud-clock rule); possible slight RR ghosting
//! on the glow (no emissive guide — the emissive-map accept).

use glam::Vec3A;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Session enable — the `--no-fireflies` kill lever (the `clouds::set_enabled`
/// pattern: set ONCE at flag-parse time, snapshotted per frame).
static ENABLED: AtomicBool = AtomicBool::new(true);
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Session count — `--fireflies N`, clamped to `MAX_FIREFLIES` at parse time
/// (loudly, in main.rs — the CB rows are sized to the max).
static COUNT: AtomicU32 = AtomicU32::new(DEFAULT_COUNT);
pub fn set_count(n: u32) {
    COUNT.store(n.min(MAX_FIREFLIES as u32), Ordering::Relaxed);
}
pub fn count() -> u32 {
    COUNT.load(Ordering::Relaxed)
}

/// Hard cap — sizes the `FrameCb` firefly rows (raise `CB_STRIDE` in lockstep;
/// past ~64 the per-pixel linear scan wants the per-leaf-tile cull + SRV-table
/// follow-ons before the cap moves again — CPU cost is ~+0.45 ms/firefly,
/// dominated by the two per-pixel scans, and the CB rows stop being the
/// right transport at hundreds).
pub const MAX_FIREFLIES: usize = 64;
pub const DEFAULT_COUNT: u32 = 32;

/// Every FF length constant below is a multiple of `Fireflies::scale` — the
/// CONTENT diagonal (`Scene::content_min/max`), NOT `Scene::diag`: the
/// procedural/stress scenes' ±60 ground quad makes `diag` ~17× the content
/// scale, and fireflies placed off it hovered high over the whole field
/// instead of flitting among the models (the first look-pass screenshot).
///
/// Vertical placement spans the WHOLE content height (`cmin.y..cmax.y` — the
/// swarm fills the model's volume, not a floor band): floor clearance
/// `FF_Y_MIN_K` must exceed the total downward displacement bound
/// (`FF_DRIFT_K · CLOUD_CURL_YSCALE + FF_BOB_K` = 0.016) so every pose stays
/// strictly above the ground BY CONSTRUCTION, and the top is inset by the
/// same bound (upward) so poses stay inside the content box — `self_test`
/// sweeps both. `FF_Y_MAX_K` survives as the MINIMUM band top: a flat
/// content box (a plane-like scene) keeps at least the original
/// `[FF_Y_MIN_K, FF_Y_MAX_K]` band instead of collapsing to a sheet.
pub const FF_Y_MIN_K: f32 = 0.02;
pub const FF_Y_MAX_K: f32 = 0.06;
/// Curl-drift displacement amplitude. The curl field's soft |v| < 1
/// normalization (see `clouds::curl_offset`) makes this an EXACT
/// displacement bound — the placement inset and the ground floor lean on it.
pub const FF_DRIFT_K: f32 = 0.04;
/// Per-axis bob amplitude (three hashed sines — the wing-beat wobble the
/// low-frequency curl field is too smooth to provide).
pub const FF_BOB_K: f32 = 0.004;
/// How fast a firefly's curl lookup point travels, in scales/second — the
/// clouds' advect precedent (the field is static; the SAMPLE point moves).
pub const FF_WIND_K: f32 = 0.02;

/// Influence radius of the point light: beyond it a firefly costs the
/// shading loop exactly one rejection test (cost scales with local density,
/// not N). The windowed falloff below is exactly 0 there — no pop.
pub const FF_RADIUS_K: f32 = 0.06;
/// Point-light intensity: irradiance at distance d is
/// `FF_E_K · scale² · w / d²` — scale-invariant at proportional distances.
pub const FF_E_K: f32 = 2.0e-4;
/// Near-field clamp on the 1/d²: bounds the peak so a firefly grazing a
/// leaf cannot push f16-breaking radiance into the upscaler planes (the
/// sun-disc headroom lesson).
pub const FF_RMIN_K: f32 = 0.005;

/// Apparent source radius for the camera glow (~1 cm at courtyard scale) —
/// sets the splat's angular width at close range; the pixel footprint takes
/// over at distance (the `sky::stars` sizing rule).
pub const FF_SRC_K: f32 = 1.0e-3;
/// Glow radiance ceiling — the `STAR_L_MAX` argument: a tiny footprint must
/// not spike the RGBA16F planes; plenty left for bloom to pick up.
pub const FF_GLOW_L_MAX: f32 = 512.0;
/// Firefly bioluminescence — yellow-green, one shared tint (per-firefly
/// variation rides the brightness w, keeping the CB rows at one float4).
pub const FF_COLOR: Vec3A = Vec3A::new(0.65, 1.0, 0.25);

/// Per-frame firefly state — pure data; all math is free functions of it.
/// Deliberately NO `Default` (the `Clouds` discipline): every construction
/// site states its policy (live / check-pinned / spin / off), and the
/// compiler enumerates the sites.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fireflies {
    /// Live firefly count — 0 is the structural off state (day, or
    /// `--no-fireflies`): every consumer's loop body is unreachable.
    pub count: u32,
    /// The CONTENT diagonal (`|content_max − content_min|`) — every firefly
    /// length is a multiple of it (see the module constants' note on why
    /// this is not `Scene::diag`). Rides the CB's `ff_scale` lane.
    pub scale: f32,
    /// Baked poses: xyz = `p_i(time)` world position, w = brightness
    /// (tier · slow pulse · night). Rows past `count` are zero.
    pub pos: [[f32; 4]; MAX_FIREFLIES],
}

impl Fireflies {
    /// The structural off state (day sessions never even reach `bake`).
    pub fn off() -> Fireflies {
        Fireflies { count: 0, scale: 1.0, pos: [[0.0; 4]; MAX_FIREFLIES] }
    }
    /// Fully explicit constructor — the self-test's handle on both arms
    /// (enabled × night) without touching the session statics. `cmin`/`cmax`
    /// is the placement box (`Scene::content_min/max`).
    pub fn new(enabled: bool, n: u32, night: f32, cmin: Vec3A, cmax: Vec3A, time: f32) -> Fireflies {
        let scale = (cmax - cmin).length().max(1e-3);
        if !enabled || night <= 0.0 || n == 0 {
            return Fireflies { count: 0, scale, ..Fireflies::off() };
        }
        bake(n.min(MAX_FIREFLIES as u32), night, cmin, cmax, time)
    }
    /// The live interactive state: session statics + the scene's TOD fade +
    /// main.rs's clock (`cloud_time` — the shared animation clock).
    pub fn live(scene: &crate::scene::Scene, time: f32) -> Fireflies {
        Fireflies::new(enabled(), count(), scene.night, scene.content_min, scene.content_max, time)
    }
    /// The pinned headless state — the clouds' `CLOUD_CHECK_TIME`, so every
    /// CPU-reference-vs-GPU gate pair compares the same swarm (day checks
    /// still snapshot `count = 0` through `scene.night`).
    pub fn check(scene: &crate::scene::Scene) -> Fireflies {
        Fireflies::new(
            enabled(),
            count(),
            scene.night,
            scene.content_min,
            scene.content_max,
            crate::clouds::CLOUD_CHECK_TIME,
        )
    }
    /// --spin's clock: a pure function of the frame index.
    pub fn spin(scene: &crate::scene::Scene, idx: u32) -> Fireflies {
        Fireflies::new(
            enabled(),
            count(),
            scene.night,
            scene.content_min,
            scene.content_max,
            idx as f32 * crate::clouds::CLOUD_SPIN_DT,
        )
    }
    /// --cinematic's clock — the `clouds::Clouds::cine` twin, on the SHARED
    /// cloud clock (the swarm drifts through the same curl field). Read
    /// `scene.night` AFTER the frame's `apply_tod`, so the swarm fades in with
    /// the stars over a time-of-day sweep.
    pub fn cine(scene: &crate::scene::Scene, out_frame: u32, fps: u32) -> Fireflies {
        Fireflies::new(
            enabled(),
            count(),
            scene.night,
            scene.content_min,
            scene.content_max,
            out_frame as f32 / fps.max(1) as f32,
        )
    }
}

/// The curl field as a unit-bounded direction: `clouds::curl_offset` rescaled
/// by its own amplitude, so |result| < 1 exactly (soft normalization) and the
/// caller's amplitude constant is the whole displacement bound. Reuses the
/// cloud field VERBATIM (same octave ids, same wavelength — clouds.rs is not
/// modified) through a synthetic time-0 `Clouds`: the field is
/// time-independent and reads only `diag`.
#[inline]
fn curl_dir(p: Vec3A, scale: f32) -> Vec3A {
    crate::clouds::curl_offset(p, &crate::clouds::Clouds::new(true, scale, 0.0))
        * (1.0 / (crate::clouds::CLOUD_CURL_AMP_K * scale))
}

/// Closed-form pose + brightness for firefly `i` at clock `time` — the whole
/// motion model. Pure function of (i, night, placement box, time); hashes
/// are `sky::pcg_mix` chains (the star field's integer hash). The drift is a
/// TIME-SHIFTED lookup into the static curl field (the clouds `advect`
/// precedent): the lookup point travels at `FF_WIND_K·scale`/s from a hashed
/// start, so each firefly wanders a decorrelated path through the shared
/// field — organic, non-repeating, and exactly bounded.
fn pose(i: u32, night: f32, cmin: Vec3A, cmax: Vec3A, time: f32) -> [f32; 4] {
    use crate::sky::{hash01, pcg_mix};
    let scale = (cmax - cmin).length().max(1e-3);
    let h0 = pcg_mix(i.wrapping_mul(0x9E37_79B9) ^ 0xF1EF_11E5);
    let h1 = pcg_mix(h0);
    let h2 = pcg_mix(h1);
    let h3 = pcg_mix(h2);
    let h4 = pcg_mix(h3);
    let h5 = pcg_mix(h4);
    let h6 = pcg_mix(h5);
    let h7 = pcg_mix(h6);

    // Base: hashed uniform over the content box's xz footprint, inset by the
    // exact displacement bound so every pose stays inside it (a footprint
    // narrower than two insets collapses to its center line — never
    // negative); y uniform over the WHOLE content height — floor-cleared by
    // FF_Y_MIN_K (above the exact downward displacement bound) and top-inset
    // by the exact UPWARD bound (the curl's vertical leg is CLOUD_CURL_YSCALE
    // of the unit direction), so every pose stays inside the box by
    // construction. A flat content box keeps at least the original
    // [FF_Y_MIN_K, FF_Y_MAX_K] band (y_hi >= y_lo since FF_Y_MAX_K > FF_Y_MIN_K).
    let inset = (FF_DRIFT_K + FF_BOB_K) * scale;
    let vinset = (FF_DRIFT_K * crate::clouds::CLOUD_CURL_YSCALE + FF_BOB_K) * scale;
    let cx = 0.5 * (cmin + cmax);
    let hx = (0.5 * (cmax.x - cmin.x) - inset).max(0.0);
    let hz = (0.5 * (cmax.z - cmin.z) - inset).max(0.0);
    let y_lo = cmin.y + FF_Y_MIN_K * scale;
    let y_hi = (cmax.y - vinset).max(cmin.y + FF_Y_MAX_K * scale);
    let base = Vec3A::new(
        cx.x + (hash01(h0) - 0.5) * 2.0 * hx,
        y_lo + hash01(h1) * (y_hi - y_lo),
        cx.z + (hash01(h2) - 0.5) * 2.0 * hz,
    );

    // Drift: the static curl field sampled at a moving, per-firefly-offset
    // point. The offset decorrelates fireflies that share a base region; the
    // 0.37/0.61 direction is a literal (the clouds wind-constant discipline).
    let lookup = base
        + Vec3A::new(0.37, 0.0, 0.61) * (FF_WIND_K * scale * time)
        + Vec3A::new(hash01(h3), 0.0, hash01(h4)) * (7.3 * scale);
    let drift = curl_dir(lookup, scale) * (FF_DRIFT_K * scale);

    // Bob: three hashed sines (ω ∈ [0.4, 1.2] rad/s, φ ∈ [0, τ)).
    let tau = std::f32::consts::TAU;
    let w1 = 0.4 + 0.8 * hash01(h3);
    let w2 = 0.4 + 0.8 * hash01(h4);
    let w3 = 0.4 + 0.8 * hash01(h5);
    let bob = Vec3A::new(
        (w1 * time + tau * hash01(h5)).sin(),
        (w2 * time + tau * hash01(h6)).sin(),
        (w3 * time + tau * hash01(h7)).sin(),
    ) * (FF_BOB_K * scale);

    let p = base + drift + bob;

    // Brightness: a per-firefly tier (0.7..1.3), a slow waxing/waning pulse
    // (the firefly blink, squared for a soft-off dwell — f(time), never
    // f(frame): the night spp-stability gate must see zero inter-frame delta
    // from a static clock), and the dusk fade.
    let tier = 0.7 + 0.6 * hash01(pcg_mix(h7));
    let q = 0.5 + 0.5 * ((0.3 + 0.5 * hash01(h6)) * time + tau * hash01(h0)).sin();
    let pulse = 0.25 + 0.75 * q * q;
    [p.x, p.y, p.z, tier * pulse * night]
}

/// Bake all poses for one frame — the only caller of `pose`.
fn bake(n: u32, night: f32, cmin: Vec3A, cmax: Vec3A, time: f32) -> Fireflies {
    let scale = (cmax - cmin).length().max(1e-3);
    let mut ff = Fireflies { count: n, scale, pos: [[0.0; 4]; MAX_FIREFLIES] };
    for i in 0..n {
        ff.pos[i as usize] = pose(i, night, cmin, cmax, time);
    }
    ff
}

/// The smooth influence window: `(1 − d²/r²)²`, exactly 0 at the radius and
/// C¹ there — no pop as a firefly crosses a pixel's influence boundary.
/// Mirrored term-for-term in `shade.hlsli::ff_light`.
#[inline(always)]
pub fn window(d2: f32, r2: f32) -> f32 {
    let x = 1.0 - d2 / r2;
    x * x
}

/// Irradiance (over π, the renderer's Lambert convention — the sun's
/// `e_over_pi` encoding) arriving at distance² `d2` from firefly `i`:
/// windowed 1/d² with the near-field clamp. 0 beyond the radius.
#[inline(always)]
pub fn irradiance(ff: &Fireflies, i: usize, d2: f32) -> f32 {
    let r2 = FF_RADIUS_K * ff.scale * (FF_RADIUS_K * ff.scale);
    if d2 >= r2 {
        return 0.0;
    }
    let rmin2 = FF_RMIN_K * ff.scale * (FF_RMIN_K * ff.scale);
    FF_E_K * ff.scale * ff.scale * ff.pos[i][3] / d2.max(rmin2) * window(d2, r2)
}

/// The camera glow: every firefly nearer than `t_max` along the ray splats an
/// energy-conserving angular Gaussian (the `sky::stars` construction at
/// finite distance — irradiance `E_K·diag²·w/s²` spread over the splat's
/// solid angle, so brightness neither crawls at 1 spp nor changes with render
/// resolution). Callers must GUARD on `ff.count > 0` before adding the
/// result (`-0.0 + 0.0 = +0.0` — the emissive bit-identity discipline).
/// `half_angle` is the ray's angular footprint (`pixel_cone/2` on the
/// primary paths). A DISPLAY-path function — never called from any gather.
pub fn glow(ff: &Fireflies, o: Vec3A, d: Vec3A, t_max: f32, half_angle: f32) -> Vec3A {
    let mut acc = Vec3A::ZERO;
    let e = FF_E_K * ff.scale * ff.scale;
    let src_r = FF_SRC_K * ff.scale;
    for i in 0..ff.count as usize {
        let to = Vec3A::from_slice(&ff.pos[i]) - o;
        let s = to.dot(d);
        // Behind the camera, or behind the primary hit: the depth test that
        // makes the glow a scene object rather than a screen decal.
        if s <= 0.0 || s >= t_max {
            continue;
        }
        // Small-angle offset between the ray and the firefly direction:
        // θ² ≈ perp²/s² (exact enough at splat scales — the stars' cos
        // shortcut, in its distance form).
        let perp2 = (to - d * s).length_squared();
        let sigma = (half_angle * 0.5).max(src_r / s);
        let theta2 = perp2 / (s * s);
        // Reject BEFORE the exp: exp(-a) < 1e-4 ⇔ a > 9.2103, and almost
        // every (pixel, firefly) pair is thousands of sigmas off-axis — the
        // unconditional exp was the measured cost of the whole feature
        // (+34 ms/frame at N=16 on the night spin; the rays were ~free).
        // 9.22 > -ln(1e-4), so every skipped pair would have failed the
        // post-test below anyway — survivors are bit-identical.
        let a = theta2 / (2.0 * sigma * sigma);
        if a > 9.22 {
            continue;
        }
        let g = (-a).exp();
        if g < 1e-4 {
            continue;
        }
        let l = (e * ff.pos[i][3] / (s * s * std::f32::consts::TAU * sigma * sigma))
            .min(FF_GLOW_L_MAX);
        acc += FF_COLOR * (l * g);
    }
    acc
}

/// Closed-form gates on the pieces every consumer leans on. Pure, DLL-free,
/// deterministic — run by `--check` next to `clouds::self_test`.
pub fn self_test() -> Result<(), String> {
    // A synthetic content box (the default scene's shape: models over a
    // modest xz footprint, ground at y = 0).
    let cmin = Vec3A::new(-12.0, 0.0, -12.0);
    let cmax = Vec3A::new(12.0, 4.0, 12.0);
    let scale = (cmax - cmin).length();

    // 1. Structural off: disabled, day, and zero-count all snapshot count 0 —
    //    the arm every flagless session takes (bit-identity by construction).
    for (en, night, n) in [(false, 1.0, 8u32), (true, 0.0, 8), (true, 1.0, 0)] {
        let ff = Fireflies::new(en, n, night, cmin, cmax, 3.0);
        if ff.count != 0 {
            return Err(format!("off arm ({en}, {night}, {n}) has count {}", ff.count));
        }
    }

    // 2. Determinism: two bakes at one clock are bit-identical (the replay /
    //    same-seed contracts consume poses as pure data).
    let a = Fireflies::new(true, 16, 1.0, cmin, cmax, 7.5);
    let b = Fireflies::new(true, 16, 1.0, cmin, cmax, 7.5);
    if a != b {
        return Err("bake is not deterministic".into());
    }
    if a.count != 16 {
        return Err(format!("expected 16 fireflies, got {}", a.count));
    }

    // 3. Bounds by construction: sweep i × t (the check clock included) —
    //    every pose inside the content footprint, strictly above the content
    //    floor, under the content top (or the flat-scene minimum band + its
    //    displacement bound, whichever is higher), w in a sane band. This is
    //    the no-clamp proof (and the FF_Y_MIN_K > vertical-displacement-bound
    //    pin). The must-fire: the full-height spread is LIVE — some pose in
    //    the sweep reaches the upper half of the content height (uniform
    //    placement over 64 × 200 samples makes a miss astronomically
    //    improbable; a regression back to the old floor band fails it
    //    structurally, since the band top sits well under half height here).
    let y_cap = cmax
        .y
        .max(cmin.y + (FF_Y_MAX_K + FF_DRIFT_K * crate::clouds::CLOUD_CURL_YSCALE + FF_BOB_K) * scale);
    let mut y_seen = f32::NEG_INFINITY;
    for i in 0..MAX_FIREFLIES as u32 {
        for step in 0..200 {
            let t = step as f32 * 3.7;
            let p = pose(i, 1.0, cmin, cmax, t);
            if p[1] <= cmin.y {
                return Err(format!("firefly {i} at t {t} below ground: y {}", p[1]));
            }
            if p[0] < cmin.x
                || p[0] > cmax.x
                || p[2] < cmin.z
                || p[2] > cmax.z
                || p[1] > y_cap + 1e-4
            {
                return Err(format!("firefly {i} at t {t} out of box: {:?}", &p[..3]));
            }
            if !(p[3] > 0.0 && p[3] <= 1.3) {
                return Err(format!("firefly {i} at t {t} brightness {} out of band", p[3]));
            }
            y_seen = y_seen.max(p[1]);
        }
    }
    if y_seen < 0.5 * (cmin.y + cmax.y) {
        return Err(format!(
            "full-height spread not live: max pose y {y_seen} under half height"
        ));
    }

    // 4. Falloff: exactly 0 at the radius (window's zero is exact in fp:
    //    d2 == r2 ⇒ x == 0), monotone inside, near-field peak bounded well
    //    under the f16 ceiling.
    let r = FF_RADIUS_K * scale;
    if window(r * r, r * r) != 0.0 {
        return Err("window not exactly 0 at the radius".into());
    }
    if irradiance(&a, 0, r * r) != 0.0 || irradiance(&a, 0, r * r * 1.5) != 0.0 {
        return Err("irradiance not 0 at/past the radius".into());
    }
    let mut prev = f32::INFINITY;
    for k in 1..=64 {
        let d2 = (k as f32 / 64.0) * r * r;
        let e = irradiance(&a, 0, d2);
        if e > prev {
            return Err(format!("irradiance not monotone at d2 {d2}"));
        }
        prev = e;
    }
    let peak = FF_E_K / (FF_RMIN_K * FF_RMIN_K);
    if !(peak.is_finite() && peak < 1000.0) {
        return Err(format!("near-field peak {peak} out of the f16-safe band"));
    }

    // 5. Glow: depth-tested away is EXACTLY zero (behind-hit and behind-
    //    camera); visible is finite, positive, and capped; the splat's
    //    integrated energy is footprint-invariant (the stars' conservation
    //    argument): peak × σ² is constant across half_angle scales while the
    //    cap is not in play.
    let ff1 = {
        let mut f = Fireflies::off();
        f.count = 1;
        f.scale = 10.0;
        f.pos[0] = [0.0, 1.0, -5.0, 1.0];
        f
    };
    let o = Vec3A::new(0.0, 1.0, 0.0);
    let d = Vec3A::new(0.0, 0.0, -1.0);
    if glow(&ff1, o, d, 4.0, 1e-3) != Vec3A::ZERO {
        return Err("glow not zero when the hit is nearer".into());
    }
    if glow(&ff1, o, -d, f32::INFINITY, 1e-3) != Vec3A::ZERO {
        return Err("glow not zero behind the camera".into());
    }
    let g1 = glow(&ff1, o, d, f32::INFINITY, 1e-3);
    if !(g1.max_element() > 0.0 && g1.is_finite()) {
        return Err(format!("on-axis glow {g1:?} not positive finite"));
    }
    // Footprint invariance where the PIXEL footprint dominates (a near
    // firefly at s = 0.5, so src_r/s = 0.02 < both half-angle sigmas): the
    // on-axis peak is l = E/(s²·τ·σ²) with g = 1, so peak × σ² must be
    // constant across half_angle — the energy the splat spreads is the same.
    let ffn = {
        let mut f = ff1;
        f.pos[0] = [0.0, 1.0, -0.5, 1.0];
        f
    };
    let (ha1, ha2) = (0.05_f32, 0.1_f32);
    let l1 = glow(&ffn, o, d, f32::INFINITY, ha1).y / FF_COLOR.y;
    let l2 = glow(&ffn, o, d, f32::INFINITY, ha2).y / FF_COLOR.y;
    let (e1, e2) = (l1 * (ha1 * 0.5) * (ha1 * 0.5), l2 * (ha2 * 0.5) * (ha2 * 0.5));
    if (e1 - e2).abs() > 1e-3 * e1.max(e2) {
        return Err(format!("glow energy not footprint-invariant: {e1} vs {e2}"));
    }
    if glow(&ff1, o, d, f32::INFINITY, 0.0).y / FF_COLOR.y > FF_GLOW_L_MAX {
        return Err("glow cap not applied at a degenerate footprint".into());
    }

    // 6. The parse-time levers round-trip (through the explicit constructor —
    //    the statics belong to the session and are not mutated here).
    if Fireflies::new(true, 999, 1.0, cmin, cmax, 0.0).count != MAX_FIREFLIES as u32 {
        return Err("count not clamped to MAX_FIREFLIES".into());
    }

    eprintln!("fireflies self-test: OK");
    Ok(())
}
