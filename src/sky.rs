//! The scene's ONE light: a sky sphere at infinity, of which the sun is a
//! bright patch.
//!
//! It is stored in two representations, split by FREQUENCY, because the two
//! bands need different sampling strategies:
//!
//! | band                          | representation                | sampled how                  |
//! |-------------------------------|-------------------------------|------------------------------|
//! | scattering dome (smooth)      | `dome()`, + order-2 SH (sh.rs)| analytic irradiance, NO rays |
//! | sun disc (sharp)              | `Sun` + `disc()`              | cone-sampled, shadow-rayed   |
//! | star field (sharp, ~5k of it) | `stars()` / `star_glow()`     | see the star row below       |
//!
//! This is not a compromise, it is forced. SH cannot be shadow-rayed, and a
//! ~2° sun is ~1e-4 of the hemisphere — gathering it by cosine sampling is pure
//! noise. Conversely the dome is genuinely low-frequency (multiple scattering
//! is what blurs it), so order-2 SH is near-lossless for its irradiance.
//!
//! # THE CENTRAL INVARIANT
//!
//! **The sun disc is delivered exactly once per light path.** A ray sees the
//! disc only if no light-sampling strategy already covers the sun along that
//! path:
//!
//! | path                         | sees            | why                                            |
//! |------------------------------|-----------------|------------------------------------------------|
//! | primary / camera miss        | `radiance()`    | backdrop — nothing else delivers it            |
//! | glass / transmission miss    | `radiance()`    | near-delta path, no light-sampling partner     |
//! | specular reflection miss     | MIS-weighted    | `direct_s` also delivers it — see `mis_weight` |
//! | hemi cells, GI leaf, SH proj | `gather()` ONLY | `direct_d` already delivers the sun's diffuse  |
//!
//! Break the last row and you get (a) a double count of light the direct loop
//! already added with its own shadow ray, and (b) a ~1e3-magnitude firefly into
//! `hemi`'s 2^18 fixed-point accumulator, which would saturate outright.
//!
//! **THE STAR ROW.** The same rule, one band lower and with the polarity
//! reversed: the star field is ALSO delivered exactly once per path, but it has
//! no light-sampling strategy at all (nothing importance-samples a star), so
//! instead of being excluded from gathers it is delivered to them in a
//! different REPRESENTATION — points to the eye (`stars()`, inside
//! `radiance()`), the field's smooth mean to the gathers (`star_glow()`, inside
//! `gather()`), carrying identical total energy either way (`STAR_FLUX`, gated
//! by enumeration in `self_test`). So `gather()` — not `dome()` — is what every
//! gather site calls, and starlight is a real, moon-independent ambient floor
//! at night. Adding `star_glow` to a `radiance()` path, or `stars` to a
//! `gather()` one, is the double count this split exists to prevent.
//!
//! **The cloud layer (src/clouds.rs, default-on, `--no-clouds`) extends the
//! table without changing it**: every `radiance()` row sees the backdrop
//! through the layer's transmittance along its own ray plus the layer's
//! scatter; the direct loop's sun (both strategies' partner) is attenuated by
//! `clouds::sun_transmittance` once per `shade()`; and the `dome()` row stays
//! CLOUD-FREE — a drifting occluder cannot live in a load-time SH projection,
//! and the gather paths must keep integrating exactly the function the SH and
//! the GI references were built from. The known-accepts live in clouds.rs's
//! header.

use glam::Vec3A;

/// The sun's angular RADIUS. The real sun is 0.27°; this is deliberately wider
/// — it keeps a little penumbra in the shadows (the old 4x4 rect subtended ~19°,
/// so shadows here have always been soft) and it drops the disc radiance ~50x,
/// which keeps reflection fireflies and the f16 upscaler wires comfortable.
pub const SUN_ANGULAR_RADIUS: f32 = 2.0 * std::f32::consts::PI / 180.0;

/// Sun irradiance / π — EXACTLY the quantity the old code spelled
/// `light.color / dist²`, i.e. `(1, 0.95, 0.85) * 150 / |(6,10,4)|²`. Authoring
/// the sun by this number is what makes removing the 1/d² falloff a non-event:
/// the direct term at the scene origin is unchanged by construction, so every
/// existing scene, gate, and screenshot keeps its exposure.
pub const SUN_E_OVER_PI: Vec3A = Vec3A::new(0.9868, 0.9375, 0.8388);

/// The moon's angular RADIUS — like the sun's, deliberately wider than the real
/// 0.26° so the disc reads at interactive resolutions and its radiance stays
/// far under the f16 upscaler wires (radiance is derived as e/Ω, so a narrower
/// moon is a brighter disc).
pub const MOON_ANGULAR_RADIUS: f32 = 0.5 * std::f32::consts::PI / 180.0;

/// Moonlight irradiance / π — ARTISTIC, not physical. A real full moon is
/// ~2e-6 of the sun (invisible at any exposure this renderer presents); ~1% of
/// the sun with a blue-shifted tint is the filmic "moonlight" convention and
/// makes moonlit shadows readable. The one moonlight brightness knob, the
/// `DOME_SCALE` pattern.
pub const MOON_E_OVER_PI: Vec3A = Vec3A::new(0.0085, 0.0095, 0.0115);

/// The night dome's floor, as a fraction of the day dome — the moonlit sky is
/// the same Rayleigh dome (around the moon), this much dimmer. Applied through
/// `scene.sky_scale`; only `scene::apply_tod` ever consumes it.
pub const MOON_DOME_FRAC: f32 = 0.01;

/// THE dome brightness knob — the single number that sets both the sky you look
/// at and the ambient it casts. Tuned so the SH irradiance lands at ~0.17
/// luminance, which is simultaneously:
///   - what the old flat AMBIENT constant was, (0.14, 0.17, 0.23), and
///   - what physics wants: a clear sky delivers ~15-25% of the sun's irradiance.
///
/// Those two agreeing is the whole point. The old sky GRADIENT was authored to
/// look good as a BACKDROP (horizon radiance 0.72-0.95) and was far too bright
/// to also serve as a LIGHT — so the renderer used a separate, 2.7x darker
/// constant for the ambient, which is exactly why `fb.gi` frames (which
/// integrate the real sky) came out brighter than sampled ones. One number now
/// feeds both, and `sky::self_test` gates the resulting irradiance into a sane
/// band so it cannot silently drift apart again.
const DOME_SCALE: f32 = 14.8;

/// Mie asymmetry — the forward-scattering aureole around the sun.
const MIE_G: f32 = 0.76;

/// Rayleigh ZENITH OPTICAL DEPTH at (680, 550, 440) nm — real magnitudes, so
/// `exp(-τ)` behaves. The 1/λ⁴ RATIO is why the sky is blue overhead and why the
/// horizon (a long path, so the blue scatters out of it) goes pale.
const BETA_R: Vec3A = Vec3A::new(0.038, 0.096, 0.244);
/// Mie is wavelength-flat — haze is white/grey, which is what desaturates the
/// horizon and makes the aureole around the sun a warm white rather than a
/// colored lobe.
const BETA_M: Vec3A = Vec3A::new(0.021, 0.021, 0.021);

/// Fraction of dome light the ground bounces back. The SH projection integrates
/// the FULL sphere (the Ramamoorthi kernel IS the clamped-cosine convolution —
/// feeding it a hemisphere-only projection would double-clamp), so the lower
/// hemisphere must be defined. A hard horizon step would ring under order-2
/// truncation, hence the smooth blend in `dome`.
const GROUND_ALBEDO: f32 = 0.28;

/// The sun: a disc at infinity. No position, no 1/d² falloff, never in the BVH.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sun {
    /// Unit vector TOWARD the sun.
    pub dir: Vec3A,
    /// Cosine of the angular radius — the disc test.
    pub cos_radius: f32,
    /// Irradiance / π. The direct loop multiplies this by N·L; Lambert's 1/π
    /// lives here, per the renderer's convention.
    pub e_over_pi: Vec3A,
    /// Radiance OF THE DISC — what an escaping ray sees. Derived once from
    /// `e_over_pi` and the solid angle, and cached, so CPU and GPU can never
    /// re-derive it differently.
    pub radiance: Vec3A,
}

impl Sun {
    /// The sun's solid angle (steradians).
    pub fn solid_angle(&self) -> f32 {
        std::f32::consts::TAU * (1.0 - self.cos_radius)
    }

    /// Build from a direction. Radiance follows from irradiance and the cone:
    /// `E = L · Ω` for a small disc, and `e_over_pi = E/π`, so `L = e_over_pi·π/Ω`.
    pub fn new(dir: Vec3A) -> Sun {
        let cos_radius = SUN_ANGULAR_RADIUS.cos();
        let omega = std::f32::consts::TAU * (1.0 - cos_radius);
        Sun {
            dir: dir.normalize_or_zero(),
            cos_radius,
            e_over_pi: SUN_E_OVER_PI,
            radiance: SUN_E_OVER_PI * (std::f32::consts::PI / omega),
        }
    }

    /// Uniform-in-cone sample from two uniforms — the EXACT replacement for the
    /// old `center + u·su + v·sv` rect sample, consuming the SAME two rng draws
    /// in the same order. That is what keeps every same-seed / replay /
    /// `VisCtl::Apply`-burn bit-identity contract intact.
    ///
    /// A pure function of (r1, r2), so `VisCtl` can keep storing the raw
    /// uniforms and reproduce the direction exactly on replay.
    #[inline(always)]
    pub fn sample_dir(&self, r1: f32, r2: f32) -> Vec3A {
        let cos_t = 1.0 - r1 * (1.0 - self.cos_radius);
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        let phi = std::f32::consts::TAU * r2;
        let (t1, t2) = crate::shade::onb(self.dir);
        t1 * (phi.cos() * sin_t) + t2 * (phi.sin() * sin_t) + self.dir * cos_t
    }

    /// `Sun::new`, with `e_over_pi` AND `radiance` scaled by the same
    /// per-channel fade (`sun_fade`) — radiance stays derived from e_over_pi,
    /// so CPU and GPU keep one source. `Sun::new` itself is untouched: this is
    /// only ever reached through `scene::apply_tod`, never on the default path
    /// (an untouched session's Sun must stay bit-identical).
    pub fn with_fade(dir: Vec3A, fade: Vec3A) -> Sun {
        let s = Sun::new(dir);
        Sun { e_over_pi: s.e_over_pi * fade, radiance: s.radiance * fade, ..s }
    }
}

/// Hermite smoothstep of `y` over `[lo, hi]` — exactly 0 at/below `lo`,
/// exactly 1 at/above `hi`. The exact endpoints are load-bearing: they are
/// what lets `sun_fade` reach a true zero (the moon handoff) and a true one
/// (daytime bit-identity).
fn rise(y: f32, lo: f32, hi: f32) -> f32 {
    let t = ((y - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The dome's hoisted sun-path transmittance at elevation `y` — the SAME
/// expression `dome()` always computed inline (`exp(-(β_R+β_M)·m)`, air mass
/// `m = 1/(y.max(0)+0.15)`), factored so the cloud layer (`clouds::along`'s
/// sunlight tint) and the dome can never disagree about how the low sun
/// reddens. Bit-identical to the old inline form.
#[inline(always)]
pub(crate) fn t_sun_path(y: f32) -> Vec3A {
    let bt = BETA_R + BETA_M;
    let m_sun = 1.0 / (y.max(0.0) + 0.15);
    (-(bt * m_sun)).exp()
}

/// Per-channel direct-sun irradiance fade for a sun at elevation `y`, relative
/// to the reference elevation `y_ref` (the default sun's — so an untouched
/// scene is exactly factor 1, though the real guard is that nothing calls this
/// on the default path).
///
/// Two factors:
/// - The dome's own sun-path transmittance, `exp(-(β_R+β_M)·m)` at the SAME
///   air-mass model `dome()` uses (`m = 1/(y.max(0)+0.15)`), as a RATIO
///   against `y_ref` and clamped ≤ 1 per channel. This is what dims AND
///   reddens the low sun — blue extinguishes first (β_R is blue-heavy), the
///   same physics that reddens the dome. It bottoms out around
///   (0.72, 0.52, 0.23) at the horizon: a sunset tint, deliberately not night.
/// - A horizon smoothstep over y ∈ [-0.05, 0] carrying the fade to a true
///   zero once the disc has fully set (sin 2° ≈ 0.035, so -0.05 is just past
///   the lower limb). For y ≥ 0 it is exactly 1 and perturbs nothing.
pub fn sun_fade(y: f32, y_ref: f32) -> Vec3A {
    let bt = BETA_R + BETA_M;
    let dm = 1.0 / (y.max(0.0) + 0.15) - 1.0 / (y_ref + 0.15);
    let t = (-(bt * dm)).exp().min(Vec3A::ONE);
    t * rise(y, -0.05, 0.0)
}

/// The full moon as the one light at infinity — a `Sun` with the moon's cone
/// and radiometrics, so the direct loop, MIS, cone sampling, the disc render,
/// and the dome's `t_sun` tint all consume it unchanged. `scene::apply_tod`
/// installs it (antipodal to the sun's arc position — a full moon IS opposite
/// the sun, so it is above the horizon exactly when the sun is not) once the
/// sun's fade reaches zero.
///
/// Its own rise ramp over y ∈ [0, 0.10] softens the handoff: at the swap the
/// moon sits ~0.05 above the horizon, mid-ramp and near-zero intensity, so
/// the switch of active light is invisible.
pub fn moon(dir: Vec3A) -> Sun {
    let cos_radius = MOON_ANGULAR_RADIUS.cos();
    let omega = std::f32::consts::TAU * (1.0 - cos_radius);
    let e = MOON_E_OVER_PI * rise(dir.y, 0.0, 0.10);
    Sun {
        dir: dir.normalize_or_zero(),
        cos_radius,
        e_over_pi: e,
        radiance: e * (std::f32::consts::PI / omega),
    }
}

/// The smooth scattering dome — NO sun disc. Single-scattering Rayleigh + Mie.
///
/// This is what every *gather* path integrates (hemi cells, GI leaf misses, the
/// SH projection) and what `radiance()` adds the disc to. Defined over the FULL
/// sphere: below the horizon it is the ground bouncing the dome back, blended
/// smoothly so order-2 SH doesn't ring on a step.
///
/// `scale` is the time-of-day brightness (`Scene::sky_scale`): exactly 1.0 in
/// every untouched session (`x * 1.0` is bit-preserving, so daytime output is
/// unchanged), falling through dusk to the `MOON_DOME_FRAC` moonlight floor.
/// At night `sun` is the MOON (see `moon()`), so the same Rayleigh model
/// renders the moonlit sky — a dim blue day sky around the moon, which is what
/// a real moonlit sky is.
pub fn dome(d: Vec3A, sun: Vec3A, scale: f32) -> Vec3A {
    // The sun's own path down to the scattering volume depends ONLY on its
    // elevation, so its transmittance is the same for every direction in the
    // frame — hoisted out of `scatter`, which would otherwise recompute this
    // Vec3A `exp` on each of its calls. `dome` is the hemisphere integrator's
    // inner loop (every proven-empty cell centroid, every GI leaf-ray miss),
    // so that is not a rounding-off saving.
    let bt = BETA_R + BETA_M;
    // ...and it is what reddens the sky when the sun is low: the blue is
    // scattered out of the sunlight before it ever reaches the air we look at.
    let t_sun = t_sun_path(sun.y);

    // The in-scatter is kept LINEAR in β, and extinction is applied as separate
    // view/sun transmittances. This matters: an earlier version divided the
    // in-scatter by the combined per-channel β_t, and since β_t is blue-heavy
    // that INVERTED the grey Mie aureole into a red one — bright enough to
    // dominate the irradiance integral and turn the whole ambient warm. Keeping
    // the scatter linear in β is what keeps blue light blue.
    let scatter = |dir: Vec3A, mu: f32| -> Vec3A {
        // Rayleigh phase 3/(16π)·(1+cos²θ) — nearly isotropic, which is why the
        // WHOLE sky glows blue and not just the region around the sun.
        let ph_r = 0.059_683_1 * (1.0 + mu * mu);
        // Henyey-Greenstein — strongly forward. This IS the aureole.
        let g2 = MIE_G * MIE_G;
        let den = (1.0 + g2 - 2.0 * MIE_G * mu).max(1e-4);
        let ph_m = 0.079_577_5 * (1.0 - g2) / (den * den.sqrt());

        // Relative air mass along the view ray: how much atmosphere it crosses.
        // Grows toward the horizon (a long, blue-depleted path); bounded so the
        // horizon stays finite.
        let m_view = 1.0 / (dir.y.abs() + 0.15);
        let t_view = (-(bt * m_view)).exp();

        (BETA_R * ph_r + BETA_M * ph_m) * m_view * t_view * t_sun * DOME_SCALE
    };

    // Below the horizon: the ground, bouncing the dome above it back. The SH
    // projection integrates the FULL sphere, so this must be defined — and it is
    // BLENDED over a band rather than stepped, because a discontinuity rings
    // under order-2 truncation (Gibbs) and would put negative lobes in the
    // ambient.
    //
    // The band is narrow (±0.05 in d.y), so almost every direction ever asked
    // for is wholly on one side of it. Both ends return early rather than
    // evaluating BOTH scatters and then lerping one of them away with a weight
    // of 0 or 1 — the sky term above the band and the ground term below it are
    // each a full `scatter`, and the discarded one was pure waste.
    let t = ((d.y + 0.05) / 0.10).clamp(0.0, 1.0);
    if t >= 1.0 {
        return scatter(d, d.dot(sun).clamp(-1.0, 1.0)) * scale;
    }
    let dm = Vec3A::new(d.x, d.y.abs(), d.z);
    let ground = scatter(dm, dm.dot(sun).clamp(-1.0, 1.0)) * GROUND_ALBEDO;
    if t <= 0.0 {
        return ground * scale;
    }
    let sky = scatter(d, d.dot(sun).clamp(-1.0, 1.0));
    ground.lerp(sky, t) * scale
}

/// The sun disc, ANTIALIASED against the ray's angular footprint.
///
/// The sun's limb really is hard — a ~650x radiance discontinuity — so the disc
/// is a step function of direction, not a gradient. But a *ray* is not a
/// direction: it carries an angular footprint (`half_angle`, from the ray cone
/// we already track for texture LOD), and a pixel that half-covers the sun
/// should receive half its radiance. Without that, the edge is a binary
/// per-ray test: jagged when still, crawling under motion at 1 spp.
///
/// `cov` is the fraction of the footprint inside the disc, box-filtered — a
/// linear ramp across the 2·half_angle band straddling the limb. Pure geometry,
/// ZERO rng draws, so every same-seed/replay contract is untouched.
/// `half_angle = 0` reproduces the hard step exactly.
#[inline(always)]
pub fn disc(d: Vec3A, sun: &Sun, half_angle: f32) -> Vec3A {
    let c = d.dot(sun.dir).clamp(-1.0, 1.0);
    if half_angle <= 1e-7 {
        return if c >= sun.cos_radius { sun.radiance } else { Vec3A::ZERO };
    }
    // Angles, not cosines: the band is a constant angular width, and near the
    // limb cos is locally linear in θ anyway — but the ramp must not shear with
    // the sun's elevation, so do it honestly.
    //
    // The radius comes from `sun.cos_radius`, NOT from SUN_ANGULAR_RADIUS: the
    // hard-step arm above tests against that same field, so deriving the ramp's
    // center from it is what guarantees the AA band straddles the disc this
    // `Sun` actually has. (trace_common.hlsli::sky_disc does the same with
    // sun.w — one rule, both renderers.)
    let radius = sun.cos_radius.clamp(-1.0, 1.0).acos();
    let theta = c.acos();
    let cov = ((radius + half_angle - theta) / (2.0 * half_angle)).clamp(0.0, 1.0);
    sun.radiance * cov
}

/// A star's nominal angular radius — used as the floor of the splat width, so
/// stars stay sub-footprint points at every real resolution.
const STAR_ANGULAR_RADIUS: f32 = 0.03 * std::f32::consts::PI / 180.0;
/// Irradiance of a tier-1.0 star. Authored as IRRADIANCE, not radiance: the
/// splat divides by its own solid angle, so a star delivers the same energy to
/// a pixel whatever the footprint — no dimming/blooming with resolution.
const STAR_E: f32 = 2.0e-6;
/// Star cells per cube-face side (6·64² cells, ~40% occupied ⇒ ~10k stars).
const STAR_CELLS: u32 = 64;
/// Radiance ceiling per star — far under f16 max, so the upscaler wires and
/// the RGBA16F sky can never be spiked by a tiny footprint.
const STAR_L_MAX: f32 = 4096.0;

/// The star field's TOTAL above-horizon flux (∫L dω over the upper hemisphere)
/// at `night = 1`, per channel — i.e. the energy `stars()` actually puts in the
/// sky. `star_glow` spreads exactly this over the hemisphere, which is what
/// makes the point field and its smooth mean the SAME field rather than two
/// independently-authored lights.
///
/// A LITERAL, mirrored in trace_common.hlsli (the clouds-wind / ripple-constant
/// idiom — the twin is identical by construction), and pinned by `self_test`'s
/// enumeration gate: that gate walks all 6·`STAR_CELLS`² cells through the same
/// occupancy/tier/tint logic `stars()` uses and fails if this drifts from the
/// field's real energy. Do not hand-tune it — retune `STAR_AMBIENT_K`.
const STAR_FLUX: Vec3A = Vec3A::new(6.87870e-3, 6.67376e-3, 6.66328e-3);

/// How much of the star field's own flux reaches the GATHER paths. 1.0 is
/// energy-honest: starlight lights the scene with exactly the energy it shows
/// in the sky. The one artistic knob here (the `DOME_SCALE` / `MOON_E_OVER_PI`
/// pattern) — and, like `MOON_E_OVER_PI`, the honest number is already an
/// artistic one, since `STAR_E` was authored to make points read at interactive
/// resolutions rather than to sit 500x under moonlight the way real starlight
/// does. Lower this if night wants a subtler floor.
pub const STAR_AMBIENT_K: f32 = 1.0;

/// The exact integer mix trace_common.hlsli's `pcg_mix` computes — the star
/// field is a pure function of this hash, so mirroring it is what makes the
/// CPU and HLSL fields the SAME sky. Change both together.
#[inline(always)]
pub(crate) fn pcg_mix(s: u32) -> u32 {
    let s = s.wrapping_mul(747796405).wrapping_add(2891336453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277803737);
    (w >> 22) ^ w
}
#[inline(always)]
pub(crate) fn hash01(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16777216.0)
}

/// Procedural twinkling stars — DISPLAY paths only, like the disc (never
/// `dome()`/gather: a point field would alias catastrophically in the SH
/// projection and hemi cells, and its energy is negligible). Deterministic,
/// ZERO rng draws; `twinkle` is a frame index (`ctx.frame` on the primary
/// miss; secondary paths pass 0 and render the fixed-phase field), so every
/// same-seed/replay contract is untouched.
///
/// `night` (`Scene::night`) gates the whole field: exactly 0.0 in an untouched
/// session, and the guard is a BRANCH, so the day sky is bit-identical by
/// construction, not by arithmetic.
///
/// Geometry: the direction's dominant-axis cube face is split into
/// `STAR_CELLS`² hash cells; an occupied cell holds one star, inset to the
/// cell's inner 80% so a single-cell lookup needs no neighbor scan. The star
/// renders as a Gaussian splat of width `max(half_angle/2, star radius)` —
/// energy-conserving in the footprint (the `disc()` AA argument taken to the
/// point limit), so stars neither crawl at 1 spp nor change brightness with
/// render resolution.
pub fn stars(d: Vec3A, half_angle: f32, night: f32, twinkle: u32) -> Vec3A {
    if night <= 0.0 || d.y <= 0.0 {
        return Vec3A::ZERO;
    }
    // Dominant-axis cube face: 0..5, plus the two in-face coordinates in
    // [-1, 1] BEFORE the perspective divide. Orientation per face is arbitrary
    // (it only feeds a hash) but must match the HLSL twin exactly.
    let ad = d.abs();
    let (face, b1, b2, ma) = if ad.x >= ad.y && ad.x >= ad.z {
        (if d.x > 0.0 { 0u32 } else { 1 }, d.y, d.z, ad.x)
    } else if ad.y >= ad.z {
        (if d.y > 0.0 { 2 } else { 3 }, d.x, d.z, ad.y)
    } else {
        (if d.z > 0.0 { 4 } else { 5 }, d.x, d.y, ad.z)
    };
    let u = (b1 / ma) * 0.5 + 0.5;
    let v = (b2 / ma) * 0.5 + 0.5;
    let n = STAR_CELLS;
    let cx = ((u * n as f32) as u32).min(n - 1);
    let cy = ((v * n as f32) as u32).min(n - 1);

    let seed = face * n * n + cy * n + cx;
    let h0 = pcg_mix(seed);
    // ~40% of cells hold a star.
    if h0 & 0xff >= 102 {
        return Vec3A::ZERO;
    }
    let h1 = pcg_mix(h0);
    let h2 = pcg_mix(h1);
    let h3 = pcg_mix(h2);

    // The star's own direction: inset sub-position in the cell, mapped back
    // through the same face parameterization and normalized.
    let su = (cx as f32 + 0.1 + 0.8 * hash01(h1)) / n as f32 * 2.0 - 1.0;
    let sv = (cy as f32 + 0.1 + 0.8 * hash01(h2)) / n as f32 * 2.0 - 1.0;
    let sdir = match face {
        0 => Vec3A::new(1.0, su, sv),
        1 => Vec3A::new(-1.0, su, sv),
        2 => Vec3A::new(su, 1.0, sv),
        3 => Vec3A::new(su, -1.0, sv),
        4 => Vec3A::new(su, sv, 1.0),
        _ => Vec3A::new(su, sv, -1.0),
    }
    .normalize();

    // Gaussian splat over angle. theta² = 2(1-cosθ) to second order — exact
    // enough at star scales and cheaper/robuster than acos.
    let sigma = (half_angle * 0.5).max(STAR_ANGULAR_RADIUS);
    let theta2 = (2.0 * (1.0 - d.dot(sdir))).max(0.0);
    let g = (-theta2 / (2.0 * sigma * sigma)).exp();
    if g < 1e-4 {
        return Vec3A::ZERO;
    }

    // Brightness tier (0.25x..2x), a warm/cool tint, and the twinkle — a
    // deterministic re-hash of the star id with the frame's 8-frame bucket
    // (~7 Hz at 60 fps), swinging ±25% about the mean.
    let tier = 0.25 * (1u32 << (h3 & 3)) as f32;
    let warm = hash01(pcg_mix(h3));
    let tint = Vec3A::new(0.75, 0.85, 1.0).lerp(Vec3A::new(1.0, 0.85, 0.7), warm);
    let tw = 0.75 + 0.25 * hash01(pcg_mix(seed ^ pcg_mix(twinkle >> 3)));

    let l = (STAR_E * tier / (std::f32::consts::TAU * sigma * sigma)).min(STAR_L_MAX);
    tint * (l * g * tw * night * rise(d.y, 0.0, 0.05))
}

/// The star field's smooth MEAN — the representation GATHER paths integrate,
/// exactly as `stars()` is the representation display paths see.
///
/// The field cannot be projected as points: `sh::Sh9::project` is a 16,384-point
/// quadrature and the stars cover ~0.067% of the sphere, so a direct projection
/// would land ~11 random hits — sampling noise that would shift with
/// `PROJ_SAMPLES`. But order-2 SH carries only DC + linear + quadratic, and a
/// near-uniform point field's entire low-order content IS its mean, so this is
/// the EXACT order-2 projection of the field, minus the noise. `hemi`'s cells
/// and GI leaf misses take the same term for the same reason (and at ~1e-3
/// radiance it comes nowhere near the 2^18 fixed-point accumulator that bans
/// the disc from those paths).
///
/// `night` (`Scene::night`) gates it exactly as it gates the points, and the
/// guard is a BRANCH — an untouched day session is bit-identical by
/// construction, not by arithmetic.
///
/// Shape: `STAR_FLUX·K` spread over the 2π sr above the horizon, carried
/// across the horizon by `dome`'s OWN blend band with the same `GROUND_ALBEDO`
/// bounce below it. Both halves are load-bearing — a hard horizon step rings
/// under order-2 truncation (Gibbs) and can put negative lobes in the ambient
/// (see `GROUND_ALBEDO`'s comment), and the ground term is *additional*
/// re-emission rather than part of the flux budget, matching `dome`'s own
/// convention, which is why the normalization is over the upper hemisphere
/// alone.
pub fn star_glow(d: Vec3A, night: f32) -> Vec3A {
    if night <= 0.0 {
        return Vec3A::ZERO;
    }
    let l = STAR_FLUX * (STAR_AMBIENT_K / std::f32::consts::TAU);
    let t = ((d.y + 0.05) / 0.10).clamp(0.0, 1.0);
    l * ((GROUND_ALBEDO + (1.0 - GROUND_ALBEDO) * t) * night)
}

/// What GATHER paths integrate: the scattering dome plus the star field's mean.
///
/// The disc-exactly-once invariant, extended one row (see the module header):
/// the star field is delivered exactly ONCE per path too, in the representation
/// that path can sample — points to the eye through `radiance()`, mean to the
/// gathers here — with identical total energy either way. No double count, and
/// no MIS partner is needed because no ray ever importance-samples a star.
///
/// Every gather site calls THIS: the SH projection (`scene::refresh_sky_sh`),
/// `hemi::sky_cell`, hemi's GI leaf miss, and both `--check` GI reference
/// estimators — the references in lockstep with hemi or the A/B would be
/// scoring two different functions.
#[inline]
pub fn gather(d: Vec3A, sun: Vec3A, scale: f32, night: f32) -> Vec3A {
    let dm = dome(d, sun, scale);
    if night <= 0.0 {
        // BITWISE the pre-feature gather. A branch, deliberately not `+ 0.0`.
        return dm;
    }
    dm + star_glow(d, night)
}

/// What an escaping ray SEES: the infinity backdrop (dome + disc + stars)
/// through the cloud layer, plus the layer's own scatter. For *display* paths
/// only — see the module's central invariant. Gather paths must call `dome()`.
///
/// `o` is the ray's ORIGIN — the cloud slab is finite-altitude geometry, so
/// unlike everything else at infinity the ray's start matters (parallax is
/// what makes clouds drift *overhead* rather than painted on). `half_angle`
/// is the ray's angular footprint (primary: `pixel_cone/2`;
/// reflection/glass continuations: their cone's spread/2), used to antialias
/// the disc's limb and to size the star splats. `scale`/`night` are the
/// scene's time-of-day state (`Scene::sky_scale`/`Scene::night` — 1.0/0.0
/// untouched); `twinkle` is the frame index on primary misses, 0 on secondary
/// paths.
///
/// The cloud rule (the disc-once table's extension): the WHOLE backdrop —
/// disc and stars included — is extinguished by the layer's transmittance
/// along this ray; the guarded arms (`--no-clouds`, and any ray whose march
/// meets no cloud) return the backdrop bit-identically.
/// `j` is the cloud march's phase dither (`clouds::dither_jk(x, y, frame, k,
/// spp)` where the caller has a pixel — per (pixel, frame, sample), so spp
/// and accumulation average the march; 0.5 — the fixed-midpoint legacy phase
/// — on pixel-less paths like the glass miss). Pure function, zero rng draws.
#[inline]
pub fn radiance(
    o: Vec3A,
    d: Vec3A,
    sun: &Sun,
    half_angle: f32,
    scale: f32,
    night: f32,
    twinkle: u32,
    cl: &crate::clouds::Clouds,
    j: f32,
) -> Vec3A {
    let dm = dome(d, sun.dir, scale);
    let backdrop = dm + disc(d, sun, half_angle) + stars(d, half_angle, night, twinkle);
    if !cl.enabled {
        return backdrop;
    }
    match crate::clouds::along(o, d, sun, dm * crate::clouds::CLOUD_AMB_K, cl, j) {
        None => backdrop,
        Some(cs) => backdrop * cs.t + cs.scatter,
    }
}

/// Balance-heuristic MIS weight for the BSDF-sampling strategy against light
/// sampling, at a direction where BOTH strategies can find the sun.
///
/// The specular sun is reachable two ways: the direct loop cone-samples the
/// light and evaluates GGX (`direct_s`), and the VNDF reflection ray can land
/// in the disc. Counting both is a real double-count — and worse, on a ROUGH
/// surface the reflection ray hits a ~1e3-radiance disc with ~1e-3 probability,
/// which is a firefly generator (and it lands in FSR's UN-denoised residual).
///
/// `w_b = p_b / (p_b + p_l)` sends the energy to whichever strategy is actually
/// good at finding it: on rough surfaces `p_b << p_l` so the reflection ray's
/// disc contribution vanishes and the low-variance light-sampled term carries
/// the highlight; on mirrors `p_b >> p_l` so the reflection ray's sharp disc
/// carries it and the light-sampled spike is suppressed. Zero extra rays, zero
/// extra rng draws — both pdfs are already in scope at both call sites.
#[inline(always)]
pub fn mis_weight(p_bsdf: f32, p_light: f32) -> f32 {
    let s = p_bsdf + p_light;
    if s > 0.0 {
        p_bsdf / s
    } else {
        0.0
    }
}

/// The pdf of the light-sampling strategy: uniform in the sun's cone.
#[inline(always)]
pub fn light_pdf(sun: &Sun) -> f32 {
    1.0 / sun.solid_angle()
}

/// The star field's real above-horizon flux, by EXACT ENUMERATION of every
/// hash cell — the oracle `STAR_FLUX` is pinned against.
///
/// This walks the same 6·`STAR_CELLS`² cells `stars()` hashes, through the same
/// occupancy test, the same tier/tint decode, and the same horizon ramp, and
/// sums `∫L dω` per star (which is `STAR_E·tier·tint·tw` exactly: the splat's
/// Gaussian integrates to `2πσ²` and its radiance carries `1/(2πσ²)`, so the
/// footprint cancels — that cancellation is the whole reason `STAR_E` is
/// authored as an irradiance).
///
/// Two deliberate approximations, both far inside the gate's tolerance: the
/// twinkle is taken at its mean `E_TW` (a frame's `tw` only redistributes ±25%
/// among ~4.9k stars, and the smooth term represents the field's expectation,
/// not one frame of it), and the horizon ramp is evaluated at the STAR's own
/// direction rather than the query direction (they differ by the ~0.03° splat
/// width). Deterministic, no rng, ~24.5k hash chains — microseconds.
fn enumerate_star_flux() -> Vec3A {
    /// Mean of `tw = 0.75 + 0.25·U[0,1)`.
    const E_TW: f32 = 0.875;
    let n = STAR_CELLS;
    let mut flux = Vec3A::ZERO;
    for face in 0..6u32 {
        for cy in 0..n {
            for cx in 0..n {
                let seed = face * n * n + cy * n + cx;
                let h0 = pcg_mix(seed);
                if h0 & 0xff >= 102 {
                    continue;
                }
                let h1 = pcg_mix(h0);
                let h2 = pcg_mix(h1);
                let h3 = pcg_mix(h2);
                let su = (cx as f32 + 0.1 + 0.8 * hash01(h1)) / n as f32 * 2.0 - 1.0;
                let sv = (cy as f32 + 0.1 + 0.8 * hash01(h2)) / n as f32 * 2.0 - 1.0;
                let sdir = match face {
                    0 => Vec3A::new(1.0, su, sv),
                    1 => Vec3A::new(-1.0, su, sv),
                    2 => Vec3A::new(su, 1.0, sv),
                    3 => Vec3A::new(su, -1.0, sv),
                    4 => Vec3A::new(su, sv, 1.0),
                    _ => Vec3A::new(su, sv, -1.0),
                }
                .normalize();
                if sdir.y <= 0.0 {
                    continue;
                }
                let tier = 0.25 * (1u32 << (h3 & 3)) as f32;
                let warm = hash01(pcg_mix(h3));
                let tint = Vec3A::new(0.75, 0.85, 1.0).lerp(Vec3A::new(1.0, 0.85, 0.7), warm);
                flux += tint * (STAR_E * tier * E_TW * rise(sdir.y, 0.0, 0.05));
            }
        }
    }
    flux
}

/// Closed-form gates, run by `--check`. No rng, no scene, no DLLs.
pub fn self_test() -> Result<(), String> {
    let sun = Sun::new(Vec3A::new(6.0, 10.0, 4.0));

    // G1: the disc's radiance must integrate back to the authored irradiance.
    // E = L·Ω·cos(0) over a small cone ⇒ e_over_pi == L·Ω/π. This is the
    // classic place to be off by 4π, and it would silently rescale the sun.
    let round = sun.radiance * (sun.solid_angle() / std::f32::consts::PI);
    if (round - sun.e_over_pi).abs().max_element() > 1e-3 {
        return Err(format!(
            "disc radiance does not round-trip: {round:?} vs e_over_pi {:?}",
            sun.e_over_pi
        ));
    }

    // G2: cone sampling stays INSIDE the cone, for the whole unit square, and
    // covers it (a mapping that collapsed to the axis would still pass "inside").
    let mut max_ang: f32 = 0.0;
    for i in 0..64 {
        for j in 0..64 {
            let r1 = (i as f32 + 0.5) / 64.0;
            let r2 = (j as f32 + 0.5) / 64.0;
            let w = sun.sample_dir(r1, r2);
            if (w.length() - 1.0).abs() > 1e-3 {
                return Err(format!("cone sample not unit: |w| = {}", w.length()));
            }
            let c = w.dot(sun.dir);
            if c < sun.cos_radius - 1e-4 {
                return Err(format!("cone sample outside the disc: cos {c} < {}", sun.cos_radius));
            }
            max_ang = max_ang.max(c.clamp(-1.0, 1.0).acos());
        }
    }
    if max_ang < SUN_ANGULAR_RADIUS * 0.9 {
        return Err(format!(
            "cone sampling collapsed: max angle {:.4} rad, expected ~{:.4}",
            max_ang, SUN_ANGULAR_RADIUS
        ));
    }

    // G3: the disc is exactly the cone the sampler draws from — the two must
    // agree or light sampling and BSDF sampling disagree about where the sun IS.
    // At half_angle = 0 the disc is the hard step it physically is.
    if disc(sun.dir, &sun, 0.0) != sun.radiance {
        return Err("the sun's own direction is not inside its disc".into());
    }
    let just_out = sun.sample_dir(1.0, 0.0);
    let eps_out = (just_out * 1.001 - sun.dir * 0.001).normalize();
    if disc(eps_out, &sun, 0.0) != Vec3A::ZERO && eps_out.dot(sun.dir) < sun.cos_radius {
        return Err("disc test disagrees with the cone boundary".into());
    }

    // G3b: the antialiased limb. Coverage must be monotone from 1 (well inside)
    // to 0 (well outside), hit exactly 1/2 ON the limb, and — the part that
    // matters — CONVERGE to the hard step as the footprint shrinks, so the AA is
    // a filter over the true disc and not a different sun.
    let ha = 0.2 * SUN_ANGULAR_RADIUS;
    let at = |ang: f32| -> f32 {
        // A direction `ang` radians off the sun, built in the sun's own frame.
        let (t1, _) = crate::shade::onb(sun.dir);
        let d = (sun.dir * ang.cos() + t1 * ang.sin()).normalize();
        disc(d, &sun, ha).x / sun.radiance.x
    };
    if at(SUN_ANGULAR_RADIUS - 2.0 * ha) < 0.999 {
        return Err("disc AA: well inside the limb is not fully covered".into());
    }
    if at(SUN_ANGULAR_RADIUS + 2.0 * ha) > 1e-6 {
        return Err("disc AA: well outside the limb is not zero".into());
    }
    let on_limb = at(SUN_ANGULAR_RADIUS);
    if (on_limb - 0.5).abs() > 1e-3 {
        return Err(format!("disc AA: coverage ON the limb is {on_limb:.4}, want 0.5"));
    }
    let mut prev = 2.0;
    for i in 0..=64 {
        let ang = SUN_ANGULAR_RADIUS * (0.5 + 1.0 * i as f32 / 64.0);
        let cov = at(ang);
        if cov > prev + 1e-6 {
            return Err("disc AA: coverage is not monotone across the limb".into());
        }
        prev = cov;
    }
    // Shrinking the footprint must reproduce the hard step (a probe just inside
    // the limb goes to full radiance, one just outside goes to zero).
    for &tiny in &[1e-3, 1e-5] {
        let (t1, _) = crate::shade::onb(sun.dir);
        let ang_in = SUN_ANGULAR_RADIUS * 0.99;
        let d_in = (sun.dir * ang_in.cos() + t1 * ang_in.sin()).normalize();
        let cov_in = disc(d_in, &sun, tiny * SUN_ANGULAR_RADIUS).x / sun.radiance.x;
        if cov_in < 0.999 {
            return Err(format!("disc AA does not converge to the hard step: {cov_in:.4}"));
        }
    }

    // G4: the dome carries NO disc. A gather path integrating `dome` must never
    // see sun radiance — that is the invariant the hemi accumulator's 2^18 fixed
    // point depends on (the disc would saturate it outright).
    //
    // The test is RELATIVE, not an absolute ceiling: `dome` legitimately peaks
    // well above its average AT the sun's direction, because that is where the
    // Mie aureole is (Henyey-Greenstein at g = 0.76 peaks ~30x isotropic). What
    // must be true is that the aureole and the DISC are orders of magnitude
    // apart — a leak would put the disc's full radiance into the dome, which is
    // ~100x over this bound rather than a hair over it.
    let at_sun = dome(sun.dir, sun.dir, 1.0).max_element();
    let leak_limit = sun.radiance.max_element() * 0.01;
    if at_sun > leak_limit {
        return Err(format!(
            "dome() at the sun's direction peaks at {at_sun:.2}, over 1% of the disc's \
             {:.0} — the disc has leaked into the dome",
            sun.radiance.max_element()
        ));
    }
    // ...and the disc really is there for the paths that SHOULD see it.
    if radiance(Vec3A::ZERO, sun.dir, &sun, 0.0, 1.0, 0.0, 0, &crate::clouds::Clouds::off(), 0.5)
        .max_element()
        < sun.radiance.max_element()
    {
        return Err("radiance() at the sun's direction has no disc".into());
    }

    // G5: the dome is non-negative and finite everywhere, INCLUDING below the
    // horizon (the SH projection integrates the full sphere).
    for i in 0..2000 {
        let a = i as f32 * 2.399_963;
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / 2000.0;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let d = Vec3A::new(r * a.cos(), z, r * a.sin());
        let v = dome(d, sun.dir, 1.0);
        if !v.is_finite() || v.min_element() < 0.0 {
            return Err(format!("dome({d:?}) = {v:?} — must be finite and non-negative"));
        }
    }

    // G6: the ambient the dome produces must land in a physically sane band.
    // A clear sky delivers roughly 15-25% of the sun's irradiance; the old flat
    // AMBIENT constant sat at 0.168 luminance, which is inside that band. This
    // is what pins DOME_SCALE — if someone retunes the dome for looks and the
    // ambient drifts out of band, the scene's exposure has silently moved.
    let sh = crate::sh::Sh9::project(|d| dome(d, sun.dir, 1.0));
    let e_up = sh.irradiance(Vec3A::Y);
    let lum = e_up.dot(Vec3A::new(0.2126, 0.7152, 0.0722));
    if !(0.10..=0.30).contains(&lum) {
        return Err(format!(
            "sky ambient luminance {lum:.4} is outside the sane band [0.10, 0.30] \
             (DOME_SCALE = {DOME_SCALE}); the old flat AMBIENT was 0.168"
        ));
    }
    // ...and it must be BLUER than it is red — that is the whole point of the
    // 1/λ⁴ Rayleigh weighting, and a channel-swap would otherwise pass silently.
    if e_up.z <= e_up.x {
        return Err(format!("sky ambient is not blue-dominant: {e_up:?}"));
    }

    // G7: the star field's DAY bit-identity. `star_glow` must be exactly ZERO
    // and `gather` must return `dome` BITWISE — both are branches on `night`,
    // so an untouched day session is unchanged by construction. If this ever
    // becomes an arithmetic `+ 0.0`, this gate is what catches the -0.0 case.
    for i in 0..2000 {
        let a = i as f32 * 2.399_963;
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / 2000.0;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let d = Vec3A::new(r * a.cos(), z, r * a.sin());
        if star_glow(d, 0.0) != Vec3A::ZERO {
            return Err(format!("star_glow({d:?}) is not exactly zero by day"));
        }
        for &scale in &[1.0f32, MOON_DOME_FRAC] {
            if gather(d, sun.dir, scale, 0.0) != dome(d, sun.dir, scale) {
                return Err(format!("gather != dome bitwise at night = 0, d = {d:?}"));
            }
        }
    }

    // G8: ENERGY EQUALITY — the load-bearing gate, and the reason the point
    // field and its smooth mean are one field rather than two lights.
    //
    // `STAR_FLUX` (the literal `star_glow` spreads, and the HLSL twin mirrors)
    // must equal the flux the field actually emits, enumerated cell by cell.
    // Then the glow, integrated back over the upper hemisphere, must return
    // that same flux — which is what pins the 1/TAU normalization and the
    // horizon shape together. Measured ~0.9% short, and that shortfall is the
    // blend band: `dome`'s ±0.05 ramp costs the upper hemisphere a little and
    // hands it to the ground bounce below, exactly as `dome`'s own does
    // (deliberate — see `star_glow`). Hence a 2% tolerance and not an ulp
    // count; a wrong normalization or horizon shape misses by tens of percent.
    let enumerated = enumerate_star_flux();
    let drift = ((STAR_FLUX - enumerated) / enumerated).abs().max_element();
    if drift > 0.02 {
        return Err(format!(
            "STAR_FLUX {STAR_FLUX:?} is {:.1}% off the field's enumerated flux \
             {enumerated:?} — set the literal (and its trace_common.hlsli twin) \
             to the enumerated value",
            drift * 100.0
        ));
    }
    // ∫ star_glow dω over d.y > 0, by the same Fibonacci quadrature `project`
    // uses (weight 4π/N, upper half only).
    const GN: usize = 40_000;
    let mut integ = Vec3A::ZERO;
    for i in 0..GN {
        let a = i as f32 * 2.399_963;
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / GN as f32;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let d = Vec3A::new(r * a.cos(), z, r * a.sin());
        if d.y > 0.0 {
            integ += star_glow(d, 1.0);
        }
    }
    integ *= 4.0 * std::f32::consts::PI / GN as f32;
    let ierr = ((integ - enumerated) / enumerated).abs().max_element();
    if ierr > 0.02 {
        return Err(format!(
            "star_glow integrates to {integ:?} over the upper hemisphere, \
             {:.1}% off the field's own flux {enumerated:?} — the points and \
             the mean are no longer the same field",
            ierr * 100.0
        ));
    }

    // G9: the glow's shape. Finite and non-negative everywhere (the SH
    // projection integrates the full sphere), the below-band value EXACTLY
    // GROUND_ALBEDO times the above-band value, and no step at the horizon —
    // a discontinuity there rings under order-2 truncation and would put
    // negative lobes in the ambient (the reason `dome` blends at all).
    let above = star_glow(Vec3A::Y, 1.0);
    let below = star_glow(Vec3A::NEG_Y, 1.0);
    if (below - above * GROUND_ALBEDO).abs().max_element() > 1e-12 {
        return Err(format!("star_glow below the horizon is {below:?}, want {above:?} * albedo"));
    }
    let mut prev = star_glow(Vec3A::new(0.0, -1.0, 0.0), 1.0).x;
    for i in 0..=400 {
        let y = -0.2 + 0.4 * i as f32 / 400.0;
        let v = star_glow(Vec3A::new((1.0f32 - y * y).max(0.0).sqrt(), y, 0.0), 1.0);
        if !v.is_finite() || v.min_element() < 0.0 {
            return Err(format!("star_glow at y = {y} is {v:?} — must be finite, non-negative"));
        }
        // Monotone up through the band, and no jump bigger than the band's own
        // slope times the step (a step would show as a large single delta).
        if v.x < prev - 1e-12 || v.x - prev > above.x * 0.05 {
            return Err(format!("star_glow is not smooth across the horizon at y = {y}"));
        }
        prev = v.x;
    }

    // G10: the NIGHT AMBIENT MUST-FIRE — anti-vacuity. Everything above passes
    // just as well with STAR_FLUX = 0, so this asserts the floor is really
    // there: at the moonlit dome's own scale, projecting `gather` must lift the
    // up-facing irradiance measurably over `dome` alone. The band is authored
    // (the DOME_SCALE precedent): starlight at K = 1 is the same ORDER as
    // moonlight, deliberately, and if it silently becomes 10x either way the
    // night exposure has moved.
    let moon_dir = Vec3A::new(-6.0, 10.0, -4.0).normalize();
    let night_dome = crate::sh::Sh9::project(|d| dome(d, moon_dir, MOON_DOME_FRAC));
    let night_gather =
        crate::sh::Sh9::project(|d| gather(d, moon_dir, MOON_DOME_FRAC, 1.0));
    let e_moon = night_dome.irradiance(Vec3A::Y);
    let e_night = night_gather.irradiance(Vec3A::Y);
    let lift = e_night - e_moon;
    if lift.min_element() <= 0.0 {
        return Err(format!(
            "starlight adds no night ambient: dome-only {e_moon:?}, with stars {e_night:?}"
        ));
    }
    let ratio = lift.dot(Vec3A::new(0.2126, 0.7152, 0.0722))
        / e_moon.dot(Vec3A::new(0.2126, 0.7152, 0.0722));
    if !(0.1..=3.0).contains(&ratio) {
        return Err(format!(
            "starlight ambient is {ratio:.2}x the moonlit dome's, outside the \
             authored band [0.1, 3.0] (STAR_AMBIENT_K = {STAR_AMBIENT_K}); \
             night exposure has moved"
        ));
    }

    eprintln!(
        "sky self-test: OK (sun {:.1}° radius, disc radiance {:.0}, ambient lum {:.3} {:?}; \
         star flux {:.3e} = {:.1}% of enumerated, night floor {:.2}x the moonlit dome)",
        SUN_ANGULAR_RADIUS.to_degrees(),
        sun.radiance.x,
        lum,
        e_up,
        STAR_FLUX.x,
        (1.0 - drift) * 100.0,
        ratio
    );
    Ok(())
}
