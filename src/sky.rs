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
//! | path                         | sees          | why                                              |
//! |------------------------------|---------------|--------------------------------------------------|
//! | primary / camera miss        | `radiance()`  | backdrop — nothing else delivers it              |
//! | glass / transmission miss    | `radiance()`  | near-delta path, no light-sampling partner       |
//! | specular reflection miss     | MIS-weighted  | `direct_s` also delivers it — see `mis_weight`   |
//! | hemi cells, GI leaf, SH proj | `dome()` ONLY | `direct_d` already delivers the sun's diffuse    |
//!
//! Break the last row and you get (a) a double count of light the direct loop
//! already added with its own shadow ray, and (b) a ~1e3-magnitude firefly into
//! `hemi`'s 2^18 fixed-point accumulator, which would saturate outright.

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
}

/// The smooth scattering dome — NO sun disc. Single-scattering Rayleigh + Mie.
///
/// This is what every *gather* path integrates (hemi cells, GI leaf misses, the
/// SH projection) and what `radiance()` adds the disc to. Defined over the FULL
/// sphere: below the horizon it is the ground bouncing the dome back, blended
/// smoothly so order-2 SH doesn't ring on a step.
pub fn dome(d: Vec3A, sun: Vec3A) -> Vec3A {
    // The sun's own path down to the scattering volume depends ONLY on its
    // elevation, so its transmittance is the same for every direction in the
    // frame — hoisted out of `scatter`, which would otherwise recompute this
    // Vec3A `exp` on each of its calls. `dome` is the hemisphere integrator's
    // inner loop (every proven-empty cell centroid, every GI leaf-ray miss),
    // so that is not a rounding-off saving.
    let bt = BETA_R + BETA_M;
    // ...and it is what reddens the sky when the sun is low: the blue is
    // scattered out of the sunlight before it ever reaches the air we look at.
    let m_sun = 1.0 / (sun.y.max(0.0) + 0.15);
    let t_sun = (-(bt * m_sun)).exp();

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
        return scatter(d, d.dot(sun).clamp(-1.0, 1.0));
    }
    let dm = Vec3A::new(d.x, d.y.abs(), d.z);
    let ground = scatter(dm, dm.dot(sun).clamp(-1.0, 1.0)) * GROUND_ALBEDO;
    if t <= 0.0 {
        return ground;
    }
    let sky = scatter(d, d.dot(sun).clamp(-1.0, 1.0));
    ground.lerp(sky, t)
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

/// What an escaping ray SEES: dome + disc. For *display* paths only — see the
/// module's central invariant. Gather paths must call `dome()`.
///
/// `half_angle` is the ray's angular footprint (primary: `pixel_cone/2`;
/// reflection/glass continuations: their cone's spread/2), used only to
/// antialias the disc's limb.
#[inline(always)]
pub fn radiance(d: Vec3A, sun: &Sun, half_angle: f32) -> Vec3A {
    dome(d, sun.dir) + disc(d, sun, half_angle)
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
    let at_sun = dome(sun.dir, sun.dir).max_element();
    let leak_limit = sun.radiance.max_element() * 0.01;
    if at_sun > leak_limit {
        return Err(format!(
            "dome() at the sun's direction peaks at {at_sun:.2}, over 1% of the disc's \
             {:.0} — the disc has leaked into the dome",
            sun.radiance.max_element()
        ));
    }
    // ...and the disc really is there for the paths that SHOULD see it.
    if radiance(sun.dir, &sun, 0.0).max_element() < sun.radiance.max_element() {
        return Err("radiance() at the sun's direction has no disc".into());
    }

    // G5: the dome is non-negative and finite everywhere, INCLUDING below the
    // horizon (the SH projection integrates the full sphere).
    for i in 0..2000 {
        let a = i as f32 * 2.399_963;
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / 2000.0;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let d = Vec3A::new(r * a.cos(), z, r * a.sin());
        let v = dome(d, sun.dir);
        if !v.is_finite() || v.min_element() < 0.0 {
            return Err(format!("dome({d:?}) = {v:?} — must be finite and non-negative"));
        }
    }

    // G6: the ambient the dome produces must land in a physically sane band.
    // A clear sky delivers roughly 15-25% of the sun's irradiance; the old flat
    // AMBIENT constant sat at 0.168 luminance, which is inside that band. This
    // is what pins DOME_SCALE — if someone retunes the dome for looks and the
    // ambient drifts out of band, the scene's exposure has silently moved.
    let sh = crate::sh::Sh9::project(|d| dome(d, sun.dir));
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

    eprintln!(
        "sky self-test: OK (sun {:.1}° radius, disc radiance {:.0}, ambient lum {:.3} {:?})",
        SUN_ANGULAR_RADIUS.to_degrees(),
        sun.radiance.x,
        lum,
        e_up
    );
    Ok(())
}
