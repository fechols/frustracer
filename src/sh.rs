//! Order-2 spherical harmonics for sky irradiance.
//!
//! The renderer's sky is one object split by FREQUENCY into two
//! representations, because the two bands need different sampling strategies:
//! the smooth scattering dome lives here (9 RGB coefficients, analytic
//! irradiance, zero rays), and the sharp sun disc stays an explicit,
//! cone-sampled, shadow-rayed light. See `shade::sky_dome` / `shade::sun_disc`.
//!
//! Order 2 is not a compromise — it is exact enough to be uninteresting.
//! Irradiance is a convolution of radiance with a clamped-cosine kernel, whose
//! own spectrum collapses above l = 2 (Ramamoorthi & Hanrahan 2001): for ANY
//! sky, 9 coefficients carry >99% of the energy a Lambertian surface can see.
//! What SH CANNOT do is the sun — a peaked source needs bands up to
//! l ≈ 180°/θ, rings under truncation, and (fatally) cannot be shadow-rayed.
//! That is why the sun is not in here.
//!
//! Convention (load-bearing — `hemi.rs`'s doc comment states the same one):
//! `irradiance()` returns cosine-weighted incoming radiance **divided by π**,
//! so a uniform sky of radiance L yields exactly L. Lambert's 1/π lives in the
//! light terms, which is what makes this a drop-in for the old `AMBIENT · ao`
//! and directly comparable to `hemi::gi`'s open-hemisphere result.
//!
//! Everything here is a pure function and draws ZERO rng — the projection is a
//! deterministic Fibonacci-sphere quadrature. That is what keeps the
//! byte-identical-build, replay, and same-seed contracts intact.

use glam::Vec3A;

/// Coefficients in an order-2 real SH basis (l = 0, 1, 2).
pub const N: usize = 9;

/// Fibonacci-sphere quadrature points for `project`. Deterministic, and the
/// count is a constant so a rebuild can never shift a coefficient.
const PROJ_SAMPLES: usize = 16384;

/// Real SH basis, evaluated at a unit direction. Any consistent basis works so
/// long as `project` and `eval`/`irradiance` share it; these are the standard
/// normalized constants, and `self_test` pins them by round-tripping each band
/// through the projection (a typo in any constant breaks orthonormality).
#[inline(always)]
fn basis(d: Vec3A) -> [f32; N] {
    let (x, y, z) = (d.x, d.y, d.z);
    [
        0.282_095,          // Y(0, 0)
        0.488_603 * y,      // Y(1,-1)
        0.488_603 * z,      // Y(1, 0)
        0.488_603 * x,      // Y(1, 1)
        1.092_548 * x * y,  // Y(2,-2)
        1.092_548 * y * z,  // Y(2,-1)
        0.315_392 * (3.0 * z * z - 1.0), // Y(2, 0)
        1.092_548 * x * z,  // Y(2, 1)
        0.546_274 * (x * x - y * y),     // Y(2, 2)
    ]
}

/// An order-2 RGB spherical-harmonic radiance field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sh9 {
    pub c: [Vec3A; N],
}

impl Sh9 {
    pub const ZERO: Sh9 = Sh9 { c: [Vec3A::ZERO; N] };

    /// Scale the field's radiance by `k`.
    ///
    /// Projection is LINEAR, so this is EXACTLY the field `project` would
    /// return for `k·f` — which is what lets `scene::apply_light_gain` brighten
    /// the sky ambient by scaling nine coefficients instead of re-projecting.
    /// That distinction is not a micro-optimization: `Sh9::project` is a
    /// 16,384-point quadrature and calling it per frame was measured as a
    /// 235 -> 17 fps stall (see `scene::apply_tod_lit`'s note).
    ///
    /// `k == 1.0` returns the field BITWISE (a branch, never `* 1.0`) — the
    /// structural-off discipline the whole light-gain feature rests on.
    pub fn scaled(self, k: f32) -> Sh9 {
        if k == 1.0 {
            return self;
        }
        let mut out = self;
        for c in &mut out.c {
            *c *= k;
        }
        out
    }

    /// Project a radiance function over the whole sphere.
    ///
    /// Deterministic quadrature (Fibonacci sphere, uniform solid-angle weight
    /// 4π/N) — NOT Monte Carlo. `f` must be the SUN-FREE dome: projecting a
    /// sharp sun here would smear it into a hemisphere-wide gradient AND
    /// double-count it against the explicit sun light.
    pub fn project(f: impl Fn(Vec3A) -> Vec3A + Sync) -> Sh9 {
        use rayon::prelude::*;
        // PARALLEL over fixed chunks with an ORDERED serial fold of the
        // partials (the two-phase BVH build discipline): deterministic
        // across runs AND thread counts — the sum is a pure function of
        // PROJ_SAMPLES/PROJ_CHUNK alone. It went parallel because the
        // world's TOD attractors re-project per MOVING frame, and the
        // serial 16k-sample pass was a ~54 ms main-thread stall (measured
        // 235 -> 17 fps flying; GPU idle). NOTE the association differs
        // from the old serial loop by design (a ±ulp move of every
        // coefficient — the reassociation class, invisible next to the
        // gates' tolerances).
        const PROJ_CHUNK: usize = 512;
        const _: () = assert!(PROJ_SAMPLES % PROJ_CHUNK == 0);
        let n = PROJ_SAMPLES as f32;
        // Golden angle — the Fibonacci spiral's azimuthal increment.
        let ga = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        let partials: Vec<[Vec3A; N]> = (0..PROJ_SAMPLES / PROJ_CHUNK)
            .into_par_iter()
            .map(|ci| {
                let mut c = [Vec3A::ZERO; N];
                for i in ci * PROJ_CHUNK..(ci + 1) * PROJ_CHUNK {
                    let fi = i as f32;
                    // z stratified over [-1, 1) at cell centers; r from the
                    // identity x²+y²+z² = 1. Equal-area in z ⇒ uniform on
                    // the sphere.
                    let z = 1.0 - 2.0 * (fi + 0.5) / n;
                    let r = (1.0 - z * z).max(0.0).sqrt();
                    let phi = ga * fi;
                    let d = Vec3A::new(r * phi.cos(), r * phi.sin(), z);
                    let l = f(d);
                    let b = basis(d);
                    for k in 0..N {
                        c[k] += l * b[k];
                    }
                }
                c
            })
            .collect();
        let mut c = [Vec3A::ZERO; N];
        for p in &partials {
            for k in 0..N {
                c[k] += p[k];
            }
        }
        let w = 4.0 * std::f32::consts::PI / n;
        for ck in &mut c {
            *ck *= w;
        }
        Sh9 { c }
    }

    /// Cosine-weighted irradiance at normal `n`, **divided by π** (see the
    /// module convention: a uniform sky of radiance L returns exactly L).
    ///
    /// The closed form of Ramamoorthi & Hanrahan 2001 §3: the clamped-cosine
    /// convolution kernel is a zonal harmonic with Â0 = π, Â1 = 2π/3,
    /// Â2 = π/4, and folding those into the basis constants collapses the
    /// whole hemisphere integral to these five numbers. `FRAC_1_PI` at the end
    /// is the convention divide, not part of the formula.
    ///
    /// Clamped at zero: a truncated SH series of a sky with a horizon step
    /// rings (Gibbs), and order 2 can undershoot into negative radiance. A
    /// negative irradiance is unphysical and would punch black holes into the
    /// ambient, so it is floored here rather than at each call site (`hemi::gi`
    /// clamps its own integrator the same way).
    pub fn irradiance(&self, n: Vec3A) -> Vec3A {
        const C1: f32 = 0.429_043;
        const C2: f32 = 0.511_664;
        const C3: f32 = 0.743_125;
        const C4: f32 = 0.886_227;
        const C5: f32 = 0.247_708;
        let (x, y, z) = (n.x, n.y, n.z);
        let e = self.c[0] * C4 - self.c[6] * C5
            + self.c[6] * (C3 * z * z)
            + self.c[8] * (C1 * (x * x - y * y))
            + (self.c[4] * (x * y) + self.c[7] * (x * z) + self.c[5] * (y * z)) * (2.0 * C1)
            + (self.c[3] * x + self.c[1] * y + self.c[2] * z) * (2.0 * C2);
        (e * std::f32::consts::FRAC_1_PI).max(Vec3A::ZERO)
    }
}

/// Closed-form gates, run by `--check`. No rng, no scene, no DLLs.
pub fn self_test() -> Result<(), String> {
    // G1: basis orthonormality. Projecting basis function k must return a unit
    // coefficient at k and zero everywhere else. This is what pins every
    // constant in `basis` — a single typo destroys it.
    for k in 0..N {
        let sh = Sh9::project(|d| Vec3A::splat(basis(d)[k]));
        for j in 0..N {
            let want = if j == k { 1.0 } else { 0.0 };
            let got = sh.c[j].x;
            if (got - want).abs() > 2e-3 {
                return Err(format!(
                    "basis not orthonormal: <Y{k}, Y{j}> = {got:.5}, want {want:.1}"
                ));
            }
        }
    }

    // G2: the convention pin. A uniform sky of radiance L must yield EXACTLY L
    // through irradiance() at every normal — this is what makes the result a
    // drop-in for `AMBIENT · ao` and comparable to hemi::gi.
    let l = Vec3A::new(0.3, 0.5, 0.9);
    let uni = Sh9::project(|_| l);
    for n in probe_normals() {
        let e = uni.irradiance(n);
        if (e - l).abs().max_element() > 1e-3 {
            return Err(format!(
                "uniform sky L={l:?} -> irradiance {e:?} at n={n:?} (want L)"
            ));
        }
    }

    // G3: accuracy vs ground truth. A cosine-weighted quadrature of a real
    // directional sky, compared against the 9-coefficient closed form. This is
    // the claim the whole design rests on — order 2 is enough for diffuse.
    // Ramamoorthi's own bound is ~1% worst case for arbitrary skies.
    let dome = |d: Vec3A| -> Vec3A {
        // A stand-in with the same shape as the real dome: a bright horizon
        // fading to a darker zenith, i.e. a strong l=1 term.
        let t = (d.y * 0.7 + 0.3).clamp(0.0, 1.0);
        Vec3A::new(0.72, 0.82, 0.95).lerp(Vec3A::new(0.18, 0.35, 0.70), t)
    };
    let sh = Sh9::project(dome);
    let mut worst: f32 = 0.0;
    let mut sum_rel = 0.0;
    let probes = probe_normals();
    for &n in &probes {
        let want = reference_irradiance(dome, n);
        let got = sh.irradiance(n);
        // Relative per channel, on a sky that is nowhere near zero.
        let rel = ((got - want) / want).abs().max_element();
        worst = worst.max(rel);
        sum_rel += rel;
    }
    let mean_rel = sum_rel / probes.len() as f32;
    if mean_rel > 0.01 || worst > 0.03 {
        return Err(format!(
            "SH irradiance vs reference: mean rel {:.3}% (limit 1%), worst {:.3}% (limit 3%)",
            mean_rel * 100.0,
            worst * 100.0
        ));
    }

    // G4: determinism. The projection is quadrature, not sampling — two runs
    // must agree BIT-for-bit or the byte-identical-build contract is a lie.
    if Sh9::project(dome) != sh {
        return Err("projection is not deterministic".into());
    }

    eprintln!(
        "sh self-test: OK (order-2 irradiance mean rel {:.3}%, worst {:.3}%)",
        mean_rel * 100.0,
        worst * 100.0
    );
    Ok(())
}

/// Brute-force cosine-weighted irradiance / π — the ground truth G3 scores
/// against. Deliberately the dumbest possible integrator.
fn reference_irradiance(f: impl Fn(Vec3A) -> Vec3A, n: Vec3A) -> Vec3A {
    const M: usize = 200_000;
    let mut sum = Vec3A::ZERO;
    let ga = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    for i in 0..M {
        let fi = i as f32;
        let z = 1.0 - 2.0 * (fi + 0.5) / M as f32;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = ga * fi;
        let d = Vec3A::new(r * phi.cos(), r * phi.sin(), z);
        let ndl = n.dot(d);
        if ndl > 0.0 {
            sum += f(d) * ndl;
        }
    }
    // Σ L·cos · (4π/M) is the irradiance E; /π is the renderer convention.
    sum * (4.0 / M as f32)
}

/// A fixed spread of normals — axis-aligned, and a few obliques that exercise
/// every cross term in the l=2 band.
fn probe_normals() -> Vec<Vec3A> {
    let mut v = vec![
        Vec3A::Y,
        Vec3A::NEG_Y,
        Vec3A::X,
        Vec3A::NEG_X,
        Vec3A::Z,
        Vec3A::NEG_Z,
    ];
    for &(x, y, z) in &[
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 2.0, 0.5),
        (0.3, -0.4, 0.9),
        (-0.6, 0.2, -0.7),
    ] {
        v.push(Vec3A::new(x, y, z).normalize());
    }
    v
}
