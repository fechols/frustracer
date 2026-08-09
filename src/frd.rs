//! FRD — the frustracer denoiser: a from-scratch, redistributable,
//! ReBLUR-class recurrent diffuse+specular denoiser (`--frd`), pure Rust +
//! our own HLSL (gpu/frd_gpu.rs + shaders/frd_*.hlsl), written to replace the
//! NRD dependency for the pre-upscale denoise slot (XeSS/FSR3 sessions).
//!
//! PROVENANCE (the clean-room rule, load-bearing): everything here is
//! designed from the PUBLISHED literature — "ReBLUR: A Hierarchical
//! Recurrent Denoiser" (Ray Tracing Gems II ch. 49, Zhdan), the GDC talk
//! "Fast Denoising with Self-Stabilizing Recurrent Blurs", and SVGF
//! (Schied et al. 2017) — never from the NRD source tree, which is licensed
//! non-redistributable and is NOT read, quoted, or transcribed (the
//! nrd.rs::oracle never-paste rule, extended to the whole engine). The wire
//! format FRD consumes is the tree's OWN (nrd_bridge.hlsl's pack/out kernels,
//! reimplemented repo-side), so during the coexistence phase the shared
//! packing math is re-exported from nrd::oracle; it physically moves here
//! when NRD retires.
//!
//! DESIGN SHAPE (v1, the plan file's 3-dispatch collapse of the published
//! 8-pass architecture):
//!   1. cs_frd_temporal — sky vote, hit-dist reconstruction, firefly
//!      pre-clamp, small pre-blur, reprojection + disocclusion, slow/fast
//!      accumulation.
//!   2. cs_frd_blur     — Poisson disk blur at a hit-distance-driven radius,
//!      history fix folded in as a radius/threshold policy.
//!   3. cs_frd_post     — second disk at 1.7x radius, writes the RECURRENT
//!      slow-history feedback, fast-history clamp + anti-lag, optional
//!      stabilization, OUT planes + prev-geometry snapshots.
//! The recurrence (pass 3's post-blurred output IS the history pass 1
//! reprojects next frame) is what lets a ~30-frame cap converge: each
//! frame's blur compounds into the accumulated footprint.
//!
//! This module is the PURE half: constants, the tuning levers, and `oracle` —
//! CPU twins of every formula the FRD shaders will run. One formula, two
//! sites (oracle + frd_common.hlsli), the nrd_bridge discipline; the F0 gate
//! (`self_test`, run by `--check`) pins the math DLL-free and hardware-free.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Defaults — the compiled constants a flagless --frd session runs. Each knob
// a `--frd-*` lever can move lives in FrdTuning; everything else is a named
// const so the HLSL literal twin has one Rust source to be checked against.
// ---------------------------------------------------------------------------

/// Diffuse accumulation cap in frames (α = 1/(1+n) ⇒ ~3% steady-state blend).
pub const MAX_ACCUM_FRAMES: f32 = 30.0;
/// Fast-history cap: the short EMA the clamp box and anti-lag read. Small by
/// design — it must track lighting changes the slow history lags on.
pub const FAST_FRAMES: f32 = 4.0;
/// Fast-history clamp box half-width in sigmas.
pub const CLAMP_SIGMA: f32 = 2.5;
/// Chroma lanes clamp at this fraction of the luma box (YCoCg pays off here:
/// chroma noise is perceptually cheaper to box tight than luma).
pub const CHROMA_BOX: f32 = 0.5;
/// Relative view-Z disocclusion tolerance (of the larger |z|), before the
/// grazing relaxation divides by n·v.
pub const Z_EPS: f32 = 0.02;
/// Diffuse normal-agreement threshold: cos 25 deg.
pub const N_COS_DIFF: f32 = 0.906_307_8;
/// Specular normal agreement at roughness 0 (mirror history is only valid on
/// a near-identical normal); relaxes to N_COS_DIFF at roughness 1.
pub const N_COS_SPEC_SMOOTH: f32 = 0.998;
/// Specular parallax sensitivity: frames of history a unit of per-frame
/// parallax (radians at the hit point, roughly) costs a smooth surface.
pub const SPEC_PARALLAX_K: f32 = 30.0;
/// Max spatial blur radius in pixels (both passes; pass 3 runs 1.7x).
pub const R_MAX: f32 = 30.0;
/// Radius floor as a fraction of R at full accumulation — converged pixels
/// keep a small maintenance blur that integrates the jittered Poisson
/// feedback instead of freezing its last sample pattern.
pub const S_MIN: f32 = 0.15;
/// History-fix window: below this accumulated-frame count the blur radius is
/// boosted toward R_FIX and the bilateral thresholds relax — a wide relaxed
/// blur on disoccluded pixels IS the history fix (no mip chain at 1080p).
pub const N_FIX: f32 = 4.0;
pub const R_FIX: f32 = 30.0;
/// GGX dominant-lobe spread proxy: tan(theta_lobe) ≈ LOBE_K * alpha, alpha =
/// roughness^2. Crude on purpose — it only sizes a blur kernel.
pub const LOBE_K: f32 = 0.75;
/// Plane-distance sensitivity: the bilateral plane test's e-folding distance
/// as a fraction of the pixel frustum size at the center's depth.
pub const PLANE_SENS: f32 = 2.0;
/// Diffuse normal-weight sharpness (pow exponent); specular sharpens toward
/// SPEC_N_POW_SMOOTH as roughness falls.
pub const N_POW_DIFF: f32 = 8.0;
pub const N_POW_SPEC_SMOOTH: f32 = 128.0;
/// Firefly pre-clamp: input luma may exceed the 3x3 neighborhood mean by at
/// most this factor before being compressed (soft, energy-aware).
pub const FIREFLY_K: f32 = 8.0;
/// Diffuse/specular blur radius gains (world hit-dist -> pixel radius).
pub const C_DIFF: f32 = 0.5;
pub const C_SPEC: f32 = 2.0;
/// The out-of-range band shared with the bridge: view_z >= 0.999 * far is
/// sky/never-denoised (nrd_bridge.hlsl's LOCKSTEP predicate — FRD keeps it
/// verbatim so cs_nrd_pack/cs_nrd_out serve either engine unchanged).
pub const RANGE_K: f32 = 0.999;
/// The meta plane's frame-count ceiling: n travels as n/63 in R8G8_UNORM
/// (frd_gpu's hist META plane; the /63.0 and *63.0 literals in
/// frd_temporal.hlsl / frd_blur.hlsl are its shader twins), so a
/// --frd-max-accum-frames past this truncates AT THE WIRE whatever the CB
/// says — the parse notes the clamp and frd_gpu's cb() applies it (the
/// MAX_FIREFLIES loud-clamp shape).
pub const META_N_MAX: f32 = 63.0;

// ---------------------------------------------------------------------------
// Tuning — the --frd-* levers (the nrd::ReblurTuning / fsr-tune shape):
// every field None by default = the compiled constants above, so a flagless
// session is bit-identical to one that never parsed a lever.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default, Debug)]
pub struct FrdTuning {
    pub max_accum_frames: Option<u32>,
    pub fast_frames: Option<u32>,
    /// 0 = the stabilization sub-step is skipped outright (the
    /// --nrd-max-stabilized-frames 0 lever's analog).
    pub max_stab_frames: Option<u32>,
    pub blur_radius: Option<f32>,
    pub clamp_sigma: Option<f32>,
    pub anti_firefly: Option<bool>,
    /// Force fp32 shader arms even where native fp16 exists (A/B lever).
    pub no_fp16: bool,
}

impl FrdTuning {
    pub fn any(&self) -> bool {
        self.max_accum_frames.is_some()
            || self.fast_frames.is_some()
            || self.max_stab_frames.is_some()
            || self.blur_radius.is_some()
            || self.clamp_sigma.is_some()
            || self.anti_firefly.is_some()
            || self.no_fp16
    }
}

static TUNING: std::sync::OnceLock<FrdTuning> = std::sync::OnceLock::new();

/// One writer: main's lever block (the nrd::set_tuning discipline).
pub fn set_tuning(t: FrdTuning) {
    let _ = TUNING.set(t);
}

pub fn tuning() -> FrdTuning {
    TUNING.get().copied().unwrap_or_default()
}

// ---------------------------------------------------------------------------
// oracle — CPU twins of the FRD shader math. Every function here mirrors a
// formula in frd_common.hlsli term for term; F0 pins the Rust side, and the
// GPU gates compare shader outputs against these on readback inputs (the
// N2 pack-vs-oracle shape). The WIRE helpers (YCoCg, the enc-2 normal pack,
// hit-dist normalization) stay in nrd::oracle while both engines share the
// bridge — re-exported here so FRD call sites read `frd::oracle::*` and the
// eventual move is mechanical.
// ---------------------------------------------------------------------------

pub mod oracle {
    use super::*;

    #[allow(unused_imports)] // consumed by the phase-B/C shader oracles
    pub use crate::nrd::oracle::{
        hit_dist_normalization, linear_to_ycocg, norm_hit_dist, sanitize_radiance,
        ycocg_to_linear,
    };

    // -- Reprojection ------------------------------------------------------

    /// The wire MV convention (nrd_bridge.hlsl's pack): mv.xy = prev − cur in
    /// PIXELS, so the previous frame's sample position is p + mv.
    pub fn reproject_px(p: [f32; 2], mv: [f32; 2]) -> [f32; 2] {
        [p[0] + mv[0], p[1] + mv[1]]
    }

    /// The 2.5D lane: mv.z = prev_z − cur_z of the SAME hit point (the
    /// tracer computed both), so the expected previous view-Z is exact — no
    /// plane reconstruction, which is strictly stronger than what SVGF has.
    pub fn expected_prev_z(z: f32, mv_z: f32) -> f32 {
        z + mv_z
    }

    // -- Disocclusion ------------------------------------------------------

    /// Relative view-Z test between the stored previous depth and the exact
    /// expectation, relaxed at grazing angles (a surface seen edge-on sweeps
    /// depth fast under 1 px of reprojection error; 1/max(n·v, 0.1) widens
    /// the tolerance up to 10x there). Symmetric in the two depths so a
    /// half-pixel of bilinear footprint can't flip the verdict by which side
    /// the max lands on.
    pub fn disocclusion_z_valid(z_expect: f32, z_prev: f32, n_dot_v: f32) -> bool {
        let scale = z_expect.abs().max(z_prev.abs()).max(1e-6);
        let rel = (z_prev - z_expect).abs() / scale;
        rel < Z_EPS / n_dot_v.max(0.1)
    }

    /// Normal-agreement threshold: diffuse history survives ~25 deg of
    /// normal delta; specular tightens toward mirror-exact as roughness
    /// falls (a mirror's history is only valid where the reflected world is
    /// the same, which the normal proxies).
    pub fn normal_cos_threshold(roughness: f32, spec: bool) -> f32 {
        if spec {
            let r = roughness.clamp(0.0, 1.0);
            N_COS_SPEC_SMOOTH + (N_COS_DIFF - N_COS_SPEC_SMOOTH) * r
        } else {
            N_COS_DIFF
        }
    }

    // -- Accumulation ------------------------------------------------------

    /// EMA blend weight from the accumulated-frame count: α = 1/(1+n). With
    /// n advancing by 1 per accepted frame this IS the running mean (the
    /// convergence pin in F0), and capping n turns it into an EMA with a
    /// ~1/(1+n_max) steady-state blend.
    pub fn accum_alpha(n: f32) -> f32 {
        1.0 / (1.0 + n.max(0.0))
    }

    /// Advance the frame count through a reprojection with confidence c in
    /// [0,1] (the summed validity of the bilinear feet): a low-confidence
    /// fetch decays the history's effective age before counting this frame.
    pub fn advance_frames(n: f32, confidence: f32, n_max: f32) -> f32 {
        (n * confidence.clamp(0.0, 1.0) + 1.0).min(n_max)
    }

    /// The specular history cap under camera parallax: a smooth surface's
    /// surface-motion reprojection is WRONG for its reflection content (the
    /// virtual image moves differently), so per-frame parallax p (≈ the
    /// angle the camera step subtends at the hit point) shortens the usable
    /// history; roughness restores it (a rough reflection is diffuse-like).
    /// v1.5's virtual-motion reprojection supersedes this cap for mirrors.
    pub fn spec_max_frames(n_max: f32, parallax: f32, roughness: f32) -> f32 {
        let r = roughness.clamp(0.0, 1.0);
        let smooth = 1.0 / (1.0 + SPEC_PARALLAX_K * parallax.max(0.0) * (1.0 - r) * (1.0 - r));
        n_max * (smooth + (1.0 - smooth) * r)
    }

    /// Welford-form luma variance update: m2' = lerp(m2, (l − f_new)(l −
    /// f_old), α) where f_old/f_new are the fast history before/after this
    /// frame's blend. NEVER the E[l²]−E[l]² form — that difference cancels
    /// catastrophically in fp16 (the plan's mandated shape); read the
    /// variance as max(m2, 0).
    pub fn welford_update(m2: f32, l: f32, f_old: f32, f_new: f32, alpha: f32) -> f32 {
        m2 + ((l - f_new) * (l - f_old) - m2) * alpha.clamp(0.0, 1.0)
    }

    // -- Fast-history clamp ------------------------------------------------

    /// Clamp a YCoCg slow-history value to the fast history's ±kσ box, the
    /// chroma lanes boxed at CHROMA_BOX of the luma width. In-box values
    /// return BITWISE (idempotence is an F0 pin — the clamp must be inert
    /// on a converged signal).
    pub fn clamp_ycocg(s: [f32; 3], fast: [f32; 3], sigma_l: f32, k: f32) -> [f32; 3] {
        let bl = k * sigma_l.max(0.0);
        let box_w = [bl, bl * CHROMA_BOX, bl * CHROMA_BOX];
        let mut out = [0.0f32; 3];
        for i in 0..3 {
            out[i] = s[i].clamp(fast[i] - box_w[i], fast[i] + box_w[i]);
        }
        out
    }

    /// Anti-lag: when the clamp had to move the slow history by e sigmas
    /// past the box (e = excess/box), cut the accumulated-frame count so the
    /// NEXT frame blends faster — the recurrent loop's lag brake. e <= 1
    /// (in box) returns n unchanged.
    pub fn antilag_frames(n: f32, e: f32) -> f32 {
        if e <= 1.0 {
            n
        } else {
            n / (1.0 + (e - 1.0))
        }
    }

    // -- Firefly pre-clamp -------------------------------------------------

    /// Soft input clamp against the 3x3 neighborhood mean: luma may exceed
    /// FIREFLY_K * mean; past that it compresses (never to zero — an
    /// outlier is attenuated, not deleted, so energy loss is bounded).
    /// Returns the scale to apply to the sample (1.0 = untouched).
    pub fn firefly_scale(luma: f32, mean3x3: f32) -> f32 {
        let cap = FIREFLY_K * mean3x3.max(1e-6);
        if luma <= cap || luma <= 0.0 {
            1.0
        } else {
            cap / luma
        }
    }

    // -- Bilateral weights -------------------------------------------------

    /// Plane-distance weight: |n_p · (X_j − X_p)| against the pixel-frustum
    /// size at the center's depth (2z·tan(fov/2)/rh per pixel, folded into
    /// `frustum_px` by the caller) — the geometry test that kills
    /// over-the-silhouette leaks a pure z-compare admits at grazing angles.
    pub fn plane_weight(plane_dist_abs: f32, frustum_px: f32) -> f32 {
        (-plane_dist_abs / (PLANE_SENS * frustum_px.max(1e-6))).exp()
    }

    /// Normal-agreement exponent, hoisted per pixel (constant across a
    /// pixel's taps): roughness-driven for specular — sharp lobes only
    /// share with near-identical normals.
    pub fn normal_pow(roughness: f32, spec: bool) -> f32 {
        if spec {
            let r = roughness.clamp(0.0, 1.0);
            N_POW_SPEC_SMOOTH + (N_POW_DIFF - N_POW_SPEC_SMOOTH) * r
        } else {
            N_POW_DIFF
        }
    }

    /// log2(e) and the per-index Gaussian coefficient 2·log2(e)/TAPS — the
    /// log2-domain constants of the FUSED tap weight below (HLSL literal
    /// twins in frd_common.hlsli, change together).
    pub const LOG2E: f32 = 1.442_695;
    pub const TAP_G2: f32 = 0.360_673_76;

    pub fn inv_z_scale(z_center: f32) -> f32 {
        LOG2E / (0.05 * z_center.abs().max(1e-4))
    }

    pub fn inv_h_scale(h_center: f32) -> f32 {
        LOG2E / (h_center + 1e-3)
    }

    /// The fused tap weight, log2 domain (the B70 EM-pipe diet): what used
    /// to be four factors — Gaussian exp(−2·r_i²) over the disk, the
    /// relative view-Z test exp(−|Δz|/(0.05·|z|)), the normal agreement
    /// saturate(n·n)^pow, and the spec hit-dist similarity
    /// exp(−|Δh|/(h+1e-3)) that preserves reflection contact hardening —
    /// as ONE exponent sum under ONE exp2. The Gaussian needs no distance:
    /// the Vogel radius r_i² = (i+0.5)/TAPS is a per-INDEX constant.
    /// Diffuse passes inv_h = 0; ndn ≤ 0 drives log2 → −inf → exp2 →
    /// exactly 0, the old pow(saturate, pw) limit.
    pub fn tap_exp2(
        i: usize,
        dz_abs: f32,
        inv_z: f32,
        ndn: f32,
        pw: f32,
        dh_abs: f32,
        inv_h: f32,
    ) -> f32 {
        -((i as f32) + 0.5) * TAP_G2 - dz_abs * inv_z + pw * ndn.clamp(0.0, 1.0).log2()
            - dh_abs * inv_h
    }

    // -- Blur radius -------------------------------------------------------

    /// Diffuse blur radius in pixels: the world-space hit distance projected
    /// to the screen (h/z, scaled by proj = rh/(2 tan(fov/2))), floored at
    /// 1 px (there is always SOME spatial support at 1 spp) and capped at
    /// `r_max` — a PARAMETER, not the R_MAX const, because the shader twin
    /// reads it off the CB (the --frd-blur-radius lever) and a const-clamped
    /// oracle would diverge from the shader exactly when the lever is set.
    pub fn radius_diffuse_px(hit_dist_w: f32, view_z: f32, proj: f32, r_max: f32) -> f32 {
        (C_DIFF * hit_dist_w / view_z.max(1e-6) * proj).clamp(1.0, r_max)
    }

    /// Specular blur radius: the GGX lobe's footprint at the reflected
    /// distance, projected. roughness → 0 ⇒ radius → 0 (mirrors get no
    /// spatial blur; temporal + clamp carry them). `r_max` = the CB lever,
    /// like the diffuse twin.
    pub fn radius_spec_px(hit_dist_w: f32, view_z: f32, roughness: f32, proj: f32, r_max: f32) -> f32 {
        let a = roughness.clamp(0.0, 1.0);
        let tan_lobe = LOBE_K * a * a;
        (C_SPEC * hit_dist_w * tan_lobe / (view_z + hit_dist_w).max(1e-6) * proj)
            .clamp(0.0, r_max)
    }

    /// Accumulation scaling: early frames get the full radius, converged
    /// pixels keep the S_MIN maintenance fraction (the +1 px floor is the
    /// caller's — radius formulas already floor where they must).
    pub fn radius_accum_scale(n: f32) -> f32 {
        (1.0 / (1.0 + n.max(0.0))).max(S_MIN)
    }

    /// The folded history fix: under N_FIX accumulated frames the effective
    /// radius is boosted toward R_FIX — a wide relaxed blur on disoccluded
    /// pixels replaces the published separate history-fix pass.
    pub fn history_fix_radius(r_eff: f32, n: f32) -> f32 {
        if n < N_FIX {
            r_eff.max(R_FIX * (1.0 - n / N_FIX))
        } else {
            r_eff
        }
    }

    // -- The tap disk ------------------------------------------------------

    /// 8 Vogel-spiral (golden-angle) disk offsets in the unit disk — OUR
    /// table (generated, not authored): r_i = sqrt((i+0.5)/8), θ_i = i·φ_g.
    /// `vogel_disk` is the analytic generator and stays the F0 reference;
    /// the SHADER runs the literal table + rotate below (one sincos per
    /// pixel per pass instead of one per tap), and the F0 pins hold the two
    /// forms together.
    pub const GOLDEN_ANGLE: f32 = 2.399_963_2; // π(3 − √5)
    pub const TAPS: usize = 8;

    pub fn vogel_disk(i: usize, n: usize, rot: f32) -> [f32; 2] {
        let r = (((i as f32) + 0.5) / n as f32).sqrt();
        let th = (i as f32) * GOLDEN_ANGLE + rot;
        [r * th.cos(), r * th.sin()]
    }

    /// The unrotated disk as literals (f64-generated from the formula
    /// above; HLSL twin FRD_VOGEL8), rotated by the per-pixel hash angle at
    /// 2 fma per tap.
    pub const VOGEL8: [[f32; 2]; 8] = [
        [0.25, 0.0],
        [-0.319290081, 0.292495887],
        [0.0488724328, -0.556876544],
        [0.402444525, 0.524917521],
        [-0.73853513, -0.130636375],
        [0.699604866, -0.445031495],
        [-0.234004003, 0.870483846],
        [-0.446271487, -0.859268154],
    ];

    pub fn vogel_rot(i: usize, cr: f32, sr: f32) -> [f32; 2] {
        let u = VOGEL8[i];
        [u[0] * cr - u[1] * sr, u[0] * sr + u[1] * cr]
    }

    // -- F0 ----------------------------------------------------------------

    /// The F0 gate (run by --check, DLL-free): closed-form anchors +
    /// monotonicity + the convergence identity for every formula above, with
    /// teeth (each pin fails on a plausible wrong form, not just on noise).
    pub fn self_test() -> Result<(), String> {
        // Reprojection: the MV convention pin — mv IS the fetch offset.
        let q = reproject_px([100.0, 50.0], [-3.5, 2.25]);
        if q != [96.5, 52.25] {
            return Err(format!("frd: reproject_px convention ({q:?})"));
        }
        if expected_prev_z(10.0, 0.75) != 10.75 {
            return Err("frd: expected_prev_z is z + mv.z".into());
        }

        // Disocclusion: exact match accepts, a 2x depth jump rejects, and
        // the grazing relaxation is MONOTONE — a tolerance that tightened at
        // grazing would reject every silhouette pixel (the boiling shape).
        if !disocclusion_z_valid(10.0, 10.0, 1.0) {
            return Err("frd: disocclusion rejects an exact match".into());
        }
        if disocclusion_z_valid(10.0, 20.0, 1.0) {
            return Err("frd: disocclusion accepts a 2x depth jump".into());
        }
        // At head-on n·v=1 a 1.9% delta passes and 2.1% fails (Z_EPS=2%);
        // at grazing n·v=0.1 the same 2.1% delta must PASS (10x relaxed).
        if !disocclusion_z_valid(10.0, 10.19, 1.0) || disocclusion_z_valid(10.0, 10.21, 1.0) {
            return Err("frd: disocclusion Z_EPS anchor".into());
        }
        if !disocclusion_z_valid(10.0, 10.21, 0.1) {
            return Err("frd: disocclusion grazing relaxation".into());
        }

        // Normal thresholds: spec at r=0 is the tightest, r=1 equals the
        // diffuse threshold, monotone between.
        let (t0, t5, t1) = (
            normal_cos_threshold(0.0, true),
            normal_cos_threshold(0.5, true),
            normal_cos_threshold(1.0, true),
        );
        if !(t0 > t5 && t5 > t1) || (t1 - N_COS_DIFF).abs() > 1e-6 || t0 != N_COS_SPEC_SMOOTH {
            return Err(format!("frd: normal_cos_threshold ladder ({t0} {t5} {t1})"));
        }

        // Accumulation: α = 1/(1+n) EMA with n advancing by 1 IS the running
        // mean — feed 1..=N and the estimate must equal (N+1)/2 exactly-ish.
        // This is the identity that makes still frames converge statistically
        // (the plan's L1 rung) and it dies under any off-by-one in n.
        let mut est = 0.0f32;
        let mut n = 0.0f32;
        for k in 1..=32 {
            let a = accum_alpha(n);
            est += (k as f32 - est) * a;
            n = advance_frames(n, 1.0, 1e9);
        }
        if (est - 16.5).abs() > 1e-3 {
            return Err(format!("frd: accumulation is not the running mean (est {est})"));
        }
        // Confidence decay: c=0 resets the age (next α = 1/2 after one
        // advance), and the cap holds.
        if advance_frames(30.0, 0.0, 30.0) != 1.0 {
            return Err("frd: zero-confidence must reset history age".into());
        }
        if advance_frames(30.0, 1.0, 30.0) != 30.0 {
            return Err("frd: frame-count cap".into());
        }

        // Specular parallax cap: parallax 0 ⇒ full history; monotone
        // decreasing in parallax on smooth surfaces; roughness 1 ⇒ immune.
        if (spec_max_frames(30.0, 0.0, 0.0) - 30.0).abs() > 1e-4 {
            return Err("frd: spec cap at zero parallax".into());
        }
        let (p1, p2) = (spec_max_frames(30.0, 0.05, 0.0), spec_max_frames(30.0, 0.2, 0.0));
        if !(p1 > p2 && p2 < 10.0) {
            return Err(format!("frd: spec parallax cap not biting ({p1} {p2})"));
        }
        if (spec_max_frames(30.0, 10.0, 1.0) - 30.0).abs() > 1e-4 {
            return Err("frd: rough surfaces must keep full history".into());
        }

        // Welford variance: a CONSTANT signal drives m2 → 0 (feed it with
        // the fast history converged onto the constant), and a ±1 square
        // wave around 0 converges to ~1. The form matters — this is the pin
        // that keeps the fp16-catastrophic E[l²]−E[l]² out.
        let mut m2 = 5.0f32;
        for _ in 0..64 {
            m2 = welford_update(m2, 2.0, 2.0, 2.0, 0.2);
        }
        if m2.abs() > 1e-4 {
            return Err(format!("frd: constant-signal variance must vanish ({m2})"));
        }
        let (mut m2, mut f) = (0.0f32, 0.0f32);
        for k in 0..256 {
            let l = if k % 2 == 0 { 1.0 } else { -1.0 };
            let f_old = f;
            f += (l - f) * 0.2;
            m2 = welford_update(m2, l, f_old, f, 0.2);
        }
        if !(0.5..=1.5).contains(&m2) {
            return Err(format!("frd: square-wave variance off ({m2})"));
        }

        // Clamp: bitwise idempotent in-box (the converged-signal contract),
        // clamps outside, chroma box tighter than luma.
        let f = [1.0f32, 0.25, -0.25];
        let inbox = [1.1f32, 0.26, -0.26];
        if clamp_ycocg(inbox, f, 0.1, CLAMP_SIGMA) != inbox {
            return Err("frd: clamp not inert in-box".into());
        }
        let big = clamp_ycocg([9.0, 9.0, 9.0], f, 0.1, CLAMP_SIGMA);
        let bl = CLAMP_SIGMA * 0.1;
        if (big[0] - (1.0 + bl)).abs() > 1e-6 || (big[1] - (0.25 + bl * CHROMA_BOX)).abs() > 1e-6 {
            return Err(format!("frd: clamp box widths ({big:?})"));
        }
        // Anti-lag: in-box e leaves n alone; e=2 halves it.
        if antilag_frames(20.0, 0.5) != 20.0 || (antilag_frames(20.0, 2.0) - 10.0).abs() > 1e-5 {
            return Err("frd: antilag_frames anchors".into());
        }

        // Firefly: under the cap the scale is EXACTLY 1.0 (the common path
        // must be bitwise untouched); an outlier compresses to the cap.
        if firefly_scale(1.0, 1.0) != 1.0 {
            return Err("frd: firefly scale must be inert under the cap".into());
        }
        let s = firefly_scale(100.0, 1.0);
        if (s * 100.0 - FIREFLY_K).abs() > 1e-4 {
            return Err(format!("frd: firefly cap ({s})"));
        }

        // Bilateral weights: 1.0 at zero distance / perfect agreement,
        // strictly decreasing, and the spec normal weight sharper than
        // diffuse at low roughness.
        if (plane_weight(0.0, 1.0) - 1.0).abs() > 1e-6
            || plane_weight(1.0, 1.0) >= plane_weight(0.5, 1.0)
        {
            return Err("frd: plane_weight shape".into());
        }
        if normal_pow(0.0, true) <= normal_pow(0.0, false)
            || (normal_pow(1.0, true) - N_POW_DIFF).abs() > 1e-4
        {
            return Err("frd: spec normal exponent ladder".into());
        }
        // The fused log2-domain tap weight: exp2(tap_exp2) must reproduce
        // the four-factor product it replaced (the pre-diet form, kept HERE
        // as the reference) to 1e-5 relative — the equivalence proof — with
        // the exact-zero limit at ndn = 0, the log2-constant derivations
        // pinned, and TEETH: a wrong-sign z term must blow the bound.
        if (LOG2E - std::f32::consts::LOG2_E).abs() > 1e-6
            || (TAP_G2 - 2.0 * LOG2E / TAPS as f32).abs() > 1e-6
        {
            return Err("frd: log2-domain constants drifted from their derivations".into());
        }
        let old_form = |i: usize,
                        dz: f32,
                        z: f32,
                        ndn: f32,
                        rough: f32,
                        spec: bool,
                        ht: f32,
                        hc: f32| {
            let d2 = ((i as f32) + 0.5) / TAPS as f32;
            let wg = (-2.0 * d2).exp();
            let wz = (-dz / (0.05 * z.abs().max(1e-4))).exp();
            let wn = ndn.clamp(0.0, 1.0).powf(normal_pow(rough, spec));
            let wh = if spec {
                (-(ht - hc).abs() / (hc + 1e-3)).exp()
            } else {
                1.0
            };
            wg * wz * wn * wh
        };
        let probes: [(usize, f32, f32, f32, f32, bool, f32, f32); 4] = [
            (0, 0.0, 10.0, 1.0, 0.5, false, 0.0, 0.0),
            (3, 0.2, 10.0, 0.95, 0.5, false, 0.0, 0.5),
            (7, 0.01, 0.5, 0.999, 0.05, true, 0.4, 0.5),
            (5, 1.0, 100.0, 0.8, 1.0, true, 2.0, 0.1),
        ];
        for &(i, dz, z, ndn, rough, spec, ht, hc) in &probes {
            let inv_h = if spec { inv_h_scale(hc) } else { 0.0 };
            let w = tap_exp2(
                i,
                dz,
                inv_z_scale(z),
                ndn,
                normal_pow(rough, spec),
                (ht - hc).abs(),
                inv_h,
            )
            .exp2();
            let r = old_form(i, dz, z, ndn, rough, spec, ht, hc);
            if (w - r).abs() > 1e-5 * r.max(1e-6) {
                return Err(format!("frd: fused tap weight diverges ({w} vs {r})"));
            }
        }
        if tap_exp2(2, 0.1, inv_z_scale(5.0), 0.0, N_POW_DIFF, 0.0, 0.0).exp2() != 0.0 {
            return Err("frd: fused weight must be exactly 0 at ndn = 0".into());
        }
        let (i, dz, z, ndn, rough, ht, hc) = (5usize, 1.0f32, 100.0f32, 0.8f32, 1.0f32, 2.0f32, 0.1f32);
        let bad = (-((i as f32) + 0.5) * TAP_G2 + dz * inv_z_scale(z)
            + normal_pow(rough, true) * ndn.log2()
            - (ht - hc).abs() * inv_h_scale(hc))
        .exp2();
        let r = old_form(i, dz, z, ndn, rough, true, ht, hc);
        if (bad - r).abs() <= 1e-5 * r.max(1e-6) {
            return Err("frd: fused-weight pin has no teeth".into());
        }

        // Radii: monotone in hit distance; diffuse floors at 1 px; spec
        // vanishes at roughness 0 and grows with roughness; the
        // accumulation scale floors at S_MIN; history fix boosts only under
        // N_FIX and never SHRINKS a radius.
        let (rd1, rd2) = (
            radius_diffuse_px(0.5, 10.0, 900.0, R_MAX),
            radius_diffuse_px(2.0, 10.0, 900.0, R_MAX),
        );
        if !(rd2 > rd1 && rd1 >= 1.0 && rd2 <= R_MAX) {
            return Err(format!("frd: diffuse radius ({rd1} {rd2})"));
        }
        if radius_diffuse_px(0.0, 10.0, 900.0, R_MAX) != 1.0 {
            return Err("frd: diffuse radius must floor at 1 px".into());
        }
        // The lever cap is a live parameter, not the const: a smaller r_max
        // must bind (the shader/oracle can't diverge under --frd-blur-radius).
        if radius_diffuse_px(2.0, 10.0, 900.0, 5.0) != 5.0 {
            return Err("frd: diffuse radius must respect the r_max lever".into());
        }
        if radius_spec_px(2.0, 10.0, 0.0, 900.0, R_MAX) != 0.0 {
            return Err("frd: mirror spec radius must be 0".into());
        }
        if radius_spec_px(2.0, 10.0, 0.8, 900.0, R_MAX)
            <= radius_spec_px(2.0, 10.0, 0.3, 900.0, R_MAX)
        {
            return Err("frd: spec radius must grow with roughness".into());
        }
        if radius_accum_scale(0.0) != 1.0 || radius_accum_scale(1e6) != S_MIN {
            return Err("frd: radius_accum_scale endpoints".into());
        }
        if history_fix_radius(2.0, 0.0) != R_FIX || history_fix_radius(2.0, N_FIX) != 2.0 {
            return Err("frd: history_fix_radius endpoints".into());
        }
        if history_fix_radius(50.0, 1.0) != 50.0 {
            return Err("frd: history fix must never shrink a radius".into());
        }

        // The Vogel disk: TAPS distinct offsets, all inside the unit disk,
        // reasonably spread (min pairwise distance — a degenerate generator
        // collapsing taps would pass an in-disk test alone), and rotation-
        // equivariant (rot only rotates, radii bitwise).
        let taps: Vec<[f32; 2]> = (0..TAPS).map(|i| vogel_disk(i, TAPS, 0.0)).collect();
        let mut min_d = f32::MAX;
        for i in 0..TAPS {
            let r = (taps[i][0] * taps[i][0] + taps[i][1] * taps[i][1]).sqrt();
            if r > 1.0 + 1e-6 {
                return Err(format!("frd: tap {i} outside the unit disk (r {r})"));
            }
            for j in 0..i {
                let (dx, dy) = (taps[i][0] - taps[j][0], taps[i][1] - taps[j][1]);
                min_d = min_d.min((dx * dx + dy * dy).sqrt());
            }
        }
        if min_d < 0.2 {
            return Err(format!("frd: tap disk degenerate (min pair dist {min_d})"));
        }
        for (i, t) in taps.iter().enumerate() {
            let rot = vogel_disk(i, TAPS, 1.234);
            let (r0, r1) = (
                (t[0] * t[0] + t[1] * t[1]).sqrt(),
                (rot[0] * rot[0] + rot[1] * rot[1]).sqrt(),
            );
            if (r0 - r1).abs() > 1e-6 {
                return Err("frd: rotation must preserve tap radii".into());
            }
        }
        // The literal table + rotate IS the analytic generator: VOGEL8 vs
        // vogel_disk at rot 0 (the f64-generation pin) and vogel_rot vs the
        // analytic form at an arbitrary angle (the trig-identity pin) —
        // what holds the shader's table to the formula above.
        let rot = 1.234f32;
        let (cr, sr) = (rot.cos(), rot.sin());
        for i in 0..TAPS {
            let a0 = vogel_disk(i, TAPS, 0.0);
            if (VOGEL8[i][0] - a0[0]).abs() > 1e-5 || (VOGEL8[i][1] - a0[1]).abs() > 1e-5 {
                return Err(format!("frd: VOGEL8[{i}] drifted from the generator"));
            }
            let ar = vogel_disk(i, TAPS, rot);
            let tr = vogel_rot(i, cr, sr);
            if (tr[0] - ar[0]).abs() > 1e-5 || (tr[1] - ar[1]).abs() > 1e-5 {
                return Err(format!("frd: vogel_rot[{i}] diverges from the analytic form"));
            }
        }

        // The shared-wire contract: FRD's re-exported packing math IS
        // nrd::oracle's (a drifted re-export would fork the bridge).
        let c = [0.3f32, 1.7, 0.05];
        if linear_to_ycocg(c) != crate::nrd::oracle::linear_to_ycocg(c) {
            return Err("frd: wire ycocg re-export drifted".into());
        }

        eprintln!(
            "frd self-test: reproject + disocclusion + accumulation + clamp + weights + \
             radii + vogel-disk OK"
        );
        Ok(())
    }
}
