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
/// The temporal-stabilization age cap — the DEFAULT since the post-QA flip
/// (2026-08-10, one commit after the mechanism: the bistro emissive QA
/// campaign measured stab 0.60 → 0.47 at zero pool-energy cost, the helmet
/// strafe lab improved on every metric, and the B70 frd region paid
/// +0.18 ms). --frd-max-stab-frames 0 is the bitwise pre-stab arm; 60
/// probed only 0.47 → 0.45 (the residual's floor), so 20 keeps the better
/// lag bound. Between FAST_FRAMES and MAX_ACCUM: a ~1/21 steady-state
/// blend of fresh output cuts the sparse-bright disk-rotation flicker
/// ~5× while the spatial-neighborhood clamp bounds lag/ghosting.
pub const MAX_STAB_FRAMES: f32 = 20.0;
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
/// The anti-lag excess denominator's RELATIVE floor: the clamp-box width is
/// floored at this fraction of the fast luma before dividing, because a
/// CONVERGED signal's box is exactly zero-width (constant input ⇒ m2 = 0)
/// while the blur pass's uniform-field output still carries a few ULPS of
/// tap-rotation-dependent fp residue — against an absolute 1e-6 floor that
/// residue reads as e ≈ 1 and mints junk sub-1 gains on perfectly converged
/// pixels (measured live by F5b: arm-order-dependent decay from a
/// bit-identical setup). A sub-1%-of-signal deviation is never a ghost.
pub const ANTILAG_E_FLOOR: f32 = 0.01;
/// Firefly pre-clamp: input luma may exceed the 8-NEIGHBOR ring mean
/// (center EXCLUDED — with the center in, a lone outlier's own contribution
/// dominates its cap and the clamp asymptotes to a fixed (1 − K/9) trim) by
/// at most this factor before being compressed (soft, energy-aware).
/// A fixed K can NEVER pass a sparse stochastic signal, only bias it down:
/// a signal with per-pixel hit probability p delivers its energy in samples
/// of luma ~L/p against a ring mean ~L, so unbiased integration needs
/// K >= 1/p — unbounded as p falls. That is exactly the folded RTGI
/// bounce's emissive transport (1-spp cosine hits on emitters), which is
/// why flags bit 5 (the GI fold, `girelax_enabled`) EXEMPTS the diffuse
/// lane instead of raising K. Specular keeps the clamp unconditionally —
/// the sun-glint outliers the v1.5.x campaigns fought are F0-demodulated
/// mirror chains, a magnitude class diffuse (albedo- and cosine-bounded)
/// cannot reach.
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
/// v1.5 specular virtual motion (the camera-translation half of the
/// bright-specular smear fix; the ngxfg_guides round-2 unfold, delta form).
/// VM_FAR_K: t_r at or past this fraction of far takes the rotation-only
/// sky limit — deliberately NOT exact far (the wire_cam_far f16 lesson: an
/// exact-far compare against a quantized hit-t never fires).
pub const VM_FAR_K: f32 = 0.99;
/// The roughness fade window: below LO the fetch is fully virtual, past HI
/// fully surface (a rough surface's "reflection" IS the surface — the
/// ngxfg ROUGH_LO/HI damping shape, FRD-local values).
pub const VM_ROUGH_LO: f32 = 0.25;
pub const VM_ROUGH_HI: f32 = 0.6;
/// Disocclusion slack per pixel of applied virtual offset: a grazing planar
/// mirror's reflector depth varies along the plane, so a far-displaced
/// virtual foot needs a proportionally wider relative-Z tolerance or every
/// mirror fetch history-restarts (noise, not trails — but noise).
pub const VM_Z_SLACK: f32 = 0.05;
/// v1.5.5 — the MAX-SPEED containment pair (the B70 1-spp max-strafe
/// blotch, diagnosed live by the AI QA Lab's lever sweep: at 300-800 px of
/// applied virtual offset the uncapped `1 + VM_Z_SLACK·|appl|` slack
/// widened the disocclusion z-test 16-40× — vacuous — so the displaced
/// fetch ACCEPTED bright glint history anywhere on the dome and the
/// recurrence compounded it into a dome-wide blob; FR_FRD_VMOTION=off
/// measured clean at the same speed, so the surface fetch is the proven
/// landing zone). VM_SLACK_CAP bounds the widening (slack ≤ 1 +
/// VM_Z_SLACK·CAP = 4 — the grazing-planar-mirror rationale only needs
/// moderate offsets; below the cap the old expression holds BITWISE), and
/// VM_APPL_UNC_K charges the applied offset itself into the history cap
/// (fetch error is MULTIPLICATIVE in the offset — a relative model error
/// δ produces δ·|appl| px per frame — so trust must shorten with reach:
/// the third installment of the v2 confidence term). Both arms only ever
/// TIGHTEN.
pub const VM_SLACK_CAP: f32 = 60.0;
pub const VM_APPL_UNC_K: f32 = 0.1;
/// The magnitude fade's window, in FRACTIONS of render height (res-
/// invariant): below LO·rh the applied virtual delta rides a multiply-by-
/// exactly-1.0 (the validated normal-speed regime, bitwise); past HI·rh
/// the fetch collapses onto the SURFACE fetch (measured clean at max
/// strafe via FR_FRD_VMOTION=off, while the uncontained fetch painted the
/// dome-wide blotch) — the FG round-4 "better a shimmer than a torn warp"
/// trade, applied to the fetch position.
pub const VM_FADE_LO: f32 = 0.08;
pub const VM_FADE_HI: f32 = 0.2;
/// v1.5.5(c) — the specular SMEAR BUDGET, in fractions of render height:
/// the parallax cap is a RATE limit (frames per radian) while the visible
/// artifact is the PRODUCT — history frames × per-frame glint screen
/// motion = the smear's length in pixels — so at extreme per-frame motion
/// even a "short" 5-8-frame history paints a dome-wide swath (the B70
/// max-strafe blotch: ~0.18·rh of glint motion per frame, and on a smooth
/// dome every fetch position passes the normal/z tests, so only the cap
/// bounds the damage). n is therefore ALSO capped by BUDGET·rh divided by
/// the per-frame motion in pixels: ~2 frames at max strafe (restart-class,
/// the measured-clean landing), ~4 at the F7 gate's 0.095·rh regime (its
/// history-dependent pins keep teeth), and inert below ~0.04·rh/frame
/// where the rate cap is already the binding one.
pub const SPEC_PX_BUDGET: f32 = 0.38;
/// v1.5.5(d) — the luma-ratio tap guard (the firefly ring clamp's SPATIAL
/// sibling, v1.5.2's documented follow-up): a blur tap brighter than
/// TAP_LUMA_K× the center contributes at most K× the center's luma — the
/// smear budget keeps history young during sustained fast motion, the
/// history-fix radius stays wide, and an 800×-mean glint tap averaged
/// into dim neighbors was the soft dome-wide bloom. Within-K taps ride a
/// multiply-by-exactly-1.0 (ordinary fields bitwise).
pub const TAP_LUMA_K: f32 = 4.0;
pub const TAP_LUMA_EPS: f32 = 1e-3;
/// Squared px dead-zone under which the virtual delta snaps to EXACTLY
/// (0,0): a parked camera's surface and virtual projections agree only to
/// fp (both points sit on one ray through the unmoved origin — the ngxfg
/// static-camera anchor measures that residue under 2e-2 px), and it would
/// otherwise defeat the integer-aligned-foot fast path and the parked
/// bit-stability. 0.05 px is far below any visible virtual motion.
pub const VM_DEADZONE2: f32 = 2.5e-3;
/// The curvature estimator's dead-zone, in |Δn|-per-2px-baseline units
/// (central difference of DECODED wire normals over p±1): one LSB of the
/// 10-bit L1-oct encode moves the decoded normal 1.4–2.4e-3, so a 2-LSB
/// central diff (~3–5e-3) is quantization, not geometry. SUBTRACTIVE
/// (dn_eff = max(dn − DZ, 0)) so κ — and therefore the fetch position —
/// stays continuous through the boundary (a hard threshold steps t_v and
/// flickers on the quantization staircase). Deliberately not
/// worst-case-proof: a false-positive κ only reverts toward the surface
/// fetch (the pre-v1.5 status quo), a false-negative streaks — the
/// asymmetry makes the tighter value right. A truly constant normal field
/// (still water, F7's mirror) is ONE encoded word ⇒ Δn ≡ 0 bitwise ⇒ κ = 0
/// with no dead-zone needed at all. Known-accept: an extreme close-up dome
/// (~1000 px of span for ~140° of swing) fades toward the dead-zone —
/// PARTIALLY mitigated by cam_step/z being large at small z (per-FRAME, so
/// the mitigation weakens as fps rises — the v1.5.3 finding below is the
/// real fix for the resolution half of that fade).
pub const VM_DN_DZ: f32 = 5e-3;
/// v1.5.3: the κ baselines' RESOLUTION anchor. The estimator's central
/// differences run over ±s/±2s TEXELS while its dead-zone is absolute
/// per-sample oct-quantization noise — so at a finer pixel pitch the same
/// dome yields HALF the per-baseline |Δn| against the same DZ, and κ
/// under-reads in proportion to render height: the fetch slides toward the
/// flat/sky arm and the v1.5.1 trailing streak reopens at exactly the
/// resolutions users play. MEASURED (2026-08-10, the AI QA Lab's live
/// helmet strafe, deflection 0.06): 1080p glint FWHM 47 → 141 px
/// (velocity-proportional — the user's own report) while the 540p lab read
/// the SAME world pass clean, and live at --lock-res 0.5 collapsed it to
/// 73 px; FR_FRD_VMOTION=off (pure surface fetch) beat the shipping vm arm
/// at 1080p — exactly what a huge-κ close-up dome predicts when κ
/// under-reads. s = max(1, round(rh/540)): 540p (the regime v1.5.1/.2 were
/// validated in) and every gate res (533x400, 800x600) keep s = 1 BITWISE;
/// 1080p samples ±2/±4 texels — the same WORLD footprint 540p's ±1/±2
/// spans — 4K ±4/±8. Integer, session-fixed (FRD runs at the locked render
/// res), so the fetch never pops mid-session.
pub const VM_BASE_RH: f32 = 540.0;

/// The v1.5.3 baseline-scale rule (see VM_BASE_RH): integer-valued f32,
/// ≥ 1, = 1 for every rh below ~810.
pub fn vm_baseline_scale(rh: u32) -> f32 {
    (rh as f32 / VM_BASE_RH + 0.5).floor().max(1.0)
}

/// FR_FRD_VMSCALE — the κ-baseline-scale lever (read once): `off` pins the
/// pre-v1.5.3 fixed 2px/4px texel baselines at every resolution (the 1080p
/// strafe-smear repro arm), an integer 1..=8 forces the scale, unset = the
/// vm_baseline_scale rule. Loud on departure; an unrecognized value is
/// LOUD and takes the rule (the FR_NGXFG discipline — a silent no-op A/B
/// walk is the failure mode the levers exist to prevent).
pub fn vm_scale_for(rh: u32) -> f32 {
    static S: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    let lever = *S.get_or_init(|| {
        let Ok(v) = std::env::var("FR_FRD_VMSCALE") else {
            return None;
        };
        if v.eq_ignore_ascii_case("off") {
            eprintln!("frd: FR_FRD_VMSCALE=off — κ baselines pinned at 2px/4px (repro arm)");
            return Some(1.0);
        }
        match v
            .parse::<f32>()
            .ok()
            .filter(|s| s.is_finite() && (1.0..=8.0).contains(s) && s.fract() == 0.0)
        {
            Some(s) => {
                eprintln!("frd: FR_FRD_VMSCALE={v} — κ baseline scale forced");
                Some(s)
            }
            None => {
                eprintln!("frd: FR_FRD_VMSCALE='{v}' is not off|1..8 — using the rule");
                None
            }
        }
    });
    lever.unwrap_or_else(|| vm_baseline_scale(rh))
}

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

/// FR_FRD_SUNPAR — the light-motion parallax lever (read once): `off` (or
/// `0`) = the pre-fix repro arm (the parked-camera moving-sun smear returns
/// on demand), any non-negative float = a host-side gain on the term (the
/// feel-test knob — the multiply happens before the CB, no shader change).
/// Unset = 1.0. Loud on departure, and an unrecognized value is LOUD and
/// takes the default (the FR_NGXFG rule — a silent no-op A/B walk is the
/// failure mode the levers exist to prevent).
pub fn sunpar_gain() -> f32 {
    static G: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        let Ok(v) = std::env::var("FR_FRD_SUNPAR") else {
            return 1.0;
        };
        let g = if v.eq_ignore_ascii_case("off") {
            Some(0.0)
        } else {
            v.parse::<f32>().ok().filter(|g| g.is_finite() && *g >= 0.0)
        };
        match g {
            Some(g) => {
                eprintln!("frd: FR_FRD_SUNPAR={v} — light-motion parallax gain {g}");
                g
            }
            None => {
                eprintln!("frd: FR_FRD_SUNPAR='{v}' is not off|<gain> — using the default 1.0");
                1.0
            }
        }
    })
}

/// FR_FRD_CURV=off — disarm the v1.5.1 curvature term (the mirror
/// equation on the unfold distance): the vm block reverts to the flat-
/// mirror t_r — the helmet sun-streak repro arm. A CB flag, never a
/// recompile; loud on departure, any other value loud + default ON.
pub fn curv_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let Ok(v) = std::env::var("FR_FRD_CURV") else {
            return true;
        };
        if v.eq_ignore_ascii_case("off") {
            eprintln!("frd: FR_FRD_CURV=off — vm curvature disarmed (flat-mirror repro arm)");
            false
        } else {
            eprintln!("frd: FR_FRD_CURV='{v}' is not `off` — curvature stays armed");
            true
        }
    })
}

/// FR_FRD_VMOTION=off — disarm the v1.5 specular virtual-motion
/// reprojection (the camera-translation reflection-smear repro arm: the
/// spec history fetch falls back to the surface MV and the parallax cap to
/// the cam_step/z term — Stage A's exact behavior). A CB flag, never a
/// recompile; loud on departure, any other value loud + default ON.
pub fn vmotion_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let Ok(v) = std::env::var("FR_FRD_VMOTION") else {
            return true;
        };
        if v.eq_ignore_ascii_case("off") {
            eprintln!("frd: FR_FRD_VMOTION=off — specular virtual motion disarmed (repro arm)");
            false
        } else {
            eprintln!("frd: FR_FRD_VMOTION='{v}' is not `off` — virtual motion stays armed");
            true
        }
    })
}

/// FR_FRD_GIRELAX=off — disarm the GI-fold diffuse relax (flags bit 5): the
/// diffuse lane goes back under the firefly ring clamp and the frd_disk
/// luma-ratio tap guard even when the shade.hlsli FLAG_NRD_GI fold is live —
/// the emissive-pool-suppression repro arm (the 2026-08-10 QA campaign: the
/// clamp cost 90%+ of the bounce-emissive pool energy in XeSS+FRD sessions).
/// A CB flag, never a recompile; loud on departure, any other value loud +
/// default ON. Specular is untouched by the bit in either state.
pub fn girelax_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let Ok(v) = std::env::var("FR_FRD_GIRELAX") else {
            return true;
        };
        if v.eq_ignore_ascii_case("off") {
            eprintln!(
                "frd: FR_FRD_GIRELAX=off — diffuse firefly/tap-guard relax under the GI fold \
                 disarmed (emissive-suppression repro arm)"
            );
            false
        } else {
            eprintln!("frd: FR_FRD_GIRELAX='{v}' is not `off` — the GI relax stays armed");
            true
        }
    })
}

/// FR_FRD_ANTILAG=off — disarm the anti-lag brake (the widened-clamp-box
/// ghost repro arm: pass 3 still records gains, pass 1 skips the multiply —
/// a CB flag, never a recompile). Loud on departure; any other value is
/// loud + the default ON.
pub fn antilag_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let Ok(v) = std::env::var("FR_FRD_ANTILAG") else {
            return true;
        };
        if v.eq_ignore_ascii_case("off") {
            eprintln!("frd: FR_FRD_ANTILAG=off — anti-lag brake disarmed (repro arm)");
            false
        } else {
            eprintln!("frd: FR_FRD_ANTILAG='{v}' is not `off` — the brake stays armed");
            true
        }
    })
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

    /// The brake's WIRE form: g = 1/max(e, 1), so `antilag_frames(n, e) ==
    /// n · antilag_gain(e)` EXACTLY (n/(1+(e−1)) is n/e past the box, n·1
    /// inside — algebra, and an F0 identity pin). g ∈ (0, 1] is what rides
    /// the R8G8_UNORM antilag plane: pass 3 stores it from the clamp
    /// excess, pass 1 applies it through `antilag_apply` next frame —
    /// which is what makes a ghost's rejection STICK instead of re-fighting
    /// a 30-frame history every frame (frd_common.hlsli twin).
    pub fn antilag_gain(e: f32) -> f32 {
        1.0 / e.max(1.0)
    }

    /// The brake's EXCESS: the |pre-clamp − fast| luma distance in box
    /// widths, with the denominator floored RELATIVELY (ANTILAG_E_FLOOR ×
    /// |fast|) before the absolute backstop — a converged signal's box is
    /// exactly zero-width while blur fp residue is ulp-scale-of-RADIANCE,
    /// so an absolute floor misreads it as excess (the F5b find).
    pub fn antilag_excess(delta_abs: f32, box_w: f32, fast_luma: f32) -> f32 {
        delta_abs / box_w.max(ANTILAG_E_FLOOR * fast_luma.abs()).max(1e-6)
    }

    /// The brake's CONSUMER, floored at the history-fix window: the cut may
    /// accelerate recovery down to the FAST-history rate (n = N_FIX ⇒
    /// α = 1/5) but never INTO the restart window below it. Cutting past
    /// N_FIX buys nothing (α is already ≥ 1/5 there) and is actively
    /// destructive: n < N_FIX re-arms the wide history-fix blur, which
    /// wipes sharp features from the recurrent feedback, which re-opens the
    /// slow-vs-fast gap, which re-fires the clamp — a PERMANENT limit
    /// cycle on any converged sharp feature (a noise-free signal's m2 is
    /// exactly 0, so its clamp box has zero width and ANY gap reads as an
    /// infinite excess). F6 caught this live: an unfloored brake kept a
    /// synthetic glint from ever converging.
    pub fn antilag_apply(n: f32, g: f32) -> f32 {
        (n * g).max(n.min(N_FIX))
    }

    // -- Temporal stabilization (pass 3, --frd-max-stab-frames) ------------

    /// The stabilization blend weight for the CURRENT frame's result:
    /// α = 1/(1 + min(n_stab, max_stab)) — the running-mean alpha with the
    /// age capped at the lever, so a converged pixel blends at least
    /// 1/(1+max_stab) of fresh signal every frame (bounded lag).
    pub fn stab_alpha(n_stab: f32, max_stab: f32) -> f32 {
        1.0 / (1.0 + n_stab.max(0.0).min(max_stab))
    }

    /// The stabilization OUT blend (frd_common.hlsli's frd_stab_out, in
    /// LOCKSTEP): the reprojected previous OUT (`hist`) is clamped per
    /// channel to the current 3x3 SPATIAL neighborhood box of the blurred
    /// result — the TAA neighborhood clamp, sound on the POST-BLUR field
    /// because it is spatially smooth (the same clamp on the raw sparse
    /// input is exactly what eats emissive upstream). This is what makes
    /// lag/ghosting structurally impossible: on a step the neighborhood IS
    /// the new value, at a stale ghost position it is dark. (The first
    /// draft used the fast ±kσ TEMPORAL box; F8's teeth killed it — m2
    /// spikes exactly during transitions, so that box went wide when it
    /// most needed to bind.) The current result blends in at stab_alpha.
    /// max_stab <= 0 or n_stab <= 0 returns `cur` by BRANCH (never
    /// lerp-by-1, which is h + (c−h)·1 — fp, not identity): the
    /// bitwise-off contract.
    pub fn stab_out(
        cur: [f32; 3],
        hist: [f32; 3],
        bmin: [f32; 3],
        bmax: [f32; 3],
        n_stab: f32,
        max_stab: f32,
    ) -> [f32; 3] {
        if max_stab <= 0.0 || n_stab <= 0.0 {
            return cur;
        }
        let a = stab_alpha(n_stab, max_stab);
        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let h = hist[i].clamp(bmin[i], bmax[i]);
            out[i] = h + (cur[i] - h) * a;
        }
        out
    }

    // -- Firefly pre-clamp -------------------------------------------------

    /// Soft input clamp against the 8-neighbor RING mean (center excluded —
    /// see FIREFLY_K's doc for why center-in is toothless): luma may exceed
    /// FIREFLY_K * mean; past that it compresses to the cap (never to a
    /// hard zero — the 1e-6 floor keeps the scale positive, and a real
    /// multi-pixel glint's ring carries glint values, so it survives
    /// proportionally). Returns the scale to apply to the sample (1.0 =
    /// untouched — the common path is bitwise inert).
    pub fn firefly_scale(luma: f32, mean3x3: f32) -> f32 {
        let cap = FIREFLY_K * mean3x3.max(1e-6);
        if luma <= cap || luma <= 0.0 {
            1.0
        } else {
            cap / luma
        }
    }

    // -- Light-motion parallax ---------------------------------------------

    /// The moving-LIGHT specular parallax (the parked-camera moving-sun
    /// smear fix, 2026-08-09): with a static camera every reprojection —
    /// surface MV or any virtual-image unfold — is the identity, so a sun
    /// moved by a TOD scrub slides its glint across a mirror with NO motion
    /// vector able to describe it; the CAP is the mechanism. On a mirror
    /// the reflected image swings at exactly 2× the light's angular delta,
    /// and that angle is already the "radians at the hit" unit
    /// `spec_max_frames`' parallax input carries (cam_step/z is the angle
    /// the camera step subtends), so the two halves SUM and
    /// SPEC_PARALLAX_K applies unchanged. Bit-equal directions return
    /// EXACTLY 0.0 — the static-sun session is structurally inert (acos of
    /// a rounded-to-0.9999999 dot is not 0, so the early-out is the
    /// contract, not an optimization).
    pub fn light_parallax(sun_prev: [f32; 3], sun_cur: [f32; 3]) -> f32 {
        if sun_prev == sun_cur {
            return 0.0;
        }
        let d = sun_prev[0] * sun_cur[0] + sun_prev[1] * sun_cur[1] + sun_prev[2] * sun_cur[2];
        2.0 * d.clamp(-1.0, 1.0).acos()
    }

    // -- v1.5 specular virtual motion --------------------------------------

    /// The roughness fade on the virtual-motion coordinate: 1 below
    /// VM_ROUGH_LO (fully virtual — a mirror's content IS the reflection),
    /// 0 past VM_ROUGH_HI (fully surface), smoothstep between (the ngxfg
    /// damping shape).
    pub fn vm_rough_fade(rough: f32) -> f32 {
        let t = ((rough - VM_ROUGH_LO) / (VM_ROUGH_HI - VM_ROUGH_LO)).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    /// The v1.5.1 curvature correction — the PARAXIAL MIRROR EQUATION
    /// verbatim (convex mirror f = −R/2, object at t_r ⇒ virtual image at
    /// t_r/(1 + 2t_r/R) behind the surface — exact at ALL object
    /// distances, not just the limits): the unfold distance a CURVED
    /// mirror actually images at. κ = 0 is a BRANCH returning t_r
    /// VERBATIM (the flat path — still water, F7's constant-normal mirror
    /// — must stay bitwise, and a formula-only identity is one fmad
    /// contraction away from not holding); κ·t_r ≫ 1 ⇒ t_v → 1/(2κ) ≈
    /// R/2, the virtual image just behind the surface ⇒ the fetch lands
    /// at the SURFACE reprojection — the physically correct answer for a
    /// convex mirror's sun glint (the helmet streak: the flat-mirror sky
    /// arm pinned the fetch to the old screen position, re-painting every
    /// vacated pixel with its own stale bright history). Safety envelope:
    /// for κ ≥ 0, t_v ∈ (0, t_r] monotone in κ — any κ error
    /// INTERPOLATES between the two already-shipped behaviors (surface
    /// fetch ↔ flat unfold), never extrapolates past either.
    pub fn virtual_dist(t_r: f32, kappa: f32) -> f32 {
        if kappa <= 0.0 {
            t_r
        } else {
            t_r / (1.0 + 2.0 * kappa * t_r)
        }
    }

    /// One screen-space CONVEX-curvature estimate feeding virtual_dist:
    /// |Δn| central-differenced over a `baseline_px` world step (the
    /// DECODED wire normals), dead-zoned SUBTRACTIVELY against the 10-bit
    /// oct quantization (VM_DN_DZ — see its doc; the SAME absolute
    /// constant at every baseline, because quantization noise is
    /// per-sample while the signal grows with the baseline — which is
    /// exactly why the 4px read rescues close-up domes the 2px read
    /// dead-zones away). Magnitude-only — concave reads as convex, which
    /// only shortens t_v toward the surface fetch, the humble direction
    /// (a signed/concave carve-out is v2).
    pub fn vm_curvature_at(dn: f32, baseline_px: f32, view_z: f32, proj: f32) -> f32 {
        (dn - VM_DN_DZ).max(0.0) * proj / (baseline_px * view_z.max(1e-4))
    }

    /// The v1.5.2 ROBUST combine over four estimates (2px + 4px baselines
    /// × both axes) → (κ, κ_lo, κ_hi). Per-axis κ = MIN across the two
    /// SCALES: macro curvature is scale-consistent (on a clean field the
    /// dead-zone skew makes the 2px read the strict min, so κ equals the
    /// v1.5.1 value exactly), while normal-map bump noise decorrelates
    /// across scales — the min knocks down single-scale spikes, which is
    /// the LEADING-streak suppressor (an over-read κ makes the fetch ride
    /// the surface FASTER than the true virtual image and paints
    /// brightness ahead of the glint — the bidirectional-streak
    /// feel-test). κ = MAX across axes (a cylinder must track its curved
    /// axis). κ_lo/κ_hi bracket all four: their projected fetch spread is
    /// the estimator's own DISAGREEMENT, and the temporal pass charges
    /// the specular history cap with it — where the model is unsure, the
    /// honest answer is a short history (neither streak can accumulate),
    /// not a confident wrong fetch. The v2 confidence term, cheap form.
    /// `scale` is the v1.5.3 baseline scale (vm_baseline_scale — the dn
    /// args are measured over ±scale/±2·scale texels; 1.0 keeps the gate
    /// resolutions on the validated regime).
    ///
    /// THE v1.5.3(b) DE-BIASED COMBINE (2026-08-10 — the residual live
    /// strafe smear after the baseline-scale fix): the subtractive
    /// dead-zone biases every estimate DOWN, and on a clean field the two
    /// scales sit at exactly k4 = k2 + skew (skew = DZ·proj/(2·b1·z) —
    /// the self-test's closed form), so the old per-axis MIN always
    /// returned the SHORT baseline: the read the DZ has eaten the most of
    /// (measured ~37% of true κ on the live helmet dome ⇒ t_v overshoots
    /// ~2.7× ⇒ the fetch lags the content ⇒ the velocity-proportional
    /// trailing residue). The clean-field identity inverts the bias: true
    /// κ = k2 + 2·skew = k4 + skew, so min(k2 + 2·skew, k4 + skew) is
    /// EXACTLY unbiased on clean fields and spike-bounded from both sides
    /// (a short-baseline bump spike leaves k4 + skew — the clean value; a
    /// long-baseline spike leaves k2 + 2·skew — also the clean value; both
    /// spiked stays the v2 known-accept). Extrapolation is GATED on both
    /// reads clearing the DZ — a constant field (still water, F7's
    /// mirror) has both reads exactly 0 and returns exactly 0 with no
    /// skew added (the bitwise flat contract), and a 2px-dead-zoned
    /// close-up dome takes the bare k4 (the v1.5.2 doc's "4px rescue",
    /// which the old min could never actually deliver: min(0, k4) = 0),
    /// never an extrapolation built on a read that saw nothing. κ_hi
    /// additionally covers the applied κ so the uncertainty bracket can
    /// only widen (the cap gets more conservative, never less).
    #[allow(clippy::too_many_arguments)]
    pub fn vm_kappa(
        dn2x: f32,
        dn4x: f32,
        dn2y: f32,
        dn4y: f32,
        view_z: f32,
        proj: f32,
        scale: f32,
    ) -> (f32, f32, f32) {
        let k2x = vm_curvature_at(dn2x, 2.0 * scale, view_z, proj);
        let k4x = vm_curvature_at(dn4x, 4.0 * scale, view_z, proj);
        let k2y = vm_curvature_at(dn2y, 2.0 * scale, view_z, proj);
        let k4y = vm_curvature_at(dn4y, 4.0 * scale, view_z, proj);
        let skew = VM_DN_DZ * proj / (4.0 * scale * view_z.max(1e-4));
        let axis = |k2: f32, k4: f32| -> f32 {
            if k2 <= 0.0 {
                if k4 <= 0.0 { 0.0 } else { k4 }
            } else if k4 <= 0.0 {
                0.0
            } else {
                (k2 + 2.0 * skew).min(k4 + skew)
            }
        };
        let kappa = axis(k2x, k4x).max(axis(k2y, k4y));
        let lo = k2x.min(k4x).min(k2y.min(k4y));
        let hi = k2x.max(k4x).max(k2y.max(k4y)).max(kappa);
        (kappa, lo, hi)
    }

    /// The v1.5 virtual-motion DELTA, pixels: project the SURFACE point and
    /// the VIRTUAL point (the reflection's image at t_surf + t_r unfolded
    /// along the primary ray; t_r >= VM_FAR_K·far takes the rotation-only
    /// direction limit, w = 0) through the SAME prev world→clip and return
    /// their difference — added to the measured-MV q, so jitter/convention
    /// offsets cancel to first order and t_r = 0 is exactly (0, 0) by
    /// construction (v == hit bitwise). A parked camera's two projections
    /// agree only to fp (both points sit on one ray through the unmoved
    /// origin), so a sub-VM_DEADZONE2 residue snaps to exactly (0, 0);
    /// either projection landing behind the prev image plane (w <= 1e-6)
    /// returns (0, 0) — the humble arm, coarser never wrong-signed.
    /// `right_s`/`up_s` carry the CamBasis pre-scaling (tan(fov/2)·aspect /
    /// tan(fov/2)) — the ngxfg_guides::virtual_prev_px conventions VERBATIM
    /// (the F0 cross-pin holds the two engines to one unfold).
    #[allow(clippy::too_many_arguments)]
    pub fn spec_virtual_delta_px(
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
        view_z: f32,
        t_r: f32,
        cam_far: f32,
        origin: glam::Vec3A,
        fwd: glam::Vec3A,
        right_s: glam::Vec3A,
        up_s: glam::Vec3A,
        m: &glam::Mat4,
    ) -> [f32; 2] {
        use glam::Vec4;
        let ndx = cx * (2.0 / w) - 1.0;
        let ndy = 1.0 - cy * (2.0 / h);
        let du = (fwd + right_s * ndx + up_s * ndy).normalize();
        let ray_t = view_z / du.dot(fwd);
        let hit = origin + du * ray_t;
        let ps = *m * Vec4::new(hit.x, hit.y, hit.z, 1.0);
        let pv = if t_r >= VM_FAR_K * cam_far {
            *m * Vec4::new(du.x, du.y, du.z, 0.0)
        } else {
            let v = origin + du * (ray_t + t_r);
            *m * Vec4::new(v.x, v.y, v.z, 1.0)
        };
        if ps.w <= 1e-6 || pv.w <= 1e-6 {
            return [0.0, 0.0];
        }
        let spx = [(ps.x / ps.w + 1.0) * 0.5 * w, (1.0 - ps.y / ps.w) * 0.5 * h];
        let vpx = [(pv.x / pv.w + 1.0) * 0.5 * w, (1.0 - pv.y / pv.w) * 0.5 * h];
        let d = [vpx[0] - spx[0], vpx[1] - spx[1]];
        if d[0] * d[0] + d[1] * d[1] < VM_DEADZONE2 {
            return [0.0, 0.0];
        }
        d
    }

    /// The virtual foot's disocclusion tolerance: the base relative-Z test
    /// widened proportionally to the APPLIED virtual offset (slack = 1 +
    /// VM_Z_SLACK · |delta_px|) — a grazing planar mirror's reflector depth
    /// varies along the plane, and the base tolerance at a far-displaced
    /// foot would history-restart every mirror fetch. slack = 1.0 is the
    /// base test EXACTLY (the F0 identity pin).
    pub fn disocclusion_z_valid_slack(z_expect: f32, z_prev: f32, n_dot_v: f32, slack: f32) -> bool {
        let scale = z_expect.abs().max(z_prev.abs()).max(1e-6);
        let rel = (z_prev - z_expect).abs() / scale;
        rel < Z_EPS * slack / n_dot_v.max(0.1)
    }

    /// v1.5.5 twins of the kernel's max-speed containment pair (see
    /// VM_SLACK_CAP's doc): the CAPPED virtual-foot slack — below the cap
    /// this is the v1.5 expression bitwise, beyond it the z-test stops
    /// widening so a far-displaced fetch must actually match — …
    pub fn vm_slack_for(appl_px: f32) -> f32 {
        1.0 + VM_Z_SLACK * appl_px.min(VM_SLACK_CAP)
    }

    /// … and the applied-offset confidence charge, in the history cap's
    /// radians-at-the-hit unit (|appl|/proj is the offset as an angle):
    /// trust shortens in proportion to how far the fetch reached.
    pub fn vm_appl_unc(appl_px: f32, proj: f32) -> f32 {
        VM_APPL_UNC_K * appl_px / proj.max(1e-4)
    }

    /// The v1.5.5 magnitude fade (see VM_FADE_LO's doc): 1.0 below LO·rh
    /// (bitwise pass-through of the applied delta), 0.0 at and past HI·rh
    /// (the surface fetch), smoothstep between.
    pub fn vm_mag_weight(dl_px: f32, rh: f32) -> f32 {
        let lo = VM_FADE_LO * rh;
        let hi = VM_FADE_HI * rh;
        let t = ((dl_px - lo) / (hi - lo)).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    /// The v1.5.5(d) tap guard's weight factor (see TAP_LUMA_K's doc):
    /// exactly 1.0 whenever the tap is within K× of the center (ordinary
    /// fields ride a bitwise multiply-by-1), else the ratio that caps the
    /// tap's delivered luma at K× the center's.
    pub fn tap_luma_guard(tap_luma: f32, center_luma: f32) -> f32 {
        (TAP_LUMA_K * (center_luma + TAP_LUMA_EPS)
            / (tap_luma + TAP_LUMA_EPS).max(1e-20))
            .min(1.0)
    }

    /// The v1.5.5(c) smear-budget frame count (see SPEC_PX_BUDGET's doc):
    /// BUDGET·rh over the glint's per-frame screen motion (parallax·proj,
    /// pixels), floored at 1 (a frame always keeps itself). Res-invariant:
    /// 2× proj with 2× rh cancels. The caller MINs it with the rate cap,
    /// so a parked camera (motion → 0) is untouched by construction.
    pub fn spec_budget_frames(parallax: f32, proj: f32, rh: f32) -> f32 {
        (SPEC_PX_BUDGET * rh / (parallax.max(0.0) * proj.max(0.0)).max(1e-4)).max(1.0)
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
        // The wire-form identity: antilag_frames(n, e) == n · antilag_gain(e)
        // over an e sweep spanning both sides of the box — the algebra the
        // R8G8_UNORM antilag plane rides on (pass 3 stores g, pass 1
        // multiplies) — and in-box g is EXACTLY 1.0 (the common path must be
        // bitwise inert through the multiply).
        for e in [0.0f32, 0.5, 1.0, 1.5, 2.0, 7.5, 100.0] {
            let (lhs, rhs) = (antilag_frames(20.0, e), 20.0 * antilag_gain(e));
            if (lhs - rhs).abs() > 1e-4 {
                return Err(format!("frd: antilag gain identity at e={e} ({lhs} vs {rhs})"));
            }
        }
        if antilag_gain(0.5) != 1.0 || antilag_gain(1.0) != 1.0 {
            return Err("frd: antilag gain must be exactly 1 in-box".into());
        }
        // The consumer's history-fix floor: g = 1 is the exact identity; a
        // total cut (g = 0) lands ON the fast-history rate (N_FIX) and
        // NEVER below it (an n already inside the window is untouched —
        // cutting into the restart window re-arms the wide history-fix
        // blur, the F6 limit cycle).
        for n in [0.0f32, 2.0, 4.0, 10.0, 30.0] {
            if antilag_apply(n, 1.0) != n {
                return Err("frd: antilag_apply must be the identity at g=1".into());
            }
        }
        if antilag_apply(30.0, 0.0) != N_FIX || antilag_apply(2.0, 0.0) != 2.0 {
            return Err("frd: antilag_apply floor must be min(n, N_FIX)".into());
        }
        if antilag_apply(30.0, 0.5) != 15.0 || antilag_apply(30.0, 0.05) != N_FIX {
            return Err("frd: antilag_apply cut/floor anchors".into());
        }
        // The excess's RELATIVE floor: a converged signal's zero-width box
        // + ulp-scale blur residue must read as NO excess (gain exactly 1
        // — the F5b junk-gain class), while a signal-scale ghost still
        // reads large, and a real box wider than the floor dominates.
        if antilag_gain(antilag_excess(3e-6, 0.0, 3.0)) != 1.0 {
            return Err("frd: ulp residue on a converged signal must not read as excess".into());
        }
        if antilag_excess(1.5, 0.0, 0.2) < 100.0 {
            return Err("frd: a signal-scale ghost must read as large excess".into());
        }
        let (e_box, e_floor) = (antilag_excess(1.0, 0.5, 3.0), antilag_excess(1.0, 0.001, 3.0));
        if (e_box - 2.0).abs() > 1e-5 || (e_floor - 1.0 / 0.03).abs() > 1e-3 {
            return Err(format!("frd: antilag_excess denominators ({e_box} {e_floor})"));
        }

        // Light parallax: bit-equal dirs EXACTLY 0 (an ABSOLUTE pin — the
        // strafe-gate lesson: a relative bound passes any fraction of a
        // large value, and acos of a fp-rounded self-dot is NOT 0 without
        // the early-out); the 2× mirror-swing factor pinned at a right
        // angle (2·π/2 = π) and at 1°; monotone; an over-unit dot (two
        // near-identical normalized dirs) must come back finite and >= 0.
        if light_parallax([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]) != 0.0 {
            return Err("frd: light_parallax must be exactly 0 for a static sun".into());
        }
        let quarter = light_parallax([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        if (quarter - std::f32::consts::PI).abs() > 1e-5 {
            return Err(format!("frd: light_parallax 2x pin at 90 deg ({quarter})"));
        }
        let one_deg = light_parallax([1.0, 0.0, 0.0], [0.999_847_7, 0.017_452_4, 0.0]);
        if (one_deg - 2.0 * 0.017_453_29).abs() > 1e-3 || one_deg >= quarter {
            return Err(format!("frd: light_parallax 1-deg anchor ({one_deg})"));
        }
        let ou = light_parallax([0.6, 0.8, 0.0], [0.600_000_1, 0.8, 0.0]);
        if !ou.is_finite() || ou < 0.0 {
            return Err(format!("frd: light_parallax over-unit dot safety ({ou})"));
        }

        // v1.5 virtual motion: the fade window's endpoints; the parked-
        // camera EXACT-zero snap (both projections share one ray through the
        // unmoved origin — the dead-zone absorbs the fp residue the ngxfg
        // static anchor bounds at 2e-2 px); t_r = 0 exact zero at a MOVED
        // camera (pv == ps bitwise); the moved-camera mirror delta must
        // FIRE with the sky arm answering differently from the finite arm
        // (anti-vacuity + the rotation-only limit is a different function);
        // the CROSS-PIN holds this unfold to ngxfg_guides::virtual_prev_px
        // — one convention, two engines, one pin; and the slack
        // disocclusion at slack 1.0 is the base test verbatim.
        {
            use crate::camera::Camera;
            use glam::{Vec3A, Vec4};
            if vm_rough_fade(0.0) != 1.0
                || vm_rough_fade(VM_ROUGH_LO) != 1.0
                || vm_rough_fade(VM_ROUGH_HI) != 0.0
                || vm_rough_fade(1.0) != 0.0
            {
                return Err("frd: vm_rough_fade endpoints".into());
            }
            let mid = vm_rough_fade(0.5 * (VM_ROUGH_LO + VM_ROUGH_HI));
            if !(mid > 0.0 && mid < 1.0) {
                return Err("frd: vm_rough_fade must be interior mid-window".into());
            }
            let (rw, rh) = (432usize, 324usize);
            let (near, far) = (0.05f32, 5000.0f32);
            let cam = Camera::look_at(Vec3A::new(4.0, 2.0, 4.0), Vec3A::new(0.0, 1.0, 0.0), 0.9);
            let f = cam.forward();
            let r = f.cross(Vec3A::Y).normalize();
            let u = r.cross(f);
            let tanh = (cam.fov_y * 0.5).tan();
            let (org, fwd) = (cam.pos, f);
            let (rgt, upv) = (r * (tanh * rw as f32 / rh as f32), u * tanh);
            let mats = crate::dlss::cam_matrices(&cam, rw, rh, near, far);
            let wc = mats.view_to_clip * mats.world_to_view;
            for &(cx, cy, t_r) in
                &[(216.5f32, 162.5f32, 3.0f32), (40.5, 300.5, 30.0), (400.5, 20.5, far)]
            {
                let d = spec_virtual_delta_px(
                    cx, cy, rw as f32, rh as f32, 7.0, t_r, far, org, fwd, rgt, upv, &wc,
                );
                if d != [0.0, 0.0] {
                    return Err(format!(
                        "frd: parked-camera virtual delta must snap to exactly 0 ({d:?} at t_r={t_r})"
                    ));
                }
            }
            let prev_cam = Camera {
                pos: cam.pos + Vec3A::new(0.12, 0.02, -0.05),
                yaw: cam.yaw + 0.01,
                pitch: cam.pitch - 0.004,
                fov_y: cam.fov_y,
            };
            let pmats = crate::dlss::cam_matrices(&prev_cam, rw, rh, near, far);
            let wp = pmats.view_to_clip * pmats.world_to_view;
            let d0 = spec_virtual_delta_px(
                216.5, 162.5, rw as f32, rh as f32, 7.0, 0.0, far, org, fwd, rgt, upv, &wp,
            );
            if d0 != [0.0, 0.0] {
                return Err("frd: t_r = 0 must be exactly zero delta at a moved camera".into());
            }
            let df = spec_virtual_delta_px(
                216.5, 162.5, rw as f32, rh as f32, 7.0, 40.0, far, org, fwd, rgt, upv, &wp,
            );
            let ds = spec_virtual_delta_px(
                216.5, 162.5, rw as f32, rh as f32, 7.0, far, far, org, fwd, rgt, upv, &wp,
            );
            if df == [0.0, 0.0] || ds == [0.0, 0.0] {
                return Err("frd: moved-camera virtual delta must fire (anti-vacuity)".into());
            }
            if (df[0] - ds[0]).abs() + (df[1] - ds[1]).abs() < 1e-3 {
                return Err("frd: the sky arm must differ from the finite arm".into());
            }
            // v1.5.1 curvature: κ ≤ 0 is the BITWISE identity (the flat
            // path's contract — still water and F7's mirror ride it); the
            // κ·t_r ≫ 1 limit is the mirror focal length 1/(2κ); monotone
            // between; the dead-zone is EXACTLY 0 below VM_DN_DZ with a
            // closed-form anchor above and max-axis symmetry; and the
            // COMPOSED pin: at heavy curvature the moved-camera sky delta
            // must snap to exactly (0,0) — the virtual image lands on the
            // surface, whose delta is sub-deadzone — while the flat sky
            // arm (ds above) provably fires.
            for t in [0.5f32, 7.0, 5000.0] {
                if virtual_dist(t, 0.0) != t || virtual_dist(t, -1.0) != t {
                    return Err("frd: virtual_dist must be the identity at kappa <= 0".into());
                }
            }
            let lim = virtual_dist(1e9, 0.7);
            if (lim - 1.0 / 1.4).abs() > 1e-3 {
                return Err(format!("frd: virtual_dist far limit must be 1/(2k) ({lim})"));
            }
            if virtual_dist(40.0, 0.1) <= virtual_dist(40.0, 0.5)
                || virtual_dist(40.0, 0.5) <= virtual_dist(40.0, 2.0)
            {
                return Err("frd: virtual_dist must be monotone decreasing in kappa".into());
            }
            if vm_curvature_at(VM_DN_DZ * 0.9, 2.0, 5.0, 406.0) != 0.0
                || vm_curvature_at(VM_DN_DZ * 0.9, 4.0, 5.0, 406.0) != 0.0
            {
                return Err("frd: vm_curvature_at must be exactly 0 inside the dead-zone".into());
            }
            let ka = vm_curvature_at(0.015, 2.0, 5.0, 406.0);
            if (ka - (0.015 - VM_DN_DZ) * 406.0 / 10.0).abs() > 1e-4 {
                return Err(format!("frd: vm_curvature_at closed-form anchor ({ka})"));
            }
            // The v1.5.2 combine: on a CLEAN linear field (dn4 = 2·dn2) the
            // dead-zone skew makes the 2px estimate the strict per-axis min,
            // so κ equals it EXACTLY; a single-scale bump spike (dn2 5×
            // while dn4 stays clean) must be SUPPRESSED to the clean
            // long-baseline value — the leading-streak teeth (an over-read
            // κ paints brightness AHEAD of the glint); axes are symmetric;
            // a single-axis field's κ_lo is exactly 0 (the honest-cylinder
            // pin: genuine ambiguity ⇒ the bracket fires ⇒ short history);
            // and the consistent field's bracket width is the exact
            // dead-zone skew DZ·proj/(4z).
            let a = 0.02f32;
            // skew = DZ·proj/(2·b1·z) at scale 1 (b1 = 2) — the exact
            // clean-field gap k4 − k2, and therefore the de-bias term.
            let skew = VM_DN_DZ * 406.0 / (4.0 * 5.0);
            // v1.5.3(b): on a clean field (dn4 = 2·dn2) BOTH extrapolated
            // arms land on the UNBIASED κ = dn2·proj/(b1·z) — no DZ term
            // at all — and the old biased 2px read must sit strictly
            // under it (the de-bias teeth).
            let clean = vm_kappa(a, 2.0 * a, 0.0, 0.0, 5.0, 406.0, 1.0);
            let unbiased = a * 406.0 / (2.0 * 5.0);
            if (clean.0 - unbiased).abs() > 1e-4 {
                return Err(format!(
                    "frd: vm_kappa clean-field κ must be the UNBIASED closed form \
                     ({} vs {unbiased})",
                    clean.0
                ));
            }
            if clean.0 <= vm_curvature_at(a, 2.0, 5.0, 406.0) {
                return Err("frd: vm_kappa de-bias has no teeth (must exceed the raw 2px read)"
                    .into());
            }
            if clean.1 != 0.0 || clean.2 < clean.0 {
                return Err("frd: vm_kappa single-axis bracket (κ_lo exactly 0, κ_hi covers κ)"
                    .into());
            }
            // A single-scale bump spike (dn2 5× while dn4 stays clean)
            // must be SUPPRESSED to the clean-field extrapolation of the
            // long baseline (k4 + skew) — the leading-streak teeth.
            let spiked = vm_kappa(5.0 * a, 2.0 * a, 0.0, 0.0, 5.0, 406.0, 1.0);
            let spiked_want = vm_curvature_at(2.0 * a, 4.0, 5.0, 406.0) + skew;
            if (spiked.0 - spiked_want).abs() > 1e-4 {
                return Err(format!(
                    "frd: vm_kappa must suppress a single-scale spike to the de-biased \
                     long-baseline value ({} vs {spiked_want})",
                    spiked.0
                ));
            }
            if spiked.0 >= vm_curvature_at(5.0 * a, 2.0, 5.0, 406.0) {
                return Err("frd: vm_kappa spike suppression has no teeth".into());
            }
            // The REAL 4px rescue (v1.5.2 documented it; its min could
            // never deliver it): a 2px read inside the dead-zone with a
            // live 4px read takes the BARE k4 — no extrapolation built on
            // a read that saw nothing (the humble arm).
            let rescued = vm_kappa(VM_DN_DZ * 0.5, 2.0 * a, 0.0, 0.0, 5.0, 406.0, 1.0);
            if rescued.0 != vm_curvature_at(2.0 * a, 4.0, 5.0, 406.0) {
                return Err("frd: vm_kappa 2px-dead-zoned axis must take the bare 4px read".into());
            }
            // The bitwise flat contract: a constant field's four zero
            // reads must yield exactly (0, 0, 0) — no skew leaks in
            // (still water / F7's mirror ride the κ=0 branch).
            if vm_kappa(0.0, 0.0, 0.0, 0.0, 5.0, 406.0, 1.0) != (0.0, 0.0, 0.0) {
                return Err("frd: vm_kappa constant field must be exactly (0,0,0)".into());
            }
            // Short-only signal (bump noise the long baseline never saw)
            // stays suppressed to 0 — the old min's spike behavior.
            if vm_kappa(5.0 * a, 0.0, 0.0, 0.0, 5.0, 406.0, 1.0).0 != 0.0 {
                return Err("frd: vm_kappa short-only spike must read 0".into());
            }
            let sym_a = vm_kappa(a, 2.0 * a, 0.003, 0.009, 5.0, 406.0, 1.0);
            let sym_b = vm_kappa(0.003, 0.009, a, 2.0 * a, 5.0, 406.0, 1.0);
            if sym_a != sym_b {
                return Err("frd: vm_kappa must be symmetric in its axes".into());
            }
            // The consistent field's RAW bracket floor is k2 (both axes
            // equal) and its κ_hi covers the de-biased κ, so the width is
            // exactly 2·skew — the de-bias magnitude made visible to the
            // uncertainty charge.
            let cons = vm_kappa(a, 2.0 * a, a, 2.0 * a, 5.0, 406.0, 1.0);
            if (cons.2 - cons.1 - 2.0 * skew).abs() > 1e-4 {
                return Err(format!(
                    "frd: consistent-field bracket must be 2× the dead-zone skew ({} vs {})",
                    cons.2 - cons.1,
                    2.0 * skew
                ));
            }
            // v1.5.3 baseline-scale rule: every gate/lab res stays at
            // scale 1 (the bitwise no-regression contract — the F1..F7C
            // harnesses run at 533x400/800x600 and the batch lab at
            // 960x540), 1080p doubles, 4K quadruples.
            for (rh_px, s) in [
                (400u32, 1.0f32),
                (540, 1.0),
                (600, 1.0),
                (810, 2.0),
                (1080, 2.0),
                (1440, 3.0),
                (2160, 4.0),
            ] {
                if vm_baseline_scale(rh_px) != s {
                    return Err(format!(
                        "frd: vm_baseline_scale({rh_px}) must be {s} (got {})",
                        vm_baseline_scale(rh_px)
                    ));
                }
            }
            // The res-invariance identity, BITWISE: at 2× the render
            // height (proj doubles) with 2× the texel offsets, a smooth
            // dome's per-baseline |Δn| is measured over the SAME world
            // span, so the same dn inputs must yield the same κ — the
            // scale factors are powers of two, so fp-exact. This is the
            // whole point of v1.5.3: the estimator (dead-zone included)
            // becomes a pure function of the world footprint, not the
            // pixel pitch.
            let k_540 = vm_kappa(a, 2.0 * a, 0.5 * a, a, 5.0, 406.0, 1.0);
            let k_1080 = vm_kappa(a, 2.0 * a, 0.5 * a, a, 5.0, 812.0, 2.0);
            if k_540 != k_1080 {
                return Err("frd: vm_kappa must be resolution-invariant under the \
                            baseline scale (2x proj + 2x scale must be bitwise)"
                    .into());
            }
            // v1.5.5 — the max-speed containment pair: below the cap the
            // slack is the v1.5 expression BITWISE (the validated normal-
            // speed regime — F7's 38 px sits under it), beyond it the
            // widening STOPS (a 500 px-displaced fetch must pass a
            // near-normal z-test instead of a vacuous one — the B70
            // blotch's mechanism); the offset charge is zero at zero,
            // linear (powers of two — bitwise), and res-invariant in the
            // radians unit.
            if vm_slack_for(38.0) != 1.0 + VM_Z_SLACK * 38.0 {
                return Err("frd: vm_slack_for below the cap must be the v1.5 slack bitwise".into());
            }
            if vm_slack_for(1000.0) != vm_slack_for(VM_SLACK_CAP)
                || vm_slack_for(1000.0) > 1.0 + VM_Z_SLACK * VM_SLACK_CAP
            {
                return Err("frd: vm_slack_for must stop widening at the cap".into());
            }
            if vm_appl_unc(0.0, 518.0) != 0.0 {
                return Err("frd: vm_appl_unc must be exactly 0 at zero offset".into());
            }
            if vm_appl_unc(256.0, 518.0) != 2.0 * vm_appl_unc(128.0, 518.0) {
                return Err("frd: vm_appl_unc must be linear in the offset".into());
            }
            if vm_appl_unc(256.0, 512.0) != vm_appl_unc(128.0, 256.0) {
                return Err("frd: vm_appl_unc must be res-invariant (an angle)".into());
            }
            // The magnitude fade: exactly 1 below LO·rh (the validated
            // regime rides a multiply-by-1.0), exactly 0 at/past HI·rh
            // (the surface-fetch collapse), monotone between, and
            // res-invariant (2× the offset at 2× the height is bitwise).
            if vm_mag_weight(0.5 * VM_FADE_LO * 1080.0, 1080.0) != 1.0
                || vm_mag_weight(0.0, 540.0) != 1.0
            {
                return Err("frd: vm_mag_weight below LO must be exactly 1".into());
            }
            if vm_mag_weight(VM_FADE_HI * 1080.0, 1080.0) != 0.0
                || vm_mag_weight(2.0 * VM_FADE_HI * 1080.0, 1080.0) != 0.0
            {
                return Err("frd: vm_mag_weight at/past HI must be exactly 0".into());
            }
            let mid = 0.5 * (VM_FADE_LO + VM_FADE_HI);
            if vm_mag_weight(mid * 1080.0, 1080.0) >= vm_mag_weight(VM_FADE_LO * 1080.0, 1080.0)
                || vm_mag_weight(mid * 1080.0, 1080.0) <= 0.0
            {
                return Err("frd: vm_mag_weight must fall monotonically inside the window".into());
            }
            if vm_mag_weight(200.0, 1080.0) != vm_mag_weight(100.0, 540.0) {
                return Err("frd: vm_mag_weight must be res-invariant".into());
            }
            // The smear budget: restart-class at max-strafe motion
            // (~0.18·rh/frame ⇒ ~2 frames), history-preserving at the F7
            // regime (~0.095·rh ⇒ ~4), floored at 1, monotone in motion,
            // and res-invariant (2× proj at 2× rh is bitwise).
            let bmax = spec_budget_frames(0.18 * 518.0 / 518.0, 518.0, 518.0);
            if !(1.5..=3.0).contains(&bmax) {
                return Err(format!(
                    "frd: spec_budget_frames at max-strafe motion must be restart-class ({bmax})"
                ));
            }
            let bf7 = spec_budget_frames(0.095, 518.0, 518.0);
            if bf7 < 3.5 {
                return Err(format!(
                    "frd: spec_budget_frames must keep the F7 regime's history ({bf7})"
                ));
            }
            if spec_budget_frames(10.0, 518.0, 518.0) != 1.0 {
                return Err("frd: spec_budget_frames must floor at 1".into());
            }
            if spec_budget_frames(0.05, 518.0, 518.0) <= spec_budget_frames(0.1, 518.0, 518.0) {
                return Err("frd: spec_budget_frames must fall as motion grows".into());
            }
            if spec_budget_frames(0.08, 1036.0, 1036.0) != spec_budget_frames(0.08, 518.0, 518.0) {
                return Err("frd: spec_budget_frames must be res-invariant".into());
            }
            // The tap guard: within K× of the center the factor is EXACTLY
            // 1.0 (ordinary fields bitwise); a 100× glint tap is crushed to
            // ~K× the center (the bloom teeth); monotone falling above K.
            if tap_luma_guard(0.5, 0.5) != 1.0
                || tap_luma_guard(1.9, 0.5) != 1.0
                || tap_luma_guard(0.0, 0.0) != 1.0
            {
                return Err("frd: tap_luma_guard within K of the center must be exactly 1".into());
            }
            let g100 = tap_luma_guard(50.0, 0.5);
            if !(0.01..=0.08).contains(&g100) {
                return Err(format!(
                    "frd: tap_luma_guard must crush a 100x glint tap to ~K/ratio ({g100})"
                ));
            }
            if tap_luma_guard(10.0, 0.5) <= tap_luma_guard(20.0, 0.5) {
                return Err("frd: tap_luma_guard must fall as the tap brightens".into());
            }
            let dc = spec_virtual_delta_px(
                216.5,
                162.5,
                rw as f32,
                rh as f32,
                7.0,
                virtual_dist(far, 1e3),
                far,
                org,
                fwd,
                rgt,
                upv,
                &wp,
            );
            if dc != [0.0, 0.0] {
                return Err(format!(
                    "frd: heavy curvature must snap the sky delta to the surface ({dc:?})"
                ));
            }
            // The cross-pin runs only where both engines take the SAME
            // branch: t_r well under VM_FAR_K·far, and exactly far (both
            // sky) — the (VM_FAR_K·far, far) band deliberately diverges
            // (the f16 sky-compare lesson is FRD's threshold, not ngxfg's).
            for &t_r in &[3.0f32, 40.0, far] {
                let d = spec_virtual_delta_px(
                    216.5, 162.5, rw as f32, rh as f32, 7.0, t_r, far, org, fwd, rgt, upv, &wp,
                );
                let v = crate::gfx::guides::virtual_prev_px(
                    216.5, 162.5, rw as f32, rh as f32, 7.0, t_r, far, org, fwd, rgt, upv, &wp,
                )
                .ok_or("frd: cross-pin virtual point behind camera")?;
                let ndx = 216.5f32 * (2.0 / rw as f32) - 1.0;
                let ndy = 1.0 - 162.5f32 * (2.0 / rh as f32);
                let du = (fwd + rgt * ndx + upv * ndy).normalize();
                let hit = org + du * (7.0 / du.dot(fwd));
                let ps = wp * Vec4::new(hit.x, hit.y, hit.z, 1.0);
                let spx = (
                    (ps.x / ps.w + 1.0) * 0.5 * rw as f32,
                    (1.0 - ps.y / ps.w) * 0.5 * rh as f32,
                );
                if (spx.0 + d[0] - v.0).abs() > 1e-2 || (spx.1 + d[1] - v.1).abs() > 1e-2 {
                    return Err(format!(
                        "frd: virtual-motion cross-pin vs ngxfg diverges at t_r={t_r}"
                    ));
                }
            }
            for &(ze, zp, ndv) in &[
                (10.0f32, 10.19f32, 1.0f32),
                (10.0, 10.21, 1.0),
                (10.0, 20.0, 1.0),
                (10.0, 10.21, 0.1),
            ] {
                if disocclusion_z_valid_slack(ze, zp, ndv, 1.0) != disocclusion_z_valid(ze, zp, ndv)
                {
                    return Err("frd: slack 1.0 must be the base disocclusion test".into());
                }
            }
            if !disocclusion_z_valid_slack(10.0, 10.3, 1.0, 2.0)
                || disocclusion_z_valid_slack(10.0, 10.3, 1.0, 1.0)
            {
                return Err("frd: disocclusion slack must widen monotonically".into());
            }
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

        // Stabilization: the off arms return `cur` EXACTLY — pinned with a
        // history so far out that the lerp-by-1 form (h + (c−h)·1) provably
        // differs in f32, so a branch→lerp regression fails; the alpha cap
        // anchor is exact (exp2-friendly values); the clamp-to-box
        // CONTAINMENT is the can't-ghost/can't-lag teeth (hist far outside
        // the spatial box ⇒ the output stays within alpha-of-cur of the box
        // however old the history claims to be); and a converged in-box
        // history pulls the output within alpha of itself (the convergence
        // direction).
        {
            let cur = [0.1f32, 0.02, -0.01];
            let wild = [1.0e8f32, -1.0e8, 1.0e8];
            let (b0, b1) = ([0.0f32, -0.5, -0.5], [0.2f32, 0.5, 0.5]);
            for (n, m) in [(0.0f32, 20.0f32), (5.0, 0.0), (-1.0, 20.0)] {
                if stab_out(cur, wild, b0, b1, n, m) != cur {
                    return Err(format!("frd: stab off arm not bitwise (n {n} max {m})"));
                }
            }
            if stab_alpha(1.0e9, 4.0) != 0.2 {
                return Err("frd: stab alpha cap anchor (1/(1+4) = 0.2 exact)".into());
            }
            let o = stab_out([0.1, 0.0, 0.0], [50.0, 0.0, 0.0], b0, b1, 60.0, 20.0);
            if o[0] > b1[0] + 1e-5 {
                return Err(format!("frd: stab containment — out {} left the box", o[0]));
            }
            let o2 = stab_out([0.13, 0.0, 0.0], [0.11, 0.0, 0.0], b0, b1, 60.0, 20.0);
            let a = stab_alpha(60.0, 20.0);
            if (o2[0] - (0.11 + (0.13 - 0.11) * a)).abs() > 1e-6 {
                return Err(format!("frd: stab convergence direction ({})", o2[0]));
            }
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
