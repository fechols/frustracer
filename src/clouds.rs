//! Animated procedural volumetric clouds — a drifting slab of 2D coverage
//! carved by 3D erosion noise, raymarched with a two-phase adaptive march
//! whose sample phase is DITHERED per (pixel, frame, sample)
//! (`dither_j`/`dither_jk` — a pure integer hash + spp stratification, so
//! "zero rng draws" still holds; do NOT "clean up" the dither back to fixed
//! midpoints: with a fixed phase the sample altitudes are ray-independent
//! and every smooth field renders as N nested step-entry contours — the
//! wedding-cake bug, twice shipped). The clouds are a
//! *modulation* of the one sky, never a light: display paths see the
//! infinity backdrop
//! (dome + disc + stars) extinguished through the layer plus the layer's own
//! sun-lit scatter, and the direct sun term is attenuated by a cheap
//! transmittance toward the sun (`sun_transmittance`) so the ground darkens
//! when a cloud crosses the sun. `sky::dome()` — and with it the SH ambient,
//! every hemi gather, and both GI reference estimators — stays CLOUD-FREE
//! (the disc-exactly-once table's last row, extended: gather paths already
//! account the sky's energy their own way, and a drifting occluder cannot be
//! baked into a load-time SH projection).
//!
//! # The contracts everything here is built on
//!
//! - **Zero rng draws, zero wall clock.** Every function is a pure function of
//!   (position/direction, `Clouds`); `Clouds::time` is OWNED by main.rs
//!   (measured dt interactively, `idx·CLOUD_SPIN_DT` under --spin,
//!   `CLOUD_CHECK_TIME` pinned in every --check*). That is what keeps every
//!   same-seed / replay / VisCtl-burn bit-identity contract intact.
//! - **Disabled ⇒ bit-identical.** `Clouds::off()` takes guarded early
//!   returns everywhere — no unconditional `+0.0`/`*1.0` — so a `--no-clouds`
//!   session reproduces the pre-cloud renderer exactly (gated by `self_test`).
//!   The same discipline holds per-ray when enabled: a direction whose march
//!   never meets a cloud returns `None` and the caller's backdrop passes
//!   through untouched, and an unshadowed surface point gets an exact `1.0`.
//! - **Scale-relative.** Every length is a multiple of `Clouds::diag`
//!   (`Scene::diag` — the codebase's epsilon convention), so the same
//!   constants work for the diag-10 auto-fit, `--stress`, and `--tile`.
//! - **CPU/GPU parity.** The noise pipeline is integer (`sky::pcg_mix` — the
//!   star field's hash, reused verbatim); only float lerps/exp can diverge
//!   (~ulps, absorbed by the statistical CPU-vs-GPU gates). The HLSL twin is
//!   `trace_common.hlsli`'s cloud block — term-for-term, change both together.
//!
//! Known-accepts (documented, deliberate): the SH ambient is static (an
//! overcast patch does not darken the ambient it stands under); clouds have no
//! motion vectors (upscalers see drift as shading change — drift is slow); a
//! camera above the slab sees no clouds (2.5D); the shadow transmittance is a
//! 2-eval estimate, softer than the marched truth, and fades to none below
//! sun elevation `CLOUD_SUN_MIN_Y + CLOUD_FADE_BAND` (a grazing sun casts no
//! cloud shadow — the slant probe degenerates there); and the MIS pair delivers
//! the sun through two slightly different transmittances of the same field
//! (light strategy: `sun_transmittance`; BSDF strategy: the march's `t` — both
//! in [0,1] along near-identical directions, so the blend is bracketed, never
//! a double count or a hole — do NOT force one T on both sides).

use crate::sky::Sun;
use glam::{Vec2, Vec3A};
use std::sync::atomic::{AtomicBool, Ordering};

/// Session enable — the `--no-clouds` kill lever, set ONCE at flag-parse time
/// before any frame exists (the `bloom::set_enabled` / `texture::set_mips`
/// pattern for session constants). The per-frame `Clouds` SNAPSHOTS it at
/// construction, so the math stays a pure function of its inputs and the
/// self-test can exercise both arms explicitly.
static ENABLED: AtomicBool = AtomicBool::new(true);
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Cloud base altitude, in scene diagonals — safely above the diag-10
/// auto-fit's geometry, low enough that shadow features live at scene scale.
pub const CLOUD_BASE_K: f32 = 1.6;
/// Slab thickness, in diagonals.
pub const CLOUD_THICK_K: f32 = 0.8;
/// Rays flatter than this (d.y) skip clouds entirely — a 2.5D slab edge-on is
/// a degenerate march, the horizon is where the dome's own haze lives, and
/// a slab seen at grazing integrates to near-total cover however sparse the
/// field is (long chords always cross a cloud), which reads as a murky band.
pub const CLOUD_MIN_DY: f32 = 0.05;
/// The fade band above `CLOUD_MIN_DY` over which the layer eases in, so the
/// skip is not a visible seam.
pub const CLOUD_FADE_BAND: f32 = 0.10;
/// Sun elevations at/below this (dir.y) cast no cloud shadow — the 2-eval
/// slant probe degenerates as the sun grazes the slab (the CLOUD_MIN_DY
/// argument, pointed at the light instead of the view ray). The shadow eases
/// out over CLOUD_FADE_BAND above it (`sun_transmittance`'s fade): at this
/// elevation the direct sun still carries ~60-70% of its irradiance
/// (`sky::sun_fade` only bottoms out at the horizon), so a hard cutoff popped
/// every ground shadow off in one TOD-scrub tick.
pub const CLOUD_SUN_MIN_Y: f32 = 0.05;
/// FBM base wavelength, in diagonals — individual clouds are scene-sized.
/// Big on purpose: at 1.4 the sky read as cirrocumulus ripples (the visible
/// features were octaves 2-3); whole puffy shapes need the BASE octave to
/// dominate the coverage decision, and it can only do that if a single
/// wavelength spans a good slice of sky.
pub const CLOUD_SCALE_K: f32 = 2.6;
/// Grid-side cap for the slab-space cloud-shadow cache. The footprint-to-
/// wavelength ratio is scale-INVARIANT (both the span and `l0 = CLOUD_SCALE_K *
/// diag` scale with `diag`), so the derived side is small in practice — this
/// only bites when a low sun spreads the projection, where the shadow is fading
/// out anyway (`CLOUD_SUN_MIN_Y` + `CLOUD_FADE_BAND`). Lives here (not the GPU
/// module) so `shadow_grid_row` and the `--check` grid gates are DLL-free.
pub const CLOUD_SHADOW_MAX: u32 = 512;
/// Wind speed in diagonals/second: a cloud crosses its own wavelength in
/// ~130 s, so something is always visibly passing overhead without the sky
/// reading as time-lapse.
pub const CLOUD_WIND_SPEED_K: f32 = 0.02;
/// Wind SHEAR: xz offset (along the wind direction) per unit altitude above
/// the base. Without it the fixed-step march samples the same 2D cover at
/// every height and the clouds read as stacked pancakes; with it the strata
/// lean downwind and merge into a billow.
pub const CLOUD_SHEAR: f32 = 0.5;
/// Coverage threshold on the FBM sum (range [0, 0.9375]) — ~25% of the sky
/// carries cloud at the default: distinct cumulus drifting through blue sky
/// (the "pass by periodically" brief), not an overcast sheet. Raising it
/// clears the sky further; it is also a COST knob — clear directions take
/// the staged cutoffs' fast path, so coverage is roughly proportional to the
/// layer's CPU price.
pub const CLOUD_THRESH: f32 = 0.60;
/// Softness of the coverage remap — the cloud's edge falloff. Tighter reads
/// as puffy cumulus; wide reads as stringy marbling.
pub const CLOUD_SOFT: f32 = 0.14;
/// Coverage octave-1 amplitude (octave 0 is 0.5). The COVERAGE field is
/// deliberately 2-octave-2D + a constant (`CLOUD_REST_MEAN`) — it decides
/// where clouds ARE; the 3D EROSION octaves decide what shape they are.
pub const CLOUD_AMP1: f32 = 0.3;
/// How deeply the 3D erosion noise carves into the 2D coverage. This is the
/// leg that makes the volume genuinely 3D: a 2D field's occupied sets at
/// different altitudes are NESTED level sets of one function, and a
/// fixed-plane march renders nesting as terraced sheets; subtracting a 3D
/// field decorrelates the isosurface per altitude — billow, not pancake.
pub const CLOUD_EROSION: f32 = 0.4;
/// Peak optical depth straight up through a density-1 slab: cores go dark
/// grey (T ≈ 3%), never charcoal — a single-scatter model with honest τ = 5
/// reads as storm cloud because nothing puts the multiply-scattered light
/// back (see CLOUD_MS).
pub const CLOUD_TAU: f32 = 3.5;
/// COARSE march steps — the occupancy scan's budget, paid by every cloudward
/// ray. The sample phase is DITHERED per (pixel, frame, sample) (`dither_jk`
/// — a pure integer hash, still zero rng draws; see `along`), never fixed:
/// with a fixed phase,
/// `dt = thick/(N·d.y)` makes the sample altitudes ray-INDEPENDENT — every
/// pixel samples the same N horizontal planes of the field, and any smooth
/// field rendered that way shows N nested contours (the wedding-cake bug).
pub const CLOUD_STEPS: u32 = 6;
/// FINE sub-steps per occupied coarse step — where the full 3D density is
/// actually integrated (6·3 = 18 effective steps inside cloud). Only
/// occupied coarse intervals pay; the quality knob if dither grain reads
/// too coarse.
pub const CLOUD_FINE: u32 = 3;
/// Grazing-ray step cap, in slab thicknesses (bounds the chord blow-up as
/// d.y → CLOUD_MIN_DY; the uncovered tail only ever LIGHTENS the cloud).
pub const CLOUD_MAX_STEP_K: f32 = 3.0;
/// Sun-probe step toward the sun from each occupied march point, in slab
/// thicknesses — ONE extra density eval per occupied step buys the
/// silver lining (thin edges toward the sun light up).
pub const CLOUD_SUN_STEP_K: f32 = 0.25;
/// Single-scatter albedo — clouds are very white.
pub const CLOUD_ALBEDO: f32 = 0.92;
/// Fraction of the view direction's own dome radiance used as the cloud's
/// ambient term — sky-colored fill (night clouds dim with the moonlit dome
/// automatically since the caller's dome already carries `sky_scale`). Near
/// 1: a cloud lit only by ambient must not read DARKER than the sky it
/// occludes, or the whole layer looks like smoke.
pub const CLOUD_AMB_K: f32 = 1.0;
/// Isotropic multiple-scattering floor on the SUN term: real clouds are
/// white from every direction because light bounces inside them; a pure
/// single-scatter phase leaves the anti-solar side charcoal. This is the
/// standard cheat (a small direction-free fraction of the sun's light),
/// applied through the same per-step `t_sun`, so cores still shade darker
/// than edges.
pub const CLOUD_MS: f32 = 0.06;
/// Two-lobe Henyey-Greenstein phase: a forward lobe (the silver lining) mixed
/// with a weak backward lobe (rim light away from the sun).
pub const CLOUD_G_FWD: f32 = 0.60;
pub const CLOUD_G_BACK: f32 = -0.15;
pub const CLOUD_LOBE_MIX: f32 = 0.7;
/// Curl-field wavelength, in diagonals — LOW frequency (~2.5× the cover
/// wavelength `l0 = CLOUD_SCALE_K·diag`): the field bends and billows WHOLE
/// clouds; a higher frequency would wrinkle them, which is the erosion's job.
pub const CLOUD_CURL_SCALE_K: f32 = 6.5;
/// Curl displacement amplitude, in diagonals. The curl vector is
/// soft-normalized to |v| < 1 (see `curl_offset`), so this is an EXACT bound
/// on the horizontal displacement — the march's interval-skip margin and the
/// G10 bound gate both lean on that.
pub const CLOUD_CURL_AMP_K: f32 = 0.8;
/// Vertical fraction of the curl displacement (the 3D leg: tops and bottoms
/// billow as a cloud drifts through the field). Kept below the horizontal
/// amplitude — every unit here widens the march's interval-skip margin by
/// `CURL_AMP_K · CURL_YSCALE · diag`, and the skip is a measured +17 ms
/// lever (see `along_k`).
pub const CLOUD_CURL_YSCALE: f32 = 0.3;

/// The pinned animation clock for every --check* suite — nonzero so the gates
/// exercise the advected field, constant so every same-seed/replay/CPU-vs-GPU
/// pair compares the SAME sky.
pub const CLOUD_CHECK_TIME: f32 = 7.5;
/// --spin's per-frame clock advance: `time = frame_idx · CLOUD_SPIN_DT` — a
/// pure function of the frame index, so spin A/Bs stay bit-repeatable.
pub const CLOUD_SPIN_DT: f32 = 1.0 / 120.0;

/// Per-frame cloud state — pure data; all math is free functions of it.
/// Deliberately NO `Default`: every construction site states its policy
/// (live / check-pinned / off), and the compiler enumerates the sites.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clouds {
    pub enabled: bool,
    /// Animation clock, seconds. main.rs owns it — see the module header.
    pub time: f32,
    /// `Scene::diag` — every cloud length is a multiple of it.
    pub diag: f32,
}

impl Clouds {
    /// Disabled — the bit-identical arm (`--no-clouds`, and every context
    /// that predates clouds conceptually).
    pub fn off() -> Clouds {
        Clouds { enabled: false, time: 0.0, diag: 1.0 }
    }
    pub fn new(enabled: bool, diag: f32, time: f32) -> Clouds {
        Clouds { enabled, time, diag }
    }
    /// The live interactive state: the session enable + main.rs's clock.
    pub fn live(diag: f32, time: f32) -> Clouds {
        Clouds::new(enabled(), diag, time)
    }
    /// The pinned headless state — one constant clock for every --check*
    /// suite, so any CPU-reference-vs-GPU pair is comparing the same sky
    /// (and `--check --no-clouds` disables it through the session flag).
    pub fn check(diag: f32) -> Clouds {
        Clouds::new(enabled(), diag, CLOUD_CHECK_TIME)
    }
    /// --spin's clock: a pure function of the frame index.
    pub fn spin(diag: f32, idx: u32) -> Clouds {
        Clouds::new(enabled(), diag, idx as f32 * CLOUD_SPIN_DT)
    }
    /// --cinematic's clock: a pure function of the OUTPUT frame index and the
    /// output frame RATE, so the sky advances in real time (a 30 fps film gets
    /// 1 s of drift per 30 frames) and any single frame is re-renderable in
    /// isolation.
    ///
    /// Deliberately NOT `spin`'s `CLOUD_SPIN_DT`: that is a benchmark cadence
    /// (1/120), which at 30 fps would run the sky at a quarter speed. And
    /// deliberately a function of the OUTPUT frame only — a cinematic frame
    /// accumulates N sub-frames at one pose, and if this took the sub-frame
    /// index those N samples would integrate N different skies and smear the
    /// clouds inside a single image (see cinematic.rs invariant 1).
    pub fn cine(diag: f32, out_frame: u32, fps: u32) -> Clouds {
        Clouds::new(enabled(), diag, out_frame as f32 / fps.max(1) as f32)
    }
}

/// The wind, in world units/second over the (x, z) plane. The direction
/// constants are LITERALS (not a normalize call) so the HLSL twin is
/// bit-identical by construction.
#[inline(always)]
fn wind(diag: f32) -> Vec2 {
    Vec2::new(0.932, 0.362) * (CLOUD_WIND_SPEED_K * diag)
}

/// The ONE advection expression: world (x, z, y) → the cloud field's uv at
/// time `t`. Everything that samples the field goes through here — which is
/// what makes the advection-identity gate exact rather than approximate.
/// The shear term leans the column downwind with altitude (see CLOUD_SHEAR);
/// it is time-independent, so the identity `advect(p, t) ==
/// advect(p, 0) − wind·t` still holds bitwise.
#[inline(always)]
fn advect(p: Vec3A, cl: &Clouds) -> Vec2 {
    let lean = CLOUD_SHEAR * (p.y - CLOUD_BASE_K * cl.diag);
    Vec2::new(p.x + 0.932 * lean, p.z + 0.362 * lean) - wind(cl.diag) * cl.time
}

/// Gates ONLY the frame term of the march-phase dither (`dither_j`'s `n`).
/// `false` = the pre-soft static dither (per-pixel grain, converged content) —
/// the fallback if the night spp-stability gate ever objects to the temporal
/// term. Mirror trace_common.hlsli's `CLOUD_TEMPORAL_DITHER` in lockstep.
pub const CLOUD_TEMPORAL_DITHER: bool = true;

/// The march-phase dither seed, in [0, 1): a PURE integer hash of the pixel
/// coordinates and the frame index — u32-exact CPU↔GPU (`sky::pcg_mix`),
/// consuming NOTHING from any shading rng stream, so every same-seed /
/// replay / VisCtl contract holds (the star-hash precedent; the frame term
/// follows the star TWINKLE precedent — replay re-shades with the fresh
/// ctx's frame, so trace-vs-replay bit-identity is untouched). `n = 0` is
/// the XOR identity, bit-identical to the pre-temporal dither.
///
/// The frame term is what turns the march grain from BAKED per-pixel
/// structure into ordinary temporal noise: plain accumulation (which
/// advances `frame` while still) converges it away, and the 1-spp upscaler
/// sessions hand it to RR/XeSS like the shading noise they already
/// integrate. This is NOT the SKY_J lesson's territory — that gate rejected
/// the sky-tile fill's DIRECTION set changing per frame (sample 0 stayed
/// static, so only spp>1 got noisier); the march phase rides the same
/// directions every frame and applies to every sample symmetrically.
/// Fresh salts: never collide with the star seeds or the lattice hashes.
#[inline(always)]
pub fn dither_j(x: u32, y: u32, n: u32) -> f32 {
    let n = if CLOUD_TEMPORAL_DITHER { n } else { 0 };
    crate::sky::hash01(crate::sky::pcg_mix(
        x.wrapping_mul(0x9E37_79B9)
            ^ y.wrapping_mul(0x85EB_CA6B)
            ^ n.wrapping_mul(0x3C6E_F372)
            ^ 0x68E3_1DA4,
    ))
}

/// The per-(pixel, frame, SAMPLE) march phase: hashed per (pixel, frame),
/// STRATIFIED across the frame's spp samples (`j0 + k/spp`, wrapped). The
/// stratification is why `--spp N` genuinely softens the march — the phase
/// dimension is a smooth 1-D integrand, and N evenly spaced phases integrate
/// it near-exactly, where N copies of ONE phase (the old shared-j rule)
/// integrated nothing. `k = 0` adds an exact 0.0 ⇒ bitwise `dither_j`, so
/// sample 0 (and any spp = 1 frame) is unchanged. The wrap is a conditional
/// subtract, not `fract` — exact, and free of the Rust-trunc vs HLSL-`frac`
/// semantics question.
#[inline(always)]
pub fn dither_jk(x: u32, y: u32, frame: u32, k: u32, spp: u32) -> f32 {
    let s = dither_j(x, y, frame) + k as f32 / spp as f32;
    if s >= 1.0 { s - 1.0 } else { s }
}

/// Value-noise corner hash — `sky::pcg_mix` (the star hash, the HLSL twin's
/// exact integer mix) over a lattice-point mix. i32 → u32 casts wrap, matching
/// HLSL's bit-preserving int → uint.
#[inline(always)]
fn cell_hash(i: i32, j: i32, oct: u32) -> f32 {
    crate::sky::hash01(crate::sky::pcg_mix(
        (i as u32).wrapping_mul(0x9E37_79B9)
            ^ (j as u32).wrapping_mul(0x85EB_CA6B)
            ^ oct.wrapping_mul(0xC2B2_AE3D),
    ))
}

/// 3D lattice corner hash — the 2D mix plus a third axis constant.
#[inline(always)]
fn cell_hash3(i: i32, j: i32, k: i32, oct: u32) -> f32 {
    crate::sky::hash01(crate::sky::pcg_mix(
        (i as u32).wrapping_mul(0x9E37_79B9)
            ^ (j as u32).wrapping_mul(0x85EB_CA6B)
            ^ (k as u32).wrapping_mul(0x27D4_EB2F)
            ^ oct.wrapping_mul(0xC2B2_AE3D),
    ))
}

/// The 4 corner hashes of `vnoise`'s cell, in (h00, h10, h01, h11) order —
/// AVX2-batched where the CPU has it, the scalar `cell_hash` loop verbatim
/// elsewhere. Lane values are BIT-EQUAL to the scalar path by construction
/// (the integer mix is exact in vector lanes; `hash01`'s u32→f32 convert and
/// power-of-two scale are exact for values < 2^24) — gated by `self_test`.
#[inline(always)]
fn corner_hashes(i: i32, j: i32, oct: u32) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if hashx::avx2() {
        return unsafe { hashx::cell_hash_x4(i, j, oct) };
    }
    [
        cell_hash(i, j, oct),
        cell_hash(i + 1, j, oct),
        cell_hash(i, j + 1, oct),
        cell_hash(i + 1, j + 1, oct),
    ]
}

/// The 8 corner hashes of `vnoise3`'s cell, in
/// (h000, h100, h010, h110, h001, h101, h011, h111) order — `corner_hashes`'s
/// 3D twin, one 8-lane pcg_mix instead of eight scalar ones.
#[inline(always)]
fn corner_hashes3(i: i32, j: i32, k: i32, oct: u32) -> [f32; 8] {
    #[cfg(target_arch = "x86_64")]
    if hashx::avx2() {
        return unsafe { hashx::cell_hash3_x8(i, j, k, oct) };
    }
    corner_hashes3_scalar(i, j, k, oct)
}

/// The scalar fallback arm, kept callable on its own so `self_test` can pin
/// the AVX2 lanes against it bitwise on hardware that has both.
#[inline(always)]
fn corner_hashes3_scalar(i: i32, j: i32, k: i32, oct: u32) -> [f32; 8] {
    [
        cell_hash3(i, j, k, oct),
        cell_hash3(i + 1, j, k, oct),
        cell_hash3(i, j + 1, k, oct),
        cell_hash3(i + 1, j + 1, k, oct),
        cell_hash3(i, j, k + 1, oct),
        cell_hash3(i + 1, j, k + 1, oct),
        cell_hash3(i, j + 1, k + 1, oct),
        cell_hash3(i + 1, j + 1, k + 1, oct),
    ]
}

/// AVX2-batched corner hashes. The vector pipeline is `sky::pcg_mix`
/// term-for-term in u32 lanes — `_mm256_mullo_epi32`/`_mm256_add_epi32` are
/// exact wrapping arithmetic, `_mm256_srlv_epi32` is the per-lane variable
/// shift `s >> ((s >> 28) + 4)` (the instruction that makes this AVX2-only:
/// SSE has no per-lane shift), and the `hash01` tail is exact in-vector
/// (`h >> 8` < 2^24, so the i32→f32 convert is lossless and the 2^-24 scale
/// is a pure exponent shift). Lane l of the 3D variant is corner
/// (l & 1, (l >> 1) & 1, (l >> 2) & 1) — `corner_hashes3_scalar`'s order.
/// Same values, same bits, fewer instructions: this module changes NOTHING
/// but time, which is why the HLSL twin and every gate are untouched.
#[cfg(target_arch = "x86_64")]
mod hashx {
    use std::arch::x86_64::*;

    /// One cached CPUID probe — the dispatch branch in the corner fetchers.
    #[inline(always)]
    pub fn avx2() -> bool {
        use std::sync::atomic::{AtomicU8, Ordering};
        static AVX2: AtomicU8 = AtomicU8::new(2);
        match AVX2.load(Ordering::Relaxed) {
            2 => {
                let has = std::is_x86_feature_detected!("avx2");
                AVX2.store(has as u8, Ordering::Relaxed);
                has
            }
            v => v != 0,
        }
    }

    /// `pcg_mix` then `hash01` over 8 u32 lanes — bit-equal to the scalar
    /// pair per lane.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn pcg_hash01_x8(seed: __m256i) -> __m256i {
        let s = _mm256_add_epi32(
            _mm256_mullo_epi32(seed, _mm256_set1_epi32(747796405u32 as i32)),
            _mm256_set1_epi32(2891336453u32 as i32),
        );
        let sh = _mm256_add_epi32(_mm256_srli_epi32(s, 28), _mm256_set1_epi32(4));
        let w = _mm256_xor_si256(_mm256_srlv_epi32(s, sh), s);
        let w = _mm256_mullo_epi32(w, _mm256_set1_epi32(277803737u32 as i32));
        let h = _mm256_xor_si256(_mm256_srli_epi32(w, 22), w);
        _mm256_srli_epi32(h, 8)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn cell_hash3_x8(i: i32, j: i32, k: i32, oct: u32) -> [f32; 8] {
        // The lane seeds are XOR combos of 6 scalar products — cheaper built
        // scalar than multiplied in-vector (6 muls vs 24).
        let a0 = (i as u32).wrapping_mul(0x9E37_79B9);
        let a1 = ((i as u32).wrapping_add(1)).wrapping_mul(0x9E37_79B9);
        let b0 = (j as u32).wrapping_mul(0x85EB_CA6B);
        let b1 = ((j as u32).wrapping_add(1)).wrapping_mul(0x85EB_CA6B);
        let c0 = (k as u32).wrapping_mul(0x27D4_EB2F) ^ oct.wrapping_mul(0xC2B2_AE3D);
        let c1 = ((k as u32).wrapping_add(1)).wrapping_mul(0x27D4_EB2F)
            ^ oct.wrapping_mul(0xC2B2_AE3D);
        let seed = _mm256_setr_epi32(
            (a0 ^ b0 ^ c0) as i32,
            (a1 ^ b0 ^ c0) as i32,
            (a0 ^ b1 ^ c0) as i32,
            (a1 ^ b1 ^ c0) as i32,
            (a0 ^ b0 ^ c1) as i32,
            (a1 ^ b0 ^ c1) as i32,
            (a0 ^ b1 ^ c1) as i32,
            (a1 ^ b1 ^ c1) as i32,
        );
        let t = pcg_hash01_x8(seed);
        let f = _mm256_mul_ps(
            _mm256_cvtepi32_ps(t),
            _mm256_set1_ps(1.0 / 16777216.0),
        );
        let mut out = [0.0_f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), f) };
        out
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn cell_hash_x4(i: i32, j: i32, oct: u32) -> [f32; 4] {
        // 128-bit lanes; `_mm_srlv_epi32` is AVX2-encoded, hence the same gate.
        let a0 = (i as u32).wrapping_mul(0x9E37_79B9);
        let a1 = ((i as u32).wrapping_add(1)).wrapping_mul(0x9E37_79B9);
        let b0 = (j as u32).wrapping_mul(0x85EB_CA6B) ^ oct.wrapping_mul(0xC2B2_AE3D);
        let b1 = ((j as u32).wrapping_add(1)).wrapping_mul(0x85EB_CA6B)
            ^ oct.wrapping_mul(0xC2B2_AE3D);
        let seed = _mm_setr_epi32(
            (a0 ^ b0) as i32,
            (a1 ^ b0) as i32,
            (a0 ^ b1) as i32,
            (a1 ^ b1) as i32,
        );
        let s = _mm_add_epi32(
            _mm_mullo_epi32(seed, _mm_set1_epi32(747796405u32 as i32)),
            _mm_set1_epi32(2891336453u32 as i32),
        );
        let sh = _mm_add_epi32(_mm_srli_epi32(s, 28), _mm_set1_epi32(4));
        let w = _mm_xor_si128(_mm_srlv_epi32(s, sh), s);
        let w = _mm_mullo_epi32(w, _mm_set1_epi32(277803737u32 as i32));
        let h = _mm_xor_si128(_mm_srli_epi32(w, 22), w);
        let t = _mm_srli_epi32(h, 8);
        let f = _mm_mul_ps(_mm_cvtepi32_ps(t), _mm_set1_ps(1.0 / 16777216.0));
        let mut out = [0.0_f32; 4];
        unsafe { _mm_storeu_ps(out.as_mut_ptr(), f) };
        out
    }
}

/// 2D value noise in [0, 1): smoothstep-faded bilerp of 4 corner hashes.
/// `floor()` then cast — NEVER a bare `as i32` truncation (negative
/// coordinates would mirror; the HLSL twin uses `floor()` for the same
/// reason).
fn vnoise(q: Vec2, oct: u32) -> f32 {
    let fx = q.x.floor();
    let fy = q.y.floor();
    let (i, j) = (fx as i32, fy as i32);
    let (tx, ty) = (q.x - fx, q.y - fy);
    let ux = tx * tx * (3.0 - 2.0 * tx);
    let uy = ty * ty * (3.0 - 2.0 * ty);
    let [h00, h10, h01, h11] = corner_hashes(i, j, oct);
    let a = h00 + (h10 - h00) * ux;
    let b = h01 + (h11 - h01) * ux;
    a + (b - a) * uy
}

/// `vnoise` plus its ANALYTIC gradient — the 2D reduction of `vnoise3_grad`.
///
/// Returns (value, d/dq). The water ripple field is built from this because
/// a ripple normal must be the gradient of a scalar height or the surface
/// shimmers with impossible normals; taking the derivative in closed form is
/// what keeps that exact (and lets the frame-generation guide pass evaluate
/// the same field at two times without a finite difference). Same corner
/// hashes as `vnoise`, so it shares the AVX2 batch path and stays u32-exact
/// between CPU and GPU.
///
/// d/dt of the smoothstep t²(3−2t) is 6t(1−t).
pub(crate) fn vnoise_vg(q: Vec2, oct: u32) -> (f32, Vec2) {
    let fx = q.x.floor();
    let fy = q.y.floor();
    let (i, j) = (fx as i32, fy as i32);
    let (tx, ty) = (q.x - fx, q.y - fy);
    let ux = tx * tx * (3.0 - 2.0 * tx);
    let uy = ty * ty * (3.0 - 2.0 * ty);
    let dux = 6.0 * tx * (1.0 - tx);
    let duy = 6.0 * ty * (1.0 - ty);
    let [h00, h10, h01, h11] = corner_hashes(i, j, oct);
    let a = h00 + (h10 - h00) * ux;
    let b = h01 + (h11 - h01) * ux;
    let v = a + (b - a) * uy;
    // ∂/∂x: the x-lerps' slopes, blended in y. ∂/∂y: the y-lerp's own slope.
    let gx = ((h10 - h00) + ((h11 - h01) - (h10 - h00)) * uy) * dux;
    let gy = (b - a) * duy;
    (v, Vec2::new(gx, gy))
}

/// 3D value noise in [0, 1): smoothstep-faded trilerp of 8 corner hashes.
/// This is the erosion field's noise — genuinely varying in all three axes,
/// which is what breaks the nested-level-set structure a 2D field is stuck
/// with. Same floor-then-cast discipline as `vnoise`.
fn vnoise3(q: Vec3A, oct: u32) -> f32 {
    let fx = q.x.floor();
    let fy = q.y.floor();
    let fz = q.z.floor();
    let (i, j, k) = (fx as i32, fy as i32, fz as i32);
    let (tx, ty, tz) = (q.x - fx, q.y - fy, q.z - fz);
    let ux = tx * tx * (3.0 - 2.0 * tx);
    let uy = ty * ty * (3.0 - 2.0 * ty);
    let uz = tz * tz * (3.0 - 2.0 * tz);
    let [h000, h100, h010, h110, h001, h101, h011, h111] = corner_hashes3(i, j, k, oct);
    let a0 = h000 + (h100 - h000) * ux;
    let b0 = h010 + (h110 - h010) * ux;
    let c0 = a0 + (b0 - a0) * uy;
    let a1 = h001 + (h101 - h001) * ux;
    let b1 = h011 + (h111 - h011) * ux;
    let c1 = a1 + (b1 - a1) * uy;
    c0 + (c1 - c0) * uz
}

/// ANALYTIC gradient of `vnoise3` w.r.t. its (unit-cell) coordinate — the
/// same 8 corner hashes, the Hermite fade's own derivative (`du = 6t(1−t)`),
/// and per-axis bilerps of the corner DIFFERENCES. C1 across cell edges
/// because the fade's derivative vanishes there (`vnoise3` is already
/// smoothstep-faded), so the curl field it feeds has no creases. Central
/// differences were rejected on cost (6 extra vnoise3 = 48 hashes); G10
/// gates this against them instead.
fn vnoise3_grad(q: Vec3A, oct: u32) -> Vec3A {
    let c = grad_cell(q);
    grad_from(corner_hashes3(c.i, c.j, c.k, oct), &c)
}

/// TWO gradients of the same cell at two octave ids — `curl_offset`'s pair.
/// One shared floor/fade computation (value-identical: the fades are pure
/// functions of `q`, so sharing them changes nothing but time), two batched
/// corner fetches.
fn vnoise3_grad2(q: Vec3A, o1: u32, o2: u32) -> (Vec3A, Vec3A) {
    let c = grad_cell(q);
    (
        grad_from(corner_hashes3(c.i, c.j, c.k, o1), &c),
        grad_from(corner_hashes3(c.i, c.j, c.k, o2), &c),
    )
}

/// The lattice cell + Hermite fades/derivatives `vnoise3_grad` shares across
/// octaves — hoisted verbatim from the old body.
struct GradCell {
    i: i32,
    j: i32,
    k: i32,
    ux: f32,
    uy: f32,
    uz: f32,
    dux: f32,
    duy: f32,
    duz: f32,
}

#[inline(always)]
fn grad_cell(q: Vec3A) -> GradCell {
    let fx = q.x.floor();
    let fy = q.y.floor();
    let fz = q.z.floor();
    let (i, j, k) = (fx as i32, fy as i32, fz as i32);
    let (tx, ty, tz) = (q.x - fx, q.y - fy, q.z - fz);
    GradCell {
        i,
        j,
        k,
        ux: tx * tx * (3.0 - 2.0 * tx),
        uy: ty * ty * (3.0 - 2.0 * ty),
        uz: tz * tz * (3.0 - 2.0 * tz),
        dux: 6.0 * tx * (1.0 - tx),
        duy: 6.0 * ty * (1.0 - ty),
        duz: 6.0 * tz * (1.0 - tz),
    }
}

/// The gradient bilerps over one cell's 8 corner hashes — the old
/// `vnoise3_grad` expressions verbatim.
#[inline(always)]
fn grad_from(h: [f32; 8], c: &GradCell) -> Vec3A {
    let [h000, h100, h010, h110, h001, h101, h011, h111] = h;
    let (ux, uy, uz) = (c.ux, c.uy, c.uz);
    // d/dx: bilerp the x-differences over (uy, uz).
    let xa = (h100 - h000) + ((h110 - h010) - (h100 - h000)) * uy;
    let xb = (h101 - h001) + ((h111 - h011) - (h101 - h001)) * uy;
    // d/dy: bilerp the y-differences over (ux, uz).
    let ya = (h010 - h000) + ((h110 - h100) - (h010 - h000)) * ux;
    let yb = (h011 - h001) + ((h111 - h101) - (h011 - h001)) * ux;
    // d/dz: bilerp the z-differences over (ux, uy).
    let za = (h001 - h000) + ((h101 - h100) - (h001 - h000)) * ux;
    let zb = (h011 - h010) + ((h111 - h110) - (h011 - h010)) * ux;
    Vec3A::new(
        c.dux * (xa + (xb - xa) * uz),
        c.duy * (ya + (yb - ya) * uz),
        c.duz * (za + (zb - za) * uy),
    )
}

/// The low-frequency **3D curl-noise wind field**, as a STATIC displacement
/// of the sampling position: `v = ∇ψ₁ × ∇ψ₂` (two 3D noise potentials,
/// octave ids 6/7 — a cross of two gradients is the cheap exactly-
/// divergence-free 3D construction), soft-normalized to |v| < 1
/// (`v/(1+|v|)` — direction-preserving, and the HARD bound is what makes
/// the march's interval-skip margin and the slab logic sound; the exact
/// div-free property only ever held to first order for a finite
/// displacement anyway). The field is TIME-INDEPENDENT and sampled at raw
/// world coordinates: clouds translate through it at wind speed, so they
/// continuously deform, wander off the straight wind line, and billow
/// vertically (`CLOUD_CURL_YSCALE`) as they drift — organic non-uniform
/// motion with zero new time terms, which is why the advection-identity
/// gate (G6) survives verbatim: `advect` never sees it. Zero rng draws;
/// pure function of position.
pub(crate) fn curl_offset(p: Vec3A, cl: &Clouds) -> Vec3A {
    let lc = CLOUD_CURL_SCALE_K * cl.diag;
    let q = p * (1.0 / lc);
    let (g6, g7) = vnoise3_grad2(q, 6, 7);
    let v = g6.cross(g7);
    let v = v * (1.0 / (1.0 + v.length()));
    Vec3A::new(v.x, CLOUD_CURL_YSCALE * v.y, v.z) * (CLOUD_CURL_AMP_K * cl.diag)
}

/// Per-octave anti-alias attenuation for a sampling footprint `w` (world
/// units — the march's STEP LENGTH): 1 (full detail) while the octave's
/// wavelength `l` is well resolved (w ≤ l/2), 0 (collapse to the octave's
/// MEAN) once w ≥ l, linear between. The mip philosophy applied to the
/// cloud field — an octave the sampling rate cannot resolve must not be
/// point-sampled, or a grazing march renders each step as its own separated
/// BEAD (the feel-test screenshot). Smooth in w, so there is no view-angle
/// seam; and a fully-attenuated octave SKIPS its noise evals, so the long
/// grazing chords get cheaper exactly where they were most expensive.
#[inline(always)]
fn oct_t(w: f32, l: f32) -> f32 {
    (2.0 - 2.0 * w / l).clamp(0.0, 1.0)
}

/// Mean contribution of the retired detail octaves — the coverage sum's
/// constant third term (`0.5·n0 + AMP1·n1 + REST_MEAN`), kept so
/// `CLOUD_THRESH`'s calibration survives the erosion rework.
pub const CLOUD_REST_MEAN: f32 = 0.1;

/// THE 2D coverage field, in [0, 1] — where clouds ARE. One function, two
/// consumers: `density_lo_f` (cover·prof — shadows, sun probes, the rough
/// march) and `density_f` (erosion-carved cover·prof — the visible march).
/// Sharing it verbatim is what makes `density ≤ density_lo` a BITWISE
/// theorem (self-test G8): erosion only ever subtracts.
///
/// The staged cutoff is the clear-sky fast path AND exact: octave 0's
/// partial sum plus the full remaining amplitude bounds the sum, so the
/// early 0.0 is the value the remap would produce — cheaper, never
/// different (attenuation only shrinks an octave's range about its mean, so
/// the bound stays conservative).
pub(crate) fn cloud_cover(p: Vec3A, cl: &Clouds, w: f32) -> f32 {
    let l0 = CLOUD_SCALE_K * cl.diag;
    let q = advect(p, cl) * (1.0 / l0);
    // Each octave: full detail (the w = 0 arithmetic VERBATIM), attenuated
    // toward its mean, or skipped entirely — three arms so the resolved path
    // never pays (or perturbs by) the attenuation math.
    let t0 = oct_t(w, l0);
    let n0 = if t0 >= 1.0 {
        vnoise(q, 0)
    } else if t0 > 0.0 {
        0.5 + (vnoise(q, 0) - 0.5) * t0
    } else {
        0.5
    };
    let c0 = 0.5 * n0;
    if c0 + CLOUD_AMP1 + CLOUD_REST_MEAN <= CLOUD_THRESH {
        return 0.0;
    }
    let t1 = oct_t(w, l0 * 0.5);
    let n1 = if t1 >= 1.0 {
        vnoise(q * 2.0, 1)
    } else if t1 > 0.0 {
        0.5 + (vnoise(q * 2.0, 1) - 0.5) * t1
    } else {
        0.5
    };
    let c1 = c0 + CLOUD_AMP1 * n1;
    ((c1 + CLOUD_REST_MEAN - CLOUD_THRESH) / CLOUD_SOFT).clamp(0.0, 1.0)
}

/// The column's local top — one formula, three consumers (`cloud_prof`, the
/// march's interval-window skip, the HLSL twins). Keep in lockstep.
#[inline(always)]
fn cloud_top(cover: f32, base: f32, thick: f32) -> f32 {
    base + thick * (0.30 + 0.70 * cover)
}

/// The coverage-driven vertical window: a fast rise off the base, a taper to
/// the column's OWN top (taller where denser — the cumulus heightfield).
/// cover → 0 collapses the window and the density with it — value-continuous.
#[inline(always)]
fn cloud_prof(py: f32, cover: f32, base: f32, thick: f32) -> f32 {
    let top_l = cloud_top(cover, base, thick);
    ((py - base) / (0.20 * thick)).clamp(0.0, 1.0)
        * ((top_l - py) / (0.30 * thick)).clamp(0.0, 1.0)
}

/// The 3D erosion factor s3 in [0, 1] — what SHAPE the cloud is. ONE octave
/// of genuine 3D value noise at l0/4 (a second at l0/8 was tried and cost
/// ~2-4 ms/frame for crinkle the step rate barely resolves); xz rides the
/// SAME `advect` (the erosion drifts and shears with its cloud, and the
/// advection-identity gate holds structurally), y raw over the same
/// wavelength. `oct_t` anti-alias like everything else — at the FINE march
/// step this octave is genuinely resolved.
fn erosion3(p: Vec3A, cl: &Clouds, w: f32) -> f32 {
    let l0 = CLOUD_SCALE_K * cl.diag;
    let le = l0 * 0.25;
    let uv = advect(p, cl);
    let qe = Vec3A::new(uv.x, p.y, uv.y) * (1.0 / le);
    let te0 = oct_t(w, le);
    if te0 >= 1.0 {
        vnoise3(qe, 5)
    } else if te0 > 0.0 {
        0.5 + (vnoise3(qe, 5) - 0.5) * te0
    } else {
        0.5
    }
}

/// Cloud density at a world point, in [0, 1] — full detail (`w = 0`).
pub(crate) fn density(p: Vec3A, cl: &Clouds) -> f32 {
    density_f(p, cl, 0.0)
}

/// The VISIBLE density at an ALREADY-WARPED point `pw` — the body every
/// consumer shares. The public `density_f` applies the exact per-point curl
/// warp; the march folds a per-RAY hoisted warp into its ray origin.
#[inline(always)]
fn density_at(pw: Vec3A, cl: &Clouds, w: f32) -> f32 {
    let base = CLOUD_BASE_K * cl.diag;
    let thick = CLOUD_THICK_K * cl.diag;
    let cover = cloud_cover(pw, cl, w);
    if cover <= 0.0 {
        return 0.0;
    }
    let prof = cloud_prof(pw.y, cover, base, thick);
    if prof <= 0.0 {
        return 0.0;
    }
    let s3 = erosion3(pw, cl, w);
    let eroded = (cover - CLOUD_EROSION * (1.0 - s3)).clamp(0.0, 1.0);
    eroded * prof
}

/// The VISIBLE cloud density at sampling footprint `w`: the shared 2D
/// coverage, carved by the 3D erosion (Nubis-style), inside the
/// coverage-driven vertical window — the whole field sampled through the
/// static 3D curl warp (`curl_offset`). The vnoise3 evals run only where
/// cover·prof > 0 — clear sky and out-of-window samples never pay them
/// (the curl warp itself is paid up front; it is what decides where the
/// field IS). `density_f(p, w) ≤ density_lo_f(p, w)` holds BITWISE (G8):
/// both wrappers apply the identical warp expression, then erosion
/// subtracts a non-negative amount before the shared prof multiply.
pub(crate) fn density_f(p: Vec3A, cl: &Clouds, w: f32) -> f32 {
    density_at(p + curl_offset(p, cl), cl, w)
}

/// The 2D SHADOW/LIGHTING field (cover·prof, no erosion): the hot one —
/// every `shade()` pays two evals (`sun_transmittance`), every occupied
/// march step one (the sun probe), and the rough reflection march runs on
/// it. Blurred by construction (the sun's angular size, the GGX lobe), so
/// the 3D carving bought nothing here and its cost stays off the hot path.
/// It sees the SAME curl warp as the visible field — the shadow must track
/// the cloud that casts it (and G8 requires the shared domain).
pub(crate) fn density_lo(p: Vec3A, cl: &Clouds) -> f32 {
    density_lo_f(p, cl, 0.0)
}

/// The lo field at an ALREADY-WARPED point — `density_at`'s twin.
#[inline(always)]
fn density_lo_at(pw: Vec3A, cl: &Clouds, w: f32) -> f32 {
    let base = CLOUD_BASE_K * cl.diag;
    let thick = CLOUD_THICK_K * cl.diag;
    let cover = cloud_cover(pw, cl, w);
    if cover <= 0.0 {
        return 0.0;
    }
    cover * cloud_prof(pw.y, cover, base, thick)
}

pub(crate) fn density_lo_f(p: Vec3A, cl: &Clouds, w: f32) -> f32 {
    density_lo_at(p + curl_offset(p, cl), cl, w)
}

/// Henyey-Greenstein phase (normalized over the sphere).
#[inline(always)]
fn hg(mu: f32, g: f32) -> f32 {
    let g2 = g * g;
    let den = (1.0 + g2 - 2.0 * g * mu).max(1e-4);
    (1.0 - g2) / (4.0 * std::f32::consts::PI * den * den.sqrt())
}

/// What a ray accumulated crossing the layer: transmittance for the backdrop
/// behind it, plus the layer's own in-scattered radiance.
#[derive(Clone, Copy, Debug)]
pub struct CloudSample {
    pub t: f32,
    pub scatter: Vec3A,
}

/// March the slab along an escaping ray. `None` ⇒ the ray never met cloud
/// (disabled, too flat, origin above the base, or every step empty) and the
/// caller's backdrop must pass through UNTOUCHED — the bit-identity arm.
///
/// `j ∈ [0, 1)` is the march-phase dither (`dither_jk(x, y, frame, k, spp)`
/// where a pixel exists; `0.5` — the fixed-midpoint legacy phase — on
/// pixel-less paths like the glass miss, which the temporal dither
/// deliberately excludes). This is THE anti-wedding-cake leg: with a
/// fixed phase every ray samples the same horizontal planes of the field and
/// any smooth field renders as nested step-entry contours. Still a pure
/// function of its inputs — zero rng draws.
///
/// `amb` is the cloud's ambient fill, `dome(d) · CLOUD_AMB_K` — the caller
/// always has `dome(d)` in hand, and routing it through here keeps the cloud's
/// fill tracking the sky's own color (and `sky_scale`, and therefore night)
/// for free. `sun` may be the MOON (`scene.sun` at night) — nothing here
/// cares, which is exactly the TOD design's point.
pub fn along(o: Vec3A, d: Vec3A, sun: &Sun, amb: Vec3A, cl: &Clouds, j: f32) -> Option<CloudSample> {
    along_k(o, d, sun, amb, cl, j, 1.0)
}

/// The two-phase adaptive march, with an extinction multiplier `sig_mul`
/// (`along` at 1.0; the self-test's monotonicity probe at other values).
///
/// Phase A: `CLOUD_STEPS` coarse dithered probes of the cheap 2D COVERAGE
/// only — deliberately not `density_lo` (a point test of the vertical
/// profile would re-quantize the lump's top surface at coarse resolution:
/// the ring bug through the back door). Phase B: an occupied coarse interval
/// gets `CLOUD_FINE` fine sub-steps of the full 3D density, anti-alias
/// filtered to the FINE step length, plus ONE sun probe (per coarse step,
/// reused — lighting is blurred by construction). Clear rays pay phase A's
/// staged-cutoff fast path only.
fn along_k(
    o: Vec3A,
    d: Vec3A,
    sun: &Sun,
    amb: Vec3A,
    cl: &Clouds,
    j: f32,
    sig_mul: f32,
) -> Option<CloudSample> {
    if !cl.enabled || d.y <= CLOUD_MIN_DY {
        return None;
    }
    let base = CLOUD_BASE_K * cl.diag;
    let thick = CLOUD_THICK_K * cl.diag;
    if o.y >= base {
        // 2.5D: the layer is only modeled from below (see the module header).
        return None;
    }
    let sigma_t = sig_mul * CLOUD_TAU / thick;
    let t0 = (base - o.y) / d.y;
    let dt_c = (thick / (CLOUD_STEPS as f32 * d.y)).min(CLOUD_MAX_STEP_K * thick);
    let dt_f = dt_c / CLOUD_FINE as f32;
    let mu = d.dot(sun.dir).clamp(-1.0, 1.0);
    let phase = CLOUD_LOBE_MIX * hg(mu, CLOUD_G_FWD) + (1.0 - CLOUD_LOBE_MIX) * hg(mu, CLOUD_G_BACK);
    // Sunlight arriving at cloud altitude: E/π rebuilt to irradiance, tinted
    // by the dome's OWN sun-path transmittance — clouds redden at sunset in
    // lockstep with the sky, and at night `sun` is the moon so this is
    // moonlight. (Under a TOD scrub `e_over_pi` already carries `sun_fade`,
    // so the low sun is tinted twice — bounded, smooth, artistic; accepted.)
    let sun_col = sun.e_over_pi * (std::f32::consts::PI) * crate::sky::t_sun_path(sun.dir.y);
    let l_sun = CLOUD_SUN_STEP_K * thick / sun.dir.y.max(0.35);
    let mut t_acc = 1.0_f32;
    let mut sc = Vec3A::ZERO;
    // The curl warp, HOISTED per RAY: one eval at the chord's mid-slab
    // point, folded into a WARPED RAY ORIGIN `ow` — every sample position
    // below is `ow + d·t`, so the inner loops carry zero extra arithmetic
    // and the skip's field-space altitude is exact (`ow.y + d.y·t`).
    // Per-coarse-step warps were measured at +21 ms/frame CPU and nearly
    // double the GPU wavefront's per-sample cost — the march would pay
    // ~6 warps/ray for along-chord variation the eye cannot separate from
    // the ray-to-ray variation it keeps (the curl wavelength is ~5× a
    // steep chord's whole span; grazing chords live in the fade band's
    // blur). density_f/_lo_f keep the exact per-point warp — G8 compares
    // those; the slab geometry (t0, dt_c) stays a function of the REAL o.
    let ow = o + curl_offset(o + d * (t0 + 0.5 * CLOUD_STEPS as f32 * dt_c), cl);
    'march: for i in 0..CLOUD_STEPS {
        // Phase A: DITHERED coarse occupancy probe — 2D cover only, sampled
        // through the ray's hoisted curl warp.
        let tc = t0 + (i as f32 + j) * dt_c;
        let pc = ow + d * tc;
        let cov = cloud_cover(pc, cl, dt_c);
        if cov <= 0.0 {
            continue;
        }
        // Interval-window skip: the coverage predicate fires for the WHOLE
        // column height, but the cloud only occupies [base, top(cover)] —
        // fine-marching the empty air above the top was measured as a
        // +17 ms/frame regression on --spin. Pure arithmetic (the analytic
        // y-extent of this coarse interval vs the column's own top, with a
        // 0.1·thick margin for cover growing between the coarse xz and a
        // fine xz — the phase-A-is-a-point-sample class; residual misses
        // are stable dither grain, not contours). The per-RAY warp hoist
        // makes the curl displacement CONSTANT along this chord, so the
        // field-space altitude below is EXACT — no conservative margin (a
        // per-point warp would force a bound-sized margin here, which was
        // measured re-adding the marching this skip exists to remove).
        let y_lo = ow.y + d.y * (t0 + i as f32 * dt_c);
        if y_lo >= cloud_top(cov, base, thick) + 0.1 * thick {
            continue;
        }
        // One sun probe AND one lighting transmittance per occupied coarse
        // step, shared by its sub-steps — sun lighting is the blurred lo
        // field's business, and a per-fine-sample exp pair was a measured
        // cost driver. The half-local-density term uses the coarse cover as
        // its density proxy (thin edges still pass light — silver lining).
        // The probe rides the ray's hoisted warp (the probe point is well
        // inside one curl wavelength of the chord).
        let rho_sun = density_lo_at(pc + sun.dir * l_sun, cl, 0.0);
        let t_sun = (-((rho_sun + 0.5 * cov) * sigma_t * l_sun)).exp();
        let s = (sun_col * ((phase + CLOUD_MS) * t_sun) + amb) * CLOUD_ALBEDO;
        for m in 0..CLOUD_FINE {
            // Phase B: fine sub-steps, same dither phase j (each sample
            // stays inside its own sub-interval for j ∈ [0,1)). The COARSE
            // cover is REUSED across the sub-interval (cover is the smooth
            // 2D placement — re-evaluating it per fine sample was a measured
            // ~4 ms/frame with no visible gain); only the 3D erosion and the
            // vertical window run at fine resolution, which is exactly the
            // detail the fine steps exist to resolve. Marched density stays
            // ≤ cov·prof — the lo field at the coarse cover — so the G8
            // bound argument carries over to the march's own samples.
            let tf = t0 + i as f32 * dt_c + (m as f32 + j) * dt_f;
            let p = ow + d * tf;
            let prof = cloud_prof(p.y, cov, base, thick);
            if prof <= 0.0 {
                continue;
            }
            let s3 = erosion3(p, cl, dt_f);
            let rho = (cov - CLOUD_EROSION * (1.0 - s3)).clamp(0.0, 1.0) * prof;
            if rho <= 0.0 {
                continue;
            }
            let a = (-(rho * sigma_t * dt_f)).exp();
            sc += s * (t_acc * (1.0 - a));
            t_acc *= a;
            // Opaque-core break: every remaining sample's contribution is
            // weighted by t_acc, so below half a percent nothing visible is
            // left to add. A discrete cut on a continuous value, but bounded
            // at 0.5% — orders under the image gates' hot threshold, and
            // both GPU kernels take the identical branch.
            if t_acc < 0.005 {
                break 'march;
            }
        }
    }
    if t_acc >= 1.0 {
        // Every sample was empty: this ray saw no cloud. None — not
        // Some{1, 0} — so the caller's backdrop stays bit-identical.
        return None;
    }
    // Ease the layer in over the flat-ray band so CLOUD_MIN_DY is not a seam.
    let fade = ((d.y - CLOUD_MIN_DY) / CLOUD_FADE_BAND).clamp(0.0, 1.0);
    Some(CloudSample { t: 1.0 + (t_acc - 1.0) * fade, scatter: sc * fade })
}

/// Steps for the ROUGH-path march (`along_rough`).
pub const CLOUD_ROUGH_STEPS: u32 = 2;

/// The march for SECONDARY specular paths (the reflection-miss site): 2 fixed
/// midpoints over the 2-octave field. A reflected sky is seen through the
/// GGX lobe's blur, so the crinkle and the fine step resolution buy variance,
/// not fidelity — the `HEMI_CONE_SPREAD`/bounce-cone philosophy applied to
/// clouds. Cost history: the FULL march here (the big reflective ground
/// plane sends ~1M rays/frame into the sky on the default scene) was once
/// the largest single share of the cloud layer's CPU cost — this 2-step
/// lo-field form with the per-ray warp fold fixed that, and the 2026-07
/// ablation A/B measures the whole function at ≈ 0 ms/frame on the spin
/// path (the layer's cost now lives ~80% in `along_k`, see CLAUDE.md).
/// Same None/bit-passthrough contract as `along`.
pub fn along_rough(o: Vec3A, d: Vec3A, sun: &Sun, amb: Vec3A, cl: &Clouds) -> Option<CloudSample> {
    if !cl.enabled || d.y <= CLOUD_MIN_DY {
        return None;
    }
    let base = CLOUD_BASE_K * cl.diag;
    let thick = CLOUD_THICK_K * cl.diag;
    if o.y >= base {
        return None;
    }
    let sigma_t = CLOUD_TAU / thick;
    let t0 = (base - o.y) / d.y;
    let dt = (thick / (CLOUD_ROUGH_STEPS as f32 * d.y)).min(CLOUD_MAX_STEP_K * thick);
    let mu = d.dot(sun.dir).clamp(-1.0, 1.0);
    let phase = CLOUD_LOBE_MIX * hg(mu, CLOUD_G_FWD) + (1.0 - CLOUD_LOBE_MIX) * hg(mu, CLOUD_G_BACK);
    let sun_col = sun.e_over_pi * (std::f32::consts::PI) * crate::sky::t_sun_path(sun.dir.y);
    let l_sun = CLOUD_SUN_STEP_K * thick / sun.dir.y.max(0.35);
    let mut t_acc = 1.0_f32;
    let mut sc = Vec3A::ZERO;
    // ONE curl warp for the whole rough march, folded into the ray origin
    // (the along_k pattern) — a reflected sky is seen through the GGX
    // lobe's blur, so per-point warps buy variance, not fidelity.
    let ow = o + curl_offset(o + d * (t0 + 0.5 * dt), cl);
    for k in 0..CLOUD_ROUGH_STEPS {
        let tk = t0 + (k as f32 + 0.5) * dt;
        let p = ow + d * tk;
        // The lo field's finest octave is l0/2 — same anti-alias rule, one
        // attenuation term (the rough march's dt is the largest of all).
        let rho = density_lo_at(p, cl, dt);
        if rho <= 0.0 {
            continue;
        }
        let a = (-(rho * sigma_t * dt)).exp();
        let rho_sun = density_lo_at(p + sun.dir * l_sun, cl, 0.0);
        let t_sun = (-((rho_sun + 0.5 * rho) * sigma_t * l_sun)).exp();
        let s = (sun_col * ((phase + CLOUD_MS) * t_sun) + amb) * CLOUD_ALBEDO;
        sc += s * (t_acc * (1.0 - a));
        t_acc *= a;
    }
    if t_acc >= 1.0 {
        return None;
    }
    let fade = ((d.y - CLOUD_MIN_DY) / CLOUD_FADE_BAND).clamp(0.0, 1.0);
    Some(CloudSample { t: 1.0 + (t_acc - 1.0) * fade, scatter: sc * fade })
}

/// The slab-space cloud-shadow grid row `[org_x, org_z, 1/cell, side]` for a
/// frame: map the scene AABB's 8 corners through the shadow projection
/// `M(p) = p + sun*(base + 0.5*thick - p.y)/sun.y` (the plane
/// `cloud_sun_transmittance` is EXACTLY a function of), bound them, pin the cell
/// to the FIELD resolution (`l0/n`, not the footprint size), derive the side
/// from the footprint and cap it at [`CLOUD_SHADOW_MAX`], then snap the origin
/// to a whole cell so the lattice is FRAME-STATIC (a grid that slid with the
/// camera would move its own interpolation error every frame, which the
/// temporal upscalers read as shimmer). The caller supplies `sun` as
/// `[x, y, z, _]`, `aabb` as content∪ground min/max, and `diag = Clouds::diag`;
/// callers guard `n == 0` / clouds-off themselves (returning a zero row). One
/// source of truth for the grid geometry the GPU tracer, the DXR pipeline, and
/// the `--check` interpolation gates all consume.
pub fn shadow_grid_row(sun: [f32; 4], aabb: ([f32; 3], [f32; 3]), diag: f32, n: u32) -> [f32; 4] {
    let base = CLOUD_BASE_K * diag;
    let thick = CLOUD_THICK_K * diag;
    let sy = sun[1].max(CLOUD_SUN_MIN_Y);
    let (mn, mx) = aabb;
    let (mut lx, mut lz) = (f32::INFINITY, f32::INFINITY);
    let (mut hx, mut hz) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for i in 0..8 {
        let c = [
            if i & 1 == 0 { mn[0] } else { mx[0] },
            if i & 2 == 0 { mn[1] } else { mx[1] },
            if i & 4 == 0 { mn[2] } else { mx[2] },
        ];
        let m = (base + 0.5 * thick - c[1]) / sy;
        let (px, pz) = (c[0] + sun[0] * m, c[2] + sun[2] * m);
        lx = lx.min(px);
        lz = lz.min(pz);
        hx = hx.max(px);
        hz = hz.max(pz);
    }
    let span = (hx - lx).max(hz - lz).max(1e-3);
    let l0 = CLOUD_SCALE_K * diag;
    let mut cell = l0 / n as f32;
    let mut side = (span / cell).ceil() as u32 + 3;
    if side > CLOUD_SHADOW_MAX {
        side = CLOUD_SHADOW_MAX;
        cell = span / (side as f32 - 3.0);
    }
    let org_x = (lx / cell).floor() * cell - cell;
    let org_z = (lz / cell).floor() * cell - cell;
    [org_x, org_z, 1.0 / cell, side as f32]
}

/// Cloud transmittance from a surface point toward the sun — the shadow term.
/// Exactly TWO density evals (the slab's quarter and three-quarter heights
/// along the sun ray), Beer over the slant path. Multiplies the WHOLE direct
/// sun contribution once per `shade()` call, hoisted out of the shadow-sample
/// loop and applied BEFORE the FSR `direct_d`/`direct_s` capture export so
/// the denoiser signals carry it. An unshadowed path returns an EXACT 1.0
/// (and `x * 1.0` is bit-preserving), so a clear-sun pixel is untouched. The
/// shadow eases out over the low-sun band (`CLOUD_SUN_MIN_Y` +
/// CLOUD_FADE_BAND), so a TOD scrub never pops it on or off.
pub fn sun_transmittance(p: Vec3A, sun_dir: Vec3A, cl: &Clouds) -> f32 {
    if !cl.enabled || sun_dir.y <= CLOUD_SUN_MIN_Y {
        return 1.0;
    }
    let base = CLOUD_BASE_K * cl.diag;
    let thick = CLOUD_THICK_K * cl.diag;
    if p.y >= base {
        return 1.0;
    }
    let s_lo = (base + 0.25 * thick - p.y) / sun_dir.y;
    let s_hi = (base + 0.75 * thick - p.y) / sun_dir.y;
    // ONE curl warp at the probes' midpoint, folded into the surface point
    // and shared by both evals — the two probes sit half a slab apart, well
    // inside one curl wavelength, and this is the hot path (two evals per
    // shade() on every lit pixel).
    let pw = p + curl_offset(p + sun_dir * (0.5 * (s_lo + s_hi)), cl);
    let rho =
        0.5 * (density_lo_at(pw + sun_dir * s_lo, cl, 0.0) + density_lo_at(pw + sun_dir * s_hi, cl, 0.0));
    if rho <= 0.0 {
        return 1.0;
    }
    let t = (-(rho * CLOUD_TAU / sun_dir.y.max(0.25))).exp();
    // Ease the shadow out over the low-sun band — the CLOUD_MIN_DY/FADE_BAND
    // pattern, pointed at the light: without it the cutoff was a hard pop
    // (deep Beer shadow through the max(0.25) slant divisor one scrub tick,
    // exact 1.0 the next, while the direct sun still carries most of its
    // irradiance). `fade >= 1` returns the Beer factor VERBATIM — `1 + (t-1)`
    // is not bit-preserving for small t, so the high-sun path must branch,
    // not multiply by a fade of 1.0.
    let fade = ((sun_dir.y - CLOUD_SUN_MIN_Y) / CLOUD_FADE_BAND).clamp(0.0, 1.0);
    if fade >= 1.0 { t } else { 1.0 + (t - 1.0) * fade }
}

/// Closed-form gates, run by `--check`. No rng, no scene, no DLLs.
pub fn self_test() -> Result<(), String> {
    let sun = Sun::new(Vec3A::new(6.0, 10.0, 4.0));
    let diag = 10.0_f32;
    // Explicit states — the self-test never reads the session flag, so it
    // exercises both arms whatever the session wired.
    let on = Clouds::new(true, diag, CLOUD_CHECK_TIME);
    let off = Clouds::off();
    let o = Vec3A::new(0.0, 1.0, 0.0);
    let amb_k = |d: Vec3A| crate::sky::dome(d, sun.dir, 1.0) * CLOUD_AMB_K;

    // A deterministic upper-hemisphere direction sweep (golden spiral).
    let dir_at = |i: usize, n: usize| -> Vec3A {
        let a = i as f32 * 2.399_963;
        let y = (i as f32 + 0.5) / n as f32; // (0, 1) — upward only
        let r = (1.0 - y * y).max(0.0).sqrt();
        Vec3A::new(r * a.cos(), y, r * a.sin())
    };

    // G1: disabled ⇒ BIT-IDENTICAL. radiance through Clouds::off must equal
    // the raw backdrop sum bitwise, and the shadow term must be exactly 1.0 —
    // the --no-clouds lever's whole guarantee.
    for i in 0..2000 {
        let d = dir_at(i, 2000);
        // The off arm must be j-independent too — sweep a phase alongside.
        let j = if i % 2 == 0 { 0.5 } else { 0.137 };
        let r = crate::sky::radiance(o, d, &sun, 5e-4, 1.0, 0.0, 0, &off, j);
        let want = crate::sky::dome(d, sun.dir, 1.0)
            + crate::sky::disc(d, &sun, 5e-4)
            + crate::sky::stars(d, 5e-4, 0.0, 0);
        if r != want {
            return Err(format!("clouds off is not bit-identical at {d:?}: {r:?} vs {want:?}"));
        }
    }
    for i in 0..100 {
        let p = Vec3A::new((i % 10) as f32 - 4.5, 0.0, (i / 10) as f32 - 4.5) * 2.0;
        if sun_transmittance(p, sun.dir, &off).to_bits() != 1.0_f32.to_bits() {
            return Err("clouds off: sun_transmittance is not exactly 1.0".into());
        }
    }

    // G2: range + finiteness over a direction × time sweep, and density stays
    // in [0, 1] (the Beer exponents assume it).
    for &t in &[0.0_f32, CLOUD_CHECK_TIME, 100.0, 3600.0] {
        let cl = Clouds::new(true, diag, t);
        for i in 0..2000 {
            let d = dir_at(i, 2000);
            if let Some(cs) = along(o, d, &sun, amb_k(d), &cl, 0.5) {
                if !(0.0..=1.0).contains(&cs.t) {
                    return Err(format!("cloud T {} out of [0,1] at t={t}, {d:?}", cs.t));
                }
                if !cs.scatter.is_finite() || cs.scatter.min_element() < 0.0 {
                    return Err(format!("cloud scatter {:?} bad at t={t}, {d:?}", cs.scatter));
                }
            }
            let p = o + d * (CLOUD_BASE_K + 0.5 * CLOUD_THICK_K) * diag;
            let rho = density(p, &cl);
            if !(0.0..=1.0).contains(&rho) {
                return Err(format!("density {rho} out of [0,1] at {p:?}"));
            }
        }
    }

    // G3: must-fires — the sky actually HAS clouds at the pinned check time
    // (a broken remap silently producing eternal clear sky must fail here),
    // and some ground point is actually shadowed.
    let mut cloudy = None;
    let mut clear = None;
    for i in 0..2000 {
        let d = dir_at(i, 2000);
        match along(o, d, &sun, amb_k(d), &on, 0.5) {
            Some(cs) if cs.t < 0.5 && cs.scatter.max_element() > 0.0 => cloudy = Some(d),
            None if d.y > CLOUD_MIN_DY + CLOUD_FADE_BAND => clear = Some(d),
            _ => {}
        }
    }
    let cloudy = cloudy.ok_or("no direction reaches T < 0.5 — the sky has no real clouds")?;
    let clear = clear.ok_or("no clear direction — coverage is total, the fast path is dead")?;
    let mut shadowed = false;
    for i in 0..400 {
        // ±3 cloud wavelengths of ground — the field must shadow SOMEWHERE
        // in a few-cloud neighborhood, not necessarily over the origin.
        let p = Vec3A::new((i % 20) as f32 - 9.5, 0.0, (i / 20) as f32 - 9.5)
            * (diag * CLOUD_SCALE_K * 0.3);
        let tc = sun_transmittance(p, sun.dir, &on);
        if !(0.0..=1.0).contains(&tc) {
            return Err(format!("sun_transmittance {tc} out of [0,1]"));
        }
        shadowed |= tc < 0.9;
    }
    if !shadowed {
        return Err("no ground point is cloud-shadowed at the check time".into());
    }

    // G4: a clear direction's backdrop passes through BIT-identically even
    // with clouds enabled (the per-ray None arm), and below the horizon band
    // the layer is structurally absent.
    for d in [clear, Vec3A::new(0.8, CLOUD_MIN_DY * 0.5, 0.6).normalize(), Vec3A::new(0.6, -0.4, 0.7).normalize()] {
        let r = crate::sky::radiance(o, d, &sun, 5e-4, 1.0, 0.0, 0, &on, 0.5);
        let want = crate::sky::dome(d, sun.dir, 1.0)
            + crate::sky::disc(d, &sun, 5e-4)
            + crate::sky::stars(d, 5e-4, 0.0, 0);
        if along(o, d, &sun, amb_k(d), &on, 0.5).is_none() && r != want {
            return Err(format!("cloud-free ray is not bit-identical at {d:?}"));
        }
    }

    // G5: T is monotone non-increasing in optical depth (the Beer identity —
    // a sign error in an exponent flips this by whole factors). Tolerance =
    // the opaque-core break's own quantum: a higher sigma can BREAK the
    // march earlier (first t_acc < 0.005) and land up to that bound ABOVE a
    // lower sigma that marched further — inherent to the discrete cut, not
    // a Beer error, and bounded at 0.005 by construction. From k = 1: a
    // ZERO multiplier correctly degenerates to the None arm (exp(0) = 1 at
    // every step — "no cloud"), which the separate zero-extinction probe
    // below covers.
    let mut prev_t = 1.0_f32 + 1e-6;
    for k in 1..=8 {
        let cs = along_k(o, cloudy, &sun, amb_k(cloudy), &on, 0.5, k as f32 * 0.5)
            .ok_or("monotonicity probe lost its cloud")?;
        if cs.t > prev_t + 5.1e-3 {
            return Err(format!("T not monotone in optical depth: {} after {}", cs.t, prev_t));
        }
        prev_t = cs.t;
    }
    // ...and sig_mul = 0 must transmit everything.
    if let Some(cs) = along_k(o, cloudy, &sun, amb_k(cloudy), &on, 0.5, 0.0) {
        if cs.t < 1.0 - 1e-6 {
            return Err(format!("zero extinction still absorbs: T = {}", cs.t));
        }
    }

    // G8: the erosion bound as a THEOREM — the visible density never exceeds
    // the shadow/lighting field, pointwise, BITWISE, at equal footprint
    // (erosion only subtracts before the shared prof multiply). This is what
    // makes phase A's cover-only occupancy test sound as a bound.
    for i in 0..500 {
        let p = Vec3A::new(
            ((i % 25) as f32 - 12.3) * 2.1,
            (CLOUD_BASE_K + 0.04 * (i % 20) as f32 * CLOUD_THICK_K) * diag,
            ((i / 25) as f32 - 9.6) * 2.7,
        );
        for &w in &[0.0_f32, 1.0, 5.0, 20.0] {
            let hi = density_f(p, &on, w);
            let lo = density_lo_f(p, &on, w);
            if hi > lo {
                return Err(format!(
                    "G8: density {hi} > density_lo {lo} at {p:?} w={w} — erosion added mass"
                ));
            }
        }
    }

    // G9: the dither phase sweep — every phase the real generator can emit
    // keeps T/scatter in range (each sample stays inside its own
    // sub-interval), and the disabled arm is j-independent (bit-identical
    // backdrop whatever the phase). The phases are sourced from `dither_jk`
    // itself over frame × (sample, spp) so the sweep exercises the real
    // per-(pixel, frame, sample) generator, not hand-picked literals.
    for &n in &[0_u32, 1, 7, 0x1234_5678] {
        for &(k, spp) in &[(0_u32, 1_u32), (1, 4), (3, 4), (127, 128)] {
            let j = dither_jk(37, 91, n, k, spp);
            if !(0.0..1.0).contains(&j) {
                return Err(format!("G9: dither_jk out of [0,1): {j} at n={n} k={k}/{spp}"));
            }
            for i in 0..250 {
                let d = dir_at(i * 8, 2000);
                if let Some(cs) = along(o, d, &sun, amb_k(d), &on, j) {
                    if !(0.0..=1.0).contains(&cs.t)
                        || !cs.scatter.is_finite()
                        || cs.scatter.min_element() < 0.0
                    {
                        return Err(format!("G9: bad T/scatter at j={j}, {d:?}"));
                    }
                }
                if along(o, d, &sun, amb_k(d), &off, j).is_some() {
                    return Err("G9: disabled clouds marched anyway".into());
                }
            }
        }
    }
    // G9b: sample 0 rides the frame hash verbatim (stratification adds an
    // exact 0.0), and the frame term actually VARIES the phase — a broken
    // n-mix silently reverting to the static dither must fail here.
    for &(x, y, f, spp) in &[(0_u32, 0_u32, 0_u32, 1_u32), (511, 288, 3, 4), (7, 7, 41, 128)] {
        if dither_jk(x, y, f, 0, spp).to_bits() != dither_j(x, y, f).to_bits() {
            return Err("G9b: dither_jk(k=0) is not bitwise dither_j".into());
        }
    }
    if CLOUD_TEMPORAL_DITHER {
        let mut seen: Vec<u32> = (0..8).map(|n| dither_j(7, 11, n).to_bits()).collect();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() < 7 {
            return Err("G9b: the frame term barely moves the phase — n-mix broken".into());
        }
    }

    // G10: the 3D curl wind field. (a) The analytic Hermite gradient against
    // central differences of vnoise3 itself — the real new math (the du/dt
    // wiring and the difference-bilerp cross terms); mixed tolerance because
    // face-normal components legitimately pass through 0. (b) The
    // soft-normalization bound |v| < 1 and the exact displacement bound
    // |offset| ≤ AMP·diag per axis (y further scaled) — what the march's
    // interval-skip margin leans on. (c) A must-fire: the field actually
    // displaces somewhere (a silently-zero curl must fail).
    let h = 1e-3_f32;
    let mut curl_fired = false;
    for i in 0..96 {
        let q = Vec3A::new(
            ((i % 8) as f32 - 3.3) * 0.73,
            (((i / 8) % 4) as f32 - 1.6) * 1.19,
            ((i / 32) as f32 - 0.9) * 2.41,
        );
        for oct in [6_u32, 7] {
            let g = vnoise3_grad(q, oct);
            let fd = Vec3A::new(
                (vnoise3(q + Vec3A::new(h, 0.0, 0.0), oct) - vnoise3(q - Vec3A::new(h, 0.0, 0.0), oct)) / (2.0 * h),
                (vnoise3(q + Vec3A::new(0.0, h, 0.0), oct) - vnoise3(q - Vec3A::new(0.0, h, 0.0), oct)) / (2.0 * h),
                (vnoise3(q + Vec3A::new(0.0, 0.0, h), oct) - vnoise3(q - Vec3A::new(0.0, 0.0, h), oct)) / (2.0 * h),
            );
            let (g, fd) = ([g.x, g.y, g.z], [fd.x, fd.y, fd.z]);
            for a in 0..3 {
                let (ga, fa) = (g[a], fd[a]);
                if (ga - fa).abs() > 2e-2 * fa.abs().max(1.0) {
                    return Err(format!(
                        "G10a: analytic gradient off FD at q={q:?} oct={oct} axis {a}: {ga} vs {fa}"
                    ));
                }
            }
        }
        let p = q * (CLOUD_CURL_SCALE_K * diag) + Vec3A::new(0.0, CLOUD_BASE_K * diag, 0.0);
        let ofs = curl_offset(p, &on);
        let amp = CLOUD_CURL_AMP_K * diag;
        if !ofs.is_finite()
            || ofs.x.abs() > amp
            || ofs.z.abs() > amp
            || ofs.y.abs() > amp * CLOUD_CURL_YSCALE
        {
            return Err(format!("G10b: curl offset {ofs:?} exceeds its bound at {p:?}"));
        }
        curl_fired |= ofs.length() > 1e-3 * diag;
    }
    if !curl_fired {
        return Err("G10c: the curl field never displaces — the wind field is dead".into());
    }

    // G6: the advection identity — the field at time t IS the t=0 field
    // translated by wind·t, exactly (one shared `advect` expression), and the
    // wind is actually nonzero.
    let w = wind(diag);
    if w.length_squared() <= 0.0 {
        return Err("the wind is zero — nothing will ever pass overhead".into());
    }
    for i in 0..200 {
        let p = Vec3A::new(
            ((i % 20) as f32 - 9.7) * 1.7,
            (CLOUD_BASE_K + 0.4) * diag,
            ((i / 20) as f32 - 4.6) * 2.3,
        );
        let a = advect(p, &on);
        let b = advect(p, &Clouds::new(true, diag, 0.0)) - w * on.time;
        if a.x.to_bits() != b.x.to_bits() || a.y.to_bits() != b.y.to_bits() {
            return Err(format!("advection identity broke at {p:?}: {a:?} vs {b:?}"));
        }
    }

    // G11: the sun-elevation shadow fade — exactly 1.0 at/below
    // CLOUD_SUN_MIN_Y (the bit-identity arm), and ~1 just above it, so a TOD
    // scrub crossing the cutoff can never pop a ground shadow (the fade's
    // whole job). G3 above re-proves a fully faded-in sun still finds a real
    // shadow, so the fade cannot silently kill the term.
    for i in 0..400 {
        let a = i as f32 * 0.7853;
        let y = CLOUD_SUN_MIN_Y + 1e-3;
        let r = (1.0 - y * y).sqrt();
        let pg = Vec3A::new((i % 20) as f32 - 9.5, 0.0, (i / 20) as f32 - 9.5)
            * (diag * CLOUD_SCALE_K * 0.3);
        let tc = sun_transmittance(pg, Vec3A::new(r * a.cos(), y, r * a.sin()), &on);
        if !(0.98..=1.0).contains(&tc) {
            return Err(format!("G11: shadow {tc} pops just above CLOUD_SUN_MIN_Y (fade broken)"));
        }
        let at = sun_transmittance(
            pg,
            Vec3A::new(r * a.cos(), CLOUD_SUN_MIN_Y, r * a.sin()),
            &on,
        );
        if at.to_bits() != 1.0_f32.to_bits() {
            return Err("G11: at/below CLOUD_SUN_MIN_Y is not exactly 1.0".into());
        }
    }

    // G12: the AVX2 corner-hash lanes are BIT-EQUAL to the scalar hashes —
    // the whole soundness argument for the SIMD path (same values, fewer
    // instructions ⇒ every other gate, the HLSL twin, and the replay/
    // same-seed contracts are untouched by construction). Sweep includes
    // negative cells (the i32→u32 wrap), zero, and large magnitudes. On a
    // non-AVX2 CPU the dispatch takes the scalar arm and this compares it to
    // itself — vacuous there, load-bearing on every machine we ship numbers
    // from.
    for n in 0..2000_u32 {
        let m = crate::sky::pcg_mix(n.wrapping_mul(0x0019_660D));
        let i = (m & 0xFFFF) as i32 - 0x8000 + if n % 5 == 0 { 1_000_000 } else { 0 };
        let j = ((m >> 8) & 0xFFFF) as i32 - 0x8000;
        let k = ((m >> 16) & 0x7FFF) as i32 - 0x4000;
        let oct = n % 8;
        let simd = corner_hashes3(i, j, k, oct);
        let scal = corner_hashes3_scalar(i, j, k, oct);
        for l in 0..8 {
            if simd[l].to_bits() != scal[l].to_bits() {
                return Err(format!(
                    "G12: corner_hashes3 lane {l} diverges from scalar at ({i},{j},{k}) oct {oct}: {} vs {}",
                    simd[l], scal[l]
                ));
            }
        }
        let simd2 = corner_hashes(i, j, oct);
        let scal2 = [
            cell_hash(i, j, oct),
            cell_hash(i + 1, j, oct),
            cell_hash(i, j + 1, oct),
            cell_hash(i + 1, j + 1, oct),
        ];
        for l in 0..4 {
            if simd2[l].to_bits() != scal2[l].to_bits() {
                return Err(format!(
                    "G12: corner_hashes lane {l} diverges from scalar at ({i},{j}) oct {oct}"
                ));
            }
        }
    }

    // G13: the slab-space cloud-shadow grid geometry (shadow_grid_row) — the
    // pure-Rust half of the cloud-shadow cache gates (the GPU fill/fetch consume
    // exactly this row). Coverage, the pinned cell, the cap, and the FRAME-
    // STATIC origin snap.
    {
        let base = CLOUD_BASE_K * diag;
        let thick = CLOUD_THICK_K * diag;
        let l0 = CLOUD_SCALE_K * diag;
        // Every projected AABB corner must land inside the grid WITH its four
        // bilinear neighbours (else a fetch edge-clamps to a wrong cell).
        let covered = |row: [f32; 4], aabb: ([f32; 3], [f32; 3]), sd: Vec3A| -> bool {
            let (org_x, org_z, inv_cell, side) = (row[0], row[1], row[2], row[3] as u32);
            let sy = sd.y.max(CLOUD_SUN_MIN_Y);
            (0..8).all(|i| {
                let c = [
                    if i & 1 == 0 { aabb.0[0] } else { aabb.1[0] },
                    if i & 2 == 0 { aabb.0[1] } else { aabb.1[1] },
                    if i & 4 == 0 { aabb.0[2] } else { aabb.1[2] },
                ];
                let m = (base + 0.5 * thick - c[1]) / sy;
                let (gx, gz) = ((c[0] + sd.x * m - org_x) * inv_cell, (c[2] + sd.z * m - org_z) * inv_cell);
                gx >= 0.0 && gx <= (side - 1) as f32 && gz >= 0.0 && gz <= (side - 1) as f32
            })
        };
        let aabb = ([-5.0_f32, 0.0, -5.0], [5.0_f32, 3.0, 5.0]);
        for &n in &[8u32, 16, 32, 64] {
            let row = shadow_grid_row([sun.dir.x, sun.dir.y, sun.dir.z, 0.0], aabb, diag, n);
            let cell = 1.0 / row[2];
            if row[3] as u32 >= CLOUD_SHADOW_MAX {
                return Err(format!("G13: an ordinary sun hit the side cap at n={n}"));
            }
            // uncapped ⇒ cell is EXACTLY l0/n (the field-resolution pin)
            if (cell - l0 / n as f32).abs() > 1e-4 * cell {
                return Err(format!("G13: cell {cell} != l0/n {} at n={n}", l0 / n as f32));
            }
            // origin snapped to a whole cell (FRAME-STATIC anchoring)
            for &o in &[row[0], row[1]] {
                if ((o / cell).round() - o / cell).abs() > 1e-3 {
                    return Err(format!("G13: origin {o} not cell-snapped (cell {cell}) at n={n}"));
                }
            }
            if !covered(row, aabb, sun.dir) {
                return Err(format!("G13: a projected AABB corner falls outside the grid at n={n}"));
            }
        }
        // The cap: a grazing sun over a tall box forces side == CLOUD_SHADOW_MAX,
        // and the cap must GROW the cell (never break coverage).
        let tall = ([-5.0_f32, 0.0, -5.0], [5.0_f32, 60.0, 5.0]);
        let low = Sun::new(Vec3A::new(1.0, 0.05, 0.0));
        let row = shadow_grid_row([low.dir.x, low.dir.y, low.dir.z, 0.0], tall, diag, 64);
        if row[3] as u32 != CLOUD_SHADOW_MAX {
            return Err(format!("G13: grazing sun over a tall box did not hit the cap (side {})", row[3]));
        }
        if !covered(row, tall, low.dir) {
            return Err("G13: the capped grid lost coverage (cap did not grow the cell)".into());
        }
    }

    // G14: the cloud-shadow interpolation-error probe — the executable form of
    // the "owed large-cloudy-scene probe". The domain reduction to F(M.x, M.z)
    // is EXACT (trace_common.hlsli); only the bilinear FETCH approximates. On a
    // check scene (smaller than one field feature 1.3*diag) the shadow is nearly
    // constant, so the error is ~0 and gating there is vacuous — this synthetic
    // grid deliberately spans SEVERAL features so the shadow varies (anti-
    // vacuity) and the fetch's real worst case shows.
    //
    // THE NUMBER: at the shipped N=16 the worst-case midpoint error is ~0.066 —
    // and it is SCALE-INVARIANT (cell = l0/N, feature = l0/2, so cell/feature =
    // 2/N regardless of CLOUD_SCALE_K or diag). So a scene LARGER than one cloud
    // feature sees ~6.6% error in the sun-transmittance term at a cloud edge
    // (a soft penumbra gradient — the accepted cost of the cache; raise --cloud-
    // shadow N to shrink it). The MEAN is far smaller (edges are sparse). The
    // gate is a REGRESSION bound: a cell-mapping / grid-plumbing bug lands the
    // error at 0.5+, an order past this. Do NOT tighten it toward the ~0 that
    // only the vacuous check-scene regime produces.
    {
        let n = 16u32;
        let aabb = ([-30.0_f32, 0.0, -30.0], [30.0_f32, 5.0, 30.0]);
        let sd = Sun::new(Vec3A::new(3.0, 6.0, 2.0)).dir;
        let sy = sd.y.max(CLOUD_SUN_MIN_Y);
        let m0 = (CLOUD_BASE_K * diag + 0.5 * CLOUD_THICK_K * diag) / sy;
        let row = shadow_grid_row([sd.x, sd.y, sd.z, 0.0], aabb, diag, n);
        let (org_x, org_z, cell, side) = (row[0], row[1], 1.0 / row[2], row[3] as u32);
        let mut best_var = 0.0_f32;
        let (mut worst, mut mean, mut mean_n) = (0.0_f32, 0.0_f64, 0u32);
        for &t in &[CLOUD_CHECK_TIME, 50.0, 150.0, 300.0] {
            let cl = Clouds::new(true, diag, t);
            // F(mx, mz): the exact transmittance of the y=0 point projecting there.
            let f = |mx: f32, mz: f32| -> f32 {
                sun_transmittance(Vec3A::new(mx - sd.x * m0, 0.0, mz - sd.z * m0), sd, &cl)
            };
            let (mut lo, mut hi) = (1.0_f32, 0.0_f32);
            let (mut w, mut m_sum, mut m_cnt) = (0.0_f32, 0.0_f64, 0u32);
            for j in 2..side - 2 {
                for i in 2..side - 2 {
                    let (mx, mz) = (org_x + i as f32 * cell, org_z + j as f32 * cell);
                    let bil = 0.25
                        * (f(mx, mz) + f(mx + cell, mz) + f(mx, mz + cell) + f(mx + cell, mz + cell));
                    let exact = f(mx + 0.5 * cell, mz + 0.5 * cell);
                    let e = (bil - exact).abs();
                    w = w.max(e);
                    m_sum += e as f64;
                    m_cnt += 1;
                    let c = f(mx, mz);
                    lo = lo.min(c);
                    hi = hi.max(c);
                }
            }
            if hi - lo > best_var {
                best_var = hi - lo;
                worst = w;
                mean = m_sum;
                mean_n = m_cnt;
            }
        }
        let mean = if mean_n > 0 { mean / mean_n as f64 } else { 0.0 };
        // Anti-vacuity: the shadow must genuinely vary across the grid at SOME
        // sampled time, else the gate proves nothing.
        if best_var < 0.05 {
            return Err(format!(
                "G14: cloud shadow never varies across the grid ({best_var}) — the interpolation probe is vacuous"
            ));
        }
        // Regression bound (~2x the measured 0.066 worst). A wiring bug blows
        // this to 0.5+; the honest N=16 fetch error stays well under it.
        if worst > 0.15 {
            return Err(format!(
                "G14: cloud-shadow bilinear WORST error {worst} exceeds 0.15 (mean {mean:.4}) — grid mapping / fetch regression (do NOT widen; fix the wiring)"
            ));
        }
        eprintln!("clouds G14: N=16 cloud-shadow fetch error worst {worst:.4} / mean {mean:.4} over a multi-feature grid (var {best_var:.3})");
    }

    // G7: the horizon fade is continuous — just above CLOUD_MIN_DY the layer
    // must be nearly invisible (fade ≈ 0), so the skip below it is no seam.
    let d_lo = Vec3A::new(0.9, CLOUD_MIN_DY + 1e-4, 0.43).normalize();
    if let Some(cs) = along(o, d_lo, &sun, amb_k(d_lo), &on, 0.5) {
        if (cs.t - 1.0).abs() > 0.02 || cs.scatter.max_element() > 0.02 {
            return Err(format!(
                "horizon fade discontinuous: T {} scatter {:?} just above the band",
                cs.t, cs.scatter
            ));
        }
    }

    eprintln!(
        "clouds self-test: OK (cloudy dir T<0.5 exists, clear dir exists, ground shadow fires, \
         wind {:.3}/s per unit diag)",
        wind(1.0).length()
    );
    Ok(())
}
