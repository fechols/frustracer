//! CPU-side FSR (Ray Regeneration + FSR4) plumbing: the per-pixel signal
//! buffers captured at primary-hit time, the demodulation/composite math, and
//! the encoding conventions of the ffx wire formats. The FFX analog of
//! `dlss.rs`; pure CPU, exercised headlessly by `--check-fsr`. FSR mode
//! reuses `dlss::GBufs` for normal/roughness, albedos, depth and motion
//! vectors — this module adds only what Ray Regeneration needs beyond that.
//!
//! Conventions (from the vendored ffx_denoiser.h / ffx_upscale.h):
//! - denoiser MV: RG = PreviousUV − CurrentUV, B = prevZ − curZ. Our
//!   `GBufs::mvec` is pixels y-down current→previous (= prev_px − cur_px), so
//!   RG is a pure (1/rw, 1/rh) scale — same direction, same y-axis, no flip.
//! - depth: "signed linear depth" — our positive-forward view-Z times
//!   `DEPTH_SIGN`.
//! - normals: RG octahedral + roughness B + material type A in RGB10A2.
//! - albedos: RGB8 sqrt-encoded (`sqrt` then 8-bit quantize); diffuse =
//!   `BaseColor * (1 - Metalness)` — exactly our `kd`.
//! - jitter: screen pixels, `JITTER_SIGN` times the renderer's sample offset.

use glam::Vec3A;
use half::f16;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering::Relaxed};

// ---------------------------------------------------------------------------
// Undocumented polarities — settled empirically on RDNA4 hardware, the same
// way xess::JITTER_SIGN and the SL jitter negation were (wrong jitter sign =
// static image wobbles at 2× the jitter amplitude; wrong MV polarity =
// directional smear under motion). They live HERE and nowhere else.
// ---------------------------------------------------------------------------

/// Jitter reported to BOTH ffx dispatches = JITTER_SIGN * the renderer's
/// sample offset (ffx_denoiser.h says "the subpixel jitter offset applied to
/// the camera", in pixels — start unnegated, unlike SL).
pub const JITTER_SIGN: f32 = 1.0;

/// Sign of the linear view-Z handed to the denoiser ("signed linear depth").
pub const DEPTH_SIGN: f32 = 1.0;

/// Per-axis sign of the upscaler's motionVectorScale. The upscaler shares the
/// denoiser's MV plane (UV-delta, prev − cur, y-down); its motionVectorScale
/// is set to (sign.x * rw, sign.y * rh) to hand FSR pixel-space MVs of the
/// same polarity. Flip an axis here if motion smears directionally on
/// hardware while the denoiser (which takes {1,1,1}) looks right.
pub const UPSCALE_MV_SIGN: (f32, f32) = (1.0, 1.0);

/// Ray Regeneration material type (A channel of the normals plane, 2 bits).
/// Semantics are provider-defined ("see docs"); 0 is the default class and
/// the first cut uses it for every surface.
pub const MAT_TYPE: f32 = 0.0;

/// Specular-albedo floor for demodulation: `ds = direct_s / max(F0, MIN)`.
/// Remodulation multiplies the un-floored wire F0 back, and the residual is
/// the exact remainder, so the floor affects only how the signal is
/// distributed between `ds` and `residual` — never the composite.
pub const MIN_SPEC_ALB: f32 = 1e-4;

// ---------------------------------------------------------------------------
// Wire encoders (pure; round-tripped by --check-fsr).
// ---------------------------------------------------------------------------

/// Saturating f32 -> f16: clamps into the finite f16 range instead of
/// overflowing to ±inf (the OIDN color-narrowing discipline). An inf on a
/// signal plane is fatal here: the demodulated `ds` can reach direct_s/1e-4
/// when a colored metal's wire F0 channel is 0 (the MIN_SPEC_ALB floor), and
/// an inf ds turns the residual remainder into inf·0 = NaN.
#[inline(always)]
pub fn f16_sat(v: f32) -> f16 {
    f16::from_f32(v.clamp(f16::MIN.to_f32(), f16::MAX.to_f32()))
}

/// f32 -> f16 -> f32 (round-to-nearest-even, saturating): the quantization
/// every f16 plane (dd/ds signals, GBufs albedo storage) applies.
#[inline(always)]
pub fn q16(v: f32) -> f32 {
    f16_sat(v).to_f32()
}

#[inline(always)]
pub fn q16v(v: Vec3A) -> Vec3A {
    Vec3A::new(q16(v.x), q16(v.y), q16(v.z))
}

/// sqrt-encode an albedo channel to the RGBA8 wire byte.
#[inline(always)]
pub fn sqrt_encode8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0).sqrt() * 255.0 + 0.5) as u8
}

/// Decode the RGBA8 wire byte back to linear (what the GPU composite and the
/// denoiser reconstruct).
#[inline(always)]
pub fn sqrt_decode8(b: u8) -> f32 {
    let s = b as f32 / 255.0;
    s * s
}

/// The exact linear albedo value the GPU side reconstructs for a CPU-side
/// f32: through f16 storage (GBufs), then the sqrt-encoded 8-bit wire.
#[inline(always)]
pub fn albedo_wire(v: f32) -> f32 {
    sqrt_decode8(sqrt_encode8(q16(v)))
}

#[inline(always)]
pub fn albedo_wire3(v: Vec3A) -> Vec3A {
    Vec3A::new(albedo_wire(v.x), albedo_wire(v.y), albedo_wire(v.z))
}

/// The GPU pack's wire chain for the sqrt-encoded RGBA8 albedo planes: ONE
/// explicit 8-bit quantization of the raw f32. Unlike `albedo_wire` there is
/// no leading f16 rounding — the GPU G-buffer pack stores f32, so the wire
/// quantization happens exactly once, at the feed. The HLSL twin is
/// trace_common.hlsli's `sqrt_wire`, used identically at the pack's sig
/// demodulation, the feed's residual, and the composite decode; the GPU
/// FSR-RR feed gates use this over the pack readback as their oracle.
#[inline(always)]
pub fn sqrt_wire(v: f32) -> f32 {
    sqrt_decode8(sqrt_encode8(v))
}

/// Unused from Rust (the CPU capture demodulates per channel inline); kept as
/// the mirror of trace_common.hlsli's live `sqrt_wire3`, so the wire-helper
/// surfaces stay 1:1 across the twins.
#[allow(dead_code)]
#[inline(always)]
pub fn sqrt_wire3(v: Vec3A) -> Vec3A {
    Vec3A::new(sqrt_wire(v.x), sqrt_wire(v.y), sqrt_wire(v.z))
}

#[inline(always)]
fn sign_nonzero(v: f32) -> f32 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}

/// Octahedral-encode a unit normal into [0,1]² (the RG of the RGB10A2
/// normals plane).
pub fn oct_encode(n: Vec3A) -> (f32, f32) {
    let inv = 1.0 / (n.x.abs() + n.y.abs() + n.z.abs()).max(1e-12);
    let (mut u, mut v) = (n.x * inv, n.y * inv);
    if n.z < 0.0 {
        let (ou, ov) = (u, v);
        u = (1.0 - ov.abs()) * sign_nonzero(ou);
        v = (1.0 - ou.abs()) * sign_nonzero(ov);
    }
    (u * 0.5 + 0.5, v * 0.5 + 0.5)
}

/// Inverse of `oct_encode` (self-test roundtrip only).
pub fn oct_decode(eu: f32, ev: f32) -> Vec3A {
    let (u, v) = (eu * 2.0 - 1.0, ev * 2.0 - 1.0);
    let z = 1.0 - u.abs() - v.abs();
    let (x, y) = if z < 0.0 {
        ((1.0 - v.abs()) * sign_nonzero(u), (1.0 - u.abs()) * sign_nonzero(v))
    } else {
        (u, v)
    };
    Vec3A::new(x, y, z).normalize()
}

/// The normal as the RGB10A2 normals plane actually STORES it: octahedral,
/// 10 bits per channel, decoded back.
///
/// This exists because the AO signal's remodulation factor stopped being a
/// constant. It used to be `shade::AMBIENT`, which every composite site could
/// simply be handed; the one sky makes it `sky_sh.irradiance(n)` — directional,
/// hence per-pixel. The GPU composite pass (`fsr_composite.hlsl`) has exactly
/// one source for that normal: the plane bytes. So every site must derive the
/// factor from the WIRE normal, not from the full-precision shading normal, or
/// the composite identity picks up a quantization-sized hole that no tolerance
/// should be widened to absorb. `--check-fsr`'s per-pixel identity gate is what
/// pins this.
pub fn wire_normal(n: Vec3A) -> Vec3A {
    let (u, v) = oct_encode(n);
    oct_decode(quant_unorm(u, 10), quant_unorm(v, 10))
}

/// Quantize a [0,1] value through an n-bit UNORM channel.
#[inline(always)]
pub fn quant_unorm(v: f32, bits: u32) -> f32 {
    let m = ((1u32 << bits) - 1) as f32;
    (v.clamp(0.0, 1.0) * m + 0.5).floor() / m
}

// ---------------------------------------------------------------------------
// Signal split + composite (the correctness core; the composite identity is
// gated by --check-fsr and mirrored by shaders/composite.hlsl).
// ---------------------------------------------------------------------------

pub struct Signals {
    /// Demodulated direct diffuse, f16-quantized (the stored/wire value).
    pub dd: Vec3A,
    /// Demodulated direct specular, f16-quantized.
    pub ds: Vec3A,
    /// Ambient-occlusion open fraction in [0,1], f16-quantized — RR's scalar
    /// AO signal. Remodulated as `ao * amb * kd`, where `amb` is the sky's SH
    /// irradiance at the pixel's WIRE normal (`wire_normal`) — the factor the
    /// sampled ambient tier multiplies the open fraction by.
    pub ao: f32,
    /// Demodulated indirect (reflection-bounce) specular, f16-quantized —
    /// divided by the same wire F0 as `ds`, so it remodulates identically.
    pub is: Vec3A,
    /// Exact f32 remainder: everything the pixel shows that is not the four
    /// remodulated signals (emissive, the glass transmission chain, all
    /// quantization slop). Passed through the denoiser untouched — which is
    /// why every NOISY term belongs in a signal, not here.
    pub residual: Vec3A,
}

/// Split a shaded primary hit into the four Ray Regeneration signals plus the
/// exact residual. `direct_d`/`direct_s`/`ao`/`ind_s` are `PrimarySurface`'s
/// captures; `color` is the pixel's final shaded color; `kd`/`f0` are the same
/// diffuse/specular albedos the G-buffer write site derives
/// (`albedo*(1-metallic)` and `lerp(0.04, albedo, metallic)`). The identity
/// `composite(sig, kd, f0) == color` holds to f32 rounding by construction
/// (the residual is defined as the remainder of the exact wire-value
/// products), and `albedo_wire3`'s leading f16 rounding makes it insensitive
/// to whether the caller hands raw f32 or the GBufs' f16-stored albedos.
///
/// The wire `kd` is the EFFECTIVE diffuse albedo `albedo*(1-metallic)*
/// (1-transmission)` (`PrimarySurface::diff_albedo`) — the exact factor
/// `color` multiplies its diffuse terms by. It used to be the raw
/// `albedo*(1-metallic)` with the remainder absorbing the difference, which
/// meant every denoiser delta on `dd`/`ao` remodulated at `kd` instead of the
/// physical `kd*(1-transmission)` — a 33x amplifier on water (transmission
/// 0.97) that smeared terrain-colored diffuse bleed across the surface in
/// FSR-RR sessions. The remaining wire approximations are sheen's
/// `(1-0.157*sheen)` energy factor and translucency's `(1-tl)` split (which
/// does NOT scale ambient, so it cannot be folded into a kd shared by the
/// `dd` and `ao` signals); both land in the residual, both zero on water.
///
/// `amb` is the AO signal's remodulation factor — `sky_sh.irradiance` at the
/// pixel's `wire_normal`. It is passed in rather than computed here so fsr.rs
/// stays sky-agnostic, but it MUST be the wire-normal value: the GPU composite
/// has only the normals plane to rebuild it from (see `wire_normal`).
pub fn split_signals(
    color: Vec3A,
    direct_d: Vec3A,
    direct_s: Vec3A,
    ao: f32,
    ind_s: Vec3A,
    kd: Vec3A,
    f0: Vec3A,
    amb: Vec3A,
) -> Signals {
    let kd_wire = albedo_wire3(kd);
    let sf0_wire = albedo_wire3(f0);
    let f0_floor = sf0_wire.max(Vec3A::splat(MIN_SPEC_ALB));
    let dd = q16v(direct_d);
    let ds = q16v(direct_s / f0_floor);
    let ao = q16(ao);
    let is = q16v(ind_s / f0_floor);
    let residual = color - dd * kd_wire - ds * sf0_wire - ao * amb * kd_wire - is * sf0_wire;
    Signals { dd, ds, ao, is, residual }
}

/// The remodulation the GPU composite pass performs (with the denoised
/// signals in place of the raw ones): the CPU twin used by the identity gate.
/// The wire quantization is applied here, exactly as `split_signals` applied
/// it — fsr_composite.hlsl mirrors this decode.
pub fn composite(sig: &Signals, kd: Vec3A, f0: Vec3A, amb: Vec3A) -> Vec3A {
    let kd_wire = albedo_wire3(kd);
    let f0_wire = albedo_wire3(f0);
    sig.dd * kd_wire + sig.ds * f0_wire + sig.ao * amb * kd_wire + sig.is * f0_wire + sig.residual
}

// ---------------------------------------------------------------------------
// Per-pixel signal buffers (FSR mode only).
// ---------------------------------------------------------------------------

/// The planes Ray Regeneration needs beyond `dlss::GBufs`, at the same
/// tile-disjoint relaxed-atomic discipline. `dd`/`ds` store f16 bits (their
/// wire is RGBA16F — storage IS the wire precision); `residual` stays f32 so
/// the composite identity survives storage exactly (its wire rounding to
/// RGBA16F happens at upload, bounded by the check's wire gate); `prev_z`
/// stays f32 like the depth plane it differences against (MV B channel).
/// Allocated once at the dynamic-resolution range max; `set_res` reinterprets
/// in place on a res step (the `GBufs::set_res` contract: contents stale
/// until the next full-depth frame rewrites every pixel).
pub struct FsrBufs {
    pub rw: usize,
    pub rh: usize,
    /// 3/px demodulated direct diffuse (f16 bits).
    pub dd: Vec<AtomicU16>,
    /// 3/px demodulated direct specular (f16 bits).
    pub ds: Vec<AtomicU16>,
    /// 1/px ambient-occlusion open fraction (f16 bits).
    pub ao: Vec<AtomicU16>,
    /// 3/px demodulated indirect (reflection) specular (f16 bits).
    pub is: Vec<AtomicU16>,
    /// 3/px exact residual (f32 bits).
    pub residual: Vec<AtomicU32>,
    /// 1/px previous-frame linear view-Z (f32 bits) — the exact hit point
    /// through the previous camera; `far` on sky (delta 0 by construction).
    pub prev_z: Vec<AtomicU32>,
}

impl FsrBufs {
    pub fn new(rw: usize, rh: usize) -> Self {
        let a16 = |n: usize| (0..n).map(|_| AtomicU16::new(0)).collect();
        let a32 = |n: usize| (0..n).map(|_| AtomicU32::new(0)).collect();
        Self {
            rw,
            rh,
            dd: a16(rw * rh * 3),
            ds: a16(rw * rh * 3),
            ao: a16(rw * rh),
            is: a16(rw * rh * 3),
            residual: a32(rw * rh * 3),
            prev_z: a32(rw * rh),
        }
    }

    /// Reinterpret at a different logical resolution within the construction
    /// capacity (dynamic render res; see `GBufs::set_res`).
    pub fn set_res(&mut self, rw: usize, rh: usize) {
        assert!(rw * rh * 3 <= self.dd.len(), "FsrBufs::set_res beyond capacity");
        self.rw = rw;
        self.rh = rh;
    }

    #[inline(always)]
    pub fn write(&self, x: usize, y: usize, sig: &Signals, prev_z: f32) {
        let i = y * self.rw + x;
        for (k, (d, s)) in [(sig.dd.x, sig.ds.x), (sig.dd.y, sig.ds.y), (sig.dd.z, sig.ds.z)]
            .into_iter()
            .enumerate()
        {
            crate::dlss::st16(&self.dd[i * 3 + k], d);
            crate::dlss::st16(&self.ds[i * 3 + k], s);
        }
        crate::dlss::st16(&self.ao[i], sig.ao);
        for (k, v) in [sig.is.x, sig.is.y, sig.is.z].into_iter().enumerate() {
            crate::dlss::st16(&self.is[i * 3 + k], v);
        }
        for (k, r) in [sig.residual.x, sig.residual.y, sig.residual.z].into_iter().enumerate() {
            self.residual[i * 3 + k].store(r.to_bits(), Relaxed);
        }
        self.prev_z[i].store(prev_z.to_bits(), Relaxed);
    }

    /// Read back every signal for the check gates.
    pub fn read(&self, x: usize, y: usize) -> Signals {
        let i = y * self.rw + x;
        let l16 = |v: &AtomicU16| f16::from_bits(v.load(Relaxed)).to_f32();
        let l32 = |v: &AtomicU32| f32::from_bits(v.load(Relaxed));
        let v3 = |b: &Vec<AtomicU16>| Vec3A::new(l16(&b[i * 3]), l16(&b[i * 3 + 1]), l16(&b[i * 3 + 2]));
        Signals {
            dd: v3(&self.dd),
            ds: v3(&self.ds),
            ao: l16(&self.ao[i]),
            is: v3(&self.is),
            residual: Vec3A::new(
                l32(&self.residual[i * 3]),
                l32(&self.residual[i * 3 + 1]),
                l32(&self.residual[i * 3 + 2]),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Denoiser tuning (the `--fsr-*` A/B levers).
// ---------------------------------------------------------------------------

/// Runtime overrides for the Ray Regeneration tuning constants
/// (`FfxApiConfigureDenoiserKey`, applied once at context creation through
/// `ffxConfigureDescDenoiserKeyValue`). Every field is `None` by default,
/// which configures NOTHING — a flagless session runs the provider's own
/// defaults, exactly as it did before these levers existed. `max_radiance` is
/// the firefly clamp, the one that matters most to a 1-spp path tracer.
#[derive(Clone, Copy, Default, Debug)]
pub struct DenoiseTuning {
    pub normal_strength: Option<f32>,
    pub stability_bias: Option<f32>,
    pub max_radiance: Option<f32>,
    pub radiance_clip_k: Option<f32>,
    pub kernel_relaxation: Option<f32>,
    pub disocclusion_threshold: Option<f32>,
}

impl DenoiseTuning {
    /// The (key, name, value) triples to configure — key ids straight from
    /// `FfxApiConfigureDenoiserKey` in ffx_denoiser.h.
    pub fn entries(&self) -> Vec<(u64, &'static str, f32)> {
        [
            (1u64, "cross-bilateral-normal-strength", self.normal_strength),
            (2, "stability-bias", self.stability_bias),
            (3, "max-radiance", self.max_radiance),
            (4, "radiance-clip-std-k", self.radiance_clip_k),
            (5, "gaussian-kernel-relaxation", self.kernel_relaxation),
            (6, "disocclusion-threshold", self.disocclusion_threshold),
        ]
        .into_iter()
        .filter_map(|(k, n, v)| v.map(|v| (k, n, v)))
        .collect()
    }

    pub fn any(&self) -> bool {
        !self.entries().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Dynamic-resolution range derivation (pure; used when the live upscaler
// query is unavailable — headless runs — and gated by --check-fsr).
// ---------------------------------------------------------------------------

/// FSR quality-mode per-dimension ratios (ffx_upscale.h enum comments).
pub const RATIO_QUALITY: f32 = 1.5;
pub const RATIO_ULTRA_PERFORMANCE: f32 = 3.0;

/// Deterministic stand-in for ffxQueryDescUpscaleGetRenderResolutionFromQualityMode:
/// floor(display / ratio), min 1. Quality seeds the ScaleCtl start;
/// UltraPerformance floors the dynamic range; the range max is the window.
pub fn fallback_render_res(out: (usize, usize), ratio: f32) -> (usize, usize) {
    (
        ((out.0 as f32 / ratio) as usize).max(1),
        ((out.1 as f32 / ratio) as usize).max(1),
    )
}

// ---------------------------------------------------------------------------
// The FSR3 upscale INPUT TRIO, staged into raw rows (pure; gated by
// --check-fsr on every platform).
//
// FSR 3.1 upscale-only takes exactly three planes — RGBA16F linear HDR colour,
// RG16F pixel-space current->previous y-down motion vectors, and R32F
// reversed-Z clip depth — and the encodings are a property of the WIRE, not of
// the graphics API. `gpu/ffx_up.rs::record_upload` is the D3D12 recording of
// them and is `cfg(windows)`; `src/mtl/` is the Metal one. This is the MATH
// both share, which is the split CLAUDE.md prescribes (share math and
// vocabulary, duplicate recording) and the reason there is no `trait Upscaler`.
//
// `pitch` is a parameter for exactly one reason: D3D12 upload heaps require
// 256-byte-aligned rows (`aligned_pitch`) while Metal's `replaceRegion:` takes
// an arbitrary `bytesPerRow`. That is the ONLY difference between the two
// backends' uploads.
//
// The Windows path deliberately still carries its own copy of these loops:
// folding it onto these functions would edit the shipping D3D12 renderer, which
// is what `check.png` guards. Consolidating them is a follow-on, not this
// change.
//
// THE ROW CASTS HAVE AN ALIGNMENT PRECONDITION, and moving these loops off the
// D3D12 path is what made it a caller's problem rather than a structural fact.
// Each row is reinterpreted as `[f16; N]` or `f32` in place, so `dst`'s base
// AND `pitch` must both carry the element alignment; `record_upload` got both
// for free from a mapped upload heap (256-aligned base, `aligned_pitch` rows)
// and had nothing to state. Here `dst` is an arbitrary slice — a `Vec<u8>` has
// layout alignment 1 — and `pitch` is arbitrary too, because Metal's
// `bytesPerRow` has no alignment rule at all where `aligned_pitch` had a
// 256-byte one. Both are cheap to assert and neither was.
// ---------------------------------------------------------------------------

/// The row-cast precondition for the three staging encoders, in one place so
/// the three cannot drift. Debug-only, like the bounds check it sits beside:
/// every caller is ours, and this is a per-frame path.
#[inline(always)]
fn debug_check_rows(dst: &[u8], pitch: usize, rw: usize, rh: usize, bpp: usize, align: usize) {
    debug_assert!(pitch >= rw * bpp, "pitch {pitch} < row {} B", rw * bpp);
    debug_assert!(dst.len() >= pitch * rh, "buffer {} B < {} B", dst.len(), pitch * rh);
    debug_assert_eq!(
        dst.as_ptr() as usize % align,
        0,
        "staging buffer base is not {align}-aligned — the row cast is UB"
    );
    debug_assert_eq!(pitch % align, 0, "pitch {pitch} is not {align}-aligned — row 1 onward is UB");
}

/// Linear HDR colour -> RGBA16F. Saturating, never `+inf` (the wire discipline
/// every f16 colour plane in this codebase follows); alpha is exactly 0.
pub fn stage_color(dst: &mut [u8], pitch: usize, accum: &[AtomicU32], rw: usize, rh: usize) {
    use rayon::prelude::*;
    debug_check_rows(dst, pitch, rw, rh, 8, std::mem::align_of::<f16>());
    dst.par_chunks_mut(pitch).take(rh).enumerate().for_each(|(y, row)| {
        let px: &mut [[f16; 4]] =
            unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, rw) };
        for (x, p) in px.iter_mut().enumerate() {
            let i = (y * rw + x) * 3;
            let ld = |k: usize| f32::from_bits(accum[i + k].load(Relaxed));
            p[0] = f16_sat(ld(0));
            p[1] = f16_sat(ld(1));
            p[2] = f16_sat(ld(2));
            p[3] = f16::from_f32(0.0);
        }
    });
}

/// Motion vectors -> RG16F. A BIT COPY: `GBufs::mvec` already stores f16 in the
/// plane's own pixel-space current->previous y-down convention, so re-encoding
/// it would only add rounding. The polarity rides `UPSCALE_MV_SIGN` at dispatch
/// (`motionVectorScale`), never here.
pub fn stage_mvec(dst: &mut [u8], pitch: usize, mvec: &[AtomicU16], rw: usize, rh: usize) {
    use rayon::prelude::*;
    debug_check_rows(dst, pitch, rw, rh, 4, std::mem::align_of::<f16>());
    dst.par_chunks_mut(pitch).take(rh).enumerate().for_each(|(y, row)| {
        let px: &mut [[f16; 2]] =
            unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, rw) };
        for (x, p) in px.iter_mut().enumerate() {
            let i = (y * rw + x) * 2;
            p[0] = f16::from_bits(mvec[i].load(Relaxed));
            p[1] = f16::from_bits(mvec[i + 1].load(Relaxed));
        }
    });
}

/// Linear view-Z -> [0,1] reversed-Z clip depth, through the ONE encoder
/// (`xess::view_z_to_clip_depth`) the XeSS path already single-sources. The
/// context is created with `DEPTH_INVERTED` to match, and sky's `view_z = far`
/// lands on exactly 0.0 — a contract `--check-xess` already gates and this
/// module's own self-test re-asserts.
pub fn stage_depth(
    dst: &mut [u8],
    pitch: usize,
    depth: &[AtomicU32],
    rw: usize,
    rh: usize,
    near: f32,
    far: f32,
) {
    use rayon::prelude::*;
    debug_check_rows(dst, pitch, rw, rh, 4, std::mem::align_of::<f32>());
    dst.par_chunks_mut(pitch).take(rh).enumerate().for_each(|(y, row)| {
        let px: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, rw) };
        for (x, p) in px.iter_mut().enumerate() {
            let z = f32::from_bits(depth[y * rw + x].load(Relaxed));
            *p = crate::xess::view_z_to_clip_depth(z, near, far);
        }
    });
}

/// The staging gates. Pure, DLL-free, GPU-free — so they run inside
/// `--check-fsr` on Windows and Linux as well as macOS, which is the point: a
/// transcription drift between the two backends' uploads is caught wherever
/// anyone builds, not only where the Metal one runs.
pub fn stage_self_test() -> Result<(), String> {
    const SENTINEL: u8 = 0xa5;
    let (near, far) = (0.17f32, 340.0f32);

    // A native and an ODD resolution — the pair `fsr_frame_check` uses, because
    // an odd width is what catches a row loop that assumed pitch == rw*bpp.
    for &(rw, rh) in &[(64usize, 48usize), (37usize, 29usize)] {
        let n = rw * rh;
        // Every 11th component is deliberately PAST the f16 range, both signs.
        // Without them nothing in this probe exercises `f16_sat`'s clamp, and
        // the expectation below is computed with `f16_sat` too — so a swap to
        // the plain `f16::from_f32` (which returns ±inf out of range) would
        // change both sides together and pass. The over-range pixels get their
        // own oracle-free assertion instead.
        let accum: Vec<AtomicU32> = (0..n * 3)
            .map(|i| {
                let v = match i % 11 {
                    0 => 1.0e30,
                    5 => -1.0e30,
                    _ => i as f32 * 0.37 - 2.0,
                };
                AtomicU32::new(v.to_bits())
            })
            .collect();
        let mvec: Vec<AtomicU16> = (0..n * 2)
            .map(|i| AtomicU16::new(f16::from_f32(i as f32 * 0.011 - 1.5).to_bits()))
            .collect();
        // Deliberately includes exact `far` (sky) and values NEARER than
        // `near`. The sub-near case needs its own arm: the ramp starts at 0.05
        // but steps by 0.31, so every index past 0 clears `near` — and index 0
        // is the sky arm, which is why the first draft's "values below near"
        // were claimed, exercised by nothing, and only found once the
        // `near_seen` counter below made the omission fail the gate.
        let depth: Vec<AtomicU32> = (0..n)
            .map(|i| {
                let z = match i % 7 {
                    0 => far,
                    3 => near * 0.5,
                    _ => 0.05 + (i as f32) * 0.31,
                };
                AtomicU32::new(z.to_bits())
            })
            .collect();

        // Over-wide pitch and over-tall buffer: everything past the sub-rect
        // must survive, which is what `renderSize` < the allocation depends on.
        let pitch_c = rw * 8 + 16;
        let pitch_m = rw * 4 + 12;
        let pitch_d = rw * 4 + 8;
        let rows = rh + 2;
        let mut c = vec![SENTINEL; pitch_c * rows];
        let mut m = vec![SENTINEL; pitch_m * rows];
        let mut d = vec![SENTINEL; pitch_d * rows];
        stage_color(&mut c, pitch_c, &accum, rw, rh);
        stage_mvec(&mut m, pitch_m, &mvec, rw, rh);
        stage_depth(&mut d, pitch_d, &depth, rw, rh, near, far);

        let (mut sky_seen, mut near_seen, mut over_seen) = (false, false, 0usize);
        for y in 0..rh {
            for x in 0..rw {
                let i = y * rw + x;
                // colour
                let o = y * pitch_c + x * 8;
                for k in 0..3 {
                    let got = f16::from_bits(u16::from_le_bytes([c[o + k * 2], c[o + k * 2 + 1]]));
                    let src = f32::from_bits(accum[i * 3 + k].load(Relaxed));
                    let want = f16_sat(src);
                    if got.to_bits() != want.to_bits() {
                        return Err(format!(
                            "stage_color {rw}x{rh} px({x},{y}).{k}: {got} != {want}"
                        ));
                    }
                    // THE SATURATION CONTRACT, stated without reference to
                    // `f16_sat`: an out-of-range radiance must land on the
                    // finite ceiling of its own sign, never ±inf. An inf here
                    // is not cosmetic — it propagates through FSR3's history
                    // and poisons every pixel the accumulation pass touches.
                    if src.abs() > f16::MAX.to_f32() {
                        over_seen += 1;
                        if !got.is_finite()
                            || got.to_f32().abs() != f16::MAX.to_f32()
                            || got.is_sign_negative() != src.is_sign_negative()
                        {
                            return Err(format!(
                                "stage_color {rw}x{rh} px({x},{y}).{k}: {src:e} encoded to \
                                 {got} — must saturate to the f16 ceiling of its own sign"
                            ));
                        }
                    }
                }
                if u16::from_le_bytes([c[o + 6], c[o + 7]]) != 0 {
                    return Err(format!("stage_color {rw}x{rh} px({x},{y}): alpha is not 0"));
                }
                // mvec — a BIT copy, so compare bits, not values (NaN-safe too)
                let o = y * pitch_m + x * 4;
                for k in 0..2 {
                    let got = u16::from_le_bytes([m[o + k * 2], m[o + k * 2 + 1]]);
                    let want = mvec[i * 2 + k].load(Relaxed);
                    if got != want {
                        return Err(format!(
                            "stage_mvec {rw}x{rh} px({x},{y}).{k}: {got:#06x} != {want:#06x} \
                             (must be a bit copy, not a re-encode)"
                        ));
                    }
                }
                // depth
                let o = y * pitch_d + x * 4;
                let got = f32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]);
                let z = f32::from_bits(depth[i].load(Relaxed));
                let want = crate::xess::view_z_to_clip_depth(z, near, far);
                if got.to_bits() != want.to_bits() {
                    return Err(format!("stage_depth {rw}x{rh} px({x},{y}): {got} != {want}"));
                }
                // THE RANGE CONTRACT, stated without reference to the encoder:
                // FSR reads this plane as normalized device depth, so a value
                // outside [0,1] is a wire violation however it arose. Asserted
                // against `got` rather than checked inside
                // `view_z_to_clip_depth` because the compare above is
                // self-referential — it re-runs the same function — and would
                // pass unchanged if that clamp were ever dropped.
                if !(0.0..=1.0).contains(&got) {
                    return Err(format!(
                        "stage_depth {rw}x{rh} px({x},{y}): view_z {z} encoded to {got}, \
                         outside the [0,1] normalized-depth range"
                    ));
                }
                if z == far {
                    sky_seen = true;
                    // THE REVERSED-Z CONTRACT: sky must land on exactly 0.0, or
                    // FSR reads the whole background as the near plane.
                    if got != 0.0 {
                        return Err(format!(
                            "stage_depth: sky (view_z == far) encoded to {got}, not exactly 0.0"
                        ));
                    }
                }
                // The near end of the same contract, and the reason the probe
                // carries sub-`near` values at all: reversed-Z puts the near
                // plane at 1.0, and anything closer is clamped there rather
                // than allowed past it. Without this the "values below `near`"
                // in the probe were exercised but never scored.
                //
                // WHAT IT DOES AND DOES NOT PIN, measured rather than assumed:
                // `view_z_to_clip_depth` clamps twice — `view_z.max(near)` and
                // then `[0,1]` on the result — and EITHER alone lands sub-near
                // input on exactly 1.0, so this fires only if both are lost. It
                // is a pin on the WIRE contract (what FSR reads at the near
                // plane), not on which line supplies it. The `[0,1]` check
                // above is the one with independent teeth: dropping the outer
                // clamp sends `z > far` negative and fires it, while the
                // equality compare — which re-runs the same function — passes.
                if z < near {
                    near_seen = true;
                    if got != 1.0 {
                        return Err(format!(
                            "stage_depth: view_z {z} is nearer than near={near} and encoded to \
                             {got}, not clamped to exactly 1.0"
                        ));
                    }
                }
            }
        }
        if !sky_seen {
            return Err("stage_self_test: no sky pixel in the probe — the 0.0 pin is vacuous".into());
        }
        if !near_seen {
            return Err(
                "stage_self_test: no sub-near view_z in the probe — the near-clamp pin is vacuous"
                    .into(),
            );
        }
        if over_seen == 0 {
            return Err(
                "stage_self_test: no out-of-range colour in the probe — the saturation pin \
                 is vacuous, and `f16_sat` would be indistinguishable from `f16::from_f32`"
                    .into(),
            );
        }

        // The sub-rect discipline: pitch padding and the rows past `rh` are
        // never touched, so one allocation at the range max can serve every
        // smaller render resolution.
        for (name, buf, pitch, bpp) in [
            ("color", &c, pitch_c, 8usize),
            ("mvec", &m, pitch_m, 4),
            ("depth", &d, pitch_d, 4),
        ] {
            for y in 0..rows {
                let tail = if y < rh { rw * bpp } else { 0 };
                for x in tail..pitch {
                    if buf[y * pitch + x] != SENTINEL {
                        return Err(format!(
                            "stage_{name} {rw}x{rh}: wrote outside the sub-rect at row {y} byte {x}"
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Provider selection (pure; gated by --check-fsr). FSR3.1 and FSR4 are the
// SAME ffx-api effect — the provider is a per-context version choice
// (ffxOverrideVersion), so "FSR3 support" is a pick over the enumeration
// ffxQueryDescGetVersions returns, not a different SDK.
// ---------------------------------------------------------------------------

/// Which FSR pipeline a session runs. `Fsr4Rr` is the full decoupled-signal
/// path (Ray Regeneration denoise -> composite -> FSR4 upscale, RDNA4 only);
/// `Fsr3` is upscale-only on the FSR 3.1 provider — no denoiser context, no
/// signal split, cross-vendor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Flavor {
    Fsr4Rr,
    Fsr3,
}

impl Flavor {
    /// The log/toggle display name. Single-sourced: every message site reads
    /// these instead of re-deriving the pair (a hardcoded title-bar "FSR4"
    /// shipped wrong for FSR3 sessions once).
    pub fn label(self) -> &'static str {
        match self {
            Flavor::Fsr4Rr => "Ray Regeneration + FSR4",
            Flavor::Fsr3 => "FSR 3.1 upscale-only",
        }
    }

    /// The title-bar short form + its denoiser suffix.
    pub fn hud(self) -> (&'static str, &'static str) {
        match self {
            Flavor::Fsr4Rr => ("FSR4", " + RayRegen"),
            Flavor::Fsr3 => ("FSR3", ""),
        }
    }
}

/// Every "major.minor.patch" triple in a provider display name, in order (the
/// names are not a documented contract — "FSR 3.1.5", "FidelityFX FSR 3.1.5"
/// and a bare "3.1.5" all parse, and a name embedding a second version, e.g.
/// "AMD 24.10.1 FSR 3.1.5", yields both so the pick can key on the right
/// major instead of whichever triple happens to come first; init prints every
/// (id, name) so a mismatch on new hardware is diagnosable in one run).
pub fn parse_provider_versions(name: &str) -> Vec<(u32, u32, u32)> {
    name.split_whitespace()
        .filter_map(|tok| {
            let tok = tok.trim_start_matches(|c: char| !c.is_ascii_digit());
            let mut it = tok.split('.').map(|p| {
                let end = p.find(|c: char| !c.is_ascii_digit()).unwrap_or(p.len());
                p[..end].parse::<u32>().ok()
            });
            match (it.next(), it.next(), it.next()) {
                (Some(Some(a)), Some(Some(b)), Some(Some(c))) => Some((a, b, c)),
                _ => None,
            }
        })
        .collect()
}

/// Choose the upscaler provider for a session. `upscalers` is the
/// versions(true) enumeration (non-empty — init errors out before this on an
/// empty one); `rr_available` says whether the Ray Regeneration (denoiser)
/// enumeration was non-empty; `force_fsr3` is the --fsr3 lever. Returns
/// (version_id, flavor) where version_id 0 means "no ffxOverrideVersion
/// chained" — the provider default, the original FSR4 create path
/// bit-for-bit.
///
/// - forced: the highest 3.x provider, or None (forced-but-absent fails
///   loudly at init — never silently un-force).
/// - RR available: (0, Fsr4Rr). The RR provider is itself the RDNA4/FSR4
///   signal and id 0 needs no name at all, so this is deliberately NOT gated
///   on parsing a 4.x display name (the names are not a contract; a driver
///   renaming its FSR4 provider must not silently downgrade the session to
///   3.1). If the default create still fails, init falls back loudly.
/// - otherwise: the highest 3.x provider or None. FSR2 is never picked.
pub fn pick_version(upscalers: &[(u64, String)], rr_available: bool, force_fsr3: bool) -> Option<(u64, Flavor)> {
    if !force_fsr3 && rr_available {
        return Some((0, Flavor::Fsr4Rr));
    }
    upscalers
        .iter()
        .filter_map(|(id, name)| {
            parse_provider_versions(name).into_iter().filter(|v| v.0 == 3).max().map(|v| (*id, v))
        })
        .max_by_key(|&(_, v)| v)
        .map(|(id, _)| (id, Flavor::Fsr3))
}

/// Choose the FRAME GENERATION provider. `fg_versions` is the device-filtered
/// framegeneration enumeration (empty = FG unsupported, the caller already
/// bailed); `fsr4_session` says whether the session's UPSCALER resolved to the
/// FSR4 family (family coherence: an FSR4-RR session prefers the 4.x ML frame
/// generation, everything else — FSR3, XeSS-fed, cross-vendor — prefers the
/// proven 3.1 interpolation). The preferred major may simply not be there
/// (4.x FG is RDNA4-gated the way RR is; a 3.1-only drop enumerates no 4.x),
/// so the other major is the fallback rather than a failure — the enumeration
/// is device-filtered, anything in it is claimed to run here. Returns
/// (version_id, display_name); version_id 0 (provider default + API pin) is
/// deliberately NOT used — FG picks are always explicit overrides, because
/// "default" would float with the drop while the session's flavor pairing is
/// the contract we print and test against.
pub fn pick_fg_version(fg_versions: &[(u64, String)], fsr4_session: bool) -> Option<(u64, String)> {
    let best_of = |major: u32| {
        fg_versions
            .iter()
            .filter_map(|(id, name)| {
                parse_provider_versions(name)
                    .into_iter()
                    .filter(|v| v.0 == major)
                    .max()
                    .map(|v| (*id, name.clone(), v))
            })
            .max_by_key(|&(_, _, v)| v)
            .map(|(id, name, _)| (id, name))
    };
    let (preferred, fallback) = if fsr4_session { (4, 3) } else { (3, 4) };
    best_of(preferred).or_else(|| best_of(fallback))
}

