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

/// Quantize a [0,1] value through an n-bit UNORM channel.
#[inline(always)]
pub fn quant_unorm(v: f32, bits: u32) -> f32 {
    let m = ((1u32 << bits) - 1) as f32;
    (v.clamp(0.0, 1.0) * m + 0.5).floor() / m
}

// ---------------------------------------------------------------------------
// Signal split + composite (the correctness core; the composite identity is
// gated by --check-fsr and mirrored by gpu/shaders/composite.hlsl).
// ---------------------------------------------------------------------------

pub struct Signals {
    /// Demodulated direct diffuse, f16-quantized (the stored/wire value).
    pub dd: Vec3A,
    /// Demodulated direct specular, f16-quantized.
    pub ds: Vec3A,
    /// Exact f32 remainder: everything the pixel shows that is not the two
    /// remodulated signals (kd*ambient, the reflection bounce, all
    /// quantization slop). Passed through the denoiser untouched.
    pub residual: Vec3A,
}

/// Split a shaded primary hit into the two Ray Regeneration signals plus the
/// exact residual. `direct_d`/`direct_s` are `PrimarySurface`'s post-average
/// lobes; `color` is the pixel's final shaded color; `kd`/`f0` are the same
/// diffuse/specular albedos the G-buffer write site derives
/// (`albedo*(1-metallic)` and `lerp(0.04, albedo, metallic)`). The identity
/// `composite(sig, kd, f0) == color` holds to f32 rounding by construction
/// (the residual is defined as the remainder of the exact wire-value
/// products), and `albedo_wire3`'s leading f16 rounding makes it insensitive
/// to whether the caller hands raw f32 or the GBufs' f16-stored albedos.
pub fn split_signals(color: Vec3A, direct_d: Vec3A, direct_s: Vec3A, kd: Vec3A, f0: Vec3A) -> Signals {
    let kd_wire = albedo_wire3(kd);
    let sf0_wire = albedo_wire3(f0);
    let dd = q16v(direct_d);
    let ds = q16v(direct_s / sf0_wire.max(Vec3A::splat(MIN_SPEC_ALB)));
    let residual = color - dd * kd_wire - ds * sf0_wire;
    Signals { dd, ds, residual }
}

/// The remodulation the GPU composite pass performs (with denoised dd/ds in
/// place of the raw signals): the CPU twin used by the identity gate. The
/// wire quantization is applied here, exactly as `split_signals` applied it —
/// composite.hlsl mirrors this decode.
pub fn composite(sig: &Signals, kd: Vec3A, f0: Vec3A) -> Vec3A {
    sig.dd * albedo_wire3(kd) + sig.ds * albedo_wire3(f0) + sig.residual
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
        for (k, r) in [sig.residual.x, sig.residual.y, sig.residual.z].into_iter().enumerate() {
            self.residual[i * 3 + k].store(r.to_bits(), Relaxed);
        }
        self.prev_z[i].store(prev_z.to_bits(), Relaxed);
    }

    /// Read back (dd, ds, residual) for the check gates.
    pub fn read(&self, x: usize, y: usize) -> Signals {
        let i = y * self.rw + x;
        let l16 = |v: &AtomicU16| f16::from_bits(v.load(Relaxed)).to_f32();
        let l32 = |v: &AtomicU32| f32::from_bits(v.load(Relaxed));
        Signals {
            dd: Vec3A::new(l16(&self.dd[i * 3]), l16(&self.dd[i * 3 + 1]), l16(&self.dd[i * 3 + 2])),
            ds: Vec3A::new(l16(&self.ds[i * 3]), l16(&self.ds[i * 3 + 1]), l16(&self.ds[i * 3 + 2])),
            residual: Vec3A::new(
                l32(&self.residual[i * 3]),
                l32(&self.residual[i * 3 + 1]),
                l32(&self.residual[i * 3 + 2]),
            ),
        }
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
