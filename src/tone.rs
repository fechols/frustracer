//! The presentation curve — the single source of truth for turning linear HDR
//! radiance into display pixels, for every present arm on both the CPU and the
//! GPU.
//!
//! Before this module the curve `(1 - exp(-c))^(1/2.2)` was open-coded in three
//! places (`render::present_px`, `gpu/shaders/tonemap.hlsl`, and the screenshot
//! readback). All three now route here, and the HLSL is a term-for-term port
//! gated against this math by `--check-gpu` — the `nppd.hlsl` / `feed.hlsl`
//! twin-gate pattern.
//!
//! # The curve
//!
//! Everything is expressed in units of **paper white** (the luminance a linear
//! radiance of 1.0 should display at — the scene is authored so 1.0 ≈ diffuse
//! white; see `scene::default_light`). With a knee `k` and headroom
//! `w = peak_nits / paper_white`:
//!
//! ```text
//! f(x) = x                                            for x <= k
//! f(x) = k + (w - k) * (1 - exp(-(x - k) / (w - k)))  for x >  k
//! ```
//!
//! Monotone, C¹ at the knee (the derivative is exactly 1 on both sides), and
//! asymptotic to `w` — it never exceeds the display's peak.
//!
//! The load-bearing property: at `k = 0, w = 1` this collapses to
//! `1 - exp(-x)`, **the curve the renderer has always shipped**. SDR is not a
//! separate path, it is the degenerate case — which is what lets one curve serve
//! the 8-bit swapchain and the scRGB one without a behavioural fork.
//! `self_test` gates that degeneracy bit-for-bit; it is the regression guard
//! that the SDR default did not move.
//!
//! # Encodings
//!
//! - **SDR** (`B8G8R8A8_UNORM` swapchain): `k=0, w=1`, then `^(1/2.2)`. The
//!   buffer is display-encoded, so the gamma is ours to apply.
//! - **scRGB** (`R16G16B16A16_FLOAT` + `DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709`):
//!   linear (G10) — **never** apply gamma, the display pipeline owns the EOTF.
//!   scRGB is *defined* so that 1.0 = 80 nits, hence `scale = paper_white / 80`.
//!   Values above 1.0 are legal and are exactly what buys us the highlights.
//! - **HDR10** (`R10G10B10A2_UNORM` + `DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020`):
//!   the same rolloff, then the Rec.709 → Rec.2020 gamut matrix and the SMPTE
//!   ST 2084 (PQ) inverse EOTF, whose signal is normalized so encoded 1.0 =
//!   10000 nits — hence `scale = paper_white / 10000`. This is the arm the
//!   swapchain-wrapper FG families take (DLSS-G and XeSS-FG both reject the
//!   scRGB fp16 swapchain but accept 10-bit PQ, the format HDR titles ship).
//!
//! An SDR display under an scRGB swapchain is therefore not a special case
//! either — it is `paper_white = 80, w = 1`, i.e. the same `1 - exp(-x)` rolloff
//! landing in [0, 1]. That is why moving the window between an HDR and an SDR
//! monitor is a two-float retune and not a swapchain rebuild.

use glam::Vec3A;

/// scRGB's definition: a value of 1.0 is 80 nits. Not tunable — it is the
/// colour space, not a preference.
pub const SCRGB_NITS: f32 = 80.0;

/// Default paper white under `--hdr`. 200 nits is the usual desktop-HDR
/// reference; `--hdr-paper-white` overrides.
pub const DEFAULT_PAPER_WHITE: f32 = 200.0;

/// Where the rolloff starts in HDR: exactly at paper white, so everything up to
/// diffuse white is reproduced *exactly* and only the highlights are compressed.
const HDR_KNEE: f32 = 1.0;

/// PQ's definition: the ST 2084 signal is normalized so encoded 1.0 = 10000
/// nits. Not tunable — it is the transfer function, not a preference.
pub const PQ_MAX_NITS: f32 = 10000.0;

// SMPTE ST 2084 constants — exact dyadic rationals, so the literals are exact
// in f32 and identical in the HLSL twin by construction (the clouds-wind
// literal idiom). Note c1 + c2 == 1 + c3 == 19.6875 exactly, which is what
// makes `pq_encode(1.0) == 1.0` bitwise (a self_test anchor).
const PQ_M1: f32 = 0.1593017578125; // 2610 / 16384
const PQ_M2: f32 = 78.84375; //        2523 / 4096 * 128
const PQ_C1: f32 = 0.8359375; //       3424 / 4096
const PQ_C2: f32 = 18.8515625; //      2413 / 4096 * 32
const PQ_C3: f32 = 18.6875; //         2392 / 4096 * 32

/// Rec.709 → Rec.2020 primaries (BT.2087 matrix). Row sums are 1 to ~1e-6, so
/// white maps to white — the self_test anchor. Literals mirrored term-for-term
/// in `tonemap.hlsl`/`hud.hlsl`; change both together.
#[inline]
pub fn m709_to_2020(v: Vec3A) -> Vec3A {
    Vec3A::new(
        0.627404 * v.x + 0.329283 * v.y + 0.043313 * v.z,
        0.069097 * v.x + 0.919540 * v.y + 0.011362 * v.z,
        0.016391 * v.x + 0.088013 * v.y + 0.895595 * v.z,
    )
}

/// SMPTE ST 2084 inverse EOTF: display luminance normalized to 10000 nits
/// (`y` in [0,1]) → the PQ-encoded signal. Saturates its input — the curve is
/// only defined on [0,1], and the rolloff already bounded the value at the
/// display's own peak, so the clamp is belt-and-braces, not a tone decision.
#[inline]
pub fn pq_encode(y: f32) -> f32 {
    let y = y.clamp(0.0, 1.0);
    let yp = y.powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * yp) / (1.0 + PQ_C3 * yp)).powf(PQ_M2)
}

/// The display encode applied after the rolloff. Which one is a property of
/// the negotiated swapchain, never of the scene.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToneMode {
    /// Linear output (scRGB) — the display pipeline owns the EOTF.
    Linear,
    /// `^(1/2.2)` display encode — the 8-bit UNORM swapchain.
    Gamma22,
    /// Rec.709→2020 matrix + ST 2084 (PQ) — the HDR10 R10G10B10A2 swapchain.
    Pq,
}

/// Parameters of the presentation curve. Pure data — the GPU gets these as root
/// constants, the CPU passes them to `map`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ToneParams {
    /// Rolloff knee, in paper-white units. Below this the response is exact.
    pub knee: f32,
    /// Asymptote, in paper-white units: `peak_nits / paper_white`. Always >= 1.
    pub headroom: f32,
    /// Output scale applied after the curve (scRGB: `paper_white / 80`;
    /// HDR10: `paper_white / 10000`).
    pub scale: f32,
    /// The display encode (see `ToneMode`). `Gamma22` only for the 8-bit UNORM
    /// swapchain — an scRGB buffer is linear and the display applies the EOTF
    /// itself; `Pq` for the HDR10 swapchain.
    pub mode: ToneMode,
}

impl ToneParams {
    /// The 8-bit SDR swapchain. Reproduces the pre-HDR renderer bit-for-bit.
    pub const SDR: ToneParams =
        ToneParams { knee: 0.0, headroom: 1.0, scale: 1.0, mode: ToneMode::Gamma22 };

    /// scRGB output on a display whose HDR is *off* (or unknown). Same rolloff
    /// as `SDR`, but linear-encoded and anchored at scRGB's own 80-nit white —
    /// so the displayed image matches `SDR` while the buffer stays f16.
    pub const SCRGB_SDR: ToneParams =
        ToneParams { knee: 0.0, headroom: 1.0, scale: 1.0, mode: ToneMode::Linear };

    /// scRGB output on an HDR display. `peak_nits` is what the display actually
    /// reports (see `gpu::hdr`), `paper_white` is where linear 1.0 lands.
    ///
    /// A display with no real headroom (`peak <= paper`) degenerates to the
    /// `SCRGB_SDR` rolloff rather than hard-clipping at the knee — a zero-width
    /// rolloff band is a divide-by-zero, and clipping would look worse than the
    /// curve we already ship.
    pub fn hdr(paper_white: f32, peak_nits: f32) -> ToneParams {
        let paper = paper_white.max(1.0);
        let scale = paper / SCRGB_NITS;
        let headroom = peak_nits / paper;
        if headroom <= HDR_KNEE {
            return ToneParams { scale, ..ToneParams::SCRGB_SDR };
        }
        ToneParams { knee: HDR_KNEE, headroom, scale, mode: ToneMode::Linear }
    }

    /// HDR10/PQ output — the arm the swapchain-wrapper FG families take. Same
    /// rolloff as `hdr()`, but the scale anchors paper white against PQ's
    /// 10000-nit signal ceiling and the encode is matrix + ST 2084. The same
    /// no-headroom degeneracy applies (a `peak <= paper` display gets the SDR
    /// rolloff, PQ-encoded).
    pub fn hdr10(paper_white: f32, peak_nits: f32) -> ToneParams {
        let paper = paper_white.max(1.0);
        let scale = paper / PQ_MAX_NITS;
        let headroom = peak_nits / paper;
        if headroom <= HDR_KNEE {
            return ToneParams { knee: 0.0, headroom: 1.0, scale, mode: ToneMode::Pq };
        }
        ToneParams { knee: HDR_KNEE, headroom, scale, mode: ToneMode::Pq }
    }

    /// Peak luminance this parameterisation will actually emit, in nits.
    pub fn peak_nits(&self) -> f32 {
        let anchor = if self.mode == ToneMode::Pq { PQ_MAX_NITS } else { SCRGB_NITS };
        self.headroom * self.scale * anchor
    }
}

/// The rolloff, one channel. `x` must be non-negative (`map` clamps).
#[inline]
fn curve(x: f32, knee: f32, headroom: f32) -> f32 {
    if x <= knee {
        return x;
    }
    let band = headroom - knee;
    knee + band * (1.0 - (-(x - knee) / band).exp())
}

/// Linear HDR radiance -> a **display-referred, paper-white-relative** value:
/// the curve and the gamma, but NOT the scale. 1.0 here means "paper white"
/// whatever the encoding, which is the domain the debug overlay's [0,1] tints
/// are authored in — so the CPU present path composites the overlay between
/// `shape` and `encode`, exactly where the pre-HDR renderer did.
///
/// The clamp to zero is what makes the branch safe: a negative radiance would
/// take the `x <= knee` arm and then hit `powf` with a negative base (NaN). It
/// changes nothing for real input — radiance is non-negative, and the old code
/// clamped the same values away at the 8-bit pack.
#[inline]
pub fn shape(c: Vec3A, p: ToneParams) -> Vec3A {
    let c = c.max(Vec3A::ZERO);
    let f = Vec3A::new(
        curve(c.x, p.knee, p.headroom),
        curve(c.y, p.knee, p.headroom),
        curve(c.z, p.knee, p.headroom),
    );
    if p.mode == ToneMode::Gamma22 {
        f.powf(1.0 / 2.2)
    } else {
        f
    }
}

/// Paper-white-relative display value -> the value the swapchain wants. For
/// scRGB that is the 80-nit anchor; for the 8-bit path `scale` is 1.0 and this
/// is the identity; for HDR10 the scale lands in PQ's 10000-nit-normalized
/// domain and the value goes through the gamut matrix + ST 2084 encode.
#[inline]
pub fn encode(v: Vec3A, p: ToneParams) -> Vec3A {
    match p.mode {
        ToneMode::Pq => {
            let n = m709_to_2020(v * p.scale);
            Vec3A::new(pq_encode(n.x), pq_encode(n.y), pq_encode(n.z))
        }
        _ => v * p.scale,
    }
}

/// The whole pipeline, no overlay — what the tonemap PS computes, and what the
/// HLSL twin is gated against.
#[inline]
pub fn map(c: Vec3A, p: ToneParams) -> Vec3A {
    encode(shape(c, p), p)
}

/// Closed-form gates on the curve. Hooked by `--check`; the HLSL twin is gated
/// against `map` itself by `--check-gpu`.
pub fn self_test() -> Result<(), String> {
    // (1) SDR degeneracy — THE regression guard. `ToneParams::SDR` must
    // reproduce the pre-HDR curve bit-for-bit, or the default renderer moved.
    for i in 0..=2000 {
        let x = i as f32 * 0.01; // 0 ..= 20, well past where the curve saturates
        let c = Vec3A::splat(x);
        let want = (Vec3A::ONE - (-c).exp()).powf(1.0 / 2.2);
        let got = map(c, ToneParams::SDR);
        if got.x.to_bits() != want.x.to_bits() {
            return Err(format!(
                "SDR degeneracy: c={x} -> {} (bits {:#x}), pre-HDR curve gives {} (bits {:#x})",
                got.x,
                got.x.to_bits(),
                want.x,
                want.x.to_bits()
            ));
        }
    }

    // (2) The scRGB-on-an-SDR-display params are the same rolloff, ungamma'd
    // and inside [0, 1] — that is what makes a monitor move a retune.
    for i in 0..=500 {
        let x = i as f32 * 0.04;
        let got = map(Vec3A::splat(x), ToneParams::SCRGB_SDR).x;
        let want = 1.0 - (-x).exp();
        if got.to_bits() != want.to_bits() {
            return Err(format!("SCRGB_SDR: c={x} -> {got}, want {want}"));
        }
        if !(0.0..=1.0).contains(&got) {
            return Err(format!("SCRGB_SDR: c={x} -> {got} outside [0,1]"));
        }
    }

    let p = ToneParams::hdr(200.0, 1000.0);
    if p.headroom != 5.0 {
        return Err(format!("hdr(200, 1000) headroom {} != 5", p.headroom));
    }

    // (3) Paper-white anchor: linear 1.0 displays at exactly paper_white nits.
    // This is the whole point of the knee sitting at 1.0 — diffuse white is
    // reproduced, not rolled off.
    let white = map(Vec3A::ONE, p).x * SCRGB_NITS;
    if (white - 200.0).abs() > 1e-3 {
        return Err(format!("paper-white anchor: linear 1.0 -> {white} nits, want 200"));
    }
    // ... and everything below the knee is exact, not merely close.
    for i in 0..=100 {
        let x = i as f32 * 0.01;
        let got = map(Vec3A::splat(x), p).x;
        let want = x * p.scale;
        if (got - want).abs() > 1e-6 {
            return Err(format!("below-knee response not exact: c={x} -> {got}, want {want}"));
        }
    }

    // (4) Monotone, and asymptotic to the headroom without ever reaching it —
    // the display's peak is a bound the curve may approach, never exceed.
    let mut prev = f32::NEG_INFINITY;
    for i in 0..=100_000 {
        let x = i as f32 * 0.01; // 0 ..= 1000: radiance far past any real scene
        let f = curve(x, p.knee, p.headroom);
        if f < prev {
            return Err(format!("curve not monotone at c={x}: {f} < {prev}"));
        }
        if f > p.headroom {
            return Err(format!("curve overshoots headroom at c={x}: {f} > {}", p.headroom));
        }
        prev = f;
    }
    if curve(1e6, p.knee, p.headroom) > p.headroom {
        return Err("curve exceeds headroom in the limit".into());
    }
    if (curve(1e6, p.knee, p.headroom) - p.headroom).abs() > 1e-3 {
        return Err("curve does not approach headroom in the limit".into());
    }
    let peak = p.peak_nits();
    if (peak - 1000.0).abs() > 1e-2 {
        return Err(format!("peak_nits {peak} != the display's 1000"));
    }

    // (5) C¹ at the knee: value continuous and slope 1 on BOTH sides. A slope
    // discontinuity here would show as a visible ring around every highlight.
    let h = 1e-4;
    let k = p.knee;
    if (curve(k, k, p.headroom) - k).abs() > 1e-6 {
        return Err("curve(knee) != knee".into());
    }
    let d_lo = (curve(k, k, p.headroom) - curve(k - h, k, p.headroom)) / h;
    let d_hi = (curve(k + h, k, p.headroom) - curve(k, k, p.headroom)) / h;
    if (d_lo - 1.0).abs() > 1e-2 {
        return Err(format!("slope below knee {d_lo} != 1"));
    }
    if (d_hi - 1.0).abs() > 1e-2 {
        return Err(format!("slope above knee {d_hi} != 1 (C1 break at the knee)"));
    }

    // (6) A display with no headroom must not divide by a zero-width rolloff
    // band; it degenerates to the SDR rolloff.
    let flat = ToneParams::hdr(200.0, 200.0);
    if flat.knee != 0.0 || flat.headroom != 1.0 {
        return Err("hdr(200, 200) must degenerate to the SDR rolloff".into());
    }
    let v = map(Vec3A::splat(3.0), flat).x;
    if !v.is_finite() {
        return Err(format!("no-headroom display produced {v}"));
    }

    // (7) The PQ encode — closed-form ST 2084 anchors. Signal 1.0 is exact by
    // construction (c1 + c2 == 1 + c3 == 19.6875, all dyadic), the 100-nit
    // point is the classic published anchor (~0.508), and 0 encodes to ~7e-7.
    if pq_encode(1.0).to_bits() != 1.0f32.to_bits() {
        return Err(format!("pq_encode(1.0) = {} != 1.0 exactly", pq_encode(1.0)));
    }
    if pq_encode(0.0) >= 1e-6 {
        return Err(format!("pq_encode(0.0) = {} >= 1e-6", pq_encode(0.0)));
    }
    let at_100 = pq_encode(100.0 / PQ_MAX_NITS);
    if (at_100 - 0.5081).abs() > 1e-3 {
        return Err(format!("pq_encode(100 nits) = {at_100}, want ~0.5081"));
    }
    // Monotone over the full signal domain, and a numeric EOTF round-trip:
    // decode(encode(y)) must land back on y — the inverse pair is the proof
    // the constants are the real ST 2084 set and not a typo'd cousin.
    let pq_decode = |e: f32| -> f32 {
        let ep = e.powf(1.0 / PQ_M2);
        ((ep - PQ_C1).max(0.0) / (PQ_C2 - PQ_C3 * ep)).powf(1.0 / PQ_M1)
    };
    let mut prev = f32::NEG_INFINITY;
    for i in 0..=1000 {
        let y = i as f32 * 1e-3;
        let e = pq_encode(y);
        if e < prev {
            return Err(format!("pq_encode not monotone at y={y}: {e} < {prev}"));
        }
        prev = e;
        let back = pq_decode(e);
        if (back - y).abs() > 1e-4 {
            return Err(format!("PQ round-trip: y={y} -> encode {e} -> decode {back}"));
        }
    }
    // The gamut matrix maps white to white (rows sum to 1) — a transposed or
    // mistyped matrix tints every neutral.
    let mw = m709_to_2020(Vec3A::ONE);
    if (mw - Vec3A::ONE).abs().max_element() > 1e-5 {
        return Err(format!("m709_to_2020(white) = {mw:?}, want white"));
    }

    // (8) The HDR10 parameterisation: paper-white anchor through the whole
    // map (linear 1.0 must display at exactly paper_white nits after the PQ
    // decode), the reported peak, and the no-headroom degeneracy.
    let p10 = ToneParams::hdr10(200.0, 1000.0);
    if p10.headroom != 5.0 {
        return Err(format!("hdr10(200, 1000) headroom {} != 5", p10.headroom));
    }
    let white_nits = pq_decode(map(Vec3A::ONE, p10).x) * PQ_MAX_NITS;
    if (white_nits - 200.0).abs() > 0.5 {
        return Err(format!("hdr10 paper-white anchor: linear 1.0 -> {white_nits} nits, want 200"));
    }
    let peak10 = p10.peak_nits();
    if (peak10 - 1000.0).abs() > 1e-2 {
        return Err(format!("hdr10 peak_nits {peak10} != 1000"));
    }
    let flat10 = ToneParams::hdr10(200.0, 200.0);
    if flat10.knee != 0.0 || flat10.headroom != 1.0 || flat10.mode != ToneMode::Pq {
        return Err("hdr10(200, 200) must degenerate to the SDR rolloff, PQ-encoded".into());
    }
    let v10 = map(Vec3A::splat(3.0), flat10).x;
    if !v10.is_finite() || !(0.0..=1.0).contains(&v10) {
        return Err(format!("no-headroom HDR10 display produced {v10}"));
    }

    eprintln!("tone self-test: OK");
    Ok(())
}
