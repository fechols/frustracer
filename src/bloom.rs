//! Glare (bloom): the optics between the scene and the sensor.
//!
//! # Why this exists
//!
//! The sun's limb is a HARD edge — a ~650x radiance discontinuity — and that is
//! physically correct. The tonemap (`render::present_px`, `1 - exp(-c)`)
//! saturates above radiance ~5, so the disc lands at a dead-flat 1.0 while the
//! Mie aureole just outside it sits around 0.86. The result reads as a crisp
//! white circle stamped on a soft gradient, which is exactly what it is.
//!
//! Photographs and eyes don't show that, and the reason is NOT that the sun's
//! edge is soft. It is that light scatters on the way IN — in the lens, in the
//! cornea, in the vitreous. A point source lands on the sensor as a bright core
//! with a wide, heavy-tailed halo. That scattering is what this module models,
//! and it is why the fix belongs here, at the display stage, rather than in
//! `sky.rs`: nothing about the sun is wrong.
//!
//! # The kernel
//!
//! Real glare point-spread functions are heavy-tailed (roughly 1/θ²) — a single
//! Gaussian is far too compact and reads as a soft blob rather than a glare.
//! Summing progressively wider blurs approximates the heavy tail well, which is
//! the whole reason the standard mip-pyramid bloom looks right: successive 2x
//! box downsamples give octave-spaced blur radii, and weighting them by
//! `LEVEL_FALLOFF^i` builds the tail.
//!
//! # Energy
//!
//! The composite is `(1 - strength)·hdr + strength·glare`, NOT `hdr + glare`.
//! Glare REDISTRIBUTES light, it does not create it — a scene that is uniformly
//! lit must come back unchanged, and `self_test` pins exactly that. This also
//! means bloom cannot brighten the image overall, so it can never be "tuned" into
//! an exposure change by accident.
//!
//! Pure math over an HDR buffer, no rng, and it runs strictly AFTER `accum` —
//! it never feeds tracing, the temporal cache, the upscaler guides, or any gate
//! that scores radiance. `--check`'s radiance A/Bs compare `accum`, so they are
//! structurally untouched by this file.

use glam::Vec3A;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Octaves of blur summed into the tail. 6 levels at 1080p reaches a ~64 px
/// radius, which is a broad, believable glare.
pub const LEVELS: usize = 6;

/// Per-octave weight. Each successive (wider) level contributes less, which is
/// what shapes the heavy tail; 1.0 would be a flat box, 0.0 a single Gaussian.
const LEVEL_FALLOFF: f32 = 0.72;

/// How much of the image is glare. The composite is energy-conserving, so this
/// is a redistribution fraction, not a gain.
const STRENGTH: f32 = 0.06;

/// Normalized level weights (sum to 1). A pure function of the constants above.
pub fn level_weights() -> [f32; LEVELS] {
    let mut w = [0.0f32; LEVELS];
    let mut k = 1.0f32;
    let mut sum = 0.0f32;
    for wi in w.iter_mut() {
        *wi = k;
        sum += k;
        k *= LEVEL_FALLOFF;
    }
    for wi in w.iter_mut() {
        *wi /= sum;
    }
    w
}

pub fn strength() -> f32 {
    STRENGTH
}

/// A mip pyramid of half-resolution steps, reused across frames.
pub struct Bloom {
    /// `mips[0]` is (w/2, h/2), `mips[i]` is (w >> (i+1), h >> (i+1)).
    mips: Vec<(usize, usize, Vec<Vec3A>)>,
    /// The full-res source, de-interleaved once per frame. Owned (not rebuilt)
    /// because it is the largest buffer here — 16 B/px, ~33 MB at 1080p — and
    /// this runs every presented frame.
    base: Vec<Vec3A>,
    w: usize,
    h: usize,
}

impl Bloom {
    pub fn new(w: usize, h: usize) -> Bloom {
        let mut mips = Vec::with_capacity(LEVELS);
        let (mut mw, mut mh) = (w, h);
        for _ in 0..LEVELS {
            mw = (mw / 2).max(1);
            mh = (mh / 2).max(1);
            mips.push((mw, mh, vec![Vec3A::ZERO; mw * mh]));
        }
        Bloom { mips, base: vec![Vec3A::ZERO; w * h], w, h }
    }

    /// Rebuild for a new resolution (upscaler res steps). Cheap; no gate depends
    /// on bloom state surviving a step — it is recomputed from scratch each
    /// frame anyway.
    pub fn set_res(&mut self, w: usize, h: usize) {
        if self.w != w || self.h != h {
            *self = Bloom::new(w, h);
        }
    }

    /// The glare halo after `apply` — level 0, `Σ wᵢ · upsample(Lᵢ)` at half res.
    /// This is the whole pyramid's product and precisely what `gpu/bloom.rs`
    /// leaves in ITS level 0, which is what makes the two comparable: the GPU
    /// gate (`gpu::bloom::self_test_gpu`) scores against exactly this.
    pub fn halo(&self) -> (usize, usize, &[Vec3A]) {
        let (w, h, ref m) = self.mips[0];
        (w, h, m)
    }

    /// Compute the glare halo for `hdr` (linear RGB, `w*h*3` floats) and write
    /// the COMPOSITE into `out` (same layout, a distinct buffer).
    pub fn apply(&mut self, hdr: &[f32], out: &mut [f32]) {
        // Split the borrows up front: `base` and `mips` are disjoint fields, and
        // the downsample chain needs one while it writes the other.
        let Bloom { mips, base, w, h } = self;
        let (w, h) = (*w, *h);
        debug_assert_eq!(hdr.len(), w * h * 3);
        let wts = level_weights();

        // 1. Downsample chain: each level is a 2x2 box of the one above.
        //    Repeated box downsampling IS the blur — the octave spacing is what
        //    shapes the heavy tail, so no separate Gaussian pass is needed.
        base.par_iter_mut().enumerate().for_each(|(p, o)| {
            *o = Vec3A::new(hdr[p * 3], hdr[p * 3 + 1], hdr[p * 3 + 2]);
        });
        let base = &*base;
        for i in 0..LEVELS {
            let (prev, cur) = mips.split_at_mut(i);
            let (mw, _, dst) = &mut cur[0];
            let mw = *mw;
            let (sw, sh, src): (usize, usize, &[Vec3A]) = if i == 0 {
                (w, h, base)
            } else {
                let p = &prev[i - 1];
                (p.0, p.1, &p.2)
            };
            dst.par_chunks_mut(mw).enumerate().for_each(|(y, row)| {
                let y0 = (2 * y).min(sh - 1);
                let y1 = (2 * y + 1).min(sh - 1);
                for (x, o) in row.iter_mut().enumerate() {
                    let x0 = (2 * x).min(sw - 1);
                    let x1 = (2 * x + 1).min(sw - 1);
                    *o = (src[y0 * sw + x0]
                        + src[y0 * sw + x1]
                        + src[y1 * sw + x0]
                        + src[y1 * sw + x1])
                        * 0.25;
                }
            });
        }

        // 2. Upsample-combine from the coarsest level down. This builds
        //    `Σ wᵢ · upsample(Lᵢ)` while only ever touching mip-sized buffers —
        //    the same sum a per-pixel gather would produce, at a third of the
        //    work. Weights sum to 1, so a uniform image survives exactly (each
        //    level is that same constant, and the weights re-sum to it).
        let last = LEVELS - 1;
        mips[last].2.par_iter_mut().for_each(|p| *p *= wts[last]);
        for i in (0..last).rev() {
            let (lo, hi) = mips.split_at_mut(i + 1);
            let (mw, mh, dst) = &mut lo[i];
            let (mw, mh) = (*mw, *mh);
            let (sw, sh, src) = (hi[0].0, hi[0].1, &hi[0].2);
            let wi = wts[i];
            dst.par_chunks_mut(mw).enumerate().for_each(|(y, row)| {
                let v = (y as f32 + 0.5) * sh as f32 / mh as f32 - 0.5;
                for (x, o) in row.iter_mut().enumerate() {
                    let u = (x as f32 + 0.5) * sw as f32 / mw as f32 - 0.5;
                    *o = *o * wi + tent(src, sw, sh, u, v);
                }
            });
        }

        // 3. Composite at full res. Energy-conserving: glare REDISTRIBUTES light,
        //    it does not add any (see the module header and `self_test` G2).
        let s = STRENGTH;
        let (gw, gh, ref glare) = mips[0];
        out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            let v = (y as f32 + 0.5) * gh as f32 / h as f32 - 0.5;
            for (x, px) in row.chunks_mut(3).enumerate() {
                let u = (x as f32 + 0.5) * gw as f32 / w as f32 - 0.5;
                let g = tent(glare, gw, gh, u, v);
                let p = (y * w + x) * 3;
                let c = Vec3A::new(hdr[p], hdr[p + 1], hdr[p + 2]);
                let o = c * (1.0 - s) + g * s;
                px[0] = o.x;
                px[1] = o.y;
                px[2] = o.z;
            }
        });
    }
}

/// Process-wide state: the pyramid is reused across frames, and `--no-bloom` is
/// the A/B lever (consumed at startup like the other render knobs).
static ENABLED: AtomicBool = AtomicBool::new(true);
static CACHE: Mutex<Option<Cached>> = Mutex::new(None);

/// Everything a presented frame's glare needs, owned across frames. Presentation
/// runs every frame, at full res, and a naive spelling of this pass allocated and
/// memcpy'd the whole image several times over per present (~83 MB at 1080p) —
/// which is exactly the cost the pyramid cache exists to avoid.
struct Cached {
    b: Bloom,
    /// The composite `Bloom::apply` writes; also what the tonemap then reads.
    out: Vec<f32>,
    /// De-interleaved source for the `resolve` path, which has to materialize
    /// `accum` into floats before a convolution can see it.
    src: Vec<f32>,
}

impl Cached {
    fn get<'a>(g: &'a mut Option<Cached>, w: usize, h: usize) -> &'a mut Cached {
        let n = w * h * 3;
        let c = g.get_or_insert_with(|| Cached {
            b: Bloom::new(w, h),
            out: vec![0.0; n],
            src: vec![0.0; n],
        });
        c.b.set_res(w, h);
        if c.out.len() != n {
            c.out = vec![0.0; n];
            c.src = vec![0.0; n];
        }
        c
    }
}

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Glare over an HDR image, handed to `f` as a borrowed slice. Under
/// `--no-bloom` the input passes through untouched — which is what makes that
/// flag bit-identical to the pre-bloom renderer by construction.
///
/// The composite lands in the process cache rather than being copied back over
/// the caller's buffer, so a presented frame allocates nothing.
pub fn with_glare<R>(hdr: &[f32], w: usize, h: usize, f: impl FnOnce(&[f32]) -> R) -> R {
    if !enabled() || w == 0 || h == 0 || hdr.len() != w * h * 3 {
        return f(hdr);
    }
    let mut g = CACHE.lock().unwrap();
    let c = Cached::get(&mut g, w, h);
    c.b.apply(hdr, &mut c.out);
    f(&c.out)
}

/// `with_glare` for a caller that must MATERIALIZE its HDR image first (the
/// `resolve` path, de-accumulating `accum`): `fill` writes the cached source
/// buffer in place, so that image never needs an allocation either. Only called
/// with glare on — the off path keeps `resolve`'s original alloc-free loop.
pub fn with_glare_filled<R>(
    w: usize,
    h: usize,
    fill: impl FnOnce(&mut [f32]),
    f: impl FnOnce(&[f32]) -> R,
) -> R {
    let mut g = CACHE.lock().unwrap();
    let c = Cached::get(&mut g, w, h);
    let Cached { b, out, src } = c;
    fill(src);
    b.apply(src, out);
    f(out)
}

/// 3x3 tent-filtered sample (weights 1,2,1 / 2,4,2 / 1,2,1 over 16), taken at
/// one source-texel spacing.
///
/// This is NOT a rounding-off nicety. A pure 2x2 box downsample has a SQUARE
/// footprint, and reconstructing it with a single bilinear tap leaves that
/// square visible: the glare core comes out as a rounded rectangle rather than a
/// round halo (very obvious on the sun, which is the one thing this pass exists
/// for). The tent isotropizes the kernel. Its weights sum to 1, so energy
/// conservation — and `self_test` G2 — is unaffected.
#[inline]
fn tent(m: &[Vec3A], mw: usize, mh: usize, u: f32, v: f32) -> Vec3A {
    let mut s = Vec3A::ZERO;
    // Corners 1, edges 2, center 4 — over 16.
    const W: [[f32; 3]; 3] = [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]];
    for (j, row) in W.iter().enumerate() {
        for (i, w) in row.iter().enumerate() {
            let du = i as f32 - 1.0;
            let dv = j as f32 - 1.0;
            s += bilinear(m, mw, mh, u + du, v + dv) * (*w / 16.0);
        }
    }
    s
}

/// Bilinear sample with clamped edges. Clamping (not wrapping, not zeroing) is
/// what keeps a uniform image uniform — a zero border would darken the frame's
/// rim and break energy conservation exactly where the gate checks it.
#[inline]
fn bilinear(m: &[Vec3A], mw: usize, mh: usize, u: f32, v: f32) -> Vec3A {
    let x0 = u.floor();
    let y0 = v.floor();
    let fx = u - x0;
    let fy = v - y0;
    let cx = |x: i32| x.clamp(0, mw as i32 - 1) as usize;
    let cy = |y: i32| y.clamp(0, mh as i32 - 1) as usize;
    let (xi, yi) = (x0 as i32, y0 as i32);
    let (a, b) = (cx(xi), cx(xi + 1));
    let (c, d) = (cy(yi), cy(yi + 1));
    let p00 = m[c * mw + a];
    let p10 = m[c * mw + b];
    let p01 = m[d * mw + a];
    let p11 = m[d * mw + b];
    p00.lerp(p10, fx).lerp(p01.lerp(p11, fx), fy)
}

/// Closed-form gates, run by `--check`. No rng, no scene, no GPU.
pub fn self_test() -> Result<(), String> {
    // G1: the weights are a normalized, strictly decreasing octave series. If
    // they ever stop summing to 1, the composite stops conserving energy.
    let w = level_weights();
    let sum: f32 = w.iter().sum();
    if (sum - 1.0).abs() > 1e-6 {
        return Err(format!("level weights sum to {sum}, want 1"));
    }
    for i in 1..LEVELS {
        if w[i] >= w[i - 1] {
            return Err("level weights are not strictly decreasing (no heavy tail)".into());
        }
    }

    // G2: ENERGY CONSERVATION — the load-bearing gate. A uniform image must come
    // back BIT-unchanged: glare redistributes light, it never creates it. This
    // is what makes bloom impossible to accidentally tune into an exposure
    // change, and it is exactly where a zero-padded border or an unnormalized
    // kernel would show up.
    let (w_px, h_px) = (64usize, 48usize);
    let mut b = Bloom::new(w_px, h_px);
    let flat = Vec3A::new(0.3, 0.5, 0.9);
    let src: Vec<f32> = (0..w_px * h_px).flat_map(|_| [flat.x, flat.y, flat.z]).collect();
    let mut out = vec![0.0f32; src.len()];
    b.apply(&src, &mut out);
    for p in 0..w_px * h_px {
        let o = Vec3A::new(out[p * 3], out[p * 3 + 1], out[p * 3 + 2]);
        if (o - flat).abs().max_element() > 1e-4 {
            let (x, y) = (p % w_px, p / w_px);
            return Err(format!(
                "uniform image not preserved at ({x},{y}): {o:?} vs {flat:?} — glare is \
                 creating or destroying energy"
            ));
        }
    }

    // G3: total energy is preserved on a NON-uniform image too (a single bright
    // point). The sum over the frame must be unchanged even though the point
    // spreads — that IS redistribution.
    let mut src2 = vec![0.0f32; w_px * h_px * 3];
    let center = ((h_px / 2) * w_px + w_px / 2) * 3;
    src2[center] = 1000.0;
    src2[center + 1] = 1000.0;
    src2[center + 2] = 1000.0;
    let before: f32 = src2.iter().sum();
    let mut out2 = vec![0.0f32; src2.len()];
    b.apply(&src2, &mut out2);
    let after: f32 = out2.iter().sum();
    if (after - before).abs() / before > 0.02 {
        return Err(format!(
            "glare does not conserve total energy: {before:.1} -> {after:.1}"
        ));
    }

    // G4: it actually SPREADS. The point must lose energy at its core and the
    // surround must gain — the whole point of the pass. A no-op kernel would
    // sail through G2/G3 and this is what catches it.
    let core = out2[center];
    if core >= 1000.0 {
        return Err("glare did not draw energy out of the bright core".into());
    }
    let far = ((h_px / 2) * w_px + w_px / 2 + 12) * 3;
    if out2[far] <= 0.0 {
        return Err("glare did not spread into the surround (kernel is a no-op)".into());
    }
    // ...and the halo must be monotone falling away from the core (a heavy tail,
    // not a ring).
    let mut prev = f32::INFINITY;
    for dx in 1..20 {
        let p = ((h_px / 2) * w_px + w_px / 2 + dx) * 3;
        let v = out2[p];
        if v > prev + 1e-6 {
            return Err(format!("glare halo is not monotone (ring artifact) at dx={dx}"));
        }
        prev = v;
    }

    eprintln!(
        "bloom self-test: OK ({LEVELS} octaves, strength {STRENGTH}, energy-conserving)"
    );
    Ok(())
}
