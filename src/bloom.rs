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
//! # Shift invariance — why the downsample kernel is a lever
//!
//! This pass is linear and MEMORYLESS: no history, no reprojection, no rng, the
//! pyramid rebuilt from scratch every presented frame. It therefore cannot
//! invent flicker out of a stable input. But it is not shift-INVARIANT, and that
//! is a defect a moving light shows as flicker.
//!
//! A 2x2 box downsample partitions the image on a fixed grid and gives no
//! partial credit: mip texel `a` is exactly the average of source pixels
//! `2a..2a+1`, so a light at `x = 2a` and the same light at `x = 2a+1` produce a
//! BIT-IDENTICAL level 0. Stack that over `LEVELS` octaves and the halo's
//! centroid quantizes to `2^(i+1)` px at octave i — a sawtooth against the
//! light's true position. `--bloom-lab` measures exactly this and checks itself
//! against that closed form (it reproduces `1, 3, 7, 15, 31, 63` px, ratio 1.00).
//!
//! `DownKernel::Wide13` is the fix and the default: five OVERLAPPING 2x2 boxes,
//! still five bilinear taps and still normalized, so adjacent mip texels share
//! half their footprint and a translating light hands energy over continuously.
//! Measured on a sun-bright disc at 4 px/frame, `--bloom-lab wobble`:
//!
//! | | box | wide13 |
//! |---|---|---|
//! | worst pixel per frame | 41.31/255 | 1.73/255 |
//! | pixels changing > 2/255 | 14387 | 0 |
//! | halo centroid slide | 6.86 px | 0.06 px |
//! | CPU pyramid, 1080p | 6.15 ms | 6.57 ms (+7%) |
//!
//! Amplitude concentrates in the FINE octaves (they carry the largest weights),
//! extent in the COARSE ones — which is why the fix had to be the kernel at
//! every level rather than simply starting the chain at full resolution. A
//! full-res level 0 would have addressed only the first octave, cost ~50 MB, and
//! forced `LEVELS` 6 -> 7 to keep the glare's reach.
//!
//! `--bloom-kernel box` restores the old kernel as an A/B lever, and it stays in
//! the tree permanently because G5's anti-vacuity arm scores against it.
//!
//! Touch the kernel, the weights, `LEVELS`, `LEVEL_FALLOFF` or `STRENGTH` ->
//! run `--bloom-lab wobble` (re-measure `WOBBLE_MAX`), `--check` (G1-G5) with a
//! golden byte-compare, `--check-gpu` (M13 both arms + the arm-discrimination
//! assertion, M13b, M12b) and `cargo test`. CPU and HLSL move in ONE commit or
//! M13 fails by construction — which is what that gate is for.
//!
//! Pure math over an HDR buffer, no rng, and it runs strictly AFTER `accum` —
//! it never feeds tracing, the temporal cache, the upscaler guides, or any gate
//! that scores radiance. `--check`'s radiance A/Bs compare `accum`, so they are
//! structurally untouched by this file.

use glam::Vec3A;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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
        self.apply_weights(hdr, out, &level_weights());
    }

    /// `apply` with the octave weights supplied by the caller.
    ///
    /// Pure plumbing: `apply` is the only shipping caller and it passes
    /// `level_weights()`, so no arithmetic moved and no rendered pixel changes.
    /// It exists so `lab` can drive a ONE-HOT weight vector and read a single
    /// octave's contribution in isolation. A one-hot vector still sums to 1, so
    /// the uniform-image invariant (`self_test` G2) holds per-octave too — which
    /// is what makes the per-octave attribution exact rather than a re-derivation.
    pub(crate) fn apply_weights(&mut self, hdr: &[f32], out: &mut [f32], wts: &[f32; LEVELS]) {
        // Split the borrows up front: `base` and `mips` are disjoint fields, and
        // the downsample chain needs one while it writes the other.
        let Bloom { mips, base, w, h } = self;
        let (w, h) = (*w, *h);
        debug_assert_eq!(hdr.len(), w * h * 3);

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
            // The kernel branch sits OUTSIDE the per-pixel loop on purpose: the
            // `Box` arm below is the pre-lever code verbatim, so its off-state
            // is structural (a branch, not a computed `* 1.0`) and the goldens
            // are byte-identical by construction rather than by fp luck.
            let wide = down_kernel() == DownKernel::Wide13;
            dst.par_chunks_mut(mw).enumerate().for_each(|(y, row)| {
                if wide {
                    for (x, o) in row.iter_mut().enumerate() {
                        *o = down13(src, sw, sh, x, y);
                    }
                    return;
                }
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

/// Which downsample kernel the chain uses.
///
/// This is the anti-flicker lever. The pass is linear and memoryless, so it
/// cannot invent flicker — but it is NOT shift-invariant, and that is a defect
/// a moving light shows as flicker. See the `--bloom-lab` block below for the
/// mechanism, the closed form, and the measurements that chose the default.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DownKernel {
    /// The 2x2 box: one bilinear tap at the 2x2 centre. Support is EXACTLY one
    /// destination texel, so a source pixel contributes 100% to one mip texel
    /// and 0% to its neighbour — no partial credit, which is what makes the
    /// halo's position a staircase in the light's position.
    Box,
    /// Five overlapping 2x2 boxes — the Jimenez/COD downsample — weighted
    /// `0.5` (centre) + `0.125` x4 (offset one SOURCE texel diagonally). Still
    /// five bilinear taps, because each box is one hardware bilinear fetch, and
    /// still normalized, so energy conservation is untouched. The overlap is the
    /// point: a translating source splits its energy between adjacent mip texels
    /// instead of jumping between them.
    Wide13,
}

/// Process-wide state: the pyramid is reused across frames, and `--no-bloom` is
/// the A/B lever (consumed at startup like the other render knobs).
static ENABLED: AtomicBool = AtomicBool::new(true);
/// `DownKernel` as a word. 0 = `Box` (the shipped kernel), 1 = `Wide13`.
static DOWN: AtomicU32 = AtomicU32::new(0);

pub fn set_down_kernel(k: DownKernel) {
    DOWN.store(k as u32, Ordering::Relaxed);
}
pub fn down_kernel() -> DownKernel {
    if DOWN.load(Ordering::Relaxed) == 1 {
        DownKernel::Wide13
    } else {
        DownKernel::Box
    }
}

/// The compile half of the kernel lever, injected ahead of `bloom.hlsl`.
///
/// Emitted from the kernel selected at PSO-build time, so the box arm's shader
/// carries NO wide-path instructions at all — that is what makes `--bloom-kernel
/// box` a true ablation rather than a branch that merely predicts well. The
/// runtime half is `BLOOM_FLAG_WIDE`, set per dispatch in `gpu::bloom::record`;
/// both must agree, so a gate can still pin the feature off at runtime inside a
/// binary that compiled it in.
pub fn kernel_defs() -> String {
    format!(
        "#define BLOOM_ALLOW_WIDE {}\n",
        u32::from(down_kernel() == DownKernel::Wide13)
    )
}

/// One 2x2 box of `src` with its top-left at `(x, y)`, edges clamped.
///
/// On the GPU this is a single bilinear tap; here it is spelled out, exactly as
/// the shipped box path spells out what `cs_down`'s one tap does. Clamping (not
/// wrapping, not zeroing) matches `bilinear` — a zero border would darken the
/// frame's rim and break the energy gate at precisely the place it checks.
#[inline]
fn box2(src: &[Vec3A], sw: usize, sh: usize, x: isize, y: isize) -> Vec3A {
    let cx = |v: isize| v.clamp(0, sw as isize - 1) as usize;
    let cy = |v: isize| v.clamp(0, sh as isize - 1) as usize;
    let (x0, x1) = (cx(x), cx(x + 1));
    let (y0, y1) = (cy(y), cy(y + 1));
    (src[y0 * sw + x0] + src[y0 * sw + x1] + src[y1 * sw + x0] + src[y1 * sw + x1]) * 0.25
}

/// `DownKernel::Wide13` for destination texel `(dx, dy)`.
///
/// Destination texel `(dx, dy)` owns source texels `(2dx, 2dx+1)`; the four
/// outer boxes are that same box shifted by one SOURCE texel diagonally, so the
/// footprint is 4x4 and adjacent destination texels overlap by half. Weights are
/// `0.5 + 4 x 0.125 = 1`, which is what keeps `self_test` G2 (a uniform image
/// survives) and G3 (a point source's total energy is preserved) intact — this
/// kernel redistributes across the grid, it does not discard.
#[inline]
fn down13(src: &[Vec3A], sw: usize, sh: usize, dx: usize, dy: usize) -> Vec3A {
    let (x, y) = (2 * dx as isize, 2 * dy as isize);
    box2(src, sw, sh, x, y) * 0.5
        + (box2(src, sw, sh, x - 1, y - 1)
            + box2(src, sw, sh, x + 1, y - 1)
            + box2(src, sw, sh, x - 1, y + 1)
            + box2(src, sw, sh, x + 1, y + 1))
            * 0.125
}
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

// ==================== the shift-variance probe (`--bloom-lab`) ====================
//
// A dev INSTRUMENT, never a gate: it always exits 0, needs no scene, no BVH, no
// GPU and no DXC, and exists to answer one question — how much does this
// pyramid's output change when the light merely MOVES?
//
// Why the question is well posed. Glare is a linear, MEMORYLESS operator here:
// no history, no reprojection, no rng (see the module header). It cannot invent
// flicker out of a stable input. What it CAN do is alias, because `apply`'s
// downsample partitions the image on a fixed grid — mip texel `a` is exactly the
// average of source pixels `2a..2a+1`, with NO partial credit. A source at
// x = 2a and the same source at x = 2a+1 therefore produce a BIT-IDENTICAL
// level 0. Stack that over `LEVELS` octaves and the halo's centroid quantizes to
// 2^(i+1) px at octave i: a sawtooth against the light's true position, which is
// what a moving highlight reads as flicker.
//
// That gives the probe a CLOSED-FORM prediction to check itself against, which
// is what keeps it honest. Octave i's texel spans 2^(i+1) source pixels, and
// both the tent and the bilinear reconstruction are symmetric, so the halo sits
// at the texel CENTRE and the centroid offset sweeps that whole texel as the
// source crosses it:
//
//     centroid_swing(i) = 2^(i+1) - 1 px   ->   1, 3, 7, 15, 31, 63
//
// A probe that reproduces that series is reading the mechanism; one that does
// not is reading its own bugs. It is printed as PREDICTION every run.

/// A synthetic light the probe translates across the fixture.
///
/// Every source is rasterized as a HARD disc and moved by whole pixels, so the
/// source itself is EXACTLY shift-invariant — bit-identical at every position,
/// merely offset. That is what makes the null hypothesis clean: any wobble the
/// probe reports was manufactured downstream, by the pyramid.
struct LabSrc {
    name: &'static str,
    radius: f32,
    radiance: f32,
}

/// The three shapes actually reported as flickering, plus the analytic control.
const LAB_SRCS: [LabSrc; 3] = [
    // The closed-form control: a single pixel is the delta the prediction above
    // is derived for, so this is the arm PREDICTION is scored on.
    LabSrc { name: "delta", radius: 0.0, radiance: 1.0e3 },
    // The sun disc — small and enormously bright, so essentially the whole
    // pyramid's energy changes grid cell at once. The maximal case.
    LabSrc { name: "sun", radius: 2.5, radiance: 4.0e4 },
    // A bistro-class emissive lamp: wider, far dimmer, sits nearer the knee.
    LabSrc { name: "lamp", radius: 4.0, radiance: 6.0e1 },
];

/// Uniform background radiance. A uniform field is preserved EXACTLY by this
/// pass (`self_test` G2), so it cannot perturb the halo — its only job is to put
/// the tonemap at a realistic operating point, which is where the visible
/// flicker actually lives. `1 - exp(-c)` saturates above ~5, so a wobble at the
/// knee and the same wobble deep in the tail look nothing alike.
const LAB_BG: f32 = 0.35;

/// Fixture size. Divisible by `2^LEVELS`, so every downsample is an exact 2x and
/// no level inherits a truncation artifact that would be mistaken for wobble.
const LAB_DIM: usize = 512;
/// Half-width of the profile window, in px, centred on the source.
const LAB_R: usize = 128;
/// Fixture for the PREDICTION control only. The coarsest octave's halo spans
/// roughly +-128 px, so it needs a frame wide enough to hold it well clear of
/// the clamped border — at `LAB_DIM` it does not fit, and the truncation reads
/// as a swing that is too small.
const LAB_PRED_DIM: usize = 1024;
/// "This pixel visibly changed" threshold, in 8-bit levels. One level is the
/// quantisation floor and would count rounding; two is the smallest step that
/// cannot be dismissed as encoding noise.
const LAB_HOT: f32 = 2.0;
/// The kernels the lab scores, in order. `Box` first so every table reads
/// before-then-after.
const LAB_ARMS: [(&str, DownKernel); 2] =
    [("box", DownKernel::Box), ("wide13", DownKernel::Wide13)];

#[inline]
fn luma(c: Vec3A) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// Paint `s` at `(cx, cy)` over a uniform `LAB_BG` field.
fn lab_paint(hdr: &mut [f32], w: usize, h: usize, cx: usize, cy: usize, s: &LabSrc) {
    for v in hdr.iter_mut() {
        *v = LAB_BG;
    }
    let r = s.radius.ceil() as isize;
    let r2 = s.radius * s.radius;
    for dy in -r..=r {
        for dx in -r..=r {
            // `<=` so radius 0.0 paints exactly the one centre pixel.
            if (dx * dx + dy * dy) as f32 > r2 {
                continue;
            }
            let (x, y) = (cx as isize + dx, cy as isize + dy);
            if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
                continue;
            }
            let p = (y as usize * w + x as usize) * 3;
            hdr[p] = s.radiance;
            hdr[p + 1] = s.radiance;
            hdr[p + 2] = s.radiance;
        }
    }
}

/// What one (fixture, weight-vector) pair scores.
#[derive(Clone, Copy, Default)]
struct Wobble {
    /// Fraction of the halo's own energy that RESHAPES per step, on the
    /// tonemapped composite — the flicker metric, in the domain the user sees.
    l1_tone: f32,
    /// The same, on the isolated linear glare — the mechanism, undistorted by
    /// the curve.
    l1_lin: f32,
    /// The WORST pixel's temporal swing anywhere in the halo, in 8-bit levels
    /// (`/255`) on the tonemapped composite — i.e. how much the most-affected
    /// pixel actually changes when the light moves one step. This is the number
    /// an eye reports as flicker, in the units the rest of the tree measures
    /// stability in. Deliberately a MAX, not a mean: a 64px halo is ~0.6% of a
    /// frame, so any frame-wide mean averages the effect away (which is exactly
    /// why `FRUSTRACER_STAB` cannot see this bug).
    worst255: f32,
    /// How many pixels change by more than `LAB_HOT` per step, averaged over
    /// positions. `worst255` alone cannot tell a single stray texel from a whole
    /// halo breathing; this is the extent, and it is what decides whether a fix
    /// is worth its cost.
    hot_px: f32,
    /// Spread of the linear glare's centroid, px. THE structural number: it is
    /// exactly 0 for a shift-invariant operator and `2^(i+1) - 1` for a box
    /// pyramid's octave i.
    centroid_swing: f32,
}

/// Sweep `src` across one full period of the coarsest grid and score how much
/// the output moves. `wts` selects the arm (shipping weights, or a one-hot
/// vector to isolate a single octave).
fn lab_measure(src: &LabSrc, wts: &[f32; LEVELS], steps: &[usize]) -> Vec<Wobble> {
    lab_measure_at(LAB_DIM, LAB_R, src, wts, steps)
}

/// `lab_measure` on a caller-chosen fixture. G5 uses a small one so it can run
/// inside `--check` on every platform; the lab uses a larger one so the coarse
/// octaves are less edge-clamped.
fn lab_measure_at(
    dim: usize,
    r: usize,
    src: &LabSrc,
    wts: &[f32; LEVELS],
    steps: &[usize],
) -> Vec<Wobble> {
    let (w, h) = (dim, dim);
    let period = 1usize << LEVELS;
    let (x0, y0) = (w / 2 - period / 2, h / 2);
    let win = 2 * r + 1;

    let mut b = Bloom::new(w, h);
    let mut hdr = vec![0.0f32; w * h * 3];
    let mut out = vec![0.0f32; w * h * 3];
    // Profiles for every position, in the SOURCE's own frame, background
    // subtracted (a uniform field passes through untouched, so the background
    // composites to exactly `LAB_BG` and the subtraction is exact).
    let mut glare = vec![0.0f32; period * win * win];
    let mut tone = vec![0.0f32; period * win * win];

    let s = STRENGTH;
    let bg_tone = luma(crate::tone::shape(
        Vec3A::splat(LAB_BG),
        crate::tone::ToneParams::SDR,
    ));

    for k in 0..period {
        let (cx, cy) = (x0 + k, y0);
        lab_paint(&mut hdr, w, h, cx, cy, src);
        b.apply_weights(&hdr, &mut out, wts);
        for j in 0..win {
            let y = cy + j - r;
            for i in 0..win {
                let x = cx + i - r;
                let p = (y * w + x) * 3;
                let o = Vec3A::new(out[p], out[p + 1], out[p + 2]);
                let c = Vec3A::new(hdr[p], hdr[p + 1], hdr[p + 2]);
                // Recover the glare component exactly: the composite is
                // `(1-s)*hdr + s*glare`, and hdr is known, so this isolates the
                // pyramid's own product from the source that is trivially
                // shift-invariant and would otherwise dilute the statistic.
                let g = (o - c * (1.0 - s)) / s;
                let d = k * win * win + j * win + i;
                glare[d] = luma(g) - LAB_BG;
                tone[d] = luma(crate::tone::shape(o, crate::tone::ToneParams::SDR)) - bg_tone;
            }
        }
    }

    // The centroid is per-position, independent of step.
    let (mut cmin, mut cmax) = (f32::INFINITY, f32::NEG_INFINITY);
    for k in 0..period {
        let base = k * win * win;
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for j in 0..win {
            for i in 0..win {
                let v = glare[base + j * win + i].max(0.0) as f64;
                num += (i as f64 - r as f64) * v;
                den += v;
            }
        }
        let c = if den > 0.0 { (num / den) as f32 } else { 0.0 };
        cmin = cmin.min(c);
        cmax = cmax.max(c);
    }

    steps
        .iter()
        .map(|&step| {
            let (mut l1t, mut l1l, mut n) = (0.0f64, 0.0f64, 0usize);
            let mut worst = 0.0f32;
            let mut hot = 0u64;
            let hot_thr = LAB_HOT / 255.0;
            for k in 0..period {
                let (a, bk) = (k * win * win, ((k + step) % period) * win * win);
                let (mut dt, mut st) = (0.0f64, 0.0f64);
                let (mut dl, mut sl) = (0.0f64, 0.0f64);
                for d in 0..win * win {
                    let et = (tone[a + d] - tone[bk + d]).abs();
                    worst = worst.max(et);
                    if et > hot_thr {
                        hot += 1;
                    }
                    dt += et as f64;
                    st += tone[a + d].abs() as f64;
                    dl += (glare[a + d] - glare[bk + d]).abs() as f64;
                    sl += glare[a + d].abs() as f64;
                }
                if st > 0.0 {
                    l1t += dt / st;
                }
                if sl > 0.0 {
                    l1l += dl / sl;
                }
                n += 1;
            }
            Wobble {
                l1_tone: (l1t / n as f64) as f32,
                l1_lin: (l1l / n as f64) as f32,
                worst255: worst * 255.0,
                hot_px: hot as f32 / period as f32,
                centroid_swing: cmax - cmin,
            }
        })
        .collect()
}

/// Centroid swing of ONE octave, measured without a window.
///
/// The windowed statistic above truncates at the coarsest octave — level 5's
/// halo spans the whole fixture, and a window centred on the source biases its
/// centroid toward the window's own centre, which reads as a swing that is too
/// SMALL. That is a probe artifact, not a kernel property, so the prediction
/// control gets its own path: a larger fixture, the centroid taken over the
/// entire image, and a sweep of exactly one period OF THAT OCTAVE (2^(i+1)
/// positions — sweeping further is redundant, the offset is periodic).
fn lab_centroid_swing(dim: usize, src: &LabSrc, octave: usize) -> f32 {
    let (w, h) = (dim, dim);
    let period = 1usize << (octave + 1);
    let (x0, y0) = (w / 2 - period / 2, h / 2);
    let mut wts = [0.0f32; LEVELS];
    wts[octave] = 1.0;

    let mut b = Bloom::new(w, h);
    let mut hdr = vec![0.0f32; w * h * 3];
    let mut out = vec![0.0f32; w * h * 3];
    let s = STRENGTH;
    let (mut cmin, mut cmax) = (f32::INFINITY, f32::NEG_INFINITY);

    for k in 0..period {
        let (cx, cy) = (x0 + k, y0);
        lab_paint(&mut hdr, w, h, cx, cy, src);
        b.apply_weights(&hdr, &mut out, &wts);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 3;
                let o = Vec3A::new(out[p], out[p + 1], out[p + 2]);
                let c = Vec3A::new(hdr[p], hdr[p + 1], hdr[p + 2]);
                let g = (luma(o) - luma(c) * (1.0 - s)) / s - LAB_BG;
                let v = g.max(0.0) as f64;
                num += (x as f64 - cx as f64) * v;
                den += v;
            }
        }
        let c = if den > 0.0 { (num / den) as f32 } else { 0.0 };
        cmin = cmin.min(c);
        cmax = cmax.max(c);
    }
    cmax - cmin
}

/// The `--bloom-lab` entry point. Always returns 0 — this is an instrument, and
/// a number with no threshold behind it yet does not belong in a gate. The
/// threshold comes back as `self_test`'s G5 once it has been measured.
pub fn lab(kind: &str) -> i32 {
    let steps: [usize; 4] = [1, 2, 4, 8];
    println!(
        "bloom-lab {kind}: {LAB_DIM}x{LAB_DIM}, {LEVELS} octaves, strength {STRENGTH}, \
         bg {LAB_BG}, window +-{LAB_R}px"
    );
    println!(
        "  sweeping one full period of the coarsest grid ({} px), hard discs moved by whole \
         pixels (the source is EXACTLY shift-invariant, so every number below is the \
         pyramid's own)",
        1usize << LEVELS
    );

    // The two controls below are statements about the SHIPPED kernel: the
    // closed form 2^(i+1)-1 is derived from the box's grid partition, so pin the
    // arm rather than inheriting whatever ran last.
    set_down_kernel(DownKernel::Box);

    // --- NULL CONTROL ------------------------------------------------------
    // Bloom is memoryless: the same position twice must produce the same halo,
    // exactly. A nonzero here means the probe is broken, not the kernel, and
    // every number after it is meaningless. Printed every run, first.
    let wts = level_weights();
    let null = lab_measure(&LAB_SRCS[0], &wts, &[0]);
    println!(
        "\n  NULL control (same position twice, must be exactly 0): l1_tone {:.3e}  \
         l1_lin {:.3e}  {}",
        null[0].l1_tone,
        null[0].l1_lin,
        if null[0].l1_tone == 0.0 && null[0].l1_lin == 0.0 {
            "OK"
        } else {
            "*** PROBE IS BROKEN — do not trust anything below ***"
        }
    );

    // --- PREDICTION CONTROL ------------------------------------------------
    // One-hot weights isolate octave i. Its centroid swing must reproduce the
    // closed form 2^(i+1) - 1 px. Agreement proves the probe reads the real
    // mechanism; disagreement means it is reading its own bugs.
    println!("\n  PREDICTION control (per-octave centroid swing vs the closed form 2^(i+1)-1):");
    println!("    octave  weight   measured   predicted   ratio");
    let mut pred_ok = true;
    for i in 0..LEVELS {
        let measured = lab_centroid_swing(LAB_PRED_DIM, &LAB_SRCS[0], i);
        let predicted = ((1usize << (i + 1)) - 1) as f32;
        let ratio = measured / predicted;
        if !(0.95..=1.05).contains(&ratio) {
            pred_ok = false;
        }
        println!(
            "    {i:>6}  {:>6.4}  {:>9.2}  {:>9.2}  {ratio:>6.2}",
            wts[i], measured, predicted
        );
    }
    println!(
        "    -> {}",
        if pred_ok {
            "OK — the probe reproduces the closed form, so it is reading the mechanism"
        } else {
            "*** MISMATCH — the probe does not track the closed form; fix it before trusting it ***"
        }
    );

    // --- THE ARM COMPARISON ------------------------------------------------
    // Both arms in ONE process, over the same fixtures. The tree's measurement
    // rule is to difference against the same run's own baseline — a
    // cross-process A/B has swamped real effects here more than once.
    println!("\n  ARM COMPARISON (all {LEVELS} octaves, the weights that render):");
    println!(
        "    kernel  fixture   step   l1_tone   l1_lin  worst/255   hot_px  centroid_swing"
    );
    for (name, k) in LAB_ARMS {
        set_down_kernel(k);
        for src in LAB_SRCS.iter() {
            let m = lab_measure(src, &wts, &steps);
            for (si, step) in steps.iter().enumerate() {
                println!(
                    "    {name:>6}  {:>7}  {step:>5}  {:>8.4}  {:>7.4}  {:>9.2}  {:>7.0}  {:>14.2}",
                    src.name,
                    m[si].l1_tone,
                    m[si].l1_lin,
                    m[si].worst255,
                    m[si].hot_px,
                    m[si].centroid_swing
                );
            }
        }
    }

    // Per-octave attribution. `l1_tone`/`worst255` and `centroid_swing` answer
    // DIFFERENT questions and do not have to agree: a source wider than an
    // octave's texel already gets partial credit there, so it barely slides —
    // but its halo's fine structure still re-registers against that octave's
    // grid every step. Amplitude concentrates in the FINE octaves, extent in the
    // COARSE ones. Read both columns before choosing a fix.
    println!("\n  per-octave contribution on the SUN fixture (one-hot weights, step 1):");
    println!("    kernel  octave  weight   l1_tone   l1_lin  worst/255   hot_px  centroid_swing");
    for (name, k) in LAB_ARMS {
        set_down_kernel(k);
        for i in 0..LEVELS {
            let mut hot = [0.0f32; LEVELS];
            hot[i] = 1.0;
            let m = lab_measure(&LAB_SRCS[1], &hot, &[1]);
            println!(
                "    {name:>6}  {i:>6}  {:>6.4}  {:>8.4}  {:>7.4}  {:>9.2}  {:>7.0}  {:>14.2}",
                wts[i], m[0].l1_tone, m[0].l1_lin, m[0].worst255, m[0].hot_px, m[0].centroid_swing
            );
        }
    }
    // --- COST --------------------------------------------------------------
    // Arms INTERLEAVED and min-of-N, at a real display resolution. Interleaving
    // is the tree's rule (a single sample is worthless, and this box thermally
    // destabilizes under sustained load, which is why this is a min and not a
    // median). This is the CPU pyramid only — the GPU pass is five bilinear taps
    // where the box arm takes one, on buffers that are half-res and smaller, and
    // must be priced separately with --gpu-timing in a presenting session.
    const COST_W: usize = 1920;
    const COST_H: usize = 1080;
    const COST_N: usize = 5;
    let mut best = [f64::INFINITY; 2];
    let mut b = Bloom::new(COST_W, COST_H);
    let mut hdr = vec![0.0f32; COST_W * COST_H * 3];
    let mut out = vec![0.0f32; COST_W * COST_H * 3];
    lab_paint(&mut hdr, COST_W, COST_H, COST_W / 2, COST_H / 2, &LAB_SRCS[1]);
    for _ in 0..COST_N {
        for (ai, (_, k)) in LAB_ARMS.iter().enumerate() {
            set_down_kernel(*k);
            let t = std::time::Instant::now();
            b.apply_weights(&hdr, &mut out, &wts);
            best[ai] = best[ai].min(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    println!("\n  CPU pyramid cost at {COST_W}x{COST_H} (min of {COST_N}, arms interleaved):");
    for (ai, (name, _)) in LAB_ARMS.iter().enumerate() {
        println!(
            "    {name:>6}  {:>6.2} ms{}",
            best[ai],
            if ai == 0 {
                String::new()
            } else {
                format!("   ({:+.0}% vs box)", 100.0 * (best[ai] / best[0] - 1.0))
            }
        );
    }

    set_down_kernel(DownKernel::Box);
    0
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

    // G2-G5 are per-KERNEL. `self_test` drives the lever itself and restores it,
    // so the result never depends on what the session happened to select — and
    // every arm must satisfy the energy contract, not just the shipping one.
    let restore = down_kernel();
    let r = self_test_arms();
    set_down_kernel(restore);
    r?;

    eprintln!(
        "bloom self-test: OK ({LEVELS} octaves, strength {STRENGTH}, energy-conserving, \
         kernels box+wide13)"
    );
    Ok(())
}

/// G2-G5, run once per downsample kernel. Split out of `self_test` so the
/// caller can restore the lever on the error path too.
fn self_test_arms() -> Result<(), String> {
    for (name, k) in LAB_ARMS {
        set_down_kernel(k);
        self_test_energy(name)?;
    }
    self_test_wobble()
}

/// G2/G3/G4 for the currently-selected kernel.
fn self_test_energy(arm: &str) -> Result<(), String> {
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
                "[{arm}] uniform image not preserved at ({x},{y}): {o:?} vs {flat:?} — glare \
                 is creating or destroying energy"
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
            "[{arm}] glare does not conserve total energy: {before:.1} -> {after:.1}"
        ));
    }

    // G4: it actually SPREADS. The point must lose energy at its core and the
    // surround must gain — the whole point of the pass. A no-op kernel would
    // sail through G2/G3 and this is what catches it.
    let core = out2[center];
    if core >= 1000.0 {
        return Err(format!("[{arm}] glare did not draw energy out of the bright core"));
    }
    let far = ((h_px / 2) * w_px + w_px / 2 + 12) * 3;
    if out2[far] <= 0.0 {
        return Err(format!(
            "[{arm}] glare did not spread into the surround (kernel is a no-op)"
        ));
    }
    // ...and the halo must be monotone falling away from the core (a heavy tail,
    // not a ring).
    let mut prev = f32::INFINITY;
    for dx in 1..20 {
        let p = ((h_px / 2) * w_px + w_px / 2 + dx) * 3;
        let v = out2[p];
        if v > prev + 1e-6 {
            return Err(format!(
                "[{arm}] glare halo is not monotone (ring artifact) at dx={dx}"
            ));
        }
        prev = v;
    }
    Ok(())
}

/// Fixture for G5 — deliberately the SAME as the lab's, so the gate and
/// `--bloom-lab` report the same number and there is never a discrepancy to
/// explain away.
///
/// It has to be this big. At 256 the coarsest mip is only 4x4 texels, so the
/// wide kernel's one-texel taps land on the clamped border constantly and the
/// arm scores 4.98 instead of its true 1.73 — a fixture artifact that would have
/// been read as residual flicker. Three sweeps at this size cost ~0.3 s, which
/// `--check` can afford.
const G5_DIM: usize = LAB_DIM;
/// G5's profile half-window. The source sweeps +-32 px about the centre, so
/// `256 +- 32 +- 128` stays inside `[0, 512)`.
const G5_R: usize = LAB_R;
/// The speed G5 scores at, px/step. The box kernel's damage peaks near here
/// (its per-octave sawtooths have periods 2, 4, 8, ... so a 4 px step lands
/// near worst-case phase for the mid octaves), and it is also `--frd-lab`'s
/// default surface speed, i.e. an ordinary pan.
const G5_STEP: usize = 4;
/// Ceiling on the worst pixel's per-step change, in 8-bit levels, for the
/// SHIPPING kernel.
///
/// MEASURED, not inherited. On this fixture the sun disc reads **1.73** under
/// the shipping wide13 kernel and **41.31** under box — a 24x separation. The
/// bound sits at 5.0: ~3x headroom over the shipping arm for platform libm
/// drift, and still 8x under the arm it has to catch. Re-measure with
/// `--bloom-lab wobble` if the fixture, `LEVELS`, `LEVEL_FALLOFF` or `STRENGTH`
/// move; never widen it to make a failing kernel pass.
///
/// For scale: 2/255 is the smallest step that is not encoding noise, and under
/// box **14387 pixels** of a single halo exceeded it every frame at this speed.
/// Under wide13 that count is **zero**.
const WOBBLE_MAX: f32 = 5.0;

/// G5: SHIFT WOBBLE — the anti-flicker gate.
///
/// Everything above pins what glare does to a STILL image. Nothing pinned what
/// it does to a MOVING one, and that gap shipped a real bug: the box downsample
/// partitions the image on a fixed grid with no partial credit, so a translating
/// highlight's halo jumped between grid cells instead of sliding. Every gate
/// stayed green, because every gate looked at one frame at a time.
///
/// Anti-vacuity, both ways, in the same gate: the shipping kernel must pass the
/// bound AND the box kernel must provably EXCEED it. The box arm survives in the
/// tree permanently as the lever's structural off-state, so that half can never
/// rot into a tautology — which is the failure mode a bound-only gate has here
/// more than once.
fn self_test_wobble() -> Result<(), String> {
    let wts = level_weights();
    let steps = [G5_STEP];
    let score = |k: DownKernel| -> Wobble {
        set_down_kernel(k);
        lab_measure_at(G5_DIM, G5_R, &LAB_SRCS[1], &wts, &steps)[0]
    };
    let wide = score(DownKernel::Wide13);
    let boxk = score(DownKernel::Box);

    // The NULL control, inline: bloom is memoryless, so the same position twice
    // must differ by exactly nothing. If this ever fires, the two numbers above
    // are measuring the harness rather than the kernel and the verdict below is
    // meaningless either way.
    set_down_kernel(DownKernel::Wide13);
    let null = lab_measure_at(G5_DIM, G5_R, &LAB_SRCS[1], &wts, &[0])[0];
    if null.worst255 != 0.0 || null.l1_tone != 0.0 {
        return Err(format!(
            "G5 harness is broken: a zero-step sweep must be bit-identical, got \
             worst {:.3e}/255, l1 {:.3e}",
            null.worst255, null.l1_tone
        ));
    }

    if wide.worst255 > WOBBLE_MAX {
        return Err(format!(
            "glare is not shift-invariant enough: the shipping kernel moves the worst pixel \
             {:.2}/255 per {G5_STEP}px step (limit {WOBBLE_MAX:.2}); box arm reads {:.2}. \
             A moving highlight will flicker",
            wide.worst255, boxk.worst255
        ));
    }
    // Teeth. If the box arm ever passes the bound, the bound stopped scoring
    // anything and the gate is proving nothing.
    if boxk.worst255 <= WOBBLE_MAX {
        return Err(format!(
            "G5 IS VACUOUS: the box kernel reads {:.2}/255, inside the {WOBBLE_MAX:.2} limit it \
             is supposed to fail. Either the fixture stopped exciting the defect or the arm \
             is not reaching the kernel — do not raise the limit, fix the probe",
            boxk.worst255
        ));
    }
    eprintln!(
        "bloom G5 shift-wobble: OK — worst pixel/step wide13 {:.2}/255 (limit {WOBBLE_MAX:.2}) \
         vs box {:.2}/255 [teeth {:.1}x], centroid slide {:.2} vs {:.2} px",
        wide.worst255, boxk.worst255, boxk.worst255 / wide.worst255.max(1e-6),
        wide.centroid_swing, boxk.centroid_swing
    );
    Ok(())
}
