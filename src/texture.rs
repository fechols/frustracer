use glam::{Vec2, Vec3A};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};

/// Exact IEC 61966-2-1 sRGB → linear transfer, tabulated per u8 code value.
static SRGB_LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let c = i as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    })
});

/// Inverse of `SRGB_LUT`: linear → sRGB-encoded u8, exact IEC transfer with
/// round-to-nearest. `encode(SRGB_LUT[c]) == c` for every code value — the
/// mip chain's constant-color roundtrip gate depends on it.
fn encode_srgb(l: f32) -> u8 {
    let c = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
    (c * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Mip generation switch, settable ONCE before scene load (--no-mips) — the
/// --bvh-ctrav "knob before build" pattern. Off = no chains are built and
/// `sample_trilinear` degenerates to `sample_bilinear` (mips empty).
static MIPS_ENABLED: AtomicBool = AtomicBool::new(true);

pub fn set_mips(on: bool) {
    MIPS_ENABLED.store(on, Relaxed);
    if !on {
        // Mips are the PREREQUISITE for anisotropy: the aniso sampler
        // prefilters the minor axis by picking its mip level, and without a
        // chain there is nothing to prefilter with (N taps of an aliased
        // level 0 is supersampling, not the feature). Enforced in both
        // orders — `set_aniso` re-checks the switch (--no-mips --aniso 16).
        MAX_ANISO.store(1, Relaxed);
    }
}

/// Ceiling on the anisotropy RATIO — the `--aniso N` upper bound and the hard
/// D3D12 limit on a sampler's `MaxAnisotropy` (`gpu/trace.rs` feeds it there
/// verbatim). Deliberately NOT the same constant as `MAX_TAPS`, which is the
/// CPU sampler's own tap budget: they happen to both be 16 today, but one is
/// a hardware limit and the other a software cost knob.
pub const MAX_ANISO_CAP: u32 = 16;

/// Max anisotropy: the longest major/minor footprint ratio the samplers will
/// resolve (`1` = off ⇒ the isotropic ray-cone lod path runs verbatim, which
/// is what makes `--no-aniso` bit-identical to the pre-aniso renderer). A
/// session knob set once before scene load, like `set_mips` — the GPU reads
/// it for the static sampler's `MaxAnisotropy` and FLAG_ANISO, the CPU for
/// `Cone::aniso`, so all three renderers filter the same footprint.
static MAX_ANISO: AtomicU32 = AtomicU32::new(MAX_ANISO_CAP);

pub fn set_aniso(n: u32) {
    let n = if MIPS_ENABLED.load(Relaxed) { n.clamp(1, MAX_ANISO_CAP) } else { 1 };
    MAX_ANISO.store(n, Relaxed);
}

pub fn max_aniso() -> f32 {
    MAX_ANISO.load(Relaxed) as f32
}

/// One mip level below the base image (level 0 lives in `Texture::{w,h,
/// texels}` unchanged — every existing consumer keeps its layout).
pub struct Mip {
    pub w: u32,
    pub h: u32,
    pub texels: Vec<[u8; 4]>,
}

/// A decoded image texture. For color (sRGB) textures the RGB channels stay
/// sRGB-encoded u8 (converted through `SRGB_LUT` at sample time); linear-data
/// maps (normal, roughness/metallic) sample through `sample_bilinear_linear`
/// instead. Alpha is linear coverage either way. RGBA8 storage keeps San
/// Miguel's texture set around 1 GB where f32 RGB would be ~4 GB.
pub struct Texture {
    pub w: u32,
    pub h: u32,
    /// Row-major, `w * h` texels. Row 0 is v = 0 — the loader pre-flips V
    /// (OBJ UVs are bottom-left origin, images top-left), so sampling and
    /// the alpha test share one convention with no per-lookup flip.
    pub texels: Vec<[u8; 4]>,
    /// Any texel with alpha < 250 — precomputed so the intersector's
    /// alpha-cutout path can skip fully opaque textures with one bool.
    /// Computed ONLY for sRGB (color-role) textures: a linear map with a
    /// stray alpha channel must never arm `Scene::any_alpha`/the cutout
    /// pipeline (the loader additionally clears it on non-Kd color roles).
    pub alpha_masked: bool,
    /// sRGB-encoded color data (albedo/emissive) vs linear data (normal,
    /// roughness/metallic). Selects the GPU DXGI format (_SRGB vs UNORM) and
    /// which CPU sampler is valid.
    pub srgb: bool,
    /// Resolved file path this texture was decoded from (empty for non-file
    /// textures). The scene cache stores paths instead of texels and
    /// re-decodes on load, so texture identity must survive the roundtrip.
    pub source: String,
    /// Mip chain, levels 1.. down to 1×1 (floor-halving; level 0 is the base
    /// fields above). Built at decode time in LINEAR space (sRGB textures
    /// decode → average → re-encode; a gamma-space box filter darkens
    /// mid-tones and the self-test rejects it). Empty under --no-mips. Like
    /// the base texels, mips are NOT persisted by the scene cache — they
    /// regenerate on every load.
    pub mips: Vec<Mip>,
}

impl Texture {
    pub fn from_image(img: image::DynamicImage, srgb: bool) -> Texture {
        let rgba = img.to_rgba8();
        let (w, h) = (rgba.width(), rgba.height());
        let texels: Vec<[u8; 4]> = rgba.pixels().map(|p| p.0).collect();
        let alpha_masked = srgb && texels.iter().any(|t| t[3] < 250);
        let mut t = Texture { w, h, texels, alpha_masked, srgb, source: String::new(), mips: Vec::new() };
        if MIPS_ENABLED.load(Relaxed) {
            t.build_mips();
        }
        t
    }

    /// Build the floor-halving box-filter chain down to 1×1. Each level
    /// averages a 2×2 tap block of the level above (taps clamp at odd
    /// edges); sRGB channels average in LINEAR space through
    /// `SRGB_LUT`/`encode_srgb`, linear maps and alpha average as raw u8
    /// with round-to-nearest.
    fn build_mips(&mut self) {
        let (mut w, mut h) = (self.w, self.h);
        while w > 1 || h > 1 {
            let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
            let src: &[[u8; 4]] = match self.mips.last() {
                Some(m) => &m.texels,
                None => &self.texels,
            };
            let mut texels = Vec::with_capacity((nw * nh) as usize);
            for y in 0..nh {
                for x in 0..nw {
                    let taps = [
                        (2 * x, 2 * y),
                        ((2 * x + 1).min(w - 1), 2 * y),
                        (2 * x, (2 * y + 1).min(h - 1)),
                        ((2 * x + 1).min(w - 1), (2 * y + 1).min(h - 1)),
                    ]
                    .map(|(tx, ty)| src[(ty * w + tx) as usize]);
                    let avg_u8 =
                        |ch: usize| ((taps.iter().map(|t| t[ch] as u32).sum::<u32>() + 2) / 4) as u8;
                    let px = if self.srgb {
                        let avg_lin = |ch: usize| {
                            taps.iter().map(|t| SRGB_LUT[t[ch] as usize]).sum::<f32>() * 0.25
                        };
                        [
                            encode_srgb(avg_lin(0)),
                            encode_srgb(avg_lin(1)),
                            encode_srgb(avg_lin(2)),
                            avg_u8(3),
                        ]
                    } else {
                        [avg_u8(0), avg_u8(1), avg_u8(2), avg_u8(3)]
                    };
                    texels.push(px);
                }
            }
            self.mips.push(Mip { w: nw, h: nh, texels });
            (w, h) = (nw, nh);
        }
    }

    /// This texture's dimension term of the ray-cone LOD:
    /// `0.5·log2(w·h)` — added to `shade::tri_lod_base`'s per-hit term to
    /// complete the lod for THIS map (maps on one material differ in size).
    #[inline]
    pub fn lod_dims(&self) -> f32 {
        0.5 * ((self.w * self.h) as f32).log2()
    }

    /// True when the image is grayscale (r == g == b on every texel) — the
    /// bump-vs-normal-map detector: MTL `map_Bump` files are a mix of true
    /// tangent-space normal maps and grayscale height maps, and treating a
    /// height map as a normal map shades garbage.
    pub fn is_grayscale(&self) -> bool {
        self.texels.iter().all(|t| t[0] == t[1] && t[1] == t[2])
    }

    /// Repeat-wrap a texture coordinate into [0, 1). Handles negatives
    /// (`-0.25 → 0.75`); non-finite inputs collapse to 0.
    #[inline]
    fn wrap(c: f32) -> f32 {
        let f = c - c.floor();
        if f.is_finite() { f } else { 0.0 }
    }

    #[inline]
    fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        self.texels[(y * self.w + x) as usize]
    }

    /// Bilinear sample, repeat wrap, sRGB → linear. Returns linear RGB.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> Vec3A {
        // Texel centers at (i + 0.5): shift, split into base + fraction.
        let x = Self::wrap(u) * self.w as f32 - 0.5;
        let y = Self::wrap(v) * self.h as f32 - 0.5;
        let (x0f, y0f) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0f, y - y0f);
        let wrap_i = |i: i64, n: u32| i.rem_euclid(n as i64) as u32;
        let x0 = wrap_i(x0f as i64, self.w);
        let x1 = wrap_i(x0f as i64 + 1, self.w);
        let y0 = wrap_i(y0f as i64, self.h);
        let y1 = wrap_i(y0f as i64 + 1, self.h);
        let lin = |t: [u8; 4]| {
            Vec3A::new(SRGB_LUT[t[0] as usize], SRGB_LUT[t[1] as usize], SRGB_LUT[t[2] as usize])
        };
        let t00 = lin(self.texel(x0, y0));
        let t10 = lin(self.texel(x1, y0));
        let t01 = lin(self.texel(x0, y1));
        let t11 = lin(self.texel(x1, y1));
        t00.lerp(t10, fx).lerp(t01.lerp(t11, fx), fy)
    }

    /// Bilinear sample of LINEAR data (normal / roughness-metallic maps) —
    /// the same loop as `sample_bilinear` with `t/255` in place of the sRGB
    /// LUT. The existing sampler is untouched (bit-identity of all color
    /// paths).
    pub fn sample_bilinear_linear(&self, u: f32, v: f32) -> Vec3A {
        let x = Self::wrap(u) * self.w as f32 - 0.5;
        let y = Self::wrap(v) * self.h as f32 - 0.5;
        let (x0f, y0f) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0f, y - y0f);
        let wrap_i = |i: i64, n: u32| i.rem_euclid(n as i64) as u32;
        let x0 = wrap_i(x0f as i64, self.w);
        let x1 = wrap_i(x0f as i64 + 1, self.w);
        let y0 = wrap_i(y0f as i64, self.h);
        let y1 = wrap_i(y0f as i64 + 1, self.h);
        let lin = |t: [u8; 4]| {
            Vec3A::new(t[0] as f32, t[1] as f32, t[2] as f32) * (1.0 / 255.0)
        };
        let t00 = lin(self.texel(x0, y0));
        let t10 = lin(self.texel(x1, y0));
        let t01 = lin(self.texel(x0, y1));
        let t11 = lin(self.texel(x1, y1));
        t00.lerp(t10, fx).lerp(t01.lerp(t11, fx), fy)
    }

    /// Nearest-texel alpha, repeat wrap — the cutout test. Nearest (not
    /// bilinear) keeps it cheap and bit-deterministic for every ray type.
    /// ALWAYS level 0: the cutout is visibility, and visibility parity
    /// (CPU / GPU RayQuery / DXR any-hit, all reading base texels) is a
    /// correctness contract — mips never touch it.
    pub fn alpha_nearest(&self, u: f32, v: f32) -> u8 {
        let x = ((Self::wrap(u) * self.w as f32) as u32).min(self.w - 1);
        let y = ((Self::wrap(v) * self.h as f32) as u32).min(self.h - 1);
        self.texel(x, y)[3]
    }

    #[inline]
    fn level_dims(&self, level: usize) -> (u32, u32, &[[u8; 4]]) {
        if level == 0 {
            (self.w, self.h, &self.texels)
        } else {
            let m = &self.mips[level - 1];
            (m.w, m.h, &m.texels)
        }
    }

    /// The bilinear loop over an arbitrary mip level — the same wrap /
    /// texel-center math as `sample_bilinear`, parameterized by level and
    /// decode. Levels 0 keep going through the original samplers so their
    /// bit-behavior can never drift.
    fn bilinear_level(&self, level: usize, u: f32, v: f32, srgb: bool) -> Vec3A {
        let (w, h, texels) = self.level_dims(level);
        let x = Self::wrap(u) * w as f32 - 0.5;
        let y = Self::wrap(v) * h as f32 - 0.5;
        let (x0f, y0f) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0f, y - y0f);
        let wrap_i = |i: i64, n: u32| i.rem_euclid(n as i64) as u32;
        let x0 = wrap_i(x0f as i64, w);
        let x1 = wrap_i(x0f as i64 + 1, w);
        let y0 = wrap_i(y0f as i64, h);
        let y1 = wrap_i(y0f as i64 + 1, h);
        let dec = |t: [u8; 4]| {
            if srgb {
                Vec3A::new(SRGB_LUT[t[0] as usize], SRGB_LUT[t[1] as usize], SRGB_LUT[t[2] as usize])
            } else {
                Vec3A::new(t[0] as f32, t[1] as f32, t[2] as f32) * (1.0 / 255.0)
            }
        };
        let at = |px: u32, py: u32| dec(texels[(py * w + px) as usize]);
        let t00 = at(x0, y0);
        let t10 = at(x1, y0);
        let t01 = at(x0, y1);
        let t11 = at(x1, y1);
        t00.lerp(t10, fx).lerp(t01.lerp(t11, fx), fy)
    }

    fn trilinear(&self, u: f32, v: f32, lod: f32, srgb: bool) -> Vec3A {
        let l = lod.min(self.mips.len() as f32);
        let l0f = l.floor();
        let f = l - l0f;
        let l0 = l0f as usize;
        let a = self.bilinear_level(l0, u, v, srgb);
        if f <= 0.0 || l0 >= self.mips.len() {
            return a;
        }
        a.lerp(self.bilinear_level(l0 + 1, u, v, srgb), f)
    }

    /// Trilinear sample of sRGB color data. `lod <= 0` (and NaN, and any
    /// texture without a chain) takes `sample_bilinear` VERBATIM — the
    /// magnification path is bit-identical to the pre-mip renderer, which is
    /// the compatibility contract `self_test` gates.
    pub fn sample_trilinear(&self, u: f32, v: f32, lod: f32) -> Vec3A {
        if !(lod > 0.0) || self.mips.is_empty() {
            return self.sample_bilinear(u, v);
        }
        self.trilinear(u, v, lod, true)
    }

    /// Trilinear sample of LINEAR data (normal / rough-metal maps).
    pub fn sample_trilinear_linear(&self, u: f32, v: f32, lod: f32) -> Vec3A {
        if !(lod > 0.0) || self.mips.is_empty() {
            return self.sample_bilinear_linear(u, v);
        }
        self.trilinear(u, v, lod, false)
    }

    /// One aniso tap. `lod <= 0` (magnification) routes to the ORIGINAL
    /// bilinear samplers, so a 1-tap aniso sample of a magnified texture is
    /// bit-equal to the pre-mip renderer — the same compatibility contract
    /// `sample_trilinear` keeps, and what the self-test pins.
    #[inline]
    fn tap(&self, u: f32, v: f32, lod: f32, srgb: bool) -> Vec3A {
        if !(lod > 0.0) {
            return if srgb { self.sample_bilinear(u, v) } else { self.sample_bilinear_linear(u, v) };
        }
        self.trilinear(u, v, lod, srgb)
    }

    /// Anisotropic sample of the elliptical footprint `(gu, gv)` — the two
    /// axis vectors in NORMALIZED-UV units (`shade::tri_grads`), which is
    /// what lets one footprint serve every map on a material regardless of
    /// its dimensions (each texture scales them by its own w/h here, exactly
    /// as `SampleGrad` does on the GPU).
    ///
    /// The classic D3D-style approximation, and the GPU's hardware aniso is
    /// the same family: pick the mip by the MINOR axis (that is the detail
    /// trilinear throws away — see `shade::tri_lod_base`'s `|n·d|` term), then
    /// average `ceil(ratio)` trilinear taps stepped along the MAJOR axis.
    /// `max_aniso` caps the ratio, so the lod floor is `|maj| / max_aniso`.
    /// A degenerate (zero-length) footprint falls back to the bilinear
    /// sampler bit-exactly.
    pub fn sample_aniso(&self, u: f32, v: f32, gu: Vec2, gv: Vec2, max_aniso: f32, srgb: bool) -> Vec3A {
        let dims = Vec2::new(self.w as f32, self.h as f32);
        let (lu, lv) = ((gu * dims).length(), (gv * dims).length());
        // The longer axis in TEXELS is the major one — the UV map's own
        // stretch can swap the world-space ellipse's axes, so this must be
        // decided here, not by the caller.
        let (maj_uv, lmaj, lmin) = if lu >= lv { (gu, lu, lv) } else { (gv, lv, lu) };
        if !(lmaj > 0.0) || self.mips.is_empty() {
            return if srgb { self.sample_bilinear(u, v) } else { self.sample_bilinear_linear(u, v) };
        }
        let ratio = (lmaj / lmin.max(1e-8)).clamp(1.0, max_aniso.max(1.0));
        let lod = (lmaj / ratio).log2();
        let n = (ratio.ceil() as u32).clamp(1, MAX_TAPS);
        let mut sum = Vec3A::ZERO;
        for i in 0..n {
            // Tap centers spread symmetrically across the major axis: the
            // i-th of n covers [i/n, (i+1)/n] of the footprint, sampled at
            // its center, so the tap set is centered on (u, v).
            let o = maj_uv * ((i as f32 + 0.5) / n as f32 - 0.5);
            sum += self.tap(u + o.x, v + o.y, lod, srgb);
        }
        sum / n as f32
    }
}

/// The CPU sampler's own tap budget per anisotropic sample — a COST knob, not
/// a limit anyone else shares (the GPU resolves the same footprint in the TMU
/// with no tap loop at all). `MAX_ANISO_CAP` is what bounds the ratio; this
/// bounds what the software sampler is willing to spend resolving it, so a
/// ratio above it degrades to a coarser-but-never-wrong average.
const MAX_TAPS: u32 = MAX_ANISO_CAP;

/// DLL-free mip/trilinear gates, run by `--check` (the sphcell precedent):
/// chain shape, linear-space filtering (a gamma-space box filter fails the
/// checker gate), constant-color sRGB roundtrip, trilinear level/lerp
/// mechanics, and the lod ≤ 0 bit-compatibility contract.
pub fn self_test() -> bool {
    let mut ok = true;
    let fail = |msg: String| {
        eprintln!("texture self-test: {msg}");
    };
    let mk = |w: u32, h: u32, texels: Vec<[u8; 4]>, srgb: bool| {
        let mut t =
            Texture { w, h, texels, alpha_masked: false, srgb, source: String::new(), mips: Vec::new() };
        t.build_mips();
        t
    };

    // Chain shape: floor-halving to 1×1, non-power-of-two included.
    let t = mk(7, 3, vec![[128, 128, 128, 255]; 21], true);
    let dims: Vec<(u32, u32)> = t.mips.iter().map(|m| (m.w, m.h)).collect();
    if dims != vec![(3, 1), (1, 1)] {
        fail(format!("7x3 chain dims {dims:?}, want [(3,1),(1,1)]"));
        ok = false;
    }

    // Constant color survives every level within 1 LSB (sRGB roundtrip).
    let c = [200u8, 100, 50, 255];
    let t = mk(8, 8, vec![c; 64], true);
    for (li, m) in t.mips.iter().enumerate() {
        for px in &m.texels {
            for ch in 0..4 {
                if (px[ch] as i32 - c[ch] as i32).abs() > 1 {
                    fail(format!("constant color drifted at level {} ({:?} vs {c:?})", li + 1, px));
                    ok = false;
                }
            }
        }
    }

    // 2×2 checker: the 1×1 mip must be the LINEAR-space average
    // (encode(0.5) ≈ 188), not the gamma-space one (≈ 128).
    let t = mk(
        2,
        2,
        vec![[255, 255, 255, 255], [0, 0, 0, 255], [0, 0, 0, 255], [255, 255, 255, 255]],
        true,
    );
    let want = encode_srgb(0.5);
    let got = t.mips[0].texels[0][0];
    if got != want || (got as i32 - 128).abs() <= 20 {
        fail(format!("checker mip = {got}, want linear-space {want} (gamma-space would be ~128)"));
        ok = false;
    }

    // Trilinear mechanics on a 4×4 gradient (levels: 2×2, 1×1).
    let grad: Vec<[u8; 4]> =
        (0..16).map(|i| [(i * 16) as u8, (255 - i * 16) as u8, 77, 255]).collect();
    let t = mk(4, 4, grad, true);
    for (u, v) in [(0.1f32, 0.2f32), (0.7, 0.9), (0.5, 0.5)] {
        // Integer lod == bilinear at that level.
        if t.sample_trilinear(u, v, 1.0) != t.bilinear_level(1, u, v, true) {
            fail(format!("trilinear(1.0) != level-1 bilinear at ({u},{v})"));
            ok = false;
        }
        // Midpoint == exact lerp of the adjacent levels.
        let mid = t.bilinear_level(1, u, v, true).lerp(t.bilinear_level(2, u, v, true), 0.5);
        if (t.sample_trilinear(u, v, 1.5) - mid).abs().max_element() > 1e-6 {
            fail(format!("trilinear(1.5) != level 1/2 midpoint at ({u},{v})"));
            ok = false;
        }
        // lod <= 0 / NaN: bit-equal the original sampler (the compatibility
        // contract — magnified views are unchanged by the mip feature).
        for lod in [0.0f32, -3.0, f32::NAN] {
            let a = t.sample_trilinear(u, v, lod);
            let b = t.sample_bilinear(u, v);
            if a.to_array().map(f32::to_bits) != b.to_array().map(f32::to_bits) {
                fail(format!("trilinear(lod={lod}) not bit-equal bilinear at ({u},{v})"));
                ok = false;
            }
            let a = t.sample_trilinear_linear(u, v, lod);
            let b = t.sample_bilinear_linear(u, v);
            if a.to_array().map(f32::to_bits) != b.to_array().map(f32::to_bits) {
                fail(format!("trilinear_linear(lod={lod}) not bit-equal at ({u},{v})"));
                ok = false;
            }
        }
    }

    // --- Anisotropic sampler ------------------------------------------------
    // A 16×16 gradient (chain: 8,4,2,1) — big enough for real lod steps.
    let img: Vec<[u8; 4]> = (0..256)
        .map(|i: u32| [(i % 16 * 17) as u8, (i / 16 * 17) as u8, 60, 255])
        .collect();
    let t = mk(16, 16, img, true);
    let (u, v) = (0.37f32, 0.62f32);

    // Ratio 1 (equal orthogonal axes): a single tap == plain trilinear at
    // that lod — aniso is a refinement of trilinear, not a different filter.
    let g = 4.0 / 16.0; // 4 texels
    let iso = t.sample_aniso(u, v, Vec2::new(g, 0.0), Vec2::new(0.0, g), 16.0, true);
    let tri = t.sample_trilinear(u, v, 4.0f32.log2());
    if iso.to_array().map(f32::to_bits) != tri.to_array().map(f32::to_bits) {
        fail("aniso(ratio 1) not bit-equal to trilinear at the same lod".into());
        ok = false;
    }

    // 4:1 stretch: the lod comes from the MINOR axis and the result is the
    // hand-averaged 4 taps along the major axis.
    let (gmaj, gmin) = (Vec2::new(8.0 / 16.0, 0.0), Vec2::new(0.0, 2.0 / 16.0));
    let got = t.sample_aniso(u, v, gmaj, gmin, 16.0, true);
    let lod = 2.0f32.log2(); // |minor| = 2 texels
    let mut want = Vec3A::ZERO;
    for i in 0..4 {
        let o = gmaj * ((i as f32 + 0.5) / 4.0 - 0.5);
        want += t.trilinear(u + o.x, v + o.y, lod, true);
    }
    want /= 4.0;
    if (got - want).abs().max_element() > 1e-6 {
        fail(format!("aniso(4:1) {got:?} != the 4-tap minor-lod average {want:?}"));
        ok = false;
    }
    // Axis order is irrelevant — major/minor is decided by texel length.
    if t.sample_aniso(u, v, gmin, gmaj, 16.0, true) != got {
        fail("aniso is not symmetric under swapping the two axes".into());
        ok = false;
    }

    // max_aniso = 1 clamps the ratio: one tap at the MAJOR-axis lod — i.e.
    // exactly what trilinear does today (the over-blur aniso exists to fix).
    let capped = t.sample_aniso(u, v, gmaj, gmin, 1.0, true);
    let major_lod = t.sample_trilinear(u, v, 8.0f32.log2());
    if capped.to_array().map(f32::to_bits) != major_lod.to_array().map(f32::to_bits) {
        fail("aniso(max=1) != trilinear at the major-axis lod".into());
        ok = false;
    }
    // …and the 16× sample must actually differ from it, or nothing is being
    // resolved (anti-vacuity for the gates above).
    if (got - capped).abs().max_element() <= 1e-6 {
        fail("aniso(16) == aniso(1): the minor axis is not being resolved".into());
        ok = false;
    }

    // Degenerate footprint (zero axes) and magnification: bit-equal to the
    // original bilinear sampler — the pre-mip compatibility contract.
    for (gu, gv) in [(Vec2::ZERO, Vec2::ZERO), (Vec2::new(1e-9, 0.0), Vec2::new(0.0, 1e-9))] {
        let a = t.sample_aniso(u, v, gu, gv, 16.0, true);
        let b = t.sample_bilinear(u, v);
        if a.to_array().map(f32::to_bits) != b.to_array().map(f32::to_bits) {
            fail(format!("aniso({gu:?},{gv:?}) not bit-equal to bilinear"));
            ok = false;
        }
        let a = t.sample_aniso(u, v, gu, gv, 16.0, false);
        let b = t.sample_bilinear_linear(u, v);
        if a.to_array().map(f32::to_bits) != b.to_array().map(f32::to_bits) {
            fail("aniso(degenerate, linear) not bit-equal to bilinear_linear".into());
            ok = false;
        }
    }

    if ok {
        eprintln!("texture self-test: OK");
    }
    ok
}
