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

/// Load-time converter kill levers (`--no-h2n` / `--no-n2h`), settable once
/// before scene load — the `set_mips`/`clouds::set_enabled` "knob before
/// build" family. h2n off restores the pre-conversion behavior exactly
/// (grayscale bump maps are dropped, `normal_tex` stays NO_TEX); n2h off
/// leaves every real normal map's alpha at 255 and `height_amp` at 0.0.
static H2N_ENABLED: AtomicBool = AtomicBool::new(true);
static N2H_ENABLED: AtomicBool = AtomicBool::new(true);

/// Read side of `set_mips`. Exists for `cli::self_test`, whose purity gate has
/// to prove the parse loop did NOT store here.
pub fn mips_enabled() -> bool {
    MIPS_ENABLED.load(Relaxed)
}

pub fn set_h2n(on: bool) {
    H2N_ENABLED.store(on, Relaxed);
}

pub fn h2n_enabled() -> bool {
    H2N_ENABLED.load(Relaxed)
}

pub fn set_n2h(on: bool) {
    N2H_ENABLED.store(on, Relaxed);
}

pub fn n2h_enabled() -> bool {
    N2H_ENABLED.load(Relaxed)
}

/// Slope-space mip filtering for normal maps (`--no-slope-mips` kills) — the
/// `set_mips` "knob before scene load" family. The lever gates the
/// `scene::finalize_normal_mips` pass, NOT `build_mips` itself: off means the
/// pass never marks a texture `normal_role`, so every chain is built by the
/// legacy filter and the off arm is bit-identical to the pre-feature renderer
/// by construction (and `texture::self_test` stays independent of the static —
/// it sets `normal_role` on its probe textures directly). NOT in the scene
/// cache's lever word: mips are derived-only, never persisted (the `--no-mips`
/// precedent).
static SLOPE_MIPS_ENABLED: AtomicBool = AtomicBool::new(true);

pub fn set_slope_mips(on: bool) {
    SLOPE_MIPS_ENABLED.store(on, Relaxed);
}

pub fn slope_mips_enabled() -> bool {
    SLOPE_MIPS_ENABLED.load(Relaxed)
}

/// High-pass knee of the normal→height solve, in CYCLES per texture tile:
/// spectral integration divides by |k|, so the lowest bins amplify normal-map
/// noise into tile-scale height swell (the classic Frankot–Chellappa
/// ill-conditioning). Bins at radial frequency r cycles are attenuated by
/// r²/(r² + K²) — wavelengths longer than ~1/K of the tile fade out, detail
/// passes untouched. The eyeball knob; 0.0 = off (the self-test's solver-core
/// setting).
pub const N2H_HIGHPASS: f32 = 2.0;

/// Ceiling on a DERIVED (n2h) heightfield's amplitude, in texel widths.
/// Aggressive normal maps (near-horizon texels — the z-clamp admits slopes
/// of ±20/texel) integrate to absurd depths (a Bistro map measured 7711
/// texels), which would balloon the swept AABBs and read as geometry, not
/// surface detail. The FIELD keeps its shape ([0,1]-normalized either way);
/// only the world amplitude clamps. Sobel-converted maps keep their exact
/// `HEIGHT_NORMAL_STRENGTH` and never hit this.
pub const N2H_AMP_CAP: f32 = 8.0;

/// Implied bump amplitude of a converted height map, in TEXEL widths: the
/// full black→white height range spans K texels of geometric height, so the
/// Sobel normal is `normalize(−K·∂h, 1)` with ∂h in height-per-texel.
/// Per-texel derivatives are the industry convention (GIMP/NVTT-class
/// filters) and deliberately NOT resolution-normalized — per-UV-unit scaling
/// would make a 4K map 4096× steeper than its content implies. The MTL's
/// `-bm` (`Material::normal_scale`) still multiplies on top at decode time,
/// and the relief march consumes the same K through `Material::height_amp`
/// so the two modes describe one surface.
pub const HEIGHT_NORMAL_STRENGTH: f32 = 2.0;

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
    /// Generated from a grayscale height map by `height_to_normal` (RGB =
    /// Sobel normal, A = the exact source height). Persisted by the scene
    /// cache so the warm-load re-decode re-applies the conversion — the
    /// cached material's `normal_tex` points at the CONVERTED texels, and a
    /// warm load that skipped the conversion would shade the raw grayscale
    /// as a normal map.
    pub h2n: bool,
    /// Height derived from this normal map by `normal_to_height` (Poisson /
    /// Frankot–Chellappa) into the alpha channel, RGB untouched. Persisted
    /// like `h2n` and for the same reason: a re-decoded file has alpha 255.
    pub n2h: bool,
    /// This texture is consumed as a NORMAL MAP (some material's `normal_tex`
    /// points at it), so `build_mips` averages SLOPES (x/z, y/z) instead of
    /// raw channel bytes — averaging unit normal VECTORS under-tilts (the
    /// mean of four tilted normals points flatter than the mean slope, and
    /// `perturb_normal`'s renormalize restores length, not tilt), which is
    /// why normal-mapped surfaces used to flatten with distance. IN-MEMORY
    /// ONLY, never serialized: set by `scene::finalize_normal_mips` from the
    /// material wiring (the one predicate identical across cold/warm/glTF/
    /// world/tile loads), false in every constructor. A texture shared with a
    /// rough/metal/emissive role is deliberately NOT marked (slope-encoded
    /// mips would corrupt those samples — coarser, never wrong).
    pub normal_role: bool,
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
        let mut t = Texture {
            w,
            h,
            texels,
            alpha_masked,
            srgb,
            source: String::new(),
            h2n: false,
            n2h: false,
            normal_role: false,
            mips: Vec::new(),
        };
        if MIPS_ENABLED.load(Relaxed) {
            t.build_mips();
        }
        t
    }

    /// Reconstruct a texture from CACHED post-conversion texels (the world
    /// sidecar's inline flavor — glTF textures have no decodable file path).
    /// Flags are restored VERBATIM, never re-derived: `from_image` would
    /// recompute `alpha_masked` from the texels and resurrect the junk-alpha
    /// cutouts the glTF loader deliberately cleared on OPAQUE materials.
    /// Mips are rebuilt fresh (never persisted — the scene-cache convention),
    /// through the same `MIPS_ENABLED` lever check as `from_image`, so
    /// `--no-mips` means the same thing on a warm world boot.
    #[allow(clippy::too_many_arguments)]
    pub fn from_cached(
        w: u32,
        h: u32,
        texels: Vec<[u8; 4]>,
        alpha_masked: bool,
        srgb: bool,
        source: String,
        h2n: bool,
        n2h: bool,
    ) -> Texture {
        let mut t = Texture {
            w,
            h,
            texels,
            alpha_masked,
            srgb,
            source,
            h2n,
            n2h,
            normal_role: false,
            mips: Vec::new(),
        };
        if MIPS_ENABLED.load(Relaxed) {
            t.build_mips();
        }
        t
    }

    /// Build the floor-halving box-filter chain down to 1×1. Each level
    /// averages a 2×2 tap block of the level above (taps clamp at odd
    /// edges); sRGB channels average in LINEAR space through
    /// `SRGB_LUT`/`encode_srgb`, linear maps and alpha average as raw u8
    /// with round-to-nearest — EXCEPT `normal_role` textures, whose RGB
    /// averages in SLOPE space: decode each tap to a tangent vector (the
    /// exact `perturb_normal` decode incl. the z ≥ 0.05 clamp), average the
    /// slopes (x/z, y/z) — slopes are the linear quantity, so their mean IS
    /// the mean tilt, where the mean of unit VECTORS under-tilts and
    /// sample-time renormalization can't recover it (a steep/flat quad's
    /// true mean slope −2.02 came back −0.77 from the byte average, 2.6×
    /// too flat — the "normal maps flatten with distance" bug) — and
    /// re-encode `normalize(s̄x, s̄y, 1)`. `NORMAL_MAP_Y_SIGN` cancels
    /// through the linear average (encode inverts decode), so the stored-y
    /// convention passes through untouched. Alpha (the n2h/h2n height)
    /// stays the raw u8 box average in every arm — height is linear, and
    /// the follow-on cavity tap wants those mips correct.
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
                    } else if self.normal_role {
                        // Slope-space average (see the doc comment above).
                        let (mut sx, mut sy) = (0.0f32, 0.0f32);
                        for t in &taps {
                            let x = t[0] as f32 / 255.0 * 2.0 - 1.0;
                            let y = t[1] as f32 / 255.0 * 2.0 - 1.0;
                            let z = (t[2] as f32 / 255.0 * 2.0 - 1.0).max(0.05);
                            sx += x / z;
                            sy += y / z;
                        }
                        let n = Vec3A::new(sx * 0.25, sy * 0.25, 1.0).normalize();
                        let enc = |c: f32| ((c * 0.5 + 0.5) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                        [enc(n.x), enc(n.y), enc(n.z), avg_u8(3)]
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

    /// Drop and rebuild the mip chain under the session's `--no-mips` lever —
    /// for passes that change how this texture's chain should be filtered
    /// after construction (`scene::finalize_normal_mips` flips `normal_role`
    /// and calls this). A no-chain session stays no-chain.
    pub fn rebuild_mips(&mut self) {
        self.mips.clear();
        if MIPS_ENABLED.load(Relaxed) {
            self.build_mips();
        }
    }

    /// True when the image is grayscale (r == g == b on every texel) — the
    /// bump-vs-normal-map detector: MTL `map_Bump` files are a mix of true
    /// tangent-space normal maps and grayscale height maps, and treating a
    /// height map as a normal map shades garbage.
    pub fn is_grayscale(&self) -> bool {
        self.texels.iter().all(|t| t[0] == t[1] && t[1] == t[2])
    }

    /// Convert a grayscale HEIGHT map into a tangent-space normal map: 3×3
    /// Sobel derivatives with WRAP addressing (matching the repeat-wrap
    /// samplers — a clamp here would seam every tiled floor), normal =
    /// normalize(−K·∂h/∂u, −K·∂h/∂v, 1) with K = `HEIGHT_NORMAL_STRENGTH`.
    /// The GREEN channel stores −n.y (OpenGL convention): derivatives here
    /// are in image-row space (+y = +v), which is the TRUE tangent frame,
    /// but `shade::perturb_normal` multiplies green by `NORMAL_MAP_Y_SIGN`
    /// (−1) on every map it decodes — so the store pre-negates to land back
    /// on the true normal (pinned end-to-end by `shade::tangent_self_test`).
    /// ALPHA keeps the exact source height byte (h ∈ [0,1], 1 = the surface
    /// plane) — the relief-render field, exact for these maps. Pure function
    /// of the texels: deterministic, zero rng — the cache roundtrip and every
    /// same-seed contract rely on it.
    pub fn height_to_normal(&self) -> Texture {
        let (w, h) = (self.w as i64, self.h as i64);
        let hf = |x: i64, y: i64| -> f32 {
            let xi = x.rem_euclid(w) as u32;
            let yi = y.rem_euclid(h) as u32;
            self.texels[(yi * self.w + xi) as usize][0] as f32 / 255.0
        };
        let enc = |c: f32| ((c * 0.5 + 0.5) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        let mut texels = Vec::with_capacity(self.texels.len());
        for y in 0..h {
            for x in 0..w {
                // Sobel: [-1,0,1] columns (left→right = +u) smoothed [1,2,1]
                // across rows, and the transpose for +v (row 0 = v 0). The
                // [-1,0,1] arm estimates 2·∂h and the smoothing sums to 4,
                // so ∂h per texel = g/8.
                let (mut gx, mut gy) = (0.0f32, 0.0f32);
                for (d, wgt) in [(-1i64, 1.0f32), (0, 2.0), (1, 1.0)] {
                    gx += wgt * (hf(x + 1, y + d) - hf(x - 1, y + d));
                    gy += wgt * (hf(x + d, y + 1) - hf(x + d, y - 1));
                }
                let k = HEIGHT_NORMAL_STRENGTH;
                let n = Vec3A::new(-k * gx / 8.0, -k * gy / 8.0, 1.0).normalize();
                let src_h = self.texels[(y * w + x) as usize][0];
                texels.push([enc(n.x), enc(-n.y), enc(n.z), src_h]);
            }
        }
        let mut t = Texture {
            w: self.w,
            h: self.h,
            texels,
            alpha_masked: false,
            srgb: false,
            source: self.source.clone(),
            h2n: true,
            n2h: false,
            normal_role: false,
            mips: Vec::new(),
        };
        if MIPS_ENABLED.load(Relaxed) {
            t.build_mips();
        }
        t
    }

    /// Derive a heightfield from this NORMAL map (Frankot–Chellappa): decode
    /// each texel's slope exactly as `shade::perturb_normal` does (Y_SIGN
    /// included, z-clamped), least-squares integrate the gradient field by
    /// FFT over the wrap-repeat domain (periodic boundary conditions are
    /// EXACT for repeat-wrap textures — this is the discrete solve, with the
    /// central-difference transfer function, not the continuous ik), and pack
    /// the [0,1]-normalized height into the ALPHA channel (RGB untouched;
    /// the PEAK maps to 1.0 = the surface plane, so relief displacement is
    /// strictly inward). Returns the peak-to-peak amplitude in TEXEL widths
    /// (`Material::height_amp` units), or 0.0 when degenerate (flat map /
    /// tiny dims) — in which case nothing is modified. The least-squares
    /// projection discards the curl (non-integrable) part of hand-authored
    /// maps; `N2H_HIGHPASS` damps the ill-conditioned lowest bins. Pure
    /// function of the texels — deterministic, zero rng (the cache re-applies
    /// it from the `n2h` flag on warm loads).
    pub fn apply_n2h(&mut self) -> f32 {
        let (w, h) = (self.w as usize, self.h as usize);
        if w < 2 || h < 2 {
            return 0.0;
        }
        let mut gx = vec![0.0f32; w * h];
        let mut gy = vec![0.0f32; w * h];
        for (i, t) in self.texels.iter().enumerate() {
            let tnx = t[0] as f32 / 255.0 * 2.0 - 1.0;
            let tny = (t[1] as f32 / 255.0 * 2.0 - 1.0) * crate::shade::NORMAL_MAP_Y_SIGN;
            let tnz = (t[2] as f32 / 255.0 * 2.0 - 1.0).max(0.05);
            gx[i] = -tnx / tnz;
            gy[i] = -tny / tnz;
        }
        let hf = n2h_solve(&gx, &gy, w, h, N2H_HIGHPASS);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &hf {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let amp = hi - lo;
        if !(amp > 1e-4) {
            return 0.0; // flat (or NaN — a corrupt map must not arm relief)
        }
        for (t, &v) in self.texels.iter_mut().zip(&hf) {
            t[3] = (((v - lo) / amp) * 255.0 + 0.5) as u8;
        }
        self.n2h = true;
        if !self.mips.is_empty() {
            self.mips.clear();
            self.build_mips();
        }
        amp
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

    /// ALU bilinear over the ALPHA channel at level 0, repeat wrap, texel
    /// centers at i+0.5 — the relief march's field sampler, in [0,1].
    /// NESTED-LERP form (`a + t·(b−a)`) deliberately: it is EXACT on
    /// constant plateaus, which is what makes a 255-alpha region an exact
    /// 1.0 field and the flat-field relief identity BITWISE (a weighted-sum
    /// form lands ~1 ulp off 1.0 and the identity dies). Level 0 fixed —
    /// visibility never touches mips (the alpha-cutout rule). Bit-mirrored
    /// by trace_common.hlsli's `height_bilinear`; change both together.
    #[inline]
    pub fn height_bilinear(&self, u: f32, v: f32) -> f32 {
        let x = Self::wrap(u) * self.w as f32 - 0.5;
        let y = Self::wrap(v) * self.h as f32 - 0.5;
        let (x0f, y0f) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0f, y - y0f);
        let wrap_i = |i: i64, n: u32| i.rem_euclid(n as i64) as u32;
        let (x0, x1) = (wrap_i(x0f as i64, self.w), wrap_i(x0f as i64 + 1, self.w));
        let (y0, y1) = (wrap_i(y0f as i64, self.h), wrap_i(y0f as i64 + 1, self.h));
        let a = |xx: u32, yy: u32| self.texel(xx, yy)[3] as f32;
        let lerp = |p: f32, q: f32, t: f32| p + t * (q - p);
        let bot = lerp(a(x0, y0), a(x1, y0), fx);
        let top = lerp(a(x0, y1), a(x1, y1), fx);
        lerp(bot, top, fy) / 255.0
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

/// Least-squares integration of a per-texel gradient field over the periodic
/// (wrap-repeat) domain — the discrete Frankot–Chellappa solve. Uses the
/// CENTRAL-DIFFERENCE transfer function `D(f) = i·sin(2πf/N)` rather than the
/// continuous `ik`, so gradients produced by central differencing (the Sobel
/// forward's single-axis case) integrate back EXACTLY; bins where both
/// transfers vanish (DC, the Nyquist checkerboards — content invisible to
/// central differencing) stay zero. `highpass` (cycles) attenuates the
/// ill-conditioned lowest bins by r²/(r²+hp²); 0.0 = off. Returns h per
/// texel in the same length units as the input slopes (texel widths).
fn n2h_solve(gx: &[f32], gy: &[f32], w: usize, h: usize, highpass: f32) -> Vec<f32> {
    use rustfft::{FftPlanner, num_complex::Complex};
    let mut planner = FftPlanner::<f32>::new();
    let fft2 = |buf: &mut Vec<Complex<f32>>, planner: &mut FftPlanner<f32>, forward: bool| {
        let fw =
            if forward { planner.plan_fft_forward(w) } else { planner.plan_fft_inverse(w) };
        for row in buf.chunks_exact_mut(w) {
            fw.process(row);
        }
        let mut tr = vec![Complex::new(0.0f32, 0.0); w * h];
        for y in 0..h {
            for x in 0..w {
                tr[x * h + y] = buf[y * w + x];
            }
        }
        let fh =
            if forward { planner.plan_fft_forward(h) } else { planner.plan_fft_inverse(h) };
        for col in tr.chunks_exact_mut(h) {
            fh.process(col);
        }
        for x in 0..w {
            for y in 0..h {
                buf[y * w + x] = tr[x * h + y];
            }
        }
    };
    let mut cgx: Vec<Complex<f32>> = gx.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut cgy: Vec<Complex<f32>> = gy.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft2(&mut cgx, &mut planner, true);
    fft2(&mut cgy, &mut planner, true);
    let mut hh = vec![Complex::new(0.0f32, 0.0); w * h];
    for fy in 0..h {
        let sy = (std::f32::consts::TAU * fy as f32 / h as f32).sin();
        let cy = if fy <= h / 2 { fy as f32 } else { fy as f32 - h as f32 };
        for fx in 0..w {
            let sx = (std::f32::consts::TAU * fx as f32 / w as f32).sin();
            let denom = sx * sx + sy * sy;
            if denom < 1e-12 {
                continue;
            }
            let i = fy * w + fx;
            // Ĥ = (conj(Dx)·Gx + conj(Dy)·Gy)/(|Dx|²+|Dy|²); conj(i·s) = −i·s.
            let num = cgx[i] * Complex::new(0.0, -sx) + cgy[i] * Complex::new(0.0, -sy);
            let mut v = num / denom;
            if highpass > 0.0 {
                let cx = if fx <= w / 2 { fx as f32 } else { fx as f32 - w as f32 };
                let r2 = cx * cx + cy * cy;
                v *= r2 / (r2 + highpass * highpass);
            }
            hh[i] = v;
        }
    }
    fft2(&mut hh, &mut planner, false);
    let norm = 1.0 / (w * h) as f32;
    hh.iter().map(|c| c.re * norm).collect()
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
        let mut t = Texture {
            w,
            h,
            texels,
            alpha_masked: false,
            srgb,
            source: String::new(),
            h2n: false,
            n2h: false,
            normal_role: false,
            mips: Vec::new(),
        };
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

    // --- Slope-space normal-map mips ---------------------------------------
    // A normal-role texture must average SLOPES, not bytes. Directly sets
    // `normal_role` (never the SLOPE_MIPS lever — that gates the scene pass,
    // and this test must stay static-independent).
    let mkn = |w: u32, h: u32, texels: Vec<[u8; 4]>, role: bool| {
        let mut t = Texture {
            w,
            h,
            texels,
            alpha_masked: false,
            srgb: false,
            source: String::new(),
            h2n: false,
            n2h: false,
            normal_role: role,
            mips: Vec::new(),
        };
        t.build_mips();
        t
    };
    let slope_of = |px: [u8; 4]| {
        let x = px[0] as f32 / 255.0 * 2.0 - 1.0;
        let z = (px[2] as f32 / 255.0 * 2.0 - 1.0).max(0.05);
        x / z
    };
    // Steep ≈ enc(normalize(-4,0,1)) and flat texels, interleaved: the true
    // mean x/z is (−4.049 + 0.004)/2 ≈ −2.02.
    let steep = [4u8, 128, 158, 0];
    let flat = [128u8, 128, 255, 255];
    let quad = vec![steep, flat, steep, flat];
    let t = mkn(2, 2, quad.clone(), true);
    let got = t.mips[0].texels[0];
    let s = slope_of(got);
    if (s + 2.02).abs() > 0.15 {
        fail(format!("slope mip x/z = {s:.3} ({got:?}), want ≈ −2.02 (mean slope preserved)"));
        ok = false;
    }
    // Anti-vacuity teeth: the legacy byte average [66,128,207] decodes to
    // x/z ≈ −0.774 — it must both differ from the slope arm's product AND
    // provably fail the gate above (i.e. the old filter under-tilts 2.6×).
    let legacy = [66u8, 128, 207];
    if got[..3] == legacy {
        fail("slope mip bit-equals the legacy byte average — slope arm not live".into());
        ok = false;
    }
    let ls = slope_of([legacy[0], legacy[1], legacy[2], 0]);
    if (ls + 2.02).abs() <= 0.15 {
        fail(format!("legacy average x/z = {ls:.3} passes the slope gate — teeth are vacuous"));
        ok = false;
    }
    // Y-axis twin (covers the y lane + renormalization symmetrically).
    let steep_y = [128u8, 4, 158, 0];
    let ty = mkn(2, 2, vec![steep_y, flat, steep_y, flat], true);
    let gy = ty.mips[0].texels[0];
    let sy = {
        let y = gy[1] as f32 / 255.0 * 2.0 - 1.0;
        let z = (gy[2] as f32 / 255.0 * 2.0 - 1.0).max(0.05);
        y / z
    };
    if (sy + 2.02).abs() > 0.15 {
        fail(format!("slope mip y/z = {sy:.3} ({gy:?}), want ≈ −2.02"));
        ok = false;
    }
    // Alpha (the height channel) stays a plain box filter in the slope arm.
    let ta = mkn(
        2,
        2,
        vec![[128, 128, 255, 0], [128, 128, 255, 255], [128, 128, 255, 0], [128, 128, 255, 255]],
        true,
    );
    if ta.mips[0].texels[0][3] != 128 {
        fail(format!("slope-arm alpha mip = {}, want box-average 128", ta.mips[0].texels[0][3]));
        ok = false;
    }
    // Constant normal map round-trips every level bit-exactly (enc(0) = 128,
    // enc(1) = 255 exact) — bounds level-chain quantization accumulation.
    let tc = mkn(8, 8, vec![[128, 128, 255, 200]; 64], true);
    for (li, m) in tc.mips.iter().enumerate() {
        for px in &m.texels {
            if *px != [128, 128, 255, 200] {
                fail(format!("constant normal map drifted at level {} ({px:?})", li + 1));
                ok = false;
            }
        }
    }
    // normal_role = false keeps the legacy path bit-exactly (the lever's
    // off-arm identity at the unit level).
    let tl = mkn(2, 2, quad, false);
    if tl.mips[0].texels[0][..3] != legacy {
        fail(format!(
            "non-role mip {:?} != legacy byte average {legacy:?}",
            &tl.mips[0].texels[0][..3]
        ));
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

    // --- height_to_normal (Sobel) -------------------------------------------
    // Expected bytes are HAND-DERIVED constants (K = 2.0), never recomputed
    // through the function under test.
    //
    // Constant field: zero gradient ⇒ exactly (128,128,255) with the source
    // height in alpha, everywhere — borders included (paired with the ramp's
    // wrap pin below, this is where a clamp bug can't hide).
    let t = mk(4, 4, vec![[100, 100, 100, 255]; 16], false).height_to_normal();
    if !(t.h2n && !t.srgb && !t.alpha_masked && !t.mips.is_empty()) {
        fail("height_to_normal flags/mips wrong on the constant field".into());
        ok = false;
    }
    if t.texels.iter().any(|&p| p != [128, 128, 255, 100]) {
        fail(format!("constant height field: {:?}, want [128,128,255,100]", t.texels[0]));
        ok = false;
    }

    // 4×1 x-ramp, reds [0,60,120,180]: interior column 1 sees ∂h/∂u =
    // (120−0)/255/2 ⇒ n = normalize(−0.4706, 0, 1) ⇒ [73,128,243]; WRAP
    // column 0 sees columns 3 and 1 ⇒ ∂h/∂u = (60−180)/255/2 < 0 ⇒
    // [182,128,243] — clamp addressing would produce a different byte.
    let ramp = |v: [u8; 4]| v.map(|c| [c, c, c, 255]).to_vec();
    let t = mk(4, 1, ramp([0, 60, 120, 180]), false).height_to_normal();
    if t.texels[1] != [73, 128, 243, 60] {
        fail(format!("x-ramp interior texel {:?}, want [73,128,243,60]", t.texels[1]));
        ok = false;
    }
    if t.texels[0] != [182, 128, 243, 0] {
        fail(format!("x-ramp wrap texel {:?}, want [182,128,243,0] (wrap pin)", t.texels[0]));
        ok = false;
    }

    // 1×4 y-ramp, same values down rows: green stores −n.y (the OpenGL-
    // convention pre-negation `perturb_normal`'s NORMAL_MAP_Y_SIGN undoes),
    // so an ASCENDING +v ramp encodes green ABOVE 128.
    let t = mk(1, 4, ramp([0, 60, 120, 180]), false).height_to_normal();
    if t.texels[1] != [128, 182, 243, 60] {
        fail(format!("y-ramp interior texel {:?}, want [128,182,243,60] (pre-negation pin)", t.texels[1]));
        ok = false;
    }

    // --- normal_to_height (Frankot–Chellappa) -------------------------------
    // Round-trip: a single-axis cosine height field, Sobel-forward (exactly
    // central differences when the field is constant along the other axis),
    // then apply_n2h on the resulting NORMAL map must recover the source
    // height up to normalization + u8 quantization. Single-frequency, so the
    // high-pass attenuation is a pure scale the normalization absorbs.
    let cosf = |i: u32, n: u32| {
        (127.5 + 100.0 * (std::f32::consts::TAU * 3.0 * i as f32 / n as f32).cos() + 0.5) as u8
    };
    let norm_bytes = |a: &[u8]| -> Vec<u8> {
        let (lo, hi) = (*a.iter().min().unwrap() as f32, *a.iter().max().unwrap() as f32);
        a.iter().map(|&v| ((v as f32 - lo) / (hi - lo) * 255.0 + 0.5) as u8).collect()
    };
    for (w, h, along_x, label) in [(32u32, 8u32, true, "+u"), (8, 32, false, "+v")] {
        let field: Vec<[u8; 4]> = (0..w * h)
            .map(|i| {
                let c = if along_x { cosf(i % w, w) } else { cosf(i / w, h) };
                [c, c, c, 255]
            })
            .collect();
        let src = mk(w, h, field.clone(), false).height_to_normal();
        let mut rec = mk(w, h, field.clone(), false).height_to_normal();
        let mut det = mk(w, h, field, false).height_to_normal();
        let amp = rec.apply_n2h();
        if !(amp > 0.0) || !rec.n2h {
            fail(format!("n2h {label} round-trip: amp {amp} (want > 0)"));
            ok = false;
            continue;
        }
        // The expected alpha is the SOURCE height normalized to [0,255] —
        // the h2n forward stored it verbatim, so normalize that.
        let want = norm_bytes(&src.texels.iter().map(|t| t[3]).collect::<Vec<_>>());
        let (mut sum, mut worst) = (0u32, 0u32);
        for (t, &wb) in rec.texels.iter().zip(&want) {
            let d = (t[3] as i32 - wb as i32).unsigned_abs();
            sum += d;
            worst = worst.max(d);
        }
        let mean = sum as f32 / want.len() as f32;
        // An inverted sign convention comes back as 255−want (mean ≈ 120) —
        // this is the end-to-end Y_SIGN pin for the INVERSE direction.
        if mean > 6.0 || worst > 20 {
            fail(format!("n2h {label} round-trip: mean |Δ| {mean:.1} worst {worst}"));
            ok = false;
        }
        // Determinism (the cache re-applies the conversion on warm loads).
        let amp2 = det.apply_n2h();
        if amp2.to_bits() != amp.to_bits()
            || det.texels.iter().zip(&rec.texels).any(|(a, b)| a[3] != b[3])
        {
            fail(format!("n2h {label} conversion is not deterministic"));
            ok = false;
        }
    }

    // Flat map: no gradient ⇒ amp 0.0, texture untouched (alpha keeps the
    // source height byte, n2h stays false so the cache never re-applies).
    let mut flat = mk(8, 8, vec![[90, 90, 90, 255]; 64], false).height_to_normal();
    if flat.apply_n2h() != 0.0 || flat.n2h || flat.texels.iter().any(|t| t[3] != 90) {
        fail("n2h flat map must be a no-op with amp 0".into());
        ok = false;
    }

    // High-pass must-fire: a frequency-1 gradient (the ill-conditioned bin)
    // is attenuated by 1/(1+hp²) = 1/5 at hp = 2 — the knob must bite there
    // and must not zero it outright.
    {
        let (w, h) = (32usize, 8usize);
        let gx: Vec<f32> = (0..w * h)
            .map(|i| -(std::f32::consts::TAU * (i % w) as f32 / w as f32).sin())
            .collect();
        let gy = vec![0.0f32; w * h];
        let span = |v: &[f32]| {
            v.iter().fold(0.0f32, |a, &x| a.max(x)) - v.iter().fold(0.0f32, |a, &x| a.min(x))
        };
        let plain = span(&n2h_solve(&gx, &gy, w, h, 0.0));
        let damped = span(&n2h_solve(&gx, &gy, w, h, 2.0));
        if !(damped < 0.35 * plain && damped > 0.05 * plain) {
            fail(format!("n2h high-pass: damped {damped:.3} vs plain {plain:.3}"));
            ok = false;
        }
    }

    if ok {
        eprintln!("texture self-test: OK");
    }
    ok
}
