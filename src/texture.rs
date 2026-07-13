use glam::Vec3A;
use std::sync::LazyLock;

/// Exact IEC 61966-2-1 sRGB → linear transfer, tabulated per u8 code value.
static SRGB_LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let c = i as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    })
});

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
}

impl Texture {
    pub fn from_image(img: image::DynamicImage, srgb: bool) -> Texture {
        let rgba = img.to_rgba8();
        let (w, h) = (rgba.width(), rgba.height());
        let texels: Vec<[u8; 4]> = rgba.pixels().map(|p| p.0).collect();
        let alpha_masked = srgb && texels.iter().any(|t| t[3] < 250);
        Texture { w, h, texels, alpha_masked, srgb, source: String::new() }
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
    pub fn alpha_nearest(&self, u: f32, v: f32) -> u8 {
        let x = ((Self::wrap(u) * self.w as f32) as u32).min(self.w - 1);
        let y = ((Self::wrap(v) * self.h as f32) as u32).min(self.h - 1);
        self.texel(x, y)[3]
    }
}
