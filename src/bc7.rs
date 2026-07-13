//! BC7 block compression of OPAQUE scene textures (`--bc7`, GPU upload only).
//!
//! Scene textures upload as RGBA8 (4 B/texel). Intel Sponza's set alone is
//! 9.1 GB of VRAM that way — CLAUDE.md's "texture-memory stress", and WDDM
//! demotes over-budget commits silently. BC7 is 8 bpp: a flat 4× cut.
//!
//! # Opaque only — the cutout carve-out
//!
//! The intersector's alpha test is a hard binary threshold on ONE texel
//! (`bvh.rs::moller_trumbore` → `alpha_nearest(u,v) < 128`, mirrored on the
//! GPU by `trace_common.hlsli::alpha_cutout`). BC7's alpha-carrying modes
//! reach only 4/8/16 alpha levels per block: a block straddling a leaf edge
//! takes endpoints ≈(0,255) and, with 2-bit indices, can represent only
//! {0, 84, 171, 255} — an authored alpha of 128 CANNOT be encoded and snaps
//! across the threshold. San Miguel's masks are antialiased (2.4% of texels
//! neither 0 nor 255; 0.11% within ±16 of the threshold), so the flips are
//! certain, not theoretical — and they are VISIBILITY divergences (a triangle
//! hits on one pipeline and misses on the other), not shading noise.
//!
//! So `should_compress` excludes every alpha-masked texture. That is airtight
//! rather than merely careful: `trace.rs`'s `mat_cutout[]` maps a material to
//! `tex + 1` ONLY when `scene.textures[tex].alpha_masked`, and `alpha_cutout`
//! early-outs on 0 — so the set of texture ids the GPU can ever `.Load()` is
//! EXACTLY the alpha-masked set, and those stay RGBA8. (`.Load` on a BC SRV
//! is legal — it returns the hardware-DECODED texel — but decoded means
//! LOSSY, which is exactly what the `< 128` threshold cannot tolerate.)
//! Every other read (albedo, normal, rough, metal, emissive) is bilinear
//! `SampleLevel` and merely gets BC7's quantization error.
//!
//! `should_compress` and that `mat_cutout` predicate agreeing IS the soundness
//! argument. If you change one, change the other.
//!
//! Belt and braces: the `opaque_*` ISPC presets carry `channels: 3` and skip
//! the alpha-carrying mode search entirely, so the hazard is excluded by the
//! ENCODER too, not only by our predicate. (Nothing reads alpha off a
//! compressed texture anyway: `!alpha_masked` sRGB textures are opaque
//! everywhere by that flag's own definition, and the linear maps —
//! normal/rough/metal/emissive — are sampled `.rgb`/`.g`/`.b` only.)
//!
//! Lossy, and deliberately so: the CPU renderer keeps sampling exact RGBA8
//! (`texture::sample_bilinear`), so the GPU-vs-CPU statistical gates
//! (`albedo A/B` ≤ 0.02/channel, the T2 radiance A/B ≤ 2%) move. The
//! exact-zero / tight alpha gates do not — they run through `alpha_cutout`,
//! which only ever reads the untouched RGBA8 masks.

use crate::texture::Texture;
use intel_tex_2::{RgbaSurface, bc7};

/// BC7's fixed tile: 4×4 texels in 16 bytes (8 bpp).
pub const BLOCK: u32 = 4;
pub const BLOCK_BYTES: usize = 16;

/// ISPC quality profile. `Fast` is the default: the encode is paid on EVERY
/// load (there is deliberately no BC7 disk cache), so the profile is a
/// load-time-vs-fidelity knob, not a one-off asset-bake setting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quality {
    UltraFast,
    Fast,
    Basic,
    Slow,
}

impl Quality {
    pub fn parse(s: &str) -> Option<Quality> {
        match s {
            "ultrafast" => Some(Quality::UltraFast),
            "fast" => Some(Quality::Fast),
            "basic" => Some(Quality::Basic),
            "slow" => Some(Quality::Slow),
            _ => None,
        }
    }

    /// The `opaque_*` presets only — see the module note: they are what keeps
    /// the alpha-mode search out of the encoder.
    fn settings(self) -> bc7::EncodeSettings {
        match self {
            Quality::UltraFast => bc7::opaque_ultra_fast_settings(),
            Quality::Fast => bc7::opaque_fast_settings(),
            Quality::Basic => bc7::opaque_basic_settings(),
            Quality::Slow => bc7::opaque_slow_settings(),
        }
    }
}

/// Blocks spanning `n` texels along one axis.
#[inline]
pub fn blocks(n: u32) -> u32 {
    n.div_ceil(BLOCK)
}

/// Tightly-packed encoded size of a `w × h` texture, in bytes.
#[inline]
pub fn encoded_len(w: u32, h: u32) -> usize {
    blocks(w) as usize * blocks(h) as usize * BLOCK_BYTES
}

/// May this texture be block-compressed?
///
/// Two conditions, both load-bearing:
///
/// - `!alpha_masked` — MUST stay identical to the `alpha_masked` arm of
///   `trace.rs`'s `mat_cutout` construction (see the module note). Anything
///   alpha-masked is a cutout mask the intersector `.Load()`s per-texel, and
///   must stay exact RGBA8.
/// - 4-aligned dims — the D3D spec: "A block-compressed texture must be
///   created as a multiple of size 4 in all dimensions." Odd-size BC7
///   resources were MEASURED to create and sample correctly on the dev
///   machine (NVIDIA, debug layer on), but that is driver tolerance, not a
///   contract — and the sound alternative (padded resource + per-material UV
///   rescale) isn't worth it: the odd-size population is San Miguel's
///   arbitrary-dim research scans (63% of its opaque MB, on a scene with no
///   memory pressure), while the scenes BC7 exists for (Bistro, Intel
///   Sponza) are 100% 4-aligned game textures.
#[inline]
pub fn should_compress(t: &Texture) -> bool {
    !t.alpha_masked && t.w % BLOCK == 0 && t.h % BLOCK == 0
}

/// Encode one opaque texture to tightly-packed BC7 blocks (row-major block
/// rows, `blocks(w) * 16` bytes per row — what the GPU upload stages).
///
/// Caller must have checked `should_compress` (which also guarantees the
/// 4-aligned dims the ISPC kernel assumes — its `calc_output_size` is
/// `ceil(w·h / 16) · 16`, which under-counts blocks on unaligned dims).
pub fn encode_opaque(t: &Texture, q: Quality) -> Vec<u8> {
    debug_assert!(should_compress(t), "bc7: encode_opaque needs should_compress");
    bc7::compress_blocks(
        &q.settings(),
        &RgbaSurface { data: t.texels.as_flattened(), width: t.w, height: t.h, stride: t.w * 4 },
    )
}

/// The DXGI format a compressed `t` uploads as — the BC7 twin of the RGBA8
/// `_SRGB` vs `_UNORM` choice in `SceneGpu::new_uploaded`, keyed on the same
/// `Texture::srgb` role (color maps decode through the sRGB transfer in
/// hardware; normal / rough-metal maps are LINEAR data and must not).
#[cfg(windows)]
pub fn dxgi_format(t: &Texture) -> windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT {
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM_SRGB};
    if t.srgb { DXGI_FORMAT_BC7_UNORM_SRGB } else { DXGI_FORMAT_BC7_UNORM }
}

/// Structural gates (run by `--check`). Fidelity is NOT checkable here —
/// there is no CPU BC7 decoder in the tree — so it is measured on the GPU
/// instead, by `--check-gpu`'s M11 section, which samples the uploaded BC7
/// texture back and diffs it against these exact texels.
pub fn self_test() -> Result<(), String> {
    // Block counts, including the unaligned cases the crate's own
    // calc_output_size gets wrong (the reason pad_replicate exists).
    for (n, want) in [(0u32, 0u32), (1, 1), (4, 1), (5, 2), (6, 2), (8, 2), (4096, 1024)] {
        if blocks(n) != want {
            return Err(format!("bc7: blocks({n}) = {} want {want}", blocks(n)));
        }
    }
    if encoded_len(6, 6) != 4 * BLOCK_BYTES {
        return Err(format!("bc7: encoded_len(6,6) = {} want 64", encoded_len(6, 6)));
    }

    let tex = |w: u32, h: u32, f: &dyn Fn(u32, u32) -> [u8; 4], srgb: bool, masked: bool| Texture {
        w,
        h,
        texels: (0..h).flat_map(|y| (0..w).map(move |x| f(x, y))).collect(),
        alpha_masked: masked,
        srgb,
        source: String::new(),
    };

    // should_compress IS the carve-out plus the spec guard. An alpha-masked
    // texture must never be offered to the encoder (the cutout soundness
    // argument), and neither may an unaligned one ("a block-compressed
    // texture must be created as a multiple of size 4 in all dimensions" —
    // odd dims measured working on the dev driver, but that is tolerance,
    // not spec).
    let opaque = tex(8, 8, &|x, y| [x as u8 * 8, y as u8 * 8, 128, 255], true, false);
    let masked = tex(8, 8, &|x, y| [x as u8 * 8, y as u8 * 8, 128, 0], true, true);
    let odd = tex(5, 3, &|x, y| [10 + x as u8, 20 + y as u8, 0, 255], true, false);
    if !should_compress(&opaque) || should_compress(&masked) || should_compress(&odd) {
        return Err(
            "bc7: should_compress must accept 4-aligned opaque and reject alpha-masked/odd-dim"
                .into(),
        );
    }

    // Encode: exact output length, and determinism — the encode is re-run on
    // every load AND re-run by the fidelity gate as its oracle, so a
    // nondeterministic encode would make both irreproducible.
    let a = encode_opaque(&opaque, Quality::Fast);
    if a.len() != encoded_len(8, 8) {
        return Err(format!("bc7: 8x8 encoded to {} B, want {}", a.len(), encoded_len(8, 8)));
    }
    if a != encode_opaque(&opaque, Quality::Fast) {
        return Err("bc7: encode is not deterministic".into());
    }
    if a.iter().all(|&b| b == 0) {
        return Err("bc7: 8x8 encoded to all zeros (kernel did not run?)".into());
    }

    // A constant block is BC7's easy case: every 4x4 tile of a flat texture
    // must encode to the SAME 16 bytes. Catches a stride/pitch mix-up (which
    // would make tiles differ) without needing a decoder.
    let flat = tex(16, 16, &|_, _| [200, 30, 30, 255], true, false);
    let enc = encode_opaque(&flat, Quality::Fast);
    if enc.chunks_exact(BLOCK_BYTES).any(|b| b != &enc[..BLOCK_BYTES]) {
        return Err("bc7: flat texture did not encode to identical blocks (stride bug?)".into());
    }

    Ok(())
}
