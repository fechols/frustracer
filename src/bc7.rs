//! BC7 block compression of OPAQUE scene textures (ON BY DEFAULT for GPU
//! sessions, GPU upload only; `--no-bc7` is the kill lever).
//!
//! Scene textures upload as RGBA8 (4 B/texel). Intel Sponza's set alone is
//! 9.1 GB of VRAM that way — CLAUDE.md's "texture-memory stress", and WDDM
//! demotes over-budget commits silently. BC7 is 8 bpp: a flat 4× cut.
//!
//! # Two encoder arms (`Bc7Mode`)
//!
//! The DEFAULT arm is the GPU compute encoder (`gpu/bc7gpu.rs` +
//! `shaders/bc7enc.hlsl`, mode-6): it runs at scene-upload time on the
//! session's own device, which is what made default-on affordable — the ispc
//! CPU encode below is ~20 s for Intel Sponza at `fast`, and there is
//! deliberately no BC7 disk cache. `--bc7-cpu` keeps this module's ispc path
//! as the A/B lever and independent quality cross-check (it shares no code
//! with the kernel). If the GPU encoder fails to construct, the upload falls
//! back LOUDLY to uncompressed RGBA8 — never an implicit CPU encode stall.
//!
//! Determinism: the CPU arm is fully deterministic (pinned below). The GPU
//! arm claims per-(device, driver) determinism only — nothing depends on
//! more (no disk cache; the M11 fidelity gate runs the session's own encoder
//! in-session).
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
/// load-time-vs-fidelity knob, not a one-off asset-bake setting. The GPU arm
/// maps the same names onto its effort tiers (`gpu/bc7gpu.rs::effort` —
/// `basic`/`slow` deepen the mode-1 partition search there).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quality {
    UltraFast,
    Fast,
    Basic,
    Slow,
}

/// The session's BC7 stance — which encoder arm, at what quality. The
/// compiled default is `Gpu(Fast)`; `--no-bc7` → `Off`; `--bc7-cpu` → the
/// ispc arm. Threaded as a parameter into `SceneGpu::new_uploaded` (never a
/// global), so CPU-only sessions — which never construct a `SceneGpu` —
/// structurally never read it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bc7Mode {
    Off,
    Gpu(Quality),
    Cpu(Quality),
}

impl Bc7Mode {
    pub fn armed(self) -> bool {
        !matches!(self, Bc7Mode::Off)
    }

    pub fn quality(self) -> Option<Quality> {
        match self {
            Bc7Mode::Off => None,
            Bc7Mode::Gpu(q) | Bc7Mode::Cpu(q) => Some(q),
        }
    }

    /// `--bc7-quality q`: set the quality on the CURRENT arm; an `Off` mode
    /// arms the GPU default (the flag implies `--bc7`, order-independent).
    pub fn with_quality(self, q: Quality) -> Bc7Mode {
        match self {
            Bc7Mode::Off | Bc7Mode::Gpu(_) => Bc7Mode::Gpu(q),
            Bc7Mode::Cpu(_) => Bc7Mode::Cpu(q),
        }
    }

    /// `--bc7`: arm the GPU default if off; an already-armed mode is kept
    /// (so `--bc7-cpu --bc7` stays on the CPU arm — `--bc7` asserts "on",
    /// not "which arm").
    pub fn armed_or_default(self) -> Bc7Mode {
        match self {
            Bc7Mode::Off => Bc7Mode::Gpu(Quality::Fast),
            m => m,
        }
    }
}

impl Quality {
    /// ASCII-case-insensitive; the CLI/settings vocabulary folds HERE.
    pub fn parse(s: &str) -> Option<Quality> {
        match s.to_ascii_lowercase().as_str() {
            "ultrafast" => Some(Quality::UltraFast),
            "fast" => Some(Quality::Fast),
            "basic" => Some(Quality::Basic),
            "slow" => Some(Quality::Slow),
            _ => None,
        }
    }

    /// `parse`'s inverse — the menu's `bc7_quality` row projects the session
    /// value back into the CLI vocabulary through this.
    pub fn name(self) -> &'static str {
        match self {
            Quality::UltraFast => "ultrafast",
            Quality::Fast => "fast",
            Quality::Basic => "basic",
            Quality::Slow => "slow",
        }
    }

    /// The GPU kernel's effort tier for this ispc `Quality` name: 0 = mode-6
    /// PCA fit, no refinement; 1 = + 2 least-squares rounds + CONDITIONAL
    /// mode-1 (top-4 partitions, only on blocks where mode 6 left visible
    /// error); 2/3 = mode-1 always, top-8/16 partitions.
    ///
    /// It lives HERE, beside the quality vocabulary it maps, rather than in
    /// either backend's encoder host: `bc7enc.hlsl` is one kernel with two
    /// hosts, so its `effort` constant is one table with two readers, and a
    /// copy per backend is the transcription this port exists to avoid.
    pub fn effort(self) -> u32 {
        match self {
            Quality::UltraFast => 0,
            Quality::Fast => 1,
            Quality::Basic => 2,
            Quality::Slow => 3,
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
/// (A third condition joined the two: height-carrying textures — `h2n`/`n2h`,
/// whose ALPHA is the relief march's field — must stay RGBA8. The `opaque_*`
/// ISPC presets skip BC7's alpha-carrying modes outright, which would flatten
/// the field to garbage; and like the cutout carve-out, the predicate here
/// must agree with the `mat_height` fill in trace.rs so the id set the relief
/// samplers can reach is exactly the RGBA8 set — the agreement IS the
/// soundness argument.)
#[inline]
pub fn should_compress(t: &Texture) -> bool {
    !t.alpha_masked && !t.h2n && !t.n2h && t.w % BLOCK == 0 && t.h % BLOCK == 0
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

/// Encode one MIP LEVEL of any dims. Levels below/off the 4-texel grid
/// (2×2, 1×1, and odd intermediates of non-pow2 chains) are edge-replicate
/// padded to whole blocks first — BC mips smaller than a block are legal
/// (storage is block-aligned; the logical size stays small) and the padded
/// texels are never sampled: the hardware clamps the filter footprint to
/// the level's logical dims. `should_compress` still gates on the BASE dims
/// only, unchanged.
pub fn encode_level(w: u32, h: u32, texels: &[[u8; 4]], q: Quality) -> Vec<u8> {
    if w % BLOCK == 0 && h % BLOCK == 0 {
        return bc7::compress_blocks(
            &q.settings(),
            &RgbaSurface { data: texels.as_flattened(), width: w, height: h, stride: w * 4 },
        );
    }
    let (pw, ph) = (blocks(w) * BLOCK, blocks(h) * BLOCK);
    let mut padded: Vec<[u8; 4]> = Vec::with_capacity((pw * ph) as usize);
    for y in 0..ph {
        let sy = y.min(h - 1);
        for x in 0..pw {
            let sx = x.min(w - 1);
            padded.push(texels[(sy * w + sx) as usize]);
        }
    }
    bc7::compress_blocks(
        &q.settings(),
        &RgbaSurface { data: padded.as_flattened(), width: pw, height: ph, stride: pw * 4 },
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
        h2n: false,
        n2h: false,
        normal_role: false,
        mips: Vec::new(),
        var_mips: Vec::new(),
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
    // Height-carrying textures (relief field in alpha) must stay RGBA8 —
    // the opaque presets would flatten the field.
    let mut hcarry = tex(8, 8, &|x, y| [x as u8 * 8, y as u8 * 8, 128, 200], false, false);
    hcarry.h2n = true;
    let mut ncarry = tex(8, 8, &|x, y| [x as u8 * 8, y as u8 * 8, 128, 200], false, false);
    ncarry.n2h = true;
    if should_compress(&hcarry) || should_compress(&ncarry) {
        return Err("bc7: should_compress must reject height-carrying (h2n/n2h) textures".into());
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

    // Bc7Mode: the flag algebra the CLI arms rely on. --bc7 arms the GPU
    // default but never switches an explicit CPU arm; --bc7-quality keys the
    // CURRENT arm and arms Gpu from Off.
    if Bc7Mode::Off.armed() || !Bc7Mode::Gpu(Quality::Fast).armed() {
        return Err("bc7: Bc7Mode::armed polarity".into());
    }
    if Bc7Mode::Off.armed_or_default() != Bc7Mode::Gpu(Quality::Fast) {
        return Err("bc7: --bc7 from Off must arm Gpu(Fast)".into());
    }
    if Bc7Mode::Cpu(Quality::Slow).armed_or_default() != Bc7Mode::Cpu(Quality::Slow) {
        return Err("bc7: --bc7 must not switch an explicit CPU arm".into());
    }
    if Bc7Mode::Off.with_quality(Quality::Basic) != Bc7Mode::Gpu(Quality::Basic)
        || Bc7Mode::Cpu(Quality::Fast).with_quality(Quality::Slow) != Bc7Mode::Cpu(Quality::Slow)
        || Bc7Mode::Gpu(Quality::Fast).with_quality(Quality::UltraFast)
            != Bc7Mode::Gpu(Quality::UltraFast)
    {
        return Err("bc7: Bc7Mode::with_quality must key the current arm".into());
    }
    if Bc7Mode::Off.quality().is_some() || Bc7Mode::Gpu(Quality::Slow).quality() != Some(Quality::Slow)
    {
        return Err("bc7: Bc7Mode::quality".into());
    }

    Ok(())
}
