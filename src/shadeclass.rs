//! Shading-class taxonomy for the `--dxr-sbt` many-record SBT ladder.
//!
//! Eight FIELD-derived classes partition materials by which `shade.hlsli`
//! arms can ever execute for them, so a per-class closest-hit shader may
//! compile the dead arms OUT (`--dxr-sbt 2`) and Intel's Thread Sorting Unit
//! gets per-class shader RECORDS to sort (`--dxr-sbt 1..3`). Pure CPU math,
//! `--check`-gated, no GPU types — the `matclass.rs`/`blas_split.rs` shape.
//!
//! THE SOUNDNESS SPINE: a class may STRIP a shade arm only when its
//! MEMBERSHIP PREDICATE forces that arm's runtime guard data-false for every
//! member. Stripped code is then provably never executed FOR THE RECORD'S
//! OWN SURFACE, so a specialized shader is semantically identical to the
//! uber one for its own geometry. `STRIPS` is the one table;
//! `classify_materials`, `strip_defines`, and `self_test`'s must-fire all
//! derive from it, so the table cannot drift from its own soundness
//! argument. Anything not provably strippable lands in `CK_UBER` — coarser,
//! never wrong.
//!
//! THE SCOPE OF THAT CLAIM, measured (2026-08-04): under `--dxr-sbt 2` the
//! flattened lap loop in a specialized record ALSO shades the record's
//! continuation surfaces (reflected/refracted children), which can belong
//! to a DIFFERENT class — a tex-opaque parent's strips drop a glass child's
//! transmission. Mode-2-vs-mode-1 drift at the SM-lp glassware pose: max
//! |d| 9.61e-3 (~3% of channels) vs the 5.96e-8 fp-noise floor — real,
//! bounded under the statistical suites, documented as the price of an
//! occupancy INSTRUMENT (gpu/dxr.rs's SBT_MODE doc carries the same note).
//! `--dxr-sbt 3` closes it BY CONSTRUCTION: every continuation TraceRays
//! into the child's OWN class record, so each surface shades under its own
//! strips and the per-member claim is the whole story again.
//!
//! Deliberately NOT keyed on `Material::class` (the matclass verdict):
//! shading never reads it and it is `IDX_DEFAULT` on every non-OBJ load
//! path (scene.rs's field doc). The fields below are populated on EVERY
//! loader. The cutout predicate is copied verbatim from the `mat_cutout`
//! fill (trace.rs) — the `bc7::should_compress` two-predicates-write-once
//! discipline: the intersector's cutout set and this class's must be one
//! decision. Note cutout is a class DISCRIMINATOR but never a `Strip`:
//! the cutout test lives in the INTERSECTOR, compiled per scene, not in a
//! shade arm.
//!
//! The reflection strip is the subtle row: `SHADE_MAT_REFL` gates BOTH the
//! VNDF bounce block AND the MIS reweight of the light-sampled specular
//! (shade.hlsli — stripping the bounce without forcing `w_l = 1` deletes
//! the highlight, the one-sky invariant's shipped-bug class; one macro
//! feeds both sites so they cannot split). Its rng pair draws INSIDE the
//! gate on both CPU and GPU, so a class whose predicate forces the gate
//! false is same-seed stream-identical with the arm stripped — no burn.

use crate::scene::{MatKind, Material, NO_TEX};
use crate::texture::Texture;

pub const N_CLASSES: usize = 8;

/// Display names, `gpu scene:` histogram order.
pub const NAMES: [&str; N_CLASSES] =
    ["lambert", "gloss", "tex-opaque", "cutout", "glass", "water", "emissive", "uber"];

pub const CK_LAMBERT: u8 = 0;
pub const CK_GLOSS: u8 = 1;
pub const CK_TEX_OPAQUE: u8 = 2;
pub const CK_CUTOUT: u8 = 3;
pub const CK_GLASS: u8 = 4;
pub const CK_WATER: u8 = 5;
pub const CK_EMISSIVE: u8 = 6;
pub const CK_UBER: u8 = 7;

/// The strippable shade arms — one variant per `SHADE_MAT_*` macro seam in
/// shade.hlsli (Commit B lands the seams; the vocabulary is fixed here so
/// the table and its gate exist first). `Refl` is the MIS-coupled one (see
/// the module doc).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strip {
    TexKind,
    Marble,
    Normal,
    RoughTex,
    MetalTex,
    EmisTex,
    Ripple,
    Sheen,
    Translucency,
    Transmission,
    Aniso,
    Refl,
    Emissive,
}

use Strip::*;

/// Per-class strip sets — THE table. A macro absent here keeps its default
/// (the verbatim data expression), so `CK_UBER`'s empty row IS today's uber
/// shader. Every entry must be implied by `classify`'s conjunction for the
/// same class; `verify_strips` proves that mechanically, and `self_test`
/// runs it over a synthetic all-classes set with boundary probes.
pub const STRIPS: [&[Strip]; N_CLASSES] = [
    // lambert: nothing optional survives — the biggest register win (since
    // the untextured detail arm, "nothing" excludes the detail block, whose
    // untextured window legitimately runs in stripped records: lambert
    // materials ARE untextured and carry the synthetic detail_scale).
    &[
        TexKind, Marble, Normal, RoughTex, MetalTex, EmisTex, Ripple, Sheen, Translucency,
        Transmission, Aniso, Refl, Emissive,
    ],
    // gloss: keeps REFL + MARBLE + ANISO (the default scene's steel teapot
    // carries anisotropy 0.8 — measured in scene.rs, not assumed away).
    &[
        TexKind, Normal, RoughTex, MetalTex, EmisTex, Ripple, Sheen, Translucency, Transmission,
        Emissive,
    ],
    // tex-opaque: keeps the map machinery + REFL.
    &[Marble, Ripple, Sheen, Translucency, Transmission, Aniso, EmisTex, Emissive],
    // cutout (foliage): keeps SHEEN + TRANSLUCENCY — leaves carry both.
    &[Marble, Ripple, Transmission, Aniso, EmisTex, Emissive],
    // glass: keeps the Snell chain + REFL + TEXKIND (flat-tint textured glass).
    &[Marble, Normal, RoughTex, MetalTex, EmisTex, Ripple, Sheen, Translucency, Aniso, Emissive],
    // water: keeps RIPPLE + TRANSMISSION + REFL + NORMAL — low-poly San
    // Miguel's water carries water_bump.png (the ripple composes ON the
    // normal map, the n_g/n_s split doc), so NORMAL must stay live here.
    &[Marble, RoughTex, MetalTex, EmisTex, Sheen, Translucency, Aniso, Emissive],
    // emissive: keeps the display add + maps + REFL (cutout ADMITTED —
    // cutout is not a shade strip, see the module doc).
    &[Marble, Ripple, Sheen, Translucency, Transmission, Aniso],
    // uber: strips nothing — the correctness backstop.
    &[],
];

/// Does `m` satisfy the guard-is-data-false obligation `s` imposes? The
/// right-hand sides are the EXACT runtime guards of the shade arms (flat
/// fields only — the reflection gate deliberately reads flat
/// metallic/roughness on both CPU and GPU, the documented statistical-A/B
/// argument, which is what makes this predicate meaningful at all).
fn strip_ok(s: Strip, m: &Material) -> bool {
    match s {
        TexKind => !matches!(m.kind, MatKind::Textured { .. }),
        Marble => !matches!(m.kind, MatKind::Marble { .. }),
        Normal => m.normal_tex == NO_TEX,
        RoughTex => m.rough_tex == NO_TEX,
        MetalTex => m.metal_tex == NO_TEX,
        EmisTex => m.emissive_tex == NO_TEX,
        Ripple => m.ripple_amp == 0.0,
        Sheen => m.sheen == 0.0,
        Translucency => m.translucency == 0.0,
        Transmission => m.transmission == 0.0,
        Aniso => m.anisotropy == 0.0,
        Refl => !(m.metallic > 0.04 || m.roughness < 0.45),
        Emissive => m.emissive == glam::Vec3A::ZERO && m.emissive_tex == NO_TEX,
    }
}

/// The `mat_cutout` predicate, verbatim (trace.rs's fill): a Textured
/// material whose kd texture is alpha-masked.
fn is_cutout(m: &Material, textures: &[Texture]) -> bool {
    match m.kind {
        MatKind::Textured { tex } => textures[tex as usize].alpha_masked,
        _ => false,
    }
}

/// First-match membership. Order is load-bearing only where domains overlap
/// (LAMBERT before GLOSS — lambert is the refl-off refinement); every arm's
/// conjunction is exactly what its class's strip set requires, and
/// `verify_strips` re-proves that from `STRIPS` rather than trusting this
/// function's spelling.
fn classify(m: &Material, textures: &[Texture]) -> u8 {
    let no_rme = m.rough_tex == NO_TEX && m.metal_tex == NO_TEX && m.emissive_tex == NO_TEX;
    let emissive_off = m.emissive == glam::Vec3A::ZERO && m.emissive_tex == NO_TEX;
    let featureless =
        m.sheen == 0.0 && m.translucency == 0.0 && m.anisotropy == 0.0 && emissive_off;
    let cutout = is_cutout(m, textures);
    let not_marble = !matches!(m.kind, MatKind::Marble { .. });

    if m.ripple_amp > 0.0
        && featureless
        && m.rough_tex == NO_TEX
        && m.metal_tex == NO_TEX
        && not_marble
        && !cutout
    {
        return CK_WATER; // NORMAL deliberately unconstrained (water_bump).
    }
    if m.transmission > 0.0
        && m.ripple_amp == 0.0
        && featureless
        && m.normal_tex == NO_TEX
        && no_rme
        && not_marble
        && !cutout
    {
        return CK_GLASS;
    }
    if !emissive_off
        && m.transmission == 0.0
        && m.ripple_amp == 0.0
        && m.sheen == 0.0
        && m.translucency == 0.0
        && m.anisotropy == 0.0
        && not_marble
    {
        return CK_EMISSIVE; // cutout admitted — not a shade strip.
    }
    if cutout
        && m.transmission == 0.0
        && m.ripple_amp == 0.0
        && m.anisotropy == 0.0
        && emissive_off
    {
        return CK_CUTOUT; // sheen/translucency deliberately unconstrained.
    }
    if matches!(m.kind, MatKind::Textured { .. })
        && !cutout
        && m.transmission == 0.0
        && m.ripple_amp == 0.0
        && featureless
    {
        return CK_TEX_OPAQUE;
    }
    if matches!(m.kind, MatKind::Diffuse)
        && m.normal_tex == NO_TEX
        && no_rme
        && m.transmission == 0.0
        && m.ripple_amp == 0.0
        && featureless
        && !(m.metallic > 0.04 || m.roughness < 0.45)
    {
        return CK_LAMBERT;
    }
    if !matches!(m.kind, MatKind::Textured { .. })
        && m.normal_tex == NO_TEX
        && no_rme
        && m.transmission == 0.0
        && m.ripple_amp == 0.0
        && m.sheen == 0.0
        && m.translucency == 0.0
        && emissive_off
    {
        return CK_GLOSS; // aniso + refl gate unconstrained (both KEPT).
    }
    CK_UBER
}

/// One class byte per material, in material order (the `mat_cutout` shape).
/// Slices rather than `&Scene` so the gate needs no scene construction;
/// trace.rs calls with `(&scene.materials, &scene.textures)`.
pub fn classify_materials(materials: &[Material], textures: &[Texture]) -> Vec<u8> {
    materials.iter().map(|m| classify(m, textures)).collect()
}

/// The `#define` block a class-k specialized library prepends (Commit B;
/// `--dxr-sbt 1` ignores it — identical code is that rung's whole point).
/// Derived FROM `STRIPS`, never hand-written per class, so the defines and
/// the soundness gate share one source. Float-valued seams force `(0.0)`,
/// bool-valued ones `(false)` — matching each macro's default expression
/// shape in shade.hlsli.
pub fn strip_defines(class: u8) -> String {
    let mut out = String::new();
    for s in STRIPS[class as usize] {
        let (name, off) = match s {
            TexKind => ("SHADE_MAT_TEXKIND", "(false)"),
            Marble => ("SHADE_MAT_MARBLE", "(false)"),
            Normal => ("SHADE_MAT_NORMAL", "(false)"),
            RoughTex => ("SHADE_MAT_ROUGHTEX", "(false)"),
            MetalTex => ("SHADE_MAT_METALTEX", "(false)"),
            EmisTex => ("SHADE_MAT_EMISTEX", "(false)"),
            Ripple => ("SHADE_MAT_RIPPLE", "(0.0)"),
            Sheen => ("SHADE_MAT_SHEEN", "(0.0)"),
            Translucency => ("SHADE_MAT_TRANSLUCENCY", "(0.0)"),
            Transmission => ("SHADE_MAT_TRANSMISSION", "(0.0)"),
            Aniso => ("SHADE_MAT_ANISO", "(0.0)"),
            Refl => ("SHADE_MAT_REFL", "(false)"),
            Emissive => ("SHADE_MAT_EMISSIVE", "(false)"),
        };
        out.push_str(&format!("#define {name}(m) {off}\n"));
    }
    out
}

/// Prove the taxonomy on a material set: every member of every class
/// satisfies every strip obligation its class imposes. This IS the
/// soundness argument — never weaken it to warn-only. Run by `self_test`
/// on the synthetic set and by the `--check-dxr` armed construction audit
/// on the live scene's.
pub fn verify_strips(materials: &[Material], classes: &[u8]) -> Result<(), String> {
    for (i, m) in materials.iter().enumerate() {
        let k = classes[i];
        for s in STRIPS[k as usize] {
            if !strip_ok(*s, m) {
                return Err(format!(
                    "material {i} in class {} violates strip {:?} — the membership \
                     predicate no longer implies the strip set (table drift)",
                    NAMES[k as usize], s
                ));
            }
        }
    }
    Ok(())
}

pub fn histogram(classes: &[u8]) -> [u32; N_CLASSES] {
    let mut h = [0u32; N_CLASSES];
    for &c in classes {
        h[c as usize] += 1;
    }
    h
}

/// `--check` gate (DLL-free, pure): totality on a synthetic all-classes set
/// with adversarial boundary probes, the strip-soundness must-fire, all-8
/// anti-vacuity, determinism, and defines-derive-from-the-table.
pub fn self_test() -> Result<(), String> {
    use glam::Vec3A;
    use image::{DynamicImage, Rgba, RgbaImage};

    // Texture 0 plain, texture 1 alpha-masked (through the real decode path
    // so `alpha_masked` is the derived fact, not a hand-set flag).
    let plain = Texture::from_image(
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, Rgba([255, 255, 255, 255]))),
        true,
    );
    let masked = Texture::from_image(
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, Rgba([255, 255, 255, 0]))),
        true,
    );
    if plain.alpha_masked || !masked.alpha_masked {
        return Err("synthetic textures did not derive alpha_masked as expected".into());
    }
    let textures = vec![plain, masked];

    let base = || Material {
        albedo: Vec3A::splat(0.5),
        roughness: 0.8,
        metallic: 0.0,
        anisotropy: 0.0,
        sheen: 0.0,
        translucency: 0.0,
        transmission: 0.0,
        trans_tint: Vec3A::splat(-1.0),
        ior: 1.5,
        ripple_amp: 0.0,
        emissive: Vec3A::ZERO,
        normal_tex: NO_TEX,
        normal_scale: 1.0,
        height_amp: 0.0,
        rough_tex: NO_TEX,
        metal_tex: NO_TEX,
        emissive_tex: NO_TEX,
        class: crate::matclass::IDX_DEFAULT as u8,
        kind: MatKind::Diffuse,
    };
    #[rustfmt::skip]
    let cases: Vec<(Material, u8)> = vec![
        (base(), CK_LAMBERT),
        (Material { metallic: 1.0, roughness: 0.2, ..base() }, CK_GLOSS),
        (Material { kind: MatKind::Marble { scale: 2.0 }, anisotropy: 0.8, ..base() }, CK_GLOSS),
        (Material { kind: MatKind::Textured { tex: 0 }, normal_tex: 3, ..base() }, CK_TEX_OPAQUE),
        (Material { kind: MatKind::Textured { tex: 1 }, sheen: 0.3, translucency: 0.35, ..base() },
         CK_CUTOUT),
        (Material { transmission: 0.9, roughness: 0.05, ..base() }, CK_GLASS),
        (Material { transmission: 0.9, ripple_amp: 0.25, normal_tex: 3, roughness: 0.05,
                    ..base() }, CK_WATER),
        (Material { emissive: Vec3A::splat(4.0), kind: MatKind::Textured { tex: 0 }, ..base() },
         CK_EMISSIVE),
        // Boundary probes that MUST fall through to uber: sheen'd glass,
        // marble water, anisotropic textured, emissive ripple.
        (Material { transmission: 0.9, sheen: 0.2, ..base() }, CK_UBER),
        (Material { ripple_amp: 0.25, kind: MatKind::Marble { scale: 1.0 }, ..base() }, CK_UBER),
        (Material { kind: MatKind::Textured { tex: 0 }, anisotropy: 0.5, ..base() }, CK_UBER),
        (Material { emissive: Vec3A::ONE, ripple_amp: 0.1, ..base() }, CK_UBER),
    ];
    let expect: Vec<u8> = cases.iter().map(|(_, k)| *k).collect();
    let materials: Vec<Material> = cases.into_iter().map(|(m, _)| m).collect();

    let got = classify_materials(&materials, &textures);
    if got != expect {
        return Err(format!("classification mismatch: got {got:?}, expected {expect:?}"));
    }
    // All 8 classes represented (anti-vacuity: a taxonomy that cannot reach
    // a class would pass every other gate while its shader slot rots).
    let h = histogram(&got);
    if h.iter().any(|&n| n == 0) {
        return Err(format!("synthetic set does not cover all classes: histogram {h:?}"));
    }
    // THE soundness must-fire.
    verify_strips(&materials, &got)?;
    // Determinism (byte-identical across runs — the build-twice discipline).
    if classify_materials(&materials, &textures) != got {
        return Err("classification is not deterministic".into());
    }
    // Defines derive from the table: one #define per stripped macro, uber
    // emits nothing.
    if !strip_defines(CK_UBER).is_empty() {
        return Err("uber must strip nothing".into());
    }
    for k in 0..N_CLASSES as u8 {
        let d = strip_defines(k);
        if d.matches("#define ").count() != STRIPS[k as usize].len() {
            return Err(format!("strip_defines({k}) drifted from STRIPS"));
        }
    }
    Ok(())
}
