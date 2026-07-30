//! MTL material classification for real (OBJ) scenes.
//!
//! San Miguel's material names are anonymous (`material_204`) and its MTL
//! specular data is exporter-flattened (242/287 materials share Ks 0.04 /
//! Ns 16), but the map_Kd texture FILENAMES are richly descriptive — Spanish
//! scene vocabulary (`madera` wood, `fierro` iron, `muros` walls, `tela`
//! cloth) plus a plant-library convention (`*lef*`/`*pet*`/`*stm*`/`bark`).
//! `classify` keys on the texture filename stem first, then the material
//! name (catches the one descriptive name, `CafeChair_Metal`), then an
//! Ns/illum heuristic for the untextured glassware, then a default equal to
//! the old hardcoded parameters. Purely a load-time decision — nothing here
//! runs per ray.

/// Per-class PBR parameters handed to `SceneBuilder::material_full`.
/// Anisotropy is deliberately absent: the tangent frame in shade.rs is
/// circumferential around world-up (lathe-spun), wrong for arbitrary OBJ
/// geometry.
#[derive(Clone, Copy)]
pub struct Pbr {
    pub roughness: f32,
    pub metallic: f32,
    pub sheen: f32,
    pub translucency: f32,
    pub transmission: f32,
    /// A liquid-water refinement of the glass tier (see `WATER`): the loader
    /// reads it to skip the dark-glass albedo lift and to fill the water
    /// `trans_tint`/`ior`/`ripple_amp` material fields. `false` for every
    /// keyword/glossy/glass class — structural bit-identity for non-water.
    pub water: bool,
}

const fn opaque(roughness: f32, metallic: f32) -> Pbr {
    Pbr { roughness, metallic, sheen: 0.0, translucency: 0.0, transmission: 0.0, water: false }
}

/// Keyword: `Tok` matches a whole lowercase token (tokens are maximal
/// alphabetic runs — digits/`_`/`.`/`-` separate), `Sub` a raw substring of
/// the lowercase stem. Whole-token matching is what keeps the short plant
/// tokens safe (`pet` never fires inside `Carpet`, `pis` not inside `piso`);
/// `Sub` exists only for fused Spanish compounds (`detmoldura`,
/// `molduraterraza`).
enum Key {
    Tok(&'static str),
    Sub(&'static str),
}

struct ClassDef {
    name: &'static str,
    keys: &'static [Key],
    pbr: Pbr,
}

use Key::{Sub, Tok};

/// Ordered table — first class with any key hit wins, so the order is
/// load-bearing: `rust` before `metal` (`metal_viejo_2` is rusty oxide, not
/// polished iron), `metal` before `clay` (`Forja_Macetas` is the wrought-iron
/// pot STAND), `clay` before `stone` (`pared_barro_afinado` is adobe), and
/// `foliage` last because its tokens are the shortest.
const CLASSES: &[ClassDef] = &[
    ClassDef {
        name: "rust",
        keys: &[Tok("rust"), Tok("viejo")],
        // Oxide is a dielectric; the remnant metallic still buys a bounce ray.
        pbr: opaque(0.85, 0.15),
    },
    ClassDef {
        name: "metal",
        keys: &[Tok("fierro"), Tok("forja"), Tok("metal")],
        pbr: opaque(0.42, 0.90),
    },
    ClassDef {
        name: "wood",
        keys: &[
            Tok("madera"),
            Tok("wood"),
            Tok("vigas"),
            Tok("puerta"),
            Tok("marco"),
            Tok("marcos"),
            Tok("triplay"),
            Tok("barandal"),
        ],
        pbr: opaque(0.62, 0.0),
    },
    ClassDef {
        name: "ceramic",
        keys: &[
            Tok("azulejo"),
            Tok("talabera"),
            Tok("ceramic"),
            Tok("tile"),
            Tok("plato"),
            Tok("cenicero"),
        ],
        // Glazed: below the 0.45 bounce gate — plates and tiles reflect.
        pbr: opaque(0.22, 0.0),
    },
    ClassDef {
        name: "clay",
        keys: &[Tok("barro"), Tok("maceta"), Tok("macetas"), Tok("jardinera")],
        pbr: opaque(0.75, 0.0),
    },
    ClassDef {
        name: "fabric",
        keys: &[Tok("tela"), Tok("carpet"), Tok("individual")],
        pbr: opaque(0.95, 0.0),
    },
    ClassDef {
        name: "leather",
        keys: &[Tok("piel")],
        pbr: opaque(0.72, 0.0),
    },
    ClassDef {
        name: "stone",
        keys: &[
            Tok("muros"),
            Tok("pared"),
            Tok("cantera"),
            Tok("concreto"),
            Tok("piedra"),
            Tok("piso"),
            Tok("losa"),
            Tok("techo"),
            Tok("columna"),
            Tok("arco"),
            Tok("arcos"),
            Tok("escalera"),
            Tok("bwk"),
            Tok("terresable"),
            Sub("moldur"),
        ],
        pbr: opaque(0.88, 0.0),
    },
    ClassDef {
        name: "foliage",
        // Plant-library tokens (BS01lef, FL19pe13, HP07stm, quercus_rubra_bark
        // ...) — every latin genus file carries one of these; `caballo` covers
        // the lone exception (Cola_Caballo horsetail). Roughness deliberately
        // stays >= 0.45: 10M foliage tris must not take the bounce ray.
        // The plain-English row (leaves/foliage/sapling/plants) is the
        // bistro + Minecraft vocabulary: bistro's stems say `Leaves_A_diff` /
        // `Paris_Foliage_01a_diff` and its material names all start
        // `Foliage_`; rungholt/vokselia carry the signal ONLY on the material
        // NAME (`Leaves`, `Sapling` — one shared atlas texture). Deliberately
        // NO bare "grass": Minecraft's `Grass` is the GROUND block, and the
        // sway mask must never mark terrain. The billboard PLANTS are named
        // exactly instead (2026-07-29 — `Tall_Grass` was the accepted miss
        // until the user asked for grass sway): `Sub` on the underscored
        // block names (`tall_grass`, `sugar_cane` — effectively exact, since
        // `tokens()` splits on `_` and a bare grass/cane token would be the
        // terrain hazard) + whole-token flower/crop names. `rose_wood`-style
        // compounds stay safe by TABLE ORDER (wood classifies first).
        keys: &[
            Tok("lef"),
            Tok("leaf"),
            Tok("pet"),
            Tok("petal"),
            Tok("stm"),
            Tok("brk"),
            Tok("bark"),
            Tok("flo"),
            Tok("cnt"),
            Tok("twg"),
            Tok("sta"),
            Tok("pe"),
            Tok("pis"),
            Tok("hoja"),
            Tok("tronco"),
            Tok("seca"),
            Tok("caballo"),
            Tok("leaves"),
            Tok("foliage"),
            Tok("sapling"),
            Tok("plants"),
            Sub("tall_grass"),
            Sub("sugar_cane"),
            Tok("dandelion"),
            Tok("rose"),
            Tok("crops"),
        ],
        // 0.3 ≈ measured leaf transmittance (chlorophyll passes ~20-30% in
        // the visible band). 0.5 was tried: visually fine, but the brighter
        // back-lit canopy makes single-probe fireflies in the unpaired GI
        // signed A/B ~40% larger for no physical gain.
        pbr: Pbr { translucency: 0.3, ..opaque(0.55, 0.0) },
    },
];

/// Ns-tier classes for materials nothing above matched. Index into `NAMES`
/// continues past `CLASSES`.
const GLASS: Pbr = Pbr { transmission: 0.9, ..opaque(0.05, 0.0) };
/// Liquid water: a refinement of the glass tier for the fountain (`materialo`
/// in San Miguel). Transmission near 1 so Fresnel owns the reflect/transmit
/// split (glass 0.9 left a `kd·(1−T)` neutral wash that read as chrome); the
/// blue-green `trans_tint`, the 1.33 IOR, and the ripple amplitude are
/// material fields the loader fills (see `scene::WATER_*`). Roughness stays
/// mirror-smooth — the ripple normals, not micro-roughness, supply the
/// water structure.
const WATER: Pbr = Pbr { transmission: 0.97, water: true, ..opaque(0.05, 0.0) };
pub const FABRIC_SHEEN: f32 = 0.5;

/// Class names in report order: the keyword classes, then the Ns tiers and
/// the default — indices returned by `classify` point in here.
pub const NAMES: &[&str] = &[
    "rust", "metal", "wood", "ceramic", "clay", "fabric", "leather", "stone", "foliage",
    "glass", "water", "glossy", "default",
];
/// Public for `foliage::leaf_materials` — the classify verdict is retained as
/// `Material::class` (a `u8` index into `NAMES`), and the sway mask compares
/// against this constant.
pub const IDX_FOLIAGE: usize = 8;
const IDX_GLASS: usize = 9;
const IDX_WATER: usize = 10;
const IDX_GLOSSY: usize = 11;
/// Public because it is the `Material::class` byte everywhere the classifier
/// does NOT run (procedural builders, the glTF loader).
pub const IDX_DEFAULT: usize = 12;

/// Blinn-Phong exponent -> perceptual GGX roughness (Brian Karis' mapping),
/// clamped to the plausible glossy band.
fn ns_to_rough(ns: f32) -> f32 {
    (2.0 / (ns + 2.0)).sqrt().clamp(0.05, 0.4)
}

fn tokens(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !c.is_ascii_alphabetic()).filter(|t| !t.is_empty())
}

/// Whole-token `water`/`agua` match (the `tokens()` matcher keeps
/// `watercolor` safe). Checked against the OBJ object/group name at the
/// loader (a `usemtl materialo` under `o Water`) and against the texture
/// stem / material name inside `classify`.
pub fn water_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    tokens(&lower).any(|t| t == "water" || t == "agua")
}

/// A chromatic MTL transmission filter (`Tf`) on an illum-4 material is the
/// structural "colored liquid, not clear glassware" signal — `materialo`'s
/// `Tf 0.5 0.4 0.2` fires; neutral glassware (`Tf 0.1 0.1 0.1`) does not. The
/// color itself is NOT used (2003-exporter Tf is untrustworthy as a hue — the
/// curated `WATER_TINT` is always the tint); only its chromaticity is the cue.
fn tf_chromatic(tf: Option<[f32; 3]>) -> bool {
    match tf {
        Some([r, g, b]) => r.max(g).max(b) - r.min(g).min(b) > 0.05,
        None => false,
    }
}

fn key_hits(keys: &[Key], key: &str) -> bool {
    keys.iter().any(|k| match k {
        Tok(t) => tokens(key).any(|tok| tok == *t),
        Sub(sub) => key.contains(sub),
    })
}

fn keyword_class(key: &str) -> Option<usize> {
    CLASSES.iter().position(|c| key_hits(c.keys, key))
}

/// A foliage BLOCK, not a class. Pavement carries leaf-litter names
/// (bistro's `Pavement_Cobble_Leaves_BLENDSHADER`), and the sway mask must
/// never mark a street — but the material must otherwise classify exactly as
/// if the foliage vocabulary had never matched. Foliage v0.1 instead pinned
/// these tokens to STONE, and that broke the night street: bistro's Ns-100
/// cobbles went glossy (Ns tier, rough ~0.14) → stone (0.88), crossing the
/// 0.45 reflection-ray gate in shade.rs — the emissive lamp/sign reflections
/// on the wet cobblestones vanished, and with no map_Pr in the MTL the flat
/// 0.88 also flattened the GGX highlights. A guarded material skips ONLY the
/// foliage row (either tier) and falls through to whatever tier it hit
/// before the foliage vocabulary existed.
const FOLIAGE_GUARD: &[Key] = &[Tok("pavement"), Tok("cobblestone"), Tok("cobble")];

/// Classify one MTL material. `tex_stem` is the lowercase filename stem of
/// map_Kd (no directory, no extension); `mat_name` the `newmtl` name;
/// `ns`/`illum` straight from tobj. `water_hint` is the loader's OBJ
/// object-name signal (`o Water` → this material); `tf` the parsed MTL
/// transmission filter. Returns (index into `NAMES`, params).
pub fn classify(
    tex_stem: Option<&str>,
    mat_name: &str,
    ns: Option<f32>,
    illum: Option<u8>,
    water_hint: bool,
    tf: Option<[f32; 3]>,
) -> (usize, Pbr) {
    // Tier 1: texture filename stem. Tier 2: material name, same table.
    // FOLIAGE_GUARD suppresses the foliage row only (see the const): a
    // guarded material classifies as if the foliage vocabulary never matched
    // — bistro's Ns-100 cobbles land in the Ns tier's glossy, under the
    // reflection gate, which is what mirrors the night street's lamps.
    let name_lower = mat_name.to_ascii_lowercase();
    let guarded = tex_stem.is_some_and(|s| key_hits(FOLIAGE_GUARD, s))
        || key_hits(FOLIAGE_GUARD, &name_lower);
    let class_at =
        |key: &str| keyword_class(key).filter(|&ci| !(guarded && ci == IDX_FOLIAGE));
    let hit = tex_stem.and_then(class_at).or_else(|| class_at(&name_lower));
    if let Some(ci) = hit {
        let mut pbr = CLASSES[ci].pbr;
        if CLASSES[ci].name == "fabric" {
            pbr.sheen = FABRIC_SHEEN;
        }
        return (ci, pbr);
    }
    // Tier 3: Ns/illum heuristic — only meaningful signal survives (the
    // exporter-flattened majority sits at Ns 16, below both bands). illum 4
    // ("transparent, ray trace on") or a very high exponent = glassware;
    // the mid band (the untextured café-chair metal at Ns 256) stays an
    // opaque glossy dielectric — transmission there would dissolve chairs.
    let ns = ns.unwrap_or(0.0);
    if illum == Some(4) || ns >= 500.0 {
        // A water REFINEMENT of the glass tier — only a material that already
        // classifies glassware can become water, so transmission stays
        // shading-only and every soundness property is unchanged. Signals:
        // an OBJ `water`/`agua` object/stem/material name, or a chromatic Tf.
        let is_water = water_hint
            || tex_stem.is_some_and(water_name)
            || water_name(mat_name)
            || tf_chromatic(tf);
        if is_water {
            return (IDX_WATER, WATER);
        }
        return (IDX_GLASS, Pbr { roughness: ns_to_rough(ns.max(500.0)), ..GLASS });
    }
    if ns >= 100.0 {
        return (IDX_GLOSSY, opaque(ns_to_rough(ns), 0.0));
    }
    // Tier 4: the old hardcode.
    (IDX_DEFAULT, opaque(0.8, 0.0))
}

/// Deterministic spot checks over the real San Miguel naming patterns —
/// run by `--check` like the other pure-module self tests. Every assert is a
/// case that once mattered: precedence (rust>metal>clay>stone), whole-token
/// safety (`pis` inside `piso`, `pet` inside `Carpet`), the fused `moldur`
/// compounds, the untextured name/Ns/illum tiers.
pub fn self_test() -> Result<(), String> {
    let expect = |stem: Option<&str>, name: &str, ns: f32, illum: u8, want: &str| {
        let got = NAMES[classify(stem, name, Some(ns), Some(illum), false, None).0];
        if got == want {
            Ok(())
        } else {
            Err(format!("{stem:?}/{name}/Ns{ns}/illum{illum}: got {got}, want {want}"))
        }
    };
    let tex = |stem: &str, want: &str| expect(Some(stem), "material_1", 16.0, 2, want);
    tex("madera_barandal_esc_2", "wood")?;
    tex("metal_viejo_2", "rust")?; // precedence: rust before metal
    tex("forja_macetas", "metal")?; // metal before clay
    tex("pared_barro_afinado", "clay")?; // clay before stone
    tex("finishes.flooring.carpet.loop.5", "fabric")?; // `pet`/`flo` must not fire inside
    tex("fl19pe13", "foliage")?;
    tex("fl12pis1", "foliage")?; // `pis` token...
    tex("piso_patio_exterior", "stone")?; // ...but not inside `piso`
    tex("detmoldura_01_color", "stone")?; // fused compounds via Sub("moldur")
    tex("molduraterraza__color", "stone")?;
    tex("silla_d_piel", "leather")?;
    tex("quercus_rubra_bark", "foliage")?;
    tex("d30_smiguel_2003_7758", "default")?;
    // The bistro/Minecraft vocabulary (foliage-sway coverage):
    tex("leaves_a_diff", "foliage")?; // linden — "leaves" plural, not "leaf"
    tex("paris_foliage_01a_diff", "foliage")?;
    tex("paris_interior_plants_01_diff", "foliage")?;
    tex("plastic_01_planters_diff", "default")?; // `planters` != `plants` (whole-token)
    // Guarded pavement is NOT stone (the foliage-v0.1 regression): it falls
    // through the keyword tiers entirely (here to the Ns default).
    tex("pavement_cobblestone_01_b_diff", "default")?;
    // Name tier: CafeChair_Metal is untextured.
    expect(None, "CafeChair_Metal", 256.0, 2, "metal")?;
    // Minecraft atlas scenes: ONE shared texture, so the stem carries no
    // signal and the material NAME is the whole vocabulary.
    expect(Some("rungholt-rgba"), "Leaves", 0.0, 2, "foliage")?;
    expect(Some("vokselia_spawn"), "Sapling", 0.0, 2, "foliage")?;
    // The GROUND block must never classify foliage — the sway mask marks
    // foliage-classed cutout materials, and terrain must not sway. This pin
    // is load-bearing on VOKSELIA, whose single atlas is alpha-masked for
    // every material (rungholt's `Grass` also fails the mask gate — it maps
    // the RGB atlas — but the class byte must hold alone).
    expect(Some("rungholt-rgba"), "Grass", 0.0, 2, "default")?;
    expect(Some("vokselia_spawn"), "Grass", 0.0, 2, "default")?;
    // The billboard plants (the exact-name vocabulary — grass sway): every
    // cutout cross-plant classifies foliage on BOTH atlas scenes.
    for plant in ["Tall_Grass", "Dandelion", "Rose", "Crops", "Sugar_Cane"] {
        expect(Some("rungholt-rgba"), plant, 0.0, 2, "foliage")?;
        expect(Some("vokselia_spawn"), plant, 0.0, 2, "foliage")?;
    }
    // Table order keeps compounds out: a rose-named WOOD hits the wood row
    // before the foliage "rose" token is ever consulted.
    expect(None, "Rose_Wood_Table", 16.0, 2, "wood")?;
    expect(None, "Foliage_Bux_Hedges46", 100.0, 2, "foliage")?;
    // Leaf-LITTER pavement: the guard suppresses the foliage "leaves" name
    // token and the material falls through to the Ns tier — bistro's Ns-100
    // cobbles are GLOSSY, which is the night street's whole look (the
    // reflection-ray gate reads the flat class roughness).
    expect(
        Some("pavement_cobblestone_01_b_diff"),
        "Pavement_Cobble_Leaves_BLENDSHADER",
        100.0,
        2,
        "glossy",
    )?;
    // ...and the roughness must stay under shade.rs's 0.45 reflection gate,
    // or the emissive lamp reflections on the cobbles die again.
    let (_, pave) = classify(
        Some("pavement_cobblestone_01_b_diff"),
        "Pavement_Cobble_Leaves_BLENDSHADER",
        Some(100.0),
        Some(2),
        false,
        None,
    );
    if pave.roughness >= 0.45 {
        return Err(format!("pavement roughness {} crossed the reflection gate", pave.roughness));
    }
    // The guard must block foliage on the NAME tier too (an untextured or
    // atlas-textured cobble material with a leafy name must never sway).
    expect(None, "Cobble_Leaves_Litter", 16.0, 2, "default")?;
    // Ns tiers: glassware (illum 4 or Ns >= 500) vs untextured opaque glossy.
    // material_79/materialn are neutral-Tf illum-4 glassware — must STAY glass.
    expect(None, "material_79", 1024.0, 4, "glass")?;
    expect(None, "materialn", 100.0, 4, "glass")?;
    expect(None, "material_267", 256.0, 2, "glossy")?;
    expect(None, "material_9", 16.0, 2, "default")?;
    let (_, g) = classify(None, "material_0", Some(4096.0), Some(2), false, None);
    if !(g.transmission > 0.0 && g.roughness <= 0.06) {
        return Err(format!("Ns 4096: transmission {} roughness {}", g.transmission, g.roughness));
    }
    let (_, f) = classify(Some("tela_mesa_b"), "material_2", Some(16.0), Some(2), false, None);
    if f.sheen != FABRIC_SHEEN || f.transmission != 0.0 {
        return Err(format!("fabric: sheen {} transmission {}", f.sheen, f.transmission));
    }
    let (_, l) = classify(Some("bs01lef"), "material_3", Some(16.0), Some(2), false, None);
    if l.translucency != 0.3 || l.roughness < 0.45 {
        return Err(format!("foliage: translucency {} roughness {}", l.translucency, l.roughness));
    }
    // Water is a glass-tier refinement, not a rival to the keyword/glossy
    // tiers. Signals, each in isolation:
    let water_named = |stem: Option<&str>, name: &str, hint: bool, tf: Option<[f32; 3]>| {
        NAMES[classify(stem, name, Some(1024.0), Some(4), hint, tf).0]
    };
    // (a) the OBJ object name (`o Water` → materialo), no texture, no Tf.
    if water_named(None, "materialo", true, None) != "water" {
        return Err("object-name water_hint should classify water".into());
    }
    // (b) a chromatic Tf alone (materialo's Tf 0.5 0.4 0.2), no name hint.
    if water_named(None, "materialo", false, Some([0.5, 0.4, 0.2])) != "water" {
        return Err("chromatic Tf should classify water".into());
    }
    // (c) a `water` stem or material name.
    if water_named(Some("agua_fuente"), "material_5", false, None) != "water" {
        return Err("water stem should classify water".into());
    }
    // Neutral Tf + no name = glassware still (the material_79 pin, restated
    // with an explicit neutral Tf).
    if water_named(None, "material_79", false, Some([0.1, 0.1, 0.1])) != "glass" {
        return Err("neutral-Tf glassware must stay glass".into());
    }
    // Refinement-only: a water name on an illum-2 opaque material does NOT
    // pull it into the glass tier.
    if NAMES[classify(None, "water_chair", Some(256.0), Some(2), true, None).0] != "glossy" {
        return Err("water hint must not override the opaque glossy tier".into());
    }
    // Token safety: `watercolor` is not water; `Water` is.
    if !water_name("Water") || water_name("watercolor_paper") {
        return Err("water_name token match wrong".into());
    }
    // The water params: transmission near 1, mirror-smooth, flagged.
    let (_, w) = classify(None, "materialo", Some(1024.0), Some(4), true, None);
    if !(w.water && w.transmission > 0.95 && w.roughness <= 0.06 && w.metallic == 0.0) {
        return Err(format!(
            "water pbr: water {} T {} rough {} metal {}",
            w.water, w.transmission, w.roughness, w.metallic
        ));
    }
    Ok(())
}
