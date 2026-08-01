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
        name: "pavement",
        // Bistro's street is ONE authored wet surface (every material is
        // named `Pavement_*`), but the exporter's Ns values are scattered
        // (100/50/30/10/1), so the Ns-tier `>= 100` bar sliced the street
        // into wet-vs-dry patches — `Pavement_Ground_Wet` and
        // `Pavement_Cobblestone_Wet_Leaves` (Ns 30/1!) rendered bone dry at
        // the 0.8 default while their Ns-100 siblings mirrored the lamps.
        // 0.14 == ns_to_rough(100), the parameters the wet half already had,
        // below shade.rs's 0.45 reflection-ray gate. Keyed on `pavement` and
        // `curbstones` ONLY — never `cobble`/`cobblestone`, which are
        // Minecraft BLOCK names (rungholt/vokselia `Cobblestone`,
        // `Cobblestone_Stairs`) that must stay dry-default: a wet Minecraft
        // street is wrong AND puts 6M+ tris under the reflection gate.
        keys: &[Tok("pavement"), Tok("curbstones")],
        pbr: opaque(0.14, 0.0),
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
        name: "bark",
        // Woody plant matter — the foliage-sway WHOLE-PLANT vocabulary (v0.5:
        // `foliage::woody_materials` keys on this class byte to group trunks
        // and branches into plants). The plant-library woody stems moved here
        // FROM the foliage row (brk/bark/tronco/twg/stm — BS04brk,
        // quercus_rubra_bark, sm_tronco, HP07stm, …); trunk/branch/branches
        // are the bistro vocabulary (Foliage_Trunk's TEXTURE stem
        // `italian_cypress_bark_diff` hits Tok("bark") at tier 1, and even at
        // the name tier bark-before-foliage resolves a name carrying both
        // `foliage` and `trunk` tokens correctly — which is why this row must
        // precede the foliage row); `log` is the Minecraft trunk block
        // (whole-token: `Wooden_Plank`'s tokens are `wooden`,`plank` — no
        // hit, so every building block stays static). Placed AFTER stone so
        // every higher class keeps precedence (`Rose_Wood_Table` → wood,
        // `madera_*`/`WOOD08` → wood, the leaf-litter BLENDSHADERs → stone
        // via their pavement stems).
        keys: &[
            Tok("brk"),
            Tok("bark"),
            Tok("tronco"),
            Tok("twg"),
            Tok("stm"),
            Tok("trunk"),
            Tok("branch"),
            Tok("branches"),
            Tok("log"),
        ],
        // Wood-like: opaque, NO translucency (deliberately not foliage's 0.3
        // — a trunk is not backlit like a leaf, and Log/tronco changing
        // shading class should move the look minimally), roughness above the
        // 0.45 bounce gate like foliage.
        pbr: opaque(0.7, 0.0),
    },
    ClassDef {
        name: "foliage",
        // Plant-library tokens (BS01lef, FL19pe13, ...) — every latin genus
        // file carries one of these; `caballo` covers the lone exception
        // (Cola_Caballo horsetail). The WOODY stems (brk/bark/tronco/twg/stm)
        // live in the `bark` row above since v0.5. Roughness deliberately
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
            Tok("flo"),
            Tok("cnt"),
            Tok("sta"),
            Tok("pe"),
            Tok("pis"),
            Tok("hoja"),
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
    "rust", "metal", "wood", "ceramic", "clay", "fabric", "leather", "pavement", "stone",
    "bark", "foliage", "glass", "water", "glossy", "default",
];
/// Public for `foliage::woody_materials` — woody plant matter (trunks,
/// branches, stems), the v0.5 whole-plant grouping anchor.
pub const IDX_BARK: usize = 9;
/// Public for `foliage::leaf_materials` — the classify verdict is retained as
/// `Material::class` (a `u8` index into `NAMES`), and the sway mask compares
/// against this constant.
pub const IDX_FOLIAGE: usize = 10;
const IDX_GLASS: usize = 11;
const IDX_WATER: usize = 12;
const IDX_GLOSSY: usize = 13;
/// Public because it is the `Material::class` byte everywhere the classifier
/// does NOT run (procedural builders, the glTF loader).
pub const IDX_DEFAULT: usize = 14;

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

/// Whole-token `glass`/`pane`/`portal` match (the `water_name` shape;
/// `glassware`/`fiberglass`/`panels` stay safe). Two consumers in
/// `classify`, both on the material NAME (the Minecraft atlas scenes carry
/// no stem signal): it VETOES the untrusted Tf-chromatic water cue
/// (vokselia's `Glass`/`Glass_Pane`/`Portal` are illum 4 with chromatic Tf
/// — without the veto they classify water and ripple), and it ADMITS a
/// named material into the transmission tier (rungholt's `Glass` is
/// illum 2 / Ns 0 — an exporter that never writes illum 4 — and rendered
/// opaque matte without it; bistro's `MASTER_Glass_*` at illum 2/7 are the
/// same shape).
fn glass_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    tokens(&lower).any(|t| t == "glass" || t == "pane" || t == "portal")
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
/// before the foliage vocabulary existed. The `pavement`-token materials now
/// land in the pavement row (which precedes foliage, so table order already
/// resolves their leafy names); the guard remains load-bearing for the
/// CROSS-TIER case (a foliage-textured stem under a cobble name must not
/// sway) and for cobble names without the pavement token
/// (`Cobble_Leaves_Litter` stays default).
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
    water_on: bool,
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
    // A glass NAME also opens the transmission tier: Minecraft-style
    // exporters write their windows as illum 2 / Ns 0 (rungholt's `Glass`,
    // bistro's `MASTER_Glass_*` at illum 2/7), which no Ns/illum signal can
    // ever admit — they rendered opaque matte.
    if illum == Some(4) || ns >= 500.0 || glass_name(mat_name) {
        // A water REFINEMENT of the glass tier — only a material that already
        // classifies glassware can become water, so transmission stays
        // shading-only and every soundness property is unchanged. Signals:
        // an OBJ `water`/`agua` object/stem/material name, or a chromatic Tf
        // — the latter VETOED by a glass name (Tf is untrusted color;
        // vokselia's chromatic-Tf `Glass` panes must not ripple). `water_on`
        // (the --no-water lever) gates ALL FOUR signals — it once gated only
        // hint+Tf at the call site, and rungholt/vokselia water (classified
        // by the NAME cue) ignored the lever entirely.
        let is_water = water_on
            && (water_hint
                || tex_stem.is_some_and(water_name)
                || water_name(mat_name)
                || (tf_chromatic(tf) && !glass_name(mat_name)));
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
        let got = NAMES[classify(stem, name, Some(ns), Some(illum), false, None, true).0];
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
    // The bark class (v0.5 whole-plant sway): woody stems moved out of the
    // foliage row — plant-library bark/stm, the bistro tree textures, the
    // Minecraft Log block.
    tex("quercus_rubra_bark", "bark")?;
    tex("hp07stm", "bark")?; // stm moved from foliage
    tex("italian_cypress_bark_diff", "bark")?; // bistro Foliage_Trunk's stem
    tex("linden_bark_a_diff", "bark")?; // bistro linden trunk's stem
    tex("paris_ivy_branch_diff", "bark")?; // bistro ivy branches' stem
    // Name-tier bark: `Foliage_Trunk` carries BOTH the `foliage` and `trunk`
    // tokens — bark-before-foliage table order resolves it (the reason the
    // bark row's position is load-bearing).
    expect(None, "Foliage_Trunk", 100.0, 2, "bark")?;
    expect(Some("rungholt-rgba"), "Log", 0.0, 2, "bark")?;
    expect(Some("vokselia_spawn"), "Log", 0.0, 2, "bark")?;
    // ...and the whole-token guard: every wooden BUILDING block stays static.
    expect(Some("rungholt-rgba"), "Wooden_Plank", 0.0, 2, "default")?;
    tex("d30_smiguel_2003_7758", "default")?;
    // The bistro/Minecraft vocabulary (foliage-sway coverage):
    tex("leaves_a_diff", "foliage")?; // linden — "leaves" plural, not "leaf"
    tex("paris_foliage_01a_diff", "foliage")?;
    tex("paris_interior_plants_01_diff", "foliage")?;
    tex("plastic_01_planters_diff", "default")?; // `planters` != `plants` (whole-token)
    // The pavement class: bistro's whole `Pavement_*` street family is ONE
    // authored wet surface — every member classifies pavement regardless of
    // the exporter's scattered Ns (100/50/30/10/1), which used to slice the
    // street into wet-vs-dry patches at the Ns tier's >= 100 bar.
    tex("pavement_cobblestone_01_b_diff", "pavement")?;
    // Name tier: stems without a pavement token (`ground_wet_01_diff`,
    // `cobble_02b_diff`) reach the row via their `Pavement_*` names — the
    // two that rendered DRY while literally named Wet.
    expect(None, "Pavement_Ground_Wet", 30.0, 2, "pavement")?;
    expect(None, "Pavement_Cobblestone_Wet_Leaves_BLENDSHADER", 1.0, 2, "pavement")?;
    expect(Some("pavement_cobblestone_03_diff"), "Pavement_Cobblestone_02", 10.0, 2, "pavement")?;
    expect(Some("paris_curbstones_01_diff"), "Pavement_Curbstones", 100.0, 2, "pavement")?;
    // Minecraft BLOCK safety: `cobble`/`cobblestone` are deliberately NOT
    // pavement keys — rungholt/vokselia's blocks must stay dry-default (a
    // wet Minecraft street is wrong AND crosses the reflection gate on 6M+
    // tris). Do not add those tokens to the pavement row.
    expect(Some("rungholt-rgba"), "Cobblestone", 0.0, 2, "default")?;
    expect(Some("vokselia_spawn"), "Cobblestone_Stairs", 0.0, 2, "default")?;
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
    // Leaf-LITTER pavement: the stem's pavement token wins at tier 1 (and
    // the pavement row precedes foliage, so even the leafy NAME resolves) —
    // never foliage, never stone.
    expect(
        Some("pavement_cobblestone_01_b_diff"),
        "Pavement_Cobble_Leaves_BLENDSHADER",
        100.0,
        2,
        "pavement",
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
        true,
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
    let (_, g) = classify(None, "material_0", Some(4096.0), Some(2), false, None, true);
    if !(g.transmission > 0.0 && g.roughness <= 0.06) {
        return Err(format!("Ns 4096: transmission {} roughness {}", g.transmission, g.roughness));
    }
    let (_, f) = classify(Some("tela_mesa_b"), "material_2", Some(16.0), Some(2), false, None, true);
    if f.sheen != FABRIC_SHEEN || f.transmission != 0.0 {
        return Err(format!("fabric: sheen {} transmission {}", f.sheen, f.transmission));
    }
    let (_, l) = classify(Some("bs01lef"), "material_3", Some(16.0), Some(2), false, None, true);
    if l.translucency != 0.3 || l.roughness < 0.45 {
        return Err(format!("foliage: translucency {} roughness {}", l.translucency, l.roughness));
    }
    // Bark Pbr: wood-like — opaque (NO leaf translucency), above the bounce
    // gate, dielectric.
    let (_, bk) = classify(Some("linden_bark_a_diff"), "material_4", Some(16.0), Some(2), false, None, true);
    if bk.translucency != 0.0 || bk.roughness < 0.45 || bk.transmission != 0.0 || bk.metallic != 0.0
    {
        return Err(format!(
            "bark pbr: translucency {} roughness {} transmission {} metallic {}",
            bk.translucency, bk.roughness, bk.transmission, bk.metallic
        ));
    }
    // Water is a glass-tier refinement, not a rival to the keyword/glossy
    // tiers. Signals, each in isolation:
    let water_named = |stem: Option<&str>, name: &str, hint: bool, tf: Option<[f32; 3]>| {
        NAMES[classify(stem, name, Some(1024.0), Some(4), hint, tf, true).0]
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
    if NAMES[classify(None, "water_chair", Some(256.0), Some(2), true, None, true).0] != "glossy" {
        return Err("water hint must not override the opaque glossy tier".into());
    }
    // Token safety: `watercolor` is not water; `Water` is.
    if !water_name("Water") || water_name("watercolor_paper") {
        return Err("water_name token match wrong".into());
    }
    // Glass-NAMED materials are glass, never water: the Tf-chromatic cue is
    // untrusted color and a trusted name vetoes it (vokselia's `Glass` panes
    // carry `Tf 0.376 0.482 0.498` — chromatic Δ 0.122 — and its `Portal`
    // `Tf 0.668 0.398 0.8`; both are illum 4, and without the veto they
    // classify water and RIPPLE).
    if water_named(None, "Glass", false, Some([0.376, 0.482, 0.498])) != "glass" {
        return Err("glass-named chromatic-Tf material must stay glass".into());
    }
    if water_named(None, "Portal", false, Some([0.668, 0.398, 0.8])) != "glass" {
        return Err("portal-named chromatic-Tf material must stay glass".into());
    }
    // ...and a glass name ADMITS an illum-2/Ns-0 material into the
    // transmission tier (rungholt's Minecraft exporter never writes illum 4
    // on glass — those windows rendered opaque matte; bistro's `MASTER_Glass_*`
    // at illum 2/7, Ns 80-200 are the same shape). Water-named materials keep
    // NOT being admitted (the water_chair pin above): water is a refinement
    // of a tier the material must reach on its own or via a GLASS name.
    expect(Some("rungholt-rgba"), "Glass", 0.0, 2, "glass")?;
    expect(Some("rungholt-rgba"), "Glass_Pane", 0.0, 2, "glass")?;
    expect(None, "MASTER_Glass_Exterior", 80.0, 2, "glass")?;
    // Glass-token safety: compounds don't fire, the real names do.
    if glass_name("glassware_shelf") || glass_name("fiberglass_panel") || glass_name("panels") {
        return Err("glass_name compound token must not match".into());
    }
    if !glass_name("Glass_Pane") || !glass_name("Portal") {
        return Err("glass_name must match Glass_Pane/Portal".into());
    }
    // The lever (`--no-water` -> water_on = false) suppresses EVERY water
    // signal — name, stem, object hint, AND chromatic Tf — regressing the
    // material to plain glassware. Pinned on the rungholt shape (a
    // name-classified `Stationary_Water`) because the lever once gated only
    // hint+Tf at the call site and name-classified Minecraft water ignored
    // `--no-water` entirely.
    if NAMES[classify(None, "Stationary_Water", Some(0.0), Some(4), true, Some([0.9, 0.8, 0.4]), false).0]
        != "glass"
    {
        return Err("--no-water must suppress every water signal (name incl.)".into());
    }
    if water_named(None, "Stationary_Water", false, None) != "water" {
        return Err("water material name should classify water when armed".into());
    }
    // The water params: transmission near 1, mirror-smooth, flagged.
    let (_, w) = classify(None, "materialo", Some(1024.0), Some(4), true, None, true);
    if !(w.water && w.transmission > 0.95 && w.roughness <= 0.06 && w.metallic == 0.0) {
        return Err(format!(
            "water pbr: water {} T {} rough {} metal {}",
            w.water, w.transmission, w.roughness, w.metallic
        ));
    }
    Ok(())
}
