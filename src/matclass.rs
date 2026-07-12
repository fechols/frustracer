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
}

const fn opaque(roughness: f32, metallic: f32) -> Pbr {
    Pbr { roughness, metallic, sheen: 0.0, translucency: 0.0, transmission: 0.0 }
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
pub const FABRIC_SHEEN: f32 = 0.5;

/// Class names in report order: the keyword classes, then the Ns tiers and
/// the default — indices returned by `classify` point in here.
pub const NAMES: &[&str] = &[
    "rust", "metal", "wood", "ceramic", "clay", "fabric", "leather", "stone", "foliage",
    "glass", "glossy", "default",
];
const IDX_GLASS: usize = 9;
const IDX_GLOSSY: usize = 10;
const IDX_DEFAULT: usize = 11;

/// Blinn-Phong exponent -> perceptual GGX roughness (Brian Karis' mapping),
/// clamped to the plausible glossy band.
fn ns_to_rough(ns: f32) -> f32 {
    (2.0 / (ns + 2.0)).sqrt().clamp(0.05, 0.4)
}

fn tokens(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !c.is_ascii_alphabetic()).filter(|t| !t.is_empty())
}

fn keyword_class(key: &str) -> Option<usize> {
    for (ci, c) in CLASSES.iter().enumerate() {
        for k in c.keys {
            let hit = match k {
                Tok(t) => tokens(key).any(|tok| tok == *t),
                Sub(sub) => key.contains(sub),
            };
            if hit {
                return Some(ci);
            }
        }
    }
    None
}

/// Classify one MTL material. `tex_stem` is the lowercase filename stem of
/// map_Kd (no directory, no extension); `mat_name` the `newmtl` name;
/// `ns`/`illum` straight from tobj. Returns (index into `NAMES`, params).
pub fn classify(
    tex_stem: Option<&str>,
    mat_name: &str,
    ns: Option<f32>,
    illum: Option<u8>,
) -> (usize, Pbr) {
    // Tier 1: texture filename stem. Tier 2: material name, same table.
    let hit = tex_stem
        .and_then(keyword_class)
        .or_else(|| keyword_class(&mat_name.to_ascii_lowercase()));
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
        let got = NAMES[classify(stem, name, Some(ns), Some(illum)).0];
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
    // Name tier: CafeChair_Metal is untextured.
    expect(None, "CafeChair_Metal", 256.0, 2, "metal")?;
    // Ns tiers: glassware (illum 4 or Ns >= 500) vs untextured opaque glossy.
    expect(None, "material_79", 1024.0, 4, "glass")?;
    expect(None, "materialn", 100.0, 4, "glass")?;
    expect(None, "material_267", 256.0, 2, "glossy")?;
    expect(None, "material_9", 16.0, 2, "default")?;
    let (_, g) = classify(None, "material_0", Some(4096.0), Some(2));
    if !(g.transmission > 0.0 && g.roughness <= 0.06) {
        return Err(format!("Ns 4096: transmission {} roughness {}", g.transmission, g.roughness));
    }
    let (_, f) = classify(Some("tela_mesa_b"), "material_2", Some(16.0), Some(2));
    if f.sheen != FABRIC_SHEEN || f.transmission != 0.0 {
        return Err(format!("fabric: sheen {} transmission {}", f.sheen, f.transmission));
    }
    let (_, l) = classify(Some("bs01lef"), "material_3", Some(16.0), Some(2));
    if l.translucency != 0.3 || l.roughness < 0.45 {
        return Err(format!("foliage: translucency {} roughness {}", l.translucency, l.roughness));
    }
    Ok(())
}
