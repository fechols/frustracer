//! World / biomes mode: the flagless-interactive default scene. Every curated
//! scene on disk loads through its normal loader (warm `.fcache` where the
//! sidecar exists), gets fitted-and-finished exactly as a positional-arg run
//! would, and is then MERGED post-hoc into one flat `Scene` — islands on a
//! ring over one shared ground plane, ordered by their theme hour so a lap
//! around the archipelago sweeps the day. The merge is `tile_scene`'s
//! flattened-replication idiom generalized to HETEROGENEOUS parts: where
//! tiling shares one base's materials/textures untouched, merging distinct
//! scenes must offset material ids (`tri_mat`), texture ids (every map-index
//! field, preserving the `NO_TEX` sentinel), and vertex indices per part.
//!
//! One sky by construction: the merged scene takes `default_sun()` and one
//! `finalize_scalars` (one SH dome). Biome identity comes from the content
//! itself plus the GLOBAL time-of-day (the flycam attractors ease `tod`
//! toward the nearest island's theme hour at the manual-scrub rate, so the
//! TOD-delta contract — accumulation reset only, histories kept — applies
//! verbatim; see flycam.rs).
//!
//! The WORLD SIDECAR (`scenes/world.fcache`, `scene_cache::try_load_world`/
//! `store_world`) caches the merged Scene + world BVH + `World` layout in
//! one file: a warm boot skips every per-part load, the merge, and the world
//! BVH build. Its key covers every curated entry's identity, PRESENCE, and
//! dependency stats (`build_world_key` — source + mtl for OBJ, source +
//! external buffers/images for glTF via `gltf_loader::dependency_files`),
//! plus the ring constants; glTF textures ride the cache as INLINE texels
//! (their synthetic sources have no file to re-decode from).
//!
//! Deliberately NOT here: instancing (the whole correctness architecture
//! assumes one flat BVH) and per-region skies (the one-sky invariant).
//! Follow-ons: per-island firefly scale, cross-part texture dedup,
//! zstd-compressing the sidecar payload.

use crate::scene::{self, MatKind, Material, Scene, NO_TEX};
use glam::{Vec2, Vec3A};

/// One curated island: a scene the world loads if its file exists. `mtris`
/// is print/budget metadata only (the real count comes from the load).
pub struct IslandSpec {
    pub name: &'static str,
    pub path: &'static str,
    /// Time-of-day this island is themed for (float hours, 0..24). The table
    /// is hour-ordered so the ring layout sweeps the day monotonically.
    pub theme_hour: f32,
    pub mtris: f32,
}

/// The curated set. Deliberate SKIPS (not entries): hairball (pathological
/// BVH depth — the stack-margin stress, not a place), intel-sponza (2.7 GB
/// of 4K PNGs — the texture-memory wall, not a poly-count call), san-miguel
/// LOW-poly (the full-res scene is the standard one; the low variant was the
/// launch-era load-time hedge), bistro Interior (interiors don't read as
/// islands from the air), rungholt house.obj (subsumed by the city).
pub const CURATED: &[IslandSpec] = &[
    IslandSpec {
        name: "powerplant",
        path: "scenes/powerplant/powerplant.obj",
        theme_hour: 6.5, // industrial dawn
        mtris: 12.8,
    },
    IslandSpec {
        name: "sponza",
        path: "scenes/sponza-khronos/Sponza.gltf",
        theme_hour: 8.5, // morning courtyard
        mtris: 0.262,
    },
    IslandSpec {
        name: "rungholt",
        path: "scenes/rungholt/rungholt.obj",
        theme_hour: 11.0, // late-morning city
        mtris: 6.7,
    },
    IslandSpec {
        name: "helmet",
        path: "scenes/damaged-helmet/DamagedHelmet.glb",
        theme_hour: 13.0, // early-afternoon pedestal piece
        mtris: 0.015,
    },
    IslandSpec {
        name: "san-miguel",
        path: "scenes/san-miguel/san-miguel.obj",
        theme_hour: 15.5, // afternoon patio
        mtris: 10.0,
    },
    IslandSpec {
        name: "bistro",
        path: "scenes/bistro/Exterior/exterior.obj",
        theme_hour: 17.5, // golden-hour street (the emissive-map showcase)
        mtris: 2.8,
    },
    IslandSpec {
        name: "vokselia",
        // FULL night, deliberately past the dusk handoff: at 22:00 the
        // antipodal moon sits ~60 deg up (a real overhead light — at the old
        // 18:45 "civil dusk" hour it grazed the horizon at ~11 deg), the star
        // field is fully in, and the fireflies are at full brightness. This
        // is the darkest island in the ring by construction, and the one
        // place a flagless lap reaches the moonlit/star-lit night tier.
        path: "scenes/vokselia-spawn/vokselia_spawn.obj",
        theme_hour: 22.0, // moonlit night garden (fireflies at full strength)
        mtris: 1.9,
    },
];

/// A placed island — what main.rs needs for camera framing and what the
/// flycam attractors are built from.
pub struct Island {
    pub name: String,
    pub center: Vec3A,
    /// Half the island's content x/z footprint — the attractor falloff scale.
    pub radius: f32,
    /// The island's content HEIGHT (full y extent). Carried separately because
    /// `radius` is an x/z measure and says nothing about how tall a subject is:
    /// framing a camera off the footprint alone puts the Damaged Helmet — as
    /// tall as it is wide — half out of frame while a flat city sits correctly.
    /// `--cinematic` uses it to fit the whole subject.
    pub height: f32,
    pub theme_hour: f32,
}

pub struct World {
    pub islands: Vec<Island>,
    /// Field half-extent for camera framing (the `tile_fh` shape).
    pub field_half: f32,
}

/// A world island's time-of-day attractor (world mode only): flying toward
/// an island eases the GLOBAL tod toward its theme hour at the manual-scrub
/// rate, so every downstream contract is the held-scrub one (accumulation
/// reset per delta, histories kept). An empty list = the feature off — every
/// non-world session passes `vec![]` and is structurally unchanged.
///
/// Lives here rather than in flycam.rs because it is derived from `Island` and
/// has two consumers with different platform reach: the flycam integrator
/// (`#[cfg(windows)]`, eases toward it at `TOD_RATE`) and `--cinematic`
/// (headless, samples it directly per output frame — see cinematic.rs's
/// invariant 3). `world::self_test` pins the circular mean, and used to have to
/// reach into a Windows-only module to do it.
pub struct TodAttractor {
    pub pos: Vec3A,
    /// Theme hour, [0, 24).
    pub hour: f32,
    /// Falloff scale (the island's content half-footprint): weight is
    /// 1 / (d² + radius²), so the blend is finite ON the island and fades
    /// with the square of distance off it.
    pub radius: f32,
}

/// Distance-weighted CIRCULAR mean of the attractor hours at `pos` (x/z
/// distance — altitude must not drag the blend). Circular because hours wrap:
/// 22h and 2h must blend toward midnight, never toward noon; each hour maps
/// to a point on the unit circle, the weighted vector sum is averaged, and
/// atan2 maps back. Returns [0, 24). Pure — `self_test` pins it.
pub fn attractor_hour(attractors: &[TodAttractor], pos: Vec3A) -> f32 {
    use std::f32::consts::TAU;
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    for a in attractors {
        let d = pos - a.pos;
        let d2 = d.x * d.x + d.z * d.z;
        let w = 1.0 / (d2 + a.radius * a.radius).max(1e-6) as f64;
        let th = (a.hour / 24.0 * TAU) as f64;
        sx += w * th.cos();
        sy += w * th.sin();
    }
    if sx == 0.0 && sy == 0.0 {
        return 0.0; // fully antipodal cancellation — pick midnight over NaN
    }
    let h = (sy.atan2(sx) / TAU as f64 * 24.0) as f32;
    h.rem_euclid(24.0)
}

/// Build the attractor set for a loaded world — the ONE construction site,
/// shared by the interactive flycam and `--cinematic`.
pub fn attractors(w: &World) -> Vec<TodAttractor> {
    w.islands
        .iter()
        .map(|i| TodAttractor { pos: i.center, hour: i.theme_hour, radius: i.radius.max(1.0) })
        .collect()
}

/// Islands-per-ring gap factor over the largest content footprint — the
/// tile_scene 1.05 gap, widened so distinct scenes read as separate places.
const PITCH_K: f32 = 1.25;

/// Ground apron beyond the field, mirroring `tile_scene`'s `fh + 6`.
const GROUND_MARGIN: f32 = 6.0;

/// How the merge went for one part — print metadata only.
enum LoadKind {
    Warm, // .fcache hit
    Cold, // parsed + BVH built + sidecar stored for the next boot
    Gltf, // glTF re-parses on world-cache misses (per-scene cache-skip by
          // design; the WORLD sidecar carries its textures inline instead)
}

impl LoadKind {
    fn label(&self) -> &'static str {
        match self {
            LoadKind::Warm => "warm",
            LoadKind::Cold => "cold",
            LoadKind::Gltf => "gltf",
        }
    }
}

/// Equal-angular ring placement for `n` hour-SORTED islands. Pure and
/// deterministic — `self_test` pins determinism and pairwise disjointness.
/// Returns (centers, field_half). Equal spacing (not hour-proportional) is
/// what makes collisions impossible by construction: adjacent chord distance
/// is ≥ ~0.97·pitch at every n (2R·sin(π/n) with R = max(pitch·n/2π, pitch)),
/// while the hour ORDER still sweeps monotonically around the lap.
fn ring_layout(footprints: &[f32]) -> (Vec<Vec3A>, f32) {
    let n = footprints.len();
    let fmax = footprints.iter().fold(1e-3f32, |a, &b| a.max(b));
    let pitch = PITCH_K * fmax;
    if n <= 1 {
        // One island is its own world — center it instead of orbiting an
        // empty origin.
        return (vec![Vec3A::ZERO; n], 0.5 * fmax);
    }
    let r = (pitch * n as f32 / std::f32::consts::TAU).max(pitch);
    let centers = (0..n)
        .map(|i| {
            let th = std::f32::consts::TAU * i as f32 / n as f32;
            Vec3A::new(r * th.cos(), 0.0, r * th.sin())
        })
        .collect();
    (centers, r + 0.5 * pitch)
}

/// Concatenate finished (fitted, grounded) Scenes at world offsets into one
/// flat Scene under one covering ground quad — the pure, testable core.
///
/// Id-offset contract per part: vertex indices rebase by the part's `vbase`
/// (the `tile_scene` expression), `tri_mat` by `mat_base`, and every
/// material texture-index field by `tex_base` EXCEPT the `NO_TEX` sentinel,
/// which must survive verbatim (branching on it is what keeps unmapped
/// materials shading bit-identically — the structural guarantee the whole
/// map system rests on). Each part's own ground material stays in the merged
/// array UNREFERENCED — remapping every tri_mat down by one to drop it buys
/// a few hundred bytes and risks an off-by-one in exchange.
fn merge_scenes(parts: Vec<(Scene, Vec3A)>, field_half: f32) -> Scene {
    let gv = scene::GROUND_VERTS;
    let gt = scene::GROUND_TRIS;
    // reserve_exact before the copy loops — the tile_scene lesson: Vec
    // doubling on multi-GB streams spikes transient memory.
    let total_v: usize = gv + parts.iter().map(|(s, _)| s.positions.len() - gv).sum::<usize>();
    let total_t: usize = gt + parts.iter().map(|(s, _)| s.indices.len() - gt).sum::<usize>();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut texcoords = Vec::new();
    let mut indices = Vec::new();
    let mut tri_mat = Vec::new();
    let mut materials: Vec<Material> = Vec::new();
    let mut textures = Vec::new();
    positions.reserve_exact(total_v);
    normals.reserve_exact(total_v);
    texcoords.reserve_exact(total_v);
    indices.reserve_exact(total_t);
    tri_mat.reserve_exact(total_t);

    // One covering ground quad FIRST (the loaders' convention — content-box
    // derivation and tile_scene both rely on ground = the first GROUND_VERTS
    // verts), same construction as tile_scene's rewrite, with its own fresh
    // ground material at index 0 (the loaders' neutral color).
    // GROUND_DROP: the covering quad must not share the islands' rest plane
    // (each part's own y = 0 seabed/floor faces land there) — see scene.rs.
    let s = field_half + GROUND_MARGIN;
    let (a, b, c, d) = (
        Vec3A::new(-s, -scene::GROUND_DROP, -s),
        Vec3A::new(-s, -scene::GROUND_DROP, s),
        Vec3A::new(s, -scene::GROUND_DROP, s),
        Vec3A::new(s, -scene::GROUND_DROP, -s),
    );
    positions.extend_from_slice(&[a, b, c, a, c, d]);
    normals.extend_from_slice(&[Vec3A::Y; scene::GROUND_VERTS]);
    texcoords.extend_from_slice(&[Vec2::ZERO; scene::GROUND_VERTS]);
    indices.push([0, 1, 2]);
    indices.push([3, 4, 5]);
    materials.push(Material {
        albedo: Vec3A::new(0.42, 0.46, 0.40),
        roughness: 0.55,
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
    });
    tri_mat.extend_from_slice(&[0, 0]);

    // Parts are consumed one at a time and dropped after their copy, so the
    // transient footprint is ~one part + the accumulating world.
    let mut sway_regions: Vec<crate::foliage::SwayRegion> = Vec::new();
    for (mut part, off) in parts {
        let vbase = positions.len() as u32;
        let mat_base = materials.len() as u32;
        let tex_base = textures.len() as u32;
        // Foliage v0.6 sway region: this part's tri range in the merged
        // stream + its content box AT THE RING OFFSET, so `foliage::attach`
        // derives plants and cells at the ISLAND's scale — the world's ~10×
        // content diagonal used to fuse whole Minecraft forests into single
        // rigidly-translating plants. Recorded for EVERY part (a region with
        // no foliage-masked tris produces no cells); ranges are ascending and
        // disjoint by construction (`region_of`'s contract).
        let tri_start = indices.len() as u32;
        let region_box = (part.content_min + off, part.content_max + off);
        positions.extend(part.positions[gv..].iter().map(|&p| p + off));
        normals.extend_from_slice(&part.normals[gv..]);
        texcoords.extend_from_slice(&part.texcoords[gv..]);
        for tri in &part.indices[gt..] {
            indices.push([
                tri[0] - gv as u32 + vbase,
                tri[1] - gv as u32 + vbase,
                tri[2] - gv as u32 + vbase,
            ]);
        }
        tri_mat.extend(part.tri_mat[gt..].iter().map(|&m| m + mat_base));
        let off_tex = |t: u32| if t == NO_TEX { NO_TEX } else { t + tex_base };
        for mut m in part.materials.drain(..) {
            m.normal_tex = off_tex(m.normal_tex);
            m.rough_tex = off_tex(m.rough_tex);
            m.metal_tex = off_tex(m.metal_tex);
            m.emissive_tex = off_tex(m.emissive_tex);
            if let MatKind::Textured { tex } = m.kind {
                m.kind = MatKind::Textured { tex: tex + tex_base };
            }
            materials.push(m);
        }
        textures.append(&mut part.textures);
        sway_regions.push(crate::foliage::SwayRegion {
            tri_start,
            tri_end: indices.len() as u32,
            cmin: region_box.0,
            cmax: region_box.1,
        });
    }

    let mut world = Scene {
        positions,
        normals,
        texcoords,
        indices,
        tri_mat,
        materials,
        textures,
        any_alpha: false,
        any_height: false,
        any_transmissive: false,
        emissive: crate::emissive::EmissiveLights::off(),
        // ONE sun at infinity for the whole archipelago — the one-sky
        // invariant. --tod / the flycam attractors move it globally later.
        sun: scene::default_sun(),
        sky_sh: crate::sh::Sh9::ZERO,
        sky_scale: 1.0,
        night: 0.0,
        sway: None,
        sway_regions,
        diag: 0.0,
        eps: 0.0,
        ao_radius: 0.0,
        detail_scales: Vec::new(),
        content_min: Vec3A::ZERO,
        content_max: Vec3A::ZERO,
    };
    // Re-derives diag/eps/ao_radius (now world-scale — the proven --tile
    // regime), the content box (unions the islands, excludes the ground),
    // the any_* gates (unions — one alpha-masked island arms cutout for the
    // world, a cost accept), and the SH dome.
    scene::finalize_scalars(&mut world);
    world
}

/// Presence probe for one curated entry, factored out of the load so the
/// WORLD KEY and the loader can never disagree about what "present" means:
/// resolve (.zst sibling), existence, and the LFS-pointer magic sniff
/// (unmaterialized pointers are small ASCII files under the binary
/// extensions — they must skip loudly, not panic inside zstd/GLB parsing).
/// Absent/unmaterialized prints ONE loud SKIP line — this runs exactly once
/// per boot per entry (warm or cold), so a fresh checkout's degradation is
/// always visible.
fn probe_part(spec: &IslandSpec) -> Option<String> {
    let resolved = scene::resolve_scene_path(spec.path);
    let p = std::path::Path::new(&resolved);
    if !p.exists() {
        eprintln!(
            "world: SKIP '{}' (~{}M tris) — {} not found (git lfs pull?)",
            spec.name, spec.mtris, spec.path
        );
        return None;
    }
    let lower = resolved.to_ascii_lowercase();
    let want_magic: Option<[u8; 4]> = if lower.ends_with(".zst") {
        Some([0x28, 0xB5, 0x2F, 0xFD])
    } else if lower.ends_with(".glb") {
        Some(*b"glTF")
    } else {
        None
    };
    if let Some(magic) = want_magic {
        use std::io::Read;
        let mut head = [0u8; 4];
        let ok = std::fs::File::open(p)
            .and_then(|mut f| f.read_exact(&mut head))
            .is_ok()
            && head == magic;
        if !ok {
            eprintln!(
                "world: SKIP '{}' — {} is not materialized (git lfs pull?)",
                spec.name, spec.path
            );
            return None;
        }
    }
    Some(resolved)
}

fn is_gltf(resolved: &str) -> bool {
    let lower = resolved.to_ascii_lowercase();
    lower.ends_with(".gltf") || lower.ends_with(".glb")
}

/// The world sidecar's multi-source key over the probe results: per curated
/// entry, identity (path + theme hour), PRESENCE, and — for present entries
/// — the resolved files loading it reads. Textures need no per-part
/// attribution here: the sidecar's own per-texture records carry path+stat
/// for every file-backed texture. Stats are taken by the key SERIALIZER
/// (`scene_cache::world_key_blob`), so store- and load-time blobs compare
/// equal iff every dependency is unchanged.
fn build_world_key(probes: &[(&'static IslandSpec, Option<String>)]) -> crate::scene_cache::WorldKey {
    let entries = probes
        .iter()
        .map(|(spec, resolved)| {
            let deps = match resolved {
                None => Vec::new(),
                Some(res) if is_gltf(res) => {
                    let mut d = vec![res.clone()];
                    for p in crate::gltf_loader::dependency_files(res) {
                        d.push(p.to_string_lossy().into_owned());
                    }
                    d
                }
                Some(res) => vec![
                    res.clone(),
                    crate::scene_cache::mtl_sibling(std::path::Path::new(res))
                        .to_string_lossy()
                        .into_owned(),
                ],
            };
            crate::scene_cache::WorldKeyEntry {
                path: spec.path.to_string(),
                theme_hour_bits: spec.theme_hour.to_bits(),
                present: resolved.is_some(),
                deps,
            }
        })
        .collect();
    crate::scene_cache::WorldKey {
        entries,
        pitch_k_bits: PITCH_K.to_bits(),
        ground_margin_bits: GROUND_MARGIN.to_bits(),
    }
}

/// Load one PRESENT island through the exact positional-arg pipeline:
/// per-scene sidecar cache for OBJ (a cold load stores one so the NEXT
/// world-cache-miss boot is warm — the per-part BVH's only use), glTF parses
/// fresh (per-scene cache-skip by design; the world sidecar carries it).
fn load_part(resolved: &str) -> (Scene, LoadKind) {
    if is_gltf(resolved) {
        return (crate::gltf_loader::load_gltf_scene(resolved), LoadKind::Gltf);
    }
    if let Some((s, _bvh)) = crate::scene_cache::try_load(resolved) {
        // The per-part BVH is discarded — the world BVH always rebuilds over
        // the merged, offset geometry.
        return (s, LoadKind::Warm);
    }
    let mut s = scene::load_obj_scene(resolved);
    scene::derive_heights(&mut s);
    // The positional-arg load order (main.rs), spray included: the sidecar
    // this stores must be byte-interchangeable with a single-scene boot's —
    // v8 world cold boots skipped the retag and wrote WRONG sidecars under
    // the same key (the v9 bump invalidated them).
    scene::reclassify_spray(&mut s);
    // Coincident-cull, same byte-interchangeability contract (main.rs's
    // positional-arg load order).
    scene::cull_coincident(&mut s);
    // Foliage-sway partition, byte-interchangeability again: the per-island
    // sidecar must hold the SAME swept tree a direct single-scene load of
    // this part would store under the identical sway_word key.
    crate::foliage::attach(&mut s);
    let bvh = crate::bvh::Bvh::build(&s);
    crate::scene_cache::store(resolved, &s, &bvh);
    (s, LoadKind::Cold)
}

/// The world sidecar's location — one file for the one world.
const WORLD_CACHE: &str = "scenes/world.fcache";

/// Build the world: probe the curated set, try the WORLD SIDECAR (a hit
/// returns the merged Scene + layout + BVH with zero per-part work), else
/// load + ring-layout + merge + build the world BVH and store the sidecar
/// for the next boot. `None` ⇔ zero islands present (the caller falls back
/// to `procedural_scene` with a loud line). Prints one line per island and
/// one summary line on the rebuild path — the boot's progress signal.
pub fn world_scene() -> Option<(Scene, World, crate::bvh::Bvh)> {
    let t0 = std::time::Instant::now();
    let probes: Vec<(&'static IslandSpec, Option<String>)> =
        CURATED.iter().map(|spec| (spec, probe_part(spec))).collect();
    if probes.iter().all(|(_, r)| r.is_none()) {
        return None;
    }
    let key = build_world_key(&probes);
    let cache_path = std::path::Path::new(WORLD_CACHE);
    // The warm path: one sidecar read collapses every per-island load, the
    // merge, and the world BVH build. Indeterminate — the ~4.6 s is one blob.
    crate::progress::phase(crate::progress::Phase::Cache, "world sidecar", 0);
    if let Some((sc, bvh, world)) = crate::scene_cache::try_load_world(cache_path, &key) {
        return Some((sc, world, bvh));
    }
    eprintln!("world: rebuilding — {} miss/stale", cache_path.display());
    // Per-island loads run CONCURRENTLY. `load_part` is a pure producer of an
    // OWNED Scene: both loaders return one, material/texture ids are
    // per-SceneBuilder counters (never global), no loader draws rng, and every
    // loader lever (texture::set_mips/set_aniso/set_h2n/set_n2h,
    // scene::set_spray/set_water/…, bvh::set_height_armed) was stored ONCE in
    // main's apply block BEFORE any load and is only read from here on.
    //
    // The fan-out rides the GLOBAL rayon pool deliberately. A pool of its own
    // — the obvious way to bound concurrency — would be INHERITED by every
    // inner par_iter (texture decode, the n2h solve, Bvh::build) through
    // `install`, narrowing the passes that already do the real parallel work;
    // that costs far more than the fan-out wins. The one place memory would
    // genuinely multiply is the n2h solve, and that is bounded structurally
    // instead, by scene.rs's shared `n2h_pool`.
    //
    // Determinism: `probes.par_iter()` is an INDEXED iterator, so map+collect
    // writes slot i from item i — positional order is a type-level property
    // here, not a reliance on collect's ordering lore (which is why this is
    // Vec<Option<_>> + flatten rather than filter_map). Ring order then comes
    // from the theme-hour sort below, which self_test pins as a STRICT order,
    // so it can never fall through to whatever order the loads finished in.
    // `with_max_len(1)` is the per-item idiom the decode sites already use:
    // one task per island, so the 12.8M-tri part never queues behind another.
    //
    // Cost: the per-part loud lines (obj materials, heightfield, spray, cache
    // hit/write) now INTERLEAVE on stderr. Nothing parses them, and the
    // readable per-island summary is the ring-ordered block below.
    let n_present = probes.iter().filter(|(_, r)| r.is_some()).count() as u32;
    // One STAGE row over the whole fan-out — a COMPLETION count, not "now
    // starting i", which has no referent when N loads are in flight. The inner
    // phase row goes indeterminate for the duration (see progress::MultiGuard,
    // whose drop is what restores sequential publishing — including when a
    // worker's panic propagates out of the `collect()` below).
    crate::progress::stage(0, n_present, "");
    crate::progress::phase(
        crate::progress::Phase::Parse,
        &format!("{n_present} islands at once"),
        0,
    );
    let loaded: Vec<Option<(&IslandSpec, Scene, LoadKind)>> = {
        use rayon::prelude::*;
        let _multi = crate::progress::multi_guard();
        probes
            .par_iter()
            .with_max_len(1)
            .map(|(spec, resolved)| {
                let res = resolved.as_ref()?;
                let (s, kind) = load_part(res);
                crate::progress::stage_done(spec.name);
                Some((*spec, s, kind))
            })
            .collect()
    };
    let mut parts: Vec<(&IslandSpec, Scene, LoadKind)> = loaded.into_iter().flatten().collect();
    // Hour order = ring order (the table is already sorted; the sort makes
    // that a property, not a convention).
    parts.sort_by(|a, b| a.0.theme_hour.total_cmp(&b.0.theme_hour));
    let heights: Vec<f32> = parts
        .iter()
        .map(|(_, s, _)| (s.content_max.y - s.content_min.y).max(1e-3))
        .collect();
    let footprints: Vec<f32> = parts
        .iter()
        .map(|(_, s, _)| {
            let e = s.content_max - s.content_min;
            e.x.max(e.z).max(1e-3)
        })
        .collect();
    let (centers, field_half) = ring_layout(&footprints);
    let mut islands = Vec::with_capacity(parts.len());
    let mut merge_in = Vec::with_capacity(parts.len());
    let mut total_tris = 0usize;
    for (i, (spec, s, kind)) in parts.into_iter().enumerate() {
        let isle = Island {
            name: spec.name.to_string(),
            center: centers[i],
            radius: 0.5 * footprints[i],
            height: heights[i],
            theme_hour: spec.theme_hour,
        };
        eprintln!(
            "world: island '{}' {:>5.1}M tris ({}) at ({:+6.1},{:+6.1}) r {:.1} h {:.1} theme {:02}:{:02}",
            isle.name,
            s.tri_count() as f64 / 1e6,
            kind.label(),
            isle.center.x,
            isle.center.z,
            isle.radius,
            isle.height,
            isle.theme_hour as u32,
            ((isle.theme_hour.fract()) * 60.0) as u32,
        );
        total_tris += s.tri_count();
        islands.push(isle);
        merge_in.push((s, centers[i]));
    }
    crate::progress::phase(crate::progress::Phase::Merge, "", 0);
    let mut world = merge_scenes(merge_in, field_half);
    eprintln!(
        "world: {} islands | {:.1}M tris | field {:.0}x{:.0} | diag {:.1} | loaded+merged in {:.1} s",
        islands.len(),
        total_tris as f64 / 1e6,
        field_half * 2.0,
        field_half * 2.0,
        world.diag,
        t0.elapsed().as_secs_f64(),
    );
    // Foliage-sway partition over the MERGED scene (world-scale content box)
    // — before the world BVH build below, whose leaf boxes sweep by it.
    // main.rs's post-match attach recomputes the identical partition (pure
    // function of the merged geometry + class bytes).
    crate::foliage::attach(&mut world);
    // The world BVH builds HERE (not main.rs's shared arm) so the sidecar
    // can store it — the single-scene cold path's `prebuilt` precedent.
    // Indeterminate (~8.8 s of parallel build — no per-node signal).
    crate::progress::phase(crate::progress::Phase::Bvh, "", 0);
    let tb = std::time::Instant::now();
    let bvh = crate::bvh::Bvh::build(&world);
    eprintln!(
        "world: BVH {} nodes built in {:.0} ms",
        bvh.nodes.len(),
        tb.elapsed().as_secs_f64() * 1000.0
    );
    crate::progress::phase(crate::progress::Phase::Sidecar, "world sidecar", 0);
    let meta = World { islands, field_half };
    crate::scene_cache::store_world(cache_path, &key, &world, &bvh, &meta);
    Some((world, meta, bvh))
}

// ---------------------------------------------------------------------------
// Self-test (pure, DLL-free, download-free — run by --check).

/// A finished mini scene in the loaders' shape: standard ground quad first,
/// then a box, with `n_tex` tiny textures and one material exercising the
/// map-index fields (mixing real ids and NO_TEX).
fn test_part(n_tex: u32, albedo_tex: u32, normal_tex: u32) -> Scene {
    use crate::texture::Texture;
    let mut b = scene::SceneBuilder::new();
    let ground = b.material(Vec3A::new(0.42, 0.46, 0.40), 0.55, 0.0);
    let s = 60.0;
    b.quad(
        Vec3A::new(-s, 0.0, -s),
        Vec3A::new(-s, 0.0, s),
        Vec3A::new(s, 0.0, s),
        Vec3A::new(s, 0.0, -s),
        ground,
    );
    for i in 0..n_tex {
        b.add_texture(Texture {
            w: 1,
            h: 1,
            texels: vec![[i as u8, 64, 32, 255]],
            alpha_masked: false,
            srgb: true,
            source: String::new(),
            h2n: false,
            n2h: false,
            normal_role: false,
            mips: Vec::new(),
        });
    }
    let m = b.material_full(Material {
        albedo: Vec3A::new(0.7, 0.7, 0.7),
        roughness: 0.5,
        metallic: 0.0,
        anisotropy: 0.0,
        sheen: 0.0,
        translucency: 0.0,
        transmission: 0.0,
        trans_tint: Vec3A::splat(-1.0),
        ior: 1.5,
        ripple_amp: 0.0,
        emissive: Vec3A::ZERO,
        normal_tex,
        normal_scale: 1.0,
        height_amp: 0.0,
        rough_tex: NO_TEX, // must survive the merge verbatim
        metal_tex: NO_TEX,
        emissive_tex: NO_TEX,
        class: crate::matclass::IDX_DEFAULT as u8,
        kind: if albedo_tex == NO_TEX {
            MatKind::Diffuse
        } else {
            MatKind::Textured { tex: albedo_tex }
        },
    });
    b.add_box(Vec3A::new(0.0, 0.5, 0.0), Vec3A::splat(0.5), m);
    b.finish(scene::default_sun())
}

/// The two table properties the PARALLEL per-island load rests on.
///
/// (1) STRICTLY INCREASING theme hours make `world_scene`'s ring sort a TOTAL
/// order, so a stable sort can never fall through to whatever order the
/// concurrent collect produced. (2) NO TWO ENTRIES MAY NAME ONE SIDECAR:
/// concurrent loads mean concurrent `scene_cache::store` calls, and that tmp
/// name is pid-suffixed — which separates PROCESSES, not two entries inside
/// one process resolving to the same file.
///
/// (2) is checked on the RESOLVED form, which is where the collision would
/// land (`probe_part` resolves, and `store`'s tmp name derives from the
/// resolved path). Comparing the raw `path` strings would MISS the one
/// aliasing this table can express: `scene::resolve_scene_path` maps `p` to
/// `p.zst` when `p` is absent AND NOTHING ELSE, so two entries resolve to one
/// file iff their paths are equal after stripping a single trailing `.zst` —
/// an `x.obj` entry alongside an `x.obj.zst` entry are distinct strings that
/// name the same sidecar. Stripping keeps this a pure function of the table;
/// calling the real resolver would make the gate depend on which files happen
/// to be materialized.
///
/// Takes a SLICE rather than reading `CURATED` directly so `self_test` can
/// prove both checks FIRE. A gate that only ever sees the real (correct)
/// table passes whether or not it means anything, and the `.zst` case is one
/// no real table exercises.
fn check_table(t: &[IslandSpec]) -> Result<(), String> {
    for i in 1..t.len() {
        if !(t[i].theme_hour > t[i - 1].theme_hour) {
            return Err(format!(
                "theme hours must strictly increase: '{}' {} follows '{}' {}",
                t[i].name,
                t[i].theme_hour,
                t[i - 1].name,
                t[i - 1].theme_hour
            ));
        }
    }
    let unzst = |p: &str| p.strip_suffix(".zst").unwrap_or(p).to_string();
    for i in 0..t.len() {
        for j in (i + 1)..t.len() {
            if unzst(t[i].path) == unzst(t[j].path) {
                return Err(format!(
                    "path aliased at {i}/{j}: '{}' and '{}' resolve to one sidecar",
                    t[i].path, t[j].path
                ));
            }
        }
    }
    Ok(())
}

pub fn self_test() -> Result<(), String> {
    let gv = scene::GROUND_VERTS;
    let gt = scene::GROUND_TRIS;
    let err = |m: String| Err(m);

    // --- CURATED: the two properties the PARALLEL per-island load rests on,
    // plus the TEETH proving each check fires (see `check_table`).
    check_table(CURATED)?;
    let spec = |path: &'static str, hour: f32| IslandSpec { name: "t", path, theme_hour: hour, mtris: 0.0 };
    let must_fail = |what: &str, t: &[IslandSpec]| match check_table(t) {
        Err(_) => Ok(()),
        Ok(()) => Err(format!("CURATED table gate did not fire on {what}")),
    };
    must_fail("equal hours", &[spec("a.obj", 1.0), spec("b.obj", 1.0)])?;
    must_fail("decreasing hours", &[spec("a.obj", 2.0), spec("b.obj", 1.0)])?;
    must_fail("an identical path", &[spec("a.obj", 1.0), spec("a.obj", 2.0)])?;
    // THE case no real table can exercise, and the whole reason the check
    // strips before comparing: two DISTINCT strings naming one sidecar.
    must_fail(".zst sibling aliasing", &[spec("a.obj", 1.0), spec("a.obj.zst", 2.0)])?;
    must_fail(".zst sibling aliasing (reversed)", &[spec("a.obj.zst", 1.0), spec("a.obj", 2.0)])?;
    // ...and the near-misses it must NOT fire on, or the strip is too eager.
    check_table(&[spec("a.obj", 1.0), spec("ab.obj", 2.0)])?;
    check_table(&[spec("a.obj.zst", 1.0), spec("b.obj.zst", 2.0)])?;
    check_table(&[spec("a.zst", 1.0), spec("a.obj", 2.0)])?;

    // --- ring_layout: determinism, disjointness, single-island centering.
    for n in 1..=8usize {
        let fp: Vec<f32> = (0..n).map(|i| 8.0 + i as f32).collect();
        let (c1, fh1) = ring_layout(&fp);
        let (c2, fh2) = ring_layout(&fp);
        if c1 != c2 || fh1 != fh2 {
            return err(format!("ring_layout nondeterministic at n={n}"));
        }
        let fmax = fp.iter().fold(0.0f32, |a, &b| a.max(b));
        for i in 0..n {
            if c1[i].y != 0.0 {
                return err(format!("ring_layout center off the plane at n={n}"));
            }
            for j in (i + 1)..n {
                let d = (c1[i] - c1[j]).length();
                if d < fmax * 0.95 {
                    return err(format!(
                        "ring_layout islands too close at n={n}: |c{i}-c{j}| = {d} < {fmax}"
                    ));
                }
            }
        }
        if n == 1 && c1[0] != Vec3A::ZERO {
            return err("ring_layout n=1 should center at origin".into());
        }
    }

    // --- merge: accounting, offsets, id remaps, NO_TEX preservation.
    // Part A: 1 texture, textured albedo + normal map on tex 0.
    // Part B: 2 textures, textured albedo on tex 1, normal map on tex 0.
    let build = || {
        vec![
            (test_part(1, 0, 0), Vec3A::new(10.0, 0.0, 0.0)),
            (test_part(2, 1, 0), Vec3A::new(-10.0, 0.0, 0.0)),
        ]
    };
    let parts = build();
    let (av, at) = (parts[0].0.positions.len(), parts[0].0.indices.len());
    let (bv, bt) = (parts[1].0.positions.len(), parts[1].0.indices.len());
    let (a_mats, b_mats) = (parts[0].0.materials.len(), parts[1].0.materials.len());
    let a_texs = parts[0].0.textures.len();
    // Reference copies for the bit-exact position check (merge consumes).
    let ref_a: Vec<Vec3A> = parts[0].0.positions[gv..].to_vec();
    let ref_b: Vec<Vec3A> = parts[1].0.positions[gv..].to_vec();
    // Part content boxes for the sway-region pin (merge consumes the parts).
    let a_cbox = (parts[0].0.content_min, parts[0].0.content_max);
    let b_cbox = (parts[1].0.content_min, parts[1].0.content_max);
    let m = merge_scenes(parts, 30.0);

    if m.positions.len() != gv + (av - gv) + (bv - gv) {
        return err(format!("merged vert count {} wrong", m.positions.len()));
    }
    if m.indices.len() != gt + (at - gt) + (bt - gt) {
        return err(format!("merged tri count {} wrong", m.indices.len()));
    }
    if m.indices[0] != [0, 1, 2] || m.indices[1] != [3, 4, 5] || m.tri_mat[..gt] != [0, 0] {
        return err("merged ground quad not first / not material 0".into());
    }
    // Positions bit-exact at source + offset.
    for (i, &p) in ref_a.iter().enumerate() {
        if m.positions[gv + i] != p + Vec3A::new(10.0, 0.0, 0.0) {
            return err(format!("part A vert {i} not source+offset"));
        }
    }
    for (i, &p) in ref_b.iter().enumerate() {
        if m.positions[gv + (av - gv) + i] != p + Vec3A::new(-10.0, 0.0, 0.0) {
            return err(format!("part B vert {i} not source+offset"));
        }
    }
    // Every rebased index must land inside its part's vertex range.
    for (t, tri) in m.indices[gt..].iter().enumerate() {
        let (lo, hi) = if t < at - gt {
            (gv, gv + (av - gv))
        } else {
            (gv + (av - gv), m.positions.len())
        };
        for &ix in tri {
            if (ix as usize) < lo || ix as usize >= hi {
                return err(format!("tri {t} index {ix} outside its part range"));
            }
        }
    }
    // Material accounting: fresh ground (1) + A's + B's.
    if m.materials.len() != 1 + a_mats + b_mats {
        return err(format!("merged material count {} wrong", m.materials.len()));
    }
    // tri_mat offsets: A's box tris reference A's box material at 1 + its
    // local id; B's at 1 + a_mats + local id.
    let a_box_mat = m.tri_mat[gt + (at - gt) - 1];
    let b_box_mat = *m.tri_mat.last().unwrap();
    if a_box_mat as usize != 1 + 1 || b_box_mat as usize != 1 + a_mats + 1 {
        return err(format!("tri_mat offsets wrong: A box {a_box_mat}, B box {b_box_mat}"));
    }
    // Texture id remaps. A's box material: albedo tex 0 -> 0, normal 0 -> 0.
    let ma = &m.materials[a_box_mat as usize];
    if ma.kind != (MatKind::Textured { tex: 0 }) || ma.normal_tex != 0 {
        return err("part A texture ids moved (tex_base 0 must be identity)".into());
    }
    // B's box material: albedo tex 1 -> 1 + a_texs, normal 0 -> a_texs, and
    // the NO_TEX fields must survive VERBATIM.
    let mb = &m.materials[b_box_mat as usize];
    if mb.kind != (MatKind::Textured { tex: 1 + a_texs as u32 }) {
        return err("part B albedo texture id not rebased".into());
    }
    if mb.normal_tex != a_texs as u32 {
        return err("part B normal texture id not rebased".into());
    }
    if mb.rough_tex != NO_TEX || mb.metal_tex != NO_TEX || mb.emissive_tex != NO_TEX {
        return err("NO_TEX sentinel not preserved through the merge".into());
    }
    if m.textures.len() != a_texs + 2 {
        return err(format!("merged texture count {} wrong", m.textures.len()));
    }
    // Content box excludes the new ground: the boxes sit at x = ±10, so the
    // content x-extent is ~[-10.5, 10.5], nowhere near the ±36 ground.
    if m.content_min.x < -11.0 || m.content_max.x > 11.0 {
        return err(format!(
            "content box polluted by ground: [{}, {}]",
            m.content_min.x, m.content_max.x
        ));
    }
    // Sway regions (foliage v0.6): one per part, ranges exactly partitioning
    // [ground, end) in part order, boxes = the part's content box AT ITS
    // OFFSET bitwise — the data `foliage::attach` derives island-scale
    // plants/cells from, so a drifted range or box silently re-fuses the
    // world's forests.
    {
        let r = &m.sway_regions;
        if r.len() != 2 {
            return err(format!("merge: {} sway regions, wanted 2", r.len()));
        }
        if r[0].tri_start as usize != gt
            || r[0].tri_end != r[1].tri_start
            || r[1].tri_end as usize != m.indices.len()
        {
            return err("merge: sway-region ranges do not partition the tri stream".into());
        }
        if r[0].tri_end as usize - gt != at - gt {
            return err("merge: region A range != part A tri count".into());
        }
        let off_a = Vec3A::new(10.0, 0.0, 0.0);
        let off_b = Vec3A::new(-10.0, 0.0, 0.0);
        if r[0].cmin != a_cbox.0 + off_a
            || r[0].cmax != a_cbox.1 + off_a
            || r[1].cmin != b_cbox.0 + off_b
            || r[1].cmax != b_cbox.1 + off_b
        {
            return err("merge: sway-region boxes != part content boxes at offset".into());
        }
    }
    // Merge determinism: fresh inputs, byte-equal outputs.
    let m2 = merge_scenes(build(), 30.0);
    if m.positions != m2.positions
        || m.indices != m2.indices
        || m.tri_mat != m2.tri_mat
        || m.diag.to_bits() != m2.diag.to_bits()
        || m.sway_regions != m2.sway_regions
    {
        return err("merge_scenes nondeterministic".into());
    }

    // --- attractor blend (flycam auto-TOD + --cinematic): the CIRCULAR mean
    // pin. 22h and 2h at equal distance must blend to MIDNIGHT — a linear mean
    // says noon, which is exactly the bug the sin/cos vector average prevents.
    let wrap = |h: f32| {
        let d = h.rem_euclid(24.0);
        d.min(24.0 - d) // circular distance to hour 0
    };
    let pair = [
        TodAttractor { pos: Vec3A::new(10.0, 0.0, 0.0), hour: 22.0, radius: 4.0 },
        TodAttractor { pos: Vec3A::new(-10.0, 0.0, 0.0), hour: 2.0, radius: 4.0 },
    ];
    let mid = attractor_hour(&pair, Vec3A::new(0.0, 5.0, 0.0)); // altitude must not matter
    if wrap(mid) > 1e-3 {
        return err(format!("22h+2h equidistant blend {mid} is not midnight"));
    }
    // Standing ON an island, its hour dominates a far one.
    let near = attractor_hour(&pair, Vec3A::new(10.0, 0.0, 0.0));
    let dn = (near - 22.0).rem_euclid(24.0);
    if dn.min(24.0 - dn) > 0.5 {
        return err(format!("on-island blend {near} not near its theme hour 22"));
    }
    // Determinism.
    if attractor_hour(&pair, Vec3A::new(3.0, 0.0, 4.0)).to_bits()
        != attractor_hour(&pair, Vec3A::new(3.0, 0.0, 4.0)).to_bits()
    {
        return err("attractor_hour nondeterministic".into());
    }

    // --- world sidecar: round-trip + staleness/corruption gates. Temp
    // files, pid-suffixed, removed on every exit path; DLL-free (the image
    // crate is an ordinary static dep).
    {
        use crate::scene_cache::{self, WorldKey, WorldKeyEntry};
        let dir = std::env::temp_dir().join(format!("frustracer-worldtest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| format!("worldtest temp dir: {e}"))?;
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(dir.clone());

        // A real decodable image for the PATH flavor (2×2, opaque so the
        // re-decode's alpha derivation matches) and a dependency file whose
        // stat the key covers.
        let png_path = dir.join("worldtest.png");
        image::RgbaImage::from_fn(2, 2, |x, y| {
            image::Rgba([40 + 100 * x as u8, 40 + 100 * y as u8, 200, 255])
        })
        .save(&png_path)
        .map_err(|e| format!("worldtest png write: {e}"))?;
        let dep_path = dir.join("worldtest.dep");
        std::fs::write(&dep_path, b"dep-v1").map_err(|e| format!("worldtest dep write: {e}"))?;

        // Two parts; part B's textures exercise BOTH flavors: tex 0 INLINE
        // (2×2 "gltf:" synthetic source — mips must round-trip via rebuild),
        // tex 1 PATH (the real PNG, re-decoded at load). Part A's 1×1
        // empty-source texture rides inline too.
        let pa = test_part(1, 0, 0);
        let mut pb = test_part(2, 1, 0);
        pb.textures[0] = crate::texture::Texture::from_cached(
            2,
            2,
            vec![[10, 20, 30, 255], [40, 50, 60, 255], [70, 80, 90, 255], [100, 110, 120, 255]],
            false,
            true,
            "gltf:image:0:srgb".to_string(),
            false,
            false,
        );
        pb.textures[1] = {
            let mut t = crate::texture::Texture::from_image(
                image::open(&png_path).map_err(|e| format!("worldtest png re-open: {e}"))?,
                true,
            );
            t.source = png_path.to_string_lossy().into_owned();
            t
        };
        let world_sc = merge_scenes(
            vec![(pa, Vec3A::new(8.0, 0.0, 0.0)), (pb, Vec3A::new(-8.0, 0.0, 0.0))],
            24.0,
        );
        let bvh = crate::bvh::Bvh::build(&world_sc);
        let meta = World {
            islands: vec![
                Island {
                    name: "alpha".into(),
                    center: Vec3A::new(8.0, 0.0, 0.0),
                    radius: 3.0,
                    height: 2.0,
                    theme_hour: 9.0,
                },
                Island {
                    name: "beta".into(),
                    center: Vec3A::new(-8.0, 0.0, 0.0),
                    radius: 4.0,
                    height: 11.0, // taller than wide — the round-trip must carry it
                    theme_hour: 22.0,
                },
            ],
            field_half: 24.0,
        };
        let dep_str = dep_path.to_string_lossy().into_owned();
        // Fresh key per use — stats are taken at SERIALIZATION time, which is
        // exactly what the dep-mutation case below relies on.
        let base_key = || WorldKey {
            entries: vec![
                WorldKeyEntry {
                    path: "scenes/alpha/alpha.obj".into(),
                    theme_hour_bits: 9.0f32.to_bits(),
                    present: true,
                    deps: vec![dep_str.clone()],
                },
                WorldKeyEntry {
                    path: "scenes/beta/beta.gltf".into(),
                    theme_hour_bits: 22.0f32.to_bits(),
                    present: false,
                    deps: Vec::new(),
                },
            ],
            pitch_k_bits: PITCH_K.to_bits(),
            ground_margin_bits: GROUND_MARGIN.to_bits(),
        };
        let key = base_key();
        let cache = dir.join("world.fcache");
        scene_cache::store_world(&cache, &key, &world_sc, &bvh, &meta);
        if !cache.exists() {
            return err("world cache: store did not write".into());
        }

        // Round-trip hit: everything faithful.
        let Some((sc2, bvh2, meta2)) = scene_cache::try_load_world(&cache, &key) else {
            return err("world cache: round-trip missed".into());
        };
        if sc2.positions != world_sc.positions
            || sc2.normals != world_sc.normals
            || sc2.texcoords != world_sc.texcoords
            || sc2.indices != world_sc.indices
            || sc2.tri_mat != world_sc.tri_mat
        {
            return err("world cache: round-trip geometry not byte-equal".into());
        }
        if sc2.sun.dir != world_sc.sun.dir {
            return err("world cache: sun direction drift".into());
        }
        // Foliage v0.6: the regions must round-trip byte-faithfully — the
        // warm-loaded scene's attach derives the same island-scale partition
        // the stored swept BVH was built from.
        if sc2.sway_regions != world_sc.sway_regions || sc2.sway_regions.len() != 2 {
            return err("world cache: sway regions drift".into());
        }
        if sc2.materials.len() != world_sc.materials.len() {
            return err("world cache: material count drift".into());
        }
        for (i, (a, b)) in world_sc.materials.iter().zip(&sc2.materials).enumerate() {
            if a.albedo != b.albedo
                || a.roughness != b.roughness
                || a.metallic != b.metallic
                || a.transmission != b.transmission
                || a.height_amp != b.height_amp
                || a.kind != b.kind
                || a.normal_tex != b.normal_tex
                || a.rough_tex != b.rough_tex
                || a.metal_tex != b.metal_tex
                || a.emissive_tex != b.emissive_tex
            {
                return err(format!("world cache: material {i} drift"));
            }
        }
        if sc2.textures.len() != world_sc.textures.len() {
            return err("world cache: texture count drift".into());
        }
        for (i, (a, b)) in world_sc.textures.iter().zip(&sc2.textures).enumerate() {
            if a.w != b.w
                || a.h != b.h
                || a.texels != b.texels
                || a.alpha_masked != b.alpha_masked
                || a.srgb != b.srgb
                || a.h2n != b.h2n
                || a.n2h != b.n2h
                || a.source != b.source
            {
                return err(format!("world cache: texture {i} drift"));
            }
            // Mips are rebuilt, never stored — they must still come back
            // identical (deterministic box filter over identical texels).
            if a.mips.len() != b.mips.len() {
                return err(format!("world cache: texture {i} mip count drift"));
            }
            for (ma, mb) in a.mips.iter().zip(&b.mips) {
                if ma.w != mb.w || ma.h != mb.h || ma.texels != mb.texels {
                    return err(format!("world cache: texture {i} mip texel drift"));
                }
            }
        }
        if !bvh2.identical(&bvh) {
            return err("world cache: BVH not identical after round-trip".into());
        }
        if meta2.islands.len() != meta.islands.len()
            || meta2.field_half.to_bits() != meta.field_half.to_bits()
        {
            return err("world cache: layout meta drift".into());
        }
        for (a, b) in meta.islands.iter().zip(&meta2.islands) {
            if a.name != b.name
                || a.center != b.center
                || a.radius.to_bits() != b.radius.to_bits()
                || a.height.to_bits() != b.height.to_bits()
                || a.theme_hour.to_bits() != b.theme_hour.to_bits()
            {
                return err("world cache: island meta drift".into());
            }
        }

        // Stale-key misses: presence flip, hour change, dropped entry,
        // layout-constant change — each a fresh key with one mutation.
        let mut k = base_key();
        k.entries[1].present = true;
        if scene_cache::try_load_world(&cache, &k).is_some() {
            return err("world cache: served despite a presence flip".into());
        }
        let mut k = base_key();
        k.entries[0].theme_hour_bits ^= 1;
        if scene_cache::try_load_world(&cache, &k).is_some() {
            return err("world cache: served despite a theme-hour change".into());
        }
        let mut k = base_key();
        k.entries.pop();
        if scene_cache::try_load_world(&cache, &k).is_some() {
            return err("world cache: served despite a dropped entry".into());
        }
        let mut k = base_key();
        k.pitch_k_bits ^= 1;
        if scene_cache::try_load_world(&cache, &k).is_some() {
            return err("world cache: served despite a layout-constant change".into());
        }

        // Corruption: bad magic, wrong WORLD_VERSION (offset 8), a zeroed
        // count (fixed offset 32 — after magic 8 + version 8 + build_key 8 +
        // lever 4 + sway 4), truncations — every one a silent miss.
        let bytes = std::fs::read(&cache).map_err(|e| format!("worldtest read-back: {e}"))?;
        let patched = dir.join("world-patched.fcache");
        let mut b2 = bytes.clone();
        b2[0] ^= 0xFF;
        std::fs::write(&patched, &b2).map_err(|e| e.to_string())?;
        if scene_cache::try_load_world(&patched, &key).is_some() {
            return err("world cache: served despite bad magic".into());
        }
        let mut b2 = bytes.clone();
        b2[8] ^= 0xFF;
        std::fs::write(&patched, &b2).map_err(|e| e.to_string())?;
        if scene_cache::try_load_world(&patched, &key).is_some() {
            return err("world cache: served despite a WORLD_VERSION change".into());
        }
        let mut b2 = bytes.clone();
        b2[32..40].fill(0); // n_verts = 0 — the zero-count guard
        std::fs::write(&patched, &b2).map_err(|e| e.to_string())?;
        if scene_cache::try_load_world(&patched, &key).is_some() {
            return err("world cache: served despite a zeroed count".into());
        }
        for cut in [bytes.len() / 8, bytes.len() / 2, bytes.len() - 9] {
            std::fs::write(&patched, &bytes[..cut]).map_err(|e| e.to_string())?;
            if scene_cache::try_load_world(&patched, &key).is_some() {
                return err(format!("world cache: served a truncation at {cut}"));
            }
        }

        // Dep-file mutation LAST (it can't be un-mutated — mtime moves): the
        // key's serialized stats drift, so the same key OBJECT now misses.
        std::fs::write(&dep_path, b"dep-v2-and-longer").map_err(|e| e.to_string())?;
        if scene_cache::try_load_world(&cache, &key).is_some() {
            return err("world cache: served despite a dependency stat change".into());
        }
    }
    Ok(())
}
