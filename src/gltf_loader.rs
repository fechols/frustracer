//! glTF 2.0 scene loader (`.gltf`/`.glb` on the positional arg). Mirrors
//! `scene::load_obj_scene`'s shape — ground quad + default light + fit to
//! diag 10 — but materials carry REAL PBR data, so `matclass` is bypassed
//! entirely: baseColor/metallic/roughness factors and the
//! normal/metallicRoughness/emissive textures map straight onto the extended
//! `Material` (factor × sample is the glTF spec and shade.rs's semantic).
//!
//! Deliberate conventions (the three glTF-vs-OBJ traps):
//! - NO V flip: glTF's UV origin is the decoded image's row 0 (top-left),
//!   the same convention our texel storage lands in AFTER the OBJ path's
//!   flip — so glTF UVs pass through untouched.
//! - NO glass albedo lift: KHR_materials_transmission baseColor is authored
//!   for tinting (unlike San Miguel's dark exporter Kd).
//! - OPAQUE materials ignore baseColor alpha per spec: their textures'
//!   `alpha_masked` is cleared post-decode (JPG-conversion junk alpha would
//!   otherwise punch cutout holes in walls).
//!
//! Ignored (warned where it matters): KHR_lights_punctual (the engine's ONE
//! sky — a scattering dome plus a sun disc at infinity — IS the lighting
//! model; see src/sky.rs), KHR_texture_transform, authored
//! TANGENTs (one tangent source: shade.rs's on-the-fly derivation),
//! COLOR_0, occlusionTexture (the engine computes its own AO), TEXCOORD_1+,
//! non-TRIANGLES primitive modes. KHR_materials_ior is logged (GLASS_IOR is
//! fixed 1.5); AlphaMode::Blend approximates as Mask; per-material
//! alphaCutoff is ignored — the engine's cutout threshold is a fixed
//! 128/255 ≈ the spec default 0.5 (bvh.rs::moller_trumbore takes no
//! per-material threshold), so non-default cutoffs are counted in the
//! summary line instead.

use crate::scene::{MatKind, Material, Scene, SceneBuilder, NO_TEX};
use crate::texture::Texture;
use glam::{Mat3A, Mat4, Vec2, Vec3A};
use std::collections::HashMap;

/// One world-space triangle primitive, post node-flattening, pre auto-fit.
struct Prim {
    positions: Vec<Vec3A>,
    normals: Vec<Vec3A>, // empty = add_mesh's position-welded fallback
    texcoords: Vec<Vec2>,
    indices: Vec<[u32; 3]>, // winding already flipped for mirrored nodes
    material: Option<usize>,
}

/// Texture roles an image is referenced through — decides srgb and whether
/// its alpha may arm the cutout pipeline.
#[derive(Default, Clone, Copy)]
struct ImgRole {
    base_color: bool,
    base_color_cutout: bool, // baseColor of a MASK/BLEND material
    emissive: bool,
    linear: bool, // normal or metallicRoughness
}

pub fn load_gltf_scene(path: &str) -> Scene {
    crate::progress::phase(crate::progress::Phase::Parse, path, 0);
    let gltf = gltf::Gltf::open(path)
        .unwrap_or_else(|e| panic!("failed to open glTF '{path}': {e}"));
    let base_dir = std::path::Path::new(path)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    let doc = gltf.document;
    let buffers = resolve_buffers(&doc, &base_dir, gltf.blob)
        .unwrap_or_else(|e| panic!("failed to resolve glTF buffers for '{path}': {e}"));
    build_scene(&doc, &buffers, &base_dir, true)
}

/// Every external FILE a glTF scene reads: the buffers' and images' resolved
/// paths, mirroring `resolve_buffers`' `.zst` and the image decode's `.webp`
/// sibling fallbacks — the world sidecar's staleness key for glTF islands
/// (the source file's own stat covers the JSON/GLB itself; a reference
/// change therefore misses via the source, a content change via the dep).
/// A buffer path that resolution would pick is recorded even if MISSING —
/// its (0,0) stat key then catches the file appearing later. Deterministic
/// order (buffers, then images, document order) — the key is blob-compared.
/// GLB embeds everything, so a self-contained one yields an empty list.
/// Parse failure is a loud warning + empty Vec: the actual load will fail on
/// its own terms (the loader convention), and an empty dep list still keys
/// on the source stat.
pub fn dependency_files(path: &str) -> Vec<std::path::PathBuf> {
    let gltf = match gltf::Gltf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("warning: glTF '{path}' failed to parse for dependency scan ({e})");
            return Vec::new();
        }
    };
    let base_dir = std::path::Path::new(path)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    let mut out = Vec::new();
    for b in gltf.document.buffers() {
        if let gltf::buffer::Source::Uri(uri) = b.source() {
            if !uri.starts_with("data:") {
                let p = base_dir.join(uri.replace("%20", " "));
                if p.exists() {
                    out.push(p);
                } else {
                    let mut z = p.into_os_string();
                    z.push(".zst");
                    out.push(std::path::PathBuf::from(z));
                }
            }
        }
    }
    for image in gltf.document.images() {
        if let gltf::image::Source::Uri { uri, .. } = image.source() {
            if !uri.starts_with("data:") {
                out.push(crate::scene::resolve_texture_path(
                    base_dir.join(uri.replace("%20", " ")),
                ));
            }
        }
    }
    out
}

/// Resolve the document's buffers like `gltf::import_buffers`, plus the
/// engine's `.zst` sibling convention: an external buffer URI whose file is
/// absent falls back to `<file>.zst` and decodes transparently. Raw glTF
/// vertex/index buffers zstd ~2-3x (measured: Intel Sponza 133.5 -> 60.7 MB),
/// unlike the already-deflated PNG/JPG textures — so committed scenes carry
/// `.bin.zst` while a plain `.bin` next to the `.gltf` still loads verbatim.
fn resolve_buffers(
    doc: &gltf::Document,
    base_dir: &std::path::Path,
    mut blob: Option<Vec<u8>>,
) -> Result<Vec<gltf::buffer::Data>, String> {
    let mut out = Vec::new();
    for b in doc.buffers() {
        let mut data = match b.source() {
            gltf::buffer::Source::Bin => blob
                .take()
                .ok_or_else(|| "GLB buffer without a bin chunk".to_string())?,
            gltf::buffer::Source::Uri(uri) => {
                if let Some(rest) = uri.strip_prefix("data:") {
                    let b64 = rest
                        .split_once(',')
                        .map(|(_, b)| b)
                        .ok_or_else(|| format!("malformed buffer data URI in buffer {}", b.index()))?;
                    base64_decode(b64)
                        .ok_or_else(|| format!("bad base64 in buffer {} data URI", b.index()))?
                } else {
                    let p = base_dir.join(uri.replace("%20", " "));
                    if p.exists() {
                        std::fs::read(&p).map_err(|e| format!("read {}: {e}", p.display()))?
                    } else {
                        let mut z = p.as_os_str().to_owned();
                        z.push(".zst");
                        let z = std::path::PathBuf::from(z);
                        let f = std::fs::File::open(&z).map_err(|e| {
                            format!(
                                "buffer '{uri}': neither {} nor {} readable ({e})",
                                p.display(),
                                z.display()
                            )
                        })?;
                        zstd::stream::decode_all(std::io::BufReader::new(f))
                            .map_err(|e| format!("zstd decode {}: {e}", z.display()))?
                    }
                }
            }
        };
        if data.len() < b.length() {
            return Err(format!(
                "buffer {} shorter than declared ({} < {})",
                b.index(),
                data.len(),
                b.length()
            ));
        }
        // import_buffers pads to 4-byte multiples; accessor math relies on it.
        while data.len() % 4 != 0 {
            data.push(0);
        }
        out.push(gltf::buffer::Data(data));
    }
    Ok(out)
}

/// The loader body, shared with `self_test` (which passes decode_images =
/// true too but has no image references). Panics on malformed files — the
/// load_obj_scene convention.
fn build_scene(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    base_dir: &std::path::Path,
    verbose: bool,
) -> Scene {
    let mut b = SceneBuilder::new();
    let ground = b.material(Vec3A::new(0.42, 0.46, 0.40), 0.55, 0.0);
    let s = 60.0;
    b.quad(
        Vec3A::new(-s, 0.0, -s),
        Vec3A::new(-s, 0.0, s),
        Vec3A::new(s, 0.0, s),
        Vec3A::new(s, 0.0, -s),
        ground,
    );
    let _ = ground;

    // --- image roles from the supported material slots ---
    let mut roles: HashMap<usize, ImgRole> = HashMap::new();
    let mut n_ior = 0u32;
    let mut n_unlit = 0u32;
    let mut n_blend = 0u32;
    let mut n_cutoff = 0u32;
    for m in doc.materials() {
        let pbr = m.pbr_metallic_roughness();
        if let Some(info) = pbr.base_color_texture() {
            let e = roles.entry(info.texture().source().index()).or_default();
            e.base_color = true;
            if m.alpha_mode() != gltf::material::AlphaMode::Opaque {
                e.base_color_cutout = true;
            }
        }
        if let Some(info) = pbr.metallic_roughness_texture() {
            roles.entry(info.texture().source().index()).or_default().linear = true;
        }
        if let Some(nt) = m.normal_texture() {
            roles.entry(nt.texture().source().index()).or_default().linear = true;
        }
        if let Some(info) = m.emissive_texture() {
            roles.entry(info.texture().source().index()).or_default().emissive = true;
        }
        if m.alpha_mode() == gltf::material::AlphaMode::Blend {
            n_blend += 1;
        }
        if m.alpha_mode() != gltf::material::AlphaMode::Opaque
            && m.alpha_cutoff().is_some_and(|c| (c - 0.5).abs() > 0.05)
        {
            n_cutoff += 1;
        }
        if m.ior().is_some_and(|i| (i - 1.5).abs() > 1e-3) {
            n_ior += 1;
        }
        if m.unlit() {
            n_unlit += 1;
        }
    }

    // --- decode wanted images (rayon), sRGB vs linear per role. An image
    // referenced both as color and linear data decodes TWICE (one entry per
    // (image, srgb) — rare, correct). ---
    let mut wanted: Vec<(usize, bool)> = Vec::new();
    for (&img, r) in &roles {
        if r.base_color || r.emissive {
            wanted.push((img, true));
        }
        if r.linear {
            wanted.push((img, false));
        }
    }
    wanted.sort_unstable(); // deterministic ids (roles is a HashMap)
    let decoded: Vec<Option<Texture>> = {
        use rayon::prelude::*;
        // Largest-first (LPT) scheduling, mirroring the OBJ decode: sort an
        // index PERMUTATION and scatter results back by original index —
        // `wanted` order defines texture ids and must not shift.
        let approx_size = |img: usize| -> u64 {
            doc.images().nth(img).map_or(0, |image| match image.source() {
                gltf::image::Source::View { view, .. } => view.length() as u64,
                gltf::image::Source::Uri { uri, .. } => {
                    if uri.starts_with("data:") {
                        uri.len() as u64
                    } else {
                        let p = crate::scene::resolve_texture_path(
                            base_dir.join(uri.replace("%20", " ")),
                        );
                        std::fs::metadata(&p).map_or(0, |m| m.len())
                    }
                }
            })
        };
        let mut order: Vec<usize> = (0..wanted.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(approx_size(wanted[i].0)));
        crate::progress::phase(crate::progress::Phase::Textures, "", order.len() as u32);
        let mut pairs: Vec<(usize, Option<Texture>)> = order
            .par_iter()
            .with_max_len(1)
            .map(|&wi| {
                crate::progress::tick();
                let (img, srgb) = wanted[wi];
                let tex = (|| {
                    let image = doc.images().nth(img)?;
                    let bytes: Vec<u8> = match image.source() {
                        gltf::image::Source::View { view, .. } => {
                            let d = &buffers[view.buffer().index()].0;
                            d[view.offset()..view.offset() + view.length()].to_vec()
                        }
                        gltf::image::Source::Uri { uri, .. } => {
                            if let Some(data) = uri.strip_prefix("data:") {
                                let b64 = data.split_once(",").map(|(_, b)| b)?;
                                base64_decode(b64)?
                            } else {
                                // .webp sibling fallback — committed scenes
                                // store textures compressed while the glTF
                                // JSON keeps its .png URIs.
                                let p = crate::scene::resolve_texture_path(
                                    base_dir.join(uri.replace("%20", " ")),
                                );
                                match std::fs::read(&p) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        eprintln!(
                                            "warning: glTF image '{}' failed to read ({e})",
                                            p.display()
                                        );
                                        return None;
                                    }
                                }
                            }
                        }
                    };
                    match image::load_from_memory(&bytes) {
                        Ok(img_d) => Some(Texture::from_image(img_d, srgb)),
                        Err(e) => {
                            eprintln!("warning: glTF image {img} failed to decode ({e})");
                            None
                        }
                    }
                })();
                (wi, tex)
            })
            .collect();
        pairs.sort_unstable_by_key(|&(wi, _)| wi);
        pairs.into_iter().map(|(_, t)| t).collect()
    };
    let mut tex_ids: HashMap<(usize, bool), u32> = HashMap::new();
    for (&(img, srgb), tex) in wanted.iter().zip(decoded) {
        if let Some(mut t) = tex {
            // Per spec, OPAQUE materials ignore baseColor alpha — only a
            // MASK/BLEND baseColor role may arm the cutout pipeline.
            if !(srgb && roles[&img].base_color_cutout) {
                t.alpha_masked = false;
            }
            t.source = format!("gltf:image:{img}:{}", if srgb { "srgb" } else { "linear" });
            let id = b.add_texture(t);
            tex_ids.insert((img, srgb), id);
        }
    }

    // --- materials (matclass bypassed — real PBR factors) ---
    let lookup = |info: Option<usize>, srgb: bool| -> u32 {
        info.and_then(|img| tex_ids.get(&(img, srgb)).copied()).unwrap_or(NO_TEX)
    };
    let mat_map: Vec<u32> = doc
        .materials()
        .map(|m| {
            let pbr = m.pbr_metallic_roughness();
            let bc = pbr.base_color_factor();
            let base_tex =
                lookup(pbr.base_color_texture().map(|i| i.texture().source().index()), true);
            let mr_tex = lookup(
                pbr.metallic_roughness_texture().map(|i| i.texture().source().index()),
                false,
            );
            let (normal_tex, normal_scale) = m
                .normal_texture()
                .map(|nt| (lookup(Some(nt.texture().source().index()), false), nt.scale()))
                .unwrap_or((NO_TEX, 1.0));
            let strength = m.emissive_strength().unwrap_or(1.0);
            let emissive = Vec3A::from_array(m.emissive_factor()) * strength;
            let emissive_tex =
                lookup(m.emissive_texture().map(|i| i.texture().source().index()), true);
            let transmission =
                m.transmission().map(|t| t.transmission_factor()).unwrap_or(0.0);
            // KHR_materials_sheen has no crate feature in gltf 1.4 — read the
            // luminance of sheenColorFactor from the raw extension JSON.
            let sheen = m
                .extension_value("KHR_materials_sheen")
                .and_then(|v| v.get("sheenColorFactor"))
                .and_then(|c| c.as_array())
                .map(|a| {
                    let f = |i: usize| a.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                    (0.2126 * f(0) + 0.7152 * f(1) + 0.0722 * f(2)).clamp(0.0, 1.0)
                })
                .unwrap_or(0.0);
            let (mut roughness, mut metallic) = (pbr.roughness_factor(), pbr.metallic_factor());
            if m.unlit() {
                // No unlit path — matte diffuse is the least-wrong stand-in.
                roughness = 1.0;
                metallic = 0.0;
            }
            let kind = if base_tex != NO_TEX {
                MatKind::Textured { tex: base_tex }
            } else {
                MatKind::Diffuse
            };
            b.material_full(Material {
                albedo: Vec3A::new(bc[0], bc[1], bc[2]), // already linear
                roughness,
                metallic,
                anisotropy: 0.0,
                sheen,
                translucency: 0.0,
                transmission,
                // glTF glass keeps the sentinel tint (baseColor is authored
                // for tinting, so albedo IS the tint) and the fixed 1.5 IOR;
                // KHR_materials_ior → `ior` and a glTF water class are noted
                // follow-ons.
                trans_tint: Vec3A::splat(-1.0),
                ior: 1.5,
                ripple_amp: 0.0,
                emissive,
                normal_tex,
                normal_scale,
                height_amp: 0.0,
                rough_tex: mr_tex, // glTF packs roughness = G ...
                metal_tex: mr_tex, // ... and metallic = B of the SAME map
                emissive_tex,
                kind,
            })
        })
        .collect();
    // The spec default material (primitive.material() == None): baseColor
    // white, metallic 1.0, roughness 1.0.
    let default_mat = b.material(Vec3A::ONE, 1.0, 1.0);

    // --- flatten the node graph of the default scene ---
    let mut prims: Vec<Prim> = Vec::new();
    let (mut n_skipped_modes, mut n_color0, mut n_tangents) = (0u32, 0u32, 0u32);
    let scene_root = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .unwrap_or_else(|| panic!("glTF has no scene"));
    let mut stack: Vec<(gltf::Node, Mat4)> = scene_root
        .nodes()
        .map(|n| (n, Mat4::IDENTITY))
        .collect();
    while let Some((node, parent)) = stack.pop() {
        let m = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        for child in node.children() {
            stack.push((child, m));
        }
        let Some(mesh) = node.mesh() else { continue };
        let m3 = Mat3A::from_mat4(m);
        let mirrored = m3.determinant() < 0.0;
        let normal_m = m3.inverse().transpose();
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                n_skipped_modes += 1;
                continue;
            }
            let reader = prim.reader(|buf| buffers.get(buf.index()).map(|d| &d.0[..]));
            let Some(pos) = reader.read_positions() else { continue };
            let positions: Vec<Vec3A> =
                pos.map(|p| m.transform_point3a(Vec3A::from_array(p))).collect();
            let normals: Vec<Vec3A> = reader
                .read_normals()
                .map(|it| {
                    it.map(|n| (normal_m * Vec3A::from_array(n)).normalize_or_zero()).collect()
                })
                .unwrap_or_default();
            let texcoords: Vec<Vec2> = reader
                .read_tex_coords(0)
                .map(|tc| tc.into_f32().map(Vec2::from_array).collect())
                .unwrap_or_default();
            if reader.read_colors(0).is_some() {
                n_color0 += 1;
            }
            if reader.read_tangents().is_some() {
                n_tangents += 1; // ignored: one tangent source (shade.rs on-the-fly)
            }
            let flat: Vec<u32> = reader
                .read_indices()
                .map(|ix| ix.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());
            let indices: Vec<[u32; 3]> = flat
                .chunks_exact(3)
                .map(|t| {
                    if mirrored {
                        // Inverse-transpose already fixes the vertex normals
                        // under reflection; the winding flip keeps geometric
                        // face normals agreeing with them.
                        [t[0], t[2], t[1]]
                    } else {
                        [t[0], t[1], t[2]]
                    }
                })
                .collect();
            prims.push(Prim {
                positions,
                normals,
                texcoords,
                indices,
                material: prim.material().index(),
            });
        }
    }

    // --- auto-fit: the add_obj_models math (glTF is +Y-up like the engine;
    // uniform scale + translation leaves normals untouched) ---
    let mut mn = Vec3A::splat(f32::INFINITY);
    let mut mx = Vec3A::splat(f32::NEG_INFINITY);
    for p in prims.iter().flat_map(|pr| &pr.positions) {
        mn = mn.min(*p);
        mx = mx.max(*p);
    }
    let diagonal = (mx - mn).length().max(1e-6);
    let scale = 10.0 / diagonal;
    let center = (mn + mx) * 0.5;
    let shift = Vec3A::new(center.x, mn.y, center.z);
    let n_prims = prims.len();
    let mut n_tris = 0usize;
    for pr in &mut prims {
        for p in &mut pr.positions {
            *p = (*p - shift) * scale;
        }
        n_tris += pr.indices.len();
    }

    for pr in prims {
        let mat = pr
            .material
            .and_then(|i| mat_map.get(i).copied())
            .unwrap_or(default_mat);
        b.add_mesh(pr.positions, pr.normals, pr.texcoords, &pr.indices, mat);
    }
    // The engine's lighting model is the ONE sky (src/sky.rs); KHR_lights_punctual
    // is deliberately ignored, exactly as MTL lights are. Shared with the OBJ and
    // procedural loaders rather than re-spelled: the sun is now a single Vec3A, so
    // a second copy would just be a direction that could silently drift out of
    // step with theirs.
    let scene = b.finish(crate::scene::default_sun());
    if verbose {
        eprintln!(
            "gltf: {} prims | {} tris | {} materials | {} textures | skipped: {} non-tri prims{}{}{}{}{}{}",
            n_prims,
            n_tris,
            mat_map.len(),
            scene.textures.len(),
            n_skipped_modes,
            if n_tangents > 0 {
                format!(" | {n_tangents} authored-tangent prims ignored")
            } else {
                String::new()
            },
            if n_color0 > 0 { format!(" | {n_color0} COLOR_0 prims ignored") } else { String::new() },
            if n_blend > 0 { format!(" | {n_blend} BLEND materials as MASK") } else { String::new() },
            if n_ior > 0 { format!(" | {n_ior} non-1.5 IORs (fixed GLASS_IOR)") } else { String::new() },
            if n_unlit > 0 { format!(" | {n_unlit} unlit as matte") } else { String::new() },
            if n_cutoff > 0 {
                format!(" | {n_cutoff} non-0.5 alphaCutoffs (fixed 128/255 threshold)")
            } else {
                String::new()
            },
        );
    }
    scene
}

/// Minimal RFC-4648 base64 decoder for data: URIs (avoids a dependency; the
/// gltf crate decodes buffer URIs itself — this is only for image URIs).
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a') as u32 + 26),
            b'0'..=b'9' => Some((c - b'0') as u32 + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'\n' && c != b'\r').collect();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let mut acc = 0u32;
        let mut n = 0;
        for &c in chunk {
            if c == b'=' {
                break;
            }
            acc = (acc << 6) | val(c)?;
            n += 1;
        }
        match n {
            4 => out.extend_from_slice(&[(acc >> 16) as u8, (acc >> 8) as u8, acc as u8]),
            3 => {
                let acc = acc << 6;
                out.extend_from_slice(&[(acc >> 16) as u8, (acc >> 8) as u8]);
            }
            2 => {
                let acc = acc << 12;
                out.push((acc >> 16) as u8);
            }
            _ => {}
        }
    }
    Some(out)
}

/// Download-free structural coverage, run by `--check` beside
/// `matclass::self_test`: a GLB assembled in code (two nodes sharing one
/// mesh, the second mirrored by a negative scale; u16 indices; PBR factors)
/// exercises node flattening, the mirrored-winding flip, index widening, and
/// the factor mapping.
pub fn self_test() -> Result<(), String> {
    let bin: Vec<u8> = {
        let mut v = Vec::new();
        for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for f in p {
                v.extend_from_slice(&f.to_le_bytes());
            }
        }
        for i in [0u16, 1, 2] {
            v.extend_from_slice(&i.to_le_bytes());
        }
        v.extend_from_slice(&[0, 0]); // pad to 4
        v
    };
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},
"buffers":[{{"byteLength":{}}}],
"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}},{{"buffer":0,"byteOffset":36,"byteLength":6}}],
"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},{{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}}],
"materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[0.5,0.25,0.125,1.0],"metallicFactor":0.75,"roughnessFactor":0.5}},"emissiveFactor":[0.5,0.5,0.5]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}}]}}],
"nodes":[{{"mesh":0}},{{"mesh":0,"scale":[-1,1,1],"translation":[3,0,0]}}],
"scenes":[{{"nodes":[0,1]}}],
"scene":0}}"#,
        bin.len()
    );
    let mut json_b = json.into_bytes();
    while json_b.len() % 4 != 0 {
        json_b.push(b' ');
    }
    let mut glb = Vec::new();
    glb.extend_from_slice(&0x46546C67u32.to_le_bytes()); // "glTF"
    glb.extend_from_slice(&2u32.to_le_bytes());
    let total = 12 + 8 + json_b.len() + 8 + bin.len();
    glb.extend_from_slice(&(total as u32).to_le_bytes());
    glb.extend_from_slice(&(json_b.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // JSON
    glb.extend_from_slice(&json_b);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E4942u32.to_le_bytes()); // BIN
    glb.extend_from_slice(&bin);

    let gltf =
        gltf::Gltf::from_slice(&glb).map_err(|e| format!("self-test GLB parse: {e}"))?;
    let doc = gltf.document;
    let buffers = resolve_buffers(&doc, std::path::Path::new("."), gltf.blob)
        .map_err(|e| format!("self-test buffers: {e}"))?;
    let sc = build_scene(&doc, &buffers, std::path::Path::new("."), false);

    // Ground (2 tris) + two instances of the 1-tri mesh.
    if sc.tri_count() != 4 {
        return Err(format!("expected 4 tris (ground 2 + 2 instances), got {}", sc.tri_count()));
    }
    // Factor mapping (material 1: ground is 0... builder order: ground,
    // then the glTF material, then the default material).
    let m = &sc.materials[1];
    let want = Vec3A::new(0.5, 0.25, 0.125);
    if (m.albedo - want).length() > 1e-6
        || (m.metallic - 0.75).abs() > 1e-6
        || (m.roughness - 0.5).abs() > 1e-6
        || (m.emissive - Vec3A::splat(0.5)).length() > 1e-6
    {
        return Err("PBR factor mapping wrong".into());
    }
    // Mirrored-node winding: both triangles' geometric normals (from vertex
    // order) must agree (+Z here) — the flip compensates the reflection.
    let face_n = |t: usize| {
        let [i0, i1, i2] = sc.indices[t];
        let (a, bb, c) = (
            sc.positions[i0 as usize],
            sc.positions[i1 as usize],
            sc.positions[i2 as usize],
        );
        (bb - a).cross(c - a).normalize_or_zero()
    };
    let (n2, n3) = (face_n(2), face_n(3));
    if n2.z <= 0.9 || n3.z <= 0.9 {
        return Err(format!("mirrored winding flip failed: n2 {n2:?} n3 {n3:?}"));
    }
    // Welded-normal fallback ran (no NORMAL attribute in the GLB): the
    // vertex normals must be finite and +Z.
    let vn = sc.normals[sc.indices[2][0] as usize];
    if vn.z <= 0.9 {
        return Err(format!("welded normal fallback wrong: {vn:?}"));
    }

    // .bin.zst sibling fallback (resolve_buffers): a minimal external-buffer
    // glTF whose buffer exists ONLY as a zstd sibling must load — this is the
    // committed-scene shape (Sponza .bins are stored compressed).
    {
        let dir = std::env::temp_dir()
            .join(format!("frustracer-gltf-zst-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| format!("zst-test tmpdir: {e}"))?;
        let tri: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut bin = Vec::with_capacity(36);
        for v in tri {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let z = zstd::stream::encode_all(&bin[..], 3)
            .map_err(|e| format!("zst-test encode: {e}"))?;
        std::fs::write(dir.join("tri.bin.zst"), z)
            .map_err(|e| format!("zst-test write bin: {e}"))?;
        // The material's baseColorTexture URI says tex.png; only tex.webp
        // exists — the resolve_texture_path sibling fallback must find it
        // (the committed-scene shape: manifests keep .png names, disk holds
        // lossless WebP).
        let px = image::RgbaImage::from_pixel(2, 2, image::Rgba([200, 30, 30, 255]));
        let mut webp_bytes: Vec<u8> = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut webp_bytes)
            .encode(px.as_raw(), 2, 2, image::ExtendedColorType::Rgba8)
            .map_err(|e| format!("webp-test encode: {e}"))?;
        std::fs::write(dir.join("tex.webp"), webp_bytes)
            .map_err(|e| format!("webp-test write: {e}"))?;
        let json = r#"{"asset":{"version":"2.0"},
"buffers":[{"uri":"tri.bin","byteLength":36}],
"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36}],
"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}],
"images":[{"uri":"tex.png"}],
"textures":[{"source":0}],
"materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}],
"meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
"nodes":[{"mesh":0}],
"scenes":[{"nodes":[0]}],
"scene":0}"#;
        std::fs::write(dir.join("tri.gltf"), json)
            .map_err(|e| format!("zst-test write gltf: {e}"))?;
        let g = gltf::Gltf::open(dir.join("tri.gltf"))
            .map_err(|e| format!("zst-test open: {e}"))?;
        let bufs = resolve_buffers(&g.document, &dir, g.blob)
            .map_err(|e| format!("zst-test resolve: {e}"))?;
        let zsc = build_scene(&g.document, &bufs, &dir, false);
        let _ = std::fs::remove_dir_all(&dir);
        // Ground quad (2) + the one decompressed triangle.
        if zsc.tri_count() != 3 {
            return Err(format!(
                "zst sibling fallback: expected 3 tris, got {}",
                zsc.tri_count()
            ));
        }
        if zsc.textures.len() != 1 || zsc.textures[0].texels[0] != [200, 30, 30, 255] {
            return Err(format!(
                "webp sibling fallback: expected 1 texture with the exact texel, got {} texture(s), first texel {:?}",
                zsc.textures.len(),
                zsc.textures.first().map(|t| t.texels[0])
            ));
        }
    }
    Ok(())
}
