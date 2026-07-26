//! Binary sidecar cache of a parsed `Scene` + built `Bvh` (`<source>.fcache`,
//! next to the OBJ/`.obj.zst` it came from). Manual POD format, std only — no
//! serde, no mmap: the read path preallocates each typed Vec exactly and
//! `read_exact`s into its byte view, so a San Miguel load drops from ~a minute
//! of parse + normal-gen + BVH build to under a second. Keyed on the RESOLVED
//! source file's size + mtime (plus the `.mtl` sibling's and each TEXTURE
//! file's — `alpha_masked` and the height-map detection/conversions are
//! texture-CONTENT decisions, so an edited texture must miss the whole
//! cache, not resurface stale flags) and `CACHE_VERSION`; any mismatch —
//! including a truncated or
//! bit-rotted sidecar — is a silent miss, never a panic: every count is
//! capped against the real file size before allocating, and the payload's
//! cross-array links are validated before anything indexes them. Texels are
//! deliberately NOT cached (decode is already rayon-parallel and takes
//! seconds): the file stores each texture's resolved path + `alpha_masked`
//! in id order and re-decodes on load, substituting a 1×1 white texture
//! (with the cached flag) on failure — preserving ids, where a fresh load
//! would shift them. Sidecars are per-machine derived artifacts: gitignored,
//! never committed.
//!
//! A second format shares the machinery: the WORLD sidecar
//! (`scenes/world.fcache`, see the section at the bottom) caches the merged
//! world Scene + BVH + layout under a multi-source key, with an INLINE
//! texture flavor for glTF's path-less textures.
//!
//! Bump `CACHE_VERSION` on ANY change to: the Scene/Material/Texture layout,
//! the Bvh build (node order is part of the contract), the OBJ loader, or
//! matclass classification.

use crate::bvh::{Bvh, BvhNode};
use crate::scene::{self, MatKind, Material, Scene};
use crate::texture::Texture;
use glam::{Vec2, Vec3A};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

// v5 supersedes two INCOMPATIBLE v4 lineages (build_key header on one side,
// .webp sibling resolution on the other) — a bare version match can't tell
// them apart, so both are invalidated.
// The TOD fields (Scene::sky_scale/night) needed NO bump: they are
// derived-only (`apply_tod` runs after load/store, the loader initializes
// them to the defaults), so the v6 on-disk layout is unchanged.
// v7: grayscale map_Bump height→normal conversion (h2n/n2h flag bytes so the
// warm-load re-decode re-applies the conversions; Material/DiskMat gained
// height_amp; texture ids shift on scenes with height maps, which used to be
// dropped and are now converted-and-kept).
// v8: spray reclassification (tinted-shadows part 2) — the load-time pass
// retags tiny transmissive components' tri_mat to appended spray materials,
// so a v7 sidecar's materials/tri_mat are stale for spray-enabled sessions.
// v9: NOT a layout change — invalidates sidecars written by world-mode cold
// boots, which skipped `reclassify_spray` before storing (world.rs's cold
// arm; fixed alongside this bump). Those sidecars are wrong under the same
// key and only a version bump can retire them.
// v10: the water class — Material/DiskMat gained trans_tint/ior/ripple_amp,
// and the loader stamps water params + skips the dark-glass albedo lift for
// the fountain, so a v9 sidecar's materials are stale (the --no-water lever
// also joins the lever word, bit 4).
pub const CACHE_VERSION: u32 = 10;
const MAGIC: [u8; 8] = *b"FRSCACH\x01";

/// Fixed on-disk material, `MatKind` flattened into (kind, param) — Marble
/// stores `scale.to_bits()`, Textured the tex id.
#[repr(C)]
#[derive(Clone, Copy)]
struct DiskMat {
    albedo: [f32; 3],
    roughness: f32,
    metallic: f32,
    anisotropy: f32,
    sheen: f32,
    translucency: f32,
    transmission: f32,
    kind: u32,
    param: u32,
    emissive: [f32; 3],
    normal_scale: f32,
    height_amp: f32,
    normal_tex: u32,
    rough_tex: u32,
    metal_tex: u32,
    emissive_tex: u32,
    trans_tint: [f32; 3],
    ior: f32,
    ripple_amp: f32,
}

/// Fixed on-disk BVH node: 32 B, no padding (the in-memory `BvhNode` is
/// 48 B from Vec3A alignment — writing it raw would serialize padding).
#[repr(C)]
#[derive(Clone, Copy)]
struct DiskNode {
    min: [f32; 3],
    max: [f32; 3],
    left_first: u32,
    count: u32,
}

/// The load-time lever word, part of the cache key: the h2n/n2h converters
/// change texel content + materials, and the heightfield session lever
/// changes the BUILT tree's AABBs (the inward sweep) — a sidecar written
/// under one lever state must never be served to a session under another
/// (the levers would silently not take effect on warm loads).
fn lever_word() -> u32 {
    crate::texture::h2n_enabled() as u32
        | (crate::texture::n2h_enabled() as u32) << 1
        | (crate::bvh::height_armed() as u32) << 2
        // Spray reclassification rewrites the cached tri_mat/materials — a
        // sidecar written under one spray state must miss under the other.
        | (crate::scene::spray_enabled() as u32) << 3
        // The water class refines the fountain's material (tint/ior/ripple +
        // no albedo lift), baked into the cached materials.
        | (crate::scene::water_enabled() as u32) << 4
}

/// (size, mtime-ns) of a file, (0, 0) if absent — the staleness key.
fn stat_key(p: &Path) -> (u64, u64) {
    let Ok(md) = std::fs::metadata(p) else { return (0, 0) };
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    (md.len(), mtime)
}

/// The `.mtl` sibling probed for staleness: strip a trailing `.zst`, then
/// swap the extension to `.mtl` (the archive convention — MTLs stay plain
/// text). A scene whose mtllib is named differently only misses MTL edits in
/// the key; CACHE_VERSION bumps cover loader-side changes.
pub(crate) fn mtl_sibling(src: &Path) -> PathBuf {
    let s = src.to_string_lossy();
    let base = s.strip_suffix(".zst").unwrap_or(&s);
    PathBuf::from(base).with_extension("mtl")
}

fn sidecar(src: &Path) -> PathBuf {
    let mut s = src.as_os_str().to_owned();
    s.push(".fcache");
    PathBuf::from(s)
}

fn write_pod<T: Copy>(w: &mut impl Write, data: &[T]) -> std::io::Result<()> {
    // Sound for the POD types used here (no padding / fully-initialized lanes).
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size_of_val(data)) };
    w.write_all(bytes)
}

fn read_pod_vec<T: Copy>(r: &mut impl Read, n: usize) -> std::io::Result<Vec<T>> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    unsafe {
        let bytes =
            std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * size_of::<T>());
        r.read_exact(bytes)?;
        v.set_len(n);
    }
    Ok(v)
}

/// `read_pod_vec` gated on the sidecar's REAL remaining byte budget: a
/// corrupt count field must be a silent miss (None), never a capacity panic
/// or a multi-GB allocation. `remaining` starts at the file length and only
/// ever shrinks, so the sum of all checked reads can't exceed the file.
fn read_pod_vec_checked<T: Copy>(
    r: &mut impl Read,
    n: usize,
    remaining: &mut u64,
) -> Option<Vec<T>> {
    let bytes = (n as u64).checked_mul(size_of::<T>() as u64)?;
    if bytes > *remaining {
        return None;
    }
    *remaining -= bytes;
    read_pod_vec(r, n).ok()
}

fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Cross-array link validation shared by both formats: a bad sidecar must be
/// a miss, not a later panic — this runs before the payload reaches code
/// that indexes unchecked (Bvh traversal, shade's texture fetches).
#[allow(clippy::too_many_arguments)]
fn validate_links(
    indices: &[[u32; 3]],
    tri_mat: &[u32],
    tri_idx: &[u32],
    disk_nodes: &[DiskNode],
    disk_mats: &[DiskMat],
    n_verts: usize,
    n_tris: usize,
    n_mats: usize,
    n_tex: usize,
    n_nodes: usize,
    n_tri_idx: usize,
) -> bool {
    use rayon::prelude::*;
    let tex_ok = |t: u32| t == scene::NO_TEX || (t as usize) < n_tex;
    indices
        .par_iter()
        .all(|t| t.iter().all(|&i| (i as usize) < n_verts))
        && tri_mat.par_iter().all(|&m| (m as usize) < n_mats)
        && tri_idx.par_iter().all(|&t| (t as usize) < n_tris)
        && disk_nodes.par_iter().all(|n| {
            if n.count > 0 {
                n.left_first as usize + n.count as usize <= n_tri_idx
            } else {
                (n.left_first as usize) + 1 < n_nodes
            }
        })
        && disk_mats.iter().all(|m| {
            tex_ok(m.normal_tex)
                && tex_ok(m.rough_tex)
                && tex_ok(m.metal_tex)
                && tex_ok(m.emissive_tex)
                && (m.kind != 2 || (m.param as usize) < n_tex)
        })
}

fn mat_to_disk(m: &Material) -> DiskMat {
    let (kind, param) = match m.kind {
        MatKind::Diffuse => (0, 0),
        MatKind::Marble { scale } => (1, scale.to_bits()),
        MatKind::Textured { tex } => (2, tex),
    };
    DiskMat {
        albedo: m.albedo.to_array(),
        roughness: m.roughness,
        metallic: m.metallic,
        anisotropy: m.anisotropy,
        sheen: m.sheen,
        translucency: m.translucency,
        transmission: m.transmission,
        kind,
        param,
        emissive: m.emissive.to_array(),
        normal_scale: m.normal_scale,
        height_amp: m.height_amp,
        normal_tex: m.normal_tex,
        rough_tex: m.rough_tex,
        metal_tex: m.metal_tex,
        emissive_tex: m.emissive_tex,
        trans_tint: m.trans_tint.to_array(),
        ior: m.ior,
        ripple_amp: m.ripple_amp,
    }
}

fn mat_from_disk(m: &DiskMat) -> Material {
    Material {
        albedo: Vec3A::from_array(m.albedo),
        roughness: m.roughness,
        metallic: m.metallic,
        anisotropy: m.anisotropy,
        sheen: m.sheen,
        translucency: m.translucency,
        transmission: m.transmission,
        trans_tint: Vec3A::from_array(m.trans_tint),
        ior: m.ior,
        ripple_amp: m.ripple_amp,
        emissive: Vec3A::from_array(m.emissive),
        normal_tex: m.normal_tex,
        normal_scale: m.normal_scale,
        height_amp: m.height_amp,
        rough_tex: m.rough_tex,
        metal_tex: m.metal_tex,
        emissive_tex: m.emissive_tex,
        kind: match m.kind {
            1 => MatKind::Marble { scale: f32::from_bits(m.param) },
            2 => MatKind::Textured { tex: m.param },
            _ => MatKind::Diffuse,
        },
    }
}

fn nodes_to_disk(bvh: &Bvh) -> Vec<DiskNode> {
    bvh.nodes
        .iter()
        .map(|n| DiskNode {
            min: n.aabb.min.to_array(),
            max: n.aabb.max.to_array(),
            left_first: n.left_first,
            count: n.count,
        })
        .collect()
}

fn nodes_from_disk(disk: &[DiskNode]) -> Vec<BvhNode> {
    disk.iter()
        .map(|n| BvhNode {
            aabb: crate::bvh::Aabb {
                min: Vec3A::from_array(n.min),
                max: Vec3A::from_array(n.max),
            },
            left_first: n.left_first,
            count: n.count,
        })
        .collect()
}

/// One texture's on-disk record, load side. `Path` re-decodes from the file
/// (the per-scene format's only flavor); `Inline` carries POST-conversion
/// texels verbatim — the world format's flavor for sources with no decodable
/// path (glTF's synthetic `"gltf:image:…"` ids, empty test sources) — and
/// only rebuilds mips.
enum TexRecord {
    Path {
        path: String,
        masked: bool,
        srgb: bool,
        h2n: bool,
        n2h: bool,
    },
    Inline {
        w: u32,
        h: u32,
        masked: bool,
        srgb: bool,
        h2n: bool,
        n2h: bool,
        source: String,
        texels: Vec<[u8; 4]>,
    },
}

/// Materialize the records IN ID ORDER; a `Path` failure keeps the slot with
/// a 1×1 white texture carrying the cached flags so material tex ids never
/// shift. Largest-first (LPT) scheduling via an index permutation, scattered
/// back so ids never shift (see the scene.rs decode note — WebP's slower
/// per-file decode makes the big-file tail dominate).
fn decode_tex_records(recs: Vec<TexRecord>) -> Vec<Texture> {
    use rayon::prelude::*;
    let mut order: Vec<usize> = (0..recs.len()).collect();
    order.sort_by_key(|&i| {
        std::cmp::Reverse(match &recs[i] {
            TexRecord::Path { path, .. } => std::fs::metadata(path).map_or(0, |m| m.len()),
            // An Inline "decode" is the mip rebuild — schedule by texel bytes.
            TexRecord::Inline { texels, .. } => (texels.len() * 4) as u64,
        })
    });
    // Per-slot cells so the parallel loop can MOVE each record out (inline
    // texel payloads must not clone); each cell is taken exactly once.
    let cells: Vec<std::sync::Mutex<Option<TexRecord>>> =
        recs.into_iter().map(|r| std::sync::Mutex::new(Some(r))).collect();
    let mut pairs: Vec<(usize, Texture)> = order
        .par_iter()
        .with_max_len(1)
        .map(|&i| {
            let rec = cells[i].lock().unwrap().take().expect("each record taken once");
            let t = match rec {
                TexRecord::Path { path, masked, srgb, h2n, n2h } => {
                    let mut t = match image::open(&path) {
                        Ok(img) => Texture::from_image(img, srgb),
                        Err(e) => {
                            eprintln!(
                                "warning: cached texture '{path}' failed to re-decode ({e}); using 1x1 white"
                            );
                            Texture {
                                w: 1,
                                h: 1,
                                texels: vec![[255; 4]],
                                alpha_masked: false,
                                srgb,
                                source: String::new(),
                                h2n: false,
                                n2h: false,
                                mips: Vec::new(),
                            }
                        }
                    };
                    // The load-time conversions are texture-CONTENT decisions
                    // baked into the cached flags — a warm load must re-apply
                    // them or the material's normal_tex points at raw
                    // grayscale texels. The 1×1-white fallback converts to a
                    // flat identity normal map — semantically the right
                    // substitute.
                    if h2n {
                        t = t.height_to_normal();
                    }
                    if n2h {
                        // Deterministic re-derivation — same texels, same
                        // field (the warm image must be bit-identical to the
                        // cold one; the returned amp is discarded: materials
                        // carry theirs in DiskMat).
                        t.apply_n2h();
                    }
                    // The loader's role-gating (e.g. an emissive-only color
                    // map never arms the cutout) is baked into the cached
                    // flag — restore it verbatim rather than re-deriving.
                    // Sound because the per-texture stat key already missed
                    // the cache if the file's content changed.
                    t.alpha_masked = masked;
                    t.source = path;
                    t
                }
                TexRecord::Inline { w, h, masked, srgb, h2n, n2h, source, texels } => {
                    Texture::from_cached(w, h, texels, masked, srgb, source, h2n, n2h)
                }
            };
            (i, t)
        })
        .collect();
    pairs.sort_unstable_by_key(|&(i, _)| i);
    pairs.into_iter().map(|(_, t)| t).collect()
}

/// Load the sidecar for an already-RESOLVED source path (see
/// `scene::resolve_scene_path`). Silent None on any mismatch or read error.
pub fn try_load(src_path: &str) -> Option<(Scene, Bvh)> {
    crate::progress::phase(crate::progress::Phase::Cache, src_path, 0);
    let src = Path::new(src_path);
    let f = std::fs::File::open(sidecar(src)).ok()?;
    // The byte budget every payload read is capped against (see
    // `read_pod_vec_checked`) — header/meta reads don't bother debiting it,
    // erring a couple hundred bytes permissive, which `read_exact`'s own
    // EOF failure covers.
    let mut remaining = f.metadata().ok()?.len();
    let mut r = BufReader::new(f);
    let t0 = std::time::Instant::now();

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).ok()?;
    if magic != MAGIC || read_u32(&mut r).ok()? != CACHE_VERSION {
        return None;
    }
    // Build parameters are part of the key: the sidecar holds a BUILT tree.
    if read_u64(&mut r).ok()? != crate::bvh::build_key() {
        return None;
    }
    if read_u32(&mut r).ok()? != lever_word() {
        return None;
    }
    let (src_size, src_mtime) = stat_key(src);
    let (mtl_size, mtl_mtime) = stat_key(&mtl_sibling(src));
    if read_u64(&mut r).ok()? != src_size
        || read_u64(&mut r).ok()? != src_mtime
        || read_u64(&mut r).ok()? != mtl_size
        || read_u64(&mut r).ok()? != mtl_mtime
    {
        return None;
    }

    let n_verts = read_u64(&mut r).ok()? as usize;
    let n_tris = read_u64(&mut r).ok()? as usize;
    let n_mats = read_u64(&mut r).ok()? as usize;
    let n_tex = read_u64(&mut r).ok()? as usize;
    let n_nodes = read_u64(&mut r).ok()? as usize;
    let n_tri_idx = read_u64(&mut r).ok()? as usize;
    // A real cache always holds at least the ground quad and a BVH root;
    // zeros here mean corruption (and would panic downstream — an empty
    // nodes Vec breaks Bvh::intersect's unconditional nodes[0] read).
    if n_verts == 0 || n_tris == 0 || n_nodes == 0 {
        return None;
    }
    let sun_v: Vec<Vec3A> = read_pod_vec_checked(&mut r, 1, &mut remaining)?;

    let positions: Vec<Vec3A> = read_pod_vec_checked(&mut r, n_verts, &mut remaining)?;
    let normals: Vec<Vec3A> = read_pod_vec_checked(&mut r, n_verts, &mut remaining)?;
    let texcoords: Vec<Vec2> = read_pod_vec_checked(&mut r, n_verts, &mut remaining)?;
    let indices: Vec<[u32; 3]> = read_pod_vec_checked(&mut r, n_tris, &mut remaining)?;
    let tri_mat: Vec<u32> = read_pod_vec_checked(&mut r, n_tris, &mut remaining)?;
    let disk_mats: Vec<DiskMat> = read_pod_vec_checked(&mut r, n_mats, &mut remaining)?;

    let mut tex_recs: Vec<TexRecord> = Vec::with_capacity(n_tex.min(4096));
    for _ in 0..n_tex {
        let len = read_u32(&mut r).ok()? as usize;
        if len as u64 > remaining {
            return None;
        }
        remaining -= len as u64;
        let mut path = vec![0u8; len];
        r.read_exact(&mut path).ok()?;
        let mut flags = [0u8; 4];
        r.read_exact(&mut flags).ok()?;
        let path = String::from_utf8(path).ok()?;
        // Texture-content staleness: alpha_masked (restored verbatim on
        // decode) and the loader's height-map detection are functions of the
        // FILE's pixels — an edited/replaced texture must miss the whole
        // cache.
        let (t_size, t_mtime) = (read_u64(&mut r).ok()?, read_u64(&mut r).ok()?);
        if stat_key(Path::new(&path)) != (t_size, t_mtime) {
            return None;
        }
        tex_recs.push(TexRecord::Path {
            path,
            masked: flags[0] != 0,
            srgb: flags[1] != 0,
            h2n: flags[2] != 0,
            n2h: flags[3] != 0,
        });
    }

    let disk_nodes: Vec<DiskNode> = read_pod_vec_checked(&mut r, n_nodes, &mut remaining)?;
    let tri_idx: Vec<u32> = read_pod_vec_checked(&mut r, n_tri_idx, &mut remaining)?;

    if !validate_links(
        &indices, &tri_mat, &tri_idx, &disk_nodes, &disk_mats, n_verts, n_tris, n_mats, n_tex,
        n_nodes, n_tri_idx,
    ) {
        return None;
    }

    let materials: Vec<Material> = disk_mats.iter().map(mat_from_disk).collect();
    let textures = decode_tex_records(tex_recs);
    let nodes = nodes_from_disk(&disk_nodes);

    let mut sc = Scene {
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
        sun: crate::sky::Sun::new(sun_v[0]),
        // Derived from the sun by finalize_scalars below — deliberately not in
        // the on-disk format, so the SH sky costs the cache nothing. The TOD
        // state (sky_scale/night) is likewise derived-only: a `--tod` override
        // applies AFTER load/store, so the cache always holds the default day.
        sky_sh: crate::sh::Sh9::ZERO,
        sky_scale: 1.0,
        night: 0.0,
        diag: 0.0,
        eps: 0.0,
        ao_radius: 0.0,
        content_min: glam::Vec3A::ZERO,
        content_max: glam::Vec3A::ZERO,
    };
    scene::finalize_scalars(&mut sc);
    eprintln!(
        "scene cache: hit — {} tris, {} nodes, {} textures in {:.0} ms",
        sc.tri_count(),
        nodes.len(),
        sc.textures.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Some((sc, Bvh::from_parts(nodes, tri_idx)))
}

/// Best-effort store (temp file + rename; a failure prints one line and the
/// run continues uncached). `src_path` must be the RESOLVED source.
pub fn store(src_path: &str, scene: &Scene, bvh: &Bvh) {
    crate::progress::phase(crate::progress::Phase::Sidecar, src_path, 0);
    let src = Path::new(src_path);
    let dst = sidecar(src);
    // Pid-suffixed so two processes loading the same scene can't interleave
    // writes into one tmp file (the rename itself is atomic either way).
    let tmp = dst.with_extension(format!("fcache.{}.tmp", std::process::id()));
    let result = (|| -> std::io::Result<()> {
        let mut w = BufWriter::new(std::fs::File::create(&tmp)?);
        w.write_all(&MAGIC)?;
        write_u32(&mut w, CACHE_VERSION)?;
        // The sidecar stores the BUILT tree, so it must also key on the build
        // PARAMETERS — a cache written at one (c_trav, max_leaf) must never be
        // served to a session asking for another, or the sweep silently
        // benchmarks the same tree every time. Distinct from CACHE_VERSION,
        // which versions the format.
        write_u64(&mut w, crate::bvh::build_key())?;
        write_u32(&mut w, lever_word())?;
        let (src_size, src_mtime) = stat_key(src);
        let (mtl_size, mtl_mtime) = stat_key(&mtl_sibling(src));
        write_u64(&mut w, src_size)?;
        write_u64(&mut w, src_mtime)?;
        write_u64(&mut w, mtl_size)?;
        write_u64(&mut w, mtl_mtime)?;
        write_u64(&mut w, scene.positions.len() as u64)?;
        write_u64(&mut w, scene.indices.len() as u64)?;
        write_u64(&mut w, scene.materials.len() as u64)?;
        write_u64(&mut w, scene.textures.len() as u64)?;
        write_u64(&mut w, bvh.nodes.len() as u64)?;
        write_u64(&mut w, bvh.tri_idx.len() as u64)?;
        // Only the DIRECTION: the sun's cone and its two radiometric values are
        // pure functions of it (sky::Sun::new), so storing them would create a
        // second source of truth that could drift from sky.rs's constants.
        write_pod(&mut w, &[scene.sun.dir])?;

        write_pod(&mut w, &scene.positions)?;
        write_pod(&mut w, &scene.normals)?;
        write_pod(&mut w, &scene.texcoords)?;
        write_pod(&mut w, &scene.indices)?;
        write_pod(&mut w, &scene.tri_mat)?;
        let disk_mats: Vec<DiskMat> = scene.materials.iter().map(mat_to_disk).collect();
        write_pod(&mut w, &disk_mats)?;
        for t in &scene.textures {
            write_u32(&mut w, t.source.len() as u32)?;
            w.write_all(t.source.as_bytes())?;
            w.write_all(&[t.alpha_masked as u8, t.srgb as u8, t.h2n as u8, t.n2h as u8])?;
            // Per-texture staleness key (see try_load): content edits to a
            // texture must invalidate the cached alpha_masked/height-map
            // decisions, which only a full reparse re-derives.
            let (t_size, t_mtime) = stat_key(Path::new(&t.source));
            write_u64(&mut w, t_size)?;
            write_u64(&mut w, t_mtime)?;
        }
        write_pod(&mut w, &nodes_to_disk(bvh))?;
        write_pod(&mut w, &bvh.tri_idx)?;
        w.flush()?;
        drop(w);
        std::fs::rename(&tmp, &dst)
    })();
    match result {
        Ok(()) => eprintln!("scene cache: wrote {}", dst.display()),
        Err(e) => {
            eprintln!("scene cache: write failed ({e}) — continuing uncached");
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

// ---------------------------------------------------------------------------
// World sidecar (`scenes/world.fcache`): the MERGED world Scene + its BVH +
// the `world::World` layout in one file, so a warm world boot skips every
// per-part load, the merge, AND the world BVH build. Same POD machinery and
// corruption discipline as the per-scene format; what differs is the KEY —
// one world has many sources, so the key is a serialized blob covering every
// curated entry's identity, PRESENCE (an island appearing on disk must miss),
// and dependency file stats — and the TEXTURE payload, which grows an INLINE
// flavor for sources with no decodable path (glTF's synthetic ids): their
// post-conversion texels are stored verbatim, mips rebuilt on load. Keyed
// additionally on `WORLD_VERSION` (world-format/layout changes — ring
// constants ride the key blob), `CACHE_VERSION` (the shared payload repr),
// `bvh::build_key()`, and `lever_word()`.

/// Bump on world-format layout changes (the key blob's shape included);
/// `CACHE_VERSION` bumps invalidate the world sidecar too (shared repr).
// 2: Island gained `height` (the content y extent) — the world layout record
//    grew a field, so every v1 sidecar must miss rather than deserialize short.
pub const WORLD_VERSION: u32 = 2;
const WORLD_MAGIC: [u8; 8] = *b"FRWORLD\x01";

/// One curated entry's contribution to the world key. `deps` holds the
/// RESOLVED paths of every file loading this island reads (source + mtl for
/// OBJ; source + external buffers/images for glTF); their (size, mtime)
/// stats are taken AT SERIALIZATION TIME, so a stored blob and a freshly
/// built one compare equal iff every dependency is byte-for-byte unchanged.
/// An absent island contributes `present: false` with empty deps — presence
/// itself is part of the key.
pub struct WorldKeyEntry {
    pub path: String,
    pub theme_hour_bits: u32,
    pub present: bool,
    pub deps: Vec<String>,
}

pub struct WorldKey {
    pub entries: Vec<WorldKeyEntry>,
    /// Ring-layout constants (f32 bits): a retuned world layout must miss.
    pub pitch_k_bits: u32,
    pub ground_margin_bits: u32,
}

/// ONE serializer for the key blob — store writes it, load re-serializes the
/// fresh key and memcmps, so any drift (presence flip, stat change, entry
/// count/order, hour bits, layout constants) is a miss by construction.
fn world_key_blob(key: &WorldKey) -> Vec<u8> {
    let mut v = Vec::new();
    let w = &mut v;
    let _ = write_u32(w, key.pitch_k_bits);
    let _ = write_u32(w, key.ground_margin_bits);
    let _ = write_u32(w, key.entries.len() as u32);
    for e in &key.entries {
        let _ = write_u32(w, e.path.len() as u32);
        let _ = w.write_all(e.path.as_bytes());
        let _ = write_u32(w, e.theme_hour_bits);
        let _ = w.write_all(&[e.present as u8]);
        let _ = write_u32(w, e.deps.len() as u32);
        for d in &e.deps {
            let _ = write_u32(w, d.len() as u32);
            let _ = w.write_all(d.as_bytes());
            let (sz, mt) = stat_key(Path::new(d));
            let _ = write_u64(w, sz);
            let _ = write_u64(w, mt);
        }
    }
    v
}

fn write_f32(w: &mut impl Write, v: f32) -> std::io::Result<()> {
    write_u32(w, v.to_bits())
}

fn read_f32(r: &mut impl Read) -> std::io::Result<f32> {
    Ok(f32::from_bits(read_u32(r)?))
}

/// Load the world sidecar. `key` is the FRESHLY built key (world.rs probes
/// presence + deps every boot — cheap stats); any mismatch, truncation, or
/// corruption is a silent None. On a hit the caller skips per-part loads,
/// the merge, and the world BVH build.
pub fn try_load_world(
    path: &Path,
    key: &WorldKey,
) -> Option<(Scene, Bvh, crate::world::World)> {
    let f = std::fs::File::open(path).ok()?;
    let mut remaining = f.metadata().ok()?.len();
    let mut r = BufReader::new(f);
    let t0 = std::time::Instant::now();

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).ok()?;
    if magic != WORLD_MAGIC
        || read_u32(&mut r).ok()? != WORLD_VERSION
        || read_u32(&mut r).ok()? != CACHE_VERSION
    {
        return None;
    }
    if read_u64(&mut r).ok()? != crate::bvh::build_key() {
        return None;
    }
    if read_u32(&mut r).ok()? != lever_word() {
        return None;
    }
    // Counts at a FIXED offset (28), before the variable-length key blob —
    // the zero-guard corruption test patches them by offset.
    let n_verts = read_u64(&mut r).ok()? as usize;
    let n_tris = read_u64(&mut r).ok()? as usize;
    let n_mats = read_u64(&mut r).ok()? as usize;
    let n_tex = read_u64(&mut r).ok()? as usize;
    let n_nodes = read_u64(&mut r).ok()? as usize;
    let n_tri_idx = read_u64(&mut r).ok()? as usize;
    if n_verts == 0 || n_tris == 0 || n_nodes == 0 {
        return None;
    }

    // Key blob: byte-compare against the fresh key's serialization.
    let fresh = world_key_blob(key);
    let blob_len = read_u32(&mut r).ok()? as usize;
    if blob_len as u64 > remaining {
        return None;
    }
    remaining -= blob_len as u64;
    let mut blob = vec![0u8; blob_len];
    r.read_exact(&mut blob).ok()?;
    if blob != fresh {
        return None;
    }

    // World layout meta.
    let n_islands = read_u32(&mut r).ok()? as usize;
    if n_islands == 0 {
        return None;
    }
    let mut islands = Vec::with_capacity(n_islands.min(256));
    for _ in 0..n_islands {
        let len = read_u32(&mut r).ok()? as usize;
        if len as u64 > remaining {
            return None;
        }
        remaining -= len as u64;
        let mut name = vec![0u8; len];
        r.read_exact(&mut name).ok()?;
        let center: Vec<Vec3A> = read_pod_vec_checked(&mut r, 1, &mut remaining)?;
        islands.push(crate::world::Island {
            name: String::from_utf8(name).ok()?,
            center: center[0],
            radius: read_f32(&mut r).ok()?,
            height: read_f32(&mut r).ok()?,
            theme_hour: read_f32(&mut r).ok()?,
        });
    }
    let field_half = read_f32(&mut r).ok()?;
    let sun_v: Vec<Vec3A> = read_pod_vec_checked(&mut r, 1, &mut remaining)?;

    let positions: Vec<Vec3A> = read_pod_vec_checked(&mut r, n_verts, &mut remaining)?;
    let normals: Vec<Vec3A> = read_pod_vec_checked(&mut r, n_verts, &mut remaining)?;
    let texcoords: Vec<Vec2> = read_pod_vec_checked(&mut r, n_verts, &mut remaining)?;
    let indices: Vec<[u32; 3]> = read_pod_vec_checked(&mut r, n_tris, &mut remaining)?;
    let tri_mat: Vec<u32> = read_pod_vec_checked(&mut r, n_tris, &mut remaining)?;
    let disk_mats: Vec<DiskMat> = read_pod_vec_checked(&mut r, n_mats, &mut remaining)?;

    let mut tex_recs: Vec<TexRecord> = Vec::with_capacity(n_tex.min(4096));
    for _ in 0..n_tex {
        let len = read_u32(&mut r).ok()? as usize;
        if len as u64 > remaining {
            return None;
        }
        remaining -= len as u64;
        let mut source = vec![0u8; len];
        r.read_exact(&mut source).ok()?;
        let source = String::from_utf8(source).ok()?;
        let mut flags = [0u8; 5];
        r.read_exact(&mut flags).ok()?;
        let (masked, srgb, h2n, n2h) =
            (flags[0] != 0, flags[1] != 0, flags[2] != 0, flags[3] != 0);
        match flags[4] {
            0 => {
                // Path flavor: the per-texture staleness stat, exactly the
                // per-scene rule.
                let (t_size, t_mtime) = (read_u64(&mut r).ok()?, read_u64(&mut r).ok()?);
                if stat_key(Path::new(&source)) != (t_size, t_mtime) {
                    return None;
                }
                tex_recs.push(TexRecord::Path { path: source, masked, srgb, h2n, n2h });
            }
            1 => {
                let w = read_u32(&mut r).ok()?;
                let h = read_u32(&mut r).ok()?;
                let n = (w as u64).checked_mul(h as u64)?;
                if w == 0 || h == 0 {
                    return None;
                }
                let texels: Vec<[u8; 4]> =
                    read_pod_vec_checked(&mut r, n as usize, &mut remaining)?;
                tex_recs.push(TexRecord::Inline { w, h, masked, srgb, h2n, n2h, source, texels });
            }
            _ => return None,
        }
    }

    let disk_nodes: Vec<DiskNode> = read_pod_vec_checked(&mut r, n_nodes, &mut remaining)?;
    let tri_idx: Vec<u32> = read_pod_vec_checked(&mut r, n_tri_idx, &mut remaining)?;

    if !validate_links(
        &indices, &tri_mat, &tri_idx, &disk_nodes, &disk_mats, n_verts, n_tris, n_mats, n_tex,
        n_nodes, n_tri_idx,
    ) {
        return None;
    }

    let materials: Vec<Material> = disk_mats.iter().map(mat_from_disk).collect();
    let textures = decode_tex_records(tex_recs);
    let nodes = nodes_from_disk(&disk_nodes);

    let mut sc = Scene {
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
        sun: crate::sky::Sun::new(sun_v[0]),
        // Derived-only fields, exactly the per-scene rule (see try_load).
        sky_sh: crate::sh::Sh9::ZERO,
        sky_scale: 1.0,
        night: 0.0,
        diag: 0.0,
        eps: 0.0,
        ao_radius: 0.0,
        content_min: glam::Vec3A::ZERO,
        content_max: glam::Vec3A::ZERO,
    };
    scene::finalize_scalars(&mut sc);
    eprintln!(
        "world cache: hit — {} islands, {:.1}M tris, {} nodes, {} textures in {:.0} ms",
        islands.len(),
        sc.tri_count() as f64 / 1e6,
        nodes.len(),
        sc.textures.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Some((
        sc,
        Bvh::from_parts(nodes, tri_idx),
        crate::world::World { islands, field_half },
    ))
}

/// Best-effort store of the merged world (temp file + rename, the per-scene
/// shape). A failure prints one line and the run continues uncached.
pub fn store_world(
    path: &Path,
    key: &WorldKey,
    scene: &Scene,
    bvh: &Bvh,
    world: &crate::world::World,
) {
    let tmp = path.with_extension(format!("fcache.{}.tmp", std::process::id()));
    let result = (|| -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut w = BufWriter::new(std::fs::File::create(&tmp)?);
        w.write_all(&WORLD_MAGIC)?;
        write_u32(&mut w, WORLD_VERSION)?;
        write_u32(&mut w, CACHE_VERSION)?;
        write_u64(&mut w, crate::bvh::build_key())?;
        write_u32(&mut w, lever_word())?;
        write_u64(&mut w, scene.positions.len() as u64)?;
        write_u64(&mut w, scene.indices.len() as u64)?;
        write_u64(&mut w, scene.materials.len() as u64)?;
        write_u64(&mut w, scene.textures.len() as u64)?;
        write_u64(&mut w, bvh.nodes.len() as u64)?;
        write_u64(&mut w, bvh.tri_idx.len() as u64)?;
        let blob = world_key_blob(key);
        write_u32(&mut w, blob.len() as u32)?;
        w.write_all(&blob)?;

        write_u32(&mut w, world.islands.len() as u32)?;
        for isle in &world.islands {
            write_u32(&mut w, isle.name.len() as u32)?;
            w.write_all(isle.name.as_bytes())?;
            write_pod(&mut w, &[isle.center])?;
            write_f32(&mut w, isle.radius)?;
            write_f32(&mut w, isle.height)?;
            write_f32(&mut w, isle.theme_hour)?;
        }
        write_f32(&mut w, world.field_half)?;
        write_pod(&mut w, &[scene.sun.dir])?;

        write_pod(&mut w, &scene.positions)?;
        write_pod(&mut w, &scene.normals)?;
        write_pod(&mut w, &scene.texcoords)?;
        write_pod(&mut w, &scene.indices)?;
        write_pod(&mut w, &scene.tri_mat)?;
        let disk_mats: Vec<DiskMat> = scene.materials.iter().map(mat_to_disk).collect();
        write_pod(&mut w, &disk_mats)?;
        for t in &scene.textures {
            write_u32(&mut w, t.source.len() as u32)?;
            w.write_all(t.source.as_bytes())?;
            // A source with no decodable file path (glTF's synthetic ids,
            // empty test sources) stores its POST-conversion texels inline —
            // there is nothing on disk to re-decode them from.
            let inline = t.source.starts_with("gltf:") || t.source.is_empty();
            w.write_all(&[
                t.alpha_masked as u8,
                t.srgb as u8,
                t.h2n as u8,
                t.n2h as u8,
                inline as u8,
            ])?;
            if inline {
                write_u32(&mut w, t.w)?;
                write_u32(&mut w, t.h)?;
                write_pod(&mut w, &t.texels)?;
            } else {
                let (t_size, t_mtime) = stat_key(Path::new(&t.source));
                write_u64(&mut w, t_size)?;
                write_u64(&mut w, t_mtime)?;
            }
        }
        write_pod(&mut w, &nodes_to_disk(bvh))?;
        write_pod(&mut w, &bvh.tri_idx)?;
        w.flush()?;
        drop(w);
        std::fs::rename(&tmp, path)
    })();
    match result {
        Ok(()) => {
            let gb = std::fs::metadata(path).map_or(0.0, |m| m.len() as f64 / (1 << 30) as f64);
            eprintln!("world cache: wrote {} ({gb:.2} GB)", path.display());
        }
        Err(e) => {
            eprintln!("world cache: write failed ({e}) — continuing uncached");
            let _ = std::fs::remove_file(&tmp);
        }
    }
}
