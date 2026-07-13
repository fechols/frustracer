//! Binary sidecar cache of a parsed `Scene` + built `Bvh` (`<source>.fcache`,
//! next to the OBJ/`.obj.zst` it came from). Manual POD format, std only — no
//! serde, no mmap: the read path preallocates each typed Vec exactly and
//! `read_exact`s into its byte view, so a San Miguel load drops from ~a minute
//! of parse + normal-gen + BVH build to under a second. Keyed on the RESOLVED
//! source file's size + mtime (plus the `.mtl` sibling's and each TEXTURE
//! file's — `alpha_masked` and the height-map skip are texture-CONTENT
//! decisions, so an edited texture must miss the whole cache, not resurface
//! stale flags) and `CACHE_VERSION`; any mismatch — including a truncated or
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
//! Bump `CACHE_VERSION` on ANY change to: the Scene/Material/Texture layout,
//! the Bvh build (node order is part of the contract), the OBJ loader, or
//! matclass classification.

use crate::bvh::{Bvh, BvhNode};
use crate::scene::{self, AreaLight, MatKind, Material, Scene};
use crate::texture::Texture;
use glam::{Vec2, Vec3A};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub const CACHE_VERSION: u32 = 4; // v4: .webp sibling texture resolution
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
    normal_tex: u32,
    rough_tex: u32,
    metal_tex: u32,
    emissive_tex: u32,
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
fn mtl_sibling(src: &Path) -> PathBuf {
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

/// Load the sidecar for an already-RESOLVED source path (see
/// `scene::resolve_scene_path`). Silent None on any mismatch or read error.
pub fn try_load(src_path: &str) -> Option<(Scene, Bvh)> {
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
    let light_v: Vec<Vec3A> = read_pod_vec_checked(&mut r, 4, &mut remaining)?;

    let positions: Vec<Vec3A> = read_pod_vec_checked(&mut r, n_verts, &mut remaining)?;
    let normals: Vec<Vec3A> = read_pod_vec_checked(&mut r, n_verts, &mut remaining)?;
    let texcoords: Vec<Vec2> = read_pod_vec_checked(&mut r, n_verts, &mut remaining)?;
    let indices: Vec<[u32; 3]> = read_pod_vec_checked(&mut r, n_tris, &mut remaining)?;
    let tri_mat: Vec<u32> = read_pod_vec_checked(&mut r, n_tris, &mut remaining)?;
    let disk_mats: Vec<DiskMat> = read_pod_vec_checked(&mut r, n_mats, &mut remaining)?;

    let mut tex_meta: Vec<(String, bool, bool)> = Vec::with_capacity(n_tex.min(4096));
    for _ in 0..n_tex {
        let len = read_u32(&mut r).ok()? as usize;
        if len as u64 > remaining {
            return None;
        }
        remaining -= len as u64;
        let mut path = vec![0u8; len];
        r.read_exact(&mut path).ok()?;
        let mut flags = [0u8; 2];
        r.read_exact(&mut flags).ok()?;
        let path = String::from_utf8(path).ok()?;
        // Texture-content staleness: alpha_masked (restored verbatim below)
        // and the loader's height-map skip are functions of the FILE's
        // pixels — an edited/replaced texture must miss the whole cache.
        let (t_size, t_mtime) = (read_u64(&mut r).ok()?, read_u64(&mut r).ok()?);
        if stat_key(Path::new(&path)) != (t_size, t_mtime) {
            return None;
        }
        tex_meta.push((path, flags[0] != 0, flags[1] != 0));
    }

    let disk_nodes: Vec<DiskNode> = read_pod_vec_checked(&mut r, n_nodes, &mut remaining)?;
    let tri_idx: Vec<u32> = read_pod_vec_checked(&mut r, n_tri_idx, &mut remaining)?;

    // Right-sized corruption: validate every cross-array link before the
    // payload reaches code that indexes unchecked (Bvh traversal, shade's
    // texture fetches) — a bad sidecar must be a miss, not a later panic.
    {
        use rayon::prelude::*;
        let tex_ok =
            |t: u32| t == scene::NO_TEX || (t as usize) < n_tex;
        let links_ok = indices
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
            });
        if !links_ok {
            return None;
        }
    }

    let materials: Vec<Material> = disk_mats
        .iter()
        .map(|m| Material {
            albedo: Vec3A::from_array(m.albedo),
            roughness: m.roughness,
            metallic: m.metallic,
            anisotropy: m.anisotropy,
            sheen: m.sheen,
            translucency: m.translucency,
            transmission: m.transmission,
            emissive: Vec3A::from_array(m.emissive),
            normal_tex: m.normal_tex,
            normal_scale: m.normal_scale,
            rough_tex: m.rough_tex,
            metal_tex: m.metal_tex,
            emissive_tex: m.emissive_tex,
            kind: match m.kind {
                1 => MatKind::Marble { scale: f32::from_bits(m.param) },
                2 => MatKind::Textured { tex: m.param },
                _ => MatKind::Diffuse,
            },
        })
        .collect();

    // Re-decode the texture list IN ID ORDER (par map preserves order); a
    // failure keeps the slot with a 1×1 white texture carrying the cached
    // alpha_masked so material tex ids never shift.
    let textures: Vec<Texture> = {
        use rayon::prelude::*;
        // Largest-first (LPT) scheduling via an index permutation, scattered
        // back so texture ids never shift (see the scene.rs decode note —
        // WebP's slower per-file decode makes the big-file tail dominate).
        let mut order: Vec<usize> = (0..tex_meta.len()).collect();
        order.sort_by_key(|&i| {
            std::cmp::Reverse(
                std::fs::metadata(&tex_meta[i].0).map_or(0, |m| m.len()),
            )
        });
        let mut pairs: Vec<(usize, Texture)> = order
            .par_iter()
            .with_max_len(1)
            .map(|&i| (i, &tex_meta[i]))
            .map(|(i, (path, masked, srgb))| (i, match image::open(path) {
                Ok(img) => {
                    let mut t = Texture::from_image(img, *srgb);
                    // The loader's role-gating (e.g. an emissive-only color
                    // map never arms the cutout) is baked into the cached
                    // flag — restore it verbatim rather than re-deriving.
                    // Sound because the per-texture stat key above already
                    // missed the cache if the file's content changed.
                    t.alpha_masked = *masked;
                    t.source = path.clone();
                    t
                }
                Err(e) => {
                    eprintln!(
                        "warning: cached texture '{path}' failed to re-decode ({e}); using 1x1 white"
                    );
                    Texture {
                        w: 1,
                        h: 1,
                        texels: vec![[255; 4]],
                        alpha_masked: *masked,
                        srgb: *srgb,
                        source: path.clone(),
                    }
                }
            }))
            .collect();
        pairs.sort_unstable_by_key(|&(i, _)| i);
        pairs.into_iter().map(|(_, t)| t).collect()
    };

    let nodes: Vec<BvhNode> = disk_nodes
        .iter()
        .map(|n| BvhNode {
            aabb: crate::bvh::Aabb {
                min: Vec3A::from_array(n.min),
                max: Vec3A::from_array(n.max),
            },
            left_first: n.left_first,
            count: n.count,
        })
        .collect();

    let mut sc = Scene {
        positions,
        normals,
        texcoords,
        indices,
        tri_mat,
        materials,
        textures,
        any_alpha: false,
        light: AreaLight {
            center: light_v[0],
            u: light_v[1],
            v: light_v[2],
            color: light_v[3],
        },
        diag: 0.0,
        eps: 0.0,
        ao_radius: 0.0,
    };
    scene::finalize_scalars(&mut sc);
    eprintln!(
        "scene cache: hit — {} tris, {} nodes, {} textures in {:.0} ms",
        sc.tri_count(),
        nodes.len(),
        sc.textures.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Some((sc, Bvh { nodes, tri_idx }))
}

/// Best-effort store (temp file + rename; a failure prints one line and the
/// run continues uncached). `src_path` must be the RESOLVED source.
pub fn store(src_path: &str, scene: &Scene, bvh: &Bvh) {
    let src = Path::new(src_path);
    let dst = sidecar(src);
    // Pid-suffixed so two processes loading the same scene can't interleave
    // writes into one tmp file (the rename itself is atomic either way).
    let tmp = dst.with_extension(format!("fcache.{}.tmp", std::process::id()));
    let result = (|| -> std::io::Result<()> {
        let mut w = BufWriter::new(std::fs::File::create(&tmp)?);
        w.write_all(&MAGIC)?;
        write_u32(&mut w, CACHE_VERSION)?;
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
        write_pod(&mut w, &[scene.light.center, scene.light.u, scene.light.v, scene.light.color])?;

        write_pod(&mut w, &scene.positions)?;
        write_pod(&mut w, &scene.normals)?;
        write_pod(&mut w, &scene.texcoords)?;
        write_pod(&mut w, &scene.indices)?;
        write_pod(&mut w, &scene.tri_mat)?;
        let disk_mats: Vec<DiskMat> = scene
            .materials
            .iter()
            .map(|m| {
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
                    normal_tex: m.normal_tex,
                    rough_tex: m.rough_tex,
                    metal_tex: m.metal_tex,
                    emissive_tex: m.emissive_tex,
                }
            })
            .collect();
        write_pod(&mut w, &disk_mats)?;
        for t in &scene.textures {
            write_u32(&mut w, t.source.len() as u32)?;
            w.write_all(t.source.as_bytes())?;
            w.write_all(&[t.alpha_masked as u8, t.srgb as u8])?;
            // Per-texture staleness key (see try_load): content edits to a
            // texture must invalidate the cached alpha_masked/height-map
            // decisions, which only a full reparse re-derives.
            let (t_size, t_mtime) = stat_key(Path::new(&t.source));
            write_u64(&mut w, t_size)?;
            write_u64(&mut w, t_mtime)?;
        }
        let disk_nodes: Vec<DiskNode> = bvh
            .nodes
            .iter()
            .map(|n| DiskNode {
                min: n.aabb.min.to_array(),
                max: n.aabb.max.to_array(),
                left_first: n.left_first,
                count: n.count,
            })
            .collect();
        write_pod(&mut w, &disk_nodes)?;
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
