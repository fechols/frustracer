//! Frozen frustum snapshot (the Y/Z keys): freeze the current view's terminal
//! quadtree as real, self-lit scene geometry so the user can fly around and see
//! what the projected tile frustums actually look like.
//!
//! One filled **near-plane quad** per leaf tile, placed at that tile's
//! proven-empty inherited `t_start` distance along the tile's four corner rays
//! (rebuilt exactly as `CamBasis::tile_frustum` does — `ray_dir` at the
//! continuous pixel-grid corners) and colored by quadtree depth with the debug
//! overlay's heat ramp. The result is a faceted "stepped depth surface"
//! floating at the capture viewpoint.
//!
//! The quads are emitted as EMISSIVE triangles (`Material::emissive`): additive,
//! self-lit, appear in reflections/through glass, and light nothing else (the
//! stars/fireflies gather-exclusion class in shade.rs), so no shadow rays and no
//! perturbation of the scene's own lighting. Because it is real geometry it
//! depth-occludes against the scene in every render mode once the BVH (and, on
//! the GPU arms, the acceleration structure) is rebuilt — the caller owns that.
//!
//! Zero rng draws, zero effect on any derived scene flag (emissive geometry is
//! not alpha/height/transmissive), so the caller deliberately skips
//! `finalize_scalars` and keeps `diag`/`eps`/`ao_radius` pinned.

use crate::overlay;
use crate::replay::ReplayCache;
use crate::scene::{MatKind, Material, Scene, NO_TEX};
use glam::{Vec2, Vec3A};

/// Emissive radiance scale applied to the display-space heat ramp so the quads
/// read bright through the tonemap (`1 - exp(-x)` ⇒ ~2 is near-white) while
/// keeping the depth hue. Tunable — the one look knob.
const FRUST_EMISSIVE: f32 = 2.0;

/// How much geometry a snapshot appended to the `Scene`, all as a contiguous
/// tail of each vector — so a recapture/clear truncates it right back off.
/// Lives in `Persist` (Copy) to survive a window-resize session re-entry.
#[derive(Clone, Copy)]
pub struct FrustArtifact {
    pub verts: u32,
    pub tris: u32,
    pub mats: u32,
}

/// A self-lit material for depth `d`'s near quads (near-black albedo so the
/// emissive term dominates; every map slot NO_TEX so it stays bit-cheap and
/// flips no scene flag).
fn emissive_material(depth: u32) -> Material {
    Material {
        albedo: Vec3A::splat(0.02),
        roughness: 1.0,
        metallic: 0.0,
        anisotropy: 0.0,
        sheen: 0.0,
        translucency: 0.0,
        transmission: 0.0,
        trans_tint: Vec3A::splat(-1.0),
        ior: 1.5,
        ripple_amp: 0.0,
        emissive: overlay::heat(depth) * FRUST_EMISSIVE,
        normal_tex: NO_TEX,
        normal_scale: 1.0,
        height_amp: 0.0,
        rough_tex: NO_TEX,
        metal_tex: NO_TEX,
        emissive_tex: NO_TEX,
        kind: MatKind::Diffuse,
    }
}

/// Append one flat triangle (vertices duplicated, the `SceneBuilder::tri` idiom).
fn push_tri(scene: &mut Scene, p: [Vec3A; 3], mat: u32) {
    let base = scene.positions.len() as u32;
    let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
    scene.positions.extend_from_slice(&p);
    scene.normals.extend_from_slice(&[n; 3]);
    scene.texcoords.extend_from_slice(&[Vec2::ZERO; 3]);
    scene.indices.push([base, base + 1, base + 2]);
    scene.tri_mat.push(mat);
}

/// Build the near-quad wireframe for every recorded leaf tile and append it to
/// `scene`. `cam` and `cache` must come from the SAME capture pose/resolution.
/// Returns the appended counts. The caller rebuilds the BVH afterward.
pub fn build(scene: &mut Scene, cam: &crate::camera::CamBasis, cache: &ReplayCache, n_leaves: u32) -> FrustArtifact {
    let v0 = scene.positions.len();
    let t0 = scene.indices.len();
    let m0 = scene.materials.len();
    // Skip degenerate near quads (a tile whose t_start ≈ 0 collapses to the
    // apex): scale-relative floor.
    let min_t = (scene.diag * 1e-3).max(1e-4);
    let o = cam.origin;
    // One emissive material per depth level, appended on first use. Quadtree
    // depth never exceeds the traversal stack (96); 32 covers every real view.
    let mut depth_mat: [Option<u32>; 32] = [None; 32];
    for i in 0..n_leaves as usize {
        let lr = cache.leaf(i);
        let t = lr.t_start;
        if t <= min_t {
            continue;
        }
        // The tile's four corner rays through its continuous pixel-grid edges,
        // TL -> TR -> BR -> BL (CamBasis::tile_frustum's order), at t_start.
        let p = [
            o + cam.ray_dir(lr.x0 as f32, lr.y0 as f32) * t,
            o + cam.ray_dir(lr.x1 as f32, lr.y0 as f32) * t,
            o + cam.ray_dir(lr.x1 as f32, lr.y1 as f32) * t,
            o + cam.ray_dir(lr.x0 as f32, lr.y1 as f32) * t,
        ];
        let di = (lr.depth as usize).min(depth_mat.len() - 1);
        let mat = match depth_mat[di] {
            Some(m) => m,
            None => {
                let m = scene.materials.len() as u32;
                scene.materials.push(emissive_material(lr.depth));
                depth_mat[di] = Some(m);
                m
            }
        };
        push_tri(scene, [p[0], p[1], p[2]], mat);
        push_tri(scene, [p[0], p[2], p[3]], mat);
    }
    FrustArtifact {
        verts: (scene.positions.len() - v0) as u32,
        tris: (scene.indices.len() - t0) as u32,
        mats: (scene.materials.len() - m0) as u32,
    }
}

/// Truncate a previously-appended artifact off the tail of every `Scene`
/// vector, returning it to its base geometry. The caller rebuilds the BVH.
pub fn clear(scene: &mut Scene, a: FrustArtifact) {
    let v = scene.positions.len() - a.verts as usize;
    scene.positions.truncate(v);
    scene.normals.truncate(v);
    scene.texcoords.truncate(v);
    let t = scene.indices.len() - a.tris as usize;
    scene.indices.truncate(t);
    scene.tri_mat.truncate(t);
    let m = scene.materials.len() - a.mats as usize;
    scene.materials.truncate(m);
}
