//! The scene's GPU wire formats — the per-material streams both backends
//! upload, packed once.
//!
//! These are `gfx` items for the reason every other row here is: `GpuMat` is a
//! `#[repr(C)]` mirror of `shade.hlsli`'s `Mat`, so a stride skew reads
//! garbage, and a SECOND copy of that mirror in `src/vk/` would be a second
//! thing to keep in step with the HLSL — the exact failure mode this module
//! exists to prevent. Nothing here names a backend type; the packing is
//! `Scene` in, bytes out, and each backend does its own uploading.
//!
//! The three tiny maps beside it (`mat_cutout`, `mat_height`, `mat_shadow`)
//! move for the same reason and carry a stronger one: each mirrors a PREDICATE
//! that also lives on the CPU side (`bvh.rs`'s cutout and `tri_height_depth`
//! gates, `Material::shadow_tint`), and `bc7::should_compress`'s agreement
//! with `mat_cutout` is load-bearing soundness rather than tidiness — see
//! `src/bc7.rs`. One statement of each, read by both backends.

use crate::scene::{MatKind, Scene};

/// The scene AABB the slab-space cloud-shadow grid spans: the content box
/// unioned with the ground quad's first few vertices (the ground is what a low
/// sun's shadow footprint actually covers).
pub fn shadow_aabb(scene: &Scene) -> ([f32; 3], [f32; 3]) {
    let (mut amn, mut amx) = (scene.content_min, scene.content_max);
    for p in scene.positions.iter().take(6) {
        amn = amn.min(*p);
        amx = amx.max(*p);
    }
    ([amn.x, amn.y, amn.z], [amx.x, amx.y, amx.z])
}

/// How many bytes the geometry/material stream set below occupies on the GPU
/// — what an upload ring is sized against.
///
/// Here rather than in either backend because it is a pure function of the
/// `Scene` and of the wire formats stated in THIS file: `GpuMat`'s size is
/// part of the answer, so a copy living next to one backend's ring would go
/// stale the next time the material struct grows. Textures are deliberately
/// excluded — they are per-image and each backend's own staging shape decides
/// what indivisible piece it must additionally be able to hold.
pub fn stream_bytes(scene: &Scene) -> usize {
    let v = scene.positions.len();
    let t = scene.indices.len();
    let m = scene.materials.len();
    v * (12 + 12 + 8) + t * (12 + 4) + m * (std::mem::size_of::<GpuMat>() + 4)
}

/// `bvh.rs::Node` packed for `StructuredBuffer<BvhNode>` (frustum.hlsli).
///
/// The frustum ladder's own tree — the software half of the tracer, since RT
/// cores cannot answer a frustum query. Here for `GpuMat`'s reason: it is a
/// `#[repr(C)]` mirror of an HLSL struct, and one mirror per HLSL struct is
/// the rule. Note the field ORDER carries meaning the names do not — the
/// `left_first`/`count` pair rides in the two float3 tails' padding lanes, so
/// a reordering that looks cosmetic changes the stride.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuBvhNode {
    pub mn: [f32; 3],
    pub left_first: u32,
    pub mx: [f32; 3],
    pub count: u32,
}

/// One node in that wire format. PER NODE, not per tree, deliberately: D3D12
/// streams the array through a staging ring (a materialized `Vec` is hundreds
/// of megabytes at 100M triangles) while Vulkan has no ring yet and collects
/// one, so the shared thing is the FIELD MAPPING — which is what a stride skew
/// would break — and each backend keeps its own iteration.
pub fn gpu_bvh_node(n: &crate::bvh::BvhNode) -> GpuBvhNode {
    GpuBvhNode {
        mn: [n.aabb.min.x, n.aabb.min.y, n.aabb.min.z],
        left_first: n.left_first,
        mx: [n.aabb.max.x, n.aabb.max.y, n.aabb.max.z],
        count: n.count,
    }
}

/// `scene.rs::Material` packed for `StructuredBuffer<Mat>` (shade.hlsli).
/// 108 B — the HLSL `Mat` mirrors this field-for-field; a stride skew reads
/// garbage, so the two must move in the same commit.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuMat {
    albedo: [f32; 3],
    roughness: f32,
    metallic: f32,
    anisotropy: f32,
    kind: u32, // 0 = diffuse, 1 = marble, 2 = textured
    scale: f32,
    sheen: f32,
    translucency: f32,
    transmission: f32,
    /// `Scene::textures` index for MAT_TEXTURED (the space1 `texs[]` slot).
    tex: u32,
    emissive: [f32; 3],
    normal_tex: u32, // NO_TEX sentinel = HLSL TEX_NONE
    rough_tex: u32,
    metal_tex: u32,
    emissive_tex: u32,
    normal_scale: f32,
    trans_tint: [f32; 3], // transmission/absorption tint; .x < 0 = "use albedo"
    ior: f32,             // Snell/Fresnel IOR (default 1.5; water 1.33)
    ripple_amp: f32,      // water ripple slope amplitude (0 = none)
    // Per-material world-space detail texel scale (Scene::detail_scales —
    // never per-face, which seams on greedy-meshed atlases). 0 = field off.
    detail_scale: f32,
    // Spec-AA: the normal map's slope-variance companion texture
    // (Scene::tex_var; TEX_NONE = none — every unmapped/lever-off material,
    // the fold's structural map-arm off state).
    normal_var_tex: u32,
}

/// Every material packed for `t6`.
pub fn gpu_materials(scene: &Scene) -> Vec<GpuMat> {
    scene
        .materials
        .iter()
        .enumerate()
        .map(|(mi, m)| GpuMat {
            albedo: [m.albedo.x, m.albedo.y, m.albedo.z],
            roughness: m.roughness,
            metallic: m.metallic,
            anisotropy: m.anisotropy,
            kind: match m.kind {
                MatKind::Diffuse => 0,
                MatKind::Marble { .. } => 1,
                MatKind::Textured { .. } => 2,
            },
            scale: match m.kind {
                MatKind::Marble { scale } => scale,
                _ => 0.0,
            },
            sheen: m.sheen,
            translucency: m.translucency,
            transmission: m.transmission,
            tex: match m.kind {
                MatKind::Textured { tex } => tex,
                _ => 0,
            },
            emissive: [m.emissive.x, m.emissive.y, m.emissive.z],
            normal_tex: m.normal_tex,
            rough_tex: m.rough_tex,
            metal_tex: m.metal_tex,
            emissive_tex: m.emissive_tex,
            normal_scale: m.normal_scale,
            trans_tint: [m.trans_tint.x, m.trans_tint.y, m.trans_tint.z],
            ior: m.ior,
            ripple_amp: m.ripple_amp,
            detail_scale: scene.detail_scales.get(mi).copied().unwrap_or(0.0),
            normal_var_tex: scene
                .tex_var
                .get(m.normal_tex as usize)
                .copied()
                .unwrap_or(crate::scene::NO_TEX),
        })
        .collect()
}

/// Per-material cutout map the `alpha_cutout` helper consumes: `tex + 1` when
/// the material is textured AND its texture has masked texels, else 0.
///
/// MIRRORS `bvh.rs`'s cutout gates, and its nonzero set must equal the one
/// `bc7::should_compress` refuses to compress — that agreement is what keeps
/// every texture the intersector can `.Load` an exact RGBA8 one.
pub fn mat_cutout(scene: &Scene) -> Vec<u32> {
    scene
        .materials
        .iter()
        .map(|m| match m.kind {
            MatKind::Textured { tex } if scene.textures[tex as usize].alpha_masked => tex + 1,
            _ => 0,
        })
        .collect()
}

/// Per-material relief map the `height_march` helper consumes: normal tex + 1
/// where the material carries a heightfield, else 0, plus the amp in texel
/// widths. The nonzero set is exactly the h2n/n2h texture set.
pub fn mat_height(scene: &Scene) -> Vec<[u32; 2]> {
    scene
        .materials
        .iter()
        .map(|m| {
            if m.height_amp > 0.0 && m.normal_tex != crate::scene::NO_TEX {
                [m.normal_tex + 1, m.height_amp.to_bits()]
            } else {
                [0, 0]
            }
        })
        .collect()
}

/// Per-material tinted-shadow data `transmit_q` consumes: rgb = the interface
/// tint (`Material::shadow_tint` — the ONE tint source, so CPU↔GPU agreement
/// is by data), a = transmission (a == 0 ⇒ opaque).
pub fn mat_shadow(scene: &Scene) -> Vec<[f32; 4]> {
    scene
        .materials
        .iter()
        .map(|m| {
            let t = m.shadow_tint();
            [t.x, t.y, t.z, m.transmission]
        })
        .collect()
}
