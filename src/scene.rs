use crate::texture::Texture;
use glam::{Vec2, Vec3A};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

/// Tinted-shadows switch, settable ONCE before scene load
/// (`--no-tinted-shadows`) — the `texture::set_mips` "knob before build"
/// family. Off = `finalize_scalars` never arms `Scene::any_transmissive`, so
/// every occlusion query runs the pre-feature binary arm bit-identically.
static TINTED_SHADOWS: AtomicBool = AtomicBool::new(true);

pub fn set_tinted_shadows(on: bool) {
    TINTED_SHADOWS.store(on, Relaxed);
}

pub fn tinted_shadows() -> bool {
    TINTED_SHADOWS.load(Relaxed)
}

/// Spray-reclassification switch (`--no-spray`), same knob-before-load
/// family. Off = `reclassify_spray` is a no-op — transmissive droplets stay
/// clear glass (the pre-spray behavior). Keys the scene cache's lever word:
/// the pass rewrites cached materials/tri_mat.
static SPRAY: AtomicBool = AtomicBool::new(true);

pub fn set_spray(on: bool) {
    SPRAY.store(on, Relaxed);
}

pub fn spray_enabled() -> bool {
    SPRAY.load(Relaxed)
}

/// Depth-tint switch (`--no-depth-tint`): Beer–Lambert attenuation over the
/// transmission chain's interior segments. Runtime shading lever (no scene
/// data changes — reads live in shade.rs / the GPU FLAG_DEPTH_TINT bit).
static DEPTH_TINT: AtomicBool = AtomicBool::new(true);

pub fn set_depth_tint(on: bool) {
    DEPTH_TINT.store(on, Relaxed);
}

pub fn depth_tint() -> bool {
    DEPTH_TINT.load(Relaxed)
}

/// How a material derives its albedo. Reflection behavior is fully described
/// by the metallic/roughness/anisotropy parameters (the old `Metal` variant is
/// subsumed by Fresnel: F0 = lerp(0.04, albedo, metallic)).
#[derive(Clone, Copy, PartialEq)]
pub enum MatKind {
    /// Constant albedo.
    Diffuse,
    /// Procedural marble: albedo from world-space fBm veining (`shade::marble`);
    /// `scale` is the feature frequency in world units.
    Marble { scale: f32 },
    /// Albedo sampled from `Scene::textures[tex]` at the hit's interpolated
    /// UV, on the CPU (`shade.rs`) and both GPU paths (`shade.hlsli` through
    /// the space1 texture table). The material's flat `albedo` stays as the
    /// untextured fallback.
    Textured { tex: u32 },
}

/// Sentinel for "no texture" in `Material`'s map-index fields — branching on
/// it is what keeps unmapped materials shading bit-identically to before the
/// map fields existed (the structural guarantee for procedural/stress
/// scenes).
pub const NO_TEX: u32 = u32::MAX;

/// Metallic/roughness PBR material (GGX microfacet; see `shade.rs`).
pub struct Material {
    pub albedo: Vec3A,
    /// Perceptual roughness; the GGX code squares it (α = roughness²).
    pub roughness: f32,
    /// 0 = dielectric (F0 = 0.04), 1 = metal (F0 = albedo, no diffuse).
    pub metallic: f32,
    /// 0 = isotropic; > 0 stretches the GGX lobe along the tangent
    /// (circumferential around world-up — a lathe-spun / brushed finish).
    pub anisotropy: f32,
    /// 0 = none; retro-reflective Charlie-sheen intensity (fabric/carpet).
    pub sheen: f32,
    /// 0 = opaque; thin-surface diffuse transmission fraction (foliage —
    /// back-lit leaves glow through).
    pub translucency: f32,
    /// 0 = opaque; thin-pane Fresnel-split transmission (glassware). The
    /// transmitted light is tinted by albedo — dark MTL glass Kd must be
    /// lifted toward white by the classifier or glass renders near-black.
    pub transmission: f32,
    /// Emitted radiance (Ke / glTF emissiveFactor). Added to color at every
    /// shading depth, OUTSIDE the kd·(1−transmission) factor; emitters do
    /// NOT light other surfaces — only the analytic sun + sky do (the
    /// stars/fireflies gather-exclusion class). Default ZERO.
    pub emissive: Vec3A,
    /// Tangent-space normal map (NO_TEX = none; linear data). Perturbs the
    /// SHADING normal only — the geometric normal keeps driving ray offsets,
    /// the translucency back ray, the hemi tier, and the glass chain (the
    /// n_g/n_s split in shade.rs).
    pub normal_tex: u32,
    /// map_Bump `-bm s` / glTF normalTexture.scale. Default 1.0.
    pub normal_scale: f32,
    /// Peak-to-peak height amplitude of `normal_tex`'s alpha-channel
    /// heightfield, in TEXEL widths (0.0 = no height data — the structural
    /// off state, like NO_TEX). Sobel-converted height maps carry exactly
    /// `texture::HEIGHT_NORMAL_STRENGTH` (the same K both modes describe);
    /// Poisson-derived heights carry the integration's own amplitude. The
    /// relief march converts to world units per hit via the UV basis.
    pub height_amp: f32,
    /// Roughness map (NO_TEX = none; samples .g — the glTF channel
    /// convention, which grayscale MTL maps satisfy via to_rgba8 gray
    /// replication). Effective roughness = `roughness` × sample: factor ×
    /// sample IS the glTF spec; with a map the flat factor comes from the
    /// MTL's own `Pr` scalar (default 1.0), bypassing the matclass constant,
    /// which stays as the no-map fallback.
    pub rough_tex: u32,
    /// Metallic map (samples .b); effective = `metallic` × sample.
    pub metal_tex: u32,
    /// Emissive map (sRGB); effective = `emissive` × sample (map present
    /// with Ke absent ⇒ factor 1.0, the map_Kd precedent).
    pub emissive_tex: u32,
    pub kind: MatKind,
}

impl Material {
    /// Per-interface throughput of a light-transport occlusion ray crossing
    /// this surface (the tinted-shadows feature): `transmission × albedo` —
    /// ZERO for opaque materials. The ONE tint source: `Bvh::transmittance`
    /// and the GPU `mat_shadow` buffer fill both read this, so CPU↔GPU
    /// agreement is by data (the fireflies-CB precedent). Deliberately the
    /// FLAT albedo — no UV/texture fetch in the occlusion inner loop
    /// (documented known-accept for textured transmissive materials). Glass
    /// Kd was already lifted toward white at load, so this composes.
    #[inline]
    pub fn shadow_tint(&self) -> Vec3A {
        self.albedo * self.transmission
    }

    /// Whether shading this material fetches ANY texture (albedo or one of
    /// the PBR maps). Untextured materials have nothing for the deferred
    /// material-sorted shading to make cache-coherent, so they shade inline.
    pub fn any_tex(&self) -> bool {
        matches!(self.kind, MatKind::Textured { .. })
            || self.normal_tex != NO_TEX
            || self.rough_tex != NO_TEX
            || self.metal_tex != NO_TEX
            || self.emissive_tex != NO_TEX
    }
}

// The rectangular AreaLight is gone. It was a 4x4 rect ~12 units away with a
// 1/d² falloff, and its GGX highlight was a mirror image of that rect — which
// is why the sun used to reflect as a SQUARE. The light is now `sky::Sun`: a
// disc at infinity, part of the one sky sphere. See src/sky.rs.

pub struct Scene {
    pub positions: Vec<Vec3A>,
    pub normals: Vec<Vec3A>,
    /// Per-vertex UVs, parallel to `positions` (zeros where a mesh has none —
    /// sound because the OBJ loader uses `single_index`, one unified stream).
    pub texcoords: Vec<Vec2>,
    pub indices: Vec<[u32; 3]>,
    pub tri_mat: Vec<u32>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    /// Any texture is alpha-masked — the intersector's one-bool gate for the
    /// alpha-cutout path (false on the procedural/stress scenes, keeping the
    /// hot loop untouched there).
    pub any_alpha: bool,
    /// Any material carries a heightfield (`height_amp` > 0 with a normal
    /// map) — the intersector's one-bool gate for the relief march, the
    /// `any_alpha` pattern (false on procedural/stress scenes). Derived by
    /// `finalize_scalars`.
    pub any_height: bool,
    /// Any material is transmissive (`transmission` > 0) AND the tinted-
    /// shadows lever is armed — the one-bool gate for `Bvh::transmittance`'s
    /// tinted arm (the `any_alpha` pattern: false on procedural/stress scenes
    /// and under `--no-tinted-shadows`, keeping those paths bit-identical).
    /// Derived by `finalize_scalars`.
    pub any_transmissive: bool,
    /// The sun: a disc at infinity, the sharp half of the one sky.
    pub sun: crate::sky::Sun,
    /// The sky's smooth dome, projected into order-2 SH once at load — the
    /// analytic replacement for the old flat `shade::AMBIENT` constant, giving
    /// every normal its own sky irradiance for free (zero rays). Derived
    /// (`finalize_scalars`), never serialized: a pure function of the sun.
    pub sky_sh: crate::sh::Sh9,
    /// Time-of-day dome brightness (`sky::dome`'s `scale`): exactly 1.0 in an
    /// untouched session (`* 1.0` is bit-preserving), falling through dusk to
    /// the `MOON_DOME_FRAC` moonlight floor. Derived, never serialized — only
    /// `apply_tod` moves it.
    pub sky_scale: f32,
    /// Star visibility (`sky::stars`' gate): exactly 0.0 in an untouched
    /// session, ramping to 1.0 after sunset. Derived, never serialized.
    pub night: f32,
    /// Bounding diagonal — the scale reference for all epsilons.
    pub diag: f32,
    /// Self-intersection offset for secondary rays.
    pub eps: f32,
    pub ao_radius: f32,
    /// CONTENT bounds: the geometry EXCLUDING the standard ground quad (the
    /// first `GROUND_VERTS` positions every loader pushes) — where the models
    /// actually are. `diag` is ground-quad-dominated on the procedural/stress
    /// scenes (a ±60 ground makes it ~17× the content scale), so anything
    /// that should live AMONG the content (the fireflies' placement box)
    /// anchors here instead. Falls back to the full AABB when the skip would
    /// be degenerate. Derived (`finalize_scalars`), never serialized.
    pub content_min: Vec3A,
    pub content_max: Vec3A,
}

impl Scene {
    pub fn tri_count(&self) -> usize {
        self.indices.len()
    }

    /// Interpolate triangle `tri`'s UV at barycentrics (u, v) — hit.u/hit.v
    /// from the intersector. Lives here (not shade.rs) so the BVH's
    /// alpha-cutout test can share it without a bvh → shade dependency.
    #[inline]
    pub fn tri_uv(&self, tri: u32, u: f32, v: f32) -> Vec2 {
        let [i0, i1, i2] = self.indices[tri as usize];
        let w = 1.0 - u - v;
        self.texcoords[i0 as usize] * w
            + self.texcoords[i1 as usize] * u
            + self.texcoords[i2 as usize] * v
    }
}

pub struct SceneBuilder {
    positions: Vec<Vec3A>,
    normals: Vec<Vec3A>,
    texcoords: Vec<Vec2>,
    indices: Vec<[u32; 3]>,
    tri_mat: Vec<u32>,
    materials: Vec<Material>,
    textures: Vec<Texture>,
}

impl SceneBuilder {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            texcoords: Vec::new(),
            indices: Vec::new(),
            tri_mat: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
        }
    }

    pub fn add_texture(&mut self, tex: Texture) -> u32 {
        self.textures.push(tex);
        (self.textures.len() - 1) as u32
    }

    pub fn material(&mut self, albedo: Vec3A, roughness: f32, metallic: f32) -> u32 {
        self.material_kind(albedo, roughness, metallic, 0.0, MatKind::Diffuse)
    }

    pub fn material_kind(
        &mut self,
        albedo: Vec3A,
        roughness: f32,
        metallic: f32,
        anisotropy: f32,
        kind: MatKind,
    ) -> u32 {
        self.material_full(Material {
            albedo,
            roughness,
            metallic,
            anisotropy,
            sheen: 0.0,
            translucency: 0.0,
            transmission: 0.0,
            emissive: Vec3A::ZERO,
            normal_tex: NO_TEX,
            normal_scale: 1.0,
            height_amp: 0.0,
            rough_tex: NO_TEX,
            metal_tex: NO_TEX,
            emissive_tex: NO_TEX,
            kind,
        })
    }

    /// Full-control material push — the OBJ classifier's entry point (the
    /// shorthands above zero the new lobe fields, which is the structural
    /// guarantee that procedural/stress scenes never exercise them).
    pub fn material_full(&mut self, m: Material) -> u32 {
        self.materials.push(m);
        (self.materials.len() - 1) as u32
    }

    /// Push a triangle with per-vertex normals (vertices are duplicated, not shared).
    pub fn tri(&mut self, p: [Vec3A; 3], n: [Vec3A; 3], mat: u32) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&p);
        self.normals.extend_from_slice(&n);
        self.texcoords.extend_from_slice(&[Vec2::ZERO; 3]);
        self.indices.push([base, base + 1, base + 2]);
        self.tri_mat.push(mat);
    }

    /// Quad p0..p3 (fan-triangulated), flat-shaded with the face normal.
    pub fn quad(&mut self, p0: Vec3A, p1: Vec3A, p2: Vec3A, p3: Vec3A, mat: u32) {
        let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        self.tri([p0, p1, p2], [n; 3], mat);
        self.tri([p0, p2, p3], [n; 3], mat);
    }

    pub fn add_box(&mut self, c: Vec3A, half: Vec3A, mat: u32) {
        let (mn, mx) = (c - half, c + half);
        let v = |x: f32, y: f32, z: f32| Vec3A::new(x, y, z);
        // 8 corners
        let c000 = v(mn.x, mn.y, mn.z);
        let c100 = v(mx.x, mn.y, mn.z);
        let c010 = v(mn.x, mx.y, mn.z);
        let c110 = v(mx.x, mx.y, mn.z);
        let c001 = v(mn.x, mn.y, mx.z);
        let c101 = v(mx.x, mn.y, mx.z);
        let c011 = v(mn.x, mx.y, mx.z);
        let c111 = v(mx.x, mx.y, mx.z);
        self.quad(c010, c110, c111, c011, mat); // +y
        self.quad(c000, c001, c101, c100, mat); // -y
        self.quad(c100, c101, c111, c110, mat); // +x
        self.quad(c001, c000, c010, c011, mat); // -x
        self.quad(c101, c001, c011, c111, mat); // +z
        self.quad(c000, c100, c110, c010, mat); // -z
    }

    pub fn add_sphere(&mut self, c: Vec3A, r: f32, mat: u32, segs: u32, rings: u32) {
        use std::f32::consts::PI;
        let pt = |ring: u32, seg: u32| -> (Vec3A, Vec3A) {
            let theta = PI * ring as f32 / rings as f32;
            let phi = 2.0 * PI * seg as f32 / segs as f32;
            let n = Vec3A::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin());
            (c + n * r, n)
        };
        for ring in 0..rings {
            for seg in 0..segs {
                let (p00, n00) = pt(ring, seg);
                let (p10, n10) = pt(ring, seg + 1);
                let (p01, n01) = pt(ring + 1, seg);
                let (p11, n11) = pt(ring + 1, seg + 1);
                if ring > 0 {
                    self.tri([p00, p10, p11], [n00, n10, n11], mat);
                }
                if ring < rings - 1 {
                    self.tri([p00, p11, p01], [n00, n11, n01], mat);
                }
            }
        }
    }

    /// Push a shared-vertex mesh (used by the OBJ path). `normals` may be empty →
    /// smooth normals are computed from area-weighted face normals. `texcoords`
    /// may be empty → zeros (untextured mesh).
    pub fn add_mesh(
        &mut self,
        positions: Vec<Vec3A>,
        mut normals: Vec<Vec3A>,
        mut texcoords: Vec<Vec2>,
        indices: &[[u32; 3]],
        mat: u32,
    ) {
        if normals.len() != positions.len() {
            // Accumulate by *position*, not index: patch-tessellated meshes
            // (the Utah teapot) duplicate the vertices along patch borders,
            // and per-index averaging would leave a one-sided normal on each
            // side of every seam. Exact-bit keys suffice — duplicates come
            // from identical source text. (+0.0 normalized so -0.0 welds.)
            let key = |p: Vec3A| {
                let q = |f: f32| if f == 0.0 { 0u32 } else { f.to_bits() };
                [q(p.x), q(p.y), q(p.z)]
            };
            let mut acc: HashMap<[u32; 3], Vec3A> = HashMap::new();
            for tri in indices {
                let [a, b, c] = *tri;
                let (pa, pb, pc) = (
                    positions[a as usize],
                    positions[b as usize],
                    positions[c as usize],
                );
                let face_n = (pb - pa).cross(pc - pa); // area-weighted (unnormalized)
                for p in [pa, pb, pc] {
                    *acc.entry(key(p)).or_insert(Vec3A::ZERO) += face_n;
                }
            }
            // Unreferenced vertices (no triangle) fall back to zero — shade()
            // substitutes the face normal for zero normals anyway.
            normals = positions
                .iter()
                .map(|p| acc.get(&key(*p)).copied().unwrap_or(Vec3A::ZERO).normalize_or_zero())
                .collect();
        }
        if texcoords.len() != positions.len() {
            texcoords = vec![Vec2::ZERO; positions.len()];
        }
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&positions);
        self.normals.extend_from_slice(&normals);
        self.texcoords.extend_from_slice(&texcoords);
        for tri in indices {
            self.indices.push([tri[0] + base, tri[1] + base, tri[2] + base]);
            self.tri_mat.push(mat);
        }
    }

    pub fn finish(self, sun: crate::sky::Sun) -> Scene {
        let mut scene = Scene {
            positions: self.positions,
            normals: self.normals,
            texcoords: self.texcoords,
            indices: self.indices,
            tri_mat: self.tri_mat,
            materials: self.materials,
            textures: self.textures,
            any_alpha: false,
            any_height: false,
            any_transmissive: false,
            sun,
            sky_sh: crate::sh::Sh9::ZERO,
            sky_scale: 1.0,
            night: 0.0,
            diag: 0.0,
            eps: 0.0,
            ao_radius: 0.0,
            content_min: Vec3A::ZERO,
            content_max: Vec3A::ZERO,
        };
        finalize_scalars(&mut scene);
        scene
    }
}

/// Post-load heightfield derivation, shared by the OBJ and glTF loaders
/// (COLD loads only — run before `scene_cache::store`, so the per-texture
/// `n2h` flags and the materials' `height_amp` persist; warm loads re-apply
/// the texture conversions from the cached flags and take `height_amp` from
/// `DiskMat`). Every normal map that doesn't already carry a heightfield
/// (Sobel-converted maps carry the exact source height) gets one derived by
/// the Frankot–Chellappa solve (`Texture::apply_n2h`) into its alpha
/// channel; `height_amp` = the texture's own amplitude × the material's
/// `-bm`/`normalTexture.scale` factor, the same composition the decoded
/// slopes carry.
pub fn derive_heights(scene: &mut Scene) {
    use std::collections::HashSet;
    let n2h_on = crate::texture::n2h_enabled();
    let mut ids: Vec<u32> = scene
        .materials
        .iter()
        .filter_map(|m| (m.normal_tex != NO_TEX).then_some(m.normal_tex))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() {
        return; // procedural/stress scenes: structurally untouched
    }
    let solve: HashSet<u32> = ids
        .iter()
        .copied()
        .filter(|&id| {
            let t = &scene.textures[id as usize];
            n2h_on && !t.h2n && !t.n2h
        })
        .collect();
    let t0 = std::time::Instant::now();
    let amps: std::collections::HashMap<u32, f32> = {
        use rayon::prelude::*;
        // Bounded pool: one 4K solve holds ~0.5 GB of complex scratch, so
        // the full rayon width would spike memory on 4K-heavy scenes.
        let workers = rayon::current_num_threads().min(8);
        let mut run = || {
            scene
                .textures
                .par_iter_mut()
                .enumerate()
                .filter(|(i, _)| solve.contains(&(*i as u32)))
                .map(|(i, t)| (i as u32, t.apply_n2h()))
                .collect()
        };
        match rayon::ThreadPoolBuilder::new().num_threads(workers).build() {
            Ok(pool) => pool.install(run),
            Err(_) => run(),
        }
    };
    let (mut n_mats, mut amp_max) = (0u32, 0.0f32);
    for m in &mut scene.materials {
        if m.normal_tex == NO_TEX {
            continue;
        }
        let base = if scene.textures[m.normal_tex as usize].h2n {
            crate::texture::HEIGHT_NORMAL_STRENGTH
        } else {
            amps.get(&m.normal_tex)
                .copied()
                .unwrap_or(m.height_amp)
                .min(crate::texture::N2H_AMP_CAP)
        };
        m.height_amp = base * m.normal_scale;
        if m.height_amp > 0.0 {
            n_mats += 1;
            amp_max = amp_max.max(m.height_amp);
        }
    }
    // `finalize_scalars` already ran inside the loader's finish(), BEFORE the
    // amps existed — refresh the intersector's one-bool gate here (the BVH
    // build that follows reads it for the AABB sweep).
    scene.any_height = n_mats > 0;
    if n_mats > 0 || !solve.is_empty() {
        eprintln!(
            "heightfield: {} materials, amp max {:.2} texels ({} maps solved in {:.0} ms)",
            n_mats,
            amp_max,
            solve.len(),
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// Spray component ceiling, relative to `Scene::diag`: a transmissive
/// connected component whose AABB diagonal is under `SPRAY_MAX_K · diag`
/// reclassifies as spray. Calibrated on San Miguel (diag 10): droplets are
/// ~3e-4·diag islands inside the `o Water` mesh, the smallest glassware
/// ~1.5e-3·diag — 6e-4 (~4 cm) sits between with ~2× margin either way (the
/// load-time histogram line is the tuning signal).
pub const SPRAY_MAX_K: f32 = 6e-4;
/// Spray look: aerated water scatters white (games ship spray as white
/// particles for the same reason) — albedo lifts toward white, transmission
/// drops to 0 (a clear millimeter droplet is invisible against a matched
/// background), translucency keeps back-lit drops glowing.
pub const SPRAY_ALBEDO_LIFT: f32 = 0.6;
pub const SPRAY_TRANSLUCENCY: f32 = 0.35;
pub const SPRAY_ROUGHNESS: f32 = 0.4;

/// Post-load spray reclassification (shared OBJ+glTF, the `derive_heights`
/// pass family — cold loads only, BEFORE the cache store, so warm loads
/// inherit the retag from the sidecar; `--no-spray` kills it and keys the
/// cache lever word). Union-find over shared vertex indices, restricted to
/// transmissive-material triangles: tiny disconnected islands (the airborne
/// droplets of a fountain) become a white scattering "spray" clone of their
/// source material, while pools/streams/glassware (large components) stay
/// glass. Purely load-time — no per-ray cost, zero rng, and a scene with no
/// transmissive materials is structurally untouched.
pub fn reclassify_spray(scene: &mut Scene) {
    if !spray_enabled() {
        return;
    }
    let nv = scene.positions.len();
    let mut parent: Vec<u32> = (0..nv as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let g = parent[parent[x as usize] as usize];
            parent[x as usize] = g;
            x = g;
        }
        x
    }
    let mut any = false;
    for (t, idx) in scene.indices.iter().enumerate() {
        if scene.materials[scene.tri_mat[t] as usize].transmission <= 0.0 {
            continue;
        }
        any = true;
        let r0 = find(&mut parent, idx[0]);
        let r1 = find(&mut parent, idx[1]);
        let r2 = find(&mut parent, idx[2]);
        parent[r1 as usize] = r0;
        parent[r2 as usize] = r0;
    }
    if !any {
        return; // no transmissive geometry: structurally untouched
    }
    // Component AABBs, keyed by root vertex.
    let mut boxes: HashMap<u32, (Vec3A, Vec3A, u32)> = HashMap::new();
    for (t, idx) in scene.indices.iter().enumerate() {
        if scene.materials[scene.tri_mat[t] as usize].transmission <= 0.0 {
            continue;
        }
        let root = find(&mut parent, idx[0]);
        let e = boxes.entry(root).or_insert((
            Vec3A::splat(f32::INFINITY),
            Vec3A::splat(f32::NEG_INFINITY),
            0,
        ));
        for &i in idx {
            let p = scene.positions[i as usize];
            e.0 = e.0.min(p);
            e.1 = e.1.max(p);
        }
        e.2 += 1;
    }
    // Size histogram (log10 of diag-relative extent) — the threshold's
    // tuning signal, printed once per load.
    let mut hist = [0u32; 8];
    let limit = SPRAY_MAX_K * scene.diag;
    let mut spray_roots: HashMap<u32, ()> = HashMap::new();
    for (&root, &(mn, mx, _)) in &boxes {
        let d = (mx - mn).length();
        let rel = (d / scene.diag).max(1e-9);
        let bucket = ((rel.log10() + 6.0).floor() as i32).clamp(0, 7) as usize;
        hist[bucket] += 1;
        if d < limit {
            spray_roots.insert(root, ());
        }
    }
    if spray_roots.is_empty() {
        eprintln!(
            "spray: 0 of {} transmissive components under {:.0e}·diag | comp size hist (log10 rel, -6..) {:?}",
            boxes.len(),
            SPRAY_MAX_K,
            hist
        );
        return;
    }
    // Retag: one spray clone per source material (deduped), all fields
    // carried over except the spray overrides.
    let mut clones: HashMap<u32, u32> = HashMap::new();
    let mut n_tris = 0u64;
    for t in 0..scene.indices.len() {
        let m = scene.tri_mat[t];
        if scene.materials[m as usize].transmission <= 0.0 {
            continue;
        }
        let root = find(&mut parent, scene.indices[t][0]);
        if !spray_roots.contains_key(&root) {
            continue;
        }
        let clone = *clones.entry(m).or_insert_with(|| {
            // Field-by-field carry-over (Material is deliberately not Clone);
            // the scoped borrow ends before the push.
            let spray = {
                let s = &scene.materials[m as usize];
                Material {
                    albedo: s.albedo.lerp(Vec3A::ONE, SPRAY_ALBEDO_LIFT),
                    roughness: SPRAY_ROUGHNESS,
                    metallic: 0.0,
                    anisotropy: s.anisotropy,
                    sheen: s.sheen,
                    translucency: SPRAY_TRANSLUCENCY,
                    transmission: 0.0,
                    emissive: s.emissive,
                    normal_tex: s.normal_tex,
                    normal_scale: s.normal_scale,
                    height_amp: s.height_amp,
                    rough_tex: s.rough_tex,
                    metal_tex: s.metal_tex,
                    emissive_tex: s.emissive_tex,
                    kind: s.kind,
                }
            };
            scene.materials.push(spray);
            (scene.materials.len() - 1) as u32
        });
        scene.tri_mat[t] = clone;
        n_tris += 1;
    }
    eprintln!(
        "spray: {} components ({} tris) retagged from {} transmissive | comp size hist (log10 rel, -6..) {:?}",
        spray_roots.len(),
        n_tris,
        boxes.len(),
        hist
    );
}

/// Spray gates, run by `--check` (the tinted_shadow_self_test pattern —
/// hand-built islands, closed-form expectations): only tiny TRANSMISSIVE
/// components retag, the clone's overrides are pinned bitwise, two islands
/// of one source material share one deduped clone, large components and
/// tiny opaque islands are untouched, and the lever-off arm is a no-op.
pub fn spray_self_test() -> Result<(), String> {
    let glass = |transmission: f32| Material {
        albedo: Vec3A::new(0.8, 0.85, 0.9),
        roughness: 0.05,
        metallic: 0.0,
        anisotropy: 0.0,
        sheen: 0.0,
        translucency: 0.0,
        transmission,
        emissive: Vec3A::ZERO,
        normal_tex: NO_TEX,
        normal_scale: 1.0,
        height_amp: 0.0,
        rough_tex: NO_TEX,
        metal_tex: NO_TEX,
        emissive_tex: NO_TEX,
        kind: MatKind::Diffuse,
    };
    // Island A: a unit-scale glass quad (2 tris sharing vertices — one
    // component). Islands B/C: two detached 1e-4-scale glass tris (droplets;
    // same source material, so they must SHARE one clone). Island D: a tiny
    // OPAQUE tri that must never retag.
    let mk = || -> Scene {
        let mut positions = vec![
            Vec3A::new(0.0, 0.0, 0.0),
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(0.0, 1.0, 0.0),
            Vec3A::new(1.0, 1.0, 0.0),
        ];
        let mut indices = vec![[0u32, 1, 2], [1, 3, 2]];
        let mut tri_mat = vec![0u32, 0];
        let mut tiny = |at: Vec3A, mat: u32,
                        positions: &mut Vec<Vec3A>,
                        indices: &mut Vec<[u32; 3]>,
                        tri_mat: &mut Vec<u32>| {
            let b = positions.len() as u32;
            positions.push(at);
            positions.push(at + Vec3A::new(1e-4, 0.0, 0.0));
            positions.push(at + Vec3A::new(0.0, 1e-4, 0.0));
            indices.push([b, b + 1, b + 2]);
            tri_mat.push(mat);
        };
        tiny(Vec3A::new(0.2, 0.2, 0.5), 0, &mut positions, &mut indices, &mut tri_mat);
        tiny(Vec3A::new(0.7, 0.7, 0.5), 0, &mut positions, &mut indices, &mut tri_mat);
        tiny(Vec3A::new(0.5, 0.5, 0.7), 1, &mut positions, &mut indices, &mut tri_mat);
        let n = positions.len();
        let mut sc = Scene {
            positions,
            normals: vec![Vec3A::Z; n],
            texcoords: vec![Vec2::ZERO; n],
            indices,
            tri_mat,
            materials: vec![glass(0.9), glass(0.0)],
            textures: Vec::new(),
            any_alpha: false,
            any_height: false,
            any_transmissive: false,
            sun: crate::sky::Sun::new(Vec3A::Y),
            sky_sh: crate::sh::Sh9::ZERO,
            sky_scale: 1.0,
            night: 0.0,
            diag: 1.0,
            eps: 1e-4,
            ao_radius: 0.03,
            content_min: Vec3A::ZERO,
            content_max: Vec3A::ZERO,
        };
        finalize_scalars(&mut sc);
        sc
    };
    let saved = spray_enabled();
    set_spray(true);
    let restore = |r: Result<(), String>| {
        set_spray(saved);
        r
    };
    let run = || -> Result<(), String> {
        let mut sc = mk();
        reclassify_spray(&mut sc);
        // Exactly ONE clone appended (B and C dedupe onto it).
        if sc.materials.len() != 3 {
            return Err(format!("expected 3 materials after retag, got {}", sc.materials.len()));
        }
        if sc.tri_mat[0] != 0 || sc.tri_mat[1] != 0 {
            return Err("large glass component must stay glass".into());
        }
        if sc.tri_mat[2] != 2 || sc.tri_mat[3] != 2 {
            return Err(format!(
                "tiny glass islands must share the spray clone (got {}, {})",
                sc.tri_mat[2], sc.tri_mat[3]
            ));
        }
        if sc.tri_mat[4] != 1 {
            return Err("tiny OPAQUE island must not retag".into());
        }
        let s = &sc.materials[2];
        let want_albedo = Vec3A::new(0.8, 0.85, 0.9).lerp(Vec3A::ONE, SPRAY_ALBEDO_LIFT);
        if s.transmission != 0.0
            || s.translucency != SPRAY_TRANSLUCENCY
            || s.roughness != SPRAY_ROUGHNESS
            || s.metallic != 0.0
            || s.albedo.to_array().map(f32::to_bits)
                != want_albedo.to_array().map(f32::to_bits)
        {
            return Err("spray clone overrides not pinned".into());
        }
        // Lever off: a no-op, bit-for-bit.
        set_spray(false);
        let mut off = mk();
        set_spray(true);
        // mk() ran finalize under lever-off (any_transmissive false is fine
        // here — the pass keys on material transmission, not the flag).
        set_spray(false);
        reclassify_spray(&mut off);
        set_spray(true);
        if off.materials.len() != 2 || off.tri_mat != vec![0, 0, 0, 0, 1] {
            return Err("lever off must be a structural no-op".into());
        }
        Ok(())
    };
    let r = run();
    restore(r)
}

/// Recompute the scale-relative scalars (`diag`/`eps`/`ao_radius`) and
/// `any_alpha` from the current geometry/textures — shared by
/// `SceneBuilder::finish` and `tile_scene`: replication changes the bounds,
/// and every epsilon in the tracer is scale-relative to `diag`.
pub fn finalize_scalars(scene: &mut Scene) {
    let mut mn = Vec3A::splat(f32::INFINITY);
    let mut mx = Vec3A::splat(f32::NEG_INFINITY);
    // The content AABB in the same pass: every loader pushes the standard
    // ground quad FIRST (GROUND_VERTS), so positions[GROUND_VERTS..] is the
    // content. Hand-built self-test scenes without that convention fall back
    // to the full bounds below.
    let mut cmn = Vec3A::splat(f32::INFINITY);
    let mut cmx = Vec3A::splat(f32::NEG_INFINITY);
    for (i, p) in scene.positions.iter().enumerate() {
        mn = mn.min(*p);
        mx = mx.max(*p);
        if i >= GROUND_VERTS {
            cmn = cmn.min(*p);
            cmx = cmx.max(*p);
        }
    }
    let diag = (mx - mn).length().max(1e-3);
    scene.diag = diag;
    scene.eps = 1e-4 * diag;
    scene.ao_radius = 0.03 * diag;
    if cmn.x <= cmx.x && (cmx - cmn).length() > 1e-3 {
        scene.content_min = cmn;
        scene.content_max = cmx;
    } else {
        scene.content_min = mn;
        scene.content_max = mx;
    }
    scene.any_alpha = scene.textures.iter().any(|t| t.alpha_masked);
    scene.any_height =
        scene.materials.iter().any(|m| m.height_amp > 0.0 && m.normal_tex != NO_TEX);
    scene.any_transmissive =
        tinted_shadows() && scene.materials.iter().any(|m| m.transmission > 0.0);

    refresh_sky_sh(scene);
}

/// Re-derive the SH ambient from the current sun/sky state. Split out of
/// `finalize_scalars` so `apply_tod` can re-run it without the O(n) position
/// scan. Deterministic quadrature, no rng — which is why `scene_cache` needs
/// no format change for it.
///
/// The DOME, not the full sky: a gather path must never see the sun disc
/// (the direct loop already delivers it, with a shadow ray). See sky.rs's
/// central invariant. At night `scene.sun` IS the moon, so the ambient is the
/// moonlit dome — the frequency split carries over unchanged.
pub fn refresh_sky_sh(scene: &mut Scene) {
    let sun = scene.sun.dir;
    let scale = scene.sky_scale;
    scene.sky_sh = crate::sh::Sh9::project(|d| crate::sky::dome(d, sun, scale));
}

/// The sun direction is UNCHANGED from the old rect light's center — so shadow
/// directions and the sun's place in the sky don't move, and the visual diff is
/// attributable to the shading model alone. Its brightness is
/// `sky::SUN_E_OVER_PI`, which is exactly the old `light.color / |center|²`.
pub fn default_sun() -> crate::sky::Sun {
    crate::sky::Sun::new(Vec3A::new(6.0, 10.0, 4.0))
}

/// Time-of-day hour → unit sun direction: the great circle in the vertical
/// plane through the default sun's azimuth. 06:00 = horizon at +az, 12:00 =
/// zenith, 18:00 = horizon at −az; the 24→0 wrap is the same point on the
/// circle. Below-horizon hours are the night half of the arc (the MOON, at
/// `−dir`, is then above the horizon — see `apply_tod`).
pub fn sun_dir_for_tod(hour: f32) -> Vec3A {
    let d0 = default_sun().dir;
    let az = Vec3A::new(d0.x, 0.0, d0.z).normalize();
    let th = (hour.rem_euclid(24.0) - 6.0) / 12.0 * std::f32::consts::PI;
    az * th.cos() + Vec3A::Y * th.sin()
}

/// The hour whose arc position labels the DEFAULT sun (~9.61 ⇒ 09:37).
/// Derived FROM the default dir; it is a LABEL — the arc is never evaluated
/// at it unless the user actually scrubs, because the float round-trip is
/// approximate and untouched-session bit-identity depends on never
/// recomputing at zero delta.
pub fn default_tod() -> f32 {
    6.0 + default_sun().dir.y.asin() * (12.0 / std::f32::consts::PI)
}

/// Set the scene's lighting to time-of-day `hour` and re-derive the SH
/// ambient. The ONLY caller of `Sun::with_fade`/`sky::moon` — an untouched
/// session never reaches this, which is the structural bit-identity guard.
///
/// Day/dusk: the sun at its arc position, irradiance faded/reddened by the
/// dome's own transmittance (`sky::sun_fade`). Once the fade hits zero the
/// one light BECOMES the full moon, antipodal — above the horizon exactly
/// when the sun is not — and the dome (whose `sun` argument is now the moon)
/// renders the moonlit sky at the `MOON_DOME_FRAC` floor. `night` gates the
/// stars in after sunset.
pub fn apply_tod(scene: &mut Scene, hour: f32) {
    let dir = sun_dir_for_tod(hour);
    let fade = crate::sky::sun_fade(dir.y, default_sun().dir.y);
    let lum = fade.dot(Vec3A::new(0.2126, 0.7152, 0.0722));
    if fade != Vec3A::ZERO {
        scene.sun = crate::sky::Sun::with_fade(dir, fade);
    } else {
        scene.sun = crate::sky::moon(-dir);
    }
    // The dome tracks the direct light down through dusk, then rests on the
    // moonlight floor (scaled by the moon's own rise so deep twilight — both
    // bodies at the horizon — stays the darkest moment, as in life).
    let moon_up = ((-dir.y - 0.0) / 0.10).clamp(0.0, 1.0);
    scene.sky_scale = lum.max(crate::sky::MOON_DOME_FRAC * moon_up);
    // Stars fade in once the sun is well below the horizon.
    let t = ((-dir.y - 0.05) / 0.10).clamp(0.0, 1.0);
    scene.night = t * t * (3.0 - 2.0 * t);
    refresh_sky_sh(scene);
}

/// Closed-form time-of-day gates, run by `--check`. No rng, no DLLs. Pins the
/// arc's anchors, the fade's identities (the bit-identity guards), the sunset
/// ordering, the moon handoff, and the star field's day-guard/determinism.
pub fn tod_self_test() -> Result<(), String> {
    use crate::sky;
    let d0 = default_sun().dir;

    // Arc anchors: 06:00 east horizon, 12:00 zenith, 18:00 west horizon, all
    // unit; the default hour reproduces the default direction; 24 h wrap.
    let az = Vec3A::new(d0.x, 0.0, d0.z).normalize();
    for (h, want) in [(6.0, az), (12.0, Vec3A::Y), (18.0, -az)] {
        let d = sun_dir_for_tod(h);
        if (d.length() - 1.0).abs() > 1e-5 {
            return Err(format!("arc dir at {h}h is not unit: {d:?}"));
        }
        if (d - want).length() > 1e-5 {
            return Err(format!("arc anchor at {h}h: {d:?}, want {want:?}"));
        }
    }
    let h0 = default_tod();
    if (h0 - 9.61).abs() > 0.02 {
        return Err(format!("default_tod {h0:.3} drifted from ~9.61"));
    }
    if sun_dir_for_tod(h0).dot(d0) < 1.0 - 1e-6 {
        return Err("arc at default_tod does not reproduce the default sun".into());
    }
    for h in [0.5f32, 3.25, 9.61, 17.75, 23.0] {
        if sun_dir_for_tod(h).dot(sun_dir_for_tod(h + 24.0)) < 1.0 - 1e-4 {
            return Err(format!("arc is not 24h-periodic at {h}h"));
        }
    }
    if sun_dir_for_tod(23.999).dot(sun_dir_for_tod(0.001)) < 1.0 - 1e-4 {
        return Err("arc is discontinuous across the 24->0 wrap".into());
    }

    // Fade identities — the daytime bit-identity guards. exp(-0) == 1 exactly,
    // and a fade of exactly ONE must leave Sun::new untouched bit-for-bit.
    if sky::sun_fade(d0.y, d0.y) != Vec3A::ONE {
        return Err("sun_fade at the reference elevation is not exactly 1".into());
    }
    let s_ref = sky::Sun::new(d0);
    let s_faded = sky::Sun::with_fade(d0, Vec3A::ONE);
    if s_faded != s_ref {
        return Err("Sun::with_fade at fade 1 is not bit-identical to Sun::new".into());
    }
    // Shape: per-channel <= 1, monotone in y, the horizon transmittance pin
    // (red > green > blue — the sunset ordering; a channel swap fails loudly),
    // and a true zero once the disc has set.
    let mut prev = Vec3A::ZERO;
    for i in 0..=40 {
        let y = -0.1 + i as f32 * (1.0 + 0.1) / 40.0;
        let f = sky::sun_fade(y, d0.y);
        if f.max_element() > 1.0 + 1e-6 || f.min_element() < 0.0 {
            return Err(format!("sun_fade({y}) = {f:?} out of [0,1]"));
        }
        if (f - prev).min_element() < -1e-6 {
            return Err(format!("sun_fade is not monotone in y at {y}"));
        }
        prev = f;
    }
    let hz = sky::sun_fade(0.0, d0.y);
    if (hz - Vec3A::new(0.7176, 0.5178, 0.2251)).abs().max_element() > 0.01 {
        return Err(format!("horizon fade {hz:?} drifted from the transmittance model"));
    }
    if !(hz.x > hz.y && hz.y > hz.z) {
        return Err(format!("horizon fade is not sunset-ordered (r>g>b): {hz:?}"));
    }
    if sky::sun_fade(-0.05, d0.y) != Vec3A::ZERO || sky::sun_fade(-0.5, d0.y) != Vec3A::ZERO {
        return Err("sun_fade below the horizon band is not exactly zero".into());
    }

    // The moon handoff, on a throwaway scene. Night: the light is the moon
    // (above the horizon, dim, disc radiance f16-safe), the dome rests on the
    // moonlight floor, stars are armed. Deep dusk at the swap point: both
    // bodies near-zero, so the handoff cannot pop.
    let mut sc = SceneBuilder::new().finish(default_sun());
    apply_tod(&mut sc, 0.0); // midnight
    if sc.sun.dir.y <= 0.5 {
        return Err(format!("midnight light is not a high moon: dir {:?}", sc.sun.dir));
    }
    let lum = |v: Vec3A| v.dot(Vec3A::new(0.2126, 0.7152, 0.0722));
    if lum(sc.sun.e_over_pi) <= 0.0 || lum(sc.sun.e_over_pi) > 0.05 {
        return Err(format!("moonlight is out of band: {:?}", sc.sun.e_over_pi));
    }
    if !sc.sun.radiance.is_finite() || sc.sun.radiance.max_element() > 4096.0 {
        return Err(format!("moon disc radiance not f16-comfortable: {:?}", sc.sun.radiance));
    }
    if sc.sky_scale <= 0.0 || sc.sky_scale > 0.05 {
        return Err(format!("night sky_scale {} out of the moonlight band", sc.sky_scale));
    }
    if sc.night != 1.0 {
        return Err(format!("midnight night factor {} != 1", sc.night));
    }
    apply_tod(&mut sc, 18.20); // just past the sun's last light
    if lum(sc.sun.e_over_pi) > 0.02 {
        return Err(format!("handoff pop: light lum {} at 18.20h", lum(sc.sun.e_over_pi)));
    }
    apply_tod(&mut sc, 12.0); // noon: near-full sun, no stars
    if sc.night != 0.0 || sc.sky_scale < 0.9 {
        return Err(format!("noon state wrong: night {} scale {}", sc.night, sc.sky_scale));
    }

    // Stars: the day guard is a hard zero; the field is deterministic, finite,
    // clamped, and its mean radiance is negligible next to even the night dome.
    let mut sum = Vec3A::ZERO;
    let mut peak = 0.0f32;
    for i in 0..2000 {
        let a = i as f32 * 2.399_963;
        let z = (i as f32 + 0.5) / 2000.0; // upper hemisphere
        let r = (1.0 - z * z).max(0.0).sqrt();
        let d = Vec3A::new(r * a.cos(), z, r * a.sin());
        if sky::stars(d, 5e-4, 0.0, 7) != Vec3A::ZERO {
            return Err("stars are not exactly zero by day (night = 0)".into());
        }
        let s = sky::stars(d, 5e-4, 1.0, 7);
        if !s.is_finite() || s.min_element() < 0.0 {
            return Err(format!("stars({d:?}) = {s:?} — must be finite, non-negative"));
        }
        if s != sky::stars(d, 5e-4, 1.0, 7) {
            return Err("star field is not deterministic".into());
        }
        sum += s;
        peak = peak.max(s.max_element());
    }
    let mean = lum(sum / 2000.0);
    if mean <= 0.0 {
        return Err("star sweep found no stars at night = 1".into());
    }
    if mean > 0.01 {
        return Err(format!("star field mean radiance {mean} is not negligible"));
    }
    if peak > 4200.0 {
        return Err(format!("star peak radiance {peak} escaped the clamp"));
    }

    eprintln!(
        "tod self-test: OK (default {h0:.2}h, horizon fade {:.2}/{:.2}/{:.2}, star mean {mean:.2e})",
        hz.x, hz.y, hz.z
    );
    Ok(())
}

/// Ground plane + a grid of boxes + three spheres + a marble Stanford Bunny
/// and a stainless Utah teapot (both embedded in the binary). Deterministic,
/// ~83k triangles.
pub fn procedural_scene() -> Scene {
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

    let palette = [
        Vec3A::new(0.85, 0.30, 0.25),
        Vec3A::new(0.90, 0.65, 0.20),
        Vec3A::new(0.30, 0.60, 0.85),
        Vec3A::new(0.45, 0.75, 0.35),
        Vec3A::new(0.75, 0.45, 0.80),
    ];
    for gx in 0..5u32 {
        for gz in 0..5u32 {
            // deterministic hash-ish variation
            let fr = (((gx * 5 + gz) as f32 * 12.9898).sin() * 43758.5453).fract().abs();
            if fr < 0.14 {
                continue; // leave a few gaps
            }
            let h = 0.5 + 2.2 * fr;
            let x = (gx as f32 - 2.0) * 2.5;
            let z = (gz as f32 - 2.0) * 2.5;
            let rough = if fr > 0.82 { 0.30 } else { 0.90 };
            let mat = b.material(palette[((gx + 2 * gz) % 5) as usize], rough, 0.0);
            b.add_box(Vec3A::new(x, h * 0.5, z), Vec3A::new(0.8, h * 0.5, 0.8), mat);
        }
    }

    let mirror = b.material(Vec3A::new(0.95, 0.95, 0.95), 0.05, 1.0);
    b.add_sphere(Vec3A::new(-7.5, 1.5, 2.0), 1.5, mirror, 40, 20);
    let red = b.material(Vec3A::new(0.85, 0.15, 0.12), 0.85, 0.0);
    b.add_sphere(Vec3A::new(7.0, 1.2, -1.0), 1.2, red, 36, 18);
    let glossy = b.material(Vec3A::new(0.20, 0.35, 0.80), 0.25, 0.0);
    b.add_sphere(Vec3A::new(2.0, 0.9, 7.5), 0.9, glossy, 32, 16);

    // Marble Stanford Bunny, front of the grid (grid ends at |x|,|z| = 5.8).
    let marble = b.material_kind(
        Vec3A::new(0.93, 0.92, 0.90),
        0.35,
        0.0,
        0.0,
        MatKind::Marble { scale: 2.4 },
    );
    let bunny = embedded_obj(include_bytes!("../assets/bunny.obj"));
    add_obj_models(&mut b, &bunny, |_| marble, 3.5, Vec3A::new(5.5, 0.0, 6.5));

    // Brushed-stainless Utah teapot, right of the grid near the red sphere:
    // metal, moderate roughness, strongly anisotropic (lathe-spun finish).
    let steel = b.material_kind(Vec3A::new(0.97, 0.96, 0.93), 0.30, 1.0, 0.8, MatKind::Diffuse);
    let teapot = embedded_obj(include_bytes!("../assets/teapot.obj"));
    add_obj_models(&mut b, &teapot, |_| steel, 3.0, Vec3A::new(7.5, 0.0, 3.5));

    b.finish(default_sun())
}

/// Verts/tris the scene loaders push before the model itself — the standard
/// ground quad (`quad()` = two `tri()` calls = 6 duplicated verts / 2 tris).
/// `tile_scene` relies on this layout to replicate only the model, and
/// `world::merge_scenes` to strip each part's ground before placing it.
pub(crate) const GROUND_VERTS: usize = 6;
pub(crate) const GROUND_TRIS: usize = 2;

/// Tile a loaded (already diag-10-fitted) OBJ scene into an `nx`×`nz` grid by
/// duplicating the transformed geometry — flattened replication, deliberately
/// NOT instancing (two-level instancing is a deferred epic; the whole
/// correctness architecture assumes one flat BVH). Tiling runs AFTER the fit,
/// in fitted units, then re-derives the scale-relative scalars over the tiled
/// extent via `finalize_scalars` — tiling before the fit would squash the
/// field back into diag 10 and shrink eps below float precision at the
/// leaves. The ground quad is rewritten to cover the grid (not replicated),
/// and `materials`/`textures` are shared untouched — geometry is the only
/// thing that multiplies. The light is pushed out and brightened
/// stress-style (direction preserved, so `render::sun_dir` is unchanged).
/// Returns the scene and the field half-extent for camera framing.
pub fn tile_scene(base: Scene, nx: u32, nz: u32) -> (Scene, f32) {
    let tiles = nx as usize * nz as usize;
    let mv = base.positions.len() - GROUND_VERTS; // model verts per tile
    let mt = base.indices.len() - GROUND_TRIS; // model tris per tile

    // Model footprint on x/z (fitted units) -> grid pitch with a small gap.
    let mut mn = Vec3A::splat(f32::INFINITY);
    let mut mx = Vec3A::splat(f32::NEG_INFINITY);
    for p in &base.positions[GROUND_VERTS..] {
        mn = mn.min(*p);
        mx = mx.max(*p);
    }
    let pitch_x = (mx.x - mn.x).max(1e-3) * 1.05;
    let pitch_z = (mx.z - mn.z).max(1e-3) * 1.05;
    let fh = (pitch_x * nx as f32).max(pitch_z * nz as f32) * 0.5;

    // reserve_exact before the copy loop — at x20 the indices Vec alone is
    // multi-GB and Vec doubling would spike transient memory.
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut texcoords = Vec::new();
    let mut indices = Vec::new();
    let mut tri_mat = Vec::new();
    positions.reserve_exact(GROUND_VERTS + tiles * mv);
    normals.reserve_exact(GROUND_VERTS + tiles * mv);
    texcoords.reserve_exact(GROUND_VERTS + tiles * mv);
    indices.reserve_exact(GROUND_TRIS + tiles * mt);
    tri_mat.reserve_exact(GROUND_TRIS + tiles * mt);

    // Ground quad rewritten to cover the grid — same construction as the
    // loaders' `quad()` (two fan triangles, 6 duplicated verts, +y normal).
    let s = fh + 6.0;
    let (a, b, c, d) = (
        Vec3A::new(-s, 0.0, -s),
        Vec3A::new(-s, 0.0, s),
        Vec3A::new(s, 0.0, s),
        Vec3A::new(s, 0.0, -s),
    );
    positions.extend_from_slice(&[a, b, c, a, c, d]);
    normals.extend_from_slice(&[Vec3A::Y; GROUND_VERTS]);
    texcoords.extend_from_slice(&[Vec2::ZERO; GROUND_VERTS]);
    indices.push([0, 1, 2]);
    indices.push([3, 4, 5]);
    tri_mat.extend_from_slice(&base.tri_mat[..GROUND_TRIS]);

    for iz in 0..nz {
        for ix in 0..nx {
            let off = Vec3A::new(
                (ix as f32 - (nx as f32 - 1.0) * 0.5) * pitch_x,
                0.0,
                (iz as f32 - (nz as f32 - 1.0) * 0.5) * pitch_z,
            );
            let vbase = positions.len() as u32;
            positions.extend(base.positions[GROUND_VERTS..].iter().map(|&p| p + off));
            normals.extend_from_slice(&base.normals[GROUND_VERTS..]);
            texcoords.extend_from_slice(&base.texcoords[GROUND_VERTS..]);
            for tri in &base.indices[GROUND_TRIS..] {
                indices.push([
                    tri[0] - GROUND_VERTS as u32 + vbase,
                    tri[1] - GROUND_VERTS as u32 + vbase,
                    tri[2] - GROUND_VERTS as u32 + vbase,
                ]);
            }
            tri_mat.extend_from_slice(&base.tri_mat[GROUND_TRIS..]);
        }
    }

    // The sun is at infinity and has no falloff, so a tiled/replicated scene
    // needs NO light rescaling at all. (This used to push the rect light out by
    // k and brighten it by k² to compensate 1/d² — a hack that existed only
    // because the "sun" was a lamp 12 units away. A sun does not get closer to
    // one end of the field.)

    let mut scene = Scene {
        positions,
        normals,
        texcoords,
        indices,
        tri_mat,
        materials: base.materials,
        textures: base.textures,
        any_alpha: false,
        any_height: false,
        any_transmissive: false,
        // The sun is at infinity: a replicated field sees the SAME sun, at the
        // same angle, with the same irradiance, everywhere. Nothing to rescale.
        sun: base.sun,
        sky_sh: crate::sh::Sh9::ZERO,
        sky_scale: base.sky_scale,
        night: base.night,
        diag: 0.0,
        eps: 0.0,
        ao_radius: 0.0,
        content_min: Vec3A::ZERO,
        content_max: Vec3A::ZERO,
    };
    finalize_scalars(&mut scene);
    eprintln!(
        "tiled scene: {nx}x{nz} = {tiles} copies | {} tris | field {:.0}x{:.0} | diag {:.1}",
        scene.tri_count(),
        fh * 2.0,
        fh * 2.0,
        scene.diag
    );
    (scene, fh)
}

/// Grid pitch of the stress field (world units between object centers).
const STRESS_SPACING: f32 = 2.2;

/// Half-extent of the `--stress n` object field on x/z. Exported so `main.rs`
/// can frame the camera without duplicating the grid math.
pub fn stress_field_half(n: usize) -> f32 {
    let side = (n as f32).sqrt().ceil().max(1.0);
    side * STRESS_SPACING * 0.5
}

/// Performance stress field: exactly `n` objects on a jittered grid — mostly
/// boxes and low-poly spheres, plus evenly spread marble bunnies and steel
/// teapots (capped at 256 mesh instances: there is no instancing, every mesh
/// is duplicated geometry). Deterministic — same sin-hash idiom as
/// `procedural_scene`, no RNG — so `--check --stress n` is reproducible.
pub fn stress_scene(n: usize) -> Scene {
    let mut b = SceneBuilder::new();
    let side = (n as f32).sqrt().ceil().max(1.0) as usize;
    let fh = stress_field_half(n);

    let ground = b.material(Vec3A::new(0.42, 0.46, 0.40), 0.55, 0.0);
    let s = fh + 6.0;
    b.quad(
        Vec3A::new(-s, 0.0, -s),
        Vec3A::new(-s, 0.0, s),
        Vec3A::new(s, 0.0, s),
        Vec3A::new(s, 0.0, -s),
        ground,
    );

    let palette = [
        Vec3A::new(0.85, 0.30, 0.25),
        Vec3A::new(0.90, 0.65, 0.20),
        Vec3A::new(0.30, 0.60, 0.85),
        Vec3A::new(0.45, 0.75, 0.35),
        Vec3A::new(0.75, 0.45, 0.80),
    ];
    // Shared materials up-front (one per look, not one per object).
    let rough: Vec<u32> = palette.iter().map(|&c| b.material(c, 0.90, 0.0)).collect();
    let glossy: Vec<u32> = palette.iter().map(|&c| b.material(c, 0.30, 0.0)).collect();
    let mirror = b.material(Vec3A::new(0.95, 0.95, 0.95), 0.05, 1.0);
    let marble = b.material_kind(
        Vec3A::new(0.93, 0.92, 0.90),
        0.35,
        0.0,
        0.0,
        MatKind::Marble { scale: 2.4 },
    );
    let steel = b.material_kind(Vec3A::new(0.97, 0.96, 0.93), 0.30, 1.0, 0.8, MatKind::Diffuse);

    // Meshes go on an even stride so the cap never starves part of the field.
    let mesh_target = (n / 50).min(256);
    let mesh_stride = if mesh_target > 0 { n.div_ceil(mesh_target) } else { usize::MAX };
    let bunny = embedded_obj(include_bytes!("../assets/bunny.obj"));
    let teapot = embedded_obj(include_bytes!("../assets/teapot.obj"));

    // Deterministic hash-ish variation, same idiom as `procedural_scene`.
    let hv = |i: usize, k: f32| (((i as f32 + 1.0) * k).sin() * 43758.5453).fract().abs();

    let (mut boxes, mut spheres, mut bunnies, mut teapots) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..n {
        let (gx, gz) = (i % side, i / side);
        let (h1, h2, h3) = (hv(i, 12.9898), hv(i, 78.2330), hv(i, 39.4250));
        let x = (gx as f32 - (side as f32 - 1.0) * 0.5) * STRESS_SPACING + (h2 - 0.5) * 0.7;
        let z = (gz as f32 - (side as f32 - 1.0) * 0.5) * STRESS_SPACING + (h3 - 0.5) * 0.7;

        if mesh_stride != usize::MAX && i % mesh_stride == mesh_stride / 2 {
            if (bunnies + teapots) % 2 == 0 {
                add_obj_models(&mut b, &bunny, |_| marble, 2.5, Vec3A::new(x, 0.0, z));
                bunnies += 1;
            } else {
                add_obj_models(&mut b, &teapot, |_| steel, 2.5, Vec3A::new(x, 0.0, z));
                teapots += 1;
            }
            continue;
        }

        let mat = if h3 > 0.97 {
            mirror
        } else if h3 > 0.94 {
            marble
        } else if h1 > 0.82 {
            glossy[(gx + 2 * gz) % 5]
        } else {
            rough[(gx + 2 * gz) % 5]
        };
        if h2 < 0.65 {
            let h = 0.5 + 2.2 * h1;
            let half = 0.4 + 0.5 * h3;
            b.add_box(Vec3A::new(x, h * 0.5, z), Vec3A::new(half, h * 0.5, half), mat);
            boxes += 1;
        } else {
            let r = 0.35 + 0.45 * h1;
            b.add_sphere(Vec3A::new(x, r, z), r, mat, 10, 5);
            spheres += 1;
        }
    }

    // No light rescaling: the sun is at infinity, so however wide the stress
    // field grows, every object sees the same sun at the same angle. (This used
    // to push a rect lamp out by k and brighten it by k² to undo 1/d².)

    let scene = b.finish(default_sun());
    eprintln!(
        "stress scene: {n} objects ({boxes} boxes, {spheres} spheres, {bunnies} bunnies, {teapots} teapots) | {} tris | field {:.0}x{:.0}",
        scene.tri_count(),
        fh * 2.0,
        fh * 2.0
    );
    scene
}

/// Resolve a scene path: a bare `.obj` argument falls back to its `.zst`
/// sibling when only that exists (the committed scene data lives in git LFS
/// zstd-compressed), so documented `model.obj` commands keep working on a
/// fresh checkout. The scene cache keys on the RESOLVED path — main.rs must
/// resolve before consulting it.
pub fn resolve_scene_path(path: &str) -> String {
    let mut p = path.to_string();
    if !std::path::Path::new(&p).exists() {
        let zst = format!("{p}.zst");
        if std::path::Path::new(&zst).exists() {
            p = zst;
        }
    }
    p
}

/// The texture flavor of the `.zst` sibling convention: MTL/glTF manifests
/// keep referencing `foo.png`, but committed scenes store textures as
/// LOSSLESS `foo.webp` (~30% smaller than PNG; decoded RGBA is bit-identical
/// — encode with `exact` so RGB under A==0 texels survives, `sample_bilinear`
/// blends them at cutout edges). When the referenced file is absent and a
/// `.webp` sibling exists, resolve to the sibling; an existing file always
/// wins verbatim, so plain-PNG scenes load unchanged.
pub fn resolve_texture_path(p: std::path::PathBuf) -> std::path::PathBuf {
    if !p.exists() {
        let w = p.with_extension("webp");
        if w.exists() {
            return w;
        }
    }
    p
}

/// Parse an MTL map-statement value: consumes a leading `-bm <s>` option
/// (bump multiplier — the only option we honor; others are skipped token by
/// token) and returns (path, bm). The LAST whitespace token is taken as the
/// path — MTL cannot quote paths, so a path with spaces is ambiguous with
/// options anyway (none of the archive scenes have one).
fn parse_map_value(v: &str) -> (String, f32) {
    let mut scale = 1.0f32;
    let toks: Vec<&str> = v.split_whitespace().collect();
    let mut i = 0;
    while i + 1 < toks.len() {
        if toks[i] == "-bm" {
            if let Ok(s) = toks[i + 1].parse() {
                scale = s;
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    (toks.last().map(|s| s.to_string()).unwrap_or_default(), scale)
}

/// Load an OBJ, auto-fit it (centered on x/z, resting on y=0, diagonal = 10),
/// and drop it onto the standard ground plane + light.
///
/// `.obj.zst` is decoded transparently (the committed scene data lives in git
/// LFS zstd-compressed — OBJ is ASCII text; see .gitattributes), and a bare
/// `.obj` argument falls back to its `.zst` sibling when only that exists, so
/// the documented `model.obj` commands keep working on a fresh checkout.
pub fn load_obj_scene(path: &str) -> Scene {
    let path = &resolve_scene_path(path);
    let opts = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    };
    let (models, materials_res) = if path.ends_with(".zst") {
        // Decode to memory, then parse the buffer; MTL references inside the
        // OBJ resolve relative to the OBJ's directory, exactly like
        // tobj::load_obj does (the .mtl files are small and stay plain text).
        let dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();
        std::fs::File::open(path)
            .map_err(|_| tobj::LoadError::OpenFileFailed)
            .and_then(|f| {
                zstd::stream::decode_all(std::io::BufReader::new(f))
                    .map_err(|_| tobj::LoadError::ReadError)
            })
            .and_then(|text| {
                tobj::load_obj_buf(&mut &text[..], &opts, |mtl| tobj::load_mtl(dir.join(mtl)))
            })
    } else {
        tobj::load_obj(path, &opts)
    }
    .unwrap_or_else(|e| panic!("failed to load OBJ '{path}': {e}"));
    let obj_mats = materials_res.unwrap_or_default();

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

    let default_mat = b.material(Vec3A::new(0.70, 0.70, 0.72), 0.8, 0.0);

    // Collect every referenced map with its color-space role, deduped by
    // (resolved path, srgb) in first-reference order — deterministic texture
    // ids are what let the scene cache store paths by id. map_Kd / map_Ke
    // are sRGB color; normal and roughness/metallic maps are LINEAR data
    // (and must never arm the alpha-cutout pipeline). MTL paths are relative
    // to the OBJ's directory and often use backslashes.
    let obj_dir = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
    // resolve_texture_path here means dedup keys, tex ids, Texture::source,
    // and the scene cache's stat keys all carry the path that actually
    // exists on disk (.webp sibling for committed scenes).
    let resolve = |t: &str| resolve_texture_path(obj_dir.join(t.replace('\\', "/")));
    struct TexReq {
        path: std::path::PathBuf,
        srgb: bool,
        kd: bool,     // referenced as map_Kd (the only role that may cutout)
        normal: bool, // referenced as a normal/bump map (grayscale check)
        other: bool,  // referenced as rough/metal/emissive
    }
    let mut reqs: Vec<TexReq> = Vec::new();
    let mut req_idx: HashMap<(std::path::PathBuf, bool), usize> = HashMap::new();
    fn add_req(
        reqs: &mut Vec<TexReq>,
        req_idx: &mut HashMap<(std::path::PathBuf, bool), usize>,
        path: std::path::PathBuf,
        srgb: bool,
        kd: bool,
        normal: bool,
        other: bool,
    ) {
        match req_idx.get(&(path.clone(), srgb)) {
            Some(&i) => {
                reqs[i].kd |= kd;
                reqs[i].normal |= normal;
                reqs[i].other |= other;
            }
            None => {
                req_idx.insert((path.clone(), srgb), reqs.len());
                reqs.push(TexReq { path, srgb, kd, normal, other });
            }
        }
    }
    for m in &obj_mats {
        if let Some(t) = &m.diffuse_texture {
            add_req(&mut reqs, &mut req_idx, resolve(t), true, true, false, false);
        }
        let norm_val =
            m.normal_texture.as_deref().or_else(|| m.unknown_param.get("norm").map(|s| s.as_str()));
        if let Some(v) = norm_val {
            let (p, _) = parse_map_value(v);
            if !p.is_empty() {
                add_req(&mut reqs, &mut req_idx, resolve(&p), false, false, true, false);
            }
        }
        for key in ["map_Pr", "map_Pm"] {
            if let Some(v) = m.unknown_param.get(key) {
                let (p, _) = parse_map_value(v);
                if !p.is_empty() {
                    add_req(&mut reqs, &mut req_idx, resolve(&p), false, false, false, true);
                }
            }
        }
        if let Some(v) = m.unknown_param.get("map_Ke") {
            let (p, _) = parse_map_value(v);
            if !p.is_empty() {
                add_req(&mut reqs, &mut req_idx, resolve(&p), true, false, false, true);
            }
        }
    }
    let mut decoded: HashMap<(std::path::PathBuf, bool), Texture> = {
        use rayon::prelude::*;
        // Largest-first (LPT) scheduling with per-item tasks: WebP lossless
        // decodes slower than PNG per file, so load time is dominated by the
        // TAIL — the last big 4K maps decoding alone. Starting the biggest
        // files first fills the stragglers with small ones. Output is a
        // HashMap and ids are assigned later in MTL order, so scheduling
        // order never shifts texture ids.
        let mut by_size: Vec<&TexReq> = reqs.iter().collect();
        by_size.sort_by_key(|r| {
            std::cmp::Reverse(std::fs::metadata(&r.path).map_or(0, |m| m.len()))
        });
        by_size
            .par_iter()
            .with_max_len(1)
            .filter_map(|r| match image::open(&r.path) {
                Ok(img) => {
                    Some(((r.path.clone(), r.srgb), Texture::from_image(img, r.srgb)))
                }
                Err(e) => {
                    eprintln!(
                        "warning: texture '{}' failed to load ({e}); using flat fallback",
                        r.path.display()
                    );
                    None
                }
            })
            .collect()
    };
    // Assign ids in request (MTL) order, not HashMap order. Grayscale
    // "normal maps" are height maps (San Miguel's map_Bump files are a mix)
    // — treating one as a normal map shades garbage, so each is CONVERTED
    // (`Texture::height_to_normal`: Sobel normal in RGB, the exact source
    // height in A) into its own texture id, recorded in `h2n_ids` so the
    // material lookup below points at the converted texels. The raw
    // grayscale copy is still kept when another linear role (rough/metal —
    // same (path, false) dedup key) wants the file: those must sample raw
    // texels, never a normal map.
    let mut tex_ids: HashMap<(std::path::PathBuf, bool), u32> = HashMap::new();
    let mut h2n_ids: HashMap<std::path::PathBuf, u32> = HashMap::new();
    let mut n_height = 0u32;
    for r in &reqs {
        let k = (r.path.clone(), r.srgb);
        if let Some(mut t) = decoded.remove(&k) {
            if r.normal && t.is_grayscale() {
                n_height += 1;
                // NO_TEX under --no-h2n: the entry still marks the file as a
                // height map so the material lookup can't wire the raw
                // grayscale as a normal map (the pre-conversion skip).
                let id = if crate::texture::h2n_enabled() {
                    let mut nt = t.height_to_normal();
                    nt.source = r.path.to_string_lossy().into_owned();
                    b.add_texture(nt)
                } else {
                    NO_TEX
                };
                h2n_ids.insert(r.path.clone(), id);
                if !r.kd && !r.other {
                    continue; // height-map only — the raw texels aren't needed
                }
            }
            if !r.kd {
                // Only Kd-role textures may arm the cutout pipeline —
                // emissive color maps with junk alpha must not flip
                // Scene::any_alpha.
                t.alpha_masked = false;
            }
            t.source = r.path.to_string_lossy().into_owned();
            let id = b.add_texture(t);
            tex_ids.insert(k, id);
        }
    }

    let mut class_counts = vec![0u32; crate::matclass::NAMES.len()];
    let (mut n_normal, mut n_rough, mut n_metal, mut n_emissive) = (0u32, 0u32, 0u32, 0u32);
    let mat_map: Vec<u32> = obj_mats
        .iter()
        .map(|m| {
            let mut kd = Vec3A::from_array(m.diffuse.unwrap_or([0.7, 0.7, 0.7]));
            let tex = m
                .diffuse_texture
                .as_ref()
                .and_then(|t| tex_ids.get(&(resolve(t), true)).copied());
            // Classify by texture filename stem (the MTL's only reliable
            // signal — see matclass.rs), falling back to the material name
            // and the Ns/illum heuristic for the untextured glassware.
            let stem = m.diffuse_texture.as_ref().map(|t| {
                let t = t.replace('\\', "/");
                let file = t.rsplit('/').next().unwrap_or(&t);
                file.split('.').next().unwrap_or(file).to_ascii_lowercase()
            });
            let (class, pbr) =
                crate::matclass::classify(stem.as_deref(), &m.name, m.shininess, m.illumination_model);
            class_counts[class] += 1;
            if pbr.transmission > 0.0 {
                // Transmitted light is tinted by albedo and San Miguel's
                // glass Kd is 0.1-0.2 dark — lift toward white or glass
                // renders near-black.
                kd = Vec3A::ONE.lerp(kd, 0.2);
            }
            let kind = match tex {
                // The texture REPLACES Kd (exporters set Kd = 1 alongside
                // map_Kd; multiplying would double-darken). Kd stays as the
                // flat fallback for paths without texture support (GPU).
                Some(tex) => MatKind::Textured { tex },
                None => MatKind::Diffuse,
            };
            // Normal map: map_Bump/bump (first-class in tobj, value may
            // carry `-bm s`) or `norm` (unknown_param); grayscale files
            // resolve to their Sobel-converted normal map (see h2n_ids
            // above), keeping the parsed `-bm` and carrying the exact
            // source height (amp = the conversion's own K, texel units).
            let norm_val = m
                .normal_texture
                .as_deref()
                .or_else(|| m.unknown_param.get("norm").map(|s| s.as_str()));
            let (normal_tex, normal_scale, height_amp) = norm_val
                .map(parse_map_value)
                .filter(|(p, _)| !p.is_empty())
                .map(|(p, s)| {
                    let rp = resolve(&p);
                    match h2n_ids.get(&rp) {
                        Some(&id) if id != NO_TEX => {
                            (id, s, crate::texture::HEIGHT_NORMAL_STRENGTH)
                        }
                        // Height map under --no-h2n: the pre-conversion skip.
                        Some(_) => (NO_TEX, 1.0, 0.0),
                        None => {
                            (tex_ids.get(&(rp, false)).copied().unwrap_or(NO_TEX), s, 0.0)
                        }
                    }
                })
                .unwrap_or((NO_TEX, 1.0, 0.0));
            // Roughness/metallic maps (PBR MTL extension, unknown_param) +
            // their scalars: factor × sample is the glTF semantic — with a
            // map the factor is the MTL's own Pr/Pm (default 1.0), bypassing
            // the matclass constant; a bare scalar also beats the heuristic.
            let linear_map = |key: &str| {
                m.unknown_param
                    .get(key)
                    .map(|v| parse_map_value(v))
                    .filter(|(p, _)| !p.is_empty())
                    .and_then(|(p, _)| tex_ids.get(&(resolve(&p), false)).copied())
                    .unwrap_or(NO_TEX)
            };
            let rough_tex = linear_map("map_Pr");
            let metal_tex = linear_map("map_Pm");
            let pr: Option<f32> = m.unknown_param.get("Pr").and_then(|v| v.trim().parse().ok());
            let pm: Option<f32> = m.unknown_param.get("Pm").and_then(|v| v.trim().parse().ok());
            let roughness =
                if rough_tex != NO_TEX { pr.unwrap_or(1.0) } else { pr.unwrap_or(pbr.roughness) };
            let metallic =
                if metal_tex != NO_TEX { pm.unwrap_or(1.0) } else { pm.unwrap_or(pbr.metallic) };
            // Emissive: Ke (first-class) + map_Ke; a map with Ke absent/zero
            // gets factor 1.0 (the map_Kd precedent — exporters zero the
            // scalar alongside the map).
            let ke = Vec3A::from_array(m.emissive.unwrap_or([0.0; 3]));
            let emissive_tex = m
                .unknown_param
                .get("map_Ke")
                .map(|v| parse_map_value(v))
                .filter(|(p, _)| !p.is_empty())
                .and_then(|(p, _)| tex_ids.get(&(resolve(&p), true)).copied())
                .unwrap_or(NO_TEX);
            let emissive =
                if emissive_tex != NO_TEX && ke == Vec3A::ZERO { Vec3A::ONE } else { ke };
            n_normal += (normal_tex != NO_TEX) as u32;
            n_rough += (rough_tex != NO_TEX) as u32;
            n_metal += (metal_tex != NO_TEX) as u32;
            n_emissive += (emissive != Vec3A::ZERO || emissive_tex != NO_TEX) as u32;
            b.material_full(Material {
                albedo: kd,
                roughness,
                metallic,
                anisotropy: 0.0,
                sheen: pbr.sheen,
                translucency: pbr.translucency,
                transmission: pbr.transmission,
                emissive,
                normal_tex,
                normal_scale,
                height_amp,
                rough_tex,
                metal_tex,
                emissive_tex,
                kind,
            })
        })
        .collect();
    if !obj_mats.is_empty() {
        let mut parts: Vec<(usize, u32)> = class_counts.iter().copied().enumerate().collect();
        parts.sort_by_key(|&(i, n)| (std::cmp::Reverse(n), i));
        let body = parts
            .iter()
            .filter(|&&(i, n)| n > 0 || crate::matclass::NAMES[i] == "default")
            .map(|&(i, n)| format!("{} {}", crate::matclass::NAMES[i], n))
            .collect::<Vec<_>>()
            .join(" | ");
        eprintln!(
            "obj materials: {} -> {} || maps: normal {} | rough {} | metal {} | emissive {} | height-maps converted {}",
            obj_mats.len(),
            body,
            n_normal,
            n_rough,
            n_metal,
            n_emissive,
            n_height
        );
    }

    add_obj_models(
        &mut b,
        &models,
        |mesh| {
            mesh.material_id
                .and_then(|id| mat_map.get(id).copied())
                .unwrap_or(default_mat)
        },
        10.0,
        Vec3A::ZERO,
    );

    b.finish(default_sun())
}

/// Fit `models` to a bounding diagonal of `target_diag` — centered on x/z,
/// resting on y = 0 — translate by `offset`, and add every mesh to the
/// builder. `mat_for` picks the material id per mesh.
fn add_obj_models(
    b: &mut SceneBuilder,
    models: &[tobj::Model],
    mat_for: impl Fn(&tobj::Mesh) -> u32,
    target_diag: f32,
    offset: Vec3A,
) {
    // Pass 1: model bounds for the fit transform.
    let mut mn = Vec3A::splat(f32::INFINITY);
    let mut mx = Vec3A::splat(f32::NEG_INFINITY);
    for m in models {
        for p in m.mesh.positions.chunks_exact(3) {
            let v = Vec3A::new(p[0], p[1], p[2]);
            mn = mn.min(v);
            mx = mx.max(v);
        }
    }
    let scale = target_diag / (mx - mn).length().max(1e-6);
    let center = (mn + mx) * 0.5;
    let xform = |p: Vec3A| (p - Vec3A::new(center.x, mn.y, center.z)) * scale + offset;

    for m in models {
        let mesh = &m.mesh;
        let positions: Vec<Vec3A> = mesh
            .positions
            .chunks_exact(3)
            .map(|p| xform(Vec3A::new(p[0], p[1], p[2])))
            .collect();
        let normals: Vec<Vec3A> = mesh
            .normals
            .chunks_exact(3)
            .map(|n| Vec3A::new(n[0], n[1], n[2]).normalize_or_zero())
            .collect();
        // V is flipped once here (OBJ UVs are bottom-left origin, decoded
        // images top-left) so texture sampling needs no per-lookup flip.
        let texcoords: Vec<Vec2> = mesh
            .texcoords
            .chunks_exact(2)
            .map(|t| Vec2::new(t[0], 1.0 - t[1]))
            .collect();
        let indices: Vec<[u32; 3]> = mesh
            .indices
            .chunks_exact(3)
            .map(|i| [i[0], i[1], i[2]])
            .collect();
        b.add_mesh(positions, normals, texcoords, &indices, mat_for(mesh));
    }
}

/// Parse an OBJ embedded in the binary (no MTL: the loader closure returns
/// empty materials).
fn embedded_obj(bytes: &[u8]) -> Vec<tobj::Model> {
    let (models, _) = tobj::load_obj_buf(
        &mut &bytes[..],
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |_| Ok((Vec::new(), Default::default())),
    )
    .expect("embedded OBJ is valid");
    models
}
