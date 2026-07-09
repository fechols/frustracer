use glam::Vec3A;
use std::collections::HashMap;

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
}

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
    pub kind: MatKind,
}

/// Rectangular area light: `center ± u ± v`, radiant intensity `color` (falls off 1/d²).
pub struct AreaLight {
    pub center: Vec3A,
    pub u: Vec3A,
    pub v: Vec3A,
    pub color: Vec3A,
}

pub struct Scene {
    pub positions: Vec<Vec3A>,
    pub normals: Vec<Vec3A>,
    pub indices: Vec<[u32; 3]>,
    pub tri_mat: Vec<u32>,
    pub materials: Vec<Material>,
    pub light: AreaLight,
    /// Bounding diagonal — the scale reference for all epsilons.
    pub diag: f32,
    /// Self-intersection offset for secondary rays.
    pub eps: f32,
    pub ao_radius: f32,
}

impl Scene {
    pub fn tri_count(&self) -> usize {
        self.indices.len()
    }
}

pub struct SceneBuilder {
    positions: Vec<Vec3A>,
    normals: Vec<Vec3A>,
    indices: Vec<[u32; 3]>,
    tri_mat: Vec<u32>,
    materials: Vec<Material>,
}

impl SceneBuilder {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            tri_mat: Vec::new(),
            materials: Vec::new(),
        }
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
        self.materials.push(Material { albedo, roughness, metallic, anisotropy, kind });
        (self.materials.len() - 1) as u32
    }

    /// Push a triangle with per-vertex normals (vertices are duplicated, not shared).
    pub fn tri(&mut self, p: [Vec3A; 3], n: [Vec3A; 3], mat: u32) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&p);
        self.normals.extend_from_slice(&n);
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
    /// smooth normals are computed from area-weighted face normals.
    pub fn add_mesh(
        &mut self,
        positions: Vec<Vec3A>,
        mut normals: Vec<Vec3A>,
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
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&positions);
        self.normals.extend_from_slice(&normals);
        for tri in indices {
            self.indices.push([tri[0] + base, tri[1] + base, tri[2] + base]);
            self.tri_mat.push(mat);
        }
    }

    pub fn finish(self, light: AreaLight) -> Scene {
        let mut mn = Vec3A::splat(f32::INFINITY);
        let mut mx = Vec3A::splat(f32::NEG_INFINITY);
        for p in &self.positions {
            mn = mn.min(*p);
            mx = mx.max(*p);
        }
        let diag = (mx - mn).length().max(1e-3);
        Scene {
            positions: self.positions,
            normals: self.normals,
            indices: self.indices,
            tri_mat: self.tri_mat,
            materials: self.materials,
            light,
            diag,
            eps: 1e-4 * diag,
            ao_radius: 0.03 * diag,
        }
    }
}

fn default_light() -> AreaLight {
    AreaLight {
        center: Vec3A::new(6.0, 10.0, 4.0),
        u: Vec3A::new(2.0, 0.0, 0.0),
        v: Vec3A::new(0.0, 0.0, 2.0),
        color: Vec3A::new(1.0, 0.95, 0.85) * 150.0,
    }
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

    b.finish(default_light())
}

/// Load an OBJ, auto-fit it (centered on x/z, resting on y=0, diagonal = 10),
/// and drop it onto the standard ground plane + light.
pub fn load_obj_scene(path: &str) -> Scene {
    let (models, materials_res) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
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
    let mat_map: Vec<u32> = obj_mats
        .iter()
        .map(|m| {
            let kd = m.diffuse.unwrap_or([0.7, 0.7, 0.7]);
            b.material(Vec3A::from_array(kd), 0.8, 0.0)
        })
        .collect();

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

    b.finish(default_light())
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
        let indices: Vec<[u32; 3]> = mesh
            .indices
            .chunks_exact(3)
            .map(|i| [i[0], i[1], i[2]])
            .collect();
        b.add_mesh(positions, normals, &indices, mat_for(mesh));
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
