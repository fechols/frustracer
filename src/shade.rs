use crate::bvh::{Bvh, Hit, Ray};
use crate::scene::Scene;
use glam::Vec3A;

#[derive(Clone, Copy)]
pub struct Quality {
    pub shadow_samples: u32,
    pub ao_samples: u32,
    pub reflections: bool,
}

impl Quality {
    pub fn preset(i: u32) -> Quality {
        match i {
            1 => Quality { shadow_samples: 1, ao_samples: 0, reflections: false },
            3 => Quality { shadow_samples: 4, ao_samples: 4, reflections: true },
            _ => Quality { shadow_samples: 2, ao_samples: 2, reflections: true },
        }
    }

    /// Cheap variant used while the camera is moving; accumulation converges
    /// the full quality once the camera is still.
    pub fn while_moving(&self) -> Quality {
        Quality {
            shadow_samples: 1,
            ao_samples: 0,
            reflections: self.reflections,
        }
    }
}

const AMBIENT: Vec3A = Vec3A::new(0.14, 0.17, 0.23);

/// Primary-hit surface data captured for the DLSS-RR G-buffers. Filled by
/// `shade` only when the caller passes `Some` — and the caller passes `Some`
/// only for primary rays; the recursive reflection call always passes `None`,
/// which is what structurally guarantees secondaries can never write it.
#[derive(Default, Clone, Copy)]
pub struct PrimarySurface {
    /// The exact world-space normal used for shading (post face-flip).
    pub n: Vec3A,
    /// Raw material albedo (the diffuse/specular split happens at the
    /// G-buffer write site).
    pub albedo: Vec3A,
    pub reflectivity: f32,
    /// Mirror-reflection ray hit t; INFINITY when the reflection ray missed;
    /// 0.0 when no reflection was traced.
    pub spec_t: f32,
}

pub fn sky(d: Vec3A, sun: Vec3A) -> Vec3A {
    let t = (d.y * 0.7 + 0.3).clamp(0.0, 1.0);
    let horizon = Vec3A::new(0.72, 0.82, 0.95);
    let zenith = Vec3A::new(0.18, 0.35, 0.70);
    let glow = d.dot(sun).max(0.0).powi(32) * Vec3A::new(1.0, 0.9, 0.7) * 0.6;
    horizon.lerp(zenith, t) + glow
}

/// Whitted-style shading. Secondary rays (shadow / AO / reflection) always use
/// tmin ≈ 0 — the quadtree's inherited tmin is a primary-frustum property and
/// must never leak in here.
pub fn shade(
    scene: &Scene,
    bvh: &Bvh,
    ray: &Ray,
    hit: &Hit,
    q: &Quality,
    rng: &mut fastrand::Rng,
    sun: Vec3A,
    depth: u32,
    rays: &mut u64,
    visits: &mut u64,
    mut prim: Option<&mut PrimarySurface>,
) -> Vec3A {
    let [i0, i1, i2] = scene.indices[hit.tri as usize];
    let w = 1.0 - hit.u - hit.v;
    let mut n = (scene.normals[i0 as usize] * w
        + scene.normals[i1 as usize] * hit.u
        + scene.normals[i2 as usize] * hit.v)
        .normalize_or_zero();
    if n == Vec3A::ZERO {
        let e1 = scene.positions[i1 as usize] - scene.positions[i0 as usize];
        let e2 = scene.positions[i2 as usize] - scene.positions[i0 as usize];
        n = e1.cross(e2).normalize_or_zero();
    }
    if n.dot(ray.d) > 0.0 {
        n = -n;
    }
    let p = ray.o + ray.d * hit.t + n * scene.eps;
    let mat = &scene.materials[scene.tri_mat[hit.tri as usize] as usize];
    if let Some(prim) = prim.as_deref_mut() {
        *prim = PrimarySurface { n, albedo: mat.albedo, reflectivity: mat.reflectivity, spec_t: 0.0 };
    }

    // Direct light: N samples on the area light.
    let mut direct = Vec3A::ZERO;
    for _ in 0..q.shadow_samples {
        let lp = scene.light.center
            + scene.light.u * (rng.f32() * 2.0 - 1.0)
            + scene.light.v * (rng.f32() * 2.0 - 1.0);
        let lv = lp - p;
        let dist2 = lv.length_squared();
        let dist = dist2.sqrt();
        let wi = lv / dist;
        let ndl = n.dot(wi);
        if ndl <= 0.0 {
            continue;
        }
        *rays += 1;
        if !bvh.occluded(scene, &Ray::new(p, wi), 0.0, dist - scene.eps, visits) {
            direct += scene.light.color * (ndl / dist2);
        }
    }
    if q.shadow_samples > 0 {
        direct /= q.shadow_samples as f32;
    }

    // Ambient occlusion: short cosine-hemisphere rays.
    let mut ao = 1.0;
    if q.ao_samples > 0 {
        let (t1, t2) = onb(n);
        let mut open = 0u32;
        for _ in 0..q.ao_samples {
            let r1 = rng.f32();
            let r2 = rng.f32();
            let phi = std::f32::consts::TAU * r1;
            let sq = r2.sqrt();
            let dir = t1 * (phi.cos() * sq) + t2 * (phi.sin() * sq) + n * (1.0 - r2).sqrt();
            *rays += 1;
            if !bvh.occluded(scene, &Ray::new(p, dir), 0.0, scene.ao_radius, visits) {
                open += 1;
            }
        }
        ao = open as f32 / q.ao_samples as f32;
    }

    let mut color = mat.albedo * (direct + AMBIENT * ao);

    // One reflection bounce.
    if q.reflections && mat.reflectivity > 0.0 && depth == 0 {
        let rdir = (ray.d - n * (2.0 * ray.d.dot(n))).normalize();
        let rray = Ray::new(p, rdir);
        *rays += 1;
        let rcol = match bvh.intersect(scene, &rray, 0.0, f32::INFINITY, visits) {
            Some(rh) => {
                if let Some(prim) = prim.as_deref_mut() {
                    prim.spec_t = rh.t;
                }
                // The recursive call gets None: only the primary surface is
                // ever captured.
                shade(scene, bvh, &rray, &rh, q, rng, sun, depth + 1, rays, visits, None)
            }
            None => {
                if let Some(prim) = prim.as_deref_mut() {
                    prim.spec_t = f32::INFINITY;
                }
                sky(rdir, sun)
            }
        };
        color = color.lerp(rcol, mat.reflectivity);
    }

    color
}

/// Branchless-ish orthonormal basis around n (Duff et al.).
#[inline(always)]
fn onb(n: Vec3A) -> (Vec3A, Vec3A) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (
        Vec3A::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        Vec3A::new(b, s + n.y * n.y * a, -n.y),
    )
}
