use crate::bvh::{Bvh, Hit, Ray};
use crate::scene::{MatKind, Scene};
use crate::stats::LocalStats;
use glam::Vec3A;

/// Which bounce effects run through the hemisphere/shaft frustum integrators
/// instead of the sampled loops. `depth` is the hemisphere subdivision budget
/// (4^depth leaf cells). All-off reproduces the sampled path bit-for-bit.
#[derive(Clone, Copy, PartialEq)]
pub struct FrustumBounce {
    pub ao: bool,
    pub gi: bool,
    pub shadows: bool,
    pub depth: u32,
}

impl FrustumBounce {
    pub const OFF: FrustumBounce = FrustumBounce { ao: false, gi: false, shadows: false, depth: 0 };
}

#[derive(Clone, Copy)]
pub struct Quality {
    pub shadow_samples: u32,
    pub ao_samples: u32,
    pub reflections: bool,
    pub fb: FrustumBounce,
}

impl Quality {
    pub fn preset(i: u32) -> Quality {
        // Hemisphere budgets scale with the preset (16/64/256 leaf cells);
        // the fb flags themselves are toggled at the call site (H key /
        // --check sections), not by the preset.
        match i {
            1 => Quality {
                shadow_samples: 1,
                ao_samples: 0,
                reflections: false,
                fb: FrustumBounce { depth: 2, ..FrustumBounce::OFF },
            },
            3 => Quality {
                shadow_samples: 4,
                ao_samples: 4,
                reflections: true,
                fb: FrustumBounce { depth: 4, ..FrustumBounce::OFF },
            },
            _ => Quality {
                shadow_samples: 2,
                ao_samples: 2,
                reflections: true,
                fb: FrustumBounce { depth: 3, ..FrustumBounce::OFF },
            },
        }
    }

    /// The pinned upscaler quality: every DLSS/XeSS/GPU-upscaler frame is a
    /// fresh 1-spp frame (1 shadow / 1 AO / reflections, fb OFF —
    /// frame-stationary noise for the temporal integrator to launder);
    /// presets don't apply. Single-sourced so the contract can't drift
    /// between the present arms, --spin, and the check gates.
    pub fn upscaler_1spp() -> Quality {
        Quality {
            shadow_samples: 1,
            ao_samples: 1,
            reflections: true,
            fb: FrustumBounce::OFF,
        }
    }

    /// Cheap variant used while the camera is moving; accumulation converges
    /// the full quality once the camera is still. Frustum bounces stay off
    /// here — they are a still-frame quality feature (v1).
    pub fn while_moving(&self) -> Quality {
        Quality {
            shadow_samples: 1,
            ao_samples: 0,
            reflections: self.reflections,
            fb: FrustumBounce::OFF,
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
    /// Perceptual roughness and metalness, straight from the material.
    pub roughness: f32,
    pub metallic: f32,
    /// Specular-reflection ray hit t; INFINITY when the reflection ray
    /// missed; 0.0 when no reflection was traced.
    pub spec_t: f32,
    /// Shadowed direct light, split by lobe, AFTER the sample average —
    /// exactly the two addends of `color = kd*(direct_d + ambient) +
    /// direct_s`. `direct_d` is albedo-free (kd multiplies it later), i.e.
    /// already the demodulated diffuse radiance FSR Ray Regeneration wants;
    /// `direct_s` includes the per-sample Fresnel. Everything else the pixel
    /// shows (kd*ambient, the reflection bounce) is reconstructed at the
    /// G-buffer write site as the exact residual `color - kd*direct_d -
    /// direct_s`-style remainder, so capture changes no shading math.
    pub direct_d: Vec3A,
    pub direct_s: Vec3A,
}

/// Capacity of a `VisRecord` (the presets top out at 4 shadow samples).
pub const VIS_MAX: usize = 8;

/// Captured view-independent visibility of one shading point: the light
/// sample points with their occlusion results, plus the sampled-AO scalar.
/// The adaptive 2×2 cells (render.rs, XeSS mode) trace this once at a
/// representative pixel and re-apply it across the cell — sharing the RAYS
/// only; every view-dependent term (N·L, albedo, GGX specular, the
/// reflection bounce) stays per-pixel.
#[derive(Clone, Copy)]
pub struct VisRecord {
    /// Light sample offsets (su, sv) in [-1,1]² — Apply pixels shade toward
    /// the SAME light points, so the only approximation is the shadow-ray
    /// origin shift within the cell (sub-pixel in world scale).
    pub light_uv: [(f32, f32); VIS_MAX],
    /// Occlusion per light sample. Below-horizon capture samples record
    /// occluded AND poison `uniform`: that occlusion was never traced (the
    /// rep's own N·L rejected it), so it must not be replayed onto a
    /// neighbor whose horizon differs — the terminator is a declassify
    /// signal exactly like penumbra.
    pub occluded: [bool; VIS_MAX],
    pub n_light: u32,
    /// Sampled-AO open fraction.
    pub ao: f32,
    /// Every light sample agreed (all lit / all blocked) and every one was
    /// actually traced — sharing is then near-exact. Fractional visibility
    /// means penumbra, a below-horizon sample means the terminator; both
    /// make the cell fall back to per-pixel rays (fractional visibility is
    /// only meaningful with >= 2 samples; one sample is trivially uniform).
    pub uniform: bool,
    /// Any capture sample fell below the rep's horizon (untraced occlusion).
    pub below_horizon: bool,
}

impl Default for VisRecord {
    fn default() -> Self {
        Self {
            light_uv: [(0.0, 0.0); VIS_MAX],
            occluded: [false; VIS_MAX],
            n_light: 0,
            ao: 1.0,
            uniform: false,
            below_horizon: false,
        }
    }
}

/// How `shade` treats the view-independent visibility rays (shadow + AO).
/// Off is the pre-adaptive behavior, bit-for-bit. Only the sampled paths
/// consult this — the frustum-bounce tiers (fb.*) never run under
/// Capture/Apply (adaptive is XeSS-mode-only, where fb is OFF).
pub enum VisCtl<'a> {
    Off,
    /// Trace normally, recording light UVs + occlusion + AO into the target.
    Capture(&'a mut VisRecord),
    /// Skip the occlusion/AO rays; reuse the record's results toward the
    /// same light points with this pixel's own geometry.
    Apply(&'a VisRecord),
}

/// Interpolated, face-flipped shading normal at a hit and the eps-offset
/// point secondary rays start from — shared by `shade` and the hemisphere
/// integrator's verification probes (they must agree exactly).
pub fn surface_point(scene: &Scene, ray: &Ray, hit: &Hit) -> (Vec3A, Vec3A) {
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
    (ray.o + ray.d * hit.t + n * scene.eps, n)
}

/// Cosine-weighted hemisphere direction from two uniform draws in a
/// right-handed tangent frame — THE construction shared by the sampled-AO
/// loop and `--check`'s A/B reference estimators (the gates compare
/// like-for-like only while every site draws identically).
#[inline(always)]
pub(crate) fn cosine_dir(n: Vec3A, t1: Vec3A, t2: Vec3A, r1: f32, r2: f32) -> Vec3A {
    let phi = std::f32::consts::TAU * r1;
    let sq = r2.sqrt();
    t1 * (phi.cos() * sq) + t2 * (phi.sin() * sq) + n * (1.0 - r2).sqrt()
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
#[allow(clippy::too_many_arguments)]
pub fn shade(
    scene: &Scene,
    bvh: &Bvh,
    ray: &Ray,
    hit: &Hit,
    sp: Option<(Vec3A, Vec3A)>,
    q: &Quality,
    rng: &mut fastrand::Rng,
    sun: Vec3A,
    depth: u32,
    ls: &mut LocalStats,
    mut prim: Option<&mut PrimarySurface>,
    mut vis: VisCtl,
    hemi_share: Option<&crate::hemi::HemiShare>,
) -> Vec3A {
    // Capture/Apply only exist for the sampled shadow/AO paths; the
    // frustum-bounce tiers would silently bypass the record.
    debug_assert!(
        matches!(vis, VisCtl::Off) || (!q.fb.shadows && !q.fb.ao && !q.fb.gi),
        "VisCtl requires the sampled shadow/AO paths (fb OFF)"
    );
    // A shared hemisphere root only means something to the fb tiers (fb
    // already implies VisCtl::Off above — the two records never mix).
    debug_assert!(
        hemi_share.is_none() || q.fb.ao || q.fb.gi,
        "hemi_share requires a frustum-bounce tier (fb.ao or fb.gi)"
    );
    debug_assert!(q.shadow_samples as usize <= VIS_MAX);
    // `sp` is the caller's precomputed surface_point(scene, ray, hit) — the
    // adaptive cell already evaluated it for the coherence test and must not
    // pay the triangle fetch + interpolation twice per pixel.
    let (p, n) = sp.unwrap_or_else(|| surface_point(scene, ray, hit));
    let mat = &scene.materials[scene.tri_mat[hit.tri as usize] as usize];
    // Effective albedo: constant, except marble which is evaluated at the
    // world-space hit point (models are static, so world space is stable).
    let albedo = match mat.kind {
        MatKind::Marble { scale } => marble(ray.o + ray.d * hit.t, scale),
        _ => mat.albedo,
    };
    if let Some(prim) = prim.as_deref_mut() {
        *prim = PrimarySurface {
            n,
            albedo,
            roughness: mat.roughness,
            metallic: mat.metallic,
            spec_t: 0.0,
            direct_d: Vec3A::ZERO,
            direct_s: Vec3A::ZERO,
        };
    }

    // Metallic/roughness lobes: F0 = 4% dielectric base lerped to albedo for
    // metals; the diffuse lobe fades out as metalness rises.
    let f0 = Vec3A::splat(0.04).lerp(albedo, mat.metallic);
    let kd = albedo * (1.0 - mat.metallic);
    // Tangent frame for the microfacet lobes. Anisotropic materials brush
    // circumferentially around world-up (a lathe-spun body); the onb fallback
    // covers the poles and all isotropic materials (frame arbitrary there).
    let (t1, t2) = if mat.anisotropy > 0.0 {
        let t = Vec3A::Y.cross(n);
        if t.length_squared() > 1e-8 {
            let t = t.normalize();
            (t, n.cross(t))
        } else {
            onb(n)
        }
    } else {
        onb(n)
    };
    let to_local = |w: Vec3A| Vec3A::new(w.dot(t1), w.dot(t2), w.dot(n));
    let (ax, ay) = ggx_alphas(mat.roughness, mat.anisotropy);
    let v = -ray.d;
    let vl = {
        let mut l = to_local(v);
        l.z = l.z.max(1e-4); // face-flip guarantees n·v >= 0; guard grazing
        l
    };
    let lambda_v = ggx_lambda(vl, ax, ay);

    // Direct light: N samples on the area light, Lambert diffuse +
    // Cook-Torrance GGX specular per sample. The renderer's convention omits
    // Lambert's 1/π (the light intensity absorbs it), so the specular term
    // carries the compensating π — the diffuse:specular ratio stays physical
    // without retuning scene brightness.
    let mut direct_d = Vec3A::ZERO;
    let mut direct_s = Vec3A::ZERO;
    // Light-shaft culling (fb.shadows): identical sampling and integrand —
    // the shaft only removes occlusion rays for samples in subrects proven
    // unoccluded, and seeds the remaining (penumbra) rays from its cut with
    // its own apex-relative tmin. Built lazily: a light fully behind the
    // surface never pays for a shaft.
    let mut shaft: Option<crate::shaft::Shaft> = None;
    // Under Apply the loop runs over the record's samples (the SAME light
    // points its capture drew); Capture/Off draw fresh points from the rng.
    let n_shadow = match &vis {
        VisCtl::Apply(r) => r.n_light,
        _ => q.shadow_samples,
    };
    for si in 0..n_shadow as usize {
        let (su, sv) = match &vis {
            VisCtl::Apply(r) => {
                // Burn the two draws Capture made for this sample — the rng
                // stream must stay aligned with a non-adaptive frame or the
                // GGX reflection draws below diverge (the spec_t / G-buffer
                // bit-identity contract).
                let _ = (rng.f32(), rng.f32());
                r.light_uv[si]
            }
            _ => (rng.f32() * 2.0 - 1.0, rng.f32() * 2.0 - 1.0),
        };
        let lp = scene.light.center + scene.light.u * su + scene.light.v * sv;
        let lv = lp - p;
        let dist2 = lv.length_squared();
        let dist = dist2.sqrt();
        let wi = lv / dist;
        let ndl = n.dot(wi);
        if ndl <= 0.0 {
            // Below the rep's horizon: no ray was traced, so this "occluded"
            // is a claim about the rep's normal, not the scene. Mark the
            // record so `uniform` fails and the cell declassifies — replaying
            // it onto a neighbor whose own N·L is positive would zero direct
            // light the neighbor actually receives (terminator darkening).
            if let VisCtl::Capture(r) = &mut vis {
                r.light_uv[si] = (su, sv);
                r.occluded[si] = true;
                r.below_horizon = true;
            }
            continue;
        }
        let occluded = match &vis {
            VisCtl::Apply(r) => {
                ls.adapt_rays_saved += 1;
                r.occluded[si]
            }
            _ if q.fb.shadows => {
                if shaft.is_none() {
                    shaft = Some(crate::shaft::build(scene, bvh, p, n, ls));
                }
                match shaft.as_ref().unwrap().classify(su, sv) {
                    crate::shaft::Class::Lit => {
                        ls.shaft_rays_skipped += 1;
                        false
                    }
                    crate::shaft::Class::Test { tmin, cut } => {
                        ls.secondary_rays += 1;
                        bvh.occluded_multi(
                            scene,
                            &Ray::new(p, wi),
                            tmin,
                            dist - scene.eps,
                            cut,
                            &mut ls.ray_nodes,
                        )
                    }
                }
            }
            _ => {
                ls.secondary_rays += 1;
                bvh.occluded(scene, &Ray::new(p, wi), 0.0, dist - scene.eps, &mut ls.ray_nodes)
            }
        };
        if let VisCtl::Capture(r) = &mut vis {
            r.light_uv[si] = (su, sv);
            r.occluded[si] = occluded;
        }
        if !occluded {
            let li = scene.light.color * (ndl / dist2);
            direct_d += li;
            let h = (wi + v).normalize_or_zero();
            let hl = to_local(h);
            if hl.z > 0.0 {
                let d = ggx_ndf(hl, ax, ay);
                let g2 = 1.0 / (1.0 + lambda_v + ggx_lambda(to_local(wi), ax, ay));
                let f = schlick(f0, wi.dot(h).max(0.0));
                // li carries ndl; D·G2·F/(4·nv·nl)·nl leaves /(4·nv).
                direct_s += li * f * (std::f32::consts::PI * d * g2 / (4.0 * vl.z * ndl));
            }
        }
    }
    if n_shadow > 0 {
        direct_d /= n_shadow as f32;
        direct_s /= n_shadow as f32;
    }
    if let Some(prim) = prim.as_deref_mut() {
        prim.direct_d = direct_d;
        prim.direct_s = direct_s;
    }
    if let VisCtl::Capture(r) = &mut vis {
        r.n_light = n_shadow;
        let k = n_shadow as usize;
        r.uniform = k == 0
            || (!r.below_horizon && r.occluded[..k].iter().all(|&o| o == r.occluded[0]));
    }

    // Diffuse ambient term. Three tiers, all through the hemisphere's OWN
    // apex-relative tmin chain when frustum-dispatched (the primary tile's
    // tmin is never involved):
    // - fb.gi: real sky+bounce irradiance/π over the hemisphere (subsumes AO —
    //   occluders contribute their radiance instead of darkening a constant).
    // - fb.ao: the AMBIENT constant modulated by frustum-dispatched AO.
    // - neither: the AMBIENT constant modulated by sampled AO.
    let ambient = if q.fb.gi {
        let (t1, t2) = onb(n);
        crate::hemi::gi(scene, bvh, p, n, t1, t2, q.fb.depth, sun, depth, hemi_share, rng, None, ls)
    } else {
        let mut ao = 1.0;
        if q.fb.ao {
            let (t1, t2) = onb(n);
            ao = crate::hemi::ao(
                scene,
                bvh,
                p,
                n,
                t1,
                t2,
                q.fb.depth,
                scene.ao_radius,
                hemi_share,
                rng,
                None,
                ls,
            );
        } else if q.ao_samples > 0 {
            if let VisCtl::Apply(r) = &vis {
                // AO is low-frequency: the shared scalar is reused outright
                // (unlike shadows there is no uniformity gate — a fractional
                // AO is its normal state, not a penumbra signal). Burn the
                // capture path's draws to keep the rng stream aligned (the
                // spec_t bit-identity contract, as in the shadow loop).
                ao = r.ao;
                for _ in 0..q.ao_samples {
                    let _ = (rng.f32(), rng.f32());
                }
                ls.adapt_rays_saved += q.ao_samples as u64;
            } else {
                let (t1, t2) = onb(n);
                let mut open = 0u32;
                for _ in 0..q.ao_samples {
                    let r1 = rng.f32();
                    let r2 = rng.f32();
                    let dir = cosine_dir(n, t1, t2, r1, r2);
                    ls.secondary_rays += 1;
                    if !bvh.occluded(
                        scene,
                        &Ray::new(p, dir),
                        0.0,
                        scene.ao_radius,
                        &mut ls.ray_nodes,
                    ) {
                        open += 1;
                    }
                }
                ao = open as f32 / q.ao_samples as f32;
                if let VisCtl::Capture(r) = &mut vis {
                    r.ao = ao;
                }
            }
        }
        AMBIENT * ao
    };

    // Ambient stays diffuse-only; metals get their environment from the
    // specular bounce ray below.
    let mut color = kd * (direct_d + ambient) + direct_s;

    // One specular bounce: a single direction importance-sampled from the
    // anisotropic GGX VNDF (Heitz 2018), so glossy surfaces see a blurred
    // environment that accumulation / DLSS-RR converges. Throughput is
    // F·G2/G1 (≤ 1 per channel — no fireflies possible). roughness → 0
    // degenerates to the old exact mirror. The gate skips near-Lambertian
    // dielectrics whose specular contribution wouldn't justify a ray.
    if q.reflections && depth == 0 && (mat.metallic > 0.04 || mat.roughness < 0.45) {
        let vh = Vec3A::new(ax * vl.x, ay * vl.y, vl.z).normalize();
        let lensq = vh.x * vh.x + vh.y * vh.y;
        let b1 = if lensq > 0.0 {
            Vec3A::new(-vh.y, vh.x, 0.0) / lensq.sqrt()
        } else {
            Vec3A::X
        };
        let b2 = vh.cross(b1);
        let r = rng.f32().sqrt();
        let phi = std::f32::consts::TAU * rng.f32();
        let p1 = r * phi.cos();
        let mut p2 = r * phi.sin();
        let s = 0.5 * (1.0 + vh.z);
        p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;
        let nh = b1 * p1 + b2 * p2 + vh * (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
        let hl = Vec3A::new(ax * nh.x, ay * nh.y, nh.z.max(1e-6)).normalize();
        let h = t1 * hl.x + t2 * hl.y + n * hl.z;
        let rdir = (2.0 * v.dot(h) * h - v).normalize();
        // Below-horizon samples are dropped (spec_t stays 0.0 = "no
        // reflection traced"), slightly darkening instead of biasing up.
        if rdir.dot(n) > 0.0 {
            let g2_over_g1 = (1.0 + lambda_v)
                / (1.0 + lambda_v + ggx_lambda(to_local(rdir), ax, ay));
            let tput = schlick(f0, v.dot(h).max(0.0)) * g2_over_g1;
            let rray = Ray::new(p, rdir);
            ls.secondary_rays += 1;
            let rcol = match bvh.intersect(scene, &rray, 0.0, f32::INFINITY, &mut ls.ray_nodes) {
                Some(rh) => {
                    if let Some(prim) = prim.as_deref_mut() {
                        prim.spec_t = rh.t;
                    }
                    // The recursive call gets None: only the primary surface
                    // is ever captured. Frustum bounces don't recurse: the
                    // reflected hit shades with fb off (the sampled ambient
                    // path), mirroring hemi's recursion-free leaf policy —
                    // otherwise every glossy pixel would pay a second full
                    // hemisphere integration (and shaft build) at depth 1.
                    let rq = Quality { fb: FrustumBounce::OFF, ..*q };
                    shade(scene, bvh, &rray, &rh, None, &rq, rng, sun, depth + 1, ls, None, VisCtl::Off, None)
                }
                None => {
                    if let Some(prim) = prim.as_deref_mut() {
                        prim.spec_t = f32::INFINITY;
                    }
                    sky(rdir, sun)
                }
            };
            color += tput * rcol;
        }
    }

    color
}

/// GGX α from perceptual roughness + Disney-style anisotropy split. The
/// floor keeps the NDF finite for near-mirror materials.
#[inline(always)]
fn ggx_alphas(roughness: f32, anisotropy: f32) -> (f32, f32) {
    const MIN_ALPHA: f32 = 5e-3;
    let alpha = roughness * roughness;
    let aspect = (1.0 - 0.9 * anisotropy).sqrt();
    ((alpha / aspect).max(MIN_ALPHA), (alpha * aspect).max(MIN_ALPHA))
}

/// Smith Λ for anisotropic GGX; `w` is in tangent space with w.z > 0.
/// G1 = 1/(1+Λ), height-correlated G2 = 1/(1+Λv+Λl).
#[inline(always)]
fn ggx_lambda(w: Vec3A, ax: f32, ay: f32) -> f32 {
    let t = ((ax * w.x) * (ax * w.x) + (ay * w.y) * (ay * w.y)) / (w.z * w.z);
    ((1.0 + t).sqrt() - 1.0) * 0.5
}

/// Anisotropic GGX NDF; `h` is the half vector in tangent space.
#[inline(always)]
fn ggx_ndf(h: Vec3A, ax: f32, ay: f32) -> f32 {
    let hx = h.x / ax;
    let hy = h.y / ay;
    let d = hx * hx + hy * hy + h.z * h.z;
    1.0 / (std::f32::consts::PI * ax * ay * d * d)
}

/// Schlick Fresnel.
#[inline(always)]
fn schlick(f0: Vec3A, cos: f32) -> Vec3A {
    f0 + (Vec3A::ONE - f0) * (1.0 - cos).clamp(0.0, 1.0).powi(5)
}

/// Procedural marble: white base cut by thin dark veins where a
/// turbulence-perturbed sine crosses zero. Deterministic (hash-based value
/// noise), evaluated in world space; `scale` sets the feature frequency.
fn marble(p: Vec3A, scale: f32) -> Vec3A {
    const BASE: Vec3A = Vec3A::new(0.93, 0.92, 0.90);
    const VEIN: Vec3A = Vec3A::new(0.10, 0.11, 0.15);
    let q = p * scale;
    let s = (q.x + 0.7 * q.y + 5.0 * fbm(q)).sin();
    // Smoothstep over a narrow band around the zero crossing: full VEIN
    // below the inner edge, full BASE above the outer, so the surface reads
    // as white stone cut by crisp dark lines instead of a gray gradient.
    let t = ((s.abs() - 0.04) / 0.18).clamp(0.0, 1.0);
    VEIN.lerp(BASE, t * t * (3.0 - 2.0 * t))
}

fn fbm(mut p: Vec3A) -> f32 {
    let mut amp = 0.5;
    let mut sum = 0.0;
    for _ in 0..5 {
        sum += amp * vnoise(p);
        p *= 2.02;
        amp *= 0.5;
    }
    sum
}

/// Trilinearly interpolated hash-lattice value noise in [0,1).
fn vnoise(p: Vec3A) -> f32 {
    let f = p.floor();
    let (ix, iy, iz) = (f.x as i32, f.y as i32, f.z as i32);
    let t = p - f;
    let s = t * t * (Vec3A::splat(3.0) - 2.0 * t);
    let c = |dx, dy, dz| hash3(ix + dx, iy + dy, iz + dz);
    let l = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = l(c(0, 0, 0), c(1, 0, 0), s.x);
    let x10 = l(c(0, 1, 0), c(1, 1, 0), s.x);
    let x01 = l(c(0, 0, 1), c(1, 0, 1), s.x);
    let x11 = l(c(0, 1, 1), c(1, 1, 1), s.x);
    l(l(x00, x10, s.y), l(x01, x11, s.y), s.z)
}

#[inline(always)]
fn hash3(x: i32, y: i32, z: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x9e37_79b1);
    h ^= h >> 16;
    (h & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Branchless-ish orthonormal basis around n (Duff et al.). Right-handed:
/// t1 × t2 = n — the hemisphere integrator's octant orientation relies on it
/// (asserted by sphcell::self_test).
#[inline(always)]
pub(crate) fn onb(n: Vec3A) -> (Vec3A, Vec3A) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (
        Vec3A::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        Vec3A::new(b, s + n.y * n.y * a, -n.y),
    )
}
