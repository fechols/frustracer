use crate::bvh::{Bvh, Hit, Ray};
use crate::scene::{MatKind, Scene, NO_TEX};
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

/// The constant ambient radiance the sampled/fb.ao tiers modulate by AO.
/// `pub` because it is the modulation factor FSR Ray Regeneration's AO signal
/// is remodulated by (fsr.rs's composite; the HLSL twin is shade.hlsli's
/// AMBIENT, and the FSR composite pass takes it as a root constant).
pub const AMBIENT: Vec3A = Vec3A::new(0.14, 0.17, 0.23);

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
    /// Open fraction of the ambient-occlusion tier, i.e. the `ao` that
    /// `ambient = AMBIENT * ao` was built from — FSR Ray Regeneration's
    /// AMBIENT_OCCLUSION signal (0 = fully occluded, 1 = fully exposed).
    /// Stays 0.0 under `fb.gi`, whose ambient is real RGB irradiance and not
    /// an AO-modulated constant: the composite then adds nothing and the
    /// residual absorbs the whole GI term, exactly as before this signal
    /// existed. (fb is pinned OFF in upscaler frames, so that is a
    /// correctness fallback, not the live path.)
    pub ao: f32,
    /// The specular bounce's contribution to `color` — `tput * rcol`, i.e.
    /// the whole reflection subtree including any glass continuation behind
    /// it. FSR's INDIRECT_SPECULAR signal (demodulated by F0 at the split,
    /// exactly like `direct_s`); its ray hit distance is `spec_t`. Zero when
    /// no reflection was traced.
    pub ind_s: Vec3A,
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
    // world-space hit point (models are static, so world space is stable)
    // and textures, sampled at the hit's interpolated UV.
    let albedo = match mat.kind {
        MatKind::Marble { scale } => marble(ray.o + ray.d * hit.t, scale),
        MatKind::Textured { tex } => {
            let uv = scene.tri_uv(hit.tri, hit.u, hit.v);
            scene.textures[tex as usize].sample_bilinear(uv.x, uv.y)
        }
        _ => mat.albedo,
    };
    // Map-driven material terms — all pure ALU on the hit's UV, ZERO rng
    // draws (materials with every map at NO_TEX shade bit-identically to
    // before the map fields existed — the structural guarantee, and what
    // keeps replay/same-seed/VisCtl burn accounting intact).
    let map_uv = if mat.normal_tex != NO_TEX
        || mat.rough_tex != NO_TEX
        || mat.metal_tex != NO_TEX
        || mat.emissive_tex != NO_TEX
    {
        Some(scene.tri_uv(hit.tri, hit.u, hit.v))
    } else {
        None
    };
    // Effective roughness/metallic = flat factor × map sample (glTF
    // semantics; roughness = .g, metallic = .b). The clamps live INSIDE the
    // map branch so unmapped materials keep their exact flat values.
    let mut rough_eff = mat.roughness;
    let mut metal_eff = mat.metallic;
    if let Some(uv) = map_uv {
        if mat.rough_tex != NO_TEX {
            let s = scene.textures[mat.rough_tex as usize].sample_bilinear_linear(uv.x, uv.y);
            rough_eff = (rough_eff * s.y).clamp(0.02, 1.0);
        }
        if mat.metal_tex != NO_TEX {
            let s = scene.textures[mat.metal_tex as usize].sample_bilinear_linear(uv.x, uv.y);
            metal_eff = (metal_eff * s.z).clamp(0.0, 1.0);
        }
    }
    // Shading normal n_s: the geometric n perturbed by the tangent-space
    // normal map; n_s ≡ n when unmapped (structural bit-identity). n keeps
    // every visibility-adjacent use — the eps-offset p, the translucency
    // back ray, the ENTIRE hemi tier (a perturbed apex normal can put the
    // own triangle inside the "open" hemisphere ⇒ false-empty), and the
    // glass chain. n_s feeds the BRDF frame, N·L, and the G-buffer guide.
    let n_s = match (mat.normal_tex != NO_TEX, map_uv) {
        (true, Some(uv)) => perturb_normal(scene, hit, n, mat, uv),
        _ => n,
    };
    if let Some(prim) = prim.as_deref_mut() {
        *prim = PrimarySurface {
            n: n_s,
            albedo,
            roughness: rough_eff,
            metallic: metal_eff,
            spec_t: 0.0,
            direct_d: Vec3A::ZERO,
            direct_s: Vec3A::ZERO,
            // Both are filled below, when their tier runs; the zeros are the
            // "no such term" values the composite adds nothing for (fb.gi's
            // ambient, or a surface whose reflection gate never fired).
            ao: 0.0,
            ind_s: Vec3A::ZERO,
        };
    }

    // Metallic/roughness lobes: F0 = 4% dielectric base lerped to albedo for
    // metals; the diffuse lobe fades out as metalness rises.
    let f0 = Vec3A::splat(0.04).lerp(albedo, metal_eff);
    // The sheen factor keeps fabric energy-conserving: 0.157 is the Charlie
    // lobe's peak directional albedo (Estevez-Kulla), so diffuse gives back
    // what the sheen adds.
    let kd = albedo * (1.0 - metal_eff) * (1.0 - 0.157 * mat.sheen);
    // Tangent frame for the microfacet lobes, built on the SHADING normal.
    // Anisotropic materials brush circumferentially around world-up (a
    // lathe-spun body); the onb fallback covers the poles and all isotropic
    // materials (frame arbitrary there).
    let (t1, t2) = if mat.anisotropy > 0.0 {
        let t = Vec3A::Y.cross(n_s);
        if t.length_squared() > 1e-8 {
            let t = t.normalize();
            (t, n_s.cross(t))
        } else {
            onb(n_s)
        }
    } else {
        onb(n_s)
    };
    let to_local = |w: Vec3A| Vec3A::new(w.dot(t1), w.dot(t2), w.dot(n_s));
    let (ax, ay) = ggx_alphas(rough_eff, mat.anisotropy);
    let v = -ray.d;
    let vl = {
        let mut l = to_local(v);
        // The face-flip guarantees n·v >= 0 for the GEOMETRIC normal; a
        // perturbed n_s can dip below — the same grazing guard covers both.
        l.z = l.z.max(1e-4);
        l
    };
    let lambda_v = ggx_lambda(vl, ax, ay);
    // Charlie-sheen inverse alpha, hoisted out of the light loop (fabric
    // reuses the material roughness as sheen roughness).
    let sheen_inv_a = 1.0 / rough_eff.clamp(0.07, 1.0);

    // Direct light: N samples on the area light, Lambert diffuse +
    // Cook-Torrance GGX specular per sample. The renderer's convention omits
    // Lambert's 1/π (the light intensity absorbs it), so the specular term
    // carries the compensating π — the diffuse:specular ratio stays physical
    // without retuning scene brightness.
    let mut direct_d = Vec3A::ZERO;
    let mut direct_s = Vec3A::ZERO;
    // Thin-surface diffuse transmission (foliage): light arriving from BEHIND
    // the surface, gathered in the ndl <= 0 arm below.
    let mut direct_t = Vec3A::ZERO;
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
        // N·L against the SHADING normal (n_s ≡ n when unmapped); the
        // shadow/translucency ray geometry below stays on the geometric n.
        let ndl = n_s.dot(wi);
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
            // Thin-surface transmission: a back-lit translucent surface
            // (leaves) receives the light through itself. The occlusion ray
            // starts on the TRANSMITTED side (p is hit + n·eps, so -2·eps
            // lands at hit - n·eps — the exact mirror of the front
            // convention; the ray departs the leaf's plane on the side it
            // starts on and never re-crosses its own triangle). Plain
            // `occluded`: no cut exists for this apex, and the shaft's
            // tangent-plane clip makes `classify` valid only for wi·n > 0.
            // Consumes no rng draws — stream alignment is untouched.
            if mat.translucency > 0.0 && ndl < 0.0 {
                let back_occluded = match &vis {
                    // The rep's traced bit is segment occlusion between the
                    // same two points — normal-independent within 2·eps.
                    VisCtl::Apply(r) => {
                        ls.adapt_rays_saved += 1;
                        r.occluded[si]
                    }
                    _ => {
                        ls.secondary_rays += 1;
                        bvh.occluded(
                            scene,
                            &Ray::new(p - n * (2.0 * scene.eps), wi),
                            0.0,
                            dist - scene.eps,
                            &mut ls.ray_nodes,
                        )
                    }
                };
                if !back_occluded {
                    direct_t += scene.light.color * ((-ndl) / dist2);
                }
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
                if mat.sheen > 0.0 {
                    // Retro-reflective fabric rim: Charlie NDF + Ashikhmin
                    // visibility (the glTF KHR_materials_sheen pair), white,
                    // direct light only. Pure ALU on values already in
                    // scope — no rng, no rays. The π compensates the
                    // renderer's dropped Lambert 1/π like the GGX term.
                    let sin2 = (1.0 - hl.z * hl.z).max(0.0);
                    let d_c = (2.0 + sheen_inv_a) * sin2.powf(sheen_inv_a * 0.5)
                        / std::f32::consts::TAU;
                    let v_ash = 1.0 / (4.0 * (ndl + vl.z - ndl * vl.z)).max(1e-4);
                    direct_s += li * (std::f32::consts::PI * mat.sheen * d_c * v_ash);
                }
            }
        }
    }
    if n_shadow > 0 {
        direct_d /= n_shadow as f32;
        direct_s /= n_shadow as f32;
        direct_t /= n_shadow as f32;
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
        let accel = crate::ftree::Accel::of(bvh);
        crate::hemi::gi(scene, accel, p, n, t1, t2, q.fb.depth, sun, depth, hemi_share, rng, None, ls)
    } else {
        let mut ao = 1.0;
        if q.fb.ao {
            let (t1, t2) = onb(n);
            ao = crate::hemi::ao(
                scene,
                crate::ftree::Accel::of(bvh),
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
        // FSR's AO signal: the open fraction itself, before the constant.
        // Assignment-only (no rng draw), so the same-seed bit-identity gates
        // are untouched.
        if let Some(prim) = prim.as_deref_mut() {
            prim.ao = ao;
        }
        AMBIENT * ao
    };

    // Ambient stays diffuse-only; metals get their environment from the
    // specular bounce ray below. Translucency splits the diffuse budget
    // between the front Lambert term and the transmitted back term
    // (energy-conserving; ambient stays front-only). kd is the sampled
    // textured albedo, so back-lit leaves glow in their own colors.
    // Transmissive glass has (almost) no diffuse response — the transmitted
    // scene replaces it (the refraction block below); the GGX highlight
    // stays unscaled.
    let tl = mat.translucency;
    let mut color = kd
        * (1.0 - mat.transmission)
        * (direct_d * (1.0 - tl) + direct_t * tl + ambient)
        + direct_s;

    // Emitted radiance — additive, OUTSIDE the kd·(1−transmission) factor,
    // at every depth (so emitters appear in reflections and through glass).
    // Guarded, not an unconditional `+ ZERO`: -0.0 + 0.0 = +0.0 would break
    // the default-material bit-identity contract. Emitters do NOT light
    // other surfaces — only the analytic area light + sky do.
    if mat.emissive != Vec3A::ZERO || mat.emissive_tex != NO_TEX {
        let e = match (mat.emissive_tex != NO_TEX, map_uv) {
            (true, Some(uv)) => {
                mat.emissive
                    * scene.textures[mat.emissive_tex as usize].sample_bilinear(uv.x, uv.y)
            }
            _ => mat.emissive,
        };
        color += e;
    }

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
        let h = t1 * hl.x + t2 * hl.y + n_s * hl.z;
        let rdir = (2.0 * v.dot(h) * h - v).normalize();
        // Below-horizon samples are dropped (spec_t stays 0.0 = "no
        // reflection traced"), slightly darkening instead of biasing up.
        // BOTH horizons: n_s (the sampled lobe's own frame) and the
        // geometric n — a perturbed lobe must not fire a ray that starts at
        // hit + eps·n but immediately re-enters the surface. Degenerates to
        // the old single test when n_s ≡ n.
        if rdir.dot(n_s) > 0.0 && rdir.dot(n) > 0.0 {
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
            let refl = tput * rcol;
            // FSR's INDIRECT_SPECULAR signal — the reflection subtree's whole
            // contribution (the recursive shade above already folded in any
            // glass continuation behind the mirror). Assignment-only.
            if let Some(prim) = prim.as_deref_mut() {
                prim.ind_s = refl;
            }
            color += refl;
        }
    }

    // Glass transmission: a Snell-refracted continuation ray per interface,
    // shading-only — visibility is untouched (glass still HITS; the frustum
    // bounds, inherited tmin, and every temporal claim stay exact). Placed
    // after the VNDF block so no existing rng draw moves; this block draws
    // none itself (both branch directions are pure functions of the hit, so
    // replay/same-seed bit-identity and VisCtl burn accounting hold). The
    // Fresnel-REFLECTED fraction at depth 0 is the VNDF bounce above (glass
    // passes its gate via roughness < 0.45); at interior interfaces it is
    // dropped — dimming, never gaining. Total internal reflection continues
    // the chain as an internal mirror bounce instead of losing the energy
    // (dead-black rims otherwise).
    if q.reflections && mat.transmission > 0.0 && depth < TRANS_MAX_DEPTH {
        // Entering or exiting? Re-derive the pre-flip normal orientation
        // (surface_point returns only the viewer-facing normal; transmissive
        // pixels are rare enough that the refetch beats widening its
        // contract everywhere).
        let [i0, i1, i2] = scene.indices[hit.tri as usize];
        let w = 1.0 - hit.u - hit.v;
        let mut n_raw = (scene.normals[i0 as usize] * w
            + scene.normals[i1 as usize] * hit.u
            + scene.normals[i2 as usize] * hit.v)
            .normalize_or_zero();
        if n_raw == Vec3A::ZERO {
            let e1 = scene.positions[i1 as usize] - scene.positions[i0 as usize];
            let e2 = scene.positions[i2 as usize] - scene.positions[i0 as usize];
            n_raw = e1.cross(e2).normalize_or_zero();
        }
        let entering = n_raw.dot(n) >= 0.0; // the viewer-flip didn't fire
        let eta = if entering { 1.0 / GLASS_IOR } else { GLASS_IOR };
        // The refraction chain stays on the GEOMETRIC normal (normal-mapped
        // glass is out of scope; a perturbed Snell axis would bend rays into
        // the surface). v·n with the same grazing guard is bit-identical to
        // the old n-frame vl.z.
        let cos_i = v.dot(n).max(1e-4);
        let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
        let hit_p = ray.o + ray.d * hit.t;
        let (tdir, torig, tput) = if k >= 0.0 {
            // Exact unpolarized dielectric Fresnel (not Schlick — it must
            // reach 1 as k -> 0 or the TIR handoff pops).
            let cos_t = k.sqrt();
            let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
            let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
            let fr = 0.5 * (rs * rs + rp * rp);
            let td = (ray.d * eta + n * (eta * cos_i - cos_t)).normalize();
            (td, hit_p - n * scene.eps, mat.transmission * (1.0 - fr))
        } else {
            // TIR: mirror about n, staying on the incident side.
            (ray.d + n * (2.0 * cos_i), hit_p + n * scene.eps, mat.transmission)
        };
        if tput > 1e-3 {
            let tray = Ray::new(torig, tdir);
            ls.secondary_rays += 1;
            let rq = Quality { fb: FrustumBounce::OFF, ..*q };
            let tcol =
                match bvh.intersect(scene, &tray, 0.0, f32::INFINITY, &mut ls.ray_nodes) {
                    Some(th) => shade(
                        scene, bvh, &tray, &th, None, &rq, rng, sun, depth + 1, ls, None,
                        VisCtl::Off, None,
                    ),
                    None => sky(tdir, sun),
                };
            // Tinted by albedo — the classifier lifts dark MTL glass Kd
            // toward white so this doesn't go black.
            color += albedo * (tput * tcol);
        }
    }

    color
}

/// Fixed glassware IOR — thin-tumbler transmission needs no per-material
/// value; the classifier's `transmission` scalar carries all the variation.
const GLASS_IOR: f32 = 1.5;
/// Interface budget for the refraction chain: front/back walls of a
/// two-walled tumbler with no TIR detour. Past it, glass shades opaque.
const TRANS_MAX_DEPTH: u32 = 4;

/// Tangent-space normal-map perturbation of the geometric normal `n`. The
/// tangent comes from the triangle's positions + UVs ON THE FLY (zero
/// storage at 100M tris; per-triangle tangents facet at UV seams — accepted,
/// the interpolated geometric normal stays smooth), Gram-Schmidt-
/// orthogonalized against n; the bitangent's handedness comes from the UV
/// winding. Degenerate UVs, a degenerate projected tangent, or a
/// perturbation past the geometric horizon all degrade to `n` — coarser,
/// never wrong. The decoded green channel is NEGATED (`NORMAL_MAP_Y_SIGN`):
/// the loader V-flips at load (image row 0 = v 0), so an OpenGL-convention
/// (+Y up) map's y axis points against our +v rows — pinned by
/// `tangent_self_test` and settled visually on San Miguel's railings.
pub const NORMAL_MAP_Y_SIGN: f32 = -1.0;

fn perturb_normal(
    scene: &Scene,
    hit: &Hit,
    n: Vec3A,
    mat: &crate::scene::Material,
    uv: glam::Vec2,
) -> Vec3A {
    let [i0, i1, i2] = scene.indices[hit.tri as usize];
    let p0 = scene.positions[i0 as usize];
    let e1 = scene.positions[i1 as usize] - p0;
    let e2 = scene.positions[i2 as usize] - p0;
    let t0 = scene.texcoords[i0 as usize];
    let d1 = scene.texcoords[i1 as usize] - t0;
    let d2 = scene.texcoords[i2 as usize] - t0;
    let det = d1.x * d2.y - d2.x * d1.y;
    if det.abs() < 1e-12 {
        return n;
    }
    let t_raw = (e1 * d2.y - e2 * d1.y) / det;
    let t = (t_raw - n * n.dot(t_raw)).normalize_or_zero();
    if t == Vec3A::ZERO {
        return n;
    }
    // Bitangent: cross(n, t) signed to agree with the UV-derived bitangent
    // direction — mirrored UVs flip the frame's handedness exactly.
    let b_raw = (e2 * d1.x - e1 * d2.x) / det;
    let b = n.cross(t) * n.cross(t).dot(b_raw).signum();
    let s = scene.textures[mat.normal_tex as usize].sample_bilinear_linear(uv.x, uv.y);
    let tn = Vec3A::new(
        (s.x * 2.0 - 1.0) * mat.normal_scale,
        (s.y * 2.0 - 1.0) * mat.normal_scale * NORMAL_MAP_Y_SIGN,
        (s.z * 2.0 - 1.0).max(0.05),
    );
    let out = (t * tn.x + b * tn.y + n * tn.z).normalize_or_zero();
    if out == Vec3A::ZERO || out.dot(n) <= 0.0 { n } else { out }
}

/// Pure self-test for the on-the-fly tangent frame + normal-map decode (run
/// by `--check` beside `matclass::self_test`): analytic tangent directions on
/// a canonical triangle, the flat-map near-identity, the green-channel sign
/// pin, mirrored-UV handedness, and the degenerate-UV skip.
pub fn tangent_self_test() -> Result<(), String> {
    use crate::scene::{AreaLight, Material, Scene};
    use crate::texture::Texture;
    let tri_scene = |texcoords: [glam::Vec2; 3], texel: [u8; 4]| -> Scene {
        let mut sc = Scene {
            positions: vec![Vec3A::ZERO, Vec3A::X, Vec3A::Y],
            normals: vec![Vec3A::Z; 3],
            texcoords: texcoords.to_vec(),
            indices: vec![[0, 1, 2]],
            tri_mat: vec![0],
            materials: vec![Material {
                albedo: Vec3A::ONE,
                roughness: 0.8,
                metallic: 0.0,
                anisotropy: 0.0,
                sheen: 0.0,
                translucency: 0.0,
                transmission: 0.0,
                emissive: Vec3A::ZERO,
                normal_tex: 0,
                normal_scale: 1.0,
                rough_tex: NO_TEX,
                metal_tex: NO_TEX,
                emissive_tex: NO_TEX,
                kind: MatKind::Diffuse,
            }],
            textures: vec![Texture {
                w: 1,
                h: 1,
                texels: vec![texel],
                alpha_masked: false,
                srgb: false,
                source: String::new(),
            }],
            any_alpha: false,
            light: AreaLight {
                center: Vec3A::Y,
                u: Vec3A::X,
                v: Vec3A::Z,
                color: Vec3A::ONE,
            },
            diag: 1.0,
            eps: 1e-4,
            ao_radius: 0.03,
        };
        crate::scene::finalize_scalars(&mut sc);
        sc
    };
    let hit = Hit { t: 1.0, tri: 0, u: 0.25, v: 0.25 };
    let uv0 = [glam::Vec2::new(0.0, 0.0), glam::Vec2::new(1.0, 0.0), glam::Vec2::new(0.0, 1.0)];
    let perturb = |sc: &Scene| {
        let uv = sc.tri_uv(0, hit.u, hit.v);
        perturb_normal(sc, &hit, Vec3A::Z, &sc.materials[0], uv)
    };

    // Flat map (128,128,255): near-identity (128/255 isn't exactly 0.5 — the
    // no-map case is the bit-identical one; the flat MAP is merely close).
    let sc = tri_scene(uv0, [128, 128, 255, 255]);
    if perturb(&sc).dot(Vec3A::Z) < 0.999 {
        return Err("flat normal map should be a near-identity perturbation".into());
    }
    // Red = +x in tangent space: UVs align u with +X, so the normal tilts
    // toward +X and stays above the horizon.
    let sc = tri_scene(uv0, [255, 128, 128, 255]);
    let out = perturb(&sc);
    if out.x < 0.5 || out.z <= 0.0 {
        return Err(format!("+x tangent tilt wrong: {out:?}"));
    }
    // Green-channel sign pin (NORMAL_MAP_Y_SIGN): +green tilts toward -Y in
    // our V-flipped storage. A sign regression flips every embossing.
    let sc = tri_scene(uv0, [128, 255, 128, 255]);
    let out = perturb(&sc);
    if out.y * NORMAL_MAP_Y_SIGN < 0.5 * NORMAL_MAP_Y_SIGN.abs() && out.y > -0.5 {
        return Err(format!("green-channel sign pin failed: {out:?}"));
    }
    // Mirrored UVs (u negated): the tangent flips with the UV winding.
    let uvm = [glam::Vec2::new(0.0, 0.0), glam::Vec2::new(-1.0, 0.0), glam::Vec2::new(0.0, 1.0)];
    let sc = tri_scene(uvm, [255, 128, 128, 255]);
    let out = perturb(&sc);
    if out.x > -0.5 {
        return Err(format!("mirrored-UV handedness wrong: {out:?}"));
    }
    // Degenerate UVs: skip — the geometric normal comes back exactly.
    let uvz = [glam::Vec2::ZERO; 3];
    let sc = tri_scene(uvz, [255, 128, 128, 255]);
    if perturb(&sc) != Vec3A::Z {
        return Err("degenerate UVs must skip the perturbation".into());
    }
    Ok(())
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
