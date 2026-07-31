use crate::bvh::{Bvh, Hit, Ray};
use crate::scene::{MatKind, Scene, NO_TEX};
use crate::stats::LocalStats;
use glam::Vec3A;

/// Which bounce effects run through the hemisphere frustum integrator instead
/// of the sampled loops. `depth` is the hemisphere subdivision budget
/// (4^depth leaf cells). All-off reproduces the sampled path bit-for-bit.
///
/// There used to be a third member here, `shadows`, dispatching light shafts
/// (a frustum from the shading point through the light RECT, whose proven-lit
/// subrects skipped their occlusion rays). It is gone: it was measured ~3x
/// SLOWER than the rays it saved, and its whole premise — a finite rectangular
/// light with four corners to build a frustum through — died with `AreaLight`.
/// The sun is a disc at infinity now.
#[derive(Clone, Copy, PartialEq)]
pub struct FrustumBounce {
    pub ao: bool,
    pub gi: bool,
    pub depth: u32,
}

impl FrustumBounce {
    pub const OFF: FrustumBounce = FrustumBounce { ao: false, gi: false, depth: 0 };
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

// The flat AMBIENT constant is gone: ambient is now the sky's own order-2 SH
// irradiance (`Scene::sky_sh`, `sh::Sh9::irradiance`), which is directional and
// agrees with what the fb.gi tier integrates with rays.
//
// It was also FSR Ray Regeneration's AO remodulation factor — the constant its
// composite multiplies the denoised open fraction by. That factor is now
// directional too, so it cannot ride in as a root constant: see
// `fsr::wire_normal` for how the three composite sites keep agreeing on it.

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
    /// The material's ripple amplitude (0.0 for everything but water). It is
    /// carried purely so the frame-generation guide pass can tell which
    /// pixels have a MOVING mirror normal: water's reflected image slides
    /// across the surface every frame with no motion in the geometry, and a
    /// guide pass that cannot see that warps it with the (zero) surface MV.
    /// Riding the pack costs nothing — `GBufExt.alb.w` was an unused lane.
    pub ripple_amp: f32,
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

/// Dev cost-attribution ablations for the CPU shade path — the twin of
/// `gpu::trace`'s `abl_has`, and read from the same `FR_ABL` variable.
///
/// `FR_ABL=noshadow,noao,norefl,noglass` neutralize one secondary-ray consumer
/// each; `nosec` arms all four. **NOT shipping levers**: every arm changes the
/// image, which is the point — you are measuring a term against its own
/// absence (the `--no-wide-levels` idiom). Unset is bit-identical, and the
/// per-ray cost when unset is one already-initialized `OnceLock` deref.
///
/// Why they exist: the primary/secondary split in `ray_nodes` says secondary
/// rays are 86-93% of ray traversal, but a counter is not a millisecond — the
/// quadtree removes 26% of ray-node visits on the default scene and the frame
/// does not get faster. These measure the TIME each consumer is worth, which is
/// the only number that can justify building sharing machinery for it.
///
/// Every arm keeps its rng draws and its `secondary_rays` increment: the point
/// is to remove the TRAVERSAL, not to change the sample pattern or make the
/// counters lie about how many rays the shading logic asked for.
#[derive(Default)]
pub struct Abl {
    /// Sun shadow ray -> unoccluded (`Vec3A::ONE`), no trace.
    pub noshadow: bool,
    /// Each AO sample -> fully open, no trace. Draws still burned.
    pub noao: bool,
    /// Skip the GGX reflection continuation (ray + recursive shade).
    pub norefl: bool,
    /// Skip the refraction/TIR continuation chain.
    pub noglass: bool,
}

pub fn abl() -> &'static Abl {
    static A: std::sync::OnceLock<Abl> = std::sync::OnceLock::new();
    A.get_or_init(|| {
        let v = std::env::var("FR_ABL").unwrap_or_default();
        let all = v.contains("nosec");
        let a = Abl {
            noshadow: all || v.contains("noshadow"),
            noao: all || v.contains("noao"),
            norefl: all || v.contains("norefl"),
            noglass: all || v.contains("noglass"),
        };
        // Loud on departure from the default — a silent ablation is how a
        // measurement gets attributed to the wrong thing (the probe-reach trap
        // recorded in CLAUDE.md's pack-split section).
        if a.noshadow || a.noao || a.norefl || a.noglass {
            eprintln!(
                "FR_ABL (cpu shade): noshadow={} noao={} norefl={} noglass={} \
                 — THE IMAGE IS DELIBERATELY WRONG, this is a cost probe",
                a.noshadow, a.noao, a.norefl, a.noglass
            );
        }
        a
    })
}

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
    /// Light-transport throughput per light sample (`Bvh::transmittance`:
    /// ONE = lit, ZERO = blocked, in between = seen through tinted glass).
    /// Below-horizon capture samples record ZERO AND poison `uniform`: that
    /// occlusion was never traced (the rep's own N·L rejected it), so it
    /// must not be replayed onto a neighbor whose horizon differs — the
    /// terminator is a declassify signal exactly like penumbra.
    pub vis: [Vec3A; VIS_MAX],
    pub n_light: u32,
    /// Sampled-AO open fraction.
    pub ao: f32,
    /// Every light sample agreed (bit-equal throughputs — all lit, all
    /// blocked, or all through the SAME glass) and every one was actually
    /// traced — sharing is then near-exact. MIXED visibility means penumbra,
    /// a below-horizon sample means the terminator; both make the cell fall
    /// back to per-pixel rays (mixed visibility is only meaningful with
    /// >= 2 samples; one sample is trivially uniform).
    pub uniform: bool,
    /// Any capture sample fell below the rep's horizon (untraced occlusion).
    pub below_horizon: bool,
}

impl Default for VisRecord {
    fn default() -> Self {
        Self {
            light_uv: [(0.0, 0.0); VIS_MAX],
            vis: [Vec3A::ONE; VIS_MAX],
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

// `sky(d, sun)` is gone. It was a two-color gradient plus a soft `dot^32` glow
// lobe — a *backdrop*, not a light, which is why the thing that actually lit the
// scene was a separate 4x4 rect lamp and the two disagreed. There are now two
// functions, and which one a ray calls is a CORRECTNESS decision, not a
// preference (see src/sky.rs's central invariant):
//
//   crate::sky::dome(d, sun)         — the smooth scattering dome, NO sun disc.
//                                      Every GATHER path: hemi cells, GI leaf
//                                      misses, the SH projection. The sun's
//                                      diffuse is already delivered by direct_d.
//   crate::sky::radiance(d, &sun)    — dome + disc. Every DISPLAY path: the
//                                      camera's own miss, and glass.
//
// The specular reflection ray is the one path where both strategies can find the
// sun, so it takes the dome plus a MIS-weighted disc.

/// Ray-cone state for texture LOD (Möller 2019 ray cones, curvature-free):
/// width at the ray origin plus per-unit-length spread — `width_at_hit =
/// w0 + t·spread`. A zero cone drives every lod to −∞, which the trilinear
/// samplers treat as mip-0 bilinear — bit-identical to the pre-mip renderer.
/// Draws ZERO rng and is a pure function of the hit, so every same-seed /
/// replay / VisCtl-burn contract is untouched.
#[derive(Clone, Copy)]
pub struct Cone {
    pub w0: f32,
    pub spread: f32,
    /// Max anisotropy for footprints read through this cone (`1.0` = off ⇒
    /// the isotropic `tri_lod_base` lod path runs verbatim, bit-identical to
    /// the pre-aniso renderer). Primary/reflection/glass cones carry the
    /// session's `texture::max_aniso()`; hemi-GI bounce cones pin 1.0 —
    /// `HEMI_CONE_SPREAD` is octant-coarse on purpose and 16 taps per bounce
    /// ray would buy nothing (over-blurred bounce albedo is variance
    /// reduction). Mirrored on the GPU by FLAG_ANISO + shade_split's `aniso`.
    pub aniso: f32,
}

impl Cone {
    /// The primary camera cone: apex at the eye, one-pixel spread, resolved
    /// anisotropically at whatever the session's `--aniso` asks for. The
    /// GPU's twin is `leaf.hlsl`/`shade_full` passing (0, pixel_cone, true).
    #[inline(always)]
    pub fn primary(cam: &crate::camera::CamBasis) -> Cone {
        Cone { w0: 0.0, spread: cam.pixel_cone(), aniso: crate::texture::max_aniso() }
    }

    /// The hemi-GI bounce cone: octant-scale and deliberately ISOTROPIC (see
    /// `aniso`). `hemi_leaf.hlsl` passes (0, HEMI_CONE_SPREAD, false).
    #[inline(always)]
    pub fn bounce() -> Cone {
        Cone { w0: 0.0, spread: HEMI_CONE_SPREAD, aniso: 1.0 }
    }
}

/// How one hit's textures get filtered — the isotropic ray-cone lod, or the
/// elliptical footprint. Built once per shaded hit and shared by all five
/// maps on the material (`Lod` carries the per-hit base term, each map adding
/// its own `Texture::lod_dims`; `Aniso` carries normalized-UV gradients, which
/// every texture scales by its own dims). Mirrored by `shade.hlsli::tex_*`.
#[derive(Clone, Copy)]
pub enum TexFilter {
    Lod(f32),
    Aniso { gu: glam::Vec2, gv: glam::Vec2, max: f32 },
}

impl TexFilter {
    /// The single texture-sampling choke point of the CPU shader.
    #[inline]
    pub fn sample(self, tx: &crate::texture::Texture, uv: glam::Vec2, srgb: bool) -> Vec3A {
        match self {
            TexFilter::Lod(base) => {
                let lod = base + tx.lod_dims();
                if srgb {
                    tx.sample_trilinear(uv.x, uv.y, lod)
                } else {
                    tx.sample_trilinear_linear(uv.x, uv.y, lod)
                }
            }
            TexFilter::Aniso { gu, gv, max } => tx.sample_aniso(uv.x, uv.y, gu, gv, max, srgb),
        }
    }
}

/// Cone spread for hemi-GI bounce hits: leaf cells are octant-scale, so the
/// bounce albedo reads a matching broad footprint (over-blurred GI albedo is
/// variance reduction, not error — coarser, never wrong). Mirrored in
/// gpu/shaders/hemi_leaf.hlsl — change both together.
pub const HEMI_CONE_SPREAD: f32 = 0.25;

/// Ray-cone LOD base term, shared contract with the HLSL mirror
/// `shade.hlsli::tex_lod_base` — change both together:
/// `0.5·log2(uv_area/world_area) + log2(cone_width) − log2(max(|n·d|, 0.05))`
/// (each map completes it with its own `Texture::lod_dims`). Computed on the
/// fly from the triangle's vertices — the loads mirror `perturb_normal`'s,
/// and a cached per-tri array would cost ~400 MB at 100M-tri tiling scale.
/// Degenerate UVs/triangles or a zero cone return −∞ → mip-0 bilinear.
pub fn tri_lod_base(scene: &Scene, tri: u32, n_dot_d: f32, cone_w: f32) -> f32 {
    let [i0, i1, i2] = scene.indices[tri as usize];
    let p0 = scene.positions[i0 as usize];
    let e1 = scene.positions[i1 as usize] - p0;
    let e2 = scene.positions[i2 as usize] - p0;
    let wa = e1.cross(e2).length(); // 2× area — the ½ cancels in the ratio
    let t0 = scene.texcoords[i0 as usize];
    let d1 = scene.texcoords[i1 as usize] - t0;
    let d2 = scene.texcoords[i2 as usize] - t0;
    let ua = (d1.x * d2.y - d2.x * d1.y).abs();
    if !(ua > 0.0) || !(wa > 0.0) || !(cone_w > 0.0) {
        return f32::NEG_INFINITY;
    }
    0.5 * (ua / wa).log2() + cone_w.log2() - n_dot_d.max(0.05).log2()
}

/// The triangle's UV basis, derived on the fly from its positions + UVs:
/// `(∂P/∂u, ∂P/∂v)`, both in the triangle's plane. Zero storage (a cached
/// per-tri array would cost ~400 MB at 100M-tri tiling scale). Called ONCE per
/// hit by `shade` and handed to BOTH consumers — the tangent frame
/// (`perturb_normal`) and the texture footprint (`tri_grads_from`).
/// Degenerate UVs (zero-area in UV space) ⇒ None.
/// Mirrored by `shade.hlsli::tri_uv_basis`.
fn tri_uv_basis(scene: &Scene, tri: u32) -> Option<(Vec3A, Vec3A)> {
    let [i0, i1, i2] = scene.indices[tri as usize];
    let p0 = scene.positions[i0 as usize];
    let e1 = scene.positions[i1 as usize] - p0;
    let e2 = scene.positions[i2 as usize] - p0;
    let t0 = scene.texcoords[i0 as usize];
    let d1 = scene.texcoords[i1 as usize] - t0;
    let d2 = scene.texcoords[i2 as usize] - t0;
    let det = d1.x * d2.y - d2.x * d1.y;
    if det.abs() < 1e-12 {
        return None;
    }
    Some(((e1 * d2.y - e2 * d1.y) / det, (e2 * d1.x - e1 * d2.x) / det))
}

/// The ray cone's texture footprint at a hit, as two UV-space gradient
/// vectors — the ANISOTROPIC refinement of `tri_lod_base`.
///
/// The cone is a circle of diameter `cone_w` perpendicular to `d`; projected
/// along `d` onto the surface it becomes an ellipse: `cone_w` across the
/// direction of travel, `cone_w / |n·d|` along it. That stretch is exactly
/// what `tri_lod_base`'s `− log2(max(|n·d|, 0.05))` term stands in for — one
/// scalar lod can only describe a circle, so it takes the MAJOR axis and
/// blurs the minor one with it. Here we keep both axes (same grazing clamp,
/// so the major axis is unchanged) and let the sampler resolve the minor one.
///
/// Returned in NORMALIZED-UV units (Cramer's rule against the triangle's UV
/// basis), so one footprint serves every map on the material — `SampleGrad`'s
/// contract, and `Texture::sample_aniso` follows it. Pure hit geometry, ZERO
/// rng draws (the same-seed / replay / VisCtl-burn gates rely on that).
/// Degenerate UVs, a degenerate basis, or a zero cone ⇒ None ⇒ the caller
/// falls back to the isotropic lod path. Mirrored by `shade.hlsli::tri_grads`.
pub fn tri_grads(
    scene: &Scene,
    tri: u32,
    n: Vec3A,
    d: Vec3A,
    cone_w: f32,
) -> Option<(glam::Vec2, glam::Vec2)> {
    tri_grads_from(tri_uv_basis(scene, tri)?, n, d, cone_w)
}

/// `tri_grads` against an ALREADY-DERIVED basis — the form `shade` calls, so
/// the hit's `tri_uv_basis` is computed exactly once and shared with the
/// tangent frame (`perturb_normal`) instead of each consumer re-fetching the
/// triangle and re-inverting. Mirrored by `shade.hlsli::tri_grads_from`.
fn tri_grads_from(
    (tu, tv): (Vec3A, Vec3A),
    n: Vec3A,
    d: Vec3A,
    cone_w: f32,
) -> Option<(glam::Vec2, glam::Vec2)> {
    if !(cone_w > 0.0) {
        return None;
    }
    // In-plane inversion: for an in-plane w, w = du·tu + dv·tv, so Cramer
    // against n gives du, dv. `den` is the basis' signed area — the same
    // degeneracy `tri_uv_basis` already rejects, re-checked because n is the
    // interpolated shading normal, not the exact face normal.
    let den = tu.cross(tv).dot(n);
    if den.abs() < 1e-12 {
        return None;
    }
    let n_d = d.dot(n);
    let across = n.cross(d);
    let (a_dir, b_dir) = if across.length_squared() > 1e-12 {
        // b: the direction of travel projected into the surface (the axis the
        // grazing stretch acts along); a: across it.
        (across.normalize(), (d - n * n_d).normalize_or_zero())
    } else {
        // Normal incidence — the footprint is a circle; any in-plane
        // orthonormal pair spans it.
        let t = (tu - n * n.dot(tu)).normalize_or_zero();
        if t == Vec3A::ZERO {
            return None;
        }
        (t, n.cross(t))
    };
    let w_min = a_dir * cone_w;
    let w_maj = b_dir * (cone_w / n_d.abs().max(0.05));
    let to_uv = |w: Vec3A| glam::Vec2::new(w.cross(tv).dot(n) / den, tu.cross(w).dot(n) / den);
    Some((to_uv(w_maj), to_uv(w_min)))
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
    cl: &crate::clouds::Clouds,
    cone: Cone,
    depth: u32,
    ls: &mut LocalStats,
    mut prim: Option<&mut PrimarySurface>,
    mut vis: VisCtl,
    hemi_share: Option<&crate::hemi::HemiShare>,
    // Firefly point lights — Some ONLY on the primary camera path
    // (render.rs::shade_traced). The reflection/glass recursion, the hemi
    // bounce tier, and both GI reference estimators pass None: like emissive
    // materials, fireflies do not light bounce surfaces (the one-sky gather
    // exclusion — the stars rule).
    ff: Option<&crate::fireflies::Fireflies>,
) -> Vec3A {
    // Capture/Apply only exist for the sampled shadow/AO paths; the
    // frustum-bounce tiers would silently bypass the record.
    debug_assert!(
        matches!(vis, VisCtl::Off) || (!q.fb.ao && !q.fb.gi),
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
    // Ray-cone texture LOD: cone width at the hit, then one per-hit base
    // term shared by every map on this material (each adds its own
    // dimension term at the sample). Untextured materials skip the
    // triangle fetch entirely; a zero cone gives −∞ ⇒ mip-0 bilinear.
    let cone_w = cone.w0 + hit.t * cone.spread;
    let any_tex = mat.any_tex();
    // ∂P/∂u, ∂P/∂v — derived at most ONCE per hit and shared by its two
    // consumers, the anisotropic footprint below and the tangent frame in
    // `perturb_normal`. Neither is reached without a texture, and the
    // isotropic path with no normal map needs no basis at all.
    let uv_basis = (any_tex && (cone.aniso > 1.0 || mat.normal_tex != NO_TEX))
        .then(|| tri_uv_basis(scene, hit.tri))
        .flatten();
    let filt = if !any_tex {
        TexFilter::Lod(f32::NEG_INFINITY)
    } else {
        // Anisotropic: the elliptical footprint, when the cone asks for it and
        // the triangle's UV basis is sound. A degenerate basis falls through
        // to the isotropic lod (coarser, never wrong).
        match (cone.aniso > 1.0)
            .then_some(uv_basis)
            .flatten()
            .and_then(|b| tri_grads_from(b, n, ray.d, cone_w))
        {
            Some((gu, gv)) => TexFilter::Aniso { gu, gv, max: cone.aniso },
            None => TexFilter::Lod(tri_lod_base(scene, hit.tri, ray.d.dot(n).abs(), cone_w)),
        }
    };
    // Effective albedo: constant, except marble which is evaluated at the
    // world-space hit point (models are static, so world space is stable)
    // and textures, sampled at the hit's interpolated UV.
    let albedo = match mat.kind {
        MatKind::Marble { scale } => marble(ray.o + ray.d * hit.t, scale),
        MatKind::Textured { tex } => {
            let uv = scene.tri_uv(hit.tri, hit.u, hit.v);
            filt.sample(&scene.textures[tex as usize], uv, true)
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
            let s = filt.sample(&scene.textures[mat.rough_tex as usize], uv, false);
            rough_eff = (rough_eff * s.y).clamp(0.02, 1.0);
        }
        if mat.metal_tex != NO_TEX {
            let s = filt.sample(&scene.textures[mat.metal_tex as usize], uv, false);
            metal_eff = (metal_eff * s.z).clamp(0.0, 1.0);
        }
    }
    // Shading normal n_s: the geometric n perturbed by the tangent-space
    // normal map; n_s ≡ n when unmapped (structural bit-identity). n keeps
    // every visibility-adjacent use — the eps-offset p, the translucency
    // back ray, the ENTIRE hemi tier (a perturbed apex normal can put the
    // own triangle inside the "open" hemisphere ⇒ false-empty), and the
    // glass chain. n_s feeds the BRDF frame, N·L, and the G-buffer guide.
    // A degenerate UV basis (`uv_basis` None) is exactly the case the old
    // in-function derivation bailed on — no tangent frame, so n_s stays n.
    let mut n_s = match (mat.normal_tex != NO_TEX, map_uv, uv_basis) {
        (true, Some(uv), Some(basis)) => perturb_normal(scene, n, mat, uv, filt, basis),
        _ => n,
    };
    // Water ripples tilt the SHADING normal on the shared cloud clock (a
    // pure-function wave field, zero rng). Composes ON the normal map: the
    // full-res fountain has none (n_s == n, ripple is the only perturbation),
    // low-poly's water_bump.png perturbs first and the ripple rides on top.
    // Geometric n is untouched (the n_g/n_s split — eps offsets, the hemi
    // tier, and the glass chain's own axis all stay on n). Structural off
    // state: ripple_amp == 0.0 leaves n_s exactly as selected above.
    if mat.ripple_amp > 0.0 {
        n_s = ripple_normal(n_s, n, ray.o + ray.d * hit.t, cl.time, mat.ripple_amp, scene.diag);
    }
    if let Some(prim) = prim.as_deref_mut() {
        *prim = PrimarySurface {
            n: n_s,
            albedo,
            roughness: rough_eff,
            metallic: metal_eff,
            spec_t: 0.0,
            ripple_amp: mat.ripple_amp,
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

    // Will the VNDF reflection ray actually be traced? MIS is a partition of
    // ONE integral between TWO strategies, so the light-sampled specular may
    // only be down-weighted if the BSDF-sampled half is really going to run —
    // otherwise `w_l` deletes energy nobody else delivers. The gate is the one
    // the reflection block below uses, hoisted verbatim (same expression, same
    // FLAT roughness/metallic — a texture-driven gate would make the two
    // conditional VNDF draws depend on the sampler and skew the same-seed A/Bs).
    // It fails for the low preset (`reflections: false`) and at every depth > 0,
    // and on those paths the light-sampled highlight is the ONLY estimator of
    // the sun's specular — so it must carry the full weight.
    let refl_ray = q.reflections && depth == 0 && (mat.metallic > 0.04 || mat.roughness < 0.45);

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
            _ => (rng.f32(), rng.f32()),
        };
        // Uniform-in-cone toward the SUN DISC — exactly two draws, in the same
        // order the old rect sampling consumed them, which is what keeps every
        // same-seed / replay / VisCtl-burn bit-identity contract intact. The
        // stored (su, sv) stay the RAW uniforms, so Apply reproduces the
        // direction bit-exactly. No `lp`, no `dist`, no 1/d²: the sun is at
        // infinity, so every shading point sees it at the same angle and the
        // same brightness (which is why the tiled/stress scenes no longer
        // rescale the light).
        let wi = scene.sun.sample_dir(su, sv);
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
                r.vis[si] = Vec3A::ZERO;
                r.below_horizon = true;
            }
            // Thin-surface transmission: a back-lit translucent surface
            // (leaves) receives the light through itself. The occlusion ray
            // starts on the TRANSMITTED side (p is hit + n·eps, so -2·eps
            // lands at hit - n·eps — the exact mirror of the front
            // convention; the ray departs the leaf's plane on the side it
            // starts on and never re-crosses its own triangle). Plain
            // `occluded` — no cut exists for this apex.
            // Consumes no rng draws — stream alignment is untouched.
            if mat.translucency > 0.0 && ndl < 0.0 {
                let back_vis = match &vis {
                    // The rep's traced throughput is segment transmittance
                    // between the same two points — normal-independent
                    // within 2·eps.
                    VisCtl::Apply(r) => {
                        ls.adapt_rays_saved += 1;
                        r.vis[si]
                    }
                    _ => {
                        ls.secondary_rays += 1;
                        // tmax = INFINITY: the sun is at infinity, so anything
                        // along the ray occludes it (the old bound was the
                        // 12-unit distance to the rect lamp). `transmittance`,
                        // not `occluded`: light rays see through tinted glass.
                        bvh.transmittance(
                            scene,
                            &Ray::new(p - n * (2.0 * scene.eps), wi),
                            0.0,
                            f32::INFINITY,
                            &mut ls.ray_nodes,
                        )
                    }
                };
                if back_vis != Vec3A::ZERO {
                    direct_t += scene.sun.e_over_pi * (-ndl) * back_vis;
                }
            }
            continue;
        }
        let vis_t = match &vis {
            VisCtl::Apply(r) => {
                ls.adapt_rays_saved += 1;
                r.vis[si]
            }
            _ => {
                ls.secondary_rays += 1;
                if abl().noshadow {
                    Vec3A::ONE // FR_ABL=noshadow: unoccluded, no traversal
                } else {
                    // tmax = INFINITY — the sun is at infinity. `transmittance`,
                    // not `occluded`: the sun ray carries a tint through glass
                    // (ONE when clear — `x * 1.0` keeps opaque scenes bitwise).
                    bvh.transmittance(scene, &Ray::new(p, wi), 0.0, f32::INFINITY, &mut ls.ray_nodes)
                }
            }
        };
        if let VisCtl::Capture(r) = &mut vis {
            r.light_uv[si] = (su, sv);
            r.vis[si] = vis_t;
        }
        if vis_t != Vec3A::ZERO {
            // No 1/d²: irradiance/π is authored directly (sky::SUN_E_OVER_PI is
            // exactly the old `light.color / |light.center|²`, so the direct
            // term at the scene origin is unchanged by construction). The
            // throughput rides `li`, so the GGX/sheen terms below inherit the
            // glass tint componentwise.
            let li = scene.sun.e_over_pi * ndl * vis_t;
            direct_d += li;
            let h = (wi + v).normalize_or_zero();
            let hl = to_local(h);
            if hl.z > 0.0 {
                let d = ggx_ndf(hl, ax, ay);
                let g2 = 1.0 / (1.0 + lambda_v + ggx_lambda(to_local(wi), ax, ay));
                let f = schlick(f0, wi.dot(h).max(0.0));
                // MIS (balance heuristic) against the VNDF reflection ray, which
                // can also land in the sun disc — see sky::mis_weight. Counting
                // both would double the sun's specular AND put ~1e3-radiance
                // fireflies in FSR's un-denoised residual. `w_l` sends the
                // energy to light sampling exactly where light sampling is the
                // better strategy (rough surfaces), and stands down on mirrors.
                // Zero rays, zero rng draws — both pdfs are already in scope.
                // The VNDF sampling pdf in solid angle:
                //   p(wi) = G1(v)·D(h) / (4·n·v),  G1(v) = 1/(1 + λ_v).
                //
                // ONLY when that ray is actually traced (`refl_ray`): with no
                // BSDF strategy in play there is nothing to share the integral
                // with, and weighting down would just lose the highlight (a
                // mirror under the low preset measured ~200x too dark).
                let w_l = if refl_ray {
                    let p_b = d / (4.0 * (1.0 + lambda_v) * vl.z.max(1e-6));
                    1.0 - crate::sky::mis_weight(p_b, crate::sky::light_pdf(&scene.sun))
                } else {
                    1.0
                };
                // li carries ndl; D·G2·F/(4·nv·nl)·nl leaves /(4·nv).
                direct_s +=
                    li * f * (std::f32::consts::PI * d * g2 * w_l / (4.0 * vl.z * ndl));
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
    // Cloud shadow: ONE transmittance toward the sun per shade() call (2
    // density evals, zero rng), scaling the WHOLE direct sun contribution —
    // diffuse, specular, and the translucent back term are the same light
    // through the same cloud. Applied BEFORE the prim capture below so FSR's
    // dd/ds signals carry it (the composite identity closes untouched), and
    // before every consumer of direct_* further down. An unshadowed path
    // returns an exact 1.0 and `x * 1.0` is bit-preserving, so a clear-sun
    // pixel — and every `--no-clouds` pixel via the guard — is untouched.
    if cl.enabled {
        let tc = crate::clouds::sun_transmittance(p, scene.sun.dir, cl);
        direct_d *= tc;
        direct_s *= tc;
        direct_t *= tc;
    }
    // Firefly point lights — the direct tier's second entry, AFTER the cloud
    // scaling (a firefly is a local light under the slab — the sun's cloud
    // transmittance must not dim it) and BEFORE the prim capture (the light
    // rides FSR-RR's denoised dd/ds lobes; the diffuse term is albedo-free
    // radiance like the sun's `li`, so the composite identity closes
    // untouched). ZERO rng draws — deterministic iteration, one HARD shadow
    // ray per in-radius firefly (the translucency/sheen/cloud-shadow
    // precedent: every same-seed / replay / VisCtl-burn contract holds with
    // no burn-accounting changes; under VisCtl the firefly rays always trace
    // their own). `w_l = 1` on the specular: a point light has zero solid
    // angle, so the VNDF ray can never deliver it — MIS does not apply (the
    // direction of the MIS rule that forbids down-weighting). A day session
    // has `count == 0` and never enters the loop — bit-identity structurally.
    if let Some(ff) = ff {
        let r_inf = crate::fireflies::FF_RADIUS_K * ff.scale;
        let r2 = r_inf * r_inf;
        for i in 0..ff.count as usize {
            let to = Vec3A::from_slice(&ff.pos[i]) - p;
            let d2 = to.length_squared();
            // The rejection test — the only cost a far firefly ever charges.
            if d2 >= r2 {
                continue;
            }
            let dist = d2.sqrt();
            let wi = to / dist;
            let ndl = n_s.dot(wi);
            if ndl <= 0.0 {
                continue;
            }
            let e = crate::fireflies::irradiance(ff, i, d2);
            if e <= 0.0 {
                continue;
            }
            // One hard shadow ray with FINITE tmax — stop 2·eps short of the
            // light point so a firefly hovering eps off a leaf never
            // self-occludes on the far end. Root `transmittance`: no cut
            // exists for this apex (the translucency-ray rule), and a firefly
            // behind glass shines through with the tint.
            ls.secondary_rays += 1;
            let vis_t = bvh.transmittance(
                scene,
                &Ray::new(p, wi),
                0.0,
                (dist - 2.0 * scene.eps).max(0.0),
                &mut ls.ray_nodes,
            );
            if vis_t == Vec3A::ZERO {
                continue;
            }
            let li = crate::fireflies::FF_COLOR * (e * ndl) * vis_t;
            direct_d += li;
            let h = (wi + v).normalize_or_zero();
            let hl = to_local(h);
            if hl.z > 0.0 {
                let dg = ggx_ndf(hl, ax, ay);
                let g2 = 1.0 / (1.0 + lambda_v + ggx_lambda(to_local(wi), ax, ay));
                let f = schlick(f0, wi.dot(h).max(0.0));
                // li carries ndl; D·G2·F/(4·nv·nl)·nl leaves /(4·nv) — the
                // sun loop's exact term shape, at full weight.
                direct_s += li * f * (std::f32::consts::PI * dg * g2 / (4.0 * vl.z * ndl));
            }
        }
    }
    if let Some(prim) = prim.as_deref_mut() {
        prim.direct_d = direct_d;
        prim.direct_s = direct_s;
    }
    if let VisCtl::Capture(r) = &mut vis {
        r.n_light = n_shadow;
        let k = n_shadow as usize;
        // Bit-equal throughputs: all-lit, all-blocked, AND all-through-the-
        // same-glass are uniform (a consistently tinted cell is not a
        // penumbra); mixed values still declassify.
        r.uniform = k == 0
            || (!r.below_horizon && r.vis[..k].iter().all(|&o| o == r.vis[0]));
    }

    // Diffuse ambient term. Three tiers, all through the hemisphere's OWN
    // apex-relative tmin chain when frustum-dispatched (the primary tile's
    // tmin is never involved):
    // - fb.gi: real sky+bounce irradiance/π over the hemisphere (subsumes AO —
    //   occluders contribute their radiance instead of darkening a constant).
    // - fb.ao: SH sky irradiance modulated by frustum-dispatched AO.
    // - neither: SH sky irradiance modulated by sampled AO.
    //
    // The lower two tiers used to multiply a flat AMBIENT constant, which gave
    // a surface facing the sun exactly the same ambient as one facing away.
    // `sky_sh` is the same integral the fb.gi tier computes with rays — minus
    // occlusion and bounce — so the three tiers now agree on what the sky IS,
    // and `sh::self_test` gates that agreement. Zero rng draws (a pure function
    // of the normal), which is what keeps the same-seed contracts intact.
    let ambient = if q.fb.gi {
        let (t1, t2) = onb(n);
        let accel = crate::ftree::Accel::of(bvh);
        crate::hemi::gi(scene, accel, p, n, t1, t2, q.fb.depth, sun, cl, depth, hemi_share, rng, None, ls)
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
                let mut open = 0.0f32;
                for _ in 0..q.ao_samples {
                    let r1 = rng.f32();
                    let r2 = rng.f32();
                    let dir = cosine_dir(n, t1, t2, r1, r2);
                    ls.secondary_rays += 1;
                    // Mean-of-components: the AO plane is a SCALAR by
                    // contract (FSR's signal, the SH-ambient modulator), so
                    // an RGB glass throughput folds to gray here. `x / 3.0`
                    // (a true divide): 3.0/3.0 == 1.0 and 0.0/3.0 == 0.0
                    // exactly, so opaque scenes accumulate the old integer
                    // counts bit-identically.
                    let tp = if abl().noao {
                        Vec3A::ONE // FR_ABL=noao: fully open, no traversal.
                                   // The draws above still ran, so the stream
                                   // and the sample directions are unchanged.
                    } else {
                        bvh.transmittance(
                            scene,
                            &Ray::new(p, dir),
                            0.0,
                            scene.ao_radius,
                            &mut ls.ray_nodes,
                        )
                    };
                    open += (tp.x + tp.y + tp.z) / 3.0;
                }
                ao = open / q.ao_samples as f32;
                if let VisCtl::Capture(r) = &mut vis {
                    r.ao = ao;
                }
            }
        }
        // FSR's AO signal: the open fraction itself, before the sky it
        // modulates. Assignment-only (no rng draw), so the same-seed
        // bit-identity gates are untouched.
        if let Some(prim) = prim.as_deref_mut() {
            prim.ao = ao;
        }
        // The shading normal: ambient is a BRDF-side quantity, and the n_g/n_s
        // split reserves the geometric normal for visibility.
        scene.sky_sh.irradiance(n_s) * ao
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
                mat.emissive * filt.sample(&scene.textures[mat.emissive_tex as usize], uv, true)
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
    // dielectrics whose specular contribution wouldn't justify a ray — and it
    // is the SAME `refl_ray` the direct loop's MIS weight consulted, hoisted so
    // the two can never disagree about whether a BSDF strategy exists.
    if refl_ray {
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
        // `!norefl` drops the ray AND the recursive shade behind it — the same
        // shape the `reflections: false` preset already produces.
        if rdir.dot(n_s) > 0.0 && rdir.dot(n) > 0.0 && !abl().norefl {
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
                    // The child cone starts at this hit's width — reflected
                    // hits read footprints grown by the full path length.
                    let rcone = Cone { w0: cone_w, spread: cone.spread, aniso: cone.aniso };
                    shade(scene, bvh, &rray, &rh, None, &rq, rng, sun, cl, rcone, depth + 1, ls, None, VisCtl::Off, None, None)
                }
                None => {
                    if let Some(prim) = prim.as_deref_mut() {
                        prim.spec_t = f32::INFINITY;
                    }
                    // The BSDF-sampling half of the MIS pair (see sky::mis_weight
                    // and the direct loop's `w_l`). The DOME passes through
                    // un-weighted — it is smooth and only this strategy sees it —
                    // but the DISC is weighted by w_b, because `direct_s` is
                    // also delivering the sun's specular. On a mirror w_b ≈ 1
                    // and this carries the (round!) sun; on a rough surface
                    // w_b ≈ 0, which is exactly what kills the firefly.
                    let hl_r = to_local(h);
                    let p_b = ggx_ndf(hl_r, ax, ay)
                        / (4.0 * (1.0 + lambda_v) * vl.z.max(1e-6));
                    let w_b =
                        crate::sky::mis_weight(p_b, crate::sky::light_pdf(&scene.sun));
                    // Stars ride un-weighted: they are BSDF-only delivery
                    // (never light-sampled), so there is no partner strategy
                    // to partition with. Twinkle phase 0 — secondary paths
                    // render the fixed-phase field (shade has no frame index,
                    // and a static starfield in a reflection is invisible).
                    //
                    // The cloud layer extinguishes this whole backdrop along
                    // the REFLECTED ray from the hit point (mirrored skies
                    // show the same clouds), including the MIS-weighted disc:
                    // the BSDF strategy's sun rides the march's T while the
                    // light strategy's rides `sun_transmittance` — two
                    // transmittances of the same field along near-identical
                    // directions, a bracketed partition, never a double count
                    // (see clouds.rs's header; do NOT force one T on both).
                    {
                        let dm = crate::sky::dome(rdir, sun, scene.sky_scale);
                        let backdrop = dm
                            + crate::sky::disc(rdir, &scene.sun, cone.spread * 0.5) * w_b
                            + crate::sky::stars(rdir, cone.spread * 0.5, scene.night, 0);
                        if cl.enabled {
                            // The ROUGH march: a reflected sky is seen
                            // through the GGX lobe — 2 steps on the 2-octave
                            // field (clouds::along_rough's cost rationale).
                            match crate::clouds::along_rough(
                                p,
                                rdir,
                                &scene.sun,
                                dm * crate::clouds::CLOUD_AMB_K,
                                cl,
                            ) {
                                None => backdrop,
                                Some(cs) => backdrop * cs.t + cs.scatter,
                            }
                        } else {
                            backdrop
                        }
                    }
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
        let eta = if entering { 1.0 / mat.ior } else { mat.ior };
        let hit_p = ray.o + ray.d * hit.t;
        // The Snell/Fresnel math over an axis `ns`. Refraction and the
        // reflected-fraction Fresnel ride `ns`; the eps offsets stay on the
        // GEOMETRIC n (a perturbed offset could push the origin to the wrong
        // side). Returns (tdir, torig, tput, is_tir).
        let snell = |ns: Vec3A| -> (Vec3A, Vec3A, f32, bool) {
            let cos_i = v.dot(ns).max(1e-4);
            let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
            if k >= 0.0 {
                // Exact unpolarized dielectric Fresnel (not Schlick — it must
                // reach 1 as k -> 0 or the TIR handoff pops).
                let cos_t = k.sqrt();
                let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
                let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
                let fr = 0.5 * (rs * rs + rp * rp);
                let td = (ray.d * eta + ns * (eta * cos_i - cos_t)).normalize();
                (td, hit_p - n * scene.eps, mat.transmission * (1.0 - fr), false)
            } else {
                // TIR: mirror about ns, staying on the incident side.
                (ray.d + ns * (2.0 * cos_i), hit_p + n * scene.eps, mat.transmission, true)
            }
        };
        // Water ripples perturb the Snell axis too (bends the basin image —
        // the dominant water cue), GUARDED: a refraction must cross to the far
        // side of the geometric surface (tdir·n < 0), a TIR mirror stay on the
        // near side (tdir·n > 0). A ripple that flips the side is rejected and
        // the arm recomputes on the geometric n (which provably satisfies both
        // tests — coarser, never wrong). Off (ripple_amp 0) runs geometric-n
        // verbatim: bit-identical to the pre-ripple chain.
        let (tdir, torig, tput, is_tir) = if mat.ripple_amp > 0.0 {
            let n_snell = ripple_normal(n, n, hit_p, cl.time, mat.ripple_amp, scene.diag);
            let r = snell(n_snell);
            let ok = if r.3 { r.0.dot(n) > 0.0 } else { r.0.dot(n) < 0.0 };
            if ok { r } else { snell(n) }
        } else {
            snell(n)
        };
        if tput > 1e-3 && !abl().noglass {
            let tray = Ray::new(torig, tdir);
            ls.secondary_rays += 1;
            let rq = Quality { fb: FrustumBounce::OFF, ..*q };
            let tcone = Cone { w0: cone_w, spread: cone.spread, aniso: cone.aniso };
            // Does the continuation travel INSIDE the medium? Entering
            // crosses in; TIR (only possible on the exit attempt) stays in; a
            // clean exit travels outside.
            let interior = entering || is_tir;
            let (mut tcol, seg) =
                match bvh.intersect(scene, &tray, 0.0, f32::INFINITY, &mut ls.ray_nodes) {
                    Some(th) => (
                        shade(
                            scene, bvh, &tray, &th, None, &rq, rng, sun, cl, tcone, depth + 1,
                            ls, None, VisCtl::Off, None, None,
                        ),
                        th.t,
                    ),
                    // The FULL sky, disc included and un-weighted: refraction is
                    // a near-delta path with no light-sampling partner, so this
                    // is the only strategy that can deliver the sun through
                    // glass. Nothing to double-count. Clouds ride along like
                    // any other display path (radiance marches from torig).
                    // An INTERIOR ray reaching sky is leaked geometry — no
                    // attenuation (seg = INF, skipped below), the status-quo
                    // shape.
                    None => (
                        crate::sky::radiance(
                            torig,
                            tdir,
                            &scene.sun,
                            cone.spread * 0.5,
                            scene.sky_scale,
                            scene.night,
                            0,
                            cl,
                            // No pixel in scope — the fixed-midpoint legacy
                            // phase (the GPU glass miss passes the same 0.5;
                            // the per-(pixel, frame, sample) temporal dither
                            // deliberately excludes this path).
                            0.5,
                        ),
                        f32::INFINITY,
                    ),
                };
            // Beer–Lambert over the interior segment (tinted-shadows part 2,
            // `--no-depth-tint` kills): albedo^(d / D_ref) — the medium is
            // exactly albedo-tinted at TRANS_DEPTH_K·diag of traversal,
            // clearer above, darker below. The depth term the per-interface
            // tint below cannot carry (1 mm of droplet ≠ 2 m of pool); that
            // interface tint STAYS, so thin glassware keeps its look —
            // dimming, never gaining. Zero rng draws (pure hit geometry);
            // shadow rays keep the per-interface tint (unordered candidates
            // have no path lengths — the clouds two-transmittance bracket
            // precedent).
            if interior && seg.is_finite() && crate::scene::depth_tint() {
                tcol *= depth_attenuation(mat.trans_tint_or(albedo), seg, scene.diag);
            }
            // Tinted by the ONE tint source (`trans_tint` for water, else the
            // albedo the classifier lifts toward white so glass isn't black).
            color += mat.trans_tint_or(albedo) * (tput * tcol);
        }
    }

    color
}

// Glassware IOR now rides `Material::ior` (default 1.5 — the old fixed
// `GLASS_IOR`; water is 1.33). `1.0/1.5f32` is bit-identical to the old
// `1.0/GLASS_IOR` const, so existing glass is unchanged.
/// Beer–Lambert reference depth, relative to `Scene::diag`: the interior
/// traversal at which a medium's transmitted light is tinted to exactly its
/// albedo (≈1 m at the OBJ fit's diag 10). Mirrored in shade.hlsli.
pub const TRANS_DEPTH_K: f32 = 0.015;

/// Componentwise `albedo^(seg / (TRANS_DEPTH_K·diag))` — the Beer–Lambert
/// attenuation of one interior segment, in the closed form whose anchors the
/// self-test pins: seg = 0 ⇒ exactly ONE (powf(_, 0) == 1), seg = D_ref ⇒
/// exactly albedo (powf(x, 1) == x), monotone decreasing in seg.
#[inline]
pub fn depth_attenuation(albedo: Vec3A, seg: f32, diag: f32) -> Vec3A {
    let e = seg / (TRANS_DEPTH_K * diag);
    Vec3A::new(albedo.x.powf(e), albedo.y.powf(e), albedo.z.powf(e))
}

/// Depth-tint math gates, run by `--check`: the closed-form anchors above
/// plus monotonicity and the ONE-albedo passthrough.
pub fn depth_tint_self_test() -> Result<(), String> {
    let a = Vec3A::new(0.82, 0.9, 0.97);
    let d_ref = TRANS_DEPTH_K * 10.0;
    let t0 = depth_attenuation(a, 0.0, 10.0);
    if t0.to_array().map(f32::to_bits) != Vec3A::ONE.to_array().map(f32::to_bits) {
        return Err(format!("seg 0 must be exactly ONE, got {t0:?}"));
    }
    let t1 = depth_attenuation(a, d_ref, 10.0);
    if t1.to_array().map(f32::to_bits) != a.to_array().map(f32::to_bits) {
        return Err(format!("seg D_ref must be exactly the albedo, got {t1:?}"));
    }
    let mut prev = f32::INFINITY;
    for i in 0..64 {
        let t = depth_attenuation(a, d_ref * i as f32 * 0.5, 10.0);
        let l = t.x + t.y + t.z;
        if !(l <= prev) || !t.x.is_finite() {
            return Err(format!("attenuation must decrease monotonically (step {i})"));
        }
        prev = l;
    }
    let w = depth_attenuation(Vec3A::ONE, 123.0, 10.0);
    if w.to_array().map(f32::to_bits) != Vec3A::ONE.to_array().map(f32::to_bits) {
        return Err("a white medium must pass through exactly".into());
    }
    Ok(())
}
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

/// `basis` is the hit's `tri_uv_basis` — derived ONCE by the caller and shared
/// with the texture footprint (`tri_grads_from`), the triangle's two `∂P/∂*`
/// consumers. A degenerate basis never reaches here (the caller keeps n).
fn perturb_normal(
    scene: &Scene,
    n: Vec3A,
    mat: &crate::scene::Material,
    uv: glam::Vec2,
    filt: TexFilter,
    (t_raw, b_raw): (Vec3A, Vec3A),
) -> Vec3A {
    let t = (t_raw - n * n.dot(t_raw)).normalize_or_zero();
    if t == Vec3A::ZERO {
        return n;
    }
    // Bitangent: cross(n, t) signed to agree with the UV-derived bitangent
    // direction — mirrored UVs flip the frame's handedness exactly.
    let b = n.cross(t) * n.cross(t).dot(b_raw).signum();
    let s = filt.sample(&scene.textures[mat.normal_tex as usize], uv, false);
    let tn = Vec3A::new(
        (s.x * 2.0 - 1.0) * mat.normal_scale,
        (s.y * 2.0 - 1.0) * mat.normal_scale * NORMAL_MAP_Y_SIGN,
        (s.z * 2.0 - 1.0).max(0.05),
    );
    let out = (t * tn.x + b * tn.y + n * tn.z).normalize_or_zero();
    if out == Vec3A::ZERO || out.dot(n) <= 0.0 { n } else { out }
}

/// Procedural water ripples — a sum of 3 directional sinusoid GRADIENTS, so
/// the field is integrable (a consistent virtual heightfield; no
/// impossible-normal shimmer). Pure f32, ZERO rng, term-for-term mirrored in
/// shade.hlsli (CPU/GPU normals then differ only by sin/cos ulps, absorbed by
/// the statistical A/Bs). All constants are LITERALS (no build-time
/// normalize) so the twin is identical by construction — the clouds-wind
/// precedent. Animated on the shared cloud clock; every length is
/// `Scene::diag`-relative (the scale-relative rule).
/// ONE domain-warped directional swell + three octaves of scrolling
/// gradient-noise chop.
///
/// The old field was three fixed sinusoids, and three plane waves beat
/// against each other on a fixed lattice: the interference pattern repeats,
/// and on a large expanse (the Minecraft ocean) it reads as tiling. Noise
/// breaks the repeat; the swell is kept because pure noise has no wave
/// DIRECTION and open water does.
///
/// Everything stays the analytic gradient of a scalar height (`ripple_height`
/// is that scalar, and the integrability gate differentiates it numerically
/// and compares). NOT curl noise, which is divergence-free — the exact
/// opposite property, and it would give normals no consistent heightfield.
/// A sum of scalars is still a scalar, so the four layers may each scroll at
/// their own velocity (which is what stops the whole field sliding rigidly)
/// without costing integrability at any instant.
const RIPPLE_SWELL_DIR: [f32; 2] = [0.932, 0.362]; // the cloud-wind direction
const RIPPLE_SWELL_LK: f32 = 5.2e-3; // swell wavelength / diag
const RIPPLE_SWELL_W: f32 = 2.1; // rad/s
const RIPPLE_SWELL_A: f32 = 0.42; // slope weight
const RIPPLE_WARP_LK: f32 = 2.4e-2; // warp wavelength / diag (low frequency)
const RIPPLE_WARP_PHI: f32 = 2.6; // radians of phase the warp can bend a crest
/// Chop octaves: wavelength/diag, slope weight, and scroll velocity in
/// wavelengths/s. Distinct directions AND rates — equal ones re-alias.
const RIPPLE_CHOP: [(f32, f32, [f32; 2]); 3] = [
    (3.3e-3, 0.30, [-0.31, 0.42]),
    (1.7e-3, 0.22, [0.55, -0.24]),
    (8.5e-4, 0.15, [-0.18, -0.61]),
];

/// The scalar virtual ripple HEIGHT at world point `p`, time `t`.
///
/// Only the self-test consumes this — shading needs the gradient, not the
/// height — but it is the definition `ripple_grad` differentiates, and the
/// gate numerically differentiates THIS and compares, which is a mechanized
/// proof of integrability the old sinusoids never had.
pub(crate) fn ripple_height(p: Vec3A, t: f32, diag: f32) -> f32 {
    let pxz = glam::Vec2::new(p.x, p.z);
    let d0 = glam::Vec2::from(RIPPLE_SWELL_DIR);
    let l_sw = RIPPLE_SWELL_LK * diag;
    let (nw, _) = crate::clouds::vnoise_vg(pxz / (RIPPLE_WARP_LK * diag), 16);
    let theta = std::f32::consts::TAU * (d0.dot(pxz) / l_sw) - RIPPLE_SWELL_W * t
        + RIPPLE_WARP_PHI * nw;
    let mut h = RIPPLE_SWELL_A * (l_sw / std::f32::consts::TAU) * theta.sin();
    for (k, &(lk, a, v)) in RIPPLE_CHOP.iter().enumerate() {
        let l = lk * diag;
        let q = pxz / l - glam::Vec2::from(v) * t;
        let (n, _) = crate::clouds::vnoise_vg(q, 17 + k as u32);
        h += a * l * n;
    }
    h
}

/// ∂h/∂x, ∂h/∂z of `ripple_height` — the analytic gradient, in closed form.
///
/// `pub(crate)` for the frame-generation guide pass, which evaluates the field
/// at two times to derive the previous frame's mirror normal
/// (`gpu::ngxfg_guides`). One CPU source of truth; the GPU twin is
/// shaders/ripple.hlsli.
pub(crate) fn ripple_grad(p: Vec3A, t: f32, diag: f32) -> glam::Vec2 {
    let pxz = glam::Vec2::new(p.x, p.z);
    let d0 = glam::Vec2::from(RIPPLE_SWELL_DIR);
    let l_sw = RIPPLE_SWELL_LK * diag;
    let l_wp = RIPPLE_WARP_LK * diag;
    // Swell, differentiated through the warp by the chain rule: the crest
    // direction picks up the warp's own gradient, which is what bends each
    // wave uniquely instead of repeating it.
    let (nw, gw) = crate::clouds::vnoise_vg(pxz / l_wp, 16);
    let theta = std::f32::consts::TAU * (d0.dot(pxz) / l_sw) - RIPPLE_SWELL_W * t
        + RIPPLE_WARP_PHI * nw;
    // d(theta)/dp = TAU/l_sw * d0 + PHI * gw / l_wp; the leading
    // A*(l_sw/TAU) of the height cancels the TAU/l_sw, leaving A*cos*d0.
    let dtheta = d0 * (std::f32::consts::TAU / l_sw) + gw * (RIPPLE_WARP_PHI / l_wp);
    let mut g = dtheta * (RIPPLE_SWELL_A * (l_sw / std::f32::consts::TAU) * theta.cos());
    // Chop: h_k = a*l*n(p/l - v t) ⇒ ∇h_k = a*∇n (the l cancels), so the
    // slope weights stay scale-free exactly like the old sinusoids'.
    for (k, &(lk, a, v)) in RIPPLE_CHOP.iter().enumerate() {
        let l = lk * diag;
        let q = pxz / l - glam::Vec2::from(v) * t;
        let (_, gn) = crate::clouds::vnoise_vg(q, 17 + k as u32);
        g += gn * a;
    }
    g
}

/// Tilt `base` by the ripple slope (× `amp`), keeping it a unit vector on the
/// +`n` side. `n` is the GEOMETRIC normal — both the horizon guard and the
/// axis the world-XZ slope is projected against. A degenerate/below-horizon
/// result falls back to `base` (coarser, never wrong). Zero rng.
fn ripple_normal(base: Vec3A, n: Vec3A, p: Vec3A, t: f32, amp: f32, diag: f32) -> Vec3A {
    // Off state returns `base` untouched (no re-normalize — an already-unit
    // base would drift by a ulp). The call sites also guard `ripple_amp > 0`,
    // so this is the structural bit-identity in-function too.
    if amp == 0.0 {
        return base;
    }
    let g = ripple_grad(p, t, diag) * amp;
    // A heightfield normal is (−∂h/∂x, 1, −∂h/∂z): subtract the in-plane
    // gradient from the base. Project the world-XZ slope into n's tangent
    // plane first so a (near-)horizontal but slightly tilted basin behaves.
    let g3 = Vec3A::new(g.x, 0.0, g.y);
    let gt = g3 - n * g3.dot(n);
    let out = (base - gt).normalize_or_zero();
    if out == Vec3A::ZERO || out.dot(n) <= 0.0 { base } else { out }
}

/// Pure self-test for the on-the-fly tangent frame + normal-map decode (run
/// by `--check` beside `matclass::self_test`): analytic tangent directions on
/// a canonical triangle, the flat-map near-identity, the green-channel sign
/// pin, mirrored-UV handedness, and the degenerate-UV skip.
pub fn tangent_self_test() -> Result<(), String> {
    use crate::scene::{Material, Scene};
    use crate::texture::Texture;
    // 1×1 single-texel map — the original cases; ramp cases below hand in a
    // full converted texture instead.
    let px = |texel: [u8; 4]| Texture {
        w: 1,
        h: 1,
        texels: vec![texel],
        alpha_masked: false,
        srgb: false,
        source: String::new(),
        h2n: false,
        n2h: false,
        mips: Vec::new(),
    };
    let tri_scene = |texcoords: [glam::Vec2; 3], tex: Texture| -> Scene {
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
                trans_tint: Vec3A::splat(-1.0),
                ior: 1.5,
                ripple_amp: 0.0,
                emissive: Vec3A::ZERO,
                normal_tex: 0,
                normal_scale: 1.0,
                height_amp: 0.0,
                rough_tex: NO_TEX,
                metal_tex: NO_TEX,
                emissive_tex: NO_TEX,
                class: crate::matclass::IDX_DEFAULT as u8,
                kind: MatKind::Diffuse,
            }],
            textures: vec![tex],
            any_alpha: false,
            any_height: false,
            any_transmissive: false,
            sun: crate::sky::Sun::new(Vec3A::Y),
            sky_sh: crate::sh::Sh9::ZERO,
            sky_scale: 1.0,
            night: 0.0,
            sway: None,
            sway_regions: Vec::new(),
            diag: 1.0,
            eps: 1e-4,
            ao_radius: 0.03,
            content_min: Vec3A::ZERO,
            content_max: Vec3A::ZERO,
        };
        crate::scene::finalize_scalars(&mut sc);
        sc
    };
    let hit = Hit { t: 1.0, tri: 0, u: 0.25, v: 0.25 };
    let uv0 = [glam::Vec2::new(0.0, 0.0), glam::Vec2::new(1.0, 0.0), glam::Vec2::new(0.0, 1.0)];
    // The `shade` caller's contract: no UV basis ⇒ no tangent frame ⇒ keep n.
    let perturb = |sc: &Scene| {
        let uv = sc.tri_uv(0, hit.u, hit.v);
        match tri_uv_basis(sc, 0) {
            Some(b) => perturb_normal(
                sc,
                Vec3A::Z,
                &sc.materials[0],
                uv,
                TexFilter::Lod(f32::NEG_INFINITY),
                b,
            ),
            None => Vec3A::Z,
        }
    };

    // Flat map (128,128,255): near-identity (128/255 isn't exactly 0.5 — the
    // no-map case is the bit-identical one; the flat MAP is merely close).
    let sc = tri_scene(uv0, px([128, 128, 255, 255]));
    if perturb(&sc).dot(Vec3A::Z) < 0.999 {
        return Err("flat normal map should be a near-identity perturbation".into());
    }
    // Red = +x in tangent space: UVs align u with +X, so the normal tilts
    // toward +X and stays above the horizon.
    let sc = tri_scene(uv0, px([255, 128, 128, 255]));
    let out = perturb(&sc);
    if out.x < 0.5 || out.z <= 0.0 {
        return Err(format!("+x tangent tilt wrong: {out:?}"));
    }
    // Green-channel sign pin (NORMAL_MAP_Y_SIGN): +green tilts toward -Y in
    // our V-flipped storage. A sign regression flips every embossing.
    let sc = tri_scene(uv0, px([128, 255, 128, 255]));
    let out = perturb(&sc);
    if out.y * NORMAL_MAP_Y_SIGN < 0.5 * NORMAL_MAP_Y_SIGN.abs() && out.y > -0.5 {
        return Err(format!("green-channel sign pin failed: {out:?}"));
    }
    // Mirrored UVs (u negated): the tangent flips with the UV winding.
    let uvm = [glam::Vec2::new(0.0, 0.0), glam::Vec2::new(-1.0, 0.0), glam::Vec2::new(0.0, 1.0)];
    let sc = tri_scene(uvm, px([255, 128, 128, 255]));
    let out = perturb(&sc);
    if out.x > -0.5 {
        return Err(format!("mirrored-UV handedness wrong: {out:?}"));
    }
    // Degenerate UVs: skip — the geometric normal comes back exactly.
    let uvz = [glam::Vec2::ZERO; 3];
    let sc = tri_scene(uvz, px([255, 128, 128, 255]));
    if perturb(&sc) != Vec3A::Z {
        return Err("degenerate UVs must skip the perturbation".into());
    }

    // --- height_to_normal, end-to-end through the REAL decode --------------
    // An 8×8 grayscale ramp ascending in +u, Sobel-converted, sampled via
    // perturb_normal (NORMAL_MAP_Y_SIGN included): the normal must tilt
    // AGAINST the ascent. The +v ramp is the pin on the conversion's green
    // pre-negation — an implementation storing raw n.y comes back +y after
    // the shader's negation and fails here.
    let gray = |v: u8| [v, v, v, 255];
    let ramp_u: Vec<[u8; 4]> = (0..64).map(|i| gray(((i % 8) * 32) as u8)).collect();
    let mut hu = px([0; 4]);
    (hu.w, hu.h, hu.texels) = (8, 8, ramp_u);
    let sc = tri_scene(uv0, hu.height_to_normal());
    let out = perturb(&sc);
    if out.x > -0.15 || out.z < 0.8 {
        return Err(format!("h2n +u ramp should tilt −x: {out:?}"));
    }
    let ramp_v: Vec<[u8; 4]> = (0..64).map(|i| gray(((i / 8) * 32) as u8)).collect();
    let mut hv = px([0; 4]);
    (hv.w, hv.h, hv.texels) = (8, 8, ramp_v);
    let sc = tri_scene(uv0, hv.height_to_normal());
    let out = perturb(&sc);
    if out.y > -0.15 || out.z < 0.8 {
        return Err(format!("h2n +v ramp should tilt −y (green pre-negation): {out:?}"));
    }

    // --- tri_grads: the anisotropic footprint ------------------------------
    // Canonical triangle (u along +X, v along +Y, unit scale — conformal),
    // so the analytic answers are exact.
    let sc = tri_scene(uv0, px([128, 128, 255, 255]));
    let n = Vec3A::Z;
    let cw = 0.01f32;

    // Normal incidence: the footprint is a CIRCLE — both axes = cone_w.
    let Some((gu, gv)) = tri_grads(&sc, 0, n, -Vec3A::Z, cw) else {
        return Err("tri_grads returned None at normal incidence".into());
    };
    if (gu.length() - cw).abs() > 1e-6 || (gv.length() - cw).abs() > 1e-6 {
        return Err(format!("normal-incidence footprint not circular: {gu:?} {gv:?}"));
    }

    // Grazing: the major axis stretches by exactly 1/|n·d| — the anisotropy
    // trilinear's `−log2(max(|n·d|, 0.05))` term can only blur away.
    for deg in [30.0f32, 60.0, 80.0] {
        let (s, c) = deg.to_radians().sin_cos();
        let d = Vec3A::new(s, 0.0, -c);
        let Some((gu, gv)) = tri_grads(&sc, 0, n, d, cw) else {
            return Err(format!("tri_grads returned None at {deg}°"));
        };
        let (maj, min) = (gu.length().max(gv.length()), gu.length().min(gv.length()));
        if (min - cw).abs() > 1e-6 || (maj - cw / c).abs() > 1e-5 {
            return Err(format!("{deg}° footprint {min}×{maj}, want {cw}×{}", cw / c));
        }
        // The REDUCTION PIN: the major axis in texels IS today's isotropic
        // lod. Aniso is a refinement of `tri_lod_base`, not a rival formula —
        // if this drifts, the two paths have diverged and --no-aniso stops
        // being a clean A/B.
        let w = 256.0f32;
        let want = tri_lod_base(&sc, 0, c, cw) + 0.5 * (w * w).log2();
        if ((maj * w).log2() - want).abs() > 1e-4 {
            return Err(format!(
                "{deg}°: major-axis lod {} != tri_lod_base {want}",
                (maj * w).log2()
            ));
        }
    }

    // Degenerate UVs / zero cone: no footprint — the caller falls back to the
    // isotropic lod (coarser, never wrong).
    if tri_grads(&tri_scene(uvz, px([128, 128, 255, 255])), 0, n, -Vec3A::Z, cw).is_some() {
        return Err("degenerate UVs must yield no footprint".into());
    }
    if tri_grads(&sc, 0, n, -Vec3A::Z, 0.0).is_some() {
        return Err("a zero cone must yield no footprint".into());
    }
    Ok(())
}

/// Pure self-test for the water ripple field (run by `--check`): the
/// structural off state (amp 0 ⇒ input returned BITWISE), the horizon guard
/// (unit-length, `·n > 0` over a direction/time sweep), a closed-form
/// single-axis anchor, and animation (two times ⇒ different normals). Zero
/// rng — a pure function of (p, t).
pub fn ripple_self_test() -> Result<(), String> {
    let n = Vec3A::Y;
    let diag = 10.0f32;
    // (a) Structural off: amp 0 returns the base normal bit-for-bit, for any
    // base — this is what makes WATER_RIPPLE_AMP = 0.0 a clean A/B.
    for &base in &[Vec3A::Y, Vec3A::new(0.1, 0.98, -0.05).normalize()] {
        let out = ripple_normal(base, n, Vec3A::new(1.3, 0.0, -2.1), 3.7, 0.0, diag);
        if out.to_array().map(f32::to_bits) != base.to_array().map(f32::to_bits) {
            return Err(format!("amp 0 must return the base verbatim: {out:?} != {base:?}"));
        }
    }
    // (b) Horizon guard + unit length over a sweep of world points and times.
    for i in 0..16 {
        let p = Vec3A::new(i as f32 * 0.37 - 3.0, 0.0, i as f32 * -0.51 + 2.0);
        for &t in &[0.0f32, 1.25, 5.5] {
            let out = ripple_normal(n, n, p, t, crate::scene::WATER_RIPPLE_AMP, diag);
            if (out.length() - 1.0).abs() > 1e-5 {
                return Err(format!("ripple normal not unit-length: {out:?}"));
            }
            if out.dot(n) <= 0.0 {
                return Err(format!("ripple normal dipped below horizon: {out:?}"));
            }
        }
    }
    let amp = crate::scene::WATER_RIPPLE_AMP;
    // (c) INTEGRABILITY, the gate the sinusoid field never had and the one
    // that matters: `ripple_grad` must be the true gradient of the scalar
    // `ripple_height`. If it ever stops being one — a stray term, a chain
    // rule dropped through the domain warp, someone reaching for curl noise —
    // the normals no longer describe any heightfield and the surface shimmers
    // with impossible normals, which no image gate would catch as a defect.
    // Central differences against the analytic gradient, over points and
    // times, at a step chosen well above f32 cancellation and well below the
    // shortest wavelength (8.5e-4·diag).
    {
        let h = 1e-3f32 * diag * 0.05;
        let mut worst = 0.0f32;
        for i in 0..16 {
            let p = Vec3A::new(i as f32 * 0.41 - 3.0, 0.0, i as f32 * -0.29 + 1.7);
            for &t in &[0.0f32, 0.8, 3.3] {
                let dx = (ripple_height(p + Vec3A::new(h, 0.0, 0.0), t, diag)
                    - ripple_height(p - Vec3A::new(h, 0.0, 0.0), t, diag))
                    / (2.0 * h);
                let dz = (ripple_height(p + Vec3A::new(0.0, 0.0, h), t, diag)
                    - ripple_height(p - Vec3A::new(0.0, 0.0, h), t, diag))
                    / (2.0 * h);
                let g = ripple_grad(p, t, diag);
                worst = worst.max((g.x - dx).abs()).max((g.y - dz).abs());
            }
        }
        if worst > 5e-3 {
            return Err(format!(
                "ripple_grad is not the gradient of ripple_height (worst |Δ| {worst}) — \
                 the field is no longer integrable"
            ));
        }
    }
    // (c2) Determinism + bounded slope. The amplitude bound is what keeps the
    // tilt inside the horizon guard by construction rather than by luck.
    {
        let mut worst = 0.0f32;
        for i in 0..64 {
            let p = Vec3A::new(i as f32 * 1.7 - 50.0, 0.0, i as f32 * -2.3 + 30.0);
            let t = i as f32 * 0.37;
            let g = ripple_grad(p, t, diag);
            if !g.is_finite() {
                return Err(format!("ripple_grad non-finite at {p:?} t={t}"));
            }
            worst = worst.max(g.length());
            if ripple_grad(p, t, diag) != g {
                return Err("ripple_grad is not deterministic".into());
            }
        }
        // Σ of the slope weights (0.42 + 0.30 + 0.22 + 0.15) plus the warp's
        // chain-rule contribution; 2.0 is the design ceiling.
        if worst > 2.0 {
            return Err(format!("ripple slope {worst} exceeds the design bound 2.0"));
        }
    }
    // (c3) APERIODICITY — the whole point of the change. The old field was a
    // sum of plane waves, so translating by a common multiple of the
    // wavelengths reproduced it; the warp and the noise octaves must not.
    // Probe a translation that IS a whole number of swell wavelengths: the
    // pure-sinusoid field would repeat exactly there.
    {
        let lam = RIPPLE_SWELL_LK * diag;
        let d0 = glam::Vec2::from(RIPPLE_SWELL_DIR);
        let p = Vec3A::new(0.3, 0.0, -0.7);
        let shift = 32.0 * lam; // 32 swell periods along the crest normal
        let q = p + Vec3A::new(d0.x * shift, 0.0, d0.y * shift);
        let a = ripple_grad(p, 0.0, diag);
        let b = ripple_grad(q, 0.0, diag);
        if (a - b).length() < 1e-3 {
            return Err(format!(
                "field repeats after {shift} (a whole number of swell wavelengths) — \
                 the warp/noise is not breaking periodicity"
            ));
        }
    }
    // (d) Animation: the same point at two times gives different normals (the
    // field advects — a still frame would freeze, which is the known accept).
    let a = ripple_normal(n, n, Vec3A::new(0.5, 0.0, 0.5), 0.0, amp, diag);
    let b = ripple_normal(n, n, Vec3A::new(0.5, 0.0, 0.5), 1.0, amp, diag);
    if (a - b).length() < 1e-4 {
        return Err("ripple must advance with time".into());
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
