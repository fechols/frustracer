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
    /// REAL-TIME GI, as a BOUNCE BUDGET rather than a switch — the ladder
    /// `--rtgi-bounces 0 | 0.5 | 1 | 1.5 | 2` (`shade::rtgi_bounces`).
    ///
    /// `>= 1.0` takes the DETERMINISTIC arm: one cosine-sampled gather IS the
    /// ambient (shaded at `hemi::BOUNCE_Q`, whose SH×AO ambient is the tail
    /// standing in for deeper bounces), and the hit shades with the budget
    /// DECREMENTED BY ONE — so the field is itself the recursion bound. It
    /// replaced a `depth == 0` gate, which could only ever express "one" and
    /// which said nothing about the reflection/glass children that must not
    /// inherit a budget at all.
    ///
    /// A FRACTIONAL remainder is the stochastic rung: the sampled SH×AO tail
    /// is kept as a control variate and a real gather is rouletted over it at
    /// that probability — unbiased for the next rung up, at that fraction of
    /// its ray cost (see the ambient tier's roulette arm for the estimator).
    /// `0.0` is the flat SH×AO ambient, i.e. the pre-RTGI renderer.
    ///
    /// ONE field, deliberately not a bool beside a probability:
    /// `Quality { rtgi_bounces: 0.0, .. }` pins the whole tier off in a single
    /// token, so the dozen gates that need a deterministic AO ambient cannot be
    /// left half-pinned by a session lever they never heard of.
    ///
    /// The still-frame fb tiers take precedence. Session-armed via
    /// `set_rtgi_bounces` (`--rtgi-bounces`, `--no-rtgi`), read by the
    /// constructors below so every session path inherits ONE decision — check
    /// harnesses pin the field per pass instead of mutating process state.
    pub rtgi_bounces: f32,
    /// May this invocation's shading add material emissive to the display
    /// color? TRUE everywhere except the RTGI bounce while cluster NEE is
    /// live that frame (the NEE-keep rule, 2026-08-08 — the XeSS feel-test:
    /// a TAA-class upscaler's neighborhood clamp rejects sparse stochastic
    /// emissive, so armed sessions keep the deterministic NEE pools and the
    /// bounce suppresses emitter-as-emitter transport instead — exactly one
    /// delivery per frame). Propagates through the bounce's own glass chain
    /// via the recursion's `..*q`; the hemi gather keeps the add (fb.gi
    /// drops NEE instead — the original inverted once-per-path rule). The
    /// GPU twin is shade.hlsli's `cam_lights || !FLAG_EMISSIVE` gate on the
    /// emissive block — no new shader argument needed.
    pub emissive_display: bool,
}

/// Session lever for real-time GI (the `scene::amb_bump` lever shape): the
/// BOUNCE BUDGET `--rtgi-bounces N`, DEFAULT 1.0 — one deterministic bounce,
/// the pre-ladder renderer — and 0.0 under `--no-rtgi`. Read by the `Quality`
/// constructors, never inside `shade()` itself, which takes its budget from
/// the Quality it was handed.
static RTGI_BOUNCES: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(1.0f32.to_bits());

pub fn set_rtgi_bounces(n: f32) {
    RTGI_BOUNCES.store(n.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

pub fn rtgi_bounces() -> f32 {
    f32::from_bits(RTGI_BOUNCES.load(std::sync::atomic::Ordering::Relaxed))
}

/// Does this session run the DETERMINISTIC level-0 gather? The `RTGI` compile
/// define and every gate that means "is the bounce tier structurally live" key
/// on this. FALSE at rung 0.5, which arms `RTGI_CORR` instead — so a gate that
/// means "is ANY GI gather live" must read `rtgi_bounces() > 0.0` (the ladder's
/// must-fires do), and one that means "is the SECOND bounce live" `> 1.0`.
pub fn rtgi_enabled() -> bool {
    rtgi_bounces() >= 1.0
}

/// The unbiasedness gate's TEETH, and the interactive A/B behind it
/// (`FR_RTGI_NOWEIGHT=1`): take a rouletted gather OUTRIGHT instead of the
/// 1/p-weighted delta — which is exactly the naive "average a 1-bounce image
/// and a 2-bounce image" design the ladder exists to avoid. It delivers only
/// `p` of the correction, so it is BIASED by construction, and the gate
/// asserts it FAILS the bound the weighted form passes. Without that arm a
/// roulette that never fired at all would pass the gate by agreeing with the
/// tail it never departed from.
static RTGI_RR_NOWEIGHT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_rtgi_rr_noweight(on: bool) {
    RTGI_RR_NOWEIGHT.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn rtgi_rr_noweight() -> bool {
    RTGI_RR_NOWEIGHT.load(std::sync::atomic::Ordering::Relaxed)
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
                rtgi_bounces: rtgi_bounces(),
                emissive_display: true,
            },
            3 => Quality {
                shadow_samples: 4,
                ao_samples: 4,
                reflections: true,
                fb: FrustumBounce { depth: 4, ..FrustumBounce::OFF },
                rtgi_bounces: rtgi_bounces(),
                emissive_display: true,
            },
            _ => Quality {
                shadow_samples: 2,
                ao_samples: 2,
                reflections: true,
                fb: FrustumBounce { depth: 3, ..FrustumBounce::OFF },
                rtgi_bounces: rtgi_bounces(),
                emissive_display: true,
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
            // RTGI rides the upscaler contract: the bounce is per-frame
            // stochastic noise the temporal integrator launders, exactly like
            // the 1-spp shadow/AO rays — and so, one rung up, is the roulette's
            // own continue/terminate decision.
            rtgi_bounces: rtgi_bounces(),
            emissive_display: true,
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
            // RTGI stays ON while moving — the bounce IS the ambient (the arm
            // never reads ao_samples), and accumulation/denoisers converge it.
            rtgi_bounces: self.rtgi_bounces,
            emissive_display: true,
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
    /// G-buffer write site, via `diff_albedo`).
    pub albedo: Vec3A,
    /// Perceptual roughness and metalness, straight from the material.
    pub roughness: f32,
    pub metallic: f32,
    /// The material's transmission — carried so the wire diffuse albedo can
    /// be the EFFECTIVE `kd·(1−transmission)` the shader's color actually
    /// multiplies (see `diff_albedo`). Default 0.0 keeps every opaque
    /// material's wire byte-identical.
    pub trans: f32,
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
    /// NRD hit-distance guides (`GBufExt.sig.w` on the GPU): the AO ray's
    /// and the sun-shadow ray's sample-0 occluder t. GPU-CAPTURED ONLY in
    /// v1 — the CPU renderer never feeds NRD, and threading a t out of
    /// `Bvh::transmittance`'s traversal would touch the CPU hot loops for a
    /// value nothing consumes; these stay 0.0 here so the PrimSurf mirror
    /// remains field-aligned (shade.hlsli owns the live capture).
    #[allow(dead_code)] // never read on the CPU by design (see above)
    pub ao_t: f32,
    #[allow(dead_code)]
    pub shadow_t: f32,
    /// `FLAG_REMOD_EXACT`'s two diffuse sub-term factors (`PrimSurf.amb_k` /
    /// `PrimSurf.m_d` on the GPU): the exact post-capture scalars shade applies
    /// to the ambient-or-bounce term (`sk*dcav`) and to the direct diffuse
    /// (`sk`), relative to the wire kd. They are blended by energy into the
    /// wire delta multiplier — see the PrimSurf comment for why one divisor
    /// cannot serve both, and why the blend belongs on the delta rather than in
    /// the captured signal.
    ///
    /// GPU-CAPTURED ONLY, for the same reason as `ao_t`/`shadow_t` above and
    /// one more: the CPU renderer has no split ambient at all (its RTGI arm
    /// composes the bounce straight into `color`, see the divergence note at
    /// the `FLAG_NRD_GI` site below) and never feeds NRD, so there is nothing
    /// here for a factor to correct. Held at the multiplicative identity so the
    /// PrimSurf mirror stays field-aligned.
    #[allow(dead_code)]
    pub amb_k: f32,
    #[allow(dead_code)]
    pub m_d: f32,
}

impl PrimarySurface {
    /// The wire diffuse albedo — the factor `color`'s diffuse terms are
    /// ACTUALLY multiplied by, `albedo·(1−metallic)·(1−transmission)`
    /// (shade's `kd·(1−transmission)`; sheen/translucency remainders stay in
    /// the residual — see `fsr::split_signals`). Both CPU wire derivation
    /// sites (the G-buffer diff_alb plane and the `split_signals` kd) MUST go
    /// through this one method: the composite identity requires them equal,
    /// and the GPU pack (`trace_common.hlsli::gbuf_write_hit`) mirrors the
    /// exact multiply order. Folding `(1−transmission)` in is what keeps the
    /// denoiser's diffuse/AO deltas remodulating at their PHYSICAL weight on
    /// glass and water — with raw kd on the wire, water (transmission 0.97)
    /// amplified every denoiser delta 33×, which smeared terrain-colored
    /// bleed across the surface in FSR-RR sessions.
    pub fn diff_albedo(&self) -> Vec3A {
        self.albedo * (1.0 - self.metallic) * (1.0 - self.trans)
    }
}

/// Capacity of a `VisRecord` (the presets top out at 4 shadow samples).
pub const VIS_MAX: usize = 8;

/// Dev cost-attribution ablations for the CPU shade path — the twin of
/// `gpu::trace`'s `abl_has`, and read from the same `FR_ABL` variable.
///
/// `FR_ABL=noshadow,noao,norefl,noglass,nogi` neutralize one secondary-ray
/// consumer each; `nosec` arms all five. **NOT shipping levers**: every arm changes the
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
    /// Skip the RTGI bounce (ray + recursive shade — ambient degrades to the
    /// unoccluded sky gather; the norefl shape for a recursive consumer).
    pub nogi: bool,
    /// REPRO ARM, not a cost probe: restore the pre-fix `surface_point` flip,
    /// which decided on the INTERPOLATED normal and so inverted the eps-offset
    /// axis across every smooth silhouette — the black-limb-band bug. Brings
    /// the band back on demand for a before/after A/B (the `nocandtmin`
    /// precedent). CPU only; the GPU twin has no lever.
    pub nofaceflip: bool,
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
            nogi: all || v.contains("nogi"),
            // Deliberately NOT in `nosec`: this is a repro arm, not a
            // secondary-ray cost probe, and folding it in would silently
            // reintroduce the bug under every cost measurement.
            nofaceflip: v.contains("nofaceflip"),
        };
        // Loud on departure from the default — a silent ablation is how a
        // measurement gets attributed to the wrong thing (the probe-reach trap
        // recorded in CLAUDE.md's pack-split section).
        if a.noshadow || a.noao || a.norefl || a.noglass || a.nogi || a.nofaceflip {
            eprintln!(
                "FR_ABL (cpu shade): noshadow={} noao={} norefl={} noglass={} nogi={} nofaceflip={} \
                 — THE IMAGE IS DELIBERATELY WRONG (cost probe / repro arm)",
                a.noshadow, a.noao, a.norefl, a.noglass, a.nogi, a.nofaceflip
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
///
/// THE FLIP IS DECIDED ON THE TRUE FACE NORMAL, not on the interpolated one,
/// and that distinction is the whole reason this reads the positions. On any
/// SMOOTH-shaded mesh the interpolated normal crosses the view horizon
/// (`n·d > 0`) in a band at every silhouette while the FACE is still
/// front-facing — that is what a smooth silhouette IS. Flipping there points
/// `n` INTO the solid, and since `n` is also the eps-offset axis, every
/// secondary ray then starts inside the surface: shadow rays self-occlude,
/// `onb(n)` aims the AO hemisphere into the geometry so `ao` comes back
/// EXACTLY 0, the direct loop's `ndl <= 0` skips every light, and the pixel
/// lands on exactly (0,0,0) — a hard black band on the limb of every smooth
/// curved surface (found on the powerplant's smokestack, whose vertex normals
/// the loader welds smooth). Only the face normal can tell that band apart
/// from a genuine backface, which is why the positions are read here.
///
/// The face normal is derived ONLY inside the `n·d > 0` branch: everywhere
/// else the two criteria provably agree, so the common path pays nothing and
/// stays bit-identical. Inside it, the face only gets to arbitrate when the
/// interpolated normal actually lies in its hemisphere (`nf·n > 0`) — a mesh
/// whose winding disagrees with its authored vertex normals keeps the old
/// unconditional flip, which is what makes this equivalent to the general
/// "orient the shading normal to the ray-oriented face normal's side" rule
/// rather than merely equivalent-where-the-data-is-clean. Mirrored by
/// `shade.hlsli::surface_point` — change both together.
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
        // Genuine backface, or a smooth silhouette's past-horizon band? Ask
        // the face — but only where the face is entitled to answer. Two
        // guards, and BOTH keep the old unconditional flip when they fire:
        //   `nf·n <= 0` — the vertex normals do NOT sit in this face's
        //     hemisphere, so the winding disagrees with the authored normals
        //     and the face cannot arbitrate. Also catches the degenerate face
        //     (zero cross ⇒ exactly 0.0) and NaN.
        //   `!(nf·d < 0)` — the face really is backfacing.
        // Without the first, a thin sheet whose winding is inverted relative
        // to its normals would render its back side with the offset INSIDE
        // the solid: the exact black band this guard exists to remove, moved
        // onto a different population. The OnceLock deref rides inside this
        // branch, so the common path pays nothing for the repro lever
        // (FR_ABL=nofaceflip).
        let e1 = scene.positions[i1 as usize] - scene.positions[i0 as usize];
        let e2 = scene.positions[i2 as usize] - scene.positions[i0 as usize];
        let nf = e1.cross(e2);
        if abl().nofaceflip || nf.dot(n) <= 0.0 || !(nf.dot(ray.d) < 0.0) {
            n = -n;
        }
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
/// shaders/hemi_leaf.hlsl — change both together.
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
pub(crate) fn tri_uv_basis(scene: &Scene, tri: u32) -> Option<(Vec3A, Vec3A)> {
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

/// The hit's barycentric REST-pose world position: `(1−u−v)·P0 + u·P1 +
/// v·P2` over `Scene::positions` — which is permanently the rest pose on
/// BOTH paths (foliage sway never moves vertices: the CPU shears the RAY
/// into rest space, the GPU rides TLAS instance transforms over rest-pose
/// BLASes). This is the world-space detail field's sample point: stable
/// under sway (grain must not crawl over a waving leaf), deliberately NOT
/// `ray.o + t·d` (displaced on both paths) and not the eps-offset `p`.
/// Mirrored by `shade.hlsli::tri_rest_point`.
fn tri_rest_point(scene: &Scene, tri: u32, u: f32, v: f32) -> Vec3A {
    let [i0, i1, i2] = scene.indices[tri as usize];
    scene.positions[i0 as usize] * (1.0 - u - v)
        + scene.positions[i1 as usize] * u
        + scene.positions[i2 as usize] * v
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
    // against n gives du, dv. `den` is the basis' signed area PROJECTED onto
    // n — re-checked (tri_uv_basis already rejected a degenerate basis)
    // because n is the interpolated shading normal, not the exact face
    // normal, and at silhouettes it can tilt nearly into the basis plane.
    // The guard is RELATIVE to the basis' own area: den/|tu×tv| is the
    // cosine between n and the basis plane's normal, and an absolute
    // threshold on den (the 1e-12 this replaced) let through cosines small
    // enough that 1/den blew the gradients up to Inf — which SampleGrad
    // turns into undefined behavior (black) and sample_aniso into a NaN
    // cascade. Written `!(x >= k)` so NaN inputs also reject.
    //
    // The `a < f32::MAX` arm is not decoration: `tri_uv_basis` admits any UV
    // det over 1e-12, so |tu| can reach ~1e12 (atlas meshes), and then
    // |tu×tv|² overflows f32 and `length()` is Inf — which would make the
    // cosine test reject UNCONDITIONALLY and silently drop those meshes to
    // the isotropic lod. Where the scale is that extreme the cosine is
    // unmeasurable, so hand the case to the finiteness backstop below
    // instead. NaN `a` fails the `<` and takes the same route (the backstop
    // rejects it there).
    let axn = tu.cross(tv);
    let den = axn.dot(n);
    let a = axn.length();
    if a < f32::MAX && !(den.abs() >= 1e-3 * a) {
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
    let (gu, gv) = (to_uv(w_maj), to_uv(w_min));
    // Overflow backstop: a huge-but-accepted basis (atlas meshes reach
    // |tu|,|tv| ~ 1e11 off a det just over tri_uv_basis' floor) can still
    // overflow the numerator products to Inf. Non-finite gradients must
    // never reach a sampler — None falls back to the isotropic lod, whose
    // log terms are bounded (coarser, never wrong).
    if !(gu.is_finite() && gv.is_finite()) {
        return None;
    }
    Some((gu, gv))
}

/// ONE cosine-sampled GI gather at a shading point: draw a direction about the
/// GEOMETRIC normal, trace, and shade the hit at `gq` — or `sky::gather` on a
/// miss, which carries NO sun disc (the once-per-path rule: `direct_d` already
/// delivers the sun with its own shadow ray, and the hemi GI leaf takes the
/// identical arm for the identical reason).
///
/// Shared by BOTH rungs of the ladder — the deterministic arm
/// (`rtgi_bounces >= 1`) and the stochastic arm's continued path — so the two
/// can never disagree about what "the gather at this level" means. That shared
/// definition is exactly what the unbiasedness gate rests on: the roulette's
/// expectation is the deterministic rung's value only if both compute the same
/// integrand from the same draws.
///
/// Cosine importance sampling makes the single sampled radiance the estimate of
/// the irradiance-convention ambient directly (no π — the `irradiance()`
/// L-in-L-out convention), so a fresh draw per frame converges to the true
/// gathered GI under accumulation and reads as laundered noise to the temporal
/// denoisers (the 1-spp contract).
#[allow(clippy::too_many_arguments)]
fn rtgi_gather(
    scene: &Scene,
    bvh: &Bvh,
    p: Vec3A,
    n: Vec3A,
    gq: &Quality,
    rng: &mut fastrand::Rng,
    sun: Vec3A,
    cl: &crate::clouds::Clouds,
    depth: u32,
    ls: &mut LocalStats,
) -> Vec3A {
    let (t1, t2) = onb(n);
    let r1 = rng.f32();
    let r2 = rng.f32();
    let dir = cosine_dir(n, t1, t2, r1, r2);
    ls.secondary_rays += 1;
    // Level 0 vs level 1-and-deeper: the pair the ladder's must-fires read as
    // its rung signature (stats.rs).
    if depth == 0 {
        ls.rtgi_rays += 1;
    } else {
        ls.rtgi_rays2 += 1;
    }
    let bray = Ray::new(p, dir);
    // FR_ABL=nogi: drop the ray AND the shade behind it (the norefl shape — a
    // recursive consumer's cost probe removes the whole continuation): the
    // gather degrades to the unoccluded sky. The draws above still ran, so the
    // stream and the sample directions are unchanged.
    let bhit = if abl().nogi {
        None
    } else {
        bvh.intersect(scene, &bray, 0.0, f32::INFINITY, &mut ls.ray_nodes)
    };
    match bhit {
        Some(bh) => shade(
            scene,
            bvh,
            &bray,
            &bh,
            None,
            gq,
            rng,
            sun,
            cl,
            Cone::bounce(),
            depth + 1,
            ls,
            None,        // secondaries never capture prim
            VisCtl::Off, // bounce rays never share visibility
            None,        // no hemi share off the fb tiers
            None,        // fireflies don't light bounce surfaces (the stars rule)
            None,        // bounce surfaces take no cluster NEE (the hemi rule)
        ),
        None => crate::sky::gather(dir, sun, scene.sky_scale, scene.night, scene.light_gain),
    }
}

/// The quality a GI gather's hit shades at: hemi's leaf policy, with the bounce
/// budget the caller chose and the emissive display-add INHERITED rather than
/// recomputed.
///
/// That inheritance is load-bearing at rung 2, where the deterministic arm runs
/// at depth 1 as well: there `el` is already `None` (the gather passes no
/// cluster NEE down), so a bare `el.is_none()` would read TRUE and re-enable
/// the emissive add on the second bounce even though NEE is live and delivering
/// it — a double count. Anding with the parent's decision keeps the NEE-keep
/// rule correct at every level while staying exactly `el.is_none()` at depth 0,
/// where every session's primary Quality has `emissive_display: true`.
fn gather_q(q: &Quality, el: Option<&crate::emissive::EmissiveLights>, budget: f32) -> Quality {
    Quality {
        emissive_display: q.emissive_display && el.is_none(),
        rtgi_bounces: budget,
        ..crate::hemi::BOUNCE_Q
    }
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
    // Emissive cluster lights (src/emissive.rs) — Some ONLY on the primary
    // camera path, and NEVER under fb.gi (the once-per-path rule INVERTED:
    // the GI gather already delivers emissive transport exactly — real
    // geometry, real soft shadows, textured emission — so GI frames keep the
    // gather and drop the lossy-cluster NEE instead; render.rs's one Some
    // site gates on !q.fb.gi). The recursion, the hemi tier, and both GI
    // references pass None like ff.
    el: Option<&crate::emissive::EmissiveLights>,
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
    let mat_idx = scene.tri_mat[hit.tri as usize] as usize;
    let mat = &scene.materials[mat_idx];
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
    // `detail` carries the Unreal-1 detail field's per-q-unit 3D gradient
    // out of the detail block for the micro-bump below; None = the field
    // never fired (lever off, window closed, s == 0, or a transmissive
    // material) — the structural off state those materials shade
    // bit-identically past.
    let mut detail: Option<Vec3A> = None;
    // The field's signed height (value − 1, mean 0) for the cavity AO below;
    // 0.0 = never fired, the structural off state (detail_cavity's callers
    // branch on `< 0`).
    let mut detail_h: f32 = 0.0;
    // Horizon-march capture (q3, dlod): Some while the AO band is open — the
    // marched sun shadow below re-samples the field along the sun's tangent
    // direction. Plain values, so nothing borrows the texture.
    let mut detail_march: Option<(Vec3A, f32)> = None;
    // Spec-AA: the detail field's discarded-octave slope variance (0.0 =
    // nothing to transfer — lever off, no field, or every window fully
    // open), folded into rough_eff at the fold site below.
    let mut s2_detail: f32 = 0.0;
    let mut albedo = match mat.kind {
        MatKind::Marble { scale } => marble(ray.o + ray.d * hit.t, scale),
        MatKind::Textured { tex } => {
            let uv = scene.tri_uv(hit.tri, hit.u, hit.v);
            filt.sample(&scene.textures[tex as usize], uv, true)
        }
        _ => mat.albedo,
    };
    // The Unreal-1 detail block — for EVERY albedo source (textured, flat,
    // marble): the field's domain is the rest-pose position over the
    // per-material texel scale, so it never needed UVs, only a scale and a
    // fade window; untextured materials get both from world space
    // (scene::derive_detail_scales' synthetic arm + the cone-footprint
    // window below). Transmissive materials are EXCLUDED (the visor/water
    // finding): their "albedo" is the transmission tint, and graining it
    // mottles glass; the bump would frost it (see DETAIL_ROUGH_*).
    if crate::scene::detail_tex() && mat.transmission == 0.0 {
        // The per-material texel scale (Scene::detail_scales — never
        // per-face, which seams on greedy-meshed atlases; see the field
        // doc). s == 0 (a Textured material with no valid UV basis
        // anywhere, `--detail-untex-scale 0`, or a hand-built scene that
        // skipped finalize) closes the window below — structural off,
        // coarser never wrong.
        let s = scene.detail_scales.get(mat_idx).copied().unwrap_or(0.0);
        let dlod = match mat.kind {
            // The albedo texture's COMPLETED isotropic lod. The iso arm's
            // base is already in `filt` (free); the aniso arm derives its
            // lod inside the sampler, so recompute the base here (L1-hot —
            // the triangle rows were just walked). dlod < 0 is
            // magnification: the texels can no longer resolve the ray
            // cone's footprint, exactly where Unreal 1 faded its detail
            // texture in.
            MatKind::Textured { tex } => {
                let base = match filt {
                    TexFilter::Lod(b) => b,
                    // The MINOR axis, not the isotropic recompute: the aniso
                    // sampler resolves the footprint down to its short axis,
                    // and the isotropic lod carries the major axis's
                    // -log2|n·d| view-tilt stretch — which closed the window
                    // on every grazing-VIEWED face while SampleGrad kept its
                    // albedo texel-sharp (the Minecraft-tops finding: block
                    // sides detailed, tops flat, a binary flip between
                    // adjacent faces of one cube). See detail_aniso_base.
                    TexFilter::Aniso { gu, gv, .. } => detail_aniso_base(gu, gv),
                };
                base + scene.textures[tex as usize].lod_dims()
            }
            // Untextured: the same window measured directly in the field's
            // own q-domain — the cone footprint in texel-equivalents
            // (`detail_untex_window`, the D2-gated single source; cone_w is
            // the footprint's MINOR axis — the extent across travel; the
            // major carries the −log2|n·d| view-tilt stretch — deliberately
            // matching the textured aniso convention above, grazing-grain
            // aliasing accepted at the same price). NOT filt's untextured
            // Lod(−∞), which would saturate every octave window wide open
            // (un-antialiased grain at every distance); s == 0 parks the
            // window closed — the bitwise pre-untextured-arm off.
            _ => detail_untex_window(cone_w, s),
        };
        // Spec-AA transfer capture (`--no-spec-aa` kills): the slope
        // variance of the detail tilt NOT applied because its windows have
        // closed. Deliberately OUTSIDE the window gate below — at
        // dlod >= DETAIL_AO_RANGE (every window shut, both arms dead) the
        // transfer is at its MAXIMUM, exactly the "distant surface must go
        // matte" regime. s == 0 (no field ever) keeps the exact-0.0 off
        // state; a fully-open window returns an IEEE-exact 0.0 (1 − 1·1),
        // so magnified pixels are bit-identical through the fold's branch.
        if crate::scene::spec_aa() && s > 0.0 {
            s2_detail = detail_var(dlod);
        }
        let ao_band = dlod < DETAIL_AO_RANGE && crate::scene::detail_ao();
        if (dlod < 0.0 || ao_band) && s > 0.0 {
            // The world-space domain: q3 = rest-pose position over s.
            let q3 = tri_rest_point(scene, hit.tri, hit.u, hit.v) / s;
            if dlod < 0.0 {
                let (f, g) = detail_field(q3, dlod);
                albedo *= f;
                detail = Some(g);
                detail_h = f - 1.0;
            }
            // The AO/relief coarse octaves fire far past the grain (their
            // 8/4-texel cells resolve until dlod = 3/2) — mid-distance
            // pools of occlusion AND relief rims: the pools' gradient
            // joins the micro-bump (scaled DETAIL_AO_BUMP_K vs the grain),
            // so mid-frequency relief is lit directionally out where the
            // grain-only bump has faded (the flat-tops finding — the eye
            // sees pool-scale structure, not texel grain, at distance).
            // Gated on the AO lever so the off arm never pays the evals
            // and `detail`/`detail_march` stay None-shaped.
            if ao_band {
                let (hp, gp) = detail_ao_field(q3, dlod);
                detail_h += hp;
                if gp != Vec3A::ZERO {
                    detail = Some(detail.unwrap_or(Vec3A::ZERO) + gp * DETAIL_AO_BUMP_K);
                }
                detail_march = Some((q3, dlod));
            }
        }
    }
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
    // Unreal-1 detail micro-bump: the SAME field that just modulated the
    // albedo tilts the SHADING normal by its analytic gradient's tangential
    // projection, so dark grains sit in concave pits. Composes ON the normal
    // map (and the ripple below rides on top in turn); geometric n untouched
    // — the n_g/n_s split. `detail` is Some only when the field fired, so
    // far pixels, degenerate-basis hits, and lever-off sessions never reach
    // this branch (no tangent frame needed here any more — the projection is
    // frame-free).
    // The bump is additionally damped by the PER-PIXEL roughness
    // (detail_bump_weight): a tight specular lobe frosts under normal
    // scatter, so polished pixels keep their mirror while matte ones keep
    // their grain — one material can be both (DamagedHelmet's visor vs
    // shell).
    // Pre-detail shading normal, retained iff detail_bump actually ran: the
    // direct loop's contrast cap (DETAIL_NDL_CAP) clamps the sun's N·L
    // relative to this — the detail feature's own off state. Sound to capture
    // BEFORE the ripple below: ripple and detail are structurally disjoint
    // (water is transmissive, and transmissive materials skip the whole
    // detail field), so on any detail pixel n_pre is the final n_s minus
    // exactly the detail tilt.
    let mut n_pre: Option<Vec3A> = None;
    if let Some(g) = detail {
        let bw = detail_bump_weight(rough_eff);
        if bw > 0.0 {
            n_pre = Some(n_s);
            n_s = detail_bump(n_s, n, g * bw);
        }
    }
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
    // Spec-AA fold (`--no-spec-aa` kills): the slope variance the mip/window
    // pipeline resolved AWAY comes back as a wider GGX lobe — the
    // Toksvig/LEAN identity α′² = α² + 2σ² (α = roughness², σ² = mean
    // per-axis slope variance), so detail maps stay in the rendering
    // equation at every distance: a distant bumpy surface shades matte-rough
    // instead of collapsing to a mirror-flat mean normal. Two sources, each
    // an exact 0.0 wherever nothing was resolved away, and the identity is
    // BY BRANCH (`s2 > 0.0`) — sqrt(sqrt(x⁴)) is NOT an f32 bitwise
    // identity:
    //  - the normal map's variance companion (scene.tex_var): level 0 is
    //    all-zero, so magnification reads an exact 0.0 through the lod ≤ 0
    //    bilinear escape; sampled through the SAME `filt` as the map itself
    //    (footprints agree by construction), guarded by perturb_normal's
    //    own triple so a hit that never decoded the map never folds it;
    //    ×normal_scale² — the decode scales slopes linearly;
    //  - the detail field's faded octaves (`s2_detail`), ×bw² — the bump
    //    applies its gradient through the same weight, so applied (bw²·wk²)
    //    plus transferred (bw²·(1−wk²)) is bw²·full at EVERY distance, and
    //    a polished surface (bw = 0) is never frosted by detail it would
    //    never have shown.
    // Sits AFTER the ripple and BEFORE the PrimarySurface capture:
    // ggx_alphas, the sheen inverse-alpha, and the denoiser guides all see
    // the widened lobe, while detail_bump_weight above read the PRE-fold
    // roughness (no feedback) and the reflection-lobe gate below keeps
    // reading the FLAT mat.roughness (the rng-schedule rule — the fold may
    // move the lobe, never the draw schedule). Zero rng draws, pure
    // function of (hit, texels) — every same-seed/replay/VisCtl contract
    // holds. Mirrored term-for-term in shade.hlsli.
    if crate::scene::spec_aa() {
        let mut s2 = 0.0f32;
        if s2_detail > 0.0 {
            let bw = detail_bump_weight(rough_eff);
            s2 += bw * bw * s2_detail;
        }
        if let (true, Some(uv), Some(_)) = (mat.normal_tex != NO_TEX, map_uv, uv_basis) {
            if let Some(&vid) = scene.tex_var.get(mat.normal_tex as usize) {
                if vid != NO_TEX {
                    let u = filt.sample(&scene.textures[vid as usize], uv, false).x;
                    s2 += crate::texture::spec_aa_decode(u) * mat.normal_scale * mat.normal_scale;
                }
            }
        }
        if s2 > 0.0 {
            rough_eff = spec_aa_fold(rough_eff, s2);
        }
    }
    if let Some(prim) = prim.as_deref_mut() {
        *prim = PrimarySurface {
            n: n_s,
            albedo,
            roughness: rough_eff,
            metallic: metal_eff,
            trans: mat.transmission,
            spec_t: 0.0,
            ripple_amp: mat.ripple_amp,
            direct_d: Vec3A::ZERO,
            direct_s: Vec3A::ZERO,
            // Both are filled below, when their tier runs; the zeros are the
            // "no such term" values the composite adds nothing for (fb.gi's
            // ambient, or a surface whose reflection gate never fired).
            ao: 0.0,
            ind_s: Vec3A::ZERO,
            // GPU-captured only (see the field docs) — 0.0 = "no data".
            ao_t: 0.0,
            shadow_t: 0.0,
            // ...except these two, whose neutral value is the multiplicative
            // identity, not 0.0 (see the field doc).
            amb_k: 1.0,
            m_d: 1.0,
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
        // DETAIL CONTRAST CAP (detail pixels only — n_pre is Some iff
        // detail_bump ran): direct contrast from a tilt δ scales as
        // tan(incidence)·δ, so the one bump strength tuned for steep-lit
        // tops (tan ≈ 0.27 at noon) overdrives grazing-lit sides (tan ≈ 3.7)
        // 14×. The cap bounds the detail tilt's modulation to ±DETAIL_NDL_CAP
        // of the PRE-detail N·L: under-cap pixels (tops) return raw bitwise,
        // grazing faces compress to a fixed contrast ceiling. The p <= 0 arm
        // kills bright speckle on the shadow side of the terminator (detail
        // may not light a pre-detail-unlit facet); both arms are continuous
        // at p = 0. Fireflies and emissive NEE below ride the SAME cap
        // (capped_ndl — one rule per light family, round 6b).
        let ndl = capped_ndl(n_s, n_pre, wi);
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
            // The detail contrast cap applies to every direct-tier light
            // (round 6b — the "other light sources" uniformity rule).
            let ndl = capped_ndl(n_s, n_pre, wi);
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
    // Emissive cluster lights (src/emissive.rs) — the direct tier's third
    // entry, the firefly block's exact shape: AFTER the cloud scaling (a
    // lamp under the slab is a local light — the sun's cloud transmittance
    // must not dim it) and BEFORE the prim capture (the light rides FSR-RR's
    // denoised dd lobe; albedo-free radiance like the sun's `li`, so the
    // composite identity closes untouched). ZERO rng draws — deterministic
    // iteration, one HARD shadow ray per in-range light. DIFFUSE-ONLY, and
    // that is a correctness decision, not a shortcut: an emitter has real
    // geometry, so its specular highlight is DELIVERED by the traced VNDF
    // ray hitting the glowing surface (the display `color += e` at every
    // depth) — a firefly-shaped `w_l = 1` specular term here would
    // double-count it, and the mirror image cannot be down-weighted (MIS'd
    // specular is the documented follow-on). The shadow ray stops
    // rc + 2·eps SHORT of the cluster center so the emitter's own bulb/
    // lamp-glass geometry cannot occlude its own light (known-accept:
    // non-emissive geometry inside the cluster sphere doesn't occlude it
    // either). Emissive-free scenes hand shade a structural None — the
    // pre-feature renderer bit-identically.
    if let Some(el) = el {
        for i in 0..el.count as usize {
            let l = &el.lights[i];
            let to = Vec3A::from(l.pos) - p;
            let d2 = to.length_squared();
            // The rejection test — a far light's only cost.
            if d2 >= l.r_infl2 {
                continue;
            }
            let dist = d2.sqrt();
            let wi = to / dist;
            // The detail contrast cap — same rule as the sun/firefly tiers.
            let ndl = capped_ndl(n_s, n_pre, wi);
            if ndl <= 0.0 {
                continue;
            }
            // The emission lobe: `wi` points from the shading point TO the
            // light, so the direction FROM the light to the receiver is -wi.
            // Bounded by 1, so this can only ever REMOVE light — no in-range
            // test, no influence radius and no tile cull is perturbed.
            let e =
                crate::emissive::irradiance(l, d2, scene.light_gain) * crate::emissive::lobe(l, -wi);
            if e == Vec3A::ZERO {
                continue;
            }
            ls.secondary_rays += 1;
            ls.emissive_rays += 1;
            let vis_t = bvh.transmittance(
                scene,
                &Ray::new(p, wi),
                0.0,
                (dist - l.rc2.sqrt() - 2.0 * scene.eps).max(0.0),
                &mut ls.ray_nodes,
            );
            if vis_t == Vec3A::ZERO {
                continue;
            }
            direct_d += e * ndl * vis_t;
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
    let mut ambient = if q.fb.gi {
        let (t1, t2) = onb(n);
        let accel = crate::ftree::Accel::of(bvh);
        crate::hemi::gi(scene, accel, p, n, t1, t2, q.fb.depth, sun, cl, depth, hemi_share, rng, None, ls)
    } else if q.rtgi_bounces >= 1.0 && !q.fb.ao {
        // REAL-TIME GI, DETERMINISTIC RUNG: one cosine-sampled gather IS the
        // ambient (`rtgi_gather`). The hit shades at hemi's leaf policy with
        // the budget DECREMENTED — so at `--rtgi-bounces 2` this same arm runs
        // again one level down and the second bounce is real, while at 1 the
        // child gets 0.0 and its SH×AO tail closes the path exactly as it
        // always has. The budget IS the recursion bound: nothing else needs to
        // know the depth, and the reflection/glass children (which set 0.0
        // explicitly) can never inherit one.
        //
        // The still-frame fb tiers take precedence (fb.gi above, fb.ao via the
        // guard here). Runs IDENTICALLY under VisCtl Off/Capture/Apply — it
        // reads and writes no VisRecord field, and both arms draw the same
        // stream, so no burn is needed and the adaptive same-seed alignment
        // holds per-pixel. `prim.ao` stays 0.0 (the fb.gi precedent: real RGB
        // irradiance is not an AO scalar — the GI term rides FSR's
        // exact-remainder residual). NOTE the GPU twin diverges here BY DESIGN
        // under FLAG_NRD_GI (shade.hlsli's shade_full): NRD sessions are
        // GPU-tracer-only, and there the bounce folds into prim.direct_d
        // (+ its t into ao_t) so ReBLUR's diffuse input carries the GI — this
        // CPU capture never feeds NRD, so it keeps the residual arm verbatim.
        //
        // The NEE-keep rule (2026-08-08, the XeSS feel-test) rides `gather_q`:
        // when cluster NEE is live this frame the gather must NOT re-deliver
        // emitter-as-emitter transport, so its whole subtree shades with the
        // emissive display-add suppressed.
        let bq = gather_q(q, el, q.rtgi_bounces - 1.0);
        rtgi_gather(scene, bvh, p, n, &bq, rng, sun, cl, depth, ls)
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
        // split reserves the geometric normal for visibility. Through
        // amb_irradiance so bumped normals get a first-order sky response
        // (the order-2 SH alone is too smooth to show texel relief).
        let tail = amb_irradiance(&scene.sky_sh, n, n_s) * ao;
        // THE STOCHASTIC RUNG (`--rtgi-bounces 0.5`, `1.5`): Russian roulette
        // on the DELTA OVER THE TAIL.
        //
        // `tail` already approximates the transport a real gather at this level
        // would compute, so continuing with probability `pr` and weighting the
        // difference by 1/pr is UNBIASED for that gather:
        //
        //     E[A] = (1-pr)*tail + pr*(tail + (G - tail)/pr) = G
        //
        // while the variance rides on (G - tail)^2 instead of G^2. That is the
        // whole reason to roulette the DELTA and not the term: the tail is a
        // control variate, and it is exactly as good as the SH×AO
        // approximation is at this surface — small in the open, largest in
        // enclosures, which is where the samples are worth spending. A plain
        // coin flip between the two rungs would instead deliver half the
        // gather, i.e. bias, which is the trap this arm exists to avoid.
        //
        // The temporal integrators launder the per-pixel decision exactly as
        // they launder the 1-spp shadow/AO rays. `G` comes from the SAME
        // `rtgi_gather` the deterministic arm calls, which is what makes the
        // expectation land on the next rung up rather than merely near it.
        //
        // A BRANCH, never a computed weight (frd_temporal.hlsl's rule): at
        // budget 0 this is today's expression BITWISE and draws nothing, which
        // is what keeps rungs 0 and 1 byte-identical to the pre-ladder
        // renderer. The fb tiers take precedence exactly as in the
        // deterministic arm above.
        if q.rtgi_bounces <= 0.0 || q.fb.ao {
            tail
        } else {
            let pr = q.rtgi_bounces.min(1.0);
            // ONE draw for the decision, then the gather's own — so a continued
            // pixel's stream is LONGER than a terminated one's. Sound because
            // the decision is a pure function of this pixel's own stream:
            // VisCtl Capture and Apply take the same branch at the same
            // position (the deterministic arm's own no-burn argument), and
            // pixels have independent streams by construction
            // (render::primary_seed). Nothing downstream of the ambient tier
            // reads a POSITION in the stream, only values.
            let u = rng.f32();
            if u >= pr {
                tail
            } else {
                // The gather's own hit is a LEAF (budget 0.0): the quantity
                // this rung estimates is "the deterministic gather whose
                // ambient is the tail", which is exactly what the rung above
                // computes. Letting it recurse would estimate a different
                // integral and quietly break the unbiasedness gate's oracle.
                let gq = gather_q(q, el, 0.0);
                let g = rtgi_gather(scene, bvh, p, n, &gq, rng, sun, cl, depth, ls);
                if rtgi_rr_noweight() {
                    // The naive coin flip — the gate's teeth (see the lever).
                    // `tail + (g - tail)` IS `g`, so take it directly rather
                    // than writing the identity out.
                    g
                } else {
                    tail + (g - tail) / pr
                }
            }
        }
    };

    // Detail cavity AO — AFTER the PrimarySurface captures (prim.ao,
    // prim.direct_d/direct_s are already written), so every FSR signal stays
    // un-cavitied and the deterministic delta lands in the exact-remainder
    // residual (fsr::split_signals subtracts the exported signals from the
    // FINAL color — texel-crisp under FSR-RR, zero wire-format contact; the
    // one asymmetry: a cavity on a reflection LAP rides the denoised ind_s
    // instead, identity closing either way). Guarded, never `* 1.0`:
    // detail_h == 0.0 on every non-fired hit and > 0 on peaks, so lever-off /
    // detail-off / dlod >= 0 / peaks are all structural. `detail_h < 0.0`
    // first, so the atomic load is skipped on the non-fired majority. Zero
    // rng draws. Direct diffuse (N·L + shadows carry the sun's contrast),
    // direct_t, emissive and the transmission chain stay untouched.
    if detail_h < 0.0 && crate::scene::detail_ao() {
        let cav = detail_cavity(detail_h);
        ambient *= cav;
        direct_s *= cav;
    }
    // REAL horizon-marched sun shadow: a closed-form occlusion trace of the
    // detail heightfield toward the sun (detail_sun_shadow — the statistical
    // micro_shadow it replaces darkened pits by depth with no direction; the
    // march tests actual upstream terrain against the sun ray, so shadows
    // fall away from the sun and lengthen as it drops). Same post-capture
    // placement: prim.direct_d is already exported, the delta rides the
    // residual. Shading-only — visibility, the BVH, and every rng stream are
    // untouched. `detail_march` is Some only under the AO lever with the
    // band open (and a sound basis for the q3 scale), so lever-off/off-band/
    // untextured/degenerate are structural. The march direction is the sun's
    // tangent-plane projection — the same tangential projection the bump
    // applies, so the march's azimuth agrees with the bump's tilt by
    // construction (no frame to keep in lockstep any more).
    if let Some((q3, ddl)) = detail_march {
        let l = scene.sun.dir;
        let lt = l - n * n.dot(l);
        direct_d *= detail_sun_shadow(q3, ddl, lt, n.dot(l));
    }

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
    // the default-material bit-identity contract. This is the DISPLAY half
    // of emissive transport (the sky::radiance analogue); the GATHER half is
    // the cluster-light NEE loop above (src/emissive.rs — direct tier, lap
    // 0, ARMED by --emissive-lights) and, under fb.gi, the hemi gather
    // picking this very term up off bounce hits. At most one of those two
    // lights other surfaces per frame (FLAG/fb gating), and this add is
    // never conditional on either.
    // `q.emissive_display` is TRUE on every path except the RTGI bounce
    // subtree while cluster NEE is live (the NEE-keep rule — see the Quality
    // field doc): there the NEE already delivers emitter-as-emitter
    // transport, so this add would double-count it.
    if q.emissive_display && (mat.emissive != Vec3A::ZERO || mat.emissive_tex != NO_TEX) {
        let e = match (mat.emissive_tex != NO_TEX, map_uv) {
            (true, Some(uv)) => {
                mat.emissive * filt.sample(&scene.textures[mat.emissive_tex as usize], uv, true)
            }
            _ => mat.emissive,
        };
        // The scene light gain (`--autoexp-mode lights`) reaches emissive
        // surfaces HERE. It has to: `mat.emissive` lives in the serialized
        // material stream, which `apply_light_gain` cannot rescale per frame
        // — and an emitter left un-gained while everything around it
        // brightens reads as the emitter DIMMING as the aperture opens.
        // Exactly 1.0 in every default and headless session, and `x * 1.0` is
        // bit-preserving, so the pre-feature arm is untouched.
        color += e * scene.light_gain;
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
                    // rtgi_bounces: 0.0 for parity with the GPU's lap-0-only shape
                    // (the depth gate already makes inheritance inert).
                    let rq = Quality { fb: FrustumBounce::OFF, rtgi_bounces: 0.0, ..*q };
                    // The child cone starts at this hit's width — reflected
                    // hits read footprints grown by the full path length.
                    let rcone = Cone { w0: cone_w, spread: cone.spread, aniso: cone.aniso };
                    shade(scene, bvh, &rray, &rh, None, &rq, rng, sun, cl, rcone, depth + 1, ls, None, VisCtl::Off, None, None, None)
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
                            + crate::sky::stars(rdir, cone.spread * 0.5, scene.night, 0, scene.light_gain);
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
            let rq = Quality { fb: FrustumBounce::OFF, rtgi_bounces: 0.0, ..*q };
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
                            ls, None, VisCtl::Off, None, None, None,
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
                            scene.light_gain,
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
/// `pub(crate)` for exactly one outside consumer: `ngxfg_guides::self_test`'s
/// reconstruction-fidelity gate scores `ripple_prev_normal` against this
/// function evaluated at `t_prev` — the ground truth it claims to reconstruct.
pub(crate) fn ripple_normal(base: Vec3A, n: Vec3A, p: Vec3A, t: f32, amp: f32, diag: f32) -> Vec3A {
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
        normal_role: false,
        mips: Vec::new(),
        var_mips: Vec::new(),
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
            emissive: crate::emissive::EmissiveLights::off(),
            sun: crate::sky::Sun::new(Vec3A::Y),
            sky_sh: crate::sh::Sh9::ZERO,
            sky_scale: 1.0,
            night: 0.0,
            light_gain: 1.0,
            light_canon: crate::scene::LightCanon::default(),
            sway: None,
            sway_regions: Vec::new(),
            diag: 1.0,
            eps: 1e-4,
            ao_radius: 0.03,
            detail_scales: Vec::new(),
            content_min: Vec3A::ZERO,
            content_max: Vec3A::ZERO,
            tex_var: Vec::new(),
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

    // --- the grazing hardening (the black-at-extreme-angles fix) -----------
    // An exact silhouette (n·d = 0) with a sound basis must still produce a
    // footprint — finite, minor axis = cone_w, major pinned at the 0.05
    // floor's 20× stretch. The guards below must not over-reject this regime.
    let Some((gu, gv)) = tri_grads(&sc, 0, n, Vec3A::X, cw) else {
        return Err("tri_grads must survive an exact silhouette (n·d = 0)".into());
    };
    if !(gu.is_finite() && gv.is_finite()) {
        return Err(format!("silhouette footprint not finite: {gu:?} {gv:?}"));
    }
    let (maj, min) = (gu.length().max(gv.length()), gu.length().min(gv.length()));
    if (min - cw).abs() > 1e-6 || (maj - cw / 0.05).abs() > 1e-4 {
        return Err(format!("silhouette footprint {min}×{maj}, want {cw}×{}", cw / 0.05));
    }
    // A shading normal tilted nearly INTO the basis plane (silhouettes on
    // smooth-shaded geometry): den/|tu×tv| = 5e-4 sits under the 1e-3
    // threshold and must REJECT — the old absolute `|den| < 1e-12` guard
    // accepted it, 1/den blew the gradients toward Inf, and SampleGrad with
    // non-finite gradients is UB (the black-surface symptom).
    let basis = (Vec3A::X, Vec3A::Y);
    let n_graze = Vec3A::new(1.0, 0.0, 5.0e-4).normalize();
    if tri_grads_from(basis, n_graze, -Vec3A::Z, cw).is_some() {
        return Err("near-in-plane shading normal must reject (den guard)".into());
    }
    // RELATIVE is the point: the same cosine on a 1e6-scaled basis carries an
    // absolute den of ~5e8, which any absolute threshold accepts — the teeth
    // against reverting the guard to `|den| < eps`.
    let big = (Vec3A::X * 1.0e6, Vec3A::Y * 1.0e6);
    if tri_grads_from(big, n_graze, -Vec3A::Z, cw).is_some() {
        return Err("den guard must be relative to the basis' area".into());
    }
    // Just ABOVE the threshold the footprint must come back finite — the
    // guard is a boundary, not a blanket reject of grazing normals.
    let n_ok = Vec3A::new(1.0, 0.0, 2.0e-3).normalize();
    match tri_grads_from(basis, n_ok, -Vec3A::Z, cw) {
        Some((gu, gv)) if gu.is_finite() && gv.is_finite() => {}
        Some(_) => return Err("above-threshold footprint must be finite".into()),
        None => return Err("cosine just above the den threshold must not reject".into()),
    }
    // A basis so large that |tu×tv| OVERFLOWS f32 must NOT be rejected by the
    // cosine test — `length()` is then Inf and a naive relative guard rejects
    // unconditionally, silently dropping atlas meshes (|tu| ~ 1e12 off
    // tri_uv_basis' 1e-12 det floor) to the isotropic lod. The cosine is
    // unmeasurable at that scale, so the finiteness backstop is what decides;
    // here it survives with a finite footprint. ANTI-VACUITY first: the
    // overflow must actually happen, or this probe proves nothing.
    let huge = (Vec3A::X * 1.0e13, Vec3A::Y * 1.0e13);
    if huge.0.cross(huge.1).length().is_finite() {
        return Err("probe is vacuous: |tu×tv| must overflow f32 here".into());
    }
    match tri_grads_from(huge, Vec3A::Z, -Vec3A::Z, cw) {
        Some((gu, gv)) if gu.is_finite() && gv.is_finite() => {}
        Some(_) => return Err("overflowing basis must not yield a non-finite footprint".into()),
        None => return Err("overflowing |tu×tv| must not reject (aniso silently off)".into()),
    }
    Ok(())
}

/// Pure self-test for `surface_point`'s face-decided flip (run by `--check`).
///
/// The case that matters is the SMOOTH SILHOUETTE band: a front-facing face
/// whose interpolated normal has already crossed the view horizon. Flipping
/// there aims the eps offset INTO the solid, which self-occludes every
/// secondary ray and renders the band exactly black. The gate asserts the
/// returned point lands OUTSIDE the face plane, and carries TEETH — the
/// pre-fix answer (`-n`) must provably fail that same bound, so the probe
/// cannot go vacuous if the guard is reverted. Genuine backfaces, degenerate
/// faces and ordinary front hits are pinned alongside so the fix cannot buy
/// the band by breaking them.
pub fn surface_point_self_test() -> Result<(), String> {
    use crate::scene::SceneBuilder;
    // A face tilted 85° off the view axis — front-facing, but only just, the
    // way a facet is at a smooth limb. Vertex normals sit 10° further round,
    // so they are PAST the view horizon while staying in the face's own
    // hemisphere (cos 10° = 0.985) — exactly a tessellated cylinder's limb.
    let d = Vec3A::new(0.0, 0.0, -1.0);
    let (s85, c85) = 85f32.to_radians().sin_cos();
    let (s95, c95) = 95f32.to_radians().sin_cos();
    let n_face = Vec3A::new(s85, 0.0, c85);
    let n_vert = Vec3A::new(s95, 0.0, c95);
    if !(n_face.dot(d) < 0.0) {
        return Err("probe setup: the face must be front-facing".into());
    }
    if !(n_vert.dot(d) > 0.0 && n_vert.dot(n_face) > 0.0) {
        return Err("probe setup: vertex normals must be past-horizon yet face-side".into());
    }
    // Triangle spanning the plane ⊥ n_face, wound so cross(e1, e2) == n_face.
    let t1 = Vec3A::Y;
    let t2 = n_face.cross(t1);
    let (p0, p1, p2) = (Vec3A::ZERO, t1, t2);

    let build = |normals: [Vec3A; 3], p: [Vec3A; 3]| {
        let mut b = SceneBuilder::new();
        let m = b.material(Vec3A::splat(0.5), 0.8, 0.0);
        b.tri(p, normals, m);
        b.finish(crate::scene::default_sun())
    };
    let hit = Hit { t: 2.0, tri: 0, u: 0.25, v: 0.25 };
    let at = |p: [Vec3A; 3]| p[0] * 0.5 + p[1] * hit.u + p[2] * hit.v;

    // (a) THE BAND. The flip must NOT fire: the face is front-facing, so the
    // interpolated normal stays outward and the offset point stays outside.
    let sc = build([n_vert; 3], [p0, p1, p2]);
    let ray = Ray::new(at([p0, p1, p2]) - d * hit.t, d);
    let (p, n) = surface_point(&sc, &ray, &hit);
    if n.dot(n_face) <= 0.0 {
        return Err(format!("smooth silhouette: normal flipped into the solid: {n:?}"));
    }
    let out = (p - p0).dot(n_face);
    if !(out > 0.0) {
        return Err(format!("smooth silhouette: ray origin {out} is inside the face plane"));
    }
    // TEETH: the pre-fix answer must fail the very bound just asserted, so a
    // revert of the guard cannot pass this gate.
    let pre_fix = (at([p0, p1, p2]) - n_vert * sc.eps - p0).dot(n_face);
    if pre_fix >= 0.0 {
        return Err(format!("probe is vacuous: the pre-fix offset {pre_fix} is not inside"));
    }

    // (b) A GENUINE backface still flips toward the ray (the case the flip
    // exists for): face normal +Z, viewed from −Z looking along +Z.
    let sc = build([Vec3A::Z; 3], [Vec3A::ZERO, Vec3A::X, Vec3A::Y]);
    let back = Ray::new(Vec3A::new(0.25, 0.25, -2.0), Vec3A::Z);
    let (p, n) = surface_point(&sc, &back, &hit);
    if n != -Vec3A::Z {
        return Err(format!("genuine backface must flip toward the ray: {n:?}"));
    }
    if !(p.z < 0.0) {
        return Err(format!("backface offset landed on the far side: {p:?}"));
    }

    // (c) An ordinary front hit is untouched — the common path, unchanged.
    let front = Ray::new(Vec3A::new(0.25, 0.25, 2.0), -Vec3A::Z);
    let (p, n) = surface_point(&sc, &front, &hit);
    if n != Vec3A::Z || !(p.z > 0.0) {
        return Err(format!("front hit must pass through untouched: {n:?} {p:?}"));
    }

    // (d) A DEGENERATE face keeps the old unconditional flip (zero cross ⇒
    // the `nf·n <= 0` arm) — coarser, never wrong.
    let sc = build([Vec3A::Z; 3], [Vec3A::ZERO; 3]);
    let (_, n) = surface_point(&sc, &back, &hit);
    if n != -Vec3A::Z {
        return Err(format!("degenerate face must keep the unconditional flip: {n:?}"));
    }

    // (e) INVERTED WINDING — vertex normals anti-aligned with `cross(e1, e2)`,
    // a modeling error the loader preserves whenever the OBJ ships a full
    // normal array. Trusting the face here would skip the flip and put the
    // offset inside the solid: the band bug moved onto a different
    // population, and worse than the pre-fix behavior. The `nf·n <= 0` guard
    // must hand these back to the unconditional flip. Face winding gives +Z
    // while the normals say −Z, and the ray travels −Z: `n·d > 0` enters the
    // branch, `nf·d < 0` would call the face front-facing and skip the flip,
    // and only the hemisphere guard rescues it. The returned `n` must oppose
    // the ray and the offset must land on the camera's side.
    let sc = build([-Vec3A::Z; 3], [Vec3A::ZERO, Vec3A::X, Vec3A::Y]);
    let (p, n) = surface_point(&sc, &front, &hit);
    if n != Vec3A::Z {
        return Err(format!("inverted winding: normal must oppose the ray: {n:?}"));
    }
    if !(p.z > 0.0) {
        return Err(format!("inverted winding: offset landed inside the solid: {p:?}"));
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

/// Unreal-1 style detail texturing — the procedural close-up field
/// (`--no-detail-tex` kills; `scene::detail_tex()` is the lever).
///
/// Three octaves of WORLD-SPACE 3D value noise multiply the sampled albedo
/// and tilt the shading normal wherever the base texture is MAGNIFIED (the
/// completed isotropic lod `dlod < 0` — texels can no longer resolve the ray
/// cone's footprint, exactly the regime Unreal 1 faded its detail texture in
/// over). Each octave k lives in its own lod window `saturate(-dlod - k)`:
/// it fades in only once it is resolvable — the window IS the anti-alias
/// (the clouds oct_t lesson) and the progressive more-detail-as-you-approach
/// ladder.
///
/// THE DOMAIN is `q3 = p_rest / s` — the hit's barycentric REST-pose world
/// position (`tri_rest_point`: stable under foliage sway, whose vertices
/// never move — the CPU shears the ray, the GPU the TLAS instances) over the
/// PER-MATERIAL texel scale `s` (`Scene::detail_scales` — a sampled median
/// of the tri_uv_basis texel-size formula over the material's triangles,
/// derived in finalize_scalars, so octave 0 stays one noise cell per
/// texel-EQUIVALENT and the field self-scales per surface). It moved off UV
/// texel space because atlas meshes (rungholt: 704 distinct vt coords for
/// 6.7M tris) repeat the same UV rect per block face, and any UV-domain
/// noise tiles in lockstep with it — blatant repeated blotches on every
/// wall; world position decorrelates by construction. And `s` is
/// deliberately NOT per-face: greedy-meshed exports make per-face texel
/// density wildly non-uniform (vokselia's Grass spans s 0.11..215 across
/// merged runs), so a per-face `s` made `q3` jump at every face boundary —
/// the block-seam artifact the first draft of this domain shipped. One `s`
/// per material keeps the field continuous across every face of a surface
/// by construction. Known accepts: grain frequency is uniform per material
/// (no within-chart density adaptation — coarser, never wrong), and the
/// finest octave's fract granularity is ~1/32 cell at THE WORLD's
/// coordinate scale (subvisible).
///
/// Grayscale and energy-neutral (value noise is symmetric about ½, so the
/// factor's mean ≈ 1 — the self-test pins it, so a retune cannot silently
/// shift exposure), bounded to [1 − ΣA, 1 + ΣA] ⊂ [0.685, 1.315] ahead of
/// the defensive floor. `clouds::vnoise3_vg` supplies value + ANALYTIC
/// gradient in one 8-corner fetch (u32-exact against the GPU's
/// `cloud_vnoise3_vg` in trace_common.hlsli), so the albedo grain and the
/// micro-bump are one coherent surface: dark crevices align with concave
/// bump. Octave salts 40..42 (ripple owns 16..19). Pure hit function, ZERO
/// rng draws — every same-seed/replay/VisCtl-burn contract holds. Mirrored
/// verbatim by `shade.hlsli::detail_field` — change both together
/// (constants are LITERALS, the clouds-wind idiom).
pub const DETAIL_AMP: f32 = 0.18;
/// Micro-bump strength, a dimensionless slope-per-texel-equivalent (the
/// gradient is per q-unit and applied by tangential projection, so the tilt
/// does not vary with mesh scale or texture resolution). Mirrored in shade.hlsli.
/// RAISED 0.35 → 1.2 by image A/B (the flat-tops campaign): tilt contrast
/// under a light at incidence θ is tan(θ)·tilt, so the old ~5° tilts read
/// only within ~5° of grazing — every flat surface had a razor-thin relief
/// window pinned at ITS grazing configuration (tops at sunset, walls at
/// noon). Real dirt facets run 30-60°; ~15-20° widens relief to most of the
/// day. COHERENCE RULE: this and DETAIL_SHADOW_HT_LO/HI describe the SAME
/// terrain steepness — move them together or shadows contradict the shading.
/// The DIRECT term additionally rides DETAIL_NDL_CAP (below): one global
/// tilt cannot fit both incidence regimes, so the crank is paired with a
/// contrast ceiling instead of being dialed back.
pub const DETAIL_BUMP_K: f32 = 2.0;

/// Direct N·L contrast ceiling for the detail bump (round 6, the
/// overdone-sides fix): the detail tilt may move the sun's diffuse N·L by at
/// most ±CAP relative to the PRE-detail N·L (`n_pre` in shade()). Contrast
/// from a tilt δ is tan(incidence)·δ — 0.27 on a noon-lit top vs 3.7 on a
/// noon-lit side — so the crank tuned for tops overdrives grazing-lit faces
/// 14×; the clamp compresses exactly the divergent-tan regime while
/// under-cap pixels (tops) keep the raw value BITWISE. The lower bound is
/// the anti-speckle floor (detail cannot extinguish a lit facet past 1−CAP),
/// and a pre-detail-unlit facet may not be lit by detail at all (min(raw,0)
/// — terminator hygiene). 0.8 → 0.5 in round 6b (the "sides still too much"
/// feel-test). Mirrored in shade.hlsli.
pub const DETAIL_NDL_CAP: f32 = 0.5;

/// The cap itself: `raw` = n_s·wi (post-detail), `p` = n_pre·wi (pre-detail).
/// Both arms are continuous at p = 0 (the bounds and the min both → ≤ 0).
/// Under-cap values return `raw` BITWISE (clamp inside its own bounds).
#[inline(always)]
pub fn detail_ndl_cap(raw: f32, p: f32) -> f32 {
    if p > 0.0 {
        raw.clamp(p * (1.0 - DETAIL_NDL_CAP), p * (1.0 + DETAIL_NDL_CAP))
    } else {
        raw.min(0.0)
    }
}

/// The cap applied to a light direction: every direct-tier N·L on a detail
/// pixel goes through this (sun, fireflies, emissive clusters — ONE rule, so
/// no light family re-opens the overdone-sides regime; the moon rides the
/// sun struct and is covered by construction). `n_pre` None (no detail) is
/// the raw dot verbatim — the structural off arm.
#[inline(always)]
pub fn capped_ndl(n_s: Vec3A, n_pre: Option<Vec3A>, wi: Vec3A) -> f32 {
    let raw = n_s.dot(wi);
    match n_pre {
        Some(np) => detail_ndl_cap(raw, np.dot(wi)),
        None => raw,
    }
}

/// The coarse pool octaves' gradient share of the micro-bump (relief RIMS),
/// relative to the grain's. The pools reach dlod < 3, so this is what keeps
/// mid-distance surfaces from going flat when the grain window closes.
/// Mirrored in shade.hlsli.
pub const DETAIL_AO_BUMP_K: f32 = 1.5;
/// The micro-bump's roughness window: zero bump at or below LO, full bump at
/// or above HI (smoothstep-free linear ramp between). A slope that reads as
/// surface grain on a matte wall FROSTS a tight specular lobe — the
/// DamagedHelmet-visor feel-test finding — so smooth surfaces keep their
/// polish. HI sits at the reflection-lobe gate's own 0.45 "glossy" threshold.
/// Evaluated on the MAP-DRIVEN per-pixel rough_eff (one material can be
/// visor-smooth and shell-rough), which is safe here because the bump draws
/// no rng — unlike the reflection gate, which must read the flat factor.
/// Mirrored in shade.hlsli.
pub const DETAIL_ROUGH_LO: f32 = 0.2;
pub const DETAIL_ROUGH_HI: f32 = 0.45;

/// The bump's roughness damping weight in [0, 1] — exact 0 at/below LO
/// (`detail_bump`'s g == 0 guard then returns the base normal verbatim),
/// exact 1 at/above HI. Mirrored by `shade.hlsli::detail_bump_weight`.
#[inline(always)]
pub fn detail_bump_weight(rough: f32) -> f32 {
    ((rough - DETAIL_ROUGH_LO) / (DETAIL_ROUGH_HI - DETAIL_ROUGH_LO)).clamp(0.0, 1.0)
}

/// Detail cavity AO strength — the feel knob (exponent scale: ambient in a
/// pool of depth |h| is scaled by exp(−K·|h|); a typical −0.15 pool at K = 2
/// reads ~0.74, the deepest combined ~−0.5 reads ~0.37). Mirrored as a
/// literal in shade.hlsli (the clouds-wind idiom). The first linear K = 1.0
/// measured a 0.6/255 mean image delta at a sunlit rungholt plaza —
/// provably live and provably invisible; don't re-timid it without an A/B.
pub const DETAIL_AO_K: f32 = 3.0;

/// Cavity factor from the detail height `h` (the grain field's value − 1.0
/// plus the coarse AO octaves — mean 0 by construction, so the
/// "neighborhood mean" is a compile-time constant and this is a zero-lookup
/// AO term): `exp(K·min(h, 0))` — pits darken, peaks return EXACTLY 1.0
/// (exp(±0.0) == 1.0, and the call sites branch on `h < 0`, so the peak
/// identity is structural twice over). Exponential rather than linear:
/// strictly positive at ANY strength (no clamp to maintain as K or the
/// field's range moves), and compounding depth is how occlusion composes.
/// Multiplies AMBIENT and DIRECT SPECULAR, AFTER the PrimarySurface
/// captures — the FSR signals stay un-cavitied and the deterministic delta
/// rides the exact-remainder residual (un-denoised, texel-crisp).
/// Deliberately NOT energy-neutral: it is occlusion. Auto-fades with the
/// field's own dlod windows (h → 0 as they close).
#[inline(always)]
pub fn detail_cavity(h: f32) -> f32 {
    (DETAIL_AO_K * h.min(0.0)).exp()
}

/// The marched shadow's height scale: TEXELS of geometric height per unit of
/// field value — the terrain's steepness for shadow-length purposes. It is
/// INCIDENCE-ADAPTIVE (round 6): `HT_eff = lerp(LO, HI, saturate(n·l))`.
/// HI is the ARTISTIC override (at the coherent ~1.2 the sun ray cleared
/// HMAX in a fraction of a texel for any sun above ~15°, so shadows existed
/// only at grazing — a shadow term that vanishes whenever the scene is lit
/// reads as absent); it applies in full exactly where it is needed — steep
/// incidence (noon tops), where the natural response is weakest. At grazing
/// incidence (noon SIDES, sunset tops) the march is already maximal at the
/// physical steepness — rise ∝ n·l is tiny — so the override fades to LO
/// (near-coherent) there instead of multiplying 5× onto an already-strong
/// response; the overdone-sides feel-test is what forced the fade. rise =
/// ndl/(|lt|·(LO + (HI−LO)·ndl)) stays strictly increasing in ndl
/// (d/dndl = LO/(…)² > 0), so "higher sun never darker" survives.
/// Mirrored in shade.hlsli.
pub const DETAIL_SHADOW_HT_LO: f32 = 1.5;
pub const DETAIL_SHADOW_HT_HI: f32 = 6.0;

/// Shadow contact hardness: `exp(−K·penetration)`. K → ∞ converges to a
/// binary hit test (which aliases at 1 spp — the relief feature affords
/// binary only because its shadows ride real visibility); the soft-contact
/// form doubles as the 2°-sun penumbra. Mirrored in shade.hlsli.
pub const DETAIL_SHADOW_K: f32 = 5.0;

/// March tap distances, texels: LINEAR near-field (continuous coverage — no
/// gap a thin ridge slips through at contact-shadow scale) then geometric
/// far-field (distant occluders are penumbra-soft under a 2° sun, so sparse
/// taps + the soft max model that for free). Mirrored in shade.hlsli.
pub const DETAIL_SHADOW_D: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 14.0, 20.0];

// The march's conservative HMAX bound (once the sun ray climbs past it no
// tap can occlude — the clouds interval-skip lesson: high-sun pixels exit
// after 2-3 taps) is COMPUTED inside `detail_sun_shadow` since the strength
// knobs: DETAIL_AMP·kd + 0.5·kao + 0.35·kao, left-assoc so the default
// reproduces the retired `DETAIL_AMP + 0.5 + 0.35` constant chain bitwise.

/// The shadow FIELD: the height the march tests against — grain octave 0
/// (1-texel-equivalent cells, salt 40) + both pool octaves (8/4-texel,
/// salts 43/44), each under its EXISTING dlod window, value-only
/// (`clouds::vnoise3` — never the grad path, the march wants no gradient).
/// The same terrain the surface shades with minus the sub-texel grain
/// octaves 1-2 (speckle-scale, irrelevant to shadows at ≥ 1-texel tap
/// distances). `q3` in the world-space q-units of `detail_field`.
/// Term-for-term HLSL twin.
fn detail_shadow_h(q3: Vec3A, dlod: f32) -> f32 {
    let mut hh = 0.0f32;
    // The shadow field is the SUM of both strength families: grain octave 0
    // rides --detail-strength, the pools --detail-ao-strength — the same
    // terrain the surface shades with stays the terrain that shadows it.
    let w0 = (-dlod).clamp(0.0, 1.0);
    if w0 > 0.0 {
        let v = crate::clouds::vnoise3(q3, 40);
        hh += DETAIL_AMP * crate::scene::detail_strength() * w0 * (2.0 * v - 1.0);
    }
    let kao = crate::scene::detail_ao_strength();
    for (div, salt, amp, lg) in [(8.0f32, 43u32, 0.5f32, 3.0f32), (4.0, 44, 0.35, 2.0)] {
        let amp = amp * kao;
        let wk = (lg - dlod).clamp(0.0, 1.0);
        if wk > 0.0 {
            let v = crate::clouds::vnoise3(q3 / div, salt);
            hh += amp * wk * (2.0 * v - 1.0);
        }
    }
    hh
}

/// REAL horizon-marched sun shadow — a closed-form occlusion trace of the
/// detail heightfield toward the sun. From the hit's q-space position the
/// sun ray is walked along the sun's tangent-plane direction, and upstream
/// terrain that rises above it occludes: `occ = max_i(h(q3 + d_i·l̂t) −
/// (h0 + d_i·rise))`, `shadow = exp(−K·max(occ, 0))`. Everything is in
/// texel-EQUIVALENT q-units (one q-unit = one texel's worth of world travel,
/// by the `s` scaling — see `detail_field`): `lt` = the sun's tangent-plane
/// projection `l − n(n·l)`, unnormalized, so `|lt|` is the same grazing
/// measure the old (t, b)-frame 2-vector had; `rise` = (n·l)/(|lt|·HT)
/// field-units per q-unit (HT maps field value to texel-equivalents of
/// height, so a low sun rises slowly ⇒ long shadows, a high sun exits on
/// the HMAX bound after 2-3 taps). Exact 1.0 bitwise when nothing occludes
/// or every window is closed; a sub-horizon sun (ndl ≤ 0, the face is
/// geometrically self-shadowed) and the exact-zenith azimuth degeneracy
/// return 1.0. Zero rng, pure hit function — every same-seed/replay/VisCtl
/// contract holds. Shading-only: no visibility contact, silhouettes stay
/// flat (deliberately NOT the --heightfield relief feature). Term-for-term
/// HLSL twin.
pub fn detail_sun_shadow(q3: Vec3A, dlod: f32, lt: Vec3A, ndl: f32) -> f32 {
    let ltl = lt.length();
    if ltl < 1e-4 || ndl <= 0.0 {
        return 1.0;
    }
    let dir = lt / ltl;
    let ht = DETAIL_SHADOW_HT_LO + (DETAIL_SHADOW_HT_HI - DETAIL_SHADOW_HT_LO) * ndl.clamp(0.0, 1.0);
    let rise = ndl / (ltl * ht);
    let h0 = detail_shadow_h(q3, dlod);
    // The early-exit bound scales WITH the strength knobs (left-assoc so
    // kd = kao = 1 reproduces DETAIL_SHADOW_HMAX's constant chain bitwise)
    // — an unscaled bound would clip cranked-field shadows.
    let hmax = DETAIL_AMP * crate::scene::detail_strength()
        + 0.5 * crate::scene::detail_ao_strength()
        + 0.35 * crate::scene::detail_ao_strength();
    let mut occ = 0.0f32;
    for d in DETAIL_SHADOW_D {
        let ray_h = h0 + d * rise;
        if ray_h > hmax {
            break;
        }
        let o = detail_shadow_h(q3 + dir * d, dlod) - ray_h;
        occ = occ.max(o);
    }
    if occ > 0.0 { (-DETAIL_SHADOW_K * occ).exp() } else { 1.0 }
}

/// Ambient bump-response amplification (`--no-amb-bump`; the HL2 radiosity-
/// basis / bent-normal / SH-dominant-light class): irradiance is a cosine
/// convolution, so even the exact sky response to a bump tilt is a few
/// percent — structurally too smooth to show texel-scale relief under sky
/// light at ANY tilt. The SH linear band already carries the dome's bright
/// direction, so the cleanest member of that trick family is amplifying the
/// deviation response: `irr(n) + K·(irr(n_s) − irr(n))`, clamped ≥ 0. First
/// order and DIRECTIONAL — facets tilted toward the bright horizon/sun
/// azimuth brighten, away darken — for one extra 9-madd SH eval. Applies to
/// the FULL n_g→n_s deviation (normal maps, detail bump, ripple), so real
/// normal-mapped scenes gain daylight relief too. Deliberately NOT energy
/// conserving (an artistic exaggeration, the MOON_E_OVER_PI class).
/// `n_s == n` (flat-shaded geometry — the majority) and lever-off return
/// the old expression verbatim, checked in that order so the atomic load is
/// skipped on the majority. Zero rng. Mirrored in shade.hlsli
/// (`amb_irradiance`, gated on FLAG_AMB_BUMP).
pub const AMB_BUMP_K: f32 = 6.0;

/// Ambient response ceiling (round 6, the overdone-sides fix): the amplified
/// delta is capped at ±CAP of the base irradiance, by a SCALAR rescale so the
/// hue never shifts. The SH irradiance derivative is largest when n is
/// PERPENDICULAR to the dome's dominant direction — i.e. on the very sides
/// the K was not tuned for (tops sit near the dominant direction, where the
/// derivative is minimal) — so the ×K response is geometrically maximal
/// exactly where it overdrives. Under-cap pixels (the tops, measured ~10-30%
/// relative on the tod-15 plaza) return the uncapped formula BITWISE.
/// 0.5 → 0.25 in round 6b: at HIGH NOON a vertical block side gets ~no
/// direct sun (n·l ≈ 0 with the sun overhead), so its light is almost
/// entirely THIS ambient term — a ±50% cap on an ambient-dominated wall is
/// a ±50% swing of its TOTAL brightness, which is the up-close side
/// contrast the feel-test kept reporting; tops are direct-dominated at
/// noon, so the same swing is a small fraction of their total and they
/// keep their look either way.
/// Mirrored in shade.hlsli.
pub const AMB_BUMP_CAP: f32 = 0.25;

#[inline(always)]
pub fn amb_irradiance(sh: &crate::sh::Sh9, n: Vec3A, n_s: Vec3A) -> Vec3A {
    if n_s == n || !crate::scene::amb_bump() {
        return sh.irradiance(n_s);
    }
    let base = sh.irradiance(n);
    let mut d = (sh.irradiance(n_s) - base) * AMB_BUMP_K;
    let m = d.abs().max_element();
    let lim = AMB_BUMP_CAP * base.max_element();
    if m > lim {
        d *= lim / m;
    }
    (base + d).max(Vec3A::ZERO)
}

/// The AO octaves open while `dlod` is below this — the coarsest pool cell is
/// 8 texels, resolvable until the footprint reaches it (log2(8) = 3), which
/// is what extends the cavity to MID-DISTANCE surfaces where the fine grain
/// (dlod < 0) has long faded.
pub const DETAIL_AO_RANGE: f32 = 3.0;

/// Coarse height octaves: pools of occlusion AND relief rims at multi-texel
/// scale — a LOWER frequency than the grain, which is what makes the cavity
/// read as AO instead of "darker grain" (the same field would just deepen
/// the speckle). Two octaves at 8- and 4-texel-equivalent cells (salts 43/44
/// — the grain owns 40..42, ripple 16..19), each in its own resolvability
/// window `clamp(log2(cell) − dlod, 0, 1)`, amplitudes summing to 0.85.
/// Same world-space `q3` domain as `detail_field` (see the block comment
/// there). Returns (height, per-q-unit gradient) like `detail_field` — the
/// gradient feeds the micro-bump (× DETAIL_AO_BUMP_K), so the pools cast
/// directional relief out to mid-distance. Chain rule: an octave samples at
/// q/div, so its per-q-unit gradient carries 1/div (and the (2v − 1) the 2).
/// Zero rng, u32-exact via the shared `vnoise3_vg`; term-for-term HLSL twin.
pub fn detail_ao_field(q3: Vec3A, dlod: f32) -> (f32, Vec3A) {
    let mut hh = 0.0f32;
    let mut g = Vec3A::ZERO;
    // --detail-ao-strength: scales both pool amplitudes (height, rims via
    // the gradient, cavity input). ·1.0 bitwise-exact.
    let kao = crate::scene::detail_ao_strength();
    for (div, salt, amp, lg) in [(8.0f32, 43u32, 0.5f32, 3.0f32), (4.0, 44, 0.35, 2.0)] {
        let amp = amp * kao;
        let wk = (lg - dlod).clamp(0.0, 1.0);
        if wk > 0.0 {
            let (v, gv) = crate::clouds::vnoise3_vg(q3 / div, salt);
            hh += amp * wk * (2.0 * v - 1.0);
            g += gv * (amp * wk * 2.0 / div);
        }
    }
    (hh, g)
}

/// Pooled per-component variance of `clouds::vnoise3_vg`'s analytic gradient
/// over its stationary distribution — the one statistical constant the
/// spec-AA detail transfer needs (each octave's applied tilt is a linear
/// scale of this gradient, so its slope variance is that scale² × this).
/// MEASURED by deterministic lattice MC over the shipping function (600k
/// pooled components, mean 7e-5 ≈ 0 as symmetry demands) and baked as a
/// literal (the clouds-wind idiom, mirrored in shade.hlsli);
/// `spec_aa_self_test` re-measures it fresh within ±10%, so a noise retune
/// cannot silently skew the transfer.
pub const VNOISE_GRAD_VAR: f32 = 0.1104;

/// Spec-AA detail transfer: the per-axis slope variance of detail tilt NOT
/// currently applied because the octave windows have closed (`--no-spec-aa`
/// kills at the capture site). Per octave the applied tilt scales linearly
/// with its window `wk`, so applied variance goes as `wk²` and the discarded
/// share is `(1 − wk²)` of the octave's full-on variance:
///  - grain (the `detail_field` ladder): full-on applied slope factor is
///    `2·DETAIL_AMP·STR·DETAIL_BUMP_K` per octave — `amp_k·2·2^k` is
///    scale-invariant across k, the chain rule's gift — windows
///    `wk = clamp(−dlod − k, 0, 1)`;
///  - pools (`detail_ao_field`, gated on the AO lever like the bump share
///    itself): factor `amp_j·AO_STR·(2/div_j)·DETAIL_AO_BUMP_K·DETAIL_BUMP_K`,
///    windows `clamp(lg_j − dlod, 0, 1)`.
/// Every fully-open window contributes an IEEE-exact 0.0 (1 − 1·1), so
/// magnified pixels transfer nothing bit-identically; at dlod ≥
/// DETAIL_AO_RANGE the transfer plateaus at the field's whole variance. The
/// consumer weights the result by `detail_bump_weight²` — a surface whose
/// bump never applies (polished visor) is never frosted by the transfer, and
/// applied + transferred = `bw²·full` at EVERY distance, which is the
/// "detail always in the rendering equation" invariant. Pure function, zero
/// rng; term-for-term HLSL twin in shade.hlsli.
pub fn detail_var(dlod: f32) -> f32 {
    let mut sum = 0.0f32;
    // Grain octaves 0..2.
    let a = 2.0 * DETAIL_AMP * crate::scene::detail_strength() * DETAIL_BUMP_K;
    for k in 0..3u32 {
        let wk = (-dlod - k as f32).clamp(0.0, 1.0);
        sum += a * a * (1.0 - wk * wk);
    }
    // Pool octaves (8- and 4-texel cells — detail_ao_field's ladder).
    if crate::scene::detail_ao() {
        let kao = crate::scene::detail_ao_strength();
        for (div, amp, lg) in [(8.0f32, 0.5f32, 3.0f32), (4.0, 0.35, 2.0)] {
            let c = amp * kao * (2.0 / div) * DETAIL_AO_BUMP_K * DETAIL_BUMP_K;
            let wk = (lg - dlod).clamp(0.0, 1.0);
            sum += c * c * (1.0 - wk * wk);
        }
    }
    sum * VNOISE_GRAD_VAR
}

/// The spec-AA fold itself: `α′² = α² + 2σ²` with `α = roughness²`, in
/// roughness form — `(r⁴ + 2σ²)^¼`, clamped at 1. The Beckmann slope-variance
/// identity (α²/2 = per-axis slope variance) applied pragmatically to GGX,
/// the LEAN/Kaplanyan fold. Monotone, ≥ r always. `σ² == 0.0` is handled by
/// the CALLER's branch, never here: `sqrt(sqrt(r⁴))` re-rounds, so the
/// off-state identity is structural, not algebraic. Mirrored term-for-term
/// in shade.hlsli.
#[inline(always)]
pub fn spec_aa_fold(rough: f32, s2: f32) -> f32 {
    (rough * rough * rough * rough + 2.0 * s2).sqrt().sqrt().min(1.0)
}

/// Spec-AA math gates, run by `--check` (the detail/ripple gate class): the
/// fold's closed-form anchor and monotone bounds, the transfer's exact-zero
/// open-window identity (the bit-identity spine of the magnification off
/// state), its plateau against an independently assembled closed form, the
/// AO-lever share, and the VNOISE_GRAD_VAR re-measure (±10%) that keeps the
/// baked literal honest against a vnoise retune.
pub fn spec_aa_self_test() -> Result<(), String> {
    // Pin the strength knobs (the detail_self_test RAII pattern) so the
    // closed forms below are checked at KNOWN amplitudes even under a
    // --detail-strength session; restored on every exit path.
    struct Pin(f32, f32, bool);
    impl Drop for Pin {
        fn drop(&mut self) {
            crate::scene::set_detail_strength(self.0);
            crate::scene::set_detail_ao_strength(self.1);
            crate::scene::set_detail_ao(self.2);
        }
    }
    let _pin = Pin(
        crate::scene::detail_strength(),
        crate::scene::detail_ao_strength(),
        crate::scene::detail_ao(),
    );
    crate::scene::set_detail_strength(1.0);
    crate::scene::set_detail_ao_strength(1.0);
    crate::scene::set_detail_ao(true);

    // --- The fold ----------------------------------------------------------
    // Closed-form anchor: fold(0, σ²) = (2σ²)^¼.
    for s2 in [1e-4f32, 0.01, 0.1, 0.5] {
        let want = (2.0 * s2).powf(0.25);
        let got = spec_aa_fold(0.0, s2);
        if (got - want).abs() > 1e-6 * want.max(1.0) {
            return Err(format!("fold(0, {s2}) = {got}, want (2σ²)^¼ = {want}"));
        }
    }
    // Monotone in both arguments, bounded to [rough, 1].
    let rs = [0.0f32, 0.02, 0.1, 0.3, 0.5, 0.8, 1.0];
    let s2s = [0.0f32, 1e-4, 1e-2, 0.1, 0.5];
    for (i, &r) in rs.iter().enumerate() {
        for (j, &s2) in s2s.iter().enumerate() {
            let f = spec_aa_fold(r, s2);
            if !(f >= r - 1e-6 && f <= 1.0) {
                return Err(format!("fold({r}, {s2}) = {f} escapes [rough, 1]"));
            }
            if i > 0 && spec_aa_fold(rs[i - 1], s2) > f + 1e-7 {
                return Err(format!("fold not monotone in rough at ({r}, {s2})"));
            }
            if j > 0 && spec_aa_fold(r, s2s[j - 1]) > f + 1e-7 {
                return Err(format!("fold not monotone in σ² at ({r}, {s2})"));
            }
        }
    }

    // --- The transfer's window identities -----------------------------------
    // Every window fully open ⇒ IEEE-exact 0.0 (a magnified pixel transfers
    // nothing, bit-identically) — including the GPU's −1e30 degenerate lod
    // and the CPU's −∞.
    for dlod in [-3.0f32, -5.0, -100.0, -1e30, f32::NEG_INFINITY] {
        let v = detail_var(dlod);
        if v.to_bits() != 0.0f32.to_bits() {
            return Err(format!("detail_var({dlod}) = {v}, want bitwise 0.0"));
        }
    }
    // Plateau at dlod ≥ DETAIL_AO_RANGE: every window shut, the field's whole
    // variance transfers. The oracle is the octave table spelled out
    // independently, not the function's own loop.
    let a = 2.0 * DETAIL_AMP * DETAIL_BUMP_K; // strengths pinned at 1.0
    let pools = {
        let c8 = 0.5 * (2.0 / 8.0) * DETAIL_AO_BUMP_K * DETAIL_BUMP_K;
        let c4 = 0.35 * (2.0 / 4.0) * DETAIL_AO_BUMP_K * DETAIL_BUMP_K;
        c8 * c8 + c4 * c4
    };
    let want_plateau = (3.0 * a * a + pools) * VNOISE_GRAD_VAR;
    let got_plateau = detail_var(DETAIL_AO_RANGE);
    if (got_plateau - want_plateau).abs() > 1e-6 * want_plateau {
        return Err(format!("plateau {got_plateau} != closed form {want_plateau}"));
    }
    if detail_var(1e30) != got_plateau {
        return Err("plateau does not saturate past DETAIL_AO_RANGE".into());
    }
    // Anti-vacuity: the transfer must be live at the pinned strengths.
    if !(got_plateau > 0.0) {
        return Err("plateau is 0 — the transfer is vacuous".into());
    }
    // Monotone nondecreasing across the whole fade band (windows only close
    // as dlod grows, so transferred variance only accumulates).
    let mut prev = 0.0f32;
    let mut x = -3.5f32;
    while x <= 3.5 {
        let v = detail_var(x);
        if v + 1e-7 < prev {
            return Err(format!("detail_var not monotone at dlod {x}"));
        }
        prev = v;
        x += 0.05;
    }
    // The AO lever removes exactly the pool share (the transfer follows the
    // bump share it mirrors).
    crate::scene::set_detail_ao(false);
    let grain_only = detail_var(DETAIL_AO_RANGE);
    crate::scene::set_detail_ao(true);
    let want_grain = 3.0 * a * a * VNOISE_GRAD_VAR;
    if (grain_only - want_grain).abs() > 1e-6 * want_grain {
        return Err(format!("AO-off plateau {grain_only} != grain-only {want_grain}"));
    }

    // --- VNOISE_GRAD_VAR re-measure ----------------------------------------
    // The same deterministic lattice that baked the literal (600k pooled
    // gradient components across ~124k cells); ±10% so a vnoise retune
    // cannot silently skew the transfer's magnitude.
    let (mut s, mut s2m, mut n) = (0.0f64, 0.0f64, 0u64);
    for i in 0..200_000u32 {
        let t = i as f32;
        let q = Vec3A::new(t * 0.618_034 + 0.123, t * 0.414_214 + 4.567, t * 0.267_949 + 9.876);
        let (_, g) = crate::clouds::vnoise3_vg(q, 40);
        for c in [g.x, g.y, g.z] {
            s += c as f64;
            s2m += (c as f64) * (c as f64);
            n += 1;
        }
    }
    let mean = s / n as f64;
    let var = (s2m / n as f64 - mean * mean) as f32;
    if (var - VNOISE_GRAD_VAR).abs() > 0.1 * VNOISE_GRAD_VAR {
        return Err(format!(
            "vnoise gradient variance measured {var:.4} vs baked {VNOISE_GRAD_VAR} (±10%) — \
             re-bake the literal (and its shade.hlsli twin)"
        ));
    }
    Ok(())
}

/// The detail window's lod base for an ANISOTROPIC footprint: the log2 length
/// of the footprint's MINOR axis, in normalized UV (unit-consistent with
/// `tri_lod_base`, which is the log2 MAJOR axis — the `tangent_self_test`
/// reduction gate pins that identity on conformal maps). The window must key
/// off what the sampler actually leaves unresolved, and `SampleGrad`/
/// `sample_aniso` resolve down to the short axis — keying off the isotropic
/// (major-axis) lod closed the window on grazing-viewed faces whose albedo
/// was still texel-sharp. Deliberately UNCAPPED by MaxAnisotropy: past the
/// cap the window opens ~log2(ratio/max) further than the sampler resolves,
/// which at the default 16 against tri_grads' 0.05 |n·d| floor is at most
/// 0.32 lod of amplitude-`DETAIL_AMP` grain at near-silhouette grazing —
/// accepted, and what keeps the HLSL twin free of an injected max constant.
/// Mirrored by `shade.hlsli::detail_aniso_base`.
pub fn detail_aniso_base(gu: glam::Vec2, gv: glam::Vec2) -> f32 {
    gu.length().min(gv.length()).max(1e-20).log2()
}

/// The UNTEXTURED materials' detail-fade window: the cone footprint in the
/// field's own texel-equivalents, `log2(cone_w / s)` — exactly 0 at
/// `cone_w == s`, and `s == 0` parks the window CLOSED (+∞ fails both the
/// `< 0` grain test and the `< DETAIL_AO_RANGE` band — the bitwise
/// pre-untextured-arm off). `cone_w` is the footprint's MINOR axis, matching
/// the textured aniso convention (`detail_aniso_base`). The ONE source for
/// the shade() call site AND the self-test's D2 anchors, so the gate pins
/// the shipping expression rather than a local twin. Mirrored as a literal
/// in `shade.hlsli` (1e30 in place of ∞ — a dead value either way, the
/// consumer re-guards on `s > 0`).
pub fn detail_untex_window(cone_w: f32, s: f32) -> f32 {
    if s > 0.0 { (cone_w / s).log2() } else { f32::INFINITY }
}

/// (albedo factor, per-q-unit gradient of the factor's sum term). See the
/// block comment above. Degenerate-lod note: a degenerate base lod is −∞ on
/// the CPU and −1e30 on the GPU — both saturate every window to 1
/// identically, and the output stays bounded at whatever q3 came in.
pub fn detail_field(q3: Vec3A, dlod: f32) -> (f32, Vec3A) {
    let mut q = q3;
    // --detail-strength: scales the whole amplitude ladder (the gradient —
    // and so the micro-bump — scales with it, linear in amp). ·1.0 is
    // bitwise-exact, so the default is the unscaled field with no branch.
    let (mut amp, mut scl) = (DETAIL_AMP * crate::scene::detail_strength(), 1.0f32);
    let mut f = 1.0f32;
    let mut g = Vec3A::ZERO;
    for k in 0..3u32 {
        let wk = (-dlod - k as f32).clamp(0.0, 1.0);
        // A real branch, mirrored in HLSL — it also skips the noise eval.
        if wk > 0.0 {
            let (v, gv) = crate::clouds::vnoise3_vg(q, 40 + k);
            f += amp * wk * (2.0 * v - 1.0);
            // Chain rule: octave k samples at q·2^k, so its per-q-unit
            // gradient carries the 2^k (`scl`), and the (2v − 1) the 2.
            g += gv * (amp * wk * 2.0 * scl);
        }
        q *= 2.0;
        amp *= 0.5;
        scl *= 2.0;
    }
    (f.max(0.05), g)
}

/// Tilt `base` (the shading normal) by the detail field's 3D gradient's
/// TANGENTIAL PROJECTION `gt = g − n(n·g)`. Under the old UV domain this was
/// `t·g.x + b·g.y` on the Gram-Schmidt (t, b) frame — an orthonormal basis
/// of the tangent plane, so the projection is the same operation with the
/// frame construction deleted (the winding sign cancels in b⊗b). A zero
/// result or a below-horizon tilt falls back to `base` (coarser, never
/// wrong — the ripple_normal shape); `gt == 0` returns `base` verbatim, the
/// structural off state in-function (it subsumes BOTH old guards: g == 0 and
/// the degenerate-tangent bail — a normal-parallel gradient has no tangent
/// component to apply).
fn detail_bump(base: Vec3A, n: Vec3A, g: Vec3A) -> Vec3A {
    let gt = g - n * n.dot(g);
    if gt == Vec3A::ZERO {
        return base;
    }
    let out = (base - gt * DETAIL_BUMP_K).normalize_or_zero();
    if out == Vec3A::ZERO || out.dot(n) <= 0.0 { base } else { out }
}

/// Detail-texture math gates, run by `--check` (the depth-tint/ripple gate
/// class): off anchors bit-exact, the single-octave window endpoint, fade
/// continuity, bounds, energy neutrality, integrability (returned gradient
/// vs central difference — the mechanized pin no image gate can see), the
/// bump guards + sign pin, the anti-tiling teeth (the world-space domain's
/// whole reason to exist), determinism.
pub fn detail_self_test() -> Result<(), String> {
    // Every gate below assumes the DEFAULT amplitudes, so the strength knobs
    // are pinned to 1.0 for the duration and restored on EVERY exit path (an
    // RAII guard — the amb-lever save/restore pattern, generalized: this fn
    // has too many early returns for manual restores). A
    // `--detail-strength 2 --check` therefore still proves the math.
    struct StrengthPin(f32, f32, f32);
    impl Drop for StrengthPin {
        fn drop(&mut self) {
            crate::scene::set_detail_strength(self.0);
            crate::scene::set_detail_ao_strength(self.1);
            crate::scene::set_detail_untex_scale(self.2);
        }
    }
    let _pin = StrengthPin(
        crate::scene::detail_strength(),
        crate::scene::detail_ao_strength(),
        crate::scene::detail_untex_scale(),
    );
    crate::scene::set_detail_strength(1.0);
    crate::scene::set_detail_ao_strength(1.0);
    crate::scene::set_detail_untex_scale(1.0);
    // A fixed non-lattice-aligned 3D q-space anchor (the old uv0 × 256 texel
    // point, given a z).
    let q0 = Vec3A::new(81.152, 189.696, 7.317);
    // (1) Off anchors, bit-exact: dlod >= 0 saturates every window to 0 —
    // factor bits == 1.0, gradient exactly zero. The call site's `dlod < 0`
    // guard therefore has a continuous in-function partner.
    for &dl in &[0.0f32, 3.0] {
        let (f, g) = detail_field(q0, dl);
        if f.to_bits() != 1.0f32.to_bits() || g != Vec3A::ZERO {
            return Err(format!("dlod {dl} must be exactly inert, got ({f}, {g:?})"));
        }
    }
    // (2) Window endpoint at dlod = −1: octave 0 fully in (w0 == 1), octaves
    // 1..2 exactly absent — the factor bit-equals a one-octave re-derivation.
    {
        let (f, _) = detail_field(q0, -1.0);
        let (v, _) = crate::clouds::vnoise3_vg(q0, 40);
        let expect = (1.0f32 + DETAIL_AMP * (2.0 * v - 1.0)).max(0.05);
        if f.to_bits() != expect.to_bits() {
            return Err(format!("dlod −1 factor {f} != single-octave {expect}"));
        }
    }
    // (3) Continuity at the fade edge — no pop crossing dlod = 0.
    {
        let (f, _) = detail_field(q0, -1e-4);
        if (f - 1.0).abs() > 1e-3 {
            return Err(format!("fade edge pops: factor {f} at dlod −1e-4"));
        }
    }
    // (4)+(5) Bounds/finiteness over a q3 × dlod sweep on a non-lattice-
    // aligned 3D slab (z drifts per sample so the sweep is genuinely 3D),
    // and energy: the mean factor at full depth must sit at 1 (a drift is an
    // exposure shift).
    {
        let (nx, ny) = (97u32, 89u32);
        let mut sum = 0.0f64;
        for i in 0..nx {
            for j in 0..ny {
                let q = Vec3A::new(
                    (i as f32 + 0.5) * 256.0 / nx as f32,
                    (j as f32 + 0.5) * 256.0 / ny as f32,
                    ((i * 13 + j * 7) % 32) as f32 * 0.37,
                );
                for &dl in &[-0.5f32, -1.5, -2.5, -4.0, -16.0] {
                    let (f, g) = detail_field(q, dl);
                    if !(0.6..=1.4).contains(&f) || !f.is_finite() || !g.is_finite() {
                        return Err(format!(
                            "factor/gradient out of bounds: ({f}, {g:?}) at {q:?} dlod {dl}"
                        ));
                    }
                    if g.length() > 2.5 {
                        return Err(format!("gradient {g:?} exceeds the design bound at {q:?}"));
                    }
                    if dl == -4.0 {
                        sum += f as f64;
                    }
                }
            }
        }
        let mean = sum / (nx as f64 * ny as f64);
        if (mean - 1.0).abs() > 0.01 {
            return Err(format!("detail factor mean {mean} drifts from 1 — exposure shift"));
        }
    }
    // (6) Integrability: the returned gradient must be the true q-space
    // derivative of the factor (dlod −4: every window saturated, so the sum
    // is the only q-dependence and the 0.05 floor is provably inactive).
    // Probe points sit with ALL THREE coords ≡ 0.125 (mod 0.25) q-units —
    // the INTERIOR of every octave's lattice cell (≥ 0.125 q-units from the
    // nearest boundary in all three octaves): value noise is only C¹ across
    // a cell boundary, so a central difference straddling one carries O(h)
    // error instead of O(h²) and would measure the probe, not the gradient.
    {
        let h = 0.02f32; // q-units — [q−h·2^k, q+h·2^k] stays in-cell everywhere
        let mut worst = 0.0f32;
        for i in 0..16u32 {
            let q = Vec3A::new(
                ((7 + i * 11) % 250) as f32 + 0.125,
                ((5 + i * 17) % 250) as f32 + 0.125,
                ((3 + i * 23) % 250) as f32 + 0.125,
            );
            let px = |d: Vec3A| detail_field(q + d, -4.0).0;
            let g_fd = Vec3A::new(
                (px(Vec3A::new(h, 0.0, 0.0)) - px(Vec3A::new(-h, 0.0, 0.0))) / (2.0 * h),
                (px(Vec3A::new(0.0, h, 0.0)) - px(Vec3A::new(0.0, -h, 0.0))) / (2.0 * h),
                (px(Vec3A::new(0.0, 0.0, h)) - px(Vec3A::new(0.0, 0.0, -h))) / (2.0 * h),
            );
            let (_, g) = detail_field(q, -4.0);
            worst = worst.max((g - g_fd).abs().max_element());
        }
        if worst > 5e-3 {
            return Err(format!(
                "detail gradient is not the factor's derivative (worst |Δ| {worst})"
            ));
        }
    }
    // (7) Bump guards: zero gradient ⇒ base verbatim; unit length + horizon
    // over a sweep; a NORMAL-PARALLEL gradient ⇒ base verbatim (the
    // tangential projection is zero — this pin replaces the retired
    // degenerate-tangent gate, whose frame no longer exists); and the sign
    // pin — positive g.x under n = +Y tilts the normal AGAINST +x, so a
    // silent sign flip fails loudly.
    {
        let n = Vec3A::Y;
        let base = Vec3A::new(0.1, 0.98, -0.05).normalize();
        let z = detail_bump(base, n, Vec3A::ZERO);
        if z.to_array().map(f32::to_bits) != base.to_array().map(f32::to_bits) {
            return Err("zero gradient must return the base verbatim".into());
        }
        for i in 0..16 {
            let g = Vec3A::new(i as f32 * 0.15 - 1.0, 0.3, 0.8 - i as f32 * 0.11);
            let out = detail_bump(n, n, g);
            if (out.length() - 1.0).abs() > 1e-5 || out.dot(n) <= 0.0 {
                return Err(format!("bump normal invalid: {out:?} for g {g:?}"));
            }
        }
        let d = detail_bump(base, n, Vec3A::Y * 0.7);
        if d.to_array().map(f32::to_bits) != base.to_array().map(f32::to_bits) {
            return Err("normal-parallel gradient must return the base verbatim".into());
        }
        let s = detail_bump(n, n, Vec3A::new(0.5, 0.0, 0.0));
        if s.x >= 0.0 {
            return Err(format!("sign pin: +g.x must tilt against +x, got {s:?}"));
        }
    }
    // (8) Determinism.
    {
        let a = detail_field(q0 * 1.31, -2.7);
        let b = detail_field(q0 * 1.31, -2.7);
        if a.0.to_bits() != b.0.to_bits() || a.1 != b.1 {
            return Err("detail_field is not deterministic".into());
        }
    }
    // (9) The bump's roughness window (the frosted-visor guard): exact 0
    // at/below LO — which the g == 0 guard turns into a verbatim base normal
    // — exact 1 at/above HI, monotone between.
    {
        for &(r, want) in &[
            (0.0f32, 0.0f32),
            (DETAIL_ROUGH_LO, 0.0),
            (DETAIL_ROUGH_HI, 1.0),
            (1.0, 1.0),
        ] {
            let w = detail_bump_weight(r);
            if w.to_bits() != want.to_bits() {
                return Err(format!("bump weight at rough {r} must be exactly {want}, got {w}"));
            }
        }
        let mut prev = -1.0f32;
        for i in 0..=20 {
            let w = detail_bump_weight(i as f32 * 0.05);
            if w < prev {
                return Err("bump weight must be monotone in roughness".into());
            }
            prev = w;
        }
        // The composition: a smooth pixel's bump must be a verbatim no-op
        // through the g·bw == 0 path.
        let n = Vec3A::Y;
        let g = Vec3A::new(0.7, 0.0, -0.4) * detail_bump_weight(DETAIL_ROUGH_LO);
        let out = detail_bump(n, n, g);
        if out.to_array().map(f32::to_bits) != n.to_array().map(f32::to_bits) {
            return Err("smooth-pixel bump must be a verbatim no-op".into());
        }
    }
    // The aniso window base keys off the MINOR axis (the Minecraft-tops
    // finding: the isotropic base's view-tilt term closed the window on
    // grazing-viewed faces whose albedo the aniso sampler kept sharp).
    {
        let iso = glam::Vec2::new(0.01, 0.0);
        let v = glam::Vec2::new(0.0, 0.01);
        let conformal = detail_aniso_base(iso, v);
        if (conformal - 0.01f32.log2()).abs() > 1e-6 {
            return Err(format!(
                "conformal aniso base {conformal} != log2(0.01) — unit drift vs tri_lod_base"
            ));
        }
        // Stretching the MAJOR axis alone (the grazing-view case) must not
        // move the window — invariance is the fix, and the teeth: the old
        // isotropic (major-axis) base moves by log2(16) here.
        let stretched = detail_aniso_base(glam::Vec2::new(0.16, 0.0), v);
        if stretched.to_bits() != conformal.to_bits() {
            return Err("major-axis stretch must not move the detail window".into());
        }
        let major = 0.16f32.log2();
        if (major - stretched).abs() < 3.9 {
            return Err("teeth: the major-axis base must differ by ~log2(16)".into());
        }
        // gu/gv symmetry, and the degenerate floor stays finite.
        if detail_aniso_base(v, glam::Vec2::new(0.16, 0.0)).to_bits() != stretched.to_bits() {
            return Err("aniso base must be symmetric in gu/gv".into());
        }
        if !detail_aniso_base(glam::Vec2::ZERO, glam::Vec2::ZERO).is_finite() {
            return Err("degenerate footprint must stay finite".into());
        }
    }
    // Detail cavity AO (detail_cavity): the pits-only occlusion factor.
    {
        // Off anchor + peaks clamp, EXACT: 1.0 + K·0.0 is a +0.0 add ⇒
        // bitwise 1.0, and the call sites' `h < 0` branch makes peaks
        // structural — but the function itself must agree.
        for h in [0.0f32, 1e-6, 0.1, 0.315] {
            if detail_cavity(h).to_bits() != 1.0f32.to_bits() {
                return Err(format!("cavity({h}) must be exactly 1.0"));
            }
        }
        // Pits darken, strictly monotone in depth.
        let mut prev = 1.0f32;
        for h in [-0.05f32, -0.1, -0.2, -0.315] {
            let c = detail_cavity(h);
            if c >= prev {
                return Err(format!("cavity must strictly decrease into pits ({h} -> {c})"));
            }
            prev = c;
        }
        // Strict positivity at any depth — the exp form's whole point (a
        // future K raise must never need a clamp audit).
        for h in [-0.315f32, -0.95, -3.0] {
            if !(detail_cavity(h) > 0.0) {
                return Err(format!("cavity({h}) must stay strictly positive"));
            }
        }
        // Continuity at 0⁻ (no pop crossing the mean): |exp(K·h) − 1| ≈ K·|h|.
        if (detail_cavity(-1e-6) - 1.0).abs() > (DETAIL_AO_K + 1.0) * 1e-6 {
            return Err("cavity must be continuous at h = 0".into());
        }
        // TEETH, field -> cavity end-to-end: a real pit of the field must
        // darken, and the SAME q3 with the window closed must be exactly
        // 1.0. The scan failing to find a pit is itself a failure (the
        // field would have lost its variance).
        let mut pit: Option<Vec3A> = None;
        'scan: for iy in 0..64 {
            for ix in 0..64 {
                let q = Vec3A::new(
                    ix as f32 * 4.0 + 0.7,
                    iy as f32 * 4.0 + 0.3,
                    ((ix * 3 + iy * 5) % 17) as f32 * 0.9,
                );
                let (f, _) = detail_field(q, -3.0);
                if f < 0.95 {
                    pit = Some(q);
                    break 'scan;
                }
            }
        }
        let Some(q) = pit else {
            return Err("no pit (f < 0.95) found at dlod = -3 — the field lost its variance".into());
        };
        let (f, _) = detail_field(q, -3.0);
        if detail_cavity(f - 1.0) >= 1.0 {
            return Err("a real field pit must produce cavity < 1".into());
        }
        let (f0, _) = detail_field(q, 0.0);
        if detail_cavity(f0 - 1.0).to_bits() != 1.0f32.to_bits() {
            return Err("a closed window must produce cavity exactly 1.0".into());
        }
        // Determinism: bit-equal across two evals.
        if detail_cavity(f - 1.0).to_bits() != detail_cavity(f - 1.0).to_bits() {
            return Err("cavity must be deterministic".into());
        }
    }
    // Horizon-marched sun shadow (detail_sun_shadow) — the real trace.
    {
        let qa = Vec3A::new(80.128, 173.312, 4.913);
        // Closed windows (dlod >= 3): the shadow field is identically zero,
        // occ never exceeds 0, factor bitwise 1.0 at any sun.
        for dlod in [DETAIL_AO_RANGE, 6.0] {
            let s = detail_sun_shadow(qa, dlod, Vec3A::new(0.6, 0.0, 0.2), 0.5);
            if s.to_bits() != 1.0f32.to_bits() {
                return Err(format!("closed-window march must be exactly 1.0, got {s}"));
            }
        }
        // Degeneracies: exact-zenith azimuth (|lt| < 1e-4) and a sub-horizon
        // sun both return bitwise 1.0.
        if detail_sun_shadow(qa, -1.0, Vec3A::ZERO, 1.0).to_bits() != 1.0f32.to_bits()
            || detail_sun_shadow(qa, -1.0, Vec3A::new(0.5, 0.0, 0.0), -0.1).to_bits()
                != 1.0f32.to_bits()
        {
            return Err("zenith/sub-horizon march must be exactly 1.0".into());
        }
        // OCCLUDER TEETH with directionality: scan for a point where a LOW
        // grazing sun from +x is occluded (< 1) while the OPPOSITE azimuth
        // at the same point is not darker than it — and require the shadow
        // to exist at all (anti-vacuity: a dead march passes every anchor
        // above while shadowing nothing).
        let ndl = 0.08f32; // low sun: rise ≈ 0.067 field-units/q-unit
        let lt = Vec3A::new(0.99, 0.0, 0.0);
        let mut found = None;
        'scan: for iy in 0..48 {
            for ix in 0..48 {
                let p = Vec3A::new(
                    ix as f32 * 5.34 + 2.3,
                    iy as f32 * 5.34 + 1.1,
                    ((ix * 7 + iy * 11) % 23) as f32 * 0.7,
                );
                let s = detail_sun_shadow(p, -1.0, lt, ndl);
                if s < 0.85 {
                    found = Some((p, s));
                    break 'scan;
                }
            }
        }
        let Some((p, s)) = found else {
            return Err("no marched shadow (< 0.85) found at a grazing sun — the march is dead".into());
        };
        if !(s > 0.0) {
            return Err("marched shadow must stay strictly positive".into());
        }
        // Directionality: the same point lit from the opposite azimuth sees
        // different upstream terrain — the factors must differ (a direction-
        // blind march would be the retired statistical term in disguise).
        let s_opp = detail_sun_shadow(p, -1.0, -lt, ndl);
        if s_opp.to_bits() == s.to_bits() {
            return Err("march must be directional (opposite azimuths bit-equal)".into());
        }
        // THIRD-AXIS LIVENESS (world-space domain): the march must occlude
        // along z exactly as along x — a port that silently dropped an axis
        // (or a field flat in z) passes the x-scan while shadowing nothing
        // on half the walls.
        {
            let ltz = Vec3A::new(0.0, 0.0, 0.99);
            let mut found_z = false;
            'zscan: for iy in 0..48 {
                for ix in 0..48 {
                    let p = Vec3A::new(
                        ix as f32 * 5.34 + 2.3,
                        ((ix * 7 + iy * 11) % 23) as f32 * 0.7,
                        iy as f32 * 5.34 + 1.1,
                    );
                    if detail_sun_shadow(p, -1.0, ltz, ndl) < 0.85 {
                        found_z = true;
                        break 'zscan;
                    }
                }
            }
            if !found_z {
                return Err("no marched shadow along +z — the third axis is dead".into());
            }
        }
        // Shadows lengthen as the sun drops: at the shadowed point, a higher
        // sun never darkens more, and somewhere the low sun is strictly
        // darker than the high one.
        let s_hi = detail_sun_shadow(p, -1.0, lt, 0.9);
        if s_hi < s {
            return Err("a higher sun must not deepen the marched shadow".into());
        }
        if !(s < s_hi) && s_hi.to_bits() == s.to_bits() {
            return Err("teeth: a low sun must shadow strictly more somewhere".into());
        }
        // Adaptive-HT monotonicity (round 6): the incidence fade must keep
        // "higher sun never darker" over the WHOLE ndl range — rise =
        // ndl/(|lt|·(LO + (HI−LO)·ndl)) is strictly increasing in ndl
        // (d/dndl = LO/(…)² > 0), and this sweep is the behavioral pin.
        let mut prev = s;
        for k in 1..=18 {
            let nd = 0.08 + 0.045 * k as f32;
            let sk = detail_sun_shadow(p, -1.0, lt, nd);
            if sk < prev {
                return Err(format!(
                    "adaptive HT broke shadow monotonicity at ndl {nd}: {sk} < {prev}"
                ));
            }
            prev = sk;
        }
        // Endpoint pins on the fade itself: the lerp must land the artistic
        // HI at steep incidence and the near-coherent LO at grazing.
        let ht_at = |nd: f32| {
            DETAIL_SHADOW_HT_LO + (DETAIL_SHADOW_HT_HI - DETAIL_SHADOW_HT_LO) * nd.clamp(0.0, 1.0)
        };
        if ht_at(0.0).to_bits() != DETAIL_SHADOW_HT_LO.to_bits()
            || ht_at(1.0).to_bits() != DETAIL_SHADOW_HT_HI.to_bits()
            || !(DETAIL_SHADOW_HT_LO < DETAIL_SHADOW_HT_HI)
        {
            return Err("adaptive HT endpoints must be exactly LO/HI with LO < HI".into());
        }
        // Determinism.
        if detail_sun_shadow(p, -1.0, lt, ndl).to_bits() != s.to_bits() {
            return Err("march must be deterministic".into());
        }
    }
    // Direct N·L contrast cap (detail_ndl_cap) — the overdone-sides ceiling.
    {
        let c = DETAIL_NDL_CAP;
        // Under-cap identity: a raw inside the bounds returns BITWISE.
        let raw = 0.55f32;
        if detail_ndl_cap(raw, 0.5).to_bits() != raw.to_bits() {
            return Err("under-cap ndl must return raw bitwise".into());
        }
        // Over-cap lands exactly on the bounds.
        if detail_ndl_cap(2.0, 0.5).to_bits() != (0.5 * (1.0 + c)).to_bits()
            || detail_ndl_cap(-0.5, 0.5).to_bits() != (0.5 * (1.0 - c)).to_bits()
        {
            return Err("over-cap ndl must land exactly on p·(1±CAP)".into());
        }
        // Terminator hygiene: a pre-detail-unlit facet may not be lit by
        // detail (bright-speckle kill), while a darker tilt passes through.
        if detail_ndl_cap(0.3, 0.0).to_bits() != 0.0f32.to_bits()
            || detail_ndl_cap(0.3, -0.2).to_bits() != 0.0f32.to_bits()
            || detail_ndl_cap(-0.2, -0.1).to_bits() != (-0.2f32).to_bits()
        {
            return Err("p <= 0 must force ndl <= 0".into());
        }
        // Continuity at p = 0: both arms collapse toward 0 (no pop crossing
        // the pre-detail terminator).
        let eps_p = 1e-6f32;
        if detail_ndl_cap(0.5, eps_p) > eps_p * (1.0 + c) {
            return Err("ndl cap must be continuous at p = 0".into());
        }
    }
    // Ambient bump response (amb_irradiance). The fn reads the live lever,
    // so the gate PINS it on for the amplification teeth and restores after
    // (the wide-tiles pattern) — a `--no-amb-bump --check` run must still
    // prove the math, then re-verify the off arm below.
    {
        let lever = crate::scene::amb_bump();
        crate::scene::set_amb_bump(true);
        // A synthetic SH with a known linear band: bright toward +x. Project
        // a directional-ish sky by hand — coefficient layout per sh.rs
        // (band 1 = Y_1{-1,0,1} ∝ y, z, x).
        let mut sh = crate::sh::Sh9::ZERO;
        sh.c[0] = Vec3A::splat(1.0);
        sh.c[3] = Vec3A::splat(0.4); // the +x linear band
        let n = Vec3A::Y;
        // Identity: n_s == n must be the plain irradiance bitwise (the
        // structural off arm — flat-shaded geometry).
        let plain = sh.irradiance(n);
        if amb_irradiance(&sh, n, n)
            .to_array()
            .map(f32::to_bits)
            != plain.to_array().map(f32::to_bits)
        {
            return Err("amb_irradiance must be plain irradiance at n_s == n".into());
        }
        // Sign + amplification teeth: a tilt toward the bright +x must
        // brighten, away must darken, and the amplified delta must exceed
        // the raw delta (K > 1 anti-vacuity).
        let toward = Vec3A::new(0.3, 0.954, 0.0).normalize();
        let away = Vec3A::new(-0.3, 0.954, 0.0).normalize();
        let a_t = amb_irradiance(&sh, n, toward);
        let a_a = amb_irradiance(&sh, n, away);
        if !(a_t.x > plain.x) || !(a_a.x < plain.x) {
            return Err("amb bump: toward-bright must brighten, away must darken".into());
        }
        let raw = sh.irradiance(toward);
        if !((a_t.x - plain.x) > (raw.x - plain.x) * 1.5) {
            return Err("amb bump: the amplified delta must exceed the raw delta".into());
        }
        // Response ceiling (round 6): where the raw ×K delta exceeds
        // ±AMB_BUMP_CAP of the base, the output delta's max channel must sit
        // ON the ceiling (scalar rescale) with the hue untouched — probed on
        // a CHROMATIC SH so the ratio teeth aren't trivially satisfied.
        {
            let mut shc = crate::sh::Sh9::ZERO;
            shc.c[0] = Vec3A::new(1.0, 0.8, 0.6);
            shc.c[3] = Vec3A::new(0.5, 0.4, 0.3);
            let plainc = shc.irradiance(n);
            let d_raw = (shc.irradiance(toward) - plainc) * AMB_BUMP_K;
            let lim = AMB_BUMP_CAP * plainc.max_element();
            if !(d_raw.abs().max_element() > lim) {
                return Err(
                    "amb cap teeth are vacuous — the synthetic SH no longer exceeds the cap"
                        .into(),
                );
            }
            let d_out = amb_irradiance(&shc, n, toward) - plainc;
            let m_out = d_out.abs().max_element();
            if !((m_out - lim).abs() <= 1e-5 * lim) {
                return Err(format!(
                    "amb cap: capped delta max {m_out} must land on the ceiling {lim}"
                ));
            }
            // Hue preservation: the rescale is scalar, so channel ratios of
            // the output delta match the raw delta's.
            let s0 = d_out.x / d_raw.x;
            if !((d_out.y / d_raw.y - s0).abs() <= 1e-5) || !((d_out.z / d_raw.z - s0).abs() <= 1e-5)
            {
                return Err("amb cap must rescale scalar (hue-preserving)".into());
            }
            // Under-cap bitwise identity: a small tilt whose ×K delta stays
            // under the ceiling must reproduce the UNCAPPED formula exactly.
            let mild = Vec3A::new(0.02, 0.9998, 0.0).normalize();
            let d_mild = (shc.irradiance(mild) - plainc) * AMB_BUMP_K;
            if !(d_mild.abs().max_element() < lim) {
                return Err("amb cap under-cap probe is not under the cap".into());
            }
            let expect = (plainc + d_mild).max(Vec3A::ZERO);
            if amb_irradiance(&shc, n, mild).to_array().map(f32::to_bits)
                != expect.to_array().map(f32::to_bits)
            {
                return Err("under-cap amb response must be the uncapped formula bitwise".into());
            }
        }
        // Clamp floor: a hostile tilt against a strong band cannot go
        // negative.
        let mut hostile = crate::sh::Sh9::ZERO;
        hostile.c[0] = Vec3A::splat(0.1);
        hostile.c[3] = Vec3A::splat(2.0);
        let neg = amb_irradiance(&hostile, n, away);
        if neg.min_element() < 0.0 {
            return Err("amb bump must clamp at zero".into());
        }
        // Determinism.
        if amb_irradiance(&sh, n, toward).to_array().map(f32::to_bits)
            != a_t.to_array().map(f32::to_bits)
        {
            return Err("amb_irradiance must be deterministic".into());
        }
        // Lever-off arm: the plain irradiance verbatim at a perturbed n_s.
        crate::scene::set_amb_bump(false);
        if amb_irradiance(&sh, n, toward).to_array().map(f32::to_bits)
            != sh.irradiance(toward).to_array().map(f32::to_bits)
        {
            crate::scene::set_amb_bump(lever);
            return Err("lever-off amb_irradiance must be plain irradiance".into());
        }
        crate::scene::set_amb_bump(lever);
    }
    // Coarse AO/relief octaves (detail_ao_field).
    {
        let qa = Vec3A::new(80.128, 173.312, 4.913);
        // Structural off past the range: both windows exactly closed —
        // height bitwise 0.0 AND gradient exactly zero (the rim bump's own
        // off arm).
        for dlod in [DETAIL_AO_RANGE, 4.0, 10.0] {
            let (v, g) = detail_ao_field(qa, dlod);
            if v.to_bits() != 0.0f32.to_bits() || g != Vec3A::ZERO {
                return Err(format!("ao field must be exactly inert at dlod = {dlod}"));
            }
        }
        // Continuity at the window edge (no pop entering range).
        if detail_ao_field(qa, DETAIL_AO_RANGE - 1e-4).0.abs() > 1e-4 {
            return Err("ao field must fade in continuously at the range edge".into());
        }
        // Bounds + anti-vacuity over a grid at full window: |h| <= 0.85, and
        // the field must actually vary (a dead field would pass every gate
        // above while darkening nothing).
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for iy in 0..48 {
            for ix in 0..48 {
                let p = Vec3A::new(
                    ix as f32 * 5.34 + 2.3,
                    iy as f32 * 5.34 + 1.1,
                    ((ix * 5 + iy * 3) % 29) as f32 * 0.8,
                );
                let (v, g) = detail_ao_field(p, -1.0);
                if !v.is_finite() || v.abs() > 0.85 || !g.is_finite() {
                    return Err(format!("ao field out of bounds at {p:?}: ({v}, {g:?})"));
                }
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if lo > -0.05 || hi < 0.05 {
            return Err(format!("ao field lost its variance (range {lo}..{hi})"));
        }
        // Integrability: the returned gradient must be the true q-space
        // derivative of the height. Probes sit at cell-INTERIOR points of
        // BOTH pool lattices in ALL THREE axes (8- and 4-unit cells:
        // p ≡ 2 (mod 4) is ≥ 1 unit from every 4-boundary and ≥ 2 from
        // every 8-boundary) — value noise is only C¹ across a cell boundary,
        // so a straddling central difference measures the probe, not the
        // gradient (the detail_field lesson).
        {
            // q-units — stays in-cell for both lattices, and small enough
            // that the O(h²) truncation sits well under the 5e-3 bound (at
            // 0.25 the pools' third derivative alone measured 0.014 — the
            // difference was measuring the probe, not the gradient).
            let hh = 0.05f32;
            let mut worst = 0.0f32;
            for i in 0..12u32 {
                let p = Vec3A::new(
                    ((6 + i * 16) % 248) as f32 + 2.0,
                    ((10 + i * 24) % 248) as f32 + 2.0,
                    ((14 + i * 40) % 248) as f32 + 2.0,
                );
                let f = |d: Vec3A| detail_ao_field(p + d, -1.0).0;
                let g_fd = Vec3A::new(
                    (f(Vec3A::new(hh, 0.0, 0.0)) - f(Vec3A::new(-hh, 0.0, 0.0))) / (2.0 * hh),
                    (f(Vec3A::new(0.0, hh, 0.0)) - f(Vec3A::new(0.0, -hh, 0.0))) / (2.0 * hh),
                    (f(Vec3A::new(0.0, 0.0, hh)) - f(Vec3A::new(0.0, 0.0, -hh))) / (2.0 * hh),
                );
                let (_, g) = detail_ao_field(p, -1.0);
                worst = worst.max((g - g_fd).abs().max_element());
            }
            if worst > 5e-3 {
                return Err(format!(
                    "ao gradient is not the height's derivative (worst |Δ| {worst})"
                ));
            }
        }
        // Determinism.
        {
            let a = detail_ao_field(qa, -1.0);
            let b = detail_ao_field(qa, -1.0);
            if a.0.to_bits() != b.0.to_bits() || a.1 != b.1 {
                return Err("ao field must be deterministic".into());
            }
        }
    }
    // ANTI-TILING TEETH — the world-space domain's reason to exist. 16
    // q-units = one 16-texel Minecraft block advance, the exact offset that
    // aliased onto the same UV rect (hence the same field values) under the
    // old UV-texel domain. Every axis must decorrelate, for the grain, the
    // AO pools, and the shadow field alike.
    {
        for axis in [Vec3A::X, Vec3A::Y, Vec3A::Z] {
            let q1 = q0 + axis * 16.0;
            if detail_field(q0, -4.0).0.to_bits() == detail_field(q1, -4.0).0.to_bits() {
                return Err(format!("grain repeats one block along {axis:?} — the domain tiles"));
            }
            if detail_ao_field(q0, -1.0).0.to_bits() == detail_ao_field(q1, -1.0).0.to_bits() {
                return Err(format!("AO pools repeat one block along {axis:?} — the domain tiles"));
            }
            if detail_shadow_h(q0, -1.0).to_bits() == detail_shadow_h(q1, -1.0).to_bits() {
                return Err(format!("shadow field repeats one block along {axis:?} — the domain tiles"));
            }
        }
    }
    // STRENGTH KNOBS (--detail-strength / --detail-ao-strength): amplitude
    // scaling must be EXACT where fp lets it be — 0 is the structural
    // off-by-amplitude, and ×2 is a power of two, so the deviation doubles
    // BITWISE (real teeth, not a tolerance). The hmax pin: at a cranked AO
    // field a low-sun march must still find shadow — an UNSCALED early-exit
    // bound would clip exactly the shadows the knob asked for.
    {
        let (f1, _) = detail_field(q0, -4.0);
        crate::scene::set_detail_strength(0.0);
        let (f0, g0) = detail_field(q0, -4.0);
        if f0.to_bits() != 1.0f32.to_bits() || g0 != Vec3A::ZERO {
            crate::scene::set_detail_strength(1.0);
            return Err("detail-strength 0 must be exactly inert".into());
        }
        crate::scene::set_detail_strength(2.0);
        let (f2, _) = detail_field(q0, -4.0);
        crate::scene::set_detail_strength(1.0);
        if (f2 - 1.0).to_bits() != (2.0 * (f1 - 1.0)).to_bits() {
            return Err(format!(
                "detail-strength 2 must double the deviation exactly ({} vs {})",
                f2 - 1.0,
                2.0 * (f1 - 1.0)
            ));
        }
        let (h1, gr1) = detail_ao_field(q0, -1.0);
        crate::scene::set_detail_ao_strength(2.0);
        let (h2, gr2) = detail_ao_field(q0, -1.0);
        crate::scene::set_detail_ao_strength(1.0);
        if h2.to_bits() != (2.0 * h1).to_bits() || gr2 != gr1 * 2.0 {
            return Err("detail-ao-strength 2 must double the pools exactly".into());
        }
        // hmax coherence: kao = 4 grows the field 4× — the march must still
        // find a sub-0.85 shadow at a grazing sun somewhere on the scan grid
        // (a stale unscaled bound breaks out before the taller terrain can
        // occlude).
        crate::scene::set_detail_ao_strength(4.0);
        let mut found4 = false;
        'k4: for iy in 0..24 {
            for ix in 0..24 {
                let p = Vec3A::new(
                    ix as f32 * 10.7 + 2.3,
                    iy as f32 * 10.7 + 1.1,
                    ((ix * 7 + iy * 11) % 23) as f32 * 0.7,
                );
                if detail_sun_shadow(p, -1.0, Vec3A::new(0.99, 0.0, 0.0), 0.08) < 0.85 {
                    found4 = true;
                    break 'k4;
                }
            }
        }
        crate::scene::set_detail_ao_strength(1.0);
        if !found4 {
            return Err("kao = 4 march found no shadow — the hmax bound did not scale".into());
        }
    }
    // PER-MATERIAL SCALE DERIVATION (Scene::detail_scales — the greedy-mesh
    // anti-seam fix): two coplanar quads of ONE material, one mapped 1:1
    // (16 texels over 1 world unit) and one stretched 4× (vokselia's merged
    // strips in miniature), must derive a SINGLE nonzero scale equal to the
    // hand-computed median of the per-triangle texel sizes — a per-face
    // value is exactly what seamed the field at every face boundary. A
    // degenerate-UV scene must derive 0.0 (the structural off).
    {
        use crate::scene::{finalize_scalars, MatKind, Material, Scene};
        use crate::texture::Texture;
        let mk = |texcoords: Vec<glam::Vec2>, kind: MatKind| -> Scene {
            let mut sc = Scene {
                positions: vec![
                    Vec3A::new(0.0, 0.0, 0.0),
                    Vec3A::new(1.0, 0.0, 0.0),
                    Vec3A::new(1.0, 0.0, 1.0),
                    Vec3A::new(0.0, 0.0, 1.0),
                    Vec3A::new(1.0, 0.0, 0.0),
                    Vec3A::new(5.0, 0.0, 0.0),
                    Vec3A::new(5.0, 0.0, 4.0),
                    Vec3A::new(1.0, 0.0, 4.0),
                ],
                normals: vec![Vec3A::Y; 8],
                texcoords,
                indices: vec![[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
                tri_mat: vec![0; 4],
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
                    normal_tex: crate::scene::NO_TEX,
                    normal_scale: 1.0,
                    height_amp: 0.0,
                    rough_tex: crate::scene::NO_TEX,
                    metal_tex: crate::scene::NO_TEX,
                    emissive_tex: crate::scene::NO_TEX,
                    class: crate::matclass::IDX_DEFAULT as u8,
                    kind,
                }],
                textures: vec![Texture {
                    w: 16,
                    h: 16,
                    texels: vec![[128u8, 128, 128, 255]; 256],
                    alpha_masked: false,
                    srgb: true,
                    source: String::new(),
                    h2n: false,
                    n2h: false,
                    normal_role: false,
                    mips: Vec::new(),
                    var_mips: Vec::new(),
                }],
                any_alpha: false,
                any_height: false,
                any_transmissive: false,
                emissive: crate::emissive::EmissiveLights::off(),
                sun: crate::sky::Sun::new(Vec3A::Y),
                sky_sh: crate::sh::Sh9::ZERO,
                sky_scale: 1.0,
                night: 0.0,
                light_gain: 1.0,
                light_canon: crate::scene::LightCanon::default(),
                sway: None,
                sway_regions: Vec::new(),
                diag: 1.0,
                eps: 1e-4,
                ao_radius: 0.03,
                detail_scales: Vec::new(),
                content_min: Vec3A::ZERO,
                content_max: Vec3A::ZERO,
                tex_var: Vec::new(),
            };
            finalize_scalars(&mut sc);
            sc
        };
        let unit = vec![
            glam::Vec2::new(0.0, 0.0),
            glam::Vec2::new(1.0, 0.0),
            glam::Vec2::new(1.0, 1.0),
            glam::Vec2::new(0.0, 1.0),
            glam::Vec2::new(0.0, 0.0),
            glam::Vec2::new(1.0, 0.0),
            glam::Vec2::new(1.0, 1.0),
            glam::Vec2::new(0.0, 1.0),
        ];
        let sc = mk(unit.clone(), MatKind::Textured { tex: 0 });
        // Per-tri texel sizes: quad A twice 1/16, quad B twice 4/16; the
        // upper median (v[len/2]) of the sorted four is 0.25.
        let s = sc.detail_scales.first().copied().unwrap_or(0.0);
        if !((s - 0.25).abs() < 1e-6) {
            return Err(format!(
                "detail_scales derivation: expected the 0.25 median, got {s}"
            ));
        }
        if sc.detail_scales.len() != 1 {
            return Err("detail_scales must be parallel to materials".into());
        }
        let sc0 = mk(vec![glam::Vec2::ZERO; 8], MatKind::Textured { tex: 0 });
        if sc0.detail_scales.first().copied().unwrap_or(1.0).to_bits() != 0.0f32.to_bits() {
            return Err("degenerate-UV material must derive detail_scale 0.0".into());
        }
        // The UNTEXTURED arm (the powerplant case): a material with NO
        // albedo map must derive the SYNTHETIC content-diag-relative scale,
        // bitwise the derivation's own expression — keyed on KIND, which is
        // why the degenerate-UV *Textured* pin above stays 0.0 untouched.
        let scd = mk(unit.clone(), MatKind::Diffuse);
        let exp = crate::scene::DETAIL_UNTEX_K
            * 1.0
            * (scd.content_max - scd.content_min).length();
        let sd = scd.detail_scales.first().copied().unwrap_or(0.0);
        if sd.to_bits() != exp.to_bits() || !(sd > 0.0) {
            return Err(format!(
                "untextured material must derive the synthetic scale {exp}, got {sd}"
            ));
        }
        // Knob 0 = the bitwise off arm: untextured materials stay 0.0 (the
        // pre-untextured-arm renderer — `--detail-untex-scale 0`).
        crate::scene::set_detail_untex_scale(0.0);
        let sck = mk(unit, MatKind::Diffuse);
        crate::scene::set_detail_untex_scale(1.0);
        if sck.detail_scales.first().copied().unwrap_or(1.0).to_bits() != 0.0f32.to_bits() {
            return Err("--detail-untex-scale 0 must keep untextured detail_scale 0.0".into());
        }
    }
    // (D2) The untextured lod window — `detail_untex_window`, the SHIPPING
    // function shade()'s untextured match arm calls (not a local twin, so
    // call-site drift is caught). Anchors are exact powers of two (fp-exact,
    // teeth not tolerance): cone_w == s sits exactly ON the window edge
    // (0.0 bitwise — grain off, AO band open), halving/doubling moves dlod
    // by exactly ∓1, and the s == 0 arm parks the window CLOSED (INFINITY
    // fails both `< 0` and `< DETAIL_AO_RANGE`).
    {
        let w = detail_untex_window;
        if w(0.037, 0.037).to_bits() != 0.0f32.to_bits() {
            return Err("untex window: cone_w == s must be exactly 0.0".into());
        }
        if w(0.5, 0.25) != 1.0 || w(0.125, 0.25) != -1.0 {
            return Err("untex window: power-of-two anchors must be exact".into());
        }
        let closed = w(1e-6, 0.0);
        if closed < 0.0 || closed < DETAIL_AO_RANGE {
            return Err("untex window: s == 0 must close the window".into());
        }
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
