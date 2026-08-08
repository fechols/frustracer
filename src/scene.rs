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

/// Water-class switch (`--no-water`), same knob-before-load family. Off = the
/// classifier never refines the glass tier into water (`classify` is handed
/// `water_hint = false, tf = None`), so the fountain classifies glassware
/// exactly as before this feature. Keys the scene cache's lever word:
/// classification is baked into the sidecar, so a lever A/B must never serve a
/// stale cache.
static WATER: AtomicBool = AtomicBool::new(true);

pub fn set_water(on: bool) {
    WATER.store(on, Relaxed);
}

pub fn water_enabled() -> bool {
    WATER.load(Relaxed)
}

/// Coincident-cull switch (`--no-coincident-cull`), same knob-before-load
/// family. Off = `cull_coincident` is a no-op — transmissive faces exactly
/// coincident with opaque faces survive (the pre-cull behavior, where the
/// CPU and GPU intersectors break the z-fight tie DIFFERENTLY and the
/// transmission chain can eps-tunnel past the coincident opaque surface).
/// Keys the scene cache's lever word: the pass rewrites cached indices.
static COINCIDENT_CULL: AtomicBool = AtomicBool::new(true);

pub fn set_coincident_cull(on: bool) {
    COINCIDENT_CULL.store(on, Relaxed);
}

pub fn coincident_cull_enabled() -> bool {
    COINCIDENT_CULL.load(Relaxed)
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

/// Detail-texture switch (`--no-detail-tex`): Unreal-1 style procedural
/// close-up detail — albedo grain + micro-bump on magnified textured hits
/// (`shade::detail_field`). Runtime shading lever like depth-tint (no scene
/// data changes — reads live in shade.rs / the GPU FLAG_DETAIL bit; no
/// cache-lever-word bit, no CACHE_VERSION move).
static DETAIL_TEX: AtomicBool = AtomicBool::new(true);

pub fn set_detail_tex(on: bool) {
    DETAIL_TEX.store(on, Relaxed);
}

pub fn detail_tex() -> bool {
    DETAIL_TEX.load(Relaxed)
}

/// Detail cavity AO switch (`--no-detail-ao`): pits of the detail field's own
/// height (its value has mean 1.0 by construction, so value − 1 IS signed
/// depth-below-neighborhood) darken the AMBIENT + SPECULAR terms — the
/// texel-scale sky-visibility contrast a flat sun-facing surface cannot get
/// from normal perturbation (SH ambient is order-2 smooth, N·L sits at the
/// cosine max). Runtime shading lever like detail-tex above (no scene data
/// changes — reads live in shade.rs / the GPU FLAG_DETAIL_AO bit; no
/// cache-lever-word bit). A no-op wherever the detail field itself never
/// fires (lever off, dlod >= 0, untextured/transmissive materials).
static DETAIL_AO: AtomicBool = AtomicBool::new(true);

pub fn set_detail_ao(on: bool) {
    DETAIL_AO.store(on, Relaxed);
}

pub fn detail_ao() -> bool {
    DETAIL_AO.load(Relaxed)
}

/// `--detail-strength K` (default 0.5 — the 2026-08-06 feel-test
/// calibration; 1.0 spells the original full-strength field, and ×1.0 is
/// the bit-identical arm): session multiplier on the detail GRAIN family's
/// amplitudes — `shade::detail_field`'s octave ladder and the grain term of
/// the shadow field. The micro-bump scales with it for free (it consumes
/// the field's gradient, linear in amplitude). Runtime shading lever, the
/// detail_tex class (no cache contact); the GPU twin is the injected
/// DETAIL_STR define (`trace::detail_defs` — kernels compile at session
/// start, restart tier).
static DETAIL_STRENGTH: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0x3f00_0000); // 0.5f32

pub fn set_detail_strength(k: f32) {
    DETAIL_STRENGTH.store(k.to_bits(), Relaxed);
}

pub fn detail_strength() -> f32 {
    f32::from_bits(DETAIL_STRENGTH.load(Relaxed))
}

/// Spec-AA switch (`--no-spec-aa`): the Toksvig/LEAN slope-variance →
/// roughness fold, keeping detail maps in the rendering equation at every
/// distance — what a mip averages away comes back as a wider GGX lobe
/// instead of vanishing. Gates (1) the variance-companion pass in
/// `finalize_normal_mips` (off ⇒ no companions ⇒ the map arm structurally
/// dead on both CPU and GPU), (2) the detail-field transfer capture in
/// shade.rs, and (3) the GPU FLAG_SPEC_AA bit. Runtime-lever class like
/// detail_tex (derived data only — no cache-lever-word bit, no
/// CACHE_VERSION move); the off arm is bit-identical to the pre-feature
/// renderer by construction. `--no-slope-mips`/`--no-mips` kill the map
/// half automatically (no `normal_role` ⇒ no variance planes); the detail
/// half is independent.
static SPEC_AA: AtomicBool = AtomicBool::new(true);

pub fn set_spec_aa(on: bool) {
    SPEC_AA.store(on, Relaxed);
}

pub fn spec_aa() -> bool {
    SPEC_AA.load(Relaxed)
}

/// `--detail-ao-strength K` (default 0.125 — the same feel-test; 1.0 = the
/// original amplitudes): session multiplier on the detail AO family's
/// amplitudes — the pool octaves (height + relief rims + cavity input) and
/// their share of the marched shadow field, whose HMAX early-exit bound
/// scales in lockstep (an unscaled bound would clip K > 1 shadows). Same
/// lever class as `detail_strength` above.
static DETAIL_AO_STRENGTH: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0x3e00_0000); // 0.125f32

pub fn set_detail_ao_strength(k: f32) {
    DETAIL_AO_STRENGTH.store(k.to_bits(), Relaxed);
}

pub fn detail_ao_strength() -> f32 {
    f32::from_bits(DETAIL_AO_STRENGTH.load(Relaxed))
}

/// `--detail-untex-scale K` (default 1.0): multiplier on the SYNTHETIC
/// texel-equivalent scale UNTEXTURED materials get in
/// `derive_detail_scales` (`DETAIL_UNTEX_K` × content diag) — the knob that
/// sizes the detail grain on albedo-map-free scenes (powerplant). 0 keeps
/// those materials at `detail_scale == 0.0`, the pre-untextured-arm
/// renderer BITWISE (the A/B off arm; `--no-detail-tex` stays the
/// whole-feature kill). Read at DERIVATION time (load), so restart tier;
/// no GPU define — the scale rides the per-material `GpuMat.detail_scale`
/// lane, and `detail_scales` is derived-never-serialized (no cache
/// contact).
static DETAIL_UNTEX_SCALE: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0x3f80_0000); // 1.0f32

pub fn set_detail_untex_scale(k: f32) {
    DETAIL_UNTEX_SCALE.store(k.to_bits(), Relaxed);
}

pub fn detail_untex_scale() -> f32 {
    f32::from_bits(DETAIL_UNTEX_SCALE.load(Relaxed))
}

/// Ambient bump response switch (`--no-amb-bump`): the sampled/SH ambient
/// tiers amplify the irradiance response to the shading normal's deviation
/// from the geometric normal (`shade::amb_irradiance` — the HL2/bent-normal
/// dominant-direction class), so normal maps and the detail micro-bump read
/// under SKY light, whose order-2 cosine-convolved irradiance is otherwise
/// too smooth to show texel-scale relief at any tilt. Runtime shading lever
/// like detail-tex above (reads live in shade.rs / the GPU FLAG_AMB_BUMP
/// bit; no cache contact). A no-op wherever n_s == n_g (flat-shaded
/// geometry) — the structural off state.
static AMB_BUMP: AtomicBool = AtomicBool::new(true);

pub fn set_amb_bump(on: bool) {
    AMB_BUMP.store(on, Relaxed);
}

pub fn amb_bump() -> bool {
    AMB_BUMP.load(Relaxed)
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
    /// Absorption/transmission tint, the SINGLE source for the per-interface
    /// glass tint, the Beer–Lambert depth attenuation, and `shadow_tint`.
    /// Sentinel `< 0` (default `splat(-1.0)`) = "use albedo" — the structural
    /// bit-identity guarantee: `trans_tint_or(albedo)` returns albedo VERBATIM
    /// for every non-water material. Water (`matclass::WATER`) sets a light
    /// blue-green so the Beer–Lambert exponent does the depth work (red
    /// extinguishes fastest); the loader does NOT white-lift a water albedo,
    /// so the tint lives here instead.
    pub trans_tint: Vec3A,
    /// Index of refraction for the transmission chain's Snell/Fresnel math.
    /// Default 1.5 (== the old global `GLASS_IOR`; `1.0/1.5f32` is bit-identical
    /// to the old const, so existing glass is unchanged). Water is 1.33.
    pub ior: f32,
    /// Procedural water-ripple slope amplitude (0.0 = no ripples — the
    /// structural off state, the `height_amp` pattern). Perturbs the SHADING
    /// normal (and, guarded, the Snell axis) with a pure-function wave field on
    /// the shared cloud clock (`shade::ripple_normal`); zero rng draws.
    pub ripple_amp: f32,
    /// Emitted radiance (Ke / glTF emissiveFactor). Added to color at every
    /// shading depth, OUTSIDE the kd·(1−transmission) factor — the DISPLAY
    /// half. Emitters CAN also light other surfaces: `--emissive-lights`
    /// arms the direct tier's clustered-light NEE (src/emissive.rs, default
    /// OFF), and under fb.gi the hemi gather delivers the transport exactly
    /// instead. Default ZERO.
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
    /// The matclass classify verdict (index into `matclass::NAMES`),
    /// `matclass::IDX_DEFAULT` everywhere the classifier does not run
    /// (procedural builders, glTF — the structural off state). Retained
    /// because the classify inputs (texture stem, MATERIAL NAME) do not
    /// survive the load: the Minecraft scenes carry their foliage signal
    /// only on the `newmtl` name (one shared atlas texture), so
    /// `foliage::leaf_materials` reads this byte instead of re-deriving from
    /// `Texture::source`. Shading never reads it — the Pbr fields already
    /// carry the class's consequences.
    pub class: u8,
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
    /// Kd was already lifted toward white at load, so this composes. Water
    /// carries its color in `trans_tint` instead of a lifted albedo, so the
    /// tint source is `trans_tint_or(albedo)` — bit-identical (`albedo`
    /// verbatim) for every material with the sentinel tint.
    #[inline]
    pub fn shadow_tint(&self) -> Vec3A {
        self.trans_tint_or(self.albedo) * self.transmission
    }

    /// The transmission/absorption tint: `trans_tint` when set (`.x >= 0`),
    /// else `albedo` returned VERBATIM. A sign test (not NaN) so the GPU
    /// mirror is a plain `>= 0.0` select. This is the ONE tint source shared
    /// by the per-interface glass tint, Beer–Lambert, and `shadow_tint`.
    #[inline]
    pub fn trans_tint_or(&self, albedo: Vec3A) -> Vec3A {
        if self.trans_tint.x >= 0.0 {
            self.trans_tint
        } else {
            albedo
        }
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
    /// Clustered emissive virtual lights (src/emissive.rs — the direct-tier
    /// NEE set derived from Ke/map_Ke/glTF-emissive triangles). Derived by
    /// `finalize_scalars`, NEVER serialized (the `sky_sh` precedent: both
    /// cache load paths and the world merge re-run finalize, so warm loads
    /// re-derive and CACHE_VERSION does not move). `count == 0` is the
    /// structural off state every emissive-free scene keeps.
    pub emissive: crate::emissive::EmissiveLights,
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
    /// Wind-swayed foliage (src/foliage.rs): the ONE cell partition every
    /// consumer shares — the CPU intersector's per-triangle displacement,
    /// the BVH build sweep, and the GPU BLAS split — plus the per-frame
    /// offsets main.rs bakes. Derived at load under `foliage::sweep_armed()`
    /// (`foliage::attach`), NEVER serialized (the sky_sh precedent); `None`
    /// is the structural off-state every headless path keeps.
    pub sway: Option<Box<crate::foliage::SceneSway>>,
    /// Foliage SWAY REGIONS (foliage v0.6): disjoint ascending tri ranges,
    /// each carrying ITS OWN content box, so `foliage::attach` derives every
    /// sway length (contact tolerance, cell pitch, ramp band, curl
    /// wavelength, amplitude) at the REGION's scale instead of the scene's.
    /// Empty = one implicit whole-scene region (every non-world scene —
    /// bit-identical to the pre-region partition). Set ONLY by
    /// `world::merge_scenes` (one region per island, the part's content box
    /// at its ring offset) and serialized ONLY in the WORLD sidecar: without
    /// this the world's ~10× content diagonal fused whole Minecraft forests
    /// into single plants whose chords span the forest — a trunk then sat on
    /// a nearly flat slice of the ramp and translated rigidly (the
    /// "world trees aren't grounded" report).
    pub sway_regions: Vec<crate::foliage::SwayRegion>,
    /// Bounding diagonal — the scale reference for all epsilons.
    pub diag: f32,
    /// Self-intersection offset for secondary rays.
    pub eps: f32,
    pub ao_radius: f32,
    /// Per-MATERIAL detail-field texel scale (world units per noise cell at
    /// octave 0), parallel to `materials` — the world-space detail domain's
    /// `s` in `q3 = p_rest / s`. A PER-MATERIAL sampled MEDIAN of the
    /// tri_uv_basis texel-size formula, never a per-face value: greedy-meshed
    /// atlas exporters make per-face texel density wildly non-uniform
    /// (vokselia's Grass spans s 0.11..215 across merged runs), and a
    /// per-face s makes `q3` jump at every face boundary — the block-seam
    /// artifact. One s per material keeps the field continuous across every
    /// face of a surface BY CONSTRUCTION. Materials with NO albedo map take
    /// a SYNTHETIC scale instead (`DETAIL_UNTEX_K` × content diag ×
    /// `--detail-untex-scale` — the field needs no UVs, only a
    /// texel-equivalent size, so powerplant-class scenes grain too). 0.0 =
    /// a Textured material with no valid basis anywhere, or the knob's off
    /// arm (the detail field's structural off). Derived
    /// (`finalize_scalars`), never serialized (the sky_sh precedent — every
    /// cache/merge path re-runs finalize).
    pub detail_scales: Vec<f32>,
    /// CONTENT bounds: the geometry EXCLUDING the standard ground quad (the
    /// first `GROUND_VERTS` positions every loader pushes) — where the models
    /// actually are. `diag` is ground-quad-dominated on the procedural/stress
    /// scenes (a ±60 ground makes it ~17× the content scale), so anything
    /// that should live AMONG the content (the fireflies' placement box)
    /// anchors here instead. Falls back to the full AABB when the skip would
    /// be degenerate. Derived (`finalize_scalars`), never serialized.
    pub content_min: Vec3A,
    pub content_max: Vec3A,
    /// Spec-AA companion map: texture id → the id of its slope-VARIANCE
    /// companion texture (`NO_TEX` = none), parallel to `textures`. Filled by
    /// `finalize_normal_mips` — companions are appended at the END of the
    /// table, so no existing id shifts (the cache-v7 argument) — and read by
    /// shade's roughness fold. Derived, never serialized (the pass runs
    /// post-cache-store on every load path); EMPTY on any scene that never
    /// ran it, so lookups must go through `.get()`.
    pub tex_var: Vec<u32>,
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
            trans_tint: Vec3A::splat(-1.0),
            ior: 1.5,
            ripple_amp: 0.0,
            emissive: Vec3A::ZERO,
            normal_tex: NO_TEX,
            normal_scale: 1.0,
            height_amp: 0.0,
            rough_tex: NO_TEX,
            metal_tex: NO_TEX,
            emissive_tex: NO_TEX,
            class: crate::matclass::IDX_DEFAULT as u8,
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
            emissive: crate::emissive::EmissiveLights::off(),
            sun,
            sky_sh: crate::sh::Sh9::ZERO,
            sky_scale: 1.0,
            night: 0.0,
            sway: None,
            sway_regions: Vec::new(),
            diag: 0.0,
            eps: 0.0,
            ao_radius: 0.0,
            detail_scales: Vec::new(),
            content_min: Vec3A::ZERO,
            content_max: Vec3A::ZERO,
            tex_var: Vec::new(),
        };
        finalize_scalars(&mut scene);
        scene
    }
}

/// Solve-width cap for the n2h heightfield derivation: one 4K Frankot–
/// Chellappa solve holds ~0.5 GB of complex scratch, so the full rayon width
/// would spike memory on 4K-heavy scenes.
const N2H_POOL_MAX: usize = 8;

/// The n2h solver's ONE shared, bounded pool. The cap has to be PROCESS-WIDE,
/// not per call: the world loader runs several islands' loads at once
/// (`world.rs`), and a fresh pool per `derive_heights` would multiply the
/// memory bound by the number of in-flight parts — 5 x 8 concurrent 4K solves
/// is exactly what the cap exists to prevent. `install` from any number of
/// callers queues into this one pool, so the ceiling is `N2H_POOL_MAX` solves
/// in flight no matter how many scenes are loading.
///
/// No wait cycle exists: `apply_n2h` -> `n2h_solve` -> `build_mips` is
/// straight-line sequential and never submits back into the global pool, so a
/// global worker blocked in `install` here is only parked, never deadlocked.
/// Lazy — a session that derives no heights builds no pool; after first use
/// `N2H_POOL_MAX` idle threads live for the process, which is the only
/// observable difference on the single-scene path.
fn n2h_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let workers = rayon::current_num_threads().min(N2H_POOL_MAX);
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .ok()
    })
    .as_ref()
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
    crate::progress::phase(crate::progress::Phase::Heights, "", solve.len() as u32);
    let t0 = std::time::Instant::now();
    let amps: std::collections::HashMap<u32, f32> = {
        use rayon::prelude::*;
        // Bounded pool — see `n2h_pool`: the cap is shared process-wide so
        // concurrent island loads can't multiply it.
        let mut run = || {
            scene
                .textures
                .par_iter_mut()
                .enumerate()
                .filter(|(i, _)| solve.contains(&(*i as u32)))
                .map(|(i, t)| {
                    let r = (i as u32, t.apply_n2h());
                    crate::progress::tick();
                    r
                })
                .collect()
        };
        match n2h_pool() {
            Some(pool) => pool.install(run),
            None => run(),
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

/// Mark every texture consumed as a NORMAL MAP and rebuild its mip chain
/// with the slope-space filter (`Texture::normal_role` — the "normal maps
/// flatten with distance" fix; `--no-slope-mips` kills). Runs ONCE from
/// `load_scene`'s post-match site (the `foliage::attach` slot), which is the
/// single point cold OBJ, warm sidecar, glTF, world-merged and tiled loads
/// all pass through — the role is defined by MATERIAL WIRING, identical in
/// every one of those, so warm and cold chains cannot diverge (loader-side
/// flagging would need 4+ sites plus a persisted flag byte plus a
/// CACHE_VERSION bump for a fact that is derivable). Runs after every
/// `apply_n2h`/`height_to_normal` mip rebuild by construction. A texture id
/// ALSO referenced by a rough/metal/emissive/albedo role is skipped loudly:
/// the dedup key is (path, srgb), so one linear map can serve two roles, and
/// slope-encoded mips would corrupt the other role's samples — coarser,
/// never wrong. Alpha (height) mips are untouched by the slope arm, so the
/// BVH height sweep and the relief march see identical data either way.
pub fn finalize_normal_mips(scene: &mut Scene) {
    use std::collections::HashSet;
    if !crate::texture::slope_mips_enabled() || !crate::texture::mips_enabled() {
        return;
    }
    let mut normal_ids: HashSet<u32> = HashSet::new();
    let mut other_ids: HashSet<u32> = HashSet::new();
    for m in &scene.materials {
        if m.normal_tex != NO_TEX {
            normal_ids.insert(m.normal_tex);
        }
        for id in [m.rough_tex, m.metal_tex, m.emissive_tex] {
            if id != NO_TEX {
                other_ids.insert(id);
            }
        }
        if let MatKind::Textured { tex } = m.kind {
            other_ids.insert(tex);
        }
    }
    if normal_ids.is_empty() {
        return; // procedural/stress scenes: structurally untouched
    }
    let shared = normal_ids.intersection(&other_ids).count();
    if shared > 0 {
        eprintln!(
            "slope-mips: {shared} normal map(s) shared with another texture role — kept on the box filter"
        );
    }
    let t0 = std::time::Instant::now();
    let n: u32 = {
        use rayon::prelude::*;
        scene
            .textures
            .par_iter_mut()
            .enumerate()
            .filter(|(i, _)| {
                let id = *i as u32;
                normal_ids.contains(&id) && !other_ids.contains(&id)
            })
            .map(|(_, t)| {
                t.normal_role = true;
                t.rebuild_mips();
                1u32
            })
            .sum()
    };
    if n > 0 {
        eprintln!(
            "slope-mips: {n} normal map chain(s) rebuilt slope-space in {:.0} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    // Spec-AA companions (`--no-spec-aa` kills): wrap each rebuilt chain's
    // slope-variance planes into a grayscale companion texture APPENDED at
    // the end of the table — existing ids never shift (the cache-v7
    // argument), and every store site runs before this pass, so a companion
    // can never reach a sidecar. Level 0 is ALL-ZERO: the base level has no
    // filtered-away variance, and the `lod <= 0` bilinear escape then reads
    // an exact 0.0, which is what makes shade's fold self-disable at
    // magnification with no extra branch. Sequential in id order —
    // deterministic table layout. The planes MOVE out of the source texture
    // (they were staging there, never sampled in place).
    if spec_aa() {
        let n_tex = scene.textures.len();
        let mut tex_var = vec![NO_TEX; n_tex];
        let mut added = 0u32;
        for id in 0..n_tex {
            if scene.textures[id].var_mips.is_empty() {
                continue;
            }
            let (w, h) = (scene.textures[id].w, scene.textures[id].h);
            let var_mips = std::mem::take(&mut scene.textures[id].var_mips);
            tex_var[id] = scene.textures.len() as u32;
            scene.textures.push(crate::texture::Texture {
                w,
                h,
                texels: vec![[0, 0, 0, 255]; (w * h) as usize],
                alpha_masked: false,
                srgb: false,
                source: String::new(),
                h2n: false,
                n2h: false,
                normal_role: false,
                mips: var_mips,
                var_mips: Vec::new(),
            });
            added += 1;
        }
        scene.tex_var = tex_var;
        if added > 0 {
            eprintln!("spec-aa: {added} slope-variance companion(s) appended");
        }
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

/// Water material fields the loader stamps onto a `matclass::WATER` material
/// (`Pbr::water`). `WATER_TINT` is the absorption/transmission color — light
/// blue-green so the Beer–Lambert exponent does the depth work: at
/// `TRANS_DEPTH_K·diag` of traversal the medium is exactly this tint, and red
/// (0.75) extinguishes fastest, so shallow rims stay clear and the deep basin
/// goes blue-green. It is the ONE look knob (screenshot-tuned; gates prove
/// soundness, never looks). `WATER_IOR` 1.33 vs glassware's 1.5 lowers the
/// face-on reflectance (2% vs 4%). `WATER_RIPPLE_AMP` is the peak ripple slope.
pub const WATER_TINT: Vec3A = Vec3A::new(0.75, 0.92, 0.96);
pub const WATER_IOR: f32 = 1.33;
pub const WATER_RIPPLE_AMP: f32 = 0.25;

/// Post-load spray reclassification (shared OBJ+glTF, the `derive_heights`
/// pass family — cold loads only, BEFORE the cache store, so warm loads
/// inherit the retag from the sidecar; `--no-spray` kills it and keys the
/// cache lever word). Union-find over shared vertex POSITIONS (exact f32
/// bits), restricted to transmissive-material triangles: tiny disconnected
/// islands (the airborne droplets of a fountain) become a white scattering
/// "spray" clone of their source material, while pools/streams/glassware
/// (large components) stay glass. Position welding, not index sharing, is
/// load-bearing: Minecraft-style exporters emit water per block, UNWELDED —
/// every face has private indices — so index connectivity saw rungholt's
/// ocean as ~150k one-block "droplets" and retagged the whole sea to matte
/// spray. Grid exports repeat the same coordinate per shared corner ⇒ same
/// bits ⇒ the ocean welds into one over-limit component. A missed weld
/// (-0.0 vs 0.0, real float drift) only degrades toward the index-based
/// status quo on that fragment — never a new failure class. Purely
/// load-time — no per-ray cost, zero rng, and a scene with no transmissive
/// materials is structurally untouched.
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
    // Canonical id per POSITION: the first vertex index seen at those exact
    // bits, in the fixed triangle scan order — a pure function of the scene.
    // Lookup-only after insert (never iterated), so HashMap order cannot
    // reach the output and the .fcache/world-sidecar bytes stay
    // run-to-run identical.
    let mut canon: HashMap<[u32; 3], u32> = HashMap::new();
    fn cid(canon: &mut HashMap<[u32; 3], u32>, positions: &[Vec3A], i: u32) -> u32 {
        let p = positions[i as usize];
        *canon.entry([p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).or_insert(i)
    }
    let mut any = false;
    for (t, idx) in scene.indices.iter().enumerate() {
        if scene.materials[scene.tri_mat[t] as usize].transmission <= 0.0 {
            continue;
        }
        any = true;
        let r0 = find(&mut parent, cid(&mut canon, &scene.positions, idx[0]));
        let r1 = find(&mut parent, cid(&mut canon, &scene.positions, idx[1]));
        let r2 = find(&mut parent, cid(&mut canon, &scene.positions, idx[2]));
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
        let root = find(&mut parent, cid(&mut canon, &scene.positions, idx[0]));
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
        let root = find(&mut parent, cid(&mut canon, &scene.positions, scene.indices[t][0]));
        if !spray_roots.contains_key(&root) {
            continue;
        }
        let clone = *clones.entry(m).or_insert_with(|| {
            // Field-by-field carry-over (Material is deliberately not Clone);
            // the scoped borrow ends before the push.
            let spray = {
                let s = &scene.materials[m as usize];
                Material {
                    // Lift from the source TINT, not the raw albedo: bitwise
                    // unchanged for glassware (sentinel ⇒ albedo), but a
                    // water-sourced droplet lifts from its light blue-green
                    // instead of the now-unlifted dark Kd (else spray would dim
                    // from ~0.93 to ~0.64 as a side effect of the water class).
                    albedo: s.trans_tint_or(s.albedo).lerp(Vec3A::ONE, SPRAY_ALBEDO_LIFT),
                    roughness: SPRAY_ROUGHNESS,
                    metallic: 0.0,
                    anisotropy: s.anisotropy,
                    sheen: s.sheen,
                    translucency: SPRAY_TRANSLUCENCY,
                    transmission: 0.0,
                    // A droplet is opaque white scatter: sentinel tint, plain
                    // IOR, no ripples.
                    trans_tint: Vec3A::splat(-1.0),
                    ior: 1.5,
                    ripple_amp: 0.0,
                    emissive: s.emissive,
                    normal_tex: s.normal_tex,
                    normal_scale: s.normal_scale,
                    height_amp: s.height_amp,
                    rough_tex: s.rough_tex,
                    metal_tex: s.metal_tex,
                    emissive_tex: s.emissive_tex,
                    class: s.class,
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

/// Drop TRANSMISSIVE triangles exactly coincident with an OPAQUE triangle —
/// same three vertex POSITIONS by f32 bits, any winding/rotation (grid
/// exporters repeat coordinates exactly, the `reclassify_spray` weld
/// precedent). Minecraft exports carry the case at scale: rungholt's ocean
/// VOLUME has ~66k bottom faces coplanar-coincident with the seabed block
/// tops (and the loader's ground quad used to share the same plane — see
/// `GROUND_DROP`). A transmissive face flush against a solid transmits
/// nothing physically, and keeping it is worse than redundant: the CPU and
/// GPU intersectors break the exact-t tie DIFFERENTLY, and when the
/// transmissive face wins, the transmission chain's eps-advanced
/// continuation starts INSIDE the solid and tunnels past it (on rungholt the
/// CPU tunneled clean through the world's floor to sky while the GPU shaded
/// the ground quad — the "water is more transparent on the CPU path"
/// report; measured 4.1% converged CPU-vs-GPU radiance at a water-dominant
/// pose, ~9% on water pixels). Culling makes both tracers shade the lit
/// seabed — agreeing AND correct. Runs at cold load beside
/// `reclassify_spray` (direct load + per world island), rides the .fcache;
/// `--no-coincident-cull` restores the pre-cull geometry and keys the cache
/// lever word. Coincident transmissive-over-TRANSMISSIVE pairs (internal
/// faces between adjacent water blocks) are deliberately KEPT — same
/// material either way, so the tie is invisible; dropping them is a
/// follow-on with its own soundness argument.
pub fn cull_coincident(scene: &mut Scene) {
    if !coincident_cull_enabled() {
        return;
    }
    // Structural off-state: no transmissive materials ⇒ untouched (the
    // procedural/stress scenes never reach the hashing below).
    if !scene.materials.iter().any(|m| m.transmission > 0.0) {
        return;
    }
    let key_of = |idx: &[u32; 3], positions: &[Vec3A]| -> [u32; 9] {
        let mut vk = [[0u32; 3]; 3];
        for (k, &i) in idx.iter().enumerate() {
            let p = positions[i as usize];
            vk[k] = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
        }
        vk.sort_unstable();
        [
            vk[0][0], vk[0][1], vk[0][2], vk[1][0], vk[1][1], vk[1][2], vk[2][0], vk[2][1],
            vk[2][2],
        ]
    };
    // Small map (transmissive tris only), scanned against by the opaque
    // pass — never the other way around: hashing all 6.7M opaque tris of a
    // Minecraft city would cost ~400 MB for nothing.
    let mut trans: HashMap<[u32; 9], Vec<u32>> = HashMap::new();
    for (t, idx) in scene.indices.iter().enumerate() {
        if scene.materials[scene.tri_mat[t] as usize].transmission > 0.0 {
            trans.entry(key_of(idx, &scene.positions)).or_default().push(t as u32);
        }
    }
    if trans.is_empty() {
        return;
    }
    // The opaque scan is the pass's whole cost — the map holds only the
    // transmissive tris, but a LOOKUP still hashes the key, so every opaque
    // tri pays key_of + SipHash (~10M on San Miguel) — hence the fan-out on
    // the global rayon pool (the load-time idiom; a world island's inner
    // par_iter deliberately inherits the one pool). Determinism: the drop
    // SET is order-independent (matches collected, deduped serially after),
    // so the culled scene — and therefore the sidecar bytes — are identical
    // to a serial scan's at any thread count.
    let matched: Vec<u32> = {
        use rayon::prelude::*;
        scene
            .indices
            .par_iter()
            .enumerate()
            .filter(|&(t, _)| scene.materials[scene.tri_mat[t] as usize].transmission <= 0.0)
            .filter_map(|(_, idx)| trans.get(&key_of(idx, &scene.positions)))
            .flat_map_iter(|list| list.iter().copied())
            .collect()
    };
    let mut drop = vec![false; scene.indices.len()];
    let mut n_drop = 0usize;
    for tt in matched {
        if !drop[tt as usize] {
            drop[tt as usize] = true;
            n_drop += 1;
        }
    }
    let n_trans: usize = trans.values().map(Vec::len).sum();
    if n_drop == 0 {
        eprintln!("coincident-cull: 0 of {n_trans} transmissive faces coincide with opaque geometry");
        return;
    }
    // Rebuild the index/material streams without the dropped faces (vertex
    // streams stay — unreferenced positions are harmless and keep every
    // other index stable).
    let mut indices = Vec::with_capacity(scene.indices.len() - n_drop);
    let mut tri_mat = Vec::with_capacity(scene.indices.len() - n_drop);
    for t in 0..scene.indices.len() {
        if !drop[t] {
            indices.push(scene.indices[t]);
            tri_mat.push(scene.tri_mat[t]);
        }
    }
    scene.indices = indices;
    scene.tri_mat = tri_mat;
    eprintln!(
        "coincident-cull: dropped {n_drop} of {n_trans} transmissive faces coincident with opaque geometry"
    );
}

/// Spray gates, run by `--check` (the tinted_shadow_self_test pattern —
/// hand-built islands, closed-form expectations): only tiny TRANSMISSIVE
/// components retag, the clone's overrides are pinned bitwise, two islands
/// of one source material share one deduped clone, large components and
/// tiny opaque islands are untouched, and the lever-off arm is a no-op.
/// Islands E/F pin the POSITION weld: an unwelded edge-to-edge strip whose
/// welded extent exceeds the limit must stay glass (the Minecraft-export
/// ocean shape), while coincident-position tiny tris still weld and retag
/// (the droplet feature survives the weld).
pub fn spray_self_test() -> Result<(), String> {
    let glass = |transmission: f32| Material {
        albedo: Vec3A::new(0.8, 0.85, 0.9),
        roughness: 0.05,
        metallic: 0.0,
        anisotropy: 0.0,
        sheen: 0.0,
        translucency: 0.0,
        transmission,
        trans_tint: Vec3A::splat(-1.0),
        ior: 1.5,
        ripple_amp: 0.0,
        emissive: Vec3A::ZERO,
        normal_tex: NO_TEX,
        normal_scale: 1.0,
        height_amp: 0.0,
        rough_tex: NO_TEX,
        metal_tex: NO_TEX,
        emissive_tex: NO_TEX,
        class: crate::matclass::IDX_DEFAULT as u8,
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
        let tiny = |at: Vec3A, mat: u32,
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
        // Island E (tris 5..13): an unwelded "ocean" strip — 4 transmissive
        // quads edge-to-edge by POSITION only (each pushes its own 4 verts,
        // private indices — the Minecraft export shape). One quad's diag
        // ≈ 5.7e-4 sits UNDER the limit (≈ 9.5e-4 at this scene's diag);
        // the welded strip ≈ 1.65e-3 sits OVER it. Index connectivity
        // retags all four as droplets; position welding must keep them
        // glass. The shared corners derive from the SAME `(col as f32)*S`
        // expression on both sides — bit-identity by construction,
        // mirroring a grid exporter's repeated coordinate text.
        const S: f32 = 0.4e-3;
        let base = Vec3A::new(0.3, 0.3, 0.5);
        for q in 0..4u32 {
            let b = positions.len() as u32;
            let xl = base.x + q as f32 * S;
            let xr = base.x + (q + 1) as f32 * S;
            positions.push(Vec3A::new(xl, base.y, base.z));
            positions.push(Vec3A::new(xr, base.y, base.z));
            positions.push(Vec3A::new(xl, base.y + S, base.z));
            positions.push(Vec3A::new(xr, base.y + S, base.z));
            indices.push([b, b + 1, b + 2]);
            indices.push([b + 1, b + 3, b + 2]);
            tri_mat.push(0);
            tri_mat.push(0);
        }
        // Island F (tris 13..15): the weld must not LOSE the droplet
        // feature — two copies of one tiny tri, positions bit-equal but
        // indices private, detached from everything else: they weld into
        // ONE still-under-limit component and retag onto the shared clone.
        for _ in 0..2 {
            let b = positions.len() as u32;
            positions.push(Vec3A::new(0.9, 0.1, 0.5));
            positions.push(Vec3A::new(0.9 + 1e-4, 0.1, 0.5));
            positions.push(Vec3A::new(0.9, 0.1 + 1e-4, 0.5));
            indices.push([b, b + 1, b + 2]);
            tri_mat.push(0);
        }
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
            emissive: crate::emissive::EmissiveLights::off(),
            sun: crate::sky::Sun::new(Vec3A::Y),
            sky_sh: crate::sh::Sh9::ZERO,
            sky_scale: 1.0,
            night: 0.0,
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
        // Island E: unwelded-but-position-connected over-limit strip stays
        // glass (the rungholt ocean shape — index-based union retags it).
        if (5..13).any(|t| sc.tri_mat[t] != 0) {
            return Err(format!(
                "unwelded over-limit strip must stay glass (tri_mat {:?})",
                &sc.tri_mat[5..13]
            ));
        }
        // Island F: coincident-position tris weld into one still-tiny
        // component and retag onto the SAME deduped clone.
        if sc.tri_mat[13] != 2 || sc.tri_mat[14] != 2 {
            return Err(format!(
                "coincident tiny tris must weld and retag onto the shared clone (got {}, {})",
                sc.tri_mat[13], sc.tri_mat[14]
            ));
        }
        let s = &sc.materials[2];
        // Sentinel-tint glass source ⇒ lift from albedo verbatim (the water
        // path is covered by the tint-source arm below).
        let want_albedo = Vec3A::new(0.8, 0.85, 0.9).lerp(Vec3A::ONE, SPRAY_ALBEDO_LIFT);
        if s.transmission != 0.0
            || s.translucency != SPRAY_TRANSLUCENCY
            || s.roughness != SPRAY_ROUGHNESS
            || s.metallic != 0.0
            || s.trans_tint.x >= 0.0
            || s.ior != 1.5
            || s.ripple_amp != 0.0
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
        let mut want_off = vec![0u32, 0, 0, 0, 1];
        want_off.extend([0; 10]); // islands E (8 tris) + F (2 tris)
        if off.materials.len() != 2 || off.tri_mat != want_off {
            return Err("lever off must be a structural no-op".into());
        }
        Ok(())
    };
    let r = run();
    restore(r)
}

/// Coincident-cull gates, run by `--check` (the spray_self_test pattern):
/// a transmissive tri whose three positions bit-equal an opaque tri's (any
/// winding) is dropped; transmissive-over-transmissive and non-coincident
/// pairs survive; opaque tris are never dropped; the lever-off arm is a
/// structural no-op.
pub fn coincident_self_test() -> Result<(), String> {
    let saved = coincident_cull_enabled();
    let restore = |r: Result<(), String>| -> Result<(), String> {
        set_coincident_cull(saved);
        r
    };
    set_coincident_cull(true);
    let run = || -> Result<(), String> {
        let mk = || -> (Vec<Vec3A>, Vec<[u32; 3]>, Vec<u32>) {
            // Tri 0: opaque base at exact grid positions. Tri 1: transmissive,
            // SAME positions, rotated winding + private verts (the Minecraft
            // water-bottom-over-seabed shape). Tri 2: transmissive, elsewhere.
            // Tri 3: transmissive, coincident with tri 2 (trans-over-trans —
            // must be KEPT). Tri 4: opaque coincident with tri 0 (opaque pair
            // — nothing drops).
            let a = Vec3A::new(0.0, 0.0, 0.0);
            let b = Vec3A::new(1.0, 0.0, 0.0);
            let c = Vec3A::new(0.0, 0.0, 1.0);
            let d = Vec3A::new(3.0, 0.5, 0.0);
            let e = Vec3A::new(4.0, 0.5, 0.0);
            let f = Vec3A::new(3.0, 0.5, 1.0);
            let mut positions = Vec::new();
            let mut indices = Vec::new();
            let push = |p0: Vec3A, p1: Vec3A, p2: Vec3A,
                        positions: &mut Vec<Vec3A>,
                        indices: &mut Vec<[u32; 3]>| {
                let b0 = positions.len() as u32;
                positions.extend_from_slice(&[p0, p1, p2]);
                indices.push([b0, b0 + 1, b0 + 2]);
            };
            push(a, b, c, &mut positions, &mut indices); // 0 opaque
            push(c, a, b, &mut positions, &mut indices); // 1 trans, rotated
            push(d, e, f, &mut positions, &mut indices); // 2 trans
            push(f, e, d, &mut positions, &mut indices); // 3 trans, coincident w/ 2
            push(b, a, c, &mut positions, &mut indices); // 4 opaque, coincident w/ 0
            (positions, indices, vec![1, 0, 0, 0, 1])
        };
        let glass = |transmission: f32| Material {
            albedo: Vec3A::new(0.8, 0.85, 0.9),
            roughness: 0.05,
            metallic: 0.0,
            anisotropy: 0.0,
            sheen: 0.0,
            translucency: 0.0,
            transmission,
            trans_tint: Vec3A::splat(-1.0),
            ior: 1.5,
            ripple_amp: 0.0,
            emissive: Vec3A::ZERO,
            normal_tex: NO_TEX,
            normal_scale: 1.0,
            height_amp: 0.0,
            rough_tex: NO_TEX,
            metal_tex: NO_TEX,
            emissive_tex: NO_TEX,
            class: crate::matclass::IDX_DEFAULT as u8,
            kind: MatKind::Diffuse,
        };
        let scene_of = |cull: bool| -> Scene {
            let (positions, indices, tri_mat) = mk();
            let n = positions.len();
            let mut sc = Scene {
                positions,
                normals: vec![Vec3A::Y; n],
                texcoords: vec![Vec2::ZERO; n],
                indices,
                tri_mat,
                materials: vec![glass(0.9), glass(0.0)],
                textures: Vec::new(),
                any_alpha: false,
                any_height: false,
                any_transmissive: false,
                emissive: crate::emissive::EmissiveLights::off(),
                sun: crate::sky::Sun::new(Vec3A::Y),
                sky_sh: crate::sh::Sh9::ZERO,
                sky_scale: 1.0,
                night: 0.0,
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
            set_coincident_cull(cull);
            cull_coincident(&mut sc);
            set_coincident_cull(true);
            sc
        };
        let sc = scene_of(true);
        // Tri 1 (trans coincident with opaque 0) dropped; everything else
        // kept in order.
        if sc.tri_mat != vec![1u32, 0, 0, 1] {
            return Err(format!(
                "cull kept the wrong faces: tri_mat {:?} (want [1, 0, 0, 1])",
                sc.tri_mat
            ));
        }
        // The survivors' first vertex indices prove tri identity (streams
        // rebuilt, verts untouched): 0, 6(d..), 9(f..), 12(b..).
        if sc.indices.iter().map(|i| i[0]).collect::<Vec<_>>() != vec![0, 6, 9, 12] {
            return Err(format!("cull dropped the wrong tri: indices {:?}", sc.indices));
        }
        let off = scene_of(false);
        if off.tri_mat != vec![1u32, 0, 0, 0, 1] || off.indices.len() != 5 {
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
    // Clustered emissive lights (src/emissive.rs) — derived HERE so every
    // load path (cold, warm sidecar, tile replication, world merge) gets
    // them re-derived for free, like sky_sh. Serial + index-ordered ⇒
    // byte-deterministic; emissive-free scenes pay O(materials) and keep
    // the structural count-0 off state. One loud line iff lights exist.
    scene.emissive = crate::emissive::derive(scene, crate::emissive::budget());
    if scene.emissive.count > 0 {
        let total: f32 = (0..scene.emissive.count as usize)
            .map(|i| {
                crate::emissive::lum(Vec3A::from(scene.emissive.lights[i].color))
                    * std::f32::consts::PI
            })
            .sum();
        eprintln!(
            "emissive lights: {} clusters (budget {}, total power {:.3} incl x{} boost){}",
            scene.emissive.count,
            crate::emissive::budget(),
            total,
            crate::emissive::EL_BOOST,
            if crate::emissive::enabled() { "" } else { " — OFF (arm with --emissive-lights)" }
        );
    }

    derive_detail_scales(scene);
    refresh_sky_sh(scene);
}

/// Re-derive the SH ambient from the current sun/sky state. Split out of
/// `finalize_scalars` so `apply_tod` can re-run it without the O(n) position
/// scan. Deterministic quadrature, no rng — which is why `scene_cache` needs
/// no format change for it.
///
/// `sky::gather`, not the full sky: a gather path must never see the sun disc
/// (the direct loop already delivers it, with a shadow ray). See sky.rs's
/// central invariant. At night `scene.sun` IS the moon, so the ambient is the
/// moonlit dome — the frequency split carries over unchanged — PLUS the star
/// field's smooth mean (`sky::star_glow`, gated by `scene.night`), which is
/// what gives night a moon-independent ambient floor. `gather` is bitwise
/// `dome` whenever `night == 0`, so every day session is untouched.
pub fn refresh_sky_sh(scene: &mut Scene) {
    let sun = scene.sun.dir;
    let scale = scene.sky_scale;
    let night = scene.night;
    scene.sky_sh = crate::sh::Sh9::project(|d| crate::sky::gather(d, sun, scale, night));
}

/// Texel-equivalent world size for materials with NO albedo texture, as a
/// fraction of the CONTENT diagonal (never `Scene::diag` — the fireflies
/// lesson: the standard ground quad inflates the full diag ~17× on
/// procedural scenes). USER-CALIBRATED 2026-08-06, two rounds: the 3e-4
/// starting point read as chunky blotches on powerplant ("needs to be
/// like 100x smaller" → 3e-6, then "maybe 2x smaller than it currently
/// is" → 1.5e-6) — the grain is genuinely fine surface texture now, at
/// the price that it resolves only at CLOSE range (the AO pools, 8× the
/// grain scale, carry the mid distance). Scaled by `--detail-untex-scale`
/// (0..=4 around this center).
pub const DETAIL_UNTEX_K: f32 = 1.5e-6;

/// Per-material detail-field texel scale (see the `detail_scales` field doc):
/// a sampled per-material MEDIAN of the per-triangle texel size —
/// deterministic (serial, fixed stride), and deliberately NOT per-face,
/// which seams on greedy-meshed atlases. Materials WITHOUT an albedo map
/// get a SYNTHETIC content-diag-relative scale instead (the untextured arm
/// below) — the detail field needs no UVs (its domain is the rest-pose
/// position over s), only a texel-equivalent size, so powerplant-class
/// scenes carry the grain too.
/// LOAD-ONLY by design: called from `finalize_scalars` alone — it landed
/// inside `refresh_sky_sh` for one commit, where the interactive TOD path's
/// throttled SH steps re-ran the whole ~2M-sample pass ~20×/s during world
/// flight (the residual half of the 235→17 fps stall).
fn derive_detail_scales(scene: &mut Scene) {
    scene.detail_scales = vec![0.0; scene.materials.len()];
    if scene.texcoords.len() == scene.positions.len() {
        let mut samples: Vec<Vec<f32>> = vec![Vec::new(); scene.materials.len()];
        let stride = (scene.indices.len() / 2_000_000).max(1);
        let mut i = 0;
        while i < scene.indices.len() {
            let m = scene.tri_mat[i] as usize;
            if let MatKind::Textured { tex } = scene.materials[m].kind {
                if let Some((bu, bv)) = crate::shade::tri_uv_basis(scene, i as u32) {
                    let t = &scene.textures[tex as usize];
                    let s = ((bu.length() / t.w as f32) * (bv.length() / t.h as f32)).sqrt();
                    if s.is_finite() && s > 0.0 {
                        samples[m].push(s);
                    }
                }
            }
            i += stride;
        }
        for (m, mut v) in samples.into_iter().enumerate() {
            if !v.is_empty() {
                v.sort_by(|a, b| a.total_cmp(b));
                scene.detail_scales[m] = v[v.len() / 2];
            }
        }
    }
    // The UNTEXTURED arm: no albedo map means no texel size to measure, so
    // those materials take DETAIL_UNTEX_K × content diag × the
    // `--detail-untex-scale` knob (0 ⇒ they stay 0.0 — the bitwise off arm).
    // Keyed on KIND, deliberately never on "collected no samples": a
    // Textured material whose UVs are all degenerate must keep its bitwise
    // 0.0 structural off (the self-test pin) — its albedo really is
    // texture-driven and a synthetic grain domain over a broken atlas is
    // not what the median rule promised. Outside the texcoords-length guard
    // above on purpose: hand-built scenes without per-vertex UVs still get
    // the untextured arm.
    let s_untex = DETAIL_UNTEX_K
        * detail_untex_scale()
        * (scene.content_max - scene.content_min).length();
    if s_untex > 0.0 && s_untex.is_finite() {
        for (m, mat) in scene.materials.iter().enumerate() {
            if !matches!(mat.kind, MatKind::Textured { .. }) {
                scene.detail_scales[m] = s_untex;
            }
        }
    }
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
    apply_tod_lit(scene, hour);
    refresh_sky_sh(scene);
}

/// The cheap closed-form half of `apply_tod` — sun/moon direction, dome
/// scale, night — WITHOUT the SH re-projection. The interactive TOD path
/// (main.rs's `sun_moved` block) calls this every write so the shadow
/// direction never lags, and re-projects the SH ambient only on
/// `SH_TOD_STEP` steps of the eased clock: the world's flight attractors
/// write tod every MOVING frame, and a per-write projection was a ~54 ms
/// main-thread stall (measured 235 -> 17 fps flying, GPU idle). Every other
/// caller takes `apply_tod`, which composes both halves — semantics
/// unchanged.
pub fn apply_tod_lit(scene: &mut Scene, hour: f32) {
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
}

/// The interactive TOD path's SH re-projection quantum, in game hours
/// (0.05 h = 3 game-minutes ≈ 0.75° of sun arc). The smooth order-2 ambient
/// stepping by this much is invisible under the temporal integrators (the
/// cloud-drift shading-change class), while it caps the projection rate at
/// ~20/s during the fastest scrub — with the parallel `Sh9::project` that
/// is ~4% of a frame instead of the old per-write stall. The sun disc,
/// shadows, sky_scale, and night never quantize (apply_tod_lit runs every
/// write).
pub const SH_TOD_STEP: f32 = 0.05;

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
    // END-TO-END STARLIGHT: the SH the renderer actually shades from must carry
    // the star field's floor, not just the moonlit dome. sky::self_test gates
    // the term itself (energy vs the enumerated field, the band); this gates
    // that `refresh_sky_sh` really routes through `sky::gather` — a revert to
    // `dome` there would pass every gate in sky.rs and silently un-light the
    // night.
    let e_night = sc.sky_sh.irradiance(Vec3A::Y);
    let moon_only = crate::sh::Sh9::project(|d| sky::dome(d, sc.sun.dir, sc.sky_scale))
        .irradiance(Vec3A::Y);
    if lum(e_night) <= lum(moon_only) * 1.05 {
        return Err(format!(
            "midnight sky_sh {e_night:?} carries no starlight over the moonlit \
             dome {moon_only:?} — refresh_sky_sh is not calling sky::gather"
        ));
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

/// How far BELOW the model's rest plane (y = 0 after the diag-10 fit) the
/// LOADED-scene ground quad sits. It used to sit exactly AT y = 0, which
/// z-fights every model face on the rest plane — rungholt's entire seabed —
/// and the CPU/GPU intersectors break an exact-t tie DIFFERENTLY, so the
/// same scene shaded a different surface per render mode (the water-
/// transparency report; see `cull_coincident`). 1e-4 of the fit diagonal
/// (1e-3 units ≈ a tenth of a Minecraft block) is far above f32 t-rounding
/// at ±60 coordinates and far below anything visible. The PROCEDURAL and
/// stress scenes deliberately keep y = 0 — no transmissive geometry, and
/// their gate images are pinned byte-identical.
pub(crate) const GROUND_DROP: f32 = 1e-3;

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
    // loaders' `quad()` (two fan triangles, 6 duplicated verts, +y normal),
    // at the loaded-scene GROUND_DROP (never on the models' rest plane).
    let s = fh + 6.0;
    let (a, b, c, d) = (
        Vec3A::new(-s, -GROUND_DROP, -s),
        Vec3A::new(-s, -GROUND_DROP, s),
        Vec3A::new(s, -GROUND_DROP, s),
        Vec3A::new(s, -GROUND_DROP, -s),
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
        emissive: crate::emissive::EmissiveLights::off(),
        // The sun is at infinity: a replicated field sees the SAME sun, at the
        // same angle, with the same irradiance, everywhere. Nothing to rescale.
        sun: base.sun,
        sky_sh: crate::sh::Sh9::ZERO,
        sky_scale: base.sky_scale,
        night: base.night,
        // Tiling re-derives the content box, so a stale partition would be
        // wrong; the caller re-attaches (foliage::attach) after tile_scene.
        sway: None,
        sway_regions: Vec::new(),
        diag: 0.0,
        eps: 0.0,
        ao_radius: 0.0,
        detail_scales: Vec::new(),
        content_min: Vec3A::ZERO,
        content_max: Vec3A::ZERO,
        tex_var: Vec::new(),
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

/// Parse an MTL `Tf r g b` transmission-filter value (three floats). Only the
/// chromaticity is used downstream (`matclass::tf_chromatic`) — the water tint
/// is the curated constant — so a malformed/short value is simply `None`.
fn parse_tf(v: &str) -> Option<[f32; 3]> {
    let mut it = v.split_whitespace().filter_map(|s| s.parse::<f32>().ok());
    match (it.next(), it.next(), it.next()) {
        (Some(r), Some(g), Some(b)) => Some([r, g, b]),
        _ => None,
    }
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
    crate::progress::phase(crate::progress::Phase::Parse, path, 0);
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

    // Water detection (a glass-tier refinement): which materials are used by
    // an OBJ object/group named `water`/`agua` — `o Water` → `materialo` in
    // both San Miguel flavors (and NOT `o Fountain`, the stone basin).
    // `--no-water` disarms the whole signal path so classification is exactly
    // the pre-feature glassware.
    let water_on = water_enabled();
    let mut water_named = vec![false; obj_mats.len()];
    if water_on {
        for m in &models {
            if let Some(id) = m.mesh.material_id {
                if id < water_named.len() && crate::matclass::water_name(&m.name) {
                    water_named[id] = true;
                }
            }
        }
    }

    let mut b = SceneBuilder::new();
    let ground = b.material(Vec3A::new(0.42, 0.46, 0.40), 0.55, 0.0);
    // GROUND_DROP: the quad must not share the fitted model's rest plane
    // (y = 0) — see the const.
    let s = 60.0;
    b.quad(
        Vec3A::new(-s, -GROUND_DROP, -s),
        Vec3A::new(-s, -GROUND_DROP, s),
        Vec3A::new(s, -GROUND_DROP, s),
        Vec3A::new(s, -GROUND_DROP, -s),
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
        crate::progress::phase(crate::progress::Phase::Textures, "", by_size.len() as u32);
        by_size
            .par_iter()
            .with_max_len(1)
            .filter_map(|r| match image::open(&r.path) {
                Ok(img) => {
                    crate::progress::tick();
                    Some(((r.path.clone(), r.srgb), Texture::from_image(img, r.srgb)))
                }
                Err(e) => {
                    crate::progress::tick();
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
        .enumerate()
        .map(|(mi, m)| {
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
            // Water refines the glass tier only: `water_hint` from the OBJ
            // object name, `tf` the parsed MTL transmission filter (tobj has
            // no first-class Tf — it rides unknown_param, the `norm`/`Pr`
            // precedent). `water_on` reaches classify itself, which gates the
            // stem/material-name cues too — gating only hint+tf here left
            // name-classified Minecraft water immune to --no-water.
            let tf = if water_on { m.unknown_param.get("Tf").and_then(|v| parse_tf(v)) } else { None };
            let (class, pbr) = crate::matclass::classify(
                stem.as_deref(),
                &m.name,
                m.shininess,
                m.illumination_model,
                water_on && water_named[mi],
                tf,
                water_on,
            );
            class_counts[class] += 1;
            if pbr.transmission > 0.0 && !pbr.water {
                // Transmitted light is tinted by albedo and San Miguel's
                // glass Kd is 0.1-0.2 dark — lift toward white or glass
                // renders near-black. Water is EXEMPT: it carries its color in
                // `trans_tint` (below), so its raw dark Kd stays, which kills
                // the neutral `kd·(1−T)` wash that read as chrome.
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
            // Water carries its color/IOR/ripples here; every other material
            // takes the sentinel tint (⇒ albedo verbatim), IOR 1.5 (bit-
            // identical to the old GLASS_IOR const), and no ripples.
            let (trans_tint, ior, ripple_amp) = if pbr.water {
                (WATER_TINT, WATER_IOR, WATER_RIPPLE_AMP)
            } else {
                (Vec3A::splat(-1.0), 1.5, 0.0)
            };
            b.material_full(Material {
                albedo: kd,
                roughness,
                metallic,
                anisotropy: 0.0,
                sheen: pbr.sheen,
                translucency: pbr.translucency,
                transmission: pbr.transmission,
                trans_tint,
                ior,
                ripple_amp,
                emissive,
                normal_tex,
                normal_scale,
                height_amp,
                rough_tex,
                metal_tex,
                emissive_tex,
                class: class as u8,
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
