//! Emissive surfaces as DIRECT-tier lights — load-time clustering of emissive
//! triangles into ≤ `MAX_EMISSIVE_LIGHTS` virtual DISC lights, sampled in
//! `shade()`'s direct loop the way the fireflies are (the proven template):
//! deterministic iteration, a windowed falloff that is exactly 0 at its
//! influence radius, ONE hard shadow ray per in-range light, ZERO rng draws,
//! CB-row transport with CPU↔GPU parity BY DATA. **Default OFF** — the
//! heightfield arming shape, not the fireflies': `--emissive-lights [N]`
//! ARMS the tier (bare = the default budget; N moves it),
//! `--no-emissive-lights` spells the default. The old LOOK FINDING (physical
//! pools faint before true nightfall) is RESOLVED by `EL_BOOST` = 2 at the
//! C_c fill (2026-08-06, the MOON_E_OVER_PI artistic precedent — user
//! feel-tested "beautiful"; it also ~doubles `r_infl2` pre-cap, so in-range
//! scan/ray counts sit above the pre-boost measurements), and the default
//! stayed OFF anyway on the user's third-round call: the CPU cost is real
//! (measured bistro PRE-boost: +5.5 ms at N=32, of which ~3.3 ms is the
//! shadow rays themselves — irreducible, they ARE the feature; the ~2.2 ms
//! per-pixel scan half is RECOVERED by the per-leaf-tile cull since
//! 2026-08-01 — `cull_tile` below, measured nocull 54.71 → cull 52.46 ms on
//! the same workload) and only ONE world island — bistro — carries emissive
//! maps, so every other session would pay derivation for count 0 and bistro
//! visitors an always-on ray tax. Emitter PLACEMENT has its own A/B lever,
//! `--el-cluster grid|som` (`ClusterMode` / `som_refine` below). Either way
//! a scene with no emissive material derives `count = 0` and every
//! consumer's loop body is unreachable, so the pre-feature renderer is
//! reproduced STRUCTURALLY (guarded branches, no unconditional `+0.0` — the
//! fireflies / `apply_tod`-unreachable precedent).
//!
//! # Why this exists (the sun-disc argument, third verse)
//!
//! Emissive materials were "half-lit": the `color += e` display add makes
//! emitters VISIBLE (including in reflections and through glass), and the
//! opt-in still-frames-only hemi GI tier picks them up as bounce light — but
//! nothing SAMPLES them, so in every default interactive session a glowing
//! street lamp lit nothing, and even under GI a small bright emitter is
//! cosine-gather-undersampled (the DamagedHelmet-visor outlier class). Sharp
//! bright features need their own sampling strategy — the exact argument that
//! made the sun an explicit light instead of dome content. The display add
//! stays untouched (it is the `sky::radiance` half); this module is the
//! gather half.
//!
//! # The once-per-path rule, INVERTED for GI
//!
//! The sun-disc rule excludes a sampled light from gathers. Here the
//! exclusion runs the OTHER way: under `fb.gi` the hemi gather already
//! delivers emissive transport EXACTLY (real geometry, real soft shadows,
//! textured emission — strictly better than any cluster), so GI frames turn
//! the NEE tier OFF instead (the CPU's one-`Some` site gates on `!q.fb.gi`;
//! the GPU clears `FLAG_EMISSIVE` at `fb_mode == 2`). AO mode keeps NEE on —
//! its ambient is sky × openness, no emissive in it. Consequence: no tier
//! double-counts, and `hemi.rs` / `hemi_leaf.hlsl` / both GI reference
//! estimators are untouched by construction.
//!
//! # The light model (disc, not point)
//!
//! Each cluster is a Lambertian disc of radius `rc` with radiance·area sum
//! `C = Σ Aᵢ·L̄ᵢ`: on-axis irradiance at distance d along the normal is
//! exactly `C / (d² + rc²)` — the `+rc²` denominator IS the near-field
//! softening (no hot spot when a shading point stands next to a large
//! emissive panel, and no separate `FF_RMIN_K`-style clamp). `color` stores
//! `C/π` (the sun's `e_over_pi` Lambert convention), and the falloff is
//! windowed by the fireflies' `(1 − d²/r²)²` — C¹ and exactly 0 at
//! `r_infl`, so a light crossing a pixel's influence boundary never pops.
//! The shadow ray stops `rc + 2·eps` SHORT of the cluster center: a ray to
//! the centroid of a bulb mesh would otherwise always be occluded by the
//! bulb itself (known-accept: non-emissive geometry INSIDE the cluster
//! sphere — a lamp housing — does not occlude its own light; the grid pitch
//! keeps clusters tight). Diffuse-only in v1: an emitter's mirror image at a
//! reflection lap cannot be down-weighted, so a `w_l = 1` specular NEE term
//! (the firefly shape — sound only for lights with no geometry) would
//! double-count against the traced VNDF ray that already hits the glowing
//! surface. Lamp highlights keep arriving via that traced ray.
//!
//! # Determinism
//!
//! `derive` is SERIAL and index-ordered — a pure function of the scene
//! (positions/indices/materials/texels/budget), byte-deterministic across
//! thread counts like the BVH build's split phase. Derived in
//! `finalize_scalars` and NEVER serialized (the `sky_sh` precedent: both
//! scene-cache load paths and the world merge re-run finalize, so warm loads
//! re-derive and `CACHE_VERSION` does not move).

use crate::scene::{Material, Scene, NO_TEX};
use crate::texture::Texture;
use glam::{Vec2, Vec3A};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Session enable — `--emissive-lights` ARMS (default OFF, the heightfield
/// shape; `fireflies::set_enabled` mechanics: live-flippable — consumers
/// read it per frame, derivation always runs so the menu toggle needs no
/// restart). The default is DUPLICATED in `cli::Opts`' constructor — flip
/// both in lockstep (the heightfield one-field-two-statics hazard: headless
/// paths that never run main's lever block inherit THIS initializer).
static ENABLED: AtomicBool = AtomicBool::new(false);
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Session budget — `--emissive-lights N`, clamped to `MAX_EMISSIVE_LIGHTS`
/// at parse time (loudly, in main.rs — the CB rows are sized to the max).
/// Restart-tier: it keys the load-time derivation.
static BUDGET: AtomicU32 = AtomicU32::new(EL_DEFAULT);
pub fn set_budget(n: u32) {
    BUDGET.store(n.min(MAX_EMISSIVE_LIGHTS as u32), Ordering::Relaxed);
}
pub fn budget() -> u32 {
    BUDGET.load(Ordering::Relaxed)
}

/// `--el-cluster grid|som` — the emitter-PLACEMENT A/B lever (the
/// `--bvh-builder` bake-off pattern, settings row included; the file value
/// validates through `parse_cluster` so a bogus save can never reach main's
/// exit(2)). `Grid` is the shipped clusterer (grid seed + agglomerative merge —
/// bit-identical to the pre-lever code, a guarded branch); `Som` refines the
/// merged centers with `som_refine` below. Restart-tier: set from main's
/// lever block BEFORE any scene load (`finalize_scalars` derives through
/// it). Derived-never-serialized (the sky_sh precedent — warm loads
/// re-derive), so no CACHE_VERSION move and the lever does not key the
/// .fcache. The judging instrument is the feature's own: a GI (H) still
/// frame at the same pose is ground truth for cluster placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterMode {
    Grid,
    Som,
}
static CLUSTER_MODE: AtomicU32 = AtomicU32::new(0);
/// Pure vocabulary check — the ONE source of the lever's legal names, shared
/// by `set_cluster_mode`, `settings::apply_to_opts`'s warn-ignore validation,
/// and `settings::self_test`'s vocabulary pin (which must stay side-effect-
/// free and so cannot call the storing flavor).
pub fn parse_cluster(name: &str) -> Option<ClusterMode> {
    match name {
        "grid" => Some(ClusterMode::Grid),
        "som" => Some(ClusterMode::Som),
        _ => None,
    }
}
pub fn set_cluster_mode(name: &str) -> Option<ClusterMode> {
    let m = parse_cluster(name)?;
    CLUSTER_MODE.store(m as u32, Ordering::Relaxed);
    Some(m)
}
pub fn cluster_mode() -> ClusterMode {
    match CLUSTER_MODE.load(Ordering::Relaxed) {
        1 => ClusterMode::Som,
        _ => ClusterMode::Grid,
    }
}
/// The lever's spelled name — `cli::lever_snapshot` reads this to prove the
/// parse never called `set_cluster_mode` (the dxr_sbt precedent).
pub fn cluster_mode_name() -> &'static str {
    match cluster_mode() {
        ClusterMode::Grid => "grid",
        ClusterMode::Som => "som",
    }
}

/// Hard cap — sizes the `FrameCb` emissive rows (raise `CB_STRIDE` in
/// lockstep). The per-TILE cull (`cull_tile`, shipped 2026-08-01) bounds
/// the per-pixel scan by the in-range count on the hybrid arms; past ~64
/// the remaining move is the SRV-table follow-on (the CB rows are the
/// 64/64-full root signature's ceiling), never a bare cap raise.
pub const MAX_EMISSIVE_LIGHTS: usize = 64;
pub const EL_DEFAULT: u32 = 32;

/// Seed-grid pitch as a fraction of the CONTENT diagonal (the fireflies'
/// scale anchor — `Scene::diag` is ground-quad-inflated ~17× on the
/// procedural scenes). Nearby emissive tris land in one cell; the pitch
/// doubles until the cell count fits `EL_MAX_CELLS`.
pub const EL_GRID_K: f32 = 0.02;
/// Seed-cluster ceiling before the agglomerative merge — bounds the O(k²)
/// merge at ~1M ops.
pub const EL_MAX_CELLS: usize = 1024;
/// Irradiance-over-π floor that DEFINES the influence radius: `r_infl` is
/// where the un-windowed `lum(C/π)/(d²+rc²)` falls to this. The one
/// cost-vs-reach knob — the per-pixel scan pays a shadow ray per light whose
/// influence sphere contains the pixel, so this trades pool extent for
/// frame time (tuned against the GI ground truth + the spin bench).
pub const EL_MIN_E: f32 = 5e-3;
/// Influence-radius cap × the content diagonal — bounds the scan cost for
/// arbitrarily bright emitters (a huge KHR emissive_strength must not make
/// one light global).
pub const EL_RMAX_K: f32 = 0.5;
/// Near-field clamp on `lum(irradiance)` — the f16-headroom bound (the
/// sun-disc / `FF_GLOW_L_MAX` lesson: the dd plane is f16, and a shading
/// point inside a bulb must not push ±Inf into the upscaler guides).
pub const EL_E_MAX: f32 = 1000.0;
/// Artistic brightness boost applied at the C_c fill (the MOON_E_OVER_PI
/// precedent — the LOOK FINDING's resolution, 2026-08-06: the PHYSICAL
/// calibration's pools read as faint before true nightfall). Multiplies the
/// cluster color AND therefore `r_infl2` (`lum(cp)/EL_MIN_E` doubles — reach
/// and per-pixel scan cost rise with it; `r_cap2` still bounds the radius).
/// The GPU needs NO edit: the el_a/el_b CB rows carry the boosted values and
/// `EL_E_MAX` is absolute and mirrored, so parity holds BY DATA. The
/// self_test power gates scale by this const — a retune moves them with it.
pub const EL_BOOST: f32 = 2.0;
/// Batch epochs for the `--el-cluster som` arm — fixed (never adaptive:
/// determinism is a count, not a convergence test). Runs unconditionally
/// under the som mode, even when the seed count already fits the budget —
/// grid-cell centroids are not Lloyd-stationary, and one code path is the
/// simpler determinism story.
pub const EL_SOM_EPOCHS: usize = 8;

/// Rec.709 luminance — the scalar the influence radius, the near-field
/// clamp, and the merge order key on. Mirrored as a literal float3 in
/// shade.hlsli's emissive block (the clouds-wind constant discipline).
#[inline(always)]
pub fn lum(c: Vec3A) -> f32 {
    c.dot(Vec3A::new(0.2126, 0.7152, 0.0722))
}

/// One clustered virtual light — exactly the two CB rows it uploads as.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmissiveLight {
    /// Power-weighted centroid of the cluster's emissive triangles (row a
    /// xyz).
    pub pos: [f32; 3],
    /// Source radius², world units (row a w): the cluster AABB's half
    /// diagonal, floored at `(2·eps)²` — the disc denominator's softening
    /// term AND the shadow ray's stop-short margin.
    pub rc2: f32,
    /// `C/π` — radiance·area over π, RGB (row b xyz; the sun's `e_over_pi`
    /// Lambert convention, so `kd · color/(d²+rc²) · ndl` composes like the
    /// sun's own `li`).
    pub color: [f32; 3],
    /// Influence radius² (row b w): the window's exact zero. Derived from
    /// `EL_MIN_E`, floored at `4·rc2` (a light always reaches past its own
    /// body), capped at `(EL_RMAX_K · content_diag)²` — the floor wins when
    /// a giant low-budget cluster puts them in conflict.
    pub r_infl2: f32,
}

/// The scene's derived light set — pure data, the fireflies' `count == 0`
/// structural-off discipline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmissiveLights {
    pub count: u32,
    pub lights: [EmissiveLight; MAX_EMISSIVE_LIGHTS],
}

impl EmissiveLights {
    /// The structural off state (every emissive-free scene).
    pub fn off() -> EmissiveLights {
        EmissiveLights {
            count: 0,
            lights: [EmissiveLight { pos: [0.0; 3], rc2: 0.0, color: [0.0; 3], r_infl2: 0.0 };
                MAX_EMISSIVE_LIGHTS],
        }
    }
}

/// Windowed disc irradiance (over π — the Lambert convention) arriving at
/// distance² `d2` from light `l`, RGB. Exactly ZERO at and past the
/// influence radius (the window's zero is exact in fp: `d2 == r2 ⇒ x == 0`).
/// Term-for-term mirrored in shade.hlsli's emissive block.
#[inline(always)]
pub fn irradiance(l: &EmissiveLight, d2: f32) -> Vec3A {
    if d2 >= l.r_infl2 {
        return Vec3A::ZERO;
    }
    let c = Vec3A::from(l.color);
    let lc = lum(c);
    if lc <= 0.0 {
        return Vec3A::ZERO;
    }
    // The disc denominator, clamped so lum(result) ≤ EL_E_MAX however close
    // the shading point stands (window ≤ 1 only tightens it).
    let inv = (1.0 / (d2 + l.rc2)).min(EL_E_MAX / lc);
    c * (inv * crate::fireflies::window(d2, l.r_infl2))
}

/// The per-leaf-tile light cull's A/B lever: `FR_ABL=noelcull` skips the
/// cull and hands every tile the full set. The `foliage::sway_abl`
/// image-NEUTRAL idiom, deliberately NOT `shade::abl`'s "the image is
/// deliberately wrong" wording — the cull is EXACT (a rejected influence
/// sphere contains no tile hit, and the windowed falloff is exactly 0 at
/// `r_infl`), so this arm is bit-identical by construction: a pure cost
/// probe, loud on departure.
pub fn cull_abl() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        let off = std::env::var("FR_ABL").unwrap_or_default().contains("noelcull");
        if off {
            eprintln!("FR_ABL (emissive): noelcull — per-tile light cull off (bit-identical cost probe)");
        }
        off
    })
}

/// Conservative per-leaf-tile cull: a COMPACTED copy holding only the
/// lights whose influence sphere can contain a hit of this tile, plus the
/// culled count (for the `el-cull` stats segment). THREE rejection tests,
/// each independently erring toward KEEP:
/// 1. the tile-frustum plane test (`TileFrustum::sphere_outside` — apex
///    planes, no far plane, so it bounds exactly the PRIMARY hits this tier
///    serves; degenerate zero normals never cull);
/// 2. the inherited-claim near ball (hits satisfy `t >= t_start`, so a
///    light with `|c − o| + r` short of the ball cannot reach one);
/// 3. the camera FORWARD half-space — `dot(forward, c − o) < −r` culls,
///    sound because forward ⊥ right/up makes `dot(forward, d) > 0` for
///    every primary direction, so every hit lies strictly forward of the
///    apex. This arm exists because (1) alone cannot exclude the ANTIPODAL
///    cone: a narrow tile's side-plane dots against a behind-the-camera
///    point are only ≈ −dist·sin(half-angle), so the common
///    walked-past-the-lamp case would otherwise never cull.
/// EXACT by the window's exact zero: a culled light would have failed every
/// pixel's `d2 >= r_infl2` test — contributing nothing, drawing no rng,
/// bumping no counter — and kept lights stay in ascending order (the same
/// subsequence ⇒ bit-identical `direct_d` sums). CPU and GPU (leaf.hlsl's
/// `el_mask` via trace_common's `el_tile_culled`) cull independently — each
/// side only needs its own conservativeness, so no bit-parity contract
/// links the two masks.
pub fn cull_tile(
    el: &EmissiveLights,
    tf: &crate::frustum::TileFrustum,
    forward: Vec3A,
    t_start: f32,
) -> (EmissiveLights, u32) {
    let mut out = EmissiveLights::off();
    let mut culled = 0u32;
    for i in 0..el.count as usize {
        let l = &el.lights[i];
        let c = Vec3A::from(l.pos);
        let r = l.r_infl2.sqrt();
        let rel = c - tf.origin;
        // Near ball + forward half-space: strict with a safety margin in
        // the KEEP direction (a light AT a boundary is kept).
        let near_out = rel.length() + r < t_start * (1.0 - 1e-5);
        let eps = 1e-5 * (1.0 + rel.abs().max_element());
        let behind = forward.dot(rel) < -r - eps;
        if near_out || behind || tf.sphere_outside(c, r) {
            culled += 1;
            continue;
        }
        out.lights[out.count as usize] = *l;
        out.count += 1;
    }
    (out, culled)
}

/// Mean emitted radiance of triangle (i0,i1,i2) for material `mat`: the flat
/// factor × the mean of 4 emissive-map taps (3 vertex UVs + the centroid —
/// the cheap deterministic estimate of a map that varies across the surface;
/// texel-footprint integration is the documented follow-on). No map — or no
/// UV stream to tap with — falls back to the factor alone.
fn tri_radiance(
    mat: &Material,
    textures: &[Texture],
    texcoords: &[Vec2],
    idx: [u32; 3],
    n_pos: usize,
) -> Vec3A {
    // The factor is the display add's own (shade.rs: `mat.emissive × tap`) —
    // deliberately NO factor-1.0 fallback for a Ke-zero material with a map:
    // the display renders that surface BLACK (factor × tap = 0, the glTF
    // factor-times-texture rule), so deriving power for it would light the
    // scene from an invisible emitter. Loaders store the factor alongside
    // every map, so the arm is unreachable from real content either way.
    let base = mat.emissive;
    if mat.emissive_tex == NO_TEX || texcoords.len() < n_pos {
        return base;
    }
    let tex = &textures[mat.emissive_tex as usize];
    let uv0 = texcoords[idx[0] as usize];
    let uv1 = texcoords[idx[1] as usize];
    let uv2 = texcoords[idx[2] as usize];
    let uvc = (uv0 + uv1 + uv2) / 3.0;
    let tap = |uv: Vec2| tex.sample_bilinear(uv.x, uv.y);
    base * ((tap(uv0) + tap(uv1) + tap(uv2) + tap(uvc)) * 0.25)
}

/// One emissive triangle's derivation record (centroid, A·L̄ power, AABB) —
/// filled by `derive_parts`' first pass, consumed by the grid binning and
/// the som refinement alike.
struct TriRec {
    centroid: Vec3A,
    power: Vec3A,
    mn: Vec3A,
    mx: Vec3A,
}

/// One seed/merged cluster during derivation.
#[derive(Clone, Copy)]
struct Cluster {
    /// Σ A·L̄ (radiance·area, RGB) — NOT yet over π.
    power: Vec3A,
    /// Σ w·centroid and Σ w, w = lum(A·L̄) — the power-weighted position.
    cacc: Vec3A,
    wsum: f32,
    mn: Vec3A,
    mx: Vec3A,
    alive: bool,
}

impl Cluster {
    fn absorb(&mut self, o: &Cluster) {
        self.power += o.power;
        self.cacc += o.cacc;
        self.wsum += o.wsum;
        self.mn = self.mn.min(o.mn);
        self.mx = self.mx.max(o.mx);
    }
    fn centroid(&self) -> Vec3A {
        self.cacc / self.wsum.max(1e-30)
    }
}

/// Derive the scene's clustered lights. Serial, index-ordered,
/// byte-deterministic. Called from `scene::finalize_scalars`.
pub fn derive(scene: &Scene, budget: u32) -> EmissiveLights {
    derive_parts(
        &scene.positions,
        &scene.indices,
        &scene.tri_mat,
        &scene.materials,
        &scene.textures,
        &scene.texcoords,
        scene.content_min,
        scene.content_max,
        scene.eps,
        budget,
        cluster_mode(),
    )
}

/// The `--el-cluster som` arm: a power-weighted BATCH SOM refinement of the
/// merged clusters. With the neighborhood radius at 0 a batch SOM is exactly
/// weighted Lloyd's/k-means, and radius 0 is deliberate — the merged centers
/// carry no lattice topology for a neighborhood term to couple (the
/// bvh::builders `som_codes` lattice learned a space-filling CURVE; that
/// purpose does not transfer, and its M7 bake-off verdict is the codebase's
/// own evidence that the coupling buys nothing measurable here). SERIAL and
/// index-ordered like the rest of the derivation: each epoch scans `recs` in
/// tri order (nearest center by `length_squared`, ties to the lowest center
/// index — strict `<` makes the first minimum win), accumulation is
/// fixed-order f32, so the result is byte-deterministic across runs and
/// thread counts. Power conserves BY CONSTRUCTION (the final pass assigns
/// every rec exactly once); the center count can only SHRINK (an emptied
/// center drops), so the budget cap holds. Zero rng draws.
fn som_refine(recs: &[TriRec], clusters: &mut Vec<Cluster>) {
    let mut centers: Vec<Vec3A> =
        clusters.iter().filter(|c| c.alive).map(|c| c.centroid()).collect();
    if centers.is_empty() {
        return;
    }
    let nearest = |centers: &[Vec3A], p: Vec3A| -> usize {
        let mut j_min = 0usize;
        let mut d_min = f32::INFINITY;
        for (j, c) in centers.iter().enumerate() {
            let d = (*c - p).length_squared();
            if d < d_min {
                d_min = d;
                j_min = j;
            }
        }
        j_min
    };
    for _ in 0..EL_SOM_EPOCHS {
        let mut acc = vec![Vec3A::ZERO; centers.len()];
        let mut wsum = vec![0.0f32; centers.len()];
        for r in recs {
            let j = nearest(&centers, r.centroid);
            let w = lum(r.power);
            acc[j] += r.centroid * w;
            wsum[j] += w;
        }
        for j in 0..centers.len() {
            // An emptied center keeps its position — it may capture records
            // again in a later epoch as its neighbors move.
            if wsum[j] > 0.0 {
                centers[j] = acc[j] / wsum[j];
            }
        }
    }
    // Final assignment pass rebuilds the cluster records around the settled
    // centers — power sums, power-weighted centroid, AABB union: exactly the
    // fields the shared finalize loop below derives rc2/r_infl2 from, so the
    // som arm inherits the disc model and the influence band for free.
    let mut rebuilt: Vec<Cluster> = centers
        .iter()
        .map(|_| Cluster {
            power: Vec3A::ZERO,
            cacc: Vec3A::ZERO,
            wsum: 0.0,
            mn: Vec3A::splat(f32::INFINITY),
            mx: Vec3A::splat(f32::NEG_INFINITY),
            alive: true,
        })
        .collect();
    for r in recs {
        let j = nearest(&centers, r.centroid);
        let w = lum(r.power);
        let c = &mut rebuilt[j];
        c.power += r.power;
        c.cacc += r.centroid * w;
        c.wsum += w;
        c.mn = c.mn.min(r.mn);
        c.mx = c.mx.max(r.mx);
    }
    rebuilt.retain(|c| c.wsum > 0.0);
    *clusters = rebuilt;
}

/// The derivation over bare parts — what `self_test` drives with synthetic
/// slices (no `Scene` construction).
#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_parts(
    positions: &[Vec3A],
    indices: &[[u32; 3]],
    tri_mat: &[u32],
    materials: &[Material],
    textures: &[Texture],
    texcoords: &[Vec2],
    cmin: Vec3A,
    cmax: Vec3A,
    eps: f32,
    budget: u32,
    mode: ClusterMode,
) -> EmissiveLights {
    let budget = budget.min(MAX_EMISSIVE_LIGHTS as u32);
    // Material precheck — the exact display-add predicate (shade.rs's
    // emissive guard), so the O(tris) pass below only runs when some
    // triangle could emit. Emissive-free scenes (procedural, --stress,
    // powerplant) pay O(materials).
    let mat_emits: Vec<bool> = materials
        .iter()
        .map(|m| m.emissive != Vec3A::ZERO || m.emissive_tex != NO_TEX)
        .collect();
    if budget == 0 || !mat_emits.iter().any(|&e| e) {
        return EmissiveLights::off();
    }

    // Per-triangle records: centroid, A·L̄ power, AABB. One pass, index
    // order.
    let mut recs: Vec<TriRec> = Vec::new();
    for (t, idx) in indices.iter().enumerate() {
        if !mat_emits[tri_mat[t] as usize] {
            continue;
        }
        let (a, b, c) =
            (positions[idx[0] as usize], positions[idx[1] as usize], positions[idx[2] as usize]);
        let area = 0.5 * (b - a).cross(c - a).length();
        if area <= 0.0 {
            continue;
        }
        let l = tri_radiance(&materials[tri_mat[t] as usize], textures, texcoords, *idx, positions.len());
        let power = l * area;
        if lum(power) <= 0.0 {
            continue;
        }
        recs.push(TriRec {
            centroid: (a + b + c) / 3.0,
            power,
            mn: a.min(b).min(c),
            mx: a.max(b).max(c),
        });
    }
    if recs.is_empty() {
        return EmissiveLights::off();
    }

    // Seed clustering: grid-bin the centroids at EL_GRID_K × the content
    // diagonal, doubling the pitch until the cell count fits (halving
    // resolution each lap — bounded, deterministic). BTreeMap keys give a
    // TOTAL order, so the cluster list below is a pure function of the
    // records.
    let diag_c = (cmax - cmin).length().max(1e-3);
    let mut pitch = EL_GRID_K * diag_c;
    let cells = loop {
        let mut cells: BTreeMap<(i64, i64, i64), Cluster> = BTreeMap::new();
        let inv = 1.0 / pitch;
        for r in &recs {
            let k = (
                (r.centroid.x * inv).floor() as i64,
                (r.centroid.y * inv).floor() as i64,
                (r.centroid.z * inv).floor() as i64,
            );
            let w = lum(r.power);
            let e = cells.entry(k).or_insert(Cluster {
                power: Vec3A::ZERO,
                cacc: Vec3A::ZERO,
                wsum: 0.0,
                mn: Vec3A::splat(f32::INFINITY),
                mx: Vec3A::splat(f32::NEG_INFINITY),
                alive: true,
            });
            e.power += r.power;
            e.cacc += r.centroid * w;
            e.wsum += w;
            e.mn = e.mn.min(r.mn);
            e.mx = e.mx.max(r.mx);
        }
        if cells.len() <= EL_MAX_CELLS {
            break cells;
        }
        pitch *= 2.0;
    };
    let mut clusters: Vec<Cluster> = cells.into_values().collect();

    // Agglomerative merge to the budget: absorb the min-power cluster into
    // its nearest-centroid living neighbor, ties broken by lowest index (a
    // total order — the BTreeMap gave the indices themselves one). O(k²),
    // k ≤ EL_MAX_CELLS.
    let mut alive = clusters.len();
    while alive > budget as usize {
        let mut i_min = usize::MAX;
        let mut p_min = f32::INFINITY;
        for (i, c) in clusters.iter().enumerate() {
            if c.alive && lum(c.power) < p_min {
                p_min = lum(c.power);
                i_min = i;
            }
        }
        let ci = clusters[i_min];
        let cen = ci.centroid();
        let mut j_min = usize::MAX;
        let mut d_min = f32::INFINITY;
        for (j, c) in clusters.iter().enumerate() {
            if j != i_min && c.alive {
                let d = (c.centroid() - cen).length_squared();
                if d < d_min {
                    d_min = d;
                    j_min = j;
                }
            }
        }
        clusters[j_min].absorb(&ci);
        clusters[i_min].alive = false;
        alive -= 1;
    }

    // The ONE mode conditional — `Grid` runs the pre-lever instruction
    // stream verbatim (the fireflies structural-off discipline).
    if mode == ClusterMode::Som {
        som_refine(&recs, &mut clusters);
    }

    // Finalize rows, cluster order preserved (deterministic).
    let mut out = EmissiveLights::off();
    let r_cap2 = (EL_RMAX_K * diag_c) * (EL_RMAX_K * diag_c);
    for c in clusters.iter().filter(|c| c.alive) {
        let cen = c.centroid();
        let half = 0.5 * (c.mx - c.mn).length();
        let rc2 = (half * half).max((2.0 * eps) * (2.0 * eps));
        let cp = c.power * (EL_BOOST * std::f32::consts::FRAC_1_PI);
        // r_infl: where lum(C/π)/(d²+rc²) falls to EL_MIN_E — floored at
        // 4·rc2 (reach past the body), capped at the scan-cost bound. The
        // FLOOR WINS when they cross (a low-budget merge over spread
        // emitters can produce a cluster wider than half the content box,
        // where 4·rc2 > r_cap2 — a bare `clamp` PANICS there, measured on
        // bistro --emissive-lights 1): capping influence under a cluster's
        // own body would window its light off inside its own extent, and
        // the scan-cost bound was per-light — a floor-sized radius on ≤ 64
        // lights is the user's own budget choice.
        let lo = 4.0 * rc2;
        let r_infl2 = (lum(cp) / EL_MIN_E - rc2).clamp(lo, r_cap2.max(lo));
        out.lights[out.count as usize] = EmissiveLight {
            pos: [cen.x, cen.y, cen.z],
            rc2,
            color: [cp.x, cp.y, cp.z],
            r_infl2,
        };
        out.count += 1;
    }
    out
}

/// Closed-form gates on everything the consumers lean on. Pure, DLL-free,
/// deterministic — run by `--check` next to `fireflies::self_test`.
pub fn self_test() -> Result<(), String> {
    use glam::Vec2;
    let mat_dark = || Material {
        albedo: Vec3A::splat(0.5),
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
        normal_tex: NO_TEX,
        normal_scale: 1.0,
        height_amp: 0.0,
        rough_tex: NO_TEX,
        metal_tex: NO_TEX,
        emissive_tex: NO_TEX,
        class: 0,
        kind: crate::scene::MatKind::Diffuse,
    };
    let mat_emit = |e: Vec3A| Material { emissive: e, ..mat_dark() };

    // A unit right triangle at `at` (area 0.5) with material `m`.
    let push_tri = |pos: &mut Vec<Vec3A>, idx: &mut Vec<[u32; 3]>, tm: &mut Vec<u32>, at: Vec3A, m: u32| {
        let b = pos.len() as u32;
        pos.push(at);
        pos.push(at + Vec3A::X);
        pos.push(at + Vec3A::Z);
        idx.push([b, b + 1, b + 2]);
        tm.push(m);
    };
    let cmin = Vec3A::splat(-16.0);
    let cmax = Vec3A::splat(16.0);
    let eps = 1e-3;

    // 1. Structural off: an emissive-free mesh and a zero budget both derive
    //    count 0 — the arm every procedural/--stress session takes.
    {
        let (mut pos, mut idx, mut tm) = (Vec::new(), Vec::new(), Vec::new());
        push_tri(&mut pos, &mut idx, &mut tm, Vec3A::ZERO, 0);
        let mats = [mat_dark()];
        let el = derive_parts(&pos, &idx, &tm, &mats, &[], &[], cmin, cmax, eps, 32, ClusterMode::Grid);
        if el.count != 0 {
            return Err(format!("emissive-free scene derived {} lights", el.count));
        }
        let mats2 = [mat_emit(Vec3A::ONE)];
        let el2 = derive_parts(&pos, &idx, &tm, &mats2, &[], &[], cmin, cmax, eps, 0, ClusterMode::Grid);
        if el2.count != 0 {
            return Err("zero budget did not derive count 0".into());
        }
    }

    // 2. Determinism + power conservation + placement: two far-apart
    //    emitters cluster separately, bit-identical across derivations, and
    //    Σ color·π over the lights equals EL_BOOST · Σ A·L̄ over the
    //    triangles exactly (fixed-order sums — the same adds in the same
    //    order; the boost lands once, at the C_c fill).
    {
        let (mut pos, mut idx, mut tm) = (Vec::new(), Vec::new(), Vec::new());
        push_tri(&mut pos, &mut idx, &mut tm, Vec3A::new(-8.0, 1.0, 0.0), 0);
        push_tri(&mut pos, &mut idx, &mut tm, Vec3A::new(8.0, 1.0, 0.0), 1);
        let mats = [mat_emit(Vec3A::new(4.0, 2.0, 1.0)), mat_emit(Vec3A::splat(6.0))];
        let a = derive_parts(&pos, &idx, &tm, &mats, &[], &[], cmin, cmax, eps, 32, ClusterMode::Grid);
        let b = derive_parts(&pos, &idx, &tm, &mats, &[], &[], cmin, cmax, eps, 32, ClusterMode::Grid);
        if a != b {
            return Err("derive is not deterministic".into());
        }
        if a.count != 2 {
            return Err(format!("expected 2 clusters, got {}", a.count));
        }
        let total_tris = (mats[0].emissive * 0.5 + mats[1].emissive * 0.5) * EL_BOOST;
        let total_lights: Vec3A = (0..a.count as usize)
            .map(|i| Vec3A::from(a.lights[i].color) * std::f32::consts::PI)
            .fold(Vec3A::ZERO, |s, v| s + v);
        if (total_lights - total_tris).abs().max_element() > 1e-4 * total_tris.max_element() {
            return Err(format!(
                "power not conserved: lights {total_lights:?} vs tris {total_tris:?}"
            ));
        }
        // Each cluster sits at its own triangle's centroid (single-tri
        // clusters — the power-weighted centroid IS the centroid).
        for i in 0..2 {
            let p = Vec3A::from(a.lights[i].pos);
            let c0 = (pos[i * 3] + pos[i * 3 + 1] + pos[i * 3 + 2]) / 3.0;
            let c1 = (pos[(1 - i) * 3] + pos[(1 - i) * 3 + 1] + pos[(1 - i) * 3 + 2]) / 3.0;
            if (p - c0).length().min((p - c1).length()) > 1e-3 {
                return Err(format!("cluster {i} not at a triangle centroid: {p:?}"));
            }
        }
    }

    // 3. Budget cap: 200 spread emitters merge to exactly the budget, and
    //    total power still conserves through the merges.
    {
        let (mut pos, mut idx, mut tm) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..200u32 {
            // A deterministic scatter over the content box (integer hash —
            // no rng anywhere in this module).
            let h = crate::sky::pcg_mix(i.wrapping_mul(0x9E37_79B9));
            let x = (h & 0xFFFF) as f32 / 65535.0 * 28.0 - 14.0;
            let z = ((h >> 16) & 0xFFFF) as f32 / 65535.0 * 28.0 - 14.0;
            push_tri(&mut pos, &mut idx, &mut tm, Vec3A::new(x, 1.0, z), 0);
        }
        let mats = [mat_emit(Vec3A::splat(2.0))];
        let el = derive_parts(&pos, &idx, &tm, &mats, &[], &[], cmin, cmax, eps, 16, ClusterMode::Grid);
        if el.count != 16 {
            return Err(format!("budget 16 derived {} lights", el.count));
        }
        let total: f32 =
            (0..16).map(|i| lum(Vec3A::from(el.lights[i].color)) * std::f32::consts::PI).sum();
        let expect = 200.0 * 0.5 * lum(Vec3A::splat(2.0)) * EL_BOOST;
        if (total - expect).abs() > 1e-3 * expect {
            return Err(format!("merge lost power: {total} vs {expect}"));
        }
        for i in 0..el.count as usize {
            let l = &el.lights[i];
            if !(l.r_infl2 >= 4.0 * l.rc2 && l.r_infl2 <= (EL_RMAX_K * 32.0 * 1.8) * (EL_RMAX_K * 32.0 * 1.8)) {
                return Err(format!("light {i} influence out of band: {}", l.r_infl2));
            }
        }
        // Budget 1 merges the whole spread field into ONE cluster wider than
        // half the content box, where the 4·rc2 floor exceeds the r_cap2
        // cap — the arm that PANICKED as a bare `clamp` (min > max; measured
        // live on bistro --emissive-lights 1). The floor must win.
        let el1 = derive_parts(&pos, &idx, &tm, &mats, &[], &[], cmin, cmax, eps, 1, ClusterMode::Grid);
        if el1.count != 1 {
            return Err(format!("budget 1 derived {} lights", el1.count));
        }
        if el1.lights[0].r_infl2 < 4.0 * el1.lights[0].rc2 {
            return Err("giant-cluster influence fell under the 4·rc2 floor".into());
        }
    }

    // 4. Falloff: exactly 0 at/past the influence radius, monotone inside
    //    (past the near-clamp plateau), lum ≤ EL_E_MAX everywhere.
    {
        let l = EmissiveLight {
            pos: [0.0; 3],
            rc2: 0.01,
            color: [3.0, 2.0, 1.0],
            r_infl2: 25.0,
        };
        if irradiance(&l, l.r_infl2) != Vec3A::ZERO || irradiance(&l, l.r_infl2 * 2.0) != Vec3A::ZERO
        {
            return Err("irradiance not 0 at/past the influence radius".into());
        }
        let mut prev = f32::INFINITY;
        for k in 1..=64 {
            let d2 = (k as f32 / 64.0) * l.r_infl2;
            let e = lum(irradiance(&l, d2));
            if e > prev {
                return Err(format!("irradiance not monotone at d2 {d2}"));
            }
            if e > EL_E_MAX {
                return Err(format!("irradiance {e} above EL_E_MAX at d2 {d2}"));
            }
            prev = e;
        }
        if lum(irradiance(&l, 0.0)) > EL_E_MAX {
            return Err("near-field clamp not applied at d2 = 0".into());
        }
    }

    // 5. The 4-tap map estimate: a CONSTANT emissive map scales the factor
    //    by exactly its texel value (255 decodes to exactly 1.0 through the
    //    sRGB LUT, so factor × 1.0 is bit-preserving), and a black map
    //    yields zero lights.
    {
        let img = image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(2, 2, vec![255u8; 16]).unwrap(),
        );
        let tex = Texture::from_image(img, true);
        let (mut pos, mut idx, mut tm) = (Vec::new(), Vec::new(), Vec::new());
        push_tri(&mut pos, &mut idx, &mut tm, Vec3A::new(0.0, 1.0, 0.0), 0);
        let uvs = vec![Vec2::new(0.25, 0.25); pos.len()];
        let mut m = mat_emit(Vec3A::splat(2.0));
        m.emissive_tex = 0;
        let mats = [m];
        let texs = [tex];
        let el = derive_parts(&pos, &idx, &tm, &mats, &texs, &uvs, cmin, cmax, eps, 32, ClusterMode::Grid);
        if el.count != 1 {
            return Err(format!("mapped emitter derived {} lights", el.count));
        }
        let want = 2.0 * 0.5 * std::f32::consts::FRAC_1_PI * EL_BOOST; // factor·area·boost/π per channel
        let got = Vec3A::from(el.lights[0].color);
        if (got - Vec3A::splat(want)).abs().max_element() > 1e-5 {
            return Err(format!("white-map power {got:?} != {want}"));
        }
        let black = image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(2, 2, vec![0u8, 0, 0, 255].repeat(4)).unwrap(),
        );
        let texs_b = [Texture::from_image(black, true)];
        let el_b = derive_parts(&pos, &idx, &tm, &mats, &texs_b, &uvs, cmin, cmax, eps, 32, ClusterMode::Grid);
        if el_b.count != 0 {
            return Err("black emissive map still derived a light".into());
        }
        // Display parity: a Ke-ZERO material with a (white) map renders
        // BLACK (factor × tap — shade.rs's display add), so it must derive
        // zero lights too; an invisible emitter must not light the scene.
        let mut m0 = mat_emit(Vec3A::ZERO);
        m0.emissive_tex = 0;
        let mats0 = [m0];
        let el_0 = derive_parts(&pos, &idx, &tm, &mats0, &texs, &uvs, cmin, cmax, eps, 32, ClusterMode::Grid);
        if el_0.count != 0 {
            return Err("Ke-zero mapped material derived a light (display renders it black)".into());
        }
    }

    // 6. The parse-time levers round-trip through the clamp.
    {
        let (mut pos, mut idx, mut tm) = (Vec::new(), Vec::new(), Vec::new());
        push_tri(&mut pos, &mut idx, &mut tm, Vec3A::ZERO, 0);
        let mats = [mat_emit(Vec3A::ONE)];
        let el = derive_parts(&pos, &idx, &tm, &mats, &[], &[], cmin, cmax, eps, 999, ClusterMode::Grid);
        if el.count != 1 {
            return Err("over-cap budget mishandled".into());
        }
    }

    // 7. The per-leaf-tile cull (cull_tile + TileFrustum::sphere_outside),
    //    against the REAL camera construction the renderer culls with — a
    //    center tile of a 1920x1080 basis. Six lights exercising every
    //    rejection arm, then the CONSERVATIVENESS pin: at every sample
    //    point a tile ray can shade (grid of pixels × t >= t_start), every
    //    CULLED light's irradiance must be exactly ZERO — the property that
    //    makes the cull bit-identical rather than approximate.
    {
        let cam = crate::camera::Camera::look_at(
            Vec3A::ZERO,
            Vec3A::new(0.0, 0.0, 10.0),
            60f32.to_radians(),
        );
        let basis = cam.basis(1920, 1080);
        let (x0, y0, x1, y1) = (944, 532, 976, 564); // 32-px center tile
        let tf = basis.tile_frustum(x0, y0, x1, y1);
        let mk = |p: [f32; 3], r_infl2: f32| EmissiveLight {
            pos: p,
            rc2: 1e-4,
            color: [1.0, 1.0, 1.0],
            r_infl2,
        };
        let mut el = EmissiveLights::off();
        // 0: on-axis in front — KEPT. 1: far off-axis — plane-culled.
        // 2: behind the camera — FORWARD-culled (the side planes provably
        // cannot: their dots are only ≈ −dist·sin(half-angle), the
        // antipodal-cone hole the third test exists for). 3: on-axis but
        // inside the inherited near ball (t_start 5) — near-ball-culled.
        // 4: center just outside the narrow tile frustum, sphere straddling
        // back in — must be KEPT (err-toward-keep). 5: on-axis far — KEPT
        // (order).
        let lights = [
            mk([0.0, 0.0, 10.0], 1.0),
            mk([100.0, 0.0, 10.0], 1.0),
            mk([0.0, 0.0, -10.0], 1.0),
            mk([0.0, 0.0, 1.0], 0.25),
            mk([0.5, 0.0, 10.0], 4.0),
            mk([0.0, 0.0, 40.0], 1.0),
        ];
        for (i, l) in lights.iter().enumerate() {
            el.lights[i] = *l;
            el.count = (i + 1) as u32;
        }
        let fwd = basis.forward();
        let t_start = 5.0f32;
        let (kept, culled) = cull_tile(&el, &tf, fwd, t_start);
        if culled == 0 || kept.count == 0 {
            return Err(format!(
                "cull anti-vacuity: kept {} culled {culled} — the layout must exercise both",
                kept.count
            ));
        }
        // The known verdicts: 0/4/5 kept in ORDER (the fp-subsequence
        // contract), 1/2/3 culled.
        let want = [lights[0], lights[4], lights[5]];
        if kept.count != 3
            || (0..3).any(|i| kept.lights[i] != want[i])
        {
            return Err(format!("cull verdicts wrong: kept {} of 6", kept.count));
        }
        // Conservativeness: every culled light invisible from every point a
        // tile ray can shade. (Culled = in `el` but not in `kept`.)
        for l in [&lights[1], &lights[2], &lights[3]] {
            for py in (y0..y1).step_by(8) {
                for px in (x0..x1).step_by(8) {
                    for t in [t_start, 6.0, 10.0, 50.0, 400.0] {
                        let p = basis.ray_dir(px as f32 + 0.5, py as f32 + 0.5) * t;
                        let d2 = (p - Vec3A::from(l.pos)).length_squared();
                        if irradiance(l, d2) != Vec3A::ZERO {
                            return Err(format!(
                                "over-cull: light at {:?} reaches a tile point at t {t}",
                                l.pos
                            ));
                        }
                    }
                }
            }
        }
        // t_start 0 disarms the near ball: light 3 comes back (light 2 stays
        // forward-culled).
        let (kept0, _) = cull_tile(&el, &tf, fwd, 0.0);
        if kept0.count != 4 {
            return Err(format!("near-ball arm: t_start 0 kept {} (want 4)", kept0.count));
        }
        // A zero-area tile degenerates every plane to a zero normal, which
        // never culls (the frustum.rs contract) — only the plane-independent
        // forward arm still fires, so exactly the behind light is culled.
        let tf0 = basis.tile_frustum(x0, y0, x0, y0);
        let (kept_d, culled_d) = cull_tile(&el, &tf0, fwd, 0.0);
        if kept_d.count != 5 || culled_d != 1 {
            return Err(format!(
                "degenerate frustum must keep all but the behind light: kept {} culled {culled_d}",
                kept_d.count
            ));
        }
        // Structural off in = structural off out.
        let (kept_e, culled_e) = cull_tile(&EmissiveLights::off(), &tf, fwd, t_start);
        if kept_e.count != 0 || culled_e != 0 {
            return Err("empty set culled something".into());
        }
    }

    // 8. The --el-cluster som arm (som_refine — power-weighted batch Lloyd,
    //    a radius-0 batch SOM): deterministic, power-conserving through the
    //    refinement (× EL_BOOST at the shared fill), budget-capped, and the
    //    shared finalize keeps the influence band. Gates 1-7 all pass
    //    ClusterMode::Grid explicitly, so they ARE the grid-arm-unmoved pin
    //    (one dispatch point, a guarded branch). Runs unconditionally —
    //    pure math, no lever needed (the self_test pattern).
    {
        // Gate 3's 200-tri deterministic scatter, rebuilt — enough spread
        // emitters that the merge to budget 16 leaves the Lloyd pass real
        // work (multi-tri clusters whose centroids genuinely move).
        let (mut pos, mut idx, mut tm) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..200u32 {
            let h = crate::sky::pcg_mix(i.wrapping_mul(0x9E37_79B9));
            let x = (h & 0xFFFF) as f32 / 65535.0 * 28.0 - 14.0;
            let z = ((h >> 16) & 0xFFFF) as f32 / 65535.0 * 28.0 - 14.0;
            push_tri(&mut pos, &mut idx, &mut tm, Vec3A::new(x, 1.0, z), 0);
        }
        let mats = [mat_emit(Vec3A::splat(2.0))];
        let a = derive_parts(&pos, &idx, &tm, &mats, &[], &[], cmin, cmax, eps, 16, ClusterMode::Som);
        let b = derive_parts(&pos, &idx, &tm, &mats, &[], &[], cmin, cmax, eps, 16, ClusterMode::Som);
        if a != b {
            return Err("som derive is not deterministic".into());
        }
        if a.count == 0 || a.count > 16 {
            return Err(format!("som budget 16 derived {} lights", a.count));
        }
        let total: f32 = (0..a.count as usize)
            .map(|i| lum(Vec3A::from(a.lights[i].color)) * std::f32::consts::PI)
            .sum();
        let expect = 200.0 * 0.5 * lum(Vec3A::splat(2.0)) * EL_BOOST;
        if (total - expect).abs() > 1e-3 * expect {
            return Err(format!("som refinement lost power: {total} vs {expect}"));
        }
        for i in 0..a.count as usize {
            let l = &a.lights[i];
            if !(l.r_infl2 >= 4.0 * l.rc2
                && l.r_infl2 <= (EL_RMAX_K * 32.0 * 1.8) * (EL_RMAX_K * 32.0 * 1.8))
            {
                return Err(format!("som light {i} influence out of band: {}", l.r_infl2));
            }
        }
        // Anti-vacuity: the refinement must actually MOVE something on this
        // geometry, or the gate proves only that the arm compiled — the
        // grid centroids are not Lloyd-stationary here by construction.
        let g = derive_parts(&pos, &idx, &tm, &mats, &[], &[], cmin, cmax, eps, 16, ClusterMode::Grid);
        if a == g {
            return Err("som arm returned the grid clustering bitwise — the refinement never ran".into());
        }
    }

    eprintln!("emissive self-test: OK");
    Ok(())
}
