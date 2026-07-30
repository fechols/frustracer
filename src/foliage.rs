//! Wind-swayed foliage — the "leaf sway" prototype of the tetrahedral-cage
//! epic (docs/design/animated-foliage.md; the enabling prior work is Gruen,
//! Benthin, Kern & McAllister, *Ray Tracing Massive Amounts of Animated
//! Geometry*, https://doi.org/10.1145/3820014 — static per-chunk BLASes,
//! per-frame instance transforms, per-frame TLAS rebuild). v0.2: ALL THREE
//! render modes consume the motion. v0.3: the CPU cost fix — GATEWAY
//! subtrees (see the swept-box bullet below).
//!
//! The prototype deliberately cuts the paper down to the smallest SOUND
//! surface:
//!
//! - **Leaves only** (a material is a "leaf" iff its retained matclass
//!   verdict is foliage (`Material::class` — the byte carries the NAME half
//!   of the vocabulary, which is all the Minecraft atlas scenes have) AND its
//!   albedo texture is alpha-masked — `leaf_materials`).
//!   Trunks/bark stay static, so disconnected cutout leaves are the only
//!   moving geometry and there is nothing to tear: the paper's clipping +
//!   shared-cage-vertex machinery (its watertightness bill) is not needed.
//! - **Translation-only, per spatial CELL** (no tets yet): `cell_partition`
//!   buckets leaf triangles by centroid into a grid over the content box —
//!   ONE partition (`Scene::sway`, derived at load, never serialized) shared
//!   by every consumer, which is the correctness spine: the CPU intersector
//!   displaces per `tri_cell`, the BVH build sweeps per `sway_pad`, and the
//!   GPU split (`split_plan`) turns each cell into a BLAS chunk + animated
//!   TLAS instance. A translation leaves normals, tangents, UVs and
//!   barycentrics untouched, and every shading path reconstructs the hit
//!   point as `o + t·d` — so the shader/shading surface is ZERO. The cost is
//!   per-cell-rigid motion (leaves in one cell sway together); the paper's
//!   per-tet affine is the follow-on.
//! - **All three arms consume the motion (v0.2), through GATEWAY subtrees
//!   (v0.3 — the design doc's "pad at cell granularity, never
//!   per-triangle")**: on the default SAH builder (`gateway_mode()`) each
//!   cell's triangles build as a rest-space-TIGHT subtree behind ONE
//!   gateway node whose box alone is swept by the displacement bound
//!   (`bvh::GATEWAY_BIT` — a truthful fat leaf to every bound/cut consumer,
//!   subtree implicitly at +1), so every frustum bound, temporal claim,
//!   structure-replay record and hemi query stays a conservative lower
//!   bound for EVERY pose while the pad lives on <= MAX_CELLS boxes instead
//!   of millions. The CPU shifts the ray into cell-rest space ONCE at the
//!   gateway entry (`Bvh::gateway_offset` in the three traversal arms — t
//!   preserved, `o + t·d` lands on the displaced surface, every ray type
//!   agrees by construction); the wavefront + DXR pipelines bind the
//!   animated-TLAS ring (`FrameParams::sway_time`) built BESIDE the
//!   pristine static TLAS. The v0.2 per-tri sweep + per-test
//!   `moller_trumbore` shift SURVIVES as the alt-builder (lbvh/ploc/som)
//!   fallback — the measured reason for v0.3 was that per-tri regime:
//!   canopy triangle tests +107%, ~80% of a +27% world-canopy CPU bill,
//!   where the gateway tree re-measures at +1-3% with FEWER tests than the
//!   unswept tree (cells are good SAH clusters). Headless gates/benches
//!   (`--check*`, `--spin`) stay at the rest pose — offsets zero, sway_time
//!   None — except the sway gates themselves; `foliage::self_test`'s
//!   gateway audit + synthetic cherry/root-gateway scenes and the
//!   displaced-hit pin gate the structure every `--check`.
//!
//! Motion is the fireflies template verbatim: the static curl field
//! (`clouds::curl_offset` — soft-normalized |v| < 1, so the amplitude
//! constant is an EXACT displacement bound), sampled at a time-shifted
//! lookup point on the shared `cloud_time` clock, plus a small hashed-phase
//! flutter keyed by the PARTITION cell index (the v0.2 re-key: v0 hashed the
//! BLAS run index, so cap-overflow runs of one cell fluttered apart and the
//! CPU/GPU poses could not agree). ZERO rng draws anywhere — poses are pure
//! functions of (cell, time), so every same-seed / replay contract holds
//! structurally. Known-accepts (the design doc's list): no MVs on sway
//! (bounded upscaler ghosting), a converging still freezes mid-gust,
//! `--sw-rays` renders the rest pose (the HLSL software rays read rest
//! positions), incompatible with `--heightfield` relief on leaf materials
//! (the relief re-march reads rest-space fields; both sweeps would compound
//! on one box).

use crate::scene::{MatKind, Scene};
use glam::Vec3A;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Session lever — DEFAULT ON (`--no-foliage-sway` kills; `--foliage-sway`
/// spells the default), the `blas_split::set_max_prims` pattern: set once
/// from the CLI before any GPU tracer is built. Default-on is safe because a
/// scene with no foliage-classed materials is structurally untouched
/// (`split_plan` returns None) and every headless gate/benchmark pins
/// `sway_time: None`.
static ARMED: AtomicBool = AtomicBool::new(true);
pub fn set_armed(on: bool) {
    ARMED.store(on, Ordering::Relaxed);
}
pub fn armed() -> bool {
    ARMED.load(Ordering::Relaxed)
}

/// `--foliage-amp` — a user taste multiplier on BOTH motion halves (curl sway
/// and flutter), default 1.0 (f32 bits in an atomic, the `set_aniso` knob
/// idiom; set once from the lever block). Applied at BAKE time only
/// (`translation`), never at split time — `SwayCell::amp` stays pure
/// geometry, so the cell partition and the startup line are knob-independent
/// and `displacement_bound` folds the mult in symmetrically.
static AMP_MULT: AtomicU32 = AtomicU32::new(f32::to_bits(1.0));
pub fn set_amp_mult(m: f32) {
    AMP_MULT.store(m.to_bits(), Ordering::Relaxed);
}
pub fn amp_mult() -> f32 {
    f32::from_bits(AMP_MULT.load(Ordering::Relaxed))
}

/// The attach predicate (`Scene::sway` exists iff this holds): the session
/// lever AND blas-split armed. Under `--no-blas-split` the GPU structurally
/// cannot animate (no per-cell instances exist), so the CPU must not either —
/// one SPACE-cycle session would otherwise render two poses; main's lever
/// block already prints the "sway is IDLE" note for that combination. Also
/// the `scene_cache::lever_word` bit — the sweep changes the BUILT tree.
pub fn sweep_armed() -> bool {
    armed() && crate::blas_split::max_prims().is_some()
}

/// The BVH-sweep multiplier: covers every pose up to the session's
/// `--foliage-amp`, floored at 1.0 so amp <= 1 sessions all share the default
/// sidecar (a pad for amp 1 contains every smaller amp's poses); the CLI
/// clamps amp to <= 8, so the pad is bounded. Keys the scene cache
/// (`scene_cache`'s sway word) — an amp > 1 session is one cold rebuild.
pub fn sweep_mult() -> f32 {
    amp_mult().max(1.0)
}

/// Gateway mode: the SAH build hosts sway cells as GATEWAY subtrees (the
/// design doc's "pad at cell granularity, never per-triangle" — profiling
/// measured the per-tri sweep at ~80% of the CPU sway bill, tests +107% on
/// a world canopy) and `moller_trumbore`'s per-test shift is OFF (the
/// gateway arm owns the one shift per cell entry). The alt builders
/// (lbvh/ploc/som) keep the v0.2 per-tri-sweep + per-test-shift path — the
/// bake-off levers stay honest and the old path doubles as a built-in A/B.
/// LIVE relaxed reads, deliberately not a OnceLock: both levers are set by
/// the CLI lever block after process start, and a OnceLock touched earlier
/// (e.g. by a self-test) would freeze pre-CLI state.
pub fn gateway_mode() -> bool {
    sweep_armed() && crate::bvh::builder() == crate::bvh::Builder::Sah
}

/// Read-only cost-probe ablations (the `shade::abl` / `FR_ABL` idiom — env
/// levers, loud on departure, one already-initialized deref when unset). They
/// exist to DECOMPOSE the CPU sway bill, whose two halves no counter can
/// separate (triangle tests are per-`moller_trumbore`, below every counter):
/// the swept-tree cost (fatter leaf boxes -> more tests) vs the intersector
/// arm's cost (the per-test `tri_cell` load + shift). Neither is a session
/// lever — flipping `armed()` instead would also move the BVH sweep and the
/// sidecar `lever_word`, conflating exactly the two things these separate.
pub struct SwayAbl {
    /// `noshift`: `moller_trumbore` skips the whole sway head (no `tri_cell`
    /// load, no origin shift) — the pre-sway intersector running on the
    /// swept tree. Renders the REST pose regardless of the bake.
    pub noshift: bool,
    /// `rest`: `bake()` early-returns, so offsets stay all-zero — the armed
    /// intersector's zero-offset fast path in an otherwise animated session.
    pub rest: bool,
}

pub fn sway_abl() -> &'static SwayAbl {
    static A: std::sync::OnceLock<SwayAbl> = std::sync::OnceLock::new();
    A.get_or_init(|| {
        let v = std::env::var("FR_SWAY_ABL").unwrap_or_default();
        let a = SwayAbl { noshift: v.contains("noshift"), rest: v.contains("rest") };
        // Loud on departure from the default — a silent ablation is how a
        // measurement gets attributed to the wrong thing.
        if a.noshift || a.rest {
            eprintln!(
                "FR_SWAY_ABL (cpu sway): noshift={} rest={} — THE IMAGE RENDERS THE \
                 WRONG POSE, this is a cost probe",
                a.noshift, a.rest
            );
        }
        a
    })
}

/// Sway displacement amplitude, in CONTENT diagonals (`Scene::content_min/max`
/// — the fireflies' scale rule: `Scene::diag` is ground-quad-inflated ~17× on
/// procedural scenes). The curl field's soft |v| < 1 normalization makes
/// `SWAY_AMP_K · scale · height_factor` an EXACT bound on the curl half of a
/// cell's displacement. Retuned 2026-07-28: the v0 0.010 (~25 cm of rigid
/// per-cell translation on San Miguel) read as an earthquake — the user's
/// verdict was "~100× too much"; 0.0003 ≈ 1 cm there, and `--foliage-amp`
/// is the taste dial.
pub const SWAY_AMP_K: f32 = 0.0003;
/// Per-axis flutter amplitude (three hashed sines per cell — the leaf-scale
/// shimmer the low-frequency curl field is too smooth to provide; per-cell
/// decorrelation is deliberate, leaves flutter independently). Retuned with
/// SWAY_AMP_K (v0's 0.0015 was ~4 cm of high-frequency jitter).
pub const SWAY_BOB_K: f32 = 0.00005;
/// How fast a cell's curl lookup point travels, in scales/second (the clouds
/// advect precedent: the field is static, the SAMPLE point moves — gusts
/// sweep across the canopy).
pub const SWAY_WIND_K: f32 = 0.03;
/// The curl field's wavelength knob: the synthetic `Clouds` handed to
/// `curl_offset` gets `diag = SWAY_FIELD_K · scale`, so the field's spatial
/// wavelength is `~6.5 · SWAY_FIELD_K · scale` (clouds::CLOUD_CURL_SCALE_K) —
/// ~0.2 content diagonals: neighboring trees sway differently, one tree sways
/// coherently. Deliberately NOT a per-cell hash offset (the fireflies
/// decorrelator): adjacent cells of one canopy must move together or the
/// cell seams read as tearing.
pub const SWAY_FIELD_K: f32 = 0.03;
/// Leaf-cell grid pitch, in content diagonals (doubled until the cell count
/// fits `MAX_CELLS` — every doubling halves per-tree motion resolution, never
/// correctness).
pub const SWAY_CELL_K: f32 = 0.03;
/// Cell-count cap: bounds the per-frame instance rewrite + TLAS rebuild and
/// the one-time per-cell BLAS builds. ~2k instances is deep inside the
/// paper's measured band (2.8M tets -> 9.66 ms; this is three orders less).
pub const MAX_CELLS: usize = 2048;
/// Height band over which sway fades in from the ground: `SWAY_GROUND_K` at
/// the content floor, full above `SWAY_HEIGHT_BAND` of the content height —
/// grass stirs, canopy sways.
pub const SWAY_HEIGHT_BAND: f32 = 0.3;
/// Ground floor of the height fade (2026-07-29 — was 0.0, "grass barely
/// stirs"; the user asked for visible grass sway in the Minecraft worlds).
/// A ground-level billboard is translated RIGIDLY, so a nonzero floor
/// accepts two texel-scale artifacts by design: the base slides laterally
/// (~30% of the canopy amplitude — the in-game Minecraft look, where the
/// whole cross wiggles) and the curl's vertical component can sink/lift it
/// by the same order (mm-to-cm at scene scale; a cross planted ON a block
/// hides it). Soundness is untouched by construction: the factor stays in
/// [SWAY_GROUND_K, 1], per-cell amp still feeds the ONE
/// `displacement_bound_with`, so sweep pads / gateway boxes / audits grow in
/// lockstep automatically.
pub const SWAY_GROUND_K: f32 = 0.3;

/// One sway cell: a leaf-triangle bucket that becomes one BLAS chunk + one
/// animated TLAS instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwayCell {
    /// Cell-center anchor the motion is sampled at (world space, rest pose).
    pub center: Vec3A,
    /// This cell's full displacement amplitude for the CURL half, in world
    /// units (`SWAY_AMP_K · scale · height_factor(center.y)`).
    pub amp: f32,
}

/// `tri_cell` sentinel: not a leaf triangle.
pub const STATIC_CELL: u16 = u16::MAX;

/// The scene-attached half of the feature (`Scene::sway` — derived at load,
/// NEVER serialized, the sky_sh precedent): ONE cell partition shared by
/// every consumer — the CPU intersector's per-triangle lookup
/// (`bvh::moller_trumbore`), the BVH build sweep (`bvh::grow_sway_sweep`),
/// and the GPU BLAS split (`split_plan`) — plus the per-frame offsets bake.
/// The single partition is the correctness spine: CPU-displaced geometry,
/// swept claim boxes, and GPU instance translations must all describe the
/// SAME motion, and they do because they all read these cells.
pub struct SceneSway {
    /// One entry per partition cell, in sorted-grid-key order (bit-equal to
    /// the v0 BTreeMap chunk-tail order, so the GPU split's cell sequence —
    /// and the 294-cell San Miguel pin — are unchanged).
    pub cells: Vec<SwayCell>,
    /// Scene-triangle-id -> partition cell; `STATIC_CELL` = static. u16 is
    /// ample: cells cap at `MAX_CELLS` = 2048 (BLAS cap-overflow RUNS are
    /// unbounded, but runs re-key onto these cells — `SwaySplit::cell_of`).
    pub tri_cell: Vec<u16>,
    /// The content diagonal every length constant multiplies.
    pub scale: f32,
    /// The grid pitch the partition settled on — the startup line's number.
    pub cell: f32,
    /// Per-cell world translation for THIS frame, as f32 bits in RELAXED
    /// atomics (x, y, z). ALL-ZERO = the rest pose, the state every headless
    /// path stays in (nothing bakes there). Interior-mutable because the
    /// session loop shadows `&mut Scene` down to `&Scene` for the frame body
    /// (main.rs's read-only-iteration contract) and the bake runs inside it;
    /// race-free by the accum-buffer discipline — written ONLY between
    /// traces on the main thread, read by the rayon workers during one.
    offsets: Vec<[AtomicU32; 3]>,
    /// The clock `offsets` was baked at (f32 bits; NAN = rest) — `bake`'s
    /// bit-equal fast path, the SwayGpu baked-slot shape.
    baked: AtomicU32,
}

impl SceneSway {
    /// This frame's translation for partition cell `c` — the intersector's
    /// read (3 relaxed loads; plain loads on x86).
    #[inline]
    pub fn offset(&self, c: u16) -> Vec3A {
        let o = &self.offsets[c as usize];
        Vec3A::new(
            f32::from_bits(o[0].load(Ordering::Relaxed)),
            f32::from_bits(o[1].load(Ordering::Relaxed)),
            f32::from_bits(o[2].load(Ordering::Relaxed)),
        )
    }

    /// Every cell's current offset (gate/test convenience — never hot).
    pub fn offsets_snapshot(&self) -> Vec<Vec3A> {
        (0..self.cells.len()).map(|c| self.offset(c as u16)).collect()
    }
}

/// Compute the partition over SCENE triangle order. `None` when the mask
/// marks nothing. Deterministic, and bit-equal to the v0 packed-order
/// derivation: the distinct-key set is order-free (sorted + deduped), the
/// coarsening loop keys on distinct-count only, and cells follow sorted-key
/// order — `BlasPlan::packed_tris` was a permutation of this same set.
pub fn cell_partition(scene: &Scene, leaf_mat: &[bool]) -> Option<SceneSway> {
    let n = scene.indices.len();
    let is_leaf = |t: usize| leaf_mat.get(scene.tri_mat[t] as usize).copied().unwrap_or(false);
    let leaf_tris: Vec<u32> = (0..n as u32).filter(|&t| is_leaf(t as usize)).collect();
    if leaf_tris.is_empty() {
        return None;
    }
    let cmin = scene.content_min;
    let cmax = scene.content_max;
    let scale = (cmax - cmin).length().max(1e-3);
    let centroid = |t: u32| -> Vec3A {
        let [a, b, c] = scene.indices[t as usize];
        (scene.positions[a as usize] + scene.positions[b as usize] + scene.positions[c as usize])
            * (1.0 / 3.0)
    };
    let key_at = |p: Vec3A, cell: f32| -> (i32, i32, i32) {
        let q = (p - cmin) * (1.0 / cell);
        (q.x.floor() as i32, q.y.floor() as i32, q.z.floor() as i32)
    };
    // Grid pitch: start at SWAY_CELL_K and double until the distinct-cell
    // count fits the cap (coarser = fewer, bigger cells — never wrong).
    let mut cell = (SWAY_CELL_K * scale).max(1e-6);
    let keys: Vec<(i32, i32, i32)> = loop {
        let mut keys: Vec<(i32, i32, i32)> =
            leaf_tris.iter().map(|&t| key_at(centroid(t), cell)).collect();
        keys.sort_unstable();
        keys.dedup();
        if keys.len() <= MAX_CELLS {
            break keys;
        }
        cell *= 2.0;
    };
    let cells: Vec<SwayCell> = keys
        .iter()
        .map(|k| {
            let center = cmin
                + Vec3A::new(
                    (k.0 as f32 + 0.5) * cell,
                    (k.1 as f32 + 0.5) * cell,
                    (k.2 as f32 + 0.5) * cell,
                );
            let amp = SWAY_AMP_K * scale * height_factor(center.y, cmin.y, cmax.y);
            SwayCell { center, amp }
        })
        .collect();
    let mut tri_cell = vec![STATIC_CELL; n];
    for &t in &leaf_tris {
        // keys is sorted + deduped, so the index IS the cell id.
        let k = key_at(centroid(t), cell);
        let c = keys.binary_search(&k).expect("leaf key must be in the partition");
        tri_cell[t as usize] = c as u16;
    }
    let offsets = (0..cells.len()).map(|_| [(); 3].map(|_| AtomicU32::new(0))).collect();
    Some(SceneSway {
        cells,
        tri_cell,
        scale,
        cell,
        offsets,
        baked: AtomicU32::new(f32::NAN.to_bits()),
    })
}

/// Attach/refresh `Scene::sway` under the session predicate. Called at every
/// scene-load site (cold OBJ/glTF, warm sidecar, world islands + merge) and
/// by the live-edit path BEFORE each BVH rebuild — appended triangles must
/// lengthen `tri_cell` or the intersector indexes out of bounds.
pub fn attach(scene: &mut Scene) {
    scene.sway = None;
    if !sweep_armed() {
        return;
    }
    let mask = leaf_materials(scene);
    scene.sway = cell_partition(scene, &mask).map(Box::new);
}

/// Bake this frame's per-cell offsets (bit-equal clock = free — a converging
/// still's frozen clock costs nothing). Takes `&self` (relaxed-atomic
/// stores) but MUST only run between traces on the main thread — the
/// accum-buffer discipline the field doc states. Each cell's translation is
/// `translation(cells[c], c, ..)` — the SAME (cell, index) pairs the GPU
/// ring uses through `SwaySplit::cell_of`, which is what makes the CPU and
/// GPU poses bit-equal.
pub fn bake(sway: &SceneSway, time: f32) {
    if sway_abl().rest {
        return; // cost probe: offsets stay all-zero (the rest pose)
    }
    if sway.baked.load(Ordering::Relaxed) == time.to_bits() {
        return;
    }
    for i in 0..sway.cells.len() {
        let t = translation(&sway.cells[i], i as u32, sway.scale, time);
        let o = &sway.offsets[i];
        o[0].store(t.x.to_bits(), Ordering::Relaxed);
        o[1].store(t.y.to_bits(), Ordering::Relaxed);
        o[2].store(t.z.to_bits(), Ordering::Relaxed);
    }
    sway.baked.store(time.to_bits(), Ordering::Relaxed);
}

/// `split_plan`'s product: which appended chunks sway, and the scale the
/// per-frame bake needs.
pub struct SwaySplit {
    /// Index of the first sway chunk in the rebuilt plan — chunks
    /// `first_chunk..first_chunk + cells.len()` are the animated instances,
    /// everything below is static.
    pub first_chunk: u32,
    /// One entry per sway chunk (RUN), in chunk order.
    pub cells: Vec<SwayCell>,
    /// Per-run PARTITION cell index — the flutter hash key (v0 hashed the
    /// RUN index, so two cap-overflow runs of one cell fluttered apart,
    /// contradicting the runs-translate-identically contract; re-keyed on
    /// the cell so runs agree AND match the CPU bake's per-cell offsets).
    pub cell_of: Vec<u32>,
    /// The content diagonal every length constant above multiplies.
    pub scale: f32,
    /// The grid pitch the split settled on (after any coarsening) — the
    /// startup line's number.
    pub cell: f32,
}

/// Per-run bake for the GPU ring: each run's translation is keyed by its
/// PARTITION cell, so runs of one overflowed cell translate identically and
/// the result is bit-equal to the CPU `bake` of that cell (same function,
/// same arguments).
pub fn translations_keyed(
    cells: &[SwayCell],
    cell_of: &[u32],
    scale: f32,
    time: f32,
) -> Vec<Vec3A> {
    cells.iter().zip(cell_of).map(|(c, &i)| translation(c, i, scale, time)).collect()
}

/// Per-material leaf mask: foliage-classified (`Material::class` — the
/// classify verdict retained at load, which is the ONLY way the Minecraft
/// scenes work: their signal lives on the `newmtl` NAME while one shared
/// atlas texture carries every material, so a texture-stem re-derivation is
/// structurally blind there) AND alpha-masked (the cutout signal — separates
/// leaves from bark/trunk, which share the foliage token table but must not
/// translate rigidly; on the atlas scenes it is the class byte doing the
/// separating, since the whole atlas is alpha-masked). Pure; derived like
/// `any_alpha`, so it keys nothing in the scene cache beyond the class byte
/// DiskMat already carries.
pub fn leaf_materials(scene: &Scene) -> Vec<bool> {
    scene
        .materials
        .iter()
        .map(|m| {
            m.class == crate::matclass::IDX_FOLIAGE as u8
                && match m.kind {
                    MatKind::Textured { tex } => scene.textures[tex as usize].alpha_masked,
                    _ => false,
                }
        })
        .collect()
}

/// Height fade: `SWAY_GROUND_K` at the content floor, 1 above
/// `SWAY_HEIGHT_BAND` of the content height.
#[inline]
fn height_factor(y: f32, cmin_y: f32, cmax_y: f32) -> f32 {
    let band = (SWAY_HEIGHT_BAND * (cmax_y - cmin_y)).max(1e-6);
    ((y - cmin_y) / band).clamp(SWAY_GROUND_K, 1.0)
}

/// Re-partition a `BlasPlan` against an attached partition (`Scene::sway`):
/// pull every leaf triangle out of the antichain chunks and append one chunk
/// per non-empty partition cell (split into `max_prims` runs if a cell
/// overflows the BLAS cap — the oversized-leaf idiom). Static chunks keep
/// their `chunk_node`; sway chunks carry `u32::MAX` (they are cells, not BVH
/// subtrees — the antichain property is deliberately given up on the sway
/// tail, which is why `blas_split::self_test` keeps gating the UNSPLIT
/// planner and this module gates its own product).
///
/// Returns `None` (plan untouched, bit-identical) when the partition maps
/// none of the plan's triangles. Deterministic: cell order is the
/// partition's sorted-grid-key order (== the v0 BTreeMap order), triangle
/// order inside a bucket follows `packed_tris` order.
pub fn split_plan(
    plan: &mut crate::blas_split::BlasPlan,
    sway: &SceneSway,
    max_prims: u32,
) -> Option<SwaySplit> {
    let cell_of_tri = |t: u32| sway.tri_cell[t as usize];
    let is_leaf = |t: u32| cell_of_tri(t) != STATIC_CELL;
    if !plan.packed_tris.iter().any(|&t| is_leaf(t)) {
        return None;
    }

    // Buckets indexed by PARTITION cell, filled in packed order (preserves
    // the intra-bucket order rule); cell order is the partition's.
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); sway.cells.len()];
    for &t in &plan.packed_tris {
        let c = cell_of_tri(t);
        if c != STATIC_CELL {
            buckets[c as usize].push(t);
        }
    }

    // Rebuild: static chunks first (leaf tris filtered out; a chunk emptied
    // entirely is dropped — a zero-prim BLAS is illegal), then one chunk per
    // cell run.
    let mut packed: Vec<u32> = Vec::with_capacity(plan.packed_tris.len());
    let mut base: Vec<u32> = vec![0];
    let mut node: Vec<u32> = Vec::new();
    for i in 0..plan.chunks() {
        let before = packed.len();
        packed.extend(plan.tris(i).iter().copied().filter(|&t| !is_leaf(t)));
        if packed.len() == before {
            continue;
        }
        base.push(packed.len() as u32);
        node.push(plan.chunk_node[i]);
    }
    let first_chunk = node.len() as u32;
    let mut cells: Vec<SwayCell> = Vec::with_capacity(buckets.len());
    let mut cell_of: Vec<u32> = Vec::with_capacity(buckets.len());
    let cap = max_prims.max(1) as usize;
    for (c, tris) in buckets.iter().enumerate() {
        if tris.is_empty() {
            continue;
        }
        // A cell over the BLAS cap splits into runs — several instances share
        // one cell anchor and translate IDENTICALLY: `cell_of` carries the
        // PARTITION cell index every run's flutter hashes (the v0.2 re-key).
        let mut off = 0;
        while off < tris.len() {
            let take = (tris.len() - off).min(cap);
            packed.extend_from_slice(&tris[off..off + take]);
            base.push(packed.len() as u32);
            node.push(u32::MAX);
            cells.push(sway.cells[c]);
            cell_of.push(c as u32);
            off += take;
        }
    }
    debug_assert_eq!(packed.len(), plan.packed_tris.len());
    plan.packed_tris = packed;
    plan.chunk_base = base;
    plan.chunk_node = node;
    Some(SwaySplit { first_chunk, cells, cell_of, scale: sway.scale, cell: sway.cell })
}

/// The curl field as a unit-bounded direction at foliage wavelength — the
/// fireflies `curl_dir` shape (a synthetic time-0 `Clouds`; the field is
/// time-independent and reads only `diag`), with `SWAY_FIELD_K` setting the
/// wavelength instead of the whole content diagonal.
#[inline]
fn curl_dir(p: Vec3A, scale: f32) -> Vec3A {
    let field = (SWAY_FIELD_K * scale).max(1e-6);
    crate::clouds::curl_offset(p, &crate::clouds::Clouds::new(true, field, 0.0))
        * (1.0 / (crate::clouds::CLOUD_CURL_AMP_K * field))
}

/// Upper bound on ANY cell's displacement at any time under multiplier
/// `mult`: the curl half is exactly `mult·amp` per axis (soft
/// normalization), the flutter half `mult·SWAY_BOB_K` per axis; `√3` folds
/// per-axis to vector length. `self_test` sweeps it at several mults.
pub fn displacement_bound_with(amp: f32, scale: f32, mult: f32) -> f32 {
    3f32.sqrt() * mult * (amp + SWAY_BOB_K * scale)
}

/// `displacement_bound_with` at the session's `--foliage-amp`.
pub fn displacement_bound(amp: f32, scale: f32) -> f32 {
    displacement_bound_with(amp, scale, amp_mult())
}

/// The BVH build-sweep pad for one triangle: its cell's displacement bound
/// at `sweep_mult()` (0 for static tris). ONE function serves the build
/// (`bvh::grow_sway_sweep` pads min AND max by it — translation is signed)
/// and the self-test's swept-containment pin — the `tri_height_depth`
/// build-vs-runtime discipline, which is the containment proof.
pub fn sway_pad(sway: &SceneSway, tri: u32) -> f32 {
    let c = sway.tri_cell[tri as usize];
    if c == STATIC_CELL {
        return 0.0;
    }
    displacement_bound_with(sway.cells[c as usize].amp, sway.scale, sweep_mult())
}

/// Closed-form translation for cell `i` at clock `time` — the whole motion
/// model, the fireflies `pose` shape. Pure function of (cell, i, time, mult);
/// hashes are `sky::pcg_mix` chains. Zero rng draws. The public `translation`
/// reads the session `--foliage-amp`; this form takes it explicitly so the
/// self-test can sweep mults without touching the global inside `--check`.
fn translation_with(c: &SwayCell, i: u32, scale: f32, time: f32, mult: f32) -> Vec3A {
    use crate::sky::{hash01, pcg_mix};
    // Gusts: the static field sampled at a lookup point moving along the one
    // wind line — spatially continuous across cells (no per-cell offset).
    let lookup = c.center + Vec3A::new(0.37, 0.0, 0.61) * (SWAY_WIND_K * scale * time);
    let sway = curl_dir(lookup, scale) * (c.amp * mult);
    // Flutter: three hashed sines per cell (ω ∈ [0.8, 2.4] rad/s — leaves are
    // quicker than fireflies), amplitude height-scaled with the curl half so
    // ground-band cells stay pinned.
    let h0 = pcg_mix(i.wrapping_mul(0x9E37_79B9) ^ 0x5EA5_1EAF);
    let h1 = pcg_mix(h0);
    let h2 = pcg_mix(h1);
    let h3 = pcg_mix(h2);
    let h4 = pcg_mix(h3);
    let h5 = pcg_mix(h4);
    let tau = std::f32::consts::TAU;
    let bob_amp = mult * SWAY_BOB_K * scale * if c.amp > 0.0 { 1.0 } else { 0.0 };
    let bob = Vec3A::new(
        ((0.8 + 1.6 * hash01(h0)) * time + tau * hash01(h3)).sin(),
        ((0.8 + 1.6 * hash01(h1)) * time + tau * hash01(h4)).sin(),
        ((0.8 + 1.6 * hash01(h2)) * time + tau * hash01(h5)).sin(),
    ) * bob_amp;
    sway + bob
}

/// `translation_with` at the session's `--foliage-amp`.
pub fn translation(c: &SwayCell, i: u32, scale: f32, time: f32) -> Vec3A {
    translation_with(c, i, scale, time, amp_mult())
}


/// `--check` gate (pure, DLL-free, GPU-free). Runs regardless of the lever —
/// the blas_split precedent, so the machinery can't rot while off. The
/// procedural/stress scenes have no leaf materials, so the split half runs on
/// a SYNTHETIC mask (every material marked) — the partition contracts are
/// mask-independent; `leaf_materials`' own anchors are closed-form.
pub fn self_test(scene: &Scene, bvh: &crate::bvh::Bvh) -> Result<(), String> {
    // -- leaf_materials anchors, closed-form on a synthetic scene: each leg
    // of the predicate (foliage class byte / textured / alpha-masked)
    // negated in isolation must kill the mask bit. The class byte cases are
    // matclass::self_test's job (the classify vocabulary); here the byte is
    // stamped directly.
    {
        let mk_tex = |alpha: bool| crate::texture::Texture {
            w: 1,
            h: 1,
            texels: vec![[255, 255, 255, if alpha { 0 } else { 255 }]],
            alpha_masked: alpha,
            srgb: true,
            source: String::new(),
            h2n: false,
            n2h: false,
            mips: Vec::new(),
        };
        let mut b = crate::scene::SceneBuilder::new();
        let t_alpha = b.add_texture(mk_tex(true));
        let t_opaque = b.add_texture(mk_tex(false));
        let white = Vec3A::ONE;
        let m_leaf = b.material_kind(white, 0.5, 0.0, 0.0, MatKind::Textured { tex: t_alpha });
        let m_bark = b.material_kind(white, 0.5, 0.0, 0.0, MatKind::Textured { tex: t_opaque });
        let m_cutout = b.material_kind(white, 0.5, 0.0, 0.0, MatKind::Textured { tex: t_alpha });
        let m_flat = b.material_kind(white, 0.5, 0.0, 0.0, MatKind::Diffuse);
        b.tri([Vec3A::ZERO, Vec3A::X, Vec3A::Y], [Vec3A::Z; 3], m_leaf);
        let mut synth = b.finish(crate::sky::Sun::new(Vec3A::Y));
        let fol = crate::matclass::IDX_FOLIAGE as u8;
        synth.materials[m_leaf as usize].class = fol;
        synth.materials[m_bark as usize].class = fol; // opaque texture = bark, static
        synth.materials[m_flat as usize].class = fol; // untextured, static
        let m = leaf_materials(&synth);
        let want = |i: u32, w: bool, what: &str| -> Result<(), String> {
            if m[i as usize] != w {
                return Err(format!("leaf_materials: {what} should be {w}"));
            }
            Ok(())
        };
        want(m_leaf, true, "foliage + textured + alpha")?;
        want(m_bark, false, "foliage + opaque texture (bark)")?;
        want(m_cutout, false, "non-foliage cutout")?;
        want(m_flat, false, "foliage + untextured")?;
    }
    // Mask length is per-material on the session scene.
    let mask = leaf_materials(scene);
    if mask.len() != scene.materials.len() {
        return Err("leaf_materials: mask length != material count".into());
    }

    // -- the partition + split structural contracts, on a synthetic all-leaf
    // mask over the session's real tree (mask-independent properties).
    let cap = crate::blas_split::DEFAULT_MAX_PRIMS;
    let base = crate::blas_split::plan(bvh, cap);
    let n_tris = base.packed_tris.len();
    if n_tris == 0 {
        return Err("empty plan".into());
    }
    let all = vec![true; scene.materials.len()];
    let none = vec![false; scene.materials.len()];

    // Off arm: an all-false mask has no partition — the structural off-state
    // (the caller never reaches split_plan without a partition).
    if cell_partition(scene, &none).is_some() {
        return Err("cell_partition: empty mask must return None".into());
    }
    let Some(part) = cell_partition(scene, &all) else {
        return Err("cell_partition: all-leaf mask produced no partition".into());
    };
    if part.cells.is_empty() || part.cells.len() > MAX_CELLS {
        return Err(format!("partition cell count {} out of band", part.cells.len()));
    }
    if part.tri_cell.len() != scene.indices.len()
        || part.offsets.len() != part.cells.len()
        || part.tri_cell.iter().any(|&c| c != STATIC_CELL && c as usize >= part.cells.len())
    {
        return Err("cell_partition: tri_cell/offsets shape broken".into());
    }
    if part.offsets_snapshot().iter().any(|&o| o != Vec3A::ZERO)
        || !f32::from_bits(part.baked.load(Ordering::Relaxed)).is_nan()
    {
        return Err("cell_partition must start at the rest pose".into());
    }

    // On arm: exact partition, mask routing, no empty chunks, cap held,
    // cells within the cap, determinism.
    let mut p = crate::blas_split::plan(bvh, cap);
    let Some(sp) = split_plan(&mut p, &part, cap) else {
        return Err("split_plan: all-leaf partition produced no split".into());
    };
    if p.packed_tris.len() != n_tris {
        return Err(format!("split lost tris: {} != {n_tris}", p.packed_tris.len()));
    }
    let mut seen = vec![false; n_tris.max(scene.indices.len())];
    for i in 0..p.chunks() {
        let prims = p.prims(i);
        if prims == 0 || prims > cap {
            return Err(format!("split chunk {i} holds {prims} prims (cap {cap})"));
        }
        let sway = i as u32 >= sp.first_chunk;
        if sway != (p.chunk_node[i] == u32::MAX) {
            return Err(format!("chunk {i}: sway tail / sentinel disagree"));
        }
        for &t in p.tris(i) {
            if seen[t as usize] {
                return Err(format!("tri {t} appears in two chunks after the split"));
            }
            seen[t as usize] = true;
        }
    }
    if p.chunks() as u32 - sp.first_chunk != sp.cells.len() as u32 {
        return Err("cells not parallel to the sway chunk tail".into());
    }
    if sp.cells.is_empty() || sp.cells.len() > MAX_CELLS + 16 {
        // +16: cap-overflow runs may exceed MAX_CELLS by the split runs —
        // bounded by construction, but the coarsening loop keys on DISTINCT
        // grid keys.
        return Err(format!("cell count {} out of band", sp.cells.len()));
    }
    // The run -> partition-cell map (the flutter re-key's data): parallel to
    // cells, in-range, and each run's SwayCell IS its partition cell's.
    if sp.cell_of.len() != sp.cells.len()
        || sp.cell_of.iter().any(|&c| c as usize >= part.cells.len())
        || sp
            .cells
            .iter()
            .zip(&sp.cell_of)
            .any(|(rc, &c)| *rc != part.cells[c as usize])
    {
        return Err("split_plan: cell_of broken (run cell != partition cell)".into());
    }
    {
        let mut q = crate::blas_split::plan(bvh, cap);
        let part2 = cell_partition(scene, &all).ok_or("determinism re-partition vanished")?;
        if part2.cells != part.cells || part2.tri_cell != part.tri_cell {
            return Err("cell_partition is not deterministic".into());
        }
        let sq = split_plan(&mut q, &part2, cap).ok_or("determinism re-split vanished")?;
        if q.packed_tris != p.packed_tris
            || q.chunk_base != p.chunk_base
            || q.chunk_node != p.chunk_node
            || sq.cells != sp.cells
            || sq.cell_of != sp.cell_of
        {
            return Err("split_plan is not deterministic".into());
        }
    }

    // Mixed mask on the real tree when the scene HAS leaf materials (San
    // Miguel/bistro --check): marked tris land in the sway tail, unmarked in
    // the static head. Prefers the session's own attached partition
    // (Scene::sway) so armed foliage-scene checks exercise the real object.
    let real_part;
    let real = match &scene.sway {
        Some(sw) => Some(&**sw),
        None => {
            real_part = cell_partition(scene, &mask);
            real_part.as_ref()
        }
    };
    if let Some(rp) = real {
        let mut q = crate::blas_split::plan(bvh, cap);
        if let Some(sq) = split_plan(&mut q, rp, cap) {
            for i in 0..q.chunks() {
                let sway = i as u32 >= sq.first_chunk;
                for &t in q.tris(i) {
                    let leaf = rp.tri_cell[t as usize] != STATIC_CELL;
                    if leaf != sway {
                        return Err(format!(
                            "tri {t} (leaf={leaf}) landed in the wrong half (chunk {i})"
                        ));
                    }
                }
            }
        }
    }

    // The flutter re-key pin: force cap-overflow runs (cap = half the
    // largest cell's population) — runs sharing a cell must translate
    // BIT-IDENTICALLY, and equal the CPU bake of that cell. Anti-vacuity:
    // the construction guarantees at least one duplicated cell.
    {
        let mut pop = vec![0u32; part.cells.len()];
        for &c in &part.tri_cell {
            if c != STATIC_CELL {
                pop[c as usize] += 1;
            }
        }
        let largest = pop.iter().copied().max().unwrap_or(0);
        if largest >= 2 {
            let small_cap = (largest / 2).max(1);
            let mut q = crate::blas_split::plan(bvh, cap);
            let sq = split_plan(&mut q, &part, small_cap)
                .ok_or("re-key split vanished at the small cap")?;
            let tk = translations_keyed(&sq.cells, &sq.cell_of, sq.scale, 7.3);
            let mut dup_seen = false;
            let mut first_run_of = vec![usize::MAX; part.cells.len()];
            for (j, &c) in sq.cell_of.iter().enumerate() {
                let f = first_run_of[c as usize];
                if f == usize::MAX {
                    first_run_of[c as usize] = j;
                } else {
                    dup_seen = true;
                    if tk[j] != tk[f] {
                        return Err(format!(
                            "runs {f}/{j} of cell {c} flutter apart — the re-key regressed"
                        ));
                    }
                }
                if tk[j] != translation(&part.cells[c as usize], c, part.scale, 7.3) {
                    return Err(format!("run {j} disagrees with the CPU bake of cell {c}"));
                }
            }
            if !dup_seen {
                return Err("re-key pin was vacuous — no cell produced two runs".into());
            }
        }
    }

    // CPU bake == GPU keyed bake, bit-for-bit (the cross-arm pose contract),
    // plus the bit-equal-clock fast path.
    {
        let pm = cell_partition(scene, &all).ok_or("bake partition vanished")?;
        bake(&pm, 7.3);
        let tk = translations_keyed(&sp.cells, &sp.cell_of, sp.scale, 7.3);
        for (j, &c) in sp.cell_of.iter().enumerate() {
            if tk[j] != pm.offset(c as u16) {
                return Err(format!("GPU run {j} != CPU offsets[{c}] after bake"));
            }
        }
        let snap = pm.offsets_snapshot();
        bake(&pm, 7.3); // bit-equal clock — must be a no-op
        if pm.offsets_snapshot() != snap {
            return Err("bake fast path moved offsets on a bit-equal clock".into());
        }
        bake(&pm, 8.1);
        if pm.cells.iter().any(|c| c.amp > 0.0) && pm.offsets_snapshot() == snap {
            return Err("bake did not move offsets on a new clock".into());
        }
    }

    // -- motion: the displacement bound, height-band pinning, determinism,
    // and time-variation — swept across `--foliage-amp` multipliers through
    // `translation_with` (never the global: the session's own setting must
    // not move under a gate run). Indexed by PARTITION cell — the flutter
    // hash key after the v0.2 re-key.
    let scale = part.scale;
    let mut moved = false;
    for (i, c) in part.cells.iter().enumerate().take(64) {
        for &mult in &[0.25f32, 1.0, 4.0] {
            let bound = displacement_bound_with(c.amp, scale, mult) + 1e-5 * scale;
            for &t in &[0.0f32, 0.37, 7.3, 123.4, 4096.0] {
                let d = translation_with(c, i as u32, scale, t, mult);
                if !d.is_finite() {
                    return Err(format!("cell {i}: non-finite translation at t={t}"));
                }
                if d.length() > bound {
                    return Err(format!(
                        "cell {i}: |d| {} exceeds the bound {} at t={t} mult={mult}",
                        d.length(),
                        bound
                    ));
                }
                if d != translation_with(c, i as u32, scale, t, mult) {
                    return Err("translation is not deterministic".into());
                }
                if d.length() > 1e-9 * scale {
                    moved = true;
                }
            }
        }
        // The amp-0 identity must be EXACT — flutter included. Synthetic
        // since the SWAY_GROUND_K floor: no real cell carries amp 0 any
        // more, but the zero-amp arm still guards `translation_with`'s
        // bob gate (and any future scene that legitimately produces one).
        let pinned = SwayCell { center: c.center, amp: 0.0 };
        if translation_with(&pinned, i as u32, scale, 7.3, 4.0) != Vec3A::ZERO {
            return Err("amp-0 cell must not move at all".into());
        }
    }
    if !moved {
        return Err("no cell moved anywhere in the sweep — the field is dead".into());
    }

    // The swept-containment pin (build-vs-motion, the height_self_test
    // shape): every pose reachable at mult <= sweep_mult() lies inside the
    // box the BVH build pads by `sway_pad` — |translation| <= pad, with both
    // signs covered because `grow_sway_sweep` pads min AND max. One function
    // (`sway_pad`) serves the build and this pin, the tri_height_depth
    // discipline.
    {
        let sm = sweep_mult();
        let mut checked = 0u32;
        for (t, &c) in part.tri_cell.iter().enumerate() {
            if c == STATIC_CELL {
                continue;
            }
            let pad = sway_pad(&part, t as u32);
            let cell0 = &part.cells[c as usize];
            for &mult in &[0.25f32, 1.0, sm] {
                for &tt in &[0.0f32, 0.37, 7.3, 123.4] {
                    let d = translation_with(cell0, c as u32, part.scale, tt, mult);
                    if d.length() > pad + 1e-5 * part.scale {
                        return Err(format!(
                            "tri {t}: |d| {} escapes the swept pad {pad} at t={tt} mult={mult}",
                            d.length()
                        ));
                    }
                }
            }
            checked += 1;
            if checked >= 64 {
                break;
            }
        }
        if checked == 0 {
            return Err("swept-containment pin was vacuous — no leaf tri checked".into());
        }
    }

    // -- GATEWAY AUDIT (v0.3): the SAH gateway tree's structural contracts on
    // the session's REAL tree. When gateways can't exist (sway None, alt
    // builder, lever off) the pin is the INVERSE: zero GATEWAY_BIT nodes —
    // the sway-less bit-identity guarantee.
    gateway_audit(scene, bvh)?;

    // -- GATEWAY machinery on synthetic ALL-FOLIAGE micro-scenes: the edges
    // no real scene guarantees. A pure-pseudo top (every tri leaf-classed,
    // >= 2 cells) MUST mint the cherry expansion (E filler must-fire — two
    // gateways can never be siblings); a 1-tri scene is 1 cell and MUST
    // build the root gateway; and a displaced-hit pin proves the traversal
    // shift end to end (ray aimed at rest + offset hits the tri with the
    // exact reconstruction point). Skipped when gateway mode is off (alt
    // builder / --no-blas-split sessions — the audit's inverse pin covers
    // those trees instead).
    if gateway_mode() {
        let synth = |clusters: &[Vec3A], per: usize| -> Scene {
            let mut b = crate::scene::SceneBuilder::new();
            // alpha_masked arms the leaf predicate; the OPAQUE texel keeps
            // the cutout from rejecting the displaced-hit pin's ray.
            let t = b.add_texture(crate::texture::Texture {
                w: 1,
                h: 1,
                texels: vec![[255, 255, 255, 255]],
                alpha_masked: true,
                srgb: true,
                source: String::new(),
                h2n: false,
                n2h: false,
                mips: Vec::new(),
            });
            let m = b.material_kind(Vec3A::ONE, 0.5, 0.0, 0.0, MatKind::Textured { tex: t });
            for &c in clusters {
                for j in 0..per {
                    let o = c + Vec3A::new(j as f32 * 0.1, 0.0, 0.0);
                    b.tri([o, o + 0.05 * Vec3A::X, o + 0.05 * Vec3A::Y], [Vec3A::Z; 3], m);
                }
            }
            let mut s = b.finish(crate::sky::Sun::new(Vec3A::Y));
            s.materials[m as usize].class = crate::matclass::IDX_FOLIAGE as u8;
            attach(&mut s);
            s
        };
        // Two clusters far apart and vertically spread (height_factor is
        // floored at SWAY_GROUND_K, so both move; the UPPER cluster carries
        // the full amplitude and is the one the displaced-hit pin targets).
        let s2 = synth(&[Vec3A::ZERO, Vec3A::new(9.0, 9.0, 9.0)], 3);
        let Some(sw2) = s2.sway.as_deref() else {
            return Err("gateway synth: attach declined an all-foliage scene".into());
        };
        if sw2.cells.len() < 2 {
            return Err(format!("gateway synth: wanted >= 2 cells, got {}", sw2.cells.len()));
        }
        let bvh2 = crate::bvh::Bvh::build(&s2);
        gateway_audit(&s2, &bvh2)?;
        if !bvh2.nodes.iter().any(|n| n.is_gateway() && n.leaf_count() == 0) {
            return Err("gateway synth: pure-pseudo top minted no cherry (E filler)".into());
        }
        // Displaced-hit pin: bake a pose, aim a ray at (rest centroid + the
        // cell's exact offset) of an UPPER-cluster tri — the gateway shift
        // must land the hit there (t preserved, o + t·d on the DISPLACED
        // surface), which is the end-to-end proof of the traversal arm.
        bake(sw2, 5.0);
        let hi = 3usize; // first tri of the second (elevated) cluster
        let tv = s2.indices[hi];
        let rest = (s2.positions[tv[0] as usize]
            + s2.positions[tv[1] as usize]
            + s2.positions[tv[2] as usize])
            / 3.0;
        let off = sw2.offset(sw2.tri_cell[hi]);
        if off == Vec3A::ZERO {
            return Err("gateway synth: baked pose has zero offset (vacuous pin)".into());
        }
        let want = rest + off;
        let ray = crate::bvh::Ray::new(want + 5.0 * Vec3A::Z, -Vec3A::Z);
        let mut vis = 0u64;
        match bvh2.intersect(&s2, &ray, 0.0, 100.0, &mut vis) {
            Some(h) => {
                let p = ray.o + h.t * ray.d;
                if h.tri != hi as u32 || (p - want).length() > 1e-4 {
                    return Err(format!(
                        "gateway synth: displaced hit wrong (tri {} d {})",
                        h.tri,
                        (p - want).length()
                    ));
                }
            }
            None => return Err("gateway synth: ray at the displaced pose missed".into()),
        }
        // One triangle = one cell = the ROOT gateway.
        let s1 = synth(&[Vec3A::new(0.0, 1.0, 0.0)], 1);
        if s1.sway.is_none() {
            return Err("gateway synth: 1-tri attach failed".into());
        }
        let bvh1 = crate::bvh::Bvh::build(&s1);
        if !bvh1.nodes[0].is_gateway() {
            return Err("gateway synth: single-cell scene did not root-gateway".into());
        }
        gateway_audit(&s1, &bvh1)?;
    }

    eprintln!(
        "foliage self-test: OK — synthetic split {} tris -> {} static + {} sway chunks \
         (cell {:.3}, scale {:.2}); scene leaf materials: {}",
        n_tris,
        sp.first_chunk,
        sp.cells.len(),
        sp.cell,
        scale,
        mask.iter().filter(|&&m| m).count()
    );
    Ok(())
}

/// The gateway tree's structural contracts (see `bvh::GATEWAY_BIT`): truthful
/// ranges == partition cells, the implicit +1 adjacency with no nesting, a
/// full reachability tiling (every node is reached from the root via child
/// links XOR lives inside exactly one gateway's subtree block — which is also
/// the proof the phase-2 stitch preserved adjacency), the swept-box identity
/// (gateway box == subtree root's rest box ± the cell's displacement bound,
/// BITWISE — combined with the swept-containment pin above, every displaced
/// pose stays inside the one swept box), E-filler shape, the exactly-one-
/// gateway-per-cell must-fire, and `tri_idx` remaining a true permutation.
fn gateway_audit(scene: &Scene, bvh: &crate::bvh::Bvh) -> Result<(), String> {
    use crate::bvh::GATEWAY_BIT;
    let gw_nodes: Vec<u32> = (0..bvh.nodes.len() as u32)
        .filter(|&i| bvh.nodes[i as usize].is_gateway())
        .collect();
    let live = scene.sway.as_deref().filter(|_| gateway_mode());
    let Some(sw) = live else {
        if !gw_nodes.is_empty() {
            return Err(format!(
                "{} GATEWAY_BIT nodes in a tree that must have none (sway {}, gateway_mode {})",
                gw_nodes.len(),
                scene.sway.is_some(),
                gateway_mode()
            ));
        }
        return Ok(());
    };
    let n_cells = sw.cells.len();
    // Cell tri counts from the partition (the truth the ranges must match).
    let mut cell_n = vec![0u32; n_cells];
    for &c in sw.tri_cell.iter() {
        if c != STATIC_CELL {
            cell_n[c as usize] += 1;
        }
    }
    let live_cells = cell_n.iter().filter(|&&n| n > 0).count();
    if live_cells > 0 && gw_nodes.iter().all(|&g| bvh.nodes[g as usize].leaf_count() == 0) {
        return Err("must-fire: leaf cells exist but no non-empty gateway was built".into());
    }

    // tri_idx must still be a permutation of 0..n (the dual-buffer emission
    // wrote every slot exactly once).
    let n = scene.indices.len();
    let mut seen = vec![false; n];
    for &t in &bvh.tri_idx {
        let t = t as usize;
        if t >= n || seen[t] {
            return Err(format!("tri_idx is not a permutation at id {t}"));
        }
        seen[t] = true;
    }

    // Reachability walk from the root: internal children pushed, gateways
    // TERMINAL (the bound-consumer view). Marks 1 = reached from root.
    let mut mark = vec![0u8; bvh.nodes.len()];
    let mut stack: Vec<u32> = if bvh.nodes.is_empty() { Vec::new() } else { vec![0] };
    while let Some(i) = stack.pop() {
        if mark[i as usize] != 0 {
            return Err(format!("node {i} reached twice from the root"));
        }
        mark[i as usize] = 1;
        let nd = &bvh.nodes[i as usize];
        if nd.count == 0 {
            stack.push(nd.left_first);
            stack.push(nd.left_first + 1);
        }
    }

    let mut cell_seen = vec![false; n_cells];
    for &g in &gw_nodes {
        let nd = &bvh.nodes[g as usize];
        let cc = nd.leaf_count() as usize;
        if cc == 0 {
            // E filler: no subtree, reached from the root, and its box is a
            // COPY of its +1 sibling gateway's (never an inverted EMPTY —
            // the quantized-ftree/NaN hazard the build comment documents).
            // Bounds-check the sibling: validate_links deliberately admits a
            // masked-count-0 gateway at the LAST node index (a corrupt-but-
            // link-valid sidecar), and the audit must Err there, not panic.
            let Some(sib) = bvh.nodes.get(g as usize + 1) else {
                return Err(format!("E filler {g} malformed (box/sibling/reachability)"));
            };
            if !sib.is_gateway()
                || nd.aabb.min.to_array().map(f32::to_bits)
                    != sib.aabb.min.to_array().map(f32::to_bits)
                || nd.aabb.max.to_array().map(f32::to_bits)
                    != sib.aabb.max.to_array().map(f32::to_bits)
                || mark[g as usize] != 1
            {
                return Err(format!("E filler {g} malformed (box/sibling/reachability)"));
            }
            continue;
        }
        if mark[g as usize] != 1 {
            return Err(format!("gateway {g} not reached from the root"));
        }
        let lf = nd.left_first as usize;
        // Truthful range == exactly one whole cell, in ascending (CSR) order.
        let c = sw.tri_cell[bvh.tri_idx[lf] as usize];
        if c == STATIC_CELL {
            return Err(format!("gateway {g} range starts on a static tri"));
        }
        if cell_seen[c as usize] {
            return Err(format!("cell {c} owns two gateways"));
        }
        cell_seen[c as usize] = true;
        if cc as u32 != cell_n[c as usize] {
            return Err(format!(
                "gateway {g}: range holds {cc} tris, cell {c} has {}",
                cell_n[c as usize]
            ));
        }
        // The range is a PERMUTATION of the cell's tris (the cell subtree's
        // SAH build partitions it in place — CSR ascending order holds only
        // at emission): same-cell membership + the count match above + the
        // global tri_idx permutation check together prove set equality.
        let range = &bvh.tri_idx[lf..lf + cc];
        if range.iter().any(|&t| sw.tri_cell[t as usize] != c) {
            return Err(format!("gateway {g} range mixes cells"));
        }
        // Subtree walk from g+1: rest-tight interior, NO nested gateway, leaf
        // ranges exactly tiling the gateway's range; marks 2 = in a subtree.
        let sub = g + 1;
        if sub as usize >= bvh.nodes.len() {
            return Err(format!("gateway {g} has no +1 subtree"));
        }
        let mut sstack = vec![sub];
        let mut tiled = vec![false; cc];
        while let Some(i) = sstack.pop() {
            if mark[i as usize] != 0 {
                return Err(format!("subtree node {i} already owned (nesting/overlap)"));
            }
            mark[i as usize] = 2;
            let snd = &bvh.nodes[i as usize];
            if snd.count & GATEWAY_BIT != 0 {
                return Err(format!("gateway nested inside gateway {g}"));
            }
            if snd.count == 0 {
                sstack.push(snd.left_first);
                sstack.push(snd.left_first + 1);
            } else {
                let f = snd.left_first as usize;
                for k in f..f + snd.count as usize {
                    if k < lf || k >= lf + cc || tiled[k - lf] {
                        return Err(format!("gateway {g} subtree leaf outside/overlapping its range"));
                    }
                    tiled[k - lf] = true;
                }
            }
        }
        if !tiled.iter().all(|&t| t) {
            return Err(format!("gateway {g} subtree leaves do not tile its range"));
        }
        // The one swept box: subtree root's rest box ± the cell's bound.
        let pad =
            displacement_bound_with(sw.cells[c as usize].amp, sw.scale, sweep_mult());
        let sb = &bvh.nodes[sub as usize].aabb;
        let (emin, emax) = (sb.min - glam::Vec3A::splat(pad), sb.max + glam::Vec3A::splat(pad));
        if emin.to_array().map(f32::to_bits) != nd.aabb.min.to_array().map(f32::to_bits)
            || emax.to_array().map(f32::to_bits) != nd.aabb.max.to_array().map(f32::to_bits)
        {
            return Err(format!("gateway {g} box != subtree rest box ± pad (bitwise)"));
        }
    }
    if let Some(c) = (0..n_cells).find(|&c| cell_n[c] > 0 && !cell_seen[c]) {
        return Err(format!("cell {c} has tris but no gateway"));
    }
    // Full tiling: every node is root-reachable XOR inside one subtree.
    if let Some(i) = mark.iter().position(|&m| m == 0) {
        return Err(format!("node {i} orphaned (neither root-reachable nor in a subtree)"));
    }
    eprintln!(
        "foliage gateway audit: OK — {} gateways over {live_cells} cells ({} E fillers)",
        gw_nodes.iter().filter(|&&g| bvh.nodes[g as usize].leaf_count() > 0).count(),
        gw_nodes.iter().filter(|&&g| bvh.nodes[g as usize].leaf_count() == 0).count(),
    );
    Ok(())
}
