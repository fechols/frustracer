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
//! - **WHOLE PLANTS (v0.5)**: sway participation is leaf OR woody
//!   (`plant_materials` — a leaf is foliage-classed AND alpha-masked, the
//!   `Material::class` byte carrying the NAME half of the vocabulary, which
//!   is all the Minecraft atlas scenes have; WOODY is the `bark` class byte
//!   alone, no alpha leg — trunks, branches, stems, Minecraft `Log`).
//!   `derive_plants` groups the masked geometry into PLANTS: union-find
//!   over shared vertices finds the connected trunk/branch meshes,
//!   proximity-merge fuses touching woody components, and disconnected leaf
//!   components attach to the overlapping plant — a tree is trunk + canopy.
//!   Each plant is a POSE GROUP, not a spatial cell: its cells stay
//!   voxel-sized (BVH/gateway bound quality) but copy the plant's
//!   anchor/chord/hash-key BITWISE, so equal pose parameters give the equal
//!   affine map and a trunk crossing many cells cannot tear — which is what
//!   makes moving OPAQUE CONNECTED geometry sound where v0.1-v0.4 could
//!   only move disconnected cutout leaves (the paper's clipping +
//!   shared-cage-vertex machinery is still not needed: intra-plant the map
//!   is one affine, and inter-plant boundaries are disconnected geometry).
//!   Leaf components touching no plant stay FIELD members — the v0.4
//!   per-voxel behavior, bit-exact when a scene has zero plants.
//! - **PER-REGION SCALE (v0.6)**: every sway length — the proximity/attach
//!   tolerances, cell pitch, ramp band, curl wavelength/scroll, amplitude —
//!   derives from a `SwayRegion`'s OWN content box (`SwayCell::scale`
//!   carries it to the runtime), one region per world island
//!   (`Scene::sway_regions`, set by `world::merge_scenes`, serialized in
//!   the world sidecar). Before this the MERGED world's ~10× content
//!   diagonal fused whole Minecraft forests into single plants whose chord
//!   spanned the forest — an individual trunk sat on a nearly flat slice of
//!   the ramp and translated rigidly (the "world trees aren't grounded"
//!   report). Merges/attaches never cross a region (grid keys carry the
//!   region id — structural); a region-less scene takes one implicit
//!   whole-scene region, BIT-IDENTICAL to the v0.5 partition (the
//!   self-test's neutrality pin). MAX_CELLS/MAX_PLANTS doubled to
//!   4096/2048 alongside — the world's ~2.2k real trees must survive as
//!   plants, because a demoted tree is a field member whose base rides the
//!   ramp, i.e. exactly the ungrounded look the plants exist to fix — with
//!   `bake`/`winds` gone rayon-parallel to keep the per-frame wind bill
//!   flat.
//! - **ROOTED HORIZONTAL SHEAR, per spatial CELL** (v0.4; no tets yet):
//!   `cell_partition` buckets leaf triangles by centroid into a grid over
//!   the content box — ONE partition (`Scene::sway`, derived at load, never
//!   serialized) shared by every consumer, which is the correctness spine:
//!   the CPU intersector displaces per `tri_cell`, the BVH build sweeps per
//!   `sway_pad`, and the GPU split (`split_plan`) turns each cell into a
//!   BLAS chunk + animated TLAS instance. The per-cell pose is the affine
//!   map `p' = p + u·(a + b·p.y)`: the baked WIND vector `u` (`u.y ≡ 0`)
//!   scaled by the cell's CHORD of the global rooting ramp — 0 exactly at
//!   the content floor, 1 above `SWAY_HEIGHT_BAND` (see `SWAY_RAMP_GAMMA`)
//!   — so a grass blade bends from its base and a canopy's crown sways more
//!   than its underside, where v0.1-v0.3 translated each cell RIGIDLY (the
//!   "whole cross wiggles" / sliding-base artifact, retired). `u.y ≡ 0`
//!   makes the map UNIPOTENT (det = 1) with the closed-form exact inverse
//!   `M⁻¹(q) = q − u·(a + b·q.y)`, which is what keeps every arm's plumbing
//!   cheap: normals, tangents, UVs and barycentrics still read rest space
//!   (error O(|u|·b), band-interior only — the v0.1 "translation leaves
//!   them untouched" accept, extended), and every shading path still
//!   reconstructs the hit point as `o + t·d`. The remaining coarseness is
//!   per-cell-rigid `u` (leaves in one cell share one wind vector; the ramp
//!   still grades them by height); the paper's per-tet affine is the
//!   follow-on. v0.5 RETRACTED the v0.4 "per-plant bases are impossible"
//!   verdict — it was scoped to leaf-only masks (disconnected billboards),
//!   and with bark participating, union-find + proximity attach DOES find
//!   plants: a PLANT cell's chord roots at the plant's OWN base over its
//!   own height (`w_max` still = the global ramp at the plant top, so tall
//!   plants move more), which retires the potted-plant known-accept — a
//!   plant on a table roots at its base, not the scene floor. FIELD cells
//!   (plant-less grass/flowers) keep the global content-floor ramp.
//! - **All three arms consume the motion (v0.2), through GATEWAY subtrees
//!   (v0.3 — the design doc's "pad at cell granularity, never
//!   per-triangle")**: on the default SAH builder (`gateway_mode()`) each
//!   cell's triangles build as a rest-space-TIGHT subtree behind ONE
//!   gateway node whose box alone is swept by the displacement bound
//!   (`bvh::GATEWAY_BIT` — a truthful fat leaf to every bound/cut consumer,
//!   subtree implicitly at +1), so every frustum bound, temporal claim,
//!   structure-replay record and hemi query stays a conservative lower
//!   bound for EVERY pose while the pad lives on <= MAX_CELLS boxes instead
//!   of millions. The CPU maps the ray into cell-rest space ONCE at the
//!   gateway entry (`Bvh::gateway_shear` + `bvh::shear_ray` in the three
//!   traversal arms — the exact unipotent inverse, t preserved because the
//!   rest-space direction is deliberately left unnormalized, `o + t·d`
//!   lands on the displaced surface, every ray type agrees by
//!   construction; `inv_d` is recomputed only when `b·d.y ≠ 0`, so
//!   above-the-band cells and horizontal rays keep the v0.3 cost); the
//!   wavefront + DXR pipelines bind the animated-TLAS ring
//!   (`FrameParams::sway_time`) built BESIDE the pristine static TLAS,
//!   with the shear riding the instance matrix's four non-identity slots
//!   (`shear_rows` — the full 3×4 was always there, v0.1-v0.3 only wrote
//!   its translation column). The v0.2 per-tri sweep + per-test
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
//! Motion is the fireflies template, now TWO scrolling octaves of it: the
//! static curl field (`clouds::curl_offset` — soft-normalized |v| < 1, so
//! the amplitude constant is an EXACT per-axis wind bound), sampled at TWO
//! lookup points advected along the one wind line on the shared
//! `cloud_time` clock — the gust octave (`SWAY_FIELD_K`/`SWAY_WIND_K`) plus
//! a ¼-wavelength, 2.5×-speed fine octave (`SWAY_FIELD2_K`/`SWAY_WIND2_K`,
//! convexly blended by `SWAY_OCT2_K`) that decorrelates NEIGHBORING plants
//! while staying continuous in space and time (the v0.4 "more random" ask;
//! a per-cell hash offset was considered and rejected — see the
//! `SWAY_FIELD_K` comment's tearing argument) — plus a hashed-phase x/z
//! flutter keyed by the PARTITION cell index (the v0.2 re-key: v0 hashed
//! the BLAS run index, so cap-overflow runs of one cell fluttered apart and
//! the CPU/GPU poses could not agree). ZERO rng draws anywhere — poses are
//! pure functions of (cell, time), so every same-seed / replay contract
//! holds structurally.
//!
//! SWAY CARRIES REAL MOTION VECTORS (v0.7 — retiring the v0.1 "no MVs on
//! sway" accept): because the pose is a unipotent shear with `u.y ≡ 0`, the
//! PREV-pose position of a current hit is closed form in the hit point alone
//! — `p_prev = p + du·(a + b·p.y)`, `du = u_prev − u_cur` — so every MV
//! write reprojects the prev-POSE point (and FSR-RR's prev-Z lane rides the
//! same point): `SwayMv`/`mv_rows`/`prev_point` here, `render::
//! sway_prev_pos` on the CPU, `gbuf_write_hit`'s SWAY_MV arm + the per-chunk
//! `sway_dmv` ring (`SwayGpu::write_mv_rows`) on both GPU pipelines. The
//! prev sway clock PAIRS with each retained prev camera (main.rs `PrevPose`
//! — set after a successful present, cleared together, so the pair cannot
//! desync); pinned/frozen clocks have du = 0 STRUCTURALLY (bit-equal ⇒
//! `mv_rows` None / FLAG_SWAY_MV clear), which is what keeps every
//! pinned-clock gate bit-identical. Gated by --check's `sway-mv` cross-pose
//! oracle (with camera-only-imposter teeth) + --check-gpu/--check-dxr
//! wiring twins on foliage scenes.
//!
//! Known-accepts (the design doc's list): a converging still freezes
//! mid-gust, `--sw-rays` renders the rest pose (the HLSL software rays read
//! rest positions — sway MVs are SUPPRESSED there, `trace::sway_defs`, or
//! they would describe motion that is not on screen), incompatible with
//! `--heightfield` relief on leaf materials (the relief re-march reads
//! rest-space fields; both sweeps would compound on one box), and OIDN's
//! temporal mode still ghosts sway (reproject.rs reprojects from DEPTH, not
//! MVs — its own documented design). v0.4 RETIRES two v0.3.1 accepts: the
//! rigid billboard base slide and the curl's vertical sink/lift (`u.y ≡ 0`
//! + the rooted ramp).

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
/// (`wind`), never at split time — the `SwayCell` chord stays pure
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

/// Wind amplitude, in CONTENT diagonals (`Scene::content_min/max` — the
/// fireflies' scale rule: `Scene::diag` is ground-quad-inflated ~17× on
/// procedural scenes). The curl field's soft |v| < 1 normalization (and the
/// CONVEX octave blend, `SWAY_OCT2_K`) makes `SWAY_AMP_K · scale · mult` an
/// EXACT per-axis bound on the curl half of a cell's WIND vector `u` — a
/// vertex's displacement is `u · w_lin(y)`, so the top of the ramp band
/// moves by up to this and the content floor by exactly zero. Tuning
/// history, so the next retune has the trail: v0 0.010 read as an
/// earthquake ("~100× too much"); 0.0003 was the approved RIGID
/// translation; v0.4's rooted shear pinned the base (plant-AVERAGE
/// displacement roughly halves), so 0.001 kept TIP motion in the approved
/// band; 0.002 is the v0.6 whole-plant retune (user: v0.6's coherent
/// trunk-to-crown motion read "a little too subtle" — with the base pinned
/// and plants posing as one body, a 2× tip amplitude stays plausible where
/// the rigid-translation era's would have slid). `--foliage-amp` is the
/// live taste dial and the three-pose screenshot check the arbiter; the
/// constant keys the sidecar through the sweep pad (the sway_word/
/// CACHE_VERSION discipline — v18 carried this retune).
pub const SWAY_AMP_K: f32 = 0.002;
/// Per-axis flutter amplitude (hashed x/z sines per cell — the leaf-scale
/// shimmer the low-frequency curl field is too smooth to provide; per-cell
/// decorrelation is deliberate, leaves flutter independently). Rides the
/// same rooting ramp as the curl half, so a bigger bob can no longer slide
/// a billboard's base — the artifact that kept it tiny pre-v0.4. Doubled
/// with SWAY_AMP_K at v0.6's retune (constant ratio: 0.15 of the gust —
/// the shimmer scales with the sway or louder wind reads eerily rigid).
pub const SWAY_BOB_K: f32 = 0.0003;
/// How fast a cell's curl lookup point travels, in scales/second (the clouds
/// advect precedent: the field is static, the SAMPLE point moves — gusts
/// sweep across the canopy).
pub const SWAY_WIND_K: f32 = 0.03;
/// Octave-2 lookup speed, scales/second: the fine field scrolls 2.5× faster
/// than the gust octave, so nearby plants decorrelate in TIME as well as
/// space.
pub const SWAY_WIND2_K: f32 = 0.075;
/// The curl field's wavelength knob: the synthetic `Clouds` handed to
/// `curl_offset` gets `diag = SWAY_FIELD_K · scale`, so the field's spatial
/// wavelength is `~6.5 · SWAY_FIELD_K · scale` (clouds::CLOUD_CURL_SCALE_K) —
/// ~0.2 content diagonals: neighboring trees sway differently, one tree sways
/// coherently. Deliberately NOT a per-cell hash offset (the fireflies
/// decorrelator): adjacent cells of one canopy must move together or the
/// cell seams read as tearing.
pub const SWAY_FIELD_K: f32 = 0.03;
/// Octave-2 wavelength: ¼ the gust octave (~0.05 content diagonals —
/// comparable to the cell pitch, so ADJACENT cells genuinely differ). The
/// v0.4 "more random" half, chosen over a per-cell hash offset because a
/// hash makes adjacent-cell deltas O(full amp) and discontinuous (the
/// `SWAY_FIELD_K` tearing argument), while a finer SCROLLING field stays
/// continuous in space and time.
pub const SWAY_FIELD2_K: f32 = SWAY_FIELD_K / 4.0;
/// Octave-2 blend weight, CONVEX: `v = (1−k)·o1 + k·o2` keeps every axis of
/// the blended wind in [−1, 1], so the whole pad algebra survives the
/// second octave unchanged. 0 = pure gust field; the randomness retreat
/// dial if canopy seams ever read as tearing.
pub const SWAY_OCT2_K: f32 = 0.35;
/// Leaf-cell grid pitch, in content diagonals (doubled until the cell count
/// fits `MAX_CELLS` — every doubling halves per-tree motion resolution, never
/// correctness). v0.6: per REGION — the pitch multiplies each region's OWN
/// scale, and the coarsening loop doubles every region's pitch together.
pub const SWAY_CELL_K: f32 = 0.03;
/// Cell-count cap: bounds the per-frame instance rewrite + TLAS rebuild and
/// the one-time per-cell BLAS builds. ~4k instances is still deep inside the
/// paper's measured band (2.8M tets -> 9.66 ms; this is three orders less).
/// 2048 → 4096 with foliage v0.6: the WORLD's islands derive plants at their
/// own scale now, and its ~2.2k real trees must survive as PLANTS — a
/// demoted tree becomes a field member whose base rides the global ramp,
/// i.e. exactly the ungrounded-trunk look the plant machinery exists to fix.
/// The per-frame wind bake went rayon-parallel alongside (`bake`/`winds`),
/// so the CPU cost of the doubled cap stays flat.
pub const MAX_CELLS: usize = 4096;
/// Height band over which the ROOTING RAMP rises from the content floor:
/// `w(y) = clamp((y − cmin.y)/band, 0, 1)^SWAY_RAMP_GAMMA` — exactly 0 AT
/// the floor (a grass blade's base is pinned, v0.4's whole point), 1 above
/// `SWAY_HEIGHT_BAND` of the content height.
pub const SWAY_HEIGHT_BAND: f32 = 0.3;
/// The rooting ramp's exponent — CONCAVE (γ < 1) on purpose. A linear ramp
/// puts a 1 m grass tip in a ~30 m Minecraft band at w ≈ 0.03 and the grass
/// reads dead — the regression the retired SWAY_GROUND_K=0.3 floor was
/// added to fix in v0.3.1 (that floor is DELETED: per-vertex rooting
/// replaced its reason to exist, and its two documented artifacts — the
/// sliding billboard base, the curl sink/lift — die with it). γ = 0.5 puts
/// that tip at w ≈ 0.18 while the base stays exactly 0. Concavity is also
/// load-bearing for the per-cell CHORD linearization: chords of a concave
/// function never exceed it, so `w_max = w(y1)` is a true per-cell bound
/// (`self_test` pins the concavity so a future convex retune can't silently
/// break the pads).
pub const SWAY_RAMP_GAMMA: f32 = 0.5;
/// Numeric floor on the chord span, as a fraction of the ramp band: a
/// paper-thin floor-hugging cell would otherwise mint `b = Δw/Δy → ∞`
/// (sqrt's infinite slope at the floor), and huge `b` amplifies the
/// absolute-y cancellation in `a + b·y`. Flooring the DIVISOR (never the
/// stored y0/y1) caps `b`; the chord stays base-anchored, so a floored
/// cell's top merely under-sways (`w_lin(y1) < w1 = w_max` — coarser,
/// never unsound).
pub const SWAY_CHORD_SPAN_K: f32 = 0.01;
/// Absolute fp slack folded into `displacement_bound_with` (fractions of
/// scale, only for live cells): the affine evaluation `u·(a + b·y)` at
/// ABSOLUTE y carries cancellation-scale rounding (~ε·|b·y|) on both the
/// GPU's hardware instance-matrix path and the CPU's `o + t·d`
/// reconstruction; the pad absorbs it. Micro against SWAY_AMP_K (1e-6 vs
/// 1e-3 scales), so BVH quality is untouched.
pub const SWAY_PAD_EPS_K: f32 = 1e-6;

/// One sway cell: a plant-triangle bucket that becomes one BLAS chunk + one
/// animated TLAS instance. v0.4: carries its rooting CHORD — a vertex at
/// height y displaces by `u · w_lin(y)`, `w_lin(y) = a + b·y`. v0.5: the
/// cell is one bucket of the composite `(plant, voxel)` partition — cells
/// of one PLANT copy `anchor`/`a`/`b`/`key` BITWISE, which is the
/// no-tearing proof for connected trunks spanning several cells (equal pose
/// parameters ⇒ the equal affine map), while FIELD cells (plant-less grass/
/// flowers) keep the v0.4 per-voxel behavior bit-exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwayCell {
    /// Curl lookup point (world space, rest pose): the PLANT's anchor for
    /// plant cells (bitwise-shared across the plant — the coherence spine),
    /// the grid-voxel center for field cells. Never a chord y reference.
    pub anchor: Vec3A,
    /// Geometric min/max VERTEX y over the cell's member triangles
    /// (vertices, not centroids: centroid bucketing lets a triangle poke
    /// outside its voxel, and every y the shear ever touches must lie in
    /// [y0, y1] for `w_max` to bound it).
    pub y0: f32,
    pub y1: f32,
    /// The chord: `w_lin(y) = a + b·y`. Plant cells copy the PLANT's chord
    /// verbatim (rooted bitwise at the plant's OWN y0 — the potted-plant
    /// fix); field cells carry the v0.4 base-anchored chord of the global
    /// ramp over their own [y0, y1] (`a = w0 − b·y0`, so a floor-touching
    /// cell roots BITWISE — `a = −fl(b·y0)` and `−x + x` is exact). b ≥ 0
    /// both ways, so `w_lin` peaks at y1 on the cell.
    pub a: f32,
    pub b: f32,
    /// `fl(a + b·y1)` for plant cells (the chord's own value at the cell
    /// top, VERBATIM — never clamped, so it is bitwise-pinnable and
    /// provably ≥ every member's w_lin since b ≥ 0); `ramp(y1)` for field
    /// cells (a chord of the CONCAVE ramp never exceeds it). Either way the
    /// bound multiplier every pad reads: `|u|·w_max` bounds every vertex's
    /// displacement (0 = the whole cell sits at its root: `wind_with` bakes
    /// an exact ZERO).
    pub w_max: f32,
    /// Flutter/wind hash key: `PLANT_KEY_BIT | plant_id` for plant cells
    /// (bitwise-shared), the cell's own final index for field cells
    /// (< MAX_CELLS, so the bit separates the namespaces structurally; with
    /// zero plants this reproduces the v0.4 per-cell keying bit-exactly).
    pub key: u32,
    /// The cell's REGION content diagonal (v0.6) — the scale every length in
    /// `wind`/`displacement_bound_with` multiplies. Bitwise-shared across a
    /// plant's cells (one plant lives in one region), and bitwise-equal to
    /// the old scene-wide `SceneSway::scale` on region-less scenes (the
    /// implicit region's box IS the content box).
    pub scale: f32,
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
    /// ample: cells cap at `MAX_CELLS` = 4096 (BLAS cap-overflow RUNS are
    /// unbounded, but runs re-key onto these cells — `SwaySplit::cell_of`).
    pub tri_cell: Vec<u16>,
    /// Region-0's settled grid pitch — the startup line's number (v0.6: the
    /// scale frame is per CELL now, `SwayCell::scale`; multi-region scenes
    /// settle every region's pitch by the one shared multiplier, so this is
    /// representative, not exhaustive).
    pub cell: f32,
    /// Whole-plant pose groups in the partition (v0.5) — startup-line /
    /// diagnostics only.
    pub n_plants: u32,
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

/// One cell's frame pose, as the intersectors consume it: a vertex at
/// height y displaces by `u·(a + b·y)`. `u.y ≡ 0` (the `wind_with`
/// contract), which is what makes the map unipotent and `bvh::shear_ray`'s
/// inverse exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellShear {
    pub u: Vec3A,
    pub a: f32,
    pub b: f32,
}

/// The four non-identity slots of the row-major 3×4 instance matrix for
/// pose (u, a, b): `[m01, t.x, m21, t.z] = [u.x·b, u.x·a, u.z·b, u.z·a]`
/// (row 1 stays identity — `u.y ≡ 0`; `x' = x + u.x·b·y + u.x·a`, likewise
/// z). ONE function serves the GPU instance patch (`SwayGpu`) and the
/// CPU↔GPU pose gate in `self_test`, so the derivation cannot fork.
pub fn shear_rows(u: Vec3A, a: f32, b: f32) -> [f32; 4] {
    [u.x * b, u.x * a, u.z * b, u.z * a]
}

impl SceneSway {
    /// This frame's WIND vector for partition cell `c` — the intersector's
    /// read (3 relaxed loads; plain loads on x86). Lane 1 is always 0.0
    /// bits (`wind_with`'s u.y ≡ 0 contract, pinned by `self_test`).
    #[inline]
    pub fn wind(&self, c: u16) -> Vec3A {
        let o = &self.offsets[c as usize];
        Vec3A::new(
            f32::from_bits(o[0].load(Ordering::Relaxed)),
            f32::from_bits(o[1].load(Ordering::Relaxed)),
            f32::from_bits(o[2].load(Ordering::Relaxed)),
        )
    }

    /// The full frame pose for partition cell `c` — the gateway entry's read.
    #[inline]
    pub fn shear(&self, c: u16) -> CellShear {
        let cl = &self.cells[c as usize];
        CellShear { u: self.wind(c), a: cl.a, b: cl.b }
    }

    /// Every cell's current wind (gate/test convenience — never hot).
    pub fn offsets_snapshot(&self) -> Vec<Vec3A> {
        (0..self.cells.len()).map(|c| self.wind(c as u16)).collect()
    }
}

/// Compute the partition over SCENE triangle order. `None` when the mask
/// marks nothing. Deterministic, and bit-equal to the v0 packed-order
/// derivation: the distinct-key set is order-free (sorted + deduped), the
/// coarsening loop keys on distinct-count only, and cells follow sorted-key
/// order — `BlasPlan::packed_tris` was a permutation of this same set.
pub fn cell_partition(
    scene: &Scene,
    plant_mat: &[bool],
    plants: &PlantSet,
    regions: &[SwayRegion],
) -> Option<SceneSway> {
    let n = scene.indices.len();
    let is_masked = |t: usize| {
        plant_mat.get(scene.tri_mat[t] as usize).copied().unwrap_or(false)
            && region_of(regions, t as u32).is_some()
    };
    let plant_tris: Vec<u32> = (0..n as u32).filter(|&t| is_masked(t as usize)).collect();
    if plant_tris.is_empty() {
        return None;
    }
    debug_assert_eq!(plants.tri_plant.len(), n, "PlantSet built for a different scene");
    // The coarsening floor is one cell per plant + one field cell per
    // region; MAX_PLANTS = MAX_CELLS/2 and a handful of regions keep it
    // well under MAX_CELLS (the termination proof).
    debug_assert!(
        regions.len() < MAX_CELLS / 4,
        "region count breaks the coarsening floor"
    );
    let reg = region_scales(regions);
    let base_pitch: Vec<f32> =
        reg.iter().map(|r| (SWAY_CELL_K * r.scale).max(1e-6)).collect();
    let centroid = |t: u32| -> Vec3A {
        let [a, b, c] = scene.indices[t as usize];
        (scene.positions[a as usize] + scene.positions[b as usize] + scene.positions[c as usize])
            * (1.0 / 3.0)
    };
    // v0.5 composite key (plant-or-FIELD, voxel); v0.6 adds the REGION and
    // makes the voxel region-local (region-cmin anchor, region pitch × the
    // SHARED coarsening multiplier). FIELD_PLANT = u32::MAX sorts LAST, so
    // the cell order is plants 0..P in id order first, then field cells in
    // (region, voxel) order — on a region-less scene the implicit region is
    // 0 everywhere and the pre-region keying reproduces bit-exactly
    // (power-of-two multipliers scale the pitch EXACTLY, so `base·mult`
    // equals the old iteratively-doubled pitch bitwise).
    let key_at = |t: u32, mult: f32| -> (u32, u32, i32, i32, i32) {
        let r = region_of(regions, t).unwrap();
        let cell = base_pitch[r as usize] * mult;
        let q = (centroid(t) - regions[r as usize].cmin) * (1.0 / cell);
        (
            plants.tri_plant[t as usize],
            r,
            q.x.floor() as i32,
            q.y.floor() as i32,
            q.z.floor() as i32,
        )
    };
    // Grid pitch: start at SWAY_CELL_K of each region's scale and double
    // EVERY region's pitch until the distinct-cell count fits the cap
    // (coarser = fewer, bigger cells — never wrong; plants are keyed apart,
    // so the floor under infinite coarsening is one cell per plant + one
    // field cell per region — MAX_PLANTS + the region debug-assert cap that
    // below MAX_CELLS, which is the termination proof).
    let mut mult = 1.0f32;
    let keys: Vec<(u32, u32, i32, i32, i32)> = loop {
        let mut keys: Vec<(u32, u32, i32, i32, i32)> =
            plant_tris.iter().map(|&t| key_at(t, mult)).collect();
        keys.sort_unstable();
        keys.dedup();
        if keys.len() <= MAX_CELLS {
            break keys;
        }
        mult *= 2.0;
    };
    // One pass fills tri_cell AND accumulates each cell's geometric VERTEX
    // y-extent — the pad's domain (vertices, not centroids: a tri bucketed
    // by centroid can poke outside its voxel, and w_max must bound every y
    // the shear touches).
    let mut tri_cell = vec![STATIC_CELL; n];
    let mut ymin = vec![f32::INFINITY; keys.len()];
    let mut ymax = vec![f32::NEG_INFINITY; keys.len()];
    for &t in &plant_tris {
        // keys is sorted + deduped, so the index IS the cell id.
        let k = key_at(t, mult);
        let c = keys.binary_search(&k).expect("plant key must be in the partition");
        tri_cell[t as usize] = c as u16;
        for &vi in &scene.indices[t as usize] {
            let y = scene.positions[vi as usize].y;
            ymin[c] = ymin[c].min(y);
            ymax[c] = ymax[c].max(y);
        }
    }
    let cells: Vec<SwayCell> = keys
        .iter()
        .enumerate()
        .map(|(c, k)| {
            let (y0, y1) = (ymin[c], ymax[c]);
            debug_assert!(y0.is_finite() && y1 >= y0, "cell {c} has no member vertices");
            let rg = &regions[k.1 as usize];
            let scale = reg[k.1 as usize].scale;
            if k.0 != FIELD_PLANT {
                // PLANT cell: the plant's anchor/chord/key BITWISE (the
                // coherence spine — equal pose parameters across the
                // plant's cells ⇒ the equal affine map ⇒ no tearing);
                // w_max = the chord's own value at the cell top, VERBATIM
                // (never clamped: bitwise-pinnable, and provably >= every
                // member's w_lin since b >= 0 and y <= y1). `scale` is the
                // plant's region's — bitwise-shared like the chord (one
                // plant, one region).
                let p = &plants.plants[k.0 as usize];
                SwayCell {
                    anchor: p.anchor,
                    y0,
                    y1,
                    a: p.a,
                    b: p.b,
                    w_max: p.a + p.b * y1,
                    key: PLANT_KEY_BIT | k.0,
                    scale,
                }
            } else {
                // FIELD cell: the v0.4 arm — voxel-center anchor,
                // base-anchored chord of the REGION ramp over [y0, y1]. The
                // divisor is floored (never the stored extent) so a
                // paper-thin floor cell can't mint b → ∞; a floored chord
                // under-sways the cell top — coarser, never unsound.
                let cell = base_pitch[k.1 as usize] * mult;
                let anchor = rg.cmin
                    + Vec3A::new(
                        (k.2 as f32 + 0.5) * cell,
                        (k.3 as f32 + 0.5) * cell,
                        (k.4 as f32 + 0.5) * cell,
                    );
                let w0 = ramp(y0, rg.cmin.y, rg.cmax.y);
                let w1 = ramp(y1, rg.cmin.y, rg.cmax.y);
                let span_floor = SWAY_CHORD_SPAN_K
                    * (SWAY_HEIGHT_BAND * (rg.cmax.y - rg.cmin.y)).max(1e-6);
                let b = if y1 > y0 { (w1 - w0) / (y1 - y0).max(span_floor) } else { 0.0 };
                let a = w0 - b * y0;
                SwayCell { anchor, y0, y1, a, b, w_max: w1, key: c as u32, scale }
            }
        })
        .collect();
    let offsets = (0..cells.len()).map(|_| [(); 3].map(|_| AtomicU32::new(0))).collect();
    Some(SceneSway {
        cells,
        tri_cell,
        cell: base_pitch[0] * mult,
        n_plants: plants.plants.len() as u32,
        offsets,
        baked: AtomicU32::new(f32::NAN.to_bits()),
    })
}

/// Attach/refresh `Scene::sway` under the session predicate. Called at every
/// scene-load site (cold OBJ/glTF, warm sidecar, world islands + merge) and
/// by the live-edit path BEFORE each BVH rebuild — appended triangles must
/// lengthen `tri_cell` or the intersector indexes out of bounds. v0.5: the
/// mask is leaf OR woody, and `derive_plants` groups trunks + attached
/// canopies into whole-plant pose groups first; the startup line's plant
/// count is the runtime measurement of the exporter-weld assumption (one
/// merged multi-tree mesh = one plant — degraded, not broken).
pub fn attach(scene: &mut Scene) {
    scene.sway = None;
    if !sweep_armed() {
        return;
    }
    let mask = plant_materials(scene);
    let woody = woody_materials(scene);
    let regions = regions_of(scene);
    let plants = derive_plants(scene, &mask, &woody, &regions);
    scene.sway = cell_partition(scene, &mask, &plants, &regions).map(Box::new);
    if let Some(sw) = &scene.sway {
        let is_woody =
            |t: usize| woody.get(scene.tri_mat[t] as usize).copied().unwrap_or(false);
        let (mut w, mut l, mut f) = (0u64, 0u64, 0u64);
        for (t, (&c, &p)) in sw.tri_cell.iter().zip(&plants.tri_plant).enumerate() {
            if c == STATIC_CELL {
                continue;
            }
            if is_woody(t) {
                w += 1;
            } else if p != FIELD_PLANT {
                l += 1;
            } else {
                f += 1;
            }
        }
        eprintln!(
            "foliage: {} plants ({w} woody tris, {l} attached leaf tris, {f} field tris) \
             -> {} cells (pitch {:.3}{})",
            sw.n_plants,
            sw.cells.len(),
            sw.cell,
            if regions.len() > 1 {
                format!(", {} regions", regions.len())
            } else {
                String::new()
            }
        );
    }
}

/// Bake this frame's per-cell WIND vectors (bit-equal clock = free — a
/// converging still's frozen clock costs nothing). Takes `&self`
/// (relaxed-atomic stores) but MUST only run between traces on the main
/// thread — the accum-buffer discipline the field doc states. Each cell's
/// wind is `wind(cells[c], ..)` — the hash key and curl anchor ride IN the
/// cell (v0.5), so the GPU ring's per-run copies of the same cells produce
/// bit-equal poses through the same pure function (the chord (a, b) is
/// build-time cell data, shared by construction).
pub fn bake(sway: &SceneSway, time: f32) {
    if sway_abl().rest {
        return; // cost probe: offsets stay all-zero (the rest pose)
    }
    if sway.baked.load(Ordering::Relaxed) == time.to_bits() {
        return;
    }
    // Rayon-parallel since the MAX_CELLS doubling (v0.6): each index writes
    // its own atomics, `wind` is pure — determinism and the accum-buffer
    // discipline both hold (still main-thread, still between traces).
    use rayon::prelude::*;
    (0..sway.cells.len()).into_par_iter().for_each(|i| {
        let u = wind(&sway.cells[i], time);
        let o = &sway.offsets[i];
        o[0].store(u.x.to_bits(), Ordering::Relaxed);
        o[1].store(u.y.to_bits(), Ordering::Relaxed);
        o[2].store(u.z.to_bits(), Ordering::Relaxed);
    });
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
    /// The grid pitch the split settled on (region-0's, see
    /// `SceneSway::cell`) — the startup line's number.
    pub cell: f32,
}

/// Per-run bake for the GPU ring. The hash key and anchor ride IN each
/// run's `SwayCell` copy (v0.5), so the re-key contract is structural: runs
/// of one cap-overflow cell are copies of one cell (same key/anchor/chord)
/// AND cells of one plant share key/anchor/chord — both pose bit-equal to
/// the CPU `bake` through the same pure function.
pub fn winds(cells: &[SwayCell], time: f32) -> Vec<Vec3A> {
    // Indexed par map — order-preserving, so determinism holds (v0.6: the
    // scale rides IN each cell; rayon for the MAX_CELLS doubling).
    use rayon::prelude::*;
    cells.par_iter().map(|c| wind(c, time)).collect()
}

/// Per-PARTITION-cell prev−cur shear-row deltas for one frame's MV write —
/// the sway half of "proper motion vectors" (the fireflies round-3 shape,
/// but into the MAIN MV planes: sway is true surface motion, exactly what
/// the upscalers are trained on). Because the pose is a unipotent shear
/// with `u.y ≡ 0`, the PREV-pose position of a CURRENT hit point is closed
/// form in the hit point alone — no rest position, no barycentrics:
/// `p_prev = p + du·(a + b·p.y)`, `du = u_prev − u_cur`. `rows[c]` is
/// `shear_rows(du, a, b)`, applied by `prev_point`.
pub struct SwayMv {
    pub rows: Vec<[f32; 4]>,
}

/// Build one frame's MV deltas: `rows[c] = shear_rows(wind(c, t_prev) −
/// wind(c, t_cur), a, b)`. `None` on bit-equal clocks (the `bake` fast-path
/// idiom) — a frozen still / pinned gate has du = 0 STRUCTURALLY, which is
/// what keeps every pinned-clock MV gate bit-identical: the consumer never
/// even branches on a zero delta, it takes the pre-feature path outright.
/// Built as `shear_rows(u_prev − u_cur, ..)` — ONE derivation shared with
/// the GPU instance patch, never `rows(u_prev) − rows(u_cur)` — so du = 0
/// yields exact zero rows and the linearity pin below stays a documentation
/// of equivalence, not a load-bearing identity. Pure function of the two
/// clocks; zero rng draws (every same-seed/replay contract holds).
pub fn mv_rows(sway: &SceneSway, t_cur: f32, t_prev: f32) -> Option<SwayMv> {
    if t_cur.to_bits() == t_prev.to_bits() {
        return None;
    }
    use rayon::prelude::*;
    let rows = sway
        .cells
        .par_iter()
        .map(|c| {
            let du = wind(c, t_prev) - wind(c, t_cur);
            shear_rows(du, c.a, c.b)
        })
        .collect();
    Some(SwayMv { rows })
}

/// Apply one cell's prev−cur rows to a CURRENT-pose point (exact: y is
/// shear-invariant, so the rows evaluate at the hit's own y verbatim). The
/// ONE Rust-side application site — the MV write (`render.rs`) and the
/// cross-pose gate oracle both call this, so the derivation cannot fork;
/// `trace_common.hlsli`'s `gbuf_write_hit` SWAY_MV arm is its term-for-term
/// HLSL twin.
#[inline]
pub fn prev_point(rows: &[f32; 4], p: Vec3A) -> Vec3A {
    Vec3A::new(p.x + rows[0] * p.y + rows[1], p.y, p.z + rows[2] * p.y + rows[3])
}

/// Synthetic all-foliage micro-scene, `per` tiny tris per cluster — ONE
/// construction serving `self_test`'s gateway/displaced-hit pins AND
/// main.rs's sway-MV cross-pose gate (`sway_mv_check`), so the two gates
/// always exercise the same partition shape. `attach` runs, so `sway` is
/// Some whenever the session lever arms (an armed run on this scene always
/// partitions: every tri is leaf-classed).
pub fn synth_sway_scene(clusters: &[Vec3A], per: usize) -> Scene {
    let mut b = crate::scene::SceneBuilder::new();
    // alpha_masked arms the leaf predicate; the OPAQUE texel keeps the
    // cutout from rejecting the displaced-hit pin's ray.
    let t = b.add_texture(crate::texture::Texture {
        w: 1,
        h: 1,
        texels: vec![[255, 255, 255, 255]],
        alpha_masked: true,
        srgb: true,
        source: String::new(),
        h2n: false,
        n2h: false,
        normal_role: false,
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

/// Per-material WOODY mask (v0.5 whole-plant sway): the bark class byte
/// alone — deliberately NO alpha leg (bark is opaque by nature, and on
/// vokselia there is no alpha signal at all: the name-classified byte is the
/// whole vocabulary, the `leaf_materials` Minecraft argument inverted).
/// Foliage-classed-but-opaque stragglers (the old "bark stays static" leg)
/// now classify `bark` upstream in matclass; anything foliage-classed and
/// opaque that remains stays static — unchanged v0.4 semantics.
pub fn woody_materials(scene: &Scene) -> Vec<bool> {
    scene.materials.iter().map(|m| m.class == crate::matclass::IDX_BARK as u8).collect()
}

/// Sway participation (what `attach`/`derive_plants`/`cell_partition`
/// consume): leaf OR woody.
pub fn plant_materials(scene: &Scene) -> Vec<bool> {
    leaf_materials(scene)
        .into_iter()
        .zip(woody_materials(scene))
        .map(|(l, w)| l || w)
        .collect()
}

/// One sway REGION (foliage v0.6): a tri range + ITS OWN content box, the
/// scale frame every sway length inside it derives from — contact tolerance
/// (`PLANT_MERGE_K`), cell pitch (`SWAY_CELL_K`), the rooting-ramp band
/// (`SWAY_HEIGHT_BAND` of the REGION's height), curl wavelength/scroll
/// (`SWAY_FIELD_K`/`SWAY_WIND_K`) and amplitude (`SWAY_AMP_K` — all via
/// `SwayCell::scale`). `Scene::sway_regions` holds one per world island
/// (`world::merge_scenes`); empty = one implicit whole-scene region, which
/// reproduces the pre-region partition BIT-EXACTLY (same keys modulo a
/// constant region component, same cells, same scale). Ranges must be
/// ascending and disjoint (`region_of`'s binary search assumes it; the world
/// sidecar's read side validates before trusting a deserialized list).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwayRegion {
    pub tri_start: u32,
    /// Exclusive.
    pub tri_end: u32,
    pub cmin: Vec3A,
    pub cmax: Vec3A,
}

/// The scene's region list, with the implicit whole-scene fallback — the ONE
/// place the empty-list convention is interpreted.
fn regions_of(scene: &Scene) -> Vec<SwayRegion> {
    if scene.sway_regions.is_empty() {
        vec![SwayRegion {
            tri_start: 0,
            tri_end: scene.indices.len() as u32,
            cmin: scene.content_min,
            cmax: scene.content_max,
        }]
    } else {
        scene.sway_regions.clone()
    }
}

/// Which region a triangle lives in (`None` = outside every region — e.g.
/// the world's covering ground quad; such a tri never participates in sway).
#[inline]
fn region_of(regions: &[SwayRegion], t: u32) -> Option<u32> {
    let i = regions.partition_point(|r| r.tri_end <= t);
    (i < regions.len() && regions[i].tri_start <= t).then_some(i as u32)
}

/// Per-region derived scales, computed once per attach: the region content
/// diagonal (the length frame) and the proximity-merge contact pitch.
struct RegScale {
    scale: f32,
    /// Contact pitch `h` (see `PLANT_MERGE_K`).
    h: f32,
    /// `0.5·h` — the AABB inflation both contact predicates share.
    pad: f32,
}

fn region_scales(regions: &[SwayRegion]) -> Vec<RegScale> {
    regions
        .iter()
        .map(|r| {
            let scale = (r.cmax - r.cmin).length().max(1e-3);
            let h = (PLANT_MERGE_K * scale).max(1e-6);
            RegScale { scale, h, pad: 0.5 * h }
        })
        .collect()
}

/// `tri_plant` sentinel: a masked triangle that belongs to no plant — a
/// FIELD member, animated exactly like v0.4 (per-voxel cell, global ramp).
pub const FIELD_PLANT: u32 = u32::MAX;
/// `SwayCell::key` bit separating plant keys from field-cell ordinals: field
/// ordinals are final cell indices < `MAX_CELLS` = 4096, so the namespaces
/// are structurally collision-free.
pub const PLANT_KEY_BIT: u32 = 1 << 31;
/// Proximity-merge CONTACT tolerance, in content diagonals. Two boxes merge
/// (or a leaf attaches) iff their 0.5·h-inflated AABBs overlap — i.e. the
/// gap is under one h. This must be CONTACT scale, not cell scale: the
/// first cut used the 0.03 sway-voxel pitch, and on rungholt (where 0.03 ·
/// diag ≈ 25 Minecraft blocks) it fused the city's every forest into 4
/// plants with 2.3M attached leaves. 0.002 · diag ≈ 1.7 blocks there ≈
/// 2 cm on a diag-10 scene — adjacent Log/leaf blocks and welded
/// trunk-piece seams merge, distinct trees don't.
pub const PLANT_MERGE_K: f32 = 0.002;
/// A component is LARGE when its AABB raster at the contact pitch would
/// exceed this many voxels (a welded multi-tree trunk mesh, a whole-canopy
/// leaf mesh). Large components skip the voxel grid and take exact
/// pairwise AABB tests instead — few of them exist by construction, so the
/// pairwise arm is cheap while the grid arm stays bounded.
pub const PLANT_LARGE_VOX: u64 = 32 * 32 * 32;
/// Hard plant cap. Load-bearing for the partition's coarsening loop: plants
/// are keyed apart, so under infinite voxel coarsening the partition floor
/// is one cell per plant plus the field cells — capping plants at half of
/// `MAX_CELLS` proves the doubling loop terminates. Overflow demotes the
/// smallest plants to FIELD membership (coarser, never wrong) with a loud
/// line.
pub const MAX_PLANTS: usize = MAX_CELLS / 2;

/// One pose group (v0.5): a trunk-connected woody component set plus its
/// proximity-attached leaf components. Every cell of this plant copies
/// `anchor`/`a`/`b` BITWISE — equal pose parameters mean the equal affine
/// map, which is the no-tearing proof for connected opaque geometry spanning
/// several cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plant {
    /// Curl lookup point: (aabb-center.x, y0, aabb-center.z) of the member
    /// union — the trunk-base position, world rest space.
    pub anchor: Vec3A,
    /// Min/max vertex y over ALL member components (woody + attached leaf —
    /// finalized AFTER leaf attach, so the chord covers the canopy).
    pub y0: f32,
    pub y1: f32,
    /// The plant chord, rooted at the plant's OWN base: `w_max` = the GLOBAL
    /// ramp at y1 (tall plants move more — the v0.4 scale rule preserved),
    /// `b = w_max / max(y1 − y0, span_floor)`, `a = 0.0 − b·y0` (bitwise
    /// root: `fl(a + b·y0) == 0.0` exactly, the −x + x argument from
    /// `SwayCell::a` — and the base is the PLANT's, which retires the v0.4
    /// "potted plants root to the scene floor" known-accept).
    pub a: f32,
    pub b: f32,
    pub w_max: f32,
    /// Member triangle count — diagnostics + the demotion sort key.
    pub tris: u32,
}

/// `derive_plants`' product: the plant list plus the scene-tri → plant map.
pub struct PlantSet {
    pub plants: Vec<Plant>,
    /// Scene-tri → plant id; `FIELD_PLANT` = field member or non-participant.
    pub tri_plant: Vec<u32>,
}

impl PlantSet {
    /// The zero-plant set — every masked tri a field member. The synthetic
    /// self_test arms use it, and it is the structural v0.4-equivalence
    /// case: with no plants, `cell_partition` reproduces the v0.4 grid,
    /// chords, anchors and flutter keys bit-exactly.
    pub fn field_only(n_tris: usize) -> PlantSet {
        PlantSet { plants: Vec::new(), tri_plant: vec![FIELD_PLANT; n_tris] }
    }
}

/// Group the masked triangles into PLANTS (v0.5). Deterministic at every
/// step — no HashMap ever decides an id, an order, or an fp accumulation
/// order (component ids feed key bytes and chord fp, so: index-keyed vecs,
/// ascending scan orders, and min-id union representatives throughout):
///
/// 1. Union-find over shared VERTEX indices of masked tris (the
///    `reclassify_spray` pattern) — trunks/branches are connected meshes and
///    come out as components; disconnected leaf cards/blocks stay singleton
///    components (vertices are never welded across tobj models, so
///    components live within one model by construction).
/// 2. A component is WOODY iff any member tri's material is bark-classed.
/// 3. Woody components whose inflated AABBs share a `PLANT_MERGE_K` grid
///    voxel merge into one plant (trunk + branch meshes; adjacent Minecraft
///    Log blocks).
/// 4. Leaf components attach to the overlapping plant with the largest
///    overlap volume (ties → smallest plant id); leaf components touching NO
///    plant stay FIELD.
/// 5. Plant extents/chords are finalized over the FULL member set.
///
/// Cost: one O(V) parent array + two passes over the masked tris — the
/// `reclassify_spray` bill; like it, this runs per world island AND once on
/// the merged world.
///
/// v0.6 REGIONS: every proximity length (`h`, `h_p`) and the ramp band are
/// the tri's REGION's (see `SwayRegion`), and merges/attaches never cross a
/// region boundary — structurally (grid keys carry the region id, the
/// pairwise arms filter on it), not merely because islands are far apart.
/// The union-find itself stays GLOBAL (regions can't share vertices — the
/// world merge rebases each part's index range — so one pass costs one
/// parent array instead of one per region); a component's region is its
/// first tri's. Masked tris OUTSIDE every region (only possible with an
/// explicit list) stay static.
pub fn derive_plants(
    scene: &Scene,
    mask: &[bool],
    woody_mat: &[bool],
    regions: &[SwayRegion],
) -> PlantSet {
    let n = scene.indices.len();
    let nv = scene.positions.len();
    let masked = |t: usize| {
        mask.get(scene.tri_mat[t] as usize).copied().unwrap_or(false)
            && region_of(regions, t as u32).is_some()
    };
    let mut tri_plant = vec![FIELD_PLANT; n];

    // 1. Vertex union-find (path halving, ascending tri order).
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
        if !masked(t) {
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
        return PlantSet { plants: Vec::new(), tri_plant };
    }

    // 2. Components in first-tri order (dense ids via an index-keyed array —
    // never a HashMap), accumulating AABB / tri count / woody flag in
    // ascending-tri fp order.
    let mut comp_of_root = vec![u32::MAX; nv];
    let mut comp_min: Vec<Vec3A> = Vec::new();
    let mut comp_max: Vec<Vec3A> = Vec::new();
    let mut comp_tris: Vec<u32> = Vec::new();
    let mut comp_woody: Vec<bool> = Vec::new();
    // The component's region = its FIRST tri's. Components cannot span
    // regions (disjoint vertex ranges per merged part), which the debug
    // build asserts below.
    let mut comp_region: Vec<u32> = Vec::new();
    let mut tri_comp = vec![u32::MAX; n];
    for (t, idx) in scene.indices.iter().enumerate() {
        if !masked(t) {
            continue;
        }
        let root = find(&mut parent, idx[0]);
        let c = if comp_of_root[root as usize] == u32::MAX {
            let c = comp_min.len() as u32;
            comp_of_root[root as usize] = c;
            comp_min.push(Vec3A::INFINITY);
            comp_max.push(Vec3A::NEG_INFINITY);
            comp_tris.push(0);
            comp_woody.push(false);
            comp_region.push(region_of(regions, t as u32).unwrap());
            c
        } else {
            debug_assert_eq!(
                comp_region[comp_of_root[root as usize] as usize],
                region_of(regions, t as u32).unwrap(),
                "connected component spans two sway regions"
            );
            comp_of_root[root as usize]
        };
        for &vi in idx {
            let p = scene.positions[vi as usize];
            comp_min[c as usize] = comp_min[c as usize].min(p);
            comp_max[c as usize] = comp_max[c as usize].max(p);
        }
        comp_tris[c as usize] += 1;
        comp_woody[c as usize] |=
            woody_mat.get(scene.tri_mat[t] as usize).copied().unwrap_or(false);
        tri_comp[t] = c;
    }
    let n_comp = comp_min.len();

    // 3. Woody proximity merge — HYBRID for bounded cost at the CONTACT
    // pitch: SMALL components (block faces, branch meshes) take a grid
    // closure — rasterize each 0.5·h-inflated AABB onto the cmin-anchored
    // grid, components sharing a voxel union (touching/overlapping boxes
    // ALWAYS share a voxel after the inflation: the overlap band is >= one
    // pitch wide, so a full grid plane crosses it); LARGE components
    // (welded multi-tree meshes, whose raster at contact pitch would
    // explode — few by construction) take exact pairwise inflated-AABB
    // tests against EVERY woody component instead. Both arms decide by the
    // same predicate (inflated boxes overlap ⇔ gap < h), so the hybrid
    // moves cost, never the partition. The union rule `parent[max] = min`
    // makes the final representative of every merged group its smallest
    // component id regardless of encounter order — the closure of a
    // symmetric relation is order-independent as a partition, and min-id
    // makes the representative so too. The voxel map may be a HashMap:
    // collisions only trigger unions (iteration-order-free).
    let reg = region_scales(regions);
    let inflated = |c: usize| {
        let pad = reg[comp_region[c] as usize].pad;
        (comp_min[c] - Vec3A::splat(pad), comp_max[c] + Vec3A::splat(pad))
    };
    let vox_span = |c: usize, pitch: f32| -> u64 {
        let e = (comp_max[c] - comp_min[c]) / pitch;
        (e.x.ceil().max(0.0) as u64 + 2)
            .saturating_mul(e.y.ceil().max(0.0) as u64 + 2)
            .saturating_mul(e.z.ceil().max(0.0) as u64 + 2)
    };
    let key_range = |lo: f32, hi: f32, org: f32, pitch: f32| -> (i32, i32) {
        ((((lo - org) / pitch).floor()) as i32, (((hi - org) / pitch).floor()) as i32)
    };
    let mut cparent: Vec<u32> = (0..n_comp as u32).collect();
    // Voxel keys carry the REGION id (region-local pitch + anchor), so
    // cross-region merges are structurally impossible — not merely unlikely
    // because islands sit far apart.
    let mut voxel_owner: std::collections::HashMap<(u32, i32, i32, i32), u32> =
        std::collections::HashMap::new();
    let mut large_woody: Vec<usize> = Vec::new();
    for c in 0..n_comp {
        if !comp_woody[c] {
            continue;
        }
        let r = comp_region[c] as usize;
        let h = reg[r].h;
        if vox_span(c, h) > PLANT_LARGE_VOX {
            large_woody.push(c);
            continue;
        }
        let org = regions[r].cmin;
        let (lo, hi) = inflated(c);
        let (x0, x1) = key_range(lo.x, hi.x, org.x, h);
        let (y0, y1) = key_range(lo.y, hi.y, org.y, h);
        let (z0, z1) = key_range(lo.z, hi.z, org.z, h);
        for x in x0..=x1 {
            for y in y0..=y1 {
                for z in z0..=z1 {
                    match voxel_owner.entry((r as u32, x, y, z)) {
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(c as u32);
                        }
                        std::collections::hash_map::Entry::Occupied(e) => {
                            let ra = find(&mut cparent, *e.get());
                            let rb = find(&mut cparent, c as u32);
                            if ra != rb {
                                cparent[ra.max(rb) as usize] = ra.min(rb);
                            }
                        }
                    }
                }
            }
        }
    }
    let boxes_overlap = |a: (Vec3A, Vec3A), b: (Vec3A, Vec3A)| -> bool {
        a.0.max(b.0).cmple(a.1.min(b.1)).all()
    };
    for &lc in &large_woody {
        let lb = inflated(lc);
        for c in 0..n_comp {
            if c == lc || !comp_woody[c] || comp_region[c] != comp_region[lc] {
                continue;
            }
            if boxes_overlap(lb, inflated(c)) {
                let ra = find(&mut cparent, lc as u32);
                let rb = find(&mut cparent, c as u32);
                if ra != rb {
                    cparent[ra.max(rb) as usize] = ra.min(rb);
                }
            }
        }
    }

    // 4. Plant ids: merged woody groups sorted by representative (= the min
    // member component id — ascending scan makes the order a pure function
    // of the mesh).
    let mut plant_of_comp = vec![FIELD_PLANT; n_comp];
    let mut rep_to_plant = vec![u32::MAX; n_comp];
    let mut plant_region: Vec<u32> = Vec::new();
    let mut n_plants = 0u32;
    for c in 0..n_comp {
        if !comp_woody[c] {
            continue;
        }
        let rep = find(&mut cparent, c as u32);
        if rep_to_plant[rep as usize] == u32::MAX {
            rep_to_plant[rep as usize] = n_plants;
            // Merges never cross regions, so any member names the plant's.
            plant_region.push(comp_region[c]);
            n_plants += 1;
        }
        plant_of_comp[c] = rep_to_plant[rep as usize];
    }

    // Per-plant WOODY member-union AABB (ascending component order — the
    // fixed fp accumulation order) for the leaf-attach overlap scores.
    let mut plant_min = vec![Vec3A::INFINITY; n_plants as usize];
    let mut plant_max = vec![Vec3A::NEG_INFINITY; n_plants as usize];
    for c in 0..n_comp {
        let p = plant_of_comp[c];
        if p != FIELD_PLANT {
            plant_min[p as usize] = plant_min[p as usize].min(comp_min[c]);
            plant_max[p as usize] = plant_max[p as usize].max(comp_max[c]);
        }
    }

    // 5. Leaf attach. Candidates come from a grid over the PLANT boxes at a
    // COARSER pitch h_p bounded by the largest plant (a coarser candidate
    // voxel can only ADD candidates, never lose one: boxes overlapping
    // after the 0.5·h contact inflation also share an h_p voxel after
    // 0.5·h_p >= 0.5·h inflation) — the EXACT contact test then decides:
    // attach to the plant with the largest overlap volume of the
    // 0.5·h-inflated boxes, requiring vol > 0 (genuine contact within h;
    // ties → smallest plant id, cand ascending). Leaf components whose own
    // raster would blow the voxel cap (welded whole-canopy meshes — few)
    // brute-force the plant list instead, same predicate. Per non-woody
    // component, each decision is a pure function of the component — order
    // free, kept ascending anyway.
    // Candidate pitch per REGION, bounded by that region's largest plant.
    let mut largest_ext = vec![0.0f32; regions.len()];
    for p in 0..n_plants as usize {
        let e = plant_max[p] - plant_min[p];
        let r = plant_region[p] as usize;
        largest_ext[r] = largest_ext[r].max(e.x.max(e.y).max(e.z));
    }
    let h_ps: Vec<f32> = (0..regions.len())
        .map(|r| reg[r].h.max(largest_ext[r] / 8.0).max(1e-6))
        .collect();
    let mut voxel_plants: std::collections::HashMap<(u32, i32, i32, i32), Vec<u32>> =
        std::collections::HashMap::new();
    for p in 0..n_plants as usize {
        let r = plant_region[p] as usize;
        let (h_p, pad_p, org) = (h_ps[r], 0.5 * h_ps[r], regions[r].cmin);
        let (lo, hi) =
            (plant_min[p] - Vec3A::splat(pad_p), plant_max[p] + Vec3A::splat(pad_p));
        let (x0, x1) = key_range(lo.x, hi.x, org.x, h_p);
        let (y0, y1) = key_range(lo.y, hi.y, org.y, h_p);
        let (z0, z1) = key_range(lo.z, hi.z, org.z, h_p);
        for x in x0..=x1 {
            for y in y0..=y1 {
                for z in z0..=z1 {
                    // Pushed in ascending plant order — per-voxel lists come
                    // out sorted by construction.
                    voxel_plants.entry((r as u32, x, y, z)).or_default().push(p as u32);
                }
            }
        }
    }
    let mut cand: Vec<u32> = Vec::new();
    for c in 0..n_comp {
        if comp_woody[c] {
            continue;
        }
        let r = comp_region[c] as usize;
        let (h_p, pad_p, org, pad) = (h_ps[r], 0.5 * h_ps[r], regions[r].cmin, reg[r].pad);
        cand.clear();
        if vox_span(c, h_p) > PLANT_LARGE_VOX {
            // Brute-force arm: same-region plants only — the grid arm's
            // region-keyed voxels made cross-region attach impossible, and
            // this arm must decide by the same predicate set.
            cand.extend((0..n_plants).filter(|&p| plant_region[p as usize] == r as u32));
        } else {
            let (lo, hi) =
                (comp_min[c] - Vec3A::splat(pad_p), comp_max[c] + Vec3A::splat(pad_p));
            let (x0, x1) = key_range(lo.x, hi.x, org.x, h_p);
            let (y0, y1) = key_range(lo.y, hi.y, org.y, h_p);
            let (z0, z1) = key_range(lo.z, hi.z, org.z, h_p);
            for x in x0..=x1 {
                for y in y0..=y1 {
                    for z in z0..=z1 {
                        if let Some(ps) = voxel_plants.get(&(r as u32, x, y, z)) {
                            cand.extend_from_slice(ps);
                        }
                    }
                }
            }
            cand.sort_unstable();
            cand.dedup();
        }
        let (lo, hi) = inflated(c);
        let mut best: Option<(f32, u32)> = None;
        for &p in &cand {
            let plo = plant_min[p as usize] - Vec3A::splat(pad);
            let phi = plant_max[p as usize] + Vec3A::splat(pad);
            let o = (hi.min(phi) - lo.max(plo)).max(Vec3A::ZERO);
            let vol = o.x * o.y * o.z;
            // vol > 0 = genuine contact; strictly-greater keeps the
            // smallest id on ties (cand is ascending).
            if vol > 0.0 && best.map_or(true, |(bv, _)| vol > bv) {
                best = Some((vol, p));
            }
        }
        if let Some((_, p)) = best {
            plant_of_comp[c] = p;
        }
    }

    // 6. Finalize plants over the FULL member set, fixed member order
    // (components ascending — one pass covers woody and attached leaf alike
    // since plant_of_comp is now total). The ramp band and chord span floor
    // are the plant's REGION's (v0.6) — a Minecraft tree's `w_max` reads the
    // island's height band, not the world's.
    let mut full_min = vec![Vec3A::INFINITY; n_plants as usize];
    let mut full_max = vec![Vec3A::NEG_INFINITY; n_plants as usize];
    let mut plant_tris = vec![0u32; n_plants as usize];
    for c in 0..n_comp {
        let p = plant_of_comp[c];
        if p != FIELD_PLANT {
            full_min[p as usize] = full_min[p as usize].min(comp_min[c]);
            full_max[p as usize] = full_max[p as usize].max(comp_max[c]);
            plant_tris[p as usize] += comp_tris[c];
        }
    }
    let mut plants: Vec<Plant> = (0..n_plants as usize)
        .map(|p| {
            let r = &regions[plant_region[p] as usize];
            let (cy0, cy1) = (r.cmin.y, r.cmax.y);
            let span_floor =
                SWAY_CHORD_SPAN_K * (SWAY_HEIGHT_BAND * (cy1 - cy0)).max(1e-6);
            let (lo, hi) = (full_min[p], full_max[p]);
            let (y0, y1) = (lo.y, hi.y);
            let center = 0.5 * (lo + hi);
            let w_max = ramp(y1, cy0, cy1);
            let b = if y1 > y0 { w_max / (y1 - y0).max(span_floor) } else { 0.0 };
            let a = 0.0 - b * y0;
            Plant {
                anchor: Vec3A::new(center.x, y0, center.z),
                y0,
                y1,
                a,
                b,
                w_max,
                tris: plant_tris[p],
            }
        })
        .collect();

    // 7. tri_plant off the component map.
    for t in 0..n {
        let c = tri_comp[t];
        if c != u32::MAX {
            tri_plant[t] = plant_of_comp[c as usize];
        }
    }

    // 8. Overflow demotion: cap the plant count so the partition's
    // coarsening loop provably converges (see MAX_PLANTS). Demote the
    // smallest plants (tris asc, id asc) to FIELD, reindex survivors
    // densely preserving id order.
    if plants.len() > MAX_PLANTS {
        let mut order: Vec<u32> = (0..plants.len() as u32).collect();
        order.sort_by_key(|&p| (plants[p as usize].tris, p));
        let n_demote = plants.len() - MAX_PLANTS;
        let mut demoted = vec![false; plants.len()];
        let mut demoted_tris = 0u64;
        for &p in order.iter().take(n_demote) {
            demoted[p as usize] = true;
            demoted_tris += plants[p as usize].tris as u64;
        }
        let mut remap = vec![FIELD_PLANT; plants.len()];
        let mut kept: Vec<Plant> = Vec::with_capacity(MAX_PLANTS);
        for (p, plant) in plants.iter().enumerate() {
            if !demoted[p] {
                remap[p] = kept.len() as u32;
                kept.push(*plant);
            }
        }
        eprintln!(
            "foliage: plant cap — demoted {n_demote} of {} plants ({demoted_tris} tris) to \
             field membership",
            plants.len()
        );
        for tp in tri_plant.iter_mut() {
            if *tp != FIELD_PLANT {
                *tp = remap[*tp as usize];
            }
        }
        plants = kept;
    }

    PlantSet { plants, tri_plant }
}

/// The rooting ramp: exactly 0 at the content floor, rising as
/// `((y − floor)/band)^γ` to 1 above `SWAY_HEIGHT_BAND` of the content
/// height — concave (γ = 0.5, see `SWAY_RAMP_GAMMA`), which the per-cell
/// chord machinery depends on. Replaces v0.3's `height_factor` and its
/// `SWAY_GROUND_K` floor: the factor is per-VERTEX now (through the cell
/// chord), so the floor's job — keeping rigid ground billboards visibly
/// alive — is done by the exponent instead, with the base truly pinned.
/// Concave on [cmin_y, ∞) because content geometry never sits below the
/// content floor (the clamp's lower knee at y = cmin_y is outside the
/// domain any chord spans).
#[inline]
fn ramp(y: f32, cmin_y: f32, cmax_y: f32) -> f32 {
    let band = (SWAY_HEIGHT_BAND * (cmax_y - cmin_y)).max(1e-6);
    ((y - cmin_y) / band).clamp(0.0, 1.0).powf(SWAY_RAMP_GAMMA)
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
    Some(SwaySplit { first_chunk, cells, cell_of, cell: sway.cell })
}

/// The curl field as a unit-bounded direction at a chosen wavelength — the
/// fireflies `curl_dir` shape (a synthetic time-0 `Clouds`; the field is
/// time-independent and reads only `diag`), with `field_k` setting the
/// wavelength instead of the whole content diagonal (v0.4 samples it at TWO
/// wavelengths — the gust and fine octaves).
#[inline]
fn curl_dir(p: Vec3A, field_k: f32, scale: f32) -> Vec3A {
    let field = (field_k * scale).max(1e-6);
    crate::clouds::curl_offset(p, &crate::clouds::Clouds::new(true, field, 0.0))
        * (1.0 / (crate::clouds::CLOUD_CURL_AMP_K * field))
}

/// Upper bound on ANY displacement a cell with bound multiplier `w_max` can
/// produce at any time under `--foliage-amp` multiplier `mult`. Per axis
/// the curl half of the WIND vector is exactly `mult·SWAY_AMP_K·scale`
/// (soft normalization + the CONVEX octave blend) and the flutter half
/// `mult·SWAY_BOB_K·scale`; only x/z are live (`u.y ≡ 0`), so `√2` folds
/// per-axis to vector length; a vertex's displacement is `u·w_lin(y)` with
/// `0 ≤ w_lin ≤ w_max` on the cell (the concave-chord argument on
/// `SwayCell::w_max`). `SWAY_PAD_EPS_K` absorbs the affine evaluation's
/// absolute-y fp slop (GPU instance matrix + CPU `o + t·d` reconstruction)
/// — zero for a dead cell so the w_max = 0 arm stays exact. `self_test`
/// sweeps it at several mults.
pub fn displacement_bound_with(w_max: f32, scale: f32, mult: f32) -> f32 {
    let eps = if w_max > 0.0 { SWAY_PAD_EPS_K * scale } else { 0.0 };
    2f32.sqrt() * mult * w_max * (SWAY_AMP_K + SWAY_BOB_K) * scale + eps
}

/// `displacement_bound_with` at the session's `--foliage-amp`. Currently
/// unwired — the build sweep and the self-tests pass their own mult
/// (`sway_pad`/`sweep_mult`) — but this is the session-amp shape the module
/// docs reason in, so it stays as the named form of that bound.
#[allow(dead_code)]
pub fn displacement_bound(w_max: f32, scale: f32) -> f32 {
    displacement_bound_with(w_max, scale, amp_mult())
}

/// The BVH build-sweep pad for one triangle: its cell's displacement bound
/// at `sweep_mult()` (0 for static tris). ONE function serves the build
/// (`bvh::grow_sway_sweep` pads min AND max by it, x/z only — the shear is
/// signed and has NO y component) and the self-test's swept-containment pin
/// — the `tri_height_depth` build-vs-runtime discipline, which is the
/// containment proof.
pub fn sway_pad(sway: &SceneSway, tri: u32) -> f32 {
    let c = sway.tri_cell[tri as usize];
    if c == STATIC_CELL {
        return 0.0;
    }
    let cl = &sway.cells[c as usize];
    displacement_bound_with(cl.w_max, cl.scale, sweep_mult())
}

/// Closed-form WIND vector for cell `i` at clock `time` — the whole motion
/// model, the fireflies `pose` shape; a vertex's displacement is
/// `u · (a + b·y)`, never `u` alone. Pure function of (cell, i, time,
/// mult); hashes are `sky::pcg_mix` chains. Zero rng draws. `u.y ≡ 0.0` BY
/// CONSTRUCTION — the det-1 / trivial-inverse contract `bvh::shear_ray`
/// and the GPU rows both lean on (pinned bitwise by `self_test`), and what
/// retires the v0.3.1 "curl sinks/lifts the billboard" accept. The public
/// `wind` reads the session `--foliage-amp`; this form takes it explicitly
/// so the self-test can sweep mults without touching the global inside
/// `--check`.
fn wind_with(c: &SwayCell, time: f32, mult: f32) -> Vec3A {
    let scale = c.scale;
    use crate::sky::{hash01, pcg_mix};
    // A cell whose whole y-extent sits at its root has w_lin ≡ 0: bake the
    // structural EXACT zero so `u != 0` fast paths (the gateway skip, the
    // GPU identity rows) hold, and the zero-cell arm needs no pad.
    if c.w_max <= 0.0 {
        return Vec3A::ZERO;
    }
    // Gusts: the static field sampled at lookup points moving along the one
    // wind line — spatially continuous across anchors (no per-cell offset).
    // Octave 2 is v0.4's decorrelator: ¼ the wavelength at 2.5× the scroll
    // speed, blended CONVEXLY so per-axis |v| ≤ 1 survives (the pad
    // algebra's premise). v0.5: the anchor is the PLANT's for plant cells —
    // every cell of one plant samples the identical points, which with the
    // shared `key` below makes the whole plant's pose bit-equal BY
    // CONSTRUCTION (the no-tearing spine).
    let wind_line = Vec3A::new(0.37, 0.0, 0.61);
    let v1 = curl_dir(c.anchor + wind_line * (SWAY_WIND_K * scale * time), SWAY_FIELD_K, scale);
    let v2 =
        curl_dir(c.anchor + wind_line * (SWAY_WIND2_K * scale * time), SWAY_FIELD2_K, scale);
    let v = v1 * (1.0 - SWAY_OCT2_K) + v2 * SWAY_OCT2_K;
    // Flutter: hashed x/z sines per KEY (plant for plant cells, cell ordinal
    // for field cells; ω ∈ [0.8, 2.4] rad/s — leaves are quicker than
    // fireflies). The SIX-hash chain is v0.2's verbatim: h1/h4 fed the
    // retired y sine and stay in the chain so the x/z phases (and with them
    // every recorded look) survive the y deletion; zero-plant scenes seed
    // from the cell index exactly as v0.4 did.
    let h0 = pcg_mix(c.key.wrapping_mul(0x9E37_79B9) ^ 0x5EA5_1EAF);
    let h1 = pcg_mix(h0);
    let h2 = pcg_mix(h1);
    let h3 = pcg_mix(h2);
    let h4 = pcg_mix(h3);
    let h5 = pcg_mix(h4);
    let _ = (h1, h4); // retired y-sine draws — chain preserved, see above
    let tau = std::f32::consts::TAU;
    let bx = ((0.8 + 1.6 * hash01(h0)) * time + tau * hash01(h3)).sin();
    let bz = ((0.8 + 1.6 * hash01(h2)) * time + tau * hash01(h5)).sin();
    (Vec3A::new(v.x, 0.0, v.z) * SWAY_AMP_K + Vec3A::new(bx, 0.0, bz) * SWAY_BOB_K)
        * (mult * scale)
}

/// `wind_with` at the session's `--foliage-amp`.
pub fn wind(c: &SwayCell, time: f32) -> Vec3A {
    wind_with(c, time, amp_mult())
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
            normal_role: false,
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
        let m_trunk = b.material_kind(white, 0.5, 0.0, 0.0, MatKind::Textured { tex: t_opaque });
        let m_trunk_flat = b.material_kind(white, 0.5, 0.0, 0.0, MatKind::Diffuse);
        b.tri([Vec3A::ZERO, Vec3A::X, Vec3A::Y], [Vec3A::Z; 3], m_leaf);
        let mut synth = b.finish(crate::sky::Sun::new(Vec3A::Y));
        let fol = crate::matclass::IDX_FOLIAGE as u8;
        let brk = crate::matclass::IDX_BARK as u8;
        synth.materials[m_leaf as usize].class = fol;
        synth.materials[m_bark as usize].class = fol; // foliage-classed opaque = static straggler
        synth.materials[m_flat as usize].class = fol; // untextured, static
        synth.materials[m_trunk as usize].class = brk;
        synth.materials[m_trunk_flat as usize].class = brk;
        let m = leaf_materials(&synth);
        let w = woody_materials(&synth);
        let pl = plant_materials(&synth);
        if m.len() != synth.materials.len() || w.len() != m.len() || pl.len() != m.len() {
            return Err("mask lengths disagree".into());
        }
        let want = |i: u32, wl: bool, ww: bool, what: &str| -> Result<(), String> {
            if m[i as usize] != wl {
                return Err(format!("leaf_materials: {what} should be {wl}"));
            }
            if w[i as usize] != ww {
                return Err(format!("woody_materials: {what} should be {ww}"));
            }
            if pl[i as usize] != (wl || ww) {
                return Err(format!("plant_materials: {what} != leaf|woody"));
            }
            Ok(())
        };
        want(m_leaf, true, false, "foliage + textured + alpha")?;
        want(m_bark, false, false, "foliage + opaque texture (straggler, static)")?;
        want(m_cutout, false, false, "non-foliage cutout")?;
        want(m_flat, false, false, "foliage + untextured")?;
        // The woody mask has NO alpha/texture leg — the bark class byte alone
        // (vokselia has no alpha signal at all; the byte is the vocabulary).
        want(m_trunk, false, true, "bark + opaque texture (trunk)")?;
        want(m_trunk_flat, false, true, "bark + untextured")?;
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
    // The synthetic arms run the FIELD path (zero plants) — bit-identical to
    // the v0.4 partition by construction, which is what keeps every
    // structural must-fire below (E filler, root gateway, displaced-hit)
    // firing unchanged. The plant machinery gets its own block further down.
    let field = PlantSet::field_only(scene.indices.len());
    let regs = regions_of(scene);

    // Off arm: an all-false mask has no partition — the structural off-state
    // (the caller never reaches split_plan without a partition).
    if cell_partition(scene, &none, &field, &regs).is_some() {
        return Err("cell_partition: empty mask must return None".into());
    }
    let Some(part) = cell_partition(scene, &all, &field, &regs) else {
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
        let part2 =
            cell_partition(scene, &all, &field, &regs).ok_or("determinism re-partition vanished")?;
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
            real_part = cell_partition(scene, &mask, &field, &regs);
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
            let tk = winds(&sq.cells, 7.3);
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
                if tk[j] != wind(&part.cells[c as usize], 7.3) {
                    return Err(format!("run {j} disagrees with the CPU bake of cell {c}"));
                }
            }
            if !dup_seen {
                return Err("re-key pin was vacuous — no cell produced two runs".into());
            }
        }
    }

    // CPU bake == GPU keyed bake, bit-for-bit (the cross-arm pose contract) —
    // wind vectors AND the derived instance-matrix rows (`shear_rows` is the
    // ONE derivation both arms call, so this pins that it cannot fork) —
    // plus the u.y ≡ 0 lane pin and the bit-equal-clock fast path.
    {
        let pm = cell_partition(scene, &all, &field, &regs).ok_or("bake partition vanished")?;
        bake(&pm, 7.3);
        let tk = winds(&sp.cells, 7.3);
        for (j, &c) in sp.cell_of.iter().enumerate() {
            if tk[j] != pm.wind(c as u16) {
                return Err(format!("GPU run {j} != CPU offsets[{c}] after bake"));
            }
            let rr = shear_rows(tk[j], sp.cells[j].a, sp.cells[j].b);
            let pc = &pm.cells[c as usize];
            let cr = shear_rows(pm.wind(c as u16), pc.a, pc.b);
            if rr.map(f32::to_bits) != cr.map(f32::to_bits) {
                return Err(format!("GPU run {j} shear rows != CPU cell {c} rows"));
            }
        }
        // The det-1 contract lands in the baked atomics too: lane 1 must be
        // exactly 0.0 bits for every cell.
        if pm.offsets_snapshot().iter().any(|o| o.y.to_bits() != 0) {
            return Err("baked wind carries a y lane — u.y ≡ 0 broken".into());
        }
        let snap = pm.offsets_snapshot();
        bake(&pm, 7.3); // bit-equal clock — must be a no-op
        if pm.offsets_snapshot() != snap {
            return Err("bake fast path moved offsets on a bit-equal clock".into());
        }
        bake(&pm, 8.1);
        if pm.cells.iter().any(|c| c.w_max > 0.0) && pm.offsets_snapshot() == snap {
            return Err("bake did not move offsets on a new clock".into());
        }
    }

    // -- motion + chords: the wind bound, u.y ≡ 0, the chord's rooting /
    // endpoint / concavity pins, displacement bounds at the cell's y
    // endpoints, determinism, and time-variation — swept across
    // `--foliage-amp` multipliers through `wind_with` (never the global:
    // the session's own setting must not move under a gate run). Indexed by
    // PARTITION cell — the flutter hash key after the v0.2 re-key.
    // (v0.6: the scale rides IN each cell — `c.scale` below.)
    let (cy0, cy1) = (scene.content_min.y, scene.content_max.y);
    let span_floor = SWAY_CHORD_SPAN_K * (SWAY_HEIGHT_BAND * (cy1 - cy0)).max(1e-6);
    let mut moved = false;
    let mut rooted_seen = false;
    for (i, c) in part.cells.iter().enumerate().take(64) {
        let scale = c.scale;
        // Chord pins (pure build-time math, mult-independent). The endpoint
        // tolerances scale with |b·y| — the chord is evaluated at ABSOLUTE
        // y, so its fp error budget is cancellation-sized, not w-sized.
        {
            let w0 = ramp(c.y0, cy0, cy1);
            let w1 = ramp(c.y1, cy0, cy1);
            if c.w_max.to_bits() != w1.to_bits() {
                return Err(format!("cell {i}: w_max != ramp(y1)"));
            }
            // The rooted-base pin, BITWISE: a floor-touching chord (w0 = 0)
            // must evaluate to exactly 0.0 at its own y0 — the fp argument
            // on SwayCell::a (a = −fl(b·y0), and −x + x is exact).
            if w0 == 0.0 {
                rooted_seen = true;
                if c.a + c.b * c.y0 != 0.0 {
                    return Err(format!(
                        "cell {i}: floor-touching chord does not root bitwise"
                    ));
                }
            }
            // Top endpoint rejoins the ramp (only when the span floor did
            // not engage — a floored chord deliberately under-sways the
            // top; see SWAY_CHORD_SPAN_K).
            let tol = |x: f32| 1e-4 * (1.0 + x.abs());
            if c.y1 - c.y0 >= span_floor && (c.a + c.b * c.y1 - w1).abs() > tol(c.b * c.y1) {
                return Err(format!("cell {i}: chord top endpoint drifted from the ramp"));
            }
            // Concavity: the chord must never exceed the ramp — the premise
            // that makes w_max a bound. Fails loudly if a future retune
            // makes the ramp convex (γ > 1) without reworking the pads.
            let ym = 0.5 * (c.y0 + c.y1);
            if c.a + c.b * ym > ramp(ym, cy0, cy1) + tol(c.b * ym) {
                return Err(format!(
                    "cell {i}: chord exceeds the ramp at the midpoint — concavity broken"
                ));
            }
        }
        for &mult in &[0.25f32, 1.0, 4.0] {
            // Per-axis soft normalization + the CONVEX octave blend make the
            // wind bound exact; displacement is u·w_lin(y), linear in y, so
            // its max over the cell is at an endpoint — checking y0 and y1
            // is a proof, not sampling.
            let u_bound = 2f32.sqrt() * mult * (SWAY_AMP_K + SWAY_BOB_K) * scale + 1e-5 * scale;
            let d_bound = displacement_bound_with(c.w_max, scale, mult) + 1e-5 * scale;
            for &t in &[0.0f32, 0.37, 7.3, 123.4, 4096.0] {
                let u = wind_with(c, t, mult);
                if !u.is_finite() {
                    return Err(format!("cell {i}: non-finite wind at t={t}"));
                }
                if u.y.to_bits() != 0 {
                    return Err(format!("cell {i}: wind carries a y component at t={t}"));
                }
                if u.length() > u_bound {
                    return Err(format!(
                        "cell {i}: |u| {} exceeds the wind bound {} at t={t} mult={mult}",
                        u.length(),
                        u_bound
                    ));
                }
                for &y in &[c.y0, c.y1] {
                    let d = u * (c.a + c.b * y);
                    if d.length() > d_bound {
                        return Err(format!(
                            "cell {i}: |d| {} exceeds the bound {} at y={y} t={t} mult={mult}",
                            d.length(),
                            d_bound
                        ));
                    }
                }
                if u != wind_with(c, t, mult) {
                    return Err("wind is not deterministic".into());
                }
                if u.length() > 1e-9 * scale {
                    moved = true;
                }
            }
        }
        // The w_max-0 identity must be EXACT — flutter included. Synthetic:
        // a real all-floor cell is rare, but the arm guards `wind_with`'s
        // structural zero (the gateway-skip / GPU-identity-row fast paths).
        let pinned = SwayCell { w_max: 0.0, ..*c };
        if wind_with(&pinned, 7.3, 4.0) != Vec3A::ZERO {
            return Err("w_max-0 cell must not move at all".into());
        }
    }
    if !moved {
        return Err("no cell moved anywhere in the sweep — the field is dead".into());
    }
    // Anti-vacuity for the rooted-base pin on the session scene is NOT
    // required (a scene whose foliage never touches the content floor is
    // legitimate — canopy-only content); the synthetic gateway scene below
    // must-fires it instead.
    let _ = rooted_seen;

    // -- MV deltas (`mv_rows`/`prev_point`): the closed-form prev-pose map
    // the sway MVs ride on. Three pins:
    //   (1) round trip — displacing a rest point by pose(t1) and applying
    //       the (t0 − t1) delta must land on the pose(t0) displacement of
    //       the same rest point (y bitwise-invariant; x/z to fp association);
    //   (2) bit-equal clocks ⇒ None (the frozen-still / pinned-gate off arm);
    //   (3) shear_rows linearity — rows(u0 − u1) == rows(u0) − rows(u1)
    //       elementwise, documenting why building from the difference is the
    //       same map (and exact at du = 0 where the subtraction form isn't
    //       guaranteed to cancel bitwise).
    {
        let (t0, t1) = (7.5f32, 9.5f32);
        if mv_rows(&part, t0, t0).is_some() {
            return Err("mv_rows: bit-equal clocks must return None".into());
        }
        let Some(mv) = mv_rows(&part, t1, t0) else {
            return Err("mv_rows: distinct clocks returned None".into());
        };
        if mv.rows.len() != part.cells.len() {
            return Err("mv_rows: row count != cell count".into());
        }
        let mut mv_moved = false;
        for (i, c) in part.cells.iter().enumerate().take(64) {
            let u0 = wind(c, t0);
            let u1 = wind(c, t1);
            // Linearity pin (elementwise, ulp-scale slack on the products).
            let direct = shear_rows(u0 - u1, c.a, c.b);
            let r0 = shear_rows(u0, c.a, c.b);
            let r1 = shear_rows(u1, c.a, c.b);
            for k in 0..4 {
                let diff = r0[k] - r1[k];
                if (direct[k] - diff).abs() > 1e-6 * (1.0 + direct[k].abs().max(diff.abs())) {
                    return Err(format!("cell {i}: shear_rows linearity broken at lane {k}"));
                }
            }
            // Round trip at both y endpoints (displacement linear in y ⇒
            // endpoints are a proof): rest r → pose(t1) point q1, then the
            // delta must carry q1 to pose(t0)'s q0.
            for &y in &[c.y0, c.y1] {
                let r = Vec3A::new(c.anchor.x + 0.25 * c.scale, y, c.anchor.z - 0.125 * c.scale);
                let w = c.a + c.b * y;
                let q1 = r + u1 * w;
                let q0 = r + u0 * w;
                let got = prev_point(&mv.rows[i], q1);
                if got.y.to_bits() != q1.y.to_bits() {
                    return Err(format!("cell {i}: prev_point moved y"));
                }
                if (got - q0).length() > 1e-5 * (1.0 + c.scale) {
                    return Err(format!(
                        "cell {i}: prev-pose round trip off by {} at y={y}",
                        (got - q0).length()
                    ));
                }
                if (q0 - q1).length() > 1e-9 * c.scale {
                    mv_moved = true;
                }
            }
        }
        if !mv_moved {
            return Err("mv_rows: no cell moved between the two clocks — dead delta".into());
        }
    }

    // The swept-containment pin (build-vs-motion, the height_self_test
    // shape): every pose reachable at mult <= sweep_mult() lies inside the
    // box the BVH build pads by `sway_pad` — |u·w_lin(y)| <= pad at BOTH y
    // endpoints (linear in y ⇒ endpoint max is a proof), with both signs
    // covered because `grow_sway_sweep` pads min AND max, and the y
    // displacement exactly 0 against the pad's zero y axis. One function
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
                    let u = wind_with(cell0, tt, mult);
                    for &y in &[cell0.y0, cell0.y1] {
                        let d = u * (cell0.a + cell0.b * y);
                        if d.y != 0.0 {
                            return Err(format!(
                                "tri {t}: y displacement {} against a zero-y pad",
                                d.y
                            ));
                        }
                        if d.length() > pad + 1e-5 * cell0.scale {
                            return Err(format!(
                                "tri {t}: |d| {} escapes the swept pad {pad} at y={y} \
                                 t={tt} mult={mult}",
                                d.length()
                            ));
                        }
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
        let synth = synth_sway_scene;
        // Two clusters far apart and vertically spread: the FLOOR cluster
        // (y ∈ [0, 0.05]) sits in the ramp's steep base — its cell roots at
        // the content floor (w0 = 0, b ≠ 0), which is what the rooted-base
        // and d-shear pins need — while the ELEVATED cluster sits ABOVE the
        // band (w0 = w1 = 1 ⇒ b = 0), the pure-translation arm.
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
        // Displaced-hit pins: bake a pose and aim rays at M(rest point) —
        // the affine pose preserves barycentric combinations, so M of any
        // point ON the rest triangle lies ON the displaced triangle — and
        // the gateway shear must land each hit there (t preserved, o + t·d
        // on the DISPLACED surface): the end-to-end proof of the traversal
        // arm. Three rays: (1) the v0.3 −Z shape at the ELEVATED centroid
        // (b = 0 — the translation fast path); (2) a SLANTED ray (d.y ≠ 0)
        // at the FLOOR cluster's tip (b ≠ 0 — the d-shear + inv_d recompute
        // actually fire; a horizontal ray leaves d_r == d and would gate
        // the new arm vacuously); (3) the same slanted shape near the FLOOR
        // cluster's BASE, where the rooted ramp must leave the surface
        // (almost) at rest — plus the base VERTEX pinned bitwise.
        bake(sw2, 5.0);
        let m_at = |tri: usize, p: Vec3A| -> Vec3A {
            let s = sw2.shear(sw2.tri_cell[tri]);
            p + s.u * (s.a + s.b * p.y)
        };
        let bary = |tri: usize, w: [f32; 3]| -> Vec3A {
            let tv = s2.indices[tri];
            s2.positions[tv[0] as usize] * w[0]
                + s2.positions[tv[1] as usize] * w[1]
                + s2.positions[tv[2] as usize] * w[2]
        };
        let shoot = |o: Vec3A, tri: usize, want: Vec3A, what: &str| -> Result<(), String> {
            let ray = crate::bvh::Ray::new(o, (want - o).normalize());
            let mut vis = 0u64;
            match bvh2.intersect(&s2, &ray, 0.0, 100.0, &mut vis) {
                Some(h) if h.tri == tri as u32 => {
                    let p = ray.o + h.t * ray.d;
                    if (p - want).length() > 1e-4 {
                        return Err(format!(
                            "gateway synth ({what}): hit off target by {}",
                            (p - want).length()
                        ));
                    }
                    Ok(())
                }
                Some(h) => Err(format!("gateway synth ({what}): hit wrong tri {}", h.tri)),
                None => Err(format!("gateway synth ({what}): ray missed the displaced pose")),
            }
        };
        // (1) elevated centroid, straight −Z (the historical shape).
        let hi = 3usize; // first tri of the second (elevated) cluster
        let want_hi = m_at(hi, bary(hi, [1.0 / 3.0; 3]));
        if (want_hi - bary(hi, [1.0 / 3.0; 3])) == Vec3A::ZERO {
            return Err("gateway synth: baked pose has zero displacement (vacuous pin)".into());
        }
        shoot(want_hi + 5.0 * Vec3A::Z, hi, want_hi, "elevated -Z")?;
        // (2) floor-cluster tip, slanted. Anti-vacuity: the cell must carry
        // a live chord slope and a live wind, or the d-shear never fires.
        let lo = 0usize;
        let s_lo = sw2.shear(sw2.tri_cell[lo]);
        if s_lo.b == 0.0 || s_lo.u == Vec3A::ZERO {
            return Err("gateway synth: floor cell has no live shear (vacuous d-shear pin)".into());
        }
        let want_tip = m_at(lo, bary(lo, [0.1, 0.1, 0.8]));
        shoot(want_tip + Vec3A::new(1.0, 2.0, 5.0), lo, want_tip, "floor tip slanted")?;
        // (3) near the floor cluster's base the surface barely moves...
        let p_base = bary(lo, [0.8, 0.1, 0.1]);
        let want_base = m_at(lo, p_base);
        if (want_base - p_base).length() > 1e-3 * sw2.cells[sw2.tri_cell[lo] as usize].scale {
            return Err("gateway synth: near-base point moved more than the rooted ramp allows".into());
        }
        shoot(want_base + Vec3A::new(1.0, 2.0, 5.0), lo, want_base, "floor base slanted")?;
        // ...and the base VERTEX (y == the content floor) is pinned BITWISE:
        // w_lin(y0) is exactly 0, so M is the identity there.
        let v0 = s2.positions[s2.indices[lo][0] as usize];
        if m_at(lo, v0) != v0 {
            return Err("gateway synth: base vertex not bitwise-rooted".into());
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

    // -- PLANT machinery (v0.5) on a synthetic whole-plant scene: a static
    // floor tri at y=0 (class DEFAULT — sets the content floor), a
    // vertex-CHAINED opaque BARK trunk strip spanning y ∈ [1, 9] over many
    // voxels (union-find must find one component), a DISCONNECTED leaf
    // cluster overlapping the trunk top (proximity attach), and one far
    // lone leaf billboard (the FIELD fallback). Gated on gateway_mode like
    // the block above (attach declines otherwise).
    if gateway_mode() {
        let mk_tex = |alpha: bool| crate::texture::Texture {
            w: 1,
            h: 1,
            texels: vec![[255, 255, 255, 255]],
            alpha_masked: alpha,
            srgb: true,
            source: String::new(),
            h2n: false,
            n2h: false,
            normal_role: false,
            mips: Vec::new(),
        };
        let mut b = crate::scene::SceneBuilder::new();
        let t_leaf = b.add_texture(mk_tex(true));
        let t_bark = b.add_texture(mk_tex(false));
        let m_static = b.material_kind(Vec3A::ONE, 0.8, 0.0, 0.0, MatKind::Diffuse);
        let m_leaf =
            b.material_kind(Vec3A::ONE, 0.5, 0.0, 0.0, MatKind::Textured { tex: t_leaf });
        let m_trunk =
            b.material_kind(Vec3A::ONE, 0.7, 0.0, 0.0, MatKind::Textured { tex: t_bark });
        // Floor (static, content min y = 0).
        b.tri([Vec3A::ZERO, Vec3A::X, Vec3A::Z], [Vec3A::Y; 3], m_static);
        // Trunk: an indexed strip — rungs at y = 1 + k, two shared vertices
        // per rung, 2 tris per segment (8 segments, y ∈ [1, 9]).
        {
            let mut pos = Vec::new();
            let mut idx = Vec::new();
            for k in 0..=8u32 {
                let y = 1.0 + k as f32;
                pos.push(Vec3A::new(0.0, y, 0.0));
                pos.push(Vec3A::new(0.3, y, 0.0));
            }
            for k in 0..8u32 {
                let b0 = 2 * k;
                idx.push([b0, b0 + 1, b0 + 2]);
                idx.push([b0 + 1, b0 + 3, b0 + 2]);
            }
            let n = vec![Vec3A::Z; pos.len()];
            let tc = vec![glam::Vec2::ZERO; pos.len()];
            b.add_mesh(pos, n, tc, &idx, m_trunk);
        }
        // Leaf cluster: disconnected cards overlapping the trunk's top —
        // coplanar with the trunk strip (z = 0), inside its x range, so the
        // inflated AABBs genuinely overlap at the CONTACT tolerance
        // (PLANT_MERGE_K is ~0.018 here — a 0.05 z offset would be a miss).
        for j in 0..4 {
            let o = Vec3A::new(0.05 * j as f32, 8.5 + 0.15 * j as f32, 0.0);
            b.tri([o, o + 0.1 * Vec3A::X, o + 0.1 * Vec3A::Y], [Vec3A::Z; 3], m_leaf);
        }
        // Lone billboard far away in x/z — must stay FIELD.
        let far = Vec3A::new(5.0, 0.5, 5.0);
        b.tri([far, far + 0.1 * Vec3A::X, far + 0.1 * Vec3A::Y], [Vec3A::Z; 3], m_leaf);
        let mut s = b.finish(crate::sky::Sun::new(Vec3A::Y));
        s.materials[m_leaf as usize].class = crate::matclass::IDX_FOLIAGE as u8;
        s.materials[m_trunk as usize].class = crate::matclass::IDX_BARK as u8;
        attach(&mut s);
        let Some(sw) = s.sway.as_deref() else {
            return Err("plant synth: attach declined the trunk scene".into());
        };
        if sw.n_plants != 1 {
            return Err(format!("plant synth: wanted 1 plant, got {}", sw.n_plants));
        }
        // tri_plant routing (re-derived — derive_plants is deterministic, so
        // this is the attach-time set): trunk + cluster in plant 0, the
        // billboard FIELD.
        let mask = plant_materials(&s);
        let woody = woody_materials(&s);
        let ps = derive_plants(&s, &mask, &woody, &regions_of(&s));
        let n_tris = s.indices.len();
        let bill = n_tris - 1; // the lone billboard is the last tri added
        for t in 1..n_tris - 1 {
            if ps.tri_plant[t] != 0 {
                return Err(format!("plant synth: tri {t} not in the plant"));
            }
        }
        if ps.tri_plant[0] != FIELD_PLANT || ps.tri_plant[bill] != FIELD_PLANT {
            return Err("plant synth: floor/billboard routing wrong".into());
        }
        let ds2 = derive_plants(&s, &mask, &woody, &regions_of(&s));
        if ds2.plants != ps.plants || ds2.tri_plant != ps.tri_plant {
            return Err("plant synth: derive_plants is not deterministic".into());
        }
        // Plant extents cover the CANOPY (finalized after leaf attach) and
        // root at the plant's OWN base — above the content floor, which is
        // the v0.4 potted-plant known-accept retired.
        let p = &ps.plants[0];
        if p.y0 != 1.0 || p.y1 < 8.9 {
            return Err(format!("plant synth: extent [{}, {}] wrong", p.y0, p.y1));
        }
        if p.b <= 0.0 || (p.a + p.b * p.y0) != 0.0 {
            return Err("plant synth: chord not bitwise-rooted at the plant base".into());
        }
        // Coherence: >= 2 cells share the plant key; after a bake they carry
        // bitwise-equal wind/(a, b)/shear_rows — the no-tearing proof — with
        // differing per-cell w_max (proves coherence isn't cell cloning).
        bake(sw, 5.0);
        let pcells: Vec<usize> = (0..sw.cells.len())
            .filter(|&c| sw.cells[c].key == (PLANT_KEY_BIT | 0))
            .collect();
        if pcells.len() < 2 {
            return Err(format!("plant synth: wanted >= 2 plant cells, got {}", pcells.len()));
        }
        let c0 = &sw.cells[pcells[0]];
        let u0 = sw.wind(pcells[0] as u16);
        if u0 == Vec3A::ZERO {
            return Err("plant synth: baked plant wind is zero (vacuous coherence pin)".into());
        }
        let rows0 = shear_rows(u0, c0.a, c0.b);
        let mut w_max_differs = false;
        for &c in &pcells[1..] {
            let cl = &sw.cells[c];
            let u = sw.wind(c as u16);
            if u != u0
                || cl.anchor != c0.anchor
                || cl.a.to_bits() != c0.a.to_bits()
                || cl.b.to_bits() != c0.b.to_bits()
                || shear_rows(u, cl.a, cl.b).map(f32::to_bits) != rows0.map(f32::to_bits)
            {
                return Err(format!("plant synth: cell {c} pose differs — the plant tears"));
            }
            w_max_differs |= cl.w_max.to_bits() != c0.w_max.to_bits();
        }
        if !w_max_differs {
            return Err("plant synth: every plant cell has one w_max — coherence pin vacuous".into());
        }
        // Per-cell bound pin: w_max is the chord's own value at the cell top,
        // VERBATIM (bitwise — never clamped).
        for &c in &pcells {
            let cl = &sw.cells[c];
            if cl.w_max.to_bits() != (cl.a + cl.b * cl.y1).to_bits() {
                return Err(format!("plant synth: cell {c} w_max != fl(a + b*y1)"));
            }
        }
        // The trunk base vertex (y == the PLANT's y0) maps to itself
        // bitwise, and a displaced MID-TRUNK point is hit by a slanted ray
        // (the first OPAQUE moving geometry; d.y != 0 so the d-shear +
        // inv_d recompute fire — b > 0 was pinned above).
        let bvh_s = crate::bvh::Bvh::build(&s);
        gateway_audit(&s, &bvh_s)?;
        let m_at = |tri: usize, pt: Vec3A| -> Vec3A {
            let sh = sw.shear(sw.tri_cell[tri]);
            pt + sh.u * (sh.a + sh.b * pt.y)
        };
        let base_v = s.positions[s.indices[1][0] as usize]; // trunk rung 0
        if base_v.y != 1.0 || m_at(1, base_v) != base_v {
            return Err("plant synth: trunk base vertex not bitwise-rooted".into());
        }
        let mid_tri = 9usize; // a mid-trunk segment (tris 1..=16 are trunk)
        let tv = s.indices[mid_tri];
        let mid = (s.positions[tv[0] as usize]
            + s.positions[tv[1] as usize]
            + s.positions[tv[2] as usize])
            / 3.0;
        let want = m_at(mid_tri, mid);
        if want == mid {
            return Err("plant synth: mid-trunk displacement is zero (vacuous pin)".into());
        }
        let ray = crate::bvh::Ray::new(want + Vec3A::new(1.0, 2.0, 5.0), {
            let o = want + Vec3A::new(1.0, 2.0, 5.0);
            (want - o).normalize()
        });
        let mut vis = 0u64;
        match bvh_s.intersect(&s, &ray, 0.0, 100.0, &mut vis) {
            Some(h) if h.tri == mid_tri as u32 => {
                let hp = ray.o + h.t * ray.d;
                if (hp - want).length() > 1e-4 {
                    return Err(format!(
                        "plant synth: displaced trunk hit off by {}",
                        (hp - want).length()
                    ));
                }
            }
            Some(h) => return Err(format!("plant synth: trunk ray hit wrong tri {}", h.tri)),
            None => return Err("plant synth: displaced trunk ray missed".into()),
        }
        // FIELD fallback: the lone billboard still moves (its own cell,
        // global-ramp chord, nonzero wind).
        let bc = sw.tri_cell[bill];
        if bc == STATIC_CELL {
            return Err("plant synth: billboard fell out of the partition".into());
        }
        let bcell = &sw.cells[bc as usize];
        if bcell.key & PLANT_KEY_BIT != 0 {
            return Err("plant synth: billboard landed in the plant".into());
        }
        if bcell.w_max <= 0.0 || sw.wind(bc) == Vec3A::ZERO {
            return Err("plant synth: field billboard does not move".into());
        }
    }

    // -- REGION machinery (v0.6) on a synthetic two-island scene: two bark
    // trunks CONTACT-close (gap 0.01 < either region's merge tolerance h,
    // so a region-blind derivation provably FUSES them — the teeth) split
    // across two regions with different content boxes. Pins: 2 plants (no
    // cross-region merge, structural), per-region scale bitwise in the
    // cells, chords on the REGION ramp band, out-of-region masked tris
    // static, the empty-list fallback bit-equal to an explicit whole-scene
    // region, and determinism.
    if gateway_mode() {
        let mk_tex = |alpha: bool| crate::texture::Texture {
            w: 1,
            h: 1,
            texels: vec![[255, 255, 255, 255]],
            alpha_masked: alpha,
            srgb: true,
            source: String::new(),
            h2n: false,
            n2h: false,
            normal_role: false,
            mips: Vec::new(),
        };
        let build = || {
            let mut b = crate::scene::SceneBuilder::new();
            let t_leaf = b.add_texture(mk_tex(true));
            let t_bark = b.add_texture(mk_tex(false));
            let m_static = b.material_kind(Vec3A::ONE, 0.8, 0.0, 0.0, MatKind::Diffuse);
            let m_leaf =
                b.material_kind(Vec3A::ONE, 0.5, 0.0, 0.0, MatKind::Textured { tex: t_leaf });
            let m_trunk =
                b.material_kind(Vec3A::ONE, 0.7, 0.0, 0.0, MatKind::Textured { tex: t_bark });
            // tri 0: static floor.
            b.tri([Vec3A::ZERO, Vec3A::X, Vec3A::Z], [Vec3A::Y; 3], m_static);
            let strip = |b: &mut crate::scene::SceneBuilder, x0: f32, y1: f32| {
                let segs = (y1 - 1.0) as u32;
                let mut pos = Vec::new();
                let mut idx = Vec::new();
                for k in 0..=segs {
                    let y = 1.0 + k as f32;
                    pos.push(Vec3A::new(x0, y, 0.0));
                    pos.push(Vec3A::new(x0 + 0.3, y, 0.0));
                }
                for k in 0..segs {
                    let b0 = 2 * k;
                    idx.push([b0, b0 + 1, b0 + 2]);
                    idx.push([b0 + 1, b0 + 3, b0 + 2]);
                }
                let n = vec![Vec3A::Z; pos.len()];
                let tc = vec![glam::Vec2::ZERO; pos.len()];
                b.add_mesh(pos, n, tc, &idx, m_trunk);
            };
            // Trunk A: tris 1..9 (y ∈ [1, 5] at x ∈ [0, 0.3]); trunk B:
            // tris 9..25 (y ∈ [1, 9] at x ∈ [0.31, 0.61] — gap 0.01).
            strip(&mut b, 0.0, 5.0);
            strip(&mut b, 0.31, 9.0);
            // tri 25: a masked leaf card OUTSIDE both regions — must stay
            // static (the region_of(None) arm).
            let far = Vec3A::new(2.0, 0.5, 2.0);
            b.tri([far, far + 0.1 * Vec3A::X, far + 0.1 * Vec3A::Y], [Vec3A::Z; 3], m_leaf);
            let mut s = b.finish(crate::sky::Sun::new(Vec3A::Y));
            s.materials[m_leaf as usize].class = crate::matclass::IDX_FOLIAGE as u8;
            s.materials[m_trunk as usize].class = crate::matclass::IDX_BARK as u8;
            s
        };
        let box_a = (Vec3A::new(-1.0, 0.0, -1.0), Vec3A::new(1.0, 6.0, 1.0));
        let box_b = (Vec3A::new(-10.0, 0.0, -10.0), Vec3A::new(10.0, 40.0, 10.0));
        let regs2 = vec![
            SwayRegion { tri_start: 0, tri_end: 9, cmin: box_a.0, cmax: box_a.1 },
            SwayRegion { tri_start: 9, tri_end: 25, cmin: box_b.0, cmax: box_b.1 },
        ];
        let mut s = build();
        s.sway_regions = regs2.clone();
        attach(&mut s);
        let Some(sw) = s.sway.as_deref() else {
            return Err("region synth: attach declined the two-region scene".into());
        };
        if sw.n_plants != 2 {
            return Err(format!(
                "region synth: wanted 2 plants (no cross-region merge), got {}",
                sw.n_plants
            ));
        }
        // TEETH: region-blind (one whole-scene region), the same trunks FUSE
        // — proof the 2-plant pin above tests the region split and not mere
        // distance.
        {
            let rmask = plant_materials(&s);
            let rwoody = woody_materials(&s);
            let whole = vec![SwayRegion {
                tri_start: 0,
                tri_end: s.indices.len() as u32,
                cmin: s.content_min,
                cmax: s.content_max,
            }];
            let fused = derive_plants(&s, &rmask, &rwoody, &whole);
            if fused.plants.len() != 1 {
                return Err(format!(
                    "region synth teeth: region-blind derivation kept {} plants — the \
                     contact-gap premise broke and the 2-plant pin is vacuous",
                    fused.plants.len()
                ));
            }
        }
        // Out-of-region masked tri is static.
        if sw.tri_cell[25] != STATIC_CELL {
            return Err("region synth: out-of-region leaf card entered the partition".into());
        }
        // Per-region scale lands in the cells bitwise — and differs across
        // the regions (anti-vacuity).
        let scale_a = (box_a.1 - box_a.0).length().max(1e-3);
        let scale_b = (box_b.1 - box_b.0).length().max(1e-3);
        let (ca, cb) = (sw.tri_cell[1], sw.tri_cell[9]);
        if ca == STATIC_CELL || cb == STATIC_CELL {
            return Err("region synth: trunk tris fell out of the partition".into());
        }
        if sw.cells[ca as usize].scale.to_bits() != scale_a.to_bits()
            || sw.cells[cb as usize].scale.to_bits() != scale_b.to_bits()
        {
            return Err("region synth: cell scale != its region's content diagonal".into());
        }
        if scale_a.to_bits() == scale_b.to_bits() {
            return Err("region synth: region scales coincide — the scale pin is vacuous".into());
        }
        // Chords read the REGION ramp band: plant 0 (trunk A, y1 = 5) on
        // box A's band, plant 1 (trunk B, y1 = 9) on box B's — bitwise, and
        // differing (box B's taller band keeps B's w_max under 1).
        {
            let rmask = plant_materials(&s);
            let rwoody = woody_materials(&s);
            let ps = derive_plants(&s, &rmask, &rwoody, &regs2);
            if ps.plants.len() != 2 {
                return Err("region synth: re-derivation lost a plant".into());
            }
            let wa = ramp(ps.plants[0].y1, box_a.0.y, box_a.1.y);
            let wb = ramp(ps.plants[1].y1, box_b.0.y, box_b.1.y);
            if ps.plants[0].w_max.to_bits() != wa.to_bits()
                || ps.plants[1].w_max.to_bits() != wb.to_bits()
            {
                return Err("region synth: plant w_max != ramp on the REGION band".into());
            }
            if wa.to_bits() == wb.to_bits() {
                return Err("region synth: region ramps coincide — the band pin is vacuous".into());
            }
        }
        // Empty-list fallback == explicit whole-scene region, bit-equal
        // (the refactor's neutrality pin: every regionless scene reproduces
        // the pre-region partition).
        {
            let mut s_implicit = build();
            attach(&mut s_implicit);
            let mut s_explicit = build();
            s_explicit.sway_regions = vec![SwayRegion {
                tri_start: 0,
                tri_end: s_explicit.indices.len() as u32,
                cmin: s_explicit.content_min,
                cmax: s_explicit.content_max,
            }];
            attach(&mut s_explicit);
            match (s_implicit.sway.as_deref(), s_explicit.sway.as_deref()) {
                (Some(a), Some(b)) => {
                    if a.cells != b.cells || a.tri_cell != b.tri_cell {
                        return Err(
                            "region synth: implicit whole-scene region != explicit".into()
                        );
                    }
                }
                _ => return Err("region synth: fallback attach vanished".into()),
            }
        }
        // Determinism: the whole regioned pipeline, twice, bit-equal.
        let mut s_d = build();
        s_d.sway_regions = regs2;
        attach(&mut s_d);
        match (s.sway.as_deref(), s_d.sway.as_deref()) {
            (Some(a), Some(b)) => {
                if a.cells != b.cells || a.tri_cell != b.tri_cell {
                    return Err("region synth: regioned attach is not deterministic".into());
                }
            }
            _ => return Err("region synth: determinism re-attach vanished".into()),
        }
    }

    // -- Plant-cell chord pins on the SESSION's real partition, when it has
    // plants (San Miguel/bistro/rungholt --check): the coherence and
    // bound-verbatim contracts on real trees.
    if let Some(sw) = scene.sway.as_deref().filter(|sw| sw.n_plants > 0) {
        use std::collections::HashMap;
        let mut seen: HashMap<u32, usize> = HashMap::new();
        for (c, cl) in sw.cells.iter().enumerate() {
            if cl.key & PLANT_KEY_BIT == 0 {
                continue;
            }
            if !(cl.b >= 0.0) || !cl.b.is_finite() {
                return Err(format!("real plant cell {c}: bad chord slope {}", cl.b));
            }
            if cl.w_max.to_bits() != (cl.a + cl.b * cl.y1).to_bits() {
                return Err(format!("real plant cell {c}: w_max != fl(a + b*y1)"));
            }
            match seen.entry(cl.key) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(c);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    let f = &sw.cells[*e.get()];
                    if cl.anchor != f.anchor
                        || cl.a.to_bits() != f.a.to_bits()
                        || cl.b.to_bits() != f.b.to_bits()
                    {
                        return Err(format!(
                            "real plant cells {}/{c} share key {:#x} but not the pose",
                            e.get(),
                            cl.key
                        ));
                    }
                }
            }
        }
    }

    eprintln!(
        "foliage self-test: OK — synthetic split {} tris -> {} static + {} sway chunks \
         (cell {:.3}, scale {:.2}); scene leaf materials: {}",
        n_tris,
        sp.first_chunk,
        sp.cells.len(),
        sp.cell,
        part.cells.first().map_or(0.0, |c| c.scale),
        mask.iter().filter(|&&m| m).count()
    );
    Ok(())
}

/// The gateway tree's structural contracts (see `bvh::GATEWAY_BIT`): truthful
/// ranges == partition cells, the implicit +1 adjacency with no nesting, a
/// full reachability tiling (every node is reached from the root via child
/// links XOR lives inside exactly one gateway's subtree block — which is also
/// the proof the phase-2 stitch preserved adjacency), the swept-box identity
/// (gateway box == subtree root's rest box ± the cell's displacement bound
/// on x/z and untouched in y — u.y ≡ 0 — BITWISE; combined with the
/// swept-containment pin above, every displaced pose stays inside the one
/// swept box), E-filler shape, the exactly-one-gateway-per-cell must-fire,
/// and `tri_idx` remaining a true permutation.
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
        // The one swept box: subtree root's rest box ± the cell's bound,
        // x/z only (the v0.4 shear has no y component — u.y ≡ 0).
        let cl = &sw.cells[c as usize];
        let pad = displacement_bound_with(cl.w_max, cl.scale, sweep_mult());
        let padv = glam::Vec3A::new(pad, 0.0, pad);
        let sb = &bvh.nodes[sub as usize].aabb;
        let (emin, emax) = (sb.min - padv, sb.max + padv);
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
