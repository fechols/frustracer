//! Wind-swayed foliage — the v0 "leaf sway" prototype of the tetrahedral-cage
//! epic (docs/design/animated-foliage.md; the enabling prior work is Gruen,
//! Benthin, Kern & McAllister, *Ray Tracing Massive Amounts of Animated
//! Geometry*, https://doi.org/10.1145/3820014 — static per-chunk BLASes,
//! per-frame instance transforms, per-frame TLAS rebuild).
//!
//! v0 deliberately cuts the paper down to the smallest SOUND surface:
//!
//! - **Leaves only** (a material is a "leaf" iff its retained matclass
//!   verdict is foliage (`Material::class` — the byte carries the NAME half
//!   of the vocabulary, which is all the Minecraft atlas scenes have) AND its
//!   albedo texture is alpha-masked — `leaf_materials`).
//!   Trunks/bark stay static, so disconnected cutout leaves are the only
//!   moving geometry and there is nothing to tear: the paper's clipping +
//!   shared-cage-vertex machinery (its watertightness bill) is not needed at
//!   all in v0.
//! - **Translation-only, per spatial CELL** (no tets yet): leaf triangles
//!   bucket by centroid into a grid over the content box, each cell becomes
//!   its own BLAS chunk (`split_plan` — appended after the blas_split
//!   antichain chunks), and each cell's DXR instance gets a per-frame
//!   TRANSLATION. A translated instance leaves normals, tangents, UVs and
//!   hardware barycentrics untouched (all affine-invariant or
//!   translation-invariant), and both GPU shading paths reconstruct the hit
//!   point as `o + t·d` — so the shader surface is ZERO. The cost is
//!   per-cell-rigid motion (leaves in one cell sway together); the paper's
//!   per-tet affine is the follow-on.
//! - **DXR sessions only** consume the motion (`FrameParams::sway_time`): the
//!   animated TLAS lives in its own ring beside the pristine static TLAS,
//!   which the wavefront tracer / every gate keeps binding — the frustum
//!   quadtree, temporal claims and structure replay therefore never see a
//!   moving triangle and stay sound BY CONSTRUCTION (the design doc's
//!   swept-box phase is not needed until the wavefront arm consumes motion).
//!
//! Motion is the fireflies template verbatim: the static curl field
//! (`clouds::curl_offset` — soft-normalized |v| < 1, so the amplitude
//! constant is an EXACT displacement bound), sampled at a time-shifted
//! lookup point on the shared `cloud_time` clock, plus a small hashed-phase
//! flutter. ZERO rng draws anywhere — poses are pure functions of
//! (cell, time), so every same-seed / replay contract holds structurally.
//! Known-accepts (the design doc's list): no MVs on sway (bounded upscaler
//! ghosting), a converging still freezes mid-gust, the CPU renderer and the
//! wavefront tracer render the REST pose, incompatible with `--heightfield`
//! relief on leaf materials (the relief re-march reads rest-space geometry).

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
/// Height band over which sway fades in from the ground: 0 at the content
/// floor, full above `SWAY_HEIGHT_BAND` of the content height — grass barely
/// stirs, canopy sways (and nothing at ground level can be dragged through
/// the floor).
pub const SWAY_HEIGHT_BAND: f32 = 0.3;

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

/// `split_plan`'s product: which appended chunks sway, and the scale the
/// per-frame bake needs.
pub struct SwaySplit {
    /// Index of the first sway chunk in the rebuilt plan — chunks
    /// `first_chunk..first_chunk + cells.len()` are the animated instances,
    /// everything below is static.
    pub first_chunk: u32,
    /// One entry per sway chunk, in chunk order.
    pub cells: Vec<SwayCell>,
    /// The content diagonal every length constant above multiplies.
    pub scale: f32,
    /// The grid pitch the split settled on (after any coarsening) — the
    /// startup line's number.
    pub cell: f32,
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

/// Height fade: 0 at the content floor, 1 above `SWAY_HEIGHT_BAND` of the
/// content height.
#[inline]
fn height_factor(y: f32, cmin_y: f32, cmax_y: f32) -> f32 {
    let band = (SWAY_HEIGHT_BAND * (cmax_y - cmin_y)).max(1e-6);
    ((y - cmin_y) / band).clamp(0.0, 1.0)
}

/// Re-partition a `BlasPlan`: pull every leaf triangle out of the antichain
/// chunks and append one chunk per non-empty grid cell (split into
/// `max_prims` runs if a cell overflows the BLAS cap — the oversized-leaf
/// idiom). Static chunks keep their `chunk_node`; sway chunks carry
/// `u32::MAX` (they are cells, not BVH subtrees — the antichain property is
/// deliberately given up on the sway tail, which is why `blas_split::self_test`
/// keeps gating the UNSPLIT planner and this module gates its own product).
///
/// Returns `None` (plan untouched, bit-identical) when the mask marks
/// nothing. Deterministic: bucket order is the sorted grid key (BTreeMap),
/// triangle order inside a bucket follows `packed_tris` order.
pub fn split_plan(
    plan: &mut crate::blas_split::BlasPlan,
    scene: &Scene,
    leaf_mat: &[bool],
    max_prims: u32,
) -> Option<SwaySplit> {
    let is_leaf =
        |t: u32| leaf_mat.get(scene.tri_mat[t as usize] as usize).copied().unwrap_or(false);
    if !plan.packed_tris.iter().any(|&t| is_leaf(t)) {
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
    let leaf_tris: Vec<u32> = plan.packed_tris.iter().copied().filter(|&t| is_leaf(t)).collect();
    let mut cell = (SWAY_CELL_K * scale).max(1e-6);
    loop {
        let mut keys: Vec<(i32, i32, i32)> =
            leaf_tris.iter().map(|&t| key_at(centroid(t), cell)).collect();
        keys.sort_unstable();
        keys.dedup();
        if keys.len() <= MAX_CELLS {
            break;
        }
        cell *= 2.0;
    }

    // Bucket leaf tris by cell — BTreeMap so the appended chunk order is a
    // pure function of the geometry (the determinism contract).
    let mut buckets: std::collections::BTreeMap<(i32, i32, i32), Vec<u32>> =
        std::collections::BTreeMap::new();
    for &t in &leaf_tris {
        buckets.entry(key_at(centroid(t), cell)).or_default().push(t);
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
    let cap = max_prims.max(1) as usize;
    for (k, tris) in &buckets {
        let center = cmin
            + Vec3A::new(
                (k.0 as f32 + 0.5) * cell,
                (k.1 as f32 + 0.5) * cell,
                (k.2 as f32 + 0.5) * cell,
            );
        let amp = SWAY_AMP_K * scale * height_factor(center.y, cmin.y, cmax.y);
        // A cell over the BLAS cap splits into runs — several instances share
        // one cell anchor and translate identically (coarse, never wrong).
        let mut off = 0;
        while off < tris.len() {
            let take = (tris.len() - off).min(cap);
            packed.extend_from_slice(&tris[off..off + take]);
            base.push(packed.len() as u32);
            node.push(u32::MAX);
            cells.push(SwayCell { center, amp });
            off += take;
        }
    }
    debug_assert_eq!(packed.len(), plan.packed_tris.len());
    plan.packed_tris = packed;
    plan.chunk_base = base;
    plan.chunk_node = node;
    Some(SwaySplit { first_chunk, cells, scale, cell })
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

/// Bake every cell's translation for one frame (the fireflies `bake` shape —
/// the one caller of `translation` outside the self-test).
pub fn translations(cells: &[SwayCell], scale: f32, time: f32) -> Vec<Vec3A> {
    cells
        .iter()
        .enumerate()
        .map(|(i, c)| translation(c, i as u32, scale, time))
        .collect()
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

    // -- the split's structural contracts, on a synthetic all-leaf mask over
    // the session's real tree (mask-independent properties).
    let cap = crate::blas_split::DEFAULT_MAX_PRIMS;
    let base = crate::blas_split::plan(bvh, cap);
    let n_tris = base.packed_tris.len();
    if n_tris == 0 {
        return Err("empty plan".into());
    }
    let all = vec![true; scene.materials.len()];
    let none = vec![false; scene.materials.len()];

    // Off arm: an all-false mask must leave the plan BYTE-identical.
    {
        let mut p = crate::blas_split::plan(bvh, cap);
        if split_plan(&mut p, scene, &none, cap).is_some() {
            return Err("split_plan: empty mask must return None".into());
        }
        if p.packed_tris != base.packed_tris
            || p.chunk_base != base.chunk_base
            || p.chunk_node != base.chunk_node
        {
            return Err("split_plan: empty mask must leave the plan untouched".into());
        }
    }

    // On arm: exact partition, mask routing, no empty chunks, cap held,
    // cells within the cap, determinism.
    let mut p = crate::blas_split::plan(bvh, cap);
    let Some(sp) = split_plan(&mut p, scene, &all, cap) else {
        return Err("split_plan: all-leaf mask produced no split".into());
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
    {
        let mut q = crate::blas_split::plan(bvh, cap);
        let sq = split_plan(&mut q, scene, &all, cap).ok_or("determinism re-split vanished")?;
        if q.packed_tris != p.packed_tris
            || q.chunk_base != p.chunk_base
            || q.chunk_node != p.chunk_node
            || sq.cells != sp.cells
        {
            return Err("split_plan is not deterministic".into());
        }
    }

    // Mixed mask on the real tree when the scene HAS leaf materials (San
    // Miguel/bistro --check): marked tris land in the sway tail, unmarked in
    // the static head.
    if mask.iter().any(|&m| m) {
        let mut q = crate::blas_split::plan(bvh, cap);
        if let Some(sq) = split_plan(&mut q, scene, &mask, cap) {
            for i in 0..q.chunks() {
                let sway = i as u32 >= sq.first_chunk;
                for &t in q.tris(i) {
                    let leaf = mask[scene.tri_mat[t as usize] as usize];
                    if leaf != sway {
                        return Err(format!(
                            "tri {t} (leaf={leaf}) landed in the wrong half (chunk {i})"
                        ));
                    }
                }
            }
        }
    }

    // -- motion: the displacement bound, height-band pinning, determinism,
    // and time-variation — swept across `--foliage-amp` multipliers through
    // `translation_with` (never the global: the session's own setting must
    // not move under a gate run).
    let scale = sp.scale;
    let mut moved = false;
    for (i, c) in sp.cells.iter().enumerate().take(64) {
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
        // A floor-band cell (amp 0) must be EXACTLY pinned — flutter included
        // (nothing at ground level may be dragged through the floor).
        let pinned = SwayCell { center: c.center, amp: 0.0 };
        if translation_with(&pinned, i as u32, scale, 7.3, 4.0) != Vec3A::ZERO {
            return Err("amp-0 cell must not move at all".into());
        }
    }
    if !moved {
        return Err("no cell moved anywhere in the sweep — the field is dead".into());
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
