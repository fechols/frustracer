use crate::bvh::{Bvh, Hit, Ray};
use crate::camera::CamBasis;
use crate::dlss;
use crate::frustum::{self, MAX_CUT};
use crate::overlay::{self, KIND_COARSE, KIND_LEAF, KIND_SKY};
use crate::replay;
use crate::scene::Scene;
use crate::shade::{self, Quality};
use crate::stats::{LocalStats, Stats};
use crate::temporal::{self, TemporalCache};
use crate::tone;
use glam::Vec3A;
use half::f16;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// Tiles at or below this size stop subdividing and trace per-pixel rays.
///
/// **32, not 8** — measured on THE WORLD (34.4M tris) on an Arc Pro B70 at
/// native 1080p, `--gpu-timing` medians over ~100 windows of 120 frames:
/// ```text
///                        (8, 32)   (32, 256)
///   at rest (replay)      4.257      3.615     -15.1%   frame span
///     leaf                2.029      1.668     -17.8%
///     sky                 0.914      0.664     -27.4%
///   moving (producing)    5.590      4.383     -21.6%   frame span
///     level 0..7 -> 0..5  1.372      0.674     -51%
/// ```
/// and NEUTRAL on a 4090 (3.883 -> 3.907, inside a 0.25 ms IQR; its frame is
/// dominated by DLSS-RR at 2.4 ms, so the tracer barely moves the span).
///
/// The mechanism is dispatch shape, not the quadtree's product. A coarser
/// frontier proves LESS space empty — and still wins, because `depth_full`
/// drops from 8 to 6, halving the ladder, while a ~540-px leaf rect genuinely
/// feeds `LEAF_GROUP` = 256 lanes where a ~32-px one only idles them. `sky`
/// gains for the same reason: fewer, larger proven-empty rects amortize
/// `cs_sky` better. **The two constants MUST move together** — 256 lanes at
/// the old 8-px frontier measured +21% on the B70 (see trace::LEAF_GROUP).
///
/// KNOWN COST, measured not assumed: temporal sky reuse under PURE YAW stops
/// firing at this frontier (a tile's query region is 4x wider per axis, so it
/// far less often lies wholly inside the old sky region). Static sky reuse is
/// unaffected. `--check` therefore pins the temporal family at `TEMPORAL_TILE`
/// — see `set_leaf_tile`.
pub const LEAF_TILE: usize = 32;

/// The frontier the temporal algorithm's structural must-fires are written
/// against. `--check` pins the temporal family here so those gates keep their
/// teeth at the frontier they were tuned for, while the rest of the suite runs
/// the shipping `LEAF_TILE`. Gating the algorithm and gating the shipping
/// config are two different jobs; conflating them would either weaken the
/// must-fires or freeze the constant.
pub const TEMPORAL_TILE: usize = 8;

static LEAF_TILE_N: AtomicU32 = AtomicU32::new(0);

/// Set the quadtree's leaf-rect cutoff (0 = "not yet resolved", which makes the
/// first `leaf_tile()` read FR_LEAF / the default).
///
/// An atomic rather than a `OnceLock` — the `texture::set_aniso` /
/// `dxr::set_inline_mode` knob idiom — because `--check` legitimately needs to
/// re-pin it between passes. Everything that consumes the frontier derives it
/// per call (`depth_full`, the temporal cell tree, the injected
/// `#define LEAF_TILE`), so a re-pin between passes is coherent; do NOT move
/// it while a tracer built against an injected value is live.
pub fn set_leaf_tile(n: usize) {
    LEAF_TILE_N.store(n as u32, Relaxed);
}

/// R&D lever (FR_LEAF, default LEAF_TILE): the quadtree's leaf-rect cutoff, so
/// the subdivision depth can be swept without a rebuild. Smaller = narrower
/// leaf frustums (deeper distance penetration, more tiles provable as sky) at
/// 4x the tiles per level down and 1/4 the pixels to amortize each tile's
/// bound-query + refine_cut over. The GPU gets the same number as an injected
/// `#define LEAF_TILE` so both intersectors agree on the frontier.
pub fn leaf_tile() -> usize {
    let n = LEAF_TILE_N.load(Relaxed);
    if n != 0 {
        return n as usize;
    }
    let v = std::env::var("FR_LEAF")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| n.is_power_of_two() && *n >= 1 && *n <= 64)
        .unwrap_or(LEAF_TILE);
    LEAF_TILE_N.store(v as u32, Relaxed);
    v
}
/// Tiles larger than this spawn their quadrants as rayon tasks; smaller ones
/// recurse sequentially (task granularity, not correctness).
const SPAWN_MIN: usize = 32;

pub struct FrameCtx<'a> {
    pub scene: &'a Scene,
    pub bvh: &'a Bvh,
    pub cam: CamBasis,
    pub q: Quality,
    /// Accumulation frame index. frame == 0 stores (implicit clear), else adds.
    pub frame: u32,
    pub jitter: bool,
    pub rw: usize,
    pub rh: usize,
    /// Linear-RGB accumulation, 3 × AtomicU32 (f32 bits) per pixel. Writes are
    /// tile-disjoint, so plain relaxed load+store is race-free.
    pub accum: &'a [AtomicU32],
    /// Per-pixel quadtree info (depth | kind) for the debug overlay.
    pub info: &'a [AtomicU32],
    /// Per-pixel primary hit t (f32 bits, INFINITY = miss/sky) for verification.
    pub tbuf: &'a [AtomicU32],
    pub stats: &'a Stats,
    pub sun: Vec3A,
    /// Per-frame cloud state (src/clouds.rs). main.rs owns the clock; every
    /// headless context pins `Clouds::check` so gate pairs compare one sky.
    pub clouds: crate::clouds::Clouds,
    /// Per-frame firefly state (src/fireflies.rs) — poses baked once per
    /// frame on the same clock as `clouds`; `count == 0` (every day session)
    /// is the structural off state. Consumed by the primary shade path and
    /// the display glow only — never the gathers (the stars rule).
    pub fireflies: crate::fireflies::Fireflies,
    /// This frame's temporal cache to fill (per-node tc / sky markers).
    pub tcache_cur: Option<&'a TemporalCache>,
    /// Ring of previous frames' caches, NEWEST FIRST, each paired with the
    /// exact basis it was traced with — consulted before each tile's bound
    /// query for a t_start head start (older entries answer regions that
    /// panned off the newest screen and back). Empty slice = no reuse. The
    /// newest entry must be the last producing full-res hybrid frame (see
    /// main.rs wiring).
    pub tcache_prev: &'a [(&'a TemporalCache, CamBasis)],
    /// false => `splat` always stores (every frame is a fresh 1-spp frame;
    /// DLSS-RR is the temporal integrator). `frame` still advances so the
    /// per-pixel RNG decorrelates across frames — pinning frame to 0 would
    /// freeze the noise pattern, which the denoiser would treat as signal.
    pub accumulate: bool,
    /// DLSS G-buffers, written at the primary-hit fill sites. None (all
    /// legacy paths) costs one never-taken branch per pixel.
    pub gbuf: Option<&'a dlss::GBufs>,
    /// FSR Ray Regeneration signal buffers (FSR mode only), written at the
    /// same fill sites — requires `gbuf` to also be Some (the split reads
    /// `PrimarySurface`, which is only captured when `gbuf` is on).
    pub fsr_buf: Option<&'a crate::fsr::FsrBufs>,
    /// Previous frame's camera basis for motion vectors (independent of the
    /// temporal cache's tprev_basis — different contract).
    pub prev_cam: Option<CamBasis>,
    /// Frame-uniform sub-pixel jitter offset in [-0.5, 0.5) (DLSS mode);
    /// None => the legacy per-pixel rng jitter controlled by `jitter`.
    pub frame_jitter: Option<(f32, f32)>,
    /// Primary samples per pixel per frame (`--spp`, 1..=dlss::MAX_SPP).
    /// Sample 0 is the frame's reported sample (today's position rule, today's
    /// rng seed, the only one that writes tbuf/info/G-buffers); samples 1..spp
    /// take `dlss::jitter_for_sample(frame, k)` inside the SAME pixel — hence
    /// inside the tile frustum — and ride the inherited t_start/cut like any
    /// leaf ray. The N colors are averaged and splatted ONCE, so a frame still
    /// contributes exactly one sample of weight to `accum` and `resolve`'s
    /// divisor stays a frame count.
    pub spp: u32,
    /// Which sample writes the per-pixel side channels (tbuf/info/G-buffers).
    /// 0 in every real frame; `--check` sweeps it 0..spp so `verify` can gate
    /// EVERY sample's ray against a tmin=0 reference, not just sample 0's.
    pub primary_sample: u32,
    /// Adaptive shading rate (XeSS mode only): leaf tiles shade in 2×2 cells
    /// that share visibility rays where coherent and supersample where noisy.
    /// Visibility stays per-pixel regardless — tbuf/G-buffers are identical
    /// to a non-adaptive frame; only radiance sampling changes.
    pub adaptive: bool,
    /// Hemi sharing (fb frames only; --no-hemi-share is the kill switch):
    /// leaf tiles shade in 2×2 cells whose coherent pixels (same triangle,
    /// bit-equal normal, measured apex-spread qualifiers) capture ONE padded
    /// hemisphere tree at the representative (`hemi::share_capture`) and all
    /// consume its record — folded analytic empties plus per-leaf (tc, cut)
    /// seeds. Visibility stays per-pixel; every pixel shoots its own rays.
    pub hemi_share: bool,
    /// Deferred material-sorted shading (--defer-shade, plain leaf path
    /// only): leaf tiles trace their pixels but don't shade; runs whose every
    /// pixel hit the SAME material merge up the quadtree (≤ DEFER_CAP px =
    /// 64×64) and flush as single sequential bursts so one material's
    /// textures stay cache-hot for the whole run. Bit-identical to the fused
    /// path by construction: the per-pixel rng rides the record, and splat /
    /// meta / G-buffer writes are single-writer and order-free.
    pub defer_shade: bool,
    /// Structure recorder for static-frame replay: Some only on full-depth
    /// uncapped hybrid frames (main.rs gates it). trace_tile records every
    /// terminal (leaf rect + inherited t_start + cut, sky rect) so a later
    /// bit-equal-basis frame can `render_frame_replay` with zero queries.
    pub replay_rec: Option<&'a replay::ReplayCache>,
    /// This frame's cut store to fill (each Split node's refine_cut output),
    /// produced in lockstep with `tcache_cur`.
    pub cut_cur: Option<&'a temporal::CutStore>,
    /// The previous PRODUCING frame's cut store — must pair with the newest
    /// `tcache_prev` ring entry (same frame, same basis). Some = cut
    /// adoption on: a tile whose direction hull is contained in an old
    /// node's cone skips its bound query and re-refines that node's cut.
    /// None is the kill switch (distance seeds still work).
    pub cut_prev: Option<&'a temporal::CutStore>,
    /// A/B/C lever (--discard-seeds): run every temporal lookup at full cost
    /// (ring retries included, cells counted) but consume NOTHING — no sky
    /// fill, no t_start seed, no query skip. Trace results are then identical
    /// to --no-temporal while still paying lookup + cache/cut production, so
    /// wall-clock differences isolate the machinery's cost from its benefit:
    /// (this − --no-temporal) = pure cost, (default − this) = gross benefit.
    pub discard_seeds: bool,
}

impl FrameCtx<'_> {
    /// Effective samples per pixel. Pinned to 1 on hemisphere-bounce (fb)
    /// frames: the bounce integrator converges by frame accumulation (still
    /// frames only), and N samples per pixel would mean N hemisphere trees per
    /// pixel — N× the cost for an estimator that already has a cheaper
    /// convergence path, plus N hemi points per pixel on the GPU (which would
    /// blow `cap_hemi_pt` and the hemi accounting gates). Upscaler frames pin
    /// fb OFF anyway, so this costs nothing on the path multi-sampling is for.
    #[inline(always)]
    fn spp(&self) -> u32 {
        if self.q.fb.ao || self.q.fb.gi {
            1
        } else {
            self.spp.max(1)
        }
    }

    #[inline(always)]
    fn splat(&self, x: usize, y: usize, c: Vec3A) {
        let i = (y * self.rw + x) * 3;
        if self.frame == 0 || !self.accumulate {
            self.accum[i].store(c.x.to_bits(), Relaxed);
            self.accum[i + 1].store(c.y.to_bits(), Relaxed);
            self.accum[i + 2].store(c.z.to_bits(), Relaxed);
        } else {
            for (k, v) in [c.x, c.y, c.z].into_iter().enumerate() {
                let a = &self.accum[i + k];
                a.store((f32::from_bits(a.load(Relaxed)) + v).to_bits(), Relaxed);
            }
        }
    }

    #[inline(always)]
    fn store_meta(&self, x: usize, y: usize, t: f32, depth: u32, kind: u32) {
        let i = y * self.rw + x;
        self.tbuf[i].store(t.to_bits(), Relaxed);
        self.info[i].store(overlay::pack_info(depth, kind), Relaxed);
    }
}

pub fn render_frame(ctx: &FrameCtx, hybrid: bool) {
    if hybrid {
        crate::zone!("trace-full");
        // Whole-tree root cut in the session's live id space (binary `[0]` or
        // the wide root slots) — every cut below is refine output in the same
        // space, translated back to binary ids only where a ray seeds.
        let root = crate::ftree::Accel::for_tiles(ctx.bvh).root_cut();
        flush_pend(ctx, trace_tile(ctx, 0, 0, ctx.rw, ctx.rh, 0.0, 0, 0, root, u32::MAX));
        crate::oracle::frame_end(ctx.rw, ctx.rh);
    } else {
        crate::zone!("trace-plain");
        // Plain per-pixel reference: the ground truth and the A/B baseline.
        (0..ctx.rh).into_par_iter().for_each(|y| {
            let mut ls = LocalStats::default();
            for x in 0..ctx.rw {
                shade_pixel(ctx, x, y, 0.0, 0, KIND_LEAF, None, &mut ls);
            }
            ctx.stats.add(&ls);
        });
    }
}

/// Outcome of one non-leaf tile's frustum work.
enum TileStep {
    Sky,
    /// Subdivide: children inherit `tc` as their ray tmin and `cut[..len]` as
    /// their BVH node cut.
    Split { tc: f32, cut: [u32; MAX_CUT], len: usize },
}

/// The per-tile frustum work: bound the nearest possible hit over the
/// inherited cut, then refine the cut for the children.
fn tile_step(
    ctx: &FrameCtx,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    t_start: f32,
    depth: u32,
    cut_in: &[u32],
    ls: &mut LocalStats,
) -> TileStep {
    let f = ctx.cam.tile_frustum(x0, y0, x1, y1);
    // The frustum structure: the 8-wide tree by default (the GPU tile
    // kernels' measured-win regime, transplanted), binary under the
    // --no-ftree(-tiles) levers. `visits` counts whichever tree's nodes.
    let accel = crate::ftree::Accel::for_tiles(ctx.bvh);
    let mut visits = 0u64;
    ls.tiles += 1;
    ls.frustum_queries += 1;
    let result = accel.nearest_within(&f, t_start, f32::INFINITY, cut_in, &mut visits);
    let step = match result {
        None => TileStep::Sky,
        Some(t_safe) => {
            let (advanced, tc) = frustum::advance_tc(t_safe, t_start, ctx.scene.eps);
            if !advanced {
                // Blocked at the inherited distance (typically a large flat
                // AABB like the ground). Still subdivide: children's smaller
                // frustums may exclude the blocker entirely — that is exactly
                // how sky tiles emerge. Worst case is bounded by the leaf cutoff.
                ls.blocked_queries += 1;
            }
            let mut cut = [0u32; MAX_CUT];
            let len = accel.refine_cut(&f, tc, f32::INFINITY, cut_in, &mut cut, &mut visits, &mut ls.cut_overflows);
            ls.cut_len_sum += len as u64;
            if crate::oracle::armed() {
                // Read-only sizing probe (Q1/Q4/Q5), taken after the real
                // refine on exactly what the children will inherit. Cut ids
                // live in the live accel's id space; the probe reads binary
                // node boxes, so translate them the way a ray seed does.
                let mut roots = [0u32; MAX_CUT];
                let roots = accel.ray_roots(&cut[..len], &mut roots);
                crate::oracle::probe_split(
                    &ctx.cam, ctx.scene, ctx.bvh, &f, x0, y0, x1, y1, depth, roots,
                );
            }
            TileStep::Split { tc, cut, len }
        }
    };
    ls.frustum_nodes += visits;
    step
}

/// The frustracer core: trace the tile's frustum until it could hit geometry,
/// then split into 4 quadrants that inherit the proven-empty distance and the
/// refined node cut. Tiles that reach `max_depth` unresolved (not sky, not
/// leaf) are sparse-filled (real point samples + cell-flood fallback);
/// `u32::MAX` means uncapped.
fn trace_tile(
    ctx: &FrameCtx,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    t_start: f32,
    depth: u32,
    path: u32,
    cut_in: &[u32],
    max_depth: u32,
) -> TilePend {
    if x0 >= x1 || y0 >= y1 {
        return TilePend::Done;
    }
    let (w, h) = (x1 - x0, y1 - y0);
    if w <= leaf_tile() && h <= leaf_tile() {
        if let Some(rec) = ctx.replay_rec {
            rec.push_leaf(x0, y0, x1, y1, t_start, depth, cut_in);
        }
        // Deferral replaces only the plain per-pixel branch; the adaptive and
        // hemi-share cell machineries keep their own paths (and replay frames
        // never come through here — they shade the recorded leaf list flat).
        // (--spp stays on the fused path: a deferred leaf stages ONE Traced per
        // pixel, so deferring a multi-sampled tile would silently drop every
        // sample but the first. The two levers compose by the fused path
        // winning — coarser, never wrong.)
        if ctx.defer_shade
            && ctx.spp() == 1
            && !ctx.adaptive
            && !(ctx.hemi_share && (ctx.q.fb.ao || ctx.q.fb.gi))
        {
            return defer_leaf(ctx, x0, y0, x1, y1, t_start, depth, cut_in);
        }
        shade_tile(ctx, x0, y0, x1, y1, t_start, depth, KIND_LEAF, cut_in);
        if crate::oracle::armed() {
            // Q3, after shading: tbuf now holds this tile's own hit distances,
            // so the farthest of them is the tile's occlusion frontier, and a
            // cut entry beyond it was carried down the ladder for nothing.
            let mut t_max = 0.0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let t = f32::from_bits(ctx.tbuf[y * ctx.rw + x].load(Relaxed));
                    if t.is_finite() {
                        t_max = t_max.max(t);
                    }
                }
            }
            let f = ctx.cam.tile_frustum(x0, y0, x1, y1);
            let mut roots = [0u32; MAX_CUT];
            let roots = crate::ftree::Accel::for_tiles(ctx.bvh).ray_roots(cut_in, &mut roots);
            crate::oracle::probe_leaf(ctx.bvh, &f, depth, roots, t_max);
        }
        return TilePend::Done;
    }
    // After the leaf check, so a cap >= the leaf depth is exactly uncapped.
    // Sparse-fill uses the inherited cut and t_start — same as the split would.
    // No temporal probe or store here: the inherited t_start already carries
    // the parent's (possibly seeded) tc, and its cache entry covers this tile.
    if depth >= max_depth {
        // Recording is gated on uncapped frames, so this arm is structurally
        // unreachable while recording — poison defensively (a capped terminal
        // has no replayable record; replaying around it would drop pixels).
        if let Some(rec) = ctx.replay_rec {
            rec.poison();
        }
        let mut ls = LocalStats::default();
        sparse_fill(ctx, x0, y0, x1, y1, t_start, depth, cut_in, &mut ls);
        ctx.stats.add(&ls);
        return TilePend::Done;
    }

    let mut ls = LocalStats::default();
    // Temporal probe: harvest a proven-empty head start from the previous
    // frame's quadtree before touching the BVH. Only the primary-path t_start
    // is affected — secondary rays never see it.
    let mut t0 = t_start;
    // (verbatim old cut to refine instead of cut_in, skip-chain age) — Some
    // means the bound query is skipped this tile.
    let mut adopted: Option<(Option<(u32, u32)>, u32)> = None;
    if !ctx.tcache_prev.is_empty() {
        let seed = temporal::lookup(ctx.tcache_prev, ctx.cut_prev, &ctx.cam, ctx.rw, ctx.rh, x0, y0, x1, y1, t_start, depth, path, &mut ls);
        // --discard-seeds: the lookup ran at full cost (cells counted above),
        // but nothing may be consumed — the trace below must be identical to
        // a --no-temporal frame for the A/B/C differencing to mean anything.
        let seed = if ctx.discard_seeds { temporal::Seed::None } else { seed };
        match seed {
            temporal::Seed::Sky => {
                // The whole frustum was proven empty last frame; still true.
                if let Some(cur) = ctx.tcache_cur {
                    cur.store(depth, path, f32::INFINITY);
                }
                if let Some(rec) = ctx.replay_rec {
                    rec.push_sky(x0, y0, x1, y1, depth);
                }
                ls.temporal_sky_tiles += 1;
                ctx.stats.add(&ls);
                fill_sky(ctx, x0, y0, x1, y1, depth);
                return TilePend::Done;
            }
            temporal::Seed::T(t) => {
                if t > t0 {
                    ls.temporal_seeds += 1;
                    t0 = t;
                }
            }
            temporal::Seed::Skip { t, age } => {
                // The completed old min predicts this tile's bound query is
                // pinned by real geometry (the blocked regime) — skip it and
                // refine the tile's own inherited cut at t. Nothing old is
                // consumed by the refine; the prediction only chooses when
                // skipping is profitable.
                if t > t0 {
                    ls.temporal_seeds += 1;
                    t0 = t;
                }
                adopted = Some((None, age));
            }
            temporal::Seed::TCut { t, off, len, age } => {
                // Identical basis: the node's own old cut is exactly valid —
                // skip the query and refine from it (already this node's
                // tightness, not the parent's).
                if t > t0 {
                    ls.temporal_seeds += 1;
                    t0 = t;
                }
                adopted = Some((Some((off, len)), age));
            }
            temporal::Seed::None => {}
        }
    }
    // Skipping tiles run only refine_cut; `cut_age` chains the skip count so
    // temporal.rs's MAX_ADOPT_AGE forces a real query before the
    // un-advanced claims decay.
    let (step, cut_age) = match adopted {
        Some((old, age)) => (adopt_step(ctx, x0, y0, x1, y1, t0, old, cut_in, &mut ls), age + 1),
        None => (tile_step(ctx, x0, y0, x1, y1, t0, depth, cut_in, &mut ls), 0),
    };
    match step {
        TileStep::Sky => {
            // Composed claim: nothing outside ball(origin, t0) by this query,
            // nothing inside it by the inherited/seeded claim — the whole
            // cone is empty, which is what +INF asserts to the next frame.
            if let Some(cur) = ctx.tcache_cur {
                cur.store(depth, path, f32::INFINITY);
            }
            if let Some(rec) = ctx.replay_rec {
                rec.push_sky(x0, y0, x1, y1, depth);
            }
            ctx.stats.add(&ls);
            fill_sky(ctx, x0, y0, x1, y1, depth);
            TilePend::Done
        }
        TileStep::Split { tc, cut, len } => {
            if let Some(cur) = ctx.tcache_cur {
                cur.store(depth, path, tc);
            }
            if let Some(ccur) = ctx.cut_cur {
                // Store-before-recurse, like the claim above. An adopted
                // node's re-refined cut is a valid standalone (claim, PVS)
                // pair for THIS frustum, so chained adoption stays sound —
                // the age caps its quality decay, not its correctness.
                if len > 0 && !ccur.store(depth, path, &cut[..len], cut_age) {
                    ls.temporal_cut_arena_full += 1;
                }
            }
            ctx.stats.add(&ls);
            // A Some bound with an empty cut should be impossible (the nearest
            // leaf survives the tc ball-cull); never degrade toward sky. (An
            // adopted refine CAN empty — adopt_step maps that to Sky, a proof.)
            debug_assert!(len > 0, "refine_cut emptied a non-sky tile");
            let child: &[u32] = if len > 0 { &cut[..len] } else { cut_in };
            let xm = x0 + w / 2;
            let ym = y0 + h / 2;
            let d = depth + 1;
            // Child paths (2 bits per level: TL=0 TR=1 BL=2 BR=3) — must match
            // temporal::rect_for_path, which replays these splits.
            let p = path << 2;
            if w.max(h) > SPAWN_MIN {
                let ((k0, k1), (k2, k3)) = rayon::join(
                    || {
                        rayon::join(
                            || trace_tile(ctx, x0, y0, xm, ym, tc, d, p, child, max_depth),
                            || trace_tile(ctx, xm, y0, x1, ym, tc, d, p | 1, child, max_depth),
                        )
                    },
                    || {
                        rayon::join(
                            || trace_tile(ctx, x0, ym, xm, y1, tc, d, p | 2, child, max_depth),
                            || trace_tile(ctx, xm, ym, x1, y1, tc, d, p | 3, child, max_depth),
                        )
                    },
                );
                merge_pend(ctx, [k0, k1, k2, k3])
            } else {
                let k0 = trace_tile(ctx, x0, y0, xm, ym, tc, d, p, child, max_depth);
                let k1 = trace_tile(ctx, xm, y0, x1, ym, tc, d, p | 1, child, max_depth);
                let k2 = trace_tile(ctx, x0, ym, xm, y1, tc, d, p | 2, child, max_depth);
                let k3 = trace_tile(ctx, xm, ym, x1, y1, tc, d, p | 3, child, max_depth);
                merge_pend(ctx, [k0, k1, k2, k3])
            }
        }
    }
}

/// Deferred-shading bucket cap: a merged same-material run stops growing at
/// 64×64 px. This is the MAXIMUM SHADING TILE — it bounds both the record
/// memory in flight and the sequential burst a single flush shades, so a
/// pathological case (the whole ground plane one material) becomes many
/// 64×64 flushes stolen across the pool, never one core shading the screen.
pub const DEFER_CAP: usize = 4096;

/// One traced-but-unshaded pixel in a deferred segment — 48 B instead of
/// buffering the whole `Traced` (~144 B): `dir` and `ray` are pure functions
/// of (fx, fy) and the frame camera, so the flush reconstructs them
/// bit-identically; only the rng STATE (advanced past any jitter draws) must
/// be carried. `t_start`/`depth` ride per record because a merged run spans
/// leaves with different inherited claims.
struct DeferPx {
    x: u16,
    y: u16,
    depth: u32,
    fx: f32,
    fy: f32,
    t_start: f32,
    /// `tri == u32::MAX` encodes a miss (only reachable through the
    /// mixed-leaf inline path — deferred segments are all-hit by
    /// construction).
    hit: Hit,
    rng: fastrand::Rng,
}

/// `trace_tile`'s return: `Done` = the subtree fully shaded (and splatted)
/// itself; `Pend` = a list of single-material pixel segments (one per
/// uniform leaf, in Z-order) whose shading is deferred upward. Ancestors
/// merge by moving SEGMENT POINTERS only — records are written once at the
/// leaf and read once at the flush; an early version that `append`ed record
/// vectors up the chain re-copied every record per merge level (~GB/frame)
/// and tripled the frame time.
/// `#[must_use]` is load-bearing, not lint hygiene: a `Pend` holds the ONLY
/// copy of its pixels' traced hits and rng state, so dropping one silently
/// leaves that rect unshaded (stale/black) rather than crashing. Every
/// recursion site must route its result to `merge_pend` or `flush_pend`.
#[must_use]
enum TilePend {
    Done,
    Pend { n: usize, segs: Vec<(u32, Vec<DeferPx>)> },
}

/// Deferred-shading leaf (--defer-shade, plain path only): trace every pixel
/// now — visibility, rng streams and recorded structure identical to the
/// fused path — and hand the records upward unshaded if all pixels hit ONE
/// material. Mixed-material or sky-containing leaves shade inline from the
/// already-traced records (the shade_cell trace-first precedent), which is
/// bit-identical because each pixel's rng is self-contained in its Traced.
fn defer_leaf(
    ctx: &FrameCtx,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    t_start: f32,
    depth: u32,
    cut: &[u32],
) -> TilePend {
    let mut ls = LocalStats::default();
    let w = x1 - x0;
    // Slot-ref cut -> binary ray roots, once per leaf (the shade_tile /
    // sparse_fill convention): this is the THIRD site where an inherited cut
    // seeds primary rays, and deferred pixels are ordinary cut-seeded primary
    // rays. Identity when the tile path runs binary.
    let mut rbuf = [0u32; MAX_CUT];
    let cut = crate::ftree::Accel::for_tiles(ctx.bvh).ray_roots(cut, &mut rbuf);
    // Probe the first pixel: a deferrable leaf is uniform in ONE textured
    // material, so a sky or untextured-material first hit already proves
    // this leaf shades inline — take the fused trace+shade path with zero
    // staging (the default scene must run at baseline speed).
    // spp == 1 here (trace_tile gates deferral on it), so this is sample 0.
    let mut first = trace_primary(ctx, x0, y0, t_start, Some(cut), &mut ls, SampleId::First);
    let mat = match &first.hit {
        Some(h) => ctx.scene.tri_mat[h.tri as usize],
        None => u32::MAX,
    };
    let deferrable = mat != u32::MAX && ctx.scene.materials[mat as usize].any_tex();
    let shade_one = |i: usize, tr: &mut Traced, ls: &mut LocalStats| {
        let (x, y) = (x0 + i % w, y0 + i / w);
        shade_traced(ctx, x, y, t_start, depth, KIND_LEAF, tr, None, ls, true, true, shade::VisCtl::Off, None);
    };
    if !deferrable {
        // Counted like any other inline-shaded leaf (`defer_mixed` is "shaded
        // inline", not "mixed material") — otherwise a scene that defers
        // nothing prints no `defer:` segment at all, which reads as "the flag
        // did nothing" when it in fact tried and rejected every leaf.
        ls.defer_mixed += 1;
        shade_one(0, &mut first, &mut ls);
        for y in y0..y1 {
            for x in x0..x1 {
                if (x, y) != (x0, y0) {
                    shade_pixel(ctx, x, y, t_start, depth, KIND_LEAF, Some(cut), &mut ls);
                }
            }
        }
        ctx.stats.add(&ls);
        return TilePend::Done;
    }
    // Textured first hit: stage full Traced records (registers → one local
    // Vec) while the leaf stays uniform; bail to direct shading on the first
    // mismatch (no reconstruction — the staged Traced are shaded as-is).
    let n = w * (y1 - y0);
    let mut trs: Vec<Traced> = Vec::with_capacity(n);
    trs.push(first);
    let mut uniform = true;
    'trace: for y in y0..y1 {
        for x in x0..x1 {
            if (x, y) == (x0, y0) {
                continue;
            }
            let tr = trace_primary(ctx, x, y, t_start, Some(cut), &mut ls, SampleId::First);
            let same = matches!(&tr.hit, Some(h) if ctx.scene.tri_mat[h.tri as usize] == mat);
            trs.push(tr);
            if !same {
                uniform = false;
                break 'trace;
            }
        }
    }
    if uniform {
        ctx.stats.add(&ls);
        let px: Vec<DeferPx> = trs
            .into_iter()
            .enumerate()
            .map(|(i, tr)| DeferPx {
                x: (x0 + i % w) as u16,
                y: (y0 + i / w) as u16,
                depth,
                fx: tr.fx,
                fy: tr.fy,
                t_start,
                hit: tr.hit.expect("uniform leaf is all-hit"),
                rng: tr.rng,
            })
            .collect();
        return TilePend::Pend { n, segs: vec![(mat, px)] };
    }
    // Mixed: shade what was staged, then finish the rest fused.
    ls.defer_mixed += 1;
    let staged = trs.len();
    for (i, tr) in trs.iter_mut().enumerate() {
        shade_one(i, tr, &mut ls);
    }
    for i in staged..n {
        let (x, y) = (x0 + i % w, y0 + i / w);
        shade_pixel(ctx, x, y, t_start, depth, KIND_LEAF, Some(cut), &mut ls);
    }
    ctx.stats.add(&ls);
    TilePend::Done
}

/// Shade a run of deferred records in order. Exactly `shade_traced` per
/// pixel — same rng stream, same meta/G-buffer/splat writes — just later:
/// `dir`/`ray` are recomputed from (fx, fy), which is bit-identical because
/// `ray_dir` and `Ray::new` are pure functions of the frame camera.
fn shade_deferred(ctx: &FrameCtx, px: &mut [DeferPx], ls: &mut LocalStats) {
    for p in px.iter_mut() {
        let dir = ctx.cam.ray_dir(p.fx, p.fy);
        let mut tr = Traced {
            fx: p.fx,
            fy: p.fy,
            dir,
            ray: Ray::new(ctx.cam.origin, dir),
            hit: (p.hit.tri != u32::MAX).then_some(p.hit),
            rng: p.rng.clone(),
            // Deferred records exist only at spp == 1 (sample 0) and only for
            // HIT pixels — the cloud-phase stratum is never read here.
            k: 0,
        };
        shade_traced(
            ctx,
            p.x as usize,
            p.y as usize,
            p.t_start,
            p.depth,
            KIND_LEAF,
            &mut tr,
            None,
            ls,
            true,
            true,
            shade::VisCtl::Off,
            None,
        );
    }
}

/// Sequential grain for shading a flushed bucket — matches the fused path's
/// SPAWN_MIN subtree (32×32 px). Without this, a cap-size flush was a ~3 ms
/// single-thread chunk on Sponza-class shading and the merge phase turned
/// every 64×64 into an end-of-frame straggler (+29% measured).
const DEFER_PAR: usize = 1024;

/// Flush a pending macro-tile: STABLE-sort the segments by material (stable
/// keeps the Z-order walk within each material group — texture UVs stay
/// spatially coherent inside the group), then shade material by material,
/// splitting the sorted segment list across rayon at DEFER_PAR grain — the
/// material runs survive inside each half; only run boundaries land on a
/// different core. This is the "tiled shader": one material's textures stay
/// hot for its whole run instead of being evicted at every 8×8 boundary.
fn flush_pend(ctx: &FrameCtx, pend: TilePend) {
    if let TilePend::Pend { n, mut segs } = pend {
        let mut ls = LocalStats::default();
        ls.defer_flushes += 1;
        ls.defer_px += n as u64;
        ctx.stats.add(&ls);
        segs.sort_by_key(|(mat, _)| *mat);
        shade_segs(ctx, &mut segs, n);
    }
}

fn shade_segs(ctx: &FrameCtx, segs: &mut [(u32, Vec<DeferPx>)], n: usize) {
    if n > DEFER_PAR && segs.len() > 1 {
        // Split at the pixel-count midpoint (whole segments — a segment is
        // never torn, so its material run and Z-order stay intact).
        let mut acc = 0usize;
        let mut cut = 0usize;
        for (i, s) in segs.iter().enumerate() {
            acc += s.1.len();
            if acc * 2 >= n {
                cut = i + 1;
                break;
            }
        }
        if cut == 0 || cut >= segs.len() {
            cut = segs.len() / 2;
        }
        let (a, b) = segs.split_at_mut(cut);
        let na: usize = a.iter().map(|s| s.1.len()).sum();
        rayon::join(|| shade_segs(ctx, a, na), || shade_segs(ctx, b, n - na));
    } else {
        let mut ls = LocalStats::default();
        for (_, px) in segs.iter_mut() {
            shade_deferred(ctx, px, &mut ls);
        }
        ctx.stats.add(&ls);
    }
}

/// Merge four children's pends: while every child is still pending and the
/// combined run fits the shading-tile cap, concatenate the SEGMENT LISTS
/// (pointer moves, no record copies) — materials may differ; the flush
/// sorts. Otherwise flush each pending child as its own ≤ DEFER_CAP unit.
fn merge_pend(ctx: &FrameCtx, kids: [TilePend; 4]) -> TilePend {
    let mergeable = kids.iter().all(|k| matches!(k, TilePend::Pend { .. }))
        && kids
            .iter()
            .map(|k| match k {
                TilePend::Pend { n, .. } => *n,
                TilePend::Done => 0,
            })
            .sum::<usize>()
            <= DEFER_CAP;
    if mergeable {
        let mut it = kids.into_iter();
        let Some(TilePend::Pend { mut n, mut segs }) = it.next() else { unreachable!() };
        for k in it {
            let TilePend::Pend { n: kn, segs: mut ks } = k else { unreachable!() };
            n += kn;
            segs.append(&mut ks);
        }
        TilePend::Pend { n, segs }
    } else {
        for k in kids {
            flush_pend(ctx, k);
        }
        TilePend::Done
    }
}

/// Temporal query skip (`Seed::Skip` / `Seed::TCut`): the previous frame
/// predicts this tile's bound query cannot meaningfully advance, so only
/// `refine_cut` runs — against the tile's own inherited cut (always valid —
/// nothing old is consumed), or, on an identical basis, against the node's
/// own stored old cut (`old`, exactly valid for the identical cone and
/// already node-tight). Children only ever receive refine_cut output, so
/// staleness cannot accumulate; cost is bounded by the cut size, no
/// root-ward traversal. An emptied refine is a sky PROOF, not the
/// debug_assert bug case: frustum ∖ ball had no surviving subtree, and
/// ball(origin, t0) is empty by the inherited/seeded claim — the whole cone
/// is empty.
fn adopt_step(
    ctx: &FrameCtx,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    t0: f32,
    old: Option<(u32, u32)>,
    cut_in: &[u32],
    ls: &mut LocalStats,
) -> TileStep {
    let f = ctx.cam.tile_frustum(x0, y0, x1, y1);
    let mut buf = [0u32; MAX_CUT];
    let input: &[u32] = match old {
        Some((off, len)) => {
            let cs = ctx.cut_prev.expect("Seed::TCut without a cut store");
            let n = cs.copy_cut(off, len, &mut buf);
            &buf[..n]
        }
        None => cut_in,
    };
    ls.tiles += 1;
    ls.temporal_cut_adopts += 1;
    let mut visits = 0u64;
    let mut cut = [0u32; MAX_CUT];
    // Same structure as tile_step — an adopted cut (this node's own previous
    // refine output) is in the same id space, both levers being startup-set.
    let accel = crate::ftree::Accel::for_tiles(ctx.bvh);
    let out = accel.refine_cut(&f, t0, f32::INFINITY, input, &mut cut, &mut visits, &mut ls.cut_overflows);
    ls.frustum_nodes += visits;
    if out == 0 {
        ls.temporal_adopt_sky += 1;
        TileStep::Sky
    } else {
        ls.cut_len_sum += out as u64;
        TileStep::Split { tc: t0, cut, len: out }
    }
}

fn shade_tile(
    ctx: &FrameCtx,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    t_start: f32,
    depth: u32,
    kind: u32,
    cut: &[u32],
) {
    let mut ls = LocalStats::default();
    // The inherited cut arrives in the frustum structure's id space (wide
    // slot-refs by default). Primary rays seed from BINARY nodes, so map it
    // ONCE per leaf tile — identity when the tile path runs binary. Every
    // consumer below (per-pixel, adaptive cells, hemi-share cells, HOT
    // top-ups, replayed leaves) sees only the translated roots.
    let mut rbuf = [0u32; MAX_CUT];
    let cut = crate::ftree::Accel::for_tiles(ctx.bvh).ray_roots(cut, &mut rbuf);
    // adaptive (XeSS) and fb never co-occur (fb is pinned OFF on upscaler
    // frames), so the two cell loops can share the branch without a tiebreak.
    debug_assert!(!(ctx.adaptive && (ctx.q.fb.ao || ctx.q.fb.gi)));
    if ctx.adaptive {
        let mut cy = y0;
        while cy < y1 {
            let cy1 = (cy + ADAPT_CELL).min(y1);
            let mut cx = x0;
            while cx < x1 {
                let cx1 = (cx + ADAPT_CELL).min(x1);
                shade_cell(ctx, cx, cy, cx1, cy1, t_start, depth, kind, cut, &mut ls);
                cx = cx1;
            }
            cy = cy1;
        }
    } else if ctx.hemi_share && (ctx.q.fb.ao || ctx.q.fb.gi) {
        // One record buffer per tile, re-captured per group — a fresh ~10 KB
        // HemiShare per 2×2 group would spend real frame time on zeroing.
        let mut hrec = crate::hemi::HemiShare::new();
        let mut cy = y0;
        while cy < y1 {
            let cy1 = (cy + ADAPT_CELL).min(y1);
            let mut cx = x0;
            while cx < x1 {
                let cx1 = (cx + ADAPT_CELL).min(x1);
                shade_hemi_cell(ctx, cx, cy, cx1, cy1, t_start, depth, kind, cut, &mut hrec, &mut ls);
                cx = cx1;
            }
            cy = cy1;
        }
    } else {
        for y in y0..y1 {
            for x in x0..x1 {
                shade_pixel(ctx, x, y, t_start, depth, kind, Some(cut), &mut ls);
            }
        }
    }
    ctx.stats.add(&ls);
}

/// One hemi-share 2×2 cell (fb still frames): per-pixel visibility ALWAYS
/// (four real primary rays with the tile's inherited cut/t_start — tbuf and
/// meta identical to a non-shared frame), then ONE padded hemisphere tree
/// captured at the representative when all four hits land on the same
/// triangle with bit-equal shading normals (⇒ bit-identical hemisphere
/// partitions) and the measured η/δ qualifiers pass. Every member — rep
/// included — consumes the record with its own fresh rays; failures (and
/// poisoned captures) shade per-pixel.
#[allow(clippy::too_many_arguments)]
fn shade_hemi_cell(
    ctx: &FrameCtx,
    cx: usize,
    cy: usize,
    cx1: usize,
    cy1: usize,
    t_start: f32,
    depth: u32,
    kind: u32,
    cut: &[u32],
    rec: &mut crate::hemi::HemiShare,
    ls: &mut LocalStats,
) {
    if (cx1 - cx, cy1 - cy) != (ADAPT_CELL, ADAPT_CELL) {
        // Odd-edge remainder: nothing to share within, plain per-pixel.
        for y in cy..cy1 {
            for x in cx..cx1 {
                shade_pixel(ctx, x, y, t_start, depth, kind, Some(cut), ls);
            }
        }
        return;
    }
    let coords = [(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)];
    // Visibility phase first: the group predicate reads real hits.
    let mut tr: [Traced; 4] =
        coords.map(|(x, y)| trace_primary(ctx, x, y, t_start, Some(cut), ls, SampleId::First));
    let sp: [Option<(Vec3A, Vec3A)>; 4] = std::array::from_fn(|i| {
        tr[i].hit.map(|h| shade::surface_point(ctx.scene, &tr[i].ray, &h))
    });
    // Rotate the representative per frame (the shade_cell precedent) so the
    // shared-apex claim shrink averages out temporally.
    let rep = (ctx.frame as usize + cx / ADAPT_CELL + cy / ADAPT_CELL) & 3;

    // Group predicate: same triangle + bit-equal shading normal (the
    // load-bearing half — it makes every member's onb(n) partition
    // bit-identical to the rep's, so per-member PSA still accounts to π),
    // then the measured spread qualifiers.
    let mut shared = false;
    if let (Some(rh), Some((rp, rn))) = (&tr[rep].hit, sp[rep]) {
        let mut delta = 0.0f32;
        let mut eta = 0.0f32;
        let mut ok = true;
        for i in 0..4 {
            if i == rep {
                continue;
            }
            match (&tr[i].hit, sp[i]) {
                (Some(ih), Some((ip, inn))) if ih.tri == rh.tri && inn == rn => {
                    let d = ip - rp;
                    eta = eta.max(rn.dot(d).abs());
                    delta = delta.max(d.length());
                }
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok
            && eta <= ctx.scene.eps * crate::hemi::SHARE_ETA_FRAC
            && delta <= ctx.scene.ao_radius * crate::hemi::SHARE_DELTA_FRAC
        {
            let (t1, t2) = shade::onb(rn);
            crate::hemi::share_capture(
                ctx.scene,
                crate::ftree::Accel::of(ctx.bvh),
                rp,
                rn,
                t1,
                t2,
                ctx.q.fb.depth,
                if ctx.q.fb.gi { f32::INFINITY } else { ctx.scene.ao_radius },
                delta,
                eta,
                ctx.q.fb.gi.then_some(ctx.sun),
                &ctx.clouds,
                rec,
                ls,
            );
            shared = !rec.poisoned;
            if shared {
                ls.hemi_share_groups += 1;
            }
        }
    }
    for i in 0..4 {
        let (x, y) = coords[i];
        if !shared && tr[i].hit.is_some() {
            ls.hemi_share_fallback += 1;
        }
        shade_traced(
            ctx,
            x,
            y,
            t_start,
            depth,
            kind,
            &mut tr[i],
            sp[i],
            ls,
            true,
            true,
            shade::VisCtl::Off,
            if shared { Some(&*rec) } else { None },
        );
    }
}

/// Adaptive shading rate (XeSS mode): cell side length. 2×2 keeps the
/// coherence guarantee tight and the shared-origin shift sub-pixel.
pub const ADAPT_CELL: usize = 2;
/// Geometric coherence vs the representative: relative Euclidean-t gap and
/// interpolated-normal agreement under which a pixel may reuse the rep's
/// visibility record (same material is also required — exact, via tri_mat).
const COH_DT: f32 = 0.02;
const COH_NDOT: f32 = 0.95;
/// Relative luminance spread (max-min over cell mean) above which the cell
/// takes a second full sample per pixel.
const HOT_SPREAD: f32 = 0.35;

/// One adaptive 2×2 cell: per-pixel visibility ALWAYS (four real primary
/// rays with the tile's inherited cut/t_start — tbuf and every G-buffer
/// guide are identical to a non-adaptive frame), adaptive shading effort:
/// - COARSE: coherent pixels reuse the representative's shadow/AO rays
///   (`VisCtl::Apply`), re-applying their own N·L/albedo/specular. Requires
///   uniform captured visibility — fractional means penumbra, where each
///   pixel pays its own rays (the self-declassifier).
/// - HOT: high in-cell luminance spread ⇒ one extra full sample per pixel
///   at its own in-pixel position, averaged locally, no meta/G-buffer writes.
/// Every pixel splats exactly once, at the end.
#[allow(clippy::too_many_arguments)]
fn shade_cell(
    ctx: &FrameCtx,
    cx: usize,
    cy: usize,
    cx1: usize,
    cy1: usize,
    t_start: f32,
    depth: u32,
    kind: u32,
    cut: &[u32],
    ls: &mut LocalStats,
) {
    if (cx1 - cx, cy1 - cy) != (ADAPT_CELL, ADAPT_CELL) {
        // Odd-edge remainder: nothing to share within, plain per-pixel.
        for y in cy..cy1 {
            for x in cx..cx1 {
                shade_pixel(ctx, x, y, t_start, depth, kind, Some(cut), ls);
                ls.adapt_partial_px += 1;
            }
        }
        return;
    }
    let coords = [(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)];
    // Visibility phase first: classification below reads real hits.
    let mut tr: [Traced; 4] =
        coords.map(|(x, y)| trace_primary(ctx, x, y, t_start, Some(cut), ls, SampleId::First));
    // Surface points once per pixel: the coherence test and shade() consume
    // the same (p, n) — recomputing per shaded pixel was a wasted triangle
    // fetch + interpolation in the hottest adaptive loop.
    let sp: [Option<(Vec3A, Vec3A)>; 4] = std::array::from_fn(|i| {
        tr[i].hit.map(|h| shade::surface_point(ctx.scene, &tr[i].ray, &h))
    });
    // Rotate the representative per frame (and stagger across cells) so the
    // shared-ray-origin bias averages out temporally.
    let rep = (ctx.frame as usize + cx / ADAPT_CELL + cy / ADAPT_CELL) & 3;

    let mut vis = shade::VisRecord::default();
    let mut out = [Vec3A::ZERO; 4];
    let (rx, ry) = coords[rep];
    out[rep] = shade_traced(
        ctx,
        rx,
        ry,
        t_start,
        depth,
        kind,
        &mut tr[rep],
        sp[rep],
        ls,
        true,
        false,
        shade::VisCtl::Capture(&mut vis),
        None,
    )
    .c;

    // Rep hit data for the coherence test (borrow tr[rep] read-only now).
    let rep_hit = tr[rep].hit;
    let rep_n = sp[rep].map(|(_, n)| n);
    let mut applied = 0u32;
    for i in 0..4 {
        if i == rep {
            continue;
        }
        let (x, y) = coords[i];
        let coherent = match (&rep_hit, &tr[i].hit) {
            (Some(rh), Some(ih)) => {
                ctx.scene.tri_mat[rh.tri as usize] == ctx.scene.tri_mat[ih.tri as usize]
                    && (ih.t - rh.t).abs() < COH_DT * rh.t.max(1e-6)
                    && sp[i].unwrap().1.dot(rep_n.unwrap()) > COH_NDOT
            }
            _ => false,
        };
        let v = if coherent && vis.uniform {
            applied += 1;
            shade::VisCtl::Apply(&vis)
        } else {
            if coherent {
                ls.adapt_penumbra += 1;
            }
            shade::VisCtl::Off
        };
        out[i] = shade_traced(
            ctx, x, y, t_start, depth, kind, &mut tr[i], sp[i], ls, true, false, v, None,
        )
        .c;
    }

    // --spp: the cell's extra samples. They come BEFORE the HOT test (which is
    // a spread test on the pixel's estimate) and they replace it: multi-
    // sampling already supersamples the footprint everywhere, and folding a
    // top-up into an N-average with the fixed 0.5/0.5 weights below would
    // misweight it. Extras never share visibility (VisCtl::Off) — the record
    // was captured at the rep's sample-0 hit, which an extra ray need not hit.
    let spp = ctx.spp();
    let hot = if spp > 1 {
        for i in 0..4 {
            let (x, y) = coords[i];
            let mut sum = out[i];
            for k in 1..spp {
                let mut t2 =
                    trace_primary(ctx, x, y, t_start, Some(cut), ls, SampleId::Extra(k));
                let s2 = shade_traced(
                    ctx,
                    x,
                    y,
                    t_start,
                    depth,
                    kind,
                    &mut t2,
                    None, // its hit differs from the first sample's
                    ls,
                    k == ctx.primary_sample,
                    false,
                    shade::VisCtl::Off,
                    None,
                );
                sum += s2.c;
            }
            out[i] = sum / spp as f32;
        }
        // The HOT top-up is what --spp REPLACES (multi-sampling already
        // supersamples the footprint everywhere, and folding a top-up into an
        // N-average with the fixed 0.5/0.5 weights below would misweight it).
        // The cell's COARSE/BASE classification is a property of the shared
        // visibility record, not of the sample count, so it still stands below.
        false
    } else {
        // HOT: in-cell luminance spread — shading noise or an in-cell edge;
        // either earns a second full sample per pixel (footprint supersampling).
        let lum = |c: Vec3A| 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
        let l = out.map(lum);
        let (lmin, lmax) =
            l.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let mean = (l[0] + l[1] + l[2] + l[3]) * 0.25;
        lmax - lmin > HOT_SPREAD * (mean + 1e-3)
    };
    if hot {
        ls.adapt_hot += 1;
        for i in 0..4 {
            let (x, y) = coords[i];
            let mut t2 = trace_primary(ctx, x, y, t_start, Some(cut), ls, SampleId::Topup);
            // The top-up traces its own salted ray — its hit differs from the
            // first sample's, so no precomputed surface point applies.
            let s2 = shade_traced(
                ctx,
                x,
                y,
                t_start,
                depth,
                kind,
                &mut t2,
                None,
                ls,
                false,
                false,
                shade::VisCtl::Off,
                None,
            );
            out[i] = (out[i] + s2.c) * 0.5;
            ls.adapt_topup += 1;
        }
    } else if applied == 3 {
        ls.adapt_coarse += 1;
    } else {
        ls.adapt_base += 1;
    }
    for i in 0..4 {
        let (x, y) = coords[i];
        ctx.splat(x, y, out[i]);
    }
}

/// G-buffer write for a primary hit at continuous sample position (fx, fy).
/// The motion vector reprojects the exact hit point through the previous
/// frame's basis: `project` takes a direction relative to that origin, and
/// the result is "current + mv = previous" in pixels, y-down. Jitter never
/// enters any projection (both bases are jitter-free and the hit point lies
/// on the jittered ray by construction), so this MV is unjittered.
fn write_gbuf_hit(
    ctx: &FrameCtx,
    x: usize,
    y: usize,
    fx: f32,
    fy: f32,
    dir: Vec3A,
    t: f32,
    prim: &shade::PrimarySurface,
    c: Vec3A,
) {
    let Some(g) = ctx.gbuf else { return };
    let (_, far) = dlss::near_far(ctx.scene.diag);
    let p = ctx.cam.origin + dir * t;
    let mv = match &ctx.prev_cam {
        Some(prev) => match prev.project(p - prev.origin) {
            Some((px, py)) => (px - fx, py - fy),
            None => (0.0, 0.0), // behind the old image plane: disocclusion
        },
        None => (0.0, 0.0),
    };
    g.write(
        x,
        y,
        &dlss::GPixel {
            normal: prim.n,
            rough: prim.roughness,
            diff_alb: prim.albedo * (1.0 - prim.metallic),
            spec_alb: Vec3A::splat(0.04).lerp(prim.albedo, prim.metallic),
            view_z: t * dir.dot(ctx.cam.forward()),
            mv,
            spec_hit_t: if prim.spec_t.is_infinite() { far } else { prim.spec_t },
        },
    );
    write_fsr(ctx, x, y, dir, t, prim, c);
}

/// The FSR (Ray Regeneration) signal write — split out of the G-buffer writes
/// because its residual is an EXACT REMAINDER against the color the frame
/// PRESENTS: FSR reconstructs color as dd⊗kd + ds⊗f0 + ao·AMBIENT⊗kd + is⊗f0 +
/// residual and never reads `accum` on the CPU-fed path. Under `--spp` the
/// presented color is the N-sample average, which only exists after
/// `shade_pixel`'s loop — hence a separate entry point it can call again with
/// the average (see there). The GPU feed kernel obeys the same rule from the
/// other side: it computes the residual against the averaged `accum`.
fn write_fsr(
    ctx: &FrameCtx,
    x: usize,
    y: usize,
    dir: Vec3A,
    t: f32,
    prim: &shade::PrimarySurface,
    c: Vec3A,
) {
    let Some(f) = ctx.fsr_buf else { return };
    let (_, far) = dlss::near_far(ctx.scene.diag);
    if !t.is_finite() {
        // Sky: EVERY signal zero (AO included — nothing is shaded here, so the
        // composite must add nothing), residual = the color itself (albedos
        // don't matter — 0 * anything), prev_z = far so the depth delta is 0.
        let sig = crate::fsr::Signals {
            dd: Vec3A::ZERO,
            ds: Vec3A::ZERO,
            ao: 0.0,
            is: Vec3A::ZERO,
            residual: c,
        };
        f.write(x, y, &sig, far);
        return;
    }
    // Previous-frame linear view-Z of the SAME hit point — the denoiser
    // MV's B channel is prev_z - cur_z. No previous camera degrades to
    // "no depth motion"; a point behind the old image plane keeps its
    // true (negative) prev view-Z — the large delta marks the
    // disocclusion for history rejection (RG is (0,0) there, above).
    let p = ctx.cam.origin + dir * t;
    let prev_z = match &ctx.prev_cam {
        Some(prev) => (p - prev.origin).dot(prev.forward()),
        None => t * dir.dot(ctx.cam.forward()),
    };
    // The AO signal's remodulation factor: the sky's SH irradiance at the WIRE
    // normal, not at `prim.n` itself. The composite pass rebuilds this from the
    // octahedral normals plane and has no other source for it, so the
    // subtraction here has to be made against the same quantized normal or the
    // composite identity picks up a quantization-sized hole (fsr::wire_normal).
    //
    // `q16v` first: on THIS (CPU-fed) path the plane is oct-encoded from the f16
    // G-buffer normal (ffx_rr::record_upload's ld16), so the f16 hop is part of
    // the wire here. The GPU feed encodes straight from the pack's f32 normal
    // and has no such hop — each path is self-consistent with its own plane,
    // which is all the identity requires.
    let amb = ctx.scene.sky_sh.irradiance(crate::fsr::wire_normal(crate::fsr::q16v(prim.n)));
    let sig = crate::fsr::split_signals(
        c,
        prim.direct_d,
        prim.direct_s,
        prim.ao,
        prim.ind_s,
        prim.albedo * (1.0 - prim.metallic),
        Vec3A::splat(0.04).lerp(prim.albedo, prim.metallic),
        amb,
    );
    f.write(x, y, &sig, prev_z);
}

/// G-buffer write for a sky/miss pixel: depth = far (finite, f16-safe), MV =
/// direction-only reprojection (exact for an environment at infinity — zero
/// under pure translation, the pan vector under rotation).
fn write_gbuf_sky(ctx: &FrameCtx, x: usize, y: usize, fx: f32, fy: f32, dir: Vec3A, c: Vec3A) {
    let Some(g) = ctx.gbuf else { return };
    let (_, far) = dlss::near_far(ctx.scene.diag);
    let mv = match &ctx.prev_cam {
        Some(prev) => match prev.project(dir) {
            Some((px, py)) => (px - fx, py - fy),
            None => (0.0, 0.0),
        },
        None => (0.0, 0.0),
    };
    g.write(
        x,
        y,
        &dlss::GPixel {
            normal: -dir,
            rough: 1.0,
            diff_alb: Vec3A::ONE,
            spec_alb: Vec3A::ZERO,
            view_z: far,
            mv,
            spec_hit_t: 0.0,
        },
    );
    // `c` is this pixel's presented color, which for a sky pixel IS the sky
    // radiance (bit-identically at spp == 1 — the caller passes what
    // `sky::radiance` returned for this same dir). Passing it rather than
    // recomputing is what lets the --spp average reach the residual.
    write_fsr(ctx, x, y, dir, f32::INFINITY, &shade::PrimarySurface::default(), c);
}

/// The traced result of one `shade_pixel` sample — returned so `sparse_fill`
/// can reuse a cell's sample as that cell's fallback fill.
struct Sample {
    c: Vec3A,
    /// Euclidean hit distance (== ray t, unit dir); INFINITY on sky.
    t: f32,
    dir: Vec3A,
    /// Primary-surface capture; default when the ray missed (unused — sky
    /// pixels take `write_gbuf_sky`) or when `ctx.gbuf` is None.
    prim: shade::PrimarySurface,
}

/// A traced-but-not-yet-shaded primary sample: `trace_primary`'s output,
/// `shade_traced`'s input. The adaptive cells trace all four pixels first so
/// coherence classification reads real hits, never guesses.
struct Traced {
    fx: f32,
    fy: f32,
    dir: Vec3A,
    ray: Ray,
    hit: Option<Hit>,
    /// Shading RNG, positioned exactly where the fused path would have it.
    rng: fastrand::Rng,
    /// The spp sample index (First/Topup → 0, Extra(k) → k) — the cloud
    /// march's stratified dither stratum (`clouds::dither_jk`). A Topup
    /// shares sample 0's stratum: top-ups exist only at spp == 1, where
    /// k/spp is 0 anyway, and the frame term still decorrelates them.
    k: u32,
}

/// Mixed into the HOT top-up sample's seed so its in-pixel position and
/// shading stream decorrelate from the cell's first samples.
const TOPUP_SALT: u64 = 0x517C_C1B7_2722_0A95;
/// Mixed (times k) into multi-sample k's seed — a different constant from
/// TOPUP_SALT so a HOT top-up and an `--spp` sample can never share a stream.
const SPP_SALT: u64 = 0xA076_1D64_78BD_642F;

/// Which sample of a pixel a primary ray is. The position rule and the rng
/// salt are one decision, so they live in one type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SampleId {
    /// Sample 0: the frame's REPORTED sample — the position rule and rng seed
    /// a single-sample frame has always used. spp == 1 is exactly this.
    First,
    /// Multi-sample k >= 1 (`--spp`): a deterministic Halton offset inside the
    /// same pixel, its own rng stream. Color only — never a side channel
    /// (unless `--check` probes it), so the guides stay tied to sample 0's
    /// reported jitter.
    Extra(u32),
    /// The adaptive HOT footprint supersample: its own random in-pixel
    /// position (spp == 1 frames only — see `shade_cell`).
    Topup,
}

impl SampleId {
    #[inline(always)]
    fn salt(self) -> u64 {
        match self {
            SampleId::First => 0,
            SampleId::Extra(k) => SPP_SALT.wrapping_mul(k as u64),
            SampleId::Topup => TOPUP_SALT,
        }
    }
}

fn trace_primary(
    ctx: &FrameCtx,
    x: usize,
    y: usize,
    t_start: f32,
    cut: Option<&[u32]>,
    ls: &mut LocalStats,
    sample: SampleId,
) -> Traced {
    let seed = (x as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F))
        .wrapping_add(ctx.frame as u64)
        .wrapping_add(sample.salt());
    let mut rng = fastrand::Rng::with_seed(seed);
    // DLSS mode: one frame-uniform low-discrepancy offset for every pixel
    // (the denoiser is told this exact offset). Legacy: per-pixel rng.
    // Either way the sample stays in [x, x+1) — the invariant leaf rays need.
    // An --spp sample takes the deterministic Halton offset for (frame, k) —
    // a pure function, which is what lets verify rebuild its ray; a HOT top-up
    // takes its own random in-pixel position. Both are footprint supersamples:
    // the reported frame jitter stays tied to sample 0, the only one that
    // writes meta/G-buffers.
    let (jx, jy) = match sample {
        SampleId::Extra(k) => {
            let (ox, oy) = dlss::jitter_for_sample(ctx.frame, k);
            (0.5 + ox, 0.5 + oy)
        }
        SampleId::Topup => (rng.f32(), rng.f32()),
        SampleId::First => match ctx.frame_jitter {
            Some((ox, oy)) => (0.5 + ox, 0.5 + oy),
            None => {
                if ctx.jitter {
                    (rng.f32(), rng.f32())
                } else {
                    (0.5, 0.5)
                }
            }
        },
    };
    let (fx, fy) = (x as f32 + jx, y as f32 + jy);
    let dir = ctx.cam.ray_dir(fx, fy);
    let ray = Ray::new(ctx.cam.origin, dir);
    ls.primary_rays += 1;
    // Counted into a local and added to BOTH: `ray_nodes` stays the total (the
    // builder bake-off and the adopt bench read it that way), `ray_nodes_prim`
    // is the camera-ray share. This is the only site that traces primaries.
    let mut pn = 0u64;
    let hit = match cut {
        // Quadtree leaf: seed the traversal from the tile's inherited cut.
        Some(roots) => ctx.bvh.intersect_multi(ctx.scene, &ray, t_start, f32::INFINITY, roots, &mut pn),
        // Plain reference path: full traversal from the root.
        None => ctx.bvh.intersect(ctx.scene, &ray, t_start, f32::INFINITY, &mut pn),
    };
    ls.ray_nodes += pn;
    ls.ray_nodes_prim += pn;
    let k = match sample {
        SampleId::Extra(k) => k,
        SampleId::First | SampleId::Topup => 0,
    };
    Traced { fx, fy, dir, ray, hit, rng, k }
}

/// Shade a traced sample. `primary` gates every per-pixel side channel
/// (tbuf/info meta, G-buffers, skip-ratio stats) — true for the pixel's
/// first sample, false for HOT top-ups so the guides stay tied to the
/// reported frame jitter. `do_splat` false lets the adaptive cell average
/// locally and splat once (two splats would break both accum semantics).
#[allow(clippy::too_many_arguments)]
fn shade_traced(
    ctx: &FrameCtx,
    x: usize,
    y: usize,
    t_start: f32,
    depth: u32,
    kind: u32,
    tr: &mut Traced,
    sp: Option<(Vec3A, Vec3A)>,
    ls: &mut LocalStats,
    primary: bool,
    do_splat: bool,
    vis: shade::VisCtl,
    hemi_share: Option<&crate::hemi::HemiShare>,
) -> Sample {
    match tr.hit {
        Some(hit) => {
            if primary {
                ctx.store_meta(x, y, hit.t, depth, kind);
                if t_start > 0.0 {
                    ls.skip_ratio_micro += (t_start / hit.t * 1e6) as u64;
                }
                ls.skip_ratio_count += 1;
            }
            let mut prim = shade::PrimarySurface::default();
            let mut c = shade::shade(
                ctx.scene,
                ctx.bvh,
                &tr.ray,
                &hit,
                sp,
                &ctx.q,
                &mut tr.rng,
                ctx.sun,
                &ctx.clouds,
                // Primary ray cone: apex at the camera, one-pixel spread.
                shade::Cone::primary(&ctx.cam),
                0,
                ls,
                if primary && ctx.gbuf.is_some() { Some(&mut prim) } else { None },
                vis,
                hemi_share,
                // The ONE Some site: fireflies light the primary camera path
                // only. Day sessions (count 0) hand shade a structural None.
                (ctx.fireflies.count > 0).then_some(&ctx.fireflies),
            );
            // Firefly glow, depth-tested against the primary hit — a firefly
            // between the camera and the surface splats over the shaded
            // color (this sample's own ray, so --spp averages it). Guarded:
            // `-0.0 + 0.0 = +0.0` would break day bit-identity (the emissive
            // discipline). Color-only — tbuf/info/meta and every exact-zero
            // verify counter are structurally blind to it; under FSR it
            // lands in the deterministic residual (the emissive accept).
            if ctx.fireflies.count > 0 {
                c += crate::fireflies::glow(
                    &ctx.fireflies,
                    tr.ray.o,
                    tr.dir,
                    hit.t,
                    ctx.cam.pixel_cone() * 0.5,
                );
            }
            if do_splat {
                ctx.splat(x, y, c);
            }
            if primary {
                write_gbuf_hit(ctx, x, y, tr.fx, tr.fy, tr.dir, hit.t, &prim, c);
            }
            Sample { c, t: hit.t, dir: tr.dir, prim }
        }
        None => {
            if primary {
                ctx.store_meta(x, y, f32::INFINITY, depth, kind);
            }
            // A DISPLAY path: the camera is looking at the sky, so it sees the
            // sun disc. Nothing else delivers it here — this is the backdrop
            // (through the cloud layer, which marches from the ray's origin
            // with this sample's own dither phase — per (pixel, frame,
            // sample), so --spp and accumulation genuinely average the march).
            let mut c = crate::sky::radiance(
                tr.ray.o,
                tr.dir,
                &ctx.scene.sun,
                ctx.cam.pixel_cone() * 0.5,
                ctx.scene.sky_scale,
                ctx.scene.night,
                ctx.frame,
                &ctx.clouds,
                crate::clouds::dither_jk(x as u32, y as u32, ctx.frame, tr.k, ctx.spp()),
            );
            // Firefly glow against the open sky — a miss has no depth to
            // test (t_max ∞). Guarded like the hit arm.
            if ctx.fireflies.count > 0 {
                c += crate::fireflies::glow(
                    &ctx.fireflies,
                    tr.ray.o,
                    tr.dir,
                    f32::INFINITY,
                    ctx.cam.pixel_cone() * 0.5,
                );
            }
            if do_splat {
                ctx.splat(x, y, c);
            }
            if primary {
                write_gbuf_sky(ctx, x, y, tr.fx, tr.fy, tr.dir, c);
            }
            Sample { c, t: f32::INFINITY, dir: tr.dir, prim: shade::PrimarySurface::default() }
        }
    }
}

/// One pixel, `ctx.spp` samples, ONE splat.
///
/// Every extra sample lands inside the same pixel, hence inside every ancestor
/// tile frustum, so it consumes the SAME inherited `t_start` and node cut as
/// sample 0 — the leaf-tile argument, unchanged. That is the whole point:
/// the quadtree's per-tile frustum work is paid once and amortizes over
/// 64·spp rays instead of 64.
///
/// The returned Sample is sample `primary_sample`'s (its t/dir/PrimarySurface
/// are what `sparse_fill` floods and the G-buffers hold), with the N-sample
/// AVERAGE as its color — so a flooded cell broadcasts the averaged radiance.
fn shade_pixel(
    ctx: &FrameCtx,
    x: usize,
    y: usize,
    t_start: f32,
    depth: u32,
    kind: u32,
    cut: Option<&[u32]>,
    ls: &mut LocalStats,
) -> Sample {
    let spp = ctx.spp();
    let mut sum = Vec3A::ZERO;
    let mut out = None;
    for k in 0..spp {
        let id = if k == 0 { SampleId::First } else { SampleId::Extra(k) };
        let mut tr = trace_primary(ctx, x, y, t_start, cut, ls, id);
        let s = shade_traced(
            ctx,
            x,
            y,
            t_start,
            depth,
            kind,
            &mut tr,
            None,
            ls,
            k == ctx.primary_sample, // side channels: tbuf/info/G-buffers/MV
            false,                   // average locally, splat once below
            shade::VisCtl::Off,
            None,
        );
        sum += s.c;
        if k == ctx.primary_sample {
            out = Some(s);
        }
    }
    let mut s = out.expect("primary_sample must be < spp");
    s.c = sum / spp as f32;
    // FSR Ray Regeneration reconstructs the presented color from its OWN
    // planes (dd⊗kd + ds⊗f0 + ao·AMBIENT⊗kd + is⊗f0 + residual — `accum` is
    // never uploaded on the CPU-fed path), so the residual `shade_traced` wrote
    // for the probe sample is a remainder against THAT sample's color, not the
    // average this frame presents. Rewrite it against the average, or FSR would
    // put sample 0's image back on screen and --spp would be a costly no-op
    // there. All four denoised signals (dd/ds/ao/is) stay the probe sample's —
    // the residual is defined as the remainder, so the identity closes exactly
    // whatever they are. The GPU pack captures them the same way, and its feed
    // kernel likewise takes the residual against averaged accum, so both feeds
    // mean exactly the same thing.
    if spp > 1 {
        write_fsr(ctx, x, y, s.dir, s.t, &s.prim, s.c);
    }
    ctx.splat(x, y, s.c);
    s
}

/// The depth cap never flat-fills tiles shallower than this (a bad estimate
/// must not paint the whole screen as a handful of quads).
pub const MIN_BUDGET_DEPTH: u32 = 2;

/// Depth-capped dynamic resolution: the same depth-first recursion as
/// `render_frame`, but every screen area stops at a uniform `max_depth` and
/// unresolved tiles there are sparse-filled. The caller (the
/// frame-budget controller in main.rs) estimates `max_depth` from the previous
/// frame's time — no wall clock is read here, so the frame is deterministic
/// for a given cap. A cap at or beyond the leaf depth is bit-identical to
/// `render_frame(ctx, true)` (same code path, the cap never fires).
pub fn render_frame_capped(ctx: &FrameCtx, max_depth: u32) {
    crate::zone!("trace-capped");
    let root = crate::ftree::Accel::for_tiles(ctx.bvh).root_cut();
    flush_pend(
        ctx,
        trace_tile(ctx, 0, 0, ctx.rw, ctx.rh, 0.0, 0, 0, root, max_depth.max(MIN_BUDGET_DEPTH)),
    );
}

/// Side length of the sparse-fill sample grid: each capped quad shoots one
/// real point sample per SAMPLE_CELL×SAMPLE_CELL cell (~0.4% of full-res
/// rays at typical caps). Denser de-blocks faster but costs budget-frame
/// time — the depth controller absorbs it by lowering the cap.
pub const SAMPLE_CELL: usize = 16;

/// Depth cap reached: fill the quad from real point samples instead of one
/// flat quad ("a pixel is not a little square"). Each SAMPLE_CELL cell traces
/// one full `shade_pixel` sample at a per-frame random pixel — stored as
/// KIND_LEAF with exact t and G-buffer, sound because every pixel of a capped
/// tile lies inside the tile frustum, so the inherited cut and `t_start`
/// apply (the leaf-tile argument). The rest of the cell floods with that
/// sample's result as the KIND_COARSE fallback, consumed only where the
/// reprojection history has nothing better (reproject.rs keeps history over
/// coarse pixels at blend weight 0).
fn sparse_fill(
    ctx: &FrameCtx,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    t_start: f32,
    depth: u32,
    cut: &[u32],
    ls: &mut LocalStats,
) {
    // Slot-ref cut -> binary ray roots, once per capped tile (the shade_tile
    // convention — sparse samples are ordinary cut-seeded primary rays).
    let mut rbuf = [0u32; MAX_CUT];
    let cut = crate::ftree::Accel::for_tiles(ctx.bvh).ray_roots(cut, &mut rbuf);
    // Sample-position RNG: per-quad, per-frame (the old flat-fill seed recipe).
    let seed = (x0 as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((y0 as u64).wrapping_mul(0xC2B2AE3D27D4EB4F))
        .wrapping_add(ctx.frame as u64);
    let mut rng = fastrand::Rng::with_seed(seed);
    ls.coarse_tiles += 1;
    let mut cy = y0;
    while cy < y1 {
        let cy1 = (cy + SAMPLE_CELL).min(y1);
        let mut cx = x0;
        while cx < x1 {
            let cx1 = (cx + SAMPLE_CELL).min(x1);
            let sx = cx + rng.usize(..cx1 - cx);
            let sy = cy + rng.usize(..cy1 - cy);
            let s = shade_pixel(ctx, sx, sy, t_start, depth, KIND_LEAF, Some(cut), ls);
            ls.coarse_samples += 1;
            ls.coarse_pixels += ((cx1 - cx) * (cy1 - cy) - 1) as u64;
            for y in cy..cy1 {
                for x in cx..cx1 {
                    if x == sx && y == sy {
                        continue;
                    }
                    ctx.store_meta(x, y, s.t, depth, KIND_COARSE);
                    ctx.splat(x, y, s.c);
                    // Shared in-cell sample surface, per-pixel sample position
                    // — the write helpers subtract (fx, fy), so the camera-
                    // motion part of the MV stays per-pixel-correct even in
                    // coarse cells.
                    let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                    if s.t.is_finite() {
                        write_gbuf_hit(ctx, x, y, fx, fy, s.dir, s.t, &s.prim, s.c);
                    } else {
                        write_gbuf_sky(ctx, x, y, fx, fy, s.dir, s.c);
                    }
                }
            }
            cx = cx1;
        }
        cy = cy1;
    }
}

/// The frustum proved this whole tile empty — fill with sky, zero rays traced.
fn fill_sky(ctx: &FrameCtx, x0: usize, y0: usize, x1: usize, y1: usize, depth: u32) {
    let mut ls = LocalStats::default();
    ls.sky_tiles = 1;
    ls.sky_pixels = ((x1 - x0) * (y1 - y0)) as u64;
    fill_sky_rows(ctx, x0, y0, x1, y1, depth);
    ctx.stats.add(&ls);
}

/// The per-pixel body of `fill_sky`, stats-free so the replay driver can fan
/// a large sky rect out across row bands without double-counting the tile.
fn fill_sky_rows(ctx: &FrameCtx, x0: usize, y0: usize, x1: usize, y1: usize, depth: u32) {
    let spp = ctx.spp();
    for y in y0..y1 {
        for x in x0..x1 {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let dir = ctx.cam.ray_dir(fx, fy);
            // The pixel's cloud march phase — per (pixel, frame, SAMPLE),
            // mirroring the per-pixel sample loops: the center rides sample
            // 0's stratum (bitwise `dither_j(x, y, frame)`), each SKY_J extra
            // takes its own stratum, so a multi-sampled sky tile averages the
            // march phase exactly like a leaf pixel does.
            let j = crate::clouds::dither_jk(x as u32, y as u32, ctx.frame, 0, spp);
            // A DISPLAY path — the camera looking at the sky — so it sees the
            // sun DISC, antialiased against the pixel's own cone (sky.rs),
            // plus the stars (twinkle-phased by the frame index — replay
            // re-shades at the same frame, so bit-identity holds).
            let mut c = crate::sky::radiance(
                ctx.cam.origin,
                dir,
                &ctx.scene.sun,
                ctx.cam.pixel_cone() * 0.5,
                ctx.scene.sky_scale,
                ctx.scene.night,
                ctx.frame,
                &ctx.clouds,
                j,
            );
            // Firefly glow against a proven-empty tile — the feature's most
            // visible case (a firefly on the night sky), still ZERO rays:
            // nothing occludes along a sky direction, so t_max is ∞ and the
            // splat needs no depth test. Guarded (the emissive discipline);
            // the splat is Gaussian at ≥ the pixel footprint, so the
            // center-direction rule stays sound exactly as it does for stars.
            if ctx.fireflies.count > 0 {
                c += crate::fireflies::glow(
                    &ctx.fireflies,
                    ctx.cam.origin,
                    dir,
                    f32::INFINITY,
                    ctx.cam.pixel_cone() * 0.5,
                );
            }
            // A proven-empty tile has always shaded ONE center direction, spp
            // or not — sound while the sky was smooth at sub-pixel scale. The
            // cloud layer broke that premise (a cover-ramp edge crosses a
            // pixel), so under clouds a multi-sampled frame averages spp
            // sample positions — still ZERO rays, sample 0 still the center,
            // side channels still the center's. The extra DIRECTION offsets
            // are the PHASE-0 Halton set, deliberately frame-INDEPENDENT like
            // the center itself (the sky fill antialiases a static function;
            // per-frame offsets put inter-frame dither on cloud edges, which
            // the spp stability gate rightly rejects at night, where nothing
            // louder masks it — a lesson about the direction SET only: the
            // march PHASE below is per-sample and frame-keyed, symmetric
            // across spp levels, which that gate accepts). The guard keeps
            // spp == 1 and --no-clouds on the old path VERBATIM, and the GPU
            // twin (cs_sky + the injected SKY_J table) mirrors the loop term
            // for term — the spp wavefront-vs-reference image A/B at frame 0
            // is the gate.
            if ctx.clouds.enabled && spp > 1 {
                for k in 1..spp {
                    let (ox, oy) = dlss::jitter_for_sample(0, k);
                    let dk = ctx.cam.ray_dir(fx + ox, fy + oy);
                    c += crate::sky::radiance(
                        ctx.cam.origin,
                        dk,
                        &ctx.scene.sun,
                        ctx.cam.pixel_cone() * 0.5,
                        ctx.scene.sky_scale,
                        ctx.scene.night,
                        ctx.frame,
                        &ctx.clouds,
                        crate::clouds::dither_jk(x as u32, y as u32, ctx.frame, k, spp),
                    );
                    // Each extra sample carries its own glow along its own
                    // direction, exactly like a leaf pixel's sample loop.
                    if ctx.fireflies.count > 0 {
                        c += crate::fireflies::glow(
                            &ctx.fireflies,
                            ctx.cam.origin,
                            dk,
                            f32::INFINITY,
                            ctx.cam.pixel_cone() * 0.5,
                        );
                    }
                }
                c *= 1.0 / spp as f32;
            }
            ctx.store_meta(x, y, f32::INFINITY, depth, KIND_SKY);
            ctx.splat(x, y, c);
            // The presented color and the FSR residual are the same value.
            write_gbuf_sky(ctx, x, y, fx, fy, dir, c);
        }
    }
}

/// Rows per parallel band when replaying a large sky rect (a depth-1 sky node
/// can span a quarter screen — one sequential fill would serialize the tail).
const REPLAY_SKY_BAND: usize = 16;

/// Re-shade a recorded terminal structure (see replay.rs). Sound ONLY when
/// `ctx.cam` is bit-equal to the recording frame's basis at the same
/// (rw, rh): the leaf cut / t_start inheritance arguments are per-frustum,
/// and bit-equality is what makes the frusta identical — the caller enforces
/// it (main.rs `replay_key`). Everything shading consumes per frame (quality,
/// frame index, jitter, G-buffers, prev_cam) comes from the fresh `ctx`, so a
/// replayed frame advances RNG/jitter exactly like a traced one; `--check`
/// gates same-seed bit-identity of tbuf/info/accum against a fresh trace.
pub fn render_frame_replay(ctx: &FrameCtx, rc: &replay::ReplayCache) {
    crate::zone!("replay");
    debug_assert!(rc.valid(), "replaying a poisoned recording");
    let (nl, ns) = rc.counts();
    (0..nl as usize).into_par_iter().for_each(|i| {
        let r = rc.leaf(i);
        let mut cut = [0u32; MAX_CUT];
        let len = rc.copy_cut(r.cut_off, r.cut_len, &mut cut);
        shade_tile(ctx, r.x0, r.y0, r.x1, r.y1, r.t_start, r.depth, KIND_LEAF, &cut[..len]);
    });
    (0..ns as usize).into_par_iter().for_each(|i| {
        let r = rc.sky(i);
        let mut ls = LocalStats::default();
        ls.sky_tiles = 1;
        ls.sky_pixels = ((r.x1 - r.x0) * (r.y1 - r.y0)) as u64;
        ctx.stats.add(&ls);
        let rows = r.y1 - r.y0;
        if rows > REPLAY_SKY_BAND {
            let bands = rows.div_ceil(REPLAY_SKY_BAND);
            (0..bands).into_par_iter().for_each(|b| {
                let by0 = r.y0 + b * REPLAY_SKY_BAND;
                let by1 = (by0 + REPLAY_SKY_BAND).min(r.y1);
                fill_sky_rows(ctx, r.x0, by0, r.x1, by1, r.depth);
            });
        } else {
            fill_sky_rows(ctx, r.x0, r.y0, r.x1, r.y1, r.depth);
        }
    });
    let mut ls = LocalStats::default();
    ls.replay_leaf_tiles = nl as u64;
    ls.replay_sky_tiles = ns as u64;
    ctx.stats.add(&ls);
}

/// Average, tonemap, and upscale the accumulation buffer into the 0RGB present
/// buffer; optionally blend the quadtree debug overlay.
pub fn resolve(
    accum: &[AtomicU32],
    info: &[AtomicU32],
    samples: u32,
    overlay_on: bool,
    present: &mut [u32],
    rw: usize,
    rh: usize,
    ww: usize,
    wh: usize,
) {
    crate::zone!("resolve");
    let inv = 1.0 / samples.max(1) as f32;

    // Glare needs the whole HDR image (it is a convolution), so the bloom path
    // materializes it — into bloom's own cached buffer, so a presented frame
    // still allocates nothing. The `--no-bloom` path keeps the original
    // per-pixel loop verbatim, which is what makes that flag bit-identical to
    // the pre-bloom renderer by construction.
    if crate::bloom::enabled() {
        crate::bloom::with_glare_filled(
            rw,
            rh,
            |hdr| {
                hdr.par_chunks_mut(3).enumerate().for_each(|(p, px)| {
                    for (k, o) in px.iter_mut().enumerate() {
                        *o = f32::from_bits(accum[p * 3 + k].load(Relaxed)) * inv;
                    }
                });
            },
            |hdr| tonemap_to(hdr, info, overlay_on, present, rw, rh, ww, wh),
        );
        return;
    }

    present.par_chunks_mut(ww).enumerate().for_each(|(wy, row)| {
        let sy = (wy * rh / wh).min(rh - 1);
        for (wx, out) in row.iter_mut().enumerate() {
            let sx = (wx * rw / ww).min(rw - 1);
            let i = (sy * rw + sx) * 3;
            let c = Vec3A::new(
                f32::from_bits(accum[i].load(Relaxed)),
                f32::from_bits(accum[i + 1].load(Relaxed)),
                f32::from_bits(accum[i + 2].load(Relaxed)),
            ) * inv;
            *out = present_px(c, info, overlay_on, sx, sy, rw, rh);
        }
    });
}

/// `resolve` for an already-averaged linear HDR buffer (3 floats/px) — the
/// OIDN output path. Same tonemap curve, overlay composite, and upscale.
pub fn resolve_hdr(
    hdr: &[f32],
    info: &[AtomicU32],
    overlay_on: bool,
    present: &mut [u32],
    rw: usize,
    rh: usize,
    ww: usize,
    wh: usize,
) {
    crate::zone!("resolve-hdr");
    // The CPU denoisers' tonemap entry (OIDN / NPPD land here directly), and
    // where their glare comes from. `resolve` does NOT funnel through this — it
    // materializes its own HDR image and calls `tonemap_to` below with the glare
    // already applied, so nothing can double-bloom.
    crate::bloom::with_glare(hdr, rw, rh, |hdr| {
        tonemap_to(hdr, info, overlay_on, present, rw, rh, ww, wh)
    });
}

/// Tonemap + overlay + upscale an HDR image into the present buffer. The one
/// CPU present loop, shared by `resolve` and `resolve_hdr`; glare is the
/// caller's business, so this can never apply it twice.
#[allow(clippy::too_many_arguments)]
fn tonemap_to(
    hdr: &[f32],
    info: &[AtomicU32],
    overlay_on: bool,
    present: &mut [u32],
    rw: usize,
    rh: usize,
    ww: usize,
    wh: usize,
) {
    present.par_chunks_mut(ww).enumerate().for_each(|(wy, row)| {
        let sy = (wy * rh / wh).min(rh - 1);
        for (wx, out) in row.iter_mut().enumerate() {
            let sx = (wx * rw / ww).min(rw - 1);
            let i = (sy * rw + sx) * 3;
            let c = Vec3A::new(hdr[i], hdr[i + 1], hdr[i + 2]);
            *out = present_px(c, info, overlay_on, sx, sy, rw, rh);
        }
    });
}

/// `resolve` for the scRGB f16 swapchain — same average, same overlay, same
/// nearest upscale; only the encode differs.
pub fn resolve_scrgb(
    accum: &[AtomicU32],
    info: &[AtomicU32],
    samples: u32,
    overlay_on: bool,
    p: tone::ToneParams,
    present: &mut [[f16; 4]],
    rw: usize,
    rh: usize,
    ww: usize,
    wh: usize,
) {
    crate::zone!("resolve-scrgb");
    let inv = 1.0 / samples.max(1) as f32;
    // Glare is a DISPLAY-stage pass on whatever the tonemap is about to read, so
    // it applies to the scRGB encode exactly as it does to the SDR one — the
    // swapchain format is not a reason to have or not have bloom. (It matters
    // more here, if anything: scRGB is the default, so a miss would silently
    // delete glare from every CPU-presented frame.) Same structure as `resolve`.
    if crate::bloom::enabled() {
        crate::bloom::with_glare_filled(
            rw,
            rh,
            |hdr| {
                hdr.par_chunks_mut(3).enumerate().for_each(|(px, o)| {
                    for (k, v) in o.iter_mut().enumerate() {
                        *v = f32::from_bits(accum[px * 3 + k].load(Relaxed)) * inv;
                    }
                });
            },
            |hdr| tonemap_to_scrgb(hdr, info, overlay_on, p, present, rw, rh, ww, wh),
        );
        return;
    }

    present.par_chunks_mut(ww).enumerate().for_each(|(wy, row)| {
        let sy = (wy * rh / wh).min(rh - 1);
        for (wx, out) in row.iter_mut().enumerate() {
            let sx = (wx * rw / ww).min(rw - 1);
            let i = (sy * rw + sx) * 3;
            let c = Vec3A::new(
                f32::from_bits(accum[i].load(Relaxed)),
                f32::from_bits(accum[i + 1].load(Relaxed)),
                f32::from_bits(accum[i + 2].load(Relaxed)),
            ) * inv;
            *out = present_px_scrgb(c, p, info, overlay_on, sx, sy, rw, rh);
        }
    });
}

/// `resolve_hdr` for the scRGB f16 swapchain — the OIDN / NPPD / XeSS-post
/// output path, which hands over an already-averaged linear HDR buffer.
pub fn resolve_hdr_scrgb(
    hdr: &[f32],
    info: &[AtomicU32],
    overlay_on: bool,
    p: tone::ToneParams,
    present: &mut [[f16; 4]],
    rw: usize,
    rh: usize,
    ww: usize,
    wh: usize,
) {
    crate::zone!("resolve-hdr-scrgb");
    // The scRGB twin of `resolve_hdr`: the CPU denoisers' glare comes from here.
    crate::bloom::with_glare(hdr, rw, rh, |hdr| {
        tonemap_to_scrgb(hdr, info, overlay_on, p, present, rw, rh, ww, wh)
    });
}

/// `tonemap_to` for the scRGB f16 swapchain — the one scRGB present loop,
/// shared by `resolve_scrgb` and `resolve_hdr_scrgb`. Glare is the caller's
/// business, so this can never apply it twice.
#[allow(clippy::too_many_arguments)]
fn tonemap_to_scrgb(
    hdr: &[f32],
    info: &[AtomicU32],
    overlay_on: bool,
    p: tone::ToneParams,
    present: &mut [[f16; 4]],
    rw: usize,
    rh: usize,
    ww: usize,
    wh: usize,
) {
    present.par_chunks_mut(ww).enumerate().for_each(|(wy, row)| {
        let sy = (wy * rh / wh).min(rh - 1);
        for (wx, out) in row.iter_mut().enumerate() {
            let sx = (wx * rw / ww).min(rw - 1);
            let i = (sy * rw + sx) * 3;
            let c = Vec3A::new(hdr[i], hdr[i + 1], hdr[i + 2]);
            *out = present_px_scrgb(c, p, info, overlay_on, sx, sy, rw, rh);
        }
    });
}

/// `resolve` for the HDR10 (R10G10B10A2 + PQ) swapchain — same average, same
/// overlay, same nearest upscale; the encode is `tone::ToneMode::Pq` and the
/// pack is the 10-bit lane layout (R low).
#[allow(clippy::too_many_arguments)]
pub fn resolve_pq(
    accum: &[AtomicU32],
    info: &[AtomicU32],
    samples: u32,
    overlay_on: bool,
    p: tone::ToneParams,
    present: &mut [u32],
    rw: usize,
    rh: usize,
    ww: usize,
    wh: usize,
) {
    crate::zone!("resolve-pq");
    let inv = 1.0 / samples.max(1) as f32;
    // Glare before the curve, exactly as in `resolve_scrgb` — the swapchain
    // encode is never a reason to have or not have bloom.
    if crate::bloom::enabled() {
        crate::bloom::with_glare_filled(
            rw,
            rh,
            |hdr| {
                hdr.par_chunks_mut(3).enumerate().for_each(|(px, o)| {
                    for (k, v) in o.iter_mut().enumerate() {
                        *v = f32::from_bits(accum[px * 3 + k].load(Relaxed)) * inv;
                    }
                });
            },
            |hdr| tonemap_to_pq(hdr, info, overlay_on, p, present, rw, rh, ww, wh),
        );
        return;
    }

    present.par_chunks_mut(ww).enumerate().for_each(|(wy, row)| {
        let sy = (wy * rh / wh).min(rh - 1);
        for (wx, out) in row.iter_mut().enumerate() {
            let sx = (wx * rw / ww).min(rw - 1);
            let i = (sy * rw + sx) * 3;
            let c = Vec3A::new(
                f32::from_bits(accum[i].load(Relaxed)),
                f32::from_bits(accum[i + 1].load(Relaxed)),
                f32::from_bits(accum[i + 2].load(Relaxed)),
            ) * inv;
            *out = present_px_pq(c, p, info, overlay_on, sx, sy, rw, rh);
        }
    });
}

/// `resolve_hdr` for the HDR10 swapchain — the OIDN / NPPD / XeSS-post output
/// path under PQ.
#[allow(clippy::too_many_arguments)]
pub fn resolve_hdr_pq(
    hdr: &[f32],
    info: &[AtomicU32],
    overlay_on: bool,
    p: tone::ToneParams,
    present: &mut [u32],
    rw: usize,
    rh: usize,
    ww: usize,
    wh: usize,
) {
    crate::zone!("resolve-hdr-pq");
    crate::bloom::with_glare(hdr, rw, rh, |hdr| {
        tonemap_to_pq(hdr, info, overlay_on, p, present, rw, rh, ww, wh)
    });
}

/// `tonemap_to` for the HDR10 swapchain — the one PQ present loop, shared by
/// `resolve_pq` and `resolve_hdr_pq`. Glare is the caller's business.
#[allow(clippy::too_many_arguments)]
fn tonemap_to_pq(
    hdr: &[f32],
    info: &[AtomicU32],
    overlay_on: bool,
    p: tone::ToneParams,
    present: &mut [u32],
    rw: usize,
    rh: usize,
    ww: usize,
    wh: usize,
) {
    present.par_chunks_mut(ww).enumerate().for_each(|(wy, row)| {
        let sy = (wy * rh / wh).min(rh - 1);
        for (wx, out) in row.iter_mut().enumerate() {
            let sx = (wx * rw / ww).min(rw - 1);
            let i = (sy * rw + sx) * 3;
            let c = Vec3A::new(hdr[i], hdr[i + 1], hdr[i + 2]);
            *out = present_px_pq(c, p, info, overlay_on, sx, sy, rw, rh);
        }
    });
}

/// Blend the quadtree debug overlay: tint by kind, darken tile borders.
///
/// The tints are **display-space [0,1] colours** — authored to look right
/// against a gamma-encoded image — so both callers composite them in that space
/// and nowhere else. The scRGB path pays a gamma round-trip to do so, which is
/// the whole reason this is factored out: compositing in linear instead would
/// tint highlights in proportion to their magnitude rather than uniformly, and
/// the overlay would not match the SDR build.
#[inline]
fn overlay_px(
    mut c: Vec3A,
    info: &[AtomicU32],
    sx: usize,
    sy: usize,
    rw: usize,
    rh: usize,
) -> Vec3A {
    let pi = info[sy * rw + sx].load(Relaxed);
    let (tint, alpha) = overlay::tint(pi);
    c = c.lerp(tint, alpha);
    let right = if sx + 1 < rw { info[sy * rw + sx + 1].load(Relaxed) } else { pi };
    let down = if sy + 1 < rh { info[(sy + 1) * rw + sx].load(Relaxed) } else { pi };
    if right != pi || down != pi {
        c *= 0.25; // tile border
    }
    c
}

/// Tonemap and pack to 0x00RRGGBB for the 8-bit SDR swapchain (the fallback
/// path when the scRGB colour space is refused). `ToneParams::SDR` is the
/// pre-HDR curve exactly — gated bit-for-bit by `tone::self_test` — and
/// `shape` has already applied the gamma, so the overlay lands in display
/// space with no extra work.
#[inline]
fn present_px(
    c: Vec3A,
    info: &[AtomicU32],
    overlay_on: bool,
    sx: usize,
    sy: usize,
    rw: usize,
    rh: usize,
) -> u32 {
    let mut c = tone::shape(c, tone::ToneParams::SDR);
    if overlay_on {
        c = overlay_px(c, info, sx, sy, rw, rh);
    }
    let c = (c.clamp(Vec3A::ZERO, Vec3A::ONE) * 255.0 + 0.5).floor();
    ((c.x as u32) << 16) | ((c.y as u32) << 8) | c.z as u32
}

/// Tonemap and encode to scRGB f16 for the `R16G16B16A16_FLOAT` swapchain.
///
/// scRGB is linear, so `shape` applies no gamma — which means the overlay must
/// be taken INTO display space and back out again to composite where its tints
/// were authored. The round-trip runs only when the overlay is on, so the
/// normal path costs nothing and is not perturbed by a pow/pow⁻¹ pair.
///
/// Deliberately **not** clamped above: values over 1.0 are legal scRGB and are
/// precisely the highlight headroom this path exists to carry. The lower clamp
/// stays — negative scRGB is out of gamut and we never intend to emit it.
#[inline]
fn present_px_scrgb(
    c: Vec3A,
    p: tone::ToneParams,
    info: &[AtomicU32],
    overlay_on: bool,
    sx: usize,
    sy: usize,
    rw: usize,
    rh: usize,
) -> [f16; 4] {
    let mut v = tone::shape(c, p);
    if overlay_on {
        let g = overlay_px(v.max(Vec3A::ZERO).powf(1.0 / 2.2), info, sx, sy, rw, rh);
        v = g.max(Vec3A::ZERO).powf(2.2);
    }
    let v = tone::encode(v, p).max(Vec3A::ZERO);
    [f16::from_f32(v.x), f16::from_f32(v.y), f16::from_f32(v.z), f16::from_f32(1.0)]
}

/// Tonemap and encode to packed 10-bit PQ for the `R10G10B10A2_UNORM`
/// swapchain (`r | g<<10 | b<<20 | 3<<30` — R in the LOW bits, R10G10B10A2's
/// lane order, opposite the SDR pack's BGRA8). The overlay composite reuses
/// the scRGB path's display-space round-trip: `shape` under PQ is still
/// paper-white-relative *light* (the ST 2084 encode lives in `tone::encode`),
/// so the same pow pair lands the tints where they were authored.
#[inline]
fn present_px_pq(
    c: Vec3A,
    p: tone::ToneParams,
    info: &[AtomicU32],
    overlay_on: bool,
    sx: usize,
    sy: usize,
    rw: usize,
    rh: usize,
) -> u32 {
    let mut v = tone::shape(c, p);
    if overlay_on {
        let g = overlay_px(v.max(Vec3A::ZERO).powf(1.0 / 2.2), info, sx, sy, rw, rh);
        v = g.max(Vec3A::ZERO).powf(2.2);
    }
    // `encode` (matrix + ST 2084) lands in [0, 1] by construction — pq_encode
    // saturates its input — so the clamp here is only the pack's own guard.
    let v = tone::encode(v, p).clamp(Vec3A::ZERO, Vec3A::ONE);
    let q = (v * 1023.0 + 0.5).floor();
    (q.x as u32) | ((q.y as u32) << 10) | ((q.z as u32) << 20) | (3 << 30)
}

pub struct VerifyReport {
    pub pixels: u64,
    /// Pixels the quadtree filled as sky while a reference ray hits geometry.
    pub false_sky: u64,
    /// Pixels whose hybrid ray (tmin = inherited t_start) missed the first
    /// surface a tmin=0 reference ray hits, or hit something behind it.
    pub overshoot: u64,
    /// Hybrid hit where reference missed (should be impossible).
    pub hybrid_extra: u64,
    /// Pixels cell-flooded by the depth cap — excluded from every counter
    /// above (a coarse pixel holds its cell sample's splatted t and may
    /// straddle sky/geometry), counted so callers can assert the capped path
    /// ran. Sparse-fill sample pixels are KIND_LEAF and are verified.
    pub coarse: u64,
    pub max_rel_err: f32,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.false_sky == 0 && self.overshoot == 0 && self.hybrid_extra == 0
    }
}

/// Ground-truth check: render one un-jittered hybrid frame, then compare every
/// pixel's primary-hit t against a tmin=0 reference ray. Detects the
/// spherical-vs-planar near-bound bug class (and now cut-dropping bugs)
/// directly. `max_depth` renders through the depth-capped driver instead of
/// the uncapped one; coarse (cell-flooded) pixels are skipped by the
/// comparison but counted, so all non-coarse pixels — including sparse-fill
/// sample pixels — must still verify exactly.
#[allow(clippy::too_many_arguments)]
pub fn verify(
    scene: &Scene,
    bvh: &Bvh,
    cam: &CamBasis,
    q: Quality,
    rw: usize,
    rh: usize,
    stats: &Stats,
    max_depth: Option<u32>,
    temporal_prev: &[(&TemporalCache, CamBasis)],
    temporal_cuts: Option<&temporal::CutStore>,
) -> VerifyReport {
    verify_sampled(scene, bvh, cam, q, rw, rh, stats, max_depth, temporal_prev, temporal_cuts, 1, 0)
}

/// `verify` at `spp` samples per pixel, gating sample `probe` (< spp): the
/// frame is rendered with `primary_sample = probe`, so tbuf holds THAT
/// sample's t, and the reference ray is rebuilt at THAT sample's sub-pixel
/// position — `dlss::jitter_for_sample(0, probe)`, a pure function of (frame,
/// k), which is the whole reason the extra samples take a deterministic offset
/// instead of an rng one. Sweeping probe over 0..spp gates every multi-sampled
/// ray against a tmin=0 reference, not just sample 0's.
#[allow(clippy::too_many_arguments)]
pub fn verify_sampled(
    scene: &Scene,
    bvh: &Bvh,
    cam: &CamBasis,
    q: Quality,
    rw: usize,
    rh: usize,
    stats: &Stats,
    max_depth: Option<u32>,
    temporal_prev: &[(&TemporalCache, CamBasis)],
    temporal_cuts: Option<&temporal::CutStore>,
    spp: u32,
    probe: u32,
) -> VerifyReport {
    crate::zone!("verify");
    assert!(probe < spp.max(1), "verify probe must name a sample the frame traces");
    // Sample `probe`'s sub-pixel position, frame 0 (verify's ctx). probe == 0
    // with jitter off is the pixel center — the historical reference ray.
    let (jx, jy) = if probe == 0 {
        (0.5, 0.5)
    } else {
        let (ox, oy) = dlss::jitter_for_sample(0, probe);
        (0.5 + ox, 0.5 + oy)
    };
    let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let ctx = FrameCtx {
        scene,
        bvh,
        cam: *cam,
        q,
        frame: 0,
        jitter: false,
        rw,
        rh,
        accum: &accum,
        info: &info,
        tbuf: &tbuf,
        stats,
        sun: sun_dir(scene),
        // verify's re-trace must shade the SAME sky as the frame it checks —
        // the pinned check state, matching every headless FrameCtx.
        clouds: crate::clouds::Clouds::check(scene.diag),
        fireflies: crate::fireflies::Fireflies::check(scene),
        tcache_cur: None,
        tcache_prev: temporal_prev,
        accumulate: true,
        gbuf: None,
        fsr_buf: None,
        prev_cam: None,
        frame_jitter: None,
        spp,
        primary_sample: probe,
        adaptive: false,
        hemi_share: false,
        replay_rec: None,
        cut_cur: None,
        cut_prev: temporal_cuts,
        discard_seeds: false,
        defer_shade: false,
    };
    match max_depth {
        Some(d) => render_frame_capped(&ctx, d),
        None => render_frame(&ctx, true),
    }

    (0..rh)
        .into_par_iter()
        .map(|y| {
            let mut rep = VerifyReport {
                pixels: 0,
                false_sky: 0,
                overshoot: 0,
                hybrid_extra: 0,
                coarse: 0,
                max_rel_err: 0.0,
            };
            let mut visits = 0u64;
            for x in 0..rw {
                // A coarse (cell-flooded) pixel carries its cell sample's t
                // and may straddle sky/geometry — exclude it from all
                // counters. Sparse-fill sample pixels are KIND_LEAF and ARE
                // gated: real rays with exact t.
                if overlay::info_kind(info[y * rw + x].load(Relaxed)) == KIND_COARSE {
                    rep.coarse += 1;
                    continue;
                }
                rep.pixels += 1;
                let dir = cam.ray_dir(x as f32 + jx, y as f32 + jy);
                let t_ref = bvh
                    .intersect(scene, &Ray::new(cam.origin, dir), 0.0, f32::INFINITY, &mut visits)
                    .map(|h| h.t);
                let t_h = f32::from_bits(tbuf[y * rw + x].load(Relaxed));
                match (t_ref, t_h.is_finite()) {
                    (Some(tr), true) => {
                        let rel = (t_h - tr).abs() / tr;
                        rep.max_rel_err = rep.max_rel_err.max(rel);
                        if rel > 1e-3 {
                            rep.overshoot += 1;
                        }
                    }
                    (Some(_), false) => {
                        let kind = overlay::info_kind(info[y * rw + x].load(Relaxed));
                        if kind == KIND_SKY {
                            rep.false_sky += 1;
                        } else {
                            rep.overshoot += 1;
                        }
                    }
                    (None, true) => rep.hybrid_extra += 1,
                    (None, false) => {}
                }
            }
            rep
        })
        .reduce(
            || VerifyReport {
                pixels: 0,
                false_sky: 0,
                overshoot: 0,
                hybrid_extra: 0,
                coarse: 0,
                max_rel_err: 0.0,
            },
            |a, b| VerifyReport {
                pixels: a.pixels + b.pixels,
                false_sky: a.false_sky + b.false_sky,
                overshoot: a.overshoot + b.overshoot,
                hybrid_extra: a.hybrid_extra + b.hybrid_extra,
                coarse: a.coarse + b.coarse,
                max_rel_err: a.max_rel_err.max(b.max_rel_err),
            },
        )
}

/// The sun's direction. Used to be `light.center.normalize()` — the direction of
/// a rect lamp 12 units away, which is why the sky's glow and the actual light
/// were two different objects that merely pointed the same way. Now it is just
/// the sun's own direction; there is one sun.
pub fn sun_dir(scene: &Scene) -> Vec3A {
    scene.sun.dir
}
