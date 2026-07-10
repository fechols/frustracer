use crate::bvh::{Bvh, Ray};
use crate::camera::CamBasis;
use crate::dlss;
use crate::frustum::{self, MAX_CUT};
use crate::overlay::{self, KIND_COARSE, KIND_LEAF, KIND_SKY};
use crate::scene::Scene;
use crate::shade::{self, Quality};
use crate::stats::{LocalStats, Stats};
use crate::temporal::{self, TemporalCache};
use glam::Vec3A;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// Tiles at or below this size stop subdividing and trace per-pixel rays.
/// Caps quadtree overhead at ~(pixels/64)·4/3 frustum queries per frame.
pub const LEAF_TILE: usize = 8;
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
    /// This frame's temporal cache to fill (per-node tc / sky markers).
    pub tcache_cur: Option<&'a TemporalCache>,
    /// The previous frame's cache and the exact basis it was traced with —
    /// consulted before each tile's bound query for a t_start head start.
    /// Must be exactly the last full-res hybrid frame (see main.rs wiring).
    pub tcache_prev: Option<(&'a TemporalCache, CamBasis)>,
    /// false => `splat` always stores (every frame is a fresh 1-spp frame;
    /// DLSS-RR is the temporal integrator). `frame` still advances so the
    /// per-pixel RNG decorrelates across frames — pinning frame to 0 would
    /// freeze the noise pattern, which the denoiser would treat as signal.
    pub accumulate: bool,
    /// DLSS G-buffers, written at the primary-hit fill sites. None (all
    /// legacy paths) costs one never-taken branch per pixel.
    pub gbuf: Option<&'a dlss::GBufs>,
    /// Previous frame's camera basis for motion vectors (independent of the
    /// temporal cache's tprev_basis — different contract).
    pub prev_cam: Option<CamBasis>,
    /// Frame-uniform sub-pixel jitter offset in [-0.5, 0.5) (DLSS mode);
    /// None => the legacy per-pixel rng jitter controlled by `jitter`.
    pub frame_jitter: Option<(f32, f32)>,
}

impl FrameCtx<'_> {
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
        trace_tile(ctx, 0, 0, ctx.rw, ctx.rh, 0.0, 0, 0, &[0], u32::MAX);
    } else {
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
    cut_in: &[u32],
    ls: &mut LocalStats,
) -> TileStep {
    let f = ctx.cam.tile_frustum(x0, y0, x1, y1);
    let mut visits = 0u64;
    ls.tiles += 1;
    ls.frustum_queries += 1;
    let result = frustum::nearest_geometry_distance(ctx.bvh, &f, t_start, cut_in, &mut visits);
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
            let len = frustum::refine_cut(ctx.bvh, &f, tc, f32::INFINITY, cut_in, &mut cut, &mut visits, &mut ls.cut_overflows);
            ls.cut_len_sum += len as u64;
            TileStep::Split { tc, cut, len }
        }
    };
    ls.frustum_nodes += visits;
    step
}

/// The frustracer core: trace the tile's frustum until it could hit geometry,
/// then split into 4 quadrants that inherit the proven-empty distance and the
/// refined node cut. Tiles that reach `max_depth` unresolved (not sky, not
/// leaf) become one flat-shaded quad; `u32::MAX` means uncapped.
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
) {
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let (w, h) = (x1 - x0, y1 - y0);
    if w <= LEAF_TILE && h <= LEAF_TILE {
        shade_tile(ctx, x0, y0, x1, y1, t_start, depth, KIND_LEAF, cut_in);
        return;
    }
    // After the leaf check, so a cap >= the leaf depth is exactly uncapped.
    // Flat-fill uses the inherited cut and t_start — same as the split would.
    // No temporal probe or store here: the inherited t_start already carries
    // the parent's (possibly seeded) tc, and its cache entry covers this tile.
    if depth >= max_depth {
        let mut ls = LocalStats::default();
        flat_fill(ctx, x0, y0, x1, y1, t_start, depth, cut_in, &mut ls);
        ctx.stats.add(&ls);
        return;
    }

    let mut ls = LocalStats::default();
    // Temporal probe: harvest a proven-empty head start from the previous
    // frame's quadtree before touching the BVH. Only the primary-path t_start
    // is affected — secondary rays never see it.
    let mut t0 = t_start;
    if let Some((prev, prev_cam)) = &ctx.tcache_prev {
        match temporal::lookup(prev, prev_cam, &ctx.cam, ctx.rw, ctx.rh, x0, y0, x1, y1, t_start, depth, path, &mut ls) {
            temporal::Seed::Sky => {
                // The whole frustum was proven empty last frame; still true.
                if let Some(cur) = ctx.tcache_cur {
                    cur.store(depth, path, f32::INFINITY);
                }
                ls.temporal_sky_tiles += 1;
                ctx.stats.add(&ls);
                fill_sky(ctx, x0, y0, x1, y1, depth);
                return;
            }
            temporal::Seed::T(t) => {
                if t > t0 {
                    ls.temporal_seeds += 1;
                    t0 = t;
                }
            }
            temporal::Seed::None => {}
        }
    }
    match tile_step(ctx, x0, y0, x1, y1, t0, cut_in, &mut ls) {
        TileStep::Sky => {
            // Composed claim: nothing outside ball(origin, t0) by this query,
            // nothing inside it by the inherited/seeded claim — the whole
            // cone is empty, which is what +INF asserts to the next frame.
            if let Some(cur) = ctx.tcache_cur {
                cur.store(depth, path, f32::INFINITY);
            }
            ctx.stats.add(&ls);
            fill_sky(ctx, x0, y0, x1, y1, depth);
        }
        TileStep::Split { tc, cut, len } => {
            if let Some(cur) = ctx.tcache_cur {
                cur.store(depth, path, tc);
            }
            ctx.stats.add(&ls);
            // A Some bound with an empty cut should be impossible (the nearest
            // leaf survives the tc ball-cull); never degrade toward sky.
            debug_assert!(len > 0, "refine_cut emptied a non-sky tile");
            let child: &[u32] = if len > 0 { &cut[..len] } else { cut_in };
            let xm = x0 + w / 2;
            let ym = y0 + h / 2;
            let d = depth + 1;
            // Child paths (2 bits per level: TL=0 TR=1 BL=2 BR=3) — must match
            // temporal::rect_for_path, which replays these splits.
            let p = path << 2;
            if w.max(h) > SPAWN_MIN {
                rayon::join(
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
            } else {
                trace_tile(ctx, x0, y0, xm, ym, tc, d, p, child, max_depth);
                trace_tile(ctx, xm, y0, x1, ym, tc, d, p | 1, child, max_depth);
                trace_tile(ctx, x0, ym, xm, y1, tc, d, p | 2, child, max_depth);
                trace_tile(ctx, xm, ym, x1, y1, tc, d, p | 3, child, max_depth);
            }
        }
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
    for y in y0..y1 {
        for x in x0..x1 {
            shade_pixel(ctx, x, y, t_start, depth, kind, Some(cut), &mut ls);
        }
    }
    ctx.stats.add(&ls);
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
) {
    let Some(g) = ctx.gbuf else { return };
    let (_, far) = dlss::near_far(ctx.scene.diag);
    let mv = match &ctx.prev_cam {
        Some(prev) => {
            let p = ctx.cam.origin + dir * t;
            match prev.project(p - prev.origin) {
                Some((px, py)) => (px - fx, py - fy),
                None => (0.0, 0.0), // behind the old image plane: disocclusion
            }
        }
        None => (0.0, 0.0),
    };
    g.write(
        x,
        y,
        &dlss::GPixel {
            normal: prim.n,
            rough: prim.roughness,
            diff_alb: prim.albedo * (1.0 - prim.metallic),
            spec_alb: {
                let f0 = Vec3A::splat(0.04).lerp(prim.albedo, prim.metallic);
                (f0.x + f0.y + f0.z) / 3.0
            },
            view_z: t * dir.dot(ctx.cam.forward()),
            mv,
            spec_hit_t: if prim.spec_t.is_infinite() { far } else { prim.spec_t },
        },
    );
}

/// G-buffer write for a sky/miss pixel: depth = far (finite, f16-safe), MV =
/// direction-only reprojection (exact for an environment at infinity — zero
/// under pure translation, the pan vector under rotation).
fn write_gbuf_sky(ctx: &FrameCtx, x: usize, y: usize, fx: f32, fy: f32, dir: Vec3A) {
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
            spec_alb: 0.0,
            view_z: far,
            mv,
            spec_hit_t: 0.0,
        },
    );
}

fn shade_pixel(
    ctx: &FrameCtx,
    x: usize,
    y: usize,
    t_start: f32,
    depth: u32,
    kind: u32,
    cut: Option<&[u32]>,
    ls: &mut LocalStats,
) {
    let seed = (x as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F))
        .wrapping_add(ctx.frame as u64);
    let mut rng = fastrand::Rng::with_seed(seed);
    // DLSS mode: one frame-uniform low-discrepancy offset for every pixel
    // (the denoiser is told this exact offset). Legacy: per-pixel rng.
    // Either way the sample stays in [x, x+1) — the invariant leaf rays need.
    let (jx, jy) = match ctx.frame_jitter {
        Some((ox, oy)) => (0.5 + ox, 0.5 + oy),
        None => {
            if ctx.jitter {
                (rng.f32(), rng.f32())
            } else {
                (0.5, 0.5)
            }
        }
    };
    let (fx, fy) = (x as f32 + jx, y as f32 + jy);
    let dir = ctx.cam.ray_dir(fx, fy);
    let ray = Ray::new(ctx.cam.origin, dir);
    ls.primary_rays += 1;
    let hit = match cut {
        // Quadtree leaf: seed the traversal from the tile's inherited cut.
        Some(roots) => ctx.bvh.intersect_multi(ctx.scene, &ray, t_start, f32::INFINITY, roots, &mut ls.ray_nodes),
        // Plain reference path: full traversal from the root.
        None => ctx.bvh.intersect(ctx.scene, &ray, t_start, f32::INFINITY, &mut ls.ray_nodes),
    };
    match hit {
        Some(hit) => {
            ctx.store_meta(x, y, hit.t, depth, kind);
            if t_start > 0.0 {
                ls.skip_ratio_micro += (t_start / hit.t * 1e6) as u64;
            }
            ls.skip_ratio_count += 1;
            let mut prim = shade::PrimarySurface::default();
            let c = shade::shade(
                ctx.scene,
                ctx.bvh,
                &ray,
                &hit,
                &ctx.q,
                &mut rng,
                ctx.sun,
                0,
                ls,
                if ctx.gbuf.is_some() { Some(&mut prim) } else { None },
            );
            ctx.splat(x, y, c);
            write_gbuf_hit(ctx, x, y, fx, fy, dir, hit.t, &prim);
        }
        None => {
            ctx.store_meta(x, y, f32::INFINITY, depth, kind);
            ctx.splat(x, y, shade::sky(dir, ctx.sun));
            write_gbuf_sky(ctx, x, y, fx, fy, dir);
        }
    }
}

/// The depth cap never flat-fills tiles shallower than this (a bad estimate
/// must not paint the whole screen as a handful of quads).
pub const MIN_BUDGET_DEPTH: u32 = 2;

/// Depth-capped dynamic resolution: the same depth-first recursion as
/// `render_frame`, but every screen area stops at a uniform `max_depth` and
/// unresolved tiles there become one flat-shaded quad each. The caller (the
/// frame-budget controller in main.rs) estimates `max_depth` from the previous
/// frame's time — no wall clock is read here, so the frame is deterministic
/// for a given cap. A cap at or beyond the leaf depth is bit-identical to
/// `render_frame(ctx, true)` (same code path, the cap never fires).
pub fn render_frame_capped(ctx: &FrameCtx, max_depth: u32) {
    trace_tile(ctx, 0, 0, ctx.rw, ctx.rh, 0.0, 0, 0, &[0], max_depth.max(MIN_BUDGET_DEPTH));
}

/// Depth cap reached: this quad becomes one "pixel" — the color of a single
/// representative ray through its center, starting at the inherited distance.
fn flat_fill(
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
    let dir = ctx.cam.ray_dir((x0 + x1) as f32 * 0.5, (y0 + y1) as f32 * 0.5);
    let ray = Ray::new(ctx.cam.origin, dir);
    let seed = (x0 as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((y0 as u64).wrapping_mul(0xC2B2AE3D27D4EB4F))
        .wrapping_add(ctx.frame as u64);
    let mut rng = fastrand::Rng::with_seed(seed);
    ls.primary_rays += 1;
    let mut prim = shade::PrimarySurface::default();
    let (c, thit) = match ctx.bvh.intersect_multi(ctx.scene, &ray, t_start, f32::INFINITY, cut, &mut ls.ray_nodes) {
        Some(hit) => {
            let c = shade::shade(
                ctx.scene,
                ctx.bvh,
                &ray,
                &hit,
                &ctx.q,
                &mut rng,
                ctx.sun,
                0,
                ls,
                if ctx.gbuf.is_some() { Some(&mut prim) } else { None },
            );
            (c, hit.t)
        }
        None => (shade::sky(dir, ctx.sun), f32::INFINITY),
    };
    ls.coarse_tiles += 1;
    ls.coarse_pixels += ((x1 - x0) * (y1 - y0)) as u64;
    for y in y0..y1 {
        for x in x0..x1 {
            ctx.store_meta(x, y, thit, depth, KIND_COARSE);
            ctx.splat(x, y, c);
            // Shared center-ray surface for the whole quad, but per-pixel
            // sample position — the write helpers subtract (fx, fy), so the
            // camera-motion part of the MV stays per-pixel-correct even in
            // coarse quads.
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            if thit.is_finite() {
                write_gbuf_hit(ctx, x, y, fx, fy, dir, thit, &prim);
            } else {
                write_gbuf_sky(ctx, x, y, fx, fy, dir);
            }
        }
    }
}

/// The frustum proved this whole tile empty — fill with sky, zero rays traced.
fn fill_sky(ctx: &FrameCtx, x0: usize, y0: usize, x1: usize, y1: usize, depth: u32) {
    let mut ls = LocalStats::default();
    ls.sky_tiles = 1;
    ls.sky_pixels = ((x1 - x0) * (y1 - y0)) as u64;
    for y in y0..y1 {
        for x in x0..x1 {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let dir = ctx.cam.ray_dir(fx, fy);
            ctx.store_meta(x, y, f32::INFINITY, depth, KIND_SKY);
            ctx.splat(x, y, shade::sky(dir, ctx.sun));
            write_gbuf_sky(ctx, x, y, fx, fy, dir);
        }
    }
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
    let inv = 1.0 / samples.max(1) as f32;
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
            // soft rolloff + gamma
            let mut c = (Vec3A::ONE - (-c).exp()).powf(1.0 / 2.2);
            if overlay_on {
                let pi = info[sy * rw + sx].load(Relaxed);
                let (tint, alpha) = overlay::tint(pi);
                c = c.lerp(tint, alpha);
                let right = if sx + 1 < rw { info[sy * rw + sx + 1].load(Relaxed) } else { pi };
                let down = if sy + 1 < rh { info[(sy + 1) * rw + sx].load(Relaxed) } else { pi };
                if right != pi || down != pi {
                    c *= 0.25; // tile border
                }
            }
            let c = (c.clamp(Vec3A::ZERO, Vec3A::ONE) * 255.0 + 0.5).floor();
            *out = ((c.x as u32) << 16) | ((c.y as u32) << 8) | c.z as u32;
        }
    });
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
    /// Pixels flat-filled by the depth cap — excluded from every counter above
    /// (a coarse quad holds one splatted center-ray t and may straddle
    /// sky/geometry), counted so callers can assert the capped path ran.
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
/// the uncapped one; coarse (flat-filled) pixels are skipped by the comparison
/// but counted, so all non-coarse pixels must still verify exactly.
pub fn verify(
    scene: &Scene,
    bvh: &Bvh,
    cam: &CamBasis,
    q: Quality,
    rw: usize,
    rh: usize,
    stats: &Stats,
    max_depth: Option<u32>,
    temporal_prev: Option<(&TemporalCache, CamBasis)>,
) -> VerifyReport {
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
        tcache_cur: None,
        tcache_prev: temporal_prev,
        accumulate: true,
        gbuf: None,
        prev_cam: None,
        frame_jitter: None,
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
                // A coarse quad carries its center ray's t on every pixel and
                // may straddle sky/geometry — exclude it from all counters.
                if overlay::info_kind(info[y * rw + x].load(Relaxed)) == KIND_COARSE {
                    rep.coarse += 1;
                    continue;
                }
                rep.pixels += 1;
                let dir = cam.ray_dir(x as f32 + 0.5, y as f32 + 0.5);
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

pub fn sun_dir(scene: &Scene) -> Vec3A {
    scene.light.center.normalize_or_zero()
}
