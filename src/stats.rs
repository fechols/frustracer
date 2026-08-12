use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Per-frame render counters. Shared across rayon tasks; hot paths batch into
/// `LocalStats` and flush once per tile so the counter cache lines don't serialize.
#[derive(Default)]
pub struct Stats {
    pub frustum_queries: AtomicU64,
    pub frustum_nodes: AtomicU64,
    pub ray_nodes: AtomicU64,
    /// The PRIMARY share of `ray_nodes` (camera rays only). `ray_nodes` stays
    /// the TOTAL — the builder bake-off scores against it meaning "all node
    /// visits" (see the two-tree bullet in CLAUDE.md, `builders.rs`, `bvh.rs`)
    /// and the adopt bench divides it — so secondary is `ray_nodes - this`.
    ///
    /// It exists because ray count and ray COST are very different things here:
    /// shadow/AO rays outnumber primaries 2.7-4.2x, but they stop at the first
    /// blocker, so what they cost in traversal cannot be inferred from the
    /// counts. Anything that removes primary traversal (a proven-covered tile,
    /// say) is bounded by this share.
    pub ray_nodes_prim: AtomicU64,
    pub primary_rays: AtomicU64,
    pub secondary_rays: AtomicU64,
    /// Shadow rays traced by the emissive cluster-light NEE loop (a subset of
    /// `secondary_rays`) — the `--check-gpu`/`--check-dxr` must-fire signal on
    /// emissive scenes (src/emissive.rs).
    pub emissive_rays: AtomicU64,
    /// LEVEL-0 GI gather rays (a subset of `secondary_rays`) — the run_check
    /// must-fire that real-time GI is live on armed sessions, exactly 0 under
    /// `--no-rtgi` / `--rtgi-bounces 0` (src/shade.rs's RTGI ambient arm).
    /// Deterministic at `--rtgi-bounces >= 1`, rouletted at 0.5.
    pub rtgi_rays: AtomicU64,
    /// LEVEL-1-AND-DEEPER GI gather rays (a subset of `secondary_rays`). The
    /// pair `(rtgi_rays, rtgi_rays2)` is the ladder's rung signature, which is
    /// what the `--check`/`--check-gpu`/`--check-vk` must-fires read: (0,0) at
    /// `--rtgi-bounces 0`, (>0,0) at 0.5 and 1, (>0,>0) at 1.5 and 2.
    pub rtgi_rays2: AtomicU64,
    /// Leaf/capped tiles that ran the emissive per-tile light cull
    /// (`emissive::cull_tile`) — the run_check must-fire that the cull is
    /// live on armed emissive scenes (hybrid arms only; the plain reference
    /// deliberately never culls).
    pub el_cull_tiles: AtomicU64,
    /// Σ lights culled by those tiles (out of tiles × scene count) — the
    /// cull's effectiveness gauge, printed as the `el-cull:` segment.
    pub el_cull_culled: AtomicU64,
    pub sky_pixels: AtomicU64,
    pub sky_tiles: AtomicU64,
    pub blocked_queries: AtomicU64,
    pub tiles: AtomicU64,
    /// Sum of (t_start / t_hit) * 1e6 over hit pixels — mean tells how much
    /// empty space the frustum pass skipped before the pixel ray even started.
    pub skip_ratio_micro: AtomicU64,
    pub skip_ratio_count: AtomicU64,
    /// Sum of refined node-cut lengths (mean = sum / non-sky queries).
    pub cut_len_sum: AtomicU64,
    /// Cut refinements that ran out of budget and emitted an internal node.
    pub cut_overflows: AtomicU64,
    /// Tiles sparse-filled at the depth cap, and their flooded (non-sample) pixels.
    pub coarse_tiles: AtomicU64,
    pub coarse_pixels: AtomicU64,
    /// Real point samples shot inside sparse-filled tiles (one per cell).
    pub coarse_samples: AtomicU64,
    /// Tiles whose t_start was raised by a previous-frame temporal seed.
    pub temporal_seeds: AtomicU64,
    /// Tiles sky-filled straight from the temporal cache (zero BVH work).
    pub temporal_sky_tiles: AtomicU64,
    /// Cells visited by the temporal region-min query (its cost).
    pub temporal_tests: AtomicU64,
    /// Tiles answered by a ring entry OTHER than the newest cache (pan-back
    /// reuse), and the sum of those entries' ages (mean = sum / hits).
    pub temporal_ring_hits: AtomicU64,
    pub temporal_ring_age_sum: AtomicU64,
    /// Tiles that adopted an old node's cut and skipped their bound query;
    /// of those, the ones whose re-refine emptied — a free sky proof.
    pub temporal_cut_adopts: AtomicU64,
    pub temporal_adopt_sky: AtomicU64,
    /// Age-capped candidates skipped by the containment search (the node is
    /// forced back onto a real query before its cut may chain on).
    pub temporal_adopt_requery: AtomicU64,
    /// Split nodes whose cut could not be stored (arena full) — consumers
    /// just miss those adoptions.
    pub temporal_cut_arena_full: AtomicU64,
    /// Shading points that ran the hemisphere frustum integrator.
    pub hemi_points: AtomicU64,
    /// Hemisphere-cell bound queries (+ refines) and their BVH node visits.
    pub hemi_queries: AtomicU64,
    pub hemi_nodes: AtomicU64,
    /// Cells resolved analytically (proven empty/open — zero rays).
    pub hemi_cells_empty: AtomicU64,
    /// Budget-depth cells that shot their one stratified ray.
    pub hemi_leaf_rays: AtomicU64,
    /// Hemi sharing (one root capture per coherent 2×2 group): groups
    /// captured, points integrated from a shared seed (rep included), and fb
    /// points that failed the group predicate and ran their own root.
    pub hemi_share_groups: AtomicU64,
    pub hemi_share_points: AtomicU64,
    pub hemi_share_fallback: AtomicU64,
    /// Adaptive-rate cells (XeSS mode): shared-visibility, per-pixel, and
    /// supersampled tiers; edge pixels that didn't form a full cell.
    pub adapt_coarse: AtomicU64,
    pub adapt_base: AtomicU64,
    pub adapt_hot: AtomicU64,
    pub adapt_partial_px: AtomicU64,
    /// Second full samples shot in HOT cells (4 per hot cell).
    pub adapt_topup: AtomicU64,
    /// Coherent cells that fell back to per-pixel rays on fractional
    /// (penumbral) shared visibility.
    pub adapt_penumbra: AtomicU64,
    /// Shadow/AO rays skipped by applying a shared VisRecord.
    pub adapt_rays_saved: AtomicU64,
    /// Structure-replay frames: leaf/sky terminals re-shaded from the
    /// recorded quadtree (zero frustum queries — see replay.rs).
    pub replay_leaf_tiles: AtomicU64,
    pub replay_sky_tiles: AtomicU64,
    /// Deferred material-sorted shading (--defer-shade): pixels shaded
    /// through merged same-material buckets, bucket flushes (mean bucket =
    /// px/flushes), and leaves that shaded INLINE instead — sky, an untextured
    /// material, or a mid-leaf material mismatch (`render::defer_leaf`).
    pub defer_px: AtomicU64,
    pub defer_flushes: AtomicU64,
    pub defer_mixed: AtomicU64,
}

#[derive(Default)]
pub struct LocalStats {
    pub frustum_queries: u64,
    pub frustum_nodes: u64,
    pub ray_nodes: u64,
    /// Primary-only share of `ray_nodes`; see `Stats::ray_nodes_prim`.
    pub ray_nodes_prim: u64,
    pub primary_rays: u64,
    pub secondary_rays: u64,
    /// Emissive-NEE subset of `secondary_rays` (see `Stats::emissive_rays`).
    pub emissive_rays: u64,
    /// Level-0 GI-gather subset of `secondary_rays` (see `Stats::rtgi_rays`).
    pub rtgi_rays: u64,
    /// Level-1-and-deeper GI gathers (see `Stats::rtgi_rays2`).
    pub rtgi_rays2: u64,
    /// Tiles that ran the emissive light cull (see `Stats::el_cull_tiles`).
    pub el_cull_tiles: u64,
    /// Lights culled by those tiles (see `Stats::el_cull_culled`).
    pub el_cull_culled: u64,
    pub sky_pixels: u64,
    pub sky_tiles: u64,
    pub blocked_queries: u64,
    pub tiles: u64,
    pub skip_ratio_micro: u64,
    pub skip_ratio_count: u64,
    pub cut_len_sum: u64,
    pub cut_overflows: u64,
    pub coarse_tiles: u64,
    pub coarse_pixels: u64,
    pub coarse_samples: u64,
    pub temporal_seeds: u64,
    pub temporal_sky_tiles: u64,
    pub temporal_tests: u64,
    pub temporal_ring_hits: u64,
    pub temporal_ring_age_sum: u64,
    pub temporal_cut_adopts: u64,
    pub temporal_adopt_sky: u64,
    pub temporal_adopt_requery: u64,
    pub temporal_cut_arena_full: u64,
    pub hemi_points: u64,
    pub hemi_queries: u64,
    pub hemi_nodes: u64,
    pub hemi_cells_empty: u64,
    pub hemi_leaf_rays: u64,
    pub hemi_share_groups: u64,
    pub hemi_share_points: u64,
    pub hemi_share_fallback: u64,
    pub adapt_coarse: u64,
    pub adapt_base: u64,
    pub adapt_hot: u64,
    pub adapt_partial_px: u64,
    pub adapt_topup: u64,
    pub adapt_penumbra: u64,
    pub adapt_rays_saved: u64,
    pub replay_leaf_tiles: u64,
    pub replay_sky_tiles: u64,
    pub defer_px: u64,
    pub defer_flushes: u64,
    pub defer_mixed: u64,
}

impl LocalStats {
    /// Fold another batch in — used by the parallel `--check` probe sweeps,
    /// which keep per-probe LocalStats and reduce them sequentially.
    pub fn merge(&mut self, o: &LocalStats) {
        self.frustum_queries += o.frustum_queries;
        self.frustum_nodes += o.frustum_nodes;
        self.ray_nodes += o.ray_nodes;
        self.ray_nodes_prim += o.ray_nodes_prim;
        self.primary_rays += o.primary_rays;
        self.secondary_rays += o.secondary_rays;
        self.emissive_rays += o.emissive_rays;
        self.rtgi_rays += o.rtgi_rays;
        self.rtgi_rays2 += o.rtgi_rays2;
        self.el_cull_tiles += o.el_cull_tiles;
        self.el_cull_culled += o.el_cull_culled;
        self.sky_pixels += o.sky_pixels;
        self.sky_tiles += o.sky_tiles;
        self.blocked_queries += o.blocked_queries;
        self.tiles += o.tiles;
        self.skip_ratio_micro += o.skip_ratio_micro;
        self.skip_ratio_count += o.skip_ratio_count;
        self.cut_len_sum += o.cut_len_sum;
        self.cut_overflows += o.cut_overflows;
        self.coarse_tiles += o.coarse_tiles;
        self.coarse_pixels += o.coarse_pixels;
        self.coarse_samples += o.coarse_samples;
        self.temporal_seeds += o.temporal_seeds;
        self.temporal_sky_tiles += o.temporal_sky_tiles;
        self.temporal_tests += o.temporal_tests;
        self.temporal_ring_hits += o.temporal_ring_hits;
        self.temporal_ring_age_sum += o.temporal_ring_age_sum;
        self.temporal_cut_adopts += o.temporal_cut_adopts;
        self.temporal_adopt_sky += o.temporal_adopt_sky;
        self.temporal_adopt_requery += o.temporal_adopt_requery;
        self.temporal_cut_arena_full += o.temporal_cut_arena_full;
        self.hemi_points += o.hemi_points;
        self.hemi_queries += o.hemi_queries;
        self.hemi_nodes += o.hemi_nodes;
        self.hemi_cells_empty += o.hemi_cells_empty;
        self.hemi_leaf_rays += o.hemi_leaf_rays;
        self.hemi_share_groups += o.hemi_share_groups;
        self.hemi_share_points += o.hemi_share_points;
        self.hemi_share_fallback += o.hemi_share_fallback;
        self.adapt_coarse += o.adapt_coarse;
        self.adapt_base += o.adapt_base;
        self.adapt_hot += o.adapt_hot;
        self.adapt_partial_px += o.adapt_partial_px;
        self.adapt_topup += o.adapt_topup;
        self.adapt_penumbra += o.adapt_penumbra;
        self.adapt_rays_saved += o.adapt_rays_saved;
        self.replay_leaf_tiles += o.replay_leaf_tiles;
        self.replay_sky_tiles += o.replay_sky_tiles;
        self.defer_px += o.defer_px;
        self.defer_flushes += o.defer_flushes;
        self.defer_mixed += o.defer_mixed;
    }
}

impl Stats {
    pub fn clear(&self) {
        self.frustum_queries.store(0, Relaxed);
        self.frustum_nodes.store(0, Relaxed);
        self.ray_nodes.store(0, Relaxed);
        self.ray_nodes_prim.store(0, Relaxed);
        self.primary_rays.store(0, Relaxed);
        self.secondary_rays.store(0, Relaxed);
        self.emissive_rays.store(0, Relaxed);
        self.rtgi_rays.store(0, Relaxed);
        self.rtgi_rays2.store(0, Relaxed);
        self.el_cull_tiles.store(0, Relaxed);
        self.el_cull_culled.store(0, Relaxed);
        self.sky_pixels.store(0, Relaxed);
        self.sky_tiles.store(0, Relaxed);
        self.blocked_queries.store(0, Relaxed);
        self.tiles.store(0, Relaxed);
        self.skip_ratio_micro.store(0, Relaxed);
        self.skip_ratio_count.store(0, Relaxed);
        self.cut_len_sum.store(0, Relaxed);
        self.cut_overflows.store(0, Relaxed);
        self.coarse_tiles.store(0, Relaxed);
        self.coarse_pixels.store(0, Relaxed);
        self.coarse_samples.store(0, Relaxed);
        self.temporal_seeds.store(0, Relaxed);
        self.temporal_sky_tiles.store(0, Relaxed);
        self.temporal_tests.store(0, Relaxed);
        self.temporal_ring_hits.store(0, Relaxed);
        self.temporal_ring_age_sum.store(0, Relaxed);
        self.temporal_cut_adopts.store(0, Relaxed);
        self.temporal_adopt_sky.store(0, Relaxed);
        self.temporal_adopt_requery.store(0, Relaxed);
        self.temporal_cut_arena_full.store(0, Relaxed);
        self.hemi_points.store(0, Relaxed);
        self.hemi_queries.store(0, Relaxed);
        self.hemi_nodes.store(0, Relaxed);
        self.hemi_cells_empty.store(0, Relaxed);
        self.hemi_leaf_rays.store(0, Relaxed);
        self.hemi_share_groups.store(0, Relaxed);
        self.hemi_share_points.store(0, Relaxed);
        self.hemi_share_fallback.store(0, Relaxed);
        self.adapt_coarse.store(0, Relaxed);
        self.adapt_base.store(0, Relaxed);
        self.adapt_hot.store(0, Relaxed);
        self.adapt_partial_px.store(0, Relaxed);
        self.adapt_topup.store(0, Relaxed);
        self.adapt_penumbra.store(0, Relaxed);
        self.adapt_rays_saved.store(0, Relaxed);
        self.replay_leaf_tiles.store(0, Relaxed);
        self.replay_sky_tiles.store(0, Relaxed);
        self.defer_px.store(0, Relaxed);
        self.defer_flushes.store(0, Relaxed);
        self.defer_mixed.store(0, Relaxed);
    }

    pub fn add(&self, l: &LocalStats) {
        if l.frustum_queries > 0 {
            self.frustum_queries.fetch_add(l.frustum_queries, Relaxed);
        }
        if l.frustum_nodes > 0 {
            self.frustum_nodes.fetch_add(l.frustum_nodes, Relaxed);
        }
        if l.ray_nodes > 0 {
            self.ray_nodes.fetch_add(l.ray_nodes, Relaxed);
        }
        if l.ray_nodes_prim > 0 {
            self.ray_nodes_prim.fetch_add(l.ray_nodes_prim, Relaxed);
        }
        if l.primary_rays > 0 {
            self.primary_rays.fetch_add(l.primary_rays, Relaxed);
        }
        if l.secondary_rays > 0 {
            self.secondary_rays.fetch_add(l.secondary_rays, Relaxed);
        }
        if l.emissive_rays > 0 {
            self.emissive_rays.fetch_add(l.emissive_rays, Relaxed);
        }
        if l.rtgi_rays > 0 {
            self.rtgi_rays.fetch_add(l.rtgi_rays, Relaxed);
        }
        if l.rtgi_rays2 > 0 {
            self.rtgi_rays2.fetch_add(l.rtgi_rays2, Relaxed);
        }
        if l.el_cull_tiles > 0 {
            self.el_cull_tiles.fetch_add(l.el_cull_tiles, Relaxed);
        }
        if l.el_cull_culled > 0 {
            self.el_cull_culled.fetch_add(l.el_cull_culled, Relaxed);
        }
        if l.sky_pixels > 0 {
            self.sky_pixels.fetch_add(l.sky_pixels, Relaxed);
        }
        if l.sky_tiles > 0 {
            self.sky_tiles.fetch_add(l.sky_tiles, Relaxed);
        }
        if l.blocked_queries > 0 {
            self.blocked_queries.fetch_add(l.blocked_queries, Relaxed);
        }
        if l.tiles > 0 {
            self.tiles.fetch_add(l.tiles, Relaxed);
        }
        if l.skip_ratio_micro > 0 {
            self.skip_ratio_micro.fetch_add(l.skip_ratio_micro, Relaxed);
        }
        if l.skip_ratio_count > 0 {
            self.skip_ratio_count.fetch_add(l.skip_ratio_count, Relaxed);
        }
        if l.cut_len_sum > 0 {
            self.cut_len_sum.fetch_add(l.cut_len_sum, Relaxed);
        }
        if l.cut_overflows > 0 {
            self.cut_overflows.fetch_add(l.cut_overflows, Relaxed);
        }
        if l.coarse_tiles > 0 {
            self.coarse_tiles.fetch_add(l.coarse_tiles, Relaxed);
        }
        if l.coarse_pixels > 0 {
            self.coarse_pixels.fetch_add(l.coarse_pixels, Relaxed);
        }
        if l.coarse_samples > 0 {
            self.coarse_samples.fetch_add(l.coarse_samples, Relaxed);
        }
        if l.temporal_seeds > 0 {
            self.temporal_seeds.fetch_add(l.temporal_seeds, Relaxed);
        }
        if l.temporal_sky_tiles > 0 {
            self.temporal_sky_tiles.fetch_add(l.temporal_sky_tiles, Relaxed);
        }
        if l.temporal_tests > 0 {
            self.temporal_tests.fetch_add(l.temporal_tests, Relaxed);
        }
        if l.temporal_ring_hits > 0 {
            self.temporal_ring_hits.fetch_add(l.temporal_ring_hits, Relaxed);
        }
        if l.temporal_ring_age_sum > 0 {
            self.temporal_ring_age_sum.fetch_add(l.temporal_ring_age_sum, Relaxed);
        }
        if l.temporal_cut_adopts > 0 {
            self.temporal_cut_adopts.fetch_add(l.temporal_cut_adopts, Relaxed);
        }
        if l.temporal_adopt_sky > 0 {
            self.temporal_adopt_sky.fetch_add(l.temporal_adopt_sky, Relaxed);
        }
        if l.temporal_adopt_requery > 0 {
            self.temporal_adopt_requery.fetch_add(l.temporal_adopt_requery, Relaxed);
        }
        if l.temporal_cut_arena_full > 0 {
            self.temporal_cut_arena_full.fetch_add(l.temporal_cut_arena_full, Relaxed);
        }
        if l.hemi_points > 0 {
            self.hemi_points.fetch_add(l.hemi_points, Relaxed);
        }
        if l.hemi_queries > 0 {
            self.hemi_queries.fetch_add(l.hemi_queries, Relaxed);
        }
        if l.hemi_nodes > 0 {
            self.hemi_nodes.fetch_add(l.hemi_nodes, Relaxed);
        }
        if l.hemi_cells_empty > 0 {
            self.hemi_cells_empty.fetch_add(l.hemi_cells_empty, Relaxed);
        }
        if l.hemi_leaf_rays > 0 {
            self.hemi_leaf_rays.fetch_add(l.hemi_leaf_rays, Relaxed);
        }
        if l.hemi_share_groups > 0 {
            self.hemi_share_groups.fetch_add(l.hemi_share_groups, Relaxed);
        }
        if l.hemi_share_points > 0 {
            self.hemi_share_points.fetch_add(l.hemi_share_points, Relaxed);
        }
        if l.hemi_share_fallback > 0 {
            self.hemi_share_fallback.fetch_add(l.hemi_share_fallback, Relaxed);
        }
        if l.adapt_coarse > 0 {
            self.adapt_coarse.fetch_add(l.adapt_coarse, Relaxed);
        }
        if l.adapt_base > 0 {
            self.adapt_base.fetch_add(l.adapt_base, Relaxed);
        }
        if l.adapt_hot > 0 {
            self.adapt_hot.fetch_add(l.adapt_hot, Relaxed);
        }
        if l.adapt_partial_px > 0 {
            self.adapt_partial_px.fetch_add(l.adapt_partial_px, Relaxed);
        }
        if l.adapt_topup > 0 {
            self.adapt_topup.fetch_add(l.adapt_topup, Relaxed);
        }
        if l.adapt_penumbra > 0 {
            self.adapt_penumbra.fetch_add(l.adapt_penumbra, Relaxed);
        }
        if l.adapt_rays_saved > 0 {
            self.adapt_rays_saved.fetch_add(l.adapt_rays_saved, Relaxed);
        }
        if l.replay_leaf_tiles > 0 {
            self.replay_leaf_tiles.fetch_add(l.replay_leaf_tiles, Relaxed);
        }
        if l.replay_sky_tiles > 0 {
            self.replay_sky_tiles.fetch_add(l.replay_sky_tiles, Relaxed);
        }
        if l.defer_px > 0 {
            self.defer_px.fetch_add(l.defer_px, Relaxed);
        }
        if l.defer_flushes > 0 {
            self.defer_flushes.fetch_add(l.defer_flushes, Relaxed);
        }
        if l.defer_mixed > 0 {
            self.defer_mixed.fetch_add(l.defer_mixed, Relaxed);
        }
    }

    pub fn summary_line(&self) -> String {
        let fq = self.frustum_queries.load(Relaxed);
        let fnodes = self.frustum_nodes.load(Relaxed);
        let rnodes = self.ray_nodes.load(Relaxed);
        // Primary vs secondary node COST, which the ray counts do not predict.
        // Empty when nothing traced, so an unchanged line stays unchanged.
        let rn_prim = self.ray_nodes_prim.load(Relaxed);
        let rsplit = if rnodes > 0 {
            format!(
                " (prim {rn_prim} = {:.0}%, sec {})",
                100.0 * rn_prim as f64 / rnodes as f64,
                rnodes - rn_prim
            )
        } else {
            String::new()
        };
        let prim = self.primary_rays.load(Relaxed);
        let sec = self.secondary_rays.load(Relaxed);
        let sky = self.sky_pixels.load(Relaxed);
        let blocked = self.blocked_queries.load(Relaxed);
        let tiles = self.tiles.load(Relaxed);
        let src = self.skip_ratio_count.load(Relaxed);
        let skip = if src > 0 {
            self.skip_ratio_micro.load(Relaxed) as f64 / src as f64 / 1e6
        } else {
            0.0
        };
        let sky_tiles = self.sky_tiles.load(Relaxed);
        let refines = fq.saturating_sub(sky_tiles);
        let cut_mean = if refines > 0 {
            self.cut_len_sum.load(Relaxed) as f64 / refines as f64
        } else {
            0.0
        };
        let ovf = self.cut_overflows.load(Relaxed);
        let coarse = self.coarse_pixels.load(Relaxed);
        let csmp = self.coarse_samples.load(Relaxed);
        let tseeds = self.temporal_seeds.load(Relaxed);
        let tsky = self.temporal_sky_tiles.load(Relaxed);
        let ttests = self.temporal_tests.load(Relaxed);
        // Bounce-integrator segments only appear when those paths ran.
        let hp = self.hemi_points.load(Relaxed);
        let hemi = if hp > 0 {
            format!(
                " | hemi: pts {hp} q {} empty {} rays {} nodes {}",
                self.hemi_queries.load(Relaxed),
                self.hemi_cells_empty.load(Relaxed),
                self.hemi_leaf_rays.load(Relaxed),
                self.hemi_nodes.load(Relaxed),
            )
        } else {
            String::new()
        };
        let shg = self.hemi_share_groups.load(Relaxed);
        let share = if shg > 0 {
            format!(
                " | hemi-share: groups {shg} pts {} fallback {}",
                self.hemi_share_points.load(Relaxed),
                self.hemi_share_fallback.load(Relaxed),
            )
        } else {
            String::new()
        };
        let adopts = self.temporal_cut_adopts.load(Relaxed);
        let adopt = if adopts > 0 {
            format!(
                " | adopt: {adopts} sky {} requery {} arena-full {}",
                self.temporal_adopt_sky.load(Relaxed),
                self.temporal_adopt_requery.load(Relaxed),
                self.temporal_cut_arena_full.load(Relaxed),
            )
        } else {
            String::new()
        };
        let rhits = self.temporal_ring_hits.load(Relaxed);
        let tring = if rhits > 0 {
            format!(
                " | tring: hits {rhits} mean-age {:.1}",
                self.temporal_ring_age_sum.load(Relaxed) as f64 / rhits as f64
            )
        } else {
            String::new()
        };
        let rl = self.replay_leaf_tiles.load(Relaxed);
        let rs = self.replay_sky_tiles.load(Relaxed);
        let replay = if rl + rs > 0 {
            format!(" | replay: leaves {rl} sky {rs}")
        } else {
            String::new()
        };
        let ac = self.adapt_coarse.load(Relaxed);
        let ab = self.adapt_base.load(Relaxed);
        let ah = self.adapt_hot.load(Relaxed);
        let adapt = if ac + ab + ah > 0 {
            format!(
                " | adapt: {ac}c/{ab}b/{ah}h (+{} edge-px) topup {} saved {} penumbra {}",
                self.adapt_partial_px.load(Relaxed),
                self.adapt_topup.load(Relaxed),
                self.adapt_rays_saved.load(Relaxed),
                self.adapt_penumbra.load(Relaxed),
            )
        } else {
            String::new()
        };
        // FR_SWAY_TRI probe (bvh::TRI_TESTS) — CUMULATIVE process totals, not
        // per-frame (the probe deliberately bypasses the LocalStats batching;
        // see its doc in bvh.rs). Zero and absent when unarmed.
        let tt = crate::bvh::TRI_TESTS.load(Relaxed);
        let tri = if tt > 0 {
            format!(
                " | tri-tests {tt} (shifted {}, cumulative)",
                crate::bvh::TRI_TESTS_SHIFTED.load(Relaxed)
            )
        } else {
            String::new()
        };
        let dpx = self.defer_px.load(Relaxed);
        let dmx = self.defer_mixed.load(Relaxed);
        let defer = if dpx + dmx > 0 {
            let df = self.defer_flushes.load(Relaxed);
            format!(
                " | defer: px {dpx} flushes {df} (mean {:.0}) mixed {dmx}",
                if df > 0 { dpx as f64 / df as f64 } else { 0.0 },
            )
        } else {
            String::new()
        };
        // Emissive per-tile light cull — absent on unarmed/emissive-free
        // sessions (the defer/adapt conditional-segment precedent).
        let ect = self.el_cull_tiles.load(Relaxed);
        let elcull = if ect > 0 {
            let ecc = self.el_cull_culled.load(Relaxed);
            format!(" | el-cull: {ect}t culled {ecc} (mean {:.1}/t)", ecc as f64 / ect as f64)
        } else {
            String::new()
        };
        // GI gather rays — absent under --no-rtgi / fb frames (the
        // conditional-segment precedent). The level-1 term appends ONLY when a
        // stochastic-or-deeper rung fired, so rungs 0 and 1 print verbatim what
        // they printed before the ladder existed.
        let grays = self.rtgi_rays.load(Relaxed);
        let grays2 = self.rtgi_rays2.load(Relaxed);
        let rtgi = if grays > 0 || grays2 > 0 {
            let l2 = if grays2 > 0 { format!("+{grays2}") } else { String::new() };
            format!(" | rtgi: rays {grays}{l2}")
        } else {
            String::new()
        };
        format!(
            "tiles {tiles} | fr-queries {fq} (blocked {blocked}) | cut mean {cut_mean:.1} (ovf {ovf}) | nodes: frustum {fnodes} + ray {rnodes}{rsplit} = {} | rays: {prim} prim + {sec} sec | sky-px (0 rays) {sky} | coarse-px {coarse} (smp {csmp}) | temporal: seeds {tseeds} sky {tsky} cells {ttests} | mean t_start/t_hit {skip:.2}{adopt}{tring}{replay}{hemi}{share}{adapt}{defer}{tri}{elcull}{rtgi}",
            fnodes + rnodes
        )
    }
}
