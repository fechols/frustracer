use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Per-frame render counters. Shared across rayon tasks; hot paths batch into
/// `LocalStats` and flush once per tile so the counter cache lines don't serialize.
#[derive(Default)]
pub struct Stats {
    pub frustum_queries: AtomicU64,
    pub frustum_nodes: AtomicU64,
    pub ray_nodes: AtomicU64,
    pub primary_rays: AtomicU64,
    pub secondary_rays: AtomicU64,
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
    pub primary_rays: u64,
    pub secondary_rays: u64,
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
        self.primary_rays += o.primary_rays;
        self.secondary_rays += o.secondary_rays;
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
        self.primary_rays.store(0, Relaxed);
        self.secondary_rays.store(0, Relaxed);
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
        if l.primary_rays > 0 {
            self.primary_rays.fetch_add(l.primary_rays, Relaxed);
        }
        if l.secondary_rays > 0 {
            self.secondary_rays.fetch_add(l.secondary_rays, Relaxed);
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
        format!(
            "tiles {tiles} | fr-queries {fq} (blocked {blocked}) | cut mean {cut_mean:.1} (ovf {ovf}) | nodes: frustum {fnodes} + ray {rnodes} = {} | rays: {prim} prim + {sec} sec | sky-px (0 rays) {sky} | coarse-px {coarse} (smp {csmp}) | temporal: seeds {tseeds} sky {tsky} cells {ttests} | mean t_start/t_hit {skip:.2}{adopt}{tring}{replay}{hemi}{share}{adapt}{defer}",
            fnodes + rnodes
        )
    }
}
