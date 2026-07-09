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
    /// Tiles flat-filled at the depth cap, and their pixels.
    pub coarse_tiles: AtomicU64,
    pub coarse_pixels: AtomicU64,
    /// Tiles whose t_start was raised by a previous-frame temporal seed.
    pub temporal_seeds: AtomicU64,
    /// Tiles sky-filled straight from the temporal cache (zero BVH work).
    pub temporal_sky_tiles: AtomicU64,
    /// Containment candidates evaluated by the temporal walk (its cost).
    pub temporal_tests: AtomicU64,
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
    pub temporal_seeds: u64,
    pub temporal_sky_tiles: u64,
    pub temporal_tests: u64,
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
        self.temporal_seeds.store(0, Relaxed);
        self.temporal_sky_tiles.store(0, Relaxed);
        self.temporal_tests.store(0, Relaxed);
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
        if l.temporal_seeds > 0 {
            self.temporal_seeds.fetch_add(l.temporal_seeds, Relaxed);
        }
        if l.temporal_sky_tiles > 0 {
            self.temporal_sky_tiles.fetch_add(l.temporal_sky_tiles, Relaxed);
        }
        if l.temporal_tests > 0 {
            self.temporal_tests.fetch_add(l.temporal_tests, Relaxed);
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
        let tseeds = self.temporal_seeds.load(Relaxed);
        let tsky = self.temporal_sky_tiles.load(Relaxed);
        let ttests = self.temporal_tests.load(Relaxed);
        format!(
            "tiles {tiles} | fr-queries {fq} (blocked {blocked}) | cut mean {cut_mean:.1} (ovf {ovf}) | nodes: frustum {fnodes} + ray {rnodes} = {} | rays: {prim} prim + {sec} sec | sky-px (0 rays) {sky} | coarse-px {coarse} | temporal: seeds {tseeds} sky {tsky} tests {ttests} | mean t_start/t_hit {skip:.2}",
            fnodes + rnodes
        )
    }
}
