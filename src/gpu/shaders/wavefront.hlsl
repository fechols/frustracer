// The wavefront quadtree: seed -> D x (prep -> level) -> sky-fill, with the
// leaf pass in leaf.hlsl. One statically recorded command list; every
// decision after the root seed is made by GPU-written counters feeding
// ExecuteIndirect. Requires trace_common.hlsli + queues.hlsli +
// frustum.hlsli pasted first.

// --sw-rays under FTREE: the wide-tree slot->binary-node map (ftree.rs
// FNode.bnode, which the quantized QFNode wire format deliberately drops),
// flat nodes x 8, bound at t1 for the level ladder ONLY — t1 (tri_idx) is
// dead in every ladder kernel, and record_wavefront rebinds the real
// tri_idx before the leaf/sky dispatches (the SKY_UNIT register-remeaning
// idiom). Read exclusively by the leaf-cut translation in level_finish.
// SW_RAYS_LEAF (not SW_RAYS): under --sw-rays --no-cut-rays the leaf never
// consumes a cut, so neither the map nor the translation compiles.
#if defined(SW_RAYS_LEAF) && defined(FTREE)
StructuredBuffer<uint> ft_bnode : register(t1);
#endif

// --- seed: reset counters + cut pool bump, enqueue the whole-screen root ---
//
// push0 = 1 suppresses the root ENQUEUE only (the counter reset and the
// degenerate-window leaf still run, and the work-graph arm depends on both).
// A graph takes its root as CPU input, so a queued one would never be
// consumed and CTR_TILE_A would sit at 1 for the whole frame — a dangling
// tile record that `--check-gpu`'s depth accounting reports, and which its
// parity-selected counter would hide at odd depth_full and expose at even.

[numthreads(1, 1, 1)]
void cs_seed(uint3 id : SV_DispatchThreadID) {
    for (uint i = 0; i < CTR_COUNT; ++i) counters[i] = 0;
    if (rw <= LEAF_TILE && rh <= LEAF_TILE) {
        // Degenerate window: the root IS a leaf (trace_tile's first check).
        LeafRec lf;
        lf.xy0 = pack_xy(0, 0);
        lf.xy1 = pack_xy(rw, rh);
        lf.t_start = 0.0;
        lf.depth = 0;
        lf.frontier = frontier_root();
        qleaf[0] = lf;
        counters[CTR_LEAF] = 1;
        return;
    }
    if (push0 != 0u) return; // the caller supplies its own root (work graph)
    TileRec root;
    root.xy0 = pack_xy(0, 0);
    root.xy1 = pack_xy(rw, rh);
    root.t_start = 0.0;
    root.cut_slot = ROOT_CUT_SLOT; // the root cut [0]
    root.meta = 1u; // cut_len 1, depth 0
    root.path = 0u;
    qin[0] = root;
    counters[CTR_TILE_A] = 1;
}

// --- replay seed: a bit-equal-basis frame skips seed + the whole ladder -----
// The previous producing frame's TERMINAL structure — qleaf/qsky/cut_pool and
// their counts CTR_LEAF/CTR_SKY/CTR_CUT — is byte-intact between producing
// frames (only cs_seed + the ladder ever write it; the leaf/sky/hemi passes
// only read it, hemi rebinds transient buffers at u7/u9, and the reference/
// feed/nppd units declare no counters or queues). So a replay frame re-runs
// ONLY the terminal fills. Zero every OTHER counter — the tile counts, the
// stats, the verify slots, and CTR_HEMI_PT, which the fb leaf pass is about to
// append into again — but KEEP the three the fills consume. (The keep-set is
// the cs_seed_probes pattern; the caller proves basis bit-equality.)
[numthreads(1, 1, 1)]
void cs_seed_replay(uint3 id : SV_DispatchThreadID) {
    for (uint i = 0; i < CTR_COUNT; ++i) {
        bool keep = i == CTR_LEAF || i == CTR_SKY || i == CTR_CUT || i == CTR_SKY_PX;
        if (!keep) counters[i] = 0;
    }
}

// --- prep-args: counter -> DispatchIndirect groups (2D split past 32768) ---
// push0 = counter to read, push1 = counter to zero (0xffffffff = none;
// zeroing the OUT queue's counter here is what makes ping-pong reuse safe),
// push2 = work items per group, push3 = args slot.

[numthreads(1, 1, 1)]
void cs_prep(uint3 id : SV_DispatchThreadID) {
    uint n = counters[push0];
    uint groups = (n + push2 - 1) / push2;
    uint gx = min(groups, 32768u);
    uint gy = (groups + 32767u) / 32768u;
    args[push3] = uint3(gx, groups > 0 ? gy : 0, 1);
    if (push1 != 0xffffffffu) counters[push1] = 0;
}

// --- prep-args, MULTIPLYING flavor: groups = counters[push0] * push2 -------
// cs_prep divides (N work items -> N/push2 groups); the sky fill needs the
// inverse, because SKY_SPLIT groups cooperate on each record. Same 2D split
// past 32768 and the same never-zero-dispatch-on-empty rule.

[numthreads(1, 1, 1)]
void cs_prep_mul(uint3 id : SV_DispatchThreadID) {
    uint groups = counters[push0] * push2;
    args[push3] = uint3(min(groups, 32768u), groups > 0 ? (groups + 32767u) / 32768u : 0u, 1);
}

// --- check builds: flood info with the exactly-once coverage sentinel ---

[numthreads(256, 1, 1)]
void cs_clear_info(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint i = flat_group(gid) * 256u + gtid.x;
    if (i < rw * rh) info[i] = 0xffffffffu;
}

// --- hemi support: H clear, batch prep, probe seeding ----------------------

[numthreads(256, 1, 1)]
void cs_clear_h(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint i = flat_group(gid) * 256u + gtid.x;
    if (i < rw * rh * 4u) hbuf[i] = 0u;
}

// Batch prep: args for the hemi root pass over points
// [push1, push1 + hemi_batch) of counters[push0], push2 items per group,
// args slot push3 — and reset the batch-scoped hemi counters (queues and cut
// pool are batch-transient; that reset is what bounds their memory).
[numthreads(1, 1, 1)]
void cs_prep_batch(uint3 id : SV_DispatchThreadID) {
    uint total = counters[push0];
    uint n = total > push1 ? min(total - push1, hemi_batch) : 0u;
    uint groups = (n + push2 - 1) / push2;
    args[push3] = uint3(min(groups, 32768u), groups > 0 ? (groups + 32767u) / 32768u : 0u, 1);
    counters[CTR_HEMI_A] = 0;
    counters[CTR_HEMI_B] = 0;
    counters[CTR_HEMI_LEAF] = 0;
    counters[CTR_HEMI_CUT] = 0;
}

// --check-gpu probe path: claim push0 pre-uploaded HemiPointRecs (the
// CPU-generated probe set). push1 = full clear (first seed); accumulate
// passes (push1 == 0) keep the verify counters and cross-seed stats so the
// multi-seed suite gates EVERY seed's rays, and zero only the batch-scoped
// transients (which cs_prep_batch would reset anyway).
[numthreads(1, 1, 1)]
void cs_seed_probes(uint3 id : SV_DispatchThreadID) {
    for (uint i = 0; i < CTR_COUNT; ++i) {
        bool keep = i == CTR_V_FALSE_EMPTY || i == CTR_V_TMIN ||
                    i == CTR_HEMI_EMPTY || i == CTR_HEMI_RAYS ||
                    i == CTR_OVERFLOW || i == CTR_CUT_FALLBACK;
        if (push1 != 0u || !keep) counters[i] = 0;
    }
    counters[CTR_HEMI_PT] = push0;
}

// --- WAVE-COOPERATIVE bound query: ONE TILE PER GROUP ---------------------
// The serial `bound_query` gives each tile one thread and a private DFS. That
// is right where tiles are many and cuts are tight, and badly wrong at shallow
// depths: level 0 is ONE tile, so the whole GPU runs ONE lane descending an
// enormous slice of the BVH (a level-0 frustum is the whole screen and its cut
// is the root), and levels 0-4 together are <= 256 tiles yet 67% of the
// ladder's cost. Measured (B70, --spin path 1080p, --stress 5000): the prep
// dispatches and args transitions across all 8 levels total 0.011 ms against
// the kernels' 1.817 — the ladder is not dispatch-bound, it is UNDER-OCCUPIED.
//
// So for shallow levels the group cooperates on one tile: 32 lanes share a
// breadth-first frontier instead of one lane walking a depth-first stack.
//
// Two deliberate differences from the serial path, both sound:
//   * BFS, not DFS. `best` is order-independent (the serial port already says
//     so), and the min is exact either way — only the PRUNING order changes,
//     so a round may visit nodes an ordered descent would have culled. That is
//     the price of 32x parallelism, and it is why this kernel is used only
//     where lanes would otherwise sit idle.
//   * No FAR->NEAR sibling ordering (ftree.hlsli's selection scan). Its whole
//     purpose is tightening `best` early within one lane's DFS; in a frontier
//     every survivor is processed next round regardless, so it would be pure
//     cost. `ft_slot`'s per-slot math is reused verbatim, so every conservative
//     slack is still the same code.
//
// Frontier storage ALIASES frustum.hlsli's per-lane stack slab (32*LANE_STACK
// u32 = 8 KB). Legal for exactly the reason the serial path already relies on:
// this phase and refine_cut's never overlap in time — here refine_cut runs
// after the query returns, on lane 0 only. Zero extra groupshared, which
// matters because groupshared is what caps resident groups.
#define WQ_CAP (16u * LANE_STACK)   // two frontiers inside the shared slab
groupshared uint gw_len[2];
groupshared uint gw_best;

// WAVE-AGGREGATED SHARED-MEMORY TRAFFIC (revert: FR_ABL=nowave).
//
// Both primitives below used to be one LDS atomic PER LANE PER CANDIDATE. In
// the FTREE round that is up to 8 per lane (the unrolled slot loop), so a
// 32-lane group could issue ~256 atomics to ONE address per BFS round. Intel's
// RT developer guide (v4, p.26) prescribes exactly this fix — "keep data within
// a wave and use wave intrinsics instead of barriers", accumulating in
// registers and combining with a wave op — and it matters more on Intel than
// elsewhere, because groupshared is carved out of the same L1 that services the
// RT unit. Measured on an Arc Pro B70: `cs_level_wide`'s 32-thread group is
// EXACTLY ONE WAVE (WaveGetLaneCount() == 32 at every group width we dispatch),
// so this collapses to a single atomic.
//
// WHY THIS IS NOT A BEHAVIOUR CHANGE. Both helpers are called at the SAME
// program points as the per-lane atomics they replace, and lanes of one wave
// reach a given point together, so the aggregate is over exactly the lanes that
// would have raced there anyway. `best` is an order-independent min and the
// queue is a SET whose slot assignment order was already unspecified (the old
// atomics raced), so only cross-WAVE interleaving changes — and on a group that
// is one wave, nothing does. Counted frustum nodes may still move where a group
// spans several waves; the image may not. Same contract as WIDE_LEVELS.
//
// CALL THEM UNCONDITIONALLY. Each takes a participation flag rather than
// sitting inside an `if`, so the wave op is never reached by a subset of lanes
// that a later edit might change. Non-participants carry an identity.

// float-min via InterlockedMin on the IEEE bit pattern: exact for NON-NEGATIVE
// finite floats, whose bit patterns order exactly as their values. Every
// candidate here is `max(distance, t_start) >= 0` or the FLT_MAX seed.
// Bit-domain core. 0xFFFFFFFF is the "no contribution" identity — it is above
// asuint(FLT_MAX), and no real candidate can collide with it because every one
// is a non-negative finite float (0xFFFFFFFF is -NaN). Callers that accumulate
// across several candidates use this directly rather than round-tripping the
// accumulator through asfloat(), which would put a NaN bit pattern in a float
// register where hardware is permitted to canonicalise it.
void gw_min_bits(uint v) {
#if defined(ABL_NO_WAVE_OPS)
    if (v != 0xFFFFFFFFu) {
        uint prev;
        InterlockedMin(gw_best, v, prev);
    }
#else
    uint m = WaveActiveMin(v);
    if (WaveIsFirstLane() && m != 0xFFFFFFFFu) {
        uint prev;
        InterlockedMin(gw_best, m, prev);
    }
#endif
}

void gw_min_if(bool take, float d) { gw_min_bits(take ? asuint(d) : 0xFFFFFFFFu); }

// Reserve `n` frontier slots for this lane (n == 0 participates but takes
// nothing). Returns the lane's base offset.
//
// The PARITY INVARIANT the binary path depends on survives: its callers pass
// n == 2, so every prefix sum is even and the wave's own base comes from an
// even total added to an even counter — accepted offsets stay even exactly as
// they did when each lane added 2 on its own.
uint gw_alloc(uint slot, uint n) {
#if defined(ABL_NO_WAVE_OPS)
    uint w = 0;
    if (n != 0) InterlockedAdd(gw_len[slot], n, w);
    return w;
#else
    uint pre = WavePrefixSum(n);
    uint sum = WaveActiveSum(n);
    uint base = 0;
    if (WaveIsFirstLane() && sum != 0) InterlockedAdd(gw_len[slot], sum, base);
    // Only the first lane's `base` is meaningful; broadcast it to the wave.
    return WaveReadLaneFirst(base) + pre;
#endif
}

float bound_query_wave(TF f, float t_start, float t_limit, uint cut_slot, uint cut_len, uint tid) {
    if (tid == 0) {
        gw_len[0] = 0;
        gw_len[1] = 0;
        gw_best = asuint(t_limit);
    }
    GroupMemoryBarrierWithGroupSync();

    // Seed the frontier from the inherited cut, lane-strided.
#ifdef FTREE
    uint n0 = cut_slot == ROOT_CUT_SLOT ? 8u : cut_len;
    for (uint r = tid; r < n0; r += 32u) {
        uint e = cut_slot == ROOT_CUT_SLOT ? r : cut_pool[cut_slot * 64u + r];
        FtNode nd = ft_nodes[e >> 3];
        // `d` is 0-initialised by ft_slot on every early-out, so it is safe to
        // read on the non-participating path below.
        float d;
        bool live = ft_slot(f, nd, e & 7u, t_start, d) && d < asfloat(gw_best);
        uint c = nd.child[e & 7u];
        bool push = live && c != FT_INVALID;      // internal: goes on the frontier
        uint w = gw_alloc(0, push ? 1u : 0u);
        bool over = push && w >= WQ_CAP;
        if (push && !over) g_stack[w] = c;
        // Terminal (a leaf's box) or frontier-full: fold in as a coarse bound.
        gw_min_if(live && (!push || over), d);
    }
#else
    for (uint r = tid; r < cut_len; r += 32u) {
        uint w = gw_alloc(0, 1u);
        if (w < WQ_CAP) g_stack[w] = cut_node(cut_slot, r);
    }
#endif

    uint cur = 0;
    [loop] while (true) {
        GroupMemoryBarrierWithGroupSync();
        // Clamped: the overflow arms below bump the counter past capacity on
        // purpose (having already folded that node in as a coarse bound), so
        // the length is an intent, not a fill level.
        uint n = min(gw_len[cur], WQ_CAP);
        if (n == 0) break; // group-uniform: every lane reads the same counter
        uint bin = cur * WQ_CAP;
        uint bout = (1u - cur) * WQ_CAP;
        for (uint i = tid; i < n; i += 32u) {
            uint idx = g_stack[bin + i];
#ifdef FTREE
            FtNode nd = ft_nodes[idx];
            // TWO PASSES, so a node costs ONE aggregated allocation and ONE
            // aggregated min instead of eight of each. Pass 1 classifies the 8
            // slots into a bitmask and a per-lane running min; pass 2 writes
            // the survivors at base + rank.
            //
            // `kid[]` and `dist[]` are indexed only by the UNROLLED loop
            // counter, so every access has a compile-time index and the arrays
            // stay in registers. That is the same rule ft_expand's selection
            // scan follows — what it measured at +58% was a DYNAMIC shuffle
            // index demoting them to scratch, which is why `rank` below is a
            // popcount of a mask rather than a running counter into an array.
            uint push_mask = 0u;
            uint kid[8];
            float dist[8];
            uint bl = 0xFFFFFFFFu;          // identity: above asuint(FLT_MAX)
            [unroll] for (uint s = 0; s < 8; ++s) {
                float d;
                bool live = ft_slot(f, nd, s, t_start, d) && d < asfloat(gw_best);
                uint c = nd.child[s];
                bool push = live && c != FT_INVALID;
                kid[s] = c;
                dist[s] = d;
                if (push) push_mask |= (1u << s);
                // Terminal (a leaf's box): fold in now. An overflowing survivor
                // is folded in by pass 2 instead.
                if (live && !push) bl = min(bl, asuint(d));
            }
            uint base = gw_alloc(1u - cur, countbits(push_mask));
            [unroll] for (uint s2 = 0; s2 < 8; ++s2) {
                if ((push_mask & (1u << s2)) == 0) continue;
                // Rank among THIS lane's survivors — a popcount of the bits
                // below s2, so no array is indexed dynamically.
                uint w = base + countbits(push_mask & ((1u << s2) - 1u));
                if (w < WQ_CAP) g_stack[bout + w] = kid[s2];
                else bl = min(bl, asuint(dist[s2]));
            }
            gw_min_bits(bl);
#else
            BvhNode node = bvh_nodes[idx];
            // Flattened from three `continue`s so the wave ops below are
            // reached by every lane in this iteration (see the helper header).
            bool live = !aabb_outside(f, node.mn, node.mx)
                     && point_aabb_max_dist(f.origin, node.mn, node.mx) > t_start;
            float d = live ? max(point_aabb_dist(f.origin, node.mn, node.mx), t_start) : 0.0;
            live = live && d < asfloat(gw_best);
            bool inner = live && node.count == 0;
            // Adds are all +2 from a zeroed base, so accepted w are even and
            // slots [0, WQ_CAP) are fully written whenever the clamp bites.
            uint w = gw_alloc(1u - cur, inner ? 2u : 0u);
            bool over = inner && (w + 1u >= WQ_CAP);
            if (inner && !over) {
                g_stack[bout + w] = node.left_first;
                g_stack[bout + w + 1u] = node.left_first + 1u;
            }
            // A leaf, or frontier full: take this node's own distance as if it
            // were a leaf — a valid lower bound for everything inside it. The
            // serial path's stack-pressure fallback, verbatim.
            gw_min_if(live && (!inner || over), d);
#endif
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid == 0) gw_len[cur] = 0;
        cur = 1u - cur;
    }
    // Every lane is past its last g_stack write before this returns, so lane 0
    // may reuse the slab for refine_cut. (The break above sits directly after a
    // group sync, and both barriers are reached by the whole group — never in
    // divergent control flow.)
    return asfloat(gw_best);
}

// --- the level kernel: render.rs::tile_step -------------------------------
// push0 = in counter (tile A or B), push1 = out counter (the other).
//
// The tail (sky / advance / refine / enqueue the 4 quadrants) is shared by the
// per-thread and per-group flavors so the two can never drift apart; only the
// bound query differs, and `lane` says which g_stack slab refine_cut owns.

// ctr_add / ctr_bump (the global-counter twins of gw_alloc/gw_min_if) live in
// ctr.hlsli beside `counters`, because leaf.hlsl aggregates its per-pixel
// hemi-point counter with the same primitives. This composes with — and does
// not replace — the HOMOGENEOUS-BATCH treatment below, which already folds a
// tile's own 4 quadrants into 1: together, 4 x 32 becomes 1. Called from
// `cs_level_wide` they run with a single active lane (level_finish is lane 0's
// job there), which is correct and simply buys nothing.
#if defined(WORKGRAPH)
// WORK-GRAPH ARM (FR_WORKGRAPH=1). Everything except the TILE children is
// unchanged: sky and leaf records still go to their UAV queues through the same
// counters, and the cut pool is still bump-allocated here — only `qout`/`push1`
// have no meaning, because a graph node hands its children to the scheduler
// instead of to a ping-pong queue. (The ping-pong is precisely what a graph
// breaks: it assumes levels SERIALISE, and the whole point of the graph is that
// they do not.)
//
// Children come back as a 4-slot array plus a live MASK rather than a packed
// list, so every write is `wg_kids[c]` under an unrolled loop counter — a
// compile-time index. A running `wg_kids[n++]` would be a dynamic index and
// would demote the array to scratch, the +58% ft_expand lesson. The caller
// compacts with a popcount rank.
void level_finish(TileRec rec, uint2 p0, uint2 p1, uint depth, uint cut_len, TF f,
                  float best, uint lane, out TileRec wg_kids[4], out uint wg_mask) {
    wg_mask = 0u;
    // Fully initialise: HLSL `out` parameters are undefined on paths that do
    // not write them, and the caller reads slots by mask.
    [unroll] for (uint wi = 0; wi < 4; ++wi) {
        TileRec z;
        z.xy0 = 0; z.xy1 = 0; z.t_start = 0.0;
        z.cut_slot = ROOT_CUT_SLOT; z.meta = 0; z.path = 0;
        wg_kids[wi] = z;
    }
    uint s;
#else
void level_finish(TileRec rec, uint2 p0, uint2 p1, uint depth, uint cut_len, TF f,
                  float best, uint lane) {
    uint s;
#endif
    if (best == FLT_MAX) {
        // Sky: the whole frustum (beyond the inherited ball, which the
        // ancestor claim covers) is empty.
        // Stat: the rect's pixel area — the empty-space proof's product
        // (pixels that will trace ZERO rays). Wave-uniform slot, per-lane n.
        ctr_add(CTR_SKY_PX, (p1.x - p0.x) * (p1.y - p0.y));
        s = ctr_add(CTR_SKY, 1u);
        if (s < cap_sky) {
            SkyRec sky;
            sky.xy0 = rec.xy0;
            sky.xy1 = rec.xy1;
            sky.depth = depth;
            sky._pad = 0;
            qsky[s] = sky;
        } else {
            InterlockedAdd(counters[CTR_OVERFLOW], 1, s);
        }
        return;
    }

    // frustum.rs::advance_tc — the shared advance/slack rule.
    bool advanced = best > rec.t_start + max(rec.t_start * 1e-4, SCENE_EPS);
    float tc = advanced ? max(best * (1.0 - 1e-4), rec.t_start) : rec.t_start;
    // Blocked at the inherited distance — still subdivide (children's smaller
    // frustums may exclude the blocker; that is how sky emerges). Hoisted out
    // of the `if` so the whole wave reaches the reduction.
    ctr_bump(CTR_BLOCKED, !advanced);
    ctr_bump(CTR_SPLIT, true);

#if !defined(SW_RAYS_LEAF) && !defined(ABL_KEEP_TERMINAL_CUT)
    // The final split has no TileRec children, so its refined cut has no
    // consumer: hardware RayQuery accepts only TMin (tc), and the software
    // root-seeded arm deliberately ignores LeafRec's cut fields too. Emit the
    // terminal children before allocating/writing a dead cut-pool slot.
    //
    // ceil(parent_extent / 2) is the largest child extent. This also handles
    // odd and one-pixel dimensions; empty quadrants retain the normal guard.
    uint pw = p1.x - p0.x;
    uint ph = p1.y - p0.y;
    if ((pw + 1u) / 2u <= LEAF_TILE && (ph + 1u) / 2u <= LEAF_TILE) {
        uint lxm = p0.x + pw / 2u;
        uint lym = p0.y + ph / 2u;
        uint ld = depth + 1u;
        uint lx0[4] = { p0.x, lxm, p0.x, lxm };
        uint ly0[4] = { p0.y, p0.y, lym, lym };
        uint lx1[4] = { lxm, p1.x, lxm, p1.x };
        uint ly1[4] = { lym, lym, p1.y, p1.y };
#if !defined(ABL_NO_QUEUE_BATCH)
        // B70 HOMOGENEOUS-BATCH BEGIN: independently revertible treatment.
        // One reservation preserves the same counter total and per-child
        // overflow reporting while avoiding four contended global atomics.
        uint live_count = 0u;
        [unroll] for (uint lc = 0; lc < 4; ++lc) {
            if (lx0[lc] != lx1[lc] && ly0[lc] != ly1[lc]) live_count++;
        }
        uint leaf_base = ctr_add(CTR_LEAF, live_count);
        uint live_rank = 0u;
        [unroll] for (uint lc = 0; lc < 4; ++lc) {
            if (lx0[lc] == lx1[lc] || ly0[lc] == ly1[lc]) continue;
            uint leaf_at = leaf_base + live_rank++;
            if (leaf_at < cap_leaf) {
                LeafRec lf;
                lf.xy0 = pack_xy(lx0[lc], ly0[lc]);
                lf.xy1 = pack_xy(lx1[lc], ly1[lc]);
                lf.t_start = tc;
                lf.depth = ld;
                // Placeholders only: this compile-time arm's leaf rays do not
                // consume cuts. ROOT_CUT_SLOT/1 remains conservative if the
                // record is inspected or its layout is shared by replay.
                lf.frontier = frontier_root();
                qleaf[leaf_at] = lf;
            } else {
                InterlockedAdd(counters[CTR_OVERFLOW], 1, s);
            }
        }
        // B70 HOMOGENEOUS-BATCH END.
#else
        [unroll] for (uint lc = 0; lc < 4; ++lc) {
            if (lx0[lc] == lx1[lc] || ly0[lc] == ly1[lc]) continue;
            InterlockedAdd(counters[CTR_LEAF], 1, s);
            if (s < cap_leaf) {
                LeafRec lf;
                lf.xy0 = pack_xy(lx0[lc], ly0[lc]);
                lf.xy1 = pack_xy(lx1[lc], ly1[lc]);
                lf.t_start = tc;
                lf.depth = ld;
                lf.frontier = frontier_root();
                qleaf[s] = lf;
            } else {
                InterlockedAdd(counters[CTR_OVERFLOW], 1, s);
            }
        }
#endif
        return;
    }
#endif

    // Refine the cut into a fresh pool slot; on pool exhaustion the children
    // inherit the PARENT's slot — an ancestor cut is valid for any descendant
    // frustum (coarse, never wrong) — the refine_cut coarse-emit analog.
    uint out_slot = ctr_add(CTR_CUT, 1u);
    uint out_len = 0;
    if (out_slot < cap_cut) {
        out_len = refine_cut(f, tc, FLT_MAX, rec.cut_slot, cut_len, out_slot, lane);
    } else {
        InterlockedAdd(counters[CTR_CUT_FALLBACK], 1, s);
    }
    if (out_len == 0) {
        // Pool exhausted, or the structurally-impossible empty refine (the
        // CPU debug_asserts it): fall back to the parent cut, never drop.
        out_slot = rec.cut_slot;
        out_len = cut_len;
    }

    // Enqueue the 4 quadrants (integer midpoint split — must match
    // trace_tile / temporal::rect_for_path exactly; TL=0 TR=1 BL=2 BR=3).
    uint xm = p0.x + (p1.x - p0.x) / 2u;
    uint ym = p0.y + (p1.y - p0.y) / 2u;
    uint d = depth + 1u;
    uint cpath = rec.path << 2;
    uint cx0[4] = { p0.x, xm, p0.x, xm };
    uint cy0[4] = { p0.y, p0.y, ym, ym };
    uint cx1[4] = { xm, p1.x, xm, p1.x };
    uint cy1[4] = { ym, ym, p1.y, p1.y };

    // The cut a leaf child consumes: the SAME (out_slot, out_len) its sibling
    // TileRecs inherit — the CPU's "leaf tiles use the inherited cut without
    // re-culling". Under --sw-rays + FTREE the refined entries are wide
    // slot-refs, but rays seed from BINARY node ids, so translate ONCE per
    // split (only when a leaf child exists) into a fresh pool slot via
    // ft_bnode — a pure relabeling: every occupied slot IS a binary node
    // (ftree self-test's slot audit). ROOT_CUT_SLOT passes through (rays
    // then traverse from the binary root) and pool exhaustion falls back to
    // it — coarse, never wrong (the refine fallback's own shape).
    uint leaf_slot = out_slot;
    uint leaf_len = out_len;
#if defined(SW_RAYS_LEAF) && defined(FTREE)
    if (out_slot != ROOT_CUT_SLOT) {
        bool any_leaf = false;
        [unroll] for (uint lc = 0; lc < 4; ++lc) {
            uint lw = cx1[lc] - cx0[lc];
            uint lh = cy1[lc] - cy0[lc];
            if (lw != 0 && lh != 0 && lw <= LEAF_TILE && lh <= LEAF_TILE)
                any_leaf = true;
        }
        if (any_leaf) {
            uint ts;
            InterlockedAdd(counters[CTR_CUT], 1, ts);
            if (ts < cap_cut) {
                for (uint i = 0; i < leaf_len; ++i) {
                    uint e = cut_pool[out_slot * 64u + i];
                    cut_pool[ts * 64u + i] = ft_bnode[(e >> 3u) * 8u + (e & 7u)];
                }
                leaf_slot = ts;
            } else {
                InterlockedAdd(counters[CTR_CUT_FALLBACK], 1, ts);
                leaf_slot = ROOT_CUT_SLOT;
            }
        }
    }
#endif

#if !defined(ABL_NO_QUEUE_BATCH)
    // B70 HOMOGENEOUS-BATCH BEGIN: independently revertible treatment.
    // Regular quadtree levels emit either four internal children or four
    // leaves. Reserve their contiguous queue range with one atomic; unusual
    // odd/aspect-ratio splits that mix the two retain the original loop below.
    uint live_children = 0u;
    uint leaf_children = 0u;
    [unroll] for (uint bc = 0; bc < 4; ++bc) {
        uint bw = cx1[bc] - cx0[bc];
        uint bh = cy1[bc] - cy0[bc];
        if (bw == 0u || bh == 0u) continue;
        live_children++;
        if (bw <= LEAF_TILE && bh <= LEAF_TILE) leaf_children++;
    }
    if (live_children != 0u && leaf_children == live_children) {
        uint leaf_base = ctr_add(CTR_LEAF, live_children);
        uint live_rank = 0u;
        [unroll] for (uint bc = 0; bc < 4; ++bc) {
            uint bw = cx1[bc] - cx0[bc];
            uint bh = cy1[bc] - cy0[bc];
            if (bw == 0u || bh == 0u) continue;
            uint leaf_at = leaf_base + live_rank++;
            if (leaf_at < cap_leaf) {
                LeafRec lf;
                lf.xy0 = pack_xy(cx0[bc], cy0[bc]);
                lf.xy1 = pack_xy(cx1[bc], cy1[bc]);
                lf.t_start = tc;
                lf.depth = d;
                lf.frontier = frontier_from_binary_cut(leaf_slot, leaf_len);
                qleaf[leaf_at] = lf;
            } else {
                InterlockedAdd(counters[CTR_OVERFLOW], 1, s);
            }
        }
        return;
    }
    // No all-tile batch arm under WORKGRAPH: there is no `push1` queue to
    // reserve from, so the mixed loop below (which forks per child) handles
    // that case with no loss — the batch existed only to fold four contended
    // atomics into one.
#if !defined(WORKGRAPH)
    if (live_children != 0u && leaf_children == 0u) {
        uint tile_base = ctr_add(push1, live_children);
        uint live_rank = 0u;
        [unroll] for (uint bc = 0; bc < 4; ++bc) {
            uint bw = cx1[bc] - cx0[bc];
            uint bh = cy1[bc] - cy0[bc];
            if (bw == 0u || bh == 0u) continue;
            uint tile_at = tile_base + live_rank++;
            if (tile_at < cap_tile) {
                TileRec child;
                child.xy0 = pack_xy(cx0[bc], cy0[bc]);
                child.xy1 = pack_xy(cx1[bc], cy1[bc]);
                child.t_start = tc;
                child.cut_slot = out_slot;
                child.meta = out_len | (d << 8);
                child.path = cpath | bc;
                qout[tile_at] = child;
            } else {
                InterlockedAdd(counters[CTR_OVERFLOW], 1, s);
            }
        }
        return;
    }
#endif
    // B70 HOMOGENEOUS-BATCH END.
#endif

    [unroll] for (uint c = 0; c < 4; ++c) {
        uint w = cx1[c] - cx0[c];
        uint h = cy1[c] - cy0[c];
        if (w == 0 || h == 0) continue; // trace_tile's empty-rect guard
        if (w <= LEAF_TILE && h <= LEAF_TILE) {
            InterlockedAdd(counters[CTR_LEAF], 1, s);
            if (s < cap_leaf) {
                LeafRec lf;
                lf.xy0 = pack_xy(cx0[c], cy0[c]);
                lf.xy1 = pack_xy(cx1[c], cy1[c]);
                lf.t_start = tc;
                lf.depth = d;
                lf.frontier = frontier_from_binary_cut(leaf_slot, leaf_len);
                qleaf[s] = lf;
            } else {
                InterlockedAdd(counters[CTR_OVERFLOW], 1, s);
            }
        } else {
            TileRec child;
            child.xy0 = pack_xy(cx0[c], cy0[c]);
            child.xy1 = pack_xy(cx1[c], cy1[c]);
            child.t_start = tc;
            child.cut_slot = out_slot;
            child.meta = out_len | (d << 8);
            child.path = cpath | c;
#if defined(WORKGRAPH)
            // `c` is the unrolled loop counter, so this is a compile-time
            // index. The graph's record storage is what bounds fanout here —
            // there is no cap_tile to overflow against.
            wg_kids[c] = child;
            wg_mask |= (1u << c);
#else
            InterlockedAdd(counters[push1], 1, s);
            if (s < cap_tile) {
                qout[s] = child;
            } else {
                InterlockedAdd(counters[CTR_OVERFLOW], 1, s);
            }
#endif
        }
    }
}

// The ExecuteIndirect ladder's two kernels. Compiled out of the work-graph
// unit, which reaches level_finish through its own node shaders and gets a
// different (children-out) signature — see workgraph.hlsl.
#if !defined(WORKGRAPH)

// One thread per tile — the deep-level flavor, where tiles are many and each
// one's inherited cut is tight enough that a private DFS is the right shape.
[numthreads(32, 1, 1)]
void cs_level(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint idx = flat_group(gid) * 32u + gtid.x;
    if (idx >= counters[push0]) return;
    TileRec rec = qin[idx];
    uint2 p0 = rect_min(rec.xy0);
    uint2 p1 = rect_max(rec.xy1);
    TF f = tile_frustum(p0.x, p0.y, p1.x, p1.y);
    uint cut_len = rec.meta & 0xffu;
    float best = bound_query(f, rec.t_start, FLT_MAX, rec.cut_slot, cut_len, gtid.x);
    level_finish(rec, p0, p1, rec.meta >> 8, cut_len, f, best, gtid.x);
}

// One GROUP per tile — the shallow-level flavor. Dispatched with one group per
// record (the prep's items-per-group is 1 here, not 32), so `flat_group` IS the
// tile index. The early-out and every barrier inside the query are
// group-uniform; only after the query does the group narrow to lane 0, which
// owns g_stack slab 0 for refine_cut.
[numthreads(32, 1, 1)]
void cs_level_wide(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint idx = flat_group(gid);
    if (idx >= counters[push0]) return;
    TileRec rec = qin[idx];
    uint2 p0 = rect_min(rec.xy0);
    uint2 p1 = rect_max(rec.xy1);
    TF f = tile_frustum(p0.x, p0.y, p1.x, p1.y);
    uint cut_len = rec.meta & 0xffu;
    float best = bound_query_wave(f, rec.t_start, FLT_MAX, rec.cut_slot, cut_len, gtid.x);
    if (gtid.x != 0) return;
    level_finish(rec, p0, p1, rec.meta >> 8, cut_len, f, best, 0u);
}

#endif // !WORKGRAPH

// --- sky fill: moved to sky.hlsl -----------------------------------------
// `cs_sky` and its lattice pass live in their own compile unit so the cloud
// lattice can take u5 (the tile queue's register, dead by then) without a
// 15th root UAV — the root signature is at 62/64 DWORDs. See sky.hlsl.
