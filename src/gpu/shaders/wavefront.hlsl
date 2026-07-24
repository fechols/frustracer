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
        lf.cut_slot = ROOT_CUT_SLOT; // the root cut [0]
        lf.cut_len = 1u;
        qleaf[0] = lf;
        counters[CTR_LEAF] = 1;
        return;
    }
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
        bool keep = i == CTR_LEAF || i == CTR_SKY || i == CTR_CUT;
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

// float-min via InterlockedMin on the IEEE bit pattern: exact for NON-NEGATIVE
// finite floats, whose bit patterns order exactly as their values. Every
// candidate here is `max(distance, t_start) >= 0` or the FLT_MAX seed.
void gw_min(float d) {
    uint prev;
    InterlockedMin(gw_best, asuint(d), prev);
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
        float d;
        if (!ft_slot(f, nd, e & 7u, t_start, d) || d >= asfloat(gw_best)) continue;
        uint c = nd.child[e & 7u];
        if (c == FT_INVALID) { gw_min(d); continue; } // terminal: a leaf's box
        uint w;
        InterlockedAdd(gw_len[0], 1u, w);
        if (w < WQ_CAP) g_stack[w] = c; else gw_min(d);
    }
#else
    for (uint r = tid; r < cut_len; r += 32u) {
        uint w;
        InterlockedAdd(gw_len[0], 1u, w);
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
            [unroll] for (uint s = 0; s < 8; ++s) {
                float d;
                if (!ft_slot(f, nd, s, t_start, d) || d >= asfloat(gw_best)) continue;
                uint c = nd.child[s];
                if (c == FT_INVALID) { gw_min(d); continue; }
                uint w;
                InterlockedAdd(gw_len[1u - cur], 1u, w);
                if (w < WQ_CAP) g_stack[bout + w] = c; else gw_min(d);
            }
#else
            BvhNode node = bvh_nodes[idx];
            if (aabb_outside(f, node.mn, node.mx)) continue;
            if (point_aabb_max_dist(f.origin, node.mn, node.mx) <= t_start) continue;
            float d = max(point_aabb_dist(f.origin, node.mn, node.mx), t_start);
            if (d >= asfloat(gw_best)) continue;
            if (node.count > 0) { gw_min(d); continue; }
            uint w;
            InterlockedAdd(gw_len[1u - cur], 2u, w);
            // Adds are all +2 from a zeroed base, so accepted w are even and
            // slots [0, WQ_CAP) are fully written whenever the clamp bites.
            if (w + 1u < WQ_CAP) {
                g_stack[bout + w] = node.left_first;
                g_stack[bout + w + 1u] = node.left_first + 1u;
            } else {
                // Frontier full: take this node's own distance as if it were a
                // leaf — a valid lower bound for everything inside it. The
                // serial path's stack-pressure fallback, verbatim.
                gw_min(d);
            }
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

void level_finish(TileRec rec, uint2 p0, uint2 p1, uint depth, uint cut_len, TF f,
                  float best, uint lane) {
    uint s;
    if (best == FLT_MAX) {
        // Sky: the whole frustum (beyond the inherited ball, which the
        // ancestor claim covers) is empty.
        InterlockedAdd(counters[CTR_SKY], 1, s);
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
    if (!advanced) {
        // Blocked at the inherited distance — still subdivide (children's
        // smaller frustums may exclude the blocker; that is how sky emerges).
        InterlockedAdd(counters[CTR_BLOCKED], 1, s);
    }
    InterlockedAdd(counters[CTR_SPLIT], 1, s);

    // Refine the cut into a fresh pool slot; on pool exhaustion the children
    // inherit the PARENT's slot — an ancestor cut is valid for any descendant
    // frustum (coarse, never wrong) — the refine_cut coarse-emit analog.
    uint out_slot;
    InterlockedAdd(counters[CTR_CUT], 1, out_slot);
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
                lf.cut_slot = leaf_slot;
                lf.cut_len = leaf_len;
                qleaf[s] = lf;
            } else {
                InterlockedAdd(counters[CTR_OVERFLOW], 1, s);
            }
        } else {
            InterlockedAdd(counters[push1], 1, s);
            if (s < cap_tile) {
                TileRec child;
                child.xy0 = pack_xy(cx0[c], cy0[c]);
                child.xy1 = pack_xy(cx1[c], cy1[c]);
                child.t_start = tc;
                child.cut_slot = out_slot;
                child.meta = out_len | (d << 8);
                child.path = cpath | c;
                qout[s] = child;
            } else {
                InterlockedAdd(counters[CTR_OVERFLOW], 1, s);
            }
        }
    }
}

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

// --- sky fill: moved to sky.hlsl -----------------------------------------
// `cs_sky` and its lattice pass live in their own compile unit so the cloud
// lattice can take u5 (the tile queue's register, dead by then) without a
// 15th root UAV — the root signature is at 62/64 DWORDs. See sky.hlsl.
