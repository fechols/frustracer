// THE QUADTREE LADDER AS A D3D12 WORK GRAPH (spike; FR_WORKGRAPH=1).
//
// Replaces `cs_seed` + N x (`cs_prep` + ExecuteIndirect) with ONE DispatchGraph.
// Nothing else moves: the leaf and sky fills still run afterwards off the same
// `qleaf`/`qsky`/`counters`, so the exactly-once coverage gate, the terminal
// accounting, and `record_terminal_fills` are untouched, and the measurement is
// isolated to the ladder.
//
// WHY THE TILE QUEUES CANNOT SURVIVE. `qin`/`qout` ping-pong per level, which
// assumes levels SERIALISE — exactly the assumption a graph exists to break.
// So tile children become node output records (the scheduler owns their
// storage) while leaf/sky records keep their UAV queues. `level_finish` hands
// them back through `out TileRec[4] + mask` under `#if defined(WORKGRAPH)`;
// every other line of it, including the cut-pool bump, is the shared code.
//
// TWO NODES, mirroring WIDE_LEVELS. Shallow levels get a whole group per tile
// (broadcasting, the `bound_query_wave` frontier); deep levels get one thread
// per tile (coalescing, the private DFS). That split is the measured shape —
// see trace::WIDE_LEVELS — and work graphs express it natively: a broadcasting
// node hands its last level's children straight to the coalescing node.
//
// *** THE UNIFORMITY RULE, and it is the easy thing to get catastrophically
// wrong here. *** Per the work-graphs spec, GetThreadNodeOutputRecords and
// OutputComplete "must be thread group uniform, not in any flow control that
// could vary within the thread group (VARYING INCLUDES THREADS EXITING)".
// `cs_level_wide` runs level_finish under `if (gtid.x != 0) return;` — doing
// that here would be undefined behaviour. So the wide node keeps ALL 32 lanes
// alive to the end: lane 0 computes the split into groupshared, a barrier
// publishes it, and then every lane calls the allocation with a per-thread
// count of 0 or 1 (a varying COUNT is explicitly allowed; varying control flow
// is not). The recursion-level branch is group-uniform because the level is a
// property of the launch, not of the thread.
//
// AND THE FANOUT PROMISE IS NOT CHECKED FOR YOU. Exceeding [MaxRecords] or
// [NodeMaxRecursionDepth] is, per spec, "undefined results ... memory
// corruption, commandlist-removed state or device-removed state" — there is no
// runtime error. A quadtree split emits at most 4 children, which is why the
// budgets below are exact rather than generous, and why WG_WIDE_LEVELS /
// WG_DEEP_LEVELS are injected from the Rust side's own depth_full rather than
// guessed here.

// Injected by trace.rs: WG_WIDE_LEVELS, WG_DEEP_LEVELS.
#ifndef WG_WIDE_LEVELS
#define WG_WIDE_LEVELS 6
#endif
#ifndef WG_DEEP_LEVELS
#define WG_DEEP_LEVELS 8
#endif

// Lane 0's split result, published to the group. 4 x 24 B + 4 B — trivial
// beside the 8 KB frustum stack slab this unit already carries, and this node
// traces no rays, so Intel's groupshared/RTU-L1 rule does not apply (see
// leaf.hlsl's header for the one that does).
groupshared TileRec wg_g_kids[4];
groupshared uint    wg_g_mask;

// Compact a 4-slot child array by popcount rank: lane `i` takes the i'th set
// bit. All indices are unrolled loop counters, so nothing is dynamically
// indexed (the ft_expand scratch-demotion lesson).
TileRec wg_pick(TileRec kids[4], uint mask, uint i) {
    TileRec out_rec = kids[0];
    [unroll] for (uint c = 0; c < 4; ++c) {
        if ((mask & (1u << c)) != 0 && countbits(mask & ((1u << c) - 1u)) == i) {
            out_rec = kids[c];
        }
    }
    return out_rec;
}

// --- shallow levels: one GROUP per tile -----------------------------------
[Shader("node")]
[NodeLaunch("broadcasting")]
[NodeIsProgramEntry]
[NodeID("TileWide")]
[NodeMaxRecursionDepth(WG_WIDE_LEVELS)]
[NodeDispatchGrid(1, 1, 1)]
[numthreads(32, 1, 1)]
void TileWide(
    uint gtid : SV_GroupThreadID,
    DispatchNodeInputRecord<TileRec> input,
    [MaxRecords(4)] NodeOutput<TileRec> TileWide,
    [MaxRecordsSharedWith(TileWide)] NodeOutput<TileRec> TileDeep)
{
    TileRec rec = input.Get();
    uint2 p0 = rect_min(rec.xy0);
    uint2 p1 = rect_max(rec.xy1);
    TF f = tile_frustum(p0.x, p0.y, p1.x, p1.y);
    uint cut_len = rec.meta & 0xffu;

    // Every lane participates — the frontier is group-cooperative.
    float best = bound_query_wave(f, rec.t_start, FLT_MAX, rec.cut_slot, cut_len, gtid);

    // Lane 0 owns g_stack slab 0 for refine_cut, exactly as cs_level_wide
    // does. It must NOT return from the shader here (see the uniformity rule
    // above) — level_finish is a function, so its own early returns are fine.
    if (gtid == 0) {
        TileRec kids[4];
        uint mask;
        level_finish(rec, p0, p1, rec.meta >> 8, cut_len, f, best, 0u, kids, mask);
        wg_g_mask = mask;
        [unroll] for (uint i = 0; i < 4; ++i) wg_g_kids[i] = kids[i];
    }
    GroupMemoryBarrierWithGroupSync();

    uint mask = wg_g_mask;                 // group-uniform
    uint n = countbits(mask);
    TileRec kids[4];
    [unroll] for (uint i = 0; i < 4; ++i) kids[i] = wg_g_kids[i];

    // Group-uniform: the recursion level is a property of the launch.
    if (GetRemainingRecursionLevels() == 0) {
        // Last wide level — hand the children to the per-thread node.
        ThreadNodeOutputRecords<TileRec> o = TileDeep.GetThreadNodeOutputRecords(gtid < n ? 1 : 0);
        if (gtid < n) o.Get(0) = wg_pick(kids, mask, gtid);
        o.OutputComplete();
    } else {
        ThreadNodeOutputRecords<TileRec> o = TileWide.GetThreadNodeOutputRecords(gtid < n ? 1 : 0);
        if (gtid < n) o.Get(0) = wg_pick(kids, mask, gtid);
        o.OutputComplete();
    }
}

// --- deep levels: one THREAD per tile --------------------------------------
// Coalescing, so the scheduler packs up to 32 tiles into one group — the shape
// cs_level gets from the ladder, without the dispatch. The record count is
// implementation-defined and must be read with Count().
[Shader("node")]
[NodeLaunch("coalescing")]
[NodeID("TileDeep")]
[NodeMaxRecursionDepth(WG_DEEP_LEVELS)]
[numthreads(32, 1, 1)]
void TileDeep(
    uint gtid : SV_GroupThreadID,
    [MaxRecords(32)] GroupNodeInputRecords<TileRec> inputs,
    [MaxRecords(128)] NodeOutput<TileRec> TileDeep)   // 32 threads x 4 children
{
    uint cnt = inputs.Count();
    uint n = 0;
    TileRec kids[4];
    uint mask = 0;
    // A lane with no record does no work but STAYS ALIVE for the allocation.
    if (gtid < cnt) {
        TileRec rec = inputs.Get(gtid);
        uint2 p0 = rect_min(rec.xy0);
        uint2 p1 = rect_max(rec.xy1);
        TF f = tile_frustum(p0.x, p0.y, p1.x, p1.y);
        uint cut_len = rec.meta & 0xffu;
        float best = bound_query(f, rec.t_start, FLT_MAX, rec.cut_slot, cut_len, gtid);
        level_finish(rec, p0, p1, rec.meta >> 8, cut_len, f, best, gtid, kids, mask);
        n = countbits(mask);
    }
    // The last level must not recurse: children beyond the declared depth are
    // memory corruption, not a caught error. This should never actually drop
    // anything — `depth_full` is chosen so the deepest split's children are all
    // LEAVES (level_finish's terminal-children path), so n is already 0 — so
    // any drop is a real bug and is counted into CTR_OVERFLOW, which every
    // suite already gates at exactly 0. A silent drop would be a hole in the
    // image, i.e. the false-sky class. Depth is a launch property, so the test
    // is group-uniform.
    if (GetRemainingRecursionLevels() == 0 && n != 0) {
        uint dropped;
        InterlockedAdd(counters[CTR_OVERFLOW], n, dropped);
        n = 0;
    }

    ThreadNodeOutputRecords<TileRec> o = TileDeep.GetThreadNodeOutputRecords(n);
    [unroll] for (uint i = 0; i < 4; ++i) {
        if (i < n) o.Get(i) = wg_pick(kids, mask, i);
    }
    o.OutputComplete();
}
