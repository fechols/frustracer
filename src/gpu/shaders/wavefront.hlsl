// The wavefront quadtree: seed -> D x (prep -> level) -> sky-fill, with the
// leaf pass in leaf.hlsl. One statically recorded command list; every
// decision after the root seed is made by GPU-written counters feeding
// ExecuteIndirect. Requires trace_common.hlsli + queues.hlsli +
// frustum.hlsli pasted first.

// --- seed: reset counters + cut pool bump, enqueue the whole-screen root ---

[numthreads(1, 1, 1)]
void cs_seed(uint3 id : SV_DispatchThreadID) {
    for (uint i = 0; i < CTR_COUNT; ++i) counters[i] = 0;
    if (rw <= 8 && rh <= 8) {
        // Degenerate window: the root IS a leaf (trace_tile's first check).
        LeafRec lf;
        lf.xy0 = pack_xy(0, 0);
        lf.xy1 = pack_xy(rw, rh);
        lf.t_start = 0.0;
        lf.depth = 0;
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

// --- the level kernel: render.rs::tile_step, one thread per tile ----------
// push0 = in counter (tile A or B), push1 = out counter (the other).

[numthreads(32, 1, 1)]
void cs_level(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint idx = flat_group(gid) * 32u + gtid.x;
    if (idx >= counters[push0]) return;
    TileRec rec = qin[idx];
    uint2 p0 = rect_min(rec.xy0);
    uint2 p1 = rect_max(rec.xy1);
    uint depth = rec.meta >> 8;
    uint cut_len = rec.meta & 0xffu;
    uint s;

    TF f = tile_frustum(p0.x, p0.y, p1.x, p1.y);
    float best = bound_query(f, rec.t_start, FLT_MAX, rec.cut_slot, cut_len, gtid.x);
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
        out_len = refine_cut(f, tc, FLT_MAX, rec.cut_slot, cut_len, out_slot, gtid.x);
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
    [unroll] for (uint c = 0; c < 4; ++c) {
        uint w = cx1[c] - cx0[c];
        uint h = cy1[c] - cy0[c];
        if (w == 0 || h == 0) continue; // trace_tile's empty-rect guard
        if (w <= 8 && h <= 8) {
            InterlockedAdd(counters[CTR_LEAF], 1, s);
            if (s < cap_leaf) {
                LeafRec lf;
                lf.xy0 = pack_xy(cx0[c], cy0[c]);
                lf.xy1 = pack_xy(cx1[c], cy1[c]);
                lf.t_start = tc;
                lf.depth = d;
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

// --- sky fill: one 64-lane group per SkyRec, grid-striding the rect -------
// render.rs::fill_sky: pixel-CENTER dirs (no jitter, no rng), KIND_SKY at
// the depth the tile was proven.

[numthreads(64, 1, 1)]
void cs_sky(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint rec_i = flat_group(gid);
    if (rec_i >= counters[push0]) return;
    SkyRec rec = qsky[rec_i];
    uint2 p0 = rect_min(rec.xy0);
    uint2 p1 = rect_max(rec.xy1);
    uint w = p1.x - p0.x;
    uint total = w * (p1.y - p0.y);
    for (uint i = gtid.x; i < total; i += 64u) {
        uint x = p0.x + i % w;
        uint y = p0.y + i / w;
        float3 dir = ray_dir(float(x) + 0.5, float(y) + 0.5);
        float3 c = sky_color(dir);
        uint pi = y * rw + x;
        uint i3 = pi * 3u;
        // The compose pass is the single accum splat site; sky pixels carry
        // their full color in `partial` with zero ambient weight.
        partial[i3 + 0u] = c.x;
        partial[i3 + 1u] = c.y;
        partial[i3 + 2u] = c.z;
        ambw[i3 + 0u] = 0.0;
        ambw[i3 + 1u] = 0.0;
        ambw[i3 + 2u] = 0.0;
        tbuf[pi] = INF;
        info[pi] = pack_info(rec.depth, KIND_SKY);
        // Pixel centers, matching the CPU sky-tile flood (leaf-tile sky
        // pixels get the jittered position in leaf.hlsl instead).
        gbuf_write_sky(pi, float(x) + 0.5, float(y) + 0.5, dir);
    }
}
