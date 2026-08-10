// Hemisphere wavefront, query half: the root pass (hemi.rs::integrate's
// tangent-half-space query + octant split) and the cell pass
// (hemi.rs::cell — bound query over the inherited cut, empty -> analytic,
// query cutoff -> leaf record, else refine + midpoint split). Requires
// trace_common.hlsli + ctr.hlsli + hemi.hlsli + frustum.hlsli + rt.hlsli
// pasted first. Both kernels run 32 threads/group, one thread per record,
// sharing frustum.hlsli's per-lane groupshared stacks.

// Verify: an empty-cell claim re-validated with the CPU check's 6-direction
// grid — a hit inside the claim is exactly the false-sky bug shape.
void check_empty_cell(float3 o, float3 a, float3 b, float3 c, float t_lim) {
    float tmax = t_lim < FLT_MAX ? t_lim * (1.0 - 1e-3) : FLT_MAX;
    const float3 W[6] = {
        float3(1, 1, 1), float3(6, 1, 1), float3(1, 6, 1),
        float3(1, 1, 6), float3(3, 3, 1), float3(1, 3, 3),
    };
    uint dummy;
    [unroll] for (uint i = 0; i < 6; ++i) {
        float3 d = normalize(a * W[i].x + b * W[i].y + c * W[i].z);
        if (occluded_q(o, d, 0.0, tmax)) {
            InterlockedAdd(counters[CTR_V_FALSE_EMPTY], 1, dummy);
        }
    }
}

void enqueue_cell(uint out_ctr, float3 a, float3 b, float3 c, float tc, uint point_id,
                  uint depth, uint path, uint cut_slot, uint cut_len) {
    uint s;
    bool leaf = out_ctr == CTR_HEMI_LEAF;
    InterlockedAdd(counters[out_ctr], 1, s);
    // Each queue guards its own CB capacity (trace.rs currently sizes them
    // equal, but the check must track the buffer actually written below).
    if (s >= (leaf ? cap_hemi_leaf : cap_hemi_cell)) {
        InterlockedAdd(counters[CTR_OVERFLOW], 1, s);
        return;
    }
    HemiCellRec rec;
    rec.a = a;
    rec.t_start = tc;
    rec.b = b;
    rec.point_id = point_id;
    rec.c = c;
    rec.meta = depth | (path << 8);
    rec.cut_slot = cut_slot;
    rec.cut_len = cut_len;
    rec._pad1 = 0;
    rec._pad2 = 0;
    if (leaf) {
        hqleaf[s] = rec;
    } else {
        hqout[s] = rec;
    }
}

// --- root pass: one thread per shading point in the batch ------------------
// push0 = batch base into hemi_pts, push1 = out counter (CTR_HEMI_A).

[numthreads(32, 1, 1)]
void cs_hemi_root(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint local = flat_group(gid) * 32u + gtid.x;
    if (local >= hemi_batch) return;
    uint idx = push0 + local;
    if (idx >= counters[CTR_HEMI_PT]) return;
    HemiPointRec pt = hemi_pts[idx];
    float t_lim = hemi_t_limit();
    float3 t1, t2;
    onb(pt.n, t1, t2);
    uint dummy;

    // Root: the tangent half-space, queried over the whole BVH. Apex t_start
    // is 0, NOT eps — ball(o, eps) is a false claim at concave corners.
    TF root = half_space_frustum(pt.o, pt.n);
    float best = bound_query(root, 0.0, t_lim, ROOT_CUT_SLOT, 1, gtid.x);
    if (best == t_lim) {
        // Whole hemisphere open within t_limit — one query, done.
        InterlockedAdd(counters[CTR_HEMI_EMPTY], 1, dummy);
        if (fb_mode == 1u) {
            hemi_add(pt.pixel, 0, PI);
        } else {
            [unroll] for (uint i = 0; i < 4; ++i) {
                float3 a, b, c;
                octant(i, pt.n, t1, t2, a, b, c);
                hemi_add3(pt.pixel, sky_cell_sum(pt.n, a, b, c, 4));
            }
        }
        if (flags & FLAG_VERIFY) {
            hemi_add(pt.pixel, 3, PI);
            for (uint i = 0; i < 4; ++i) {
                float3 a, b, c;
                octant(i, pt.n, t1, t2, a, b, c);
                check_empty_cell(pt.o, a, b, c, t_lim);
            }
        }
        return;
    }

    // Advance from 0 + refine [0] exactly like tile_step, then the 4 octants.
    bool adv = best > SCENE_EPS; // advance_tc with t_start = 0
    float tc = adv ? max(best * (1.0 - 1e-4), 0.0) : 0.0;
    uint slot;
    InterlockedAdd(counters[CTR_HEMI_CUT], 1, slot);
    uint olen = 0;
    if (slot < cap_hemi_cut) {
        olen = refine_cut(root, tc, t_lim, ROOT_CUT_SLOT, 1, slot, gtid.x);
    } else {
        InterlockedAdd(counters[CTR_CUT_FALLBACK], 1, dummy);
    }
    uint use_slot = slot;
    uint use_len = olen;
    if (olen == 0) {
        use_slot = ROOT_CUT_SLOT;
        use_len = 1;
    }
    [unroll] for (uint i = 0; i < 4; ++i) {
        float3 a, b, c;
        octant(i, pt.n, t1, t2, a, b, c);
        enqueue_cell(push1, a, b, c, tc, idx, 1, i, use_slot, use_len);
    }
}

// --- cell pass: one thread per spherical-triangle cell ----------------------
// push0 = in counter, push1 = out counter.

[numthreads(32, 1, 1)]
void cs_hemi_cell(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint idx = flat_group(gid) * 32u + gtid.x;
    if (idx >= counters[push0]) return;
    HemiCellRec rec = hqin[idx];
    HemiPointRec pt = hemi_pts[rec.point_id];
    uint depth = rec.meta & 0xffu;
    uint path = rec.meta >> 8;
    float t_lim = hemi_t_limit();
    uint dummy;

    TF f = tri_cell_frustum(pt.o, rec.a, rec.b, rec.c);
    float best = bound_query(f, rec.t_start, t_lim, rec.cut_slot, rec.cut_len, gtid.x);

    // There is deliberately NO cut-vs-full-tree bound compare here (the
    // CPU's cut-miss gate compared cut-seeded RAYS, which no longer exist —
    // GPU rays traverse from the root). A bound compare is unsound as an
    // exact-zero gate: the cut legitimately encodes ANCESTOR culling (the
    // root's tangent half-space) that this cell's own 3-plane frustum cannot
    // express, so a conservative full-tree query can descend into
    // below-horizon boxes and report a lower bound than the (tighter,
    // correct) cut. Cut fidelity is validated where its claims are consumed:
    // false-empty (real rays through claimed-empty cells) and
    // tmin-overshoot (real rays against claimed balls) — both exact-zero.

    if (best == t_lim) {
        // Empty within t_limit: analytic resolve (AO: exact PSA; GI: sky with
        // centroid refinement), zero rays.
        InterlockedAdd(counters[CTR_HEMI_EMPTY], 1, dummy);
        if (fb_mode == 1u) {
            hemi_add(pt.pixel, 0, psa3(rec.a, rec.b, rec.c, pt.n));
        } else {
            uint budget = depth < 5u ? 5u - depth : 0u;
            hemi_add3(pt.pixel, sky_cell_sum(pt.n, rec.a, rec.b, rec.c, budget));
        }
        if (flags & FLAG_VERIFY) {
            hemi_add(pt.pixel, 3, psa3(rec.a, rec.b, rec.c, pt.n));
            check_empty_cell(pt.o, rec.a, rec.b, rec.c, t_lim);
        }
        return;
    }

    // advance_tc; blocked cells carry t_start through and still subdivide.
    bool adv = best > rec.t_start + max(rec.t_start * 1e-4, SCENE_EPS);
    float tc = adv ? max(best * (1.0 - 1e-4), rec.t_start) : rec.t_start;

    if (depth + 1u >= fb_depth) {
        // Query cutoff (LEAF_LEVELS = 1): one leaf record -> 4 stratified
        // rays in the leaf pass, all seeded with this cell's tc.
        enqueue_cell(CTR_HEMI_LEAF, rec.a, rec.b, rec.c, tc, rec.point_id, depth, path,
                     rec.cut_slot, rec.cut_len);
        return;
    }

    uint slot;
    InterlockedAdd(counters[CTR_HEMI_CUT], 1, slot);
    uint olen = 0;
    if (slot < cap_hemi_cut) {
        olen = refine_cut(f, tc, t_lim, rec.cut_slot, rec.cut_len, slot, gtid.x);
    } else {
        InterlockedAdd(counters[CTR_CUT_FALLBACK], 1, dummy);
    }
    uint use_slot = slot;
    uint use_len = olen;
    if (olen == 0) {
        // Pool exhausted (or the structurally-impossible empty refine): the
        // parent cut is valid for any child cell — coarse, never wrong.
        use_slot = rec.cut_slot;
        use_len = rec.cut_len;
    }
    float3 mab, mbc, mca;
    midpoints3(rec.a, rec.b, rec.c, mab, mbc, mca);
    uint d = depth + 1u;
    uint p = path << 2;
    enqueue_cell(push1, rec.a, mab, mca, tc, rec.point_id, d, p | 0u, use_slot, use_len);
    enqueue_cell(push1, mab, rec.b, mbc, tc, rec.point_id, d, p | 1u, use_slot, use_len);
    enqueue_cell(push1, mca, mbc, rec.c, tc, rec.point_id, d, p | 2u, use_slot, use_len);
    enqueue_cell(push1, mab, mbc, mca, tc, rec.point_id, d, p | 3u, use_slot, use_len);
}
