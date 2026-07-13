// ftree.rs on the GPU: the 8-wide frustum tree as the bound_query/refine_cut
// backend, compiled in by `#define FTREE` (the ALPHA_CUTOUT pattern — the
// session picks its frustum structure at kernel-assembly time). Pasted AFTER
// frustum.hlsli, whose swappable half (BvhNode/t0, bound_query, refine_cut)
// is `#ifndef FTREE`-guarded; the TF frustum type, the plane/distance helpers
// and the per-lane g_stack slab are shared — every conservative slack is the
// same code, term-for-term.
//
// Same function SIGNATURES as the binary versions, so the four call sites
// (wavefront.hlsl x2, hemi_wave.hlsl x4) compile unchanged. Cut entries are
// SLOT-REFS, `(node << 3) | slot` (ftree.rs's encoding); they ride the same
// 64-u32 cut_pool slots. ROOT_CUT_SLOT expands to the wide root's occupied
// slots in place: entry(0, s) == s, so the sentinel needs no seed changes —
// the occupancy test (bnode == FT_INVALID) skips the empty tail. Rays never
// consume these cuts on the GPU (every ray is DXR RayQuery), so no slot->
// binary translation exists here at all.
//
// The traversal stack is the SAME per-lane 64-u32 slab frustum.hlsli owns,
// holding child node ids only (no stored distance): 8-way expansion can hold
// up to depth*7 pending siblings, so survivors push FAR->NEAR (an 8-slot
// insertion sort in registers keeps the nearest popping first) and stack
// pressure falls back to the coarse bound `best = d` — a valid lower bound,
// exactly the binary version's overflow rule. A dead node popped without a
// stored-d skip just re-tests 8 slot boxes and dies; the ftree can afford it.

#ifdef FTREE

// ftree.rs::FNode, uploaded VERBATIM (repr(C), 256 B, no padding): SoA boxes
// so the 8 slot tests are contiguous loads, `child` = wide child id or
// FT_INVALID for terminal slots, `bnode` = the binary node id (occupancy
// marker here; FT_INVALID = empty slot, its box is inverted-empty).
struct FtNode {
    float mnx[8];
    float mny[8];
    float mnz[8];
    float mxx[8];
    float mxy[8];
    float mxz[8];
    uint child[8];
    uint bnode[8];
};
StructuredBuffer<FtNode> ft_nodes : register(t0);

#define FT_INVALID 0xffffffffu

// Test one slot's box against the frustum/ball/best prunes. False = culled
// (empty slot, outside a plane, or inside the proven-empty ball); true hands
// back d = max(dist, t_start), the slot's conservative bound candidate.
bool ft_slot(TF f, FtNode nd, uint s, float t_start, out float d) {
    d = 0.0;
    if (nd.bnode[s] == FT_INVALID) return false;
    float3 mn = float3(nd.mnx[s], nd.mny[s], nd.mnz[s]);
    float3 mx = float3(nd.mxx[s], nd.mxy[s], nd.mxz[s]);
    if (aabb_outside(f, mn, mx)) return false;
    if (point_aabb_max_dist(f.origin, mn, mx) <= t_start) return false;
    d = max(point_aabb_dist(f.origin, mn, mx), t_start);
    return true;
}

// Expand one wide node: test all 8 slots, tighten `best` with terminal
// survivors, and push internal survivors FAR->NEAR onto the lane stack (so
// the nearest pops first — the pruning power the stored-d skip would buy).
// Ordering is a selection scan over a live-bitmask: every sd/sc access has a
// COMPILE-TIME index after unrolling, so the arrays stay in registers — an
// insertion sort's dynamic shuffle indices demote them to scratch memory,
// which measured +58% on the hemi bench (hemi_cell is the hottest caller).
// Stack pressure takes the slot's own d as a coarse leaf — sound, see header.
void ft_expand(TF f, uint ni, float t_start, inout float best, uint base, inout uint sp) {
    FtNode nd = ft_nodes[ni];
    float sd[8];
    uint sc[8];
    uint live = 0;
    [unroll] for (uint s = 0; s < 8; ++s) {
        sd[s] = 0.0;
        sc[s] = 0;
        float d;
        if (!ft_slot(f, nd, s, t_start, d) || d >= best) continue;
        if (nd.child[s] == FT_INVALID) {
            best = d; // terminal slot: a binary leaf's box distance
            continue;
        }
        sd[s] = d;
        sc[s] = nd.child[s];
        live |= 1u << s;
    }
    while (live != 0) {
        // Extract the farthest live slot (scalar pd/pc carry the values, so
        // no dynamic array reads anywhere).
        uint pick = 0;
        uint pc = 0;
        float pd = -1.0;
        [unroll] for (uint s = 0; s < 8; ++s) {
            if ((live & (1u << s)) != 0 && sd[s] > pd) {
                pd = sd[s];
                pc = sc[s];
                pick = s;
            }
        }
        live &= ~(1u << pick);
        if (sp + 1 > LANE_STACK) {
            best = pd; // coarse: the slot's d lower-bounds its subtree
            continue;
        }
        g_stack[base + sp++] = pc;
    }
}

// frustum.rs::nearest_geometry_distance(_within) over the wide tree — the
// same contract as the binary version above (returns exactly t_limit for
// "no geometry"), the same result values (identical per-box math over the
// same binary-node boxes; min is order-independent).
float bound_query(TF f, float t_start, float t_limit, uint cut_slot, uint cut_len, uint lane) {
    float best = t_limit;
    uint base = lane * LANE_STACK;
    uint sp = 0;
    uint n = cut_slot == ROOT_CUT_SLOT ? 8u : cut_len;
    for (uint r = 0; r < n; ++r) {
        uint e = cut_slot == ROOT_CUT_SLOT ? r : cut_pool[cut_slot * 64u + r];
        uint ni = e >> 3;
        uint s = e & 7u;
        FtNode nd = ft_nodes[ni];
        float d;
        if (!ft_slot(f, nd, s, t_start, d) || d >= best) continue;
        if (nd.child[s] == FT_INVALID) {
            best = d;
            continue;
        }
        if (sp + 1 > LANE_STACK) {
            best = d;
            continue;
        }
        g_stack[base + sp++] = nd.child[s];
    }
    while (sp > 0) {
        ft_expand(f, g_stack[base + --sp], t_start, best, base, sp);
    }
    return best;
}

// frustum.rs::refine_cut over slot-ref entries — the same three drop rules,
// the same never-drop-on-d>=best law, the same out+work <= 64 coarse-emit
// invariant. Expanding an internal slot pushes its wide child's occupied
// slots (up to 8 at once — overflow coarsens a little earlier than the
// binary 2-way split; the entry is emitted, never dropped).
uint refine_cut(TF f, float t_ball, float t_far, uint parent_slot, uint parent_len, uint out_slot, uint lane) {
    uint base = lane * LANE_STACK;
    uint wlen = 0;
    uint n = parent_slot == ROOT_CUT_SLOT ? 8u : min(parent_len, LANE_STACK);
    for (uint i = 0; i < n; ++i) {
        g_stack[base + wlen++] = parent_slot == ROOT_CUT_SLOT ? i : cut_pool[parent_slot * 64u + i];
    }
    bool has_far = t_far < FLT_MAX;
    uint olen = 0;
    uint out_base = out_slot * 64u;
    while (wlen > 0) {
        uint e = g_stack[base + --wlen];
        uint ni = e >> 3;
        uint s = e & 7u;
        FtNode nd = ft_nodes[ni];
        if (nd.bnode[s] == FT_INVALID) continue;
        float3 mn = float3(nd.mnx[s], nd.mny[s], nd.mnz[s]);
        float3 mx = float3(nd.mxx[s], nd.mxy[s], nd.mxz[s]);
        if (aabb_outside(f, mn, mx)) continue;
        if (point_aabb_max_dist(f.origin, mn, mx) <= t_ball) continue;
        if (has_far && point_aabb_dist(f.origin, mn, mx) >= t_far) continue;
        uint c = nd.child[s];
        if (c != FT_INVALID) {
            FtNode cn = ft_nodes[c];
            uint n_occ = 0;
            [unroll] for (uint cs = 0; cs < 8; ++cs) {
                if (cn.bnode[cs] != FT_INVALID) ++n_occ;
            }
            if (olen + wlen + n_occ <= LANE_STACK) {
                [unroll] for (uint cs2 = 0; cs2 < 8; ++cs2) {
                    if (cn.bnode[cs2] != FT_INVALID) {
                        g_stack[base + wlen++] = (c << 3) | cs2;
                    }
                }
                continue;
            }
        }
        cut_pool[out_base + olen++] = e;
    }
    return olen;
}

#endif // FTREE
