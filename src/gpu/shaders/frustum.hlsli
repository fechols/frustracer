// frustum.rs on the GPU: tile frustums, the conservative nearest-distance
// bound query, advance_tc, and refine_cut — the software-BVH half of the
// tracer (RT cores can't answer frustum queries; see CLAUDE.md invariants).
// Requires trace_common.hlsli pasted first. Used by the level kernel
// (32 threads/group, one thread per tile).
//
// Ports are term-for-term transliterations of frustum.rs; every conservative
// slack (aabb_outside's 1e-5 outward push, advance_tc's 1e-4 shave) carries
// over unchanged. `precise` on the accumulating plane/distance math keeps
// DXC's fast-math from re-associating the slacks away.
//
// The CPU `visit` is recursive nearer-child-first; here it's an explicit
// stack in groupshared, partitioned per lane (32 x 64 x 4 B = 8 KB/group).
// Visit ORDER differs from the CPU (correctness is order-independent; node
// counts differ). The same per-lane slab is reused by refine_cut's work
// stack — the two phases never overlap in time.

struct BvhNode {
    float3 mn;
    uint left_first; // count == 0: children at left_first, left_first+1
    float3 mx;
    uint count;      // > 0: leaf over tri_idx[left_first..+count]
};
StructuredBuffer<BvhNode> bvh_nodes : register(t0);

#define LANE_STACK 64u
groupshared uint g_stack[32 * LANE_STACK];

// Screen-tile frustum: apex + 4 inward unit normals through it.
struct TF {
    float3 origin;
    float3 nrm[4];
};

float3 frustum_plane(float3 a, float3 b, float3 center) {
    precise float3 n = cross(a, b);
    n = dot(n, center) < 0.0 ? -n : n;
    // Degenerate (near-parallel corner rays) -> zero normal -> never culls.
    return normalize_or_zero(n);
}

// camera.rs::tile_frustum: planes through the tile's continuous pixel-grid
// EDGES (footprints, not centers) so jittered samples stay inside.
TF tile_frustum(uint x0, uint y0, uint x1, uint y1) {
    float3 c0 = ray_dir(float(x0), float(y0));
    float3 c1 = ray_dir(float(x1), float(y0));
    float3 c2 = ray_dir(float(x1), float(y1));
    float3 c3 = ray_dir(float(x0), float(y1));
    float3 center = c0 + c1 + c2 + c3;
    TF f;
    f.origin = cam_origin.xyz;
    f.nrm[0] = frustum_plane(c0, c1, center);
    f.nrm[1] = frustum_plane(c1, c2, center);
    f.nrm[2] = frustum_plane(c2, c3, center);
    f.nrm[3] = frustum_plane(c3, c0, center);
    return f;
}

// frustum.rs::tri_cell — a spherical-triangle hemisphere cell: 3 planes
// through the apex; the 4th is zero (a zero normal never culls).
TF tri_cell_frustum(float3 origin, float3 a, float3 b, float3 c) {
    float3 center = a + b + c;
    TF f;
    f.origin = origin;
    f.nrm[0] = frustum_plane(a, b, center);
    f.nrm[1] = frustum_plane(b, c, center);
    f.nrm[2] = frustum_plane(c, a, center);
    f.nrm[3] = float3(0.0, 0.0, 0.0);
    return f;
}

// frustum.rs::half_space — the hemisphere root (the surface tangent plane).
TF half_space_frustum(float3 origin, float3 n) {
    TF f;
    f.origin = origin;
    f.nrm[0] = n;
    f.nrm[1] = float3(0.0, 0.0, 0.0);
    f.nrm[2] = float3(0.0, 0.0, 0.0);
    f.nrm[3] = float3(0.0, 0.0, 0.0);
    return f;
}

// Conservative: true only if the box is fully outside some side plane.
bool aabb_outside(TF f, float3 mn, float3 mx) {
    [unroll] for (uint i = 0; i < 4; ++i) {
        float3 n = f.nrm[i];
        // positive vertex: the corner farthest along the plane normal
        float3 pv = select(n >= 0.0, mx, mn);
        precise float3 rel = pv - f.origin;
        float3 ar = abs(rel);
        precise float eps = 1e-5 * (1.0 + max(ar.x, max(ar.y, ar.z)));
        precise float d = dot(n, rel);
        if (d < -eps) return true;
    }
    return false;
}

float point_aabb_dist(float3 p, float3 mn, float3 mx) {
    precise float3 v = clamp(p, mn, mx) - p;
    return length(v);
}

float point_aabb_max_dist(float3 p, float3 mn, float3 mx) {
    precise float3 v = max(abs(p - mn), abs(p - mx));
    return length(v);
}

uint cut_node(uint slot, uint i) {
    return slot == ROOT_CUT_SLOT ? 0u : cut_pool[slot * 64u + i];
}

// frustum.rs::nearest_geometry_distance(_within): the conservative
// nearest-possible-hit bound over the inherited cut. Seeds best = t_limit;
// "no geometry" (sky when t_limit is the FLT_MAX sentinel) <=> returns
// exactly t_limit. Never a planar near clip: nodes are skipped only when
// fully inside ball(origin, t_start); candidates clamp UP to t_start.
float bound_query(TF f, float t_start, float t_limit, uint cut_slot, uint cut_len, uint lane) {
    float best = t_limit;
    uint base = lane * LANE_STACK;
    for (uint r = 0; r < cut_len; ++r) {
        uint sp = 0;
        g_stack[base + sp++] = cut_node(cut_slot, r);
        while (sp > 0) {
            uint idx = g_stack[base + --sp];
            BvhNode node = bvh_nodes[idx];
            if (aabb_outside(f, node.mn, node.mx)) continue;
            if (point_aabb_max_dist(f.origin, node.mn, node.mx) <= t_start) continue;
            float d = max(point_aabb_dist(f.origin, node.mn, node.mx), t_start);
            if (d >= best) continue;
            if (node.count > 0) {
                // Leaf: the box distance lower-bounds its triangles.
                best = d;
                continue;
            }
            if (sp + 2 > LANE_STACK) {
                // Stack pressure (a BVH chain deeper than the slab): take
                // the node's own distance as if it were a leaf — a valid
                // lower bound for everything inside, refine_cut's
                // coarse-emit analog. Overflowing instead would corrupt the
                // neighboring lane's stack (the CPU visit recurses on the
                // heap and has no such limit).
                best = d;
                continue;
            }
            uint l = node.left_first;
            BvhNode ln = bvh_nodes[l];
            BvhNode rn = bvh_nodes[l + 1];
            float dl = point_aabb_dist(f.origin, ln.mn, ln.mx);
            float dr = point_aabb_dist(f.origin, rn.mn, rn.mx);
            // Push farther first so the nearer child pops first (tightens
            // `best` early, same intent as the CPU's ordered recursion).
            if (dl <= dr) {
                g_stack[base + sp++] = l + 1;
                g_stack[base + sp++] = l;
            } else {
                g_stack[base + sp++] = l;
                g_stack[base + sp++] = l + 1;
            }
        }
    }
    return best;
}

// frustum.rs::refine_cut. Writes surviving node ids straight into
// cut_pool[out_slot]; returns the length. Invariant out+work <= 64: an
// internal node splits only while the budget allows, else it is emitted
// coarsely — NEVER dropped, and never distance-to-best pruned. `t_far` is
// the clamped-consumer drop (sound only when every bound query on the cut
// uses t_limit <= t_far and every ray tmax <= t_far); the primary path and
// GI pass FLT_MAX, which never drops; hemi AO passes ao_radius.
uint refine_cut(TF f, float t_ball, float t_far, uint parent_slot, uint parent_len, uint out_slot, uint lane) {
    uint base = lane * LANE_STACK;
    uint wlen = min(parent_len, LANE_STACK);
    for (uint i = 0; i < wlen; ++i) {
        g_stack[base + i] = cut_node(parent_slot, i);
    }
    bool has_far = t_far < FLT_MAX;
    uint olen = 0;
    uint out_base = out_slot * 64u;
    while (wlen > 0) {
        uint idx = g_stack[base + --wlen];
        BvhNode node = bvh_nodes[idx];
        if (aabb_outside(f, node.mn, node.mx)) continue;
        if (point_aabb_max_dist(f.origin, node.mn, node.mx) <= t_ball) continue;
        if (has_far && point_aabb_dist(f.origin, node.mn, node.mx) >= t_far) continue;
        if (node.count == 0 && olen + wlen + 2 <= LANE_STACK) {
            g_stack[base + wlen++] = node.left_first;
            g_stack[base + wlen++] = node.left_first + 1;
        } else {
            cut_pool[out_base + olen++] = idx;
        }
    }
    return olen;
}
