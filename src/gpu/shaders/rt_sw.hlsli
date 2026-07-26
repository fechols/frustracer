// bvh.rs's ray traversal on the GPU — the software twin of rt.hlsli, pasted
// IN ITS PLACE when --sw-rays is armed (trace.rs selects the file; rt.hlsli
// is untouched, so the off arm assembles the exact pre-lever source lists).
// Same three primitive signatures (trace_closest / occluded_q / transmit_q)
// over the already-uploaded software BVH (SwTreesGpu: bvh_nodes at t0,
// bvh.tri_idx at t1 — the buffers the frustum bound queries ride; rays were
// their missing consumer), plus trace_closest_frontier (leaf unit only) —
// bvh.rs::intersect_multi behind an opaque provider seam, which RayQuery
// cannot express: DXR accepts (o, d, TMin, TMax) and nothing else, so the
// quadtree's node cut never reached a GPU ray before this file.
// Requires trace_common.hlsli pasted first; the consuming unit must inject
// SW_TRAV_STACK (trace.rs, lockstep with bvh::TRAV_STACK and the build's
// max_depth hard assert). bvh.rs is the source of truth — every loop below
// is a term-for-term transliteration; fix bugs THERE first.
//
// No tri_of() remap anywhere: sw_tri holds scene triangle ids directly (the
// chunk remap is a hardware-BLAS artifact). No interval widening and no
// winner re-march under HEIGHTFIELD either: those exist in rt.hlsli because
// a RayQuery can only commit the PLANE t — here candidate_reject's marched
// (t', u', v') is tracked as the running best directly, exactly the CPU,
// whose soundness rides the build-time swept AABBs (grow_height_sweep); the
// box-vs-segment cull is exact for marched hits because an accepted t' lies
// inside (tmin, tmax) AND inside the swept box, so the ray∩box interval
// meets the segment by construction. This arm therefore has neither of
// rt.hlsli's relief known-accepts (plane-t mis-sort, interval widening).
//
// Stacks are per-lane LOCAL arrays (scratch/private memory): a 96-deep
// dynamically indexed stack can never live in registers, so scratch is the
// expected home — the ft_expand +58% lesson is about AVOIDABLE demotion of
// small arrays. Groupshared at 32 lanes x 96 x 8 B = 24 KB/group would
// newly LDS-cap the leaf kernel (VGPR-bound with zero LDS today);
// groupshared-vs-scratch is the documented sweep, not the default.
//
// slab_t NaN note: an axis-parallel ray whose origin plane coincides with a
// box face produces 0 * inf = NaN lanes, and HLSL min/max resolve them like
// glam's SIMD minps (the other operand wins) — both arms of the same-seed
// wavefront-vs-reference A/B compile this one function, so they agree with
// each other bitwise; CPU-vs-GPU stays the statistical gate class.

#ifndef HAVE_BVH_NODES // hemi_wave pastes frustum.hlsli, which declares these
struct BvhNode {
    float3 mn;
    uint left_first; // count == 0: children at left_first, left_first+1
    float3 mx;
    uint count;      // > 0: leaf over sw_tri[left_first..+count]
};
StructuredBuffer<BvhNode> bvh_nodes : register(t0);
#endif
// bvh.tri_idx — scene triangle ids in leaf slices (SwTreesGpu's t1; dead in
// every kernel until this file). Each tri lives in exactly ONE leaf slice
// (bvh.rs:880) — the transmittance product needs no NO_DUPLICATE_ANYHIT
// analog, unlike the RayQuery arm.
StructuredBuffer<uint> sw_tri : register(t1);

struct HitInfo {
    float t;
    uint tri;
    float u, v; // moller-trumbore == DXR convention: p = (1-u-v)p0 + u·p1 + v·p2
};

// bvh.rs::slab_t — entry t if the ray hits the box within (tmin, tmax), else
// the miss sentinel. FLT_MAX stands in for the CPU's +INF: every consumer
// compares `< tmax` / `< SW_MISS`, and no real entry distance reaches 3.4e38
// (scenes are diag ~10..70).
#define SW_MISS FLT_MAX

float sw_slab_t(float3 mn, float3 mx, float3 o, float3 inv_d, float tmin, float tmax) {
    float3 t1 = (mn - o) * inv_d;
    float3 t2 = (mx - o) * inv_d;
    float3 lo = min(t1, t2);
    float3 hi = max(t1, t2);
    float t_enter = max(max(lo.x, max(lo.y, lo.z)), tmin);
    float t_exit = min(min(hi.x, min(hi.y, hi.z)), tmax);
    return t_exit >= t_enter ? t_enter : SW_MISS;
}

// moller_trumbore's pure-intersection head (two-sided — |det| < 1e-10 is the
// only rejection; no backface cull; t <= 0 rejects). The relief/cutout tail
// is candidate_reject (trace_common.hlsli), called by the loops below at the
// exact point the CPU calls it. Reads the space1 position/index aliases —
// the space0 root SRVs belong to shade.hlsli, which pastes AFTER this file
// (and hemi_wave never pastes it at all).
bool sw_moller(uint tri, float3 o, float3 d, out float t, out float u, out float v) {
    t = 0.0; u = 0.0; v = 0.0;
    float3 v0 = uv_positions[uv_indices[tri * 3u]];
    float3 e1 = uv_positions[uv_indices[tri * 3u + 1u]] - v0;
    float3 e2 = uv_positions[uv_indices[tri * 3u + 2u]] - v0;
    float3 p = cross(d, e2);
    float det = dot(e1, p);
    if (abs(det) < 1e-10) return false;
    float inv = 1.0 / det;
    float3 s = o - v0;
    u = dot(s, p) * inv;
    if (u < 0.0 || u > 1.0) return false;
    float3 q = cross(s, e1);
    v = dot(d, q) * inv;
    if (v < 0.0 || u + v > 1.0) return false;
    t = dot(e2, q) * inv;
    return t > 0.0;
}

// Anti-vacuity stats (rt.hlsli's helpers verbatim): counted only by compile
// units that bind `counters` (ctr.hlsli pastes before this file in the
// wavefront kernels; the reference kernel never binds the slot).
void count_alpha_rej() {
#ifdef HAVE_COUNTERS
    uint _d;
    InterlockedAdd(counters[CTR_ALPHA_REJ], 1u, _d);
#endif
}

void count_height_rej() {
#ifdef HAVE_COUNTERS
    uint _d;
    InterlockedAdd(counters[CTR_HEIGHT_REJ], 1u, _d);
#endif
}

// bvh.rs::intersect_from — closest hit from subtree `start`, shrinking tmax.
// The stack carries each deferred far child's ENTRY DISTANCE beside its
// index: a node pushed when tmax was loose is often already beaten by the
// time it pops, and the stored distance kills it in one compare instead of
// one subtree re-descent (the CPU's comment, bvh.rs:647-652).
void sw_intersect_from(float3 o, float3 d, float3 inv_d, float tmin,
                       inout float tmax, uint start,
                       inout bool found, inout HitInfo best) {
    uint stk_n[SW_TRAV_STACK];
    float stk_d[SW_TRAV_STACK];
    uint sp = 0u;
    uint node_idx = start;
    for (;;) {
        BvhNode node = bvh_nodes[node_idx];
        if (node.count > 0u) {
            for (uint i = 0u; i < node.count; ++i) {
                uint tri = sw_tri[node.left_first + i];
                float tt, uu, vv;
                if (sw_moller(tri, o, d, tt, uu, vv)) {
                    uint rej = candidate_reject(tri, o, d, tt, uu, vv);
                    if (rej == 0u) {
                        if (tt > tmin && tt < tmax) {
                            tmax = tt;
                            found = true;
                            best.t = tt; best.tri = tri; best.u = uu; best.v = vv;
                        }
                    } else if (rej == 1u) {
                        count_alpha_rej();
                    } else {
                        count_height_rej();
                    }
                }
            }
            // Pop the nearest deferred node tmax has not already killed.
            for (;;) {
                if (sp == 0u) return;
                sp -= 1u;
                if (stk_d[sp] < tmax) { node_idx = stk_n[sp]; break; }
            }
        } else {
            uint l = node.left_first;
            BvhNode nl = bvh_nodes[l];
            BvhNode nr = bvh_nodes[l + 1u];
            float dl = sw_slab_t(nl.mn, nl.mx, o, inv_d, tmin, tmax);
            float dr = sw_slab_t(nr.mn, nr.mx, o, inv_d, tmin, tmax);
            uint near_i = l; uint far_i = l + 1u;
            float dnear = dl; float dfar = dr;
            if (!(dl <= dr)) { near_i = l + 1u; far_i = l; dnear = dr; dfar = dl; }
            if (dnear < SW_MISS) {
                node_idx = near_i;
                if (dfar < SW_MISS) {
                    // Capacity guaranteed by the build's max_depth <= TRAV_STACK
                    // hard assert (bvh.rs:401) — <= 1 push per level.
                    stk_n[sp] = far_i; stk_d[sp] = dfar; sp += 1u;
                }
            } else {
                for (;;) {
                    if (sp == 0u) return;
                    sp -= 1u;
                    if (stk_d[sp] < tmax) { node_idx = stk_n[sp]; break; }
                }
            }
        }
    }
}

// bvh.rs::occluded_from — any accepted hit in (tmin, tmax); index-only stack
// (no tmax to shrink), LEFTMOST-first descent verbatim (deliberately not
// near-first — the CPU's order, bvh.rs:780-794).
bool sw_occluded_from(float3 o, float3 d, float3 inv_d, float tmin, float tmax, uint start) {
    uint stk[SW_TRAV_STACK];
    uint sp = 0u;
    uint node_idx = start;
    for (;;) {
        BvhNode node = bvh_nodes[node_idx];
        if (node.count > 0u) {
            for (uint i = 0u; i < node.count; ++i) {
                uint tri = sw_tri[node.left_first + i];
                float tt, uu, vv;
                if (sw_moller(tri, o, d, tt, uu, vv)) {
                    uint rej = candidate_reject(tri, o, d, tt, uu, vv);
                    if (rej == 0u) {
                        if (tt > tmin && tt < tmax) return true;
                    } else if (rej == 1u) {
                        count_alpha_rej();
                    } else {
                        count_height_rej();
                    }
                }
            }
        } else {
            uint l = node.left_first;
            BvhNode nl = bvh_nodes[l];
            BvhNode nr = bvh_nodes[l + 1u];
            bool hl = sw_slab_t(nl.mn, nl.mx, o, inv_d, tmin, tmax) < SW_MISS;
            bool hr = sw_slab_t(nr.mn, nr.mx, o, inv_d, tmin, tmax) < SW_MISS;
            if (hl) {
                if (hr) { stk[sp] = l + 1u; sp += 1u; }
                node_idx = l;
                continue;
            }
            if (hr) { node_idx = l + 1u; continue; }
        }
        if (sp == 0u) return false;
        sp -= 1u;
        node_idx = stk[sp];
    }
}

bool trace_closest(float3 o, float3 d, float tmin, float tmax, out HitInfo h) {
    h = (HitInfo)0;
#ifdef SCENE_EMPTY
    return false;
#endif
    float3 inv_d = 1.0 / d; // Ray::new's d.recip() — ±inf on zero lanes
    bool found = false;
    sw_intersect_from(o, d, inv_d, tmin, tmax, 0u, found, h);
    return found;
}

bool occluded_q(float3 o, float3 d, float tmin, float tmax) {
#ifdef SCENE_EMPTY
    return false;
#endif
    if (tmax <= tmin) return false; // rt.hlsli:113 parity (degenerate segments)
    float3 inv_d = 1.0 / d;
    BvhNode root = bvh_nodes[0];
    if (sw_slab_t(root.mn, root.mx, o, inv_d, tmin, tmax) >= SW_MISS) return false;
    return sw_occluded_from(o, d, inv_d, tmin, tmax, 0u);
}

// Light-transport occlusion (bvh.rs::transmittance): RGB throughput of the
// segment — ONE clear, ZERO on any opaque hit, the mat_shadow tint product
// per transmissive interface crossed. Without TRANS_SHADOW this is
// occluded_q verbatim — the structural bit-identity arm (the
// !any_transmissive branch on the CPU).

#ifdef TRANS_SHADOW

void count_trans_pass() {
#ifdef HAVE_COUNTERS
    uint _d;
    InterlockedAdd(counters[CTR_TRANS_PASS], 1u, _d);
#endif
}

// bvh.rs::transmittance_from — occluded_from's traversal with the tinted
// leaf test: opaque (mat_shadow.a == 0) returns true (=> ZERO at the
// caller); a transmissive interface multiplies tp and traversal CONTINUES;
// tp decaying under SHADOW_TP_MIN counts as opaque (the deep-glass bound).
bool sw_transmit_from(float3 o, float3 d, float3 inv_d, float tmin, float tmax,
                      uint start, inout float3 tp) {
    uint stk[SW_TRAV_STACK];
    uint sp = 0u;
    uint node_idx = start;
    for (;;) {
        BvhNode node = bvh_nodes[node_idx];
        if (node.count > 0u) {
            for (uint i = 0u; i < node.count; ++i) {
                uint tri = sw_tri[node.left_first + i];
                float tt, uu, vv;
                if (sw_moller(tri, o, d, tt, uu, vv)) {
                    uint rej = candidate_reject(tri, o, d, tt, uu, vv);
                    if (rej == 0u) {
                        if (tt > tmin && tt < tmax) {
                            float4 ms = mat_shadow[uv_tri_mat[tri]];
                            if (ms.a <= 0.0) return true;
                            tp *= ms.rgb;
                            count_trans_pass();
                            if (max(tp.x, max(tp.y, tp.z)) < SHADOW_TP_MIN)
                                return true;
                        }
                    } else if (rej == 1u) {
                        count_alpha_rej();
                    } else {
                        count_height_rej();
                    }
                }
            }
        } else {
            uint l = node.left_first;
            BvhNode nl = bvh_nodes[l];
            BvhNode nr = bvh_nodes[l + 1u];
            bool hl = sw_slab_t(nl.mn, nl.mx, o, inv_d, tmin, tmax) < SW_MISS;
            bool hr = sw_slab_t(nr.mn, nr.mx, o, inv_d, tmin, tmax) < SW_MISS;
            if (hl) {
                if (hr) { stk[sp] = l + 1u; sp += 1u; }
                node_idx = l;
                continue;
            }
            if (hr) { node_idx = l + 1u; continue; }
        }
        if (sp == 0u) return false;
        sp -= 1u;
        node_idx = stk[sp];
    }
}

float3 transmit_q(float3 o, float3 d, float tmin, float tmax) {
#ifdef SCENE_EMPTY
    return float3(1.0, 1.0, 1.0);
#endif
    if (tmax <= tmin) return float3(1.0, 1.0, 1.0);
    float3 inv_d = 1.0 / d;
    BvhNode root = bvh_nodes[0];
    if (sw_slab_t(root.mn, root.mx, o, inv_d, tmin, tmax) >= SW_MISS)
        return float3(1.0, 1.0, 1.0);
    float3 tp = float3(1.0, 1.0, 1.0);
    if (sw_transmit_from(o, d, inv_d, tmin, tmax, 0u, tp))
        return float3(0.0, 0.0, 0.0);
    return tp;
}

#else

float3 transmit_q(float3 o, float3 d, float tmin, float tmax) {
    return occluded_q(o, d, tmin, tmax)
        ? float3(0.0, 0.0, 0.0)
        : float3(1.0, 1.0, 1.0);
}

#endif // TRANS_SHADOW

// Software provider for the opaque traversal-frontier API. This is
// bvh.rs::intersect_multi — closest hit seeded from a tile's node cut instead
// of the root. Every leaf-pixel sample lies inside the tile frustum, so the
// refine_cut output at the tile's own tc covers everything its ancestors did
// not cull. Leaf unit only: TraversalFrontier/cut_pool come from
// continuation+queues, pasted there.
//
// v1 iterates roots in POOL ORDER with the per-root slab gate + the running
// tmax shrinking across roots. Every root is visited — the bvh.rs:629
// soundness tail argument (dropping a root drops geometry -> false sky);
// order is PERF-only. The CPU's ascending slab_t sort (bvh.rs:611-628,
// measured worth 5.4% on San Miguel CPU-side) is the documented follow-up.
#ifdef SW_RAYS_LEAF
bool trace_closest_frontier(TraversalFrontier frontier,
                            float3 o, float3 d, float tmin, float tmax,
                            out HitInfo h) {
    h = (HitInfo)0;
#ifdef SCENE_EMPTY
    return false;
#endif
    // Invalid provider/cookie and explicit root both degrade conservatively.
    if (frontier_backend_is_root(frontier))
        return trace_closest(o, d, tmin, tmax, h);
    uint cut_slot = frontier_backend_slot(frontier);
    uint cut_len = frontier_backend_len(frontier);
    float3 inv_d = 1.0 / d;
    bool found = false;
    for (uint r = 0u; r < cut_len; ++r) {
        uint root = cut_pool[cut_slot * 64u + r];
        BvhNode rn = bvh_nodes[root];
        if (sw_slab_t(rn.mn, rn.mx, o, inv_d, tmin, tmax) < SW_MISS)
            sw_intersect_from(o, d, inv_d, tmin, tmax, root, found, h);
    }
    return found;
}
#endif // SW_RAYS_LEAF
