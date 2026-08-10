// Software semantic prototype of an opaque traversal-frontier handle.
//
// A terminal beam probe mints one TraversalFrontier and every primary ray in
// that leaf tile (including all SPP samples) reuses it. The public shader seam
// deliberately does not expose a BVH node, cut-pool slot, or cut length: a
// future native provider may replace these opaque words with driver-owned
// traversal state without changing LeafRec or the ray call site.
//
// Software-provider lifetime:
//   * bound to this TraceGpu's exact acceleration-structure build;
//   * immutable and forkable after publication;
//   * valid through terminal fills and exact-basis structure replay;
//   * invalid after the next producing seed/reset or AS rebuild.
//
// t_start is intentionally not in this token. It is an independently proven
// empty-distance bound and remains valid when frontier allocation coarsens to
// an ancestor or falls all the way back to the root.

#ifndef FRUSTRACER_CONTINUATION_HLSLI
#define FRUSTRACER_CONTINUATION_HLSLI

#define ROOT_CUT_SLOT 0xffffffffu

// "FRC", provider ABI v1. A native backend can use another cookie/domain.
#define FRONTIER_COOKIE_V1 0x46524301u
#define FRONTIER_ROOT_TOKEN 0xffffffffu
#define FRONTIER_MAX_ENTRIES 64u

struct TraversalFrontier {
    uint2 opaque;
};

TraversalFrontier frontier_root() {
    TraversalFrontier h;
    h.opaque = uint2(FRONTIER_ROOT_TOKEN, FRONTIER_COOKIE_V1);
    return h;
}

// Producer/backend hook. v1 packs (pool slot, length) into a private token;
// callers outside this provider must not depend on that representation.
TraversalFrontier frontier_from_binary_cut(uint slot, uint len) {
    if (slot == ROOT_CUT_SLOT || len == 0u || len > FRONTIER_MAX_ENTRIES)
        return frontier_root();
    TraversalFrontier h;
    h.opaque = uint2((slot << 6u) | (len - 1u), FRONTIER_COOKIE_V1);
    return h;
}

bool frontier_backend_valid(TraversalFrontier h) {
    return h.opaque.y == FRONTIER_COOKIE_V1;
}

bool frontier_backend_is_root(TraversalFrontier h) {
    if (!frontier_backend_valid(h) || h.opaque.x == FRONTIER_ROOT_TOKEN)
        return true;
    // A corrupted/out-of-domain token is a conservative root request, never
    // an out-of-bounds arena read. Pool epochs are enforced by the host-side
    // producer/replay lifetime described above.
    return (h.opaque.x >> 6u) >= cap_cut;
}

uint frontier_backend_slot(TraversalFrontier h) {
    return h.opaque.x >> 6u;
}

uint frontier_backend_len(TraversalFrontier h) {
    return (h.opaque.x & 63u) + 1u;
}

// One call per terminal record, never per ray. Both software A/B arms pay the
// same three atomics (the root control adds zero), so telemetry does not give
// continuation traversal a cheaper timed path. Cost is independent of pixels
// and SPP.
//
// What is counted is frontiers CONSUMED, not frontiers minted. Without
// SW_RAYS_LEAF the leaf shader traverses from the root and never hands the
// token to a provider, so the contribution is zero BY CONSTRUCTION — which is
// what --check-gpu's "counters fired without the continuation lever" gate
// relies on. Producers do not know which arm is compiled: `level_finish`'s
// mixed splits (one child <= LEAF_TILE while a sibling is not, reachable when
// a parent extent is exactly 2*LEAF_TILE+1) mint a real frontier in every arm,
// so keying `active` on the token alone would let the root control report
// handles nothing ever resumed from. The decode work above the flag is left
// intact and only the contribution is zeroed, so both arms still execute the
// same three atomics and the same token math.
void frontier_record_reuse(TraversalFrontier h, uint ray_count) {
#ifdef SW_RAYS
    uint active = frontier_backend_is_root(h) ? 0u : 1u;
#ifndef SW_RAYS_LEAF
    active = 0u;
#endif
    uint _d;
    InterlockedAdd(counters[CTR_FRONTIER_HANDLES], active, _d);
    InterlockedAdd(counters[CTR_FRONTIER_RAYS], active * ray_count, _d);
    InterlockedAdd(counters[CTR_FRONTIER_ENTRIES],
                   active * frontier_backend_len(h), _d);
#endif
}

#endif // FRUSTRACER_CONTINUATION_HLSLI
