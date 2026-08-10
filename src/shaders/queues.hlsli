// Primary-quadtree queue records + per-pixel planes. Requires
// trace_common.hlsli + ctr.hlsli + continuation.hlsli pasted first.
//
// Records are fixed-size self-contained POD (work-graph-shaped: a future
// SM 6.8 backend consumes these same structs as node records; the counters/
// prep-args scaffolding is exactly what it would delete). Queue counters are
// read for slot assignment and dispatch sizing only — never for semantics.

// A tile awaiting its bound query + refine (internal tiles only — leaf/sky
// classification happens at enqueue). 24 bytes.
struct TileRec {
    uint xy0;      // x0 | y0 << 16
    uint xy1;      // x1 | y1 << 16
    float t_start; // inherited proven-empty distance (== ray tmin)
    uint cut_slot; // cut-pool slot; 0xffffffff = the root cut [0]
    uint meta;     // cut_len | depth << 8
    uint path;     // 2 bits/level quadrant path (TL=0 TR=1 BL=2 BR=3)
};

// A leaf tile (both dims <= LEAF_TILE): per-pixel rays with TMin = t_start.
// The opaque frontier rides along for --continuation-rays / --sw-rays.
// SW_RAYS_LEAF consumes it through trace_closest_frontier; RayQuery arms
// ignore it because current DXR exposes no traversal-resume input. Under
// FTREE the producer translates wide slots to the binary ray-BVH domain
// before minting the token. 24 bytes (one struct, no lever-forked stride;
// trace.rs::LEAF_REC_BYTES and main.rs's readback move in lockstep).
struct LeafRec {
    uint xy0;
    uint xy1;
    float t_start;
    uint depth;
    TraversalFrontier frontier;
};

// A tile whose whole frustum was proven empty. 16 bytes.
struct SkyRec {
    uint xy0;
    uint xy1;
    uint depth;
    uint _pad;
};

// Hemisphere shading point, appended by the leaf pass when fb_mode > 0;
// consumed by the hemi wavefront (hemi.hlsli units bind the same buffer).
// 32 bytes. o = hit + n·eps (the secondary-ray origin); n = the exact
// shading normal (the octant frame and PSA axis need it unquantized).
struct HemiPointRec {
    float3 o;
    uint pixel; // y * rw + x
    float3 n;
    uint _pad;
};

// Per-pixel planes.
RWStructuredBuffer<float> accum : register(u0);
RWStructuredBuffer<float> tbuf  : register(u1);
RWStructuredBuffer<uint>  info  : register(u2);

// SKY_UNIT (sky.hlsl) takes u5 for the cloud lattice instead: the tile queues
// are dead by the time the sky fill runs, and each compile unit re-declares
// what its registers mean in that phase (the codebase idiom — see the header
// on `TRACE_COMMON_HLSLI`'s concat sites). The leaf kernel keeps these
// declarations and simply never touches them.
#ifndef SKY_UNIT
RWStructuredBuffer<TileRec> qin      : register(u5);
RWStructuredBuffer<TileRec> qout     : register(u6);
#endif
RWStructuredBuffer<LeafRec> qleaf    : register(u7);
RWStructuredBuffer<SkyRec>  qsky     : register(u8);
RWStructuredBuffer<uint>    cut_pool : register(u9); // 64 x u32 slots

// Compose inputs: `partial` = this frame's ambient-free color (or the full
// color when fb is off / for sky), `ambw` = the primary kd weight the hemi
// ambient multiplies (0 when fb off), `hbuf` = the hemisphere accumulator
// (4 x u32 fixed-point 2^18: AO mass / GI rgb + verify PSA in .w).
RWStructuredBuffer<float> partial : register(u10);
RWStructuredBuffer<float> ambw    : register(u11);

// THE fb-OFF SPLAT. With no hemisphere pass there is no ambient term to fold
// in later, so `compose` degenerates to `accum[i] = partial[i]` — a
// full-screen 12 B read + 12 B write, its own dispatch and its own barrier,
// moving data between two buffers for nothing (and `ambw` is 12 B/px of zeros
// that compose.hlsl already declines to read). So leaf and sky write accum
// DIRECTLY and trace.rs does not dispatch compose at all.
//
// This is compose.hlsl's store-or-add rule verbatim, and it lives here so the
// two cannot drift. BIT-IDENTICAL to going through partial: an f32 store
// followed by an f32 load returns the same bits, which is what the replay
// gate's zero-tolerance accum compare confirms.
//
// Safe for the same reason compose's own += is: leaf and sky rects partition
// the screen exactly (the --check-gpu exactly-once coverage gate), so every
// pixel has a single writer and the read-modify-write needs no atomic.
// reference.hlsl has always written accum this way.
void accum_splat(uint pi, float3 c) {
    uint i3 = pi * 3u;
    if (frame == 0u || (flags & FLAG_ACCUM) == 0u) {
        accum[i3 + 0u] = c.x;
        accum[i3 + 1u] = c.y;
        accum[i3 + 2u] = c.z;
    } else {
        accum[i3 + 0u] += c.x;
        accum[i3 + 1u] += c.y;
        accum[i3 + 2u] += c.z;
    }
}
RWStructuredBuffer<uint>  hbuf    : register(u12);
RWStructuredBuffer<HemiPointRec> hemi_pts : register(u13);

#define HEMI_FIXED 262144.0 // 2^18 fixed-point scale for hbuf

uint2 rect_min(uint xy0) { return uint2(xy0 & 0xffffu, xy0 >> 16); }
uint2 rect_max(uint xy1) { return uint2(xy1 & 0xffffu, xy1 >> 16); }
uint pack_xy(uint x, uint y) { return x | (y << 16); }
