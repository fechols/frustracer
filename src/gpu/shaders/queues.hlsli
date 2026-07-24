// Primary-quadtree queue records + per-pixel planes. Requires
// trace_common.hlsli + ctr.hlsli pasted first.
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
// The cut rides along for --sw-rays (SW_RAYS_LEAF: trace_closest_multi seeds
// traversal from these BINARY node ids — under FTREE the emitter translates
// the slot-ref cut via ft_bnode at enqueue); RayQuery arms ignore it —
// hardware traversal from the root subsumes cut-seeding there and the
// inherited ball claim rides in t_start alone. 24 bytes (one struct, no
// lever-forked stride — the fields are always written, read only under the
// lever; trace.rs::LEAF_REC_BYTES and main.rs's readback move in lockstep).
struct LeafRec {
    uint xy0;
    uint xy1;
    float t_start;
    uint depth;
    uint cut_slot; // ROOT_CUT_SLOT = traverse from the binary root
    uint cut_len;
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
RWStructuredBuffer<uint>  hbuf    : register(u12);
RWStructuredBuffer<HemiPointRec> hemi_pts : register(u13);

#define ROOT_CUT_SLOT 0xffffffffu
#define HEMI_FIXED 262144.0 // 2^18 fixed-point scale for hbuf

uint2 rect_min(uint xy0) { return uint2(xy0 & 0xffffu, xy0 >> 16); }
uint2 rect_max(uint xy1) { return uint2(xy1 & 0xffffu, xy1 >> 16); }
uint pack_xy(uint x, uint y) { return x | (y << 16); }
