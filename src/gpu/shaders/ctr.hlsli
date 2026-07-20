// Counters + indirect-args + per-dispatch push constants — the scaffolding
// every wavefront kernel shares (primary quadtree AND hemisphere passes).
// Requires trace_common.hlsli pasted first.

// counters[] slot assignments
#define CTR_TILE_A       0u
#define CTR_TILE_B       1u
#define CTR_LEAF         2u
#define CTR_SKY          3u
#define CTR_CUT          4u  // primary cut-pool bump allocator
#define CTR_OVERFLOW     5u  // failed queue enqueue — gated == 0 (structural sizing)
#define CTR_CUT_FALLBACK 6u  // cut pool exhausted -> parent slot reused (reported, sound)
#define CTR_SPLIT        7u  // stat: tiles split
#define CTR_BLOCKED      8u  // stat: blocked (unadvanced) splits
#define CTR_HEMI_PT      9u  // hemi shading-point queue count (frame-wide)
#define CTR_HEMI_A       10u // hemi cell ping-pong counts (per batch)
#define CTR_HEMI_B       11u
#define CTR_HEMI_LEAF    12u // hemi leaf-cell count (per batch)
#define CTR_HEMI_CUT     13u // hemi cut-pool bump (per batch)
#define CTR_HEMI_EMPTY   14u // stat: hemi cells resolved analytically
#define CTR_HEMI_RAYS    15u // stat: hemi leaf rays
#define CTR_V_FALSE_EMPTY 16u // verify: empty-cell claims contradicted by rays
#define CTR_V_TMIN        17u // verify: leaf-ray tmin skipped real geometry
#define CTR_ALPHA_REJ     18u // stat: alpha-cutout candidate rejections (ALPHA_CUTOUT scenes)
#define CTR_HEIGHT_REJ    19u // stat: relief-march candidate rejections (HEIGHTFIELD scenes)
#define CTR_TRANS_PASS    20u // stat: tinted-shadow candidate passes (TRANS_SHADOW scenes)
#define CTR_COUNT        24u

// Compile units that paste this file bind `counters` — rt.hlsli's cutout
// loop keys its stat increment on this (the reference kernel and the DXR
// library never declare the buffer and must not touch the slot).
#define HAVE_COUNTERS 1

RWStructuredBuffer<uint>  counters : register(u3);
RWStructuredBuffer<uint3> args     : register(u4);

// Per-dispatch push constants (b1); meanings are per-kernel.
cbuffer Push : register(b1) { uint push0; uint push1; uint push2; uint push3; }

// Flat work-item index for group-per-record and thread-per-record passes.
// prep writes 2D groups with x capped at 32768 (the 65,535 groups/dim
// limit), so flat = gy * 32768 + gx everywhere.
uint flat_group(uint3 gid) { return gid.y * 32768u + gid.x; }
