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
#define CTR_FRONTIER_HANDLES 21u // non-root opaque handles consumed (once/leaf record)
#define CTR_FRONTIER_RAYS    22u // primary ray samples reusing those handles
#define CTR_FRONTIER_ENTRIES 23u // summed software-frontier widths
#define CTR_SKY_PX       24u // stat: pixels inside proven-empty sky rects (zero rays)
#define CTR_RTGI_RAYS    25u // stat: real-time-GI bounce rays (shade_full's RTGI block;
                             // wavefront leaf units only — reference/DXR carry no counters)
#define CTR_COUNT        26u

// --- WIDTH_PROBE slots (FR_WIDTH=1 — trace::width_defs) --------------------
// Each kernel reports its COMPILED wave width (WaveGetLaneCount()) — the
// per-shader SIMD choice IGC makes from register pressure, i.e. the visible
// half of the occupancy story the FR_ABL probes measure behaviorally.
// DELIBERATELY >= CTR_COUNT: every zero loop (cs_seed / cs_seed_replay /
// cs_seed_probes) runs `i < CTR_COUNT`, and every gate readback reads
// CTR_COUNT*4 bytes — so these slots are never zeroed, never gated, and
// survive to the end-of-session readback BY CONSTRUCTION. LOCKSTEP with
// trace.rs's consts AND the counters buffer size (CTR_TOTAL * 4).
#define CTR_W_LEAF      26u // cs_leaf (leaf vs leaf-fb: whichever ran last)
#define CTR_W_SKY       27u // cs_sky
#define CTR_W_LEVEL     28u // cs_level / cs_level_wide (last writer wins)
#define CTR_W_HEMI      29u // cs_hemi_leaf
#define CTR_W_REFERENCE 30u // cs_reference (declares its OWN u3 view — below)
#define CTR_TOTAL       31u

// Compile units that paste this file bind `counters` — rt.hlsli's cutout
// loop keys its stat increment on this (the reference kernel and the DXR
// library never declare HAVE_COUNTERS and must not touch the GATED stat
// slots < CTR_COUNT; the WIDTH_PROBE exception lets the reference unit
// declare its own u3 view writing width slots >= CTR_COUNT ONLY, which no
// gate reads — the invariant this note protects is the stat slots, not the
// register).
#define HAVE_COUNTERS 1

RWStructuredBuffer<uint>  counters : register(u3);
RWStructuredBuffer<uint3> args     : register(u4);

// Per-dispatch push constants (b1); meanings are per-kernel.
cbuffer Push : register(b1) { uint push0; uint push1; uint push2; uint push3; }

// Flat work-item index for group-per-record and thread-per-record passes.
// prep writes 2D groups with x capped at 32768 (the 65,535 groups/dim
// limit), so flat = gy * 32768 + gx everywhere.
uint flat_group(uint3 gid) { return gid.y * 32768u + gid.x; }

// --- wave-aggregated counter bumps (revert: FR_ABL=nowave) -----------------
//
// Every queue slot and statistic here used to cost one global atomic PER
// THREAD, all landing on the same handful of addresses: `cs_level` gives each
// tile its own thread, and leaf.hlsl's hemi-point counter fires once per hit
// PIXEL (~2M to a single address at 1080p). Folding a wave into one atomic is
// Intel's own prescription — their RT developer guide (v4, p.26) says to keep
// data within a wave and use wave intrinsics rather than shared-memory
// traffic. VERDICT SPLIT (2026-08-01/09): THIS half — GLOBAL-memory counters,
// where contention is genuinely expensive — measured neutral and keeps its
// wave forms; the FRONTIER half (wavefront.hlsl's gw_alloc/gw_min_bits, whose
// targets are cheap on-chip GROUPSHARED words) measured a cross-vendor
// regression once the A/B was fully armed and reverted to plain atomics —
// see the gw block's header for the numbers and the FR_ABL=wavegw re-arm.
//
// NOT A BEHAVIOUR CHANGE. These are called at the same program points as the
// per-lane atomics they replace, so the aggregate covers exactly the lanes that
// would have raced there anyway; the totals published are identical and queue
// slot ORDER was already unspecified (the old atomics raced). WIDTH CAVEAT
// (2026-08-04): "WaveGetLaneCount() is 32 at every group width" was the
// TRIVIAL wave_probe's answer; FR_WIDTH's in-kernel report shows the REAL fat
// kernels compile SIMD16 on the B70 (leaf/hemi/reference — the per-shader
// pressure choice). Correctness here never depended on 32 — the intrinsics
// are width-agnostic — but do not cite this comment for occupancy math.
//
// `slot` MUST be wave-uniform — every caller passes a compile-time CTR_*
// constant or the `push1` cbuffer value. A per-lane slot would silently
// aggregate across different counters.

// Allocating bump: returns this lane's base, same total as the per-lane form.
// Pass n == 0 to participate without reserving (that is how callers keep the
// wave op out of divergent control flow).
uint ctr_add(uint slot, uint n) {
#if defined(ABL_NO_WAVE_OPS)
    uint w = 0;
    if (n != 0) InterlockedAdd(counters[slot], n, w);
    return w;
#else
    uint pre = WavePrefixSum(n);
    uint sum = WaveActiveSum(n);
    uint base = 0;
    if (WaveIsFirstLane() && sum != 0) InterlockedAdd(counters[slot], sum, base);
    // Only the first lane's `base` is meaningful; broadcast it to the wave.
    return WaveReadLaneFirst(base) + pre;
#endif
}

// Statistics bump: the result is never read, so this wants a count, not a
// prefix. Kept separate so the cheaper reduction is the one that gets used.
void ctr_bump(uint slot, bool take) {
#if defined(ABL_NO_WAVE_OPS)
    if (take) {
        uint s;
        InterlockedAdd(counters[slot], 1u, s);
    }
#else
    uint n = WaveActiveCountBits(take);
    if (WaveIsFirstLane() && n != 0) {
        uint s;
        InterlockedAdd(counters[slot], n, s);
    }
#endif
}
