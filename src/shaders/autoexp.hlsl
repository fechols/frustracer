// Auto-exposure's luminance meter (gpu/autoexp.rs; src/autoexp.rs is the
// consumer and carries the rationale): mean log2-luminance of the tonemap
// source, i.e. the geometric mean the EV controller keys on. A LOG mean on
// purpose — this renderer's sun disc is ~44,000 radiance against a ~1.0 scene,
// and a linear mean (the bloom pyramid's coarsest mip) would be nothing BUT
// the sun; in log space those pixels contribute a bounded ~15 stops each.
//
// fxc cs_5_0 like the rest of the presentation stage (no DXC dependency — the
// meter must exist in CPU-renderer/plain sessions too), so no wave intrinsics:
// a classic two-level groupshared tree reduction, no atomics — each group owns
// its output record, kernel B folds them alone.
//
// The 1e-6 luma floor and the 0.2126/0.7152/0.0722 Rec.709 weights mirror
// autoexp::log2_luma term-for-term (the clouds-wind literal idiom; the
// self-test's uniform-image anchor is the cross-check).

Texture2D<float4> src : register(t0);
RWStructuredBuffer<float> tiles : register(u0);  // cs_reduce out / cs_fold in
RWStructuredBuffer<float> result : register(u1); // [0] = mean log2 luma

cbuffer Params : register(b0) {
    uint src_w;
    uint src_h;
    float inv_samples; // 1/frame_count on the accumulation arm, else 1.0
    uint tiles_x;      // groups per row (cs_reduce's flat-id pitch)
    uint ntiles;       // total groups (cs_fold's fold length)
    uint _p0;
    uint _p1;
    uint _p2;
};

groupshared float s[256];

// One group per 64x64-pixel tile; each of the 16x16 threads accumulates its
// strided 4x4 pixels, then a tree reduction writes one partial sum per tile.
// Out-of-bounds pixels contribute 0 (cs_fold divides by the true pixel count,
// so edge tiles are simply lighter records, not a bias).
[numthreads(16, 16, 1)]
void cs_reduce(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gi : SV_GroupIndex) {
    float acc = 0.0;
    uint bx = gid.x * 64u;
    uint by = gid.y * 64u;
    [unroll]
    for (uint dy = 0; dy < 4; ++dy) {
        [unroll]
        for (uint dx = 0; dx < 4; ++dx) {
            uint x = bx + gtid.x + dx * 16u;
            uint y = by + gtid.y + dy * 16u;
            if (x < src_w && y < src_h) {
                float3 c = max(src.Load(int3(x, y, 0)).rgb * inv_samples, 0.0);
                float l = dot(c, float3(0.2126, 0.7152, 0.0722));
                acc += log2(max(l, 1e-6));
            }
        }
    }
    s[gi] = acc;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint k = 128u; k > 0u; k >>= 1u) {
        if (gi < k) s[gi] += s[gi + k];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0) tiles[gid.y * tiles_x + gid.x] = s[0];
}

// One group folds every tile record and divides by the pixel count.
[numthreads(256, 1, 1)]
void cs_fold(uint gi : SV_GroupIndex) {
    float acc = 0.0;
    for (uint i = gi; i < ntiles; i += 256u) acc += tiles[i];
    s[gi] = acc;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint k = 128u; k > 0u; k >>= 1u) {
        if (gi < k) s[gi] += s[gi + k];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0) result[0] = s[0] / float(src_w * src_h);
}
