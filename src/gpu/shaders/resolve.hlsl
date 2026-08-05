// accum -> RGBA16F HDR texture at 1/samples. The divide happens HERE, before
// the f16 texture, so long accumulations can't overflow half range; the
// tonemap PS then runs with inv_samples pinned to 1.0. Requires
// trace_common.hlsli pasted first.

RWStructuredBuffer<float> accum : register(u0);
RWTexture2D<float4> hdr : register(u14); // typed UAV -> descriptor table (base = NUM_UAVS)

#ifdef WAVEVIZ
// FR_WAVEVIZ: tbuf carries wave-ticket bits while the overlay is live (the
// covered kernels' LAST tbuf touch); this pass hashes ticket -> color and
// blends it over the scene so wave footprints are visible on the geometry
// they shaded. No wave ops here — safe at this unit's cs_6_3 floor (the DXR
// twin's target).
RWStructuredBuffer<float> tbuf : register(u1);

float3 wv_hash_color(uint t) {
    // pcg-style integer mix (sky.rs's pcg_mix shape) -> 3 channels in
    // [0.25, 1.0] so neighboring tickets land on clearly distinct hues and
    // nothing goes black except the sentinel.
    uint h = t * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    float3 c = float3(float(h & 1023u), float((h >> 10u) & 1023u),
                      float((h >> 20u) & 1023u)) * (1.0 / 1023.0);
    return 0.25 + 0.75 * c;
}
#endif

cbuffer Push : register(b1) { float inv_samples; uint _pp0; uint _pp1; uint _pp2; }

[numthreads(8, 8, 1)]
void cs_resolve(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint i3 = (id.y * rw + id.x) * 3u;
    float3 c = float3(accum[i3], accum[i3 + 1u], accum[i3 + 2u]) * inv_samples;
#ifdef WAVEVIZ
    if (flags & FLAG_WAVEVIZ) {
        uint t = asuint(tbuf[id.y * rw + id.x]);
        // 0xFFFFFFFE = the WAVEVIZ_CHS miss sentinel (no hit shader ran) —
        // darken instead of hashing so miss regions read as "no data".
        c = (t == 0xFFFFFFFEu) ? c * 0.15 : lerp(c, wv_hash_color(t), 0.65);
    }
#endif
    hdr[id.xy] = float4(c, 1.0);
}
