// accum -> RGBA16F HDR texture at 1/samples. The divide happens HERE, before
// the f16 texture, so long accumulations can't overflow half range; the
// tonemap PS then runs with inv_samples pinned to 1.0. Requires
// trace_common.hlsli pasted first.

RWStructuredBuffer<float> accum : register(u0);
RWTexture2D<float4> hdr : register(u14); // typed UAV -> descriptor table (base = NUM_UAVS)

cbuffer Push : register(b1) { float inv_samples; uint _pp0; uint _pp1; uint _pp2; }

[numthreads(8, 8, 1)]
void cs_resolve(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint i3 = (id.y * rw + id.x) * 3u;
    float3 c = float3(accum[i3], accum[i3 + 1u], accum[i3 + 2u]) * inv_samples;
    hdr[id.xy] = float4(c, 1.0);
}
