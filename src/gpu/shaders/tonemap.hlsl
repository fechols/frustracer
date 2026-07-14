// Tonemap linear-HDR radiance to the SDR backbuffer, replicating the CPU
// resolve exactly: c = accum * inv_samples; out = (1 - exp(-c))^(1/2.2).
// inv_samples is 1.0 when the source is a per-frame radiance image (DLSS-RR
// output) and 1/frame_count when it is the CPU accumulation sum.
Texture2D<float4> src : register(t0);
// The glare halo (gpu/bloom.rs level 0, half-res). Zero-strength when bloom is
// off, in which case this is never sampled and the arm below is skipped.
Texture2D<float4> glare : register(t1);
SamplerState samp_lin : register(s0); // linear, clamp

cbuffer Params : register(b0) {
    float inv_samples;
    float bloom_strength; // 0 = off
    float2 bloom_texel;   // 1 / glare dims (the tent's tap spacing)
};

float4 vsmain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

// 3x3 tent (1,2,1 / 2,4,2 / 1,2,1 over 16) — bloom.rs::tent. The tent is what
// keeps the glare core ROUND: a 2x2 box downsample has a square footprint, and a
// single bilinear tap would leave it looking like a rounded rectangle.
float3 tent(float2 uv) {
    float2 t = bloom_texel;
    float3 s = 0.0;
    s += glare.SampleLevel(samp_lin, uv + float2(-t.x, -t.y), 0).rgb * 1.0;
    s += glare.SampleLevel(samp_lin, uv + float2( 0.0, -t.y), 0).rgb * 2.0;
    s += glare.SampleLevel(samp_lin, uv + float2( t.x, -t.y), 0).rgb * 1.0;
    s += glare.SampleLevel(samp_lin, uv + float2(-t.x,  0.0), 0).rgb * 2.0;
    s += glare.SampleLevel(samp_lin, uv,                      0).rgb * 4.0;
    s += glare.SampleLevel(samp_lin, uv + float2( t.x,  0.0), 0).rgb * 2.0;
    s += glare.SampleLevel(samp_lin, uv + float2(-t.x,  t.y), 0).rgb * 1.0;
    s += glare.SampleLevel(samp_lin, uv + float2( 0.0,  t.y), 0).rgb * 2.0;
    s += glare.SampleLevel(samp_lin, uv + float2( t.x,  t.y), 0).rgb * 1.0;
    return s * (1.0 / 16.0);
}

float4 psmain(float4 pos : SV_Position) : SV_Target {
    float2 dims;
    src.GetDimensions(dims.x, dims.y);
    float3 c = src.Load(int3(pos.xy, 0)).rgb * inv_samples;
    if (bloom_strength > 0.0) {
        // Energy-conserving composite (bloom.rs): glare REDISTRIBUTES light, it
        // never adds any, so a uniformly lit frame comes back unchanged.
        float2 uv = (pos.xy + 0.5) / dims;
        c = lerp(c, tent(uv), bloom_strength);
    }
    float3 mapped = pow(saturate(1.0 - exp(-c)), 1.0 / 2.2);
    return float4(mapped, 1.0);
}
