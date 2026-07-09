// Tonemap linear-HDR radiance to the SDR backbuffer, replicating the CPU
// resolve exactly: c = accum * inv_samples; out = (1 - exp(-c))^(1/2.2).
// inv_samples is 1.0 when the source is a per-frame radiance image (DLSS-RR
// output) and 1/frame_count when it is the CPU accumulation sum.
Texture2D<float4> src : register(t0);

cbuffer Params : register(b0) {
    float inv_samples;
    float3 _pad;
};

float4 vsmain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 psmain(float4 pos : SV_Position) : SV_Target {
    float3 c = src.Load(int3(pos.xy, 0)).rgb * inv_samples;
    float3 mapped = pow(saturate(1.0 - exp(-c)), 1.0 / 2.2);
    return float4(mapped, 1.0);
}
