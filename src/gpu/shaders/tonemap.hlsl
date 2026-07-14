// The presentation curve on the GPU — a term-for-term port of `tone::map`
// (src/tone.rs), which is the single source of truth. Gated against it by
// --check-gpu M12 at <= 1 f16 ulp; the Rust side is gated closed-form by
// --check's `tone::self_test`. Do not "improve" the math here alone.
//
//   f(x) = x                                            for x <= knee
//   f(x) = knee + band * (1 - exp(-(x - knee) / band))  for x >  knee,  band = headroom - knee
//
// SDR is the degenerate case (knee 0, headroom 1, gamma on) and reduces to the
// pre-HDR curve `(1 - exp(-c))^(1/2.2)` exactly — which is why there is no
// separate SDR shader.
//
// inv_samples is 1.0 when the source is a per-frame radiance image (every
// upscaler output, the GPU tracer, DXR) and 1/frame_count when it is the CPU
// accumulation sum (present_hdr).
Texture2D<float4> src : register(t0);

cbuffer Params : register(b0) {
    float inv_samples;
    float knee;      // rolloff start, in paper-white units
    float headroom;  // asymptote = peak_nits / paper_white; 1.0 on SDR
    float scale;     // scRGB: paper_white / 80. SDR: 1.0
    float gamma_on;  // 1.0 only for the 8-bit UNORM swapchain; scRGB is linear
};

float4 vsmain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float curve(float x) {
    if (x <= knee) return x;
    float band = headroom - knee;
    return knee + band * (1.0 - exp(-(x - knee) / band));
}

float4 psmain(float4 pos : SV_Position) : SV_Target {
    // max(0) mirrors tone::map's clamp: a negative radiance would take the
    // below-knee arm and then reach pow() with a negative base (NaN).
    float3 c = max(src.Load(int3(pos.xy, 0)).rgb * inv_samples, 0.0);
    float3 f = float3(curve(c.r), curve(c.g), curve(c.b));
    if (gamma_on > 0.5) f = pow(f, 1.0 / 2.2);
    // No saturate: under scRGB, values above 1.0 are legal and ARE the highlight
    // headroom. The curve is bounded by `headroom` on its own.
    return float4(f * scale, 1.0);
}
