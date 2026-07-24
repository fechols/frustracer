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
// The glare halo (gpu/bloom.rs level 0, half-res). Zero-strength when bloom is
// off, in which case this is never sampled and the arm below is skipped.
Texture2D<float4> glare : register(t1);
SamplerState samp_lin : register(s0); // linear, clamp

// Field order is the root-constant DWORD order tonemap.rs writes; the float2
// sits at offset 8 so it cannot straddle a 16-byte boundary (the fsr_composite
// bug). Eight contiguous DWORDs.
cbuffer Params : register(b0) {
    float inv_samples;
    float bloom_strength; // 0 = off
    float2 bloom_texel;   // 1 / glare dims (the tent's tap spacing)
    float knee;      // rolloff start, in paper-white units
    float headroom;  // asymptote = peak_nits / paper_white; 1.0 on SDR
    float scale;     // scRGB: paper_white / 80. SDR: 1.0. HDR10: paper_white / 10000
    float mode;      // 0 = scRGB linear, 1 = 8-bit gamma 2.2, 2 = HDR10 PQ (tone::ToneMode)
};

// Rec.709 -> Rec.2020 primaries. Literals mirror tone::m709_to_2020
// term-for-term; change both together.
float3 m709_to_2020(float3 v) {
    return float3(
        0.627404 * v.r + 0.329283 * v.g + 0.043313 * v.b,
        0.069097 * v.r + 0.919540 * v.g + 0.011362 * v.b,
        0.016391 * v.r + 0.088013 * v.g + 0.895595 * v.b);
}

// SMPTE ST 2084 inverse EOTF (luminance normalized to 10000 nits -> PQ
// signal). Literals mirror tone::pq_encode term-for-term.
float3 pq_encode(float3 y) {
    float3 yp = pow(saturate(y), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * yp) / (1.0 + 18.6875 * yp), 78.84375);
}

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

float curve(float x) {
    if (x <= knee) return x;
    float band = headroom - knee;
    return knee + band * (1.0 - exp(-(x - knee) / band));
}

float4 psmain(float4 pos : SV_Position) : SV_Target {
    float2 dims;
    src.GetDimensions(dims.x, dims.y);
    // max(0) mirrors tone::map's clamp: a negative radiance would take the
    // below-knee arm and then reach pow() with a negative base (NaN).
    float3 c = max(src.Load(int3(pos.xy, 0)).rgb * inv_samples, 0.0);
    if (bloom_strength > 0.0) {
        // Glare is applied to the LINEAR radiance, BEFORE the curve — it is a
        // model of light scattering in the lens/eye, which happens to the light
        // itself, not to the displayed pixel. (It also has to be: the curve is
        // where a 44,000-radiance sun disc gets compressed to the display's
        // peak, and glare's whole job is to redistribute that energy into the
        // surrounding pixels while it still EXISTS.)
        //
        // Energy-conserving composite (bloom.rs): glare REDISTRIBUTES light, it
        // never adds any, so a uniformly lit frame comes back unchanged — which
        // is what keeps bloom from being tunable into an exposure change.
        float2 uv = (pos.xy + 0.5) / dims;
        c = lerp(c, tent(uv), bloom_strength);
    }
    float3 f = float3(curve(c.r), curve(c.g), curve(c.b));
    if (mode > 1.5) {
        // HDR10: paper-white-relative -> PQ's 10000-nit-normalized domain,
        // then gamut matrix + ST 2084 — tone::encode's Pq arm verbatim.
        return float4(pq_encode(m709_to_2020(f * scale)), 1.0);
    }
    if (mode > 0.5) f = pow(f, 1.0 / 2.2);
    // No saturate: under scRGB, values above 1.0 are legal and ARE the highlight
    // headroom. The curve is bounded by `headroom` on its own.
    return float4(f * scale, 1.0);
}
