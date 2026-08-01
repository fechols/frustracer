// HUD/menu overlay composite (src/gpu/hud.rs): the Slint software renderer's
// PREMULTIPLIED, display-space (sRGB-encoded) RGBA8 buffer, alpha-blended
// over the tonemapped backbuffer as a second fullscreen draw inside
// `fullscreen_to_backbuffer` — one insertion point, every present arm.
//
// The PSO blends ONE / INV_SRC_ALPHA (premultiplied), so this shader's only
// job is to hand the blender the texel in the BACKBUFFER's space:
//  - Gamma 2.2 (B8G8R8A8 SDR AND R10G10B10A2 Sdr10, mode == 1): the
//    backbuffer is display space, which is exactly what Slint authored —
//    pass through (the RTV width is the PSO's business, not this shader's).
//  - HDR10 (R10G10B10A2 + G2084, mode == 2): the backbuffer is PQ-encoded —
//    un-premultiply, decode the sRGB-ish 2.2 curve into paper-white-relative
//    light, scale into PQ's 10000-nit domain (`scale` = paper_white/10000),
//    gamut matrix + ST 2084 (literals mirror tone.rs), re-premultiply.
//    Blending premultiplied PQ values is the display-space-blend compromise
//    the SDR arm already makes, at HDR10's own encoding.
//  (mode == 0 was the scRGB-linear arm, retired with the f16 swapchain.)
//
// The cbuffer DECLARATION mirrors tonemap.hlsl's Params exactly — this pass
// reuses the tonemap root signature and is recorded through `Passes::record`,
// so the 8 root constants arrive in that layout; only scale/mode are read.
Texture2D<float4> src : register(t0);

cbuffer Params : register(b0) {
    float inv_samples;
    float bloom_strength;
    float2 bloom_texel;
    float knee;
    float headroom;
    float scale;
    float mode; // 1 = gamma 2.2 (SDR/Sdr10), 2 = HDR10 PQ (0 retired with scRGB)
}

// tone.rs twins (see tonemap.hlsl — same literals, change all three together).
float3 m709_to_2020(float3 v) {
    return float3(
        0.627404 * v.r + 0.329283 * v.g + 0.043313 * v.b,
        0.069097 * v.r + 0.919540 * v.g + 0.011362 * v.b,
        0.016391 * v.r + 0.088013 * v.g + 0.895595 * v.b);
}

float3 pq_encode(float3 y) {
    float3 yp = pow(saturate(y), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * yp) / (1.0 + 18.6875 * yp), 78.84375);
}

float4 vsmain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 psmain(float4 pos : SV_Position) : SV_Target {
    float4 c = src.Load(int3(pos.xy, 0));
    if (mode > 1.5) {
        // HDR10: un-premultiply, decode into paper-white-relative light, then
        // the PQ encode chain (scale = paper_white / 10000).
        float3 rgb = c.a > 0.0 ? c.rgb / c.a : float3(0.0, 0.0, 0.0);
        rgb = pow(max(rgb, 0.0), 2.2) * scale;
        rgb = pq_encode(m709_to_2020(rgb));
        c = float4(rgb * c.a, c.a);
    }
    // mode == 1 (SDR/Sdr10): the backbuffer is display space — pass through.
    return c;
}
