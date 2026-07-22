// HUD/menu overlay composite (src/gpu/hud.rs): the Slint software renderer's
// PREMULTIPLIED, display-space (sRGB-encoded) RGBA8 buffer, alpha-blended
// over the tonemapped backbuffer as a second fullscreen draw inside
// `fullscreen_to_backbuffer` — one insertion point, every present arm.
//
// The PSO blends ONE / INV_SRC_ALPHA (premultiplied), so this shader's only
// job is to hand the blender the texel in the BACKBUFFER's space:
//  - SDR (B8G8R8A8, gamma_on != 0): the backbuffer is display space, which is
//    exactly what Slint authored — pass through.
//  - scRGB (RGBA16F, gamma_on == 0): the backbuffer is LINEAR with 1.0 = 80
//    nits / `scale` = paper_white/80 (see tone.rs) — un-premultiply, decode
//    the sRGB-ish 2.2 curve, scale so UI white lands at paper white, and
//    re-premultiply. The blend then runs in linear space; the slight
//    AA-fringe difference vs the SDR path's display-space blend is the same
//    accepted compromise `render::present_px_scrgb` documents for the debug
//    overlay.
//
// The cbuffer DECLARATION mirrors tonemap.hlsl's Params exactly — this pass
// reuses the tonemap root signature and is recorded through `Passes::record`,
// so the 8 root constants arrive in that layout; only scale/gamma_on are read.
Texture2D<float4> src : register(t0);

cbuffer Params : register(b0) {
    float inv_samples;
    float bloom_strength;
    float2 bloom_texel;
    float knee;
    float headroom;
    float scale;
    float gamma_on;
}

float4 vsmain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 psmain(float4 pos : SV_Position) : SV_Target {
    float4 c = src.Load(int3(pos.xy, 0));
    if (gamma_on == 0.0) {
        float3 rgb = c.a > 0.0 ? c.rgb / c.a : float3(0.0, 0.0, 0.0);
        rgb = pow(max(rgb, 0.0), 2.2) * scale;
        c = float4(rgb * c.a, c.a);
    }
    return c;
}
