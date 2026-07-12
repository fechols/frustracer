// Feed pass: fan the tracer's G-buffer pack (gbuf, u15) + the 1-spp radiance
// (accum) out into an upscaler's input textures — the GPU-resident
// replacement for rr.rs/xr.rs::record_upload's CPU convert-and-copy. One
// thread per pixel; the target textures are typed UAVs reached through the
// second range of the u14 descriptor table (registers u16..u22, heap slots
// 1..7 — trace.rs::wire_feed builds the descriptors per session).
// Requires trace_common.hlsli pasted first.
//
// Register/type layout is shared by both kernels (same compile unit); each
// kernel writes only its planes and DXC strips the rest. Where both write
// the same register the TYPE matches; the VALUE may differ (u18 depth is
// reversed-Z clip depth for XeSS but raw linear view-Z for RR).

RWStructuredBuffer<float> accum : register(u0); // rw*rh*3 linear HDR (1-spp store)

RWTexture2D<float4> feed_color   : register(u16); // RGBA16F  (both)
RWTexture2D<float4> feed_nr      : register(u17); // RGBA16F  normal.xyz + rough (RR)
RWTexture2D<float>  feed_depth   : register(u18); // R32F     (both; encoding differs)
RWTexture2D<float2> feed_mvec    : register(u19); // RG16F    (both)
RWTexture2D<float4> feed_alb     : register(u20); // RGBA8    diffuse albedo (RR)
RWTexture2D<float4> feed_spec    : register(u21); // RGBA8    specular albedo F0 (RR)
RWTexture2D<float>  feed_spechit : register(u22); // R16F     spec hit distance (RR)

// xess.rs::view_z_to_clip_depth: linear view-Z -> [0,1] reversed-Z clip
// depth. `precise` keeps DXC from FMA-contracting near*(far-z) — sky's
// view_z == far must land EXACTLY on 0.0 (the CPU encode's contract).
float view_z_to_clip_depth(float view_z, float near, float far) {
    precise float z = max(view_z, near);
    precise float num = near * (far - z);
    precise float den = z * (far - near);
    // The quotient must be precise too, or DXC lowers it to rcp+mul
    // (observed: 2-ulp drift on ~0.1% of pixels).
    precise float q = num / den;
    return saturate(q);
}

[numthreads(8, 8, 1)]
void cs_feed_xess(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint pi = id.y * rw + id.x;
    uint i3 = pi * 3u;
    GBufPx g = gbuf[pi];
    feed_color[id.xy] = float4(accum[i3], accum[i3 + 1u], accum[i3 + 2u], 1.0);
    feed_mvec[id.xy] = g.mv.xy;
    feed_depth[id.xy] = view_z_to_clip_depth(g.alb_z.w, CAM_NEAR, CAM_FAR);
}

// NPPD pre-denoise frames (--gpu --nppd): the color plane comes from
// cs_nppd_out (nppd.hlsl) instead — this writes only the guide planes.
[numthreads(8, 8, 1)]
void cs_feed_xess_dm(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    GBufPx g = gbuf[id.y * rw + id.x];
    feed_mvec[id.xy] = g.mv.xy;
    feed_depth[id.xy] = view_z_to_clip_depth(g.alb_z.w, CAM_NEAR, CAM_FAR);
}

[numthreads(8, 8, 1)]
void cs_feed_rr(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint pi = id.y * rw + id.x;
    uint i3 = pi * 3u;
    GBufPx g = gbuf[pi];
    feed_color[id.xy] = float4(accum[i3], accum[i3 + 1u], accum[i3 + 2u], 1.0);
    feed_nr[id.xy] = g.nr;
    feed_depth[id.xy] = g.alb_z.w; // linear view-Z, RR's LINEAR_DEPTH contract
    feed_mvec[id.xy] = g.mv.xy;
    // UNORM stores clamp to [0,1] in hardware (the CPU path's to_unorm8).
    feed_alb[id.xy] = float4(g.alb_z.xyz, 1.0);
    feed_spec[id.xy] = float4(g.spec.xyz, 1.0);
    feed_spechit[id.xy] = g.spec.w;
}
