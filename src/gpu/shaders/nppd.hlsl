// NPPD GPU staging (the --gpu --nppd composition): term-for-term ports of
// src/nppd.rs's pure-math half, writing the NCHW fp32 plane buffers ONNX
// Runtime consumes in place (nppd::NppdGpu binds them as tensors — zero CPU
// traffic). Requires trace_common.hlsli pasted first.
//
// Plane layout (all f32, plane-major over the /32-PADDED dims pw x ph):
//   nppd_frame  10 planes  (nppd::pack_inputs: depth|normal|diffuse|color)
//   nppd_state  38 planes  (the recurrent state; the graph writes it back)
//   nppd_warped 38 planes  (the backward-warped state the graph reads)
//   nppd_out     3 planes  (the graph's denoised color)
//
// Hazard notes: cs_nppd_pack / cs_nppd_warp / cs_nppd_zero touch pairwise-
// disjoint buffers, so they record back-to-back with no barriers; the reads
// of gbuf/accum are fenced by record_frame's trailing global UAV barrier, and
// everything ORT wrote (state, out) arrived in a PREVIOUS ExecuteCommandLists
// batch on the shared queue — cross-batch visibility is implicit.

RWStructuredBuffer<float> accum : register(u0); // rw*rh*3 linear HDR (1-spp store)

RWStructuredBuffer<float> nppd_frame  : register(u23);
RWStructuredBuffer<float> nppd_state  : register(u24);
RWStructuredBuffer<float> nppd_warped : register(u25);
RWStructuredBuffer<float> nppd_out    : register(u26);

RWTexture2D<float4> nppd_feed_color : register(u16); // the XeSS color plane

#define NPPD_CT 38u

// nppd::pad_dims — round up to the /32 padding quantum.
uint pad32(uint v) { return (v + 31u) & ~31u; }

// nppd::pack_inputs: one thread per PADDED pixel; the border replicates the
// edge texel (clamped source coords — a zero border would be a synthetic
// edge the U-Net denoises against). ch0 is ln(1 + 1/d) of the EUCLIDEAN
// camera distance recovered from view-Z through the SOURCE pixel's center
// ray (sky -> exact 0); ch1-3 the camera-space normal in the UNIT
// fwd / fwd x Y / r x f basis (NOT the pre-scaled CB rows); ch4-6 diffuse
// albedo; ch7-9 the raw 1-spp HDR store.
[numthreads(8, 8, 1)]
void cs_nppd_pack(uint3 id : SV_DispatchThreadID) {
    uint pw = pad32(rw), ph = pad32(rh);
    if (id.x >= pw || id.y >= ph) return;
    uint sx = min(id.x, rw - 1u), sy = min(id.y, rh - 1u);
    uint si = sy * rw + sx;
    uint pp = pw * ph;
    uint o = id.y * pw + id.x;
    GBufPx g = gbuf[si];

    float view_z = g.alb_z.w;
    float d0 = 0.0;
    if (view_z < CAM_FAR * 0.99) {
        float3 dir = ray_dir(float(sx) + 0.5, float(sy) + 0.5);
        float t = view_z / dot(dir, cam_forward.xyz);
        d0 = log(1.0 + 1.0 / max(t, 1e-8));
    }
    nppd_frame[o] = d0;

    float3 f = cam_forward.xyz;
    float3 r = normalize(cross(f, float3(0.0, 1.0, 0.0)));
    float3 u = cross(r, f);
    nppd_frame[o + pp]      = dot(g.nr.xyz, f);
    nppd_frame[o + 2u * pp] = dot(g.nr.xyz, r);
    nppd_frame[o + 3u * pp] = dot(g.nr.xyz, u);

    nppd_frame[o + 4u * pp] = g.alb_z.x;
    nppd_frame[o + 5u * pp] = g.alb_z.y;
    nppd_frame[o + 6u * pp] = g.alb_z.z;

    uint i3 = si * 3u;
    nppd_frame[o + 7u * pp] = accum[i3];
    nppd_frame[o + 8u * pp] = accum[i3 + 1u];
    nppd_frame[o + 9u * pp] = accum[i3 + 2u];
}

// nppd::warp_temporal: backward-warp state -> warped through this frame's
// motion vectors (MV_SIGN = (1,1): GBufPx.mv IS the fetch offset), bilinear
// with grid_sample-style ZEROS padding outside the logical rw x rh interior;
// the padded border writes 0 (a stale buffer never leaks). Tap order and
// weight formulas mirror the CPU loop exactly.
[numthreads(8, 8, 1)]
void cs_nppd_warp(uint3 id : SV_DispatchThreadID) {
    uint pw = pad32(rw), ph = pad32(rh);
    if (id.x >= pw || id.y >= ph) return;
    uint pp = pw * ph;
    uint o = id.y * pw + id.x;
    if (id.x >= rw || id.y >= rh) {
        for (uint c = 0u; c < NPPD_CT; c++) nppd_warped[o + c * pp] = 0.0;
        return;
    }
    GBufPx g = gbuf[id.y * rw + id.x];
    float px = float(id.x) + g.mv.x;
    float py = float(id.y) + g.mv.y;
    float x0f = floor(px), y0f = floor(py);
    float fx = px - x0f, fy = py - y0f;
    int x0 = int(x0f), y0 = int(y0f);
    float w00 = (1.0 - fx) * (1.0 - fy);
    float w10 = fx * (1.0 - fy);
    float w01 = (1.0 - fx) * fy;
    float w11 = fx * fy;
    bool i00 = x0 >= 0 && y0 >= 0 && x0 < int(rw) && y0 < int(rh);
    bool i10 = x0 + 1 >= 0 && y0 >= 0 && x0 + 1 < int(rw) && y0 < int(rh);
    bool i01 = x0 >= 0 && y0 + 1 >= 0 && x0 < int(rw) && y0 + 1 < int(rh);
    bool i11 = x0 + 1 >= 0 && y0 + 1 >= 0 && x0 + 1 < int(rw) && y0 + 1 < int(rh);
    for (uint c = 0u; c < NPPD_CT; c++) {
        uint base = c * pp;
        float acc = 0.0;
        if (i00) acc += w00 * nppd_state[base + uint(y0) * pw + uint(x0)];
        if (i10) acc += w10 * nppd_state[base + uint(y0) * pw + uint(x0 + 1)];
        if (i01) acc += w01 * nppd_state[base + uint(y0 + 1) * pw + uint(x0)];
        if (i11) acc += w11 * nppd_state[base + uint(y0 + 1) * pw + uint(x0 + 1)];
        nppd_warped[o + c * pp] = acc;
    }
}

// State reset: zero the WARPED buffer (recorded INSTEAD of the warp — the
// graph fully rewrites nppd_state every run, so only what it READS needs
// zeroing; warp-of-anything is skipped entirely).
[numthreads(8, 8, 1)]
void cs_nppd_zero(uint3 id : SV_DispatchThreadID) {
    uint pw = pad32(rw), ph = pad32(rh);
    if (id.x >= pw || id.y >= ph) return;
    uint pp = pw * ph;
    uint o = id.y * pw + id.x;
    for (uint c = 0u; c < NPPD_CT; c++) nppd_warped[o + c * pp] = 0.0;
}

// nppd::crop_to_interleaved, GPU-fed: cut the logical interior out of the
// padded planar network output straight into the XeSS color plane (replaces
// cs_feed_xess's color write in NPPD frames; recorded inside record_feed's
// NPSR->UAV window).
[numthreads(8, 8, 1)]
void cs_nppd_out(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint pw = pad32(rw), ph = pad32(rh);
    uint pp = pw * ph;
    uint o = id.y * pw + id.x;
    nppd_feed_color[id.xy] =
        float4(nppd_out[o], nppd_out[o + pp], nppd_out[o + 2u * pp], 1.0);
}
