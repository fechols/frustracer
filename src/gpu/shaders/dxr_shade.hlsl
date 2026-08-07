// The deferred shading half of --dxr-inline 3 (thin CHS): reference.hlsl's
// per-pixel body with the primary trace replaced by a hit-record read. The
// raygen (dxr.hlsl's mode-3 arm) fired the bare-hit TraceRay and stored
// {t, tri, u, v, inst}; this kernel re-derives the SAME rng/jitter/ray
// (identical pasted sample_pos/ray_dir — integer rng half bit-exact by
// construction; the float half may fma-differ sub-ulp across the lib/cs
// compilations, which is why no mode-3-vs-other-mode gate may ever be
// bit-exact) and runs shade_full with rt.hlsli's inline RayQuery
// secondaries — in a PLAIN COMPUTE stage, which is the whole point: the
// 2026-08 Intel campaign measured Arc executing the identical code 3-4.5x
// slower hosted in a raygen/closest-hit stage than in compute
// (FR_ABL=nosec collapsed mode 1 from 2.395 to 0.478 ms — the existence
// proof this kernel is built on).
//
// Requires trace_common.hlsli + skylod.hlsli + rt.hlsli + ripple.hlsli +
// shade.hlsli pasted first (gpu/dxr.rs concatenates; this unit pastes NO
// rt_dxr.hlsli and must never contain a TraceRay — cargo test pins that).
//
// One sample per PASS (push0 = s, the b1 root constants): gpu/dxr.rs records
// spp pass pairs [DispatchRays -> barrier -> this kernel -> barrier].
// Cross-pass accumulation rides `csum` (u8 — the wavefront's qsky register,
// free in DXR sessions like the hit record's u7): pass 0 stores, later
// passes load-add-store, and the LAST pass (s == spp-1) divides by spp and
// performs the one store-or-add accum splat per frame (dxr.hlsl's exact
// splat rule; probe_sample is clamped <= spp-1, so the side channels always
// fire on some pass).

RWStructuredBuffer<float> accum : register(u0); // rw*rh*3, CPU layout parity
RWStructuredBuffer<float> tbuf  : register(u1); // primary-hit t, INF = sky
RWStructuredBuffer<uint>  info  : register(u2); // pack_info(depth, kind)

// LOCKSTEP with dxr.hlsl's mode-3 block — same struct, same register; a
// field skew reads garbage.
struct HitRec {
    float t; // < 0 = miss (the HitPayload wire convention, NOT tbuf's INF)
    uint tri;
    float u;
    float v;
    uint inst; // committed InstanceID (sway-MV lane); 0 on miss
};
RWStructuredBuffer<HitRec> hitrec : register(u7);
RWStructuredBuffer<float>  csum   : register(u8); // rw*rh*3 cross-pass sum

cbuffer Push : register(b1) { uint push0; uint push1; uint push2; uint push3; }

#ifdef WIDTH_PROBE
// FR_WIDTH: the DXR width sink (dxr.rs's width_buf at the u3 root param,
// bound only when armed). Slot 1 = this kernel; slot 0 = the raygen.
RWStructuredBuffer<uint> dxr_width : register(u3);
#endif

[numthreads(8, 8, 1)]
void cs_dxr_shade(uint3 id : SV_DispatchThreadID) {
#ifdef WIDTH_PROBE
    // The deferred kernel's compiled wave width — THE number of the cliff
    // hunt: this is the cs_6_5 unit that measured 1.9x the reference kernel
    // doing less work, before the early-out so it always reports.
    if (id.x == 0u && id.y == 0u) dxr_width[1] = WaveGetLaneCount();
#endif
    // --dual-gpu: the dispatch covers only [push1, push2) rows, so lift the
    // thread id back into absolute screen space (the raygen's `band_id`) and
    // bound against the band's END rather than `rh` — the grid is rounded up
    // to the 8x8 group, so the tail would otherwise shade rows this device
    // does not own from a STALE hit record. Unbanded, push1 = 0 and
    // push2 = rh, which is the pre-feature test verbatim.
    id.y += push1;
    if (id.x >= rw || id.y >= push2)
        return;
    uint pi = id.y * rw + id.x;
    uint s = push0;
    uint rng = rng_init(id.x, id.y, frame, s);
    float2 sp = sample_pos(id.x, id.y, s, rng);
    float3 dir = ray_dir(sp.x, sp.y);

    HitRec rec = hitrec[pi];
    float3 col;
    float t;
    // MISS SENTINEL FIRST: the record's miss is t < 0 (miss_hit's wire
    // convention); it must convert to the sky arm's t = INF before ANY
    // consumer — a raw -1 in tbuf would classify every sky pixel as a hit
    // and fail --check-dxr's T1 loudly. cargo test pins this ordering.
    if (rec.t < 0.0) {
#if SKY_LOD > 1
        col = sky_radiance_lod(dir, id.x, id.y);
#else
        col = sky_radiance(cam_origin.xyz, dir, pixel_cone * 0.5, frame,
                           cloud_dither_k(id.xy, frame, s, spp));
#endif
        t = INF;
        if (s == probe_sample)
            gbuf_write_sky(pi, sp.x, sp.y, dir);
    } else {
        HitInfo h;
        h.t = rec.t;
        h.tri = rec.tri;
        h.u = rec.u;
        h.v = rec.v;
#ifdef SWAY_MV
        h.inst = rec.inst;
#endif
        PrimSurf ps;
        // Full emissive mask — the DXR pipeline has no quadtree tile to
        // cull from (dxr.hlsl's raygen/chs_shade arms, verbatim).
        col = shade_full(cam_origin.xyz, dir, h, rng,
                         uint2(0xffffffffu, 0xffffffffu), ps);
        t = h.t;
        if (s == probe_sample)
            gbuf_write_hit(pi, sp.x, sp.y, dir, t, ps SWAY_ARG(rec.inst));
    }
#ifndef ABL_NO_FF_CODE
    if (flags & FLAG_FIREFLIES)
        col += ff_glow(cam_origin.xyz, dir, t, pixel_cone * 0.5);
#endif
    if (s == probe_sample) {
        tbuf[pi] = t;
        info[pi] = pack_info(0u, KIND_LEAF);
    }

    // Cross-pass accumulation, load-before-store (no same-address RAW to
    // reason about): pass 0's `col` seeds the sum, later passes add the
    // stored prefix, the last pass divides and splats. Exactly ONE
    // store-or-add on accum per frame — the dxr.hlsl/reference.hlsl splat
    // contract verbatim.
    uint i3 = pi * 3u;
    float3 c = col;
    if (s > 0u)
        c += float3(csum[i3 + 0u], csum[i3 + 1u], csum[i3 + 2u]);
    if (s + 1u < spp) {
        csum[i3 + 0u] = c.x;
        csum[i3 + 1u] = c.y;
        csum[i3 + 2u] = c.z;
    } else {
        c *= (1.0 / float(spp));
        // splat: frame 0 (or non-accumulating) stores — the implicit clear.
        if (frame == 0u || (flags & FLAG_ACCUM) == 0u) {
            accum[i3 + 0u] = c.x;
            accum[i3 + 1u] = c.y;
            accum[i3 + 2u] = c.z;
        } else {
            accum[i3 + 0u] += c.x;
            accum[i3 + 1u] += c.y;
            accum[i3 + 2u] += c.z;
        }
    }
}
