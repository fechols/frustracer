// The DispatchRays entry points: reference.hlsl's per-pixel loop split
// across the raygen / closest-hit / miss stages of the DXR pipeline.
// Requires trace_common.hlsli + rt_dxr.hlsli + shade.hlsli pasted first
// (gpu/dxr.rs concatenates). Recursion shape: raygen's primary TraceRay is
// depth 1; chs_shade's shadow/AO/reflection rays are depth 2; chs_hit and
// the misses fire nothing — MaxTraceRecursionDepth = 2 in the RTPSO.

RWStructuredBuffer<float> accum : register(u0); // rw*rh*3, CPU layout parity
RWStructuredBuffer<float> tbuf  : register(u1); // primary-hit t, INF = sky
RWStructuredBuffer<uint>  info  : register(u2); // pack_info(depth, kind)

[shader("raygeneration")]
void raygen() {
    uint2 id = DispatchRaysIndex().xy;
    uint pi = id.y * rw + id.x;

    // --spp: one TraceRay per sample, averaged into a single accum
    // store-or-add (two splats would break the accum semantics, exactly as on
    // the CPU). No tile claim exists in this pipeline — every ray starts at
    // the TLAS root with TMin = 0 — so this is plain supersampling: it buys
    // the ~1/N variance the upscaler wants, not the quadtree amortization the
    // --cpu/--gpu paths get.
    float3 csum = 0.0;
    for (uint s = 0u; s < spp; ++s) {
        uint rng = rng_init(id.x, id.y, frame, s);
        float2 sp = sample_pos(id.x, id.y, s, rng);
        float3 dir = ray_dir(sp.x, sp.y);

        RayDesc r;
        r.Origin = cam_origin.xyz; r.Direction = dir; r.TMin = 0.0; r.TMax = FLT_MAX;
        RayPayload p;
        p.color = float3(0.0, 0.0, 0.0);
        p.t = INF;
        p.rng = rng;
        p.sp = sp;
        // The probe sample owns every per-pixel side channel; chs_shade reads
        // this bit (it fires once per SAMPLE now, not once per pixel).
        p.prim = (s == probe_sample) ? 1u : 0u;
        TraceRay(tlas, OPAQUE_RF, 0xffu, 0u, 0u, 0u, r, p);

        csum += p.color;
        if (p.prim != 0u) {
            tbuf[pi] = p.t;
            info[pi] = pack_info(0u, KIND_LEAF);
            // Sky G-buffer capture (FLAG_GBUF-gated inside the helper — plain
            // sessions are bit-untouched, and no rng draw is consumed). The
            // hit half lives in chs_shade, where the PrimSurf is.
            if (isinf(p.t))
                gbuf_write_sky(pi, sp.x, sp.y, dir);
        }
    }
    float3 c = csum * (1.0 / float(spp));

    uint i3 = pi * 3u;
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

[shader("closesthit")]
void chs_shade(inout RayPayload p, in BuiltInTriangleIntersectionAttributes a) {
    HitInfo h;
    h.t = RayTCurrent();
    h.tri = PrimitiveIndex();
    h.u = a.barycentrics.x;
    h.v = a.barycentrics.y;
    PrimSurf ps;
    p.color = shade_full(WorldRayOrigin(), WorldRayDirection(), h, p.rng, ps);
    p.t = h.t;
    // G-buffer capture: a pure copy of already-computed values, zero rng
    // draws, FLAG_GBUF-gated inside the helper. Under --spp chs_shade fires
    // once per SAMPLE (the reflection ray still routes to HgHit and the
    // occlusion ray to the null record, so it is once per sample and no
    // more), and only the probe sample may write the pack — otherwise the
    // guides would drift off the jitter the upscaler was told about. The
    // sample position rides the payload: raygen owns the jitter policy, this
    // stage just reports where the ray was.
    if (p.prim == 0u) return;
    uint2 id = DispatchRaysIndex().xy;
    gbuf_write_hit(id.y * rw + id.x, p.sp.x, p.sp.y, WorldRayDirection(), h.t, ps);
}

[shader("closesthit")]
void chs_hit(inout HitPayload p, in BuiltInTriangleIntersectionAttributes a) {
    p.t = RayTCurrent();
    p.tri = PrimitiveIndex();
    p.u = a.barycentrics.x;
    p.v = a.barycentrics.y;
}

[shader("miss")]
void miss_radiance(inout RayPayload p) {
    p.color = sky_color(WorldRayDirection());
    p.t = INF;
}

[shader("miss")]
void miss_shadow(inout ShadowPayload p) {
    p.hit = 0u;
}

[shader("miss")]
void miss_hit(inout HitPayload p) {
    p.t = -1.0;
}

#ifdef ALPHA_CUTOUT
// Alpha-cutout any-hit shaders (alpha-masked scenes only; the BLAS drops
// OPAQUE and the OPAQUE_RF ray flags drop FORCE_OPAQUE so these run). One
// per payload type — a hit group's any-hit must share its ray's payload —
// all three deferring to the SAME trace_common.hlsli::alpha_cutout the
// RayQuery candidate loops use, so both intersectors agree bit-for-bit.
// IgnoreHit() == moller_trumbore returning None: the candidate is removed,
// traversal continues, TMax stays unshrunk.
[shader("anyhit")]
void ah_shade(inout RayPayload p, in BuiltInTriangleIntersectionAttributes a) {
    if (alpha_cutout(PrimitiveIndex(), a.barycentrics.x, a.barycentrics.y))
        IgnoreHit();
}

[shader("anyhit")]
void ah_hit(inout HitPayload p, in BuiltInTriangleIntersectionAttributes a) {
    if (alpha_cutout(PrimitiveIndex(), a.barycentrics.x, a.barycentrics.y))
        IgnoreHit();
}

[shader("anyhit")]
void ah_shadow(inout ShadowPayload p, in BuiltInTriangleIntersectionAttributes a) {
    if (alpha_cutout(PrimitiveIndex(), a.barycentrics.x, a.barycentrics.y))
        IgnoreHit();
}
#endif // ALPHA_CUTOUT
