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

    uint rng = rng_init(id.x, id.y, frame, 0u);
    float jx = 0.5, jy = 0.5;
    if (flags & FLAG_FRAME_JITTER) {
        jx = 0.5 + frame_jitter.x;
        jy = 0.5 + frame_jitter.y;
    } else if (flags & FLAG_JITTER) {
        jx = rng_next(rng);
        jy = rng_next(rng);
    }
    float3 dir = ray_dir(float(id.x) + jx, float(id.y) + jy);

    RayDesc r;
    r.Origin = cam_origin.xyz; r.Direction = dir; r.TMin = 0.0; r.TMax = FLT_MAX;
    RayPayload p;
    p.color = float3(0.0, 0.0, 0.0);
    p.t = INF;
    p.rng = rng;
    TraceRay(tlas, OPAQUE_RF, 0xffu, 0u, 0u, 0u, r, p);

    uint pi = id.y * rw + id.x;
    uint i3 = pi * 3u;
    // splat: frame 0 (or non-accumulating) stores — the implicit clear.
    if (frame == 0u || (flags & FLAG_ACCUM) == 0u) {
        accum[i3 + 0u] = p.color.x;
        accum[i3 + 1u] = p.color.y;
        accum[i3 + 2u] = p.color.z;
    } else {
        accum[i3 + 0u] += p.color.x;
        accum[i3 + 1u] += p.color.y;
        accum[i3 + 2u] += p.color.z;
    }
    tbuf[pi] = p.t;
    info[pi] = pack_info(0u, KIND_LEAF);
    // Sky G-buffer capture (FLAG_GBUF-gated inside the helper — plain
    // sessions are bit-untouched, and no rng draw is consumed). The hit
    // half lives in chs_shade, where the PrimSurf is.
    if (isinf(p.t))
        gbuf_write_sky(pi, float(id.x) + jx, float(id.y) + jy, dir);
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
    // draws, FLAG_GBUF-gated inside the helper. chs_shade fires exactly
    // once per pixel structurally (the reflection ray routes to HgHit, the
    // occlusion ray to the null record). The (fx, fy) sample position is
    // recomputed from the frame-uniform jitter — upscaler frames always run
    // FLAG_FRAME_JITTER; on FLAG_JITTER (rng-drawn, plain accumulation)
    // frames the offset here is the pixel center, which is harmless: (fx,
    // fy) feeds only gbuf_mv, and those frames run with prev_cam unset, so
    // the MV is (0,0) regardless.
    uint2 id = DispatchRaysIndex().xy;
    float jx = 0.5, jy = 0.5;
    if (flags & FLAG_FRAME_JITTER) {
        jx = 0.5 + frame_jitter.x;
        jy = 0.5 + frame_jitter.y;
    }
    gbuf_write_hit(id.y * rw + id.x, float(id.x) + jx, float(id.y) + jy,
                   WorldRayDirection(), h.t, ps);
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
    // The raygen primary ray's miss: a DISPLAY path (the backdrop), so it sees
    // the sun disc. Reflection/glass continuations route to miss_hit_info and
    // are handled inside shade.hlsli, per the sky.rs invariant.
    p.color = sky_radiance(WorldRayDirection(), pixel_cone * 0.5);
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
