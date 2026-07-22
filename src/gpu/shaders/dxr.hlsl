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
        // prim packs TWO things (payload stays 32 B): bit 0 = the probe bit
        // (the sample that owns every per-pixel side channel; chs_shade reads
        // it), bits 1.. = the sample index s — the miss shader has the pixel
        // (DispatchRaysIndex) and the frame (CB) but not s, and the cloud
        // march phase is per (pixel, frame, SAMPLE). spp <= MAX_SPP = 128
        // fits with 23 bits to spare.
        p.prim = (s << 1) | ((s == probe_sample) ? 1u : 0u);
        TraceRay(tlas, OPAQUE_RF, 0xffu, 0u, 0u, 0u, r, p);

        // Firefly glow, depth-tested against the payload's t (INF on a miss)
        // — the wavefront kernels' composite, term for term.
        if (flags & FLAG_FIREFLIES)
            p.color += ff_glow(cam_origin.xyz, dir, p.t, pixel_cone * 0.5);
        csum += p.color;
        if ((p.prim & 1u) != 0u) {
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
    // tri_of == PrimitiveIndex() in the single-BLAS build; the chunk remap
    // under --blas-split (trace_common.hlsli).
    h.tri = tri_of(InstanceID(), PrimitiveIndex());
    h.u = a.barycentrics.x;
    h.v = a.barycentrics.y;
#ifdef HEIGHTFIELD
    // Re-march the committed hit for the displaced (t', u', v') — an any-hit
    // can only accept/ignore at the plane t, so the hardware sorts by plane t
    // (the depth-band mis-ordering known-accept) and the closest-hit derives
    // the true marched hit. Deterministic: same inputs as the any-hit's run.
    if (flags & FLAG_HEIGHT)
        height_march(h.tri, WorldRayOrigin(), WorldRayDirection(), h.t, h.u, h.v);
#endif
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
    // stage just reports where the ray was. Bit 0 of prim is the probe bit —
    // the high bits carry the sample index for the miss shader's cloud phase.
    if ((p.prim & 1u) == 0u) return;
    uint2 id = DispatchRaysIndex().xy;
    gbuf_write_hit(id.y * rw + id.x, p.sp.x, p.sp.y, WorldRayDirection(), h.t, ps);
}

[shader("closesthit")]
void chs_hit(inout HitPayload p, in BuiltInTriangleIntersectionAttributes a) {
    p.t = RayTCurrent();
    p.tri = tri_of(InstanceID(), PrimitiveIndex());
    p.u = a.barycentrics.x;
    p.v = a.barycentrics.y;
#ifdef HEIGHTFIELD
    if (flags & FLAG_HEIGHT)
        height_march(p.tri, WorldRayOrigin(), WorldRayDirection(), p.t, p.u, p.v);
#endif
}

[shader("miss")]
void miss_radiance(inout RayPayload p) {
    // The raygen primary ray's miss: a DISPLAY path (the backdrop), so it sees
    // the sun disc. Reflection/glass continuations route to miss_hit_info and
    // are handled inside shade.hlsli, per the sky.rs invariant. The cloud
    // march phase is per (pixel, frame, SAMPLE) — the sample index rides
    // prim's high bits (raygen packs `(s << 1) | probe`).
    p.color = sky_radiance(WorldRayOrigin(), WorldRayDirection(), pixel_cone * 0.5, frame,
                           cloud_dither_k(DispatchRaysIndex().xy, frame, p.prim >> 1u, spp));
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

#if defined(ALPHA_CUTOUT) || defined(HEIGHTFIELD) || defined(TRANS_SHADOW)
// Cutout/relief/tinted-shadow any-hit shaders (alpha-masked, height-carrying
// or transmissive scenes; the BLAS drops OPAQUE and the SHADOW_RF — plus,
// for cutout/relief, OPAQUE_RF — ray flags drop FORCE_OPAQUE so these run;
// on a transmission-only scene closest rays keep FORCE_OPAQUE and ah_shade/
// ah_hit compile but stay inert). One per payload type — a hit group's
// any-hit must share its ray's payload — all three deferring to the SAME
// trace_common.hlsli::candidate_reject the RayQuery candidate loops use, so
// both intersectors agree bit-for-bit. IgnoreHit() == moller_trumbore
// returning None: the candidate is removed, traversal continues, TMax stays
// unshrunk — which is exactly what makes relief silhouettes real on this
// pipeline too. The marched t'/bary are discarded here (an any-hit cannot
// move the committed t); chs_shade/chs_hit re-derive them. ah_shadow keeps
// its own body below: it consumes the marched t for the logical-tmax
// re-check and accumulates the tinted-shadow throughput.
bool ah_reject(uint tri, float u, float v) {
    float t = RayTCurrent();
    return candidate_reject(tri, WorldRayOrigin(), WorldRayDirection(), t, u, v) != 0u;
}

[shader("anyhit")]
void ah_shade(inout RayPayload p, in BuiltInTriangleIntersectionAttributes a) {
    if (ah_reject(tri_of(InstanceID(), PrimitiveIndex()), a.barycentrics.x, a.barycentrics.y))
        IgnoreHit();
}

[shader("anyhit")]
void ah_hit(inout HitPayload p, in BuiltInTriangleIntersectionAttributes a) {
    if (ah_reject(tri_of(InstanceID(), PrimitiveIndex()), a.barycentrics.x, a.barycentrics.y))
        IgnoreHit();
}

// The shadow flavor additionally re-checks the MARCHED t against the
// segment's LOGICAL far bound (ShadowPayload::tmax): occluded_q widens the
// hardware TMax by one relief depth so underside candidates whose plane t
// lies just beyond the segment still surface (rt_dxr.hlsli), and this test —
// the mirror of the RayQuery loops' explicit `ct < tmax` — is what keeps a
// marched hit BEYOND the segment from occluding it. Unwidened rays never
// trip it (their candidates already satisfy plane t <= tmax and, without a
// live march, t' == plane t).
[shader("anyhit")]
void ah_shadow(inout ShadowPayload p, in BuiltInTriangleIntersectionAttributes a) {
    float t = RayTCurrent();
    float u = a.barycentrics.x;
    float v = a.barycentrics.y;
    uint tri = tri_of(InstanceID(), PrimitiveIndex());
    if (candidate_reject(tri, WorldRayOrigin(), WorldRayDirection(), t, u, v) != 0u
        || t >= p.tmax)
        IgnoreHit();
#ifdef TRANS_SHADOW
    // Tinted shadows: a transmissive candidate multiplies the payload tint
    // and is IGNORED (payload writes persist through IgnoreHit — traversal
    // continues, the standard tinted-shadow pattern; rt.hlsli::transmit_q is
    // the RayQuery twin). Opaque candidates fall through and commit. The
    // ordering matters: a candidate rejected above (cutout texel, relief
    // escape, beyond the logical tmax) must NOT tint.
    float4 ms = mat_shadow[uv_tri_mat[tri]];
    if (ms.a > 0.0) {
        p.tint *= ms.rgb;
        IgnoreHit();
    }
#endif
}
#endif // ALPHA_CUTOUT || HEIGHTFIELD || TRANS_SHADOW
