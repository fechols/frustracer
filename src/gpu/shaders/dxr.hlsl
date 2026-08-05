// The DispatchRays entry points: reference.hlsl's per-pixel loop split
// across the raygen / closest-hit / miss stages of the DXR pipeline.
// Requires trace_common.hlsli + rt_dxr.hlsli + shade.hlsli pasted first
// (gpu/dxr.rs concatenates). Recursion shape: raygen's primary TraceRay is
// depth 1; chs_shade's shadow/AO/reflection rays are depth 2; chs_hit and
// the misses fire nothing — MaxTraceRecursionDepth = 2 in the RTPSO.
//
// FR_DXR_INLINE (gpu/dxr.rs) reshapes that: mode 1 keeps the primary
// TraceRay but the pasted trace primitives are rt.hlsli's inline RayQuery,
// so chs_shade's secondaries never re-enter the pipeline (recursion depth 1);
// mode 2 additionally takes the DXR_INLINE_SEC == 2 arm in raygen below —
// no TraceRay anywhere, DispatchRays as a bare launch grid; mode 3 (thin
// CHS) inverts the split — raygen fires ONLY the bare-hit primary and writes
// a hit record, and dxr_shade.hlsl (a separate cs_6_5 unit) shades from the
// record, one sample per pass pair (see the mode-3 block below).

RWStructuredBuffer<float> accum : register(u0); // rw*rh*3, CPU layout parity
RWStructuredBuffer<float> tbuf  : register(u1); // primary-hit t, INF = sky
RWStructuredBuffer<uint>  info  : register(u2); // pack_info(depth, kind)

#if defined(WIDTH_PROBE_RAYGEN) || defined(WAVEVIZ)
// FR_WIDTH / FR_WAVEVIZ: the DXR probe sink (dxr.rs's width_buf, bound at
// the otherwise-unbound u3 root param only when armed). Slot 0 = raygen
// width, slot 1 = cs_dxr_shade width (that unit re-declares its own view),
// slot 2 = the WAVEVIZ ticket counter. WaveGetLaneCount() in a raygen is
// spec-legal (lib_6_5 — the armed modes' floor); the value is the COMPILED
// SIMD width, which is the prize — mode 2's raygen IS the codegen lottery
// victim.
RWStructuredBuffer<uint> dxr_width : register(u3);
#endif

#if defined(DXR_INLINE_SEC) && DXR_INLINE_SEC == 3
// Mode 3 (thin CHS + deferred compute shade — the W4 Intel finding: Arc
// executes a fat shader inside an RT pipeline stage at 3-4.5x its compute
// cost, so the pipeline does ONLY what it is uniquely good at — the coherent
// hardware primary — and the fat shade_full + inline-RayQuery secondaries run
// in a compute pass, dxr_shade.hlsl). One sample per PASS: the sample index
// arrives in push0 (the b1 root constants — gpu/dxr.rs sets it before each
// pass pair), never a loop here.
//
// u7 is the wavefront's qleaf register — never declared in any DXR unit, so
// reusing it needs no root-signature change (the cloud-cache u5/u6
// precedent). LOCKSTEP: dxr_shade.hlsl re-declares HitRec/hitrec verbatim —
// the two units share the buffer bytes, and a field skew reads garbage.
struct HitRec {
    float t; // < 0 = miss (the HitPayload wire convention, NOT tbuf's INF)
    uint tri;
    float u;
    float v;
    uint inst; // committed InstanceID (sway-MV lane); 0 on miss
};
RWStructuredBuffer<HitRec> hitrec : register(u7);
// Per-pass sample index. resolve.hlsl declares the same cbuffer in its own
// compile unit; nothing else in the LIB assembly declares b1.
cbuffer Push : register(b1) { uint push0; uint push1; uint push2; uint push3; }
#endif

[shader("raygeneration")]
void raygen() {
    uint2 id = DispatchRaysIndex().xy;
    uint pi = id.y * rw + id.x;
#ifdef WIDTH_PROBE_RAYGEN
    // FR_WIDTH: this raygen's compiled wave width, once per dispatch.
    if (all(id == 0u)) dxr_width[0] = WaveGetLaneCount();
#endif

#if defined(DXR_INLINE_SEC) && DXR_INLINE_SEC == 3
    // The THIN arm: derive the sample's ray (rng/jitter re-derived here
    // solely to reproduce it — the deferred kernel re-derives the identical
    // stream for shading), fire the bare-hit TraceRay (hit group 1 = HgHit,
    // miss 2 = miss_hit — textually rt_dxr.hlsli's trace_closest, which is
    // compiled out in inline modes), and store the record. No shading, no
    // accum/tbuf/info writes — the deferred kernel owns every output.
    // Cutout + relief correctness is inherited: HgHit carries the ah_hit
    // any-hit and chs_hit re-marches relief, exactly as the mode-0
    // continuation path always did.
    uint s = push0;
    uint rng = rng_init(id.x, id.y, frame, s);
    float2 sp = sample_pos(id.x, id.y, s, rng);
    float3 dir = ray_dir(sp.x, sp.y);
    RayDesc r;
    r.Origin = cam_origin.xyz;
    r.Direction = dir;
    r.TMin = height_tmin(0.0);
    r.TMax = height_tmax(FLT_MAX);
    HitPayload p;
    p.t = -1.0; p.tri = 0u; p.u = 0.0; p.v = 0.0;
    p.tmin = 0.0; p.tmax = FLT_MAX;
    p.inst = 0u;
    TraceRay(tlas, OPAQUE_RF, 0xffu, 1u, 0u, 2u, r, p);
    HitRec rec;
    rec.t = p.t;
    rec.tri = p.tri;
    rec.u = p.u;
    rec.v = p.v;
    rec.inst = p.inst;
    hitrec[pi] = rec;
#else
    // --spp: one TraceRay per sample, averaged into a single accum
    // store-or-add (two splats would break the accum semantics, exactly as on
    // the CPU). No tile claim exists in this pipeline — every ray starts at
    // the TLAS root with TMin = 0 — so this is plain supersampling: it buys
    // the ~1/N variance the upscaler wants, not the quadtree amortization the
    // --cpu/--gpu paths get.
    float3 csum = 0.0;
#if defined(BALLAST_N) && (BALLAST_N > 0) && defined(DXR_INLINE_SEC) && (DXR_INLINE_SEC == 2)
    // FR_BALLAST=dxr:N — reference.hlsl's ballast, mirrored into the mode-2
    // raygen (identical code, identical compiled width — FR_WIDTH reads 16
    // for both on the B70), so the two knee curves' offset measures what the
    // RT launch regime confiscates, in floats. Same three-part liveness
    // argument as reference.hlsl's blocks: fold branch-dead on a cbuffer
    // value, loop recurrence consuming the traced t, [unroll] compile-time
    // indices. The compound guard confines all three blocks to the mode-2
    // arm — a ballast whose update compiled out would "measure" a flat curve.
    float ballast[BALLAST_N];
    [unroll] for (uint bi = 0u; bi < BALLAST_N; ++bi)
        ballast[bi] = float(bi + 1u) * 0.618034 + float(id.x * 7919u + id.y) * 1e-6;
#endif
    for (uint s = 0u; s < spp; ++s) {
        uint rng = rng_init(id.x, id.y, frame, s);
        float2 sp = sample_pos(id.x, id.y, s, rng);
        float3 dir = ray_dir(sp.x, sp.y);

#if defined(DXR_INLINE_SEC) && DXR_INLINE_SEC == 2
        // FR_DXR_INLINE=2: NO TraceRay anywhere — the primary is rt.hlsli's
        // inline trace_closest and shade_full runs right here, i.e. the
        // payload path's raygen + chs_shade + miss_radiance fused (same rng
        // stream, same splat, same probe-sample side channels; the relief
        // re-march lives inside the inline trace_closest's committed block,
        // exactly as in reference.hlsl). Against the compute reference
        // kernel this isolates pure DispatchRays-vs-Dispatch launch cost;
        // against mode 1 it isolates the primary TraceRay + closest-hit
        // stage.
        HitInfo h;
        float3 col;
        float t;
        if (trace_closest(cam_origin.xyz, dir, 0.0, FLT_MAX, h)) {
            PrimSurf ps;
            // Full emissive mask — the DXR pipeline has no quadtree tile to
            // cull from (and measures the whole feature in the noise band).
            col = shade_full(cam_origin.xyz, dir, h, rng, uint2(0xffffffffu, 0xffffffffu), ps);
            t = h.t;
            if (s == probe_sample)
                gbuf_write_hit(pi, sp.x, sp.y, dir, t, ps SWAY_ARG(h.inst));
        } else {
#if SKY_LOD > 1
            col = sky_radiance_lod(dir, id.x, id.y);
#else
            col = sky_radiance(cam_origin.xyz, dir, pixel_cone * 0.5, frame,
                               cloud_dither_k(id, frame, s, spp));
#endif
            t = INF;
            if (s == probe_sample)
                gbuf_write_sky(pi, sp.x, sp.y, dir);
        }
#ifndef ABL_NO_FF_CODE
        if (flags & FLAG_FIREFLIES)
            col += ff_glow(cam_origin.xyz, dir, t, pixel_cone * 0.5);
#endif
#if defined(BALLAST_N) && (BALLAST_N > 0) && defined(DXR_INLINE_SEC) && (DXR_INLINE_SEC == 2)
        // The recurrence — every element loop-carried across the next
        // iteration's trace_closest (reference.hlsl's update, verbatim).
        [unroll] for (uint bi = 0u; bi < BALLAST_N; ++bi)
            ballast[bi] = ballast[bi] * 1.0000001 + (t + float(bi)) * 1e-30;
#endif
        csum += col;
        if (s == probe_sample) {
            tbuf[pi] = t;
            info[pi] = pack_info(0u, KIND_LEAF);
        }
#else
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
#ifndef ABL_NO_FF_CODE
        if (flags & FLAG_FIREFLIES)
            p.color += ff_glow(cam_origin.xyz, dir, p.t, pixel_cone * 0.5);
#endif
        csum += p.color;
        if ((p.prim & 1u) != 0u) {
#if defined(WAVEVIZ) && defined(WAVEVIZ_CHS)
            // FR_WAVEVIZ=chs: the closest-hit owns tbuf while live — keep
            // the chs-written ticket on hit pixels, mark misses with the
            // sentinel (resolve darkens it: "no hit shader ran here").
            if (flags & FLAG_WAVEVIZ) {
                if (isinf(p.t)) tbuf[pi] = asfloat(0xFFFFFFFEu);
            } else {
                tbuf[pi] = p.t;
            }
#else
            tbuf[pi] = p.t;
#endif
            info[pi] = pack_info(0u, KIND_LEAF);
            // Sky G-buffer capture (FLAG_GBUF-gated inside the helper — plain
            // sessions are bit-untouched, and no rng draw is consumed). The
            // hit half lives in chs_shade, where the PrimSurf is.
            if (isinf(p.t))
                gbuf_write_sky(pi, sp.x, sp.y, dir);
        }
#endif // DXR_INLINE_SEC == 2
    }
    float3 c = csum * (1.0 / float(spp));
#if defined(BALLAST_N) && (BALLAST_N > 0) && defined(DXR_INLINE_SEC) && (DXR_INLINE_SEC == 2)
    // Never true — write_cb clamps spp to 1..=MAX_SPP — but the compiler
    // cannot prove that, so the array is not dead (reference.hlsl's fold).
    if (spp == 0xdeadu) {
        float bacc = 0.0;
        [unroll] for (uint bi = 0u; bi < BALLAST_N; ++bi) bacc += ballast[bi];
        c += bacc;
    }
#endif
#if defined(WAVEVIZ) && !defined(WAVEVIZ_CHS)
    // FR_WAVEVIZ: this RAYGEN wave's ticket, minted at kernel END and written
    // as the pixel's LAST tbuf touch. Mode 2 has zero TraceRay, so this IS
    // launch packing; mode 1's composition here is the post-TraceRay
    // continuation's (repacked or not — exactly the question). Slot 2 of the
    // probe buffer is the DXR ticket counter.
    if (flags & FLAG_WAVEVIZ) {
        uint wv_t = 0u;
        if (WaveIsFirstLane()) InterlockedAdd(dxr_width[2], 1u, wv_t);
        tbuf[pi] = asfloat(WaveReadLaneFirst(wv_t));
    }
#endif

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
#endif // DXR_INLINE_SEC == 3 (thin arm)
}

[shader("closesthit")]
void chs_shade(inout RayPayload p, in BuiltInTriangleIntersectionAttributes a) {
#if defined(WAVEVIZ) && defined(WAVEVIZ_CHS)
    // FR_WAVEVIZ=chs: THIS closest-hit wave's ticket — the composition AFTER
    // whatever hit-stage packing the driver did (the TSU question, drawn).
    // Written first so the raygen's post-return sentinel logic (see the
    // mode-0/1 arm) can leave it standing on hit pixels.
    if (flags & FLAG_WAVEVIZ) {
        uint wv_t = 0u;
        if (WaveIsFirstLane()) InterlockedAdd(dxr_width[2], 1u, wv_t);
        uint2 wvid = DispatchRaysIndex().xy;
        tbuf[wvid.y * rw + wvid.x] = asfloat(WaveReadLaneFirst(wv_t));
    }
#endif
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
#ifdef DXR_SBT_RECURSE
    // A recursion-tagged continuation (--dxr-sbt 3, trace_shade): shade THIS
    // surface at the payload's own depth and cone (the sp lanes, bit-punned),
    // return the radiance, and touch NO side channel — the probe bit is 0 by
    // construction, so the guard below skips them, but the early return also
    // keeps the flow shape obvious. This invocation ran in the SURFACE'S OWN
    // class record — the recursive dispatch is the SBT arithmetic, not code.
    if ((p.prim & 0x80000000u) != 0u) {
        PrimSurf psr;
        float3 w3, o3, n3;
        p.color = shade_split(WorldRayOrigin(), WorldRayDirection(), h, p.rng,
                              shadow_samples, ao_samples, reflections != 0u, false,
                              p.sp.y, pixel_cone, true, true,
                              uint2(0xffffffffu, 0xffffffffu), w3, o3, n3, psr,
                              asuint(p.sp.x));
        p.t = h.t;
        return;
    }
#endif
    PrimSurf ps;
    // Full emissive mask — no tile exists on this pipeline (see raygen).
    p.color = shade_full(WorldRayOrigin(), WorldRayDirection(), h, p.rng, uint2(0xffffffffu, 0xffffffffu), ps);
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
    // SWAY_ARG: the CHS has the instance intrinsic directly (the same id
    // tri_of consumed above) — no HitInfo hop needed on this path.
    gbuf_write_hit(id.y * rw + id.x, p.sp.x, p.sp.y, WorldRayDirection(), h.t, ps
                   SWAY_ARG(InstanceID()));
}

[shader("closesthit")]
void chs_hit(inout HitPayload p, in BuiltInTriangleIntersectionAttributes a) {
    p.t = RayTCurrent();
    p.tri = tri_of(InstanceID(), PrimitiveIndex());
    p.u = a.barycentrics.x;
    p.v = a.barycentrics.y;
#if defined(DXR_INLINE_SEC) && DXR_INLINE_SEC == 3
    p.inst = InstanceID();
#endif
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
#if SKY_LOD > 1
    // The amortized cloud lattice (record_frame fills it before DispatchRays),
    // read through the same sky_radiance_lod cs_leaf uses. DispatchRaysIndex()
    // is legal in a miss shader (used on the line below already).
    uint2 mid = DispatchRaysIndex().xy;
    p.color = sky_radiance_lod(WorldRayDirection(), mid.x, mid.y);
#else
    p.color = sky_radiance(WorldRayOrigin(), WorldRayDirection(), pixel_cone * 0.5, frame,
                           cloud_dither_k(DispatchRaysIndex().xy, frame, p.prim >> 1u, spp));
#endif
    p.t = INF;
}

[shader("miss")]
void miss_shadow(inout ShadowPayload p) {
    p.hit = 0u;
}

#ifdef DXR_SBT_RECURSE
// The recursion continuations' miss (index 3): a SENTINEL, nothing more —
// t = INF and the color untouched. The PARENT owns its miss arm (a
// reflection miss needs the parent lobe's MIS weight, a glass miss the
// fixed-phase sky), so computing any sky here would be wasted work at best
// and a double count at worst.
[shader("miss")]
void miss_rec(inout RayPayload p) {
    p.t = INF;
}
#endif

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
// its own body below: it consumes the marched t for the logical interval
// re-check and accumulates the tinted-shadow throughput.
bool ah_reject(uint tri, float u, float v, float tmin, float tmax) {
    float t = RayTCurrent();
    return candidate_reject(tri, WorldRayOrigin(), WorldRayDirection(), t, u, v) != 0u
        || t <= tmin || t >= tmax;
}

[shader("anyhit")]
void ah_shade(inout RayPayload p, in BuiltInTriangleIntersectionAttributes a) {
    // Raygen primaries are logically the complete positive ray.
    if (ah_reject(tri_of(InstanceID(), PrimitiveIndex()), a.barycentrics.x,
                  a.barycentrics.y, 0.0, FLT_MAX))
        IgnoreHit();
}

[shader("anyhit")]
void ah_hit(inout HitPayload p, in BuiltInTriangleIntersectionAttributes a) {
    if (ah_reject(tri_of(InstanceID(), PrimitiveIndex()), a.barycentrics.x,
                  a.barycentrics.y, p.tmin, p.tmax))
        IgnoreHit();
}

// The shadow flavor additionally re-checks the MARCHED t against the
// segment's LOGICAL bounds (ShadowPayload::tmin/tmax). With relief live,
// occluded_q/transmit_q enumerate the full positive base-triangle ray because
// no finite +/-world-depth widening covers grazing incidence. These tests —
// the mirror of the RayQuery loops' explicit `ct > tmin && ct < tmax` — keep
// displaced hits outside the segment from occluding it. Without a live march,
// the helpers preserve the original hardware interval and t == plane t.
[shader("anyhit")]
void ah_shadow(inout ShadowPayload p, in BuiltInTriangleIntersectionAttributes a) {
    float t = RayTCurrent();
    float u = a.barycentrics.x;
    float v = a.barycentrics.y;
    uint tri = tri_of(InstanceID(), PrimitiveIndex());
    if (candidate_reject(tri, WorldRayOrigin(), WorldRayDirection(), t, u, v) != 0u
        || t <= p.tmin || t >= p.tmax)
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
