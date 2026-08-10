// The vanilla GPU tracer: one thread per pixel, RayQuery from the TLAS root
// with TMin = 0, full shading — render.rs's plain per-pixel reference (the
// R-key baseline) re-hosted. Doubles as the on-GPU reference for the
// wavefront gates (M3+) and validates the whole shading/RT/present stack
// with zero queue machinery. Requires trace_common.hlsli + shade.hlsli
// pasted first.

RWStructuredBuffer<float> accum : register(u0); // rw*rh*3, CPU layout parity
RWStructuredBuffer<float> tbuf  : register(u1); // primary-hit t, INF = sky
RWStructuredBuffer<uint>  info  : register(u2); // pack_info(depth, kind)

#ifdef WIDTH_PROBE
// FR_WIDTH: the counters buffer, bound at u3 by bind_common for every
// dispatch on this pipeline (record_reference included). Width slots
// (>= CTR_COUNT) ONLY — deliberately NOT ctr.hlsli and NOT HAVE_COUNTERS,
// so rt.hlsli's gated stat increments stay compiled out of this unit (the
// ctr.hlsli contract note; CTR_W_REFERENCE is injected by trace::width_defs).
RWStructuredBuffer<uint> width_ctr : register(u3);
#endif

[numthreads(8, 8, 1)]
void cs_reference(uint3 id : SV_DispatchThreadID) {
#ifdef WIDTH_PROBE
    // Report this kernel's COMPILED wave width once (thread 0,0) — the
    // per-shader SIMD choice the driver made for THIS kernel's pressure.
    if (id.x == 0u && id.y == 0u) width_ctr[CTR_W_REFERENCE] = WaveGetLaneCount();
#endif
#ifdef WAVEVIZ
    // --waveviz: this wave's ID is its first lane's PIXEL — unique per wave
    // within a frame (a pixel belongs to exactly one wave) and identical
    // across frames whenever the packing is identical, so a parked view is
    // color-STABLE and residual shimmer is real repacking. Minted before the
    // early-out (converged). An arrival-order atomic ticket was tried first
    // and strobed: wave scheduling order is nondeterministic per frame.
    uint wv_t = WaveReadLaneFirst(id.y * rw + id.x);
#endif
    if (id.x >= rw || id.y >= rh) return;

    uint pi = id.y * rw + id.x;
    // --spp: the leaf kernel's sample loop, minus the tile claim (this kernel
    // traces from the root with TMin = 0). Same seeding, same probe rule — the
    // same-seed wavefront-vs-reference A/B has to keep holding at every spp.
    float3 csum = 0.0;
#if defined(BALLAST_N) && (BALLAST_N > 0)
    // FR_BALLAST: BALLAST_N synthetic live floats — the register-cliff
    // bisection lever. The liveness argument, one part per compiler escape:
    //  (a) NOT DEAD: the fold below reads every element under a branch on a
    //      cbuffer value (spp == 0xdead) the compiler cannot disprove; spp
    //      is clamped to 1..=MAX_SPP at write_cb, so the branch never runs
    //      and the image stays BIT-IDENTICAL at every real spp.
    //  (b) NOT SINKABLE / REMATERIALIZABLE: each element is a LOOP
    //      RECURRENCE whose increment consumes `t` — the traced distance —
    //      so reconstructing an element after the loop would require
    //      re-tracing, and sinking the chain out of the loop would mean
    //      distributing the loop across a RayQuery, which neither DXC nor
    //      IGC performs.
    //  (c) NOT MEMORY-DEMOTED: [unroll] makes every index compile-time, so
    //      the array lives in registers unless the compiler SPILLS — which
    //      is exactly the pressure being simulated.
    float ballast[BALLAST_N];
    [unroll] for (uint bi = 0u; bi < BALLAST_N; ++bi)
        ballast[bi] = float(bi + 1u) * 0.618034 + float(id.x * 7919u + id.y) * 1e-6;
#endif
    for (uint s = 0u; s < spp; ++s) {
        uint rng = rng_init(id.x, id.y, frame, s);
        float2 sp = sample_pos(id.x, id.y, s, rng);
        float3 dir = ray_dir(sp.x, sp.y);
        bool prim = (s == probe_sample);

        HitInfo hit;
        float3 c;
        float t;
        PrimSurf ps;
        if (trace_closest(cam_origin.xyz, dir, 0.0, FLT_MAX, hit)) {
            // Full emissive mask: the reference deliberately never culls —
            // it is the unculled oracle the leaf kernel's tile mask is
            // proven against (the same-seed A/B stays exact because a
            // culled light contributes exactly zero).
            c = shade_full(cam_origin.xyz, dir, hit, rng, uint2(0xffffffffu, 0xffffffffu), ps);
            t = hit.t;
            if (prim) gbuf_write_hit(pi, sp.x, sp.y, dir, hit.t, ps SWAY_ARG(hit.inst));
        } else {
            // A DISPLAY path: the camera's own miss sees the sun DISC (sky.rs).
            // Cloud phase per (pixel, frame, SAMPLE) — the leaf kernel's twin.
#if SKY_LOD > 1
            // The amortized cloud lattice (record_sky_lod fills it before this
            // dispatch), read through the EXACT same sky_radiance_lod cs_leaf
            // uses — this is what keeps the wavefront-vs-reference image A/B
            // bit-identical at the default-ON K. Keep textually in lockstep with
            // leaf.hlsl's arm.
            c = sky_radiance_lod(dir, id.x, id.y);
#else
            c = sky_radiance(cam_origin.xyz, dir, pixel_cone * 0.5, frame,
                             cloud_dither_k(id.xy, frame, s, spp));
#endif
            t = INF;
            if (prim) gbuf_write_sky(pi, sp.x, sp.y, dir);
        }
        // Firefly glow — the leaf kernel's composite, term for term (the
        // same-seed wavefront-vs-reference bit gate rides on that; the
        // ABL_NO_FF_CODE probe strips BOTH kernels, so it still does).
#ifndef ABL_NO_FF_CODE
        if (flags & FLAG_FIREFLIES) c += ff_glow(cam_origin.xyz, dir, t, pixel_cone * 0.5);
#endif
#if defined(BALLAST_N) && (BALLAST_N > 0)
        // The recurrence: every element's PREVIOUS value is consumed after
        // the next iteration's trace_closest, i.e. all BALLAST_N floats are
        // loop-carried across the traversal — the footprint being simulated.
        [unroll] for (uint bi = 0u; bi < BALLAST_N; ++bi)
            ballast[bi] = ballast[bi] * 1.0000001 + (t + float(bi)) * 1e-30;
#endif
        csum += c;
        if (prim) {
            tbuf[pi] = t;
            info[pi] = pack_info(0u, KIND_LEAF);
        }
    }
    float3 c = csum * (1.0 / float(spp));
#if defined(BALLAST_N) && (BALLAST_N > 0)
    // Never true — write_cb clamps spp to 1..=MAX_SPP (128) — but the
    // compiler cannot prove that, so the array is not dead. Bit-identical
    // image at every real spp.
    if (spp == 0xdeadu) {
        float bacc = 0.0;
        [unroll] for (uint bi = 0u; bi < BALLAST_N; ++bi) bacc += ballast[bi];
        c += bacc;
    }
#endif
#ifdef WAVEVIZ
    // The kernel's LAST tbuf touch: this pixel now carries its wave's ticket
    // (asfloat bit-cast — float(t) would quantize past 2^24) for the resolve
    // colorizer. accum keeps the real image, so toggling off is clean.
    if (flags & FLAG_WAVEVIZ) tbuf[pi] = asfloat(wv_t);
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
}
