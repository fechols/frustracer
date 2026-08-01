// The vanilla GPU tracer: one thread per pixel, RayQuery from the TLAS root
// with TMin = 0, full shading — render.rs's plain per-pixel reference (the
// R-key baseline) re-hosted. Doubles as the on-GPU reference for the
// wavefront gates (M3+) and validates the whole shading/RT/present stack
// with zero queue machinery. Requires trace_common.hlsli + shade.hlsli
// pasted first.

RWStructuredBuffer<float> accum : register(u0); // rw*rh*3, CPU layout parity
RWStructuredBuffer<float> tbuf  : register(u1); // primary-hit t, INF = sky
RWStructuredBuffer<uint>  info  : register(u2); // pack_info(depth, kind)

[numthreads(8, 8, 1)]
void cs_reference(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;

    uint pi = id.y * rw + id.x;
    // --spp: the leaf kernel's sample loop, minus the tile claim (this kernel
    // traces from the root with TMin = 0). Same seeding, same probe rule — the
    // same-seed wavefront-vs-reference A/B has to keep holding at every spp.
    float3 csum = 0.0;
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
        csum += c;
        if (prim) {
            tbuf[pi] = t;
            info[pi] = pack_info(0u, KIND_LEAF);
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
