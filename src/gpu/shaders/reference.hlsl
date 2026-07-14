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

    uint pi = id.y * rw + id.x;
    HitInfo hit;
    float3 c;
    float t;
    PrimSurf ps;
    if (trace_closest(cam_origin.xyz, dir, 0.0, FLT_MAX, hit)) {
        c = shade_full(cam_origin.xyz, dir, hit, rng, ps);
        t = hit.t;
        gbuf_write_hit(pi, float(id.x) + jx, float(id.y) + jy, dir, hit.t, ps);
    } else {
        c = sky_radiance(dir, pixel_cone * 0.5);
        t = INF;
        gbuf_write_sky(pi, float(id.x) + jx, float(id.y) + jy, dir);
    }

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
    tbuf[pi] = t;
    info[pi] = pack_info(0u, KIND_LEAF);
}
