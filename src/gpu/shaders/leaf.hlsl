// The wavefront leaf pass: one 64-lane group per LeafRec, one thread per
// pixel. Each pixel traces its own jittered primary ray with
// **RayQuery TMin = the tile's inherited t_start** — the ball claim
// transfers exactly (hit acceptance is strictly beyond TMin, and tc was
// shaved by 1e-4 at the advance) — then runs the full shade() port. This is
// M4 landing with M3: the shading code is the already-gated shade_full from
// the reference kernel; only TMin and the info depth/kind differ.
// Secondary rays inside shade_full keep their own tmin chains (never the
// tile's — the CLAUDE.md invariant).
// Requires trace_common.hlsli + queues.hlsli + shade.hlsli pasted first.
// push0 = CTR_LEAF.

[numthreads(64, 1, 1)]
void cs_leaf(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint rec_i = flat_group(gid);
    if (rec_i >= counters[push0]) return;
    LeafRec rec = qleaf[rec_i];
    uint2 p0 = rect_min(rec.xy0);
    uint2 p1 = rect_max(rec.xy1);
    uint w = p1.x - p0.x;
    uint npx = w * (p1.y - p0.y); // <= 64 == the group width
    if (gtid.x >= npx) return;
    uint x = p0.x + gtid.x % w;
    uint y = p0.y + gtid.x / w;

    // Identical seeding/jitter policy to the reference kernel — a leaf pixel
    // and a reference pixel draw the same streams (the same-seed A/B relies
    // on it).
    uint rng = rng_init(x, y, frame, 0u);
    float jx = 0.5, jy = 0.5;
    if (flags & FLAG_FRAME_JITTER) {
        jx = 0.5 + frame_jitter.x;
        jy = 0.5 + frame_jitter.y;
    } else if (flags & FLAG_JITTER) {
        jx = rng_next(rng);
        jy = rng_next(rng);
    }
    float3 dir = ray_dir(float(x) + jx, float(y) + jy);

    uint pi = y * rw + x;
    HitInfo hit;
    float3 c;
    float3 aw = 0.0;
    float t;
    PrimSurf ps;
    if (trace_closest(cam_origin.xyz, dir, rec.t_start, FLT_MAX, hit)) {
        if (fb_mode > 0u) {
            // Hemi mode: shade everything except the primary ambient; hand
            // the surface point to the hemisphere wavefront.
            float3 o_h, n_h;
            c = shade_split(cam_origin.xyz, dir, hit, rng, shadow_samples, ao_samples,
                            reflections != 0u, true, 0.0, pixel_cone, true, aw, o_h, n_h, ps);
            uint s;
            InterlockedAdd(counters[CTR_HEMI_PT], 1, s);
            if (s < cap_hemi_pt) {
                HemiPointRec pt;
                pt.o = o_h;
                pt.pixel = pi;
                pt.n = n_h;
                pt._pad = 0;
                hemi_pts[s] = pt;
            } else {
                InterlockedAdd(counters[CTR_OVERFLOW], 1, s);
            }
        } else {
            c = shade_full(cam_origin.xyz, dir, hit, rng, ps);
        }
        t = hit.t;
        gbuf_write_hit(pi, float(x) + jx, float(y) + jy, dir, hit.t, ps);
    } else {
        c = sky_color(dir);
        t = INF;
        gbuf_write_sky(pi, float(x) + jx, float(y) + jy, dir);
    }

    uint i3 = pi * 3u;
    // The compose pass is the single accum splat site.
    partial[i3 + 0u] = c.x;
    partial[i3 + 1u] = c.y;
    partial[i3 + 2u] = c.z;
    ambw[i3 + 0u] = aw.x;
    ambw[i3 + 1u] = aw.y;
    ambw[i3 + 2u] = aw.z;
    tbuf[pi] = t;
    info[pi] = pack_info(rec.depth, KIND_LEAF);
}
