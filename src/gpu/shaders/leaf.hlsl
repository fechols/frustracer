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
//
// `LEAF_NO_FB` compiles the hemi arm OUT (trace.rs builds this kernel twice —
// see `pso_leaf` / `pso_leaf_fb`). It is not a code-size nicety: `fb_mode` is
// a cbuffer value, so a runtime branch inlines shade_split's (large) body at
// BOTH call sites and the kernel's register allocation is the max over the
// two. VGPR count sets occupancy directly on RDNA, so the fb arm was costing
// every fb-OFF frame its latency hiding. Measured (--gpu-timing, leaf+sky at
// spp=16): -11% on AMD, -16% on NVIDIA — a real win on BOTH, so this is not
// an AMD-specific hack. fb frames take the untouched `pso_leaf_fb`.

[numthreads(LEAF_GROUP, 1, 1)]
void cs_leaf(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint rec_i = flat_group(gid);
    if (rec_i >= counters[push0]) return;
    LeafRec rec = qleaf[rec_i];
    uint2 p0 = rect_min(rec.xy0);
    uint2 p1 = rect_max(rec.xy1);
    uint w = p1.x - p0.x;
    uint npx = w * (p1.y - p0.y);
    // Grid-stride over the tile's pixels, so LEAF_GROUP is a knob instead of
    // being welded to "64 >= the largest tile". A leaf tile is NOT 8x8:
    // depth_full is driven by the WIDER screen axis, so at 1920x1080 a leaf is
    // 1920/2^8 = 7.5 by 1080/2^8 = 4.2 -- about 32 px. Dispatched into 64
    // lanes, half of them used to return immediately. That is nearly free on a
    // wave32 GPU (the all-idle second wave retires at once) but NOT on wave64,
    // where the idle lanes sit in the SAME wave and cost half the RT
    // throughput. See trace.rs's LEAF_GROUP note for the measurement.
    for (uint k = gtid.x; k < npx; k += LEAF_GROUP) {
    uint x = p0.x + k % w;
    uint y = p0.y + k / w;

    uint pi = y * rw + x;
    // --spp: every sample of this pixel reuses the tile's inherited t_start
    // (and, on the CPU, its cut) — they all lie inside the same pixel, hence
    // inside the same tile frustum. The N colors average into ONE partial
    // write, so the compose pass stays the single accum splat site and its
    // store-or-add semantics are untouched. spp == 1 is bit-identical: the
    // loop runs once, salt 0, and the 1.0 divide is exact.
    float3 csum = 0.0, awsum = 0.0;
    for (uint s = 0u; s < spp; ++s) {
        // Identical seeding/jitter policy to the reference kernel — a leaf
        // pixel and a reference pixel draw the same streams (the same-seed A/B
        // relies on it). Sample index rides the salt slot.
        uint rng = rng_init(x, y, frame, s);
        float2 sp = sample_pos(x, y, s, rng);
        float3 dir = ray_dir(sp.x, sp.y);
        // The probe sample owns every per-pixel side channel (tbuf/info and
        // the G-buffer pack, whose guides must stay tied to the jitter the
        // upscaler was told about). 0 in every real frame.
        bool prim = (s == probe_sample);

        HitInfo hit;
        float3 c;
        float3 aw = 0.0;
        float t;
        PrimSurf ps;
        if (trace_closest(cam_origin.xyz, dir, rec.t_start, FLT_MAX, hit)) {
#ifdef LEAF_NO_FB
            {
                c = shade_full(cam_origin.xyz, dir, hit, rng, ps);
            }
#else
            if (fb_mode > 0u) {
                // Hemi mode: shade everything except the primary ambient; hand
                // the surface point to the hemisphere wavefront. One point per
                // PIXEL, never per sample (cap_hemi_pt is rw*rh, and the CPU
                // pins spp to 1 on fb frames anyway).
                float3 o_h, n_h;
                c = shade_split(cam_origin.xyz, dir, hit, rng, shadow_samples, ao_samples,
                                reflections != 0u, true, 0.0, pixel_cone, true, aw, o_h, n_h, ps);
                if (prim) {
                    uint q;
                    InterlockedAdd(counters[CTR_HEMI_PT], 1, q);
                    if (q < cap_hemi_pt) {
                        HemiPointRec pt;
                        pt.o = o_h;
                        pt.pixel = pi;
                        pt.n = n_h;
                        pt._pad = 0;
                        hemi_pts[q] = pt;
                    } else {
                        InterlockedAdd(counters[CTR_OVERFLOW], 1, q);
                    }
                }
            } else {
                c = shade_full(cam_origin.xyz, dir, hit, rng, ps);
            }
#endif
            t = hit.t;
            if (prim) gbuf_write_hit(pi, sp.x, sp.y, dir, hit.t, ps);
        } else {
            // A DISPLAY path (the camera looking at the sky), so it sees the sun
            // DISC — sky.rs's disc-exactly-once rule. The half-angle is the ray's
            // own footprint, which is what antialiases the limb.
            c = sky_radiance(dir, pixel_cone * 0.5);
            t = INF;
            if (prim) gbuf_write_sky(pi, sp.x, sp.y, dir);
        }
        csum += c;
        awsum += aw;
        if (prim) {
            tbuf[pi] = t;
            info[pi] = pack_info(rec.depth, KIND_LEAF);
        }
    }

    uint i3 = pi * 3u;
    float inv = 1.0 / float(spp);
    // The compose pass is the single accum splat site.
    partial[i3 + 0u] = csum.x * inv;
    partial[i3 + 1u] = csum.y * inv;
    partial[i3 + 2u] = csum.z * inv;
    ambw[i3 + 0u] = awsum.x * inv;
    ambw[i3 + 1u] = awsum.y * inv;
    ambw[i3 + 2u] = awsum.z * inv;
    } // grid-stride over the tile's pixels
}
