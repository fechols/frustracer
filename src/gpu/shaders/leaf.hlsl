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
//
// THIS KERNEL HAS ZERO GROUPSHARED, AND THAT IS LOAD-BEARING ON INTEL.
// It does not paste frustum.hlsli (see trace.rs's `leaf_of` source list), so
// unlike the level kernels it carries no LDS at all — and Intel's Arc RT
// developer guide (v4, p.26) states that groupshared is allocated out of the
// same L1 that services the ray-tracing unit, so LDS in a RayQuery kernel
// degrades ray throughput directly, "even if [it] is running on a different
// queue". Every ray this renderer traces comes from here or hemi_leaf.hlsl,
// and neither has any. Do NOT introduce a groupshared scratchpad here as a
// tiling/caching optimisation without measuring ray cost on Arc first; the
// per-lane software traversal stacks under --sw-rays were kept in private
// scratch for the register/occupancy reason (rt_sw.hlsli's header), and this
// is the second, independent reason that was the right call. Cross-lane work
// belongs in wave intrinsics, which need no LDS — see ctr.hlsli's ctr_add.

[numthreads(LEAF_GROUP, 1, 1)]
void cs_leaf(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint rec_i = flat_group(gid);
    if (rec_i >= counters[push0]) return;
    LeafRec rec = qleaf[rec_i];
    uint2 p0 = rect_min(rec.xy0);
    uint2 p1 = rect_max(rec.xy1);
    uint w = p1.x - p0.x;
    uint npx = w * (p1.y - p0.y);
#ifdef SW_RAYS
    // One beam-produced opaque handle is shared by every ray/sample in this
    // record. The root control executes the same telemetry atomics with zero
    // values, keeping the continuation-vs-root timing comparison symmetric.
    if (gtid.x == 0u) frontier_record_reuse(rec.frontier, npx * spp);
#endif
    // Per-TILE emissive light cull (emissive::cull_tile's GPU twin): one
    // conservative uint2 mask over the ≤64 cluster lights, from the tile
    // RECT + inherited t_start — group-uniform inputs, so every thread
    // computes the identical mask. Placed HERE, before the grid-stride loop,
    // where control flow is still group-uniform (npx can be < LEAF_GROUP, so
    // some lanes never enter the loop). Deliberately a SERIAL uniform loop,
    // not a wave-cooperative build: a lane-strided WaveActiveBitOr silently
    // drops lights at FR_LGROUP below the wave width (inactive lanes), and
    // ≤64 iterations of pure ALU once per ~LEAF_TILE²·spp pixels is noise
    // (the wave-aggregation campaign's own verdict on micro-coordination).
    // The cull is EXACT — a culled light fails every pixel's d2 >= r_infl2
    // test — so the reference kernel's full mask stays the bit-identical
    // A/B oracle. ABL_NO_EL_CULL (FR_ABL=noelcull) is the full-mask probe.
    uint2 el_mask = uint2(0xffffffffu, 0xffffffffu);
#ifndef ABL_NO_EL_CULL
    if (flags & FLAG_EMISSIVE) {
        TF tf = tile_frustum(p0.x, p0.y, p1.x, p1.y);
        el_mask = uint2(0u, 0u);
        for (uint ei = 0u; ei < EL_COUNT; ++ei) {
            if (el_tile_culled(tf, rec.t_start, ei)) continue;
            if (ei < 32u) el_mask.x |= 1u << ei;
            else         el_mask.y |= 1u << (ei - 32u);
        }
    }
#endif
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
        // WHAT THE INHERITED t_start IS ACTUALLY WORTH — measured, so nobody
        // has to assume. Ablation: trace from 0 instead (sound; t_start lower-
        // bounds the nearest hit, so the same hit is found and only traversal
        // is paid), keeping the rest of the quadtree intact. `--spin path`
        // 1080p spp=16, 3 interleaved reps, medians:
        //   Arc Pro B70   default +1.7%   stress +1.1%   powerplant +1.7%
        //   RTX 4090      default -3.5%   stress -7.1%   powerplant +5.2%
        // Intel is consistent and small; NVIDIA straddles zero, i.e. free and
        // worth nothing — the same verdict CLAUDE.md already records for AMD's
        // documented TMin re-origining. So on the ONE GPU where the quadtree
        // beats a plain per-pixel reference (Intel, see trace::WIDE_LEVELS and
        // the --spp cost model), the inherited bound explains only 7-21% of
        // that advantage; the rest is tiles proven empty tracing NO rays.
        // Re-run by making this a 0.0 and rebuilding.
#ifdef SW_RAYS_LEAF
        // Software simulation of the proposed hardware seam: the ray stage
        // consumes an opaque, beam-produced traversal frontier. It cannot
        // inspect the cut arena or BVH ids. --no-cut-rays leaves this define
        // absent and gives the same SW intersector a root traversal instead.
        if (trace_closest_frontier(rec.frontier, cam_origin.xyz, dir,
                                   rec.t_start, FLT_MAX, hit)) {
#else
        if (trace_closest(cam_origin.xyz, dir, rec.t_start, FLT_MAX, hit)) {
#endif
#ifdef LEAF_NO_FB
            {
                c = shade_full(cam_origin.xyz, dir, hit, rng, el_mask, ps);
            }
#else
            if (fb_mode > 0u) {
                // Hemi mode: shade everything except the primary ambient; hand
                // the surface point to the hemisphere wavefront. One point per
                // PIXEL, never per sample (cap_hemi_pt is rw*rh, and the CPU
                // pins spp to 1 on fb frames anyway).
                float3 o_h, n_h;
                c = shade_split(cam_origin.xyz, dir, hit, rng, shadow_samples, ao_samples,
                                reflections != 0u, true, 0.0, pixel_cone, true, true,
                                el_mask, aw, o_h, n_h, ps);
                // Wave-aggregated: this fires once per HIT PIXEL, ~2M times to
                // a single address at 1080p, from a 256-thread group. The
                // reservation is hoisted out of the `if (prim)` so the whole
                // wave reaches it — non-participants take 0 slots.
                {
                    uint q = ctr_add(CTR_HEMI_PT, prim ? 1u : 0u);
                    if (prim && q < cap_hemi_pt) {
                        HemiPointRec pt;
                        pt.o = o_h;
                        pt.pixel = pi;
                        pt.n = n_h;
                        pt._pad = 0;
                        hemi_pts[q] = pt;
                    } else if (prim) {
                        // Overflow only — a lane that reserved nothing is not
                        // an overflow, so `prim` has to be re-tested here.
                        InterlockedAdd(counters[CTR_OVERFLOW], 1, q);
                    }
                }
            } else {
                c = shade_full(cam_origin.xyz, dir, hit, rng, el_mask, ps);
            }
#endif
            t = hit.t;
            if (prim) gbuf_write_hit(pi, sp.x, sp.y, dir, hit.t, ps SWAY_ARG(hit.inst));
        } else {
            // A DISPLAY path (the camera looking at the sky), so it sees the sun
            // DISC — sky.rs's disc-exactly-once rule. The half-angle is the ray's
            // own footprint, which is what antialiases the limb. The cloud march
            // phase is per (pixel, frame, SAMPLE) — render.rs's dither_jk twin.
#if SKY_LOD > 1
            // Sky inside a LEAF tile — not a proven-empty region, but the cloud
            // field does not care about geometry, so the same screen-space
            // lattice serves it. Measured: the rect-only lattice reached just
            // 27% of the cloud bill; this is where most of the rest was.
            c = sky_radiance_lod(dir, x, y);
#else
            c = sky_radiance(cam_origin.xyz, dir, pixel_cone * 0.5, frame,
                             cloud_dither_k(uint2(x, y), frame, s, spp));
#endif
            t = INF;
            if (prim) gbuf_write_sky(pi, sp.x, sp.y, dir);
        }
        // Firefly glow, depth-tested against this sample's own hit (INF on a
        // miss) — render.rs::shade_traced's composite, term for term. Guarded
        // by the flag, so day frames add nothing (bit-identity).
        if (flags & FLAG_FIREFLIES) c += ff_glow(cam_origin.xyz, dir, t, pixel_cone * 0.5);
        csum += c;
        awsum += aw;
        if (prim) {
            tbuf[pi] = t;
            info[pi] = pack_info(rec.depth, KIND_LEAF);
        }
    }

    float inv = 1.0 / float(spp);
#ifdef LEAF_NO_FB
    // fb OFF (this kernel IS the fb-off PSO): straight into accum, and
    // trace.rs skips the compose dispatch entirely — see `accum_splat`.
    // `awsum` is identically zero on this arm, so nothing is lost.
    accum_splat(pi, csum * inv);
#else
    // fb ON: the hemi wavefront runs after this pass and compose folds its
    // ambient in, so the handoff through partial/ambw stays.
    uint i3 = pi * 3u;
    partial[i3 + 0u] = csum.x * inv;
    partial[i3 + 1u] = csum.y * inv;
    partial[i3 + 2u] = csum.z * inv;
    ambw[i3 + 0u] = awsum.x * inv;
    ambw[i3 + 1u] = awsum.y * inv;
    ambw[i3 + 2u] = awsum.z * inv;
#endif
    } // grid-stride over the tile's pixels
}
