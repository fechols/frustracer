// The sky-tile fill: `cs_sky_lod` (the amortized cloud lattice) + `cs_sky`
// (the per-pixel composite). Requires trace_common.hlsli + ctr.hlsli +
// queues.hlsli pasted first, with SKY_UNIT defined.
//
// WHY ITS OWN COMPILE UNIT: `cloud_lod` takes u5, which is the tile queue's
// register. The tile queues are dead by the time the sky fill runs, and each
// compile unit re-declares what its registers mean in that phase — the same
// idiom `record_hemi` uses when it rebinds u7/u9 to the hemi buffers. The
// alternative was a 15th root UAV, and the root signature is at 62/64 DWORDs.
//
// THE IDEA. A proven-empty tile pays the full dome+cloud integral per pixel,
// and the cloud march is 35% of a whole sample on the B70 while casting ZERO
// rays. The quadtree's proof hands us a contiguous region every pixel of which
// is known — before any shading — to need that same expensive term. So
// evaluate it on a lattice at 1/SKY_LOD^2 of pixel rate and interpolate,
// keeping the SHARP half (sun limb, stars) per-pixel and exact.
//
// SKY_LOD == 1 compiles the lattice out entirely and runs the original
// per-pixel path verbatim — the off state is bit-identical by construction,
// not by tolerance (the --no-clouds / --aniso 1 precedent).

// The lattice declaration + accessors live in skylod.hlsli (shared with
// leaf.hlsl, which shades the sky pixels inside leaf tiles). This unit owns
// only the lattice FILL and the proven-empty-rect composite.

// --- pass 1: fill the lattice over the whole screen ------------------------
// A plain full-screen dispatch: the lattice covers every pixel because BOTH
// consumers read it, and the screen size is known on the CPU so no indirect
// args are needed. Runs before leaf+sky, with one barrier.

[numthreads(64, 1, 1)]
void cs_sky_lod(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
#if SKY_LOD > 1
    uint i = flat_group(gid) * 64u + gtid.x;
    uint lw = sky_lod_w();
    if (i >= lw * sky_lod_h()) return;
    uint lx = i % lw;
    uint ly = i / lw;
    uint px = lx << SKY_LOD_LOG;
    uint py = ly << SKY_LOD_LOG;
    float3 dir = ray_dir(float(px) + 0.5, float(py) + 0.5);
    // The march phase is the lattice POINT's own stratum — the same pure
    // function of (pixel, frame, sample) the per-pixel path uses, evaluated
    // where the samples actually are.
    cloud_lod[i] = sky_cloud4(cam_origin.xyz, dir,
                             cloud_dither_k(uint2(px, py), frame, 0u, spp));
#endif
}

// --- pass 2: the per-pixel composite ---------------------------------------
// render.rs::fill_sky: pixel-CENTRE dirs (no jitter, no rng), KIND_SKY at the
// depth the tile was proven.
//
// A sky RECT is not a tile-sized thing. It is whatever rect the quadtree proved
// empty, so its area is (rw*rh)/4^d for the depth d it was proven at — a
// depth-2 sky tile at 1080p is 480x270 = 129,600 pixels. This kernel used to
// run ONE group per record and grid-stride that whole rect inside it, which is
// ~2,025 serial laps for one group while the rest of the GPU idles. That was
// invisible for as long as a sky pixel was a cheap dome+disc evaluation; the
// volumetric cloud march made each sky pixel ~100x dearer and turned the
// imbalance into the single dominant cost of the whole wavefront tracer.
// Measured (--spin path, 1080p, GPU time, cloud march neutralized in THIS
// kernel only, leaf keeping it):
//   Arc Pro B70  default 8.95 -> 2.58 ms   stress 5000 13.07 -> 3.87 ms
//   RTX 4090     default 4.99 -> 1.65 ms   stress 5000  7.32 -> 2.84 ms
// The DXR pipeline never had it — DispatchRays gives every sky pixel its own
// thread — which is exactly why `--dxr` measured FASTER than `--gpu` on Arc
// with clouds on while `--no-clouds` measured the reverse.
//
// SKY_SPLIT groups now share each record's rect, interleaved by group so the
// stride stays coalesced. {sub*SKY_GROUP + gtid.x} over sub in [0,SKY_SPLIT)
// and gtid in [0,SKY_GROUP) is exactly [0, SKY_GROUP*SKY_SPLIT), so striding by
// that product PARTITIONS [0,total): every pixel is still written by exactly
// one (record, sub, lane) triple, and --check-gpu's queue-accounting and
// coverage gates are unmoved by construction.
//
// Note SKY_GROUP is NOT LEAF_GROUP and must not be unified with it — see the
// sweep table on the consts in trace.rs. A leaf tile is ~32 px so one wave
// covers it; a sky rect is thousands, so lanes are parallelism, not waste.

[numthreads(SKY_GROUP, 1, 1)]
void cs_sky(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint g = flat_group(gid);
    uint rec_i = g / SKY_SPLIT;
    uint sub = g % SKY_SPLIT;
    if (rec_i >= counters[push0]) return;
    SkyRec rec = qsky[rec_i];
    uint2 p0 = rect_min(rec.xy0);
    uint2 p1 = rect_max(rec.xy1);
    uint w = p1.x - p0.x;
    uint total = w * (p1.y - p0.y);
    // Small rects leave most of their SKY_SPLIT groups with nothing to do; they
    // fail this bound on the first test and retire. That is the same "empty
    // dispatches cost nothing" trade the level ladder already makes.
    for (uint i = sub * SKY_GROUP + gtid.x; i < total; i += SKY_GROUP * SKY_SPLIT) {
        uint x = p0.x + i % w;
        uint y = p0.y + i / w;
        // Sample 0 sits where the FRAME SAYS it sits — render.rs::fill_sky_rows
        // term for term, and the same `sample_pos` rule cs_leaf / reference /
        // dxr already use. A proven-empty tile traces no rays but still writes
        // this pixel's color and G-buffers, and every temporal consumer is told
        // ONE sub-pixel offset per frame; hard-coding the center here while
        // declaring a Halton offset anyway is a registration lie (the MV stays
        // correct — it subtracts whatever position we pass — but the
        // reconstructor places the content at center+jitter when it was sampled
        // at center, and the jitter moves every frame). Same rng stream a leaf
        // pixel at these coordinates would take, so the legacy-jitter arm now
        // agrees with reference.hlsl too.
        uint srng = rng_init(x, y, frame, 0u);
        float2 sp = sample_pos(x, y, 0u, srng);
        float3 dir = ray_dir(sp.x, sp.y);
        // March-phase dither per (pixel, frame, SAMPLE) — the center rides
        // sample 0's stratum, mirroring render.rs::fill_sky_rows verbatim.
        float cj = cloud_dither_k(uint2(x, y), frame, 0u, spp);
#if SKY_LOD > 1
        // The amortized path: sharp half per pixel, expensive half interpolated
        // off the lattice. One fetch serves every sample of this pixel — the
        // extra --spp samples differ only in their BACKDROP direction, which is
        // where the sub-pixel structure the spp path exists for actually lives.
        float4 cl = sky_cloud_lerp(x, y);
        float3 c = sky_compose(sky_backdrop(dir, pixel_cone * 0.5, frame), cl);
#else
        float3 c = sky_radiance(cam_origin.xyz, dir, pixel_cone * 0.5, frame, cj);
#endif
        // Firefly glow against a proven-empty tile — still zero rays (t_max
        // ∞, nothing occludes a sky direction). render.rs::fill_sky_rows,
        // term for term; the flag guard keeps day frames bit-identical.
        if (flags & FLAG_FIREFLIES)
            c += ff_glow(cam_origin.xyz, dir, INF, pixel_cone * 0.5);
        // Under CLOUDS the sky has sub-pixel structure (a cover-ramp edge
        // crosses a pixel), so a multi-sampled frame averages spp sample
        // positions — render.rs::fill_sky_rows, term for term (still zero
        // rays; side channels stay sample 0's). The DIRECTION offsets are the
        // PHASE-0 Halton table (SKY_J, injected by trace::spp_defs),
        // frame-independent AND ANCHORED TO THE PIXEL CENTER — deliberately
        // NOT translated with sample 0's declared position above.
        //
        // Both halves matter at night, where the spp stability gate (spp=4
        // strictly quieter than spp=1) has teeth. Writing d for the per-frame
        // offset and G for the sky's screen gradient, the first-order
        // inter-frame difference is:
        //
        //   extras move, sample 0 fixed    spp1 0          spp4 (d0-d1)*G
        //   whole set translated rigidly   spp1 (d0-d1)*G  spp4 (d0-d1)*G
        //   THIS: sample 0 moves, anchored spp1 (d0-d1)*G  spp4 (d0-d1)*G/4
        //
        // Only the last row leaves spp=4 quieter: averaging over a rigidly
        // translated stratified set does not attenuate a translation, only the
        // aliasing residual. The march PHASE is per-sample and frame-keyed,
        // symmetric across spp levels, which that gate accepts. The guard
        // keeps spp == 1 and cloudless skies on the old path verbatim.
        if ((flags & FLAG_CLOUDS) && spp > 1u) {
            for (uint s = 1u; s < spp; ++s) {
                float2 o = SKY_J[s];
                float3 ds = ray_dir(float(x) + 0.5 + o.x, float(y) + 0.5 + o.y);
#if SKY_LOD > 1
                c += sky_compose(sky_backdrop(ds, pixel_cone * 0.5, frame), cl);
#else
                c += sky_radiance(cam_origin.xyz, ds, pixel_cone * 0.5, frame,
                                  cloud_dither_k(uint2(x, y), frame, s, spp));
#endif
                // Each extra sample carries its own glow along its own
                // direction, exactly like a leaf pixel's sample loop.
                if (flags & FLAG_FIREFLIES)
                    c += ff_glow(cam_origin.xyz, ds, INF, pixel_cone * 0.5);
            }
            c *= 1.0 / float(spp);
        }
        uint pi = y * rw + x;
        // Sky pixels have no ambient weight in either mode. With fb OFF that
        // makes compose a pure copy, so splat straight into accum (see
        // `accum_splat`); the branch is on a cbuffer value, uniform across the
        // dispatch. cs_sky is ONE PSO — unlike cs_leaf it has no fb-off
        // compile arm to hang this on.
        if (fb_mode == 0u) {
            accum_splat(pi, c);
        } else {
            // fb ON: compose still runs over EVERY pixel, so sky must keep
            // feeding it — a sky pixel that skipped `partial` would have
            // compose read whatever was there last frame.
            uint i3 = pi * 3u;
            partial[i3 + 0u] = c.x;
            partial[i3 + 1u] = c.y;
            partial[i3 + 2u] = c.z;
            ambw[i3 + 0u] = 0.0;
            ambw[i3 + 1u] = 0.0;
            ambw[i3 + 2u] = 0.0;
        }
        tbuf[pi] = INF;
        info[pi] = pack_info(rec.depth, KIND_SKY);
        // Sample 0's DECLARED position, matching the CPU sky-tile flood and
        // leaf.hlsl alike — one convention, and `dir` is its exact preimage,
        // which is what makes the direction-reprojection MV self-consistent.
        gbuf_write_sky(pi, sp.x, sp.y, dir);
    }
}

// --- the cloud shadow cache fill -------------------------------------------
// One thread per cell of the slab-space grid (trace_common.hlsli explains why
// the reduction to (M.x, M.z) is EXACT). Dispatched once per frame before any
// shading, with one barrier; every shading path then reads it through
// `cloud_sun_transmittance`, so leaf AND reference both get it and the
// same-seed image A/B keeps its teeth.

[numthreads(64, 1, 1)]
void cs_cloud_shadow(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
#if CLOUD_SHADOW_N > 0
    uint i = flat_group(gid) * 64u + gtid.x;
    // The live side rides the CB; the dispatch covers the cap, so the tail
    // retires here.
    uint n = uint(cloud_grid.w);
    if (n == 0u || i >= n * n) return;
    uint cx = i % n;
    uint cy = i / n;
    float cell = 1.0 / cloud_grid.z;
    float2 mxz = cloud_grid.xy + float2(float(cx), float(cy)) * cell;
    cloud_shadow[i] = ((flags & FLAG_CLOUDS) == 0u || sun.y <= CLOUD_SUN_MIN_Y)
                          ? 1.0
                          : cloud_shadow_at(mxz);
#endif
}
