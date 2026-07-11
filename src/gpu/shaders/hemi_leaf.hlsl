// Hemisphere leaf pass (hemi.rs::leaf_rays at LEAF_LEVELS = 1): 4 threads
// per leaf cell, one stratified Arvo ray per midpoint sub-cell, RayQuery
// with TMin = the cell's inherited tc from the shading-point apex (its own
// tmin chain — never the primary tile's). AO: any-hit clamped to ao_radius;
// GI: closest hit shaded at the BOUNCE_Q policy (1 shadow sample, no AO, no
// reflection — structurally recursion-free), sky on miss. Results land in
// the fixed-point H accumulator (order-independent adds => reproducible).
// RNG is keyed by (pixel, hemi path, frame, HEMI_SALT) — the path, never the
// atomic slot index, so results are queue-order independent.
// Requires trace_common.hlsli + ctr.hlsli + hemi.hlsli + rt.hlsli +
// shade.hlsli pasted first.

[numthreads(32, 1, 1)]
void cs_hemi_leaf(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid = flat_group(gid) * 32u + gtid.x;
    uint rec_i = tid >> 2;
    if (rec_i >= counters[CTR_HEMI_LEAF]) return;
    HemiCellRec rec = hqleaf[rec_i];
    HemiPointRec pt = hemi_pts[rec.point_id];
    uint sub = tid & 3u;
    uint dummy;

    float3 mab, mbc, mca;
    midpoints3(rec.a, rec.b, rec.c, mab, mbc, mca);
    float3 a = sub == 0u ? rec.a : (sub == 1u ? mab : (sub == 2u ? mca : mab));
    float3 b = sub == 0u ? mab : (sub == 1u ? rec.b : (sub == 2u ? mbc : mbc));
    float3 c = sub == 0u ? mca : (sub == 1u ? mbc : (sub == 2u ? rec.c : mca));

    uint path = ((rec.meta >> 8) << 2) | sub;
    uint rng = rng_init(pt.pixel, path, frame, HEMI_SALT);
    float u0 = rng_next(rng);
    float u1 = rng_next(rng);
    float3 d = sample_tri(a, b, c, u0, u1);
    float weight = max(dot(d, pt.n), 0.0) * solid_angle3(a, b, c);
    float t_lim = hemi_t_limit();
    float tc = rec.t_start;
    InterlockedAdd(counters[CTR_HEMI_RAYS], 1, dummy);

    if (flags & FLAG_VERIFY) {
        hemi_add(pt.pixel, 3, psa3(a, b, c, pt.n));
        // tmin soundness: a tmin=0 reference ray must not hit strictly
        // inside the claimed-empty ball (hemi.rs::verify_leaf_ray).
        HitInfo vh;
        if (trace_closest(pt.o, d, 0.0, FLT_MAX, vh) && vh.t < tc * (1.0 - 1e-3)) {
            InterlockedAdd(counters[CTR_V_TMIN], 1, dummy);
        }
    }

    if (fb_mode == 1u) {
        if (!occluded_q(pt.o, d, tc, t_lim)) {
            hemi_add(pt.pixel, 0, weight);
        }
    } else {
        HitInfo h;
        if (trace_closest(pt.o, d, tc, FLT_MAX, h)) {
            float3 w3, o3, n3;
            PrimSurf ps_unused; // bounce rays never capture (secondary-ray rule)
            float3 l = shade_split(pt.o, d, h, rng, 1u, 0u, false, false, w3, o3, n3, ps_unused);
            hemi_add3(pt.pixel, l * weight);
        } else {
            hemi_add3(pt.pixel, sky_color(d) * weight);
        }
    }
}
