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
        //
        // ONLY for a sample the integrand actually uses. Arvo sampling of a
        // horizon-adjacent cell can land fp-epsilon BELOW the tangent plane;
        // such a direction carries `weight == max(dot(d,n),0) == 0` and
        // contributes literally nothing, and no claim was ever made about it
        // — the hemi ROOT CUT *is* the tangent half-space, so the bound query
        // proves things about the open hemisphere and nothing else. Traced
        // from tmin=0 it grazes back down onto the apex's OWN surface (which
        // sits at -eps) at t = eps/|d.n|, an artifact of the eps offset
        // rather than occlusion, and at a grazing angle that t is huge:
        // measured on Intel Arc, d.n = -4.31e-4 put the own ground plane at
        // t = 39.36 inside a correctly-claimed empty ball of 57.26 (a "31%
        // violation" that is really a zero-weight ray hitting the floor it
        // stands on). NVIDIA/AMD round the sample the other way and never
        // trip it, which is luck, not soundness — the guard is the invariant.
        if (dot(d, pt.n) > 0.0) {
            HitInfo vh;
            if (trace_closest(pt.o, d, 0.0, FLT_MAX, vh) && vh.t < tc * (1.0 - 1e-3)) {
                InterlockedAdd(counters[CTR_V_TMIN], 1, dummy);
            }
        }
    }

    if (fb_mode == 1u) {
        // transmit_q (hemi.rs's tinted-shadows twin): AO is a LIGHT query,
        // so glass passes its tint, folded to gray by the mean-of-components
        // rule (exact 1.0/0.0 on opaque scenes via the true divide).
        float3 tp = transmit_q(pt.o, d, tc, t_lim);
        float w_open = weight * ((tp.x + tp.y + tp.z) / 3.0);
        if (w_open > 0.0) {
            hemi_add(pt.pixel, 0, w_open);
        }
    } else {
        HitInfo h;
        if (trace_closest(pt.o, d, tc, FLT_MAX, h)) {
            float3 w3, o3, n3;
            PrimSurf ps_unused; // bounce rays never capture (secondary-ray rule)
            // Bounce cone: octant-scale spread (shade.rs::HEMI_CONE_SPREAD
            // — the CPU hemi leaf shades with the same value), and ISOTROPIC
            // (aniso false — hemi.rs pins Cone::aniso = 1.0): the cell
            // footprint is coarse by design, so resolving it anisotropically
            // would buy nothing.
            // fireflies false — bounce surfaces take no firefly light (the
            // CPU hemi tier's ff = None; the emissive precedent).
            // n_ao MIRRORS hemi.rs::BOUNCE_Q.ao_samples and is not optional
            // detail: at 0 the bounce surface's own sky ambient is unoccluded
            // (shade.hlsli leaves `ao` at 1.0), so an arcade interior returns
            // open-field radiance and the whole GI integral flattens to a
            // constant. Keep these two in lockstep.
            // Full emissive mask — inert: cam_lights=false means the
            // emissive block never runs on bounce laps (the gather IS the
            // emissive transport under GI).
            float3 l = shade_split(pt.o, d, h, rng, 1u, 1u, false, false,
                                   0.0, HEMI_CONE_SPREAD, false, false,
                                   uint2(0xffffffffu, 0xffffffffu), w3, o3, n3, ps_unused);
            hemi_add3(pt.pixel, l * weight);
        } else {
            // GATHER, not the full sky: a GI leaf ray landing in the sun disc
            // would double-count direct_d AND saturate this 2^18 fixed-point
            // accumulator outright (sky.rs's invariant). The star field is the
            // opposite case — nothing else delivers it to a bounce, and its
            // mean is ~1e-3 — so sky_gather carries it in.
            hemi_add3(pt.pixel, sky_gather(d) * weight);
        }
    }
}
