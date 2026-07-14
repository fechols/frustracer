// The single accum splat site: partial + ambW * ambient(H), store-or-add per
// the frame/accumulate flags — the GPU analog of "each pixel is written by
// exactly one task per frame" with the implicit frame-0 clear. Requires
// trace_common.hlsli + ctr.hlsli + queues.hlsli pasted first.
//
// hemi.rs mapping: AO ambient = sky_irradiance(n) * (mass / pi) (no clamps —
// the estimator is unbiased, truncation would bias it); GI ambient =
// max(rgb / pi, 0) per component.
//
// The AO arm's sky term is NOT applied here: compose has no normal, and the
// sky's irradiance is directional. shade.hlsli folds it into `ambw` at the
// leaf site (where n_s is in scope) whenever fb_mode == 1, so this pass stays
// a pure weight x mass multiply. The GI arm must NOT be pre-multiplied — it
// integrates the sky itself.

[numthreads(256, 1, 1)]
void cs_compose(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint pi = flat_group(gid) * 256u + gtid.x;
    if (pi >= rw * rh) return;
    uint i3 = pi * 3u;
    float3 c = float3(partial[i3], partial[i3 + 1u], partial[i3 + 2u]);
    if (fb_mode != 0u) {
        float3 aw = float3(ambw[i3], ambw[i3 + 1u], ambw[i3 + 2u]);
        float3 ambient;
        if (fb_mode == 1u) {
            // aw already carries sh_irradiance(n_s) — see the header note.
            ambient = float(hbuf[pi * 4u]) / HEMI_FIXED / PI;
        } else {
            ambient = max(
                float3(float(hbuf[pi * 4u]), float(hbuf[pi * 4u + 1u]), float(hbuf[pi * 4u + 2u]))
                    / HEMI_FIXED / PI,
                0.0);
        }
        c += aw * ambient;
    }
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
