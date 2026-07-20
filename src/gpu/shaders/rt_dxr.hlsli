// DXR pipeline (DispatchRays) implementations of the rt.hlsli trace
// primitives — TraceRay through the shader binding table instead of inline
// RayQuery, so shade.hlsli runs unmodified inside the closest-hit shader
// with every ray on the hardware pipeline. Requires trace_common.hlsli
// pasted first; the entry points (dxr.hlsl) follow shade.hlsli.
//
// SBT contract (mirrored by the byte layout in gpu/dxr.rs — keep in
// lockstep):
//   hit groups: 0 = HgShade (chs_shade), 1 = HgHit (chs_hit),
//               2 = null (opaque scenes) / HgOcclude (any-hit only,
//                   ALPHA_CUTOUT / HEIGHTFIELD / TRANS_SHADOW scenes)
//   miss:       0 = miss_radiance, 1 = miss_shadow, 2 = miss_hit
//
// ALPHA_CUTOUT / HEIGHTFIELD (per-scene, trace.rs::alpha_defs /
// height_defs): the BLAS drops OPAQUE and every TraceRay drops FORCE_OPAQUE
// so the ah_* any-hit shaders (dxr.hlsl) run the shared candidate_reject
// test (relief march + cutout at the displaced uv); opaque scenes compile
// the FORCE_OPAQUE flags verbatim and keep the null occlusion record.
// TRANS_SHADOW (trace.rs::trans_defs, tinted shadows) drops FORCE_OPAQUE on
// the OCCLUSION rays only (SHADOW_RF below) — ah_shadow accumulates glass
// tints into the payload while closest rays keep OPAQUE_RF (glass still
// HITS for visibility).

#if defined(ALPHA_CUTOUT) || defined(HEIGHTFIELD)
#define OPAQUE_RF RAY_FLAG_NONE
#else
#define OPAQUE_RF RAY_FLAG_FORCE_OPAQUE
#endif

// Occlusion rays additionally drop FORCE_OPAQUE under TRANS_SHADOW (tinted
// shadows): ah_shadow must run to accumulate the glass tint into the
// payload. CLOSEST rays keep OPAQUE_RF — on a transmission-only scene they
// still fly FORCE_OPAQUE (glass HITS for visibility; the ray flag overrides
// the BLAS's non-opaque geometry flag), so the extra compiled any-hits stay
// inert there.
#if defined(ALPHA_CUTOUT) || defined(HEIGHTFIELD) || defined(TRANS_SHADOW)
#define SHADOW_RF RAY_FLAG_NONE
#else
#define SHADOW_RF RAY_FLAG_FORCE_OPAQUE
#endif

RaytracingAccelerationStructure tlas : register(t7);

struct HitInfo {
    float t;
    uint tri;
    float u, v; // DXR barycentrics == moller-trumbore: p = (1-u-v)p0 + u·p1 + v·p2
};

// Radiance ray (raygen -> chs_shade / miss_radiance). rng rides IN so the
// closest-hit continues the pixel's stream exactly where raygen's jitter
// draws left it (the reference kernel's one-stream contract); color and the
// primary t (INF = sky, the tbuf convention) ride out.
//
// --spp: raygen fires one of these per SAMPLE, so the sample's position (sp)
// and its identity ride in too: prim = `(s << 1) | probe_bit` — bit 0 marks
// the probe sample (the one that owns tbuf/info/the G-buffer pack), the high
// bits carry the sample index for the miss shader's per-sample cloud march
// phase (cloud_dither_k). chs_shade can no longer recompute the position
// from frame_jitter alone, and it no longer fires exactly once per pixel.
struct RayPayload {
    float3 color;
    float t;
    uint rng;
    float2 sp;
    uint prim;
};

struct ShadowPayload {
    uint hit;
    // The segment's LOGICAL far bound. Relief widens the hardware TMax below
    // (a marched underside hit inside tmax can belong to a candidate whose
    // PLANE t lies beyond it — the hardware culls at the plane t), so
    // ah_shadow re-checks the MARCHED t against this — the mirror of the
    // RayQuery loops' explicit `ct < tmax`. Opaque scenes never run an
    // any-hit and ignore the field.
    float tmax;
    // Tinted shadows (TRANS_SHADOW): the running product of glass tints
    // ah_shadow accumulates while IgnoreHit()-ing transmissive candidates
    // (payload writes persist through IgnoreHit — the standard tinted-shadow
    // pattern). Any-hit order is hardware-arbitrary; the product commutes,
    // and a committed opaque hit zeroes the result regardless. The BLAS
    // carries NO_DUPLICATE_ANYHIT_INVOCATION when TRANS_SHADOW is armed —
    // the multiply, unlike the cutout reject, is not idempotent.
    float3 tint;
};

// trace_closest's wire format; t < 0 = miss.
struct HitPayload {
    float t;
    uint tri;
    float u;
    float v;
};

// rt.hlsli::trace_closest, DispatchRays flavor — the reflection ray inside
// shade_split. Hit group 1 records bare hit info: the lap loop shades the
// reflected surface itself (the CPU's depth-1 recursion), so routing it to
// chs_shade would double-shade.
//
// Relief TMax contract: every trace_closest call in this pipeline passes
// tmax = FLT_MAX (shade.hlsli), so the relief TMax widening lives on the
// occlusion path only. A future FINITE-tmax closest-hit ray would need the
// ShadowPayload treatment (widen + payload-carried logical bound), because
// the hardware culls candidates at the PLANE t while the marched t can land
// one relief depth earlier.
bool trace_closest(float3 o, float3 d, float tmin, float tmax, out HitInfo h) {
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = tmin; r.TMax = tmax;
    HitPayload p;
    p.t = -1.0; p.tri = 0u; p.u = 0.0; p.v = 0.0;
    TraceRay(tlas, OPAQUE_RF, 0xffu, 1u, 0u, 2u, r, p);
    h.t = max(p.t, 0.0); h.tri = p.tri; h.u = p.u; h.v = p.v;
    return p.t >= 0.0;
}

// rt.hlsli::occluded_q. The closest-hit stage is skipped — only miss_shadow
// (and, in ALPHA_CUTOUT/HEIGHTFIELD scenes, HgOcclude's any-hit rejecting
// masked/escaped candidates) can run, so an untouched payload IS the hit
// answer. Relief widens the hardware TMax by one relief depth (rt.hlsli's
// height_tmax rationale — the AO segment is this pipeline's finite-tmax
// case); ah_shadow re-checks the marched t against the LOGICAL bound riding
// the payload. TMin stays unwidened: every occluded_q call here passes
// tmin = 0 (shade.hlsli), so there is nothing to widen.
bool occluded_q(float3 o, float3 d, float tmin, float tmax) {
    if (tmax <= tmin) return false;
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = tmin;
    r.TMax = tmax;
#ifdef HEIGHTFIELD
    if (flags & FLAG_HEIGHT) r.TMax = tmax + height_max;
#endif
    ShadowPayload p;
    p.hit = 1u;
    p.tmax = tmax;
    p.tint = float3(1.0, 1.0, 1.0);
    // NOTE: under TRANS_SHADOW ah_shadow passes transmissive candidates, so
    // this binary query sees THROUGH glass on this pipeline. Every DXR light
    // ray goes through transmit_q below and no geometric oracle runs here —
    // if one ever does, give it its own hit group.
    TraceRay(tlas,
             SHADOW_RF | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
                 RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0xffu, 2u, 0u, 1u, r, p);
    return p.hit != 0u;
}

// rt.hlsli::transmit_q, DispatchRays flavor (tinted shadows): the RGB
// throughput of segment (tmin, tmax) — ONE when clear, ZERO on any opaque
// hit, the product of ah_shadow's accumulated glass tints otherwise. Without
// TRANS_SHADOW this degenerates to occluded_q's binary answer bit-exactly
// (the tint stays the initialized ONE and only hit/miss decides).
float3 transmit_q(float3 o, float3 d, float tmin, float tmax) {
    if (tmax <= tmin) return float3(1.0, 1.0, 1.0);
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = tmin;
    r.TMax = tmax;
#ifdef HEIGHTFIELD
    if (flags & FLAG_HEIGHT) r.TMax = tmax + height_max;
#endif
    ShadowPayload p;
    p.hit = 1u;
    p.tmax = tmax;
    p.tint = float3(1.0, 1.0, 1.0);
    TraceRay(tlas,
             SHADOW_RF | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
                 RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0xffu, 2u, 0u, 1u, r, p);
    // The SHADOW_TP_MIN floor lands at the return here — an any-hit cannot
    // end the search mid-traversal like the RayQuery loop does; tints are
    // ≤ 1 per component so the two placements agree (trace_common.hlsli).
    if (p.hit != 0u || max(p.tint.x, max(p.tint.y, p.tint.z)) < SHADOW_TP_MIN)
        return float3(0.0, 0.0, 0.0);
    return p.tint;
}
