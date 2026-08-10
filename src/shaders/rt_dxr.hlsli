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

// FR_DXR_INLINE (gpu/dxr.rs — the W2 Intel-campaign lever): rt.hlsli is
// pasted BEFORE this file, so tlas + HitInfo + the three trace primitives
// already exist as inline RayQuery; this file then contributes only the
// payload structs + ray flags the DispatchRays stages still need (raygen's
// primary TraceRay in mode 1; in mode 2 the chs_*/miss_*/ah_* entry points
// are dead-but-exported and no ray can reach them).
// DXR_SBT_RECURSE (--dxr-sbt 3, gpu/dxr.rs) is the HYBRID paste: rt.hlsli
// rides along for the inline-RayQuery occlusion primitives (shadow/AO rays
// never re-enter the pipeline — that is what caps MaxTraceRecursionDepth at
// 5), so tlas/HitInfo and the three TraceRay-flavor primitives below compile
// out exactly as under the inline modes, and this file contributes the
// payload structs + `trace_shade` — the recursive shading TraceRay the
// class-sorted continuations ride.
#if !defined(DXR_INLINE_SEC) && !defined(DXR_SBT_RECURSE)
RaytracingAccelerationStructure tlas : register(t7);

struct HitInfo {
    float t;
    uint tri;
    float u, v; // DXR barycentrics == moller-trumbore: p = (1-u-v)p0 + u·p1 + v·p2
#ifdef SWAY_MV
    // Signature parity with rt.hlsli. This TraceRay flavor serves only the
    // reflection continuations inside chs_shade — never the G-buffer write,
    // whose chs_shade call passes InstanceID() directly — so 0 is fine.
    uint inst;
#endif
};
#endif

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
    // The segment's LOGICAL bounds. With relief live, the hardware interval is
    // the full positive ray: a marched hit inside either bound can belong to
    // a base-plane candidate arbitrarily far outside it at grazing incidence.
    // ah_shadow re-checks the MARCHED t against these — the mirror of the
    // RayQuery loops' explicit `ct > tmin && ct < tmax`. Opaque scenes never
    // run an any-hit and ignore the fields.
    float tmin;
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
    // Logical interval, consumed by ah_hit after relief moves the candidate t.
    float tmin;
    float tmax;
#if defined(DXR_INLINE_SEC) && DXR_INLINE_SEC == 3
    // Mode 3 (thin CHS): the committed InstanceID, carried into the hit
    // record so the deferred compute kernel can feed the sway-MV G-buffer
    // lane (reference.hlsl's `hit.inst` form — the payload hop is the only
    // way out of the pipeline, raygen has no InstanceID()). 24 -> 28 B, under
    // the RTPSO's 32 B MaxPayloadSizeInBytes; guarded so modes 0-2
    // preprocess to today's bytes.
    uint inst;
#endif
};

// Recursive shading TraceRay (--dxr-sbt 3): the continuation fires at
// RayContribution 0, so the instance's InstanceContributionToHitGroupIndex
// (= class * 3, baked at upload) lands it in the HIT SURFACE'S OWN class
// shading record — the per-class recursive dispatch IS the SBT arithmetic,
// zero multipliers. The miss is miss_rec (index 3): t = INF and NOTHING
// else — the PARENT owns its miss arm, because a reflection miss needs the
// PARENT lobe's MIS weight and a glass miss the fixed-phase sky, neither of
// which a miss shader can know. depth and the cone origin ride the sp lanes
// (bit-punned — no payload growth past the 32 B config), prim carries only
// the recursion tag (bit 31; probe bit 0, so the child CHS skips every
// side channel through the existing guard). rng rides in AND back out —
// the payload round-trip is what keeps the stream in the CPU DFS's exact
// draw order.
#ifdef DXR_SBT_RECURSE
float3 trace_shade(float3 o, float3 d, inout uint rng, uint depth, float cone_o,
                   out float t_out) {
    RayDesc r;
    r.Origin = o;
    r.Direction = d;
    r.TMin = height_tmin(0.0);
    r.TMax = height_tmax(FLT_MAX);
    RayPayload p;
    p.color = float3(0.0, 0.0, 0.0);
    p.t = INF;
    p.rng = rng;
    p.sp = float2(asfloat(depth), cone_o);
    p.prim = 0x80000000u;
    TraceRay(tlas, OPAQUE_RF, 0xffu, 0u, 0u, 3u, r, p);
    rng = p.rng;
    t_out = p.t;
    return p.color;
}
#endif // DXR_SBT_RECURSE

// The three TraceRay-flavor primitives. Compiled out under FR_DXR_INLINE —
// rt.hlsli's inline RayQuery bodies (pasted above) serve shade.hlsli instead,
// so every secondary runs inline inside chs_shade (mode 1) or raygen (mode 2)
// and the SBT's HgHit/miss/occlusion machinery goes dead-but-exported — and
// under DXR_SBT_RECURSE, whose occlusion is inline (rt.hlsli again) and whose
// closest continuations are trace_shade above, never these.
#if !defined(DXR_INLINE_SEC) && !defined(DXR_SBT_RECURSE)

// rt.hlsli::trace_closest, DispatchRays flavor — the reflection ray inside
// shade_split. Hit group 1 records bare hit info: the lap loop shades the
// reflected surface itself (the CPU's depth-1 recursion), so routing it to
// chs_shade would double-shade.
//
// Relief interval contract: hardware enumeration uses the complete positive
// base-triangle ray (trace_common.hlsli::height_tmin/height_tmax), while the
// payload carries the caller's logical bounds for ah_hit to apply to the
// MARCHED t. Current reflection callers use tmax=FLT_MAX, but this also keeps
// future finite-tmax and nonzero-tmin calls sound at grazing incidence.
bool trace_closest(float3 o, float3 d, float tmin, float tmax, out HitInfo h) {
    RayDesc r;
    r.Origin = o; r.Direction = d;
    r.TMin = height_tmin(tmin); r.TMax = height_tmax(tmax);
    HitPayload p;
    p.t = -1.0; p.tri = 0u; p.u = 0.0; p.v = 0.0;
    p.tmin = tmin; p.tmax = tmax;
    TraceRay(tlas, OPAQUE_RF, 0xffu, 1u, 0u, 2u, r, p);
    h.t = max(p.t, 0.0); h.tri = p.tri; h.u = p.u; h.v = p.v;
#ifdef SWAY_MV
    h.inst = 0u; // dead — see the field's comment
#endif
    return p.t >= 0.0;
}

// rt.hlsli::occluded_q. The closest-hit stage is skipped — only miss_shadow
// (and, in ALPHA_CUTOUT/HEIGHTFIELD scenes, HgOcclude's any-hit rejecting
// masked/escaped candidates) can run, so an untouched payload IS the hit
// answer. Relief enumerates the full positive hardware ray (a finite
// world-depth widening is not conservative at grazing); ah_shadow re-checks
// the marched t against BOTH logical bounds riding the payload. Today's
// shade.hlsli occlusion calls pass tmin=0, while carrying it explicitly keeps
// this primitive correct for a future nonzero-tmin caller.
bool occluded_q(float3 o, float3 d, float tmin, float tmax) {
    if (tmax <= tmin) return false;
    RayDesc r;
    r.Origin = o; r.Direction = d;
    r.TMin = height_tmin(tmin); r.TMax = height_tmax(tmax);
    ShadowPayload p;
    p.hit = 1u;
    p.tmin = tmin;
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
    r.Origin = o; r.Direction = d;
    r.TMin = height_tmin(tmin); r.TMax = height_tmax(tmax);
    ShadowPayload p;
    p.hit = 1u;
    p.tmin = tmin;
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

// _t twin (NRD's sig.w hit-distance guide): the all-TraceRay arm does NOT
// capture — carrying the occluder t out would grow ShadowPayload (an ABI
// every ah_shadow/miss_shadow site shares), for the --dxr-inline 0
// measurement escape only; the DEFAULT inline-1 pipeline captures via
// rt.hlsli's bodies. first_t = tmax (= "miss"/no data); ReBLUR's
// hit-distance-reconstruction mode covers it. Documented known-accept.
float3 transmit_q_t(float3 o, float3 d, float tmin, float tmax, out float first_t) {
    first_t = tmax;
    return transmit_q(o, d, tmin, tmax);
}

#endif // !DXR_INLINE_SEC && !DXR_SBT_RECURSE
