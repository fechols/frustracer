// DXR pipeline (DispatchRays) implementations of the rt.hlsli trace
// primitives — TraceRay through the shader binding table instead of inline
// RayQuery, so shade.hlsli runs unmodified inside the closest-hit shader
// with every ray on the hardware pipeline. Requires trace_common.hlsli
// pasted first; the entry points (dxr.hlsl) follow shade.hlsli.
//
// SBT contract (mirrored by the byte layout in gpu/dxr.rs — keep in
// lockstep):
//   hit groups: 0 = HgShade (chs_shade), 1 = HgHit (chs_hit), 2 = null
//   miss:       0 = miss_radiance, 1 = miss_shadow, 2 = miss_hit

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
struct RayPayload {
    float3 color;
    float t;
    uint rng;
};

struct ShadowPayload {
    uint hit;
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
bool trace_closest(float3 o, float3 d, float tmin, float tmax, out HitInfo h) {
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = tmin; r.TMax = tmax;
    HitPayload p;
    p.t = -1.0; p.tri = 0u; p.u = 0.0; p.v = 0.0;
    TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xffu, 1u, 0u, 2u, r, p);
    h.t = max(p.t, 0.0); h.tri = p.tri; h.u = p.u; h.v = p.v;
    return p.t >= 0.0;
}

// rt.hlsli::occluded_q. Hit group 2 is a null SBT record and the closest-hit
// stage is skipped anyway — only miss_shadow can run, so an untouched
// payload IS the hit answer.
bool occluded_q(float3 o, float3 d, float tmin, float tmax) {
    if (tmax <= tmin) return false;
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = tmin; r.TMax = tmax;
    ShadowPayload p;
    p.hit = 1u;
    TraceRay(tlas,
             RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
                 RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0xffu, 2u, 0u, 1u, r, p);
    return p.hit != 0u;
}
