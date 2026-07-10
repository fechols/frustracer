// DXR inline-RayQuery helpers shared by every kernel that shoots actual rays
// (leaf/reference shading, hemi leaf rays, verify probes). Requires
// trace_common.hlsli pasted first.
//
// One BLAS over scene.indices in order + one identity-instance TLAS, so
// CommittedPrimitiveIndex() == tri and tri_mat/indices/normals index
// directly. Geometry is OPAQUE; no cull flags (moller_trumbore is two-sided).

RaytracingAccelerationStructure tlas : register(t7);

struct HitInfo {
    float t;
    uint tri;
    float u, v; // moller-trumbore == DXR convention: p = (1-u-v)p0 + u·p1 + v·p2
};

bool trace_closest(float3 o, float3 d, float tmin, float tmax, out HitInfo h) {
    h = (HitInfo)0;
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = tmin; r.TMax = tmax;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xffu, r);
    q.Proceed();
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        h.t = q.CommittedRayT();
        h.tri = q.CommittedPrimitiveIndex();
        float2 b = q.CommittedTriangleBarycentrics();
        h.u = b.x; h.v = b.y;
        return true;
    }
    return false;
}

bool occluded_q(float3 o, float3 d, float tmin, float tmax) {
    if (tmax <= tmin) return false;
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = tmin; r.TMax = tmax;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xffu, r);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}
