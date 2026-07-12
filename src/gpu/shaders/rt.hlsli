// DXR inline-RayQuery helpers shared by every kernel that shoots actual rays
// (leaf/reference shading, hemi leaf rays, verify probes). Requires
// trace_common.hlsli pasted first.
//
// One BLAS over scene.indices in order + one identity-instance TLAS, so
// CommittedPrimitiveIndex() == tri and tri_mat/indices/normals index
// directly. No cull flags (moller_trumbore is two-sided). Geometry is OPAQUE
// unless the scene has alpha-masked textures: then trace.rs builds the BLAS
// with FLAG_NONE and prepends #define ALPHA_CUTOUT 1, and the wrappers below
// run a candidate loop mirroring bvh.rs::moller_trumbore's rejection — the
// cutout only REMOVES hits, so every frustum/tmin/temporal claim carries
// over (the bvh.rs soundness argument). Without the define these compile
// byte-identical to the always-opaque originals.

RaytracingAccelerationStructure tlas : register(t7);

struct HitInfo {
    float t;
    uint tri;
    float u, v; // moller-trumbore == DXR convention: p = (1-u-v)p0 + u·p1 + v·p2
};

#ifdef ALPHA_CUTOUT

// Anti-vacuity stat: cutout rejections, counted only by compile units that
// bind `counters` (ctr.hlsli pastes before this file in the wavefront
// kernels; the reference kernel / DXR library never touch the slot).
void count_alpha_rej() {
#ifdef HAVE_COUNTERS
    uint _d;
    InterlockedAdd(counters[CTR_ALPHA_REJ], 1u, _d);
#endif
}

bool trace_closest(float3 o, float3 d, float tmin, float tmax, out HitInfo h) {
    h = (HitInfo)0;
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = tmin; r.TMax = tmax;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xffu, r);
    while (q.Proceed()) {
        // Only non-opaque candidates surface here (BLAS flag NONE in alpha
        // scenes). Committing shrinks TMax; rejecting leaves it unshrunk —
        // exactly moller_trumbore returning None.
        if (!alpha_cutout(q.CandidatePrimitiveIndex(),
                          q.CandidateTriangleBarycentrics().x,
                          q.CandidateTriangleBarycentrics().y))
            q.CommitNonOpaqueTriangleHit();
        else
            count_alpha_rej();
    }
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
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xffu, r);
    while (q.Proceed()) {
        if (!alpha_cutout(q.CandidatePrimitiveIndex(),
                          q.CandidateTriangleBarycentrics().x,
                          q.CandidateTriangleBarycentrics().y))
            q.CommitNonOpaqueTriangleHit();
        else
            count_alpha_rej();
    }
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

#else

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

#endif // ALPHA_CUTOUT
