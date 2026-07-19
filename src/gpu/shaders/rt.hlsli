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

#if defined(ALPHA_CUTOUT) || defined(HEIGHTFIELD)

// Anti-vacuity stat: cutout/relief rejections, counted only by compile units
// that bind `counters` (ctr.hlsli pastes before this file in the wavefront
// kernels; the reference kernel / DXR library never touch the slot).
void count_alpha_rej() {
#ifdef HAVE_COUNTERS
    uint _d;
    InterlockedAdd(counters[CTR_ALPHA_REJ], 1u, _d);
#endif
}

void count_height_rej() {
#ifdef HAVE_COUNTERS
    uint _d;
    InterlockedAdd(counters[CTR_HEIGHT_REJ], 1u, _d);
#endif
}

// Relief widens the hardware ray interval on BOTH ends — the hardware culls
// candidates at the PLANE t, and a marched hit can sit up to one relief
// depth on the far side of its plane t in either direction: a below/interior
// hit lands EARLIER than its plane t (so an inherited t_start ∈ (t_plane,
// t'] would drop a candidate the CPU accepts — the TMin side), and a hit
// with marched t' < tmax can belong to a candidate whose plane t lies BEYOND
// tmax (an underside hit just inside an AO/shadow segment's far end — the
// TMax side). The marched t is re-checked against the ORIGINAL bounds
// explicitly in the loops (mirroring intersect_from's `tt > tmin &&
// tt < tmax`), so the widening only ever surfaces candidates, never accepts
// out-of-range hits.
float height_tmin(float tmin) {
#ifdef HEIGHTFIELD
    if (flags & FLAG_HEIGHT) return max(0.0, tmin - height_max);
#endif
    return tmin;
}

float height_tmax(float tmax) {
#ifdef HEIGHTFIELD
    // +height_max saturates INF/FLT_MAX harmlessly (the unbounded rays).
    if (flags & FLAG_HEIGHT) return tmax + height_max;
#endif
    return tmax;
}

bool trace_closest(float3 o, float3 d, float tmin, float tmax, out HitInfo h) {
    h = (HitInfo)0;
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = height_tmin(tmin); r.TMax = height_tmax(tmax);
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xffu, r);
    while (q.Proceed()) {
        // Only non-opaque candidates surface here (BLAS flag NONE in
        // alpha/height scenes). Committing shrinks TMax; rejecting leaves it
        // unshrunk — exactly moller_trumbore returning None. A marched hit
        // COMMITS AT THE PLANE t (the only t a RayQuery can commit):
        // candidates whose plane-t's interleave within one relief depth can
        // mis-sort — bounded, the documented known-accept — and the winner
        // is re-marched below for its true (t', u', v').
        float ct = q.CandidateTriangleRayT();
        float cu = q.CandidateTriangleBarycentrics().x;
        float cv = q.CandidateTriangleBarycentrics().y;
        uint rej = candidate_reject(q.CandidatePrimitiveIndex(), o, d, ct, cu, cv);
        if (rej == 0u && ct > tmin && ct < tmax)
            q.CommitNonOpaqueTriangleHit();
        else if (rej == 1u)
            count_alpha_rej();
        else
            count_height_rej();
    }
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        h.t = q.CommittedRayT();
        h.tri = q.CommittedPrimitiveIndex();
        float2 b = q.CommittedTriangleBarycentrics();
        h.u = b.x; h.v = b.y;
        // Re-march the committed winner for the displaced (t', u', v') the
        // shading consumes (deterministic — same inputs as the in-loop run).
        candidate_reject(h.tri, o, d, h.t, h.u, h.v);
        return true;
    }
    return false;
}

bool occluded_q(float3 o, float3 d, float tmin, float tmax) {
    if (tmax <= tmin) return false;
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = height_tmin(tmin); r.TMax = height_tmax(tmax);
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xffu, r);
    while (q.Proceed()) {
        float ct = q.CandidateTriangleRayT();
        float cu = q.CandidateTriangleBarycentrics().x;
        float cv = q.CandidateTriangleBarycentrics().y;
        uint rej = candidate_reject(q.CandidatePrimitiveIndex(), o, d, ct, cu, cv);
        if (rej == 0u && ct > tmin && ct < tmax)
            q.CommitNonOpaqueTriangleHit();
        else if (rej == 1u)
            count_alpha_rej();
        else
            count_height_rej();
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
