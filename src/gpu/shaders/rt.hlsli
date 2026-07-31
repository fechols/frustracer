// DXR inline-RayQuery helpers shared by every kernel that shoots actual rays
// (leaf/reference shading, hemi leaf rays, verify probes). Requires
// trace_common.hlsli pasted first.
//
// The scene is one BLAS per maximal BVH subtree of <= 64k tris, each an
// identity instance (BLAS_SPLIT — the DEFAULT), so a primitive index is an
// index into a CHUNK: every one below goes through trace_common.hlsli's
// tri_of(), the chunk remap. Under --no-blas-split it is one BLAS over
// scene.indices in order and tri_of compiles to the identity, which is the
// build that made CommittedPrimitiveIndex() == tri true. Never index
// tri_mat/indices/normals by a raw primitive index — go through tri_of().
// No cull flags (moller_trumbore is two-sided). Geometry is OPAQUE
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

// Relief uses trace_common.hlsli's height_tmin/height_tmax hardware interval:
// the full positive ray while the live height toggle is on. The hardware
// culls at the BASE-PLANE t, but the displaced t can differ by
// depth/abs(dot(d,n)), so no finite world-depth widening is conservative at
// grazing incidence. Every candidate loop below re-checks the marched `ct`
// against the ORIGINAL logical bounds (`ct > tmin && ct < tmax`), mirroring
// intersect_from and preventing the wider enumeration from accepting an
// out-of-segment hit.

// CAND_TMIN0 — the AMD candidate-loop TMin workaround (trace.rs injects it on
// an AMD adapter; see cand_defs there for the vendor gate and the measurement).
//
// IT LIVES OUTSIDE THE CUTOUT/RELIEF GUARD BELOW ON PURPOSE. Its callers do
// not share one condition: trace_closest/occluded_q are inside that guard, but
// transmit_q is under `#ifdef TRANS_SHADOW` — an INDEPENDENT per-scene arm — so
// defining it inside would leave a glass-bearing scene with no alpha-masked
// texture (an ordinary architectural OBJ: windows, opaque materials) failing to
// compile every ray-shooting kernel on `undeclared identifier 'cand_tmin'`.
// That shipped for exactly as long as it took to review, because every gated
// scene happens to carry BOTH arms or NEITHER (san-miguel and THE WORLD have
// foliage; procedural/stress/powerplant have no transmissive material at all),
// and `FR_ABL=noalpha` on san-miguel is what reproduces it. The shader-source
// test in trace.rs pins the ordering so it cannot come back.
//
// THE BUG, measured on a Radeon AI PRO R9700: in an inline RayQuery driven by
// a Proceed() candidate loop, a NONZERO r.TMin makes the reported hit distance
// come back as `t_true + TMin`. AMD's RT hardware re-origins the ray at TMin
// (documented in CLAUDE.md, and normally invisible); on this path the driver
// appears to add that offset back to a value that already carried it, so the
// error is exactly +TMin and shows up in CandidateTriangleRayT/CommittedRayT
// alike. NVIDIA, same source, same scene, is bit-exact.
//
// It is not cosmetic: the wavefront's leaf primaries pass the tile's inherited
// t_start as TMin, so on any alpha-cutout or relief scene EVERY hit pixel got
// a t that was ~6.5 units long on san-miguel-low-poly — which moves the shading
// point (p = o + d*t), the G-buffer view-z, and the motion vectors. It read as
// `tmin-overshoot 341393 | max rel t err 8.70e-1 | mv_selftest median 8.428e-1`
// in --check-gpu --prefer-amd, with `claim-violation 0` throughout: the
// empty-space proof was always sound, only the reported distance was wrong.
//
// THE FIX is the one this file already implements for relief: hand the hardware
// the full positive ray and let the loop's own `ct > tmin && ct < tmax` re-check
// enforce the logical interval. With TMin = 0 there is no re-origining, so
// there is no offset to get wrong — which makes this robust to whatever the
// driver does rather than a compensation that would silently invert if AMD
// fixes it (subtracting TMin back off would then read t TOO SMALL, the
// false-near-hit direction). TMax is left alone: it is not implicated, and
// keeping it bounds traversal.
//
// The cost is the near-prune: candidates in [0, tmin) now surface and are
// rejected by the loop instead of being culled by the hardware. That is
// affordable precisely HERE, because the inherited bound is already measured
// "free and worth nothing" on AMD (it re-origins rather than prunes) — the
// same hardware property that causes the bug pays for the workaround.
//
// Unarmed it is `height_tmin` verbatim, so the FORCE_OPAQUE arms below (which
// take the raw tmin, having no candidate loop to re-check anything) stay the
// only paths that bypass the relief interval — as before.
float cand_tmin(float logical_tmin) {
#ifdef CAND_TMIN0
    return 0.0;
#else
    return height_tmin(logical_tmin);
#endif
}

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

bool trace_closest(float3 o, float3 d, float tmin, float tmax, out HitInfo h) {
    h = (HitInfo)0;
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = cand_tmin(tmin); r.TMax = height_tmax(tmax);
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xffu, r);
    while (q.Proceed()) {
        // Only non-opaque candidates surface here (BLAS flag NONE in
        // alpha/height scenes). Committing shrinks TMax; rejecting leaves it
        // unshrunk — exactly moller_trumbore returning None. A marched hit
        // COMMITS AT THE PLANE t (the only t a RayQuery can commit):
        // candidates whose plane-t's interleave within one relief prism's
        // displaced ray interval can mis-sort — spatially bounded, although
        // its ray-t width grows at grazing — and the winner is re-marched
        // below for its true (t', u', v'). This is the documented plane-order
        // known-accept; full hardware interval enumeration does not repair it.
        float ct = q.CandidateTriangleRayT();
        float cu = q.CandidateTriangleBarycentrics().x;
        float cv = q.CandidateTriangleBarycentrics().y;
        uint rej = candidate_reject(
            tri_of(q.CandidateInstanceID(), q.CandidatePrimitiveIndex()), o, d, ct, cu, cv);
        if (rej == 0u && ct > tmin && ct < tmax)
            q.CommitNonOpaqueTriangleHit();
        else if (rej == 1u)
            count_alpha_rej();
        else
            count_height_rej();
    }
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        h.t = q.CommittedRayT();
        h.tri = tri_of(q.CommittedInstanceID(), q.CommittedPrimitiveIndex());
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
    r.Origin = o; r.Direction = d; r.TMin = cand_tmin(tmin); r.TMax = height_tmax(tmax);
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xffu, r);
    while (q.Proceed()) {
        float ct = q.CandidateTriangleRayT();
        float cu = q.CandidateTriangleBarycentrics().x;
        float cv = q.CandidateTriangleBarycentrics().y;
        uint rej = candidate_reject(
            tri_of(q.CandidateInstanceID(), q.CandidatePrimitiveIndex()), o, d, ct, cu, cv);
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
        h.tri = tri_of(q.CommittedInstanceID(), q.CommittedPrimitiveIndex());
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

// Light-transport occlusion (bvh.rs::Bvh::transmittance — tinted shadows):
// the RGB throughput of segment (tmin, tmax). ONE when clear, ZERO on any
// opaque hit, the product of mat_shadow tints per transmissive interface
// crossed. Every LIGHT ray (sun shadow, translucency back, firefly, AO,
// hemi-AO leaf) calls this; occluded_q keeps binary any-geometry semantics
// for the geometric verify oracles (check_empty_cell). Without TRANS_SHADOW
// this is occluded_q verbatim — the structural bit-identity arm.
// SHADOW_TP_MIN lives in trace_common.hlsli (shared with rt_dxr.hlsli).

#ifdef TRANS_SHADOW

void count_trans_pass() {
#ifdef HAVE_COUNTERS
    uint _d;
    InterlockedAdd(counters[CTR_TRANS_PASS], 1u, _d);
#endif
}

float3 transmit_q(float3 o, float3 d, float tmin, float tmax) {
    if (tmax <= tmin) return float3(1.0, 1.0, 1.0);
    RayDesc r;
    r.Origin = o; r.Direction = d; r.TMin = cand_tmin(tmin); r.TMax = height_tmax(tmax);
    // ACCEPT_FIRST_HIT still applies — but only to COMMITTED hits, and only
    // opaque candidates commit: a transmissive candidate multiplies the
    // running tint and is NOT committed, so traversal keeps enumerating.
    // Candidate order is hardware-arbitrary; the product commutes, and any
    // opaque commit zeroes the result regardless of what multiplied in. The
    // BLAS builds with NO_DUPLICATE_ANYHIT_INVOCATION when TRANS_SHADOW is
    // armed (trace.rs::geometry_desc) — without it a candidate may surface
    // twice and the multiply, unlike the cutout reject, is not idempotent.
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xffu, r);
    float3 tp = float3(1.0, 1.0, 1.0);
    while (q.Proceed()) {
        float ct = q.CandidateTriangleRayT();
        float cu = q.CandidateTriangleBarycentrics().x;
        float cv = q.CandidateTriangleBarycentrics().y;
        uint tri = tri_of(q.CandidateInstanceID(), q.CandidatePrimitiveIndex());
        uint rej = candidate_reject(tri, o, d, ct, cu, cv);
        if (rej == 0u && ct > tmin && ct < tmax) {
            float4 ms = mat_shadow[uv_tri_mat[tri]];
            if (ms.a > 0.0) {
                tp *= ms.rgb;
                count_trans_pass();
                if (max(tp.x, max(tp.y, tp.z)) < SHADOW_TP_MIN)
                    return float3(0.0, 0.0, 0.0);
            } else {
                q.CommitNonOpaqueTriangleHit();
            }
        }
#if defined(ALPHA_CUTOUT) || defined(HEIGHTFIELD)
        else if (rej == 1u)
            count_alpha_rej();
        else if (rej == 2u)
            count_height_rej();
#endif
    }
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT
        ? float3(0.0, 0.0, 0.0)
        : tp;
}

#else

float3 transmit_q(float3 o, float3 d, float tmin, float tmax) {
    return occluded_q(o, d, tmin, tmax)
        ? float3(0.0, 0.0, 0.0)
        : float3(1.0, 1.0, 1.0);
}

#endif // TRANS_SHADOW
