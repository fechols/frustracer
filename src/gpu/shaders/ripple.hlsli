// The procedural water ripple field — the ONE GPU copy.
//
// shade.rs::ripple_grad is the CPU source of truth; this is its term-for-term
// twin (only cos ulps differ). Constants are LITERALS matching shade.rs, so
// the pair is identical by construction — the clouds-wind precedent.
//
// It lives in its own file because it has THREE consumers, not two: the dxc
// units paste it ahead of shade.hlsli (whose `ripple_normal` calls it), and
// the fxc `cs_5_0` frame-generation guide kernel (gpu/ngxfg_guides.rs) pastes
// it to evaluate the field at two times per water pixel — hand-transcribing a
// field this size into a third copy is how twins drift. Written cs_5_0-clean
// for that reason: no wave ops, no SM6 intrinsics.
//
// The field is integrable BY CONSTRUCTION — it is the analytic gradient of a
// scalar height, which is what makes the perturbed normals describe a
// consistent virtual heightfield instead of shimmering with impossible
// normals. Never replace it with a divergence-free (curl) field: that is the
// opposite property and looks correct only until the surface is lit.
//
// Pure ALU, ZERO rng draws — which is what keeps every same-seed/replay/
// VisCtl-burn contract intact. Animated on CLOUD_TIME; every length is
// SCENE_DIAG-relative (the scale-relative rule).

// Normally supplied by trace_common.hlsli, which every dxc unit pastes first.
// The guide kernel pastes this file alone, hence the guard.
#ifndef TAU
#define TAU 6.28318530717959
#endif

static const float2 RIPPLE_DIR[3] = {
    float2(0.932, 0.362), float2(-0.588, 0.809), float2(0.259, -0.966)
};
static const float3 RIPPLE_LAMBDA_K = float3(5.2e-3, 2.9e-3, 1.6e-3);
static const float3 RIPPLE_W = float3(2.1, 3.4, 4.9);
static const float3 RIPPLE_A = float3(0.45, 0.32, 0.23);

// ∂h/∂x, ∂h/∂z of the virtual ripple height at world point `p`, time `t`.
float2 ripple_grad(float3 p, float t, float diag) {
    float2 g = float2(0.0, 0.0);
    float2 pxz = float2(p.x, p.z);
    [unroll]
    for (int i = 0; i < 3; i++) {
        float ph = TAU * (dot(RIPPLE_DIR[i], pxz) / (RIPPLE_LAMBDA_K[i] * diag)) - RIPPLE_W[i] * t;
        g += RIPPLE_DIR[i] * (RIPPLE_A[i] * cos(ph));
    }
    return g;
}
