// The procedural water ripple field — the ONE GPU copy.
//
// shade.rs::ripple_height / ripple_grad are the CPU source of truth; this is
// their term-for-term twin (only sin/cos ulps differ). Constants are LITERALS
// matching shade.rs, so the pair is identical by construction — the
// clouds-wind precedent.
//
// It lives in its own file because it has THREE consumers, not two: the dxc
// units paste it ahead of shade.hlsli (whose `ripple_normal` calls it), and
// the fxc `cs_5_0` frame-generation guide kernel (gpu/ngxfg_guides.rs) pastes
// it to evaluate the field at two times per water pixel — hand-transcribing a
// field this size into a third copy is how twins drift. Written cs_5_0-clean
// and SELF-CONTAINED for that reason: no wave ops, no SM6 intrinsics, and it
// defines its own hash so it needs no trace_common.hlsli prelude.
//
// THE SHAPE: one domain-warped directional swell + three octaves of scrolling
// gradient-noise chop. Three fixed sinusoids (the previous field) beat against
// each other on a fixed lattice, so the interference repeats and a large
// expanse reads as TILED; noise breaks the repeat, and the swell survives
// because pure noise has no wave direction and open water does.
//
// The field is integrable BY CONSTRUCTION — it is the analytic gradient of
// `ripple_height`, which is what makes the perturbed normals a consistent
// virtual heightfield instead of shimmering with impossible normals. Never
// replace it with a divergence-free (curl) field: that is the opposite
// property and looks correct only until the surface is lit. A sum of scalars
// is still a scalar, so each layer may scroll at its own velocity (which is
// what stops the whole field sliding rigidly) at no cost to integrability.
//
// Pure ALU, ZERO rng draws — which is what keeps every same-seed/replay/
// VisCtl-burn contract intact. Animated on CLOUD_TIME; every length is
// SCENE_DIAG-relative (the scale-relative rule).

// Normally supplied by trace_common.hlsli, which every dxc unit pastes first.
// The guide kernel pastes this file alone, hence the guards.
#ifndef TAU
#define TAU 6.28318530717959
#endif

// Uniquely named so it cannot collide with trace_common.hlsli's pcg_mix in
// the units that paste both; identical mix, so the lattice is the same one.
uint rip_pcg_mix(uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// clouds::cell_hash — the u32-exact lattice hash both tracers already share.
float rip_cell_hash(int i, int j, uint oct) {
    uint h = rip_pcg_mix(uint(i) * 0x9E3779B9u ^ uint(j) * 0x85EBCA6Bu
                       ^ oct * 0xC2B2AE3Du);
    return float(h >> 8u) * (1.0 / 16777216.0);
}

// clouds::vnoise_vg — 2D value noise plus its ANALYTIC gradient.
// d/dt of the smoothstep t*t*(3-2t) is 6t(1-t).
void rip_vnoise_vg(float2 q, uint oct, out float v, out float2 g) {
    float fx = floor(q.x);
    float fy = floor(q.y);
    int i = int(fx);
    int j = int(fy);
    float tx = q.x - fx;
    float ty = q.y - fy;
    float ux = tx * tx * (3.0 - 2.0 * tx);
    float uy = ty * ty * (3.0 - 2.0 * ty);
    float dux = 6.0 * tx * (1.0 - tx);
    float duy = 6.0 * ty * (1.0 - ty);
    float h00 = rip_cell_hash(i, j, oct);
    float h10 = rip_cell_hash(i + 1, j, oct);
    float h01 = rip_cell_hash(i, j + 1, oct);
    float h11 = rip_cell_hash(i + 1, j + 1, oct);
    float a = h00 + (h10 - h00) * ux;
    float b = h01 + (h11 - h01) * ux;
    v = a + (b - a) * uy;
    g = float2(((h10 - h00) + ((h11 - h01) - (h10 - h00)) * uy) * dux, (b - a) * duy);
}

static const float2 RIPPLE_SWELL_DIR = float2(0.932, 0.362); // the cloud-wind direction
static const float RIPPLE_SWELL_LK = 5.2e-3;  // swell wavelength / diag
static const float RIPPLE_SWELL_W  = 2.1;     // rad/s
static const float RIPPLE_SWELL_A  = 0.42;    // slope weight
static const float RIPPLE_WARP_LK  = 2.4e-2;  // warp wavelength / diag (low frequency)
static const float RIPPLE_WARP_PHI = 2.6;     // radians a warp can bend a crest
// Chop octaves: wavelength/diag, slope weight, scroll velocity (wavelengths/s).
static const float3 RIPPLE_CHOP_LA[3] = {
    float3(3.3e-3, 0.30, 0.0), float3(1.7e-3, 0.22, 0.0), float3(8.5e-4, 0.15, 0.0)
};
static const float2 RIPPLE_CHOP_V[3] = {
    float2(-0.31, 0.42), float2(0.55, -0.24), float2(-0.18, -0.61)
};

// dh/dx, dh/dz of the virtual ripple height at world point `p`, time `t`.
float2 ripple_grad(float3 p, float t, float diag) {
    float2 pxz = float2(p.x, p.z);
    float l_sw = RIPPLE_SWELL_LK * diag;
    float l_wp = RIPPLE_WARP_LK * diag;
    float nw;
    float2 gw;
    rip_vnoise_vg(pxz / l_wp, 16u, nw, gw);
    float theta = TAU * (dot(RIPPLE_SWELL_DIR, pxz) / l_sw) - RIPPLE_SWELL_W * t
                + RIPPLE_WARP_PHI * nw;
    // The height's leading A*(l_sw/TAU) cancels d(theta)'s TAU/l_sw on the
    // swell term, leaving A*cos*d0 plus the warp's chain-rule contribution.
    float2 dtheta = RIPPLE_SWELL_DIR * (TAU / l_sw) + gw * (RIPPLE_WARP_PHI / l_wp);
    float2 g = dtheta * (RIPPLE_SWELL_A * (l_sw / TAU) * cos(theta));
    [unroll]
    for (int k = 0; k < 3; k++) {
        float l = RIPPLE_CHOP_LA[k].x * diag;
        float2 q = pxz / l - RIPPLE_CHOP_V[k] * t;
        float n;
        float2 gn;
        rip_vnoise_vg(q, 17u + uint(k), n, gn);
        g += gn * RIPPLE_CHOP_LA[k].y; // h_k = a*l*n ⇒ ∇h_k = a*∇n (l cancels)
    }
    return g;
}
