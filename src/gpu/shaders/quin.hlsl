// Registered multi-engine consensus — the fuse (--quinlight).
//
// A port of quinlight-player's consensus_registered.comp (the UNSEEDED,
// single-level path: `use_seed == 0`, HW bilinear, engines as SRVs). For each
// output pixel it (1) reads all N engine images, (2) registers every
// non-anchor engine to the anchor with a small-window Lucas-Kanade solve and
// bilinearly warps it into the anchor's frame, then (3) reduces the
// {anchor} u {warped non-anchors} stack with a per-channel winsorized mean.
//
// The engines are the session's upscalers run over the SAME traced frame —
// DLSS-RR / FSR4-RR / XeSS / FSR 3.1, all at window res, all RGBA16F. This is
// a SPATIAL, STATELESS pass: no history, no motion vectors, no depth, no
// jitter. Everything temporal already happened inside the engines.
//
// Monolithic on purpose: it writes NO intermediate flow or warped images, so
// memory traffic stays at "N engine reads + 1 write" — the extra cost is
// purely the per-pixel windowed solves (compute, which is what we want to
// spend).
//
// What was deliberately NOT ported: the coarse-to-fine pyramid seed (3 more
// shaders + its host glue). It exists upstream to capture LARGE inter-engine
// motion at 8K; inter-engine disagreement on one frame is sub-pixel to a few
// pixels, which MAX_DISP covers. If engines ever visibly disagree by more,
// that is the signal to port it.

#define MAX_ENGINES 4
#define HALF_WINDOW 2                                  // 5x5 solve window
#define MAX_TAPS ((2 * HALF_WINDOW + 1) * (2 * HALF_WINDOW + 1))
#define HALO (HALF_WINDOW + 1)                         // window + the +/-1 gradient ring
#define TILE (16 + 2 * HALO)                           // 22x22 float3 = 5.8 KiB groupshared
#ifndef QUIN_ITERS
#define QUIN_ITERS 2                                   // iter 0 is the integer tap (unseeded)
#endif

static const float MAX_DISP = 2.0;   // total displacement clamp, px
static const float SIGMA = 0.75;     // bilateral edge bandwidth, log2 stops
static const float B_WEIGHT = 1.0;   // structure-tensor B-channel weight
static const float LM_DAMP_REL = 1.0e-3;
static const float LM_DAMP_ABS = 1e-9;
static const float LOGL_EPS = 1e-6;
static const float3 LUMA = float3(0.299, 0.587, 0.114); // Rec.601, the bilateral guide

// Winsorization window: tol = TAU*median + ABS. Taken verbatim from upstream —
// quinlight's 1.0 is paper white and so is ours (scene::default_light), so a
// RELATIVE tolerance plus a near-black absolute floor mean the same thing in
// both.
static const float WINSOR_TAU = 0.1;
static const float WINSOR_ABS = 0.02;
// Upstream clamps samples to [0, 100] because its pixels are PQ display units
// (1.0 = 100 nits, so 100 IS its 10000-nit peak). Ours are LINEAR
// scene-referred — paper white ~1.0, a physical sun disc ~44000 — so a 100
// clamp would crush every highlight in the scene. The clamp's only job here is
// "a finite sample survives the RGBA16F store finite", so clamp at the output
// format's own ceiling.
static const float CLAMP_HI = 65504.0; // f16 max

cbuffer QuinCb : register(b0) {
    uint g_w;
    uint g_h;
    uint g_n;      // engines actually wired this session (1..MAX_ENGINES)
    uint g_anchor; // the engine that defines the spatial frame — never warped
};

Texture2D<float4> g_eng[MAX_ENGINES] : register(t0);
RWTexture2D<float4> g_dst : register(u0);
SamplerState g_lin : register(s0); // static: linear, clamp-to-edge

groupshared float3 s_anc[TILE * TILE];

// NOT isnan()/isinf(): DXC folds those away without strict IEEE mode, exactly
// as glslc does. Bit-test the exponent instead. A non-finite engine value is
// MISSING (blown HDR highlights emit NaN/Inf) — it must not enter the stack as
// a 0 addend, which would drag the mean toward black.
bool finite1(float x) { return ((asuint(x) >> 23) & 0xFFu) != 0xFFu; }
float clamp1(float x) { return clamp(x, 0.0, CLAMP_HI); }
float luma_of(float3 c) { return max(dot(c, LUMA), 0.0); }

// Ascending insertion sort + median — the winsorization centre.
float median_of(inout float a[MAX_ENGINES], uint m) {
    for (uint i = 1u; i < m; ++i) {
        float v = a[i];
        uint j = i;
        while (j > 0u && a[j - 1u] > v) {
            a[j] = a[j - 1u];
            j -= 1u;
        }
        a[j] = v;
    }
    uint mid = m / 2u;
    return (m & 1u) ? a[mid] : 0.5 * (a[mid - 1u] + a[mid]);
}

// Robust mean: clamp each of the `m` finite samples to median +/- tol, average.
// m == 1 is a passthrough; m == 2 is EXACTLY the plain mean (two samples are
// symmetric about their own median, so the clamp is a no-op on both) — which
// is why winsorization only starts rejecting outliers at m >= 3.
float winsorized_mean(inout float a[MAX_ENGINES], uint m) {
    if (m == 0u) return 0.0;
    if (m == 1u) return a[0];
    float center = median_of(a, m);
    float tol = WINSOR_TAU * max(center, 0.0) + WINSOR_ABS;
    float sum = 0.0;
    for (uint i = 0u; i < m; ++i) sum += clamp(a[i], center - tol, center + tol);
    return sum / (float)m;
}

// Clamp-to-edge integer fetch.
float4 texel(uint i, int2 c) {
    return g_eng[i].Load(int3(clamp(c, int2(0, 0), int2(g_w - 1u, g_h - 1u)), 0));
}
// Clamp-to-edge bilinear fetch at a float pixel coord — THE WARP. The +0.5
// puts an integer coord on the texel's centre, so an integer displacement
// reproduces `texel` exactly (gated).
float4 warp(uint i, float2 xy) {
    return g_eng[i].SampleLevel(g_lin, (xy + 0.5) / float2((float)g_w, (float)g_h), 0.0);
}

#define SANC(ox, oy) s_anc[((int)tid.y + HALO + (oy)) * TILE + ((int)tid.x + HALO + (ox))]

[numthreads(16, 16, 1)]
void cs_quin(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID,
             uint3 dtid : SV_DispatchThreadID, uint li : SV_GroupIndex) {
    int2 p = int2(dtid.xy);
    int w = (int)g_w;
    int h = (int)g_h;
    uint n = min(g_n, (uint)MAX_ENGINES);
    uint anchor = (n == 0u) ? 0u : min(g_anchor, n - 1u);

    // Degenerate engine counts. `n` is a root constant, so this branch is
    // UNIFORM across the workgroup — every thread returns here and none
    // reaches the barrier below, which is what makes the early-out legal.
    if (n <= 1u) {
        if (p.x < w && p.y < h) g_dst[p] = (n == 1u) ? texel(0, p) : float4(0, 0, 0, 1);
        return;
    }

    // Stage the anchor plane into groupshared BEFORE the divergent bounds
    // return, so all 256 threads reach the barrier (a barrier in divergent
    // control flow is undefined). Only the anchor is tiled — the engine warp
    // taps stay global reads, since they move with the solved displacement.
    int2 base = int2(gid.xy) * 16 - HALO;
    for (uint k = li; k < (uint)(TILE * TILE); k += 256u) {
        int2 tc = int2((int)k % TILE, (int)k / TILE);
        s_anc[k] = texel(anchor, base + tc).rgb;
    }
    GroupMemoryBarrierWithGroupSync();

    if (p.x >= w || p.y >= h) return;

    float4 anchor_c = texel(anchor, p); // the anchor is never warped: identity registration
    float3 cw = float3(1.0, 1.0, B_WEIGHT);

    // ---- Structure tensor over the window. ANCHOR-ONLY, so it is independent
    // of the engine being registered: built ONCE and reused for every engine
    // AND every refine iteration. Only the residual RHS is per-engine.
    float twgt[MAX_TAPS];
    float l_centre = luma_of(anchor_c.rgb);
    float log_lp = log2(l_centre + LOGL_EPS);
    float inv_two_sigma2 = 1.0 / (2.0 * SIGMA * SIGMA);
    float sxx = 0.0, syy = 0.0, sxy = 0.0, sw = 0.0;
    uint t = 0u;
    [unroll] for (int dy = -HALF_WINDOW; dy <= HALF_WINDOW; ++dy) {
        [unroll] for (int dx = -HALF_WINDOW; dx <= HALF_WINDOW; ++dx) {
            float3 a_c = SANC(dx, dy);
            float3 ix = 0.5 * (SANC(dx + 1, dy) - SANC(dx - 1, dy));
            float3 iy = 0.5 * (SANC(dx, dy + 1) - SANC(dx, dy - 1));
            // Bilateral edge weight in log2-luma: taps across a luminance edge
            // from the centre are down-weighted, so the solve does not average
            // two different surfaces' motion.
            float dl = log2(luma_of(a_c) + LOGL_EPS) - log_lp;
            float wgt = exp(-(dl * dl) * inv_two_sigma2);
            twgt[t] = wgt;
            t += 1u;
            float3 wix = cw * ix;
            float3 wiy = cw * iy;
            sxx += wgt * dot(wix, ix);
            syy += wgt * dot(wiy, iy);
            sxy += wgt * dot(wix, iy);
            sw += wgt;
        }
    }
    // Levenberg damping, luminance-relative: det >= alpha^2 > 0, so the system
    // is unconditionally invertible and EVERY pixel solves (a flat, textureless
    // region simply solves to ~zero displacement instead of blowing up).
    // Unless the anchor itself is non-finite — then guard det, forcing inv = 0,
    // so every step is zero and each engine folds in UNWARPED: the pixel
    // degrades to the unregistered mean, never to black.
    float le = l_centre + LOGL_EPS;
    float alpha = sw * (LM_DAMP_REL * le * le) + LM_DAMP_ABS;
    float ta = sxx + alpha;
    float td = syy + alpha;
    float tb = sxy;
    float det = ta * td - tb * tb;
    float inv = (finite1(det) && det > 0.0) ? 1.0 / det : 0.0;

    // ---- Reduce stack: the anchor (finite channels only), then each warped
    // non-anchor. Channels are independent — a NaN in R does not lose G and B.
    float r[MAX_ENGINES], g[MAX_ENGINES], b[MAX_ENGINES];
    uint mr = 0u, mg = 0u, mb = 0u;
    if (finite1(anchor_c.r)) { r[mr] = clamp1(anchor_c.r); mr += 1u; }
    if (finite1(anchor_c.g)) { g[mg] = clamp1(anchor_c.g); mg += 1u; }
    if (finite1(anchor_c.b)) { b[mb] = clamp1(anchor_c.b); mb += 1u; }

    for (uint i = 0u; i < n; ++i) {
        if (i == anchor) continue;

        // Solve (uu, vv) such that engine_i(q - uu) ~= anchor(q) over the
        // window; the warped sample is then engine_i at (p - uu). Unseeded:
        // start at (0,0), iteration 0 uses the integer taps (no warp), and
        // every step is TOTAL-clamped to +/- MAX_DISP.
        float uu = 0.0;
        float vv = 0.0;
        [loop] for (uint iter = 0u; iter < (uint)QUIN_ITERS; ++iter) {
            float rxt = 0.0, ryt = 0.0;
            uint tt = 0u;
            [unroll] for (int qy = -HALF_WINDOW; qy <= HALF_WINDOW; ++qy) {
                [unroll] for (int qx = -HALF_WINDOW; qx <= HALF_WINDOW; ++qx) {
                    float3 a_c = SANC(qx, qy);
                    float3 ix = 0.5 * (SANC(qx + 1, qy) - SANC(qx - 1, qy));
                    float3 iy = 0.5 * (SANC(qx, qy + 1) - SANC(qx, qy - 1));
                    float2 q = float2(p + int2(qx, qy));
                    float3 ew = (iter == 0u) ? texel(i, p + int2(qx, qy)).rgb
                                             : warp(i, q - float2(uu, vv)).rgb;
                    float3 it = a_c - ew; // brightness-constancy residual
                    float wgt = twgt[tt]; // the anchor-only weight cached above
                    tt += 1u;
                    rxt += wgt * dot(cw * ix, it);
                    ryt += wgt * dot(cw * iy, it);
                }
            }
            float r1 = -rxt;
            float r2 = -ryt;
            float du = (td * r1 - tb * r2) * inv;
            float dv = (-tb * r1 + ta * r2) * inv;
            uu = clamp(uu + du, -MAX_DISP, MAX_DISP);
            vv = clamp(vv + dv, -MAX_DISP, MAX_DISP);
        }

        float4 ew = warp(i, float2(p) - float2(uu, vv));
        if (finite1(ew.r)) { r[mr] = clamp1(ew.r); mr += 1u; }
        if (finite1(ew.g)) { g[mg] = clamp1(ew.g); mg += 1u; }
        if (finite1(ew.b)) { b[mb] = clamp1(ew.b); mb += 1u; }
    }

    g_dst[p] = float4(winsorized_mean(r, mr), winsorized_mean(g, mg), winsorized_mean(b, mb), 1.0);
}
