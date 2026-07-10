// Shared prelude for every GPU-tracer kernel. There is no #include — trace.rs
// concatenates the .hlsli sources ahead of each kernel before DXC sees them,
// so this file must stay self-contained and order-independent apart from
// coming first.
//
// Contract notes (mirrors of the CPU renderer, keep in lockstep):
// - ray_dir == camera.rs::CamBasis::ray_dir (normalized — distance == ray t).
// - sky_color == shade.rs::sky.
// - pack_info == overlay.rs::pack_info; KIND_* == overlay.rs.
// - The RNG is a per-pixel counter-based PCG stream seeded from
//   (x, y, frame, salt) — deliberately NOT the CPU's WyRand: sequences
//   differ, means match, and no exact-zero gate depends on the sequence.
//   Draw ORDER within a pixel mirrors the CPU exactly (jitter, then shadow
//   pairs, then AO pairs, then the two reflection draws).

#define PI  3.14159265358979
#define TAU 6.28318530717959
#define FLT_MAX 3.402823466e38
#define INF (asfloat(0x7f800000u))

#define KIND_LEAF    0u
#define KIND_SKY     1u
#define KIND_BLOCKED 2u
#define KIND_COARSE  3u

#define FLAG_ACCUM        1u   // frame > 0 adds instead of stores
#define FLAG_JITTER       2u   // per-pixel rng jitter (legacy accumulation)
#define FLAG_FRAME_JITTER 4u   // frame-uniform jitter from frame_jitter (DLSS/XeSS)
#define FLAG_VERIFY       8u   // check builds: hemi claim re-validation + PSA accounting

cbuffer Frame : register(b0) {
    float4 cam_origin;   // xyz; w = inv_w
    float4 cam_forward;  // xyz (unit); w = inv_h
    float4 cam_right;    // xyz pre-scaled by tan(fov/2)*aspect
    float4 cam_up;       // xyz pre-scaled by tan(fov/2)
    float4 sun;          // xyz (unit)
    float4 light_center; // xyz; w = scene eps
    float4 light_u;      // xyz; w = ao_radius
    float4 light_v;      // xyz
    float4 light_color;  // xyz (radiant intensity)
    uint rw; uint rh; uint frame; uint flags;
    uint shadow_samples; uint ao_samples; uint reflections; uint _pad0;
    float2 frame_jitter; float _pad1; float _pad2;
    // Wavefront queue capacities (resolution-derived, trace.rs computes them;
    // structural worst cases — the overflow counter is gated == 0).
    uint cap_tile; uint cap_leaf; uint cap_sky; uint cap_cut;
    // Hemisphere bounce state: fb_mode 0 = off, 1 = AO, 2 = GI; fb_depth =
    // the subdivision budget (Quality presets 2/3/4); hemi_batch = points
    // per batch (bounds transient queue memory); cap_hemi_pt = point-queue
    // capacity (rw*rh).
    uint fb_mode; uint fb_depth; uint hemi_batch; uint cap_hemi_pt;
    // Hemi per-batch queue capacities (batch-bounded, cannot overflow).
    uint cap_hemi_cell; uint cap_hemi_leaf; uint cap_hemi_cut; uint _pad3;
}

#define SCENE_EPS  (light_center.w)
#define AO_RADIUS  (light_u.w)

uint pack_info(uint depth, uint kind) { return (depth & 0xffu) | (kind << 8); }

// --- RNG: pcg-style hash stream ---------------------------------------------

uint pcg_mix(uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

uint rng_init(uint x, uint y, uint frame_idx, uint salt) {
    return pcg_mix(x * 0x9E3779B9u ^ y * 0xC2B2AE3Du ^ frame_idx * 0x27D4EB2Fu ^ salt);
}

float rng_next(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    w = (w >> 22u) ^ w;
    return float(w >> 8u) * (1.0 / 16777216.0);
}

// --- Camera ------------------------------------------------------------------

// camera.rs::ray_dir: pixel-grid coords (y down), normalized result.
float3 ray_dir(float fx, float fy) {
    float ndx = fx * cam_origin.w * 2.0 - 1.0;
    float ndy = 1.0 - fy * cam_forward.w * 2.0;
    return normalize(cam_forward.xyz + cam_right.xyz * ndx + cam_up.xyz * ndy);
}

// --- Sky (shade.rs::sky) ------------------------------------------------------

float3 sky_color(float3 d) {
    float t = saturate(d.y * 0.7 + 0.3);
    const float3 horizon = float3(0.72, 0.82, 0.95);
    const float3 zenith  = float3(0.18, 0.35, 0.70);
    float g = max(dot(d, sun.xyz), 0.0);
    // powi(32) by squaring, matching the CPU's exact operation chain.
    float g2 = g * g; float g4 = g2 * g2; float g8 = g4 * g4;
    float g16 = g8 * g8;
    float3 glow = (g16 * g16) * float3(1.0, 0.9, 0.7) * 0.6;
    return lerp(horizon, zenith, t) + glow;
}

// glam normalize_or_zero.
float3 normalize_or_zero(float3 v) {
    float l2 = dot(v, v);
    return l2 > 1e-30 ? v * rsqrt(l2) : float3(0.0, 0.0, 0.0);
}

// shade.rs::AMBIENT — also consumed by the compose pass (hemi AO mode).
static const float3 AMBIENT = float3(0.14, 0.17, 0.23);

// Duff et al. orthonormal basis; right-handed (t1 x t2 = n) — the hemisphere
// octant orientation relies on it (sphcell::self_test asserts the CPU twin).
void onb(float3 n, out float3 t1, out float3 t2) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    t1 = float3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    t2 = float3(b, s + n.y * n.y * a, -n.y);
}
