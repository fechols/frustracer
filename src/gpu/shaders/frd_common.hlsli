// FRD shader common — the HLSL twin of src/frd.rs::oracle. One formula, two
// sites (the oracle's F0 pins + the GPU gates keep them in lockstep); every
// constant here mirrors a named const in frd.rs — change both together.
//
// CLEAN-ROOM: written from the published literature (RTG II ch. 49, the GDC
// self-stabilizing-recurrent-blurs talk, SVGF) — never from the NRD source.
// The WIRE helpers (YCoCg layout, enc-2 normals, hit-dist normalization) are
// the bridge's own (nrd_bridge.hlsl) and FRD consumes their planes verbatim.
//
// FRD_FP16: 1 = native fp16 filter math (compiled with -enable-16bit-types,
// the phase-D arm), 0 = fp32 throughout. Accumulator-critical math (variance,
// tap sums) stays fp32 in BOTH arms (the Welford rule).

#if FRD_FP16
typedef float16_t fh;
typedef float16_t3 fh3;
typedef float16_t4 fh4;
#else
typedef float fh;
typedef float3 fh3;
typedef float4 fh4;
#endif

// frd.rs constants (LOCKSTEP).
static const float FRD_MAX_ACCUM = 30.0;
static const float FRD_FAST_FRAMES = 4.0;
static const float FRD_CLAMP_SIGMA = 2.5;
static const float FRD_CHROMA_BOX = 0.5;
static const float FRD_Z_EPS = 0.02;
static const float FRD_N_COS_DIFF = 0.9063078;
static const float FRD_N_COS_SPEC_SMOOTH = 0.998;
static const float FRD_SPEC_PARALLAX_K = 30.0;
static const float FRD_RANGE_K = 0.999;
static const float FRD_FIREFLY_K = 8.0;
static const float FRD_ANTILAG_E_FLOOR = 0.01;
// v1.5 specular virtual motion (frd.rs VM_* twins).
static const float FRD_VM_FAR_K = 0.99;
static const float FRD_VM_ROUGH_LO = 0.25;
static const float FRD_VM_ROUGH_HI = 0.6;
static const float FRD_VM_Z_SLACK = 0.05;
static const float FRD_VM_DEADZONE2 = 2.5e-3;
static const float FRD_VM_DN_DZ = 5e-3;

// The wire's enc-2 normal+roughness decode (nrd_pack_nr's inverse — the
// nrd.rs::oracle decode twin): L1-octahedral xy, SIGNED roughness in z.
void frd_decode_nr(float4 p, out float3 n, out float rough) {
    float t = p.z * 2.0 - 1.0;
    float x = p.x - p.y;
    float y = p.x + p.y - 1.0;
    float zs = t < 0.0 ? -1.0 : 1.0;
    n = normalize(float3(x, y, zs * (1.0 - abs(x) - abs(y))));
    rough = abs(t);
}

// oracle::disocclusion_z_valid — relative view-Z test vs the exact z + mv.z
// expectation, grazing-relaxed by 1/max(n·v, 0.1).
bool frd_z_valid(float z_expect, float z_prev, float ndv) {
    float scale = max(max(abs(z_expect), abs(z_prev)), 1e-6);
    float rel = abs(z_prev - z_expect) / scale;
    return rel < FRD_Z_EPS / max(ndv, 0.1);
}

// oracle::virtual_dist — v1.5.1: the PARAXIAL MIRROR EQUATION on the
// unfold distance (convex f = −R/2, exact at all object distances): the
// distance a CURVED mirror actually images at. κ = 0 is a BRANCH
// returning t_r VERBATIM (the flat path — still water — stays bitwise;
// a formula identity is one fmad contraction away from not holding);
// κ·t_r ≫ 1 ⇒ t_v → 1/(2κ) ≈ R/2, the image just behind the surface ⇒
// the fetch lands at the SURFACE reprojection — the correct answer for a
// convex mirror's sun glint (the helmet-streak fix: the flat sky arm
// pinned vacated pixels to their own stale bright history). For κ ≥ 0,
// t_v ∈ (0, t_r] monotone — κ errors interpolate between the two shipped
// behaviors, never past either.
float frd_virtual_dist(float t_r, float kappa) {
    return kappa <= 0.0 ? t_r : t_r / (1.0 + 2.0 * kappa * t_r);
}

// oracle::vm_curvature — |Δn| central-differenced over a 2px baseline per
// axis (decoded wire normals), dead-zoned SUBTRACTIVELY against the
// 10-bit oct quantization (continuous through the boundary — a hard
// threshold steps the fetch on the quantization staircase), MAX of the
// axes (errs surface-fetch-ward only), over the 2px world step 2z/proj.
// Magnitude-only: concave reads convex, which only shortens t_v toward
// the surface fetch — the humble direction.
float frd_vm_curvature(float dn_x, float dn_y, float view_z, float proj) {
    float dn = max(max(dn_x, dn_y) - FRD_VM_DN_DZ, 0.0);
    return dn * proj / (2.0 * max(view_z, 1e-4));
}

// oracle::disocclusion_z_valid_slack — the VIRTUAL foot's widened test
// (slack = 1 + VM_Z_SLACK · |applied offset px|): a grazing planar mirror's
// reflector depth varies along the plane, so a far-displaced virtual foot
// needs proportionally more relative-Z tolerance or every mirror fetch
// history-restarts. slack 1.0 IS frd_z_valid exactly.
bool frd_z_valid_slack(float z_expect, float z_prev, float ndv, float slack) {
    float scale = max(max(abs(z_expect), abs(z_prev)), 1e-6);
    float rel = abs(z_prev - z_expect) / scale;
    return rel < FRD_Z_EPS * slack / max(ndv, 0.1);
}

// oracle::normal_cos_threshold.
float frd_ncos_thresh(float rough, bool spec) {
    if (spec) {
        float r = saturate(rough);
        return FRD_N_COS_SPEC_SMOOTH + (FRD_N_COS_DIFF - FRD_N_COS_SPEC_SMOOTH) * r;
    }
    return FRD_N_COS_DIFF;
}

// oracle::spec_max_frames — the parallax-capped specular history.
float frd_spec_max_frames(float n_max, float parallax, float rough) {
    float r = saturate(rough);
    float smooth_cap = 1.0 / (1.0 + FRD_SPEC_PARALLAX_K * max(parallax, 0.0) * (1.0 - r) * (1.0 - r));
    return n_max * (smooth_cap + (1.0 - smooth_cap) * r);
}

// oracle::clamp_ycocg — luma box ± k·sigma, chroma boxed at CHROMA_BOX of it.
// In-box values return BITWISE (the converged-signal contract).
float3 frd_clamp_ycocg(float3 s, float3 fast, float sigma_l, float k) {
    float bl = k * max(sigma_l, 0.0);
    float3 bw = float3(bl, bl * FRD_CHROMA_BOX, bl * FRD_CHROMA_BOX);
    return clamp(s, fast - bw, fast + bw);
}

// oracle::firefly_scale — the soft input clamp vs the 3x3 neighborhood
// luma mean: under FIREFLY_K x mean the scale is EXACTLY 1.0 (the common
// path is bitwise untouched); past it the sample compresses to the cap —
// attenuated, never deleted. A radiance scale on the YCoCg lanes only;
// the .w hit-dist lane is NEVER scaled.
float frd_firefly_scale(float luma, float mean3x3) {
    float cap = FRD_FIREFLY_K * max(mean3x3, 1e-6);
    return (luma <= cap || luma <= 0.0) ? 1.0 : cap / luma;
}

// oracle::antilag_gain — the anti-lag brake's wire form: pass 3 stores
// g = 1/max(e, 1) (e = the clamp's |pre − fast| luma distance in box
// widths) into the antilag plane; pass 1 applies it via frd_antilag_apply.
// antilag_frames(n, e) == n·g EXACTLY (the F0 identity pin), and in-box g
// is exactly 1.0.
float frd_antilag_gain(float e) {
    return 1.0 / max(e, 1.0);
}

// oracle::antilag_excess — |pre − fast| in box widths, the denominator
// floored RELATIVELY (E_FLOOR·|fast|) before the absolute backstop: a
// converged signal's box is exactly zero-width (m2 = 0) while blur fp
// residue is ulp-scale-of-RADIANCE, so an absolute floor misreads it as
// excess and mints junk gains on perfectly converged pixels (the F5b
// find). A sub-1%-of-signal deviation is never a ghost.
float frd_antilag_excess(float delta_abs, float box_w, float fast_luma) {
    return delta_abs / max(max(box_w, FRD_ANTILAG_E_FLOOR * abs(fast_luma)), 1e-6);
}

// --- The spatial passes' twins (frd.rs constants, LOCKSTEP) ---------------

static const float FRD_S_MIN = 0.15;
static const float FRD_N_FIX = 4.0;
static const float FRD_R_FIX = 30.0;
static const float FRD_LOBE_K = 0.75;
static const float FRD_C_DIFF = 0.5;
static const float FRD_C_SPEC = 2.0;
static const float FRD_N_POW_DIFF = 8.0;
static const float FRD_N_POW_SPEC_SMOOTH = 128.0;
static const float FRD_GOLDEN = 2.3999632;
static const uint FRD_TAPS = 8;

// The wire's hit-distance normalization factor (nrd_bridge.hlsl's
// nrd_norm_hit_dist denominator, same {3, 0.1, 20} literals) — FRD stores
// the wire's NORMALIZED .w and denormalizes where a world radius is needed.
float frd_hitdist_denorm_factor(float view_z, float rough) {
    float f = 1.0 - exp2(-200.0 * rough * rough);
    float smc = f * sqrt(saturate(rough));
    return (3.0 + abs(view_z) * 0.1) * lerp(20.0, 1.0, smc);
}

// oracle::radius_diffuse_px / radius_spec_px / radius_accum_scale /
// history_fix_radius — r_max rides the CB (the --frd-blur-radius lever).
float frd_radius_diffuse(float hit_w, float view_z, float proj, float r_max) {
    return clamp(FRD_C_DIFF * hit_w / max(view_z, 1e-6) * proj, 1.0, r_max);
}

float frd_radius_spec(float hit_w, float view_z, float rough, float proj, float r_max) {
    float a = saturate(rough);
    float tan_lobe = FRD_LOBE_K * a * a;
    return clamp(FRD_C_SPEC * hit_w * tan_lobe / max(view_z + hit_w, 1e-6) * proj, 0.0, r_max);
}

float frd_accum_scale(float n) {
    return max(1.0 / (1.0 + max(n, 0.0)), FRD_S_MIN);
}

float frd_history_fix(float r_eff, float n) {
    return n < FRD_N_FIX ? max(r_eff, FRD_R_FIX * (1.0 - n / FRD_N_FIX)) : r_eff;
}

// oracle::antilag_apply — the anti-lag brake's consumer, FLOORED at the
// history-fix window: the cut accelerates recovery down to the
// fast-history rate but never INTO the restart window (n < N_FIX re-arms
// the wide history-fix blur, which wipes sharp features from the recurrent
// feedback and re-fires the clamp — the F6-caught limit cycle; a converged
// noise-free signal's m2 is exactly 0, so its clamp box has zero width and
// ANY gap reads as an infinite excess — the floor is what keeps sharp
// converged features stable under the brake).
float frd_antilag_apply(float n, float g) {
    return max(n * g, min(n, FRD_N_FIX));
}

// oracle::normal_pow — the normal-agreement exponent, hoisted per pixel
// (constant across a pixel's taps; specular sharpens toward
// N_POW_SPEC_SMOOTH as roughness → 0).
float frd_normal_pow(float rough, bool spec) {
    if (spec) {
        float r = saturate(rough);
        return FRD_N_POW_SPEC_SMOOTH + (FRD_N_POW_DIFF - FRD_N_POW_SPEC_SMOOTH) * r;
    }
    return FRD_N_POW_DIFF;
}

// --- The FUSED tap weight, log2 domain (oracle::tap_exp2 + inv_z_scale /
// inv_h_scale — the B70 EM-pipe diet). What used to be four factors —
// Gaussian exp(−2·r_i²) · z-relative exp(−|Δz|/(0.05·|z|)) ·
// saturate(n·n)^pow · hit-dist exp(−|Δh|/(h+1e-3)) — is ONE exponent sum
// under ONE exp2: per tap per signal the transcendental bill drops from
// sincos+sqrt+pow+3·exp to log2+exp2. The Gaussian needs no distance at
// all: the Vogel radius r_i² = (i+0.5)/TAPS is a per-INDEX constant, so
// exp(−2·d²) is −(i+0.5)·FRD_TAP_G2 in log2 domain, and the old per-tap
// length()/sqrt is gone with it.
static const float FRD_LOG2E = 1.442695;    // log2(e)
static const float FRD_TAP_G2 = 0.36067376; // 2·log2(e)/TAPS

float frd_inv_z_scale(float z_center) {
    return FRD_LOG2E / (0.05 * max(abs(z_center), 1e-4));
}

float frd_inv_h_scale(float h_center) {
    return FRD_LOG2E / (h_center + 1e-3);
}

// Diffuse passes inv_h = 0 (v.w never influences a diffuse tap — the wire
// keeps it finite); saturate(ndn) ≤ 0 drives log2 → −inf → exp2 → exactly
// 0, the old pow(saturate(ndn), pw) limit.
float frd_tap_exp2(uint i, float dz_abs, float inv_z, float ndn, float pw, float dh_abs, float inv_h) {
    return -(float(i) + 0.5) * FRD_TAP_G2 - dz_abs * inv_z
        + pw * log2(saturate(ndn)) - dh_abs * inv_h;
}

// oracle::VOGEL8 / vogel_rot — the unrotated Vogel disk as LITERALS (r_i =
// sqrt((i+0.5)/8), θ_i = i·2.3999632; f64-generated, F0-pinned against the
// analytic generator), rotated by the per-(pixel, frame) hash angle at 2
// fma per tap — ONE sincos per pixel per pass instead of one per tap.
static const float2 FRD_VOGEL8[8] = {
    float2(0.25, 0.0),
    float2(-0.319290081, 0.292495887),
    float2(0.0488724328, -0.556876544),
    float2(0.402444525, 0.524917521),
    float2(-0.73853513, -0.130636375),
    float2(0.699604866, -0.445031495),
    float2(-0.234004003, 0.870483846),
    float2(-0.446271487, -0.859268154),
};

float2 frd_vogel_rot(uint i, float cr, float sr) {
    float2 u = FRD_VOGEL8[i];
    return float2(u.x * cr - u.y * sr, u.x * sr + u.y * cr);
}

// The per-(pixel, frame) rotation angle: a pure integer hash (the clouds
// dither discipline: zero rng draws, deterministic).
float frd_disk_rot(uint2 p, uint frame, uint salt) {
    uint h = p.x * 73856093u ^ p.y * 19349663u ^ (frame + salt) * 83492791u;
    h = (h ^ (h >> 16)) * 0x7feb352du;
    h = (h ^ (h >> 15)) * 0x846ca68bu;
    h = h ^ (h >> 16);
    return float(h & 0xFFFFu) * (6.2831853 / 65536.0);
}
