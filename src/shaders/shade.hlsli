// Scene resources + the shade.rs port. Requires trace_common.hlsli +
// rt.hlsli pasted first (trace.rs does the concatenation).
//
// This is shade.rs::shade with VisCtl::Off — the exact interactive non-bounce
// path — with the recursion flattened into a bounded DFS loop. The CPU
// recursion tree: the VNDF reflection bounce fires at depth 0 only; the
// glass transmission chain fires at any depth < TRANS_MAX_DEPTH — so only
// the ROOT can have two children (reflection + transmission), one stash slot
// is provably sufficient, and the loop runs at most 1 + TRANS_MAX_DEPTH +
// TRANS_MAX_DEPTH = 9 laps in the CPU's exact DFS order (reflection subtree
// first, then the root's transmission chain). Every ray here goes through
// the shared trace primitives (inline RayQuery, or TraceRay in the DXR
// library — transmission continuations route to HgHit, so the whole tree
// stays at TraceRay recursion depth 2); secondary rays use tmin = 0 from the
// eps-offset surface point, exactly like the CPU (the quadtree's inherited
// tmin is a primary-frustum property — it must never leak into shading; see
// CLAUDE.md invariants). Hemi mode (fb_mode > 0) splits the PRIMARY
// surface's ambient out: shade_split returns the ambient-free color plus the
// kd weight and surface point the hemi wavefront consumes; every non-root
// lap shades with its own sampled ambient (the CPU forces fb OFF past depth
// 0).

// Dev cost-attribution ablations (FR_ABL=noshadow|noao|norefl|noglass|nosec —
// trace::abl_defs): neutralize ONE secondary-ray consumer's TRAVERSAL while
// keeping every rng draw and all control flow (shade.rs::Abl's discipline —
// the delta prices the rays and nothing else). Cost probes, never levers: the
// image changes by design (the nogbuf class). They reach every unit that
// pastes this file — leaf, reference, AND the DXR library — which is what
// makes a primary-vs-secondary split inside the one opaque DispatchRays
// region measurable at all. The off arm of every wrapper is the verbatim
// call, so an unarmed session compiles byte-identical sources.
#if defined(ABL_NOSEC) || defined(ABL_NOSHADOW)
#define ABL_TQ_SHADOW(o, d, t0, t1) (float3(1.0, 1.0, 1.0))
#else
#define ABL_TQ_SHADOW(o, d, t0, t1) transmit_q(o, d, t0, t1)
#endif
#if defined(ABL_NOSEC) || defined(ABL_NOAO)
#define ABL_TQ_AO(o, d, t0, t1) (float3(1.0, 1.0, 1.0))
#else
#define ABL_TQ_AO(o, d, t0, t1) transmit_q(o, d, t0, t1)
#endif
// _T flavors of the two occlusion wrappers: the same queries, additionally
// reporting the occluder t (the NRD hit-distance capture — sig.w; sample 0
// only, the side-channel rule). The ablation arms report t1 (= miss), so a
// neutralized traversal cannot resurface as a live-looking guide.
float3 abl_tq_pass_t(float t1, out float first_t) {
    first_t = t1;
    return float3(1.0, 1.0, 1.0);
}
#if defined(ABL_NOSEC) || defined(ABL_NOSHADOW)
#define ABL_TQ_SHADOW_T(o, d, t0, t1, ot) abl_tq_pass_t(t1, ot)
#else
#define ABL_TQ_SHADOW_T(o, d, t0, t1, ot) transmit_q_t(o, d, t0, t1, ot)
#endif
#if defined(ABL_NOSEC) || defined(ABL_NOAO)
#define ABL_TQ_AO_T(o, d, t0, t1, ot) abl_tq_pass_t(t1, ot)
#else
#define ABL_TQ_AO_T(o, d, t0, t1, ot) transmit_q_t(o, d, t0, t1, ot)
#endif
#if defined(ABL_NOSEC) || defined(ABL_NOREFL)
#define ABL_TRACE_REFL(o, d, t0, t1, h) false
#else
#define ABL_TRACE_REFL(o, d, t0, t1, h) trace_closest(o, d, t0, t1, h)
#endif
#if defined(ABL_NOSEC) || defined(ABL_NOGLASS)
#define ABL_TRACE_GLASS(o, d, t0, t1, h) false
#else
#define ABL_TRACE_GLASS(o, d, t0, t1, h) trace_closest(o, d, t0, t1, h)
#endif
#if defined(ABL_NOSEC) || defined(ABL_NOGI)
// FR_ABL=nogi: drop the RTGI bounce ray AND the bounce shade behind it (the
// norefl shape for a recursive consumer) — ambient degrades to the unoccluded
// sky gather. The 2 direction draws above the trace still run (shade.rs).
#define ABL_TRACE_GI(o, d, t0, t1, h) false
#else
#define ABL_TRACE_GI(o, d, t0, t1, h) trace_closest(o, d, t0, t1, h)
#endif

// t0 (bvh nodes) and t1 (tri_idx) belong to the frustum kernels.
StructuredBuffer<float3> positions : register(t2);
StructuredBuffer<float3> normals   : register(t3);
StructuredBuffer<uint>   indices   : register(t4); // flat, 3 per tri (also the BLAS index buffer)
StructuredBuffer<uint>   tri_mat   : register(t5);

#define MAT_DIFFUSE  0u
#define MAT_MARBLE   1u
#define MAT_TEXTURED 2u

// scene.rs::NO_TEX — "no map" sentinel for the texture-index fields.
#define TEX_NONE 0xffffffffu

// Mirrors trace.rs::GpuMat field-for-field (108 B) — a stride skew reads
// garbage; the two must move in the same commit.
struct Mat {
    float3 albedo;
    float roughness;
    float metallic;
    float anisotropy;
    uint kind;
    float scale; // marble feature frequency
    float sheen;
    float translucency;
    float transmission;
    uint tex; // Scene::textures index (MAT_TEXTURED: the space1 texs[] slot)
    float3 emissive;   // Ke; added at every lap, outside the kd*(1-transmission) factor
    uint normal_tex;   // tangent-space normal map (TEX_NONE = none; UNORM SRV)
    uint rough_tex;    // roughness map, samples .g (glTF channel convention)
    uint metal_tex;    // metallic map, samples .b
    uint emissive_tex; // emissive map (sRGB SRV)
    float normal_scale;
    float3 trans_tint; // transmission/absorption tint; sentinel .x < 0 = "use albedo"
    float ior;         // Snell/Fresnel IOR (default 1.5; water 1.33)
    float ripple_amp;  // water ripple slope amplitude (0 = none)
    // Per-material world-space detail texel scale (Scene::detail_scales —
    // never per-face, which seams on greedy-meshed atlases). 0 = field off.
    float detail_scale;
    // Spec-AA: the normal map's slope-variance companion texture
    // (Scene::tex_var; TEX_NONE = none — the fold's map-arm off state).
    uint normal_var_tex;
};
StructuredBuffer<Mat> materials : register(t6);

// Material::trans_tint_or — the ONE tint source (per-interface glass tint,
// Beer–Lambert, shadow tint): trans_tint when set, else albedo verbatim.
float3 trans_tint_or(Mat m, float3 albedo) {
    return m.trans_tint.x >= 0.0 ? m.trans_tint : albedo;
}

// shade.rs's glassware constants: the interface budget for the refraction
// chain (front/back walls of a two-walled tumbler, no TIR detour). Past the
// budget, glass shades opaque. IOR now rides mat.ior (default 1.5; water
// 1.33) — GLASS_IOR stays only as the documented default.
#define GLASS_IOR 1.5
#define TRANS_MAX_DEPTH 4u
// shade.rs::TRANS_DEPTH_K — the Beer–Lambert reference depth (relative to
// SCENE_DIAG) at which transmitted light is tinted to exactly the albedo.
#define TRANS_DEPTH_K 0.015

// --- shade.rs helper ports ----------------------------------------------------

float3 cosine_dir(float3 n, float3 t1, float3 t2, float r1, float r2) {
    float phi = TAU * r1;
    float sq = sqrt(r2);
    return t1 * (cos(phi) * sq) + t2 * (sin(phi) * sq) + n * sqrt(1.0 - r2);
}

void ggx_alphas(float roughness, float anisotropy, out float ax, out float ay) {
    const float MIN_ALPHA = 5e-3;
    float alpha = roughness * roughness;
    float aspect = sqrt(1.0 - 0.9 * anisotropy);
    ax = max(alpha / aspect, MIN_ALPHA);
    ay = max(alpha * aspect, MIN_ALPHA);
}

// Smith lambda for anisotropic GGX; w in tangent space, w.z > 0.
float ggx_lambda(float3 w, float ax, float ay) {
    float t = ((ax * w.x) * (ax * w.x) + (ay * w.y) * (ay * w.y)) / (w.z * w.z);
    return (sqrt(1.0 + t) - 1.0) * 0.5;
}

float ggx_ndf(float3 h, float ax, float ay) {
    float hx = h.x / ax;
    float hy = h.y / ay;
    float d = hx * hx + hy * hy + h.z * h.z;
    return 1.0 / (PI * ax * ay * d * d);
}

float3 schlick(float3 f0, float c) {
    float m = saturate(1.0 - c);
    float m2 = m * m;
    return f0 + (1.0 - f0) * (m2 * m2 * m);
}

// shade.rs::hash3/vnoise/fbm/marble — deterministic world-space veining.
float hash3(int x, int y, int z) {
    uint h = uint(x) * 0x8da6b343u ^ uint(y) * 0xd8163841u ^ uint(z) * 0xcb1ab31fu;
    h ^= h >> 13u;
    h *= 0x9e3779b1u;
    h ^= h >> 16u;
    return float(h & 0x00ffffffu) / 16777216.0;
}

float vnoise(float3 p) {
    float3 f = floor(p);
    int ix = int(f.x), iy = int(f.y), iz = int(f.z);
    float3 t = p - f;
    float3 s = t * t * (3.0 - 2.0 * t);
    float x00 = lerp(hash3(ix, iy, iz),         hash3(ix + 1, iy, iz),         s.x);
    float x10 = lerp(hash3(ix, iy + 1, iz),     hash3(ix + 1, iy + 1, iz),     s.x);
    float x01 = lerp(hash3(ix, iy, iz + 1),     hash3(ix + 1, iy, iz + 1),     s.x);
    float x11 = lerp(hash3(ix, iy + 1, iz + 1), hash3(ix + 1, iy + 1, iz + 1), s.x);
    return lerp(lerp(x00, x10, s.y), lerp(x01, x11, s.y), s.z);
}

float fbm(float3 p) {
    float amp = 0.5;
    float sum = 0.0;
    [unroll] for (int i = 0; i < 5; ++i) {
        sum += amp * vnoise(p);
        p *= 2.02;
        amp *= 0.5;
    }
    return sum;
}

float3 marble(float3 p, float scale) {
    const float3 BASE = float3(0.93, 0.92, 0.90);
    const float3 VEIN = float3(0.10, 0.11, 0.15);
    float3 q = p * scale;
    float s = sin(q.x + 0.7 * q.y + 5.0 * fbm(q));
    float t = saturate((abs(s) - 0.04) / 0.18);
    return lerp(VEIN, BASE, t * t * (3.0 - 2.0 * t));
}

// shade.rs::detail_field / detail_bump — Unreal-1 style detail texturing,
// mirrored verbatim (constants are LITERALS, the clouds-wind idiom — change
// both together). Three octaves of WORLD-SPACE 3D value noise multiply the
// albedo and tilt the shading normal where the base texture is MAGNIFIED
// (dlod < 0); octave k's window saturate(-dlod - k) IS its anti-alias and
// the progressive fade ladder. THE DOMAIN is q3 = p_rest / s: the hit's
// barycentric REST-pose position (tri_rest_point — positions[] is the rest
// buffer, sway rides TLAS instance transforms) over the per-hit texel world
// size s from tri_uv_basis, so octave 0 stays one cell per texel-EQUIVALENT
// and atlas meshes (rungholt) stop tiling the noise in lockstep with their
// repeated UV rects. Value + analytic gradient come from ONE
// cloud_vnoise3_vg eval per octave (trace_common.hlsli, pasted ahead of
// this file in every unit), so grain and bump are one coherent surface.
// Zero rng draws. Degenerate lod is -inf on the CPU and -1e30 here — every
// window saturates to 1 identically, output bounded at whatever q3 came in.
#define DETAIL_AMP      0.18
#define DETAIL_BUMP_K   2.0
// Session strength multipliers (--detail-strength / --detail-ao-strength),
// injected by trace::detail_defs into every unit that pastes this file; the
// #ifndef fallbacks MATCH the compiled defaults (0.5 grain / 0.125 AO — the
// 2026-08-06 feel-test calibration; 1.0 spells the original full-strength
// field, and ×1.0 is the bitwise-exact arm), so a unit the injection ever
// missed would still agree with the CPU's default statics — the probe-reach
// rule's fail-safe. shade.rs reads the scene:: statics — CPU/GPU move
// together by value.
#ifndef DETAIL_STR
#define DETAIL_STR 0.5
#endif
#ifndef DETAIL_AO_STR
#define DETAIL_AO_STR 0.125
#endif
// Direct N·L contrast ceiling for the detail bump (round 6, shade.rs::
// DETAIL_NDL_CAP): the tilt may move the sun's diffuse by at most ±CAP of
// the PRE-detail N·L — under-cap pixels (tops) keep the raw value bitwise,
// grazing-lit faces (tan(incidence) large) compress to the ceiling.
#define DETAIL_NDL_CAP  0.5
// shade.rs::detail_ndl_cap — raw = n_s·wi (post-detail), p = n_pre·wi.
float detail_ndl_cap(float raw, float p) {
    return (p > 0.0)
        ? clamp(raw, p * (1.0 - DETAIL_NDL_CAP), p * (1.0 + DETAIL_NDL_CAP))
        : min(raw, 0.0);
}
#define DETAIL_AO_BUMP_K 1.5
#define DETAIL_ROUGH_LO 0.2
#define DETAIL_ROUGH_HI 0.45

// shade.rs::detail_bump_weight — the bump's roughness damping (the
// frosted-visor guard): 0 at/below LO (detail_bump's g == 0 guard then
// returns the base verbatim), 1 at/above HI. Reads the MAP-DRIVEN per-pixel
// rough_eff — safe, the bump draws no rng.
float detail_bump_weight(float rough) {
    return saturate((rough - DETAIL_ROUGH_LO) / (DETAIL_ROUGH_HI - DETAIL_ROUGH_LO));
}
// shade.rs::SPEC_AA_S2_CAP / VNOISE_GRAD_VAR — the spec-AA literals
// (texture.rs::SPEC_AA_S2_CAP is the encode's cap; VNOISE_GRAD_VAR is the
// measured per-component gradient variance of the shared vnoise — both
// mirrored, the clouds-wind idiom; spec_aa_self_test re-measures the
// latter, so change all copies together).
#define SPEC_AA_S2_CAP 0.5
#define VNOISE_GRAD_VAR 0.1104
// shade.rs::detail_var — the spec-AA detail transfer: per-axis slope
// variance of detail tilt NOT applied because its octave windows have
// closed (applied variance scales wk², the discarded share is 1 − wk² of
// each octave's full-on variance; grain full-on factor 2·AMP·STR·BUMP_K is
// scale-invariant across k). Fully-open windows contribute an IEEE-exact
// 0.0 — magnified pixels transfer nothing bit-identically; past
// DETAIL_AO_RANGE the transfer plateaus at the field's whole variance. The
// consumer weights by detail_bump_weight² (applied + transferred = bw²·full
// at every distance — the invariant). Term-for-term CPU mirror.
float detail_var(float dlod) {
    float sum = 0.0;
    float a = 2.0 * DETAIL_AMP * DETAIL_STR * DETAIL_BUMP_K;
    [unroll] for (uint k = 0u; k < 3u; ++k) {
        float wk = saturate(-dlod - float(k));
        sum += a * a * (1.0 - wk * wk);
    }
    if (flags & FLAG_DETAIL_AO) {
        float c8 = 0.5 * DETAIL_AO_STR * (2.0 / 8.0) * DETAIL_AO_BUMP_K * DETAIL_BUMP_K;
        float wk = saturate(3.0 - dlod);
        sum += c8 * c8 * (1.0 - wk * wk);
        float c4 = 0.35 * DETAIL_AO_STR * (2.0 / 4.0) * DETAIL_AO_BUMP_K * DETAIL_BUMP_K;
        wk = saturate(2.0 - dlod);
        sum += c4 * c4 * (1.0 - wk * wk);
    }
    return sum * VNOISE_GRAD_VAR;
}
// shade.rs::detail_aniso_base — the detail window's lod base under the aniso
// filter: log2 of the footprint's MINOR axis (what SampleGrad leaves
// unresolved), deliberately uncapped by MaxAnisotropy (see the shade.rs doc
// for the ~0.32-lod grazing known-accept that buys a max-free twin).
float detail_aniso_base(float2 gu, float2 gv) {
    return log2(max(min(length(gu), length(gv)), 1e-20));
}
// shade.rs::DETAIL_AO_K / detail_cavity — cavity AO from the field's own
// signed height h = value − 1 (mean 0 by construction — zero extra lookups):
// pits darken ambient + direct specular, peaks return exactly 1.0 (callers
// branch on h < 0). Not energy-neutral: it is occlusion.
#define DETAIL_AO_K 3.0
float detail_cavity(float h) {
    return exp(DETAIL_AO_K * min(h, 0.0));
}
// FLAG_REMOD_EXACT's energy blend — the ONE site the wire delta multiplier's
// formula lives (nrd::remod::blend is the CPU twin the N9 gate scores against;
// keep the two in lockstep). The diffuse channel carries two sub-terms with
// different exact factors (see PrimSurf), so the multiplier that makes the
// DELTA exact is their energy-weighted mean:
//
//     m = (E_a*k_a + E_b*k_b) / (E_a + E_b)
//
// A convex combination, so m is bounded by [min(k_a,k_b), max(k_a,k_b)] BY
// CONSTRUCTION — it can never amplify, and it is exactly k_a == k_b whenever
// the two factors agree (every non-pit pixel, and every pixel under
// --no-detail-ao), which is what keeps the common path free of any weighting
// at all. A zero-energy pixel contributes nothing to the recompose either way,
// so the degenerate denominator returns k_a rather than dividing.
//
// Energy is the channel MEAN, not a perceptual luma: this weights how much of
// the delta each sub-term owns, which is a radiometric question.
float remod_blend(float3 e_a, float k_a, float3 e_b, float k_b) {
    float a = dot(max(e_a, 0.0), float3(1.0, 1.0, 1.0) / 3.0);
    float b = dot(max(e_b, 0.0), float3(1.0, 1.0, 1.0) / 3.0);
    float s = a + b;
    return s > 0.0 ? (a * k_a + b * k_b) / s : k_a;
}
// shade.rs::DETAIL_AO_RANGE / detail_ao_field — coarse height octaves
// (8/4-texel-equivalent cells, salts 43/44, windows log2(cell) − dlod): the
// lower-frequency pools that make the cavity read as AO and reach
// mid-distance, returning their per-q-unit GRADIENT too (chain rule: an
// octave samples at q/div, so it carries 1/div) — the relief-rim share of
// the micro-bump (× DETAIL_AO_BUMP_K). Term-for-term CPU mirror.
#define DETAIL_AO_RANGE 3.0
void detail_ao_field(float3 q3, float dlod, out float hh, out float3 g) {
    hh = 0.0;
    g = float3(0.0, 0.0, 0.0);
    float v;
    float3 gv;
    float wk = saturate(3.0 - dlod);
    if (wk > 0.0) {
        cloud_vnoise3_vg(q3 / 8.0, 43u, v, gv);
        hh += (0.5 * DETAIL_AO_STR) * wk * (2.0 * v - 1.0);
        g += gv * ((0.5 * DETAIL_AO_STR) * wk * 2.0 / 8.0);
    }
    wk = saturate(2.0 - dlod);
    if (wk > 0.0) {
        cloud_vnoise3_vg(q3 / 4.0, 44u, v, gv);
        hh += (0.35 * DETAIL_AO_STR) * wk * (2.0 * v - 1.0);
        g += gv * ((0.35 * DETAIL_AO_STR) * wk * 2.0 / 4.0);
    }
}
// shade.rs::DETAIL_SHADOW_* / detail_shadow_h / detail_sun_shadow — the REAL
// horizon-marched sun shadow: a closed-form occlusion trace of the detail
// heightfield toward the sun (replaces the retired statistical
// detail_micro_shadow). The shadow field = grain octave 0 + both pools
// under their existing windows (sub-texel grain octaves are speckle-scale),
// value-only via cloud_vnoise3 (never the grad path); lt = the sun's
// tangent-plane projection l − n(n·l), UNNORMALIZED (|lt| is the grazing
// measure); the sun ray rises (n·l)/(|lt|·HT) field-units per q-unit and
// taps that clear the conservative HMAX bound exit early (the clouds
// interval-skip shape). Soft contact exp(−K·occ): K → inf is a binary hit
// test, which aliases at 1 spp; the softness doubles as the 2°-sun
// penumbra. Exact 1.0 when nothing occludes / windows closed / zenith
// azimuth / sub-horizon sun. HT is INCIDENCE-ADAPTIVE (round 6):
// lerp(LO, HI, saturate(ndl)) — the artistic HI applies where the natural
// response is weakest (noon tops) and fades to the near-coherent LO at
// grazing, where rise ∝ ndl already makes shadows maximal (the
// overdone-sides fix). Term-for-term CPU mirror; zero rng.
#define DETAIL_SHADOW_HT_LO 1.5
#define DETAIL_SHADOW_HT_HI 6.0
#define DETAIL_SHADOW_K  5.0
// The march's HMAX bound is computed in detail_sun_shadow (it scales with
// the strength knobs); the retired constant was 1.03 = 0.18 + 0.5 + 0.35.
float detail_shadow_h(float3 q3, float dlod) {
    float hh = 0.0;
    // Grain rides DETAIL_STR, pools DETAIL_AO_STR — the same terrain the
    // surface shades with stays the terrain that shadows it (shade.rs twin).
    float w0 = saturate(-dlod);
    if (w0 > 0.0) {
        float v = cloud_vnoise3(q3, 40u);
        hh += DETAIL_AMP * DETAIL_STR * w0 * (2.0 * v - 1.0);
    }
    float wk = saturate(3.0 - dlod);
    if (wk > 0.0) {
        float v = cloud_vnoise3(q3 / 8.0, 43u);
        hh += (0.5 * DETAIL_AO_STR) * wk * (2.0 * v - 1.0);
    }
    wk = saturate(2.0 - dlod);
    if (wk > 0.0) {
        float v = cloud_vnoise3(q3 / 4.0, 44u);
        hh += (0.35 * DETAIL_AO_STR) * wk * (2.0 * v - 1.0);
    }
    return hh;
}
float detail_sun_shadow(float3 q3, float dlod, float3 lt, float ndl) {
    float ltl = length(lt);
    if (ltl < 1e-4 || ndl <= 0.0) return 1.0;
    float3 dir = lt / ltl;
    float ht = DETAIL_SHADOW_HT_LO + (DETAIL_SHADOW_HT_HI - DETAIL_SHADOW_HT_LO) * saturate(ndl);
    float rise = ndl / (ltl * ht);
    float h0 = detail_shadow_h(q3, dlod);
    // The early-exit bound scales WITH the strength knobs (shade.rs twin —
    // left-assoc so 1.0/1.0 reproduces DETAIL_SHADOW_HMAX's chain bitwise).
    float hmax = DETAIL_AMP * DETAIL_STR + 0.5 * DETAIL_AO_STR + 0.35 * DETAIL_AO_STR;
    float occ = 0.0;
    static const float taps[8] = { 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 14.0, 20.0 };
    [unroll] for (uint i = 0u; i < 8u; ++i) {
        float ray_h = h0 + taps[i] * rise;
        if (ray_h > hmax) break;
        occ = max(occ, detail_shadow_h(q3 + dir * taps[i], dlod) - ray_h);
    }
    return occ > 0.0 ? exp(-DETAIL_SHADOW_K * occ) : 1.0;
}
// shade.rs::AMB_BUMP_K / amb_irradiance — ambient bump-response
// amplification (FLAG_AMB_BUMP; the HL2/bent-normal dominant-direction
// class): the order-2 irradiance is a cosine convolution, too smooth to
// show texel relief at any tilt, so the sampled/SH ambient amplifies the
// deviation response irr(n) + K·(irr(n_s) − irr(n)), clamped ≥ 0. n_s == n
// (flat-shaded geometry) and flag-off return the plain expression verbatim.
// The amplified delta is CAPPED at ±AMB_BUMP_CAP of the base by a SCALAR
// rescale (hue-preserving; round 6): the SH derivative is maximal when n ⊥
// the dome's dominant direction — the sides — so the ×K tuned for tops
// overdrives there. Under-cap pixels return the uncapped formula bitwise.
// CAP 0.5 -> 0.25 (round 6b): a noon block side gets ~no direct sun, so a
// ±50% cap on its ambient WAS a ±50% swing of its total brightness.
#define AMB_BUMP_K 6.0
#define AMB_BUMP_CAP 0.25
float3 amb_irradiance(float3 n, float3 n_s) {
    if (all(n_s == n) || !(flags & FLAG_AMB_BUMP)) return sh_irradiance(n_s);
    float3 base = sh_irradiance(n);
    float3 d = (sh_irradiance(n_s) - base) * AMB_BUMP_K;
    float m = max(abs(d.x), max(abs(d.y), abs(d.z)));
    float lim = AMB_BUMP_CAP * max(base.x, max(base.y, base.z));
    if (m > lim) d *= lim / m;
    return max(base + d, 0.0);
}
void detail_field(float3 q3, float dlod, out float f, out float3 g) {
    float3 q = q3;
    // --detail-strength scales the whole ladder (the gradient — and so the
    // micro-bump — scales with it). shade.rs twin.
    float amp = DETAIL_AMP * DETAIL_STR;
    float scl = 1.0;
    f = 1.0;
    g = float3(0.0, 0.0, 0.0);
    [unroll] for (uint k = 0u; k < 3u; ++k) {
        float wk = saturate(-dlod - float(k));
        // A real branch, mirrored with the CPU — it also skips the noise eval.
        if (wk > 0.0) {
            float v;
            float3 gv;
            cloud_vnoise3_vg(q, 40u + k, v, gv);
            f += amp * wk * (2.0 * v - 1.0);
            // Chain rule: octave k samples at q*2^k, so its per-q-unit
            // gradient carries the 2^k (scl), and the (2v - 1) the 2.
            g += gv * (amp * wk * 2.0 * scl);
        }
        q *= 2.0;
        amp *= 0.5;
        scl *= 2.0;
    }
    f = max(f, 0.05);
}

// Tilt the shading normal by the detail field's 3D gradient's TANGENTIAL
// PROJECTION gt = g − n(n·g) — identical to the retired (t, b)-frame form
// (the frame was orthonormal, so t·g.x + b·g.y WAS this projection; the
// winding sign cancels in b⊗b). gt == 0 subsumes both old guards (zero g
// and the degenerate tangent). Zero result / below-horizon fall back to
// base (the ripple_normal shape).
float3 detail_bump(float3 base, float3 n, float3 g) {
    float3 gt = g - n * dot(n, g);
    if (all(gt == float3(0.0, 0.0, 0.0))) return base;
    float3 outn = normalize_or_zero(base - gt * DETAIL_BUMP_K);
    if (all(outn == float3(0.0, 0.0, 0.0)) || dot(outn, n) <= 0.0) return base;
    return outn;
}

// shade.rs::surface_point: interpolated, face-flipped shading normal and the
// eps-offset origin for secondary rays.
//
// THE FLIP IS DECIDED ON THE TRUE FACE NORMAL (shade.rs mirror — read its doc
// for the full argument): on a SMOOTH-shaded mesh the interpolated normal
// crosses the view horizon in a band at every silhouette while the FACE is
// still front-facing, and flipping there aims `n` — which is also the
// eps-offset axis — INTO the solid, so every shadow/AO ray starts inside the
// surface and the pixel resolves to exactly (0,0,0). The face normal is read
// ONLY inside the branch; everywhere else the two criteria agree, so the
// common path is unchanged bit-for-bit. Inside it the face only arbitrates
// when the interpolated normal lies in its hemisphere — a mesh whose winding
// disagrees with its authored normals keeps the old unconditional flip.
void surface_point(float3 ro, float3 rd, HitInfo hit, out float3 p, out float3 n) {
    uint3 idx = uint3(indices[hit.tri * 3u], indices[hit.tri * 3u + 1u], indices[hit.tri * 3u + 2u]);
    float w = 1.0 - hit.u - hit.v;
    n = normalize_or_zero(normals[idx.x] * w + normals[idx.y] * hit.u + normals[idx.z] * hit.v);
    if (all(n == float3(0.0, 0.0, 0.0))) {
        float3 e1 = positions[idx.y] - positions[idx.x];
        float3 e2 = positions[idx.z] - positions[idx.x];
        n = normalize_or_zero(cross(e1, e2));
    }
    if (dot(n, rd) > 0.0) {
        // Genuine backface, or a smooth silhouette's past-horizon band? Ask
        // the face, and only where it is entitled to answer. Both guards keep
        // the old unconditional flip when they fire: nf·n <= 0 (winding
        // disagrees with the authored normals — also the degenerate face's
        // exact 0.0, and NaN) and !(nf·d < 0) (the face really is backfacing).
        float3 e1 = positions[idx.y] - positions[idx.x];
        float3 e2 = positions[idx.z] - positions[idx.x];
        float3 nf = cross(e1, e2);
        if (dot(nf, n) <= 0.0 || !(dot(nf, rd) < 0.0)) n = -n;
    }
    p = ro + rd * hit.t + n * SCENE_EPS;
}

// Ray-cone texture LOD base term — the term-for-term mirror of
// shade.rs::tri_lod_base (change both together):
//   0.5*log2(uv_area/world_area) + log2(cone_w) - log2(max(|n.d|, 0.05))
// Each map completes it with its own dimension term (tex_lod below, the
// Texture::lod_dims mirror). Degenerate UVs/triangles or a zero cone return
// a large negative lod: SampleLevel clamps to level 0 = the exact old
// bilinear (the magnification-compat contract). Pure hit-geometry ALU,
// ZERO rng draws — the same-seed gates rely on that.
float tex_lod_base(uint tri, float n_dot_d, float cone_w) {
    uint3 idx = uint3(indices[tri * 3u], indices[tri * 3u + 1u], indices[tri * 3u + 2u]);
    float3 p0 = positions[idx.x];
    float3 e1 = positions[idx.y] - p0;
    float3 e2 = positions[idx.z] - p0;
    float wa = length(cross(e1, e2)); // 2x area — the 1/2 cancels in the ratio
    float2 t0 = uv_buf[idx.x];
    float2 d1 = uv_buf[idx.y] - t0;
    float2 d2 = uv_buf[idx.z] - t0;
    float ua = abs(d1.x * d2.y - d2.x * d1.y);
    if (!(ua > 0.0) || !(wa > 0.0) || !(cone_w > 0.0)) return -1e30;
    return 0.5 * log2(ua / wa) + log2(cone_w) - log2(max(n_dot_d, 0.05));
}

// Texture::lod_dims mirror: complete the base term for one map's dims.
float tex_lod(uint ti, float lod_base) {
    uint tw, th;
    texs[NonUniformResourceIndex(ti)].GetDimensions(tw, th);
    return lod_base + 0.5 * log2(float(tw * th));
}

// shade.rs::HEMI_CONE_SPREAD mirror — the cone spread hemi-GI bounce hits
// shade with (octant-scale footprint; over-blur is variance reduction).
#define HEMI_CONE_SPREAD 0.25

// shade.rs::tri_uv_basis mirror: (dP/du, dP/dv) from the triangle's positions
// + UVs, derived on the fly (zero storage). Called ONCE per lap by shade_split
// and handed to BOTH consumers — the tangent frame (perturb_normal) and the
// texture footprint (tri_grads_from) — same as on the CPU. Degenerate => false.
bool tri_uv_basis(uint tri, out float3 tu, out float3 tv) {
    uint3 idx = uint3(indices[tri * 3u], indices[tri * 3u + 1u], indices[tri * 3u + 2u]);
    float3 p0 = positions[idx.x];
    float3 e1 = positions[idx.y] - p0;
    float3 e2 = positions[idx.z] - p0;
    float2 t0 = uv_buf[idx.x];
    float2 d1 = uv_buf[idx.y] - t0;
    float2 d2 = uv_buf[idx.z] - t0;
    float det = d1.x * d2.y - d2.x * d1.y;
    tu = 0.0;
    tv = 0.0;
    if (abs(det) < 1e-12) return false;
    tu = (e1 * d2.y - e2 * d1.y) / det;
    tv = (e2 * d1.x - e1 * d2.x) / det;
    return true;
}

// shade.rs::tri_rest_point — the hit's barycentric REST-pose world position
// over positions[], which IS the rest buffer (sway rides TLAS instance
// transforms; vertices never move). The world-space detail field's sample
// point: stable under sway, deliberately NOT ro + t*rd (displaced) and not
// the eps-offset p.
float3 tri_rest_point(uint tri, float u, float v) {
    uint3 idx = uint3(indices[tri * 3u], indices[tri * 3u + 1u], indices[tri * 3u + 2u]);
    return positions[idx.x] * (1.0 - u - v) + positions[idx.y] * u + positions[idx.z] * v;
}

// shade.rs::tri_grads_from mirror (change both together): the ray cone's
// ELLIPTICAL footprint at the hit, as two UV-space gradient vectors, against
// an ALREADY-DERIVED UV basis (the caller derives it once and shares it with
// perturb_normal — the triangle's two dP/d* consumers). The cone is a circle
// of diameter cone_w perpendicular to d; projected along d onto the surface it
// is cone_w across the direction of travel and cone_w / |n.d| along it — the
// stretch tex_lod_base's -log2(max(|n.d|, 0.05)) term can only blur away.
// Normalized-UV units, so one footprint serves every map on the material
// (SampleGrad scales by each texture's own dims — the tex_lod_base / tex_lod
// split in gradient form). Pure hit geometry, ZERO rng draws.
bool tri_grads_from(float3 tu, float3 tv, float3 n, float3 d, float cone_w,
                    out float2 gu, out float2 gv) {
    gu = 0.0;
    gv = 0.0;
    if (!(cone_w > 0.0)) return false;
    // RELATIVE degeneracy guard (shade.rs mirror): den/|tu x tv| is the
    // cosine between the shading normal and the basis plane's normal. At
    // silhouettes the interpolated n tilts nearly into the basis plane and
    // an absolute den threshold let 1/den blow the gradients up — and
    // SampleGrad with huge/non-finite gradients is UB (black). Written
    // !(x >= k) so NaN inputs reject too; false = the iso-lod fallback.
    // The a < 3e38 arm: tri_uv_basis admits UV dets down to 1e-12, so |tu|
    // reaches ~1e12 on atlas meshes and |tu x tv|^2 overflows to Inf, which
    // would make the cosine test reject UNCONDITIONALLY (aniso silently off
    // there). At that scale the cosine is unmeasurable — hand it to the
    // finiteness backstop below instead. NaN fails the < and takes the same
    // route. Plain magnitude compare, NOT isfinite (DXC folds that away).
    float3 axn = cross(tu, tv);
    float den = dot(axn, n);
    float a = length(axn);
    if (a < 3.0e38 && !(abs(den) >= 1e-3 * a)) return false;
    float n_d = dot(d, n);
    float3 across = cross(n, d);
    float3 a_dir, b_dir;
    if (dot(across, across) > 1e-12) {
        a_dir = normalize(across);              // across the direction of travel
        b_dir = normalize_or_zero(d - n * n_d); // along it (the stretched axis)
    } else {
        // Normal incidence: the footprint is a circle — any in-plane
        // orthonormal pair spans it.
        a_dir = normalize_or_zero(tu - n * dot(n, tu));
        if (all(a_dir == float3(0.0, 0.0, 0.0))) return false;
        b_dir = cross(n, a_dir);
    }
    float3 w_min = a_dir * cone_w;
    float3 w_maj = b_dir * (cone_w / max(abs(n_d), 0.05));
    // Cramer against the UV basis: w = du*tu + dv*tv for in-plane w.
    gu = float2(dot(cross(w_maj, tv), n), dot(cross(tu, w_maj), n)) / den;
    gv = float2(dot(cross(w_min, tv), n), dot(cross(tu, w_min), n)) / den;
    // Overflow backstop (shade.rs mirror): a huge-but-accepted basis can
    // still overflow the numerator products. The exponent bit test, NOT
    // isfinite() — DXC folds isfinite away without strict IEEE (the
    // quin.hlsl finite1 lesson). Non-finite gradients must never reach
    // SampleGrad.
    uint4 ge = uint4(asuint(gu), asuint(gv)) & 0x7f800000u;
    if (any(ge == 0x7f800000u)) {
        gu = 0.0;
        gv = 0.0;
        return false;
    }
    return true;
}

// How one hit's textures get filtered — the shade.rs::TexFilter mirror, built
// once per lap and shared by all five maps on the material.
struct TexFilt {
    float  lod_base; // isotropic: the per-hit ray-cone lod term (-1e30 = mip 0)
    float2 gu;       // anisotropic: the footprint's two axes, in UV units
    float2 gv;
    bool   aniso;
};

// The single texture-sampling choke point of the GPU shader (shade.rs's
// TexFilter::sample). SampleGrad is what makes samp_aniso mean anything —
// SampleLevel gives the TMU no gradients to be anisotropic about.
float3 tex_sample(uint ti, float2 uv, TexFilt f) {
    if (f.aniso) {
        return texs[NonUniformResourceIndex(ti)].SampleGrad(samp_aniso, uv, f.gu, f.gv).rgb;
    }
    return texs[NonUniformResourceIndex(ti)].SampleLevel(samp_lin, uv, tex_lod(ti, f.lod_base)).rgb;
}

// The per-lap filter for a hit: the elliptical footprint when the session and
// this ray both want it and the triangle's UV basis is sound, else the
// isotropic ray-cone lod (coarser, never wrong). `aniso` is the RAY's
// decision (hemi bounce laps pass false); FLAG_ANISO is the session's.
// `has_basis`/(tu, tv) is the caller's ONE tri_uv_basis derivation, shared
// with perturb_normal — mirrors the shade.rs factoring.
TexFilt tex_filter(uint tri, float3 n, float3 rd, float cone_w, bool mat_tex, bool aniso,
                   bool has_basis, float3 tu, float3 tv) {
    TexFilt f;
    f.lod_base = -1e30;
    f.gu = 0.0;
    f.gv = 0.0;
    f.aniso = false;
    if (!mat_tex) return f;
    if (aniso && (flags & FLAG_ANISO) && has_basis &&
        tri_grads_from(tu, tv, n, rd, cone_w, f.gu, f.gv)) {
        f.aniso = true;
        return f;
    }
    f.lod_base = tex_lod_base(tri, abs(dot(n, rd)), cone_w);
    return f;
}

// shade.rs::perturb_normal — tangent-space normal mapping with the tangent
// derived on the fly from the triangle's positions + UVs (zero storage),
// Gram-Schmidt vs n, bitangent handedness from the UV winding. Degenerate
// UVs/tangent or a past-horizon perturbation degrade to n. The green channel
// is negated (shade.rs::NORMAL_MAP_Y_SIGN): the loader V-flips at load, so
// OpenGL-convention (+Y up) maps point against our +v rows.
// (t_raw, b_raw) is the caller's ONE tri_uv_basis derivation, shared with the
// texture footprint (tri_grads_from) — a degenerate basis never reaches here.
#define NORMAL_MAP_Y_SIGN (-1.0)
float3 perturb_normal(float3 n, Mat mat, float2 uv, TexFilt filt,
                      float3 t_raw, float3 b_raw) {
    float3 t = normalize_or_zero(t_raw - n * dot(n, t_raw));
    if (all(t == float3(0.0, 0.0, 0.0))) return n;
    float3 bx = cross(n, t);
    float3 b = bx * (dot(bx, b_raw) >= 0.0 ? 1.0 : -1.0);
    float3 s = tex_sample(mat.normal_tex, uv, filt);
    float3 tn = float3((s.x * 2.0 - 1.0) * mat.normal_scale,
                       (s.y * 2.0 - 1.0) * mat.normal_scale * NORMAL_MAP_Y_SIGN,
                       max(s.z * 2.0 - 1.0, 0.05));
    float3 outn = normalize_or_zero(t * tn.x + b * tn.y + n * tn.z);
    if (all(outn == float3(0.0, 0.0, 0.0)) || dot(outn, n) <= 0.0) return n;
    return outn;
}

// The ripple FIELD itself (`ripple_grad` + its constants) lives in
// ripple.hlsli, pasted ahead of this file — it has a third consumer, the fxc
// frame-generation guide kernel. Only the shading glue stays here.
//
// Tilt `base` by the ripple slope (× amp), unit on the +n side; a
// degenerate/below-horizon result falls back to base. `n` is the geometric
// normal (horizon guard + projection axis).
float3 ripple_normal(float3 base, float3 n, float3 p, float t, float amp, float diag) {
    if (amp == 0.0) return base; // off state: no re-normalize (unit base drift)
    float2 g = ripple_grad(p, t, diag) * amp;
    float3 g3 = float3(g.x, 0.0, g.y);
    float3 gt = g3 - n * dot(g3, n);
    float3 outn = normalize_or_zero(base - gt);
    if (all(outn == float3(0.0, 0.0, 0.0)) || dot(outn, n) <= 0.0) return base;
    return outn;
}

// shade.rs's `snell` closure — one Snell/Fresnel evaluation over axis `ns`.
// Refraction and the reflected-fraction Fresnel ride `ns`; the eps offsets
// ride the GEOMETRIC n. Returns (tdir, torig, ttw, is_tir) via out params.
void glass_snell(float3 rd, float3 v, float3 ns, float3 n, float3 hit_p,
                 float eta, float transmission, out float3 tdir, out float3 torig,
                 out float ttw, out bool is_tir) {
    float cos_i = max(dot(v, ns), 1e-4);
    float k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if (k >= 0.0) {
        // Exact unpolarized dielectric Fresnel (not Schlick — it must reach 1
        // as k -> 0 or the TIR handoff pops).
        float cos_t = sqrt(k);
        float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
        float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
        float fr = 0.5 * (rs * rs + rp * rp);
        tdir = normalize(rd * eta + ns * (eta * cos_i - cos_t));
        torig = hit_p - n * SCENE_EPS;
        ttw = transmission * (1.0 - fr);
        is_tir = false;
    } else {
        tdir = rd + ns * (2.0 * cos_i);
        torig = hit_p + n * SCENE_EPS;
        ttw = transmission;
        is_tir = true;
    }
}

// --- The shade() port ----------------------------------------------------------

// --dxr-sbt rung-2 SPECIALIZATION SEAMS (the SHADE_MAT_* macros): every
// material-arm guard below routes through ONE overridable function-like
// macro whose DEFAULT is the verbatim data expression — an unarmed compile
// is therefore semantically identical in all five pasting units (the
// wavefront/reference/hemi kernels never override these), and a
// class-specialized DXR library (shadeclass::strip_defines) force-folds
// exactly its provably-dead arms to (false)/(0.0). Soundness is the STRIPS
// table's membership contract: a class only strips arms whose guards are
// data-false for every member, so the fold removes code that never ran —
// same image, fewer registers. Guard and dependents fold off the SAME
// macro, lockstep by construction; float-valued seams return the field
// (force (0.0)), bool-valued ones the predicate (force (false)).
#ifndef SHADE_MAT_TEXKIND
#define SHADE_MAT_TEXKIND(m) ((m).kind == MAT_TEXTURED)
#endif
#ifndef SHADE_MAT_MARBLE
#define SHADE_MAT_MARBLE(m) ((m).kind == MAT_MARBLE)
#endif
#ifndef SHADE_MAT_NORMAL
#define SHADE_MAT_NORMAL(m) ((m).normal_tex != TEX_NONE)
#endif
#ifndef SHADE_MAT_ROUGHTEX
#define SHADE_MAT_ROUGHTEX(m) ((m).rough_tex != TEX_NONE)
#endif
#ifndef SHADE_MAT_METALTEX
#define SHADE_MAT_METALTEX(m) ((m).metal_tex != TEX_NONE)
#endif
#ifndef SHADE_MAT_EMISTEX
#define SHADE_MAT_EMISTEX(m) ((m).emissive_tex != TEX_NONE)
#endif
#ifndef SHADE_MAT_RIPPLE
#define SHADE_MAT_RIPPLE(m) ((m).ripple_amp)
#endif
#ifndef SHADE_MAT_SHEEN
#define SHADE_MAT_SHEEN(m) ((m).sheen)
#endif
#ifndef SHADE_MAT_TRANSLUCENCY
#define SHADE_MAT_TRANSLUCENCY(m) ((m).translucency)
#endif
#ifndef SHADE_MAT_TRANSMISSION
#define SHADE_MAT_TRANSMISSION(m) ((m).transmission)
#endif
#ifndef SHADE_MAT_ANISO
#define SHADE_MAT_ANISO(m) ((m).anisotropy)
#endif
// THE MIS-COUPLED SEAM — the one-sky invariant's shipped-bug class: this
// macro feeds BOTH the VNDF bounce block AND the direct loop's MIS reweight
// (`refl_ray` is the one gate both read). Stripping the bounce without
// forcing w_l = 1.0 would delete the light-sampled specular highlight;
// routing both through ONE macro makes that split impossible. Classes may
// strip it only when their predicate forces metallic <= 0.04 AND
// roughness >= 0.45 (the gate's own expression) — and the rng pair draws
// INSIDE the gate on both CPU and GPU, so the strip is same-seed
// stream-identical, no burn.
#ifndef SHADE_MAT_REFL
#define SHADE_MAT_REFL(m) ((m).metallic > 0.04 || (m).roughness < 0.45)
#endif
#ifndef SHADE_MAT_EMISSIVE
#define SHADE_MAT_EMISSIVE(m) (any((m).emissive != 0.0) || SHADE_MAT_EMISTEX(m))
#endif

// Whitted shading of a committed primary hit, reflection bounce included.
// (ro, rd) is the ray that produced `hit`; rd unit-length. RNG draw order
// matches shade.rs exactly per lap: shadow pairs, AO pairs, reflection pair.
// ONE cross-pipeline exception (RTGI): shade_full's bounce draws land AFTER
// this DFS returns, where the CPU arm draws them at its ambient-tier
// position — permitted because CPU and GPU are different streams by design
// (sequences differ, means match); within each pipeline the order is fixed.
// Quality is explicit (n_shadow / n_ao / refl) so the hemi leaf pass can run
// the BOUNCE_Q policy (1/0/false) through the same code.
// `split_ambient` (hemi mode, lap 0 only): the primary ambient term is
// omitted — no AO draws — and amb_w = kd*(1-transmission), (amb_o, amb_n) =
// the surface point are returned for the hemi wavefront; every non-root lap
// shades its own sampled ambient (the CPU forces fb OFF past depth 0).
// `prim` is the PRIMARY surface capture (shade.rs::PrimarySurface, lap 0
// only) for the G-buffer pack — a pure copy of already-computed values, so
// exposing it adds NO rng draws (the same-seed A/B gates rely on that).
// `aniso`: may THIS ray's footprint be resolved anisotropically (the CPU's
// Cone::aniso > 1)? Primary/reflection/glass laps yes, hemi-GI bounce laps no
// (their cone is octant-coarse by design). Gated by FLAG_ANISO on top.
// `cam_lights`: may the PRIMARY surface (lap 0) take camera-path point/
// cluster light NEE — fireflies AND emissive cluster lights (the CPU's
// `ff: Option<&Fireflies>` / `el: Option<&EmissiveLights>` pair — Some only
// on the camera path)? Camera laps yes, hemi bounce laps no — fireflies
// never light bounce surfaces (the gather exclusion), and under GI the
// gather IS the emissive transport. FLAG_FIREFLIES / FLAG_EMISSIVE gate on
// top per family, so day / emissive-free kernels are bit-identical whatever
// the call site passes.
// `el_mask`: which emissive cluster lights (bit ei = light ei) this pixel's
// NEE may sample — the wavefront leaf kernel's per-TILE conservative cull
// (trace_common::el_tile_culled; the CPU's cull_tile twin), full ~0 from
// every tile-less path (reference, DXR, hemi). EXACT, not approximate: a
// culled light would have failed every pixel's own d2 >= r_infl2 test, so
// any conservative mask shades bit-identically — which is what keeps the
// same-seed wavefront(culled)-vs-reference(full) A/B at 0.00e0.
// DXR_SBT_RECURSE (--dxr-sbt 3, the DXR library only — no other unit defines
// it): the flattened DFS below becomes a TRUE recursion. Both continuation
// branches fire rt_dxr.hlsli's trace_shade — a TraceRay whose SBT arithmetic
// lands the child in ITS OWN class's closest-hit — and consume the returned
// radiance, so the lap loop degenerates to one iteration (next_set is never
// set; the stash is dead code the compiler folds) and the hardware ray stack
// replaces it. The invocation's own recursion depth arrives as `depth0`
// (payload-carried), so the depth-0-only gates (refl_ray, the TRANS_MAX_DEPTH
// chain bound) keep the CPU's exact policy per surface.
float3 shade_split(float3 ro, float3 rd, HitInfo hit, inout uint rng,
                   uint n_shadow, uint n_ao, bool refl, bool split_ambient,
                   float cone_w0, float cone_spread, bool aniso, bool cam_lights,
                   uint2 el_mask,
                   out float3 amb_w, out float3 amb_o, out float3 amb_n,
                   out PrimSurf prim
#ifdef DXR_SBT_RECURSE
                   , uint depth0
#endif
) {
    amb_w = 0.0;
    amb_o = 0.0;
    amb_n = 0.0;
    prim = (PrimSurf)0;
    float3 total = 0.0;
    float3 tput = 1.0;
    // Ray-cone origin width for the CURRENT lap's ray — advances to the
    // hit's width at every continuation (the CPU recursion's Cone{w0}).
    float cone_o = cone_w0;
    // Flattened-DFS state: `depth` is the CPU recursion depth of the surface
    // this lap shades; the one stash slot holds the root's transmission
    // child while the reflection subtree runs (only the root can have two
    // children — reflection is depth-0-only). Max laps = 1 root +
    // TRANS_MAX_DEPTH reflection-branch nodes + TRANS_MAX_DEPTH root-chain
    // nodes = 9, the CPU's exact DFS order.
#ifdef DXR_SBT_RECURSE
    uint depth = depth0;
#else
    uint depth = 0u;
#endif
    bool have_stash = false;
    float3 st_o = 0.0, st_d = 0.0, st_tput = 0.0;
    HitInfo st_hit = (HitInfo)0;
    uint st_depth = 0u;
    float st_cone = 0.0;
    // Is THIS lap inside the root's reflection subtree? Everything such a lap
    // adds to `total` is part of the CPU's `tput * rcol` — i.e. the
    // INDIRECT_SPECULAR signal — and the subtree is more than one lap: a
    // reflected ray that hits glass continues its own transmission chain.
    // The stash only ever holds the ROOT's transmission child, which is not.
    bool in_refl = false;
    bool st_in_refl = false;
    [loop] for (uint lap = 0u; lap < 1u + 2u * TRANS_MAX_DEPTH; ++lap) {
        float3 p, n;
        surface_point(ro, rd, hit, p, n);
        Mat mat = materials[tri_mat[hit.tri]];
        // Cone width at this hit + the per-hit lod base term shared by every
        // map on this material (shade.rs's cone_w / lod_base pair).
        float cone_w = cone_o + hit.t * cone_spread;
        bool mat_tex = SHADE_MAT_TEXKIND(mat) || SHADE_MAT_NORMAL(mat) ||
                       SHADE_MAT_ROUGHTEX(mat) || SHADE_MAT_METALTEX(mat) ||
                       SHADE_MAT_EMISTEX(mat);
        // dP/du, dP/dv — derived at most ONCE per lap and shared by its two
        // consumers, the anisotropic footprint and the tangent frame. Neither
        // is reached without a texture, and the isotropic path with no normal
        // map needs no basis at all (the shade.rs `uv_basis` factoring).
        float3 tu = 0.0, tv = 0.0;
        bool has_basis = false;
        if (mat_tex && (aniso || SHADE_MAT_NORMAL(mat))) {
            has_basis = tri_uv_basis(hit.tri, tu, tv);
        }
        TexFilt filt = tex_filter(hit.tri, n, rd, cone_w, mat_tex, aniso, has_basis, tu, tv);
        // Albedo: the shade.rs match. A texture REPLACES Kd (exporters set
        // Kd = 1 alongside map_Kd); hardware bilinear + the SRGB SRV format
        // reproduce texture.rs::sample_bilinear (decode per texel, then
        // filter — precision-level differences only, absorbed by the
        // statistical CPU-vs-GPU gates). The detail block after the match
        // adds the Unreal-1 field for EVERY albedo source (shade.rs's hook,
        // mirrored): `dgr` carries its per-q-unit 3D gradient to the
        // micro-bump below, `ddo` the fired flag — both dead when
        // FLAG_DETAIL is off or the window is closed. Under shadeclass
        // stripping only the textured WINDOW arm folds away with
        // SHADE_MAT_TEXKIND; the untextured arm legitimately survives (a
        // lambert/gloss record's materials ARE untextured).
        float3 dgr = float3(0.0, 0.0, 0.0);
        bool ddo = false;
        // The field's signed height (value - 1, mean 0) for the cavity AO
        // below; 0.0 = never fired, the structural off state (the cavity
        // sites branch on `< 0`).
        float dh = 0.0;
        // Horizon-march capture (shade.rs::detail_march): live while the AO
        // band is open — the marched sun shadow below re-samples the field
        // along the sun's tangent direction from the SAME q3.
        float3 mq3 = float3(0.0, 0.0, 0.0);
        float mdl = 0.0;
        bool dmarch = false;
        // Spec-AA: the detail field's discarded-octave slope variance
        // (0.0 = nothing to transfer), folded into rough_eff below.
        float s2_detail = 0.0;
        float3 albedo = mat.albedo;
        if (SHADE_MAT_MARBLE(mat)) {
            albedo = marble(ro + rd * hit.t, mat.scale);
        } else if (SHADE_MAT_TEXKIND(mat)) {
            float2 auv = tri_uv(hit.tri, hit.u, hit.v);
            albedo = tex_sample(mat.tex, auv, filt);
        }
        // The Unreal-1 detail block (shade.rs's site, mirrored): the field's
        // domain is the rest-pose position over the per-material scale, so
        // it needs no UVs — untextured materials ride the synthetic scale
        // (derive_detail_scales' untextured arm via Mat.detail_scale) and
        // the world-space window below. Transmissive materials EXCLUDED
        // (the visor/water finding): graining the transmission tint mottles
        // glass.
        if ((flags & FLAG_DETAIL) && SHADE_MAT_TRANSMISSION(mat) == 0.0) {
            // The PER-MATERIAL texel scale (Scene::detail_scales via
            // Mat.detail_scale — never per-face, which seams on
            // greedy-meshed atlases). s == 0 closes the window — structural
            // off.
            float s = mat.detail_scale;
            float dlod;
            if (SHADE_MAT_TEXKIND(mat)) {
                // The albedo texture's COMPLETED lod: the iso arm's base
                // rides in filt (free); the aniso arm keys off the MINOR
                // axis its gradients already carry — the isotropic recompute
                // carried the major axis's -log2|n·d| view-tilt stretch,
                // which closed the window on grazing-viewed faces whose
                // albedo SampleGrad kept sharp (the Minecraft-tops finding).
                float lb = filt.aniso ? detail_aniso_base(filt.gu, filt.gv)
                                      : filt.lod_base;
                uint tw, th;
                texs[NonUniformResourceIndex(mat.tex)].GetDimensions(tw, th);
                dlod = lb + 0.5 * log2(float(tw * th)); // Texture::lod_dims
            } else {
                // Untextured: the same window measured directly in the
                // field's own q-domain — log2 of the cone footprint in
                // texel-equivalents. cone_w is the footprint's MINOR axis,
                // matching the textured aniso convention above (grazing
                // aliasing accepted at the same price); exact 0 at
                // cone_w == s. NOT filt's untextured -1e30 base, which
                // would saturate every octave window wide open. s == 0
                // parks the window closed (the bitwise off arm).
                dlod = (s > 0.0) ? log2(cone_w / s) : 1e30;
            }
            // Spec-AA transfer capture (shade.rs's site, mirrored):
            // deliberately OUTSIDE the window gate below — at dlod >=
            // DETAIL_AO_RANGE (every window shut, both arms dead) the
            // transfer is at its MAXIMUM, the "distant surface must go
            // matte" regime. s == 0 keeps the exact-0.0 off state; a
            // fully-open window contributes an IEEE-exact 0.0, so magnified
            // pixels are bit-identical through the fold's branch.
            if ((flags & FLAG_SPEC_AA) && s > 0.0) {
                s2_detail = detail_var(dlod);
            }
            bool ao_band = dlod < DETAIL_AO_RANGE && (flags & FLAG_DETAIL_AO);
            if ((dlod < 0.0 || ao_band) && s > 0.0) {
                // World-space domain: q3 = rest position over s.
                float3 q3 = tri_rest_point(hit.tri, hit.u, hit.v) / s;
                if (dlod < 0.0) {
                    float df;
                    detail_field(q3, dlod, df, dgr);
                    albedo *= df;
                    ddo = true;
                    dh = df - 1.0;
                }
                // The AO/relief coarse octaves fire far past the grain
                // (8/4-texel-equivalent cells resolve until dlod = 3/2) —
                // mid-distance pools AND relief rims: their gradient joins
                // the micro-bump (× DETAIL_AO_BUMP_K), so pool-scale relief
                // is lit directionally out where the grain has faded
                // (shade.rs's site, mirrored). Gated on the AO lever so the
                // off arm never pays the evals and ddo/dmarch stay
                // false-shaped.
                if (ao_band) {
                    float hp;
                    float3 gp;
                    detail_ao_field(q3, dlod, hp, gp);
                    dh += hp;
                    if (any(gp != float3(0.0, 0.0, 0.0))) {
                        dgr += gp * DETAIL_AO_BUMP_K;
                        ddo = true;
                    }
                    mq3 = q3;
                    mdl = dlod;
                    dmarch = true;
                }
            }
        }
        // Map-driven material terms — the shade.rs block, ZERO rng draws
        // (materials with every map at TEX_NONE run bit-identically to the
        // pre-map kernel; the same-seed wavefront-vs-reference gates rely on
        // that). Linear maps ride UNORM SRVs (no sRGB decode).
        float2 map_uv = float2(0.0, 0.0);
        if (SHADE_MAT_NORMAL(mat) || SHADE_MAT_ROUGHTEX(mat) ||
            SHADE_MAT_METALTEX(mat) || SHADE_MAT_EMISTEX(mat)) {
            map_uv = tri_uv(hit.tri, hit.u, hit.v);
        }
        float rough_eff = mat.roughness;
        float metal_eff = mat.metallic;
        if (SHADE_MAT_ROUGHTEX(mat)) {
            rough_eff = clamp(rough_eff * tex_sample(mat.rough_tex, map_uv, filt).g, 0.02, 1.0);
        }
        if (SHADE_MAT_METALTEX(mat)) {
            metal_eff = clamp(metal_eff * tex_sample(mat.metal_tex, map_uv, filt).b, 0.0, 1.0);
        }
        // Shading normal n_s (n_s == n when unmapped): the BRDF frame, N·L,
        // and the guide use it; n keeps the ray offsets, the translucency
        // back ray, the hemi handoff, and the glass chain — the shade.rs
        // n_g/n_s split.
        // A degenerate UV basis is exactly the case perturb_normal used to
        // bail on internally — no tangent frame, so n_s stays n.
        float3 n_s = n;
        if (SHADE_MAT_NORMAL(mat) && has_basis) {
            n_s = perturb_normal(n, mat, map_uv, filt, tu, tv);
        }
        // Unreal-1 detail micro-bump (shade.rs's hook, mirrored): the SAME
        // field that modulated the albedo tilts n_s by its gradient's
        // tangential projection, composed ON the normal map and UNDER the
        // ripple. `ddo` fires only on magnified textured hits with a sound
        // basis, so far pixels never reach this (no tangent frame needed —
        // the projection is frame-free). Damped by the PER-PIXEL roughness
        // (detail_bump_weight — a tight specular lobe frosts under normal
        // scatter: the visor keeps its mirror, the shell its grain).
        // Pre-detail shading normal, live iff detail_bump ran (dcap): the
        // direct loop's DETAIL_NDL_CAP clamps the sun's N·L relative to it.
        // Captured BEFORE the ripple — sound because ripple and detail are
        // structurally disjoint (water is transmissive; transmissive skips
        // the detail field). shade.rs::n_pre.
        float3 n_pd = n_s;
        bool dcap = false;
        if (ddo) {
            float bw = detail_bump_weight(rough_eff);
            if (bw > 0.0) {
                n_pd = n_s;
                n_s = detail_bump(n_s, n, dgr * bw);
                dcap = true;
            }
        }
        // Water ripples tilt the SHADING normal on the shared cloud clock
        // (pure ALU, no rng). Composes on the normal map; geometric n
        // untouched. Off (ripple_amp 0) leaves n_s exactly as selected.
        if (SHADE_MAT_RIPPLE(mat) > 0.0) {
            n_s = ripple_normal(n_s, n, ro + rd * hit.t, CLOUD_TIME, mat.ripple_amp, SCENE_DIAG);
        }
        // Spec-AA fold (shade.rs's site, mirrored term-for-term): the slope
        // variance the mip/window pipeline resolved AWAY widens the GGX lobe
        // — α′² = α² + 2σ², so detail stays in the rendering equation at
        // every distance. Both sources are exactly 0.0 wherever nothing was
        // resolved away, and the identity is BY BRANCH (s2 > 0.0):
        //  - the normal map's variance companion, through the SAME filt as
        //    the map itself (level-0 all-zero ⇒ magnification reads exact
        //    0.0); ×normal_scale² — the decode scales slopes linearly;
        //  - the detail transfer ×bw² (applied bw²·wk² + transferred
        //    bw²·(1−wk²) = bw²·full at EVERY distance; a polished surface
        //    is never frosted by detail it would never have shown).
        // ggx_alphas, the sheen inverse-alpha, and the prim guide below all
        // see the widened lobe; detail_bump_weight above read the PRE-fold
        // roughness; the reflection gate keeps its FLAT fields (the rng
        // rule). Zero rng draws.
        if (flags & FLAG_SPEC_AA) {
            float s2 = 0.0;
            if (s2_detail > 0.0) {
                float bw = detail_bump_weight(rough_eff);
                s2 += bw * bw * s2_detail;
            }
            if (SHADE_MAT_NORMAL(mat) && has_basis && mat.normal_var_tex != TEX_NONE) {
                float u = tex_sample(mat.normal_var_tex, map_uv, filt).x;
                s2 += u * u * SPEC_AA_S2_CAP * mat.normal_scale * mat.normal_scale;
            }
            if (s2 > 0.0) {
                rough_eff = min(
                    sqrt(sqrt(rough_eff * rough_eff * rough_eff * rough_eff + 2.0 * s2)), 1.0);
            }
        }
        if (lap == 0u) {
            // shade.rs — spec_t stays 0 unless the lap-0 reflection below
            // traces (hit t / INF on miss). trans/ripple ride the seams so a
            // stripped class folds the export to its (data-identical) zero.
            prim.n = n_s;
            prim.rough = rough_eff;
            prim.albedo = albedo;
            prim.metallic = metal_eff;
            prim.trans = SHADE_MAT_TRANSMISSION(mat);
            prim.ripple_amp = SHADE_MAT_RIPPLE(mat);
            // FLAG_REMOD_EXACT's two sub-term factors — the multiplicative
            // identity until the post-capture block below knows sk/dcav. The
            // blend sites read them whether or not that block ran, so they
            // must be set on EVERY lap-0 path (an unset factor would scale the
            // fold by garbage).
            prim.amb_k = 1.0;
            prim.m_d = 1.0;
        }

        float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metal_eff);
        // 0.157 = Charlie peak directional albedo (shade.rs energy comp) —
        // the kd factor and the direct sheen arm fold off the ONE seam.
        float3 kd = albedo * (1.0 - metal_eff) * (1.0 - 0.157 * SHADE_MAT_SHEEN(mat));

        // Tangent frame on the SHADING normal: anisotropic materials brush
        // circumferentially around world-up; onb covers poles and isotropic.
        float3 t1, t2;
        if (SHADE_MAT_ANISO(mat) > 0.0) {
            float3 t = cross(float3(0.0, 1.0, 0.0), n_s);
            if (dot(t, t) > 1e-8) {
                t1 = normalize(t);
                t2 = cross(n_s, t1);
            } else {
                onb(n_s, t1, t2);
            }
        } else {
            onb(n_s, t1, t2);
        }
        float ax, ay;
        ggx_alphas(rough_eff, SHADE_MAT_ANISO(mat), ax, ay);
        float3 v = -rd;
        // The face-flip guarantees n·v >= 0 for the GEOMETRIC normal; a
        // perturbed n_s can dip below — the grazing guard covers both.
        float3 vl = float3(dot(v, t1), dot(v, t2), max(dot(v, n_s), 1e-4));
        float lambda_v = ggx_lambda(vl, ax, ay);
        // Charlie-sheen inverse alpha, hoisted like the CPU.
        float sheen_inv_a = 1.0 / clamp(rough_eff, 0.07, 1.0);

        // Will the VNDF reflection ray actually be traced? MIS partitions ONE
        // integral between TWO strategies, so the light-sampled specular may
        // only be down-weighted if the BSDF-sampled half really runs — else
        // `w_l` deletes energy nobody else delivers (shade.rs::refl_ray, same
        // expression, same FLAT roughness/metallic). False for the low preset
        // and at every depth > 0.
        bool refl_ray = (depth == 0u && refl && SHADE_MAT_REFL(mat));

        // Direct light: N cone samples toward the SUN DISC at infinity (no
        // position, no 1/d^2 — sky.rs). Lambert (1/pi omitted by convention,
        // absorbed into sun_e = irradiance/pi) + Cook-Torrance GGX with the
        // compensating pi.
        float3 direct_d = 0.0;
        float3 direct_s = 0.0;
        float3 direct_t = 0.0; // thin-surface back transmission (foliage)
        // Sun shadow sample 0's occluder t (the NRD/SIGMA penumbra guide).
        // 0 = no front shadow ray fired — the backlit/ndl<=0 arm — which IS
        // SIGMA's "NoL <= 0 -> 0" convention; miss = INF (CAM_FAR at the pack).
        float sh_t0 = 0.0;
        for (uint si = 0u; si < n_shadow; ++si) {
            // The SAME two draws, in the same order, the rect sampling consumed.
            float su = rng_next(rng);
            float sv = rng_next(rng);
            float3 wi = sun_sample_dir(su, sv);
            // N·L against the SHADING normal; the shadow/translucency ray
            // geometry stays on the geometric n (shade.rs). Detail pixels
            // ride the contrast cap (shade.rs's n_pre clamp — tan(incidence)
            // makes one bump strength overdrive grazing-lit faces 14×; the
            // p <= 0 arm is terminator hygiene).
            float ndl = dot(n_s, wi);
            if (dcap) ndl = detail_ndl_cap(ndl, dot(n_pd, wi));
            if (ndl <= 0.0) {
                // shade.rs translucency arm: a back-lit leaf receives the
                // light through itself; the occlusion ray starts on the
                // transmitted side (p - 2*eps*n = hit - n*eps). The rng
                // draws above already happened — order matches the CPU.
                if (SHADE_MAT_TRANSLUCENCY(mat) > 0.0 && ndl < 0.0) {
                    // transmit_q (shade.rs's tinted-shadows twin): the back
                    // ray carries a tint through glass; ONE when clear, so
                    // `x * 1.0` keeps opaque scenes bitwise.
                    float3 back_vis = ABL_TQ_SHADOW(p - n * (2.0 * SCENE_EPS), wi, 0.0, INF);
                    if (any(back_vis != 0.0)) {
                        direct_t += sun_e.xyz * (-ndl) * back_vis;
                    }
                }
                continue;
            }
            // tmax = INF: the sun is at infinity, so anything along the ray
            // occludes it. transmit_q: the sun ray carries a tint through
            // glass (tinted shadows); the throughput rides `li`, so the
            // GGX/sheen terms inherit it componentwise (shade.rs). Sample 0
            // routes through the _T twin — same query body, so the
            // transmittance value (and every rng draw) is untouched.
            float3 vis_t;
            if (si == 0u) vis_t = ABL_TQ_SHADOW_T(p, wi, 0.0, INF, sh_t0);
            else vis_t = ABL_TQ_SHADOW(p, wi, 0.0, INF);
            if (any(vis_t != 0.0)) {
                float3 li = sun_e.xyz * ndl * vis_t;
                direct_d += li;
                float3 h = normalize_or_zero(wi + v);
                float3 hl = float3(dot(h, t1), dot(h, t2), dot(h, n_s));
                if (hl.z > 0.0) {
                    float dn = ggx_ndf(hl, ax, ay);
                    float3 wil = float3(dot(wi, t1), dot(wi, t2), dot(wi, n_s));
                    float g2 = 1.0 / (1.0 + lambda_v + ggx_lambda(wil, ax, ay));
                    float3 f = schlick(f0, max(dot(wi, h), 0.0));
                    // MIS (balance heuristic) against the VNDF reflection ray,
                    // which can also land in the disc — sky::mis_weight. The
                    // VNDF pdf is G1(v)*D(h)/(4*n.v), G1 = 1/(1 + lambda_v).
                    // ONLY when that ray is traced: with no BSDF strategy in
                    // play there is nothing to share the integral with, and
                    // weighting down would simply lose the highlight.
                    float w_l = 1.0;
                    if (refl_ray) {
                        float p_b = dn / (4.0 * (1.0 + lambda_v) * max(vl.z, 1e-6));
                        w_l = 1.0 - sky_mis_weight(p_b, sky_light_pdf());
                    }
                    direct_s += li * f * (PI * dn * g2 * w_l / (4.0 * vl.z * ndl));
                    if (SHADE_MAT_SHEEN(mat) > 0.0) {
                        // Charlie NDF + Ashikhmin visibility (shade.rs).
                        float sin2 = max(1.0 - hl.z * hl.z, 0.0);
                        float d_c = (2.0 + sheen_inv_a) * pow(sin2, sheen_inv_a * 0.5) / TAU;
                        float v_ash = 1.0 / max(4.0 * (ndl + vl.z - ndl * vl.z), 1e-4);
                        direct_s += li * (PI * SHADE_MAT_SHEEN(mat) * d_c * v_ash);
                    }
                }
            }
        }
        if (n_shadow > 0u) {
            direct_d /= float(n_shadow);
            direct_s /= float(n_shadow);
            direct_t /= float(n_shadow);
        }
        // Cloud shadow (shade.rs, same insertion point): ONE transmittance
        // toward the sun per lap, scaling the whole direct sun contribution —
        // applied BEFORE the lap-0 prim export so the FSR dd/ds signals carry
        // it. The guard is FLAG_CLOUDS (inside the helper) and the unshadowed
        // arm's exact 1.0, so clouds-off stays bit-identical.
        if (flags & FLAG_CLOUDS) {
            float cloud_t = cloud_sun_transmittance(p);
            direct_d *= cloud_t;
            direct_s *= cloud_t;
            direct_t *= cloud_t;
        }
        // Firefly point lights (shade.rs, same insertion point): AFTER the
        // cloud scaling (a firefly is a local light under the slab) and
        // BEFORE the lap-0 prim export (the light rides FSR-RR's denoised
        // dd/ds lobes). Lap 0 only — the CPU recursion passes ff = None.
        // ZERO rng draws: deterministic iteration, one HARD shadow ray per
        // in-radius firefly, finite tmax (stop 2·eps short of the light
        // point). `w_l = 1` on the specular — a point light has zero solid
        // angle, so the VNDF ray can never deliver it; MIS does not apply.
        // ABL_NO_FF_CODE (FR_ABL=noffcode) compiles the block out — the
        // register-pressure probe: a day frame executes none of this either
        // way, so the A/B isolates the allocation the dead arm reserves
        // (trace.rs's abl_defs row carries the contract).
#ifndef ABL_NO_FF_CODE
        if (cam_lights && lap == 0u && (flags & FLAG_FIREFLIES)) {
            float r_inf = FF_RADIUS_K * ff_scale;
            float ff_r2 = r_inf * r_inf;
            float ff_rmin2 = (FF_RMIN_K * ff_scale) * (FF_RMIN_K * ff_scale);
            for (uint fi = 0u; fi < ff_count; ++fi) {
                float3 fto = ff[fi].xyz - p;
                float fd2 = dot(fto, fto);
                // The rejection test — a far firefly's only cost.
                if (fd2 >= ff_r2) continue;
                float fdist = sqrt(fd2);
                float3 fwi = fto / fdist;
                // The detail contrast cap applies to every direct-tier
                // light (shade.rs::capped_ndl, round 6b).
                float fndl = dot(n_s, fwi);
                if (dcap) fndl = detail_ndl_cap(fndl, dot(n_pd, fwi));
                if (fndl <= 0.0) continue;
                // Windowed 1/d² (fireflies.rs::irradiance — exactly 0 at the
                // radius, C¹ there, near-field clamped under the f16 ceiling).
                float fx = 1.0 - fd2 / ff_r2;
                float fe = FF_E_K * ff_scale * ff_scale * ff[fi].w
                           / max(fd2, ff_rmin2) * (fx * fx);
                if (fe <= 0.0) continue;
                float3 fvis = ABL_TQ_SHADOW(p, fwi, 0.0, max(fdist - 2.0 * SCENE_EPS, 0.0));
                if (all(fvis == 0.0))
                    continue;
                float3 fli = FF_COLOR * (fe * fndl) * fvis;
                direct_d += fli;
                float3 fh = normalize_or_zero(fwi + v);
                float3 fhl = float3(dot(fh, t1), dot(fh, t2), dot(fh, n_s));
                if (fhl.z > 0.0) {
                    float fdn = ggx_ndf(fhl, ax, ay);
                    float3 fwil = float3(dot(fwi, t1), dot(fwi, t2), dot(fwi, n_s));
                    float fg2 = 1.0 / (1.0 + lambda_v + ggx_lambda(fwil, ax, ay));
                    float3 ffr = schlick(f0, max(dot(fwi, fh), 0.0));
                    // fli carries ndl; D·G2·F/(4·nv·nl)·nl leaves /(4·nv) —
                    // the sun loop's exact term shape, at full weight.
                    direct_s += fli * ffr * (PI * fdn * fg2 / (4.0 * vl.z * fndl));
                }
            }
        }
#endif
        // Emissive cluster lights (shade.rs, same insertion point): the
        // direct tier's third entry, AFTER the cloud scaling (a lamp under
        // the slab is a local light) and BEFORE the lap-0 prim export (the
        // light rides FSR-RR's denoised dd lobe). Lap 0 only — the CPU
        // recursion passes el = None. ZERO rng draws: deterministic
        // iteration, one HARD shadow ray per in-range light, stopping
        // rc + 2·eps short of the cluster center so the emitter's own bulb
        // geometry cannot occlude its own light. DIFFUSE-ONLY: the traced
        // VNDF ray already delivers the emitter's specular image (the
        // display `total += emis` at every lap), so a w_l = 1 specular term
        // here would double-count it (src/emissive.rs).
        // ABL_NO_EL_CODE (FR_ABL=noelcode) compiles the block out — the
        // register-pressure probe, the noffcode shape; the same tag also
        // emits ABL_NO_EL_CULL so leaf.hlsl's hoist goes with it (the
        // subsumption lives in trace.rs's abl_defs table). el_mask then goes
        // unused here — an unused parameter is legal, signatures stable.
#ifndef ABL_NO_EL_CODE
        if (cam_lights && lap == 0u && (flags & FLAG_EMISSIVE)) {
            for (uint ei = 0u; ei < EL_COUNT; ++ei) {
                // Tile-culled lights: skipped before the CB fetch. Ascending
                // order preserved, so the kept lights' direct_d sums are the
                // full loop's exact subsequence — bit-identical.
                if (((ei < 32u ? el_mask.x >> ei : el_mask.y >> (ei - 32u)) & 1u) == 0u)
                    continue;
                float3 eto = el_a[ei].xyz - p;
                float ed2 = dot(eto, eto);
                float er2 = el_b[ei].w;
                // The rejection test — a far light's only cost.
                if (ed2 >= er2) continue;
                float edist = sqrt(ed2);
                float3 ewi = eto / edist;
                // Same detail contrast cap as the sun/firefly tiers.
                float endl = dot(n_s, ewi);
                if (dcap) endl = detail_ndl_cap(endl, dot(n_pd, ewi));
                if (endl <= 0.0) continue;
                // Windowed disc irradiance (emissive.rs::irradiance — the
                // +rc² denominator is the near-field softening, the window
                // exactly 0 at r_infl, lum clamped under EL_E_MAX).
                float3 ecol = el_b[ei].xyz;
                float elum = dot(ecol, EL_LUM_W);
                if (elum <= 0.0) continue;
                float einv = min(1.0 / (ed2 + el_a[ei].w), EL_E_MAX / elum);
                float ex = 1.0 - ed2 / er2;
                float3 ee = ecol * (einv * ex * ex);
                float3 evis = ABL_TQ_SHADOW(
                    p, ewi, 0.0,
                    max(edist - sqrt(el_a[ei].w) - 2.0 * SCENE_EPS, 0.0));
                if (all(evis == 0.0)) continue;
                direct_d += ee * endl * evis;
            }
        }
#endif
        if (lap == 0u) {
            // shade.rs's post-average lobe export (the FSR-RR signal split
            // demodulates these at the G-buffer write) — pure copies, zero
            // rng draws.
            prim.direct_d = direct_d;
            prim.direct_s = direct_s;
            prim.shadow_t = sh_t0;
        }
        // Detail cavity AO — AFTER the prim captures (shade.rs's site): the
        // FSR signals stay un-cavitied and the deterministic delta rides the
        // exact-remainder residual (texel-crisp under FSR-RR; a reflection
        // LAP's cavity rides the denoised ind_s instead — the documented
        // asymmetry, identity closing either way). Guarded, never `* 1.0`:
        // dh == 0.0 on every non-fired hit and > 0 on peaks, so lever-off /
        // detail-off / dlod >= 0 / peaks leave the expression DAG untouched.
        float dcav = 1.0;
        if (dh < 0.0 && (flags & FLAG_DETAIL_AO)) {
            dcav = detail_cavity(dh);
            direct_s *= dcav;
        }
        // REAL horizon-marched sun shadow on the DIRECT diffuse (shade.rs's
        // site — post-capture, the delta rides the residual; replaces the
        // retired statistical micro_shadow). The march direction is the
        // sun's tangent-plane projection — the same projection the bump
        // applies, so the azimuths agree by construction (no frame to keep
        // in lockstep).
        if (dmarch) {
            float3 slt = sun.xyz - n * dot(n, sun.xyz);
            direct_d *= detail_sun_shadow(mq3, mdl, slt, dot(n, sun.xyz));
        }
        // Diffuse budget split, front vs transmitted (ambient front-only —
        // shade.rs composition). Transmissive glass has (almost) no diffuse
        // response — the transmitted scene replaces it (the chain below);
        // the GGX highlight stays unscaled. kt == 1.0 exactly when
        // transmission == 0 (every procedural/stress material), so the
        // multiply is bit-neutral there.
        float3 diffuse_d = direct_d * (1.0 - SHADE_MAT_TRANSLUCENCY(mat))
                         + direct_t * SHADE_MAT_TRANSLUCENCY(mat);
        float kt = 1.0 - SHADE_MAT_TRANSMISSION(mat);

        // EXACT REMODULATION (FLAG_REMOD_EXACT) — RE-capture the two lobes now
        // that every post-capture factor has been applied, so the bridge's
        // kd/f0 become the EXACT divisors and the denoiser's delta lands at its
        // true physical weight. See trace.rs::FLAG_REMOD_EXACT for what the
        // 1/m-weighted correction was costing (a raw fraction of every bright
        // 1-spp bounce spike, un-denoised).
        //
        // A REASSIGNMENT, not a relocation: the originals at the `lap == 0u`
        // block above stay put, so the flag-clear arm is textually and
        // numerically today's code — the guarded-never-`* 1.0` discipline this
        // file already states for dcav, and what keeps the off arm's expression
        // DAG bit-identical for the M9b/N6 accum gates.
        //
        // WHY `diffuse_d` AND NOT `direct_d * (1 - tl)`: diffuse_d already
        // carries the translucency BACK-RAY term (direct_t * tl), which has no
        // wire lane of its own and is therefore 100% un-denoised stochastic
        // residual today. Capturing the sum pulls it into the denoised channel
        // for free — the one place this fix also removes noise rather than
        // merely reweighting it.
        //
        // The specular lane needs no factor of its own: `ds`'s exact factor is
        // dcav (applied at the cavity block) and `is`'s is exactly 1, so
        // folding dcav into ds leaves BOTH sub-terms remodulating at f0. That
        // lands the free side of the split on `is`, the reflection bounce —
        // the noisy one — which is why m_s stays 1.
        //
        // The DIFFUSE lane cannot do the same, because its two sub-terms
        // disagree (diffuse_d by sk, the ambient/bounce by sk*dcav), so both
        // factors are carried and blended by energy downstream — see PrimSurf.
        // The captured SIGNAL stays the clean demodulated radiance either way.
        if (lap == 0u && (flags & FLAG_REMOD_EXACT)) {
            float sk = 1.0 - 0.157 * SHADE_MAT_SHEEN(mat);
            prim.direct_d = diffuse_d;  // dsun + the tl split; sk rides m_d
            prim.direct_s = direct_s;   // carries dcav (m_s == 1)
            prim.m_d = sk;              // the direct-diffuse sub-term's factor
            prim.amb_k = sk * dcav;     // the ambient/bounce sub-term's factor
        }

        if (split_ambient && lap == 0u) {
            // Hemi mode: the primary ambient is integrated by the hemisphere
            // wavefront (compose applies amb_w * ambient later). No AO draws
            // here — the CPU's sampled loop is skipped the same way. The
            // ambient sits INSIDE the (1 - transmission) factor (shade.rs),
            // so kt folds into the weight.
            float3 c = tput * (kd * kt * diffuse_d + direct_s);
            total += c;
            if (in_refl) prim.ind_s += c;
            amb_w = tput * kd * kt;
            // The cavity darkens the composed ambient through its weight —
            // the CPU's `ambient *= cav` under fb (compose multiplies later).
            if (dh < 0.0 && (flags & FLAG_DETAIL_AO)) amb_w *= dcav;
            // fb_mode 1 (AO) scales the SKY's irradiance by an openness
            // scalar, so the sky term folds into the weight HERE, where n_s is
            // in scope — compose has no normal. fb_mode 2 (GI) integrates the
            // sky itself and must NOT be pre-multiplied by it (that would
            // square the sky). See compose.hlsl. Through amb_irradiance so
            // bumped normals get the amplified sky response (shade.rs's site).
            if (fb_mode == 1u) amb_w *= amb_irradiance(n, n_s);
            amb_o = p;
            amb_n = n;
        } else {
            // Sampled AO modulating the sky's own irradiance (sh.rs::Sh9 —
            // shade.rs's `sky_sh.irradiance(n_s) * ao`). The SHADING normal:
            // ambient is a BRDF-side quantity, while the AO ray directions
            // below keep the GEOMETRIC n (visibility), per the n_g/n_s split.
            float ao = 1.0;
            // AO sample 0's occluder t (the NRD hit-distance guide): 0 = no
            // AO ray at this preset, miss = AO_RADIUS (the query's tmax).
            float ao_t0 = 0.0;
            if (n_ao > 0u) {
                float3 at1, at2;
                onb(n, at1, at2);
                float open = 0.0;
                for (uint ai = 0u; ai < n_ao; ++ai) {
                    float r1 = rng_next(rng);
                    float r2 = rng_next(rng);
                    // Mean-of-components (shade.rs): the AO plane is a
                    // SCALAR, so a glass throughput folds to gray. The true
                    // divide keeps 3.0/3.0 == 1.0 and 0.0/3.0 == 0.0 exact —
                    // opaque scenes accumulate the old integer counts
                    // bit-identically. Sample 0 routes through the _T twin —
                    // same query body, transmittance and rng untouched.
                    float3 dir_ao = cosine_dir(n, at1, at2, r1, r2);
                    float3 tp;
                    if (ai == 0u) tp = ABL_TQ_AO_T(p, dir_ao, 0.0, AO_RADIUS, ao_t0);
                    else tp = ABL_TQ_AO(p, dir_ao, 0.0, AO_RADIUS);
                    open += (tp.x + tp.y + tp.z) / 3.0;
                }
                ao = open / float(n_ao);
            }
            // FSR's AO signal — lap 0 only (later laps compute their own AO
            // for their own ambient; the capture is the PRIMARY surface's).
            // The factor it remodulates by is no longer a constant: it is the
            // sky's own SH irradiance at n_s (feed.hlsl / fsr_composite.hlsl
            // rebuild it from the WIRE normal — see fsr::wire_normal).
            if (lap == 0u) {
                prim.ao = ao;
                prim.ao_t = ao_t0;
            }
            // The ambient term hoisted so the cavity can scale it without
            // touching prim.ao (the un-cavitied FSR signal) — same
            // subexpression, same tree; the off arm's DAG identity is what
            // the same-seed byte gates verify.
            float3 amb_t = amb_irradiance(n, n_s) * ao;
            // FLAG_REMOD_EXACT, the SAMPLED-ambient arm's blend site: the two
            // diffuse sub-terms are known here, so weight their factors before
            // the cavity is applied below (the weight asks "how much of the
            // delta does each own", which is a pre-cavity question — the cavity
            // is what the factor CARRIES, not part of the share).
            //
            // KNOWN-APPROXIMATE, and stated where it bites: shade's ambient is
            // amb_irradiance(n, n_s) — the AMB_BUMP-amplified response — while
            // the bridge rebuilds sh_irradiance(n_s). That ratio is per-channel
            // RGB and needs the GEOMETRIC normal, which is not on the wire, so
            // no scalar lane can carry it. It is structurally ZERO on the live
            // NRD path (the split-ambient arm leaves prim.ao at 0 under RTGI,
            // so the bridge's ao*amb term vanishes); this arm only runs in a
            // non-RTGI NRD session.
            if (lap == 0u && (flags & FLAG_REMOD_EXACT)) {
                prim.m_d = remod_blend(diffuse_d, prim.m_d, amb_t, prim.amb_k);
            }
            if (dh < 0.0 && (flags & FLAG_DETAIL_AO)) amb_t *= dcav;
            float3 c = tput * (kd * kt * (diffuse_d + amb_t) + direct_s);
            total += c;
            if (in_refl) prim.ind_s += c;
        }

        // Emitted radiance — additive per lap, OUTSIDE the kd*kt factor, so
        // emitters appear in reflections and through glass (shade.rs). The
        // guard keeps emissive-free materials bit-identical.
        // THE NEE-KEEP GATE (shade.rs Quality::emissive_display's GPU twin,
        // no new argument): camera laps (cam_lights=true) always add; a
        // BOUNCE invocation (cam_lights=false) adds only when cluster NEE is
        // NOT live this frame — hemi's fb.gi bounce keeps the add because
        // FLAG_EMISSIVE clears at fb_mode==2 (the gather delivers), the RTGI
        // bounce suppresses under a live flag (NEE delivers), and an unarmed
        // frame's clear flag keeps the RTGI bounce as the only delivery.
        if (SHADE_MAT_EMISSIVE(mat) && (cam_lights || (flags & FLAG_EMISSIVE) == 0u)) {
            float3 emis = mat.emissive;
            if (SHADE_MAT_EMISTEX(mat)) {
                emis *= tex_sample(mat.emissive_tex, map_uv, filt);
            }
            total += tput * emis;
            if (in_refl) prim.ind_s += tput * emis;
        }

        // Continuation bookkeeping: at most one of the two branches below
        // becomes `next`; the root's second branch (transmission behind a
        // traced reflection) goes to the stash.
        bool next_set = false;
        float3 nx_o = 0.0, nx_d = 0.0, nx_tput = 0.0;
        HitInfo nx_hit = (HitInfo)0;
        uint nx_depth = 0u;
        // Both possible children originate at THIS hit: their cone starts at
        // this lap's width (the CPU's Cone{w0: cone_w} recursion).
        float nx_cone = cone_w;
        // A continuation inherits this lap's subtree unless the reflection
        // branch below sets it — that branch IS the root of the ind_s subtree.
        bool nx_in_refl = in_refl;

        // (a) One specular bounce — ROOT only (shade.rs gates depth == 0):
        // GGX VNDF importance sample (Heitz 2018), throughput F*G2/G1 <= 1.
        // The two rng draws stay conditional on the same gate as the CPU's —
        // and it is the SAME `refl_ray` the direct loop's MIS weight consulted.
        if (refl_ray) {
            float3 vh = normalize(float3(ax * vl.x, ay * vl.y, vl.z));
            float lensq = vh.x * vh.x + vh.y * vh.y;
            float3 b1 =
                lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) / sqrt(lensq) : float3(1.0, 0.0, 0.0);
            float3 b2 = cross(vh, b1);
            float r = sqrt(rng_next(rng));
            float phi = TAU * rng_next(rng);
            float p1 = r * cos(phi);
            float p2 = r * sin(phi);
            float s = 0.5 * (1.0 + vh.z);
            p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * p2;
            float3 nh = b1 * p1 + b2 * p2 + vh * sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0));
            float3 hl = normalize(float3(ax * nh.x, ay * nh.y, max(nh.z, 1e-6)));
            float3 h = t1 * hl.x + t2 * hl.y + n_s * hl.z;
            float3 rdir = normalize(2.0 * dot(v, h) * h - v);
            // Below-horizon samples are dropped (slight darkening, no upward
            // bias; spec_t stays 0.0 = "no reflection traced") — but the
            // transmission branch below still runs, like the CPU. BOTH
            // horizons: n_s (the lobe's frame) and the geometric n (a
            // perturbed lobe must not fire a ray that re-enters the surface).
            if (dot(rdir, n_s) > 0.0 && dot(rdir, n) > 0.0) {
                float3 rdl = float3(dot(rdir, t1), dot(rdir, t2), dot(rdir, n_s));
                float g2_over_g1 =
                    (1.0 + lambda_v) / (1.0 + lambda_v + ggx_lambda(rdl, ax, ay));
                float3 rtput = tput * schlick(f0, max(dot(v, h), 0.0)) * g2_over_g1;
#ifdef DXR_SBT_RECURSE
                // Recursive class dispatch: the reflected surface shades in
                // ITS OWN class's closest-hit (trace_shade — miss_rec hands
                // back t = INF and NO sky, because the MIS weight below is
                // THIS lobe's and only this invocation can compute it).
                // ind_s = rtput × the child's whole returned radiance — the
                // CPU's literal `tput * rcol`, one multiply at the return
                // (the recursion's own fp association; the statistical
                // suites absorb the reassociation vs the flattened fold).
                float rec_t;
                float3 rcol = trace_shade(p, rdir, rng, depth + 1u, cone_w, rec_t);
                if (!isinf(rec_t)) {
                    prim.spec_t = rec_t; // depth 0 == the captured surface
                    float3 rc = rtput * rcol;
                    total += rc;
                    prim.ind_s += rc;
                } else {
#else
                HitInfo rh;
                if (ABL_TRACE_REFL(p, rdir, 0.0, FLT_MAX, rh)) {
                    prim.spec_t = rh.t; // depth 0 == the captured surface
                    nx_o = p;
                    nx_d = rdir;
                    nx_hit = rh;
                    nx_tput = rtput;
                    nx_depth = 1u;
                    next_set = true;
                    nx_in_refl = true; // everything below this is ind_s
                } else {
#endif
                    prim.spec_t = INF; // reflection missed (shade.rs)
                    // The BSDF-sampling half of the MIS pair. The DOME passes
                    // through un-weighted (only this strategy sees it); the DISC
                    // is weighted, because direct_s is also delivering the sun's
                    // specular. Mirror: w_b ~ 1, this carries the round sun.
                    // Rough: w_b ~ 0, which is what kills the firefly.
                    float3 hl_r = float3(dot(h, t1), dot(h, t2), dot(h, n_s));
                    float p_b_r =
                        ggx_ndf(hl_r, ax, ay) / (4.0 * (1.0 + lambda_v) * max(vl.z, 1e-6));
                    float w_b = sky_mis_weight(p_b_r, sky_light_pdf());
                    // The reflection subtree is just the sky here — so it is
                    // still the ind_s signal (the CPU's tput * rcol). Stars
                    // ride un-weighted (BSDF-only delivery, no MIS partner),
                    // twinkle phase 0 — the CPU's secondary-path convention.
                    //
                    // The cloud layer extinguishes this whole backdrop along
                    // the REFLECTED ray from the hit point (mirrored skies
                    // show the same clouds), MIS-weighted disc included: the
                    // BSDF sun rides the march's T, the light-sampled sun
                    // rides cloud_sun_transmittance — two transmittances of
                    // one field along near-identical directions, a bracketed
                    // partition (clouds.rs header; never force one T on both).
                    float3 dm_r = sky_dome(rdir);
                    float3 sky_r = dm_r + sky_disc(rdir, cone_spread * 0.5) * w_b
                        + sky_stars(rdir, cone_spread * 0.5, 0u);
                    if (flags & FLAG_CLOUDS) {
                        // The ROUGH march — reflected sky through the GGX
                        // lobe (clouds_along_rough's cost rationale).
                        float rct;
                        float3 rcs;
                        if (clouds_along_rough(p, rdir, dm_r * CLOUD_AMB_K, rct, rcs)) {
                            sky_r = sky_r * rct + rcs;
                        }
                    }
                    float3 rc = rtput * sky_r;
                    total += rc;
                    prim.ind_s += rc;
                }
            }
        }

        // (b) Glass transmission — the shade.rs Snell chain, shading-only
        // (glass still HITS: frustum bounds / inherited tmin / temporal
        // claims are untouched) and drawing ZERO rng, placed after the
        // reflection draws so the stream never moves. The Fresnel-reflected
        // fraction at the root is the VNDF bounce above; at interior
        // interfaces it is dropped — dimming, never gaining. TIR continues
        // as an internal mirror bounce.
        if (refl && SHADE_MAT_TRANSMISSION(mat) > 0.0 && depth < TRANS_MAX_DEPTH) {
            // Entering or exiting? Re-derive the pre-flip normal orientation
            // (surface_point returns only the viewer-facing normal).
            uint3 idx = uint3(indices[hit.tri * 3u], indices[hit.tri * 3u + 1u],
                              indices[hit.tri * 3u + 2u]);
            float w = 1.0 - hit.u - hit.v;
            float3 n_raw =
                normalize_or_zero(normals[idx.x] * w + normals[idx.y] * hit.u +
                                  normals[idx.z] * hit.v);
            if (all(n_raw == float3(0.0, 0.0, 0.0))) {
                float3 e1 = positions[idx.y] - positions[idx.x];
                float3 e2 = positions[idx.z] - positions[idx.x];
                n_raw = normalize_or_zero(cross(e1, e2));
            }
            bool entering = dot(n_raw, n) >= 0.0; // the viewer-flip didn't fire
            float eta = entering ? 1.0 / mat.ior : mat.ior;
            float3 hit_p = ro + rd * hit.t;
            float3 tdir, torig;
            float ttw;
            bool is_tir;
            // Water ripples perturb the Snell axis too (guarded): a refraction
            // must cross the geometric surface (tdir·n < 0), a TIR mirror stay
            // on the near side (tdir·n > 0). A ripple that flips the side is
            // rejected and the arm recomputes on geometric n (which always
            // passes both). Off (ripple_amp 0) runs geometric-n verbatim.
            if (SHADE_MAT_RIPPLE(mat) > 0.0) {
                float3 n_snell = ripple_normal(n, n, hit_p, CLOUD_TIME, mat.ripple_amp, SCENE_DIAG);
                glass_snell(rd, v, n_snell, n, hit_p, eta, mat.transmission, tdir, torig, ttw, is_tir);
                bool ok = is_tir ? (dot(tdir, n) > 0.0) : (dot(tdir, n) < 0.0);
                if (!ok) {
                    glass_snell(rd, v, n, n, hit_p, eta, mat.transmission, tdir, torig, ttw, is_tir);
                }
            } else {
                glass_snell(rd, v, n, n, hit_p, eta, mat.transmission, tdir, torig, ttw, is_tir);
            }
            if (ttw > 1e-3) {
                // Tinted by the ONE tint source (trans_tint for water, else
                // the albedo the classifier lifts toward white).
                float3 t_tput = tput * trans_tint_or(mat, albedo) * ttw;
#ifdef DXR_SBT_RECURSE
                // Recursive: the chain continues in the child's own class
                // CHS; Beer–Lambert multiplies the RETURNED radiance over
                // the child's reported segment — the CPU's exact
                // association ("multiplies the child's returned radiance"),
                // where the flattened arm below folds it into the child's
                // throughput instead. The miss arm is the parent's (fixed
                // fixed-phase sky, unattenuated — CPU parity), never
                // miss_rec's.
                float trec_t;
                float3 tcol = trace_shade(torig, tdir, rng, depth + 1u, cone_w, trec_t);
                if (!isinf(trec_t)) {
                    if ((entering || is_tir) && (flags & FLAG_DEPTH_TINT)) {
                        t_tput *= pow(max(trans_tint_or(mat, albedo), 1e-6),
                                      trec_t / (TRANS_DEPTH_K * SCENE_DIAG));
                    }
                    float3 tc = t_tput * tcol;
                    total += tc;
                    if (in_refl) prim.ind_s += tc;
                } else {
                    float3 tc = t_tput * sky_radiance(torig, tdir, cone_spread * 0.5, 0u, 0.5);
                    total += tc;
                    if (in_refl) prim.ind_s += tc;
                }
#else
                HitInfo th;
                if (ABL_TRACE_GLASS(torig, tdir, 0.0, FLT_MAX, th)) {
                    // Beer–Lambert over the interior segment (shade.rs's
                    // depth_attenuation twin — the flattened DFS folds it
                    // into the child's THROUGHPUT, since the child hit is
                    // already traced here; same product, different fp
                    // association, absorbed by the statistical CPU-vs-GPU
                    // gates). Entering crosses in; TIR (k < 0) stays in; a
                    // clean exit travels outside. The sky-miss arm below
                    // stays unattenuated (leaked geometry — CPU parity).
                    if ((entering || is_tir) && (flags & FLAG_DEPTH_TINT)) {
                        t_tput *= pow(max(trans_tint_or(mat, albedo), 1e-6),
                                      th.t / (TRANS_DEPTH_K * SCENE_DIAG));
                    }
                    if (!next_set) {
                        nx_o = torig;
                        nx_d = tdir;
                        nx_hit = th;
                        nx_tput = t_tput;
                        nx_depth = depth + 1u;
                        next_set = true;
                    } else {
                        // Root only: park the transmission child while the
                        // reflection subtree runs (CPU DFS order). Its cone
                        // origin is THIS hit's width — the reflection laps
                        // must not advance it.
                        st_o = torig;
                        st_d = tdir;
                        st_hit = th;
                        st_tput = t_tput;
                        st_depth = depth + 1u;
                        st_cone = cone_w;
                        st_in_refl = in_refl; // root's own chain: not ind_s
                        have_stash = true;
                    }
                } else {
                    // The FULL sky, disc included and un-weighted: refraction is
                    // a near-delta path with no light-sampling partner, so this
                    // is the only strategy that can deliver the sun through
                    // glass. Nothing to double-count.
                    // No pixel in scope — the fixed-midpoint legacy phase
                    // (the CPU glass miss passes the same 0.5).
                    float3 tc = t_tput * sky_radiance(torig, tdir, cone_spread * 0.5, 0u, 0.5);
                    total += tc;
                    if (in_refl) prim.ind_s += tc;
                }
#endif // !DXR_SBT_RECURSE (the flattened transmission arm)
            }
        }

        if (!next_set) {
            if (!have_stash) break;
            nx_o = st_o;
            nx_d = st_d;
            nx_hit = st_hit;
            nx_tput = st_tput;
            nx_depth = st_depth;
            nx_cone = st_cone;
            nx_in_refl = st_in_refl;
            have_stash = false;
        }
        // Continuation laps shade with reflections ineligible past the root
        // (the depth gate) and with fb OFF (the split_ambient lap-0 gate) —
        // the CPU's recursive `shade(depth + 1)` policy.
        ro = nx_o;
        rd = nx_d;
        hit = nx_hit;
        tput = nx_tput;
        depth = nx_depth;
        cone_o = nx_cone;
        in_refl = nx_in_refl;
    }
    return total;
}

// The plain (non-hemi) entry: quality straight from the frame constants;
// the ray cone is the primary one (apex at the camera, one-pixel spread).
// `el_mask` forwards verbatim — the leaf kernel passes its tile cull, every
// tile-less caller (reference, DXR raygen/chs) the full ~0 mask.
//
// REAL-TIME GI (#ifdef RTGI, --no-rtgi compiles it out — the off arm is the
// verbatim pre-RTGI call): when FLAG_RTGI is live the primary runs in
// split-ambient mode (amb_w = tput·kd·kt·dcav, NO SH pre-multiply — the flag
// derivation guarantees fb_mode == 0 whenever the bit is set, so the
// fb_mode==1 pre-multiply in the split arm never fires here) and ONE
// cosine-sampled bounce ray IS the ambient: hit → a second shade_split at the
// BOUNCE_Q literals (hemi_leaf.hlsl:99's call, lockstep — 1 shadow / 1 AO /
// no refl / octant cone / no cam lights; its SH×AO ambient is the tail
// standing in for deeper bounces), miss → sky_gather (NO sun disc — the
// once-per-path rule; direct_d already delivers the sun). `c += amb_w * li`
// with no π: cosine importance sampling makes the sampled radiance the
// irradiance-convention estimate directly (shade.rs's RTGI arm — the CPU
// source of truth; its rng draws sit at the ambient-tier position while
// these land after the DFS, a permitted cross-pipeline stream divergence).
// Bounce rays cannot re-bounce: the inner call is shade_split, never this
// entry. Flag-off frames pass split_ambient=false — value-identical to the
// pre-feature call (the runtime lever); fb frames clear the flag (the hemi
// tiers take precedence).
float3 shade_full(float3 ro, float3 rd, HitInfo hit, inout uint rng, uint2 el_mask,
                  out PrimSurf prim) {
    float3 w, o, n;
    // Camera rays (and their reflection/glass continuations) resolve their
    // footprint anisotropically when the session asks for it — FLAG_ANISO.
#ifdef RTGI
    bool rtgi = (flags & FLAG_RTGI) != 0u;
    float3 c = shade_split(ro, rd, hit, rng, shadow_samples, ao_samples, reflections != 0u,
                           rtgi, 0.0, pixel_cone, true, true, el_mask, w, o, n, prim
#ifdef DXR_SBT_RECURSE
                           // The DEPTH-0 root: chs_shade's recursion-tagged
                           // continuations bypass this entry and call shade_split
                           // with the payload's own depth.
                           , 0u
#endif
    );
    if (rtgi) {
        // Exactly the CPU arm's 2 direction draws, then the bounce shade's
        // stream-shared draws (hit: 1 shadow pair + 1 AO pair; miss: none).
        float gr1 = rng_next(rng);
        float gr2 = rng_next(rng);
        float3 bt1, bt2;
        onb(n, bt1, bt2);
        float3 bd = cosine_dir(n, bt1, bt2, gr1, gr2);
#ifdef HAVE_COUNTERS
        // Liveness stat (the CTR_TRANS_PASS shape — raw atomic in shading
        // code; reference/DXR units carry no counters and compile this out).
        uint _g;
        InterlockedAdd(counters[CTR_RTGI_RAYS], 1u, _g);
#endif
        HitInfo bh;
        float3 li;
        bool bhit = ABL_TRACE_GI(o, bd, 0.0, FLT_MAX, bh);
        if (bhit) {
            float3 w3, o3, n3;
            PrimSurf ps_unused; // bounce rays never capture (secondary-ray rule)
            li = shade_split(o, bd, bh, rng, 1u, 1u, false, false,
                             0.0, HEMI_CONE_SPREAD, false, false,
                             uint2(0xffffffffu, 0xffffffffu), w3, o3, n3, ps_unused
#ifdef DXR_SBT_RECURSE
                             // The bounce surface shades at depth 1 (the
                             // CPU's `depth + 1`), so its glass chain keeps
                             // the CPU's interface budget and the recursion
                             // stays under the declared 5.
                             , 1u
#endif
            );
        } else {
            li = sky_gather(bd);
        }
        c += w * li;
        if ((flags & FLAG_NRD_GI) != 0u) {
            // NRD diffuse fold: the bounce IS this frame's ambient (the split
            // arm leaves prim.ao at 0, so the bridge's D = dd + ao*amb
            // collapses to D = dd, which now carries it) — ReBLUR denoises
            // the GI instead of it riding the un-denoised residual, and the
            // exact-linear delta recompose is untouched because cs_nrd_out
            // reads D back from the packed plane. The bounce ray's own t is
            // the diffuse hit-dist guide (under RTGI no AO ray exists at
            // depth 0, so ao_t was 0 on EVERY pixel — ReBLUR's AREA_3X3
            // reconstruction had nothing to reconstruct from); miss stores
            // CAM_FAR, the shadow_t INF-clamp convention. Assignment-only,
            // zero rng — accum and the same-seed A/Bs are untouched, and
            // FSR-RR sessions (FLAG_FSR_SIG without this bit) keep dd = pure
            // direct diffuse for AMD's own denoiser.
            // FLAG_REMOD_EXACT, the RTGI arm's blend site — the one place the
            // bounce's own factor (prim.amb_k = sk*dcav) meets the direct
            // diffuse's (prim.m_d = sk), so the delta multiplier is weighted
            // here, BEFORE the add, while prim.direct_d still holds only the
            // direct term. WITHOUT this the bounce enters accum at kd*sk*dcav
            // while the bridge remodulates its correction at kd — on a cavity
            // pit that is up to 3.3x, and the leftover raw fraction of every
            // 1-spp spike never reaches the denoiser at all.
            //
            // The captured signal stays RAW `li` in both arms, so the fold
            // line below is textually and numerically today's code: every
            // correction rides the multiplier, and the denoiser input is the
            // clean demodulated radiance it was before.
            if (flags & FLAG_REMOD_EXACT) {
                prim.m_d = remod_blend(prim.direct_d, prim.m_d, li, prim.amb_k);
            }
            prim.direct_d += li;
            prim.ao_t = bhit ? bh.t : CAM_FAR;
        }
    }
    return c;
#else
    return shade_split(ro, rd, hit, rng, shadow_samples, ao_samples, reflections != 0u,
                       false, 0.0, pixel_cone, true, true, el_mask, w, o, n, prim
#ifdef DXR_SBT_RECURSE
                       // The DEPTH-0 root: chs_shade's recursion-tagged
                       // continuations bypass this entry and call shade_split
                       // with the payload's own depth.
                       , 0u
#endif
    );
#endif
}
