// NRD bridge kernels (--nrd): the two ends of the ReBLUR pre-upscale denoise
// stage. cs_nrd_pack fans the G-buffer pack into NRD's five input planes;
// cs_nrd_out folds the denoised channels back into the upscaler's color plane
// in DELTA form. Compiled as its own unit — [defs, TRACE_COMMON_HLSLI,
// FSR_WIRE_HLSLI, NRD_BRIDGE_HLSL] — at cs_6_3 (no rays; must stay
// cs_6_3-clean so the DXR chain's unit can compile it at its floor, the
// feed.hlsl precedent). Planes ride descriptor set trace::NRD_FEED_SET over
// the shared u16..u27 registers (this unit's own typed declarations).
//
// THE PACKING MATH IS REIMPLEMENTED from NRD v4.17.3's documented semantics
// (YCoCg rotation, the encoding-2 L1-octahedral normal pack, ReBLUR's
// hit-distance normalization) — NEVER paste the licensed NRD.hlsli here.
// One formula, three sites: nrd.rs::oracle (the CPU twins the N0/N2 gates
// pin), cs_nrd_pack, cs_nrd_out. Change all three together.
//
// NRD.hlsli IS NEVERTHELESS AVAILABLE TO THIS UNIT — and the two statements do
// not conflict, because the mechanisms are different. gfx::shaders::nrd_header
// `include_str!`s NVIDIA's header out of the SUBMODULE at build time and joins
// it ahead of this file; nothing of theirs is committed here, and this is the
// only unit that gets it. So `NRD_SG_ReJitter` below is THEIR code running as
// theirs, while the pack math above stays OURS — deliberately, because the
// N0/N2 gates score the shader against nrd::oracle's CPU twins, and swapping
// the shader side to NVIDIA's would leave those gates comparing NVIDIA's code
// against itself. Do not "unify" them.
//
// THE DELTA-FORM RECOMPOSE is the correctness spine:
//     color' = accum + (D_out − D_in)·kd + (S_out − S_in)·f0
// with kd = gbuf_ext.alb.xyz (the EXACT f32 factor shade multiplied its
// diffuse terms by) and f0 = max(sqrt_wire3(gbuf_ext.spec.xyz), 1e-4) (the
// EXACT demodulation divisor the pack's sig lanes were divided by,
// trace_common.hlsli's gbuf_write_hit) — so remodulation is exact, the
// residual (everything not in the two folds) is structurally untouched, and
// a PASSTHROUGH denoiser (OUT planes byte-copied from IN) makes both deltas
// exactly 0.0, reproducing cs_feed_xess's color plane BYTE-IDENTICALLY (the
// N3 control-arm gate). The diffuse fold is dd + ao·sh_irr(sky_sh, n) —
// shade's own ambient additionally carries the detail-cavity factor at some
// pixels; the mismatch only shifts what the denoiser sees between "signal"
// and "residual" (a quality nuance), never the recomposed color, because the
// delta form subtracts the same fold it added.
//
// SKY: NRD does not write output pixels outside CommonSettings.denoisingRange
// (= far * 0.999, nrd_gpu.rs — the shader's 0.999 * CAM_FAR literal is its
// LOCKSTEP twin), so the OUT planes are STALE there — cs_nrd_out passes accum
// through verbatim for every out-of-range pixel: sky (view_z == CAM_FAR) AND
// the [0.999·far, far) hit band, or stale bytes leak into the recompose.

RWStructuredBuffer<float> accum : register(u0); // rw*rh*3 linear HDR (1-spp)

// THE FOLD (2026-08-09): cs_nrd_pack ALSO writes the engine's mvec/depth
// guide planes — what record_feed_nrd used to dispatch cs_feed_xess_dm for —
// so an NRD frame runs one full-screen guide pass instead of two (the second
// GBufCore load + a dispatch + the color plane's redundant transition
// round-trip, ~0.15-0.25 ms on the B70). The engine planes take feed.hlsl's
// own registers/types (u18/u19), which is why nrd_in_viewz moved u18 → u26
// (free in an NRD session: record-time policy refuses RR/FSR-RR, whose
// plane sets claim u20..u22 — never 26 on THIS set).
RWTexture2D<float4> nrd_color_out : register(u16); // the upscaler's color plane
RWTexture2D<float4> nrd_in_nr : register(u17);     // R10G10B10A2: NRD enc-2
RWTexture2D<float> feed_depth : register(u18);     // R32F reversed-Z clip (ENGINE)
RWTexture2D<float2> feed_mvec : register(u19);     // RG16F pixel MVs (ENGINE)
RWTexture2D<float4> nrd_out_spec : register(u20);  // RGBA16F (NRD-written)
RWTexture2D<float4> nrd_in_mv : register(u23);     // RGBA16F 2.5D MV
RWTexture2D<float4> nrd_in_diff : register(u24);   // RGBA16F YCoCg + normHitDist
RWTexture2D<float4> nrd_in_spec : register(u25);   // RGBA16F YCoCg + normHitDist
RWTexture2D<float> nrd_in_viewz : register(u26);   // R32F linear view-Z
RWTexture2D<float4> nrd_out_diff : register(u27);  // RGBA16F (NRD-written)

// ReBLUR's HitDistanceParameters (A, B, C) — LITERALS, in LOCKSTEP with the
// ReblurSettings defaults gpu/nrd_gpu.rs passes at SetDenoiserSettings (and
// nrd.rs::oracle's pins). If a session ever tunes them, they move to the CB.
static const float3 NRD_HITDIST_PARAMS = float3(3.0, 0.1, 20.0);
static const float NRD_FP16_MAX_B = 65504.0;

// _NRD_LinearToYCoCg / _NRD_YCoCgToLinear semantics (nrd.rs::oracle twins).
float3 nrd_ycocg(float3 c) {
    return float3(
        0.25 * c.x + 0.5 * c.y + 0.25 * c.z,
        0.5 * c.x - 0.5 * c.z,
        -0.25 * c.x + 0.5 * c.y - 0.25 * c.z);
}

float3 nrd_ycocg_inv(float3 c) {
    float t = c.x - c.z;
    return max(float3(t + c.y, c.x + c.z, t - c.y), 0.0);
}

// NRD_FrontEnd_PackNormalAndRoughness at NRD_NORMAL_ENCODING=2 semantics:
// L1-octahedral xy, SIGNED roughness (n.z's sign, floor 1.5/512) in z,
// materialID/3 in the 2-bit A (we always store 0).
float4 nrd_pack_nr(float3 n, float rough) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float ry = n.y * 0.5 + 0.5;
    float rx = n.x * 0.5 + ry;
    ry -= n.x * 0.5;
    float r = max(rough, 1.5 / 512.0);
    float s = n.z < 0.0 ? -r : r;
    return float4(rx, ry, s * 0.5 + 0.5, 0.0);
}

// REBLUR_FrontEnd_GetNormHitDist semantics: saturate(hitDist / f) floored at
// 1e-6, f = (A + |viewZ|·B) · lerp(C, 1, smc(roughness)). Callers gate the
// "no ray" 0 sentinel to 0 THEMSELVES (0 = invalid ⇒ NRD's hit-distance
// reconstruction; the floor is only for lobes that really traced).
float nrd_norm_hit_dist(float hit_dist, float view_z, float rough) {
    float f = 1.0 - exp2(-200.0 * rough * rough);
    float smc = f * sqrt(saturate(rough));
    float norm = (NRD_HITDIST_PARAMS.x + abs(view_z) * NRD_HITDIST_PARAMS.y)
        * lerp(NRD_HITDIST_PARAMS.z, 1.0, smc);
    return max(saturate(hit_dist / norm), 1e-6);
}

// Exponent bit test, NOT isnan()/isinf() — DXC folds those away without
// strict IEEE mode (the quin.hlsl finite1 lesson). Semantics equal the Rust
// oracle's is_finite (exponent all-ones = Inf or NaN).
bool nrd_finite1(float v) { return ((asuint(v) >> 23) & 0xFFu) != 0xFFu; }

// THE RE-JITTER TAP (FLAG_NRD_REJITTER, 2026-08-10). NVIDIA's own
// `NRD_SG_ReJitter` — reachable because this unit, and ONLY this unit, joins
// NRD.hlsli straight out of the submodule (gfx::shaders::nrd_header). It
// returns a float2 Jacobian in [1/NRD_REJITTER_AMPLITUDE, NRD_REJITTER_
// AMPLITUDE] = the ratio of the BRDF at THIS pixel's normal to the mean BRDF
// over its 4 neighbours: exactly the texel-scale variation a spatial blur
// averaged away, which is the parked-camera "detail flattens as it converges"
// class. A pure multiplier, so it composes with our recompose without any
// convention change — unlike the SG RESOLVE half, see cs_nrd_out.
//
// THE DIRECTIONS ARE ANALYTIC, and that bounds what this can do. ReJitter
// needs the diffuse and specular DOMINANT LIGHT directions; NRD's own pipeline
// carries them in the SH1 planes, packed from the actual sampled ray. We do
// not export ray directions (see the SG note in cs_nrd_out), so we hand it the
// analytic pair: N for the cosine lobe, and NVIDIA's own
// `_NRD_GetSpecularDominantDirection` for GGX. Both are smooth functions of
// (N, V, roughness) — which is what the Jacobian wants, since it measures how
// the BRDF responds to the NORMAL varying across the neighbourhood.
//
// BUT THE SUBSTITUTION IS NOT NEUTRAL, AND THE DIFFUSE HALF IS ONE-SIDED.
// `_NRD_ComputeBrdfs` weights by `saturate(dot(N_tap, Ld))`, so handing it
// Ld = N_center evaluates the CENTER at dot(N,N) = 1 — the maximum of that
// factor — while every neighbour evaluates at dot(N_nb, N_c) <= 1. The diffuse
// Jacobian is therefore biased ABOVE 1 wherever normals vary: it sharpens, and
// essentially never attenuates, up to NRD's own 2.0 clamp. (The specular half
// is anchored the same way — Ls is derived FROM the center N, so the center
// sits at its lobe's peak — though ReJitter's `lerp(V, Ls, roughness)` step
// muddies it.) A true dominant LIGHT direction is generally not N, which is
// what makes NVIDIA's own Jacobian two-sided. So read this as "amplify the
// denoiser delta where normals vary", not as NVIDIA's symmetric restoration —
// and read the measured +5.2% Laplacian / +0.2% mean the same way: contrast
// AND level move together, which is the signature of a one-sided multiplier.
// THE FOLLOW-ON, if the default is kept: the sun direction is already in the
// CB and is a far better diffuse-dominant proxy for this content than N, and
// it restores two-sidedness for free. Measure it before adopting — the
// analytic pair is the CHEAP form, and the reason to prefer it is that it
// cannot be wrong about a direction it never claimed to know.
//
// `c1` is handed UNIT length rather than `direction * c0`: ReJitter consumes
// only `_NRD_SG_ExtractDirection`, which normalizes, so the two agree exactly
// for c0 > 0 — and the unit form stays well-defined at c0 == 0 (a black pixel),
// where `direction * c0` is the zero vector and the extraction degenerates.
struct RejitterTaps {
    float z, ze, zw, zn, zs;
    float3 n, ne, nw, nn, ns;
};

// Neighbour fetch, edge-clamped. Reads BOTH planes: viewZ from GBufCore (always
// written) and the world normal from GBufExt — which under FLAG_SKY_EXT_SKIP is
// UNWRITTEN at sky texels, so the caller must reject out-of-range neighbours
// BEFORE trusting a normal. That is not merely defensive: stale ext bytes are
// exactly what the armed check-gpu NaN-sentinel gate proves stay unread.
void nrd_tap(uint2 p, int dx, int dy, out float z, out float3 n) {
    int2 q = int2(clamp(int(p.x) + dx, 0, int(rw) - 1), clamp(int(p.y) + dy, 0, int(rh) - 1));
    uint qi = uint(q.y) * rw + uint(q.x);
    z = gbuf[qi].core.z;
    n = gbuf_ext[qi].nr.xyz;
}

float3 nrd_sanitize(float3 c) {
    if (!(nrd_finite1(c.x) && nrd_finite1(c.y) && nrd_finite1(c.z))) {
        return float3(0.0, 0.0, 0.0);
    }
    return clamp(c, 0.0, NRD_FP16_MAX_B);
}

[numthreads(8, 8, 1)]
void cs_nrd_pack(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint pi = id.y * rw + id.x;
    GBufCore c = gbuf[pi];

    // Guides. MV: the pack's pixel-space prev−cur plus the 2.5D view-Z delta
    // (prev_z − view_z, both already captured); CommonSettings'
    // motionVectorScale = {1/rectW, 1/rectH, 1} converts xy to the UV units
    // NRD expects — the plane stays in the tree's own pixel convention.
    nrd_in_viewz[id.xy] = c.core.z;
    nrd_in_mv[id.xy] = float4(c.core.x, c.core.y, c.core.w - c.core.z, 0.0);
    // The folded engine guides — cs_feed_xess_dm's two stores VERBATIM (one
    // shared view_z_to_clip_depth in trace_common.hlsli, so the depth ulp
    // gates see one encode).
    feed_mvec[id.xy] = c.core.xy;
    feed_depth[id.xy] = view_z_to_clip_depth(c.core.z, CAM_NEAR, CAM_FAR);

    // OUT-OF-RANGE (sky + the [0.999·far, far) band — cs_nrd_out's EXACT
    // predicate): NRD never writes these texels (denoisingRange), so the
    // remaining planes carry deterministic constants and the ext record is
    // NEVER read here — which is what lets FLAG_SKY_EXT_SKIP elide the sky
    // ext store entirely (stale bytes must stay unread BY CONSTRUCTION, the
    // armed check-gpu gate's NaN-sentinel proof). The nr constant is any
    // NaN-free encoding (never nrd_pack_nr of a stale/zero normal — 0/0);
    // viewZ carries the neighbor rejection, so the value is inert.
    if (c.core.z >= 0.999 * CAM_FAR) {
        nrd_in_nr[id.xy] = nrd_pack_nr(float3(0.0, 1.0, 0.0), 1.0);
        nrd_in_diff[id.xy] = float4(0.0, 0.0, 0.0, 0.0);
        nrd_in_spec[id.xy] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    GBufExt g = gbuf_ext[pi];
    nrd_in_nr[id.xy] = nrd_pack_nr(g.nr.xyz, g.nr.w);

    // The two folded noisy channels (sky rows carry all-zero sig lanes, so
    // they pack to zero radiance; denoisingRange excludes them anyway).
    float3 dd = float3(f16tof32(g.sig.x & 0xffffu), f16tof32(g.sig.x >> 16),
                       f16tof32(g.sig.y & 0xffffu));
    float3 ds = float3(f16tof32(g.sig.y >> 16), f16tof32(g.sig.z & 0xffffu),
                       f16tof32(g.sig.z >> 16));
    float ao = f16tof32(g.sig2.x & 0xffffu);
    float3 is = float3(f16tof32(g.sig2.x >> 16), f16tof32(g.sig2.y & 0xffffu),
                       f16tof32(g.sig2.y >> 16));
    float ao_t = f16tof32(g.sig.w & 0xffffu);

    float3 amb = sh_irradiance(g.nr.xyz);
    float3 diff = nrd_sanitize(dd + ao * amb);
    float3 spec = nrd_sanitize(ds + is);

    // Hit-dist lanes: 0 = "no ray traced" (NRD-invalid, reconstruction
    // covers it) — the AO lane when no AO ray fired (or a no-capture
    // intersector arm reported plain tmax == AO_RADIUS: still a valid miss),
    // the spec lane when the reflection gate never fired (spec.w == 0).
    float nh_d = ao_t > 0.0 ? nrd_norm_hit_dist(ao_t, c.core.z, 1.0) : 0.0;
    float nh_s = g.spec.w > 0.0 ? nrd_norm_hit_dist(g.spec.w, c.core.z, g.nr.w) : 0.0;

    nrd_in_diff[id.xy] = float4(nrd_ycocg(diff), nh_d);
    nrd_in_spec[id.xy] = float4(nrd_ycocg(spec), nh_s);
}

[numthreads(8, 8, 1)]
void cs_nrd_out(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint pi = id.y * rw + id.x;
    float3 base = float3(accum[pi * 3u], accum[pi * 3u + 1u], accum[pi * 3u + 2u]);
    GBufCore c = gbuf[pi];
    // Out-of-range: NRD never wrote these OUT texels — pass accum through
    // verbatim (cs_feed_xess's own value there, so N3's byte compare holds
    // too). The predicate is CommonSettings.denoisingRange's EXACT bound
    // (far * 0.999, nrd_gpu.rs::common_settings — LOCKSTEP), not CAM_FAR:
    // NRD skips every pixel past denoisingRange, sky (view_z == CAM_FAR)
    // included, and a hit whose view-Z lands in [0.999·far, far) would
    // otherwise delta-recompose from OUT texels NRD never wrote. That band
    // forgoes denoising (accum verbatim) — coarser, never garbage.
    if (c.core.z >= 0.999 * CAM_FAR) {
        nrd_color_out[id.xy] = float4(base, 1.0);
        return;
    }
    GBufExt g = gbuf_ext[pi];
    float3 d_in = nrd_ycocg_inv(nrd_in_diff[id.xy].xyz);
    float3 d_out = nrd_ycocg_inv(nrd_out_diff[id.xy].xyz);
    float3 s_in = nrd_ycocg_inv(nrd_in_spec[id.xy].xyz);
    float3 s_out = nrd_ycocg_inv(nrd_out_spec[id.xy].xyz);
    float3 kd = g.alb.xyz;
    float3 f0 = max(sqrt_wire3(g.spec.xyz), float3(1e-4, 1e-4, 1e-4));

    // RE-JITTER — the micro-detail a converged history's narrow blur averaged
    // away, restored from NVIDIA's own local-BRDF Jacobian (see nrd_tap).
    //
    // WHY THE JACOBIAN MULTIPLIES THE DELTA AND NOT THE OUTPUT. NRD's own
    // Composition example writes `resolved * rejitter * materialFactor`,
    // because THERE the denoised value REPLACES the raw one. Our recompose is
    // additive against the raw frame: `base` already carries this pixel's own
    // shading at its own normal, i.e. the true per-pixel BRDF response is
    // already in it, and the only denoiser-produced quantity is the DELTA. So
    // the smeared part is the delta and the correction belongs there. It also
    // makes the N3 control arm hold BY CONSTRUCTION rather than by tolerance:
    // a passthrough denoiser gives an exactly-0.0 delta, and 0.0 times a finite
    // clamped Jacobian is 0.0, so `base` comes back bitwise.
    //
    // WHY THERE IS NO `NRD_SG_ResolveDiffuse`/`ResolveSpecular` HERE, and this
    // is a convention mismatch rather than an omission: those functions take
    // INCIDENT radiance from the dominant direction and INTEGRATE a BRDF to
    // produce the outgoing term (`_NRD_DiffuseTerm`/`_NRD_GeometryTerm` are
    // applied inside them). The lanes we pack are already BRDF-integrated,
    // albedo-demodulated OUTGOING lobes — shade's own `direct_d`/`direct_s`
    // plus the AO·sky and reflection folds — so resolving them would apply the
    // BRDF a second time. Using the resolve would mean re-cutting the pack to
    // carry incident radiance and dropping the delta-form spine with it; that
    // is a G-buffer project, not a back-end swap.
    float2 j = float2(1.0, 1.0);
    if (flags & FLAG_NRD_REJITTER) {
        RejitterTaps t;
        t.z = c.core.z;
        t.n = g.nr.xyz;
        nrd_tap(id.xy, 1, 0, t.ze, t.ne);
        nrd_tap(id.xy, -1, 0, t.zw, t.nw);
        nrd_tap(id.xy, 0, 1, t.zn, t.nn);
        nrd_tap(id.xy, 0, -1, t.zs, t.ns);
        // Reject BEFORE reading a neighbour normal into the math. A neighbour
        // past denoisingRange (sky, or the [0.999·far, far) band) may have an
        // UNWRITTEN ext record under FLAG_SKY_EXT_SKIP, so its normal can be
        // any bytes at all. NRD's own Z test would disarm the Jacobian for
        // exactly these pixels anyway — `isSymmetrical` needs all four
        // neighbours inside the threshold, and an out-of-range Z is orders
        // past it — so pre-testing changes no result, it only keeps stale
        // bytes out of an arithmetic expression (the armed NaN-sentinel gate's
        // "unread BY CONSTRUCTION" contract).
        float lim = 0.999 * CAM_FAR;
        bool ok = t.ze < lim && t.zw < lim && t.zn < lim && t.zs < lim;
        if (ok) {
            float3 V = -ray_dir(float(id.x) + 0.5, float(id.y) + 0.5);
            float rough = g.nr.w;
            float NoV = abs(dot(t.n, V));
            float3 Ld = t.n;
            float3 Ls = _NRD_GetSpecularDominantDirection(
                t.n, V, _NRD_GetSpecularDominantFactor(NoV, rough));
            // sh0 IS DELIBERATELY ZERO, and that is a statement about the
            // function rather than a shortcut: ReJitter touches its two SGs
            // ONLY through `_NRD_SG_ExtractDirection`, so c0/chroma/
            // normHitDist are dead and the DENOISED RADIANCE NEVER ENTERS the
            // Jacobian — it is a pure function of (N, V, roughness) and the
            // four neighbours' (N, Z). Passing the real OUT planes here would
            // compile to the same thing and read as though the denoised value
            // mattered. Still built through NVIDIA's own constructor rather
            // than by assigning their struct's fields, which keeps the
            // clean-room posture (we call their code, we do not restate it).
            NRD_SG dsg = REBLUR_BackEnd_UnpackSh(float4(0.0, 0.0, 0.0, 0.0), Ld);
            NRD_SG ssg = REBLUR_BackEnd_UnpackSh(float4(0.0, 0.0, 0.0, 0.0), Ls);
            j = NRD_SG_ReJitter(dsg, ssg, V, rough,
                                t.z, t.ze, t.zw, t.zn, t.zs,
                                t.n, t.ne, t.nw, t.nn, t.ns);
        }
    }

    // precise: the deltas must stay literal subtractions — a passthrough's
    // exact 0.0 (identical loads) times anything is 0.0, and base + 0.0 is
    // base bitwise, which is what the N3 byte gate rests on.
    precise float3 col = base + (d_out - d_in) * j.x * kd + (s_out - s_in) * j.y * f0;
    nrd_color_out[id.xy] = float4(clamp(col, 0.0, NRD_FP16_MAX_B), 1.0);
}
