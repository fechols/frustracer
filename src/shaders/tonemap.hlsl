// The presentation curve on the GPU — a term-for-term port of `tone::map`
// (src/tone.rs), which is the single source of truth. Gated against it by
// --check-gpu M12 at <= 1 f16 ulp; the Rust side is gated closed-form by
// --check's `tone::self_test`. Do not "improve" the math here alone.
//
//   f(x) = x                                            for x <= knee
//   f(x) = knee + band * (1 - exp(-(x - knee) / band))  for x >  knee,  band = headroom - knee
//
// SDR is the degenerate case (knee 0, headroom 1, gamma on) and reduces to the
// pre-HDR curve `(1 - exp(-c))^(1/2.2)` exactly — which is why there is no
// separate SDR shader.
//
// inv_samples is 1.0 when the source is a per-frame radiance image (every
// upscaler output, the GPU tracer, DXR) and 1/frame_count when it is the CPU
// accumulation sum (present_hdr).
Texture2D<float4> src : register(t0);
// The glare halo (gpu/bloom.rs level 0, half-res). Zero-strength when bloom is
// off, in which case this is never sampled and the arm below is skipped.
Texture2D<float4> glare : register(t1);
SamplerState samp_lin : register(s0); // linear, clamp

// Field order is the root-constant DWORD order tonemap.rs writes; the float2
// sits at offset 8 so it cannot straddle a 16-byte boundary (the fsr_composite
// bug). TEN contiguous DWORDs — hud.hlsl mirrors this layout field-for-field
// and waveviz.hlsl pads to it; all three move together.
cbuffer Params : register(b0) {
    float inv_samples;
    float bloom_strength; // 0 = off
    float2 bloom_texel;   // 1 / glare dims (the tent's tap spacing)
    float knee;      // rolloff start, in paper-white units
    float headroom;  // asymptote = peak_nits / paper_white; 1.0 on SDR
    float scale;     // Gamma22 (SDR/Sdr10): 1.0. HDR10: paper_white / 10000
    float mode;      // 1 = gamma 2.2 (8-bit AND Sdr10), 2 = HDR10 PQ (tone::ToneMode;
                     // 0 was scRGB-linear, retired with the f16 swapchain)
    float exposure;  // ToneParams::exposure — the aperture (autoexp.rs). 1.0 = off.
    float guard_strength; // the spike guard (autoexp::guard_strength_live). 0 = off.
};

// ---------------------------------------------------------------------------
// The SPIKE GUARD — a per-pixel exemption from the aperture's boost.
// ---------------------------------------------------------------------------
// A term-for-term twin of `autoexp::guard_scale`, which is the single source of
// truth; --check-gpu's M12 scores this against it. The K/radius constants
// arrive as INJECTED DEFINES from autoexp::guard_defs() so the FR_AEXP_GUARD_*
// sweep levers provably reach the shader; the fallbacks below keep the file
// compilable standalone (--check-spirv assembles it with no injector).
#ifndef AEXP_GUARD_K_MEAN
#define AEXP_GUARD_K_MEAN 4.0
#endif
#ifndef AEXP_GUARD_K_MAX
#define AEXP_GUARD_K_MAX 2.0
#endif
#ifndef AEXP_GUARD_RAMP
#define AEXP_GUARD_RAMP 2.0
#endif
#ifndef AEXP_GUARD_R
#define AEXP_GUARD_R 4
#endif
#ifndef AEXP_GUARD_MIN_RING
#define AEXP_GUARD_MIN_RING 4
#endif

// Rec.709 luma — autoexp::luma's weights, shared with autoexp.hlsl's meter so
// the guard and the aperture agree about what "bright" means.
float aexp_luma(float3 c) { return dot(max(c, 0.0), float3(0.2126, 0.7152, 0.0722)); }

float aexp_guard_scale(float l, float ring_mean, float ring_max, uint nv,
                       float e, float strength) {
    // `!(x > y)` throughout so a NaN in any input takes the off arm. Each of
    // these is a BRANCH returning `e` bitwise, never a computed lerp — the
    // disarmed stream must be the pre-feature one.
    if (!(e > 1.0) || !(strength > 0.0) || nv < uint(AEXP_GUARD_MIN_RING) || !(l > 0.0)) return e;
    float cap = max(AEXP_GUARD_K_MEAN * ring_mean, AEXP_GUARD_K_MAX * ring_max);
    // A NaN in either reduction takes this arm, so `cap` is a real number below.
    if (!(l > cap)) return e;
    // The saturated end of the ramp is tested DIRECTLY, not reached through
    // l/cap: an exactly-black ring gives cap == 0, which is a lit dot on
    // absolute black — the most extreme outlier there is, and the one a
    // `cap > 0` guard would silently hand the full boost to. This exempts it,
    // and keeps x/0 out of the expression on both sides of the port.
    float t = (l >= cap * AEXP_GUARD_RAMP)
        ? 1.0
        : saturate((l / cap - 1.0) / (AEXP_GUARD_RAMP - 1.0));
    float w = t * t * (3.0 - 2.0 * t) * strength;
    if (!(w > 0.0)) return e;
    // 1 + (E-1)(1-w), NOT lerp(E, 1, w): both factors are non-negative here, so
    // the result is >= 1.0 BY CONSTRUCTION and a full exemption lands on
    // exactly 1.0. That bound is the feature's whole safety argument.
    return 1.0 + (e - 1.0) * (1.0 - w);
}

// Rec.709 -> Rec.2020 primaries. Literals mirror tone::m709_to_2020
// term-for-term; change both together.
float3 m709_to_2020(float3 v) {
    return float3(
        0.627404 * v.r + 0.329283 * v.g + 0.043313 * v.b,
        0.069097 * v.r + 0.919540 * v.g + 0.011362 * v.b,
        0.016391 * v.r + 0.088013 * v.g + 0.895595 * v.b);
}

// SMPTE ST 2084 inverse EOTF (luminance normalized to 10000 nits -> PQ
// signal). Literals mirror tone::pq_encode term-for-term.
float3 pq_encode(float3 y) {
    float3 yp = pow(saturate(y), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * yp) / (1.0 + 18.6875 * yp), 78.84375);
}

float4 vsmain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

// 3x3 tent (1,2,1 / 2,4,2 / 1,2,1 over 16) — bloom.rs::tent. The tent is what
// keeps the glare core ROUND: a 2x2 box downsample has a square footprint, and a
// single bilinear tap would leave it looking like a rounded rectangle.
float3 tent(float2 uv) {
    float2 t = bloom_texel;
    float3 s = 0.0;
    s += glare.SampleLevel(samp_lin, uv + float2(-t.x, -t.y), 0).rgb * 1.0;
    s += glare.SampleLevel(samp_lin, uv + float2( 0.0, -t.y), 0).rgb * 2.0;
    s += glare.SampleLevel(samp_lin, uv + float2( t.x, -t.y), 0).rgb * 1.0;
    s += glare.SampleLevel(samp_lin, uv + float2(-t.x,  0.0), 0).rgb * 2.0;
    s += glare.SampleLevel(samp_lin, uv,                      0).rgb * 4.0;
    s += glare.SampleLevel(samp_lin, uv + float2( t.x,  0.0), 0).rgb * 2.0;
    s += glare.SampleLevel(samp_lin, uv + float2(-t.x,  t.y), 0).rgb * 1.0;
    s += glare.SampleLevel(samp_lin, uv + float2( 0.0,  t.y), 0).rgb * 2.0;
    s += glare.SampleLevel(samp_lin, uv + float2( t.x,  t.y), 0).rgb * 1.0;
    return s * (1.0 / 16.0);
}

float curve(float x) {
    if (x <= knee) return x;
    float band = headroom - knee;
    return knee + band * (1.0 - exp(-(x - knee) / band));
}

float4 psmain(float4 pos : SV_Position) : SV_Target {
    float2 dims;
    src.GetDimensions(dims.x, dims.y);
    // max(0) mirrors tone::map's clamp: a negative radiance would take the
    // below-knee arm and then reach pow() with a negative base (NaN).
    float3 c0 = max(src.Load(int3(pos.xy, 0)).rgb * inv_samples, 0.0);
    float3 c = c0;
    if (bloom_strength > 0.0) {
        // Glare is applied to the LINEAR radiance, BEFORE the curve — it is a
        // model of light scattering in the lens/eye, which happens to the light
        // itself, not to the displayed pixel. (It also has to be: the curve is
        // where a 44,000-radiance sun disc gets compressed to the display's
        // peak, and glare's whole job is to redistribute that energy into the
        // surrounding pixels while it still EXISTS.)
        //
        // Energy-conserving composite (bloom.rs): glare REDISTRIBUTES light, it
        // never adds any, so a uniformly lit frame comes back unchanged — which
        // is what keeps bloom from being tunable into an exposure change.
        // `pos.xy` is SV_Position, which ALREADY points at the pixel CENTRE
        // (16.5 for pixel 16) — the `src.Load(int3(pos.xy, 0))` above relies on
        // exactly that, since Load truncates the .5 away. So this must NOT add
        // another half: `(pos.xy + 0.5)` sampled the glare half a full-res texel
        // down-right, which on the half-res pyramid is a QUARTER of a glare
        // texel, and disagreed with the CPU twin (bloom.rs's `u = (x + 0.5) *
        // gw / w - 0.5`, which lands on 7.75 where this landed on 8.0). Small on
        // a blur, and gated now by M13b's symmetry arm rather than by argument.
        float2 uv = pos.xy / dims;
        c = lerp(c, tent(uv), bloom_strength);
    }
    // THE SPIKE GUARD: withhold the aperture's boost from pixels that are
    // outliers against their own surround, so brightening a dark scene does not
    // also amplify its stochastic dots. Bounded by construction — the exemption
    // only ever relaxes toward 1.0, so a false positive is presented at exactly
    // its --no-auto-exposure brightness.
    //
    // Both conditions are cbuffer values and therefore UNIFORM, so a disarmed
    // frame executes the pre-feature instruction stream — no taps, no branch
    // divergence, nothing to price.
    float e_px = exposure;
    if (exposure > 1.0 && guard_strength > 0.0) {
        // The ring is read PRE-GLARE, off `src`, and so is the centre luma:
        // bloom's whole job is spreading an outlier into its neighbourhood, so
        // measuring after it would let a firefly's own halo raise the ring mean
        // and shield it. Detection is bloom-independent, and costs nothing
        // extra — the taps are `src` loads either way.
        int2 p0 = int2(pos.xy);
        int2 lim = int2(int(dims.x) - 1, int(dims.y) - 1);
        float sum = 0.0, mx = 0.0;
        uint nv = 0u;
        // A DONUT at radius R, not the adjacent 3x3 ring the sibling clamps
        // use: this runs after the upscaler, so a 1-px firefly at trace res is
        // a 2-3 px blob here and an adjacent ring would sample the blob's own
        // body — making ring_max hot and letting the K_MAX arm shield exactly
        // the thing being hunted.
        [unroll] for (int dy = -1; dy <= 1; ++dy) {
            [unroll] for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue; // CENTRE EXCLUDED (frd's lesson)
                int2 q = p0 + int2(dx, dy) * AEXP_GUARD_R;
                // EXCLUDED, never clamped-to-edge: a replicated ring would
                // measure the border pixel against copies of itself.
                if (q.x < 0 || q.y < 0 || q.x > lim.x || q.y > lim.y) continue;
                float v = aexp_luma(src.Load(int3(q, 0)).rgb * inv_samples);
                sum += v;
                mx = max(mx, v);
                nv += 1u;
            }
        }
        float mean = (nv > 0u) ? sum / float(nv) : 0.0;
        e_px = aexp_guard_scale(aexp_luma(c0), mean, mx, nv, exposure, guard_strength);
    }
    // Exposure: pre-curve, AFTER the glare blend — the lerp is linear, so
    // scaling the blend equals exposing source and glare consistently, and it
    // matches the CPU order (bloom composites into the HDR slice before
    // tone::shape applies exposure). Branched at 1.0 like tone::shape, so the
    // unexposed stream is bit-identical (M12's sdr/sdr10/hdr10 rows).
    if (e_px != 1.0) c *= e_px;
    float3 f = float3(curve(c.r), curve(c.g), curve(c.b));
    if (mode > 1.5) {
        // HDR10: paper-white-relative -> PQ's 10000-nit-normalized domain,
        // then gamut matrix + ST 2084 — tone::encode's Pq arm verbatim.
        return float4(pq_encode(m709_to_2020(f * scale)), 1.0);
    }
    // Gamma 2.2, display-encoded — both the 8-bit and Sdr10 wires (the RTV
    // format quantizes; the curve is bounded by headroom = 1, and the UNORM
    // ROP clamps anyway).
    return float4(pow(f, 1.0 / 2.2) * scale, 1.0);
}
