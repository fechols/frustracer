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
#define FLAG_GBUF         16u  // G-buffer pack writes on (upscaler sessions ONLY —
                               // the pack buffer is a 64-byte dummy otherwise and
                               // root UAVs have no bounds check: this gate is
                               // memory safety, not an optimization)
#define FLAG_HAS_PREV     32u  // the prev_* rows carry last frame's camera basis
#define FLAG_FSR_SIG      64u  // FSR-RR sessions: demodulated direct-light
                               // signals in GBufPx.sig + the prev-camera
                               // view-Z in mv.z (zeros otherwise — RR/XeSS
                               // sessions keep their pack bytes unchanged)
#define FLAG_ANISO        128u // anisotropic filtering on (--aniso > 1; a
                               // SESSION constant from texture::max_aniso()).
                               // WHICH laps use it is a call-site decision
                               // (shade_split's `aniso` arg — hemi bounce
                               // laps pass false), not this flag alone.

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
    // pixel_cone: primary ray-cone spread (CamBasis::pixel_cone verbatim,
    // the trilinear texture LOD's single source — shade.hlsli::tex_lod_base).
    float2 frame_jitter; float pixel_cone; float _pad2;
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
    // Previous frame's camera basis for G-buffer motion vectors (upscaler
    // sessions; zeroed with FLAG_HAS_PREV clear when there is no prev frame).
    // The scene-static near/far planes (dlss::near_far) ride the w slots.
    float4 prev_origin;  // xyz; w = prev inv_w
    float4 prev_forward; // xyz (unit); w = prev inv_h
    float4 prev_right;   // xyz pre-scaled; w = NEAR
    float4 prev_up;      // xyz pre-scaled; w = FAR
}

#define SCENE_EPS  (light_center.w)
#define AO_RADIUS  (light_u.w)
#define CAM_NEAR   (prev_right.w)
#define CAM_FAR    (prev_up.w)

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

// --- G-buffer pack (upscaler sessions; dlss.rs::GPixel on the GPU) ------------

// Primary-hit surface capture — the shade.rs::PrimarySurface mirror. spec_t:
// reflection-ray hit t, INF when the reflection missed, 0 when none was traced.
// direct_d/direct_s are the post-average direct-light lobes (lap 0 only; the
// two addends of color = kd*(direct_d + ambient) + direct_s) — assignment-only
// copies, zero rng draws, so the same-seed bit-identity gates hold.
struct PrimSurf {
    float3 n;      // shading normal, post face-flip
    float rough;
    float3 albedo; // raw material albedo (diffuse/specular split at the write)
    float metallic;
    float spec_t;
    float3 direct_d; // albedo-free direct diffuse (kd multiplies later)
    float3 direct_s; // direct specular incl. per-sample Fresnel
};

// dlss::GBufs re-hosted as one interleaved plane; the feed kernels fan it out
// into the upscalers' input textures. 80 B/px (GBUF_STRIDE in trace.rs —
// keep in lockstep).
struct GBufPx {
    float4 nr;    // normal.xyz, roughness
    float4 alb_z; // diff_alb.xyz = albedo*(1-metallic), view_z = t*dot(dir, forward)
    float4 spec;  // spec_alb.xyz = lerp(0.04, albedo, metallic) (RGB F0), spec_hit_t
    float4 mv;    // xy = motion vector in render-res pixels (y-down, current ->
                  // previous); z = prev-camera linear view-Z of the SAME hit
                  // point (FLAG_FSR_SIG — the denoiser MV's B channel differences
                  // it against alb_z.w; sky stores CAM_FAR so the delta is 0);
                  // else 0. w = 0.
    uint4 sig;    // f16x2 packs (dd.x|dd.y, dd.z|ds.x, ds.y|ds.z, 0) of the
                  // DEMODULATED FSR-RR signals — fsr::split_signals' twin,
                  // f16 IS the wire precision. FLAG_FSR_SIG; else 0.
};
RWStructuredBuffer<GBufPx> gbuf : register(u15);

// f32 -> f16 bits with round-to-nearest-even — NOT the f32tof16 intrinsic
// (the legacy DXIL op truncates). The CPU twin is half::f16::from_f32;
// nppd.hlsl's q16 round-trip and the sig packing below both build on it.
uint f16bits_rtne(float v) {
    uint x = asuint(v);
    uint s = (x >> 16u) & 0x8000u;
    x &= 0x7FFFFFFFu;
    uint h;
    if (x >= 0x47800000u) {            // >= 2^16: overflow -> inf; keep nan
        h = x > 0x7F800000u ? 0x7E00u : 0x7C00u;
    } else if (x >= 0x38800000u) {     // f16 normal range [2^-14, 2^16)
        h = (((x >> 23u) - 112u) << 10u) | ((x >> 13u) & 0x3FFu);
        uint rem = x & 0x1FFFu;
        // RTNE; a mantissa carry may bump the exponent, [65520, 65536)
        // correctly lands on inf.
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h += 1u;
    } else if (x >= 0x33000000u) {     // f16 subnormal range [2^-25, 2^-14)
        uint shift = 126u - (x >> 23u); // 14..24
        uint m = (x & 0x7FFFFFu) | 0x800000u;
        h = m >> shift;
        uint rem = m & ((1u << shift) - 1u);
        uint halfb = 1u << (shift - 1u);
        if (rem > halfb || (rem == halfb && (h & 1u) != 0u)) h += 1u;
    } else {                           // < 2^-25 underflows to signed zero
        h = 0u;
    }
    return s | h;
}

// fsr::f16_sat's twin: saturate into the finite f16 range (an inf on a signal
// plane turns the residual remainder into inf*0 = NaN downstream).
uint f16bits_sat(float v) { return f16bits_rtne(clamp(v, -65504.0, 65504.0)); }

uint pack_h2(float lo, float hi) { return f16bits_sat(lo) | (f16bits_sat(hi) << 16u); }

// fsr's GPU wire chain for the sqrt-encoded RGBA8 albedo planes: one explicit
// 8-bit quantization (fsr::sqrt_wire — the GPU pack stores f32, so unlike the
// CPU's albedo_wire there is no leading f16 rounding). Used identically at
// the sig demodulation here, the feed's residual, and the composite decode —
// consistency of those three sites IS the composite identity.
float sqrt_enc8(float v) { return floor(sqrt(saturate(v)) * 255.0 + 0.5) / 255.0; }
float sqrt_wire(float v) {
    float enc = sqrt_enc8(v);
    return enc * enc;
}
float3 sqrt_wire3(float3 v) { return float3(sqrt_wire(v.x), sqrt_wire(v.y), sqrt_wire(v.z)); }

// camera.rs::CamBasis::project against the PREVIOUS basis: the continuous
// image point a world direction passes through, y-down pixels. ok = false
// means at/behind the old image plane — consumed as mv (0,0) (disocclusion).
float2 project_prev(float3 d, out bool ok) {
    float df = dot(d, prev_forward.xyz);
    ok = df > 0.0;
    if (!ok) return float2(0.0, 0.0);
    float ndx = dot(d, prev_right.xyz) / (dot(prev_right.xyz, prev_right.xyz) * df);
    float ndy = dot(d, prev_up.xyz) / (dot(prev_up.xyz, prev_up.xyz) * df);
    return float2((ndx + 1.0) * 0.5 / prev_origin.w, (1.0 - ndy) * 0.5 / prev_forward.w);
}

float2 gbuf_mv(float3 d, float fx, float fy) {
    if ((flags & FLAG_HAS_PREV) == 0u) return float2(0.0, 0.0);
    bool ok;
    float2 p = project_prev(d, ok);
    return ok ? p - float2(fx, fy) : float2(0.0, 0.0);
}

// render.rs::write_gbuf_hit. (fx, fy) is the jittered sample position — the
// MV is unjittered by construction (both bases are jitter-free and the hit
// point lies on the jittered ray).
void gbuf_write_hit(uint pi, float fx, float fy, float3 dir, float t, PrimSurf ps) {
    if ((flags & FLAG_GBUF) == 0u) return;
    GBufPx g;
    g.nr = float4(ps.n, ps.rough);
    g.alb_z = float4(ps.albedo * (1.0 - ps.metallic), t * dot(dir, cam_forward.xyz));
    float3 spec_alb = lerp(float3(0.04, 0.04, 0.04), ps.albedo, ps.metallic);
    g.spec = float4(spec_alb, isinf(ps.spec_t) ? CAM_FAR : ps.spec_t);
    float3 hit_rel = cam_origin.xyz + dir * t - prev_origin.xyz;
    g.mv = float4(gbuf_mv(hit_rel, fx, fy), 0.0, 0.0);
    g.sig = uint4(0u, 0u, 0u, 0u);
    if (flags & FLAG_FSR_SIG) {
        // render.rs's fsr_buf fill: prev-camera linear view-Z of the SAME
        // hit point (no prev camera degrades to "no depth motion"); the
        // demodulation divides direct_s by the un-floored WIRE F0 —
        // fsr::split_signals with sqrt_wire in place of albedo_wire (the
        // pack stores f32; the wire quantization happens exactly once).
        g.mv.z = (flags & FLAG_HAS_PREV) ? dot(hit_rel, prev_forward.xyz) : g.alb_z.w;
        float3 sf0w = sqrt_wire3(spec_alb);
        float3 dd = ps.direct_d;
        float3 ds = ps.direct_s / max(sf0w, float3(1e-4, 1e-4, 1e-4));
        g.sig = uint4(pack_h2(dd.x, dd.y), pack_h2(dd.z, ds.x), pack_h2(ds.y, ds.z), 0u);
    }
    gbuf[pi] = g;
}

// render.rs::write_gbuf_sky: depth = far (finite, f16-safe); MV = direction-
// only reprojection (exact for an environment at infinity). FSR sessions:
// sig = 0 (residual = the sky color itself downstream) and prev-Z = far so
// the depth delta is exactly 0 — the sky contract --check-fsr pins.
void gbuf_write_sky(uint pi, float fx, float fy, float3 dir) {
    if ((flags & FLAG_GBUF) == 0u) return;
    GBufPx g;
    g.nr = float4(-dir, 1.0);
    g.alb_z = float4(1.0, 1.0, 1.0, CAM_FAR);
    g.spec = float4(0.0, 0.0, 0.0, 0.0);
    g.mv = float4(gbuf_mv(dir, fx, fy), 0.0, 0.0);
    g.sig = uint4(0u, 0u, 0u, 0u);
    if (flags & FLAG_FSR_SIG) {
        g.mv.z = CAM_FAR;
    }
    gbuf[pi] = g;
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

// --- Scene textures + UV stream (space1; the RP_SCENE_TEX table) --------------
// Self-contained here (NOT on shade.hlsli's declarations) because the cutout
// test runs inside the trace primitives — hemi_wave.hlsl calls occluded_q
// without pasting shade.hlsli. uv_indices/uv_tri_mat are second descriptors
// over the same indices/tri_mat resources as the t4/t5 space0 root SRVs.

StructuredBuffer<float2> uv_buf     : register(t0, space1); // Scene::texcoords
StructuredBuffer<uint>   uv_indices : register(t1, space1); // flat, 3 per tri
StructuredBuffer<uint>   uv_tri_mat : register(t2, space1); // material per tri
// Per-material cutout map: tex + 1 when textured AND alpha_masked, else 0
// (the bvh.rs::moller_trumbore gate chain folded to one fetch).
StructuredBuffer<uint>   mat_cutout : register(t3, space1);
// R8G8B8A8 (_SRGB for color maps, _UNORM for linear-data normal/rough-metal
// maps — Texture::srgb), carrying the FULL CPU-generated mip chain (built in
// texture.rs::build_mips, uploaded verbatim — CPU-trilinear parity: both
// samplers read identical texels at identical ray-cone lods).
Texture2D<float4>        texs[]     : register(t4, space1);
// Static: trilinear, repeat wrap — texture.rs::sample_trilinear in hardware
// (sRGB decode per texel via the SRGB SRV format, texel centers at i + 0.5;
// every sample passes an explicit ray-cone lod to SampleLevel, and lod <= 0
// reads level 0 only = the old bilinear). The cutout test below deliberately
// uses .Load instead — visibility never touches mips.
SamplerState             samp_lin   : register(s0, space1);
// Static: hardware anisotropic (MaxAnisotropy = the session's --aniso), fed
// the elliptical footprint through SampleGrad — texture.rs::sample_aniso in
// hardware. SampleLevel hands the TMU one scalar lod and no gradients, so an
// aniso filter THERE would be a silent no-op: the gradients are the feature.
SamplerState             samp_aniso : register(s1, space1);

// texture.rs::wrap — repeat into [0,1], non-finite collapses to 0 (fp can
// round c - floor(c) up to exactly 1.0 for c just below an integer; the
// consumer's min(w-1) clamp absorbs it, same as the CPU).
float uv_wrap(float c) {
    float f = c - floor(c);
    return isfinite(f) ? f : 0.0;
}

// scene.rs::tri_uv — barycentric UV interpolation at a hit.
float2 tri_uv(uint tri, float u, float v) {
    uint i0 = uv_indices[tri * 3u];
    uint i1 = uv_indices[tri * 3u + 1u];
    uint i2 = uv_indices[tri * 3u + 2u];
    return uv_buf[i0] * (1.0 - u - v) + uv_buf[i1] * u + uv_buf[i2] * v;
}

// bvh.rs::moller_trumbore's alpha-cutout test: true = reject the candidate.
// texture.rs::alpha_nearest mirrored in ALU over a .Load (not a sampler), so
// the RayQuery candidate loop and the DXR any-hit agree bit-for-bit with each
// other; `.a` of an SRGB SRV is linear per D3D spec, and alpha*255 < 127.5
// reproduces the CPU's `u8 < 128` through the UNORM n/255 representation.
bool alpha_cutout(uint tri, float u, float v) {
    uint cm = mat_cutout[uv_tri_mat[tri]];
    if (cm == 0u) return false;
    float2 uv = tri_uv(tri, u, v);
    uint ti = NonUniformResourceIndex(cm - 1u);
    uint w, h;
    texs[ti].GetDimensions(w, h);
    uint x = min(uint(uv_wrap(uv.x) * float(w)), w - 1u);
    uint y = min(uint(uv_wrap(uv.y) * float(h)), h - 1u);
    return texs[ti].Load(int3(int(x), int(y), 0)).a * 255.0 < 127.5;
}
