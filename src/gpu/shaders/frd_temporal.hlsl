// cs_frd_temporal — FRD pass 1 of 3: reprojection + disocclusion + slow/fast
// accumulation over the bridge's wire planes (all radiance stays in the
// wire's YCoCg space; cs_nrd_out decodes). Compiled as
// [FRD defines, frd_common.hlsli, this] by FrdGpu on its own root signature
// (the bloom pattern): one SRV table, one UAV table, root constants.
//
// Output goes to the TRANSIENT accum pair — pass 2 (cs_frd_blur) reads it;
// pass 3 (cs_frd_post) writes the recurrent slow feedback + the clamp. The
// n == 0 branch still writes accum = x BITWISE (never lerp(h, x, 1.0), which
// is h + (x−h)·1 — fp, not identity); the byte-identity control arm is now
// FrdGpu's force_passthrough (the F3 gate's hook — the spatial passes fire
// on reset frames BY DESIGN, that wide blur IS the history fix). Phase C
// pending inside this pass: hit-dist reconstruction, the 3x3 valid-foot
// fallback; phase D the fp16 arm + LDS tile. The firefly pre-clamp and the
// anti-lag brake are BUILT (2026-08-09, the bright-specular-smear fix —
// flags bits 1/3, --frd-[no-]anti-firefly / FR_FRD_ANTILAG=off), and the
// specular parallax carries the light-motion term (light_par — the
// parked-camera moving-sun case no MV can express).

// SRVs (prev fast/meta/geometry = the OTHER parity set; slow is
// SINGLE-buffered — pass 3 rewrites it after this pass reads it, the
// recurrent feedback).
Texture2D<float4> in_mv : register(t0);      // wire 2.5D MV (prev−cur px, z = prev_z−cur_z)
Texture2D<float4> in_nr : register(t1);      // wire enc-2 normal+roughness
Texture2D<float> in_viewz : register(t2);    // linear view-Z
Texture2D<float4> in_diff : register(t3);    // YCoCg + normHitDist
Texture2D<float4> in_spec : register(t4);
Texture2D<float4> prev_slow_diff : register(t5);
Texture2D<float4> prev_slow_spec : register(t6);
Texture2D<float4> prev_fast_diff : register(t7); // YCoCg + luma m2 (Welford)
Texture2D<float4> prev_fast_spec : register(t8);
Texture2D<float2> prev_meta : register(t9);      // n_diff/63, n_spec/63
Texture2D<float4> prev_nr : register(t10);       // last frame's wire nr snapshot
Texture2D<float> prev_vz : register(t11);        // last frame's view-Z snapshot
// The anti-lag plane (single-buffered, the slow discipline: pass 3 wrote it
// AFTER this pass read it last frame): per-signal g = 1/max(e, 1) from the
// clamp excess — the recurrent brake's feedback path (pass 3 has no meta
// UAV, so the gain rides its own plane).
Texture2D<float2> prev_antilag : register(t12);

// UAVs: the transient accum pair (pass 2's input) + THIS parity's history.
RWTexture2D<float4> accum_diff : register(u0);
RWTexture2D<float4> accum_spec : register(u1);
RWTexture2D<float4> cur_fast_diff : register(u2);
RWTexture2D<float4> cur_fast_spec : register(u3);
RWTexture2D<float2> cur_meta : register(u4);
RWTexture2D<float4> cur_nr : register(u5);
RWTexture2D<float> cur_vz : register(u6);

cbuffer FrdCb : register(b0) {
    uint rw;
    uint rh;
    uint flags; // bit 0 = history reset | 1 = firefly pre-clamp | 3 = anti-lag
    uint frame;
    float cam_far;
    float max_accum;   // tuning (frd.rs constants unless --frd-* moved them)
    float fast_frames;
    float clamp_sigma;
    float cam_step;    // |O_cur − O_prev| world units (specular parallax)
    float3 cam_fwd;    // world-space camera forward (the n·v proxy)
    // The blur passes' lanes, declared here only so light_par lands at the
    // shared dword-16 offset (root constants map by declaration order).
    float proj;
    float r_max;
    float pass_scale;
    uint salt;
    // 2× the light's per-frame angular delta, radians — the moving-sun
    // specular parallax (oracle::light_parallax; SUMS with cam_step/z, the
    // same radians-at-the-hit unit, so SPEC_PARALLAX_K applies unchanged).
    float light_par;
};

// One signal's temporal step — shared by diff/spec so the feet loop's
// geometry tests run once (the caller passes per-signal history + caps).
struct Accum {
    float4 slow;
    float4 fast;
    float n;
};

Accum frd_accumulate(
    float4 x, float4 hist_slow, float4 hist_fast, float n_prev, float conf, float n_max
) {
    Accum a;
    float n_eff = n_prev * saturate(conf);
    if (n_eff <= 0.0) {
        // The bitwise passthrough branch (reset / disocclusion / first
        // frame): OUT = IN exactly, history restarts on this sample.
        a.slow = x;
        a.fast = float4(x.xyz, 0.0);
        a.n = min(1.0, n_max);
        return a;
    }
    float alpha = 1.0 / (1.0 + n_eff);
    a.slow = lerp(hist_slow, x, alpha);
    float alpha_f = 1.0 / (1.0 + min(n_eff, fast_frames));
    float f_old = hist_fast.x;
    float3 fast_c = lerp(hist_fast.xyz, x.xyz, alpha_f);
    // Welford (oracle::welford_update): m2' = m2 + ((l−f_new)(l−f_old) − m2)·α
    // — NEVER E[l²]−E[l²] (fp16-catastrophic cancellation).
    float m2 = hist_fast.w + ((x.x - fast_c.x) * (x.x - f_old) - hist_fast.w) * alpha_f;
    a.fast = float4(fast_c, m2);
    a.n = min(n_eff + 1.0, n_max);
    // The fast-history clamp lives in pass 3 (cs_frd_post), applied to the
    // POST-BLURRED result — clamping here would box the pre-blur signal
    // twice per frame trip through the recurrent loop.
    return a;
}

[numthreads(FRD_GTX, FRD_GTY, 1)]
void cs_frd_temporal(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint2 p = id.xy;
    float z = in_viewz[p];
    float4 din = in_diff[p];
    float4 sin_ = in_spec[p];
    float4 nrw = in_nr[p];
    // Snapshots for the NEXT frame's disocclusion, sky included (a sky texel
    // must reject as prev geometry too — its stored z fails the relative
    // test against any hit).
    cur_nr[p] = nrw;
    cur_vz[p] = z;
    // Out-of-range (sky + the [0.999·far, far) band — the bridge's LOCKSTEP
    // predicate): cs_nrd_out passes accum through and never reads OUT here;
    // write the passthrough anyway so the planes stay deterministic.
    if (z >= FRD_RANGE_K * cam_far) {
        accum_diff[p] = din;
        accum_spec[p] = sin_;
        cur_fast_diff[p] = float4(din.xyz, 0.0);
        cur_fast_spec[p] = float4(sin_.xyz, 0.0);
        cur_meta[p] = float2(0.0, 0.0);
        return;
    }
    float3 n;
    float rough;
    frd_decode_nr(nrw, n, rough);
    // n·v proxy: the camera forward stands in for the per-pixel ray (within
    // fov/2 of it) — v points surface→camera = −fwd.
    float ndv = saturate(dot(n, -cam_fwd));

    // Firefly pre-clamp (oracle::firefly_scale, flag bit 1 — the
    // --frd-[no-]anti-firefly lever): soft-clamp each input's luma to
    // FIREFLY_K × its 3x3 neighborhood mean BEFORE anything accumulates, so
    // one sun-glint outlier (the bridge's F0 demodulation amplifies specular
    // 25×+) can neither seed the slow history nor inflate the Welford m2
    // that sizes its OWN clamp box (the box-as-wide-as-the-outlier failure —
    // m2 sees the clamped sample automatically because frd_accumulate
    // computes it from x). Radiance lanes only — .w is the hit-dist guide.
    // Edge texels replicate (clamped coords); the mean includes the center,
    // so an isolated glint beside pure sky clamps to (8/9)·luma at worst.
    if ((flags & 2u) != 0u) {
        float md = 0.0, ms = 0.0;
        [unroll]
        for (int dy = -1; dy <= 1; dy++) {
            [unroll]
            for (int dx = -1; dx <= 1; dx++) {
                int2 np = clamp(
                    int2(p) + int2(dx, dy), int2(0, 0), int2(int(rw) - 1, int(rh) - 1));
                md += in_diff[np].x;
                ms += in_spec[np].x;
            }
        }
        din.xyz *= frd_firefly_scale(din.x, md * (1.0 / 9.0));
        sin_.xyz *= frd_firefly_scale(sin_.x, ms * (1.0 / 9.0));
    }

    // Reprojection: the wire MV IS the fetch offset (prev − cur, pixels).
    float4 mv = in_mv[p];
    float2 q = float2(p) + 0.5 + mv.xy;
    float z_expect = z + mv.z;

    // Manual bilinear over the 4 feet, each validity-tested (a hardware
    // bilinear would blend across disocclusions). Two validity ladders:
    // diffuse (25 deg) and specular (roughness-tightened).
    float4 hs_d = 0.0, hs_s = 0.0, hf_d = 0.0, hf_s = 0.0;
    float2 nprev = 0.0;
    float2 al = 0.0;
    float w_d = 0.0, w_s = 0.0;
    bool reset = (flags & 1u) != 0u;
    if (!reset) {
        float2 fq = q - 0.5;
        int2 base = int2(floor(fq));
        float2 fr = fq - float2(base);
        float ncd = frd_ncos_thresh(rough, false);
        float ncs = frd_ncos_thresh(rough, true);
        [unroll]
        for (uint k = 0; k < 4; k++) {
            int2 fp = base + int2(k & 1u, k >> 1);
            if (fp.x < 0 || fp.y < 0 || fp.x >= int(rw) || fp.y >= int(rh)) continue;
            float b = ((k & 1u) != 0u ? fr.x : 1.0 - fr.x) * ((k >> 1) != 0u ? fr.y : 1.0 - fr.y);
            // Order is cost, not math: a zero-weight foot (integer-aligned
            // reprojection — a PARKED camera hits 3 of 4) and a z-rejected
            // foot (disocclusion) contribute exactly nothing, so both tests
            // run before the prev_nr load + decode (the loop's one rsqrt).
            if (b == 0.0) continue;
            float pz = prev_vz[fp];
            if (!frd_z_valid(z_expect, pz, ndv)) continue;
            float3 pn;
            float pr;
            frd_decode_nr(prev_nr[fp], pn, pr);
            float nn = dot(n, pn);
            float wd = nn > ncd ? b : 0.0;
            float ws = nn > ncs ? b : 0.0;
            // The anti-lag plane rides the same validated feet as meta
            // (one float2 load serves both signals).
            float2 alv = prev_antilag[fp];
            if (wd > 0.0) {
                hs_d += wd * prev_slow_diff[fp];
                hf_d += wd * prev_fast_diff[fp];
                nprev.x += wd * prev_meta[fp].x;
                al.x += wd * alv.x;
                w_d += wd;
            }
            if (ws > 0.0) {
                hs_s += ws * prev_slow_spec[fp];
                hf_s += ws * prev_fast_spec[fp];
                nprev.y += ws * prev_meta[fp].y;
                al.y += ws * alv.y;
                w_s += ws;
            }
        }
    }
    if (w_d > 1e-4) {
        hs_d /= w_d;
        hf_d /= w_d;
        nprev.x /= w_d;
        al.x /= w_d;
    }
    if (w_s > 1e-4) {
        hs_s /= w_s;
        hf_s /= w_s;
        nprev.y /= w_s;
        al.y /= w_s;
    }
    // Anti-lag (flag bit 3, FR_FRD_ANTILAG=off is the repro arm): where
    // pass 3's clamp had to move a signal e box-widths, its stored
    // g = 1/max(e,1) cuts the reprojected age — antilag_frames(n, e) ==
    // n·g exactly (the F0 identity), so the NEXT frame blends fast and the
    // ghost's rejection sticks. A no-valid-feet pixel leaves al at 0 with
    // nprev already 0 — the multiply is inert there by construction.
    if ((flags & 8u) != 0u) {
        nprev *= al;
    }

    // Accumulate. Meta stores n/63 in RG8 (cap 63 > every legal max_accum).
    Accum ad = frd_accumulate(din, hs_d, hf_d, nprev.x * 63.0, w_d, max_accum);
    // Camera-translation parallax + the light-motion term (light_par —
    // 2× the sun's per-frame angular delta): a parked camera under a TOD
    // scrub has cam_step 0 while the glint slides, and no MV of any kind
    // can express content motion — the CAP is the mechanism there.
    float parallax = cam_step / max(z, 1e-4) + light_par;
    float n_smax = frd_spec_max_frames(max_accum, parallax, rough);
    Accum as_ = frd_accumulate(sin_, hs_s, hf_s, nprev.y * 63.0, w_s, n_smax);

    accum_diff[p] = ad.slow;
    accum_spec[p] = as_.slow;
    cur_fast_diff[p] = ad.fast;
    cur_fast_spec[p] = as_.fast;
    cur_meta[p] = float2(ad.n / 63.0, as_.n / 63.0);
}
