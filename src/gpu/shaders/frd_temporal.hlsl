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
    uint flags; // bit 0 = reset | 1 = firefly | 2 = vmotion | 3 = anti-lag | 4 = vm curvature
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
    // v1.5 specular virtual motion (oracle::spec_virtual_delta_px): the
    // prev world→clip matrix + this frame's CamBasis, so the pass can
    // unfold the reflection's virtual image and reproject it through the
    // PREVIOUS camera. Padding aligns the matrix to a register (root
    // constants map by declaration order — frd_gpu's cb() mirrors it).
    float3 _vm_pad;    // dwords 17-19
    float4x4 vm_m;     // world → PREV clip (glam columns; column-major mul)
    float3 cam_org;    // camera origin, world
    float3 cam_rgt;    // right · tan(fov/2) · aspect (CamBasis pre-scaling)
    float3 cam_up;     // up · tan(fov/2)
};

// The vm block's shared prev-camera projection: the virtual point at
// unfold distance t along the primary ray (t past the FAR_K threshold
// takes the rotation-only DIRECTION limit, w = 0 — the flat-sky arm; the
// never-exact-far threshold is the wire_cam_far f16 lesson). Returns
// (px, py, w) — callers test .z > 1e-6 (the humble arm) before consuming
// the coordinates.
float3 frd_vm_prev_px(float3 du, float ray_t, float t) {
    float4 pc = t >= FRD_VM_FAR_K * cam_far
        ? mul(vm_m, float4(du, 0.0))
        : mul(vm_m, float4(cam_org + du * (ray_t + t), 1.0));
    return float3(
        (pc.x / pc.w + 1.0) * 0.5 * float(rw), (1.0 - pc.y / pc.w) * 0.5 * float(rh), pc.w);
}

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
    // FIREFLY_K × its 8-NEIGHBOR ring mean BEFORE anything accumulates, so
    // one sun-glint outlier (the bridge's F0 demodulation amplifies specular
    // 25×+) can neither seed the slow history nor inflate the Welford m2
    // that sizes its OWN clamp box (the box-as-wide-as-the-outlier failure —
    // m2 sees the clamped sample automatically because frd_accumulate
    // computes it from x). THE CENTER IS EXCLUDED from the mean, and that
    // exclusion is the teeth: with the center in, a lone outlier's own
    // contribution dominates its cap (cap ≈ (K/9)·L asymptotically — an 11%
    // trim at K=8 however extreme the outlier), where the ring mean crushes
    // it to K× its surround while a REAL multi-pixel glint (whose ring
    // carries glint values) survives proportionally. Radiance lanes only —
    // .w is the hit-dist guide. Edge texels replicate (clamped coords).
    // Known-accept: a genuinely isolated 1-px highlight attenuates hard — at
    // 1 spp it is indistinguishable from a firefly, and the spatial passes
    // rebuild it from the ring.
    if ((flags & 2u) != 0u) {
        float md = 0.0, ms = 0.0;
        [unroll]
        for (int dy = -1; dy <= 1; dy++) {
            [unroll]
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int2 np = clamp(
                    int2(p) + int2(dx, dy), int2(0, 0), int2(int(rw) - 1, int(rh) - 1));
                md += in_diff[np].x;
                ms += in_spec[np].x;
            }
        }
        din.xyz *= frd_firefly_scale(din.x, md * (1.0 / 8.0));
        sin_.xyz *= frd_firefly_scale(sin_.x, ms * (1.0 / 8.0));
    }

    // Reprojection: the wire MV IS the fetch offset (prev − cur, pixels).
    float4 mv = in_mv[p];
    float2 q = float2(p) + 0.5 + mv.xy;
    float z_expect = z + mv.z;

    // v1.5 SPECULAR VIRTUAL MOTION (flag bit 2, FR_FRD_VMOTION=off is the
    // repro arm): a mirror pixel's CONTENT is the virtual image at
    // t_surf + t_r along the primary ray, which under camera TRANSLATION
    // moves at a different screen velocity than the surface (pure rotation
    // the surface MV already handles — both points share the ray through
    // the unmoved origin). The DELTA form — project the surface point AND
    // the virtual point through the SAME prev world→clip and add only
    // their difference to the measured-MV q — cancels jitter/convention
    // offsets to first order; a parked camera's sub-VM_DEADZONE2 fp
    // residue snaps to exactly 0 (keeping the integer-aligned-foot fast
    // path). t_r comes back off the wire's normalized hit-dist
    // (frd_hitdist_denorm_factor — exact below saturation; a saturated
    // big-far sky reflection underestimates t_r, degrading toward the
    // surface fetch: coarser, never wrong-signed). t_r ≥ VM_FAR_K·far
    // takes the rotation-only sky limit (project the DIRECTION, w = 0 —
    // never exact far, the f16 sky-compare lesson); roughness fades the
    // COORDINATE toward the surface (one fetch, never two); a projection
    // behind the prev image plane takes delta 0 — the humble arm.
    float2 q_spec = q;
    float vm_slack = 1.0;
    float vm_unc = 0.0;
    // The unfold's t_r comes off the ACCUMULATED hit-dist at the
    // surface-reprojected texel when one exists (the slow plane's EMA'd .w
    // — temporally stable), falling back to this frame's sample: a raw
    // 1-spp .w jitters per frame, and a fetch position that is a function
    // of per-frame noise fetches INCOHERENTLY under motion — measured live
    // as bright-reflection smear while strafing (each frame blends history
    // from a different offset).
    int2 qn = clamp(int2(floor(q)), int2(0, 0), int2(int(rw) - 1, int(rh) - 1));
    float nh_hist = prev_slow_spec[qn].w;
    float t_r = (nh_hist > 0.0 ? nh_hist : sin_.w) * frd_hitdist_denorm_factor(z, rough);
    float vm_w = 1.0 - smoothstep(FRD_VM_ROUGH_LO, FRD_VM_ROUGH_HI, rough);
    if ((flags & 4u) != 0u && t_r > 0.0 && vm_w > 0.0) {
        // v1.5.1/.2 CURVATURE (flag bit 4 = 16u; FR_FRD_CURV=off is the
        // flat-mirror repro arm): a convex mirror images its
        // reflection at the MIRROR-EQUATION distance, not the ray-traced
        // one — on the helmet the sun's image sits ~R/2 behind the
        // surface, so its screen motion rides the SURFACE, and the flat
        // sky arm's screen-pinned fetch re-painted every vacated pixel
        // with its own stale bright history (the strafe streak). v1.5.2:
        // κ comes from FOUR dead-zoned central differences of the DECODED
        // wire normals — 2px AND 4px baselines per axis (frd_vm_kappa:
        // per-axis MIN across scales kills normal-map bump spikes, the
        // LEADING-streak class; MAX across axes keeps cylinders; the
        // κ_lo/κ_hi bracket's projected fetch spread charges the history
        // cap below — unsure ⇒ short history, never a confident wrong
        // fetch). A constant field is ONE encoded word ⇒ Δn ≡ 0 bitwise
        // at every baseline ⇒ frd_virtual_dist's κ=0 branch ⇒ the flat
        // path verbatim — still water and F7's mirror are structurally
        // untouched.
        float t_v = t_r;
        float t_bhi = t_r; // the bracket's LONGER unfold (κ_lo)
        float t_blo = t_r; // the bracket's shorter unfold (κ_hi)
        if ((flags & 16u) != 0u) {
            int2 xm = int2(max(int(p.x) - 1, 0), int(p.y));
            int2 xp = int2(min(int(p.x) + 1, int(rw) - 1), int(p.y));
            int2 ym = int2(int(p.x), max(int(p.y) - 1, 0));
            int2 yp = int2(int(p.x), min(int(p.y) + 1, int(rh) - 1));
            int2 xm2 = int2(max(int(p.x) - 2, 0), int(p.y));
            int2 xp2 = int2(min(int(p.x) + 2, int(rw) - 1), int(p.y));
            int2 ym2 = int2(int(p.x), max(int(p.y) - 2, 0));
            int2 yp2 = int2(int(p.x), min(int(p.y) + 2, int(rh) - 1));
            float3 nxm, nxp, nym, nyp, nxm2, nxp2, nym2, nyp2;
            float rr_;
            frd_decode_nr(in_nr[xm], nxm, rr_);
            frd_decode_nr(in_nr[xp], nxp, rr_);
            frd_decode_nr(in_nr[ym], nym, rr_);
            frd_decode_nr(in_nr[yp], nyp, rr_);
            frd_decode_nr(in_nr[xm2], nxm2, rr_);
            frd_decode_nr(in_nr[xp2], nxp2, rr_);
            frd_decode_nr(in_nr[ym2], nym2, rr_);
            frd_decode_nr(in_nr[yp2], nyp2, rr_);
            float3 k3 = frd_vm_kappa(
                length(nxp - nxm),
                length(nxp2 - nxm2),
                length(nyp - nym),
                length(nyp2 - nym2),
                z,
                proj);
            t_v = frd_virtual_dist(t_r, k3.x);
            t_bhi = frd_virtual_dist(t_r, k3.y);
            t_blo = frd_virtual_dist(t_r, k3.z);
        }
        float ndx = (float(p.x) + 0.5) * (2.0 / float(rw)) - 1.0;
        float ndy = 1.0 - (float(p.y) + 0.5) * (2.0 / float(rh));
        float3 du = normalize(cam_fwd + cam_rgt * ndx + cam_up * ndy);
        float ray_t = z / dot(du, cam_fwd);
        // The sky test runs INSIDE frd_vm_prev_px on each unfold distance
        // — a heavily curved surface's "sky" reflection is a NEAR virtual
        // image (that re-route IS the helmet fix); the surface anchor is
        // the t = 0 point form.
        float3 sp3 = frd_vm_prev_px(du, ray_t, 0.0);
        float3 vp3 = frd_vm_prev_px(du, ray_t, t_v);
        if (sp3.z > 1e-6 && vp3.z > 1e-6) {
            float2 d = vp3.xy - sp3.xy;
            if (dot(d, d) < FRD_VM_DEADZONE2) {
                d = 0.0;
            }
            float2 appl = d * vm_w;
            q_spec = q + appl;
            vm_slack = 1.0 + FRD_VM_Z_SLACK * length(appl);
            // The uncertainty bracket: where the four κ estimates
            // disagree (bumpy visor, dead-zone-straddling close-ups,
            // cylinders' genuine ambiguity), the spread of their
            // projected fetches is real model uncertainty — charged into
            // the history cap as parallax, so NEITHER streak direction
            // can accumulate. A self-consistent dome pays ~nothing and
            // keeps its long, correctly-fetched history.
            if (t_bhi > t_blo) {
                float3 blo = frd_vm_prev_px(du, ray_t, t_blo);
                float3 bhi = frd_vm_prev_px(du, ray_t, t_bhi);
                if (blo.z > 1e-6 && bhi.z > 1e-6) {
                    vm_unc = length(bhi.xy - blo.xy) * vm_w / max(proj, 1e-4);
                }
            }
        }
    }
    bool spec_shared = all(q_spec == q);

    // Manual bilinear over the 4 feet, each validity-tested (a hardware
    // bilinear would blend across disocclusions). Two validity ladders:
    // diffuse (25 deg) and specular (roughness-tightened; a DISPLACED
    // virtual fetch runs its own loop below).
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
            float ws = spec_shared && nn > ncs ? b : 0.0;
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
    // The DISPLACED specular fetch (vm_live with a non-zero applied delta):
    // its own 4-foot loop at q_spec, the widened relative-Z test (see
    // vm_slack), and the spec normal ladder — the diffuse loop above ran
    // untouched at q, and when q_spec == q this loop never runs (the shared
    // loop carried spec, so the parked/rough/off arms are Stage A verbatim).
    if (!reset && !spec_shared) {
        float2 fq = q_spec - 0.5;
        int2 base = int2(floor(fq));
        float2 fr = fq - float2(base);
        float ncs = frd_ncos_thresh(rough, true);
        [unroll]
        for (uint k = 0; k < 4; k++) {
            int2 fp = base + int2(k & 1u, k >> 1);
            if (fp.x < 0 || fp.y < 0 || fp.x >= int(rw) || fp.y >= int(rh)) continue;
            float b = ((k & 1u) != 0u ? fr.x : 1.0 - fr.x) * ((k >> 1) != 0u ? fr.y : 1.0 - fr.y);
            if (b == 0.0) continue;
            float pz = prev_vz[fp];
            // "The virtual fetch must still land on the same reflector":
            // the foot's stored SURFACE z against our surface's expected
            // prev z, slack-widened by the applied offset.
            if (!frd_z_valid_slack(z_expect, pz, ndv, vm_slack)) continue;
            float3 pn;
            float pr;
            frd_decode_nr(prev_nr[fp], pn, pr);
            if (dot(n, pn) <= ncs) continue;
            hs_s += b * prev_slow_spec[fp];
            hf_s += b * prev_fast_spec[fp];
            nprev.y += b * prev_meta[fp].y;
            al.y += b * prev_antilag[fp].y;
            w_s += b;
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
    // g = 1/max(e,1) cuts the reprojected age — FLOORED at the history-fix
    // window (frd_antilag_apply: recover at the fast-history rate, never
    // restart — the unfloored multiply limit-cycled sharp converged
    // features, the F6 find), so the NEXT frame blends fast and the
    // ghost's rejection sticks. A no-valid-feet pixel has nprev 0, which
    // the apply maps to 0 — inert there by construction.
    float2 nfr = nprev * 63.0;
    if ((flags & 8u) != 0u) {
        nfr = float2(frd_antilag_apply(nfr.x, al.x), frd_antilag_apply(nfr.y, al.y));
    }

    // Accumulate. Meta stores n/63 in RG8 (cap 63 > every legal max_accum).
    Accum ad = frd_accumulate(din, hs_d, hf_d, nfr.x, w_d, max_accum);
    // The specular parallax cap: the crude cam_step/z translation term +
    // the light-motion term (light_par — 2× the sun's per-frame angular
    // delta; a parked camera under a TOD scrub has cam_step 0 while the
    // glint slides, and no MV can express content motion — the CAP is the
    // mechanism there) + the v1.5.2 curvature-uncertainty term (vm_unc —
    // see the bracket above). DELIBERATELY NOT the measured
    // surface-vs-virtual divergence: that is ALWAYS ≤ cam_step/z
    // (z_eff > z), so using it LENGTHENS history under translation exactly
    // in proportion to how much the planar-still-mirror unfold is trusted
    // — and rippled water / curved reflectors / hit-dist noise all break
    // that trust (measured live as a strafe smear regression). v1.5 is
    // the corrected FETCH under the SAME conservative cap; vm_unc only
    // ever TIGHTENS it; relaxing needs v2's model-confidence term.
    float parallax = cam_step / max(z, 1e-4) + light_par + vm_unc;
    float n_smax = frd_spec_max_frames(max_accum, parallax, rough);
    Accum as_ = frd_accumulate(sin_, hs_s, hf_s, nfr.y, w_s, n_smax);

    accum_diff[p] = ad.slow;
    accum_spec[p] = as_.slow;
    cur_fast_diff[p] = ad.fast;
    cur_fast_spec[p] = as_.fast;
    cur_meta[p] = float2(ad.n / 63.0, as_.n / 63.0);
}
