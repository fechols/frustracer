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
// meta RGBA8: .rg = n_diff/63, n_spec/63 | .ba = the stabilization ages
// n_stab_diff/63, n_stab_spec/63 (0 with stabilization off).
Texture2D<float4> prev_meta : register(t9);
Texture2D<float4> prev_nr : register(t10);       // last frame's wire nr snapshot
Texture2D<float> prev_vz : register(t11);        // last frame's view-Z snapshot
// The anti-lag plane (single-buffered, the slow discipline: pass 3 wrote it
// AFTER this pass read it last frame): per-signal g = 1/max(e, 1) from the
// clamp excess — the recurrent brake's feedback path (pass 3 has no meta
// UAV, so the gain rides its own plane).
Texture2D<float2> prev_antilag : register(t12);
// The stabilization history IS last frame's OUT planes (nothing writes them
// between FRD frames — cs_nrd_out only reads; single-buffered, the slow
// discipline: pass 3 rewrites them after this pass reads). Only consulted
// under stab_max > 0.
Texture2D<float4> prev_out_diff : register(t13);
Texture2D<float4> prev_out_spec : register(t14);

// UAVs: the transient accum pair (pass 2's input) + THIS parity's history.
RWTexture2D<float4> accum_diff : register(u0);
RWTexture2D<float4> accum_spec : register(u1);
RWTexture2D<float4> cur_fast_diff : register(u2);
RWTexture2D<float4> cur_fast_spec : register(u3);
RWTexture2D<float4> cur_meta : register(u4);
RWTexture2D<float4> cur_nr : register(u5);
RWTexture2D<float> cur_vz : register(u6);
// The reprojected stabilization history (pass 3's t7/t8): written only
// under stab_max > 0 — the ONE deliberately non-deterministic plane pair
// (unread by construction when unwritten: pass 3's blend keys on the meta
// stab age, which is 0/reseed exactly where the gather didn't run).
RWTexture2D<float4> stab_out_diff : register(u7);
RWTexture2D<float4> stab_out_spec : register(u8);

cbuffer FrdCb : register(b0) {
    uint rw;
    uint rh;
    uint flags; // bit 0 = reset | 1 = firefly | 2 = vmotion | 3 = anti-lag | 4 = vm curvature
                // | 5 = GI fold live (diffuse firefly/tap-guard relax)
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
    // vm_scale = the v1.5.3 κ-baseline scale (oracle::vm_baseline_scale
    // of the render height; FR_FRD_VMSCALE overrides host-side) — an
    // integer-valued float, 1.0 at every ≤~800p gate/lab res.
    float vm_scale;    // dword 17
    // --frd-max-stab-frames (0 = the stabilization sub-step off — the
    // bitwise pre-stab arm; rode the matrix-alignment pad, no layout shift).
    float stab_max;    // dword 18
    float _pad19;      // dword 19
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
        cur_meta[p] = float4(0.0, 0.0, 0.0, 0.0);
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
    // THE GI-FOLD EXEMPTION (flag bit 5, FR_FRD_GIRELAX=off is the repro
    // arm): when the diffuse lane carries the folded RTGI bounce
    // (shade.hlsli's FLAG_NRD_GI add), a sparse 1-spp emissive/bounce hit IS
    // a lone outlier against a dim ring — the exact shape this clamp exists
    // to crush — and a fixed K can never pass it (a sparse signal with hit
    // probability p needs K >= 1/p to integrate unbiased). Diffuse therefore
    // rides UNCLAMPED under the fold; measured (bistro night, the emissive
    // QA campaign): the clamp cost 90%+ of the bounce-emissive pool energy.
    // SPECULAR is untouched — the sun-glint outliers the v1.5.x campaigns
    // fought are specular (F0-demodulated mirror chains), and diffuse
    // magnitudes are albedo-and-cosine-bounded so the trail class can't
    // reach it. The skip is a BRANCH, never a computed 1.0 — bitwise-inert
    // with the bit clear.
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
        if ((flags & 32u) == 0u)
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
            // v1.5.3: offsets scale with the render height (±vs/±2vs
            // texels = the same WORLD footprint at every res) so the
            // dead-zoned |Δn| the DZ was sized against is res-invariant;
            // vs = 1 at every gate/lab res is the bitwise pre-v1.5.3 path.
            int vs = max(int(vm_scale), 1);
            int2 xm = int2(max(int(p.x) - vs, 0), int(p.y));
            int2 xp = int2(min(int(p.x) + vs, int(rw) - 1), int(p.y));
            int2 ym = int2(int(p.x), max(int(p.y) - vs, 0));
            int2 yp = int2(int(p.x), min(int(p.y) + vs, int(rh) - 1));
            int2 xm2 = int2(max(int(p.x) - 2 * vs, 0), int(p.y));
            int2 xp2 = int2(min(int(p.x) + 2 * vs, int(rw) - 1), int(p.y));
            int2 ym2 = int2(int(p.x), max(int(p.y) - 2 * vs, 0));
            int2 yp2 = int2(int(p.x), min(int(p.y) + 2 * vs, int(rh) - 1));
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
                proj,
                float(vs));
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
            // v1.5.5 — the MAGNITUDE FADE (the FG round-4 dt-fade shape,
            // res-invariant in fractions of rh): fetch error is
            // multiplicative in the offset, so past FADE_HI·rh the virtual
            // fetch collapses onto the SURFACE fetch — the arm
            // FR_FRD_VMOTION=off measured CLEAN at max strafe while the
            // uncontained vm fetch painted the dome-wide blotch. Below
            // FADE_LO·rh the weight is exactly 1.0 (the validated
            // normal-speed regime rides the multiply-by-1 bitwise).
            float dl = length(d) * vm_w;
            float w_mag =
                1.0 - smoothstep(FRD_VM_FADE_LO * rh, FRD_VM_FADE_HI * rh, dl);
            float2 appl = d * (vm_w * w_mag);
            q_spec = q + appl;
            float alen = length(appl);
            // v1.5.5 — the MAX-SPEED containment pair (the B70 1-spp
            // max-strafe blotch: at 300-800 px of applied offset the
            // uncapped slack widened the z-test 16-40x — vacuous — so the
            // displaced fetch ACCEPTED bright glint history anywhere and
            // the recurrence compounded it into a dome-wide blob;
            // FR_FRD_VMOTION=off measured clean at the same speed, so the
            // surface fetch is the proven landing zone). (a) The slack CAP:
            // the grazing-planar-mirror rationale only needs the widening
            // at moderate offsets — beyond FRD_VM_SLACK_CAP px the fetch
            // must pass a near-normal z-test, so a wrong far-displaced
            // fetch history-restarts instead of being believed (below the
            // cap this line is the old expression BITWISE). (b) The
            // magnitude charge below: fetch error is MULTIPLICATIVE in the
            // offset (a relative model error δ ⇒ δ·|appl| px per frame),
            // so |appl| itself joins the history cap as parallax — the
            // third installment of the v2 confidence term. Both arms only
            // ever TIGHTEN (shorter history / stricter test — coarser,
            // never a confident wrong fetch). Twins: frd.rs
            // vm_slack_for/vm_appl_unc — change all together.
            vm_slack = 1.0 + FRD_VM_Z_SLACK * min(alen, FRD_VM_SLACK_CAP);
            vm_unc = FRD_VM_APPL_UNC_K * alen / max(proj, 1e-4);
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
                    vm_unc += length(bhi.xy - blo.xy) * vm_w / max(proj, 1e-4);
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
    // Stabilization gather (stab_max > 0 only): last frame's OUT through
    // the SAME validated feet, + the stab ages off meta .ba.
    float4 so_d = 0.0, so_s = 0.0;
    float2 nsprev = 0.0;
    bool stab_on = stab_max > 0.0;
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
                if (stab_on) {
                    so_d += wd * prev_out_diff[fp];
                    nsprev.x += wd * prev_meta[fp].z;
                }
                w_d += wd;
            }
            if (ws > 0.0) {
                hs_s += ws * prev_slow_spec[fp];
                hf_s += ws * prev_fast_spec[fp];
                nprev.y += ws * prev_meta[fp].y;
                al.y += ws * alv.y;
                if (stab_on) {
                    so_s += ws * prev_out_spec[fp];
                    nsprev.y += ws * prev_meta[fp].w;
                }
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
            if (stab_on) {
                so_s += b * prev_out_spec[fp];
                nsprev.y += b * prev_meta[fp].w;
            }
            w_s += b;
        }
    }
    if (w_d > 1e-4) {
        hs_d /= w_d;
        hf_d /= w_d;
        nprev.x /= w_d;
        al.x /= w_d;
        if (stab_on) {
            so_d /= w_d;
            nsprev.x /= w_d;
        }
    }
    if (w_s > 1e-4) {
        hs_s /= w_s;
        hf_s /= w_s;
        nprev.y /= w_s;
        al.y /= w_s;
        if (stab_on) {
            so_s /= w_s;
            nsprev.y /= w_s;
        }
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
    // v1.5.5(c) — the SMEAR BUDGET (oracle::spec_budget_frames, in
    // lockstep): the rate cap above bounds frames-per-radian while the
    // visible artifact is frames × per-frame glint motion in PIXELS, so at
    // max strafe (~0.18·rh/frame) even 5-8 frames paint a dome-wide swath
    // — and on a smooth dome every fetch passes the normal/z ladders, so
    // the cap is the only bound. n also obeys BUDGET·rh over the motion:
    // restart-class at max strafe, inert wherever the rate cap is already
    // the binding one (parked, TOD scrubs, slow strafes).
    n_smax = min(n_smax,
                 max(FRD_SPEC_PX_BUDGET * rh
                         / max(max(parallax, 0.0) * max(proj, 0.0), 1e-4),
                     1.0));
    Accum as_ = frd_accumulate(sin_, hs_s, hf_s, nfr.y, w_s, n_smax);

    accum_diff[p] = ad.slow;
    accum_spec[p] = as_.slow;
    cur_fast_diff[p] = ad.fast;
    cur_fast_spec[p] = as_.fast;
    // Stabilization ages (meta .ba): advance where the gather validated
    // feet, RESEED at 1 on reset/no-feet — this frame's out then seeds next
    // frame's blend, and pass 3's blend keys on the REPROJECTED age (its
    // meta lane − 1 ≤ 0 on a reseed frame), so the unwritten/garbage
    // transient is structurally unread. stab_max == 0 writes 0 lanes and
    // never touches the transients — the bitwise-off arm.
    float2 ns_new = float2(0.0, 0.0);
    if (stab_on) {
        float nsd = !reset && w_d > 1e-4 ? min(nsprev.x * 63.0 + 1.0, 63.0) : 1.0;
        float nss = !reset && w_s > 1e-4 ? min(nsprev.y * 63.0 + 1.0, 63.0) : 1.0;
        ns_new = float2(nsd / 63.0, nss / 63.0);
        stab_out_diff[p] = so_d;
        stab_out_spec[p] = so_s;
    }
    cur_meta[p] = float4(ad.n / 63.0, as_.n / 63.0, ns_new.x, ns_new.y);
}
