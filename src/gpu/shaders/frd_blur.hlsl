// FRD passes 2 + 3 (cs_frd_blur / cs_frd_post) — the adaptive spatial
// blurs. One Vogel-disk pass each over a hit-distance-driven, accumulation-
// scaled radius with bilateral rejection; the HISTORY FIX is folded in as a
// radius policy (n < N_FIX boosts toward R_FIX — a wide relaxed blur on
// disoccluded pixels replaces the published separate pass). cs_frd_post
// additionally runs at pass_scale x the radius with its own rotation salt,
// applies the fast-history clamp, and writes the RECURRENT slow feedback —
// pass 3's output IS the history pass 1 reprojects next frame, which is what
// lets a ~30-frame cap converge (each frame's blur compounds into the
// accumulated footprint — the self-stabilizing recurrent design).
//
// Tap rotation is a pure integer hash of (pixel, frame, salt) — the clouds
// dither discipline: deterministic, ZERO rng draws, and the per-frame term
// is what the recurrent feedback integrates (different taps every frame).

// Pass 2 SRVs.
Texture2D<float4> b_src_diff : register(t0); // pass 2: accum | pass 3: blur
Texture2D<float4> b_src_spec : register(t1);
Texture2D<float4> b_nr : register(t2);       // wire enc-2 (this frame's)
Texture2D<float> b_viewz : register(t3);
Texture2D<float2> b_meta : register(t4);     // THIS frame's n (pass 1's write)
// Pass 3 only (the clamp's fast history — this frame's, pass 1's write).
Texture2D<float4> b_fast_diff : register(t5);
Texture2D<float4> b_fast_spec : register(t6);

RWTexture2D<float4> b_out_diff : register(u0); // pass 2: blur | pass 3: wire OUT
RWTexture2D<float4> b_out_spec : register(u1);
// Pass 3 only: the recurrent slow feedback.
RWTexture2D<float4> b_slow_diff : register(u2);
RWTexture2D<float4> b_slow_spec : register(u3);
// Pass 3 only: per-signal anti-lag gains g = 1/max(e, 1) for NEXT frame's
// pass 1 (single-buffered, the slow discipline — pass 1 read it before this
// pass rewrites it). Clamp-not-run and sky pixels store exactly 1.0.
RWTexture2D<float2> b_antilag : register(u4);

cbuffer FrdCb : register(b0) {
    uint rw;
    uint rh;
    uint flags;
    uint frame;
    float cam_far;
    float max_accum;
    float fast_frames;
    float clamp_sigma;
    float cam_step;
    float3 cam_fwd;
    float proj;       // world→pixel: r_px = (r_world / z) · proj
    float r_max;      // --frd-blur-radius (default frd.rs R_MAX)
    float pass_scale; // 1.0 (blur) | 1.7 (post)
    uint salt;        // disk rotation salt (decorrelates the two passes)
    float light_par;  // pass 1's lane (declared for the shared layout)
};

// One signal's disk blur, weights in log2 domain (frd_tap_exp2 — one
// log2+exp2 per tap, the B70 diet). The center tap carries weight 1 so a
// zero-neighbor pixel degrades to passthrough (the kernel is total).
// `inv_h` = 0 disables the spec hit-dist term (a diffuse tap's v.w then
// contributes exactly 0 — the wire keeps it finite).
float4 frd_disk(
    Texture2D<float4> src, uint2 p, float4 center, float radius, float z, float3 n,
    float pw, float inv_z, float inv_h, float cr, float sr
) {
    if (radius < 0.5) {
        return center;
    }
    float4 sum = center;
    float wsum = 1.0;
    [unroll]
    for (uint i = 0; i < FRD_TAPS; i++) {
        float2 off = frd_vogel_rot(i, cr, sr) * radius;
        int2 q = int2(floor(float2(p) + 0.5 + off));
        if (q.x < 0 || q.y < 0 || q.x >= int(rw) || q.y >= int(rh)) continue;
        float zq = b_viewz[q];
        if (zq >= FRD_RANGE_K * cam_far) continue; // never blend sky into geometry
        float3 nq;
        float rq;
        frd_decode_nr(b_nr[q], nq, rq);
        float4 v = src[q];
        float w = exp2(frd_tap_exp2(
            i, abs(zq - z), inv_z, dot(n, nq), pw, abs(v.w - center.w), inv_h));
        sum += w * v;
        wsum += w;
    }
    return sum / wsum;
}

// Shared per-pixel prologue: geometry + the two radii + the hoisted
// per-pixel weight constants (the tap loop's exponent inputs). Takes the
// kernels' already-loaded center values — never re-loads them.
struct BlurCtx {
    float z;
    float3 n;
    float rough;
    float r_d;
    float r_s;
    float pw_s;  // spec normal exponent (roughness-driven, per pixel)
    float inv_z; // z-term scale (frd_inv_z_scale)
    float2 nn;   // meta n (diff, spec)
    bool sky;
};

BlurCtx frd_blur_ctx(uint2 p, float4 ad, float4 as_) {
    BlurCtx c;
    c.z = b_viewz[p];
    c.sky = c.z >= FRD_RANGE_K * cam_far;
    c.n = float3(0.0, 1.0, 0.0);
    c.rough = 1.0;
    c.r_d = 0.0;
    c.r_s = 0.0;
    c.pw_s = FRD_N_POW_DIFF;
    c.inv_z = 0.0;
    c.nn = float2(0.0, 0.0);
    if (c.sky) return c;
    frd_decode_nr(b_nr[p], c.n, c.rough);
    c.nn = b_meta[p] * 63.0;
    float hd = ad.w * frd_hitdist_denorm_factor(c.z, 1.0);
    float hs = as_.w * frd_hitdist_denorm_factor(c.z, c.rough);
    c.r_d = frd_history_fix(
        frd_radius_diffuse(hd, c.z, proj, r_max) * frd_accum_scale(c.nn.x), c.nn.x);
    c.r_s = frd_history_fix(
        frd_radius_spec(hs, c.z, c.rough, proj, r_max) * frd_accum_scale(c.nn.y), c.nn.y);
    c.pw_s = frd_normal_pow(c.rough, true);
    c.inv_z = frd_inv_z_scale(c.z);
    return c;
}

[numthreads(FRD_GBX, FRD_GBY, 1)]
void cs_frd_blur(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint2 p = id.xy;
    float4 ad = b_src_diff[p];
    float4 as_ = b_src_spec[p];
    BlurCtx c = frd_blur_ctx(p, ad, as_);
    if (c.sky) {
        b_out_diff[p] = ad;
        b_out_spec[p] = as_;
        return;
    }
    float sr, cr;
    sincos(frd_disk_rot(p, frame, salt), sr, cr);
    b_out_diff[p] = frd_disk(
        b_src_diff, p, ad, c.r_d * pass_scale, c.z, c.n, FRD_N_POW_DIFF, c.inv_z, 0.0, cr, sr);
    b_out_spec[p] = frd_disk(
        b_src_spec, p, as_, c.r_s * pass_scale, c.z, c.n, c.pw_s, c.inv_z,
        frd_inv_h_scale(as_.w), cr, sr);
}

[numthreads(FRD_GBX, FRD_GBY, 1)]
void cs_frd_post(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint2 p = id.xy;
    float4 ad = b_src_diff[p];
    float4 as_ = b_src_spec[p];
    BlurCtx c = frd_blur_ctx(p, ad, as_);
    if (c.sky) {
        b_out_diff[p] = ad;
        b_out_spec[p] = as_;
        b_slow_diff[p] = ad;
        b_slow_spec[p] = as_;
        b_antilag[p] = float2(1.0, 1.0);
        return;
    }
    float sr, cr;
    sincos(frd_disk_rot(p, frame, salt), sr, cr);
    float4 od = frd_disk(
        b_src_diff, p, ad, c.r_d * pass_scale, c.z, c.n, FRD_N_POW_DIFF, c.inv_z, 0.0, cr, sr);
    float4 os = frd_disk(
        b_src_spec, p, as_, c.r_s * pass_scale, c.z, c.n, c.pw_s, c.inv_z,
        frd_inv_h_scale(as_.w), cr, sr);
    // Fast-history clamp, once the fast EMA is trustworthy (n past the fast
    // window): box the radiance lanes to fast ± kσ (the .w hit-dist lane is
    // not radiance — never clamped). The clamped value is ALSO what feeds
    // back — a ghost the clamp rejected must not survive in the recurrence.
    // Beside each clamp, RECORD how hard it had to fight: e = the pre-clamp
    // luma's distance from fast in box widths, g = frd_antilag_gain(e) into
    // the antilag plane — next frame's pass 1 cuts the reprojected age by
    // g, the recurrent brake (in-box and clamp-not-run store exactly 1.0).
    float2 g = float2(1.0, 1.0);
    float4 fd = b_fast_diff[p];
    float4 fs = b_fast_spec[p];
    if (c.nn.x >= fast_frames) {
        float sig = sqrt(max(fd.w, 0.0));
        g.x = frd_antilag_gain(frd_antilag_excess(abs(od.x - fd.x), clamp_sigma * sig, fd.x));
        od.xyz = frd_clamp_ycocg(od.xyz, fd.xyz, sig, clamp_sigma);
    }
    if (c.nn.y >= fast_frames) {
        float sig = sqrt(max(fs.w, 0.0));
        g.y = frd_antilag_gain(frd_antilag_excess(abs(os.x - fs.x), clamp_sigma * sig, fs.x));
        os.xyz = frd_clamp_ycocg(os.xyz, fs.xyz, sig, clamp_sigma);
    }
    b_out_diff[p] = od;
    b_out_spec[p] = os;
    b_slow_diff[p] = od;
    b_slow_spec[p] = os;
    b_antilag[p] = g;
}
