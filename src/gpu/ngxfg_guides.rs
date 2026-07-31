//! FG guide conversion for the raw-NGX DLSS-G evaluate (`--fg` DLSS sessions,
//! `gpu/mod.rs::ngxfg_dispatch`): ONE fused compute pass producing the two
//! planes the FG snippet needs but the RR plane set does not carry —
//!
//! 1. **Clip depth** (round 1): DLSS-RR consumes the depth plane through the
//!    RR-specific LINEAR-depth tag, so it holds unbounded linear view-Z
//!    (0..far). The FG snippet's plain `Depth` input has DLSS-SR's contract —
//!    a [0,1] depth buffer CONSISTENT WITH THE SUPPLIED CAMERA MATRICES
//!    (ProgrammingGuideDLSS_G.md §5.1) — and dlssg-to-fsr3 confirms real SL
//!    titles feed a projection-matrix-consistent depth buffer (it derives
//!    near/far back out of the matrix). The mapping here is the EXACT
//!    z-mapping of the `Mat4::perspective_lh` matrix riding the same
//!    dispatch: `d = A + B/z`, `A = far/(far−near)`, `B = −near·far/(far−near)`
//!    — near → 0, sky (`z = far`) → 1, `depthInverted` 0. Deliberately NOT
//!    `xess::view_z_to_clip_depth` (REVERSED-Z, inconsistent with these
//!    matrices).
//!
//! 2. **Reflection-aware motion vectors** (round 2 — the DamagedHelmet
//!    sky-reflection swim): the MV plane describes SURFACE motion, but a
//!    mirror pixel's CONTENT is the reflection — a VIRTUAL IMAGE at path
//!    depth `t_surface + t_reflection` (planar unfold: the image lies along
//!    the primary ray, beyond the surface; reflected sky ⇒ `t_v ≈ far` ⇒
//!    near-zero translation parallax — exactly the "reflection drifts
//!    opposite the surface" the user observed strafing). Warping with the
//!    surface MV drags the reflection with the helmet on every generated
//!    frame; real frames snap it back = swimming at half the present rate.
//!    We can compute the virtual-image MV EXACTLY because the reflection
//!    ray's hit distance is already captured per pixel (`spec_hit_t`, the RR
//!    guide plane; 0 = no reflection ray, `far` = reflected sky — pinned in
//!    `shade.rs::PrimarySurface::spec_t`). The output is
//!    `lerp(mv_surface, mv_virtual, w)` with `w` = how much of the pixel's
//!    LOOK is the reflection: `lum(spec_alb)/(lum(diff_alb)+lum(spec_alb))`
//!    damped above `ROUGH_LO..ROUGH_HI` roughness — on the metal helmet
//!    diffuse ≈ 0 so w ≈ 1. (True RADIANCE-weighted w — the user's exact
//!    formulation, which would also catch a blinding glint on a glossy
//!    dielectric — needs a dd/ds/ind_s-style capture armed in DLSS sessions,
//!    the FLAG_FSR_SIG precedent: documented follow-on, not v1.) The blended
//!    plane is FG-ONLY — RR keeps its own MV plane untouched (RR is trained
//!    for surface MVs + the spec-hit guide).
//!
//! 3. **Firefly glow motion vectors** (round 3 — the night-swarm strobe on
//!    generated frames): the glow splat is a color-only add applied AFTER the
//!    G-buffer capture (render.rs `shade_traced`, leaf/sky/dxr twins), so a
//!    glow-dominated pixel carries the MV of the surface (or sky) BEHIND it
//!    while its bright CONTENT translates several pixels per rendered frame
//!    (`fireflies::pose` on the shared cloud clock). Worse, on smooth/metal
//!    pixels round 2's material-driven weight confidently hands FG the
//!    virtual-reflection MV at exactly those pixels. Poses are CLOSED-FORM,
//!    so the fix is exact: the CPU bakes per-firefly SCREEN-SPACE splat rows
//!    (`ff_guide_rows` — current pixel, prev-frame pixel through the same
//!    world→prev-clip matrix `GuideParams` carries, view-Z, sigma in pixels,
//!    center luminance) and the kernel lerps toward the firefly's own
//!    `mv_i = prev_px − cur_px` where glow luminance dominates
//!    (`w_ff = S/(S + FF_MV_L_REF)`). The MV is CONSTANT across a splat
//!    (rigid translation — exact at the center, and the right first-order
//!    warp for a translating blob; per-pixel reprojection would contract the
//!    splat toward a point). The weight is analytic, never an accum read: a
//!    1-spp color denominator is stochastic and would flicker the MV plane
//!    frame to frame (the round-2 radiance-weighted-w follow-on note applies
//!    here too). Splat rows pair prev↔cur BY INDEX; a count mismatch (first
//!    armed frame, a live count edit) reprojects the CURRENT pose instead —
//!    camera-motion-only MV, coarser never wrong-signed. `ffc == 0` (day /
//!    `--no-fireflies` / FR_NGXFG_FFMV=off) skips the whole block: the
//!    pre-round-3 instruction stream bit-identically. Known-accepts:
//!    firefly SPECULAR highlights on surfaces still ride the surface/virtual
//!    blend (highlight motion is half-vector geometry); RR itself still sees
//!    no firefly MVs (its plane is untouched by contract); ffx/XeSS-FG
//!    families unchanged.
//!
//! Compiled with D3DCompile (fxc, cs_5_0) like `gpu/bloom.rs` — a CPU-fed RR
//! session never loads DXC, and this pass must run in all three RR arms
//! (CPU-fed, wavefront-fed, DXR-fed): it records inside `ngxfg_dispatch`,
//! the one site they share. `self_test` (run by `--check`, DLL- and
//! GPU-free) pins the Rust mirrors; the HLSL is their literal twin — change
//! both together (the clouds-wind idiom).

use super::d3d12::{self, Result};
use super::tonemap::compile;
use glam::{Mat4, Vec3A, Vec4};
use windows::core::s;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// Roughness band over which the virtual-image blend fades out: past
/// ROUGH_HI the one-VNDF-sample `spec_hit_t` is too stochastic to steer a
/// warp (and the blurred reflection hides the surface-MV error anyway).
/// Mirrored as HLSL literals below — change all four together.
pub const ROUGH_LO: f32 = 0.25;
pub const ROUGH_HI: f32 = 0.6;

/// The firefly-MV dominance reference (round 3): a pixel whose summed glow
/// luminance S reaches this value takes half the firefly MV, and a bright
/// splat core (lum near the `FF_GLOW_L_MAX` cap, ~446 in luminance) lands at
/// w ≈ 0.999. A night-scene constant — tune once on the 4090 with
/// FR_NGXFG_SHOW=interp if the skirt still strobes (raise = tighter cores,
/// lower = wider capture that starts stealing genuinely-moving background
/// pixels in the halo). Uploaded through the FF cbuffer so this const is the
/// one source (no HLSL literal to co-edit).
pub const FF_MV_L_REF: f32 = 0.25;

// The Rust twins are `clip_depth` / `virtual_prev_px` / `rmv_weight` below —
// change both together.
const HLSL: &str = r#"
Texture2D<float>  zsrc   : register(t0); // linear view-Z (P_DEPTH)
Texture2D<float2> mvsrc  : register(t1); // surface MVs, prev - cur, pixels (P_MVEC)
Texture2D<float>  hitsrc : register(t2); // reflection-ray hit distance (P_SPEC_HIT)
Texture2D<float4> dalb   : register(t3); // diffuse albedo, linear (P_ALBEDO)
Texture2D<float4> salb   : register(t4); // specular albedo / F0, linear (P_SPEC_ALBEDO)
Texture2D<float4> nrough : register(t5); // normal.xyz + roughness.w (P_NORMAL_ROUGH)
Texture2D<float>  ripsrc : register(t6); // water ripple amplitude, 0 elsewhere (P_RIPPLE)
RWTexture2D<float>  zdst  : register(u0); // [0,1] clip depth out
RWTexture2D<float2> mvdst : register(u1); // FG motion vectors out
cbuffer C : register(b0) {
    uint   w; uint h; float A; float B; // clip depth d = A + B/z
    float4 m0; float4 m1; float4 m2; float4 m3; // world -> PREV clip (glam columns)
    float4 org;  // camera origin
    float4 fwd;  // unit forward
    float4 rgt;  // right * tan(fov/2) * aspect (CamBasis pre-scaling)
    float4 upv;  // up * tan(fov/2)
    uint   rmv; float cam_far; float t_cur; float t_prev; // ripple clock (rides the old _pad)
    float  diag; uint ripplemv; float2 _pad2;
}
// Round 3: the CPU-baked firefly splat table (FfTable — layout in lockstep,
// see the size assert beside it). ffc == 0 is the structural off.
cbuffer FF : register(b1) {
    uint   ffc;      // live compacted rows
    float  ff_lref;  // FF_MV_L_REF (Rust is the one source)
    float2 _ffpad;
    float4 ffa[64];  // cur_px.x, cur_px.y, view_z, sigma_px
    float4 ffb[64];  // mv.x, mv.y, lum_center, 0
}
[numthreads(8, 8, 1)]
void cs_guides(uint3 id : SV_DispatchThreadID) {
    if (id.x >= w || id.y >= h) return;
    float z = zsrc[id.xy];
    zdst[id.xy] = saturate(A + B / max(z, 1e-6f));
    float2 mv = mvsrc[id.xy];
    float t_r = hitsrc[id.xy];
    if (rmv != 0 && t_r > 0.0f && z > 0.0f) {
        // Virtual-image reprojection — the Rust twin is virtual_prev_px.
        // Pixel center; the surface MV is anchored at the jittered sample
        // position instead, a sub-pixel mismatch the blend absorbs.
        float2 c = float2(id.xy) + 0.5f;
        float ndx = c.x * (2.0f / w) - 1.0f;
        float ndy = 1.0f - c.y * (2.0f / h);
        float3 du = normalize(fwd.xyz + rgt.xyz * ndx + upv.xyz * ndy);
        float ray_t = z / dot(du, fwd.xyz); // view-Z -> Euclidean ray t
        // A MISSED reflection is the SKY — a virtual image at INFINITY, whose
        // translation parallax is exactly zero (only rotation moves it). The
        // pack cannot say "infinity" though: it clamps a missed reflection to
        // CAM_FAR because that lane's OTHER consumer is the depth delta, which
        // wants far. Feeding that finite distance into the point form below
        // gives the sky real parallax it does not have — at far = 2*diag (~138
        // world units) and a 3841 px render width that is ~28 px of motion
        // error per world unit of camera translation, which warps the sun's
        // specular highlight tens of pixels on every generated frame and snaps
        // it back on the next real one. Measured on the world's DamagedHelmet:
        // strafing is far worse than rotating, which is the signature.
        //
        // So take the analytic LIMIT of the same formula instead of a finite
        // stand-in: as t_r -> inf, V -> org + du*inf, i.e. a pure DIRECTION.
        // Projecting a direction is the point projection with the translation
        // column dropped (a w = 0 homogeneous point), which yields exactly the
        // rotation-only reprojection the sky deserves.
        bool sky_refl = t_r >= cam_far;
        float3 hit_p = org.xyz + du * ray_t;
        // ROUND 4 — the mirror itself MOVES on water.
        //
        // The unfold above is normal-FREE because mirroring the reflected
        // point across the surface plane sends it back down the primary ray;
        // that is only true while the normal HOLDS STILL. Water's ripple
        // tilts the shading normal every rendered frame on the cloud clock,
        // so the reflected skyline slides across the surface with ZERO motion
        // in the geometry — and the MV plane, which only knows camera motion,
        // reports zero. A parked camera over rippling water therefore hands
        // FG two different reflections and no reason to think anything moved,
        // which is the half-cadence strobe. (Same shape as the firefly
        // strobe below, and the opposite of the reflection swim, which was a
        // wrong MV rather than a missing one.)
        //
        // The field is closed-form, so the PREVIOUS frame's normal is
        // computable rather than remembered. Reflect the current content
        // direction back off it and the whole thing collapses to one
        // expression covering both branches:
        //
        //   d = reflect(reflect(du, n_c), n_p)
        //
        // At n_p == n_c this is EXACTLY du (reflect is an involution), so the
        // finite arm reduces to org + du*(ray_t + t_r) and the sky arm to du
        // — today's kernel, bit-for-bit. That identity is the whole safety
        // argument, and the Rust twin gates it.
        float3 d = du;
        float amp = ripsrc[id.xy];
        if (ripplemv != 0 && amp > 0.0f) {
            float3 n_c = normalize(nrough[id.xy].xyz);
            // First-order previous normal: the ripple perturbs by SUBTRACTING
            // the in-plane height gradient, so stepping the gradient back in
            // time steps the normal back with it. Exact at dt = 0; the error
            // is second order in amp*|g| (<= ~0.8 deg on water's ~14 deg
            // tilt), well under the 1-3 deg/frame being corrected.
            float2 gd = amp * (ripple_grad(hit_p, t_prev, diag)
                             - ripple_grad(hit_p, t_cur, diag));
            float3 g3 = float3(gd.x, 0.0f, gd.y);
            float3 n_p = n_c - (g3 - n_c * dot(g3, n_c));
            float nl = dot(n_p, n_p);
            // A degenerate or horizon-crossing reconstruction falls back to
            // n_c, i.e. to the exact pre-round-4 unfold. Coarser, never
            // wrong-signed — the humble arm.
            if (nl > 1e-12f) {
                n_p *= rsqrt(nl);
                if (dot(n_p, n_c) > 0.0f) d = reflect(reflect(du, n_c), n_p);
            }
        }
        float3 V = hit_p + d * t_r; // planar-unfold virtual point
        float4 pc = sky_refl ? (m0 * d.x + m1 * d.y + m2 * d.z)
                             : (m0 * V.x + m1 * V.y + m2 * V.z + m3);
        if (pc.w > 1e-6f) {
            float2 pndc = pc.xy / pc.w;
            float2 ppx = float2((pndc.x + 1.0f) * 0.5f * w,
                                (1.0f - pndc.y) * 0.5f * h);
            float2 mv_v = ppx - c; // current -> previous, pixels (the plane's convention)
            const float3 LUM = float3(0.2126f, 0.7152f, 0.0722f);
            float ld = dot(dalb[id.xy].rgb, LUM);
            float ls = dot(salb[id.xy].rgb, LUM);
            float rough = nrough[id.xy].w;
            // ROUND 4b — Fresnel, on ripple pixels only.
            //
            // ls/(ld+ls) is an F0-luminance PROXY for "how much of this pixel
            // is the mirror image", and on water it is wrong in both
            // directions at once: F0 is a flat 0.04 while the actual
            // reflectance runs from ~2% face-on to ~100% at grazing. Water
            // also keeps its dark unlifted Kd (the class exists to avoid the
            // neutral wash), so the proxy lands at a middling 0.15-0.45
            // EVERYWHERE — which would apply a fraction of the correct MV at
            // exactly the grazing angles where the sliding skyline lives, and
            // halve the strobe instead of killing it.
            //
            // Scoped inside amp > 0: every non-water pixel keeps the proxy
            // and stays bit-identical to rounds 2/3.
            if (amp > 0.0f) {
                float cos_i = saturate(abs(dot(du, normalize(nrough[id.xy].xyz))));
                float f = 1.0f - cos_i;
                float f5 = f * f * f * f * f;
                ls *= (0.04f + 0.96f * f5) / 0.04f; // Schlick / F0 -> reflectance ratio
            }
            float wgt = saturate(ls / (ld + ls + 1e-4f))
                      * (1.0f - smoothstep(0.25f, 0.6f, rough)); // ROUGH_LO..ROUGH_HI
            mv = lerp(mv, mv_v, wgt);
        }
    }
    // Round 3 (firefly glow MVs): a glow-dominated pixel's CONTENT is the
    // firefly, not the surface behind it — the Rust twin is ff_mv_blend,
    // change both together. Runs AFTER the virtual-reflection blend: the
    // glow overlays surface and sky alike, and the blend above is exactly
    // the confident wrong answer at a bright highlight on smooth metal.
    if (ffc != 0) {
        float S = 0.0f;
        float2 mv_ff = float2(0.0f, 0.0f);
        float2 cpx = float2(id.xy) + 0.5f;
        for (uint i = 0; i < ffc; i++) {
            float4 fa = ffa[i];
            if (fa.z >= z) continue; // occluded by the primary hit (sky z = far passes)
            float2 dpx = cpx - fa.xy;
            float ax = dot(dpx, dpx) / (2.0f * fa.w * fa.w);
            if (ax > 9.22f) continue; // reject BEFORE exp (the fireflies +34 ms lesson)
            // The 1e-4 skirt makes the reject EXACT (9.22 > -ln(1e-4)), so
            // the weight is continuous at the cut instead of stepping.
            float g = exp(-ax) - 1e-4f;
            if (g <= 0.0f) continue;
            float wi = ffb[i].z * g;
            S += wi;
            mv_ff += wi * ffb[i].xy;
        }
        if (S > 0.0f) {
            mv = lerp(mv, mv_ff / S, S / (S + ff_lref));
        }
    }
    mvdst[id.xy] = mv;
}
"#;

/// The CPU mirror of `cs_guides`' depth half — the value
/// `perspective_lh(_, _, near, far)` produces as `clip.z / clip.w` at view
/// depth `view_z` (see `self_test` for the matrix-consistency pin).
pub fn clip_depth(view_z: f32, near: f32, far: f32) -> f32 {
    let a = far / (far - near);
    let b = -near * far / (far - near);
    (a + b / view_z.max(1e-6)).clamp(0.0, 1.0)
}

/// The CPU mirror of `cs_guides`' virtual-image reprojection: the PREVIOUS
/// frame's pixel position of the virtual point behind pixel center (cx, cy),
/// or None when it lands behind the previous image plane. `right_s`/`up_s`
/// carry the CamBasis pre-scaling (tan(fov/2)·aspect / tan(fov/2));
/// `m` = world → previous clip (glam column-vector convention).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn virtual_prev_px(
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    view_z: f32,
    t_r: f32,
    cam_far: f32,
    origin: Vec3A,
    fwd: Vec3A,
    right_s: Vec3A,
    up_s: Vec3A,
    m: &Mat4,
) -> Option<(f32, f32)> {
    let ndx = cx * (2.0 / w) - 1.0;
    let ndy = 1.0 - cy * (2.0 / h);
    let du = (fwd + right_s * ndx + up_s * ndy).normalize();
    let ray_t = view_z / du.dot(fwd);
    let v = origin + du * (ray_t + t_r);
    // t_r >= cam_far IS the pack's "reflection missed" encoding: the sky is at
    // infinity, so project the DIRECTION (w = 0 — the translation column drops
    // out) for rotation-only parallax. See the cs_guides twin for why a finite
    // stand-in warps the sun's highlight.
    let pc = if t_r >= cam_far {
        *m * Vec4::new(du.x, du.y, du.z, 0.0)
    } else {
        *m * Vec4::new(v.x, v.y, v.z, 1.0)
    };
    if pc.w <= 1e-6 {
        return None;
    }
    let (px, py) = (pc.x / pc.w, pc.y / pc.w);
    Some(((px + 1.0) * 0.5 * w, (1.0 - py) * 0.5 * h))
}

/// Round 4: the previous frame's ripple-perturbed shading normal, first
/// order — the CPU mirror of the `n_p` block in `cs_guides`.
///
/// `ripple_normal` tilts by SUBTRACTING the in-plane height gradient, so
/// stepping the gradient back in time steps the normal back with it. Exact at
/// `t_prev == t_cur` (returns `n_c` bitwise, which is what makes the whole
/// round-4 path collapse to round 2/3); second-order in `amp·|g|` otherwise.
/// A degenerate or horizon-crossing result falls back to `n_c` — coarser,
/// never wrong-signed.
pub fn ripple_prev_normal(
    n_c: Vec3A,
    hit_p: Vec3A,
    t_cur: f32,
    t_prev: f32,
    amp: f32,
    diag: f32,
) -> Vec3A {
    if amp <= 0.0 {
        return n_c;
    }
    let gd = (crate::shade::ripple_grad(hit_p, t_prev, diag)
        - crate::shade::ripple_grad(hit_p, t_cur, diag))
        * amp;
    let g3 = Vec3A::new(gd.x, 0.0, gd.y);
    let n_p = n_c - (g3 - n_c * g3.dot(n_c));
    let nl = n_p.length_squared();
    if nl <= 1e-12 {
        return n_c;
    }
    let n_p = n_p / nl.sqrt();
    if n_p.dot(n_c) > 0.0 { n_p } else { n_c }
}

/// Round 4: `virtual_prev_px` for a mirror that MOVED between frames.
///
/// The still-mirror form is normal-free because reflecting a point across the
/// surface plane and back sends it down the primary ray; once the normal
/// moves that no longer holds. Reflecting the current content direction off
/// the PREVIOUS normal generalizes both branches into one expression, and
/// `reflect` being an involution makes `n_p == n_c` reduce to `du` exactly —
/// so at `amp == 0` or `t_prev == t_cur` this IS `virtual_prev_px`, which the
/// gates assert bitwise rather than approximately.
#[allow(clippy::too_many_arguments)]
pub fn ripple_virtual_prev_px(
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    view_z: f32,
    t_r: f32,
    cam_far: f32,
    origin: Vec3A,
    fwd: Vec3A,
    right_s: Vec3A,
    up_s: Vec3A,
    m: &Mat4,
    n_c: Vec3A,
    t_cur: f32,
    t_prev: f32,
    amp: f32,
    diag: f32,
) -> Option<(f32, f32)> {
    // The off arm CALLS the still-mirror twin, so "bit-identical when the
    // feature is off" is true by construction and not by re-derivation.
    if amp <= 0.0 {
        return virtual_prev_px(
            cx, cy, w, h, view_z, t_r, cam_far, origin, fwd, right_s, up_s, m,
        );
    }
    let ndx = cx * (2.0 / w) - 1.0;
    let ndy = 1.0 - cy * (2.0 / h);
    let du = (fwd + right_s * ndx + up_s * ndy).normalize();
    let ray_t = view_z / du.dot(fwd);
    let hit_p = origin + du * ray_t;
    let n_p = ripple_prev_normal(n_c, hit_p, t_cur, t_prev, amp, diag);
    let refl = |i: Vec3A, n: Vec3A| i - n * (2.0 * i.dot(n));
    let d = refl(refl(du, n_c), n_p);
    let v = hit_p + d * t_r;
    let pc = if t_r >= cam_far {
        *m * Vec4::new(d.x, d.y, d.z, 0.0)
    } else {
        *m * Vec4::new(v.x, v.y, v.z, 1.0)
    };
    if pc.w <= 1e-6 {
        return None;
    }
    let (px, py) = (pc.x / pc.w, pc.y / pc.w);
    Some(((px + 1.0) * 0.5 * w, (1.0 - py) * 0.5 * h))
}

/// Round 4b: the Fresnel scale the kernel applies to the specular luminance
/// on ripple pixels only. F0 = 0.04 is a flat proxy for a reflectance that
/// actually runs ~2% face-on to ~100% at grazing, and water's dark unlifted
/// albedo puts the raw proxy at a middling value everywhere — which would
/// apply a fraction of the correct MV exactly where the sliding skyline is.
pub fn fresnel_spec_scale(cos_i: f32) -> f32 {
    let f = (1.0 - cos_i.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let f5 = f * f * f * f * f;
    (0.04 + 0.96 * f5) / 0.04
}

/// The CPU mirror of `cs_guides`' blend weight: how much of the pixel's look
/// is the reflection — specular vs diffuse albedo luminance, damped over
/// [ROUGH_LO, ROUGH_HI].
pub fn rmv_weight(lum_diff: f32, lum_spec: f32, rough: f32) -> f32 {
    let t = ((rough - ROUGH_LO) / (ROUGH_HI - ROUGH_LO)).clamp(0.0, 1.0);
    let smooth = 1.0 - t * t * (3.0 - 2.0 * t);
    (lum_spec / (lum_diff + lum_spec + 1e-4)).clamp(0.0, 1.0) * smooth
}

/// Round 3: the CPU-baked firefly splat table — the `FF` cbuffer's exact
/// byte layout (16-byte header + two float4 rows per `MAX_FIREFLIES` slot;
/// the HLSL declares `ffa[64]`/`ffb[64]`, in lockstep with the array bound
/// here). Rows are COMPACTED at bake: `ffc` counts live rows only, and 0 is
/// the structural off every day / `--no-fireflies` / lever-off frame takes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfTable {
    pub ffc: u32,
    /// `FF_MV_L_REF`, uploaded so the Rust const is the one source.
    pub lref: f32,
    pub _pad: [f32; 2],
    /// cur_px.x, cur_px.y, view_z, sigma_px.
    pub a: [[f32; 4]; crate::fireflies::MAX_FIREFLIES],
    /// mv.x, mv.y (prev − cur, pixels — the plane's convention), center
    /// luminance, 0.
    pub b: [[f32; 4]; crate::fireflies::MAX_FIREFLIES],
}
const _: () = assert!(std::mem::size_of::<FfTable>() == 16 + 2 * 64 * 16);
/// Per-slot stride in the upload ring: the table size rounded up to the
/// 256-byte CBV placement alignment.
pub const FF_CB_STRIDE: usize = 2304;
const _: () =
    assert!(FF_CB_STRIDE >= std::mem::size_of::<FfTable>() && FF_CB_STRIDE % 256 == 0);

impl FfTable {
    pub fn empty() -> FfTable {
        FfTable {
            ffc: 0,
            lref: FF_MV_L_REF,
            _pad: [0.0; 2],
            a: [[0.0; 4]; crate::fireflies::MAX_FIREFLIES],
            b: [[0.0; 4]; crate::fireflies::MAX_FIREFLIES],
        }
    }
}

/// World point → pixel through a world→clip matrix, the kernel's own NDC →
/// pixel mapping (and `virtual_prev_px`'s — one convention, three users).
/// None = at/behind the camera plane; the caller DROPS the row so the
/// surface MV survives rather than inventing one.
fn px_of(m: &Mat4, p: Vec3A, w: f32, h: f32) -> Option<(f32, f32)> {
    let pc = *m * Vec4::new(p.x, p.y, p.z, 1.0);
    if pc.w <= 1e-6 {
        return None;
    }
    Some(((pc.x / pc.w + 1.0) * 0.5 * w, (1.0 - pc.y / pc.w) * 0.5 * h))
}

/// Bake the frame's firefly splat rows in SCREEN space — everything the
/// kernel needs per firefly is a per-frame constant, so the per-pixel loop
/// degenerates to a 2-D Gaussian test and no third copy of `ff_glow` exists
/// (CPU `fireflies::glow` + `trace_common.hlsli::ff_glow` stay the only
/// twins; the σ/lum parity gate in `self_test` ties this bake to them).
/// `world_to_clip` = current frame, `world_to_prev_clip` = the same `m` the
/// GuideParams carry. Pure — zero rng, zero statics.
#[allow(clippy::too_many_arguments)]
pub fn ff_guide_rows(
    cur: &crate::fireflies::Fireflies,
    prev: &crate::fireflies::Fireflies,
    w: f32,
    h: f32,
    fov_y: f32,
    org: Vec3A,
    fwd: Vec3A,
    world_to_clip: &Mat4,
    world_to_prev_clip: &Mat4,
) -> FfTable {
    use crate::fireflies::{FF_COLOR, FF_E_K, FF_GLOW_L_MAX, FF_SRC_K};
    let mut t = FfTable::empty();
    if cur.count == 0 {
        return t;
    }
    // Prev poses pair by INDEX (bake writes row i from pose(i) — the roster
    // is fixed). A count mismatch (first frame after arming, a live count
    // edit) reprojects the CURRENT pose instead: the camera-motion-only MV
    // of a momentarily stationary firefly — coarser, never wrong-signed.
    let prev = if prev.count == cur.count { prev } else { cur };
    // camera.rs::pixel_cone — the angular pitch of one pixel, the unit the
    // glow's σ converts through.
    let pixel_cone = 2.0 * (fov_y * 0.5).tan() / h;
    let e = FF_E_K * cur.scale * cur.scale;
    let src_r = FF_SRC_K * cur.scale;
    let lum709 = 0.2126 * FF_COLOR.x + 0.7152 * FF_COLOR.y + 0.0722 * FF_COLOR.z;
    let mut n = 0usize;
    for i in 0..cur.count as usize {
        let wgt = cur.pos[i][3];
        if wgt <= 0.0 {
            continue;
        }
        let p = Vec3A::from_slice(&cur.pos[i]);
        let z = (p - org).dot(fwd);
        if z <= 0.0 {
            continue;
        }
        let Some((cx, cy)) = px_of(world_to_clip, p, w, h) else { continue };
        let Some((px, py)) =
            px_of(world_to_prev_clip, Vec3A::from_slice(&prev.pos[i]), w, h)
        else {
            continue;
        };
        // ff_glow's exact σ/radiance, converted to pixels/luminance: σ_ang =
        // max(pixel_cone/4, src_r/s) — callers hand glow half_angle =
        // pixel_cone/2 and it halves again — and the FF_GLOW_L_MAX cap.
        let s = (p - org).length();
        let sigma_ang = (pixel_cone * 0.25).max(src_r / s);
        let lum = lum709
            * (e * wgt / (s * s * std::f32::consts::TAU * sigma_ang * sigma_ang))
                .min(FF_GLOW_L_MAX);
        if lum <= 1e-3 * FF_MV_L_REF {
            continue;
        }
        t.a[n] = [cx, cy, z, sigma_ang / pixel_cone];
        t.b[n] = [px - cx, py - cy, lum, 0.0];
        n += 1;
    }
    t.ffc = n as u32;
    t
}

/// The CPU mirror of `cs_guides`' round-3 loop — change both together. An
/// `ffc == 0` table (and an all-rejected pixel) returns `mv` BIT-identically,
/// which is what the off-arm gate pins.
pub fn ff_mv_blend(mv: (f32, f32), cx: f32, cy: f32, z_pixel: f32, t: &FfTable) -> (f32, f32) {
    if t.ffc == 0 {
        return mv;
    }
    let (mut s, mut ax, mut ay) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..t.ffc as usize {
        let fa = t.a[i];
        if fa[2] >= z_pixel {
            continue;
        }
        let (dx, dy) = (cx - fa[0], cy - fa[1]);
        let a = (dx * dx + dy * dy) / (2.0 * fa[3] * fa[3]);
        if a > 9.22 {
            continue;
        }
        let g = (-a).exp() - 1e-4;
        if g <= 0.0 {
            continue;
        }
        let wi = t.b[i][2] * g;
        s += wi;
        ax += wi * t.b[i][0];
        ay += wi * t.b[i][1];
    }
    if s <= 0.0 {
        return mv;
    }
    let wff = s / (s + t.lref);
    (mv.0 + (ax / s - mv.0) * wff, mv.1 + (ay / s - mv.1) * wff)
}

/// Root constants for `cs_guides` — field order IS the cbuffer layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuideParams {
    pub w: u32,
    pub h: u32,
    pub a: f32,
    pub b: f32,
    /// world → PREVIOUS clip, glam columns.
    pub m: [[f32; 4]; 4],
    pub org: [f32; 4],
    pub fwd: [f32; 4],
    pub rgt: [f32; 4],
    pub upv: [f32; 4],
    pub rmv: u32,
    /// The far plane the pack clamps a MISSED reflection to (`spec_hit_t`'s
    /// "reflected sky" value). The kernel needs it to tell a genuine hit at
    /// distance from a miss, because one lane carries both — see the sky-miss
    /// branch in `cs_guides`. Rides the old `_pad`, so the layout is unmoved.
    pub cam_far: f32,
    /// The ripple clock, current and previous EVALUATED frame (round 4).
    /// Equal values are the structural off state: the gradient delta is then
    /// exactly zero, the reconstructed previous normal is `n_c`, and the
    /// kernel reduces to the round-2/3 unfold bit-for-bit. That is also the
    /// first-frame default — `t_prev` is never 0.0 on a frame with no
    /// history, because a bogus multi-second delta is precisely the confident
    /// wrong MV this pass exists to avoid. Rides the old `_pad`.
    pub t_cur: f32,
    pub t_prev: f32,
    /// `Scene::diag` — every ripple length is scale-relative.
    pub diag: f32,
    /// `FR_NGXFG_RIPPLEMV=off` clears it (0 = the pre-round-4 kernel stream).
    pub ripplemv: u32,
    pub _pad2: [f32; 2],
}
const PARAM_DWORDS: u32 = (std::mem::size_of::<GuideParams>() / 4) as u32;
const _: () = assert!(std::mem::size_of::<GuideParams>() == 44 * 4);

/// Kernel-order indices into the source-plane array `ensure` takes — must
/// match `rr::RrResources::guide_inputs` (which returns them in this order).
pub const SRC_PLANES: usize = 7;
const SRC_FORMATS: [DXGI_FORMAT; SRC_PLANES] = [
    DXGI_FORMAT_R32_FLOAT,          // linear view-Z
    DXGI_FORMAT_R16G16_FLOAT,       // surface MVs
    DXGI_FORMAT_R16_FLOAT,          // spec hit distance
    DXGI_FORMAT_R8G8B8A8_UNORM,     // diffuse albedo
    DXGI_FORMAT_R8G8B8A8_UNORM,     // specular albedo
    DXGI_FORMAT_R16G16B16A16_FLOAT, // normal + roughness
    DXGI_FORMAT_R16_FLOAT,          // ripple amplitude (round 4)
];

/// One dispatch: read six RR planes, write the [0,1] clip-depth plane and
/// the reflection-blended MV plane the NGX evaluate consumes. Planes +
/// descriptors are (re)built by `ensure` at feature-creation time (first
/// dispatch / after a resize), so no descriptor is ever rewritten under an
/// in-flight frame.
pub struct GuidePass {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// [0..6] = SRVs of the RR planes (kernel order), [6] = clip UAV,
    /// [7] = mv UAV.
    heap: ID3D12DescriptorHeap,
    inc: u32,
    /// Render-res R32F / RG16F; rest UNORDERED_ACCESS between frames
    /// (`record` hands them to the evaluate as NON_PIXEL_SHADER_RESOURCE,
    /// `restore` puts them back).
    clip: Option<ID3D12Resource>,
    mv: Option<ID3D12Resource>,
    w: u32,
    h: u32,
    /// Round 3's firefly table ring: `FRAMES_IN_FLIGHT × FF_CB_STRIDE` on an
    /// upload heap, persistently mapped, bound as root CBV b1. Res-
    /// independent, so `ensure` never touches it. The frame slot's fence was
    /// waited in `begin_frame`, so writing slot k's region never races an
    /// in-flight read.
    ff_cb: ID3D12Resource,
    ff_ptr: *mut u8,
}

impl GuidePass {
    pub fn new(device: &ID3D12Device) -> Result<GuidePass> {
        // Root signature: [0] SRV table (t0..t5), [1] UAV table (u0..u1),
        // [2] the GuideParams root constants (b0). The bloom shape, minus
        // its sampler.
        let srv_range = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: SRC_PLANES as u32,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        }];
        let uav_range = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: 2,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        }];
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: srv_range.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: uav_range.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: PARAM_DWORDS,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            // [3] the firefly splat table (round 3) — a root CBV: 2 KB of
            // rows is far past the root-constant budget, and a root
            // descriptor needs only a GPU VA (no heap change).
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 1, RegisterSpace: 0 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            ..Default::default()
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut errb: Option<ID3DBlob> = None;
        unsafe {
            D3D12SerializeRootSignature(
                &rs_desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                Some(&mut errb),
            )
        }
        .map_err(|e| format!("ngxfg-guides: D3D12SerializeRootSignature: {e}"))?;
        let blob = blob.unwrap();
        let root_sig: ID3D12RootSignature = unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    blob.GetBufferPointer() as *const u8,
                    blob.GetBufferSize(),
                ),
            )
        }
        .map_err(|e| format!("ngxfg-guides: CreateRootSignature: {e}"))?;

        // ripple.hlsli AHEAD of the kernel: the ONE GPU copy of the field,
        // shared with the dxc shading units (see its header). It is written
        // cs_5_0-clean and self-defines TAU, so it pastes standalone here —
        // there is no trace_common.hlsli prelude on this pass.
        let src = format!("{}\n{}", super::trace::RIPPLE_HLSLI, HLSL);
        let cs = compile(&src, s!("cs_guides"), s!("cs_5_0"), "ngxfg-guides cs_guides")?;
        let pso: ID3D12PipelineState = unsafe {
            device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root_sig),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: cs.GetBufferPointer(),
                    BytecodeLength: cs.GetBufferSize(),
                },
                ..Default::default()
            })
        }
        .map_err(|e| format!("ngxfg-guides: CreateComputePipelineState: {e}"))?;

        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: SRC_PLANES as u32 + 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("ngxfg-guides: CreateDescriptorHeap: {e}"))?;
        let inc = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };

        let ff_cb = {
            let mut res: Option<ID3D12Resource> = None;
            unsafe {
                device.CreateCommittedResource(
                    &d3d12::upload_heap(),
                    D3D12_HEAP_FLAG_NONE,
                    &d3d12::buffer_desc((FF_CB_STRIDE * d3d12::FRAMES_IN_FLIGHT) as u64),
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut res,
                )
            }
            .map_err(|e| format!("ngxfg-guides: firefly CB ring: {e}"))?;
            res.unwrap()
        };
        let mut ff_ptr = std::ptr::null_mut();
        unsafe { ff_cb.Map(0, None, Some(&mut ff_ptr)) }
            .map_err(|e| format!("ngxfg-guides: firefly CB map: {e}"))?;

        Ok(GuidePass {
            root_sig,
            pso,
            heap,
            inc,
            clip: None,
            mv: None,
            w: 0,
            h: 0,
            ff_cb,
            ff_ptr: ff_ptr as *mut u8,
        })
    }

    /// (Re)build the output planes at the feature's render res and rewrite
    /// ALL descriptors — the SRVs must be rewritten even at unchanged dims,
    /// because a resize rebuilt `RrResources` and `srcs` are new resources.
    /// Only called at NGX-feature creation (first dispatch ever, or the
    /// first after a resize's `wait_idle`), so no in-flight frame references
    /// these descriptors.
    pub fn ensure(
        &mut self,
        device: &ID3D12Device,
        w: u32,
        h: u32,
        srcs: [&ID3D12Resource; SRC_PLANES],
    ) -> Result<()> {
        if self.clip.is_none() || self.w != w || self.h != h {
            self.clip = Some(d3d12::committed_tex(
                device,
                w,
                h,
                DXGI_FORMAT_R32_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?);
            // RG16F typed UAV STORE is an optional D3D12 cap (the mandatory
            // set is RGBA32/RGBA16/RGBA8/R32/R16/R8 — two-channel formats
            // need CheckFormatSupport, which trace::wire_feed's format gate
            // performs for the feed kernels). Deliberately unchecked here:
            // this pass exists only in raw-NGX DLSS-G sessions, i.e. NVIDIA
            // with the NDA SDK, where support is universal. Port the pass to
            // a cross-vendor path and it needs the wire_feed gate.
            self.mv = Some(d3d12::committed_tex(
                device,
                w,
                h,
                DXGI_FORMAT_R16G16_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?);
            self.w = w;
            self.h = h;
        }
        let base = unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() };
        let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + (i * self.inc) as usize,
        };
        for (i, (src, fmt)) in srcs.iter().zip(SRC_FORMATS).enumerate() {
            unsafe {
                device.CreateShaderResourceView(
                    *src,
                    Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: fmt,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                        },
                    }),
                    at(i as u32),
                );
            }
        }
        let uav = |res: &ID3D12Resource, fmt, slot: u32| unsafe {
            device.CreateUnorderedAccessView(
                res,
                None,
                Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: fmt,
                    ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                    ..Default::default()
                }),
                at(slot),
            );
        };
        uav(self.clip.as_ref().unwrap(), DXGI_FORMAT_R32_FLOAT, SRC_PLANES as u32);
        uav(self.mv.as_ref().unwrap(), DXGI_FORMAT_R16G16_FLOAT, SRC_PLANES as u32 + 1);
        Ok(())
    }

    /// Record the conversion. The source planes already rest
    /// NON_PIXEL_SHADER_RESOURCE (the state a compute SRV read wants); both
    /// output planes go UAV → NON_PIXEL_SHADER_RESOURCE for the evaluate
    /// that follows. Pair with `restore` after the evaluate. `slot` is the
    /// frame's in-flight slot (its ring region is fence-free by
    /// `begin_frame`); `ff` is the frame's baked splat table — pass
    /// `FfTable::empty()` for the structural off.
    pub fn record(
        &self,
        list: &ID3D12GraphicsCommandList,
        p: &GuideParams,
        slot: usize,
        ff: &FfTable,
    ) {
        let gpu0 = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        let uavs = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu0.ptr + (SRC_PLANES as u32 * self.inc) as u64,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                ff as *const FfTable as *const u8,
                self.ff_ptr.add(slot * FF_CB_STRIDE),
                std::mem::size_of::<FfTable>(),
            );
            list.SetComputeRootSignature(&self.root_sig);
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetPipelineState(&self.pso);
            list.SetComputeRootDescriptorTable(0, gpu0);
            list.SetComputeRootDescriptorTable(1, uavs);
            list.SetComputeRoot32BitConstants(2, PARAM_DWORDS, p as *const _ as *const _, 0);
            list.SetComputeRootConstantBufferView(
                3,
                self.ff_cb.GetGPUVirtualAddress() + (slot * FF_CB_STRIDE) as u64,
            );
            list.Dispatch(self.w.div_ceil(8), self.h.div_ceil(8), 1);
            list.ResourceBarrier(&[
                d3d12::transition(
                    self.clip.as_ref().unwrap(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
                d3d12::transition(
                    self.mv.as_ref().unwrap(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
            ]);
        }
    }

    /// Back to the UAV rest state after the evaluate consumed the planes.
    pub fn restore(&self, list: &ID3D12GraphicsCommandList) {
        unsafe {
            list.ResourceBarrier(&[
                d3d12::transition(
                    self.clip.as_ref().unwrap(),
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
                d3d12::transition(
                    self.mv.as_ref().unwrap(),
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
            ]);
        }
    }

    pub fn clip(&self) -> &ID3D12Resource {
        self.clip.as_ref().expect("ensure before clip")
    }
    pub fn mv(&self) -> &ID3D12Resource {
        self.mv.as_ref().expect("ensure before mv")
    }
}

/// `TONE_HLSL`'s two kernels: compress the color handed to NGX, then expand
/// its output back. Kept beside the guide kernel because it shares that
/// pass's fxc `cs_5_0` toolchain and its dispatch site.
const TONE_HLSL: &str = r#"
Texture2D<float4>   src : register(t0);
RWTexture2D<float4> dst : register(u0);
cbuffer C : register(b0) { uint tw; uint th; uint mode; float k; };

// mode 1 = linear scale (exact round trip, isolates absolute MAGNITUDE),
// 3 = log2(1+v)*k (compresses ratios like Reinhard but stays WELL
// CONDITIONED — its inverse's relative error is constant instead of
// exploding at the ceiling), anything else = Reinhard v/(1+v).
float3 tone_fwd(float3 v) {
    if (mode == 1u) return v * k;
    if (mode == 3u) return log2(1.0f + v) * k;
    return v / (1.0f + v);
}
float3 tone_inv(float3 v) {
    if (mode == 1u) return v / max(k, 1e-8f);
    if (mode == 3u) return exp2(v / max(k, 1e-8f)) - 1.0f;
    // Reinhard's inverse blows up as v -> 1; clamp so a saturated texel comes
    // back large-but-finite instead of inf (which would poison the tonemap).
    float3 s = min(v, 0.9995f);
    return s / max(1.0f - s, 1e-6f);
}

[numthreads(8, 8, 1)]
void cs_tone_compress(uint3 id : SV_DispatchThreadID) {
    if (id.x >= tw || id.y >= th) return;
    float4 c = src[id.xy];
    dst[id.xy] = float4(tone_fwd(max(c.rgb, 0.0f)), c.a);
}

// In place on the NGX output: one texel per thread, so the read-write is
// safe without a copy.
[numthreads(8, 8, 1)]
void cs_tone_expand(uint3 id : SV_DispatchThreadID) {
    if (id.x >= tw || id.y >= th) return;
    float4 c = dst[id.xy];
    dst[id.xy] = float4(tone_inv(max(c.rgb, 0.0f)), c.a);
}
"#;

/// The linear-scale mode's factor. 810 (the sun disc) -> ~3.2, so a squared
/// difference lands ~10 instead of ~6.6e5.
pub const TONE_SCALE_K: f32 = 1.0 / 256.0;

/// The log mode's factor: `log2(1 + 810)/16` ~ 0.60, so the whole scene lands
/// in roughly [0, 0.7]. Chosen for CONDITIONING, not range — the inverse's
/// derivative is `ln2 * exp2(c/k) / k`, i.e. proportional to the VALUE, so
/// one f16 quantum costs a constant ~0.5% relative error everywhere instead
/// of Reinhard's blow-up near the ceiling (which is what banded the outer
/// bloom edge).
pub const TONE_LOG_K: f32 = 1.0 / 16.0;

/// `FR_NGXFG_TONEMAP` — the input curve for raw-NGX frame generation.
/// **DEFAULT `reinhard`** (shipped 2026-07-31, promoted from the diagnostic it
/// started as): every `--fg` session compresses the color handed to NGX and
/// expands its output in place. `off` is the kill arm — raw linear
/// `rr.output`, the sun-disc strobe returns on demand — and `scale`/`log`
/// remain the diagnostic arms whose measurements are recorded below.
///
/// WHY IT EXISTS. Frame generation warps the sun disc's neighborhood into
/// smooth "heat haze" on GENERATED frames only, at a parked camera, in every
/// render arm. Diagnosed by elimination (2026-07-31): the camera block
/// (including a full `FR_NGXFG_CAM=identity` substitution), matrix majority,
/// jitter, resolution, HDR declaration, depth convention, MV convention,
/// magnitude (the far dimmer MOON does it too), per-frame variance
/// (`--spp 16`), clouds, bloom and DRS were each ruled out by MEASUREMENT —
/// and the ffx FI interpolator is CLEAN on identical content, which localizes
/// it to what we hand NGX. (That cross-interpolator A/B is the same move that
/// cracked the reflection-swim bug; see `jitter_mode`.)
///
/// The one uncontrolled difference that survives: ffx FI interpolates the
/// POST-tonemap backbuffer (display-range), while raw NGX is handed
/// `rr.output` — scene-referred LINEAR radiance, where the sun disc is ~810
/// against a scene of order 1. `colorBuffersHDR` tells NGX the buffer is HDR;
/// it does not tell it the values are unbounded.
///
/// SHAPE. Compress `rr.output` into a scratch, hand THAT to NGX, then expand
/// its output IN PLACE. Presentation is therefore untouched — the tonemap
/// reads `n.out` and sees linear values exactly as it does with the pass off —
/// so the lever is an A/B of ONE variable with no second code path downstream,
/// and the REAL half of every present pair is bit-identical in every mode
/// (compress only READS `rr.output`; expand only writes `n.out`).
///
/// MEASURED (2026-07-31, 4090, parked and rotating): `scale` converges at a
/// parked camera but DOUBLE-GHOSTS the disc under rotation; `reinhard` is
/// correct in both. So the defect is the unbounded RATIOS, not the absolute
/// magnitude — reducing magnitude while preserving every ratio does not give
/// NGX's flow what it needs. That is the ffx-vs-NGX difference, confirmed
/// from the other side, and it says the permanent fix is to hand FG a
/// display-range image (as standard DLSS-G integrations do) rather than
/// scene-referred linear.
///
/// `log` MEASURED WORSE than `reinhard` (ghosting + banding), and the reason
/// is MIDTONE PLACEMENT, not precision: at `TONE_LOG_K` a scene value of 1
/// lands at 0.0625 where Reinhard puts it at 0.5, so ordinary content is
/// crushed against black and the flow estimator has almost no signal. Log
/// cannot retune out of this — placing 1 at 0.5 needs k = 0.5, which sends
/// 810 back above 1 and unbounds it again. So the answer to this probe's
/// second question is: **NGX wants a display-CURVE-shaped input, not merely a
/// bounded one** — and Reinhard working is not luck, `v/(1+v)` IS a classic
/// tonemap operator. The permanent fix should hand FG a genuinely tonemapped
/// image (the standard DLSS-G integration), not any cheap compression.
///
/// A SECOND, DISTINCT ARTIFACT survives every mode and is NOT ours: straight
/// piecewise-linear banding in the smooth aureole, ONLY under camera motion,
/// settling when the camera stops. That rules out this probe's round-trip
/// quantization, which is a function of the VALUE and would be identical
/// parked (and which would band along iso-radiance contours — curved, not
/// straight). It is NGX's own optical flow hitting the aperture problem: flow
/// is unconstrained in a textureless gradient, so per-block estimates drift
/// independently and the block boundaries become visible steps. Parked, the
/// frames are near-identical, flow converges to zero, and it vanishes.
/// UNTESTED: whether ffx FI bands in the same region (if it does, the smooth
/// HDR gradient is simply hard for any interpolator and this is not worth
/// chasing).
///
/// THREE MODES, testing three different sub-hypotheses:
/// - `scale`: a pure linear scale by `TONE_SCALE_K`. Preserves every contrast
///   RATIO and moves only absolute magnitude, so it isolates numeric overflow
///   inside NGX — a squared difference of ~810 is 6.6e5, past f16's 65504
///   ceiling. Round trip exact up to f16 rounding.
/// - `reinhard`: `v/(1+v)`, compressing the ratios themselves — the arm the
///   ffx comparison points at, the one that WORKS, and THE SHIPPING DEFAULT.
///   KNOWN-ACCEPT it carries: its inverse is ill-conditioned near the ceiling
///   (`1/(1-v)^2` reaches ~1e6, so one f16 quantum of the stored value expands
///   to ~490 radiance), which can band the outer bloom edge on GENERATED
///   frames only. The tracked follow-on that would delete it — hand FG a
///   genuinely display-referred image and present its output through a
///   matching curve, i.e. no inverse pass at all — is blocked on
///   `fullscreen_to_backbuffer` not being parameterized on a render target and
///   on the bloom pyramid running on linear input; reinhard-by-default is the
///   measured-correct answer until then.
/// - `log`: `log2(1+v)*TONE_LOG_K`, same ratio compression with CONSTANT
///   relative error, so it should match `reinhard` without the banding. It is
///   also the experiment that distinguishes "NGX needs a specific tonemap
///   curve" from "NGX needs merely bounded input" — if a curve this unlike
///   Reinhard works equally well, any monotone compression suffices and the
///   permanent fix does not have to route the real tonemap through FG.
pub struct TonePass {
    root_sig: ID3D12RootSignature,
    pso_compress: ID3D12PipelineState,
    pso_expand: ID3D12PipelineState,
    /// [0] = SRV of the linear color, [1] = UAV of `scratch`, [2] = UAV of
    /// the NGX output (the in-place expand target).
    heap: ID3D12DescriptorHeap,
    inc: u32,
    /// Display-res RGBA16F, matching `rr.output` — NGX reads it as `color`.
    scratch: Option<ID3D12Resource>,
    w: u32,
    h: u32,
    /// COM identities of the (src, out) pair the heap's descriptors currently
    /// name — what lets the per-dispatch `ensure` skip the descriptor rewrite
    /// on the steady state (see the method).
    described: (usize, usize),
}

impl TonePass {
    pub fn new(device: &ID3D12Device) -> Result<TonePass> {
        // [0] SRV table (t0), [1] UAV table (u0), [2] root constants (b0).
        // The GuidePass shape with one descriptor per table: the UAV table's
        // base is what selects scratch-vs-output between the two dispatches.
        let srv_range = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        }];
        let uav_range = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        }];
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: srv_range.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: uav_range.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 4,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            ..Default::default()
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut errb: Option<ID3DBlob> = None;
        unsafe {
            D3D12SerializeRootSignature(
                &rs_desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                Some(&mut errb),
            )
        }
        .map_err(|e| format!("ngxfg-tone: D3D12SerializeRootSignature: {e}"))?;
        let blob = blob.unwrap();
        let root_sig: ID3D12RootSignature = unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    blob.GetBufferPointer() as *const u8,
                    blob.GetBufferSize(),
                ),
            )
        }
        .map_err(|e| format!("ngxfg-tone: CreateRootSignature: {e}"))?;

        let mut pso_of = |entry: windows::core::PCSTR,
                          what: &str|
         -> Result<ID3D12PipelineState> {
            let cs = compile(TONE_HLSL, entry, s!("cs_5_0"), what)?;
            unsafe {
                device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root_sig),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: cs.GetBufferPointer(),
                        BytecodeLength: cs.GetBufferSize(),
                    },
                    ..Default::default()
                })
            }
            .map_err(|e| format!("ngxfg-tone: CreateComputePipelineState({what}): {e}").into())
        };
        let pso_compress = pso_of(s!("cs_tone_compress"), "ngxfg-tone cs_tone_compress")?;
        let pso_expand = pso_of(s!("cs_tone_expand"), "ngxfg-tone cs_tone_expand")?;

        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 3,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("ngxfg-tone: CreateDescriptorHeap: {e}"))?;
        let inc = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };

        Ok(TonePass {
            root_sig,
            pso_compress,
            pso_expand,
            heap,
            inc,
            scratch: None,
            w: 0,
            h: 0,
            described: (0, 0),
        })
    }

    /// Idempotent per-dispatch prologue — called EVERY frame, unlike
    /// `GuidePass::ensure` (which runs only at feature create/recreate): this
    /// pass has no recreate hook of its own, so it re-checks here. The scratch
    /// rebuilds when the DISPLAY dims move, and the three descriptors rewrite
    /// only when the scratch or the (src, out) COM identities changed — a
    /// descriptor in a shader-visible heap is read at EXECUTE time, so
    /// editing one the steady state's in-flight frames still reference would
    /// be a race; the identities only ever move across drained points
    /// (feature creation, or resize_output's `wait_idle` rebuilding
    /// `rr.output`/the FG out), which is what makes the rewrite safe exactly
    /// when it happens. The steady-state frame is two pointer compares.
    pub fn ensure(
        &mut self,
        device: &ID3D12Device,
        w: u32,
        h: u32,
        src: &ID3D12Resource,
        out: &ID3D12Resource,
    ) -> Result<()> {
        let key = (
            windows::core::Interface::as_raw(src) as usize,
            windows::core::Interface::as_raw(out) as usize,
        );
        let mut rewrite = self.described != key;
        if self.scratch.is_none() || self.w != w || self.h != h {
            self.scratch = Some(d3d12::committed_tex(
                device,
                w,
                h,
                DXGI_FORMAT_R16G16B16A16_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?);
            self.w = w;
            self.h = h;
            rewrite = true;
        }
        if rewrite {
            let base = unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() };
            let at =
                |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + (i * self.inc) as usize };
            unsafe {
                device.CreateShaderResourceView(src, None, at(0));
                device.CreateUnorderedAccessView(self.scratch.as_ref().unwrap(), None, None, at(1));
                device.CreateUnorderedAccessView(out, None, None, at(2));
            }
            self.described = key;
        }
        Ok(())
    }

    pub fn scratch(&self) -> &ID3D12Resource {
        self.scratch.as_ref().expect("ensure before scratch")
    }

    fn dispatch(&self, list: &ID3D12GraphicsCommandList, pso: &ID3D12PipelineState, uav: u32, mode: u8) {
        let gpu = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        let at = |i: u32| D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + (i * self.inc) as u64 };
        unsafe {
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootSignature(&self.root_sig);
            list.SetPipelineState(pso);
            list.SetComputeRootDescriptorTable(0, at(0));
            list.SetComputeRootDescriptorTable(1, at(uav));
            list.SetComputeRoot32BitConstant(2, self.w, 0);
            list.SetComputeRoot32BitConstant(2, self.h, 1);
            list.SetComputeRoot32BitConstant(2, mode as u32, 2);
            // Per-mode factor: Reinhard takes none (it is parameter-free).
            let k = match mode {
                1 => TONE_SCALE_K,
                3 => TONE_LOG_K,
                _ => 0.0,
            };
            list.SetComputeRoot32BitConstant(2, k.to_bits(), 3);
            list.Dispatch(self.w.div_ceil(8), self.h.div_ceil(8), 1);
        }
    }

    /// `rr.output` (NON_PIXEL_SHADER_RESOURCE, as the NGX pre-barrier already
    /// leaves it) -> `scratch` (UNORDERED_ACCESS). Caller transitions scratch
    /// to NON_PIXEL_SHADER_RESOURCE before the evaluate reads it.
    pub fn record_compress(&self, list: &ID3D12GraphicsCommandList, mode: u8) {
        self.dispatch(list, &self.pso_compress, 1, mode);
    }

    /// The NGX output, in place, while it is still UNORDERED_ACCESS. The UAV
    /// barrier is issued HERE rather than by the caller: it is what orders the
    /// evaluate's writes before this read, and separating it from the dispatch
    /// it protects is how that gets dropped in a later edit.
    pub fn record_expand(&self, list: &ID3D12GraphicsCommandList, out: &ID3D12Resource, mode: u8) {
        let b = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(out) },
                }),
            },
        };
        unsafe { list.ResourceBarrier(&[b]) };
        self.dispatch(list, &self.pso_expand, 2, mode);
    }
}

/// Pure-math gates (run by `--check`, DLL- and GPU-free).
///
/// Depth half: `clip_depth` IS the z-mapping of the `perspective_lh` matrix
/// the NGX dispatch carries (matrix-consistency sweep + the
/// near/far/harmonic-midpoint anchors).
///
/// MV half: the virtual-image reprojection against `CamBasis::project`
/// itself — (b) a static camera reprojects every virtual point to its own
/// pixel (zero MV at any reflection depth); (c) `t_r = 0` degenerates to the
/// surface reprojection `render.rs::write_gbuf_hit` computes, pinned against
/// the REAL `CamBasis::project` on a dollied+yawed pose (the continuity pin
/// that proves the mirrored matrix math matches the plane's convention);
/// (d) lateral strafe with the reflection at `far` moves the virtual pixel
/// far less than the surface pixel — the user's strafe observation, as a
/// gate; (e) the blend-weight anchors.
///
/// Round-3 (firefly MV) half, on the `ff_guide_rows`/`ff_mv_blend` twins:
/// off arms exact (day/disabled/zero-count bake ffc 0; empty-table blend
/// bit-identical), the moving-firefly gate with its anti-vacuity and TEETH
/// pins (the pass-through surface MV must FAIL the bound), occlusion +
/// behind-camera drops, the weight-continuity sweep (the skirt pin), and
/// the projection-route/σ-lum anchors against `CamBasis::project` and the
/// shipped `fireflies::glow`.
pub fn self_test() -> std::result::Result<(), String> {
    use crate::camera::Camera;
    // ---- depth half ----
    let cases = [(0.0421f32, 421.0f32), (0.1, 10_000.0), (1.0, 100.0)];
    for &(near, far) in &cases {
        let d_near = clip_depth(near, near, far);
        let d_far = clip_depth(far, near, far);
        if d_near.abs() > 1e-6 {
            return Err(format!("clip_depth(near) = {d_near}, want 0 (near={near} far={far})"));
        }
        if (d_far - 1.0).abs() > 1e-6 {
            return Err(format!("clip_depth(far) = {d_far}, want 1 (near={near} far={far})"));
        }
        let zm = 2.0 * near * far / (near + far);
        let d_mid = clip_depth(zm, near, far);
        if (d_mid - 0.5).abs() > 1e-5 {
            return Err(format!("clip_depth(harmonic mid) = {d_mid}, want 0.5"));
        }
        let m = Mat4::perspective_lh(1.0, 16.0 / 9.0, near, far);
        let mut prev = -1.0f32;
        for i in 0..64 {
            let t = i as f32 / 63.0;
            let z = near * (far / near).powf(t);
            let clip = m * Vec4::new(0.0, 0.0, z, 1.0);
            let want = clip.z / clip.w;
            let got = clip_depth(z, near, far);
            if (got - want).abs() > 1e-5 {
                return Err(format!(
                    "clip_depth({z}) = {got} but perspective_lh gives {want} \
                     (near={near} far={far})"
                ));
            }
            if got < prev {
                return Err(format!("clip_depth not monotone at z={z}"));
            }
            prev = got;
        }
    }
    if clip_depth(1e30, 0.1, 10_000.0) != 1.0 {
        return Err("clip_depth beyond far must saturate to 1.0".into());
    }

    // ---- MV half ----
    let (rw, rh) = (432usize, 324usize);
    let (near, far) = (0.05f32, 5000.0f32);
    let cam = Camera::look_at(Vec3A::new(4.0, 2.0, 4.0), Vec3A::new(0.0, 1.0, 0.0), 0.9);
    // The dispatch's basis derivation (fc.right/up are unit; pre-scale like
    // CamBasis).
    let basis_fields = |c: &Camera| {
        let f = c.forward();
        let r = f.cross(Vec3A::Y).normalize();
        let u = r.cross(f);
        let tanh = (c.fov_y * 0.5).tan();
        (c.pos, f, r * (tanh * rw as f32 / rh as f32), u * tanh)
    };
    let (org, fwd, rgt, upv) = basis_fields(&cam);
    let mats = crate::dlss::cam_matrices(&cam, rw, rh, near, far);
    let world_to_clip = mats.view_to_clip * mats.world_to_view;

    // (b) static camera: prev == current ⇒ the virtual point reprojects to
    // its own pixel at ANY reflection depth (a still frame's reflection has
    // zero MV — the model's sanity anchor).
    for &(cx, cy, t_r) in
        &[(216.5f32, 162.5f32, 0.0f32), (40.5, 300.5, 3.0), (400.5, 20.5, far)]
    {
        let (px, py) = virtual_prev_px(
            cx, cy, rw as f32, rh as f32, 7.0, t_r, far, org, fwd, rgt, upv, &world_to_clip,
        )
        .ok_or("static virtual point behind its own camera")?;
        if (px - cx).abs() > 2e-2 || (py - cy).abs() > 2e-2 {
            return Err(format!(
                "static camera: virtual pixel drifted ({cx},{cy}) -> ({px},{py}) at t_r={t_r}"
            ));
        }
    }

    // A genuinely moved previous pose for (c) and (d').
    let prev_cam = Camera {
        pos: cam.pos + Vec3A::new(0.12, 0.02, -0.05),
        yaw: cam.yaw + 0.01,
        pitch: cam.pitch - 0.004,
        fov_y: cam.fov_y,
    };
    let prev_basis = prev_cam.basis(rw, rh);
    let prev_mats = crate::dlss::cam_matrices(&prev_cam, rw, rh, near, far);
    let world_to_prev_clip = prev_mats.view_to_clip * prev_mats.world_to_view;

    // (c) t_r = 0 degenerates to the surface reprojection — pinned against
    // CamBasis::project (the exact function the MV fill sites use), which
    // proves the matrix route and the basis route agree in the plane's own
    // convention.
    for &(cx, cy, view_z) in &[(100.5f32, 80.5f32, 3.0f32), (300.5, 250.5, 11.0), (16.5, 16.5, 0.8)]
    {
        let ndx = cx * (2.0 / rw as f32) - 1.0;
        let ndy = 1.0 - cy * (2.0 / rh as f32);
        let du = (fwd + rgt * ndx + upv * ndy).normalize();
        let p = org + du * (view_z / du.dot(fwd));
        let (ex, ey) = prev_basis
            .project(p - prev_basis.origin)
            .ok_or("surface point behind prev camera")?;
        let (px, py) = virtual_prev_px(
            cx, cy, rw as f32, rh as f32, view_z, 0.0, far, org, fwd, rgt, upv,
            &world_to_prev_clip,
        )
        .ok_or("virtual point behind prev camera")?;
        if (px - ex).abs() > 0.05 || (py - ey).abs() > 0.05 {
            return Err(format!(
                "t_r=0 continuity: matrix route ({px},{py}) vs CamBasis::project ({ex},{ey})"
            ));
        }
    }

    // (d) PURE LATERAL STRAFE with a MISSED reflection (the sky). The camera
    // only translates, and the sky is at infinity, so the correct virtual MV
    // is EXACTLY ZERO — hence an ABSOLUTE bound, not a fraction of the surface
    // MV. That distinction is the whole gate: the old `mv_virt <= 0.05 *
    // mv_surf` form passed while production visibly warped, because zero is
    // the right answer and any percentage of a large surface MV clears a
    // percentage bar. It also ran at far = 5000 while the renderer ships
    // far = 2*diag (~138 on the world) — 36x more distant, so "reflection at
    // far" impersonated infinity in the gate far better than it ever did in
    // the product. Both halves are fixed here: the bound is absolute, and the
    // sweep includes a PRODUCTION-scale far.
    //
    // The sky-miss branch makes this exact rather than merely small: a
    // direction reprojected under a pure translation is unchanged.
    for &far_probe in &[far, 2.0 * 69.0, 2.0 * 10.0] {
        let strafe_cam = Camera { pos: cam.pos + rgt.normalize() * 0.3, ..cam };
        let strafe_mats = crate::dlss::cam_matrices(&strafe_cam, rw, rh, near, far_probe);
        let world_to_strafe_clip = strafe_mats.view_to_clip * strafe_mats.world_to_view;
        let strafe_basis = strafe_cam.basis(rw, rh);
        let (cx, cy, view_z) = (216.5f32, 162.5f32, 2.5f32);
        let ndx = cx * (2.0 / rw as f32) - 1.0;
        let ndy = 1.0 - cy * (2.0 / rh as f32);
        let du = (fwd + rgt * ndx + upv * ndy).normalize();
        let p = org + du * (view_z / du.dot(fwd));
        let (sx, sy) = strafe_basis
            .project(p - strafe_basis.origin)
            .ok_or("strafe: surface point behind camera")?;
        let mv_surf = ((sx - cx).powi(2) + (sy - cy).powi(2)).sqrt();
        // t_r == far_probe IS the miss encoding the pack produces.
        let (vx, vy) = virtual_prev_px(
            cx, cy, rw as f32, rh as f32, view_z, far_probe, far_probe, org, fwd, rgt, upv,
            &world_to_strafe_clip,
        )
        .ok_or("strafe: virtual point behind camera")?;
        let mv_virt = ((vx - cx).powi(2) + (vy - cy).powi(2)).sqrt();
        if mv_surf < 5.0 {
            return Err(format!(
                "strafe gate is vacuous — surface MV only {mv_surf} px (probe too gentle)"
            ));
        }
        if mv_virt > 0.05 {
            return Err(format!(
                "strafe (far={far_probe}): reflected-sky virtual MV {mv_virt} px, want ~0 \
                 (surface MV {mv_surf} px) — a missed reflection is at INFINITY and must \
                 not translate; this is the sun-highlight jump on generated frames"
            ));
        }
        // GATE TEETH. The pre-fix behaviour is still reachable: cam_far =
        // INFINITY makes the miss branch unreachable, leaving the old point
        // form that placed the sky at a finite `far`. At a PRODUCTION-scale
        // far that must BLOW this bound — if it does not, the probe is too
        // gentle to have caught the bug that shipped, and a future regression
        // would pass unnoticed. (At the old test's far = 5000 it legitimately
        // does not fire, which is precisely why that value hid the defect.)
        if far_probe < 1000.0 {
            let (ox, oy) = virtual_prev_px(
                cx, cy, rw as f32, rh as f32, view_z, far_probe, f32::INFINITY, org, fwd, rgt,
                upv, &world_to_strafe_clip,
            )
            .ok_or("strafe: pre-fix virtual point behind camera")?;
            let mv_old = ((ox - cx).powi(2) + (oy - cy).powi(2)).sqrt();
            if mv_old <= 0.05 {
                return Err(format!(
                    "strafe (far={far_probe}) gate is TOOTHLESS: the pre-fix point form \
                     also lands at {mv_old} px, so this probe could never have caught the \
                     sky-reflection warp"
                ));
            }
        }
    }

    // (e) blend-weight anchors: metal (diffuse ~0) ⇒ ~1; white dielectric ⇒
    // ~F0 fraction; past ROUGH_HI ⇒ exactly 0.
    let w_metal = rmv_weight(0.0, 0.55, 0.2);
    if w_metal < 0.95 {
        return Err(format!("rmv_weight(metal) = {w_metal}, want ~1"));
    }
    let w_diel = rmv_weight(0.8, 0.04, 0.2);
    if !(0.01..=0.1).contains(&w_diel) {
        return Err(format!("rmv_weight(white dielectric) = {w_diel}, want ~0.05"));
    }
    if rmv_weight(0.0, 0.55, ROUGH_HI + 0.01) != 0.0 {
        return Err("rmv_weight past ROUGH_HI must be exactly 0".into());
    }

    // ---- round 3: firefly glow MVs ----
    use crate::fireflies::{Fireflies, FF_COLOR, FF_GLOW_L_MAX};
    let lum709 = |c: glam::Vec3A| 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
    let (fcmin, fcmax) = (Vec3A::new(-6.0, 0.0, -6.0), Vec3A::new(6.0, 4.0, 6.0));

    // (b) OFF ARMS EXACT: day / disabled / zero count all bake ffc == 0, and
    // an empty table's blend returns the input BIT-identically — the
    // structural-off proof the kernel's `if (ffc != 0)` guard mirrors.
    for (en, night, cnt) in [(true, 0.0f32, 8u32), (false, 1.0, 8), (true, 1.0, 0)] {
        let ff = Fireflies::new(en, cnt, night, fcmin, fcmax, 5.0);
        let t = ff_guide_rows(
            &ff, &ff, rw as f32, rh as f32, cam.fov_y, org, fwd, &world_to_clip,
            &world_to_clip,
        );
        if t.ffc != 0 {
            return Err(format!(
                "ff off arm (en={en} night={night} n={cnt}): ffc = {}, want 0",
                t.ffc
            ));
        }
    }
    {
        let mv_in = (0.123f32, -4.56f32);
        let out = ff_mv_blend(mv_in, 100.5, 100.5, 3.0, &FfTable::empty());
        if out != mv_in {
            return Err(format!("ff empty-table blend must be bit-identical: {out:?}"));
        }
    }

    // (a) MOVING FIREFLY, STATIC CAMERA: the blend at a splat center must
    // land on the firefly's own closed-form displacement. Real bakes (the
    // same `pose` production runs); dt scanned so the probe is never
    // vacuous — the anti-vacuity assert fails loudly if no firefly moves
    // >= 5 px, rather than the gate silently passing on nothing.
    {
        let bake_at = |dt: f32| {
            let cur = Fireflies::new(true, 6, 1.0, fcmin, fcmax, 9.0);
            let prv = Fireflies::new(true, 6, 1.0, fcmin, fcmax, 9.0 - dt);
            ff_guide_rows(
                &cur, &prv, rw as f32, rh as f32, cam.fov_y, org, fwd, &world_to_clip,
                &world_to_clip,
            )
        };
        let scan = |t: &FfTable| {
            let mut best = (0usize, 0.0f32);
            for i in 0..t.ffc as usize {
                let m = (t.b[i][0].powi(2) + t.b[i][1].powi(2)).sqrt();
                if m > best.1 {
                    best = (i, m);
                }
            }
            best
        };
        let mut table = bake_at(0.5);
        let mut best = scan(&table);
        for dt in [1.0f32, 2.0, 4.0, 8.0] {
            if best.1 >= 5.0 {
                break;
            }
            table = bake_at(dt);
            best = scan(&table);
        }
        if best.1 < 5.0 {
            return Err(format!(
                "ff moving gate is vacuous — max firefly MV {} px after the dt scan \
                 (probe too gentle)",
                best.1
            ));
        }
        let i = best.0;
        let (cx, cy) = (table.a[i][0], table.a[i][1]);
        let want = (table.b[i][0], table.b[i][1]);
        let got = ff_mv_blend((0.0, 0.0), cx, cy, far, &table);
        let err = ((got.0 - want.0).powi(2) + (got.1 - want.1).powi(2)).sqrt();
        // The asymptotic weight never quite reaches 1 (w = S/(S+lref)), so
        // the bound carries a relative and an absolute term.
        let bound = 0.1 * best.1 + 0.5;
        if err > bound {
            return Err(format!(
                "ff moving: blend ({},{}) vs closed-form MV ({},{}) — err {err} px > {bound}",
                got.0, got.1, want.0, want.1
            ));
        }
        // GATE TEETH: the pre-fix answer — the pass-through surface MV
        // (0,0) — must FAIL the same bound, or this probe could never have
        // caught the strobe it exists for.
        if best.1 <= bound {
            return Err(format!(
                "ff moving gate is TOOTHLESS: the pass-through (0,0) also lands within \
                 {bound} px of the {won} px displacement",
                won = best.1
            ));
        }
    }

    // (c) OCCLUSION + BEHIND-CAMERA: a firefly behind the primary hit leaves
    // the surface MV bit-identical; a firefly behind the CURRENT camera —
    // or one whose PREVIOUS projection fails — is dropped at bake, so the
    // surface MV survives rather than an invented one.
    {
        let mut t = FfTable::empty();
        t.ffc = 1;
        t.a[0] = [200.5, 150.5, 10.0, 3.0];
        t.b[0] = [8.0, -3.0, 400.0, 0.0];
        let mv_in = (1.25f32, -0.5f32);
        let occ = ff_mv_blend(mv_in, 200.5, 150.5, 5.0, &t);
        if occ != mv_in {
            return Err(format!("ff occluded blend must be bit-identical: {occ:?}"));
        }
        let vis = ff_mv_blend(mv_in, 200.5, 150.5, far, &t);
        if (vis.0 - mv_in.0).abs() < 1.0 {
            return Err(format!("ff visible blend did not move: {vis:?}"));
        }

        let mut one = Fireflies::off();
        one.count = 1;
        one.scale = (fcmax - fcmin).length();
        // Behind the CURRENT camera (cam at (4,2,4) looking at the origin).
        one.pos[0] = [8.0, 3.0, 8.0, 1.0];
        let tb = ff_guide_rows(
            &one, &one, rw as f32, rh as f32, cam.fov_y, org, fwd, &world_to_clip,
            &world_to_clip,
        );
        if tb.ffc != 0 {
            return Err("ff behind current camera must drop at bake".into());
        }
        // Visible now, but the PREV camera faces away — the prev projection
        // fails and the row drops.
        one.pos[0] = [0.4, 1.3, -0.2, 1.0];
        let away = Camera::look_at(cam.pos, cam.pos + Vec3A::new(4.0, 1.0, 4.0), cam.fov_y);
        let away_mats = crate::dlss::cam_matrices(&away, rw, rh, near, far);
        let away_clip = away_mats.view_to_clip * away_mats.world_to_view;
        let tb = ff_guide_rows(
            &one, &one, rw as f32, rh as f32, cam.fov_y, org, fwd, &world_to_clip, &away_clip,
        );
        if tb.ffc != 0 {
            return Err("ff prev-projection failure must drop at bake".into());
        }
    }

    // (d) WEIGHT CONTINUITY: radial sweep across a capped-brightness splat —
    // monotone, near-1 core, EXACTLY 0 past the 9.22 cut, and no step: the
    // pin the (exp − 1e-4)⁺ skirt exists for (drop the skirt and the reject
    // boundary steps the weight by ~0.17 on a bright splat).
    {
        let mut t = FfTable::empty();
        let sigma = 3.0f32;
        t.ffc = 1;
        t.a[0] = [200.5, 150.5, 1.0, sigma];
        t.b[0] = [10.0, 0.0, lum709(FF_COLOR) * FF_GLOW_L_MAX, 0.0];
        let mut prev_w = f32::INFINITY;
        // 0.005σ spacing: the smooth tail's per-step delta stays ~0.005,
        // half the 0.01 bound, while the skirt-less cut discontinuity is
        // ~0.17 at ANY spacing — the pin separates slope from step.
        for k in 0..=1100 {
            let r = k as f32 * 0.005 * sigma;
            let out = ff_mv_blend((0.0, 0.0), 200.5 + r, 150.5, far, &t);
            let wk = out.0 / 10.0;
            if k == 0 && wk <= 0.99 {
                return Err(format!("ff weight core w(0) = {wk}, want > 0.99"));
            }
            if wk > prev_w + 1e-6 {
                return Err(format!("ff weight not monotone at r = {r} px ({wk} > {prev_w})"));
            }
            if prev_w.is_finite() && (prev_w - wk).abs() >= 0.01 {
                return Err(format!(
                    "ff weight steps {} at r = {r} px — the skirt pin",
                    prev_w - wk
                ));
            }
            if r >= 4.35 * sigma && wk != 0.0 {
                return Err(format!("ff weight past the 9.22 cut must be exactly 0 ({wk})"));
            }
            prev_w = wk;
        }
    }

    // (e) PROJECTION-ROUTE ANCHOR + (f) σ/LUM PARITY: one synthetic firefly
    // at a known point — the bake's matrix projections must agree with
    // CamBasis::project (the MV fill sites' own function) in the plane's
    // pixel convention, and the baked center luminance must equal the
    // shipped `fireflies::glow` evaluated on-axis (ties the σ/pixel_cone
    // conversion to the one glow implementation; drift here = the splat and
    // its MV capture disagree about where the glow IS).
    {
        let p = Vec3A::new(0.4, 1.3, -0.2);
        let mut one = Fireflies::off();
        one.count = 1;
        one.scale = (fcmax - fcmin).length();
        one.pos[0] = [p.x, p.y, p.z, 1.0];
        let tb = ff_guide_rows(
            &one, &one, rw as f32, rh as f32, cam.fov_y, org, fwd, &world_to_clip,
            &world_to_prev_clip,
        );
        if tb.ffc != 1 {
            return Err(format!("ff anchor: ffc = {}, want 1", tb.ffc));
        }
        let basis = cam.basis(rw, rh);
        let (ex, ey) =
            basis.project(p - basis.origin).ok_or("ff anchor point behind camera")?;
        if (tb.a[0][0] - ex).abs() > 0.05 || (tb.a[0][1] - ey).abs() > 0.05 {
            return Err(format!(
                "ff anchor cur px: bake ({},{}) vs CamBasis::project ({ex},{ey})",
                tb.a[0][0], tb.a[0][1]
            ));
        }
        let (px, py) = prev_basis
            .project(p - prev_basis.origin)
            .ok_or("ff anchor point behind prev camera")?;
        let (wmx, wmy) = (px - ex, py - ey);
        if (tb.b[0][0] - wmx).abs() > 0.1 || (tb.b[0][1] - wmy).abs() > 0.1 {
            return Err(format!(
                "ff anchor mv: bake ({},{}) vs basis route ({wmx},{wmy})",
                tb.b[0][0], tb.b[0][1]
            ));
        }
        let pixel_cone = 2.0 * (cam.fov_y * 0.5).tan() / rh as f32;
        let d = (p - org).normalize();
        let g = crate::fireflies::glow(&one, org, d, f32::INFINITY, pixel_cone * 0.5);
        let want_lum = lum709(g);
        let got_lum = tb.b[0][2];
        if (got_lum - want_lum).abs() > 1e-3 * want_lum.max(1e-6) {
            return Err(format!(
                "ff σ/lum parity: bake {got_lum} vs fireflies::glow {want_lum}"
            ));
        }
    }

    // ---- ROUND 4: ripple-aware virtual MVs (water) ----
    //
    // The bug: water's mirror normal moves every frame on the cloud clock, so
    // its reflected image slides across a surface whose geometry — and
    // therefore whose MV — is perfectly still. A parked camera hands frame
    // generation two different reflections and a zero MV.
    let diag = 69.0f32; // the world's scale; every ripple length is diag-relative
    let amp = crate::scene::WATER_RIPPLE_AMP;
    let n_flat = Vec3A::Y; // an ocean: flat geometry, all the motion in the field

    // (r-a) OFF-ARM BIT-IDENTITY. amp == 0 must be the still-mirror twin
    // BITWISE, not approximately — the round-4 path is opt-in per pixel and
    // every non-water pixel in the world takes this arm.
    for &(cx, cy, view_z, t_r) in &[
        (10.5f32, 20.5f32, 3.0f32, 4.0f32),
        (216.5, 162.5, 12.0, 0.5),
        (400.5, 300.5, 2.0, far),
    ] {
        let want = virtual_prev_px(
            cx, cy, rw as f32, rh as f32, view_z, t_r, far, org, fwd, rgt, upv,
            &world_to_prev_clip,
        );
        let got = ripple_virtual_prev_px(
            cx, cy, rw as f32, rh as f32, view_z, t_r, far, org, fwd, rgt, upv,
            &world_to_prev_clip, n_flat, 1.0, 2.0, 0.0, diag,
        );
        if want.map(|(a, b)| (a.to_bits(), b.to_bits()))
            != got.map(|(a, b)| (a.to_bits(), b.to_bits()))
        {
            return Err(format!("ripple off-arm not bit-identical: {want:?} vs {got:?}"));
        }
    }

    // (r-b) THE GENERALIZATION PIN. At t_prev == t_cur the reconstructed
    // previous normal is n_c, and reflect() being an involution makes the
    // content direction exactly du — so the whole thing must collapse onto
    // the still-mirror unfold even with the ripple ARMED. Both branches
    // (finite hit and reflected sky) and several depths, since one expression
    // now serves both.
    for &(view_z, t_r) in &[(3.0f32, 4.0f32), (12.0, 0.5), (2.0, far), (7.0, far * 0.999)] {
        let want = virtual_prev_px(
            216.5, 162.5, rw as f32, rh as f32, view_z, t_r, far, org, fwd, rgt, upv,
            &world_to_prev_clip,
        );
        let got = ripple_virtual_prev_px(
            216.5, 162.5, rw as f32, rh as f32, view_z, t_r, far, org, fwd, rgt, upv,
            &world_to_prev_clip, n_flat, 5.0, 5.0, amp, diag,
        );
        match (want, got) {
            (Some((wx, wy)), Some((gx, gy))) => {
                if (wx - gx).abs() > 1e-3 || (wy - gy).abs() > 1e-3 {
                    return Err(format!(
                        "ripple dt=0 must equal the still-mirror unfold: \
                         ({wx},{wy}) vs ({gx},{gy}) at view_z={view_z} t_r={t_r}"
                    ));
                }
            }
            (a, b) => return Err(format!("ripple dt=0 branch mismatch: {a:?} vs {b:?}")),
        }
    }

    // (r-c) THE MUST-FIRE, WITH TEETH. A PARKED camera (prev == current
    // matrices) over rippling water: the surface MV is exactly zero, the
    // still-mirror unfold is exactly zero — and the true content motion is
    // not. Scan dt until the reconstruction reports real motion; never
    // reaching it means the probe is vacuous, which is itself a failure.
    {
        let (cx, cy, view_z, t_r) = (216.5f32, 162.5f32, 6.0f32, 30.0f32);
        let mut fired = None;
        for k in 1..=24 {
            let dt = k as f32 / 60.0;
            let (mx, my) = ripple_virtual_prev_px(
                cx, cy, rw as f32, rh as f32, view_z, t_r, far, org, fwd, rgt, upv,
                &world_to_clip, // PARKED: prev matrices == current
                n_flat, dt, 0.0, amp, diag,
            )
            .ok_or("ripple parked probe went behind the image plane")?;
            let mv = ((mx - cx).powi(2) + (my - cy).powi(2)).sqrt();
            if mv >= 5.0 {
                fired = Some((dt, mv));
                break;
            }
        }
        let Some((dt, mv)) = fired else {
            return Err(
                "ripple parked probe never reached 5 px of motion — the gate is vacuous".into(),
            );
        };
        // TEETH: the still-mirror answer at the SAME parked pose. It is the
        // pre-round-4 behavior AND the surface MV, and both are ~0 — if this
        // passed the bound the probe would be measuring nothing.
        let (ox, oy) = virtual_prev_px(
            cx, cy, rw as f32, rh as f32, view_z, t_r, far, org, fwd, rgt, upv, &world_to_clip,
        )
        .ok_or("ripple teeth probe behind the image plane")?;
        let mv_old = ((ox - cx).powi(2) + (oy - cy).powi(2)).sqrt();
        if mv_old >= 1.0 {
            return Err(format!(
                "ripple teeth: the still-mirror unfold moved {mv_old} px at a PARKED \
                 camera — it must be ~0, so this probe cannot show round 4 works"
            ));
        }
        // ORACLE: an inverse round-trip, not a re-derivation of the same
        // formula. Rebuild the ray through the answer pixel, reflect it off
        // the reconstructed previous normal, and it must return the CURRENT
        // content direction. A sign error or a wrong normal fails this while
        // a duplicated formula would not.
        let ndx = cx * (2.0 / rw as f32) - 1.0;
        let ndy = 1.0 - cy * (2.0 / rh as f32);
        let du = (fwd + rgt * ndx + upv * ndy).normalize();
        let hit_p = org + du * (view_z / du.dot(fwd));
        let n_p = ripple_prev_normal(n_flat, hit_p, dt, 0.0, amp, diag);
        let refl = |i: Vec3A, n: Vec3A| i - n * (2.0 * i.dot(n));
        let r_cur = refl(du, n_flat);
        let back = refl(refl(r_cur, n_p), n_p); // reflect off n_p and back
        if (back - r_cur).length() > 1e-5 {
            return Err(format!(
                "ripple oracle: reflecting off n_p twice did not return the current \
                 content direction (err {})",
                (back - r_cur).length()
            ));
        }
        if n_p.dot(n_flat) > 0.999_999 {
            return Err("ripple oracle: n_p never left n_c — the field is not animating".into());
        }
        eprintln!(
            "ngxfg-guides: ripple parked-camera MV fires at dt={dt:.3}s \
             ({mv:.2} px vs still-mirror {mv_old:.4})"
        );
    }

    // (r-d) CONTINUITY AT THE STRUCTURAL CUT. The kernel branches on
    // amp > 0, so the blend must approach the off arm as amp shrinks rather
    // than stepping at the boundary.
    {
        let (cx, cy, view_z, t_r) = (216.5f32, 162.5f32, 6.0f32, 30.0f32);
        let at = |a: f32| {
            ripple_virtual_prev_px(
                cx, cy, rw as f32, rh as f32, view_z, t_r, far, org, fwd, rgt, upv,
                &world_to_clip, n_flat, 0.05, 0.0, a, diag,
            )
        };
        let (bx, by) = at(0.0).ok_or("ripple continuity: off arm missing")?;
        let mut prev_d = f32::INFINITY;
        for k in 0..6 {
            let a = amp * 0.5f32.powi(k);
            let (px, py) = at(a).ok_or("ripple continuity: armed arm missing")?;
            let d = ((px - bx).powi(2) + (py - by).powi(2)).sqrt();
            if d > prev_d + 1e-3 {
                return Err(format!("ripple continuity not monotone in amp at amp={a}"));
            }
            prev_d = d;
        }
        if prev_d > 0.5 {
            return Err(format!("ripple amp -> 0 left a {prev_d} px step at the cut"));
        }
    }

    // (r-e) HUMILITY. An adversarial gradient that would flip the normal
    // through the horizon must fall back to n_c exactly — coarser, never
    // wrong-signed. (A huge amp is the cheap way to force it.)
    {
        let hit_p = org + fwd * 5.0;
        let n_p = ripple_prev_normal(n_flat, hit_p, 0.0, 40.0, 1.0e6, diag);
        if n_p.dot(n_flat) <= 0.0 {
            return Err("ripple humility: n_p crossed the horizon instead of falling back".into());
        }
    }

    // (r-f) THE MODE BOUNDARY. t_r just under cam_far (a real, very distant
    // hit) and t_r == cam_far (the sky encoding) take DIFFERENT projection
    // routes — point vs direction. They must not disagree wildly, or a
    // ripple that nudges a reflection across the boundary snaps the MV.
    {
        let (cx, cy, view_z) = (216.5f32, 162.5f32, 6.0f32);
        let near_far = ripple_virtual_prev_px(
            cx, cy, rw as f32, rh as f32, view_z, far * (1.0 - 1e-4), far, org, fwd, rgt, upv,
            &world_to_clip, n_flat, 0.05, 0.0, amp, diag,
        )
        .ok_or("ripple boundary: finite arm missing")?;
        let at_far = ripple_virtual_prev_px(
            cx, cy, rw as f32, rh as f32, view_z, far, far, org, fwd, rgt, upv,
            &world_to_clip, n_flat, 0.05, 0.0, amp, diag,
        )
        .ok_or("ripple boundary: sky arm missing")?;
        let d = ((near_far.0 - at_far.0).powi(2) + (near_far.1 - at_far.1).powi(2)).sqrt();
        // The two differ by the parallax of a point at `far` vs one at
        // infinity, which is bounded by |hit_p - org| / far in angle.
        if d > 2.0 {
            return Err(format!("ripple finite/sky boundary jumps {d} px"));
        }
    }

    // (r-g) FRESNEL ANCHORS (round 4b). Grazing water is nearly all mirror;
    // face-on is nearly none of it. The proxy this replaces is flat.
    {
        if (fresnel_spec_scale(1.0) - 1.0).abs() > 1e-6 {
            return Err("fresnel_spec_scale(face-on) must be exactly 1 (F0/F0)".into());
        }
        let grazing = fresnel_spec_scale(0.0);
        if (grazing - 25.0).abs() > 1e-3 {
            return Err(format!("fresnel_spec_scale(grazing) = {grazing}, want 1/0.04 = 25"));
        }
        // The point of the change: a dark-albedo water pixel at grazing must
        // hand the MV over almost entirely, where the flat proxy did not.
        let (ld, ls) = (0.10f32, 0.04f32);
        let w_flat = rmv_weight(ld, ls, 0.05);
        let w_graze = rmv_weight(ld, ls * fresnel_spec_scale(0.05), 0.05);
        if w_flat > 0.5 {
            return Err(format!("water proxy weight {w_flat} — probe no longer representative"));
        }
        if w_graze < 0.85 {
            return Err(format!("fresnel grazing weight {w_graze}, want ~1"));
        }
        // Past ROUGH_HI the damping still wins outright, Fresnel or not.
        if rmv_weight(ld, ls * fresnel_spec_scale(0.0), ROUGH_HI + 0.01) != 0.0 {
            return Err("fresnel must not resurrect the past-ROUGH_HI zero".into());
        }
    }

    eprintln!("ngxfg-guides self-test: OK");
    Ok(())
}
