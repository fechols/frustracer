//! CPU mirrors of the frame-generation guide pass's reprojection math.
//!
//! These are the Rust twins of `cs_guides` (`shaders/` via
//! `gpu/ngxfg_guides.rs`) — pure `glam`, no device, no resource, no API. They
//! live here because they are CROSS-PINNED from two directions and one of the
//! pinners is platform-neutral: `frd::oracle`'s virtual-motion family checks
//! its own unfold against `virtual_prev_px` so that FRD and the NGX-FG guide
//! pass provably share ONE virtual-image convention ("one unfold, two engines,
//! one pin" — see the `--fg` round-2 and FRD v1.5.1 notes in CLAUDE.md).
//!
//! Left in `gpu/ngxfg_guides.rs` that pin was reachable only on Windows, so
//! `--check`'s FRD gate silently lost a tooth everywhere else. A gate that
//! evaporates on a platform is worse than no gate, because the suite still
//! prints green.

use glam::{Mat4, Vec3A, Vec4};

/// The CPU mirror of `cs_guides`' virtual-image reprojection: the PREVIOUS
/// frame's pixel position of the virtual point behind pixel center (cx, cy),
/// or None when it lands behind the previous image plane. `right_s`/`up_s`
/// carry the CamBasis pre-scaling (tan(fov/2)·aspect / tan(fov/2));
/// `m` = world → previous clip (glam column-vector convention).
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

// The last of the shader-source/probe gates that `cargo test` could not reach
// off Windows, moved here for the reason in this module's header: it lived in
// `gpu/ngxfg_guides.rs` beside the fade it justifies, and calls nothing but
// `crate::shade` and `crate::scene`, so the platform gate was incidental.
// The empirical basis of the round-4 dt story, pinned so it cannot silently
// rot: the ripple field stays COHERENT at low-framerate clock deltas (the
// waves are slow and wide — normal swing ~1.4 deg mean at 150 ms) and the
// UNFADED first-order reconstruction stays near-exact out to 250 ms. The
// `ripple_dt_weight` fade therefore exists for the resulting MV's MAGNITUDE
// (200-550 px/frame at 8K density — past what the FG warp consumes), NOT for
// reconstruction accuracy — if a field retune ever breaks either premise,
// this test is what says the fade needs re-deriving.
#[cfg(test)]
mod ripple_probe {
    use glam::Vec3A;

    #[test]
    fn field_coherence_stats() {
        let diag = 69.0f32; // THE WORLD's scale
        let amp = crate::scene::WATER_RIPPLE_AMP;
        let n = Vec3A::Y;
        let ang = |a: Vec3A, b: Vec3A| a.dot(b).clamp(-1.0, 1.0).acos().to_degrees();
        for &dt in &[1.0f32 / 60.0, 1.0 / 30.0, 0.05, 0.1, 0.15, 0.25] {
            let (mut sw_max, mut sw_sum) = (0f32, 0f32);
            let (mut er_max, mut er_sum) = (0f32, 0f32);
            let mut cnt = 0u32;
            for i in 0..12 {
                for j in 0..12 {
                    let p = Vec3A::new(i as f32 * 2.13 - 12.0, 0.0, j as f32 * 1.87 - 11.0);
                    for pi in 0..6 {
                        let t0 = pi as f32 * 2.29;
                        let t1 = t0 + dt;
                        let n_c = crate::shade::ripple_normal(n, n, p, t1, amp, diag);
                        let truth = crate::shade::ripple_normal(n, n, p, t0, amp, diag);
                        // The UNFADED first-order reconstruction (the pre-fade
                        // kernel, replicated so the shipping fade doesn't hide it).
                        let gd = (crate::shade::ripple_grad(p, t0, diag)
                            - crate::shade::ripple_grad(p, t1, diag))
                            * amp;
                        let g3 = Vec3A::new(gd.x, 0.0, gd.y);
                        let n_p = (n_c - (g3 - n_c * g3.dot(n_c))).normalize();
                        let swing = ang(n_c, truth);
                        let err = ang(n_p, truth);
                        sw_max = sw_max.max(swing);
                        sw_sum += swing;
                        er_max = er_max.max(err);
                        er_sum += err;
                        cnt += 1;
                    }
                }
            }
            // ~px per radian at 8K (4320 px over fov_y 0.9); reflected content
            // moves at 2x the normal swing, sky content maps 1:1 through it.
            let px8k = 4470.0f32;
            let mv = |deg: f32| 2.0 * deg.to_radians() * px8k;
            eprintln!(
                "dt={dt:.3}s | normal swing mean {:.3} max {:.3} deg | rec err mean {:.4} \
                 max {:.4} deg | reflected MV @8K mean {:.0} max {:.0} px | MV err mean \
                 {:.1} max {:.1} px",
                sw_sum / cnt as f32,
                sw_max,
                er_sum / cnt as f32,
                er_max,
                mv(sw_sum / cnt as f32),
                mv(sw_max),
                mv(er_sum / cnt as f32),
                mv(er_max)
            );
            // COHERENCE: the field must remain smooth at low-framerate deltas
            // (measured max 3.5 deg at 150 ms / 5.3 deg at 250 ms — a field
            // that swings tens of degrees per frame would re-open the
            // confident-wrong-MV class the fade's zero answer relies on
            // being mild).
            if sw_max > 12.0 {
                panic!(
                    "ripple field swings {sw_max} deg over dt={dt} — no longer the \
                     slow-wide-waves regime the round-4 fade was derived in"
                );
            }
            // ACCURACY: the unfaded reconstruction stays near-exact at every
            // dt (measured <= 0.06 deg at 250 ms; 0.5 deg is ~10x headroom).
            // If this fails, the fade is no longer "magnitude, not math".
            if er_max > 0.5 {
                panic!(
                    "ripple reconstruction err {er_max} deg at dt={dt} — the first-order \
                     model broke, re-derive the round-4 story"
                );
            }
        }
    }
}
