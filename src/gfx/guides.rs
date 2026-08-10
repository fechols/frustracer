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
