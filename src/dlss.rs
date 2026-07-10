//! CPU-side DLSS Ray Reconstruction plumbing: per-pixel G-buffers captured at
//! primary-hit time, the frame-uniform Halton jitter sequence, and the camera
//! matrices/constants the denoiser consumes. Everything here is pure CPU and
//! feeds the GPU seam (`gpu::GpuContext::present_rr`); nothing touches the
//! tracer's tmin/cut machinery.

use crate::camera::{CamBasis, Camera};
use glam::{Mat4, Vec3, Vec3A, Vec4};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// One pixel's worth of G-buffer data, computed once and stored via
/// `GBufs::write`.
pub struct GPixel {
    /// World-space shading normal (the exact normal used for shading).
    pub normal: Vec3A,
    /// Linear roughness (the material's perceptual roughness; 1 = diffuse).
    pub rough: f32,
    /// Diffuse albedo = mat.albedo * (1 - metallic), linear.
    pub diff_alb: Vec3A,
    /// Specular albedo (achromatic: mean of F0 = lerp(0.04, albedo, metallic)).
    pub spec_alb: f32,
    /// Linear view-space Z (t * dot(dir, forward)), NOT Euclidean t.
    /// Sky pixels store `far` (finite — survives f16 textures).
    pub view_z: f32,
    /// Screen-space motion vector in pixels, y-down, current -> previous.
    pub mv: (f32, f32),
    /// Specular (mirror reflection) hit distance in world units; `far` when
    /// the reflection ray missed, 0 when no reflection was traced.
    pub spec_hit_t: f32,
}

/// Per-pixel G-buffers in the same `Vec<AtomicU32>` f32-bit-pattern layout as
/// the accumulation buffer — writes are tile-disjoint, so relaxed stores are
/// race-free for exactly the same reason (render.rs:30).
pub struct GBufs {
    pub rw: usize,
    pub rh: usize,
    /// 4/px: world normal xyz + roughness w (packed to match the
    /// kBufferTypeNormalRoughness texture layout).
    pub normal_rough: Vec<AtomicU32>,
    /// 3/px linear diffuse albedo.
    pub diff_alb: Vec<AtomicU32>,
    /// 1/px achromatic specular albedo.
    pub spec_alb: Vec<AtomicU32>,
    /// 1/px linear view-space Z.
    pub depth: Vec<AtomicU32>,
    /// 2/px motion vector (pixels, y-down, current -> previous).
    pub mvec: Vec<AtomicU32>,
    /// 1/px specular hit distance.
    pub spec_hit_t: Vec<AtomicU32>,
}

impl GBufs {
    pub fn new(rw: usize, rh: usize) -> Self {
        let alloc = |n: usize| (0..n).map(|_| AtomicU32::new(0)).collect();
        Self {
            rw,
            rh,
            normal_rough: alloc(rw * rh * 4),
            diff_alb: alloc(rw * rh * 3),
            spec_alb: alloc(rw * rh),
            depth: alloc(rw * rh),
            mvec: alloc(rw * rh * 2),
            spec_hit_t: alloc(rw * rh),
        }
    }

    /// Nearest-upscale the OIDN guide planes — diffuse albedo, specular
    /// albedo, normal+roughness: exactly the fields `oidn::run_filter`
    /// reads — from `src` into this buffer's resolution. The post-upscale
    /// denoise path runs OIDN at window res while the frame's G-buffers
    /// live at render res; the guides follow the same nearest mapping
    /// `render::resolve` uses for color, so guide texels stay bit-equal to
    /// their source texels (gated by --check-xess).
    pub fn upscale_guides_from(&self, src: &GBufs) {
        let (dw, dh) = (self.rw, self.rh);
        let (sw, sh) = (src.rw, src.rh);
        (0..dh).into_par_iter().for_each(|y| {
            let sy = (y * sh / dh).min(sh - 1);
            for x in 0..dw {
                let sx = (x * sw / dw).min(sw - 1);
                let si = sy * sw + sx;
                let di = y * dw + x;
                for k in 0..3 {
                    self.diff_alb[di * 3 + k]
                        .store(src.diff_alb[si * 3 + k].load(Relaxed), Relaxed);
                }
                self.spec_alb[di].store(src.spec_alb[si].load(Relaxed), Relaxed);
                for k in 0..4 {
                    self.normal_rough[di * 4 + k]
                        .store(src.normal_rough[si * 4 + k].load(Relaxed), Relaxed);
                }
            }
        });
    }

    /// Reinterpret the buffers at a different logical resolution within the
    /// construction capacity — XeSS mode's dynamic render res. Contents are
    /// stale until the next frame writes every pixel, which every XeSS frame
    /// does (full-depth trace: hit and sky fill sites both write).
    pub fn set_res(&mut self, rw: usize, rh: usize) {
        assert!(rw * rh * 4 <= self.normal_rough.len(), "GBufs::set_res beyond capacity");
        self.rw = rw;
        self.rh = rh;
    }

    #[inline(always)]
    pub fn write(&self, x: usize, y: usize, p: &GPixel) {
        let i = y * self.rw + x;
        let s = |a: &AtomicU32, v: f32| a.store(v.to_bits(), Relaxed);
        s(&self.normal_rough[i * 4], p.normal.x);
        s(&self.normal_rough[i * 4 + 1], p.normal.y);
        s(&self.normal_rough[i * 4 + 2], p.normal.z);
        s(&self.normal_rough[i * 4 + 3], p.rough);
        s(&self.diff_alb[i * 3], p.diff_alb.x);
        s(&self.diff_alb[i * 3 + 1], p.diff_alb.y);
        s(&self.diff_alb[i * 3 + 2], p.diff_alb.z);
        s(&self.spec_alb[i], p.spec_alb);
        s(&self.depth[i], p.view_z);
        s(&self.mvec[i * 2], p.mv.0);
        s(&self.mvec[i * 2 + 1], p.mv.1);
        s(&self.spec_hit_t[i], p.spec_hit_t);
    }
}

/// Radical-inverse (van der Corput) in the given base.
pub fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let ib = 1.0 / base as f32;
    while index > 0 {
        f *= ib;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Streamline guidance: phase length ≈ 8·(upscale ratio)²; Quality mode
/// needs 8·1.5² = 18, so 32 covers DLAA through Quality with headroom.
pub const JITTER_PHASE: u32 = 32;

/// Frame-uniform sub-pixel jitter offset in [-0.5, 0.5): the SAME offset is
/// applied to every pixel of the frame (index 0 of the Halton sequence is
/// skipped — it is (0, 0), which wastes a phase slot on the center sample).
/// Offsets stay inside the pixel footprint, so sample positions remain in
/// [x, x+1) — exactly the jitter invariant the quadtree and temporal cache
/// already require (CLAUDE.md: jitter in [0,1) on pixel coords).
pub fn jitter_for(frame_idx: u32) -> (f32, f32) {
    let n = (frame_idx % JITTER_PHASE) + 1;
    (halton(n, 2) - 0.5, halton(n, 3) - 0.5)
}

/// Reporting values for the denoiser's projection matrices — the ray tracer
/// has no clip planes. Single source for Constants and the sky-depth
/// sentinel so they can never disagree. Note Scene::diag includes the ground
/// plane (~170 for the procedural scene), not just the model.
pub fn near_far(diag: f32) -> (f32, f32) {
    (1e-3 * diag, 2.0 * diag)
}

/// Deterministic stand-in for SL's Quality-mode optimal render resolution in
/// headless runs (no interposer to query): exact 2/3, floored. The
/// interactive path never uses this — it takes the size
/// slDLSSDGetOptimalSettings returns.
pub fn headless_render_res(w: usize, h: usize) -> (usize, usize) {
    (w * 2 / 3, h * 2 / 3)
}

/// View + projection matrices built from the *exact* construction
/// `Camera::basis` uses (camera.rs:33-35), so the matrix camera is the ray
/// camera: f = forward, r = f × Y normalized, u = r × f. Camera space is
/// (x=r, y=u, z=f), z positive forward — left-handed D3D-style. glam is
/// column-major; the transpose to Streamline's row-major float4x4 happens at
/// the shim boundary (gpu/mod.rs), nowhere else.
#[derive(Clone, Copy)]
pub struct CamMatrices {
    pub world_to_view: Mat4,
    pub view_to_clip: Mat4,
}

pub fn cam_matrices(cam: &Camera, rw: usize, rh: usize, near: f32, far: f32) -> CamMatrices {
    let f = cam.forward();
    let r = f.cross(Vec3A::Y).normalize();
    let u = r.cross(f);
    let pos = cam.pos;
    // Column-major: columns are the images of the basis vectors; row i of the
    // rotation part is the camera axis i.
    let world_to_view = Mat4::from_cols(
        Vec4::new(r.x, u.x, f.x, 0.0),
        Vec4::new(r.y, u.y, f.y, 0.0),
        Vec4::new(r.z, u.z, f.z, 0.0),
        Vec4::new(-r.dot(pos), -u.dot(pos), -f.dot(pos), 1.0),
    );
    let aspect = rw as f32 / rh as f32;
    let view_to_clip = Mat4::perspective_lh(cam.fov_y, aspect, near, far);
    CamMatrices { world_to_view, view_to_clip }
}

/// Previous-frame state for motion vectors + reprojection matrices. Kept
/// SEPARATE from the temporal cache's `tprev_basis` on purpose: that one has
/// its own contract ("the exact basis of the last cache-producing frame") and
/// is deliberately dropped on non-participating frames. Two variables, two
/// contracts.
pub struct DlssPrev {
    pub basis: CamBasis,
    pub mats: CamMatrices,
}

/// Everything the denoiser needs per frame besides the buffers themselves.
pub struct FrameConstants {
    pub view_to_clip: Mat4,
    pub clip_to_view: Mat4,
    pub clip_to_prev_clip: Mat4,
    pub prev_clip_to_clip: Mat4,
    pub world_to_view: Mat4,
    pub view_to_world: Mat4,
    /// The sample-position jitter offset the renderer actually used, pixels.
    pub jitter: (f32, f32),
    /// No usable history: first frame, mode/quality change, teleport.
    pub reset: bool,
    pub pos: Vec3A,
    pub right: Vec3A,
    pub up: Vec3A,
    pub forward: Vec3A,
    pub near: f32,
    pub far: f32,
    pub fov_y: f32,
    pub aspect: f32,
    pub rw: usize,
    pub rh: usize,
}

pub fn frame_constants(
    cam: &Camera,
    mats: &CamMatrices,
    prev: Option<&CamMatrices>,
    jitter: (f32, f32),
    reset: bool,
    near: f32,
    far: f32,
    rw: usize,
    rh: usize,
) -> FrameConstants {
    let world_to_view = mats.world_to_view;
    let view_to_clip = mats.view_to_clip;
    let clip_to_view = view_to_clip.inverse();
    let view_to_world = world_to_view.inverse();
    // Column-vector composition, applied right to left:
    // current clip -> view -> world -> prev view -> prev clip.
    let clip_to_prev_clip = match prev {
        Some(p) => p.view_to_clip * p.world_to_view * view_to_world * clip_to_view,
        None => Mat4::IDENTITY,
    };
    let f = cam.forward();
    let r = f.cross(Vec3A::Y).normalize();
    let u = r.cross(f);
    FrameConstants {
        view_to_clip,
        clip_to_view,
        clip_to_prev_clip,
        prev_clip_to_clip: clip_to_prev_clip.inverse(),
        world_to_view,
        view_to_world,
        jitter,
        reset,
        pos: cam.pos,
        right: r,
        up: u,
        forward: f,
        near,
        far,
        fov_y: cam.fov_y,
        aspect: rw as f32 / rh as f32,
        rw,
        rh,
    }
}

/// Debug: dump the G-buffers as PNGs (albedo/normal/roughness/depth/MV/spec)
/// so the capture can be eyeballed before the GPU side even runs.
pub fn dump_gbufs(g: &GBufs, prefix: &str, far: f32) {
    let (rw, rh) = (g.rw, g.rh);
    let load = |a: &AtomicU32| f32::from_bits(a.load(Relaxed));
    let to8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;

    let mut alb = Vec::with_capacity(rw * rh * 3);
    let mut nrm = Vec::with_capacity(rw * rh * 3);
    let mut misc = Vec::with_capacity(rw * rh * 3); // r=rough, g=depth/far, b=spec_hit/far
    let mut mv = Vec::with_capacity(rw * rh * 3);
    for i in 0..rw * rh {
        for k in 0..3 {
            // linear -> rough sRGB for visibility
            alb.push(to8(load(&g.diff_alb[i * 3 + k]).powf(1.0 / 2.2)));
            nrm.push(to8(load(&g.normal_rough[i * 4 + k]) * 0.5 + 0.5));
        }
        misc.push(to8(load(&g.normal_rough[i * 4 + 3])));
        misc.push(to8(load(&g.depth[i]) / far));
        misc.push(to8(load(&g.spec_hit_t[i]) / far));
        mv.push(to8(load(&g.mvec[i * 2]) / 32.0 + 0.5));
        mv.push(to8(load(&g.mvec[i * 2 + 1]) / 32.0 + 0.5));
        mv.push(128);
    }
    for (name, data) in [("albedo", &alb), ("normal", &nrm), ("misc", &misc), ("mv", &mv)] {
        let path = format!("{prefix}_{name}.png");
        if let Err(e) =
            image::save_buffer(&path, data, rw as u32, rh as u32, image::ColorType::Rgb8)
        {
            eprintln!("failed to save {path}: {e}");
        } else {
            eprintln!("wrote {path}");
        }
    }
}

/// Deterministic MV/depth/matrix self-test: given two frames' G-buffers and
/// camera bases (B = A + a small dolly), reconstruct each B pixel's world
/// position from its view-Z, follow its motion vector into frame A, and
/// reconstruct there too — the two world points must agree. Tests MV sign,
/// depth, and both bases jointly; immune to shading noise. Also spot-checks
/// the ray/matrix identity (clip x/w vs pixel coordinate).
///
/// Edges and disocclusions legitimately fail, so the gate is on the median
/// and a 90th-percentile bound, never the max.
pub fn mv_selftest(
    ga: &GBufs,
    basis_a: &CamBasis,
    gb: &GBufs,
    basis_b: &CamBasis,
    mats_b: &CamMatrices,
    diag: f32,
    far: f32,
) -> bool {
    let (rw, rh) = (gb.rw, gb.rh);
    let load = |buf: &[AtomicU32], i: usize| f32::from_bits(buf[i].load(Relaxed));
    // Reconstruct the world point of a pixel from its stored view-Z: the ray
    // through the (center) sample position scaled so its view-Z matches.
    let reconstruct = |basis: &CamBasis, fx: f32, fy: f32, view_z: f32| -> Vec3A {
        let dir = basis.ray_dir(fx, fy);
        let t = view_z / dir.dot(basis.forward());
        basis.origin + dir * t
    };

    let mut errs: Vec<f32> = Vec::with_capacity(rw * rh / 4);
    for y in 0..rh {
        for x in 0..rw {
            let i = y * rw + x;
            let zb = load(&gb.depth, i);
            if !(zb > 0.0) || zb >= far * 0.99 {
                continue; // sky
            }
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let pb = reconstruct(basis_b, fx, fy, zb);
            let (mx, my) = (load(&gb.mvec, i * 2), load(&gb.mvec, i * 2 + 1));
            let (px, py) = (fx + mx, fy + my);
            let (ax, ay) = (px as usize, py as usize);
            if px < 0.5 || py < 0.5 || ax + 1 >= rw || ay + 1 >= rh {
                continue; // reprojects off the old screen
            }
            let za = load(&ga.depth, ay * rw + ax);
            if za >= far * 0.99 {
                continue; // old pixel was sky (disocclusion)
            }
            let pa = reconstruct(basis_a, px, py, za);
            errs.push((pa - pb).length());
        }
    }
    if errs.is_empty() {
        eprintln!("mv_selftest: no comparable pixels — FAIL");
        return false;
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = errs[errs.len() / 2];
    let p90 = errs[errs.len() * 9 / 10];
    let ok = median < 1e-3 * diag && p90 < 1e-2 * diag;
    eprintln!(
        "mv_selftest: {} px | median err {:.3e} (limit {:.3e}) | p90 {:.3e} (limit {:.3e}) -> {}",
        errs.len(),
        median,
        1e-3 * diag,
        p90,
        1e-2 * diag,
        if ok { "OK" } else { "FAIL" }
    );

    // Ray/matrix identity: for a few pixels, projecting the reconstructed
    // world point through world_to_view * view_to_clip must land back on the
    // pixel: clip.x/w == fx*2/rw - 1, clip.y/w == 1 - fy*2/rh.
    let mut ident_ok = true;
    for (x, y) in [(rw / 4, rh / 4), (rw / 2, rh / 2), (3 * rw / 4, 2 * rh / 3)] {
        let i = y * rw + x;
        let zb = load(&gb.depth, i);
        if zb >= far * 0.99 {
            continue;
        }
        let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
        let p = reconstruct(basis_b, fx, fy, zb);
        let clip = mats_b.view_to_clip * mats_b.world_to_view * Vec4::from((Vec3::from(p), 1.0));
        let ndx = clip.x / clip.w;
        let ndy = clip.y / clip.w;
        let want_x = fx * 2.0 / rw as f32 - 1.0;
        let want_y = 1.0 - fy * 2.0 / rh as f32;
        if (ndx - want_x).abs() > 1e-3 || (ndy - want_y).abs() > 1e-3 {
            eprintln!(
                "mv_selftest: matrix/ray identity FAIL at ({x},{y}): clip ({ndx:.5},{ndy:.5}) vs ray ({want_x:.5},{want_y:.5})"
            );
            ident_ok = false;
        }
    }
    if ident_ok {
        eprintln!("mv_selftest: matrix/ray identity OK");
    }
    ok && ident_ok
}
