//! CPU-side DLSS Ray Reconstruction plumbing: per-pixel G-buffers captured at
//! primary-hit time, the frame-uniform Halton jitter sequence, and the camera
//! matrices/constants the denoiser consumes. Everything here is pure CPU and
//! feeds the GPU seam (`gpu::GpuContext::present_rr`); nothing touches the
//! tracer's tmin/cut machinery.

use crate::camera::{CamBasis, Camera};
use glam::{Mat4, Vec3, Vec3A, Vec4};
use half::f16;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering::Relaxed};

/// Load an f16-bit-pattern atomic as f32. The G-buffer planes (except depth)
/// store f16 — every consumer widens through here so the conversion has one
/// home; never hand-roll `f16::from_bits` at a call site.
#[inline(always)]
pub fn ld16(a: &AtomicU16) -> f32 {
    f16::from_bits(a.load(Relaxed)).to_f32()
}

/// Store an f32 into an f16-bit-pattern atomic (round-to-nearest-even). The
/// single narrowing point for CPU-rendered G-buffer writes.
#[inline(always)]
pub fn st16(a: &AtomicU16, v: f32) {
    a.store(f16::from_f32(v).to_bits(), Relaxed)
}

/// One pixel's worth of G-buffer data, computed once and stored via
/// `GBufs::write`.
pub struct GPixel {
    /// World-space shading normal (the exact normal used for shading).
    pub normal: Vec3A,
    /// Linear roughness (the material's perceptual roughness; 1 = diffuse).
    pub rough: f32,
    /// Diffuse albedo = mat.albedo * (1 - metallic), linear.
    pub diff_alb: Vec3A,
    /// Specular albedo: full RGB F0 = lerp(0.04, albedo, metallic), linear.
    pub spec_alb: Vec3A,
    /// Linear view-space Z (t * dot(dir, forward)), NOT Euclidean t.
    /// Sky pixels store `far` (finite — survives f16 textures).
    pub view_z: f32,
    /// Screen-space motion vector in pixels, y-down, current -> previous.
    pub mv: (f32, f32),
    /// Specular (mirror reflection) hit distance in world units; `far` when
    /// the reflection ray missed, 0 when no reflection was traced.
    pub spec_hit_t: f32,
    /// The material's ripple amplitude — nonzero only on water. Read by the
    /// frame-generation guide pass, which needs to know the mirror normal is
    /// MOVING; every other consumer ignores it. See `shade::PrimarySurface`.
    pub ripple_amp: f32,
}

/// Per-pixel G-buffers as atomic bit patterns — writes are tile-disjoint, so
/// relaxed stores are race-free for exactly the same reason as the
/// accumulation buffer (render.rs:30). Every plane except `depth` stores f16
/// bits (`ld16`/`st16`): the GPU wire is already f16/unorm8 for those planes
/// (rr.rs/xr.rs formats) and the check gates compare at ≤ 1 f16 ulp, so the
/// narrowing just moves from upload time into storage. `depth` stays f32 —
/// its wire format is R32_FLOAT on both upscalers, reprojection and the XeSS
/// sky-encodes-exactly-0.0 contract read it, and the RR feed gate is
/// bit-equal f32.
pub struct GBufs {
    pub rw: usize,
    pub rh: usize,
    /// True when only `mvec` + `depth` are ever consumed (the FSR 3.1
    /// upscale-only session): the four guide/denoiser planes are zero-length
    /// and `write` skips their encodes — the same flavor gate that skips
    /// `FsrBufs`, saving ~22 B/px of stores + f16 ALU per frame and ~3/4 of
    /// the allocation.
    slim: bool,
    /// 4/px: world normal xyz + roughness w (packed to match the
    /// kBufferTypeNormalRoughness texture layout).
    pub normal_rough: Vec<AtomicU16>,
    /// 3/px linear diffuse albedo.
    pub diff_alb: Vec<AtomicU16>,
    /// 3/px RGB specular albedo (F0).
    pub spec_alb: Vec<AtomicU16>,
    /// 1/px linear view-space Z (f32 bits — see the struct doc).
    pub depth: Vec<AtomicU32>,
    /// 2/px motion vector (pixels, y-down, current -> previous).
    pub mvec: Vec<AtomicU16>,
    /// 1/px specular hit distance.
    pub spec_hit_t: Vec<AtomicU16>,
    /// 1/px ripple amplitude (water tag for the FG guide pass). A guide
    /// plane, so the slim variant leaves it zero-length like the rest.
    pub ripple_amp: Vec<AtomicU16>,
}

impl GBufs {
    pub fn new(rw: usize, rh: usize) -> Self {
        Self::sized(rw, rh, false)
    }

    /// The mvec+depth-only variant (see the `slim` field doc).
    pub fn new_slim(rw: usize, rh: usize) -> Self {
        Self::sized(rw, rh, true)
    }

    fn sized(rw: usize, rh: usize, slim: bool) -> Self {
        let alloc = |n: usize| (0..n).map(|_| AtomicU16::new(0)).collect();
        let guide = |n: usize| alloc(if slim { 0 } else { n });
        Self {
            rw,
            rh,
            slim,
            normal_rough: guide(rw * rh * 4),
            diff_alb: guide(rw * rh * 3),
            spec_alb: guide(rw * rh * 3),
            depth: (0..rw * rh).map(|_| AtomicU32::new(0)).collect(),
            mvec: alloc(rw * rh * 2),
            spec_hit_t: guide(rw * rh),
            ripple_amp: guide(rw * rh),
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
        debug_assert!(!self.slim && !src.slim, "guide planes absent on a slim GBufs");
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
                for k in 0..3 {
                    self.spec_alb[di * 3 + k]
                        .store(src.spec_alb[si * 3 + k].load(Relaxed), Relaxed);
                }
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
        // mvec is allocated in every variant (slim keeps only mvec + depth).
        assert!(rw * rh * 2 <= self.mvec.len(), "GBufs::set_res beyond capacity");
        self.rw = rw;
        self.rh = rh;
    }

    #[inline(always)]
    pub fn write(&self, x: usize, y: usize, p: &GPixel) {
        let i = y * self.rw + x;
        if !self.slim {
            st16(&self.normal_rough[i * 4], p.normal.x);
            st16(&self.normal_rough[i * 4 + 1], p.normal.y);
            st16(&self.normal_rough[i * 4 + 2], p.normal.z);
            st16(&self.normal_rough[i * 4 + 3], p.rough);
            st16(&self.diff_alb[i * 3], p.diff_alb.x);
            st16(&self.diff_alb[i * 3 + 1], p.diff_alb.y);
            st16(&self.diff_alb[i * 3 + 2], p.diff_alb.z);
            st16(&self.spec_alb[i * 3], p.spec_alb.x);
            st16(&self.spec_alb[i * 3 + 1], p.spec_alb.y);
            st16(&self.spec_alb[i * 3 + 2], p.spec_alb.z);
            st16(&self.spec_hit_t[i], p.spec_hit_t);
            st16(&self.ripple_amp[i], p.ripple_amp);
        }
        self.depth[i].store(p.view_z.to_bits(), Relaxed);
        st16(&self.mvec[i * 2], p.mv.0);
        st16(&self.mvec[i * 2 + 1], p.mv.1);
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

/// Streamline guidance: phase length ≈ 8·(upscale ratio)². Both DRS paths
/// (DLSS-RR and XeSS) can shed to a 3× ratio at their range floor, which
/// needs 8·3² = 72; a longer phase is benign at shallow ratios (finer
/// coverage, slower cycle), a too-short one causes convergence artifacts
/// when the controller parks deep.
pub const JITTER_PHASE: u32 = 72;

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

/// Hard cap on `--spp`. Not a math limit — it is the size of the GPU constant
/// buffer's jitter table (MAX_SPP × 8 B inside `FrameCb`, which must fit
/// `CB_STRIDE`). Raising it is those two constants in lockstep; 128 already
/// puts a 1080p frame in the seconds, and the returns are 1/√N (see the
/// `--check` spp bench: the quadtree's fixed cost is fully amortized by
/// ~8-16 spp, past which every sample pays full marginal price).
pub const MAX_SPP: u32 = 128;

/// Halton index stride between the samples of one multi-sampled frame.
/// Coprime with JITTER_PHASE (72 = 8·9, so any odd non-multiple of 3 works),
/// which is what keeps the extra samples spread across the whole phase
/// instead of clustering on a coset of it.
const SPP_STRIDE: u32 = 25;

/// Sub-pixel offset for sample `k` of frame `frame_idx`, in [-0.5, 0.5).
///
/// `k == 0` IS the frame's reported jitter — `jitter_for` verbatim, so the
/// sequence the upscalers see (and its 72-phase coverage, which the DRS ratio
/// floors depend on) is untouched by multi-sampling, and spp == 1 stays
/// bit-identical to a single-sample frame.
///
/// `k > 0` walks the SAME Halton sequence at a phase-coprime stride, but its
/// index runs FREE — deliberately NOT reduced mod JITTER_PHASE. The phase
/// bounds the sequence the *upscaler* sees; within one frame the samples must
/// be distinct positions, and Halton is infinite, so `n` just keeps climbing
/// (a wrap would make sample 72 land exactly on sample 0 at spp > 72 —
/// supersampling the same point twice). Deterministic, low-discrepancy, inside
/// the pixel footprint (so the tile-frustum/cut inheritance still covers it),
/// and a pure function of (frame, k) — which is what lets `render::verify`
/// reconstruct every extra sample's ray and gate it.
pub fn jitter_for_sample(frame_idx: u32, k: u32) -> (f32, f32) {
    if k == 0 {
        return jitter_for(frame_idx);
    }
    let n = (frame_idx % JITTER_PHASE) + 1 + k * SPP_STRIDE;
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
/// contracts. Carries the previous `Camera` so a DRS step can rebuild the
/// basis/matrices at the NEW resolution — the pose is unchanged, only the
/// pixel mapping, which is what keeps MVs in current-res pixels across a
/// step (dropping prev instead would false-flag whole-frame motion or force
/// a history reset).
pub struct DlssPrev {
    pub basis: CamBasis,
    pub mats: CamMatrices,
    pub cam: Camera,
    /// The sway clock of the SAME frame (main.rs's `PrevPose` pairing rule —
    /// the sway-MV deltas must describe the exact pose pair the retained
    /// camera does). None = no sway partition in the scene.
    pub sway_t: Option<f32>,
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

/// Debug: dump the G-buffers as PNGs (albedo/spec-albedo/normal/roughness/
/// depth/MV/spec-hit) so the capture can be eyeballed before the GPU side
/// even runs.
pub fn dump_gbufs(g: &GBufs, prefix: &str, far: f32) {
    let (rw, rh) = (g.rw, g.rh);
    let load = ld16;
    let to8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;

    let mut alb = Vec::with_capacity(rw * rh * 3);
    let mut spec = Vec::with_capacity(rw * rh * 3);
    let mut nrm = Vec::with_capacity(rw * rh * 3);
    let mut misc = Vec::with_capacity(rw * rh * 3); // r=rough, g=depth/far, b=spec_hit/far
    let mut mv = Vec::with_capacity(rw * rh * 3);
    for i in 0..rw * rh {
        for k in 0..3 {
            // linear -> rough sRGB for visibility
            alb.push(to8(load(&g.diff_alb[i * 3 + k]).powf(1.0 / 2.2)));
            spec.push(to8(load(&g.spec_alb[i * 3 + k]).powf(1.0 / 2.2)));
            nrm.push(to8(load(&g.normal_rough[i * 4 + k]) * 0.5 + 0.5));
        }
        misc.push(to8(load(&g.normal_rough[i * 4 + 3])));
        misc.push(to8(f32::from_bits(g.depth[i].load(Relaxed)) / far));
        misc.push(to8(load(&g.spec_hit_t[i]) / far));
        mv.push(to8(load(&g.mvec[i * 2]) / 32.0 + 0.5));
        mv.push(to8(load(&g.mvec[i * 2 + 1]) / 32.0 + 0.5));
        mv.push(128);
    }
    for (name, data) in
        [("albedo", &alb), ("spec_albedo", &spec), ("normal", &nrm), ("misc", &misc), ("mv", &mv)]
    {
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
    // Depth is f32 bits; the MV plane is f16 (its quantization error is
    // ≤ ~0.001 px for typical few-pixel MVs — far inside the geometric gates).
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
            let (mx, my) = (ld16(&gb.mvec[i * 2]), ld16(&gb.mvec[i * 2 + 1]));
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

    // SKY ARM. The loop above `continue`s on sky (there is no world point to
    // reconstruct), so nothing here ever validated a sky pixel — which is how
    // `sparse_fill` shipped writing each coarse sky pixel's MV and normal from
    // a DIFFERENT pixel's direction (the cell's random sample, up to
    // SAMPLE_CELL away and re-randomized every frame).
    //
    // What is checkable exactly: `write_gbuf_sky` stores `normal = -dir`, and
    // `dir` must be the preimage of the (fx, fy) the same call subtracts to
    // form the MV. So reconstruct the pixel's own direction and compare. The
    // bound is set by the f16 normal plane (~5e-4 per component), and the
    // defect it targets is ~2e-2 rad — nearly two orders clear.
    //
    // NOTE what this deliberately does NOT claim to catch: sub-pixel
    // REGISTRATION. A half-pixel error is ~1e-3 rad here, at the plane's own
    // noise floor, and under pure translation the sky MV is exactly zero
    // whether or not the sample position is declared honestly. That defect is
    // observable only in color — see the `sky registration` gate in main.rs.
    //
    // must_fire = false: a skyless pose passes VACUOUSLY here (with the note),
    // per the loaded-scene/--stress structural-skip convention — mv_selftest
    // has no structural flag of its own, and its check-gpu/-dxr callers
    // already carry a structural-gated `skies == 0` must-fire beside this
    // call, so sky-presence teeth are not lost where they are mandated.
    let sky_ok = sky_dir_check("mv_selftest sky", gb, basis_b, far, false);
    ok && ident_ok && sky_ok
}

/// The sky half of the G-buffer contract, factored out so both `mv_selftest`
/// (full-depth frames) and the capped-frame gate in main.rs (which is what
/// reaches `sparse_fill`'s coarse flood) run the identical check.
///
/// `write_gbuf_sky` stores `normal = -dir` and forms the MV by subtracting the
/// (fx, fy) that `dir` came from, so the two must stay a matched pair: `dir`
/// is the preimage of the position. Reconstructing the pixel's own direction
/// and comparing catches any path that pairs them wrongly — which is exactly
/// what `sparse_fill` did, reusing a cell sample's direction from a pixel up
/// to SAMPLE_CELL away against this pixel's own position.
///
/// Limit 5e-3 rad against an f16 plane whose own quantization is ~3.5e-4
/// (measured) — 14x of headroom, and the defect class it targets is ~2e-2.
///
/// `must_fire` is the anti-vacuity policy: a pose with zero sky pixels proves
/// nothing, and whether that FAILS (structural runs — the default scenes have
/// sky by construction) or passes with a note (loaded scenes / --stress, the
/// suite-wide structural-skip convention) is the CALLER's decision, exactly
/// like every other must-fire in the check suites. The mismatch half
/// (`bad == 0`) gates unconditionally either way.
pub fn sky_dir_check(label: &str, gb: &GBufs, basis: &CamBasis, far: f32, must_fire: bool) -> bool {
    const LIMIT: f32 = 5e-3;
    let (rw, rh) = (gb.rw, gb.rh);
    let (mut px, mut bad, mut worst) = (0u64, 0u64, 0.0f32);
    for y in 0..rh {
        for x in 0..rw {
            let i = y * rw + x;
            if f32::from_bits(gb.depth[i].load(Relaxed)) < far * 0.99 {
                continue;
            }
            px += 1;
            let want = basis.ray_dir(x as f32 + 0.5, y as f32 + 0.5);
            let got = Vec3A::new(
                -ld16(&gb.normal_rough[i * 4]),
                -ld16(&gb.normal_rough[i * 4 + 1]),
                -ld16(&gb.normal_rough[i * 4 + 2]),
            );
            // Both are unit, so the chord length is the angle to first order.
            let err = (want - got).length();
            if err > LIMIT {
                bad += 1;
            }
            worst = worst.max(err);
        }
    }
    let ok = bad == 0 && (px > 0 || !must_fire);
    eprintln!(
        "{label}: {px} px | dir mismatch {bad} | worst {worst:.3e} (limit {LIMIT:.1e}) -> {}",
        if !ok {
            "FAIL"
        } else if px == 0 {
            "OK (vacuous — no sky px, structural skip)"
        } else {
            "OK"
        }
    );
    ok
}
