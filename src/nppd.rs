//! NPPD (Neural Partitioning Pyramids for Denoising Monte Carlo Renderings,
//! Bálint et al., SIGGRAPH 2023) — the neural denoiser path next to OIDN and
//! DLSS-RR. Same footprint policy as Streamline/OIDN/XeSS: nothing links any
//! SDK; `onnxruntime.dll` (+ `DirectML.dll`) is loaded at runtime from the
//! SDK bin directory (default `SDKs\onnxruntime\bin`), every entry point goes
//! through a fn-pointer table, and headless builds / DLL-free machines are
//! unaffected. Unlike OIDN there is not even a per-export `GetProcAddress`
//! table: ONE export (`OrtGetApiBase`) bootstraps the whole versioned
//! `OrtApi` struct of function pointers.
//!
//! The model is a one-time offline export (`tools/nppd-export/export.py`):
//! the pretrained Lightning checkpoint restructured into an inference-only
//! ONNX graph with the recurrent state as explicit I/O —
//! `frame_input [1,S,10,H,W]` (depth ‖ normal ‖ diffuse-albedo ‖ color) plus
//! `temporal_warped [1,C_t,H,W]` in, `output [1,3,H,W]` plus
//! `temporal_out [1,C_t,H,W]` out. The network's own radiance normalization
//! (`normalize_radiance`/`clip_logp1`) lives INSIDE the graph; the packer
//! reproduces the two Noisebase dataloader transforms the checkpoints were
//! trained on (settled from `noisebase/projective.py`, not empirical):
//! depth is `ln(1 + 1/d)` of the EUCLIDEAN camera distance (view-Z converted
//! back through the per-pixel ray direction; sky = 0), and normals are
//! CAMERA-space `(n·forward, n·right, n·up)` — the same right-handed
//! `f`/`f×Y`/`r×f` basis `dlss::cam_matrices` builds, which is exactly
//! noisebase's `W`/`U = W×up`/`V = U×W`. Albedo and color pass through raw
//! (linear, HDR 1-spp). The motion-vector backward warp of the recurrent
//! state (`F.grid_sample` upstream) is deliberately excised from the graph
//! and done here in Rust (`warp_temporal`) — it sidesteps DirectML
//! GridSample support risk and keeps the warp math DLL-free-testable
//! (`self_test`); `GBufs::mvec`'s "pixels, current → previous" IS noisebase's
//! `motion = prev - current` convention, so the fetch offset is the vector
//! itself.
//!
//! All pure math in this file (padding, NCHW packing, the warp) is gated by
//! `nppd::self_test` (run by `--check`, DLL-free); the live ORT path is gated
//! by `--check-nppd` (needs the DLLs + an exported model).

use crate::camera::CamBasis;
use crate::dlss::GBufs;
use glam::Vec3A;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// Temporal state channels: previous color (3) + output (3) + feature (32) —
/// upstream `Model.temporal_init` (model.py:164).
pub const C_T: usize = 38;

/// Channels in the per-sample frame stack, in pack order (the upstream
/// encoder concat order: depth, normal, diffuse, color):
/// 0 = log-depth `ln(1 + 1/d)` (sky = 0), 1-3 = camera-space normal,
/// 4-6 = diffuse albedo (linear), 7-9 = color (linear HDR, 1 spp).
pub const CH_FRAME: usize = 10;

/// The PartitioningPyramid downsamples by 2 five times (K = 5 → /16 at the
/// coarsest level); pad to /32 so every level divides exactly with margin.
pub const PAD_MULT: usize = 32;

/// Backward-warp fetch: `prev_px = cur_px + MV_SIGN·mv`. NOT empirical —
/// `GBufs::mvec` is defined as "pixels, y-down, current → previous"
/// (dlss.rs::GPixel), i.e. the motion vector IS the fetch offset, the same
/// convention `dlss::mv_selftest` walks (`px = fx + mx`). Both sides of this
/// warp are ours, so the sign is fixed by that contract and gated in
/// `self_test`; flipping it presents as directional smear under motion.
pub const MV_SIGN: (f32, f32) = (1.0, 1.0);

/// Round `w`×`h` up to the padding quantum. The network sees the padded
/// frame; `crop_to_interleaved` cuts the logical interior back out.
pub fn pad_dims(w: usize, h: usize) -> (usize, usize) {
    let up = |v: usize| v.div_ceil(PAD_MULT) * PAD_MULT;
    (up(w), up(h))
}

#[inline(always)]
fn load(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Relaxed))
}

/// Stage the 10-channel frame stack NCHW-planar into `out`
/// (`CH_FRAME · pw · ph` floats, channel-major). Sources: `g` at its own
/// `rw×rh` (the logical resolution) and `accum` as the raw linear-RGB
/// bit-pattern buffer holding exactly one 1-spp frame (the `accumulate =
/// false` store semantics — no sample divide here). `basis` must be the
/// frame's own camera basis: depth is converted view-Z → Euclidean t through
/// the pixel-center ray direction and log-encoded `ln(1 + 1/t)` (sky, i.e.
/// view_z ≥ ~far, encodes to exactly 0 — the d → ∞ limit), and the world
/// normal is rotated into the (n·forward, n·right, n·up) camera frame — the
/// two Noisebase training transforms (module docs). The `pw×ph` padding
/// border is filled by edge replication (clamped source coordinates), not
/// zeros — a zero border is a hard synthetic edge the U-Net would denoise
/// against; a replicated one is conv-friendly and vanishes at the crop.
pub fn pack_inputs(
    accum: &[AtomicU32],
    g: &GBufs,
    basis: &CamBasis,
    far: f32,
    pw: usize,
    ph: usize,
    out: &mut [f32],
) {
    let (w, h) = (g.rw, g.rh);
    assert_eq!(accum.len(), w * h * 3);
    assert_eq!(out.len(), CH_FRAME * pw * ph);
    debug_assert!(pw >= w && ph >= h);
    let fwd = basis.forward();
    // Unit camera right/up — the dlss::cam_matrices construction, which is
    // bit-for-bit noisebase's U = normalize(W × up), V = normalize(U × W)
    // with up = +Y.
    let right = fwd.cross(Vec3A::Y).normalize();
    let up = right.cross(fwd);
    // One rayon task per output plane-row: index = c·ph + y. Each task writes
    // a disjoint `pw` slice sequentially, which keeps stores streaming per
    // plane despite the channel-major layout.
    out.par_chunks_mut(pw).enumerate().for_each(|(row, dst)| {
        let (c, y) = (row / ph, row % ph);
        let sy = y.min(h - 1);
        for (x, v) in dst.iter_mut().enumerate() {
            let sx = x.min(w - 1);
            let i = sy * w + sx;
            *v = match c {
                0 => {
                    let view_z = load(&g.depth[i]);
                    if view_z >= far * 0.99 {
                        0.0
                    } else {
                        // Euclidean camera distance from stored view-Z (the
                        // dlss.rs encode ran t·dot(dir, forward) — undo it at
                        // the pixel center, the mv_selftest reconstruction).
                        let dir = basis.ray_dir(sx as f32 + 0.5, sy as f32 + 0.5);
                        let t = view_z / dir.dot(fwd);
                        (1.0 + 1.0 / t.max(1e-8)).ln()
                    }
                }
                1..=3 => {
                    let n = Vec3A::new(
                        load(&g.normal_rough[i * 4]),
                        load(&g.normal_rough[i * 4 + 1]),
                        load(&g.normal_rough[i * 4 + 2]),
                    );
                    match c {
                        1 => n.dot(fwd),
                        2 => n.dot(right),
                        _ => n.dot(up),
                    }
                }
                4..=6 => load(&g.diff_alb[i * 3 + (c - 4)]),
                _ => f32::from_bits(accum[i * 3 + (c - 7)].load(Relaxed)),
            };
        }
    });
}

/// Backward-warp the recurrent state one frame forward: for every interior
/// pixel, fetch `prev` bilinearly at `cur + MV_SIGN·mv` (texel-index space —
/// an mv of (0,0) is the identity), with **zeros padding** outside the
/// logical `w×h` interior — the exact `F.grid_sample(padding_mode="zeros")`
/// semantics the graph was trained with (an out-of-bounds tap contributes 0
/// with its weight unrenormalized, so edge fetches fade instead of smearing).
/// `prev` and `out` are `c_t` planes of `pw·ph`; only the `w×h` interior of
/// `prev` is trusted data, and the whole of `out` (padding included) is
/// rewritten — zeros in the border, so a stale ping-pong buffer never leaks.
pub fn warp_temporal(
    prev: &[f32],
    mvec: &[AtomicU32],
    c_t: usize,
    w: usize,
    h: usize,
    pw: usize,
    ph: usize,
    out: &mut [f32],
) {
    assert_eq!(mvec.len(), w * h * 2);
    assert_eq!(prev.len(), c_t * pw * ph);
    assert_eq!(out.len(), c_t * pw * ph);
    out.par_chunks_mut(pw).enumerate().for_each(|(row, dst)| {
        let (c, y) = (row / ph, row % ph);
        if y >= h {
            dst.fill(0.0);
            return;
        }
        let plane = &prev[c * pw * ph..(c + 1) * pw * ph];
        for (x, v) in dst.iter_mut().enumerate() {
            if x >= w {
                *v = 0.0;
                continue;
            }
            let i = y * w + x;
            let px = x as f32 + MV_SIGN.0 * load(&mvec[i * 2]);
            let py = y as f32 + MV_SIGN.1 * load(&mvec[i * 2 + 1]);
            let (x0, y0) = (px.floor(), py.floor());
            let (fx, fy) = (px - x0, py - y0);
            let (x0, y0) = (x0 as i64, y0 as i64);
            let mut acc = 0.0f32;
            for (tx, ty, wgt) in [
                (x0, y0, (1.0 - fx) * (1.0 - fy)),
                (x0 + 1, y0, fx * (1.0 - fy)),
                (x0, y0 + 1, (1.0 - fx) * fy),
                (x0 + 1, y0 + 1, fx * fy),
            ] {
                if tx >= 0 && ty >= 0 && (tx as usize) < w && (ty as usize) < h {
                    acc += wgt * plane[ty as usize * pw + tx as usize];
                }
            }
            *v = acc;
        }
    });
}

/// Cut the logical `w×h` interior out of the padded NCHW 3-channel network
/// output and transpose it to the interleaved 3/px layout
/// `render::resolve_hdr` consumes (the OIDN output contract).
pub fn crop_to_interleaved(src: &[f32], pw: usize, ph: usize, w: usize, h: usize, out: &mut [f32]) {
    assert!(src.len() >= 3 * pw * ph);
    assert_eq!(out.len(), w * h * 3);
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, dst)| {
        for (x, px) in dst.chunks_exact_mut(3).enumerate() {
            for (k, v) in px.iter_mut().enumerate() {
                *v = src[k * pw * ph + y * pw + x];
            }
        }
    });
}

/// Closed-form gates over every pure-math piece above — run by `--check`
/// (DLL-free) and as the first stage of `--check-nppd`.
pub fn self_test() -> Result<(), String> {
    let fail = |what: &str| Err(format!("nppd::self_test: {what}"));

    // pad_dims: exact table.
    for (w, h, want) in [
        (1920usize, 1080usize, (1920usize, 1088usize)),
        (800, 600, (800, 608)),
        (33, 33, (64, 64)),
        (32, 32, (32, 32)),
        (1, 1, (32, 32)),
    ] {
        if pad_dims(w, h) != want {
            return fail(&format!("pad_dims({w},{h}) = {:?}, want {want:?}", pad_dims(w, h)));
        }
    }

    // pack_inputs: every channel lands at its NCHW offset, the depth/normal
    // Noisebase transforms hold at analytic anchors, and replicate-padding
    // texels are bit-equal their clamped-coordinate source.
    let (w, h) = (5usize, 5usize);
    let (pw, ph) = pad_dims(w, h);
    let far = 100.0f32;
    // Camera down +X, fov 90°: fwd = (1,0,0), right = (0,0,1), up = (0,1,0).
    let cam = crate::camera::Camera {
        pos: Vec3A::ZERO,
        yaw: 0.0,
        pitch: 0.0,
        fov_y: std::f32::consts::FRAC_PI_2,
    };
    let basis = cam.basis(w, h);
    let (fwd, right, up) = (Vec3A::X, Vec3A::Z, Vec3A::Y);
    let g = GBufs::new(w, h);
    let accum: Vec<AtomicU32> = (0..w * h * 3).map(|_| AtomicU32::new(0)).collect();
    // Distinct, position- and channel-dependent values everywhere.
    let val = |c: usize, x: usize, y: usize| (c * 1000 + y * 100 + x) as f32 + 0.25;
    let view_z = |x: usize, y: usize| 1.0 + (y * w + x) as f32 * 0.125;
    let normal = |x: usize, y: usize| {
        Vec3A::new(0.6, (x as f32) * 0.1 - 0.2, (y as f32) * 0.1 - 0.2).normalize()
    };
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            // Center pixel gets the sky sentinel — its transform anchor is
            // exact 0; every other pixel a real view-Z.
            let z = if (x, y) == (2, 2) { far } else { view_z(x, y) };
            g.depth[i].store(z.to_bits(), Relaxed);
            let n = normal(x, y);
            for (k, nk) in [n.x, n.y, n.z].into_iter().enumerate() {
                g.normal_rough[i * 4 + k].store(nk.to_bits(), Relaxed);
            }
            for k in 0..3 {
                g.diff_alb[i * 3 + k].store(val(4 + k, x, y).to_bits(), Relaxed);
                accum[i * 3 + k].store(val(7 + k, x, y).to_bits(), Relaxed);
            }
            g.normal_rough[i * 4 + 3].store(999.0f32.to_bits(), Relaxed); // roughness: must NOT be packed
        }
    }
    let mut stack = vec![f32::NAN; CH_FRAME * pw * ph];
    pack_inputs(&accum, &g, &basis, far, pw, ph, &mut stack);
    for c in 0..CH_FRAME {
        for y in 0..ph {
            for x in 0..pw {
                let (sx, sy) = (x.min(w - 1), y.min(h - 1));
                let got = stack[c * pw * ph + y * pw + x];
                // The oracle recomputes the two transforms independently of
                // the packer's indexing; the hand-computed anchors below pin
                // the transforms themselves.
                let want = match c {
                    0 => {
                        if (sx, sy) == (2, 2) {
                            0.0
                        } else {
                            let dir = basis.ray_dir(sx as f32 + 0.5, sy as f32 + 0.5);
                            (1.0 + dir.dot(fwd) / view_z(sx, sy)).ln()
                        }
                    }
                    1 => normal(sx, sy).dot(fwd),
                    2 => normal(sx, sy).dot(right),
                    3 => normal(sx, sy).dot(up),
                    _ => val(c, sx, sy),
                };
                if (got - want).abs() > 1e-6 {
                    return fail(&format!("pack ch{c} at ({x},{y}): {got} != {want}"));
                }
            }
        }
    }
    // Anchors: a center-adjacent probe via a second pack with center non-sky.
    // Center pixel (2,2) of the 5×5 grid: ray = forward exactly, so t =
    // view_z and depth = ln(1 + 1/z) in closed form; a +Y world normal must
    // land entirely in the camera-up channel.
    g.depth[2 * w + 2].store(1.0f32.to_bits(), Relaxed);
    for (k, nk) in [0.0f32, 1.0, 0.0].into_iter().enumerate() {
        g.normal_rough[(2 * w + 2) * 4 + k].store(nk.to_bits(), Relaxed);
    }
    pack_inputs(&accum, &g, &basis, far, pw, ph, &mut stack);
    let center = 2 * pw + 2;
    if (stack[center] - 2.0f32.ln()).abs() > 1e-6 {
        return fail(&format!("pack depth anchor: {} != ln 2", stack[center]));
    }
    for (c, want) in [(1usize, 0.0f32), (2, 0.0), (3, 1.0)] {
        let got = stack[c * pw * ph + center];
        if (got - want).abs() > 1e-6 {
            return fail(&format!("pack normal anchor ch{c}: {got} != {want}"));
        }
    }

    // crop_to_interleaved: planar → interleaved roundtrip on an independent
    // synthetic 3-plane buffer (channel-blind — the `val` oracle).
    let mut planar = vec![0.0f32; 3 * pw * ph];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                planar[c * pw * ph + y * pw + x] = val(c, x, y);
            }
        }
    }
    let mut inter = vec![f32::NAN; w * h * 3];
    crop_to_interleaved(&planar, pw, ph, w, h, &mut inter);
    for y in 0..h {
        for x in 0..w {
            for k in 0..3 {
                let got = inter[(y * w + x) * 3 + k];
                let want = val(k, x, y);
                if got.to_bits() != want.to_bits() {
                    return fail(&format!("crop ch{k} at ({x},{y}): {got} != {want}"));
                }
            }
        }
    }

    // Warp gates on a c_t = 3 state with per-channel-distinct contents.
    let c_t = 3usize;
    let plane = |c: usize, x: usize, y: usize| (c * 10000 + y * 100 + x) as f32;
    let mut prev = vec![0.0f32; c_t * pw * ph];
    for c in 0..c_t {
        for y in 0..h {
            for x in 0..w {
                prev[c * pw * ph + y * pw + x] = plane(c, x, y);
            }
        }
    }
    let mv = |dx: f32, dy: f32| -> Vec<AtomicU32> {
        (0..w * h * 2)
            .map(|i| AtomicU32::new(if i % 2 == 0 { dx } else { dy }.to_bits()))
            .collect()
    };
    let mut out = vec![f32::NAN; c_t * pw * ph];

    // Identity: mv = 0 ⇒ interior bit-equal, padding zeroed.
    warp_temporal(&prev, &mv(0.0, 0.0), c_t, w, h, pw, ph, &mut out);
    for c in 0..c_t {
        for y in 0..ph {
            for x in 0..pw {
                let got = out[c * pw * ph + y * pw + x];
                let want = if x < w && y < h { plane(c, x, y) } else { 0.0 };
                if got.to_bits() != want.to_bits() {
                    return fail(&format!("warp identity ch{c} at ({x},{y}): {got} != {want}"));
                }
            }
        }
    }

    // Integer shift, all channels together: mv = (2, 1) fetches (x+2, y+1)
    // exactly; taps past the interior fall to zero (zeros padding). This is
    // also the MV-sign convention gate: `GBufs::mvec` stores current →
    // previous, so a state feature seen at prev pixel (x+2, y+1) must land at
    // current pixel (x, y) — the `dlss::mv_selftest` walk.
    warp_temporal(&prev, &mv(2.0, 1.0), c_t, w, h, pw, ph, &mut out);
    for c in 0..c_t {
        for y in 0..h {
            for x in 0..w {
                let got = out[c * pw * ph + y * pw + x];
                let want =
                    if x + 2 < w && y + 1 < h { plane(c, x + 2, y + 1) } else { 0.0 };
                if got.to_bits() != want.to_bits() {
                    return fail(&format!("warp shift ch{c} at ({x},{y}): {got} != {want}"));
                }
            }
        }
    }

    // Bilinear midpoint on the x-linear ramp: mv = (0.5, 0) at an interior
    // pixel averages the two x-neighbors — analytic value x + 0.5 on the
    // `plane` ramp (unit slope in x).
    warp_temporal(&prev, &mv(0.5, 0.0), c_t, w, h, pw, ph, &mut out);
    for c in 0..c_t {
        let got = out[c * pw * ph + pw + 1]; // (x=1, y=1)
        let want = 0.5 * (plane(c, 1, 1) + plane(c, 2, 1));
        if (got - want).abs() > 1e-3 {
            return fail(&format!("warp midpoint ch{c}: {got} != {want}"));
        }
    }

    // Far out-of-bounds: everything lands on zeros.
    warp_temporal(&prev, &mv(100.0, 100.0), c_t, w, h, pw, ph, &mut out);
    if out.iter().any(|v| *v != 0.0) {
        return fail("warp far-oob: nonzero output");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Runtime loader + ONNX Runtime session (Windows-only, needs the DLLs).
// ---------------------------------------------------------------------------

#[cfg(windows)]
pub use ort::{NppdContext, NppdDevice, NppdGpu};

#[cfg(windows)]
mod ort {
    use super::*;
    use std::ffi::{c_char, c_void, CStr};
    use std::time::Instant;
    use windows::core::{PCSTR, PCWSTR};
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{
        GetProcAddress, LoadLibraryExW, LOAD_WITH_ALTERED_SEARCH_PATH,
    };

    /// The `GetApi` version this binding's `OrtApi` layout was transcribed
    /// for (onnxruntime v1.17.3's onnxruntime_c_api.h). The struct is
    /// append-only versioned, so any onnxruntime.dll >= 1.17 serves this
    /// request; a runtime too old to know version 17 returns null (loud
    /// error at init, never a mis-laid call).
    pub const ORT_API_VERSION: u32 = 17;

    /// The exported graph's tensor names (`tools/nppd-export/export.py`).
    const IN_FRAME: &CStr = c"frame_input";
    const IN_TEMPORAL: &CStr = c"temporal_warped";
    const OUT_COLOR: &CStr = c"output";
    const OUT_TEMPORAL: &CStr = c"temporal_out";

    // onnxruntime_c_api.h enum values (C ints).
    const LOG_LEVEL_ERROR: i32 = 3; // OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR
    const EXECUTION_SEQUENTIAL: i32 = 0; // ExecutionMode::ORT_SEQUENTIAL
    const OPT_ENABLE_ALL: i32 = 99; // GraphOptimizationLevel::ORT_ENABLE_ALL
    const ALLOCATOR_ARENA: i32 = 1; // OrtAllocatorType::OrtArenaAllocator
    const MEM_TYPE_DEFAULT: i32 = 0; // OrtMemType::OrtMemTypeDefault
    const ELEMENT_F32: i32 = 1; // ONNXTensorElementDataType::..._FLOAT

    /// Which execution provider to ask for.
    #[derive(Clone, Copy, PartialEq)]
    pub enum NppdDevice {
        /// DirectML on adapter 0, CPU EP if that fails — the default.
        Auto,
        /// CPU EP only.
        Cpu,
        /// DirectML on the given adapter, hard error if it fails.
        Dml(i32),
    }

    // Opaque ORT handle types — never constructed, only pointed at.
    macro_rules! opaque {
        ($($n:ident),*) => {$(
            #[repr(C)]
            pub struct $n {
                _private: [u8; 0],
            }
        )*};
    }
    opaque!(
        OrtStatus,
        OrtEnv,
        OrtSession,
        OrtSessionOptions,
        OrtValue,
        OrtMemoryInfo,
        OrtAllocator,
        OrtIoBinding,
        OrtRunOptions
    );

    #[repr(C)]
    struct OrtApiBase {
        get_api: unsafe extern "system" fn(u32) -> *const OrtApi,
        get_version_string: unsafe extern "system" fn() -> *const c_char,
    }

    /// `OrtApi` transcribed MECHANICALLY from onnxruntime v1.17.3's
    /// `onnxruntime_c_api.h` (the field list was extracted from the header by
    /// script, not typed by hand — ordering IS the ABI; see the pin above).
    /// Unused slots are pointer-sized placeholders carrying the header name.
    /// All entries are ORT_API_CALL (stdcall on x86, the plain C convention
    /// on x64) — Rust's `extern "system"`.
    #[repr(C)]
    struct OrtApi {
        _f000: *const c_void, // CreateStatus
        /// [1] GetErrorCode
        get_error_code: unsafe extern "system" fn(*const OrtStatus) -> i32,
        /// [2] GetErrorMessage
        get_error_message: unsafe extern "system" fn(*const OrtStatus) -> *const c_char,
        /// [3] CreateEnv
        create_env: unsafe extern "system" fn(i32, *const c_char, *mut *mut OrtEnv) -> *mut OrtStatus,
        _f004: *const c_void, // CreateEnvWithCustomLogger
        _f005: *const c_void, // EnableTelemetryEvents
        _f006: *const c_void, // DisableTelemetryEvents
        /// [7] CreateSession
        create_session: unsafe extern "system" fn(*mut OrtEnv, *const u16, *mut OrtSessionOptions, *mut *mut OrtSession) -> *mut OrtStatus,
        _f008: *const c_void, // CreateSessionFromArray
        /// [9] Run
        run: unsafe extern "system" fn(*mut OrtSession, *const c_void, *const *const c_char, *const *mut OrtValue, usize, *const *const c_char, usize, *mut *mut OrtValue) -> *mut OrtStatus,
        /// [10] CreateSessionOptions
        create_session_options: unsafe extern "system" fn(*mut *mut OrtSessionOptions) -> *mut OrtStatus,
        _f011: *const c_void, // SetOptimizedModelFilePath
        _f012: *const c_void, // CloneSessionOptions
        /// [13] SetSessionExecutionMode
        set_session_execution_mode: unsafe extern "system" fn(*mut OrtSessionOptions, i32) -> *mut OrtStatus,
        _f014: *const c_void, // EnableProfiling
        _f015: *const c_void, // DisableProfiling
        _f016: *const c_void, // EnableMemPattern
        /// [17] DisableMemPattern
        disable_mem_pattern: unsafe extern "system" fn(*mut OrtSessionOptions) -> *mut OrtStatus,
        _f018: *const c_void, // EnableCpuMemArena
        _f019: *const c_void, // DisableCpuMemArena
        _f020: *const c_void, // SetSessionLogId
        _f021: *const c_void, // SetSessionLogVerbosityLevel
        _f022: *const c_void, // SetSessionLogSeverityLevel
        /// [23] SetSessionGraphOptimizationLevel
        set_session_graph_optimization_level: unsafe extern "system" fn(*mut OrtSessionOptions, i32) -> *mut OrtStatus,
        _f024: *const c_void, // SetIntraOpNumThreads
        _f025: *const c_void, // SetInterOpNumThreads
        _f026: *const c_void, // CreateCustomOpDomain
        _f027: *const c_void, // CustomOpDomain_Add
        _f028: *const c_void, // AddCustomOpDomain
        _f029: *const c_void, // RegisterCustomOpsLibrary
        /// [30] SessionGetInputCount
        session_get_input_count: unsafe extern "system" fn(*mut OrtSession, *mut usize) -> *mut OrtStatus,
        /// [31] SessionGetOutputCount
        session_get_output_count: unsafe extern "system" fn(*mut OrtSession, *mut usize) -> *mut OrtStatus,
        _f032: *const c_void, // SessionGetOverridableInitializerCount
        _f033: *const c_void, // SessionGetInputTypeInfo
        _f034: *const c_void, // SessionGetOutputTypeInfo
        _f035: *const c_void, // SessionGetOverridableInitializerTypeInfo
        /// [36] SessionGetInputName
        session_get_input_name: unsafe extern "system" fn(*mut OrtSession, usize, *mut OrtAllocator, *mut *mut c_char) -> *mut OrtStatus,
        /// [37] SessionGetOutputName
        session_get_output_name: unsafe extern "system" fn(*mut OrtSession, usize, *mut OrtAllocator, *mut *mut c_char) -> *mut OrtStatus,
        _f038: *const c_void, // SessionGetOverridableInitializerName
        /// [39] CreateRunOptions
        create_run_options: unsafe extern "system" fn(*mut *mut OrtRunOptions) -> *mut OrtStatus,
        _f040: *const c_void, // RunOptionsSetRunLogVerbosityLevel
        _f041: *const c_void, // RunOptionsSetRunLogSeverityLevel
        _f042: *const c_void, // RunOptionsSetRunTag
        _f043: *const c_void, // RunOptionsGetRunLogVerbosityLevel
        _f044: *const c_void, // RunOptionsGetRunLogSeverityLevel
        _f045: *const c_void, // RunOptionsGetRunTag
        _f046: *const c_void, // RunOptionsSetTerminate
        _f047: *const c_void, // RunOptionsUnsetTerminate
        _f048: *const c_void, // CreateTensorAsOrtValue
        /// [49] CreateTensorWithDataAsOrtValue
        create_tensor_with_data_as_ort_value: unsafe extern "system" fn(*const OrtMemoryInfo, *mut c_void, usize, *const i64, usize, i32, *mut *mut OrtValue) -> *mut OrtStatus,
        _f050: *const c_void, // IsTensor
        /// [51] GetTensorMutableData
        get_tensor_mutable_data: unsafe extern "system" fn(*mut OrtValue, *mut *mut c_void) -> *mut OrtStatus,
        _f052: *const c_void, // FillStringTensor
        _f053: *const c_void, // GetStringTensorDataLength
        _f054: *const c_void, // GetStringTensorContent
        _f055: *const c_void, // CastTypeInfoToTensorInfo
        _f056: *const c_void, // GetOnnxTypeFromTypeInfo
        _f057: *const c_void, // CreateTensorTypeAndShapeInfo
        _f058: *const c_void, // SetTensorElementType
        _f059: *const c_void, // SetDimensions
        _f060: *const c_void, // GetTensorElementType
        _f061: *const c_void, // GetDimensionsCount
        _f062: *const c_void, // GetDimensions
        _f063: *const c_void, // GetSymbolicDimensions
        _f064: *const c_void, // GetTensorShapeElementCount
        _f065: *const c_void, // GetTensorTypeAndShape
        _f066: *const c_void, // GetTypeInfo
        _f067: *const c_void, // GetValueType
        /// [68] CreateMemoryInfo
        create_memory_info: unsafe extern "system" fn(*const c_char, i32, i32, i32, *mut *mut OrtMemoryInfo) -> *mut OrtStatus,
        /// [69] CreateCpuMemoryInfo
        create_cpu_memory_info: unsafe extern "system" fn(i32, i32, *mut *mut OrtMemoryInfo) -> *mut OrtStatus,
        _f070: *const c_void, // CompareMemoryInfo
        _f071: *const c_void, // MemoryInfoGetName
        _f072: *const c_void, // MemoryInfoGetId
        _f073: *const c_void, // MemoryInfoGetMemType
        _f074: *const c_void, // MemoryInfoGetType
        _f075: *const c_void, // AllocatorAlloc
        /// [76] AllocatorFree
        allocator_free: unsafe extern "system" fn(*mut OrtAllocator, *mut c_void) -> *mut OrtStatus,
        _f077: *const c_void, // AllocatorGetInfo
        /// [78] GetAllocatorWithDefaultOptions
        get_allocator_with_default_options: unsafe extern "system" fn(*mut *mut OrtAllocator) -> *mut OrtStatus,
        _f079: *const c_void, // AddFreeDimensionOverride
        _f080: *const c_void, // GetValue
        _f081: *const c_void, // GetValueCount
        _f082: *const c_void, // CreateValue
        _f083: *const c_void, // CreateOpaqueValue
        _f084: *const c_void, // GetOpaqueValue
        _f085: *const c_void, // KernelInfoGetAttribute_float
        _f086: *const c_void, // KernelInfoGetAttribute_int64
        _f087: *const c_void, // KernelInfoGetAttribute_string
        _f088: *const c_void, // KernelContext_GetInputCount
        _f089: *const c_void, // KernelContext_GetOutputCount
        _f090: *const c_void, // KernelContext_GetInput
        _f091: *const c_void, // KernelContext_GetOutput
        /// [92] ReleaseEnv
        release_env: unsafe extern "system" fn(*mut OrtEnv),
        /// [93] ReleaseStatus
        release_status: unsafe extern "system" fn(*mut OrtStatus),
        /// [94] ReleaseMemoryInfo
        release_memory_info: unsafe extern "system" fn(*mut OrtMemoryInfo),
        /// [95] ReleaseSession
        release_session: unsafe extern "system" fn(*mut OrtSession),
        /// [96] ReleaseValue
        release_value: unsafe extern "system" fn(*mut OrtValue),
        /// [97] ReleaseRunOptions
        release_run_options: unsafe extern "system" fn(*mut OrtRunOptions),
        _f098: *const c_void, // ReleaseTypeInfo
        _f099: *const c_void, // ReleaseTensorTypeAndShapeInfo
        /// [100] ReleaseSessionOptions
        release_session_options: unsafe extern "system" fn(*mut OrtSessionOptions),
        _f101: *const c_void, // ReleaseCustomOpDomain
        _f102: *const c_void, // GetDenotationFromTypeInfo
        _f103: *const c_void, // CastTypeInfoToMapTypeInfo
        _f104: *const c_void, // CastTypeInfoToSequenceTypeInfo
        _f105: *const c_void, // GetMapKeyType
        _f106: *const c_void, // GetMapValueType
        _f107: *const c_void, // GetSequenceElementType
        _f108: *const c_void, // ReleaseMapTypeInfo
        _f109: *const c_void, // ReleaseSequenceTypeInfo
        _f110: *const c_void, // SessionEndProfiling
        _f111: *const c_void, // SessionGetModelMetadata
        _f112: *const c_void, // ModelMetadataGetProducerName
        _f113: *const c_void, // ModelMetadataGetGraphName
        _f114: *const c_void, // ModelMetadataGetDomain
        _f115: *const c_void, // ModelMetadataGetDescription
        _f116: *const c_void, // ModelMetadataLookupCustomMetadataMap
        _f117: *const c_void, // ModelMetadataGetVersion
        _f118: *const c_void, // ReleaseModelMetadata
        _f119: *const c_void, // CreateEnvWithGlobalThreadPools
        _f120: *const c_void, // DisablePerSessionThreads
        _f121: *const c_void, // CreateThreadingOptions
        _f122: *const c_void, // ReleaseThreadingOptions
        _f123: *const c_void, // ModelMetadataGetCustomMetadataMapKeys
        /// [124] AddFreeDimensionOverrideByName
        add_free_dimension_override_by_name: unsafe extern "system" fn(*mut OrtSessionOptions, *const c_char, i64) -> *mut OrtStatus,
        _f125: *const c_void, // GetAvailableProviders
        _f126: *const c_void, // ReleaseAvailableProviders
        _f127: *const c_void, // GetStringTensorElementLength
        _f128: *const c_void, // GetStringTensorElement
        _f129: *const c_void, // FillStringTensorElement
        _f130: *const c_void, // AddSessionConfigEntry
        _f131: *const c_void, // CreateAllocator
        _f132: *const c_void, // ReleaseAllocator
        /// [133] RunWithBinding
        run_with_binding: unsafe extern "system" fn(*mut OrtSession, *const OrtRunOptions, *const OrtIoBinding) -> *mut OrtStatus,
        /// [134] CreateIoBinding
        create_io_binding: unsafe extern "system" fn(*mut OrtSession, *mut *mut OrtIoBinding) -> *mut OrtStatus,
        /// [135] ReleaseIoBinding
        release_io_binding: unsafe extern "system" fn(*mut OrtIoBinding),
        /// [136] BindInput
        bind_input: unsafe extern "system" fn(*mut OrtIoBinding, *const c_char, *const OrtValue) -> *mut OrtStatus,
        /// [137] BindOutput
        bind_output: unsafe extern "system" fn(*mut OrtIoBinding, *const c_char, *const OrtValue) -> *mut OrtStatus,
        _f138: *const c_void, // BindOutputToDevice
        _f139: *const c_void, // GetBoundOutputNames
        _f140: *const c_void, // GetBoundOutputValues
        _f141: *const c_void, // ClearBoundInputs
        _f142: *const c_void, // ClearBoundOutputs
        _f143: *const c_void, // TensorAt
        _f144: *const c_void, // CreateAndRegisterAllocator
        _f145: *const c_void, // SetLanguageProjection
        _f146: *const c_void, // SessionGetProfilingStartTimeNs
        _f147: *const c_void, // SetGlobalIntraOpNumThreads
        _f148: *const c_void, // SetGlobalInterOpNumThreads
        _f149: *const c_void, // SetGlobalSpinControl
        _f150: *const c_void, // AddInitializer
        _f151: *const c_void, // CreateEnvWithCustomLoggerAndGlobalThreadPools
        _f152: *const c_void, // SessionOptionsAppendExecutionProvider_CUDA
        _f153: *const c_void, // SessionOptionsAppendExecutionProvider_ROCM
        _f154: *const c_void, // SessionOptionsAppendExecutionProvider_OpenVINO
        _f155: *const c_void, // SetGlobalDenormalAsZero
        _f156: *const c_void, // CreateArenaCfg
        _f157: *const c_void, // ReleaseArenaCfg
        _f158: *const c_void, // ModelMetadataGetGraphDescription
        _f159: *const c_void, // SessionOptionsAppendExecutionProvider_TensorRT
        _f160: *const c_void, // SetCurrentGpuDeviceId
        _f161: *const c_void, // GetCurrentGpuDeviceId
        _f162: *const c_void, // KernelInfoGetAttributeArray_float
        _f163: *const c_void, // KernelInfoGetAttributeArray_int64
        _f164: *const c_void, // CreateArenaCfgV2
        _f165: *const c_void, // AddRunConfigEntry
        _f166: *const c_void, // CreatePrepackedWeightsContainer
        _f167: *const c_void, // ReleasePrepackedWeightsContainer
        _f168: *const c_void, // CreateSessionWithPrepackedWeightsContainer
        _f169: *const c_void, // CreateSessionFromArrayWithPrepackedWeightsContainer
        _f170: *const c_void, // SessionOptionsAppendExecutionProvider_TensorRT_V2
        _f171: *const c_void, // CreateTensorRTProviderOptions
        _f172: *const c_void, // UpdateTensorRTProviderOptions
        _f173: *const c_void, // GetTensorRTProviderOptionsAsString
        _f174: *const c_void, // ReleaseTensorRTProviderOptions
        _f175: *const c_void, // EnableOrtCustomOps
        _f176: *const c_void, // RegisterAllocator
        _f177: *const c_void, // UnregisterAllocator
        _f178: *const c_void, // IsSparseTensor
        _f179: *const c_void, // CreateSparseTensorAsOrtValue
        _f180: *const c_void, // FillSparseTensorCoo
        _f181: *const c_void, // FillSparseTensorCsr
        _f182: *const c_void, // FillSparseTensorBlockSparse
        _f183: *const c_void, // CreateSparseTensorWithValuesAsOrtValue
        _f184: *const c_void, // UseCooIndices
        _f185: *const c_void, // UseCsrIndices
        _f186: *const c_void, // UseBlockSparseIndices
        _f187: *const c_void, // GetSparseTensorFormat
        _f188: *const c_void, // GetSparseTensorValuesTypeAndShape
        _f189: *const c_void, // GetSparseTensorValues
        _f190: *const c_void, // GetSparseTensorIndicesTypeShape
        _f191: *const c_void, // GetSparseTensorIndices
        _f192: *const c_void, // HasValue
        _f193: *const c_void, // KernelContext_GetGPUComputeStream
        _f194: *const c_void, // GetTensorMemoryInfo
        /// [195] GetExecutionProviderApi
        get_execution_provider_api: unsafe extern "system" fn(*const c_char, u32, *mut *const c_void) -> *mut OrtStatus,
        _f196: *const c_void, // SessionOptionsSetCustomCreateThreadFn
        _f197: *const c_void, // SessionOptionsSetCustomThreadCreationOptions
        _f198: *const c_void, // SessionOptionsSetCustomJoinThreadFn
        _f199: *const c_void, // SetGlobalCustomCreateThreadFn
        _f200: *const c_void, // SetGlobalCustomThreadCreationOptions
        _f201: *const c_void, // SetGlobalCustomJoinThreadFn
        /// [202] SynchronizeBoundInputs
        synchronize_bound_inputs: unsafe extern "system" fn(*mut OrtIoBinding) -> *mut OrtStatus,
        /// [203] SynchronizeBoundOutputs
        synchronize_bound_outputs: unsafe extern "system" fn(*mut OrtIoBinding) -> *mut OrtStatus,
        // Fields past 203 exist in the real struct but are never read; a
        // shorter Rust declaration is sound (we only ever hold a pointer).
    }

    /// `OrtDmlApi`, transcribed from onnxruntime v1.17.3's
    /// dml_provider_factory.h (all 6 entries; append-only like OrtApi). The
    /// COM pointers (IDMLDevice*, ID3D12CommandQueue*, ID3D12Resource*)
    /// travel as raw `*mut c_void` — this module never links d3d12/directml,
    /// the caller hands in `.as_raw()` pointers it keeps alive.
    #[repr(C)]
    struct OrtDmlApi {
        /// [0] SessionOptionsAppendExecutionProvider_DML
        append_dml: unsafe extern "system" fn(*mut OrtSessionOptions, i32) -> *mut OrtStatus,
        /// [1] SessionOptionsAppendExecutionProvider_DML1 — the interop form:
        /// ORT executes on the GIVEN queue (must be DIRECT or COMPUTE, same
        /// parent device as the IDMLDevice).
        append_dml1: unsafe extern "system" fn(*mut OrtSessionOptions, *mut c_void, *mut c_void) -> *mut OrtStatus,
        /// [2] CreateGPUAllocationFromD3DResource — wraps an ID3D12Resource
        /// (default-heap buffer, ALLOW_UNORDERED_ACCESS, resting in the
        /// UNORDERED_ACCESS state) as a pointer usable as OrtValue data.
        create_gpu_allocation_from_d3d_resource: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> *mut OrtStatus,
        /// [3] FreeGPUAllocation
        free_gpu_allocation: unsafe extern "system" fn(*mut c_void) -> *mut OrtStatus,
        /// [4] GetD3D12ResourceFromAllocation
        get_d3d12_resource_from_allocation: unsafe extern "system" fn(*mut OrtAllocator, *mut c_void, *mut *mut c_void) -> *mut OrtStatus,
        /// [5] SessionOptionsAppendExecutionProvider_DML2
        append_dml2: unsafe extern "system" fn(*mut OrtSessionOptions, *mut c_void) -> *mut OrtStatus,
    }

    fn load_dll(dir: &str, name: &str) -> Result<HMODULE, String> {
        let path = format!("{dir}\\{name}");
        let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
        // ALTERED_SEARCH_PATH so each DLL's own imports resolve next to it.
        unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, LOAD_WITH_ALTERED_SEARCH_PATH) }
            .map_err(|e| format!("failed to load {path}: {e}"))
    }

    /// Consume an `OrtStatus*`: null = success, anything else = copied
    /// message + release.
    fn check(api: &OrtApi, status: *mut OrtStatus, what: &str) -> Result<(), String> {
        if status.is_null() {
            return Ok(());
        }
        let (code, msg) = unsafe {
            let code = (api.get_error_code)(status);
            let m = (api.get_error_message)(status);
            let msg = if m.is_null() {
                String::new()
            } else {
                CStr::from_ptr(m).to_string_lossy().into_owned()
            };
            (api.release_status)(status);
            (code, msg)
        };
        Err(format!("nppd {what}: ort error {code}: {msg}"))
    }

    /// Which execution provider a session is opened with. `Dml1` is the
    /// interop form: ORT executes on the GIVEN queue (DIRECT or COMPUTE, same
    /// parent ID3D12Device as the IDMLDevice) — the GPU-resident path.
    #[derive(Clone, Copy)]
    enum SessionEp {
        Cpu,
        Dml(i32),
        Dml1 {
            dml_device: *mut c_void,
            queue: *mut c_void,
        },
    }

    /// Create a session with the model's dynamic h/w axes FROZEN to the
    /// padded dims via `AddFreeDimensionOverrideByName` — the exported graph
    /// is launch-bound on ~1600 tiny shape-math ops (the pyramid's dynamic
    /// slices), and pinning the dims lets ORT's optimizer constant-fold them
    /// (measured 93 -> 70 ms at 1280x736 on DML). The session then only
    /// accepts tensors at exactly (pw, ph); a real res change rebuilds it.
    /// The DML EP contract (dml_provider_factory docs): memory pattern OFF,
    /// sequential execution.
    fn open_session(
        api: &'static OrtApi,
        env: *mut OrtEnv,
        model_wide: &[u16],
        ep: SessionEp,
        pw: usize,
        ph: usize,
    ) -> Result<*mut OrtSession, String> {
        let mut opts: *mut OrtSessionOptions = std::ptr::null_mut();
        check(api, unsafe { (api.create_session_options)(&mut opts) }, "session options")?;
        let r = (|| {
            check(api, unsafe {
                (api.set_session_graph_optimization_level)(opts, OPT_ENABLE_ALL)
            }, "opt level")?;
            check(api, unsafe {
                (api.add_free_dimension_override_by_name)(opts, c"h".as_ptr(), ph as i64)
            }, "freeze h")?;
            check(api, unsafe {
                (api.add_free_dimension_override_by_name)(opts, c"w".as_ptr(), pw as i64)
            }, "freeze w")?;
            if !matches!(ep, SessionEp::Cpu) {
                check(api, unsafe { (api.disable_mem_pattern)(opts) }, "mem pattern")?;
                check(api, unsafe {
                    (api.set_session_execution_mode)(opts, EXECUTION_SEQUENTIAL)
                }, "execution mode")?;
                let dml_api = get_dml_api(api)?;
                match ep {
                    SessionEp::Dml(dev) => {
                        check(api, unsafe { (dml_api.append_dml)(opts, dev) }, "DML append")?
                    }
                    SessionEp::Dml1 { dml_device, queue } => check(api, unsafe {
                        (dml_api.append_dml1)(opts, dml_device, queue)
                    }, "DML1 append (shared device/queue)")?,
                    SessionEp::Cpu => unreachable!(),
                }
            }
            Ok(())
        })();
        if let Err(e) = r {
            unsafe { (api.release_session_options)(opts) };
            return Err(e);
        }
        let mut session: *mut OrtSession = std::ptr::null_mut();
        let r = check(api, unsafe {
            (api.create_session)(env, model_wide.as_ptr(), opts, &mut session)
        }, "create_session");
        unsafe { (api.release_session_options)(opts) };
        r.map(|()| session)
    }

    /// Fetch the DML provider API table (the DirectML build of the runtime).
    fn get_dml_api(api: &'static OrtApi) -> Result<&'static OrtDmlApi, String> {
        let mut dml_api: *const c_void = std::ptr::null();
        check(api, unsafe {
            (api.get_execution_provider_api)(c"DML".as_ptr(), ORT_API_VERSION, &mut dml_api)
        }, "DML provider api (is this the DirectML build of onnxruntime.dll?)")?;
        Ok(unsafe { &*dml_api.cast() })
    }

    /// Canonicalize the DLL directory (ALTERED_SEARCH_PATH only alters the
    /// search for absolute paths), load the DLLs, and bootstrap the versioned
    /// `OrtApi` from the ONE export. `DirectML.dll` is loaded first,
    /// best-effort (ORT delay-loads it; a system copy may also exist); its
    /// handle comes back for the GPU path's `DMLCreateDevice`.
    fn ort_bootstrap(dll_dir: &str) -> Result<(&'static OrtApi, String, Option<HMODULE>), String> {
        let dir = std::fs::canonicalize(dll_dir)
            .map(|p| p.to_string_lossy().trim_start_matches(r"\\?\").to_string())
            .unwrap_or_else(|_| dll_dir.to_string());
        let dml_dll = load_dll(&dir, "DirectML.dll").ok();
        let h_dll = load_dll(&dir, "onnxruntime.dll")?;
        let base = unsafe {
            let sym = GetProcAddress(h_dll, PCSTR(c"OrtGetApiBase".as_ptr().cast()))
                .ok_or("onnxruntime.dll: missing export OrtGetApiBase")?;
            let get_base: unsafe extern "system" fn() -> *const OrtApiBase =
                std::mem::transmute(sym);
            &*get_base()
        };
        let ort_version = unsafe {
            CStr::from_ptr((base.get_version_string)()).to_string_lossy().into_owned()
        };
        let api_ptr = unsafe { (base.get_api)(ORT_API_VERSION) };
        if api_ptr.is_null() {
            return Err(format!(
                "onnxruntime {ort_version} does not serve API version {ORT_API_VERSION} \
                 (need >= 1.17)"
            ));
        }
        Ok((unsafe { &*api_ptr }, ort_version, dml_dll))
    }

    /// I/O sanity: the exported graph's names and counts, so a stale or
    /// foreign .onnx fails at init, not inside Run.
    fn verify_io(api: &'static OrtApi, session: *mut OrtSession) -> Result<(), String> {
        let mut alloc: *mut OrtAllocator = std::ptr::null_mut();
        check(api, unsafe { (api.get_allocator_with_default_options)(&mut alloc) }, "allocator")?;
        let mut n_in = 0usize;
        let mut n_out = 0usize;
        check(api, unsafe { (api.session_get_input_count)(session, &mut n_in) }, "input count")?;
        check(api, unsafe { (api.session_get_output_count)(session, &mut n_out) }, "output count")?;
        if (n_in, n_out) != (2, 2) {
            return Err(format!("model has {n_in} inputs / {n_out} outputs, want 2 / 2"));
        }
        for (i, want, which) in [
            (0usize, IN_FRAME, "input"),
            (1, IN_TEMPORAL, "input"),
            (0, OUT_COLOR, "output"),
            (1, OUT_TEMPORAL, "output"),
        ] {
            let mut name: *mut c_char = std::ptr::null_mut();
            let get = if which == "input" {
                api.session_get_input_name
            } else {
                api.session_get_output_name
            };
            check(api, unsafe { get(session, i, alloc, &mut name) }, "io name")?;
            let got = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
            let _ = check(api, unsafe { (api.allocator_free)(alloc, name.cast()) }, "free name");
            if got != want.to_string_lossy() {
                return Err(format!(
                    "model {which} {i} is '{got}', want '{}' — re-export?",
                    want.to_string_lossy()
                ));
            }
        }
        Ok(())
    }

    pub struct NppdContext {
        api: &'static OrtApi,
        env: *mut OrtEnv,
        session: *mut OrtSession,
        mem_info: *mut OrtMemoryInfo,
        /// Padded-capacity pixel count (`pad_dims` of the construction res) —
        /// every buffer below is allocated once at this size; `set_res` only
        /// reinterprets within it.
        cap_px: usize,
        w: usize,
        h: usize,
        pw: usize,
        ph: usize,
        /// NCHW frame stack, `CH_FRAME` planes.
        staging_frame: Vec<f32>,
        /// The recurrent state as of the last denoise, `C_T` planes. The
        /// graph writes `temporal_out` STRAIGHT into this buffer (a
        /// pre-created output tensor — no post-run copy of the ~C_T-plane
        /// state).
        state: Vec<f32>,
        /// The warped state handed to the graph, `C_T` planes.
        warped: Vec<f32>,
        /// Planar network color output (3 planes, padded).
        out_planar: Vec<f32>,
        /// Cropped interleaved output, the `last_output` contract.
        staging_out: Vec<f32>,
        temporal_valid: bool,
        pub device_desc: &'static str,
        /// onnxruntime's own version string (init log + version pin sanity).
        pub ort_version: String,
        /// Wall time of the last `denoise` (warp + pack + session run + crop).
        pub last_ms: f64,
        /// Session-rebuild parameters (the dims are frozen per session; a
        /// padded-res change in `set_res` reopens with these).
        model_wide: Vec<u16>,
        model_path: String,
        dml_dev: Option<i32>,
    }

    // Raw ORT handles are plain pointers into a thread-safe C runtime; the
    // context is used from one thread at a time (the render loop).
    unsafe impl Send for NppdContext {}

    impl NppdContext {
        /// Load the DLLs from `dll_dir`, create the env + session for the
        /// exported model at `model_path`, and allocate staging for `w`×`h`
        /// (also the capacity for later `set_res`). `device` picks the
        /// execution provider; `Auto` falls back to CPU with a note on
        /// stderr, `Dml(_)` makes DML failure a hard error.
        pub fn new(
            dll_dir: &str,
            model_path: &str,
            w: usize,
            h: usize,
            device: NppdDevice,
        ) -> Result<Self, String> {
            Self::with_capacity(dll_dir, model_path, w, h, w, h, device)
        }

        /// `new` with the staging capacity (the `set_res` ceiling) decoupled
        /// from the initial resolution: the session dims are frozen at
        /// `w`×`h`'s padding (see `open_session`), so opening at the res the
        /// first denoise will use avoids an immediate rebuild — the XeSS
        /// composition opens at the locked render res with window-res
        /// capacity, keeping X-off standalone J and dynamic-DRS steps within
        /// bounds (those still rebuild the session, as documented).
        #[allow(clippy::too_many_arguments)]
        pub fn with_capacity(
            dll_dir: &str,
            model_path: &str,
            w: usize,
            h: usize,
            cap_w: usize,
            cap_h: usize,
            device: NppdDevice,
        ) -> Result<Self, String> {
            let (api, ort_version, _dml_dll) = ort_bootstrap(dll_dir)?;

            let mut env: *mut OrtEnv = std::ptr::null_mut();
            check(api, unsafe {
                (api.create_env)(LOG_LEVEL_ERROR, c"frustracer".as_ptr(), &mut env)
            }, "create_env")?;

            let (pw, ph) = pad_dims(w, h);
            let wide: Vec<u16> =
                model_path.encode_utf16().chain(std::iter::once(0)).collect();
            let try_session = |ep: SessionEp| -> Result<*mut OrtSession, String> {
                open_session(api, env, &wide, ep, pw, ph).map_err(|e| {
                    format!("{e} (model: {model_path} — run tools/nppd-export/export.py?)")
                })
            };

            let (session, device_desc, dml_dev) = match device {
                NppdDevice::Cpu => (try_session(SessionEp::Cpu), "cpu", None),
                NppdDevice::Dml(dev) => (try_session(SessionEp::Dml(dev)), "dml", Some(dev)),
                NppdDevice::Auto => match try_session(SessionEp::Dml(0)) {
                    Ok(s) => (Ok(s), "dml", Some(0)),
                    Err(e) => {
                        eprintln!("nppd: DirectML unavailable ({e}); falling back to CPU EP");
                        (try_session(SessionEp::Cpu), "cpu", None)
                    }
                },
            };
            let session = match session {
                Ok(s) => s,
                Err(e) => {
                    unsafe { (api.release_env)(env) };
                    return Err(e);
                }
            };

            let cleanup = |api: &OrtApi| unsafe {
                (api.release_session)(session);
                (api.release_env)(env);
            };

            if let Err(e) = verify_io(api, session) {
                cleanup(api);
                return Err(format!("nppd model check: {e}"));
            }

            let mut mem_info: *mut OrtMemoryInfo = std::ptr::null_mut();
            if let Err(e) = check(api, unsafe {
                (api.create_cpu_memory_info)(ALLOCATOR_ARENA, MEM_TYPE_DEFAULT, &mut mem_info)
            }, "cpu memory info") {
                cleanup(api);
                return Err(e);
            }

            let (cpw, cph) = pad_dims(cap_w.max(w), cap_h.max(h));
            let cap_px = cpw * cph;
            eprintln!("nppd: onnxruntime {ort_version} | {device_desc} | model {model_path}");
            Ok(Self {
                api,
                env,
                session,
                mem_info,
                cap_px,
                w,
                h,
                pw,
                ph,
                staging_frame: vec![0.0; CH_FRAME * cap_px],
                state: vec![0.0; C_T * cap_px],
                warped: vec![0.0; C_T * cap_px],
                out_planar: vec![0.0; 3 * cap_px],
                staging_out: vec![0.0; w * h * 3],
                temporal_valid: false,
                device_desc,
                ort_version,
                last_ms: 0.0,
                model_wide: wide,
                model_path: model_path.to_string(),
                dml_dev,
            })
        }

        /// Forget the recurrent state — the next `denoise` starts from a
        /// zero temporal tensor (upstream `temporal_init` semantics). Called
        /// on every predicate that resets `frame`, never on camera motion.
        pub fn reset_temporal(&mut self) {
            self.temporal_valid = false;
        }

        /// Whether the recurrent state currently carries history (the check
        /// suite's must-fire probe).
        pub fn temporal_valid(&self) -> bool {
            self.temporal_valid
        }

        /// Reinterpret the staging buffers at a new logical resolution within
        /// the construction capacity (the XeSS-composition path; the
        /// standalone mode is fixed window-res). A res change invalidates the
        /// recurrent state — its planes are laid out for the old pitch — and,
        /// when the PADDED dims change, rebuilds the session (the dims are
        /// frozen per session; a DML rebuild re-specializes the graph,
        /// hundreds of ms — rare outside `--lock-res dynamic` ramps, which
        /// the startup note already warns about).
        pub fn set_res(&mut self, w: usize, h: usize) -> Result<(), String> {
            if (w, h) == (self.w, self.h) {
                return Ok(());
            }
            let (pw, ph) = pad_dims(w, h);
            if pw * ph > self.cap_px || w == 0 || h == 0 {
                return Err(format!(
                    "nppd: set_res {w}x{h} (padded {pw}x{ph}) exceeds capacity {} px",
                    self.cap_px
                ));
            }
            if (pw, ph) != (self.pw, self.ph) {
                let ep = match self.dml_dev {
                    Some(dev) => SessionEp::Dml(dev),
                    None => SessionEp::Cpu,
                };
                let session = open_session(self.api, self.env, &self.model_wide, ep, pw, ph)
                    .map_err(|e| format!("{e} (model: {})", self.model_path))?;
                unsafe { (self.api.release_session)(self.session) };
                self.session = session;
            }
            (self.w, self.h, self.pw, self.ph) = (w, h, pw, ph);
            if self.staging_out.len() < w * h * 3 {
                self.staging_out.resize(w * h * 3, 0.0);
            }
            self.temporal_valid = false;
            Ok(())
        }

        /// One recurrent denoise step: warp the state by this frame's motion
        /// vectors, pack the frame stack, run the graph (writing the new
        /// state in place), crop the padded planar output to interleaved.
        /// `accum` holds exactly one 1-spp linear-HDR frame at `g`'s res;
        /// `basis`/`far` are the frame's own camera (the pack transforms).
        pub fn denoise(
            &mut self,
            accum: &[AtomicU32],
            g: &GBufs,
            basis: &CamBasis,
            far: f32,
        ) -> Result<&[f32], String> {
            crate::zone!("nppd-denoise");
            let t0 = Instant::now();
            let (w, h, pw, ph) = (self.w, self.h, self.pw, self.ph);
            assert_eq!((g.rw, g.rh), (w, h), "gbuf/nppd resolution mismatch");
            let n_state = C_T * pw * ph;

            if self.temporal_valid {
                warp_temporal(
                    &self.state[..n_state],
                    &g.mvec[..w * h * 2],
                    C_T,
                    w,
                    h,
                    pw,
                    ph,
                    &mut self.warped[..n_state],
                );
            } else {
                self.warped[..n_state].fill(0.0);
            }
            pack_inputs(accum, g, basis, far, pw, ph, &mut self.staging_frame[..CH_FRAME * pw * ph]);

            self.run_graph()?;

            crop_to_interleaved(
                &self.out_planar[..3 * pw * ph],
                pw,
                ph,
                w,
                h,
                &mut self.staging_out[..w * h * 3],
            );
            self.temporal_valid = true;
            self.last_ms = t0.elapsed().as_secs_f64() * 1000.0;
            Ok(&self.staging_out[..w * h * 3])
        }

        /// Wrap the staging buffers in ORT tensors and run the session. The
        /// two outputs are PRE-CREATED tensors over `out_planar` and `state`,
        /// so the graph writes results in place — no post-run copies.
        fn run_graph(&mut self) -> Result<(), String> {
            let api = self.api;
            let (pw, ph) = (self.pw as i64, self.ph as i64);
            let mk_tensor = |data: &mut [f32], dims: &[i64]| -> Result<*mut OrtValue, String> {
                let mut v: *mut OrtValue = std::ptr::null_mut();
                check(api, unsafe {
                    (api.create_tensor_with_data_as_ort_value)(
                        self.mem_info,
                        data.as_mut_ptr().cast(),
                        std::mem::size_of_val(data),
                        dims.as_ptr(),
                        dims.len(),
                        ELEMENT_F32,
                        &mut v,
                    )
                }, "create tensor")?;
                Ok(v)
            };

            let n_frame = CH_FRAME * (pw * ph) as usize;
            let n_state = C_T * (pw * ph) as usize;
            let mut values: Vec<*mut OrtValue> = Vec::with_capacity(4);
            let release_all = |vals: &[*mut OrtValue]| {
                for &v in vals {
                    if !v.is_null() {
                        unsafe { (api.release_value)(v) };
                    }
                }
            };
            let r = (|| {
                values.push(mk_tensor(
                    &mut self.staging_frame[..n_frame],
                    &[1, 1, CH_FRAME as i64, ph, pw],
                )?);
                values.push(mk_tensor(&mut self.warped[..n_state], &[1, C_T as i64, ph, pw])?);
                values.push(mk_tensor(
                    &mut self.out_planar[..3 * (pw * ph) as usize],
                    &[1, 3, ph, pw],
                )?);
                values.push(mk_tensor(&mut self.state[..n_state], &[1, C_T as i64, ph, pw])?);

                let in_names = [IN_FRAME.as_ptr(), IN_TEMPORAL.as_ptr()];
                let out_names = [OUT_COLOR.as_ptr(), OUT_TEMPORAL.as_ptr()];
                let inputs = [values[0], values[1]];
                let mut outputs = [values[2], values[3]];
                check(api, unsafe {
                    (api.run)(
                        self.session,
                        std::ptr::null(),
                        in_names.as_ptr(),
                        inputs.as_ptr(),
                        2,
                        out_names.as_ptr(),
                        2,
                        outputs.as_mut_ptr(),
                    )
                }, "run")
            })();
            release_all(&values);
            r
        }

        /// The most recent denoised output (valid after any successful
        /// `denoise` at the current resolution) — lets the caller
        /// re-composite the overlay without re-running the network.
        pub fn last_output(&self) -> &[f32] {
            &self.staging_out[..self.w * self.h * 3]
        }

        /// The current recurrent state plane view (check-suite probes only).
        pub fn state(&self) -> &[f32] {
            &self.state[..C_T * self.pw * self.ph]
        }
    }

    impl Drop for NppdContext {
        fn drop(&mut self) {
            unsafe {
                (self.api.release_memory_info)(self.mem_info);
                (self.api.release_session)(self.session);
                (self.api.release_env)(self.env);
            }
        }
    }

    // ---- GPU-resident NPPD (the --gpu --nppd composition) ----

    // Minimal COM plumbing for the one interface created here (IDMLDevice
    // from DMLCreateDevice) — nothing links DirectML, and windows-rs has no
    // IDMLDevice binding to borrow.
    #[repr(C)]
    struct Guid {
        data1: u32,
        data2: u16,
        data3: u16,
        data4: [u8; 8],
    }
    /// IDMLDevice — DirectML.h
    /// `DML_DECLARE_INTERFACE("6dbd6437-96fd-423f-a98c-ae5e7c2a573f")`.
    const IID_IDML_DEVICE: Guid = Guid {
        data1: 0x6dbd6437,
        data2: 0x96fd,
        data3: 0x423f,
        data4: [0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f],
    };
    unsafe fn com_release(p: *mut c_void) {
        #[repr(C)]
        struct Vtbl {
            _query_interface: *const c_void,
            _add_ref: *const c_void,
            release: unsafe extern "system" fn(*mut c_void) -> u32,
        }
        unsafe { ((*(*(p as *mut *const Vtbl))).release)(p) };
    }

    const ALLOCATOR_DEVICE: i32 = 0; // OrtAllocatorType::OrtDeviceAllocator

    /// The same exported graph with ONNX Runtime executing on the CALLER's
    /// D3D12 queue (`SessionOptionsAppendExecutionProvider_DML1` — the queue
    /// must be DIRECT or COMPUTE with the same parent ID3D12Device as the
    /// IDMLDevice created here) and the four I/O tensors bound ONCE over
    /// caller-owned D3D12 buffers (`CreateGPUAllocationFromD3DResource` +
    /// IoBinding): zero per-frame CPU traffic (measured 94 → 26 ms at
    /// 1280×736 vs the CPU-staged fp32 path). Bindings are fixed by design —
    /// the caller's warp kernel reads `state` and writes `warped`, the graph
    /// reads `warped` and writes `state` back in place, so there is no
    /// ping-pong swap and no per-frame rebinding. The dims are frozen to
    /// `pad_dims(w, h)` (see `open_session`); the render res is LOCKED per
    /// `--gpu` session, so the session is built exactly once.
    ///
    /// Buffer contract: default-heap raw buffers with ALLOW_UNORDERED_ACCESS,
    /// resting in the UNORDERED_ACCESS state (what DML binds); `frame` =
    /// `CH_FRAME` planes, `warped`/`state` = `C_T` planes, `out` = 3 planes,
    /// each plane `pw·ph` f32, NCHW plane-major. The caller keeps the
    /// resources alive for this struct's whole life and orders its own
    /// writes/reads around `run()` by submitting to the shared queue first —
    /// single-queue FIFO is the only synchronization.
    pub struct NppdGpu {
        api: &'static OrtApi,
        dml_api: &'static OrtDmlApi,
        env: *mut OrtEnv,
        session: *mut OrtSession,
        mem_info: *mut OrtMemoryInfo,
        binding: *mut OrtIoBinding,
        run_opts: *mut OrtRunOptions,
        /// IDMLDevice created over the caller's device (owned; COM-released).
        dml_device: *mut c_void,
        /// GPU allocations wrapping [frame, warped, out, state].
        allocs: [*mut c_void; 4],
        values: [*mut OrtValue; 4],
        pub ort_version: String,
    }

    // Same argument as NppdContext: raw handles into a thread-safe C
    // runtime, used from one thread at a time.
    unsafe impl Send for NppdGpu {}

    impl NppdGpu {
        /// `d3d_device`/`queue` are `ID3D12Device*`/`ID3D12CommandQueue*` as
        /// raw pointers (`.as_raw()`); `frame`/`warped`/`out`/`state` are the
        /// four `ID3D12Resource*` buffers sized for `pad_dims(w, h)` per the
        /// struct docs. Failure releases everything it created (Drop).
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            dll_dir: &str,
            model_path: &str,
            d3d_device: *mut c_void,
            queue: *mut c_void,
            w: usize,
            h: usize,
            frame: *mut c_void,
            warped: *mut c_void,
            out: *mut c_void,
            state: *mut c_void,
        ) -> Result<Self, String> {
            let (api, ort_version, dml_dll) = ort_bootstrap(dll_dir)?;
            let dml_dll =
                dml_dll.ok_or_else(|| format!("DirectML.dll not found in {dll_dir}"))?;
            let dml_api = get_dml_api(api)?;

            // DMLCreateDevice over the caller's device — the DML1 form needs
            // an IDMLDevice with the same parent as the queue.
            let dml_create: unsafe extern "system" fn(
                *mut c_void,
                u32,
                *const Guid,
                *mut *mut c_void,
            ) -> i32 = unsafe {
                std::mem::transmute(
                    GetProcAddress(dml_dll, PCSTR(c"DMLCreateDevice".as_ptr().cast()))
                        .ok_or("DirectML.dll: missing export DMLCreateDevice")?,
                )
            };
            let mut dml_device: *mut c_void = std::ptr::null_mut();
            let hr = unsafe { dml_create(d3d_device, 0, &IID_IDML_DEVICE, &mut dml_device) };
            if hr < 0 || dml_device.is_null() {
                return Err(format!("DMLCreateDevice failed: 0x{hr:08x}"));
            }

            // From here on Drop cleans up whatever exists — fill in order.
            let mut me = Self {
                api,
                dml_api,
                env: std::ptr::null_mut(),
                session: std::ptr::null_mut(),
                mem_info: std::ptr::null_mut(),
                binding: std::ptr::null_mut(),
                run_opts: std::ptr::null_mut(),
                dml_device,
                allocs: [std::ptr::null_mut(); 4],
                values: [std::ptr::null_mut(); 4],
                ort_version,
            };

            check(api, unsafe {
                (api.create_env)(LOG_LEVEL_ERROR, c"frustracer".as_ptr(), &mut me.env)
            }, "create_env")?;

            let (pw, ph) = pad_dims(w, h);
            let wide: Vec<u16> =
                model_path.encode_utf16().chain(std::iter::once(0)).collect();
            me.session = open_session(
                api,
                me.env,
                &wide,
                SessionEp::Dml1 { dml_device, queue },
                pw,
                ph,
            )
            .map_err(|e| {
                format!("{e} (model: {model_path} — run tools/nppd-export/export.py?)")
            })?;
            verify_io(api, me.session).map_err(|e| format!("nppd model check: {e}"))?;

            check(api, unsafe {
                (api.create_memory_info)(
                    c"DML".as_ptr(),
                    ALLOCATOR_DEVICE,
                    0,
                    MEM_TYPE_DEFAULT,
                    &mut me.mem_info,
                )
            }, "DML memory info")?;

            let dims_frame = [1i64, 1, CH_FRAME as i64, ph as i64, pw as i64];
            let dims_ct = [1i64, C_T as i64, ph as i64, pw as i64];
            let dims_out = [1i64, 3, ph as i64, pw as i64];
            let io: [(*mut c_void, &[i64]); 4] = [
                (frame, &dims_frame),
                (warped, &dims_ct),
                (out, &dims_out),
                (state, &dims_ct),
            ];
            for (slot, (res, dims)) in io.into_iter().enumerate() {
                check(api, unsafe {
                    (dml_api.create_gpu_allocation_from_d3d_resource)(res, &mut me.allocs[slot])
                }, "gpu allocation")?;
                let bytes = dims.iter().product::<i64>() as usize * 4;
                check(api, unsafe {
                    (api.create_tensor_with_data_as_ort_value)(
                        me.mem_info,
                        me.allocs[slot],
                        bytes,
                        dims.as_ptr(),
                        dims.len(),
                        ELEMENT_F32,
                        &mut me.values[slot],
                    )
                }, "gpu tensor")?;
            }

            check(api, unsafe {
                (api.create_io_binding)(me.session, &mut me.binding)
            }, "io binding")?;
            check(api, unsafe {
                (api.bind_input)(me.binding, IN_FRAME.as_ptr(), me.values[0])
            }, "bind frame_input")?;
            check(api, unsafe {
                (api.bind_input)(me.binding, IN_TEMPORAL.as_ptr(), me.values[1])
            }, "bind temporal_warped")?;
            check(api, unsafe {
                (api.bind_output)(me.binding, OUT_COLOR.as_ptr(), me.values[2])
            }, "bind output")?;
            check(api, unsafe {
                (api.bind_output)(me.binding, OUT_TEMPORAL.as_ptr(), me.values[3])
            }, "bind temporal_out")?;
            check(api, unsafe { (api.create_run_options)(&mut me.run_opts) }, "run options")?;

            eprintln!(
                "nppd-gpu: onnxruntime {} | DML1 (shared device/queue) | model {model_path}",
                me.ort_version
            );
            Ok(me)
        }

        /// One inference over the fixed bindings. The caller must have
        /// SUBMITTED (ExecuteCommandLists) the work that writes `frame` and
        /// `warped` to the shared queue before calling — ORT's DML work is
        /// enqueued behind it, and anything the caller submits afterwards
        /// reads the outputs in queue order.
        pub fn run(&mut self) -> Result<(), String> {
            crate::zone!("nppd-gpu-run");
            check(self.api, unsafe {
                (self.api.run_with_binding)(self.session, self.run_opts, self.binding)
            }, "run_with_binding")
        }

        /// Block the CPU until the bound outputs are GPU-complete — the
        /// check-suite readback path only; the frame loop never calls this.
        pub fn sync_outputs(&self) -> Result<(), String> {
            check(self.api, unsafe {
                (self.api.synchronize_bound_outputs)(self.binding)
            }, "synchronize outputs")
        }
    }

    impl Drop for NppdGpu {
        fn drop(&mut self) {
            unsafe {
                if !self.run_opts.is_null() {
                    (self.api.release_run_options)(self.run_opts);
                }
                if !self.binding.is_null() {
                    (self.api.release_io_binding)(self.binding);
                }
                for v in self.values {
                    if !v.is_null() {
                        (self.api.release_value)(v);
                    }
                }
                for a in self.allocs {
                    if !a.is_null() {
                        let s = (self.dml_api.free_gpu_allocation)(a);
                        if !s.is_null() {
                            (self.api.release_status)(s);
                        }
                    }
                }
                if !self.mem_info.is_null() {
                    (self.api.release_memory_info)(self.mem_info);
                }
                if !self.session.is_null() {
                    (self.api.release_session)(self.session);
                }
                if !self.env.is_null() {
                    (self.api.release_env)(self.env);
                }
                com_release(self.dml_device);
            }
        }
    }
}
