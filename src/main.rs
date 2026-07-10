mod bvh;
mod camera;
mod dlss;
mod frustum;
mod hemi;
// The presentation stack (D3D12 + Streamline) is Windows-only; everything
// headless (--check, --check-dlss) stays cross-platform.
#[cfg(windows)]
mod gpu;
#[cfg(windows)]
mod input;
// OIDN loads its DLLs through the Win32 loader; the denoiser itself is
// CPU/GPU-agnostic but the SDK drop and load path here are Windows-only.
#[cfg(windows)]
mod oidn;
mod overlay;
mod render;
mod reproject;
mod scene;
mod shade;
mod shaft;
mod sphcell;
mod stats;
mod temporal;
// The loader half is Windows-only (LoadLibrary); the FFI structs, depth
// encoding, and the dynamic-res controller are pure and feed --check-xess.
mod xess;

use camera::Camera;
use glam::Vec3A;
use rayon::prelude::*;
use render::FrameCtx;
use shade::Quality;
use stats::Stats;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::time::{Duration, Instant};

const W: usize = 1920;
const H: usize = 1080;
const MAX_SAMPLES: u32 = 1024;
/// Frame budget for dynamic-resolution mode: 60 FPS minus resolve/present
/// headroom. Not a per-tile deadline: a log4-proportional controller turns the
/// previous frame's time against this target into a uniform quadtree depth cap
/// for the next frame; tiles reaching the cap unresolved are sparse-filled
/// (render::sparse_fill). Cost roughly quadruples per level, so
/// log4(budget/elapsed) reads "levels of headroom" directly.
const RENDER_BUDGET: Duration = Duration::from_millis(15);
/// Controller gain on the log4 error.
const DEPTH_GAIN: f32 = 0.6;
/// Max upward step per frame — creep up (>= 3 frames per level)...
const STEP_UP_MAX: f32 = 0.4;
/// ...but drop more than a full level in one step after a blown frame.
const STEP_DOWN_MAX: f32 = 1.5;

/// CLI options beyond the OBJ path / --check.
pub struct Opts {
    /// Want DLSS (default on; auto-falls back when unsupported).
    pub dlss: bool,
    /// D3D12 debug layer + Streamline verbose logging.
    pub gpu_debug: bool,
    /// Directory holding sl.interposer.dll + plugins (M3+).
    pub sl_path: String,
    /// Start with OIDN denoising on (N toggles at runtime; default off —
    /// DLSS-RR stays the primary denoiser).
    pub oidn: bool,
    /// Directory holding OpenImageDenoise.dll + its core/device DLLs.
    pub oidn_path: String,
    /// OIDN device type (oidn.h OIDNDeviceType; 0 = auto-pick fastest).
    pub oidn_device: i32,
    /// OIDN RT-filter quality (`oidn::QUALITY_*`; default balanced — HIGH is
    /// documented for final frames, the flag lets stills opt in).
    pub oidn_quality: i32,
    /// Declare the OIDN albedo/normal guides noise-free (default on — they
    /// are deterministic primary-hit values; --oidn-no-clean-aux is the
    /// empirical escape hatch, same policy as the sign/flag constants).
    pub oidn_clean_aux: bool,
    /// OIDN temporal reprojection history (M toggles at runtime; default on —
    /// off means the plain accumulation-average mode that shimmers while
    /// moving).
    pub oidn_temporal: bool,
    /// Start with XeSS-SR dynamic-resolution upscaling (X toggles; implies
    /// DLSS off — the XeSS context lives on the native, non-SL pipeline).
    pub xess: bool,
    /// Directory holding libxess.dll.
    pub xess_path: String,
    /// Start XeSS mode with the OIDN denoise placed AFTER the upscale
    /// (requires --xess; N cycles placement at runtime).
    pub oidn_post: bool,
    /// XeSS internal autoexposure (XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE;
    /// default off — A/B lever, init-time only).
    pub xess_autoexposure: bool,
    /// Adaptive shading rate on XeSS frames (default on; --no-adaptive forces
    /// uniform per-pixel shading — visibility is per-pixel either way, only
    /// the 2×2-cell shadow/AO sharing and HOT top-ups are disabled).
    pub adaptive: bool,
}

/// OIDN placement in XeSS mode (N cycles off → pre → post). Pre denoises the
/// 1-spp frame at the dynamic render res before upscaling — the recommended
/// default (guides match, subpixel jitter detail preserved, cheaper). Post
/// denoises the upscaled window-res frame — the A/B experiment; costs a
/// synchronous GPU readback plus a window-res denoise every frame and
/// presents through the CPU blit path.
#[derive(Clone, Copy, PartialEq)]
enum XessOidn {
    Off,
    Pre,
    Post,
}

/// The presentation/denoise mode a frame renders for, resolved ONCE per frame
/// from the toggle flags (precedence dlss > xess > oidn > plain). Every
/// mode-dependent FrameCtx field reads this single value — the per-field
/// if/else chains it replaced each re-encoded the precedence by ordering and
/// could silently disagree when a toggle handler missed a reset.
#[derive(Clone, Copy, PartialEq)]
enum RenderMode {
    Dlss,
    Xess,
    /// Plain-mode OIDN; `temporal` is the reprojected-history sub-mode
    /// (fresh 1-spp frames on a free-running rng index). Both sub-modes
    /// fill the window-res OIDN G-buffers.
    Oidn { temporal: bool },
    Plain,
}

fn main() {
    let mut obj: Option<String> = None;
    let mut check = false;
    let mut opts = Opts {
        dlss: true,
        gpu_debug: false,
        sl_path: std::env::var("FRUSTRACER_SL_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\streamline-sdk\bin\x64").to_string()
        }),
        oidn: false,
        oidn_path: std::env::var("FRUSTRACER_OIDN_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\oidn.x64.windows\bin").to_string()
        }),
        oidn_device: 0,
        oidn_quality: oidn::QUALITY_BALANCED,
        oidn_clean_aux: true,
        oidn_temporal: true,
        xess: false,
        xess_path: std::env::var("FRUSTRACER_XESS_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\XeSS-SDK\bin").to_string()
        }),
        oidn_post: false,
        xess_autoexposure: false,
        adaptive: true,
    };
    let mut check_dlss = false;
    let mut dlss_dump = false;
    let mut check_oidn = false;
    let mut oidn_dump = false;
    let mut check_xess = false;
    let mut xess_dump = false;
    let mut stress: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--check" => check = true,
            "--check-dlss" => check_dlss = true,
            "--dlss-dump" => {
                check_dlss = true;
                dlss_dump = true;
            }
            "--dlss" => opts.dlss = true,
            "--no-dlss" => opts.dlss = false,
            "--check-oidn" => check_oidn = true,
            "--oidn-dump" => {
                check_oidn = true;
                oidn_dump = true;
            }
            "--oidn" => opts.oidn = true,
            "--no-oidn" => opts.oidn = false,
            "--oidn-no-temporal" => opts.oidn_temporal = false,
            "--check-xess" => check_xess = true,
            "--xess-dump" => {
                check_xess = true;
                xess_dump = true;
            }
            "--xess" => opts.xess = true,
            "--no-xess" => opts.xess = false,
            "--oidn-post" => opts.oidn_post = true,
            "--xess-autoexposure" => opts.xess_autoexposure = true,
            "--no-adaptive" => opts.adaptive = false,
            "--xess-path" => {
                opts.xess_path = args.next().unwrap_or_else(|| {
                    eprintln!("--xess-path needs a directory argument");
                    std::process::exit(2);
                })
            }
            "--oidn-path" => {
                opts.oidn_path = args.next().unwrap_or_else(|| {
                    eprintln!("--oidn-path needs a directory argument");
                    std::process::exit(2);
                })
            }
            "--oidn-device" => {
                // Names map to oidn.h OIDNDeviceType values.
                opts.oidn_device = match args.next().as_deref() {
                    Some("default") => 0,
                    Some("cpu") => 1,
                    Some("sycl") => 2,
                    Some("cuda") => 3,
                    Some("hip") => 4,
                    _ => {
                        eprintln!("--oidn-device needs one of: default cpu sycl cuda hip");
                        std::process::exit(2);
                    }
                }
            }
            "--oidn-quality" => {
                opts.oidn_quality = match args.next().as_deref() {
                    Some("fast") => oidn::QUALITY_FAST,
                    Some("balanced") => oidn::QUALITY_BALANCED,
                    Some("high") => oidn::QUALITY_HIGH,
                    _ => {
                        eprintln!("--oidn-quality needs one of: fast balanced high");
                        std::process::exit(2);
                    }
                }
            }
            "--oidn-no-clean-aux" => opts.oidn_clean_aux = false,
            "--gpu-debug" => opts.gpu_debug = true,
            "--sl-path" => {
                opts.sl_path = args.next().unwrap_or_else(|| {
                    eprintln!("--sl-path needs a directory argument");
                    std::process::exit(2);
                })
            }
            "--stress" => {
                stress = Some(
                    args.next()
                        .and_then(|s| s.parse::<usize>().ok())
                        .filter(|&n| n > 0)
                        .unwrap_or_else(|| {
                            eprintln!("--stress needs an object count, e.g. --stress 5000");
                            std::process::exit(2);
                        }),
                )
            }
            "--help" | "-h" => {
                eprintln!("usage: frustracer [model.obj] [--stress <n>] [--check] [--check-dlss] [--dlss-dump] [--no-dlss] [--check-oidn] [--oidn-dump] [--oidn] [--check-xess] [--xess-dump] [--xess] [--gpu-debug] [--sl-path <dir>] [--oidn-path <dir>] [--oidn-device <d>] [--xess-path <dir>]");
                eprintln!("  --stress <n>  procedural stress field of n objects (perf test; composes with --check)");
                eprintln!("  --check       headless: verify hybrid vs reference, benchmark, write check.png");
                eprintln!("  --check-dlss  headless: G-buffer MV/depth/matrix self-test (no GPU needed)");
                eprintln!("  --dlss-dump   --check-dlss plus G-buffer PNG dumps (albedo/spec_albedo/normal/misc/mv)");
                eprintln!("  --no-dlss     skip Streamline/DLSS; plain D3D12 presentation");
                eprintln!("  --check-oidn  headless: OIDN denoise self-test (needs the OIDN DLLs)");
                eprintln!("  --oidn-dump   --check-oidn plus before/after/G-buffer PNG dumps");
                eprintln!("  --oidn        start with OIDN denoising on (N toggles; implies DLSS off)");
                eprintln!("  --oidn-no-temporal  start OIDN without the temporal reprojection history (M toggles)");
                eprintln!("  --oidn-path   OIDN DLL directory (default: SDKs\\oidn.x64.windows\\bin)");
                eprintln!("  --oidn-device OIDN device: default|cpu|sycl|cuda|hip");
                eprintln!("  --oidn-quality OIDN RT-filter quality: fast|balanced|high (default balanced)");
                eprintln!("  --oidn-no-clean-aux  don't declare the albedo/normal guides noise-free (A/B lever)");
                eprintln!("  --check-xess  headless: XeSS dynamic-res contract self-test (no GPU or DLL needed)");
                eprintln!("  --xess-dump   --check-xess plus G-buffer PNG dumps");
                eprintln!("  --xess        start with XeSS-SR dynamic-res upscaling (X toggles; implies DLSS off;");
                eprintln!("                N cycles the OIDN denoise: off -> pre-upscale -> post-upscale)");
                eprintln!("  --oidn-post   start XeSS mode with OIDN placed AFTER the upscale (requires --xess)");
                eprintln!("  --xess-autoexposure  let XeSS compute exposure internally (A/B lever)");
                eprintln!("  --no-adaptive disable the adaptive shading rate in XeSS mode (uniform per-pixel shading;");
                eprintln!("                visibility is per-pixel either way)");
                eprintln!("  --xess-path   XeSS DLL directory (default: SDKs\\XeSS-SDK\\bin)");
                eprintln!("  --gpu-debug   D3D12 debug layer + verbose Streamline logging");
                eprintln!("  --sl-path     Streamline DLL directory (default: SDKs\\streamline-sdk\\bin\\x64)");
                return;
            }
            _ => obj = Some(a),
        }
    }

    if obj.is_some() && stress.is_some() {
        eprintln!("--stress and an OBJ path are mutually exclusive — pick one scene source");
        std::process::exit(2);
    }
    if opts.xess && opts.dlss {
        // The XeSS context needs the native (non-proxied) device/queue —
        // Streamline's manual-hooking proxies never coexist with it.
        opts.dlss = false;
        eprintln!("xess: --xess implies --no-dlss (native D3D12 pipeline)");
    }

    eprintln!("frustracer — loading scene...");
    let scene = match (&obj, stress) {
        (Some(p), _) => scene::load_obj_scene(p),
        (None, Some(n)) => scene::stress_scene(n),
        (None, None) => scene::procedural_scene(),
    };
    // The stress field keeps the default look direction but pulls the camera
    // back/up to overlook the field; /8 (not the field half-extent itself)
    // trades the nearest rows off the bottom of the frame for less sky.
    let cam0 = match stress {
        Some(n) => scaled_camera((scene::stress_field_half(n) / 8.0).max(1.0)),
        None => default_camera(),
    };
    let t0 = Instant::now();
    let bvh = bvh::Bvh::build(&scene);
    eprintln!(
        "scene: {} tris | BVH: {} nodes built in {:.0} ms",
        scene.tri_count(),
        bvh.nodes.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );

    if check {
        let code = run_check(&scene, &bvh, cam0, stress.is_none());
        std::process::exit(code);
    }
    if check_dlss {
        let code = run_check_dlss(&scene, &bvh, cam0, dlss_dump);
        std::process::exit(code);
    }
    if check_xess {
        // Must-fire structural gates are tuned to the default scene's
        // topology — skipped under --stress, mirroring run_check.
        let code = run_check_xess(&scene, &bvh, cam0, xess_dump, stress.is_none());
        std::process::exit(code);
    }
    if check_oidn {
        #[cfg(windows)]
        {
            // Must-fire structural gates are tuned to the default scene's
            // topology — skipped under --stress, mirroring run_check.
            let code = run_check_oidn(&scene, &bvh, cam0, &opts, oidn_dump, stress.is_none());
            std::process::exit(code);
        }
        #[cfg(not(windows))]
        {
            let _ = oidn_dump;
            eprintln!("--check-oidn requires Windows (the OIDN SDK drop is Win64-only here)");
            std::process::exit(2);
        }
    }
    #[cfg(windows)]
    run_window(&scene, &bvh, &opts, cam0);
    #[cfg(not(windows))]
    {
        let _ = (&opts, cam0);
        eprintln!("the interactive window requires Windows (D3D12 + DLSS); use --check / --check-dlss");
        std::process::exit(2);
    }
}

/// Headless DLSS G-buffer verification: renders two DLSS-style frames (a
/// small forward dolly apart — the same move `--check` T2 uses), then checks
/// motion vectors, depth, and the camera matrices jointly by reconstructing
/// world positions through both frames. No GPU or Streamline involved — this
/// validates the CPU capture before/without the denoiser.
fn run_check_dlss(scene: &scene::Scene, bvh: &bvh::Bvh, cam0: Camera, dump: bool) -> i32 {
    // Halton structural checks: known prefixes, offsets in [-0.5, 0.5).
    let mut halton_ok = true;
    for (i, base, want) in [(1u32, 2u32, 0.5f32), (2, 2, 0.25), (3, 2, 0.75), (1, 3, 1.0 / 3.0), (2, 3, 2.0 / 3.0)] {
        let got = dlss::halton(i, base);
        if (got - want).abs() > 1e-6 {
            eprintln!("halton({i},{base}) = {got}, want {want}");
            halton_ok = false;
        }
    }
    for idx in 0..64 {
        let (ox, oy) = dlss::jitter_for(idx);
        if !(-0.5..0.5).contains(&ox) || !(-0.5..0.5).contains(&oy) {
            eprintln!("jitter_for({idx}) = ({ox},{oy}) out of [-0.5,0.5)");
            halton_ok = false;
        }
    }
    eprintln!("halton/jitter checks: {}", if halton_ok { "OK" } else { "FAIL" });

    // Frame A at the given camera; frame B a 0.02·diag forward dolly later,
    // with A as its previous frame. Run once at the native test res and once
    // at the Quality-mode render res stand-in (odd width — also exercises
    // odd-dim quadtree splits), since the interactive DLSS path now traces
    // at a sub-native resolution.
    let mv_native_ok = mv_check_at(scene, bvh, cam0, 800, 600, if dump { Some("dlss_gbuf") } else { None });
    let (qw, qh) = dlss::headless_render_res(800, 600);
    let mv_quality_ok = mv_check_at(scene, bvh, cam0, qw, qh, None);
    // Step-wise DRS: an arbitrary quantized size inside a typical RR range —
    // the varying-res MV/depth/matrix contract (the extent tagging itself is
    // SL-side and validated interactively; headless stays SL-free).
    let (dw, dh) = xess::quantize_res(0.55, (800, 600), (266, 200), (800, 600));
    let mv_drs_ok = mv_check_at(scene, bvh, cam0, dw, dh, None);

    if halton_ok && mv_native_ok && mv_quality_ok && mv_drs_ok {
        eprintln!("DLSS CHECK PASSED");
        0
    } else {
        eprintln!("DLSS CHECK FAILED");
        1
    }
}

/// One MV/depth/matrix pass at an arbitrary render resolution: frame A at
/// `cam0`, frame B a 0.02·diag forward dolly later with A as its previous
/// frame, gated by `dlss::mv_selftest`. Zero jitter so samples sit on pixel
/// centers — the reconstruction in the self-test assumes centers. Shared by
/// --check-dlss (the fixed DLSS-style resolutions) and --check-xess (a sweep
/// of dynamic render resolutions).
fn mv_check_at(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    rw: usize,
    rh: usize,
    dump_prefix: Option<&str>,
) -> bool {
    let (near, far) = dlss::near_far(scene.diag);
    let q = Quality {
        shadow_samples: 1,
        ao_samples: 1,
        reflections: true,
        fb: shade::FrustumBounce::OFF,
    };
    let stats = Stats::default();
    eprintln!("MV/depth/matrix self-test at {rw}x{rh}:");
    let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let basis_a = cam0.basis(rw, rh);
    let ga = dlss::GBufs::new(rw, rh);
    let gb = dlss::GBufs::new(rw, rh);
    let render_dlss_frame =
        |g: &dlss::GBufs, basis: camera::CamBasis, prev: Option<camera::CamBasis>, frame: u32| {
            let ctx = FrameCtx {
                scene,
                bvh,
                cam: basis,
                q,
                frame,
                jitter: false,
                rw,
                rh,
                accum: &accum,
                info: &info,
                tbuf: &tbuf,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: None,
                tcache_prev: None,
                accumulate: false,
                gbuf: Some(g),
                prev_cam: prev,
                frame_jitter: Some((0.0, 0.0)),
                adaptive: false,
            };
            render::render_frame(&ctx, true);
        };
    render_dlss_frame(&ga, basis_a, None, 0);

    let mut cam_b = cam0;
    cam_b.pos += cam0.forward() * (0.02 * scene.diag);
    let basis_b = cam_b.basis(rw, rh);
    render_dlss_frame(&gb, basis_b, Some(basis_a), 1);

    let mats_b = dlss::cam_matrices(&cam_b, rw, rh, near, far);
    let ok = dlss::mv_selftest(&ga, &basis_a, &gb, &basis_b, &mats_b, scene.diag, far);

    if let Some(prefix) = dump_prefix {
        dlss::dump_gbufs(&gb, prefix, far);
    }
    ok
}

/// Headless XeSS verification — DLL- and GPU-free (the pure half of xess.rs
/// plus the same G-buffer machinery --check-dlss gates): proves the
/// dynamic-resolution contract before the SDK ever runs. Gates: the
/// view-Z → clip-depth encoding roundtrips and hits its endpoints exactly
/// (sky = far must land on 1.0), `quantize_res` respects the range clamps /
/// height quantum / window aspect, the scale controller clamps, sheds fast,
/// creeps slowly and respects the deadband on a scripted frame-time
/// sequence, and the MV/depth/matrix self-test passes at a sweep of dynamic
/// render resolutions (quantized 16:9 steps plus an odd-dimension literal).
fn run_check_xess(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    dump: bool,
    structural: bool,
) -> i32 {
    let (near, far) = dlss::near_far(scene.diag);
    let mut all_ok = true;

    // 1. Depth encoding (reversed-Z): monotone decreasing, roundtrips,
    // exact endpoints.
    {
        let mut pass = true;
        let mut prev_d = 2.0f32;
        for i in 0..=256 {
            let z = near + (far - near) * (i as f32 / 256.0);
            let d = xess::view_z_to_clip_depth(z, near, far);
            let z2 = xess::clip_depth_to_view_z(d, near, far);
            if d > prev_d {
                eprintln!("depth encoding not monotone decreasing at z={z}");
                pass = false;
            }
            if (z2 - z).abs() > 1e-3 * z {
                eprintln!("depth roundtrip z={z} -> d={d} -> {z2}");
                pass = false;
            }
            prev_d = d;
        }
        if xess::view_z_to_clip_depth(near, near, far) != 1.0 {
            eprintln!("depth(near) != 1");
            pass = false;
        }
        if xess::view_z_to_clip_depth(far, near, far) != 0.0 {
            eprintln!("depth(far) != 0 (sky sentinel would drift)");
            pass = false;
        }
        eprintln!("depth encoding: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 2. quantize_res: height quantum, window aspect, hard range clamps.
    {
        let mut pass = true;
        let out = (1920usize, 1080usize);
        let min = (640usize, 360usize);
        let max = (1920usize, 1080usize);
        for i in 0..=40 {
            let s = 0.2 + 0.9 * (i as f32 / 40.0); // sweeps below min and above max
            let (rw, rh) = xess::quantize_res(s, out, min, max);
            if rw < min.0 || rh < min.1 || rw > max.0 || rh > max.1 {
                eprintln!("quantize_res({s}) = {rw}x{rh} escapes range");
                pass = false;
            }
            let aspect = rw as f32 / rh as f32;
            let want = out.0 as f32 / out.1 as f32;
            if (aspect - want).abs() > 0.02 {
                eprintln!("quantize_res({s}) = {rw}x{rh} aspect {aspect} vs {want}");
                pass = false;
            }
            if rh % xess::RES_STEP != 0 && rh != min.1 && rh != max.1 {
                eprintln!("quantize_res({s}) height {rh} off the quantum");
                pass = false;
            }
        }
        eprintln!("quantize_res: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 3. Scale controller on a scripted frame-time sequence (no wall clock).
    {
        let mut pass = true;
        let budget = 15.0f32;
        let mut ctl = xess::ScaleCtl::new(720, 360, 1080, 1080);
        let s0 = ctl.scale();
        ctl.update(6.0 * budget, budget); // blown frame
        if ctl.scale() >= s0 {
            eprintln!("controller did not shed after a blown frame");
            pass = false;
        }
        let shed = ctl.scale();
        ctl.update(0.2 * budget, budget); // one cheap frame
        if ctl.scale() > shed * 1.05 {
            eprintln!("controller climbed more than the slow-up bound in one frame");
            pass = false;
        }
        for _ in 0..2000 {
            ctl.update(0.2 * budget, budget); // sustained cheap: creep to the top
        }
        if (ctl.scale() - 1.0).abs() > 1e-3 {
            eprintln!("controller failed to reach the max clamp ({})", ctl.scale());
            pass = false;
        }
        for _ in 0..100 {
            ctl.update(100.0 * budget, budget); // sustained blown: floor clamp
        }
        if (ctl.scale() - 360.0 / 1080.0).abs() > 1e-3 {
            eprintln!("controller failed to hold the min clamp ({})", ctl.scale());
            pass = false;
        }
        let mut parked = xess::ScaleCtl::new(720, 360, 1080, 1080);
        let sp = parked.scale();
        for _ in 0..50 {
            parked.update(0.75 * budget, budget); // >60% of budget: deadband
        }
        if parked.scale() != sp {
            eprintln!("controller climbed inside the deadband");
            pass = false;
        }
        // Step limiter: first target adopts; the dwell holds against both
        // growth and non-emergency sheds; an emergency may bypass only to
        // SHED; dwell expiry adopts the current target in one jump.
        let mut lim = xess::StepLimiter::new();
        if lim.apply((1280, 720), false) != (1280, 720) {
            eprintln!("step limiter: first apply did not adopt");
            pass = false;
        }
        if lim.apply((1216, 684), false) != (1280, 720) {
            eprintln!("step limiter: dwell did not hold a shed");
            pass = false;
        }
        if lim.apply((1408, 792), true) != (1280, 720) {
            eprintln!("step limiter: emergency bypassed for GROWTH");
            pass = false;
        }
        if lim.apply((960, 540), true) != (960, 540) {
            eprintln!("step limiter: emergency shed did not bypass");
            pass = false;
        }
        let mut adopted = (0, 0);
        for _ in 0..=xess::STEP_DWELL {
            adopted = lim.apply((1152, 648), false);
        }
        if adopted != (1152, 648) {
            eprintln!("step limiter: dwell expiry did not adopt");
            pass = false;
        }
        eprintln!("scale controller + step limiter: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 3b. Guide nearest-upscale (the post-OIDN placement feeds window-res
    // OIDN with albedo/normal guides pulled from the render-res G-buffers):
    // every destination texel of the three copied planes must bit-equal its
    // nearest source texel — indexing/stride/plane-mix-up guard. Checked at
    // identity, integer, and non-integer ratios.
    {
        let mut pass = true;
        let (sw, sh) = (64usize, 36usize);
        let src = dlss::GBufs::new(sw, sh);
        for i in 0..sw * sh {
            for k in 0..3 {
                src.diff_alb[i * 3 + k].store((i * 3 + k) as u32, Relaxed);
            }
            for k in 0..3 {
                src.spec_alb[i * 3 + k].store(0x5000_0000 + (i * 3 + k) as u32, Relaxed);
            }
            for k in 0..4 {
                src.normal_rough[i * 4 + k].store(0x6000_0000 + (i * 4 + k) as u32, Relaxed);
            }
        }
        let check = |dst: &dlss::GBufs| -> bool {
            let (dw, dh) = (dst.rw, dst.rh);
            for y in 0..dh {
                let sy = (y * sh / dh).min(sh - 1);
                for x in 0..dw {
                    let sx = (x * sw / dw).min(sw - 1);
                    let (si, di) = (sy * sw + sx, y * dw + x);
                    for k in 0..3 {
                        if dst.diff_alb[di * 3 + k].load(Relaxed)
                            != src.diff_alb[si * 3 + k].load(Relaxed)
                        {
                            return false;
                        }
                    }
                    for k in 0..3 {
                        if dst.spec_alb[di * 3 + k].load(Relaxed)
                            != src.spec_alb[si * 3 + k].load(Relaxed)
                        {
                            return false;
                        }
                    }
                    for k in 0..4 {
                        if dst.normal_rough[di * 4 + k].load(Relaxed)
                            != src.normal_rough[si * 4 + k].load(Relaxed)
                        {
                            return false;
                        }
                    }
                }
            }
            true
        };
        for (dw, dh) in [(sw, sh), (2 * sw, 2 * sh), (100, 54)] {
            let dst = dlss::GBufs::new(dw, dh);
            dst.upscale_guides_from(&src);
            if !check(&dst) {
                eprintln!("guide upscale {sw}x{sh} -> {dw}x{dh}: texel mismatch");
                pass = false;
            }
        }
        eprintln!("guide nearest-upscale: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 4. Adaptive shading rate: the same frame twice (identical seed, res,
    // frame-uniform jitter), BASE vs ADAPTIVE. Visibility must be
    // bit-identical — adaptivity may never touch it (the transplanted
    // bug-class gate); radiance may differ only by the shared-visibility
    // approximation in coherent cells; the counters must fire (default
    // scene) and account for every leaf-shaded pixel.
    {
        let (rw, rh) = (768usize, 432usize);
        // shadow_samples = 2 so the uniformity test has teeth: the penumbra
        // self-declassifier can only fire with >= 2 light samples (a single
        // sample is trivially uniform). The interactive XeSS preset is 1/1 —
        // there, penumbra correlation at cell scale is laundered temporally.
        let q = Quality {
            shadow_samples: 2,
            ao_samples: 2,
            reflections: true,
            fb: shade::FrustumBounce::OFF,
        };
        // Accumulate several jittered frames per side: two single 1-spp
        // frames differ only by the shared-visibility approximation (Apply
        // pixels reuse the rep's occlusion; the rng stream stays aligned via
        // burned draws), which is noise, not bias — the averages expose the
        // actual approximation error, and the SIGNED mean is the bias
        // detector (noise cancels, systematic error doesn't). Rep rotation
        // across frames is part of the contract under test.
        const AB_FRAMES: u32 = 16;
        let render_avg = |adaptive: bool, stats: &Stats| -> (Vec<f32>, Vec<u32>, dlss::GBufs) {
            let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
            let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
            let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
            let g = dlss::GBufs::new(rw, rh);
            for f in 0..AB_FRAMES {
                let ctx = FrameCtx {
                    scene,
                    bvh,
                    cam: cam0.basis(rw, rh),
                    q,
                    frame: f,
                    jitter: false,
                    rw,
                    rh,
                    accum: &accum,
                    info: &info,
                    tbuf: &tbuf,
                    stats,
                    sun: render::sun_dir(scene),
                    tcache_cur: None,
                    tcache_prev: None,
                    accumulate: true,
                    gbuf: Some(&g),
                    prev_cam: None,
                    frame_jitter: Some(dlss::jitter_for(f)),
                    adaptive,
                };
                render::render_frame(&ctx, true);
            }
            let inv = 1.0 / AB_FRAMES as f32;
            (
                accum.iter().map(|a| f32::from_bits(a.load(Relaxed)) * inv).collect(),
                tbuf.iter().map(|a| a.load(Relaxed)).collect(),
                g,
            )
        };
        let stats_b = Stats::default();
        let (col_b, t_b, g_b) = render_avg(false, &stats_b);
        let stats_a = Stats::default();
        let (col_a, t_a, g_a) = render_avg(true, &stats_a);

        let mism = t_a.iter().zip(&t_b).filter(|(a, b)| a != b).count();
        let vis_ok = mism == 0;
        eprintln!(
            "adaptive visibility bit-identity: {} px differ -> {}",
            mism,
            if vis_ok { "OK" } else { "FAIL (adaptivity touched visibility)" }
        );
        all_ok &= vis_ok;

        // Every G-buffer guide plane must be bit-identical too — including
        // spec_hit_t, the only rng-dependent plane (Apply burns the draws it
        // skips precisely so the GGX reflection sample stays aligned). This
        // holds per frame; the last frame's buffers are what's compared.
        let planes: [(&str, &[AtomicU32], &[AtomicU32]); 6] = [
            ("normal_rough", &g_a.normal_rough, &g_b.normal_rough),
            ("diff_alb", &g_a.diff_alb, &g_b.diff_alb),
            ("spec_alb", &g_a.spec_alb, &g_b.spec_alb),
            ("depth", &g_a.depth, &g_b.depth),
            ("mvec", &g_a.mvec, &g_b.mvec),
            ("spec_hit_t", &g_a.spec_hit_t, &g_b.spec_hit_t),
        ];
        for (name, pa, pb) in planes {
            let gm = pa
                .iter()
                .zip(pb)
                .filter(|(a, b)| a.load(Relaxed) != b.load(Relaxed))
                .count();
            if gm != 0 {
                eprintln!("adaptive G-buffer bit-identity: {name}: {gm} texels differ -> FAIL");
                all_ok = false;
            }
        }
        eprintln!("adaptive G-buffer bit-identity (6 planes): checked");

        let lum =
            |c: &[f32], i: usize| 0.2126 * c[i * 3] + 0.7152 * c[i * 3 + 1] + 0.0722 * c[i * 3 + 2];
        let mut dsum = 0.0f64;
        let mut ssum = 0.0f64;
        let mut bsum = 0.0f64;
        for i in 0..rw * rh {
            let d = (lum(&col_a, i) - lum(&col_b, i)) as f64;
            dsum += d.abs();
            ssum += d;
            bsum += lum(&col_b, i).abs() as f64;
        }
        let rel = dsum / bsum.max(1e-9);
        let signed = ssum / bsum.max(1e-9);
        // |Δ| bounds residual noise + local approximation; the signed mean is
        // the bias gate (shared visibility/AO must not brighten or darken).
        let rad_ok = rel < 0.02 && signed.abs() < 0.005;
        eprintln!(
            "adaptive radiance A/B ({AB_FRAMES} frames): mean |Δ| {rel:.4} (limit 0.02) | signed {signed:+.4} (limit ±0.005) -> {}",
            if rad_ok { "OK" } else { "FAIL" }
        );
        all_ok &= rad_ok;

        let ld = |c: &std::sync::atomic::AtomicU64| c.load(Relaxed);
        let (coarse, base, hot) =
            (ld(&stats_a.adapt_coarse), ld(&stats_a.adapt_base), ld(&stats_a.adapt_hot));
        let partial = ld(&stats_a.adapt_partial_px);
        let topup = ld(&stats_a.adapt_topup);
        let prim = ld(&stats_a.primary_rays);
        let acct_ok = 4 * (coarse + base + hot) + partial + topup == prim;
        eprintln!(
            "adaptive accounting: {coarse}c/{base}b/{hot}h cells + {partial} edge-px + {topup} topup vs {prim} primaries -> {}",
            if acct_ok { "OK" } else { "FAIL" }
        );
        all_ok &= acct_ok;
        if structural {
            let pen = ld(&stats_a.adapt_penumbra);
            let saved = ld(&stats_a.adapt_rays_saved);
            let fired = coarse > 0 && hot > 0 && pen > 0 && saved > 0;
            eprintln!(
                "adaptive structural: coarse {coarse} hot {hot} penumbra {pen} rays-saved {saved} -> {}",
                if fired { "OK" } else { "FAIL (must-fire counters missing)" }
            );
            all_ok &= fired;
        }
        let base_clean =
            ld(&stats_b.adapt_coarse) + ld(&stats_b.adapt_base) + ld(&stats_b.adapt_hot) == 0;
        if !base_clean {
            eprintln!("adaptive counters fired on a non-adaptive frame -> FAIL");
        }
        all_ok &= base_clean;
    }

    // 5. MV/depth/matrix contract at a sweep of dynamic render resolutions:
    // two quantized 16:9 steps the controller would actually pick, plus an
    // odd-dimension literal (any res inside the range is legal).
    let out = (768usize, 432usize);
    let min = (out.0 / 3, out.1 / 3);
    for (rw, rh) in [
        xess::quantize_res(0.5, out, min, out),
        xess::quantize_res(0.8, out, min, out),
        (515, 289),
    ] {
        all_ok &= mv_check_at(scene, bvh, cam0, rw, rh, if dump { Some("xess_gbuf") } else { None });
    }

    if all_ok {
        eprintln!("XESS CHECK PASSED");
        0
    } else {
        eprintln!("XESS CHECK FAILED");
        1
    }
}

/// Headless OIDN verification: accumulate a few jittered hybrid frames with
/// G-buffer capture (the exact interactive OIDN-mode contract), denoise, and
/// gate on structural properties of the output — finite everywhere, actually
/// changed, measurably smoother (mean |Laplacian| of luminance must drop),
/// mean value preserved within 2×. A second denoise after one more
/// accumulated frame proves the commit-once/execute-many filter reuse.
/// Unlike --check/--check-dlss this needs the (license-clean, gitignored)
/// OIDN runtime DLLs on disk.
#[cfg(windows)]
fn run_check_oidn(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    opts: &Opts,
    dump: bool,
    structural: bool,
) -> i32 {
    let (rw, rh) = (800usize, 600usize);
    let mut octx = match oidn::OidnContext::new(
        &opts.oidn_path,
        rw,
        rh,
        opts.oidn_device,
        opts.oidn_quality,
        opts.oidn_clean_aux,
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            eprintln!(
                "OIDN DLLs expected at {} (override with --oidn-path or FRUSTRACER_OIDN_PATH)",
                opts.oidn_path
            );
            return 1;
        }
    };
    eprintln!("oidn: device {}", octx.device_desc);

    let q = Quality::preset(2);
    let stats = Stats::default();
    let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let g = dlss::GBufs::new(rw, rh);
    let basis = cam0.basis(rw, rh);
    let render_accum_frame = |frame: u32| {
        let ctx = FrameCtx {
            scene,
            bvh,
            cam: basis,
            q,
            frame,
            jitter: frame > 0,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: None,
            accumulate: true,
            gbuf: Some(&g),
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
        };
        render::render_frame(&ctx, true);
    };
    const WARM: u32 = 4;
    for f in 0..WARM {
        render_accum_frame(f);
    }

    let inv = 1.0 / WARM as f32;
    let input: Vec<f32> =
        accum.iter().map(|a| f32::from_bits(a.load(Relaxed)) * inv).collect();
    let out = match octx.denoise(&accum, WARM, &g) {
        Ok(o) => o.to_vec(),
        Err(e) => {
            eprintln!("{e}");
            eprintln!("OIDN CHECK FAILED");
            return 1;
        }
    };
    eprintln!("oidn: denoised {rw}x{rh} in {:.1} ms", octx.last_ms);

    let mut ok = true;
    if !out.iter().all(|v| v.is_finite()) {
        eprintln!("oidn: output contains non-finite values");
        ok = false;
    }
    let max_diff =
        input.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    if max_diff <= 1e-6 {
        eprintln!("oidn: output identical to input (max diff {max_diff:.2e}) — filter did nothing");
        ok = false;
    }

    // Noise metric on interior pixels: denoising must strictly reduce the
    // mean |Laplacian| of luminance while roughly preserving the mean value.
    let lum = |b: &[f32], x: usize, y: usize| {
        let i = (y * rw + x) * 3;
        0.2126 * b[i] + 0.7152 * b[i + 1] + 0.0722 * b[i + 2]
    };
    let roughness = |b: &[f32]| -> f64 {
        let mut s = 0.0f64;
        for y in 1..rh - 1 {
            for x in 1..rw - 1 {
                let l = 4.0 * lum(b, x, y)
                    - lum(b, x - 1, y)
                    - lum(b, x + 1, y)
                    - lum(b, x, y - 1)
                    - lum(b, x, y + 1);
                s += l.abs() as f64;
            }
        }
        s / ((rw - 2) * (rh - 2)) as f64
    };
    let (r_in, r_out) = (roughness(&input), roughness(&out));
    let mean = |b: &[f32]| b.iter().map(|&v| v as f64).sum::<f64>() / b.len() as f64;
    let ratio = mean(&out) / mean(&input).max(1e-9);
    eprintln!("oidn: mean |laplacian| {r_in:.4} -> {r_out:.4} | mean value ratio {ratio:.3}");
    if r_out >= r_in {
        eprintln!("oidn: output not smoother than input");
        ok = false;
    }
    if !(0.5..=2.0).contains(&ratio) {
        eprintln!("oidn: mean value ratio {ratio:.3} outside [0.5, 2.0]");
        ok = false;
    }

    // One more accumulated frame + re-denoise: the per-frame reuse contract
    // (images bound and filter committed once; only buffer contents change).
    render_accum_frame(WARM);
    match octx.denoise(&accum, WARM + 1, &g) {
        Ok(o) => {
            if !o.iter().all(|v| v.is_finite()) {
                eprintln!("oidn: second denoise produced non-finite values");
                ok = false;
            }
            eprintln!("oidn: second denoise (filter reuse) {:.1} ms", octx.last_ms);
        }
        Err(e) => {
            eprintln!("{e}");
            ok = false;
        }
    }

    // ---- Temporal reprojection gates (G0-G5): scene-level validation of
    // reproject::History against the exact interactive temporal contract
    // (fresh 1-spp frames: accumulate=false, jitter=true, free-running seq).
    // Deterministic — the per-pixel RNG is seeded from (x, y, frame) only.
    // Camera moves reuse the --check constants (0.02·diag dolly, +0.05 yaw,
    // cap d=4). Pure CPU math on the shared buffers; the OIDN DLLs above are
    // the only external dependency of this check.
    let (_, far) = dlss::near_far(scene.diag);
    // G0: the closed-form self-test also runs here so --check-oidn is
    // self-contained (it is the check that owns reproject.rs).
    match reproject::self_test() {
        Ok(()) => eprintln!("reproject self-test: OK"),
        Err(e) => {
            eprintln!("reproject self-test: FAIL — {e}");
            ok = false;
        }
    }
    // G2 thresholds, tuned on the default scene (structural-gated):
    // forward dolly keeps the screen edges on-screen, so rejections are
    // silhouette disocclusions + sky/geom transitions — small but nonzero.
    const REJ_FRAC_MAX: f64 = 0.10;
    // World-point agreement is pure geometry (shading-noise-immune): the
    // point the previous frame stored at the fetched texel, projected back
    // into the CURRENT screen, must land within ~a texel of the consuming
    // pixel. Screen-space pixels make the gate scale-invariant — a 3D
    // distance gate has an error floor of one texel's world footprint, which
    // grows with depth/grazing angle and broke on the stress scene. The
    // error floor by construction: fresh frames jitter per-pixel, so both
    // depths belong to random points inside their pixels (~0.7 px each) plus
    // 0.5 px nearest-tap rounding — median ~1 px is geometry agreeing; a
    // real reprojection defect (sign, basis) measures tens to hundreds.
    const WP_MEDIAN_PX: f32 = 1.5;
    const WP_P90_PX: f32 = 4.0;
    // An 8-deep history cuts 1-spp luminance error ~3x vs a converged
    // reference; 0.7 leaves margin for bilinear blur and view-dependence.
    const CONV_GAIN_MAX: f64 = 0.7;
    const CONV_ABS_MAX: f64 = 0.2;

    let render_fresh = |basis: &camera::CamBasis, seq: u32, cap: Option<u32>| {
        let ctx = FrameCtx {
            scene,
            bvh,
            cam: *basis,
            q,
            frame: seq,
            jitter: true,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: None,
            accumulate: false,
            gbuf: Some(&g),
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
        };
        match cap {
            Some(d) => render::render_frame_capped(&ctx, d),
            None => render::render_frame(&ctx, true),
        }
    };
    let load = |a: &AtomicU32| f32::from_bits(a.load(Relaxed));
    // G5: resets are the only writers of L == 1 (coarse_kept may legitimately
    // carry an inherited L of 1, hence the upper slack).
    let l_accounting = |hist: &reproject::History, st: &reproject::UpdateStats, what: &str| {
        let ones = hist.sample_counts().iter().filter(|&&l| l == 1.0).count() as u64;
        let lo = st.rejected + st.coarse_reset;
        let hi = lo + st.coarse_kept;
        if ones < lo || ones > hi {
            eprintln!("reproject {what}: #(L==1) = {ones} outside [{lo}, {hi}]");
            return false;
        }
        true
    };
    let finite = |hist: &reproject::History, what: &str| {
        if hist.color().iter().all(|v| v.is_finite()) {
            true
        } else {
            eprintln!("reproject {what}: history contains non-finite values");
            false
        }
    };

    // G1: static replay. Fresh frames at a fixed basis take the identity
    // path and the history must be the exact running mean (fp tolerance).
    let mut hist = reproject::History::new(rw, rh);
    let mut sum = vec![0f64; rw * rh * 3];
    const T_WARM: u32 = 8;
    for f in 0..T_WARM {
        render_fresh(&basis, f, None);
        for (s, a) in sum.iter_mut().zip(accum.iter()) {
            *s += load(a) as f64;
        }
        let st = hist.update(&basis, &accum, &g, &info, far, MAX_SAMPLES as f32);
        let good = if f == 0 {
            st.rejected == (rw * rh) as u64 && !st.identity
        } else {
            st.identity && st.rejected == 0 && st.coarse_kept + st.coarse_reset == 0
        };
        if !good {
            eprintln!(
                "reproject static f{f}: identity {} rejected {} coarse {}/{}",
                st.identity, st.rejected, st.coarse_kept, st.coarse_reset
            );
            ok = false;
        }
        if f == 3 {
            // Snapshot gates at frame 4 (the plan's G1 point): L uniform,
            // history == sum/4 within fp-rounding of a running mean.
            if st.len_min != 4.0 || st.len_max != 4.0 {
                eprintln!("reproject static: L range [{}, {}], want [4, 4]", st.len_min, st.len_max);
                ok = false;
            }
            let mut max_rel = 0f64;
            for (i, &v) in hist.color().iter().enumerate() {
                let m = sum[i] / 4.0;
                max_rel = max_rel.max((v as f64 - m).abs() / m.abs().max(1.0));
            }
            if max_rel >= 1e-4 {
                eprintln!("reproject static: max |hist - mean| = {max_rel:.2e} (limit 1e-4)");
                ok = false;
            }
        }
        ok &= l_accounting(&hist, &st, &format!("static f{f}"));
    }
    ok &= finite(&hist, "static");

    // G2: forward dolly against an 8-deep history.
    let prev_depth: Vec<f32> = g.depth.iter().map(load).collect();
    let mut cam_b = cam0;
    cam_b.pos += cam0.forward() * (0.02 * scene.diag);
    let basis_b = cam_b.basis(rw, rh);
    render_fresh(&basis_b, T_WARM, None);
    let fresh_b: Vec<f32> = accum.iter().map(load).collect();
    let t_upd = Instant::now();
    let st_b = hist.update(&basis_b, &accum, &g, &info, far, MAX_SAMPLES as f32);
    // Print-only perf diagnostic — the one wall-clock read in this check.
    // Never gate on it: everything gated must stay deterministic.
    let upd_ms = t_upd.elapsed().as_secs_f64() * 1000.0;
    // Dump snapshot here (post-dolly, pre-G4): the G4 budget frame legitimately
    // floods the history with flat quads wherever it was disoccluded.
    let hist_dolly: Vec<f32> = if dump { hist.color().to_vec() } else { Vec::new() };
    let depth_b: Vec<f32> = g.depth.iter().map(load).collect();
    let rough_b: Vec<f32> = (0..rw * rh).map(|i| load(&g.normal_rough[i * 4 + 3])).collect();
    let rej_frac = st_b.rejected as f64 / (rw * rh) as f64;
    eprintln!(
        "reproject dolly: accepted {} ({} sky) rejected {} ({:.2}%) | L {:.0}..{:.0} | update {upd_ms:.2} ms",
        st_b.accepted,
        st_b.sky_accepted,
        st_b.rejected,
        rej_frac * 100.0,
        st_b.len_min,
        st_b.len_max
    );
    if rej_frac >= REJ_FRAC_MAX {
        eprintln!("reproject dolly: rejection fraction {rej_frac:.3} >= {REJ_FRAC_MAX}");
        ok = false;
    }
    if structural && st_b.rejected == 0 {
        eprintln!("reproject dolly: expected silhouette disocclusions > 0 — rejection never fired");
        ok = false;
    }
    ok &= l_accounting(&hist, &st_b, "dolly");
    ok &= finite(&hist, "dolly");
    // World-point agreement on a sparse deterministic subset of accepted
    // geometry pixels: the point this frame saw, reprojected, must be the
    // point the previous frame stored at that texel (both reconstructed from
    // depth — immune to shading noise).
    {
        let sky_z = 0.99 * far;
        let mut errs: Vec<f32> = Vec::new();
        for y in 0..rh {
            for x in 0..rw {
                let i = y * rw + x;
                if (x * 7 + y * 13) % 97 != 0 || hist.mask()[i] == 0 || depth_b[i] >= sky_z {
                    continue;
                }
                let dir = basis_b.ray_dir(x as f32 + 0.5, y as f32 + 0.5);
                let p_cur = basis_b.origin + dir * (depth_b[i] / dir.dot(basis_b.forward()));
                let Some((px, py)) = basis.project(p_cur - basis.origin) else { continue };
                let (tx, ty) = (px.round() as i64, py.round() as i64);
                if tx < 0 || ty < 0 || tx >= rw as i64 || ty >= rh as i64 {
                    continue;
                }
                let ti = ty as usize * rw + tx as usize;
                let pz = prev_depth[ti];
                if pz >= sky_z {
                    continue;
                }
                let pdir = basis.ray_dir(tx as f32 + 0.5, ty as f32 + 0.5);
                let p_prev = basis.origin + pdir * (pz / pdir.dot(basis.forward()));
                let Some((qx, qy)) = basis_b.project(p_prev - basis_b.origin) else { continue };
                errs.push(((qx - (x as f32 + 0.5)).powi(2) + (qy - (y as f32 + 0.5)).powi(2)).sqrt());
            }
        }
        errs.sort_by(|a, b| a.total_cmp(b));
        if errs.is_empty() {
            eprintln!("reproject dolly: world-point sample set is empty");
            ok = false;
        } else {
            let med = errs[errs.len() / 2];
            let p90 = errs[errs.len() * 9 / 10];
            eprintln!(
                "reproject dolly world-point agreement ({} samples): median {med:.3} px p90 {p90:.3} px",
                errs.len(),
            );
            if med >= WP_MEDIAN_PX || p90 >= WP_P90_PX {
                eprintln!("reproject dolly: world-point agreement out of bounds");
                ok = false;
            }
        }
    }
    // Convergence-beats-1-spp: on accepted diffuse geometry pixels the
    // history must track a 16-frame converged reference strictly better than
    // the fresh 1-spp frame does (self-normalizing against scene brightness).
    if structural {
        for f in 0..16u32 {
            let ctx = FrameCtx {
                scene,
                bvh,
                cam: basis_b,
                q,
                frame: f,
                jitter: f > 0,
                rw,
                rh,
                accum: &accum,
                info: &info,
                tbuf: &tbuf,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: None,
                tcache_prev: None,
                accumulate: true,
                gbuf: Some(&g),
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
            };
            render::render_frame(&ctx, true);
        }
        let ref16: Vec<f32> = accum.iter().map(|a| load(a) / 16.0).collect();
        let lum3 = |b: &[f32], i: usize| {
            0.2126 * b[i * 3] as f64 + 0.7152 * b[i * 3 + 1] as f64 + 0.0722 * b[i * 3 + 2] as f64
        };
        let sky_z = 0.99 * far;
        let (mut e_hist, mut e_fresh, mut n) = (0f64, 0f64, 0u64);
        for i in 0..rw * rh {
            if hist.mask()[i] == 0 || depth_b[i] >= sky_z || rough_b[i] < 0.5 {
                continue;
            }
            e_hist += (lum3(hist.color(), i) - lum3(&ref16, i)).abs();
            e_fresh += (lum3(&fresh_b, i) - lum3(&ref16, i)).abs();
            n += 1;
        }
        if n == 0 {
            eprintln!("reproject dolly: convergence sample set is empty");
            ok = false;
        } else {
            e_hist /= n as f64;
            e_fresh /= n as f64;
            eprintln!(
                "reproject dolly vs 16-frame reference ({n} diffuse px): hist err {e_hist:.4} | 1-spp err {e_fresh:.4} (gain limit {CONV_GAIN_MAX})"
            );
            if e_hist >= CONV_GAIN_MAX * e_fresh || e_hist >= CONV_ABS_MAX {
                eprintln!("reproject dolly: history does not beat 1-spp against the reference");
                ok = false;
            }
        }
    }

    // G3: pure yaw. Sky reprojects exactly under rotation (direction-only),
    // and the newly-exposed edge band reprojects off the old screen.
    {
        let mut h3 = reproject::History::new(rw, rh);
        for f in 0..2u32 {
            render_fresh(&basis, 100 + f, None);
            h3.update(&basis, &accum, &g, &info, far, MAX_SAMPLES as f32);
        }
        let mut cam_y = cam0;
        cam_y.yaw += 0.05;
        let basis_y = cam_y.basis(rw, rh);
        render_fresh(&basis_y, 102, None);
        let st = h3.update(&basis_y, &accum, &g, &info, far, MAX_SAMPLES as f32);
        eprintln!(
            "reproject yaw: accepted {} ({} sky) rejected {}",
            st.accepted, st.sky_accepted, st.rejected
        );
        if structural && st.sky_accepted == 0 {
            eprintln!("reproject yaw: expected sky reprojection > 0 — the sky path didn't fire");
            ok = false;
        }
        if structural && st.rejected == 0 {
            eprintln!("reproject yaw: expected the exposed edge band to reject");
            ok = false;
        }
        ok &= l_accounting(&h3, &st, "yaw");
        ok &= finite(&h3, "yaw");
    }

    // G4: a depth-capped budget frame while moving — coarse quads over
    // previously-visible geometry must keep the history (blend weight 0),
    // and the moving-frame length cap must hold. Reuses the dolly history
    // (prev state = frame B); same d=4 cap as --check.
    {
        let mut cam_c = cam_b;
        cam_c.pos += cam0.forward() * (0.02 * scene.diag);
        let basis_c = cam_c.basis(rw, rh);
        let smp0 = stats.coarse_samples.load(Relaxed);
        render_fresh(&basis_c, 200, Some(4));
        let smp_c = stats.coarse_samples.load(Relaxed) - smp0;
        let coarse_px = info
            .iter()
            .filter(|i| overlay::info_kind(i.load(Relaxed)) == overlay::KIND_COARSE)
            .count();
        let st = hist.update(&basis_c, &accum, &g, &info, far, MAX_SAMPLES as f32);
        eprintln!(
            "reproject capped d=4: coarse px {} samples {} | kept {} reset {} | L {:.0}..{:.0}",
            coarse_px, smp_c, st.coarse_kept, st.coarse_reset, st.len_min, st.len_max
        );
        if coarse_px == 0 {
            eprintln!("reproject capped: no coarse pixels — the capped path didn't run");
            ok = false;
        }
        // Coarse pixels imply per-cell point samples on any scene: the
        // samples must have gone through the normal (non-coarse) blend path.
        if coarse_px > 0 && (smp_c == 0 || st.accepted + st.rejected == 0) {
            eprintln!(
                "reproject capped: samples {} accepted {} rejected {} — sparse samples didn't blend",
                smp_c, st.accepted, st.rejected
            );
            ok = false;
        }
        if structural && st.coarse_kept == 0 {
            eprintln!("reproject capped: expected coarse-kept > 0 — the mask rule didn't fire");
            ok = false;
        }
        if st.len_max > reproject::L_MAX {
            eprintln!("reproject capped: L max {} exceeds the moving cap {}", st.len_max, reproject::L_MAX);
            ok = false;
        }
        ok &= l_accounting(&hist, &st, "capped");
        ok &= finite(&hist, "capped");
    }

    if dump {
        let mut present = vec![0u32; rw * rh];
        render::resolve_hdr(&input, &info, false, &mut present, rw, rh, rw, rh);
        save_png("oidn_before.png", &present, rw, rh);
        render::resolve_hdr(&out, &info, false, &mut present, rw, rh, rw, rh);
        save_png("oidn_after.png", &present, rw, rh);
        render::resolve_hdr(&hist_dolly, &info, false, &mut present, rw, rh, rw, rh);
        save_png("oidn_hist.png", &present, rw, rh);
        dlss::dump_gbufs(&g, "oidn_gbuf", far);
        eprintln!("wrote oidn_before.png / oidn_after.png / oidn_hist.png / oidn_gbuf_*.png");
    }

    if ok {
        eprintln!("OIDN CHECK PASSED");
        0
    } else {
        eprintln!("OIDN CHECK FAILED");
        1
    }
}

fn default_camera() -> Camera {
    scaled_camera(1.0)
}

/// The default view pushed back along its own look direction: eye and target
/// scale by `k`, so a `k`-times-wider scene gets the same framing.
fn scaled_camera(k: f32) -> Camera {
    Camera::look_at(
        Vec3A::new(11.0, 6.5, 13.0) * k,
        Vec3A::new(0.0, 1.2 * k, 0.0),
        55f32.to_radians(),
    )
}

/// One deterministic probe for the bounce-integrator check sections: a
/// primary-hit surface point (with its shading normal) on the check camera.
struct Probe {
    x: usize,
    y: usize,
    p: Vec3A,
    n: Vec3A,
}

/// The deterministic probe sweep shared by the hemi-AO, hemi-GI, and shaft
/// check sections — every section MUST run on this exact set (the A/B gates
/// compare like-for-like only because integrator and reference share points).
fn collect_probes(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam: &camera::CamBasis,
    rw: usize,
    rh: usize,
) -> Vec<Probe> {
    let mut vis = 0u64;
    let mut probes = Vec::new();
    for y in 0..rh {
        for x in 0..rw {
            if (x * 7 + y * 13) % 397 != 0 {
                continue;
            }
            let dir = cam.ray_dir(x as f32 + 0.5, y as f32 + 0.5);
            let ray = bvh::Ray::new(cam.origin, dir);
            let Some(hit) = bvh.intersect(scene, &ray, 0.0, f32::INFINITY, &mut vis) else {
                continue;
            };
            let (p, n) = shade::surface_point(scene, &ray, &hit);
            probes.push(Probe { x, y, p, n });
        }
    }
    probes
}

/// Per-pixel deterministic RNG seed for the probe sweeps (`s` salts the
/// stratified repeats and separates integrator from reference streams).
fn px_seed(x: usize, y: usize, s: u64) -> u64 {
    (x as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F))
        .wrapping_add(s)
}

/// Headless end-to-end check: correctness counters (must be 0), an A/B
/// benchmark of hybrid vs plain, and a rendered check.png. `structural`
/// additionally gates the scene-topology assertions (coarse pixels at fixed
/// caps, temporal seeds/sky-tiles firing) — they are tuned to the default
/// procedural scene; a `--stress` scene keeps only the scene-agnostic
/// zero-counter invariants.
fn run_check(scene: &scene::Scene, bvh: &bvh::Bvh, cam0: Camera, structural: bool) -> i32 {
    let (rw, rh) = (800usize, 600usize);
    let cam = cam0.basis(rw, rh);
    let q = Quality::preset(2);
    let stats = Stats::default();

    // Spherical-cell math self-test — closed-form identities the hemisphere
    // bounce integrator is built on (Ω/PSA anchors, exact partition,
    // in-cell sampling).
    let sph_ok = match sphcell::self_test() {
        Ok(()) => {
            eprintln!("sphcell self-test: OK");
            true
        }
        Err(e) => {
            eprintln!("sphcell self-test: FAIL — {e}");
            false
        }
    };

    // Reprojection-history math self-test — closed-form gates the OIDN
    // temporal mode is built on (projection roundtrip, static replay = exact
    // running mean, analytic strafe, behind-plane/depth rejection, coarse
    // keep/reset, L-accounting).
    let reproj_ok = match reproject::self_test() {
        Ok(()) => {
            eprintln!("reproject self-test: OK");
            true
        }
        Err(e) => {
            eprintln!("reproject self-test: FAIL — {e}");
            false
        }
    };

    let rep = render::verify(scene, bvh, &cam, q, rw, rh, &stats, None, None);
    eprintln!(
        "verify full-depth ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e}",
        rep.pixels, rep.false_sky, rep.overshoot, rep.hybrid_extra, rep.max_rel_err
    );
    // The capped driver is the uncapped one with an extra depth check (a cap
    // past the leaf depth is bit-identical by construction), so verify it at a
    // cap that actually sparse-fills: every non-coarse pixel — including the
    // per-cell point samples, which are KIND_LEAF and thus inside the gates —
    // must still match the reference exactly, and both coarse pixels and
    // samples must exist (deterministic — no wall clock involved).
    let smp0 = stats.coarse_samples.load(Relaxed);
    let rep_c = render::verify(scene, bvh, &cam, q, rw, rh, &stats, Some(4), None);
    let smp_c = stats.coarse_samples.load(Relaxed) - smp0;
    eprintln!(
        "verify capped d=4 ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e} | coarse px {} samples {}",
        rep_c.pixels, rep_c.false_sky, rep_c.overshoot, rep_c.hybrid_extra, rep_c.max_rel_err, rep_c.coarse, smp_c
    );
    let mut capped_ok = rep_c.ok() && (!structural || rep_c.coarse > 0);
    if structural && rep_c.coarse == 0 {
        eprintln!("verify capped d=4: expected coarse pixels, found none — capped path not exercised");
    }
    // Coarse tiles exist above, so their per-cell samples must too — this is
    // sound on any scene, not just the default topology (no structural gate).
    if rep_c.coarse > 0 && smp_c == 0 {
        eprintln!("verify capped d=4: coarse pixels without point samples — sparse fill didn't fire");
        capped_ok = false;
    }

    // Hemisphere frustum AO: soundness gates (reference rays re-validate
    // every empty-cell claim and leaf-ray tmin on a deterministic probe set —
    // the false-sky / tmin-overshoot analogs) plus an A/B error measurement
    // against high-sample cosine AO at the same surface points. The
    // integrator is unbiased, so the signed mean is a bias detector.
    let probes = collect_probes(scene, bvh, &cam, rw, rh);
    let hemi_ok = {
        const SEEDS: u64 = 8; // hemi AO averaged over stratified frames
        const REF_SAMPLES: u32 = 1024;
        // Per-probe work is independent (per-pixel seeds); results are
        // collected in probe order and folded sequentially, so the sweep is
        // deterministic and parallel.
        let results: Vec<_> = probes
            .par_iter()
            .map(|pr| {
                let mut hv = hemi::VerifyCounters::default();
                let mut ls = stats::LocalStats::default();
                let mut vis = 0u64;
                let (t1, t2) = shade::onb(pr.n);
                let mut ao_h = 0.0;
                for s in 0..SEEDS {
                    let mut rng = fastrand::Rng::with_seed(px_seed(pr.x, pr.y, s));
                    ao_h += hemi::ao(
                        scene,
                        bvh,
                        pr.p,
                        pr.n,
                        t1,
                        t2,
                        q.fb.depth,
                        scene.ao_radius,
                        &mut rng,
                        if s == 0 { Some(&mut hv) } else { None },
                        &mut ls,
                    );
                }
                let ao_h = ao_h / SEEDS as f32;
                // Reference: cosine-sampled AO, the same construction shade()
                // uses, from the same eps-offset point.
                let mut rng = fastrand::Rng::with_seed(px_seed(pr.x, pr.y, 0xA0));
                let mut open = 0u32;
                for _ in 0..REF_SAMPLES {
                    let r1 = rng.f32();
                    let r2 = rng.f32();
                    let d = shade::cosine_dir(pr.n, t1, t2, r1, r2);
                    if !bvh.occluded(scene, &bvh::Ray::new(pr.p, d), 0.0, scene.ao_radius, &mut vis)
                    {
                        open += 1;
                    }
                }
                let ao_r = open as f32 / REF_SAMPLES as f32;
                ((ao_h - ao_r) as f64, hv, ls)
            })
            .collect();
        let mut hv = hemi::VerifyCounters::default();
        let mut ls = stats::LocalStats::default();
        let (mut sum_abs, mut sum_signed, mut worst) = (0f64, 0f64, 0f32);
        for (d, phv, pls) in &results {
            hv.merge(phv);
            ls.merge(pls);
            sum_abs += d.abs();
            sum_signed += *d;
            worst = worst.max(d.abs() as f32);
        }
        let nprobes = results.len() as u64;
        let mean_abs = sum_abs / nprobes.max(1) as f64;
        let mean_signed = sum_signed / nprobes.max(1) as f64;
        eprintln!(
            "hemi AO ({nprobes} probes, depth {}): psa-viol {} | false-empty {} | tmin-overshoot {} | cut-miss {} | max psa err {:.2e}",
            q.fb.depth, hv.psa_violations, hv.false_empty, hv.tmin_overshoot, hv.cut_miss, hv.max_psa_err
        );
        eprintln!(
            "hemi AO vs {REF_SAMPLES}-sample cosine: mean |Δ| {mean_abs:.4} (limit 0.02) | mean Δ {mean_signed:+.4} (limit ±0.005) | worst {worst:.3} | per point: {:.1} queries, {:.1} rays, {:.1} cells empty",
            ls.hemi_queries as f64 / ls.hemi_points.max(1) as f64,
            ls.hemi_leaf_rays as f64 / ls.hemi_points.max(1) as f64,
            ls.hemi_cells_empty as f64 / ls.hemi_points.max(1) as f64,
        );
        let mut ok = hv.ok() && mean_abs < 0.02 && mean_signed.abs() < 0.005;
        if structural && (ls.hemi_cells_empty == 0 || ls.hemi_leaf_rays == 0) {
            eprintln!("hemi AO: expected both empty cells and leaf rays > 0 — a path didn't fire");
            ok = false;
        }
        ok
    };

    // Hemisphere frustum GI: the same soundness gates (with t_limit = ∞, so
    // empty-cell claims are true sky claims) plus an A/B against a
    // cosine-sampled reference implementing the SAME depth-1 bounce policy
    // (hemi::BOUNCE_Q) — the comparison isolates integrator error from policy
    // error. Luminance-relative gate; the signed mean detects bias.
    let gi_ok = {
        let sun = render::sun_dir(scene);
        const SEEDS: u64 = 8;
        const REF_SAMPLES: u32 = 512;
        let lum = |c: Vec3A| c.dot(Vec3A::new(0.2126, 0.7152, 0.0722));
        // Same probe set and parallel/sequential-fold structure as hemi AO.
        let results: Vec<_> = probes
            .par_iter()
            .map(|pr| {
                let mut hv = hemi::VerifyCounters::default();
                let mut ls = stats::LocalStats::default();
                let mut vis = 0u64;
                let (t1, t2) = shade::onb(pr.n);
                let mut e_h = Vec3A::ZERO;
                for s in 0..SEEDS {
                    let mut rng = fastrand::Rng::with_seed(px_seed(pr.x, pr.y, s));
                    e_h += hemi::gi(
                        scene,
                        bvh,
                        pr.p,
                        pr.n,
                        t1,
                        t2,
                        q.fb.depth,
                        sun,
                        0,
                        &mut rng,
                        if s == 0 { Some(&mut hv) } else { None },
                        &mut ls,
                    );
                }
                let e_h = e_h / SEEDS as f32;
                // Reference: cosine-sampled hemisphere, identical depth-1
                // policy (miss → sky, hit → shade at BOUNCE_Q, depth 1).
                let mut rng = fastrand::Rng::with_seed(px_seed(pr.x, pr.y, 0xE0));
                let mut lsr = stats::LocalStats::default();
                let mut e_r = Vec3A::ZERO;
                for _ in 0..REF_SAMPLES {
                    let r1 = rng.f32();
                    let r2 = rng.f32();
                    let d = shade::cosine_dir(pr.n, t1, t2, r1, r2);
                    let bray = bvh::Ray::new(pr.p, d);
                    e_r += match bvh.intersect(scene, &bray, 0.0, f32::INFINITY, &mut vis) {
                        None => shade::sky(d, sun),
                        Some(h) => shade::shade(
                            scene,
                            bvh,
                            &bray,
                            &h,
                            None,
                            &hemi::BOUNCE_Q,
                            &mut rng,
                            sun,
                            1,
                            &mut lsr,
                            None,
                            shade::VisCtl::Off,
                        ),
                    };
                }
                let e_r = e_r / REF_SAMPLES as f32;
                let rel = ((lum(e_h) - lum(e_r)) / lum(e_r).max(0.05)) as f64;
                (rel, hv, ls)
            })
            .collect();
        let mut hv = hemi::VerifyCounters::default();
        let mut ls = stats::LocalStats::default();
        let (mut sum_rel, mut sum_signed, mut worst) = (0f64, 0f64, 0f32);
        for (rel, phv, pls) in &results {
            hv.merge(phv);
            ls.merge(pls);
            sum_rel += rel.abs();
            sum_signed += *rel;
            worst = worst.max(rel.abs() as f32);
        }
        let nprobes = results.len() as u64;
        let mean_rel = sum_rel / nprobes.max(1) as f64;
        let mean_signed = sum_signed / nprobes.max(1) as f64;
        eprintln!(
            "hemi GI ({nprobes} probes, depth {}): psa-viol {} | false-empty {} | tmin-overshoot {} | cut-miss {} | max psa err {:.2e}",
            q.fb.depth, hv.psa_violations, hv.false_empty, hv.tmin_overshoot, hv.cut_miss, hv.max_psa_err
        );
        eprintln!(
            "hemi GI vs {REF_SAMPLES}-sample cosine (same depth-1 policy): mean rel {mean_rel:.4} (limit 0.05) | signed {mean_signed:+.4} (limit ±0.01) | worst {worst:.3} | per point: {:.1} queries, {:.1} rays, {:.1} cells empty",
            ls.hemi_queries as f64 / ls.hemi_points.max(1) as f64,
            ls.hemi_leaf_rays as f64 / ls.hemi_points.max(1) as f64,
            ls.hemi_cells_empty as f64 / ls.hemi_points.max(1) as f64,
        );
        let mut ok = hv.ok() && mean_rel < 0.05 && mean_signed.abs() < 0.01;
        if structural && (ls.hemi_cells_empty == 0 || ls.hemi_leaf_rays == 0) {
            eprintln!("hemi GI: expected both empty cells and leaf rays > 0 — a path didn't fire");
            ok = false;
        }
        ok
    };

    // Light-shaft shadow culling: proven-lit subrect claims re-validated by
    // reference occlusion rays (corners + center of each lit leaf), sample
    // classification re-checked against full-tree occlusion (exact-match —
    // the shaft must never change a shadow result, only skip proven rays),
    // and the leaf rects must exactly tile the light's param square.
    let shaft_ok = {
        // Same probe set and parallel/sequential-fold structure as hemi AO.
        let results: Vec<_> = probes
            .par_iter()
            .map(|pr| {
                let mut ls = stats::LocalStats::default();
                let mut vis = 0u64;
                let (mut false_lit, mut mismatch, mut area_bad) = (0u64, 0u64, 0u64);
                let (mut samples, mut skipped) = (0u64, 0u64);
                let (p, n) = (pr.p, pr.n);
                let s = shaft::build(scene, bvh, p, n, &mut ls);
                let mut area = 0.0f32;
                for l in s.leaves() {
                    area += (l.r[2] - l.r[0]) * (l.r[3] - l.r[1]);
                    if l.lit {
                        // Corners + center of the lit subrect must be reachable.
                        let pts = [
                            (l.r[0], l.r[1]),
                            (l.r[2], l.r[1]),
                            (l.r[2], l.r[3]),
                            (l.r[0], l.r[3]),
                            ((l.r[0] + l.r[2]) * 0.5, (l.r[1] + l.r[3]) * 0.5),
                        ];
                        for (su, sv) in pts {
                            let lp = scene.light.center + scene.light.u * su + scene.light.v * sv;
                            let lv = lp - p;
                            let dist = lv.length();
                            let wi = lv / dist;
                            // The shaft claim covers the tangent upper
                            // half-space only — exactly the samples shade
                            // ever consults it for (ndl > 0).
                            if wi.dot(n) <= 0.0 {
                                continue;
                            }
                            if bvh.occluded(
                                scene,
                                &bvh::Ray::new(p, wi),
                                0.0,
                                dist - scene.eps,
                                &mut vis,
                            ) {
                                false_lit += 1;
                            }
                        }
                    }
                }
                if (area - 4.0).abs() > 1e-4 {
                    area_bad += 1;
                }
                // Deterministic sample sweep: the shaft-classified occlusion
                // result must equal the full-tree result for every sample.
                let mut rng = fastrand::Rng::with_seed(
                    (pr.x as u64).wrapping_mul(31).wrapping_add(pr.y as u64),
                );
                for _ in 0..16 {
                    let (su, sv) = (rng.f32() * 2.0 - 1.0, rng.f32() * 2.0 - 1.0);
                    let lp = scene.light.center + scene.light.u * su + scene.light.v * sv;
                    let lv = lp - p;
                    let dist = lv.length();
                    let wi = lv / dist;
                    if wi.dot(n) <= 0.0 {
                        continue; // shade never consults the shaft below the horizon
                    }
                    let full = bvh.occluded(
                        scene,
                        &bvh::Ray::new(p, wi),
                        0.0,
                        dist - scene.eps,
                        &mut vis,
                    );
                    samples += 1;
                    let got = match s.classify(su, sv) {
                        shaft::Class::Lit => {
                            skipped += 1;
                            false
                        }
                        shaft::Class::Test { tmin, cut } => bvh.occluded_multi(
                            scene,
                            &bvh::Ray::new(p, wi),
                            tmin,
                            dist - scene.eps,
                            cut,
                            &mut vis,
                        ),
                    };
                    if got != full {
                        mismatch += 1;
                    }
                }
                (false_lit, mismatch, area_bad, samples, skipped, ls)
            })
            .collect();
        let mut ls = stats::LocalStats::default();
        let (mut false_lit, mut mismatch, mut area_bad) = (0u64, 0u64, 0u64);
        let (mut samples, mut skipped) = (0u64, 0u64);
        for (fl, mm, ab, sa, sk, pls) in &results {
            false_lit += fl;
            mismatch += mm;
            area_bad += ab;
            samples += sa;
            skipped += sk;
            ls.merge(pls);
        }
        let nprobes = results.len() as u64;
        eprintln!(
            "shaft shadows ({nprobes} probes): false-lit {false_lit} | class mismatch {mismatch} | area-bad {area_bad} | rays skipped {skipped}/{samples} ({:.0}%) | {:.1} queries/point",
            skipped as f64 * 100.0 / samples.max(1) as f64,
            ls.shaft_queries as f64 / nprobes.max(1) as f64,
        );
        let mut ok = false_lit == 0 && mismatch == 0 && area_bad == 0;
        if structural && skipped == 0 {
            eprintln!("shaft shadows: expected skipped rays > 0 — the lit path didn't fire");
            ok = false;
        }
        ok
    };

    let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();

    const BENCH_FRAMES: u32 = 8;
    for (label, hybrid, hemi_ao, hemi_gi, shafts) in [
        ("hybrid ", true, false, false, false),
        ("hemi-ao", true, true, false, false),
        ("hemi-gi", true, false, true, false),
        ("shafts ", true, false, false, true),
        ("plain  ", false, false, false, false),
    ] {
        stats.clear();
        let mut bq = q;
        bq.fb.ao = hemi_ao;
        bq.fb.gi = hemi_gi;
        bq.fb.shadows = shafts;
        let ctx = FrameCtx {
            scene,
            bvh,
            cam,
            q: bq,
            frame: 0,
            jitter: false,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: None,
            accumulate: true,
            gbuf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
        };
        let t = Instant::now();
        for _ in 0..BENCH_FRAMES {
            render::render_frame(&ctx, hybrid);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / BENCH_FRAMES as f64;
        eprintln!("{label}: {ms:6.1} ms/frame | per {BENCH_FRAMES} frames: {}", stats.summary_line());
        // Save the plain-hybrid and hemi-GI images while their buffers are
        // fresh (frame stays 0, so accum holds exactly the last frame).
        if hybrid && !hemi_ao && !hemi_gi && !shafts {
            let mut present = vec![0u32; rw * rh];
            render::resolve(&accum, &info, 1, false, &mut present, rw, rh, rw, rh);
            save_png("check.png", &present, rw, rh);
        } else if hemi_gi {
            let mut present = vec![0u32; rw * rh];
            render::resolve(&accum, &info, 1, false, &mut present, rw, rh, rw, rh);
            save_png("check_gi.png", &present, rw, rh);
        }
    }
    eprintln!("wrote check.png + check_gi.png");

    // Smoke-test dynamic resolution: one depth-capped frame per plausible cap
    // must complete; coarse coverage shrinks monotonically-ish as the cap
    // deepens until the leaf depth makes it a normal hybrid frame.
    for cap in [3u32, 5, 7] {
        stats.clear();
        let ctx = FrameCtx {
            scene,
            bvh,
            cam,
            q,
            frame: 0,
            jitter: false,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: None,
            accumulate: true,
            gbuf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
        };
        let t = Instant::now();
        render::render_frame_capped(&ctx, cap);
        eprintln!(
            "dynamic (depth cap {cap}): {:5.1} ms | coarse tiles {} covering {} px",
            t.elapsed().as_secs_f64() * 1000.0,
            stats.coarse_tiles.load(Relaxed),
            stats.coarse_pixels.load(Relaxed),
        );
    }

    // Temporal cache: warm it with one full-depth frame at camera A, then
    // verify frames that consume it. Every pixel still gets the tmin=0
    // reference-ray treatment, so a bad seed surfaces as false-sky/overshoot.
    // All deterministic — no wall clock.
    let tcache = temporal::TemporalCache::new(rw, rh);
    {
        let ctx = FrameCtx {
            scene,
            bvh,
            cam,
            q,
            frame: 0,
            jitter: false,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: Some(&tcache),
            tcache_prev: None,
            accumulate: true,
            gbuf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
        };
        render::render_frame(&ctx, true);
    }
    let mut temporal_ok = true;
    let mut temporal_pass = |label: &str, basis: &camera::CamBasis, max_depth: Option<u32>, want_seeds: bool, want_sky: bool| {
        stats.clear();
        let rep = render::verify(scene, bvh, basis, q, rw, rh, &stats, max_depth, Some((&tcache, cam)));
        let seeds = stats.temporal_seeds.load(Relaxed);
        let sky = stats.temporal_sky_tiles.load(Relaxed);
        let tests = stats.temporal_tests.load(Relaxed);
        eprintln!(
            "verify temporal {label} ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e} | seeds {seeds} sky-tiles {sky} cells {tests} coarse px {}",
            rep.pixels, rep.false_sky, rep.overshoot, rep.hybrid_extra, rep.max_rel_err, rep.coarse
        );
        let mut ok = rep.ok();
        if structural && want_seeds && seeds == 0 {
            eprintln!("verify temporal {label}: expected temporal seeds > 0 — the path didn't fire");
            ok = false;
        }
        if structural && want_sky && sky == 0 {
            eprintln!("verify temporal {label}: expected temporal sky-tiles > 0 — the sky path didn't fire");
            ok = false;
        }
        if structural && max_depth.is_some() && rep.coarse == 0 {
            eprintln!("verify temporal {label}: expected coarse pixels, found none — capped path not exercised");
            ok = false;
        }
        temporal_ok &= ok;
    };
    // T1: identical basis — the static-accumulation fast path. Every sky tile
    // must come from the cache and at least one node must seed.
    temporal_pass("static", &cam, None, true, true);
    // T2: pure forward dolly. Seeds must fire (the root, at minimum: its
    // extreme dirs are the old corners ± fp on the old screen boundary plus
    // the focus of expansion). Sky is NOT asserted: at this δ the λ_max tilt
    // drags every sky tile's query box toward the FOE, across the sky
    // boundary into finite cells — expected, not a regression.
    let mut cam_b = cam0;
    cam_b.pos += cam0.forward() * (0.02 * scene.diag);
    let basis_b = cam_b.basis(rw, rh);
    temporal_pass("dolly", &basis_b, None, true, false);
    // T3: the same dolly through the depth-capped driver (the root's seed is
    // cap-independent).
    temporal_pass("dolly capped d=4", &basis_b, Some(4), true, false);
    // T4: translate + rotate — the root leaves the old screen and this
    // scene's finite bound landscape is single-valued (the ground AABB blocks
    // everything at one distance), so only correctness is asserted.
    let mut cam_c = cam_b;
    cam_c.yaw += 0.05;
    temporal_pass("dolly+yaw", &cam_c.basis(rw, rh), None, false, false);
    // T5: pure rotation — the region-min query's structural win: δ = 0, the
    // old proven-empty balls are unchanged in world space, and panned-into
    // sky tiles overlap only old sky cells → free. Seeds are NOT asserted:
    // with a single-valued finite landscape, min == inherited everywhere.
    let mut cam_y = cam0;
    cam_y.yaw += 0.05;
    let basis_y = cam_y.basis(rw, rh);
    temporal_pass("yaw", &basis_y, None, false, true);

    // Informational A/B: static (the accumulation-frame path) and pure yaw
    // (the rotation path), each cold vs seeded. Not gated — the win is
    // scene-dependent.
    for (label, basis, prev) in [
        ("static cold", cam, None),
        ("static warm", cam, Some((&tcache, cam))),
        ("yaw cold   ", basis_y, None),
        ("yaw warm   ", basis_y, Some((&tcache, cam))),
    ] {
        stats.clear();
        let ctx = FrameCtx {
            scene,
            bvh,
            cam: basis,
            q,
            frame: 0,
            jitter: false,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: prev,
            accumulate: true,
            gbuf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
        };
        let t = Instant::now();
        render::render_frame(&ctx, true);
        eprintln!(
            "temporal A/B {label}: {:5.1} ms | frustum nodes {} | queries {} | temporal: seeds {} sky {} cells {}",
            t.elapsed().as_secs_f64() * 1000.0,
            stats.frustum_nodes.load(Relaxed),
            stats.frustum_queries.load(Relaxed),
            stats.temporal_seeds.load(Relaxed),
            stats.temporal_sky_tiles.load(Relaxed),
            stats.temporal_tests.load(Relaxed),
        );
    }

    if sph_ok && reproj_ok && hemi_ok && gi_ok && shaft_ok && rep.ok() && capped_ok && temporal_ok {
        eprintln!("CHECK PASSED");
        0
    } else {
        eprintln!("CHECK FAILED");
        1
    }
}

/// Extract the Win32 HWND from the SDL2 window for swapchain creation.
#[cfg(windows)]
fn sdl_hwnd(window: &sdl2::video::Window) -> windows::Win32::Foundation::HWND {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let handle = window.window_handle().expect("window handle").as_raw();
    match handle {
        RawWindowHandle::Win32(h) => {
            windows::Win32::Foundation::HWND(h.hwnd.get() as *mut core::ffi::c_void)
        }
        _ => unreachable!("non-Win32 window handle on Windows"),
    }
}

#[cfg(windows)]
fn run_window(scene: &scene::Scene, bvh: &bvh::Bvh, opts: &Opts, cam0: Camera) {
    // Opt into per-monitor DPI awareness so Windows doesn't stretch the window
    // on scaled displays — W×H stays W×H physical pixels, matching the swapchain.
    sdl2::hint::set("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2");
    let sdl = sdl2::init().expect("SDL init failed");
    let video = sdl.video().expect("SDL video failed");
    let mut window = video
        .window(
            "frustracer — R: hybrid/plain  T: dynamic-res  O: overlay  B: gpu-tonemap  G: dlss  X: xess  N: oidn  H: hemi-bounce  1-3: quality  C: verify  P: screenshot",
            W as u32,
            H as u32,
        )
        .position_centered()
        .build()
        .expect("failed to open window");
    let mut inp = input::Input::new(&sdl).expect("SDL event pump failed");
    let mut gpu = gpu::GpuContext::new(
        sdl_hwnd(&window),
        W as u32,
        H as u32,
        &gpu::GpuOptions {
            dlss: opts.dlss,
            sl_dir: opts.sl_path.clone(),
            xess: opts.xess,
            xess_dir: opts.xess_path.clone(),
            xess_autoexposure: opts.xess_autoexposure,
            debug: opts.gpu_debug,
        },
    )
    .expect("GPU init failed");

    let mut cam = cam0;
    let accum: Vec<AtomicU32> = (0..W * H * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..W * H).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..W * H).map(|_| AtomicU32::new(0)).collect();
    let mut present = vec![0u32; W * H];
    let stats = Stats::default();
    // Temporal cache pair: cur is filled by this frame, prev (last full-res
    // hybrid frame + the exact basis it traced with) seeds it. Half-res and
    // plain frames don't participate and drop prev — a cache from any older
    // frame or another resolution is never consulted.
    let tcaches = [temporal::TemporalCache::new(W, H), temporal::TemporalCache::new(W, H)];
    let mut tcur = 0usize;
    let mut tprev_ok = false;
    let mut tprev_basis = cam.basis(W, H); // placeholder until tprev_ok
    let mut tprev_res = (0usize, 0usize); // resolution the prev cache was traced at

    let mut frame: u32 = 0;
    let mut hybrid = true;
    let mut dynamic = true;
    let mut overlay_on = false;
    let mut gpu_tonemap = false;
    // Hemisphere frustum bounces (H cycles off → AO → GI): still-frame
    // quality — moving/DLSS frames keep the sampled path.
    let mut bounce_mode = 0u32;
    let mut preset = 2u32;
    let mut prev_rw = W;

    // DLSS Ray Reconstruction state. In DLSS mode every frame is a fresh
    // 1-spp hybrid frame at RR's Quality-mode render resolution (RR upscales
    // + denoises to the window size) and RR is the only temporal integrator:
    // no CPU accumulation, no half-res moving mode, no depth-cap budget.
    let mut dlss_on = gpu.dlss_ready();
    let (drw, drh) =
        gpu.rr_render_res().map(|(a, b)| (a as usize, b as usize)).unwrap_or((W, H));
    // Step-wise DRS (shares xess::ScaleCtl / quantize_res — pure controller
    // math common to both upscalers). Steps are made RARE (the quantization
    // plus the StepLimiter dwell is the hysteresis) because RR re-initializes
    // its internal denoiser on an input-res change — but a step is a scale
    // change, not a scene change: the res-step block below does NOT reset
    // (no dlss_reset, no prev drop; history survives via the extent tags).
    // A degenerate reported range (min == max) means the driver offers no
    // DRS — fixed res, no controller.
    let dlss_range = gpu.rr_res_range();
    let dlss_drs = dlss_range.map(|(_, min, max)| min != max).unwrap_or(false);
    let mut dlss_ctl = dlss_range.filter(|_| dlss_drs).map(|(_, min, max)| {
        let start_h = (H * 2 / 3).clamp(min.1 as usize, max.1 as usize);
        xess::ScaleCtl::new(start_h, min.1 as usize, max.1 as usize, H)
    });
    // G-buffer capacity = the range max; reinterpreted per step via set_res.
    let mut gbufs = {
        let (gw, gh) = dlss_range
            .map(|(_, _, max)| (max.0 as usize, max.1 as usize))
            .unwrap_or((drw, drh));
        dlss::GBufs::new(gw.max(drw), gh.max(drh))
    };
    gbufs.set_res(drw, drh); // fixed-res default until the controller speaks
    let mut dlss_idx: u32 = 0;
    let mut dlss_prev: Option<dlss::DlssPrev> = None;
    let mut dlss_reset = true;
    // Applied-step rate limiters, one per DRS path: bound how often the
    // upscalers take a history hit (see xess::StepLimiter).
    let mut dlss_lim = xess::StepLimiter::new();
    let mut xess_lim = xess::StepLimiter::new();
    // FRUSTRACER_STAB=1: numeric stability meter — every 15th upscaled frame
    // is read back and diffed against the previous capture; a static camera
    // on a converged pipeline trends to ~0, temporal instability ("dancing")
    // shows as a persistently high mean.
    let stab_on = std::env::var("FRUSTRACER_STAB").is_ok();
    let mut stab_prev: Option<Vec<u32>> = None;
    let mut stab_n = 0u32;
    let (dlss_near, dlss_far) = dlss::near_far(scene.diag);
    if dlss_on {
        eprintln!(
            "dlss: Ray Reconstruction ON (G toggles), dynamic resolution {}",
            if dlss_drs {
                "ON (step-wise; history survives steps)"
            } else {
                "unavailable (degenerate range) — fixed render res"
            }
        );
    }

    // XeSS-SR state (--xess sessions; X toggles). The all-in "a pixel is a
    // sample" mode: every frame is a fresh jittered 1-spp full-depth hybrid
    // trace at a dynamic render resolution picked by the scale controller
    // (quantized inside the SDK's queried input range), and XeSS's temporal
    // accumulation of the jittered sample stream is the ONLY spatial
    // reconstruction — the depth-cap/quad-fill budget path, the half-res
    // moving mode, and CPU accumulation never run. N composes an OIDN
    // pre-denoise at the same dynamic render res (XeSS-SR is a TAA-upscaler,
    // not a denoiser); the render-res G-buffers are reinterpreted in place
    // on a res step (GBufs::set_res), and `xess_prev` is its own MV-basis
    // contract — dropped on any res step, since the previous frame's pixel
    // grid no longer matches.
    let mut xess_on = gpu.xess_ready();
    let xess_range = gpu.xess_res_range(); // (optimal, min, max)
    let mut xess_ctl = xess_range.map(|(_, min, max)| {
        // Start at ~2/3 scale (the DLSS-Quality neighborhood), not the SDK's
        // "optimal" — with the ULTRA_PERFORMANCE init that widens the range,
        // optimal is the 1/3-scale floor and would open blurry. The
        // controller corrects from here either way.
        let start_h = (H * 2 / 3).clamp(min.1 as usize, max.1 as usize);
        xess::ScaleCtl::new(start_h, min.1 as usize, max.1 as usize, H)
    });
    let mut xess_gbufs: Option<dlss::GBufs> = None; // capacity = range max, lazily allocated
    let mut xess_idx: u32 = 0;
    // The previous frame's CAMERA (not basis): the MV basis is derived at
    // each frame's own resolution, so it stays correct across DRS steps
    // without dropping history.
    let mut xess_prev: Option<Camera> = None;
    let mut xess_reset = true;
    // OIDN placement in XeSS mode (independent of the plain-mode `oidn_on`);
    // xess_hdr is the post-placement's window-res readback staging.
    let mut xess_oidn = XessOidn::Off;
    let mut xess_hdr: Vec<f32> = Vec::new();
    if xess_on {
        eprintln!("xess: dynamic super-resolution ON (X toggles; N cycles OIDN off/pre/post)");
        if !opts.adaptive {
            eprintln!("xess: adaptive shading rate OFF (--no-adaptive; uniform per-pixel shading)");
        }
    }

    // OIDN state — the secondary denoiser (N toggles, mutually exclusive
    // with DLSS). It keeps the normal render loop (temporal cache, budget
    // frames, hemi bounces) at forced full-res (the half-res moving mode
    // writes a half-res prefix that would misalign the full-res G-buffers)
    // and denoises each rendered frame. Two sub-modes (M toggles): temporal
    // (default) renders fresh 1-spp frames and folds them into a reprojected
    // EMA history (reproject.rs) that is the sole accumulator and denoiser
    // input; plain denoises the accumulation average and shimmers while
    // moving. Context, W×H G-buffers (~116 MB) and history (~77 MB) are
    // lazily created on first enable; a failed init is remembered and not
    // retried per keypress.
    let mut oidn_on = false;
    let mut oidn_temporal = opts.oidn_temporal;
    let mut oidn_ctx: Option<oidn::OidnContext> = None;
    let mut oidn_gbufs: Option<dlss::GBufs> = None;
    let mut oidn_hist: Option<reproject::History> = None;
    // Free-running frame index for the temporal mode's per-pixel RNG seed —
    // the dlss_idx pattern: `frame` is pinned to 0 while moving, which would
    // freeze the noise pattern and defeat the history average.
    let mut oidn_seq: u32 = 0;
    let mut last_hist = reproject::UpdateStats::default();
    let mut hist_ms = 0.0f64;
    // Previous frame's XeSS pre-denoise cost (set_res + history + filter):
    // it scales with the render area just like the trace, so the resolution
    // controller must see it or a slow OIDN device (e.g. CPU) lets the scale
    // creep to the range max while total frame time blows the budget.
    let mut pre_ms = 0.0f64;
    let mut oidn_failed = false;
    let oidn_try_enable = |oidn_ctx: &mut Option<oidn::OidnContext>,
                           oidn_gbufs: &mut Option<dlss::GBufs>,
                           oidn_hist: &mut Option<reproject::History>| {
        if oidn_ctx.is_none() {
            // XeSS sessions must not let OIDN auto-pick its SYCL device: the
            // SYCL runtime and libxess.dll drag conflicting Intel compute
            // stacks into one process and abort() natively at first use
            // (observed: OIDN 2.5 SYCL + XeSS SDK 2.0.2). Auto in a XeSS
            // session means CUDA then CPU; an explicit --oidn-device is
            // honored as given.
            let devices: &[i32] = if opts.xess && opts.oidn_device == 0 {
                &[3, 1] // cuda, cpu
            } else {
                &[opts.oidn_device]
            };
            for &d in devices {
                match oidn::OidnContext::new(
                    &opts.oidn_path,
                    W,
                    H,
                    d,
                    opts.oidn_quality,
                    opts.oidn_clean_aux,
                ) {
                    Ok(c) => {
                        eprintln!("oidn: ready on {} device", c.device_desc);
                        *oidn_ctx = Some(c);
                        break;
                    }
                    Err(e) => eprintln!("{e}"),
                }
            }
            if oidn_ctx.is_none() {
                eprintln!(
                    "oidn: DLLs expected at {} (--oidn-path / FRUSTRACER_OIDN_PATH)",
                    opts.oidn_path
                );
            }
        }
        if oidn_ctx.is_some() && oidn_gbufs.is_none() {
            *oidn_gbufs = Some(dlss::GBufs::new(W, H));
        }
        if oidn_ctx.is_some() && oidn_hist.is_none() {
            *oidn_hist = Some(reproject::History::new(W, H));
        }
        oidn_ctx.is_some()
    };
    if opts.oidn || opts.oidn_post {
        if oidn_try_enable(&mut oidn_ctx, &mut oidn_gbufs, &mut oidn_hist) {
            if xess_on {
                // XeSS sessions: --oidn = pre-upscale placement, --oidn-post
                // = post-upscale; the plain-mode oidn_on stays independent.
                xess_oidn = if opts.oidn_post { XessOidn::Post } else { XessOidn::Pre };
                eprintln!(
                    "oidn: {}-upscale denoise ON (N cycles off/pre/post)",
                    if opts.oidn_post { "POST" } else { "PRE" }
                );
            } else if opts.oidn_post {
                eprintln!("oidn: --oidn-post requires a live --xess session; ignoring");
            } else {
                oidn_on = true;
                if dlss_on {
                    dlss_on = false;
                    eprintln!("dlss: Ray Reconstruction OFF (--oidn; G re-enables)");
                }
                eprintln!(
                    "oidn: denoising ON, temporal reprojection {} (N / M toggle)",
                    if oidn_temporal { "ON" } else { "OFF" }
                );
            }
        } else {
            oidn_failed = true;
        }
    }
    let mut prev_budget = false;
    // Depth cap that fully resolves the screen (tiles reach LEAF_TILE): 7 at 1024.
    let depth_full: f32 = ((W.max(H) as f32) / render::LEAF_TILE as f32).log2().ceil();
    // Fractional depth-cap estimate for budget frames. Mid-range prior: one
    // slightly-coarse first frame beats a hitch. Deliberately not reset when
    // the camera stops — the last value is the best prior for the same
    // neighborhood, and the controller corrects within a frame on resume.
    let mut depth_est: f32 = 4.0;
    let mut last = Instant::now();
    let mut last_title = Instant::now();
    let mut last_stats = Instant::now();
    // Presented-frames-per-second, recomputed once per second (frame ms in
    // the title is render time only; this is the actual present rate).
    let mut fps = 0.0f64;
    let mut fps_frames = 0u32;
    let mut fps_t = Instant::now();
    let mut last_ms = 0.0f64;
    let mut shot = 0u32;

    loop {
        let now = Instant::now();
        let dt = (now - last).as_secs_f32().min(0.1);
        last = now;

        let edges = inp.poll();
        if edges.quit {
            break;
        }
        let moved = inp.apply_movement(&mut cam, dt, scene.diag);
        if edges.toggle_hybrid {
            hybrid = !hybrid;
            frame = 0;
            dlss_reset = true; // noise statistics change across the toggle
        }
        if edges.toggle_dynamic {
            if dlss_on {
                eprintln!(
                    "dynamic-res in DLSS mode is {}",
                    if dlss_drs {
                        "always on (the scale controller drives it, step-wise)"
                    } else {
                        "unavailable (driver reported no DRS range)"
                    }
                );
            } else if xess_on {
                eprintln!("dynamic-res is always on in XeSS mode (the scale controller drives it)");
            } else {
                dynamic = !dynamic;
                frame = 0;
            }
        }
        if edges.toggle_overlay {
            if dlss_on || xess_on {
                eprintln!("overlay unavailable in DLSS/XeSS mode (lives in the CPU resolve)");
            } else {
                overlay_on = !overlay_on;
            }
        }
        if edges.toggle_gpu_tone {
            gpu_tonemap = !gpu_tonemap;
            let note = if oidn_on { " (no effect in OIDN mode — presents via the CPU resolve)" } else { "" };
            eprintln!("tonemap: {}{note}", if gpu_tonemap { "GPU" } else { "CPU" });
        }
        if edges.toggle_bounce {
            bounce_mode = (bounce_mode + 1) % 4;
            frame = 0;
            eprintln!(
                "hemisphere frustum bounces: {}",
                ["OFF", "AO (still frames)", "GI (still frames)", "GI + shadow shafts (still frames)"]
                    [bounce_mode as usize]
            );
        }
        if edges.toggle_dlss {
            if gpu.dlss_ready() {
                dlss_on = !dlss_on;
                frame = 0;
                dlss_reset = true;
                dlss_prev = None;
                // Fresh limiter: a re-enabled session adopts the controller's
                // target immediately instead of dwelling on the stale res.
                dlss_lim = xess::StepLimiter::new();
                if dlss_on && oidn_on {
                    oidn_on = false;
                    eprintln!("oidn: OFF (DLSS enabled)");
                }
                if dlss_on && xess_on {
                    // Structurally unreachable (XeSS sessions never init SL),
                    // kept for the day both live on one pipeline.
                    xess_on = false;
                    xess_prev = None;
                    eprintln!("xess: OFF (DLSS enabled)");
                }
                eprintln!("dlss: Ray Reconstruction {}", if dlss_on { "ON" } else { "OFF" });
            } else {
                eprintln!("dlss: not available");
            }
        }
        if edges.toggle_xess {
            if gpu.xess_ready() {
                xess_on = !xess_on;
                frame = 0;
                xess_reset = true;
                xess_prev = None;
                // Fresh limiter: a re-enabled session adopts the controller's
                // target immediately instead of dwelling on the stale res.
                xess_lim = xess::StepLimiter::new();
                if xess_on && dlss_on {
                    dlss_on = false;
                    dlss_prev = None;
                    dlss_reset = true;
                    eprintln!("dlss: Ray Reconstruction OFF (XeSS enabled)");
                }
                if xess_on && oidn_on {
                    // Plain-mode OIDN is unreachable while XeSS presents (the
                    // xess_on present arm wins); clear it so M and the stats
                    // line don't keep acting on a denoiser that isn't running.
                    oidn_on = false;
                    eprintln!("oidn: OFF (XeSS enabled; N cycles the XeSS placement)");
                }
                eprintln!(
                    "xess: {}",
                    if xess_on {
                        match xess_oidn {
                            XessOidn::Off => "ON",
                            XessOidn::Pre => "ON (OIDN pre-denoise)",
                            XessOidn::Post => "ON (OIDN post-denoise)",
                        }
                    } else {
                        "OFF"
                    }
                );
            } else {
                eprintln!("xess: not available (start with --xess and the SDK DLL on disk)");
            }
        }
        if edges.toggle_oidn {
            if xess_on {
                // XeSS mode: N cycles the OIDN placement (off → pre → post).
                let next = match xess_oidn {
                    XessOidn::Off => XessOidn::Pre,
                    XessOidn::Pre => XessOidn::Post,
                    XessOidn::Post => XessOidn::Off,
                };
                if next == XessOidn::Off {
                    xess_oidn = next;
                    frame = 0;
                    eprintln!("oidn: OFF (raw XeSS)");
                } else if oidn_failed {
                    eprintln!("oidn: unavailable (earlier init failed; restart with --oidn-path to retry)");
                } else if oidn_try_enable(&mut oidn_ctx, &mut oidn_gbufs, &mut oidn_hist) {
                    xess_oidn = next;
                    frame = 0;
                    eprintln!(
                        "oidn: {} the XeSS upscale (N cycles off → pre → post)",
                        if next == XessOidn::Pre {
                            "PRE-denoise at render res, before"
                        } else {
                            "POST-denoise at window res, after"
                        }
                    );
                } else {
                    oidn_failed = true;
                }
            } else if oidn_on {
                oidn_on = false;
                frame = 0;
                eprintln!("oidn: OFF");
            } else if oidn_failed {
                eprintln!("oidn: unavailable (earlier init failed; restart with --oidn-path to retry)");
            } else if oidn_try_enable(&mut oidn_ctx, &mut oidn_gbufs, &mut oidn_hist) {
                oidn_on = true;
                frame = 0;
                if dlss_on {
                    dlss_on = false;
                    dlss_prev = None;
                    dlss_reset = true;
                    eprintln!("dlss: Ray Reconstruction OFF (OIDN enabled)");
                }
                eprintln!(
                    "oidn: ON{}, temporal reprojection {} (M toggles)",
                    if xess_on { " (XeSS pre-denoise at the dynamic render res)" } else { "" },
                    if oidn_temporal { "ON" } else { "OFF" }
                );
            } else {
                oidn_failed = true;
            }
        }
        // True only when M actually flipped oidn_temporal (not the Post/no-op
        // arms) — the single source for the reset predicates below, so they
        // can't drift from the handler's own condition.
        let mut temporal_flipped = false;
        if edges.toggle_temporal {
            if xess_on && xess_oidn == XessOidn::Post {
                eprintln!("oidn: no temporal history in POST placement (XeSS itself is the temporal integrator there)");
            } else if oidn_on || (xess_on && xess_oidn == XessOidn::Pre) {
                oidn_temporal = !oidn_temporal;
                temporal_flipped = true;
                // Required in BOTH directions: accum semantics flip between
                // "last 1-spp frame" (temporal) and "pure sum" (plain) — a
                // stale sample count would divide a 1-spp frame to near-black.
                frame = 0;
                eprintln!(
                    "oidn: temporal reprojection {}",
                    if oidn_temporal { "ON" } else { "OFF (plain accumulation)" }
                );
            } else {
                eprintln!("oidn: temporal toggle is OIDN-only (N enables OIDN, then M toggles)");
            }
        }
        if let Some(p) = edges.quality {
            preset = p;
            frame = 0;
            dlss_reset = true;
        }
        // Any XeSS frame after this point sends reset_history = 1: the
        // upscaler's accumulated history mixes shading statistics, so every
        // predicate that resets `frame`/the OIDN history also resets it —
        // EXCEPT camera motion, which is exactly what the temporal upscaler
        // exists to survive.
        if edges.toggle_hybrid
            || edges.toggle_bounce
            || edges.quality.is_some()
            || edges.toggle_xess
            || edges.toggle_oidn
            || temporal_flipped
        {
            xess_reset = true;
        }
        if moved {
            frame = 0;
        }
        // Reprojection-history invalidation: any setting change that alters
        // shading or mode semantics drops the history; camera motion and the
        // budget↔normal transition deliberately do NOT (surviving motion is
        // the history's whole purpose; coarse budget pixels are handled
        // per-pixel by the KIND_COARSE rule). Over-invalidating on no-op
        // edges (e.g. T in DLSS mode) is accepted for the simple predicate.
        let hist_stale = edges.toggle_hybrid
            || edges.toggle_dynamic
            || edges.toggle_bounce
            || edges.quality.is_some()
            || edges.toggle_oidn
            || edges.toggle_dlss
            || edges.toggle_xess
            || temporal_flipped;
        if hist_stale {
            if let Some(h) = &mut oidn_hist {
                h.invalidate();
            }
        }

        // All toggle handlers have run: resolve this frame's mode once.
        // Everything below reads `mode`, never the flag soup.
        let mode = if dlss_on {
            RenderMode::Dlss
        } else if xess_on {
            RenderMode::Xess
        } else if oidn_on {
            RenderMode::Oidn { temporal: oidn_temporal }
        } else {
            RenderMode::Plain
        };
        // DLSS/XeSS: an upscaler owns temporal integration — fresh 1-spp
        // frames, fixed cheap preset, no budget path, no CPU accumulation.
        let upscaled = matches!(mode, RenderMode::Dlss | RenderMode::Xess);

        // Cheap while moving, converge while still. Dynamic-res mode keeps
        // full resolution buffers and full quality — the estimated depth cap
        // floats the effective resolution instead. DLSS mode traces every
        // frame uncapped at RR's fixed render resolution (RR requires clean
        // per-pixel G-buffers) with frame-stationary quality.
        let use_budget = moved && hybrid && dynamic && !upscaled;
        // Emergency-shed predicate for the step limiters: a badly blown
        // previous frame may bypass the dwell (shed only, never grow).
        let blown = last_ms as f32 > 1.5 * RENDER_BUDGET.as_secs_f32() * 1000.0;
        let (rw, rh) = match mode {
            // Step-wise DRS when the driver reported a range; the fixed
            // optimal res otherwise. Same controller math as XeSS, with the
            // step limiter bounding how often RR takes a history hit.
            RenderMode::Dlss if dlss_drs => {
                let (_, min, max) = dlss_range.unwrap();
                let target = xess::quantize_res(
                    dlss_ctl.as_ref().unwrap().scale(),
                    (W, H),
                    (min.0 as usize, min.1 as usize),
                    (max.0 as usize, max.1 as usize),
                );
                dlss_lim.apply(target, blown)
            }
            RenderMode::Dlss => (drw, drh),
            RenderMode::Xess => {
                // XeSS mode: dynamic resolution, no block filling — the scale
                // controller's estimate quantized into the SDK's input range.
                // Every frame is a full-depth per-pixel trace at this size.
                let (_, min, max) = xess_range.unwrap();
                let target = xess::quantize_res(
                    xess_ctl.as_ref().unwrap().scale(),
                    (W, H),
                    (min.0 as usize, min.1 as usize),
                    (max.0 as usize, max.1 as usize),
                );
                xess_lim.apply(target, blown)
            }
            // OIDN mode never drops to half-res: the G-buffers are full-res
            // and a half-res frame renders into a prefix with a different
            // stride. Budget (dynamic-res) frames are full-res and fine.
            RenderMode::Oidn { .. } => (W, H),
            RenderMode::Plain if moved && !use_budget => (W / 2, H / 2),
            RenderMode::Plain => (W, H),
        };
        if rw != prev_rw {
            frame = 0;
            prev_rw = rw;
        }
        if xess_on {
            // Keep the render-res G-buffers on this frame's resolution. On a
            // res step (rare — the quantization is the hysteresis) the
            // buffers are reinterpreted in place and the previous frame's MV
            // basis is dropped: its pixel grid no longer matches, and XeSS's
            // history is reset with it (the temporal cache drops itself via
            // tprev_res).
            if xess_gbufs.is_none() {
                let (_, _, max) = xess_range.unwrap();
                xess_gbufs = Some(dlss::GBufs::new(max.0 as usize, max.1 as usize));
            }
            let g = xess_gbufs.as_mut().unwrap();
            if (g.rw, g.rh) != (rw, rh) {
                // A step is a scale change, not a scene change: no
                // reset_history, no prev drop — the MV basis is derived from
                // the prev CAMERA at each frame's own res, so it stays
                // correct across the step, and XeSS's DRS carries its
                // accumulation across extent changes by design. (Resetting
                // here was the "dancing": every step wiped the history and
                // the image re-converged patchily.)
                g.set_res(rw, rh);
                eprintln!("xess: drs step -> {rw}x{rh}");
            }
        }
        if mode == RenderMode::Dlss && (gbufs.rw, gbufs.rh) != (rw, rh) {
            // Same contract for RR: rebuild the previous frame's basis and
            // matrices at the NEW resolution (same pose, new pixel mapping)
            // so MVs land in current-res pixels; history survives via the
            // extents. The temporal cache still drops itself via tprev_res —
            // that one is a correctness contract, not a quality choice.
            gbufs.set_res(rw, rh);
            if let Some(p) = &mut dlss_prev {
                p.basis = p.cam.basis(rw, rh);
                p.mats = dlss::cam_matrices(&p.cam, rw, rh, dlss_near, dlss_far);
            }
            eprintln!("dlss: drs step -> {rw}x{rh}");
        }
        if use_budget != prev_budget {
            frame = 0; // budget frames hold coarse fills — never accumulate onto them
            prev_budget = use_budget;
        }
        let base_q = Quality::preset(preset);
        let mut q = if upscaled {
            // Fixed cheap preset: the temporal denoisers/upscalers want
            // frame-stationary noise statistics.
            Quality {
                shadow_samples: 1,
                ao_samples: 1,
                reflections: true,
                fb: shade::FrustumBounce::OFF,
            }
        } else if moved && !use_budget {
            base_q.while_moving()
        } else {
            base_q
        };
        if !upscaled && !moved {
            q.fb.ao = bounce_mode == 1;
            q.fb.gi = bounce_mode >= 2;
            q.fb.shadows = bounce_mode == 3;
        }

        // Temporal-OIDN mode renders fresh 1-spp frames (the DLSS pattern);
        // the reprojected history in the present chain is the accumulator.
        let oidn_t = mode == RenderMode::Oidn { temporal: true };
        // DLSS and XeSS modes never idle: fresh jittered 1-spp frames are
        // what their temporal accumulators integrate — super-resolution in
        // XeSS's case, which converges while "still" instead of on the CPU.
        let rendered = upscaled || frame < MAX_SAMPLES;
        // Hoisted out of the render arm: the OIDN present branch needs the
        // exact basis this frame traced with for the history update.
        let basis = cam.basis(rw, rh);
        if rendered {
            stats.clear();
            // Budget (moving) frames are full-res, so the cache stays live
            // through motion and across the moving→static transition. In
            // DLSS mode "full participation res" is RR's render resolution;
            // the tprev_res check drops the cache whenever the resolution
            // changes (e.g. the G toggle), per the temporal invariant.
            // XeSS mode participates at whatever res this frame traces:
            // every frame is a full-depth hybrid frame at one fixed res, so
            // the producer/consumer contract holds; the tprev_res check
            // below drops the prev cache across any res step.
            let temporal_on = hybrid
                && (upscaled || (rw, rh) == (W, H));
            let (tcache_cur, tcache_prev) = if temporal_on {
                tcaches[tcur].clear();
                (
                    Some(&tcaches[tcur]),
                    if tprev_ok && tprev_res == (rw, rh) {
                        Some((&tcaches[tcur ^ 1], tprev_basis))
                    } else {
                        None
                    },
                )
            } else {
                (None, None)
            };
            let ctx = FrameCtx {
                scene,
                bvh,
                cam: basis,
                q,
                frame: match mode {
                    RenderMode::Dlss => dlss_idx,
                    RenderMode::Xess => xess_idx, // free-running, the dlss_idx pattern
                    // free-running: decorrelates the RNG while `frame` is pinned
                    RenderMode::Oidn { temporal: true } => oidn_seq,
                    _ => frame,
                },
                // DLSS/XeSS ignore `jitter` (frame_jitter wins in
                // trace_primary); temporal OIDN always jitters its fresh
                // 1-spp frames; the accumulating modes jitter after the
                // first (pilot) sample.
                jitter: oidn_t || (!upscaled && frame > 0),
                rw,
                rh,
                accum: &accum,
                info: &info,
                tbuf: &tbuf,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur,
                tcache_prev,
                accumulate: !upscaled && !oidn_t,
                gbuf: match mode {
                    RenderMode::Dlss => Some(&gbufs),
                    RenderMode::Xess => xess_gbufs.as_ref(),
                    // Both OIDN sub-modes fill the window-res G-buffers.
                    RenderMode::Oidn { .. } => oidn_gbufs.as_ref(),
                    RenderMode::Plain => None,
                },
                prev_cam: match mode {
                    RenderMode::Dlss => dlss_prev.as_ref().map(|p| p.basis),
                    // Basis derived at THIS frame's res — correct across
                    // DRS steps by construction.
                    RenderMode::Xess => xess_prev.map(|c| c.basis(rw, rh)),
                    _ => None,
                },
                frame_jitter: match mode {
                    RenderMode::Dlss => Some(dlss::jitter_for(dlss_idx)),
                    RenderMode::Xess => Some(dlss::jitter_for(xess_idx)),
                    _ => None,
                },
                // Adaptive shading rate: XeSS only — its temporal
                // accumulation launders the spatially varying sampling. On
                // RR the 2×2-correlated shadow noise and per-frame cell
                // reclassification presented as patchy "dancing" (RR's
                // network preserves block-correlated noise as structure
                // instead of integrating it). Revisit with an RR-friendly
                // classifier before re-enabling. --no-adaptive forces
                // uniform per-pixel shading (visibility is per-pixel
                // either way).
                adaptive: mode == RenderMode::Xess && opts.adaptive,
            };
            let t = Instant::now();
            if use_budget {
                render::render_frame_capped(&ctx, depth_est.floor() as u32);
            } else {
                render::render_frame(&ctx, hybrid);
            }
            last_ms = t.elapsed().as_secs_f64() * 1000.0;
            if temporal_on {
                tprev_basis = basis;
                tprev_res = (rw, rh);
                tprev_ok = true;
                tcur ^= 1;
            } else {
                tprev_ok = false;
            }
            if use_budget {
                // Update the cap estimate from this frame only — non-budget
                // frames (half-res, plain, converging) have incomparable cost.
                let target = RENDER_BUDGET.as_secs_f32() * 1000.0;
                let err = (target / (last_ms as f32).max(0.1)).log2() * 0.5; // log4
                let step = (DEPTH_GAIN * err).clamp(-STEP_DOWN_MAX, STEP_UP_MAX);
                // Deadband: don't climb while already using >60% of the budget
                // (the next level costs ~4x) — parks at the deepest cap that
                // fits instead of flapping across the boundary.
                if step < 0.0 || (last_ms as f32) < 0.6 * target {
                    depth_est = (depth_est + step).clamp(render::MIN_BUDGET_DEPTH as f32, depth_full);
                }
            }
            if xess_on {
                // The scale controller only ever sees XeSS frames — a
                // comparable cost model (full-depth trace, cost ~ area).
                // While still, the temporal cache makes frames cheaper and
                // the scale creeps toward the range max: super-resolution.
                // The previous frame's pre-denoise cost rides along: it is
                // area-proportional work the chosen resolution buys, and a
                // controller blind to it would creep past the budget on a
                // slow OIDN device.
                if let Some(ctl) = &mut xess_ctl {
                    ctl.update(
                        (last_ms + pre_ms) as f32,
                        RENDER_BUDGET.as_secs_f32() * 1000.0,
                    );
                }
            }
            if mode == RenderMode::Dlss {
                // Same controller, DLSS flavor. RR has no CPU pre-pass, so
                // the trace time alone is the area-proportional cost.
                if let Some(ctl) = &mut dlss_ctl {
                    ctl.update(last_ms as f32, RENDER_BUDGET.as_secs_f32() * 1000.0);
                }
            }
            frame += 1;
            oidn_seq = oidn_seq.wrapping_add(1);
        } else {
            std::thread::sleep(Duration::from_millis(8)); // converged — idle
        }

        // GPU tonemap consumes the raw HDR accumulation directly, but only
        // for full-res frames without the overlay — half-res upscale and the
        // overlay composite live in the CPU resolve. OIDN mode presents its
        // denoised output through the CPU path, so it excludes the GPU tonemap.
        let use_gpu_tone = gpu_tonemap && !oidn_on && rw == W && !(overlay_on && hybrid);
        if dlss_on {
            // DLSS-RR: hand the 1-spp radiance + G-buffers to the denoiser;
            // it outputs denoised HDR which the GPU tonemap presents.
            // Everything SL sees — matrices, jitter, MVs, mvec_scale — lives
            // in render-res pixel space; only the RR output is window-sized.
            let mats = dlss::cam_matrices(&cam, rw, rh, dlss_near, dlss_far);
            let fc = dlss::frame_constants(
                &cam,
                &mats,
                dlss_prev.as_ref().map(|p| &p.mats),
                dlss::jitter_for(dlss_idx),
                dlss_reset,
                dlss_near,
                dlss_far,
                rw,
                rh,
            );
            match gpu.present_rr(&accum, &gbufs, &fc, dlss_idx) {
                Ok(()) => {
                    dlss_prev = Some(dlss::DlssPrev { basis: cam.basis(rw, rh), mats, cam });
                    dlss_reset = false;
                    dlss_idx = dlss_idx.wrapping_add(1);
                }
                Err(e) => {
                    // A Streamline failure (e.g. out of VRAM) shouldn't kill
                    // the app: the aborted frame never reached the GPU, so
                    // fall back to the CPU pipeline; the next loop iteration
                    // presents normally. G retries.
                    eprintln!("dlss: present failed ({e}); Ray Reconstruction disabled (G to retry)");
                    dlss_on = false;
                    dlss_prev = None;
                    dlss_reset = true;
                    frame = 0;
                }
            }
        } else if xess_on {
            // XeSS-SR: hand the fresh 1-spp frame (optionally OIDN-denoised
            // first, at the same dynamic render res) + MV/depth to the
            // upscaler; it accumulates the jittered sample stream into the
            // window-sized image. Everything XeSS sees is in input-res pixel
            // space; only its output is window-sized. The denoised/upscaled
            // output is never written back into accum or the history.
            let xg = xess_gbufs.as_ref().expect("xess_on without gbufs");
            let n = rw * rh * 3;
            let jit = dlss::jitter_for(xess_idx);
            if xess_oidn == XessOidn::Post {
                // POST placement (the A/B experiment): raw 1-spp → XeSS
                // upscale → readback → OIDN at window res with
                // nearest-upscaled guides → CPU tonemap → present_cpu (the
                // frame's single Present). Costs a synchronous readback and
                // a window-res denoise; no pre-EMA history in this ordering.
                // Post's denoise is window-res (constant, not area-
                // proportional) — shedding render resolution wouldn't reduce
                // it, so the scale controller must not see it.
                pre_ms = 0.0;
                if xess_hdr.len() != W * H * 3 {
                    xess_hdr.resize(W * H * 3, 0.0);
                }
                match gpu.upscale_xess_to_cpu(
                    &gpu::xr::ColorSrc::Accum(&accum[..n]),
                    xg,
                    rw,
                    rh,
                    jit,
                    xess_reset,
                    dlss_near,
                    dlss_far,
                    &mut xess_hdr,
                ) {
                    Ok(()) => {
                        xess_prev = Some(cam);
                        xess_reset = false;
                        xess_idx = xess_idx.wrapping_add(1);
                        let octx = oidn_ctx.as_mut().expect("xess_oidn without context");
                        let og = oidn_gbufs.as_ref().expect("xess_oidn without gbufs");
                        og.upscale_guides_from(xg);
                        let result = octx
                            .set_res(W, H)
                            .and_then(|()| octx.denoise_hdr(&xess_hdr, og).map(drop));
                        match result {
                            Ok(()) => render::resolve_hdr(
                                octx.last_output(),
                                &info,
                                false,
                                &mut present,
                                W,
                                H,
                                W,
                                H,
                            ),
                            Err(e) => {
                                eprintln!(
                                    "oidn: post-denoise failed ({e}); presenting the raw upscale (N to retry)"
                                );
                                xess_oidn = XessOidn::Off;
                                render::resolve_hdr(
                                    &xess_hdr, &info, false, &mut present, W, H, W, H,
                                );
                            }
                        }
                        gpu.present_cpu(&present).expect("GPU present failed");
                    }
                    Err(e) => {
                        eprintln!("xess: upscale failed ({e}); XeSS disabled (X to retry)");
                        xess_on = false;
                        xess_prev = None;
                        xess_reset = true;
                        frame = 0;
                    }
                }
            } else {
                // OFF / PRE placements: the GPU tonemap present path. The
                // pre-pass is resolution-agile — set_res rebinds the filter
                // on a res step (cheap; the weights stay loaded), and the
                // reprojected history reallocates (a fresh history is an
                // invalidated one; XeSS's own accumulation hides the blip).
                let denoised = if xess_oidn == XessOidn::Pre {
                    let octx = oidn_ctx.as_mut().expect("xess_oidn without context");
                    let t_pre = Instant::now();
                    let result = octx.set_res(rw, rh).and_then(|()| {
                        if oidn_temporal {
                            let hist = oidn_hist.as_mut().expect("xess_oidn without history");
                            if hist.res() != (rw, rh) {
                                *hist = reproject::History::new(rw, rh);
                            }
                            let t0 = Instant::now();
                            last_hist = hist.update(
                                &basis,
                                &accum[..n],
                                xg,
                                &info[..rw * rh],
                                dlss_far,
                                MAX_SAMPLES as f32,
                            );
                            hist_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            octx.denoise_hdr(hist.color(), xg).map(drop)
                        } else {
                            // accum holds exactly the last 1-spp frame (store
                            // semantics in XeSS mode) — denoise it per-frame.
                            octx.denoise(&accum[..n], 1, xg).map(drop)
                        }
                    });
                    pre_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
                    match result {
                        Ok(()) => Some(()),
                        Err(e) => {
                            eprintln!("oidn: pre-denoise failed ({e}); raw XeSS input (N to retry)");
                            xess_oidn = XessOidn::Off;
                            pre_ms = 0.0;
                            // The color source flips denoised→raw this same
                            // frame; the upscaler must not integrate across
                            // the statistics flip (mirrors the keyboard
                            // Pre→Off transition, which resets via the
                            // toggle_oidn edge).
                            xess_reset = true;
                            None
                        }
                    }
                } else {
                    pre_ms = 0.0;
                    None
                };
                let color = match denoised {
                    Some(()) => gpu::xr::ColorSrc::Hdr(oidn_ctx.as_ref().unwrap().last_output()),
                    None => gpu::xr::ColorSrc::Accum(&accum[..n]),
                };
                match gpu.present_xess(&color, xg, rw, rh, jit, xess_reset, dlss_near, dlss_far) {
                    Ok(()) => {
                        xess_prev = Some(cam);
                        xess_reset = false;
                        xess_idx = xess_idx.wrapping_add(1);
                    }
                    Err(e) => {
                        // Nothing reached the GPU (abort_frame) — fall back to
                        // the CPU pipeline next iteration; X retries.
                        eprintln!("xess: present failed ({e}); XeSS disabled (X to retry)");
                        xess_on = false;
                        xess_prev = None;
                        xess_reset = true;
                        frame = 0;
                    }
                }
            }
        } else if oidn_on {
            // OIDN: denoise on the CPU-resolve path — temporal mode folds the
            // fresh 1-spp frame into the reprojected history and denoises
            // that; plain mode denoises the accumulation average. The
            // denoised HDR is never written back into accum or the history.
            // On an idle (converged) iteration the cached present buffer is
            // re-presented without re-denoising; an overlay toggle while idle
            // re-composites the retained denoised HDR instead.
            let octx = oidn_ctx.as_mut().expect("oidn_on without context");
            if rendered {
                let og = oidn_gbufs.as_ref().expect("oidn_on without gbufs");
                // Returning from XeSS mode leaves the filter/history bound at
                // a render res — rebind at window res (no-op otherwise).
                let result = octx.set_res(W, H).and_then(|()| {
                    if oidn_t {
                        let hist = oidn_hist.as_mut().expect("oidn_on without history");
                        if hist.res() != (W, H) {
                            *hist = reproject::History::new(W, H);
                        }
                        let t0 = Instant::now();
                        last_hist =
                            hist.update(&basis, &accum, og, &info, dlss_far, MAX_SAMPLES as f32);
                        hist_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        octx.denoise_hdr(hist.color(), og).map(drop)
                    } else {
                        octx.denoise(&accum, frame.max(1), og).map(drop)
                    }
                });
                match result {
                    Ok(()) => render::resolve_hdr(
                        octx.last_output(),
                        &info,
                        overlay_on && hybrid,
                        &mut present,
                        W,
                        H,
                        W,
                        H,
                    ),
                    Err(e) => {
                        eprintln!("oidn: denoise failed ({e}); OFF (N to retry)");
                        oidn_on = false;
                        if oidn_t {
                            // accum holds one 1-spp frame, not a sum: resolve
                            // it as 1 sample and restart accumulation cleanly.
                            frame = 0;
                            if let Some(h) = &mut oidn_hist {
                                h.invalidate();
                            }
                        }
                        render::resolve(
                            &accum,
                            &info,
                            if oidn_t { 1 } else { frame.max(1) },
                            overlay_on && hybrid,
                            &mut present,
                            rw,
                            rh,
                            W,
                            H,
                        );
                    }
                }
            } else if edges.toggle_overlay {
                render::resolve_hdr(
                    octx.last_output(),
                    &info,
                    overlay_on && hybrid,
                    &mut present,
                    W,
                    H,
                    W,
                    H,
                );
            }
            gpu.present_cpu(&present).expect("GPU present failed");
        } else if use_gpu_tone {
            gpu.present_hdr(&accum, frame.max(1)).expect("GPU present failed");
        } else {
            render::resolve(
                &accum,
                &info,
                frame.max(1),
                overlay_on && hybrid,
                &mut present,
                rw,
                rh,
                W,
                H,
            );
            gpu.present_cpu(&present).expect("GPU present failed");
        }
        // Stability meter (FRUSTRACER_STAB=1): quantifies temporal
        // instability of the upscaled output — hold the camera still and a
        // healthy pipeline trends toward ~0; "dancing" holds a high mean.
        // Reads back the GPU output synchronously; diagnostics only.
        if stab_on && upscaled && rendered {
            stab_n = stab_n.wrapping_add(1);
            if stab_n % 15 == 0 {
                let cap = if dlss_on { gpu.read_rr_output() } else { gpu.read_xess_output() };
                if let Ok(px) = cap {
                    if let Some(prev) = &stab_prev {
                        if prev.len() == px.len() {
                            let sum: u64 = px
                                .iter()
                                .zip(prev)
                                .map(|(a, b)| {
                                    let d = |s: u32| {
                                        ((a >> s) & 0xff).abs_diff((b >> s) & 0xff) as u64
                                    };
                                    d(16) + d(8) + d(0)
                                })
                                .sum();
                            eprintln!(
                                "stab: mean |Δ| {:.2}/255 over 15 frames (window-res output; render {}x{})",
                                sum as f64 / (px.len() * 3) as f64,
                                rw,
                                rh
                            );
                        }
                    }
                    stab_prev = Some(px);
                }
            }
        }
        fps_frames += 1;
        if (now - fps_t).as_secs_f64() >= 1.0 {
            fps = fps_frames as f64 / (now - fps_t).as_secs_f64();
            fps_frames = 0;
            fps_t = now;
        }

        if edges.screenshot {
            if dlss_on {
                // The denoised image exists only on the GPU — read the RR
                // output back and tonemap it (same curve, 1 spp). On failure
                // fall back to a fresh CPU resolve of the noisy input.
                match gpu.read_rr_output() {
                    Ok(px) => present.copy_from_slice(&px),
                    Err(e) => {
                        eprintln!("screenshot: RR readback failed ({e}); saving noisy 1-spp resolve");
                        render::resolve(&accum, &info, 1, false, &mut present, rw, rh, W, H);
                    }
                }
            } else if xess_on && xess_oidn != XessOidn::Post {
                // Same story as DLSS: the upscaled image lives only on the
                // GPU. (POST placement presents via the CPU path, so its
                // present buffer is already current — plain save.)
                match gpu.read_xess_output() {
                    Ok(px) => present.copy_from_slice(&px),
                    Err(e) => {
                        eprintln!("screenshot: XeSS readback failed ({e}); saving noisy 1-spp resolve");
                        render::resolve(&accum, &info, 1, false, &mut present, rw, rh, W, H);
                    }
                }
            } else if use_gpu_tone {
                // The present buffer is stale in GPU-tonemap mode; resolve
                // fresh for the screenshot.
                render::resolve(&accum, &info, frame.max(1), false, &mut present, rw, rh, W, H);
            }
            let name = format!("screenshot_{shot}.png");
            save_png(&name, &present, W, H);
            eprintln!("saved {name}");
            shot += 1;
        }
        if edges.verify {
            eprintln!("verifying current view...");
            let vstats = Stats::default();
            // Cache-free on purpose: an independent ground-truth oracle.
            let rep = render::verify(scene, bvh, &cam.basis(W, H), base_q, W, H, &vstats, None, None);
            eprintln!(
                "verify ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e} -> {}",
                rep.pixels,
                rep.false_sky,
                rep.overshoot,
                rep.hybrid_extra,
                rep.max_rel_err,
                if rep.ok() { "OK" } else { "MISMATCH" }
            );
        }

        if (now - last_title).as_secs_f64() > 0.25 {
            last_title = now;
            let coarse_px = stats.coarse_pixels.load(Relaxed);
            let coarse = if use_budget {
                format!(" | cap {} coarse {}%", depth_est.floor() as u32, coarse_px * 100 / (rw * rh) as u64)
            } else {
                String::new()
            };
            let _ = window.set_title(&format!(
                "frustracer | {:.0} fps | {}{}{} | {}x{} | {:.1} ms | {} | quality {}{}{}{}{}{}",
                fps,
                if hybrid { "hybrid" } else { "plain" },
                if !dlss_on && !xess_on && hybrid && dynamic { "+dyn" } else { "" },
                if dlss_on {
                    if dlss_drs {
                        format!(" | DLSS: dyn {}% + RR", rh * 100 / H)
                    } else if (drw, drh) == (W, H) {
                        // Native render res means the optimal-settings query
                        // fell back to DLAA; anything smaller is Quality mode.
                        " | DLSS: DLAA + RR".to_string()
                    } else {
                        " | DLSS: Quality + RR".to_string()
                    }
                } else if xess_on {
                    format!(
                        " | XeSS: dyn {}%{}",
                        rh * 100 / H,
                        match (xess_oidn, oidn_ctx.as_ref()) {
                            (XessOidn::Pre, Some(c)) =>
                                format!(" +OIDN(pre) {} {:.1} ms", c.device_desc, c.last_ms),
                            (XessOidn::Post, Some(c)) =>
                                format!(" +OIDN(post) {} {:.1} ms", c.device_desc, c.last_ms),
                            _ => String::new(),
                        }
                    )
                } else if oidn_on {
                    match oidn_ctx.as_ref() {
                        Some(c) => format!(
                            " | OIDN{}: {} {:.1} ms",
                            if oidn_temporal { "+T" } else { "" },
                            c.device_desc,
                            c.last_ms
                        ),
                        None => " | OIDN".to_string(),
                    }
                } else {
                    " | DLSS: off".to_string()
                },
                rw,
                rh,
                last_ms,
                if dlss_on || xess_on {
                    "1 spp".to_string()
                } else {
                    format!("{} spp", frame.min(MAX_SAMPLES))
                },
                preset,
                coarse,
                if dlss_on || xess_on {
                    ""
                } else {
                    ["", " | hemi-AO", " | hemi-GI", " | hemi-GI+shafts"][bounce_mode as usize]
                },
                if use_gpu_tone && !dlss_on && !xess_on { " | gpu-tone" } else { "" },
                if overlay_on { " | overlay" } else { "" },
                if !dlss_on && !xess_on && frame >= MAX_SAMPLES { " | converged" } else { "" },
            ));
        }
        if (now - last_stats).as_secs_f64() > 1.0 && frame <= MAX_SAMPLES && frame > 0 {
            last_stats = now;
            eprintln!(
                "[{}] {:.1} ms | {}{}{}",
                if hybrid { "hybrid" } else { "plain" },
                last_ms,
                stats.summary_line(),
                if xess_on || (dlss_on && dlss_drs) {
                    format!(
                        " | {} {}x{} ({}%)",
                        if xess_on { "xess" } else { "dlss-drs" },
                        rw,
                        rh,
                        rh * 100 / H
                    )
                } else {
                    String::new()
                },
                if (oidn_on || (xess_on && xess_oidn == XessOidn::Pre)) && oidn_temporal {
                    format!(
                        " | hist {:.1} ms acc {} rej {} coarse {}/{} L {:.0}..{:.0}",
                        hist_ms,
                        last_hist.accepted,
                        last_hist.rejected,
                        last_hist.coarse_kept,
                        last_hist.coarse_reset,
                        last_hist.len_min,
                        last_hist.len_max
                    )
                } else {
                    String::new()
                }
            );
        }
    }
}

fn save_png(name: &str, present: &[u32], w: usize, h: usize) {
    let mut rgb = Vec::with_capacity(w * h * 3);
    for px in present {
        rgb.push((px >> 16) as u8);
        rgb.push((px >> 8) as u8);
        rgb.push(*px as u8);
    }
    if let Err(e) = image::save_buffer(name, &rgb, w as u32, h as u32, image::ColorType::Rgb8) {
        eprintln!("failed to save {name}: {e}");
    }
}
