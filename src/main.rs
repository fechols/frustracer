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
mod overlay;
mod render;
mod scene;
mod shade;
mod shaft;
mod sphcell;
mod stats;
mod temporal;

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
/// for the next frame; tiles reaching the cap unresolved become single
/// flat-shaded quads. Cost roughly quadruples per level, so
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
    };
    let mut check_dlss = false;
    let mut dlss_dump = false;
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
                eprintln!("usage: frustracer [model.obj] [--stress <n>] [--check] [--check-dlss] [--dlss-dump] [--no-dlss] [--gpu-debug] [--sl-path <dir>]");
                eprintln!("  --stress <n>  procedural stress field of n objects (perf test; composes with --check)");
                eprintln!("  --check       headless: verify hybrid vs reference, benchmark, write check.png");
                eprintln!("  --check-dlss  headless: G-buffer MV/depth/matrix self-test (no GPU needed)");
                eprintln!("  --dlss-dump   --check-dlss plus G-buffer PNG dumps (albedo/normal/misc/mv)");
                eprintln!("  --no-dlss     skip Streamline/DLSS; plain D3D12 presentation");
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
    let (near, far) = dlss::near_far(scene.diag);
    let q = Quality {
        shadow_samples: 1,
        ao_samples: 1,
        reflections: true,
        fb: shade::FrustumBounce::OFF,
    };
    let stats = Stats::default();

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
    // with A as its previous frame. Zero jitter so samples sit on pixel
    // centers — the reconstruction in the self-test assumes centers. Run
    // once at the native test res and once at the Quality-mode render res
    // stand-in (odd width — also exercises odd-dim quadtree splits), since
    // the interactive DLSS path now traces at a sub-native resolution.
    let mv_pass = |rw: usize, rh: usize, dump: bool| -> bool {
        eprintln!("MV/depth/matrix self-test at {rw}x{rh}:");
        let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
        let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
        let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
        let basis_a = cam0.basis(rw, rh);
        let ga = dlss::GBufs::new(rw, rh);
        let gb = dlss::GBufs::new(rw, rh);
        let render_dlss_frame = |g: &dlss::GBufs, basis: camera::CamBasis, prev: Option<camera::CamBasis>, frame: u32| {
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

        if dump {
            dlss::dump_gbufs(&gb, "dlss_gbuf", far);
        }
        ok
    };
    let mv_native_ok = mv_pass(800, 600, dump);
    let (qw, qh) = dlss::headless_render_res(800, 600);
    let mv_quality_ok = mv_pass(qw, qh, false);

    if halton_ok && mv_native_ok && mv_quality_ok {
        eprintln!("DLSS CHECK PASSED");
        0
    } else {
        eprintln!("DLSS CHECK FAILED");
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

    let rep = render::verify(scene, bvh, &cam, q, rw, rh, &stats, None, None);
    eprintln!(
        "verify full-depth ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e}",
        rep.pixels, rep.false_sky, rep.overshoot, rep.hybrid_extra, rep.max_rel_err
    );
    // The capped driver is the uncapped one with an extra depth check (a cap
    // past the leaf depth is bit-identical by construction), so verify it at a
    // cap that actually flat-fills: every non-coarse pixel must still match
    // the reference exactly, and coarse pixels must exist (deterministic —
    // no wall clock involved).
    let rep_c = render::verify(scene, bvh, &cam, q, rw, rh, &stats, Some(4), None);
    eprintln!(
        "verify capped d=4 ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e} | coarse px {}",
        rep_c.pixels, rep_c.false_sky, rep_c.overshoot, rep_c.hybrid_extra, rep_c.max_rel_err, rep_c.coarse
    );
    let capped_ok = rep_c.ok() && (!structural || rep_c.coarse > 0);
    if structural && rep_c.coarse == 0 {
        eprintln!("verify capped d=4: expected coarse pixels, found none — capped path not exercised");
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
                            &hemi::BOUNCE_Q,
                            &mut rng,
                            sun,
                            1,
                            &mut lsr,
                            None,
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

    if sph_ok && hemi_ok && gi_ok && shaft_ok && rep.ok() && capped_ok && temporal_ok {
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
            "frustracer — R: hybrid/plain  T: dynamic-res  O: overlay  B: gpu-tonemap  H: hemi-bounce  1-3: quality  C: verify  P: screenshot",
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
    let gbufs = dlss::GBufs::new(drw, drh);
    let mut dlss_idx: u32 = 0;
    let mut dlss_prev: Option<dlss::DlssPrev> = None;
    let mut dlss_reset = true;
    let (dlss_near, dlss_far) = dlss::near_far(scene.diag);
    if dlss_on {
        eprintln!("dlss: Ray Reconstruction ON (G toggles)");
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
                eprintln!("dynamic-res is a no-op in DLSS mode (fixed render resolution)");
            } else {
                dynamic = !dynamic;
                frame = 0;
            }
        }
        if edges.toggle_overlay {
            if dlss_on {
                eprintln!("overlay unavailable in DLSS mode (lives in the CPU resolve)");
            } else {
                overlay_on = !overlay_on;
            }
        }
        if edges.toggle_gpu_tone {
            gpu_tonemap = !gpu_tonemap;
            eprintln!("tonemap: {}", if gpu_tonemap { "GPU" } else { "CPU" });
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
                eprintln!("dlss: Ray Reconstruction {}", if dlss_on { "ON" } else { "OFF" });
            } else {
                eprintln!("dlss: not available");
            }
        }
        if let Some(p) = edges.quality {
            preset = p;
            frame = 0;
            dlss_reset = true;
        }
        if moved {
            frame = 0;
        }

        // Cheap while moving, converge while still. Dynamic-res mode keeps
        // full resolution buffers and full quality — the estimated depth cap
        // floats the effective resolution instead. DLSS mode traces every
        // frame uncapped at RR's fixed render resolution (RR requires clean
        // per-pixel G-buffers) with frame-stationary quality.
        let use_budget = moved && hybrid && dynamic && !dlss_on;
        let (rw, rh) = if dlss_on {
            (drw, drh)
        } else if moved && !use_budget {
            (W / 2, H / 2)
        } else {
            (W, H)
        };
        if rw != prev_rw {
            frame = 0;
            prev_rw = rw;
        }
        if use_budget != prev_budget {
            frame = 0; // budget frames hold flat quads — never accumulate onto them
            prev_budget = use_budget;
        }
        let base_q = Quality::preset(preset);
        let mut q = if dlss_on {
            // Fixed cheap preset: RR wants frame-stationary noise statistics.
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
        if !dlss_on && !moved {
            q.fb.ao = bounce_mode == 1;
            q.fb.gi = bounce_mode >= 2;
            q.fb.shadows = bounce_mode == 3;
        }

        if dlss_on || frame < MAX_SAMPLES {
            stats.clear();
            let basis = cam.basis(rw, rh);
            // Budget (moving) frames are full-res, so the cache stays live
            // through motion and across the moving→static transition. In
            // DLSS mode "full participation res" is RR's render resolution;
            // the tprev_res check drops the cache whenever the resolution
            // changes (e.g. the G toggle), per the temporal invariant.
            let temporal_on = hybrid && (rw, rh) == if dlss_on { (drw, drh) } else { (W, H) };
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
                frame: if dlss_on { dlss_idx } else { frame },
                jitter: !dlss_on && frame > 0,
                rw,
                rh,
                accum: &accum,
                info: &info,
                tbuf: &tbuf,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur,
                tcache_prev,
                accumulate: !dlss_on,
                gbuf: if dlss_on { Some(&gbufs) } else { None },
                prev_cam: if dlss_on { dlss_prev.as_ref().map(|p| p.basis) } else { None },
                frame_jitter: if dlss_on { Some(dlss::jitter_for(dlss_idx)) } else { None },
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
            frame += 1;
        } else {
            std::thread::sleep(Duration::from_millis(8)); // converged — idle
        }

        // GPU tonemap consumes the raw HDR accumulation directly, but only
        // for full-res frames without the overlay — half-res upscale and the
        // overlay composite live in the CPU resolve.
        let use_gpu_tone = gpu_tonemap && rw == W && !(overlay_on && hybrid);
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
                    dlss_prev = Some(dlss::DlssPrev { basis: cam.basis(rw, rh), mats });
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
                if !dlss_on && hybrid && dynamic { "+dyn" } else { "" },
                if dlss_on {
                    // Native render res means the optimal-settings query fell
                    // back to DLAA; anything smaller is Quality mode.
                    if (drw, drh) == (W, H) { " | DLSS: DLAA + RR" } else { " | DLSS: Quality + RR" }
                } else {
                    " | DLSS: off"
                },
                rw,
                rh,
                last_ms,
                if dlss_on { "1 spp".to_string() } else { format!("{} spp", frame.min(MAX_SAMPLES)) },
                preset,
                coarse,
                if dlss_on {
                    ""
                } else {
                    ["", " | hemi-AO", " | hemi-GI", " | hemi-GI+shafts"][bounce_mode as usize]
                },
                if use_gpu_tone && !dlss_on { " | gpu-tone" } else { "" },
                if overlay_on { " | overlay" } else { "" },
                if !dlss_on && frame >= MAX_SAMPLES { " | converged" } else { "" },
            ));
        }
        if (now - last_stats).as_secs_f64() > 1.0 && frame <= MAX_SAMPLES && frame > 0 {
            last_stats = now;
            eprintln!(
                "[{}] {:.1} ms | {}",
                if hybrid { "hybrid" } else { "plain" },
                last_ms,
                stats.summary_line()
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
