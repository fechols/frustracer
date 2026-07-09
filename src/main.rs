mod bvh;
mod camera;
mod frustum;
mod overlay;
mod render;
mod scene;
mod shade;
mod stats;
mod temporal;

use camera::Camera;
use glam::Vec3A;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use render::FrameCtx;
use shade::Quality;
use stats::Stats;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::time::{Duration, Instant};

const W: usize = 1024;
const H: usize = 768;
const MAX_SAMPLES: u32 = 1024;
/// Frame budget for dynamic-resolution mode: 60 FPS minus resolve/present
/// headroom. Not a per-tile deadline: a log4-proportional controller turns the
/// previous frame's time against this target into a uniform quadtree depth cap
/// for the next frame; tiles reaching the cap unresolved become single
/// flat-shaded quads. Cost roughly quadruples per level, so
/// log4(budget/elapsed) reads "levels of headroom" directly.
const RENDER_BUDGET: Duration = Duration::from_millis(14);
/// Controller gain on the log4 error.
const DEPTH_GAIN: f32 = 0.6;
/// Max upward step per frame — creep up (>= 3 frames per level)...
const STEP_UP_MAX: f32 = 0.4;
/// ...but drop more than a full level in one step after a blown frame.
const STEP_DOWN_MAX: f32 = 1.5;

fn main() {
    let mut obj: Option<String> = None;
    let mut check = false;
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "--check" => check = true,
            "--help" | "-h" => {
                eprintln!("usage: frustracer [model.obj] [--check]");
                eprintln!("  --check  headless: verify hybrid vs reference, benchmark, write check.png");
                return;
            }
            _ => obj = Some(a),
        }
    }

    eprintln!("frustracer — loading scene...");
    let scene = match &obj {
        Some(p) => scene::load_obj_scene(p),
        None => scene::procedural_scene(),
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
        let code = run_check(&scene, &bvh);
        std::process::exit(code);
    }
    run_window(&scene, &bvh);
}

fn default_camera() -> Camera {
    Camera::look_at(
        Vec3A::new(11.0, 6.5, 13.0),
        Vec3A::new(0.0, 1.2, 0.0),
        55f32.to_radians(),
    )
}

/// Headless end-to-end check: correctness counters (must be 0), an A/B
/// benchmark of hybrid vs plain, and a rendered check.png.
fn run_check(scene: &scene::Scene, bvh: &bvh::Bvh) -> i32 {
    let (rw, rh) = (800usize, 600usize);
    let cam0 = default_camera();
    let cam = cam0.basis(rw, rh);
    let q = Quality::preset(2);
    let stats = Stats::default();

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
    let capped_ok = rep_c.ok() && rep_c.coarse > 0;
    if rep_c.coarse == 0 {
        eprintln!("verify capped d=4: expected coarse pixels, found none — capped path not exercised");
    }

    let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();

    const BENCH_FRAMES: u32 = 8;
    for (label, hybrid) in [("hybrid", true), ("plain ", false)] {
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
        };
        let t = Instant::now();
        for _ in 0..BENCH_FRAMES {
            render::render_frame(&ctx, hybrid);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / BENCH_FRAMES as f64;
        eprintln!("{label}: {ms:6.1} ms/frame | per {BENCH_FRAMES} frames: {}", stats.summary_line());
        if hybrid {
            // Save the hybrid image while its buffers are fresh.
            let mut present = vec![0u32; rw * rh];
            render::resolve(&accum, &info, 1, false, &mut present, rw, rh, rw, rh);
            save_png("check.png", &present, rw, rh);
        }
    }
    eprintln!("wrote check.png");

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
            "verify temporal {label} ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e} | seeds {seeds} sky-tiles {sky} tests {tests} coarse px {}",
            rep.pixels, rep.false_sky, rep.overshoot, rep.hybrid_extra, rep.max_rel_err, rep.coarse
        );
        let mut ok = rep.ok();
        if want_seeds && seeds == 0 {
            eprintln!("verify temporal {label}: expected temporal seeds > 0 — the path didn't fire");
            ok = false;
        }
        if want_sky && sky == 0 {
            eprintln!("verify temporal {label}: expected temporal sky-tiles > 0 — the sky path didn't fire");
            ok = false;
        }
        if max_depth.is_some() && rep.coarse == 0 {
            eprintln!("verify temporal {label}: expected coarse pixels, found none — capped path not exercised");
            ok = false;
        }
        temporal_ok &= ok;
    };
    // T1: identical basis — the static-accumulation fast path. Every sky tile
    // must come from the cache and at least one node must seed.
    temporal_pass("static", &cam, None, true, true);
    // T2: pure forward dolly. Seeds must fire (the root, at minimum: its
    // segment is provably inside the old root under an interior translation).
    // Sky reuse is NOT asserted — a same-position old sky cell genuinely
    // cannot cover the translated tile's infinite tail (it exits through
    // their shared plane), and shallower old cells aren't sky; temporal sky
    // is structurally a static-camera win.
    let mut cam_b = cam0;
    cam_b.pos += cam0.forward() * (0.02 * scene.diag);
    let basis_b = cam_b.basis(rw, rh);
    temporal_pass("dolly", &basis_b, None, true, false);
    // T3: the same dolly through the depth-capped driver.
    temporal_pass("dolly capped d=4", &basis_b, Some(4), false, false);
    // T4: translate + rotate — containment legitimately fires sparsely under
    // rotation, so only correctness is asserted.
    let mut cam_c = cam_b;
    cam_c.yaw += 0.05;
    temporal_pass("dolly+yaw", &cam_c.basis(rw, rh), None, false, false);

    // Informational A/B: the same static frame cold vs seeded (this is the
    // accumulation-frame path). Not gated — the win is scene-dependent.
    for (label, prev) in [("cold", None), ("warm", Some((&tcache, cam)))] {
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
            tcache_prev: prev,
        };
        let t = Instant::now();
        render::render_frame(&ctx, true);
        eprintln!(
            "temporal A/B {label}: {:5.1} ms | frustum nodes {} | queries {} | temporal: seeds {} sky {} tests {}",
            t.elapsed().as_secs_f64() * 1000.0,
            stats.frustum_nodes.load(Relaxed),
            stats.frustum_queries.load(Relaxed),
            stats.temporal_seeds.load(Relaxed),
            stats.temporal_sky_tiles.load(Relaxed),
            stats.temporal_tests.load(Relaxed),
        );
    }

    if rep.ok() && capped_ok && temporal_ok {
        eprintln!("CHECK PASSED");
        0
    } else {
        eprintln!("CHECK FAILED: hybrid image diverges from reference");
        1
    }
}

fn run_window(scene: &scene::Scene, bvh: &bvh::Bvh) {
    let mut window = Window::new(
        "frustracer — R: hybrid/plain  T: dynamic-res  O: overlay  1-3: quality  C: verify  P: screenshot",
        W,
        H,
        WindowOptions::default(),
    )
    .expect("failed to open window");

    let mut cam = default_camera();
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

    let mut frame: u32 = 0;
    let mut hybrid = true;
    let mut dynamic = true;
    let mut overlay_on = false;
    let mut preset = 2u32;
    let mut prev_mouse: Option<(f32, f32)> = None;
    let mut prev_rw = W;
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
    let mut last_ms = 0.0f64;
    let mut shot = 0u32;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let dt = (now - last).as_secs_f32().min(0.1);
        last = now;

        let moved = handle_input(&window, &mut cam, dt, scene.diag, &mut prev_mouse);
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            hybrid = !hybrid;
            frame = 0;
        }
        if window.is_key_pressed(Key::T, KeyRepeat::No) {
            dynamic = !dynamic;
            frame = 0;
        }
        if window.is_key_pressed(Key::O, KeyRepeat::No) {
            overlay_on = !overlay_on;
        }
        for (k, p) in [(Key::Key1, 1), (Key::Key2, 2), (Key::Key3, 3)] {
            if window.is_key_pressed(k, KeyRepeat::No) {
                preset = p;
                frame = 0;
            }
        }
        if moved {
            frame = 0;
        }

        // Cheap while moving, converge while still. Dynamic-res mode keeps
        // full resolution buffers and full quality — the estimated depth cap
        // floats the effective resolution instead.
        let use_budget = moved && hybrid && dynamic;
        let (rw, rh) = if moved && !use_budget { (W / 2, H / 2) } else { (W, H) };
        if rw != prev_rw {
            frame = 0;
            prev_rw = rw;
        }
        if use_budget != prev_budget {
            frame = 0; // budget frames hold flat quads — never accumulate onto them
            prev_budget = use_budget;
        }
        let base_q = Quality::preset(preset);
        let q = if moved && !use_budget { base_q.while_moving() } else { base_q };

        if frame < MAX_SAMPLES {
            stats.clear();
            let basis = cam.basis(rw, rh);
            // Budget (moving) frames are full-res, so the cache stays live
            // through motion and across the moving→static transition.
            let temporal_on = hybrid && rw == W;
            let (tcache_cur, tcache_prev) = if temporal_on {
                tcaches[tcur].clear();
                (
                    Some(&tcaches[tcur]),
                    if tprev_ok { Some((&tcaches[tcur ^ 1], tprev_basis)) } else { None },
                )
            } else {
                (None, None)
            };
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
                tcache_cur,
                tcache_prev,
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
        window
            .update_with_buffer(&present, W, H)
            .expect("window update failed");

        if window.is_key_pressed(Key::P, KeyRepeat::No) {
            let name = format!("screenshot_{shot}.png");
            save_png(&name, &present, W, H);
            eprintln!("saved {name}");
            shot += 1;
        }
        if window.is_key_pressed(Key::C, KeyRepeat::No) {
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
            window.set_title(&format!(
                "frustracer | {}{} | {}x{} | {:.1} ms | {} spp | quality {}{}{}{}",
                if hybrid { "hybrid" } else { "plain" },
                if hybrid && dynamic { "+dyn" } else { "" },
                rw,
                rh,
                last_ms,
                frame.min(MAX_SAMPLES),
                preset,
                coarse,
                if overlay_on { " | overlay" } else { "" },
                if frame >= MAX_SAMPLES { " | converged" } else { "" },
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

fn handle_input(
    window: &Window,
    cam: &mut Camera,
    dt: f32,
    diag: f32,
    prev_mouse: &mut Option<(f32, f32)>,
) -> bool {
    let mut moved = false;

    let mut speed = diag * 0.25 * dt;
    if window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift) {
        speed *= 4.0;
    }
    let f = cam.forward();
    let r = f.cross(Vec3A::Y).normalize();
    let mut delta = Vec3A::ZERO;
    if window.is_key_down(Key::W) {
        delta += f;
    }
    if window.is_key_down(Key::S) {
        delta -= f;
    }
    if window.is_key_down(Key::D) {
        delta += r;
    }
    if window.is_key_down(Key::A) {
        delta -= r;
    }
    if window.is_key_down(Key::E) || window.is_key_down(Key::Space) {
        delta += Vec3A::Y;
    }
    if window.is_key_down(Key::Q) {
        delta -= Vec3A::Y;
    }
    if delta != Vec3A::ZERO {
        cam.pos += delta.normalize() * speed;
        moved = true;
    }

    // Hold left mouse to look.
    let mpos = window.get_mouse_pos(MouseMode::Pass);
    if window.get_mouse_down(MouseButton::Left) {
        if let (Some((x, y)), Some((px, py))) = (mpos, *prev_mouse) {
            let (dx, dy) = (x - px, y - py);
            if dx != 0.0 || dy != 0.0 {
                cam.yaw += dx * 0.004;
                cam.pitch = (cam.pitch - dy * 0.004).clamp(-1.5, 1.5);
                moved = true;
            }
        }
    }
    *prev_mouse = mpos;

    moved
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
