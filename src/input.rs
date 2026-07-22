//! SDL3 input: the main-thread event drain — toggles from KeyDown edges
//! (repeat filtered), quit, and window size changes. Camera movement/look
//! deliberately does NOT live here: SDL state only updates at pump time and
//! the main thread blocks for whole traces, so flight is integrated at
//! 500 Hz wall-clock on the flycam thread (src/flycam.rs) instead.

use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::EventPump;

/// One-frame key edges (KeyDown, no repeat) plus quit.
#[derive(Default)]
pub struct Edges {
    pub quit: bool,
    pub toggle_hybrid: bool,   // R
    pub toggle_dynamic: bool,  // T
    pub toggle_overlay: bool,  // O
    pub toggle_gpu_tone: bool, // B
    pub toggle_dlss: bool,     // G
    pub toggle_xess: bool,     // X (XeSS-SR dynamic-res upscaling)
    pub toggle_fsr: bool,      // K (FSR Ray Regeneration + FSR4; F belongs to DXR)
    pub toggle_oidn: bool,     // N (Open Image Denoise)
    pub toggle_nppd: bool,     // J (NPPD neural denoiser)
    pub toggle_dxr: bool,      // F (DXR DispatchRays pipeline)
    pub cycle_mode: bool,      // SPACE (render mode: CPU -> GPU wavefront -> DXR)
    pub toggle_temporal: bool, // M (OIDN temporal reprojection)
    pub toggle_bounce: bool,   // H (hemisphere frustum bounces)
    pub toggle_height: bool,   // V (heightfield relief vs normal-mapped)
    pub verify: bool,          // C
    pub screenshot: bool,      // P
    pub cycle_spp: bool,       // U (samples per pixel: 1 -> 2 -> 4 -> 8 -> 1)
    pub capture_frustum: bool, // Y (freeze the current view's quadtree frustums as scene geometry)
    pub clear_frustum: bool,   // Z (remove the frozen frustum snapshot)
    pub quality: Option<u32>,  // 1/2/3
    pub toggle_fullscreen: bool, // F11 (borderless desktop fullscreen)
    /// Newest window client size from this frame's Resized/PixelSizeChanged
    /// events (maximize, restore, fullscreen, drag — the last event in the
    /// drain wins). The consumer debounces and commits via `size_in_pixels()`.
    pub size_changed: Option<(u32, u32)>,
    /// The window may now be on a different monitor — `DisplayChanged` (SDL's
    /// own "you moved to another display") or `Moved` (a window can straddle
    /// two monitors and change which one owns it without SDL firing
    /// `DisplayChanged`). Either way the HDR capabilities under us may have
    /// changed, so the consumer re-probes the display.
    ///
    /// Note this canNOT catch the user toggling Windows HDR on the monitor the
    /// window is already sitting on — no window event fires for that at all.
    /// The consumer's periodic re-probe is what covers it.
    pub display_changed: bool,
}

pub struct Input {
    pump: EventPump,
}

impl Input {
    pub fn new(sdl: &sdl3::Sdl) -> Result<Self, String> {
        Ok(Self { pump: sdl.event_pump().map_err(|e| e.to_string())? })
    }

    /// Drain the event queue, collecting edges. Call once per frame.
    pub fn poll(&mut self) -> Edges {
        let mut e = Edges::default();
        for ev in self.pump.poll_iter() {
            match ev {
                Event::Quit { .. } => e.quit = true,
                Event::KeyDown { keycode: Some(k), repeat: false, .. } => match k {
                    Keycode::Escape => e.quit = true,
                    Keycode::R => e.toggle_hybrid = true,
                    Keycode::T => e.toggle_dynamic = true,
                    Keycode::O => e.toggle_overlay = true,
                    Keycode::B => e.toggle_gpu_tone = true,
                    Keycode::G => e.toggle_dlss = true,
                    Keycode::X => e.toggle_xess = true,
                    Keycode::K => e.toggle_fsr = true,
                    Keycode::N => e.toggle_oidn = true,
                    Keycode::J => e.toggle_nppd = true,
                    Keycode::F => e.toggle_dxr = true,
                    Keycode::Space => e.cycle_mode = true,
                    Keycode::M => e.toggle_temporal = true,
                    Keycode::H => e.toggle_bounce = true,
                    Keycode::V => e.toggle_height = true,
                    Keycode::C => e.verify = true,
                    Keycode::P => e.screenshot = true,
                    Keycode::U => e.cycle_spp = true,
                    Keycode::Y => e.capture_frustum = true,
                    Keycode::Z => e.clear_frustum = true,
                    Keycode::_1 | Keycode::Kp1 => e.quality = Some(1),
                    Keycode::_2 | Keycode::Kp2 => e.quality = Some(2),
                    Keycode::_3 | Keycode::Kp3 => e.quality = Some(3),
                    Keycode::F11 => e.toggle_fullscreen = true,
                    _ => {}
                },
                // SDL3 split SDL2's SizeChanged into Resized (logical) and
                // PixelSizeChanged (physical). Arm on either — this edge only
                // starts the settle debounce; the authoritative size is read
                // from `size_in_pixels()` at commit time.
                Event::Window {
                    win_event:
                        sdl3::event::WindowEvent::Resized(w, h)
                        | sdl3::event::WindowEvent::PixelSizeChanged(w, h),
                    ..
                } => e.size_changed = Some((w.max(0) as u32, h.max(0) as u32)),
                Event::Window {
                    win_event:
                        sdl3::event::WindowEvent::DisplayChanged(_)
                        | sdl3::event::WindowEvent::Moved(_, _),
                    ..
                } => e.display_changed = true,
                _ => {}
            }
        }
        e
    }
}
