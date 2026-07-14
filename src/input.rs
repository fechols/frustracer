//! SDL2 input: the main-thread event drain — toggles from KeyDown edges
//! (repeat filtered), quit, and window size changes. Camera movement/look
//! deliberately does NOT live here: SDL state only updates at pump time and
//! the main thread blocks for whole traces, so flight is integrated at
//! 500 Hz wall-clock on the flycam thread (src/flycam.rs) instead.

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::EventPump;

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
    pub toggle_temporal: bool, // M (OIDN temporal reprojection)
    pub toggle_bounce: bool,   // H (hemisphere frustum bounces)
    pub verify: bool,          // C
    pub screenshot: bool,      // P
    pub cycle_spp: bool,       // U (samples per pixel: 1 -> 2 -> 4 -> 8 -> 1)
    pub quality: Option<u32>,  // 1/2/3
    pub toggle_fullscreen: bool, // F11 (borderless desktop fullscreen)
    /// Newest window client size from this frame's SizeChanged events
    /// (maximize, restore, fullscreen, drag — the last event in the drain
    /// wins). The consumer debounces and commits via `drawable_size()`.
    pub size_changed: Option<(u32, u32)>,
}

pub struct Input {
    pump: EventPump,
}

impl Input {
    pub fn new(sdl: &sdl2::Sdl) -> Result<Self, String> {
        Ok(Self { pump: sdl.event_pump()? })
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
                    Keycode::M => e.toggle_temporal = true,
                    Keycode::H => e.toggle_bounce = true,
                    Keycode::C => e.verify = true,
                    Keycode::P => e.screenshot = true,
                    Keycode::U => e.cycle_spp = true,
                    Keycode::Num1 | Keycode::Kp1 => e.quality = Some(1),
                    Keycode::Num2 | Keycode::Kp2 => e.quality = Some(2),
                    Keycode::Num3 | Keycode::Kp3 => e.quality = Some(3),
                    Keycode::F11 => e.toggle_fullscreen = true,
                    _ => {}
                },
                Event::Window {
                    win_event: sdl2::event::WindowEvent::SizeChanged(w, h), ..
                } => e.size_changed = Some((w.max(0) as u32, h.max(0) as u32)),
                _ => {}
            }
        }
        e
    }
}
