//! SDL3 input: the main-thread event drain — toggles from KeyDown edges
//! (repeat filtered), quit, and window size changes. Camera movement/look
//! deliberately does NOT live here: SDL state only updates at pump time and
//! the main thread blocks for whole traces, so flight is integrated at
//! 500 Hz wall-clock on the flycam thread (src/flycam.rs) instead.
//!
//! Two modes, switched by the pause menu (`poll`'s `menu` argument): menu
//! CLOSED = the historical toggle-edge drain; menu OPEN = window-level
//! events (quit/resize/display/F11) and ESC keep their edges, and EVERY
//! other event is translated + dispatched into the Slint menu
//! (hud/events.rs) — toggle keys structurally cannot fire under the menu.
//! Note this only silences SDL-side toggles: the flycam thread reads raw OS
//! key state and is separately paused by the menu's state machine.

use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::EventPump;

/// One-frame key edges (KeyDown, no repeat) plus quit.
#[derive(Default)]
pub struct Edges {
    pub quit: bool,
    /// ESC — opens the pause menu (closed) / back / close (open). The menu
    /// path is what quit used to be; window-X still sets `quit`.
    pub esc: bool,
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
    pub toggle_hud: bool,      // F1 (compass/clock/keymap overlay)
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

impl Edges {
    /// OR another frame's edges in (the pause menu synthesizes key edges so
    /// its rows ride the exact key-handler reset semantics).
    pub fn merge(&mut self, o: &Edges) {
        self.quit |= o.quit;
        self.esc |= o.esc;
        self.toggle_hybrid |= o.toggle_hybrid;
        self.toggle_dynamic |= o.toggle_dynamic;
        self.toggle_overlay |= o.toggle_overlay;
        self.toggle_gpu_tone |= o.toggle_gpu_tone;
        self.toggle_dlss |= o.toggle_dlss;
        self.toggle_xess |= o.toggle_xess;
        self.toggle_fsr |= o.toggle_fsr;
        self.toggle_oidn |= o.toggle_oidn;
        self.toggle_nppd |= o.toggle_nppd;
        self.toggle_dxr |= o.toggle_dxr;
        self.cycle_mode |= o.cycle_mode;
        self.toggle_temporal |= o.toggle_temporal;
        self.toggle_bounce |= o.toggle_bounce;
        self.toggle_height |= o.toggle_height;
        self.verify |= o.verify;
        self.screenshot |= o.screenshot;
        self.cycle_spp |= o.cycle_spp;
        self.capture_frustum |= o.capture_frustum;
        self.clear_frustum |= o.clear_frustum;
        self.quality = self.quality.or(o.quality);
        self.toggle_hud |= o.toggle_hud;
        self.toggle_fullscreen |= o.toggle_fullscreen;
        self.size_changed = self.size_changed.or(o.size_changed);
        self.display_changed |= o.display_changed;
    }
}

pub struct Input {
    pump: EventPump,
}

impl Input {
    pub fn new(sdl: &sdl3::Sdl) -> Result<Self, String> {
        Ok(Self { pump: sdl.event_pump().map_err(|e| e.to_string())? })
    }

    /// Drain the event queue, collecting edges. Call once per frame.
    /// `menu` = Some while the pause menu is open — see the module header.
    pub fn poll(&mut self, menu: Option<&crate::hud::Hud>) -> Edges {
        let mut e = Edges::default();
        for ev in self.pump.poll_iter() {
            // Window-level events keep their edges in BOTH modes: quitting,
            // resizing, and monitor changes must work under an open menu.
            match &ev {
                Event::Quit { .. } => {
                    e.quit = true;
                    continue;
                }
                // SDL3 split SDL2's SizeChanged into Resized (logical) and
                // PixelSizeChanged (physical). Arm on either — this edge only
                // starts the settle debounce; the authoritative size is read
                // from `size_in_pixels()` at commit time.
                Event::Window {
                    win_event:
                        sdl3::event::WindowEvent::Resized(w, h)
                        | sdl3::event::WindowEvent::PixelSizeChanged(w, h),
                    ..
                } => {
                    e.size_changed = Some(((*w).max(0) as u32, (*h).max(0) as u32));
                    continue;
                }
                Event::Window {
                    win_event:
                        sdl3::event::WindowEvent::DisplayChanged(_)
                        | sdl3::event::WindowEvent::Moved(_, _),
                    ..
                } => {
                    e.display_changed = true;
                    continue;
                }
                _ => {}
            }
            if let Some(hud) = menu {
                // Menu open: ESC/F11 stay ours, everything else goes to Slint.
                match &ev {
                    Event::KeyDown { keycode: Some(Keycode::Escape), repeat: false, .. } => {
                        e.esc = true
                    }
                    Event::KeyDown { keycode: Some(Keycode::F11), repeat: false, .. } => {
                        e.toggle_fullscreen = true
                    }
                    _ => {
                        if matches!(
                            &ev,
                            Event::MouseButtonDown { .. } | Event::MouseButtonUp { .. }
                        ) && std::env::var_os("FRUSTRACER_HUD_STATS").is_some()
                        {
                            eprintln!("hud-input: {ev:?}");
                        }
                        crate::hud::events::forward(hud.slint_window(), &ev)
                    }
                }
                continue;
            }
            if let Event::KeyDown { keycode: Some(k), repeat: false, .. } = ev {
                match k {
                    Keycode::Escape => e.esc = true,
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
                    Keycode::F1 => e.toggle_hud = true,
                    Keycode::F11 => e.toggle_fullscreen = true,
                    _ => {}
                }
            }
        }
        e
    }
}
