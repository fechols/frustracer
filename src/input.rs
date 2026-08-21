//! SDL3 input: the per-frame event drain — toggles from KeyDown edges
//! (repeat filtered), quit, and window size changes. Camera movement/look
//! deliberately does NOT live here: SDL state only updates at pump time and
//! the main thread blocks for whole traces, so flight is integrated at
//! 500 Hz wall-clock on the flycam thread (src/flycam.rs) instead.
//!
//! ONE ROUTING TABLE, TWO PUMPS (B6b rung 4). `Edges::feed` is the per-event
//! body; `Input::poll` is the Windows session's loop over it, draining the
//! `EventPump` it owns on the main thread. The Linux window cannot do that —
//! its pump lives on the main thread and its session on the render thread
//! (`vk::present`'s header says why) — so `present::Win::pump` forwards every
//! drained `sdl3::event::Event` across and the render thread calls `feed` on
//! each. The table is textually one; only the loop around it is per window.
//!
//! Two modes, switched by the pause menu (`Mode`; `poll`'s `menu` argument): menu
//! CLOSED = the historical toggle-edge drain; menu OPEN = window-level
//! events (quit/resize/display/F11) and ESC keep their edges, and EVERY
//! other event is translated + dispatched into the Slint menu
//! (hud/events.rs) — toggle keys structurally cannot fire under the menu.
//! The open mode has two sub-modes, gated by `Hud::text_editing()`: while a
//! settings TextInput has focus, arrows/Enter forward to Slint (cursor
//! movement, `accepted`) and WASD arrives as TextInput characters; otherwise
//! arrows + WASD + Enter become the `menu_*` NAVIGATION edges (consumed, not
//! forwarded — the session drives Hud::nav/adjust/activate with them, the
//! same cursor the controller's D-pad moves). Key REPEAT is deliberately
//! allowed on the nav keys — free OS auto-repeat for held navigation.
//! Note this only silences SDL-side toggles: the flycam thread reads raw OS
//! key state and is separately paused by the menu's state machine.
//!
//! GAMEPAD MENU EDGES ride the same table on the Linux window: SDL's gamepad
//! subsystem is opened by `vk::present::Win` (never by the Windows session,
//! whose `pad.rs` reads XInput on the main thread instead), so the
//! `ControllerButtonDown` arms below fire there and are structurally dead on
//! Windows. Start toggles, South/East activate/back, the D-pad navigates —
//! `pad::PadEdges`'s vocabulary. Held-repeat is NOT here (SDL buttons do not
//! auto-repeat); that is a later slice.

use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::EventPump;

/// Which drain `Edges::feed` runs — the pause menu's two modes (module header).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Menu CLOSED: the toggle-edge drain.
    Closed,
    /// Menu OPEN: window-level events, ESC and F11 keep their edges; the nav
    /// keys become `menu_*` edges unless a text field is `editing`; everything
    /// else is handed to `forward` (the Slint window).
    Open { editing: bool },
}

impl Mode {
    /// The mode for this frame's drain: `menu` is `Some` while the pause menu
    /// is open — `Hud::text_editing` picks the sub-mode.
    pub fn of(menu: Option<&crate::hud::Hud>) -> Mode {
        match menu {
            Some(hud) => Mode::Open { editing: hud.text_editing() },
            None => Mode::Closed,
        }
    }
}

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
    pub toggle_waveviz: bool,  // I (FR_WAVEVIZ wave-footprint overlay; armed sessions only)
    pub verify: bool,          // C
    pub screenshot: bool,      // P
    pub cycle_spp: bool,       // U (samples per pixel: 1 -> 2 -> 4 -> 8 -> 1)
    pub capture_frustum: bool, // Y (freeze the current view's quadtree frustums as scene geometry)
    pub clear_frustum: bool,   // Z (remove the frozen frustum snapshot)
    pub quality: Option<u32>,  // 1/2/3
    pub toggle_hud: bool,      // F1 (compass/clock/keymap overlay)
    pub toggle_fullscreen: bool, // F11 (borderless desktop fullscreen)
    // Menu-open navigation edges (arrows/WASD/Enter while no text field has
    // focus — see the module header). The session ORs them with the pad's.
    pub menu_up: bool,
    pub menu_down: bool,
    pub menu_left: bool,
    pub menu_right: bool,
    pub menu_activate: bool,
    /// Gamepad Start (the Linux window's SDL gamepad; `pad.rs` on Windows):
    /// toggles the menu in either mode.
    pub menu_toggle: bool,
    /// Gamepad East ("B"): back / close while the menu is open.
    pub menu_back: bool,
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

/// The menu-open navigation keys (arrows + WASD + Enter).
fn nav_key(k: Keycode) -> bool {
    matches!(
        k,
        Keycode::Up
            | Keycode::Down
            | Keycode::Left
            | Keycode::Right
            | Keycode::W
            | Keycode::A
            | Keycode::S
            | Keycode::D
            | Keycode::Return
            | Keycode::KpEnter
    )
}

impl Edges {
    /// Route ONE event — the per-event body both windows share. `mode` is the
    /// menu's state for this frame; `forward` receives the events the open
    /// menu owns (the caller dispatches them into its Slint window).
    pub fn feed(&mut self, ev: &Event, mode: Mode, forward: &mut dyn FnMut(&Event)) {
        let e = self;
        {
            // Window-level events keep their edges in BOTH modes: quitting,
            // resizing, and monitor changes must work under an open menu.
            match ev {
                Event::Quit { .. } => {
                    e.quit = true;
                    return;
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
                    return;
                }
                Event::Window {
                    win_event:
                        sdl3::event::WindowEvent::DisplayChanged(_)
                        | sdl3::event::WindowEvent::Moved(_, _),
                    ..
                } => {
                    e.display_changed = true;
                    return;
                }
                // Gamepad Start toggles the menu in either mode (the Linux
                // window's pad; see the module header).
                Event::ControllerButtonDown { button: sdl3::gamepad::Button::Start, .. } => {
                    e.menu_toggle = true;
                    return;
                }
                _ => {}
            }
            if let Mode::Open { editing } = mode {
                // Menu open: ESC/F11 stay ours, nav keys become menu edges
                // unless a text field owns the keyboard, everything else
                // goes to Slint.
                match ev {
                    Event::KeyDown { keycode: Some(Keycode::Escape), repeat: false, .. } => {
                        e.esc = true
                    }
                    Event::KeyDown { keycode: Some(Keycode::F11), repeat: false, .. } => {
                        e.toggle_fullscreen = true
                    }
                    // Navigation (repeat ALLOWED — OS auto-repeat drives held
                    // keys). Consumed here, never also forwarded; the matching
                    // KeyUps are consumed below for symmetry. While a text
                    // field is focused these fall through to forward: arrows
                    // move the cursor, Enter fires `accepted`, and the "wasd"
                    // characters arrive as TextInput events (letter KeyDowns
                    // were always dropped by events.rs's `special`).
                    Event::KeyDown { keycode: Some(k), .. } if !editing && nav_key(*k) => {
                        match k {
                            Keycode::Up | Keycode::W => e.menu_up = true,
                            Keycode::Down | Keycode::S => e.menu_down = true,
                            Keycode::Left | Keycode::A => e.menu_left = true,
                            Keycode::Right | Keycode::D => e.menu_right = true,
                            _ => e.menu_activate = true, // Return | KpEnter
                        }
                    }
                    Event::KeyUp { keycode: Some(k), .. } if !editing && nav_key(*k) => {}
                    // Gamepad menu navigation — `pad::PadEdges`'s vocabulary.
                    Event::ControllerButtonDown { button, .. } => {
                        use sdl3::gamepad::Button as B;
                        match button {
                            B::DPadUp => e.menu_up = true,
                            B::DPadDown => e.menu_down = true,
                            B::DPadLeft => e.menu_left = true,
                            B::DPadRight => e.menu_right = true,
                            B::South => e.menu_activate = true,
                            B::East => e.menu_back = true,
                            _ => {}
                        }
                    }
                    _ => {
                        if matches!(
                            ev,
                            Event::MouseButtonDown { .. } | Event::MouseButtonUp { .. }
                        ) && std::env::var_os("FRUSTRACER_HUD_STATS").is_some()
                        {
                            eprintln!("hud-input: {ev:?}");
                        }
                        forward(ev)
                    }
                }
                return;
            }
            if let Event::KeyDown { keycode: Some(k), repeat: false, .. } = ev {
                match *k {
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
                    Keycode::I => e.toggle_waveviz = true,
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
    }
}

/// The Windows session's drain: owns the `EventPump` and loops `Edges::feed`
/// over it on the main thread. The Linux window has no `Input` — its pump is
/// `vk::present::Win` and its loop is in `window_frames` — hence dead there.
#[cfg_attr(not(windows), allow(dead_code))]
pub struct Input {
    pump: EventPump,
}

#[cfg_attr(not(windows), allow(dead_code))]
impl Input {
    pub fn new(sdl: &sdl3::Sdl) -> Result<Self, String> {
        Ok(Self { pump: sdl.event_pump().map_err(|e| e.to_string())? })
    }

    /// Drain the event queue, collecting edges. Call once per frame.
    /// `menu` = Some while the pause menu is open — see the module header.
    pub fn poll(&mut self, menu: Option<&crate::hud::Hud>) -> Edges {
        let mut e = Edges::default();
        for ev in self.pump.poll_iter() {
            // The mode is re-read per event: a forwarded click can focus or
            // blur a text field, and the next key in the same drain must see
            // that (the per-event `editing` read this always did).
            let mode = Mode::of(menu);
            let mut fwd = |ev: &Event| {
                if let Some(hud) = menu {
                    crate::hud::events::forward(hud.slint_window(), ev)
                }
            };
            e.feed(&ev, mode, &mut fwd);
        }
        e
    }
}

/// The routing table, gated without a window or a Slint instance: `feed` is
/// driven with hand-built `sdl3::event::Event` values (plain enums, every
/// field public) and the edges it sets — and does NOT set — are asserted in
/// both modes. Each positive has its negative, so the teeth are inherent.
pub fn self_test() -> Result<(), String> {
    use sdl3::event::WindowEvent as WE;
    use sdl3::gamepad::Button as B;
    use sdl3::keyboard::Mod;

    let key = |k: Keycode, repeat: bool| Event::KeyDown {
        timestamp: 0,
        window_id: 0,
        keycode: Some(k),
        scancode: None,
        keymod: Mod::empty(),
        repeat,
        which: 0,
        raw: 0,
    };
    let keyup = |k: Keycode| Event::KeyUp {
        timestamp: 0,
        window_id: 0,
        keycode: Some(k),
        scancode: None,
        keymod: Mod::empty(),
        repeat: false,
        which: 0,
        raw: 0,
    };
    let pad = |b: B| Event::ControllerButtonDown { timestamp: 0, which: 0, button: b };
    let quit = Event::Quit { timestamp: 0 };
    let resized = Event::Window { timestamp: 0, window_id: 0, win_event: WE::Resized(640, 360) };
    let moved = Event::Window { timestamp: 0, window_id: 0, win_event: WE::Moved(1, 2) };
    let motion = Event::MouseMotion {
        timestamp: 0,
        window_id: 0,
        which: 0,
        mousestate: sdl3::mouse::MouseState::from_sdl_state(0),
        x: 1.0,
        y: 2.0,
        xrel: 0.0,
        yrel: 0.0,
    };

    // Run one event through `feed`, counting forwards.
    let run = |ev: &Event, mode: Mode| -> (Edges, usize) {
        let mut e = Edges::default();
        let mut n = 0usize;
        e.feed(ev, mode, &mut |_| n += 1);
        (e, n)
    };
    let closed = Mode::Closed;
    let open = Mode::Open { editing: false };
    let editing = Mode::Open { editing: true };

    // CLOSED: the toggle edges, and repeat is filtered.
    let (e, n) = run(&key(Keycode::F1, false), closed);
    if !e.toggle_hud || n != 0 {
        return Err("closed: F1 must set toggle_hud and forward nothing".into());
    }
    let (e, _) = run(&key(Keycode::F1, true), closed);
    if e.toggle_hud {
        return Err("closed: a repeat F1 must not edge".into());
    }
    let (e, _) = run(&key(Keycode::Escape, false), closed);
    if !e.esc {
        return Err("closed: ESC must set esc".into());
    }
    let (e, _) = run(&key(Keycode::F11, false), closed);
    if !e.toggle_fullscreen {
        return Err("closed: F11 must set toggle_fullscreen".into());
    }
    let (e, _) = run(&key(Keycode::H, false), closed);
    if !e.toggle_bounce {
        return Err("closed: H must set toggle_bounce".into());
    }
    let (e, _) = run(&key(Keycode::Up, false), closed);
    if e.menu_up {
        return Err("closed: Up must NOT be a menu edge".into());
    }
    let (e, n) = run(&motion, closed);
    if n != 0 || e.menu_activate {
        return Err("closed: mouse motion must be dropped, not forwarded".into());
    }

    // OPEN: F1 is forwarded rather than edged; ESC/F11 keep theirs; nav keys
    // become menu edges (repeat ALLOWED); everything else forwards.
    let (e, n) = run(&key(Keycode::F1, false), open);
    if e.toggle_hud || n != 1 {
        return Err("open: F1 must be forwarded, not an edge".into());
    }
    let (e, n) = run(&key(Keycode::Escape, false), open);
    if !e.esc || n != 0 {
        return Err("open: ESC must stay an edge and not forward".into());
    }
    let (e, n) = run(&key(Keycode::F11, false), open);
    if !e.toggle_fullscreen || n != 0 {
        return Err("open: F11 must stay an edge and not forward".into());
    }
    let (e, n) = run(&key(Keycode::Up, true), open);
    if !e.menu_up || n != 0 {
        return Err("open: a held Up must be menu_up (repeat allowed), not forwarded".into());
    }
    let (e, _) = run(&key(Keycode::W, false), open);
    if !e.menu_up {
        return Err("open: W must be menu_up".into());
    }
    let (e, _) = run(&key(Keycode::S, false), open);
    if !e.menu_down {
        return Err("open: S must be menu_down".into());
    }
    let (e, _) = run(&key(Keycode::A, false), open);
    if !e.menu_left {
        return Err("open: A must be menu_left".into());
    }
    let (e, _) = run(&key(Keycode::Right, false), open);
    if !e.menu_right {
        return Err("open: Right must be menu_right".into());
    }
    let (e, n) = run(&key(Keycode::Return, false), open);
    if !e.menu_activate || n != 0 {
        return Err("open: Return must be menu_activate".into());
    }
    let (e, n) = run(&keyup(Keycode::Return), open);
    if e.menu_activate || n != 0 {
        return Err("open: a nav KeyUp is consumed, neither edge nor forward".into());
    }
    let (e, n) = run(&key(Keycode::H, false), open);
    if e.toggle_bounce || n != 1 {
        return Err("open: H must be forwarded, never a toggle".into());
    }
    let (_, n) = run(&motion, open);
    if n != 1 {
        return Err("open: mouse motion must be forwarded".into());
    }

    // OPEN + EDITING: nav keys forward (cursor movement / `accepted`).
    let (e, n) = run(&key(Keycode::Up, false), editing);
    if e.menu_up || n != 1 {
        return Err("editing: Up must forward, not edge".into());
    }
    let (e, n) = run(&key(Keycode::Return, false), editing);
    if e.menu_activate || n != 1 {
        return Err("editing: Return must forward".into());
    }
    let (e, n) = run(&key(Keycode::Escape, false), editing);
    if !e.esc || n != 0 {
        return Err("editing: ESC still an edge".into());
    }

    // BOTH modes: window-level events keep their edges and never forward.
    for (name, mode) in [("closed", closed), ("open", open), ("editing", editing)] {
        let (e, n) = run(&quit, mode);
        if !e.quit || n != 0 {
            return Err(format!("{name}: Quit must set quit and not forward"));
        }
        let (e, n) = run(&resized, mode);
        if e.size_changed != Some((640, 360)) || n != 0 {
            return Err(format!("{name}: Resized must set size_changed"));
        }
        let (e, n) = run(&moved, mode);
        if !e.display_changed || n != 0 {
            return Err(format!("{name}: Moved must set display_changed"));
        }
        let (e, n) = run(&pad(B::Start), mode);
        if !e.menu_toggle || n != 0 {
            return Err(format!("{name}: pad Start must set menu_toggle"));
        }
    }

    // Gamepad: nav only while open; dropped while closed (flight owns it).
    let (e, n) = run(&pad(B::DPadDown), open);
    if !e.menu_down || n != 0 {
        return Err("open: pad DPadDown must be menu_down".into());
    }
    let (e, _) = run(&pad(B::South), open);
    if !e.menu_activate {
        return Err("open: pad South must be menu_activate".into());
    }
    let (e, _) = run(&pad(B::East), open);
    if !e.menu_back {
        return Err("open: pad East must be menu_back".into());
    }
    let (e, n) = run(&pad(B::DPadDown), closed);
    if e.menu_down || n != 0 {
        return Err("closed: pad DPadDown must be dropped".into());
    }
    Ok(())
}
