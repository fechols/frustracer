//! The window, its event pump, and the frame loop's presentation half —
//! B6b rungs 1 (the window) and 2 (the input).
//!
//! WHAT RUNG 1 ADDED TO THE HEADLESS PRESENT PATH. `swapchain.rs` acquires,
//! renders, presents and lets the engine recycle, all over
//! `VK_EXT_headless_surface` — a real present path with nothing to scan out.
//! What it could not do is the two things a surface implies: a WINDOW (so a
//! human can look at this backend, which nobody had) and PACING (there is no
//! vblank without a compositor). Everything below the surface is unchanged and
//! unmoved — that is the whole reason the headless rung came first.
//!
//! SDL, NOT A NEW WINDOWING CRATE. The Windows session already runs on SDL3, so
//! using it here means one windowing library in the tree rather than two, and
//! the crate carries exactly what the handoff needs: `vulkan_instance_extensions`
//! names the platform surface extension itself (so nothing here mentions xlib,
//! xcb or wayland), and its `ash` feature makes `vulkan_create_surface` hand
//! back an `ash::vk::SurfaceKHR` — our own type, no transmute at the one place
//! this tree crosses from a windowing library into its backend.
//!
//! RUNG 2 MAKES `pump` THE INPUT SOURCE, and the shape is forced rather than
//! chosen. `SDL_PumpEvents` may only be called from the thread that made the
//! window, and Wayland exposes no off-thread keyboard state — so the Windows
//! trick (sample the OS from the integrator, which is what makes displacement
//! independent of frame time there) has no portable equivalent. Something must
//! ask the OS at input rate and the renderer cannot, so `pump` writes a
//! `flycam::Mirror` from the MAIN thread while the frame loop runs elsewhere.
//! MEASURED, and it is the whole purchase of the split: the pump interval is
//! p50 1.06 / p99 1.08 ms on its own thread against p50 10.4–16.7 / p99
//! 11.3–18.8 ms once per frame (`FR_VK_PUMP_INLINE=1`, which keeps that arm
//! alive as arm B). That interval bounds how short a key tap can be before its
//! down and its up land in one drain and the press is lost outright.
//!
//! WHAT `pump` STILL DOES NOT DO is `input.rs`'s job: the three-tier `Edges`
//! drain (SPACE, F, H, P, F1 …) answers a pause menu this backend has no peer
//! of yet. Continuous STATE for a 500 Hz integrator and per-FRAME EDGES for a
//! session are two mechanisms on Windows too (`GetAsyncKeyState` beside
//! `input.rs`), so keeping them apart here is that shape, not a fork of it.

use ash::vk;
use std::time::{Duration, Instant};

use crate::vk::device::Vk;
use crate::vk::display::{self, Passes};
use crate::vk::headless::VkHeadless;
use crate::vk::spirv::Spirv;
use crate::vk::swapchain::{self, Swapchain};

/// The SDL side: context, window, and the event pump.
///
/// Held together because SDL's own lifetimes require it — the video subsystem
/// must outlive the window, and dropping the context tears down the
/// connection — and because the ORDER matters at bring-up: the window has to
/// exist (with the Vulkan flag, which is what loads SDL's Vulkan library)
/// before `instance_extensions` can answer, and the instance has to exist
/// before `surface` can be asked for.
pub struct Win {
    /// Held for its LIFETIME rather than read, the `ash::Entry` rule: dropping
    /// the context shuts SDL down under the window and the surface.
    #[allow(dead_code)]
    sdl: sdl3::Sdl,
    #[allow(dead_code)]
    video: sdl3::VideoSubsystem,
    window: sdl3::video::Window,
    pump: sdl3::EventPump,
    /// The gamepad subsystem, and the one open pad if there is one. SDL
    /// delivers axis/button events only for an OPENED device, and only while
    /// the handle lives — so this is held rather than dropped after `open`.
    gamepads: sdl3::GamepadSubsystem,
    pad: Option<sdl3::gamepad::Gamepad>,
    /// Drained into before dispatch, and REUSED: `poll_iter` borrows the pump
    /// for the whole loop, so opening a gamepad (which needs `&mut self`)
    /// cannot happen inside it. Cleared rather than reallocated, so a 500 Hz
    /// pump allocates nothing in steady state.
    evs: Vec<sdl3::event::Event>,
}

impl Win {
    /// Open a window sized `w`x`h`.
    ///
    /// `.vulkan()` on the builder is what makes SDL load its Vulkan library,
    /// which every call below depends on — without it
    /// `SDL_Vulkan_GetInstanceExtensions` has nothing to report.
    ///
    /// DELIBERATELY NOT `.resizable()`. Rung 1 has no resize path — the
    /// swapchain, the tracer and the display pipelines are all built once at
    /// one extent — so advertising the capability invites a resize whose only
    /// possible outcomes are bad: a driver that reports SUBOPTIMAL keeps
    /// presenting the OLD extent and lets the compositor scale it (MEASURED on
    /// RADV: 1280x720 stretched into a 320x240 window, still ~105 fps, a wrong
    /// image that still looks like a picture — the very failure the FFX-extent
    /// check in `run_window_vk` refuses), and a driver that reports
    /// OUT_OF_DATE stops the session. Not asking for it is the honest shape
    /// until rung 2 can rebuild.
    ///
    /// It is not a GUARANTEE — a tiling compositor resizes whatever it likes —
    /// which is why `Lost::Stale` exists underneath as well.
    pub fn open(w: u32, h: u32, title: &str) -> Result<Win, String> {
        let sdl = sdl3::init().map_err(|e| format!("SDL_Init: {e}"))?;
        let video = sdl.video().map_err(|e| format!("SDL video subsystem: {e}"))?;
        let window = video
            .window(title, w, h)
            .position_centered()
            .vulkan()
            .build()
            .map_err(|e| format!("SDL_CreateWindow: {e}"))?;
        let pump = sdl.event_pump().map_err(|e| format!("SDL_GetEventPump: {e}"))?;
        // The gamepad subsystem is initialized whether or not one is plugged
        // in: SDL only reports a device ARRIVING to a subsystem that exists,
        // so deferring this would mean a pad connected mid-session is one the
        // session never hears about.
        let gamepads = sdl.gamepad().map_err(|e| format!("SDL gamepad subsystem: {e}"))?;
        Ok(Win { sdl, video, window, pump, gamepads, pad: None, evs: Vec::new() })
    }

    /// The instance extensions SDL needs to make a surface for this window —
    /// `VK_KHR_surface` plus one platform extension it picks itself.
    ///
    /// Fed to `Vk::new`, which UNIONS rather than appends: `VK_KHR_surface` is
    /// already in the backend's own list on every box that can run V19, so a
    /// concatenation would name it twice.
    pub fn instance_extensions(&self) -> Result<Vec<String>, String> {
        self.window
            .vulkan_instance_extensions()
            .map_err(|e| format!("SDL_Vulkan_GetInstanceExtensions: {e}"))
    }

    /// Create the `VkSurfaceKHR`.
    ///
    /// OWNERSHIP PASSES TO THE CALLER: SDL does not destroy this when the
    /// window drops — `vkDestroySurfaceKHR` is the application's job — and
    /// `Swapchain::from_surface` takes it on, so the surface dies with
    /// `Swapchain::destroy`.
    pub fn surface(&self, vkd: &Vk) -> Result<vk::SurfaceKHR, String> {
        // SAFETY: the instance is live for the whole call and outlives the
        // surface, which is the contract `vulkan_create_surface` states.
        unsafe { self.window.vulkan_create_surface(vkd.instance.handle()) }
            .map_err(|e| format!("SDL_Vulkan_CreateSurface: {e}"))
    }

    /// The window's CURRENT pixel size, which is what a swapchain is built
    /// against.
    ///
    /// `size_in_pixels`, never `size`: they differ by the display scale on a
    /// HiDPI output, and a swapchain built at logical size on a 2x display is
    /// a quarter-resolution image the compositor then stretches.
    pub fn size(&self) -> (u32, u32) {
        self.window.size_in_pixels()
    }

    /// Drain the event queue into `m`. `false` means the user asked to quit.
    ///
    /// THE ONLY WRITER OF THE MIRROR, and the only thing on this thread that
    /// may touch SDL at all — which is the whole reason rung 2 has three
    /// threads: `SDL_PumpEvents` belongs to the thread that made the window,
    /// and that thread must never be the one blocked in a trace. Events still
    /// have to be DRAINED whether or not they are acted on: a queue nothing
    /// reads grows without bound and, on some compositors, a window that never
    /// pumps is declared unresponsive.
    ///
    /// WHAT IS DELIBERATELY ABSENT is the toggle-edge drain (`input.rs`'s
    /// `Edges` — SPACE, F, H, P, F1 …). That is a per-FRAME question answered
    /// against a pause menu this backend has no peer of yet; this is
    /// continuous STATE for a 500 Hz integrator. Windows keeps the same two
    /// mechanisms apart for the same reason (`input.rs` beside
    /// `GetAsyncKeyState`), so this is that shape rather than a fork of it.
    pub fn pump(&mut self, m: &crate::flycam::Mirror) -> bool {
        use sdl3::event::{Event, WindowEvent};
        use sdl3::gamepad::Axis;
        use sdl3::keyboard::Keycode;
        use sdl3::mouse::MouseButton;
        // Taken and PUT BACK at the end of the pass (there is no early return
        // between — that is what `quit` is for), so the allocation travels
        // with it and a steady-state pump allocates nothing. Draining rather
        // than consuming keeps that capacity: `into_iter` would hand it to the
        // loop and drop it.
        let mut evs = std::mem::take(&mut self.evs);
        evs.clear();
        evs.extend(self.pump.poll_iter());
        m.pumped();
        let mut quit = false;
        for ev in evs.drain(..) {
            match ev {
                Event::Quit { .. }
                | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => quit = true,

                // REPEATS FILTERED. A held key is already down as far as the
                // mirror is concerned, so an auto-repeat is pure noise — and
                // the whole point of the compare-before-store inside `key` is
                // that an idle session writes nothing.
                Event::KeyDown { scancode: Some(sc), repeat: false, .. } => {
                    if let Some(a) = crate::flycam::action_for_scancode(sc) {
                        m.key(a, true);
                    }
                }
                Event::KeyUp { scancode: Some(sc), .. } => {
                    if let Some(a) = crate::flycam::action_for_scancode(sc) {
                        m.key(a, false);
                    }
                }

                // `xrel`/`yrel` on the ABSOLUTE cursor, deliberately NOT
                // relative-mouse mode: Windows differences the accelerated OS
                // cursor out of `GetCursorPos` and therefore stops at the
                // screen edge, and this is the same quantity with the same
                // edge behaviour. Capturing the pointer would be a different
                // feel, which is the wrong thing for a parity rung.
                Event::MouseMotion { xrel, yrel, .. } => m.look(xrel, yrel),
                // A button event only reaches us when it landed on our window,
                // so this IS `drag_may_start`'s hit-test — the compositor
                // already did it.
                Event::MouseButtonDown { mouse_btn: MouseButton::Left, .. } => m.set_drag(true),
                Event::MouseButtonUp { mouse_btn: MouseButton::Left, .. } => m.set_drag(false),

                Event::Window { win_event: WindowEvent::FocusGained, .. } => m.set_focused(true),
                Event::Window { win_event: WindowEvent::FocusLost, .. } => m.set_focused(false),

                // A gamepad must be OPENED before SDL delivers its axes, and
                // the handle has to be held for as long as we want them — so
                // `pad` is a field rather than a local. First one wins, which
                // is XInput slot 0's rule on the other side.
                Event::ControllerDeviceAdded { which, .. } => self.open_pad(which, m),
                Event::ControllerDeviceRemoved { which, .. } => {
                    if self.pad.as_ref().is_some_and(|p| p.id().is_ok_and(|id| id == which)) {
                        self.pad = None;
                        m.pad_present(false);
                    }
                }
                Event::ControllerAxisMotion { axis, value, .. } => {
                    let i = match axis {
                        Axis::LeftX => 0,
                        Axis::LeftY => 1,
                        Axis::RightX => 2,
                        Axis::RightY => 3,
                        Axis::TriggerLeft => 4,
                        Axis::TriggerRight => 5,
                    };
                    m.pad_axis(i, value);
                }
                Event::ControllerButtonDown { button, .. } => pad_button(m, button, true),
                Event::ControllerButtonUp { button, .. } => pad_button(m, button, false),
                _ => {}
            }
        }
        self.evs = evs;
        !quit
    }

    /// Open a newly-arrived gamepad, if we do not already have one.
    ///
    /// NEVER FATAL: a pad that will not open is a pad the session flies
    /// without, and saying so once beats either a panic or silence.
    fn open_pad(&mut self, which: u32, m: &crate::flycam::Mirror) {
        if self.pad.is_some() {
            return;
        }
        match self.gamepads.open(sdl3::joystick::JoystickId::new(which)) {
            Ok(g) => {
                eprintln!("vk: gamepad — {}", g.name().unwrap_or_else(|| "unnamed".into()));
                self.pad = Some(g);
                m.pad_present(true);
            }
            Err(e) => eprintln!("vk: a gamepad arrived but would not open ({e}) — flying without"),
        }
    }
}

/// The four pad buttons the integrator knows, in `Mirror`'s bit order.
/// Everything else on the pad is deliberately unbound here — face buttons
/// belong to a menu this backend has no peer of yet.
fn pad_button(m: &crate::flycam::Mirror, b: sdl3::gamepad::Button, down: bool) {
    use sdl3::gamepad::Button;
    let i = match b {
        Button::LeftShoulder => 0,
        Button::RightShoulder => 1,
        Button::DPadLeft => 2,
        Button::DPadRight => 3,
        _ => return,
    };
    m.pad_button(i, down);
}

/// Presented-frame interval statistics — the measurement rung 2 owed this rung.
///
/// WALL CLOCK ON THE CPU, and the limits of that are worth stating rather than
/// leaving to be discovered: this measures the interval between `present`
/// RETURNING, which under FIFO is governed by the compositor's vblank, so it
/// answers "is the cadence steady and what is it" and NOT "where does the GPU
/// time go". The latter needs a `vkCmdWriteTimestamp` instrument that does not
/// exist on this backend yet (there is no peer of `gpu/gputime.rs`); it is a
/// slice of its own and this one does not pretend to it.
///
/// p50 and p99 rather than a mean alone, because a mean is exactly the statistic
/// that cannot see a hitch: one 200 ms stall in a 60-frame second moves the mean
/// by 3 ms and the p99 to 200.
pub struct Pacing {
    intervals: Vec<f32>,
    last: Option<Instant>,
    since_report: Instant,
    period: Duration,
    /// Pump-interval samples, in ms — one per frame, from `Mirror::pump_gap`.
    /// Empty unless the caller notes them, which the headless V19 gate does
    /// not.
    ///
    /// THIS IS THE NUMBER THE THREAD SPLIT IS FOR: pumping on its own thread
    /// makes it a millisecond, pumping on the render thread makes it a frame.
    /// It rides the pacing report because the two are one question — a cadence
    /// that only looks at the keyboard once per frame is not the same thing as
    /// a cadence.
    pump_gaps: Vec<f32>,
}

impl Pacing {
    pub fn new(period_s: f32) -> Pacing {
        Pacing {
            intervals: Vec::new(),
            last: None,
            since_report: Instant::now(),
            period: Duration::from_secs_f32(period_s),
            pump_gaps: Vec::new(),
        }
    }

    /// Record the interval between the two most recent pump passes. Optional
    /// by design: the headless gate has no pump.
    pub fn note_pump_gap(&mut self, gap: Duration) {
        self.pump_gaps.push(gap.as_secs_f32() * 1000.0);
    }

    /// Record one presented frame. Returns a report line when the period is up.
    ///
    /// The FIRST frame records no interval, which is not an off-by-one to fix:
    /// there is no previous present to measure against, and seeding it with a
    /// zero would put a spurious 0 ms sample in the p-values.
    pub fn tick(&mut self) -> Option<String> {
        let now = Instant::now();
        if let Some(prev) = self.last {
            self.intervals.push((now - prev).as_secs_f32() * 1000.0);
        } else {
            // THE PERIOD STARTS AT THE FIRST PRESENT, not at construction. A
            // `Pacing` is built before the scene loads, and a cold world boot
            // is ~13 s — so a clock started in `new` is already expired when
            // the first frame lands, and the first report is a ONE-SAMPLE
            // report whose mean, p50 and p99 are necessarily the same number.
            // That is not a statistic, it is one interval wearing three hats,
            // and it was the first line the window printed.
            self.since_report = now;
        }
        self.last = Some(now);
        if now - self.since_report < self.period || self.intervals.is_empty() {
            return None;
        }
        let mut v = std::mem::take(&mut self.intervals);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        let mean = v.iter().sum::<f32>() / n as f32;
        // NEAREST-RANK, and the saturating index is what makes it total at
        // n == 1: `ceil(0.99 * 1) - 1 == 0`, and for n where the product lands
        // exactly on an integer the -1 keeps it in range.
        let pick = |q: f32| v[(((q * n as f32).ceil() as usize).max(1) - 1).min(n - 1)];
        self.since_report = now;
        let mut a = std::mem::take(&mut self.pump_gaps);
        let input = if a.is_empty() {
            String::new()
        } else {
            a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            let m = a.len();
            let apick =
                |q: f32| a[(((q * m as f32).ceil() as usize).max(1) - 1).min(m - 1)];
            format!(" | pump gap p50 {:.2} p99 {:.2} ms", apick(0.50), apick(0.99))
        };
        Some(format!(
            "present: {:.1} fps | interval mean {:.2} p50 {:.2} p99 {:.2} ms over {} frame(s){}",
            1000.0 / mean,
            mean,
            pick(0.50),
            pick(0.99),
            n,
            input
        ))
    }
}

/// What one `Presenter::present` did.
///
/// A THIRD ANSWER BESIDE Ok AND Err, because the surface going out of date is
/// neither. It is not success — nothing reached the screen — and it is not a
/// failure: the spec's answer is "rebuild the swapchain", which rung 1 cannot
/// do and rung 2 will. Collapsing it into `Err` is what made a compositor
/// resize read as a crash (`window: vkQueuePresentKHR: ERROR_OUT_OF_DATE_KHR`,
/// exit 2); collapsing it into `Ok` would spin forever presenting nothing.
pub enum Frame {
    Presented,
    /// The surface and the swapchain disagree — see `swapchain::Lost::Stale`.
    Stale,
}

/// The swapchain, the display pipelines, and one frame's present.
pub struct Presenter {
    pub sc: Swapchain,
    passes: Passes,
    pub pacing: Pacing,
}

impl Presenter {
    /// Build over a window's surface.
    ///
    /// THE PIPELINES ARE BUILT AT THE NEGOTIATED FORMAT, never at a chosen one:
    /// `pick_format` takes the surface's own preference order, and on both ICDs
    /// measured here that is `R8G8B8A8_UNORM` — the opposite byte order from the
    /// one V18 renders. A pipeline whose rendering format disagrees with the
    /// swapchain's is ALSO the one defect here that only the validation layer
    /// names (RADV segfaults on the invalid pipeline rather than erroring), so
    /// reading the format back off the swapchain rather than passing one in is
    /// what makes the two agree by construction.
    pub fn new(hg: &VkHeadless, sp: &Spirv, surface: vk::SurfaceKHR, w: u32, h: u32)
        -> Result<Presenter, String>
    {
        let sc = Swapchain::from_surface(hg, surface, w, h)?;
        let passes = Passes::new(hg, sp, sc.fmt)?;
        Ok(Presenter { sc, passes, pacing: Pacing::new(2.0) })
    }

    /// Point the display stage at the image it should tonemap.
    ///
    /// Called ONCE, not per frame: FFX writes the same output image every
    /// frame, so the descriptor is stable and rewriting it per frame would be
    /// pure descriptor traffic. It rests in `GENERAL`, which is what
    /// `bind_source` declares — see `Fsr3::output_view`.
    pub fn bind_source(&self, vkd: &Vk, view: vk::ImageView) {
        self.passes.bind_source(vkd, view);
    }

    /// Tonemap `src` into the next swapchain image and present it.
    ///
    /// THE SHAPE IS RUNG 2'S, unchanged: acquire, `record_to` (which transitions
    /// in from `UNDEFINED` — correct for a freshly acquired image, whose
    /// contents are undefined by contract — and out to `TRANSFER_SRC_OPTIMAL`),
    /// one further barrier to `PRESENT_SRC_KHR`, `run_present` with the
    /// semaphore pair, then present. `wait_submit` before returning is what
    /// makes the next frame's `reset_command_pool` legal: one command buffer,
    /// one fence, no frames in flight. That costs a real pipeline bubble and it
    /// is rung 1's honest shape — overlapping frames needs a per-frame command
    /// buffer and a fence ring, which is a change to `VkHeadless` and belongs
    /// with the thread split rather than under it.
    /// A STALE ACQUIRE RETURNS BEFORE RECORDING ANYTHING, which is what makes
    /// the early return safe: no command buffer was submitted and no semaphore
    /// was signalled, so there is nothing outstanding for the caller to drain.
    /// A stale PRESENT is the other way round — the submit already ran and
    /// `wait_submit` still has to happen — so that arm falls through to the
    /// wait and reports afterwards.
    pub fn present(&mut self, hg: &VkHeadless, params: display::Params) -> Result<Frame, String> {
        self.passes.set_params(&hg.vk, params)?;
        let idx = match self.sc.acquire() {
            Ok(i) => i,
            Err(swapchain::Lost::Stale) => return Ok(Frame::Stale),
            Err(e) => return Err(e.to_string()),
        };
        let (wait, stages) = self.sc.wait_pair();
        let signal = self.sc.signal(idx);
        let (img, view) = (self.sc.images[idx], self.sc.view(idx));
        let (w, h) = (self.sc.w, self.sc.h);
        let passes = &self.passes;
        let sc = &self.sc;
        hg.run_present(&wait, &stages, &signal, |d, cmd| {
            passes.record_to(d, cmd, img, view, w, h, true);
            sc.to_present_layout(d, cmd, idx);
        })?;
        let stale = match self.sc.present(&hg.vk, idx) {
            Ok(()) => false,
            Err(swapchain::Lost::Stale) => true,
            Err(e) => return Err(e.to_string()),
        };
        hg.wait_submit()?;
        if stale {
            return Ok(Frame::Stale);
        }
        if let Some(line) = self.pacing.tick() {
            eprintln!("{line}");
        }
        Ok(Frame::Presented)
    }

    pub fn destroy(&self, vkd: &Vk) {
        self.sc.destroy(vkd);
        self.passes.destroy(vkd);
    }
}
