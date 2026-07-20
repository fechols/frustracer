//! 500 Hz wall-clock camera integrator (keyboard + mouse + Xbox controller).
//!
//! Camera motion used to be integrated once per rendered frame with
//! dt = frame time, but the main thread blocks for the whole trace (100+ ms
//! on heavy scenes), so a key tap counted as a full frame of motion when it
//! happened to span the event pump — or was lost outright. This thread
//! samples input every ~2 ms through Win32 calls that read live OS state
//! from any thread (`GetAsyncKeyState`, `GetCursorPos`, `XInputGetState`;
//! SDL's own key/mouse state only updates when the blocked main thread
//! pumps, so it is useless here) and integrates with the MEASURED tick dt.
//! Displacement is therefore an exact function of wall-clock time
//! regardless of render framerate — timer jitter changes granularity,
//! never totals. Rate inputs (the controller's right stick) are where this
//! matters most: "how long the stick was held at this angle" is integrated
//! at 2 ms granularity instead of per-frame over/undershoot.
//!
//! The render loop consumes exactly ONE `snapshot()` per loop iteration and
//! uses only that snapshot everywhere in the iteration (trace pose == MV
//! pose == prev-capture pose — the temporal/replay/upscaler bit-equality
//! contracts). Any future teleport/reset must write through `set()`, never
//! a session-local `Camera` copy.

use crate::camera::Camera;
use glam::Vec3A;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use windows::core::PCWSTR;
use windows::Win32::Foundation::{CloseHandle, ERROR_SUCCESS, HANDLE, HWND, POINT, RECT};
use windows::Win32::Graphics::Gdi::ScreenToClient;
use windows::Win32::Media::{timeBeginPeriod, timeEndPeriod};
use windows::Win32::System::Threading::{
    CreateWaitableTimerExW, GetCurrentThread, SetThreadPriority, SetWaitableTimer,
    WaitForSingleObject, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, INFINITE,
    THREAD_PRIORITY_ABOVE_NORMAL, TIMER_ALL_ACCESS,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, MapVirtualKeyW, MAPVK_VSC_TO_VK_EX, VK_CONTROL, VK_DOWN, VK_LBUTTON,
    VK_LEFT, VK_OEM_COMMA, VK_OEM_PERIOD, VK_RBUTTON, VK_RIGHT, VK_SHIFT, VK_UP,
};
use windows::Win32::UI::Input::XboxController::{
    XInputGetState, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
    XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE,
    XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE,
    XINPUT_GAMEPAD_TRIGGER_THRESHOLD, XINPUT_STATE,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetClientRect, GetCursorPos, GetForegroundWindow, GetSystemMetrics, WindowFromPoint,
    SM_SWAPBUTTON,
};

/// Integrator tick period. 2 ms = 500 Hz. Total displacement is exact at any
/// tick rate (measured dt); the rate only bounds input-to-camera latency and
/// the piecewise-constant error at key/stick transitions.
const TICK_MS: u32 = 2;
/// Right-stick look rate at full deflection, radians/second.
const LOOK_RATE: f32 = 2.5;
/// Stick response exponent: magnitude^2 keeps the center precise while full
/// deflection still reaches 1.0.
const STICK_CURVE: i32 = 2;
/// Probing a disconnected XInput slot is documented-slow — back off ~2 s
/// between probes (in ticks) after ERROR_DEVICE_NOT_CONNECTED.
const PAD_REPROBE_TICKS: u32 = 2000 / TICK_MS;
/// Time-of-day scrub rate: game-hours per real second held (`.`/`,` or D-pad
/// right/left). The Ctrl(/16)/Shift(/8)/bumper divisors apply, for fine scrub.
const TOD_RATE: f32 = 1.0;
/// Seconds for a slow-factor modifier (Ctrl/LB, Shift/RB) to ramp between
/// full speed and its divided speed. The ramp is smoothstep-shaped, so
/// engaging AND releasing both ease instead of stepping the speed 8-16x in
/// one tick.
const SLOW_EASE_S: f32 = 0.25;

/// One integrator snapshot: the camera pose of record plus the time-of-day
/// hour, taken under ONE lock so a render iteration sees a consistent pair.
#[derive(Clone, Copy)]
pub struct FlyState {
    pub cam: Camera,
    /// Hours, wrapped into [0, 24). Consumed by the session loop, which turns
    /// a delta into `scene::apply_tod` + the shading-change resets.
    pub tod: f32,
}

pub struct FlyCam {
    shared: Arc<Shared>,
    handle: Option<std::thread::JoinHandle<()>>,
}

struct Shared {
    state: Mutex<FlyState>,
    stop: AtomicBool,
    /// Integration gate. A LONG FRAME must still integrate — that is the
    /// whole feature. But a session rebuild (resize / F11 re-entry: kernel
    /// compile + scene upload + BLAS build, seconds on a big scene) presents
    /// no frames at all, so flying through it is flying blind, with nothing
    /// on screen to correct against. `run_window` pauses across the rebuild
    /// and `session` resumes once its frame loop is actually running; the
    /// thread spawns paused so the first session's init is covered too.
    paused: AtomicBool,
}

impl FlyCam {
    /// Spawn the integrator for the session window, PAUSED (see `resume`).
    /// `hwnd` rides as isize because windows::HWND is a raw pointer and not
    /// Send; the thread only ever compares it / hands it to read-only queries.
    pub fn spawn(hwnd: isize, cam0: Camera, tod0: f32, diag: f32) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(FlyState { cam: cam0, tod: tod0 }),
            stop: AtomicBool::new(false),
            paused: AtomicBool::new(true),
        });
        let s2 = shared.clone();
        let handle = std::thread::Builder::new()
            .name("flycam".into())
            .spawn(move || integrate_loop(&s2, hwnd, diag))
            .expect("flycam thread spawn failed");
        Self { shared, handle: Some(handle) }
    }

    /// The pose + time-of-day of record. Call ONCE per render-loop iteration
    /// and use only the returned snapshot for the whole iteration.
    pub fn snapshot(&self) -> FlyState {
        *self.shared.state.lock().unwrap()
    }

    /// Stop/start integrating. Paused ticks keep advancing the integrator's
    /// dt clock, so resuming costs one tick of motion — never the whole
    /// paused span dumped into one step. Held keys during a pause are simply
    /// not integrated (the camera stays bit-untouched, so a paused span never
    /// invalidates replay either).
    pub fn pause(&self) {
        self.shared.paused.store(true, Relaxed);
    }

    pub fn resume(&self) {
        self.shared.paused.store(false, Relaxed);
    }

    /// Write-through for teleports/resets. Nothing calls it today; it exists
    /// so a future pose write goes through the shared camera instead of a
    /// session-local copy the integrator would immediately overwrite.
    #[allow(dead_code)]
    pub fn set(&self, cam: Camera) {
        self.shared.state.lock().unwrap().cam = cam;
    }
}

impl Drop for FlyCam {
    fn drop(&mut self) {
        self.shared.stop.store(true, Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join(); // returns within one tick (~2 ms)
        }
    }
}

/// The ~2 ms tick source. Plain `thread::sleep(2ms)` quantizes to the
/// ~15.6 ms system tick (~64 Hz effective); the high-resolution waitable
/// timer (Win10 1803+) delivers ~0.5 ms precision for one blocked wait and
/// opts out of timer coalescing. Fallback when unavailable: raise the global
/// timer resolution for this process (timeBeginPeriod) and sleep. Either
/// way integration uses measured dt, so the tick source never affects
/// displacement — only granularity.
struct Ticker {
    timer: Option<HANDLE>,
}

impl Ticker {
    fn new() -> Self {
        unsafe {
            if let Ok(t) = CreateWaitableTimerExW(
                None,
                PCWSTR::null(),
                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                TIMER_ALL_ACCESS.0,
            ) {
                // Relative first due time (negative, 100 ns units), then the
                // millisecond period takes over.
                let due = -(TICK_MS as i64) * 10_000;
                if SetWaitableTimer(t, &due, TICK_MS as i32, None, None, false).is_ok() {
                    return Self { timer: Some(t) };
                }
                let _ = CloseHandle(t);
            }
            eprintln!("flycam: high-res waitable timer unavailable; timeBeginPeriod(1) fallback");
            timeBeginPeriod(1);
            Self { timer: None }
        }
    }

    fn wait(&self) {
        match self.timer {
            Some(t) => unsafe {
                WaitForSingleObject(t, INFINITE);
            },
            None => std::thread::sleep(Duration::from_millis(TICK_MS as u64)),
        }
    }
}

impl Drop for Ticker {
    fn drop(&mut self) {
        unsafe {
            match self.timer {
                Some(t) => {
                    let _ = CloseHandle(t);
                }
                None => {
                    timeEndPeriod(1);
                }
            }
        }
    }
}

/// Movement virtual keys, resolved once from the physical Set-1 scancodes
/// SDL used (`is_scancode_pressed`) via the current layout, so e.g. AZERTY
/// keeps flying on the same physical keys. Fallback to the US-layout VK if
/// the mapping comes back empty.
struct Keys {
    w: u16,
    a: u16,
    s: u16,
    d: u16,
    q: u16,
    e: u16,
    /// Time-of-day reverse / fast-forward (`,` / `.` physical keys).
    comma: u16,
    period: u16,
}

impl Keys {
    fn new() -> Self {
        let vk = |sc: u32, fallback: u16| {
            let v = unsafe { MapVirtualKeyW(sc, MAPVK_VSC_TO_VK_EX) } as u16;
            if v == 0 { fallback } else { v }
        };
        Self {
            w: vk(0x11, b'W' as u16),
            a: vk(0x1E, b'A' as u16),
            s: vk(0x1F, b'S' as u16),
            d: vk(0x20, b'D' as u16),
            q: vk(0x10, b'Q' as u16),
            e: vk(0x12, b'E' as u16),
            // Space deliberately absent: it is the render-mode cycle
            // (input.rs), and a flight key polled here would ALSO fire on
            // every mode switch — the camera bumped upward per press.
            comma: vk(0x33, VK_OEM_COMMA.0),
            period: vk(0x34, VK_OEM_PERIOD.0),
        }
    }
}

/// One decoded controller sample: sticks deadzoned + curved to [-1, 1],
/// triggers thresholded to [0, 1], shoulder buttons raw.
struct Pad {
    lx: f32,
    ly: f32,
    rx: f32,
    ry: f32,
    lt: f32,
    rt: f32,
    lb: bool,
    rb: bool,
    /// D-pad left/right: time-of-day reverse / fast-forward.
    dpad_l: bool,
    dpad_r: bool,
}

/// Radial deadzone + response curve: the 2D magnitude below the deadzone is
/// dead; above it, rescaled so deadzone-edge -> 0 and full tilt -> 1, then
/// raised to STICK_CURVE for center precision. Direction is preserved.
fn stick(x: i16, y: i16, deadzone: f32) -> (f32, f32) {
    let (fx, fy) = (x as f32, y as f32);
    let mag = (fx * fx + fy * fy).sqrt();
    if mag <= deadzone {
        return (0.0, 0.0);
    }
    let m = ((mag - deadzone) / (32767.0 - deadzone)).min(1.0).powi(STICK_CURVE);
    (fx / mag * m, fy / mag * m)
}

fn trigger(t: u8) -> f32 {
    let thr = XINPUT_GAMEPAD_TRIGGER_THRESHOLD.0 as u8; // metadata mistypes it as button flags
    t.saturating_sub(thr) as f32 / (255 - thr) as f32
}

fn poll_pad(backoff: &mut u32) -> Option<Pad> {
    if *backoff > 0 {
        *backoff -= 1;
        return None;
    }
    let mut st = XINPUT_STATE::default();
    if unsafe { XInputGetState(0, &mut st) } != ERROR_SUCCESS.0 {
        *backoff = PAD_REPROBE_TICKS;
        return None;
    }
    let g = st.Gamepad;
    let (lx, ly) = stick(g.sThumbLX, g.sThumbLY, XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE.0 as f32);
    let (rx, ry) = stick(g.sThumbRX, g.sThumbRY, XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE.0 as f32);
    Some(Pad {
        lx,
        ly,
        rx,
        ry,
        lt: trigger(g.bLeftTrigger),
        rt: trigger(g.bRightTrigger),
        lb: (g.wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER).0 != 0,
        rb: (g.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER).0 != 0,
        dpad_l: (g.wButtons & XINPUT_GAMEPAD_DPAD_LEFT).0 != 0,
        dpad_r: (g.wButtons & XINPUT_GAMEPAD_DPAD_RIGHT).0 != 0,
    })
}

/// A look-drag may only latch when the press lands on our window's client
/// area: not on the title bar / borders (SDL never entered button-down state
/// from WM_NC* clicks either) and not on another window overlapping ours.
/// Once latched, the drag keeps tracking off-window until release — the SDL
/// mouse-capture behavior.
fn drag_may_start(hwnd: HWND, pt: POINT) -> bool {
    unsafe {
        if WindowFromPoint(pt) != hwnd {
            return false;
        }
        let mut c = pt;
        if !ScreenToClient(hwnd, &mut c).as_bool() {
            return false;
        }
        let mut rc = RECT::default();
        if GetClientRect(hwnd, &mut rc).is_err() {
            return false;
        }
        c.x >= 0 && c.y >= 0 && c.x < rc.right && c.y < rc.bottom
    }
}

fn integrate_loop(shared: &Shared, hwnd: isize, diag: f32) {
    // Above the rayon workers. The renderer saturates every core at normal
    // priority for the whole trace, and this thread needs ~10 us every 2 ms.
    // A starved tick never loses displacement (dt is measured, so the total
    // is exact regardless) — it coarsens the SAMPLING at key/stick
    // transitions, which is precisely what the 500 Hz rate buys. Best-effort:
    // a failure here costs granularity, not correctness.
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    }

    let ticker = Ticker::new();
    let keys = Keys::new();
    // Windows swaps the mouse buttons at the MESSAGE layer, but
    // GetAsyncKeyState reports the PHYSICAL button — so for a swapped-button
    // (left-handed) user the primary button is VK_RBUTTON. SDL's
    // `mouse_state().left()` was the logical primary, so reading VK_LBUTTON
    // unconditionally would have moved drag-look to their non-primary button.
    // Resolved once, like the key layout.
    let look_btn = unsafe {
        if GetSystemMetrics(SM_SWAPBUTTON) != 0 { VK_RBUTTON.0 } else { VK_LBUTTON.0 }
    };
    let hwnd = HWND(hwnd as *mut core::ffi::c_void);
    let mut last = Instant::now();
    let mut drag: Option<(i32, i32)> = None;
    let mut pad_backoff = 0u32;
    // Slow-modifier ramp positions in [0, 1]: 0 = full speed, 1 = fully
    // engaged divisor. Advanced every focused tick (even idle ones, so a
    // modifier held before movement starts is already engaged).
    let mut slow_ctrl = 0.0f32;
    let mut slow_shift = 0.0f32;

    loop {
        ticker.wait();
        if shared.stop.load(Relaxed) {
            break;
        }
        let now = Instant::now();
        // Same suspend/hitch clamp the old per-frame dt had. Note this runs
        // BEFORE both gates below, so a paused/unfocused span advances the
        // clock instead of accumulating into the first tick after it.
        let dt = (now - last).as_secs_f32().min(0.1);
        last = now;

        // Focus gate: GetAsyncKeyState/XInput are global — only act when our
        // window is foreground, and drop any latched drag on focus loss.
        // Pause gate: no frames are being presented (session rebuild), so
        // integrating would fly the camera blind. Both drop the drag.
        if shared.paused.load(Relaxed) || unsafe { GetForegroundWindow() } != hwnd {
            drag = None;
            continue;
        }

        let down = |vk: u16| unsafe { GetAsyncKeyState(vk as i32) } as u16 & 0x8000 != 0;
        let pad = poll_pad(&mut pad_backoff);

        // --- mouse look: absolute cursor deltas while a left-drag is
        // latched (the same accelerated OS cursor SDL reported, so the
        // 0.004 rad/px feel is unchanged).
        let (mut dx, mut dy) = (0.0f32, 0.0f32);
        let mut pt = POINT::default();
        if unsafe { GetCursorPos(&mut pt) }.is_err() || !down(look_btn) {
            drag = None;
        } else if let Some((px, py)) = drag {
            (dx, dy) = ((pt.x - px) as f32, (pt.y - py) as f32);
            drag = Some((pt.x, pt.y));
        } else if drag_may_start(hwnd, pt) {
            drag = Some((pt.x, pt.y)); // latch; first tick contributes no delta
        }

        // --- shared slow factor: Ctrl (or LB) /16, Shift (or RB) /8 — the
        // flight-speed divisors, also applied to the TOD scrub rate so the
        // same chord means "finer" everywhere. Eased: each modifier ramps
        // over SLOW_EASE_S with smoothstep shaping, and the divisor is
        // applied in log2 space (exp2 of a lerped exponent), so engagement
        // glides through the intermediate speeds and the rest states stay
        // EXACT (2^0 = 1, 2^-4 = 1/16, 2^-3 = 1/8 — no powf rounding).
        let ramp = |t: f32, held: bool| {
            if held { (t + dt / SLOW_EASE_S).min(1.0) } else { (t - dt / SLOW_EASE_S).max(0.0) }
        };
        slow_ctrl = ramp(slow_ctrl, down(VK_CONTROL.0) || pad.as_ref().is_some_and(|p| p.lb));
        slow_shift = ramp(slow_shift, down(VK_SHIFT.0) || pad.as_ref().is_some_and(|p| p.rb));
        let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
        let slow = (-4.0 * smooth(slow_ctrl) - 3.0 * smooth(slow_shift)).exp2();
        // --- flight speed: diag * 0.25 / s, times the slow factor.
        let speed = diag * 0.25 * dt * slow;

        // Keyboard direction flags (unit directions, normalized below —
        // exactly the old apply_movement).
        // Arrows alias WASD. They are layout-independent VKs, so unlike the
        // letter keys they need no scancode round-trip.
        let kw = down(keys.w) || down(VK_UP.0);
        let ks = down(keys.s) || down(VK_DOWN.0);
        let kd = down(keys.d) || down(VK_RIGHT.0);
        let ka = down(keys.a) || down(VK_LEFT.0);
        let kup = down(keys.e);
        let kdn = down(keys.q);
        let key_any = kw || ks || kd || ka || kup || kdn;

        // --- time-of-day scrub: `.`/D-pad-right forward, `,`/D-pad-left
        // reverse (both held = 0). Wall-clock-exact by the same measured-dt
        // argument as displacement.
        let t_fwd = down(keys.period) || pad.as_ref().is_some_and(|p| p.dpad_r);
        let t_rev = down(keys.comma) || pad.as_ref().is_some_and(|p| p.dpad_l);
        let tod_dir = (t_fwd as i32 - t_rev as i32) as f32;

        let pad_move = pad.as_ref().is_some_and(|p| p.lx != 0.0 || p.ly != 0.0 || p.lt != 0.0 || p.rt != 0.0);
        let pad_look = pad.as_ref().is_some_and(|p| p.rx != 0.0 || p.ry != 0.0);
        if !key_any && !pad_move && !pad_look && dx == 0.0 && dy == 0.0 && tod_dir == 0.0 {
            continue; // nothing to integrate; the shared state stays bit-untouched
        }

        let mut st = shared.state.lock().unwrap();
        let cam = &mut st.cam;
        let f = cam.forward();
        let r = f.cross(Vec3A::Y).normalize();
        let mut step = Vec3A::ZERO;
        if key_any {
            let mut delta = Vec3A::ZERO;
            if kw {
                delta += f;
            }
            if ks {
                delta -= f;
            }
            if kd {
                delta += r;
            }
            if ka {
                delta -= r;
            }
            if kup {
                delta += Vec3A::Y;
            }
            if kdn {
                delta -= Vec3A::Y;
            }
            if delta != Vec3A::ZERO {
                step += delta.normalize() * speed;
            }
        }
        if let Some(p) = &pad {
            // Analog flight: deflection IS the speed control (deliberately
            // not normalized like the keys); full tilt == key speed.
            step += (r * p.lx + f * p.ly) * speed;
            step += Vec3A::Y * ((p.rt - p.lt) * speed);
        }
        if step != Vec3A::ZERO {
            cam.pos += step;
        }

        // Look: mouse per-pixel (tick-rate independent by construction) +
        // right stick as a rate (rad/s x deflection x dt — the term that
        // needed wall-clock integration). Stick right = look right, stick
        // up = look up, matching the mouse-drag feel.
        let mut yaw_d = dx * 0.004;
        let mut pitch_d = -dy * 0.004;
        if let Some(p) = &pad {
            yaw_d += p.rx * LOOK_RATE * dt;
            pitch_d += p.ry * LOOK_RATE * dt;
        }
        if yaw_d != 0.0 || pitch_d != 0.0 {
            cam.yaw += yaw_d;
            cam.pitch = (cam.pitch + pitch_d).clamp(-1.5, 1.5);
        }

        // Time-of-day: 1 game-hour per held second (times the slow factor),
        // wrapped into [0, 24). Written only on an actual scrub, so an idle
        // session's snapshot compares bit-equal forever.
        if tod_dir != 0.0 {
            st.tod = (st.tod + tod_dir * TOD_RATE * dt * slow).rem_euclid(24.0);
        }
    }
}
