//! Main-thread Xbox-controller EDGES for the pause menu: Start toggles the
//! menu, A/B activate/back, D-pad + left stick navigate (with held-repeat).
//! This is deliberately separate from the flycam thread's level-triggered
//! flight integration (src/flycam.rs): menu input is discrete edges at frame
//! cadence, flight is continuous state at 500 Hz — different contracts.
//!
//! Concurrency: `XInputGetState` is a stateless read-only snapshot, so two
//! readers are safe. While the menu is OPEN this is the only reader (the
//! flycam's pause gate runs before its own poll); while CLOSED both threads
//! read, harmlessly. Headless paths never construct a `MenuPad`.

use std::time::{Duration, Instant};
use windows::Win32::Foundation::{ERROR_SUCCESS, HWND};
use windows::Win32::UI::Input::XboxController::{
    XInputGetState, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_DPAD_DOWN,
    XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT, XINPUT_GAMEPAD_DPAD_UP,
    XINPUT_GAMEPAD_START, XINPUT_STATE,
};
use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;

/// Held-direction auto-repeat: first refire after the delay, then at the
/// rate. The menu hold-loop polls at ~140 Hz, normal frames at 10-100 ms —
/// the timer is wall-clock so the cadence is the same either way.
const REPEAT_DELAY: Duration = Duration::from_millis(400);
const REPEAT_RATE: Duration = Duration::from_millis(120);

/// Left-stick digital-navigation threshold (of i16 full scale ±32767).
/// Deliberately ~half deflection — well above the analog flight deadzone, so
/// stick drift that the flycam would happily integrate never navigates.
const STICK_NAV: i16 = 16000;

/// Probing a disconnected XInput slot is documented-slow — back off ~2 s
/// between probes (flycam.rs's PAD_REPROBE_TICKS, but wall-clock: this poll
/// runs at frame cadence, not a fixed tick rate).
const REPROBE: Duration = Duration::from_secs(2);

// Synthesized state-word bits (D-pad OR stick for the directions).
const B_UP: u16 = 1 << 0;
const B_DOWN: u16 = 1 << 1;
const B_LEFT: u16 = 1 << 2;
const B_RIGHT: u16 = 1 << 3;
const B_A: u16 = 1 << 4;
const B_B: u16 = 1 << 5;
const B_START: u16 = 1 << 6;

/// One frame's controller edges. Directions fire on the press edge AND on
/// each held-repeat deadline; buttons fire on the press edge only.
#[derive(Default, Clone, Copy)]
pub struct PadEdges {
    pub start: bool,
    pub a: bool,
    pub b: bool,
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
}

pub struct MenuPad {
    hwnd: HWND,
    /// Previous synthesized state word (edge detection).
    prev: u16,
    /// Per-direction next-refire deadline while held (up/down/left/right).
    repeat: [Option<Instant>; 4],
    /// Disconnected-slot backoff: no probe before this instant.
    reprobe_at: Option<Instant>,
}

impl MenuPad {
    pub fn new(hwnd: HWND) -> Self {
        Self { hwnd, prev: 0, repeat: [None; 4], reprobe_at: None }
    }

    /// Sample controller 0 and return this frame's edges. Call once per
    /// frame-loop iteration (both the normal loop and the menu hold-loop).
    pub fn poll(&mut self) -> PadEdges {
        let mut e = PadEdges::default();
        // Focus gate (the flycam's): a background window gets nothing, and
        // clearing the state means refocusing can never see a stale edge.
        if unsafe { GetForegroundWindow() } != self.hwnd {
            self.prev = 0;
            self.repeat = [None; 4];
            return e;
        }
        let now = Instant::now();
        if let Some(t) = self.reprobe_at {
            if now < t {
                return e;
            }
            self.reprobe_at = None;
        }
        let mut st = XINPUT_STATE::default();
        if unsafe { XInputGetState(0, &mut st) } != ERROR_SUCCESS.0 {
            self.reprobe_at = Some(now + REPROBE);
            self.prev = 0;
            self.repeat = [None; 4];
            return e;
        }
        let g = &st.Gamepad;
        let btn = |m| (g.wButtons & m).0 != 0;
        let mut cur = 0u16;
        if btn(XINPUT_GAMEPAD_DPAD_UP) || g.sThumbLY > STICK_NAV {
            cur |= B_UP;
        }
        if btn(XINPUT_GAMEPAD_DPAD_DOWN) || g.sThumbLY < -STICK_NAV {
            cur |= B_DOWN;
        }
        if btn(XINPUT_GAMEPAD_DPAD_LEFT) || g.sThumbLX < -STICK_NAV {
            cur |= B_LEFT;
        }
        if btn(XINPUT_GAMEPAD_DPAD_RIGHT) || g.sThumbLX > STICK_NAV {
            cur |= B_RIGHT;
        }
        if btn(XINPUT_GAMEPAD_A) {
            cur |= B_A;
        }
        if btn(XINPUT_GAMEPAD_B) {
            cur |= B_B;
        }
        if btn(XINPUT_GAMEPAD_START) {
            cur |= B_START;
        }
        let newly = cur & !self.prev;
        self.prev = cur;

        // Buttons: pure edges.
        e.a = newly & B_A != 0;
        e.b = newly & B_B != 0;
        e.start = newly & B_START != 0;

        // Directions: edge fires immediately and arms the repeat; a held
        // direction refires each time its deadline passes. `now + RATE` (not
        // `deadline + RATE`) so a long frame never banks a burst of catch-up
        // fires.
        let dirs = [B_UP, B_DOWN, B_LEFT, B_RIGHT];
        let mut fired = [false; 4];
        for (i, &bit) in dirs.iter().enumerate() {
            if cur & bit == 0 {
                self.repeat[i] = None;
                continue;
            }
            if newly & bit != 0 {
                fired[i] = true;
                self.repeat[i] = Some(now + REPEAT_DELAY);
            } else if let Some(t) = self.repeat[i] {
                if now >= t {
                    fired[i] = true;
                    self.repeat[i] = Some(now + REPEAT_RATE);
                }
            }
        }
        e.up = fired[0];
        e.down = fired[1];
        e.left = fired[2];
        e.right = fired[3];
        e
    }
}
