//! 500 Hz wall-clock camera integrator (keyboard + mouse + gamepad).
//!
//! Camera motion used to be integrated once per rendered frame with
//! dt = frame time, but the main thread blocks for the whole trace (100+ ms
//! on heavy scenes), so a key tap counted as a full frame of motion when it
//! happened to span the event pump — or was lost outright. This thread
//! samples input every ~2 ms and integrates with the MEASURED tick dt.
//! Displacement is therefore an exact function of wall-clock time
//! regardless of render framerate — timer jitter changes granularity,
//! never totals. Rate inputs (the controller's right stick) are where this
//! matters most: "how long the stick was held at this angle" is integrated
//! at 2 ms granularity instead of per-frame over/undershoot.
//!
//! ONE INTEGRATOR, TWO SOURCES (B6b rung 2). Everything below `Source` is
//! platform-free: the smoothstep slow ramp, the eased movement vector, the
//! TOD scrub and the `Shared` atomics are ONE implementation that both
//! windows run. What differs is only how a tick's input is SAMPLED, and the
//! two ways are not alike:
//!
//! - **Windows (`src_win32`)** reads live OS state from any thread —
//!   `GetAsyncKeyState`, `GetCursorPos`, `XInputGetState`. SDL's own
//!   key/mouse state only updates when the blocked main thread pumps, so it
//!   is useless here.
//! - **Linux (`MirrorSource`)** cannot do that: Wayland exposes no
//!   off-thread keyboard state and `SDL_PumpEvents` is main-thread-only. So
//!   the main thread pumps into a `Mirror` of atomics and this thread reads
//!   THAT. The pump is the only writer, the integrator the only reader, and
//!   the look accumulators drain with a `swap(0)` so no motion is
//!   double-counted and none is lost.
//!
//! The seam is `Source` + `Tick`, and it is cut exactly where the OS reads
//! already were: nothing below it moved. `Tick` carries the clock as well as
//! the wait, which is what lets `self_test` drive the REAL loop at scripted
//! tick rates rather than a copy of it.
//!
//! The render loop consumes exactly ONE `snapshot()` per loop iteration and
//! uses only that snapshot everywhere in the iteration (trace pose == MV
//! pose == prev-capture pose — the temporal/replay/upscaler bit-equality
//! contracts). Any future teleport/reset must write through `set()`, never
//! a session-local `Camera` copy.

use crate::camera::Camera;
use glam::Vec3A;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
#[cfg(windows)]
const PAD_REPROBE_TICKS: u32 = 2000 / TICK_MS;
/// Time-of-day scrub rate: game-hours per real second held (`.`/`,` or D-pad
/// right/left). The Ctrl(/16)/Shift(/8)/bumper divisors apply, for fine scrub.
const TOD_RATE: f32 = 1.0;
/// Seconds for a slow-factor modifier (Ctrl/LB, Shift/RB) to ramp between
/// full speed and its divided speed. The ramp is smoothstep-shaped, so
/// engaging AND releasing both ease instead of stepping the speed 8-16x in
/// one tick.
const SLOW_EASE_S: f32 = 0.25;

/// Stick deadzones and the trigger threshold, in the i16/[0, 32767] units
/// BOTH pad backends report. The values are XInput's
/// (`XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE`, `..._RIGHT_...`, and
/// `XINPUT_GAMEPAD_TRIGGER_THRESHOLD` rescaled from its 0..255 range), so the
/// two pads feel identical rather than merely both working — `self_test`'s
/// Windows arm pins the first two against the crate constants they came from.
const PAD_DEADZONE_L: f32 = 7849.0;
const PAD_DEADZONE_R: f32 = 8689.0;
/// 30/255 of full scale — XInput's threshold expressed in SDL's range.
const TRIGGER_THRESHOLD_16: f32 = 30.0 / 255.0 * 32767.0;

/// One integrator snapshot: the camera pose of record plus the time-of-day
/// hour, taken under ONE lock so a render iteration sees a consistent pair.
#[derive(Clone, Copy)]
pub struct FlyState {
    pub cam: Camera,
    /// Hours, wrapped into [0, 24). Consumed by the session loop, which turns
    /// a delta into `scene::apply_tod` + the shading-change resets.
    pub tod: f32,
}

// The attractor type and its circular mean moved to world.rs, where the data
// they are derived from lives and where `--cinematic` (which has no window and
// so cannot depend on a windowed module) can reach them. Re-exported here so
// every existing call site — and the module's own doc comments — keep reading
// as they did.
pub use crate::world::{attractor_hour, TodAttractor};

// ---------------------------------------------------------------------------
// The keymap — ONE table, two platforms
// ---------------------------------------------------------------------------

/// What a physical key MEANS to the integrator. The mapping from keys to
/// these is per-platform; everything downstream speaks only in these.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Fwd,
    Back,
    Left,
    Right,
    Up,
    Down,
    SlowCtrl,
    SlowShift,
    TodRev,
    TodFwd,
}

/// The actions, in bit order. `Mirror`'s bitset is indexed by position here,
/// so this list IS the wire format between the pump and the integrator.
pub const ACTIONS: [Action; 10] = [
    Action::Fwd,
    Action::Back,
    Action::Left,
    Action::Right,
    Action::Up,
    Action::Down,
    Action::SlowCtrl,
    Action::SlowShift,
    Action::TodRev,
    Action::TodFwd,
];

/// One action's physical keys on each platform.
///
/// TWO SPELLINGS OF "LAYOUT-INDEPENDENT", because the platforms disagree
/// about what a key is. Windows' `GetAsyncKeyState` takes a VIRTUAL key, which
/// moves with the layout, so a letter key has to be resolved from its Set-1
/// SCANCODE through `MapVirtualKeyW` (that is what keeps AZERTY flying on the
/// same physical keys) while arrows and modifiers are layout-independent VKs
/// already. SDL reports a scancode directly, so it needs no round-trip at all.
/// Both halves live here rather than in their sources so `self_test` can check
/// the table on every platform, not just the native one.
pub struct KeyRow {
    pub action: Action,
    /// For the message, and for the self-test's failure text.
    pub name: &'static str,
    /// (Set-1 scancode, US-layout VK fallback if the layout maps it to none).
    pub win_scan: &'static [(u32, u16)],
    /// Layout-independent VKs that alias this action on Windows.
    pub win_vk: &'static [u16],
    /// `SDL_Scancode` values — pinned against the `Scancode` enum itself by
    /// `self_test`'s unix arm, so these numbers cannot rot silently.
    pub sdl_scan: &'static [i32],
}

// Windows VKs used below, as plain numbers so this table compiles everywhere.
// `self_test`'s Windows arm pins each against the `windows` crate constant.
const VK_SHIFT_N: u16 = 0x10;
const VK_CONTROL_N: u16 = 0x11;
const VK_LEFT_N: u16 = 0x25;
const VK_UP_N: u16 = 0x26;
const VK_RIGHT_N: u16 = 0x27;
const VK_DOWN_N: u16 = 0x28;
const VK_OEM_COMMA_N: u16 = 0xBC;
const VK_OEM_PERIOD_N: u16 = 0xBE;

/// The one keymap. Space is deliberately absent: it is the render-mode cycle
/// (`input.rs`), and a flight key polled here would ALSO fire on every mode
/// switch — the camera bumped upward per press.
pub const KEYMAP: [KeyRow; 10] = [
    KeyRow {
        action: Action::Fwd,
        name: "forward (W / Up)",
        win_scan: &[(0x11, b'W' as u16)],
        win_vk: &[VK_UP_N],
        sdl_scan: &[26, 82],
    },
    KeyRow {
        action: Action::Back,
        name: "back (S / Down)",
        win_scan: &[(0x1F, b'S' as u16)],
        win_vk: &[VK_DOWN_N],
        sdl_scan: &[22, 81],
    },
    KeyRow {
        action: Action::Left,
        name: "left (A / Left)",
        win_scan: &[(0x1E, b'A' as u16)],
        win_vk: &[VK_LEFT_N],
        sdl_scan: &[4, 80],
    },
    KeyRow {
        action: Action::Right,
        name: "right (D / Right)",
        win_scan: &[(0x20, b'D' as u16)],
        win_vk: &[VK_RIGHT_N],
        sdl_scan: &[7, 79],
    },
    KeyRow {
        action: Action::Up,
        name: "up (E)",
        win_scan: &[(0x12, b'E' as u16)],
        win_vk: &[],
        sdl_scan: &[8],
    },
    KeyRow {
        action: Action::Down,
        name: "down (Q)",
        win_scan: &[(0x10, b'Q' as u16)],
        win_vk: &[],
        sdl_scan: &[20],
    },
    KeyRow {
        action: Action::SlowCtrl,
        name: "slow /16 (Ctrl)",
        win_scan: &[],
        win_vk: &[VK_CONTROL_N],
        // Left and right Ctrl. SDL reports them as distinct scancodes; the
        // Windows VK is the merged one, which is why the arities differ.
        sdl_scan: &[224, 228],
    },
    KeyRow {
        action: Action::SlowShift,
        name: "slow /8 (Shift)",
        win_scan: &[],
        win_vk: &[VK_SHIFT_N],
        sdl_scan: &[225, 229],
    },
    KeyRow {
        action: Action::TodRev,
        name: "time reverse (,)",
        win_scan: &[(0x33, VK_OEM_COMMA_N)],
        win_vk: &[],
        sdl_scan: &[54],
    },
    KeyRow {
        action: Action::TodFwd,
        name: "time forward (.)",
        win_scan: &[(0x34, VK_OEM_PERIOD_N)],
        win_vk: &[],
        sdl_scan: &[55],
    },
];

/// The SDL scancode -> action lookup the pump uses.
///
/// A MATCH ON THE ENUM, not a scan of `KEYMAP`'s numbers, and that is the
/// point: this is compiler-checked against the crate's own `Scancode`, and
/// `self_test` asserts the two agree in BOTH directions — so `KEYMAP`'s
/// integers are pinned to the enum rather than merely believed.
#[cfg(all(unix, not(target_os = "macos")))]
pub fn action_for_scancode(sc: sdl3::keyboard::Scancode) -> Option<Action> {
    use sdl3::keyboard::Scancode as S;
    Some(match sc {
        S::W | S::Up => Action::Fwd,
        S::S | S::Down => Action::Back,
        S::A | S::Left => Action::Left,
        S::D | S::Right => Action::Right,
        S::E => Action::Up,
        S::Q => Action::Down,
        S::LCtrl | S::RCtrl => Action::SlowCtrl,
        S::LShift | S::RShift => Action::SlowShift,
        S::Comma => Action::TodRev,
        S::Period => Action::TodFwd,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// The seam: one tick's input, and where it comes from
// ---------------------------------------------------------------------------

/// One decoded controller sample: sticks deadzoned + curved to [-1, 1],
/// triggers thresholded to [0, 1], shoulder buttons raw.
pub struct Pad {
    pub lx: f32,
    pub ly: f32,
    pub rx: f32,
    pub ry: f32,
    pub lt: f32,
    pub rt: f32,
    pub lb: bool,
    pub rb: bool,
    /// D-pad left/right: time-of-day reverse / fast-forward.
    pub dpad_l: bool,
    pub dpad_r: bool,
}

/// Everything one tick of the integrator needs from the OS, already resolved
/// to meanings. Built by a `Source`; consumed by `integrate_loop` and nothing
/// else.
#[derive(Default)]
pub struct Raw {
    pub fwd: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub slow_ctrl: bool,
    pub slow_shift: bool,
    pub tod_rev: bool,
    pub tod_fwd: bool,
    /// Mouse-look delta in PIXELS for this tick — already differenced, and
    /// already zero unless a look-drag is latched. The drag latch is the
    /// source's, because what may START one is a platform question.
    pub look: (f32, f32),
    pub pad: Option<Pad>,
}

impl Raw {
    /// Every input at rest. What an unfocused tick gets while a QA drive runs
    /// (which is exempt from the focus gate) so background keys, mouse and pad
    /// cannot leak into a scripted run.
    fn none() -> Raw {
        Raw::default()
    }

    fn key_any(&self) -> bool {
        self.fwd || self.back || self.left || self.right || self.up || self.down
    }
}

/// Where a tick's input comes from. Implemented once per window.
///
/// `focused` is asked FIRST and separately because a gated tick must not
/// sample the OS at all — that ordering is what keeps a paused session from
/// advancing a pad's reprobe backoff, and it is why the drag latch gets its
/// own hook rather than being inferred from a zero delta.
pub trait Source {
    /// Does our window have input focus?
    fn focused(&mut self) -> bool;
    /// Sample one tick. Called only on ticks that will actually integrate.
    fn tick(&mut self) -> Raw;
    /// Drop any latched look-drag — a gated tick, or focus loss.
    fn drop_drag(&mut self);
}

/// The tick source, and the integrator's clock with it.
///
/// ONE TRAIT FOR BOTH because they are one question: "how long was the last
/// tick" is only answerable by whatever did the waiting, and splitting them
/// would let a test script the rate without scripting the dt — which is
/// exactly the bug this whole thread exists to prevent, reintroduced in the
/// harness that is supposed to catch it.
pub trait Tick {
    /// Run once on the integrator thread before the first wait.
    fn on_thread_start(&self) {}
    /// Block until the next tick; return the elapsed time since the previous
    /// return (the full loop period, work included).
    fn wait(&mut self) -> Duration;
}

// ---------------------------------------------------------------------------
// The mirror — a pump's writes, an integrator's reads
// ---------------------------------------------------------------------------

/// Input state published by a main-thread event pump for the integrator to
/// read (B6b rung 2, the Linux window).
///
/// WHY THIS EXISTS AT ALL: `GetAsyncKeyState`'s portable equivalent does not.
/// Wayland has no off-thread keyboard state and `SDL_PumpEvents` may only be
/// called from the thread that made the window, so the integrator cannot ask
/// the OS anything — the pump has to tell it. `Relaxed` is enough throughout:
/// no field here orders another, and a tick that reads a key one 2 ms window
/// late is indistinguishable from the key having been pressed 2 ms later.
///
/// WHO WRITES WHAT, stated exactly because the obvious summary ("one writer")
/// is not true and the next edit will rest on this. The PUMP owns `keys`,
/// `focused`, the pad fields and the pump stamps outright — nothing else ever
/// writes them, which is what licenses `key`'s load-then-conditional-store
/// (a plain RMW that a second writer would corrupt). The DRAG LATCH and the
/// look accumulators have two writers: the pump latches and clears them, and
/// the INTEGRATOR clears them too, through `drop_drag` on every gated tick.
/// That pair is race-free in the only way that matters — every write is an
/// atomic `swap`/`fetch_add`/`store`, and both writers are trying to reach the
/// same state (drag down, motion discarded) — but it costs one thing worth
/// naming: a button-down landing in the same instant as a gated tick's clear
/// can lose the press, and the user re-clicks. Anything that wants finer
/// behaviour than that must stop making the latch shared, not add ordering.
///
/// `look` ACCUMULATES rather than latching a position, and that difference is
/// load-bearing: the pump runs faster than the integrator, so a tick must
/// collect every motion event since the last one — `drain_look`'s `swap(0)` is
/// what makes each pixel counted exactly once.
///
/// COMPILED EVERYWHERE, WRITTEN ONLY ON LINUX: `self_test`'s T6 exercises this
/// as pure data on every platform (it is the wire between two threads, and a
/// wire is checkable without either of them), so it is not `#[cfg]`-ed away —
/// but no Windows caller writes it, hence the attribute rather than a warning
/// nobody can act on.
#[cfg_attr(not(all(unix, not(target_os = "macos"))), allow(dead_code))]
pub struct Mirror {
    /// One bit per `ACTIONS` position.
    keys: AtomicU32,
    /// Mouse-look motion accumulated since the last drain, in 1/`LOOK_FIXED`
    /// of a pixel. See `LOOK_FIXED` for why this is not just a pixel count.
    look_x: AtomicI32,
    look_y: AtomicI32,
    /// A look-drag is latched: the primary button went down over our window
    /// and has not come up. The pump owns the "over our window" half — a
    /// button event only reaches us when it lands on us.
    drag: AtomicBool,
    focused: AtomicBool,
    /// lx, ly, rx, ry, lt, rt as raw i16 in SDL's own range, undecoded: the
    /// deadzone and the curve live in `MirrorSource` beside the Windows
    /// path's, so the two pads cannot drift apart in feel.
    pad_axes: [AtomicI32; 6],
    /// lb, rb, dpad_l, dpad_r in bits 0..3; bit 31 = a pad is connected.
    pad_btn: AtomicU32,
    /// The last PUMP PASS, and the gap before it, in nanoseconds since `base`.
    ///
    /// WHAT THIS MEASURES, and two drafts got it wrong before this one. "Time
    /// since the last input EVENT" is idle time — it read ~2 s in both arms.
    /// "Age at the moment the frame samples" is ~0 in BOTH arms too, because
    /// the inline arm pumps immediately before sampling; freshness at that
    /// instant was never the difference. The difference is HOW OFTEN THE OS IS
    /// ASKED: 1 ms on the pump thread against a frame on the render thread —
    /// and that interval is exactly the bound on how short a key tap can be
    /// before its down and its up land in the same drain and the press is lost
    /// outright, which is the bug the 500 Hz integrator exists for.
    pumped_at: AtomicU64,
    pump_gap: AtomicU64,
    base: Instant,
}

/// `pad_btn` bit 31: a gamepad is connected. Separate from the axes being
/// zero, which is also what a centred stick reads.
const PAD_PRESENT: u32 = 1 << 31;

/// Sub-pixel units per pixel in the look accumulators.
///
/// SDL3 REPORTS `xrel`/`yrel` AS f32 AND MEANS IT. Windows differences the
/// integer `GetCursorPos`, so there is nothing below a pixel to lose there —
/// but Wayland carries pointer motion as `wl_fixed` (1/256 px) and then
/// DIVIDES it by the output scale, so a fractional-scale desktop delivers
/// deltas like 0.667 as a matter of course. Accumulating those as `as i32`
/// truncated every one of them to nothing: a slow drag, or any drag at all on
/// a high-poll-rate mouse whose per-event delta never reaches 1.0, moved the
/// view not at all — and it would have looked like a dead code path rather
/// than a rounding choice. 256 matches Wayland's own fixed-point, so a scale
/// of 1 round-trips exactly; the accumulator's headroom is then ±8.4 M pixels
/// between drains, against the ~2 ms of motion actually collected.
const LOOK_FIXED: f32 = 256.0;

impl Default for Mirror {
    fn default() -> Self {
        Mirror::new()
    }
}

#[cfg_attr(not(all(unix, not(target_os = "macos"))), allow(dead_code))]
impl Mirror {
    pub fn new() -> Mirror {
        Mirror {
            keys: AtomicU32::new(0),
            look_x: AtomicI32::new(0),
            look_y: AtomicI32::new(0),
            drag: AtomicBool::new(false),
            focused: AtomicBool::new(true),
            pad_axes: Default::default(),
            pad_btn: AtomicU32::new(0),
            pumped_at: AtomicU64::new(0),
            pump_gap: AtomicU64::new(0),
            base: Instant::now(),
        }
    }

    /// One pump pass happened, whether or not it found anything. Called by the
    /// pump and nothing else.
    pub fn pumped(&self) {
        let now = self.base.elapsed().as_nanos() as u64;
        let prev = self.pumped_at.swap(now, Relaxed);
        if prev != 0 {
            self.pump_gap.store(now.saturating_sub(prev), Relaxed);
        }
    }

    /// The most recent interval between pump passes. Sampled once per frame by
    /// the render loop and reported; nothing reads it to make a decision.
    pub fn pump_gap(&self) -> Duration {
        Duration::from_nanos(self.pump_gap.load(Relaxed))
    }

    /// A key went down or up. Idempotent by construction, so an OS
    /// auto-repeat's extra downs are harmless — the pump filters them anyway,
    /// because a held key should write nothing at all.
    pub fn key(&self, action: Action, down: bool) {
        let Some(i) = ACTIONS.iter().position(|a| *a == action) else { return };
        let bit = 1u32 << i;
        // Load-then-conditional-store: a held key repeats no writes, so an
        // idle session's mirror is as read-only as the camera it feeds.
        let cur = self.keys.load(Relaxed);
        let next = if down { cur | bit } else { cur & !bit };
        if next != cur {
            self.keys.store(next, Relaxed);
        }
    }

    /// Relative mouse motion, in pixels.
    ///
    /// GATED ON THE DRAG, which is what makes the accumulator mean "motion
    /// this drag has not yet been charged for" rather than "motion since
    /// whenever". Without it, moving the cursor across the window and THEN
    /// pressing would snap the view by the whole traverse.
    /// SUB-PIXEL, in `LOOK_FIXED` units — see that constant for why a plain
    /// `as i32` here silently deleted fractional motion.
    pub fn look(&self, dx: f32, dy: f32) {
        if (dx == 0.0 && dy == 0.0) || !self.drag.load(Relaxed) {
            return;
        }
        // `round`, not a truncation: two 0.5 px events must sum to 1 px, and
        // truncating toward zero would make them sum to 0 forever.
        self.look_x.fetch_add((dx * LOOK_FIXED).round() as i32, Relaxed);
        self.look_y.fetch_add((dy * LOOK_FIXED).round() as i32, Relaxed);
    }

    /// The primary mouse button, over our window.
    pub fn set_drag(&self, down: bool) {
        if self.drag.swap(down, Relaxed) != down {
            // EITHER transition clears. Release discards motion the drag never
            // got charged for (the pump can outrun a tick); press clears
            // whatever a race let through between the two stores above.
            self.look_x.store(0, Relaxed);
            self.look_y.store(0, Relaxed);
        }
    }

    /// Focus arrived or left. Losing it PARKS EVERY HELD INPUT, structurally.
    ///
    /// SDL does synthesize a key-up for each held key when a window loses
    /// keyboard focus, so in practice the bitset clears itself — but "in
    /// practice" is the wrong footing for this one. The Windows source reads
    /// live OS state and so cannot hold a stale key by construction; here a
    /// key-up that never arrives leaves the bit set, and the camera then flies
    /// the instant focus returns, with nothing on the keyboard pressed. That
    /// is the failure the focus gate exists to prevent, reintroduced one layer
    /// down. One store makes it a branch instead of a trust.
    pub fn set_focused(&self, f: bool) {
        if self.focused.swap(f, Relaxed) != f && !f {
            self.keys.store(0, Relaxed);
            // The pad needs the same treatment for the same reason and one
            // exception: SDL stops delivering its events to an unfocused window
            // by default, so a stick held across the transition stays latched
            // at its last deflection. Buttons and axes park; PAD_PRESENT does
            // NOT, because nothing re-announces a device that never left and
            // the pad would be gone for the rest of the session.
            self.pad_btn.fetch_and(PAD_PRESENT, Relaxed);
            for a in &self.pad_axes {
                a.store(0, Relaxed);
            }
            self.set_drag(false);
        }
    }

    /// A pad axis, in SDL's raw i16 range. `i` indexes lx, ly, rx, ry, lt, rt.
    pub fn pad_axis(&self, i: usize, v: i16) {
        let Some(a) = self.pad_axes.get(i) else { return };
        let _ = a.swap(v as i32, Relaxed);
    }

    /// A pad button. `i` indexes lb, rb, dpad_l, dpad_r.
    pub fn pad_button(&self, i: u32, down: bool) {
        if i > 3 {
            return;
        }
        let bit = 1u32 << i;
        let cur = self.pad_btn.load(Relaxed);
        let next = if down { cur | bit } else { cur & !bit };
        if next != cur {
            self.pad_btn.store(next, Relaxed);
        }
    }

    /// A pad arrived or left. Leaving zeroes the axes: a disconnect mid-tilt
    /// would otherwise fly the camera forever.
    pub fn pad_present(&self, present: bool) {
        let cur = self.pad_btn.load(Relaxed);
        let next = if present { cur | PAD_PRESENT } else { 0 };
        if next != cur {
            self.pad_btn.store(next, Relaxed);
            if !present {
                for a in &self.pad_axes {
                    a.store(0, Relaxed);
                }
            }
        }
    }

    /// The raw state, for the `--qa` `pos` verb: (key bits, focused, drag
    /// latched). A readout, never a decision — and the one that separates "the
    /// key never reached the mirror" from "the tick was gated", which is a
    /// distinction no pose alone can make.
    pub fn debug_state(&self) -> (u32, bool, bool) {
        (self.keys.load(Relaxed), self.focused.load(Relaxed), self.drag.load(Relaxed))
    }

    fn drain_look(&self) -> (f32, f32) {
        (
            self.look_x.swap(0, Relaxed) as f32 / LOOK_FIXED,
            self.look_y.swap(0, Relaxed) as f32 / LOOK_FIXED,
        )
    }

    fn held(&self, action: Action) -> bool {
        ACTIONS
            .iter()
            .position(|a| *a == action)
            .is_some_and(|i| self.keys.load(Relaxed) & (1 << i) != 0)
    }
}

/// The `Source` over a `Mirror` — the Linux window's, and the one a scripted
/// harness can drive without an OS at all.
pub struct MirrorSource {
    m: Arc<Mirror>,
}

impl MirrorSource {
    pub fn new(m: Arc<Mirror>) -> MirrorSource {
        MirrorSource { m }
    }
}

impl Source for MirrorSource {
    fn focused(&mut self) -> bool {
        self.m.focused.load(Relaxed)
    }

    fn tick(&mut self) -> Raw {
        let btn = self.m.pad_btn.load(Relaxed);
        let ax = |i: usize| self.m.pad_axes[i].load(Relaxed) as i16;
        let pad = (btn & PAD_PRESENT != 0).then(|| {
            let (lx, ly) = stick(ax(0), ax(1), PAD_DEADZONE_L);
            let (rx, ry) = stick(ax(2), ax(3), PAD_DEADZONE_R);
            Pad {
                lx,
                // SDL reports stick Y DOWN-positive, XInput up-positive. The
                // integrator's basis is the latter (`ly` is forward), so the
                // flip belongs here, at the one place that knows which pad
                // this is — not in the shared math.
                ly: -ly,
                rx,
                ry: -ry,
                lt: trigger16(ax(4)),
                rt: trigger16(ax(5)),
                lb: btn & 1 != 0,
                rb: btn & 2 != 0,
                dpad_l: btn & 4 != 0,
                dpad_r: btn & 8 != 0,
            }
        });
        // The drag gate is the mirror's, so a motion event that arrived while
        // no button was down is discarded rather than applied late.
        let look = if self.m.drag.load(Relaxed) { self.m.drain_look() } else { (0.0, 0.0) };
        Raw {
            fwd: self.m.held(Action::Fwd),
            back: self.m.held(Action::Back),
            left: self.m.held(Action::Left),
            right: self.m.held(Action::Right),
            up: self.m.held(Action::Up),
            down: self.m.held(Action::Down),
            slow_ctrl: self.m.held(Action::SlowCtrl),
            slow_shift: self.m.held(Action::SlowShift),
            tod_rev: self.m.held(Action::TodRev),
            tod_fwd: self.m.held(Action::TodFwd),
            look,
            pad,
        }
    }

    fn drop_drag(&mut self) {
        self.m.set_drag(false);
    }
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

/// Smoothstep shaping for a ramp position in [0, 1]. Exact at both endpoints
/// (0*0*3 = 0, 1*1*(3-2) = 1), which is what makes the rest states below exact.
fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// The flight-speed divisor for the two slow modifiers' ramp positions:
/// Ctrl /16, Shift /8, composing to /128 when both are held.
///
/// APPLIED IN LOG2 SPACE — one `exp2` of a lerped EXPONENT rather than two
/// `powf`s of a lerped base. The rest states come out EXACT (2^0 = 1,
/// 2^-4 = 1/16, 2^-3 = 1/8, 2^-7 = 1/128), which is what keeps an UNMODIFIED
/// session flying at exactly its speed and every bit-comparison downstream
/// meaningful; `self_test`'s T2 pins all four.
///
/// MEASURED, because the obvious claim about this is wrong and was written
/// down here once: the arithmetic-space spelling
/// `(1/16).powf(smooth(ctrl)) * (1/8).powf(smooth(shift))` is exact at all
/// four rest states TOO — `powf` is correctly rounded at exponent 0 and 1 —
/// and differs only mid-ramp (56% of a 1000x1000 sweep, worst relative
/// 3.6e-7). So T2 does not distinguish the two spellings and does not claim
/// to: what it pins is the -4 / -3 EXPONENTS and the smoothstep's exact
/// endpoints, which is the thing a retune would actually break.
fn slow_factor(ctrl: f32, shift: f32) -> f32 {
    (-4.0 * smooth(ctrl) - 3.0 * smooth(shift)).exp2()
}

/// A trigger in SDL's 0..32767 range. XInput's 0..255 twin is
/// `src_win32::trigger`; both threshold at the same fraction of full scale.
fn trigger16(t: i16) -> f32 {
    let t = t.max(0) as f32;
    if t <= TRIGGER_THRESHOLD_16 {
        return 0.0;
    }
    (t - TRIGGER_THRESHOLD_16) / (32767.0 - TRIGGER_THRESHOLD_16)
}

/// The Linux tick source: plain sleep. The waitable-timer machinery the
/// Windows half needs has no counterpart to earn here — `nanosleep` already
/// overshoots a 2 ms request by tens of microseconds, well inside the tick —
/// and integration uses measured dt regardless, so this affects granularity
/// and never displacement.
///
/// NOT thread-priority-raised, unlike Windows. Raising it needs privileges we
/// do not ask for, and a thread that uses ~10 us per 2 ms accrues enough CFS
/// credit to be scheduled promptly against saturated rayon workers — but that
/// is a prediction, and the tick-dt distribution this rung measures is what
/// says whether it held.
#[cfg(all(unix, not(target_os = "macos")))]
pub struct SleepTicker {
    last: Instant,
}

#[cfg(all(unix, not(target_os = "macos")))]
impl Default for SleepTicker {
    fn default() -> Self {
        SleepTicker::new()
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
impl SleepTicker {
    pub fn new() -> SleepTicker {
        SleepTicker { last: Instant::now() }
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
impl Tick for SleepTicker {
    fn wait(&mut self) -> Duration {
        std::thread::sleep(Duration::from_millis(TICK_MS as u64));
        let now = Instant::now();
        let dt = now - self.last;
        self.last = now;
        dt
    }
}

// ---------------------------------------------------------------------------
// The thread, and its shared state
// ---------------------------------------------------------------------------

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
    /// World auto-TOD sticky-off: set by the first manual scrub (integrator
    /// side) OR a pause-menu `set_tod` (main thread) — either way the user
    /// took the clock, and the attractors easing it back would undo them.
    /// Shared (not an integrator local) precisely so the menu can set it.
    manual_tod: AtomicBool,
    /// Camera speed in world units/s (f32 bits), written by the integrator
    /// each moving tick and read by the audio mixer's wind synth at the
    /// audio callback rate — an Arc so the mixer can hold its own handle
    /// (the render thread being blocked in a trace must not gate the wind).
    /// Deliberately NOT a FlyState field: the one-lock snapshot contract
    /// stays exactly what it was. Idle steady state writes nothing (the
    /// snapshot bit-compare discipline); pause/unfocus store one 0.0.
    speed: Arc<AtomicU32>,
    /// QA drive override (the --qa `drive` verb): axes in [-1, 1] as f32
    /// bits (x = camera right, y = world up, z = forward) + ticks left at
    /// the 500 Hz rate. Injected as SYNTHETIC PAD AXES after the OS
    /// sampling (the analog path — deflection scales speed), so a scripted
    /// strafe rides the exact integration the sticks do. Exempt from the
    /// FOCUS gate (an unfocused window is a scripted driver's normal
    /// state; while driving unfocused the OS keys/mouse/pad are zeroed so
    /// background typing still cannot leak in) but NEVER from the pause
    /// gate (paused = no frames presented = flying blind).
    drive: [AtomicU32; 3],
    drive_ticks: AtomicU32,
    /// Keyboard flight ease in seconds (f32 bits) — see `camera::move_ease`.
    /// Shared (not a spawn-time constant) so the pause menu's Live row can
    /// move it mid-session; 0 is the hard-step arm. Read once per tick.
    /// Deliberately does NOT reach the analog stick / trigger / QA-drive
    /// path, which is already a throttle.
    move_ease: AtomicU32,
    /// Tick accounting since the last `tick_stats()` read: count, summed dt in
    /// nanoseconds, and the WORST single dt.
    ///
    /// MAX RATHER THAN A p99, and that is the stricter claim: the failure being
    /// hunted is a STARVED tick — the integrator descheduled behind saturated
    /// rayon workers — and one 40 ms gap is exactly what a p99 over hundreds of
    /// 2 ms ticks would hide. Displacement survives it either way (dt is
    /// measured), so what this bounds is input GRANULARITY, which is the thing
    /// the 500 Hz rate was bought for.
    ticks: AtomicU64,
    tick_ns: AtomicU64,
    tick_max_ns: AtomicU64,
    /// Raised by `set()` (a teleport), consumed and cleared by the
    /// integrator: the ramp must be parked so a pose write ARRIVES STOPPED.
    /// Without it a `tp` mid-flight keeps coasting through the settle, which
    /// would make a scripted `tp` -> `sync` -> `screenshot` non-repeatable.
    motion_reset: AtomicBool,
}

// Several of these answer to the Windows session and the `--qa` socket only —
// `pause`/`resume` to its resize re-entry, `drive`/`set`/`set_tod` to the
// socket's verbs, `speed_handle` to the audio wind. The Linux window reaches
// none of them yet (its rungs are 3 and later), and the module is no longer
// `#[cfg(windows)]`, so say so rather than let a real dead item hide in the
// noise. The `qa`/`settings` modules carry the same attribute for the same
// reason.
#[cfg_attr(not(windows), allow(dead_code))]
impl FlyCam {
    /// Spawn the integrator over an arbitrary source + tick, PAUSED (see
    /// `resume`). The platform constructors below are what sessions call;
    /// this one is also what `self_test` drives, which is the whole reason it
    /// is generic rather than two copies.
    ///
    /// `attractors` arms world-mode auto-TOD (empty = off, the structural
    /// non-world state); the first manual scrub disables it for the session.
    /// `move_ease` seeds the keyboard flight ease in seconds (`--move-ease`).
    /// FACTORIES, NOT VALUES, and `tools/win-cross-check.sh` is what proved
    /// they have to be: the Windows source holds an `HWND` and its ticker an
    /// `HANDLE`, both raw pointers and therefore neither `Send` — so
    /// constructing them here and moving them in does not compile on the
    /// platform this machine cannot build. It is also what the pre-split code
    /// did (only `hwnd: isize` ever crossed the boundary; `Ticker::new` and
    /// the key-layout resolve ran inside `integrate_loop`), so this is that
    /// property restored rather than a new constraint.
    pub fn spawn<S, T>(
        mk_src: impl FnOnce() -> S + Send + 'static,
        mk_tick: impl FnOnce() -> T + Send + 'static,
        cam0: Camera,
        tod0: f32,
        diag: f32,
        attractors: Vec<TodAttractor>,
        move_ease: f32,
    ) -> Self
    where
        S: Source,
        T: Tick,
    {
        let shared = Arc::new(Shared::new(cam0, tod0, move_ease));
        let s2 = shared.clone();
        let handle = std::thread::Builder::new()
            .name("flycam".into())
            .spawn(move || {
                let (mut src, mut ticker) = (mk_src(), mk_tick());
                integrate_loop(&s2, &mut src, &mut ticker, diag, &attractors)
            })
            .expect("flycam thread spawn failed");
        Self { shared, handle: Some(handle) }
    }

    /// The Windows session's: live OS state read from this thread, and the
    /// high-resolution waitable timer. `hwnd` rides as isize because
    /// windows::HWND is a raw pointer and not Send; the thread only ever
    /// compares it / hands it to read-only queries.
    #[cfg(windows)]
    pub fn spawn_win32(
        hwnd: isize,
        cam0: Camera,
        tod0: f32,
        diag: f32,
        attractors: Vec<TodAttractor>,
        move_ease: f32,
    ) -> Self {
        FlyCam::spawn(
            move || src_win32::Win32Source::new(hwnd),
            src_win32::Ticker::new,
            cam0,
            tod0,
            diag,
            attractors,
            move_ease,
        )
    }

    /// The Linux window's: a main-thread pump's `Mirror`, and a sleeping
    /// ticker. The caller keeps its own handle on the mirror — it is the pump
    /// that writes it.
    #[cfg(all(unix, not(target_os = "macos")))]
    pub fn spawn_mirror(
        mirror: Arc<Mirror>,
        cam0: Camera,
        tod0: f32,
        diag: f32,
        attractors: Vec<TodAttractor>,
        move_ease: f32,
    ) -> Self {
        FlyCam::spawn(
            move || MirrorSource::new(mirror),
            SleepTicker::new,
            cam0,
            tod0,
            diag,
            attractors,
            move_ease,
        )
    }

    /// The pose + time-of-day of record. Call ONCE per render-loop iteration
    /// and use only the returned snapshot for the whole iteration.
    pub fn snapshot(&self) -> FlyState {
        *self.shared.state.lock().unwrap()
    }

    /// A clone of the integrator's live camera-speed atomic (world units/s
    /// as f32 bits) — the audio wind synth's input. See `Shared::speed`.
    pub fn speed_handle(&self) -> Arc<AtomicU32> {
        self.shared.speed.clone()
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

    /// Write-through for teleports/resets — the --qa `tp`/`look` verbs'
    /// entry (written for exactly this before anything called it): a pose
    /// write goes through the shared camera instead of a session-local copy
    /// the integrator would immediately overwrite.
    pub fn set(&self, cam: Camera) {
        // A teleport must ARRIVE STOPPED — otherwise the keyboard ease keeps
        // coasting through the new pose and a scripted `tp` -> `sync` ->
        // `screenshot` drifts during the settle.
        self.shared.motion_reset.store(true, Relaxed);
        self.shared.state.lock().unwrap().cam = cam;
    }

    /// Keyboard flight ease in seconds — the pause menu's Live row (and the
    /// `--move-ease` seed). 0 restores the hard step; negative is clamped
    /// away so the integrator's `> 0.0` arm is the only predicate.
    pub fn set_move_ease(&self, secs: f32) {
        if secs.is_finite() {
            self.shared.move_ease.store(secs.max(0.0).to_bits(), Relaxed);
        }
    }

    /// The live keyboard flight ease, for the menu row's displayed value and
    /// the base its adjust steps from.
    pub fn move_ease(&self) -> f32 {
        f32::from_bits(self.shared.move_ease.load(Relaxed))
    }

    /// Arm the QA drive: axes in [-1, 1] for `ticks` 500 Hz integrator
    /// ticks (`ticks` 0 = stop). Non-finite axes are refused here — NaN
    /// survives `clamp` (the diamondmine lesson: `"nan".parse()` succeeds),
    /// and a NaN position is a wedged session.
    pub fn drive(&self, x: f32, y: f32, z: f32, ticks: u32) -> bool {
        if !(x.is_finite() && y.is_finite() && z.is_finite()) {
            return false;
        }
        self.shared.drive[0].store(x.clamp(-1.0, 1.0).to_bits(), Relaxed);
        self.shared.drive[1].store(y.clamp(-1.0, 1.0).to_bits(), Relaxed);
        self.shared.drive[2].store(z.clamp(-1.0, 1.0).to_bits(), Relaxed);
        self.shared.drive_ticks.store(ticks, Relaxed);
        true
    }

    // Read by the Vulkan window's `--qa pos` verb; D3D12's own readout is the
    // HUD's, so this is dead on Windows and says so.
    #[cfg_attr(windows, allow(dead_code))]
    /// Tick count, mean dt (ms) and worst dt (ms) since the last call, which
    /// RESETS the window — so a caller polling once a second reports that
    /// second rather than the session's average, and one early stall does not
    /// hide behind a million clean ticks.
    pub fn tick_stats(&self) -> (u64, f64, f64) {
        let n = self.shared.ticks.swap(0, Relaxed);
        let sum = self.shared.tick_ns.swap(0, Relaxed);
        let max = self.shared.tick_max_ns.swap(0, Relaxed);
        let mean = if n == 0 { 0.0 } else { sum as f64 / n as f64 / 1e6 };
        (n, mean, max as f64 / 1e6)
    }

    /// Time-of-day write-through (the pause menu's TOD control): the thread
    /// owns the hour like it owns the pose, so a menu set rides the same
    /// channel a scrub does — the session's existing `snap.tod != cur_tod`
    /// detection applies it (scene::apply_tod + refresh_sky + frame reset,
    /// histories kept). Also takes the clock for the session, like the first
    /// manual scrub — a menu set IS a manual choice, and the world-mode
    /// attractors easing it back would undo it.
    pub fn set_tod(&self, hours: f32) {
        self.shared.manual_tod.store(true, Relaxed);
        self.shared.state.lock().unwrap().tod = hours.rem_euclid(24.0);
    }
}

impl Shared {
    fn new(cam0: Camera, tod0: f32, move_ease: f32) -> Shared {
        Shared {
            state: Mutex::new(FlyState { cam: cam0, tod: tod0 }),
            stop: AtomicBool::new(false),
            paused: AtomicBool::new(true),
            manual_tod: AtomicBool::new(false),
            speed: Arc::new(AtomicU32::new(0)),
            drive: [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)],
            drive_ticks: AtomicU32::new(0),
            move_ease: AtomicU32::new(move_ease.max(0.0).to_bits()),
            ticks: AtomicU64::new(0),
            tick_ns: AtomicU64::new(0),
            tick_max_ns: AtomicU64::new(0),
            motion_reset: AtomicBool::new(false),
        }
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

// ---------------------------------------------------------------------------
// The integrator — platform-free from here down
// ---------------------------------------------------------------------------

fn integrate_loop<S: Source, T: Tick>(
    shared: &Shared,
    src: &mut S,
    ticker: &mut T,
    diag: f32,
    attractors: &[TodAttractor],
) {
    ticker.on_thread_start();

    // Slow-modifier ramp positions in [0, 1]: 0 = full speed, 1 = fully
    // engaged divisor. Advanced every focused tick (even idle ones, so a
    // modifier held before movement starts is already engaged).
    let mut slow_ctrl = 0.0f32;
    let mut slow_shift = 0.0f32;
    // Keyboard flight ramp, CAMERA-RELATIVE (x = right, y = up, z = forward) —
    // the analog stick's own basis, which is what lets it be advanced BEFORE
    // the pose lock is taken and keeps the idle early-out below a pure input
    // test. |mv| <= 1 by construction (see camera::move_ease); the ramp reaches
    // a BITWISE zero on release, which is what re-arms that early-out.
    let mut mv = Vec3A::ZERO;
    // Last speed value stored (world units/s). Compare-before-store keeps
    // the idle steady state write-free (the snapshot bit-compare
    // discipline); the pause/unfocus and no-input paths park it at 0.0 once.
    let mut prev_speed = 0.0f32;

    loop {
        // Same suspend/hitch clamp the old per-frame dt had. Note the clock
        // advances BEFORE both gates below, so a paused/unfocused span
        // advances it instead of accumulating into the first tick after it.
        let waited = ticker.wait();
        let dt = waited.as_secs_f32().min(0.1);
        if shared.stop.load(Relaxed) {
            break;
        }
        // BEFORE the gates, and counting the UNCLAMPED wait: a tick starved by
        // the render threads is precisely the one that never reaches the
        // integration below, so accounting for it after the gates would report
        // only the ticks that already went well.
        {
            let ns = waited.as_nanos() as u64;
            shared.ticks.fetch_add(1, Relaxed);
            shared.tick_ns.fetch_add(ns, Relaxed);
            shared.tick_max_ns.fetch_max(ns, Relaxed);
        }

        // Focus gate: the OS inputs are global — only act when our window has
        // focus, and drop any latched drag on focus loss.
        // Pause gate: no frames are being presented (session rebuild), so
        // integrating would fly the camera blind. Both drop the drag.
        // A live QA drive is EXEMPT from the focus gate only (a scripted
        // driver's window is normally unfocused); the OS inputs are then
        // zeroed below so background keys/mouse still cannot leak in.
        let drive_ticks = shared.drive_ticks.load(Relaxed);
        let focused = src.focused();
        if shared.paused.load(Relaxed) || (!focused && drive_ticks == 0) {
            src.drop_drag();
            // Park the flight ramp with the drag: a gated span presents no
            // frames (or is not ours), so resuming must not dump a coast into
            // it. Closing the menu with W still held re-eases from rest.
            mv = Vec3A::ZERO;
            if prev_speed != 0.0 {
                prev_speed = 0.0;
                shared.speed.store(0.0f32.to_bits(), Relaxed);
            }
            continue;
        }

        // A teleport arrives stopped. Load-then-conditional-store rather than
        // an unconditional swap, so the steady state stays read-only.
        if shared.motion_reset.load(Relaxed) {
            shared.motion_reset.store(false, Relaxed);
            mv = Vec3A::ZERO;
        }
        let ease_s = f32::from_bits(shared.move_ease.load(Relaxed));

        // The OS, sampled once. An unfocused tick (a QA drive, which is exempt
        // from the gate above) never asks it at all — that is what `Raw::none`
        // says, and it is the `focused &&` conjunct the per-key read used to
        // carry, hoisted to the one place that knows.
        let raw = if focused { src.tick() } else { Raw::none() };
        let pad = raw.pad.as_ref();
        // QA drive axes for this tick (decremented per tick, so a requested
        // run is exact in 500 Hz ticks whatever the render rate).
        let mut drv = Vec3A::ZERO;
        if drive_ticks > 0 {
            let g = |i: usize| f32::from_bits(shared.drive[i].load(Relaxed));
            drv = Vec3A::new(g(0), g(1), g(2));
            shared.drive_ticks.store(drive_ticks - 1, Relaxed);
        }
        let drv_any = drv != Vec3A::ZERO;

        let (dx, dy) = raw.look;

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
        slow_ctrl = ramp(slow_ctrl, raw.slow_ctrl || pad.is_some_and(|p| p.lb));
        slow_shift = ramp(slow_shift, raw.slow_shift || pad.is_some_and(|p| p.rb));
        let slow = slow_factor(slow_ctrl, slow_shift);
        // --- flight speed: diag * 0.1875 / s, times the slow factor.
        let speed = diag * 0.1875 * dt * slow;

        // Keyboard direction flags (unit directions, normalized below —
        // exactly the old apply_movement). Arrows alias WASD; which physical
        // keys reach these is `KEYMAP`'s answer and the source's job.
        let (kw, ks, kd, ka) = (raw.fwd, raw.back, raw.right, raw.left);
        let (kup, kdn) = (raw.up, raw.down);
        let key_any = raw.key_any();

        // --- keyboard flight ease. The target is built from the key flags
        // ALONE, in the camera-relative basis (x = right, y = up, z = forward)
        // the analog stick already uses, so the ramp advances without the pose
        // lock and the early-out below stays a pure input test. `speed` is
        // applied later and already carries dt + the eased slow factor, so the
        // two eases compose.
        //
        // ONE eased vector, not a scalar plus a latched direction: that is what
        // makes a direction change SLEW — W->S passes through the origin (a
        // momentary stop, the inertia) instead of flipping at full speed.
        let target = if key_any {
            Vec3A::new(
                (kd as i32 - ka as i32) as f32,
                (kup as i32 - kdn as i32) as f32,
                (kw as i32 - ks as i32) as f32,
            )
            .normalize_or_zero()
        } else {
            Vec3A::ZERO
        };
        mv = crate::camera::move_ease(mv, target, dt, ease_s);

        // --- time-of-day scrub: `.`/D-pad-right forward, `,`/D-pad-left
        // reverse (both held = 0). Wall-clock-exact by the same measured-dt
        // argument as displacement.
        let t_fwd = raw.tod_fwd || pad.is_some_and(|p| p.dpad_r);
        let t_rev = raw.tod_rev || pad.is_some_and(|p| p.dpad_l);
        let tod_dir = (t_fwd as i32 - t_rev as i32) as f32;

        if tod_dir != 0.0 {
            shared.manual_tod.store(true, Relaxed);
        }
        let auto_tod = !shared.manual_tod.load(Relaxed) && !attractors.is_empty();

        let pad_move =
            pad.is_some_and(|p| p.lx != 0.0 || p.ly != 0.0 || p.lt != 0.0 || p.rt != 0.0);
        let pad_look = pad.is_some_and(|p| p.rx != 0.0 || p.ry != 0.0);
        // auto_tod must integrate with NO input held (the ease keeps running
        // after the flight stops) — but once converged it writes nothing (see
        // below), so an idle converged session still compares bit-equal.
        // `mv` is tested AFTER this tick's update, which is what keeps a coast
        // running: skipping a tick would freeze the ramp mid-glide and the
        // camera would resume from it on the next key press. On the tick the
        // ramp snaps to a bitwise ZERO the test re-arms, and since a rested
        // ramp contributes exactly nothing there is no motion to lose — that
        // exact arrival is why the ease may not be an asymptotic smoother.
        if !key_any
            && mv == Vec3A::ZERO
            && !pad_move
            && !pad_look
            && !drv_any
            && dx == 0.0
            && dy == 0.0
            && tod_dir == 0.0
            && !auto_tod
        {
            if prev_speed != 0.0 {
                prev_speed = 0.0;
                shared.speed.store(0.0f32.to_bits(), Relaxed);
            }
            continue; // nothing to integrate; the shared state stays bit-untouched
        }

        let mut st = shared.state.lock().unwrap();
        let cam = &mut st.cam;
        let f = cam.forward();
        let r = f.cross(Vec3A::Y).normalize();
        let mut step = Vec3A::ZERO;
        if ease_s > 0.0 {
            // Eased keyboard flight. The DIRECTION is normalized in WORLD
            // space, exactly as the pre-ease code did — `f` carries a Y
            // component when pitched, so the camera basis is not orthonormal
            // and normalizing the coefficients instead would run a
            // forward+vertical combination fast. So the only thing the ease
            // changes is the magnitude, through a smoothstep of the ramp.
            let s = crate::camera::move_scale(mv);
            if s > 0.0 {
                let dir = r * mv.x + Vec3A::Y * mv.y + f * mv.z;
                if dir != Vec3A::ZERO {
                    step += dir.normalize() * (s * speed);
                }
            }
        } else if key_any {
            // Off arm (`--move-ease 0`): the pre-ease hard step, verbatim.
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
        if let Some(p) = pad {
            // Analog flight: deflection IS the speed control (deliberately
            // not normalized like the keys); full tilt == key speed.
            step += (r * p.lx + f * p.ly) * speed;
            step += Vec3A::Y * ((p.rt - p.lt) * speed);
        }
        if drv_any {
            // The QA drive rides the pad's analog math verbatim.
            step += (r * drv.x + f * drv.z + Vec3A::Y * drv.y) * speed;
        }
        if step != Vec3A::ZERO {
            cam.pos += step;
        }
        // Publish the tick's speed for the audio wind (units/s — dt-invariant
        // by the same measured-dt argument as displacement). A look-only tick
        // parks it at 0.0; the compare keeps steady states write-free.
        let sp = if step != Vec3A::ZERO { step.length() / dt.max(1e-4) } else { 0.0 };
        if sp != prev_speed {
            prev_speed = sp;
            shared.speed.store(sp.to_bits(), Relaxed);
        }

        // Look: mouse per-pixel (tick-rate independent by construction) +
        // right stick as a rate (rad/s x deflection x dt — the term that
        // needed wall-clock integration). Stick right = look right, stick
        // up = look up, matching the mouse-drag feel.
        let mut yaw_d = dx * 0.004;
        let mut pitch_d = -dy * 0.004;
        if let Some(p) = pad {
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
        } else if auto_tod {
            // World auto-TOD: ease toward the islands' blended theme hour at
            // the MANUAL scrub rate (no slow factor — that chord belongs to
            // the user's hand), along the shorter wrap direction. Within one
            // step of the target it SNAPS to the exact target value, so the
            // next tick's delta is exactly 0.0 and the write stops — a still
            // camera therefore converges tod, the session loop stops seeing
            // deltas, and plain accumulation converges like any still frame.
            // Stepping by `st.tod + d` instead of snapping would leave an
            // ulp-scale residue that re-writes (and re-resets accumulation)
            // every tick, forever.
            let target = attractor_hour(attractors, st.cam.pos);
            let mut d = target - st.tod;
            d -= (d / 24.0).round() * 24.0;
            if d != 0.0 {
                let step = TOD_RATE * dt;
                st.tod = if d.abs() <= step {
                    target
                } else {
                    (st.tod + step * d.signum()).rem_euclid(24.0)
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The Windows source — live OS state, read from this thread
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod src_win32 {
    use super::*;
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
        GetAsyncKeyState, MapVirtualKeyW, MAPVK_VSC_TO_VK_EX, VK_LBUTTON, VK_RBUTTON,
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

    /// The ~2 ms tick source. Plain `thread::sleep(2ms)` quantizes to the
    /// ~15.6 ms system tick (~64 Hz effective); the high-resolution waitable
    /// timer (Win10 1803+) delivers ~0.5 ms precision for one blocked wait and
    /// opts out of timer coalescing. Fallback when unavailable: raise the
    /// global timer resolution for this process (timeBeginPeriod) and sleep.
    /// Either way integration uses measured dt, so the tick source never
    /// affects displacement — only granularity.
    pub struct Ticker {
        timer: Option<HANDLE>,
        last: Instant,
    }

    impl Ticker {
        pub fn new() -> Self {
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
                        return Self { timer: Some(t), last: Instant::now() };
                    }
                    let _ = CloseHandle(t);
                }
                eprintln!(
                    "flycam: high-res waitable timer unavailable; timeBeginPeriod(1) fallback"
                );
                timeBeginPeriod(1);
                Self { timer: None, last: Instant::now() }
            }
        }
    }

    impl Tick for Ticker {
        /// Above the rayon workers. The renderer saturates every core at
        /// normal priority for the whole trace, and this thread needs ~10 us
        /// every 2 ms. A starved tick never loses displacement (dt is
        /// measured, so the total is exact regardless) — it coarsens the
        /// SAMPLING at key/stick transitions, which is precisely what the
        /// 500 Hz rate buys. Best-effort: a failure here costs granularity,
        /// not correctness.
        fn on_thread_start(&self) {
            unsafe {
                let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
            }
        }

        fn wait(&mut self) -> Duration {
            match self.timer {
                Some(t) => unsafe {
                    WaitForSingleObject(t, INFINITE);
                },
                None => std::thread::sleep(Duration::from_millis(TICK_MS as u64)),
            }
            let now = Instant::now();
            let dt = now - self.last;
            self.last = now;
            dt
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
    /// in `KEYMAP` via the current layout, so e.g. AZERTY keeps flying on the
    /// same physical keys. Fallback to the US-layout VK if the mapping comes
    /// back empty.
    struct Keys {
        /// Parallel to `ACTIONS`: every VK that fires this action.
        vks: [Vec<u16>; ACTIONS.len()],
    }

    impl Keys {
        fn new() -> Self {
            let mut vks: [Vec<u16>; ACTIONS.len()] = Default::default();
            for (i, action) in ACTIONS.iter().enumerate() {
                let row = KEYMAP.iter().find(|r| r.action == *action).expect("KEYMAP covers ACTIONS");
                for (scan, fallback) in row.win_scan {
                    let v = unsafe { MapVirtualKeyW(*scan, MAPVK_VSC_TO_VK_EX) } as u16;
                    vks[i].push(if v == 0 { *fallback } else { v });
                }
                vks[i].extend_from_slice(row.win_vk);
            }
            Self { vks }
        }
    }

    /// A trigger in XInput's 0..255 range. `super::trigger16` is the SDL
    /// twin; both threshold at the same fraction of full scale.
    fn trigger(t: u8) -> f32 {
        let thr = XINPUT_GAMEPAD_TRIGGER_THRESHOLD.0 as u8; // metadata mistypes it as button flags
        t.saturating_sub(thr) as f32 / (255 - thr) as f32
    }

    // NOTE: src/pad.rs is a SECOND XInput reader (main thread, pause-menu
    // edges). Safe because XInputGetState is a stateless read-only snapshot —
    // and while the menu is open this thread's pause gate skips its own poll
    // entirely, so the two never even overlap in that state.
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
    /// area: not on the title bar / borders (SDL never entered button-down
    /// state from WM_NC* clicks either) and not on another window overlapping
    /// ours. Once latched, the drag keeps tracking off-window until release —
    /// the SDL mouse-capture behavior.
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

    pub struct Win32Source {
        hwnd: HWND,
        keys: Keys,
        look_btn: u16,
        drag: Option<(i32, i32)>,
        pad_backoff: u32,
    }

    impl Win32Source {
        pub fn new(hwnd: isize) -> Win32Source {
            // Windows swaps the mouse buttons at the MESSAGE layer, but
            // GetAsyncKeyState reports the PHYSICAL button — so for a
            // swapped-button (left-handed) user the primary button is
            // VK_RBUTTON. SDL's `mouse_state().left()` was the logical
            // primary, so reading VK_LBUTTON unconditionally would have moved
            // drag-look to their non-primary button. Resolved once, like the
            // key layout.
            let look_btn = unsafe {
                if GetSystemMetrics(SM_SWAPBUTTON) != 0 { VK_RBUTTON.0 } else { VK_LBUTTON.0 }
            };
            Win32Source {
                hwnd: HWND(hwnd as *mut core::ffi::c_void),
                keys: Keys::new(),
                look_btn,
                drag: None,
                pad_backoff: 0,
            }
        }
    }

    fn down(vk: u16) -> bool {
        unsafe { GetAsyncKeyState(vk as i32) as u16 & 0x8000 != 0 }
    }

    impl Source for Win32Source {
        fn focused(&mut self) -> bool {
            let fg = unsafe { GetForegroundWindow() };
            fg == self.hwnd
        }

        fn tick(&mut self) -> Raw {
            let held = |a: Action| {
                ACTIONS
                    .iter()
                    .position(|x| *x == a)
                    .is_some_and(|i| self.keys.vks[i].iter().any(|vk| down(*vk)))
            };
            let pad = poll_pad(&mut self.pad_backoff);

            // --- mouse look: absolute cursor deltas while a left-drag is
            // latched (the same accelerated OS cursor SDL reported, so the
            // 0.004 rad/px feel is unchanged).
            let (mut dx, mut dy) = (0.0f32, 0.0f32);
            let mut pt = POINT::default();
            if unsafe { GetCursorPos(&mut pt) }.is_err() || !down(self.look_btn) {
                self.drag = None;
            } else if let Some((px, py)) = self.drag {
                (dx, dy) = ((pt.x - px) as f32, (pt.y - py) as f32);
                self.drag = Some((pt.x, pt.y));
            } else if drag_may_start(self.hwnd, pt) {
                self.drag = Some((pt.x, pt.y)); // latch; first tick contributes no delta
            }

            Raw {
                fwd: held(Action::Fwd),
                back: held(Action::Back),
                left: held(Action::Left),
                right: held(Action::Right),
                up: held(Action::Up),
                down: held(Action::Down),
                slow_ctrl: held(Action::SlowCtrl),
                slow_shift: held(Action::SlowShift),
                tod_rev: held(Action::TodRev),
                tod_fwd: held(Action::TodFwd),
                look: (dx, dy),
                pad,
            }
        }

        fn drop_drag(&mut self) {
            self.drag = None;
        }
    }

    /// The numbers `KEYMAP` and the pad constants spell out, against the
    /// `windows` crate's own — the lockstep pin for a table that had to be
    /// written as plain integers so every platform could check its shape.
    pub fn pin_constants() -> Result<(), String> {
        use windows::Win32::UI::Input::KeyboardAndMouse::{
            VK_CONTROL, VK_DOWN, VK_LEFT, VK_OEM_COMMA, VK_OEM_PERIOD, VK_RIGHT, VK_SHIFT, VK_UP,
        };
        let pins: [(&str, u16, u16); 8] = [
            ("VK_SHIFT", VK_SHIFT_N, VK_SHIFT.0),
            ("VK_CONTROL", VK_CONTROL_N, VK_CONTROL.0),
            ("VK_LEFT", VK_LEFT_N, VK_LEFT.0),
            ("VK_UP", VK_UP_N, VK_UP.0),
            ("VK_RIGHT", VK_RIGHT_N, VK_RIGHT.0),
            ("VK_DOWN", VK_DOWN_N, VK_DOWN.0),
            ("VK_OEM_COMMA", VK_OEM_COMMA_N, VK_OEM_COMMA.0),
            ("VK_OEM_PERIOD", VK_OEM_PERIOD_N, VK_OEM_PERIOD.0),
        ];
        for (name, ours, theirs) in pins {
            if ours != theirs {
                return Err(format!("flycam: KEYMAP's {name} is {ours:#x}, the SDK says {theirs:#x}"));
            }
        }
        if PAD_DEADZONE_L != XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE.0 as f32 {
            return Err("flycam: PAD_DEADZONE_L disagrees with XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE".into());
        }
        if PAD_DEADZONE_R != XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE.0 as f32 {
            return Err("flycam: PAD_DEADZONE_R disagrees with XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// A scripted `Source`: one closure per tick, plus the focus flag and a count
/// of `drop_drag` calls (which is how the gated-tick contract is observed).
///
/// The closure is handed the `Shared` it is driving, so a scenario can sample
/// what the PREVIOUS tick published — the speed atomic is the only window onto
/// the ramp, which is a loop local by design.
struct ScriptSource<F: FnMut(u32, &Shared) -> Raw> {
    f: F,
    shared: Arc<Shared>,
    n: u32,
    focused: bool,
    drops: u32,
}

impl<F: FnMut(u32, &Shared) -> Raw> Source for ScriptSource<F> {
    fn focused(&mut self) -> bool {
        self.focused
    }
    fn tick(&mut self) -> Raw {
        let r = (self.f)(self.n, &self.shared);
        self.n += 1;
        r
    }
    fn drop_drag(&mut self) {
        self.drops += 1;
    }
}

/// A scripted `Tick`: a FIXED dt and a tick budget, stopping the loop itself.
///
/// It carries the clock as well as the wait — which is the entire reason the
/// two are one trait — so a scenario can be run at 2 ms and again at 20 ms
/// through THE REAL `integrate_loop`, with no wall-clock time passing and no
/// dependence on how the test machine schedules.
struct ScriptTick {
    shared: Arc<Shared>,
    dt: Duration,
    left: u32,
}

impl Tick for ScriptTick {
    fn wait(&mut self) -> Duration {
        if self.left == 0 {
            self.shared.stop.store(true, Relaxed);
        } else {
            self.left -= 1;
        }
        self.dt
    }
}

/// One scenario. Returns the shared state (pose, tod, speed) and how many
/// ticks were gated.
fn run_ticks<F: FnMut(u32, &Shared) -> Raw>(
    dt: Duration,
    ticks: u32,
    ease: f32,
    diag: f32,
    focused: bool,
    paused: bool,
    attractors: &[TodAttractor],
    f: F,
) -> (Arc<Shared>, u32) {
    // Origin, looking down +X: `forward()` is then exactly (1, 0, 0), so a
    // forward step lands exactly on `pos.x` and the displacement assertions
    // below are about the integrator rather than about trigonometry.
    let cam0 = Camera { pos: Vec3A::ZERO, yaw: 0.0, pitch: 0.0, fov_y: 1.0 };
    let shared = Arc::new(Shared::new(cam0, 12.0, ease));
    shared.paused.store(paused, Relaxed);
    let mut src = ScriptSource { f, shared: shared.clone(), n: 0, focused, drops: 0 };
    let mut ticker = ScriptTick { shared: shared.clone(), dt, left: ticks };
    integrate_loop(&shared, &mut src, &mut ticker, diag, attractors);
    (shared, src.drops)
}

/// Forward held, nothing else.
fn raw_fwd() -> Raw {
    Raw { fwd: true, ..Raw::default() }
}

fn speed_of(shared: &Shared) -> f32 {
    f32::from_bits(shared.speed.load(Relaxed))
}

fn rel(a: f32, b: f32) -> f32 {
    ((a - b) / b.abs().max(1e-6)).abs()
}

/// The flycam gate — the integrator's math, pinned. Run by `--check` on every
/// platform (the loop is platform-free; only the sources are not), which is
/// what makes the Windows half of B6b rung 2's split checkable from Linux.
///
/// NONE OF THIS EXISTED before the split, and that is the point: the refactor
/// that made two sources possible is exactly the one that could have changed
/// the math silently, since no gate anywhere reached this file.
pub fn self_test() -> Result<(), String> {
    let diag = 100.0f32;
    // diag * 0.1875 per second, the flight speed.
    let per_s = diag * 0.1875;

    // --- T1: displacement is an exact function of WALL-CLOCK TIME, not of
    // tick count. One second of forward at 500 Hz and at 50 Hz.
    //
    // NOT bit-equality, and the reason is worth stating: 500 additions of a
    // small step and 50 additions of a large one are the same sum in exact
    // arithmetic and different roundings in f32. The claim is the physics, so
    // the bound is a relative one — five orders of magnitude tighter than the
    // bug it is here to catch.
    let (a, _) =
        run_ticks(Duration::from_millis(2), 500, 0.0, diag, true, false, &[], |_, _| raw_fwd());
    let (b, _) =
        run_ticks(Duration::from_millis(20), 50, 0.0, diag, true, false, &[], |_, _| raw_fwd());
    let (da, db) = (a.state.lock().unwrap().cam.pos.x, b.state.lock().unwrap().cam.pos.x);
    if rel(da, per_s) > 1e-5 || rel(db, per_s) > 1e-5 {
        return Err(format!(
            "T1: one second of forward should move {per_s} units; 500x2ms moved {da}, 50x20ms \
             moved {db}"
        ));
    }
    if rel(da, db) > 1e-5 {
        return Err(format!("T1: the two tick rates disagree — {da} vs {db}"));
    }
    // ANTI-VACUITY, and it names the bug: an integrator that used the NOMINAL
    // 2 ms instead of the measured dt would move a tenth as far in the 50 Hz
    // arm. If that value were within the bound above, the bound would be
    // proving nothing.
    if rel(db, per_s * 0.1) < 1e-3 {
        return Err("T1 is vacuous: a fixed-dt integrator would pass it".into());
    }

    // --- T2: the slow-factor rest states are EXACT. See `slow_factor`.
    let pins = [
        ("none", slow_factor(0.0, 0.0), 1.0f32),
        ("ctrl", slow_factor(1.0, 0.0), 0.0625),
        ("shift", slow_factor(0.0, 1.0), 0.125),
        ("both", slow_factor(1.0, 1.0), 0.0078125),
    ];
    for (name, got, want) in pins {
        if got != want {
            return Err(format!("T2: slow factor at rest ({name}) is {got}, not exactly {want}"));
        }
    }
    // The two modifiers compose by multiplication, exactly (2^-4 * 2^-3).
    if slow_factor(1.0, 1.0) != slow_factor(1.0, 0.0) * slow_factor(0.0, 1.0) {
        return Err("T2: the two slow modifiers do not compose exactly".into());
    }
    // ANTI-VACUITY, and it names what T2 actually protects: the EXPONENTS.
    // A retune of the divisor is what moves them, and it must not pass.
    // (Measured, and the reason `slow_factor`'s header says so: this does NOT
    // catch the arithmetic-space spelling, which is exact at all four rest
    // states as well. A gate is worth what it can fail on, not what it looks
    // like it covers.)
    if (-4.1f32 * smooth(1.0)).exp2() == 0.0625 {
        return Err("T2 is vacuous: the exactness test cannot distinguish a wrong exponent".into());
    }

    // --- T3a: RELEASE reaches a bitwise zero ramp. That exact arrival is what
    // re-arms the idle early-out, and therefore what keeps a parked camera
    // bit-stable enough for the replay path to engage. Hold forward for 0.5 s,
    // release for 1.0 s, then require the camera to have stopped dead.
    let ease = 0.4f32;
    let (c, _) = run_ticks(Duration::from_millis(2), 750, ease, diag, true, false, &[], |n, _| {
        if n < 250 { raw_fwd() } else { Raw::default() }
    });
    if speed_of(&c) != 0.0 {
        return Err(format!(
            "T3a: 1.0 s after release the speed is {}, not exactly 0 — the ramp did not arrive \
             at a bitwise zero, so the idle early-out never re-arms and a parked camera keeps \
             invalidating replay",
            speed_of(&c)
        ));
    }
    if c.state.lock().unwrap().cam.pos.x <= 0.0 {
        return Err("T3a: the camera never moved forward at all".into());
    }

    // --- T3b: a REVERSAL slews through a near-stop instead of flipping. ONE
    // eased vector is what buys that; a scalar speed plus a latched direction
    // would pass straight from full forward to full back with no dip, and read
    // as a camera that changes direction without decelerating.
    let mut dip = f32::INFINITY;
    let mut peak = 0.0f32;
    let mut x_at_switch = 0.0f32;
    let (d, _) = run_ticks(Duration::from_millis(2), 1000, ease, diag, true, false, &[], |n, sh| {
        // The PREVIOUS tick's published speed — the only window onto a ramp
        // that is deliberately a loop local. Safe to take the pose lock here:
        // the integrator takes it AFTER sampling the source, never across it.
        let v = speed_of(sh);
        if n < 500 {
            peak = peak.max(v);
        } else {
            if n == 500 {
                x_at_switch = sh.state.lock().unwrap().cam.pos.x;
            } else if n > 505 {
                // Skip a few ticks after the switch: the dip is what is being
                // measured, and the tick the keys change on still carries the
                // forward ramp.
                dip = dip.min(v);
            }
        }
        if n < 500 { raw_fwd() } else { Raw { back: true, ..Raw::default() } }
    });
    if peak <= 0.0 {
        return Err("T3b: the forward phase never reached a speed to reverse from".into());
    }
    if !(dip < peak * 0.05) {
        return Err(format!(
            "T3b: the reversal's slowest tick still ran at {dip} units/s against a full speed of \
             {peak} — the ramp flipped instead of slewing through a stop"
        ));
    }
    if d.state.lock().unwrap().cam.pos.x >= x_at_switch {
        return Err(format!(
            "T3b: holding back left the camera at {} against {x_at_switch} at the reversal",
            d.state.lock().unwrap().cam.pos.x
        ));
    }

    // --- T4: the gates park EVERYTHING. Keys held, but unfocused (and again
    // paused): no displacement, no speed, and the drag dropped every tick.
    for (label, focused, paused) in
        [("unfocused", false, false), ("paused", true, true), ("both", false, true)]
    {
        let (g, drops) =
            run_ticks(Duration::from_millis(2), 400, ease, diag, focused, paused, &[], |_, _| {
                Raw { fwd: true, look: (10.0, 10.0), ..Raw::default() }
            });
        let st = g.state.lock().unwrap();
        if st.cam.pos != Vec3A::ZERO || st.cam.yaw != 0.0 || st.cam.pitch != 0.0 {
            return Err(format!("T4 ({label}): a gated tick moved the camera"));
        }
        if speed_of(&g) != 0.0 {
            return Err(format!("T4 ({label}): a gated tick left the speed at {}", speed_of(&g)));
        }
        if drops != 400 {
            return Err(format!(
                "T4 ({label}): the drag was dropped on {drops} of 400 gated ticks"
            ));
        }
    }

    // --- T5: ONE keymap, two platforms.
    if ACTIONS.len() > 32 {
        return Err("T5: more actions than Mirror's 32-bit key set can carry".into());
    }
    for (i, action) in ACTIONS.iter().enumerate() {
        if KEYMAP[i].action != *action {
            return Err(format!(
                "T5: KEYMAP[{i}] is {:?} where ACTIONS says {action:?} — the Mirror bitset is \
                 indexed by ACTIONS position, so the two orders are one order",
                KEYMAP[i].action
            ));
        }
    }
    for row in &KEYMAP {
        if row.win_scan.is_empty() && row.win_vk.is_empty() {
            return Err(format!("T5: {} has no Windows key", row.name));
        }
        if row.sdl_scan.is_empty() {
            return Err(format!("T5: {} has no SDL key", row.name));
        }
    }
    // No key may mean two things. Each of these fired a real class of edit:
    // adding an alias to one row while forgetting it belongs to another.
    for (i, a) in KEYMAP.iter().enumerate() {
        for b in KEYMAP.iter().skip(i + 1) {
            if let Some(sc) = a.sdl_scan.iter().find(|s| b.sdl_scan.contains(s)) {
                return Err(format!("T5: SDL scancode {sc} is both {} and {}", a.name, b.name));
            }
            if let Some(vk) = a.win_vk.iter().find(|v| b.win_vk.contains(v)) {
                return Err(format!("T5: VK {vk:#x} is both {} and {}", a.name, b.name));
            }
            if let Some((sc, _)) = a.win_scan.iter().find(|(s, _)| {
                b.win_scan.iter().any(|(t, _)| t == s)
            }) {
                return Err(format!("T5: Set-1 scancode {sc:#x} is both {} and {}", a.name, b.name));
            }
        }
    }
    #[cfg(windows)]
    src_win32::pin_constants()?;
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use sdl3::keyboard::Scancode;
        // Both directions. Forward: every number in the table resolves to a
        // scancode the pump maps back to the SAME action. Backward: the pump
        // maps nothing ELSE into an action (a stray arm would be invisible to
        // the forward check).
        for row in &KEYMAP {
            for n in row.sdl_scan {
                let sc = Scancode::from_i32(*n)
                    .ok_or_else(|| format!("T5: {n} is not an SDL scancode ({})", row.name))?;
                match action_for_scancode(sc) {
                    Some(a) if a == row.action => {}
                    other => {
                        return Err(format!(
                            "T5: SDL scancode {n} is {} in KEYMAP and {other:?} in \
                             action_for_scancode",
                            row.name
                        ))
                    }
                }
            }
        }
        for n in 0..512i32 {
            let Some(sc) = Scancode::from_i32(n) else { continue };
            let Some(a) = action_for_scancode(sc) else { continue };
            let row = KEYMAP.iter().find(|r| r.action == a).expect("KEYMAP covers ACTIONS");
            if !row.sdl_scan.contains(&n) {
                return Err(format!(
                    "T5: action_for_scancode maps SDL scancode {n} to {} but KEYMAP does not list \
                     it",
                    row.name
                ));
            }
        }
    }

    // --- T6: the Mirror is a faithful wire. One writer, one reader, and the
    // look accumulator drains EXACTLY ONCE — the property that makes a pump
    // running faster than the integrator lose no motion and double-count none.
    let m = Arc::new(Mirror::new());
    let mut ms = MirrorSource::new(m.clone());
    for (i, action) in ACTIONS.iter().enumerate() {
        m.key(*action, true);
        let raw = ms.tick();
        let held = [
            raw.fwd,
            raw.back,
            raw.left,
            raw.right,
            raw.up,
            raw.down,
            raw.slow_ctrl,
            raw.slow_shift,
            raw.tod_rev,
            raw.tod_fwd,
        ];
        if held.iter().filter(|h| **h).count() != 1 || !held[i] {
            return Err(format!("T6: {action:?} set bit {i} but the source read something else"));
        }
        m.key(*action, false);
    }
    m.set_drag(true);
    m.look(7.0, -3.0);
    m.look(1.0, 1.0);
    let r1 = ms.tick();
    let r2 = ms.tick();
    if r1.look != (8.0, -2.0) {
        return Err(format!("T6: two motion events read back as {:?}, not (8, -2)", r1.look));
    }
    if r2.look != (0.0, 0.0) {
        return Err(format!("T6: the look accumulator did not drain — second read {:?}", r2.look));
    }
    // A drag that ends discards motion it never consumed, so the next press
    // does not begin with a jump.
    m.look(50.0, 50.0);
    m.set_drag(false);
    m.set_drag(true);
    if ms.tick().look != (0.0, 0.0) {
        return Err("T6: motion survived the end of a drag".into());
    }
    // And motion with NO drag live is never recorded at all — otherwise
    // crossing the window and then pressing would snap the view by the whole
    // traverse.
    m.set_drag(false);
    m.look(400.0, 400.0);
    m.set_drag(true);
    if ms.tick().look != (0.0, 0.0) {
        return Err("T6: motion outside a drag reached the integrator".into());
    }
    // SUB-PIXEL MOTION SURVIVES THE WIRE. SDL3 reports `xrel`/`yrel` as f32
    // and Wayland means it (wl_fixed, then divided by the output scale), so
    // fractional deltas are the ordinary case rather than the exotic one —
    // and an accumulator that truncated them counted a slow drag as no drag
    // at all. Four quarter-pixels are one pixel, exactly.
    m.set_drag(false);
    m.set_drag(true);
    for _ in 0..4 {
        m.look(0.25, -0.25);
    }
    let sub = ms.tick().look;
    if sub != (1.0, -1.0) {
        return Err(format!(
            "T6: four 0.25 px motion events read back as {sub:?}, not (1, -1) — the look \
             accumulator is quantizing to whole pixels, which on a fractional-scale output \
             discards every event a slow drag produces"
        ));
    }

    // A pad that leaves mid-tilt must stop the camera, not fly it forever.
    m.pad_present(true);
    m.pad_axis(1, -30000);
    if ms.tick().pad.is_none_or(|p| p.ly <= 0.5) {
        return Err("T6: a tilted stick did not reach the source".into());
    }
    m.pad_present(false);
    if ms.tick().pad.is_some() {
        return Err("T6: a disconnected pad still reports axes".into());
    }

    // FOCUS LOSS PARKS EVERY HELD INPUT. The integrator's focus gate already
    // refuses to act on them, but the mirror must not still be HOLDING them
    // when focus returns — SDL's synthesized key-ups are what normally clears
    // this, and a source that merely trusts them is one alt-tab away from
    // flying the camera with nothing pressed.
    m.set_focused(true);
    m.key(Action::Fwd, true);
    m.set_drag(true);
    m.pad_present(true);
    m.pad_axis(0, 30000);
    m.pad_button(0, true);
    m.set_focused(false);
    let (keys, focused, drag) = m.debug_state();
    if keys != 0 || focused || drag {
        return Err(format!(
            "T6: focus loss left keys {keys:#010x}, focused {focused}, drag {drag} — a held \
             key that outlives the focus it was pressed under flies the camera on the way back"
        ));
    }
    match ms.tick().pad {
        // PAD_PRESENT survives on purpose: nothing re-announces a device that
        // never physically left, so clearing it would lose the pad for the
        // rest of the session.
        None => return Err("T6: focus loss unplugged a pad that is still connected".into()),
        Some(p) if p.lx != 0.0 || p.lb => {
            return Err("T6: focus loss left a stick deflected or a shoulder held".into())
        }
        Some(_) => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn flycam_self_test() {
        super::self_test().expect("flycam self_test");
    }
}
