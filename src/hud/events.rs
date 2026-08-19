//! SDL → Slint event translation for the pause menu. Only runs while the
//! menu is OPEN (input.rs's forwarding mode): pointer motion/buttons/wheel,
//! printable text via SDL's TextInput (never synthesized from KeyDown — that
//! would double every character), and a small non-printable key table.
//! ESC is deliberately NOT forwarded — the session's menu state machine owns
//! it (close / back), as does F11 and the window events.
//!
//! `translate` is the pure half (one SDL event → zero, one or two Slint
//! `WindowEvent`s, handed to a sink) and `forward` is the one-line dispatch
//! over it. The split exists so `self_test` can pin the table — including
//! its NEGATIVE, the letter-KeyDown that must translate to nothing — with no
//! Slint window in the process (B6b rung 4).

use sdl3::event::Event;
use slint::platform::{Key, PointerEventButton, WindowEvent};
use slint::LogicalPosition;

fn key_text(k: Key) -> slint::SharedString {
    let c: char = k.into();
    let mut buf = [0u8; 4];
    slint::SharedString::from(&*c.encode_utf8(&mut buf))
}

fn special(k: sdl3::keyboard::Keycode) -> Option<Key> {
    use sdl3::keyboard::Keycode as K;
    Some(match k {
        K::Return | K::KpEnter => Key::Return,
        K::Backspace => Key::Backspace,
        K::Delete => Key::Delete,
        K::Tab => Key::Tab,
        K::Left => Key::LeftArrow,
        K::Right => Key::RightArrow,
        K::Up => Key::UpArrow,
        K::Down => Key::DownArrow,
        K::Home => Key::Home,
        K::End => Key::End,
        K::PageUp => Key::PageUp,
        K::PageDown => Key::PageDown,
        _ => return None,
    })
}

fn button(b: sdl3::mouse::MouseButton) -> PointerEventButton {
    use sdl3::mouse::MouseButton as M;
    match b {
        M::Left => PointerEventButton::Left,
        M::Right => PointerEventButton::Right,
        M::Middle => PointerEventButton::Middle,
        _ => PointerEventButton::Other,
    }
}

/// Translate one SDL event into the Slint `WindowEvent`s it means, handing
/// each to `sink` in dispatch order. Unhandled event kinds produce nothing
/// (the menu doesn't want them). Returns whether anything was produced.
pub fn translate(ev: &Event, sink: &mut dyn FnMut(WindowEvent)) -> bool {
    match ev {
        // SDL3 mouse coordinates are f32 already (subpixel precision).
        Event::MouseMotion { x, y, .. } => {
            sink(WindowEvent::PointerMoved { position: LogicalPosition::new(*x, *y) });
            true
        }
        Event::MouseButtonDown { mouse_btn, x, y, .. } => {
            sink(WindowEvent::PointerPressed {
                position: LogicalPosition::new(*x, *y),
                button: button(*mouse_btn),
            });
            true
        }
        Event::MouseButtonUp { mouse_btn, x, y, .. } => {
            sink(WindowEvent::PointerReleased {
                position: LogicalPosition::new(*x, *y),
                button: button(*mouse_btn),
            });
            true
        }
        // SDL3 folded SDL2's precise_x/y into the (now float) x/y deltas.
        Event::MouseWheel { x, y, mouse_x, mouse_y, .. } => {
            sink(WindowEvent::PointerScrolled {
                position: LogicalPosition::new(*mouse_x, *mouse_y),
                delta_x: x * 40.0,
                delta_y: y * 40.0,
            });
            true
        }
        // Printable characters arrive HERE (SDL text input is started while
        // the menu is open); KeyDown only carries the non-printable keys.
        Event::TextInput { text, .. } => {
            let t = slint::SharedString::from(text.as_str());
            sink(WindowEvent::KeyPressed { text: t.clone() });
            sink(WindowEvent::KeyReleased { text: t });
            true
        }
        Event::KeyDown { keycode: Some(k), .. } => {
            if let Some(key) = special(*k) {
                sink(WindowEvent::KeyPressed { text: key_text(key) });
                true
            } else {
                false
            }
        }
        Event::KeyUp { keycode: Some(k), .. } => {
            if let Some(key) = special(*k) {
                sink(WindowEvent::KeyReleased { text: key_text(key) });
                true
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Translate + dispatch one SDL event into the Slint window.
pub fn forward(win: &slint::Window, ev: &Event) {
    let _ = translate(ev, &mut |we| win.dispatch_event(we));
}

/// The translation table, pinned without a Slint window: pointer motion,
/// text input's press+release pair with the SAME text (the "é" round trip —
/// non-ASCII survives `SharedString`), the special-key table, and the
/// NEGATIVE the module header rests on: a letter KeyDown translates to
/// nothing, because letters arrive as TextInput and synthesising them from
/// KeyDown would double every character.
pub fn self_test() -> Result<(), String> {
    use sdl3::keyboard::{Keycode, Mod};
    let key = |k: Keycode| Event::KeyDown {
        timestamp: 0,
        window_id: 0,
        keycode: Some(k),
        scancode: None,
        keymod: Mod::empty(),
        repeat: false,
        which: 0,
        raw: 0,
    };
    let run = |ev: &Event| -> Vec<WindowEvent> {
        let mut out = Vec::new();
        translate(ev, &mut |we| out.push(we));
        out
    };

    let motion = Event::MouseMotion {
        timestamp: 0,
        window_id: 0,
        which: 0,
        mousestate: sdl3::mouse::MouseState::from_sdl_state(0),
        x: 3.5,
        y: 7.25,
        xrel: 0.0,
        yrel: 0.0,
    };
    match run(&motion).as_slice() {
        [WindowEvent::PointerMoved { position }] if position.x == 3.5 && position.y == 7.25 => {}
        other => return Err(format!("MouseMotion → {other:?}, want one PointerMoved at (3.5,7.25)")),
    }

    let text = Event::TextInput { timestamp: 0, window_id: 0, text: "é".to_string() };
    match run(&text).as_slice() {
        [WindowEvent::KeyPressed { text: a }, WindowEvent::KeyReleased { text: b }]
            if a.as_str() == "é" && b.as_str() == "é" => {}
        other => return Err(format!("TextInput \"é\" → {other:?}, want pressed+released \"é\"")),
    }

    match run(&key(Keycode::Return)).as_slice() {
        [WindowEvent::KeyPressed { text }] if text.as_str() == key_text(Key::Return).as_str() => {}
        other => return Err(format!("KeyDown Return → {other:?}, want KeyPressed(Return)")),
    }
    match run(&key(Keycode::Left)).as_slice() {
        [WindowEvent::KeyPressed { text }] if text.as_str() == key_text(Key::LeftArrow).as_str() => {}
        other => return Err(format!("KeyDown Left → {other:?}, want KeyPressed(LeftArrow)")),
    }

    // The negative: a letter KeyDown is NOT a key event for Slint.
    let got = run(&key(Keycode::A));
    if !got.is_empty() {
        return Err(format!("KeyDown A → {got:?}, must translate to NOTHING (letters are TextInput)"));
    }
    // And ESC is not in the table either — the session owns it.
    let got = run(&key(Keycode::Escape));
    if !got.is_empty() {
        return Err(format!("KeyDown Escape → {got:?}, must translate to nothing"));
    }
    // An event kind the menu does not want produces nothing and says so.
    let mut n = 0;
    if translate(&Event::Quit { timestamp: 0 }, &mut |_| n += 1) || n != 0 {
        return Err("Quit must translate to nothing".into());
    }
    Ok(())
}
