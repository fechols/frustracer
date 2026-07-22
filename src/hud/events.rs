//! SDL → Slint event translation for the pause menu. Only runs while the
//! menu is OPEN (input.rs's forwarding mode): pointer motion/buttons/wheel,
//! printable text via SDL's TextInput (never synthesized from KeyDown — that
//! would double every character), and a small non-printable key table.
//! ESC is deliberately NOT forwarded — the session's menu state machine owns
//! it (close / back), as does F11 and the window events.

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

/// Translate + dispatch one SDL event into the Slint window. Unhandled event
/// kinds are simply dropped (the menu doesn't want them).
pub fn forward(win: &slint::Window, ev: &Event) {
    let dispatched = match ev {
        // SDL3 mouse coordinates are f32 already (subpixel precision).
        Event::MouseMotion { x, y, .. } => {
            win.dispatch_event(WindowEvent::PointerMoved {
                position: LogicalPosition::new(*x, *y),
            });
            true
        }
        Event::MouseButtonDown { mouse_btn, x, y, .. } => {
            win.dispatch_event(WindowEvent::PointerPressed {
                position: LogicalPosition::new(*x, *y),
                button: button(*mouse_btn),
            });
            true
        }
        Event::MouseButtonUp { mouse_btn, x, y, .. } => {
            win.dispatch_event(WindowEvent::PointerReleased {
                position: LogicalPosition::new(*x, *y),
                button: button(*mouse_btn),
            });
            true
        }
        // SDL3 folded SDL2's precise_x/y into the (now float) x/y deltas.
        Event::MouseWheel { x, y, mouse_x, mouse_y, .. } => {
            win.dispatch_event(WindowEvent::PointerScrolled {
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
            win.dispatch_event(WindowEvent::KeyPressed { text: t.clone() });
            win.dispatch_event(WindowEvent::KeyReleased { text: t });
            true
        }
        Event::KeyDown { keycode: Some(k), .. } => {
            if let Some(key) = special(*k) {
                win.dispatch_event(WindowEvent::KeyPressed { text: key_text(key) });
                true
            } else {
                false
            }
        }
        Event::KeyUp { keycode: Some(k), .. } => {
            if let Some(key) = special(*k) {
                win.dispatch_event(WindowEvent::KeyReleased { text: key_text(key) });
                true
            } else {
                false
            }
        }
        _ => false,
    };
    let _ = dispatched;
}
