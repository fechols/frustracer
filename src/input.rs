//! SDL2 input: replaces minifb's polling. Held keys (movement) come from the
//! keyboard state each frame; toggles come from KeyDown edges (repeat
//! filtered); look is a left-button mouse drag, same feel as before.

use crate::camera::Camera;
use glam::Vec3A;
use sdl2::event::Event;
use sdl2::keyboard::{Keycode, Scancode};
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
    pub toggle_bounce: bool,   // H (hemisphere frustum bounces)
    pub verify: bool,          // C
    pub screenshot: bool,      // P
    pub quality: Option<u32>,  // 1/2/3
}

pub struct Input {
    pump: EventPump,
    prev_mouse: Option<(f32, f32)>,
}

impl Input {
    pub fn new(sdl: &sdl2::Sdl) -> Result<Self, String> {
        Ok(Self { pump: sdl.event_pump()?, prev_mouse: None })
    }

    /// Drain the event queue, collecting edges. Call once per frame before
    /// `apply_movement`.
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
                    Keycode::H => e.toggle_bounce = true,
                    Keycode::C => e.verify = true,
                    Keycode::P => e.screenshot = true,
                    Keycode::Num1 | Keycode::Kp1 => e.quality = Some(1),
                    Keycode::Num2 | Keycode::Kp2 => e.quality = Some(2),
                    Keycode::Num3 | Keycode::Kp3 => e.quality = Some(3),
                    _ => {}
                },
                _ => {}
            }
        }
        e
    }

    /// WASD/EQ/Space movement + left-drag look. Returns true if the camera
    /// changed (the accumulation-reset signal).
    pub fn apply_movement(&mut self, cam: &mut Camera, dt: f32, diag: f32) -> bool {
        let mut moved = false;
        let ks = self.pump.keyboard_state();
        let down = |sc: Scancode| ks.is_scancode_pressed(sc);

        let mut speed = diag * 0.25 * dt;
        if down(Scancode::LShift) || down(Scancode::RShift) {
            speed *= 4.0;
        }
        let f = cam.forward();
        let r = f.cross(Vec3A::Y).normalize();
        let mut delta = Vec3A::ZERO;
        if down(Scancode::W) {
            delta += f;
        }
        if down(Scancode::S) {
            delta -= f;
        }
        if down(Scancode::D) {
            delta += r;
        }
        if down(Scancode::A) {
            delta -= r;
        }
        if down(Scancode::E) || down(Scancode::Space) {
            delta += Vec3A::Y;
        }
        if down(Scancode::Q) {
            delta -= Vec3A::Y;
        }
        if delta != Vec3A::ZERO {
            cam.pos += delta.normalize() * speed;
            moved = true;
        }

        let ms = self.pump.mouse_state();
        let mpos = (ms.x() as f32, ms.y() as f32);
        if ms.left() {
            if let Some((px, py)) = self.prev_mouse {
                let (dx, dy) = (mpos.0 - px, mpos.1 - py);
                if dx != 0.0 || dy != 0.0 {
                    cam.yaw += dx * 0.004;
                    cam.pitch = (cam.pitch - dy * 0.004).clamp(-1.5, 1.5);
                    moved = true;
                }
            }
        }
        self.prev_mouse = Some(mpos);

        moved
    }
}
