//! The CPU half of the HUD/pause-menu overlay: Slint's SOFTWARE renderer
//! drawing into a persistent premultiplied-RGBA8 buffer behind a custom
//! `slint::platform::Platform` (no winit, no window of its own — the SDL
//! window and D3D12 swapchain stay exactly what they are). The GPU half
//! (`gpu/hud.rs`) uploads the DIRTY RECTANGLES this module reports and
//! composites them over the tonemapped frame in every present arm.
//!
//! Slint is used under its Royalty-Free license (the project is not GPL;
//! see Cargo.toml's dependency comment).
//!
//! Dirty-rect discipline, CPU side: `RepaintBufferType::ReusedBuffer` makes
//! the renderer preserve the buffer across frames and re-rasterize ONLY the
//! dirty region, whose rectangles come back from `render()`; a frame where
//! nothing changed returns from `draw_if_needed` without rendering at all —
//! zero raster, zero bytes packed, zero upload. `Hud::frame` additionally
//! quantizes its inputs (whole degrees, whole minutes) so property churn
//! only happens on visibly-different values.
//!
//! Lifetime: one `Hud` per PROCESS, owned by `run_window` beside `fly` — it
//! survives session re-entries (resize/F11), so menu/HUD state needs no
//! `Persist` mirror; `set_size` follows the window.

pub mod events;
mod ui;

use crate::camera::Camera;
use slint::platform::software_renderer::{
    MinimalSoftwareWindow, PremultipliedRgbaColor, RepaintBufferType,
};
use slint::platform::{Platform, WindowAdapter};
use slint::ComponentHandle;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

/// How long the keymap panel lingers after the camera stops moving before
/// fading out (it fades IN on motion — help appears exactly when the user is
/// flying; a short linger keeps brief pauses from strobing it).
const HELP_LINGER: std::time::Duration = std::time::Duration::from_millis(2500);

/// One changed region of the HUD buffer, in pixels.
#[derive(Clone, Copy, Debug)]
pub struct DirtyRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// A frame's changed pixels: each rect's rows tightly packed (`w*4` bytes per
/// row), concatenated in rect order. `gpu/hud.rs` consumes this layout.
pub struct HudFrame {
    pub rects: Vec<DirtyRect>,
    pub bytes: Vec<u8>,
}

/// Menu events for the session loop, queued by the Slint callbacks during
/// event dispatch (inside `input::Input::poll`'s forwarding mode) and drained
/// once per frame by `take_actions`.
pub enum HudAction {
    Resume,
    Quit,
    OpenSettings,
    Back,
    Group(String),
    /// (row id, ±1)
    Adjust(String, i32),
    /// (row id, committed text)
    TextEdit(String, String),
}

/// One settings row, built by main.rs from `settings::menu_items()` +
/// `menu_value` and handed to `set_rows`; `control` mirrors
/// `settings::Control` as the markup's string tag.
pub struct MenuRow {
    pub id: String,
    pub label: String,
    pub value: String,
    pub restart: bool,
    pub control: &'static str,
}

#[derive(Clone, Copy, PartialEq)]
enum MenuPage {
    Main,
    Settings,
}

struct FrustPlatform {
    start: Instant,
}

thread_local! {
    /// Handed from `create_window_adapter` (which Slint calls during
    /// component instantiation) back to `Hud::new`. Main-thread only, like
    /// every Slint object.
    static SLINT_WINDOW: RefCell<Option<Rc<MinimalSoftwareWindow>>> = const { RefCell::new(None) };
}

impl Platform for FrustPlatform {
    fn create_window_adapter(&self) -> Result<Rc<dyn WindowAdapter>, slint::PlatformError> {
        // ReusedBuffer is the dirty-rect contract: buffer contents persist
        // across renders and `render()` returns only what changed.
        let w = MinimalSoftwareWindow::new(RepaintBufferType::ReusedBuffer);
        SLINT_WINDOW.with(|s| *s.borrow_mut() = Some(w.clone()));
        Ok(w)
    }

    fn duration_since_start(&self) -> core::time::Duration {
        self.start.elapsed()
    }
}

pub struct Hud {
    ui: ui::HudUi,
    window: Rc<MinimalSoftwareWindow>,
    buf: Vec<PremultipliedRgbaColor>,
    w: u32,
    h: u32,
    /// Compass+clock visibility (F1 / the menu's Display toggle).
    visible: bool,
    /// Force the next frame to report the whole window dirty (first frame,
    /// resize, or a present-arm error dropped staged rects).
    force_full: bool,
    /// Quantized last-set property values — churn only on visible change.
    last_heading: i32,
    last_minute: i32,
    last_help: bool,
    last_hud_on: bool,
    /// Last time the camera was moving (drives the keymap panel's fade).
    last_move: Option<Instant>,
    /// Pause-menu state (Rust owns it; the ui properties mirror it).
    menu_open: bool,
    page: MenuPage,
    group: String,
    actions: Rc<RefCell<Vec<HudAction>>>,
}

const CLEAR: PremultipliedRgbaColor =
    PremultipliedRgbaColor { red: 0, green: 0, blue: 0, alpha: 0 };

impl Hud {
    pub fn new(w: u32, h: u32, visible: bool) -> Result<Self, String> {
        // set_platform is once-per-process; Hud is constructed once, in
        // run_window. A second construction is a programming error we'd
        // rather hear about than paper over.
        slint::platform::set_platform(Box::new(FrustPlatform { start: Instant::now() }))
            .map_err(|e| format!("slint set_platform: {e:?}"))?;
        let ui = ui::HudUi::new().map_err(|e| format!("slint HudUi: {e}"))?;
        let window = SLINT_WINDOW
            .with(|s| s.borrow_mut().take())
            .ok_or_else(|| "slint window adapter was not created".to_string())?;
        window.set_size(slint::PhysicalSize::new(w, h));
        ui.show().map_err(|e| format!("slint show: {e}"))?;
        // Menu plumbing: the callbacks queue typed actions; the session loop
        // drains them right after the event poll that fired them.
        let actions: Rc<RefCell<Vec<HudAction>>> = Rc::new(RefCell::new(Vec::new()));
        {
            let a = actions.clone();
            ui.on_menu_action(move |act| {
                let act = act.as_str();
                let parsed = match act {
                    "resume" => Some(HudAction::Resume),
                    "exit" => Some(HudAction::Quit),
                    "settings" => Some(HudAction::OpenSettings),
                    "back" => Some(HudAction::Back),
                    _ => act.strip_prefix("group:").map(|g| HudAction::Group(g.to_string())),
                };
                if let Some(p) = parsed {
                    a.borrow_mut().push(p);
                }
            });
        }
        {
            let a = actions.clone();
            ui.on_row_adjust(move |id, dir| {
                a.borrow_mut().push(HudAction::Adjust(id.to_string(), dir));
            });
        }
        {
            let a = actions.clone();
            ui.on_text_edited(move |id, text| {
                a.borrow_mut().push(HudAction::TextEdit(id.to_string(), text.to_string()));
            });
        }
        ui.set_groups(slint::ModelRc::new(slint::VecModel::from(
            crate::settings::GROUPS.iter().map(|g| slint::SharedString::from(*g)).collect::<Vec<_>>(),
        )));
        Ok(Self {
            ui,
            window,
            buf: vec![CLEAR; (w * h) as usize],
            w,
            h,
            visible,
            force_full: true,
            last_heading: i32::MIN,
            last_minute: i32::MIN,
            last_help: false,
            last_hud_on: !visible, // != visible so the first frame sets it
            last_move: None,
            menu_open: false,
            page: MenuPage::Main,
            group: "Display".to_string(),
            actions,
        })
    }

    // ── Pause-menu state machine (the session loop drives it via ESC + the
    // drained actions; open/close pairs with FlyCam pause/resume there).

    pub fn menu_open(&self) -> bool {
        self.menu_open
    }

    pub fn open_menu(&mut self) {
        self.menu_open = true;
        self.page = MenuPage::Main;
        self.ui.set_menu_open(true);
        self.ui.set_menu_page("main".into());
        // FRUSTRACER_MENU_PROBE=1 (dev): drive the action queue as if the
        // user clicked Settings and stepped a restart-tier row — the
        // synthetic-input E2E (SDL rewrites posted click coordinates to the
        // real cursor, so a harness cannot exercise the TouchArea hit-test;
        // hover proves that path, this proves everything downstream of it).
        if std::env::var_os("FRUSTRACER_MENU_PROBE").is_some() {
            let mut a = self.actions.borrow_mut();
            a.push(HudAction::OpenSettings);
            a.push(HudAction::Adjust("vsync".into(), 1));
        }
    }

    pub fn close_menu(&mut self) {
        self.menu_open = false;
        self.ui.set_menu_open(false);
    }

    /// ESC while the menu is open: settings page backs out to the main page
    /// (returns true = consumed); on the main page it is unconsumed and the
    /// session closes the menu.
    pub fn escape(&mut self) -> bool {
        if self.page == MenuPage::Settings {
            self.page = MenuPage::Main;
            self.ui.set_menu_page("main".into());
            return true;
        }
        false
    }

    pub fn open_settings_page(&mut self) {
        self.page = MenuPage::Settings;
        self.ui.set_menu_page("settings".into());
    }

    pub fn back_to_main(&mut self) {
        self.page = MenuPage::Main;
        self.ui.set_menu_page("main".into());
    }

    pub fn group(&self) -> &str {
        &self.group
    }

    pub fn set_group(&mut self, g: &str) {
        self.group = g.to_string();
        self.ui.set_menu_group(g.into());
    }

    pub fn set_rows(&mut self, rows: Vec<MenuRow>) {
        let converted: Vec<ui::SettingRow> = rows
            .into_iter()
            .map(|r| ui::SettingRow {
                id: r.id.as_str().into(),
                label: r.label.as_str().into(),
                value: r.value.as_str().into(),
                restart: r.restart,
                control: r.control.into(),
            })
            .collect();
        self.ui.set_rows(slint::ModelRc::new(slint::VecModel::from(converted)));
    }

    pub fn take_actions(&mut self) -> Vec<HudAction> {
        std::mem::take(&mut *self.actions.borrow_mut())
    }

    /// The Slint window, for input.rs's event forwarding.
    pub fn slint_window(&self) -> &slint::Window {
        use slint::platform::WindowAdapter;
        self.window.window()
    }

    pub fn visible(&self) -> bool {
        self.visible
    }

    /// F1 / menu toggle. The next `frame` repaints the affected region
    /// (Slint dirties the compass area when `hud-on` flips).
    pub fn set_visible(&mut self, on: bool) {
        self.visible = on;
    }

    /// Window resized (session re-entry): new buffer, full repaint.
    pub fn set_size(&mut self, w: u32, h: u32) {
        self.w = w;
        self.h = h;
        self.buf = vec![CLEAR; (w * h) as usize];
        self.window.set_size(slint::PhysicalSize::new(w, h));
        self.force_full = true;
    }

    /// A present arm failed after staging — the staged rects may be gone, so
    /// re-upload everything next frame.
    pub fn request_full_redraw(&mut self) {
        self.force_full = true;
    }

    /// Once per frame: feed the pose/clock/motion state, tick Slint's
    /// timers/animations, render if anything is dirty, and return the changed
    /// rects + their pixels (None = nothing changed, upload nothing).
    pub fn frame(&mut self, cam: &Camera, tod: f32, moving: bool) -> Option<HudFrame> {
        // Compass heading: the camera's forward projected to the ground
        // plane; north = +Z, east = +X (the sun-arc azimuth convention),
        // quantized to whole degrees so a still camera dirties nothing.
        let f = cam.forward();
        let heading = f.x.atan2(f.z).to_degrees().rem_euclid(360.0).round() as i32 % 360;
        if heading != self.last_heading {
            self.last_heading = heading;
            self.ui.set_heading(heading as f32);
        }
        // Clock: whole minutes.
        let minute = ((tod.rem_euclid(24.0) * 60.0) as i32).clamp(0, 24 * 60 - 1);
        if minute != self.last_minute {
            self.last_minute = minute;
            self.ui.set_clock(format!("{:02}:{:02}", minute / 60, minute % 60).into());
        }
        // Keymap panel: on while moving, lingering HELP_LINGER after.
        if moving {
            self.last_move = Some(Instant::now());
        }
        let help = self.visible
            && self.last_move.is_some_and(|t| t.elapsed() < HELP_LINGER);
        if help != self.last_help {
            self.last_help = help;
            self.ui.set_help_on(help);
        }
        if self.visible != self.last_hud_on {
            self.last_hud_on = self.visible;
            self.ui.set_hud_on(self.visible);
        }

        slint::platform::update_timers_and_animations();

        let mut rects: Vec<DirtyRect> = Vec::new();
        let (buf, w) = (&mut self.buf, self.w);
        self.window.draw_if_needed(|renderer| {
            let region = renderer.render(buf, w as usize);
            for (pos, size) in region.iter() {
                if size.width > 0 && size.height > 0 {
                    rects.push(DirtyRect {
                        x: pos.x.max(0) as u32,
                        y: pos.y.max(0) as u32,
                        w: size.width,
                        h: size.height,
                    });
                }
            }
        });
        if self.force_full {
            // First frame / resize / post-error: the GPU texture is stale or
            // undefined regardless of what Slint thinks changed.
            self.force_full = false;
            rects = vec![DirtyRect { x: 0, y: 0, w: self.w, h: self.h }];
        }
        // Clamp to the buffer FIRST so the packed bytes and the rect list can
        // never disagree about a row's length (Slint shouldn't produce
        // out-of-bounds rects, but the packing layout is a cross-module
        // contract with gpu/hud.rs — make it true by construction).
        for r in &mut rects {
            r.x = r.x.min(self.w);
            r.y = r.y.min(self.h);
            r.w = r.w.min(self.w - r.x);
            r.h = r.h.min(self.h - r.y);
        }
        rects.retain(|r| r.w > 0 && r.h > 0);
        if rects.is_empty() {
            return None;
        }

        // Pack each rect's rows tightly, in rect order (gpu/hud.rs's layout).
        let bytes_len: usize = rects.iter().map(|r| (r.w * r.h * 4) as usize).sum();
        // FRUSTRACER_HUD_STATS=1: one line per NON-EMPTY upload — the
        // dirty-rect acceptance probe. A still HUD must print nothing at all
        // between clock-minute ticks; a ticking digit is a few KB, never a
        // window-sized copy.
        static STATS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *STATS.get_or_init(|| std::env::var_os("FRUSTRACER_HUD_STATS").is_some()) {
            let list: Vec<String> =
                rects.iter().map(|r| format!("{}x{}+{}+{}", r.w, r.h, r.x, r.y)).collect();
            eprintln!("hud: {} dirty rect(s), {} bytes [{}]", rects.len(), bytes_len, list.join(" "));
            // FRUSTRACER_HUD_STATS also dumps the CPU buffer (straight alpha)
            // each dirty frame — the ground truth for "what did Slint render".
            let px: Vec<u8> = self
                .buf
                .iter()
                .flat_map(|p| {
                    // Un-premultiply for a viewable PNG.
                    let a = p.alpha as u32;
                    let un = |c: u8| if a == 0 { 0 } else { ((c as u32 * 255) / a).min(255) as u8 };
                    [un(p.red), un(p.green), un(p.blue), p.alpha]
                })
                .collect();
            let _ = image::save_buffer(
                "hud-buffer-dump.png",
                &px,
                self.w,
                self.h,
                image::ColorType::Rgba8,
            );
        }
        let mut bytes = Vec::with_capacity(bytes_len);
        let src: &[u8] = unsafe {
            std::slice::from_raw_parts(self.buf.as_ptr() as *const u8, self.buf.len() * 4)
        };
        for r in &rects {
            for row in r.y..r.y + r.h {
                let o = (row as usize * self.w as usize + r.x as usize) * 4;
                bytes.extend_from_slice(&src[o..o + r.w as usize * 4]);
            }
        }
        Some(HudFrame { rects, bytes })
    }
}
