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
//! quantizes its inputs (whole degrees, whole minutes, the SPACE/F-only
//! render-mode string, 125 ms frame-time buckets) so property churn only
//! happens on visibly-different values — and the FPS graph's Slint writes
//! gate on the hud-live fade besides, so a FADED graph freezes (the ring
//! keeps sampling Rust-side) and a settled HUD still uploads nothing.
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

/// The compass + clock's own linger: they wake on camera OR time-of-day
/// activity and fade when both go idle. Longer than HELP_LINGER — heading
/// and hour are exactly what you glance at just AFTER stopping.
const HUD_LINGER: std::time::Duration = std::time::Duration::from_millis(4000);

/// FPS graph: `GRAPH_BARS` buckets of `GRAPH_TICK` each — a 5 s window. Bars
/// carry bucket-average (rendered FPS, FG-added FPS) on the markup's fixed
/// 0..120 scale, so the 60-fps reference line sits statically at mid-strip
/// and a spike clamps instead of rescaling the whole history (auto-range
/// would churn a static element). The FG half is the frame-generation
/// surplus — presented minus rendered, from the family's own multiplier —
/// stacked as a violet segment on the base bar.
const GRAPH_BARS: usize = 40;
const GRAPH_TICK: std::time::Duration = std::time::Duration::from_millis(125);

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
    last_hud_live: bool,
    /// Render-mode label last pushed ("" so the first frame sets it).
    last_mode: &'static str,
    /// FPS graph: ring of bucket-average (rendered FPS, FG-added FPS)
    /// (oldest first), the open bucket's accumulators, and the persistent
    /// VecModel whose rows mirror `hist` while the HUD is live
    /// (`push_graph_rows`). The ring samples even while faded — only the
    /// Slint writes gate on hud_live — so a woken graph shows live recent
    /// history, not a stale pre-fade picture. `acc_mult` sums the per-frame
    /// presented-per-rendered multiplier (1.0 without FG), so a bucket's
    /// presented FPS is `1000·Σmult/Σms` even when FG flips mid-bucket.
    hist: [(f32, f32); GRAPH_BARS],
    acc_sum: f32,
    acc_n: u32,
    acc_mult: f32,
    last_bucket: Option<Instant>,
    graph: Rc<slint::VecModel<ui::FpsBar>>,
    last_fps_txt: String,
    last_ms_txt: String,
    /// Last time the camera was moving (drives the keymap panel's fade).
    last_move: Option<Instant>,
    /// Last camera OR time-of-day activity (drives the compass/clock fade).
    /// Seeded at construction so the HUD shows itself once at boot.
    last_active: Option<Instant>,
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
        // The graph model is created ONCE and updated via set_row_data —
        // re-setting a fresh ModelRc per tick would tear down and rebuild
        // all GRAPH_BARS for-items instead of dirtying their properties.
        let graph = Rc::new(slint::VecModel::from(vec![
            ui::FpsBar { base: 0.0, fg: 0.0 };
            GRAPH_BARS
        ]));
        ui.set_fps_bars(slint::ModelRc::from(graph.clone()));
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
            last_hud_live: true,   // the markup default; first frame reconciles
            last_mode: "",
            hist: [(0.0, 0.0); GRAPH_BARS],
            acc_sum: 0.0,
            acc_n: 0,
            acc_mult: 0.0,
            last_bucket: None,
            graph,
            last_fps_txt: String::new(),
            last_ms_txt: String::new(),
            last_move: None,
            last_active: Some(Instant::now()), // show once at boot, then fade
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

    /// Mirror the FPS ring + readouts into the Slint properties, per row and
    /// guarded by value, so an unchanged row/string dirties nothing (the
    /// wake resync would otherwise repaint a frozen graph for free).
    fn push_graph_rows(&mut self) {
        use slint::Model;
        for (i, &(base, fg)) in self.hist.iter().enumerate() {
            let row = ui::FpsBar { base, fg };
            if self.graph.row_data(i) != Some(row.clone()) {
                self.graph.set_row_data(i, row);
            }
        }
        // Readouts: FPS is what the monitor receives (base + FG surplus —
        // equal to base without FG); ms stays the RENDER frame time, so the
        // pair deliberately doesn't invert under FG.
        let (base, fg) = self.hist[GRAPH_BARS - 1];
        let fps = base + fg;
        let (fps_txt, ms_txt) = if base > 0.0 {
            (format!("{:.0}", fps.min(999.0)), format!("{:.1} ms", 1000.0 / base))
        } else {
            ("--".to_string(), String::new())
        };
        if fps_txt != self.last_fps_txt {
            self.ui.set_fps_now(fps_txt.as_str().into());
            self.last_fps_txt = fps_txt;
        }
        if ms_txt != self.last_ms_txt {
            self.ui.set_ms_now(ms_txt.as_str().into());
            self.last_ms_txt = ms_txt;
        }
    }

    /// Once per frame: feed the pose/clock/motion state, tick Slint's
    /// timers/animations, render if anything is dirty, and return the changed
    /// rects + their pixels (None = nothing changed, upload nothing).
    /// `tod_moved` = the clock changed this frame (scrub / attractors / menu)
    /// — it wakes the compass+clock fade like camera motion does. `mode` is
    /// the render-mode label ("CPU" | "GPU" | "DXR"), quantized by nature
    /// (SPACE/F only) and a WAKE source like motion; `last_ms` is the
    /// PREVIOUS frame's render wall-clock ms (`<= 0.0` before the first
    /// present — skipped, never a fake bar); `fg_mult` is the FG family's
    /// presented-per-rendered multiplier for that same frame
    /// (`GpuContext::fg_display_mult`, 1.0 when no FG inserts) — the FG
    /// graph segment is `(mult − 1) ×` the base rate.
    pub fn frame(
        &mut self,
        cam: &Camera,
        tod: f32,
        moving: bool,
        tod_moved: bool,
        mode: &'static str,
        last_ms: f32,
        fg_mult: f32,
    ) -> Option<HudFrame> {
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
        // Render mode: changes only on SPACE/F transitions (quantized by
        // nature), and a change WAKES the compass HUD — a faded label update
        // would upload invisible pixels and the switch would go unseen.
        if mode != self.last_mode {
            self.last_mode = mode;
            self.ui.set_mode_label(mode.into());
            self.last_active = Some(Instant::now());
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
        // Compass + clock: awake on camera OR clock activity (and while the
        // pause menu is up — its TOD row wants live clock feedback), asleep
        // HUD_LINGER after both go idle. While asleep nothing updates their
        // properties (an idle camera moves neither heading nor hour), so the
        // faded state costs zero repaints.
        if moving || tod_moved {
            self.last_active = Some(Instant::now());
        }
        let hud_live =
            self.menu_open || self.last_active.is_some_and(|t| t.elapsed() < HUD_LINGER);
        if hud_live != self.last_hud_live {
            self.last_hud_live = hud_live;
            self.ui.set_hud_live(hud_live);
            // Waking: snap the graph to the ring sampled while faded (the
            // fade-in repaints the panel anyway, so this costs nothing new).
            if hud_live && self.visible {
                self.push_graph_rows();
            }
        }
        if self.visible != self.last_hud_on {
            self.last_hud_on = self.visible;
            self.ui.set_hud_on(self.visible);
        }
        // FPS graph. Rust-side sampling is always-on (a GRAPH_BARS-float
        // rotate per tick) so the graph is CURRENT the instant the HUD
        // wakes; the Slint row writes gate on hud_live + visible — the
        // idle-clean contract — and sampling pauses under the menu hold
        // (present_again re-enters at ~140 Hz with a STALE last_ms: pause
        // means pause). A frame slower than GRAPH_TICK closes its own
        // bucket (the acc_n > 0 guard), degrading tick cadence to frame
        // cadence instead of dropping data.
        if last_ms > 0.0 && !self.menu_open {
            self.acc_sum += last_ms;
            self.acc_n += 1;
            self.acc_mult += fg_mult.max(1.0);
        }
        if self.acc_n > 0 && self.last_bucket.map_or(true, |t| t.elapsed() >= GRAPH_TICK) {
            self.last_bucket = Some(Instant::now());
            self.hist.rotate_left(1);
            let base = 1000.0 * self.acc_n as f32 / self.acc_sum;
            // FG surplus: presented minus rendered over the same wall time.
            let fg = (1000.0 * (self.acc_mult - self.acc_n as f32) / self.acc_sum).max(0.0);
            self.hist[GRAPH_BARS - 1] = (base, fg);
            self.acc_sum = 0.0;
            self.acc_n = 0;
            self.acc_mult = 0.0;
            if hud_live && self.visible {
                self.push_graph_rows();
            }
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
