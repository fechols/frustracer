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

// The CPU→GPU wire MOVED to `gfx::hud_frame` (B6b rung 4) so the Vulkan half
// and its headless gate can name it without this module's slint/sdl3 cfg;
// re-exported here so every existing call site is untouched (gfx's rule).
pub use crate::gfx::hud_frame::{DirtyRect, HudFrame};

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
    /// A CLI flag overrode this row's saved value this session — the cyan
    /// "cli" badge (main computes the set once at startup from a pre-parse /
    /// post-parse Opts diff; see settings::cli_overrides).
    pub cli: bool,
    pub control: &'static str,
}

#[derive(Clone, Copy, PartialEq)]
enum MenuPage {
    Main,
    Settings,
}

/// Menu selection cursor (controller/keyboard navigation). Rust owns it like
/// the rest of the menu state; the ui's `sel-*` properties mirror it for the
/// highlight styling. Mouse hover is deliberately a SEPARATE visual (tint vs
/// the selection's gold border) — the two input methods never fight.
#[derive(Clone, Copy, PartialEq)]
enum Sel {
    /// Main page: 0 Resume, 1 Settings, 2 Exit.
    Main(usize),
    /// Settings page, CATEGORIES column: group tab index (into
    /// settings::GROUPS). Up/Down walks the tabs and switches the group live;
    /// Right/A crosses into the rows.
    Tab(usize),
    /// Settings page, CATEGORIES column: the Back button, below the tabs
    /// (Down past the last tab lands here; A returns to the main page).
    BackBtn,
    /// Settings page, ROWS column: row index in the current group. Up/Down
    /// moves between settings; Left/Right changes the value; B returns to the
    /// categories column.
    Row(usize),
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
    /// `--cam-readout`: the pose plate is armed. OFF by default, and the only
    /// thing that reaches its Slint properties — an unarmed session cannot
    /// dirty a pixel of it.
    cam_on: bool,
    /// Its four last-pushed strings, value-guarded like every other readout
    /// here: a PARKED camera re-formats the same text and writes nothing, so
    /// the plate costs zero raster while the pilot lines up a screenshot.
    last_cam: [String; 4],
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
    /// Navigation cursor (pad/keyboard); `sync_sel` mirrors it to the ui.
    sel: Sel,
    /// (id, control tag) per current row — `adjust`/`activate` dispatch on
    /// the control tag exactly like the row TouchArea/ArrowButton handlers.
    rows_meta: Vec<(String, &'static str)>,
    /// True while the settings TextInput has focus (mirrored from the ui's
    /// `edit-focus` callback): input.rs's gate for keyboard nav — typing
    /// "wasd" into a text field must not move the selection.
    text_editing: Rc<std::cell::Cell<bool>>,
    /// Loading-screen last-set values (value-guarded like the compass/clock so
    /// a steady phase re-raster dirties nothing). Seeded to sentinels the first
    /// `loading_frame` can't match.
    last_load_stage: String,
    last_load_phase: String,
    last_load_detail: String,
    last_load_count: String,
    last_load_frac: f32,
    last_load_marquee: f32,
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
        let text_editing = Rc::new(std::cell::Cell::new(false));
        {
            let t = text_editing.clone();
            ui.on_edit_focus(move |f| t.set(f));
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
            cam_on: false,
            last_cam: [const { String::new() }; 4],
            last_move: None,
            last_active: Some(Instant::now()), // show once at boot, then fade
            menu_open: false,
            page: MenuPage::Main,
            group: "Display".to_string(),
            actions,
            sel: Sel::Main(0),
            rows_meta: Vec::new(),
            text_editing,
            last_load_stage: String::new(),
            last_load_phase: String::new(),
            last_load_detail: String::new(),
            last_load_count: String::new(),
            last_load_frac: f32::NAN, // != any real fraction, so first set fires
            last_load_marquee: -1.0,
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
        self.set_sel(Sel::Main(0));
        self.text_editing.set(false);
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
        self.text_editing.set(false);
    }

    /// ESC / pad-B while the menu is open — a hierarchical back-out matching
    /// the two-column settings model: rows focus -> categories, categories
    /// focus -> main page, main page -> unconsumed (the session closes the
    /// menu). Returns true = consumed (stay open).
    pub fn escape(&mut self) -> bool {
        if self.page == MenuPage::Settings {
            // Rows -> categories: land back on the active group's tab.
            if let Sel::Row(_) = self.sel {
                let g = crate::settings::GROUPS
                    .iter()
                    .position(|g| *g == self.group)
                    .unwrap_or(0);
                self.set_sel(Sel::Tab(g));
                self.text_editing.set(false);
                return true;
            }
            // Categories (tabs / Back) -> main page.
            self.page = MenuPage::Main;
            self.ui.set_menu_page("main".into());
            self.set_sel(Sel::Main(0));
            self.text_editing.set(false);
            return true;
        }
        false
    }

    pub fn open_settings_page(&mut self) {
        self.page = MenuPage::Settings;
        self.ui.set_menu_page("settings".into());
        // Land on the active group's tab — NOT Row(0): the rows rebuild one
        // session-loop iteration later (menu_rows_stale), so the row list is
        // stale at this instant.
        let g = crate::settings::GROUPS.iter().position(|g| *g == self.group).unwrap_or(0);
        self.set_sel(Sel::Tab(g));
    }

    pub fn back_to_main(&mut self) {
        self.page = MenuPage::Main;
        self.ui.set_menu_page("main".into());
        self.set_sel(Sel::Main(0));
        self.text_editing.set(false);
    }

    pub fn group(&self) -> &str {
        &self.group
    }

    pub fn set_group(&mut self, g: &str) {
        self.group = g.to_string();
        self.ui.set_menu_group(g.into());
    }

    pub fn set_rows(&mut self, rows: Vec<MenuRow>) {
        self.rows_meta = rows.iter().map(|r| (r.id.clone(), r.control)).collect();
        let converted: Vec<ui::SettingRow> = rows
            .into_iter()
            .map(|r| ui::SettingRow {
                id: r.id.as_str().into(),
                label: r.label.as_str().into(),
                value: r.value.as_str().into(),
                restart: r.restart,
                cli: r.cli,
                control: r.control.into(),
            })
            .collect();
        self.ui.set_rows(slint::ModelRc::new(slint::VecModel::from(converted)));
        // A rebuild recreates every row element: clamp a row cursor to the
        // new length (a group switch can shrink the list) and clear any
        // text-focus latch (the focused TextInput no longer exists).
        if let Sel::Row(i) = self.sel {
            let n = self.rows_meta.len();
            let sel = if n == 0 { Sel::BackBtn } else { Sel::Row(i.min(n - 1)) };
            self.set_sel(sel);
        }
        self.text_editing.set(false);
    }

    // ── Controller/keyboard menu navigation (src/pad.rs edges + input.rs's
    // menu_* key edges). Rust owns the cursor; activation pushes the SAME
    // HudAction variants the TouchArea callbacks push, onto the same queue —
    // one action path, no semantics drift.

    /// True while the settings TextInput has focus — input.rs's keyboard-nav
    /// gate (WASD/arrows must edit text, not move the selection).
    pub fn text_editing(&self) -> bool {
        self.text_editing.get()
    }

    /// Mirror the cursor to the ui's `sel-*` properties. Equal-value sets
    /// are Slint no-ops, so an unchanged cursor dirties nothing.
    fn set_sel(&mut self, sel: Sel) {
        self.sel = sel;
        self.ui.set_sel_main(if let Sel::Main(i) = sel { i as i32 } else { -1 });
        self.ui.set_sel_tab(if let Sel::Tab(i) = sel { i as i32 } else { -1 });
        self.ui.set_sel_back(sel == Sel::BackBtn);
        self.ui.set_sel_row(if let Sel::Row(i) = sel { i as i32 } else { -1 });
    }

    /// Focus a category tab AND switch the displayed group to it live, so the
    /// highlight and the rows panel never desync (the old up/down-through-tabs
    /// bug left them out of sync). A redundant switch — the group is already
    /// active — pushes no action.
    fn select_tab(&mut self, j: usize) {
        self.set_sel(Sel::Tab(j));
        let g = crate::settings::GROUPS[j];
        if g != self.group {
            self.actions.borrow_mut().push(HudAction::Group(g.to_string()));
        }
    }

    /// Cross from the categories column into the rows column (Right / A on a
    /// tab). No-op if the current group has no rows.
    fn enter_rows(&mut self) {
        if !self.rows_meta.is_empty() {
            self.set_sel(Sel::Row(0));
        }
    }

    /// Up (-1) / Down (+1). Main page: 3-item cyclic menu. Settings page is a
    /// TWO-COLUMN model, so up/down stays WITHIN the focused column:
    ///   Categories (left): Tab(0)..Tab(G-1) then the Back button; landing on
    ///     a tab switches the group so the rows panel updates live.
    ///   Rows (right): Row(0)..Row(N-1).
    /// Both columns CLAMP at their ends — Left/Right/B cross columns, not
    /// up/down (held auto-repeat wants a stop, not a teleport across columns).
    pub fn nav(&mut self, dy: i32) {
        match self.page {
            MenuPage::Main => {
                let i = if let Sel::Main(i) = self.sel { i } else { 0 };
                self.set_sel(Sel::Main((i as i32 + dy).rem_euclid(3) as usize));
            }
            MenuPage::Settings => {
                let g = crate::settings::GROUPS.len();
                match self.sel {
                    // Categories column: tabs, then the Back button, clamped.
                    Sel::Tab(i) => {
                        let next = i.min(g - 1) as i32 + dy;
                        if next < 0 {
                            self.select_tab(0);
                        } else if (next as usize) < g {
                            self.select_tab(next as usize);
                        } else {
                            self.set_sel(Sel::BackBtn);
                        }
                    }
                    Sel::BackBtn => {
                        if dy < 0 {
                            self.select_tab(g - 1);
                        }
                    }
                    // Rows column, clamped.
                    Sel::Row(i) => {
                        let n = self.rows_meta.len();
                        if n == 0 {
                            self.set_sel(Sel::BackBtn);
                        } else {
                            self.set_sel(Sel::Row((i as i32 + dy).clamp(0, n as i32 - 1) as usize));
                        }
                    }
                    Sel::Main(_) => self.select_tab(0),
                }
            }
        }
    }

    /// Left (-1) / Right (+1). Categories column: Right crosses into the rows
    /// (Left is a no-op — nothing further left; use B to leave the page).
    /// Rows column: change the highlighted setting's value (the row's < / >
    /// arrows).
    pub fn adjust(&mut self, dir: i32) {
        match self.sel {
            Sel::Tab(_) | Sel::BackBtn => {
                if dir > 0 {
                    self.enter_rows();
                }
            }
            Sel::Row(i) => {
                if let Some((id, control)) = self.rows_meta.get(i) {
                    // Mirror the arrow buttons: cycle/step take both
                    // directions, cyclefwd only its lone ">" arrow, toggle
                    // flips on either direction, text has no arrows.
                    let push = match *control {
                        "cycle" | "step" => Some(dir),
                        "toggle" => Some(1),
                        "cyclefwd" if dir > 0 => Some(1),
                        _ => None,
                    };
                    if let Some(d) = push {
                        self.actions.borrow_mut().push(HudAction::Adjust(id.clone(), d));
                    }
                }
            }
            Sel::Main(_) => {}
        }
    }

    /// A / Enter on the selected item. Main page: the button's own action.
    /// Settings page: a category tab crosses into the rows (like Right); the
    /// Back button returns to main; a row toggles/steps its value. Text rows
    /// stay mouse-edited (v1 — no programmatic TextInput focus).
    pub fn activate(&mut self) {
        match self.sel {
            Sel::Main(0) => self.actions.borrow_mut().push(HudAction::Resume),
            Sel::Main(1) => self.actions.borrow_mut().push(HudAction::OpenSettings),
            Sel::Main(_) => self.actions.borrow_mut().push(HudAction::Quit),
            Sel::Tab(_) => self.enter_rows(),
            Sel::BackBtn => self.actions.borrow_mut().push(HudAction::Back),
            Sel::Row(i) => {
                if let Some((id, control)) = self.rows_meta.get(i) {
                    if matches!(*control, "toggle" | "cyclefwd" | "cycle" | "step") {
                        self.actions.borrow_mut().push(HudAction::Adjust(id.clone(), 1));
                    }
                }
            }
        }
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

    /// `--cam-readout`: arm the pose plate. Separate from `new` because the
    /// two construction sites (loading screen, session) would otherwise both
    /// have to thread a diagnostic flag they have no other use for.
    pub fn set_cam_readout(&mut self, on: bool) {
        self.cam_on = on;
        self.ui.set_cam_on(on);
    }

    /// Format and push the pose plate. `ev` is the aperture in STOPS as the
    /// session holds it (`autoexp`'s eased EV) — the plate reports the scale
    /// and the arm alongside the pose because the aperture's behaviour is a
    /// function of the whole frame's content, so a pose without its lighting
    /// regime does not reproduce anything.
    ///
    /// The QUATERNION convention, stated because a bare 4-tuple is unusable
    /// without it: the camera basis is X = right, Y = up, Z = forward (the
    /// same right-handed basis `Camera::basis` builds), and this is that
    /// rotation. `Camera` itself stores yaw/pitch, so the quaternion is
    /// DERIVED and always roll-free — it cannot express a roll the camera
    /// cannot have.
    fn push_cam(&mut self, cam: &Camera, tod: f32, ev: f32) {
        use glam::{Mat3, Quat, Vec3, Vec3A};
        let p = cam.pos;
        let f = cam.forward();
        let r = f.cross(Vec3A::Y).normalize();
        let u = r.cross(f);
        let q: Quat =
            Quat::from_mat3(&Mat3::from_cols(Vec3::from(r), Vec3::from(u), Vec3::from(f)));
        // A paste-ready `--cam`: eye + a target one unit down the forward
        // ray. `Camera::look_at` re-derives yaw/pitch from exactly that
        // difference, so the round trip is the pose, not an approximation.
        let t = p + f;
        let lines = [
            format!("pos   {:.3}  {:.3}  {:.3}", p.x, p.y, p.z),
            format!("quat  {:.4}  {:.4}  {:.4}  {:.4}", q.x, q.y, q.z, q.w),
            format!(
                "tod {:05.2}   ev {:+.3}  scale {:.2}x  ({})",
                tod.rem_euclid(24.0),
                crate::autoexp::ev_total(ev),
                crate::autoexp::exposure(ev),
                crate::autoexp::mode().as_str()
            ),
            format!(
                "--cam {:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                p.x, p.y, p.z, t.x, t.y, t.z
            ),
        ];
        // Value-guarded, per line (the push_graph_rows discipline).
        if lines[0] != self.last_cam[0] {
            self.ui.set_cam_pos(lines[0].as_str().into());
            self.last_cam[0] = lines[0].clone();
        }
        if lines[1] != self.last_cam[1] {
            self.ui.set_cam_quat(lines[1].as_str().into());
            self.last_cam[1] = lines[1].clone();
        }
        if lines[2] != self.last_cam[2] {
            self.ui.set_cam_exp(lines[2].as_str().into());
            self.last_cam[2] = lines[2].clone();
        }
        if lines[3] != self.last_cam[3] {
            self.ui.set_cam_flag(lines[3].as_str().into());
            self.last_cam[3] = lines[3].clone();
        }
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
        aexp_ev: f32,
    ) -> Option<HudFrame> {
        // --cam-readout, before the quantized readouts below: this one is
        // DELIBERATELY unquantized (a pose is what it is) and is the only
        // thing here that a parked camera still updates, because its aperture
        // line keeps moving while the controller eases.
        if self.cam_on {
            self.push_cam(cam, tod, aexp_ev);
        }
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

        self.raster()
    }

    /// Advance Slint timers/animations, re-rasterize only the dirty region,
    /// and pack it into a `HudFrame` (or `None` when nothing changed). The
    /// pack layout is a cross-module contract with the backend hud modules —
    /// `gfx::hud_frame::pack_rects` is the ONE copy, shared by `frame` (the
    /// session HUD), `loading_frame` (the loading screen) and V21's synthetic
    /// fixture, so none of them can disagree about a row's byte length.
    fn raster(&mut self) -> Option<HudFrame> {
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
        // Clamp to the buffer, drop empties and pack each rect's rows tightly
        // in rect order — `gfx::hud_frame::pack_rects`, the one writer of the
        // layout (Slint shouldn't produce out-of-bounds rects, but the packer
        // clamps FIRST so the bytes and the rect list agree by construction).
        let src: &[u8] = unsafe {
            std::slice::from_raw_parts(self.buf.as_ptr() as *const u8, self.buf.len() * 4)
        };
        let frame = crate::gfx::hud_frame::pack_rects(src, self.w, self.h, rects)?;
        // FRUSTRACER_HUD_STATS=1: one line per NON-EMPTY upload — the
        // dirty-rect acceptance probe. A still HUD must print nothing at all
        // between clock-minute ticks; a ticking digit is a few KB, never a
        // window-sized copy.
        static STATS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *STATS.get_or_init(|| std::env::var_os("FRUSTRACER_HUD_STATS").is_some()) {
            let list: Vec<String> = frame
                .rects
                .iter()
                .map(|r| format!("{}x{}+{}+{}", r.w, r.h, r.x, r.y))
                .collect();
            eprintln!(
                "hud: {} dirty rect(s), {} bytes [{}]",
                frame.rects.len(),
                frame.bytes.len(),
                list.join(" ")
            );
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
        Some(frame)
    }

    /// Composite the HUD over an SDR `0x00RRGGBB` present buffer, premultiplied
    /// `over`, in DISPLAY space — the same compromise `hud.hlsl`'s SDR arm
    /// makes when it blends against a gamma-encoded backbuffer.
    ///
    /// `--cinematic` only, and the ONE path in the tree that puts the HUD into
    /// a saved image (P screenshots and `--check` PNGs deliberately read
    /// pre-composite sources). The whole buffer composites: dirty rects are a
    /// GPU-upload optimization, and a capture has no persistent target to be
    /// incremental against.
    pub fn composite_sdr(&self, present: &mut [u32], w: usize, h: usize) {
        if w != self.w as usize || h != self.h as usize {
            eprintln!(
                "hud: composite of a {}x{} overlay into a {w}x{h} frame — skipped",
                self.w, self.h
            );
            return;
        }
        if present.len() < w * h || self.buf.len() < w * h {
            return;
        }
        for (dst, p) in present.iter_mut().zip(self.buf.iter()) {
            *dst = crate::cinematic::over_sdr(*dst, p.red, p.green, p.blue, p.alpha);
        }
    }

    /// Run the wall-clock fade animations to rest, so a capture never catches
    /// the HUD mid-fade-in. Slint's animations are driven by real time, so a
    /// cold first frame is always partway through one; this pumps frames until
    /// two consecutive ones report nothing dirty (capped, so a permanently
    /// animating element can never hang a render).
    /// `fill_secs` additionally pumps the FPS graph. The graph is 40 bars of
    /// 125 ms of WALL CLOCK, so a HUD that has only just been settled shows an
    /// empty one — the fades finish in ~0.3 s and there is nothing to draw. A
    /// capture wants a populated graph, and 5 s of pumping (once, at
    /// construction — the capture caches the Hud) is the only way to get one.
    /// The frame times fed are plausible and slightly varied so the bars read
    /// as a live trace rather than a flat wall.
    pub fn settle(&mut self, cam: &Camera, tod: f32, mode: &'static str, fill_secs: f32) {
        let t0 = Instant::now();
        let mut quiet = 0;
        while quiet < 2 && t0.elapsed() < std::time::Duration::from_secs(2) {
            match self.frame(cam, tod, true, true, mode, 1000.0 / 60.0, 1.0, 0.0) {
                Some(_) => quiet = 0,
                None => quiet += 1,
            }
            std::thread::sleep(std::time::Duration::from_millis(8));
        }
        if fill_secs <= 0.0 {
            return;
        }
        let t1 = Instant::now();
        let mut i = 0u32;
        while t1.elapsed().as_secs_f32() < fill_secs {
            // ~55-70 fps, gently varying: a believable trace.
            let ms = 15.5 + 2.5 * (i as f32 * 0.21).sin() + 1.0 * (i as f32 * 0.07).cos();
            let _ = self.frame(cam, tod, true, true, mode, ms, 1.0, 0.0);
            std::thread::sleep(std::time::Duration::from_millis(8));
            i += 1;
        }
    }

    /// Is the loading page up? `session()` reads it to tell a FIRST entry (the
    /// page is still live, and its eager init must keep repainting it) from a
    /// resize re-entry, which needs no extra parameter to say so.
    pub fn is_loading(&self) -> bool {
        self.ui.get_loading()
    }

    /// Show or hide the loading page. Hiding it reveals the frame behind — a
    /// full-window change (the page's scrim covers everything), so the next
    /// `frame()`/`raster()` repaints wholesale.
    pub fn set_loading(&mut self, on: bool) {
        if self.ui.get_loading() != on {
            self.ui.set_loading(on);
            self.force_full = true;
        }
    }

    /// Rasterize one loading-screen frame from a progress snapshot. `marquee`
    /// is a Rust-driven `[0,1]` sweep, consumed only while the current phase
    /// is indeterminate (`snap.frac < 0`). Every property write is value-
    /// guarded like the compass/clock, so a steady phase between ticks dirties
    /// nothing; the marquee is the one thing that legitimately animates (its
    /// job is to show liveness through a long unmetered phase like the world
    /// BVH build).
    pub fn loading_frame(
        &mut self,
        snap: &crate::progress::Snapshot,
        marquee: f32,
    ) -> Option<HudFrame> {
        if !self.ui.get_loading() {
            self.ui.set_loading(true);
            self.force_full = true;
        }
        // Outer stage row: "i / n  name" — only when the loader set a stage
        // (a single-scene load leaves it blank). The name is dropped WITH its
        // separator when empty, which is the world loader's opening state: it
        // arms the row at (0, n, "") and the name is whichever island finished
        // LAST, so there is no honest one to show until the first lands. A
        // bare "0 / 7" is the truth there; "0 / 7  " trailed two spaces.
        let stage = if snap.stage_total == 0 {
            String::new()
        } else if snap.stage_name.is_empty() {
            format!("{} / {}", snap.stage_done, snap.stage_total)
        } else {
            format!("{} / {}  {}", snap.stage_done, snap.stage_total, snap.stage_name)
        };
        if stage != self.last_load_stage {
            self.ui.set_load_stage(stage.as_str().into());
            self.last_load_stage = stage;
        }
        if self.last_load_phase != snap.phase_label {
            self.ui.set_load_phase(snap.phase_label.into());
            self.last_load_phase = snap.phase_label.to_string();
        }
        if self.last_load_detail != snap.detail {
            self.ui.set_load_detail(snap.detail.as_str().into());
            self.last_load_detail = snap.detail.clone();
        }
        let count = if snap.total > 0 {
            format!("{} / {}", snap.done, snap.total)
        } else {
            String::new()
        };
        if count != self.last_load_count {
            self.ui.set_load_count(count.as_str().into());
            self.last_load_count = count;
        }
        if snap.frac < 0.0 {
            // Indeterminate: sweep the marquee; flip frac to the marquee
            // sentinel once on entry.
            if self.last_load_frac >= 0.0 {
                self.ui.set_load_frac(-1.0);
                self.last_load_frac = -1.0;
            }
            if (marquee - self.last_load_marquee).abs() > 1.0 / 256.0 {
                let m = marquee.clamp(0.0, 1.0);
                self.ui.set_load_marquee(m);
                self.last_load_marquee = m;
            }
        } else {
            // Determinate: quantize to 1/256 so sub-pixel churn is idle.
            let q = (snap.frac.clamp(0.0, 1.0) * 256.0).round() / 256.0;
            if (q - self.last_load_frac).abs() > 0.5 / 256.0 {
                self.ui.set_load_frac(q);
                self.last_load_frac = q;
            }
        }
        self.raster()
    }
}
