//! JSON-persisted user settings (`frustracer-settings.json` next to the exe).
//!
//! Precedence is strictly layered: compiled defaults < settings file < CLI
//! flags. That falls out of ordering — `main()` applies the file's values
//! right after the `Opts` defaults literal and BEFORE the parse loop, so a
//! flag parsed later simply overwrites the same field (and the global statics
//! the same way: the file calls the exact setters the flag arms call, and a
//! later flag arm stores over them). The file is only ever WRITTEN by the
//! pause menu's edits (auto-save on change) — keyboard toggles (N/G/X/…) are
//! experimentation, not preferences, and deliberately never persist.
//!
//! Every leaf field is `Option<T>` with `None` = "the user never set this in
//! the menu": only `Some` fields are applied at startup and only `Some`
//! fields are serialized, so the file stays a sparse record of deliberate
//! choices and the compiled defaults remain the single source of truth for
//! everything untouched.
//!
//! **Headless runs ignore the file entirely** (`headless_args`): any
//! `--check*` / `--*-dump` / `--spin*` invocation — and `--no-settings` —
//! drops it with one loud line. The gates' whole value is that a command
//! line fully determines the run; a config file varying them with invisible
//! machine state would break exactly that. Reproducible-viewpoint machinery
//! (`--cam`, `--stress`, `--tile`) is likewise excluded from the schema:
//! those are per-invocation experiment shapes, not preferences.
//!
//! Enum-ish fields serialize as the CLI's OWN strings (`"quality"`,
//! `"fsr4"`, `"cuda"`, …) so the file reads like a command line and the two
//! vocabularies cannot mean different things. The lists are pinned against
//! their consumers by `self_test` (run by `--check`) — the drift guard for
//! the accepted duplication with main.rs's parse loop; each parse arm there
//! carries a "mirrored in settings.rs" cross-reference and vice versa.
//!
//! A file value that fails validation is a loud line + IGNORED (the compiled
//! default rules) — never an exit. This deliberately differs from the CLI's
//! exit(2): a stale or hand-edited file must not brick the app.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Bumped when a field changes meaning (serde ignores unknown fields and
/// defaults missing ones, so additions don't need a bump). `save` stamps it.
pub const VERSION: u32 = 1;

pub const FILE_NAME: &str = "frustracer-settings.json";

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

macro_rules! opt_fields {
    ($(#[$m:meta])* pub struct $name:ident { $($(#[$fm:meta])* pub $f:ident : $t:ty,)* }) => {
        $(#[$m])*
        #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
        #[serde(default)]
        pub struct $name {
            $(
                $(#[$fm])*
                #[serde(skip_serializing_if = "Option::is_none")]
                pub $f: Option<$t>,
            )*
        }
    };
}

/// The whole settings file. Group structs mirror the pause menu's pages.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub version: u32,
    pub display: Display,
    pub renderer: Renderer,
    pub upscaler: Upscaler,
    pub effects: Effects,
    pub scene: SceneChoice,
    pub advanced: Advanced,
}

opt_fields! {
    /// Presentation / window (all restart-tier except `hud`).
    pub struct Display {
        /// --vsync / --no-vsync
        pub vsync: bool,
        /// --hdr / --no-hdr (the scRGB f16 swapchain)
        pub hdr: bool,
        /// --hdr-paper-white <nits>
        pub hdr_paper_white: f32,
        /// --hdr-peak <nits>
        pub hdr_peak: f32,
        /// HUD compass+clock visibility (no CLI flag; live, F1 toggles).
        pub hud: bool,
    }
}

opt_fields! {
    /// Render mode + per-frame quality knobs.
    pub struct Renderer {
        /// "cpu" | "gpu" | "dxr" (--cpu / --gpu / --dxr; sets mode_explicit)
        pub mode: String,
        /// --lock-res: quality|balanced|performance|ultra-performance|native|
        /// dynamic or a ratio in (0,1] as a string (restart-tier)
        pub lock_res: String,
        /// --spp 1..=128 (live: U cycles)
        pub spp: u32,
        /// Quality preset 1..=3 (the 1/2/3 keys; no CLI flag; live)
        pub preset: u32,
        /// "off" | "ao" | "gi" (the H key; no CLI flag; live, still frames)
        pub bounce: String,
        /// --heightfield / --no-heightfield (ARMS relief; restart — keys the
        /// scene cache and the BVH build)
        pub heightfield: bool,
    }
}

opt_fields! {
    /// Upscaler chain + denoisers.
    pub struct Upscaler {
        /// "auto" (probe the full chain) | "dlss" | "fsr4" | "fsr3" | "xess"
        /// (force-start at that level) | "none" (--no-upscale). Restart-tier:
        /// the chain is wired at GpuContext::new.
        pub chain: String,
        /// --quinlight (restart)
        pub quinlight: bool,
        /// --quin-anchor <n> (restart)
        pub quin_anchor: u32,
        /// "nvidia" | "amd" | "intel" (--prefer-*; restart)
        pub prefer: String,
        /// --oidn (live: N toggles)
        pub oidn: bool,
        /// --oidn-no-temporal inverse (live: M toggles)
        pub oidn_temporal: bool,
        /// --oidn-post (live: N cycles placement in XeSS mode)
        pub oidn_post: bool,
        /// "fast" | "balanced" | "high" (--oidn-quality; restart)
        pub oidn_quality: String,
        /// "default" | "cpu" | "sycl" | "cuda" | "hip" (--oidn-device; restart)
        pub oidn_device: String,
        /// --oidn-no-clean-aux inverse (restart)
        pub oidn_clean_aux: bool,
        /// --nppd (live: J toggles)
        pub nppd: bool,
        /// "auto" | "cpu" | "dml" | "dml:<n>" (--nppd-device; restart)
        pub nppd_device: String,
        /// --xess-autoexposure (restart)
        pub xess_autoexposure: bool,
        /// --no-adaptive inverse (restart)
        pub adaptive: bool,
        pub fsr_max_radiance: f32,
        pub fsr_stability_bias: f32,
        pub fsr_radiance_clip_k: f32,
        pub fsr_disocclusion_threshold: f32,
        pub fsr_normal_strength: f32,
        pub fsr_kernel_relaxation: f32,
    }
}

opt_fields! {
    /// Sky/light/texture effects — the "knob before scene load" family plus
    /// the per-frame atomics (bloom/clouds/fireflies are live-flippable).
    pub struct Effects {
        /// --no-bloom inverse (live, display-stage — no reset)
        pub bloom: bool,
        /// --no-clouds inverse (live: frame=0, histories kept)
        pub clouds: bool,
        /// --no-fireflies inverse (live: frame=0, histories kept)
        pub fireflies: bool,
        /// --fireflies N (1..=64; live)
        pub fireflies_count: u32,
        /// --tod <hour> (live: the scrub keys / menu slider)
        pub tod: f32,
        /// --aniso 1..=16 (restart: baked into the GPU static sampler)
        pub aniso: u32,
        /// --no-mips inverse (restart: load-time)
        pub mips: bool,
        /// --no-h2n inverse (restart: load-time, keys the scene cache)
        pub h2n: bool,
        /// --no-n2h inverse (restart: load-time, keys the scene cache)
        pub n2h: bool,
        /// --no-tinted-shadows inverse (restart: load-time)
        pub tinted_shadows: bool,
        /// --no-spray inverse (restart: keys the scene cache)
        pub spray: bool,
        /// --no-depth-tint inverse (restart)
        pub depth_tint: bool,
        /// --no-water inverse (restart: keys the scene cache)
        pub water: bool,
    }
}

opt_fields! {
    /// What to render at boot (restart-tier; the CLI's positional arg and
    /// --world/--no-world). Applied only when the CLI named NO scene source —
    /// a CLI scene path or --world always replaces the file's choice outright
    /// (never a conflict error from a file value).
    pub struct SceneChoice {
        /// Path of a model to load (the positional arg)
        pub scene_path: String,
        /// --world / --no-world
        pub world: bool,
    }
}

opt_fields! {
    /// A/B levers, build knobs, debug, SDK paths (all restart-tier; the
    /// levers exist for attributable A/B measurement — a mid-session flip
    /// would corrupt exactly the comparisons they exist for).
    pub struct Advanced {
        /// "sah" | "lbvh" | "ploc" | "som"
        pub bvh_builder: String,
        pub bvh_ctrav: f32,
        /// >= 2
        pub bvh_maxleaf: u32,
        /// 1 | 3
        pub bvh_axes: u32,
        /// Triangles per BLAS; 0 = --no-blas-split (0 is not a legal CLI cap,
        /// so it is free to mean "off" here)
        pub blas_split: u32,
        pub ftree: bool,
        pub ftree_tiles: bool,
        pub temporal: bool,
        pub replay: bool,
        pub adopt: bool,
        pub discard_seeds: bool,
        pub defer_shade: bool,
        pub hemi_share: bool,
        pub cut_rays: bool,
        pub cut_hemi: bool,
        /// --bc7 (the `fast` profile unless bc7_quality names one)
        pub bc7: bool,
        /// "ultrafast" | "fast" | "basic" | "slow" (implies bc7)
        pub bc7_quality: String,
        /// --fsr4's REQUIRED semantics for a "fsr4" chain force
        pub fsr4_required: bool,
        pub gpu_debug: bool,
        pub pix_markers: bool,
        pub gpu_timing: bool,
        pub sl_path: String,
        pub oidn_path: String,
        pub nppd_path: String,
        pub nppd_model: String,
        pub xess_path: String,
        pub ffx_path: String,
        pub dxc_path: String,
        pub pix_path: String,
    }
}

// ---------------------------------------------------------------------------
// Load / save
// ---------------------------------------------------------------------------

/// Where the file lives: next to the exe (the in-tree convention — SDK paths
/// are manifest-relative, the scene cache writes sidecars next to sources;
/// there is no AppData convention anywhere in this project), CWD fallback.
pub fn path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
        .join(FILE_NAME)
}

/// Should this invocation ignore the settings file? Any headless gate /
/// benchmark flag, or the explicit `--no-settings` opt-out. Pure so
/// `self_test` can pin it.
pub fn headless_args<I: IntoIterator<Item = S>, S: AsRef<str>>(args: I) -> bool {
    args.into_iter().any(|a| {
        let a = a.as_ref();
        a == "--no-settings"
            || a.starts_with("--check")
            || a.ends_with("-dump")
            || a == "--spin"
            || a == "--spin-frames"
    })
}

/// Read + parse only — no side effects. Missing file = silent defaults;
/// unreadable/corrupt file = one loud line + defaults, never a panic.
pub fn load() -> Settings {
    let p = path();
    let text = match std::fs::read_to_string(&p) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Settings::default(),
        Err(e) => {
            eprintln!("settings: cannot read {} ({e}) — using defaults", p.display());
            return Settings::default();
        }
    };
    // Editors (and PowerShell's utf8 encoding) prepend a BOM, which
    // serde_json rejects as "expected value" — a hand-edited file must
    // still load, so strip it.
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    match serde_json::from_str::<Settings>(text) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("settings: {} is not valid ({e}) — using defaults", p.display());
            Settings::default()
        }
    }
}

/// Atomic-ish write: temp file + rename (Windows rename won't clobber, so the
/// stale destination is removed first — the small race is acceptable for a
/// preferences file). Failure is a loud line, never fatal.
pub fn save(s: &Settings) {
    let mut stamped = s.clone();
    stamped.version = VERSION;
    let p = path();
    let json = match serde_json::to_string_pretty(&stamped) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("settings: serialize failed ({e}) — not saved");
            return;
        }
    };
    let tmp = p.with_extension("json.tmp");
    let write = std::fs::write(&tmp, &json).and_then(|()| {
        let _ = std::fs::remove_file(&p);
        std::fs::rename(&tmp, &p)
    });
    if let Err(e) = write {
        eprintln!("settings: cannot write {} ({e}) — not saved", p.display());
        let _ = std::fs::remove_file(&tmp);
    }
}

// ---------------------------------------------------------------------------
// Vocabulary parsers (shared by apply + self_test — the pin against drift)
// ---------------------------------------------------------------------------

/// "cpu" | "gpu" | "dxr" -> (dxr, gpu) exactly as the --cpu/--gpu/--dxr arms
/// set them (mirrored in main.rs's parse loop).
pub fn parse_mode(s: &str) -> Option<(bool, bool)> {
    match s {
        "cpu" => Some((false, false)),
        "gpu" => Some((false, true)),
        "dxr" => Some((true, false)),
        _ => None,
    }
}

/// The chain vocabulary. Returns the level to force, None-inner = "auto"
/// (probe all), Some(None) never occurs; "none" is the empty chain.
pub enum ChainChoice {
    Auto,
    Force(crate::upchain::UpLevel),
    None,
}

pub fn parse_chain(s: &str) -> Option<ChainChoice> {
    use crate::upchain::UpLevel;
    match s {
        "auto" => Some(ChainChoice::Auto),
        "dlss" => Some(ChainChoice::Force(UpLevel::Dlss)),
        "fsr4" => Some(ChainChoice::Force(UpLevel::Fsr4)),
        "fsr3" => Some(ChainChoice::Force(UpLevel::Fsr3)),
        "xess" => Some(ChainChoice::Force(UpLevel::Xess)),
        "none" => Some(ChainChoice::None),
        _ => None,
    }
}

pub fn parse_prefer(s: &str) -> Option<crate::gpu::adapter::Prefer> {
    use crate::gpu::adapter::Prefer;
    match s {
        "nvidia" => Some(Prefer::Nvidia),
        "amd" => Some(Prefer::Amd),
        "intel" => Some(Prefer::Intel),
        _ => None,
    }
}

/// Names map to oidn.h OIDNDeviceType values (mirrors --oidn-device).
pub fn parse_oidn_device(s: &str) -> Option<i32> {
    match s {
        "default" => Some(0),
        "cpu" => Some(1),
        "sycl" => Some(2),
        "cuda" => Some(3),
        "hip" => Some(4),
        _ => None,
    }
}

pub fn parse_oidn_quality(s: &str) -> Option<i32> {
    match s {
        "fast" => Some(crate::oidn::QUALITY_FAST),
        "balanced" => Some(crate::oidn::QUALITY_BALANCED),
        "high" => Some(crate::oidn::QUALITY_HIGH),
        _ => None,
    }
}

/// Mirrors --nppd-device: "auto" = None (DML then CPU), "cpu" = Some(-1),
/// "dml" = Some(0), "dml:<n>" = Some(n). Outer None = invalid.
pub fn parse_nppd_device(s: &str) -> Option<Option<i32>> {
    match s {
        "auto" => Some(None),
        "cpu" => Some(Some(-1)),
        "dml" => Some(Some(0)),
        _ => s
            .strip_prefix("dml:")
            .and_then(|n| n.parse::<i32>().ok())
            .filter(|n| *n >= 0)
            .map(|n| Some(Some(n)))?,
    }
}

pub fn parse_bounce(s: &str) -> Option<u32> {
    match s {
        "off" => Some(0),
        "ao" => Some(1),
        "gi" => Some(2),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

/// Side channels `apply_to_opts` cannot reach through `&mut Opts`: parse-loop
/// locals in `main()` that the file's values must also feed. `main()` merges
/// these BEFORE its parse loop runs, so CLI flags still override.
#[derive(Default)]
pub struct AppliedFx {
    /// The file forced an FSR level — flips the default adapter preference to
    /// AMD exactly like --fsr/--fsr3 (only when no explicit prefer exists).
    pub fsr_forced: bool,
    /// scene.scene_path: applies only if the CLI named no scene source.
    pub scene_path: Option<String>,
    /// scene.world: same rule.
    pub world: Option<bool>,
}

fn warn(field: &str, val: &str) {
    eprintln!("settings: {field} = '{val}' is not a valid value — ignored");
}

/// Write the file's `Some` fields into `opts` through the same conversions
/// the CLI parse arms use. Called after the `Opts` defaults literal and
/// before the parse loop; a flag parsed later overwrites. (Session-start
/// state with no `Opts` field — preset, bounce, hud, tod-as-live-state —
/// is consumed by `run_window`/`session` directly from the `Settings` the
/// caller keeps.)
pub fn apply_to_opts(s: &Settings, opts: &mut crate::Opts) -> AppliedFx {
    let mut fx = AppliedFx::default();

    // Display
    let d = &s.display;
    if let Some(v) = d.vsync {
        opts.vsync = v;
    }
    if let Some(v) = d.hdr {
        opts.hdr = v;
    }
    if let Some(v) = d.hdr_paper_white {
        if v >= 1.0 {
            opts.hdr_paper_white = v;
        } else {
            warn("display.hdr_paper_white", &v.to_string());
        }
    }
    if let Some(v) = d.hdr_peak {
        if v >= 1.0 {
            opts.hdr_peak = Some(v);
        } else {
            warn("display.hdr_peak", &v.to_string());
        }
    }

    // Renderer
    let r = &s.renderer;
    if let Some(m) = r.mode.as_deref() {
        match parse_mode(m) {
            Some((dxr, gpu)) => {
                opts.dxr = dxr;
                opts.gpu = gpu;
                opts.mode_explicit = true;
            }
            None => warn("renderer.mode", m),
        }
    }
    if let Some(l) = r.lock_res.as_deref() {
        // Mirrors --lock-res: one value, BOTH scales, and it counts as an
        // explicit choice for the vendor-default policy.
        if l == "dynamic" {
            opts.lock_scale = None;
            opts.gpu_lock_scale = None;
            opts.lock_res_explicit = true;
        } else {
            match crate::xess::lock_scale(l) {
                Some(scale) => {
                    opts.lock_scale = Some(scale);
                    opts.gpu_lock_scale = Some(scale);
                    opts.lock_res_explicit = true;
                }
                None => warn("renderer.lock_res", l),
            }
        }
    }
    if let Some(n) = r.spp {
        if (1..=crate::dlss::MAX_SPP).contains(&n) {
            opts.spp = n;
        } else {
            warn("renderer.spp", &n.to_string());
        }
    }
    if let Some(v) = r.heightfield {
        crate::bvh::set_height_armed(v);
        crate::bvh::set_height_on(v);
    }
    // r.preset / r.bounce: session-start state, consumed by run_window.

    // Upscaler
    let u = &s.upscaler;
    if let Some(c) = u.chain.as_deref() {
        match parse_chain(c) {
            Some(ChainChoice::Auto) => opts.chain = crate::upchain::UpChain::ALL,
            Some(ChainChoice::Force(level)) => {
                opts.chain.force(level);
                if matches!(level, crate::upchain::UpLevel::Fsr4 | crate::upchain::UpLevel::Fsr3) {
                    fx.fsr_forced = true;
                }
            }
            Some(ChainChoice::None) => opts.chain = crate::upchain::UpChain::NONE,
            None => warn("upscaler.chain", c),
        }
    }
    if let Some(v) = u.quinlight {
        opts.quin = v;
    }
    if let Some(v) = u.quin_anchor {
        opts.quin_anchor = Some(v);
    }
    if let Some(p) = u.prefer.as_deref() {
        match parse_prefer(p) {
            Some(v) => opts.prefer = Some(v),
            None => warn("upscaler.prefer", p),
        }
    }
    if let Some(v) = u.oidn {
        opts.oidn = v;
    }
    if let Some(v) = u.oidn_temporal {
        opts.oidn_temporal = v;
    }
    if let Some(v) = u.oidn_post {
        opts.oidn_post = v;
    }
    if let Some(q) = u.oidn_quality.as_deref() {
        match parse_oidn_quality(q) {
            Some(v) => opts.oidn_quality = v,
            None => warn("upscaler.oidn_quality", q),
        }
    }
    if let Some(dev) = u.oidn_device.as_deref() {
        match parse_oidn_device(dev) {
            Some(v) => opts.oidn_device = v,
            None => warn("upscaler.oidn_device", dev),
        }
    }
    if let Some(v) = u.oidn_clean_aux {
        opts.oidn_clean_aux = v;
    }
    if let Some(v) = u.nppd {
        opts.nppd = v;
    }
    if let Some(dev) = u.nppd_device.as_deref() {
        match parse_nppd_device(dev) {
            Some(v) => opts.nppd_device = v,
            None => warn("upscaler.nppd_device", dev),
        }
    }
    if let Some(v) = u.xess_autoexposure {
        opts.xess_autoexposure = v;
    }
    if let Some(v) = u.adaptive {
        opts.adaptive = v;
    }
    let t = &mut opts.fsr_tune;
    if let Some(v) = u.fsr_max_radiance {
        t.max_radiance = Some(v);
    }
    if let Some(v) = u.fsr_stability_bias {
        t.stability_bias = Some(v);
    }
    if let Some(v) = u.fsr_radiance_clip_k {
        t.radiance_clip_k = Some(v);
    }
    if let Some(v) = u.fsr_disocclusion_threshold {
        t.disocclusion_threshold = Some(v);
    }
    if let Some(v) = u.fsr_normal_strength {
        t.normal_strength = Some(v);
    }
    if let Some(v) = u.fsr_kernel_relaxation {
        t.kernel_relaxation = Some(v);
    }

    // Effects: tod rides Opts; the rest are global statics (apply_globals).
    if let Some(h) = s.effects.tod {
        if h.is_finite() {
            opts.tod = Some(h.rem_euclid(24.0));
        } else {
            warn("effects.tod", &h.to_string());
        }
    }

    // Scene choice: side-channel — main() applies these only where the CLI
    // named no scene source (see AppliedFx).
    fx.scene_path = s.scene.scene_path.clone();
    fx.world = s.scene.world;

    // Advanced
    let a = &s.advanced;
    if let Some(b) = a.bvh_builder.as_deref() {
        if matches!(b, "sah" | "lbvh" | "ploc" | "som") {
            opts.bvh_builder = b.to_string();
        } else {
            warn("advanced.bvh_builder", b);
        }
    }
    if let Some(v) = a.bvh_ctrav {
        if v.is_finite() && v >= 0.0 {
            opts.c_trav = v;
        } else {
            warn("advanced.bvh_ctrav", &v.to_string());
        }
    }
    if let Some(n) = a.bvh_maxleaf {
        if n >= 2 {
            opts.max_leaf = n as usize;
        } else {
            warn("advanced.bvh_maxleaf", &n.to_string());
        }
    }
    if let Some(n) = a.bvh_axes {
        if n == 1 || n == 3 {
            opts.split_axes = n as usize;
        } else {
            warn("advanced.bvh_axes", &n.to_string());
        }
    }
    if let Some(n) = a.blas_split {
        // 0 = --no-blas-split (not a legal CLI cap, so free to mean "off").
        opts.blas_split = if n == 0 { None } else { Some(n) };
    }
    if let Some(v) = a.ftree {
        opts.ftree = v;
    }
    if let Some(v) = a.ftree_tiles {
        opts.ftree_tiles = v;
    }
    if let Some(v) = a.temporal {
        opts.temporal = v;
    }
    if let Some(v) = a.replay {
        opts.replay = v;
    }
    if let Some(v) = a.adopt {
        opts.adopt = v;
    }
    if let Some(v) = a.discard_seeds {
        opts.discard_seeds = v;
    }
    if let Some(v) = a.defer_shade {
        opts.defer_shade = v;
    }
    if let Some(v) = a.hemi_share {
        opts.hemi_share = v;
    }
    if let Some(v) = a.cut_rays {
        opts.cut_rays = v;
    }
    if let Some(v) = a.cut_hemi {
        opts.cut_hemi = v;
    }
    if let Some(v) = a.bc7 {
        if v {
            opts.bc7 = opts.bc7.or(Some(crate::bc7::Quality::Fast));
        } else {
            opts.bc7 = None;
        }
    }
    if let Some(q) = a.bc7_quality.as_deref() {
        match crate::bc7::Quality::parse(q) {
            Some(v) => opts.bc7 = Some(v),
            None => warn("advanced.bc7_quality", q),
        }
    }
    if let Some(v) = a.fsr4_required {
        opts.fsr4_required = v;
    }
    if let Some(v) = a.gpu_debug {
        opts.gpu_debug = v;
    }
    if let Some(v) = a.pix_markers {
        opts.pix_markers = v;
    }
    if let Some(v) = a.gpu_timing {
        opts.gpu_timing = v;
    }
    if let Some(p) = a.sl_path.clone() {
        opts.sl_path = p;
    }
    if let Some(p) = a.oidn_path.clone() {
        opts.oidn_path = p;
    }
    if let Some(p) = a.nppd_path.clone() {
        opts.nppd_path = p;
    }
    if let Some(p) = a.nppd_model.clone() {
        opts.nppd_model = p;
    }
    if let Some(p) = a.xess_path.clone() {
        opts.xess_path = p;
    }
    if let Some(p) = a.ffx_path.clone() {
        opts.ffx_path = p;
    }
    if let Some(p) = a.dxc_path.clone() {
        opts.dxc_path = p;
    }
    if let Some(p) = a.pix_path.clone() {
        opts.pix_path = p;
    }

    fx
}

/// The process-global "knob before scene load" statics, set through the SAME
/// setters the flag arms call (a later flag arm stores over them — CLI wins).
/// Ordering note: mips before aniso — `set_mips(false)` forces aniso to 1 and
/// `set_aniso` re-checks the mips switch, the CLI's own ordering contract
/// (texture.rs).
pub fn apply_globals(s: &Settings) {
    let e = &s.effects;
    if let Some(v) = e.mips {
        crate::texture::set_mips(v);
    }
    if let Some(n) = e.aniso {
        if (1..=crate::texture::MAX_ANISO_CAP).contains(&n) {
            crate::texture::set_aniso(n);
        } else {
            warn("effects.aniso", &n.to_string());
        }
    }
    if let Some(v) = e.h2n {
        crate::texture::set_h2n(v);
    }
    if let Some(v) = e.n2h {
        crate::texture::set_n2h(v);
    }
    if let Some(v) = e.tinted_shadows {
        crate::scene::set_tinted_shadows(v);
    }
    if let Some(v) = e.spray {
        crate::scene::set_spray(v);
    }
    if let Some(v) = e.depth_tint {
        crate::scene::set_depth_tint(v);
    }
    if let Some(v) = e.water {
        crate::scene::set_water(v);
    }
    if let Some(v) = e.bloom {
        crate::bloom::set_enabled(v);
    }
    if let Some(v) = e.clouds {
        crate::clouds::set_enabled(v);
    }
    if let Some(v) = e.fireflies {
        crate::fireflies::set_enabled(v);
    }
    if let Some(n) = e.fireflies_count {
        if n > crate::fireflies::MAX_FIREFLIES as u32 {
            eprintln!(
                "settings: effects.fireflies_count {n} clamped to {} (the CB row cap)",
                crate::fireflies::MAX_FIREFLIES
            );
        }
        crate::fireflies::set_count(n);
    }
}

// ---------------------------------------------------------------------------
// Self-test (run by --check — DLL-free, pure)
// ---------------------------------------------------------------------------

pub fn self_test() -> Result<(), String> {
    // Round-trip: a default (all-None) Settings survives serde bit-exactly
    // and serializes SPARSE — no "null" leaves in the file.
    let d = Settings::default();
    let j = serde_json::to_string_pretty(&d).map_err(|e| format!("serialize: {e}"))?;
    if j.contains("null") {
        return Err("default Settings serializes non-sparse (a null leaked)".into());
    }
    let back: Settings = serde_json::from_str(&j).map_err(|e| format!("roundtrip parse: {e}"))?;
    if back != d {
        return Err("default Settings did not round-trip".into());
    }

    // Forward compat: unknown fields ignored, missing groups defaulted, and
    // a partial file leaves everything else None.
    let partial: Settings =
        serde_json::from_str(r#"{"renderer":{"spp":4,"future_field":true},"version":1}"#)
            .map_err(|e| format!("partial parse: {e}"))?;
    if partial.renderer.spp != Some(4) || partial.display.vsync.is_some() {
        return Err("partial file did not default correctly".into());
    }

    // A populated file round-trips value-exactly.
    let mut full = Settings::default();
    full.display.hud = Some(false);
    full.renderer.mode = Some("gpu".into());
    full.renderer.lock_res = Some("balanced".into());
    full.upscaler.chain = Some("fsr3".into());
    full.effects.fireflies_count = Some(24);
    full.advanced.blas_split = Some(0);
    let j = serde_json::to_string(&full).map_err(|e| format!("serialize full: {e}"))?;
    let back: Settings = serde_json::from_str(&j).map_err(|e| format!("full parse: {e}"))?;
    if back != full {
        return Err("populated Settings did not round-trip".into());
    }

    // Vocabulary pins: every string the menu/file may hold must be accepted
    // by its consumer — the drift guard against main.rs's parse arms.
    for m in ["cpu", "gpu", "dxr"] {
        parse_mode(m).ok_or_else(|| format!("mode vocab '{m}' rejected"))?;
    }
    if parse_mode("vulkan").is_some() {
        return Err("mode vocab accepted garbage".into());
    }
    for c in ["auto", "dlss", "fsr4", "fsr3", "xess", "none"] {
        parse_chain(c).ok_or_else(|| format!("chain vocab '{c}' rejected"))?;
    }
    for p in ["nvidia", "amd", "intel"] {
        parse_prefer(p).ok_or_else(|| format!("prefer vocab '{p}' rejected"))?;
    }
    for d in ["default", "cpu", "sycl", "cuda", "hip"] {
        parse_oidn_device(d).ok_or_else(|| format!("oidn_device vocab '{d}' rejected"))?;
    }
    for q in ["fast", "balanced", "high"] {
        parse_oidn_quality(q).ok_or_else(|| format!("oidn_quality vocab '{q}' rejected"))?;
    }
    for n in ["auto", "cpu", "dml", "dml:2"] {
        parse_nppd_device(n).ok_or_else(|| format!("nppd_device vocab '{n}' rejected"))?;
    }
    if parse_nppd_device("dml:-1").is_some() {
        return Err("nppd_device accepted a negative adapter".into());
    }
    for b in ["off", "ao", "gi"] {
        parse_bounce(b).ok_or_else(|| format!("bounce vocab '{b}' rejected"))?;
    }
    // lock_res / bc7_quality delegate to their real consumers — pin that the
    // menu's option lists stay inside what those accept.
    for l in ["quality", "balanced", "performance", "ultra-performance", "native", "0.75"] {
        crate::xess::lock_scale(l).ok_or_else(|| format!("lock_res vocab '{l}' rejected"))?;
    }
    for q in ["ultrafast", "fast", "basic", "slow"] {
        crate::bc7::Quality::parse(q).ok_or_else(|| format!("bc7 vocab '{q}' rejected"))?;
    }

    // The headless predicate: gates must never see the file.
    for probe in [
        vec!["--check"],
        vec!["--check-gpu", "--stress"],
        vec!["--dlss-dump"],
        vec!["--spin", "path"],
        vec!["--no-settings"],
    ] {
        if !headless_args(probe.iter().copied()) {
            return Err(format!("headless_args missed {probe:?}"));
        }
    }
    if headless_args(["--tod", "17.5", "model.obj"].iter().copied()) {
        return Err("headless_args fired on an interactive command line".into());
    }

    Ok(())
}
