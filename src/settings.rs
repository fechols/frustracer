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
        /// --hdr10 / --no-hdr10 (force the 10-bit PQ swapchain — the encode
        /// the wrapper-FG families take automatically on an HDR-on display).
        /// True implies `hdr` at apply time, the CLI arm's own semantics.
        pub hdr10: bool,
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
        /// --fg / --no-fg (frame generation for the wired upscaler family;
        /// ON by default). Restart-tier — the swapchain wrap happens at
        /// creation. The file drives the DEFAULT arm only: it never sets
        /// `fg_explicit`, so fg=true here neither arms the explicit-only
        /// SL DLSS-G fallback nor makes a CLI --quinlight fatal (it disarms
        /// with the loud line instead) — a menu click is a preference, not
        /// a being-told.
        pub fg: bool,
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
        /// --cloud-shadow N (restart: 0 = off; 2..=64 cells/λ; GPU shading
        /// cache, snapshotted at TraceGpu/DxrGpu construction)
        pub cloud_shadow: u32,
        /// --sky-lod K (restart: 1 = off; power of two 2..=32; GPU sky-march
        /// lattice pitch)
        pub sky_lod: u32,
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
        /// BC7 texture compression (ON by default — the GPU encoder at
        /// `fast`; false = --no-bc7. The --bc7-cpu A/B arm is CLI-only.)
        pub bc7: bool,
        /// "ultrafast" | "fast" | "basic" | "slow" (implies bc7 on)
        pub bc7_quality: String,
        /// --dxr-inline 0|1|2 (the DXR ray-dispatch mode; default 1 = inline
        /// RayQuery secondaries)
        pub dxr_inline: u32,
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
            // One clause covers the whole --cinematic-* family, which is why
            // every sub-flag carries that prefix. (--spin needed two arms
            // because --spin-frames does not extend "--spin" as a prefix.)
            || a.starts_with("--cinematic")
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
        if !v {
            opts.hdr10 = false;
            opts.scrgb = false;
        }
    }
    if let Some(v) = d.hdr10 {
        opts.hdr10 = v;
        if v {
            // The CLI arm's semantics: PQ is an HDR mode, so forcing it turns
            // the wide swapchain on (a file with hdr=false + hdr10=true reads
            // as "the PQ flavor", not a contradiction — hdr10 is the more
            // specific choice and wins).
            opts.hdr = true;
            opts.scrgb = false;
        } else {
            // `hdr10: false` in the file is the menu's OFF state, and since PQ
            // became the HDR-display default that has to mean "scRGB", not
            // "whatever the probe picks" — otherwise the row could not turn PQ
            // off at all. Mirrors the `--no-hdr10` arm.
            opts.scrgb = true;
        }
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
        // Mirrors --lock-res: one value, every render mode, and it counts as
        // an explicit choice for the vendor-default policy.
        if l == "dynamic" {
            opts.lock_scale = None;
            opts.lock_res_explicit = true;
        } else {
            match crate::xess::lock_scale(l) {
                Some(scale) => {
                    opts.lock_scale = Some(scale);
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
        opts.heightfield = v;
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
    if let Some(v) = u.fg {
        // The DEFAULT arm only — deliberately never `fg_explicit`: the
        // explicit-only consumers (the SL DLSS-G fallback, the --quinlight
        // exit(2)) key on being TOLD on the command line. CLI --fg/--no-fg
        // parse after this and override per the precedence rule.
        opts.fg = v;
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
    // bc7_quality applies BEFORE the bc7 toggle: a settings file has no flag
    // order, so "quality implies on" (the CLI rule) must not defeat an
    // explicit bc7=false — quality first, then the toggle, makes off always
    // win while bc7=true keeps the quality (`armed_or_default` preserves an
    // armed mode).
    if let Some(q) = a.bc7_quality.as_deref() {
        match crate::bc7::Quality::parse(q) {
            Some(v) => opts.bc7 = opts.bc7.with_quality(v),
            None => warn("advanced.bc7_quality", q),
        }
    }
    if let Some(v) = a.bc7 {
        // The file toggle mirrors --bc7/--no-bc7: true arms the default
        // (keeping an already-armed mode's arm/quality — precedence lets a
        // later CLI --bc7-cpu still win), false is the kill lever.
        if v {
            opts.bc7 = opts.bc7.armed_or_default();
        } else {
            opts.bc7 = crate::bc7::Bc7Mode::Off;
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

    // Effects: the "knob before scene load" levers. These used to be written
    // STRAIGHT into their process-global statics by a separate `apply_globals`,
    // moments before the parse loop wrote the same statics again through the
    // same setters — two writers, layered by call ordering alone. They are
    // ordinary `Opts` fields now (see cli.rs's header), so the file simply
    // seeds them and a CLI flag simply overwrites them; `main`'s lever block is
    // the one place any of it reaches a global.
    let e = &s.effects;
    if let Some(v) = e.mips {
        opts.mips = v;
    }
    if let Some(n) = e.aniso {
        if (1..=crate::texture::MAX_ANISO_CAP).contains(&n) {
            opts.aniso = n;
        } else {
            warn("effects.aniso", &n.to_string());
        }
    }
    if let Some(n) = e.cloud_shadow {
        if n == 0 || (2..=64).contains(&n) {
            opts.cloud_shadow = n;
        } else {
            warn("effects.cloud_shadow", &n.to_string());
        }
    }
    if let Some(k) = e.sky_lod {
        if k.is_power_of_two() && (1..=32).contains(&k) {
            opts.sky_lod = k;
        } else {
            warn("effects.sky_lod", &k.to_string());
        }
    }
    if let Some(v) = e.h2n {
        opts.h2n = v;
    }
    if let Some(v) = e.n2h {
        opts.n2h = v;
    }
    if let Some(v) = e.tinted_shadows {
        opts.tinted_shadows = v;
    }
    if let Some(v) = e.spray {
        opts.spray = v;
    }
    if let Some(v) = e.depth_tint {
        opts.depth_tint = v;
    }
    if let Some(v) = e.water {
        opts.water = v;
    }
    if let Some(v) = e.bloom {
        opts.bloom = v;
    }
    if let Some(v) = e.clouds {
        opts.clouds = v;
    }
    if let Some(v) = e.fireflies {
        opts.fireflies = v;
    }
    if let Some(n) = e.fireflies_count {
        if n > crate::fireflies::MAX_FIREFLIES as u32 {
            eprintln!(
                "settings: effects.fireflies_count {n} clamped to {} (the CB row cap)",
                crate::fireflies::MAX_FIREFLIES
            );
        }
        opts.fireflies_count = n;
    }
    if let Some(n) = s.advanced.dxr_inline {
        if n <= 2 {
            opts.dxr_inline = n;
        } else {
            warn("advanced.dxr_inline", &n.to_string());
        }
    }

    fx
}

// ---------------------------------------------------------------------------
// Pause-menu descriptor (the UI binds rows to this; src/hud consumes it)
// ---------------------------------------------------------------------------

/// How a menu row is manipulated. `CycleFwd` mirrors a cycle KEY (one click =
/// one press — SPACE/U/H/N semantics, forward only); `Cycle` is a ±-steppable
/// option list; `Step*` are ± numeric steps; `Text` is a free-text field
/// (restart-tier paths). One mechanism per row, zero drift from the keys.
pub enum Control {
    Toggle { default: bool },
    Cycle { options: &'static [&'static str], default_ix: usize },
    CycleFwd,
    StepU { min: u32, max: u32, step: u32, default: u32 },
    StepF { min: f32, max: f32, step: f32, default: f32 },
    Text,
}

/// Live rows apply instantly through the SAME code paths as their keys
/// (synthesized `Edges` / the shared atomics); Restart rows edit the settings
/// file and badge "restart" — they apply next launch.
#[derive(PartialEq, Clone, Copy)]
pub enum Tier {
    Live,
    Restart,
}

pub struct MenuItem {
    pub id: &'static str,
    pub label: &'static str,
    pub group: &'static str,
    pub tier: Tier,
    pub control: Control,
    /// Read the persisted value (None = never set — the default rules).
    pub get: fn(&Settings) -> Option<String>,
    /// Write the persisted value (invalid input = ignored, the load() rule).
    pub set: fn(&mut Settings, &str),
}

pub const GROUPS: &[&str] = &["Display", "Renderer", "Upscaler", "Effects", "Scene", "Advanced"];

macro_rules! acc_bool {
    ($($p:ident).+) => {
        (
            |s: &Settings| s.$($p).+.map(|v| if v { "on" } else { "off" }.to_string()),
            |s: &mut Settings, v: &str| {
                s.$($p).+ = match v {
                    "on" => Some(true),
                    "off" => Some(false),
                    _ => s.$($p).+,
                }
            },
        )
    };
}
macro_rules! acc_u32 {
    ($($p:ident).+) => {
        (
            |s: &Settings| s.$($p).+.map(|v| v.to_string()),
            |s: &mut Settings, v: &str| {
                if let Ok(n) = v.parse::<u32>() {
                    s.$($p).+ = Some(n);
                }
            },
        )
    };
}
macro_rules! acc_f32 {
    ($($p:ident).+) => {
        (
            |s: &Settings| s.$($p).+.map(|v| format!("{v}")),
            |s: &mut Settings, v: &str| {
                if let Ok(n) = v.parse::<f32>() {
                    if n.is_finite() {
                        s.$($p).+ = Some(n);
                    }
                }
            },
        )
    };
}
macro_rules! acc_str {
    ($($p:ident).+) => {
        (
            |s: &Settings| s.$($p).+.clone(),
            |s: &mut Settings, v: &str| s.$($p).+ = Some(v.to_string()),
        )
    };
}

macro_rules! item {
    ($id:literal, $label:literal, $group:literal, $tier:expr, $control:expr, $acc:expr) => {{
        let (g, s): (fn(&Settings) -> Option<String>, fn(&mut Settings, &str)) = $acc;
        MenuItem { id: $id, label: $label, group: $group, tier: $tier, control: $control, get: g, set: s }
    }};
}

/// The whole menu, one declarative table. IDs are stable (the UI round-trips
/// them through callbacks); labels carry the key mnemonic for live rows.
/// EVERY CLI-exposable option appears here except the headless/bench harness
/// (--check*/--spin/--*-dump/--cam/--stress/--tile — excluded by design; see
/// the module header).
pub fn menu_items() -> &'static [MenuItem] {
    use Control::*;
    use Tier::*;
    static ITEMS: std::sync::OnceLock<Vec<MenuItem>> = std::sync::OnceLock::new();
    ITEMS.get_or_init(|| {
        vec![
            // ── Display
            item!("hud", "HUD compass + clock (F1)", "Display", Live, Toggle { default: true }, acc_bool!(display.hud)),
            item!("overlay", "quadtree overlay (O, CPU mode)", "Display", Live, Toggle { default: false }, ((|_| None), (|_, _| {}))),
            item!("gpu_tone", "GPU tonemap A/B (B)", "Display", Live, Toggle { default: false }, ((|_| None), (|_, _| {}))),
            item!("vsync", "v-sync", "Display", Restart, Toggle { default: true }, acc_bool!(display.vsync)),
            item!("hdr", "scRGB (HDR) swapchain", "Display", Restart, Toggle { default: true }, acc_bool!(display.hdr)),
            item!("hdr10", "HDR10 (PQ) swapchain", "Display", Restart, Toggle { default: false }, acc_bool!(display.hdr10)),
            item!("hdr_paper_white", "paper white (nits)", "Display", Restart, StepF { min: 80.0, max: 1000.0, step: 20.0, default: 200.0 }, acc_f32!(display.hdr_paper_white)),
            item!("hdr_peak", "peak override (nits)", "Display", Restart, StepF { min: 400.0, max: 4000.0, step: 100.0, default: 1000.0 }, acc_f32!(display.hdr_peak)),
            // ── Renderer
            item!("mode", "render mode (SPACE cycles)", "Renderer", Live, CycleFwd, acc_str!(renderer.mode)),
            item!("preset", "quality preset (1-3)", "Renderer", Live, Cycle { options: &["1", "2", "3"], default_ix: 1 }, acc_u32!(renderer.preset)),
            item!("spp", "samples per pixel (U cycles)", "Renderer", Live, CycleFwd, acc_u32!(renderer.spp)),
            item!("bounce", "hemi bounce (H cycles)", "Renderer", Live, CycleFwd, acc_str!(renderer.bounce)),
            item!("hybrid", "hybrid tracer (R)", "Renderer", Live, Toggle { default: true }, ((|_| None), (|_, _| {}))),
            item!("dynamic", "dynamic res (T, CPU mode)", "Renderer", Live, Toggle { default: true }, ((|_| None), (|_, _| {}))),
            item!("height_on", "relief rendering (V, armed only)", "Renderer", Live, Toggle { default: false }, ((|_| None), (|_, _| {}))),
            item!("lock_res", "render res lock", "Renderer", Restart, Cycle { options: &["quality", "balanced", "performance", "ultra-performance", "native", "dynamic"], default_ix: 0 }, acc_str!(renderer.lock_res)),
            item!("heightfield", "arm heightfield relief", "Renderer", Restart, Toggle { default: false }, acc_bool!(renderer.heightfield)),
            // ── Upscaler
            item!("chain", "upscaler chain start", "Upscaler", Restart, Cycle { options: &["auto", "dlss", "fsr4", "fsr3", "xess", "none"], default_ix: 0 }, acc_str!(upscaler.chain)),
            item!("fg", "frame generation", "Upscaler", Restart, Toggle { default: true }, acc_bool!(upscaler.fg)),
            item!("dlss", "DLSS-RR vs plain (G)", "Upscaler", Live, Toggle { default: true }, ((|_| None), (|_, _| {}))),
            item!("xess", "XeSS vs plain (X)", "Upscaler", Live, Toggle { default: false }, ((|_| None), (|_, _| {}))),
            item!("fsr", "FSR vs plain (K)", "Upscaler", Live, Toggle { default: false }, ((|_| None), (|_, _| {}))),
            item!("oidn", "OIDN denoise (N cycles)", "Upscaler", Live, CycleFwd, acc_bool!(upscaler.oidn)),
            item!("oidn_temporal", "OIDN temporal history (M)", "Upscaler", Live, Toggle { default: true }, acc_bool!(upscaler.oidn_temporal)),
            item!("nppd", "NPPD neural denoise (J)", "Upscaler", Live, Toggle { default: false }, acc_bool!(upscaler.nppd)),
            item!("prefer", "adapter preference", "Upscaler", Restart, Cycle { options: &["nvidia", "amd", "intel"], default_ix: 0 }, acc_str!(upscaler.prefer)),
            item!("quinlight", "quinlight consensus fuse", "Upscaler", Restart, Toggle { default: false }, acc_bool!(upscaler.quinlight)),
            item!("quin_anchor", "quinlight anchor engine", "Upscaler", Restart, StepU { min: 0, max: 3, step: 1, default: 0 }, acc_u32!(upscaler.quin_anchor)),
            item!("oidn_quality", "OIDN quality", "Upscaler", Restart, Cycle { options: &["fast", "balanced", "high"], default_ix: 1 }, acc_str!(upscaler.oidn_quality)),
            item!("oidn_device", "OIDN device", "Upscaler", Restart, Cycle { options: &["default", "cpu", "sycl", "cuda", "hip"], default_ix: 0 }, acc_str!(upscaler.oidn_device)),
            item!("oidn_clean_aux", "OIDN clean guides", "Upscaler", Restart, Toggle { default: true }, acc_bool!(upscaler.oidn_clean_aux)),
            item!("nppd_device", "NPPD device", "Upscaler", Restart, Cycle { options: &["auto", "cpu", "dml"], default_ix: 0 }, acc_str!(upscaler.nppd_device)),
            item!("xess_autoexposure", "XeSS autoexposure", "Upscaler", Restart, Toggle { default: false }, acc_bool!(upscaler.xess_autoexposure)),
            item!("adaptive", "adaptive shading (XeSS)", "Upscaler", Restart, Toggle { default: true }, acc_bool!(upscaler.adaptive)),
            item!("fsr_max_radiance", "FSR-RR max radiance", "Upscaler", Restart, StepF { min: 0.0, max: 64.0, step: 2.0, default: 10.0 }, acc_f32!(upscaler.fsr_max_radiance)),
            item!("fsr_stability_bias", "FSR-RR stability bias", "Upscaler", Restart, StepF { min: 0.0, max: 1.0, step: 0.05, default: 0.5 }, acc_f32!(upscaler.fsr_stability_bias)),
            item!("fsr_radiance_clip_k", "FSR-RR radiance clip k", "Upscaler", Restart, StepF { min: 0.0, max: 10.0, step: 0.5, default: 1.0 }, acc_f32!(upscaler.fsr_radiance_clip_k)),
            item!("fsr_disocclusion_threshold", "FSR-RR disocclusion thr", "Upscaler", Restart, StepF { min: 0.0, max: 1.0, step: 0.05, default: 0.1 }, acc_f32!(upscaler.fsr_disocclusion_threshold)),
            item!("fsr_normal_strength", "FSR-RR normal strength", "Upscaler", Restart, StepF { min: 0.0, max: 2.0, step: 0.1, default: 1.0 }, acc_f32!(upscaler.fsr_normal_strength)),
            item!("fsr_kernel_relaxation", "FSR-RR kernel relaxation", "Upscaler", Restart, StepF { min: 0.0, max: 1.0, step: 0.05, default: 0.5 }, acc_f32!(upscaler.fsr_kernel_relaxation)),
            // ── Effects
            item!("tod", "time of day", "Effects", Live, StepF { min: 0.0, max: 24.0, step: 0.5, default: 12.0 }, acc_f32!(effects.tod)),
            item!("bloom", "bloom (glare)", "Effects", Live, Toggle { default: true }, acc_bool!(effects.bloom)),
            item!("clouds", "volumetric clouds", "Effects", Live, Toggle { default: true }, acc_bool!(effects.clouds)),
            item!("fireflies", "fireflies (night)", "Effects", Live, Toggle { default: true }, acc_bool!(effects.fireflies)),
            item!("fireflies_count", "firefly count", "Effects", Live, StepU { min: 8, max: 64, step: 8, default: 32 }, acc_u32!(effects.fireflies_count)),
            item!("aniso", "max anisotropy", "Effects", Restart, Cycle { options: &["1", "2", "4", "8", "16"], default_ix: 4 }, acc_u32!(effects.aniso)),
            item!("cloud_shadow", "cloud shadow cache (cells/λ; off)", "Effects", Restart, Cycle { options: &["off", "8", "16", "32", "64"], default_ix: 2 }, ((|s: &Settings| s.effects.cloud_shadow.map(|v| if v == 0 { "off".into() } else { v.to_string() })), (|s: &mut Settings, v: &str| {
                s.effects.cloud_shadow = if v == "off" { Some(0) } else { v.parse().ok() };
            }))),
            item!("sky_lod", "sky march lattice (1/K px; off)", "Effects", Restart, Cycle { options: &["off", "2", "4", "8", "16"], default_ix: 2 }, ((|s: &Settings| s.effects.sky_lod.map(|v| if v <= 1 { "off".into() } else { v.to_string() })), (|s: &mut Settings, v: &str| {
                s.effects.sky_lod = if v == "off" { Some(1) } else { v.parse().ok() };
            }))),
            item!("mips", "texture mip chains", "Effects", Restart, Toggle { default: true }, acc_bool!(effects.mips)),
            item!("h2n", "height-to-normal convert", "Effects", Restart, Toggle { default: true }, acc_bool!(effects.h2n)),
            item!("n2h", "normal-to-height derive", "Effects", Restart, Toggle { default: true }, acc_bool!(effects.n2h)),
            item!("tinted_shadows", "tinted glass shadows", "Effects", Restart, Toggle { default: true }, acc_bool!(effects.tinted_shadows)),
            item!("spray", "spray reclassification", "Effects", Restart, Toggle { default: true }, acc_bool!(effects.spray)),
            item!("depth_tint", "water depth tint", "Effects", Restart, Toggle { default: true }, acc_bool!(effects.depth_tint)),
            item!("water", "water material class", "Effects", Restart, Toggle { default: true }, acc_bool!(effects.water)),
            // ── Scene
            item!("world", "world mode (flagless boot)", "Scene", Restart, Toggle { default: true }, acc_bool!(scene.world)),
            item!("scene_path", "scene path", "Scene", Restart, Text, acc_str!(scene.scene_path)),
            // ── Advanced
            item!("bvh_builder", "BVH builder", "Advanced", Restart, Cycle { options: &["sah", "lbvh", "ploc", "som"], default_ix: 0 }, acc_str!(advanced.bvh_builder)),
            item!("bvh_ctrav", "BVH SAH traversal cost", "Advanced", Restart, StepF { min: 0.0, max: 8.0, step: 0.5, default: 3.0 }, acc_f32!(advanced.bvh_ctrav)),
            item!("bvh_maxleaf", "BVH max leaf tris", "Advanced", Restart, StepU { min: 2, max: 32, step: 2, default: 8 }, acc_u32!(advanced.bvh_maxleaf)),
            item!("bvh_axes", "BVH SAH axes", "Advanced", Restart, Cycle { options: &["1", "3"], default_ix: 1 }, acc_u32!(advanced.bvh_axes)),
            item!("blas_split", "BLAS split (tris; off = one BLAS)", "Advanced", Restart, Cycle { options: &["off", "64", "4096", "65536", "262144"], default_ix: 3 }, ((|s: &Settings| s.advanced.blas_split.map(|v| if v == 0 { "off".into() } else { v.to_string() })), (|s: &mut Settings, v: &str| {
                s.advanced.blas_split = if v == "off" { Some(0) } else { v.parse().ok() };
            }))),
            item!("ftree", "8-wide frustum tree", "Advanced", Restart, Toggle { default: true }, acc_bool!(advanced.ftree)),
            item!("ftree_tiles", "wide tree for CPU tiles", "Advanced", Restart, Toggle { default: false }, acc_bool!(advanced.ftree_tiles)),
            item!("temporal", "temporal reuse", "Advanced", Restart, Toggle { default: true }, acc_bool!(advanced.temporal)),
            item!("replay", "structure replay", "Advanced", Restart, Toggle { default: true }, acc_bool!(advanced.replay)),
            item!("adopt", "query skip / cut adoption", "Advanced", Restart, Toggle { default: true }, acc_bool!(advanced.adopt)),
            item!("discard_seeds", "discard temporal seeds (A/B)", "Advanced", Restart, Toggle { default: false }, acc_bool!(advanced.discard_seeds)),
            item!("defer_shade", "deferred shading (A/B)", "Advanced", Restart, Toggle { default: false }, acc_bool!(advanced.defer_shade)),
            item!("hemi_share", "hemi sharing", "Advanced", Restart, Toggle { default: true }, acc_bool!(advanced.hemi_share)),
            item!("cut_rays", "cut-seeded rays", "Advanced", Restart, Toggle { default: true }, acc_bool!(advanced.cut_rays)),
            item!("cut_hemi", "cut-seeded hemi rays", "Advanced", Restart, Toggle { default: false }, acc_bool!(advanced.cut_hemi)),
            item!("bc7", "BC7 texture compression", "Advanced", Restart, Toggle { default: true }, acc_bool!(advanced.bc7)),
            item!("bc7_quality", "BC7 quality", "Advanced", Restart, Cycle { options: &["ultrafast", "fast", "basic", "slow"], default_ix: 1 }, acc_str!(advanced.bc7_quality)),
            item!("dxr_inline", "DXR dispatch mode (0/1/2)", "Advanced", Restart, Cycle { options: &["0", "1", "2"], default_ix: 1 }, acc_u32!(advanced.dxr_inline)),
            item!("fsr4_required", "require FSR4 (exit if absent)", "Advanced", Restart, Toggle { default: false }, acc_bool!(advanced.fsr4_required)),
            item!("gpu_debug", "D3D12 debug layer", "Advanced", Restart, Toggle { default: false }, acc_bool!(advanced.gpu_debug)),
            item!("pix_markers", "PIX markers", "Advanced", Restart, Toggle { default: false }, acc_bool!(advanced.pix_markers)),
            item!("gpu_timing", "GPU timestamp table", "Advanced", Restart, Toggle { default: false }, acc_bool!(advanced.gpu_timing)),
            item!("sl_path", "Streamline DLL dir", "Advanced", Restart, Text, acc_str!(advanced.sl_path)),
            item!("oidn_path", "OIDN DLL dir", "Advanced", Restart, Text, acc_str!(advanced.oidn_path)),
            item!("nppd_path", "ONNX Runtime DLL dir", "Advanced", Restart, Text, acc_str!(advanced.nppd_path)),
            item!("nppd_model", "NPPD model path", "Advanced", Restart, Text, acc_str!(advanced.nppd_model)),
            item!("xess_path", "XeSS DLL dir", "Advanced", Restart, Text, acc_str!(advanced.xess_path)),
            item!("ffx_path", "FidelityFX DLL dir", "Advanced", Restart, Text, acc_str!(advanced.ffx_path)),
            item!("dxc_path", "DXC DLL dir", "Advanced", Restart, Text, acc_str!(advanced.dxc_path)),
            item!("pix_path", "PIX runtime dir", "Advanced", Restart, Text, acc_str!(advanced.pix_path)),
        ]
    })
}

/// The session's LIVE state, snapshotted by main.rs each menu-visible frame —
/// what Live-tier rows display, and the base current-value for their adjusts.
#[derive(Clone, Copy, Default)]
pub struct LiveView {
    /// 0 = cpu, 1 = gpu wavefront, 2 = dxr.
    pub mode: u8,
    pub hybrid: bool,
    pub dynamic: bool,
    pub overlay: bool,
    pub gpu_tone: bool,
    pub preset: u32,
    pub spp: u32,
    /// 0 off / 1 AO / 2 GI.
    pub bounce: u32,
    pub height_armed: bool,
    pub height_on: bool,
    pub dlss: bool,
    pub xess: bool,
    pub fsr: bool,
    /// 0 off / 1 on (pre) / 2 post.
    pub oidn: u8,
    pub oidn_temporal: bool,
    pub nppd: bool,
    pub tod: f32,
    pub hud: bool,
    pub bloom: bool,
    pub clouds: bool,
    pub fireflies: bool,
    pub fireflies_count: u32,
}

/// What a Live-tier adjust does to the session — main.rs maps these onto the
/// exact key-handler paths (`Edges` synthesis / shared atomics), so reset
/// semantics cannot drift from the keys.
pub enum MenuFx {
    /// Restart-tier: persisted; applies next launch.
    Restart,
    /// Synthesize this frame's Edges field (one key press).
    CycleMode,
    ToggleHybrid,
    ToggleDynamic,
    ToggleOverlay,
    ToggleGpuTone,
    ToggleDlss,
    ToggleXess,
    ToggleFsr,
    ToggleOidn,
    ToggleOidnTemporal,
    ToggleNppd,
    ToggleBounce,
    ToggleHeight,
    CycleSpp,
    Quality(u32),
    SetTod(f32),
    ToggleBloom,
    ToggleClouds,
    ToggleFireflies,
    FirefliesCount(u32),
    ToggleHud,
    None,
}

/// The value string a row displays.
pub fn menu_value(item: &MenuItem, s: &Settings, live: &LiveView) -> String {
    if item.tier == Tier::Live {
        return match item.id {
            "hud" => onoff(live.hud),
            "overlay" => onoff(live.overlay),
            "gpu_tone" => onoff(live.gpu_tone),
            "mode" => ["cpu", "gpu", "dxr"][live.mode.min(2) as usize].into(),
            "preset" => live.preset.to_string(),
            "spp" => live.spp.to_string(),
            "bounce" => ["off", "ao", "gi"][live.bounce.min(2) as usize].into(),
            "hybrid" => onoff(live.hybrid),
            "dynamic" => onoff(live.dynamic),
            "height_on" => {
                if live.height_armed { onoff(live.height_on) } else { "unarmed".into() }
            }
            "dlss" => onoff(live.dlss),
            "xess" => onoff(live.xess),
            "fsr" => onoff(live.fsr),
            "oidn" => ["off", "on", "post"][live.oidn.min(2) as usize].into(),
            "oidn_temporal" => onoff(live.oidn_temporal),
            "nppd" => onoff(live.nppd),
            "tod" => format!("{:02}:{:02}", live.tod as u32 % 24, (live.tod.fract() * 60.0) as u32),
            "bloom" => onoff(live.bloom),
            "clouds" => onoff(live.clouds),
            "fireflies" => onoff(live.fireflies),
            "fireflies_count" => live.fireflies_count.to_string(),
            _ => "?".into(),
        };
    }
    // Restart tier: the persisted value, or the compiled default.
    match (item.get)(s) {
        Some(v) => v,
        None => match &item.control {
            Control::Toggle { default } => format!("{} (default)", onoff(*default)),
            Control::Cycle { options, default_ix } => format!("{} (default)", options[*default_ix]),
            Control::StepU { default, .. } => format!("{default} (default)"),
            Control::StepF { default, .. } => format!("{default} (default)"),
            _ => "(default)".into(),
        },
    }
}

fn onoff(v: bool) -> String {
    if v { "on" } else { "off" }.to_string()
}

/// Apply one click/step to a row: persist the new value into `s` (Live rows
/// persist too — a menu click IS a preference, unlike a key press) and return
/// the live action for main.rs. `dir` is ±1 (Toggle ignores it).
pub fn menu_adjust(item: &MenuItem, dir: i32, s: &mut Settings, live: &LiveView) -> MenuFx {
    if item.tier == Tier::Live {
        return match item.id {
            "hud" => {
                (item.set)(s, &onoff(!live.hud));
                MenuFx::ToggleHud
            }
            "overlay" => MenuFx::ToggleOverlay,
            "gpu_tone" => MenuFx::ToggleGpuTone,
            "mode" => {
                // One SPACE press; the landing mode depends on availability,
                // so nothing is persisted here (mode is persisted only via
                // the file's renderer.mode, which advanced users set).
                MenuFx::CycleMode
            }
            "preset" => {
                let n = (live.preset as i32 + dir).clamp(1, 3) as u32;
                if n == live.preset {
                    return MenuFx::None;
                }
                (item.set)(s, &n.to_string());
                MenuFx::Quality(n)
            }
            "spp" => {
                // One U press (the cycle wraps at the top like the key).
                MenuFx::CycleSpp
            }
            "bounce" => MenuFx::ToggleBounce,
            "hybrid" => MenuFx::ToggleHybrid,
            "dynamic" => MenuFx::ToggleDynamic,
            "height_on" => {
                if live.height_armed { MenuFx::ToggleHeight } else { MenuFx::None }
            }
            "dlss" => MenuFx::ToggleDlss,
            "xess" => MenuFx::ToggleXess,
            "fsr" => MenuFx::ToggleFsr,
            "oidn" => {
                (item.set)(s, &onoff(live.oidn == 0));
                MenuFx::ToggleOidn
            }
            "oidn_temporal" => {
                (item.set)(s, &onoff(!live.oidn_temporal));
                MenuFx::ToggleOidnTemporal
            }
            "nppd" => {
                (item.set)(s, &onoff(!live.nppd));
                MenuFx::ToggleNppd
            }
            "tod" => {
                let step = match &item.control {
                    Control::StepF { step, .. } => *step,
                    _ => 0.5,
                };
                let t = (live.tod + dir as f32 * step).rem_euclid(24.0);
                (item.set)(s, &format!("{t}"));
                MenuFx::SetTod(t)
            }
            "bloom" => {
                (item.set)(s, &onoff(!live.bloom));
                MenuFx::ToggleBloom
            }
            "clouds" => {
                (item.set)(s, &onoff(!live.clouds));
                MenuFx::ToggleClouds
            }
            "fireflies" => {
                (item.set)(s, &onoff(!live.fireflies));
                MenuFx::ToggleFireflies
            }
            "fireflies_count" => {
                let (min, max, step) = match &item.control {
                    Control::StepU { min, max, step, .. } => (*min, *max, *step),
                    _ => (8, 64, 8),
                };
                let n = (live.fireflies_count as i64 + dir as i64 * step as i64)
                    .clamp(min as i64, max as i64) as u32;
                if n == live.fireflies_count {
                    return MenuFx::None;
                }
                (item.set)(s, &n.to_string());
                MenuFx::FirefliesCount(n)
            }
            _ => MenuFx::None,
        };
    }
    // Restart tier: pure Settings edit through the row's control semantics.
    match &item.control {
        Control::Toggle { default } => {
            let cur = (item.get)(s).map(|v| v == "on").unwrap_or(*default);
            (item.set)(s, &onoff(!cur));
        }
        Control::Cycle { options, default_ix } => {
            let cur = (item.get)(s)
                .and_then(|v| options.iter().position(|o| *o == v))
                .unwrap_or(*default_ix);
            let next = (cur as i32 + dir).rem_euclid(options.len() as i32) as usize;
            (item.set)(s, options[next]);
        }
        Control::StepU { min, max, step, default } => {
            let cur = (item.get)(s).and_then(|v| v.parse::<u32>().ok()).unwrap_or(*default);
            let n = (cur as i64 + dir as i64 * *step as i64).clamp(*min as i64, *max as i64) as u32;
            (item.set)(s, &n.to_string());
        }
        Control::StepF { min, max, step, default } => {
            let cur = (item.get)(s).and_then(|v| v.parse::<f32>().ok()).unwrap_or(*default);
            let n = (cur + dir as f32 * step).clamp(*min, *max);
            (item.set)(s, &format!("{n}"));
        }
        _ => {}
    }
    MenuFx::Restart
}

/// Free-text commit (Text rows — paths, scene). Restart-tier by construction.
pub fn menu_text_edit(item: &MenuItem, value: &str, s: &mut Settings) -> MenuFx {
    if matches!(item.control, Control::Text) {
        (item.set)(s, value.trim());
        return MenuFx::Restart;
    }
    MenuFx::None
}

pub fn item_by_id(id: &str) -> Option<&'static MenuItem> {
    menu_items().iter().find(|i| i.id == id)
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
    full.upscaler.fg = Some(false);
    full.effects.fireflies_count = Some(24);
    full.effects.cloud_shadow = Some(0);
    full.effects.sky_lod = Some(8);
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

    // Menu descriptor invariants: unique ids, known groups, valid cycle
    // defaults, and every restart-tier Cycle vocabulary accepted by the SAME
    // consumer the startup apply path uses — the menu cannot offer an option
    // the file loader would warn-and-ignore.
    let items = menu_items();
    let mut seen = std::collections::HashSet::new();
    for it in items {
        if !seen.insert(it.id) {
            return Err(format!("menu id '{}' duplicated", it.id));
        }
        if !GROUPS.contains(&it.group) {
            return Err(format!("menu id '{}' names unknown group '{}'", it.id, it.group));
        }
        if let Control::Cycle { options, default_ix } = &it.control {
            if *default_ix >= options.len() {
                return Err(format!("menu id '{}' default_ix out of range", it.id));
            }
        }
        let ok = match (it.id, &it.control) {
            ("chain", Control::Cycle { options, .. }) => {
                options.iter().all(|o| parse_chain(o).is_some())
            }
            ("lock_res", Control::Cycle { options, .. }) => {
                options.iter().all(|o| *o == "dynamic" || crate::xess::lock_scale(o).is_some())
            }
            ("prefer", Control::Cycle { options, .. }) => {
                options.iter().all(|o| parse_prefer(o).is_some())
            }
            ("oidn_device", Control::Cycle { options, .. }) => {
                options.iter().all(|o| parse_oidn_device(o).is_some())
            }
            ("oidn_quality", Control::Cycle { options, .. }) => {
                options.iter().all(|o| parse_oidn_quality(o).is_some())
            }
            ("nppd_device", Control::Cycle { options, .. }) => {
                options.iter().all(|o| parse_nppd_device(o).is_some())
            }
            ("bc7_quality", Control::Cycle { options, .. }) => {
                options.iter().all(|o| crate::bc7::Quality::parse(o).is_some())
            }
            ("bvh_builder", Control::Cycle { options, .. }) => {
                options.iter().all(|o| matches!(*o, "sah" | "lbvh" | "ploc" | "som"))
            }
            // The cloud-cache Cycles map "off" ↔ 0/1 and otherwise carry a
            // value apply_globals accepts (the vocab-vs-validation drift guard).
            ("cloud_shadow", Control::Cycle { options, .. }) => options
                .iter()
                .all(|o| *o == "off" || o.parse::<u32>().is_ok_and(|n| (2..=64).contains(&n))),
            ("sky_lod", Control::Cycle { options, .. }) => options.iter().all(|o| {
                *o == "off" || o.parse::<u32>().is_ok_and(|k| k.is_power_of_two() && (2..=32).contains(&k))
            }),
            _ => true,
        };
        if !ok {
            return Err(format!("menu id '{}' offers an option its consumer rejects", it.id));
        }
    }
    // A restart Toggle round-trips through its accessors: adjust flips from
    // the compiled default, a second adjust flips back.
    let vsync = item_by_id("vsync").ok_or("menu item 'vsync' missing")?;
    let mut probe = Settings::default();
    let live = LiveView::default();
    if !matches!(menu_adjust(vsync, 1, &mut probe, &live), MenuFx::Restart) {
        return Err("restart-tier adjust did not report Restart".into());
    }
    if probe.display.vsync != Some(false) {
        return Err("vsync adjust from default(true) should persist Some(false)".into());
    }
    menu_adjust(vsync, 1, &mut probe, &live);
    if probe.display.vsync != Some(true) {
        return Err("second vsync adjust should flip back to Some(true)".into());
    }

    // The headless predicate: gates must never see the file.
    for probe in [
        vec!["--check"],
        vec!["--check-gpu", "--stress"],
        vec!["--dlss-dump"],
        vec!["--spin", "path"],
        vec!["--cinematic", "tour"],
        vec!["--cinematic-samples", "64"],
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
