//! Session vocabulary: the small enums and constants that the command line,
//! the settings file and the gates talk about, which have no backend content
//! at all.
//!
//! These lived in `gpu/adapter.rs`, `gpu/dual.rs`, `gpu/d3d12.rs` and
//! `oidn.rs` — every one of them a `#[cfg(windows)]` module — while their only
//! non-backend consumers (`cli.rs`, `settings.rs`) are compiled on every
//! platform. That is what broke the Linux build: `cli.rs` needs to PARSE
//! `--prefer-intel` long before anything opens a device, and parsing an enum
//! is not a D3D12 operation. Each type keeps its old spelling via a `pub use`
//! at the old site, so the backend code reads exactly as it did.

/// Which vendor's adapter to prefer when several are present (`--prefer-nvidia`
/// / `--prefer-amd` / `--prefer-intel`; `--fsr` flips the default to AMD).
///
/// A PREFERENCE, never a requirement: a box without that vendor falls back to
/// the first hardware adapter, and the caller's own feature probe is the real
/// gate. This is exactly why `main::vendor_defaults` keys off the PICKED
/// adapter's vendor — a fact — and never off this — a request.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prefer {
    Nvidia,
    Amd,
    Intel,
}

/// Which tracer the SECONDARY adapter runs under `--dual-gpu` (`--dual-gpu-arm
/// wave|dxr`). The primary's arm is the session's; the secondary's follows its
/// own adapter's vendor unless this forces it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arm {
    Wave,
    Dxr,
}

impl Arm {
    /// For the announce lines and the gate reports.
    pub fn name(self) -> &'static str {
        match self {
            Arm::Wave => "wavefront",
            Arm::Dxr => "DXR",
        }
    }
}

/// What the swapchain's bytes MEAN — one 10-bit format, two transfer curves,
/// plus the legacy 8-bit rung. Decided once at swapchain creation and read by
/// every encode site, so an arm cannot present the wrong wire.
///
/// A RUNTIME FACT, never the CLI flag: the HDR10 declare can be refused and
/// the frame-generation swapchain wrap can force a rebuild at a lower rung, so
/// callers read the negotiated value (`D3d::space`) rather than what was asked
/// for. What varies with it is the tonemap encode (`tone::ToneMode`), the CPU
/// blit format, and nothing upstream of the present.
///
/// The mapping to an actual backend format is deliberately NOT here (a
/// `DXGI_FORMAT` may not appear in `gfx/`); it is an `impl` in the backend —
/// see `gpu::d3d12::PresentSpace::format`. Vulkan's twin will be the same
/// `impl` shape over `VK_FORMAT_A2B10G10R10_UNORM_PACK32`, whose channel order
/// already matches what `render::present_px_sdr10` packs.
///
/// Unlike its neighbours here — which `cli.rs` and `settings.rs` parse on every
/// platform — this one's consumers are both present paths (`gpu::d3d12`, and
/// `CpuPresent`'s staging wires), so today it has no caller off Windows. The
/// Vulkan swapchain in M4 is the second consumer that retires the attribute.
#[cfg_attr(not(windows), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PresentSpace {
    /// 8-bit `B8G8R8A8_UNORM`, display-encoded (gamma 2.2 is ours to apply).
    /// The `--no-hdr` lever and the ladders' last rung.
    Sdr,
    /// 10-bit `R10G10B10A2_UNORM`, NO colour space declared — DXGI reads a
    /// UNORM chain as gamma-2.2/`G22_NONE_P709` by default, so this is
    /// deep-colour SDR: the same display-encoded image as `Sdr` at 10-bit
    /// tonal resolution. The HDR-off-display default.
    Sdr10,
    /// 10-bit PQ — `R10G10B10A2_UNORM`, `G2084_NONE_P2020`. The
    /// HDR-on-display default.
    Hdr10,
}

/// OIDN's RT-filter quality tier (`--oidn-quality fast|balanced|high`).
///
/// The VALUES are OIDN's own ABI (`OIDNQuality`), which is why they are bare
/// `i32` rather than an enum: `oidn.rs` passes them straight to
/// `oidnSetFilterInt`, and inventing a Rust enum here would add a mapping that
/// could drift from the header. They live in `gfx` because `settings.rs`
/// round-trips them through the settings file on every platform, while
/// `oidn.rs` is Windows-only today.
pub const OIDN_QUALITY_FAST: i32 = 4;
pub const OIDN_QUALITY_BALANCED: i32 = 5;
pub const OIDN_QUALITY_HIGH: i32 = 6;
