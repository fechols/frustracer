//! What the display can actually do — the only place the renderer learns real
//! luminance numbers, and the input to `tone::ToneParams::hdr`.
//!
//! Everything upstream of the fullscreen pass is scene-referred: unitless linear
//! radiance where 1.0 ≈ diffuse white. Turning that into light requires two
//! absolute numbers the scene cannot know — where paper white sits (a taste
//! setting, `--hdr-paper-white`) and how far above it the panel can go (a fact,
//! probed here). Without the probe the highlight rolloff would be aimed at a
//! guess, and would clip or crush on any display that disagreed.
//!
//! We deliberately do **not** use `IDXGISwapChain::GetContainingOutput`: under
//! DLSS the swapchain is a Streamline proxy, and whether SL forwards that call
//! is not a question worth depending on. `MonitorFromWindow` on the HWND we
//! already own, matched against `DXGI_OUTPUT_DESC1::Monitor`, sidesteps it.

use crate::tone::ToneParams;
use windows::core::Interface;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Dxgi::Common::DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
use windows::Win32::Graphics::Dxgi::{IDXGIAdapter4, IDXGIOutput6, DXGI_OUTPUT_DESC1};
use windows::Win32::Graphics::Gdi::{MonitorFromWindow, MONITOR_DEFAULTTONEAREST};

/// What the monitor the window currently sits on reports.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DisplayHdr {
    /// Windows HDR is switched on for this output. When false the output is
    /// SDR and no amount of scRGB will produce a highlight.
    pub enabled: bool,
    /// Peak luminance in nits (`MaxLuminance`) — a small-window highlight peak.
    pub max_nits: f32,
    /// Sustained full-screen-white luminance in nits (`MaxFullFrameLuminance`).
    /// Usually far below `max_nits` on OLED and edge-lit panels.
    pub max_full_frame_nits: f32,
}

impl DisplayHdr {
    /// What we get when the probe finds nothing (no matching output, a failed
    /// GetDesc1). SDR — the safe answer, never a guessed HDR peak.
    pub const SDR: DisplayHdr =
        DisplayHdr { enabled: false, max_nits: 80.0, max_full_frame_nits: 80.0 };

    /// The presentation curve for this display.
    ///
    /// `peak_override` is `--hdr-peak`, and it **wins over the probe** — including
    /// over `enabled == false`. That is the point of the lever: the probe can be
    /// wrong (an odd output enumeration, a driver that doesn't report G2084 on a
    /// panel that is plainly HDR), and an override that silently did nothing in
    /// exactly that case would be no escape hatch at all.
    ///
    /// Without an override we aim the rolloff at `max_nits`: the curve only ever
    /// *approaches* its asymptote, and the values that reach up there are small
    /// highlights — exactly the regime `MaxLuminance` describes. Aiming at
    /// `max_full_frame_nits` instead would throw away most of the headroom the
    /// panel has for precisely this content.
    pub fn tone(&self, paper_white: f32, peak_override: Option<f32>) -> ToneParams {
        match peak_override {
            Some(peak) => ToneParams::hdr(paper_white, peak),
            // scRGB on an SDR output: same rolloff as the 8-bit path, linear-
            // encoded at scRGB's own 80-nit white. Displays identically to SDR.
            None if !self.enabled => ToneParams::SCRGB_SDR,
            None => ToneParams::hdr(paper_white, self.max_nits),
        }
    }

    fn from_desc(d: &DXGI_OUTPUT_DESC1) -> DisplayHdr {
        // A non-finite luminance would ride straight into `ToneParams::hdr` and
        // come out as a NaN headroom — `NaN <= HDR_KNEE` is false, so the
        // degeneracy guard there does NOT catch it, and every pixel above the
        // knee would map to NaN (a black frame, silently). Zero is fine and IS
        // handled downstream (headroom 0 degenerates to the SDR rolloff); only
        // NaN/inf need rejecting, and the honest answer for a display that
        // cannot describe itself is SDR.
        let sane = |v: f32| if v.is_finite() && v >= 0.0 { Some(v) } else { None };
        let (Some(max_nits), Some(max_full_frame_nits)) =
            (sane(d.MaxLuminance), sane(d.MaxFullFrameLuminance))
        else {
            eprintln!(
                "hdr: display reported a non-finite peak luminance ({}, full-frame {}) — treating as SDR",
                d.MaxLuminance, d.MaxFullFrameLuminance
            );
            return DisplayHdr::SDR;
        };
        DisplayHdr {
            // G2084 (PQ) is how Windows reports "HDR is on for this output",
            // regardless of the colour space *we* present in — an scRGB
            // swapchain on an HDR output still reads back as G2084 here.
            enabled: d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
            max_nits,
            max_full_frame_nits,
        }
    }
}

/// Probe the output the window currently sits on. Never fails — an unprobeable
/// display is reported SDR, which costs a highlight, not a frame.
pub fn probe(adapter: &IDXGIAdapter4, hwnd: HWND) -> DisplayHdr {
    let mon = unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) };
    for i in 0.. {
        let Ok(out) = (unsafe { adapter.EnumOutputs(i) }) else {
            break; // DXGI_ERROR_NOT_FOUND — end of the list
        };
        let Ok(out6) = out.cast::<IDXGIOutput6>() else {
            continue;
        };
        let Ok(desc) = (unsafe { out6.GetDesc1() }) else {
            continue;
        };
        if desc.Monitor == mon {
            return DisplayHdr::from_desc(&desc);
        }
    }
    DisplayHdr::SDR
}
