//! What the display can actually do — the only place the renderer learns real
//! luminance numbers, and the input to `tone::ToneParams::hdr10`.
//!
//! Everything upstream of the fullscreen pass is scene-referred: unitless linear
//! radiance where 1.0 ≈ diffuse white. Turning that into light requires two
//! absolute numbers the scene cannot know — where paper white sits (a taste
//! setting, `--hdr-paper-white`) and how far above it the panel can go (a fact,
//! probed here). Without the probe the highlight rolloff would be aimed at a
//! guess, and would clip or crush on any display that disagreed.
//!
//! We deliberately do **not** use `IDXGISwapChain::GetContainingOutput`: an
//! FG-family session's swapchain is a wrapper proxy (ffx FI / XeSS-FG), and
//! whether a proxy forwards that call is not a question worth depending on.
//! `MonitorFromWindow` on the HWND we already own, matched against
//! `DXGI_OUTPUT_DESC1::Monitor`, sidesteps it.
//!
//! **The monitor need not hang off the adapter we render on.** A display is
//! enumerated by the adapter it is PHYSICALLY PLUGGED INTO, and on a
//! multi-GPU box those are routinely different devices: render on the AMD card
//! (what `--fsr`/`--fsr4` prefers) with the panel cabled to the NVIDIA one, and
//! the AMD adapter's output list does not contain that monitor at all. DWM
//! cross-adapter-copies the presented frame to the display, so presentation
//! works fine — but a probe that searched only the render adapter found
//! nothing, returned `SDR`, and handed an HDR panel the flat SDR rolloff:
//! driven at paper white with zero highlight headroom. So the search falls
//! THROUGH to every adapter on the box. Reading
//! `GetDesc1` off the adapter that owns the output is not a workaround — that
//! adapter is the authority on that display, and the luminance numbers are a
//! property of the panel, not of whatever device we happen to be drawing with.

use crate::tone::ToneParams;
use windows::core::Interface;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Dxgi::Common::DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
use windows::Win32::Graphics::Dxgi::{
    IDXGIAdapter4, IDXGIFactory6, IDXGIOutput6, DXGI_ADAPTER_FLAG3_NONE,
    DXGI_ADAPTER_FLAG3_SOFTWARE, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, DXGI_OUTPUT_DESC1,
};
use windows::Win32::Graphics::Gdi::{MonitorFromWindow, HMONITOR, MONITOR_DEFAULTTONEAREST};

/// What the monitor the window currently sits on reports.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DisplayHdr {
    /// Windows HDR is switched on for this output. When false the output is
    /// SDR and no swapchain encoding will produce a highlight.
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

    /// The presentation curve for the HDR10/PQ swapchain (the HDR-on-display
    /// default + `--hdr10`).
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
    ///
    /// HDR-off gets the SDR rolloff anchored at paper white
    /// (`hdr10(paper, paper)` is the degenerate knee-0 curve, PQ-encoded), so
    /// a `--hdr10` session on an SDR output displays like SDR through DWM's
    /// own PQ handling rather than blowing highlights. (The Sdr/Sdr10 arms use
    /// `ToneParams::SDR` directly and never call here — a gamma encode has no
    /// peak to retune.)
    pub fn tone_pq(&self, paper_white: f32, peak_override: Option<f32>) -> ToneParams {
        match peak_override {
            Some(peak) => ToneParams::hdr10(paper_white, peak),
            None if !self.enabled => ToneParams::hdr10(paper_white, paper_white),
            None => ToneParams::hdr10(paper_white, self.max_nits),
        }
    }

    fn from_desc(d: &DXGI_OUTPUT_DESC1) -> DisplayHdr {
        // A non-finite luminance would ride straight into `ToneParams::hdr10` and
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
            // regardless of the colour space *we* present in — an SDR-declared
            // swapchain on an HDR output still reads back as G2084 here.
            enabled: d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
            max_nits,
            max_full_frame_nits,
        }
    }
}

/// Search ONE adapter's outputs for `mon`. `None` = this adapter does not drive
/// that monitor (the ordinary answer for every adapter but one).
fn search(adapter: &IDXGIAdapter4, mon: HMONITOR) -> Option<DisplayHdr> {
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
            return Some(DisplayHdr::from_desc(&desc));
        }
    }
    None
}

/// A factory for the cross-adapter fallback, cached across calls.
///
/// `probe` runs on a 1 Hz poll (`GpuContext::refresh_display` — the only thing
/// that catches the user toggling Windows HDR on the monitor the window is
/// already sitting on), and on a cross-adapter box the fallback runs EVERY
/// poll, so creating a factory each time would be a needless recurring cost.
/// Staleness is the reason this can't just be created once and kept forever: a
/// DXGI factory does not see displays attached after its creation, which is
/// exactly the event the poll exists to notice — hence the documented
/// `IsCurrent` check, re-creating only when the topology actually moved.
///
/// Main-thread-only by construction (`probe` takes an HWND and is called from
/// `GpuContext`), so a `thread_local` is the right scope and no lock is needed.
fn fallback_factory() -> Option<IDXGIFactory6> {
    use std::cell::RefCell;
    thread_local! {
        static FACTORY: RefCell<Option<IDXGIFactory6>> = const { RefCell::new(None) };
    }
    FACTORY.with(|slot| {
        let mut slot = slot.borrow_mut();
        let stale = match slot.as_ref() {
            Some(f) => !unsafe { f.IsCurrent() }.as_bool(),
            None => true,
        };
        if stale {
            *slot = super::adapter::create_factory(false).ok();
        }
        slot.clone()
    })
}

/// Probe the output the window currently sits on. Never fails — an unprobeable
/// display is reported SDR, which costs a highlight, not a frame.
///
/// `adapter` is the RENDER adapter and is only a fast path: it is searched
/// first because on a single-GPU box (and on any box where the panel is cabled
/// to the card we draw with) it is the whole answer. When it does not own the
/// monitor we search every other adapter — see the module header for why that
/// is the correct question rather than a fallback.
pub fn probe(adapter: &IDXGIAdapter4, hwnd: HWND) -> DisplayHdr {
    let mon = unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) };
    if let Some(d) = search(adapter, mon) {
        return d;
    }
    let Some(factory) = fallback_factory() else {
        note_once(
            "hdr: the render adapter does not drive this monitor and no DXGI factory could be \
             created to find the adapter that does — treating the display as SDR (--hdr-peak \
             overrides)",
        );
        return DisplayHdr::SDR;
    };
    for i in 0.. {
        let Ok(other) = (unsafe {
            factory.EnumAdapterByGpuPreference::<IDXGIAdapter4>(
                i,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            )
        }) else {
            break;
        };
        // A software adapter drives no physical panel; skipping it keeps a
        // WARP/Basic-Render entry from ever answering for a real display.
        let software = unsafe { other.GetDesc3() }
            .map(|d| (d.Flags & DXGI_ADAPTER_FLAG3_SOFTWARE) != DXGI_ADAPTER_FLAG3_NONE)
            .unwrap_or(false);
        if software {
            continue;
        }
        if let Some(d) = search(&other, mon) {
            let name = unsafe { other.GetDesc3() }
                .map(|desc| {
                    let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(128);
                    String::from_utf16_lossy(&desc.Description[..len])
                })
                .unwrap_or_else(|_| "another adapter".into());
            note_once(&format!(
                "hdr: this monitor is driven by {name}, not the render adapter — probing HDR \
                 capability there (cross-adapter present; DWM copies the frame)"
            ));
            return d;
        }
    }
    note_once(
        "hdr: no adapter on this box enumerates the monitor the window is on — treating the \
         display as SDR (--hdr-peak overrides the probe)",
    );
    DisplayHdr::SDR
}

/// One-shot for the lines above. They describe a STANDING property of the box,
/// and `probe` is on a 1 Hz poll — printing per call would be a scrolling wall
/// saying the same thing. Latching per distinct message keeps the loud-fallback
/// discipline (the fact is visible) without the spam.
fn note_once(msg: &str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let mut g = match SEEN.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(), // a poisoned latch must not silence a diagnostic
    };
    if g.get_or_insert_with(HashSet::new).insert(msg.to_string()) {
        eprintln!("{msg}");
    }
}
