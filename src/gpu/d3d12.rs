//! Core D3D12 machinery: device, queue, swapchain, per-frame ring, fence
//! sync, and the small barrier/copy helpers everything else records with.
//!
//! Frame pacing: FRAMES_IN_FLIGHT per-frame slots {command allocator, fence
//! value, upload ranges}. The fence is waited *before touching a slot's
//! upload memory*, not after present — v1 is effectively synchronous (the CPU
//! trace dwarfs GPU work), but CPU/GPU overlap later is a loop reorder, not a
//! rewrite.

use std::mem::ManuallyDrop;
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

pub const FRAMES_IN_FLIGHT: usize = 2;
pub const BACKBUFFERS: u32 = 3;
/// Backbuffer count for sessions that PAIR-PRESENT (raw-NGX DLSS-G: two
/// Presents per rendered frame). The single-present sessions reuse a buffer
/// 3 presents apart; at 3 buffers a pair-presenting session reuses one only
/// 1.5 FRAMES apart — and under vsync with the DXGI present queue full
/// (max latency 3), that is re-rendering into a buffer still QUEUED FOR
/// SCANOUT: stale frames flicker through, no debug layer objects (it is a
/// timing race, not an API error). 6 restores the exact
/// 3-buffers-per-present ratio the shipped path has always had. quinlight's
/// own pair-present ran its second present through a dedicated fence ring
/// (PAIR_PRESENT_FENCES) — the port dropped that; this is the counting-safe
/// equivalent.
pub const PAIR_BACKBUFFERS: u32 = 6;

/// The SDR swapchain format — 8-bit, display-encoded (the tonemap PS and
/// `render::present_px` apply the gamma; there is no hardware sRGB encode).
pub const SWAPCHAIN_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;

/// The scRGB HDR swapchain format: linear, Rec.709 primaries, 1.0 = 80 nits,
/// values above 1.0 legal. Paired with `DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709`
/// via `SetColorSpace1`. See `crate::tone`.
pub const SWAPCHAIN_FORMAT_HDR: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

/// The HDR10 swapchain format: 10-bit PQ-encoded, Rec.2020 primaries, paired
/// with `DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020`. This is the arm the
/// swapchain-wrapper FG families take (DLSS-G and XeSS-FG both reject the
/// scRGB fp16 swapchain; 10-bit PQ is the format HDR FG titles actually ship),
/// plus the `--hdr10` A/B lever. See `tone::ToneMode::Pq`.
pub const SWAPCHAIN_FORMAT_HDR10: DXGI_FORMAT = DXGI_FORMAT_R10G10B10A2_UNORM;

/// The format the CPU-present blit texture is uploaded in. **Deliberately its
/// own constant, not `SWAPCHAIN_FORMAT`**: `BlitUpload` packs pixels as
/// `u32 0x00RRGGBB`, whose little-endian byte order is B,G,R,X — a layout that
/// is only valid for B8G8R8A8. If it followed the swapchain format it would
/// silently reinterpret those bytes as f16 under `--hdr`. The HDR blit path has
/// its own f16 texture (`BlitUpload::new_hdr`) instead.
pub const BLIT_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;

/// scRGB f16 for the CPU-present blit under `--hdr` — `render::present_px_scrgb`
/// has already applied the curve and the encode, so the blit PS stays a
/// passthrough.
pub const BLIT_FORMAT_HDR: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

/// 10-bit PQ for the CPU-present blit under HDR10 — `render::present_px_pq`
/// owns the whole encode (curve + matrix + ST 2084 + the 10-bit pack), so the
/// blit PS stays a passthrough here too. Its own const per the `BLIT_FORMAT`
/// doctrine: the packed `u32` lane order (R low) is only valid for R10G10B10A2.
pub const BLIT_FORMAT_HDR10: DXGI_FORMAT = DXGI_FORMAT_R10G10B10A2_UNORM;

/// Which color space presentation was actually negotiated in. Runtime fact,
/// never the CLI flag — the scRGB/HDR10 declares can be refused and the FG
/// wrap can force a rebuild, so callers must read `D3d::space`. What varies
/// with it is the tonemap encode (`tone::ToneMode`), the CPU blit format, and
/// nothing upstream of the present.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PresentSpace {
    /// 8-bit `B8G8R8A8_UNORM`, display-encoded (gamma 2.2 is ours to apply).
    Sdr,
    /// f16 scRGB — linear, `G10_NONE_P709`, 1.0 = 80 nits.
    Scrgb,
    /// 10-bit PQ — `R10G10B10A2_UNORM`, `G2084_NONE_P2020`.
    Hdr10,
}

impl PresentSpace {
    pub fn format(self) -> DXGI_FORMAT {
        match self {
            PresentSpace::Sdr => SWAPCHAIN_FORMAT,
            PresentSpace::Scrgb => SWAPCHAIN_FORMAT_HDR,
            PresentSpace::Hdr10 => SWAPCHAIN_FORMAT_HDR10,
        }
    }
}

pub type Result<T> = std::result::Result<T, String>;

/// Fallible `OnceLock` get-or-init for lazily allocated GPU buffers
/// (interior mutability so recording paths keep `&self`). Two racing
/// initializers may both construct; the loser's value is dropped by the
/// discarded `set` — benign, both then return the winner's buffer.
pub fn get_or_try_init<T>(
    lock: &std::sync::OnceLock<T>,
    init: impl FnOnce() -> Result<T>,
) -> Result<&T> {
    if lock.get().is_none() {
        let _ = lock.set(init()?);
    }
    Ok(lock.get().unwrap())
}

fn err<T>(ctx: &str, e: windows::core::Error) -> Result<T> {
    Err(format!("{ctx}: {e}"))
}

/// Declare the swapchain's color space (scRGB G10 or HDR10 G2084 — `Sdr` is
/// DXGI's default reading and declares nothing). `CheckColorSpaceSupport`
/// first: `SetColorSpace1` on an unsupported space is an error, and we want to
/// know *before* committing to the buffer so the caller can rebuild at SDR.
///
/// Idempotent and cheap — the display-change retune re-runs it, since a swapchain
/// that moves to a different output can in principle need re-declaring.
fn declare_colorspace(swapchain: &IDXGISwapChain3, space: PresentSpace) -> Result<()> {
    let (cs, name) = match space {
        PresentSpace::Sdr => return Ok(()),
        PresentSpace::Scrgb => (DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709, "G10_NONE_P709"),
        PresentSpace::Hdr10 => (DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, "G2084_NONE_P2020"),
    };
    let support = unsafe { swapchain.CheckColorSpaceSupport(cs) }
        .map_err(|e| format!("CheckColorSpaceSupport: {e}"))?;
    if support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT.0 as u32 == 0 {
        return Err(format!("{name} not present-supported"));
    }
    unsafe { swapchain.SetColorSpace1(cs) }.map_err(|e| format!("SetColorSpace1({name}): {e}"))
}

pub struct FrameSlot {
    pub allocator: ID3D12CommandAllocator,
    pub fence_value: u64,
}

pub struct D3d {
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    /// Declared before `swapchain`: the buffer refs pin the native swapchain,
    /// so they must release first (fields drop in declaration order) or SL
    /// warns about destroying the proxy while the native is still referenced.
    pub backbuffers: Vec<ID3D12Resource>,
    pub swapchain: IDXGISwapChain3,
    pub rtv_heap: ID3D12DescriptorHeap,
    pub rtv_size: u32,
    pub list: ID3D12GraphicsCommandList,
    pub slots: Vec<FrameSlot>,
    pub fence: ID3D12Fence,
    pub fence_event: HANDLE,
    pub next_fence: u64,
    pub frame_index: usize,
    pub width: u32,
    pub height: u32,
    /// Swapchain buffer count this chain was created with (BACKBUFFERS, or
    /// PAIR_BACKBUFFERS for pair-presenting raw-NGX FG sessions) — resize
    /// must pass the same count.
    nbuf: u32,
    /// The format the swapchain was actually created with. Runtime, not the
    /// const, because `--hdr` picks scRGB f16 (and the wrapper-FG arm HDR10) —
    /// and because `ResizeBuffers` and both fullscreen PSOs must agree with
    /// whatever was chosen. `space` falls back to `Sdr` when a colour space
    /// was requested but refused (or the FG wrap forced a rebuild), so callers
    /// must read THIS, never the CLI flag.
    pub format: DXGI_FORMAT,
    pub space: PresentSpace,
    /// Present sync interval (1 = v-sync, 0 = uncapped for benchmarking).
    sync_interval: u32,
    /// DXGI_PRESENT_ALLOW_TEARING when the swapchain was created with the
    /// tearing flag (required together — the flag is only legal on such a
    /// swapchain, and only with sync interval 0 in windowed mode).
    present_flags: DXGI_PRESENT,
}

/// Drain the D3D12 debug layer's message queue to stderr.
///
/// The layer writes to `OutputDebugString`, NOT to stdout/stderr — so with a
/// debugger unattached, `--gpu-debug` armed the validation and then threw every
/// finding away. That is not a theoretical gap: a compute shader reading a
/// resource left in PIXEL_SHADER_RESOURCE (the wrong state — compute needs
/// NON_PIXEL) is a debug-layer ERROR that drivers wave through, so it presented
/// perfectly and shipped, twice, because nobody was listening. Call this after
/// every submit under `--gpu-debug` and the layer earns its keep.
///
/// A no-op without the layer (the InfoQueue only exists when it is on).
pub fn drain_debug(device: &ID3D12Device) {
    let Ok(q) = device.cast::<ID3D12InfoQueue>() else {
        return;
    };
    let n = unsafe { q.GetNumStoredMessages() };
    for i in 0..n {
        let mut len = 0usize;
        if unsafe { q.GetMessage(i, None, &mut len) }.is_err() || len == 0 {
            continue;
        }
        let mut buf = vec![0u8; len];
        let msg = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
        if unsafe { q.GetMessage(i, Some(msg), &mut len) }.is_err() {
            continue;
        }
        let m = unsafe { &*msg };
        let sev = match m.Severity {
            D3D12_MESSAGE_SEVERITY_CORRUPTION => "CORRUPTION",
            D3D12_MESSAGE_SEVERITY_ERROR => "ERROR",
            D3D12_MESSAGE_SEVERITY_WARNING => "WARNING",
            _ => continue, // INFO/MESSAGE: pure noise, and voluminous.
        };
        let text = unsafe {
            std::slice::from_raw_parts(m.pDescription as *const u8, m.DescriptionByteLength)
        };
        eprintln!("d3d12 {sev}: {}", String::from_utf8_lossy(text).trim_end_matches('\0'));
    }
    if n > 0 {
        unsafe { q.ClearStoredMessages() };
    }
}

/// Create the native D3D12 device on the picked adapter (debug layer first —
/// it must be enabled before device creation).
pub fn create_device(adapter: &IDXGIAdapter4, debug: bool) -> Result<ID3D12Device> {
    if debug {
        let mut dbg: Option<ID3D12Debug> = None;
        if unsafe { D3D12GetDebugInterface(&mut dbg) }.is_ok() {
            if let Some(d) = dbg {
                unsafe { d.EnableDebugLayer() };
                // GPU-BASED VALIDATION, not just the basic layer. The basic layer
                // checks barrier bookkeeping (before-state must match) but does
                // NOT check the state a resource is actually IN when a shader
                // reads it through a descriptor table — that is a GBV-only check,
                // and it is precisely the class of bug that shipped here (a
                // compute shader sampling a texture left in PIXEL_SHADER_RESOURCE
                // instead of NON_PIXEL_SHADER_RESOURCE). Without GBV, --gpu-debug
                // was quietly blind to the thing it most needed to see.
                //
                // GBV is SLOW (patched shaders, per-dispatch checks). That is fine:
                // this is an opt-in correctness flag, not a benchmark path.
                match d.cast::<ID3D12Debug1>() {
                    Ok(d1) => {
                        unsafe { d1.SetEnableGPUBasedValidation(true) };
                        eprintln!(
                            "gpu: D3D12 debug layer + GPU-based validation enabled \
                             (messages drain to stderr; expect it to be slow)"
                        );
                    }
                    Err(_) => eprintln!(
                        "gpu: D3D12 debug layer enabled, but GPU-based validation is \
                         unavailable (no ID3D12Debug1) — resource-state-at-use is NOT checked"
                    ),
                }
            }
        }
    }
    let mut device: Option<ID3D12Device> = None;
    if let Err(e) = unsafe { D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, &mut device) } {
        return err("D3D12CreateDevice", e);
    }
    Ok(device.unwrap())
}

/// Create a direct command queue. In the Streamline path the `device` passed
/// here is the SL *proxy* device — that hook is what lets DLSS Frame
/// Generation own presentation pacing later.
pub fn create_queue(device: &ID3D12Device) -> Result<ID3D12CommandQueue> {
    unsafe {
        device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })
    }
    .map_err(|e| format!("CreateCommandQueue: {e}"))
}

/// Frame-generation swapchain wrap hook (see `with_queue`). `wrap` receives
/// the present queue and the fully negotiated swapchain (post colour-space
/// fallback) and returns the frame-interpolation proxy the session presents
/// through from then on; Err hands the ORIGINAL swapchain back so a failed
/// wrap degrades to a normal session. `fallback` is the presentation space
/// that normal session would have requested before the FG wrapper forced
/// HDR10/SDR; `with_queue` restores it if every wrap attempt fails. `unwind`
/// destroys whatever FG context the last successful `wrap` created — called
/// only after every proxy reference has been released, when the post-wrap
/// colour-space re-declare fails and the session must rebuild at SDR (a
/// mis-declared present is never acceptable; SDR *with* FG beats it). d3d12.rs
/// stays SDK-agnostic — both closures live with the FG runtime that owns the
/// proxy.
pub struct FgHook<'a> {
    pub fallback: PresentSpace,
    pub wrap: &'a mut dyn FnMut(
        &ID3D12CommandQueue,
        IDXGISwapChain3,
    ) -> std::result::Result<IDXGISwapChain3, (IDXGISwapChain3, String)>,
    pub unwind: &'a mut dyn FnMut(),
}

impl D3d {
    /// Build the swapchain + frame machinery around an existing device/queue
    /// (M3 passes the Streamline proxy queue here so present is hooked).
    pub fn with_queue(
        factory: &IDXGIFactory6,
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        hwnd: HWND,
        width: u32,
        height: u32,
        vsync: bool,
        want: PresentSpace,
        fg_hook: Option<FgHook>,
        pair_present: bool,
    ) -> Result<Self> {
        let nbuf = if pair_present { PAIR_BACKBUFFERS } else { BACKBUFFERS };
        // Uncapped presentation needs DXGI tearing support (windowed flip
        // model otherwise paces on the compositor even at sync interval 0),
        // and the Present flag is only legal on a swapchain created with the
        // matching creation flag.
        let tearing = !vsync && {
            let mut sup = windows::core::BOOL(0);
            unsafe {
                factory.CheckFeatureSupport(
                    DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                    &mut sup as *mut _ as *mut core::ffi::c_void,
                    std::mem::size_of::<windows::core::BOOL>() as u32,
                )
            }
            .is_ok()
                && sup.as_bool()
        };
        if !vsync {
            if tearing {
                eprintln!("present: v-sync off (tearing swapchain, uncapped frame rate)");
            } else {
                eprintln!(
                    "present: v-sync off requested but DXGI tearing is unsupported — \
                     presenting at sync interval 0 (the compositor may still pace frames)"
                );
            }
        }
        let sc_desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: want.format(),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: nbuf,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            Flags: if tearing { DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32 } else { 0 },
            ..Default::default()
        };
        // One creation site, so the IDXGISwapChain1 the factory hands back CANNOT
        // outlive the cast: an HWND may own only one flip-model swapchain, so a
        // stray IDXGISwapChain1 binding still holding the old one is enough to
        // make the SDR-fallback re-create below fail with DXGI_ERROR_INVALID_CALL.
        // Scoping it inside this closure is what guarantees the old swapchain is
        // fully released before a new one is asked for.
        let create = |fmt: DXGI_FORMAT| -> Result<IDXGISwapChain3> {
            let mut d = sc_desc;
            d.Format = fmt;
            let sc1: IDXGISwapChain1 =
                unsafe { factory.CreateSwapChainForHwnd(&queue, hwnd, &d, None, None) }
                    .map_err(|e| format!("CreateSwapChainForHwnd({fmt:?}): {e}"))?;
            sc1.cast().map_err(|e| format!("IDXGISwapChain3 cast: {e}"))
        };
        let mut space = want;
        let mut swapchain = create(space.format())?;

        // scRGB/HDR10: the buffer alone means nothing — DXGI interprets a
        // swapchain as sRGB unless told otherwise, so the colour space
        // declaration is what makes the values mean what we think. If it is
        // refused we fall back to the 8-bit swapchain rather than present into
        // a misinterpreted buffer (which would wash the whole image out). Loud
        // line, full fallback, no degraded half-mode.
        //
        // scRGB is the DEFAULT, not an HDR-only path: handing Windows an f16
        // linear surface is the right thing on any monitor. On an HDR display it
        // carries real highlights; on an SDR one DWM tone-maps/clamps it and can
        // still drive a 10-bit panel at full precision, where the old 8-bit
        // backbuffer would have banded. What varies with the display is the
        // CURVE (crate::tone), not the swapchain. HDR10 is the wrapper-FG arm
        // (+ the --hdr10 lever): the FG proxies reject scRGB fp16 but take
        // 10-bit PQ, the format HDR FG titles ship.
        if space != PresentSpace::Sdr {
            match declare_colorspace(&swapchain, space) {
                Ok(()) => match space {
                    PresentSpace::Scrgb => eprintln!(
                        "present: scRGB swapchain (R16G16B16A16_FLOAT, G10_NONE_P709) — \
                         no 8-bit quantization; Windows drives the panel's real bit depth"
                    ),
                    PresentSpace::Hdr10 => eprintln!(
                        "present: HDR10 swapchain (R10G10B10A2_UNORM, G2084_NONE_P2020, PQ)"
                    ),
                    PresentSpace::Sdr => unreachable!(),
                },
                Err(e) => {
                    eprintln!("present: {space:?} refused ({e}) — 8-bit B8G8R8A8 swapchain");
                    space = PresentSpace::Sdr;
                    // Release the old swapchain BEFORE asking for its replacement:
                    // the HWND still owns it until the last reference dies.
                    drop(swapchain);
                    swapchain = create(space.format())?;
                }
            }
        }

        // Frame generation: wrap the negotiated swapchain with the runtime's
        // frame-interpolation proxy. This must sit AFTER the colour-space
        // fallback (the proxy clones the final desc, and a post-wrap SDR
        // re-create would orphan it) and BEFORE RTV creation (GetBuffer on the
        // proxy returns the PROXY's backbuffers — the real presentation chain
        // lives inside it, and RTVs built from the pre-wrap chain would render
        // into buffers nothing ever presents).
        //
        // Failure ladder, in order of what is worth keeping:
        //  - wrap fails at Hdr10: the proxy itself may be rejecting the 10-bit
        //    format (XeSS-FG's InitFromSwapChain is format-picky). FG is the
        //    reason this session exists, so rebuild at SDR and wrap AGAIN —
        //    SDR with FG beats HDR10 without it.
        //  - wrap fails otherwise: restore the non-FG presentation request
        //    (including its normal SDR fallback) and run without FG.
        //  - wrap succeeds but the colour-space re-declare through the proxy
        //    fails: NEVER present mis-declared. Drop the proxy, `unwind` the
        //    FG context (every proxy ref must be gone before the FG runtime
        //    can destroy it, and the wrap consumed the app chain, so unwind
        //    must run before a new chain can exist on this HWND), rebuild at
        //    SDR, wrap again.
        if let Some(hook) = fg_hook {
            // If the wrapper did not force a different request, initial
            // negotiation has already told us the effective non-FG space.
            // Retrying a colour space DXGI just refused buys only another
            // swapchain rebuild and duplicate error line.
            let fallback = if hook.fallback == want { space } else { hook.fallback };
            // A wrapper-only format must not survive when the wrapper does
            // not. XeSS-FG is attempted by default on Intel before its
            // optional DLLs are known to exist: if both wraps fail, rebuild
            // the plain session in the scRGB space `--hdr` originally
            // requested. Negotiate that space exactly like the initial path,
            // including the safe SDR fallback if DXGI refuses it.
            let restore_without_fg =
                |sc: IDXGISwapChain3,
                 current: PresentSpace,
                 target: PresentSpace|
                 -> Result<(IDXGISwapChain3, PresentSpace)> {
                    if current == target {
                        return Ok((sc, current));
                    }
                    drop(sc);
                    let fresh = match create(target.format()) {
                        Ok(sc) => sc,
                        Err(e) if target != PresentSpace::Sdr => {
                            eprintln!(
                                "present: could not restore the non-FG {target:?} swapchain \
                                 ({e}) — using SDR"
                            );
                            return Ok((create(PresentSpace::Sdr.format())?, PresentSpace::Sdr));
                        }
                        Err(e) => return Err(e),
                    };
                    if target != PresentSpace::Sdr {
                        if let Err(e) = declare_colorspace(&fresh, target) {
                            eprintln!(
                                "present: non-FG {target:?} colour space refused ({e}) — using SDR"
                            );
                            drop(fresh);
                            return Ok((create(PresentSpace::Sdr.format())?, PresentSpace::Sdr));
                        }
                    }
                    eprintln!(
                        "present: frame generation unavailable — restored the requested \
                         non-FG {target:?} swapchain"
                    );
                    Ok((fresh, target))
                };
            let rebuild_sdr_and_rewrap =
                |sc: IDXGISwapChain3,
                 unwind: bool,
                 hook_wrap: &mut dyn FnMut(
                    &ID3D12CommandQueue,
                    IDXGISwapChain3,
                )
                    -> std::result::Result<IDXGISwapChain3, (IDXGISwapChain3, String)>,
                 hook_unwind: &mut dyn FnMut()|
                 -> Result<(IDXGISwapChain3, bool)> {
                    drop(sc);
                    if unwind {
                        hook_unwind();
                    }
                    let fresh = create(PresentSpace::Sdr.format())?;
                    Ok(match hook_wrap(&queue, fresh) {
                        Ok(proxy) => (proxy, true),
                        Err((orig, e2)) => {
                            eprintln!(
                                "fg: swapchain wrap failed at SDR too ({e2}) — \
                                 frame generation disabled"
                            );
                            (orig, false)
                        }
                    })
                };
            swapchain = match (hook.wrap)(&queue, swapchain) {
                Ok(proxy) => {
                    // The proxy's internal chain was created fresh from our
                    // desc; a colour-space declaration does not survive that.
                    // Re-assert it through the proxy (it forwards
                    // SetColorSpace1 to the real chain).
                    match declare_colorspace(&proxy, space) {
                        Ok(()) => proxy,
                        Err(e) => {
                            eprintln!(
                                "present: {space:?} re-declare through the FG proxy refused ({e}) — \
                                 rebuilding at SDR (never a mis-declared present)"
                            );
                            space = PresentSpace::Sdr;
                            let (candidate, wrapped) =
                                rebuild_sdr_and_rewrap(proxy, true, hook.wrap, hook.unwind)?;
                            if wrapped {
                                candidate
                            } else {
                                let (plain, restored) =
                                    restore_without_fg(candidate, space, fallback)?;
                                space = restored;
                                plain
                            }
                        }
                    }
                }
                Err((orig, e)) => {
                    if space == PresentSpace::Hdr10 {
                        eprintln!(
                            "fg: swapchain wrap rejects the HDR10 chain ({e}) — rebuilding at SDR"
                        );
                        space = PresentSpace::Sdr;
                        let (candidate, wrapped) =
                            rebuild_sdr_and_rewrap(orig, false, hook.wrap, hook.unwind)?;
                        if wrapped {
                            candidate
                        } else {
                            let (plain, restored) =
                                restore_without_fg(candidate, space, fallback)?;
                            space = restored;
                            plain
                        }
                    } else {
                        eprintln!("fg: swapchain wrap failed ({e}) — frame generation disabled");
                        let (plain, restored) = restore_without_fg(orig, space, fallback)?;
                        space = restored;
                        plain
                    }
                }
            };
        }
        let format = space.format();

        let rtv_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: nbuf,
                ..Default::default()
            })
        }
        .map_err(|e| format!("CreateDescriptorHeap(RTV): {e}"))?;
        let rtv_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) };
        let rtv0 = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        let mut backbuffers = Vec::with_capacity(nbuf as usize);
        for i in 0..nbuf {
            let buf: ID3D12Resource = unsafe { swapchain.GetBuffer(i) }
                .map_err(|e| format!("swapchain GetBuffer({i}): {e}"))?;
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv0.ptr + (i * rtv_size) as usize,
            };
            unsafe { device.CreateRenderTargetView(&buf, None, handle) };
            backbuffers.push(buf);
        }

        let mut slots = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let allocator: ID3D12CommandAllocator = unsafe {
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }
            .map_err(|e| format!("CreateCommandAllocator: {e}"))?;
            slots.push(FrameSlot { allocator, fence_value: 0 });
        }
        let list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &slots[0].allocator, None)
        }
        .map_err(|e| format!("CreateCommandList: {e}"))?;
        unsafe { list.Close() }.map_err(|e| format!("initial list Close: {e}"))?;

        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .map_err(|e| format!("CreateFence: {e}"))?;
        let fence_event = unsafe { CreateEventW(None, false, false, None) }
            .map_err(|e| format!("CreateEventW: {e}"))?;

        Ok(Self {
            device,
            queue,
            swapchain,
            backbuffers,
            rtv_heap,
            rtv_size,
            list,
            slots,
            fence,
            fence_event,
            next_fence: 1,
            frame_index: 0,
            width,
            height,
            nbuf,
            format,
            space,
            sync_interval: if vsync { 1 } else { 0 },
            present_flags: if tearing { DXGI_PRESENT_ALLOW_TEARING } else { DXGI_PRESENT(0) },
        })
    }

    pub fn rtv_handle(&self, backbuffer: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let start = unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start.ptr + (backbuffer * self.rtv_size) as usize }
    }

    /// scRGB f16 live? The historical `hdr` bit — sites that mean "is the
    /// backbuffer linear f16" (the ffx FG backbuffer-format report, the f16
    /// blit) keep reading exactly this; sites that need the third arm switch
    /// on `space` instead.
    #[inline]
    pub fn hdr(&self) -> bool {
        self.space == PresentSpace::Scrgb
    }

    /// Resize the swapchain to a new client size. Drains the GPU, releases
    /// the backbuffer refs (DXGI requires zero outstanding references — the
    /// `backbuffers` Vec is their sole holder), ResizeBuffers with the SAME
    /// creation flags (a tearing swapchain must keep the tearing flag or the
    /// call fails), and recreates the RTVs in place (the heap holds a fixed
    /// `nbuf` descriptors — overwritten, never reallocated).
    /// `frame_index` needs no reset: it is only the frames-in-flight slot
    /// counter; the backbuffer index is queried fresh at every present.
    /// Works through the Streamline proxy swapchain too — ResizeBuffers is
    /// an SL hook and the manual-hooking guide's documented resize pattern.
    pub fn resize(&mut self, w: u32, h: u32) -> Result<()> {
        self.wait_idle()?;
        self.backbuffers.clear();
        let flags = if self.present_flags == DXGI_PRESENT_ALLOW_TEARING {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        } else {
            DXGI_SWAP_CHAIN_FLAG(0)
        };
        unsafe { self.swapchain.ResizeBuffers(self.nbuf, w, h, self.format, flags) }
            .map_err(|e| format!("ResizeBuffers({w}x{h}): {e}"))?;
        // ResizeBuffers preserves the colour space, so this re-declare is belt and
        // braces — which is exactly why a refusal must NOT kill the resize. The
        // swapchain we just resized is still the one we were already
        // presenting to; erroring here would take down the session over a
        // redundant call. Loud line, carry on.
        if self.space != PresentSpace::Sdr {
            if let Err(e) = declare_colorspace(&self.swapchain, self.space) {
                eprintln!(
                    "present: {:?} re-declare after resize failed ({e}) — keeping the existing colour space",
                    self.space
                );
            }
        }
        let rtv0 = unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        for i in 0..self.nbuf {
            let buf: ID3D12Resource = unsafe { self.swapchain.GetBuffer(i) }
                .map_err(|e| format!("resize GetBuffer({i}): {e}"))?;
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv0.ptr + (i * self.rtv_size) as usize,
            };
            unsafe { self.device.CreateRenderTargetView(&buf, None, handle) };
            self.backbuffers.push(buf);
        }
        self.width = w;
        self.height = h;
        Ok(())
    }

    fn wait_for_fence(&self, value: u64) -> Result<()> {
        if value == 0 || unsafe { self.fence.GetCompletedValue() } >= value {
            return Ok(());
        }
        unsafe { self.fence.SetEventOnCompletion(value, self.fence_event) }
            .map_err(|e| format!("SetEventOnCompletion: {e}"))?;
        unsafe { WaitForSingleObject(self.fence_event, INFINITE) };
        Ok(())
    }

    /// Wait until this frame's slot is free, reset its allocator + the list.
    /// Returns the slot index; upload memory for that slot is safe to touch
    /// after this returns.
    pub fn begin_frame(&mut self) -> Result<usize> {
        let slot = self.frame_index % FRAMES_IN_FLIGHT;
        self.wait_for_fence(self.slots[slot].fence_value)?;
        unsafe { self.slots[slot].allocator.Reset() }
            .map_err(|e| format!("allocator Reset: {e}"))?;
        unsafe { self.list.Reset(&self.slots[slot].allocator, None) }
            .map_err(|e| format!("list Reset: {e}"))?;
        // The fence wait above is what makes this slot's timestamps safe to
        // map: --gpu-timing reports frame N at the top of frame N+2, never
        // stalling the pipeline to do it.
        super::gputime::begin_frame(&self.device, &self.queue, slot);
        Ok(slot)
    }

    /// Close, execute, present, signal. Call after recording the frame.
    pub fn end_frame(&mut self, slot: usize) -> Result<()> {
        super::gputime::resolve(&self.list, slot);
        unsafe { self.list.Close() }.map_err(|e| format!("list Close: {e}"))?;
        let lists = [Some(self.list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { self.queue.ExecuteCommandLists(&lists) };
        unsafe { self.swapchain.Present(self.sync_interval, self.present_flags) }
            .ok()
            .map_err(|e| format!("Present: {e}"))?;
        let v = self.next_fence;
        self.next_fence += 1;
        unsafe { self.queue.Signal(&self.fence, v) }.map_err(|e| format!("queue Signal: {e}"))?;
        self.slots[slot].fence_value = v;
        self.frame_index += 1;
        // Report whatever the layer found in this frame's recording (a no-op
        // when it isn't on). Validation errors are useless unheard.
        drain_debug(&self.device);
        Ok(())
    }

    /// Close, execute, signal, and BLOCK until the GPU finishes — a frame
    /// submission without a Present, for work whose output the CPU reads
    /// back immediately (the post-upscale denoise path). The slot's fence
    /// bookkeeping matches `end_frame`, so the frame ring stays consistent
    /// and the caller's Map of a readback buffer is safe on return.
    pub fn submit_and_wait(&mut self, slot: usize) -> Result<()> {
        crate::zone!("gpu-wait");
        super::gputime::resolve(&self.list, slot);
        unsafe { self.list.Close() }.map_err(|e| format!("list Close: {e}"))?;
        let lists = [Some(self.list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { self.queue.ExecuteCommandLists(&lists) };
        let v = self.next_fence;
        self.next_fence += 1;
        unsafe { self.queue.Signal(&self.fence, v) }.map_err(|e| format!("queue Signal: {e}"))?;
        self.slots[slot].fence_value = v;
        self.frame_index += 1;
        self.wait_for_fence(v)
    }

    /// Close + execute the list MID-frame — no Present, no fence wait — then
    /// reset it on the same slot's allocator and keep recording. Sound: an
    /// allocator may back multiple closed lists (only the allocator reset in
    /// `begin_frame` needs the fence), and the GPU consumes the submissions
    /// in queue order — single-queue FIFO is the synchronization. For work
    /// that must be ON the queue before an external submitter (ONNX
    /// Runtime's DML EP in the NPPD composition) appends its own.
    /// The slot's fence IS advanced past the executed half: `abort_frame`
    /// leaves the fence untouched, so without this a post-split error would
    /// let the next `begin_frame` reset the allocator under the still-running
    /// submission. On the success path `end_frame` overwrites it with a later
    /// value — the Signal is pure error-path insurance, never waited on here.
    /// Mid-frame present — the frame-generation pair-present's first half:
    /// Close + Execute + Present, then Reset the list on the SAME slot
    /// allocator (split_frame's legality argument — only ALLOCATOR reset
    /// needs the fence) so the caller records the frame's second half. The
    /// slot's fence signals once, at the final end_frame; under vsync the two
    /// Presents land one vblank apart, which IS the pacing.
    pub fn present_mid(&mut self, slot: usize) -> Result<()> {
        unsafe { self.list.Close() }.map_err(|e| format!("mid Close: {e}"))?;
        let lists = [Some(self.list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { self.queue.ExecuteCommandLists(&lists) };
        unsafe { self.swapchain.Present(self.sync_interval, self.present_flags) }
            .ok()
            .map_err(|e| format!("mid Present: {e}"))?;
        unsafe { self.list.Reset(&self.slots[slot].allocator, None) }
            .map_err(|e| format!("mid list Reset: {e}"))
    }

    pub fn split_frame(&mut self, slot: usize) -> Result<()> {
        unsafe { self.list.Close() }.map_err(|e| format!("split Close: {e}"))?;
        let lists = [Some(self.list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { self.queue.ExecuteCommandLists(&lists) };
        let v = self.next_fence;
        self.next_fence += 1;
        unsafe { self.queue.Signal(&self.fence, v) }.map_err(|e| format!("queue Signal: {e}"))?;
        self.slots[slot].fence_value = v;
        unsafe { self.list.Reset(&self.slots[slot].allocator, None) }
            .map_err(|e| format!("split list Reset: {e}"))
    }

    pub fn wait_idle(&mut self) -> Result<()> {
        let v = self.next_fence;
        self.next_fence += 1;
        unsafe { self.queue.Signal(&self.fence, v) }.map_err(|e| format!("queue Signal: {e}"))?;
        self.wait_for_fence(v)
    }

    /// Abandon a partially recorded frame: close the list without executing
    /// it (recorded barriers/copies never reach the GPU, so tracked resource
    /// states stay truthful) and leave the slot's fence untouched so the next
    /// begin_frame reuses it. Recovery path for a mid-frame error.
    pub fn abort_frame(&mut self) {
        let _ = unsafe { self.list.Close() };
    }

    /// Record and synchronously execute a one-off command list outside the
    /// frame loop (drains in-flight work before borrowing slot 0's allocator
    /// and drains again after). For rare operations like readbacks.
    pub fn run_once<F: FnOnce(&ID3D12GraphicsCommandList)>(&mut self, f: F) -> Result<()> {
        self.wait_idle()?;
        unsafe { self.slots[0].allocator.Reset() }
            .map_err(|e| format!("run_once allocator Reset: {e}"))?;
        unsafe { self.list.Reset(&self.slots[0].allocator, None) }
            .map_err(|e| format!("run_once list Reset: {e}"))?;
        f(&self.list);
        unsafe { self.list.Close() }.map_err(|e| format!("run_once list Close: {e}"))?;
        let lists = [Some(self.list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { self.queue.ExecuteCommandLists(&lists) };
        self.wait_idle()
    }
}

/// One-off blocking submission — the seam `SceneGpu`'s chunked scene upload
/// streams through, implemented by both harnesses (`HeadlessGpu` for the
/// check suites, `D3d` for interactive sessions). Each call records,
/// executes, and BLOCKS until the GPU finishes, which is what lets one
/// staging ring be reused across calls instead of committing a full second
/// copy of the scene in upload heaps.
pub trait Submit {
    fn run_list(
        &mut self,
        f: &mut dyn FnMut(&ID3D12GraphicsCommandList) -> Result<()>,
    ) -> Result<()>;
}

impl Submit for D3d {
    fn run_list(
        &mut self,
        f: &mut dyn FnMut(&ID3D12GraphicsCommandList) -> Result<()>,
    ) -> Result<()> {
        let mut rec = Ok(());
        self.run_once(|l| rec = f(l))?;
        rec
    }
}

impl Drop for D3d {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        let _ = unsafe { CloseHandle(self.fence_event) };
    }
}

/// Transition barrier. The `transmute_copy` pattern is the standard
/// windows-rs idiom: the union field is `ManuallyDrop<Option<ID3D12Resource>>`
/// and we hand it a borrowed refcount that ManuallyDrop keeps from releasing.
pub fn transition(
    res: &ID3D12Resource,
    from: D3D12_RESOURCE_STATES,
    to: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(res) },
                StateBefore: from,
                StateAfter: to,
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            }),
        },
    }
}

/// UAV barrier: all UAV writes before it complete before any UAV access
/// after it. `None` = global (covers every UAV) — the trace pipeline's
/// between-dispatch fence, cheaper to reason about than per-resource lists.
pub fn uav_barrier(res: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: match res {
                    Some(r) => unsafe { std::mem::transmute_copy(r) },
                    None => ManuallyDrop::new(None),
                },
            }),
        },
    }
}

/// Copy location for a texture subresource — either side of a
/// CopyTextureRegion (upload dst, readback src).
pub fn loc_subresource(res: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    loc_subresource_mip(res, 0)
}

/// `loc_subresource` for an explicit mip level (scene-texture chains; every
/// other texture in the renderer is single-mip).
pub fn loc_subresource_mip(res: &ID3D12Resource, mip: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(res) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: mip },
    }
}

/// Copy location for a buffer with a placed footprint — either side of a
/// CopyTextureRegion (upload src, readback dst).
pub fn loc_footprint(
    res: &ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(res) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
    }
}

pub fn default_heap() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }
}

pub fn upload_heap() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }
}

pub fn readback_heap() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }
}

pub fn tex2d_desc(w: u32, h: u32, format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: w as u64,
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
    }
}

pub fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

/// Default-heap buffer with explicit flags/state — the trace pipeline's
/// UAV queues, pools, and per-pixel planes.
pub fn committed_buffer(
    device: &ID3D12Device,
    size: u64,
    flags: D3D12_RESOURCE_FLAGS,
    initial: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource> {
    let mut desc = buffer_desc(size);
    desc.Flags = flags;
    let mut res: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &default_heap(),
            D3D12_HEAP_FLAG_NONE,
            &desc,
            initial,
            None,
            &mut res,
        )
    }
    .map_err(|e| format!("CreateCommittedResource(buffer {size}B): {e}"))?;
    Ok(res.unwrap())
}

pub fn committed_tex(
    device: &ID3D12Device,
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
    flags: D3D12_RESOURCE_FLAGS,
    initial: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource> {
    committed_tex_mips(device, w, h, 1, format, flags, initial)
}

/// `committed_tex` with a mip chain (scene textures; every other texture
/// stays single-mip through the wrapper above).
pub fn committed_tex_mips(
    device: &ID3D12Device,
    w: u32,
    h: u32,
    mips: u16,
    format: DXGI_FORMAT,
    flags: D3D12_RESOURCE_FLAGS,
    initial: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource> {
    let mut desc = tex2d_desc(w, h, format, flags);
    desc.MipLevels = mips;
    let mut res: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &default_heap(),
            D3D12_HEAP_FLAG_NONE,
            &desc,
            initial,
            None,
            &mut res,
        )
    }
    .map_err(|e| format!("CreateCommittedResource(tex {w}x{h} m{mips} {format:?}): {e}"))?;
    Ok(res.unwrap())
}

/// Persistently-mapped upload buffer.
pub struct UploadBuffer {
    pub resource: ID3D12Resource,
    pub ptr: *mut u8,
    pub size: usize,
}

// The mapped pointer is only written from the render thread between
// begin_frame (fence wait) and end_frame; D3D12 upload heaps are always
// CPU-visible write-combined memory.
unsafe impl Send for UploadBuffer {}

impl UploadBuffer {
    pub fn new(device: &ID3D12Device, size: usize) -> Result<Self> {
        let mut res: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &upload_heap(),
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size as u64),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut res,
            )
        }
        .map_err(|e| format!("CreateCommittedResource(upload {size}): {e}"))?;
        let resource = res.unwrap();
        let mut ptr = std::ptr::null_mut();
        unsafe { resource.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("Map: {e}"))?;
        Ok(Self { resource, ptr: ptr as *mut u8, size })
    }
}

/// GPU→CPU staging buffer for rare readbacks (screenshots). Created on
/// demand; mapped only for the duration of the read.
pub struct ReadbackBuffer {
    pub resource: ID3D12Resource,
}

impl ReadbackBuffer {
    pub fn new(device: &ID3D12Device, size: usize) -> Result<Self> {
        let mut res: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &readback_heap(),
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size as u64),
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut res,
            )
        }
        .map_err(|e| format!("CreateCommittedResource(readback {size}): {e}"))?;
        Ok(Self { resource: res.unwrap() })
    }
}

pub const ROW_ALIGN: usize = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as usize; // 256

pub fn aligned_pitch(row_bytes: usize) -> usize {
    (row_bytes + ROW_ALIGN - 1) & !(ROW_ALIGN - 1)
}

pub fn footprint(
    format: DXGI_FORMAT,
    w: u32,
    h: u32,
    bytes_per_px: usize,
    offset: u64,
) -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        Offset: offset,
        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
            Format: format,
            Width: w,
            Height: h,
            Depth: 1,
            RowPitch: aligned_pitch(w as usize * bytes_per_px) as u32,
        },
    }
}

/// Placed footprint of a BLOCK-COMPRESSED region (BC7 — 4×4 texels per 16-byte
/// block). The plain `footprint` above cannot express this: its pitch is
/// `w · bytes_per_px`, and a BC row is `ceil(w/4) · 16` bytes regardless of
/// what a "byte per texel" would mean.
///
/// `w`/`h` stay in TEXELS (that is what D3D12_SUBRESOURCE_FOOTPRINT wants for
/// BC formats). The debug layer requires them to be multiples of 4 UNLESS they
/// equal the resource's own dimensions — callers pass the full texture width
/// and a band height that is a whole number of block rows except on the last
/// band, which reaches `h` exactly, so both cases are legal. The matching
/// `CopyTextureRegion` DstY must likewise be a multiple of 4 (whole block rows).
pub fn footprint_block(
    format: DXGI_FORMAT,
    w: u32,
    h: u32,
    offset: u64,
) -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        Offset: offset,
        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
            Format: format,
            Width: w,
            Height: h,
            Depth: 1,
            RowPitch: block_pitch(w) as u32,
        },
    }
}

/// Aligned bytes of one BC7 block ROW (4 texel rows) of a `w`-texel-wide image.
pub fn block_pitch(w: u32) -> usize {
    aligned_pitch(crate::bc7::blocks(w) as usize * crate::bc7::BLOCK_BYTES)
}
