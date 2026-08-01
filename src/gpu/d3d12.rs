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

/// The 10-bit swapchain format — ONE format, two transfer curves: declared
/// `G2084_NONE_P2020` it is HDR10/PQ (the HDR-on-display default, and the arm
/// the swapchain-wrapper FG families take — 10-bit PQ is the format HDR FG
/// titles actually ship); left undeclared it reads as gamma-2.2
/// (`G22_NONE_P709`, DXGI's default interpretation of a UNORM chain) — the
/// Sdr10 deep-colour arm, the HDR-off-display default. 4 B/px either way,
/// which is the point: the old scRGB f16 chain was 8, and the present is the
/// whole frame budget when the display hangs off a different GPU than the
/// renderer. See `tone::ToneMode`.
pub const SWAPCHAIN_FORMAT_10BIT: DXGI_FORMAT = DXGI_FORMAT_R10G10B10A2_UNORM;

/// The format the CPU-present blit texture is uploaded in. **Deliberately its
/// own constant, not `SWAPCHAIN_FORMAT`**: `BlitUpload` packs pixels as
/// `u32 0x00RRGGBB`, whose little-endian byte order is B,G,R,X — a layout that
/// is only valid for B8G8R8A8. If it followed the swapchain format it would
/// silently reinterpret those bytes under the 10-bit chain. The 10-bit blit
/// path has its own texture (`BlitUpload::new_10bit`) instead.
pub const BLIT_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;

/// 10-bit for the CPU-present blit under Sdr10/Hdr10 — `render::present_px_pq`
/// / `present_px_sdr10` own the whole encode (curve + encode + the 10-bit
/// pack), so the blit PS stays a passthrough here too. Its own const per the
/// `BLIT_FORMAT` doctrine: the packed `u32` lane order (R low) is only valid
/// for R10G10B10A2.
pub const BLIT_FORMAT_10BIT: DXGI_FORMAT = DXGI_FORMAT_R10G10B10A2_UNORM;

/// Which color space presentation was actually negotiated in. Runtime fact,
/// never the CLI flag — the HDR10 declare can be refused and the FG wrap can
/// force a rebuild, so callers must read `D3d::space`. What varies with it is
/// the tonemap encode (`tone::ToneMode`), the CPU blit format, and nothing
/// upstream of the present.
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

impl PresentSpace {
    pub fn format(self) -> DXGI_FORMAT {
        match self {
            PresentSpace::Sdr => SWAPCHAIN_FORMAT,
            PresentSpace::Sdr10 | PresentSpace::Hdr10 => SWAPCHAIN_FORMAT_10BIT,
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

/// Declare the swapchain's color space (HDR10 G2084 only — `Sdr` AND `Sdr10`
/// are DXGI's default gamma-2.2 reading and declare nothing: an explicit
/// `SetColorSpace1(G22)` would buy zero information and add two failure edges,
/// proxy forwarding through the FG wrappers and a needless refusal path).
/// `CheckColorSpaceSupport` first: `SetColorSpace1` on an unsupported space is
/// an error, and we want to know *before* committing so the caller can fall
/// back (to Sdr10 — same buffer, default reading).
///
/// Idempotent and cheap — the FG wrap path re-asserts it through the proxy
/// (the proxy's fresh internal chain does not inherit a declaration). A live
/// re-declare when the window moves between displays is the noted follow-on;
/// today the declaration is session-static.
fn declare_colorspace(swapchain: &IDXGISwapChain3, space: PresentSpace) -> Result<()> {
    let (cs, name) = match space {
        PresentSpace::Sdr | PresentSpace::Sdr10 => return Ok(()),
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
    /// const, because `ResizeBuffers` and both fullscreen PSOs must agree with
    /// whatever was chosen. `space` falls back (Hdr10 -> Sdr10 on a refused
    /// G2084; any 10-bit -> Sdr when the FG wrap forces a rebuild), so callers
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

/// Create a direct command queue.
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
/// wrap degrades to a normal session, restored to the space negotiation had
/// already settled on. (Two fields died with scRGB: `fallback` — the FG
/// families no longer force a different space than the non-FG session would
/// take, so there is no separate "what the session wanted" to carry — and
/// `unwind`: a G2084 re-declare failure through the proxy relabels the
/// session Sdr10 and keeps the proxy, the one path that used to tear the FG
/// context down.) d3d12.rs stays SDK-agnostic — the closure lives with the FG
/// runtime that owns the proxy.
pub struct FgHook<'a> {
    pub wrap: &'a mut dyn FnMut(
        &ID3D12CommandQueue,
        IDXGISwapChain3,
    ) -> std::result::Result<IDXGISwapChain3, (IDXGISwapChain3, String)>,
}

impl D3d {
    /// Build the swapchain + frame machinery around an existing device/queue.
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

        // ONE 10-bit format (R10G10B10A2, 4 B/px), two transfer curves: the
        // G2084/PQ declaration is what makes it HDR10; undeclared it reads as
        // gamma-2.2 (Sdr10, deep-colour SDR). The caller decides which by the
        // display probe — PQ on an HDR-on output, Sdr10 everywhere else (see
        // GpuContext::new). 10-bit wins on BYTES: the old scRGB f16 chain was
        // 8 B/px, and the present is the whole frame budget when the display
        // hangs off a different GPU than the renderer and DWM has to copy
        // every frame across (measured 8K cross-adapter: ~80 -> ~51 ms per
        // present just from halving the format). 10-bit gamma keeps the
        // deep-colour quality f16 used to buy on SDR outputs (an 8-bit
        // backbuffer banded). What varies with the display is the CURVE
        // (crate::tone) and the declaration, never the format.
        //
        // A refused G2084 declare is a RELABEL, not a rebuild: the 10-bit
        // buffer already exists and Sdr10 is its default reading — a failed
        // SetColorSpace1 leaves the chain in exactly the state Sdr10 wants.
        if space != PresentSpace::Sdr {
            match declare_colorspace(&swapchain, space) {
                Ok(()) => match space {
                    PresentSpace::Sdr10 => eprintln!(
                        "present: 10-bit SDR swapchain (R10G10B10A2_UNORM, default gamma-2.2 \
                         reading — deep colour at 4 B/px)"
                    ),
                    PresentSpace::Hdr10 => eprintln!(
                        "present: HDR10 swapchain (R10G10B10A2_UNORM, G2084_NONE_P2020, PQ)"
                    ),
                    PresentSpace::Sdr => unreachable!(),
                },
                Err(e) => {
                    // Only Hdr10 can land here — Sdr10 declares nothing.
                    eprintln!(
                        "present: G2084 refused ({e}) — presenting 10-bit SDR \
                         (gamma-2.2 default reading; --no-hdr forces 8-bit)"
                    );
                    space = PresentSpace::Sdr10;
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
        //  - wrap fails at a 10-bit space (Sdr10 or Hdr10): the proxy itself
        //    may be rejecting the format (XeSS-FG's InitFromSwapChain is
        //    format-picky), and both spaces share R10G10B10A2, so there is no
        //    intermediate rung. FG is the reason this session exists, so
        //    rebuild at 8-bit SDR and wrap AGAIN — SDR with FG beats 10-bit
        //    without it.
        //  - wrap fails at Sdr: restore the non-FG presentation request and
        //    run without FG.
        //  - wrap succeeds but the G2084 re-declare through the proxy fails
        //    (only reachable at Hdr10 — Sdr10 declares nothing): NEVER present
        //    mis-declared, but a failed declare leaves the proxy's fresh chain
        //    at its default gamma-2.2 reading, which IS Sdr10 — so relabel to
        //    Sdr10 and keep the proxy. No rebuild, no unwind.
        if let Some(hook) = fg_hook {
            // If every wrap attempt fails, the plain session restores the
            // space initial negotiation settled on — NOT the pre-negotiation
            // request: retrying a colour space DXGI just refused buys only
            // another swapchain rebuild and a duplicate error line. Captured
            // here because the ladder below mutates `space` on its way down.
            let fallback = space;
            // A wrapper-only space must not survive when the wrapper does
            // not. XeSS-FG is attempted by default on Intel before its
            // optional DLLs are known to exist: if both wraps fail, rebuild
            // the plain session in the space the session originally
            // requested. Negotiate that space exactly like the initial path,
            // including the safe fallback if DXGI refuses it.
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
                    if let Err(e) = declare_colorspace(&fresh, target) {
                        // Only Hdr10 declares; a refusal leaves the fresh
                        // 10-bit chain at its default gamma-2.2 reading, which
                        // IS Sdr10 — relabel, keep the chain.
                        eprintln!(
                            "present: non-FG G2084 refused ({e}) — presenting 10-bit SDR"
                        );
                        return Ok((fresh, PresentSpace::Sdr10));
                    }
                    eprintln!(
                        "present: frame generation unavailable — restored the requested \
                         non-FG {target:?} swapchain"
                    );
                    Ok((fresh, target))
                };
            let rebuild_sdr_and_rewrap =
                |sc: IDXGISwapChain3,
                 hook_wrap: &mut dyn FnMut(
                    &ID3D12CommandQueue,
                    IDXGISwapChain3,
                )
                    -> std::result::Result<IDXGISwapChain3, (IDXGISwapChain3, String)>|
                 -> Result<(IDXGISwapChain3, bool)> {
                    drop(sc);
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
                            // Only Hdr10 can land here (Sdr10 declares
                            // nothing). The proxy just accepted the 10-bit
                            // format and its fresh chain sits at the default
                            // gamma-2.2 reading — relabel to Sdr10 and keep
                            // the proxy. Never a mis-declared present, and
                            // never a needless rebuild.
                            eprintln!(
                                "present: G2084 re-declare through the FG proxy refused ({e}) — \
                                 presenting 10-bit SDR through the proxy"
                            );
                            space = PresentSpace::Sdr10;
                            proxy
                        }
                    }
                }
                Err((orig, e)) => {
                    if space != PresentSpace::Sdr {
                        // Sdr10 and Hdr10 share one format, so a format-picky
                        // wrap has no intermediate rung — rebuild at 8-bit.
                        eprintln!(
                            "fg: swapchain wrap rejects the 10-bit chain ({e}) — rebuilding at SDR"
                        );
                        space = PresentSpace::Sdr;
                        let (candidate, wrapped) = rebuild_sdr_and_rewrap(orig, hook.wrap)?;
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

    /// Resize the swapchain to a new client size. Drains the GPU, releases
    /// the backbuffer refs (DXGI requires zero outstanding references — the
    /// `backbuffers` Vec is their sole holder), ResizeBuffers with the SAME
    /// creation flags (a tearing swapchain must keep the tearing flag or the
    /// call fails), and recreates the RTVs in place (the heap holds a fixed
    /// `nbuf` descriptors — overwritten, never reallocated).
    /// `frame_index` needs no reset: it is only the frames-in-flight slot
    /// counter; the backbuffer index is queried fresh at every present.
    /// Works through the FG-family proxy swapchains too — they forward
    /// ResizeBuffers to the chain they wrap.
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
