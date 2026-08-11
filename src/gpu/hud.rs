//! GPU half of the HUD/menu overlay (the CPU half — Slint software
//! rendering — is `crate::hud`): a window-sized premultiplied RGBA8 texture,
//! dirty-rect uploads, and a premultiplied-alpha fullscreen composite drawn
//! inside `fullscreen_to_backbuffer` after the tonemap pass — ONE insertion
//! point, so every present arm (CPU, GPU, DXR, every upscaler) gets the HUD.
//!
//! TRUE dirty rectangles, all three layers: Slint only re-rasterizes the
//! dirty region (ReusedBuffer), `crate::hud` packs only those rects' bytes,
//! and this module memcpys only those bytes into the upload ring and records
//! ONE `CopyTextureRegion` per rect (non-zero DstX/DstY + a source
//! `D3D12_BOX` — the `footprint_block` DstY precedent extended). A frame
//! with an unchanged HUD stages nothing, copies nothing, and only pays the
//! composite draw; a fully hidden HUD pays nothing at all.
//!
//! Alignment: every rect shares ONE full-window placed footprint per
//! frame-in-flight slice (offset = slot·slice, 512-aligned by construction;
//! row pitch 256-aligned via `aligned_pitch`), with the rect named by the
//! source box — so per-rect copies need no per-rect offset alignment.
//!
//! Interior mutability throughout (`fullscreen_to_backbuffer` is `&self`):
//! staged rects live in a RefCell, consumed at record time — which is after
//! `begin_frame`'s fence wait, making the slot's ring memory safe to write.

use super::d3d12::{
    aligned_pitch, committed_tex, footprint, loc_footprint, loc_subresource, transition, Result,
    UploadBuffer, FRAMES_IN_FLIGHT,
};
use super::tonemap::{self, Passes};
use crate::hud::{DirtyRect, HudFrame};
use std::cell::{Cell, RefCell};
use windows::core::s;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;

use crate::gfx::shaders::HUD_HLSL;

/// D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT — a placed footprint's Offset must
/// be 512-aligned, so the per-frame-in-flight slice is rounded up to it.
const PLACEMENT_ALIGN: usize = 512;

#[derive(Default)]
struct Pending {
    rects: Vec<DirtyRect>,
    /// Each rect's rows, tightly packed (rect.w * 4 bytes per row), in rect
    /// order — `crate::hud::Hud::frame`'s packing.
    bytes: Vec<u8>,
}

pub struct HudGpu {
    texture: ID3D12Resource,
    upload: UploadBuffer,
    pub pso: ID3D12PipelineState,
    /// Aligned pitch of one full-window row in the ring.
    pitch: usize,
    /// 512-aligned full-window slice per frame in flight (worst case: the
    /// first frame and any resize are fully dirty).
    slice: usize,
    w: u32,
    h: u32,
    pending: RefCell<Pending>,
    /// The texture holds valid pixels (the first stage happened) — drawing
    /// before that would composite uninitialized memory.
    uploaded: Cell<bool>,
    /// Whether the composite draw runs (the HUD/menu is on screen). Fed per
    /// frame by the session; staging keeps happening while hidden so the
    /// buffer stays current and re-showing needs no special case.
    pub visible: Cell<bool>,
    /// The compiled hud.hlsl blobs, kept so `HudFi` (built after this at
    /// boot AND on every resize) reuses them instead of recompiling the
    /// same source — hud.hlsl stays the single encoder either way, this
    /// just makes the "same shader" claim structural.
    vs: ID3DBlob,
    ps: ID3DBlob,
}

impl HudGpu {
    pub fn new(device: &ID3D12Device, passes: &Passes, rtv_format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, w: u32, h: u32) -> Result<Self> {
        let texture = committed_tex(
            device,
            w,
            h,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        passes.create_srv(device, &texture, DXGI_FORMAT_R8G8B8A8_UNORM, tonemap::SRV_SLOT_OVERLAY);
        let pitch = aligned_pitch(w as usize * 4);
        let slice = (pitch * h as usize + PLACEMENT_ALIGN - 1) & !(PLACEMENT_ALIGN - 1);
        let upload = UploadBuffer::new(device, slice * FRAMES_IN_FLIGHT)?;
        let vs = tonemap::compile(HUD_HLSL, s!("vsmain"), s!("vs_5_0"), "hud vs")?;
        let ps = tonemap::compile(HUD_HLSL, s!("psmain"), s!("ps_5_0"), "hud ps")?;
        let pso = tonemap::fullscreen_pso_blend(
            device,
            rtv_format,
            &passes.root_sig,
            &vs,
            &ps,
            tonemap::premultiplied_blend(),
        )?;
        Ok(Self {
            texture,
            upload,
            pso,
            pitch,
            slice,
            w,
            h,
            pending: RefCell::new(Pending::default()),
            uploaded: Cell::new(false),
            visible: Cell::new(false),
            vs,
            ps,
        })
    }

    /// Stage a frame's dirty rects (main loop, once per frame, before the
    /// present arm records). Appends — an arm that errors out after staging
    /// leaves the rects for the next recorded frame rather than dropping
    /// them.
    pub fn stage(&self, frame: HudFrame) {
        if frame.rects.is_empty() {
            return;
        }
        let mut p = self.pending.borrow_mut();
        p.rects.extend_from_slice(&frame.rects);
        p.bytes.extend_from_slice(&frame.bytes);
    }

    /// True when the composite draw should record: something is on screen and
    /// the texture holds real pixels.
    pub fn drawable(&self) -> bool {
        self.visible.get() && self.uploaded.get()
    }

    /// Consume the staged rects: memcpy each into this slot's ring slice at
    /// its natural position and record one CopyTextureRegion per rect.
    /// Called from `fullscreen_to_backbuffer` (post-begin_frame, so the
    /// slot's fence has been waited).
    pub fn record_upload(&self, list: &ID3D12GraphicsCommandList, slot: usize) {
        let mut p = self.pending.borrow_mut();
        if p.rects.is_empty() {
            return;
        }
        let base = slot * self.slice;
        let dst =
            unsafe { std::slice::from_raw_parts_mut(self.upload.ptr.add(base), self.slice) };
        unsafe {
            list.ResourceBarrier(&[transition(
                &self.texture,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
        }
        let fp = footprint(DXGI_FORMAT_R8G8B8A8_UNORM, self.w, self.h, 4, base as u64);
        let mut src_off = 0usize;
        for r in &p.rects {
            // Defensive clamp — a rect from a stale (pre-resize) frame must
            // not index outside the ring.
            let x = r.x.min(self.w);
            let y = r.y.min(self.h);
            let rw = r.w.min(self.w - x);
            let rh = r.h.min(self.h - y);
            let row_bytes = r.w as usize * 4;
            let copy_bytes = rw as usize * 4;
            for row in 0..rh as usize {
                let d = (y as usize + row) * self.pitch + x as usize * 4;
                dst[d..d + copy_bytes]
                    .copy_from_slice(&p.bytes[src_off + row * row_bytes..][..copy_bytes]);
            }
            src_off += row_bytes * r.h as usize;
            if rw == 0 || rh == 0 {
                continue;
            }
            unsafe {
                list.CopyTextureRegion(
                    &loc_subresource(&self.texture),
                    x,
                    y,
                    0,
                    &loc_footprint(&self.upload.resource, fp),
                    Some(&D3D12_BOX {
                        left: x,
                        top: y,
                        front: 0,
                        right: x + rw,
                        bottom: y + rh,
                        back: 1,
                    }),
                );
            }
        }
        unsafe {
            list.ResourceBarrier(&[transition(
                &self.texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
        p.rects.clear();
        p.bytes.clear();
        self.uploaded.set(true);
    }
}

/// The frame-generation UI target: a window-sized DISPLAY-SPACE premultiplied
/// render of the HUD, registered with the FG proxy (ffx FI's UI resource /
/// XeSS-FG's RES_UI tag) so the proxy composites the UI AFTER interpolation
/// instead of warping baked HUD pixels by scene motion on generated frames.
///
/// A pre-pass, not a second encoder: `record` draws the SAME HudGpu texture
/// through the SAME hud.hlsl shader (SRV_SLOT_OVERLAY, the tonemap root
/// signature, the live ToneParams) that the baked backbuffer composite uses —
/// hud.hlsl stays the single display-space encoder, so the registered texture
/// matches the baked draw bit-for-bit per texel in every present space. The
/// only differences are the OPAQUE write (the target holds the UI alone —
/// empty HUD texels are (0,0,0,0) premul, which the proxy's premul blend
/// leaves invisible; no clear needed, the fullscreen triangle writes every
/// pixel) and the format: RGBA16F under HDR10 (8-bit PQ bands, and the
/// backbuffer's R10G10B10A2 has 2-bit alpha — disqualified as a UI blend
/// source), RGBA8 under SDR/Sdr10 (mode 1 is a passthrough bit-copy).
///
/// Rests in PIXEL_SHADER_RESOURCE — the state the registration declares
/// (RES_STATE_PIXEL_READ / the xefg incoming_state).
pub struct HudFi {
    texture: ID3D12Resource,
    /// Keeps the RTV descriptor alive — `rtv` points into it.
    _rtv_heap: ID3D12DescriptorHeap,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pso: ID3D12PipelineState,
    w: u32,
    h: u32,
}

impl HudFi {
    pub fn new(
        device: &ID3D12Device,
        passes: &Passes,
        hud: &HudGpu,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        w: u32,
        h: u32,
    ) -> Result<Self> {
        let texture = committed_tex(
            device,
            w,
            h,
            format,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        let rtv_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })
        }
        .map_err(|e| format!("hud-fi RTV heap: {e}"))?;
        let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        unsafe { device.CreateRenderTargetView(&texture, None, rtv) };
        // HudGpu's own compiled blobs — same source, same entries; only the
        // PSO differs (RT format + opaque write vs premul blend).
        let pso = tonemap::fullscreen_pso_blend(
            device,
            format,
            &passes.root_sig,
            &hud.vs,
            &hud.ps,
            tonemap::default_blend(),
        )?;
        Ok(Self { texture, _rtv_heap: rtv_heap, rtv, pso, w, h })
    }

    /// Render the HUD texture into the display-space target (call after
    /// `HudGpu::record_upload` on the same list — single-queue order is the
    /// only sync the SRV read needs).
    pub fn record(&self, list: &ID3D12GraphicsCommandList, passes: &Passes, tone: crate::tone::ToneParams) {
        unsafe {
            list.ResourceBarrier(&[transition(
                &self.texture,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )]);
        }
        passes.record(
            list,
            &self.pso,
            tonemap::SRV_SLOT_OVERLAY,
            1.0,
            (0.0, 0.0, 0.0),
            tone,
            self.rtv,
            self.w,
            self.h,
        );
        unsafe {
            list.ResourceBarrier(&[transition(
                &self.texture,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
    }

    /// The registered resource (rests PIXEL_SHADER_RESOURCE).
    pub fn resource(&self) -> &ID3D12Resource {
        &self.texture
    }
}
