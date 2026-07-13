//! FSR 3.1 upscale-only resources: the three standard temporal-upscaler
//! input planes (color, motion, depth) plus the window-res output —
//! `xr.rs::XessResources`' plane set with `ffx_rr.rs`' conventions. The
//! FSR3 flavor has no denoiser, so none of the Ray Regeneration planes
//! (normals, albedos, signals, residual) exist here: the frame's shaded
//! HDR color goes straight into the upscale dispatch. Like the RR path,
//! dynamic input resolution is first-class: planes are allocated once at
//! the range MAX and each frame uploads (and the dispatch reads, via its
//! per-dispatch `renderSize`) only the top-left `rw×rh` sub-rect.
//!
//! Formats: color RGBA16F, motion RG16F (pixel-space f16 bit copy of
//! `GBufs::mvec` — no re-encode), depth R32F (the codebase-wide f32 depth
//! wire contract; `xess::view_z_to_clip_depth` reversed-Z, gated by
//! --check-xess).

use super::d3d12::{
    aligned_pitch, committed_tex, footprint, get_or_try_init, loc_footprint, loc_subresource,
    transition, D3d, Result, UploadBuffer, FRAMES_IN_FLIGHT,
};
use super::ffx_sys::{FfxShimRes, RES_STATE_COMPUTE_READ, RES_STATE_UNORDERED_ACCESS};
use crate::dlss::GBufs;
use crate::fsr;
use crate::xess;
use half::f16;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

struct Plane {
    tex: ID3D12Resource,
    format: DXGI_FORMAT,
    bpp: usize,
    offset: usize, // into each upload slot, sized for the max input res
}

// Upload plane order.
const P_COLOR: usize = 0; // RGBA16F linear HDR (the frame's 1-spp shade)
const P_MVEC: usize = 1; // RG16F pixel-space MVs, current→previous, y-down
const P_DEPTH: usize = 2; // R32F reversed-Z clip depth
const N_PLANES: usize = 3;

pub struct Fsr3Resources {
    pub max_w: u32, // input plane allocation size (range max)
    pub max_h: u32,
    planes: [Plane; N_PLANES],
    /// FSR 3.1 output at window res — the tonemap SRV target (screenshots
    /// read it back through the shared GpuContext::read_hdr_output path).
    pub upscaled: ID3D12Resource,
    upload: std::sync::OnceLock<UploadBuffer>,
    device: ID3D12Device,
    slot_stride: usize,
}

impl Fsr3Resources {
    pub fn new(device: &ID3D12Device, max_w: u32, max_h: u32, ow: u32, oh: u32) -> Result<Self> {
        let specs: [(DXGI_FORMAT, usize); N_PLANES] = [
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 8),
            (DXGI_FORMAT_R16G16_FLOAT, 4),
            (DXGI_FORMAT_R32_FLOAT, 4),
        ];
        let mut offset = 0usize;
        let mut planes = Vec::with_capacity(N_PLANES);
        for (format, bpp) in specs {
            // ALLOW_UNORDERED_ACCESS: the GPU-fed sessions' feed kernel
            // writes these planes directly (typed UAV stores) — the xr.rs
            // precedent. The CPU upload path is unaffected; the rest state
            // stays NON_PIXEL_SHADER_RESOURCE either way.
            let tex = committed_tex(
                device,
                max_w,
                max_h,
                format,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )?;
            planes.push(Plane { tex, format, bpp, offset });
            offset += aligned_pitch(max_w as usize * bpp) * max_h as usize;
        }
        let slot_stride = offset;
        let upscaled = committed_tex(
            device,
            ow,
            oh,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        let planes: [Plane; N_PLANES] = planes.try_into().map_err(|_| "plane count".to_string())?;
        Ok(Self {
            max_w,
            max_h,
            planes,
            upscaled,
            upload: std::sync::OnceLock::new(),
            device: device.clone(),
            slot_stride,
        })
    }

    fn upload_buf(&self) -> Result<&UploadBuffer> {
        get_or_try_init(&self.upload, || {
            UploadBuffer::new(&self.device, self.slot_stride * FRAMES_IN_FLIGHT)
        })
    }

    /// Convert this frame's `rw×rh` sub-rect into the slot's upload memory
    /// (rayon, disjoint row slices) and record the copies + barriers — the
    /// `xr.rs::record_upload` pattern over the FSR3 plane set. `accum` is
    /// the tracer's store-semantics buffer (in FSR mode it holds exactly
    /// the last 1-spp HDR frame).
    #[allow(clippy::too_many_arguments)]
    pub fn record_upload(
        &self,
        d3d: &D3d,
        slot: usize,
        accum: &[AtomicU32],
        g: &GBufs,
        rw: usize,
        rh: usize,
        near: f32,
        far: f32,
    ) -> Result<()> {
        crate::zone!("fsr-upload");
        let _ev = super::pix::scope(&d3d.list, c"fsr-upload");
        let upload = self.upload_buf()?;
        debug_assert_eq!((g.rw, g.rh), (rw, rh));
        debug_assert!(rw <= self.max_w as usize && rh <= self.max_h as usize);
        let base = slot * self.slot_stride;
        let w = rw;

        let plane_mem = |p: &Plane| -> &mut [u8] {
            let pitch = aligned_pitch(w * p.bpp);
            unsafe { std::slice::from_raw_parts_mut(upload.ptr.add(base + p.offset), pitch * rh) }
        };
        let ld32 = |b: &[AtomicU32], i: usize| f32::from_bits(b[i].load(Relaxed));

        // Linear HDR color -> RGBA16F, saturating (an extreme HDR value must
        // narrow to f16::MAX, never +inf — the wire discipline every f16
        // color plane in this codebase follows).
        plane_mem(&self.planes[P_COLOR])
            .par_chunks_mut(aligned_pitch(w * 8))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [[f16; 4]] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = (y * w + x) * 3;
                    p[0] = fsr::f16_sat(ld32(accum, i));
                    p[1] = fsr::f16_sat(ld32(accum, i + 1));
                    p[2] = fsr::f16_sat(ld32(accum, i + 2));
                    p[3] = f16::from_f32(0.0);
                }
            });

        // Motion vectors: f16 storage -> RG16F, bit copy. The plane keeps
        // GBufs::mvec's pixel-space current→previous y-down convention; the
        // dispatch's motionVectorScale is (UPSCALE_MV_SIGN.x, .y), so
        // post-scale FSR sees the same pixel-space MVs the RR flavor hands
        // it via (sign·rw)×UV-delta — one polarity constant, two flavors.
        plane_mem(&self.planes[P_MVEC])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [[f16; 2]] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = (y * w + x) * 2;
                    p[0] = f16::from_bits(g.mvec[i].load(Relaxed));
                    p[1] = f16::from_bits(g.mvec[i + 1].load(Relaxed));
                }
            });

        // Linear view-Z -> [0,1] reversed-Z clip depth (the XeSS
        // single-source encoder; the upscaler context is created with
        // DEPTH_INVERTED to match; sky's view_z = far lands exactly on 0).
        plane_mem(&self.planes[P_DEPTH])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    *p = xess::view_z_to_clip_depth(ld32(&g.depth, y * w + x), near, far);
                }
            });

        // Batch barriers to COPY_DEST, record the sub-rect copies, batch back.
        let to_copy: Vec<_> = self
            .planes
            .iter()
            .map(|p| {
                transition(
                    &p.tex,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )
            })
            .collect();
        unsafe { d3d.list.ResourceBarrier(&to_copy) };
        for p in &self.planes {
            let fp = footprint(p.format, rw as u32, rh as u32, p.bpp, (base + p.offset) as u64);
            unsafe {
                d3d.list.CopyTextureRegion(
                    &loc_subresource(&p.tex),
                    0,
                    0,
                    0,
                    &loc_footprint(&upload.resource, fp),
                    None,
                )
            };
        }
        let to_srv: Vec<_> = self
            .planes
            .iter()
            .map(|p| {
                transition(
                    &p.tex,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )
            })
            .collect();
        unsafe { d3d.list.ResourceBarrier(&to_srv) };
        Ok(())
    }

    fn shim(res: &ID3D12Resource, state: u32) -> FfxShimRes {
        FfxShimRes { resource: res.as_raw(), state }
    }

    /// The input planes as GPU feed targets, in the XeSS wiring order
    /// (color, mvec, depth) — the FSR3 feed IS the XeSS feed: same plane
    /// set, same formats, same depth encode, so `FeedKind::Fsr3` runs
    /// `cs_feed_xess` into these.
    pub fn plane_resources(&self) -> [(&ID3D12Resource, DXGI_FORMAT); N_PLANES] {
        [
            (&self.planes[P_COLOR].tex, self.planes[P_COLOR].format),
            (&self.planes[P_MVEC].tex, self.planes[P_MVEC].format),
            (&self.planes[P_DEPTH].tex, self.planes[P_DEPTH].format),
        ]
    }

    /// The upscale dispatch's resource references in (color, depth, mvec,
    /// output) order — the same tuple shape `ffx_rr.rs::upscale_res` hands
    /// the shared desc builder, in the ffx states the resources will be in
    /// when the recorded work executes (the caller wraps the dispatch in
    /// `barrier_upscale_*`).
    pub fn upscale_res(&self) -> (FfxShimRes, FfxShimRes, FfxShimRes, FfxShimRes) {
        (
            Self::shim(&self.planes[P_COLOR].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.planes[P_DEPTH].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.planes[P_MVEC].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.upscaled, RES_STATE_UNORDERED_ACCESS),
        )
    }

    pub fn barrier_upscale_begin(&self, list: &ID3D12GraphicsCommandList) {
        unsafe {
            list.ResourceBarrier(&[transition(
                &self.upscaled,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )])
        };
    }

    pub fn barrier_upscale_end(&self, list: &ID3D12GraphicsCommandList) {
        unsafe {
            list.ResourceBarrier(&[transition(
                &self.upscaled,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )])
        };
    }
}
