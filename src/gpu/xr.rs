//! XeSS-SR resources: three input planes (color, motion vectors, depth), a
//! window-res output UAV, and the persistently-mapped upload ring — the
//! `rr.rs` structure minus Streamline. The defining difference from DLSS-RR:
//! the input planes are allocated once at the *maximum* input resolution the
//! range query allows, and every frame uploads (and XeSS reads, via
//! `input_width/height`) only the top-left `rw×rh` sub-rect — dynamic
//! resolution as the SDK intends it, no reallocation on a res step.

use super::d3d12::{
    aligned_pitch, committed_tex, footprint, loc_footprint, loc_subresource, transition, D3d,
    ReadbackBuffer, Result, UploadBuffer, FRAMES_IN_FLIGHT,
};
use crate::dlss::GBufs;
use crate::xess;
use half::f16;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// Per-frame color input: either the raw 1-spp frame straight from the
/// tracer's store-semantics accum (`accum` in XeSS mode holds exactly the
/// last frame) or an already-denoised linear-HDR image (the OIDN pre-pass).
pub enum ColorSrc<'a> {
    Accum(&'a [AtomicU32]),
    Hdr(&'a [f32]),
}

struct Plane {
    tex: ID3D12Resource,
    format: DXGI_FORMAT,
    bpp: usize,
    offset: usize, // into each upload slot, sized for the max input res
}

pub struct XessResources {
    pub max_w: u32, // input plane allocation size (range max)
    pub max_h: u32,
    ow: u32, // output res (window)
    oh: u32,
    planes: [Plane; 3], // color, mvec, depth
    pub output: ID3D12Resource, // RGBA16F UAV, written by XeSS, read by the tonemap
    upload: UploadBuffer,
    slot_stride: usize,
    /// Persistent window-res staging for the post-upscale denoise path —
    /// the upscaled HDR is pulled back to the CPU every frame there, so the
    /// buffer must not be a per-call allocation like the screenshot one.
    readback: ReadbackBuffer,
}

const P_COLOR: usize = 0;
const P_MVEC: usize = 1;
const P_DEPTH: usize = 2;

impl XessResources {
    pub fn new(d3d: &D3d, max_w: u32, max_h: u32, ow: u32, oh: u32) -> Result<Self> {
        let specs: [(DXGI_FORMAT, usize); 3] = [
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 8), // linear HDR color (1-spp or denoised)
            (DXGI_FORMAT_R16G16_FLOAT, 4),       // motion vectors (input-res pixels)
            (DXGI_FORMAT_R32_FLOAT, 4),          // [0,1] clip depth
        ];
        let mut offset = 0usize;
        let mut planes = Vec::with_capacity(3);
        for (format, bpp) in specs {
            let tex = committed_tex(
                &d3d.device,
                max_w,
                max_h,
                format,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )?;
            planes.push(Plane { tex, format, bpp, offset });
            offset += aligned_pitch(max_w as usize * bpp) * max_h as usize;
        }
        let slot_stride = offset;
        let upload = UploadBuffer::new(&d3d.device, slot_stride * FRAMES_IN_FLIGHT)?;
        let output = committed_tex(
            &d3d.device,
            ow,
            oh,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        let planes: [Plane; 3] = planes.try_into().map_err(|_| "plane count".to_string())?;
        let readback =
            ReadbackBuffer::new(&d3d.device, aligned_pitch(ow as usize * 8) * oh as usize)?;
        Ok(Self { max_w, max_h, ow, oh, planes, output, upload, slot_stride, readback })
    }

    /// Copy the window-res output into the persistent readback buffer —
    /// recorded after the XeSS dispatch when the caller wants the upscaled
    /// HDR on the CPU (post-upscale denoise) instead of a backbuffer pass.
    /// Leaves `output` in PIXEL_SHADER_RESOURCE, its rest state.
    pub fn record_readback(&self, d3d: &D3d) {
        unsafe {
            d3d.list.ResourceBarrier(&[transition(
                &self.output,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let fp = footprint(DXGI_FORMAT_R16G16B16A16_FLOAT, self.ow, self.oh, 8, 0);
            d3d.list.CopyTextureRegion(
                &loc_footprint(&self.readback.resource, fp),
                0,
                0,
                0,
                &loc_subresource(&self.output),
                None,
            );
            d3d.list.ResourceBarrier(&[transition(
                &self.output,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
    }

    /// Map the readback buffer and expand its RGBA16F rows into linear-RGB
    /// f32 triples. Only valid after the submission that recorded
    /// `record_readback` completed (D3d::submit_and_wait).
    pub fn read_back(&self, out: &mut [f32]) -> Result<()> {
        let (w, h) = (self.ow as usize, self.oh as usize);
        assert_eq!(out.len(), w * h * 3);
        let pitch = aligned_pitch(w * 8);
        let mut ptr = std::ptr::null_mut();
        unsafe { self.readback.resource.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| format!("readback Map: {e}"))?;
        let base = ptr as usize; // usize crosses the rayon closure; rows are disjoint
        out.par_chunks_mut(w * 3).take(h).enumerate().for_each(|(y, row)| {
            let src: &[[f16; 4]] = unsafe {
                std::slice::from_raw_parts((base as *const u8).add(y * pitch) as *const _, w)
            };
            for (x, px) in src.iter().enumerate() {
                row[x * 3] = f32::from(px[0]);
                row[x * 3 + 1] = f32::from(px[1]);
                row[x * 3 + 2] = f32::from(px[2]);
            }
        });
        unsafe { self.readback.resource.Unmap(0, None) };
        Ok(())
    }

    /// Convert this frame's `rw×rh` sub-rect into the slot's upload memory
    /// (rayon, disjoint row slices) and record the copies + barriers. The
    /// footprint pitch is per-frame (aligned rw·bpp), so smaller frames copy
    /// proportionally less — a res step costs nothing on this path.
    pub fn record_upload(
        &self,
        d3d: &D3d,
        slot: usize,
        color: &ColorSrc,
        g: &GBufs,
        rw: usize,
        rh: usize,
        near: f32,
        far: f32,
    ) {
        debug_assert_eq!((g.rw, g.rh), (rw, rh));
        debug_assert!(rw <= self.max_w as usize && rh <= self.max_h as usize);
        let base = slot * self.slot_stride;
        let w = rw;

        let plane_mem = |p: &Plane| -> &mut [u8] {
            let pitch = aligned_pitch(w * p.bpp);
            unsafe {
                std::slice::from_raw_parts_mut(self.upload.ptr.add(base + p.offset), pitch * rh)
            }
        };
        let load = |b: &[AtomicU32], i: usize| f32::from_bits(b[i].load(Relaxed));

        // Linear HDR color -> RGBA16F.
        plane_mem(&self.planes[P_COLOR])
            .par_chunks_mut(aligned_pitch(w * 8))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [[f16; 4]] =
                    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = (y * w + x) * 3;
                    let (r, g_, b) = match color {
                        ColorSrc::Accum(a) => (load(a, i), load(a, i + 1), load(a, i + 2)),
                        ColorSrc::Hdr(h) => (h[i], h[i + 1], h[i + 2]),
                    };
                    p[0] = f16::from_f32(r);
                    p[1] = f16::from_f32(g_);
                    p[2] = f16::from_f32(b);
                    p[3] = f16::from_f32(1.0);
                }
            });

        // Motion vectors: 2 × f32 (input-res pixels) -> RG16F.
        plane_mem(&self.planes[P_MVEC])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [[f16; 2]] =
                    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = (y * w + x) * 2;
                    p[0] = f16::from_f32(load(&g.mvec, i));
                    p[1] = f16::from_f32(load(&g.mvec, i + 1));
                }
            });

        // Linear view-Z -> [0,1] reversed-Z clip depth (xess::view_z_to_clip_depth
        // is the single source; sky's view_z = far lands exactly on 0).
        plane_mem(&self.planes[P_DEPTH])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [f32] =
                    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let z = load(&g.depth, y * w + x);
                    *p = xess::view_z_to_clip_depth(z, near, far);
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
                    &loc_footprint(&self.upload.resource, fp),
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
    }

    /// Raw resource pointers for the execute params (color, mvec, depth).
    pub fn input_ptrs(&self) -> (*mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void)
    {
        use windows::core::Interface;
        (
            self.planes[P_COLOR].tex.as_raw(),
            self.planes[P_MVEC].tex.as_raw(),
            self.planes[P_DEPTH].tex.as_raw(),
        )
    }
}
