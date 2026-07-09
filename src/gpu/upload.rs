//! CPU→GPU frame uploads. Each path owns a default-heap texture plus a
//! persistently-mapped upload buffer with one 256-aligned slice per frame in
//! flight; rows are written on the CPU (rayon for the HDR conversion), then a
//! CopyTextureRegion + barrier pair is recorded. Textures live in
//! PIXEL_SHADER_RESOURCE between frames.

use super::d3d12::{
    aligned_pitch, committed_tex, footprint, loc_footprint, loc_subresource, transition, D3d,
    Result, UploadBuffer, FRAMES_IN_FLIGHT, SWAPCHAIN_FORMAT,
};
use half::f16;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The CPU-tonemapped u32 0RGB frame, presented via a B8G8R8A8 texture.
/// u32 0x00RRGGBB is little-endian bytes [B,G,R,0] — exactly B8G8R8A8 layout;
/// the blit shader forces alpha to 1.
pub struct BlitUpload {
    pub texture: ID3D12Resource,
    upload: UploadBuffer,
    pitch: usize,
    slice_size: usize,
    w: u32,
    h: u32,
}

impl BlitUpload {
    pub fn new(d3d: &D3d, w: u32, h: u32) -> Result<Self> {
        let texture = committed_tex(
            &d3d.device,
            w,
            h,
            SWAPCHAIN_FORMAT,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        let pitch = aligned_pitch(w as usize * 4);
        let slice_size = pitch * h as usize;
        let upload = UploadBuffer::new(&d3d.device, slice_size * FRAMES_IN_FLIGHT)?;
        Ok(Self { texture, upload, pitch, slice_size, w, h })
    }

    pub fn record(&self, d3d: &D3d, slot: usize, pixels: &[u32]) {
        debug_assert_eq!(pixels.len(), (self.w * self.h) as usize);
        let base = slot * self.slice_size;
        let dst = unsafe {
            std::slice::from_raw_parts_mut(self.upload.ptr.add(base), self.slice_size)
        };
        let row_bytes = self.w as usize * 4;
        for y in 0..self.h as usize {
            let src_row = &pixels[y * self.w as usize..(y + 1) * self.w as usize];
            let src_bytes =
                unsafe { std::slice::from_raw_parts(src_row.as_ptr() as *const u8, row_bytes) };
            dst[y * self.pitch..y * self.pitch + row_bytes].copy_from_slice(src_bytes);
        }
        let fp = footprint(SWAPCHAIN_FORMAT, self.w, self.h, 4, base as u64);
        unsafe {
            d3d.list.ResourceBarrier(&[transition(
                &self.texture,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            d3d.list.CopyTextureRegion(
                &loc_subresource(&self.texture),
                0,
                0,
                0,
                &loc_footprint(&self.upload.resource, fp),
                None,
            );
            d3d.list.ResourceBarrier(&[transition(
                &self.texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
    }
}

/// The linear-HDR accumulation (3 × f32 bits per pixel) uploaded as RGBA16F.
/// The GPU tonemap divides by the sample count, so the raw sum is uploaded.
pub struct HdrUpload {
    pub texture: ID3D12Resource,
    upload: UploadBuffer,
    pitch: usize,
    slice_size: usize,
    w: u32,
    h: u32,
}

impl HdrUpload {
    pub fn new(d3d: &D3d, w: u32, h: u32) -> Result<Self> {
        let texture = committed_tex(
            &d3d.device,
            w,
            h,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        let pitch = aligned_pitch(w as usize * 8);
        let slice_size = pitch * h as usize;
        let upload = UploadBuffer::new(&d3d.device, slice_size * FRAMES_IN_FLIGHT)?;
        Ok(Self { texture, upload, pitch, slice_size, w, h })
    }

    /// `accum` is the renderer's accumulation buffer (rw*rh*3 f32 bit
    /// patterns). Rows are converted in parallel; disjoint row slices make
    /// the parallel writes race-free.
    pub fn record(&self, d3d: &D3d, slot: usize, accum: &[AtomicU32]) {
        let (w, h) = (self.w as usize, self.h as usize);
        debug_assert!(accum.len() >= w * h * 3);
        let base = slot * self.slice_size;
        let dst =
            unsafe { std::slice::from_raw_parts_mut(self.upload.ptr.add(base), self.slice_size) };
        let one = f16::from_f32(1.0);
        dst.par_chunks_mut(self.pitch).take(h).enumerate().for_each(|(y, row)| {
            let row_px: &mut [[f16; 4]] = unsafe {
                std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut [f16; 4], w)
            };
            for (x, px) in row_px.iter_mut().enumerate() {
                let i = (y * w + x) * 3;
                px[0] = f16::from_f32(f32::from_bits(accum[i].load(Relaxed)));
                px[1] = f16::from_f32(f32::from_bits(accum[i + 1].load(Relaxed)));
                px[2] = f16::from_f32(f32::from_bits(accum[i + 2].load(Relaxed)));
                px[3] = one;
            }
        });
        let fp = footprint(DXGI_FORMAT_R16G16B16A16_FLOAT, self.w, self.h, 8, base as u64);
        unsafe {
            d3d.list.ResourceBarrier(&[transition(
                &self.texture,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            d3d.list.CopyTextureRegion(
                &loc_subresource(&self.texture),
                0,
                0,
                0,
                &loc_footprint(&self.upload.resource, fp),
                None,
            );
            d3d.list.ResourceBarrier(&[transition(
                &self.texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
    }
}
