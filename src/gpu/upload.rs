//! CPU→GPU frame uploads. Each path owns a default-heap texture plus a
//! persistently-mapped upload buffer with one 256-aligned slice per frame in
//! flight; rows are written on the CPU (rayon for the HDR conversion), then a
//! CopyTextureRegion + barrier pair is recorded. Textures live in
//! PIXEL_SHADER_RESOURCE between frames.

use super::d3d12::{
    aligned_pitch, committed_tex, footprint, loc_footprint, loc_subresource, transition, D3d,
    Result, UploadBuffer, BLIT_FORMAT, BLIT_FORMAT_10BIT, FRAMES_IN_FLIGHT,
};
use half::f16;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The already-display-encoded CPU frame. Two wire formats, one texture:
///
/// - **SDR** (`BLIT_FORMAT`): u32 `0x00RRGGBB`, whose little-endian bytes are
///   [B,G,R,0] — exactly B8G8R8A8, so the frame memcpy's row-by-row with no
///   swizzle. The blit shader forces alpha to 1.
/// - **10-bit** (`BLIT_FORMAT_10BIT`, Sdr10 AND Hdr10 sessions): u32
///   `A2:B10:G10:R10` (R in the LOW bits — R10G10B10A2's lane order, opposite
///   BGRA8's) straight from `render::present_px_sdr10` / `present_px_pq`,
///   which already applied the curve, the overlay, and the encode (gamma-2.2
///   or matrix + ST 2084 respectively).
///
/// Either way the blit PS is a passthrough — the CPU owns the encode on this
/// path, which is what keeps the debug overlay's display-space semantics exact.
pub struct BlitUpload {
    pub texture: ID3D12Resource,
    pub format: DXGI_FORMAT,
    upload: UploadBuffer,
    pitch: usize,
    slice_size: usize,
    bpp: usize,
    w: u32,
    h: u32,
}

impl BlitUpload {
    pub fn new(d3d: &D3d, w: u32, h: u32) -> Result<Self> {
        Self::with_format(d3d, w, h, BLIT_FORMAT, 4)
    }

    /// The 10-bit variant, for an Sdr10/Hdr10 session's CPU-presented arms.
    pub fn new_10bit(d3d: &D3d, w: u32, h: u32) -> Result<Self> {
        Self::with_format(d3d, w, h, BLIT_FORMAT_10BIT, 4)
    }

    fn with_format(d3d: &D3d, w: u32, h: u32, format: DXGI_FORMAT, bpp: usize) -> Result<Self> {
        let texture = committed_tex(
            &d3d.device,
            w,
            h,
            format,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        let pitch = aligned_pitch(w as usize * bpp);
        let slice_size = pitch * h as usize;
        let upload = UploadBuffer::new(&d3d.device, slice_size * FRAMES_IN_FLIGHT)?;
        Ok(Self { texture, format, upload, pitch, slice_size, bpp, w, h })
    }

    pub fn record(&self, d3d: &D3d, slot: usize, pixels: &[u32]) {
        debug_assert_eq!(self.format, BLIT_FORMAT, "u32 0RGB into a non-B8G8R8A8 blit texture");
        self.copy_rows(d3d, slot, unsafe {
            std::slice::from_raw_parts(pixels.as_ptr() as *const u8, std::mem::size_of_val(pixels))
        });
    }

    /// Packed 10-bit, for the Sdr10/Hdr10 CPU present arms
    /// (`render::present_px_sdr10` / `present_px_pq` own the pack:
    /// `r | g<<10 | b<<20 | 3<<30`).
    pub fn record_10bit(&self, d3d: &D3d, slot: usize, pixels: &[u32]) {
        debug_assert_eq!(
            self.format, BLIT_FORMAT_10BIT,
            "packed 10-bit into a non-R10G10B10A2 blit texture"
        );
        self.copy_rows(d3d, slot, unsafe {
            std::slice::from_raw_parts(pixels.as_ptr() as *const u8, std::mem::size_of_val(pixels))
        });
    }

    /// Row-by-row memcpy into the frame's upload slice, then the copy + barrier
    /// pair. Format-agnostic: the caller's bytes are already what the texture
    /// wants, so the only thing that varies is the row stride.
    fn copy_rows(&self, d3d: &D3d, slot: usize, src: &[u8]) {
        let row_bytes = self.w as usize * self.bpp;
        debug_assert_eq!(src.len(), row_bytes * self.h as usize);
        let base = slot * self.slice_size;
        let dst = unsafe {
            std::slice::from_raw_parts_mut(self.upload.ptr.add(base), self.slice_size)
        };
        for y in 0..self.h as usize {
            dst[y * self.pitch..y * self.pitch + row_bytes]
                .copy_from_slice(&src[y * row_bytes..(y + 1) * row_bytes]);
        }
        let fp = footprint(self.format, self.w, self.h, self.bpp, base as u64);
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
