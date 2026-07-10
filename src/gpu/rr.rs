//! DLSS Ray Reconstruction resources: one texture per SL buffer tag, a
//! single persistently-mapped upload ring (per-frame slots), the CPU→GPU
//! conversion loops, and the ResourceTag table handed to Streamline each
//! frame. Input planes are allocated once at the optimal-settings query's
//! range MAX; every frame uploads (and tags, via sl::Extent) only the
//! top-left `rw×rh` sub-rect — step-wise dynamic resolution, the same
//! allocate-max/name-a-sub-rect pattern as gpu/xr.rs. The output stays at
//! the window resolution with its own full extent.

use super::d3d12::{
    aligned_pitch, committed_tex, footprint, loc_footprint, loc_subresource, transition, D3d,
    Result, UploadBuffer, FRAMES_IN_FLIGHT,
};
use super::streamline_sys as sl;
use crate::dlss::GBufs;
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
    offset: usize, // into each upload slot
}

pub struct RrResources {
    /// Input plane allocation size — the DRS range max.
    pub max_w: u32,
    pub max_h: u32,
    pub ow: u32, // output res (RR output)
    pub oh: u32,
    /// The optimal-settings query result: optimal render size and the DRS
    /// range every dynamic frame must stay inside. opt == min == max when
    /// the query fell back (DLAA) — the degenerate range main.rs treats as
    /// "DRS off".
    pub opt: (u32, u32),
    pub min: (u32, u32),
    pub max: (u32, u32),
    planes: [Plane; 7], // upload order: color, normal_rough, depth, mvec, albedo, spec_albedo, spec_hit
    pub output: ID3D12Resource, // RGBA16F UAV, written by RR, read by the tonemap
    upload: UploadBuffer,
    slot_stride: usize,
}

const P_COLOR: usize = 0;
const P_NORMAL_ROUGH: usize = 1;
const P_DEPTH: usize = 2;
const P_MVEC: usize = 3;
const P_ALBEDO: usize = 4;
const P_SPEC_ALBEDO: usize = 5;
const P_SPEC_HIT: usize = 6;

impl RrResources {
    pub fn new(
        d3d: &D3d,
        opt: (u32, u32),
        min: (u32, u32),
        max: (u32, u32),
        ow: u32,
        oh: u32,
    ) -> Result<Self> {
        let (max_w, max_h) = max;
        let specs: [(DXGI_FORMAT, usize); 7] = [
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 8), // noisy color
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 8), // normal.xyz + roughness.w (packed mode)
            (DXGI_FORMAT_R32_FLOAT, 4),          // linear view-Z
            (DXGI_FORMAT_R16G16_FLOAT, 4),       // motion vectors (pixels)
            (DXGI_FORMAT_R8G8B8A8_UNORM, 4),     // diffuse albedo (linear)
            (DXGI_FORMAT_R8G8B8A8_UNORM, 4),     // specular albedo (linear)
            (DXGI_FORMAT_R16_FLOAT, 2),          // specular hit distance
        ];
        let mut offset = 0usize;
        let mut planes = Vec::with_capacity(7);
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
        let planes: [Plane; 7] = planes.try_into().map_err(|_| "plane count".to_string())?;
        Ok(Self { max_w, max_h, ow, oh, opt, min, max, planes, output, upload, slot_stride })
    }

    /// Convert the CPU frame's `rw×rh` sub-rect into the slot's upload
    /// memory (rayon, disjoint row slices) and record the copies + barriers.
    /// The footprint pitch is per-frame (aligned rw·bpp) — smaller frames
    /// copy proportionally less; a res step costs nothing on this path.
    /// Inputs live in NON_PIXEL_SHADER_RESOURCE between frames — exactly the
    /// state they are tagged with, which manual hooking requires to be
    /// correct at *use*.
    pub fn record_upload(
        &self,
        d3d: &D3d,
        slot: usize,
        accum: &[AtomicU32],
        g: &GBufs,
        rw: usize,
        rh: usize,
    ) {
        let (w, h) = (rw, rh);
        debug_assert_eq!((g.rw, g.rh), (w, h));
        debug_assert!(w <= self.max_w as usize && h <= self.max_h as usize);
        let base = slot * self.slot_stride;

        let plane_mem = |p: &Plane| -> &mut [u8] {
            let pitch = aligned_pitch(w * p.bpp);
            unsafe {
                std::slice::from_raw_parts_mut(self.upload.ptr.add(base + p.offset), pitch * h)
            }
        };
        let load = |b: &[AtomicU32], i: usize| f32::from_bits(b[i].load(Relaxed));
        let to_unorm8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;

        // Noisy HDR color: accum under store semantics (3 × f32) -> RGBA16F.
        plane_mem(&self.planes[P_COLOR])
            .par_chunks_mut(aligned_pitch(w * 8))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [[f16; 4]] =
                    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = (y * w + x) * 3;
                    p[0] = f16::from_f32(load(accum, i));
                    p[1] = f16::from_f32(load(accum, i + 1));
                    p[2] = f16::from_f32(load(accum, i + 2));
                    p[3] = f16::from_f32(1.0);
                }
            });

        // World normal + roughness, packed (normalRoughnessMode = ePacked).
        plane_mem(&self.planes[P_NORMAL_ROUGH])
            .par_chunks_mut(aligned_pitch(w * 8))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [[f16; 4]] =
                    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = (y * w + x) * 4;
                    p[0] = f16::from_f32(load(&g.normal_rough, i));
                    p[1] = f16::from_f32(load(&g.normal_rough, i + 1));
                    p[2] = f16::from_f32(load(&g.normal_rough, i + 2));
                    p[3] = f16::from_f32(load(&g.normal_rough, i + 3));
                }
            });

        // Linear view-Z: f32 bit patterns pass through untouched (R32_FLOAT).
        plane_mem(&self.planes[P_DEPTH])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [u32] =
                    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    *p = g.depth[y * w + x].load(Relaxed);
                }
            });

        // Motion vectors: 2 × f32 (pixels) -> RG16F.
        plane_mem(&self.planes[P_MVEC])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(h)
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

        // Diffuse albedo (linear RGBA8) and specular albedo (achromatic).
        plane_mem(&self.planes[P_ALBEDO])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    row[x * 4] = to_unorm8(load(&g.diff_alb, i));
                    row[x * 4 + 1] = to_unorm8(load(&g.diff_alb, i + 1));
                    row[x * 4 + 2] = to_unorm8(load(&g.diff_alb, i + 2));
                    row[x * 4 + 3] = 255;
                }
            });
        plane_mem(&self.planes[P_SPEC_ALBEDO])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    let s = to_unorm8(load(&g.spec_alb, y * w + x));
                    row[x * 4] = s;
                    row[x * 4 + 1] = s;
                    row[x * 4 + 2] = s;
                    row[x * 4 + 3] = 255;
                }
            });

        // Specular hit distance -> R16F.
        plane_mem(&self.planes[P_SPEC_HIT])
            .par_chunks_mut(aligned_pitch(w * 2))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [f16] =
                    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    *p = f16::from_f32(load(&g.spec_hit_t, y * w + x));
                }
            });

        // Batch barriers to COPY_DEST, record all copies, batch back.
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
            let fp = footprint(p.format, w as u32, h as u32, p.bpp, (base + p.offset) as u64);
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

    /// The full tag table for slEvaluateFeature(DLSS-RR). States must be the
    /// states the resources are in when SL *uses* them (manual hooking).
    /// Under DRS every input tag carries the current render-size extent
    /// `{0,0,rw,rh}` (the RR guide tags ALL inputs with the input extent)
    /// and the output its full window-size extent — SL derives the ratio
    /// and low-res-MV handling from these, not from the texture dims.
    pub fn tags(&self, rw: usize, rh: usize) -> [sl::SlShimResourceTag; 8] {
        let npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 as u32;
        let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;
        let tag = |tex: &ID3D12Resource, buffer_type: u32, state: u32, w: u32, h: u32| {
            sl::SlShimResourceTag {
                resource: tex.as_raw(),
                state,
                buffer_type,
                lifecycle: sl::LIFECYCLE_VALID_UNTIL_PRESENT,
                extent_top: 0,
                extent_left: 0,
                extent_width: w,
                extent_height: h,
            }
        };
        let (rw, rh) = (rw as u32, rh as u32);
        [
            tag(&self.planes[P_COLOR].tex, sl::BUFFER_SCALING_INPUT_COLOR, npsr, rw, rh),
            tag(&self.output, sl::BUFFER_SCALING_OUTPUT_COLOR, uav, self.ow, self.oh),
            tag(&self.planes[P_DEPTH].tex, sl::BUFFER_LINEAR_DEPTH, npsr, rw, rh),
            tag(&self.planes[P_MVEC].tex, sl::BUFFER_MOTION_VECTORS, npsr, rw, rh),
            tag(&self.planes[P_ALBEDO].tex, sl::BUFFER_ALBEDO, npsr, rw, rh),
            tag(&self.planes[P_SPEC_ALBEDO].tex, sl::BUFFER_SPECULAR_ALBEDO, npsr, rw, rh),
            tag(&self.planes[P_NORMAL_ROUGH].tex, sl::BUFFER_NORMAL_ROUGHNESS, npsr, rw, rh),
            tag(&self.planes[P_SPEC_HIT].tex, sl::BUFFER_SPECULAR_HIT_DISTANCE, npsr, rw, rh),
        ]
    }
}
