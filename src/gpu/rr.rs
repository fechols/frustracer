//! DLSS Ray Reconstruction resources: one texture per raw-NGX DLSSD eval
//! input, a single persistently-mapped upload ring (per-frame slots,
//! allocated lazily on first CPU upload — --gpu feed sessions write the
//! planes in place and never pay for it), and the CPU→GPU conversion loops.
//! Input planes are allocated once at the optimal-settings query's range
//! MAX; every frame uploads (and evaluates, via InRenderSubrectDimensions)
//! only the top-left `rw×rh` sub-rect — step-wise dynamic resolution, the
//! same allocate-max/name-a-sub-rect pattern as gpu/xr.rs. The output stays
//! at the window resolution.

use super::d3d12::{
    aligned_pitch, committed_tex, footprint, get_or_try_init, loc_footprint, loc_subresource,
    transition, D3d, Result, UploadBuffer, FRAMES_IN_FLIGHT,
};
use crate::dlss::{ld16, GBufs};
use half::f16;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering::Relaxed};
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
    // upload order: color, normal_rough, depth, mvec, albedo, spec_albedo,
    // spec_hit, ripple. The last is NOT an RR input — RR is never tagged with
    // it and never reads it; it exists so the frame-generation guide pass can
    // see which pixels carry a moving (rippling) mirror normal.
    planes: [Plane; 8],
    pub output: ID3D12Resource, // RGBA16F UAV, written by RR, read by the tonemap
    /// CPU-upload ring, lazily allocated on first `record_upload` — a --gpu
    /// feed session never touches it (the feed kernel writes the planes in
    /// place), so its slot_stride × FRAMES_IN_FLIGHT bytes stay uncommitted.
    upload: std::sync::OnceLock<UploadBuffer>,
    device: ID3D12Device,
    slot_stride: usize,
}

const P_COLOR: usize = 0;
const P_NORMAL_ROUGH: usize = 1;
const P_DEPTH: usize = 2;
const P_MVEC: usize = 3;
const P_ALBEDO: usize = 4;
const P_SPEC_ALBEDO: usize = 5;
const P_SPEC_HIT: usize = 6;
/// Water ripple amplitude — a guide-pass-only plane (see the `planes` doc).
const P_RIPPLE: usize = 7;

impl RrResources {
    pub fn new(
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        opt: (u32, u32),
        min: (u32, u32),
        max: (u32, u32),
        ow: u32,
        oh: u32,
    ) -> Result<Self> {
        let (max_w, max_h) = max;
        let specs: [(DXGI_FORMAT, usize); 8] = [
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 8), // noisy color
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 8), // normal.xyz + roughness.w (packed mode)
            (DXGI_FORMAT_R32_FLOAT, 4),          // linear view-Z
            (DXGI_FORMAT_R16G16_FLOAT, 4),       // motion vectors (pixels)
            (DXGI_FORMAT_R8G8B8A8_UNORM, 4),     // diffuse albedo (linear)
            (DXGI_FORMAT_R8G8B8A8_UNORM, 4),     // specular albedo (linear)
            (DXGI_FORMAT_R16_FLOAT, 2),          // specular hit distance
            (DXGI_FORMAT_R16_FLOAT, 2),          // ripple amp (FG guide pass only)
        ];
        let mut offset = 0usize;
        let mut planes = Vec::with_capacity(8);
        for (format, bpp) in specs {
            // ALLOW_UNORDERED_ACCESS: the --gpu feed kernel writes these
            // planes directly (the CPU path keeps the upload copies — the
            // rest state stays NON_PIXEL_SHADER_RESOURCE either way, which is
            // what the SL tags declare at evaluate).
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
        let output = committed_tex(
            device,
            ow,
            oh,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        let planes: [Plane; 8] = planes.try_into().map_err(|_| "plane count".to_string())?;
        Ok(Self {
            max_w,
            max_h,
            ow,
            oh,
            opt,
            min,
            max,
            planes,
            output,
            upload: std::sync::OnceLock::new(),
            device: device.clone(),
            slot_stride,
        })
    }

    /// The upload ring, allocated on first use.
    fn upload_buf(&self) -> Result<&UploadBuffer> {
        get_or_try_init(&self.upload, || {
            UploadBuffer::new(&self.device, self.slot_stride * FRAMES_IN_FLIGHT)
        })
    }

    /// The input planes as (resource, format) in the FEED register order
    /// (color, normal_rough, depth, mvec, albedo, spec_albedo, spec_hit —
    /// u16..u22) — the --gpu feed path's wiring/barrier handle.
    pub fn plane_resources(&self) -> [(&ID3D12Resource, DXGI_FORMAT); 8] {
        [
            (&self.planes[P_COLOR].tex, self.planes[P_COLOR].format),
            (&self.planes[P_NORMAL_ROUGH].tex, self.planes[P_NORMAL_ROUGH].format),
            (&self.planes[P_DEPTH].tex, self.planes[P_DEPTH].format),
            (&self.planes[P_MVEC].tex, self.planes[P_MVEC].format),
            (&self.planes[P_ALBEDO].tex, self.planes[P_ALBEDO].format),
            (&self.planes[P_SPEC_ALBEDO].tex, self.planes[P_SPEC_ALBEDO].format),
            (&self.planes[P_SPEC_HIT].tex, self.planes[P_SPEC_HIT].format),
            (&self.planes[P_RIPPLE].tex, self.planes[P_RIPPLE].format),
        ]
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
    ) -> Result<()> {
        crate::zone!("rr-upload");
        let _ev = super::pix::scope(&d3d.list, c"rr-upload");
        let upload = self.upload_buf()?;
        let (w, h) = (rw, rh);
        debug_assert_eq!((g.rw, g.rh), (w, h));
        debug_assert!(w <= self.max_w as usize && h <= self.max_h as usize);
        let base = slot * self.slot_stride;

        let plane_mem = |p: &Plane| -> &mut [u8] {
            let pitch = aligned_pitch(w * p.bpp);
            unsafe {
                std::slice::from_raw_parts_mut(upload.ptr.add(base + p.offset), pitch * h)
            }
        };
        let load = |b: &[AtomicU32], i: usize| f32::from_bits(b[i].load(Relaxed));
        // f16-stored planes go to their f16 textures as raw bit copies.
        let bits16 = |b: &[AtomicU16], i: usize| f16::from_bits(b[i].load(Relaxed));
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
        // Storage is already f16 — a straight bit copy.
        plane_mem(&self.planes[P_NORMAL_ROUGH])
            .par_chunks_mut(aligned_pitch(w * 8))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [[f16; 4]] =
                    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = (y * w + x) * 4;
                    p[0] = bits16(&g.normal_rough, i);
                    p[1] = bits16(&g.normal_rough, i + 1);
                    p[2] = bits16(&g.normal_rough, i + 2);
                    p[3] = bits16(&g.normal_rough, i + 3);
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

        // Motion vectors (pixels): f16 storage -> RG16F, bit copy.
        plane_mem(&self.planes[P_MVEC])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [[f16; 2]] =
                    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = (y * w + x) * 2;
                    p[0] = bits16(&g.mvec, i);
                    p[1] = bits16(&g.mvec, i + 1);
                }
            });

        // Diffuse albedo (linear RGBA8) and specular albedo (RGB F0).
        plane_mem(&self.planes[P_ALBEDO])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    row[x * 4] = to_unorm8(ld16(&g.diff_alb[i]));
                    row[x * 4 + 1] = to_unorm8(ld16(&g.diff_alb[i + 1]));
                    row[x * 4 + 2] = to_unorm8(ld16(&g.diff_alb[i + 2]));
                    row[x * 4 + 3] = 255;
                }
            });
        plane_mem(&self.planes[P_SPEC_ALBEDO])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    row[x * 4] = to_unorm8(ld16(&g.spec_alb[i]));
                    row[x * 4 + 1] = to_unorm8(ld16(&g.spec_alb[i + 1]));
                    row[x * 4 + 2] = to_unorm8(ld16(&g.spec_alb[i + 2]));
                    row[x * 4 + 3] = 255;
                }
            });

        // Specular hit distance: f16 storage -> R16F, bit copy.
        plane_mem(&self.planes[P_SPEC_HIT])
            .par_chunks_mut(aligned_pitch(w * 2))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [f16] =
                    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    *p = bits16(&g.spec_hit_t, y * w + x);
                }
            });

        // Ripple amplitude: f16 storage -> R16F, bit copy. RR never reads
        // this (it is not among the tags) — the frame-generation guide pass
        // does, to find the pixels whose mirror normal moves between frames.
        plane_mem(&self.planes[P_RIPPLE])
            .par_chunks_mut(aligned_pitch(w * 2))
            .take(h)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [f16] =
                    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    *p = bits16(&g.ripple_amp, y * w + x);
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

    /// The raw-NGX frame-generation inputs: (depth, mvec) — the same linear
    /// view-Z + pixel-space MV planes the SL tags name, resting
    /// NON_PIXEL_SHADER_RESOURCE.
    pub fn fg_inputs(&self) -> (&ID3D12Resource, &ID3D12Resource) {
        (&self.planes[P_DEPTH].tex, &self.planes[P_MVEC].tex)
    }

    /// The guide-conversion pass's source planes, in `ngxfg_guides`' kernel
    /// order (its `SRC_FORMATS` mirrors these planes' formats — change both
    /// together). All rest NON_PIXEL_SHADER_RESOURCE, the state a compute
    /// SRV read wants.
    pub fn guide_inputs(&self) -> [&ID3D12Resource; crate::gpu::ngxfg_guides::SRC_PLANES] {
        [
            &self.planes[P_DEPTH].tex,
            &self.planes[P_MVEC].tex,
            &self.planes[P_SPEC_HIT].tex,
            &self.planes[P_ALBEDO].tex,
            &self.planes[P_SPEC_ALBEDO].tex,
            &self.planes[P_NORMAL_ROUGH].tex,
            &self.planes[P_RIPPLE].tex,
        ]
    }

}
