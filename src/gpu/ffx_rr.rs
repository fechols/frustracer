//! FSR-mode resources: the Ray Regeneration input planes, the two denoised
//! signal UAVs, the remodulation composite pass, and the FSR4 upscaled
//! output — `xr.rs`'s structure with ffx conventions. Like XeSS, dynamic
//! input resolution is first-class: every plane is allocated once at the
//! range MAX and each frame uploads (and both ffx dispatches read, via their
//! per-dispatch `renderSize`) only the top-left `rw×rh` sub-rect — no
//! reallocation on a res step. All wire encodings here (UV-delta MVs, oct
//! normals, sqrt albedos, reversed-Z clip depth) have their pure twins in
//! fsr.rs / xess.rs and are gated by --check-fsr / --check-xess.

use super::d3d12::{
    aligned_pitch, committed_tex, footprint, get_or_try_init, loc_footprint, loc_subresource,
    transition, D3d, Result, UploadBuffer, FRAMES_IN_FLIGHT,
};
use super::ffx_sys::{FfxShimRes, RES_STATE_COMPUTE_READ, RES_STATE_UNORDERED_ACCESS};
use super::tonemap;
use crate::dlss::GBufs;
use crate::fsr::{self, FsrBufs};
use crate::xess;
use half::f16;
use rayon::prelude::*;
use windows::core::{s, Interface};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use std::sync::atomic::Ordering::Relaxed;

const COMPOSITE_HLSL: &str = include_str!("shaders/fsr_composite.hlsl");

struct Plane {
    tex: ID3D12Resource,
    format: DXGI_FORMAT,
    bpp: usize,
    offset: usize, // into each upload slot, sized for the max input res
}

// Upload plane order.
const P_DEPTH_LIN: usize = 0; // R32F signed linear view-Z (denoiser)
const P_DEPTH_CLIP: usize = 1; // R32F reversed-Z clip depth (upscaler)
const P_MVEC: usize = 2; // RGBA16F: RG UV-delta prev-cur, B prevZ-curZ
const P_NORMALS: usize = 3; // RGB10A2: oct normal + roughness + mat type
const P_DIFF_ALB: usize = 4; // RGBA8 sqrt-encoded kd
const P_SPEC_ALB: usize = 5; // RGBA8 sqrt-encoded F0
const P_DD_IN: usize = 6; // RGBA16F demodulated direct diffuse
const P_DS_IN: usize = 7; // RGBA16F demodulated direct specular
const P_RESIDUAL: usize = 8; // RGBA16F pass-through (composite input only)
const N_PLANES: usize = 9;

pub struct FsrResources {
    pub max_w: u32, // input plane allocation size (range max)
    pub max_h: u32,
    planes: [Plane; N_PLANES],
    /// Denoised signal UAVs (max render size; ffx writes the sub-rect).
    dd_out: ID3D12Resource,
    ds_out: ID3D12Resource,
    /// Remodulated render-res color (composite CS output, upscaler input).
    composite: ID3D12Resource,
    /// FSR4 output at window res — the tonemap SRV target (screenshots read
    /// it back through the shared GpuContext::read_hdr_output path).
    pub upscaled: ID3D12Resource,
    upload: std::sync::OnceLock<UploadBuffer>,
    device: ID3D12Device,
    slot_stride: usize,
    // Composite pass state.
    comp_root: ID3D12RootSignature,
    comp_pso: ID3D12PipelineState,
    /// 5 SRVs (dd_out, ds_out, diff_alb, spec_alb, residual) + 1 UAV
    /// (composite), created once — the textures never change.
    comp_heap: ID3D12DescriptorHeap,
}

impl FsrResources {
    pub fn new(device: &ID3D12Device, max_w: u32, max_h: u32, ow: u32, oh: u32) -> Result<Self> {
        let specs: [(DXGI_FORMAT, usize); N_PLANES] = [
            (DXGI_FORMAT_R32_FLOAT, 4),
            (DXGI_FORMAT_R32_FLOAT, 4),
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 8),
            (DXGI_FORMAT_R10G10B10A2_UNORM, 4),
            (DXGI_FORMAT_R8G8B8A8_UNORM, 4),
            (DXGI_FORMAT_R8G8B8A8_UNORM, 4),
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 8),
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 8),
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 8),
        ];
        let mut offset = 0usize;
        let mut planes = Vec::with_capacity(N_PLANES);
        for (format, bpp) in specs {
            let tex = committed_tex(
                device,
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
        let signal_uav = || {
            committed_tex(
                device,
                max_w,
                max_h,
                DXGI_FORMAT_R16G16B16A16_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )
        };
        let dd_out = signal_uav()?;
        let ds_out = signal_uav()?;
        let composite = signal_uav()?;
        let upscaled = committed_tex(
            device,
            ow,
            oh,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        let planes: [Plane; N_PLANES] = planes.try_into().map_err(|_| "plane count".to_string())?;

        // Composite pass: table of 5 SRVs + 1 UAV, two root constants (rw, rh).
        let ranges = [
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 5,
                BaseShaderRegister: 0,
                ..Default::default()
            },
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                NumDescriptors: 1,
                BaseShaderRegister: 0,
                OffsetInDescriptorsFromTableStart: 5,
                ..Default::default()
            },
        ];
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: ranges.len() as u32,
                        pDescriptorRanges: ranges.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 2,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let sig_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            ..Default::default()
        };
        let mut blob = None;
        let mut errs = None;
        unsafe {
            D3D12SerializeRootSignature(&sig_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut errs))
        }
        .map_err(|e| format!("fsr composite root sig serialize: {e}"))?;
        let blob = blob.unwrap();
        let comp_root: ID3D12RootSignature = unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()),
            )
        }
        .map_err(|e| format!("fsr composite root sig: {e}"))?;

        let cs = tonemap::compile(COMPOSITE_HLSL, s!("cs"), s!("cs_5_0"), "fsr_composite")?;
        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: unsafe { std::mem::transmute_copy(&comp_root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: unsafe { cs.GetBufferPointer() },
                BytecodeLength: unsafe { cs.GetBufferSize() },
            },
            ..Default::default()
        };
        let comp_pso: ID3D12PipelineState = unsafe { device.CreateComputePipelineState(&pso_desc) }
            .map_err(|e| format!("fsr composite PSO: {e}"))?;

        let comp_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 6,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("fsr composite heap: {e}"))?;
        let inc = unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
        let cpu0 = unsafe { comp_heap.GetCPUDescriptorHandleForHeapStart() };
        let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu0.ptr + (i * inc) as usize };
        let srv = |res: &ID3D12Resource, format: DXGI_FORMAT, i: u32| {
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                },
            };
            unsafe { device.CreateShaderResourceView(res, Some(&desc), at(i)) };
        };
        srv(&dd_out, DXGI_FORMAT_R16G16B16A16_FLOAT, 0);
        srv(&ds_out, DXGI_FORMAT_R16G16B16A16_FLOAT, 1);
        srv(&planes[P_DIFF_ALB].tex, DXGI_FORMAT_R8G8B8A8_UNORM, 2);
        srv(&planes[P_SPEC_ALB].tex, DXGI_FORMAT_R8G8B8A8_UNORM, 3);
        srv(&planes[P_RESIDUAL].tex, DXGI_FORMAT_R16G16B16A16_FLOAT, 4);
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        unsafe { device.CreateUnorderedAccessView(&composite, None, Some(&uav_desc), at(5)) };

        Ok(Self {
            max_w,
            max_h,
            planes,
            dd_out,
            ds_out,
            composite,
            upscaled,
            upload: std::sync::OnceLock::new(),
            device: device.clone(),
            slot_stride,
            comp_root,
            comp_pso,
            comp_heap,
        })
    }

    fn upload_buf(&self) -> Result<&UploadBuffer> {
        get_or_try_init(&self.upload, || {
            UploadBuffer::new(&self.device, self.slot_stride * FRAMES_IN_FLIGHT)
        })
    }

    /// Convert this frame's `rw×rh` sub-rect into the slot's upload memory
    /// (rayon, disjoint row slices) and record the copies + barriers — the
    /// `xr.rs::record_upload` pattern over the ffx plane set.
    #[allow(clippy::too_many_arguments)]
    pub fn record_upload(
        &self,
        d3d: &D3d,
        slot: usize,
        g: &GBufs,
        f: &FsrBufs,
        rw: usize,
        rh: usize,
        near: f32,
        far: f32,
    ) -> Result<()> {
        crate::zone!("fsr-upload");
        let _ev = super::pix::scope(&d3d.list, c"fsr-upload");
        let upload = self.upload_buf()?;
        debug_assert_eq!((g.rw, g.rh), (rw, rh));
        debug_assert_eq!((f.rw, f.rh), (rw, rh));
        debug_assert!(rw <= self.max_w as usize && rh <= self.max_h as usize);
        let base = slot * self.slot_stride;
        let w = rw;

        let plane_mem = |p: &Plane| -> &mut [u8] {
            let pitch = aligned_pitch(w * p.bpp);
            unsafe { std::slice::from_raw_parts_mut(upload.ptr.add(base + p.offset), pitch * rh) }
        };
        let ld32 = |b: &[std::sync::atomic::AtomicU32], i: usize| f32::from_bits(b[i].load(Relaxed));
        let ld16 = crate::dlss::ld16;

        // Signed linear view-Z (denoiser) — DEPTH_SIGN is the polarity knob.
        plane_mem(&self.planes[P_DEPTH_LIN])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    *p = fsr::DEPTH_SIGN * ld32(&g.depth, y * w + x);
                }
            });

        // Reversed-Z clip depth (upscaler) — the XeSS single-source encoder;
        // the upscaler context is created with DEPTH_INVERTED to match.
        plane_mem(&self.planes[P_DEPTH_CLIP])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    *p = xess::view_z_to_clip_depth(ld32(&g.depth, y * w + x), near, far);
                }
            });

        // Motion vectors: our pixel-space current→previous MVs are exactly
        // the denoiser's PreviousUV − CurrentUV after a (1/rw, 1/rh) scale
        // (same direction, same y-down axis); B carries the linear-depth
        // delta from the captured prev_z.
        plane_mem(&self.planes[P_MVEC])
            .par_chunks_mut(aligned_pitch(w * 8))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [[f16; 4]] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = y * w + x;
                    let mv = (ld16(&g.mvec[i * 2]), ld16(&g.mvec[i * 2 + 1]));
                    let dz = ld32(&f.prev_z, i) - ld32(&g.depth, i);
                    p[0] = f16::from_f32(mv.0 / rw as f32);
                    p[1] = f16::from_f32(mv.1 / rh as f32);
                    p[2] = f16::from_f32(dz);
                    p[3] = f16::from_f32(0.0);
                }
            });

        // Octahedral normal + roughness + material type -> RGB10A2.
        plane_mem(&self.planes[P_NORMALS])
            .par_chunks_mut(aligned_pitch(w * 4))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = y * w + x;
                    let n = glam::Vec3A::new(
                        ld16(&g.normal_rough[i * 4]),
                        ld16(&g.normal_rough[i * 4 + 1]),
                        ld16(&g.normal_rough[i * 4 + 2]),
                    );
                    let rough = ld16(&g.normal_rough[i * 4 + 3]);
                    let (u, v) = fsr::oct_encode(n);
                    let q10 = |v: f32| (v.clamp(0.0, 1.0) * 1023.0 + 0.5) as u32;
                    let q2 = |v: f32| (v.clamp(0.0, 1.0) * 3.0 + 0.5) as u32;
                    *p = q10(u) | (q10(v) << 10) | (q10(rough) << 20) | (q2(fsr::MAT_TYPE) << 30);
                }
            });

        // sqrt-encoded albedos -> RGBA8 (fsr::sqrt_encode8 is the twin the
        // CPU identity gate and composite.hlsl's decode mirror).
        for (plane, buf) in [(P_DIFF_ALB, &g.diff_alb), (P_SPEC_ALB, &g.spec_alb)] {
            plane_mem(&self.planes[plane])
                .par_chunks_mut(aligned_pitch(w * 4))
                .take(rh)
                .enumerate()
                .for_each(|(y, row)| {
                    let px: &mut [[u8; 4]] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                    for (x, p) in px.iter_mut().enumerate() {
                        let i = (y * w + x) * 3;
                        p[0] = fsr::sqrt_encode8(ld16(&buf[i]));
                        p[1] = fsr::sqrt_encode8(ld16(&buf[i + 1]));
                        p[2] = fsr::sqrt_encode8(ld16(&buf[i + 2]));
                        p[3] = 255;
                    }
                });
        }

        // Signals: f16 storage -> RGBA16F, bit copy (A undefined on input).
        for (plane, buf) in [(P_DD_IN, &f.dd), (P_DS_IN, &f.ds)] {
            plane_mem(&self.planes[plane])
                .par_chunks_mut(aligned_pitch(w * 8))
                .take(rh)
                .enumerate()
                .for_each(|(y, row)| {
                    let px: &mut [[f16; 4]] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                    for (x, p) in px.iter_mut().enumerate() {
                        let i = (y * w + x) * 3;
                        p[0] = f16::from_bits(buf[i].load(Relaxed));
                        p[1] = f16::from_bits(buf[i + 1].load(Relaxed));
                        p[2] = f16::from_bits(buf[i + 2].load(Relaxed));
                        p[3] = f16::from_f32(0.0);
                    }
                });
        }

        // Residual: f32 storage -> RGBA16F (the identity's only wire
        // rounding; saturating so an extreme HDR remainder never becomes inf
        // on the wire).
        plane_mem(&self.planes[P_RESIDUAL])
            .par_chunks_mut(aligned_pitch(w * 8))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [[f16; 4]] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = (y * w + x) * 3;
                    p[0] = fsr::f16_sat(ld32(&f.residual, i));
                    p[1] = fsr::f16_sat(ld32(&f.residual, i + 1));
                    p[2] = fsr::f16_sat(ld32(&f.residual, i + 2));
                    p[3] = f16::from_f32(0.0);
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

    /// The denoiser dispatch's resource references, in the ffx states the
    /// resources will actually be in when the recorded work executes (inputs
    /// NON_PIXEL_SHADER_RESOURCE = compute read; outputs UAV — the caller
    /// wraps the dispatch in `barrier_denoise_*`).
    pub fn denoise_res(
        &self,
    ) -> (FfxShimRes, FfxShimRes, FfxShimRes, FfxShimRes, FfxShimRes, FfxShimRes, FfxShimRes, FfxShimRes, FfxShimRes) {
        (
            Self::shim(&self.planes[P_DEPTH_LIN].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.planes[P_MVEC].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.planes[P_NORMALS].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.planes[P_SPEC_ALB].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.planes[P_DIFF_ALB].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.planes[P_DD_IN].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.dd_out, RES_STATE_UNORDERED_ACCESS),
            Self::shim(&self.planes[P_DS_IN].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.ds_out, RES_STATE_UNORDERED_ACCESS),
        )
    }

    /// The upscale dispatch's resource references (color = composite output,
    /// clip depth, the shared MV plane, the window-res output UAV).
    pub fn upscale_res(&self) -> (FfxShimRes, FfxShimRes, FfxShimRes, FfxShimRes) {
        (
            Self::shim(&self.composite, RES_STATE_COMPUTE_READ),
            Self::shim(&self.planes[P_DEPTH_CLIP].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.planes[P_MVEC].tex, RES_STATE_COMPUTE_READ),
            Self::shim(&self.upscaled, RES_STATE_UNORDERED_ACCESS),
        )
    }

    pub fn barrier_denoise_begin(&self, list: &ID3D12GraphicsCommandList) {
        unsafe {
            list.ResourceBarrier(&[
                transition(&self.dd_out, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition(&self.ds_out, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ])
        };
    }

    pub fn barrier_denoise_end(&self, list: &ID3D12GraphicsCommandList) {
        unsafe {
            list.ResourceBarrier(&[
                transition(&self.dd_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition(&self.ds_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ])
        };
    }

    /// Record the remodulation compute pass over the `rw×rh` sub-rect.
    /// Binds everything from scratch (heap, root sig, PSO) — this doubles as
    /// the state restore after the ffx dispatch, same rationale as
    /// `Passes::record`.
    pub fn record_composite(&self, list: &ID3D12GraphicsCommandList, rw: u32, rh: u32) {
        let _ev = super::pix::scope(list, c"fsr-composite");
        unsafe {
            list.ResourceBarrier(&[transition(
                &self.composite,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            list.SetDescriptorHeaps(&[Some(self.comp_heap.clone())]);
            list.SetComputeRootSignature(&self.comp_root);
            list.SetPipelineState(&self.comp_pso);
            list.SetComputeRootDescriptorTable(0, self.comp_heap.GetGPUDescriptorHandleForHeapStart());
            list.SetComputeRoot32BitConstant(1, rw, 0);
            list.SetComputeRoot32BitConstant(1, rh, 1);
            list.Dispatch(rw.div_ceil(8), rh.div_ceil(8), 1);
            list.ResourceBarrier(&[transition(
                &self.composite,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
        }
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
