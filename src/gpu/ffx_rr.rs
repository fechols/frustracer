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

const COMPOSITE_HLSL: &str = include_str!("../shaders/fsr_composite.hlsl");

/// One Ray Regeneration dispatch's resource set: the shared inputs plus an
/// in/out pair per subscribed signal (`ffx_sys::SIGNALS`).
pub struct DenoiseRes {
    pub depth_lin: FfxShimRes,
    pub mvec: FfxShimRes,
    pub normals: FfxShimRes,
    pub spec_alb: FfxShimRes,
    pub diff_alb: FfxShimRes,
    pub dd_in: FfxShimRes,
    pub dd_out: FfxShimRes,
    pub ds_in: FfxShimRes,
    pub ds_out: FfxShimRes,
    pub ao_in: FfxShimRes,
    pub ao_out: FfxShimRes,
    pub is_in: FfxShimRes,
    pub is_out: FfxShimRes,
}

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
const P_AO_IN: usize = 9; // R16F ambient-occlusion open fraction [0,1]
const P_IS_IN: usize = 10; // RGBA16F demodulated indirect specular (A = hit t)
const N_PLANES: usize = 11;

/// Composite-pass root constants: the sky's 9 SH rows as float4 (the cbuffer
/// declares `float4 sky_sh[9]`, so HLSL's 16-byte row stride is the layout —
/// a float3 array would pad to the same 4 DWORDs per row while inviting the
/// straddle bug that shipped once here), then rw and rh.
const COMP_CONSTS: u32 = 4 * crate::sh::N as u32 + 2;

pub struct FsrResources {
    pub max_w: u32, // input plane allocation size (range max)
    pub max_h: u32,
    planes: [Plane; N_PLANES],
    /// Denoised signal UAVs (max render size; ffx writes the sub-rect).
    dd_out: ID3D12Resource,
    ds_out: ID3D12Resource,
    ao_out: ID3D12Resource,
    is_out: ID3D12Resource,
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
    /// 7 SRVs (dd_out, ds_out, diff_alb, spec_alb, residual, ao_out, is_out) +
    /// 1 UAV (composite), created once — the textures never change. The ORDER
    /// is fsr_composite.hlsl's t0..t6 and is pinned by the composite gate.
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
            (DXGI_FORMAT_R16_FLOAT, 2),
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 8),
        ];
        let mut offset = 0usize;
        let mut planes = Vec::with_capacity(N_PLANES);
        for (format, bpp) in specs {
            // ALLOW_UNORDERED_ACCESS: GPU-fed sessions write these planes
            // directly from the cs_feed_fsr_rr kernel (typed UAV stores) —
            // the xr.rs precedent. The CPU upload path is unaffected; the
            // rest state stays NON_PIXEL_SHADER_RESOURCE either way.
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
        let is_out = signal_uav()?;
        // The AO signal is scalar in and out (R16F, like its input plane).
        let ao_out = committed_tex(
            device,
            max_w,
            max_h,
            DXGI_FORMAT_R16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        )?;
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

        // Composite pass: table of 8 SRVs + 1 UAV, and COMP_CONSTS root
        // constants (the sky's 9 SH rows, then rw/rh).
        //
        // The AO signal's remodulation factor used to be one float3 here
        // (shade::AMBIENT). The one sky makes it directional — sky_sh.irradiance
        // at the pixel's normal — so the pass reads the NORMALS plane (the 8th
        // SRV) and carries the sky itself in its constants.
        let ranges = [
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 8,
                BaseShaderRegister: 0,
                ..Default::default()
            },
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                NumDescriptors: 1,
                BaseShaderRegister: 0,
                OffsetInDescriptorsFromTableStart: 8,
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
                        Num32BitValues: COMP_CONSTS,
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

        // sh.hlsli (the SH evaluator) and fsr_wire.hlsli (the octahedral wire
        // encoding) are the SHARED halves — the tracer's kernels and feed.hlsl
        // paste the same two. This pass has no shade.hlsli prelude, but it must
        // not therefore own private copies of either: the composite identity is
        // precisely the claim that it and feed.hlsl agree.
        let comp_src =
            [super::trace::SH_HLSLI, super::trace::FSR_WIRE_HLSLI, COMPOSITE_HLSL].join("\n");
        let cs = tonemap::compile(&comp_src, s!("cs"), s!("cs_5_0"), "fsr_composite")?;
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
                NumDescriptors: 9,
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
        // t0..t7 in fsr_composite.hlsl's declaration order. t7 (the normals
        // plane) is the AO signal's remodulation input — the pass rebuilds
        // sky_sh.irradiance(n) from it, which is why the split site had to
        // subtract the factor at the WIRE normal (fsr::wire_normal).
        srv(&dd_out, DXGI_FORMAT_R16G16B16A16_FLOAT, 0);
        srv(&ds_out, DXGI_FORMAT_R16G16B16A16_FLOAT, 1);
        srv(&planes[P_DIFF_ALB].tex, DXGI_FORMAT_R8G8B8A8_UNORM, 2);
        srv(&planes[P_SPEC_ALB].tex, DXGI_FORMAT_R8G8B8A8_UNORM, 3);
        srv(&planes[P_RESIDUAL].tex, DXGI_FORMAT_R16G16B16A16_FLOAT, 4);
        srv(&ao_out, DXGI_FORMAT_R16_FLOAT, 5);
        srv(&is_out, DXGI_FORMAT_R16G16B16A16_FLOAT, 6);
        srv(&planes[P_NORMALS].tex, DXGI_FORMAT_R10G10B10A2_UNORM, 7);
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        unsafe { device.CreateUnorderedAccessView(&composite, None, Some(&uav_desc), at(8)) };

        Ok(Self {
            max_w,
            max_h,
            planes,
            dd_out,
            ds_out,
            ao_out,
            is_out,
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

        // AO: f16 storage -> R16F, bit copy (the open fraction, already in
        // [0,1]).
        plane_mem(&self.planes[P_AO_IN])
            .par_chunks_mut(aligned_pitch(w * 2))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [f16] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    *p = f16::from_bits(f.ao[y * w + x].load(Relaxed));
                }
            });

        // Indirect specular: RGB the demodulated reflection radiance (f16 bit
        // copy), A the reflection ray's hit distance — the channel layout
        // ffx_denoiser.h's INDIRECT_SPECULAR signal documents.
        plane_mem(&self.planes[P_IS_IN])
            .par_chunks_mut(aligned_pitch(w * 8))
            .take(rh)
            .enumerate()
            .for_each(|(y, row)| {
                let px: &mut [[f16; 4]] = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut _, w) };
                for (x, p) in px.iter_mut().enumerate() {
                    let i = y * w + x;
                    let i3 = i * 3;
                    p[0] = f16::from_bits(f.is[i3].load(Relaxed));
                    p[1] = f16::from_bits(f.is[i3 + 1].load(Relaxed));
                    p[2] = f16::from_bits(f.is[i3 + 2].load(Relaxed));
                    p[3] = f16::from_bits(g.spec_hit_t[i].load(Relaxed));
                }
            });

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

    /// The eleven input planes as GPU feed targets for `cs_feed_fsr_rr`, in
    /// upload-plane order (depth_lin, depth_clip, mvec, normals, diff_alb,
    /// spec_alb, dd_in, ds_in, residual, ao_in, is_in) — `wire_session_feed`
    /// maps them to the FEED_* registers explicitly, and the check suites'
    /// rewire gates pin the mapping.
    pub fn plane_resources(&self) -> [(&ID3D12Resource, DXGI_FORMAT); N_PLANES] {
        [
            P_DEPTH_LIN,
            P_DEPTH_CLIP,
            P_MVEC,
            P_NORMALS,
            P_DIFF_ALB,
            P_SPEC_ALB,
            P_DD_IN,
            P_DS_IN,
            P_RESIDUAL,
            P_AO_IN,
            P_IS_IN,
        ]
        .map(|p| (&self.planes[p].tex, self.planes[p].format))
    }

    /// The frame-generation prepare's inputs: (reversed-Z clip depth, the MV
    /// plane). This plane's RG carries UV deltas (prev − cur), so the
    /// prepare's mv_scale is (rw, rh) — the same conversion the upscale desc
    /// applies for this flavor.
    pub fn fg_inputs(&self) -> (&ID3D12Resource, &ID3D12Resource) {
        (&self.planes[P_DEPTH_CLIP].tex, &self.planes[P_MVEC].tex)
    }

    fn shim(res: &ID3D12Resource, state: u32) -> FfxShimRes {
        FfxShimRes { resource: res.as_raw(), state }
    }

    /// The denoiser dispatch's resource references, in the ffx states the
    /// resources will actually be in when the recorded work executes (inputs
    /// NON_PIXEL_SHADER_RESOURCE = compute read; outputs UAV — the caller
    /// wraps the dispatch in `barrier_denoise_*`).
    pub fn denoise_res(&self) -> DenoiseRes {
        let read = |p: usize| Self::shim(&self.planes[p].tex, RES_STATE_COMPUTE_READ);
        let write = |r: &ID3D12Resource| Self::shim(r, RES_STATE_UNORDERED_ACCESS);
        DenoiseRes {
            depth_lin: read(P_DEPTH_LIN),
            mvec: read(P_MVEC),
            normals: read(P_NORMALS),
            spec_alb: read(P_SPEC_ALB),
            diff_alb: read(P_DIFF_ALB),
            dd_in: read(P_DD_IN),
            dd_out: write(&self.dd_out),
            ds_in: read(P_DS_IN),
            ds_out: write(&self.ds_out),
            ao_in: read(P_AO_IN),
            ao_out: write(&self.ao_out),
            is_in: read(P_IS_IN),
            is_out: write(&self.is_out),
        }
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

    /// Every denoised output the dispatch writes — one per subscribed signal.
    fn outs(&self) -> [&ID3D12Resource; 4] {
        [&self.dd_out, &self.ds_out, &self.ao_out, &self.is_out]
    }

    pub fn barrier_denoise_begin(&self, list: &ID3D12GraphicsCommandList) {
        let b: Vec<_> = self
            .outs()
            .iter()
            .map(|r| {
                transition(
                    r,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )
            })
            .collect();
        unsafe { list.ResourceBarrier(&b) };
    }

    pub fn barrier_denoise_end(&self, list: &ID3D12GraphicsCommandList) {
        let b: Vec<_> = self
            .outs()
            .iter()
            .map(|r| {
                transition(
                    r,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )
            })
            .collect();
        unsafe { list.ResourceBarrier(&b) };
    }

    /// Gate hook (`--check-gpu` / `--check-dxr`): make the denoiser an IDENTITY
    /// by copying each signal's input plane into its denoised-output UAV. With
    /// pass-through signals, `record_composite` must reproduce the traced color
    /// — which is the composite identity, and the only way to exercise
    /// fsr_composite.hlsl without an RDNA4 denoiser in the loop. The four pairs
    /// are same-format, same-dimension by construction (P_DD_IN/P_DS_IN/P_IS_IN
    /// are RGBA16F like their outs; P_AO_IN and ao_out are both R16F).
    pub fn record_signal_passthrough(&self, list: &ID3D12GraphicsCommandList) {
        let pairs: [(&ID3D12Resource, &ID3D12Resource); 4] = [
            (&self.planes[P_DD_IN].tex, &self.dd_out),
            (&self.planes[P_DS_IN].tex, &self.ds_out),
            (&self.planes[P_AO_IN].tex, &self.ao_out),
            (&self.planes[P_IS_IN].tex, &self.is_out),
        ];
        unsafe {
            let pre: Vec<_> = pairs
                .iter()
                .flat_map(|(src, dst)| {
                    [
                        transition(
                            src,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        ),
                        transition(
                            dst,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        ),
                    ]
                })
                .collect();
            list.ResourceBarrier(&pre);
            for (src, dst) in pairs {
                list.CopyResource(dst, src);
            }
            // Back to the rest state both the composite SRVs and the next
            // denoise dispatch expect.
            let post: Vec<_> = pairs
                .iter()
                .flat_map(|(src, dst)| {
                    [
                        transition(
                            src,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        ),
                        transition(
                            dst,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        ),
                    ]
                })
                .collect();
            list.ResourceBarrier(&post);
        }
    }

    /// The composite CS's output texture (RGBA16F, render-res sub-rect) — the
    /// gate reads it back; the upscaler consumes it in a live session.
    pub fn composite_tex(&self) -> &ID3D12Resource {
        &self.composite
    }

    /// Record the remodulation compute pass over the `rw×rh` sub-rect.
    /// Binds everything from scratch (heap, root sig, PSO) — this doubles as
    /// the state restore after the ffx dispatch, same rationale as
    /// `Passes::record`.
    pub fn record_composite(
        &self,
        list: &ID3D12GraphicsCommandList,
        rw: u32,
        rh: u32,
        sky_sh: &crate::sh::Sh9,
    ) {
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
            // DWORD order is fsr_composite.hlsl's cbuffer layout, which LEADS
            // with the float4 array so the block packs contiguously:
            // sky_sh[0..9] (4 DWORDs each, .w unused) | rw | rh. The sky is the
            // AO signal's remodulation factor — the split subtracted
            // irradiance(n_wire)*ao*kd, so the shader must add back exactly
            // that, which means it needs these coefficients and the normals
            // plane, not a constant.
            for (i, c) in sky_sh.c.iter().enumerate() {
                let base = (i * 4) as u32;
                for (k, v) in [c.x, c.y, c.z, 0.0].into_iter().enumerate() {
                    list.SetComputeRoot32BitConstant(1, v.to_bits(), base + k as u32);
                }
            }
            list.SetComputeRoot32BitConstant(1, rw, COMP_CONSTS - 2);
            list.SetComputeRoot32BitConstant(1, rh, COMP_CONSTS - 1);
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
