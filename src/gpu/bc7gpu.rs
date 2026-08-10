//! The GPU BC7 encoder (`shaders/bc7enc.hlsl` is the kernel): a mode-6
//! compute encode that runs at scene-upload time inside `SceneGpu::
//! new_uploaded`'s band loop, replacing the ~20 s-per-Sponza CPU ispc encode
//! (`src/bc7.rs` keeps that path as the `--bc7-cpu` A/B arm — it shares no
//! code with this one, which is what makes it an independent cross-check).
//!
//! Compiled with D3DCompile (fxc, cs_5_0) like bloom/ngxfg_guides — no DXC
//! dependency, so the encoder exists before any tracer kernel does and
//! `--check-gpu` can gate it without the wavefront. Binding is the M11
//! root-view pattern: no descriptor heap at all — the staged RGBA8 rows bind
//! as a raw root SRV straight off the upload ring's VA (upload heaps are
//! GENERIC_READ), the block buffer as a raw root UAV.
//!
//! Determinism: per (device, driver, bytecode) only — the kernel is
//! per-block independent (no atomics, no cross-thread state), so one machine
//! re-encodes identically, but there is NO cross-vendor bit claim and
//! nothing depends on one (no BC7 disk cache; the M11 fidelity gate runs the
//! session's own encoder in-session). The CPU arm keeps its full determinism
//! pin in `bc7::self_test`.

use super::d3d12::{self, Result, transition};
use super::tonemap::compile;
use crate::bc7;
use windows::core::s;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

const BC7ENC_HLSL: &str = include_str!("../shaders/bc7enc.hlsl");

/// The kernel's effort tier for an ispc `Quality` name: 0 = mode-6 PCA fit,
/// no refinement; 1 = + 2 least-squares rounds + CONDITIONAL mode-1 (top-4
/// partitions, only on blocks where mode 6 left visible error); 2/3 =
/// mode-1 always, top-8/16 partitions.
pub fn effort(q: bc7::Quality) -> u32 {
    match q {
        bc7::Quality::UltraFast => 0,
        bc7::Quality::Fast => 1,
        bc7::Quality::Basic => 2,
        bc7::Quality::Slow => 3,
    }
}

pub struct Bc7Enc {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Encoded-block output, resting in UNORDERED_ACCESS between uses.
    pub block_buf: ID3D12Resource,
    pub block_cap: usize,
}

impl Bc7Enc {
    /// `block_cap` = bytes of the largest single encode target (the max over
    /// compressible mips of `block_pitch(mw) * blocks(mh)`).
    pub fn new(device: &ID3D12Device, block_cap: usize) -> Result<Bc7Enc> {
        // Root signature: [0] raw root SRV (t0, the staged texel rows),
        // [1] raw root UAV (u0, the block buffer), [2] 8 root constants (b0).
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 8,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            ..Default::default()
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut errb: Option<ID3DBlob> = None;
        unsafe {
            D3D12SerializeRootSignature(
                &rs_desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                Some(&mut errb),
            )
        }
        .map_err(|e| format!("bc7gpu: D3D12SerializeRootSignature: {e}"))?;
        let blob = blob.unwrap();
        let root_sig: ID3D12RootSignature = unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    blob.GetBufferPointer() as *const u8,
                    blob.GetBufferSize(),
                ),
            )
        }
        .map_err(|e| format!("bc7gpu: CreateRootSignature: {e}"))?;

        let cs = compile(BC7ENC_HLSL, s!("cs_bc7_encode"), s!("cs_5_0"), "bc7enc")?;
        let pso: ID3D12PipelineState = unsafe {
            device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root_sig),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: cs.GetBufferPointer(),
                    BytecodeLength: cs.GetBufferSize(),
                },
                ..Default::default()
            })
        }
        .map_err(|e| format!("bc7gpu: CreateComputePipelineState: {e}"))?;

        let block_buf = d3d12::committed_buffer(
            device,
            block_cap.max(bc7::BLOCK_BYTES) as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        Ok(Bc7Enc { root_sig, pso, block_buf, block_cap })
    }

    /// Record the encode of one band: `staged_rows` RGBA8 texel rows of an
    /// `mw`-wide mip, resident in `src` (an upload-heap buffer — GENERIC_READ
    /// covers the SRV read, no transition) at `src_pitch` bytes/row, into
    /// `band_rows` block rows of `block_buf` (which must be resting in
    /// UNORDERED_ACCESS — `record_copy_out` returns it there).
    pub fn record_encode(
        &self,
        l: &ID3D12GraphicsCommandList,
        src: &ID3D12Resource,
        mw: u32,
        staged_rows: u32,
        src_pitch: u32,
        band_rows: u32,
        effort: u32,
    ) {
        let bw = bc7::blocks(mw);
        let row_blocks = (d3d12::block_pitch(mw) / bc7::BLOCK_BYTES) as u32;
        debug_assert!(
            band_rows as usize * d3d12::block_pitch(mw) <= self.block_cap,
            "bc7gpu: band over block_buf capacity"
        );
        let consts: [u32; 8] = [mw, staged_rows, src_pitch, bw, band_rows, row_blocks, effort, 0];
        unsafe {
            l.SetComputeRootSignature(&self.root_sig);
            l.SetPipelineState(&self.pso);
            l.SetComputeRootShaderResourceView(0, src.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(1, self.block_buf.GetGPUVirtualAddress());
            l.SetComputeRoot32BitConstants(2, 8, consts.as_ptr() as *const _, 0);
            l.Dispatch(bw.div_ceil(8), band_rows.div_ceil(8), 1);
        }
    }

    /// Copy the encoded band out of `block_buf` into mip `mip` of `dst` at
    /// texel row `y0` (a multiple of 4 — whole block rows; `h_tex` reaches
    /// the mip's own height on the final band, the other legal form). The
    /// UAV→COPY_SOURCE transition is the write-flush; the buffer returns to
    /// UNORDERED_ACCESS for the next band.
    pub fn record_copy_out(
        &self,
        l: &ID3D12GraphicsCommandList,
        dst: &ID3D12Resource,
        mip: u32,
        y0: u32,
        fmt: DXGI_FORMAT,
        mw: u32,
        h_tex: u32,
    ) {
        let fp = d3d12::footprint_block(fmt, mw, h_tex, 0);
        unsafe {
            l.ResourceBarrier(&[transition(
                &self.block_buf,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyTextureRegion(
                &d3d12::loc_subresource_mip(dst, mip),
                0,
                y0,
                0,
                &d3d12::loc_footprint(&self.block_buf, fp),
                None,
            );
            l.ResourceBarrier(&[transition(
                &self.block_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    }
}
