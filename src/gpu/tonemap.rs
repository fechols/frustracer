//! Fullscreen-pass pipelines: `blit` (CPU-tonemapped B8G8R8A8 → backbuffer)
//! and `tonemap` (linear HDR RGBA16F → backbuffer, replicating the CPU
//! resolve curve). Shaders are compiled at startup with D3DCompile (fxc,
//! SM 5.0) from HLSL embedded via include_str! — no build-time toolchain.

use super::d3d12::{Result, SWAPCHAIN_FORMAT};
use windows::core::{s, PCSTR};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{ID3DBlob, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

const BLIT_HLSL: &str = include_str!("shaders/blit.hlsl");
const TONEMAP_HLSL: &str = include_str!("shaders/tonemap.hlsl");

pub struct Passes {
    pub root_sig: ID3D12RootSignature,
    pub blit_pso: ID3D12PipelineState,
    pub tonemap_pso: ID3D12PipelineState,
    /// Shader-visible SRV heap; slot 0 = blit texture, slot 1 = HDR source.
    pub srv_heap: ID3D12DescriptorHeap,
    pub srv_size: u32,
}

pub const SRV_SLOT_BLIT: u32 = 0;
pub const SRV_SLOT_HDR: u32 = 1;
pub const SRV_SLOT_RR: u32 = 2;
pub const SRV_SLOT_XESS: u32 = 3;
/// The GPU-resident tracer's resolved HDR output (gpu/trace.rs).
pub const SRV_SLOT_GPU: u32 = 4;
/// The DXR pipeline's resolved HDR output (gpu/dxr.rs).
pub const SRV_SLOT_DXR: u32 = 5;
/// The FSR4 upscaled output (gpu/ffx_rr.rs).
pub const SRV_SLOT_FSR: u32 = 6;
/// The glare halo (gpu/bloom.rs level 0). Always bound — the tonemap PS declares
/// t1 unconditionally, so the table must be valid even under `--no-bloom`, where
/// `bloom_strength = 0` simply never samples it.
pub const SRV_SLOT_BLOOM: u32 = 7;
/// Room for future debug views. Also sizes `gpu/bloom.rs`'s source-slot region:
/// the glare pyramid keeps a permanent SRV per tonemap slot in its own heap.
pub const SRV_HEAP_CAPACITY: u32 = 12;

pub(super) fn compile(src: &str, entry: PCSTR, target: PCSTR, what: &str) -> Result<ID3DBlob> {
    let mut blob: Option<ID3DBlob> = None;
    let mut errs: Option<ID3DBlob> = None;
    let hr = unsafe {
        D3DCompile(
            src.as_ptr() as *const _,
            src.len(),
            None,
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut blob,
            Some(&mut errs),
        )
    };
    if let Err(e) = hr {
        let msg = errs
            .map(|b| unsafe {
                String::from_utf8_lossy(std::slice::from_raw_parts(
                    b.GetBufferPointer() as *const u8,
                    b.GetBufferSize(),
                ))
                .into_owned()
            })
            .unwrap_or_default();
        return Err(format!("D3DCompile({what}): {e}\n{msg}"));
    }
    Ok(blob.unwrap())
}

fn bytecode(blob: &ID3DBlob) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: unsafe { blob.GetBufferPointer() },
        BytecodeLength: unsafe { blob.GetBufferSize() },
    }
}

fn default_blend() -> D3D12_BLEND_DESC {
    let rt = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: false.into(),
        LogicOpEnable: false.into(),
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
    };
    D3D12_BLEND_DESC {
        AlphaToCoverageEnable: false.into(),
        IndependentBlendEnable: false.into(),
        RenderTarget: [rt; 8],
    }
}

fn fullscreen_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &ID3DBlob,
    ps: &ID3DBlob,
) -> Result<ID3D12PipelineState> {
    let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
        VS: bytecode(vs),
        PS: bytecode(ps),
        BlendState: default_blend(),
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC::default(),
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let mut desc = desc;
    desc.RTVFormats[0] = SWAPCHAIN_FORMAT;
    unsafe { device.CreateGraphicsPipelineState(&desc) }
        .map_err(|e| format!("CreateGraphicsPipelineState: {e}"))
}

impl Passes {
    pub fn new(device: &ID3D12Device) -> Result<Self> {
        // Root signature: [0] SRV table (t0 = source), [1] 4 root constants (b0),
        // [2] SRV table (t1 = the glare halo), + one static linear/clamp sampler
        // (0 DWORDs) for the tent tap.
        let ranges = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        }];
        let bloom_range = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 1,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        }];
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: ranges.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 4,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: bloom_range.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
        ];
        let samp = [D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxLOD: D3D12_FLOAT32_MAX,
            ShaderRegister: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            ..Default::default()
        }];
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 1,
            pStaticSamplers: samp.as_ptr(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut errb: Option<ID3DBlob> = None;
        unsafe {
            D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut errb))
        }
        .map_err(|e| format!("D3D12SerializeRootSignature: {e}"))?;
        let blob = blob.unwrap();
        let root_sig: ID3D12RootSignature = unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()),
            )
        }
        .map_err(|e| format!("CreateRootSignature: {e}"))?;

        let blit_vs = compile(BLIT_HLSL, s!("vsmain"), s!("vs_5_0"), "blit vs")?;
        let blit_ps = compile(BLIT_HLSL, s!("psmain"), s!("ps_5_0"), "blit ps")?;
        let tm_vs = compile(TONEMAP_HLSL, s!("vsmain"), s!("vs_5_0"), "tonemap vs")?;
        let tm_ps = compile(TONEMAP_HLSL, s!("psmain"), s!("ps_5_0"), "tonemap ps")?;
        let blit_pso = fullscreen_pso(device, &root_sig, &blit_vs, &blit_ps)?;
        let tonemap_pso = fullscreen_pso(device, &root_sig, &tm_vs, &tm_ps)?;

        let srv_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: SRV_HEAP_CAPACITY,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("CreateDescriptorHeap(SRV): {e}"))?;
        let srv_size = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };

        Ok(Self { root_sig, blit_pso, tonemap_pso, srv_heap, srv_size })
    }

    pub fn create_srv(&self, device: &ID3D12Device, res: &ID3D12Resource, format: DXGI_FORMAT, slot: u32) {
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        let start = unsafe { self.srv_heap.GetCPUDescriptorHandleForHeapStart() };
        let handle = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start.ptr + (slot * self.srv_size) as usize };
        unsafe { device.CreateShaderResourceView(res, Some(&desc), handle) };
    }

    pub fn gpu_srv(&self, slot: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let start = unsafe { self.srv_heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE { ptr: start.ptr + ((slot * self.srv_size) as u64) }
    }

    /// Record a fullscreen pass: PSO + root sig + SRV slot (+ optional
    /// inv_samples root constant) drawing 3 vertices to the bound RTV.
    /// Re-binds everything each call — this doubles as the post-Streamline
    /// state restore required by eDisableCLStateTracking.
    /// `bloom` is `(strength, texel_w, texel_h)`; strength 0 disables the arm and
    /// the halo SRV is never sampled (but the table is still bound — the PS
    /// declares t1 unconditionally).
    pub fn record(
        &self,
        list: &ID3D12GraphicsCommandList,
        pso: &ID3D12PipelineState,
        srv_slot: u32,
        inv_samples: f32,
        bloom: (f32, f32, f32),
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        w: u32,
        h: u32,
    ) {
        unsafe {
            list.SetPipelineState(pso);
            list.SetGraphicsRootSignature(&self.root_sig);
            list.SetDescriptorHeaps(&[Some(self.srv_heap.clone())]);
            list.SetGraphicsRootDescriptorTable(0, self.gpu_srv(srv_slot));
            list.SetGraphicsRootDescriptorTable(2, self.gpu_srv(SRV_SLOT_BLOOM));
            let consts = [inv_samples, bloom.0, bloom.1, bloom.2];
            list.SetGraphicsRoot32BitConstants(1, 4, consts.as_ptr() as *const _, 0);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: w as i32,
                bottom: h as i32,
            }]);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.DrawInstanced(3, 1, 0, 0);
        }
    }
}
