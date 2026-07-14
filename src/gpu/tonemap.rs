//! Fullscreen-pass pipelines: `blit` (CPU-tonemapped B8G8R8A8 → backbuffer)
//! and `tonemap` (linear HDR RGBA16F → backbuffer, replicating the CPU
//! resolve curve). Shaders are compiled at startup with D3DCompile (fxc,
//! SM 5.0) from HLSL embedded via include_str! — no build-time toolchain.

use super::d3d12::Result;
use crate::tone::ToneParams;
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
/// The registered-consensus fuse of every wired upscaler (--quinlight,
/// gpu/quin.rs).
pub const SRV_SLOT_QUIN: u32 = 7;
/// Room for the DLSS-RR output SRV and future debug views.
const SRV_HEAP_CAPACITY: u32 = 8;

/// M12: run the REAL tonemap PS over a synthetic linear-HDR image and hand back
/// what it wrote — the twin gate that stops `tonemap.hlsl` from drifting away
/// from `tone::map`, the same discipline `feed.hlsl` and `nppd.hlsl` are held to.
///
/// Headless (no swapchain), so it renders into an offscreen target of whichever
/// format the caller names: `SWAPCHAIN_FORMAT` gates the 8-bit SDR encode,
/// `SWAPCHAIN_FORMAT_HDR` gates the scRGB one. Both curves come out of the same
/// shader, so gating either would catch a drifted port — gating both also pins
/// the encode.
///
/// `src` is w*h*3 linear f32; returns w*h RGB read back from the target.
pub fn selftest(
    hg: &mut super::trace::HeadlessGpu,
    src: &[f32],
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
    tone: ToneParams,
) -> Result<Vec<[f32; 3]>> {
    use super::d3d12::{
        aligned_pitch, committed_tex, footprint, loc_footprint, loc_subresource, transition,
        ReadbackBuffer, UploadBuffer,
    };
    use half::f16;
    assert_eq!(src.len(), (w * h * 3) as usize);
    let hdr = format == DXGI_FORMAT_R16G16B16A16_FLOAT;
    let bpp = if hdr { 8usize } else { 4usize };

    let passes = Passes::new(&hg.device, format)?;

    // The source is a real RGBA16F texture bound to the real SRV slot the
    // tonemap PS reads in `present_hdr` — not a debug binding that could hide a
    // wiring bug.
    let sw = w as usize;
    let src_tex = committed_tex(
        &hg.device,
        w,
        h,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let src_pitch = aligned_pitch(sw * 8);
    let up = UploadBuffer::new(&hg.device, src_pitch * h as usize)?;
    for y in 0..h as usize {
        let row: &mut [[f16; 4]] = unsafe {
            std::slice::from_raw_parts_mut(up.ptr.add(y * src_pitch) as *mut [f16; 4], sw)
        };
        for (x, px) in row.iter_mut().enumerate() {
            let i = (y * sw + x) * 3;
            *px = [
                f16::from_f32(src[i]),
                f16::from_f32(src[i + 1]),
                f16::from_f32(src[i + 2]),
                f16::from_f32(1.0),
            ];
        }
    }
    passes.create_srv(&hg.device, &src_tex, DXGI_FORMAT_R16G16B16A16_FLOAT, SRV_SLOT_HDR);

    let target = committed_tex(
        &hg.device,
        w,
        h,
        format,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    )?;
    let rtv_heap: ID3D12DescriptorHeap = unsafe {
        hg.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: 1,
            ..Default::default()
        })
    }
    .map_err(|e| format!("selftest RTV heap: {e}"))?;
    let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    unsafe { hg.device.CreateRenderTargetView(&target, None, rtv) };

    let pitch = aligned_pitch(sw * bpp);
    let rb = ReadbackBuffer::new(&hg.device, pitch * h as usize)?;
    let src_fp = footprint(DXGI_FORMAT_R16G16B16A16_FLOAT, w, h, 8, 0);
    let dst_fp = footprint(format, w, h, bpp, 0);

    hg.run(|list| unsafe {
        list.CopyTextureRegion(
            &loc_subresource(&src_tex),
            0,
            0,
            0,
            &loc_footprint(&up.resource, src_fp),
            None,
        );
        list.ResourceBarrier(&[transition(
            &src_tex,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )]);
        // inv_samples = 1.0: `src` is already a per-frame radiance image.
        passes.record(list, &passes.tonemap_pso, SRV_SLOT_HDR, 1.0, tone, rtv, w, h);
        list.ResourceBarrier(&[transition(
            &target,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyTextureRegion(
            &loc_footprint(&rb.resource, dst_fp),
            0,
            0,
            0,
            &loc_subresource(&target),
            None,
        );
    })?;

    let mut ptr = std::ptr::null_mut();
    unsafe { rb.resource.Map(0, None, Some(&mut ptr)) }
        .map_err(|e| format!("selftest readback Map: {e}"))?;
    let mut out = vec![[0.0f32; 3]; (w * h) as usize];
    for y in 0..h as usize {
        let row = unsafe { (ptr as *const u8).add(y * pitch) };
        for x in 0..sw {
            out[y * sw + x] = if hdr {
                let px: &[f16; 4] = unsafe { &*(row.add(x * 8) as *const [f16; 4]) };
                [px[0].into(), px[1].into(), px[2].into()]
            } else {
                // B8G8R8A8_UNORM — B and R are swapped on the wire.
                let px: &[u8; 4] = unsafe { &*(row.add(x * 4) as *const [u8; 4]) };
                [px[2] as f32 / 255.0, px[1] as f32 / 255.0, px[0] as f32 / 255.0]
            };
        }
    }
    unsafe { rb.resource.Unmap(0, None) };
    Ok(out)
}

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

/// Root constants the fullscreen PS reads (b0). `inv_samples` plus the four
/// `tone::ToneParams` fields — the presentation curve is uniform state, not
/// baked into the shader, which is what lets a display change be a retune.
const NUM_ROOT_CONSTS: u32 = 5;

fn fullscreen_pso(
    device: &ID3D12Device,
    rtv_format: DXGI_FORMAT,
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
    desc.RTVFormats[0] = rtv_format;
    unsafe { device.CreateGraphicsPipelineState(&desc) }
        .map_err(|e| format!("CreateGraphicsPipelineState: {e}"))
}

impl Passes {
    /// `rtv_format` is the swapchain's actual format (`D3d::format`) — both PSOs
    /// bake it, so it must come from the swapchain, never from the CLI flag.
    pub fn new(device: &ID3D12Device, rtv_format: DXGI_FORMAT) -> Result<Self> {
        // Root signature: [0] SRV table (t0, pixel), [1] root constants (b0).
        let ranges = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
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
                        Num32BitValues: NUM_ROOT_CONSTS,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
        ];
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
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
        let blit_pso = fullscreen_pso(device, rtv_format, &root_sig, &blit_vs, &blit_ps)?;
        let tonemap_pso = fullscreen_pso(device, rtv_format, &root_sig, &tm_vs, &tm_ps)?;

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

    /// Record a fullscreen pass: PSO + root sig + SRV slot + the inv_samples and
    /// `ToneParams` root constants, drawing 3 vertices to the bound RTV.
    /// Re-binds everything each call — this doubles as the post-Streamline
    /// state restore required by eDisableCLStateTracking.
    ///
    /// `tone` is ignored by the blit PSO (its source is already encoded); it is
    /// passed unconditionally so the two PSOs can share one root signature.
    pub fn record(
        &self,
        list: &ID3D12GraphicsCommandList,
        pso: &ID3D12PipelineState,
        srv_slot: u32,
        inv_samples: f32,
        tone: ToneParams,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        w: u32,
        h: u32,
    ) {
        unsafe {
            list.SetPipelineState(pso);
            list.SetGraphicsRootSignature(&self.root_sig);
            list.SetDescriptorHeaps(&[Some(self.srv_heap.clone())]);
            list.SetGraphicsRootDescriptorTable(0, self.gpu_srv(srv_slot));
            // Layout must match tonemap.hlsl's cbuffer exactly.
            let consts: [f32; NUM_ROOT_CONSTS as usize] = [
                inv_samples,
                tone.knee,
                tone.headroom,
                tone.scale,
                if tone.gamma { 1.0 } else { 0.0 },
            ];
            list.SetGraphicsRoot32BitConstants(
                1,
                NUM_ROOT_CONSTS,
                consts.as_ptr() as *const _,
                0,
            );
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
