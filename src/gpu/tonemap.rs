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

use crate::gfx::shaders::{BLIT_HLSL, TONEMAP_HLSL, WAVEVIZ_HLSL};

pub struct Passes {
    pub root_sig: ID3D12RootSignature,
    pub blit_pso: ID3D12PipelineState,
    pub tonemap_pso: ID3D12PipelineState,
    /// The --waveviz overlay composite (the HUD's shape: its own PS + PSO on
    /// this root signature, premultiplied blend, drawn after the tonemap
    /// draw). Built only in ARMED sessions — unarmed sessions compile
    /// nothing and record nothing.
    pub waveviz_pso: Option<ID3D12PipelineState>,
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
/// The registered-consensus fuse of every wired upscaler (--quinlight,
/// gpu/quin.rs).
pub const SRV_SLOT_QUIN: u32 = 8;
/// The HUD/menu overlay texture (gpu/hud.rs) — NOT a tonemap source (bloom
/// never reads it; it composites OVER whatever was tonemapped), so it takes a
/// plain `create_srv`, never `wire_tonemap_src`.
pub const SRV_SLOT_OVERLAY: u32 = 9;
/// The raw-NGX DLSS-G interpolated frame (`--fg` DLSS sessions with the NDA
/// SDK present) — presented BEFORE the real frame each pair-present.
pub const SRV_SLOT_NGXFG: u32 = 10;
/// Room for future debug views. Also sizes `gpu/bloom.rs`'s source-slot region:
/// the glare pyramid keeps a permanent SRV per tonemap slot in its own heap.
pub const SRV_HEAP_CAPACITY: u32 = 12;

/// M12: run the REAL tonemap PS over a synthetic linear-HDR image and hand back
/// what it wrote — the twin gate that stops `tonemap.hlsl` from drifting away
/// from `tone::map`, the same discipline `feed.hlsl` and `nppd.hlsl` are held to.
///
/// Headless (no swapchain), so it renders into an offscreen target of whichever
/// format the caller names: `SWAPCHAIN_FORMAT` gates the 8-bit SDR encode,
/// `SWAPCHAIN_FORMAT_10BIT` the Sdr10 (Gamma22 through a 10-bit RTV) and
/// HDR10/PQ ones. All the curves come out of the same shader, so gating any
/// would catch a drifted port — gating each wire also pins each encode.
///
/// `src` is w*h*3 linear f32; returns w*h RGB read back from the target.
///
/// `bloom` is the PS's `(strength, texel.x, texel.y)` triple, normally
/// `(0, 0, 0)` — the curve is what this gate exists for and glare is M13's job.
/// The one caller that arms it is M12b's pre-glare arm: the glare SRV is bound
/// to `src` itself here, so a non-zero strength makes the PS blend a real tent
/// of the source into the colour, which is exactly the halo the spike guard
/// must NOT be measuring its ring against.
pub fn selftest(
    hg: &mut super::trace::HeadlessGpu,
    src: &[f32],
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
    tone: ToneParams,
    bloom: (f32, f32, f32),
) -> Result<Vec<[f32; 3]>> {
    use super::d3d12::{
        aligned_pitch, committed_tex, footprint, loc_footprint, loc_subresource, transition,
        ReadbackBuffer, UploadBuffer,
    };
    use half::f16;
    assert_eq!(src.len(), (w * h * 3) as usize);
    let bpp = 4usize; // every present format is 4 B/px now (BGRA8 / R10G10B10A2)

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
    // `record` binds the glare table unconditionally (the PS declares t1 whether
    // or not it samples it), so the slot must hold a REAL descriptor even though
    // strength 0 means it is never read — an uninitialized one is a GBV error.
    passes.create_srv(&hg.device, &src_tex, DXGI_FORMAT_R16G16B16A16_FLOAT, SRV_SLOT_BLOOM);

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
        // inv_samples = 1.0: `src` is already a per-frame radiance image. Bloom
        // strength 0: this gate scores the CURVE, and glare is a separate pass
        // with its own gate (--check-gpu M13) — mixing them would let a bloom
        // regression masquerade as a tonemap one, and vice versa.
        passes.record(list, &passes.tonemap_pso, SRV_SLOT_HDR, 1.0, bloom, tone, rtv, w, h);
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
            out[y * sw + x] = if format == DXGI_FORMAT_R10G10B10A2_UNORM {
                // R10G10B10A2 — R is the LOW 10 bits (unlike BGRA8's byte
                // order, where B leads).
                let px: &[u8; 4] = unsafe { &*(row.add(x * 4) as *const [u8; 4]) };
                let v = u32::from_le_bytes(*px);
                [
                    (v & 1023) as f32 / 1023.0,
                    ((v >> 10) & 1023) as f32 / 1023.0,
                    ((v >> 20) & 1023) as f32 / 1023.0,
                ]
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

pub(super) fn default_blend() -> D3D12_BLEND_DESC {
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

/// PREMULTIPLIED alpha-over (src + dest·(1−src.a)) — the HUD overlay's blend
/// (gpu/hud.rs). Slint's software renderer hands us premultiplied pixels, so
/// SrcBlend is ONE, not SRC_ALPHA.
pub(super) fn premultiplied_blend() -> D3D12_BLEND_DESC {
    let mut b = default_blend();
    b.RenderTarget[0].BlendEnable = true.into();
    b.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE;
    b.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    b.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    b.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
    b
}

/// Root constants the fullscreen PS reads (b0), in tonemap.hlsl's cbuffer
/// order: `inv_samples`, the three glare fields (strength + the tent's texel
/// step), then the five `tone::ToneParams` fields (knee/headroom/scale/mode/
/// exposure) — the presentation curve is uniform state, not baked into the
/// shader, which is what lets a display change be a retune.
const NUM_ROOT_CONSTS: u32 = 10;

fn fullscreen_pso(
    device: &ID3D12Device,
    rtv_format: DXGI_FORMAT,
    root_sig: &ID3D12RootSignature,
    vs: &ID3DBlob,
    ps: &ID3DBlob,
) -> Result<ID3D12PipelineState> {
    fullscreen_pso_blend(device, rtv_format, root_sig, vs, ps, default_blend())
}

/// `fullscreen_pso` with an explicit blend state — the HUD overlay's
/// premultiplied-alpha composite (gpu/hud.rs) is the one non-opaque pass.
pub(super) fn fullscreen_pso_blend(
    device: &ID3D12Device,
    rtv_format: DXGI_FORMAT,
    root_sig: &ID3D12RootSignature,
    vs: &ID3DBlob,
    ps: &ID3DBlob,
    blend: D3D12_BLEND_DESC,
) -> Result<ID3D12PipelineState> {
    let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
        VS: bytecode(vs),
        PS: bytecode(ps),
        BlendState: blend,
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
        // Root signature: [0] SRV table (t0 = source), [1] TONE_CONSTS root
        // constants (b0), [2] SRV table (t1 = the glare halo), + one static
        // linear/clamp sampler (0 DWORDs) for the tent tap.
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
                        Num32BitValues: NUM_ROOT_CONSTS,
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
            // [3] t2 as a ROOT SRV (2 DWORDs, no descriptor): the --waveviz
            // ticket buffer, bound by GPU VA only for the waveviz draw. The
            // tonemap/hud/blit shaders never declare t2, so leaving this
            // param unset on their draws is legal.
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 2, RegisterSpace: 0 },
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
        // The spike guard's calibration constants ride INJECTED DEFINES (the
        // spp_defs/detail_defs idiom) rather than the cbuffer: they are a
        // restart-tier probe, so paying four root DWORDs — and four more pads
        // in every shader that mirrors this layout — to make them live would
        // be the wrong trade. `guard_strength` is the one that IS live (a menu
        // row), and that one is a cbuffer field. Pasting also means the FR_
        // sweep levers provably REACH the shader, which is the whole point.
        let tm_src = format!("{}{}", crate::autoexp::guard_defs(), TONEMAP_HLSL);
        let tm_vs = compile(&tm_src, s!("vsmain"), s!("vs_5_0"), "tonemap vs")?;
        let tm_ps = compile(&tm_src, s!("psmain"), s!("ps_5_0"), "tonemap ps")?;
        let blit_pso = fullscreen_pso(device, rtv_format, &root_sig, &blit_vs, &blit_ps)?;
        let tonemap_pso = fullscreen_pso(device, rtv_format, &root_sig, &tm_vs, &tm_ps)?;
        // --waveviz: armed sessions only — the unarmed session builds no
        // extra PSO and the funnel's draw predicate never fires.
        let waveviz_pso = if super::trace::waveviz_on() {
            let wv_vs = compile(WAVEVIZ_HLSL, s!("vsmain"), s!("vs_5_0"), "waveviz vs")?;
            let wv_ps = compile(WAVEVIZ_HLSL, s!("psmain"), s!("ps_5_0"), "waveviz ps")?;
            Some(fullscreen_pso_blend(
                device,
                rtv_format,
                &root_sig,
                &wv_vs,
                &wv_ps,
                premultiplied_blend(),
            )?)
        } else {
            None
        };

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

        Ok(Self { root_sig, blit_pso, tonemap_pso, waveviz_pso, srv_heap, srv_size })
    }

    /// The --waveviz overlay draw (waveviz.hlsl's own b0 layout — root
    /// constants are per-draw, so the tonemap/hud draws are untouched):
    /// tickets bound as the t2 root SRV by VA, nearest window→render mapping
    /// from the four dims, ToneParams supplying the PQ arm's scale/mode.
    /// Caller brackets the ticket buffer UNORDERED_ACCESS ↔
    /// PIXEL_SHADER_RESOURCE around this draw.
    #[allow(clippy::too_many_arguments)]
    pub fn record_waveviz(
        &self,
        list: &ID3D12GraphicsCommandList,
        pso: &ID3D12PipelineState,
        tickets_va: u64,
        rw: u32,
        rh: u32,
        ww: u32,
        wh: u32,
        tone: ToneParams,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        unsafe {
            list.SetPipelineState(pso);
            list.SetGraphicsRootSignature(&self.root_sig);
            list.SetDescriptorHeaps(&[Some(self.srv_heap.clone())]);
            // Tables 0/2 are unread by this PS but bound to valid slots
            // anyway (the record() habit — nothing is left dangling).
            list.SetGraphicsRootDescriptorTable(0, self.gpu_srv(SRV_SLOT_HDR));
            list.SetGraphicsRootDescriptorTable(2, self.gpu_srv(SRV_SLOT_BLOOM));
            list.SetGraphicsRootShaderResourceView(3, tickets_va);
            // waveviz.hlsl's Params layout: rw rh ww wh | scale mode pad x4.
            let consts: [u32; NUM_ROOT_CONSTS as usize] = [
                rw,
                rh,
                ww,
                wh,
                tone.scale.to_bits(),
                match tone.mode {
                    crate::tone::ToneMode::Gamma22 => 1.0f32,
                    crate::tone::ToneMode::Pq => 2.0f32,
                }
                .to_bits(),
                0,
                0,
                0,
                0,
            ];
            list.SetGraphicsRoot32BitConstants(1, NUM_ROOT_CONSTS, consts.as_ptr() as *const _, 0);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: ww as f32,
                Height: wh as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: ww as i32,
                bottom: wh as i32,
            }]);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.DrawInstanced(3, 1, 0, 0);
        }
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
    /// Re-binds everything each call — this doubles as the state restore
    /// after an upscaler/denoiser evaluate clobbered the list's bindings.
    ///
    /// `bloom` is `(strength, texel_w, texel_h)`; strength 0 disables the arm and
    /// the halo SRV is never sampled (but the table is still bound — the PS
    /// declares t1 unconditionally).
    ///
    /// `tone` is ignored by the blit PSO (its source is already encoded); it is
    /// passed unconditionally so the two PSOs can share one root signature.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        list: &ID3D12GraphicsCommandList,
        pso: &ID3D12PipelineState,
        srv_slot: u32,
        inv_samples: f32,
        bloom: (f32, f32, f32),
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
            list.SetGraphicsRootDescriptorTable(2, self.gpu_srv(SRV_SLOT_BLOOM));
            // Layout must match tonemap.hlsl's cbuffer exactly.
            let consts: [f32; NUM_ROOT_CONSTS as usize] = [
                inv_samples,
                bloom.0,
                bloom.1,
                bloom.2,
                tone.knee,
                tone.headroom,
                tone.scale,
                // Mode literals shared with tonemap.hlsl / hud.hlsl (change
                // all three together). 0 was scRGB-linear, retired with f16.
                match tone.mode {
                    crate::tone::ToneMode::Gamma22 => 1.0,
                    crate::tone::ToneMode::Pq => 2.0,
                },
                tone.exposure,
                // The spike guard's strength, read from the lever here rather
                // than threaded through ToneParams (the `bloom::enabled()`
                // habit — a display-stage process lever the funnel consults).
                // Exactly 0.0 when the guard is off, which is the shader's own
                // structural off arm.
                crate::autoexp::guard_strength_live(),
            ];
            list.SetGraphicsRoot32BitConstants(1, NUM_ROOT_CONSTS, consts.as_ptr() as *const _, 0);
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
