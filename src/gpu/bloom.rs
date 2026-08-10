//! The GPU half of the glare pass (`src/bloom.rs` is the source of truth and
//! carries the rationale; `shaders/bloom.hlsl` is the kernel mirror).
//!
//! A mip pyramid of RGBA16F half-steps, built from whatever texture the tonemap
//! is about to read, then folded back down with weighted tent-upsamples. The
//! final tent-sample to full res and the energy-conserving composite happen in
//! the tonemap PS, which saves a pass and a full-res texture.
//!
//! Compiled with D3DCompile (fxc, cs_5_0) like the rest of `gpu/tonemap.rs` — no
//! DXC dependency, so the CPU-renderer + plain-presentation sessions that never
//! load DXC still get glare.

use super::d3d12::Result;
use super::tonemap::{compile, SRV_HEAP_CAPACITY};
use windows::core::s;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::gfx::shaders::BLOOM_HLSL;

/// Must match `bloom::LEVELS`.
pub const LEVELS: usize = crate::bloom::LEVELS;

pub struct BloomGpu {
    /// (w, h, texture) per level; `[0]` is half-res.
    levels: Vec<(u32, u32, ID3D12Resource)>,
    root_sig: ID3D12RootSignature,
    down_pso: ID3D12PipelineState,
    up_pso: ID3D12PipelineState,
    /// Shader-visible heap: per level an SRV then a UAV (2 descriptors each),
    /// then one PERMANENT slot per tonemap SRV slot (see `create_source_srv`).
    heap: ID3D12DescriptorHeap,
    inc: u32,
    w: u32,
    h: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    dst_w: u32,
    dst_h: u32,
    src_w: u32,
    src_h: u32,
    weight: f32,
    _pad: [f32; 3],
}

impl BloomGpu {
    pub fn new(device: &ID3D12Device, w: u32, h: u32) -> Result<BloomGpu> {
        // Root signature: [0] SRV table (t0), [1] UAV table (u0), [2] 8 root
        // constants (b0), + one static linear/clamp sampler (0 DWORDs).
        // Clamp, not wrap: a wrapping or zero border would break the uniform-image
        // energy gate exactly at the frame's rim.
        let srv_range = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        }];
        let uav_range = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
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
                        pDescriptorRanges: srv_range.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: uav_range.as_ptr(),
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
                        Num32BitValues: 8,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let samp = [D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxLOD: D3D12_FLOAT32_MAX,
            ShaderRegister: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
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
            D3D12SerializeRootSignature(
                &rs_desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                Some(&mut errb),
            )
        }
        .map_err(|e| format!("bloom: D3D12SerializeRootSignature: {e}"))?;
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
        .map_err(|e| format!("bloom: CreateRootSignature: {e}"))?;

        let cs_down = compile(BLOOM_HLSL, s!("cs_down"), s!("cs_5_0"), "bloom cs_down")?;
        let cs_up = compile(BLOOM_HLSL, s!("cs_up"), s!("cs_5_0"), "bloom cs_up")?;
        let pso = |cs: &ID3DBlob, what: &str| -> Result<ID3D12PipelineState> {
            let d = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: unsafe { std::mem::transmute_copy(&root_sig) },
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: unsafe { cs.GetBufferPointer() },
                    BytecodeLength: unsafe { cs.GetBufferSize() },
                },
                ..Default::default()
            };
            unsafe { device.CreateComputePipelineState(&d) }
                .map_err(|e| format!("bloom: CreateComputePipelineState({what}): {e}"))
        };
        let down_pso = pso(&cs_down, "down")?;
        let up_pso = pso(&cs_up, "up")?;

        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                // Per level an SRV + a UAV, then one slot per tonemap SRV slot.
                NumDescriptors: LEVELS as u32 * 2 + SRV_HEAP_CAPACITY,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("bloom: CreateDescriptorHeap: {e}"))?;
        let inc = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };

        let mut b = BloomGpu {
            levels: Vec::new(),
            root_sig,
            down_pso,
            up_pso,
            heap,
            inc,
            w: 0,
            h: 0,
        };
        b.alloc(device, w, h)?;
        Ok(b)
    }

    fn alloc(&mut self, device: &ID3D12Device, w: u32, h: u32) -> Result<()> {
        self.levels.clear();
        let (mut mw, mut mh) = (w, h);
        for i in 0..LEVELS {
            mw = (mw / 2).max(1);
            mh = (mh / 2).max(1);
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: mw as u64,
                Height: mh,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let heap_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut res: Option<ID3D12Resource> = None;
            unsafe {
                device.CreateCommittedResource(
                    &heap_props,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut res,
                )
            }
            .map_err(|e| format!("bloom: CreateCommittedResource(level {i}): {e}"))?;
            let res = res.unwrap();

            let cpu = unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() };
            let srv_h = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu.ptr + ((i * 2) as u32 * self.inc) as usize,
            };
            let uav_h = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu.ptr + ((i * 2 + 1) as u32 * self.inc) as usize,
            };
            unsafe {
                device.CreateShaderResourceView(
                    &res,
                    Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV {
                                MipLevels: 1,
                                ..Default::default()
                            },
                        },
                    }),
                    srv_h,
                );
                device.CreateUnorderedAccessView(
                    &res,
                    None,
                    Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                        ..Default::default()
                    }),
                    uav_h,
                );
            }
            self.levels.push((mw, mh, res));
        }
        self.w = w;
        self.h = h;
        Ok(())
    }

    pub fn set_res(&mut self, device: &ID3D12Device, w: u32, h: u32) -> Result<()> {
        if self.w != w || self.h != h {
            self.alloc(device, w, h)?;
        }
        Ok(())
    }

    /// The SRV of level 0 — what the tonemap PS tent-samples for the halo.
    pub fn glare_srv_source(&self) -> &ID3D12Resource {
        &self.levels[0].2
    }
    pub fn glare_dims(&self) -> (u32, u32) {
        (self.levels[0].0, self.levels[0].1)
    }

    fn gpu(&self, idx: usize) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let s = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE { ptr: s.ptr + (idx as u32 * self.inc) as u64 }
    }

    fn uav_barrier(list: &ID3D12GraphicsCommandList, res: &ID3D12Resource) {
        let b = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(res) },
                }),
            },
        };
        unsafe { list.ResourceBarrier(&[b]) };
    }

    fn transition(
        list: &ID3D12GraphicsCommandList,
        res: &ID3D12Resource,
        from: D3D12_RESOURCE_STATES,
        to: D3D12_RESOURCE_STATES,
    ) {
        unsafe { list.ResourceBarrier(&[super::d3d12::transition(res, from, to)]) };
    }

    /// Register a tonemap source so the pyramid can read it.
    ///
    /// D3D12 allows exactly ONE CBV_SRV_UAV heap bound at a time and the levels'
    /// UAVs live in ours, so the source's SRV has to be in this heap too. It is
    /// CREATED here, once, in a permanent slot per tonemap SRV slot — NOT copied
    /// across at present time. Two reasons, either sufficient:
    ///
    ///   - `CopyDescriptors` may not READ from a shader-visible heap (they are
    ///     allowed to live in write-combine memory), and `tonemap::Passes`'s heap
    ///     is shader-visible. A per-present copy out of it is simply illegal.
    ///   - Rewriting a descriptor a still-executing frame references is UB, and
    ///     with frames in flight the previous present's dispatches are exactly
    ///     that — so a copy would also need a ring to be safe.
    ///
    /// A permanent slot has neither problem. Call it wherever the matching
    /// `Passes::create_srv` is called (gpu/mod.rs::wire_tonemap_src pairs them).
    pub fn create_source_srv(
        &self,
        device: &ID3D12Device,
        res: &ID3D12Resource,
        format: DXGI_FORMAT,
        slot: u32,
    ) {
        let base = unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() };
        let h = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + (Self::src_slot(slot) as u32 * self.inc) as usize,
        };
        unsafe {
            device.CreateShaderResourceView(
                res,
                Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                    },
                }),
                h,
            )
        };
    }

    /// Our heap slot holding the SRV for tonemap SRV slot `slot`.
    fn src_slot(slot: u32) -> usize {
        LEVELS * 2 + slot as usize
    }

    /// Record the pyramid: downsample to the coarsest level, then fold back down
    /// with weighted tent-upsamples. Leaves level 0 holding the glare halo, in
    /// PIXEL_SHADER_RESOURCE state for the tonemap.
    ///
    /// `srv_slot` is the TONEMAP slot of the source (registered by
    /// `create_source_srv`), whose resource state is the caller's business — it
    /// must be in NON_PIXEL_SHADER_RESOURCE, these being compute dispatches.
    /// Source DIMENSIONS are deliberately not a parameter: `cs_down` derives its
    /// tap from the destination's normalized UV and never reads them, so passing
    /// them in only invited the reader to think this pass adapts to a source size
    /// when it does not.
    pub fn record(&self, list: &ID3D12GraphicsCommandList, srv_slot: u32) {
        let src_slot = Self::src_slot(srv_slot);
        let wts = crate::bloom::level_weights();
        let groups = |n: u32| n.div_ceil(8);
        let last = LEVELS - 1;

        unsafe {
            list.SetComputeRootSignature(&self.root_sig);
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetPipelineState(&self.down_pso);
        }

        // Downsample. Every level starts in UNORDERED_ACCESS and ends as an SRV
        // for the next lap. `weight` pre-scales ONLY the coarsest level (it seeds
        // the upsample recurrence — see bloom.hlsl).
        for i in 0..LEVELS {
            let (dw, dh, res) = &self.levels[i];
            let src_desc = if i == 0 { self.gpu(src_slot) } else { self.gpu((i - 1) * 2) };
            let p = Params {
                dst_w: *dw,
                dst_h: *dh,
                // cs_down taps in NORMALIZED uv derived from the destination, so
                // it never reads src_size — the source's dims are genuinely not
                // its business (which is what lets level 0 take a source of any
                // size). Only cs_up needs them, for its tent's texel spacing.
                src_w: 0,
                src_h: 0,
                weight: if i == last { wts[last] } else { 1.0 },
                _pad: [0.0; 3],
            };
            unsafe {
                list.SetComputeRootDescriptorTable(0, src_desc);
                list.SetComputeRootDescriptorTable(1, self.gpu(i * 2 + 1));
                list.SetComputeRoot32BitConstants(2, 8, &p as *const _ as *const _, 0);
                list.Dispatch(groups(*dw), groups(*dh), 1);
            }
            Self::transition(
                list,
                res,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
        }

        // Upsample-combine, coarsest first: acc_i = w_i * L_i + tent(acc_{i+1}).
        // Level i is read-modify-written, so it goes back to UAV state while the
        // level above it stays an SRV.
        unsafe { list.SetPipelineState(&self.up_pso) };
        for i in (0..last).rev() {
            let (dw, dh, res) = &self.levels[i];
            let (sw, sh) = (self.levels[i + 1].0, self.levels[i + 1].1);
            Self::transition(
                list,
                res,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
            let p = Params {
                dst_w: *dw,
                dst_h: *dh,
                src_w: sw,
                src_h: sh,
                weight: wts[i],
                _pad: [0.0; 3],
            };
            unsafe {
                list.SetComputeRootDescriptorTable(0, self.gpu((i + 1) * 2));
                list.SetComputeRootDescriptorTable(1, self.gpu(i * 2 + 1));
                list.SetComputeRoot32BitConstants(2, 8, &p as *const _ as *const _, 0);
                list.Dispatch(groups(*dw), groups(*dh), 1);
            }
            Self::uav_barrier(list, res);
            Self::transition(
                list,
                res,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
        }

        // Level 0 now holds the halo; the tonemap PS reads it.
        Self::transition(
            list,
            &self.levels[0].2,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
    }

    /// Level 0's dims and resource — the gate reads the halo straight out of it.
    fn level0(&self) -> (u32, u32, &ID3D12Resource) {
        let (w, h, ref r) = self.levels[0];
        (w, h, r)
    }

    /// Put level 0 back in UAV state for next frame (the tonemap left it as a
    /// pixel-shader resource).
    pub fn restore(&self, list: &ID3D12GraphicsCommandList) {
        Self::transition(
            list,
            &self.levels[0].2,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        for (_, _, res) in self.levels.iter().skip(1) {
            Self::transition(
                list,
                res,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
        }
    }
}

/// GPU-vs-CPU parity for the glare pyramid — the `--check-gpu` half of the gate
/// family (`bloom::self_test` pins the math itself, closed-form and DLL-free).
///
/// Without this, `bloom.hlsl` was the one CPU/GPU mirror in the renderer with no
/// numeric gate: a swapped octave weight, a wrong barrier, a mis-derived tent
/// spacing or a bad descriptor slot would all still PRESENT — just wrongly — and
/// no suite would notice, because nothing else reads the presented image back.
/// Every other mirror here has one (the albedo A/B, the feed-plane ulp gates, the
/// wavefront-vs-reference bit A/B), so this closes the hole.
///
/// It scores level 0 — the halo, i.e. the entire pyramid's product: box
/// downsample, octave weights, tent-upsample recurrence, and every state
/// transition between them. The final tent-to-full-res and the composite live in
/// the tonemap PS and are a pasted copy of the same `tent`.
///
/// Tolerances are WIRING tolerances, in the spirit of the BC7 PSNR gate: the GPU
/// accumulates the pyramid in f16 through hardware bilinear taps while the CPU is
/// f32 throughout, so exact agreement was never the claim — but every error class
/// this is here to catch (a weight, a barrier, a slot, a pitch) moves the halo by
/// tens of percent or more, not by an f16 ulp.
pub fn self_test_gpu(hg: &mut super::trace::HeadlessGpu) -> Result<()> {
    use super::d3d12::{self, ReadbackBuffer, UploadBuffer};

    // Dimensions divisible by 2^LEVELS, so every downsample is an exact 2x and
    // the CPU's explicit 2x2 box and the GPU's single bilinear tap describe the
    // same footprint. (Off-by-2^k dims make them differ by design — a half-texel
    // shift — which is a real difference this gate should not be scoring.)
    const W: usize = 256;
    const H: usize = 128;
    assert!(W % (1 << LEVELS) == 0 && H % (1 << LEVELS) == 0);

    // A deterministic HDR image with the structure glare actually has to carry:
    // a very bright compact core (the sun — this is what the heavy tail is for),
    // a dim background, and a gradient so a channel or axis swap can't cancel.
    let mut src = vec![0.0f32; W * H * 3];
    for y in 0..H {
        for x in 0..W {
            let p = (y * W + x) * 3;
            let (fx, fy) = (x as f32 / W as f32, y as f32 / H as f32);
            src[p] = 0.05 + 0.30 * fx;
            src[p + 1] = 0.04 + 0.20 * fy;
            src[p + 2] = 0.03;
            if (x as i32 - 96).abs() <= 1 && (y as i32 - 40).abs() <= 1 {
                src[p] = 800.0;
                src[p + 1] = 700.0;
                src[p + 2] = 500.0;
            }
        }
    }

    // CPU reference: run the real pyramid and keep its halo.
    let mut cpu = crate::bloom::Bloom::new(W, H);
    let mut out = vec![0.0f32; W * H * 3];
    cpu.apply(&src, &mut out);
    let (hw, hh, cpu_halo) = cpu.halo();

    // GPU source: an RGBA16F texture holding the same image, in the state the
    // pyramid's compute dispatches require.
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: W as u64,
        Height: H as u32,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let mut tex: Option<ID3D12Resource> = None;
    unsafe {
        hg.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut tex,
        )
    }
    .map_err(|e| format!("bloom gate: CreateCommittedResource(src): {e}"))?;
    let tex = tex.unwrap();

    let bpp = 8; // RGBA16F
    let pitch = d3d12::aligned_pitch(W * bpp);
    let up = UploadBuffer::new(&hg.device, pitch * H)?;
    for y in 0..H {
        for x in 0..W {
            let p = (y * W + x) * 3;
            let px = [src[p], src[p + 1], src[p + 2], 1.0].map(half::f16::from_f32);
            let off = y * pitch + x * bpp;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    px.as_ptr() as *const u8,
                    up.ptr.add(off),
                    bpp,
                );
            }
        }
    }
    let fp = d3d12::footprint(DXGI_FORMAT_R16G16B16A16_FLOAT, W as u32, H as u32, bpp, 0);
    hg.run(|list| unsafe {
        list.CopyTextureRegion(
            &d3d12::loc_subresource(&tex),
            0,
            0,
            0,
            &d3d12::loc_footprint(&up.resource, fp),
            None,
        );
        list.ResourceBarrier(&[d3d12::transition(
            &tex,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        )]);
    })?;

    // Run the real pyramid, through the real entry points.
    let gpu = BloomGpu::new(&hg.device, W as u32, H as u32)?;
    gpu.create_source_srv(&hg.device, &tex, DXGI_FORMAT_R16G16B16A16_FLOAT, 0);
    hg.run(|list| gpu.record(list, 0))?;

    // Read level 0 back. `record` leaves it in PIXEL_SHADER_RESOURCE.
    let (gw, gh, lvl0) = gpu.level0();
    if gw as usize != hw || gh as usize != hh {
        return Err(format!(
            "bloom gate: halo dims disagree — gpu {gw}x{gh}, cpu {hw}x{hh}"
        ));
    }
    let gpitch = d3d12::aligned_pitch(gw as usize * bpp);
    let rb = ReadbackBuffer::new(&hg.device, gpitch * gh as usize)?;
    let gfp = d3d12::footprint(DXGI_FORMAT_R16G16B16A16_FLOAT, gw, gh, bpp, 0);
    hg.run(|list| unsafe {
        list.ResourceBarrier(&[d3d12::transition(
            lvl0,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyTextureRegion(
            &d3d12::loc_footprint(&rb.resource, gfp),
            0,
            0,
            0,
            &d3d12::loc_subresource(lvl0),
            None,
        );
        list.ResourceBarrier(&[d3d12::transition(
            lvl0,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )]);
    })?;
    let mut ptr = std::ptr::null_mut();
    unsafe { rb.resource.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("Map: {e}"))?;
    let bytes =
        unsafe { std::slice::from_raw_parts(ptr as *const u8, gpitch * gh as usize) }.to_vec();
    unsafe { rb.resource.Unmap(0, None) };

    // Score. Relative per channel, against the CPU's own magnitude — the halo
    // spans ~4 orders of magnitude between the core and the far tail, so an
    // absolute bound would be meaningless at one end or the other.
    let (mut sum_rel, mut worst, mut n, mut peak) = (0.0f64, 0.0f32, 0usize, 0.0f32);
    for y in 0..gh as usize {
        for x in 0..gw as usize {
            let off = y * gpitch + x * bpp;
            let g = [0usize, 1, 2].map(|k| {
                let b = [bytes[off + k * 2], bytes[off + k * 2 + 1]];
                half::f16::from_bits(u16::from_le_bytes(b)).to_f32()
            });
            let c = cpu_halo[y * gw as usize + x];
            let c = [c.x, c.y, c.z];
            for k in 0..3 {
                peak = peak.max(c[k]);
                // Skip channels too dim for f16 to resolve: below ~1e-3 the
                // storage quantum IS the value, and a "relative" error there
                // measures f16, not the port.
                if c[k] < 1e-3 {
                    continue;
                }
                let rel = ((g[k] - c[k]) / c[k]).abs();
                sum_rel += rel as f64;
                worst = worst.max(rel);
                n += 1;
            }
        }
    }
    if n == 0 || peak <= 0.0 {
        return Err("bloom gate: the CPU halo is empty — the probe image never bloomed".into());
    }
    let mean_rel = (sum_rel / n as f64) as f32;
    // A wrong weight/barrier/slot/pitch moves the halo by tens of percent or
    // more; f16 accumulation through 6 octaves of hardware bilinear costs well
    // under one. Never widen these to make a failing port pass.
    if mean_rel > 0.02 || worst > 0.10 {
        return Err(format!(
            "bloom gate: GPU halo vs CPU — mean rel {:.4} (limit 0.02), worst {:.4} \
             (limit 0.10) over {n} channels",
            mean_rel, worst
        ));
    }
    eprintln!(
        "check-gpu: bloom pyramid vs CPU ({gw}x{gh} halo): mean rel {:.4} | worst {:.4} | \
         peak {:.1} | {n} channels scored",
        mean_rel, worst, peak
    );
    Ok(())
}
