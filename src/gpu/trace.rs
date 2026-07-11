//! GPU-resident tracer (M1: toolchain + dispatch plumbing). This module owns
//! everything the wavefront pipeline shares: the capability gates (DXR 1.1 +
//! SM 6.5 — hard requirements, the CPU path is the fallback), the one compute
//! root signature every kernel binds, the dispatch-only command signature
//! that makes ExecuteIndirect act as DispatchIndirect, and a headless
//! device harness for `--check-gpu` (swapchain-free by construction).
//!
//! Root signature layout (root descriptors throughout — no descriptor-heap
//! management for buffer-only passes; the TLAS binds directly as a root SRV):
//!   param 0                 root CBV  b0   frame constants
//!   param 1                 constants b1   4 DWORDs per-dispatch push
//!   param 2 .. 2+NUM_UAVS   root UAV  u0.. queues/pools/planes
//!   param 10 .. 10+NUM_SRVS root SRV  t0.. BVH/scene/TLAS
//!   param 18                table          u8 = the RGBA16F HDR output
//!                                          (typed texture UAVs can't be root
//!                                          descriptors — the one exception)

use super::adapter;
use super::d3d12::{self, committed_buffer, transition, uav_barrier, Result};
use super::dxc::Dxc;
use crate::bvh::Bvh;
use crate::camera::CamBasis;
use crate::scene::{MatKind, Scene};
use crate::shade::Quality;
use glam::Vec3A;
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32_UINT};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

pub const RP_FRAME_CBV: u32 = 0;
pub const RP_PUSH: u32 = 1;
pub const RP_UAV0: u32 = 2;
pub const NUM_UAVS: u32 = 14;
pub const RP_SRV0: u32 = RP_UAV0 + NUM_UAVS;
pub const NUM_SRVS: u32 = 8;
pub const RP_TEX: u32 = RP_SRV0 + NUM_SRVS;
// The G-buffer pack root UAV (register u15), appended AFTER the table so the
// established param indices never renumber. 53/64 root-signature DWORDs.
pub const RP_GBUF: u32 = RP_TEX + 1;
/// Upscaler feed-target texture UAVs: registers u16..u22, riding the u14
/// descriptor table as a second range at heap slots 1..7 (0 DWORDs extra).
/// The register/type layout is shared by both feed kernels (feed.hlsl).
pub const NUM_FEED: u32 = 7;
pub const FEED_COLOR: u32 = 16; // RGBA16F (both upscalers)
pub const FEED_NR: u32 = 17; // RGBA16F normal+rough (RR)
pub const FEED_DEPTH: u32 = 18; // R32F (both; encoding differs per kernel)
pub const FEED_MVEC: u32 = 19; // RG16F (both)
pub const FEED_ALB: u32 = 20; // RGBA8 diffuse albedo (RR)
pub const FEED_SPEC: u32 = 21; // RGBA8 specular albedo (RR)
pub const FEED_SPECHIT: u32 = 22; // R16F spec hit distance (RR)

// SRV register assignments (t0..t7) — shared across every kernel; a kernel
// declares only what it reads, DXC strips the rest.
pub const SRV_BVH_NODES: u32 = 0;
pub const SRV_TRI_IDX: u32 = 1;
pub const SRV_POSITIONS: u32 = 2;
pub const SRV_NORMALS: u32 = 3;
pub const SRV_INDICES: u32 = 4;
pub const SRV_TRI_MAT: u32 = 5;
pub const SRV_MATERIALS: u32 = 6;
pub const SRV_TLAS: u32 = 7;

// UAV register assignments (u0..): per-pixel planes, then queue machinery.
// u5/u6/u7/u9 are generic binding points: tile queues + primary cut pool
// during the quadtree levels, hemi cell/leaf queues + hemi cut pool during
// the hemisphere passes (rebound per dispatch phase).
pub const UAV_ACCUM: u32 = 0;
pub const UAV_TBUF: u32 = 1;
pub const UAV_INFO: u32 = 2;
pub const UAV_COUNTERS: u32 = 3;
pub const UAV_ARGS: u32 = 4;
pub const UAV_QIN: u32 = 5;
pub const UAV_QOUT: u32 = 6;
pub const UAV_QLEAF: u32 = 7;
pub const UAV_QSKY: u32 = 8;
pub const UAV_CUT: u32 = 9;
pub const UAV_PARTIAL: u32 = 10;
pub const UAV_AMBW: u32 = 11;
pub const UAV_HBUF: u32 = 12;
pub const UAV_HEMI_PTS: u32 = 13;

// counters[] slots — mirror of ctr.hlsli.
pub const CTR_TILE_A: u32 = 0;
pub const CTR_TILE_B: u32 = 1;
pub const CTR_LEAF: u32 = 2;
pub const CTR_SKY: u32 = 3;
pub const CTR_CUT: u32 = 4;
pub const CTR_OVERFLOW: u32 = 5;
pub const CTR_CUT_FALLBACK: u32 = 6;
pub const CTR_SPLIT: u32 = 7;
pub const CTR_BLOCKED: u32 = 8;
pub const CTR_HEMI_PT: u32 = 9;
pub const CTR_HEMI_A: u32 = 10;
pub const CTR_HEMI_B: u32 = 11;
pub const CTR_HEMI_LEAF: u32 = 12;
pub const CTR_HEMI_CUT: u32 = 13;
pub const CTR_HEMI_EMPTY: u32 = 14;
pub const CTR_HEMI_RAYS: u32 = 15;
pub const CTR_V_FALSE_EMPTY: u32 = 16;
pub const CTR_V_TMIN: u32 = 17;
pub const CTR_COUNT: u32 = 24;

// Indirect-args buffer slots: level d at slot d (depth_full <= 11 asserted
// at init); hemi + leaf/sky passes at the top.
const ARG_HEMI_ROOT: u32 = 11;
const ARG_HEMI_CELL: u32 = 12;
const ARG_HEMI_LEAF: u32 = 13;
const ARG_LEAF: u32 = 14;
const ARG_SKY: u32 = 15;
const NO_RESET: u32 = 0xffff_ffff;

/// Hemisphere points per batch: bounds the transient hemi queue/pool memory
/// (queues are sized to batch x 4^(depth-1) — bounded, cannot overflow;
/// ~300 MB at this size). Bigger batches amortize the barrier-serialized
/// per-batch drains — 4096 measured 294 ms/frame for 1080p GI, 16384 is the
/// sweet spot on a 24 GB card.
pub const HEMI_BATCH: u32 = 16384;
/// Max fb.depth the hemi queue sizing supports (presets top out at 4).
const HEMI_MAX_DEPTH: u32 = 4;

const CB_STRIDE: usize = 512; // root-CBV alignment (FrameCb is 304 bytes)

/// Quadtree depth to the leaf frontier: smallest D with
/// max(rw, rh) / 2^D <= LEAF_TILE (temporal.rs uses the same formula).
pub fn depth_full(rw: u32, rh: u32) -> u32 {
    let m = rw.max(rh) as u64;
    let mut d = 0;
    let mut s = 8u64;
    while s < m {
        s *= 2;
        d += 1;
    }
    d
}

const SMOKE_HLSL: &str = include_str!("shaders/smoke.hlsl");
const TRACE_COMMON_HLSLI: &str = include_str!("shaders/trace_common.hlsli");
const CTR_HLSLI: &str = include_str!("shaders/ctr.hlsli");
const QUEUES_HLSLI: &str = include_str!("shaders/queues.hlsli");
const FRUSTUM_HLSLI: &str = include_str!("shaders/frustum.hlsli");
const RT_HLSLI: &str = include_str!("shaders/rt.hlsli");
const SHADE_HLSLI: &str = include_str!("shaders/shade.hlsli");
const HEMI_HLSLI: &str = include_str!("shaders/hemi.hlsli");
const REFERENCE_HLSL: &str = include_str!("shaders/reference.hlsl");
const RESOLVE_HLSL: &str = include_str!("shaders/resolve.hlsl");
const WAVEFRONT_HLSL: &str = include_str!("shaders/wavefront.hlsl");
const LEAF_HLSL: &str = include_str!("shaders/leaf.hlsl");
const HEMI_WAVE_HLSL: &str = include_str!("shaders/hemi_wave.hlsl");
const HEMI_LEAF_HLSL: &str = include_str!("shaders/hemi_leaf.hlsl");
const COMPOSE_HLSL: &str = include_str!("shaders/compose.hlsl");
const FEED_HLSL: &str = include_str!("shaders/feed.hlsl");

/// What the GPU tracer requires, queried once. RayQuery in compute needs
/// RaytracingTier 1.1 AND shader model 6.5; missing either is a clean
/// "use the CPU path" story, never a degraded half-mode.
pub struct Caps {
    pub rt_tier: i32,
    pub shader_model: i32,
}

pub fn query_caps(device: &ID3D12Device) -> Result<Caps> {
    let mut o5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5,
            &mut o5 as *mut _ as *mut _,
            std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32,
        )
    }
    .map_err(|e| format!("CheckFeatureSupport(OPTIONS5): {e}"))?;
    // Highest-supported query: seed with the max we understand; the runtime
    // clamps DOWN to what it supports (an old runtime errors on unknown
    // values, so retry with the floor before giving up).
    let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL_6_7 };
    let sm_probe = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            &mut sm as *mut _ as *mut _,
            std::mem::size_of::<D3D12_FEATURE_DATA_SHADER_MODEL>() as u32,
        )
    };
    if sm_probe.is_err() {
        sm.HighestShaderModel = D3D_SHADER_MODEL_6_5;
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                &mut sm as *mut _ as *mut _,
                std::mem::size_of::<D3D12_FEATURE_DATA_SHADER_MODEL>() as u32,
            )
        }
        .map_err(|e| format!("CheckFeatureSupport(SHADER_MODEL): {e}"))?;
    }
    Ok(Caps { rt_tier: o5.RaytracingTier.0, shader_model: sm.HighestShaderModel.0 })
}

/// Errors with the specific missing capability (the message main.rs surfaces
/// before falling back to the CPU path).
pub fn require_caps(device: &ID3D12Device) -> Result<Caps> {
    let caps = query_caps(device)?;
    let mut missing = Vec::new();
    if caps.rt_tier < D3D12_RAYTRACING_TIER_1_1.0 {
        missing.push(format!(
            "DXR raytracing tier 1.1 (inline RayQuery) — device reports tier {}",
            caps.rt_tier
        ));
    }
    if caps.shader_model < D3D_SHADER_MODEL_6_5.0 {
        missing.push(format!(
            "shader model 6.5 — device reports 0x{:x}",
            caps.shader_model
        ));
    }
    if device.cast::<ID3D12Device5>().is_err() {
        missing.push("ID3D12Device5 (acceleration-structure builds)".into());
    }
    if missing.is_empty() {
        Ok(caps)
    } else {
        Err(format!("GPU tracing unsupported here: {}", missing.join("; ")))
    }
}

/// The shared compute root signature (layout in the module docs).
pub fn create_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature> {
    let mut params: Vec<D3D12_ROOT_PARAMETER> = Vec::new();
    params.push(D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    });
    params.push(D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: 1,
                RegisterSpace: 0,
                Num32BitValues: 4,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    });
    for i in 0..NUM_UAVS {
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: i, RegisterSpace: 0 },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
    }
    for i in 0..NUM_SRVS {
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: i, RegisterSpace: 0 },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
    }
    // The one descriptor table (1 DWORD total, both ranges): u14 = the typed
    // RGBA16F output texture (resolve pass), u16..u22 = the upscaler feed
    // targets (heap slots 1..7; wire_feed builds the descriptors, null
    // elsewhere — RS 1.0 descriptors are volatile, only accessed slots must
    // be valid). u15 is skipped: it's the RP_GBUF root UAV.
    // `ranges` must outlive serialization below.
    let ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: 1,
            BaseShaderRegister: NUM_UAVS,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: NUM_FEED,
            BaseShaderRegister: NUM_UAVS + 2,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 1,
        },
    ];
    params.push(D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 2,
                pDescriptorRanges: ranges.as_ptr(),
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    });
    // RP_GBUF: the G-buffer pack (u15), appended last (see the const note).
    params.push(D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: NUM_UAVS + 1, RegisterSpace: 0 },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    });
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let mut blob = None;
    let mut errb = None;
    unsafe {
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut errb))
    }
    .map_err(|e| format!("D3D12SerializeRootSignature(compute): {e}"))?;
    let blob = blob.unwrap();
    unsafe {
        device.CreateRootSignature(
            0,
            std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()),
        )
    }
    .map_err(|e| format!("CreateRootSignature(compute): {e}"))
}

/// Dispatch-only command signature: ExecuteIndirect over one 12-byte
/// (x, y, z) record IS D3D12's DispatchIndirect. Null root signature —
/// no root-argument changes ride the indirect stream.
pub fn create_dispatch_signature(device: &ID3D12Device) -> Result<ID3D12CommandSignature> {
    let arg = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: 12,
        NumArgumentDescs: 1,
        pArgumentDescs: &arg,
        NodeMask: 0,
    };
    let mut sig: Option<ID3D12CommandSignature> = None;
    unsafe { device.CreateCommandSignature(&desc, None, &mut sig) }
        .map_err(|e| format!("CreateCommandSignature(dispatch): {e}"))?;
    Ok(sig.unwrap())
}

pub fn compute_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    dxil: &[u8],
    what: &str,
) -> Result<ID3D12PipelineState> {
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: dxil.as_ptr() as *const _,
            BytecodeLength: dxil.len(),
        },
        ..Default::default()
    };
    unsafe { device.CreateComputePipelineState(&desc) }
        .map_err(|e| format!("CreateComputePipelineState({what}): {e}"))
}

/// Minimal device/queue/list/fence harness for `--check-gpu` — no window, no
/// swapchain, no Streamline. Interactive mode uses `D3d` instead; everything
/// recorded against this harness records identically against that one.
pub struct HeadlessGpu {
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    alloc: ID3D12CommandAllocator,
    pub list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: HANDLE,
    next: u64,
    pub adapter_name: String,
}

impl HeadlessGpu {
    pub fn new(debug: bool) -> Result<Self> {
        let factory = adapter::create_factory(debug).map_err(|e| format!("factory: {e}"))?;
        let pick = adapter::pick(&factory, false)?;
        let device = d3d12::create_device(&pick.adapter, debug)?;
        let queue = d3d12::create_queue(&device)?;
        let alloc: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                .map_err(|e| format!("CreateCommandAllocator: {e}"))?;
        let list: ID3D12GraphicsCommandList =
            unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None) }
                .map_err(|e| format!("CreateCommandList: {e}"))?;
        unsafe { list.Close() }.map_err(|e| format!("initial Close: {e}"))?;
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .map_err(|e| format!("CreateFence: {e}"))?;
        let event =
            unsafe { CreateEventW(None, false, false, None) }.map_err(|e| format!("event: {e}"))?;
        Ok(Self { device, queue, alloc, list, fence, event, next: 1, adapter_name: pick.name })
    }

    /// Record + execute + block. The `--check-gpu` cadence: correctness
    /// first, wall-clock timing is a separate explicit segment.
    pub fn run<F: FnOnce(&ID3D12GraphicsCommandList)>(&mut self, f: F) -> Result<()> {
        unsafe { self.alloc.Reset() }.map_err(|e| format!("alloc Reset: {e}"))?;
        unsafe { self.list.Reset(&self.alloc, None) }.map_err(|e| format!("list Reset: {e}"))?;
        f(&self.list);
        unsafe { self.list.Close() }.map_err(|e| format!("list Close: {e}"))?;
        let lists = [Some(self.list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { self.queue.ExecuteCommandLists(&lists) };
        let v = self.next;
        self.next += 1;
        unsafe { self.queue.Signal(&self.fence, v) }.map_err(|e| format!("Signal: {e}"))?;
        if unsafe { self.fence.GetCompletedValue() } < v {
            unsafe { self.fence.SetEventOnCompletion(v, self.event) }
                .map_err(|e| format!("SetEventOnCompletion: {e}"))?;
            unsafe { WaitForSingleObject(self.event, INFINITE) };
        }
        Ok(())
    }

    /// Copy `size` bytes out of `src` (currently in `state`) and map them.
    pub fn read_buffer(
        &mut self,
        src: &ID3D12Resource,
        state: D3D12_RESOURCE_STATES,
        size: usize,
    ) -> Result<Vec<u8>> {
        let rb = d3d12::ReadbackBuffer::new(&self.device, size)?;
        self.run(|list| unsafe {
            if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                list.ResourceBarrier(&[transition(src, state, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            }
            list.CopyBufferRegion(&rb.resource, 0, src, 0, size as u64);
            if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                list.ResourceBarrier(&[transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, state)]);
            }
        })?;
        let mut ptr = std::ptr::null_mut();
        unsafe { rb.resource.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("Map: {e}"))?;
        let out = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) }.to_vec();
        unsafe { rb.resource.Unmap(0, None) };
        Ok(out)
    }
}

impl Drop for HeadlessGpu {
    fn drop(&mut self) {
        // Drain before releasing (the run() calls already block, but be safe
        // against an early-error exit mid-record).
        let v = self.next;
        if unsafe { self.queue.Signal(&self.fence, v) }.is_ok()
            && unsafe { self.fence.GetCompletedValue() } < v
            && unsafe { self.fence.SetEventOnCompletion(v, self.event) }.is_ok()
        {
            unsafe { WaitForSingleObject(self.event, INFINITE) };
        }
        let _ = unsafe { CloseHandle(self.event) };
    }
}

/// M1 gate: the full dispatch-plumbing chain — seed writes a counter, prep
/// turns it into DispatchIndirect args, ExecuteIndirect runs the consumer,
/// readback verifies every element and the exact group roundup. This is the
/// same seed → prep → indirect-consume shape every level of the real
/// wavefront uses.
pub fn smoke_test(hg: &mut HeadlessGpu, dxc: &Dxc, debug: bool) -> Result<()> {
    const FILL_N: u32 = 555; // deliberately not a multiple of 64

    let root_sig = create_root_signature(&hg.device)?;
    let cmd_sig = create_dispatch_signature(&hg.device)?;
    let seed = compute_pso(
        &hg.device,
        &root_sig,
        &dxc.compile(SMOKE_HLSL, "cs_seed", "cs_6_5", "smoke seed", debug)?,
        "smoke seed",
    )?;
    let prep = compute_pso(
        &hg.device,
        &root_sig,
        &dxc.compile(SMOKE_HLSL, "cs_prep", "cs_6_5", "smoke prep", debug)?,
        "smoke prep",
    )?;
    let fill = compute_pso(
        &hg.device,
        &root_sig,
        &dxc.compile(SMOKE_HLSL, "cs_fill", "cs_6_5", "smoke fill", debug)?,
        "smoke fill",
    )?;

    let uaf = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let counters = committed_buffer(&hg.device, 8, uaf, ua)?;
    let args = committed_buffer(&hg.device, 12, uaf, ua)?;
    let outbuf = committed_buffer(&hg.device, FILL_N as u64 * 4, uaf, ua)?;

    hg.run(|list| unsafe {
        list.SetComputeRootSignature(&root_sig);
        let push = [FILL_N, 0, 0, 0];
        list.SetComputeRoot32BitConstants(RP_PUSH, 4, push.as_ptr() as *const _, 0);
        list.SetComputeRootUnorderedAccessView(RP_UAV0, counters.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(RP_UAV0 + 1, args.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(RP_UAV0 + 2, outbuf.GetGPUVirtualAddress());

        list.SetPipelineState(&seed);
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[uav_barrier(None)]);

        list.SetPipelineState(&prep);
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[uav_barrier(None)]);
        list.ResourceBarrier(&[transition(&args, ua, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)]);

        list.SetPipelineState(&fill);
        list.ExecuteIndirect(&cmd_sig, 1, &args, 0, None, 0);
        list.ResourceBarrier(&[uav_barrier(None)]);
    })?;

    let out = hg.read_buffer(&outbuf, ua, FILL_N as usize * 4)?;
    for i in 0..FILL_N {
        let got = u32::from_le_bytes(out[i as usize * 4..][..4].try_into().unwrap());
        let want = i ^ 0x00C0_FFEE;
        if got != want {
            return Err(format!("smoke: outbuf[{i}] = {got:#x}, expected {want:#x}"));
        }
    }
    let a = hg.read_buffer(&args, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, 12)?;
    let groups: Vec<u32> =
        a.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let want = [FILL_N.div_ceil(64), 1, 1];
    if groups != want {
        return Err(format!("smoke: indirect args {groups:?}, expected {want:?}"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scene on the GPU: SoA geometry/material buffers (SRVs), the software BVH
// (frustum kernels, M3), and the DXR BLAS/TLAS (every actual ray).
// ---------------------------------------------------------------------------

/// bvh.rs::BvhNode packed to 32 bytes for StructuredBuffer<BvhNode>.
#[repr(C)]
struct GpuBvhNode {
    mn: [f32; 3],
    left_first: u32,
    mx: [f32; 3],
    count: u32,
}

/// scene.rs::Material packed for StructuredBuffer<Mat> (shade.hlsli).
#[repr(C)]
struct GpuMat {
    albedo: [f32; 3],
    roughness: f32,
    metallic: f32,
    anisotropy: f32,
    kind: u32, // 0 = diffuse, 1 = marble
    scale: f32,
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// One pending default-heap upload: staging src, dest, size, post-copy state.
struct PendingUpload {
    src: d3d12::UploadBuffer,
    dst: ID3D12Resource,
    size: usize,
    after: D3D12_RESOURCE_STATES,
}

struct SceneStaging {
    uploads: Vec<PendingUpload>,
    scratch: ID3D12Resource,
    /// TLAS instance descs live in an upload buffer for the build's duration.
    _instance: d3d12::UploadBuffer,
    instance_va: u64,
    blas_scratch_size: u64,
    tlas_scratch_size: u64,
}

pub struct SceneGpu {
    pub bvh_nodes: ID3D12Resource,
    pub tri_idx: ID3D12Resource,
    pub positions: ID3D12Resource,
    pub normals: ID3D12Resource,
    pub indices: ID3D12Resource,
    pub tri_mat: ID3D12Resource,
    pub materials: ID3D12Resource,
    pub blas: ID3D12Resource,
    pub tlas: ID3D12Resource,
    n_verts: u32,
    n_tris: u32,
    staging: Option<SceneStaging>,
}

fn upload_pair(
    device: &ID3D12Device,
    bytes: &[u8],
    after: D3D12_RESOURCE_STATES,
) -> Result<(ID3D12Resource, PendingUpload)> {
    let size = bytes.len().max(4);
    let dst = committed_buffer(
        device,
        size as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let src = d3d12::UploadBuffer::new(device, size)?;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), src.ptr, bytes.len()) };
    let pending = PendingUpload { src, dst: dst.clone(), size, after };
    Ok((dst, pending))
}

impl SceneGpu {
    pub fn new(device: &ID3D12Device, scene: &Scene, bvh: &Bvh) -> Result<Self> {
        let device5: ID3D12Device5 = device
            .cast()
            .map_err(|e| format!("ID3D12Device5 (require_caps should have gated): {e}"))?;

        // --- pack CPU-side data ---
        let nodes: Vec<GpuBvhNode> = bvh
            .nodes
            .iter()
            .map(|n| GpuBvhNode {
                mn: [n.aabb.min.x, n.aabb.min.y, n.aabb.min.z],
                left_first: n.left_first,
                mx: [n.aabb.max.x, n.aabb.max.y, n.aabb.max.z],
                count: n.count,
            })
            .collect();
        let positions: Vec<[f32; 3]> = scene.positions.iter().map(|p| [p.x, p.y, p.z]).collect();
        let normals: Vec<[f32; 3]> = scene.normals.iter().map(|n| [n.x, n.y, n.z]).collect();
        let indices: Vec<u32> = scene.indices.iter().flatten().copied().collect();
        let materials: Vec<GpuMat> = scene
            .materials
            .iter()
            .map(|m| GpuMat {
                albedo: [m.albedo.x, m.albedo.y, m.albedo.z],
                roughness: m.roughness,
                metallic: m.metallic,
                anisotropy: m.anisotropy,
                kind: match m.kind {
                    MatKind::Diffuse => 0,
                    MatKind::Marble { .. } => 1,
                },
                scale: match m.kind {
                    MatKind::Marble { scale } => scale,
                    _ => 0.0,
                },
            })
            .collect();

        let srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let mut uploads = Vec::new();
        let mut up = |bytes: &[u8]| -> Result<ID3D12Resource> {
            let (dst, p) = upload_pair(device, bytes, srv)?;
            uploads.push(p);
            Ok(dst)
        };
        let bvh_nodes = up(as_bytes(&nodes))?;
        let tri_idx = up(as_bytes(&bvh.tri_idx))?;
        let positions_b = up(as_bytes(&positions))?;
        let normals_b = up(as_bytes(&normals))?;
        let indices_b = up(as_bytes(&indices))?;
        let tri_mat = up(as_bytes(&scene.tri_mat))?;
        let materials_b = up(as_bytes(&materials))?;

        // --- acceleration-structure sizing ---
        let n_verts = scene.positions.len() as u32;
        let n_tris = scene.indices.len() as u32;
        let geom = geometry_desc(&positions_b, &indices_b, n_verts, n_tris);
        let blas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: &geom,
            },
        };
        let mut blas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        unsafe {
            device5.GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &mut blas_info)
        };
        let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: 0,
            },
        };
        let mut tlas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        unsafe {
            device5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut tlas_info)
        };

        let uaf = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let as_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        let blas =
            committed_buffer(device, blas_info.ResultDataMaxSizeInBytes, uaf, as_state)?;
        let tlas =
            committed_buffer(device, tlas_info.ResultDataMaxSizeInBytes, uaf, as_state)?;
        let scratch = committed_buffer(
            device,
            blas_info.ScratchDataSizeInBytes.max(tlas_info.ScratchDataSizeInBytes),
            uaf,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;

        // Identity-instance TLAS: InstanceID 0, mask 0xff, contribution 0,
        // no flags (geometry is OPAQUE; two-sidedness comes from tracing
        // with no cull flags).
        let instance = d3d12::UploadBuffer::new(
            device,
            std::mem::size_of::<D3D12_RAYTRACING_INSTANCE_DESC>(),
        )?;
        let mut idesc: D3D12_RAYTRACING_INSTANCE_DESC = unsafe { std::mem::zeroed() };
        idesc.Transform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        idesc._bitfield1 = 0xff << 24; // InstanceID 0 | InstanceMask 0xff
        idesc._bitfield2 = 0;
        idesc.AccelerationStructure = unsafe { blas.GetGPUVirtualAddress() };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &idesc as *const _ as *const u8,
                instance.ptr,
                std::mem::size_of::<D3D12_RAYTRACING_INSTANCE_DESC>(),
            )
        };
        let instance_va = unsafe { instance.resource.GetGPUVirtualAddress() };

        Ok(Self {
            bvh_nodes,
            tri_idx,
            positions: positions_b,
            normals: normals_b,
            indices: indices_b,
            tri_mat,
            materials: materials_b,
            blas,
            tlas,
            n_verts,
            n_tris,
            staging: Some(SceneStaging {
                uploads,
                scratch,
                _instance: instance,
                instance_va,
                blas_scratch_size: blas_info.ScratchDataSizeInBytes,
                tlas_scratch_size: tlas_info.ScratchDataSizeInBytes,
            }),
        })
    }

    /// Record the one-time init: buffer copies, transitions, BLAS + TLAS
    /// builds. Call once, execute, then `free_staging`.
    pub fn record_upload(&self, list: &ID3D12GraphicsCommandList) -> Result<()> {
        let st = self.staging.as_ref().ok_or("scene staging already freed")?;
        let list4: ID3D12GraphicsCommandList4 =
            list.cast().map_err(|e| format!("ID3D12GraphicsCommandList4: {e}"))?;
        let mut barriers = Vec::new();
        for u in &st.uploads {
            unsafe { list.CopyBufferRegion(&u.dst, 0, &u.src.resource, 0, u.size as u64) };
            barriers.push(transition(&u.dst, D3D12_RESOURCE_STATE_COPY_DEST, u.after));
        }
        unsafe { list.ResourceBarrier(&barriers) };

        let geom = geometry_desc(&self.positions, &self.indices, self.n_verts, self.n_tris);
        let scratch_va = unsafe { st.scratch.GetGPUVirtualAddress() };
        let _ = (st.blas_scratch_size, st.tlas_scratch_size);
        let blas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: unsafe { self.blas.GetGPUVirtualAddress() },
            Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: &geom,
                },
            },
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_va,
        };
        unsafe { list4.BuildRaytracingAccelerationStructure(&blas_desc, None) };
        // The TLAS build reads the BLAS; scratch is also reused.
        unsafe { list.ResourceBarrier(&[uav_barrier(None)]) };
        let tlas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: unsafe { self.tlas.GetGPUVirtualAddress() },
            Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: st.instance_va,
                },
            },
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_va,
        };
        unsafe { list4.BuildRaytracingAccelerationStructure(&tlas_desc, None) };
        unsafe { list.ResourceBarrier(&[uav_barrier(None)]) };
        Ok(())
    }

    /// Drop the staging buffers + scratch after the upload list executed.
    pub fn free_staging(&mut self) {
        self.staging = None;
    }
}

fn geometry_desc(
    positions: &ID3D12Resource,
    indices: &ID3D12Resource,
    n_verts: u32,
    n_tris: u32,
) -> D3D12_RAYTRACING_GEOMETRY_DESC {
    D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        // OPAQUE == the kernels' FORCE_OPAQUE assumption (no any-hit ever).
        Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: DXGI_FORMAT_R32_UINT,
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: n_tris * 3,
                VertexCount: n_verts,
                IndexBuffer: unsafe { indices.GetGPUVirtualAddress() },
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: unsafe { positions.GetGPUVirtualAddress() },
                    StrideInBytes: 12,
                },
            },
        },
    }
}

// ---------------------------------------------------------------------------
// Frame constants + the tracer itself.
// ---------------------------------------------------------------------------

pub const FLAG_ACCUM: u32 = 1;
pub const FLAG_JITTER: u32 = 2;
pub const FLAG_FRAME_JITTER: u32 = 4;
pub const FLAG_VERIFY: u32 = 8;
/// G-buffer pack writes on. Set ONLY when the pack is full-size (upscaler
/// sessions) — root UAVs have no bounds check and the plain-session pack is
/// a 64-byte dummy, so this flag is memory safety, not an optimization.
pub const FLAG_GBUF: u32 = 16;
pub const FLAG_HAS_PREV: u32 = 32;

/// Mirror of `cbuffer Frame` in trace_common.hlsli (304 bytes, 16-aligned
/// rows — float3s ride in float4 slots with scalars packed in .w).
#[repr(C)]
#[derive(Clone, Copy)]
struct FrameCb {
    cam_origin: [f32; 4],
    cam_forward: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    sun: [f32; 4],
    light_center: [f32; 4],
    light_u: [f32; 4],
    light_v: [f32; 4],
    light_color: [f32; 4],
    rw: u32,
    rh: u32,
    frame: u32,
    flags: u32,
    shadow_samples: u32,
    ao_samples: u32,
    reflections: u32,
    _pad0: u32,
    frame_jitter: [f32; 2],
    _pad1: f32,
    _pad2: f32,
    cap_tile: u32,
    cap_leaf: u32,
    cap_sky: u32,
    cap_cut: u32,
    fb_mode: u32,
    fb_depth: u32,
    hemi_batch: u32,
    cap_hemi_pt: u32,
    cap_hemi_cell: u32,
    cap_hemi_leaf: u32,
    cap_hemi_cut: u32,
    _pad3: u32,
    // Previous frame's camera basis for G-buffer MVs; near/far ride the w
    // slots of the last two rows (scene-static, from dlss::near_far).
    prev_origin: [f32; 4],  // xyz; w = prev inv_w
    prev_forward: [f32; 4], // xyz; w = prev inv_h
    prev_right: [f32; 4],   // xyz; w = near
    prev_up: [f32; 4],      // xyz; w = far
}
// The HLSL cbuffer is hand-mirrored across 7 concatenated compile units —
// a size drift here corrupts every field after the drift point.
const _: () = assert!(std::mem::size_of::<FrameCb>() == 304);

/// Everything that varies per frame, CPU-side.
pub struct FrameParams {
    pub cam: CamBasis,
    pub frame: u32,
    pub accumulate: bool,
    pub jitter: bool,
    pub frame_jitter: Option<(f32, f32)>,
    /// Previous frame's camera basis for G-buffer motion vectors (upscaler
    /// sessions; None = mv (0,0), consumed as disocclusion).
    pub prev_cam: Option<CamBasis>,
    pub q: Quality,
    /// Check builds: hemi claim re-validation + PSA accounting on the GPU.
    pub verify: bool,
}

/// Which upscaler the feed pass targets — selects the kernel (and thereby
/// the plane set and the u18 depth encoding).
#[derive(Clone, Copy, PartialEq)]
pub enum FeedKind {
    Xess,
    Rr,
}

/// 0 = off, 1 = AO, 2 = GI (GI subsumes AO, mirroring shade.rs's tiering).
fn fb_mode_of(q: &Quality) -> u32 {
    if q.fb.gi {
        2
    } else if q.fb.ao {
        1
    } else {
        0
    }
}

pub struct TraceGpu {
    pub root_sig: ID3D12RootSignature,
    pub cmd_sig: ID3D12CommandSignature,
    pso_reference: ID3D12PipelineState,
    pso_resolve: ID3D12PipelineState,
    pso_seed: ID3D12PipelineState,
    pso_prep: ID3D12PipelineState,
    pso_clear_info: ID3D12PipelineState,
    pso_level: ID3D12PipelineState,
    pso_sky: ID3D12PipelineState,
    pso_leaf: ID3D12PipelineState,
    pso_clear_h: ID3D12PipelineState,
    pso_prep_batch: ID3D12PipelineState,
    pso_seed_probes: ID3D12PipelineState,
    pso_hemi_root: ID3D12PipelineState,
    pso_hemi_cell: ID3D12PipelineState,
    pso_hemi_leaf: ID3D12PipelineState,
    pso_compose: ID3D12PipelineState,
    pso_feed_xess: Option<ID3D12PipelineState>,
    pso_feed_rr: Option<ID3D12PipelineState>,
    /// The wired upscaler feed targets (wire_feed): plane resources cloned
    /// for record_feed's barriers, plus which feed kernel consumes them.
    feed: Option<(FeedKind, Vec<ID3D12Resource>)>,
    pub scene: SceneGpu,
    /// Per-pixel planes, CPU-layout parity (accum = 3 f32/px, tbuf = f32/px,
    /// info = u32/px) so readback compares are direct memcmp-shaped.
    pub accum: ID3D12Resource,
    pub tbuf: ID3D12Resource,
    pub info: ID3D12Resource,
    /// Wavefront machinery: counters + indirect args + ping-pong tile queues
    /// + leaf/sky queues + the cut pool, all sized to the structural worst
    /// case (see caps) so the primary queues cannot overflow.
    pub counters: ID3D12Resource,
    args: ID3D12Resource,
    qa: ID3D12Resource,
    qb: ID3D12Resource,
    pub qleaf: ID3D12Resource,
    pub qsky: ID3D12Resource,
    cut_pool: ID3D12Resource,
    /// Compose planes + the hemisphere wavefront's buffers.
    partial: ID3D12Resource,
    ambw: ID3D12Resource,
    pub hbuf: ID3D12Resource,
    pub hemi_pts: ID3D12Resource,
    hq_a: ID3D12Resource,
    hq_b: ID3D12Resource,
    hq_leaf: ID3D12Resource,
    hemi_cut: ID3D12Resource,
    /// RGBA16F resolve target; rests in PIXEL_SHADER_RESOURCE between frames
    /// (the tonemap PS reads it via SRV_SLOT_GPU).
    pub hdr: ID3D12Resource,
    /// The G-buffer pack (GBufPx, 64 B/px) — full-size in upscaler sessions,
    /// a 64-byte dummy otherwise (`gbuf_full` gates FLAG_GBUF, which is what
    /// keeps the write helpers from scribbling past the dummy).
    pub gbuf: ID3D12Resource,
    gbuf_full: bool,
    uav_heap: ID3D12DescriptorHeap,
    frame_cb: d3d12::UploadBuffer,
    cb_base: FrameCb,
    pub rw: u32,
    pub rh: u32,
    /// Quadtree depth to the leaf frontier (levels recorded per frame).
    pub depth_full: u32,
    pub cap_leaf: u32,
    pub cap_sky: u32,
}

impl TraceGpu {
    pub fn new(
        device: &ID3D12Device,
        dxc: &Dxc,
        scene: &Scene,
        bvh: &Bvh,
        rw: u32,
        rh: u32,
        gbuf_full: bool,
        debug: bool,
    ) -> Result<Self> {
        require_caps(device)?;
        let root_sig = create_root_signature(device)?;
        let cmd_sig = create_dispatch_signature(device)?;

        let reference_src = [TRACE_COMMON_HLSLI, RT_HLSLI, SHADE_HLSLI, REFERENCE_HLSL].join("\n");
        let resolve_src = [TRACE_COMMON_HLSLI, RESOLVE_HLSL].join("\n");
        let wavefront_src =
            [TRACE_COMMON_HLSLI, CTR_HLSLI, QUEUES_HLSLI, FRUSTUM_HLSLI, WAVEFRONT_HLSL].join("\n");
        let leaf_src =
            [TRACE_COMMON_HLSLI, CTR_HLSLI, QUEUES_HLSLI, RT_HLSLI, SHADE_HLSLI, LEAF_HLSL]
                .join("\n");
        let hemi_wave_src =
            [TRACE_COMMON_HLSLI, CTR_HLSLI, HEMI_HLSLI, FRUSTUM_HLSLI, RT_HLSLI, HEMI_WAVE_HLSL]
                .join("\n");
        let hemi_leaf_src =
            [TRACE_COMMON_HLSLI, CTR_HLSLI, HEMI_HLSLI, RT_HLSLI, SHADE_HLSLI, HEMI_LEAF_HLSL]
                .join("\n");
        let compose_src = [TRACE_COMMON_HLSLI, CTR_HLSLI, QUEUES_HLSLI, COMPOSE_HLSL].join("\n");
        let feed_src = [TRACE_COMMON_HLSLI, FEED_HLSL].join("\n");
        let mut pso = |src: &str, entry: &str, what: &str| -> Result<ID3D12PipelineState> {
            compute_pso(device, &root_sig, &dxc.compile(src, entry, "cs_6_5", what, debug)?, what)
        };
        let pso_reference = pso(&reference_src, "cs_reference", "reference")?;
        let pso_resolve = pso(&resolve_src, "cs_resolve", "resolve")?;
        let pso_seed = pso(&wavefront_src, "cs_seed", "seed")?;
        let pso_prep = pso(&wavefront_src, "cs_prep", "prep")?;
        let pso_clear_info = pso(&wavefront_src, "cs_clear_info", "clear_info")?;
        let pso_level = pso(&wavefront_src, "cs_level", "level")?;
        let pso_sky = pso(&wavefront_src, "cs_sky", "sky")?;
        let pso_leaf = pso(&leaf_src, "cs_leaf", "leaf")?;
        let pso_clear_h = pso(&wavefront_src, "cs_clear_h", "clear_h")?;
        let pso_prep_batch = pso(&wavefront_src, "cs_prep_batch", "prep_batch")?;
        let pso_seed_probes = pso(&wavefront_src, "cs_seed_probes", "seed_probes")?;
        let pso_hemi_root = pso(&hemi_wave_src, "cs_hemi_root", "hemi_root")?;
        let pso_hemi_cell = pso(&hemi_wave_src, "cs_hemi_cell", "hemi_cell")?;
        let pso_hemi_leaf = pso(&hemi_leaf_src, "cs_hemi_leaf", "hemi_leaf")?;
        let pso_compose = pso(&compose_src, "cs_compose", "compose")?;
        // Feed kernels exist only when the pack is full-size (an upscaler
        // session); plain sessions never record a feed.
        let (pso_feed_xess, pso_feed_rr) = if gbuf_full {
            (
                Some(pso(&feed_src, "cs_feed_xess", "feed_xess")?),
                Some(pso(&feed_src, "cs_feed_rr", "feed_rr")?),
            )
        } else {
            (None, None)
        };

        let scene_gpu = SceneGpu::new(device, scene, bvh)?;

        let uaf = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let px = rw as u64 * rh as u64;
        let accum = committed_buffer(device, px * 12, uaf, ua)?;
        let tbuf = committed_buffer(device, px * 4, uaf, ua)?;
        let info = committed_buffer(device, px * 4, uaf, ua)?;

        // Structural worst-case queue sizing (see the plan/CLAUDE.md notes):
        // rects at depth d number at most 4^d; internal tiles live at depth
        // < D; every terminal (leaf or sky) tile contains at least one
        // depth-D path cell, so terminals number at most 4^D; split tiles
        // allocate one cut slot each, at most (4^D - 1) / 3.
        let dd = depth_full(rw, rh);
        if dd > 11 {
            // TraceGpu::new failures fall back to the CPU renderer with the
            // reason on stderr — a giant multi-monitor span must not abort.
            return Err(format!(
                "window {rw}x{rh} needs quadtree depth {dd} > 11 indirect-arg slots (max 16384 px)"
            ));
        }
        let cap_tile = if dd >= 1 { 1u64 << (2 * (dd - 1)) } else { 1 };
        let cap_leaf = 1u64 << (2 * dd);
        let cap_sky = cap_leaf;
        let cap_cut = ((1u64 << (2 * dd)) - 1) / 3 + 1;
        let counters = committed_buffer(device, CTR_COUNT as u64 * 4, uaf, ua)?;
        let args = committed_buffer(device, 16 * 12, uaf, ua)?;
        let qa = committed_buffer(device, cap_tile * 24, uaf, ua)?;
        let qb = committed_buffer(device, cap_tile * 24, uaf, ua)?;
        let qleaf = committed_buffer(device, cap_leaf * 16, uaf, ua)?;
        let qsky = committed_buffer(device, cap_sky * 16, uaf, ua)?;
        let cut_pool = committed_buffer(device, cap_cut * 256, uaf, ua)?;

        // Compose planes + hemisphere wavefront (batch-bounded transients:
        // a batch point has at most 4^(depth-1) cells at one level, and one
        // cut slot per split — 1 root + 4 + 16 interior at the deepest
        // preset).
        let partial = committed_buffer(device, px * 12, uaf, ua)?;
        let ambw = committed_buffer(device, px * 12, uaf, ua)?;
        let hbuf = committed_buffer(device, px * 16, uaf, ua)?;
        let hemi_pts = committed_buffer(device, px * 32, uaf, ua)?;
        let cap_hemi_cell = HEMI_BATCH as u64 * (1u64 << (2 * (HEMI_MAX_DEPTH - 1)));
        let cap_hemi_cut = HEMI_BATCH as u64 * (((1u64 << (2 * (HEMI_MAX_DEPTH - 1))) - 1) / 3 + 1);
        let hq_a = committed_buffer(device, cap_hemi_cell * 64, uaf, ua)?;
        let hq_b = committed_buffer(device, cap_hemi_cell * 64, uaf, ua)?;
        let hq_leaf = committed_buffer(device, cap_hemi_cell * 64, uaf, ua)?;
        let hemi_cut = committed_buffer(device, cap_hemi_cut * 256, uaf, ua)?;

        let hdr = d3d12::committed_tex(
            device,
            rw,
            rh,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            uaf,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        // The G-buffer pack: dlss::GBufs interleaved on the GPU. Full-size
        // only in upscaler sessions — plain sessions bind a 64-byte dummy
        // and never set FLAG_GBUF (root UAVs have no bounds check).
        let gbuf = committed_buffer(device, if gbuf_full { px * 64 } else { 64 }, uaf, ua)?;
        // Slot 0 = hdr (u14, resolve); slots 1..7 = the upscaler feed targets
        // (u16..u22), wired per session by wire_feed — null until then (RS 1.0
        // descriptors are volatile; only accessed slots must be valid).
        let uav_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1 + NUM_FEED,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("CreateDescriptorHeap(trace UAV): {e}"))?;
        unsafe {
            device.CreateUnorderedAccessView(
                &hdr,
                None,
                None,
                uav_heap.GetCPUDescriptorHandleForHeapStart(),
            )
        };

        let frame_cb =
            d3d12::UploadBuffer::new(device, CB_STRIDE * d3d12::FRAMES_IN_FLIGHT)?;

        // Debug names — what PIX / the debug layer show for barriers, UAV
        // hazards, and device-removed pages.
        let name = |res: &ID3D12Resource, n: &str| {
            let w: Vec<u16> = n.encode_utf16().chain(std::iter::once(0)).collect();
            let _ = unsafe { res.SetName(windows::core::PCWSTR(w.as_ptr())) };
        };
        for (res, n) in [
            (&accum, "trace.accum"),
            (&tbuf, "trace.tbuf"),
            (&info, "trace.info"),
            (&counters, "trace.counters"),
            (&args, "trace.args"),
            (&qa, "trace.tile_queue_a"),
            (&qb, "trace.tile_queue_b"),
            (&qleaf, "trace.leaf_queue"),
            (&qsky, "trace.sky_queue"),
            (&cut_pool, "trace.cut_pool"),
            (&partial, "trace.partial"),
            (&ambw, "trace.ambw"),
            (&hbuf, "trace.hemi_accum"),
            (&hemi_pts, "trace.hemi_points"),
            (&hq_a, "trace.hemi_queue_a"),
            (&hq_b, "trace.hemi_queue_b"),
            (&hq_leaf, "trace.hemi_leaf_queue"),
            (&hemi_cut, "trace.hemi_cut_pool"),
            (&hdr, "trace.hdr"),
            (&gbuf, "trace.gbuf_pack"),
        ] {
            name(res, n);
        }

        // Scene-static CB fields, prefilled once. near/far ride the prev
        // block's w slots (dlss::near_far — the G-buffer sky depth source).
        let sun = crate::render::sun_dir(scene);
        let (near, far) = crate::dlss::near_far(scene.diag);
        let v4 = |v: Vec3A, w: f32| [v.x, v.y, v.z, w];
        let cb_base = FrameCb {
            cam_origin: [0.0; 4],
            cam_forward: [0.0; 4],
            cam_right: [0.0; 4],
            cam_up: [0.0; 4],
            sun: v4(sun, 0.0),
            light_center: v4(scene.light.center, scene.eps),
            light_u: v4(scene.light.u, scene.ao_radius),
            light_v: v4(scene.light.v, 0.0),
            light_color: v4(scene.light.color, 0.0),
            rw,
            rh,
            frame: 0,
            flags: 0,
            shadow_samples: 0,
            ao_samples: 0,
            reflections: 0,
            _pad0: 0,
            frame_jitter: [0.0, 0.0],
            _pad1: 0.0,
            _pad2: 0.0,
            cap_tile: cap_tile as u32,
            cap_leaf: cap_leaf as u32,
            cap_sky: cap_sky as u32,
            cap_cut: cap_cut as u32,
            fb_mode: 0,
            fb_depth: 2,
            hemi_batch: HEMI_BATCH,
            cap_hemi_pt: rw * rh,
            cap_hemi_cell: cap_hemi_cell as u32,
            cap_hemi_leaf: cap_hemi_cell as u32,
            cap_hemi_cut: cap_hemi_cut as u32,
            _pad3: 0,
            prev_origin: [0.0; 4],
            prev_forward: [0.0; 4],
            prev_right: [0.0, 0.0, 0.0, near],
            prev_up: [0.0, 0.0, 0.0, far],
        };

        Ok(Self {
            root_sig,
            cmd_sig,
            pso_reference,
            pso_resolve,
            pso_seed,
            pso_prep,
            pso_clear_info,
            pso_level,
            pso_sky,
            pso_leaf,
            pso_clear_h,
            pso_prep_batch,
            pso_seed_probes,
            pso_hemi_root,
            pso_hemi_cell,
            pso_hemi_leaf,
            pso_compose,
            pso_feed_xess,
            pso_feed_rr,
            feed: None,
            scene: scene_gpu,
            accum,
            tbuf,
            info,
            counters,
            args,
            qa,
            qb,
            qleaf,
            qsky,
            cut_pool,
            partial,
            ambw,
            hbuf,
            hemi_pts,
            hq_a,
            hq_b,
            hq_leaf,
            hemi_cut,
            hdr,
            gbuf,
            gbuf_full,
            uav_heap,
            frame_cb,
            cb_base,
            rw,
            rh,
            depth_full: dd,
            cap_leaf: cap_leaf as u32,
            cap_sky: cap_sky as u32,
        })
    }

    /// Write this frame's constants into the given ring slot.
    pub fn write_cb(&self, slot: usize, p: &FrameParams) {
        let (origin, forward, right, up, inv_w, inv_h) = p.cam.gpu_fields();
        let mut cb = self.cb_base;
        cb.cam_origin = [origin.x, origin.y, origin.z, inv_w];
        cb.cam_forward = [forward.x, forward.y, forward.z, inv_h];
        cb.cam_right = [right.x, right.y, right.z, 0.0];
        cb.cam_up = [up.x, up.y, up.z, 0.0];
        cb.frame = p.frame;
        cb.flags = (p.accumulate as u32 * FLAG_ACCUM)
            | (p.jitter as u32 * FLAG_JITTER)
            | (p.frame_jitter.is_some() as u32 * FLAG_FRAME_JITTER)
            | (p.verify as u32 * FLAG_VERIFY)
            | (self.gbuf_full as u32 * FLAG_GBUF)
            | (p.prev_cam.is_some() as u32 * FLAG_HAS_PREV);
        cb.shadow_samples = p.q.shadow_samples;
        cb.ao_samples = p.q.ao_samples;
        cb.reflections = p.q.reflections as u32;
        cb.frame_jitter = match p.frame_jitter {
            Some((x, y)) => [x, y],
            None => [0.0, 0.0],
        };
        cb.fb_mode = fb_mode_of(&p.q);
        cb.fb_depth = p.q.fb.depth.clamp(1, HEMI_MAX_DEPTH);
        if let Some(pc) = &p.prev_cam {
            // The near/far riding the w slots of the last two rows come from
            // cb_base and must survive the overwrite.
            let (po, pf, pr, pu, piw, pih) = pc.gpu_fields();
            cb.prev_origin = [po.x, po.y, po.z, piw];
            cb.prev_forward = [pf.x, pf.y, pf.z, pih];
            cb.prev_right = [pr.x, pr.y, pr.z, cb.prev_right[3]];
            cb.prev_up = [pu.x, pu.y, pu.z, cb.prev_up[3]];
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                &cb as *const FrameCb as *const u8,
                self.frame_cb.ptr.add(slot * CB_STRIDE),
                std::mem::size_of::<FrameCb>(),
            )
        };
    }

    /// Bind the shared root signature + everything every kernel might read.
    unsafe fn bind_common(&self, list: &ID3D12GraphicsCommandList, slot: usize) {
        unsafe {
            list.SetComputeRootSignature(&self.root_sig);
            list.SetComputeRootConstantBufferView(
                RP_FRAME_CBV,
                self.frame_cb.resource.GetGPUVirtualAddress() + (slot * CB_STRIDE) as u64,
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_ACCUM,
                self.accum.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_TBUF,
                self.tbuf.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_INFO,
                self.info.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_COUNTERS,
                self.counters.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_ARGS,
                self.args.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_QLEAF,
                self.qleaf.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_QSKY,
                self.qsky.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_CUT,
                self.cut_pool.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_PARTIAL,
                self.partial.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_AMBW,
                self.ambw.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_HBUF,
                self.hbuf.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_HEMI_PTS,
                self.hemi_pts.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(RP_GBUF, self.gbuf.GetGPUVirtualAddress());
            let s = &self.scene;
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_BVH_NODES,
                s.bvh_nodes.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_TRI_IDX,
                s.tri_idx.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_POSITIONS,
                s.positions.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_NORMALS,
                s.normals.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_INDICES,
                s.indices.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_TRI_MAT,
                s.tri_mat.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_MATERIALS,
                s.materials.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_TLAS,
                s.tlas.GetGPUVirtualAddress(),
            );
        }
    }

    /// Record the vanilla full-screen reference trace (M2; also the on-GPU
    /// reference for the wavefront gates). Ends with a global UAV barrier.
    pub fn record_reference(&self, list: &ID3D12GraphicsCommandList, slot: usize) {
        let _ev = super::pix::scope(list, c"reference");
        unsafe {
            self.bind_common(list, slot);
            list.SetPipelineState(&self.pso_reference);
            list.Dispatch(self.rw.div_ceil(8), self.rh.div_ceil(8), 1);
            list.ResourceBarrier(&[uav_barrier(None)]);
        }
    }

    unsafe fn push(&self, list: &ID3D12GraphicsCommandList, v: [u32; 4]) {
        unsafe { list.SetComputeRoot32BitConstants(RP_PUSH, 4, v.as_ptr() as *const _, 0) };
    }

    unsafe fn args_to_indirect(&self, list: &ID3D12GraphicsCommandList) {
        unsafe {
            list.ResourceBarrier(&[
                uav_barrier(None),
                transition(
                    &self.args,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                ),
            ]);
        }
    }

    unsafe fn args_to_uav(&self, list: &ID3D12GraphicsCommandList) {
        unsafe {
            list.ResourceBarrier(&[
                uav_barrier(None),
                transition(
                    &self.args,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
            ]);
        }
    }

    /// Record one wavefront quadtree frame: seed -> depth_full x (prep-args
    /// -> ExecuteIndirect level) -> leaf + sky fills -> (hemi batches when
    /// fb is on) -> compose (the single accum splat). Statically recorded —
    /// the GPU makes every scheduling decision through the counters; empty
    /// levels and empty hemi batches dispatch zero groups. `clear_sentinel`
    /// floods `info` with the exactly-once coverage sentinel (check builds).
    pub fn record_wavefront(
        &self,
        list: &ID3D12GraphicsCommandList,
        slot: usize,
        p: &FrameParams,
        clear_sentinel: bool,
    ) {
        let fb_mode = fb_mode_of(&p.q);
        let _ev = super::pix::scope(list, c"wavefront");
        unsafe {
            self.bind_common(list, slot);
            // Seed sees level 0's queue arrangement (qin = A).
            list.SetComputeRootUnorderedAccessView(RP_UAV0 + UAV_QIN, self.qa.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(RP_UAV0 + UAV_QOUT, self.qb.GetGPUVirtualAddress());
            list.SetPipelineState(&self.pso_seed);
            list.Dispatch(1, 1, 1);
            if clear_sentinel {
                let px = self.rw * self.rh;
                let groups = px.div_ceil(256);
                list.SetPipelineState(&self.pso_clear_info);
                list.Dispatch(groups.min(32768), groups.div_ceil(32768), 1);
            }
            if fb_mode > 0 {
                let groups = (self.rw * self.rh * 4).div_ceil(256);
                list.SetPipelineState(&self.pso_clear_h);
                list.Dispatch(groups.min(32768), groups.div_ceil(32768), 1);
            }
            list.ResourceBarrier(&[uav_barrier(None)]);

            for d in 0..self.depth_full {
                let _ev = super::pix::scope_fmt(list, format_args!("level {d}"));
                let (in_ctr, out_ctr) = if d % 2 == 0 {
                    (CTR_TILE_A, CTR_TILE_B)
                } else {
                    (CTR_TILE_B, CTR_TILE_A)
                };
                let (qin, qout) =
                    if d % 2 == 0 { (&self.qa, &self.qb) } else { (&self.qb, &self.qa) };
                // prep: this level's count -> indirect args; zero the OUT
                // counter the level kernel is about to append into.
                list.SetPipelineState(&self.pso_prep);
                self.push(list, [in_ctr, out_ctr, 32, d]);
                list.Dispatch(1, 1, 1);
                self.args_to_indirect(list);
                list.SetComputeRootUnorderedAccessView(
                    RP_UAV0 + UAV_QIN,
                    qin.GetGPUVirtualAddress(),
                );
                list.SetComputeRootUnorderedAccessView(
                    RP_UAV0 + UAV_QOUT,
                    qout.GetGPUVirtualAddress(),
                );
                list.SetPipelineState(&self.pso_level);
                self.push(list, [in_ctr, out_ctr, 0, 0]);
                list.ExecuteIndirect(&self.cmd_sig, 1, &self.args, d as u64 * 12, None, 0);
                self.args_to_uav(list);
            }

            // Leaf + sky fills (disjoint pixels — no barrier between them).
            {
                let _ev = super::pix::scope(list, c"leaf+sky");
                list.SetPipelineState(&self.pso_prep);
                self.push(list, [CTR_LEAF, NO_RESET, 1, ARG_LEAF]);
                list.Dispatch(1, 1, 1);
                self.push(list, [CTR_SKY, NO_RESET, 1, ARG_SKY]);
                list.Dispatch(1, 1, 1);
                self.args_to_indirect(list);
                list.SetPipelineState(&self.pso_leaf);
                self.push(list, [CTR_LEAF, 0, 0, 0]);
                list.ExecuteIndirect(&self.cmd_sig, 1, &self.args, ARG_LEAF as u64 * 12, None, 0);
                list.SetPipelineState(&self.pso_sky);
                self.push(list, [CTR_SKY, 0, 0, 0]);
                list.ExecuteIndirect(&self.cmd_sig, 1, &self.args, ARG_SKY as u64 * 12, None, 0);
                self.args_to_uav(list);
            }

            if fb_mode > 0 {
                // Every hit pixel appended a shading point; batch over the
                // worst case (all of them).
                self.record_hemi(list, self.rw * self.rh, p.q.fb.depth);
            }
            self.record_compose(list);
        }
    }

    /// The hemisphere wavefront over the points in the hemi queue, in
    /// HEMI_BATCH slices (each batch resets the transient cell queues + cut
    /// pool — that reset is what bounds the memory). `max_points` sizes the
    /// statically recorded batch count; batches past the GPU-side count
    /// dispatch zero groups. Caller must have bind_common'd already.
    fn record_hemi(&self, list: &ID3D12GraphicsCommandList, max_points: u32, fb_depth: u32) {
        let _ev = super::pix::scope(list, c"hemi");
        let n_batches = max_points.div_ceil(HEMI_BATCH);
        let levels = fb_depth.clamp(2, HEMI_MAX_DEPTH) - 1;
        unsafe {
            // Hemi buffer arrangement: u7 = hemi leaf queue, u9 = hemi cut
            // pool (the primary qleaf/cut_pool are done for this frame).
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_QLEAF,
                self.hq_leaf.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_CUT,
                self.hemi_cut.GetGPUVirtualAddress(),
            );
            for b in 0..n_batches {
                let base = b * HEMI_BATCH;
                // Batch prep: root args + reset the batch-scoped counters.
                list.SetPipelineState(&self.pso_prep_batch);
                self.push(list, [CTR_HEMI_PT, base, 32, ARG_HEMI_ROOT]);
                list.Dispatch(1, 1, 1);
                self.args_to_indirect(list);
                list.SetComputeRootUnorderedAccessView(
                    RP_UAV0 + UAV_QOUT,
                    self.hq_a.GetGPUVirtualAddress(),
                );
                list.SetPipelineState(&self.pso_hemi_root);
                self.push(list, [base, CTR_HEMI_A, 0, 0]);
                list.ExecuteIndirect(&self.cmd_sig, 1, &self.args, ARG_HEMI_ROOT as u64 * 12, None, 0);
                self.args_to_uav(list);

                for l in 0..levels {
                    let (in_ctr, out_ctr) = if l % 2 == 0 {
                        (CTR_HEMI_A, CTR_HEMI_B)
                    } else {
                        (CTR_HEMI_B, CTR_HEMI_A)
                    };
                    let (qin, qout) =
                        if l % 2 == 0 { (&self.hq_a, &self.hq_b) } else { (&self.hq_b, &self.hq_a) };
                    list.SetPipelineState(&self.pso_prep);
                    self.push(list, [in_ctr, out_ctr, 32, ARG_HEMI_CELL]);
                    list.Dispatch(1, 1, 1);
                    self.args_to_indirect(list);
                    list.SetComputeRootUnorderedAccessView(
                        RP_UAV0 + UAV_QIN,
                        qin.GetGPUVirtualAddress(),
                    );
                    list.SetComputeRootUnorderedAccessView(
                        RP_UAV0 + UAV_QOUT,
                        qout.GetGPUVirtualAddress(),
                    );
                    list.SetPipelineState(&self.pso_hemi_cell);
                    self.push(list, [in_ctr, out_ctr, 0, 0]);
                    list.ExecuteIndirect(&self.cmd_sig, 1, &self.args, ARG_HEMI_CELL as u64 * 12, None, 0);
                    self.args_to_uav(list);
                }

                // Leaf rays: 4 threads per leaf cell (numthreads 32 => 8
                // records per group).
                list.SetPipelineState(&self.pso_prep);
                self.push(list, [CTR_HEMI_LEAF, NO_RESET, 8, ARG_HEMI_LEAF]);
                list.Dispatch(1, 1, 1);
                self.args_to_indirect(list);
                list.SetPipelineState(&self.pso_hemi_leaf);
                self.push(list, [0, 0, 0, 0]);
                list.ExecuteIndirect(&self.cmd_sig, 1, &self.args, ARG_HEMI_LEAF as u64 * 12, None, 0);
                self.args_to_uav(list);
            }
        }
    }

    /// partial + ambW * ambient(H) -> accum (store-or-add): the single splat.
    fn record_compose(&self, list: &ID3D12GraphicsCommandList) {
        let _ev = super::pix::scope(list, c"compose");
        unsafe {
            let groups = (self.rw * self.rh).div_ceil(256);
            list.SetPipelineState(&self.pso_compose);
            list.Dispatch(groups.min(32768), groups.div_ceil(32768), 1);
            list.ResourceBarrier(&[uav_barrier(None)]);
        }
    }

    /// One frame, hybrid (wavefront quadtree) or the vanilla reference —
    /// the R-key A/B. (The reference kernel has no hemi tiers: with fb on
    /// it renders the sampled-ambient path.)
    pub fn record_frame(&self, list: &ID3D12GraphicsCommandList, slot: usize, p: &FrameParams, hybrid: bool) {
        if hybrid {
            self.record_wavefront(list, slot, p, false);
        } else {
            self.record_reference(list, slot);
        }
    }

    /// --check-gpu probe path: upload CPU-generated shading points, run ONLY
    /// the hemisphere passes over them (fb settings from the CB written by
    /// `write_cb` — the CB `frame` seeds the Arvo draws, so calling again
    /// with `clear = false` and a different frame ACCUMULATES another
    /// independent estimate into H, mirroring the CPU suite's multi-seed
    /// A/B; the verify/stat counters accumulate the same way, so the
    /// exact-zero gates cover every seed). Probe i's results land at
    /// hbuf[i]; `pixel` is the probe index.
    pub fn run_hemi_probes(
        &self,
        hg: &mut HeadlessGpu,
        slot: usize,
        probes: &[(Vec3A, Vec3A)],
        fb_depth: u32,
        clear: bool,
    ) -> Result<()> {
        assert!(probes.len() <= (self.rw * self.rh) as usize);
        let mut bytes = Vec::with_capacity(probes.len() * 32);
        for (i, (o, n)) in probes.iter().enumerate() {
            for v in [o.x, o.y, o.z] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes.extend_from_slice(&(i as u32).to_le_bytes());
            for v in [n.x, n.y, n.z] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }
        let staging = d3d12::UploadBuffer::new(&hg.device, bytes.len())?;
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), staging.ptr, bytes.len()) };
        let n = probes.len() as u32;
        hg.run(|list| unsafe {
            let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            list.ResourceBarrier(&[transition(&self.hemi_pts, ua, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyBufferRegion(&self.hemi_pts, 0, &staging.resource, 0, bytes.len() as u64);
            list.ResourceBarrier(&[transition(&self.hemi_pts, D3D12_RESOURCE_STATE_COPY_DEST, ua)]);

            self.bind_common(list, slot);
            list.SetPipelineState(&self.pso_seed_probes);
            // push1: full counter clear on the first seed only — accumulate
            // passes keep the verify/stat counters so the exact-zero gates
            // observe every seed's rays, not just the last seed's.
            self.push(list, [n, clear as u32, 0, 0]);
            list.Dispatch(1, 1, 1);
            if clear {
                let groups = (self.rw * self.rh * 4).div_ceil(256);
                list.SetPipelineState(&self.pso_clear_h);
                list.Dispatch(groups.min(32768), groups.div_ceil(32768), 1);
            }
            list.ResourceBarrier(&[uav_barrier(None)]);
            self.record_hemi(list, n, fb_depth);
        })
    }

    /// Wire the upscaler feed targets into the descriptor heap (slots 1..7 =
    /// registers u16..u22) and remember them for record_feed's barriers.
    /// `targets` = (shader register, plane, format). Gated on typed-UAV-store
    /// support per format (optional in D3D12) — an Err here means the caller
    /// falls back to plain presentation, loudly.
    pub fn wire_feed(
        &mut self,
        device: &ID3D12Device,
        kind: FeedKind,
        targets: &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
    ) -> Result<()> {
        for &(reg, _, format) in targets {
            let mut fs = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: format, ..Default::default() };
            unsafe {
                device.CheckFeatureSupport(
                    D3D12_FEATURE_FORMAT_SUPPORT,
                    &mut fs as *mut _ as *mut _,
                    std::mem::size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>() as u32,
                )
            }
            .map_err(|e| format!("CheckFeatureSupport(format {}): {e}", format.0))?;
            if fs.Support2.0 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE.0 == 0 {
                return Err(format!(
                    "feed target u{reg}: format {} lacks typed UAV store on this device",
                    format.0
                ));
            }
        }
        let inc = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        } as usize;
        let base = unsafe { self.uav_heap.GetCPUDescriptorHandleForHeapStart() };
        let mut planes = Vec::with_capacity(targets.len());
        for &(reg, res, format) in targets {
            // A register outside the feed range would silently overwrite the
            // hdr descriptor (slot 0) or write past the heap end — descriptor
            // writes have no bounds check of their own, so gate in release.
            if !(NUM_UAVS + 2..NUM_UAVS + 2 + NUM_FEED).contains(&reg) {
                return Err(format!("feed target u{reg} outside u16..u22"));
            }
            let slot = (reg - (NUM_UAVS + 1)) as usize; // u16 -> heap slot 1
            let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            unsafe {
                device.CreateUnorderedAccessView(
                    res,
                    None,
                    Some(&desc),
                    D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + slot * inc },
                )
            };
            planes.push(res.clone());
        }
        self.feed = Some((kind, planes));
        Ok(())
    }

    /// Fan the pack + accum out into the wired upscaler input planes — the
    /// GPU-resident replacement for rr/xr::record_upload. Record AFTER
    /// record_frame on the same list (its trailing global UAV barrier fences
    /// the pack/accum writes). The planes transition NPSR -> UAV -> NPSR; the
    /// back-transition is both the write->read sync and what keeps the
    /// upscalers' state-at-use contracts truthful (RR's tags and XeSS's
    /// bindings both declare NON_PIXEL_SHADER_RESOURCE).
    pub fn record_feed(&self, list: &ID3D12GraphicsCommandList, slot: usize) -> Result<()> {
        let Some((kind, planes)) = &self.feed else {
            return Err("feed targets not wired".into());
        };
        let pso = match kind {
            FeedKind::Xess => self.pso_feed_xess.as_ref(),
            FeedKind::Rr => self.pso_feed_rr.as_ref(),
        }
        .ok_or("feed PSO missing (TraceGpu built without gbuf)")?;
        let _ev = super::pix::scope(list, c"feed");
        unsafe {
            let npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            let to_uav: Vec<_> = planes.iter().map(|p| transition(p, npsr, ua)).collect();
            list.ResourceBarrier(&to_uav);
            self.bind_common(list, slot);
            list.SetDescriptorHeaps(&[Some(self.uav_heap.clone())]);
            list.SetComputeRootDescriptorTable(
                RP_TEX,
                self.uav_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            list.SetPipelineState(pso);
            list.Dispatch(self.rw.div_ceil(8), self.rh.div_ceil(8), 1);
            let back: Vec<_> = planes.iter().map(|p| transition(p, ua, npsr)).collect();
            list.ResourceBarrier(&back);
        }
        Ok(())
    }

    /// accum -> HDR texture at 1/samples; leaves hdr in PIXEL_SHADER_RESOURCE
    /// for the tonemap PS.
    pub fn record_resolve(&self, list: &ID3D12GraphicsCommandList, slot: usize, samples: u32) {
        let _ev = super::pix::scope(list, c"resolve");
        unsafe {
            list.ResourceBarrier(&[transition(
                &self.hdr,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            self.bind_common(list, slot);
            list.SetDescriptorHeaps(&[Some(self.uav_heap.clone())]);
            list.SetComputeRootDescriptorTable(
                RP_TEX,
                self.uav_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            let push = [1.0f32 / samples.max(1) as f32, 0.0, 0.0, 0.0];
            list.SetComputeRoot32BitConstants(RP_PUSH, 4, push.as_ptr() as *const _, 0);
            list.SetPipelineState(&self.pso_resolve);
            list.Dispatch(self.rw.div_ceil(8), self.rh.div_ceil(8), 1);
            list.ResourceBarrier(&[transition(
                &self.hdr,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
    }
}
