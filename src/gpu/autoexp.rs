//! The GPU half of auto-exposure's luminance meter (`src/autoexp.rs` owns the
//! controller and the rationale; `shaders/autoexp.hlsl` is the kernel).
//!
//! A two-level mean-log2-luminance reduction over whatever texture the tonemap
//! is about to read, recorded from `fullscreen_to_backbuffer` beside the glare
//! pyramid — so every GPU present arm meters through this ONE site, on the
//! pre-glare, pre-exposure linear source (open-loop: exposure lives only in
//! the presentation curve downstream, so there is no servo feedback).
//!
//! Compiled with D3DCompile (fxc, cs_5_0) like bloom — no DXC dependency, so
//! CPU-renderer + plain-presentation sessions meter too.
//!
//! The readback is the `gputime.rs` frame-ring discipline, never a stall: the
//! result is copied into this frame's slot window of one persistent readback
//! buffer, and collected when the ring returns to that slot — `D3d::begin_frame`
//! has already fence-waited it, so the map reads completed bytes at exactly
//! FRAMES_IN_FLIGHT frames of latency (invisible inside the controller's ~1 s
//! EMA). An aborted frame leaves its slot's PREVIOUS value to be re-collected
//! once — one duplicate stale sample into the EMA, accepted (the alternative is
//! plumbing cleanup through every abort_frame arm).

use std::cell::Cell;

use super::d3d12::{self, ReadbackBuffer, Result, FRAMES_IN_FLIGHT};
use super::tonemap::{compile, SRV_HEAP_CAPACITY};
use windows::core::s;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

const AUTOEXP_HLSL: &str = include_str!("shaders/autoexp.hlsl");

/// One `cs_reduce` group covers a TILE x TILE pixel block.
const TILE: u32 = 64;
/// Tile-record capacity, sized once for 8K (7680x4320 -> 120x68 = 8160): 33 KB
/// buys never resizing, and covers XeSS super-resolution render targets past
/// the window size. A source past 8K skips metering (see `record`).
const MAX_TILES: u32 = (7680 / TILE) * (4320 / TILE);

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    src_w: u32,
    src_h: u32,
    inv_samples: f32,
    tiles_x: u32,
    ntiles: u32,
    _pad: [u32; 3],
}

pub struct AutoExpGpu {
    root_sig: ID3D12RootSignature,
    reduce_pso: ID3D12PipelineState,
    fold_pso: ID3D12PipelineState,
    /// Shader-visible heap: ONE permanent source-SRV slot per tonemap SRV slot
    /// (the bloom `create_source_srv` discipline — `CopyDescriptors` may not
    /// read a shader-visible heap, and rewriting a descriptor an in-flight
    /// frame references is UB). The UAVs are root descriptors and need none.
    heap: ID3D12DescriptorHeap,
    inc: u32,
    /// `MAX_TILES` floats — cs_reduce's per-tile partial sums.
    tiles: ID3D12Resource,
    /// One float — the folded mean.
    result: ID3D12Resource,
    /// `4 * FRAMES_IN_FLIGHT` bytes, one slot window per in-flight frame.
    readback: ReadbackBuffer,
    /// Which slot windows hold an uncollected copy (Cell: the present recorder
    /// takes `&self`).
    pending: [Cell<bool>; FRAMES_IN_FLIGHT],
}

impl AutoExpGpu {
    pub fn new(device: &ID3D12Device) -> Result<AutoExpGpu> {
        // Root signature: [0] SRV table (t0), [1] root UAV u0 (tiles),
        // [2] root UAV u1 (result), [3] 8 root constants. Buffer UAVs ride
        // root descriptors (no bounds check needed — the dispatch grid is
        // derived from the same dims the kernel guards on), so only the
        // texture SRV needs a heap.
        let srv_range = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        }];
        let root_uav = |reg: u32| D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: reg, RegisterSpace: 0 },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
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
            root_uav(0),
            root_uav(1),
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
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
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
        .map_err(|e| format!("autoexp: D3D12SerializeRootSignature: {e}"))?;
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
        .map_err(|e| format!("autoexp: CreateRootSignature: {e}"))?;

        let cs_reduce = compile(AUTOEXP_HLSL, s!("cs_reduce"), s!("cs_5_0"), "autoexp cs_reduce")?;
        let cs_fold = compile(AUTOEXP_HLSL, s!("cs_fold"), s!("cs_5_0"), "autoexp cs_fold")?;
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
                .map_err(|e| format!("autoexp: CreateComputePipelineState({what}): {e}"))
        };
        let reduce_pso = pso(&cs_reduce, "reduce")?;
        let fold_pso = pso(&cs_fold, "fold")?;

        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: SRV_HEAP_CAPACITY,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("autoexp: CreateDescriptorHeap: {e}"))?;
        let inc = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };

        let tiles = d3d12::committed_buffer(
            device,
            MAX_TILES as u64 * 4,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        let result = d3d12::committed_buffer(
            device,
            4,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        let readback = ReadbackBuffer::new(device, 4 * FRAMES_IN_FLIGHT)?;

        Ok(AutoExpGpu {
            root_sig,
            reduce_pso,
            fold_pso,
            heap,
            inc,
            tiles,
            result,
            readback,
            pending: std::array::from_fn(|_| Cell::new(false)),
        })
    }

    /// Register a tonemap source (the bloom twin — `wire_tonemap_src` pairs the
    /// three heaps so a source can never silently lose its meter).
    pub fn create_source_srv(
        &self,
        device: &ID3D12Device,
        res: &ID3D12Resource,
        format: DXGI_FORMAT,
        slot: u32,
    ) {
        let base = unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() };
        let h = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + (slot * self.inc) as usize };
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

    /// Record the reduction + the copy into `slot`'s readback window. The
    /// source must be in NON_PIXEL_SHADER_RESOURCE (the caller's bloom-style
    /// PSR<->NPSR bracket). A source too large for the tile buffer (past 8K)
    /// skips silently — no meter beats a wrong one, and the controller simply
    /// holds.
    pub fn record(
        &self,
        list: &ID3D12GraphicsCommandList,
        srv_slot: u32,
        src_w: u32,
        src_h: u32,
        inv_samples: f32,
        slot: usize,
    ) {
        let tx = src_w.div_ceil(TILE);
        let ty = src_h.div_ceil(TILE);
        let ntiles = tx * ty;
        debug_assert!(ntiles <= MAX_TILES, "autoexp: {src_w}x{src_h} exceeds the 8K tile budget");
        if ntiles == 0 || ntiles > MAX_TILES {
            return;
        }
        let p = Params {
            src_w,
            src_h,
            inv_samples,
            tiles_x: tx,
            ntiles,
            _pad: [0; 3],
        };
        let gpu_base = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        let src_desc =
            D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu_base.ptr + (srv_slot * self.inc) as u64 };
        unsafe {
            list.SetComputeRootSignature(&self.root_sig);
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootDescriptorTable(0, src_desc);
            list.SetComputeRootUnorderedAccessView(1, self.tiles.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, self.result.GetGPUVirtualAddress());
            list.SetComputeRoot32BitConstants(3, 8, &p as *const _ as *const _, 0);
            list.SetPipelineState(&self.reduce_pso);
            list.Dispatch(tx, ty, 1);
        }
        // cs_fold reads what cs_reduce wrote.
        let uav_barrier = |res: &ID3D12Resource| {
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
        };
        uav_barrier(&self.tiles);
        unsafe {
            list.SetPipelineState(&self.fold_pso);
            list.Dispatch(1, 1, 1);
        }
        uav_barrier(&self.result);
        unsafe {
            list.ResourceBarrier(&[d3d12::transition(
                &self.result,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyBufferRegion(&self.readback.resource, slot as u64 * 4, &self.result, 0, 4);
            list.ResourceBarrier(&[d3d12::transition(
                &self.result,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
        self.pending[slot].set(true);
    }

    /// Collect `slot`'s value — call BEFORE recording into the same slot, at a
    /// point where `D3d::begin_frame`'s fence wait has retired that slot's
    /// frame (the gputime contract; `fullscreen_to_backbuffer` qualifies).
    /// Maps the 4-byte sub-range and unmaps with an empty written range.
    pub fn collect(&self, slot: usize) -> Option<f32> {
        if !self.pending[slot].replace(false) {
            return None;
        }
        let begin = slot * 4;
        let range = D3D12_RANGE { Begin: begin, End: begin + 4 };
        let mut ptr = std::ptr::null_mut();
        unsafe { self.readback.resource.Map(0, Some(&range), Some(&mut ptr)) }.ok()?;
        let v = unsafe { std::ptr::read_unaligned((ptr as *const u8).add(begin) as *const f32) };
        unsafe {
            self.readback.resource.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        }
        v.is_finite().then_some(v)
    }
}
