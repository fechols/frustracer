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
//!   param RP_SCENE_TEX      table          t0..t6 space1 + Texture2D[] at
//!                                          t7.. space1 (scene textures; the
//!                                          only other exception — texture
//!                                          SRVs can't be root descriptors)

use super::adapter;
use super::d3d12::{self, committed_buffer, transition, uav_barrier, Result};
use super::dxc::Dxc;
// The kernel assembly — the HLSL corpus, the `#define` generators, and the
// tuning knobs those defines carry — lives in `gfx::shaders`, because it is
// pure string work that decides what the shaders MEAN and both backends must
// hand their compilers byte-identical sources. Re-exported here rather than
// merely imported: `dxr.rs`, `mod.rs`, `dual.rs` and main.rs's gates all spell
// these `trace::…`, and this module remains the place a reader looks for
// anything the wavefront pipeline knows. See that module's header for what
// stayed behind and why.
pub use crate::gfx::shaders::*;
use crate::bc7;
use crate::blas_split::ChunkWindow;
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
/// Upscaler feed-target texture UAVs: registers u16..u27, riding the u14
/// descriptor table as a second range at heap slots 1..12 (0 DWORDs extra).
/// The register/type layout is shared by every feed kernel (feed.hlsl); a
/// register's VALUE/format may differ per kernel, the HLSL type never does.
pub const NUM_FEED: u32 = 12;
/// One descriptor SET is the whole RP_TEX table: the u14 resolve target at
/// offset 0, then the NUM_FEED feed registers.
pub const FEED_SET_STRIDE: u32 = 1 + NUM_FEED;
/// How many feed sets the heap holds. Normal sessions wire ONE (the session's
/// one upscaler). `--quinlight` wires several engines at once, and their plane
/// sets OVERLAP in registers (RR and FSR4-RR both claim u16/u17/u18/u20..u22) —
/// so each engine gets its OWN set of descriptors and each feed dispatch binds
/// the table at its set's offset. Rewriting one set of descriptors between two
/// dispatches recorded into the SAME command list would be a bug: descriptors
/// are read at execute time, so the last write would win for both.
///
/// 4 = the 3 engine sets + the NRD bridge set. The engine ceiling is still 3
/// (DLSS-RR, FSR4-RR, and the XeSS/FSR3 trio — XeSS and FSR 3.1 take a
/// byte-identical plane set, so they share one feed, see
/// `ffx_up::upscale_res_shared`); set `NRD_FEED_SET` (= 3, the LAST set) holds
/// the `--nrd` bridge's plane descriptors (nrd_bridge.hlsl's IN_*/OUT_* over
/// the shared u16..u27 registers), wired via `wire_nrd_feed`. The bump is heap
/// arithmetic only — TEX_HEAP_BASE derives, the root signature is untouched.
pub const FEED_SETS: u32 = 4;
/// The NRD bridge's descriptor set (the last one — engine sets stay 0..2).
pub const NRD_FEED_SET: u32 = 3;
pub const FEED_COLOR: u32 = 16; // RGBA16F (RR/XeSS); RGBA16F residual (FSR-RR)
pub const FEED_NR: u32 = 17; // RGBA16F normal+rough (RR); RGB10A2 oct-normals (FSR-RR)
pub const FEED_DEPTH: u32 = 18; // R32F (all; encoding differs per kernel)
pub const FEED_MVEC: u32 = 19; // RG16F (RR/XeSS)
pub const FEED_ALB: u32 = 20; // RGBA8 diffuse albedo (RR; sqrt-encoded FSR-RR)
pub const FEED_SPEC: u32 = 21; // RGBA8 specular albedo (RR; sqrt-encoded FSR-RR)
pub const FEED_SPECHIT: u32 = 22; // R16F spec hit distance (RR); R32F linear depth (FSR-RR)
pub const FEED_FSR_MVEC: u32 = 23; // RGBA16F FSR-RR mvec (UV-delta RG + depth-delta B)
pub const FEED_FSR_DD: u32 = 24; // RGBA16F FSR-RR demodulated direct diffuse
pub const FEED_FSR_DS: u32 = 25; // RGBA16F FSR-RR demodulated direct specular
pub const FEED_FSR_AO: u32 = 26; // R16F FSR-RR ambient-occlusion open fraction
pub const FEED_FSR_IS: u32 = 27; // RGBA16F FSR-RR indirect specular (A = hit t)
// GPU-resident NPPD staging (the --gpu --nppd composition): four raw-buffer
// root UAVs at u28..u31 (nppd.hlsl's literals — lockstep), appended after
// RP_GBUF so nothing renumbers (61/64 root-signature DWORDs). Bound only
// when the session built them.
pub const RP_NPPD_FRAME: u32 = RP_GBUF + 1;
pub const RP_NPPD_STATE: u32 = RP_GBUF + 2;
pub const RP_NPPD_WARPED: u32 = RP_GBUF + 3;
pub const RP_NPPD_OUT: u32 = RP_GBUF + 4;
pub const NPPD_REG_BASE: u32 = NUM_UAVS + 2 + NUM_FEED; // u28 (feed ends at u27)
/// Scene textures + UV stream: one SRV descriptor table in register space1
/// (collision-free with every space0 register above), appended after the
/// NPPD params so nothing renumbers (62/64 root-signature DWORDs). Range 0 =
/// t0..t6 space1 (texcoords, indices alias, tri_mat alias, mat_cutout,
/// positions alias, mat_height, mat_shadow);
/// range 1 = t7.. space1, UNBOUNDED — the per-scene Texture2D array (must be
/// the table's last range). Descriptors live in the same shader-visible heap
/// as the u14/feed slots (only one CBV_SRV_UAV heap is bindable at a time),
/// starting at slot TEX_HEAP_BASE.
pub const RP_SCENE_TEX: u32 = RP_NPPD_OUT + 1;
/// The G-buffer pack's guide/signal half (register u32), appended LAST so no
/// established param index renumbers. See `GBufExt` in trace_common.hlsli for
/// why the pack is two buffers rather than one struct with skipped members —
/// short version: a partial member store of the old 88-byte record measured
/// SLOWER than storing all of it (unaligned scatter / write-allocate), while
/// a separate 16 B/px core buffer stores contiguous and coalesced.
///
/// **THIS CLOSES THE ROOT SIGNATURE AT 64/64 DWORDs.** It was 62. A root UAV
/// costs 2, and there is now no slack: any future root parameter must displace
/// one or move into a descriptor table. That was a deliberate trade — several
/// designs here reused registers specifically to avoid spending it (the cloud
/// caches on u5/u6, the feed descriptor SETS) — bought here because `gbuf` is
/// a root UAV with no bounds checking, so typed structs turn an offset mistake
/// into a compile error instead of silent memory corruption.
pub const RP_GBUF_EXT: u32 = RP_SCENE_TEX + 1;
/// First heap slot of the scene-texture table (after every feed set).
pub const TEX_HEAP_BASE: u32 = FEED_SETS * FEED_SET_STRIDE;
/// Buffer-SRV descriptors preceding the Texture2D array in the table
/// (t0..t9 space1: texcoords, indices alias, tri_mat alias, mat_cutout,
/// positions alias, mat_height, mat_shadow, blas_tri, chunk_base, sway_dmv —
/// texs[] starts at t10). Must stay in lockstep with trace_common.hlsli's
/// space1 declarations: this const IS the texs[] base register.
pub const TEX_TABLE_BUFS: u32 = 10;

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

// Root-CBV alignment (256 B). FrameCb is 4576 bytes — 288 of struct plus the
// MAX_SPP-entry jitter table (--spp), the SH sky rows, the MAX_FIREFLIES
// pose rows, and the MAX_EMISSIVE_LIGHTS cluster-light row pairs, which are
// what set the size (raise the stride in lockstep with any cap; the const
// asserts below police both directions).
pub(crate) const CB_STRIDE: usize = 4608;



/// What the GPU tracer requires, queried once. RayQuery in compute needs
/// RaytracingTier 1.1 AND shader model 6.5; missing either is a clean
/// "use the CPU path" story, never a degraded half-mode.
///
/// The wave/work-graph fields below are REPORTING ONLY — nothing in
/// `require_caps` gates on them. `WaveOps` is true on every device that can
/// reach this code (it has been core since SM 6.0, and we require 6.5), so
/// making it a requirement would add a fallback path that can never run;
/// `--check-gpu` asserts it instead. `work_graphs_tier` exists so the
/// `FR_WORKGRAPH` spike can refuse to arm on a runtime that lacks the feature
/// rather than failing at state-object creation.
pub struct Caps {
    pub rt_tier: i32,
    pub shader_model: i32,
    pub binding_tier: i32,
    /// D3D12_OPTIONS1. The lane count is a RANGE, not a width: the driver
    /// picks per shader inside it, so this bounds what `WaveGetLaneCount()`
    /// can return and never predicts it. Measured spread that matters here:
    /// Arc A-series reports [8, 32], Arc B-series (Xe2 dropped SIMD8) [16, 32],
    /// NVIDIA [32, 32], AMD RDNA [32, 64].
    pub wave_ops: bool,
    pub wave_lane_min: u32,
    pub wave_lane_max: u32,
    pub total_lanes: u32,
    /// D3D12_OPTIONS21, 0 when the runtime predates the struct (the query
    /// itself errors there, which is not a failure — see `query_caps`).
    pub work_graphs_tier: i32,
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
    // clamps DOWN to what it supports (an old runtime ERRORS on an enum value
    // it does not know, rather than clamping, so walk the seed down).
    //
    // THE SEED IS A CEILING, NOT A QUESTION — this reported 6.7 on an Arc Pro
    // B70 purely because 6.7 was what we asked about. That is fine while the
    // only consumers are `>= 6.5` / `>= 6.3` gates, and actively wrong once
    // anything wants SM 6.8 (work graphs compile at `lib_6_8`), so the ladder
    // starts above what we currently use. Keep 6.5 as the last rung: it is the
    // wavefront tracer's own floor, so a runtime that rejects even that is a
    // clean CPU-fallback story.
    let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL::default();
    let mut sm_err = None;
    for seed in [
        D3D_SHADER_MODEL_6_9,
        D3D_SHADER_MODEL_6_8,
        D3D_SHADER_MODEL_6_7,
        D3D_SHADER_MODEL_6_5,
    ] {
        sm.HighestShaderModel = seed;
        match unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                &mut sm as *mut _ as *mut _,
                std::mem::size_of::<D3D12_FEATURE_DATA_SHADER_MODEL>() as u32,
            )
        } {
            Ok(()) => {
                sm_err = None;
                break;
            }
            Err(e) => sm_err = Some(e),
        }
    }
    if let Some(e) = sm_err {
        return Err(format!("CheckFeatureSupport(SHADER_MODEL): {e}"));
    }
    // Resource binding tier: the root signature's unbounded scene-texture
    // SRV range needs tier 2+ (tier 3 on all RT-capable hardware in
    // practice — belt-and-braces with the loud-fallback story).
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            &mut o as *mut _ as *mut _,
            std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
        )
    }
    .map_err(|e| format!("CheckFeatureSupport(OPTIONS): {e}"))?;
    // Wave ops + lane-count range. Reporting only (see Caps), so a failure
    // here must not sink a session that would otherwise run: fall back to the
    // all-zero default, which reads as "unknown" everywhere it is printed.
    let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    if unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            &mut o1 as *mut _ as *mut _,
            std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>() as u32,
        )
    }
    .is_err()
    {
        o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    }
    // OPTIONS21 postdates the D3D12 runtimes we still support, and an unknown
    // feature enum ERRORS rather than zero-filling — so this query failing is
    // the ordinary "no work graphs here" answer, not a problem to report.
    let mut o21 = D3D12_FEATURE_DATA_D3D12_OPTIONS21::default();
    if unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS21,
            &mut o21 as *mut _ as *mut _,
            std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS21>() as u32,
        )
    }
    .is_err()
    {
        o21 = D3D12_FEATURE_DATA_D3D12_OPTIONS21::default();
    }
    Ok(Caps {
        rt_tier: o5.RaytracingTier.0,
        shader_model: sm.HighestShaderModel.0,
        binding_tier: o.ResourceBindingTier.0,
        wave_ops: o1.WaveOps.as_bool(),
        wave_lane_min: o1.WaveLaneCountMin,
        wave_lane_max: o1.WaveLaneCountMax,
        total_lanes: o1.TotalLaneCount,
        work_graphs_tier: o21.WorkGraphsTier.0,
    })
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
    if caps.binding_tier < D3D12_RESOURCE_BINDING_TIER_2.0 {
        missing.push(format!(
            "resource binding tier 2 (unbounded texture table) — device reports tier {}",
            caps.binding_tier
        ));
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
    // RP_NPPD_*: the NPPD staging buffers (u28..u31), appended after RP_GBUF
    // for the same no-renumber reason; bound only in NPPD sessions (unbound
    // root UAVs are fine as long as no dispatched kernel touches them).
    for i in 0..4 {
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: NPPD_REG_BASE + i,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
    }
    // RP_SCENE_TEX: scene textures + UV stream, all in space1 (see the const
    // note). The Texture2D range is unbounded (NumDescriptors u32::MAX, legal
    // as the last range) — the heap slice is sized per scene at init.
    // `tex_ranges` must outlive serialization below.
    let tex_ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: TEX_TABLE_BUFS,
            BaseShaderRegister: 0,
            RegisterSpace: 1,
            OffsetInDescriptorsFromTableStart: 0,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: u32::MAX,
            BaseShaderRegister: TEX_TABLE_BUFS,
            RegisterSpace: 1,
            OffsetInDescriptorsFromTableStart: TEX_TABLE_BUFS,
        },
    ];
    params.push(D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 2,
                pDescriptorRanges: tex_ranges.as_ptr(),
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    });
    // RP_GBUF_EXT: the pack's guide/signal half (u32), appended LAST. This is
    // the parameter that takes the signature to 64/64 DWORDs — see the const.
    params.push(D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 32, RegisterSpace: 0 },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    });
    // Two static samplers (0 root-signature DWORDs), both repeat-wrap, one per
    // filter path — the GPU mirrors of texture.rs's two samplers:
    //   s0 samp_lin   trilinear, fed an explicit ray-cone lod by SampleLevel
    //                 (lod 0 on a MIP_LINEAR filter reads level 0 only, so
    //                 single-mip textures and the magnification clamp behave
    //                 exactly like the old bilinear sampler) — the --no-aniso
    //                 path and every isotropic (hemi-bounce) lap.
    //   s1 samp_aniso hardware anisotropic, fed the elliptical footprint by
    //                 SampleGrad (shade.hlsli::tri_grads — SampleLevel gives
    //                 the TMU no gradients, so aniso there would be a no-op).
    //                 MaxAnisotropy is the session's --aniso, the same cap
    //                 texture::sample_aniso clamps its tap count to.
    // The alpha-cutout test deliberately uses NEITHER — it is a nearest-texel
    // .Load (trace_common.hlsli::alpha_cutout): filtering never touches
    // visibility.
    let base = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        MipLODBias: 0.0,
        MaxAnisotropy: 0,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        BorderColor: D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ShaderRegister: 0,
        RegisterSpace: 1,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let samplers = [
        base,
        D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_ANISOTROPIC,
            MaxAnisotropy: crate::texture::max_aniso() as u32,
            ShaderRegister: 1,
            ..base
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: samplers.as_ptr(),
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

/// Minimal device/queue/list/fence harness for `--check-gpu` and the
/// cinematic capture — no window, no swapchain. Interactive mode uses `D3d`
/// instead; everything recorded against this harness records identically
/// against that one. Raw-NGX DLSS-RR evaluates on it directly (no queue hook
/// needed — the retired Streamline flavor existed only for SL's proxy).
pub struct HeadlessGpu {
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    alloc: ID3D12CommandAllocator,
    pub list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: HANDLE,
    next: u64,
    pub adapter_name: String,
    /// What the picked adapter IS — the cinematic caller's vendor gate for
    /// the raw-NGX DLSS probe (raw NGX needs no queue hook, so the harness
    /// carries no upscaler plumbing of its own; the retired `new_sl` flavor
    /// existed only to hook the queue for SL's evaluate).
    pub vendor: adapter::Vendor,
}

impl HeadlessGpu {
    pub fn new(debug: bool, prefer: adapter::Prefer) -> Result<Self> {
        let factory = adapter::create_factory(debug).map_err(|e| format!("factory: {e}"))?;
        let pick = adapter::pick(&factory, prefer)?;
        Self::from_pick(&pick, debug)
    }

    /// Open a harness on an ALREADY-CHOSEN adapter.
    ///
    /// The `--dual-gpu` entry point, and the reason it exists rather than a
    /// second `Prefer`: `adapter::pick` can only ever land on one adapter, and
    /// it RECORDS `PICKED` (the session-vendor global). Calling it twice would
    /// leave the process claiming the secondary's vendor as the session's,
    /// silently retuning `main::vendor_defaults` and the `--spin` warm-up
    /// against the wrong device. `adapter::enumerate` + this constructor is the
    /// pair that opens a second device without touching that global — which is
    /// exactly why `enumerate` deliberately does not record it either.
    ///
    /// Everything below is already adapter-parameterized, so `new` is now just
    /// `pick` + this.
    pub fn from_pick(pick: &adapter::AdapterPick, debug: bool) -> Result<Self> {
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
        Ok(Self {
            device,
            queue,
            alloc,
            list,
            fence,
            event,
            next: 1,
            adapter_name: pick.name.clone(),
            vendor: pick.vendor,
        })
    }

    /// Record + execute, WITHOUT blocking. Returns the fence value to pass to
    /// `wait`.
    ///
    /// The `--dual-gpu` primitive, and the reason it has to exist: the whole
    /// claim of split rendering is that two devices work AT THE SAME TIME, so
    /// driving them through the blocking `run` would serialize them and
    /// measure the SUM of two tracers instead of the max — reporting the
    /// feature as a large regression while it was in fact working.
    ///
    /// CONTRACT: the caller must `wait` on the previous submit before calling
    /// this again on the same harness. Resetting a command allocator whose
    /// lists are still executing is undefined behaviour, and this is the one
    /// place in the codebase where that ordering is the caller's to keep
    /// rather than the callee's.
    pub fn submit<F: FnOnce(&ID3D12GraphicsCommandList)>(&mut self, f: F) -> Result<u64> {
        unsafe { self.alloc.Reset() }.map_err(|e| format!("alloc Reset: {e}"))?;
        unsafe { self.list.Reset(&self.alloc, None) }.map_err(|e| format!("list Reset: {e}"))?;
        // A dual-GPU frame nests THIS inside the primary's open one, so the
        // restore is load-bearing: without it the primary's remaining
        // timestamps name the secondary's query heap. See `gputime::CURRENT`.
        let prev = super::gputime::begin_frame(&self.device, &self.queue, 0);
        f(&self.list);
        super::gputime::resolve(&self.list, 0);
        super::gputime::restore(prev);
        unsafe { self.list.Close() }.map_err(|e| format!("list Close: {e}"))?;
        let cl: ID3D12CommandList = self.list.cast().map_err(|e| format!("cast: {e}"))?;
        unsafe { self.queue.ExecuteCommandLists(&[Some(cl)]) };
        let v = self.next;
        self.next += 1;
        unsafe { self.queue.Signal(&self.fence, v) }.map_err(|e| format!("Signal: {e}"))?;
        Ok(v)
    }

    /// Has this `submit` value already completed? A NON-blocking query.
    ///
    /// The dual-GPU balancer's one exact signal. Asking "was `wait` fast?"
    /// cannot answer it: waiting on an already-signalled fence still burns
    /// tens of nanoseconds, so a duration test reads as "the primary was
    /// still running" on essentially every frame — which had the balancer
    /// growing the secondary's share on a box where it should shrink it to
    /// zero. Measure the condition, not a proxy for it.
    pub fn completed(&self, v: u64) -> bool {
        let done = unsafe { self.fence.GetCompletedValue() };
        done >= v
    }

    /// Block until a `submit` value completes.
    pub fn wait(&mut self, v: u64) -> Result<()> {
        if unsafe { self.fence.GetCompletedValue() } < v {
            unsafe { self.fence.SetEventOnCompletion(v, self.event) }
                .map_err(|e| format!("SetEventOnCompletion: {e}"))?;
            unsafe { WaitForSingleObject(self.event, INFINITE) };
        }
        d3d12::drain_debug(&self.device);
        Ok(())
    }

    /// Record + execute + block. The `--check-gpu` cadence: correctness
    /// first, wall-clock timing is a separate explicit segment.
    pub fn run<F: FnOnce(&ID3D12GraphicsCommandList)>(&mut self, f: F) -> Result<()> {
        unsafe { self.alloc.Reset() }.map_err(|e| format!("alloc Reset: {e}"))?;
        unsafe { self.list.Reset(&self.alloc, None) }.map_err(|e| format!("list Reset: {e}"))?;
        // --gpu-timing: every run() blocks on the fence below, so the previous
        // run's timestamps are complete by the time we get here — the same
        // wait-then-collect the frame ring gives `D3d::begin_frame`, with one
        // slot instead of FRAMES_IN_FLIGHT. Without this the headless suites
        // (the deterministic workloads, and the only place a per-pass number
        // means anything) would record no timings at all.
        // Nested inside the primary's frame on the dual-GPU band readback —
        // restore, exactly as `submit` does.
        let prev = super::gputime::begin_frame(&self.device, &self.queue, 0);
        f(&self.list);
        super::gputime::resolve(&self.list, 0);
        super::gputime::restore(prev);
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
        // Surface whatever the debug layer found (no-op unless --gpu-debug). The
        // `--check-gpu` suites record most of the renderer's command lists, so
        // this is where a state or barrier error gets caught cheaply.
        d3d12::drain_debug(&self.device);
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

impl d3d12::Submit for HeadlessGpu {
    fn run_list(
        &mut self,
        f: &mut dyn FnMut(&ID3D12GraphicsCommandList) -> Result<()>,
    ) -> Result<()> {
        let mut rec = Ok(());
        self.run(|l| rec = f(l))?;
        rec
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
/// One row per distinct thread-group width the tracer dispatches:
/// `(group_width, wave_lane_count, waves_per_group)`.
///
/// Answers the question `D3D12_OPTIONS1` cannot: the caps report a lane-count
/// RANGE, and the driver picks inside it per shader. The widths probed are the
/// ones that actually ship — `cs_level*`/`cs_hemi_*` at 32, `cs_sky` at
/// `SKY_GROUP`, `cs_leaf` at `LEAF_GROUP` — so the table lines up with the
/// `--gpu-timing` regions rather than describing hypothetical kernels.
///
/// Reporting only. Nothing branches on the result: the wave-aggregated paths
/// in wavefront.hlsl/leaf.hlsl are correct at ANY lane count (they aggregate
/// over whatever the active wave turns out to be), which is exactly why this
/// is a diagnostic and not a tuning input.
pub fn wave_probe(hg: &mut HeadlessGpu, dxc: &Dxc, debug: bool) -> Result<Vec<(u32, u32, u32)>> {
    let root_sig = create_root_signature(&hg.device)?;
    let uaf = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    let mut widths = vec![32u32, SKY_GROUP, leaf_group()];
    widths.sort_unstable();
    widths.dedup();

    let mut rows = Vec::with_capacity(widths.len());
    for w in widths {
        let src = format!("#define PROBE_GROUP {w}\n{WAVEPROBE_HLSL}");
        let what = format!("wave probe g{w}");
        let pso = compute_pso(
            &hg.device,
            &root_sig,
            &dxc.compile(&src, "cs_wave_probe", "cs_6_5", &what, debug)?,
            &what,
        )?;
        let out = committed_buffer(&hg.device, 3 * 4, uaf, ua)?;
        hg.run(|list| unsafe {
            list.SetComputeRootSignature(&root_sig);
            let push = [0u32; 4];
            list.SetComputeRoot32BitConstants(RP_PUSH, 4, push.as_ptr() as *const _, 0);
            list.SetComputeRootUnorderedAccessView(RP_UAV0, out.GetGPUVirtualAddress());
            list.SetPipelineState(&pso);
            list.Dispatch(1, 1, 1);
            list.ResourceBarrier(&[uav_barrier(None)]);
        })?;
        let b = hg.read_buffer(&out, ua, 12)?;
        let v: Vec<u32> =
            b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        rows.push((v[1], v[0], v[2]));
    }
    Ok(rows)
}

/// `FR_WORKGRAPH=1` — run the quadtree ladder as a D3D12 work graph instead of
/// `cs_seed` + depth_full x (`cs_prep` + ExecuteIndirect). R&D lever, never a
/// CLI flag (the `FR_LEAF`/`FR_ABL` idiom): shipped features get flags, spikes
/// get env levers. Read once per session.
pub(crate) fn work_graph_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FR_WORKGRAPH").map(|v| v == "1").unwrap_or(false))
}

/// queues.hlsli's `ROOT_CUT_SLOT` — the sentinel meaning "the root cut [0]".
/// Only the work-graph entry record needs it CPU-side; the ladder's root is
/// built by `cs_seed` on the GPU.
const WG_ROOT_CUT_SLOT: u32 = 0xffff_ffff;

/// The CPU twin of queues.hlsli's `TileRec` — the graph's entry record is
/// handed to `DispatchGraph` from CPU memory, so this layout is load-bearing
/// (24 B, packed like C). `cs_seed` builds the same record on the GPU for the
/// ladder; keep the two in step.
#[repr(C)]
#[derive(Clone, Copy)]
struct TileRecCpu {
    xy0: u32,
    xy1: u32,
    t_start: f32,
    cut_slot: u32,
    meta: u32,
    path: u32,
}

/// The ladder as a work graph (spike). Owns its state object and the opaque
/// backing memory the scheduler needs.
pub(crate) struct WorkGraph {
    _so: ID3D12StateObject,
    ident: D3D12_PROGRAM_IDENTIFIER,
    backing: ID3D12Resource,
    backing_size: u64,
    entry: u32,
    /// The first use of a backing allocation must carry
    /// `SET_WORK_GRAPH_FLAG_INITIALIZE` so the driver can prepare its opaque
    /// contents; afterwards it must NOT, because re-initialising every frame
    /// would redo that work. The memory is opaque — never clear it by hand.
    initialized: std::cell::Cell<bool>,
}

impl WorkGraph {
    const PROGRAM: windows::core::PCWSTR = windows::core::w!("QuadtreeGraph");
    const ENTRY_NODE: windows::core::PCWSTR = windows::core::w!("TileWide");

    /// Compile + create. `wide`/`deep` are the recursion depths each node may
    /// reach; they are DECLARED, and exceeding them at runtime is memory
    /// corruption rather than a caught error (work-graphs spec), so they come
    /// from the same `depth_full` the ladder uses, never a guess.
    fn create(
        device: &ID3D12Device,
        root_sig: &ID3D12RootSignature,
        dxc: &Dxc,
        src: &str,
        debug: bool,
    ) -> Result<WorkGraph> {
        let dxil = dxc.compile(src, "", "lib_6_8", "work graph", debug)?;

        let lib = D3D12_DXIL_LIBRARY_DESC {
            DXILLibrary: D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr() as *const _,
                BytecodeLength: dxil.len(),
            },
            NumExports: 0, // export everything
            pExports: std::ptr::null_mut(),
        };
        let grs = D3D12_GLOBAL_ROOT_SIGNATURE { pGlobalRootSignature: unsafe { std::mem::transmute_copy(root_sig) } };
        let wg = D3D12_WORK_GRAPH_DESC {
            ProgramName: Self::PROGRAM,
            Flags: D3D12_WORK_GRAPH_FLAG_INCLUDE_ALL_AVAILABLE_NODES,
            NumEntrypoints: 0,
            pEntrypoints: std::ptr::null(),
            NumExplicitlyDefinedNodes: 0,
            pExplicitlyDefinedNodes: std::ptr::null(),
        };
        let subs = [
            D3D12_STATE_SUBOBJECT {
                Type: D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY,
                pDesc: &lib as *const _ as *const _,
            },
            D3D12_STATE_SUBOBJECT {
                Type: D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE,
                pDesc: &grs as *const _ as *const _,
            },
            D3D12_STATE_SUBOBJECT {
                Type: D3D12_STATE_SUBOBJECT_TYPE_WORK_GRAPH,
                pDesc: &wg as *const _ as *const _,
            },
        ];
        let so_desc = D3D12_STATE_OBJECT_DESC {
            Type: D3D12_STATE_OBJECT_TYPE_EXECUTABLE,
            NumSubobjects: subs.len() as u32,
            pSubobjects: subs.as_ptr(),
        };
        let dev9: ID3D12Device9 = device
            .cast()
            .map_err(|e| format!("work graph: ID3D12Device9 unavailable: {e}"))?;
        let so: ID3D12StateObject = unsafe { dev9.CreateStateObject(&so_desc) }
            .map_err(|e| format!("work graph: CreateStateObject: {e}"))?;

        let props: ID3D12StateObjectProperties1 = so
            .cast()
            .map_err(|e| format!("work graph: ID3D12StateObjectProperties1: {e}"))?;
        let wgp: ID3D12WorkGraphProperties = so
            .cast()
            .map_err(|e| format!("work graph: ID3D12WorkGraphProperties: {e}"))?;

        let index = unsafe { wgp.GetWorkGraphIndex(Self::PROGRAM) };
        let mut req = D3D12_WORK_GRAPH_MEMORY_REQUIREMENTS::default();
        unsafe { wgp.GetWorkGraphMemoryRequirements(index, &mut req) };
        let entry = unsafe {
            wgp.GetEntrypointIndex(
                index,
                D3D12_NODE_ID { Name: Self::ENTRY_NODE, ArrayIndex: 0 },
            )
        };
        // An export-list or node-name mistake yields an EMPTY graph that
        // creates SUCCESSFULLY and whose identifier is zeroed — so the entry
        // lookup, not creation, is where a typo actually surfaces.
        if entry == u32::MAX {
            return Err("work graph: entry node \"TileWide\" not found (empty graph?)".into());
        }

        // Zero is a legal requirement (the driver may need none), in which
        // case a null range is passed at SetProgram time.
        let backing_size = req.MinSizeInBytes.max(1);
        let backing = committed_buffer(
            device,
            backing_size,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        eprintln!(
            "gpu work-graph: armed (backing {:.2} MB, min {} / max {} / gran {})",
            backing_size as f64 / (1024.0 * 1024.0),
            req.MinSizeInBytes,
            req.MaxSizeInBytes,
            req.SizeGranularityInBytes
        );
        Ok(WorkGraph {
            ident: unsafe { props.GetProgramIdentifier(Self::PROGRAM) },
            _so: so,
            backing,
            backing_size,
            entry,
            initialized: std::cell::Cell::new(false),
        })
    }

    /// Record the ladder. The caller has already bound the root signature and
    /// every root argument — `SetProgram` is a successor to `SetPipelineState`
    /// and does not reset root arguments, but the ORDER (root signature, then
    /// arguments, then program) is the one the spec leaves unambiguous.
    unsafe fn record(&self, list: &ID3D12GraphicsCommandList, root: TileRecCpu) -> Result<()> {
        let l10: ID3D12GraphicsCommandList10 = list
            .cast()
            .map_err(|e| format!("work graph: ID3D12GraphicsCommandList10: {e}"))?;
        unsafe {
            let mut set = D3D12_SET_PROGRAM_DESC {
                Type: D3D12_PROGRAM_TYPE_WORK_GRAPH,
                ..Default::default()
            };
            set.Anonymous.WorkGraph = D3D12_SET_WORK_GRAPH_DESC {
                ProgramIdentifier: self.ident,
                Flags: if self.initialized.get() {
                    D3D12_SET_WORK_GRAPH_FLAGS(0)
                } else {
                    D3D12_SET_WORK_GRAPH_FLAG_INITIALIZE
                },
                BackingMemory: D3D12_GPU_VIRTUAL_ADDRESS_RANGE {
                    StartAddress: self.backing.GetGPUVirtualAddress(),
                    SizeInBytes: self.backing_size,
                },
                NodeLocalRootArgumentsTable: Default::default(),
            };
            l10.SetProgram(&set);
            self.initialized.set(true);

            let mut dg = D3D12_DISPATCH_GRAPH_DESC {
                Mode: D3D12_DISPATCH_MODE_NODE_CPU_INPUT,
                ..Default::default()
            };
            dg.Anonymous.NodeCPUInput = D3D12_NODE_CPU_INPUT {
                EntrypointIndex: self.entry,
                NumRecords: 1,
                // Copied by the driver during recording, so a stack local is
                // fine — it is not referenced after DispatchGraph returns.
                pRecords: &root as *const _ as *const _,
                RecordStrideInBytes: std::mem::size_of::<TileRecCpu>() as u64,
            };
            l10.DispatchGraph(&dg);
        }
        Ok(())
    }
}

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
#[derive(Clone, Copy)]
struct GpuBvhNode {
    mn: [f32; 3],
    left_first: u32,
    mx: [f32; 3],
    count: u32,
}

/// scene.rs::Material packed for StructuredBuffer<Mat> (shade.hlsli).
/// 108 B — the HLSL `Mat` mirrors this field-for-field; a stride skew reads
/// garbage, so the two must move in the same commit.
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuMat {
    albedo: [f32; 3],
    roughness: f32,
    metallic: f32,
    anisotropy: f32,
    kind: u32, // 0 = diffuse, 1 = marble, 2 = textured
    scale: f32,
    sheen: f32,
    translucency: f32,
    transmission: f32,
    /// `Scene::textures` index for MAT_TEXTURED (the space1 `texs[]` slot).
    tex: u32,
    emissive: [f32; 3],
    normal_tex: u32, // NO_TEX sentinel = HLSL TEX_NONE
    rough_tex: u32,
    metal_tex: u32,
    emissive_tex: u32,
    normal_scale: f32,
    trans_tint: [f32; 3], // transmission/absorption tint; .x < 0 = "use albedo"
    ior: f32,             // Snell/Fresnel IOR (default 1.5; water 1.33)
    ripple_amp: f32,      // water ripple slope amplitude (0 = none)
    // Per-material world-space detail texel scale (Scene::detail_scales —
    // never per-face, which seams on greedy-meshed atlases). 0 = field off.
    detail_scale: f32,
    // Spec-AA: the normal map's slope-variance companion texture
    // (Scene::tex_var; TEX_NONE = none — every unmapped/lever-off material,
    // the fold's structural map-arm off state).
    normal_var_tex: u32,
}

/// Bytes of reusable staging streamed per blocking submit — bounds the
/// upload's transient commit to one chunk instead of a full second copy of
/// every scene stream (which at 100M tris was ~7 GB of upload heaps ON TOP of
/// ~7 GB of repack Vecs).
const STAGE_CHUNK: usize = 256 << 20;

/// Steady-state byte total of the scene's buffer streams (excludes textures,
/// acceleration structures, and the wavefront-only software trees — those
/// live in `SwTreesGpu`) — sizes the staging ring and the init report.
fn scene_stream_bytes(scene: &Scene) -> usize {
    let v = scene.positions.len();
    let t = scene.indices.len();
    let m = scene.materials.len();
    v * (12 + 12 + 8) + t * (12 + 4) + m * (size_of::<GpuMat>() + 4)
}

/// The software acceleration structure(s) that ride t0 for the FRUSTUM
/// queries (their only consumer — every actual ray is DXR RayQuery).
/// WAVEFRONT-ONLY, which is why these live outside the shared `SceneGpu`
/// core: DxrGpu never binds any of them, so a DXR session simply never
/// uploads them (~2.3 GB at 100M tris — what the old `SwAccel::None`
/// dummies bought, now structural). `ftree_nodes` is the per-consumer split
/// ON the GPU: the tile kernels compile `#define FTREE` and bind the wide
/// tree at t0 (long queries, wide wins big), while `record_hemi` rebinds the
/// binary tree for the hemi kernels (hemi bound queries terminate in ~10
/// visits — a binary pop is 1 box test where a wide pop is always 8, and the
/// wide tree measured +35% there).
pub struct SwTreesGpu {
    pub bvh_nodes: ID3D12Resource,
    pub tri_idx: ID3D12Resource,
    /// The 8-wide frustum tree, uploaded in its QUANTIZED wire format
    /// (ftree::QFNode, 112 B — the HLSL FtNode mirror; self_test audits
    /// containment): the per-processor split verdict — the CPU keeps the
    /// f32 nodes, the GPU trades decode ALU for -56% tree bandwidth/VRAM.
    /// None when the ftree lever is off.
    pub ftree_nodes: Option<ID3D12Resource>,
    /// --sw-rays + FTREE only: the wide-tree slot→binary-node map (flat
    /// nodes × 8 u32s — the FNode.bnode field the quantized wire format
    /// deliberately drops), bound at t1 during the level ladder so
    /// level_finish can translate a slot-ref leaf cut into the binary node
    /// ids the software ray traversal seeds from.
    pub ft_bnode: Option<ID3D12Resource>,
}

impl SwTreesGpu {
    /// Upload the binary BVH (+ the wide tree when built) through a
    /// buffer-sized staging ring. Separate from `SceneGpu::new_uploaded` by
    /// design: the shared core is per-SESSION (both tracers hold an Rc),
    /// these trees are per-TraceGpu.
    fn new_uploaded(
        device: &ID3D12Device,
        bvh: &Bvh,
        ft: Option<&crate::ftree::FTree>,
        sub: &mut dyn d3d12::Submit,
    ) -> Result<Self> {
        let srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let bytes = bvh.nodes.len() * size_of::<GpuBvhNode>()
            + bvh.tri_idx.len() * 4
            + ft.map_or(0, |f| f.quantized_bytes());
        let ring = d3d12::UploadBuffer::new(device, STAGE_CHUNK.min(bytes.max(4096)))?;
        let bvh_nodes = stream_buffer(
            device,
            sub,
            &ring,
            &bvh.nodes,
            |n| GpuBvhNode {
                mn: [n.aabb.min.x, n.aabb.min.y, n.aabb.min.z],
                left_first: n.left_first,
                mx: [n.aabb.max.x, n.aabb.max.y, n.aabb.max.z],
                count: n.count,
            },
            srv,
        )?;
        let tri_idx = stream_buffer(device, sub, &ring, &bvh.tri_idx, |t| *t, srv)?;
        let ftree_nodes = match ft {
            Some(f) => {
                let qn = f.quantized();
                Some(stream_buffer(device, sub, &ring, &qn, |n| *n, srv)?)
            }
            None => None,
        };
        // --sw-rays leaf-cut translation map (cut-consuming lever + ftree
        // sessions only — --no-cut-rays compiles the translation out too).
        let ft_bnode = match ft {
            Some(f) if sw_rays_leaf() => {
                let flat = f.bnode_flat();
                Some(stream_buffer(device, sub, &ring, &flat, |t| *t, srv)?)
            }
            _ => None,
        };
        let bnode_bytes = ft_bnode.as_ref().map_or(0, |_| ft.map_or(0, |f| f.nodes.len() * 32));
        eprintln!(
            "gpu sw-trees: {} MB (binary bvh + tri idx{}{})",
            (bytes + bnode_bytes) >> 20,
            match ft {
                Some(f) => format!(", ftree {} MB", f.quantized_bytes() >> 20),
                None => String::new(),
            },
            if ft_bnode.is_some() {
                format!(", sw-rays bnode {} MB", bnode_bytes >> 20)
            } else {
                String::new()
            }
        );
        Ok(Self { bvh_nodes, tri_idx, ftree_nodes, ft_bnode })
    }
}

/// Stream `src` into a new default-heap buffer through `ring`, `map`ping each
/// element into the mapped staging pointer chunk-by-chunk (identity for
/// layout-compatible streams, a repack for Vec3A→float3 / BvhNode→GpuBvhNode).
/// Each chunk is one blocking `Submit::run_list`; the final chunk's list also
/// records the COPY_DEST→`after` transition. Empty streams get a 4-byte dummy
/// created directly in `after`.
///
/// CONTRACT: `map` is called exactly ONCE per element, in `src` order — the
/// `blas_indices` stream's chunk cursor (SceneGpu::new_uploaded) depends on
/// it. Parallelizing the map or retrying a ring chunk would silently desync
/// that cursor into wrong BLAS geometry.
fn stream_buffer<T: Copy, U: Copy>(
    device: &ID3D12Device,
    sub: &mut dyn d3d12::Submit,
    ring: &d3d12::UploadBuffer,
    src: &[T],
    map: impl Fn(&T) -> U,
    after: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource> {
    if src.is_empty() {
        return committed_buffer(device, 4, D3D12_RESOURCE_FLAG_NONE, after);
    }
    let total = std::mem::size_of::<U>() * src.len();
    let dst = committed_buffer(
        device,
        total as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let per = (ring.size / std::mem::size_of::<U>()).max(1);
    let mut e = 0usize;
    while e < src.len() {
        let n = per.min(src.len() - e);
        unsafe {
            let out = std::slice::from_raw_parts_mut(ring.ptr as *mut U, n);
            for (o, s) in out.iter_mut().zip(&src[e..e + n]) {
                *o = map(s);
            }
        }
        let off = (e * std::mem::size_of::<U>()) as u64;
        let bytes = (n * std::mem::size_of::<U>()) as u64;
        let last = e + n == src.len();
        sub.run_list(&mut |l| {
            unsafe { l.CopyBufferRegion(&dst, off, &ring.resource, 0, bytes) };
            if last {
                unsafe {
                    l.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_COPY_DEST, after)])
                };
            }
            Ok(())
        })?;
        e += n;
    }
    Ok(dst)
}

/// FR_SPLIT_AUDIT=1 helper: copy a whole u32 stream buffer back to the CPU and
/// diff it against the source slice it was streamed from. Diagnostic-only —
/// never in a shipping path; the state round-trip mirrors stream_buffer's
/// `after` (NON_PIXEL_SHADER_RESOURCE).
fn audit_split_streams(
    device: &ID3D12Device,
    sub: &mut dyn d3d12::Submit,
    buf: &ID3D12Resource,
    want: &[u32],
    name: &str,
) -> Result<()> {
    let bytes = want.len() * 4;
    let rb = d3d12::ReadbackBuffer::new(device, bytes)?;
    sub.run_list(&mut |l| {
        unsafe {
            l.ResourceBarrier(&[transition(
                buf,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyBufferRegion(&rb.resource, 0, buf, 0, bytes as u64);
            l.ResourceBarrier(&[transition(
                buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
        }
        Ok(())
    })?;
    let mut ptr = std::ptr::null_mut();
    unsafe { rb.resource.Map(0, None, Some(&mut ptr)) }
        .map_err(|e| format!("split-audit Map: {e}"))?;
    let got = unsafe { std::slice::from_raw_parts(ptr as *const u32, want.len()) };
    let bad = got.iter().zip(want).filter(|(g, w)| g != w).count();
    if bad == 0 {
        eprintln!("split-audit: {name} MATCHES the CPU plan ({} u32s)", want.len());
    } else {
        let first = got.iter().zip(want).position(|(g, w)| g != w).unwrap();
        eprintln!(
            "split-audit: {name} DIVERGES — {bad} of {} u32s differ, first at [{first}] \
             (gpu {} vs plan {})",
            want.len(),
            got[first],
            want[first]
        );
    }
    unsafe { rb.resource.Unmap(0, None) };
    Ok(())
}

/// The SHARED scene core: everything both GPU tracers read — streams,
/// materials, textures, the driver BLAS/TLAS, the blas-split remap.
/// Immutable after upload; held as `Rc<SceneGpu>` by TraceGpu AND DxrGpu
/// `--dxr-sbt`'s per-chunk class labels (see `SceneGpu::sbt_class`).
pub struct SbtClassInfo {
    /// One shading class per chunk, parallel to the refined plan's chunks —
    /// `blas_split::ClassRefine::chunk_class`, kept CPU-side for the audit
    /// (the contribution itself is baked into the TLAS instance descs).
    pub chunk_class: Vec<u8>,
    /// Per-class chunk counts (`shadeclass::histogram`) — the arrival line
    /// + the `--check-dxr` ≥2-classes must-fire.
    pub histo: [u32; crate::shadeclass::N_CLASSES],
}

/// (cached in GpuContext, so the second tracer and every resize re-entry
/// skip the upload + BLAS build entirely). The wavefront-only software
/// trees live in `SwTreesGpu`, deliberately outside the core.
pub struct SceneGpu {
    pub positions: ID3D12Resource,
    pub normals: ID3D12Resource,
    pub indices: ID3D12Resource,
    pub tri_mat: ID3D12Resource,
    pub materials: ID3D12Resource,
    /// Never read, but MUST be held: the TLAS instance descs bake only the
    /// (compacted) BLAS's GPU VA — dropping this resource would free the
    /// memory the TLAS points into. Under `--blas-split` this is the ARENA all
    /// chunk BLASes were sub-allocated from, so the same rule covers all of
    /// them at once.
    #[allow(dead_code)]
    pub blas: ID3D12Resource,
    pub tlas: ID3D12Resource,
    /// `--foliage-sway` with leaf cells present (else None — every other
    /// session is structurally the pre-sway build): the animated-TLAS ring
    /// an animated frame rebuilds per clock tick — BOTH pipelines since
    /// v0.2. The STATIC `tlas` above stays the rest pose and keeps serving
    /// every headless gate/bench (sway_time None); the frustum/temporal
    /// machinery is sound under motion because the uploaded software BVH's
    /// leaf boxes are swept (`bvh::grow_sway_sweep`).
    pub sway: Option<SwayGpu>,
    /// `--dxr-sbt` armed on a `--blas-split` upload (else None — the
    /// `sway: None` structural-off shape): the per-chunk shading classes the
    /// TLAS instances carry as `InstanceContributionToHitGroupIndex` (× the
    /// 3 ray types), plus the histogram the `--check-dxr` construction audit
    /// and the `gpu scene:` line read. The wavefront pipeline ignores hit
    /// groups entirely (RayQuery), so the grown TLAS is transparent to it —
    /// the remap contract is untouched by construction (instance-keyed, not
    /// multi-geometry: PrimitiveIndex() restarts per GEOMETRY, which would
    /// have broken `tri_of` on both pipelines).
    pub sbt_class: Option<SbtClassInfo>,
    /// `--blas-split` only (4-byte dummies otherwise): the reordered index
    /// stream the chunk BLASes were built over, and the remap the shaders read
    /// to recover a triangle id from `(InstanceID, PrimitiveIndex)` —
    /// `blas_tri[chunk_base[inst] + prim]`. Held by SRVs in the space1 table.
    pub blas_tri: ID3D12Resource,
    pub chunk_base: ID3D12Resource,
    /// Packed slots (== `n_tris` when armed, 0 when not) and chunk count — the
    /// SRV element counts, and the must-fire the GPU gates read.
    n_packed: u32,
    pub n_chunks: u32,
    /// `Scene::texcoords` as float2 per vertex (parallel to positions; zeros
    /// on procedural scenes — 4 dummy bytes, never read there).
    pub texcoords: ID3D12Resource,
    /// Per-material cutout map: `tex + 1` when the material is textured AND
    /// its texture has masked texels, else 0 (mirrors the bvh.rs cutout
    /// gates: MatKind::Textured -> Texture::alpha_masked).
    pub mat_cutout: ID3D12Resource,
    /// Per-material relief map: `normal_tex + 1` + `height_amp` bits where
    /// the material carries a heightfield, else zeros (mirrors the bvh.rs
    /// tri_height_depth gates).
    pub mat_height: ID3D12Resource,
    /// Per-material tinted-shadow map: rgb = `Material::shadow_tint`, a =
    /// transmission (a == 0 ⇒ opaque) — `transmit_q`'s per-interface data.
    pub mat_shadow: ID3D12Resource,
    /// One RGBA8 Texture2D per `Scene::textures` entry, 1 mip (the CPU
    /// samples bilinear with no mip chain — parity over aliasing); _SRGB for
    /// color textures, _UNORM for linear-data maps (Texture::srgb).
    pub textures: Vec<ID3D12Resource>,
    /// The sway-MV delta table's stand-in when `sway` is None: 16 bytes so
    /// the t9 space1 slot always has a describable float4 (the blas_tri
    /// 4-byte-dummy discipline — the table's shape never moves with the
    /// feature; the kernels compile without SWAY_MV and never read it).
    /// Deliberately NOT one of the per-material buffers: those degrade to
    /// 4-byte dummies on material-free scenes, and a 16-byte view over one
    /// is the over-long-view device removal.
    dmv_dummy: ID3D12Resource,
    n_verts: u32,
    n_tris: u32,
    n_mats: u32,
}

impl SceneGpu {
    /// Create AND upload the shared scene core in one call: every stream
    /// chunks through one reusable `STAGE_CHUNK` staging ring (blocking
    /// submits via `sub`), then the BLAS + TLAS build rides a final submit —
    /// scratch and staging are gone by the time this returns, so peak commit
    /// is steady-state + one chunk, not steady-state × 2. Runs ONCE per
    /// session (GpuContext caches the Rc); the wavefront's software trees
    /// upload separately per TraceGpu (`SwTreesGpu::new_uploaded`).
    /// `bc7_mode`: block-compress the OPAQUE scene textures (the DEFAULT —
    /// `Gpu(Fast)`: the compute encoder in gpu/bc7gpu.rs runs per band inside
    /// this upload; `Cpu(q)` = the `--bc7-cpu` ispc A/B arm, pre-encoded
    /// below; `Off` = `--no-bc7`, RGBA8 everywhere). Alpha-masked cutout
    /// textures stay RGBA8 in every mode — see src/bc7.rs.
    /// `bvh` is the CHUNKING source for `--blas-split` and is read only when
    /// that lever is armed.
    pub fn new_uploaded(
        device: &ID3D12Device,
        scene: &Scene,
        bvh: &Bvh,
        sub: &mut dyn d3d12::Submit,
        bc7_mode: bc7::Bc7Mode,
    ) -> Result<Self> {
        let device5: ID3D12Device5 = device
            .cast()
            .map_err(|e| format!("ID3D12Device5 (require_caps should have gated): {e}"))?;

        let srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        // The texture band loop below needs at least ONE full aligned row in
        // the ring (`scene_stream_bytes` deliberately excludes textures, and
        // its geometry-only size can undershoot a wide texture's pitch on a
        // small mesh — the band `.max(1)` would then overrun the mapping).
        // A BC7 texture's "row" is a 4-texel-tall BLOCK row: on the CPU
        // (ispc) arm the ring stages ENCODED block rows (block pitch); on the
        // GPU arm it stages the 4 SOURCE texel rows a block row encodes from
        // (4 aligned RGBA8 rows). Mispredicting this here is exactly the
        // overrun the comment above warns about.
        let max_tex_pitch = scene
            .textures
            .iter()
            .map(|t| match bc7_mode {
                bc7::Bc7Mode::Cpu(_) if bc7::should_compress(t) => d3d12::block_pitch(t.w),
                bc7::Bc7Mode::Gpu(_) if bc7::should_compress(t) => {
                    4 * d3d12::aligned_pitch(t.w as usize * 4)
                }
                _ => d3d12::aligned_pitch(t.w as usize * 4),
            })
            .max()
            .unwrap_or(0);
        let ring = d3d12::UploadBuffer::new(
            device,
            STAGE_CHUNK
                .min(scene_stream_bytes(scene).max(4096))
                .max(max_tex_pitch),
        )?;

        let positions_b = stream_buffer(
            device,
            sub,
            &ring,
            &scene.positions,
            |p| [p.x, p.y, p.z],
            srv,
        )?;
        let normals_b =
            stream_buffer(device, sub, &ring, &scene.normals, |n| [n.x, n.y, n.z], srv)?;
        // [u32;3] tris flatten to the u32 index stream by layout.
        let indices_b = stream_buffer(device, sub, &ring, &scene.indices, |t| *t, srv)?;
        let tri_mat = stream_buffer(device, sub, &ring, &scene.tri_mat, |t| *t, srv)?;
        let texcoords_b = stream_buffer(
            device,
            sub,
            &ring,
            &scene.texcoords,
            |t| [t.x, t.y],
            srv,
        )?;
        let materials: Vec<GpuMat> = scene
            .materials
            .iter()
            .enumerate()
            .map(|(mi, m)| GpuMat {
                albedo: [m.albedo.x, m.albedo.y, m.albedo.z],
                roughness: m.roughness,
                metallic: m.metallic,
                anisotropy: m.anisotropy,
                kind: match m.kind {
                    MatKind::Diffuse => 0,
                    MatKind::Marble { .. } => 1,
                    MatKind::Textured { .. } => 2,
                },
                scale: match m.kind {
                    MatKind::Marble { scale } => scale,
                    _ => 0.0,
                },
                sheen: m.sheen,
                translucency: m.translucency,
                transmission: m.transmission,
                tex: match m.kind {
                    MatKind::Textured { tex } => tex,
                    _ => 0,
                },
                emissive: [m.emissive.x, m.emissive.y, m.emissive.z],
                normal_tex: m.normal_tex,
                rough_tex: m.rough_tex,
                metal_tex: m.metal_tex,
                emissive_tex: m.emissive_tex,
                normal_scale: m.normal_scale,
                trans_tint: [m.trans_tint.x, m.trans_tint.y, m.trans_tint.z],
                ior: m.ior,
                ripple_amp: m.ripple_amp,
                detail_scale: scene.detail_scales.get(mi).copied().unwrap_or(0.0),
                normal_var_tex: scene
                    .tex_var
                    .get(m.normal_tex as usize)
                    .copied()
                    .unwrap_or(crate::scene::NO_TEX),
            })
            .collect();
        let materials_b = stream_buffer(device, sub, &ring, &materials, |m| *m, srv)?;
        // Per-material cutout map the alpha_cutout helper consumes.
        let mat_cutout: Vec<u32> = scene
            .materials
            .iter()
            .map(|m| match m.kind {
                MatKind::Textured { tex } if scene.textures[tex as usize].alpha_masked => tex + 1,
                _ => 0,
            })
            .collect();
        let mat_cutout_b = stream_buffer(device, sub, &ring, &mat_cutout, |m| *m, srv)?;
        // Per-material relief map the height_march helper consumes: normal
        // tex + 1 where the material carries a heightfield, else 0, plus the
        // amp (texel widths). The nonzero set is exactly the h2n/n2h texture
        // set — the same predicate `bc7::should_compress` excludes, so every
        // texture the march can `.Load` is RGBA8 (the mat_cutout agreement
        // argument verbatim).
        let mat_height: Vec<[u32; 2]> = scene
            .materials
            .iter()
            .map(|m| {
                if m.height_amp > 0.0 && m.normal_tex != crate::scene::NO_TEX {
                    [m.normal_tex + 1, m.height_amp.to_bits()]
                } else {
                    [0, 0]
                }
            })
            .collect();
        let mat_height_b = stream_buffer(device, sub, &ring, &mat_height, |m| *m, srv)?;
        // Per-material tinted-shadow data `transmit_q` consumes: rgb = the
        // interface tint (`Material::shadow_tint` — the ONE tint source, so
        // CPU↔GPU agreement is by data), a = transmission (a == 0 ⇒ opaque).
        let mat_shadow: Vec<[f32; 4]> = scene
            .materials
            .iter()
            .map(|m| {
                let t = m.shadow_tint();
                [t.x, t.y, t.z, m.transmission]
            })
            .collect();
        let mat_shadow_b = stream_buffer(device, sub, &ring, &mat_shadow, |m| *m, srv)?;
        // BC7 (ON BY DEFAULT — --no-bc7 kills): block-compress the OPAQUE
        // 4-aligned textures on upload (8 bpp vs 32 — Intel Sponza's set is
        // 4.6 GB of VRAM as RGBA8). Alpha-masked cutout textures are EXCLUDED
        // and stay RGBA8: the intersector `.Load()`s their alpha per texel
        // against a hard `< 128` threshold, and BC7 quantizes alpha across it
        // (a .Load on a BC SRV returns the DECODED — lossy — texel).
        // `bc7::should_compress`'s masked arm is the same predicate as
        // `mat_cutout` below — see src/bc7.rs for why that agreement IS the
        // soundness argument.
        //
        // There is deliberately no BC7 disk cache: the encode runs every load
        // — affordable because the DEFAULT arm is the GPU compute encoder
        // (gpu/bc7gpu.rs), dispatched per band inside the upload loop below.
        // The `--bc7-cpu` ispc arm pre-encodes here instead (LPT largest-first
        // for the same reason the DECODE sites sort — cost is ~linear in
        // texels; results scatter back by texture id, which must never shift).
        let mut compress: Vec<bool> = scene
            .textures
            .iter()
            .map(|t| bc7_mode.armed() && bc7::should_compress(t))
            .collect();
        let mut bc7_blocks: Vec<Option<Vec<Vec<u8>>>> =
            scene.textures.iter().map(|_| None).collect();
        let mut enc_ms = 0.0f64;
        // Every ENCODED level counts (base + mips, ~4/3 of base) — the
        // Mtexel/s print divides by this, and both arms encode the chain.
        let enc_texels: u64 = compress
            .iter()
            .zip(&scene.textures)
            .filter(|&(&c, _)| c)
            .map(|(_, t)| {
                std::iter::once((t.w, t.h))
                    .chain(t.mips.iter().map(|m| (m.w, m.h)))
                    .map(|(w, h)| w as u64 * h as u64)
                    .sum::<u64>()
            })
            .sum();
        if let bc7::Bc7Mode::Cpu(q) = bc7_mode {
            use rayon::prelude::*;
            let t0 = std::time::Instant::now();
            let mut order: Vec<usize> =
                (0..scene.textures.len()).filter(|&i| compress[i]).collect();
            order.sort_by_key(|&i| {
                std::cmp::Reverse(scene.textures[i].w as u64 * scene.textures[i].h as u64)
            });
            let done: Vec<(usize, Vec<Vec<u8>>)> = order
                .par_iter()
                .map(|&i| {
                    let t = &scene.textures[i];
                    // Every level of the chain — a BC7 resource cannot mix
                    // formats per mip, and the CPU/GPU trilinear parity
                    // wants the same chain depth on both sides.
                    let mut levels = vec![bc7::encode_opaque(t, q)];
                    levels.extend(t.mips.iter().map(|m| bc7::encode_level(m.w, m.h, &m.texels, q)));
                    (i, levels)
                })
                .collect();
            for (i, blocks) in done {
                bc7_blocks[i] = Some(blocks);
            }
            enc_ms = t0.elapsed().as_secs_f64() * 1e3;
        }
        // GPU arm: one encoder + one block buffer (sized to the largest whole
        // compressible mip), reused across every band — the ring's own reuse
        // discipline. Construction failure is a LOUD line + uncompressed
        // RGBA8, never an implicit CPU-encode stall (the default-on contract;
        // --check-gpu's bc7-gpu gate turns the same failure into a suite
        // FAIL so it can't rot silently).
        let mut gpu_enc: Option<super::bc7gpu::Bc7Enc> = None;
        if let bc7::Bc7Mode::Gpu(_) = bc7_mode {
            if compress.iter().any(|&c| c) {
                let block_cap = scene
                    .textures
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| compress[*i])
                    .flat_map(|(_, t)| {
                        std::iter::once((t.w, t.h))
                            .chain(t.mips.iter().map(|m| (m.w, m.h)))
                            .map(|(w, h)| d3d12::block_pitch(w) * bc7::blocks(h) as usize)
                    })
                    .max()
                    .unwrap_or(0);
                match super::bc7gpu::Bc7Enc::new(device, block_cap) {
                    Ok(e) => gpu_enc = Some(e),
                    Err(e) => {
                        eprintln!(
                            "bc7: GPU encoder unavailable ({e}) — textures upload UNCOMPRESSED \
                             RGBA8 (--bc7-cpu forces the CPU encode)"
                        );
                        compress.iter_mut().for_each(|c| *c = false);
                    }
                }
            }
        }
        let gpu_effort = bc7_mode.quality().map(super::bc7gpu::effort).unwrap_or(1);

        // Scene textures: RGBA8 Texture2Ds — _SRGB for color textures (the
        // per-texel decode of texture.rs::sample_bilinear in hardware) and
        // plain _UNORM for linear-data maps (normal / rough-metal; the CPU
        // samples those via sample_bilinear_linear). Texels upload raw (row
        // 0 = v0, the V flip is baked at OBJ load) in row bands through the
        // same staging ring — no per-texture staging commit. Compressed ones
        // upload as BC7 (same _SRGB/_UNORM role split): the DEFAULT Gpu arm
        // stages the SOURCE texel rows and encodes per band on the GPU (ring
        // SRV → block_buf → CopyTextureRegion — the blocks never touch the
        // CPU); the --bc7-cpu arm stages its pre-encoded 4-texel-tall block
        // rows, dropped as we go so steady-state RAM is unchanged.
        //
        // The band-loop arm per (texture, mip): what one ring "unit" is.
        enum TexArm {
            Rgba,   // unit = 1 texel row, straight CopyTextureRegion
            CpuBlk, // unit = 1 ENCODED block row, straight CopyTextureRegion
            GpuEnc, // unit = 4 SOURCE texel rows, dispatch + copy-out
        }
        let mut textures_v = Vec::new();
        for (i, t) in scene.textures.iter().enumerate() {
            let fmt = if compress[i] {
                bc7::dxgi_format(t)
            } else if t.srgb {
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
            } else {
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM
            };
            let arm = if !compress[i] {
                TexArm::Rgba
            } else if bc7_blocks[i].is_some() {
                TexArm::CpuBlk
            } else {
                TexArm::GpuEnc
            };
            let n_mips = 1 + t.mips.len();
            let dst = d3d12::committed_tex_mips(
                device,
                t.w,
                t.h,
                n_mips as u16,
                fmt,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            // Every mip is its own subresource, streamed through the same
            // ring in row bands. The COPY_DEST → NPSR transition rides the
            // last band of the LAST mip.
            for mip in 0..n_mips {
                let (mw, mh, mip_texels): (u32, u32, &[[u8; 4]]) = if mip == 0 {
                    (t.w, t.h, &t.texels)
                } else {
                    let m = &t.mips[mip - 1];
                    (m.w, m.h, &m.texels)
                };
                let row_pitch = d3d12::aligned_pitch(mw as usize * 4);
                let (pitch, rows_total, row_h) = match arm {
                    // A BC7 "row" is a block row: 4 texel rows — encoded
                    // ceil(w/4)*16 B on the CPU arm, 4 aligned source rows
                    // on the GPU arm (per-mip dims either way).
                    TexArm::CpuBlk => {
                        (d3d12::block_pitch(mw), bc7::blocks(mh) as usize, bc7::BLOCK as usize)
                    }
                    TexArm::GpuEnc => {
                        (4 * row_pitch, bc7::blocks(mh) as usize, bc7::BLOCK as usize)
                    }
                    TexArm::Rgba => (row_pitch, mh as usize, 1),
                };
                let band = (ring.size / pitch).max(1).min(rows_total);
                let mut r0 = 0usize;
                while r0 < rows_total {
                    let rows = band.min(rows_total - r0);
                    for r in 0..rows {
                        match arm {
                            TexArm::CpuBlk => {
                                let src_pitch = bc7::blocks(mw) as usize * bc7::BLOCK_BYTES;
                                let b = bc7_blocks[i].as_ref().unwrap();
                                let src = &b[mip][(r0 + r) * src_pitch..(r0 + r + 1) * src_pitch];
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        src.as_ptr(),
                                        ring.ptr.add(r * pitch),
                                        src_pitch,
                                    )
                                };
                            }
                            TexArm::GpuEnc => {
                                // Up to 4 source texel rows per block row;
                                // the mip's bottom edge simply stages fewer
                                // (the kernel edge-replicates via its clamp).
                                for k in 0..bc7::BLOCK as usize {
                                    let y = (r0 + r) * bc7::BLOCK as usize + k;
                                    if y >= mh as usize {
                                        break;
                                    }
                                    let row =
                                        &mip_texels[y * mw as usize..(y + 1) * mw as usize];
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            row.as_flattened().as_ptr(),
                                            ring.ptr.add(r * pitch + k * row_pitch),
                                            mw as usize * 4,
                                        )
                                    };
                                }
                            }
                            TexArm::Rgba => {
                                let y = r0 + r;
                                let row = &mip_texels[y * mw as usize..(y + 1) * mw as usize];
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        row.as_flattened().as_ptr(),
                                        ring.ptr.add(r * pitch),
                                        mw as usize * 4,
                                    )
                                };
                            }
                        }
                    }
                    // The dst y offset and the footprint height are always in
                    // TEXELS. For BC7 both are whole block rows (DstY a
                    // multiple of 4, as the debug layer requires) except the
                    // final band, which runs to the mip's `mh` exactly — the
                    // bottom edge of the subresource, the other form the
                    // layer accepts.
                    let y0 = r0 * row_h;
                    let h_tex = (rows * row_h).min(mh as usize - y0) as u32;
                    let last = mip + 1 == n_mips && r0 + rows == rows_total;
                    let t_enc = matches!(arm, TexArm::GpuEnc).then(std::time::Instant::now);
                    sub.run_list(&mut |l| {
                        match arm {
                            TexArm::GpuEnc => {
                                let enc = gpu_enc.as_ref().expect("GpuEnc arm without encoder");
                                // h_tex is exactly the texel rows staged above.
                                enc.record_encode(
                                    l,
                                    &ring.resource,
                                    mw,
                                    h_tex,
                                    row_pitch as u32,
                                    rows as u32,
                                    gpu_effort,
                                );
                                enc.record_copy_out(l, &dst, mip as u32, y0 as u32, fmt, mw, h_tex);
                            }
                            TexArm::CpuBlk | TexArm::Rgba => {
                                let fp = match arm {
                                    TexArm::CpuBlk => d3d12::footprint_block(fmt, mw, h_tex, 0),
                                    _ => d3d12::footprint(fmt, mw, h_tex, 4, 0),
                                };
                                unsafe {
                                    l.CopyTextureRegion(
                                        &d3d12::loc_subresource_mip(&dst, mip as u32),
                                        0,
                                        y0 as u32,
                                        0,
                                        &d3d12::loc_footprint(&ring.resource, fp),
                                        None,
                                    )
                                };
                            }
                        }
                        if last {
                            unsafe {
                                l.ResourceBarrier(&[transition(
                                    &dst,
                                    D3D12_RESOURCE_STATE_COPY_DEST,
                                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                )])
                            };
                        }
                        Ok(())
                    })?;
                    if let Some(t0) = t_enc {
                        enc_ms += t0.elapsed().as_secs_f64() * 1e3;
                    }
                    r0 += rows;
                }
            }
            // The blocks are on the GPU now — drop them so peak RAM carries at
            // most the BC7 set (~0.25x the RGBA8 texels), never a second copy.
            bc7_blocks[i] = None;
            textures_v.push(dst);
        }
        drop(gpu_enc);

        if !scene.textures.is_empty() {
            let raw = scene.textures.iter().map(|t| t.w as u64 * t.h as u64 * 4).sum::<u64>();
            // Analytic per-texture VRAM: the CPU arm's actual block vecs are
            // exactly `encoded_len` sized, so one formula serves both arms.
            let live = scene
                .textures
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let levels = std::iter::once((t.w, t.h))
                        .chain(t.mips.iter().map(|m| (m.w, m.h)));
                    if compress[i] {
                        levels.map(|(w, h)| bc7::encoded_len(w, h) as u64).sum::<u64>()
                    } else {
                        levels.map(|(w, h)| w as u64 * h as u64 * 4).sum::<u64>()
                    }
                })
                .sum::<u64>();
            let n_bc7 = compress.iter().filter(|&&c| c).count();
            let bc7_note = if n_bc7 > 0 {
                // Mtexel/s is the "is a load-time encode real-time?" number.
                // The GPU arm's ms is the summed wall time of its encode
                // submits (blocking submits, so wall ≈ GPU time).
                let arm = if matches!(bc7_mode, bc7::Bc7Mode::Cpu(_)) { "cpu" } else { "gpu" };
                format!(
                    ", {} BC7 + {} RGBA8, was {} MB | bc7 encode ({}) {:.0} ms ({:.0} Mtexel/s)",
                    n_bc7,
                    scene.textures.len() - n_bc7,
                    raw >> 20,
                    arm,
                    enc_ms,
                    enc_texels as f64 / 1e6 / (enc_ms / 1e3).max(1e-9),
                )
            } else {
                String::new()
            };
            eprintln!(
                "gpu: {} textures uploaded ({} MB{}{})",
                scene.textures.len(),
                live >> 20,
                if scene.any_alpha { ", alpha cutout on" } else { "" },
                bc7_note,
            );
        }

        // --- acceleration-structure sizing ---
        let n_verts = scene.positions.len() as u32;
        let n_tris = scene.indices.len() as u32;
        // OPAQUE drops when ANY conditional-hit feature is armed — the
        // any-hit/candidate machinery is shared (candidate_reject), and the
        // tinted-shadow pass needs candidates to surface in `transmit_q`.
        // Derived from the same three predicates that compile those arms in,
        // so the flag and the shaders cannot disagree (see `non_opaque`).
        let non_opaque = non_opaque(scene);

        // --blas-split: one BLAS per maximal BVH subtree under the cap, each
        // instanced identity into the TLAS. Everything below this block is the
        // single-BLAS build, reached whenever the lever is off — bit-identical
        // to the pre-feature path, which is what keeps every unarmed gate's
        // baseline valid.
        if let Some(cap) = crate::blas_split::max_prims() {
            let t0 = std::time::Instant::now();
            let mut plan = crate::blas_split::plan(bvh, cap);
            // --foliage-sway: pull leaf triangles out of the antichain chunks
            // into per-cell chunks appended at the tail — each cell is one
            // animated TLAS instance (src/foliage.rs). The partition comes
            // ATTACHED on the scene (`Scene::sway`, the one partition the CPU
            // intersector and BVH sweep already consumed — the cross-arm pose
            // contract); the static TLAS below still holds every chunk at
            // identity, the rest pose the headless gates trace.
            let mut sway_split = match &scene.sway {
                Some(sw) => {
                    let sp = crate::foliage::split_plan(&mut plan, sw, cap);
                    match &sp {
                        Some(s) => eprintln!(
                            "foliage-sway: {} leaf tris -> {} cells (grid {:.3}, {} static chunks kept)",
                            plan.packed_tris.len() as u32 - plan.chunk_base[s.first_chunk as usize],
                            s.cells.len(),
                            s.cell,
                            s.first_chunk
                        ),
                        None => eprintln!(
                            "foliage-sway: attached partition maps no plan triangle — sway idle"
                        ),
                    }
                    sp
                }
                None => {
                    if crate::foliage::sweep_armed() {
                        eprintln!(
                            "foliage-sway: no leaf-classified (foliage + cutout) materials in \
                             this scene — sway idle"
                        );
                    }
                    None
                }
            };
            // --dxr-sbt: refine the antichain chunks into per-class
            // SUB-CHUNKS (blas_split::refine_by_class) so each TLAS instance
            // carries its shading class as InstanceContributionToHitGroupIndex
            // (× the 3 ray types — dxr.rs's class-major SBT). INSTANCE-keyed
            // rather than multi-geometry because PrimitiveIndex() restarts
            // per GEOMETRY: per-class geometry descs would have broken
            // `tri_of(inst, prim)` on BOTH pipelines and dragged a
            // GeometryIndex() SM 6.5 floor into the lib_6_3 mode-0 path;
            // this way the remap contract survives with zero shader edits.
            // Runs AFTER split_plan (the sway tail is RELABELED, never split
            // — the cells-parallel-to-tail contract, foliage.rs; the moved
            // tail start is patched back before SwayGpu::new below) and
            // BEFORE the 2^24 ceiling check, so the ceiling validates the
            // GROWN count. Head sub-chunks stay under the cap by
            // construction (a sub-span of an under-cap span); windows/the
            // index stream/FR_SPLIT_AUDIT all derive from the MUTATED plan
            // through the same accessors, so they stay coherent with no
            // further changes (`Windows::tri` is the one rewrite rule).
            let sbt_class = if crate::gpu::dxr::dxr_sbt_mode() > 0 {
                let mat_class =
                    crate::shadeclass::classify_materials(&scene.materials, &scene.textures);
                // The live-scene soundness audit — the same must-fire
                // shadeclass::self_test runs synthetically. A taxonomy bug
                // here would otherwise surface as a wrong image three
                // suites later; fail the upload instead.
                crate::shadeclass::verify_strips(&scene.materials, &mat_class)
                    .map_err(|e| format!("--dxr-sbt: {e}"))?;
                let n_before = plan.chunks();
                let r = crate::blas_split::refine_by_class(
                    &mut plan,
                    sway_split.as_ref().map(|s| s.first_chunk),
                    crate::shadeclass::CK_UBER,
                    crate::shadeclass::N_CLASSES,
                    |t| mat_class[scene.tri_mat[t as usize] as usize],
                );
                if let (Some(sp), Some(ft)) = (sway_split.as_mut(), r.first_tail) {
                    sp.first_chunk = ft;
                }
                let histo = crate::shadeclass::histogram(&r.chunk_class);
                let mut parts = String::new();
                for (k, n) in histo.iter().enumerate() {
                    if *n > 0 {
                        parts.push_str(&format!(" {}:{n}", crate::shadeclass::NAMES[k]));
                    }
                }
                eprintln!(
                    "dxr-sbt: {n_before} chunks -> {} class sub-chunks |{parts}",
                    plan.chunks()
                );
                Some(SbtClassInfo { chunk_class: r.chunk_class, histo })
            } else {
                None
            };
            // The chunk index rides InstanceID (24 bits). No real scene comes
            // near this at a 64k cap; a tiny cap on a huge scene could, and a
            // silent wrap would remap every triangle in the overflowing chunks.
            if plan.chunks() > (1 << 24) {
                return Err(format!(
                    "--blas-split {cap}: {} chunks exceeds the 2^24 InstanceID ceiling \
                     (raise the cap)",
                    plan.chunks()
                ));
            }
            // A triangle-free scene plans zero chunks, and every size below
            // would be zero (a 0-byte committed_buffer is invalid, a 0-instance
            // TLAS pointless). Unreachable today — every scene carries at least
            // the ground quad — so this is the assert, not a fallback.
            if plan.chunks() == 0 {
                return Err("--blas-split: the scene has no triangles to chunk".into());
            }
            // PER-CHUNK VERTEX WINDOWING (2026-08-01 — the bistro-dusk shard
            // fix; the planner and the defect write-up live at
            // blas_split::plan_windows / SPLIT_INDEX_CEILING, pinned by
            // blas_split::self_test). Every chunk's BLAS indices are kept
            // under the ceiling — REBASE slides the vertex window to the
            // chunk's min id (free, the common case); a chunk whose id RANGE
            // itself clears the ceiling GATHERS its ≤ 3·cap used vertices
            // into a small transient side buffer (tile seams / the world's
            // cross-island chunks — 9 chunks / 1.5 MB on tiled san-miguel, 1
            // chunk / 201 KB on THE WORLD). build_split_blas windows each
            // geometry desc to match. FR_SPLIT_NOREBASE=1 restores absolute
            // index values — the repro arm; FR_SPLIT_AUDIT=1 memcmps all
            // three streamed buffers against the CPU plan.
            //
            // The stream stays transient-free (no 12 B/tri CPU copy): the map
            // closure walks a chunk cursor instead, sound because
            // stream_buffer maps elements STRICTLY IN ORDER (the ring loop is
            // sequential and each ring chunk zips in order).
            let no_rebase = std::env::var("FR_SPLIT_NOREBASE").map_or(false, |v| v == "1");
            if no_rebase {
                eprintln!("blas-split: FR_SPLIT_NOREBASE=1 — absolute BLAS index values");
            }
            let wins = crate::blas_split::plan_windows(
                &plan,
                &scene.indices,
                |v| {
                    let p = scene.positions[v as usize];
                    [p.x, p.y, p.z]
                },
                no_rebase,
            );
            if wins.gathered() > 0 {
                eprintln!(
                    "blas-split: {} chunk(s) vertex-gathered ({} KB side buffer) — id range \
                     over the {} ceiling (the RDNA4 index-value workaround)",
                    wins.gathered(),
                    (wins.aux.len() * 12) >> 10,
                    crate::blas_split::SPLIT_INDEX_CEILING,
                );
            }
            // The gathered side buffer, like the reordered index stream below,
            // feeds only the builds (a built AS is self-contained).
            let aux_positions = if wins.aux.is_empty() {
                None
            } else {
                Some(stream_buffer(device, sub, &ring, &wins.aux, |v| *v, srv)?)
            };
            // The reordered index stream, mapped straight out of the plan — no
            // transient CPU copy of the whole buffer (12 B/tri would be 1 GB on
            // a tiled scene). Positions are SHARED with the single-BLAS path:
            // chunks reference the one vertex buffer, so nothing duplicates.
            let cursor = std::cell::Cell::new((0usize, 0usize)); // (element idx, chunk)
            let blas_indices = stream_buffer(
                device,
                sub,
                &ring,
                &plan.packed_tris,
                |&t| {
                    let (idx, mut c) = cursor.get();
                    while idx >= plan.chunk_base[c + 1] as usize {
                        c += 1;
                    }
                    cursor.set((idx + 1, c));
                    wins.tri(c, scene.indices[t as usize])
                },
                srv,
            )?;
            let blas_tri_b = stream_buffer(device, sub, &ring, &plan.packed_tris, |t| *t, srv)?;
            let chunk_base_b = stream_buffer(device, sub, &ring, &plan.chunk_base, |b| *b, srv)?;
            // FR_SPLIT_AUDIT=1 — one-shot diagnostic (the bistro-dusk shard
            // hunt): read the two remap buffers straight back and memcmp
            // against the CPU plan, so "the GPU sees wrong remap DATA" can be
            // ruled in/out without touching a shader. Loud either way.
            if std::env::var("FR_SPLIT_AUDIT").map_or(false, |v| v == "1") {
                audit_split_streams(device, sub, &blas_tri_b, &plan.packed_tris, "blas_tri")?;
                audit_split_streams(device, sub, &chunk_base_b, &plan.chunk_base, "chunk_base")?;
                // The reordered index stream too — the buffer the BLAS builds
                // actually consume (transiently ~413 MB of expected values;
                // diagnostic-only). Expected carries the same per-chunk rebase
                // the stream applied.
                let expected: Vec<u32> = (0..plan.chunks())
                    .flat_map(|i| {
                        plan.tris(i)
                            .iter()
                            .flat_map(|&t| wins.tri(i, scene.indices[t as usize]))
                            .collect::<Vec<u32>>()
                    })
                    .collect();
                audit_split_streams(device, sub, &blas_indices, &expected, "blas_indices")?;
            }

            let split = build_split_blas(
                device,
                &device5,
                sub,
                &positions_b,
                &blas_indices,
                &plan,
                &wins.win,
                aux_positions.as_ref(),
                n_verts,
                non_opaque,
                scene.any_transmissive,
                sbt_class.as_ref().map(|c| c.chunk_class.as_slice()),
            )?;
            // The gathered side buffer fed only the builds, exactly like the
            // reordered index stream below.
            drop(aux_positions);
            // The animated-TLAS ring (--foliage-sway with leaf cells): rest
            // template + FRAMES_IN_FLIGHT instance/TLAS slots + one scratch,
            // all beside the static TLAS — never replacing it.
            let sway = match sway_split {
                Some(sp) => Some(SwayGpu::new(
                    device,
                    sp,
                    &split.instances,
                    split.tlas_size,
                    split.tlas_scratch,
                )?),
                None => None,
            };
            let (blas, tlas, report) = (split.arena, split.tlas, split.report);
            // A built AS is self-contained: the reordered index stream existed
            // only to feed the builds (which `sub` ran to completion), so it
            // goes back now rather than resting for the session. `indices_b`
            // (original order) stays — that one is what SHADING reads.
            drop(blas_indices);
            let (lo, mean, hi) = plan.stats();
            eprintln!(
                "gpu scene: streams {} MB | {} | blas-split {} chunks (prims min {lo} mean {mean:.0} max {hi}, cap {cap}) in {:.0} ms{}",
                (scene_stream_bytes(scene) + plan.packed_tris.len() * 16) >> 20,
                report,
                plan.chunks(),
                t0.elapsed().as_secs_f64() * 1e3,
                match adapter::vram_info(device) {
                    Some((usage, budget)) =>
                        format!(" | vram {} / {} MB", usage >> 20, budget >> 20),
                    None => String::new(),
                }
            );
            drop(ring);
            return Ok(Self {
                positions: positions_b,
                normals: normals_b,
                indices: indices_b,
                tri_mat,
                materials: materials_b,
                blas,
                tlas,
                sway,
                sbt_class,
                blas_tri: blas_tri_b,
                chunk_base: chunk_base_b,
                n_packed: plan.packed_tris.len() as u32,
                n_chunks: plan.chunks() as u32,
                texcoords: texcoords_b,
                mat_cutout: mat_cutout_b,
                mat_height: mat_height_b,
                mat_shadow: mat_shadow_b,
                textures: textures_v,
                dmv_dummy: committed_buffer(device, 16, D3D12_RESOURCE_FLAG_NONE, srv)?,
                n_verts,
                n_tris,
                n_mats: scene.materials.len() as u32,
            });
        }

        let geom = geometry_desc(
            &positions_b,
            &indices_b,
            n_verts,
            n_tris,
            non_opaque,
            scene.any_transmissive,
        );
        // ALLOW_COMPACTION: the build lands in a worst-case-sized buffer,
        // then a compact copy (~40-50% smaller in practice) replaces it and
        // the original drops before this function returns — the compacted
        // size is what buys DXR headroom on 100M-tri scenes.
        let blas_flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
        let blas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: blas_flags,
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
        let tlas =
            committed_buffer(device, tlas_info.ResultDataMaxSizeInBytes, uaf, as_state)?;
        // The worst-case-sized build target, scratch, and the TLAS instance
        // desc all live only to the end of this function — peak commit is
        // steady-state (compacted BLAS) + one worst-case BLAS + scratch.
        let blas_full =
            committed_buffer(device, blas_info.ResultDataMaxSizeInBytes, uaf, as_state)?;
        let scratch = committed_buffer(
            device,
            blas_info.ScratchDataSizeInBytes.max(tlas_info.ScratchDataSizeInBytes),
            uaf,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;

        // Submit 1: BLAS build, emitting the compacted size (a u64 written to
        // a tiny UAV buffer, copied to readback in the same list — the
        // blocking submit doubles as the fence).
        let csize_buf =
            committed_buffer(device, 8, uaf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
        let csize_rb = d3d12::ReadbackBuffer::new(device, 8)?;
        sub.run_list(&mut |list| {
            let list4: ID3D12GraphicsCommandList4 =
                list.cast().map_err(|e| format!("ID3D12GraphicsCommandList4: {e}"))?;
            let blas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: unsafe { blas_full.GetGPUVirtualAddress() },
                Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    Flags: blas_flags,
                    NumDescs: 1,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: &geom,
                    },
                },
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: unsafe { scratch.GetGPUVirtualAddress() },
            };
            let postbuild = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: unsafe { csize_buf.GetGPUVirtualAddress() },
                InfoType:
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
            };
            unsafe { list4.BuildRaytracingAccelerationStructure(&blas_desc, Some(&[postbuild])) };
            unsafe {
                list.ResourceBarrier(&[
                    uav_barrier(None),
                    transition(
                        &csize_buf,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                ])
            };
            unsafe { list.CopyBufferRegion(&csize_rb.resource, 0, &csize_buf, 0, 8) };
            Ok(())
        })?;
        let compacted_size = {
            let mut ptr = std::ptr::null_mut();
            unsafe { csize_rb.resource.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("compacted-size Map: {e}"))?;
            let v = unsafe { (ptr as *const u64).read_unaligned() };
            unsafe { csize_rb.resource.Unmap(0, None) };
            v
        };

        // Submit 2: compact copy into an exact-size buffer, then the TLAS
        // build against the COMPACTED BLAS. A degenerate reported size keeps
        // the full build (never wrong, just bigger).
        let use_compact =
            compacted_size > 0 && compacted_size < blas_info.ResultDataMaxSizeInBytes;
        let blas = if use_compact {
            committed_buffer(device, compacted_size, uaf, as_state)?
        } else {
            blas_full.clone()
        };

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

        sub.run_list(&mut |list| {
            let list4: ID3D12GraphicsCommandList4 =
                list.cast().map_err(|e| format!("ID3D12GraphicsCommandList4: {e}"))?;
            if use_compact {
                unsafe {
                    list4.CopyRaytracingAccelerationStructure(
                        blas.GetGPUVirtualAddress(),
                        blas_full.GetGPUVirtualAddress(),
                        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
                    )
                };
                unsafe { list.ResourceBarrier(&[uav_barrier(None)]) };
            }
            let tlas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: unsafe { tlas.GetGPUVirtualAddress() },
                Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                    Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                    NumDescs: 1,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        InstanceDescs: instance_va,
                    },
                },
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: unsafe { scratch.GetGPUVirtualAddress() },
            };
            unsafe { list4.BuildRaytracingAccelerationStructure(&tlas_desc, None) };
            unsafe { list.ResourceBarrier(&[uav_barrier(None)]) };
            Ok(())
        })?;
        drop(instance);
        drop(scratch);
        drop(blas_full);
        drop(ring);

        eprintln!(
            "gpu scene: streams {} MB | blas {} MB{} | transient scratch {} MB (freed){}",
            scene_stream_bytes(scene) >> 20,
            if use_compact { compacted_size } else { blas_info.ResultDataMaxSizeInBytes } >> 20,
            if use_compact {
                format!(" (compacted from {})", blas_info.ResultDataMaxSizeInBytes >> 20)
            } else {
                String::new()
            },
            blas_info.ScratchDataSizeInBytes.max(tlas_info.ScratchDataSizeInBytes) >> 20,
            match adapter::vram_info(device) {
                Some((usage, budget)) =>
                    format!(" | vram {} / {} MB", usage >> 20, budget >> 20),
                None => String::new(),
            }
        );

        Ok(Self {
            positions: positions_b,
            normals: normals_b,
            indices: indices_b,
            tri_mat,
            materials: materials_b,
            blas,
            tlas,
            // Sway needs per-cell instances, which only the split path has —
            // `--no-blas-split --foliage-sway` already printed its note in
            // main's lever block.
            sway: None,
            // --dxr-sbt likewise needs per-chunk instances: DxrGpu::new sees
            // None and degrades the lever loudly (the sway note precedent).
            sbt_class: None,
            // Unarmed: no remap exists and the kernels compile without
            // BLAS_SPLIT, so these are never read — 4-byte dummies keep the
            // descriptor table's shape uniform across both paths.
            blas_tri: committed_buffer(device, 4, D3D12_RESOURCE_FLAG_NONE, srv)?,
            chunk_base: committed_buffer(device, 4, D3D12_RESOURCE_FLAG_NONE, srv)?,
            // ZERO, not 1: these are SRV ELEMENT counts, and the table's
            // `chunk_base` view is sized `n_chunks + 1` (the plan's sentinel).
            // A 1 here would describe 8 bytes of a 4-byte dummy — an
            // out-of-range buffer SRV, which is exactly the kind of invalid
            // view that takes the device out at heap-write time.
            n_packed: 0,
            n_chunks: 0,
            texcoords: texcoords_b,
            mat_cutout: mat_cutout_b,
            mat_height: mat_height_b,
            mat_shadow: mat_shadow_b,
            textures: textures_v,
            dmv_dummy: committed_buffer(device, 16, D3D12_RESOURCE_FLAG_NONE, srv)?,
            n_verts,
            n_tris,
            n_mats: scene.materials.len() as u32,
        })
    }

    /// Write the RP_SCENE_TEX table's descriptors into `heap` at slots
    /// `base..`: 10 buffer SRVs (texcoords, indices, tri_mat, mat_cutout,
    /// positions, mat_height, mat_shadow, blas_tri, chunk_base, sway_dmv —
    /// t0..t9 space1) then one Texture2D SRV per scene texture (t10..
    /// space1). The heap must be sized `base + TEX_TABLE_BUFS +
    /// textures.len()`.
    pub fn write_scene_descriptors(
        &self,
        device: &ID3D12Device,
        heap: &ID3D12DescriptorHeap,
        base: u32,
    ) {
        let inc = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        let start = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let slot = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + ((base + i) as usize * inc as usize),
        };
        let buf_srv = |res: &ID3D12Resource, stride: u32, elems: u32, at: u32| {
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: 0,
                        NumElements: elems.max(1),
                        StructureByteStride: stride,
                        Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                    },
                },
            };
            unsafe { device.CreateShaderResourceView(res, Some(&desc), slot(at)) };
        };
        buf_srv(&self.texcoords, 8, self.n_verts, 0);
        buf_srv(&self.indices, 4, self.n_tris * 3, 1);
        buf_srv(&self.tri_mat, 4, self.n_tris, 2);
        buf_srv(&self.mat_cutout, 4, self.n_mats, 3);
        // Second descriptor over positions (the indices/tri_mat pattern) +
        // the per-material relief map — the march's intersector-scope inputs.
        buf_srv(&self.positions, 12, self.n_verts, 4);
        buf_srv(&self.mat_height, 8, self.n_mats, 5);
        buf_srv(&self.mat_shadow, 16, self.n_mats, 6);
        // --blas-split's (InstanceID, PrimitiveIndex) -> tri remap. Unarmed
        // sessions bind the 4-byte dummies: the kernels compile without
        // BLAS_SPLIT and never read them, but the table's shape (and so
        // TEX_TABLE_BUFS, and so texs[]'s base register) stays session-
        // independent. Both element counts are ZERO-BASED on purpose and lean
        // on `elems.max(1)` above: a view must describe no more than the 4-byte
        // dummy holds, and an over-long one takes the device out at heap-write
        // time (n_chunks + 1 with n_chunks = 1 did exactly that once).
        buf_srv(&self.blas_tri, 4, self.n_packed, 7);
        buf_srv(&self.chunk_base, 4, self.n_chunks + 1, 8);
        // Foliage-sway MV deltas: the whole FRAMES_IN_FLIGHT × n_inst float4
        // ring (the frame's slot base rides the CB's sway_mv_base). Unarmed
        // scenes bind the 16-byte dummy — the kernels compile without SWAY_MV
        // and never read it, the blas_tri discipline.
        let (dmv_res, dmv_elems) = match &self.sway {
            Some(sw) => (&sw.dmv_ring.resource, sw.dmv_elems()),
            None => (&self.dmv_dummy, 1),
        };
        buf_srv(dmv_res, 16, dmv_elems, 9);
        for (i, tex) in self.textures.iter().enumerate() {
            let tex_desc = unsafe { tex.GetDesc() };
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                // The resource's own creation format: _SRGB for color
                // textures, _UNORM for linear-data maps.
                Format: tex_desc.Format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        // The whole CPU-generated chain (1 on chainless
                        // textures — the 1×1 fallbacks and --no-mips).
                        MipLevels: tex_desc.MipLevels as u32,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            };
            unsafe {
                device.CreateShaderResourceView(
                    tex,
                    Some(&desc),
                    slot(TEX_TABLE_BUFS + i as u32),
                )
            };
        }
    }
}


/// The sway-MV arming pair: `Some((t_cur, t_prev))` iff this frame renders an
/// animated pose AND holds a paired prev pose to reproject into — sway clock
/// present, prev sway clock present and BIT-different (a frozen still /
/// pinned gate has du = 0 structurally), prev camera present (an MV is only
/// defined against one). ONE predicate for the CB flag and the dmv-ring fill
/// (both tracers), so the two cannot disagree.
pub(crate) fn sway_mv_pair(p: &FrameParams) -> Option<(f32, f32)> {
    let (tc, tp) = (p.sway_time?, p.sway_prev_time?);
    if tc.to_bits() == tp.to_bits() || p.prev_cam.is_none() {
        return None;
    }
    Some((tc, tp))
}

/// The scene AABB the slab-space cloud-shadow grid spans: the content box
/// unioned with the ground quad's first few vertices (the ground is what a low
/// sun's shadow footprint actually covers). Shared by TraceGpu/DxrGpu so both
/// derive the same grid.
pub(crate) fn scene_shadow_aabb(scene: &Scene) -> ([f32; 3], [f32; 3]) {
    let (mut amn, mut amx) = (scene.content_min, scene.content_max);
    for p in scene.positions.iter().take(6) {
        amn = amn.min(*p);
        amx = amx.max(*p);
    }
    ([amn.x, amn.y, amn.z], [amx.x, amx.y, amx.z])
}

/// D3D12 requires every acceleration structure to start on a 256-byte boundary
/// (`D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT`), which is what
/// lets all chunk BLASes share ONE committed arena instead of one resource
/// each.
const AS_ALIGN: u64 = 256;

fn align_as(x: u64) -> u64 {
    (x + AS_ALIGN - 1) & !(AS_ALIGN - 1)
}

/// `--foliage-sway`'s GPU half (src/foliage.rs owns the math, the design doc
/// is docs/design/animated-foliage.md): an animated-TLAS ring BESIDE the
/// static TLAS. Per animated frame the rest-pose instance template is copied
/// into this ring's slot, the sway rows' translations are patched in
/// (translation-only — normals/UVs/barycentrics untouched, hit points are
/// `o + t·d` on both GPU paths), and a full TLAS rebuild is recorded on the
/// frame's own list (`PREFER_FAST_TRACE` at a few hundred to ~2k instances —
/// three orders under the paper's measured 2.8M-instance/9.66 ms point).
/// The static TLAS is never touched: it is the rest pose every headless
/// gate/bench traces (sway_time None). Since v0.2 BOTH ray pipelines consume
/// the ring (wavefront `TraceGpu` and DXR `DxrGpu` — each stashes the
/// frame's clock and binds `tlas_va(slot)` when Some); frustum claims /
/// temporal ring / structure replay stay sound because the software BVH's
/// leaf boxes are SWEPT by the displacement bound at build
/// (`bvh::grow_sway_sweep`), not because the pose is static. Ring depth =
/// FRAMES_IN_FLIGHT (the frame_cb contract); scratch is ONE buffer —
/// successive builds are serialized by each frame's trailing queue-level
/// UAV barrier.
pub struct SwayGpu {
    /// Chunks (== instances) `first_chunk..` are the sway cells.
    first_chunk: u32,
    cells: Vec<crate::foliage::SwayCell>,
    /// Per-run PARTITION cell index — the flutter hash key (the v0.2 re-key:
    /// runs of one overflowed cell translate identically, bit-equal to the
    /// CPU bake of that cell). Never read on THIS side — the per-frame wind
    /// bake keys off `cells`, whose entries already carry the re-key — kept
    /// as the run → partition-cell map of the split plan.
    #[allow(dead_code)]
    cell_of: Vec<u32>,
    /// Rest-pose instance descs — identity transforms, compacted-BLAS VAs
    /// baked, InstanceID = chunk index; the per-frame patch source.
    tpl: Vec<D3D12_RAYTRACING_INSTANCE_DESC>,
    /// FRAMES_IN_FLIGHT slots of `tpl.len()` instance descs, persistently
    /// mapped (the frame_cb ring shape).
    inst_ring: d3d12::UploadBuffer,
    /// Sway-MV deltas: FRAMES_IN_FLIGHT slots × tpl.len() float4 shear-row
    /// deltas (`foliage::shear_rows(u_prev − u_cur, a, b)` per sway chunk),
    /// persistently mapped (the inst_ring shape), zero-filled once at init so
    /// static chunks' rows stay the exact-identity zeros forever. Read by
    /// `gbuf_write_hit`'s SWAY_MV arm through the space1 t9 SRV at
    /// `sway_mv_base.x + InstanceID`.
    dmv_ring: d3d12::UploadBuffer,
    /// One animated TLAS per in-flight slot.
    tlas: Vec<ID3D12Resource>,
    scratch: ID3D12Resource,
    /// The clock each slot's TLAS was last built at (NAN = never): a frozen
    /// clock — a converging still — records NOTHING after the first two
    /// frames, so accumulation integrates one pose at zero rebuild cost.
    baked: Vec<std::cell::Cell<f32>>,
}

impl SwayGpu {
    fn new(
        device: &ID3D12Device,
        sp: crate::foliage::SwaySplit,
        tpl: &[D3D12_RAYTRACING_INSTANCE_DESC],
        tlas_size: u64,
        tlas_scratch: u64,
    ) -> Result<Self> {
        let uaf = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let as_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        let n = tpl.len();
        let sz = std::mem::size_of::<D3D12_RAYTRACING_INSTANCE_DESC>();
        let inst_ring = d3d12::UploadBuffer::new(device, d3d12::FRAMES_IN_FLIGHT * n * sz)?;
        let dmv_ring = d3d12::UploadBuffer::new(device, d3d12::FRAMES_IN_FLIGHT * n * 16)?;
        // Zero-fill EXPLICITLY (never lean on fresh-commit zeroing): static
        // chunks' rows are the identity by being zero, and `write_mv_rows`
        // only ever rewrites the sway chunks' tail.
        unsafe { std::ptr::write_bytes(dmv_ring.ptr, 0, d3d12::FRAMES_IN_FLIGHT * n * 16) };
        let mut tlas = Vec::with_capacity(d3d12::FRAMES_IN_FLIGHT);
        for _ in 0..d3d12::FRAMES_IN_FLIGHT {
            tlas.push(committed_buffer(device, tlas_size, uaf, as_state)?);
        }
        let scratch = committed_buffer(
            device,
            tlas_scratch,
            uaf,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        eprintln!(
            "foliage-sway: animated-TLAS ring armed — {} instances ({} sway cells), {} KB x {} \
             slots + {} KB scratch",
            n,
            sp.cells.len(),
            tlas_size >> 10,
            d3d12::FRAMES_IN_FLIGHT,
            tlas_scratch >> 10,
        );
        Ok(SwayGpu {
            first_chunk: sp.first_chunk,
            cells: sp.cells,
            cell_of: sp.cell_of,
            tpl: tpl.to_vec(),
            inst_ring,
            dmv_ring,
            tlas,
            scratch,
            baked: (0..d3d12::FRAMES_IN_FLIGHT).map(|_| std::cell::Cell::new(f32::NAN)).collect(),
        })
    }

    /// The TLAS a frame binding slot `slot` should trace — valid only after
    /// `record_rebuild` ran for that slot at least once this session.
    pub fn tlas_va(&self, slot: usize) -> u64 {
        unsafe { self.tlas[slot].GetGPUVirtualAddress() }
    }

    /// Forget every slot's baked clock — the replay-invalidation class: a
    /// frame recorded-but-aborted (present error) marked its slot baked at
    /// RECORD time, but the build never executed, and the skip fast-path
    /// would then bind a TLAS that was never written. Call from every GPU
    /// (wavefront AND DXR) present-error arm.
    pub fn invalidate(&self) {
        for b in &self.baked {
            b.set(f32::NAN);
        }
    }

    /// Bake this clock's winds, patch the slot's instance descs with each
    /// run's rooted-shear rows (foliage v0.4 — the affine `p' = p + u·(a +
    /// b·p.y)` rides the four non-identity slots of the row-major 3×4;
    /// row 1 stays identity from the `tpl` copy because `u.y ≡ 0`), and
    /// record the TLAS rebuild — a bit-equal clock whose slot already holds
    /// this pose records nothing (the converging-still fast path). Call
    /// BEFORE any ray dispatch that binds `tlas_va(slot)`. DXR transforms
    /// the ray into object space without renormalizing, so hit t and
    /// PrimitiveIndex are unaffected — the same argument as the CPU's
    /// `shear_ray`.
    pub fn record_rebuild(
        &self,
        list: &ID3D12GraphicsCommandList,
        slot: usize,
        time: f32,
    ) -> Result<()> {
        if self.baked[slot].get().to_bits() == time.to_bits() {
            return Ok(());
        }
        let list4: ID3D12GraphicsCommandList4 =
            list.cast().map_err(|e| format!("ID3D12GraphicsCommandList4: {e}"))?;
        let n = self.tpl.len();
        let sz = std::mem::size_of::<D3D12_RAYTRACING_INSTANCE_DESC>();
        let d = crate::foliage::winds(&self.cells, time);
        // Slot-fenced by begin_frame (the frame_cb contract), so the CPU
        // write cannot race the GPU read from 2 frames ago.
        unsafe {
            let dst =
                self.inst_ring.ptr.add(slot * n * sz) as *mut D3D12_RAYTRACING_INSTANCE_DESC;
            std::ptr::copy_nonoverlapping(self.tpl.as_ptr(), dst, n);
            for (j, u) in d.iter().enumerate() {
                // ONE shared derivation with the CPU↔GPU pose gate
                // (foliage::shear_rows) — the matrix cannot fork from the
                // self_test's oracle. Runs re-key onto partition cells, so
                // runs of one overflowed cell get bit-identical rows.
                let cl = &self.cells[j];
                let [uxb, uxa, uzb, uza] = crate::foliage::shear_rows(*u, cl.a, cl.b);
                let inst = dst.add(self.first_chunk as usize + j);
                (*inst).Transform[1] = uxb;
                (*inst).Transform[3] = uxa;
                (*inst).Transform[9] = uzb;
                (*inst).Transform[11] = uza;
            }
        }
        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: unsafe { self.tlas[slot].GetGPUVirtualAddress() },
            Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                NumDescs: n as u32,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: unsafe { self.inst_ring.resource.GetGPUVirtualAddress() }
                        + (slot * n * sz) as u64,
                },
            },
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: unsafe { self.scratch.GetGPUVirtualAddress() },
        };
        unsafe { list4.BuildRaytracingAccelerationStructure(&desc, None) };
        // One queue-level UAV barrier carries BOTH orderings: this build
        // completes before the frame's rays read the TLAS, and before the
        // next frame's build rewrites the shared scratch.
        unsafe { list.ResourceBarrier(&[uav_barrier(None)]) };
        self.baked[slot].set(time);
        Ok(())
    }

    /// The dmv ring's SRV element count (whole ring — the frame picks its
    /// slot via the CB's sway_mv_base).
    pub fn dmv_elems(&self) -> u32 {
        (d3d12::FRAMES_IN_FLIGHT * self.tpl.len()) as u32
    }

    /// Instances per ring slot — the CB `sway_mv_base` multiplier
    /// (`slot · n_inst`), pub because DxrGpu computes its base too.
    pub fn n_inst(&self) -> u32 {
        self.tpl.len() as u32
    }

    /// This frame's per-chunk prev−cur shear rows into `slot`'s ring section
    /// — the sway-MV delta table (`foliage::mv_rows`' per-run form: runs of
    /// one cap-overflow cell get bit-identical rows through the same pure
    /// `wind`, the record_rebuild re-key contract). Bit-equal clocks write
    /// nothing and return false — the caller must then NOT arm FLAG_SWAY_MV
    /// (du = 0 structurally). Slot-fenced like inst_ring; static chunks'
    /// zeros persist from init, only the sway tail is rewritten. Keyless on
    /// purpose (recomputed whenever armed): ~2×cells closed-form wind evals,
    /// and a skip cache would be a second `baked` to invalidate.
    pub fn write_mv_rows(&self, slot: usize, t_cur: f32, t_prev: f32) -> bool {
        if t_cur.to_bits() == t_prev.to_bits() {
            return false;
        }
        let up = crate::foliage::winds(&self.cells, t_prev);
        let uc = crate::foliage::winds(&self.cells, t_cur);
        let n = self.tpl.len();
        unsafe {
            let base = self.dmv_ring.ptr.add(slot * n * 16) as *mut [f32; 4];
            for (j, (p, c)) in up.iter().zip(uc.iter()).enumerate() {
                let cl = &self.cells[j];
                *base.add(self.first_chunk as usize + j) =
                    crate::foliage::shear_rows(*p - *c, cl.a, cl.b);
            }
        }
        true
    }
}

/// `build_split_blas`'s product. `instances` is the rest-pose instance-desc
/// array (identity transforms, compacted-BLAS VAs baked, InstanceID = chunk
/// index) — the static TLAS was built from exactly these bytes, and the
/// foliage-sway ring patches a copy per frame; `tlas_size`/`tlas_scratch` are
/// the prebuild numbers a per-frame rebuild allocates by (same NumDescs).
struct SplitBuild {
    arena: ID3D12Resource,
    tlas: ID3D12Resource,
    report: String,
    instances: Vec<D3D12_RAYTRACING_INSTANCE_DESC>,
    tlas_size: u64,
    tlas_scratch: u64,
}

/// `--blas-split`: build one BLAS per plan chunk and a TLAS instancing them all
/// identity. Returns the arena holding every compacted chunk BLAS, the TLAS,
/// the `gpu scene:` line's size report, and the rest-pose instance template
/// (see `SplitBuild`).
///
/// Shape mirrors the single-BLAS build one level up — build worst-case with
/// ALLOW_COMPACTION emitting postbuild sizes, read them back, compact into an
/// exact-size arena, then build the TLAS against the compacted addresses — and
/// keeps its affordances for the same reasons: compaction is 40-50% of BLAS
/// memory, and at a few hundred chunks the extra queries are noise. Builds run
/// SERIALLY through one max-sized scratch buffer with a UAV barrier between
/// them; the alternative (a scratch slot per chunk) trades hundreds of MB of
/// transient VRAM for parallelism the build is not bound by.
#[allow(clippy::too_many_arguments)]
fn build_split_blas(
    device: &ID3D12Device,
    device5: &ID3D12Device5,
    sub: &mut dyn d3d12::Submit,
    positions: &ID3D12Resource,
    blas_indices: &ID3D12Resource,
    plan: &crate::blas_split::BlasPlan,
    // Per-chunk vertex windowing (the RDNA4 index-value workaround — see the
    // rebase comment at the blas_indices stream): the chunk's indices were
    // rebased/gathered per this, so its geometry desc must window the same
    // way. All Rebase(0) under FR_SPLIT_NOREBASE.
    chunk_win: &[ChunkWindow],
    aux_positions: Option<&ID3D12Resource>,
    n_verts: u32,
    non_opaque: bool,
    any_transmissive: bool,
    // --dxr-sbt: per-chunk shading class, baked into each instance's
    // InstanceContributionToHitGroupIndex as `class * 3` (the class-major
    // [shade, hit, occlude] SBT stride in dxr.rs — keep in lockstep). None
    // (the lever off) leaves `_bitfield2 = 0`, the byte-identical off-state
    // instance template.
    chunk_class: Option<&[u8]>,
) -> Result<SplitBuild> {
    let n = plan.chunks();
    let uaf = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    let as_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    let flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;

    // Per-chunk geometry descs: the SHARED vertex buffer, this chunk's slice of
    // the reordered index stream. Held for the whole function — the build descs
    // point at them.
    let index_va = unsafe { blas_indices.GetGPUVirtualAddress() };
    let pos_va = unsafe { positions.GetGPUVirtualAddress() };
    let aux_va = aux_positions.map(|r| unsafe { r.GetGPUVirtualAddress() });
    let geoms: Vec<D3D12_RAYTRACING_GEOMETRY_DESC> = (0..n)
        .map(|i| {
            let (vcount, vstart) = match chunk_win[i] {
                // The window opens at the rebase base (12 B/vertex; 4-byte
                // alignment holds — the format's component size) …
                ChunkWindow::Rebase(base) => (n_verts - base, pos_va + base as u64 * 12),
                // … or at the chunk's slice of the gathered side buffer.
                ChunkWindow::Gather { base, count } => (
                    count,
                    aux_va.expect("gather chunks imply an aux buffer") + base as u64 * 12,
                ),
            };
            let mut g = geometry_desc(
                positions,
                blas_indices,
                vcount,
                plan.prims(i),
                non_opaque,
                any_transmissive,
            );
            g.Anonymous.Triangles.IndexBuffer =
                index_va + plan.chunk_base[i] as u64 * 12;
            g.Anonymous.Triangles.VertexBuffer.StartAddress = vstart;
            g
        })
        .collect();

    // Sizing pass: worst-case offsets into the build arena + the scratch high
    // water mark. One prebuild query per chunk — a few hundred calls.
    let mut build_off = Vec::with_capacity(n);
    let mut total_build = 0u64;
    let mut scratch_max = 0u64;
    for g in &geoms {
        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: flags,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: g,
            },
        };
        let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut info) };
        build_off.push(total_build);
        total_build = align_as(total_build + info.ResultDataMaxSizeInBytes);
        scratch_max = scratch_max.max(info.ScratchDataSizeInBytes);
    }

    let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
        NumDescs: n as u32,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            InstanceDescs: 0,
        },
    };
    let mut tlas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut tlas_info) };

    // VRAM pre-flight: the transient peak is the worst-case arena + scratch +
    // the TLAS, and the caller has already committed the scene streams. Over
    // budget is a LOUD failure here rather than a WDDM demotion that silently
    // renders at a tenth the speed.
    //
    // It CANNOT degrade to the single-BLAS build: the `--blas-split` lever is
    // session-global and both tracers bake `blas_defs()` into their
    // kernels/RTPSO, so a degraded ONE-BLAS core would have any armed shader
    // (already compiled, or compiled later against the shared Rc<SceneGpu>)
    // reading `blas_tri[chunk_base[inst] + prim]` through 4-byte dummies —
    // every hit remapped to garbage. (Falling back WOULD be possible by
    // binding an identity remap instead of the dummies, at 4 B/tri and one
    // indirection; deliberately not done — an untested path reachable only by
    // exhausting VRAM is exactly how the dummy-SRV device removal got in.) So
    // the core upload fails, and the session's normal shape takes over: a
    // loud line and the CPU renderer.
    if let Some((usage, budget)) = adapter::vram_info(device) {
        let want = total_build + scratch_max + tlas_info.ResultDataMaxSizeInBytes;
        if usage + want > budget {
            return Err(format!(
                "--blas-split: {} chunks need {} MB of acceleration structure on top of \
                 {} MB already committed, over the {} MB budget — free VRAM first: \
                 --lock-res <scale> (smaller render-res buffers), a smaller scene, or \
                 re-enable BC7 if --no-bc7 dropped it (opaque textures 8 bpp vs 32). \
                 Do NOT drop --blas-split on Intel — the \
                 single-BLAS build's scratch ask removes the device on large scenes. \
                 (The GPU tracers cannot start, so the session falls back to the CPU \
                 renderer)",
                n,
                want >> 20,
                usage >> 20,
                budget >> 20
            ));
        }
    }

    let build_arena = committed_buffer(device, total_build, uaf, as_state)?;
    let scratch = committed_buffer(
        device,
        scratch_max.max(tlas_info.ScratchDataSizeInBytes),
        uaf,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )?;
    let build_va = unsafe { build_arena.GetGPUVirtualAddress() };
    let scratch_va = unsafe { scratch.GetGPUVirtualAddress() };

    // Submit 1: every chunk build, each emitting its compacted size into one
    // u64-per-chunk buffer, copied to readback in the same list.
    let csize_buf = committed_buffer(
        device,
        (n * 8) as u64,
        uaf,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )?;
    let csize_rb = d3d12::ReadbackBuffer::new(device, n * 8)?;
    let csize_va = unsafe { csize_buf.GetGPUVirtualAddress() };
    sub.run_list(&mut |list| {
        let list4: ID3D12GraphicsCommandList4 =
            list.cast().map_err(|e| format!("ID3D12GraphicsCommandList4: {e}"))?;
        for (i, g) in geoms.iter().enumerate() {
            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: build_va + build_off[i],
                Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    Flags: flags,
                    NumDescs: 1,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: g,
                    },
                },
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch_va,
            };
            let postbuild = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: csize_va + (i * 8) as u64,
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
            };
            unsafe { list4.BuildRaytracingAccelerationStructure(&desc, Some(&[postbuild])) };
            // The scratch buffer is shared, so consecutive builds MUST NOT
            // overlap — this barrier is the serialization, not an optimization
            // to remove. (The bistro-dusk shard hunt proved this sync SOUND on
            // RDNA4 by fencing every build individually and measuring no
            // change — the shards were the index-value defect, not a race.)
            unsafe { list.ResourceBarrier(&[uav_barrier(None)]) };
        }
        unsafe {
            list.ResourceBarrier(&[transition(
                &csize_buf,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )])
        };
        unsafe { list.CopyBufferRegion(&csize_rb.resource, 0, &csize_buf, 0, (n * 8) as u64) };
        Ok(())
    })?;

    let csizes: Vec<u64> = {
        let mut ptr = std::ptr::null_mut();
        unsafe { csize_rb.resource.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| format!("compacted-size Map: {e}"))?;
        let v = (0..n)
            .map(|i| unsafe { (ptr as *const u64).add(i).read_unaligned() })
            .collect();
        unsafe { csize_rb.resource.Unmap(0, None) };
        v
    };

    // Compacted layout. A degenerate reported size (0, or no smaller than the
    // build) keeps that chunk uncompacted — never wrong, just bigger — exactly
    // as the single-BLAS path does.
    let mut final_off = Vec::with_capacity(n);
    let mut total_final = 0u64;
    let mut compact = Vec::with_capacity(n);
    for i in 0..n {
        let built = if i + 1 < n {
            build_off[i + 1] - build_off[i]
        } else {
            total_build - build_off[i]
        };
        let c = csizes[i] > 0 && csizes[i] < built;
        compact.push(c);
        final_off.push(total_final);
        total_final = align_as(total_final + if c { csizes[i] } else { built });
    }
    let arena = committed_buffer(device, total_final, uaf, as_state)?;
    let arena_va = unsafe { arena.GetGPUVirtualAddress() };

    // Identity instances, InstanceID = chunk index — the (InstanceID,
    // PrimitiveIndex) -> tri remap's first coordinate, and the stable handle a
    // cut-driven TLAS rebuild would address chunks by. Under --dxr-sbt,
    // `_bitfield2`'s low 24 bits carry InstanceContributionToHitGroupIndex =
    // class * 3 (flags stay 0 in the high 8); record address = hit-table
    // start + stride * (RayContribution + InstanceContribution), multipliers
    // are literal 0 at every TraceRay site, so the ray-type indices {0,1,2}
    // keep their meaning inside each class's triplet. The sway ring copies
    // these descs verbatim per slot, so animated instances inherit their
    // contribution for free.
    let idescs: Vec<D3D12_RAYTRACING_INSTANCE_DESC> = (0..n)
        .map(|i| {
            let mut idesc: D3D12_RAYTRACING_INSTANCE_DESC = unsafe { std::mem::zeroed() };
            idesc.Transform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            idesc._bitfield1 = (0xffu32 << 24) | (i as u32 & 0x00ff_ffff);
            idesc._bitfield2 = chunk_class.map_or(0, |c| c[i] as u32 * 3);
            idesc.AccelerationStructure = arena_va + final_off[i];
            idesc
        })
        .collect();
    let instances = d3d12::UploadBuffer::new(
        device,
        n * std::mem::size_of::<D3D12_RAYTRACING_INSTANCE_DESC>(),
    )?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            idescs.as_ptr() as *const u8,
            instances.ptr,
            n * std::mem::size_of::<D3D12_RAYTRACING_INSTANCE_DESC>(),
        )
    };
    let tlas = committed_buffer(device, tlas_info.ResultDataMaxSizeInBytes, uaf, as_state)?;
    let instances_va = unsafe { instances.resource.GetGPUVirtualAddress() };

    // Submit 2: compact every chunk into the exact-size arena, then build the
    // TLAS against the COMPACTED addresses.
    sub.run_list(&mut |list| {
        let list4: ID3D12GraphicsCommandList4 =
            list.cast().map_err(|e| format!("ID3D12GraphicsCommandList4: {e}"))?;
        for i in 0..n {
            unsafe {
                list4.CopyRaytracingAccelerationStructure(
                    arena_va + final_off[i],
                    build_va + build_off[i],
                    if compact[i] {
                        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT
                    } else {
                        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE
                    },
                )
            };
        }
        unsafe { list.ResourceBarrier(&[uav_barrier(None)]) };
        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: unsafe { tlas.GetGPUVirtualAddress() },
            Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                NumDescs: n as u32,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: instances_va,
                },
            },
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_va,
        };
        unsafe { list4.BuildRaytracingAccelerationStructure(&desc, None) };
        unsafe { list.ResourceBarrier(&[uav_barrier(None)]) };
        Ok(())
    })?;

    let report = format!(
        "blas {} MB (compacted from {}) | tlas {} MB | transient scratch {} MB (freed)",
        total_final >> 20,
        total_build >> 20,
        tlas_info.ResultDataMaxSizeInBytes >> 20,
        scratch_max.max(tlas_info.ScratchDataSizeInBytes) >> 20,
    );
    Ok(SplitBuild {
        arena,
        tlas,
        report,
        instances: idescs,
        tlas_size: tlas_info.ResultDataMaxSizeInBytes,
        tlas_scratch: tlas_info.ScratchDataSizeInBytes,
    })
}

fn geometry_desc(
    positions: &ID3D12Resource,
    indices: &ID3D12Resource,
    n_verts: u32,
    n_tris: u32,
    non_opaque: bool,
    any_transmissive: bool,
) -> D3D12_RAYTRACING_GEOMETRY_DESC {
    D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        // OPAQUE == the kernels' FORCE_OPAQUE assumption (no any-hit ever) —
        // the per-scene fast path. Alpha-masked/height/transmissive scenes
        // build with NONE so candidates surface to the candidate loops /
        // any-hit shaders (compiled in under the same per-scene predicates).
        // Transmissive scenes ALSO need NO_DUPLICATE_ANYHIT_INVOCATION:
        // D3D12 may legally surface the same triangle more than once to
        // any-hit/candidate code without it, and the tint MULTIPLY is not
        // idempotent (the cutout/relief rejects were, which is why they
        // never needed the flag — a duplicate reject rejects the same way).
        Flags: if non_opaque {
            if any_transmissive {
                D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION
            } else {
                D3D12_RAYTRACING_GEOMETRY_FLAG_NONE
            }
        } else {
            D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
        },
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
/// a GBUF_STRIDE-byte dummy, so this flag is memory safety, not an
/// optimization.
pub const FLAG_GBUF: u32 = 16;
pub const FLAG_HAS_PREV: u32 = 32;
/// FSR-RR sessions: the pack additionally carries the demodulated
/// direct-light signals (GBufExt.sig) and the prev-camera view-Z
/// (GBufCore.core.w);
/// zeros under every other wiring — RR/XeSS packs stay byte-identical.
pub const FLAG_FSR_SIG: u32 = 64;
/// Anisotropic texture filtering on (the session's `--aniso` > 1). A session
/// constant, not a per-frame decision — set from `texture::max_aniso()`, the
/// same source the static aniso sampler's MaxAnisotropy and the CPU's
/// `Cone::aniso` read, so all three renderers filter the same footprint.
/// Which *rays* use it is decided per call site, not by this flag
/// (`shade_split`'s `aniso` arg — hemi bounce laps pass false).
pub const FLAG_ANISO: u32 = 128;
/// Volumetric clouds on (`--no-clouds` clears it). The cloud state rides two
/// otherwise-zero cam-row w lanes — `cam_right.w` = scene diag, `cam_up.w` =
/// the animation clock (`SCENE_DIAG`/`CLOUD_TIME` in trace_common.hlsli, the
/// SCENE_EPS/AO_RADIUS alias pattern) — so no CB offset moves.
pub const FLAG_CLOUDS: u32 = 256;

/// Heightfield relief march on — the V toggle × any_height × the
/// --no-heightfield lever; per-frame runtime gate over the per-scene
/// HEIGHTFIELD compile-in (trace_common.hlsli mirror).
pub const FLAG_HEIGHT: u32 = 512;

/// Firefly point lights live this frame (src/fireflies.rs — count > 0, which
/// already folds in the session enable and the night fade: a day session
/// never sets it, so day kernels are bit-identical by construction). Poses
/// ride the CB's `ff` rows, CPU-baked — the HLSL re-derives nothing.
pub const FLAG_FIREFLIES: u32 = 1024;

/// Beer–Lambert depth tint over the transmission chain's interior segments
/// (`--no-depth-tint` clears it; shade.hlsli branches inside the
/// transmission arm, which non-transmissive scenes never enter — no compile
/// define needed).
pub const FLAG_DEPTH_TINT: u32 = 2048;

/// The pack's guide/signal half is stored this frame — set when any WIRED feed
/// kind consumes it (RR, FSR-RR) or GPU-resident NPPD is live. XeSS and
/// FSR 3.1 read only the core (mv + view_z), so their sessions skip 72 of the
/// old 88 B/px: measured 0.411 ms of pure store cost in `leaf` on the world,
/// of which this recovers ~0.34. Derived per frame like `fsr_sig` (one
/// subscriber is enough — `--quinlight` wires several kinds at once), never a
/// construction-time constant: the tracer is built BEFORE `wire_feed` runs.
///
/// FLAG_FSR_SIG implies this — FSR-RR reads the sig lanes, which live in ext.
pub const FLAG_GBUF_EXT: u32 = 4096;

/// Foliage-sway MVs live this frame: the pack's MV/prev-Z reproject each
/// hit's PREV-POSE point (`p + du·(a + b·p.y)` off the sway_dmv table)
/// instead of the current one. Armed only when the SWAY_MV compile-in is
/// present AND `sway_mv_pair` holds (prev clock present, bit-different, prev
/// camera present) AND the frame's `write_mv_rows` filled the slot — every
/// pinned-clock gate and frozen still runs the flag-off branch, which is
/// today's expressions verbatim (a branch, never an add-zero: −0.0 + 0.0 =
/// +0.0, the fireflies lesson).
pub const FLAG_SWAY_MV: u32 = 8192;

/// Emissive cluster lights live this frame (src/emissive.rs): the scene
/// derived clusters AND the session lever is on AND the frame is not GI
/// (`fb_mode != 2` — the GI gather already delivers emissive transport
/// exactly, so GI frames keep the gather and drop the cluster NEE; the
/// inverted once-per-path rule). An emissive-free scene never sets the bit,
/// so its kernels are bit-identical by construction (the FLAG_FIREFLIES
/// shape).
pub const FLAG_EMISSIVE: u32 = 16384;
/// FR_WAVEVIZ live overlay (armed session AND the I key / headless spin):
/// covered kernels overwrite tbuf with their wave ticket and the resolve
/// stage blends the ticket hash over the scene. Runtime half of the WAVEVIZ
/// compile-in — an armed-but-OFF frame runs the normal tbuf writes, so
/// toggling off recovers the clean frame immediately.
// RENUMBERED at the 2026-08-06 merge: both parallel sessions claimed bit
// 32768 (the sibling branched before FLAG_DETAIL/FLAG_DETAIL_AO/
// FLAG_AMB_BUMP landed); detail keeps 32768..131072, waveviz takes the
// next free bit. Lockstep with trace_common.hlsli's FLAG_WAVEVIZ.
pub const FLAG_WAVEVIZ: u32 = 262144;

/// Real-time GI live this frame (`--no-rtgi` clears the session lever;
/// `with_frame` additionally clears the bit on fb frames — the hemi tiers
/// take precedence — so shade_full's bounce block can key on the bit alone).
/// Lockstep with trace_common.hlsli's FLAG_RTGI.
pub const FLAG_RTGI: u32 = 524288;

/// Spec-AA (`--no-spec-aa` clears it): the slope-variance → roughness fold —
/// mip-averaged normal-map detail (the variance companion, gated per
/// material on Mat.normal_var_tex) and the detail field's faded octaves
/// (detail_var) widen the GGX lobe instead of vanishing with distance. The
/// FLAG_DETAIL runtime-lever shape; lockstep with trace_common.hlsli's
/// FLAG_SPEC_AA.
pub const FLAG_SPEC_AA: u32 = 1048576;

/// An NRD (ReBLUR) bridge is wired this frame, so shade_full's RTGI bounce
/// folds into the `prim.direct_d` capture (+ the bounce ray's t into `ao_t`)
/// — the bridge's diffuse input carries the GI instead of the un-denoised
/// residual. Runtime like FLAG_RTGI (NRD arms at session start and sheds
/// mid-session), and DISTINCT from FLAG_FSR_SIG: FSR-RR sessions arm that
/// bit too, and their dd must stay pure direct diffuse — it is AMD's own
/// denoiser's input. Lockstep with trace_common.hlsli's FLAG_NRD_GI.
pub const FLAG_NRD_GI: u32 = 2097152;

/// Sky pixels SKIP the GBufExt store (bit 22 — the B70 NRD-cost recovery,
/// 2026-08-09): armed only when NRD is the SOLE ext subscriber, because at a
/// sky texel the bridge needs nothing from ext — cs_nrd_pack takes its own
/// canonical-constant sky branch (never reading the possibly-stale bytes) and
/// cs_nrd_out returns at its 0.999·CAM_FAR predicate before the ext load.
/// Every OTHER ext consumer (cs_feed_rr, cs_feed_fsr_rr, nppd.hlsl, the pack
/// readback gates) reads ext full-screen INCLUDING sky, which is why the
/// derivation vetoes on any of them. Measured: the sky ext store was
/// +0.33–0.51 ms/frame on the B70 at native 1080p. Lockstep with
/// trace_common.hlsli's FLAG_SKY_EXT_SKIP.
pub const FLAG_SKY_EXT_SKIP: u32 = 4194304;

/// Unreal-1 detail texturing (`--no-detail-tex` clears it): procedural
/// close-up albedo grain + micro-bump on MAGNIFIED hits — textured AND
/// untextured since the untextured arm (shade.hlsli's post-match detail
/// block: textured materials window off their albedo texture's lod,
/// untextured off the cone footprint in synthetic texel-equivalents,
/// Mat.detail_scale > 0 either way — the FLAG_DEPTH_TINT shape, no compile
/// define needed).
pub const FLAG_DETAIL: u32 = 32768;

/// Detail cavity AO (`--no-detail-ao` clears it): the detail field's pits
/// darken ambient + direct specular (shade.hlsli branches behind `dh < 0`,
/// which only the fired field sets — the FLAG_DETAIL runtime-lever shape).
pub const FLAG_DETAIL_AO: u32 = 65536;

/// Ambient bump response (`--no-amb-bump` clears it): shade.hlsli's
/// `amb_irradiance` amplifies the SH ambient's response to the n_g → n_s
/// deviation (normal maps + detail bump + ripple) — flat-shaded geometry
/// (n_s == n) takes the plain expression verbatim, the runtime-lever shape.
pub const FLAG_AMB_BUMP: u32 = 131072;

/// `GBufCore` stride in bytes — lockstep with trace_common.hlsli (one float4:
/// mv.xy | view_z | prev_z).
pub const GBUF_STRIDE: u64 = 16;

/// `GBufExt` stride in bytes — lockstep with trace_common.hlsli
/// (nr | alb | spec | sig | sig2 = 3 float4 + 1 uint4 + 1 uint2).
pub const GBUF_EXT_STRIDE: u64 = 72;

/// Mirror of `cbuffer Frame` in trace_common.hlsli (304 bytes, 16-aligned
/// rows — float3s ride in float4 slots with scalars packed in .w).
/// pub(crate): gpu/dxr.rs shares the layout (its lib pastes the same
/// trace_common.hlsli); fields stay module-private — outside constructors go
/// through `FrameCb::base`/`with_frame`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FrameCb {
    cam_origin: [f32; 4],
    cam_forward: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    /// The sun (sky::Sun). Replaces the old five rows (sun + the rect light's
    /// center/u/v/color): a disc at infinity needs a direction, a cone, and two
    /// radiometric values. `scene.eps` / `ao_radius` used to ride in
    /// `light_center.w` / `light_u.w`; they are rehomed onto these rows' w slots.
    pub(crate) sun: [f32; 4], // xyz = unit dir; w = cos(angular radius)
    sun_e: [f32; 4], // xyz = irradiance/π (the direct loop's multiplier); w = scene eps
    sun_l: [f32; 4], // xyz = DISC radiance (what an escaping ray sees); w = ao_radius
    rw: u32,
    rh: u32,
    frame: u32,
    flags: u32,
    shadow_samples: u32,
    ao_samples: u32,
    reflections: u32,
    /// The fireflies' CONTENT-diagonal scale (`Fireflies::scale` — every FF_*
    /// length multiplies it; deliberately NOT SCENE_DIAG, which the ground
    /// quad inflates ~17× on the procedural scenes). Rode what was `_pad0`.
    ff_scale: f32,
    frame_jitter: [f32; 2],
    /// Primary ray-cone spread (CamBasis::pixel_cone — the CPU value
    /// verbatim, single source for the trilinear LOD parity).
    pixel_cone: f32,
    /// Time-of-day dome brightness (`Scene::sky_scale` — exactly 1.0 in an
    /// untouched session; `x * 1.0` is bit-preserving, so the day sky gates
    /// are unmoved). Rides what was `_pad2`, so no offset moves.
    sky_scale: f32,
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
    /// Live firefly count (rode what was `_pad3`, so no offset moves) —
    /// 0 in every day/`--no-fireflies` session; FLAG_FIREFLIES mirrors it.
    ff_count: u32,
    // Previous frame's camera basis for G-buffer MVs; near/far ride the w
    // slots of the last two rows (scene-static, from dlss::near_far).
    prev_origin: [f32; 4],  // xyz; w = prev inv_w
    prev_forward: [f32; 4], // xyz; w = prev inv_h
    prev_right: [f32; 4],   // xyz; w = near
    prev_up: [f32; 4],      // xyz; w = far
    // --spp: samples per pixel this frame, and which one writes the per-pixel
    // side channels (tbuf/info/pack). See trace_common.hlsli.
    spp: u32,
    probe_sample: u32,
    /// Star visibility (`Scene::night` — exactly 0.0 in an untouched session;
    /// the HLSL star branch is guarded on it, so day kernels are bit-identical
    /// by construction). Rides what was `_pad4`.
    night: f32,
    /// Scene-wide max relief depth in world units (`bvh::height_max_world` —
    /// 0.0 = no height data, which is also how FLAG_HEIGHT's `with_frame`
    /// predicate reads `any_height`). The wavefront TMin widening constant.
    height_max: f32,
    /// Sample offsets from `dlss::jitter_for_sample` (the ONE Halton source —
    /// no radical-inverse port in HLSL), two per 16-byte row.
    jitters: [[f32; 4]; (crate::dlss::MAX_SPP as usize) / 2],
    /// The sky dome in order-2 SH (`scene.sky_sh`, `sh::N` = 9 RGB rows, .w
    /// unused) — the GPU's copy of the analytic ambient the CPU reads through
    /// `Sh9::irradiance`. Appended after every scalar so no offset above moves.
    sky_sh: [[f32; 4]; crate::sh::N],
    /// Firefly poses (src/fireflies.rs — xyz = world position, w =
    /// brightness), the CPU's baked f32s verbatim so both renderers light
    /// from bit-equal positions. Appended LAST (the sky_sh precedent); rows
    /// past `ff_count` are zero and never read (the HLSL loops on the count).
    ff: [[f32; 4]; crate::fireflies::MAX_FIREFLIES],
    /// Cloud shadow cache transform: [origin.x, origin.z, 1/cell, side].
    /// Appended LAST (the sky_sh / ff precedent) so no offset above moves.
    /// pub(crate) so DxrGpu::write_cb can fill it (the shared shadow_grid_row).
    pub(crate) cloud_grid: [f32; 4],
    /// Sway-MV delta table base: x = the frame's ring-slot offset in float4
    /// elements (`slot · n_inst` — the shader reads `sway_dmv[x +
    /// InstanceID]`), yzw unused. Appended LAST (the cloud_grid precedent).
    /// Set through `arm_sway_mv` beside FLAG_SWAY_MV so the pair cannot
    /// split.
    sway_mv_base: [u32; 4],
    /// Emissive cluster lights (src/emissive.rs): x = count, yzw unused.
    /// Scene-static — filled by `base` from `Scene::emissive`; FLAG_EMISSIVE
    /// mirrors it per frame (× the live lever × fb_mode != 2).
    el_meta: [u32; 4],
    /// Cluster row a: xyz = power-weighted centroid, w = rc² (source
    /// radius²) — the CPU's derived f32s VERBATIM, so both renderers light
    /// from bit-equal clusters (parity BY DATA, the ff precedent). Rows past
    /// the count are zero and never read (the HLSL loops on the count).
    el_a: [[f32; 4]; crate::emissive::MAX_EMISSIVE_LIGHTS],
    /// Cluster row b: xyz = C/π (radiance·area over π), w = r_infl²
    /// (the window's exact zero). Appended LAST so no offset above moves.
    el_b: [[f32; 4]; crate::emissive::MAX_EMISSIVE_LIGHTS],
    /// Dual-GPU tile ownership (`--dual-gpu`): xy = the bitmask of
    /// level-`z` quadtree tiles THIS device renders (x = tiles 0..31,
    /// y = 32..63), z = the split depth, w unused. Appended LAST so no
    /// offset above moves.
    ///
    /// `z == 0` is the unsplit session — `level_finish` branches around the
    /// test entirely, so every single-GPU frame is bit-identical to the
    /// pre-feature renderer by construction (the `apply_tod`/`night`
    /// precedent). 64 bits caps the split depth at `MAX_SPLIT_DEPTH` = 3.
    split: [u32; 4],
}

/// Deepest quadtree level a `--dual-gpu` split may be assigned at: 4^3 = 64
/// tiles, exactly the 64 bits of the CB's `split.xy` mask. Also the point of
/// diminishing returns — 1/64 of the screen is finer than the balancer can
/// usefully act on, since every reassignment invalidates that device's
/// structure replay.
pub(crate) const MAX_SPLIT_DEPTH: u32 = 3;

/// Which level-`depth` quadtree tiles THIS device renders (`--dual-gpu`).
///
/// A per-DEVICE property that changes only when the balancer reassigns tiles,
/// which is why it lives on `TraceGpu` rather than in `FrameParams` beside the
/// per-frame jitter: the tracer already owns the other things fixed for a
/// device (its resolution, its queues), and the split belongs with them.
///
/// `depth == 0` is the whole screen — `ALL`, the unsplit default, in which
/// `level_finish` branches around the ownership test entirely and the frame is
/// bit-identical to the pre-feature renderer.
///
/// Equality is part of the structure-replay key: a device whose assignment
/// changed must NOT replay the terminal queues it recorded for the old one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TileSplit {
    /// Bit `i` = the level-`depth` tile whose quadtree path is `i`. Unused
    /// above `4^depth`.
    pub mask: u64,
    /// The level `mask` indexes; 0 = unsplit.
    pub depth: u32,
}

impl Default for TileSplit {
    fn default() -> Self {
        Self::ALL
    }
}

impl TileSplit {
    /// The whole screen — one device, no split, every ownership test skipped.
    pub const ALL: TileSplit = TileSplit { mask: u64::MAX, depth: 0 };

    /// Tiles at `depth`, as a count.
    pub fn tiles_at(depth: u32) -> u32 {
        1u32 << (2 * depth)
    }

    /// A CONTIGUOUS band of whole tile ROWS at `depth`: rows `[row0, row1)` of
    /// the `2^depth × 2^depth` grid.
    ///
    /// This is the shape mixed-mode dual-GPU requires — a DXR partner renders a
    /// rectangle, so the wavefront side's tiles must form one too (an
    /// interleaved mask cannot be a single `DispatchRays`). It is also the
    /// cheapest cross-adapter transfer: one contiguous row range, one copy.
    ///
    /// A tile's row is the interleaved "bottom" bit of its path — bit 1 of each
    /// level's 2-bit code (TL=0 TR=1 BL=2 BR=3), most significant level first.
    pub fn rows(depth: u32, row0: u32, row1: u32) -> TileSplit {
        let mut mask = 0u64;
        for path in 0..Self::tiles_at(depth) {
            let mut row = 0u32;
            for lvl in 0..depth {
                // Level `lvl` contributes its B bit; the FIRST level split is
                // the most significant row bit.
                let shift = 2 * (depth - 1 - lvl);
                row = (row << 1) | ((path >> (shift + 1)) & 1);
            }
            if row >= row0 && row < row1 {
                mask |= 1u64 << path;
            }
        }
        TileSplit { mask, depth }
    }

    /// The complement within `depth` — the partner device's assignment. Their
    /// union must be every tile and their intersection empty, which is what
    /// makes the two halves partition the screen exactly.
    pub fn complement(&self) -> TileSplit {
        if self.depth == 0 {
            return TileSplit { mask: 0, depth: 0 };
        }
        let all = if Self::tiles_at(self.depth) >= 64 {
            u64::MAX
        } else {
            (1u64 << Self::tiles_at(self.depth)) - 1
        };
        TileSplit { mask: !self.mask & all, depth: self.depth }
    }

    /// Does this assignment own the level-`depth` tile CONTAINING pixel
    /// `(x, y)` of an `rw x rh` screen?
    ///
    /// The twin of `trace_common.hlsli`'s `split_owns_px`, written as the same
    /// forward midpoint recursion so the two cannot drift. Two consumers: the
    /// shader side bands `cs_compose` (the one per-pixel pass in the tracer),
    /// and the CPU side derives the cross-adapter transfer's row ranges from it.
    ///
    /// `depth == 0` is the unsplit whole screen and answers true without
    /// touching the mask, matching the branch every other consumer takes.
    pub fn owns_px(&self, x: u32, y: u32, rw: u32, rh: u32) -> bool {
        if self.depth == 0 {
            return true;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (0u32, 0u32, rw, rh);
        let mut path = 0u32;
        for _ in 0..self.depth {
            let xm = x0 + (x1 - x0) / 2;
            let ym = y0 + (y1 - y0) / 2;
            let cx = u32::from(x >= xm);
            let cy = u32::from(y >= ym);
            path = (path << 2) | (cy * 2 + cx);
            if cx == 1 {
                x0 = xm;
            } else {
                x1 = xm;
            }
            if cy == 1 {
                y0 = ym;
            } else {
                y1 = ym;
            }
        }
        // The shader's conservative arm, mirrored: an out-of-range path renders
        // a tile twice rather than dropping it. Unreachable at
        // `depth <= MAX_SPLIT_DEPTH`, kept so the twins stay textually equal.
        if path >= 64 {
            return true;
        }
        (self.mask >> path) & 1 != 0
    }

    /// The two devices' assignments for a secondary share of `rows` out of
    /// `2^depth` tile rows: `(primary, secondary)`.
    ///
    /// **A share of ZERO returns `(ALL, None)`, and that is the safety
    /// property the whole feature rests on.** Not `rows(depth, 0, side)` — a
    /// full mask at a nonzero depth is functionally the same but structurally
    /// different: `SPLIT_DEPTH != 0` makes `level_finish` run the ownership
    /// test and `cs_compose` run `split_owns_px` per pixel. `ALL` is depth 0,
    /// which every consumer branches AROUND, so share 0 is the pre-feature
    /// renderer instruction-for-instruction.
    ///
    /// That is what makes it honest to ship a feature whose correct answer on
    /// a bandwidth-starved box is "give the secondary nothing": arming it
    /// costs exactly nothing when the balancer converges to zero. The `None`
    /// is the other half — the caller must skip the secondary's submit and
    /// the transfer outright, not hand it an empty mask and pay the schedule.
    ///
    /// A share at or above `side` would leave the PRIMARY with nothing, which
    /// no consumer expects (it still presents), so it is clamped to `side-1`.
    pub fn for_share(rows: u32, depth: u32) -> (TileSplit, Option<TileSplit>) {
        if rows == 0 || depth == 0 {
            return (TileSplit::ALL, None);
        }
        let side = 1u32 << depth;
        let rows = rows.min(side - 1);
        let prim = TileSplit::rows(depth, 0, side - rows);
        (prim, Some(prim.complement()))
    }

    /// The CONTIGUOUS pixel row range `[y0, y1)` this assignment owns, or
    /// `None` if it is not a whole-tile-row band.
    ///
    /// This is what makes the cross-adapter transfer one `CopyBufferRegion`
    /// per buffer instead of a scatter: every per-pixel plane is indexed
    /// `y*rw + x`, so a row band is a contiguous BYTE range at
    /// `y0*rw*stride` for `(y1-y0)*rw*stride` bytes. `TileSplit::rows` is
    /// built to satisfy this; an interleaved mask deliberately answers `None`
    /// so a caller cannot silently copy the wrong bytes for one.
    ///
    /// Returns `None` rather than a bounding box on purpose — a bounding box
    /// would be a plausible-looking answer that copies pixels the partner
    /// owns, which is exactly the overlap the whole design forbids.
    pub fn row_range(&self, rw: u32, rh: u32) -> Option<(u32, u32)> {
        if self.depth == 0 {
            return Some((0, rh));
        }
        let side = 1u32 << self.depth;
        // Per grid row: how many of its tiles are owned. A band must own all
        // of them or none — a partially-owned row is not a row band.
        let mut owned = [0u32; 8];
        debug_assert!(side as usize <= owned.len());
        for path in 0..Self::tiles_at(self.depth) {
            if (self.mask >> path) & 1 == 0 {
                continue;
            }
            // The tile's grid row: the interleaved "bottom" bit of each
            // level's 2-bit code, most significant level first — the same
            // extraction `rows()` builds the mask from.
            let mut row = 0u32;
            for lvl in 0..self.depth {
                let shift = 2 * (self.depth - 1 - lvl);
                row = (row << 1) | ((path >> (shift + 1)) & 1);
            }
            owned[row as usize] += 1;
        }
        let mut first = None;
        let mut last = 0u32;
        for r in 0..side {
            match owned[r as usize] {
                0 => {
                    // A gap AFTER the band started means two disjoint bands.
                    if first.is_some() && r <= last {
                        return None;
                    }
                }
                n if n == side => {
                    if first.is_none() {
                        first = Some(r);
                    } else if r != last + 1 {
                        return None; // non-contiguous
                    }
                    last = r;
                }
                _ => return None, // partially-owned row
            }
        }
        let first = first?;
        // Every tile in a grid row shares that row's y extent (the y half of
        // the midpoint recursion depends only on the row bits), so any tile of
        // the row gives the band's edges.
        let top = Self::first_path_of_row(self.depth, first);
        let bot = Self::first_path_of_row(self.depth, last);
        let (_, y0, _, _) = rect_for_path(self.depth, top, rw, rh);
        let (_, _, _, y1) = rect_for_path(self.depth, bot, rw, rh);
        Some((y0, y1))
    }

    /// The lowest path index whose grid row is `row` — the inverse of the row
    /// extraction above, taking every x bit as 0.
    fn first_path_of_row(depth: u32, row: u32) -> u32 {
        let mut path = 0u32;
        for lvl in 0..depth {
            let bit = (row >> (depth - 1 - lvl)) & 1;
            path = (path << 2) | (bit << 1);
        }
        path
    }

    /// The CB row: xy = the mask, z = depth, w unused.
    fn cb_row(&self) -> [u32; 4] {
        [self.mask as u32, (self.mask >> 32) as u32, self.depth, 0]
    }
}

/// The screen rect of level-`depth` tile `path`, replaying `trace_tile` /
/// `level_finish`'s integer midpoint splits exactly (`xm = x0 + (x1-x0)/2`,
/// TL=0 TR=1 BL=2 BR=3).
///
/// Test-side only: it exists so `split_self_test` can check the ownership
/// mask's BIT math against the actual tile GEOMETRY, which is the pair that
/// can drift. The renderer never calls it — the shader derives child rects as
/// it descends.
fn rect_for_path(depth: u32, path: u32, rw: u32, rh: u32) -> (u32, u32, u32, u32) {
    let (mut x0, mut y0, mut x1, mut y1) = (0u32, 0u32, rw, rh);
    for lvl in 0..depth {
        let code = (path >> (2 * (depth - 1 - lvl))) & 3;
        let xm = x0 + (x1 - x0) / 2;
        let ym = y0 + (y1 - y0) / 2;
        if code & 1 == 0 {
            x1 = xm;
        } else {
            x0 = xm;
        }
        if (code >> 1) & 1 == 0 {
            y1 = ym;
        } else {
            y0 = ym;
        }
    }
    (x0, y0, x1, y1)
}

/// Pure-math gates for the `--dual-gpu` tile split. DLL- and GPU-free, and run
/// by every `--check` regardless of the lever — the blas-split rule, so the
/// machinery cannot rot while the feature is off.
///
/// What this actually protects: `TileSplit::rows` derives a tile's ROW from
/// interleaved path bits, while the renderer derives a tile's RECT by
/// recursive midpoint splits. Those are two independent derivations of the same
/// thing, and if they disagree a device renders tiles it does not own (wasted
/// work) or, worse, neither device renders a tile (a hole in the image). The
/// geometry cross-check below is the gate that ties them together.
pub fn split_self_test() -> std::result::Result<(), String> {
    // The unsplit default must be exactly the state every consumer branches
    // around — depth 0. If this drifts, single-GPU frames stop being
    // bit-identical and every existing gate silently changes meaning.
    if TileSplit::ALL.depth != 0 {
        return Err("TileSplit::ALL must be depth 0 (the branched-around unsplit state)".into());
    }

    // The documented level-1 claim: a horizontal half-split IS the level-1
    // quadrant boundary, top band = TL | TR = paths 0 and 1.
    let top = TileSplit::rows(1, 0, 1);
    if top.mask != 0b0011 {
        return Err(format!(
            "rows(1,0,1) must be TL|TR = 0b0011, got {:#06b} — the top band is not the \
             level-1 quadrant pair, so a half-split is no longer a quadtree subtree",
            top.mask
        ));
    }

    for depth in 1..=MAX_SPLIT_DEPTH {
        let n = TileSplit::tiles_at(depth);
        let side = 1u32 << depth;
        let full = TileSplit::rows(depth, 0, side);
        let all_bits = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
        if full.mask != all_bits {
            return Err(format!(
                "rows({depth},0,{side}) must cover all {n} tiles, got {:#x}",
                full.mask
            ));
        }

        for r in 0..=side {
            let a = TileSplit::rows(depth, 0, r);
            let b = TileSplit::rows(depth, r, side);

            // PARTITION: disjoint, and together every tile. This is what makes
            // the two devices' work cover the screen exactly once — the
            // property `--check-gpu`'s exactly-once coverage gate asserts on
            // the GPU, checked here in closed form at every split position.
            if a.mask & b.mask != 0 {
                return Err(format!(
                    "depth {depth} row {r}: the two bands overlap ({:#x}) — those tiles \
                     would be rendered twice",
                    a.mask & b.mask
                ));
            }
            if a.mask | b.mask != all_bits {
                return Err(format!(
                    "depth {depth} row {r}: the two bands leave {:#x} unrendered — a hole \
                     in the image (the false-sky class)",
                    all_bits & !(a.mask | b.mask)
                ));
            }
            // The partner's assignment must be derivable as the complement,
            // since that is how the second device is actually configured.
            if a.complement().mask != b.mask || a.complement().depth != depth {
                return Err(format!(
                    "depth {depth} row {r}: complement() disagrees with rows() — \
                     {:#x} vs {:#x}",
                    a.complement().mask,
                    b.mask
                ));
            }

            // GEOMETRY: the mask's bit math vs the renderer's rect recursion,
            // at a power-of-two resolution AND an odd one (where the integer
            // midpoint rounds and the bands are NOT equal height).
            for &(rw, rh) in &[(1920u32, 1080u32), (533, 400)] {
                // The seam: the largest y1 among the top band's tiles must be
                // the smallest y0 among the bottom band's. Anything else is a
                // gap or an overlap in SCREEN space even if the masks
                // partition in INDEX space.
                let mut top_max_y1 = 0u32;
                let mut bot_min_y0 = u32::MAX;
                for path in 0..n {
                    let (_, y0, _, y1) = rect_for_path(depth, path, rw, rh);
                    let in_a = (a.mask >> path) & 1 == 1;
                    if in_a {
                        top_max_y1 = top_max_y1.max(y1);
                    } else {
                        bot_min_y0 = bot_min_y0.min(y0);
                    }
                }
                if r > 0 && r < side && top_max_y1 != bot_min_y0 {
                    return Err(format!(
                        "depth {depth} row {r} at {rw}x{rh}: band seam disagrees — top ends \
                         at y={top_max_y1}, bottom starts at y={bot_min_y0}. The mask's row \
                         bits and the midpoint-split rects have drifted apart."
                    ));
                }
            }
        }

        // OWNS_PX: the FORWARD pixel->path recursion against the BACKWARD
        // path->rect one — the same two-independent-derivations check the
        // seam test above applies to `rows`.
        //
        // It matters because `cs_compose` is a flat per-pixel dispatch that
        // bands itself with the shader twin of `owns_px`, while the tiles
        // themselves descend through the rect recursion. A drift between the
        // two blanks or double-writes a band on fb frames only — and no image
        // gate can see it, since fb-off frames never dispatch compose at all.
        for &(rw, rh) in &[(1920u32, 1080u32), (533, 400)] {
            // A deliberately MIXED assignment (every other tile): a uniform
            // mask would pass a recursion that always answered the same way.
            let s = TileSplit { mask: 0x5555_5555_5555_5555u64 & all_bits, depth };
            for path in 0..n {
                let (x0, y0, x1, y1) = rect_for_path(depth, path, rw, rh);
                if x0 >= x1 || y0 >= y1 {
                    continue; // degenerate at this resolution — owns no pixels
                }
                let want = (s.mask >> path) & 1 == 1;
                for &(px, py) in &[
                    (x0, y0),
                    (x1 - 1, y0),
                    (x0, y1 - 1),
                    (x1 - 1, y1 - 1),
                    (x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2),
                ] {
                    let got = s.owns_px(px, py, rw, rh);
                    if got != want {
                        return Err(format!(
                            "depth {depth} at {rw}x{rh}: owns_px({px},{py}) = {got}, but that \
                             pixel lies in tile path {path}, whose mask bit is {want}. The \
                             pixel->path recursion and the path->rect one have drifted — \
                             cs_compose would band on a different grid than the tiles do."
                        ));
                    }
                }
            }
        }
    }

    // ROW_RANGE: the transfer's byte range. Two bands' pixel rows must
    // partition [0, rh) exactly and meet at the same seam the tile rects do —
    // an off-by-one here copies a row twice or leaves one stale, which is the
    // hole/overlap class again, one level down in the stack.
    for depth in 1..=MAX_SPLIT_DEPTH {
        let side = 1u32 << depth;
        for &(rw, rh) in &[(1920u32, 1080u32), (533, 400)] {
            for r in 1..side {
                let a = TileSplit::rows(depth, 0, r);
                let b = a.complement();
                let (ay0, ay1) = a.row_range(rw, rh).ok_or_else(|| {
                    format!("depth {depth} row {r} at {rw}x{rh}: rows() produced a mask row_range calls non-contiguous")
                })?;
                let (by0, by1) = b.row_range(rw, rh).ok_or_else(|| {
                    format!("depth {depth} row {r} at {rw}x{rh}: the complement of a row band must also be one")
                })?;
                if ay0 != 0 || by1 != rh || ay1 != by0 {
                    return Err(format!(
                        "depth {depth} row {r} at {rw}x{rh}: bands [{ay0},{ay1}) and [{by0},{by1}) \
                         do not partition [0,{rh}) — the transfer would copy a row twice or leave \
                         one stale"
                    ));
                }
                // And the seam must be where the TILES say it is, not merely
                // self-consistent: row_range and rect_for_path are again two
                // derivations of one number.
                let top = TileSplit::first_path_of_row(depth, r);
                let (_, ty0, _, _) = rect_for_path(depth, top, rw, rh);
                if ay1 != ty0 {
                    return Err(format!(
                        "depth {depth} row {r} at {rw}x{rh}: band seam {ay1} disagrees with the \
                         tile rect's y0 {ty0}"
                    ));
                }
            }
        }
    }

    // An INTERLEAVED mask must refuse rather than answer with a bounding box:
    // a plausible-looking range there would copy pixels the partner owns.
    let inter = TileSplit { mask: 0b0101, depth: 1 };
    if inter.row_range(1920, 1080).is_some() {
        return Err(
            "an interleaved (non-row-band) mask must return None from row_range — a bounding \
             box would silently copy the partner's pixels"
                .into(),
        );
    }
    // A partially-owned row must refuse for the same reason.
    let partial = TileSplit { mask: 0b0001, depth: 1 };
    if partial.row_range(1920, 1080).is_some() {
        return Err("a partially-owned tile row must return None from row_range".into());
    }

    // THE ZERO-SHARE IDENTITY — the safety property the whole feature rests
    // on, and the reason it is honest to ship a split whose correct answer on
    // a bandwidth-starved box is "give the secondary nothing".
    //
    // Share 0 must return the DEPTH-0 unsplit state, not a full mask at a
    // nonzero depth. The two render the same image, but only depth 0 is the
    // state every consumer branches AROUND: at any nonzero depth
    // `level_finish` runs the ownership test and `cs_compose` runs
    // `split_owns_px` for every pixel. Arming the feature must cost exactly
    // nothing when the balancer converges to zero.
    for d in 0..=MAX_SPLIT_DEPTH {
        let (p, s) = TileSplit::for_share(0, d);
        if p != TileSplit::ALL || s.is_some() {
            return Err(format!(
                "for_share(0, {d}) must be (ALL, None) — a share of zero has to take the \
                 pre-feature path, not a full mask at depth {d} that still runs the ownership \
                 test on every tile and every compose pixel"
            ));
        }
    }
    for d in 1..=MAX_SPLIT_DEPTH {
        let side = 1u32 << d;
        for rows in 1..side {
            let (p, s) = TileSplit::for_share(rows, d);
            let s = s.ok_or_else(|| format!("for_share({rows}, {d}) dropped the secondary"))?;
            // The pair must still partition, and the SECONDARY must be the one
            // that grows with the share — invert this and every safety
            // property inverts with it (the balancer's "down is safe"
            // direction would then hand the slow device MORE work).
            if p.complement() != s {
                return Err(format!("for_share({rows}, {d}) is not a complementary pair"));
            }
            let (y0, y1) = s
                .row_range(1920, 1080)
                .ok_or_else(|| format!("for_share({rows}, {d}) secondary is not a row band"))?;
            let got = ((y1 - y0) as f32 / 1080.0 * side as f32).round() as u32;
            if got != rows {
                return Err(format!(
                    "for_share({rows}, {d}): the secondary got {got}/{side} rows, not {rows} — \
                     the share is oriented at the wrong device"
                ));
            }
        }
        // Asking for the whole screen must leave the primary something: it is
        // the device that presents.
        let (p, _) = TileSplit::for_share(side, d);
        if p.mask == 0 {
            return Err(format!("for_share({side}, {d}) starved the primary"));
        }
    }

    // The unsplit default must answer true everywhere without consulting the
    // mask: that is the branch `cs_compose` short-circuits on, and if it ever
    // returned false a single-GPU fb frame would come back black.
    for &(x, y) in &[(0u32, 0u32), (1919, 1079), (960, 540)] {
        if !TileSplit::ALL.owns_px(x, y, 1920, 1080) {
            return Err(format!(
                "TileSplit::ALL must own every pixel; ({x},{y}) came back unowned — an \
                 unsplit fb frame would compose nothing there"
            ));
        }
    }

    // A depth past the mask's width must be refused rather than silently
    // truncated — the CB carries 64 bits and nothing else.
    if TileSplit::tiles_at(MAX_SPLIT_DEPTH) != 64 {
        return Err(format!(
            "MAX_SPLIT_DEPTH={MAX_SPLIT_DEPTH} implies {} tiles, but the CB mask holds 64",
            TileSplit::tiles_at(MAX_SPLIT_DEPTH)
        ));
    }
    Ok(())
}
// The HLSL cbuffer is hand-mirrored across 7 concatenated compile units —
// a size drift here corrupts every field after the drift point.
// 304 (the pre-sun size) − 32 (two rect-light rows dropped) + 16 (the spp
// block) + 8·MAX_SPP (the jitter table) + 16·9 (the SH sky) +
// 16·MAX_FIREFLIES (the firefly pose rows) + 16 + 32·MAX_EMISSIVE_LIGHTS
// (the emissive cluster meta + row pairs).
const _: () = assert!(
    std::mem::size_of::<FrameCb>()
        == 320 - 32
            + 8 * crate::dlss::MAX_SPP as usize
            + 16 * crate::sh::N
            + 16 * crate::fireflies::MAX_FIREFLIES
            + 16 // cloud_grid
            + 16 // sway_mv_base
            + 16 // el_meta
            + 32 * crate::emissive::MAX_EMISSIVE_LIGHTS
            + 16 // split (dual-GPU tile ownership)
);
// ...and the whole thing must still fit a CB ring slot.
const _: () = assert!(std::mem::size_of::<FrameCb>() <= CB_STRIDE);

impl FrameCb {
    /// The scene-static base: sun/light/eps/ao_radius, near/far riding the
    /// prev rows' w slots, rw/rh. Queue capacities zero — the wavefront
    /// tracer overwrites its own; the DXR pipeline never reads them.
    pub(crate) fn base(scene: &Scene, rw: u32, rh: u32) -> FrameCb {
        let sun = crate::render::sun_dir(scene);
        let (near, far) = crate::dlss::near_far(scene.diag);
        let v4 = |v: Vec3A, w: f32| [v.x, v.y, v.z, w];
        let mut sky_sh = [[0.0f32; 4]; crate::sh::N];
        for (dst, c) in sky_sh.iter_mut().zip(scene.sky_sh.c.iter()) {
            *dst = [c.x, c.y, c.z, 0.0];
        }
        // Emissive cluster rows: the CPU's derived f32s verbatim — both
        // renderers light from bit-equal clusters (parity BY DATA, the ff
        // precedent). Scene-static, so they ride the base.
        let mut el_a = [[0.0f32; 4]; crate::emissive::MAX_EMISSIVE_LIGHTS];
        let mut el_b = [[0.0f32; 4]; crate::emissive::MAX_EMISSIVE_LIGHTS];
        for i in 0..scene.emissive.count as usize {
            let l = &scene.emissive.lights[i];
            el_a[i] = [l.pos[0], l.pos[1], l.pos[2], l.rc2];
            el_b[i] = [l.color[0], l.color[1], l.color[2], l.r_infl2];
        }
        FrameCb {
            sky_sh,
            cam_origin: [0.0; 4],
            cam_forward: [0.0; 4],
            cam_right: [0.0; 4],
            cam_up: [0.0; 4],
            sun: v4(sun, scene.sun.cos_radius),
            sun_e: v4(scene.sun.e_over_pi, scene.eps),
            sun_l: v4(scene.sun.radiance, scene.ao_radius),
            rw,
            rh,
            frame: 0,
            flags: 0,
            shadow_samples: 0,
            ao_samples: 0,
            reflections: 0,
            ff_scale: 1.0,
            frame_jitter: [0.0, 0.0],
            pixel_cone: 0.0,
            sky_scale: scene.sky_scale,
            cap_tile: 0,
            cap_leaf: 0,
            cap_sky: 0,
            cap_cut: 0,
            fb_mode: 0,
            fb_depth: 2,
            hemi_batch: HEMI_BATCH,
            cap_hemi_pt: rw * rh,
            cap_hemi_cell: 0,
            cap_hemi_leaf: 0,
            cap_hemi_cut: 0,
            ff_count: 0,
            ff: [[0.0; 4]; crate::fireflies::MAX_FIREFLIES],
            cloud_grid: [0.0; 4],
            // Unsplit: depth 0 means every consumer branches around the
            // ownership test, so the whole feature is off by default.
            split: [0; 4],
            sway_mv_base: [0; 4],
            el_meta: [scene.emissive.count, 0, 0, 0],
            el_a,
            el_b,
            prev_origin: [0.0; 4],
            prev_forward: [0.0; 4],
            prev_right: [0.0, 0.0, 0.0, near],
            prev_up: [0.0, 0.0, 0.0, far],
            spp: 1,
            probe_sample: 0,
            night: scene.night,
            height_max: crate::bvh::height_max_world(scene),
            jitters: [[0.0; 4]; (crate::dlss::MAX_SPP as usize) / 2],
        }
    }

    /// Re-derive the sun/sky rows from the scene after a TOD change
    /// (`scene::apply_tod`) — the shared body of `TraceGpu::refresh_sky` /
    /// `DxrGpu::refresh_sky`. Whole rows are copied from a fresh base, so the
    /// rehomed w slots (sun_e.w = eps, sun_l.w = ao_radius) are preserved by
    /// construction; every other field (queue caps included) is untouched.
    pub(crate) fn refresh_sky_rows(&mut self, scene: &Scene, rw: u32, rh: u32) {
        let fresh = FrameCb::base(scene, rw, rh);
        self.sun = fresh.sun;
        self.sun_e = fresh.sun_e;
        self.sun_l = fresh.sun_l;
        self.sky_sh = fresh.sky_sh;
        self.sky_scale = fresh.sky_scale;
        self.night = fresh.night;
    }

    /// The per-frame fields folded onto the static base — the single source
    /// for the FrameParams -> cbuffer mapping (both dispatch flavors).
    pub(crate) fn with_frame(
        &self,
        p: &FrameParams,
        gbuf_full: bool,
        fsr_sig: bool,
        gbuf_ext: bool,
        nrd_sig: bool,
        sky_ext_skip: bool,
    ) -> FrameCb {
        let (origin, forward, right, up, inv_w, inv_h) = p.cam.gpu_fields();
        let mut cb = *self;
        cb.cam_origin = [origin.x, origin.y, origin.z, inv_w];
        cb.cam_forward = [forward.x, forward.y, forward.z, inv_h];
        // The cloud state rides the cam rows' free w lanes (SCENE_DIAG /
        // CLOUD_TIME in the HLSL) — per-frame values on per-frame rows.
        cb.cam_right = [right.x, right.y, right.z, p.clouds.diag];
        cb.cam_up = [up.x, up.y, up.z, p.clouds.time];
        cb.frame = p.frame;
        cb.flags = (p.accumulate as u32 * FLAG_ACCUM)
            | (p.jitter as u32 * FLAG_JITTER)
            | (p.frame_jitter.is_some() as u32 * FLAG_FRAME_JITTER)
            | (p.verify as u32 * FLAG_VERIFY)
            | (gbuf_full as u32 * FLAG_GBUF)
            | (p.prev_cam.is_some() as u32 * FLAG_HAS_PREV)
            | ((gbuf_full && fsr_sig) as u32 * FLAG_FSR_SIG)
            // FSR-RR reads the sig lanes, which live in ext — so the sig flag
            // implies the ext flag by construction, not by convention.
            | ((gbuf_full && (gbuf_ext || fsr_sig)) as u32 * FLAG_GBUF_EXT)
            // The NRD RTGI fold rides the sig capture (it edits the lanes the
            // sig store writes), so it requires the sig flag by construction.
            | ((gbuf_full && fsr_sig && nrd_sig) as u32 * FLAG_NRD_GI)
            // Sky ext-store skip: only meaningful when the ext store runs at
            // all, so it requires the GBUF flag by construction (the branch
            // sits behind gbuf_write_sky's own FLAG_GBUF/FLAG_GBUF_EXT gates).
            | ((gbuf_full && sky_ext_skip) as u32 * FLAG_SKY_EXT_SKIP)
            | ((crate::texture::max_aniso() > 1.0) as u32 * FLAG_ANISO)
            | (p.clouds.enabled as u32 * FLAG_CLOUDS)
            // count > 0 already folds in the session enable + the night fade
            // (fireflies.rs::new) — a day session never sets the bit, so day
            // kernels are bit-identical by construction.
            | ((p.fireflies.count > 0) as u32 * FLAG_FIREFLIES)
            // The V toggle read at CB-build time (height_max > 0 encodes
            // any_height from base()) — no FrameParams plumbing needed, and
            // the HEIGHTFIELD compile-in stays per-scene.
            | ((crate::bvh::height_on() && self.height_max > 0.0) as u32 * FLAG_HEIGHT)
            // The --no-depth-tint lever, read at CB-build time like the V
            // toggle — the branch lives inside shade.hlsli's transmission
            // arm, which non-transmissive scenes never enter.
            | (crate::scene::depth_tint() as u32 * FLAG_DEPTH_TINT)
            // Emissive cluster NEE: the scene derived clusters (el rows ride
            // the base) × the live lever × NOT a GI frame — under fb.gi the
            // hemi gather already delivers emissive transport exactly, so
            // the cluster tier stands down (the inverted once-per-path
            // rule). NEE STAYS LIVE under RTGI (the NEE-keep rule): the
            // bounce's emissive display-add suppresses on this very bit
            // instead (shade.hlsli's `cam_lights || !FLAG_EMISSIVE` gate),
            // so exactly one mechanism delivers per frame. Emissive-free
            // scenes never set the bit.
            | ((self.el_meta[0] > 0
                && crate::emissive::enabled()
                && fb_mode_of(&p.q) != 2) as u32
                * FLAG_EMISSIVE)
            // Real-time GI: the session lever (baked as the RTGI compile
            // define; this runtime bit covers the fb stand-down) × NOT a
            // hemi frame — the still-frame tiers take precedence, so
            // shade_full's bounce block keys on the bit alone.
            | ((p.q.rtgi && fb_mode_of(&p.q) == 0) as u32 * FLAG_RTGI)
            // The --no-detail-tex lever, read at CB-build time (the
            // depth-tint shape) — shade.hlsli's post-match detail block,
            // gated per material on Mat.detail_scale > 0 (untextured
            // materials carry the synthetic scale since the untextured arm).
            | (crate::scene::detail_tex() as u32 * FLAG_DETAIL)
            | (crate::scene::detail_ao() as u32 * FLAG_DETAIL_AO)
            | (crate::scene::spec_aa() as u32 * FLAG_SPEC_AA)
            | (crate::scene::amb_bump() as u32 * FLAG_AMB_BUMP)
            // FR_WAVEVIZ live toggle, read at CB-build time like the V
            // toggle — unarmed sessions compile no WAVEVIZ block, so the
            // bit is only ever consumed where the code exists.
            | ((waveviz_on() && waveviz_live()) as u32 * FLAG_WAVEVIZ);
        cb.shadow_samples = p.q.shadow_samples;
        cb.ao_samples = p.q.ao_samples;
        cb.reflections = p.q.reflections as u32;
        cb.frame_jitter = match p.frame_jitter {
            Some((x, y)) => [x, y],
            None => [0.0, 0.0],
        };
        cb.pixel_cone = p.cam.pixel_cone();
        // Firefly poses: the CPU's baked f32 rows verbatim (CPU↔GPU positions
        // bit-equal by DATA — the HLSL re-derives nothing). Rows past the
        // count stay the base's zeros.
        cb.ff_count = p.fireflies.count;
        cb.ff_scale = p.fireflies.scale;
        for i in 0..p.fireflies.count as usize {
            cb.ff[i] = p.fireflies.pos[i];
        }
        cb.fb_mode = fb_mode_of(&p.q);
        cb.fb_depth = p.q.fb.depth.clamp(1, HEMI_MAX_DEPTH);
        // --spp. Pinned to 1 on fb frames, exactly like FrameCtx::spp(): the
        // leaf pass appends one hemi point per PIXEL (cap_hemi_pt = rw*rh),
        // and N hemispheres per pixel is the wrong way to converge a bounce.
        cb.spp = if cb.fb_mode > 0 { 1 } else { p.spp.clamp(1, crate::dlss::MAX_SPP) };
        cb.probe_sample = p.probe_sample.min(cb.spp - 1);
        for k in 0..cb.spp {
            let (x, y) = crate::dlss::jitter_for_sample(p.frame, k);
            let (row, half) = ((k / 2) as usize, (k % 2) as usize * 2);
            cb.jitters[row][half] = x;
            cb.jitters[row][half + 1] = y;
        }
        if let Some(pc) = &p.prev_cam {
            // The near/far riding the w slots of the last two rows come from
            // the base and must survive the overwrite.
            let (po, pf, pr, pu, piw, pih) = pc.gpu_fields();
            cb.prev_origin = [po.x, po.y, po.z, piw];
            cb.prev_forward = [pf.x, pf.y, pf.z, pih];
            cb.prev_right = [pr.x, pr.y, pr.z, cb.prev_right[3]];
            cb.prev_up = [pu.x, pu.y, pu.z, cb.prev_up[3]];
        }
        cb
    }

    /// Arm the sway-MV correction for this frame: FLAG_SWAY_MV + the frame's
    /// dmv-ring slot base, one call so the pair cannot split (a flag without
    /// its base indexes slot 0's stale rows). Callers (both tracers'
    /// write_cb) gate on `sway_mv_pair` + the session's SWAY_MV compile-in +
    /// the slot fill having run.
    pub(crate) fn arm_sway_mv(&mut self, base: u32) {
        self.flags |= FLAG_SWAY_MV;
        self.sway_mv_base = [base, 0, 0, 0];
    }

    /// Copy into a persistently-mapped CB ring slot.
    pub(crate) fn store(&self, ptr: *mut u8) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                self as *const FrameCb as *const u8,
                ptr,
                std::mem::size_of::<FrameCb>(),
            )
        };
    }
}

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
    /// --spp: primary samples per pixel this frame (1..=dlss::MAX_SPP; the CB
    /// pins it to 1 when fb is on). The samples share the tile's inherited
    /// t_start and average into one partial write — accum semantics unchanged.
    pub spp: u32,
    /// Which sample writes tbuf/info/the G-buffer pack. 0 in every real frame;
    /// the check suites sweep it 0..spp so every sample's ray is gated.
    pub probe_sample: u32,
    /// Per-frame cloud state (src/clouds.rs) — enable bit + clock + diag,
    /// mapped onto FLAG_CLOUDS and the cam rows' w lanes by `with_frame`.
    pub clouds: crate::clouds::Clouds,
    /// Per-frame firefly state (src/fireflies.rs) — CPU-baked poses, mapped
    /// onto FLAG_FIREFLIES + the `ff`/`ff_count` CB rows by `with_frame`
    /// (count 0 — every day session — writes neither).
    pub fireflies: crate::fireflies::Fireflies,
    /// --foliage-sway clock for THIS frame (the shared cloud_time), or None =
    /// trace the static rest-pose TLAS. Consumed by BOTH ray pipelines since
    /// v0.2 (each rebuilds `SceneGpu::sway`'s ring TLAS on its list and
    /// binds it — DxrGpu via its sway_t stash, TraceGpu via `record_sway`);
    /// every headless gate/bench passes None — which, plus `sway: None` on
    /// unarmed uploads, is the structural off-state (src/foliage.rs).
    pub sway_time: Option<f32>,
    /// The PREVIOUS frame's sway clock, paired with `prev_cam`'s frame by
    /// main.rs (the PrevPose rule — set beside the camera after a successful
    /// present, cleared with it, so the pair cannot desync). Some + bit-
    /// different from `sway_time` + `prev_cam` Some arms the sway-MV
    /// correction (`sway_mv_pair`); None — every headless gate/bench/spin
    /// site — is the structural camera-only arm.
    pub sway_prev_time: Option<f32>,
    /// Structure-replay enable (opts.replay). When true AND this frame's basis
    /// bit-equals the previous producing frame's, `record_frame` re-dispatches
    /// the persisted terminal queues instead of re-running seed + the ladder
    /// (the GPU mirror of src/replay.rs). Replay frames re-shade fresh — the
    /// leaf shader's MV write included — so the sway-MV fill and CB arming
    /// run on them like any producing frame. NOT a global atomic: every
    /// headless gate/bench sets it false so nothing silently switches paths
    /// under a measurement.
    pub replay: bool,
}

/// Which upscaler the feed pass targets — selects the kernel (and thereby
/// the plane set and the u18 depth encoding). `Fsr3` is the XeSS feed over
/// FSR 3.1's planes: same three targets, same formats, same reversed-Z
/// depth encode — it compiles to `cs_feed_xess` and exists as its own kind
/// only so the wiring is explicit (and the NPPD composition stays
/// XeSS-only). `FsrRr` is the nine-plane Ray Regeneration + FSR4 feed
/// (`cs_feed_fsr_rr`) — the only kind that arms FLAG_FSR_SIG.
#[derive(Clone, Copy, PartialEq)]
pub enum FeedKind {
    Xess,
    Rr,
    Fsr3,
    FsrRr,
}

/// The kernel a feed kind runs. Shared by TraceGpu and DxrGpu, which hold the
/// same three PSOs and now both fan over a LIST of wired engines
/// (`--quinlight`), so the mapping had to stop being an inline match.
/// `nppd_dm` (XeSS + GPU-resident NPPD only) substitutes the depth+mvec variant
/// for the plain XeSS feed.
pub(crate) fn feed_pso<'a>(
    kind: FeedKind,
    nppd_dm: Option<&'a ID3D12PipelineState>,
    xess: Option<&'a ID3D12PipelineState>,
    rr: Option<&'a ID3D12PipelineState>,
    fsr_rr: Option<&'a ID3D12PipelineState>,
) -> Option<&'a ID3D12PipelineState> {
    match kind {
        // Fsr3 IS the XeSS feed (same planes, same encodings) — the kind exists
        // only so the wiring is explicit.
        FeedKind::Xess => nppd_dm.or(xess),
        FeedKind::Fsr3 => xess,
        FeedKind::Rr => rr,
        FeedKind::FsrRr => fsr_rr,
    }
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

/// GPU-resident NPPD staging: the four NCHW fp32 plane buffers (at the
/// /32-padded dims — `nppd::pad_dims`) that `nppd::NppdGpu` binds as ORT
/// tensors, plus the nppd.hlsl kernels that fill/consume them. Default-heap
/// raw buffers, ALLOW_UNORDERED_ACCESS, resting in UNORDERED_ACCESS — the
/// DML binding contract; they never transition.
pub struct NppdRes {
    pub frame: ID3D12Resource,
    pub state: ID3D12Resource,
    pub warped: ID3D12Resource,
    pub out: ID3D12Resource,
    pub pw: u32,
    pub ph: u32,
    pso_pack: ID3D12PipelineState,
    pso_warp: ID3D12PipelineState,
    pso_zero: ID3D12PipelineState,
    pso_out: ID3D12PipelineState,
    pso_feed_dm: ID3D12PipelineState,
}

/// The --nrd bridge stage: the two kernels flanking NRD's own passes
/// (nrd_bridge.hlsl). PSOs only — the NRD plane TEXTURES belong to the
/// session's NrdGpu (gpu/nrd_gpu.rs) or a gate's locals, wired into
/// descriptor set NRD_FEED_SET via `wire_nrd_feed` (the NppdRes shape,
/// minus buffers). The old guides-only engine feed PSO is gone — the FOLD
/// moved cs_feed_xess_dm's stores into cs_nrd_pack itself.
pub struct NrdRes {
    // pub(crate): DxrGpu's record twins bind the same PSOs (compiled with
    // ITS root-signature object — same layout, the feed_pso discipline).
    pub(crate) pso_pack: ID3D12PipelineState,
    pub(crate) pso_out: ID3D12PipelineState,
}

impl NrdRes {
    /// DxrGpu's constructor — it compiles the same nrd_bridge.hlsl unit at
    /// its own cs_6_3 floor with its own root-signature object.
    pub(crate) fn from_psos(
        pso_pack: ID3D12PipelineState,
        pso_out: ID3D12PipelineState,
    ) -> Self {
        Self { pso_pack, pso_out }
    }
}

pub struct TraceGpu {
    pub root_sig: ID3D12RootSignature,
    pub cmd_sig: ID3D12CommandSignature,
    pso_reference: ID3D12PipelineState,
    pso_resolve: ID3D12PipelineState,
    pso_seed: ID3D12PipelineState,
    /// The replay seed (cs_seed_replay): zeroes every counter EXCEPT the
    /// terminal counts, so a bit-equal-basis frame re-dispatches the persisted
    /// leaf/sky queues without re-running the ladder.
    pso_seed_replay: ID3D12PipelineState,
    /// The basis whose terminal structure the queues currently hold — set by
    /// EVERY record_wavefront (incl. verify/check callers), cleared by
    /// invalidate_replay (aborts, hemi-probe seeds) and by TraceGpu recreation.
    /// `Cell` because the record_* methods take `&self`. record_frame replays
    /// only when `p.replay && last_struct == Some((p.cam, self.split))`.
    ///
    /// The SPLIT is in the key, not just the basis: under `--dual-gpu` the
    /// terminal queues describe the tiles this device owned when it recorded
    /// them, so a rebalance must force a fresh trace. Keying on the basis
    /// alone would re-dispatch the old assignment's leaves and leave the
    /// reassigned tiles unwritten — a hole in the image, the false-sky class.
    last_struct: std::cell::Cell<Option<(CamBasis, TileSplit)>>,
    /// Which level-`depth` tiles this device renders (`--dual-gpu`).
    /// `TileSplit::ALL` in every single-GPU session, which is the state in
    /// which the ownership test is branched around entirely.
    split: std::cell::Cell<TileSplit>,
    /// Whether the last `set_split` was refused, so a refusal that repeats
    /// prints ONCE. `set_split` is called every frame from `record_trace`, and
    /// every refusal condition (a depth past the leaf frontier at a small
    /// render resolution, `--waveviz`) holds for as long as that condition
    /// does — an unlatched `eprintln!` there is two lines per frame at
    /// whatever the frame rate is, which buries the one line that mattered.
    split_refused: std::cell::Cell<bool>,
    /// --foliage-sway clock for the frame being recorded (the DxrGpu shape,
    /// but set by record_wavefront/record_wavefront_replay from their OWN
    /// FrameParams — never write_cb, so the check/bench sites that skip
    /// write_cb can't read a stale value). bind_common picks the animated
    /// ring slot's TLAS when (this, scene.sway) are both Some; the reference
    /// kernel shares bind_common, so an R/C comparison is same-TLAS by
    /// construction.
    sway_t: std::cell::Cell<Option<f32>>,
    /// Whether SWAY_MV compiled into this tracer's kernels (ring armed AND
    /// not --sw-rays — see `sway_defs`). The write_cb/record_sway arming
    /// gate: a flag the kernels never read is one thing, a flag armed
    /// without the compile-in is a stale-slot-0 read.
    sway_mv_on: bool,
    /// The live SKY_SPLIT (see the const): the shader define and the
    /// multiplying prep's push constant MUST be the same number, so it is read
    /// once at kernel assembly and stored, never re-derived at the dispatch.
    sky_split: u32,
    pso_prep: ID3D12PipelineState,
    /// prep-args, multiplying flavor (groups = counter * push2) — the sky fill,
    /// where SKY_SPLIT groups cooperate on each record.
    pso_prep_mul: ID3D12PipelineState,
    pso_clear_info: ID3D12PipelineState,
    pso_level: ID3D12PipelineState,
    /// The wave-cooperative level kernel (one GROUP per tile) used for the
    /// shallow levels — see WIDE_LEVELS.
    pso_level_wide: ID3D12PipelineState,
    /// The ladder as a work graph (`FR_WORKGRAPH=1`). None on every ordinary
    /// session, and also whenever the runtime/driver lacks work graphs or the
    /// state object fails to build — a spike must degrade to the shipping
    /// ladder with one loud line, never take the session down.
    work_graph: Option<WorkGraph>,
    pso_sky: ID3D12PipelineState,
    /// The amortized cloud lattice pass (sky.hlsl); only dispatched at SKY_LOD > 1.
    pso_sky_lod: ID3D12PipelineState,
    pso_cloud_shadow: ID3D12PipelineState,
    /// (scatter.rgb, transmittance) at 1/SKY_LOD pixel pitch, bound at u5 for
    /// the sky passes — the tile queue's register, dead by then.
    pub(crate) cloud_lod: ID3D12Resource,
    /// Slab-space cloud shadow cache (u6 during shading — the qout register,
    /// dead by then and suppressed in the shading units by SKY_UNIT). pub(crate)
    /// so --check-gpu reads it back for the fill-vs-oracle gate.
    pub(crate) cloud_shadow: ID3D12Resource,
    /// Scene AABB (content box unioned with the ground quad), for the per-frame
    /// slab-space grid extent.
    scene_aabb: ([f32; 3], [f32; 3]),
    /// The cloud-cache levers SNAPSHOTTED at construction (see CLOUD_SHADOW /
    /// SKY_LOD): the kernels are compiled and the buffers sized against these,
    /// so every per-frame record path must read the field — never the live
    /// static, which a mid-process A/B (two TraceGpu instances) can have flipped
    /// out from under a kernel that doesn't compile the fill.
    cloud_shadow_n: u32,
    sky_lod_k: u32,
    pso_leaf: ID3D12PipelineState,
    /// The same kernel with the hemi arm compiled IN — used only by fb frames
    /// (H). See leaf.hlsl's LEAF_NO_FB note: keeping the two apart is what
    /// keeps the common path's VGPR count (and so RDNA's occupancy) low.
    pso_leaf_fb: ID3D12PipelineState,
    pso_clear_h: ID3D12PipelineState,
    pso_prep_batch: ID3D12PipelineState,
    pso_seed_probes: ID3D12PipelineState,
    pso_hemi_root: ID3D12PipelineState,
    pso_hemi_cell: ID3D12PipelineState,
    pso_hemi_leaf: ID3D12PipelineState,
    pso_compose: ID3D12PipelineState,
    pso_feed_xess: Option<ID3D12PipelineState>,
    pso_feed_rr: Option<ID3D12PipelineState>,
    pso_feed_fsr_rr: Option<ID3D12PipelineState>,
    /// The wired upscaler feed targets (wire_feed): plane resources cloned for
    /// record_feed's barriers, plus which feed kernel consumes them. ONE entry
    /// per wired engine — normally exactly one, several under `--quinlight`.
    /// The INDEX is the engine's descriptor set (see FEED_SETS): its planes'
    /// UAVs live at that set's slots, and its feed dispatch binds RP_TEX there.
    feed: Vec<(FeedKind, Vec<ID3D12Resource>)>,
    /// Kept for the per-set descriptor-table handles record_feed computes.
    device: ID3D12Device,
    /// GPU-resident NPPD staging (the --gpu --nppd composition) — buffers
    /// nppd::NppdGpu wraps as ORT tensors, plus the staging kernels.
    pub nppd: Option<NppdRes>,
    /// The --nrd bridge kernels (pack/out around NRD's passes) — built when
    /// the session (or a gate) asks; wire_nrd_feed points set NRD_FEED_SET
    /// at the frame's NRD planes.
    pub nrd: Option<NrdRes>,
    /// Keeps the wired NRD planes alive for the descriptors' lifetime (the
    /// wire_feed discipline — descriptors don't own).
    nrd_wired: Vec<ID3D12Resource>,
    /// The wired NRD set's u16 target — the ENGINE's color plane, which RESTS
    /// in NON_PIXEL_SHADER_RESOURCE (the upscaler-eval contract) and must be
    /// bracketed NPSR→UA→NPSR around cs_nrd_out's write. Held separately from
    /// `nrd_wired` because record_nrd_out needs the RESOURCE, not a keepalive.
    nrd_color: Option<ID3D12Resource>,
    /// The wired NRD set's u18/u19 targets — the ENGINE's depth/mvec guide
    /// planes the FOLDED cs_nrd_pack writes (record_feed_nrd's retired job).
    /// Same NPSR rest-state contract as `nrd_color`, bracketed around the
    /// pack dispatch; the bridge's own IN planes rest UA and are never
    /// transitioned (the NrdGpu pool doctrine).
    nrd_guides: Vec<ID3D12Resource>,
    /// The shared scene core — one Rc per tracer, cached in GpuContext (the
    /// second tracer and resize re-entries skip the upload + BLAS build).
    pub scene: std::rc::Rc<SceneGpu>,
    /// The wavefront-only software trees (binary BVH + optional wide tree) —
    /// per-TraceGpu, deliberately outside the shared core (DXR never binds
    /// them).
    sw: SwTreesGpu,
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
    /// The G-buffer pack's CORE half (`GBufCore`, GBUF_STRIDE = 16 B/px) —
    /// full-size in upscaler sessions, a stride-sized dummy otherwise
    /// (`gbuf_full` gates FLAG_GBUF, which is what keeps the write helpers
    /// from scribbling past the dummy).
    pub gbuf: ID3D12Resource,
    /// The pack's guide/signal half (`GBufExt`, GBUF_EXT_STRIDE = 72 B/px).
    /// Allocated whenever `gbuf` is, but WRITTEN only under FLAG_GBUF_EXT —
    /// see that flag for why the split exists and what it measured.
    pub gbuf_ext: ID3D12Resource,
    /// See `force_fsr_sig` — the `--dual-gpu` secondary's only way to store the
    /// signal lanes its partner's feed will read.
    force_fsr_sig: std::cell::Cell<bool>,
    /// See `force_nrd_sig` — the secondary's half of the NRD RTGI fold (and
    /// the check-gpu fold gate's arming hook). An OVERRIDE, not an OR: the
    /// N6 gate must force the fold OFF on a tracer whose NRD planes the
    /// earlier bridge gates already wired (Some(false) beats the wiring
    /// term), and None restores wiring-derived behavior.
    force_nrd_sig: std::cell::Cell<Option<bool>>,
    /// Test hook — see `force_gbuf_ext`. `Cell` because the record/write
    /// methods take `&self` (the `last_struct` precedent).
    force_gbuf_ext: std::cell::Cell<bool>,
    /// See `sky_ext_skip` — the armed-skip gate's hook (the `force_nrd_sig`
    /// Option-override shape: Some beats the derivation, None restores it).
    force_sky_ext_skip: std::cell::Cell<Option<bool>>,
    gbuf_full: bool,
    uav_heap: ID3D12DescriptorHeap,
    /// GPU handle of the RP_SCENE_TEX table's first descriptor (heap slot
    /// TEX_HEAP_BASE) — bound by bind_common for every trace dispatch.
    tex_table: D3D12_GPU_DESCRIPTOR_HANDLE,
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
        scene_gpu: std::rc::Rc<SceneGpu>,
        rw: u32,
        rh: u32,
        gbuf_full: bool,
        nppd: bool,
        nrd: bool,
        debug: bool,
        sub: &mut dyn d3d12::Submit,
    ) -> Result<Self> {
        require_caps(device)?;
        abl_announce();
        // The vendor of THIS device — never `picked_vendor()`, which is a
        // process-global and under --dual-gpu names whichever device was picked
        // last. Every vendor-keyed decision below (the AMD candidate-TMin
        // workaround, the Intel work-graph refusal) is a property of the adapter
        // these kernels will run on, so it is derived from the device itself.
        let vendor = adapter::vendor_of_device(device);
        let root_sig = create_root_signature(device)?;
        let cmd_sig = create_dispatch_signature(device)?;

        // Alpha-masked scenes compile the cutout candidate loops into the
        // trace primitives (rt.hlsli); height-carrying scenes likewise
        // compile the relief march in (runtime-gated by FLAG_HEIGHT — the V
        // toggle); transmissive scenes compile transmit_q's tinted candidate
        // loop in (TRANS_SHADOW). Scenes with none compile the FORCE_OPAQUE
        // originals verbatim (modulo leading blank lines) — procedural/stress
        // sessions are structurally untouched (the bit gates rely on that).
        // Dev cost-attribution ablations ride `abl_defs()`, which dxr.rs pastes
        // too so the two arms stay comparable.
        let empty_def = empty_defs(scene);
        // SWAY_MV: suppressed under --sw-rays — the software rays render the
        // REST pose, so sway MVs would describe motion that is not on screen
        // (sway_defs' doc). DXR takes the same predicate verbatim.
        //
        // The argument is the UPLOADED ring's existence, read off `SceneGpu`
        // here rather than re-derived from the partition/lever/split chain:
        // the define and the resource it describes have to be ONE decision
        // (the `non_opaque` discipline), and the assembly is portable now, so
        // the fact travels to it as a bool instead of the backend type.
        let sway_def = if sw_rays() { "" } else { sway_defs(scene_gpu.sway.is_some()) };
        let defs = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            empty_def,
            alpha_defs(scene),
            height_defs(scene),
            trans_defs(scene),
            cand_defs(vendor),
            blas_defs(),
            sway_def,
            abl_defs(),
            rtgi_defs()
        );
        let defs = defs.as_str();
        // The session's frustum structure: `#define FTREE` swaps frustum.hlsli's
        // binary bound_query/refine_cut for ftree.hlsli's wide bodies (same
        // signatures — the call sites don't know), and the FNode array uploads
        // at t0 in place of the binary nodes. --no-ftree keeps the binary path.
        let ftree_on = crate::ftree::FTREE_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let ft_defs = if ftree_on { "#define FTREE 1" } else { "" };
        // The per-lane frustum stack depth — the tracer's ONLY groupshared, and
        // therefore the only thing that can cap resident GROUPS in the level and
        // hemi kernels, which RGA shows are nowhere near VGPR-limited
        // (cs_level_wide 54 VGPR against cs_leaf's 216). Injected into exactly
        // the units that paste frustum.hlsli; the work graph inherits it through
        // wavefront_src. See lane_stack() before sweeping it.
        let ls_defs = format!("#define LANE_STACK {}u{}", lane_stack(), stack_layout_def());
        let ls_defs = ls_defs.as_str();
        // The cbuffer's jitter-table size (--spp) — every unit sees the cbuffer.
        let sd = spp_defs();
        let sd = sd.as_str();
        // The detail strength knobs — every unit that pastes SHADE_HLSLI.
        let dd = detail_defs();
        let dd = dd.as_str();
        // --sw-rays: rt_sw.hlsli pastes in place of rt.hlsli in every
        // ray-shooting unit (leaf x2, reference, hemi_wave, hemi_leaf); the
        // wavefront unit gets the SW_RAYS define too (level_finish's leaf-cut
        // translation arm). SW_TRAV_STACK rides in from bvh::TRAV_STACK so
        // the HLSL stacks stay in lockstep with the build's max_depth assert.
        let sw_on = sw_rays();
        let rt_src: &str = if sw_on { RT_SW_HLSLI } else { RT_HLSLI };
        let sw_defs = if sw_on {
            format!(
                "#define SW_RAYS 1\n#define SW_TRAV_STACK {}u",
                crate::bvh::TRAV_STACK
            )
        } else {
            String::new()
        };
        let sw_defs = sw_defs.as_str();
        // The leaf unit's cut consumption composes with --no-cut-rays exactly
        // as the CPU's intersect_multi short-circuit does: software traversal
        // from the root, the scalar t_start kept. The wavefront unit shares
        // the define (level_finish's leaf-cut translation compiles only when
        // the leaf actually consumes it).
        let sw_leaf_defs = if sw_rays_leaf() { "#define SW_RAYS_LEAF 1" } else { "" };
        // Snapshot the cloud-cache levers ONCE: the kernels below are compiled
        // against these values and the buffers sized against them, and every
        // per-frame record path reads the stored fields — so a mid-process A/B
        // that flips the static between two constructions can never desync a
        // kernel from its fill dispatch (the device-hang class record_cloud_shadow
        // documents).
        let cloud_shadow_v = cloud_shadow_n();
        let sky_lod_v = sky_lod();
        // The cloud-shadow cache is compiled into every unit that shades (leaf,
        // reference) plus the unit that fills it (sky). wavefront/hemi get 0 and
        // keep the exact per-pixel expression — they must not declare u6, which
        // is the tile queue there. (DXR compiles it in through its own assembly.)
        let csn = format!("#define CLOUD_SHADOW_N {cloud_shadow_v}");
        // The sky-lod lattice defines, shared by the sky-fill unit, both leaf
        // kernels, AND the reference kernel — so reference's sky pixels compose
        // through the identical `sky_radiance_lod` and the exact-zero
        // wavefront-vs-reference image A/B stays bit-identical at the default-ON
        // K. SKY_UNIT is a harmless no-op in a unit that pastes no queues.hlsli
        // (reference), where u5 is free anyway.
        let sky_lod_defs = format!(
            "#define SKY_UNIT 1\n#define SKY_LOD {sky_lod_v}\n#define SKY_LOD_LOG {}",
            sky_lod_v.trailing_zeros()
        );
        // The reference kernel swaps to rt_sw with the wavefront: the
        // exact-zero wavefront-vs-reference gates require ONE intersector on
        // both sides (the "same intersector, same seeds" contract). It also
        // reads the cloud lattice (SKYLOD_HLSLI at u5, filled by record_sky_lod)
        // and the cloud-shadow cache (csn, filled by record_cloud_shadow), so it
        // shades sky exactly as the leaf kernel does.
        // WIDTH_PROBE defines, pushed CONDITIONALLY into every unit below —
        // never as an empty join element (the feed-unit byte-identity rule):
        // unarmed assemblies must be byte-identical to the pre-lever sources.
        let wd = width_defs();
        let bd = ballast_defs();
        let wv = waveviz_defs();
        let mut reference_parts: Vec<&str> = Vec::new();
        if !wd.is_empty() {
            reference_parts.push(wd.as_str());
        }
        if !bd.is_empty() {
            reference_parts.push(bd.as_str());
        }
        if !wv.is_empty() {
            reference_parts.push(wv.as_str());
        }
        reference_parts.extend([
            csn.as_str(),
            sky_lod_defs.as_str(),
            defs,
            sw_defs,
            sd,
            dd,
            TRACE_COMMON_HLSLI,
            SKYLOD_HLSLI,
            rt_src,
            RIPPLE_HLSLI,
            SHADE_HLSLI,
            REFERENCE_HLSL,
        ]);
        let reference_src = reference_parts.join("\n");
        // The waveviz overlay deliberately does NOT live in this resolve
        // unit any more: it composites at the present funnel (tonemap.rs's
        // waveviz PSO), which is what makes it work under every upscaler —
        // the resolve runs only on the plain arms.
        let resolve_src = [sd, TRACE_COMMON_HLSLI, RESOLVE_HLSL].join("\n");
        let (sky_group, sky_split) = (SKY_GROUP, SKY_SPLIT);
        let sky_defs = format!(
            "#define SKY_GROUP {sky_group}\n#define SKY_SPLIT {sky_split}\n#define LEAF_TILE {}",
            crate::render::leaf_tile()
        );
        let wavefront_ablation_defs = wavefront_ablation_defs();
        let mut wavefront_parts: Vec<&str> = Vec::new();
        if !wd.is_empty() {
            wavefront_parts.push(wd.as_str());
        }
        wavefront_parts.extend([
            sky_defs.as_str(),
            wavefront_ablation_defs.as_str(),
            empty_def,
            ft_defs,
            ls_defs,
            sw_defs,
            sw_leaf_defs,
            sd,
            TRACE_COMMON_HLSLI,
            CTR_HLSLI,
            QUEUES_HLSLI,
            FRUSTUM_HLSLI,
            FTREE_HLSLI,
            WAVEFRONT_HLSL,
        ]);
        let wavefront_src = wavefront_parts.join("\n");
        // The sky fill is its own unit so `cloud_lod` can take u5 (SKY_UNIT
        // suppresses queues.hlsli's tile-queue declarations there). Assembled by
        // the shared `sky_unit_src` so the DXR pipeline's fill kernels cannot
        // drift from these.
        let sky_src = sky_unit_src(sky_lod_v, cloud_shadow_v);
        // Two leaf kernels from the one source. `fb_mode` is a cbuffer value,
        // so leaving the hemi arm as a runtime branch inlines shade_split at
        // both call sites and the kernel's register allocation is the MAX of
        // the two — which on RDNA costs occupancy (and therefore latency
        // hiding) in every fb-OFF frame, i.e. essentially all of them.
        // `LEAF_NO_FB` compiles that arm out; record_wavefront picks per frame.
        // The leaf kernel shades the sky pixels inside leaf tiles, so it reads
        // the same lattice: SKY_UNIT yields u5 (it never touches the tile
        // queues) and skylod.hlsli supplies the accessors.
        let leaf_of = |extra: &str| {
            let lg = format!("#define LEAF_GROUP {}", leaf_group());
            let mut parts: Vec<&str> = Vec::new();
            if !wd.is_empty() {
                parts.push(wd.as_str());
            }
            if !wv.is_empty() {
                parts.push(wv.as_str());
            }
            parts.extend([
                lg.as_str(),
                extra,
                csn.as_str(),
                sky_lod_defs.as_str(),
                defs,
                sw_defs,
                sw_leaf_defs,
                sd,
                dd,
                TRACE_COMMON_HLSLI,
                CTR_HLSLI,
                QUEUES_HLSLI,
                SKYLOD_HLSLI,
                rt_src,
                RIPPLE_HLSLI,
                SHADE_HLSLI,
                LEAF_HLSL,
            ]);
            parts.join("\n")
        };
        let leaf_src = leaf_of("#define LEAF_NO_FB 1");
        let leaf_fb_src = leaf_of("");
        // Hemi kernels stay on the BINARY tree deliberately (no ft_defs):
        // hemi bound queries terminate in ~10 visits, where a wide pop's
        // unconditional 8 slot tests lose to the binary pop's 1 — measured
        // +35% ms on the hemi-gi bench with the wide tree, against -54% on
        // the tile path. record_hemi rebinds the binary buffer at t0.
        let hemi_wave_src = [
            defs,
            ls_defs,
            sw_defs,
            sd,
            TRACE_COMMON_HLSLI,
            CTR_HLSLI,
            HEMI_HLSLI,
            FRUSTUM_HLSLI,
            rt_src,
            HEMI_WAVE_HLSL,
        ]
        .join("\n");
        let mut hemi_leaf_parts: Vec<&str> = Vec::new();
        if !wd.is_empty() {
            hemi_leaf_parts.push(wd.as_str());
        }
        hemi_leaf_parts.extend([
            defs,
            sw_defs,
            sd,
            dd,
            TRACE_COMMON_HLSLI,
            CTR_HLSLI,
            HEMI_HLSLI,
            rt_src,
            RIPPLE_HLSLI,
            SHADE_HLSLI,
            HEMI_LEAF_HLSL,
        ]);
        let hemi_leaf_src = hemi_leaf_parts.join("\n");
        let compose_src = [sd, TRACE_COMMON_HLSLI, CTR_HLSLI, QUEUES_HLSLI, COMPOSE_HLSL].join("\n");
        // abl_defs FIRST so a feed ablation is not silently inert. It was:
        // an `FR_ABL=nopack` probe reported `feed` unchanged and that was read
        // as "the pack read is free" — but the define never reached this unit,
        // so the probe compared identical code against itself. The shipping
        // split then measured feed 0.544 -> 0.231 ms, i.e. the read very much
        // is not free. An ablation that cannot reach its target is worse than
        // no ablation, because it answers confidently.
        let feed_src =
            [abl_defs().as_str(), sd, TRACE_COMMON_HLSLI, FSR_WIRE_HLSLI, FEED_HLSL].join("\n");
        let pso = |src: &str, entry: &str, what: &str| -> Result<ID3D12PipelineState> {
            compute_pso(device, &root_sig, &dxc.compile(src, entry, "cs_6_5", what, debug)?, what)
        };
        let pso_reference = pso(&reference_src, "cs_reference", "reference")?;
        let pso_resolve = pso(&resolve_src, "cs_resolve", "resolve")?;
        let pso_seed = pso(&wavefront_src, "cs_seed", "seed")?;
        let pso_seed_replay = pso(&wavefront_src, "cs_seed_replay", "seed_replay")?;
        let pso_prep = pso(&wavefront_src, "cs_prep", "prep")?;
        let pso_prep_mul = pso(&wavefront_src, "cs_prep_mul", "prep_mul")?;
        let pso_clear_info = pso(&wavefront_src, "cs_clear_info", "clear_info")?;
        let pso_level = pso(&wavefront_src, "cs_level", "level")?;
        let pso_level_wide = pso(&wavefront_src, "cs_level_wide", "level_wide")?;
        // The work-graph ladder (FR_WORKGRAPH=1). Same pasted sources as the
        // ladder plus its node shaders, with WORKGRAPH switching level_finish's
        // child emission from `qout` to out-params — the tile logic is NOT
        // forked. Any failure here (no OPTIONS21 support, no lib_6_8, a state
        // object the driver rejects) falls back to the ladder with one loud
        // line: a spike must never cost a session.
        let work_graph = if work_graph_on() {
            let caps = query_caps(device)?;
            // MEASURED REFUSAL, not a guess, and the first real use of
            // `adapter::picked_vendor()`. Intel driver 32.0.101.8515 reports
            // WorkGraphsTier 1.0 and builds the state object happily — then
            // takes an access violation inside DispatchGraph, with the debug
            // layer AND GPU-based validation silent. The identical graph runs
            // on NVIDIA and passes every gate bit-identically, so this is the
            // driver, not the shader. The backing-memory ask is the corroborating
            // tell: 517 MB on Arc against 82 MB on NVIDIA for the same graph.
            // RE-TESTED 2026-08-01 on 32.0.101.8805 (arm deleted locally, the
            // procedure below): the IDENTICAL AV — exit 0xC0000005 at the
            // first graph dispatch of --check-gpu, backing ask still
            // 517.62 MB, state object still builds. Re-test on a driver newer
            // than 8805 and delete this arm if it passes; keying on THIS
            // DEVICE's adapter (a fact) rather than --prefer-* (a request that
            // can fall back) is the vendor_defaults rule — and under
            // --dual-gpu the per-device form is the only correct one, since
            // the Intel device must refuse while its NVIDIA partner does not.
            if caps.work_graphs_tier == 0 {
                eprintln!(
                    "gpu work-graph: FR_WORKGRAPH=1 but the device reports no work-graph \
                     support — running the ExecuteIndirect ladder"
                );
                None
            } else if vendor == adapter::Vendor::Intel {
                eprintln!(
                    "gpu work-graph: FR_WORKGRAPH=1 refused on Intel — drivers 8515 AND 8805 \
                     report WorkGraphsTier 1.0 but fault inside DispatchGraph (the same graph \
                     passes every gate on NVIDIA). Running the ExecuteIndirect ladder; retry on \
                     a newer driver by deleting this arm in trace.rs"
                );
                None
            } else {
                // Levels 0..WIDE_LEVELS run one group per tile, the rest one
                // thread per tile — the ladder's own split. Each node recurses
                // (levels - 1) times; declared depths are clamped to >= 1
                // because 0 means "not recursive at all" to the compiler, which
                // would make the node's self-output an illegal cycle.
                let dfull = depth_full(rw, rh);
                let wl = if WIDE_LEVELS_ON.load(std::sync::atomic::Ordering::Relaxed) {
                    wide_levels()
                } else {
                    0
                };
                let wide = dfull.min(wl).max(1);
                // A node declaring N may recurse N times, i.e. run N+1 levels,
                // so the wide node hands off after `wide` launches. DEEP is
                // deliberately GENEROUS (the whole depth): over-declaring only
                // reserves more backing memory, whereas under-declaring makes
                // the deepest node drop its children — which the shader counts
                // into CTR_OVERFLOW, so it fails a gate rather than corrupting
                // an image, but it is still a failure worth not courting. Both
                // are >= 1 because 0 reads as "not recursive" to the compiler
                // and would make the node's self-output an illegal cycle.
                let wg_src = format!(
                    "#define WORKGRAPH 1\n#define WG_WIDE_LEVELS {}\n#define WG_DEEP_LEVELS {}\n{}\n{}",
                    wide.saturating_sub(1).max(1),
                    dfull.saturating_sub(1).max(1),
                    wavefront_src,
                    WORKGRAPH_HLSL
                );
                match WorkGraph::create(device, &root_sig, dxc, &wg_src, debug) {
                    Ok(g) => Some(g),
                    Err(e) => {
                        eprintln!("gpu work-graph: {e} — running the ExecuteIndirect ladder");
                        None
                    }
                }
            }
        } else {
            None
        };
        let pso_sky = pso(&sky_src, "cs_sky", "sky")?;
        let pso_sky_lod = pso(&sky_src, "cs_sky_lod", "sky_lod")?;
        let pso_cloud_shadow = pso(&sky_src, "cs_cloud_shadow", "cloud_shadow")?;
        let pso_leaf = pso(&leaf_src, "cs_leaf", "leaf")?;
        let pso_leaf_fb = pso(&leaf_fb_src, "cs_leaf", "leaf-fb")?;
        let pso_clear_h = pso(&wavefront_src, "cs_clear_h", "clear_h")?;
        let pso_prep_batch = pso(&wavefront_src, "cs_prep_batch", "prep_batch")?;
        let pso_seed_probes = pso(&wavefront_src, "cs_seed_probes", "seed_probes")?;
        let pso_hemi_root = pso(&hemi_wave_src, "cs_hemi_root", "hemi_root")?;
        let pso_hemi_cell = pso(&hemi_wave_src, "cs_hemi_cell", "hemi_cell")?;
        let pso_hemi_leaf = pso(&hemi_leaf_src, "cs_hemi_leaf", "hemi_leaf")?;
        let pso_compose = pso(&compose_src, "cs_compose", "compose")?;
        // Feed kernels exist only when the pack is full-size (an upscaler
        // session); plain sessions never record a feed.
        let (pso_feed_xess, pso_feed_rr, pso_feed_fsr_rr) = if gbuf_full {
            (
                Some(pso(&feed_src, "cs_feed_xess", "feed_xess")?),
                Some(pso(&feed_src, "cs_feed_rr", "feed_rr")?),
                Some(pso(&feed_src, "cs_feed_fsr_rr", "feed_fsr_rr")?),
            )
        } else {
            (None, None, None)
        };
        // NRD bridge kernels: --nrd sessions + the check-gpu bridge gates.
        // abl_defs FIRST (the probe-reach lesson — an ablation define that
        // cannot reach its unit answers confidently).
        let nrd_res = if gbuf_full && nrd {
            let nrd_src =
                [abl_defs().as_str(), sd, TRACE_COMMON_HLSLI, FSR_WIRE_HLSLI, NRD_BRIDGE_HLSL]
                    .join("\n");
            Some(NrdRes {
                pso_pack: pso(&nrd_src, "cs_nrd_pack", "nrd_pack")?,
                pso_out: pso(&nrd_src, "cs_nrd_out", "nrd_out")?,
            })
        } else {
            None
        };
        // NPPD staging kernels: only in --gpu --nppd (XeSS) sessions.
        let nppd_psos = if gbuf_full && nppd {
            let nppd_src = [sd, TRACE_COMMON_HLSLI, NPPD_HLSL].join("\n");
            Some((
                pso(&nppd_src, "cs_nppd_pack", "nppd_pack")?,
                pso(&nppd_src, "cs_nppd_warp", "nppd_warp")?,
                pso(&nppd_src, "cs_nppd_zero", "nppd_zero")?,
                pso(&nppd_src, "cs_nppd_out", "nppd_out")?,
                pso(&feed_src, "cs_feed_xess_dm", "feed_xess_dm")?,
            ))
        } else {
            None
        };

        // Built here, uploaded, dropped — the GPU session needs no CPU copy
        // (CPU hemi has its own lazy global; a --gpu session never runs it).
        let ft = ftree_on.then(|| {
            let t0 = std::time::Instant::now();
            let ft = crate::ftree::FTree::build(bvh);
            eprintln!(
                "gpu ftree: {} wide nodes ({} MB quantized) collapsed in {:.0} ms (tile kernels bind it at t0; hemi stays binary)",
                ft.nodes.len(),
                ft.quantized_bytes() >> 20,
                t0.elapsed().as_secs_f64() * 1000.0
            );
            ft
        });
        // The shared core arrived pre-uploaded (Rc from GpuContext's cache);
        // only the wavefront's own software trees upload here.
        let sw = SwTreesGpu::new_uploaded(device, bvh, ft.as_ref(), sub)?;
        drop(ft);

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
        // --sw-rays + FTREE: level_finish translates each leaf-emitting
        // split's slot-ref cut into a SECOND fresh slot of binary node ids —
        // at most one extra slot per split, so doubling keeps the pool
        // structurally overflow-free (CTR_OVERFLOW stays gated == 0; the
        // exhaustion arm degrades to root seeding, counted, never wrong).
        let cap_cut = if sw_rays_leaf() && ftree_on { cap_cut * 2 } else { cap_cut };
        // continuation.hlsli packs slot<<6 | (len-1) and reserves all-ones
        // for root. This is enormously above the structural depth-11 cap,
        // but keep the opaque provider's wire proof beside the allocation.
        assert!(
            cap_cut < (1u64 << 26),
            "cut arena exceeds the traversal-frontier token domain"
        );
        // CTR_TOTAL, not CTR_COUNT: the tail holds the WIDTH_PROBE slots.
        // Unconditional — 20 bytes buys a lever-independent buffer shape (an
        // FR_WIDTH session and a plain one bind identical resources).
        let counters = committed_buffer(device, CTR_TOTAL as u64 * 4, uaf, ua)?;
        let args = committed_buffer(device, 16 * 12, uaf, ua)?;
        let qa = committed_buffer(device, cap_tile * 24, uaf, ua)?;
        let qb = committed_buffer(device, cap_tile * 24, uaf, ua)?;
        let qleaf = committed_buffer(device, cap_leaf * LEAF_REC_BYTES, uaf, ua)?;
        let qsky = committed_buffer(device, cap_sky * 16, uaf, ua)?;
        let cut_pool = committed_buffer(device, cap_cut * 256, uaf, ua)?;
        // The amortized cloud lattice (sky.hlsl): one float4 per lattice point,
        // one point of border past each far edge. 2.1 MB at 1080p/K=4; a single
        // float4 at K=1, where the kernel compiles the lattice out entirely.
        let lw = (rw >> sky_lod_v.trailing_zeros()) as u64 + 2;
        let lh = (rh >> sky_lod_v.trailing_zeros()) as u64 + 2;
        let cloud_lod = committed_buffer(device, (lw * lh).max(1) * 16, uaf, ua)?;
        // Slab-space cloud shadow cache: N*N scalar transmittances. 16 KB at
        // N=64 — the field's finest feature is 1.3*diag (wider than the scene),
        // so a coarse grid is nearly exact. See cloud_shadow_n.
        let scene_aabb = scene_shadow_aabb(scene);
        // Sized at the cap (1 MB): the side is derived per frame, so the
        // allocation cannot track it.
        let csn_n =
            if cloud_shadow_v > 0 { crate::clouds::CLOUD_SHADOW_MAX as u64 } else { 1 };
        let cloud_shadow = committed_buffer(device, csn_n * csn_n * 4, uaf, ua)?;

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
        // only in upscaler sessions — plain sessions bind a stride-sized
        // dummy and never set FLAG_GBUF (root UAVs have no bounds check).
        let gbuf =
            committed_buffer(device, if gbuf_full { px * GBUF_STRIDE } else { GBUF_STRIDE }, uaf, ua)?;
        // The guide/signal half. Allocated full-size whenever the pack is
        // full-size, NOT only when a guide-consuming kind is wired: the kind
        // arrives after construction (and `--check-gpu` rewires one tracer
        // across all four), so the buffer must always be able to receive the
        // stores. FLAG_GBUF_EXT gates the WRITES, which is where the cost is.
        let gbuf_ext = committed_buffer(
            device,
            if gbuf_full { px * GBUF_EXT_STRIDE } else { GBUF_EXT_STRIDE },
            uaf,
            ua,
        )?;
        // NPPD plane buffers at the /32-padded dims (~340 MB at 1080p/quality
        // — the recurrent state dominates, same as the CPU path's staging).
        let nppd_res = match nppd_psos {
            Some((pso_pack, pso_warp, pso_zero, pso_out, pso_feed_dm)) => {
                let (pw, ph) = crate::nppd::pad_dims(rw as usize, rh as usize);
                let ppx = pw as u64 * ph as u64;
                let ct = crate::nppd::C_T as u64;
                let ch = crate::nppd::CH_FRAME as u64;
                Some(NppdRes {
                    frame: committed_buffer(device, ppx * 4 * ch, uaf, ua)?,
                    state: committed_buffer(device, ppx * 4 * ct, uaf, ua)?,
                    warped: committed_buffer(device, ppx * 4 * ct, uaf, ua)?,
                    out: committed_buffer(device, ppx * 4 * 3, uaf, ua)?,
                    pw: pw as u32,
                    ph: ph as u32,
                    pso_pack,
                    pso_warp,
                    pso_zero,
                    pso_out,
                    pso_feed_dm,
                })
            }
            None => None,
        };
        // FEED_SETS copies of the RP_TEX table, back to back: each set is the
        // hdr resolve UAV (u14) at its offset 0, then NUM_FEED feed targets
        // (u16.., wired per session by wire_feed — null until then; RS 1.0
        // descriptors are volatile, so only ACCESSED slots must be valid).
        // Slots TEX_HEAP_BASE.. = the RP_SCENE_TEX scene table (4 buffer SRVs +
        // one Texture2D per scene texture — same heap by necessity: only one
        // CBV_SRV_UAV heap can be bound at a time).
        let uav_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: TEX_HEAP_BASE
                    + TEX_TABLE_BUFS
                    + scene_gpu.textures.len() as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("CreateDescriptorHeap(trace UAV): {e}"))?;
        write_resolve_uavs(device, &uav_heap, &hdr);
        scene_gpu.write_scene_descriptors(device, &uav_heap, TEX_HEAP_BASE);
        let tex_table = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: unsafe { uav_heap.GetGPUDescriptorHandleForHeapStart() }.ptr
                + TEX_HEAP_BASE as u64
                    * unsafe {
                        device.GetDescriptorHandleIncrementSize(
                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        )
                    } as u64,
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
        if let Some(n2) = &nppd_res {
            name(&n2.frame, "trace.nppd_frame");
            name(&n2.state, "trace.nppd_state");
            name(&n2.warped, "trace.nppd_warped");
            name(&n2.out, "trace.nppd_out");
        }

        // Scene-static CB fields, prefilled once (near/far ride the prev
        // block's w slots — dlss::near_far, the G-buffer sky depth source),
        // plus this tracer's queue capacities.
        let mut cb_base = FrameCb::base(scene, rw, rh);
        cb_base.cap_tile = cap_tile as u32;
        cb_base.cap_leaf = cap_leaf as u32;
        cb_base.cap_sky = cap_sky as u32;
        cb_base.cap_cut = cap_cut as u32;
        cb_base.cap_hemi_cell = cap_hemi_cell as u32;
        cb_base.cap_hemi_leaf = cap_hemi_cell as u32;
        cb_base.cap_hemi_cut = cap_hemi_cut as u32;

        // Where this construction left the local segment: the window planes +
        // sw trees land on top of whatever is already committed — the shared
        // scene core, and in a SPACE-cycled session the OTHER tracer too.
        // WDDM demotes over-budget commits silently (10-100× slowdown, no
        // error — the adapter::vram_info note), so the number prints here.
        if let Some((usage, budget)) = adapter::vram_info(device) {
            eprintln!(
                "gpu tracer: wavefront planes+trees committed | vram {} / {} MB",
                usage >> 20,
                budget >> 20
            );
        }

        Ok(Self {
            root_sig,
            cmd_sig,
            pso_reference,
            pso_resolve,
            pso_seed,
            pso_seed_replay,
            nrd: nrd_res,
            nrd_wired: Vec::new(),
            nrd_color: None,
            nrd_guides: Vec::new(),
            last_struct: std::cell::Cell::new(None),
            split: std::cell::Cell::new(TileSplit::ALL),
            split_refused: std::cell::Cell::new(false),
            sway_t: std::cell::Cell::new(None),
            sway_mv_on: !sway_def.is_empty(),
            sky_split,
            pso_prep,
            pso_prep_mul,
            pso_clear_info,
            pso_level,
            pso_level_wide,
            work_graph,
            pso_sky,
            pso_sky_lod,
            pso_cloud_shadow,
            cloud_lod,
            cloud_shadow,
            scene_aabb,
            cloud_shadow_n: cloud_shadow_v,
            sky_lod_k: sky_lod_v,
            pso_leaf,
            pso_leaf_fb,
            pso_clear_h,
            pso_prep_batch,
            pso_seed_probes,
            pso_hemi_root,
            pso_hemi_cell,
            pso_hemi_leaf,
            pso_compose,
            pso_feed_xess,
            pso_feed_rr,
            pso_feed_fsr_rr,
            feed: Vec::new(),
            device: device.clone(),
            nppd: nppd_res,
            scene: scene_gpu,
            sw,
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
            gbuf_ext,
            force_gbuf_ext: std::cell::Cell::new(false),
            force_fsr_sig: std::cell::Cell::new(false),
            force_nrd_sig: std::cell::Cell::new(None),
            force_sky_ext_skip: std::cell::Cell::new(None),
            gbuf_full,
            uav_heap,
            tex_table,
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
    /// The slab-space grid for this frame: map the scene AABB's 8 corners
    /// through M(p) = p + sun*(base + 0.5*thick - p.y)/sun.y (the projection
    /// the shadow is EXACTLY a function of), take the bounding box, and snap
    /// the origin to a whole cell. Snapping is what makes the lattice
    /// FRAME-STATIC: a grid that slid with the camera would move its own
    /// interpolation error every frame, which the temporal upscalers read as
    /// shimmer (the sky fill's frame-independent offsets, same lesson).
    fn cloud_grid_row(&self, p: &FrameParams) -> [f32; 4] {
        // Read the SNAPSHOT (not the live static): the buffer was sized and the
        // fill kernel compiled against this value in `new()`.
        if self.cloud_shadow_n == 0 || !p.clouds.enabled {
            return [0.0; 4];
        }
        crate::clouds::shadow_grid_row(self.cb_base.sun, self.scene_aabb, p.clouds.diag, self.cloud_shadow_n)
    }

    pub fn write_cb(&self, slot: usize, p: &FrameParams) {
        let mut cb = self.cb_base.with_frame(
            p,
            self.gbuf_full,
            self.fsr_sig(),
            self.gbuf_ext_needed(),
            self.nrd_sig(),
            self.sky_ext_skip(),
        );
        cb.cloud_grid = self.cloud_grid_row(p);
        cb.split = self.split.get().cb_row();
        // Sway MVs: the SAME predicate record_sway's fill uses (sway_mv_pair
        // + the compile-in), so the flag and the slot's rows cannot disagree.
        // Known-accept: a record_rebuild FAILURE after this CB was written
        // leaves one degraded rest-pose frame with sway MVs armed — the
        // already-loud rebuild-failure path.
        if self.sway_mv_on && sway_mv_pair(p).is_some() {
            if let Some(sw) = &self.scene.sway {
                cb.arm_sway_mv(slot as u32 * sw.n_inst());
            }
        }
        cb.store(unsafe { self.frame_cb.ptr.add(slot * CB_STRIDE) });
    }

    /// Re-derive the base CB's sun/sky rows after a TOD change
    /// (`FrameCb::refresh_sky_rows`). No fence hazard: `write_cb` stores a
    /// fresh ring slot per frame.
    pub fn refresh_sky(&mut self, scene: &Scene) {
        self.cb_base.refresh_sky_rows(scene, self.rw, self.rh);
    }

    /// Whether ANY wired feed consumes the pack's FSR signal lanes (under
    /// --quinlight, FSR4-RR can be one engine among several — one subscriber is
    /// enough to arm FLAG_FSR_SIG, and the capture is assignment-only, so the
    /// other engines' frames stay bit-identical).
    pub fn fsr_sig(&self) -> bool {
        // `nrd_wired`, deliberately NOT `nrd.is_some()`: the check-gpu pack
        // tracer builds the bridge PSOs unconditionally, and arming the sig
        // capture on mere PSO presence would run M9b's "off" baseline with
        // the capture ON — a vacuous bit-identity A/B. Wiring is the intent.
        self.force_fsr_sig.get()
            || !self.nrd_wired.is_empty()
            || self.feed.iter().any(|(k, _)| matches!(k, FeedKind::FsrRr))
    }

    /// Whether ANY consumer reads the pack's guide/signal half this frame.
    /// Same one-subscriber-is-enough shape as `fsr_sig` (and for the same
    /// `--quinlight` reason), plus NPPD, whose staging kernels read the
    /// normal and albedo lanes even though its feed kind is XeSS.
    ///
    /// XeSS and FSR 3.1 alone answer FALSE, which is the whole win: their
    /// sessions store 16 B/px instead of 88.
    ///
    /// PUBLIC for `--dual-gpu`, which must copy exactly the lanes the PRIMARY's
    /// feed will read: the secondary traced its band's pack on its own device,
    /// so a fed frame carries the pack across as well as `accum`. Copying the
    /// ext half unconditionally would be 72 B/px of waste on the link that is
    /// already the binding constraint; omitting it when a guide-consuming feed
    /// is wired hands the engine the primary's stale normals for those rows.
    /// Whether the pack buffers are FULL-SIZE (`rw*rh` strided) rather than the
    /// single-element dummies a plain session allocates.
    ///
    /// PUBLIC for `--dual-gpu`, and it is a memory-safety read there rather
    /// than an optimization: a plain (`--no-upscale`) session's `gbuf` is
    /// GBUF_STRIDE bytes TOTAL, so copying a band into it is an out-of-bounds
    /// `CopyBufferRegion`. That does not fault at record time — the command is
    /// simply invalid and the whole list fails to Close, which surfaces as
    /// `list Close: The parameter is incorrect` and then a permanently broken
    /// allocator. Measured exactly that way.
    pub fn pack_full(&self) -> bool {
        self.gbuf_full
    }

    pub fn gbuf_ext_needed(&self) -> bool {
        // `nrd_wired` for the same reason fsr_sig carries it: cs_nrd_pack
        // reads gbuf_ext FULL-SCREEN, so --dual-gpu's fed_strides must carry
        // the EXT band for the secondary's rows in an NRD session (without
        // it the bridge packs stale normals/albedo/sig for that band).
        self.force_gbuf_ext.get()
            || self.nppd.is_some()
            || !self.nrd_wired.is_empty()
            || self.feed.iter().any(|(k, _)| matches!(k, FeedKind::Rr | FeedKind::FsrRr))
    }

    /// Force the guide/signal half to be stored even with no guide-consuming
    /// feed wired.
    ///
    /// TWO legitimate consumers, and neither derives the flag the normal way
    /// because neither owns the feed that would set it. (1) The
    /// `--check-gpu`/`--check-dxr` pack gates: their consumer is a CPU
    /// READBACK, and they trace their coverage frames before any `wire_feed`
    /// call, so without this the normals/albedo they gate would be unwritten.
    /// (2) `--dual-gpu`'s SECONDARY tracer: its feed list is empty by
    /// construction — the upscaler lives on the primary — so it must be told
    /// to store the same pack half the PRIMARY's feed will read off its band
    /// after the transfer. Both mirror a flag that is a property of the
    /// CONSUMER, which in these two cases is not this tracer.
    pub fn force_gbuf_ext(&self, on: bool) {
        self.force_gbuf_ext.set(on);
    }

    /// The `fsr_sig` twin of `force_gbuf_ext`, and it exists for the same
    /// reason: an FSR4-RR session's pack carries the demodulated dd/ds/ao/ind_s
    /// lanes, `--dual-gpu`'s secondary has no FSR feed of its own to derive
    /// that from, and a band transferred without them hands the denoiser zeros
    /// for every signal in the secondary's rows — which reads as a black band
    /// through the composite, not as a subtle error.
    pub fn force_fsr_sig(&self, on: bool) {
        self.force_fsr_sig.set(on);
    }

    /// Whether this frame's sig capture folds the RTGI bounce into the dd
    /// lane for the NRD bridge (FLAG_NRD_GI). Wiring-derived like `fsr_sig`'s
    /// nrd term — never PSO presence (the M9b baseline-teeth argument) —
    /// unless the override is set (dual-GPU mirror / the N6 gate).
    pub fn nrd_sig(&self) -> bool {
        self.force_nrd_sig.get().unwrap_or(!self.nrd_wired.is_empty())
    }

    /// Whether sky pixels may SKIP the ext store this frame (FLAG_SKY_EXT_SKIP
    /// — see the const's comment). Derived TRUE only when NRD is the sole ext
    /// subscriber: the bridge's own sky branches make the bytes unread, while
    /// an RR/FsrRr feed, NPPD, or a forced-ext consumer (the pack readback
    /// gates, the dual-GPU secondary) reads ext at sky and vetoes. The
    /// override is the armed-skip gate's hook (`force_nrd_sig`'s Option
    /// shape — Some(true) must beat the force_gbuf_ext veto).
    pub fn sky_ext_skip(&self) -> bool {
        self.force_sky_ext_skip.get().unwrap_or(
            !self.nrd_wired.is_empty()
                && self.nppd.is_none()
                && !self.force_gbuf_ext.get()
                && !self
                    .feed
                    .iter()
                    .any(|(k, _)| matches!(k, FeedKind::Rr | FeedKind::FsrRr)),
        )
    }

    pub fn force_sky_ext_skip(&self, v: Option<bool>) {
        self.force_sky_ext_skip.set(v);
    }

    /// The `nrd_sig` half of the dual-GPU mirror (the `force_fsr_sig` shape):
    /// the secondary has no NRD wiring of its own, and a band packed WITHOUT
    /// the fold beside a primary that folds would hand ReBLUR direct-only
    /// diffuse for the secondary's rows — a per-band denoise-semantics seam.
    /// Also the check-gpu N6 gate's hook, which is why it is an OVERRIDE
    /// (Some(false) must beat live wiring); None = wiring-derived.
    pub fn force_nrd_sig(&self, v: Option<bool>) {
        self.force_nrd_sig.set(v);
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
            list.SetComputeRootUnorderedAccessView(
                RP_GBUF_EXT,
                self.gbuf_ext.GetGPUVirtualAddress(),
            );
            if let Some(n) = &self.nppd {
                list.SetComputeRootUnorderedAccessView(
                    RP_NPPD_FRAME,
                    n.frame.GetGPUVirtualAddress(),
                );
                list.SetComputeRootUnorderedAccessView(
                    RP_NPPD_STATE,
                    n.state.GetGPUVirtualAddress(),
                );
                list.SetComputeRootUnorderedAccessView(
                    RP_NPPD_WARPED,
                    n.warped.GetGPUVirtualAddress(),
                );
                list.SetComputeRootUnorderedAccessView(
                    RP_NPPD_OUT,
                    n.out.GetGPUVirtualAddress(),
                );
            }
            let s = &self.scene;
            // Tile dispatches consume the wide frustum tree at t0 when the
            // session carries one (`#define FTREE` matched at kernel-assembly
            // time); record_hemi rebinds the binary tree for the hemi phase.
            // The software trees live in self.sw, not the shared core.
            let t0 = self.sw.ftree_nodes.as_ref().unwrap_or(&self.sw.bvh_nodes);
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_BVH_NODES,
                t0.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_TRI_IDX,
                self.sw.tri_idx.GetGPUVirtualAddress(),
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
            // --foliage-sway: an animated frame traces the ring slot's TLAS;
            // everything else (rest-pose gates, None-clock frames, scenes
            // with no sway cells) the pristine static TLAS. The rebuild was
            // recorded by record_wavefront/_replay BEFORE this bind.
            let tlas_va = match (self.sway_t.get(), &s.sway) {
                (Some(_), Some(sw)) => sw.tlas_va(slot),
                _ => s.tlas.GetGPUVirtualAddress(),
            };
            list.SetComputeRootShaderResourceView(RP_SRV0 + SRV_TLAS, tlas_va);
            // The scene-texture table (t0..t3 + texs[] in space1). The heap
            // must be set before the table; resolve/feed re-setting the same
            // heap later is redundant-but-legal.
            list.SetDescriptorHeaps(&[Some(self.uav_heap.clone())]);
            list.SetComputeRootDescriptorTable(RP_SCENE_TEX, self.tex_table);
        }
    }

    /// --foliage-sway: stash the frame's clock and record the ring rebuild
    /// (a bit-equal clock records nothing — the converging-still fast path).
    /// MUST run before `bind_common`, which reads the stash to pick the
    /// TLAS. Set from the record path's OWN FrameParams (never write_cb —
    /// the check/bench sites that skip write_cb must not read stale). A
    /// rebuild failure degrades THIS frame to the static rest pose with one
    /// loud line, never a dead session.
    fn record_sway(&self, list: &ID3D12GraphicsCommandList, slot: usize, p: &FrameParams) {
        self.sway_t.set(p.sway_time);
        if let (Some(t), Some(sw)) = (p.sway_time, &self.scene.sway) {
            if let Err(e) = sw.record_rebuild(list, slot, t) {
                eprintln!("foliage-sway: ring rebuild failed ({e}) — rest pose this frame");
                self.sway_t.set(None);
            } else if self.sway_mv_on {
                // Sway MVs: fill the slot's prev−cur rows under the same
                // predicate write_cb arms FLAG_SWAY_MV with. Runs on replay
                // frames too (record_wavefront_replay calls record_sway —
                // replays re-shade fresh, MV write included).
                if let Some((tc, tp)) = sway_mv_pair(p) {
                    sw.write_mv_rows(slot, tc, tp);
                }
            }
        }
    }

    /// Record the vanilla full-screen reference trace (M2; also the on-GPU
    /// reference for the wavefront gates). Ends with a global UAV barrier.
    /// Takes no FrameParams, so it traces whatever TLAS the last
    /// `record_sway` stash selects — in a verify/R pair that is the SAME
    /// (possibly animated) TLAS the wavefront half traced, by construction.
    pub fn record_reference(&self, list: &ID3D12GraphicsCommandList, slot: usize) {
        let _ev = super::pix::scope(list, c"reference");
        unsafe {
            self.bind_common(list, slot);
            // --sw-rays: the reference kernel's software traversal walks the
            // BINARY tree at t0 (bind_common binds the wide one in ftree
            // sessions — a structure the ray loops can't descend).
            if sw_rays() {
                list.SetComputeRootShaderResourceView(
                    RP_SRV0 + SRV_BVH_NODES,
                    self.sw.bvh_nodes.GetGPUVirtualAddress(),
                );
            }
            self.record_cloud_shadow(list);
            // The reference kernel shades its own sky pixels through
            // sky_radiance_lod when SKY_LOD > 1 (mirroring cs_leaf), so it needs
            // the lattice filled + bound too — this is what keeps the exact-zero
            // wavefront-vs-reference image A/B bit-identical at the default-ON K.
            self.record_sky_lod(list);
            list.SetPipelineState(&self.pso_reference);
            {
                // The bare dispatch, mirroring `dxr-rays`' bracket shape
                // EXACTLY (bind/PSO/cache-fills/trailing-barrier all outside,
                // end timestamp pre-barrier like its twin): the outer
                // `reference` region also contains bind_common + the two
                // cloud-cache fills + the PSO set + the barrier, so a
                // reference-vs-dxr-rays read compares two DIFFERENT bracket
                // shapes — the finding-1 audit's bracket-asymmetry catch.
                // `reference-kernel` is the like-for-like row.
                let _k = super::pix::scope(list, c"reference-kernel");
                list.Dispatch(self.rw.div_ceil(8), self.rh.div_ceil(8), 1);
            }
            list.ResourceBarrier(&[uav_barrier(None)]);
        }
    }

    /// Fill the slab-space cloud shadow cache and leave it bound at u6 for the
    /// whole shading phase. EVERY path that compiles the cache in must call
    /// this: the reference kernel reads it through `cloud_sun_transmittance`
    /// exactly as the leaf kernel does, and binding only one of them leaves the
    /// other reading the tile queue as floats (a device hang, found the hard
    /// way — and it hid until a `--tod` pose made the shadow non-trivial enough
    /// to reach the fetch at all).
    unsafe fn record_cloud_shadow(&self, list: &ID3D12GraphicsCommandList) {
        if self.cloud_shadow_n == 0 {
            return;
        }
        unsafe {
            let _e = super::pix::scope(list, c"cloud-shadow");
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_QOUT,
                self.cloud_shadow.GetGPUVirtualAddress(),
            );
            // The live side rides the CB (it is derived per frame), so dispatch
            // the cap and let the kernel retire the tail.
            let groups =
                (crate::clouds::CLOUD_SHADOW_MAX * crate::clouds::CLOUD_SHADOW_MAX).div_ceil(64);
            list.SetPipelineState(&self.pso_cloud_shadow);
            list.Dispatch(groups.min(32768), groups.div_ceil(32768), 1);
            list.ResourceBarrier(&[uav_barrier(None)]);
        }
    }

    /// Fill the amortized cloud lattice (full-screen) and leave it bound at u5
    /// for the sky passes. Both `cs_sky` (proven-empty rects) and `cs_leaf`'s
    /// miss branch read it, AND the reference kernel does — so, like
    /// record_cloud_shadow, EVERY unit that compiles the lattice in must call
    /// this or its u5 (the tile queue's register, dead by shading) reads garbage
    /// as float4 (the same device-hang class). Snapshotted k: the buffer was
    /// sized and the kernel compiled against it in `new()`.
    unsafe fn record_sky_lod(&self, list: &ID3D12GraphicsCommandList) {
        let k = self.sky_lod_k;
        if k <= 1 {
            return;
        }
        unsafe {
            let _e = super::pix::scope(list, c"sky-lod");
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_QIN,
                self.cloud_lod.GetGPUVirtualAddress(),
            );
            let pts = ((self.rw / k) + 2) * ((self.rh / k) + 2);
            let groups = pts.div_ceil(64);
            list.SetPipelineState(&self.pso_sky_lod);
            list.Dispatch(groups.min(32768), groups.div_ceil(32768), 1);
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
        self.record_sway(list, slot, p);
        unsafe {
            self.bind_common(list, slot);
            // --sw-rays + cut consumption: the ladder reads ft_bnode at t1
            // (tri_idx's register — dead in every ladder kernel) for
            // level_finish's leaf-cut translation; the leaf/sky block below
            // rebinds the real tri_idx before any ray fires (the SKY_UNIT
            // register-remeaning idiom, phase-scoped).
            if let Some(bn) = &self.sw.ft_bnode {
                list.SetComputeRootShaderResourceView(
                    RP_SRV0 + SRV_TRI_IDX,
                    bn.GetGPUVirtualAddress(),
                );
            }
            // Seed sees level 0's queue arrangement (qin = A).
            list.SetComputeRootUnorderedAccessView(RP_UAV0 + UAV_QIN, self.qa.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(RP_UAV0 + UAV_QOUT, self.qb.GetGPUVirtualAddress());
            // push0 = 1 tells the seed to skip the root ENQUEUE: the work
            // graph's root arrives as CPU input, so a queued one would never
            // be consumed and would leave CTR_TILE_A dangling all frame. Set
            // explicitly rather than inherited — root constants are undefined
            // until written, so the ladder arm needs the 0 just as much.
            self.push(list, [self.work_graph.is_some() as u32, 0, 0, 0]);
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

            // THE WORK-GRAPH LADDER (FR_WORKGRAPH=1). One DispatchGraph in
            // place of depth_full x (prep + ExecuteIndirect). cs_seed above
            // still ran and is still needed: it zeroes the counters and handles
            // the degenerate rw,rh <= LEAF_TILE window. It does NOT enqueue a
            // root here (push0 = 1 above) — the graph's root is CPU input, and
            // an unconsumed queued one would leave CTR_TILE_A at 1 for the
            // whole frame. Everything downstream — leaf, sky, hemi, compose —
            // is untouched and reads the same queues, which is what keeps the
            // terminal accounting and coverage gates in force.
            if let Some(wg) = &self.work_graph {
                let _ev = super::pix::scope(list, c"workgraph");
                // Degenerate window: cs_seed already emitted the single leaf,
                // so there is no tile to expand and the graph must not run.
                let lt = crate::render::leaf_tile() as u32;
                if self.rw > lt || self.rh > lt {
                    let root = TileRecCpu {
                        xy0: 0,
                        xy1: (self.rw & 0xffff) | (self.rh << 16),
                        t_start: 0.0,
                        cut_slot: WG_ROOT_CUT_SLOT,
                        meta: 1, // cut_len 1, depth 0 — cs_seed's own root
                        path: 0,
                    };
                    if let Err(e) = wg.record(list, root) {
                        eprintln!("gpu work-graph: {e}");
                    }
                }
                list.ResourceBarrier(&[uav_barrier(None)]);
            }

            // The graph ran the whole ladder in one dispatch, so this loop is
            // empty there — no level kernels, no prep, no ExecuteIndirect.
            let ladder_levels = if self.work_graph.is_some() { 0 } else { self.depth_full };
            for d in 0..ladder_levels {
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
                //
                // DO NOT "optimize" this scaffolding away. It reads like pure
                // serialization — a 1-thread dispatch, a UAV barrier, an args
                // UAV->INDIRECT transition, then the reverse afterwards, times
                // depth_full — and it was long assumed to be why the ladder
                // costs what it does. It is not. Measured with nested timing
                // regions around the two halves (B70, --spin path 1080p, sums
                // over all 8 levels): prep + BOTH transitions **0.011 ms**
                // against the kernel's 0.428 (default scene) and 1.817
                // (--stress 5000) — 0.6% of the ladder. Barriers and
                // transitions are cheap here; per-level counters and a static
                // Dispatch would buy eleven microseconds. The ladder's cost is
                // the level KERNEL, running with too few threads at shallow
                // depths (levels 0-4 are <= 256 tiles yet 67% of the ladder,
                // because a shallow tile's frustum covers a quarter-screen and
                // its cut is barely refined, so each of those few lanes
                // descends an enormous slice of the BVH). That is what the
                // wave-cooperative level kernel addresses.
                // Shallow levels take the wave-cooperative kernel: ONE GROUP
                // per tile (items-per-group 1) instead of one thread per tile
                // (32). See WIDE_LEVELS for why the crossover exists at all;
                // --no-wide-levels (WIDE_LEVELS_ON) forces the serial ladder.
                let wide = WIDE_LEVELS_ON.load(std::sync::atomic::Ordering::Relaxed)
                    && d < wide_levels();
                let per_group = if wide { 1 } else { 32 };
                list.SetPipelineState(&self.pso_prep);
                self.push(list, [in_ctr, out_ctr, per_group, d]);
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
                list.SetPipelineState(if wide {
                    &self.pso_level_wide
                } else {
                    &self.pso_level
                });
                self.push(list, [in_ctr, out_ctr, 0, 0]);
                list.ExecuteIndirect(&self.cmd_sig, 1, &self.args, d as u64 * 12, None, 0);
                self.args_to_uav(list);
            }

            // Leaf + sky fills: shared with the replay path, so the two cannot
            // drift (the level_finish shared-tail idiom).
            self.record_terminal_fills(list, fb_mode);

            if fb_mode > 0 {
                // Every hit pixel appended a shading point; batch over the
                // worst case (all of them).
                self.record_hemi(list, self.rw * self.rh, p.q.fb.depth);
            }
            // fb OFF: leaf/sky already splatted into accum (see
            // queues.hlsli::accum_splat), so compose would be a pure
            // buffer-to-buffer copy. record_terminal_fills' args_to_uav
            // already ends with a GLOBAL uav barrier, so dropping the
            // dispatch drops nothing the resolve/feed depended on.
            if fb_mode > 0 {
                self.record_compose(list);
            }
        }
        // The queues now hold this basis's terminal structure — a later
        // bit-equal-basis frame under `p.replay` can re-dispatch it (record_frame
        // decides). Truthful for every producer, verify/check callers included.
        self.last_struct.set(Some((p.cam, self.split.get())));
    }

    /// The leaf + sky fills (disjoint pixels — no barrier between them) plus the
    /// per-frame cloud caches they consume. Shared by record_wavefront and
    /// record_wavefront_replay; must be preceded by the seed (real or replay)
    /// that set CTR_LEAF/CTR_SKY and by a UAV barrier.
    unsafe fn record_terminal_fills(&self, list: &ID3D12GraphicsCommandList, fb_mode: u32) {
        unsafe {
            let _ev = super::pix::scope(list, c"leaf+sky");
            list.SetPipelineState(&self.pso_prep);
            self.push(list, [CTR_LEAF, NO_RESET, 1, ARG_LEAF]);
            list.Dispatch(1, 1, 1);
            // Sky takes the MULTIPLYING prep: SKY_SPLIT groups share each
            // record so one huge proven-empty rect can't serialize on one
            // group (see SKY_SPLIT — this was ~70% of the tracer's frame).
            list.SetPipelineState(&self.pso_prep_mul);
            self.push(list, [CTR_SKY, NO_RESET, self.sky_split, ARG_SKY]);
            list.Dispatch(1, 1, 1);
            self.args_to_indirect(list);
            // fb frames need the hemi arm; every other frame takes the
            // slim kernel (leaf.hlsl's LEAF_NO_FB).
            // The slab-space cloud shadow cache, ahead of ALL shading and
            // left bound at u6 (the qout register, dead once the ladder has
            // drained and suppressed in the shading units by SKY_UNIT).
            self.record_cloud_shadow(list);
            // The cloud lattice, full-screen and ahead of BOTH consumers:
            // `cs_sky` (proven-empty rects) and `cs_leaf`'s miss branch
            // (sky inside leaf tiles). u5 = the tile queue's register, dead
            // once the ladder has drained; both re-declare it (SKY_UNIT).
            self.record_sky_lod(list);
            // --sw-rays: the leaf kernel's software ray traversal walks
            // the BINARY tree at t0 (bind_common bound the wide one for
            // the ladder — a structure the ray loops can't descend) and
            // the real tri_idx at t1 (the ladder may hold ft_bnode there)
            // — the record_hemi rebind precedent.
            if sw_rays() {
                list.SetComputeRootShaderResourceView(
                    RP_SRV0 + SRV_BVH_NODES,
                    self.sw.bvh_nodes.GetGPUVirtualAddress(),
                );
                list.SetComputeRootShaderResourceView(
                    RP_SRV0 + SRV_TRI_IDX,
                    self.sw.tri_idx.GetGPUVirtualAddress(),
                );
            }
            let leaf_pso = if fb_mode > 0 { &self.pso_leaf_fb } else { &self.pso_leaf };
            list.SetPipelineState(leaf_pso);
            self.push(list, [CTR_LEAF, 0, 0, 0]);
            {
                // Nested sub-brackets (the level-ladder idiom). No barrier
                // separates the two EIs, so `sky` under-reports whatever
                // overlapped `leaf`; `leaf` is honest — its end timestamp
                // is bottom-of-pipe at the leaf EI's drain.
                let _e = super::pix::scope(list, c"leaf");
                list.ExecuteIndirect(&self.cmd_sig, 1, &self.args, ARG_LEAF as u64 * 12, None, 0);
            }
            list.SetPipelineState(&self.pso_sky);
            self.push(list, [CTR_SKY, 0, 0, 0]);
            {
                let _e = super::pix::scope(list, c"sky");
                list.ExecuteIndirect(&self.cmd_sig, 1, &self.args, ARG_SKY as u64 * 12, None, 0);
            }
            self.args_to_uav(list);
        }
    }

    /// Structure replay: a frame whose basis bit-equals the previous producing
    /// frame's re-dispatches the persisted terminal queues (qleaf/qsky/cut_pool
    /// + CTR_LEAF/CTR_SKY/CTR_CUT) and skips seed + the whole level ladder — the
    /// GPU mirror of render::render_frame_replay. The structure is a pure
    /// function of (scene, BVH, basis, rw, rh); spp/jitter/frame/fb/quality/
    /// clouds all ride the CB, so a replay frame's shading is fresh. The CALLER
    /// (record_frame) proves the bit-equality; this leaves last_struct untouched
    /// (a replay does not change what the queues hold).
    pub fn record_wavefront_replay(
        &self,
        list: &ID3D12GraphicsCommandList,
        slot: usize,
        p: &FrameParams,
        clear_sentinel: bool,
    ) {
        let fb_mode = fb_mode_of(&p.q);
        let _ev = super::pix::scope(list, c"wavefront-replay");
        // Replay skips the ladder, NOT the rays: leaf fills re-trace against
        // whatever TLAS is bound, so the ring rebuild records here too (free
        // on a frozen clock — the bit-equal skip).
        self.record_sway(list, slot, p);
        unsafe {
            self.bind_common(list, slot);
            // Root UAVs u5/u6 are NOT bound by bind_common; the replay list is
            // fresh, so bind qa/qb before any dispatch (the cloud passes below
            // rebind them as the lattice/cache, exactly as the full path does).
            list.SetComputeRootUnorderedAccessView(RP_UAV0 + UAV_QIN, self.qa.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(RP_UAV0 + UAV_QOUT, self.qb.GetGPUVirtualAddress());
            // Keep CTR_LEAF/CTR_SKY/CTR_CUT (the persisted terminal counts),
            // zero everything else the fills/hemi would otherwise accumulate.
            list.SetPipelineState(&self.pso_seed_replay);
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
            self.record_terminal_fills(list, fb_mode);
            if fb_mode > 0 {
                self.record_hemi(list, self.rw * self.rh, p.q.fb.depth);
                self.record_compose(list); // see the record_wavefront twin
            }
        }
    }

    /// This device's `--dual-gpu` tile assignment.
    // Read by the stage-4 balancer (which rebalances against the CURRENT
    // assignment) and by the transfer, which copies exactly the rows this
    // device owns. The setter is what stage 1 exercises.
    #[allow(dead_code)]
    pub fn split(&self) -> TileSplit {
        self.split.get()
    }

    /// Assign which level-`depth` tiles this device renders.
    ///
    /// No explicit replay invalidation is needed — the split is IN the replay
    /// key, so a changed assignment simply fails the bit-equality test and the
    /// next frame traces fresh. That is deliberately not the same as clearing
    /// the key: flipping back to a previous assignment at an unchanged basis
    /// legitimately replays, which is what makes a balancer that oscillates
    /// between two splits cost nothing extra.
    ///
    /// RETURNS WHETHER THE SPLIT WAS APPLIED, and a refusal leaves the device
    /// on `TileSplit::ALL` — the whole screen, the pre-feature dispatch.
    /// Refusing by KEEPING the current assignment is what shipped, and it is
    /// unsound in both directions: keep a partial split and the rows the
    /// caller thinks it handed to the other device go unrendered (a hole);
    /// keep the whole screen while the OTHER device did arm and its band
    /// overwrites rows this one also traced, which under accumulation is two
    /// devices' sample counts fighting over the same pixels. Handing the
    /// caller a `false` lets it do the only correct thing — put BOTH devices
    /// back on the whole screen and take the single-GPU path for the frame.
    /// The message prints once per refusal episode (`split_refused`), because
    /// every condition here is a property of the resolution or a lever and so
    /// repeats on every frame it holds for.
    ///
    /// A depth above `MAX_SPLIT_DEPTH` cannot be represented in the CB's 64-bit
    /// mask, so it is refused loudly rather than silently rendering a subset
    /// (a hole in the image) — the loud-degrade rule.
    ///
    /// FLAG_WAVEVIZ is refused with it, the `DxrGpu::set_band` rule and for
    /// the same reason: the overlay's ticket is `WaveReadLaneFirst` of the
    /// first lane's pixel index, i.e. a property of how the DRIVER packed the
    /// launch, and a split changes that packing by construction. It is worse
    /// here than on the DXR side, because `tbuf` — where the tickets live — is
    /// not among the planes the band transfer carries, so the other device's
    /// rows would show this device's PREVIOUS frame's tickets. A `--spin
    /// --waveviz` run would then print a compactness line computed over them:
    /// an instrument reporting fabricated numbers, which is strictly worse
    /// than one that declines to run.
    ///
    /// A depth at or below the LEAF FRONTIER is refused for a subtler reason,
    /// and it is a soundness guard rather than a sanity check. `level_finish`
    /// has a second terminal path — the batch that emits four LEAF children
    /// directly when both child extents fit `LEAF_TILE` — which does not pass
    /// through the child-ownership test. Requiring `depth < depth_full` puts
    /// that batch strictly BELOW the split depth (it only fires at the leaf
    /// frontier, whose children are at `depth_full`), so its leaves always lie
    /// inside an already-owned tile and need no test of their own. Drop this
    /// guard and that path silently emits all four leaves on both devices.
    /// A split that fine has nothing to balance anyway — its tiles would be
    /// single leaf tiles.
    pub fn set_split(&self, split: TileSplit) -> bool {
        let full = depth_full(self.rw, self.rh);
        let why = if split.depth > MAX_SPLIT_DEPTH {
            Some(format!(
                "depth {} exceeds MAX_SPLIT_DEPTH={} (the CB mask is 64 bits)",
                split.depth, MAX_SPLIT_DEPTH
            ))
        } else if split.depth >= full && split.depth != 0 {
            Some(format!(
                "depth {} is at or below the leaf frontier (depth_full={} at {}x{})",
                split.depth, full, self.rw, self.rh
            ))
        } else if split.depth != 0 && waveviz_on() {
            Some(
                "--waveviz measures launch PACKING, which a split changes by construction (and \
                 `tbuf`, which carries the tickets, does not cross the band)"
                    .into(),
            )
        } else {
            None
        };
        if let Some(why) = why {
            if !self.split_refused.replace(true) {
                eprintln!("dual-gpu: {why} — rendering the whole screen on this device");
            }
            self.split.set(TileSplit::ALL);
            return false;
        }
        // Clear on a NON-TRIVIAL accept only. `TileSplit::ALL` is depth 0 and
        // passes every test above, and the both-or-neither degrade arm calls
        // it on BOTH devices on every refused frame — so an unguarded clear
        // re-arms the announce each frame, and a session refusing every frame
        // (`--waveviz --dual-gpu`) prints two lines per frame instead of two
        // lines per episode. The `DxrGpu::set_band` twin.
        if split.depth != 0 {
            self.split_refused.set(false);
        }
        self.split.set(split);
        true
    }

    /// Drop the replay key. Called when a recorded-but-never-executed producing
    /// frame is aborted (the queues then hold the OLD basis while `last_struct`
    /// would claim the NEW one), and when a hemi-probe seed zeroes the terminal
    /// counts out from under the persisted structure.
    pub fn invalidate_replay(&self) {
        self.last_struct.set(None);
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
            // pool (the primary qleaf/cut_pool are done for this frame) —
            // and t0 back to the BINARY tree: the hemi kernels compile the
            // binary bound_query (short queries lose on the wide tree; see
            // SwTreesGpu), while bind_common bound the wide one for the tiles.
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_BVH_NODES,
                self.sw.bvh_nodes.GetGPUVirtualAddress(),
            );
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
            // Structure replay: a bit-equal basis under `p.replay` re-dispatches
            // the persisted terminal queues instead of re-running seed + the
            // ladder. record_reference is a verified non-clobber (its unit
            // declares no counters/queues), so the R toggle round-trips free.
            if p.replay && self.last_struct.get() == Some((p.cam, self.split.get())) {
                self.record_wavefront_replay(list, slot, p, false);
            } else {
                self.record_wavefront(list, slot, p, false);
            }
        } else {
            // The R-toggle reference frame consumes the SAME clock the
            // wavefront would (record_reference itself takes no params).
            self.record_sway(list, slot, p);
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
        // cs_seed_probes zeroes CTR_LEAF/CTR_SKY, so the persisted terminal
        // structure is no longer valid after this — drop the replay key.
        self.invalidate_replay();
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

    /// Wire ONE engine's upscaler feed targets into the descriptor heap and
    /// remember them for record_feed's barriers. `targets` = (shader register,
    /// plane, format). Gated on typed-UAV-store support per format (optional in
    /// D3D12) — an Err here means the caller falls back to plain presentation,
    /// loudly.
    ///
    /// REPLACES the wiring with this one engine (descriptor set 0) — the
    /// single-upscaler contract, and what lets `--check-gpu` rewire the same
    /// tracer from one feed kind to the next. Use `wire_feed_add` to wire
    /// several engines at once (--quinlight).
    pub fn wire_feed(
        &mut self,
        device: &ID3D12Device,
        kind: FeedKind,
        targets: &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
    ) -> Result<()> {
        self.feed.clear();
        self.wire_feed_add(device, kind, targets)
    }

    /// APPENDS one engine: it claims the next descriptor set, so a --quinlight
    /// session calls this once per engine and their (overlapping) registers land
    /// in disjoint heap slots.
    pub fn wire_feed_add(
        &mut self,
        device: &ID3D12Device,
        kind: FeedKind,
        targets: &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
    ) -> Result<()> {
        let set = self.feed.len() as u32;
        let planes = wire_feed_targets(device, &self.uav_heap, set, targets)?;
        self.feed.push((kind, planes));
        Ok(())
    }

    /// Fan the pack + accum out into the wired upscaler input planes — the
    /// GPU-resident replacement for rr/xr::record_upload. Record AFTER
    /// record_frame on the same list (its trailing global UAV barrier fences
    /// the pack/accum writes). The planes transition NPSR -> UAV -> NPSR; the
    /// back-transition is both the write->read sync and what keeps the
    /// upscalers' state-at-use contracts truthful (RR's tags and XeSS's
    /// bindings both declare NON_PIXEL_SHADER_RESOURCE).
    /// `nppd_color = true` (NPPD frames only): the guide planes come from the
    /// depth+mvec feed variant and the color plane from `cs_nppd_out` (the
    /// denoised planar buffer ORT wrote in an earlier submission on the same
    /// queue) — two dispatches writing disjoint planes inside one barrier
    /// window.
    /// One dispatch per WIRED engine (normal sessions wire exactly one;
    /// `--quinlight` wires several and each takes its own descriptor set).
    pub fn record_feed(
        &self,
        list: &ID3D12GraphicsCommandList,
        slot: usize,
        nppd_color: bool,
    ) -> Result<()> {
        if self.feed.is_empty() {
            return Err("feed targets not wired".into());
        }
        let nppd = if nppd_color {
            // NPPD composes with XeSS only — and never with --quinlight, whose
            // engines each own a descriptor set (the color-crop dispatch below
            // rides the single feed's set).
            if self.feed.len() != 1 || !matches!(self.feed[0].0, FeedKind::Xess) {
                return Err("NPPD feed composition is XeSS-only".into());
            }
            Some(self.nppd.as_ref().ok_or("NPPD staging not built")?)
        } else {
            None
        };
        let mut feeds: Vec<(&ID3D12PipelineState, u32, &[ID3D12Resource])> = Vec::new();
        for (set, (kind, planes)) in self.feed.iter().enumerate() {
            let pso = feed_pso(
                *kind,
                nppd.map(|n| &n.pso_feed_dm),
                self.pso_feed_xess.as_ref(),
                self.pso_feed_rr.as_ref(),
                self.pso_feed_fsr_rr.as_ref(),
            )
            .ok_or("feed PSO missing (TraceGpu built without gbuf)")?;
            feeds.push((pso, set as u32, planes.as_slice()));
        }
        record_feed_dispatch(
            list,
            &self.device,
            &self.uav_heap,
            &feeds,
            nppd.map(|n| &n.pso_out),
            self.rw,
            self.rh,
            &|| unsafe { self.bind_common(list, slot) },
        );
        Ok(())
    }

    /// NPPD pre-inference staging: pack the G-buffer + 1-spp radiance into
    /// the NCHW frame buffer and backward-warp the recurrent state (or zero
    /// the warped buffer when `state_valid` is false — a reset). Record AFTER
    /// `record_frame` on the same list (its trailing global UAV barrier
    /// fences the pack/accum writes); the three kernels touch pairwise-
    /// disjoint buffers, so no barriers in between. The caller must SUBMIT
    /// this list before `NppdGpu::run()` — single-queue order is the sync.
    pub fn record_nppd_pre(
        &self,
        list: &ID3D12GraphicsCommandList,
        slot: usize,
        state_valid: bool,
    ) -> Result<()> {
        let n = self.nppd.as_ref().ok_or("NPPD staging not built")?;
        let _ev = super::pix::scope(list, c"nppd-stage");
        unsafe {
            self.bind_common(list, slot);
            list.SetPipelineState(&n.pso_pack);
            list.Dispatch(n.pw / 8, n.ph / 8, 1);
            list.SetPipelineState(if state_valid { &n.pso_warp } else { &n.pso_zero });
            list.Dispatch(n.pw / 8, n.ph / 8, 1);
        }
        Ok(())
    }

    /// Point descriptor set NRD_FEED_SET at the frame's NRD planes (the
    /// bridge kernels' u16..u27 — see nrd_bridge.hlsl's register map). The
    /// engine feed sets (0..2) are untouched; NRD is a bridge, never a
    /// FeedKind, so `fsr_sig()`/`record_feed` semantics cannot be perturbed
    /// by wiring it.
    pub fn wire_nrd_feed(
        &mut self,
        device: &ID3D12Device,
        targets: &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
    ) -> Result<()> {
        self.nrd_wired = wire_feed_targets(device, &self.uav_heap, NRD_FEED_SET, targets)?;
        // u16 = the engine's color plane: record_nrd_out brackets its write
        // with NPSR↔UA transitions (the plane rests NON_PIXEL_SHADER_RESOURCE
        // — the upscaler-eval contract every other writer honors through
        // record_feed_dispatch's own transitions).
        self.nrd_color = targets.iter().find(|t| t.0 == 16).map(|t| t.1.clone());
        // u18/u19 = the engine's depth/mvec guide planes the folded pack
        // writes — same rest state, bracketed around record_nrd_pack.
        self.nrd_guides = targets
            .iter()
            .filter(|t| t.0 == 18 || t.0 == 19)
            .map(|t| t.1.clone())
            .collect();
        Ok(())
    }

    /// Un-wire the NRD set: drops the plane keepalives (and with them the
    /// last refs once NrdGpu is gone) and disarms `fsr_sig`'s nrd term, so a
    /// shed session stops paying the sig/sig2 pack stores. The set-3
    /// descriptors go stale, which is fine — nothing binds them once
    /// record_nrd_* stops being called.
    pub fn clear_nrd_wired(&mut self) {
        self.nrd_wired.clear();
        self.nrd_color = None;
        self.nrd_guides.clear();
    }

    /// Whether THIS tracer can run the NRD bridge (PSOs built AND planes
    /// wired) — the presenters' per-arm predicate, so a session whose other
    /// arm armed NRD runs this one plain instead of shedding on the first
    /// "NRD bridge not built" frame.
    pub fn nrd_armed(&self) -> bool {
        self.nrd.is_some() && !self.nrd_wired.is_empty()
    }

    /// The bridge's front half: gbuf/gbuf_ext/accum -> NRD's five IN planes,
    /// PLUS the engine's mvec/depth guide planes (the fold — cs_feed_xess_dm's
    /// stores moved into cs_nrd_pack, so record_feed_nrd is gone). Runs after
    /// record_frame (its trailing global UAV barrier fences the pack writes);
    /// the caller owns the IN planes' UAV barrier before NRD's own passes
    /// consume them. Only the ENGINE guide planes transition here — the
    /// bridge's IN planes rest UA (the NrdGpu pool doctrine).
    pub fn record_nrd_pack(&self, list: &ID3D12GraphicsCommandList, slot: usize) -> Result<()> {
        let n = self.nrd.as_ref().ok_or("NRD bridge not built")?;
        let _ev = super::pix::scope(list, c"nrd-pack");
        unsafe {
            let npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            let pre: Vec<_> = self.nrd_guides.iter().map(|r| transition(r, npsr, ua)).collect();
            if !pre.is_empty() {
                list.ResourceBarrier(&pre);
            }
            self.bind_common(list, slot);
            list.SetDescriptorHeaps(&[Some(self.uav_heap.clone())]);
            list.SetComputeRootDescriptorTable(
                RP_TEX,
                feed_set_handle(&self.device, &self.uav_heap, NRD_FEED_SET),
            );
            list.SetPipelineState(&n.pso_pack);
            list.Dispatch(self.rw.div_ceil(8), self.rh.div_ceil(8), 1);
            let post: Vec<_> = self.nrd_guides.iter().map(|r| transition(r, ua, npsr)).collect();
            if !post.is_empty() {
                list.ResourceBarrier(&post);
            }
        }
        Ok(())
    }

    /// The bridge's back half: the delta-form recompose into the upscaler's
    /// color plane (u16 of the NRD set — the same texture the engine set's
    /// color slot names). The caller fences NRD's OUT-plane writes first.
    /// The color plane RESTS in NON_PIXEL_SHADER_RESOURCE (the state the
    /// upscaler eval reads it in; record_feed_dispatch round-trips it the
    /// same way), so the UAV write is bracketed NPSR→UA→NPSR here — gate
    /// callers must create their stand-in color plane resting NPSR too.
    pub fn record_nrd_out(&self, list: &ID3D12GraphicsCommandList, slot: usize) -> Result<()> {
        let n = self.nrd.as_ref().ok_or("NRD bridge not built")?;
        let color = self.nrd_color.as_ref().ok_or("NRD color plane not wired")?;
        let _ev = super::pix::scope(list, c"nrd-out");
        unsafe {
            let npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            list.ResourceBarrier(&[transition(color, npsr, ua)]);
            self.bind_common(list, slot);
            list.SetDescriptorHeaps(&[Some(self.uav_heap.clone())]);
            list.SetComputeRootDescriptorTable(
                RP_TEX,
                feed_set_handle(&self.device, &self.uav_heap, NRD_FEED_SET),
            );
            list.SetPipelineState(&n.pso_out);
            list.Dispatch(self.rw.div_ceil(8), self.rh.div_ceil(8), 1);
            list.ResourceBarrier(&[transition(color, ua, npsr)]);
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

/// Owner-independent half of `wire_feed`: gate every target format on typed
/// UAV store support (optional in D3D12), bounds-check the registers into
/// the feed range, and write the TEXTURE2D UAV descriptors into `uav_heap`
/// at slot reg - 15 (u16 -> slot 1, after the resolve target at slot 0).
/// Returns the plane list the record-side barriers need. Shared by TraceGpu
/// and DxrGpu — both bind the same root layout, so the heap contract is
/// identical.
/// The resolve target (u14) sits at offset 0 of the RP_TEX table — so it must
/// be present at the head of EVERY feed set, since a feed dispatch binds the
/// table at its own set's offset and the range layout is the same one the
/// kernel declares. Cheap: a descriptor, not a resource.
pub(crate) fn write_resolve_uavs(
    device: &ID3D12Device,
    uav_heap: &ID3D12DescriptorHeap,
    hdr: &ID3D12Resource,
) {
    let inc =
        unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) }
            as usize;
    let base = unsafe { uav_heap.GetCPUDescriptorHandleForHeapStart() };
    for set in 0..FEED_SETS as usize {
        unsafe {
            device.CreateUnorderedAccessView(
                hdr,
                None,
                None,
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: base.ptr + set * FEED_SET_STRIDE as usize * inc,
                },
            )
        };
    }
}

/// The GPU handle a feed dispatch binds RP_TEX at: the start of `set`'s copy of
/// the table.
pub(crate) fn feed_set_handle(
    device: &ID3D12Device,
    uav_heap: &ID3D12DescriptorHeap,
    set: u32,
) -> D3D12_GPU_DESCRIPTOR_HANDLE {
    let inc =
        unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) }
            as u64;
    D3D12_GPU_DESCRIPTOR_HANDLE {
        ptr: unsafe { uav_heap.GetGPUDescriptorHandleForHeapStart() }.ptr
            + (set * FEED_SET_STRIDE) as u64 * inc,
    }
}

pub(crate) fn wire_feed_targets(
    device: &ID3D12Device,
    uav_heap: &ID3D12DescriptorHeap,
    set: u32,
    targets: &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
) -> Result<Vec<ID3D12Resource>> {
    if set >= FEED_SETS {
        return Err(format!("feed set {set} >= FEED_SETS ({FEED_SETS})"));
    }
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
    let inc =
        unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) }
            as usize;
    let base = unsafe { uav_heap.GetCPUDescriptorHandleForHeapStart() };
    let mut planes = Vec::with_capacity(targets.len());
    for &(reg, res, format) in targets {
        // A register outside the feed range would silently overwrite the
        // hdr descriptor (slot 0) or write past the heap end — descriptor
        // writes have no bounds check of their own, so gate in release.
        if !(NUM_UAVS + 2..NUM_UAVS + 2 + NUM_FEED).contains(&reg) {
            return Err(format!(
                "feed target u{reg} outside u{}..u{}",
                NUM_UAVS + 2,
                NUM_UAVS + 1 + NUM_FEED
            ));
        }
        // u16 -> slot 1 of the set (slot 0 is the set's resolve-target copy).
        let slot = (set * FEED_SET_STRIDE + (reg - (NUM_UAVS + 1))) as usize;
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
    Ok(planes)
}

/// Owner-independent half of `record_feed`: NPSR -> UAV transitions, the
/// owner's root binds (`bind` — invoked after the transitions, must set the
/// root signature + common roots), the descriptor table, the feed dispatch
/// (+ an optional second dispatch writing disjoint planes inside the same
/// barrier window — TraceGpu's NPPD color crop), and the NPSR
/// back-transitions that double as the write->read sync and the upscalers'
/// state-at-use contract.
/// `feeds` is one entry per wired engine: (its PSO, its descriptor set, its
/// planes). Normal sessions pass exactly one; `--quinlight` passes one per
/// engine, and each dispatch binds RP_TEX at ITS set — which is precisely why
/// the sets exist (rewriting one set's descriptors between two dispatches in
/// the same list would let the last write win for both).
///
/// All the engines' planes share ONE barrier window: they are pairwise disjoint
/// resources and each is written by exactly one dispatch.
pub(crate) fn record_feed_dispatch(
    list: &ID3D12GraphicsCommandList,
    device: &ID3D12Device,
    uav_heap: &ID3D12DescriptorHeap,
    feeds: &[(&ID3D12PipelineState, u32, &[ID3D12Resource])],
    extra_pso: Option<&ID3D12PipelineState>,
    rw: u32,
    rh: u32,
    bind: &dyn Fn(),
) {
    let _ev = super::pix::scope(list, c"feed");
    unsafe {
        let npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let all = || feeds.iter().flat_map(|(_, _, planes)| planes.iter());
        let to_uav: Vec<_> = all().map(|p| transition(p, npsr, ua)).collect();
        list.ResourceBarrier(&to_uav);
        bind();
        list.SetDescriptorHeaps(&[Some(uav_heap.clone())]);
        for (pso, set, _) in feeds {
            list.SetComputeRootDescriptorTable(RP_TEX, feed_set_handle(device, uav_heap, *set));
            list.SetPipelineState(*pso);
            list.Dispatch(rw.div_ceil(8), rh.div_ceil(8), 1);
        }
        if let Some(p2) = extra_pso {
            // NPPD's color crop: a second dispatch writing planes DISJOINT from
            // the feed's, inside the same barrier window. XeSS-only, so it rides
            // the single feed's set, which record_feed's guard pins.
            list.SetPipelineState(p2);
            list.Dispatch(rw.div_ceil(8), rh.div_ceil(8), 1);
        }
        let back: Vec<_> = all().map(|p| transition(p, ua, npsr)).collect();
        list.ResourceBarrier(&back);
    }
}

// ---------------------------------------------------------------------------
// M11: BC7 encode fidelity, measured on the GPU (runs whenever BC7 is armed
// — the default — on a scene with compressible textures).

/// The whole M11 kernel: `.Load` one texel of the BC7 SRV (legal on BC —
/// returns the hardware-decoded value) and store it back as packed RGBA8.
/// BC7 DECODE is required by the D3D spec to be bit-exact, so
/// `round(load * 255)` recovers the decoder's exact 8-bit values — the diff
/// against the CPU source texels then measures the ENCODER's loss and
/// nothing else.
const BC7_READ_HLSL: &str = r#"
Texture2D<float4> src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer C : register(b0) { uint W; uint H; }
[numthreads(8, 8, 1)]
void cs_bc7_read(uint3 id : SV_DispatchThreadID) {
    if (id.x >= W || id.y >= H) return;
    uint4 q = (uint4)round(saturate(src.Load(int3(int(id.x), int(id.y), 0))) * 255.0);
    dst.Store((id.y * W + id.x) * 4u, q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u));
}
"#;

pub struct Bc7Fidelity {
    pub textures: usize,
    /// Mean |decoded − source| per RGB channel sample, in 8-bit LSB.
    pub mean_abs: f64,
    /// Worst single-channel diff, LSB.
    pub max_abs: u32,
    /// Worst per-texture RGB PSNR, dB.
    pub worst_psnr: f64,
}

/// Encode every compressible texture with the session's arm — the GPU
/// (default) arm runs the SESSION'S OWN encoder module on the session's own
/// device (no determinism bridge needed: this literally is the encoder that
/// uploads); the `--bc7-cpu` arm re-encodes with the deterministic ispc path
/// (`bc7::self_test` pins it, so those blocks ARE the session's). Each lands
/// in a plain `BC7_UNORM` Texture2D (deliberately never `_SRGB`: the decode
/// kernel must read raw code values, not the transfer function), is
/// GPU-decoded back with `BC7_READ_HLSL`, and diffed against the CPU RGBA8
/// source.
///
/// RGB only: nothing ever samples a compressed texture's alpha (the cutout
/// path reads only the alpha-masked RGBA8 set, and shade.hlsli consumes
/// .rgb/.g/.b), and "opaque" merely means every alpha ≥ 250 — a 252 would
/// quantize and show up here as false loss.
///
/// `Ok(None)` = BC7 off, or nothing compressible (untextured scene, or
/// every texture masked/odd-dim).
pub fn bc7_fidelity(
    scene: &Scene,
    mode: bc7::Bc7Mode,
    hg: &mut HeadlessGpu,
) -> Result<Option<Bc7Fidelity>> {
    use super::d3d12::Submit;
    let Some(q) = mode.quality() else {
        return Ok(None);
    };
    let cpu_arm = matches!(mode, bc7::Bc7Mode::Cpu(_));
    let ids: Vec<usize> =
        (0..scene.textures.len()).filter(|&i| bc7::should_compress(&scene.textures[i])).collect();
    if ids.is_empty() {
        return Ok(None);
    }
    let device = hg.device.clone();

    // CPU arm: the session's exact blocks, re-encoded (LPT largest-first,
    // the upload path's scheduling). GPU arm: the encoder itself, encoding
    // in-gate below.
    let mut cpu_blocks: Vec<Option<Vec<u8>>> = scene.textures.iter().map(|_| None).collect();
    if cpu_arm {
        use rayon::prelude::*;
        let mut order = ids.clone();
        order.sort_by_key(|&i| {
            std::cmp::Reverse(scene.textures[i].w as u64 * scene.textures[i].h as u64)
        });
        let done: Vec<(usize, Vec<u8>)> =
            order.par_iter().map(|&i| (i, bc7::encode_opaque(&scene.textures[i], q))).collect();
        for (i, b) in done {
            cpu_blocks[i] = Some(b);
        }
    }
    let gpu_enc = if cpu_arm {
        None
    } else {
        let block_cap = ids
            .iter()
            .map(|&i| {
                let t = &scene.textures[i];
                d3d12::block_pitch(t.w) * bc7::blocks(t.h) as usize
            })
            .max()
            .unwrap();
        // In the gate, encoder-construction failure is a hard error (the
        // caller FAILs the suite) — gates never silently skip.
        Some(super::bc7gpu::Bc7Enc::new(&device, block_cap)?)
    };

    let rk = bc7_read_kernel(&device)?;

    // One staging pair reused across textures (the blocking submits fence
    // it). The CPU arm stages encoded block rows; the GPU arm stages the
    // RGBA8 source rows the encoder reads.
    let max_stage = ids
        .iter()
        .map(|&i| {
            let t = &scene.textures[i];
            if cpu_arm {
                d3d12::block_pitch(t.w) * bc7::blocks(t.h) as usize
            } else {
                d3d12::aligned_pitch(t.w as usize * 4) * t.h as usize
            }
        })
        .max()
        .unwrap();
    let max_out = ids
        .iter()
        .map(|&i| scene.textures[i].w as u64 * scene.textures[i].h as u64 * 4)
        .max()
        .unwrap();
    let stage = d3d12::UploadBuffer::new(&device, max_stage)?;
    let out = committed_buffer(
        &device,
        max_out,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )?;

    let (mut sum_abs, mut n_samples, mut max_abs, mut worst_psnr) = (0f64, 0u64, 0u32, f64::MAX);
    for &i in &ids {
        let t = &scene.textures[i];
        let fmt = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC7_UNORM;
        let tex = d3d12::committed_tex(
            &device,
            t.w,
            t.h,
            fmt,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        if let Some(enc) = &cpu_blocks[i] {
            let src_pitch = bc7::blocks(t.w) as usize * bc7::BLOCK_BYTES;
            let pitch = d3d12::block_pitch(t.w);
            for r in 0..bc7::blocks(t.h) as usize {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        enc[r * src_pitch..].as_ptr(),
                        stage.ptr.add(r * pitch),
                        src_pitch,
                    )
                };
            }
            let fp = d3d12::footprint_block(fmt, t.w, t.h, 0);
            hg.run_list(&mut |l| {
                unsafe {
                    l.CopyTextureRegion(
                        &d3d12::loc_subresource(&tex),
                        0,
                        0,
                        0,
                        &d3d12::loc_footprint(&stage.resource, fp),
                        None,
                    );
                    l.ResourceBarrier(&[transition(
                        &tex,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    )]);
                }
                Ok(())
            })?;
        } else {
            let enc = gpu_enc.as_ref().unwrap();
            let row_pitch = d3d12::aligned_pitch(t.w as usize * 4);
            for y in 0..t.h as usize {
                let row = &t.texels[y * t.w as usize..(y + 1) * t.w as usize];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        row.as_flattened().as_ptr(),
                        stage.ptr.add(y * row_pitch),
                        t.w as usize * 4,
                    )
                };
            }
            let (tw, th) = (t.w, t.h);
            hg.run_list(&mut |l| {
                enc.record_encode(
                    l,
                    &stage.resource,
                    tw,
                    th,
                    row_pitch as u32,
                    bc7::blocks(th),
                    super::bc7gpu::effort(q),
                );
                enc.record_copy_out(l, &tex, 0, 0, fmt, tw, th);
                unsafe {
                    l.ResourceBarrier(&[transition(
                        &tex,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    )])
                };
                Ok(())
            })?;
        }
        let dec = bc7_decode_tex(hg, &rk, &tex, t.w, t.h, &out)?;
        let mut sq = 0f64;
        for (px, src) in t.texels.iter().enumerate() {
            for c in 0..3 {
                let d = dec[px * 4 + c].abs_diff(src[c]) as u32;
                sum_abs += d as f64;
                sq += (d as f64) * (d as f64);
                max_abs = max_abs.max(d);
            }
            n_samples += 3;
        }
        let mse = sq / (t.texels.len() as f64 * 3.0);
        let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { 99.0 };
        // Per-texture diagnostic (which texture holds the worst PSNR): the
        // FR_ABL read-only-probe idiom, off by default.
        if std::env::var_os("FR_BC7_DUMP").is_some() {
            eprintln!(
                "bc7 fid: {:>7.1} dB  {}x{} srgb={} {}",
                psnr, t.w, t.h, t.srgb, t.source
            );
        }
        worst_psnr = worst_psnr.min(psnr);
    }
    Ok(Some(Bc7Fidelity {
        textures: ids.len(),
        mean_abs: sum_abs / n_samples as f64,
        max_abs,
        worst_psnr,
    }))
}

/// The M11 `.Load`-decode kernel, packaged so `bc7_fidelity` and the
/// structural gate below share one implementation: [0] table of one SRV
/// (t0), [1] root UAV (u0), [2] two root constants (b0: W, H).
struct Bc7ReadKernel {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
}

fn bc7_read_kernel(device: &ID3D12Device) -> Result<Bc7ReadKernel> {
    let ranges = [D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        ..Default::default()
    }];
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
    unsafe { D3D12SerializeRootSignature(&sig_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None) }
        .map_err(|e| format!("bc7 read root sig serialize: {e}"))?;
    let blob = blob.unwrap();
    let root: ID3D12RootSignature = unsafe {
        device.CreateRootSignature(
            0,
            std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()),
        )
    }
    .map_err(|e| format!("bc7 read root sig: {e}"))?;
    let cs = super::tonemap::compile(
        BC7_READ_HLSL,
        windows::core::s!("cs_bc7_read"),
        windows::core::s!("cs_5_0"),
        "bc7_read",
    )?;
    let pso: ID3D12PipelineState = unsafe {
        device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: cs.GetBufferPointer(),
                BytecodeLength: cs.GetBufferSize(),
            },
            ..Default::default()
        })
    }
    .map_err(|e| format!("bc7 read PSO: {e}"))?;
    let heap: ID3D12DescriptorHeap = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            ..Default::default()
        })
    }
    .map_err(|e| format!("bc7 read heap: {e}"))?;
    Ok(Bc7ReadKernel { root, pso, heap })
}

/// Hardware-decode a `BC7_UNORM` texture (resting in NPSR) back to packed
/// RGBA8 bytes via the spec-bit-exact `.Load` path. `out` must be an
/// UNORDERED_ACCESS buffer of at least `w*h*4` bytes.
fn bc7_decode_tex(
    hg: &mut HeadlessGpu,
    rk: &Bc7ReadKernel,
    tex: &ID3D12Resource,
    w: u32,
    h: u32,
    out: &ID3D12Resource,
) -> Result<Vec<u8>> {
    use super::d3d12::Submit;
    let device = hg.device.clone();
    let fmt = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC7_UNORM;
    unsafe {
        device.CreateShaderResourceView(
            tex,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: fmt,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                },
            }),
            rk.heap.GetCPUDescriptorHandleForHeapStart(),
        )
    };
    hg.run_list(&mut |l| {
        unsafe {
            l.SetDescriptorHeaps(&[Some(rk.heap.clone())]);
            l.SetComputeRootSignature(&rk.root);
            l.SetPipelineState(&rk.pso);
            l.SetComputeRootDescriptorTable(0, rk.heap.GetGPUDescriptorHandleForHeapStart());
            l.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            l.SetComputeRoot32BitConstants(2, 2, [w, h].as_ptr() as *const _, 0);
            l.Dispatch(w.div_ceil(8), h.div_ceil(8), 1);
        }
        Ok(())
    })?;
    hg.read_buffer(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, (w * h) as usize * 4)
}

/// The `--check-gpu` STRUCTURAL gate for the GPU BC7 encoder — synthetic
/// textures, so it fires on every scene including the untextured procedural
/// default (where M11 skips). Three teeth:
///
/// 1. A flat 16x16 whose channels all agree in parity (all-even) must come
///    back through the hardware decoder BIT-EXACT — mode 6 represents such a
///    color exactly via e0 == e1, so any loss here is a wiring bug, not
///    quantization.
/// 2. Every block of that flat texture must be byte-identical to block 0 —
///    the CPU self_test's stride-bug catch, ported: a tight-packed store
///    (ignoring `block_pitch`) or a wrong row stride makes blocks differ.
/// 3. A gradient ramp at both effort tiers must clear 30 dB RGB PSNR — a
///    sanity bar the mode-6 fit passes with a wide margin (a broken index /
///    anchor / endpoint path lands far below it).
pub fn bc7_gpu_self_test(hg: &mut HeadlessGpu) -> Result<()> {
    use super::d3d12::Submit;
    let device = hg.device.clone();
    let fmt = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC7_UNORM;
    let enc = super::bc7gpu::Bc7Enc::new(&device, d3d12::block_pitch(64) * bc7::blocks(64) as usize)
        .map_err(|e| format!("bc7-gpu: encoder construction: {e}"))?;
    let rk = bc7_read_kernel(&device)?;
    let out = committed_buffer(
        &device,
        64 * 64 * 4,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )?;
    let stage = d3d12::UploadBuffer::new(&device, d3d12::aligned_pitch(64 * 4) * 64)?;

    // Encode w×h texels on the GPU; return (raw block-buffer bytes, decoded
    // RGBA8) — the pair every tooth below is scored on.
    let encode = |hg: &mut HeadlessGpu,
                      w: u32,
                      h: u32,
                      texels: &[[u8; 4]],
                      effort: u32|
     -> Result<(Vec<u8>, Vec<u8>)> {
        let row_pitch = d3d12::aligned_pitch(w as usize * 4);
        for y in 0..h as usize {
            let row = &texels[y * w as usize..(y + 1) * w as usize];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row.as_flattened().as_ptr(),
                    stage.ptr.add(y * row_pitch),
                    w as usize * 4,
                )
            };
        }
        let tex = d3d12::committed_tex(
            &device,
            w,
            h,
            fmt,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        hg.run_list(&mut |l| {
            enc.record_encode(l, &stage.resource, w, h, row_pitch as u32, bc7::blocks(h), effort);
            enc.record_copy_out(l, &tex, 0, 0, fmt, w, h);
            unsafe {
                l.ResourceBarrier(&[transition(
                    &tex,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )])
            };
            Ok(())
        })?;
        let blocks = hg.read_buffer(
            &enc.block_buf,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            d3d12::block_pitch(w) * bc7::blocks(h) as usize,
        )?;
        let dec = bc7_decode_tex(hg, &rk, &tex, w, h, &out)?;
        Ok((blocks, dec))
    };

    // Tooth 1 + 2: the all-even flat block.
    let flat_c = [200u8, 30, 30, 255];
    let flat: Vec<[u8; 4]> = vec![flat_c; 16 * 16];
    let (blocks, dec) = encode(hg, 16, 16, &flat, 1)?;
    let pitch = d3d12::block_pitch(16);
    let b0 = &blocks[0..bc7::BLOCK_BYTES];
    if b0.iter().all(|&b| b == 0) {
        return Err("bc7-gpu: flat block encoded to all zeros (kernel did not run?)".into());
    }
    for by in 0..bc7::blocks(16) as usize {
        for bx in 0..bc7::blocks(16) as usize {
            let b = &blocks[by * pitch + bx * bc7::BLOCK_BYTES..][..bc7::BLOCK_BYTES];
            if b != b0 {
                return Err(format!(
                    "bc7-gpu: flat texture block ({bx},{by}) differs from block 0 (stride bug? \
                     the store must honor block_pitch)"
                ));
            }
        }
    }
    for (px, d) in dec.chunks_exact(4).enumerate() {
        if d[0] != flat_c[0] || d[1] != flat_c[1] || d[2] != flat_c[2] {
            return Err(format!(
                "bc7-gpu: all-even flat color must round-trip BIT-EXACT; px {px} decoded \
                 ({},{},{}) want ({},{},{})",
                d[0], d[1], d[2], flat_c[0], flat_c[1], flat_c[2]
            ));
        }
    }

    // Tooth 3: gradient ramp, every effort tier (0/1 = mode-6 depth, 2/3 =
    // the always-mode-1 arms — a smooth ramp must never get WORSE for
    // trying the two-subset mode, since the chooser keeps the lower SSE).
    let ramp: Vec<[u8; 4]> = (0..64u32)
        .flat_map(|y| {
            (0..64u32).map(move |x| [(x * 4 + 1) as u8, (y * 4) as u8, (x * 2 + y * 2) as u8, 255])
        })
        .collect();
    for effort in [0u32, 1, 2, 3] {
        let (_, dec) = encode(hg, 64, 64, &ramp, effort)?;
        let mut sq = 0f64;
        for (px, s) in ramp.iter().enumerate() {
            for c in 0..3 {
                let d = dec[px * 4 + c].abs_diff(s[c]) as f64;
                sq += d * d;
            }
        }
        let mse = sq / (ramp.len() as f64 * 3.0);
        let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { 99.0 };
        if psnr < 30.0 {
            return Err(format!(
                "bc7-gpu: ramp PSNR {psnr:.1} dB < 30 at effort {effort} (encoder math broken?)"
            ));
        }
    }

    // Tooth 4: a two-CLUSTER block (a red-ish pair left, a blue-ish pair
    // right — four colors no single line can fit). Mode 6 alone leaves
    // ~20-LSB errors; a small max error therefore proves the mode-1 arm
    // FIRED and that its partition/anchor tables and packing agree with the
    // hardware decoder (a wrong table entry decodes texels against the
    // wrong subset and the error explodes).
    let cl = |x: usize, y: usize| -> [u8; 4] {
        match (x < 2, y % 2 == 0) {
            (true, true) => [200, 0, 0, 255],
            (true, false) => [220, 20, 20, 255],
            (false, true) => [0, 0, 200, 255],
            (false, false) => [20, 20, 220, 255],
        }
    };
    let pair: Vec<[u8; 4]> =
        (0..4usize).flat_map(|y| (0..4usize).map(move |x| cl(x, y))).collect();
    let (_, dec) = encode(hg, 4, 4, &pair, 1)?;
    let mut worst = 0u32;
    for (px, s) in pair.iter().enumerate() {
        for c in 0..3 {
            worst = worst.max(dec[px * 4 + c].abs_diff(s[c]) as u32);
        }
    }
    if worst > 6 {
        return Err(format!(
            "bc7-gpu: two-cluster block max err {worst} LSB > 6 — the mode-1 arm did not fire \
             or its partition/anchor/packing disagree with the hardware decoder"
        ));
    }
    Ok(())
}

