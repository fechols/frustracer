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
use crate::bc7;
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
/// 3 is the ceiling, not a guess: the engines that need a DISTINCT plane set are
/// DLSS-RR, FSR4-RR, and the XeSS/FSR3 trio — XeSS and FSR 3.1 take a
/// byte-identical plane set, so they share one feed (see `ffx_up::upscale_res_shared`).
pub const FEED_SETS: u32 = 3;
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
pub const NPPD_REG_BASE: u32 = NUM_UAVS + 2 + NUM_FEED; // u26
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
/// First heap slot of the scene-texture table (after every feed set).
pub const TEX_HEAP_BASE: u32 = FEED_SETS * FEED_SET_STRIDE;
/// Buffer-SRV descriptors preceding the Texture2D array in the table
/// (t0..t6 space1: texcoords, indices alias, tri_mat alias, mat_cutout,
/// positions alias, mat_height, mat_shadow — texs[] starts at t7).
pub const TEX_TABLE_BUFS: u32 = 7;

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
pub const CTR_ALPHA_REJ: u32 = 18;
pub const CTR_HEIGHT_REJ: u32 = 19;
/// Tinted-shadow candidate passes (TRANS_SHADOW scenes) — the anti-vacuity
/// stat proving occlusion rays really crossed transmissive surfaces.
pub const CTR_TRANS_PASS: u32 = 20;
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

// Root-CBV alignment (256 B). FrameCb is 2480 bytes — 288 of struct plus the
// MAX_SPP-entry jitter table (--spp), the SH sky rows, and the MAX_FIREFLIES
// pose rows, which are what set the size (raise the stride in lockstep with
// either cap; the const asserts below police both directions).
pub(crate) const CB_STRIDE: usize = 2560;

/// The leaf kernel's thread-group width — ONE WAVE on both vendors, and that
/// is the whole point.
///
/// A leaf tile is not 8x8. `depth_full` is driven by the WIDER screen axis, so
/// at 1920x1080 a leaf rect is 1920/2^8 = 7.5 by 1080/2^8 = 4.2 — about **32
/// pixels**, never 64. The kernel used to dispatch 64 lanes per tile and let
/// the surplus half return immediately, which is nearly free on a wave32 GPU
/// (the all-idle second wave retires at once) and expensive on wave64, where
/// those lanes sit in the SAME wave and waste half its RT throughput. That one
/// mismatch was most of the AMD-vs-NVIDIA gap: per extra sample the leaf kernel
/// cost 2.27x its own reference kernel on RDNA but only 1.24x on Ada, for
/// identical work.
///
/// leaf.hlsl grid-strides over the tile's pixels, so this is a free knob.
/// Measured (--gpu-timing, leaf+sky, 1080p; 64 -> 32):
///   spp=1   AMD 1.63 -> 1.01 ms (-38%)   NVIDIA 2.24 -> 1.38 ms (-38%)
///   spp=16  AMD 19.7 -> 11.4 ms (-42%)   NVIDIA 10.2 ->  7.6 ms (-25%)
/// i.e. a win on BOTH vendors, not an AMD-specific hack — a 64-thread group
/// reserves registers for 64 threads on Ada too, so halving it doubles the
/// blocks in flight.
///
/// 32 is a floor, not a tuning parameter: RDNA's wave is 32 lanes MINIMUM, so
/// a 16-wide group is a half-empty wave again (measured worse — 1.31 ms AMD).
/// And it never loses at other resolutions: a tile larger than 32 px simply
/// takes a second full lap, which is the same lane utilization a 64-wide group
/// would have had.
const LEAF_GROUP_DEF: &str = "#define LEAF_GROUP 32";

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
/// Order-2 SH irradiance, standalone (no cbuffer of its own — the coefficients
/// are a parameter). pub(crate) because gpu/ffx_rr.rs prepends it to the FSR
/// composite pass, which needs the SAME evaluator but binds its own 9 rows.
pub(crate) const SH_HLSLI: &str = include_str!("shaders/sh.hlsli");
/// The FSR plane wire encodings (octahedral normals + their 10-bit quantum).
/// pub(crate) for the same reason: feed.hlsl writes those planes and
/// fsr_composite.hlsl reads them back, and the composite identity is exactly
/// the claim that the two agree — so they share one copy.
pub(crate) const FSR_WIRE_HLSLI: &str = include_str!("shaders/fsr_wire.hlsli");
// pub(crate): gpu/dxr.rs pastes the same prelude/shading/resolve sources
// into its DXR library (the kernels are single-sourced on disk).
//
// sh.hlsli leads: trace_common's `sh_irradiance` is just the frame's cbuffer
// bound to it. Folding it in here rather than at each of the dozen concat sites
// keeps the prelude one name.
pub(crate) const TRACE_COMMON_HLSLI: &str = concat!(
    include_str!("shaders/sh.hlsli"),
    "\n",
    include_str!("shaders/trace_common.hlsli")
);
const CTR_HLSLI: &str = include_str!("shaders/ctr.hlsli");
const QUEUES_HLSLI: &str = include_str!("shaders/queues.hlsli");
const FRUSTUM_HLSLI: &str = include_str!("shaders/frustum.hlsli");
// The 8-wide frustum tree's bound_query/refine_cut, `#ifdef FTREE`-guarded —
// pasted right after FRUSTUM_HLSLI (whose binary halves are `#ifndef FTREE`);
// the ftree_defs prelude picks the structure per session.
const FTREE_HLSLI: &str = include_str!("shaders/ftree.hlsli");
const RT_HLSLI: &str = include_str!("shaders/rt.hlsli");
pub(crate) const SHADE_HLSLI: &str = include_str!("shaders/shade.hlsli");
const HEMI_HLSLI: &str = include_str!("shaders/hemi.hlsli");
const REFERENCE_HLSL: &str = include_str!("shaders/reference.hlsl");
pub(crate) const RESOLVE_HLSL: &str = include_str!("shaders/resolve.hlsl");
const WAVEFRONT_HLSL: &str = include_str!("shaders/wavefront.hlsl");
const LEAF_HLSL: &str = include_str!("shaders/leaf.hlsl");
const HEMI_WAVE_HLSL: &str = include_str!("shaders/hemi_wave.hlsl");
const HEMI_LEAF_HLSL: &str = include_str!("shaders/hemi_leaf.hlsl");
const COMPOSE_HLSL: &str = include_str!("shaders/compose.hlsl");
pub(crate) const FEED_HLSL: &str = include_str!("shaders/feed.hlsl");
const NPPD_HLSL: &str = include_str!("shaders/nppd.hlsl");

/// What the GPU tracer requires, queried once. RayQuery in compute needs
/// RaytracingTier 1.1 AND shader model 6.5; missing either is a clean
/// "use the CPU path" story, never a degraded half-mode.
pub struct Caps {
    pub rt_tier: i32,
    pub shader_model: i32,
    pub binding_tier: i32,
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
    Ok(Caps {
        rt_tier: o5.RaytracingTier.0,
        shader_model: sm.HighestShaderModel.0,
        binding_tier: o.ResourceBindingTier.0,
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
    pub fn new(debug: bool, prefer: adapter::Prefer) -> Result<Self> {
        let factory = adapter::create_factory(debug).map_err(|e| format!("factory: {e}"))?;
        let pick = adapter::pick(&factory, prefer)?;
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
        // --gpu-timing: every run() blocks on the fence below, so the previous
        // run's timestamps are complete by the time we get here — the same
        // wait-then-collect the frame ring gives `D3d::begin_frame`, with one
        // slot instead of FRAMES_IN_FLIGHT. Without this the headless suites
        // (the deterministic workloads, and the only place a per-pass number
        // means anything) would record no timings at all.
        super::gputime::begin_frame(&self.device, &self.queue, 0);
        f(&self.list);
        super::gputime::resolve(&self.list, 0);
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
/// 80 B — the HLSL `Mat` mirrors this field-for-field; a stride skew reads
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
}

/// Bytes of reusable staging streamed per blocking submit — bounds the
/// upload's transient commit to one chunk instead of a full second copy of
/// every scene stream (which at 100M tris was ~7 GB of upload heaps ON TOP of
/// ~7 GB of repack Vecs).
const STAGE_CHUNK: usize = 256 << 20;

/// Which software acceleration structure(s) ride t0 for the FRUSTUM queries
/// (their only consumer — every actual ray is DXR RayQuery). `Bvh` = the
/// binary tree alone; `Both` = binary PLUS the 8-wide frustum tree — the
/// per-consumer split ON the GPU: the tile kernels compile `#define FTREE`
/// and bind the wide tree at t0 (long queries, wide wins big), while
/// `record_hemi` rebinds the binary tree for the hemi kernels (hemi bound
/// queries terminate in ~10 visits — a binary pop is 1 box test where a wide
/// pop is always 8, and the wide tree measured +35% there). `None` =
/// DXR-only, dummies.
#[derive(Clone, Copy)]
pub enum SwAccel<'a> {
    Bvh(&'a Bvh),
    Both(&'a Bvh, &'a crate::ftree::FTree),
    None,
}

/// Steady-state byte total of the scene's buffer streams (excludes textures
/// and acceleration structures) — sizes the staging ring and the init report.
fn scene_stream_bytes(scene: &Scene, sw_bvh: SwAccel) -> usize {
    let v = scene.positions.len();
    let t = scene.indices.len();
    let m = scene.materials.len();
    let bin = |b: &Bvh| b.nodes.len() * size_of::<GpuBvhNode>() + b.tri_idx.len() * 4;
    let bvh = match sw_bvh {
        SwAccel::Bvh(b) => bin(b),
        SwAccel::Both(b, ft) => bin(b) + ft.quantized_bytes(),
        SwAccel::None => 0,
    };
    bvh + v * (12 + 12 + 8) + t * (12 + 4) + m * (size_of::<GpuMat>() + 4)
}

/// Stream `src` into a new default-heap buffer through `ring`, `map`ping each
/// element into the mapped staging pointer chunk-by-chunk (identity for
/// layout-compatible streams, a repack for Vec3A→float3 / BvhNode→GpuBvhNode).
/// Each chunk is one blocking `Submit::run_list`; the final chunk's list also
/// records the COPY_DEST→`after` transition. Empty streams get a 4-byte dummy
/// created directly in `after`.
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

pub struct SceneGpu {
    pub bvh_nodes: ID3D12Resource,
    pub tri_idx: ID3D12Resource,
    /// The 8-wide frustum tree (SwAccel::Both sessions): bound at t0 by the
    /// TILE dispatches in place of `bvh_nodes`; the hemi dispatches rebind
    /// the binary tree (record_hemi) — the per-consumer split on the GPU.
    pub ftree_nodes: Option<ID3D12Resource>,
    pub positions: ID3D12Resource,
    pub normals: ID3D12Resource,
    pub indices: ID3D12Resource,
    pub tri_mat: ID3D12Resource,
    pub materials: ID3D12Resource,
    /// Never read, but MUST be held: the TLAS instance desc bakes only the
    /// (compacted) BLAS's GPU VA — dropping this resource would free the
    /// memory the TLAS points into.
    #[allow(dead_code)]
    pub blas: ID3D12Resource,
    pub tlas: ID3D12Resource,
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
    n_verts: u32,
    n_tris: u32,
    n_mats: u32,
}

impl SceneGpu {
    /// Create AND upload the scene in one call: every stream chunks through
    /// one reusable `STAGE_CHUNK` staging ring (blocking submits via `sub`),
    /// then the BLAS + TLAS build rides a final submit — scratch and staging
    /// are gone by the time this returns, so peak commit is
    /// steady-state + one chunk, not steady-state × 2. `sw_bvh: None` is the
    /// DXR-only session: the software BVH (frustum kernels' tree) is never
    /// bound there, so `bvh_nodes`/`tri_idx` become 4-byte dummies (~2.3 GB
    /// saved at 100M tris).
    /// `bc7_q`: block-compress the OPAQUE scene textures at this ISPC quality
    /// before upload (`--bc7`; `None` = today's RGBA8 everywhere). Alpha-masked
    /// cutout textures stay RGBA8 either way — see src/bc7.rs.
    pub fn new_uploaded(
        device: &ID3D12Device,
        scene: &Scene,
        sw_bvh: SwAccel,
        sub: &mut dyn d3d12::Submit,
        bc7_q: Option<bc7::Quality>,
    ) -> Result<Self> {
        let device5: ID3D12Device5 = device
            .cast()
            .map_err(|e| format!("ID3D12Device5 (require_caps should have gated): {e}"))?;

        let srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        // The texture band loop below needs at least ONE full aligned row in
        // the ring (`scene_stream_bytes` deliberately excludes textures, and
        // its geometry-only size can undershoot a wide texture's pitch on a
        // small mesh — the band `.max(1)` would then overrun the mapping).
        // A BC7 texture's "row" is a 4-texel-tall BLOCK row, so its pitch is
        // the block pitch, not w*4 — mispredicting this here is exactly the
        // overrun the comment above warns about.
        let max_tex_pitch = scene
            .textures
            .iter()
            .map(|t| {
                if bc7_q.is_some() && bc7::should_compress(t) {
                    d3d12::block_pitch(t.w)
                } else {
                    d3d12::aligned_pitch(t.w as usize * 4)
                }
            })
            .max()
            .unwrap_or(0);
        let ring = d3d12::UploadBuffer::new(
            device,
            STAGE_CHUNK
                .min(scene_stream_bytes(scene, sw_bvh).max(4096))
                .max(max_tex_pitch),
        )?;

        let up_bin = |sub: &mut dyn d3d12::Submit, bvh: &Bvh| -> Result<(ID3D12Resource, ID3D12Resource)> {
            Ok((
                stream_buffer(
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
                )?,
                stream_buffer(device, sub, &ring, &bvh.tri_idx, |t| *t, srv)?,
            ))
        };
        let (bvh_nodes, tri_idx, ftree_nodes) = match sw_bvh {
            SwAccel::Bvh(bvh) => {
                let (n, t) = up_bin(sub, bvh)?;
                (n, t, None)
            }
            // The per-consumer split: BOTH structures upload — the tile
            // kernels bind the wide tree at t0 (bind_common), the hemi
            // kernels the binary one (record_hemi's rebind). The wide tree
            // uploads in its QUANTIZED wire format (ftree::QFNode, 112 B —
            // the HLSL FtNode mirror; self_test audits containment): the
            // per-processor split verdict — the CPU keeps the f32 nodes,
            // the GPU trades decode ALU for -56% tree bandwidth/VRAM.
            SwAccel::Both(bvh, ft) => {
                let (n, t) = up_bin(sub, bvh)?;
                let qn = ft.quantized();
                let f = stream_buffer(device, sub, &ring, &qn, |n| *n, srv)?;
                (n, t, Some(f))
            }
            SwAccel::None => (
                committed_buffer(device, 4, D3D12_RESOURCE_FLAG_NONE, srv)?,
                committed_buffer(device, 4, D3D12_RESOURCE_FLAG_NONE, srv)?,
                None,
            ),
        };
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
            .map(|m| GpuMat {
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
        // --bc7: block-compress the OPAQUE 4-aligned textures before
        // uploading them (8 bpp vs 32 — Intel Sponza's set is 4.6 GB of VRAM
        // as RGBA8). Alpha-masked cutout textures are EXCLUDED and stay
        // RGBA8: the intersector `.Load()`s their alpha per texel against a
        // hard `< 128` threshold, and BC7 quantizes alpha across it (a .Load
        // on a BC SRV returns the DECODED — lossy — texel).
        // `bc7::should_compress`'s masked arm is the same predicate as
        // `mat_cutout` below — see src/bc7.rs for why that agreement IS the
        // soundness argument.
        //
        // There is deliberately no BC7 disk cache: the encode runs every load.
        // Largest-first (LPT) scheduling for the same reason the DECODE sites
        // sort (scene.rs / scene_cache.rs) — cost is ~linear in texels, so the
        // few 4K maps would otherwise dominate the tail while the rest idle.
        // Results scatter back by texture id, which must never shift.
        let mut bc7_blocks: Vec<Option<Vec<Vec<u8>>>> =
            scene.textures.iter().map(|_| None).collect();
        let mut enc_ms = 0.0f64;
        let mut enc_texels = 0u64;
        if let Some(q) = bc7_q {
            use rayon::prelude::*;
            let t0 = std::time::Instant::now();
            let mut order: Vec<usize> = (0..scene.textures.len())
                .filter(|&i| bc7::should_compress(&scene.textures[i]))
                .collect();
            order.sort_by_key(|&i| {
                std::cmp::Reverse(scene.textures[i].w as u64 * scene.textures[i].h as u64)
            });
            enc_texels =
                order.iter().map(|&i| scene.textures[i].w as u64 * scene.textures[i].h as u64).sum();
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

        if !scene.textures.is_empty() {
            let raw = scene.textures.iter().map(|t| t.w as u64 * t.h as u64 * 4).sum::<u64>();
            let live = scene
                .textures
                .iter()
                .enumerate()
                .map(|(i, t)| match &bc7_blocks[i] {
                    Some(b) => b.iter().map(|l| l.len() as u64).sum::<u64>(),
                    None => {
                        let base = t.w as u64 * t.h as u64;
                        let mips: u64 =
                            t.mips.iter().map(|m| m.w as u64 * m.h as u64).sum();
                        (base + mips) * 4
                    }
                })
                .sum::<u64>();
            let n_bc7 = bc7_blocks.iter().filter(|b| b.is_some()).count();
            let bc7_note = if n_bc7 > 0 {
                // Mtexel/s is the "is a load-time encode real-time?" number.
                format!(
                    ", {} BC7 + {} RGBA8, was {} MB | bc7 encode {:.0} ms ({:.0} Mtexel/s)",
                    n_bc7,
                    scene.textures.len() - n_bc7,
                    raw >> 20,
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

        // Scene textures: RGBA8 Texture2Ds — _SRGB for color textures (the
        // per-texel decode of texture.rs::sample_bilinear in hardware) and
        // plain _UNORM for linear-data maps (normal / rough-metal; the CPU
        // samples those via sample_bilinear_linear). Texels upload raw (row
        // 0 = v0, the V flip is baked at OBJ load) in row bands through the
        // same staging ring — no per-texture staging commit. Under --bc7 the
        // opaque ones instead upload as BC7 (same _SRGB/_UNORM role split),
        // staged in 4-texel-tall BLOCK rows; the blocks are dropped as we go,
        // so steady-state RAM is unchanged.
        let mut textures_v = Vec::new();
        for (i, t) in scene.textures.iter().enumerate() {
            let fmt = match &bc7_blocks[i] {
                Some(_) => bc7::dxgi_format(t),
                None => {
                    if t.srgb {
                        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
                    } else {
                        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM
                    }
                }
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
                let (pitch, src_pitch, rows_total, row_h) = match &bc7_blocks[i] {
                    // A BC7 "row" is a block row: 4 texel rows in
                    // ceil(w/4)*16 B — per-mip dims.
                    Some(_) => (
                        d3d12::block_pitch(mw),
                        bc7::blocks(mw) as usize * bc7::BLOCK_BYTES,
                        bc7::blocks(mh) as usize,
                        bc7::BLOCK as usize,
                    ),
                    None => (
                        d3d12::aligned_pitch(mw as usize * 4),
                        mw as usize * 4,
                        mh as usize,
                        1,
                    ),
                };
                let band = (ring.size / pitch).max(1).min(rows_total);
                let mut r0 = 0usize;
                while r0 < rows_total {
                    let rows = band.min(rows_total - r0);
                    for r in 0..rows {
                        let src: &[u8] = match &bc7_blocks[i] {
                            Some(b) => &b[mip][(r0 + r) * src_pitch..(r0 + r + 1) * src_pitch],
                            None => {
                                let y = r0 + r;
                                let row = &mip_texels[y * mw as usize..(y + 1) * mw as usize];
                                row.as_flattened()
                            }
                        };
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                src.as_ptr(),
                                ring.ptr.add(r * pitch),
                                src_pitch,
                            )
                        };
                    }
                    // The dst y offset and the footprint height are always in
                    // TEXELS. For BC7 both are whole block rows (DstY a
                    // multiple of 4, as the debug layer requires) except the
                    // final band, which runs to the mip's `mh` exactly — the
                    // bottom edge of the subresource, the other form the
                    // layer accepts.
                    let y0 = r0 * row_h;
                    let h_tex = (rows * row_h).min(mh as usize - y0) as u32;
                    let fp = match &bc7_blocks[i] {
                        Some(_) => d3d12::footprint_block(fmt, mw, h_tex, 0),
                        None => d3d12::footprint(fmt, mw, h_tex, 4, 0),
                    };
                    let last = mip + 1 == n_mips && r0 + rows == rows_total;
                    sub.run_list(&mut |l| {
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
                    r0 += rows;
                }
            }
            // The blocks are on the GPU now — drop them so peak RAM carries at
            // most the BC7 set (~0.25x the RGBA8 texels), never a second copy.
            bc7_blocks[i] = None;
            textures_v.push(dst);
        }

        // --- acceleration-structure sizing ---
        let n_verts = scene.positions.len() as u32;
        let n_tris = scene.indices.len() as u32;
        // OPAQUE drops when ANY conditional-hit feature is armed — the
        // any-hit/candidate machinery is shared (candidate_reject), and the
        // tinted-shadow pass needs candidates to surface in `transmit_q`.
        let non_opaque = scene.any_alpha
            || (scene.any_height && crate::bvh::height_armed())
            || scene.any_transmissive;
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
            scene_stream_bytes(scene, sw_bvh) >> 20,
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
            bvh_nodes,
            tri_idx,
            ftree_nodes,
            positions: positions_b,
            normals: normals_b,
            indices: indices_b,
            tri_mat,
            materials: materials_b,
            blas,
            tlas,
            texcoords: texcoords_b,
            mat_cutout: mat_cutout_b,
            mat_height: mat_height_b,
            mat_shadow: mat_shadow_b,
            textures: textures_v,
            n_verts,
            n_tris,
            n_mats: scene.materials.len() as u32,
        })
    }

    /// Write the RP_SCENE_TEX table's descriptors into `heap` at slots
    /// `base..`: 7 buffer SRVs (texcoords, indices, tri_mat, mat_cutout,
    /// positions, mat_height, mat_shadow — t0..t6 space1) then one
    /// Texture2D SRV per scene texture (t7.. space1).
    /// The heap must be sized `base + TEX_TABLE_BUFS + textures.len()`.
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

/// Per-scene HLSL prelude: alpha-masked scenes compile the cutout candidate
/// loops / any-hit shaders in; opaque scenes compile byte-identical sources
/// to the pre-cutout tracer. Shared with dxr.rs (the DXR library concat).
pub(crate) fn alpha_defs(scene: &Scene) -> &'static str {
    if scene.any_alpha { "#define ALPHA_CUTOUT 1" } else { "" }
}

/// The relief twin of `alpha_defs`: height-carrying scenes compile the march
/// + candidate loops / any-hit shaders in (runtime-gated by FLAG_HEIGHT —
/// the V toggle); scenes without height data compile byte-identical sources
/// to the pre-relief tracer. Shared with dxr.rs.
pub(crate) fn height_defs(scene: &Scene) -> &'static str {
    if scene.any_height && crate::bvh::height_armed() { "#define HEIGHTFIELD 1" } else { "" }
}

/// The tinted-shadows twin: transmissive scenes compile `transmit_q`'s
/// candidate loop / the ah_shadow tint arm in (`Scene::any_transmissive`
/// already folds the `--no-tinted-shadows` lever); scenes without
/// transmissive materials compile byte-identical sources to the binary
/// occlusion tracer. Shared with dxr.rs.
pub(crate) fn trans_defs(scene: &Scene) -> &'static str {
    if scene.any_transmissive { "#define TRANS_SHADOW 1" } else { "" }
}

/// HLSL prelude every compile unit takes: the `--spp` jitter table's row count
/// (`FrameCb::jitters`, hand-mirrored in trace_common.hlsli's cbuffer). The
/// SIZE is derived from `dlss::MAX_SPP` rather than written twice — a literal
/// there would be a third constant to raise in lockstep, and a shader reading
/// past a too-small array is silent (no gate can see it). Injected like
/// ALPHA_CUTOUT / FTREE.
pub(crate) fn spp_defs() -> String {
    // The sky-fill's extra-sample offsets (cs_sky under FLAG_CLOUDS at
    // spp > 1): PHASE-0 Halton, deliberately frame-INDEPENDENT — a proven-
    // empty tile antialiases a static function, and per-frame offsets put
    // inter-frame dither on cloud edges that the spp stability gate (rightly)
    // rejects at night. Injected as literals ({:.9e} — 10 significant digits,
    // past f32's 9 — so the HLSL parses back the exact bits) because the CB
    // jitter table carries the FRAME's phase and the CB has no room for a
    // second one; the frame-0 gates still match the reference kernel exactly
    // (its jitters[] ARE jitter_for_sample(0, s) there). The CPU twin is
    // fill_sky_rows' jitter_for_sample(0, k).
    let sky_j: String = (0..crate::dlss::MAX_SPP)
        .map(|k| {
            let (x, y) = crate::dlss::jitter_for_sample(0, k);
            format!("float2({x:.9e}, {y:.9e})")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "#define MAX_SPP {}u\n#define JITTER_ROWS {}\n#define MAX_FIREFLIES {}\nstatic const float2 SKY_J[MAX_SPP] = {{ {} }};",
        crate::dlss::MAX_SPP,
        crate::dlss::MAX_SPP / 2,
        // The firefly pose-row count (fireflies.rs::MAX_FIREFLIES), injected
        // for the same reason as JITTER_ROWS: a hand-mirrored literal would
        // be a second constant to raise in lockstep, and a shader reading
        // past a too-small cbuffer array is silent.
        crate::fireflies::MAX_FIREFLIES,
        sky_j
    )
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
/// direct-light signals (GBufPx.sig) and the prev-camera view-Z (mv.z);
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

/// GBufPx stride in bytes — lockstep with trace_common.hlsli's struct
/// (nr | alb_z | spec | mv | sig | sig2 = 4 float4 + 1 uint4 + 1 uint2).
pub const GBUF_STRIDE: u64 = 88;

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
    sun: [f32; 4],   // xyz = unit dir; w = cos(angular radius)
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
}
// The HLSL cbuffer is hand-mirrored across 7 concatenated compile units —
// a size drift here corrupts every field after the drift point.
// 304 (the pre-sun size) − 32 (two rect-light rows dropped) + 16 (the spp
// block) + 8·MAX_SPP (the jitter table) + 16·9 (the SH sky) +
// 16·MAX_FIREFLIES (the firefly pose rows).
const _: () = assert!(
    std::mem::size_of::<FrameCb>()
        == 320 - 32
            + 8 * crate::dlss::MAX_SPP as usize
            + 16 * crate::sh::N
            + 16 * crate::fireflies::MAX_FIREFLIES
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
    pub(crate) fn with_frame(&self, p: &FrameParams, gbuf_full: bool, fsr_sig: bool) -> FrameCb {
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
            | (crate::scene::depth_tint() as u32 * FLAG_DEPTH_TINT);
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
        rw: u32,
        rh: u32,
        gbuf_full: bool,
        nppd: bool,
        debug: bool,
        bc7_q: Option<bc7::Quality>,
        sub: &mut dyn d3d12::Submit,
    ) -> Result<Self> {
        require_caps(device)?;
        let root_sig = create_root_signature(device)?;
        let cmd_sig = create_dispatch_signature(device)?;

        // Alpha-masked scenes compile the cutout candidate loops into the
        // trace primitives (rt.hlsli); height-carrying scenes likewise
        // compile the relief march in (runtime-gated by FLAG_HEIGHT — the V
        // toggle); transmissive scenes compile transmit_q's tinted candidate
        // loop in (TRANS_SHADOW). Scenes with none compile the FORCE_OPAQUE
        // originals verbatim (modulo leading blank lines) — procedural/stress
        // sessions are structurally untouched (the bit gates rely on that).
        let defs =
            format!("{}\n{}\n{}", alpha_defs(scene), height_defs(scene), trans_defs(scene));
        let defs = defs.as_str();
        // The session's frustum structure: `#define FTREE` swaps frustum.hlsli's
        // binary bound_query/refine_cut for ftree.hlsli's wide bodies (same
        // signatures — the call sites don't know), and the FNode array uploads
        // at t0 in place of the binary nodes. --no-ftree keeps the binary path.
        let ftree_on = crate::ftree::FTREE_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let ft_defs = if ftree_on { "#define FTREE 1" } else { "" };
        // The cbuffer's jitter-table size (--spp) — every unit sees the cbuffer.
        let sd = spp_defs();
        let sd = sd.as_str();
        let reference_src =
            [defs, sd, TRACE_COMMON_HLSLI, RT_HLSLI, SHADE_HLSLI, REFERENCE_HLSL].join("\n");
        let resolve_src = [sd, TRACE_COMMON_HLSLI, RESOLVE_HLSL].join("\n");
        let wavefront_src = [
            ft_defs,
            sd,
            TRACE_COMMON_HLSLI,
            CTR_HLSLI,
            QUEUES_HLSLI,
            FRUSTUM_HLSLI,
            FTREE_HLSLI,
            WAVEFRONT_HLSL,
        ]
        .join("\n");
        // Two leaf kernels from the one source. `fb_mode` is a cbuffer value,
        // so leaving the hemi arm as a runtime branch inlines shade_split at
        // both call sites and the kernel's register allocation is the MAX of
        // the two — which on RDNA costs occupancy (and therefore latency
        // hiding) in every fb-OFF frame, i.e. essentially all of them.
        // `LEAF_NO_FB` compiles that arm out; record_wavefront picks per frame.
        let leaf_of = |extra: &str| {
            [
                LEAF_GROUP_DEF,
                extra,
                defs,
                sd,
                TRACE_COMMON_HLSLI,
                CTR_HLSLI,
                QUEUES_HLSLI,
                RT_HLSLI,
                SHADE_HLSLI,
                LEAF_HLSL,
            ]
            .join("\n")
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
            sd,
            TRACE_COMMON_HLSLI,
            CTR_HLSLI,
            HEMI_HLSLI,
            FRUSTUM_HLSLI,
            RT_HLSLI,
            HEMI_WAVE_HLSL,
        ]
        .join("\n");
        let hemi_leaf_src = [
            defs,
            sd,
            TRACE_COMMON_HLSLI,
            CTR_HLSLI,
            HEMI_HLSLI,
            RT_HLSLI,
            SHADE_HLSLI,
            HEMI_LEAF_HLSL,
        ]
        .join("\n");
        let compose_src = [sd, TRACE_COMMON_HLSLI, CTR_HLSLI, QUEUES_HLSLI, COMPOSE_HLSL].join("\n");
        let feed_src = [sd, TRACE_COMMON_HLSLI, FSR_WIRE_HLSLI, FEED_HLSL].join("\n");
        let pso = |src: &str, entry: &str, what: &str| -> Result<ID3D12PipelineState> {
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
        let sw = match &ft {
            Some(t) => SwAccel::Both(bvh, t),
            None => SwAccel::Bvh(bvh),
        };
        let scene_gpu = SceneGpu::new_uploaded(device, scene, sw, sub, bc7_q)?;
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
        // only in upscaler sessions — plain sessions bind a stride-sized
        // dummy and never set FLAG_GBUF (root UAVs have no bounds check).
        let gbuf =
            committed_buffer(device, if gbuf_full { px * GBUF_STRIDE } else { GBUF_STRIDE }, uaf, ua)?;
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
    pub fn write_cb(&self, slot: usize, p: &FrameParams) {
        self.cb_base
            .with_frame(p, self.gbuf_full, self.fsr_sig())
            .store(unsafe { self.frame_cb.ptr.add(slot * CB_STRIDE) });
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
    fn fsr_sig(&self) -> bool {
        self.feed.iter().any(|(k, _)| matches!(k, FeedKind::FsrRr))
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
            let t0 = s.ftree_nodes.as_ref().unwrap_or(&s.bvh_nodes);
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_BVH_NODES,
                t0.GetGPUVirtualAddress(),
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
            // The scene-texture table (t0..t3 + texs[] in space1). The heap
            // must be set before the table; resolve/feed re-setting the same
            // heap later is redundant-but-legal.
            list.SetDescriptorHeaps(&[Some(self.uav_heap.clone())]);
            list.SetComputeRootDescriptorTable(RP_SCENE_TEX, self.tex_table);
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
                // fb frames need the hemi arm; every other frame takes the
                // slim kernel (leaf.hlsl's LEAF_NO_FB).
                let leaf_pso =
                    if fb_mode > 0 { &self.pso_leaf_fb } else { &self.pso_leaf };
                list.SetPipelineState(leaf_pso);
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
            // pool (the primary qleaf/cut_pool are done for this frame) —
            // and t0 back to the BINARY tree: the hemi kernels compile the
            // binary bound_query (short queries lose on the wide tree; see
            // SwAccel), while bind_common bound the wide one for the tiles.
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_BVH_NODES,
                self.scene.bvh_nodes.GetGPUVirtualAddress(),
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
// M11 (--bc7): encode fidelity, measured on the GPU.

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

/// Re-encode every compressible texture (deterministic — `bc7::self_test`
/// pins it, so these blocks ARE the session's), upload each as a plain
/// `BC7_UNORM` Texture2D (deliberately never `_SRGB`: the kernel must read
/// raw code values, not the transfer function), GPU-decode it back with
/// `BC7_READ_HLSL`, and diff against the CPU RGBA8 source.
///
/// RGB only: nothing ever samples a compressed texture's alpha (the cutout
/// path reads only the alpha-masked RGBA8 set, and shade.hlsli consumes
/// .rgb/.g/.b), and "opaque" merely means every alpha ≥ 250 — a 252 would
/// quantize and show up here as false loss.
///
/// `Ok(None)` = nothing compressible (untextured scene, or every texture
/// masked/odd-dim).
pub fn bc7_fidelity(
    scene: &Scene,
    q: bc7::Quality,
    hg: &mut HeadlessGpu,
) -> Result<Option<Bc7Fidelity>> {
    use super::d3d12::Submit;
    let ids: Vec<usize> =
        (0..scene.textures.len()).filter(|&i| bc7::should_compress(&scene.textures[i])).collect();
    if ids.is_empty() {
        return Ok(None);
    }
    let device = hg.device.clone();

    // The session's exact blocks, re-encoded (LPT largest-first, the upload
    // path's scheduling).
    let blocks_by_id: Vec<(usize, Vec<u8>)> = {
        use rayon::prelude::*;
        let mut order = ids.clone();
        order.sort_by_key(|&i| {
            std::cmp::Reverse(scene.textures[i].w as u64 * scene.textures[i].h as u64)
        });
        order.par_iter().map(|&i| (i, bc7::encode_opaque(&scene.textures[i], q))).collect()
    };

    // Root signature: [0] table of one SRV (t0), [1] root UAV (u0),
    // [2] two root constants (b0: W, H).
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
        .map_err(|e| format!("bc7 fidelity root sig serialize: {e}"))?;
    let blob = blob.unwrap();
    let root: ID3D12RootSignature = unsafe {
        device.CreateRootSignature(
            0,
            std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()),
        )
    }
    .map_err(|e| format!("bc7 fidelity root sig: {e}"))?;
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
    .map_err(|e| format!("bc7 fidelity PSO: {e}"))?;
    let heap: ID3D12DescriptorHeap = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            ..Default::default()
        })
    }
    .map_err(|e| format!("bc7 fidelity heap: {e}"))?;

    // One staging pair reused across textures (the blocking submits fence it).
    let max_stage = ids
        .iter()
        .map(|&i| {
            let t = &scene.textures[i];
            d3d12::block_pitch(t.w) * bc7::blocks(t.h) as usize
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
    for (i, enc) in &blocks_by_id {
        let t = &scene.textures[*i];
        let fmt = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC7_UNORM;
        let tex = d3d12::committed_tex(
            &device,
            t.w,
            t.h,
            fmt,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
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
        unsafe {
            device.CreateShaderResourceView(
                &tex,
                Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: fmt,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                    },
                }),
                heap.GetCPUDescriptorHandleForHeapStart(),
            )
        };
        hg.run_list(&mut |l| {
            unsafe {
                l.SetDescriptorHeaps(&[Some(heap.clone())]);
                l.SetComputeRootSignature(&root);
                l.SetPipelineState(&pso);
                l.SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
                l.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
                l.SetComputeRoot32BitConstants(2, 2, [t.w, t.h].as_ptr() as *const _, 0);
                l.Dispatch(t.w.div_ceil(8), t.h.div_ceil(8), 1);
            }
            Ok(())
        })?;
        let dec = hg.read_buffer(
            &out,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            (t.w * t.h) as usize * 4,
        )?;
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
        worst_psnr = worst_psnr.min(psnr);
    }
    Ok(Some(Bc7Fidelity {
        textures: blocks_by_id.len(),
        mean_abs: sum_abs / n_samples as f64,
        max_abs,
        worst_psnr,
    }))
}
