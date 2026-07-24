//! The by-the-book DXR pipeline (`--dxr`, the F key): an RTPSO state object,
//! a shader binding table, and DispatchRays with raygen / closest-hit / miss
//! shaders — hardware ray tracing dispatched the way the DXR spec draws it,
//! next to trace.rs's compute-wavefront + inline-RayQuery flavor. The CPU
//! renderer stays the reference. Shading parity is inherited, not re-ported:
//! the DXR library pastes the SAME trace_common.hlsli + shade.hlsli the
//! compute tracer runs, with rt_dxr.hlsli swapping the two trace primitives
//! from RayQuery to TraceRay — so the F toggle on a converged frame is an
//! intersector/dispatch A/B, not a shading A/B. Scene buffers, the BLAS/TLAS
//! (SceneGpu), the compute root signature (as the DXR GLOBAL root
//! signature — same registers), the FrameCb layout, and the resolve kernel
//! are all shared with trace.rs.
//!
//! SBT layout (rt_dxr.hlsli mirrors the indices — keep in lockstep):
//!   raygen @ 0    | miss @ 64: [radiance, shadow, hit_info]
//!   hit groups @ 192: [HgShade, HgHit, null (occlusion; the any-hit-only
//!   HgOcclude instead on alpha-masked scenes — see ALPHA_CUTOUT)]
//!
//! FR_DXR_INLINE (dxr_inline_mode below) is the W2 experiment lever: the
//! same pipeline with its rays moved onto inline RayQuery, one stage at a
//! time — the layout above is unchanged in every mode.

use super::d3d12::{self, committed_buffer, transition, uav_barrier, Result};
use super::dxc::Dxc;
use super::trace::{
    self, FrameCb, FrameParams, SceneGpu, CB_STRIDE, RP_FRAME_CBV, RP_GBUF, RP_PUSH,
    RP_SCENE_TEX, RP_SRV0, RP_TEX, RP_UAV0, SRV_INDICES, SRV_MATERIALS, SRV_NORMALS,
    SRV_POSITIONS, SRV_TLAS, SRV_TRI_MAT, TEX_HEAP_BASE, TEX_TABLE_BUFS, UAV_ACCUM, UAV_INFO,
    UAV_QIN, UAV_QOUT, UAV_TBUF,
};
use crate::scene::Scene;
use windows::core::{Interface, PCWSTR};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;

const RT_DXR_HLSLI: &str = include_str!("shaders/rt_dxr.hlsli");
const DXR_HLSL: &str = include_str!("shaders/dxr.hlsl");

const IDENT: usize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES as usize; // 32
/// Table starts are 64-aligned (D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
/// identifier-only records make the stride the 32-byte identifier itself.
const SBT_MISS: usize = 64;
const SBT_HIT: usize = 192;
const SBT_SIZE: usize = SBT_HIT + 3 * IDENT;

/// What DispatchRays needs, queried once. Tier 1.0 suffices (the compute
/// tracer's RayQuery needs 1.1; this pipeline predates it) and the library
/// compiles as lib_6_3. Missing caps are a clean "stay on the CPU" story.
pub fn require_caps(device: &ID3D12Device) -> Result<()> {
    let caps = trace::query_caps(device)?;
    let mut missing = Vec::new();
    if caps.rt_tier < D3D12_RAYTRACING_TIER_1_0.0 {
        missing.push(format!(
            "DXR raytracing tier 1.0 (DispatchRays) — device reports tier {}",
            caps.rt_tier
        ));
    }
    // 0x63 == D3D_SHADER_MODEL_6_3 (absent from the windows 0.62 bindings).
    if caps.shader_model < 0x63 {
        missing.push(format!("shader model 6.3 — device reports 0x{:x}", caps.shader_model));
    }
    if device.cast::<ID3D12Device5>().is_err() {
        missing.push("ID3D12Device5 (CreateStateObject/DispatchRays)".into());
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!("DXR pipeline unsupported here: {}", missing.join("; ")))
    }
}

/// `--dxr-inline` (default **1** — the W2 promotion): which of this
/// pipeline's rays ride recursive TraceRay vs inline RayQuery, without
/// leaving DispatchRays.
///   0 — all TraceRay: the original by-the-book pipeline, kept as the A/B
///       escape (bit-identical library to the pre-lever build).
///   1 — THE DEFAULT: primary TraceRay -> chs_shade, every secondary
///       shade.hlsli fires (shadow/AO/reflection/transmission/translucency)
///       an inline RayQuery inside the hit shader (rt.hlsli's bodies compile
///       in place of rt_dxr.hlsli's TraceRay flavors);
///       MaxTraceRecursionDepth 1. Promoted because it strictly DOMINATES
///       mode 0 at every measured point on both vendors — never slower,
///       −68 to −81% at the shipping spp=1 — while the payload/closest-hit/
///       SBT machinery keeps doing its real job for the primary.
///   2 — everything inline in raygen (dxr.hlsl's DXR_INLINE_SEC == 2 arm):
///       no TraceRay anywhere, DispatchRays as a bare launch grid over the
///       reference loop. The measurement arm that proved launch overhead is
///       ≈ zero — and the right MANUAL pick for a high-spp Intel DXR
///       session (mode 1's fat hit shader pays occupancy per sample: B70
///       marginal 2.2 ms/sample vs mode 2's 1.11).
/// Measured (--spin path 1080p spp=1, GPU frame span ms, default/stress/
/// SM-lp): B70 mode 0 9.05/5.30/6.75 -> mode 1 2.35/1.64/1.94 -> mode 2
/// 1.41/1.22/1.29; 4090 1.34/0.79/1.18 -> 0.26/0.25/0.34 -> 0.29/0.27/0.34.
/// Armed modes compile lib_6_5 and need RT tier 1.1 (the wavefront's own
/// floor); lesser hardware degrades to 0 with one loud line — the default is
/// a preference, never a requirement (NOT the --fsr4 shape). The RTPSO/SBT
/// layout is identical in every mode: unreached hit groups and misses stay
/// exported (identifier-only records, no cost). Set from main's parse via
/// `set_inline_mode` (the texture::set_aniso knob-before-anything idiom);
/// legal values 0..=2, main exits 2 on anything else.
static INLINE_MODE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

pub fn set_inline_mode(n: u32) {
    INLINE_MODE.store(n, std::sync::atomic::Ordering::Relaxed);
}

fn dxr_inline_mode() -> u32 {
    INLINE_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

pub struct DxrGpu {
    root_sig: ID3D12RootSignature,
    state: ID3D12StateObject,
    sbt: d3d12::UploadBuffer,
    pso_resolve: ID3D12PipelineState,
    /// The cloud-cache fill kernels + their buffers (u5/u6), armed per the
    /// snapshotted levers. `None` when the respective cache is off. bound before
    /// DispatchRays and left resident for the whole ray dispatch.
    pso_sky_lod: Option<ID3D12PipelineState>,
    pso_cloud_shadow: Option<ID3D12PipelineState>,
    cloud_lod: ID3D12Resource,
    cloud_shadow: ID3D12Resource,
    /// Snapshotted cloud-cache levers (see the assembly comment) + the scene
    /// AABB the slab-space shadow grid spans.
    sky_lod_k: u32,
    cloud_shadow_n: u32,
    scene_aabb: ([f32; 3], [f32; 3]),
    /// The shared scene core — the SAME Rc the wavefront tracer holds (cached
    /// in GpuContext), so a session running both pays the scene VRAM once.
    pub scene: std::rc::Rc<SceneGpu>,
    /// Per-pixel planes, CPU-layout parity (accum = 3 f32/px, tbuf = f32/px,
    /// info = u32/px) — the same readback-compare shape as the compute tracer.
    pub accum: ID3D12Resource,
    pub tbuf: ID3D12Resource,
    pub info: ID3D12Resource,
    /// The G-buffer pack at RP_GBUF: 64 B/px `GBufPx` when the session
    /// composes with an upscaler (`gbuf_full`), a 64-byte dummy otherwise —
    /// FLAG_GBUF is clear then, but root-descriptor UAVs have no bounds
    /// check, so the plain-mode dummy is memory safety, not an optimization
    /// (the trace.rs precedent).
    pub gbuf: ID3D12Resource,
    gbuf_full: bool,
    /// RGBA16F resolve target; rests in PIXEL_SHADER_RESOURCE between frames
    /// (the tonemap PS reads it via SRV_SLOT_DXR).
    pub hdr: ID3D12Resource,
    uav_heap: ID3D12DescriptorHeap,
    /// GPU handle of the RP_SCENE_TEX table (heap slot TEX_HEAP_BASE).
    tex_table: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// Upscaler feed kernels (compiled only when `gbuf_full`; every kind so
    /// --check-dxr can rewire between them like --check-gpu does) and the
    /// wired planes record_feed barriers over.
    pso_feed_xess: Option<ID3D12PipelineState>,
    pso_feed_rr: Option<ID3D12PipelineState>,
    pso_feed_fsr_rr: Option<ID3D12PipelineState>,
    /// One entry per wired engine; the index IS its descriptor set (see
    /// trace::FEED_SETS). Normally one — several under --quinlight.
    feed: Vec<(trace::FeedKind, Vec<ID3D12Resource>)>,
    /// For the per-set descriptor-table handles record_feed computes.
    device: ID3D12Device,
    frame_cb: d3d12::UploadBuffer,
    cb_base: FrameCb,
    pub rw: u32,
    pub rh: u32,
}

impl DxrGpu {
    pub fn new(
        device: &ID3D12Device,
        dxc: &Dxc,
        scene: &Scene,
        scene_gpu: std::rc::Rc<SceneGpu>,
        rw: u32,
        rh: u32,
        gbuf_full: bool,
        debug: bool,
    ) -> Result<Self> {
        require_caps(device)?;
        let device5: ID3D12Device5 =
            device.cast().map_err(|e| format!("ID3D12Device5: {e}"))?;
        let root_sig = trace::create_root_signature(device)?;

        // --dxr-inline (see dxr_inline_mode): armed modes compile RayQuery
        // into the library, which needs the wavefront's caps floor, not this
        // pipeline's — gate here so a tier-1.0 box degrades to the TraceRay
        // path with one loud line instead of a DXC error. The default (1)
        // stays QUIET; only a departure prints a lever line (the blas-split
        // precedent).
        let inline_mode = {
            let m = dxr_inline_mode();
            if m > 0 {
                let caps = trace::query_caps(device)?;
                if caps.rt_tier < D3D12_RAYTRACING_TIER_1_1.0 || caps.shader_model < 0x65 {
                    eprintln!(
                        "dxr: --dxr-inline {m} unavailable — inline RayQuery needs RT tier 1.1 \
                         + SM 6.5 (device: tier {}, SM 0x{:x}); running the all-TraceRay \
                         pipeline",
                        caps.rt_tier, caps.shader_model
                    );
                    0
                } else {
                    if m == 2 {
                        eprintln!(
                            "dxr: --dxr-inline 2 — everything inline in raygen (DispatchRays \
                             as a bare launch grid; the default is 1, inline secondaries)"
                        );
                    }
                    m
                }
            } else {
                eprintln!(
                    "dxr: --dxr-inline 0 — all-TraceRay dispatch (the pre-lever pipeline; \
                     the default is 1, inline RayQuery secondaries)"
                );
                0
            }
        };

        // Alpha-masked and height-carrying scenes compile the ah_* any-hit
        // shaders + non-opaque ray flags in (trace.rs::alpha_defs /
        // height_defs — the same per-scene predicates that drop OPAQUE from
        // the BLAS); scenes with neither compile verbatim.
        let non_opaque = scene.any_alpha
            || (scene.any_height && crate::bvh::height_armed())
            || scene.any_transmissive;
        // The cbuffer's --spp jitter-table size, injected like alpha_defs.
        let sd = trace::spp_defs();
        let sd = sd.as_str();
        let defs = format!(
            "{}\n{}\n{}\n{}",
            trace::alpha_defs(scene),
            trace::height_defs(scene),
            trace::trans_defs(scene),
            trace::blas_defs()
        );
        // The cloud shading caches, snapshotted at construction like TraceGpu:
        // the library is compiled against these, the buffers sized against them,
        // and record_frame's fills / write_cb's grid all read the fields, so a
        // mid-process A/B (the --check-dxr on/off gate flips the static between
        // two DxrGpu builds) can never desync a shader from its fill dispatch.
        let sky_lod_k = trace::sky_lod();
        let cloud_shadow_n = trace::cloud_shadow_n();
        // The two cache defines arm trace_common's cached cloud_sun_transmittance
        // (u6) + skylod.hlsli's sky_radiance_lod (u5) for EVERY shade path in the
        // library — parity inherited, not re-ported. u5/u6 are unbound in the DXR
        // root signature today (the wavefront's tile queues), so binding dedicated
        // buffers there needs no root-signature change.
        let cloud_defs = format!(
            "#define CLOUD_SHADOW_N {cloud_shadow_n}\n#define SKY_LOD {sky_lod_k}\n#define SKY_LOD_LOG {}",
            sky_lod_k.trailing_zeros()
        );
        // Mode 0 assembles EXACTLY the shipping sequence (the lever's
        // off-state is byte-identical source, not merely equivalent); armed
        // modes prepend the define and paste rt.hlsli's RayQuery primitives
        // ahead of rt_dxr.hlsli, whose TraceRay flavors + tlas/HitInfo
        // compile out under DXR_INLINE_SEC. The two cache defines ride in every
        // mode (the shipping sequence now carries them; mode 0 stays
        // byte-identical ACROSS inline modes, the lever's actual contract).
        let inline_def = format!("#define DXR_INLINE_SEC {inline_mode}");
        let mut parts = vec![defs.as_str(), sd, cloud_defs.as_str()];
        if inline_mode > 0 {
            parts.push(inline_def.as_str());
        }
        parts.push(trace::TRACE_COMMON_HLSLI);
        // skylod.hlsli after trace_common (needs sky_compose/sky_backdrop/rw); no
        // SKY_UNIT — this unit pastes no queues.hlsli, so u5 is free anyway.
        parts.push(trace::SKYLOD_HLSLI);
        if inline_mode > 0 {
            parts.push(trace::RT_HLSLI);
        }
        parts.extend([RT_DXR_HLSLI, trace::SHADE_HLSLI, DXR_HLSL]);
        let lib_src = parts.join("\n");
        let lib_target = if inline_mode > 0 { "lib_6_5" } else { "lib_6_3" };
        let dxil = dxc.compile(&lib_src, "", lib_target, "dxr library", debug)?;
        let resolve_src = [sd, trace::TRACE_COMMON_HLSLI, trace::RESOLVE_HLSL].join("\n");
        let pso_resolve = trace::compute_pso(
            device,
            &root_sig,
            &dxc.compile(&resolve_src, "cs_resolve", "cs_6_3", "dxr resolve", debug)?,
            "dxr resolve",
        )?;
        // The cloud-cache FILL kernels (cs_sky_lod / cs_cloud_shadow), from the
        // SHARED sky_unit_src so they cannot drift from the wavefront's — plain
        // compute (no rays), so cs_6_3 like the resolve/feed kernels. Compiled
        // only when the respective cache is armed; None otherwise.
        let (pso_sky_lod, pso_cloud_shadow) = if sky_lod_k > 1 || cloud_shadow_n > 0 {
            let sky_unit = trace::sky_unit_src(sky_lod_k, cloud_shadow_n);
            let mk = |entry: &str, name: &str| -> Result<Option<ID3D12PipelineState>> {
                Ok(Some(trace::compute_pso(
                    device,
                    &root_sig,
                    &dxc.compile(&sky_unit, entry, "cs_6_3", name, debug)?,
                    name,
                )?))
            };
            (
                if sky_lod_k > 1 { mk("cs_sky_lod", "dxr sky_lod")? } else { None },
                if cloud_shadow_n > 0 { mk("cs_cloud_shadow", "dxr cloud_shadow")? } else { None },
            )
        } else {
            (None, None)
        };
        // Upscaler sessions: the same feed kernels the wavefront runs, at
        // this pipeline's cs_6_3 cap floor (feed.hlsl needs nothing newer).
        let (pso_feed_xess, pso_feed_rr, pso_feed_fsr_rr) = if gbuf_full {
            let feed_src =
                [sd, trace::TRACE_COMMON_HLSLI, trace::FSR_WIRE_HLSLI, trace::FEED_HLSL].join("\n");
            let pso = |entry: &str, name: &str| -> Result<ID3D12PipelineState> {
                trace::compute_pso(
                    device,
                    &root_sig,
                    &dxc.compile(&feed_src, entry, "cs_6_3", name, debug)?,
                    name,
                )
            };
            (
                Some(pso("cs_feed_xess", "dxr feed_xess")?),
                Some(pso("cs_feed_rr", "dxr feed_rr")?),
                Some(pso("cs_feed_fsr_rr", "dxr feed_fsr_rr")?),
            )
        } else {
            (None, None, None)
        };

        // --- RTPSO. Every pDesc payload (and every name string) lives in a
        // local that outlives CreateStateObject.
        let wname = |s: &str| -> Vec<u16> { s.encode_utf16().chain(std::iter::once(0)).collect() };
        let hg_shade_name = wname("HgShade");
        let hg_hit_name = wname("HgHit");
        let hg_occlude_name = wname("HgOcclude");
        let chs_shade_name = wname("chs_shade");
        let chs_hit_name = wname("chs_hit");
        let ah_shade_name = wname("ah_shade");
        let ah_hit_name = wname("ah_hit");
        let ah_shadow_name = wname("ah_shadow");

        let lib_desc = D3D12_DXIL_LIBRARY_DESC {
            DXILLibrary: D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr() as *const _,
                BytecodeLength: dxil.len(),
            },
            // No export list: every [shader("...")] entry exports.
            NumExports: 0,
            pExports: std::ptr::null_mut(),
        };
        // ALPHA_CUTOUT scenes attach the cutout any-hit to every hit group;
        // HgOcclude carries ONLY an any-hit (legal — SKIP_CLOSEST_HIT skips
        // just that stage, any-hit still runs during traversal: the standard
        // alpha-tested-shadow pattern, and the untouched-payload = occluded
        // convention holds: all-rejected => miss_shadow writes 0).
        let ahs = |name: &Vec<u16>| {
            if non_opaque { PCWSTR(name.as_ptr()) } else { PCWSTR::null() }
        };
        let hit_group = |export: &Vec<u16>, chs: PCWSTR, ah: PCWSTR| D3D12_HIT_GROUP_DESC {
            HitGroupExport: PCWSTR(export.as_ptr()),
            Type: D3D12_HIT_GROUP_TYPE_TRIANGLES,
            AnyHitShaderImport: ah,
            ClosestHitShaderImport: chs,
            IntersectionShaderImport: PCWSTR::null(),
        };
        let hg_shade = hit_group(
            &hg_shade_name,
            PCWSTR(chs_shade_name.as_ptr()),
            ahs(&ah_shade_name),
        );
        let hg_hit =
            hit_group(&hg_hit_name, PCWSTR(chs_hit_name.as_ptr()), ahs(&ah_hit_name));
        let hg_occlude = hit_group(
            &hg_occlude_name,
            PCWSTR::null(),
            PCWSTR(ah_shadow_name.as_ptr()),
        );
        // RayPayload {float3 + float + uint + float2 + uint} = 32 B is the
        // largest payload (the float2/uint tail is --spp: the sample's own
        // position, and prim = `(sample << 1) | probe_bit` — the index rides
        // the high bits so the miss shader can key the per-sample cloud march
        // phase without growing the payload); triangle barycentrics = 8 B.
        let shader_cfg = D3D12_RAYTRACING_SHADER_CONFIG {
            MaxPayloadSizeInBytes: 32,
            MaxAttributeSizeInBytes: 8,
        };
        // raygen -> chs_shade (1); its shadow/AO/reflection rays (2); chs_hit
        // and the misses fire nothing. The CPU's depth-1 recursion is the
        // flattened lap loop inside chs_shade, not payload recursion. Under
        // FR_DXR_INLINE the secondaries are inline RayQuery, so the deepest
        // TraceRay is raygen's primary (mode 1) or none at all (mode 2 —
        // depth 1 stays declared: 0 is legal but is a separate micro-variant,
        // not worth a DispatchRays-validation seam until mode 2 shows a win).
        let pipe_cfg = D3D12_RAYTRACING_PIPELINE_CONFIG {
            MaxTraceRecursionDepth: if inline_mode > 0 { 1 } else { 2 },
        };
        let grs = D3D12_GLOBAL_ROOT_SIGNATURE {
            pGlobalRootSignature: unsafe { std::mem::transmute_copy(&root_sig) },
        };
        let sub = |t: D3D12_STATE_SUBOBJECT_TYPE, p: *const std::ffi::c_void| {
            D3D12_STATE_SUBOBJECT { Type: t, pDesc: p }
        };
        let mut subobjects = vec![
            sub(D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &lib_desc as *const _ as *const _),
            sub(D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hg_shade as *const _ as *const _),
            sub(D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hg_hit as *const _ as *const _),
            sub(
                D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG,
                &shader_cfg as *const _ as *const _,
            ),
            sub(
                D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG,
                &pipe_cfg as *const _ as *const _,
            ),
            sub(D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &grs as *const _ as *const _),
        ];
        // HgOcclude imports ah_shadow, which only exports under
        // ALPHA_CUTOUT/HEIGHTFIELD — the subobject exists exactly when the
        // library exports it.
        if non_opaque {
            subobjects.push(sub(
                D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP,
                &hg_occlude as *const _ as *const _,
            ));
        }
        let so_desc = D3D12_STATE_OBJECT_DESC {
            Type: D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
            NumSubobjects: subobjects.len() as u32,
            pSubobjects: subobjects.as_ptr(),
        };
        let state: ID3D12StateObject = unsafe { device5.CreateStateObject(&so_desc) }
            .map_err(|e| format!("CreateStateObject(DXR pipeline): {e}"))?;

        // --- SBT: bare 32-byte identifiers (the global root signature
        // carries every binding, so records need no local root arguments).
        let props: ID3D12StateObjectProperties =
            state.cast().map_err(|e| format!("ID3D12StateObjectProperties: {e}"))?;
        let ident = |name: &str| -> Result<[u8; IDENT]> {
            let wn = wname(name);
            let p = unsafe { props.GetShaderIdentifier(PCWSTR(wn.as_ptr())) };
            if p.is_null() {
                return Err(format!("GetShaderIdentifier({name}): not found in the RTPSO"));
            }
            Ok(unsafe { *(p as *const [u8; IDENT]) })
        };
        let sbt = d3d12::UploadBuffer::new(device, SBT_SIZE)?;
        unsafe { std::ptr::write_bytes(sbt.ptr, 0, SBT_SIZE) };
        let put = |off: usize, id: [u8; IDENT]| unsafe {
            std::ptr::copy_nonoverlapping(id.as_ptr(), sbt.ptr.add(off), IDENT);
        };
        put(0, ident("raygen")?);
        put(SBT_MISS, ident("miss_radiance")?);
        put(SBT_MISS + IDENT, ident("miss_shadow")?);
        put(SBT_MISS + 2 * IDENT, ident("miss_hit")?);
        put(SBT_HIT, ident("HgShade")?);
        put(SBT_HIT + IDENT, ident("HgHit")?);
        // Hit group 2 (occlusion rays): the zeroed null record on opaque
        // scenes (SKIP_CLOSEST_HIT + FORCE_OPAQUE never run a shader from
        // it); the any-hit-only HgOcclude on alpha-masked/height scenes.
        if non_opaque {
            put(SBT_HIT + 2 * IDENT, ident("HgOcclude")?);
        }

        // The shared core arrived pre-uploaded (Rc from GpuContext's cache).
        // The DXR pipeline never binds the software BVH (see bind_common —
        // t0/t1 stay unset), and those trees now live OUTSIDE the core
        // (trace::SwTreesGpu, per-TraceGpu), so a DXR session structurally
        // never pays their upload (~2.3 GB at 100M tris).
        let uaf = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let px = rw as u64 * rh as u64;
        let accum = committed_buffer(device, px * 12, uaf, ua)?;
        let tbuf = committed_buffer(device, px * 4, uaf, ua)?;
        let info = committed_buffer(device, px * 4, uaf, ua)?;
        let gbuf = committed_buffer(
            device,
            if gbuf_full { px * trace::GBUF_STRIDE } else { trace::GBUF_STRIDE },
            uaf,
            ua,
        )?;
        let hdr = d3d12::committed_tex(
            device,
            rw,
            rh,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            uaf,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        // The cloud caches (sized exactly as TraceGpu's). One float4 per lattice
        // point + a one-point border; N*N scalar transmittances at the cap.
        let lw = (rw >> sky_lod_k.trailing_zeros()) as u64 + 2;
        let lh = (rh >> sky_lod_k.trailing_zeros()) as u64 + 2;
        let cloud_lod = committed_buffer(device, (lw * lh).max(1) * 16, uaf, ua)?;
        let csn_n =
            if cloud_shadow_n > 0 { crate::clouds::CLOUD_SHADOW_MAX as u64 } else { 1 };
        let cloud_shadow = committed_buffer(device, csn_n * csn_n * 4, uaf, ua)?;
        let scene_aabb = trace::scene_shadow_aabb(scene);
        // FEED_SETS copies of the RP_TEX table (hdr resolve target at each set's
        // offset 0, then that set's feed planes — wired later), then slots
        // TEX_HEAP_BASE.. = the RP_SCENE_TEX scene table — the tracer's heap
        // layout exactly.
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
        .map_err(|e| format!("CreateDescriptorHeap(dxr UAV): {e}"))?;
        trace::write_resolve_uavs(device, &uav_heap, &hdr);
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
        let frame_cb = d3d12::UploadBuffer::new(device, CB_STRIDE * d3d12::FRAMES_IN_FLIGHT)?;

        let name = |res: &ID3D12Resource, n: &str| {
            let w: Vec<u16> = n.encode_utf16().chain(std::iter::once(0)).collect();
            let _ = unsafe { res.SetName(PCWSTR(w.as_ptr())) };
        };
        name(&accum, "dxr.accum");
        name(&tbuf, "dxr.tbuf");
        name(&info, "dxr.info");
        name(&hdr, "dxr.hdr");
        name(&gbuf, if gbuf_full { "dxr.gbuf" } else { "dxr.gbuf_dummy" });

        name(&cloud_lod, "dxr.cloud_lod");
        name(&cloud_shadow, "dxr.cloud_shadow");

        Ok(Self {
            root_sig,
            state,
            sbt,
            pso_resolve,
            pso_sky_lod,
            pso_cloud_shadow,
            cloud_lod,
            cloud_shadow,
            sky_lod_k,
            cloud_shadow_n,
            scene_aabb,
            scene: scene_gpu,
            accum,
            tbuf,
            info,
            gbuf,
            gbuf_full,
            hdr,
            uav_heap,
            tex_table,
            pso_feed_xess,
            pso_feed_rr,
            pso_feed_fsr_rr,
            feed: Vec::new(),
            device: device.clone(),
            frame_cb,
            cb_base: FrameCb::base(scene, rw, rh),
            rw,
            rh,
        })
    }

    pub fn write_cb(&self, slot: usize, p: &FrameParams) {
        // One FSR4-RR subscriber among the wired engines is enough to arm the
        // pack's signal lanes (--quinlight can wire several).
        let fsr_sig = self.feed.iter().any(|(k, _)| matches!(k, trace::FeedKind::FsrRr));
        let mut cb = self.cb_base.with_frame(p, self.gbuf_full, fsr_sig);
        // The per-frame slab-space shadow grid (the shared shadow_grid_row —
        // one source of truth with TraceGpu); zero when the cache is off or
        // clouds are disabled (cloud_sun_transmittance then takes its exact arm).
        if self.cloud_shadow_n > 0 && p.clouds.enabled {
            cb.cloud_grid = crate::clouds::shadow_grid_row(
                self.cb_base.sun,
                self.scene_aabb,
                p.clouds.diag,
                self.cloud_shadow_n,
            );
        }
        cb.store(unsafe { self.frame_cb.ptr.add(slot * CB_STRIDE) });
    }

    /// Re-derive the base CB's sun/sky rows after a TOD change —
    /// `TraceGpu::refresh_sky`'s twin (`FrameCb::refresh_sky_rows`).
    pub fn refresh_sky(&mut self, scene: &Scene) {
        self.cb_base.refresh_sky_rows(scene, self.rw, self.rh);
    }

    /// The DXR twin of TraceGpu::wire_feed — same heap layout, same typed-store
    /// gate, same semantics: REPLACES the wiring with this one engine (so
    /// --check-dxr can rewire from one feed kind to the next).
    pub fn wire_feed(
        &mut self,
        device: &ID3D12Device,
        kind: trace::FeedKind,
        targets: &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
    ) -> Result<()> {
        self.feed.clear();
        self.wire_feed_add(device, kind, targets)
    }

    /// APPENDS one engine, claiming the next descriptor set (--quinlight).
    pub fn wire_feed_add(
        &mut self,
        device: &ID3D12Device,
        kind: trace::FeedKind,
        targets: &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
    ) -> Result<()> {
        let set = self.feed.len() as u32;
        let planes = trace::wire_feed_targets(device, &self.uav_heap, set, targets)?;
        self.feed.push((kind, planes));
        Ok(())
    }

    /// Fan the pack + accum out into the wired upscaler input planes — one
    /// dispatch per wired engine. Record AFTER record_frame on the same list
    /// (its trailing global UAV barrier fences the pack/accum writes).
    pub fn record_feed(&self, list: &ID3D12GraphicsCommandList, slot: usize) -> Result<()> {
        if self.feed.is_empty() {
            return Err("feed targets not wired".into());
        }
        let mut feeds: Vec<(&ID3D12PipelineState, u32, &[ID3D12Resource])> = Vec::new();
        for (set, (kind, planes)) in self.feed.iter().enumerate() {
            let pso = trace::feed_pso(
                *kind,
                None,
                self.pso_feed_xess.as_ref(),
                self.pso_feed_rr.as_ref(),
                self.pso_feed_fsr_rr.as_ref(),
            )
            .ok_or("feed PSO missing (DxrGpu built without gbuf)")?;
            feeds.push((pso, set as u32, planes.as_slice()));
        }
        trace::record_feed_dispatch(
            list,
            &self.device,
            &self.uav_heap,
            &feeds,
            None,
            self.rw,
            self.rh,
            &|| unsafe { self.bind_common(list, slot) },
        );
        Ok(())
    }

    /// The DXR subset of the shared root layout. t0/t1 (software BVH) and the
    /// wavefront queue UAVs stay unbound — no shader in this library touches
    /// them, and unaccessed root descriptors are legal to leave unset.
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
            list.SetComputeRootUnorderedAccessView(RP_GBUF, self.gbuf.GetGPUVirtualAddress());
            let s = &self.scene;
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
            // The scene-texture table (t0..t3 + texs[] in space1) — heap
            // before table, same as the tracer's bind_common.
            list.SetDescriptorHeaps(&[Some(self.uav_heap.clone())]);
            list.SetComputeRootDescriptorTable(RP_SCENE_TEX, self.tex_table);
        }
    }

    /// One DispatchRays over the full target. Ends with a global UAV barrier
    /// so the resolve (or a readback) sees the splats.
    pub fn record_frame(&self, list: &ID3D12GraphicsCommandList, slot: usize) -> Result<()> {
        let list4: ID3D12GraphicsCommandList4 =
            list.cast().map_err(|e| format!("ID3D12GraphicsCommandList4: {e}"))?;
        let _ev = super::pix::scope(list, c"dxr");
        unsafe {
            self.bind_common(list, slot);
            // The cloud caches, ahead of the ray dispatch and left bound at
            // u6/u5 for the whole DispatchRays. record_frame is the ONE dispatch
            // site (session chains AND --check-dxr route through it), so the
            // "every compiling path must fill or the shader reads garbage as
            // float4 = device hang" contract (trace.rs::record_cloud_shadow's
            // war story) holds structurally. The root-descriptor bindings
            // persist on the list, so the DispatchRays below reads them.
            if let Some(pso) = &self.pso_cloud_shadow {
                let _e = super::pix::scope(list, c"dxr-cloud-shadow");
                list.SetComputeRootUnorderedAccessView(
                    RP_UAV0 + UAV_QOUT,
                    self.cloud_shadow.GetGPUVirtualAddress(),
                );
                let groups = (crate::clouds::CLOUD_SHADOW_MAX * crate::clouds::CLOUD_SHADOW_MAX)
                    .div_ceil(64);
                list.SetPipelineState(pso);
                list.Dispatch(groups.min(32768), groups.div_ceil(32768), 1);
                list.ResourceBarrier(&[uav_barrier(None)]);
            }
            if let Some(pso) = &self.pso_sky_lod {
                let _e = super::pix::scope(list, c"dxr-sky-lod");
                list.SetComputeRootUnorderedAccessView(
                    RP_UAV0 + UAV_QIN,
                    self.cloud_lod.GetGPUVirtualAddress(),
                );
                let k = self.sky_lod_k;
                let pts = ((self.rw / k) + 2) * ((self.rh / k) + 2);
                let groups = pts.div_ceil(64);
                list.SetPipelineState(pso);
                list.Dispatch(groups.min(32768), groups.div_ceil(32768), 1);
                list.ResourceBarrier(&[uav_barrier(None)]);
            }
            list4.SetPipelineState1(&self.state);
            let va = self.sbt.resource.GetGPUVirtualAddress();
            let desc = D3D12_DISPATCH_RAYS_DESC {
                RayGenerationShaderRecord: D3D12_GPU_VIRTUAL_ADDRESS_RANGE {
                    StartAddress: va,
                    SizeInBytes: IDENT as u64,
                },
                MissShaderTable: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
                    StartAddress: va + SBT_MISS as u64,
                    SizeInBytes: (3 * IDENT) as u64,
                    StrideInBytes: IDENT as u64,
                },
                HitGroupTable: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
                    StartAddress: va + SBT_HIT as u64,
                    SizeInBytes: (3 * IDENT) as u64,
                    StrideInBytes: IDENT as u64,
                },
                CallableShaderTable: Default::default(),
                Width: self.rw,
                Height: self.rh,
                Depth: 1,
            };
            list4.DispatchRays(&desc);
            list.ResourceBarrier(&[uav_barrier(None)]);
        }
        Ok(())
    }

    /// accum -> hdr at 1/samples (trace.rs's resolve kernel + curve); hdr
    /// ends in PIXEL_SHADER_RESOURCE for the tonemap blit.
    pub fn record_resolve(&self, list: &ID3D12GraphicsCommandList, slot: usize, samples: u32) {
        let _ev = super::pix::scope(list, c"dxr-resolve");
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
