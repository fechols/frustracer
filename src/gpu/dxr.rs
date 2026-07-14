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

use super::d3d12::{self, committed_buffer, transition, uav_barrier, Result};
use super::dxc::Dxc;
use super::trace::{
    self, FrameCb, FrameParams, SceneGpu, CB_STRIDE, RP_FRAME_CBV, RP_GBUF, RP_PUSH,
    RP_SCENE_TEX, RP_SRV0, RP_TEX, RP_UAV0, SRV_INDICES, SRV_MATERIALS, SRV_NORMALS,
    SRV_POSITIONS, SRV_TLAS, SRV_TRI_MAT, TEX_HEAP_BASE, TEX_TABLE_BUFS, UAV_ACCUM, UAV_INFO,
    UAV_TBUF,
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

pub struct DxrGpu {
    root_sig: ID3D12RootSignature,
    state: ID3D12StateObject,
    sbt: d3d12::UploadBuffer,
    pso_resolve: ID3D12PipelineState,
    pub scene: SceneGpu,
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
    feed: Option<(trace::FeedKind, Vec<ID3D12Resource>)>,
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
        rw: u32,
        rh: u32,
        gbuf_full: bool,
        debug: bool,
        bc7_q: Option<crate::bc7::Quality>,
        submit: &mut dyn d3d12::Submit,
    ) -> Result<Self> {
        require_caps(device)?;
        let device5: ID3D12Device5 =
            device.cast().map_err(|e| format!("ID3D12Device5: {e}"))?;
        let root_sig = trace::create_root_signature(device)?;

        // Alpha-masked scenes compile the ah_* any-hit shaders + non-opaque
        // ray flags in (trace.rs::alpha_defs — the same per-scene predicate
        // that drops OPAQUE from the BLAS); opaque scenes compile verbatim.
        let any_alpha = scene.any_alpha;
        // The cbuffer's --spp jitter-table size, injected like alpha_defs.
        let sd = trace::spp_defs();
        let sd = sd.as_str();
        let lib_src = [
            trace::alpha_defs(scene),
            sd,
            trace::TRACE_COMMON_HLSLI,
            RT_DXR_HLSLI,
            trace::SHADE_HLSLI,
            DXR_HLSL,
        ]
        .join("\n");
        let dxil = dxc.compile(&lib_src, "", "lib_6_3", "dxr library", debug)?;
        let resolve_src = [sd, trace::TRACE_COMMON_HLSLI, trace::RESOLVE_HLSL].join("\n");
        let pso_resolve = trace::compute_pso(
            device,
            &root_sig,
            &dxc.compile(&resolve_src, "cs_resolve", "cs_6_3", "dxr resolve", debug)?,
            "dxr resolve",
        )?;
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
            if any_alpha { PCWSTR(name.as_ptr()) } else { PCWSTR::null() }
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
        // position and its probe bit); triangle barycentrics = 8 B.
        let shader_cfg = D3D12_RAYTRACING_SHADER_CONFIG {
            MaxPayloadSizeInBytes: 32,
            MaxAttributeSizeInBytes: 8,
        };
        // raygen -> chs_shade (1); its shadow/AO/reflection rays (2); chs_hit
        // and the misses fire nothing. The CPU's depth-1 recursion is the
        // flattened lap loop inside chs_shade, not payload recursion.
        let pipe_cfg = D3D12_RAYTRACING_PIPELINE_CONFIG { MaxTraceRecursionDepth: 2 };
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
        // HgOcclude imports ah_shadow, which only exports under ALPHA_CUTOUT
        // — the subobject exists exactly when the library exports it.
        if any_alpha {
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
        // it); the any-hit-only HgOcclude on alpha-masked scenes.
        if any_alpha {
            put(SBT_HIT + 2 * IDENT, ident("HgOcclude")?);
        }

        // SwAccel::None: the DXR pipeline never binds the software BVH (see
        // bind_common — t0/t1 stay unset), so its ~32 B/node upload is
        // skipped entirely (~2.3 GB at 100M tris).
        let scene_gpu =
            SceneGpu::new_uploaded(device, scene, crate::gpu::trace::SwAccel::None, submit, bc7_q)?;

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
        // Slot 0 = hdr (the resolve target), slots 1..7 = the feed planes
        // (wired later), slots TEX_HEAP_BASE.. = the RP_SCENE_TEX scene
        // table — the tracer's heap layout exactly.
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
        unsafe {
            device.CreateUnorderedAccessView(
                &hdr,
                None,
                None,
                uav_heap.GetCPUDescriptorHandleForHeapStart(),
            )
        };
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

        Ok(Self {
            root_sig,
            state,
            sbt,
            pso_resolve,
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
            feed: None,
            frame_cb,
            cb_base: FrameCb::base(scene, rw, rh),
            rw,
            rh,
        })
    }

    pub fn write_cb(&self, slot: usize, p: &FrameParams) {
        let fsr_sig = matches!(&self.feed, Some((trace::FeedKind::FsrRr, _)));
        self.cb_base
            .with_frame(p, self.gbuf_full, fsr_sig)
            .store(unsafe { self.frame_cb.ptr.add(slot * CB_STRIDE) });
    }

    /// Wire the upscaler feed targets (registers u16..u22) into this
    /// pipeline's descriptor heap — the DXR twin of TraceGpu::wire_feed,
    /// same heap layout, same typed-store gate.
    pub fn wire_feed(
        &mut self,
        device: &ID3D12Device,
        kind: trace::FeedKind,
        targets: &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
    ) -> Result<()> {
        self.feed = Some((kind, trace::wire_feed_targets(device, &self.uav_heap, targets)?));
        Ok(())
    }

    /// Fan the pack + accum out into the wired upscaler input planes. Record
    /// AFTER record_frame on the same list (its trailing global UAV barrier
    /// fences the pack/accum writes).
    pub fn record_feed(&self, list: &ID3D12GraphicsCommandList, slot: usize) -> Result<()> {
        let Some((kind, planes)) = &self.feed else {
            return Err("feed targets not wired".into());
        };
        let pso = match kind {
            // Fsr3 IS the XeSS feed (same planes, formats, depth encode).
            trace::FeedKind::Xess | trace::FeedKind::Fsr3 => self.pso_feed_xess.as_ref(),
            trace::FeedKind::Rr => self.pso_feed_rr.as_ref(),
            trace::FeedKind::FsrRr => self.pso_feed_fsr_rr.as_ref(),
        }
        .ok_or("feed PSO missing (DxrGpu built without gbuf)")?;
        trace::record_feed_dispatch(list, &self.uav_heap, pso, None, planes, self.rw, self.rh, &|| unsafe {
            self.bind_common(list, slot)
        });
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
