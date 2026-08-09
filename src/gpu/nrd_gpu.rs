//! NrdGpu — the D3D12 host for NRD's own compute passes (`--nrd`). NRD is a
//! CPU-side dispatch scheduler with embedded DXIL (src/nrd.rs); everything
//! GPU-facing lives HERE: the texture pools `GetInstanceDesc` asks the app to
//! allocate, one SHARED root signature (NRD's recommended layout: a root CBV
//! + static samplers + one SRV/UAV table sized to the per-set maxima), one
//! PSO per `PipelineDesc` from the DLL-served DXIL blobs, a shader-visible
//! descriptor ring + a persistently-mapped CB ring (~31 dispatches/frame,
//! 864 B CB — measured by --check-nrd N1), and a per-resource state tracker
//! (every texture RESTS in UNORDERED_ACCESS between frames, the NppdRes
//! doctrine; TEXTURE bindings transition to NON_PIXEL_SHADER_RESOURCE for the
//! dispatch that reads them).
//!
//! The bloom private-heap pattern at the DXC tier: NRD's passes bind their
//! OWN heap/root signature, and every downstream pass (record_feed, the
//! upscaler evals, tonemap) rebinds from scratch, so nothing needs restoring.
//! `record()` snapshots the DispatchDescs into a Vec first — the slice
//! GetComputeDispatches returns is owned by the instance and borrows it, and
//! the recording loop needs the rest of `self`.

#![allow(dead_code)]

use super::d3d12;
use crate::nrd;
use windows::core::s;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

type Result<T> = std::result::Result<T, String>;

/// nrd::Format -> DXGI (the NRDDescs.h enum order — transcription-pinned).
fn dxgi_format(f: u32) -> Result<DXGI_FORMAT> {
    Ok(match f {
        0 => DXGI_FORMAT_R8_UNORM,
        1 => DXGI_FORMAT_R8_SNORM,
        2 => DXGI_FORMAT_R8_UINT,
        3 => DXGI_FORMAT_R8_SINT,
        4 => DXGI_FORMAT_R8G8_UNORM,
        5 => DXGI_FORMAT_R8G8_SNORM,
        6 => DXGI_FORMAT_R8G8_UINT,
        7 => DXGI_FORMAT_R8G8_SINT,
        8 => DXGI_FORMAT_R8G8B8A8_UNORM,
        9 => DXGI_FORMAT_R8G8B8A8_SNORM,
        10 => DXGI_FORMAT_R8G8B8A8_UINT,
        11 => DXGI_FORMAT_R8G8B8A8_SINT,
        12 => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        13 => DXGI_FORMAT_R16_UNORM,
        14 => DXGI_FORMAT_R16_SNORM,
        15 => DXGI_FORMAT_R16_UINT,
        16 => DXGI_FORMAT_R16_SINT,
        17 => DXGI_FORMAT_R16_FLOAT,
        18 => DXGI_FORMAT_R16G16_UNORM,
        19 => DXGI_FORMAT_R16G16_SNORM,
        20 => DXGI_FORMAT_R16G16_UINT,
        21 => DXGI_FORMAT_R16G16_SINT,
        22 => DXGI_FORMAT_R16G16_FLOAT,
        23 => DXGI_FORMAT_R16G16B16A16_UNORM,
        24 => DXGI_FORMAT_R16G16B16A16_SNORM,
        25 => DXGI_FORMAT_R16G16B16A16_UINT,
        26 => DXGI_FORMAT_R16G16B16A16_SINT,
        27 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        28 => DXGI_FORMAT_R32_UINT,
        29 => DXGI_FORMAT_R32_SINT,
        30 => DXGI_FORMAT_R32_FLOAT,
        31 => DXGI_FORMAT_R32G32_UINT,
        32 => DXGI_FORMAT_R32G32_SINT,
        33 => DXGI_FORMAT_R32G32_FLOAT,
        34 => DXGI_FORMAT_R32G32B32_UINT,
        35 => DXGI_FORMAT_R32G32B32_SINT,
        36 => DXGI_FORMAT_R32G32B32_FLOAT,
        37 => DXGI_FORMAT_R32G32B32A32_UINT,
        38 => DXGI_FORMAT_R32G32B32A32_SINT,
        39 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        40 => DXGI_FORMAT_R10G10B10A2_UNORM,
        41 => DXGI_FORMAT_R10G10B10A2_UINT,
        42 => DXGI_FORMAT_R11G11B10_FLOAT,
        43 => DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
        _ => return Err(format!("nrd: unknown Format {f}")),
    })
}

/// One registered resource: the texture, its staging SRV/UAV slots, and its
/// tracked state (rests in UNORDERED_ACCESS).
struct Reg {
    res: ID3D12Resource,
    state: std::cell::Cell<D3D12_RESOURCE_STATES>,
}

const CB_ALIGN: usize = 256;
const RING_FRAMES: usize = 3; // >= d3d12 frames in flight
const MAX_DISPATCHES: usize = 48; // N1 measured 31; headroom, asserted

pub struct NrdGpu {
    pub nrd: nrd::Nrd,
    root_sig: ID3D12RootSignature,
    psos: Vec<ID3D12PipelineState>,
    /// Registered resources: [permanent pool.. , transient pool.., app planes]
    regs: Vec<Reg>,
    perm_base: usize,
    trans_base: usize,
    /// App-plane indices into `regs` (IN_MV, IN_NR, IN_VIEWZ, IN_DIFF,
    /// IN_SPEC, OUT_DIFF, OUT_SPEC, OUT_VALIDATION).
    plane_idx: [usize; 8],
    /// Non-shader-visible staging heap: per reg, slot 2i = SRV, 2i+1 = UAV.
    staging: ID3D12DescriptorHeap,
    /// Shader-visible ring: RING_FRAMES * MAX_DISPATCHES * set_stride.
    ring: ID3D12DescriptorHeap,
    set_stride: u32, // perSetTexturesMaxNum + perSetStorageTexturesMaxNum
    max_tex: u32,
    /// Persistently-mapped CB upload ring.
    cb: d3d12::UploadBuffer,
    cb_slot_size: usize,
    inc: u32,
    pub rw: u32,
    pub rh: u32,
    frame_counter: std::cell::Cell<u64>,
}

/// The frame's CommonSettings from the tree's own camera facts.
/// `dlss::CamMatrices` is glam — COLUMN-major with COLUMN vectors, which is
/// NRD's stated convention (NRDSettings.h), so `to_cols_array()` IS the wire
/// format: no transpose anywhere (the deliberate contrast with
/// `gpu::row_major`'s SL boundary — verify with FR_NRD_DEBUG's validation
/// overlay if reprojection ever looks dead). MV scale: the pack stores
/// pixel-space prev−cur (+ the 2.5D view-Z delta), NRD wants
/// `pixelUvPrev = pixelUv + mv.xy` in UV units — {1/rw, 1/rh, 1}.
/// denoisingRange: sky stores view_z == far, so anything at/past far*0.999
/// is out of range and NRD leaves those texels alone. cs_nrd_out's
/// pass-accum-through predicate is `view_z >= 0.999 * CAM_FAR` — the SAME
/// bound, LOCKSTEP (a shader predicate of plain CAM_FAR shipped once and
/// recomposed the [0.999·far, far) hit band from OUT texels NRD never wrote).
pub fn common_settings(
    mats: &crate::dlss::CamMatrices,
    prev_mats: &crate::dlss::CamMatrices,
    jitter: (f32, f32),
    prev_jitter: (f32, f32),
    rw: u32,
    rh: u32,
    far: f32,
    frame_index: u32,
    reset: bool,
) -> nrd::CommonSettings {
    let mut cs = nrd::CommonSettings::default();
    cs.view_to_clip_matrix = mats.view_to_clip.to_cols_array();
    cs.view_to_clip_matrix_prev = prev_mats.view_to_clip.to_cols_array();
    cs.world_to_view_matrix = mats.world_to_view.to_cols_array();
    cs.world_to_view_matrix_prev = prev_mats.world_to_view.to_cols_array();
    cs.motion_vector_scale = [1.0 / rw as f32, 1.0 / rh as f32, 1.0];
    cs.camera_jitter = [jitter.0, jitter.1];
    cs.camera_jitter_prev = [prev_jitter.0, prev_jitter.1];
    cs.resource_size = [rw as u16, rh as u16];
    cs.resource_size_prev = [rw as u16, rh as u16];
    cs.rect_size = [rw as u16, rh as u16];
    cs.rect_size_prev = [rw as u16, rh as u16];
    cs.denoising_range = far * 0.999;
    cs.frame_index = frame_index;
    cs.accumulation_mode = if reset { nrd::ACCUM_RESTART } else { nrd::ACCUM_CONTINUE };
    if std::env::var_os("FR_NRD_DEBUG").is_some() {
        cs.enable_validation = 1;
    }
    cs
}

/// The session's ReblurSettings: defaults + the ONE departure the 1-spp path
/// demands (the hit-dist params stay at their {3, 0.1, 20} defaults, which is
/// what keeps them in LOCKSTEP with nrd_bridge.hlsl's literals), then the
/// `--nrd-*` tuning overrides (all-None flagless — bit-identical settings).
pub fn reblur_settings() -> nrd::ReblurSettings {
    let mut rs = nrd::ReblurSettings::default();
    // Pixels whose reflection gate never fired carry hit-dist 0 ("no data")
    // — reconstruction fills them from neighbors (required below ~1 rpp).
    rs.hit_distance_reconstruction_mode = nrd::HITDIST_RECONSTRUCTION_AREA_3X3;
    nrd::tuning().apply(&mut rs);
    rs
}

impl NrdGpu {
    pub fn new(device: &ID3D12Device, dll_dir: &str, rw: u32, rh: u32) -> Result<Self> {
        let denoisers = [nrd::DenoiserDesc {
            identifier: 0,
            denoiser: nrd::DENOISER_REBLUR_DIFFUSE_SPECULAR,
        }];
        let inst = nrd::Nrd::new(dll_dir, &denoisers)?;
        let d = inst.instance_desc();

        // --- The shared root signature NRD recommends: root CBV + the two
        // static samplers + one table [SRV range | UAV range] at the per-set
        // maxima. Per-dispatch tables bind a ring window laid out the same way.
        let max_tex = d.descriptor_pool_desc.per_set_textures_max_num.max(1);
        let max_uav = d.descriptor_pool_desc.per_set_storage_textures_max_num.max(1);
        let cb_space = d.constant_buffer_and_samplers_space_index;
        let res_space = d.resources_space_index;
        let ranges = [
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: max_tex,
                BaseShaderRegister: d.resources_base_register_index,
                RegisterSpace: res_space,
                OffsetInDescriptorsFromTableStart: 0,
            },
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                NumDescriptors: max_uav,
                BaseShaderRegister: d.resources_base_register_index,
                RegisterSpace: res_space,
                OffsetInDescriptorsFromTableStart: max_tex,
            },
        ];
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: d.constant_buffer_register_index,
                        RegisterSpace: cb_space,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
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
        ];
        // InstanceDesc.samplers is [NEAREST_CLAMP, LINEAR_CLAMP] at
        // samplersBaseRegisterIndex — verified by --check-nrd's samplers==2.
        let samplers = unsafe { std::slice::from_raw_parts(d.samplers, d.samplers_num as usize) };
        let mut static_samp = Vec::new();
        for (i, &s) in samplers.iter().enumerate() {
            static_samp.push(D3D12_STATIC_SAMPLER_DESC {
                Filter: if s == nrd::SAMPLER_LINEAR_CLAMP {
                    D3D12_FILTER_MIN_MAG_MIP_LINEAR
                } else {
                    D3D12_FILTER_MIN_MAG_MIP_POINT
                },
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MaxLOD: f32::MAX,
                ShaderRegister: d.samplers_base_register_index + i as u32,
                RegisterSpace: cb_space,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                ..Default::default()
            });
        }
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: static_samp.len() as u32,
            pStaticSamplers: static_samp.as_ptr(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut errb: Option<ID3DBlob> = None;
        unsafe {
            windows::Win32::Graphics::Direct3D12::D3D12SerializeRootSignature(
                &rs_desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                Some(&mut errb),
            )
        }
        .map_err(|e| format!("nrd: serialize root sig: {e}"))?;
        let blob = blob.unwrap();
        let root_sig: ID3D12RootSignature = unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()),
            )
        }
        .map_err(|e| format!("nrd: create root sig: {e}"))?;

        // --- PSOs from the embedded DXIL blobs.
        let pipes = unsafe { std::slice::from_raw_parts(d.pipelines, d.pipelines_num as usize) };
        let mut psos = Vec::with_capacity(pipes.len());
        for (i, p) in pipes.iter().enumerate() {
            let cs = &p.compute_shader_dxil;
            if cs.bytecode.is_null() || cs.size == 0 {
                return Err(format!("nrd: pipeline {i} has no DXIL (rebuild the DLL)"));
            }
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: unsafe { std::mem::transmute_copy(&root_sig) },
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: cs.bytecode,
                    BytecodeLength: cs.size as usize,
                },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = unsafe { device.CreateComputePipelineState(&desc) }
                .map_err(|e| format!("nrd: pipeline {i} PSO: {e}"))?;
            psos.push(pso);
        }

        // --- Texture pools + our IN/OUT planes, all resting UNORDERED_ACCESS.
        let uaf = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let ust = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let mut regs: Vec<Reg> = Vec::new();
        fn push_tex(
            device: &ID3D12Device,
            regs: &mut Vec<Reg>,
            fmt: DXGI_FORMAT,
            w: u32,
            h: u32,
        ) -> Result<usize> {
            let uaf = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            let ust = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            let res = d3d12::committed_tex(device, w.max(1), h.max(1), fmt, uaf, ust)?;
            regs.push(Reg { res, state: std::cell::Cell::new(ust) });
            Ok(regs.len() - 1)
        }
        let _ = (uaf, ust);
        let perm = unsafe { std::slice::from_raw_parts(d.permanent_pool, d.permanent_pool_size as usize) };
        let trans = unsafe { std::slice::from_raw_parts(d.transient_pool, d.transient_pool_size as usize) };
        let perm_base = 0usize;
        for t in perm {
            let df = t.downsample_factor.max(1) as u32;
            push_tex(device, &mut regs, dxgi_format(t.format)?, rw.div_ceil(df), rh.div_ceil(df))?;
        }
        let trans_base = regs.len();
        for t in trans {
            let df = t.downsample_factor.max(1) as u32;
            push_tex(device, &mut regs, dxgi_format(t.format)?, rw.div_ceil(df), rh.div_ceil(df))?;
        }
        // App planes: IN_MV, IN_NR, IN_VIEWZ, IN_DIFF, IN_SPEC, OUT_DIFF,
        // OUT_SPEC, OUT_VALIDATION (RGBA8 debug — cheap, always present so
        // FR_NRD_DEBUG needs no realloc).
        let plane_fmts = [
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R10G10B10A2_UNORM,
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R8G8B8A8_UNORM,
        ];
        let mut plane_idx = [0usize; 8];
        for (k, &f) in plane_fmts.iter().enumerate() {
            plane_idx[k] = push_tex(device, &mut regs, f, rw, rh)?;
        }

        // --- Staging heap (CPU-only): SRV + UAV per reg.
        let inc = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        let staging: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: (regs.len() * 2) as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("nrd: staging heap: {e}"))?;
        let sbase = unsafe { staging.GetCPUDescriptorHandleForHeapStart() };
        for (i, r) in regs.iter().enumerate() {
            let srv = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: sbase.ptr + (2 * i) as usize * inc as usize };
            let uav = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: sbase.ptr + (2 * i + 1) as usize * inc as usize,
            };
            unsafe {
                device.CreateShaderResourceView(&r.res, None, srv);
                device.CreateUnorderedAccessView(&r.res, None, None, uav);
            }
        }

        // --- Shader-visible ring.
        let set_stride = max_tex + max_uav;
        let ring: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: (RING_FRAMES * MAX_DISPATCHES) as u32 * set_stride,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("nrd: ring heap: {e}"))?;

        // --- CB upload ring, persistently mapped.
        let cb_slot_size =
            (d.constant_buffer_max_data_size as usize).next_multiple_of(CB_ALIGN).max(CB_ALIGN);
        let cb = d3d12::UploadBuffer::new(device, RING_FRAMES * MAX_DISPATCHES * cb_slot_size)
            .map_err(|e| format!("nrd: cb ring: {e}"))?;

        eprintln!(
            "nrd: v{}.{}.{} — {} pipelines, pools {}+{}, set {}t+{}u, cb {} B",
            inst.version.0,
            inst.version.1,
            inst.version.2,
            psos.len(),
            perm.len(),
            trans.len(),
            max_tex,
            max_uav,
            d.constant_buffer_max_data_size
        );
        let _ = s!(""); // keep the s! import alive for future named objects
        Ok(Self {
            nrd: inst,
            root_sig,
            psos,
            regs,
            perm_base,
            trans_base,
            plane_idx,
            staging,
            ring,
            set_stride,
            max_tex,
            cb,
            cb_slot_size,
            inc,
            rw,
            rh,
            frame_counter: std::cell::Cell::new(0),
        })
    }

    pub fn plane_in_mv(&self) -> &ID3D12Resource {
        &self.regs[self.plane_idx[0]].res
    }
    pub fn plane_in_nr(&self) -> &ID3D12Resource {
        &self.regs[self.plane_idx[1]].res
    }
    pub fn plane_in_viewz(&self) -> &ID3D12Resource {
        &self.regs[self.plane_idx[2]].res
    }
    pub fn plane_in_diff(&self) -> &ID3D12Resource {
        &self.regs[self.plane_idx[3]].res
    }
    pub fn plane_in_spec(&self) -> &ID3D12Resource {
        &self.regs[self.plane_idx[4]].res
    }
    pub fn plane_out_diff(&self) -> &ID3D12Resource {
        &self.regs[self.plane_idx[5]].res
    }
    pub fn plane_out_spec(&self) -> &ID3D12Resource {
        &self.regs[self.plane_idx[6]].res
    }

    fn reg_for(&self, r: &nrd::ResourceDesc) -> Result<usize> {
        Ok(match r.ty {
            nrd::RES_PERMANENT_POOL => self.perm_base + r.index_in_pool as usize,
            nrd::RES_TRANSIENT_POOL => self.trans_base + r.index_in_pool as usize,
            nrd::RES_IN_MV => self.plane_idx[0],
            nrd::RES_IN_NORMAL_ROUGHNESS => self.plane_idx[1],
            nrd::RES_IN_VIEWZ => self.plane_idx[2],
            nrd::RES_IN_DIFF_RADIANCE_HITDIST => self.plane_idx[3],
            nrd::RES_IN_SPEC_RADIANCE_HITDIST => self.plane_idx[4],
            nrd::RES_OUT_DIFF_RADIANCE_HITDIST => self.plane_idx[5],
            nrd::RES_OUT_SPEC_RADIANCE_HITDIST => self.plane_idx[6],
            nrd::RES_OUT_VALIDATION => self.plane_idx[7],
            other => return Err(format!("nrd: unmapped ResourceType {other}")),
        })
    }

    /// Record one frame's NRD passes. `slot` picks the descriptor/CB ring
    /// window (the caller's frame-in-flight slot). The caller has already
    /// filled the IN planes (cs_nrd_pack) and fenced them (queue order); the
    /// OUT planes are ready for cs_nrd_out when this returns. Every touched
    /// resource is restored to UNORDERED_ACCESS.
    pub fn record(
        &mut self,
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        slot: usize,
        cs: &nrd::CommonSettings,
        rs: &nrd::ReblurSettings,
    ) -> Result<()> {
        self.nrd.set_common_settings(cs)?;
        self.nrd.set_reblur_settings(0, rs)?;
        // Snapshot: the returned slice borrows the instance and is overwritten
        // by the next call — and the loop below needs &self.
        let dispatches: Vec<nrd::DispatchDesc> = {
            let ds = self.nrd.compute_dispatches(&[0])?;
            ds.iter()
                .map(|d| nrd::DispatchDesc {
                    name: d.name,
                    identifier: d.identifier,
                    resources: d.resources,
                    resources_num: d.resources_num,
                    constant_buffer_data: d.constant_buffer_data,
                    constant_buffer_data_size: d.constant_buffer_data_size,
                    constant_buffer_data_matches_previous_dispatch: d
                        .constant_buffer_data_matches_previous_dispatch,
                    pipeline_index: d.pipeline_index,
                    grid_width: d.grid_width,
                    grid_height: d.grid_height,
                })
                .collect()
        };
        if dispatches.len() > MAX_DISPATCHES {
            return Err(format!(
                "nrd: {} dispatches > ring capacity {MAX_DISPATCHES}",
                dispatches.len()
            ));
        }
        let _ev = super::pix::scope(list, c"nrd");
        let ring_gpu0 = unsafe { self.ring.GetGPUDescriptorHandleForHeapStart() };
        let ring_cpu0 = unsafe { self.ring.GetCPUDescriptorHandleForHeapStart() };
        let stage_cpu0 = unsafe { self.staging.GetCPUDescriptorHandleForHeapStart() };
        let frame_base = (slot % RING_FRAMES) * MAX_DISPATCHES;
        unsafe {
            list.SetComputeRootSignature(&self.root_sig);
            list.SetDescriptorHeaps(&[Some(self.ring.clone())]);
        }
        let mut cb_off_last = None;
        for (di, dsp) in dispatches.iter().enumerate() {
            // Barriers: TEXTURE bindings read as SRV, STORAGE as UAV; a
            // global UAV barrier orders back-to-back UAV writers/readers.
            let resources =
                unsafe { std::slice::from_raw_parts(dsp.resources, dsp.resources_num as usize) };
            let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
            for r in resources {
                let idx = self.reg_for(r)?;
                let want = if r.descriptor_type == nrd::DESC_TEXTURE {
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                } else {
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                };
                let cur = self.regs[idx].state.get();
                if cur != want {
                    barriers.push(d3d12::transition(&self.regs[idx].res, cur, want));
                    self.regs[idx].state.set(want);
                }
            }
            // The global UAV barrier (None) fences the previous dispatch's
            // storage writes for this one's reads — cheap (the ladder's
            // 11 µs-for-24-barriers measurement) and always correct.
            barriers.push(D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
            });
            unsafe { list.ResourceBarrier(&barriers) };

            // Descriptor set: SRVs at [0..), UAVs at [max_tex..), in the
            // ranges' concatenated resource order.
            let set = frame_base + di;
            let set_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: ring_cpu0.ptr + (set as u32 * self.set_stride) as usize * self.inc as usize,
            };
            let (mut srv_i, mut uav_i) = (0u32, 0u32);
            for r in resources {
                let idx = self.reg_for(r)?;
                let (slot_off, stage_off) = if r.descriptor_type == nrd::DESC_TEXTURE {
                    let o = srv_i;
                    srv_i += 1;
                    (o, (2 * idx) as u32)
                } else {
                    let o = self.max_tex + uav_i;
                    uav_i += 1;
                    (o, (2 * idx + 1) as u32)
                };
                unsafe {
                    device.CopyDescriptorsSimple(
                        1,
                        D3D12_CPU_DESCRIPTOR_HANDLE {
                            ptr: set_cpu.ptr + (slot_off * self.inc) as usize,
                        },
                        D3D12_CPU_DESCRIPTOR_HANDLE {
                            ptr: stage_cpu0.ptr + (stage_off * self.inc) as usize,
                        },
                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    )
                };
            }

            // Constants: NRD flags identical-CB runs; reuse the last upload.
            let cb_off = if dsp.constant_buffer_data_size > 0 {
                let reuse = dsp.constant_buffer_data_matches_previous_dispatch != 0;
                match (reuse, cb_off_last) {
                    (true, Some(off)) => Some(off),
                    _ => {
                        let off = ((slot % RING_FRAMES) * MAX_DISPATCHES + di) * self.cb_slot_size;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                dsp.constant_buffer_data,
                                self.cb.ptr.add(off),
                                dsp.constant_buffer_data_size as usize,
                            )
                        };
                        cb_off_last = Some(off);
                        Some(off)
                    }
                }
            } else {
                None
            };
            unsafe {
                if let Some(off) = cb_off {
                    list.SetComputeRootConstantBufferView(
                        0,
                        self.cb.resource.GetGPUVirtualAddress() + off as u64,
                    );
                }
                list.SetComputeRootDescriptorTable(
                    1,
                    D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: ring_gpu0.ptr + (set as u32 * self.set_stride) as u64 * self.inc as u64,
                    },
                );
                list.SetPipelineState(&self.psos[dsp.pipeline_index as usize]);
                list.Dispatch(dsp.grid_width as u32, dsp.grid_height as u32, 1);
            }
        }
        // Restore the rest state (+ one trailing global UAV barrier so the
        // bridge's cs_nrd_out sees the OUT writes).
        let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
        for r in &self.regs {
            let cur = r.state.get();
            if cur != D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
                barriers.push(d3d12::transition(&r.res, cur, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
                r.state.set(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
        }
        barriers.push(D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        });
        unsafe { list.ResourceBarrier(&barriers) };
        self.frame_counter.set(self.frame_counter.get() + 1);
        Ok(())
    }
}

