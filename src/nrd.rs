//! NRD (NVIDIA Real-time Denoisers) — the runtime FFI for `--nrd`, the
//! pre-upscale ReBLUR denoiser that cleans the 1-spp signal at render res
//! before XeSS / FSR 3.1 upscale it (DLSS-RR / FSR4-RR sessions already
//! denoise and never arm this).
//!
//! FOOTPRINT POLICY (the xess.rs/OIDN shape): nothing links NRD and NO NRD
//! header or shader is committed — the NVIDIA RTX SDKs license forbids source
//! redistribution, which is also why `install-prerequisites.bat nrd` BUILDS
//! the pinned tag locally instead of downloading a binary (NVIDIA publishes
//! none). `NRD.dll` is `LoadLibraryExW`'d at runtime from `--nrd-path`
//! (default `SDKs\NRD\bin`, env `FRUSTRACER_NRD_PATH`); a missing DLL is a
//! loud shed, never an error, and every `--check*` except the DLL half of
//! `--check-nrd` stays DLL-free.
//!
//! TRANSCRIPTION CONTRACT: every `#[repr(C)]` struct, enum value, and default
//! below is transcribed from NRD **v4.17.3** `Include/{NRD.h, NRDDescs.h,
//! NRDSettings.h}` — the tag `install-prerequisites.bat`'s `NRD_TAG` pins.
//! Regenerate against the pinned tag as a whole, never hand-edit piecemeal
//! (the nppd.rs `OrtApi` discipline). The `PIN_*` size asserts carry the
//! MSVC-x64 `sizeof`/`offsetof` ground truth measured by compiling a sizer
//! against the pinned headers (2026-08-08); `NRD_API` is `extern "C"`
//! dllexport in the default shared build, so the seven entry points resolve
//! unmangled. C++ references in the signatures are ABI pointers on x64.
//! `Nrd::new` additionally gates `GetLibraryDesc` at runtime — version 4.17
//! AND normalEncoding == 2 (R10G10B10A2 oct) AND roughnessEncoding == 1
//! (linear), the build contract the install script's cmake flags fix — so a
//! drifted DLL sheds loudly instead of corrupting.
//!
//! The `oracle` module holds pure-Rust twins of the NRD.hlsli packing math
//! the bridge kernels reimplement (YCoCg, the 10-10-10-2 oct normal encode,
//! ReBLUR's hit-distance normalization). They exist for the gates: N0 pins
//! the math DLL-free in `--check-nrd`, and M3's pack-vs-oracle gate compares
//! `cs_nrd_pack`'s planes against them on the same readback inputs. The
//! formulas are REIMPLEMENTED from the spec — never paste NRD.hlsli into the
//! tree or a shader concat (license).

#![allow(dead_code)]

use std::ffi::c_void;

/// The pinned NRD version — must match `install-prerequisites.bat`'s NRD_TAG.
pub const PIN_MAJOR: u8 = 4;
pub const PIN_MINOR: u8 = 17;

// ---------------------------------------------------------------------------
// Enums (values transcribed from NRDDescs.h / NRDSettings.h; carried as raw
// integers in the structs so an unexpected DLL value can never be UB — named
// consts below are the vocabulary our call sites use).
// ---------------------------------------------------------------------------

pub const RESULT_SUCCESS: u32 = 0;
pub const RESULT_FAILURE: u32 = 1;
pub const RESULT_INVALID_ARGUMENT: u32 = 2;
pub const RESULT_UNSUPPORTED: u32 = 3;
pub const RESULT_NON_UNIQUE_IDENTIFIER: u32 = 4;

// ResourceType (NRDDescs.h order — the wire between GetComputeDispatches and
// our binding loop; only the ones our denoisers reach are named).
pub const RES_IN_MV: u32 = 0;
pub const RES_IN_NORMAL_ROUGHNESS: u32 = 1;
pub const RES_IN_VIEWZ: u32 = 2;
pub const RES_IN_DIFF_CONFIDENCE: u32 = 3;
pub const RES_IN_SPEC_CONFIDENCE: u32 = 4;
pub const RES_IN_DISOCCLUSION_THRESHOLD_MIX: u32 = 5;
pub const RES_IN_DIFF_RADIANCE_HITDIST: u32 = 6;
pub const RES_IN_SPEC_RADIANCE_HITDIST: u32 = 7;
pub const RES_IN_DIFF_HITDIST: u32 = 8;
pub const RES_IN_SPEC_HITDIST: u32 = 9;
pub const RES_IN_DIFF_DIRECTION_HITDIST: u32 = 10;
pub const RES_IN_DIFF_SH0: u32 = 11;
pub const RES_IN_DIFF_SH1: u32 = 12;
pub const RES_IN_SPEC_SH0: u32 = 13;
pub const RES_IN_SPEC_SH1: u32 = 14;
pub const RES_IN_PENUMBRA: u32 = 15;
pub const RES_IN_TRANSLUCENCY: u32 = 16;
pub const RES_IN_SIGNAL: u32 = 17;
pub const RES_OUT_DIFF_RADIANCE_HITDIST: u32 = 18;
pub const RES_OUT_SPEC_RADIANCE_HITDIST: u32 = 19;
pub const RES_OUT_DIFF_SH0: u32 = 20;
pub const RES_OUT_DIFF_SH1: u32 = 21;
pub const RES_OUT_SPEC_SH0: u32 = 22;
pub const RES_OUT_SPEC_SH1: u32 = 23;
pub const RES_OUT_DIFF_HITDIST: u32 = 24;
pub const RES_OUT_SPEC_HITDIST: u32 = 25;
pub const RES_OUT_DIFF_DIRECTION_HITDIST: u32 = 26;
pub const RES_OUT_SHADOW_TRANSLUCENCY: u32 = 27;
pub const RES_OUT_SIGNAL: u32 = 28;
pub const RES_OUT_VALIDATION: u32 = 29;
pub const RES_TRANSIENT_POOL: u32 = 30;
pub const RES_PERMANENT_POOL: u32 = 31;

// Denoiser (NRDDescs.h order).
pub const DENOISER_REBLUR_DIFFUSE_SPECULAR: u32 = 6;
pub const DENOISER_SIGMA_SHADOW: u32 = 16;
pub const DENOISER_REFERENCE: u32 = 18;

// Format (NRDDescs.h order; the u32 our pool-allocation maps to DXGI).
pub const FORMAT_R8_UNORM: u32 = 0;
pub const FORMAT_RGBA8_UNORM: u32 = 8;
pub const FORMAT_R16_SFLOAT: u32 = 17;
pub const FORMAT_RG16_SFLOAT: u32 = 22;
pub const FORMAT_RGBA16_SFLOAT: u32 = 27;
pub const FORMAT_R32_SFLOAT: u32 = 30;
pub const FORMAT_R10_G10_B10_A2_UNORM: u32 = 40;
pub const FORMAT_R11_G11_B10_UFLOAT: u32 = 42;
pub const FORMAT_MAX_NUM: u32 = 44;

// DescriptorType.
pub const DESC_TEXTURE: u32 = 0; // SRV
pub const DESC_STORAGE_TEXTURE: u32 = 1; // UAV

// Sampler (InstanceDesc::samplers order).
pub const SAMPLER_NEAREST_CLAMP: u32 = 0;
pub const SAMPLER_LINEAR_CLAMP: u32 = 1;

// NormalEncoding / RoughnessEncoding (u8 in LibraryDesc).
pub const NORMAL_ENCODING_R10G10B10A2_UNORM: u8 = 2;
pub const ROUGHNESS_ENCODING_LINEAR: u8 = 1;

// AccumulationMode (u8 in CommonSettings).
pub const ACCUM_CONTINUE: u8 = 0;
pub const ACCUM_RESTART: u8 = 1;
pub const ACCUM_CLEAR_AND_RESTART: u8 = 2;

// HitDistanceReconstructionMode (u8 in ReblurSettings).
pub const HITDIST_RECONSTRUCTION_OFF: u8 = 0;
pub const HITDIST_RECONSTRUCTION_AREA_3X3: u8 = 1;

pub const NRD_FP16_MAX: f32 = 65504.0;
pub const NRD_EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Structs (MSVC x64 layout; sizes pinned by the const asserts at the bottom).
// ---------------------------------------------------------------------------

pub type Identifier = u32;

/// Opaque; only ever held behind a pointer.
#[repr(C)]
pub struct Instance {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct AllocationCallbacks {
    // Zeroed = "use NRD's own allocator" (CreateInstance installs defaults).
    pub allocate: Option<unsafe extern "system" fn(*mut c_void, usize, usize) -> *mut c_void>,
    pub reallocate:
        Option<unsafe extern "system" fn(*mut c_void, *mut c_void, usize, usize) -> *mut c_void>,
    pub free: Option<unsafe extern "system" fn(*mut c_void, *mut c_void)>,
    pub user_arg: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SpirvBindingOffsets {
    pub sampler_offset: u32,
    pub texture_offset: u32,
    pub constant_buffer_offset: u32,
    pub storage_texture_and_buffer_offset: u32,
}

#[repr(C)]
pub struct LibraryDesc {
    pub spirv_binding_offsets: SpirvBindingOffsets,
    pub supported_denoisers: *const u32,
    pub supported_denoisers_num: u32,
    pub version_major: u8,
    pub version_minor: u8,
    pub version_build: u8,
    pub normal_encoding: u8,
    pub roughness_encoding: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DenoiserDesc {
    pub identifier: Identifier,
    pub denoiser: u32,
}

#[repr(C)]
pub struct InstanceCreationDesc {
    pub allocation_callbacks: AllocationCallbacks,
    pub denoisers: *const DenoiserDesc,
    pub denoisers_num: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TextureDesc {
    pub format: u32,
    pub downsample_factor: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ResourceDesc {
    pub descriptor_type: u32,
    pub ty: u32, // ResourceType
    pub index_in_pool: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ResourceRangeDesc {
    pub descriptor_type: u32,
    pub descriptors_num: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ComputeShaderDesc {
    pub bytecode: *const c_void,
    pub size: u64,
}

#[repr(C)]
pub struct PipelineDesc {
    pub compute_shader_dxbc: ComputeShaderDesc,
    pub compute_shader_dxil: ComputeShaderDesc,
    pub compute_shader_spirv: ComputeShaderDesc,
    pub resource_ranges: *const ResourceRangeDesc,
    pub resource_ranges_num: u32,
    pub has_constant_data: u8,
    pub shader_identifier: [u8; 256],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DescriptorPoolDesc {
    pub per_set_textures_max_num: u32,
    pub per_set_storage_textures_max_num: u32,
    pub total_textures_num: u32,
    pub total_storage_textures_num: u32,
    pub sets_max_num: u32,
}

#[repr(C)]
pub struct InstanceDesc {
    pub constant_buffer_and_samplers_space_index: u32,
    pub resources_space_index: u32,
    pub constant_buffer_register_index: u32,
    pub samplers_base_register_index: u32,
    pub resources_base_register_index: u32,
    pub constant_buffer_max_data_size: u32,
    pub samplers: *const u32,
    pub samplers_num: u32,
    pub shader_entry_point: *const i8,
    pub pipelines: *const PipelineDesc,
    pub pipelines_num: u32,
    pub permanent_pool: *const TextureDesc,
    pub permanent_pool_size: u32,
    pub transient_pool: *const TextureDesc,
    pub transient_pool_size: u32,
    pub descriptor_pool_desc: DescriptorPoolDesc,
}

#[repr(C)]
pub struct DispatchDesc {
    pub name: *const i8,
    pub identifier: Identifier,
    pub resources: *const ResourceDesc,
    pub resources_num: u32,
    pub constant_buffer_data: *const u8,
    pub constant_buffer_data_size: u32,
    pub constant_buffer_data_matches_previous_dispatch: u8,
    pub pipeline_index: u16,
    pub grid_width: u16,
    pub grid_height: u16,
}

#[repr(C)]
pub struct CommonSettings {
    // Column-major, vector-is-a-column, NON-jittered (NRDSettings.h).
    pub view_to_clip_matrix: [f32; 16],
    pub view_to_clip_matrix_prev: [f32; 16],
    pub world_to_view_matrix: [f32; 16],
    pub world_to_view_matrix_prev: [f32; 16],
    pub world_prev_to_world_matrix: [f32; 16],
    pub motion_vector_scale: [f32; 3],
    pub camera_jitter: [f32; 2],
    pub camera_jitter_prev: [f32; 2],
    pub resource_size: [u16; 2],
    pub resource_size_prev: [u16; 2],
    pub rect_size: [u16; 2],
    pub rect_size_prev: [u16; 2],
    pub view_z_scale: f32,
    pub time_delta_between_frames: f32,
    pub denoising_range: f32,
    pub disocclusion_threshold: f32,
    pub disocclusion_threshold_alternate: f32,
    pub camera_attached_reflection_material_id: f32,
    pub strand_material_id: f32,
    pub history_fix_alternate_pixel_stride_material_id: f32,
    pub strand_thickness: f32,
    pub split_screen: f32,
    pub printf_at: [u16; 2],
    pub debug: f32,
    pub rect_origin: [u32; 2],
    pub frame_index: u32,
    pub accumulation_mode: u8,
    pub is_motion_vector_in_world_space: u8,
    pub is_history_confidence_available: u8,
    pub is_disocclusion_threshold_mix_available: u8,
    pub enable_validation: u8,
}

impl Default for CommonSettings {
    fn default() -> Self {
        let ident = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        Self {
            view_to_clip_matrix: [0.0; 16],
            view_to_clip_matrix_prev: [0.0; 16],
            world_to_view_matrix: [0.0; 16],
            world_to_view_matrix_prev: [0.0; 16],
            world_prev_to_world_matrix: ident,
            motion_vector_scale: [1.0, 1.0, 0.0],
            camera_jitter: [0.0; 2],
            camera_jitter_prev: [0.0; 2],
            resource_size: [0; 2],
            resource_size_prev: [0; 2],
            rect_size: [0; 2],
            rect_size_prev: [0; 2],
            view_z_scale: 1.0,
            time_delta_between_frames: 0.0,
            denoising_range: 500000.0,
            disocclusion_threshold: 0.01,
            disocclusion_threshold_alternate: 0.05,
            camera_attached_reflection_material_id: 999.0,
            strand_material_id: 999.0,
            history_fix_alternate_pixel_stride_material_id: 999.0,
            strand_thickness: 80e-6,
            split_screen: 0.0,
            printf_at: [9999, 9999],
            debug: 0.0,
            rect_origin: [0; 2],
            frame_index: 0,
            accumulation_mode: ACCUM_CONTINUE,
            is_motion_vector_in_world_space: 0,
            is_history_confidence_available: 0,
            is_disocclusion_threshold_mix_available: 0,
            enable_validation: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReblurHitDistanceParameters {
    pub a: f32, // (units) constant
    pub b: f32, // viewZ-linear scale
    pub c: f32, // roughness scale
}

impl Default for ReblurHitDistanceParameters {
    fn default() -> Self {
        Self { a: 3.0, b: 0.1, c: 20.0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReblurAntilagSettings {
    pub luminance_sigma_scale: f32,
    pub luminance_sensitivity: f32,
}

impl Default for ReblurAntilagSettings {
    fn default() -> Self {
        Self { luminance_sigma_scale: 2.0, luminance_sensitivity: 3.0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReblurResponsiveAccumulationSettings {
    pub roughness_threshold: f32,
    pub min_accumulated_frame_num: u32,
}

impl Default for ReblurResponsiveAccumulationSettings {
    fn default() -> Self {
        Self { roughness_threshold: 0.0, min_accumulated_frame_num: 3 }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReblurConvergenceSettings {
    pub s: f32,
    pub b: f32,
    pub p: f32,
}

impl Default for ReblurConvergenceSettings {
    fn default() -> Self {
        Self { s: 1.0, b: 0.2, p: 0.8 }
    }
}

#[repr(C)]
pub struct ReblurSettings {
    pub hit_distance_parameters: ReblurHitDistanceParameters,
    pub antilag_settings: ReblurAntilagSettings,
    pub responsive_accumulation_settings: ReblurResponsiveAccumulationSettings,
    pub convergence_settings: ReblurConvergenceSettings,
    pub max_accumulated_frame_num: u32,
    pub max_fast_accumulated_frame_num: u32,
    pub max_stabilized_frame_num: u32,
    pub history_fix_frame_num: u32,
    pub history_fix_base_pixel_stride: u32,
    pub history_fix_alternate_pixel_stride: u32,
    pub fast_history_clamping_sigma_scale: f32,
    pub diffuse_prepass_blur_radius: f32,
    pub specular_prepass_blur_radius: f32,
    pub min_hit_distance_weight: f32,
    pub min_blur_radius: f32,
    pub max_blur_radius: f32,
    pub lobe_angle_fraction: f32,
    pub roughness_fraction: f32,
    pub plane_distance_sensitivity: f32,
    pub specular_probability_thresholds_for_mv_modification: [f32; 2],
    pub firefly_suppressor_min_relative_scale: f32,
    pub min_material_for_diffuse: f32,
    pub min_material_for_specular: f32,
    pub checkerboard_mode: u8,
    pub hit_distance_reconstruction_mode: u8,
    pub enable_anti_firefly: u8,
    pub use_prepass_only_for_specular_motion_estimation: u8,
    pub return_history_length_instead_of_occlusion: u8,
}

impl Default for ReblurSettings {
    fn default() -> Self {
        Self {
            hit_distance_parameters: Default::default(),
            antilag_settings: Default::default(),
            responsive_accumulation_settings: Default::default(),
            convergence_settings: Default::default(),
            max_accumulated_frame_num: 30,
            max_fast_accumulated_frame_num: 6,
            max_stabilized_frame_num: 63,
            history_fix_frame_num: 3,
            history_fix_base_pixel_stride: 14,
            history_fix_alternate_pixel_stride: 14,
            fast_history_clamping_sigma_scale: 2.0,
            diffuse_prepass_blur_radius: 30.0,
            specular_prepass_blur_radius: 50.0,
            min_hit_distance_weight: 0.1,
            min_blur_radius: 1.0,
            max_blur_radius: 30.0,
            lobe_angle_fraction: 0.15,
            roughness_fraction: 0.15,
            plane_distance_sensitivity: 0.02,
            specular_probability_thresholds_for_mv_modification: [0.5, 0.9],
            firefly_suppressor_min_relative_scale: 2.0,
            min_material_for_diffuse: 4.0,
            min_material_for_specular: 4.0,
            checkerboard_mode: 0,
            hit_distance_reconstruction_mode: HITDIST_RECONSTRUCTION_OFF,
            enable_anti_firefly: 1,
            use_prepass_only_for_specular_motion_estimation: 0,
            return_history_length_instead_of_occlusion: 0,
        }
    }
}

/// Runtime `ReblurSettings` overrides (the `--nrd-*` tuning levers, the
/// `fsr::DenoiseTuning` shape): every field is `None` by default, which
/// changes NOTHING — a flagless session sends exactly the settings it always
/// did (defaults + the AREA_3X3 departure in `nrd_gpu::reblur_settings`).
/// These exist because ReBLUR's compile-time performance mode covers only the
/// shader-internal half of the cost; `max_stabilized_frames = 0` is the one
/// lever that genuinely DROPS a pass (TemporalStabilization), and
/// `prepass_radius = 0` disables both prepasses. Written once from main's
/// lever block (`set_tuning`), read by `nrd_gpu::reblur_settings()` — the N4
/// gate inherits whatever the session runs, the `--cam` user's-own-risk class.
#[derive(Clone, Copy, Default, Debug)]
pub struct ReblurTuning {
    pub max_stabilized_frames: Option<u32>,
    pub prepass_radius: Option<f32>,
    pub anti_firefly: Option<bool>,
    pub max_accum_frames: Option<u32>,
}

impl ReblurTuning {
    pub fn any(&self) -> bool {
        self.max_stabilized_frames.is_some()
            || self.prepass_radius.is_some()
            || self.anti_firefly.is_some()
            || self.max_accum_frames.is_some()
    }

    /// Fold the overrides into a settings struct (None leaves the field).
    pub fn apply(&self, rs: &mut ReblurSettings) {
        if let Some(n) = self.max_stabilized_frames {
            rs.max_stabilized_frame_num = n;
        }
        if let Some(r) = self.prepass_radius {
            rs.diffuse_prepass_blur_radius = r;
            rs.specular_prepass_blur_radius = r;
        }
        if let Some(b) = self.anti_firefly {
            rs.enable_anti_firefly = b as u8;
        }
        if let Some(n) = self.max_accum_frames {
            rs.max_accumulated_frame_num = n;
        }
    }
}

static TUNING: std::sync::OnceLock<ReblurTuning> = std::sync::OnceLock::new();

/// One writer: main's lever block. A second call is ignored (OnceLock), which
/// is fine — the block runs once per process.
pub fn set_tuning(t: ReblurTuning) {
    let _ = TUNING.set(t);
}

pub fn tuning() -> ReblurTuning {
    TUNING.get().copied().unwrap_or_default()
}

#[repr(C)]
pub struct SigmaSettings {
    pub light_direction: [f32; 3],
    pub plane_distance_sensitivity: f32,
    pub max_stabilized_frame_num: u32,
}

impl Default for SigmaSettings {
    fn default() -> Self {
        Self {
            light_direction: [0.0; 3],
            plane_distance_sensitivity: 0.02,
            max_stabilized_frame_num: 5,
        }
    }
}

// The measured MSVC-x64 ground truth (sizer against the pinned headers).
const _: () = {
    assert!(std::mem::size_of::<AllocationCallbacks>() == 32);
    assert!(std::mem::size_of::<SpirvBindingOffsets>() == 16);
    assert!(std::mem::size_of::<LibraryDesc>() == 40);
    assert!(std::mem::offset_of!(LibraryDesc, supported_denoisers) == 16);
    assert!(std::mem::offset_of!(LibraryDesc, version_major) == 28);
    assert!(std::mem::offset_of!(LibraryDesc, normal_encoding) == 31);
    assert!(std::mem::size_of::<DenoiserDesc>() == 8);
    assert!(std::mem::size_of::<InstanceCreationDesc>() == 48);
    assert!(std::mem::size_of::<TextureDesc>() == 8);
    assert!(std::mem::size_of::<ResourceDesc>() == 12);
    assert!(std::mem::offset_of!(ResourceDesc, index_in_pool) == 8);
    assert!(std::mem::size_of::<ResourceRangeDesc>() == 8);
    assert!(std::mem::size_of::<ComputeShaderDesc>() == 16);
    assert!(std::mem::size_of::<PipelineDesc>() == 320);
    assert!(std::mem::offset_of!(PipelineDesc, resource_ranges) == 48);
    assert!(std::mem::offset_of!(PipelineDesc, has_constant_data) == 60);
    assert!(std::mem::offset_of!(PipelineDesc, shader_identifier) == 61);
    assert!(std::mem::size_of::<DescriptorPoolDesc>() == 20);
    assert!(std::mem::size_of::<InstanceDesc>() == 112);
    assert!(std::mem::offset_of!(InstanceDesc, constant_buffer_max_data_size) == 20);
    assert!(std::mem::offset_of!(InstanceDesc, samplers) == 24);
    assert!(std::mem::offset_of!(InstanceDesc, shader_entry_point) == 40);
    assert!(std::mem::offset_of!(InstanceDesc, pipelines) == 48);
    assert!(std::mem::offset_of!(InstanceDesc, permanent_pool) == 64);
    assert!(std::mem::offset_of!(InstanceDesc, transient_pool) == 80);
    assert!(std::mem::offset_of!(InstanceDesc, descriptor_pool_desc) == 92);
    assert!(std::mem::size_of::<DispatchDesc>() == 56);
    assert!(std::mem::offset_of!(DispatchDesc, resources) == 16);
    assert!(std::mem::offset_of!(DispatchDesc, constant_buffer_data) == 32);
    assert!(
        std::mem::offset_of!(DispatchDesc, constant_buffer_data_matches_previous_dispatch) == 44
    );
    assert!(std::mem::offset_of!(DispatchDesc, pipeline_index) == 46);
    assert!(std::mem::offset_of!(DispatchDesc, grid_height) == 50);
    assert!(std::mem::size_of::<CommonSettings>() == 432);
    assert!(std::mem::offset_of!(CommonSettings, motion_vector_scale) == 320);
    assert!(std::mem::offset_of!(CommonSettings, camera_jitter) == 332);
    assert!(std::mem::offset_of!(CommonSettings, resource_size) == 348);
    assert!(std::mem::offset_of!(CommonSettings, view_z_scale) == 364);
    assert!(std::mem::offset_of!(CommonSettings, split_screen) == 400);
    assert!(std::mem::offset_of!(CommonSettings, printf_at) == 404);
    assert!(std::mem::offset_of!(CommonSettings, rect_origin) == 412);
    assert!(std::mem::offset_of!(CommonSettings, frame_index) == 420);
    assert!(std::mem::offset_of!(CommonSettings, accumulation_mode) == 424);
    assert!(std::mem::offset_of!(CommonSettings, enable_validation) == 428);
    assert!(std::mem::size_of::<ReblurSettings>() == 128);
    assert!(std::mem::offset_of!(ReblurSettings, max_accumulated_frame_num) == 40);
    assert!(std::mem::offset_of!(ReblurSettings, fast_history_clamping_sigma_scale) == 64);
    assert!(
        std::mem::offset_of!(ReblurSettings, specular_probability_thresholds_for_mv_modification)
            == 100
    );
    assert!(std::mem::offset_of!(ReblurSettings, checkerboard_mode) == 120);
    assert!(std::mem::offset_of!(ReblurSettings, enable_anti_firefly) == 122);
    assert!(std::mem::offset_of!(ReblurSettings, return_history_length_instead_of_occlusion) == 124);
    assert!(std::mem::size_of::<SigmaSettings>() == 20);
};

// ---------------------------------------------------------------------------
// Loader (the xess.rs shape: LoadLibraryExW + GetProcAddress fn table).
// ---------------------------------------------------------------------------

mod loader {
    use super::*;
    use windows::core::{PCSTR, PCWSTR};
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{
        GetProcAddress, LoadLibraryExW, LOAD_WITH_ALTERED_SEARCH_PATH,
    };

    /// The resolved NRD entry points (extern "C" ⇒ undecorated x64 exports).
    pub(super) struct Api {
        pub create_instance:
            unsafe extern "system" fn(*const InstanceCreationDesc, *mut *mut Instance) -> u32,
        pub destroy_instance: unsafe extern "system" fn(*mut Instance),
        pub get_library_desc: unsafe extern "system" fn() -> *const LibraryDesc,
        pub get_instance_desc: unsafe extern "system" fn(*const Instance) -> *const InstanceDesc,
        pub set_common_settings:
            unsafe extern "system" fn(*mut Instance, *const CommonSettings) -> u32,
        pub set_denoiser_settings:
            unsafe extern "system" fn(*mut Instance, Identifier, *const c_void) -> u32,
        pub get_compute_dispatches: unsafe extern "system" fn(
            *mut Instance,
            *const Identifier,
            u32,
            *mut *const DispatchDesc,
            *mut u32,
        ) -> u32,
    }

    fn load_dll(dir: &str, name: &str) -> Result<HMODULE, String> {
        let path = format!("{dir}\\{name}");
        let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
        unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, LOAD_WITH_ALTERED_SEARCH_PATH) }
            .map_err(|e| format!("failed to load {path}: {e}"))
    }

    macro_rules! resolve {
        ($h:expr, $name:literal) => {{
            let sym = unsafe { GetProcAddress($h, PCSTR(concat!($name, "\0").as_ptr())) }
                .ok_or_else(|| format!("NRD.dll: missing export {}", $name))?;
            #[allow(clippy::missing_transmute_annotations)]
            let f = unsafe { std::mem::transmute(sym) };
            f
        }};
    }

    pub(super) fn load(dll_dir: &str) -> Result<Api, String> {
        // Absolute path: ALTERED_SEARCH_PATH only helps absolute paths.
        let dir = std::fs::canonicalize(dll_dir)
            .map(|p| p.to_string_lossy().trim_start_matches(r"\\?\").to_string())
            .unwrap_or_else(|_| dll_dir.to_string());
        let h = load_dll(&dir, "NRD.dll")?;
        // The HMODULE is never freed (the SL/OIDN policy — fn pointers must
        // stay valid for the process lifetime).
        Ok(Api {
            create_instance: resolve!(h, "CreateInstance"),
            destroy_instance: resolve!(h, "DestroyInstance"),
            get_library_desc: resolve!(h, "GetLibraryDesc"),
            get_instance_desc: resolve!(h, "GetInstanceDesc"),
            set_common_settings: resolve!(h, "SetCommonSettings"),
            set_denoiser_settings: resolve!(h, "SetDenoiserSettings"),
            get_compute_dispatches: resolve!(h, "GetComputeDispatches"),
        })
    }
}

/// A live NRD instance over a runtime-loaded NRD.dll.
pub struct Nrd {
    api: loader::Api,
    instance: *mut Instance,
    pub version: (u8, u8, u8),
}

impl Nrd {
    /// Load NRD.dll from `dll_dir`, gate the version/encoding pins, and
    /// create an instance over `denoisers`. Every failure is a `String` the
    /// caller sheds loudly with — never a panic.
    pub fn new(dll_dir: &str, denoisers: &[DenoiserDesc]) -> Result<Self, String> {
        let api = loader::load(dll_dir)?;
        let lib = unsafe { (api.get_library_desc)() };
        if lib.is_null() {
            return Err("NRD.dll: GetLibraryDesc returned null".into());
        }
        let lib = unsafe { &*lib };
        // The drift gate: the transcribed structs and the bridge kernels'
        // packing math are only valid against the pinned version + the
        // encodings the install script's cmake flags fixed.
        if lib.version_major != PIN_MAJOR || lib.version_minor != PIN_MINOR {
            return Err(format!(
                "NRD.dll v{}.{}.{} != pinned {PIN_MAJOR}.{PIN_MINOR} — rebuild via \
                 install-prerequisites.bat nrd (or /force after a repo update)",
                lib.version_major, lib.version_minor, lib.version_build
            ));
        }
        if lib.normal_encoding != NORMAL_ENCODING_R10G10B10A2_UNORM
            || lib.roughness_encoding != ROUGHNESS_ENCODING_LINEAR
        {
            return Err(format!(
                "NRD.dll encodings (normal {}, roughness {}) != pinned (2, 1) — the DLL was \
                 built without the install script's cmake pins; rebuild via \
                 install-prerequisites.bat nrd /force",
                lib.normal_encoding, lib.roughness_encoding
            ));
        }
        let version = (lib.version_major, lib.version_minor, lib.version_build);
        let creation = InstanceCreationDesc {
            allocation_callbacks: AllocationCallbacks {
                allocate: None,
                reallocate: None,
                free: None,
                user_arg: std::ptr::null_mut(),
            },
            denoisers: denoisers.as_ptr(),
            denoisers_num: denoisers.len() as u32,
        };
        let mut instance: *mut Instance = std::ptr::null_mut();
        let r = unsafe { (api.create_instance)(&creation, &mut instance) };
        if r != RESULT_SUCCESS || instance.is_null() {
            return Err(format!("NRD CreateInstance failed (result {r})"));
        }
        Ok(Self { api, instance, version })
    }

    /// The instance's resource/pipeline requirements. Valid for the
    /// instance's lifetime (NRD owns the memory).
    pub fn instance_desc(&self) -> &InstanceDesc {
        unsafe { &*(self.api.get_instance_desc)(self.instance) }
    }

    pub fn set_common_settings(&mut self, cs: &CommonSettings) -> Result<(), String> {
        let r = unsafe { (self.api.set_common_settings)(self.instance, cs) };
        if r != RESULT_SUCCESS {
            return Err(format!("NRD SetCommonSettings failed (result {r})"));
        }
        Ok(())
    }

    pub fn set_reblur_settings(
        &mut self,
        id: Identifier,
        s: &ReblurSettings,
    ) -> Result<(), String> {
        let r = unsafe {
            (self.api.set_denoiser_settings)(self.instance, id, s as *const _ as *const c_void)
        };
        if r != RESULT_SUCCESS {
            return Err(format!("NRD SetDenoiserSettings failed (result {r})"));
        }
        Ok(())
    }

    /// The frame's dispatch list. IMPORTANT: the returned slice is owned by
    /// the instance and is overwritten by the NEXT call — consume (record)
    /// it before calling again.
    pub fn compute_dispatches(&mut self, ids: &[Identifier]) -> Result<&[DispatchDesc], String> {
        let mut descs: *const DispatchDesc = std::ptr::null();
        let mut num: u32 = 0;
        let r = unsafe {
            (self.api.get_compute_dispatches)(
                self.instance,
                ids.as_ptr(),
                ids.len() as u32,
                &mut descs,
                &mut num,
            )
        };
        if r != RESULT_SUCCESS {
            return Err(format!("NRD GetComputeDispatches failed (result {r})"));
        }
        if num == 0 {
            return Ok(&[]);
        }
        if descs.is_null() {
            return Err("NRD GetComputeDispatches returned null with nonzero count".into());
        }
        Ok(unsafe { std::slice::from_raw_parts(descs, num as usize) })
    }
}

impl Drop for Nrd {
    fn drop(&mut self) {
        if !self.instance.is_null() {
            unsafe { (self.api.destroy_instance)(self.instance) };
        }
    }
}

// ---------------------------------------------------------------------------
// oracle — pure-Rust twins of the NRD.hlsli packing math the bridge kernels
// reimplement. One formula, three consumers (this module, cs_nrd_pack,
// cs_nrd_out) — the N0/N2 gates are what keep them in lockstep. Formulas
// reimplemented from NRD v4.17.3 Shaders/NRD.hlsli semantics; the licensed
// file itself is never committed or pasted.
// ---------------------------------------------------------------------------

pub mod oracle {
    use super::{NRD_EPS, NRD_FP16_MAX};

    /// `_NRD_LinearToYCoCg`: the reversible luma/chroma rotation ReBLUR
    /// packs radiance in (front end).
    pub fn linear_to_ycocg(c: [f32; 3]) -> [f32; 3] {
        let y = 0.25 * c[0] + 0.5 * c[1] + 0.25 * c[2];
        let co = 0.5 * c[0] - 0.5 * c[2];
        let cg = -0.25 * c[0] + 0.5 * c[1] - 0.25 * c[2];
        [y, co, cg]
    }

    /// `_NRD_YCoCgToLinear` (back end) — clamps negatives to 0, exactly as
    /// the HLSL does.
    pub fn ycocg_to_linear(c: [f32; 3]) -> [f32; 3] {
        let t = c[0] - c[2];
        let g = c[0] + c[2];
        let r = t + c[1];
        let b = t - c[1];
        [r.max(0.0), g.max(0.0), b.max(0.0)]
    }

    /// The front-end sanitize arm of `REBLUR_FrontEnd_PackRadianceAndNormHitDist`.
    pub fn sanitize_radiance(c: [f32; 3]) -> [f32; 3] {
        if c.iter().any(|v| !v.is_finite()) {
            return [0.0; 3];
        }
        [
            c[0].clamp(0.0, NRD_FP16_MAX),
            c[1].clamp(0.0, NRD_FP16_MAX),
            c[2].clamp(0.0, NRD_FP16_MAX),
        ]
    }

    /// `_NRD_EncodeNormalRoughness101010` — L1-normalized octahedral xy with
    /// the n.z SIGN folded into a signed roughness in z (roughness floored at
    /// 1.5/512 so the sign bit survives). The 2-bit A channel (materialID/3)
    /// is appended by the caller; we always store materialID = 0.
    pub fn encode_normal_roughness_101010(n: [f32; 3], roughness: f32) -> [f32; 3] {
        let l1 = n[0].abs() + n[1].abs() + n[2].abs();
        let (nx, ny, nz) = (n[0] / l1, n[1] / l1, n[2] / l1);
        let ry = ny * 0.5 + 0.5;
        let rx = nx * 0.5 + ry;
        let ry = ry - nx * 0.5;
        let r = roughness.max(1.5 / 512.0);
        let s = if nz < 0.0 { -r } else { r };
        [rx, ry, s * 0.5 + 0.5]
    }

    /// `_NRD_DecodeNormalRoughness101010` — returns (unnormalized normal,
    /// roughness); callers normalize.
    pub fn decode_normal_roughness_101010(p: [f32; 3]) -> ([f32; 3], f32) {
        let t = p[2] * 2.0 - 1.0;
        let x = p[0] - p[1];
        let y = p[0] + p[1] - 1.0;
        let zsign = if t < 0.0 { -1.0f32 } else { 1.0 };
        let z = zsign * (1.0 - x.abs() - y.abs());
        ([x, y, z], t.abs())
    }

    /// `_NRD_GetSpecMagicCurve(roughness, power)`.
    pub fn spec_magic_curve(roughness: f32, power: f32) -> f32 {
        let f = 1.0 - (-200.0 * roughness * roughness).exp2();
        f * roughness.clamp(0.0, 1.0).powf(power)
    }

    /// `_REBLUR_GetHitDistanceNormalization(viewZ, hitDistParams, roughness)`.
    pub fn hit_dist_normalization(view_z: f32, params: [f32; 3], roughness: f32) -> f32 {
        let smc = spec_magic_curve(roughness, 0.5);
        (params[0] + view_z.abs() * params[1]) * (params[2] + (1.0 - params[2]) * smc)
    }

    /// `REBLUR_FrontEnd_GetNormHitDist` — saturate(hitDist / f), floored at
    /// NRD_EPS ("0 means no data; if this is called the lobe was traced").
    pub fn norm_hit_dist(hit_dist: f32, view_z: f32, params: [f32; 3], roughness: f32) -> f32 {
        let f = hit_dist_normalization(view_z, params, roughness);
        (hit_dist / f).clamp(0.0, 1.0).max(NRD_EPS)
    }

    /// N0: the DLL-free math gates. Pins round-trips and closed-form anchors
    /// so a bridge-kernel port or an upstream formula drift fails loudly.
    pub fn self_test() -> Result<(), String> {
        // YCoCg round trip: exact algebraic inverse for non-negative colors;
        // fp gives ~ulp noise, bound relative to magnitude.
        let colors: [[f32; 3]; 6] = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.25, 0.5, 0.75],
            [44000.0, 12.0, 0.001],
            [0.0, 3.5, 0.0],
            [7.25, 0.0, 19.5],
        ];
        for c in colors {
            let back = ycocg_to_linear(linear_to_ycocg(c));
            // The rotation's cancellation error scales with the LARGEST
            // channel, not each channel's own value (an 0.001 next to 44000
            // is below f32's resolution of their difference) — the spp-gate
            // lesson: bound relative to the vector's own magnitude. The
            // delta-form recompose in cs_nrd_out cancels exactly this error.
            let maxc = c[0].abs().max(c[1].abs()).max(c[2].abs());
            let tol = 1e-5f32.max(maxc * 1e-6);
            for k in 0..3 {
                if (back[k] - c[k]).abs() > tol {
                    return Err(format!("nrd: YCoCg round trip {c:?} -> {back:?}"));
                }
            }
        }
        // A uniform gray is pure luma: Co = Cg = 0 exactly.
        let g = linear_to_ycocg([0.5, 0.5, 0.5]);
        if g[0] != 0.5 || g[1] != 0.0 || g[2] != 0.0 {
            return Err(format!("nrd: YCoCg gray anchor {g:?}"));
        }

        // Normal encode/decode round trip over a sphere sweep: direction
        // recovered to < 0.5 deg, roughness to the 1.5/512 floor's own step,
        // n.z sign preserved bitwise through the signed-roughness channel.
        let mut worst_dot = 1.0f32;
        for i in 0..64 {
            for j in 0..32 {
                let phi = i as f32 / 64.0 * std::f32::consts::TAU;
                let cth = 1.0 - 2.0 * (j as f32 + 0.5) / 32.0;
                let sth = (1.0 - cth * cth).sqrt();
                let n = [sth * phi.cos(), sth * phi.sin(), cth];
                for rough in [0.0f32, 0.05, 0.5, 1.0] {
                    let enc = encode_normal_roughness_101010(n, rough);
                    // The wire is R10G10B10A2 UNORM — quantize like the store.
                    let q = |v: f32| (v.clamp(0.0, 1.0) * 1023.0).round() / 1023.0;
                    let (dn, dr) = decode_normal_roughness_101010([q(enc[0]), q(enc[1]), q(enc[2])]);
                    let len = (dn[0] * dn[0] + dn[1] * dn[1] + dn[2] * dn[2]).sqrt();
                    let dot = (dn[0] * n[0] + dn[1] * n[1] + dn[2] * n[2]) / len;
                    worst_dot = worst_dot.min(dot);
                    if dot < 0.99996 {
                        // ~0.5 deg
                        return Err(format!("nrd: normal round trip {n:?} dot {dot}"));
                    }
                    let expect_r = rough.max(1.5 / 512.0);
                    if (dr - expect_r).abs() > 1.5 / 512.0 {
                        return Err(format!("nrd: roughness round trip {rough} -> {dr}"));
                    }
                    if (dn[2] < 0.0) != (n[2] < 0.0) && n[2].abs() > 1e-3 {
                        return Err(format!("nrd: n.z sign lost for {n:?}"));
                    }
                }
            }
        }

        // Hit-distance normalization anchors (params = ReBLUR defaults).
        let p = [3.0, 0.1, 20.0];
        // roughness 1: smc(1) = (1 - 2^-200) * 1 = 1 ⇒ lerp(C, 1, 1) = 1
        // ⇒ f = (3 + 10*0.1) * 1 = 4 ⇒ hitDist 2 normalizes to 0.5.
        let v = norm_hit_dist(2.0, 10.0, p, 1.0);
        if (v - 0.5).abs() > 1e-5 {
            return Err(format!("nrd: normHitDist anchor 0.5 != {v}"));
        }
        // hitDist 0 ⇒ the NRD_EPS floor exactly (the "lobe was traced" pin).
        let v0 = norm_hit_dist(0.0, 10.0, p, 1.0);
        if v0 != NRD_EPS {
            return Err(format!("nrd: normHitDist zero floor != {v0}"));
        }
        // Saturation: anything past f pins at 1.0.
        let v1 = norm_hit_dist(1e9, 10.0, p, 0.3);
        if v1 != 1.0 {
            return Err(format!("nrd: normHitDist saturation != {v1}"));
        }
        // Monotone in hitDist below saturation.
        let (a, b) = (norm_hit_dist(0.5, 10.0, p, 0.5), norm_hit_dist(1.0, 10.0, p, 0.5));
        if b <= a {
            return Err(format!("nrd: normHitDist not monotone ({a} vs {b})"));
        }
        // Rough-vs-smooth: at equal hitDist a SMOOTH surface normalizes to a
        // SMALLER value (its normalization f is C=20x larger) — the property
        // that makes the guide roughness-adaptive.
        let (sm, rg) = (norm_hit_dist(2.0, 10.0, p, 0.0), norm_hit_dist(2.0, 10.0, p, 1.0));
        if sm >= rg {
            return Err(format!("nrd: normHitDist roughness adaptivity ({sm} vs {rg})"));
        }

        eprintln!(
            "nrd self-test: ycocg + normal-enc2 (worst dot {worst_dot:.6}) + norm-hit-dist OK"
        );
        Ok(())
    }
}
