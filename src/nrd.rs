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
    // --- The three SUB-STRUCTS, which had zero plumbing until 2026-08-10.
    // `antilagSettings` (--nrd-antilag S,K): the luminance-gradient antilag —
    // sigma scale reduces the measured delta by local variance, sensitivity
    // is inverse (smaller = more sensitive).
    pub antilag_sigma_scale: Option<f32>,
    pub antilag_sensitivity: Option<f32>,
    // `responsiveAccumulationSettings` (--nrd-responsive R[,N]): below
    // `roughness_threshold` the history length scales WITH roughness
    // (`maxAccum *= smoothstep(0, 1, rough/threshold)`), which is NVIDIA's
    // stated animated-water lever — this scene's fountain/ocean water is
    // roughness 0.05 with a ripple field that swings the mirror normal every
    // frame, i.e. exactly the case. `min_frames` keeps a floor of history at
    // roughness 0 so a clean signal is not thrown away entirely.
    pub responsive_roughness: Option<f32>,
    pub responsive_min_frames: Option<u32>,
    // `convergenceSettings` (--nrd-convergence S,B,P): ReBLUR drives its
    // denoising by `f = 1/(1 + k·N)` over N accumulated frames, and since
    // v4.17 `k = s·lerp(b, 1, saturate(N/(1 + p·maxAccum)))`. b < 1 means "do
    // MORE denoising on a short history" — deliberately blurrier just after a
    // reset, on the grounds that blurry beats dirty. That is the same shape
    // measured on the FRD arm during the 2026-08-10 darkening campaign (a
    // MOVING frame is flattered by young-history wide blur bleeding light
    // outward, while a parked frame converges onto a genuinely dimmer truth),
    // so this triple is the tuning lever for that symptom on the NRD arm:
    // raising `b` toward 1 narrows the young-history blur and makes the two
    // regimes agree. NVIDIA ship a Desmos sandbox for the curve.
    pub convergence_s: Option<f32>,
    pub convergence_b: Option<f32>,
    pub convergence_p: Option<f32>,
}

impl ReblurTuning {
    pub fn any(&self) -> bool {
        self.max_stabilized_frames.is_some()
            || self.prepass_radius.is_some()
            || self.anti_firefly.is_some()
            || self.max_accum_frames.is_some()
            || self.antilag_sigma_scale.is_some()
            || self.antilag_sensitivity.is_some()
            || self.responsive_roughness.is_some()
            || self.responsive_min_frames.is_some()
            || self.convergence_s.is_some()
            || self.convergence_b.is_some()
            || self.convergence_p.is_some()
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
        if let Some(v) = self.antilag_sigma_scale {
            rs.antilag_settings.luminance_sigma_scale = v;
        }
        if let Some(v) = self.antilag_sensitivity {
            rs.antilag_settings.luminance_sensitivity = v;
        }
        if let Some(v) = self.responsive_roughness {
            rs.responsive_accumulation_settings.roughness_threshold = v;
        }
        if let Some(v) = self.responsive_min_frames {
            rs.responsive_accumulation_settings.min_accumulated_frame_num = v;
        }
        if let Some(v) = self.convergence_s {
            rs.convergence_settings.s = v;
        }
        if let Some(v) = self.convergence_b {
            rs.convergence_settings.b = v;
        }
        if let Some(v) = self.convergence_p {
            rs.convergence_settings.p = v;
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

/// Runtime `CommonSettings` overrides — the ReblurTuning shape one level up,
/// for the fields that belong to the shared settings block rather than to
/// ReBLUR. `split_screen` is NRD's OWN noisy-vs-denoised wipe: at X the left
/// X of the frame presents the RAW input and the rest the denoised output, in
/// one library-side pass with no new resources, which replaces the hand-built
/// two-capture A/B this project keeps rebuilding. All-None = a flagless
/// session sends exactly what it always did.
#[derive(Clone, Copy, Default, Debug)]
pub struct CommonTuning {
    pub split_screen: Option<f32>,
}

impl CommonTuning {
    pub fn any(&self) -> bool {
        self.split_screen.is_some()
    }
}

static COMMON_TUNING: std::sync::OnceLock<CommonTuning> = std::sync::OnceLock::new();

/// One writer: main's lever block (the `set_tuning` contract).
pub fn set_common_tuning(t: CommonTuning) {
    let _ = COMMON_TUNING.set(t);
}

pub fn common_tuning() -> CommonTuning {
    COMMON_TUNING.get().copied().unwrap_or_default()
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

// The DLL half is Windows-only: NRD ships `NRD.dll` with DXIL embedded for
// D3D12. `oracle` below is the portable twin and stays compiled everywhere —
// it is what --check-nrd's DLL-free N0 gate scores, and it must not evaporate
// off Windows just because the loader cannot. (A Vulkan NRD build is a
// different artifact, carrying SPIRV; see SpirvBindingOffsets above.)
#[cfg(windows)]
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
#[cfg(windows)]
pub struct Nrd {
    api: loader::Api,
    instance: *mut Instance,
    pub version: (u8, u8, u8),
}

#[cfg(windows)]
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

#[cfg(windows)]
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

/// N9-CPU: EXACT REMODULATION's identity, as closed-form math (DLL-free, so it
/// runs in `--check` on every platform).
///
/// IT EXISTS BECAUSE THE GPU GATES ARE SCENE-DEPENDENT AND THE CHECK SCENE HAS
/// NONE OF THE MATERIALS INVOLVED. N6b measures the procedural scene's every
/// factor at exactly 1.0 (no sheen, no translucency, no cavity pits, no detail
/// sun-shadow), so the whole feature is INERT there and a green suite proves
/// only that it is harmless. San Miguel exercises it (m_d down to 0.9087 over
/// 1514 pixels) but that is a downloaded scene, not a gate the tree can rely
/// on. This module owes nothing to scene content: it sweeps the factors
/// directly.
///
/// THE MODEL, and it is deliberately a model rather than a re-derivation of the
/// shader. The denoiser is taken to scale the diffuse channel by `l` — the one
/// assumption that makes "the correct answer" well defined at all, since the
/// wire hands the denoiser `A + B` and there is no decomposition to recover.
/// Under it the truth is simply `l * base`, and the question the gate asks is
/// how close each remodulation lands to that.
pub mod remod {
    /// `shade.hlsli::remod_blend` — the ONE formula, mirrored. The energy is
    /// the channel MEAN (a radiometric share, not a perceptual luma) and the
    /// degenerate denominator returns `k_a` rather than dividing.
    pub fn blend(e_a: [f32; 3], k_a: f32, e_b: [f32; 3], k_b: f32) -> f32 {
        let m = |e: [f32; 3]| (e[0].max(0.0) + e[1].max(0.0) + e[2].max(0.0)) / 3.0;
        let (a, b) = (m(e_a), m(e_b));
        let s = a + b;
        if s > 0.0 { (a * k_a + b * k_b) / s } else { k_a }
    }

    /// A primary hit's material, in shade's own vocabulary.
    #[derive(Clone, Copy)]
    pub struct Hit {
        pub albedo: [f32; 3],
        pub metallic: f32,
        pub trans: f32,
        pub sheen: f32,
        /// Translucency: splits the direct diffuse front/back.
        pub tl: f32,
        /// `detail_cavity(dh)` — applied to the ambient/bounce sub-term.
        pub dcav: f32,
        /// `detail_sun_shadow(..)` — applied to the direct diffuse, INSIDE the
        /// captured signal (it is not a delta multiplier).
        pub dsun: f32,
    }

    impl Hit {
        /// The wire diffuse albedo the bridge remodulates at.
        pub fn wire_kd(&self) -> [f32; 3] {
            let k = (1.0 - self.metallic) * (1.0 - self.trans);
            [self.albedo[0] * k, self.albedo[1] * k, self.albedo[2] * k]
        }
        /// The Charlie energy term shade folds into its own `kd`.
        pub fn sk(&self) -> f32 {
            1.0 - 0.157 * self.sheen
        }
    }

    fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }
    fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn v_mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
    }
    fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
        [a[0] * s, a[1] * s, a[2] * s]
    }
    /// The f16 the sig lanes really are — so "exact" below means exact to the
    /// wire's own quantum and the tolerances are that quantum, not a fudge.
    fn q16(a: [f32; 3]) -> [f32; 3] {
        [
            half::f16::from_f32(a[0]).to_f32(),
            half::f16::from_f32(a[1]).to_f32(),
            half::f16::from_f32(a[2]).to_f32(),
        ]
    }

    /// What the pack captures and what shade's accum actually received.
    /// `dd_pre` is `direct_d` BEFORE `detail_sun_shadow`, `dt` the
    /// translucency back-ray term, `amb` the ambient (or RTGI bounce)
    /// sub-term BEFORE the cavity.
    pub struct Split {
        /// shade's own diffuse contribution to accum.
        pub base: [f32; 3],
        /// The bridge's `D_in` = the wire dd lane + the reconstructed ambient.
        pub d_in: [f32; 3],
        /// The delta multiplier riding sig.w's high half (1.0 pre-fix).
        pub m_d: f32,
    }

    /// `exact` selects the arm: true = FLAG_REMOD_EXACT's re-capture, false =
    /// the pre-2026-08-10 capture (dd taken before dsun and before the
    /// translucency split, remodulated at a bare kd).
    pub fn split(h: &Hit, dd_pre: [f32; 3], dt: [f32; 3], amb: [f32; 3], exact: bool) -> Split {
        let (sk, kd) = (h.sk(), h.wire_kd());
        let dd_post = v_scale(dd_pre, h.dsun);
        let diffuse_d = v_add(v_scale(dd_post, 1.0 - h.tl), v_scale(dt, h.tl));
        // shade: kd_local * kt * (diffuse_d + amb*dcav), and kd_local*kt is
        // exactly wire_kd*sk.
        let base = v_mul(v_scale(kd, sk), v_add(diffuse_d, v_scale(amb, h.dcav)));
        if exact {
            let m_d = blend(diffuse_d, sk, amb, sk * h.dcav);
            Split { base, d_in: v_add(q16(diffuse_d), amb), m_d }
        } else {
            Split { base, d_in: v_add(q16(dd_pre), amb), m_d: 1.0 }
        }
    }

    /// The bridge's delta-form recompose (`nrd_bridge.hlsl`).
    pub fn compose(base: [f32; 3], kd: [f32; 3], d_in: [f32; 3], d_out: [f32; 3], m_d: f32) -> [f32; 3] {
        v_add(base, v_scale(v_mul(v_sub(d_out, d_in), kd), m_d))
    }

    /// The model truth: the denoiser scaled the diffuse channel by `l`, so
    /// both sub-terms scale and shade's own composition scales with them.
    pub fn truth(base: [f32; 3], l: f32) -> [f32; 3] {
        v_scale(base, l)
    }

    /// One row of the sweep: the relative error each arm lands at.
    pub fn score(h: &Hit, dd_pre: [f32; 3], dt: [f32; 3], amb: [f32; 3], l: f32) -> (f32, f32) {
        let kd = h.wire_kd();
        let err = |exact: bool| {
            let s = split(h, dd_pre, dt, amb, exact);
            let out = v_scale(s.d_in, l);
            let got = compose(s.base, kd, s.d_in, out, s.m_d);
            let want = truth(s.base, l);
            let (mut e, mut mag) = (0.0f32, 0.0f32);
            for c in 0..3 {
                e = e.max((got[c] - want[c]).abs());
                mag = mag.max(want[c].abs());
            }
            e / mag.max(1e-6)
        };
        (err(true), err(false))
    }

    /// N9-CPU. Sweeps sheen x tl x dcav x dsun x trans, achromatic and
    /// chromatic, and asserts BOTH directions: the exact arm holds the
    /// identity, and the pre-fix arm PROVABLY FAILS it wherever any factor
    /// departs from 1 — the anti-vacuity half, without which the gate passes
    /// on a feature that never ran.
    pub fn self_test() -> Result<(), String> {
        // The blend's own contracts first (it is the piece with no GPU twin
        // the gates can reach).
        let one = [1.0f32; 3];
        if blend(one, 0.5, [0.0; 3], 0.1) != 0.5 {
            return Err("remod: blend with zero b-energy != k_a".into());
        }
        if blend([0.0; 3], 0.5, [0.0; 3], 0.1) != 0.5 {
            return Err("remod: degenerate blend != k_a".into());
        }
        if blend(one, 0.5, one, 0.5) != 0.5 {
            return Err("remod: blend of equal factors is not that factor".into());
        }
        let mid = blend(one, 1.0, one, 0.4);
        if (mid - 0.7).abs() > 1e-6 {
            return Err(format!("remod: equal-energy blend 0.7 != {mid}"));
        }
        // CONVEXITY — the property that makes m_d an attenuation and never an
        // amplifier, swept rather than argued.
        for i in 0..17 {
            let e = [i as f32 * 0.25; 3];
            let m = blend(one, 0.9, e, 0.3);
            if !(0.3 - 1e-6..=0.9 + 1e-6).contains(&m) {
                return Err(format!("remod: blend {m} escaped [0.3, 0.9]"));
            }
        }

        let (mut worst_exact, mut worst_pre, mut rows, mut teeth) = (0.0f32, 0.0f32, 0, 0);
        let mut min_pre_on_live = f32::INFINITY;
        for &sheen in &[0.0f32, 0.5, 1.0] {
            for &tl in &[0.0f32, 0.5, 1.0] {
                for &dcav in &[1.0f32, 0.5, 0.1] {
                    for &dsun in &[1.0f32, 0.3] {
                        for &trans in &[0.0f32, 0.97] {
                            for &l in &[0.5f32, 1.0, 2.0, 8.0] {
                                let h = Hit {
                                    albedo: [0.8, 0.6, 0.4],
                                    metallic: 0.0,
                                    trans,
                                    sheen,
                                    tl,
                                    dcav,
                                    dsun,
                                };
                                // ACHROMATIC sub-terms: the scalar blend is
                                // exact only when the two sub-terms share a
                                // hue (otherwise the per-channel factor is
                                // channel-dependent and one lane cannot carry
                                // it). This row is the exactness claim.
                                let (e_ex, e_pre) =
                                    score(&h, [0.4; 3], [0.25; 3], [0.3; 3], l);
                                rows += 1;
                                worst_exact = worst_exact.max(e_ex);
                                worst_pre = worst_pre.max(e_pre);
                                // A row is "live" when some factor departs
                                // from 1 AND the denoiser actually moved
                                // (l == 1 leaves every arm exact by the delta
                                // form's own construction — the pre-fix bug is
                                // a bug in the CORRECTION, not in `base`).
                                let live = (sheen > 0.0 || tl > 0.0 || dcav < 1.0 || dsun < 1.0)
                                    && (l - 1.0).abs() > 1e-6;
                                if live {
                                    teeth += 1;
                                    min_pre_on_live = min_pre_on_live.min(e_pre);
                                }
                            }
                        }
                    }
                }
            }
        }
        if worst_exact > 1.0 / 2048.0 {
            return Err(format!(
                "remod: the exact arm missed the identity by {worst_exact:.2e} (> one f16 \
                 capture quantum) over {rows} rows"
            ));
        }
        // THE TEETH. Without this the whole sweep passes on a build where
        // FLAG_REMOD_EXACT does nothing at all.
        if teeth == 0 {
            return Err("remod: the sweep produced no live rows (vacuous)".into());
        }
        // THE ANCHOR, not a threshold. The sweep's MILDEST live row is
        // sheen-only (tl = dcav = dsun = 1) at the smallest departure of `l`,
        // and its pre-fix error has a closed form: the correction enters at
        // kd instead of kd*sk, so
        //     err = (l-1)*(1-sk) / (l*sk)
        // which at sheen 0.5 (sk = 0.9215) and l = 2 is 4.26e-2. Asserting
        // that identity is worth more than asserting "> 5%": it pins WHY the
        // pre-fix arm is wrong and by exactly how much, and it cannot be
        // satisfied by a sweep that merely produces large numbers.
        let sk_a = 1.0 - 0.157 * 0.5f32;
        let l_a = 2.0f32;
        let want_a = (l_a - 1.0) * (1.0 - sk_a) / (l_a * sk_a);
        let anchor = Hit {
            albedo: [0.8, 0.6, 0.4],
            metallic: 0.0,
            trans: 0.0,
            sheen: 0.5,
            tl: 0.0,
            dcav: 1.0,
            dsun: 1.0,
        };
        let (a_ex, a_pre) = score(&anchor, [0.4; 3], [0.25; 3], [0.3; 3], l_a);
        if (a_pre - want_a).abs() > 1e-3 {
            return Err(format!(
                "remod: the sheen-only pre-fix anchor {a_pre:.4e} != the closed form \
                 {want_a:.4e} — the model and the arithmetic disagree"
            ));
        }
        if a_ex > 1.0 / 2048.0 {
            return Err(format!("remod: the exact arm missed the anchor by {a_ex:.2e}"));
        }
        // ...and no live row may be MILDER than that anchor, which is what
        // stops the sweep from silently losing its hardest cases (a factor
        // list edited down to only near-unity values would pass every other
        // assertion here).
        if min_pre_on_live < want_a * 0.99 {
            return Err(format!(
                "remod: a live row's PRE-FIX error {min_pre_on_live:.2e} fell below the \
                 sheen-only anchor {want_a:.2e} — the sweep stopped exercising the defect"
            ));
        }

        // CHROMATIC sub-terms: the scalar lane cannot be exact, so the claim
        // is the BOUND (m_d stays inside the two factors) plus a strict
        // aggregate improvement over the pre-fix arm.
        let (mut sum_ex, mut sum_pre, mut n) = (0.0f64, 0.0f64, 0usize);
        for &sheen in &[0.0f32, 0.6] {
            for &dcav in &[1.0f32, 0.2] {
                for &l in &[0.5f32, 4.0] {
                    let h = Hit {
                        albedo: [0.7, 0.7, 0.7],
                        metallic: 0.0,
                        trans: 0.0,
                        sheen,
                        tl: 0.0,
                        dcav,
                        dsun: 1.0,
                    };
                    // Deliberately opposite hues, the worst case for a scalar.
                    let (dd, amb) = ([0.9, 0.05, 0.0], [0.0, 0.05, 0.9]);
                    let s = split(&h, dd, [0.0; 3], amb, true);
                    let (lo, hi) = (h.sk() * h.dcav, h.sk());
                    if !(lo - 1e-6..=hi + 1e-6).contains(&s.m_d) {
                        return Err(format!(
                            "remod: chromatic m_d {} escaped [{lo}, {hi}]",
                            s.m_d
                        ));
                    }
                    let (e_ex, e_pre) = score(&h, dd, [0.0; 3], amb, l);
                    sum_ex += e_ex as f64;
                    sum_pre += e_pre as f64;
                    n += 1;
                }
            }
        }
        if sum_ex >= sum_pre {
            return Err(format!(
                "remod: on chromatic rows the exact arm is no better than pre-fix \
                 ({:.3e} vs {:.3e})",
                sum_ex / n as f64,
                sum_pre / n as f64
            ));
        }

        eprintln!(
            "remod self-test: {rows} rows, exact worst {worst_exact:.2e}, pre-fix worst \
             {worst_pre:.2e} (teeth {teeth} live rows, min pre-fix err \
             {min_pre_on_live:.2e}), chromatic mean {:.2e} vs {:.2e} OK",
            sum_ex / n as f64,
            sum_pre / n as f64
        );
        Ok(())
    }
}

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

    // ---- THE RESIDUAL SPIKE CAP (FLAG_NRD_RCLAMP) -------------------------
    //
    // CPU twins of `nrd_bridge.hlsl`'s clamp — the NRD_HITDIST_PARAMS
    // precedent, and here they carry a second job: N10 scores the real shader
    // against `rclamp_scale`, so the K values, the centre exclusion, the
    // out-of-range exclusion and the min-ring floor are pinned across shader
    // and CPU AT ONCE rather than by the shader agreeing with itself.

    /// Ring-mean multiplier — FRD's `FIREFLY_K`, deliberately the same number
    /// so the two clamps in this tree speak one value. Decoupling them means
    /// two named constants with two doc comments, never one with a caveat.
    pub const RCLAMP_K_MEAN: f32 = 8.0;
    /// Ring-MAX multiplier, and the term that makes the cap safe: a lone
    /// one-sample spike has mean ~ max so the mean term binds, while a
    /// RESOLVED feature (a bright emissive texel, a caustic edge) has at least
    /// one bright neighbour, so max >> mean and this term protects it.
    pub const RCLAMP_K_MAX: f32 = 3.0;
    /// The `hard` diagnostic arm's multiplier, applied to BOTH terms: `cap` is
    /// a max and `max >= mean` always, so relaxing only the mean one leaves
    /// `K_MAX * ring_max` binding and the "hard" arm is barely harder than the
    /// default (measured: it gated nothing). At 1.0 on both, `cap == ring_max`
    /// and the test becomes "is this pixel a strict local maximum".
    pub const RCLAMP_K_HARD: f32 = 1.0;
    /// A 3-neighbour estimate is not a neighbourhood.
    pub const RCLAMP_MIN_RING: usize = 4;

    /// The bridge's own luma — `nrd_ycocg(c).x`, the luma the DENOISER reasons
    /// in, deliberately not a Rec.709 perceptual weight.
    pub fn rclamp_luma(c: [f32; 3]) -> f32 {
        0.25 * c[0] + 0.5 * c[1] + 0.25 * c[2]
    }

    /// The scale applied to the RESIDUAL, given this pixel's BASE luma and its
    /// 8-neighbour ring of base lumas (`None` = out of range / excluded).
    /// Returns 1.0 when the cap does not fire.
    ///
    /// THE TEST IS ON BASE AND THE CORRECTION LANDS ON THE RESIDUAL, which
    /// reads like a units error and is not. Model a neighbour's residual as its
    /// base times this pixel's residual FRACTION `f = rl/bl`; then
    /// `rl > K*mean(ring_R)` reduces exactly to `bl > K*mean(ring_base)` — the
    /// `f` cancels — so the outlier test is a statement about base, which we
    /// have a ring for, while the scale applies to the residual, the only part
    /// the denoiser did not clean. Everything stays in one unit, and no
    /// neighbour residuals (hence no neighbour `gbuf_ext` reads, hence N7's
    /// unwritten-sky-ext contract) are needed.
    pub fn rclamp_scale(base_luma: f32, ring: &[Option<f32>], hard: bool) -> f32 {
        let (mut sum, mut mx, mut n) = (0.0f32, 0.0f32, 0usize);
        for v in ring.iter().flatten() {
            sum += *v;
            mx = mx.max(*v);
            n += 1;
        }
        if n < RCLAMP_MIN_RING || !(base_luma > 0.0) {
            return 1.0;
        }
        let (km, kx) = if hard {
            (RCLAMP_K_HARD, RCLAMP_K_HARD)
        } else {
            (RCLAMP_K_MEAN, RCLAMP_K_MAX)
        };
        let cap = (km * (sum / n as f32)).max(kx * mx);
        if base_luma > cap { cap / base_luma } else { 1.0 }
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

        // ---- the residual spike cap's closed forms (the N10 twin) ---------
        let flat = [Some(1.0f32); 8];
        // A pixel level with its ring never fires: cap = max(8, 3) * 1 = 8.
        if rclamp_scale(1.0, &flat, false) != 1.0 {
            return Err("nrd: rclamp fired on a flat neighbourhood".into());
        }
        // A lone 100x spike does: ring mean = max = 1, cap = 8, scale = 0.08.
        let s = rclamp_scale(100.0, &flat, false);
        if (s - 0.08).abs() > 1e-6 {
            return Err(format!("nrd: rclamp lone-spike scale 0.08 != {s}"));
        }
        // THE K_MAX TEETH — a RESOLVED feature must survive. One bright
        // neighbour at 40 puts ring mean at ~5.9 (cap 47.1 via K_MEAN) but
        // K_MAX*max = 120, so the max term binds and a 100 centre passes.
        let mut resolved = [Some(1.0f32); 8];
        resolved[0] = Some(40.0);
        if rclamp_scale(100.0, &resolved, false) != 1.0 {
            return Err("nrd: rclamp clamped a resolved feature (K_MAX inert?)".into());
        }
        // ...and the same centre against the same ring DOES clamp under the
        // hard arm, where cap == ring_max. Without this the two arms could
        // share a multiplier and nothing would notice.
        let sh = rclamp_scale(100.0, &resolved, true);
        if (sh - 0.4).abs() > 1e-6 {
            return Err(format!("nrd: rclamp hard scale 0.4 != {sh}"));
        }
        // The hard arm is exactly "strict local maximum": a centre at the
        // ring's own max must NOT fire (cap == max == bl).
        if rclamp_scale(40.0, &resolved, true) != 1.0 {
            return Err("nrd: rclamp hard fired on a non-maximum".into());
        }
        // Out-of-range neighbours are EXCLUDED, not zeroed — zeroing would
        // drag the mean down and clamp along every horizon. With 5 of 8
        // excluded the surviving 3 still set the cap, unchanged.
        let mut sparse = [None; 8];
        for e in sparse.iter_mut().take(4) {
            *e = Some(1.0);
        }
        if rclamp_scale(100.0, &sparse, false) != 0.08 {
            return Err("nrd: rclamp excluded neighbours changed the mean".into());
        }
        // ...and below the min-ring floor it stands down entirely.
        let mut thin = [None; 8];
        for e in thin.iter_mut().take(RCLAMP_MIN_RING - 1) {
            *e = Some(1.0);
        }
        if rclamp_scale(100.0, &thin, false) != 1.0 {
            return Err("nrd: rclamp fired below the min-ring floor".into());
        }
        // Only ever an attenuation, at every input.
        for bl in [0.0f32, 1e-6, 0.5, 7.9, 8.1, 1e6] {
            let sc = rclamp_scale(bl, &flat, false);
            if !(0.0..=1.0).contains(&sc) {
                return Err(format!("nrd: rclamp scale {sc} out of (0,1] at bl {bl}"));
            }
        }

        eprintln!(
            "nrd self-test: ycocg + normal-enc2 (worst dot {worst_dot:.6}) + norm-hit-dist \
             + rclamp OK"
        );
        Ok(())
    }
}
