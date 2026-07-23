// Flat C API over the Streamline SDK for Rust FFI.
//
// Why a shim: SL structs are versioned BaseStructures with GUIDs and C++
// constructors (Constants v2, DLSSDOptions v3), FrameToken carries a vtable,
// and ViewportHandle is itself a BaseStructure. Hand-mirroring those layouts
// in #[repr(C)] Rust would corrupt silently on any SDK update; including the
// real headers here makes layout/GUIDs/versions automatic.
//
// The shim dynamically loads sl.interposer.dll (LoadLibraryW by absolute
// path) — nothing links against sl.interposer.lib, so the executable starts
// with no Streamline DLLs present and headless runs never touch SL.
//
// All functions return the raw sl::Result as int32_t (0 == eOk). Handles are
// opaque void*. All matrices are row-major float[16] (the Rust side
// transposes from glam's column-major at this boundary).

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*SlShimLogCb)(uint32_t type /*sl::LogType*/, const char* msg);

typedef struct SlShimInitDesc {
    const wchar_t* interposer_path; // absolute path to sl.interposer.dll
    const wchar_t* plugins_path;    // Preferences::pathsToPlugins (bin/x64)
    uint32_t app_id;                // NGX application id (dev placeholder ok)
    int32_t show_console;           // bool
    uint32_t log_level;             // sl::LogLevel (0 off, 1 default, 2 verbose)
    SlShimLogCb log_cb;             // may be null
    const uint32_t* features;       // sl::Feature ids to load
    uint32_t num_features;
} SlShimInitDesc;

// ---- lifecycle ----
int32_t slshim_load_and_init(const SlShimInitDesc* desc);
int32_t slshim_shutdown(void);

// ---- support / device / proxies ----
// luid: the DXGI adapter LUID bytes (8 bytes); may be null for a global check.
int32_t slshim_is_feature_supported(uint32_t feature, const void* luid, uint32_t luid_size);
int32_t slshim_set_d3d_device(void* native_d3d12_device);
int32_t slshim_upgrade_interface(void** iface_inout);
int32_t slshim_get_native_interface(void* proxy, void** native_out);

// ---- DLSS-RR options / settings ----
typedef struct SlShimDlssdOptions {
    uint32_t mode;      // sl::DLSSMode (0=Off ... 6=DLAA)
    uint32_t output_width;
    uint32_t output_height;
    uint32_t preset;    // sl::DLSSDPreset applied to ALL quality modes
    int32_t normal_roughness_packed; // 1 => ePacked (roughness in normal.w)
    int32_t use_camera_matrices;     // 1 => matrices below are valid
    float world_to_view[16];         // row-major
    float view_to_world[16];         // row-major
} SlShimDlssdOptions;

typedef struct SlShimDlssdOptimal {
    uint32_t render_width, render_height;
    float sharpness;
    uint32_t render_width_min, render_height_min;
    uint32_t render_width_max, render_height_max;
} SlShimDlssdOptimal;

int32_t slshim_dlssd_get_optimal_settings(const SlShimDlssdOptions* o, SlShimDlssdOptimal* out);
int32_t slshim_dlssd_set_options(uint32_t viewport, const SlShimDlssdOptions* o);
int32_t slshim_dlssd_get_state(uint32_t viewport, uint64_t* vram_bytes_out);
int32_t slshim_allocate_resources(void* cmdlist, uint32_t feature, uint32_t viewport);
int32_t slshim_free_resources(uint32_t feature, uint32_t viewport);

// ---- per frame ----
// frame_index may be null (SL internal counting). Token is owned by SL.
int32_t slshim_new_frame_token(const uint32_t* frame_index, void** token_out);

typedef struct SlShimConstants {
    float view_to_clip[16];      // row-major, jitter-free
    float clip_to_view[16];
    float clip_to_prev_clip[16];
    float prev_clip_to_clip[16];
    float jitter[2];             // pixel-space jitter offset
    float mvec_scale[2];         // normalizes MV values to [-1,1]
    float cam_pos[3];
    float cam_up[3];
    float cam_right[3];
    float cam_fwd[3];
    float cam_near, cam_far;
    float cam_fov;               // vertical, radians
    float cam_aspect;
    int32_t depth_inverted;      // sl::Boolean (0/1)
    int32_t camera_motion_included;
    int32_t mvec_3d;
    int32_t reset;
    int32_t ortho;
    int32_t mvec_jittered;
} SlShimConstants;

int32_t slshim_set_constants(void* token, uint32_t viewport, const SlShimConstants* c);

typedef struct SlShimResourceTag {
    void* resource;       // ID3D12Resource*
    uint32_t state;       // D3D12_RESOURCE_STATES at the time SL uses it
    uint32_t buffer_type; // sl::kBufferType*
    uint32_t lifecycle;   // sl::ResourceLifecycle
    // Active sub-rect of the resource (sl::Extent) — dynamic resolution.
    // All-zero means "whole resource" (SL's own null-extent convention),
    // preserving the pre-DRS behavior.
    uint32_t extent_top;
    uint32_t extent_left;
    uint32_t extent_width;
    uint32_t extent_height;
} SlShimResourceTag;

#define SLSHIM_MAX_TAGS 16
int32_t slshim_tag_resources(void* token, uint32_t viewport, const SlShimResourceTag* tags,
                             uint32_t n, void* cmdlist);

int32_t slshim_evaluate(uint32_t feature, void* token, uint32_t viewport, void* cmdlist);

// ---- DLSS-G frame generation + Reflex/PCL (its hard prerequisites) ----
//
// DLSS-G interpolates the presented backbuffer through the SL proxy swapchain
// the session already presents on; it consumes the SAME depth/mvec tags and
// per-frame constants the RR path sets. Reflex sleep + the PCL present
// markers are REQUIRED at runtime (DLSSGStatus::eFailReflexNotDetectedAt-
// Runtime otherwise), and the marker token must carry the SAME frame index
// slSetConstants used.

typedef struct SlShimDlssgOptions {
    uint32_t mode;                   // sl::DLSSGMode: 0 off, 1 on, 2 auto
    uint32_t num_frames_to_generate; // 1 = 2x
    uint32_t flags;                  // sl::DLSSGFlags
} SlShimDlssgOptions;
int32_t slshim_dlssg_set_options(uint32_t viewport, const SlShimDlssgOptions* o);

typedef struct SlShimDlssgState {
    uint32_t status;              // sl::DLSSGStatus bits; 0 == eOk
    uint32_t min_width_or_height; // minimum supported backbuffer dimension
    uint32_t frames_presented;    // presents since the previous get_state
    uint32_t frames_max;          // numFramesToGenerateMax (1 = 2x ceiling)
} SlShimDlssgState;
int32_t slshim_dlssg_get_state(uint32_t viewport, SlShimDlssgState* out);

// mode: sl::ReflexMode (0 off, 1 low latency, 2 low latency + boost). Must be
// called at least once (even at eOff) for DLSS-G; sleep every frame.
int32_t slshim_reflex_set_options(uint32_t mode);
int32_t slshim_reflex_sleep(void* token);
// marker: sl::PCLMarker ordinal (0 simStart, 1 simEnd, 2 renderSubmitStart,
// 3 renderSubmitEnd, 4 presentStart, 5 presentEnd).
int32_t slshim_pcl_set_marker(uint32_t marker, void* token);

#ifdef __cplusplus
} // extern "C"
#endif
