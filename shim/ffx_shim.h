// Flat C API over the AMD FidelityFX SDK (ffx-api) for Rust FFI.
//
// Why a shim (vs XeSS-style hand-transcribed structs): the ffx-api surface is
// pNext-chained descriptors plus FfxApiResource with an embedded description
// struct that ffxApiGetResourceDX12 derives from ID3D12Resource::GetDesc().
// One C++ TU compiled against the vendored MIT headers absorbs the chaining,
// the description derivation, and any header drift; Rust only ever sees the
// flat structs below (mirrored in gpu/ffx_sys.rs).
//
// The shim dynamically loads amd_fidelityfx_loader_dx12.dll (LoadLibraryExW
// with DLL_LOAD_DIR search so the loader finds the provider DLLs next to
// itself) — nothing links against amd_fidelityfx_loader_dx12.lib, so the
// executable starts with no FFX DLLs present and headless runs (--check*)
// never touch FFX.
//
// All functions return the raw ffxReturnCode_t as int32_t (0 == OK), except
// the shim-private negatives below. Handles are opaque void*. Matrices are
// float[16] in FfxApiMatrix4x4's own layout (row-major storage, row-vector
// convention) — per the header's convention table, glam's column-major /
// column-vector matrices memcpy into this directly, NO transpose (deliberate
// contrast with the SL shim, whose boundary transposes via gpu::row_major).

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Shim-private error codes (ffxReturnCode_t is non-negative).
#define FFXSHIM_ERR_LOAD_LIBRARY   -1000  // LoadLibraryExW failed
#define FFXSHIM_ERR_GET_PROC       -1001  // an ffx* export is missing
#define FFXSHIM_ERR_NOT_LOADED     -1002  // call before ffxshim_load
#define FFXSHIM_ERR_BAD_ARG        -1003

typedef void (*FfxShimLogCb)(uint32_t type /*FfxApiMsgType*/, const wchar_t* msg);

// ---- lifecycle ----
int32_t ffxshim_load(const wchar_t* loader_dll_path);
void    ffxshim_unload(void);
// Global debug message routing for one effect id (loader-level configure,
// null context). Call after load; harmless if the provider ignores it.
int32_t ffxshim_set_debug(uint64_t effect_id, FfxShimLogCb cb, uint32_t level);

// ---- support probe / version enumeration ----
// is_upscaler selects the create-desc type (denoiser vs upscale) whose
// providers are enumerated for `device`. In/out count protocol matches
// ffxQueryDescGetVersions: pass *inout_count = capacity of ids/names (or 0 to
// just get the count back). Zero returned count == effect unsupported here.
int32_t ffxshim_query_versions(int32_t is_upscaler, void* device,
                               uint64_t* inout_count,
                               uint64_t* ids /*may be null*/,
                               const char** names /*may be null*/);

// ---- context creation ----
// signal_flags: FfxApiDenoiserSignalFlags combination (shim re-exports the
// two we use below). version_id 0 = provider default, else chained
// ffxOverrideVersion. max_w/max_h: ffxCreateContextDescDenoiser.maxRenderSize
// (dynamic resolution: dispatches name any renderSize <= this).
#define FFXSHIM_SIGNAL_DIRECT_DIFFUSE   (1u << 1)
#define FFXSHIM_SIGNAL_DIRECT_SPECULAR  (1u << 2)
int32_t ffxshim_create_denoiser(void* device, uint32_t max_w, uint32_t max_h,
                                uint32_t signal_flags, uint32_t flags,
                                uint64_t version_id, void** ctx_out);

// flags: FfxApiCreateContextUpscaleFlags. The shim always chains the
// ffxCreateContextDescUpscaleVersion (API version pin) itself.
#define FFXSHIM_UPSCALE_HDR                 (1u << 0)
#define FFXSHIM_UPSCALE_DEPTH_INVERTED      (1u << 3)
#define FFXSHIM_UPSCALE_DYNAMIC_RESOLUTION  (1u << 6)
#define FFXSHIM_UPSCALE_DEBUG_CHECKING      (1u << 7)
int32_t ffxshim_create_upscaler(void* device, uint32_t max_render_w, uint32_t max_render_h,
                                uint32_t out_w, uint32_t out_h, uint32_t flags,
                                uint64_t version_id, void** ctx_out);

// Destroys either context kind; *ctx is nulled on success.
int32_t ffxshim_destroy(void** ctx);

// ---- queries / configure ----
// FfxApiUpscaleQualityMode -> render resolution for a display size, asked of
// a live upscaler context (Rust falls back to the documented fixed ratios if
// this fails).
int32_t ffxshim_upscaler_render_res(void* upscaler_ctx, uint32_t display_w, uint32_t display_h,
                                    uint32_t quality_mode, uint32_t* out_rw, uint32_t* out_rh);
// ffxConfigureDescDenoiserKeyValue escape hatch (count/data per the header).
int32_t ffxshim_denoiser_kv(void* denoiser_ctx, uint64_t key, uint64_t count, const void* data);

// ---- per-frame dispatches ----
// A D3D12 resource reference: the FfxApiResourceDescription is derived inside
// the shim from GetDesc() (planes are allocated at range max; the dispatch's
// render_w/render_h names the active top-left sub-rect).
typedef struct FfxShimRes {
    void*    resource; // ID3D12Resource*; null = absent (zero-init resource)
    uint32_t state;    // FfxApiResourceState the resource is in at dispatch
} FfxShimRes;

// One Ray Regeneration dispatch: common inputs + the two direct signals,
// chained internally as ffxDispatchDescDenoiser -> DirectDiffuse ->
// DirectSpecular. Conventions (from ffx_denoiser.h):
//   linear_depth    R:  signed linear view-space Z
//   motion_vectors  RG: PreviousUV - CurrentUV, B: prevZ - curZ (mv_scale {1,1,1})
//   normals         RG: octahedral normal, B: linear roughness, A: material type
//   albedos         RGB sqrt-encoded unless non_gamma_albedo
//   jitter          screen pixels; cam_pos_delta = prev - cur (world)
//   view/projection FfxApiMatrix4x4 layout (see file header note on glam)
typedef struct FfxShimDenoiseDesc {
    void* cmdlist;                    // ID3D12GraphicsCommandList*
    FfxShimRes linear_depth;
    FfxShimRes motion_vectors;
    FfxShimRes normals;
    FfxShimRes specular_albedo;
    FfxShimRes diffuse_albedo;
    FfxShimRes dd_in, dd_out;         // direct diffuse signal
    FfxShimRes ds_in, ds_out;         // direct specular signal
    float mv_scale[3];
    float jitter[2];
    float cam_pos_delta[3];
    float view[16];
    float projection[16];
    float depth_bounds_min, depth_bounds_max; // absolute linear view-Z band
    uint32_t render_w, render_h;      // dynamic render size this frame
    uint32_t frame_index;
    int32_t reset;                    // FFX_DENOISER_DISPATCH_RESET
    int32_t non_gamma_albedo;         // FFX_DENOISER_DISPATCH_NON_GAMMA_ALBEDO
} FfxShimDenoiseDesc;
int32_t ffxshim_denoise(void* denoiser_ctx, const FfxShimDenoiseDesc* d);

// One FSR4 upscale dispatch (ffxDispatchDescUpscale). renderSize is the
// dynamic input sub-rect; output is always the full out_w x out_h target.
typedef struct FfxShimUpscaleDesc {
    void* cmdlist;
    FfxShimRes color;                 // render-res composite (HDR linear)
    FfxShimRes depth;                 // 32-bit clip depth at render res
    FfxShimRes motion_vectors;        // shared RGBA16F plane; RG consumed
    FfxShimRes output;                // presentation-res UAV
    float jitter[2];
    float mv_scale[2];
    uint32_t render_w, render_h;
    uint32_t out_w, out_h;
    int32_t enable_sharpening;
    float sharpness;
    float frame_time_delta_ms;
    float pre_exposure;               // must be > 0
    int32_t reset;
    float cam_near, cam_far;
    float cam_fovy;                   // vertical, radians
    float view_space_to_meters;
    uint32_t flags;                   // FfxApiDispatchFsrUpscaleFlags
} FfxShimUpscaleDesc;
int32_t ffxshim_upscale(void* upscaler_ctx, const FfxShimUpscaleDesc* d);

#ifdef __cplusplus
} // extern "C"
#endif
