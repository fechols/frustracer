// DLSS Ray Reconstruction (DLSSD) via RAW NGX (no Streamline) — flat C API
// for Rust FFI.
//
// Why this exists: Streamline earned its keep as the DLSS-RR evaluator only —
// its DLSS-G present layer declines to insert (the CLAUDE.md --fg open-issue
// record) while the raw-NGX FG path (dlssg_shim) is verified generating. This
// shim moves RR onto the same raw path, which retires the SL interposer, the
// proxy device/queue/swapchain split, the slInit-before-DXGI ordering
// constraint, and the shared-in-process-NGX-state trap class outright: with
// SL gone this process has exactly one NGX client family, refcounted through
// ngx_shared.
//
// TWO-HANDLE API: a SESSION (shared init + the NGX-owned capability-parameter
// map + the availability probe) and a FEATURE (created at fixed allocation
// dims; per-frame subrects carry the real render res — DRS without recreate).
// GpuContext holds one session + one feature; the cinematic capture holds one
// session + a feature per shot resolution.
//
// BUILD-OPTIONAL: same gate as dlssg_shim — build.rs compiles this TU only
// when FRUSTRACER_DLSS_SDK points at a DLSS SDK carrying the DLSSD headers,
// and emits cfg(dlss_ngx); without it the Rust side stubs to "unavailable"
// (no DLSS at all — the chain falls to FSR4/XeSS/FSR3).
//
// All functions return 0 on success; negative shim-private codes below
// (same values as dlssg_shim's FRDLSSG_* family).

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FRDLSSD_OK               0
#define FRDLSSD_ERR_INTERNAL    -1
#define FRDLSSD_ERR_INIT        -2
#define FRDLSSD_ERR_UNSUPPORTED -3  // no RR support / old driver — fall through the chain
#define FRDLSSD_ERR_D3D12       -4
#define FRDLSSD_ERR_DISPATCH    -5

// Refcounted shared NGX init + GetCapabilityParameters + the
// SuperSamplingDenoising.Available probe. UNSUPPORTED = the chain's level-1
// probe failing — fall through, never an error.
int32_t frdlssd_open(void* device, void** out_session);

// The DLSSD optimal-settings query at MaxQuality for a target resolution:
// the (optimal, min, max) render-res triple the DRS controller consumes
// (the slDLSSDGetOptimalSettings replacement — same degenerate-collapse
// handling stays on the Rust side).
typedef struct FrDlssdOptimal {
    uint32_t opt_w, opt_h;
    uint32_t min_w, min_h;
    uint32_t max_w, max_h;
} FrDlssdOptimal;
int32_t frdlssd_optimal(void* session, uint32_t target_w, uint32_t target_h,
                        FrDlssdOptimal* out);

// CreateFeature on a shim-owned transient queue, fence-waited before return.
// rend_* = ALLOCATION dims (the DRS range max — per-frame subrect dims name
// the real res); dlaa selects PerfQuality DLAA vs MaxQuality; depth_hw =
// NVSDK_NGX_DLSS_Depth_Type (0 = linear view-Z, our wire); flags =
// NVSDK_NGX_DLSS_Feature_Flags (IsHDR | MVLowRes mirrors the retired SL
// options). Denoise_Mode DLUnified + Roughness_Mode PACKED (roughness rides
// normals.w — the packed RGBA16F plane rr.rs always carried) + preset E on
// every per-mode preset key are set unconditionally.
int32_t frdlssd_create(void* session, uint32_t rend_w, uint32_t rend_h,
                       uint32_t target_w, uint32_t target_h, int32_t dlaa,
                       uint32_t depth_hw, int32_t flags, void** out_feature);

// ReleaseFeature + CreateFeature ONLY (session untouched; the caller drains
// the queue first — in-flight evaluates reference the feature). Failure
// leaves the old feature RELEASED and *feature nulled: the caller marks the
// session failed-loud.
int32_t frdlssd_recreate(void* session, void** feature, uint32_t rend_w,
                         uint32_t rend_h, uint32_t target_w, uint32_t target_h,
                         int32_t dlaa, uint32_t depth_hw, int32_t flags);

// One RR evaluate recorded into the caller's OPEN command list. States at
// execute: every input plane NON_PIXEL_SHADER_RESOURCE, output
// UNORDERED_ACCESS (the NGX convention — identical to today's rr.rs rest
// states, so the callers' barriers carry over unchanged).
typedef struct FrDlssdDispatch {
    void* cmdlist;       // ID3D12GraphicsCommandList*
    void* color;         // ID3D12Resource* — rend-max dims RGBA16F noisy HDR
    void* output;        // ID3D12Resource* — target dims RGBA16F (UAV state)
    void* depth;         // ID3D12Resource* — R32F linear view-Z (Depth_Type_Linear)
    void* motion;        // ID3D12Resource* — RG16F pixel-space MVs
    void* diff_albedo;   // ID3D12Resource* — RGBA8
    void* spec_albedo;   // ID3D12Resource* — RGBA8
    void* normal_rough;  // ID3D12Resource* — RGBA16F, roughness in .w (packed)
    void* spec_hit;      // ID3D12Resource* — R16F specular hit distance
    // Row-major float4x4 (gpu::row_major is the one transpose boundary on the
    // Rust side). Per-evaluate — this is what the retired SL path re-sent
    // options every frame for (the SpecularHitDistance path reads them).
    float world_to_view[16];
    float view_to_clip[16];
    float jitter[2];      // polarity = the Rust FR_NGXRR_JITTER decision, passed as-is
    float mv_scale[2];    // {1,1} default (pixels); FR_NGXRR_MV walks it
    uint32_t rend_w, rend_h;  // InRenderSubrectDimensions — this frame's real res (DRS)
    int32_t reset;
    float frame_time_ms;  // InFrameTimeDeltaInMsec; 0 = unset (helper defaults)
} FrDlssdDispatch;
int32_t frdlssd_evaluate(void* session, void* feature, const FrDlssdDispatch* d);

// GPU idle required (the caller waits the queue).
void frdlssd_release_feature(void* feature);

// After all features are released. Drops the shared-init ref (Shutdown1 on
// the last one — with SL gone, we own NGX).
void frdlssd_close(void* session);

#ifdef __cplusplus
}  // extern "C"
#endif
