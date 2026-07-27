// DLSS Frame Generation via RAW NGX (no Streamline) — flat C API for Rust FFI.
//
// Why this exists: the Streamline DLSS-G integration (sl_shim) validates
// end-to-end on this project yet SL's closed dlfg present layer declines to
// insert generated frames, with no diagnostic surface (see CLAUDE.md's --fg
// open-issue record). ../quinlight-player proved the escape on this same
// machine: drive NVSDK_NGX_Feature_FrameGeneration DIRECTLY — the app
// evaluates interpolation into ITS OWN texture on ITS OWN command list and
// presents real + generated frames itself (pair-present). Nothing can
// silently decline. This shim is an adaptation of quinlight-player's
// cpp/dlssg_shim_d3d12.cpp (same author's project) with one substantive
// change: frustracer HAS a real camera, so the dispatch ABI carries the
// actual matrices/basis/jitter instead of quinlight's video-player identity
// transforms.
//
// BUILD-OPTIONAL: the NDA-tier DLSS SDK (nvsdk_ngx_*_dlssg.h + the
// nvsdk_ngx_d import lib + nvngx_dlssg.dll) is NOT redistributable and never
// committed. build.rs compiles this TU only when FRUSTRACER_DLSS_SDK points
// at an SDK (default: ..\quinlight-player\SDKs\DLSS-SDK) and emits
// cfg(dlssg_ngx); without it the Rust side stubs to "unavailable".
//
// All functions return 0 on success; negative shim-private codes below.

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FRDLSSG_OK               0
#define FRDLSSG_ERR_INTERNAL    -1
#define FRDLSSG_ERR_INIT        -2
#define FRDLSSG_ERR_UNSUPPORTED -3  // non-RTX-40+/old driver/HAGS off — drop the engine
#define FRDLSSG_ERR_D3D12       -4
#define FRDLSSG_ERR_DISPATCH    -5

// Create the NGX FrameGeneration feature. disp_* = the interpolation
// input/output resolution (the color texture's); rend_* = the MV/depth guide
// resolution. Warm-up (feature creation records GPU work + capability probe)
// runs on a shim-owned transient queue, fence-waited before return.
int32_t frdlssg_create(void* device, uint32_t disp_w, uint32_t disp_h,
                       uint32_t rend_w, uint32_t rend_h, int32_t color_hdr,
                       void** out_handle);

// One interpolation evaluate recorded into the caller's OPEN command list.
// The feature retains the previous color internally: pass the CURRENT frame's
// color + guides and the output receives the frame BETWEEN previous and
// current. States at execute: color/motion/depth NON_PIXEL_SHADER_RESOURCE,
// output UNORDERED_ACCESS (the DLSS-SR NGX convention).
typedef struct FrDlssgDispatch {
    void* cmdlist;  // ID3D12GraphicsCommandList*
    void* color;    // ID3D12Resource* — disp dims, the frame to pair with prev
    void* motion;   // ID3D12Resource* — rend dims RG16F pixel-space MVs
    void* depth;    // ID3D12Resource* — rend dims R32F depth (monotone)
    void* output;   // ID3D12Resource* — disp dims, receives the interpolated frame
    uint64_t frame_id;   // +1 per real frame
    int32_t reset;       // camera cut / history reset
    // Real camera data (row-major float4x4 — the SL/NGX family convention;
    // gpu::row_major is the one transpose boundary on the Rust side).
    float view_to_clip[16];
    float clip_to_view[16];
    float clip_to_prev_clip[16];
    float prev_clip_to_clip[16];
    float jitter[2];      // the RAW sample offset — raw NGX does NOT want SL's
                          // negation (measured 2026-07-26: the negated form
                          // strobes specular highlights). One convention, one place.
    float mv_scale[2];    // pixel MVs -> [-1,1]: {1/rend_w, 1/rend_h}
    float cam_pos[3];
    float cam_up[3];
    float cam_right[3];
    float cam_fwd[3];
    float cam_near, cam_far, cam_fov, cam_aspect;
    uint32_t rend_w, rend_h;  // this frame's guide extent (must equal create's)
    int32_t depth_inverted;
} FrDlssgDispatch;
int32_t frdlssg_dispatch(void* handle, const FrDlssgDispatch* d);

// GPU idle required (the caller waits the queue). Pairs the shared NGX init.
void frdlssg_destroy(void* handle);

#ifdef __cplusplus
}  // extern "C"
#endif
