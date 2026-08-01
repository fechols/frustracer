// DLSS Frame Generation via RAW NGX — flat C API for Rust FFI.
//
// Why raw NGX: the retired Streamline DLSS-G integration validated
// end-to-end yet SL's closed dlfg present layer declined to insert generated
// frames, with no diagnostic surface (git history keeps the record; the SL
// retirement then deleted the interposer entirely). ../quinlight-player
// proved the escape on this same machine: drive
// NVSDK_NGX_Feature_FrameGeneration DIRECTLY — the app evaluates
// interpolation into ITS OWN texture on ITS OWN command list and presents
// real + generated frames itself (pair-present). Nothing can silently
// decline. This shim is an adaptation of quinlight-player's
// cpp/dlssg_shim_d3d12.cpp (same author's project) with one substantive
// change: frustracer HAS a real camera, so the dispatch ABI carries the
// actual matrices/basis/jitter instead of quinlight's video-player identity
// transforms.
//
// BUILD-OPTIONAL: the DLSS SDK (nvsdk_ngx_*_dlssg.h + the nvsdk_ngx_d
// import lib + nvngx_dlssg.dll) is NOT redistributable and never committed.
// build.rs compiles this TU (beside dlssd_shim — one SDK, one gate) only
// when FRUSTRACER_DLSS_SDK points at an SDK (default:
// ..\quinlight-player\SDKs\DLSS-SDK) and emits cfg(dlss_ngx); without it
// the Rust side stubs BOTH DLSS features to "unavailable".
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
    float mv_scale[2];    // {1,1}: MvecScale converts stored MVs to PIXELS and
                          // ours already are (CLAUDE.md trap 5 — the {1/rend}
                          // form starved the snippet of motion ~2000x; the
                          // Rust side owns the lever-controlled value)
    float cam_pos[3];
    float cam_up[3];
    float cam_right[3];
    float cam_fwd[3];
    float cam_near, cam_far, cam_fov, cam_aspect;
    uint32_t rend_w, rend_h;  // this frame's guide extent (must equal create's)
    int32_t depth_inverted;
} FrDlssgDispatch;
int32_t frdlssg_dispatch(void* handle, const FrDlssgDispatch* d);

// FEATURE-scoped rebuild at new dims (ReleaseFeature + CreateFeature ONLY —
// params/init/device untouched; the caller drains the queue first). The only
// mid-session teardown: frdlssg_destroy tears at state shared with other
// in-process NGX clients (see the .cpp's ownership notes).
int32_t frdlssg_recreate(void* handle, uint32_t disp_w, uint32_t disp_h,
                         uint32_t rend_w, uint32_t rend_h);

// GPU idle required (the caller waits the queue). Pairs the shared NGX init.
void frdlssg_destroy(void* handle);

#ifdef __cplusplus
}  // extern "C"
#endif
