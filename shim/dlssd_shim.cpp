// Raw-NGX DLSS Ray Reconstruction (DLSSD) shim. See dlssd_shim.h for the
// contract and why it exists (the Streamline retirement). Modeled on
// dlssg_shim.cpp — same transient-queue CreateFeature shape, same shared
// refcounted init (ngx_shared), same ownership discipline: the capability
// parameter map is NGX's OWN (GetCapabilityParameters, never
// AllocateParameters) and is never destroyed; the only things a handle
// exclusively owns are its feature (frdlssd_release_feature) and its
// shared-init ref (frdlssd_close).
//
// Parameter conventions deliberately NOT decided here: jitter polarity and
// mv_scale ride the dispatch verbatim (the Rust side owns the FR_NGXRR_*
// levers — SL wanted negated jitter + {1/rw,1/rh}, raw NGX FG wants raw +
// {1,1} pixels, and raw DLSSD was never validated in this codebase; both
// prior conventions shipped wrong once, so the polarity is settled
// empirically above, not assumed here).

#include "dlssd_shim.h"
#include "ngx_shared.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_params.h>
#include <nvsdk_ngx_helpers_dlssd.h>

namespace {

struct Session {
    ID3D12Device*        device = nullptr;
    NVSDK_NGX_Parameter* params = nullptr;  // NGX-owned capability map — never destroyed
};

struct Feature {
    NVSDK_NGX_Handle* handle = nullptr;
};

template <typename T>
void release(T*& p) {
    if (p) {
        p->Release();
        p = nullptr;
    }
}

// CreateFeature records GPU work — run it on a shim-owned transient queue,
// submitted + fence-waited before return (the dlssg_shim create_feature
// shape, generalized over the create params).
int32_t create_on_transient_queue(ID3D12Device* dev, NVSDK_NGX_Parameter* params,
                                  NVSDK_NGX_DLSSD_Create_Params* cp,
                                  NVSDK_NGX_Handle** out_handle) {
    ID3D12CommandQueue*        queue = nullptr;
    ID3D12CommandAllocator*    alloc = nullptr;
    ID3D12GraphicsCommandList* cmd   = nullptr;
    ID3D12Fence*               fence = nullptr;
    auto cleanup = [&]() {
        release(fence);
        release(cmd);
        release(alloc);
        release(queue);
    };
    auto fail = [&](int32_t code) {
        cleanup();
        return code;
    };

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr,
                                      IID_PPV_ARGS(&cmd))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
        std::fprintf(stderr, "[fr-dlssd] warm-up queue/list/fence creation failed\n");
        return fail(FRDLSSD_ERR_D3D12);
    }

    NVSDK_NGX_Result create_r =
        NGX_D3D12_CREATE_DLSSD_EXT(cmd, 0, 0, out_handle, params, cp);
    if (NVSDK_NGX_FAILED(create_r)) {
        std::fprintf(stderr,
            "[fr-dlssd] NGX_D3D12_CREATE_DLSSD_EXT failed: 0x%08X (rend=%ux%u target=%ux%u)\n",
            (unsigned)create_r, cp->InWidth, cp->InHeight, cp->InTargetWidth,
            cp->InTargetHeight);
        return fail(FRDLSSD_ERR_UNSUPPORTED);
    }

    // Past this point *out_handle is a live NGX feature: a failed submit must
    // release it (releasing a feature whose warm-up list never executed is the
    // same thing frdlssg_destroy does) or it leaks driver-side.
    auto fail_created = [&](int32_t code) {
        NVSDK_NGX_D3D12_ReleaseFeature(*out_handle);
        *out_handle = nullptr;
        return fail(code);
    };

    if (FAILED(cmd->Close())) return fail_created(FRDLSSD_ERR_D3D12);
    ID3D12CommandList* lists[] = {cmd};
    queue->ExecuteCommandLists(1, lists);
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event || FAILED(queue->Signal(fence, 1)) ||
        FAILED(fence->SetEventOnCompletion(1, event))) {
        if (event) CloseHandle(event);
        std::fprintf(stderr, "[fr-dlssd] warm-up fence submit failed\n");
        return fail_created(FRDLSSD_ERR_D3D12);
    }
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
    cleanup();
    return FRDLSSD_OK;
}

// Fill the create params + stamp preset E on every per-mode preset key (the
// retired SL shim set one preset across all six modes; the raw keys are
// per-mode, so the loop is explicit here).
void fill_create(Session* s, NVSDK_NGX_DLSSD_Create_Params* cp, uint32_t rend_w,
                 uint32_t rend_h, uint32_t target_w, uint32_t target_h,
                 int32_t dlaa, uint32_t depth_hw, int32_t flags) {
    cp->InDenoiseMode   = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    cp->InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
    cp->InUseHWDepth    = depth_hw != 0 ? NVSDK_NGX_DLSS_Depth_Type_HW
                                        : NVSDK_NGX_DLSS_Depth_Type_Linear;
    cp->InWidth         = rend_w;
    cp->InHeight        = rend_h;
    cp->InTargetWidth   = target_w;
    cp->InTargetHeight  = target_h;
    cp->InPerfQualityValue = dlaa != 0 ? NVSDK_NGX_PerfQuality_Value_DLAA
                                       : NVSDK_NGX_PerfQuality_Value_MaxQuality;
    cp->InFeatureCreateFlags   = flags;
    cp->InEnableOutputSubrects = false;

    static const char* kPresetKeys[] = {
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance,
        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality,
    };
    for (const char* key : kPresetKeys) {
        NVSDK_NGX_Parameter_SetUI(s->params, key,
                                  NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E);
    }
}

}  // namespace

extern "C" int32_t frdlssd_open(void* device_v, void** out_session) {
    if (!out_session) return FRDLSSD_ERR_INTERNAL;
    *out_session = nullptr;
    if (!device_v) return FRDLSSD_ERR_INTERNAL;
    auto* dev = static_cast<ID3D12Device*>(device_v);

    // Unconditional refcounted init — every raw-NGX consumer takes its own
    // ref, so teardown order between DLSSD and DLSSG cannot matter (the
    // probe-first/owns_init dance existed to detect a Streamline-owned NGX;
    // with SL retired it would be a destroy-ordering hazard instead).
    if (ngx_shared_init(dev) != 0) return FRDLSSD_ERR_INIT;

    auto* s   = new Session();
    s->device = dev;

    NVSDK_NGX_Result cap_r = NVSDK_NGX_D3D12_GetCapabilityParameters(&s->params);
    if (NVSDK_NGX_FAILED(cap_r)) {
        std::fprintf(stderr, "[fr-dlssd] GetCapabilityParameters failed: 0x%08X\n",
                     (unsigned)cap_r);
        frdlssd_close(s);
        return FRDLSSD_ERR_INIT;
    }

    // The RR analog of FrameGeneration.Available: pre-RTX hardware and old
    // drivers land here — map to UNSUPPORTED so the chain falls through to
    // the native levels, never an error.
    int rr_available = 0;
    NVSDK_NGX_Result avail_r = s->params->Get(
        NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &rr_available);
    if (NVSDK_NGX_FAILED(avail_r) || rr_available == 0) {
        int needs_update = -1, init_result = 0;
        s->params->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver,
                       &needs_update);
        s->params->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult,
                       &init_result);
        std::fprintf(stderr,
            "[fr-dlssd] SuperSamplingDenoising.Available=%d (0x%08X); "
            "FeatureInitResult=0x%08X NeedsUpdatedDriver=%d\n",
            rr_available, (unsigned)avail_r, (unsigned)init_result, needs_update);
        frdlssd_close(s);
        return FRDLSSD_ERR_UNSUPPORTED;
    }

    *out_session = s;
    return FRDLSSD_OK;
}

extern "C" int32_t frdlssd_optimal(void* session, uint32_t target_w, uint32_t target_h,
                                   FrDlssdOptimal* out) {
    if (!session || !out || target_w == 0 || target_h == 0) return FRDLSSD_ERR_INTERNAL;
    auto* s = static_cast<Session*>(session);
    float sharpness = 0.0f;
    NVSDK_NGX_Result r = NGX_DLSSD_GET_OPTIMAL_SETTINGS(
        s->params, target_w, target_h, NVSDK_NGX_PerfQuality_Value_MaxQuality,
        &out->opt_w, &out->opt_h, &out->max_w, &out->max_h, &out->min_w, &out->min_h,
        &sharpness);
    if (NVSDK_NGX_FAILED(r)) {
        std::fprintf(stderr, "[fr-dlssd] NGX_DLSSD_GET_OPTIMAL_SETTINGS failed: 0x%08X\n",
                     (unsigned)r);
        return FRDLSSD_ERR_UNSUPPORTED;
    }
    return FRDLSSD_OK;
}

extern "C" int32_t frdlssd_create(void* session, uint32_t rend_w, uint32_t rend_h,
                                  uint32_t target_w, uint32_t target_h, int32_t dlaa,
                                  uint32_t depth_hw, int32_t flags, void** out_feature) {
    if (!out_feature) return FRDLSSD_ERR_INTERNAL;
    *out_feature = nullptr;
    if (!session || rend_w == 0 || rend_h == 0 || target_w == 0 || target_h == 0)
        return FRDLSSD_ERR_INTERNAL;
    auto* s = static_cast<Session*>(session);

    NVSDK_NGX_DLSSD_Create_Params cp{};
    fill_create(s, &cp, rend_w, rend_h, target_w, target_h, dlaa, depth_hw, flags);

    auto* f = new Feature();
    int32_t cr = create_on_transient_queue(s->device, s->params, &cp, &f->handle);
    if (cr != FRDLSSD_OK) {
        delete f;
        return cr;
    }
    *out_feature = f;
    return FRDLSSD_OK;
}

extern "C" int32_t frdlssd_recreate(void* session, void** feature, uint32_t rend_w,
                                    uint32_t rend_h, uint32_t target_w, uint32_t target_h,
                                    int32_t dlaa, uint32_t depth_hw, int32_t flags) {
    if (!session || !feature || !*feature || rend_w == 0 || rend_h == 0 ||
        target_w == 0 || target_h == 0)
        return FRDLSSD_ERR_INTERNAL;
    auto* s = static_cast<Session*>(session);
    auto* f = static_cast<Feature*>(*feature);

    // ReleaseFeature + CreateFeature only (the frdlssg_recreate contract) —
    // and BOTH dim pairs travel: a window resize moves render and target
    // together, and rebuilding at old-target x new-render is the measured
    // FAIL_InvalidParameter (0xBAD00005) class.
    if (f->handle) {
        NVSDK_NGX_D3D12_ReleaseFeature(f->handle);
        f->handle = nullptr;
    }
    NVSDK_NGX_DLSSD_Create_Params cp{};
    fill_create(s, &cp, rend_w, rend_h, target_w, target_h, dlaa, depth_hw, flags);
    int32_t cr = create_on_transient_queue(s->device, s->params, &cp, &f->handle);
    if (cr != FRDLSSD_OK) {
        // The old feature is gone and no new one exists — hand the dead
        // handle back deleted so the caller can't evaluate on it.
        delete f;
        *feature = nullptr;
    }
    return cr;
}

// The Rust twin (gpu/ngxrr.rs) asserts the identical literals.
static_assert(offsetof(FrDlssdDispatch, world_to_view) == 72,
              "FrDlssdDispatch: layout disagrees with gpu/ngxrr.rs");
static_assert(sizeof(FrDlssdDispatch) == 232,
              "FrDlssdDispatch: size disagrees with gpu/ngxrr.rs");

extern "C" int32_t frdlssd_evaluate(void* session, void* feature,
                                    const FrDlssdDispatch* d) {
    if (!session || !feature || !d) return FRDLSSD_ERR_INTERNAL;
    auto* s = static_cast<Session*>(session);
    auto* f = static_cast<Feature*>(feature);
    if (!f->handle) return FRDLSSD_ERR_INTERNAL;
    // DLUnified requires the guide planes too (albedos/normals/spec-hit) —
    // check them beside the classic five rather than letting NGX turn a null
    // into an opaque evaluate failure.
    if (!d->cmdlist || !d->color || !d->output || !d->depth || !d->motion ||
        !d->diff_albedo || !d->spec_albedo || !d->normal_rough || !d->spec_hit)
        return FRDLSSD_ERR_INTERNAL;

    auto* cmd = static_cast<ID3D12GraphicsCommandList*>(d->cmdlist);

    // The matrix params are stored as POINTERS on the parameter map and read
    // synchronously inside EvaluateFeature below — local copies keep the
    // lifetime obvious regardless of what the caller's struct does after.
    float w2v[16], v2c[16];
    std::memcpy(w2v, d->world_to_view, sizeof w2v);
    std::memcpy(v2c, d->view_to_clip, sizeof v2c);

    NVSDK_NGX_D3D12_DLSSD_Eval_Params eval{};
    eval.pInDiffuseAlbedo  = static_cast<ID3D12Resource*>(d->diff_albedo);
    eval.pInSpecularAlbedo = static_cast<ID3D12Resource*>(d->spec_albedo);
    eval.pInNormals        = static_cast<ID3D12Resource*>(d->normal_rough);
    eval.pInRoughness      = nullptr;  // packed mode: roughness rides normals.w
    eval.pInColor          = static_cast<ID3D12Resource*>(d->color);
    eval.pInOutput         = static_cast<ID3D12Resource*>(d->output);
    eval.pInDepth          = static_cast<ID3D12Resource*>(d->depth);
    eval.pInMotionVectors  = static_cast<ID3D12Resource*>(d->motion);
    eval.InJitterOffsetX   = d->jitter[0];
    eval.InJitterOffsetY   = d->jitter[1];
    eval.InRenderSubrectDimensions = {d->rend_w, d->rend_h};
    eval.InReset           = d->reset;
    eval.InMVScaleX        = d->mv_scale[0];
    eval.InMVScaleY        = d->mv_scale[1];
    eval.pInSpecularHitDistance = static_cast<ID3D12Resource*>(d->spec_hit);
    eval.pInWorldToViewMatrix   = w2v;
    eval.pInViewToClipMatrix    = v2c;
    eval.InFrameTimeDeltaInMsec = d->frame_time_ms;
    // Everything else (alpha, the ColorBefore*/After* research inputs,
    // exposure, ray directions, disocclusion mask) stays null/0 — the helper
    // defaults InPreExposure/InExposureScale to 1.0 on 0.0.

    NVSDK_NGX_Result r = NGX_D3D12_EVALUATE_DLSSD_EXT(cmd, f->handle, s->params, &eval);
    if (NVSDK_NGX_FAILED(r)) {
        std::fprintf(stderr,
            "[fr-dlssd] NGX_D3D12_EVALUATE_DLSSD_EXT failed: 0x%08X (rend=%ux%u reset=%d)\n",
            (unsigned)r, d->rend_w, d->rend_h, d->reset);
        return FRDLSSD_ERR_DISPATCH;
    }
    return FRDLSSD_OK;
}

extern "C" void frdlssd_release_feature(void* feature) {
    if (!feature) return;
    auto* f = static_cast<Feature*>(feature);
    if (f->handle) {
        NVSDK_NGX_D3D12_ReleaseFeature(f->handle);
        f->handle = nullptr;
    }
    delete f;
}

extern "C" void frdlssd_close(void* session) {
    if (!session) return;
    auto* s = static_cast<Session*>(session);
    // params is NGX's own capability map (GetCapabilityParameters, never
    // AllocateParameters) — never destroyed; see dlssg_shim.cpp's ownership
    // notes for the crash that rule encodes.
    s->params = nullptr;
    s->device = nullptr;
    ngx_shared_shutdown();
    delete s;
}
