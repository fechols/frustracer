// Raw-NGX DLSS Frame Generation shim. See dlssg_shim.h for the contract and
// provenance (an adaptation of ../quinlight-player/cpp/dlssg_shim_d3d12.cpp +
// ngx_d3d12_init.cpp — the proven-working integration on this machine).
//
// Parameter decisions inherited from quinlight (each empirically settled
// there): mvecScale normalizes pixel MVs to [-1,1] = {1/rend_w, 1/rend_h};
// motionVectorsInvalidValue = FLT_MAX ("no pixel is ever invalid" — 0 would
// tag every static pixel invalid, and FLT_MAX is unrepresentable in an fp16
// MV texture); NativeBackbufferFormat fp16 is accepted (unlike SL DLSS-G's
// swapchain-format policing — there is no swapchain on this path at all).
// The one substantive change: real camera data rides the dispatch, where the
// video player passed identity transforms.

#include "dlssg_shim.h"
#include "ngx_shared.h"

#include <cfloat>
#include <cstdio>
#include <cstring>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_params.h>
#include <nvsdk_ngx_helpers_dlssg.h>

namespace {

// The refcounted one-init-per-device NGX bring-up lives in ngx_shared.{h,cpp}
// — shared with the DLSSD (ray reconstruction) consumer.

struct Ctx {
    ID3D12Device*        device  = nullptr;
    NVSDK_NGX_Parameter* params  = nullptr;
    NVSDK_NGX_Handle*    feature = nullptr;
    uint32_t disp_w = 0, disp_h = 0, rend_w = 0, rend_h = 0;
    bool color_hdr = false;
    /// Whether this Ctx holds a ref on the shared refcounted init (false
    /// only before frdlssg_create's init succeeds — the destroy-on-failure
    /// guard). Always true on a live handle since the SL retirement made the
    /// init unconditional.
    bool owns_init = false;
};

template <typename T>
void release(T*& p) {
    if (p) {
        p->Release();
        p = nullptr;
    }
}

void copy4x4(float dst[4][4], const float src[16]) {
    std::memcpy(dst, src, sizeof(float) * 16);
}

// Create s->feature at s's current dims over a shim-owned transient queue
// (the C ABI takes no command list), submitted + fence-waited before return.
// FEATURE-scoped by design: touches nothing but s->feature, so
// frdlssg_recreate (a render-res move) can reuse it MID-SESSION while the
// shared NGX state (the params map, the DLSSD session's feature) stays live.
// Failure returns the error code and leaves the Ctx intact — the CALLER
// decides between full teardown (create) and failed-but-alive (recreate).
int32_t create_feature(Ctx* s) {
    ID3D12Device*              dev   = s->device;
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
        std::fprintf(stderr, "[fr-dlssg] warm-up queue/list/fence creation failed\n");
        return fail(FRDLSSG_ERR_D3D12);
    }

    NVSDK_NGX_DLSSG_Create_Params cp{};
    cp.Width                    = s->disp_w;
    cp.Height                   = s->disp_h;
    cp.NativeBackbufferFormat   = (unsigned int)DXGI_FORMAT_R16G16B16A16_FLOAT;
    cp.RenderWidth              = s->rend_w;
    cp.RenderHeight             = s->rend_h;
    cp.DynamicResolutionScaling = false;

    NVSDK_NGX_Result create_r =
        NGX_D3D12_CREATE_DLSSG(cmd, 0, 0, &s->feature, s->params, &cp);
    if (NVSDK_NGX_FAILED(create_r)) {
        std::fprintf(stderr,
            "[fr-dlssg] NGX_D3D12_CREATE_DLSSG failed: 0x%08X (disp=%ux%u rend=%ux%u)\n",
            (unsigned)create_r, s->disp_w, s->disp_h, s->rend_w, s->rend_h);
        return fail(FRDLSSG_ERR_UNSUPPORTED);
    }

    if (FAILED(cmd->Close())) return fail(FRDLSSG_ERR_D3D12);
    ID3D12CommandList* lists[] = {cmd};
    queue->ExecuteCommandLists(1, lists);
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event || FAILED(queue->Signal(fence, 1)) ||
        FAILED(fence->SetEventOnCompletion(1, event))) {
        if (event) CloseHandle(event);
        std::fprintf(stderr, "[fr-dlssg] warm-up fence submit failed\n");
        return fail(FRDLSSG_ERR_D3D12);
    }
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
    cleanup();
    return FRDLSSG_OK;
}

}  // namespace

extern "C" int32_t frdlssg_create(void* device_v, uint32_t disp_w, uint32_t disp_h,
                                  uint32_t rend_w, uint32_t rend_h, int32_t color_hdr,
                                  void** out_handle) {
    if (!out_handle) return FRDLSSG_ERR_INTERNAL;
    *out_handle = nullptr;
    if (!device_v || disp_w == 0 || disp_h == 0 || rend_w == 0 || rend_h == 0)
        return FRDLSSG_ERR_INTERNAL;
    auto* dev = static_cast<ID3D12Device*>(device_v);

    auto* s = new Ctx();
    s->device = dev;
    s->disp_w = disp_w;
    s->disp_h = disp_h;
    s->rend_w = rend_w;
    s->rend_h = rend_h;
    s->color_hdr = color_hdr != 0;

    // Unconditional refcounted init — every raw-NGX consumer (this and the
    // DLSSD/RR session) takes its OWN ref through ngx_shared, so teardown
    // order between the two cannot matter: Shutdown1 runs only when the last
    // ref drops. (The probe-first/owns_init dance this replaces existed to
    // detect a Streamline-owned in-process NGX; with SL retired it became a
    // hazard — a consumer that merely rode another's init would see NGX shut
    // down under its live feature if the owner died first.)
    if (ngx_shared_init(dev) != 0) {
        frdlssg_destroy(s); // owns_init still false — no ref to drop
        return FRDLSSG_ERR_INIT;
    }
    s->owns_init = true;
    NVSDK_NGX_Result cap_r = NVSDK_NGX_D3D12_GetCapabilityParameters(&s->params);
    if (NVSDK_NGX_FAILED(cap_r)) {
        std::fprintf(stderr, "[fr-dlssg] GetCapabilityParameters failed: 0x%08X\n",
                     (unsigned)cap_r);
        frdlssg_destroy(s);
        return FRDLSSG_ERR_INIT;
    }

    // The DLSS-G analog of SuperSampling.Available: non-RTX-40+/old driver/
    // HAGS-off all land here (or at CreateFeature below) — map to UNSUPPORTED
    // so the session degrades to frame-generation-off, never an error.
    int fg_available = 0;
    NVSDK_NGX_Result avail_r =
        s->params->Get(NVSDK_NGX_Parameter_FrameGeneration_Available, &fg_available);
    if (NVSDK_NGX_FAILED(avail_r) || fg_available == 0) {
        int needs_update = -1, init_result = 0;
        s->params->Get(NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, &needs_update);
        s->params->Get(NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult, &init_result);
        std::fprintf(stderr,
            "[fr-dlssg] FrameGeneration.Available=%d (0x%08X); FeatureInitResult=0x%08X "
            "NeedsUpdatedDriver=%d — needs RTX 40+, a recent driver, and HAGS on\n",
            fg_available, (unsigned)avail_r, (unsigned)init_result, needs_update);
        frdlssg_destroy(s);
        return FRDLSSG_ERR_UNSUPPORTED;
    }

    // The feature's creation work runs on a shim-owned transient queue
    // inside create_feature (shared with frdlssg_recreate). Initial-create
    // failures tear the whole Ctx down — nothing else references it yet.
    int32_t cr = create_feature(s);
    if (cr != FRDLSSG_OK) {
        frdlssg_destroy(s);
        return cr;
    }

    *out_handle = s;
    return FRDLSSG_OK;
}

extern "C" int32_t frdlssg_recreate(void* handle, uint32_t disp_w, uint32_t disp_h,
                                    uint32_t rend_w, uint32_t rend_h) {
    if (!handle || disp_w == 0 || disp_h == 0 || rend_w == 0 || rend_h == 0)
        return FRDLSSG_ERR_INTERNAL;
    auto* s = static_cast<Ctx*>(handle);
    // FEATURE-scoped by contract: ReleaseFeature + CreateFeature at the new
    // render res, params/init/device untouched. The first attempt at this
    // used frdlssg_destroy + frdlssg_create — and destroy calls
    // NVSDK_NGX_D3D12_DestroyParameters on the map GetCapabilityParameters
    // returned, which in an SL session is SHARED with the in-process
    // Streamline NGX state: every subsequent sl.dlss_d (RR) evaluate failed
    // FeatureNotFound (0xBAD00004). Mid-session teardown may release ONLY
    // what this Ctx exclusively owns — the feature handle. The caller
    // drains the queue first (in-flight evaluates reference the feature).
    if (s->feature) {
        NVSDK_NGX_D3D12_ReleaseFeature(s->feature);
        s->feature = nullptr;
    }
    // DISPLAY dims move too, and must: this is the ONLY teardown a WINDOW
    // resize may use (frdlssg_destroy breaks the shared SL NGX state — see
    // above), and a resize changes both sizes. Updating rend alone rebuilt
    // the feature as cp.Width/Height = the OLD display with the NEW, larger
    // RenderWidth/Height, which NGX rejects at evaluate with
    // FAIL_InvalidParameter (0xBAD00005) — measured on an 8K resize.
    s->disp_w = disp_w;
    s->disp_h = disp_h;
    s->rend_w = rend_w;
    s->rend_h = rend_h;
    // Failure leaves the Ctx alive (params/init still serve a later retry or
    // the session-end destroy); the Rust side marks the session failed-loud.
    return create_feature(s);
}

// The Rust twin (gpu/mod.rs::ngxfg) asserts the identical literals.
static_assert(offsetof(FrDlssgDispatch, view_to_clip) == 52,
              "FrDlssgDispatch: layout disagrees with gpu/mod.rs::ngxfg");
static_assert(sizeof(FrDlssgDispatch) == 400,
              "FrDlssgDispatch: size disagrees with gpu/mod.rs::ngxfg");

extern "C" int32_t frdlssg_dispatch(void* handle, const FrDlssgDispatch* d) {
    if (!handle || !d) return FRDLSSG_ERR_INTERNAL;
    Ctx& s = *static_cast<Ctx*>(handle);
    if (!d->cmdlist || !d->color || !d->motion || !d->depth || !d->output)
        return FRDLSSG_ERR_INTERNAL;
    if (d->rend_w != s.rend_w || d->rend_h != s.rend_h) return FRDLSSG_ERR_INTERNAL;

    auto* cmd = static_cast<ID3D12GraphicsCommandList*>(d->cmdlist);

    NVSDK_NGX_D3D12_DLSSG_Eval_Params eval{};
    eval.pBackbuffer        = static_cast<ID3D12Resource*>(d->color);
    eval.pDepth             = static_cast<ID3D12Resource*>(d->depth);
    eval.pMVecs             = static_cast<ID3D12Resource*>(d->motion);
    eval.pOutputInterpFrame = static_cast<ID3D12Resource*>(d->output);
    // pHudless/pUI/pUIAlpha stay null (the HUD-baked known-accept, like the
    // other FG families); pOutputRealFrame/pOutputDisableInterpolation stay
    // null — frustracer owns present/pacing.

    NVSDK_NGX_DLSSG_Opt_Eval_Params opt{};
    opt.multiFrameCount = 1;  // 2x: one generated frame per real pair
    opt.multiFrameIndex = 1;

    // Real camera data (quinlight fed identities — video has no camera; a
    // renderer does, and the snippet's reprojection terms can earn their
    // keep). clipToLensClip stays identity (no lens distortion here).
    copy4x4(opt.cameraViewToClip, d->view_to_clip);
    copy4x4(opt.clipToCameraView, d->clip_to_view);
    copy4x4(opt.clipToPrevClip, d->clip_to_prev_clip);
    copy4x4(opt.prevClipToClip, d->prev_clip_to_clip);
    static constexpr float kIdentity[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                                            0, 0, 1, 0, 0, 0, 0, 1};
    copy4x4(opt.clipToLensClip, kIdentity);

    opt.jitterOffset[0] = d->jitter[0];
    opt.jitterOffset[1] = d->jitter[1];
    opt.mvecScale[0]    = d->mv_scale[0];
    opt.mvecScale[1]    = d->mv_scale[1];
    opt.cameraPinholeOffset[0] = 0.0f;
    opt.cameraPinholeOffset[1] = 0.0f;

    std::memcpy(opt.cameraPos,   d->cam_pos,   sizeof(float) * 3);
    std::memcpy(opt.cameraUp,    d->cam_up,    sizeof(float) * 3);
    std::memcpy(opt.cameraRight, d->cam_right, sizeof(float) * 3);
    std::memcpy(opt.cameraFwd,   d->cam_fwd,   sizeof(float) * 3);
    opt.cameraNear        = d->cam_near;
    opt.cameraFar         = d->cam_far;
    opt.cameraFOV         = d->cam_fov;
    opt.cameraAspectRatio = d->cam_aspect;

    opt.colorBuffersHDR        = s.color_hdr;
    opt.depthInverted          = d->depth_inverted != 0;
    opt.cameraMotionIncluded   = true;  // our MVs reproject the exact hit point
    opt.reset                  = d->reset != 0;
    opt.automodeOverrideReset  = false;
    opt.notRenderingGameFrames = false;
    opt.orthoProjection        = false;

    opt.motionVectorsInvalidValue = FLT_MAX;  // "no pixel is ever invalid"
    opt.motionVectorsDilated      = false;
    opt.menuDetectionEnabled      = false;

    // Guide extents (rend dims differ from disp dims in upscaler sessions);
    // color/output subrects stay zero = full resource.
    opt.mvecsSubrectBase = {0, 0};
    opt.mvecsSubrectSize = {s.rend_w, s.rend_h};
    opt.depthSubrectBase = {0, 0};
    opt.depthSubrectSize = {s.rend_w, s.rend_h};

    s.params->Set(NVSDK_NGX_DLSSG_Parameter_BackbufferFrameID,
                  (unsigned long long)d->frame_id);

    NVSDK_NGX_Result r = NGX_D3D12_EVALUATE_DLSSG(cmd, s.feature, s.params, &eval, &opt);
    if (NVSDK_NGX_FAILED(r)) {
        std::fprintf(stderr,
            "[fr-dlssg] NGX_D3D12_EVALUATE_DLSSG failed: 0x%08X (frame_id=%llu reset=%d)\n",
            (unsigned)r, (unsigned long long)d->frame_id, d->reset);
        return FRDLSSG_ERR_DISPATCH;
    }
    return FRDLSSG_OK;
}

extern "C" void frdlssg_destroy(void* handle) {
    if (!handle) return;
    auto* s = static_cast<Ctx*>(handle);
    if (s->feature) {
        NVSDK_NGX_D3D12_ReleaseFeature(s->feature);
        s->feature = nullptr;
    }
    // params is DELIBERATELY not destroyed: it always comes from
    // GetCapabilityParameters (above — never AllocateParameters), so the map
    // is NGX's own and is SHARED with every other in-process NGX client, i.e.
    // Streamline. Destroying it is the exact ownership mistake `owns_init`
    // guards on the init side, and it cost a crash: on an 8K resize this ran,
    // the RR rebuild's slDLSSDGetOptimalSettings then failed eErrorNGXFailed,
    // and SL's Present hook read the freed map — an access violation inside
    // _nvngx's feature-table lookup, swallowed by its SEH and surfacing to us
    // as Present returning E_ABORT. Releasing the FEATURE above is the whole
    // of what this Ctx exclusively owns (the frdlssg_recreate contract, which
    // routed around this instead of fixing it). Should a future path ever
    // AllocateParameters, THAT map would be ours — track it with an
    // owns_params flag then, mirroring owns_init.
    s->params = nullptr;
    if (s->device) {
        if (s->owns_init) {
            ngx_shared_shutdown();
        }
        s->device = nullptr;
    }
    delete s;
}
