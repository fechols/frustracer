// FidelityFX SDK 1.1.4 — the VULKAN half of frustracer's FSR3 arm.
//
// Ported from ../quinlight-player/cpp/fsr3_shim.cpp, the working reference this
// integration was blueprinted from (as the DLSS, FG and XeSS shims in this tree
// were, and as shim/ffx_msvc_compat.h already was). Three of the facts below
// come from that reference and from NOTHING in the public FFX headers; each
// would otherwise have been found the expensive way, so they are recorded at the
// line that depends on them rather than only here.
//
// See shim/ffx_fsr3_vk.h for the ABI and for why images live in Rust.

#include "ffx_fsr3_vk.h"

#include <cstdio>
#include <cstdlib>

#include <vulkan/vulkan.h>

#include <FidelityFX/host/ffx_interface.h>
#include <FidelityFX/host/ffx_fsr3upscaler.h>
#include <FidelityFX/host/backends/vk/ffx_vk.h>

// FINDING 1, and the reason this is not simply "add ffx_vk.cpp to the build":
// ffx_vk.cpp does NOT link standalone. It references
// ffxSetFrameGenerationConfigToSwapchainVK, which is DEFINED in the SDK's
// FrameInterpolationSwapchain/ translation units — a whole frame-generation
// stack we do not compile and do not want (Windows keeps ffx-api v2.3.0 for FG).
// Stubbing the one symbol is the entire cost of leaving that half out. It can
// never be reached: nothing here calls it, and FFX only would through a
// swapchain we never hand it.
struct FfxFrameGenerationConfig;
extern "C" FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(
    const FfxFrameGenerationConfig*) {
    return FFX_ERROR_INVALID_ENUM;
}

namespace {

struct Fsr3Ctx {
    void* scratch = nullptr;
    size_t scratch_size = 0;
    FfxFsr3UpscalerContext ctx{};
    bool ctx_alive = false;
    VkDeviceContext vk_device_context{};
};

// FINDING 2: FFX reports validation failures and unsupported-capability details
// ONLY through this callback. Without it the SDK fails silently and context
// creation surfaces a generic error code with the real cause hidden — the
// `--gpu-debug` lesson in another vendor's currency (an instrument that exists
// and is not installed). stderr, because a shim has no logging facility.
void ffx_message(FfxMsgType type, const wchar_t* message) {
    std::fprintf(stderr, "[fsr3-vk][ffx type=%d] %ls\n", (int)type, message);
    std::fflush(stderr);
}

// Built by hand rather than through ffxGetImageResourceDescriptionVK, which
// would need a whole VkImageCreateInfo across the C ABI to say what these six
// fields already say.
FfxResourceDescription rdesc_tex2d(uint32_t w, uint32_t h, FfxSurfaceFormat fmt,
                                   FfxResourceUsage usage) {
    FfxResourceDescription d{};
    d.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    d.format = fmt;
    d.width = w;
    d.height = h;
    d.depth = 1;
    d.mipCount = 1;
    d.flags = FFX_RESOURCE_FLAGS_NONE;
    d.usage = usage;
    return d;
}

// FFX rebuilds its own image view from the format in the description and
// ignores any view the caller holds, so each of these must name what
// src/vk/fsr3.rs actually allocated. The three shared formats are the SDK's,
// not ours — they are what its internal passes are compiled against.
FfxResource wrap(uint64_t image, uint32_t w, uint32_t h, FfxSurfaceFormat fmt,
                 FfxResourceUsage usage, FfxResourceStates state,
                 const wchar_t* name) {
    return ffxGetResourceVK(reinterpret_cast<void*>(static_cast<uintptr_t>(image)),
                            rdesc_tex2d(w, h, fmt, usage), name, state);
}

constexpr FfxResourceUsage READ = FFX_RESOURCE_USAGE_READ_ONLY;
constexpr FfxResourceUsage UAV = FFX_RESOURCE_USAGE_UAV;
constexpr FfxResourceStates ST_READ = FFX_RESOURCE_STATE_COMPUTE_READ;
constexpr FfxResourceStates ST_UAV = FFX_RESOURCE_STATE_UNORDERED_ACCESS;

}  // namespace

extern "C" int32_t frshim_fsr3vk_create(void* physical_device, void* device,
                                        void* get_device_proc_addr,
                                        uint32_t max_render_w,
                                        uint32_t max_render_h,
                                        uint32_t upscale_w, uint32_t upscale_h,
                                        uint32_t flags, void** out_handle) {
    if (!out_handle) return FRSHIM_FSR3VK_ERR_INTERNAL;
    *out_handle = nullptr;
    if (!physical_device || !device || !get_device_proc_addr)
        return FRSHIM_FSR3VK_ERR_INTERNAL;
    if (!max_render_w || !max_render_h || !upscale_w || !upscale_h)
        return FRSHIM_FSR3VK_ERR_INTERNAL;

    // No queue is taken, deliberately: FFX records into the command buffer
    // handed to dispatch and derives everything else from the device context.
    auto* s = new Fsr3Ctx();

    s->scratch_size = ffxGetScratchMemorySizeVK(
        static_cast<VkPhysicalDevice>(physical_device),
        FFX_FSR3UPSCALER_CONTEXT_COUNT);

    // FINDING 3, and the one most likely to be "simplified" back into a bug:
    // the scratch buffer must be ZERO-INITIALIZED. ffxGetInterfaceVK reads its
    // BackendContext_VK refCount at offset 0 BEFORE clearing the buffer, and
    // rejects a non-zero value as an already-live context with
    // FFX_ERROR_BACKEND_API_ERROR. calloc, never malloc.
    s->scratch = std::calloc(1, s->scratch_size);
    if (!s->scratch) {
        frshim_fsr3vk_destroy(s);
        return FRSHIM_FSR3VK_ERR_OOM;
    }

    s->vk_device_context.vkDevice = static_cast<VkDevice>(device);
    s->vk_device_context.vkPhysicalDevice =
        static_cast<VkPhysicalDevice>(physical_device);
    s->vk_device_context.vkDeviceProcAddr =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(get_device_proc_addr);

    FfxFsr3UpscalerContextDescription desc{};
    desc.flags = 0;
    if (flags & FRSHIM_FSR3VK_HDR)
        desc.flags |= FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE;
    if (flags & FRSHIM_FSR3VK_DEPTH_INVERTED)
        desc.flags |= FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED;
    if (flags & FRSHIM_FSR3VK_DEPTH_INFINITE)
        desc.flags |= FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE;
    if (flags & FRSHIM_FSR3VK_AUTO_EXPOSURE)
        desc.flags |= FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE;
    desc.maxRenderSize = {max_render_w, max_render_h};
    desc.maxUpscaleSize = {upscale_w, upscale_h};
    desc.fpMessage = &ffx_message;

    FfxDevice ffx_dev = ffxGetDeviceVK(&s->vk_device_context);
    FfxErrorCode fr =
        ffxGetInterfaceVK(&desc.backendInterface, ffx_dev, s->scratch,
                          s->scratch_size, FFX_FSR3UPSCALER_CONTEXT_COUNT);
    if (fr != FFX_OK) {
        std::fprintf(stderr, "[fsr3-vk] ffxGetInterfaceVK failed: 0x%08x\n",
                     (unsigned)fr);
        std::fflush(stderr);
        frshim_fsr3vk_destroy(s);
        return FRSHIM_FSR3VK_ERR_INTERFACE;
    }

    fr = ffxFsr3UpscalerContextCreate(&s->ctx, &desc);
    if (fr != FFX_OK) {
        std::fprintf(stderr,
                     "[fsr3-vk] ffxFsr3UpscalerContextCreate failed: 0x%08x\n",
                     (unsigned)fr);
        std::fflush(stderr);
        frshim_fsr3vk_destroy(s);
        return FRSHIM_FSR3VK_ERR_CREATE;
    }
    s->ctx_alive = true;

    *out_handle = s;
    return FRSHIM_FSR3VK_OK;
}

extern "C" int32_t frshim_fsr3vk_dispatch(
    void* handle, void* cmd_buf, uint64_t color, uint64_t depth,
    uint64_t motion, uint64_t output, uint64_t shared_dilated_depth,
    uint64_t shared_dilated_motion, uint64_t shared_recon_prev_depth,
    uint32_t render_w, uint32_t render_h, uint32_t upscale_w,
    uint32_t upscale_h, float jitter_x, float jitter_y, float mv_scale_x,
    float mv_scale_y, float frame_time_delta_ms, float camera_near,
    float camera_far, float camera_fov_y, int32_t reset) {
    if (!handle || !cmd_buf) return FRSHIM_FSR3VK_ERR_INTERNAL;
    Fsr3Ctx& s = *static_cast<Fsr3Ctx*>(handle);

    FfxFsr3UpscalerDispatchDescription d{};
    d.commandList = ffxGetCommandListVK(static_cast<VkCommandBuffer>(cmd_buf));

    // Inputs at render res. Colour is RGBA16F and depth R32F because that is
    // what this renderer's wire is: `accum` through fsr::f16_sat, and
    // xess::view_z_to_clip_depth's reversed-Z clip depth. (The reference feeds
    // an R16 luma-derived pseudo-depth instead, which is why it sets no
    // DEPTH_INVERTED and we do.)
    d.color = wrap(color, render_w, render_h,
                   FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, READ, ST_READ,
                   L"color");
    d.depth = wrap(depth, render_w, render_h, FFX_SURFACE_FORMAT_R32_FLOAT,
                   READ, ST_READ, L"depth");
    d.motionVectors = wrap(motion, render_w, render_h,
                           FFX_SURFACE_FORMAT_R16G16_FLOAT, READ, ST_READ,
                           L"motion");
    d.output = wrap(output, upscale_w, upscale_h,
                    FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, UAV, ST_UAV,
                    L"output");

    // FFX's cross-frame temporal state. Mandatory in fact and optional-looking
    // in the header — see ffx_fsr3_vk.h. Always at RENDER res; the formats are
    // the SDK's own.
    d.dilatedDepth =
        wrap(shared_dilated_depth, render_w, render_h,
             FFX_SURFACE_FORMAT_R32_FLOAT, UAV, ST_UAV, L"dilated_depth");
    d.dilatedMotionVectors =
        wrap(shared_dilated_motion, render_w, render_h,
             FFX_SURFACE_FORMAT_R16G16_FLOAT, UAV, ST_UAV, L"dilated_motion");
    d.reconstructedPrevNearestDepth =
        wrap(shared_recon_prev_depth, render_w, render_h,
             FFX_SURFACE_FORMAT_R32_UINT, UAV, ST_UAV, L"recon_prev_depth");

    // Genuinely optional and genuinely unused: this renderer has no
    // transparency pass to build a reactive mask from, and exposure is either
    // FFX's own (AUTO_EXPOSURE) or the fixed paper-white anchor.
    d.exposure = {};
    d.reactive = {};
    d.transparencyAndComposition = {};

    // Signs come from the caller so that src/fsr.rs's JITTER_SIGN and
    // UPSCALE_MV_SIGN remain the single statement of each convention across
    // both FidelityFX generations. Hardcoding either here would make this the
    // second place a polarity is decided.
    d.jitterOffset = {jitter_x, jitter_y};
    d.motionVectorScale = {mv_scale_x, mv_scale_y};
    d.renderSize = {render_w, render_h};
    d.upscaleSize = {upscale_w, upscale_h};
    d.enableSharpening = false;
    d.sharpness = 0.0f;
    d.frameTimeDelta = frame_time_delta_ms;
    d.preExposure = 1.0f;
    d.reset = (reset != 0);
    d.cameraNear = camera_near;
    d.cameraFar = camera_far;
    d.cameraFovAngleVertical = camera_fov_y;
    d.viewSpaceToMetersFactor = 1.0f;
    d.flags = 0;

    FfxErrorCode fr = ffxFsr3UpscalerContextDispatch(&s.ctx, &d);
    if (fr != FFX_OK) {
        std::fprintf(stderr,
                     "[fsr3-vk] ffxFsr3UpscalerContextDispatch failed: 0x%08x\n",
                     (unsigned)fr);
        std::fflush(stderr);
        return FRSHIM_FSR3VK_ERR_DISPATCH;
    }
    return FRSHIM_FSR3VK_OK;
}

extern "C" void frshim_fsr3vk_destroy(void* handle) {
    if (!handle) return;
    auto* s = static_cast<Fsr3Ctx*>(handle);
    // The caller has already made the device idle (src/vk/fsr3.rs's destroy),
    // and on the create-failure paths above nothing was ever submitted.
    if (s->ctx_alive) ffxFsr3UpscalerContextDestroy(&s->ctx);
    if (s->scratch) std::free(s->scratch);
    delete s;
}
