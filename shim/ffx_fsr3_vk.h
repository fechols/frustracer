// FidelityFX SDK 1.1.4 — the VULKAN half of frustracer's FSR3 arm.
//
// Peer of shim/ffx_fsr3.cpp (backend-neutral) and of the Metal `FfxInterface`
// that will land beside it. Each per-backend half drags in an API's headers and
// so cannot compile where the other's are absent — which is the whole reason
// this is a separate translation unit rather than more of the neutral one.
//
// THE C ABI IS DELIBERATELY NARROW: every FFX struct stays on the C++ side, as
// in shim/ffx_shim.cpp ("the pNext desc chains and FfxApiResource descriptions
// exist only there, never mirrored in Rust"). Rust sees opaque handles and
// scalars, so a v1.1.x struct layout can never drift out of step with a Rust
// transcription of it — there is no transcription.
//
// IMAGES ARE OWNED BY RUST, not by this shim, which is the one place we depart
// from the quinlight-player reference this is ported from. `src/vk/fsr3.rs`
// creates all seven through `ash` (the same `Vk::image` every other Vulkan
// resource in this backend goes through) and hands the handles down, so there is
// no second hand-written allocator here and image lifetime stays where the rest
// of the backend's does. This shim calls only `ffx*` entry points; it needs
// <vulkan/vulkan.h> for TYPES alone.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return codes. Distinct values rather than a bare -1 so a failure names its
// own phase in the gate's output; FFX's own FfxErrorCode is printed alongside
// by the message callback (see the .cpp — without that callback the SDK fails
// silently and only a generic code surfaces).
enum {
    FRSHIM_FSR3VK_OK = 0,
    FRSHIM_FSR3VK_ERR_INTERNAL = 1,  // null handle / bad argument from our side
    FRSHIM_FSR3VK_ERR_OOM = 2,       // the scratch allocation failed
    FRSHIM_FSR3VK_ERR_INTERFACE = 3, // ffxGetInterfaceVK
    FRSHIM_FSR3VK_ERR_CREATE = 4,    // ffxFsr3UpscalerContextCreate
    FRSHIM_FSR3VK_ERR_DISPATCH = 5,  // ffxFsr3UpscalerContextDispatch
};

// Context-creation flags. OURS, not FFX's — the FFX constants stay in the .cpp
// so this header owes nothing to the SDK being present.
enum {
    FRSHIM_FSR3VK_HDR = 1u << 0,            // ENABLE_HIGH_DYNAMIC_RANGE
    FRSHIM_FSR3VK_DEPTH_INVERTED = 1u << 1, // ENABLE_DEPTH_INVERTED (reversed-Z)
    FRSHIM_FSR3VK_DEPTH_INFINITE = 1u << 2, // ENABLE_DEPTH_INFINITE
    FRSHIM_FSR3VK_AUTO_EXPOSURE = 1u << 3,  // ENABLE_AUTO_EXPOSURE
};

// Dispatchable Vulkan handles (VkDevice, VkPhysicalDevice, VkCommandBuffer) are
// real pointers and cross as `void*`; NON-dispatchable ones (VkImage) are
// 64-bit by definition and cross as `uint64_t`. Keeping the two spellings apart
// is what stops a 32-bit build from silently truncating one of them.
//
// `get_device_proc_addr` is `PFN_vkGetDeviceProcAddr` handed in from ash's
// loader. Passing it rather than referencing the symbol is what keeps this TU
// free of any direct Vulkan call.
int32_t frshim_fsr3vk_create(void* physical_device, void* device,
                             void* get_device_proc_addr, uint32_t max_render_w,
                             uint32_t max_render_h, uint32_t upscale_w,
                             uint32_t upscale_h, uint32_t flags,
                             void** out_handle);

// One upscale into a caller-owned, already-recording command buffer.
//
// The three `shared_*` images are FFX's own temporal state and MUST be supplied:
// nothing in ffx_fsr3upscaler.h marks them mandatory — the dispatch struct lists
// them beside the genuinely optional reactive / transparency masks — but FFX
// reads and writes them every frame and they must persist across frames. Their
// formats are fixed by the SDK (R32_SFLOAT / R16G16_SFLOAT / R32_UINT) and are
// asserted against the descriptions built below.
//
// Every image must already be in VK_IMAGE_LAYOUT_GENERAL. FFX rebuilds its own
// views from the format in the FfxResourceDescription and ignores any view the
// caller may hold, so those formats must match what was actually allocated.
int32_t frshim_fsr3vk_dispatch(void* handle, void* cmd_buf, uint64_t color,
                               uint64_t depth, uint64_t motion, uint64_t output,
                               uint64_t shared_dilated_depth,
                               uint64_t shared_dilated_motion,
                               uint64_t shared_recon_prev_depth,
                               uint32_t render_w, uint32_t render_h,
                               uint32_t upscale_w, uint32_t upscale_h,
                               float jitter_x, float jitter_y, float mv_scale_x,
                               float mv_scale_y, float frame_time_delta_ms,
                               float camera_near, float camera_far,
                               float camera_fov_y, int32_t reset);

// Caller must have made the device idle first (`src/vk/fsr3.rs` does).
void frshim_fsr3vk_destroy(void* handle);

#ifdef __cplusplus
}  // extern "C"
#endif
