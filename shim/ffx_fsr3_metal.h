// FidelityFX SDK 1.1.4 — the METAL half of frustracer's FSR3 arm.
//
// Peer of shim/ffx_fsr3.cpp (backend-neutral) and shim/ffx_fsr3_vk.cpp (the
// Vulkan half). Each per-backend half drags in an API's headers and so cannot
// compile where the other's are absent — which is the whole reason each is its
// own translation unit rather than more of the neutral one.
//
// THE ASYMMETRY WITH THE VULKAN HALF IS THE WHOLE STORY. FidelityFX ships
// `ffx_vk` and `ffx_dx12` and nothing else, so the Vulkan shim is a thin C ABI
// over a STOCK backend while this file must ALSO CARRY THAT BACKEND: a complete
// `FfxInterface` (~23 callbacks) implemented against Apple Metal, in
// shim/ffx_fsr3_metal.mm. That is ~1350 lines of Objective-C++ this tree owns
// with no upstream, and it is the single largest maintenance surface the Metal
// arm has.
//
// THE C ABI IS DELIBERATELY NARROW, as in shim/ffx_shim.cpp and the Vulkan
// half: every FFX struct stays on the C++ side, so Rust sees opaque handles and
// scalars and a v1.1.x struct layout can never drift out of step with a Rust
// transcription of it — there is no transcription.
//
// TEXTURE OWNERSHIP DIFFERS FROM THE VULKAN HALF, and deliberately. There,
// `src/vk/fsr3.rs` owns all seven images and hands the handles down. Here the
// four I/O planes (colour, depth, motion, output) are still Rust's, but FFX's
// three cross-frame temporals are SHIM-OWNED, because one of them —
// `reconstructedPrevNearestDepth`, an R32_UINT image-atomic target — must be
// BUFFER-BACKED with a row stride derived from the device's linear-texture
// alignment. That requirement is not a property of Metal or of FFX: it is a
// property of how spirv-cross EMULATED image atomics when this tree transpiled
// the SPIR-V (see the .mm), so the authority on it is the same file that loads
// those metallibs. Putting it in Rust would export a transpiler artifact into
// the backend's resource vocabulary for no gain.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return codes. Distinct values rather than a bare -1 so a failure names its
// own phase in the gate's output; FFX's own FfxErrorCode is printed alongside
// by the message callback (see the .mm — without that callback the SDK fails
// silently and only a generic code surfaces).
enum {
    FRSHIM_FSR3MTL_OK = 0,
    FRSHIM_FSR3MTL_ERR_INTERNAL = 1,  // null handle / bad argument from our side
    FRSHIM_FSR3MTL_ERR_DIMS = 2,      // a zero render or upscale extent
    FRSHIM_FSR3MTL_ERR_INTERFACE = 3, // assembling the FfxInterface
    FRSHIM_FSR3MTL_ERR_CREATE = 4,    // ffxFsr3UpscalerContextCreate
    FRSHIM_FSR3MTL_ERR_DISPATCH = 5,  // ffxFsr3UpscalerContextDispatch
    FRSHIM_FSR3MTL_ERR_METAL = 6,     // a Metal object the shim owns came back nil
};

// Context-creation flags. OURS, not FFX's — the FFX constants stay in the .mm
// so this header owes nothing to the SDK being present. Values match
// shim/ffx_fsr3_vk.h's set deliberately: `src/fsr.rs` states each convention
// once and both backends read the same statement.
enum {
    FRSHIM_FSR3MTL_HDR = 1u << 0,            // ENABLE_HIGH_DYNAMIC_RANGE
    FRSHIM_FSR3MTL_DEPTH_INVERTED = 1u << 1, // ENABLE_DEPTH_INVERTED (reversed-Z)
    FRSHIM_FSR3MTL_DEPTH_INFINITE = 1u << 2, // ENABLE_DEPTH_INFINITE
    FRSHIM_FSR3MTL_AUTO_EXPOSURE = 1u << 3,  // ENABLE_AUTO_EXPOSURE
};

// One transpiled FFX FSR3 compute permutation. `hash` is FNV-1a-64 over the
// SPIR-V bytes — the key `build.rs::generate_fsr3_metallibs` emits and the .mm
// recomputes over the very same bytes from the very same blob accessor. THAT
// HASH IS THE ONLY THING THE TWO SIDES AGREE ON, so neither may change alone.
//
// `data`/`size` is the combined `[12-byte LE threadgroup header][metallib]`
// blob. Metal needs the workgroup size HOST-side at dispatch (Vulkan and DXIL
// both reflect it out of the bytecode), so build.rs parses `OpExecutionMode
// LocalSize` and prepends it; the shim strips the same 12 bytes back off.
//
// The caller owns the bytes (they are `include_bytes!`'d into the Rust binary
// and are therefore 'static); the shim only borrows them, for the lifetime of
// the handle.
typedef struct FrShimFsr3MetalLib {
    uint64_t hash;
    const uint8_t* data;
    uint32_t size;
} FrShimFsr3MetalLib;

// `device` is an `id<MTLDevice>`. Creates the FSR3 upscaler context on the
// hand-written `ffx_metal` backend, plus the three shim-owned temporals (at
// MAX render size — a smaller dispatch declares its own sub-rect).
//
// `libs`/`lib_count` is the embedded metallib table, borrowed for the lifetime
// of the handle. Context creation is what drives `fpCreatePipeline` across every
// FSR3 pass, so a mis-keyed or truncated table fails HERE rather than at the
// first dispatch that happens to need the missing permutation.
//
// `out_pipelines` (optional) receives how many compute pipeline states were
// actually built. It exists because "context created OK" is NOT the same claim
// as "the metallibs work", and the two are indistinguishable from the return
// code alone: it is the number the gate asserts a floor on, and the number that
// says how much of the table one creation really covered. MEASURED on an M1:
// ELEVEN of the 80 permutations, one per FSR3 pass at the option word its flags
// select — a sweep of all eight context-flag combinations reaches only 14
// distinct blobs, because most passes ignore most option bits and the 40 fp32
// variants are never requested at all (the caps report fp16 support). So the
// table gate proves all 80 are well-formed and this proves eleven of them
// become pipelines; nothing here proves the other 69 ever run.
int32_t frshim_fsr3_metal_create(void* device, uint32_t max_render_w,
                                 uint32_t max_render_h, uint32_t upscale_w,
                                 uint32_t upscale_h, uint32_t flags,
                                 const FrShimFsr3MetalLib* libs,
                                 uint32_t lib_count, uint32_t* out_pipelines,
                                 void** out_handle);

// One upscale recorded into the caller's `id<MTLCommandBuffer>`, which must
// have NO encoder open — the shim opens and closes its own.
//
// The four planes are `id<MTLTexture>`: `color` (RGBA16F) / `depth` (R32F) /
// `motion` (RG16F) at render res, `output` (RGBA16F) at upscale res. Their
// formats are fixed by what this renderer's wire is (see `fsr::stage_*`), and
// FFX rebuilds its own views from the format in the FfxResourceDescription, so
// those formats must match what was actually allocated.
//
// Jitter and motion-vector polarity come from the CALLER, so `fsr::JITTER_SIGN`
// and `fsr::UPSCALE_MV_SIGN` stay the single statement of each convention
// across all three backends.
int32_t frshim_fsr3_metal_dispatch(void* handle, void* cmd_buffer, void* color,
                                   void* depth, void* motion, void* output,
                                   uint32_t render_w, uint32_t render_h,
                                   uint32_t upscale_w, uint32_t upscale_h,
                                   float jitter_x, float jitter_y,
                                   float mv_scale_x, float mv_scale_y,
                                   float frame_time_delta_ms, float camera_near,
                                   float camera_far, float camera_fov_y,
                                   int32_t reset);

// Caller must have waited for the GPU first (`src/mtl/fsr3.rs` does — every
// command buffer there is committed with `waitUntilCompleted`).
void frshim_fsr3_metal_destroy(void* handle);

#ifdef __cplusplus
}  // extern "C"
#endif
