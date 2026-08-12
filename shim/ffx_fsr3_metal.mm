// FidelityFX SDK 1.1.4 — the METAL half of frustracer's FSR3 arm, and the
// `ffx_metal` backend it needs in order to exist at all.
//
// See shim/ffx_fsr3_metal.h for the ABI, for why the temporals are shim-owned,
// and for the FNV-1a contract with build.rs.
//
// Ported from ../quinlight-player/cpp/fsr3_shim_metal.mm, the working reference
// this integration was blueprinted from (as the DLSS, FG, XeSS and FSR3-Vulkan
// shims in this tree were). Objective-C++ with MANUAL reference counting — this
// file is compiled WITHOUT -fobjc-arc, mirroring the raw-handle style of the
// other shims, so every `new*`/`alloc` here is a +1 that some line must release.
//
// WHY THIS IS ~1350 LINES AND ffx_vk.cpp IS 4692. The mapping is what shrinks
// it, and each line of it is a real Metal property rather than a shortcut:
//   - Resource states / barriers -> NO-OPS. One MTLComputeCommandEncoder with
//     Metal's default (serial, hazard-tracked) dispatch already serializes
//     dependent passes and inserts the memory barriers FFX would otherwise ask
//     for explicitly.
//   - Descriptor sets / pipeline layouts -> discrete setTexture/setBuffer/
//     setBytes at the binding index. build.rs transpiles with
//     `--msl-decoration-binding`, so a Metal argument index EQUALS the FFX/VK
//     binding number; static samplers were remapped `binding - 1000` into 0..15
//     (Metal rejects `[[sampler(N)]]` above 15, and FFX's only sampler is at
//     1001). THAT REMAP IS A CONTRACT WITH build.rs — the same subtraction is
//     applied on both sides or every sampler binds to the wrong slot.
//   - Scratch sub-allocation -> a real C++ object. We own ffxGetInterfaceMetal,
//     so FFX never touches the scratch except through these callbacks.
//
// THE FOUR VALIDATION-ONLY BUGS. Every one of the four marked GOTCHA below was
// found ONLY under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`, and each masks
// the next (the layer aborts on the first error per command buffer), so a green
// run without them proves considerably less than it appears to. They are
// carried here verbatim rather than re-derived.

#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <FidelityFX/host/ffx_interface.h>
#include <FidelityFX/host/ffx_fsr3upscaler.h>

#include "ffx_fsr3_metal.h"

// The shared FFX shaderblob accessor (one of the backend-neutral TUs compiled
// into this same static lib) exposes this. It is the SAME accessor the Vulkan
// backend uses, which is what makes the FNV-1a key well defined: both sides
// hash bytes that came from here.
extern "C" FfxErrorCode ffxGetPermutationBlobByIndex(FfxEffect effectId,
                                                     FfxPass passId,
                                                     FfxBindStage bindStage,
                                                     uint32_t permutationOptions,
                                                     FfxShaderBlob* outBlob);

namespace {

// --------------------------------------------------------------------- FNV-1a
// Byte-identical to build.rs::fnv1a64. The two sides MUST agree on the metallib
// key; there is no other handshake between the transpiled table and the loader.
uint64_t fnv1a64(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// ----------------------------------------------------------------- format map
// FfxSurfaceFormat -> MTLPixelFormat, the Metal twin of
// ffxGetVkFormatFromSurfaceFormat. TYPELESS maps to the concrete float/uint
// variant exactly as the VK and DX12 backends do.
MTLPixelFormat mtlFormat(FfxSurfaceFormat fmt) {
    switch (fmt) {
        case FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
        case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT: return MTLPixelFormatRGBA32Float;
        case FFX_SURFACE_FORMAT_R32G32B32A32_UINT:  return MTLPixelFormatRGBA32Uint;
        case FFX_SURFACE_FORMAT_R16G16B16A16_TYPELESS:
        case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT: return MTLPixelFormatRGBA16Float;
        case FFX_SURFACE_FORMAT_R32G32_TYPELESS:
        case FFX_SURFACE_FORMAT_R32G32_FLOAT:       return MTLPixelFormatRG32Float;
        case FFX_SURFACE_FORMAT_R8_UINT:            return MTLPixelFormatR8Uint;
        case FFX_SURFACE_FORMAT_R32_TYPELESS:
        case FFX_SURFACE_FORMAT_R32_UINT:           return MTLPixelFormatR32Uint;
        case FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
        case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:     return MTLPixelFormatRGBA8Unorm;
        case FFX_SURFACE_FORMAT_R8G8B8A8_SNORM:     return MTLPixelFormatRGBA8Snorm;
        case FFX_SURFACE_FORMAT_R8G8B8A8_SRGB:      return MTLPixelFormatRGBA8Unorm_sRGB;
        case FFX_SURFACE_FORMAT_B8G8R8A8_TYPELESS:
        case FFX_SURFACE_FORMAT_B8G8R8A8_UNORM:     return MTLPixelFormatBGRA8Unorm;
        case FFX_SURFACE_FORMAT_B8G8R8A8_SRGB:      return MTLPixelFormatBGRA8Unorm_sRGB;
        case FFX_SURFACE_FORMAT_R11G11B10_FLOAT:    return MTLPixelFormatRG11B10Float;
        case FFX_SURFACE_FORMAT_R10G10B10A2_TYPELESS:
        case FFX_SURFACE_FORMAT_R10G10B10A2_UNORM:  return MTLPixelFormatRGB10A2Unorm;
        case FFX_SURFACE_FORMAT_R16G16_TYPELESS:
        case FFX_SURFACE_FORMAT_R16G16_FLOAT:       return MTLPixelFormatRG16Float;
        case FFX_SURFACE_FORMAT_R16G16_UINT:        return MTLPixelFormatRG16Uint;
        case FFX_SURFACE_FORMAT_R16G16_SINT:        return MTLPixelFormatRG16Sint;
        case FFX_SURFACE_FORMAT_R16_TYPELESS:
        case FFX_SURFACE_FORMAT_R16_FLOAT:          return MTLPixelFormatR16Float;
        case FFX_SURFACE_FORMAT_R16_UINT:           return MTLPixelFormatR16Uint;
        case FFX_SURFACE_FORMAT_R16_UNORM:          return MTLPixelFormatR16Unorm;
        case FFX_SURFACE_FORMAT_R16_SNORM:          return MTLPixelFormatR16Snorm;
        case FFX_SURFACE_FORMAT_R8_TYPELESS:
        case FFX_SURFACE_FORMAT_R8_UNORM:           return MTLPixelFormatR8Unorm;
        case FFX_SURFACE_FORMAT_R8G8_TYPELESS:
        case FFX_SURFACE_FORMAT_R8G8_UNORM:         return MTLPixelFormatRG8Unorm;
        case FFX_SURFACE_FORMAT_R8G8_UINT:          return MTLPixelFormatRG8Uint;
        case FFX_SURFACE_FORMAT_R32_FLOAT:          return MTLPixelFormatR32Float;
        case FFX_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP: return MTLPixelFormatRGB9E5Float;
        // Metal has no 3-channel pixel format at all; the 4-channel one is what
        // every other backend's TYPELESS mapping lands on too.
        case FFX_SURFACE_FORMAT_R32G32B32_FLOAT:    return MTLPixelFormatRGBA32Float;
        default:                                    return MTLPixelFormatInvalid;
    }
}

uint32_t formatBytes(MTLPixelFormat f) {
    switch (f) {
        case MTLPixelFormatRGBA32Float:
        case MTLPixelFormatRGBA32Uint:
            return 16;
        case MTLPixelFormatRG32Float:
        case MTLPixelFormatRGBA16Float:
            return 8;
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Snorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
        case MTLPixelFormatRG16Float:
        case MTLPixelFormatRG16Uint:
        case MTLPixelFormatRG16Sint:
        case MTLPixelFormatR32Uint:
        case MTLPixelFormatR32Float:
        case MTLPixelFormatRG11B10Float:
        case MTLPixelFormatRGB10A2Unorm:
        case MTLPixelFormatRGB9E5Float:
            return 4;
        case MTLPixelFormatR16Float:
        case MTLPixelFormatR16Uint:
        case MTLPixelFormatR16Unorm:
        case MTLPixelFormatR16Snorm:
        case MTLPixelFormatRG8Unorm:
        case MTLPixelFormatRG8Uint:
            return 2;
        case MTLPixelFormatR8Uint:
        case MTLPixelFormatR8Unorm:
            return 1;
        default:
            return 4;
    }
}

// ------------------------------------------------------------- backend types
struct Resource {
    id<MTLTexture> texture = nil;
    id<MTLBuffer> buffer = nil;
    bool owned = false;  // created by us (release on destroy) vs registered (external)
    FfxResourceDescription desc{};
    FfxResourceStates state = FFX_RESOURCE_STATE_COMMON;
};

struct PipelineMetal {
    id<MTLComputePipelineState> pso = nil;
    MTLSize threadsPerTG = MTLSizeMake(1, 1, 1);
    id<MTLSamplerState> samplers[4] = {nil, nil, nil, nil};
    uint32_t samplerCount = 0;
};

// Our backend scratch — a real C++ object we own, NOT FFX's packed scratch
// buffer. Legal because ffxGetInterfaceMetal is ours: FFX only ever reaches
// this through the callbacks below, and never inspects its layout.
struct BackendContext_Metal {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> initQueue = nil;  // one-time resource-init clears only

    const FrShimFsr3MetalLib* libs = nullptr;
    uint32_t libCount = 0;

    // FFX_MAX_RESOURCE_COUNT slots, index 0 reserved as null. Statics grow up
    // from 1, per-frame dynamics (RegisterResource) shrink down from the top —
    // the ffx_vk allocation scheme, mirrored so FFX's own expectations hold.
    std::vector<Resource> resources;
    uint32_t nextStatic = 1;
    uint32_t nextDynamic = FFX_MAX_RESOURCE_COUNT - 1;

    std::vector<FfxGpuJobDescription> jobs;

    // Constant-buffer staging ring (StageConstantBufferData writes here; the
    // executor binds out of it with setBytes).
    std::vector<uint8_t> cbRing;
    size_t cbCursor = 0;

    // GOTCHA 2 (validation-only), and the one with the longest reach:
    // spirv-cross EMULATES Metal image atomics with a plain MTLBuffer aliased
    // over the texture's memory, bound at the SAME slot number in buffer space
    // and addressed `alignedWidth*y + x`, where `alignedWidth` comes from the
    // `spvLinearTextureAlignment` function constant. So every R32_UINT surface
    // FFX uses for atomics (the SPD global counter; reconstructed-prev-depth)
    // must be BUFFER-BACKED at that row stride, and every pipeline must be
    // specialized with function constant 65535 set to the device's alignment.
    // Miss either and the atomic argument is simply left unbound.
    uint32_t linearTexAlign = 64;
    std::vector<std::pair<id<MTLTexture>, id<MTLBuffer>>> aliases;

    // How many compute pipeline states CreatePipelineMetal actually built. The
    // gate asserts a floor on it, because a create that returned FFX_OK having
    // built nothing is indistinguishable from a healthy one by return code —
    // and it is the honest measure of how much of the metallib table a single
    // context creation covers (see ffx_fsr3_metal.h: eleven of eighty).
    uint32_t pipelinesBuilt = 0;

    id<MTLBuffer> aliasFor(id<MTLTexture> t) {
        for (auto& a : aliases)
            if (a.first == t) return a.second;
        return nil;
    }
};

BackendContext_Metal* ctx(FfxInterface* bi) {
    return reinterpret_cast<BackendContext_Metal*>(bi->scratchBuffer);
}

// --------------------------------------------------------- resource helpers
id<MTLTexture> makeTexture(BackendContext_Metal* bc, const FfxResourceDescription& d) {
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = mtlFormat(d.format);
    td.width = d.width;
    td.height = d.height;
    td.depth = 1;
    td.mipmapLevelCount = d.mipCount ? d.mipCount : 1;
    td.storageMode = MTLStorageModePrivate;
    // GOTCHA 4: RenderTarget is not optional. On a reset dispatch FFX issues
    // FFX_GPU_JOB_CLEAR_FLOAT on these, which the executor services with a
    // render-pass load-clear — a path Metal's validation layer aborts on
    // without this usage bit.
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
               MTLTextureUsageRenderTarget;
    id<MTLTexture> t = [bc->device newTextureWithDescriptor:td];
    [td release];
    return t;  // +1, owned by the caller
}

// One-time value-initialization for a Private texture: fill a Shared staging
// buffer with the byte and blit it into mip 0, then block. FFX's value-inits
// are all zero; this covers any byte fill.
void initTextureValue(BackendContext_Metal* bc, id<MTLTexture> tex, MTLPixelFormat fmt,
                      uint32_t w, uint32_t h, unsigned char value) {
    uint32_t bpp = formatBytes(fmt);
    NSUInteger bytesPerRow = (NSUInteger)bpp * w;
    NSUInteger total = bytesPerRow * h;
    id<MTLBuffer> staging = [bc->device newBufferWithLength:total
                                                    options:MTLResourceStorageModeShared];
    if (!staging) return;
    memset([staging contents], value, total);
    id<MTLCommandBuffer> cb = [bc->initQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging
             sourceOffset:0
        sourceBytesPerRow:bytesPerRow
      sourceBytesPerImage:total
               sourceSize:MTLSizeMake(w, h, 1)
                toTexture:tex
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    [staging release];
}

id<MTLSamplerState> makeSampler(BackendContext_Metal* bc, const FfxSamplerDescription& s) {
    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    MTLSamplerMinMagFilter f = (s.filter == FFX_FILTER_TYPE_MINMAGMIP_POINT)
                                   ? MTLSamplerMinMagFilterNearest
                                   : MTLSamplerMinMagFilterLinear;
    sd.minFilter = f;
    sd.magFilter = f;
    sd.mipFilter = (s.filter == FFX_FILTER_TYPE_MINMAGMIP_LINEAR)
                       ? MTLSamplerMipFilterLinear
                       : MTLSamplerMipFilterNearest;
    auto addr = [](FfxAddressMode m) -> MTLSamplerAddressMode {
        switch (m) {
            case FFX_ADDRESS_MODE_WRAP:        return MTLSamplerAddressModeRepeat;
            case FFX_ADDRESS_MODE_MIRROR:      return MTLSamplerAddressModeMirrorRepeat;
            case FFX_ADDRESS_MODE_BORDER:      return MTLSamplerAddressModeClampToBorderColor;
            case FFX_ADDRESS_MODE_MIRROR_ONCE: return MTLSamplerAddressModeMirrorClampToEdge;
            case FFX_ADDRESS_MODE_CLAMP:
            default:                           return MTLSamplerAddressModeClampToEdge;
        }
    };
    sd.sAddressMode = addr(s.addressModeU);
    sd.tAddressMode = addr(s.addressModeV);
    sd.rAddressMode = addr(s.addressModeW);
    id<MTLSamplerState> ss = [bc->device newSamplerStateWithDescriptor:sd];
    [sd release];
    return ss;
}

}  // namespace

// ==================================================== FfxInterface callbacks
extern "C" {

static FfxVersionNumber GetSDKVersionMetal(FfxInterface*) {
    return FFX_SDK_MAKE_VERSION(FFX_SDK_VERSION_MAJOR, FFX_SDK_VERSION_MINOR,
                                FFX_SDK_VERSION_PATCH);
}

static FfxErrorCode GetEffectGpuMemoryUsageMetal(FfxInterface*, FfxUInt32,
                                                 FfxEffectMemoryUsage* out) {
    if (out) {
        out->totalUsageInBytes = 0;
        out->aliasableUsageInBytes = 0;
    }
    return FFX_OK;
}

static FfxErrorCode CreateBackendContextMetal(FfxInterface* bi, FfxEffect,
                                              FfxEffectBindlessConfig*, FfxUInt32* outId) {
    BackendContext_Metal* bc = ctx(bi);
    bc->nextStatic = 1;
    bc->nextDynamic = FFX_MAX_RESOURCE_COUNT - 1;
    bc->resources.assign(FFX_MAX_RESOURCE_COUNT, Resource{});
    bc->cbRing.assign(256 * 1024, 0);
    bc->cbCursor = 0;
    if (outId) *outId = 0;  // one effect context (FFX_FSR3UPSCALER_CONTEXT_COUNT == 1)
    return FFX_OK;
}

// APPLE SILICON, HARDCODED — and this is a PAIR with build.rs's wave64 skip.
// Apple GPUs are SIMD-32, so reporting a 32-lane wave is what makes FFX request
// exactly the non-wave64 permutation set build.rs transpiled. Widen this and
// FFX asks for a hash that was never emitted (and the wave64 variants would
// mis-execute at width 32 anyway); narrow the transpile and the same happens
// from the other side. `src/mtl/device.rs::Mtl::new` refuses a non-unified
// device, which is what keeps `deviceCoherentMemorySupported` honest here.
static FfxErrorCode GetDeviceCapabilitiesMetal(FfxInterface*, FfxDeviceCapabilities* caps) {
    caps->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_6;
    caps->waveLaneCountMin = 32;
    caps->waveLaneCountMax = 32;
    caps->fp16Supported = true;
    caps->raytracingSupported = false;
    caps->deviceCoherentMemorySupported = true;  // unified memory
    caps->dedicatedAllocationSupported = false;
    caps->bufferMarkerSupported = false;
    caps->extendedSynchronizationSupported = false;
    caps->shaderStorageBufferArrayNonUniformIndexing = true;
    return FFX_OK;
}

static FfxErrorCode DestroyBackendContextMetal(FfxInterface* bi, FfxUInt32) {
    BackendContext_Metal* bc = ctx(bi);
    for (auto& r : bc->resources) {
        if (r.owned) {
            if (r.texture) [r.texture release];
            if (r.buffer) [r.buffer release];
        }
        r = Resource{};
    }
    return FFX_OK;
}

static FfxErrorCode CreateResourceMetal(FfxInterface* bi,
                                        const FfxCreateResourceDescription* desc, FfxUInt32,
                                        FfxResourceInternal* out) {
    BackendContext_Metal* bc = ctx(bi);
    uint32_t index = bc->nextStatic++;
    if (index >= bc->nextDynamic) return FFX_ERROR_OUT_OF_RANGE;
    Resource& r = bc->resources[index];
    r = Resource{};
    r.owned = true;
    r.desc = desc->resourceDescription;
    r.state = desc->initialState;

    const FfxResourceInitData& init = desc->initData;
    if (desc->resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER) {
        NSUInteger size = desc->resourceDescription.width;  // the union's buffer-size arm
        // Shared for both heap types: on unified memory an UPLOAD heap and a
        // DEFAULT one are the same allocation, and MapResource below hands FFX
        // `[buffer contents]` unconditionally.
        r.buffer = [bc->device newBufferWithLength:size
                                           options:MTLResourceStorageModeShared];
        if (!r.buffer) return FFX_ERROR_BACKEND_API_ERROR;
        if (init.type == FFX_RESOURCE_INIT_DATA_TYPE_BUFFER && init.buffer && init.size)
            memcpy([r.buffer contents], init.buffer, init.size);
        else if (init.type == FFX_RESOURCE_INIT_DATA_TYPE_VALUE)
            memset([r.buffer contents], init.value, size);
    } else if (mtlFormat(desc->resourceDescription.format) == MTLPixelFormatR32Uint) {
        // GOTCHA 2, FFX's own side of it: R32_UINT is the image-atomic format
        // (the SPD global-atomic counter, reconstructed-prev-depth), so these
        // MUST be buffer-backed or spirv-cross's emulated atomic argument is
        // left unbound and Metal rejects the dispatch. Setting `r.buffer` is
        // sufficient — the UAV-texture bind in the executor reads it directly.
        // The row stride uses the device's linear-texture alignment so it
        // agrees with the shaders' spvLinearTextureAlignmentOverride constant.
        uint32_t w = desc->resourceDescription.width;
        uint32_t h = desc->resourceDescription.height;
        uint32_t align = bc->linearTexAlign;
        uint32_t bytesPerRow = ((w * 4 + align - 1) / align) * align;
        r.buffer = [bc->device newBufferWithLength:(NSUInteger)bytesPerRow * h
                                           options:MTLResourceStorageModePrivate];
        if (!r.buffer) return FFX_ERROR_BACKEND_API_ERROR;
        MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
        td.pixelFormat = MTLPixelFormatR32Uint;
        td.width = w;
        td.height = h;
        td.storageMode = MTLStorageModePrivate;
        // NO RenderTarget: Metal forbids it on a buffer-backed texture. That is
        // precisely why GOTCHA 3 exists below — the clear path must not be the
        // render-pass one for these.
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        r.texture = [r.buffer newTextureWithDescriptor:td offset:0 bytesPerRow:bytesPerRow];
        [td release];
        if (!r.texture) return FFX_ERROR_BACKEND_API_ERROR;
        if (init.type == FFX_RESOURCE_INIT_DATA_TYPE_VALUE) {
            // Private storage, so fill the backing buffer by blit rather than a
            // host write.
            id<MTLCommandBuffer> cb = [bc->initQueue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit fillBuffer:r.buffer
                       range:NSMakeRange(0, [r.buffer length])
                       value:init.value];
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    } else {
        r.texture = makeTexture(bc, desc->resourceDescription);
        if (!r.texture) return FFX_ERROR_BACKEND_API_ERROR;
        if (init.type == FFX_RESOURCE_INIT_DATA_TYPE_VALUE)
            initTextureValue(bc, r.texture, mtlFormat(desc->resourceDescription.format),
                             desc->resourceDescription.width,
                             desc->resourceDescription.height, init.value);
        else if (init.type == FFX_RESOURCE_INIT_DATA_TYPE_BUFFER && init.buffer && init.size)
            // FFX never takes this arm for a texture in FSR3; zeroing is the
            // conservative stand-in rather than a silent skip.
            initTextureValue(bc, r.texture, mtlFormat(desc->resourceDescription.format),
                             desc->resourceDescription.width,
                             desc->resourceDescription.height, 0);
    }
    out->internalIndex = (int32_t)index;
    return FFX_OK;
}

static FfxErrorCode DestroyResourceMetal(FfxInterface* bi, FfxResourceInternal resource,
                                         FfxUInt32) {
    BackendContext_Metal* bc = ctx(bi);
    if (resource.internalIndex <= 0 || resource.internalIndex >= (int32_t)bc->resources.size())
        return FFX_OK;
    Resource& r = bc->resources[resource.internalIndex];
    if (r.owned) {
        if (r.texture) [r.texture release];
        if (r.buffer) [r.buffer release];
    }
    r = Resource{};
    return FFX_OK;
}

static FfxErrorCode RegisterResourceMetal(FfxInterface* bi, const FfxResource* in, FfxUInt32,
                                          FfxResourceInternal* out) {
    BackendContext_Metal* bc = ctx(bi);
    if (!in || in->resource == nullptr) {
        out->internalIndex = 0;  // the reserved null slot
        return FFX_OK;
    }
    // THE SAME GUARD ITS STATIC TWIN CARRIES (CreateResourceMetal). Unreachable
    // with FSR3 — ~10 registrations per dispatch against 512 slots, all rewound
    // by UnregisterResourcesMetal every frame — and the stock ffx_vk backend is
    // no better here (an FFX_ASSERT, compiled out in release). It is a line
    // because the ASYMMETRY inside one file reads as "the dynamic side was
    // checked", and the failure it prevents is a silent out-of-bounds write
    // into `resources` rather than an error.
    if (bc->nextDynamic <= bc->nextStatic) return FFX_ERROR_OUT_OF_RANGE;
    uint32_t index = bc->nextDynamic--;
    Resource& r = bc->resources[index];
    r = Resource{};
    r.owned = false;  // external — never released here
    r.desc = in->description;
    r.state = in->state;
    if (in->description.type == FFX_RESOURCE_TYPE_BUFFER) {
        r.buffer = (id<MTLBuffer>)in->resource;
    } else {
        r.texture = (id<MTLTexture>)in->resource;
        // GOTCHA 2 again: if this is a buffer-backed image-atomic target,
        // attach its alias so the UAV bind also binds the atomic buffer. This
        // is what makes the SHIM-owned reconstructed-prev-depth work when FFX
        // registers it as an external resource each frame.
        r.buffer = bc->aliasFor(r.texture);
    }
    out->internalIndex = (int32_t)index;
    return FFX_OK;
}

static FfxResource GetResourceMetal(FfxInterface* bi, FfxResourceInternal resource) {
    BackendContext_Metal* bc = ctx(bi);
    FfxResource out{};
    if (resource.internalIndex <= 0 || resource.internalIndex >= (int32_t)bc->resources.size())
        return out;
    Resource& r = bc->resources[resource.internalIndex];
    out.resource = r.texture ? (void*)r.texture : (void*)r.buffer;
    out.description = r.desc;
    out.state = r.state;
    return out;
}

static FfxErrorCode UnregisterResourcesMetal(FfxInterface* bi, FfxCommandList, FfxUInt32) {
    BackendContext_Metal* bc = ctx(bi);
    // Clear the per-frame dynamic slots (external — nothing to release) and
    // rewind the allocator.
    for (uint32_t i = bc->nextDynamic + 1; i < FFX_MAX_RESOURCE_COUNT; ++i)
        bc->resources[i] = Resource{};
    bc->nextDynamic = FFX_MAX_RESOURCE_COUNT - 1;
    return FFX_OK;
}

static FfxErrorCode RegisterStaticResourceMetal(FfxInterface*,
                                                const FfxStaticResourceDescription*, FfxUInt32) {
    return FFX_OK;  // the FSR3 upscaler uses no bindless static table
}

static FfxResourceDescription GetResourceDescriptionMetal(FfxInterface* bi,
                                                          FfxResourceInternal resource) {
    BackendContext_Metal* bc = ctx(bi);
    if (resource.internalIndex <= 0 || resource.internalIndex >= (int32_t)bc->resources.size())
        return FfxResourceDescription{};
    return bc->resources[resource.internalIndex].desc;
}

static FfxErrorCode MapResourceMetal(FfxInterface* bi, FfxResourceInternal resource, void** ptr) {
    BackendContext_Metal* bc = ctx(bi);
    if (resource.internalIndex <= 0 ||
        resource.internalIndex >= (int32_t)bc->resources.size() || !ptr)
        return FFX_ERROR_INVALID_ARGUMENT;
    Resource& r = bc->resources[resource.internalIndex];
    *ptr = r.buffer ? [r.buffer contents] : nullptr;  // unified memory: no map call
    return *ptr ? FFX_OK : FFX_ERROR_INVALID_ARGUMENT;
}

static FfxErrorCode UnmapResourceMetal(FfxInterface*, FfxResourceInternal) { return FFX_OK; }

static FfxErrorCode StageConstantBufferDataMetal(FfxInterface* bi, void* data, FfxUInt32 size,
                                                 FfxConstantBuffer* cb) {
    BackendContext_Metal* bc = ctx(bi);
    if (!cb) return FFX_ERROR_INVALID_ARGUMENT;
    if (size > bc->cbRing.size()) return FFX_ERROR_OUT_OF_RANGE;
    if (bc->cbCursor + size > bc->cbRing.size())
        bc->cbCursor = 0;  // wrap; the ring only has to survive one dispatch
    uint8_t* dst = bc->cbRing.data() + bc->cbCursor;
    if (data && size) memcpy(dst, data, size);
    bc->cbCursor += (size + 255) & ~size_t(255);
    cb->data = reinterpret_cast<uint32_t*>(dst);
    cb->num32BitEntries = size / sizeof(uint32_t);
    return FFX_OK;
}

static FfxErrorCode CreatePipelineMetal(FfxInterface* bi, FfxEffect effect, FfxPass pass,
                                        uint32_t permutationOptions,
                                        const FfxPipelineDescription* desc, FfxUInt32,
                                        FfxPipelineState* out) {
    BackendContext_Metal* bc = ctx(bi);

    FfxShaderBlob blob{};
    FfxErrorCode err = bi->fpGetPermutationBlobByIndex(effect, pass,
                                                       FFX_BIND_COMPUTE_SHADER_STAGE,
                                                       permutationOptions, &blob);
    if (err != FFX_OK || !blob.data || !blob.size) return FFX_ERROR_BACKEND_API_ERROR;

    // The transpiled metallib, located by FNV-1a over the very bytes the
    // accessor just returned — the same bytes build.rs hashed.
    uint64_t key = fnv1a64(blob.data, blob.size);
    const FrShimFsr3MetalLib* lib = nullptr;
    for (uint32_t i = 0; i < bc->libCount; ++i)
        if (bc->libs[i].hash == key) {
            lib = &bc->libs[i];
            break;
        }
    if (!lib || lib->size < 12) {
        // Names the pass AND the hash: with the hash, `nm`-style grepping of
        // the emitted table says whether the permutation was never transpiled
        // or was transpiled from different bytes.
        std::fprintf(stderr,
                     "[fsr3-metal] no metallib for pass %u (hash %016llx, options %#x) — "
                     "the caps hardcode and build.rs's wave64 skip have diverged\n",
                     (unsigned)pass, (unsigned long long)key, permutationOptions);
        std::fflush(stderr);
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // `[12-byte LE threadgroup x,y,z][metallib]`.
    uint32_t tx, ty, tz;
    memcpy(&tx, lib->data + 0, 4);
    memcpy(&ty, lib->data + 4, 4);
    memcpy(&tz, lib->data + 8, 4);

    PipelineMetal* pm = new PipelineMetal();
    pm->threadsPerTG = MTLSizeMake(tx ? tx : 1, ty ? ty : 1, tz ? tz : 1);

    // DISPATCH_DATA_DESTRUCTOR_DEFAULT COPIES the bytes, so the queue argument
    // is irrelevant and the borrow of `lib->data` ends here.
    dispatch_data_t dd = dispatch_data_create(lib->data + 12, lib->size - 12,
                                              dispatch_get_main_queue(),
                                              DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError* nerr = nil;
    id<MTLLibrary> mtllib = [bc->device newLibraryWithData:dd error:&nerr];
    dispatch_release(dd);
    if (!mtllib) {
        std::fprintf(stderr, "[fsr3-metal] newLibraryWithData failed for pass %u: %s\n",
                     (unsigned)pass,
                     nerr ? [[nerr localizedDescription] UTF8String] : "?");
        std::fflush(stderr);
        delete pm;
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // ALWAYS the specialized variant, never newFunctionWithName: alone. Some
    // permutations declare spirv-cross's optional `spvLinearTextureAlignment`
    // function constant, and Metal then REFUSES the plain variant at pipeline
    // creation. Passing empty values would leave the constant undefined, which
    // is why 65535 is set below for every pipeline; permutations that do not
    // declare it ignore it.
    MTLFunctionConstantValues* fcv = [[MTLFunctionConstantValues alloc] init];
    uint32_t align = bc->linearTexAlign;
    [fcv setConstantValue:&align type:MTLDataTypeUInt atIndex:65535];
    id<MTLFunction> fn = [mtllib newFunctionWithName:@"main0" constantValues:fcv error:&nerr];
    [fcv release];
    if (!fn) {
        std::fprintf(stderr, "[fsr3-metal] newFunctionWithName main0 failed for pass %u: %s\n",
                     (unsigned)pass,
                     nerr ? [[nerr localizedDescription] UTF8String] : "?");
        std::fflush(stderr);
        [mtllib release];
        delete pm;
        return FFX_ERROR_BACKEND_API_ERROR;
    }
    pm->pso = [bc->device newComputePipelineStateWithFunction:fn error:&nerr];
    [fn release];
    [mtllib release];
    if (!pm->pso) {
        std::fprintf(stderr, "[fsr3-metal] pipeline state failed for pass %u: %s\n",
                     (unsigned)pass,
                     nerr ? [[nerr localizedDescription] UTF8String] : "?");
        std::fflush(stderr);
        delete pm;
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // Static samplers arrive in ascending slot order (1000, 1001, ...) and our
    // metallibs put them at `slot - 1000`, so sampler j binds at index j. THE
    // SAME SUBTRACTION build.rs::remap_ffx_samplers APPLIES — the two must
    // agree or every sampler lands on the wrong slot.
    pm->samplerCount = (uint32_t)desc->samplerCount;
    if (pm->samplerCount > 4) {
        // FSR3 declares one (s_LinearClamp); four is headroom, not a budget. A
        // dropped sampler binds nothing at its index and the shader reads black,
        // which is invisible in an image — so say it rather than clamp quietly.
        std::fprintf(stderr,
                     "[fsr3-metal] pass %u asks for %u static samplers, PipelineMetal holds 4 "
                     "— the rest were DROPPED and will read as unbound\n",
                     (unsigned)pass, (unsigned)desc->samplerCount);
        std::fflush(stderr);
        pm->samplerCount = 4;
    }
    for (uint32_t j = 0; j < pm->samplerCount; ++j) pm->samplers[j] = makeSampler(bc, desc->samplers[j]);

    bc->pipelinesBuilt++;
    out->pipeline = (FfxPipeline)pm;
    out->rootSignature = nullptr;
    out->cmdSignature = nullptr;

    // Flatten the blob reflection into the pipeline's binding tables.
    // `slotIndex` is the FFX/VK binding number, which IS our Metal argument
    // index (--msl-decoration-binding); an arrayed binding (count > 1) becomes
    // consecutive entries sharing slotIndex with an incrementing arrayIndex.
    //
    // THE `name` IS REQUIRED, not decoration: the FSR3 component's
    // patchResourceBindings() resolves each slot's resourceIdentifier by
    // `wcscmp` against it and returns FFX_ERROR_INVALID_ARGUMENT if it is
    // empty. Nothing in the public headers says so.
    auto widen = [](const char* src, wchar_t* dst) {
        size_t i = 0;
        for (; src && src[i] && i < FFX_RESOURCE_NAME_SIZE - 1; ++i) dst[i] = (wchar_t)src[i];
        dst[i] = L'\0';
    };

    // THE DESTINATION BOUNDS ARE CHECKED BEFORE EACH WRITE, not asserted after
    // it. Upstream's ffx_vk does the latter (FFX_ASSERT on the final count, so
    // nothing at all in a release build) and the arrays are generous —
    // FFX_MAX_NUM_SRVS/UAVS are 64 against FSR3's handful — but a blob is DATA,
    // this file has no upstream to inherit a fix from, and the failure mode of
    // a flatten that runs past the end is a silent overwrite of the pipeline
    // state's own fields. Truncating instead is loud and bounded.
    // `continue` rather than `break`, so the count is warned about ONCE per
    // category rather than once per surviving outer iteration; the loops are a
    // handful of entries either way.
    auto overflow = [&](bool dropped, const char* what, uint32_t cap) {
        if (!dropped) return;
        std::fprintf(stderr,
                     "[fsr3-metal] pass %u declares more %s than FfxPipelineState holds "
                     "(cap %u) — the excess was DROPPED, so this pipeline will read unbound "
                     "resources\n",
                     (unsigned)pass, what, cap);
        std::fflush(stderr);
    };
    bool dropped = false;

    uint32_t flat = 0;
    for (uint32_t i = 0; i < blob.srvTextureCount; ++i)
        for (uint32_t k = 0; k < blob.boundSRVTextureCounts[i]; ++k) {
            if (flat >= FFX_MAX_NUM_SRVS) { dropped = true; continue; }
            out->srvTextureBindings[flat].slotIndex = blob.boundSRVTextures[i];
            out->srvTextureBindings[flat].arrayIndex = k;
            widen(blob.boundSRVTextureNames[i], out->srvTextureBindings[flat].name);
            ++flat;
        }
    out->srvTextureCount = flat;
    overflow(dropped, "texture SRVs", FFX_MAX_NUM_SRVS);

    dropped = false;
    flat = 0;
    for (uint32_t i = 0; i < blob.uavTextureCount; ++i)
        for (uint32_t k = 0; k < blob.boundUAVTextureCounts[i]; ++k) {
            if (flat >= FFX_MAX_NUM_UAVS) { dropped = true; continue; }
            out->uavTextureBindings[flat].slotIndex = blob.boundUAVTextures[i];
            out->uavTextureBindings[flat].arrayIndex = k;
            widen(blob.boundUAVTextureNames[i], out->uavTextureBindings[flat].name);
            ++flat;
        }
    out->uavTextureCount = flat;
    overflow(dropped, "texture UAVs", FFX_MAX_NUM_UAVS);

    dropped = false;
    flat = 0;
    for (uint32_t i = 0; i < blob.srvBufferCount; ++i) {
        if (flat >= FFX_MAX_NUM_SRVS) { dropped = true; continue; }
        out->srvBufferBindings[flat].slotIndex = blob.boundSRVBuffers[i];
        out->srvBufferBindings[flat].arrayIndex = 0;
        widen(blob.boundSRVBufferNames[i], out->srvBufferBindings[flat].name);
        ++flat;
    }
    out->srvBufferCount = flat;
    overflow(dropped, "buffer SRVs", FFX_MAX_NUM_SRVS);

    dropped = false;
    flat = 0;
    for (uint32_t i = 0; i < blob.uavBufferCount; ++i) {
        if (flat >= FFX_MAX_NUM_UAVS) { dropped = true; continue; }
        out->uavBufferBindings[flat].slotIndex = blob.boundUAVBuffers[i];
        out->uavBufferBindings[flat].arrayIndex = 0;
        widen(blob.boundUAVBufferNames[i], out->uavBufferBindings[flat].name);
        ++flat;
    }
    out->uavBufferCount = flat;
    overflow(dropped, "buffer UAVs", FFX_MAX_NUM_UAVS);

    dropped = false;
    flat = 0;
    for (uint32_t i = 0; i < blob.cbvCount; ++i) {
        if (flat >= FFX_MAX_NUM_CONST_BUFFERS) { dropped = true; continue; }
        out->constantBufferBindings[flat].slotIndex = blob.boundConstantBuffers[i];
        out->constantBufferBindings[flat].arrayIndex = 1;
        widen(blob.boundConstantBufferNames[i], out->constantBufferBindings[flat].name);
        ++flat;
    }
    out->constCount = flat;
    overflow(dropped, "constant buffers", FFX_MAX_NUM_CONST_BUFFERS);

    return FFX_OK;
}

static FfxErrorCode DestroyPipelineMetal(FfxInterface*, FfxPipelineState* pipeline, FfxUInt32) {
    if (!pipeline || !pipeline->pipeline) return FFX_OK;
    PipelineMetal* pm = (PipelineMetal*)pipeline->pipeline;
    if (pm->pso) [pm->pso release];
    for (uint32_t j = 0; j < pm->samplerCount; ++j)
        if (pm->samplers[j]) [pm->samplers[j] release];
    delete pm;
    pipeline->pipeline = nullptr;
    return FFX_OK;
}

static FfxErrorCode ScheduleGpuJobMetal(FfxInterface* bi, const FfxGpuJobDescription* job) {
    BackendContext_Metal* bc = ctx(bi);
    // A value copy: the cbs[].data pointers alias the ring, which stays valid
    // until ExecuteGpuJobs rewinds it.
    bc->jobs.push_back(*job);
    return FFX_OK;
}

static void texFromIndex(BackendContext_Metal* bc, int32_t idx, id<MTLTexture>* out) {
    *out = (idx > 0 && idx < (int32_t)bc->resources.size()) ? bc->resources[idx].texture : nil;
}

static id<MTLBuffer> bufFromIndex(BackendContext_Metal* bc, int32_t idx) {
    return (idx > 0 && idx < (int32_t)bc->resources.size()) ? bc->resources[idx].buffer : nil;
}

static FfxErrorCode ExecuteGpuJobsMetal(FfxInterface* bi, FfxCommandList commandList, FfxUInt32) {
    BackendContext_Metal* bc = ctx(bi);
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)commandList;
    if (!cmd) return FFX_ERROR_INVALID_ARGUMENT;

    // ONE compute encoder across all compute jobs, opened lazily. Metal's
    // default hazard tracking serializes dependent dispatches inside it, which
    // is exactly what makes FFX_GPU_JOB_BARRIER a no-op below. Blit and clear
    // jobs must close it first — hence `endCompute`.
    id<MTLComputeCommandEncoder> enc = nil;
    auto endCompute = [&]() {
        if (enc) {
            [enc endEncoding];
            enc = nil;
        }
    };

    for (const FfxGpuJobDescription& job : bc->jobs) {
        switch (job.jobType) {
            case FFX_GPU_JOB_COMPUTE: {
                const FfxComputeJobDescription& c = job.computeJobDescriptor;
                PipelineMetal* pm = (PipelineMetal*)c.pipeline.pipeline;
                if (!pm || !pm->pso) break;
                if (!enc) enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pm->pso];

                for (uint32_t i = 0; i < c.pipeline.srvTextureCount; ++i) {
                    id<MTLTexture> t = nil;
                    texFromIndex(bc, c.srvTextures[i].resource.internalIndex, &t);
                    [enc setTexture:t atIndex:c.pipeline.srvTextureBindings[i].slotIndex];
                }
                for (uint32_t i = 0; i < c.pipeline.uavTextureCount; ++i) {
                    int32_t idx = c.uavTextures[i].resource.internalIndex;
                    uint32_t slot = c.pipeline.uavTextureBindings[i].slotIndex;
                    id<MTLTexture> t = nil;
                    texFromIndex(bc, idx, &t);
                    [enc setTexture:t atIndex:slot];
                    // GOTCHA 2's bind side: spirv-cross also expects the
                    // texture's backing buffer at the SAME slot in buffer space
                    // for atomic_*_explicit. Non-atomic UAVs have no alias and
                    // skip this.
                    id<MTLBuffer> alias = bufFromIndex(bc, idx);
                    if (alias) [enc setBuffer:alias offset:0 atIndex:slot];
                }
                for (uint32_t i = 0; i < c.pipeline.srvBufferCount; ++i) {
                    id<MTLBuffer> b = bufFromIndex(bc, c.srvBuffers[i].resource.internalIndex);
                    [enc setBuffer:b
                            offset:c.srvBuffers[i].offset
                           atIndex:c.pipeline.srvBufferBindings[i].slotIndex];
                }
                for (uint32_t i = 0; i < c.pipeline.uavBufferCount; ++i) {
                    id<MTLBuffer> b = bufFromIndex(bc, c.uavBuffers[i].resource.internalIndex);
                    [enc setBuffer:b
                            offset:c.uavBuffers[i].offset
                           atIndex:c.pipeline.uavBufferBindings[i].slotIndex];
                }
                for (uint32_t i = 0; i < c.pipeline.constCount; ++i) {
                    uint32_t bytes = c.cbs[i].num32BitEntries * 4;
                    uint32_t slot = c.pipeline.constantBufferBindings[i].slotIndex;
                    // GOTCHA 1 (validation-only): spirv-cross rounds an MSL
                    // constant-buffer struct up to 16-byte alignment, so the
                    // shader's argument is sized align(bytes,16) — cbFSR3Upscaler
                    // is 148 bytes and the MSL struct is 160. FFX's `data`
                    // pointer backs only `bytes`, so passing the larger length
                    // over-reads it (UB) while passing the smaller one is
                    // REJECTED as a too-small bind. Copy into a zero-padded
                    // temp: the pad is struct padding no shader reads.
                    uint32_t padded = (bytes + 15u) & ~15u;
                    if (padded == bytes) {
                        [enc setBytes:c.cbs[i].data length:bytes atIndex:slot];
                    } else {
                        uint8_t tmp[512];  // FSR3's cbuffers are far under this
                        if (padded <= sizeof(tmp)) {
                            memcpy(tmp, c.cbs[i].data, bytes);
                            memset(tmp + bytes, 0, padded - bytes);
                            [enc setBytes:tmp length:padded atIndex:slot];
                        } else {
                            // Unreachable today. Binding the unpadded size is
                            // the conservative fallback: validation names it
                            // loudly, where a silent over-read would not be.
                            [enc setBytes:c.cbs[i].data length:bytes atIndex:slot];
                        }
                    }
                }
                for (uint32_t j = 0; j < pm->samplerCount; ++j)
                    [enc setSamplerState:pm->samplers[j] atIndex:j];

                [enc dispatchThreadgroups:MTLSizeMake(c.dimensions[0], c.dimensions[1],
                                                      c.dimensions[2])
                    threadsPerThreadgroup:pm->threadsPerTG];
                break;
            }
            case FFX_GPU_JOB_CLEAR_FLOAT: {
                endCompute();
                const FfxClearFloatJobDescription& cl = job.clearJobDescriptor;
                int32_t idx = cl.target.internalIndex;
                id<MTLTexture> t = nil;
                texFromIndex(bc, idx, &t);
                if (!t) break;
                // GOTCHA 3 (validation-only): a buffer-backed image-atomic
                // texture CANNOT carry MTLTextureUsageRenderTarget (Metal
                // forbids the combination), so the render-pass load-clear below
                // is illegal on exactly those surfaces. FFX only ever
                // zero-clears them (the reset reconstructed-depth init), and
                // every byte of a 0.0f fill is zero, so a blit fill of the
                // backing buffer is EXACT rather than an approximation.
                id<MTLBuffer> backing = bufFromIndex(bc, idx);
                if (backing && !(t.usage & MTLTextureUsageRenderTarget)) {
                    // A byte fill can only express a clear whose every byte is
                    // the same, and zero is the only such float FFX asks for.
                    // Checked rather than assumed: the assumption holds today
                    // and the code cannot notice if it stops, and what it would
                    // become is a SILENTLY wrong clear — a nonzero request
                    // serviced as zero.
                    if (cl.color[0] != 0.0f || cl.color[1] != 0.0f || cl.color[2] != 0.0f ||
                        cl.color[3] != 0.0f) {
                        std::fprintf(stderr,
                                     "[fsr3-metal] CLEAR_FLOAT(%g,%g,%g,%g) on a buffer-backed "
                                     "image-atomic target, which only a byte fill can service "
                                     "— cleared to ZERO instead\n",
                                     cl.color[0], cl.color[1], cl.color[2], cl.color[3]);
                        std::fflush(stderr);
                    }
                    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                    [blit fillBuffer:backing range:NSMakeRange(0, [backing length]) value:0];
                    [blit endEncoding];
                    break;
                }
                MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
                rp.colorAttachments[0].texture = t;
                rp.colorAttachments[0].loadAction = MTLLoadActionClear;
                rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                rp.colorAttachments[0].clearColor =
                    MTLClearColorMake(cl.color[0], cl.color[1], cl.color[2], cl.color[3]);
                id<MTLRenderCommandEncoder> re = [cmd renderCommandEncoderWithDescriptor:rp];
                [re endEncoding];
                break;
            }
            case FFX_GPU_JOB_COPY: {
                endCompute();
                const FfxCopyJobDescription& cp = job.copyJobDescriptor;
                if (cp.src.internalIndex <= 0 ||
                    cp.src.internalIndex >= (int32_t)bc->resources.size() ||
                    cp.dst.internalIndex <= 0 ||
                    cp.dst.internalIndex >= (int32_t)bc->resources.size())
                    break;
                Resource& src = bc->resources[cp.src.internalIndex];
                Resource& dst = bc->resources[cp.dst.internalIndex];
                id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                if (src.buffer && dst.buffer) {
                    NSUInteger n = cp.size ? cp.size : [src.buffer length];
                    [blit copyFromBuffer:src.buffer
                            sourceOffset:cp.srcOffset
                                toBuffer:dst.buffer
                       destinationOffset:cp.dstOffset
                                    size:n];
                } else if (src.texture && dst.texture) {
                    [blit copyFromTexture:src.texture toTexture:dst.texture];
                }
                [blit endEncoding];
                break;
            }
            case FFX_GPU_JOB_BARRIER:
            case FFX_GPU_JOB_DISCARD:
            default:
                // Metal auto-tracks hazards within a serial compute encoder, so
                // an explicit barrier has nothing to add. See the encoder note
                // at the top of this function.
                break;
        }
    }
    endCompute();

    bc->jobs.clear();
    bc->cbCursor = 0;
    return FFX_OK;
}

// Breadcrumbs (FFX 1.1's post-mortem debugging) — unused here, but the function
// pointers must be non-null: FFX calls them unconditionally.
static FfxErrorCode BreadcrumbsAllocBlockMetal(FfxInterface*, uint64_t,
                                               FfxBreadcrumbsBlockData*) {
    return FFX_OK;
}
static void BreadcrumbsFreeBlockMetal(FfxInterface*, FfxBreadcrumbsBlockData*) {}
static void BreadcrumbsWriteMetal(FfxInterface*, FfxCommandList, uint32_t, uint64_t, void*, bool) {}
static void BreadcrumbsPrintDeviceInfoMetal(FfxInterface*, FfxAllocationCallbacks*, bool, char**,
                                            size_t*) {}

}  // extern "C"

// ======================================================== the interface table
static FfxErrorCode ffxGetInterfaceMetal(FfxInterface* backendInterface,
                                         BackendContext_Metal* bc) {
    if (!backendInterface) return FFX_ERROR_INVALID_ARGUMENT;
    backendInterface->fpGetSDKVersion = GetSDKVersionMetal;
    backendInterface->fpGetEffectGpuMemoryUsage = GetEffectGpuMemoryUsageMetal;
    backendInterface->fpCreateBackendContext = CreateBackendContextMetal;
    backendInterface->fpGetDeviceCapabilities = GetDeviceCapabilitiesMetal;
    backendInterface->fpDestroyBackendContext = DestroyBackendContextMetal;
    backendInterface->fpCreateResource = CreateResourceMetal;
    backendInterface->fpDestroyResource = DestroyResourceMetal;
    backendInterface->fpMapResource = MapResourceMetal;
    backendInterface->fpUnmapResource = UnmapResourceMetal;
    backendInterface->fpRegisterResource = RegisterResourceMetal;
    backendInterface->fpGetResource = GetResourceMetal;
    backendInterface->fpUnregisterResources = UnregisterResourcesMetal;
    backendInterface->fpRegisterStaticResource = RegisterStaticResourceMetal;
    backendInterface->fpGetResourceDescription = GetResourceDescriptionMetal;
    backendInterface->fpStageConstantBufferDataFunc = StageConstantBufferDataMetal;
    backendInterface->fpCreatePipeline = CreatePipelineMetal;
    backendInterface->fpDestroyPipeline = DestroyPipelineMetal;
    backendInterface->fpScheduleGpuJob = ScheduleGpuJobMetal;
    backendInterface->fpExecuteGpuJobs = ExecuteGpuJobsMetal;
    backendInterface->fpBreadcrumbsAllocBlock = BreadcrumbsAllocBlockMetal;
    backendInterface->fpBreadcrumbsFreeBlock = BreadcrumbsFreeBlockMetal;
    backendInterface->fpBreadcrumbsWrite = BreadcrumbsWriteMetal;
    backendInterface->fpBreadcrumbsPrintDeviceInfo = BreadcrumbsPrintDeviceInfoMetal;
    backendInterface->fpGetPermutationBlobByIndex = ffxGetPermutationBlobByIndex;
    // Frame generation and a caller-supplied constant allocator are both
    // Windows/ffx-api territory; FFX null-checks these two.
    backendInterface->fpSwapChainConfigureFrameGeneration = nullptr;
    backendInterface->fpRegisterConstantBufferAllocator = nullptr;
    // OUR object, not an FFX-sized scratch allocation. `scratchBufferSize` is
    // informational here — nothing in FFX allocates from it, because every
    // access goes through the callbacks above.
    backendInterface->scratchBuffer = bc;
    backendInterface->scratchBufferSize = sizeof(BackendContext_Metal);
    backendInterface->device = (FfxDevice)bc->device;
    return FFX_OK;
}

// ==================================================== the frshim_fsr3_metal_*
namespace {

// FFX reports validation failures and unsupported-capability details ONLY
// through this callback. Without it the SDK fails silently and context creation
// surfaces a generic error code with the real cause hidden — the `--gpu-debug`
// lesson in another vendor's currency (an instrument that exists and is not
// installed). The Vulkan half installs the same one.
void ffx_message(FfxMsgType type, const wchar_t* message) {
    std::fprintf(stderr, "[fsr3-metal][ffx type=%d] %ls\n", (int)type, message);
    std::fflush(stderr);
}

struct ShimTex {
    id<MTLTexture> tex = nil;
    id<MTLBuffer> backing = nil;  // non-nil only for buffer-backed image-atomic targets
    uint32_t w = 0, h = 0;
};

struct Fsr3MetalCtx {
    id<MTLDevice> device = nil;
    uint32_t max_render_w = 0, max_render_h = 0, upscale_w = 0, upscale_h = 0;

    BackendContext_Metal* backend = nullptr;
    FfxFsr3UpscalerContext ffx{};
    bool ffx_alive = false;

    // FFX's three cross-frame temporals. Shim-owned — see ffx_fsr3_metal.h for
    // why this differs from the Vulkan half. Allocated at MAX render size; a
    // smaller dispatch declares its own sub-rect and FFX reads only that.
    ShimTex dilated_depth;     // R32_FLOAT
    ShimTex dilated_motion;    // RG16_FLOAT
    ShimTex recon_prev_depth;  // R32_UINT, buffer-backed (GOTCHA 2)
};

ShimTex makeShimTex(id<MTLDevice> dev, uint32_t w, uint32_t h, MTLPixelFormat fmt) {
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.pixelFormat = fmt;
    td.width = w;
    td.height = h;
    td.storageMode = MTLStorageModePrivate;
    // GOTCHA 4: same RenderTarget requirement as the FFX-managed path in
    // makeTexture — on a reset dispatch FFX clears these through a render-pass
    // load-clear, which Metal's validation layer aborts on without it.
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
               MTLTextureUsageRenderTarget;
    ShimTex s;
    s.tex = [dev newTextureWithDescriptor:td];
    s.w = w;
    s.h = h;
    [td release];
    return s;
}

// A buffer-backed texture for an image-atomic target: the texture and its
// MTLBuffer share memory, so the shader's `texture2d<uint>` view and its
// `device atomic_uint*` alias address the same pixels. The row stride MUST use
// the device's linear-texture alignment so it agrees with spvImage2DAtomicCoord,
// which computes its address from the spvLinearTextureAlignment constant that
// CreatePipelineMetal sets from the same number.
ShimTex makeBufferBackedTex(id<MTLDevice> dev, uint32_t w, uint32_t h, MTLPixelFormat fmt,
                            uint32_t bpp, uint32_t align) {
    uint32_t bytesPerRow = ((w * bpp + align - 1) / align) * align;
    ShimTex s;
    id<MTLBuffer> buf = [dev newBufferWithLength:(NSUInteger)bytesPerRow * h
                                         options:MTLResourceStorageModePrivate];
    if (!buf) return s;
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.pixelFormat = fmt;
    td.width = w;
    td.height = h;
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    s.tex = [buf newTextureWithDescriptor:td offset:0 bytesPerRow:bytesPerRow];
    s.backing = buf;  // owned by the ShimTex; released in destroy
    s.w = w;
    s.h = h;
    [td release];
    return s;
}

// Built by hand rather than through any ffxGet*ResourceDescription helper,
// which would need a whole texture-descriptor struct across the C ABI to say
// what these six fields already say. Mirrors the Vulkan half's rdesc_tex2d.
FfxResourceDescription rdesc(uint32_t w, uint32_t h, FfxSurfaceFormat fmt, FfxResourceUsage usage) {
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

// The `name` is what FFX prints in its own diagnostics, so a mis-registered
// plane names itself rather than surfacing as a bare error code.
FfxResource wrapTex(id<MTLTexture> t, uint32_t w, uint32_t h, FfxSurfaceFormat fmt,
                    FfxResourceUsage usage, FfxResourceStates state, const wchar_t* name) {
    FfxResource r{};
    r.resource = (void*)t;
    r.description = rdesc(w, h, fmt, usage);
    r.state = state;
    if (name) {
        size_t i = 0;
        for (; name[i] && i < FFX_RESOURCE_NAME_SIZE - 1; ++i) r.name[i] = name[i];
        r.name[i] = L'\0';
    }
    return r;
}

}  // namespace

extern "C" int32_t frshim_fsr3_metal_create(void* device, uint32_t max_render_w,
                                            uint32_t max_render_h, uint32_t upscale_w,
                                            uint32_t upscale_h, uint32_t flags,
                                            const FrShimFsr3MetalLib* libs, uint32_t lib_count,
                                            uint32_t* out_pipelines, void** out_handle) {
    if (out_pipelines) *out_pipelines = 0;
    if (!out_handle) return FRSHIM_FSR3MTL_ERR_INTERNAL;
    *out_handle = nullptr;
    if (!device) return FRSHIM_FSR3MTL_ERR_INTERNAL;
    if (!max_render_w || !max_render_h || !upscale_w || !upscale_h)
        return FRSHIM_FSR3MTL_ERR_DIMS;
    if (!libs || !lib_count) return FRSHIM_FSR3MTL_ERR_INTERNAL;

    id<MTLDevice> dev = (id<MTLDevice>)device;
    auto* s = new Fsr3MetalCtx();
    s->device = dev;
    s->max_render_w = max_render_w;
    s->max_render_h = max_render_h;
    s->upscale_w = upscale_w;
    s->upscale_h = upscale_h;

    auto* bc = new BackendContext_Metal();
    bc->device = dev;
    bc->initQueue = [dev newCommandQueue];
    bc->libs = libs;
    bc->libCount = lib_count;
    bc->linearTexAlign =
        (uint32_t)[dev minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatR32Uint];
    if (bc->linearTexAlign < 4) bc->linearTexAlign = 4;
    s->backend = bc;
    if (!bc->initQueue) {
        frshim_fsr3_metal_destroy(s);
        return FRSHIM_FSR3MTL_ERR_METAL;
    }

    // The temporals FIRST, so the recon-prev-depth alias is registered before
    // ffxFsr3UpscalerContextCreate can register any resource against it.
    s->dilated_depth = makeShimTex(dev, max_render_w, max_render_h, MTLPixelFormatR32Float);
    s->dilated_motion = makeShimTex(dev, max_render_w, max_render_h, MTLPixelFormatRG16Float);
    s->recon_prev_depth = makeBufferBackedTex(dev, max_render_w, max_render_h,
                                              MTLPixelFormatR32Uint, 4, bc->linearTexAlign);
    if (!s->dilated_depth.tex || !s->dilated_motion.tex || !s->recon_prev_depth.tex ||
        !s->recon_prev_depth.backing) {
        frshim_fsr3_metal_destroy(s);
        return FRSHIM_FSR3MTL_ERR_METAL;
    }
    bc->aliases.emplace_back(s->recon_prev_depth.tex, s->recon_prev_depth.backing);

    // NOT ZEROED, and that is the SDK's declared contract rather than an
    // omission: ffx_fsr3upscaler.cpp creates all three of these with
    // {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}, and ffx_vk.cpp's own
    // CreateResourceVK skips its upload job entirely for that type. So residue
    // in texels a reset frame never wrote is content FFX has told every backend
    // it does not care about.
    //
    // MEASURED, because zeroing them looks obviously right and is not: it moved
    // the cross-context divergence of an ACCUMULATING frame from a median of
    // 388 differing channels to 1050 (8 samples each, same session) — i.e. no
    // benefit, inside the same wide band, for three blocking blits per context
    // creation. Whereas a RESET-only frame is EXACTLY bit-identical across
    // contexts either way, which is what proves nothing uninitialised is read
    // on a fresh frame at all. See `--check-fsr3`'s U4 for the two claims that
    // measurement supports.

    FfxFsr3UpscalerContextDescription desc{};
    desc.flags = 0;
    if (flags & FRSHIM_FSR3MTL_HDR) desc.flags |= FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE;
    if (flags & FRSHIM_FSR3MTL_DEPTH_INVERTED) desc.flags |= FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED;
    if (flags & FRSHIM_FSR3MTL_DEPTH_INFINITE) desc.flags |= FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE;
    if (flags & FRSHIM_FSR3MTL_AUTO_EXPOSURE) desc.flags |= FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE;
    desc.maxRenderSize = {max_render_w, max_render_h};
    desc.maxUpscaleSize = {upscale_w, upscale_h};
    desc.fpMessage = &ffx_message;

    if (ffxGetInterfaceMetal(&desc.backendInterface, bc) != FFX_OK) {
        frshim_fsr3_metal_destroy(s);
        return FRSHIM_FSR3MTL_ERR_INTERFACE;
    }

    // THIS is what drives fpCreatePipeline across every FSR3 pass, so a
    // mis-keyed metallib table fails here rather than at some later dispatch.
    FfxErrorCode fr = ffxFsr3UpscalerContextCreate(&s->ffx, &desc);
    if (fr != FFX_OK) {
        std::fprintf(stderr, "[fsr3-metal] ffxFsr3UpscalerContextCreate failed: 0x%08x\n",
                     (unsigned)fr);
        std::fflush(stderr);
        frshim_fsr3_metal_destroy(s);
        return FRSHIM_FSR3MTL_ERR_CREATE;
    }
    s->ffx_alive = true;

    if (out_pipelines) *out_pipelines = bc->pipelinesBuilt;
    *out_handle = s;
    return FRSHIM_FSR3MTL_OK;
}

extern "C" int32_t frshim_fsr3_metal_dispatch(void* handle, void* cmd_buffer, void* color,
                                              void* depth, void* motion, void* output,
                                              uint32_t render_w, uint32_t render_h,
                                              uint32_t upscale_w, uint32_t upscale_h,
                                              float jitter_x, float jitter_y, float mv_scale_x,
                                              float mv_scale_y, float frame_time_delta_ms,
                                              float camera_near, float camera_far,
                                              float camera_fov_y, int32_t reset) {
    if (!handle) return FRSHIM_FSR3MTL_ERR_INTERNAL;
    Fsr3MetalCtx& s = *static_cast<Fsr3MetalCtx*>(handle);
    if (!cmd_buffer || !color || !depth || !motion || !output)
        return FRSHIM_FSR3MTL_ERR_INTERNAL;
    if (!render_w || !render_h || render_w > s.max_render_w || render_h > s.max_render_h)
        return FRSHIM_FSR3MTL_ERR_DIMS;

    const FfxSurfaceFormat kColor = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    const FfxResourceUsage READ = FFX_RESOURCE_USAGE_READ_ONLY;
    const FfxResourceUsage UAV = FFX_RESOURCE_USAGE_UAV;
    const FfxResourceStates ST_READ = FFX_RESOURCE_STATE_COMPUTE_READ;
    const FfxResourceStates ST_UAV = FFX_RESOURCE_STATE_UNORDERED_ACCESS;

    FfxFsr3UpscalerDispatchDescription d{};
    d.commandList = (FfxCommandList)cmd_buffer;

    // Inputs at render res. The formats are what this renderer's wire IS —
    // `accum` through fsr::f16_sat, and xess::view_z_to_clip_depth's reversed-Z
    // clip depth — and they must match what Rust actually allocated, because
    // FFX rebuilds its own views from these descriptions.
    d.color = wrapTex((id<MTLTexture>)color, render_w, render_h, kColor, READ, ST_READ, L"color");
    d.depth = wrapTex((id<MTLTexture>)depth, render_w, render_h,
                      FFX_SURFACE_FORMAT_R32_FLOAT, READ, ST_READ, L"depth");
    d.motionVectors = wrapTex((id<MTLTexture>)motion, render_w, render_h,
                              FFX_SURFACE_FORMAT_R16G16_FLOAT, READ, ST_READ, L"motion");
    d.output = wrapTex((id<MTLTexture>)output, upscale_w, upscale_h, kColor, UAV, ST_UAV,
                       L"output");

    // FFX's cross-frame temporal state. Mandatory in fact and optional-looking
    // in the header, which lists them beside the genuinely optional reactive
    // and transparency masks. Always at RENDER res; the formats are the SDK's.
    d.dilatedDepth = wrapTex(s.dilated_depth.tex, render_w, render_h,
                             FFX_SURFACE_FORMAT_R32_FLOAT, UAV, ST_UAV, L"dilated_depth");
    d.dilatedMotionVectors = wrapTex(s.dilated_motion.tex, render_w, render_h,
                                     FFX_SURFACE_FORMAT_R16G16_FLOAT, UAV, ST_UAV,
                                     L"dilated_motion");
    d.reconstructedPrevNearestDepth = wrapTex(s.recon_prev_depth.tex, render_w, render_h,
                                              FFX_SURFACE_FORMAT_R32_UINT, UAV, ST_UAV,
                                              L"recon_prev_depth");

    // Genuinely optional and genuinely unused: this renderer has no
    // transparency pass to build a reactive mask from, and exposure is either
    // FFX's own (AUTO_EXPOSURE) or the fixed paper-white anchor.
    d.exposure = {};
    d.reactive = {};
    d.transparencyAndComposition = {};

    // Signs come from the caller so that src/fsr.rs's JITTER_SIGN and
    // UPSCALE_MV_SIGN stay the single statement of each convention across all
    // three backends. Hardcoding either here would make this a second place a
    // polarity is decided.
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

    FfxErrorCode fr = ffxFsr3UpscalerContextDispatch(&s.ffx, &d);
    if (fr != FFX_OK) {
        std::fprintf(stderr, "[fsr3-metal] ffxFsr3UpscalerContextDispatch failed: 0x%08x\n",
                     (unsigned)fr);
        std::fflush(stderr);
        return FRSHIM_FSR3MTL_ERR_DISPATCH;
    }
    return FRSHIM_FSR3MTL_OK;
}

extern "C" void frshim_fsr3_metal_destroy(void* handle) {
    if (!handle) return;
    auto* s = static_cast<Fsr3MetalCtx*>(handle);
    // The caller has already waited for the GPU (src/mtl/fsr3.rs commits every
    // command buffer with waitUntilCompleted).
    //
    // TEARDOWN IS STRICTLY INSIDE-OUT: the FFX context first, then the
    // temporals it was registered against, then the backend it calls back into.
    // The middle one is the ordering that is easy to get backwards, because
    // releasing our own textures first LOOKS independent — FFX's release path
    // does null every external resource identifier before it touches anything
    // (ffx_fsr3upscaler.cpp's fsr3UpscalerRelease), and fpUnregisterResources
    // already ran at the end of the last dispatch, so the dynamic slots holding
    // these three are empty by then. That makes the inverted order safe in
    // FACT, as a property of a frozen SDK's internals rather than of this file.
    // This order is safe by CONSTRUCTION, and costs nothing.
    if (s->ffx_alive) ffxFsr3UpscalerContextDestroy(&s->ffx);
    if (s->dilated_depth.tex) [s->dilated_depth.tex release];
    if (s->dilated_motion.tex) [s->dilated_motion.tex release];
    if (s->recon_prev_depth.tex) [s->recon_prev_depth.tex release];
    if (s->recon_prev_depth.backing) [s->recon_prev_depth.backing release];
    if (s->backend) {
        if (s->backend->initQueue) [s->backend->initQueue release];
        delete s->backend;
    }
    delete s;
}
