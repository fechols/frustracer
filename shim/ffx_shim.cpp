// See ffx_shim.h for the design rationale. This TU is the only place the real
// ffx-api structs exist; keep every struct construction here.

#include "ffx_shim.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <cstring>
#include <string>

#include "../SDKs/fidelityfx-sdk/api/include/ffx_api.h"
#include "../SDKs/fidelityfx-sdk/api/include/ffx_api_types.h"
#include "../SDKs/fidelityfx-sdk/api/include/dx12/ffx_api_dx12.h"
#include "../SDKs/fidelityfx-sdk/denoisers/include/ffx_denoiser.h"
#include "../SDKs/fidelityfx-sdk/upscalers/include/ffx_upscale.h"

namespace {

struct Api {
    HMODULE dll = nullptr;
    PfnFfxCreateContext  create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxConfigure      configure = nullptr;
    PfnFfxQuery          query = nullptr;
    PfnFfxDispatch       dispatch = nullptr;
};
Api g_api;

FfxShimLogCb g_log_cb = nullptr;

void log_trampoline(uint32_t type, const wchar_t* msg) {
    if (g_log_cb) g_log_cb(type, msg);
}

// FfxApiResource from a shim (resource, state) pair. Null resource yields the
// zero-initialized FfxApiResource the headers document for absent inputs.
FfxApiResource shim_res(const FfxShimRes& r) {
    if (!r.resource) return FfxApiResource{};
    return ffxApiGetResourceDX12(static_cast<ID3D12Resource*>(r.resource), r.state);
}

} // namespace

extern "C" {

int32_t ffxshim_load(const wchar_t* loader_dll_path) {
    if (!loader_dll_path) return FFXSHIM_ERR_BAD_ARG;
    if (g_api.dll) return FFX_API_RETURN_OK; // idempotent
    // Preload the provider DLLs by ABSOLUTE path from the loader's directory
    // (the OIDN precedent: LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR only covers the
    // loader's STATIC imports — its runtime LoadLibrary("amd_fidelityfx_
    // <provider>_dx12.dll") calls search the process default order, which
    // does NOT include the loader's own directory, and every provider then
    // silently fails to resolve = NO_PROVIDER for every effect. A module
    // already loaded resolves by name from the module list, so preloading
    // makes the loader's name-based lookups land). Providers are globbed
    // rather than hardcoded so a drop that adds or renames a provider DLL
    // keeps resolving, and paths are dynamic std::wstring — a deep checkout
    // must never MAX_PATH-truncate into silent no-op preloads.
    {
        std::wstring dir(loader_dll_path);
        const size_t slash = dir.find_last_of(L"\\/");
        if (slash != std::wstring::npos) {
            const std::wstring loader_name = dir.substr(slash + 1);
            dir.resize(slash + 1);
            WIN32_FIND_DATAW fd;
            HANDLE find = FindFirstFileW((dir + L"amd_fidelityfx_*_dx12.dll").c_str(), &fd);
            if (find != INVALID_HANDLE_VALUE) {
                do {
                    if (_wcsicmp(fd.cFileName, loader_name.c_str()) == 0) continue;
                    LoadLibraryExW((dir + fd.cFileName).c_str(), nullptr,
                                   LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
                } while (FindNextFileW(find, &fd));
                FindClose(find);
            }
        }
    }
    HMODULE dll = LoadLibraryExW(loader_dll_path, nullptr,
                                 LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!dll) return FFXSHIM_ERR_LOAD_LIBRARY;
    Api api;
    api.dll = dll;
    api.create    = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(dll, "ffxCreateContext"));
    api.destroy   = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(dll, "ffxDestroyContext"));
    api.configure = reinterpret_cast<PfnFfxConfigure>(GetProcAddress(dll, "ffxConfigure"));
    api.query     = reinterpret_cast<PfnFfxQuery>(GetProcAddress(dll, "ffxQuery"));
    api.dispatch  = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(dll, "ffxDispatch"));
    if (!api.create || !api.destroy || !api.configure || !api.query || !api.dispatch) {
        FreeLibrary(dll);
        return FFXSHIM_ERR_GET_PROC;
    }
    g_api = api;
    return FFX_API_RETURN_OK;
}

void ffxshim_unload(void) {
    if (g_api.dll) FreeLibrary(g_api.dll);
    g_api = Api{};
    g_log_cb = nullptr;
}

int32_t ffxshim_set_debug(uint64_t effect_id, FfxShimLogCb cb, uint32_t level) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    g_log_cb = cb;
    ffxConfigureDescGlobalDebug desc{};
    desc.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG;
    desc.effectId    = effect_id;
    desc.fpMessage   = cb ? &log_trampoline : nullptr;
    desc.debugLevel  = level;
    return static_cast<int32_t>(g_api.configure(nullptr, &desc.header));
}

int32_t ffxshim_query_versions(int32_t is_upscaler, void* device,
                               uint64_t* inout_count, uint64_t* ids, const char** names) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!inout_count) return FFXSHIM_ERR_BAD_ARG;
    ffxQueryDescGetVersions q{};
    q.header.type    = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = is_upscaler ? FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE
                                   : FFX_API_CREATE_CONTEXT_DESC_TYPE_DENOISER;
    q.device       = device;
    q.outputCount  = inout_count;
    q.versionIds   = ids;
    q.versionNames = names;
    return static_cast<int32_t>(g_api.query(nullptr, &q.header));
}

int32_t ffxshim_create_denoiser(void* device, uint32_t max_w, uint32_t max_h,
                                uint32_t signal_flags, uint32_t flags,
                                uint64_t version_id, void** ctx_out) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!device || !ctx_out) return FFXSHIM_ERR_BAD_ARG;

    ffxCreateContextDescDenoiser desc{};
    desc.header.type   = FFX_API_CREATE_CONTEXT_DESC_TYPE_DENOISER;
    desc.version       = FFX_DENOISER_VERSION;
    desc.maxRenderSize = {max_w, max_h};
    desc.signalFlags   = signal_flags;
    desc.checkerboardSignalFlags = 0;
    desc.flags         = flags;

    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device      = static_cast<ID3D12Device*>(device);
    desc.header.pNext   = &backend.header;

    ffxOverrideVersion ver{};
    if (version_id != 0) {
        ver.header.type     = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
        ver.versionId       = version_id;
        backend.header.pNext = &ver.header;
    }

    ffxContext ctx = nullptr;
    ffxReturnCode_t rc = g_api.create(&ctx, &desc.header, nullptr);
    if (rc == FFX_API_RETURN_OK) *ctx_out = ctx;
    return static_cast<int32_t>(rc);
}

int32_t ffxshim_create_upscaler(void* device, uint32_t max_render_w, uint32_t max_render_h,
                                uint32_t out_w, uint32_t out_h, uint32_t flags,
                                uint64_t version_id, void** ctx_out) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!device || !ctx_out) return FFXSHIM_ERR_BAD_ARG;

    ffxCreateContextDescUpscale desc{};
    desc.header.type    = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    desc.flags          = flags;
    desc.maxRenderSize  = {max_render_w, max_render_h};
    desc.maxUpscaleSize = {out_w, out_h};
    desc.fpMessage      = g_log_cb ? &log_trampoline : nullptr;

    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device      = static_cast<ID3D12Device*>(device);
    desc.header.pNext   = &backend.header;

    // Version chaining — the two descs are mutually exclusive. With no
    // override, pin the API version we were compiled against (the header
    // provides an explicit desc for this, unlike the denoiser whose create
    // desc carries the version inline): the pin guards the *default-provider*
    // choice against a future loader substituting a different major. An
    // explicit override IS an exact provider choice already, and the pin desc
    // (which names 4.x) is one a 3.1 provider may reject — so an overridden
    // create chains ffxOverrideVersion alone.
    ffxCreateContextDescUpscaleVersion apiver{};
    ffxOverrideVersion ver{};
    if (version_id != 0) {
        ver.header.type    = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
        ver.versionId      = version_id;
        backend.header.pNext = &ver.header;
    } else {
        apiver.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
        apiver.version      = FFX_UPSCALER_VERSION;
        backend.header.pNext = &apiver.header;
    }

    ffxContext ctx = nullptr;
    ffxReturnCode_t rc = g_api.create(&ctx, &desc.header, nullptr);
    if (rc == FFX_API_RETURN_OK) *ctx_out = ctx;
    return static_cast<int32_t>(rc);
}

int32_t ffxshim_destroy(void** ctx) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!ctx || !*ctx) return FFXSHIM_ERR_BAD_ARG;
    ffxContext c = *ctx;
    ffxReturnCode_t rc = g_api.destroy(&c, nullptr);
    if (rc == FFX_API_RETURN_OK) *ctx = nullptr;
    return static_cast<int32_t>(rc);
}

int32_t ffxshim_upscaler_render_res(void* upscaler_ctx, uint32_t display_w, uint32_t display_h,
                                    uint32_t quality_mode, uint32_t* out_rw, uint32_t* out_rh) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!upscaler_ctx || !out_rw || !out_rh) return FFXSHIM_ERR_BAD_ARG;
    ffxQueryDescUpscaleGetRenderResolutionFromQualityMode q{};
    q.header.type      = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE;
    q.displayWidth     = display_w;
    q.displayHeight    = display_h;
    q.qualityMode      = quality_mode;
    q.pOutRenderWidth  = out_rw;
    q.pOutRenderHeight = out_rh;
    ffxContext ctx = upscaler_ctx;
    return static_cast<int32_t>(g_api.query(&ctx, &q.header));
}

int32_t ffxshim_denoiser_kv(void* denoiser_ctx, uint64_t key, uint64_t count, const void* data) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!denoiser_ctx) return FFXSHIM_ERR_BAD_ARG;
    ffxConfigureDescDenoiserKeyValue desc{};
    desc.header.type = FFX_API_CONFIGURE_DESC_TYPE_DENOISER_KEYVALUE;
    desc.key   = key;
    desc.count = count;
    desc.data  = data;
    ffxContext ctx = denoiser_ctx;
    return static_cast<int32_t>(g_api.configure(&ctx, &desc.header));
}

int32_t ffxshim_denoise(void* denoiser_ctx, const FfxShimDenoiseDesc* d) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!denoiser_ctx || !d || !d->cmdlist) return FFXSHIM_ERR_BAD_ARG;

    ffxDispatchDescDenoiser common{};
    common.header.type    = FFX_API_DISPATCH_DESC_TYPE_DENOISER;
    common.commandList    = d->cmdlist;
    common.linearDepth    = shim_res(d->linear_depth);
    common.motionVectors  = shim_res(d->motion_vectors);
    common.normals        = shim_res(d->normals);
    common.specularAlbedo = shim_res(d->specular_albedo);
    common.diffuseAlbedo  = shim_res(d->diffuse_albedo);
    common.motionVectorScale = {d->mv_scale[0], d->mv_scale[1], d->mv_scale[2]};
    common.jitterOffsets     = {d->jitter[0], d->jitter[1]};
    common.cameraPositionDelta = {d->cam_pos_delta[0], d->cam_pos_delta[1], d->cam_pos_delta[2]};
    static_assert(sizeof(common.view) == 16 * sizeof(float), "FfxApiMatrix4x4 layout");
    std::memcpy(&common.view, d->view, sizeof(common.view));
    std::memcpy(&common.projection, d->projection, sizeof(common.projection));
    common.linearDepthBounds = {d->depth_bounds_min, d->depth_bounds_max};
    common.renderSize = {d->render_w, d->render_h};
    common.frameIndex = d->frame_index;
    common.flags = (d->reset ? FFX_DENOISER_DISPATCH_RESET : 0u)
                 | (d->non_gamma_albedo ? FFX_DENOISER_DISPATCH_NON_GAMMA_ALBEDO : 0u);

    // Per-signal descs chain behind the common desc; signal set must match
    // the context's creation signalFlags exactly.
    ffxDispatchDescDenoiserDirectDiffuse dd{};
    dd.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DIRECT_DIFFUSE;
    dd.signal.input  = shim_res(d->dd_in);
    dd.signal.output = shim_res(d->dd_out);
    dd.signal.checkerboardOrigin = 0;

    ffxDispatchDescDenoiserDirectSpecular ds{};
    ds.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DIRECT_SPECULAR;
    ds.signal.input  = shim_res(d->ds_in);
    ds.signal.output = shim_res(d->ds_out);
    ds.signal.checkerboardOrigin = 0;

    common.header.pNext = &dd.header;
    dd.header.pNext     = &ds.header;

    ffxContext ctx = denoiser_ctx;
    return static_cast<int32_t>(g_api.dispatch(&ctx, &common.header));
}

int32_t ffxshim_upscale(void* upscaler_ctx, const FfxShimUpscaleDesc* d) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!upscaler_ctx || !d || !d->cmdlist) return FFXSHIM_ERR_BAD_ARG;

    ffxDispatchDescUpscale desc{};
    desc.header.type   = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    desc.commandList   = d->cmdlist;
    desc.color         = shim_res(d->color);
    desc.depth         = shim_res(d->depth);
    desc.motionVectors = shim_res(d->motion_vectors);
    desc.exposure      = FfxApiResource{};                    // auto-exposure off, pre_exposure below
    desc.reactive      = FfxApiResource{};
    desc.transparencyAndComposition = FfxApiResource{};
    desc.output        = shim_res(d->output);
    desc.jitterOffset      = {d->jitter[0], d->jitter[1]};
    desc.motionVectorScale = {d->mv_scale[0], d->mv_scale[1]};
    desc.renderSize        = {d->render_w, d->render_h};
    desc.upscaleSize       = {d->out_w, d->out_h};
    desc.enableSharpening  = d->enable_sharpening != 0;
    desc.sharpness         = d->sharpness;
    desc.frameTimeDelta    = d->frame_time_delta_ms;
    desc.preExposure       = d->pre_exposure > 0.0f ? d->pre_exposure : 1.0f;
    desc.reset             = d->reset != 0;
    desc.cameraNear        = d->cam_near;
    desc.cameraFar         = d->cam_far;
    desc.cameraFovAngleVertical = d->cam_fovy;
    desc.viewSpaceToMetersFactor = d->view_space_to_meters > 0.0f ? d->view_space_to_meters : 1.0f;
    desc.flags             = d->flags;

    ffxContext ctx = upscaler_ctx;
    return static_cast<int32_t>(g_api.dispatch(&ctx, &desc.header));
}

} // extern "C"
