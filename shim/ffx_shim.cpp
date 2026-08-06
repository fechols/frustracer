// See ffx_shim.h for the design rationale. This TU is the only place the real
// ffx-api structs exist; keep every struct construction here.

#include "ffx_shim.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <cstring>
#include <cstdlib>   // atexit — the post-main probe at the end of this file
#include <string>

#include "../SDKs/fidelityfx-sdk/api/include/ffx_api.h"
#include "../SDKs/fidelityfx-sdk/api/include/ffx_api_types.h"
#include "../SDKs/fidelityfx-sdk/api/include/dx12/ffx_api_dx12.h"
#include "../SDKs/fidelityfx-sdk/denoisers/include/ffx_denoiser.h"
#include "../SDKs/fidelityfx-sdk/upscalers/include/ffx_upscale.h"
#include "../SDKs/fidelityfx-sdk/framegeneration/include/ffx_framegeneration.h"
#include "../SDKs/fidelityfx-sdk/framegeneration/include/dx12/ffx_api_framegeneration_dx12.h"

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

// The luminance patch the FG trampoline applies (see ffxshim_fg_configure).
float g_fg_min_max_luminance[2] = {0.0f, 0.0f};

// The FI swapchain drives generation through this callback at present time,
// handing a dispatch desc it built (presentColor, outputs, frameID, reset,
// backbuffer transfer function). Route it into the FG effect context; patch
// the luminance range when the display probe supplied a real one (the
// swapchain's own value is a guess — it cannot see the panel).
ffxReturnCode_t fg_dispatch_trampoline(ffxDispatchDescFrameGeneration* params, void* uctx) {
    if (g_fg_min_max_luminance[1] > 0.0f) {
        params->minMaxLuminance[0] = g_fg_min_max_luminance[0];
        params->minMaxLuminance[1] = g_fg_min_max_luminance[1];
    }
    ffxContext ctx = uctx;
    return g_api.dispatch(&ctx, &params->header);
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

// The desc is hand-mirrored in ffx_sys.rs (this is the shim's OWN struct, not
// an ffx one — mirroring it is the design). Pin the two places a mismatch
// could hide: `signal_flags` is a u32 wedged between 8-aligned members, so it
// opens a padding hole that both languages must agree on. The Rust side
// asserts the identical literals; a divergence fails one build or the other
// instead of silently shifting every resource pointer.
static_assert(offsetof(FfxShimDenoiseDesc, linear_depth) == 16,
              "FfxShimDenoiseDesc: signal_flags padding hole disagrees with ffx_sys.rs");
static_assert(sizeof(FfxShimDenoiseDesc) == 416,
              "FfxShimDenoiseDesc: size disagrees with ffx_sys.rs");

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

    // Per-signal descs chain behind the common desc. The header requires the
    // chained set to match the context's creation signalFlags EXACTLY (each
    // desc is required "if and only if" its flag is set) — hence the flag word
    // travels on the dispatch desc; the shim keeps no per-context state.
    // All descs are stack locals, alive through g_api.dispatch below.
    ffxDispatchDescDenoiserAmbientOcclusion ao{};
    ao.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_AMBIENT_OCCLUSION;
    ao.signal.input  = shim_res(d->ao_in);
    ao.signal.output = shim_res(d->ao_out);
    ao.signal.checkerboardOrigin = 0;

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

    ffxDispatchDescDenoiserIndirectSpecular is{};
    is.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_INDIRECT_SPECULAR;
    is.signal.input  = shim_res(d->is_in);
    is.signal.output = shim_res(d->is_out);
    is.signal.checkerboardOrigin = 0;

    // Stitch in flag order: whichever signals are subscribed, in one chain.
    ffxDispatchDescHeader* chain[4];
    uint32_t n = 0;
    if (d->signal_flags & FFXSHIM_SIGNAL_AMBIENT_OCCLUSION)  chain[n++] = &ao.header;
    if (d->signal_flags & FFXSHIM_SIGNAL_DIRECT_DIFFUSE)     chain[n++] = &dd.header;
    if (d->signal_flags & FFXSHIM_SIGNAL_DIRECT_SPECULAR)    chain[n++] = &ds.header;
    if (d->signal_flags & FFXSHIM_SIGNAL_INDIRECT_SPECULAR)  chain[n++] = &is.header;
    // A flag with no desc built above (dominant light, indirect diffuse,
    // specular occlusion) would dispatch a set that does not match the
    // creation flags — refuse rather than let the provider misbehave.
    const uint32_t supported = FFXSHIM_SIGNAL_AMBIENT_OCCLUSION | FFXSHIM_SIGNAL_DIRECT_DIFFUSE
                             | FFXSHIM_SIGNAL_DIRECT_SPECULAR | FFXSHIM_SIGNAL_INDIRECT_SPECULAR;
    if (d->signal_flags & ~supported) return FFXSHIM_ERR_BAD_ARG;
    if (n == 0) return FFXSHIM_ERR_BAD_ARG;

    common.header.pNext = chain[0];
    for (uint32_t i = 0; i + 1 < n; ++i) chain[i]->pNext = chain[i + 1];
    chain[n - 1]->pNext = nullptr;

    ffxContext ctx = denoiser_ctx;
    return static_cast<int32_t>(g_api.dispatch(&ctx, &common.header));
}

int32_t ffxshim_preload_dir(const wchar_t* extra_dir) {
    if (!extra_dir) return FFXSHIM_ERR_BAD_ARG;
    // Same glob + absolute-path preload as ffxshim_load's own directory pass,
    // with one addition: skip any basename ALREADY in the module list
    // (GetModuleHandleW). The Windows loader resolves same-basename loads to
    // the existing module anyway, so the skip changes nothing semantically —
    // it exists so the primary directory's DLLs stay authoritative REGARDLESS
    // of whether this runs before or after ffxshim_load, and so the intent is
    // explicit rather than an accident of loader behavior.
    std::wstring dir(extra_dir);
    if (dir.empty()) return FFXSHIM_ERR_BAD_ARG;
    if (dir.back() != L'\\' && dir.back() != L'/') dir += L'\\';
    WIN32_FIND_DATAW fd;
    HANDLE find = FindFirstFileW((dir + L"amd_fidelityfx_*_dx12.dll").c_str(), &fd);
    if (find == INVALID_HANDLE_VALUE) return FFX_API_RETURN_OK; // nothing there = fine
    do {
        if (GetModuleHandleW(fd.cFileName)) continue;
        LoadLibraryExW((dir + fd.cFileName).c_str(), nullptr,
                       LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    } while (FindNextFileW(find, &fd));
    FindClose(find);
    return FFX_API_RETURN_OK;
}

int32_t ffxshim_query_versions_fg(void* device, uint64_t* inout_count,
                                  uint64_t* ids, const char** names) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!inout_count) return FFXSHIM_ERR_BAD_ARG;
    ffxQueryDescGetVersions q{};
    q.header.type    = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    q.device       = device;
    q.outputCount  = inout_count;
    q.versionIds   = ids;
    q.versionNames = names;
    return static_cast<int32_t>(g_api.query(nullptr, &q.header));
}

int32_t ffxshim_fg_swapchain_wrap(void* game_queue, void** inout_swapchain, void** out_sc_ctx) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!game_queue || !inout_swapchain || !*inout_swapchain || !out_sc_ctx)
        return FFXSHIM_ERR_BAD_ARG;

    ffxCreateContextDescFrameGenerationSwapChainWrapDX12 desc{};
    desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_WRAP_DX12;
    desc.swapchain   = reinterpret_cast<IDXGISwapChain4**>(inout_swapchain);
    desc.gameQueue   = static_cast<ID3D12CommandQueue*>(game_queue);

    // The API-version pin (the upscaler-create precedent): omitting it locks
    // newer swapchain APIs out per the header's own comment.
    ffxCreateContextDescFrameGenerationSwapChainVersionDX12 ver{};
    ver.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_DX12;
    ver.version      = FFX_FRAMEGENERATION_SWAPCHAIN_DX12_VERSION;
    desc.header.pNext = &ver.header;

    ffxContext ctx = nullptr;
    ffxReturnCode_t rc = g_api.create(&ctx, &desc.header, nullptr);
    if (rc == FFX_API_RETURN_OK) *out_sc_ctx = ctx;
    return static_cast<int32_t>(rc);
}

int32_t ffxshim_fg_swapchain_wait(void* sc_ctx) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!sc_ctx) return FFXSHIM_ERR_BAD_ARG;
    ffxDispatchDescFrameGenerationSwapChainWaitForPresentsDX12 d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_WAIT_FOR_PRESENTS_DX12;
    ffxContext ctx = sc_ctx;
    return static_cast<int32_t>(g_api.dispatch(&ctx, &d.header));
}

int32_t ffxshim_fg_swapchain_ui(void* sc_ctx, const FfxShimRes* ui, int32_t premul) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!sc_ctx || !ui) return FFXSHIM_ERR_BAD_ARG;
    ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceDX12 desc{};
    desc.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_REGISTERUIRESOURCE_DX12;
    desc.uiResource  = shim_res(*ui);
    desc.flags       = premul ? FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_USE_PREMUL_ALPHA : 0u;
    ffxContext ctx = sc_ctx;
    return static_cast<int32_t>(g_api.configure(&ctx, &desc.header));
}

int32_t ffxshim_create_fg(void* device, uint32_t display_w, uint32_t display_h,
                          uint32_t max_render_w, uint32_t max_render_h,
                          uint32_t backbuffer_format, uint32_t flags,
                          uint64_t version_id, void** out_ctx) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!device || !out_ctx) return FFXSHIM_ERR_BAD_ARG;

    ffxCreateContextDescFrameGeneration desc{};
    desc.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    desc.flags            = flags;
    desc.displaySize      = {display_w, display_h};
    desc.maxRenderSize    = {max_render_w, max_render_h};
    desc.backBufferFormat = backbuffer_format;

    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device      = static_cast<ID3D12Device*>(device);
    desc.header.pNext   = &backend.header;

    // Version chaining. NOT the upscaler's mutual-exclusion rule, which this
    // used to copy verbatim -- FG's two version descs answer DIFFERENT
    // questions and both belong on the chain:
    //   ffxCreateContextDescFrameGenerationVersion is the API level this TU
    //     compiled against, and the runtime REQUIRES it to expose the current
    //     entry points. Omitting it is not silent: the library prints "An
    //     instance of ffxCreateContextDescFrameGenerationVersion must be
    //     attached ... to access new API functions" and the context comes up
    //     on a legacy path.
    //   ffxOverrideVersion picks WHICH enumerated provider backs the context
    //     (fsr::pick_fg_version -- 4.x ML FG for an FSR4 session, 3.1
    //     interpolation otherwise).
    // Choosing a provider says nothing about which API level we speak, so the
    // override chains AHEAD of the pin rather than replacing it.
    ffxCreateContextDescFrameGenerationVersion apiver{};
    ffxOverrideVersion ver{};
    apiver.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
    apiver.version     = FFX_FRAMEGENERATION_VERSION;
    if (version_id != 0) {
        ver.header.type      = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
        ver.versionId        = version_id;
        backend.header.pNext = &ver.header;
        ver.header.pNext     = &apiver.header;
    } else {
        backend.header.pNext = &apiver.header;
    }

    ffxContext ctx = nullptr;
    ffxReturnCode_t rc = g_api.create(&ctx, &desc.header, nullptr);
    if (rc == FFX_API_RETURN_OK) *out_ctx = ctx;
    return static_cast<int32_t>(rc);
}

// The FG twins of the denoise-desc layout pins — ffx_sys.rs asserts the
// identical literals (see the comment there for which holes these guard).
static_assert(offsetof(FfxShimFgConfig, hudless) == 16,
              "FfxShimFgConfig: layout disagrees with ffx_sys.rs");
static_assert(sizeof(FfxShimFgConfig) == 72,
              "FfxShimFgConfig: size disagrees with ffx_sys.rs");
static_assert(offsetof(FfxShimFgPrepare, depth) == 72,
              "FfxShimFgPrepare: layout disagrees with ffx_sys.rs");
static_assert(sizeof(FfxShimFgPrepare) == 152,
              "FfxShimFgPrepare: size disagrees with ffx_sys.rs");

int32_t ffxshim_fg_configure(void* fg_ctx, const FfxShimFgConfig* c) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!fg_ctx || !c || !c->swapchain) return FFXSHIM_ERR_BAD_ARG;

    g_fg_min_max_luminance[0] = c->min_max_luminance[0];
    g_fg_min_max_luminance[1] = c->min_max_luminance[1];

    ffxConfigureDescFrameGeneration desc{};
    desc.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    desc.swapChain   = c->swapchain;
    desc.presentCallback                    = nullptr; // default composition
    desc.presentCallbackUserContext         = nullptr;
    desc.frameGenerationCallback            = &fg_dispatch_trampoline;
    desc.frameGenerationCallbackUserContext = fg_ctx;
    desc.frameGenerationEnabled = c->enabled != 0;
    desc.allowAsyncWorkloads    = c->allow_async != 0;
    desc.HUDLessColor           = shim_res(c->hudless);
    desc.flags                  = c->flags;
    desc.onlyPresentGenerated   = c->only_present_generated != 0;
    desc.generationRect         = {static_cast<int32_t>(c->rect_left),
                                   static_cast<int32_t>(c->rect_top),
                                   static_cast<int32_t>(c->rect_w),
                                   static_cast<int32_t>(c->rect_h)};
    desc.frameID                = c->frame_id;

    ffxContext ctx = fg_ctx;
    return static_cast<int32_t>(g_api.configure(&ctx, &desc.header));
}

int32_t ffxshim_fg_prepare(void* fg_ctx, const FfxShimFgPrepare* p) {
    if (!g_api.dll) return FFXSHIM_ERR_NOT_LOADED;
    if (!fg_ctx || !p || !p->cmdlist) return FFXSHIM_ERR_BAD_ARG;

    ffxDispatchDescFrameGenerationPrepareV2 desc{};
    desc.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
    desc.frameID     = p->frame_id;
    desc.flags       = p->flags;
    desc.commandList = p->cmdlist;
    desc.renderSize  = {p->render_w, p->render_h};
    desc.jitterOffset      = {p->jitter[0], p->jitter[1]};
    desc.motionVectorScale = {p->mv_scale[0], p->mv_scale[1]};
    desc.frameTimeDelta    = p->frame_time_delta_ms;
    desc.reset             = p->reset != 0;
    desc.cameraNear        = p->cam_near;
    desc.cameraFar         = p->cam_far;
    desc.cameraFovAngleVertical  = p->cam_fovy;
    desc.viewSpaceToMetersFactor = p->view_space_to_meters > 0.0f ? p->view_space_to_meters : 1.0f;
    desc.depth         = shim_res(p->depth);
    desc.motionVectors = shim_res(p->motion_vectors);
    std::memcpy(desc.cameraPosition, p->cam_pos,   sizeof(desc.cameraPosition));
    std::memcpy(desc.cameraUp,       p->cam_up,    sizeof(desc.cameraUp));
    std::memcpy(desc.cameraRight,    p->cam_right, sizeof(desc.cameraRight));
    std::memcpy(desc.cameraForward,  p->cam_fwd,   sizeof(desc.cameraForward));

    ffxContext ctx = fg_ctx;
    return static_cast<int32_t>(g_api.dispatch(&ctx, &desc.header));
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

// ---------------------------------------------------------------------------
// Deliberate fault for the crash handler's gate (FR_CRASH_TEST=cpp, see
// src/crash.rs). Nothing else calls it, and nothing arms it by default.
//
// It lives HERE because this TU is the shim built unconditionally on Windows —
// the DLSS shims compile only when the (non-redistributable) SDK is present —
// so a C++ frame is guaranteed to exist in every build. A crash report naming
// these functions with `shim/ffx_shim.cpp:LINE` is the end-to-end proof that
// the Rust code and the cc-built C++ share one frustracer.pdb; without it the
// handler's "prints C++ frames too" claim would be untested.
//
// Two frames, both noinline, so the report has to show a real C++ -> C++ call
// chain rather than a single leaf that could be a coincidence.
// ---------------------------------------------------------------------------
__declspec(noinline) void frshim_crash_test_inner(volatile int* p) {
    *p = 0x5A;  // deliberate null-pointer write
}

__declspec(noinline) void frshim_crash_test(void) {
    frshim_crash_test_inner(nullptr);
}

// --- late-phase hook: AFTER main returns -----------------------------------
//
// Rust cannot register an atexit handler, and the phase after main returns
// (CRT atexit + static destructors, then DLL_PROCESS_DETACH) is exactly the
// window a "did the crash handler outlive the crash?" question is about. So
// the shim owns the registration and calls back into Rust.
//
// atexit handlers run in REVERSE registration order, and crash::install()
// registers this one FIRST — so it runs LAST, which is what makes it a
// meaningful probe of the latest reachable moment.
typedef void (*FrShimLateCb)(void);
static FrShimLateCb g_late_cb = nullptr;

static void frshim_late_trampoline(void) {
    if (g_late_cb) g_late_cb();
}

__declspec(noinline) void frshim_register_atexit(FrShimLateCb cb) {
    g_late_cb = cb;
    atexit(&frshim_late_trampoline);
}

} // extern "C"
