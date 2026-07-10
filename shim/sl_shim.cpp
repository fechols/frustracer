// Streamline shim implementation. See sl_shim.h for the design rationale.
//
// sl.interposer.dll is loaded dynamically (never linked), so the inline
// helpers in sl_dlss_d.h — which reference the link-time slGetFeatureFunction
// symbol — must NOT be used; per-feature functions are fetched by name
// through our own function-pointer table (ManualHooking guide §1.1).

#include "sl_shim.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>

// sl_dlss_d.h's inline helpers declare `slGetFeatureFunction` as an extern
// symbol we deliberately don't link. Only the types are needed; declaring the
// include AFTER a macro rename would be fragile — instead just include it:
// the inline helpers are never instantiated, so no link error can occur.
#include <sl_dlss_d.h>

#include <string>
#include <vector>
#include <cstring>

namespace {

HMODULE g_interposer = nullptr;
SlShimLogCb g_log_cb = nullptr;

// Core function-pointer table (PFun_* are function types; add pointers).
PFun_slInit* p_slInit = nullptr;
PFun_slShutdown* p_slShutdown = nullptr;
PFun_slIsFeatureSupported* p_slIsFeatureSupported = nullptr;
PFun_slSetD3DDevice* p_slSetD3DDevice = nullptr;
PFun_slUpgradeInterface* p_slUpgradeInterface = nullptr;
PFun_slGetNativeInterface* p_slGetNativeInterface = nullptr;
PFun_slGetFeatureFunction* p_slGetFeatureFunction = nullptr;
PFun_slGetNewFrameToken* p_slGetNewFrameToken = nullptr;
PFun_slSetConstants* p_slSetConstants = nullptr;
PFun_slSetTagForFrame* p_slSetTagForFrame = nullptr;
PFun_slEvaluateFeature* p_slEvaluateFeature = nullptr;
PFun_slAllocateResources* p_slAllocateResources = nullptr;
PFun_slFreeResources* p_slFreeResources = nullptr;

// DLSS-D functions, fetched by name after the device is set (lazy).
PFun_slDLSSDGetOptimalSettings* p_slDLSSDGetOptimalSettings = nullptr;
PFun_slDLSSDSetOptions* p_slDLSSDSetOptions = nullptr;
PFun_slDLSSDGetState* p_slDLSSDGetState = nullptr;

// Preferences keeps pointers alive past slInit? SL copies the struct, but be
// conservative: keep everything it points at in static storage.
std::wstring g_plugins_path;
const wchar_t* g_plugins_paths[1] = {};
std::vector<sl::Feature> g_features;

void log_trampoline(sl::LogType type, const char* msg) {
    if (g_log_cb) g_log_cb(static_cast<uint32_t>(type), msg);
}

int32_t to_i32(sl::Result r) { return static_cast<int32_t>(r); }

sl::Boolean to_bool(int32_t v) { return v ? sl::Boolean::eTrue : sl::Boolean::eFalse; }

void copy_mat(sl::float4x4& dst, const float m[16]) {
    // Both sides are row-major float[16] at this boundary.
    std::memcpy(&dst, m, sizeof(float) * 16);
}

// Fetch a per-feature function; returns eErrorFeatureMissing-shaped failure
// as the raw sl::Result from slGetFeatureFunction.
template <typename T>
int32_t fetch(sl::Feature feature, const char* name, T*& out) {
    if (out) return 0;
    if (!p_slGetFeatureFunction) return -1003; // shim: init not called

    void* fn = nullptr;
    sl::Result r = p_slGetFeatureFunction(feature, name, fn);
    if (r != sl::Result::eOk) return to_i32(r);
    out = reinterpret_cast<T*>(fn);
    return 0;
}

void build_dlssd_options(const SlShimDlssdOptions* o, sl::DLSSDOptions& opt) {
    opt.mode = static_cast<sl::DLSSMode>(o->mode);
    opt.outputWidth = o->output_width;
    opt.outputHeight = o->output_height;
    // RR requires an HDR pipeline; the pre-tonemap linear radiance we feed it
    // is exactly that, so this is forced rather than configurable.
    opt.colorBuffersHDR = sl::Boolean::eTrue;
    opt.normalRoughnessMode = o->normal_roughness_packed
                                  ? sl::DLSSDNormalRoughnessMode::ePacked
                                  : sl::DLSSDNormalRoughnessMode::eUnpacked;
    if (o->use_camera_matrices) {
        copy_mat(opt.worldToCameraView, o->world_to_view);
        copy_mat(opt.cameraViewToWorld, o->view_to_world);
    }
    const auto preset = static_cast<sl::DLSSDPreset>(o->preset);
    opt.dlaaPreset = preset;
    opt.qualityPreset = preset;
    opt.balancedPreset = preset;
    opt.performancePreset = preset;
    opt.ultraPerformancePreset = preset;
    opt.ultraQualityPreset = preset;
}

} // namespace

extern "C" {

int32_t slshim_load_and_init(const SlShimInitDesc* desc) {
    if (g_interposer) return 0; // already initialized
    g_interposer = LoadLibraryW(desc->interposer_path);
    if (!g_interposer) return -1000; // distinct from any sl::Result

    #define SLSHIM_RESOLVE(name)                                                     \
        p_##name = reinterpret_cast<PFun_##name*>(GetProcAddress(g_interposer, #name)); \
        if (!p_##name) return -1001;
    SLSHIM_RESOLVE(slInit)
    SLSHIM_RESOLVE(slShutdown)
    SLSHIM_RESOLVE(slIsFeatureSupported)
    SLSHIM_RESOLVE(slSetD3DDevice)
    SLSHIM_RESOLVE(slUpgradeInterface)
    SLSHIM_RESOLVE(slGetNativeInterface)
    SLSHIM_RESOLVE(slGetFeatureFunction)
    SLSHIM_RESOLVE(slGetNewFrameToken)
    SLSHIM_RESOLVE(slSetConstants)
    SLSHIM_RESOLVE(slSetTagForFrame)
    SLSHIM_RESOLVE(slEvaluateFeature)
    SLSHIM_RESOLVE(slAllocateResources)
    SLSHIM_RESOLVE(slFreeResources)
    #undef SLSHIM_RESOLVE

    g_log_cb = desc->log_cb;
    g_plugins_path = desc->plugins_path ? desc->plugins_path : L"";
    g_plugins_paths[0] = g_plugins_path.c_str();
    g_features.assign(desc->features, desc->features + desc->num_features);

    sl::Preferences pref{};
    pref.showConsole = desc->show_console != 0;
    pref.logLevel = static_cast<sl::LogLevel>(desc->log_level);
    pref.pathsToPlugins = g_plugins_paths;
    pref.numPathsToPlugins = g_plugins_path.empty() ? 0 : 1;
    pref.logMessageCallback = desc->log_cb ? &log_trampoline : nullptr;
    // Manual hooking + frame-based tagging; keep the default CL-state-
    // tracking opt-out (we re-bind all state after evaluate anyway); no OTA
    // so dev builds are reproducible.
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                 sl::PreferenceFlags::eUseManualHooking |
                 sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    pref.featuresToLoad = g_features.data();
    pref.numFeaturesToLoad = static_cast<uint32_t>(g_features.size());
    pref.applicationId = desc->app_id;
    pref.renderAPI = sl::RenderAPI::eD3D12;

    return to_i32(p_slInit(pref, sl::kSDKVersion));
}

int32_t slshim_shutdown(void) {
    int32_t r = 0;
    if (p_slShutdown) r = to_i32(p_slShutdown());
    if (g_interposer) {
        FreeLibrary(g_interposer);
        g_interposer = nullptr;
    }
    p_slDLSSDGetOptimalSettings = nullptr;
    p_slDLSSDSetOptions = nullptr;
    p_slDLSSDGetState = nullptr;
    return r;
}

int32_t slshim_is_feature_supported(uint32_t feature, const void* luid, uint32_t luid_size) {
    sl::AdapterInfo info{};
    info.deviceLUID = const_cast<uint8_t*>(static_cast<const uint8_t*>(luid));
    info.deviceLUIDSizeInBytes = luid_size;
    return to_i32(p_slIsFeatureSupported(static_cast<sl::Feature>(feature), info));
}

int32_t slshim_set_d3d_device(void* native_d3d12_device) {
    return to_i32(p_slSetD3DDevice(native_d3d12_device));
}

int32_t slshim_upgrade_interface(void** iface_inout) {
    return to_i32(p_slUpgradeInterface(iface_inout));
}

int32_t slshim_get_native_interface(void* proxy, void** native_out) {
    return to_i32(p_slGetNativeInterface(proxy, native_out));
}

int32_t slshim_dlssd_get_optimal_settings(const SlShimDlssdOptions* o, SlShimDlssdOptimal* out) {
    int32_t f = fetch(sl::kFeatureDLSS_RR, "slDLSSDGetOptimalSettings", p_slDLSSDGetOptimalSettings);
    if (f) return f;
    sl::DLSSDOptions opt{};
    build_dlssd_options(o, opt);
    sl::DLSSDOptimalSettings s{};
    sl::Result r = p_slDLSSDGetOptimalSettings(opt, s);
    if (r != sl::Result::eOk) return to_i32(r);
    out->render_width = s.optimalRenderWidth;
    out->render_height = s.optimalRenderHeight;
    out->sharpness = s.optimalSharpness;
    out->render_width_min = s.renderWidthMin;
    out->render_height_min = s.renderHeightMin;
    out->render_width_max = s.renderWidthMax;
    out->render_height_max = s.renderHeightMax;
    return 0;
}

int32_t slshim_dlssd_set_options(uint32_t viewport, const SlShimDlssdOptions* o) {
    int32_t f = fetch(sl::kFeatureDLSS_RR, "slDLSSDSetOptions", p_slDLSSDSetOptions);
    if (f) return f;
    sl::DLSSDOptions opt{};
    build_dlssd_options(o, opt);
    return to_i32(p_slDLSSDSetOptions(sl::ViewportHandle(viewport), opt));
}

int32_t slshim_dlssd_get_state(uint32_t viewport, uint64_t* vram_bytes_out) {
    int32_t f = fetch(sl::kFeatureDLSS_RR, "slDLSSDGetState", p_slDLSSDGetState);
    if (f) return f;
    sl::DLSSDState st{};
    sl::Result r = p_slDLSSDGetState(sl::ViewportHandle(viewport), st);
    if (r != sl::Result::eOk) return to_i32(r);
    *vram_bytes_out = st.estimatedVRAMUsageInBytes;
    return 0;
}

int32_t slshim_allocate_resources(void* cmdlist, uint32_t feature, uint32_t viewport) {
    return to_i32(p_slAllocateResources(static_cast<sl::CommandBuffer*>(cmdlist),
                                        static_cast<sl::Feature>(feature),
                                        sl::ViewportHandle(viewport)));
}

int32_t slshim_free_resources(uint32_t feature, uint32_t viewport) {
    return to_i32(p_slFreeResources(static_cast<sl::Feature>(feature), sl::ViewportHandle(viewport)));
}

int32_t slshim_new_frame_token(const uint32_t* frame_index, void** token_out) {
    sl::FrameToken* token = nullptr;
    sl::Result r = p_slGetNewFrameToken(token, frame_index);
    *token_out = token;
    return to_i32(r);
}

int32_t slshim_set_constants(void* token, uint32_t viewport, const SlShimConstants* c) {
    sl::Constants k{};
    copy_mat(k.cameraViewToClip, c->view_to_clip);
    copy_mat(k.clipToCameraView, c->clip_to_view);
    copy_mat(k.clipToPrevClip, c->clip_to_prev_clip);
    copy_mat(k.prevClipToClip, c->prev_clip_to_clip);
    k.jitterOffset = {c->jitter[0], c->jitter[1]};
    k.mvecScale = {c->mvec_scale[0], c->mvec_scale[1]};
    k.cameraPinholeOffset = {0.0f, 0.0f};
    k.cameraPos = {c->cam_pos[0], c->cam_pos[1], c->cam_pos[2]};
    k.cameraUp = {c->cam_up[0], c->cam_up[1], c->cam_up[2]};
    k.cameraRight = {c->cam_right[0], c->cam_right[1], c->cam_right[2]};
    k.cameraFwd = {c->cam_fwd[0], c->cam_fwd[1], c->cam_fwd[2]};
    k.cameraNear = c->cam_near;
    k.cameraFar = c->cam_far;
    k.cameraFOV = c->cam_fov;
    k.cameraAspectRatio = c->cam_aspect;
    k.motionVectorsInvalidValue = sl::INVALID_FLOAT;
    k.depthInverted = to_bool(c->depth_inverted);
    k.cameraMotionIncluded = to_bool(c->camera_motion_included);
    k.motionVectors3D = to_bool(c->mvec_3d);
    k.reset = to_bool(c->reset);
    k.orthographicProjection = to_bool(c->ortho);
    k.motionVectorsDilated = sl::Boolean::eFalse;
    k.motionVectorsJittered = to_bool(c->mvec_jittered);
    return to_i32(p_slSetConstants(k, *static_cast<sl::FrameToken*>(token),
                                   sl::ViewportHandle(viewport)));
}

int32_t slshim_tag_resources(void* token, uint32_t viewport, const SlShimResourceTag* tags,
                             uint32_t n, void* cmdlist) {
    if (n > SLSHIM_MAX_TAGS) return -1002;
    // Resources must stay alive for the slSetTagForFrame call; the tags
    // reference them by pointer. Stack storage is fine — SL copies what it
    // needs during the call.
    sl::Resource res[SLSHIM_MAX_TAGS];
    sl::ResourceTag rt[SLSHIM_MAX_TAGS];
    for (uint32_t i = 0; i < n; ++i) {
        res[i] = sl::Resource(sl::ResourceType::eTex2d, tags[i].resource, tags[i].state);
        // Dynamic resolution: a nonzero extent names the active sub-rect of
        // the (max-size) resource. Zero extent = whole resource — SL's own
        // null convention, and the pre-DRS behavior. The Extent is copied
        // by value into the tag by the ctor, so a stack temporary is fine.
        if (tags[i].extent_width != 0 && tags[i].extent_height != 0) {
            sl::Extent ext{tags[i].extent_top, tags[i].extent_left, tags[i].extent_width,
                           tags[i].extent_height};
            rt[i] = sl::ResourceTag(&res[i], static_cast<sl::BufferType>(tags[i].buffer_type),
                                    static_cast<sl::ResourceLifecycle>(tags[i].lifecycle), &ext);
        } else {
            rt[i] = sl::ResourceTag(&res[i], static_cast<sl::BufferType>(tags[i].buffer_type),
                                    static_cast<sl::ResourceLifecycle>(tags[i].lifecycle));
        }
    }
    return to_i32(p_slSetTagForFrame(*static_cast<sl::FrameToken*>(token),
                                     sl::ViewportHandle(viewport), rt, n,
                                     static_cast<sl::CommandBuffer*>(cmdlist)));
}

int32_t slshim_evaluate(uint32_t feature, void* token, uint32_t viewport, void* cmdlist) {
    sl::ViewportHandle vp(viewport);
    const sl::BaseStructure* inputs[] = {&vp};
    return to_i32(p_slEvaluateFeature(static_cast<sl::Feature>(feature),
                                      *static_cast<sl::FrameToken*>(token), inputs, 1,
                                      static_cast<sl::CommandBuffer*>(cmdlist)));
}

} // extern "C"
