// See ngx_shared.h. Extracted verbatim from dlssg_shim.cpp when the DLSSD
// (ray reconstruction) consumer arrived — the refcount was built for exactly
// this second consumer.

#include "ngx_shared.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>

namespace {

constexpr const char* kProjectId     = "b3c1a4d8-52fe-4a0f-9d31-8e7c6f2ab914";
constexpr const char* kEngineVersion = "0.1.0";

std::mutex    g_ngx_mutex;
int           g_ngx_refcount = 0;
ID3D12Device* g_ngx_device   = nullptr;

}  // namespace

int ngx_shared_init(ID3D12Device* device) {
    std::lock_guard<std::mutex> lk(g_ngx_mutex);
    if (g_ngx_refcount > 0) {
        g_ngx_refcount++;
        return 0;
    }
    // NGX wants a WRITABLE app-data path (logs/model cache) — a null path
    // fails with 0xBAD0000F FAIL_UnableToWriteToAppDataPath (measured).
    std::wstring app_data;
    if (const char* lad = std::getenv("LOCALAPPDATA")) {
        std::error_code ec;
        std::filesystem::path p = std::filesystem::path(lad) / "frustracer" / "ngx";
        std::filesystem::create_directories(p, ec);
        if (!ec) app_data = p.wstring();
    }
    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init_with_ProjectID(
        kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, kEngineVersion,
        app_data.empty() ? nullptr : app_data.c_str(), device);
    if (NVSDK_NGX_FAILED(r)) {
        std::fprintf(stderr, "[fr-ngx] NGX D3D12 init failed: 0x%08X\n", (unsigned)r);
        return -1;
    }
    g_ngx_device   = device;
    g_ngx_refcount = 1;
    return 0;
}

void ngx_shared_shutdown() {
    std::lock_guard<std::mutex> lk(g_ngx_mutex);
    if (g_ngx_refcount <= 0) return;
    if (--g_ngx_refcount == 0 && g_ngx_device) {
        NVSDK_NGX_D3D12_Shutdown1(g_ngx_device);
        g_ngx_device = nullptr;
    }
}
