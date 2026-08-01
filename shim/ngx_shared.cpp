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
        if (device != g_ngx_device) {
            // One NGX init per process, keyed to the FIRST consumer's device;
            // a second device would silently ride (and later Shutdown1) the
            // wrong one — the cross-keying class this module exists to
            // prevent. Loud because today's consumers share one device by
            // construction; if this ever prints, that assumption broke.
            std::fprintf(stderr,
                "[fr-ngx] WARNING: consumer passed a different ID3D12Device "
                "(%p, NGX keyed to %p) — sharing the existing init\n",
                static_cast<void*>(device), static_cast<void*>(g_ngx_device));
        }
        g_ngx_refcount++;
        return 0;
    }
    // NGX wants a WRITABLE app-data path (logs/model cache) — a null path
    // fails with 0xBAD0000F FAIL_UnableToWriteToAppDataPath (measured).
    // Read the env var WIDE: getenv is the ANSI codepage, which mangles a
    // non-ASCII username into a path create_directories then fails on.
    std::wstring app_data;
    if (const wchar_t* lad = _wgetenv(L"LOCALAPPDATA")) {
        std::error_code ec;
        std::filesystem::path p = std::filesystem::path(lad) / L"frustracer" / L"ngx";
        std::filesystem::create_directories(p, ec);
        if (!ec) app_data = p.wstring();
    }
    if (app_data.empty()) {
        // The null fallback is the GUARANTEED 0xBAD0000F failure — name the
        // real cause here rather than letting the generic init line carry it.
        std::fprintf(stderr,
            "[fr-ngx] no writable %%LOCALAPPDATA%%\\frustracer\\ngx (env var "
            "missing or directory creation failed) — NGX init will fail\n");
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
