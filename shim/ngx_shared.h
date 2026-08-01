// Refcounted process-wide raw-NGX init, shared by every raw-NGX consumer
// (dlssg_shim = frame generation, dlssd_shim = ray reconstruction). One NGX
// init per device: two differently-keyed NGX inits on one device silently
// break each other — whichever runs second sees the first's app identity and
// its own feature snippet stays unloaded (the quinlight lesson). Every
// consumer takes a ref through ngx_shared_init and releases it through
// ngx_shared_shutdown, so teardown order between consumers cannot matter:
// Shutdown1 runs only when the LAST ref drops.
//
// Internal C++ linkage — both consumers are C++ TUs in one cc::Build archive.

#pragma once

struct ID3D12Device;

// Refcounted NVSDK_NGX_D3D12_Init_with_ProjectID (writable %LOCALAPPDATA%
// app-data path — a null path fails with 0xBAD0000F). Returns 0 on success,
// -1 on init failure. A refcount > 0 ignores `device` (one device per
// process is the operating assumption of everything above this).
int ngx_shared_init(ID3D12Device* device);

// Drops one ref; NVSDK_NGX_D3D12_Shutdown1 when the count reaches zero.
void ngx_shared_shutdown();
