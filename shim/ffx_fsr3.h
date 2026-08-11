// FidelityFX SDK 1.1.4 — C ABI for frustracer's FSR3 arm (Vulkan + Metal).
//
// Peer of shim/ffx_shim.h, which serves the Windows ffx-api v2.3.0 path. The two
// share no types on purpose: v1.1.x and v2 are different C APIs, not versions of
// one, so a shared header would have to be an abstraction over both and would
// bury exactly the differences that matter (see shim/ffx_fsr3.cpp's header).
//
// Nothing here is loaded at runtime. Unlike every vendor SDK this tree consumes,
// FidelityFX 1.1.x is COMPILED FROM SOURCE and statically linked, so there is no
// DLL to find, no fn-pointer table, and no absent-library degrade at run time —
// the degrade is at BUILD time (`cfg(ffx_fsr3_src)`), and a build without the SDK
// simply has no FSR3.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// FFX_SDK_MAKE_VERSION(major, minor, patch) of the headers these objects were
// compiled against. The gate compares it to the pin.
uint32_t frshim_ffx_sdk_version(void);

// FFX_SDK_DEFAULT_CONTEXT_SIZE as the SDK headers see it, after
// shim/ffx_msvc_compat.h's override. Proves the force-include reached them.
uint32_t frshim_ffx_context_size(void);

#ifdef __cplusplus
}  // extern "C"
#endif
