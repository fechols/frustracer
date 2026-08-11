// FidelityFX SDK 1.1.4 — the backend-neutral half of frustracer's FSR3 arm.
//
// WHY A SECOND FIDELITYFX AT ALL: the shipping Windows path is ffx-api v2.3.0
// (shim/ffx_shim.cpp), which reaches FSR3/FSR4 through AMD's signed prebuilt
// DX12 provider DLLs. That generation has no Vulkan backend (its own readme
// lists "Vulkan is currently not supported in SDK" under known issues) and, more
// decisively, v2 REMOVED the `FfxInterface` custom-backend seam — so there is no
// way to write a Metal implementation against it. v1.1.4 is the last generation
// that is MIT source with a stock first-party `ffx_vk` and a seam. Windows keeps
// v2.3.0; Vulkan and Metal use this.
//
// This translation unit holds only what BOTH backends share. The per-backend
// halves (`ffx_vk.cpp` from the SDK for Vulkan; a hand-written `FfxInterface`
// for Metal) land beside it, because each drags in an API's headers and neither
// can compile where the other's are absent.

#include "ffx_fsr3.h"

#include <FidelityFX/host/ffx_interface.h>

extern "C" {

// The COMPILED SDK version, read from the headers the objects were actually
// built against — deliberately not a string the Rust side could hold its own
// copy of. `install-prerequisites.sh`'s FFX_SRC_TAG and the gate's expectation
// are a pin pair (the NRD_TAG / src/nrd.rs shape), and this is what makes a
// silent drift between them fail loudly: fetch a different tag without moving
// the pin and the number coming back stops matching.
uint32_t frshim_ffx_sdk_version(void) {
    return (uint32_t)FFX_SDK_MAKE_VERSION(
        FFX_SDK_VERSION_MAJOR, FFX_SDK_VERSION_MINOR, FFX_SDK_VERSION_PATCH);
}

// The context-size constant AS THE SDK HEADERS SEE IT after shim/ffx_msvc_compat.h
// has had its say. That header raises it from the upstream 128 KB to 1 MB because
// a 4-byte wchar_t (i.e. everywhere that is not MSVC) bloats the private FSR3
// context past the smaller bound. Reporting it lets the gate assert the override
// actually reached the SDK's own headers rather than trusting that a force-include
// landed — the compat header is applied by a build.rs flag, which is exactly the
// kind of wiring that can silently stop being passed.
uint32_t frshim_ffx_context_size(void) {
    return (uint32_t)FFX_SDK_DEFAULT_CONTEXT_SIZE;
}

}  // extern "C"
