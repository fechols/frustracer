//! The Metal backend — third peer of `src/gpu/` (D3D12) and `src/vk/` (Vulkan),
//! named for its API exactly as those are.
//!
//! **Scope today: FSR3 upscaling, headless and gated. There is no Metal
//! tracer.** The G-buffer this upscales comes from the CPU renderer
//! (`src/dlss.rs`), which runs on every platform, and the FidelityFX shaders
//! come from the SPIR-V permutations committed under
//! `SDKs/FidelityFX-SDK-prebuilt/` and transpiled to `.metallib` at build time.
//! So this module needs neither DXC nor any of our own HLSL — which is what
//! makes it independent of both the Vulkan port and the (unbuilt) Metal shader
//! route.
//!
//! **NO TRAIT, and that is the architecture, not an omission.** `src/gfx/`
//! contains zero traits by design (`gfx/mod.rs:11-19`) and nothing under it may
//! name a backend type. The tree's answer to "two backends share an upscaler"
//! is shared MATH and VOCABULARY with the RECORDING duplicated per API: the
//! three input-plane encodings live in `fsr::stage_*` and are gated on every
//! platform by `--check-fsr`, while the D3D12 recording
//! (`gpu/ffx_up.rs::record_upload`) and the Metal one here stay separate. Two
//! `Fsr3` implementations is the correct shape; a `trait Upscaler` is not.
//!
//! Nothing here is reachable from `src/gfx/`, and no interactive path
//! constructs it — `--check-fsr3` is the only entry point.

pub mod device;
pub mod fsr3;
