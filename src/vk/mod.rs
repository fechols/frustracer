//! The Vulkan backend.
//!
//! Peer of `src/gpu/` (D3D12), over the shared `src/gfx/` core. The split is
//! the one the whole tree already runs on — ONE source of truth for the math
//! and the vocabulary, one implementation of the plumbing per API — so what
//! lives here is *recording*: devices, queues, memory, descriptor sets,
//! pipelines, command buffers. Kernel assembly, the constant-buffer layout,
//! the FLAG_* vocabulary and every tuning knob come from `gfx::` and are
//! shared byte-for-byte with the D3D12 backend rather than mirrored.
//!
//! THE HLSL IS NOT PORTED, AND THAT IS MEASURED, NOT HOPED FOR. `src/shaders/`
//! is one corpus with THREE code generators now: the same concatenated source
//! that `gpu/dxc.rs` compiles to DXIL, `crate::spirv` compiles to SPIR-V with a
//! fixed flag set and zero edits to any `.hlsl` — and `mtl::msl` carries that
//! SPIR-V on to `.metallib` through spirv-cross. See `crate::spirv`'s header
//! for the flag set, the register-space -> descriptor-set mapping it implies,
//! and the measurement.

pub mod bc7;
pub mod device;
pub mod display;
pub mod fsr3;
pub mod headless;
// The HUD's GPU half (B6b rung 4): the overlay image and its dirty-rect
// uploads; the composite pipeline is `display`'s. cfg-free like `display`,
// because its wire (`gfx::hud_frame`) is, and V21 drives it with no Slint.
pub mod hud;
pub mod layout;
pub mod nrd;
// The window (B6b rung 1). NOT `cfg(unix)` like its siblings: it is the one
// module here that depends on `sdl3`, which is scoped to non-macOS unix in
// Cargo.toml so a macOS build does not compile SDL3 from source for a presenter
// that platform has no tracer to feed.
#[cfg(not(target_os = "macos"))]
pub mod present;
// `reflect` MOVED OUT for the same reason `spirv` below did, and is re-exported
// on the same terms: it names no `ash` type, its consumers are Vulkan, Metal
// AND the device-free `--check-spirv` S0, and that last one now runs on Windows
// where this backend does not exist. See `crate::reflect`'s header.
pub(crate) use crate::reflect;
pub mod scene;
// `spirv` MOVED OUT of this backend (it is `crate::spirv` now) and is
// re-exported here so every `vk::spirv::…` call site is untouched — the
// items-move-rather-than-being-copied rule `gfx/mod.rs` states, applied to a
// module that was misfiled rather than shared. It names no `ash` type and
// never did: it is the corpus's second CODE GENERATOR, and since the Metal
// backend's `--check-msl` consumes the same SPIR-V it produces, leaving it
// under `vk/` would have made the Metal gate import the Vulkan backend.
pub(crate) use crate::spirv;
pub mod stage;
pub mod swapchain;
pub mod textures;
pub mod tracer;
