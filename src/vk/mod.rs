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
//! is one corpus with two code generators: the same concatenated source that
//! `gpu/dxc.rs` compiles to DXIL, `vk/spirv.rs` compiles to SPIR-V with a
//! fixed flag set and zero edits to any `.hlsl`. See that module's header for
//! the flag set, the register-space -> descriptor-set mapping it implies, and
//! the measurement.

pub mod bc7;
pub mod device;
pub mod display;
pub mod fsr3;
pub mod headless;
pub mod layout;
pub mod nrd;
// The window (B6b rung 1). NOT `cfg(unix)` like its siblings: it is the one
// module here that depends on `sdl3`, which is scoped to non-macOS unix in
// Cargo.toml so a macOS build does not compile SDL3 from source for a presenter
// that platform has no tracer to feed.
#[cfg(not(target_os = "macos"))]
pub mod present;
pub mod reflect;
pub mod scene;
pub mod spirv;
pub mod stage;
pub mod swapchain;
pub mod textures;
pub mod tracer;
