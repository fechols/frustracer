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
pub mod layout;
pub mod nrd;
pub mod reflect;
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
pub mod textures;
pub mod tracer;
