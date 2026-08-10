//! The backend-neutral half of the GPU story.
//!
//! `src/gpu/` is the D3D12 backend and `src/vk/` will be the Vulkan one; this
//! module is what they SHARE. The rule is the one the CPU↔GPU split already
//! runs on — `shade.rs` ↔ `shade.hlsli`, `nrd::oracle` ↔ `nrd_bridge.hlsl`,
//! `fsr::split_signals` ↔ `cs_feed_fsr_rr`, `tone.rs` ↔ `tonemap.hlsl`: ONE
//! source of truth for the math and the vocabulary, one implementation of the
//! plumbing per API. D3D12 ↔ Vulkan is that same shape.
//!
//! THE HARD RULE: nothing under `gfx/` may name a backend type. No
//! `windows::`, no `ash::`, no `DXGI_FORMAT`, no `VkFormat`. That is what lets
//! this module compile on every platform — which is not a portability nicety
//! but the reason the headless CPU gates (`--check`, `--check-dlss`,
//! `--check-fsr`, and every `*::self_test`) can run somewhere without a GPU
//! vendor's SDK. A backend-specific detail hanging off a shared type belongs
//! in that backend's file as an `impl` (see `PresentSpace::format`, which lives
//! in `gpu/d3d12.rs` precisely because `DXGI_FORMAT` may not appear here).
//!
//! Items move here rather than being copied: the old home re-exports the new
//! one (`pub use crate::gfx::…`), so every existing call site is untouched and
//! there is still exactly one definition. A second definition that "keeps the
//! backends independent" is the failure mode this module exists to prevent.

pub mod guides;
pub mod vocab;
