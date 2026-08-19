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
//!
//! THE HLSL CORPUS IS SHARED AND LIVES AT `src/shaders/`, not under either
//! backend. It moved out of `src/gpu/shaders/` once the SPIR-V spike measured
//! what it does under `dxc -spirv`: **373 of 373 assembled units compiled and
//! passed `spirv-val`, with zero edits to any `.hlsl`.** So the corpus is not
//! "D3D12 shaders we will have to port" — it is one body of source with two
//! code generators, exactly like every other row in the list above, and
//! `src/gpu/` naming it was a statement about where the compiler lived rather
//! than about the shaders. The Rust that ASSEMBLES it (there is no `#include`
//! anywhere — each kernel is a concatenation plus generated `#define` blocks)
//! followed it into `gfx::shaders`, which is now the ONLY module that names an
//! `.hlsl` file: `src/gpu/` holds zero `include_str!`s, so "where does this
//! kernel's source come from" has exactly one answer.
//! What made the translation free is worth recording so nobody re-derives
//! it: `-fvk-{b,t,u,s}-shift` maps HLSL register spaces onto Vulkan descriptor
//! sets without touching a declaration, and `-fvk-use-dx-layout` keeps the
//! 5632-byte `FrameCb` byte-compatible so ONE Rust packer serves both backends
//! — at the cost of requiring `scalarBlockLayout` on the Vulkan device, which
//! is core in 1.2. The one permanent exception is `workgraph.hlsl`: a
//! `VK_AMDX_shader_enqueue` translation exists, but that extension is a vendor
//! provisional, and the file is already a default-off env lever (`FR_WORKGRAPH`)
//! measured as a wash — so it stays D3D12-only by choice, not by obstacle.

pub mod denoise;
pub mod frame;
pub mod guides;
pub mod hud_frame;
pub mod scene;
pub mod shaders;
pub mod texweb;
pub mod vocab;
