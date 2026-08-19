//! The WebGPU backend (Stage C of the browser port) — the `wgpu` crate as
//! the host API, so the recorder that will drive the browser ALSO runs
//! natively: over Vulkan on the CI runner (lavapipe, the `--check-vk`
//! recipe), over Vulkan/D3D12/Metal on real boxes. Raw web-sys would have
//! left the browser backend as the only untestable backend in the tree.
//!
//! Today this is Stage C1: the device probe and the headless smoke —
//! `--check-wgpu`, stages J0..J3 (`J*` per docs/history/README.md's gate
//! map; `U*` was the obvious pick and is taken by `--check-fsr3`). The
//! smoke is `gfx::shaders::SMOKE_HLSL`, the SAME kernel `--check-gpu`,
//! `--check-vk` and `--check-mtl` prove — "three backends, one kernel"
//! becomes four — but routed through the BROWSER's real shader chain:
//! DXC -> SPIR-V (vulkan1.1) -> `wgsl::normalize` -> `wgsl::spv_to_wgsl`
//! -> a `wgpu::ShaderModule` from the WGSL text. `--check-wgsl` proves the
//! corpus VALIDATES; this proves the translator's output EXECUTES, and
//! those are different claims.
//!
//! The directory is `webgpu`, not `wgpu`: `mod wgpu` would shadow the crate
//! name inside the module and force `::wgpu::` on every path.
//!
//! NOTHING here is reused from `vk/` — `mod vk` is cfg(unix) and this
//! module runs on Windows too, so the pick machinery and the error type are
//! textual twins of vk::device's, not imports. The shared seams are the
//! backend-neutral ones: `crate::spirv` (DXC + the register-shift rule) and
//! `crate::wgsl` (the naga chain).

pub mod device;
pub mod headless;
