//! The Metal backend — third peer of `src/gpu/` (D3D12) and `src/vk/` (Vulkan),
//! named for its API exactly as those are.
//!
//! **Scope today: TWO upscalers over one G-buffer, headless and gated. There is
//! no Metal tracer.** The G-buffer comes from the CPU renderer
//! (`src/dlss.rs`), which runs on every platform.
//!
//! * `fsr3` — FidelityFX 1.1.4, through a hand-written `FfxInterface`
//!   (`shim/ffx_fsr3_metal.mm`) over shaders transpiled to `.metallib` at build
//!   time from the SPIR-V permutations committed under
//!   `SDKs/FidelityFX-SDK-prebuilt/`. Needs neither DXC nor any of our own
//!   HLSL, which is what makes it independent of both the Vulkan port and the
//!   Metal shader route. Gated by `--check-fsr3`.
//! * `mfx` — Apple's `MTLFXTemporalScaler`. A system framework: no shim, no
//!   transpile, no SDK, so it works on a BARE CLONE where the FSR3 arm needs
//!   `install-prerequisites.sh fsr3src` plus spirv-cross and Xcode at build
//!   time. Gated by `--check-metalfx`.
//!
//! Two upscalers on identical inputs is not redundancy — it is the instrument.
//! A single arm's gate can assert magnitudes (energy, history-differs,
//! not-a-bilinear-stretch) and `--check-fsr3` says plainly that no magnitude of
//! its own can tell a correctly-signed jitter from a mirrored one. Correlating
//! the two arms' deviations from a common reference CAN, and that is what
//! settled `mfx::JITTER_SIGN` from a guess into a measurement.
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
//! `planes::Trio` is NOT a counter-example, and the distinction is the API.
//! That rule's unit is the API — its own sentence names D3D12's recording and
//! Metal's as the two that stay apart — and `fsr3` and `mfx` are the same one,
//! making the same `replaceRegion` calls against the same texture type. What
//! the sharing buys is the cross-check above: it is only evidence if the two
//! arms provably see the same bytes, and two independent allocation sites
//! agreeing is not a proof. `planes.rs`'s header carries the condition under
//! which it must be un-shared again — the first time it needs a conditional.
//!
//! Nothing here is reachable from `src/gfx/`, and no interactive path
//! constructs it — `--check-fsr3` is the only entry point.

pub mod device;
pub mod fsr3;
pub mod mfx;
pub mod planes;
