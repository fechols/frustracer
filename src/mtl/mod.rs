//! The Metal backend — third peer of `src/gpu/` (D3D12) and `src/vk/` (Vulkan),
//! named for its API exactly as those are.
//!
//! **Scope today: FOUR consumers of one G-buffer — two upscalers, a denoiser
//! and a frame interpolator — plus the cinematic capture stage they feed, and
//! since C2 a fifth thing that is not a consumer at all: the corpus's own
//! kernels, compiled, bound and DISPATCHED.** The G-buffer comes from the CPU
//! renderer (`src/dlss.rs`), which runs on every platform; there is still no
//! Metal TRACER, but the machinery one needs now runs.
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
//! * `mfxdn` — Apple's `MTLFXTemporalDenoisedScaler`, the platform-native
//!   answer to the "NRD for pre-upscale denoising" half of the original macOS
//!   ask. It takes the RAY-RECONSTRUCTION G-buffer — normal, roughness, both
//!   albedos, specular hit distance — which `dlss::GBufs` already fills
//!   unconditionally on every platform, so it needed staging code and no
//!   renderer change at all. macOS 26.0+; gated by `--check-metalfx` X4-X5.
//! * `mfxfi` — Apple's `MTLFXFrameInterpolator`. Wired and dispatching, and
//!   MEASURED to be a passthrough without a presentation path; its header
//!   carries the evidence and X6 reports rather than asserts.
//!
//! **The denoiser is what made a QUALITY claim possible here.** Every gate in
//! `mfx` and `fsr3` scores wiring, because an upscaler has no directional claim
//! — `--check-fsr3`'s own notes record that its quality comparison is
//! report-only, since mean-|d| against a converged reference rewards blur and
//! inverts between scenes. A denoiser's claim has a direction: noise must go
//! DOWN. That is what let X5 settle the one convention Apple does not document
//! (world-space normals: 33.8% less high-frequency content than the plain
//! scaler, against 28.8% for view space) instead of arguing about it.
//!
//! Several arms on identical inputs is not redundancy — it is the instrument.
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
//! Metal's as the two that stay apart — and every arm here is the same one,
//! making the same `replaceRegion` calls against the same texture type. What
//! the sharing buys is the cross-check above: it is only evidence if the two
//! arms provably see the same bytes, and two independent allocation sites
//! agreeing is not a proof. `planes.rs`'s header carries the condition under
//! which it must be un-shared again — the first time it needs a conditional.
//!
//! `planes::Guides` is the five DENOISER planes, and it is a separate struct
//! rather than five `Option`s on `Trio` precisely because of that un-share
//! condition — its own doc gives the argument.
//!
//! `msl`, `bind`, `smoke`, `texprobe` and `mtl4` are the modules here that are
//! not G-buffer consumers. They are the tracer's own rungs rather than more
//! arms beside the others:
//!
//! * `msl` (C1) — the corpus's third code generator, HLSL -> SPIR-V -> MSL ->
//!   `.metallib`, gated by `--check-msl`. It renders nothing and binds nothing
//!   by design. Read its header before touching the arg set: every flag in it
//!   is a measurement, including one that `build.rs` passes for FidelityFX and
//!   this route must not.
//! * `bind` (C2) — how one of those `.metallib`s is BOUND. The argument-buffer
//!   map is derived off the compiled function and cross-checked against an
//!   independent read of the module's own SPIR-V, because it cannot be written
//!   down: spirv-cross assigns `[[id(n)]]` densely and PER ENTRY POINT, so the
//!   three kernels of one file disagree about where the same buffer lives.
//! * `smoke` (C2) — the chain that proves it: seed -> prep -> **indirect**
//!   fill, the same `smoke.hlsl` D3D12 and Vulkan each run as their own first
//!   dispatch. Gated by `--check-mtl`, which is to `--check-msl` what
//!   `--check-vk` is to `--check-spirv`.
//! * `texprobe` (C3) — the list `bind.rs` used to name as out of scope, bound
//!   and DISPATCHED: a texture, samplers, a second and third descriptor set,
//!   and the tier-2 unbounded `texs[]` array. Its subject `mtlbind.hlsl` had
//!   been in the shared corpus and executed by nothing, which is the exact gap
//!   the `samp_lin` miscompile hid in for eight months. `--check-mtl` K6/K7/K8.
//! * `mtl4` (D4) — the same smoke chain through the **Metal 4** API:
//!   `MTL4CommandQueue`, `MTL4CommandBuffer`, `MTL4CommandAllocator`,
//!   `MTL4ArgumentTable` and `MTLResidencySet`. It is what makes `grep -rn MTL4
//!   src/ shim/` answer for the first time. The pipeline objects are SHARED
//!   with `smoke`, the verifier is `smoke::verify` UNCHANGED, and K10 compares
//!   the two paths' readbacks word for word — one shader, one pipeline, two
//!   submission APIs, identical bytes. `--check-mtl` K9/K10.
//!
//!   D4b then gave it the one thing MTL4 removed and D4 left unreplaced: an
//!   ERROR CHANNEL. `MTL4CommitFeedback` is both halves of it — MTL4 has no
//!   synchronous `error` and no `waitUntilCompleted` — so the wait is built ON
//!   the handler rather than beside it, and the shared event is gone. Reach is
//!   structural rather than asserted: nothing else can unblock the submission,
//!   which `FR_MTL4_NO_FEEDBACK` proves by timing out. `--check-mtl` K11.
//!
//! It also retired the premise the tracer plan was written on: **spirv-cross
//! lowers `RayQuery` to Metal**, so `leaf`/`reference`/`leaf_fb`/`hemi_leaf`
//! compile with hardware ray tracing intact and a Metal tracer does not need
//! `--sw-rays`.
//!
//! Nothing here is reachable from `src/gfx/`. The entry points are
//! `--check-fsr3`, `--check-metalfx`, `--check-msl`, `--check-mtl`, and — since
//! B4 — `--cinematic`, whose CPU arm reconstructs through `mfxdn` (or `mfx`)
//! when one is available. That last one is the first thing on this platform
//! that produces a PICTURE rather than a number; `--check-mtl` is the first
//! that runs one of OUR OWN shaders on the GPU.

pub mod bind;
pub mod device;
pub mod fsr3;
pub mod mfx;
pub mod mfxdn;
pub mod mfxfi;
pub mod msl;
pub mod mtl4;
pub mod planes;
pub mod smoke;
pub mod texprobe;
