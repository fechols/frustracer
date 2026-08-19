# The browser port — WASM + WebGPU (gate prefix: W* for `--check-wgsl`; J* reserved for `--check-wgpu` — U* was the obvious pick and is TAKEN by `--check-fsr3`, see the README's namespace table)

The campaign record for frustracer-in-the-browser: a GitHub Pages-hosted build
running the wavefront tracer on WebGPU, no download. The full staged plan
(shader track W + runtime track A–G) lives with the session that authored it;
this file records what LANDED and what was MEASURED, newest last.

## The shape of the port, decided 2026-08-18

- **The browser render path is `--gpu --sw-rays --no-upscale --frd`** — the
  wavefront tracer over the software BVH (`rt_sw.hlsli` declares no
  acceleration structure), plain presentation, the in-house FRD denoiser.
  Zero DLLs, zero hardware RT, pure compute.
- **Host API: the `wgpu` crate** (Stage C, not yet landed) — the same
  recorder runs natively over Vulkan, so a `--check-wgpu` gate can run on
  llvmpipe in CI, the `--check-vk` recipe. Raw web-sys would leave the
  browser backend as the only untestable backend in the tree.
- **v1 scope: GPU wavefront only.** rayon ≥1.7 degrades to inline-sequential
  on wasm without atomics, so the CPU tracer is a wasm-threads v2 (needs
  SharedArrayBuffer → COOP/COEP → the coi-serviceworker shim on Pages), not
  a blocker.
- **Assets: offline-prebaked bundles** (`--bake-web`, Stage E): fetch + zstd
  (max level — decode speed is level-independent) + upload; no in-browser BVH
  build or mip. Big scenes ride as GitHub Release assets (CORS `*`), small
  ones on Pages.
- **Shipping boundary artifact: SPIR-V blobs + manifest**, translated to WGSL
  at page load by naga (inside wgpu). The wasm build still assembles the HLSL
  corpus (`gfx::shaders` is cfg-free) and hash-compares units against the
  manifest — "one corpus" stays asserted in the browser. `src/wgsl.rs` is the
  one seam; the gate and the page share one lockfile-pinned translator.

## Stage 0 (landed before this file existed)

`4553c34` (0a: `--check-spirv` + `reflect.rs` on Windows), `af225cb`/`e45a19a`
(0b: the flat indirect-args buffer + `ARG_STRIDE`), `5fb02f0` (`FR_NOPRECISE`
priced `precise`-loss: LOW risk, re-run per browser-tier scene). See
shader-toolchains.md.

## Stage A + the wasm compile guard (landed 2026-08-18)

`cargo check --target wasm32-unknown-unknown` is green, and CI's `check-wasm`
job holds it so. What it took, in full:

- `.cargo/config.toml`: `[build] rustflags` → `[target.'cfg(not(wasm32))']`
  (target-cpu=native handed to a wasm compile), plus a
  `[target.wasm32-unknown-unknown]` table carrying
  `--cfg getrandom_backend="wasm_js"`. CAUTION: an env `RUSTFLAGS` replaces
  the target tables wholesale — the check-wasm job must leave it unset.
- `Cargo.toml`: `intel_tex_2` → `cfg(not(wasm32))` (its build script panics
  on target_family wasm; prebuilt x86 ISPC objects); `getrandom/wasm_js`
  under the wasm table (tobj → ahash's entropy source); `naga` as a base dep.
- `build.rs`: wasm early-out BEFORE `require_nrd()` — a bare checkout (no
  submodules) must build for wasm, and the check-wasm job checks out without
  submodules so reordering fails CI. The check-cfg declarations hoisted above
  it (a target that skips the script body still compiles the attributes).
- `src/bc7.rs`: encoder internals cfg'd off wasm; the vocabulary
  (`Bc7Mode`/`should_compress`) compiles everywhere. `src/nrd.rs`: a wasm
  `imp` stub (open = the error message) + the x64 size-assert block gated on
  `target_pointer_width = "64"` (pointers make those 64-bit facts).
  `src/main.rs`: `--check-spirv` fails by name on wasm (exit 2); cinematic's
  ArmPick::Gpu takes the macOS substitution.
- The zstd wasm arm needs a clang that emits wasm32 (ubuntu runners have
  one; a Windows box needs LLVM — found at C:\Program Files\LLVM here).

Only 12 crate errors stood between a never-tried target and green, in 4
files: the module cfg map was already almost right, which is the payoff of
the `cfg(any(unix, windows))`-spelled-deliberately discipline.

## `--check-wgsl` W0–W4 (landed 2026-08-18) and naga's verdict

`src/wgsl.rs` (naga behind one seam, uncfg'd — the page runs it at load) +
`web_units()` (16 units: the 9 trace units minus feed, FRD temporal/blur,
bloom, autoexp, tonemap/blit/hud) + `run_check_wgsl` W0–W4 in main.rs.
`web_defs()` (`WEB` + `ABL_NO_WAVE_OPS`) is PREPENDED at collection — the
native off-state is structural (native assembly untouched by construction),
and the probe-reach question is moot by shape. The wave-ops tooth is
end-to-end: `wgsl::validate` grants an EMPTY capability set, so a unit the
prelude missed fails W4 on its leaked subgroup ops.

**The web arm compiles at `-fspv-target-env=vulkan1.1`** (appended after
`spirv_args()`'s vulkan1.3 — last flag wins in DXC), and this was the first
measurement's biggest lever: SPIR-V ≥ 1.4 lets DXC emit `OpCopyLogical`,
which naga's spv-in does not implement — 6 of the first run's 14 failures.
The two things that forced 1.3 (subgroups, RayQuery) are absent from the web
corpus by construction, so 1.1 is free there.

**Measured 2026-08-18 (procedural scene): 16 units → 34 modules, 34/34
spirv-val clean, 28/34 naga-parse, 21/34 full validate + WGSL round-trip.**
The 13 failures are SIX named defects — the F-stage work-list:

| defect | modules | class |
|---|---|---|
| `ShaderNonUniform` — the bindless `texs[]` | reference, leaf, leaf_fb, hemi_leaf | **F2/WEB_TEX**, the planned centerpiece: WebGPU has no texture arrays of any kind |
| `ft_nodes` member[1] at offset 12 fails WGSL storage alignment | cs_level, cs_level_wide | the FTree wide-node struct wants the Stage-0b treatment (flatten or pad, HLSL + ftree.rs upload in lockstep) |
| `FrdCb` member[9] at offset 36 fails WGSL uniform 16-alignment | frd-temporal, frd-blur ×2 | pad/reorder FRD's CB, Rust packer + frd_common.hlsli in lockstep |
| infinite float literal | cs_sky | WGSL forbids INF literals; find the folded constant, spell it finite (or bitcast at runtime) |
| a local of a type WGSL can't hold locally | cs_sky_lod | investigate ([17] unnamed in the error; likely an array-of-textures or pointer local) |
| naga atomic-upgrade: "expected to find a global variable" | cs_hemi_root, cs_hemi_cell | the mixed atomic/non-atomic access pattern the plan flagged; characterize with a micro-unit, then restructure the access |

Everything else — the whole display stack, compose, resolve, wavefront's
seed/prep entries, autoexp, bloom, FRD's post entry — already survives the
full chain untouched. `FrameCb` (5632 B) passes as a WGSL UNIFORM unchanged,
confirming the F3 no-change decision: its 16-byte rows + vec4-stride arrays
are dx-layout ≡ WGSL-uniform by construction.

W5–W7 (limits/budget audit, hostile-construct scan, the cross-platform
corpus golden) are not yet built; the gate is deliberately NOT in CI until
the six fixes land and it holds green — a red gate in CI gates nothing.

## The corpus fixes, round 1 (landed 2026-08-18): 21/34 -> 30/34

Five of the six defects fell in one session; the survivors are all one
class (`texs[]`/ShaderNonUniform ×4 — the F2/WEB_TEX campaign). Native
proof after every change below: cargo test 28, `--check-gpu` OK on real
hardware, plain `--check` LAST with goldens byte-identical.

- **FrdCb** — `cam_fwd` now leads its row (dwords 8-10) with `cam_step` in
  the tail dword: a float3 at offset 36 is legal cbuffer packing but
  illegal WGSL uniform layout. Everything from `proj` (12) on is unmoved,
  so the shared dword-16 `light_par` contract holds. THREE-way lockstep:
  frd_temporal.hlsl + frd_blur.hlsl + frd_gpu.rs `cb()`.
- **INF/ISINF** — WGSL has no infinity (no literal, no `isinf`). Under WEB,
  `INF` = FLT_MAX and `ISINF(x)` = `x >= FLT_MAX` (true for the sentinel
  AND for genuine overflow, so miss-detection keeps its meaning; no
  `== INF`/`< INF` compare exists in the corpus). Native keeps the bit
  pattern and the intrinsic behind the #ifdef.
- **ft_nodes** — org/sca ride as SIX scalars (bytes 0-23 identical, the
  Stage-0b flatten): a vec3 at offset 12 is illegal WGSL STORAGE layout.
  QFNode's 112-B wire and ftree.rs upload untouched; only `ft_slot_box`
  reassembles.
- **GBufExt** — compiled OUT under WEB (decl + both stores): no RR/FSR-RR/
  NPPD exists in a browser session so FLAG_GBUF_EXT can never arm, the
  72-B record is unWGSLable anyway (vec4 members ⇒ element align 16 ⇒
  stride 72 illegal), and dropping it relieves a storage-buffer binding.
  CONTRACT: the WebGPU session must never set FLAG_GBUF_EXT.
- **positions/normals/uv_positions** — vec3-element storage arrays carry
  stride 16 in WGSL vs the tightly-packed 12-B wire; the web arm re-reads
  the same bytes as scalar arrays behind `pos_at`/`nrm_at`/`uv_pos`
  accessors (native declarations and SRV strides untouched).
- **`wgsl::normalize` — the naga-normalization passes**, the round's big
  find. naga's spv-in spills any expression used outside its defining
  structured body into a synthesized local; for POINTERS that local is
  invalid WGSL (cs_sky, cs_sky_lod), it breaks the atomic-upgrade walk
  ("expected to find a global variable", both hemi_wave entries), and its
  cached spill-LOADS get reused across sibling bodies ("used by a
  statement before it was introduced", both level kernels). No spirv-opt
  pass and no DXC -O level removes the triggering shapes (measured). Fix:
  (1) `split_chains` clones every cross-block OpAccessChain to sit beside
  each use — pointer expressions never cross a block; (2) `spill_values`
  de-SSAs cross-block VALUES (store-after-def + load-beside-use, phi
  edges load in their predecessor, phi results spill after the phi group)
  — naga never spills at all. Drivers re-promote to SSA, so runtime cost
  is nil. Both passes run BEFORE spirv-val (the reference validator
  checks the rewrite — it caught a literal/id collision in the first
  draft), are byte-deterministic (BTree containers), and only touch
  operand slots with certain grammar (under-rewriting is safe; blanket
  sweeps are how the OpExtInst literal got corrupted). Teeth: W0 carries
  a hand-assembled cross-body-chain module that must FAIL raw and PASS
  split; deleting spill_values fails W4 on both level kernels; the W2
  line prints the normalization id count (probe-reach).
- `FR_WGSL_DUMP=<dir>` — FR_SPIRV_DUMP's twin for THIS corpus: .spv
  always, .types.txt after naga parse (validator errors name types by
  handle index), .wgsl once the writer ran.

## Round 2 (landed 2026-08-18, same session): WEB_TEX shader side — 34/34, GATE IN CI

The bindless replacement (F2's shader half) took the corpus green:

- **`src/gfx/texweb.rs`** — the plan half (`plan(scene)`: exact-size
  Texture2DArray buckets keyed (w, h, levels, srgb), a 16-B meta row per
  texture carrying dims + `bucket<<16|layer` + texel-payload offset, and a
  mip-0 RGBA8 word payload for exactly the `mat_cutout` ∪ `mat_height`
  Load-path set) and the codegen half (`hlsl(plan)`: decls at t10/t11/t12+
  in space1 — the registers the bindless decl vacates — plus the
  `web_tex_dims`/`web_tex_a8`/`web_tex_grad`/`web_tex_level` dispatch
  helpers, samplers passed as parameters because the block pastes ahead of
  trace_common). Deterministic by construction (BTreeMap bucket order,
  ascending layers). `texweb::self_test` rides `--check`. The plan half is
  the SHARED artifact: the native FR_WEB_TEX arm, the wgpu backend and
  `--bake-web` all consume it — none of those exist yet (round 3).
- The six choke points carry `#ifdef WEB_TEX` arms: `tex_sample` /
  `tex_lod` / the detail-lod dims (shade.hlsli), `alpha_cutout` /
  `height_bilinear` / `height_march` (trace_common). The Load-path arms
  are byte-exact: `(word>>24) < 128` ≡ `.a*255 < 127.5`, and
  `float(word>>24)` is exactly the byte the native bilinear recovers.
- **Mat flattened** (the ft_nodes treatment): its three float3s land
  16-aligned by luck but the 108-B array STRIDE is illegal once any member
  aligns to 16; scalars + `mat_albedo`/`mat_emissive`/`mat_trans_tint`
  keep bytes 0-107 identical.
- **ISFINITE** joins INF/ISINF: `isfinite` lowers to OpIsNan (no WGSL
  target — surfaced by bistro's heightfield arm); the WEB arm is the
  exponent bit test, exactly IEEE isfinite.
- **HONEST LIMIT**: one bucket = one sampled-texture binding; WebGPU's
  DEFAULT limit is 16/stage. Real scenes want a raised
  maxSampledTexturesPerShaderStage (W5's audit + C1's probe own this).

**Measured: 34/34 full round-trip on procedural, damaged-helmet, bistro
(cutout + heightfield + transmissive), and san-miguel (water).** The gate
joined the check-vulkan CI job with positive-assertion greps (PASSED + W1
loaded + spirv-val ran + normalize touched > 0 ids).

Helmet-scene note, measured in passing: `--check-gpu
scenes/damaged-helmet/DamagedHelmet.glb` fails two gates (two-device d2 r1
localized divergence 2487 hot > 720; N9 nrd residual-sign ulp-scale) —
IDENTICALLY on unmodified master, to the same numbers. Pre-existing,
scene-specific, not a web-port regression; unowned as of this writing.

Round 3 (the remaining F2 half + the ladder): the native `FR_WEB_TEX`
byte-gate (SceneGpu bucket upload + A/B vs the bindless arm, `--no-bc7` on
both so the binding topology is the only delta), then W5–W7, then Stage C1
(the wgpu device probe).

## Stage C1 (2026-08-19): the wgpu host lands — `--check-wgpu` J0–J3

wgpu 30 + pollster 1 under `[target.'cfg(any(unix, windows))'.dependencies]`
(compute-only features: std/parking_lot/wgsl/vulkan/dx12/metal — no gles, no
webgpu until Stage D). The lockstep held with zero effort: wgpu-core 30 wants
naga ^30, so `cargo tree -i naga` shows ONE naga node shared by the gate and
the runtime. New modules `src/webgpu/{mod,device,headless}.rs`; the gate is
J0 (pure pick/limits self-test) | J1 (adapter + device + the granted-limits
probe) | J2 (SMOKE_HLSL through DXC -> vulkan1.1 SPIR-V -> wgsl::normalize ->
spv_to_wgsl -> ShaderModule — spv_to_wgsl's first live consumer) | J3 (the
indirect-dispatch smoke, verdicts verbatim from the Vulkan twin). "Three
backends, one kernel" is now four. In the check-vulkan CI job after
--check-wgsl (lavapipe + the DXC drop are already there), positive greps
PASSED + a J0–J3 line loop.

**The finding the first run bought** (the gate caught the design being
wrong): a dispatch's usage scope is computed from the pipeline LAYOUT, not
the shader's static use. The Vulkan twin's one-layout-for-everything shape is
ILLEGAL in WebGPU — the shared layout carried `args` as read-write storage,
STORAGE_READ_WRITE is exclusive, and the fill dispatch's scope (layout
bindings + the INDIRECT buffer) conflicted with itself:

    Attempted to use Buffer with 'smoke args' label with conflicting usages.
    Current usage BufferUses(STORAGE_READ_WRITE) and new usage
    BufferUses(INDIRECT).

Layouts are now PER PIPELINE, cut (with the bind groups) from one
`SMOKE_USES` table — which independently confirms Stage C2's planned
per-entry-point layout shape by measurement rather than by reading the spec.
The uncaptured-error handler did its job on the same run: the error became a
FAIL line + a counted sweep instead of wgpu's default native PANIC.

**The day-one limits probe, answered for native** (the reason C1 exists):
every adapter on the dev box grants the raised `max_bindings_per_bind_group
= 2003` ask (derived `binding_of(U,2)+1` — the DXC shifts put u0 at 2000),
and every limit the tracer needs has orders-of-magnitude headroom. Measured
7/7 arms green, `FR_WGPU_ADAPTER` x `WGPU_BACKEND`:

| adapter | backend | max bind/group | storage buf/stage | storage tex/stage | sampled tex/stage |
|---|---|---|---|---|---|
| RTX 4090 | Vulkan | 4294967295 | 524288 | 524288 | 524288 |
| RTX 4090 | Dx12 | 4294967295 | 262144 | 262144 | 393212 |
| Arc Pro B70 | Vulkan | 33554432 | 3355442 | 3355442 | 3355441 |
| Arc Pro B70 | Dx12 | 4294967295 | 262144 | 262144 | 393212 |
| Radeon iGPU | Vulkan | 4294967295 | 429391872 | 429391872 | 429391871 |
| Radeon iGPU | Dx12 | 4294967295 | 262144 | 262144 | 393212 |
| WARP (software) | Dx12 | 4294967295 | 262144 | 262144 | 393212 |

(WebGPU defaults, what a browser grants before any ask: 1000 / 8 / 4 / 16.)
So the binding-ceiling question is a BROWSER question only — whether Dawn/
WebKit grant a raised maxBindingsPerBindGroup decides C2's compact-or-ask
choice, and nothing native constrains us meanwhile. WARP running the whole
indirect chain correctly is a pleasant extra: a zero-GPU Windows box still
exercises J3.

Traps for the next stage: wgpu enumerates each physical GPU once per backend
(the same 4090 twice), so candidate names carry the backend and a bare
`FR_WGPU_ADAPTER=radeon` on a dual-backend box is CORRECTLY ambiguous;
`InstanceDescriptor` lost its `Default` in wgpu 30 (use
`new_without_display_handle_from_env`, which also honors `WGPU_BACKEND`
free); the sentinel poison must be `queue.write_buffer` (clear_buffer only
writes zeros — the opposite of a poison); `entry_point: None` survives naga
renaming the sole @compute entry; and the error-scope guard's `pop()` is a
future — a dropped guard EATS the error, so the block_on is load-bearing.

Next: FR_WEB_TEX native byte-gate, W5–W7, Stage C2 (the tracer recorder over
per-entry layouts — now a measured requirement, see above).
