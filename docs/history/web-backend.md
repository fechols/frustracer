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

## Round 3 (2026-08-19): FR_WEB_TEX — the native runtime half, `--check-gpu` M15/M15b

The upload/binding half of F2, on native D3D12, with the gate that proves the
browser texture path renders right without a browser.

**The machinery** (all in gpu/trace.rs unless noted):

- Root signature: a STATIC middle SRV range in the RP_SCENE_TEX table —
  space1 `texweb::META_REG..` (t10 meta, t11 texels, t12.. buckets), sized
  `WEB_TEX_SLOTS = 2 + WEB_TEX_MAX_BUCKETS (64)`; the unbounded space2
  `texs[]` range stays last, its heap offset shifted by WEB_TEX_SLOTS. A
  range costs zero root DWORDs, so the signature stays 64/64. Registers are
  pinned three ways: texweb.rs consts are the single source, `hlsl()`
  formats from them, `texweb::self_test` string-pins the emitted text, and
  trace.rs carries a const assert.
- The lever: `FR_WEB_TEX=1` (loud, FR_WIDE-rule) seeds a process global with
  `set_web_tex`/`web_tex` for the gate's in-process A/B — snapshotted at
  `TraceGpu::new` (the set_sky_lod contract). No CB bit, no TraceKeys field:
  define + resources + descriptors are one constructor-scope decision
  (the FR_DXR_LEAN shape), off-state structural (a conditional prepend).
- Armed `TraceGpu::new`: prepends `texweb::hlsl(&plan)` to exactly the NINE
  units `web_units` wraps, at the srcs→pso seam (the collection shape —
  probe-reach-immune; feed/nrd/nppd keep texs[], which stays resident);
  `WebTexGpu::new_uploaded` (the SwTreesGpu model) streams meta/texels and
  band-uploads every bucket layer's full mip chain (subresource =
  layer*levels+mip) through `d3d12::committed_tex_array`; descriptors land in
  the wavefront's own heap slice — never `write_scene_descriptors`, so DXR is
  structurally unmoved. >64 buckets falls back to bindless with one loud line.
- BC7: an array resource is ONE format and `should_compress` varies within a
  bucket key, so buckets are always RGBA8; the gates run both arms over one
  `Bc7Mode::Off` core so binding topology is the only delta.

**The gate.** M15 brings its own scene (`scene::texweb_check_scene` — the
default check scene is texture-free and would be vacuous): a multi-layer
bucket + a second bucket + a cutout + an h2n heightfield + a grazing floor
tile (the SampleGrad arm). M15b repeats the A/B on the session scene.
Verdicts are TWO-TIER: EXACT on tbuf/info/counters (geometry + the byte-exact
Load paths) and on the uploaded bytes (`web_tex_audit` — a per-(texture, mip)
readback compare of every bucket layer against its bindless resource);
ULP-BOUNDED on accum (violation = |a−b| > max(1e-6, 1e-5·max|·|)). Teeth are
same-program (arm-B-vs-arm-B, where bit-identity IS structural): a bl
layer-swap poison and an ofs+1 payload poison must each push the image past
the bound and restore to bit-identity; a planted `web_tex_a8 → 0` break made
M15 FAIL 5198/19266/3 (tbuf/accum/counters) — both ways proven.

**Why accum is bounded, measured (bistro, RTX 4090):** the first run's
"byte-identical" claim failed by 43 of 1.44M accum channels — and every
discriminator said *not our bytes*: identical 43 with `--aniso 1` (not the
SampleGrad path), 408 with `--no-mips` (not mip filtering — level-0 bilinear
only), and the upload audit read back ALL 187 textures × every mip (≈1 GB)
byte-identical. The diffs are 1–8 ulp on small radiance values in one screen
region: the texweb preamble changes DXC's instruction fusing and the shading
math rounds differently at the last bit — the FR_NOPRECISE class, recorded by
Stage 0 as low-risk. The bound sits ~100× above that noise and ~1000× below
any routing bug (a wrong layer/lod/offset moves channels O(0.01..1) — the
poisons prove the bound catches exactly that).

**Measured green:** procedural (M15: 3 buckets/4 layers/2048 payload words,
all-exact, poisons bit; M15b skips loudly), bistro (M15b: 21 buckets/187
layers/250M payload words — a 1 GB `web_texels` buffer works — tbuf/info/
counters exact, audit 0 bytes, accum 43 bits / 0 violations), san-miguel
(M15 green; M15b = the >64-bucket loud fallback, exercised live: 165 bucket
keys — a W5 datum, though san-miguel is outside the web ring; bistro's 21 is
the browser-bound number). Per vendor: RTX 4090 as above; **Arc Pro B70
all-exact including accum** (0 bits — Intel's compiler fuses identically);
AMD iGPU UNMEASURED — the suite DEVICE_HUNGs at the spp=128 wavefront probe
before M15, IDENTICALLY on unmodified master (a TDR on the 22×-slower iGPU;
pre-existing, not the port's). Bistro also fails N9 (nrd residual-sign) —
verified IDENTICAL on unmodified master, the helmet-N9 class: pre-existing,
scene-keyed, not the port's.

Closing ladder, all green: cargo test 39/39, `--check-dxr` PASS (the heap
gained WEB_TEX_SLOTS), `--check-wgsl` 34/34 (the texweb consts refactor kept
the emitted block byte-stable — self_test pins the registers),
`FR_WEB_TEX=1 --gpu --spin still` on bistro (both announce lines, 30 frames),
`cargo check --target wasm32-unknown-unknown`, and plain `--check` LAST with
`check.png`/`check_gi.png` byte-identical.

Next: W5 (limits/budget audit — now with two measured anchors: 21 buckets on
bistro, 165 on san-miguel), W6, W7, Stage C2 (the tracer recorder over
per-entry layouts).

## Round 4 (2026-08-20): W5–W7 — the corpus audits that close the W track

Three stages onto `--check-wgsl`, all machinery in `src/wgsl.rs` (uncfg'd —
every new fn is pure and wasm-compilable, and its teeth run in W0 AND in
plain `--check`'s self-test sweep, which is the three OS CI jobs' only view
of them; they carry no DXC and never run the wgsl gate).

**W5 — the per-entry layout audit.** `wgsl::profile(&naga::Module)` counts
DECLARED resource globals per class (declared ≈ used — DXC strips dead
resources, and declared is what a `BindGroupLayout` pays for), plus
groupshared bytes (`AddressSpace::WorkGroup` + `TypeInner::try_size`), the
`Frame` uniform's byte span, and IR-level hostiles; it asserts the
one-module-one-entry corpus invariant. `wgsl::BUDGET` is **the C2 ask-limits
contract, not the WebGPU defaults** — `audit()` pins the corpus under what
the browser session will `required_limits`-request, with buckets audited
against the scene's own plan (scene-keyed) rather than any fixed row.

Measured (first W5 prints): procedural — worst sb 22/32, st 9/12 (both on
frd_temporal's side of the corpus as predicted), **fixed sampled 15
(frd_temporal — over the drafted 12; the row was trued to 16 from this
print, the plan's own rule)**, samplers 1, ub 2, groupshared max 2060 B
(cs_level_wide; cs_level 2048), Frame 5616 → stride 5632 (== CB_STRIDE,
the cross-language cbuffer pin, live). Bistro — worst sb 26, buckets 21
classified through DXC OpNames (the runner's reach probe owns that half; a
W0 tooth pins the name-prefix half every run, because THE DEFAULT SCENE
PLANS 0 BUCKETS — it is texture-free, so a bare CI run never exercises the
classifier via the corpus). **The trace units' sampled textures are pure
buckets** (0 fixed + 21 on bistro): the worst per-entry sampled total is the
scene's bucket count itself — bistro 21 vs the untouched 16/stage default,
the honest number the browser bucket story hangs on. (The first draft of the
stage line summed frd's fixed 15 with the trace units' 21 buckets into a
fictional 36 — two different modules' worsts; rewritten to the real
per-entry max.)

**W6 — the hostile-construct scan.** `wgsl::scan_wgsl` (identifier-exact
tokens verified against naga 30's WGSL writer: `binding_array`,
`wgpu_binding_array`, `f16`, `ray_query`, `acceleration_structure`,
`subgroup*`) runs over every W4-emitted text — belt to `validate()`'s
`Capabilities::empty()` braces, so it survives a naga bump that widens a
default. The every-run tooth: a PLANTED WEB_TEX-off leaf (web_defs without
the texweb block, so the bindless `texs[]` survives) is compiled and pushed
through the chain each run; a reflect probe first proves it really carries
the unbounded table (else the arm proves nothing), then the chain must
refuse it — measured: W3 parse refuses it (the historical ShaderNonUniform
class). A clean sweep is a FAIL: a real bindless leak would sail through the
same way.

**W7 — the tracked corpus golden.** `goldens/web_corpus.txt` (plain git,
`.gitattributes -text`): header + per unit `hlsl=fnv1a64:<hash>` over the
\r-stripped assembled source (CRLF working trees vs LF CI — hash-strip,
compare-strip, and the -text pin are three independent guards) + one profile
line per compiled entry + a module-count tail. LF-only, integers-only,
deterministic order — `golden_entry_line`'s exact format is itself
self_test-pinned so format drift cannot masquerade as corpus churn. Missing
golden = loud FAIL (a tracked file's absence is a defect; a SKIP would be
permanently-green vacuity); scene-keyed runs SKIP loudly (the golden pins
the default corpus). `--write-golden` (cli) refuses: without `--check-wgsl`
(exit 2, guarded right after arg parsing — the first draft guarded at the
gate dispatch and a bare `--write-golden` opened an interactive session,
measured), on a scene-keyed run (exit 2), on the W1-SKIP path (exit 2), and
unless W0–W6 are green in the same run (exit 1). Teeth proven live: one
corrupted golden byte → FAIL with the first differing line pair; restore →
match.

CI: the check-vulkan wgsl block gains `for s in W5 W6 W7` positive greps
(trailing space — `SKIP W7` cannot match) + a `WEB_TEX-off` grep for the
planted arm. The golden's first Linux compare on that job is the
cross-platform proof (counts/spans/workgroups are semantically forced and
the DXC drop is version-pinned; if Linux DXC ever disagrees on a count,
record and split per-field, never per-OS).

Closing ladder green: cargo test 39/39, `--check-wgsl` bare (W0–W7,
golden written then matched) + bistro (buckets live, W7 SKIP),
`--check-spirv`, `--check-gpu`, wasm `cargo check`, plain `--check` LAST
(now also running `wgsl::self_test` in the sweep), goldens byte-identical.

Next: Stage C2 — the wgpu tracer recorder over per-entry layouts (the W
track is closed; W5's profile data is the exact per-entry binding
information C2's `webgpu/layout.rs` consumes).
