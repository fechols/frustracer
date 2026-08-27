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

## Stage C2a (2026-08-20): the browser's TRACER renders — `--check-wgpu` J6

C1 proved a wgpu device executes the translator's output on the SMOKE
kernel. This is the tracer: the browser's own corpus, compiled through the
browser's own chain, bound through layouts DERIVED from the modules wgpu
compiles, rendering the reference kernel — and scored against the CPU
reference by `--check-vk` V6's two verdicts, to the line.

Scope is deliberately the reference kernel + resolve + the two cloud caches,
which is the rung the Vulkan port landed first and for its stated reason:
the reference kernel is the smallest thing that can be WRONG in an
interesting way. The wavefront ladder, replay and display are C2b/C2c; the
`Variant` enum, the push RING and the per-(entry, variant) bind-group table
are already shaped for them, so those arms are new arms rather than new
mechanisms.

### What is derived, and from what

`vk/layout.rs` reads `crate::reflect` (a SPIR-V parse). `webgpu/layout.rs`
reads the new `wgsl::bindings`, which walks the **naga IR** — one step
further down, and for two reasons that are not style:

- **WebGPU puts read-only-vs-read-write in the layout ENTRY.** Vulkan has one
  `STORAGE_BUFFER` type and spells read-only as a SPIR-V decoration the
  layout never sees (`reflect::DescKind::StorageBuffer` says exactly this).
  The access mode a WGSL global carries is what a browser's compiler reads,
  so taking the layout from the same module the text came from makes the two
  unable to disagree. Deriving it from `t` vs `u` would restate the
  register-shift rule with nothing pinning the two statements together.
- **`reflect` is `cfg(any(unix, windows))`.** It reads SPIR-V, which the
  shipped page will not carry. `wgsl::bindings` is pure and compiles on
  wasm, so Stage E can BAKE its output and the browser builds the same
  layouts from data. A derivation that only ran natively would leave the
  browser's layouts hand-written — the liability the whole derivation exists
  to remove.

`wgsl::BindKind` is a THIRD vocabulary (not naga's, not wgpu's) for
`reflect::DescKind`'s own argument: the module is pure, its output is
bake-able, and a storage format it does not map is an ERROR rather than a
pass-through.

**One layout per ENTRY POINT, one bind group per (entry, variant).** Not the
spec's no-PARTIALLY_BOUND clause alone — C1 measured it: a dispatch's usage
scope comes from the pipeline LAYOUT, so a shared layout carrying `args` as
read-write storage makes the indirect fill conflict with itself. Bind groups
are still built once at construction, so a dispatch costs a
`set_bind_group` and no descriptor traffic.

**`b1` is a dynamic-offset uniform RING.** D3D12 has root constants, Vulkan
has `vkCmdUpdateBuffer`; WebGPU has neither, and every host write in a
recording happens before the submit. All push rows a frame needs are written
up front at the device's own `min_uniform_buffer_offset_alignment` and
selected per dispatch by offset. This is the ONE hand-made entry in a derived
layout, so it is a parameter of `layout::build_unit` named at exactly one
call site.

**The device ask is `wgsl::BUDGET`.** The limits C2 requests and the limits
W5 proved the corpus fits under are ONE constant, asserted equal in
`webgpu::device::self_test` field by field — a corpus audited green can never
be run under a limit it was never audited against. Two rows are scene
properties and cannot come from BUDGET (`device::Ask`): the WEB_TEX bucket
count and the largest single buffer. Both are computed with no device in
hand, because the ask is what a device is created WITH — which is what a page
must do too.

The corpus itself is assembled by `gfx::shaders::web_keys` / `web_unit` /
`web_trace_unit`, now SHARED with `--check-wgsl`'s `web_units`. The text this
tracer compiles is byte-for-byte the text that gate validates and
`goldens/web_corpus.txt` pins. Without that sharing the golden would pin text
nothing executes and the executed text would be pinned by nothing.

### Two browser constraints the corpus did not know about

Both were MEASURED as failures on the first J6 run, and both are facts about
WebGPU rather than wgpu quirks.

**1. `read_write` storage textures.** WebGPU allows that access for exactly
three formats (`r32uint`, `r32sint`, `r32float`). `cs_resolve` only ever
STORES to `hdr`, but DXC emits no `NonReadable`, so naga reads the access as
`LOAD | STORE`, the derived layout says `ReadWrite`, and
`createBindGroupLayout` refuses:

```
Binding index 2014: ReadWrite access to storage textures with format
Rgba32Float is not supported
```

Fixed by a third SPIR-V pass, `wgsl::mark_write_only`, joining
`split_chains` and `spill_values` in `normalize`: decorate every storage
image that no `OpImageRead` / `OpImageSparseRead` / `OpImageFetch` /
`OpImageTexelPointer` names as `NonReadable`. It states a fact about the
shader that HLSL has no syntax for. Sound by construction — a module that
genuinely reads and writes one is left alone and still refused downstream,
correctly, because a browser cannot run it. spirv-val accepts the added
decoration (W2 stayed 34/34). Teeth are word-level over a synthetic stream
in `wgsl::self_test` (decorated when never read; untouched when read;
untouched with no storage image at all), and the end-to-end tooth is
structural: if the pass stops firing, J6 cannot create the layout — which is
how it was found.

**2. The storage texture's FORMAT.** WebGPU's layout entry names the format
and requires it to equal the texture's. HLSL has no format syntax for a UAV,
so DXC picks `rgba32f` for an unannotated `float4` — and `hdr` is RGBA16F.
`resolve.hlsl` now carries `[[vk::image_format("rgba16f")]]` under `#ifdef
WEB`: SPIR-V-only and WEB-only, so DXIL and the native SPIR-V corpus stay
byte-identical (Vulkan's layout carries no format, so the annotation would
buy it nothing and would move recorded numbers for stages that never wanted
it). W7's golden moved by exactly one line — the resolve unit's HLSL hash —
with every COUNT unchanged, which is the signal that matters.

### And one about reading the result back

`hit 13655 | sky 0 | class-mismatch 5545` on the first scored frame, with a
max relative `t` error of 2.01e-6 on the pixels it DID classify. The render
was already right; the READING was wrong. WGSL has no infinity, so the `WEB`
arm of `trace_common.hlsli` defines `INF` as `FLT_MAX` — a browser-corpus
miss writes a FINITE float, and `t.is_finite()`, the classifier every native
gate uses, reports sky as geometry.

The predicate now lives once, in `gfx::shaders::WEB_INF` / `web_miss`, for
every host-side consumer — this gate today, Stage D's overlay tomorrow. A
cargo test pins three claims no GPU can reach: the HLSL still spells
`FLT_MAX` as `3.402823466e38`, that literal is exactly `f32::MAX`, and the
`WEB` arm still routes `INF` to it.

### Measured (RTX 4090 / Vulkan)

| scene | res | frames | hit / sky | class-mismatch | rel-t > 1e-3 | max rel t | radiance mean rel |
|---|---|---|---|---|---|---|---|
| procedural (79.7k tris) | 400x300 | 16 | 85346 / 34654 | 0 | 1 | 1.43e-2 | 0.213% |
| procedural | 160x120 | 16 | 13655 / 5545 | 0 | 0 | 2.01e-6 | 0.441% |
| damaged-helmet (15.5k, 5 tex / 2 buckets) | 400x300 | 16 | 85414 / 34586 | 0 | 0 | 8.35e-6 | 0.074% |
| bistro (2.84M, 187 tex / 21 buckets) | 200x150 | 4 | 21342 / 8658 | 0 | 0 | 2.31e-5 | 0.024% |

Bars are V6's, unchanged: class-mismatch ≤ 0.05%, rel-t violations ≤ 0.01%,
radiance ≤ 2%, non-finite exactly 0. The single 1.43e-2 rel-t pixel at
400x300 is one grazing hit out of 120000 — both sides run the SAME software
intersector here (`rt_sw.hlsli` is a port of `bvh.rs`'s traversal, not a
second one), so V6's two-intersector grazing-edge set does not exist on this
backend and the residual is f32 ordering.

Resolution and frame count follow the ADAPTER (a departure from V6, stated
on the line): a software adapter running a software intersector gets 160x120
and 4 frames, because lavapipe in CI is two orders of magnitude off hardware.
Every verdict still runs; the numbers there are coverage, not quotable.
`FR_WGPU_RES` / `FR_WGPU_AB_FRAMES` override both.

### THE BROWSER GO/NO-GO TABLE (J1, per scene)

Every row where the ask exceeds the WebGPU default is a row a page must
request and a browser may refuse.

| limit | default | procedural | helmet | bistro |
|---|---|---|---|---|
| `max_bindings_per_bind_group` | 1000 | **3002** | **3002** | **3002** |
| `max_storage_buffers_per_shader_stage` | 8 | **32** | **32** | **32** |
| `max_storage_textures_per_shader_stage` | 4 | **12** | **12** | **12** |
| `max_sampled_textures_per_shader_stage` | 16 | 16 | **18** | **37** |
| `max_storage_buffer_binding_size` | 128 MB | 128 MB | 128 MB | **955.6 MB** |
| `max_buffer_size` | 256 MB | 256 MB | 256 MB | **955.6 MB** |

The 4090 grants all of it (`max_bindings_per_bind_group` 4294967295,
per-stage counts 524288, `max_buffer_size` 4294967295). Whether a BROWSER
does is still the open question C1 named — but it is now asked with real
numbers instead of estimates, and the binding ceiling is the one to watch:
3002 is 3x the default and comes straight from DXC's `s` register shift.

**And a size finding for Stage E/H.** Bistro's browser texture plan is
**~5.2 GB of RGBA8** (187 textures with full mip chains — uploaded and
rendered here on a 24 GB card). That is over wasm32's entire 4 GB address
space on its own, so BC7 (`texture-compression-bc`, filed under Stage L) is
not a download optimization for bistro — it is a PRECONDITION for bistro
existing in a browser at all. Stage H's world-ring budget must be measured
against the texture plan, not the geometry.

### Ladder run

cargo test, `--check-wgsl` (W0–W7; golden regenerated for the one moved
hash, then matched), `--check-wgpu` on three scenes, `--check-spirv`,
`--check-gpu`, `--check-dxr`, wasm `cargo check`, plain `--check` LAST with
goldens byte-identical. CI: `--check-wgpu`'s J-loop gains J6 plus two
positive greps for its verdict lines (a tracer that built and never rendered
would otherwise still print the upload line and read as PASSED). The gate
arms `--sw-rays` itself, exactly as `--check-wgsl` does, so a bare invocation
is sufficient.

## Stage C2b (2026-08-20, same session): the WAVEFRONT QUADTREE — `--check-wgpu` J7

The browser's actual render path. Seed -> depth_full x (prep-args -> level)
-> the leaf and sky terminal fills, statically recorded: every scheduling
decision after the seed is a GPU-written counter feeding
`dispatch_workgroups_indirect`, so an empty level dispatches zero groups
rather than being skipped by the CPU. There is no readback anywhere in the
frame, which is the property the whole design rests on.

**The result, on the first run and on all three scenes: BITWISE IDENTICAL to
the reference kernel.** 0 of 360000 channels differ at 400x300; claim-
violation, false-sky and tmin-overshoot all exactly 0; the leaf and sky
rects partition the screen exactly; no info sentinel survives; both tile
queues drain; overflow and cut-pool fallbacks both 0.

That bitwise result is stronger than either native backend's own bar and the
reason is structural rather than luck: under `--sw-rays` there is only ONE
intersector. `--check-vk` V7 must tolerate a grazing-edge set because
hardware RayQuery is not `moller_trumbore`; here both kernels run
`rt_sw.hlsli`, which is a port of `bvh.rs`'s traversal and not a second
implementation, so agreement is exact or it is a bug.

### The push RING, and the hazard it deletes

The Vulkan twin rewrites `b1` twice per level with `vkCmdUpdateBuffer`, and
its `push()` carries TWO barriers — the read-after-write one, and a
write-after-read one whose omission silently cost the entire ladder past
level 0 on that backend's first run (the transfer executed ahead of the
dispatch it textually followed, so `cs_prep` read the NEXT level's `push3`
and wrote its indirect args to the wrong slot; nothing faulted and validation
was clean).

WebGPU cannot make that mistake, because it cannot express the operation: a
host write lands before the submit, full stop. So the ladder is built as
DATA first — a `Vec<Step>` naming each dispatch's entry, variant, push value
and launch shape — and walked twice: once to write every row into the ring,
once to record. **Each step gets its own ring row**, so a row's lifetime is
exactly one dispatch and the WAR hazard has nowhere to live. The stride is
the device's own granted `min_uniform_buffer_offset_alignment`, read back
rather than assumed to be the 256-byte default.

### No barriers, again

The whole ladder — 26 dispatches at depth 4, half of them indirect, over
queues written by one dispatch and consumed by the next — records into ONE
compute pass with no explicit synchronization at all. WebGPU's per-dispatch
usage scopes are the edges. That this is sufficient was J3's claim on the
smoke; J7 is the claim at scale, and it holds. The one thing that had to be
true for it: `cs_level` must not DECLARE `args`, or its layout would carry
that buffer as read-write storage while the same dispatch reads it as
INDIRECT — the self-conflict C1 measured. DXC strips it, and the derived
layout is what makes that observable rather than assumed.

### Teeth, proven both ways — and a hole the plant found in the gate

A planted ping-pong bug (both ladder variants bound queue A as `qin`) turned
J7 red on three verdicts at once: rect area 0/120000, 120000 info sentinels,
4 dangling tiles.

**But the image A/B compared CLEAN on that plant** — 0/360000 channels
differing — because the ladder emitted zero terminal records and `accum`
still held the reference frame nothing had overwritten. An operation that
never happened compares clean against its own oracle: the M3d lesson,
re-learned on a fourth backend. The gate now POISONS `accum` and `tbuf` with
`0xEEEEEEEE` before the wavefront frame and asserts zero survivors; on the
re-planted run that reads 360000 accum + 120000 tbuf survivors and fails,
and on correct code it reads 0 and 0. `write_buffer`, not `clear_buffer` —
WebGPU can only clear to zeros, and zero is a legitimate radiance.

### Measured (RTX 4090 / Vulkan, release)

| scene | res | depth | leaves / sky | splits / blocked / cuts | coverage | soundness | bitwise |
|---|---|---|---|---|---|---|---|
| procedural | 400x300 | 4 (even) | 192 / 4 | 65 / 64 / 113 | 120000/120000 | 0 / 0 / 0 | 0/360000 |
| procedural | 160x120 | 3 (odd) | 48 / 4 | 17 / 16 / 29 | 19200/19200 | 0 / 0 / 0 | 0/57600 |
| damaged-helmet | 400x300 | 4 | 192 / 4 | 65 / 64 / 113 | 120000/120000 | 0 / 0 / 0 | 0/360000 |
| bistro (2.84M) | 200x150 | 3 | 48 / 4 | 17 / 16 / 29 | 30000/30000 | 0 / 0 / 0 | 0/90000 |

Both DEPTH PARITIES are covered, which matters because the drained-queue
check is parity-selected (`cs_prep` zeroes only the OUT counter, so which
tile queue must be empty at the end depends on `depth_full % 2`). 400x300
gives depth 4, 160x120 gives 3 — a parity-selected gate is half a gate until
both parities have run.

### Not in this rung

The hemisphere tier (`cs_hemi_root`/`cs_hemi_cell`/`cs_hemi_leaf`/
`cs_compose` and their two queue parities) and structure REPLAY are Stage
C2c. `render_wavefront` REFUSES a frame that asks for a bounce tier rather
than quietly rendering without one — an ambient term dropped in silence is
exactly the `BOUNCE_Q.ao_samples` failure mode this tree already has a rule
about. The `Variant` enum, the step program and the per-(entry, variant)
bind-group table are shaped so both arrive as new arms rather than new
mechanisms.

## Stage C2 review round (2026-08-20, same session): what a read-through found

C2a/C2b were read end to end before landing. Ten findings; the two that
mattered are below, and both were the same shape — a claim stated more
strongly in a comment than the code could support.

### J7's image A/B was gated on a statistic that cancels

The A/B computed a BITWISE difference count and printed it, then gated on
`|Σref − Σwav| / Σref ≤ 0.5%`. A signed sum over every channel of the frame
cancels: any PERMUTATION of the ladder's output scores exactly zero on it.

MEASURED rather than argued. Rotating the ladder's accumulator by 137 pixels
— every channel in the frame misplaced, `0/360000` becoming `359985/360000`
bitwise — gives:

| verdict | planted arm |
|---|---|
| sum rel diff | **0.0000%** (bit-unchanged, as a permutation must be) |
| coverage (rect partition, sentinels, dangling queue) | clean |
| all three soundness counters | 0 |
| poison survivors | 0 accum, 0 tbuf |

Every other J7 verdict passes that plant, and for a structural reason: a
permutation writes every pixel, so the poison sentinel — the thing added in
C2b precisely to catch a ladder that never ran — cannot see it either. The
gate would have shipped green on a renderer that put every tile's light in
the wrong place.

The gated statistic is now the per-channel MEAN ABSOLUTE difference,
expressed as a fraction of the frame's own mean (so the bar travels across
scenes and exposures) with the reference mean floored so a dark frame gets no
free pass. Bar 0.1%; the plant scores **36.04%**. The sum ratio is still
printed as a SEPARATE diagnosis — large sum with small mean-abs is uniform
bias, the reverse is misplaced light — and J6 keeps the sum ratio as its
gated one, correctly: its two sides draw independent sample streams, so
convergence is the only answerable question there.

Bitwise identity is reported, not gated. It measures 0 on all three scenes
and both depth parities on the 4090, but no CI adapter has reported yet, and
a claim this gate has not earned is not one to assert.

### `mark_write_only` claimed soundness it did not have

The pass decorates a storage image `NonReadable` when it sees no read — and a
WRONG decoration is undefined behaviour rather than an error, because
spirv-val does not check the claim and naga simply believes it. The first
version asked an ENUMERATED question ("did one of four opcode shapes name
it"), so an image reaching a read through `OpCopyObject`, `OpPhi`, `OpSelect`
or a function parameter was invisible and its variable would have been
decorated while genuinely read.

It now asks the TOTAL one: **is every occurrence of the four read opcodes in
this module attributable to a variable I know?** Image values are followed
through the laundering opcodes to a fixed point (a fixed point, not one
forward pass, because `OpPhi` may name a value defined later); an image
handed to `OpFunctionCall` or stored through `OpStore` escapes analysis and
marks its variable READ; and a read that resolves to nothing makes the pass
**decorate nothing at all** and hand the module on untouched. Declining to
claim is always available and always safe — the cost is the same loud
`createBindGroupLayout` refusal that motivated the pass.

The precision that keeps that bail-out from firing on ordinary work is
tracking SAMPLED images too: a fetch that resolves to a sampled variable is
simply not this pass's business, while one that resolves to NOTHING is the
alarm. Without that distinction the bail would disable the pass on every unit
that reads a texture. Four new `wgsl::self_test` arms, all DXC-free and
wasm-safe: laundered-through-`OpCopyObject`, escaped-into-`OpFunctionCall`,
unresolvable-read-bails, and sampled-fetch-does-not-suppress.

Separately, the undecorated-module arm advanced its insertion point to the
LAST type/constant/function instruction rather than the first (`max` inside a
loop over a monotonic index), which would have spliced an `OpDecorate` into
the middle of a function body — opcode 59 is `OpVariable`, which lives there
too. Unreachable through DXC, which decorates every resource, but it was
written as the total-function arm and was not one.

### The rest

- **The ask is resolution-keyed now.** `ask_for` covered the scene's buffers
  and not the TRACER's, which are resolution-derived — `accum` alone (12 B
  per pixel) is the largest buffer at 400x300 on the default scene, and
  passes the 128 MB default `max_storage_buffer_binding_size` at 8K. The
  ladder's caps come from one `webgpu::tracer::caps_for` that the ask and the
  allocation both call, so they cannot disagree; the ftree row's per-node size
  is `size_of::<QFNode>()` rather than a transcribed 128.
- **The device is scene-keyed from J1, not from J6**, and three doc sites said
  otherwise. The smoke KERNEL is scene-free; the DEVICE J2/J3 run on is not.
- **The dummy buffer could not serve its own uniform fall-through** (usage was
  `STORAGE|COPY_SRC|COPY_DST`). Unreachable today — only `b0`/`b1` are uniform
  and both always resolve — but the arm exists to be a net.
- Dead code removed (`EntryLayout::live_groups`, `StorageFmt::name`), a
  swallowed readback error named, and `run_steps` no longer requires a ladder
  on the reference-only path.

Re-verified after the fixes: `cargo test` 40/40, `--check-wgsl` W0–W7 (34/34,
golden matches), `--check-wgpu` on procedural / helmet / bistro and both depth
parities, `--check-spirv`, `--check-gpu`, `--check-dxr`, wasm `cargo check`,
then plain `--check` LAST with the tracked goldens byte-identical.

Next: C2c — the hemisphere tiers and replay (J8/J9), then the display stage
(J18), and then Stage D: the web shell, and the first frustracer pixels in a
browser tab.

---

## Stage C2c (2026-08-21): the hemisphere tiers and structure replay

The last two render-path arms the WebGPU tracer did not have. Nine more entry
points (21 total), two more bind-group variants, eight more buffers, and two
gates: **J8** the bounce tiers, **J9** replay.

The port itself is `vk/tracer.rs`'s design read off the page — the hemi units
re-declaring u5/u6/u7/u9 as `HemiCellRec` queues, the root pass running under
the ODD parity because it WRITES `hqout`, the per-batch reset that bounds the
memory, `cs_leaf` compiled twice from one source because `LEAF_NO_FB` is a
register-pressure decision an fb frame cannot take. All of that transferred
without incident. Three things did not.

### The push ring was a literal, and the hemisphere tail is not literal-sized

`PUSH_ROWS = 64` carried the ladder. Every push row must be written before the
submit — the WebGPU constraint this backend is built around — so a dispatch
cannot reuse its predecessor's row, and the ring's length IS the longest
program. The hemisphere tail is ~10 dispatches per `HEMI_BATCH` slice of the
framebuffer: 30000 px at 200x150 is 8 batches, 720p is 57. The literal would
have refused **every fb frame at every resolution**.

`push_rows_for(depth_full, px, hemi)` now derives it, and `run_steps` gates
against what was allocated rather than against a constant that can drift from
it. The refusal was already loud, which is what made this a sizing decision
rather than a soundness one.

### One blanket `clear_buffer` broke two gates, and only one of them said so

`run_steps` opened every frame with `enc.clear_buffer(&self.counters, ...)` —
added for the ladder, where it is what makes a "> 0" must-fire mean anything on
counters no kernel resets. Two kernels spell a KEEP-SET in HLSL, and a
host-side clear silently overrides it:

- **`cs_seed_replay` keeps CTR_LEAF / CTR_SKY / CTR_CUT / CTR_SKY_PX** — the
  entire terminal structure it is about to re-dispatch. Cleared, the fills
  launched zero groups and the replay wrote NOTHING. J9 said so immediately:
  120000 poison survivors, all three diffs at full count.
- **`cs_seed_probes` keeps the verify and stats counters across accumulate
  seeds**, so the exact-zero gates observe every seed's rays. Cleared, J8
  scored one seed in eight — **and PASSED**. The measurement: leaf-rays
  **344 → 2752** and empty-cells **19 → 152** once the clear was lifted.
  Exactly 8x, and the gate never said a word.

That second one is the transferable half. The loud failure was found in the
first run of a new gate; the quiet one was found only because fixing the loud
one required understanding what the keep-set was FOR. A gate scoring 1/8 of its
evidence reports the same verdicts as a gate scoring all of it — every counter
was still exactly 0, because the seven unobserved seeds were also correct.

The fix makes the clear a parameter, so the caller passes what its seed kernel
expects and the two statements of the keep-set cannot drift apart. The probes
path passes the same `clear` flag it hands `cs_seed_probes` as `push1` — one
decision, spelled once on each side.

### A fixed probe stride is a bar that means different things on different arms

J8's probe set uses V8's construction: a scattered lattice, `surface_point`ed
so both sides integrate at the exact same `(o, n)`. V8's strides are fixed at
(41, 53). This gate renders at 400x300 on hardware and **160x120 on a software
adapter — the CI arm** — where those strides yield SIX probes.

Every bar in J8 is a statement about a MEAN, and a mean's noise floor scales as
1/sqrt(N). At six probes the AO signed mean measured **-0.0050 against its
+/-0.005 bar**: a gate failing on sampling noise. Measured across three
resolutions the reading was **+0.0006 (46 probes) / -0.0049 (12) / -0.0050
(6)** — it straddles zero, which is what says the estimator is unbiased and the
SAMPLE was the defect.

The stride is derived from the frame now (9 candidate positions per axis), so
the probe count sits near 50 on every adapter class and a bar means the same
thing wherever it runs. Both arms then read **-0.0008**, hardware and software
alike. Widening the tolerance was the available alternative and would have
hidden the real bias this bar exists to catch.

### Teeth, both ways

- **The root-parity inversion** — the failure `vk/tracer.rs` calls silent, and
  it is: the root writes `hqout`, so running it under the EVEN variant lands
  its output in a queue nothing reads. Planted, J8 answers `leaf-rays 0`,
  `psa-viol 45/56` with max error **3.14** (the whole hemisphere unaccounted),
  AO mean |d| **0.5912** against a 0.02 bar and GI **100%** against 5%. The
  `leaf-rays 0` must-fire is the universal detector — psa-viol caught only
  45 of 56 probes in the AO arm.
- **The stale cbuffer** — `render_wavefront_replay` with its `write_cb`
  removed. J9's warm arm answers **45399 words** differing, and **arm 1 stays
  completely clean** (0/0/0). That is the two arms proven non-redundant rather
  than argued: arm 1 replays the same params, so both frames use the same
  cbuffer and it structurally cannot see this.

### Measured

Procedural at 400x300 (hardware) and 160x120 (the CI arm), 21 units,
9021.5 KB of WGSL, ~285 MB of tracer buffers of which ~280 MB is the
hemisphere tier's per-batch transients (three 64 MB cell queues and an 88 MB
cut pool — all under the 128 MB default `max_storage_buffer_binding_size`, so
the tier costs device memory without costing a limits row).

| | 400x300 | 160x120 |
|---|---|---|
| probes | 54 | 56 |
| psa-viol / false-empty / tmin-overshoot | 0 / 0 / 0 | 0 / 0 / 0 |
| AO mean \|d\| (limit 0.02) | 0.0070 | 0.0071 |
| AO signed mean (limit ±0.005) | -0.0008 | -0.0008 |
| GI mean rel (limit 5%) | 2.85% | 2.35% |
| hemi-points vs hit-px | 85346 = 85346 | 13655 = 13655 |
| J9 tbuf / info / accum diff | 0 / 0 / 0 | 0 / 0 / 0 |

The `hemi-points == hit-px` identity is exact rather than a tolerance, and it
is spelled with `gfx::shaders::web_miss` rather than `is_finite` — a
browser-corpus miss is FLT_MAX, the C2 constraint that once read 5545 of 19200
sky pixels as geometry. An accounting identity is exactly what that predicate
breaks silently.

Next: the display stage (J18) — `tonemap.hlsl` and `blit.hlsl` DRAWN rather
than dispatched, which needs `webgpu/layout.rs` to learn stage visibility and
to union a vs/ps pair into one pipeline layout. Then the shader bake, and then
Stage D: the web shell, and the first frustracer pixels in a browser tab.
