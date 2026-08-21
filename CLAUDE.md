# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

A **frustracer** — a frustum tracer. The screen is a quadtree: each tile's frustum is traced
against a BVH for a conservative nearest-possible-hit distance; on contact the tile splits
into 4 children that inherit that distance as their ray `tmin` **and a node cut**, bottoming
out at per-pixel rays seeded from the cut. Tiles whose frustum provably hits nothing are
filled with sky, tracing **zero rays** — proving space empty is the algorithm's product. The
same divide-and-conquer dispatches hemisphere bounce lighting. Lighting is **one sky**: a
Rayleigh+Mie scattering dome carried as order-2 SH, plus a sun disc that is cone-sampled and
shadow-rayed.

Three renderers over one shading model — the CPU tracer (the reference), a GPU-resident
wavefront tracer, and a by-the-book DXR pipeline — plus Vulkan and Metal backends, an
always-on temporal upscaler chain, and a pre-upscale denoiser slot.

`README.md` has the algorithm write-up.

## Where the documentation lives

Four tiers, in the order to consult them.

1. **Module headers.** `src/<module>.rs`'s doc comment is the spec for that feature: its
   invariants, its known-accepts, its measured costs, and its own "touch this → run that"
   list. This is the authoritative source. **Read it before changing a module.**
2. **`docs/history/`** — the campaign records: write-ups, measurement tables, the reasoning
   behind every default, and the bug each gate was written for. 15 files, one per subsystem;
   `docs/history/README.md` indexes them and maps gate-ID prefixes (`V*`, `N*`, `M*`, …) to
   the suite that owns them. *Reference*, not contract, and **not auto-loaded**.
3. **`CLAUDE_Historical.md`** — the notebook those were extracted from. Now a ~336 KB index:
   its `## Commands` block keeps every flag with a pointer to the file holding its story, and
   it still carries the cross-cutting prose sections (`## Correctness invariants`,
   `## Architecture notes`, the HDR/upscaler/real-scenes sections). Grep, never read whole:
   ```
   grep -rn 'CACHE_VERSION' docs/history/ CLAUDE_Historical.md   # a topic, anywhere
   grep -n  -- '--check-vk' CLAUDE_Historical.md                 # which file holds a flag
   grep -rn -- '--check-vk' docs/history/                        # that flag's full story
   grep -n  '^## ' CLAUDE_Historical.md                          # its prose sections
   ```
4. **`README.md`** (the algorithm) and `docs/design/`.

Older comments across this repo cite *"the Profiling section in CLAUDE.md"*, *"Real scenes in
CLAUDE.md"* and similar. `CLAUDE.md` is this file; those sections are in
`CLAUDE_Historical.md` or under `docs/history/` — grep both.

Reference the archive in backticks. **Never prefix it with an at-sign** — that syntax
inlines an entire file into context and would silently undo this split.

**The live HLSL and the `self_test` functions are the executable specification.** Where prose
and code disagree, the code is right.

## Build and run

```
cargo build --release
cargo run --release                    # interactive; boots THE WORLD (src/world.rs)
cargo run --release -- model.obj       # or .gltf/.glb
cargo run --release -- --help          # the live flag list
```

Rust edition 2024, single crate, ~150k lines. `[profile.quick]` (`lto = false`,
`codegen-units = 16`) rebuilds a one-line touch in ~25 s vs release's ~45 s and is what CI
builds with — but **no benchmark number may ever come from it.** Every measurement in this
project is only meaningful under `release`'s LTO settings.

Render modes: `--cpu` (the reference tracer), `--gpu` (wavefront), `--dxr` (the default on
NVIDIA/AMD; Intel defaults to the wavefront). In-session **SPACE** cycles all three and **F**
toggles CPU↔DXR; the CLI flags only pick the starting mode. **H** cycles hemisphere bounces
(off → AO → GI, still frames only), **ESC** opens the settings menu, **F1** the HUD.

Headless entry points, none of which need a window: `--check*` (the gates), `--spin`
(deterministic benchmark), `--cinematic` (media capture), `--frd-lab` (denoiser instrument),
`--qa [port]` (live control socket, driven by the `frqa` binary).

`src/cli.rs` and `--help` are the live authority on flags — §"Flag index" below is an index,
not documentation.

## Gates — the test suite

**There are essentially no unit tests.** `--check*` is the suite. The one exception is
`cargo test`: 25 shader-source gates in `src/gfx/shaders.rs` plus a field-coherence probe in
`src/gfx/guides.rs`, asserting ordering statements inside the HLSL that no CPU-only gate can
reach — and including a clean-room **licence** scan that our assembled shaders contain none
of NVIDIA's NRD entry names.

| gate | needs | covers |
|---|---|---|
| `--check` | nothing (DLL-free) | the CPU suite + every module `self_test`; writes the goldens |
| `--check-gpu` | real GPU + DXC | the wavefront tracer — **not in CI** |
| `--check-dxr` | real RT GPU + DXC | the DXR pipeline — **not in CI** |
| `--check-vk` | unix + Vulkan | the Vulkan backend, stages V0–V20 |
| `--check-spirv` | DXC/spirv-val (any OS) | the whole shader corpus → SPIR-V |
| `--check-wgsl` | DXC (any OS; naga is built in) | the BROWSER corpus → SPIR-V → naga-validated WGSL round-trip + W5 layout audit + W6 hostile scan + W7 tracked corpus golden (`goldens/web_corpus.txt`, regenerated via `--write-golden`; W0 is DXC-free) |
| `--check-wgpu` | any wgpu adapter (llvmpipe/WARP count) + DXC for J2+ | that chain's output EXECUTING on a WebGPU device — adapter/limits probe, the indirect smoke, **J6 the browser's reference tracer vs the CPU**, **J7 the wavefront quadtree vs that reference** (J0 is pure; the smoke KERNEL is scene-free but the DEVICE is scene-keyed from J1 on — the ask lands in `required_limits`) |
| `--check-msl` | macOS + spirv-cross | the corpus → MSL → metallib |
| `--check-mtl` | macOS + Metal device | the backend binding and DISPATCHING those metallibs |
| `--check-metalfx` | macOS | MetalFX temporal upscale/denoise |
| `--check-fsr3` | macOS | FSR3 over the hand-written Metal backend |
| `--check-dlss` `--check-xess` `--check-fsr` `--check-nrd` | nothing | DLL-free contract self-tests |
| `--check-oidn` `--check-nppd` | those DLLs + model | denoiser wiring end to end |

Each takes an optional scene and composes with `--stress N`. Exit **2** = environment
(missing DLL/GPU), **1** = a gate failed.

**CI** (`.github/workflows/ci.yml`, 5 jobs, all `--profile quick`): Windows/Linux/macOS run
`--check`, `--check-dlss`, `--check-xess`, `--check-fsr`, `--check-nrd`; a Vulkan job runs
`--check-spirv` + `--check-wgsl` + `--check-wgpu` + `--check-vk` with anti-vacuity greps; a Metal job runs
the four Metal gates; a wasm job holds `cargo check --target wasm32-unknown-unknown` green
from a bare checkout. No clippy or fmt job.

**The "touch X → run Y" convention.** ~35 such run-lists live in module headers and the
archive — find the one for what you touched and follow it. The floor is `--check` +
`cargo test` + a golden byte-compare.

## Invariants that fail silently

These are the rules where breaking them leaves **the suite green and the output wrong**.
Everything else you can discover by reading code; these you cannot.

### Soundness — the bug class the whole design guards

- These counters must be **exactly 0**, and are never a tolerance to widen: `false-sky`,
  `tmin-overshoot`, `claim-violation`, `hybrid-extra`, `psa-viol`, `false-empty`, `cut-miss`.
  A nonzero one is a real bug.
- Distances are **Euclidean from the shared camera origin** and all ray directions are
  **normalized**, so distance == ray parameter `t`. The proven-empty region is frustum ∩
  **ball** — never reintroduce a planar near clip.
- **Secondary rays never see the tile's inherited tmin.** It is a primary-frustum property.
  The hemisphere integrator keeps this by construction: its own apex, its own tmin chain
  starting at 0 (not eps — a ball(o, eps) claim is false at concave corners).
- **Never add distance-to-best pruning to `refine_cut`.** A far node can be the nearest thing
  in a sibling's frustum; pruning it surfaces as false sky. `d >= best` belongs to the bound
  query only.
- A **"blocked" query must still subdivide** — that is how sky tiles emerge. Children consume
  only `refine_cut` output.
- Frustum-vs-AABB culling may only err toward "intersecting". Erring inclusive costs
  efficiency, never correctness.

### One sky

- **The sun disc is delivered exactly once per light path.** Camera and glass misses see
  dome + disc; the specular reflection miss takes an MIS-weighted disc; **every gather path
  calls `sky::gather`, which carries no disc.** Breaking that double-counts light the direct
  loop already delivered *and* fires a ~1e3-radiance value into the hemisphere's fixed-point
  accumulator, which saturates.
- MIS may only down-weight the light-sampled specular when the BSDF-sampled half will
  actually run. Where the reflection-ray gate fails (low preset, any `depth > 0`), light
  sampling is the only estimator and must carry `w_l = 1`.
- **`BOUNCE_Q.ao_samples` must stay > 0.** At 0 every bounce surface is lit as if in an open
  field, GI collapses toward a flat ambient — brighter than no GI and visibly worse — and
  **every gate passes**, because estimator and oracle share the policy constant.

### Shading is not forked

`shade.rs` is the source of truth, `shade.hlsli` its one GPU port, and the CPU path is the
reference. Both GPU pipelines paste the same source. Never fork them.

### Feature-addition discipline

This project adds features in a recurring shape. Follow it.

- **Zero rng draws**, or the same-seed / replay / VisCtl-burn contracts break. If a path must
  draw, **burn the draws it skips** so the stream stays aligned.
- **The off-state must be structural** — a BRANCH, never a computed `* 1.0` or `+ 0.0`
  (`x + 0.0 != x` for `-0.0`). Prove it by byte-comparing the goldens.
- A GPU runtime lever needs **both halves**: the compile define *and* the CB flag bit. A
  compile-only lever fires inside gates that pinned the feature off.
- Constants duplicated across sites — a default in `cli::defaults()` + a module static + a
  settings-menu row; a CPU literal + its HLSL twin — **move in lockstep**. `cli::self_test`
  and `settings::self_test` pin several against each other; extend them rather than adding an
  unpinned fourth copy.
- Bump `scene_cache::CACHE_VERSION` on any Scene/Material/Texture on-disk repr, BVH build,
  loader or matclass change; add a `lever_word` bit for a lever that changes loaded data. A
  stale sidecar is served silently otherwise.

### Gate design

- **Anti-vacuity: a gate that cannot fire proves nothing.** Prove teeth **both ways** — the
  correct code passes AND the broken code provably fails. Several gates here shipped green
  while scoring nothing.
- A probe reporting "no effect" must first be proven to have **reached** its target. The
  `FR_ABL` probe-reach trap fired four separate times: an ablation define that never reached
  its compile unit compared identical code against itself and answered confidently. `FR_ABL`
  now announces the arms it matched; "matched GPU arms: (none)" on a non-empty value is the
  alarm.
- Pick the statistic from the failure you are hunting. A signed mean catches bias that an
  absolute mean cancels; a relative bound travels across scenes where an absolute one does
  not. Both mistakes have shipped.

### The goldens

- `check.png` / `check_gi.png` are **tracked** goldens of the default scene, byte-compared by
  hand — several bit-identity claims rest on nothing but a clean `git status`.
- A scene-keyed run writes `check-<tag>.png` instead (gitignored). **But `--tod` and
  `--rtgi-bounces` have no tag and DO overwrite the goldens.** Run the plain `--check`
  **last** in any sweep, or `git checkout -- check.png check_gi.png` afterwards.
- They are a **Windows** contract: `sinf`/`cosf`/`expf`/`powf` come from the system libm and
  may differ per platform. A cross-platform **pixel** diff is not a regression; a **counter**
  diff is. Never commit another platform's goldens.

## Working rules

**Source encoding.** Every file is UTF-8 **with no BOM**, and the comments are dense with
`—`, `×`, `→`, `≤`, `Δ`, `π`. **Never round-trip a source file through a PowerShell
pipeline**: PowerShell 5.1's `Get-Content` decodes a BOM-less file as cp1252 while
`Out-File`/`>` writes UTF-8 *with* BOM, so `Get-Content f | ... | Out-File f` double-encodes
every non-ASCII character and prepends a BOM. It still compiles and every gate passes — the
damage is comments-only — so nothing catches it. Use the Edit tool or `python3`; pass
`-Encoding utf8` on **both** ends if you must use PowerShell. Install the guard once per
clone: `cp tools/hooks/pre-commit .git/hooks/pre-commit` (not via `core.hooksPath`, which
would orphan git-lfs's four hooks).

**Line endings.** The working tree is CRLF (`core.autocrlf=true`); git stores LF. A raw
`cmp` between a working file and its blob will differ at every line ending — that is
expected, not corruption.

**Shell.** PowerShell is primary; the Bash tool is also available and takes POSIX syntax.
Never build via Git Bash (it shadows `link.exe`).

**git lfs.** Run `git lfs install` once per clone or scene checkouts are pointer files.
Verify `git check-attr filter <scene file>` says `lfs` before committing one — a scene blob
reaching plain git history is permanent bloat.

**Measurement discipline.** Each of these was learned by publishing a wrong number.

- Always `--release`, never `[profile.quick]`.
- **Interleave arms and take medians.** A single sample is worthless on NVIDIA (one unchanged
  config spans 1.42–1.98 ms); the Intel B70 and the R9700 repeat to ±0.002.
- **On Intel Arc, discard a warm-up run per shader variant.** The driver returns an
  unoptimized binary and recompiles ~5–8 s later on a wall clock, so a short run is 100%
  fallback and reads ~4.7× slow — and it *repeats*, so two agreeing runs prove nothing. Every
  point in a config sweep is a new variant.
- **Parse the LAST `--gpu-timing` table**, not the first — the first covers frames 0–119, the
  coldest window there is. Prefer the windowed column over the cumulative one.
- On this CPU box, sustained 32-thread load thermally destabilizes: use **min-of-N with
  cooldowns** and short runs, never the median.
- **Difference against the same run's own baseline**, never across processes — cross-session
  noise has swamped real effects more than once.
- An instrument at the wrong **resolution** cannot see what it was built for. Check that a
  probe can express the effect before trusting a null result.

**Before asking the user to look at something**, drive it yourself: `--frd-lab` (batch) and
`--qa` + `frqa` (live socket) exist so visual regressions can be reproduced and measured
without a human feel-test.

## Flag index

An **index, not documentation** — one line per family, pointing into the archive. 220 flags;
`--help` and `src/cli.rs` are authoritative. Flag names and keyword values are ASCII-case-
insensitive; paths, scene files and `settings:<Group>` names are taken verbatim. Most `--x` features have a `--no-x` twin that
spells the opposite; A/B levers are generally bit-identical when off.

**Scene & camera** — `<path>.obj|.gltf|.glb` (positional) · `--world` / `--no-world` ·
`--stress N` · `--tile NxM` · `--cam ex,ey,ez,tx,ty,tz` · `--tod H`

**Render mode & resolution** — `--cpu` `--gpu` `--dxr` · `--dxr-inline 0|1|2|3` ·
`--dxr-sbt 0|1|2|3` · `--lock-res native|quality|dynamic|<ratio>` · `--spp N` ·
`--sw-rays` / `--continuation-rays` · `--dual-gpu N` `--dual-gpu-auto|-arm|-depth`

**Upscalers & frame generation** — `--dlss` `--fsr` `--fsr4` (required, not preferred)
`--fsr3` `--xess` `--no-upscale` · `--quinlight` `--quin-anchor N` · `--fg` / `--no-fg` ·
`--fsr-max-radiance` and the `--fsr-*` denoiser tuning family · `--xess-autoexposure`
`--no-adaptive`

**Denoisers** (one slot) — `--nrd` (default) + the `--nrd-*` tuning family + `--nrd-perf` ·
`--frd` + `--frd-*` · `--oidn` + `--oidn-*` · `--nppd` + `--nppd-*` · `--rr-emis-demod`
(the DLSS-RR emissive-demodulation A/B arm, default OFF — emission rides RR's temporal
integration; armed, an on-screen emitter no longer lifts the frame but emitters shimmer)

**Lighting & sky** — `--rtgi-bounces 0..2` / `--rtgi` / `--no-rtgi` · `--emissive-lights [N]`
`--el-cluster grid|som` · `--fireflies N` · `--no-clouds` `--cloud-shadow N` `--sky-lod K` ·
`--no-amb-bump` `--no-spec-aa` · `--no-bloom` `--bloom-kernel box|wide` · `--auto-exposure` `--autoexp-mode
lights|tonemap` `--exposure-bias EV` `--autoexp-spike-guard` / `-strength`

**Materials & texturing** — `--no-mips` `--aniso N` `--no-slope-mips` · `--no-bc7`
`--bc7-cpu` `--bc7-quality` · `--normal-strength` `--no-h2n` `--no-n2h` `--heightfield` ·
`--no-detail-tex` `--detail-strength` `--no-detail-ao` `--detail-ao-strength`
`--detail-untex-scale` · `--no-tinted-shadows` `--no-spray` `--no-depth-tint` `--no-water`
`--no-coincident-cull`

**Geometry & acceleration** — `--bvh-builder sah|lbvh|ploc|som` `--bvh-ctrav` `--bvh-axes`
`--bvh-maxleaf` · `--blas-split [N]` / `--no-blas-split` · `--no-ftree` `--ftree-tiles`
`--no-wide-levels` · `--no-cut-rays` `--cut-hemi` · `--foliage-sway` `--foliage-amp`

**Temporal reuse** (A/B levers) — `--no-temporal` `--no-replay` `--no-adopt`
`--discard-seeds` `--no-hemi-share`

**Display & session** — `--no-hdr` `--hdr10` `--no-hdr10` `--hdr-paper-white` `--hdr-peak` ·
`--no-vsync` · `--move-ease S` / `--no-move-ease` · `--no-audio` · `--no-settings` ·
`--prefer-nvidia|amd|intel`

**Headless modes** — the 16 `--check*` gates · `--write-golden` (regenerates the `--check-wgsl`
W7 corpus golden; refused unless W0–W6 green) · `--spin still|path` + `--spin-frames|-warmup|
-hybrid|-plain` · `--cinematic <preset>` + the `--cinematic-*` family (res/fps/frames/samples/
island/gi/overlay/hud/hdr/exposure/out/encode/dry-run) · `--frd-lab <kind>` + `--frd-lab-*` ·
`--bloom-lab [wobble]` (glare shift-variance probe; scene-free, GPU-free) · `--qa [port]`

**Diagnostics** — `--gpu-debug` (debug layer + GBV) · `--gpu-timing` · `--pix-markers` ·
`--waveviz [chs]` · `--cam-readout` (HUD pose plate: position, quaternion, a paste-ready
`--cam`, TOD + live aperture — so a screenshot reproduces a pose) · `--no-crash-handler` ·
`--dlss-dump` `--xess-dump` `--oidn-dump` `--nppd-dump`

**SDK paths** — `--dxc-path` `--ffx-path` `--fg-path` `--nrd-path` `--oidn-path`
`--xess-path` `--nppd-path` `--pix-path`, each with a `FRUSTRACER_*_PATH` env twin.

**Env levers** (~95 `FR_*`, all default-off, loud when armed, bit-identical when off) —
ablation and probes: `FR_ABL` `FR_BALLAST` `FR_WIDTH` `FR_ORACLE` `FR_RANGE` `FR_REF` ·
tuning sweeps: `FR_LEAF` `FR_LGROUP` `FR_LSTACK` `FR_WIDE` `FR_STACK_LAYOUT` `FR_FRD_GROUP` ·
per-feature A/B: `FR_NGXFG_*` `FR_NGXRR_*` `FR_NRD_*` `FR_FRD_*` `FR_MFX*` `FR_AEXP_*`
`FR_MTL_*` (the `--check-mtl` plants: eight teeth and measurements over the argument-buffer
map, the threadgroup size and residency) · `FR_MTL4_*` (the Metal 4 path's five: the argument
table's bind point, the inter-dispatch barriers and the commit-feedback handler are TEETH,
residency is a measurement, and `FR_MTL4_OFF` forces the SKIP branch on a box that has
MTL4) · `FR_FFX_MSL` — **the one
BUILD-time lever**, so it needs `cargo build` and the gate run to see it alike, and a stale
binary measures the other arm
`FR_DXR_LEAN` `FR_DXR_STACK` `FR_FG_*` `FR_RTGI_NOWEIGHT` `FR_SWAY_*` `FR_WEB_TEX`
(the wavefront samples texweb buckets — the browser texture plan on native D3D12;
`--check-gpu` M15 owns it) · Vulkan: `FR_VK_*`
(device pick, validation, res parity, drop-binding teeth, the window's pump arm) · WebGPU:
`FR_WGPU_ADAPTER` (adapter pick, index or name substring; `WGPU_BACKEND` rides along free)
`FR_WGPU_RES` `FR_WGPU_AB_FRAMES` `FR_WGPU_MAP` (J6's frame, its A/B depth, and the
REAL-vs-dummy bind report — the `FR_VK_RES`/`FR_VK_AB_FRAMES`/`FR_VK_MAP` twins) · dumps: `FR_DUMP_HLSL`
`FR_SPIRV_DUMP|LIST` `FR_MSL_LIST` `FR_CHECK_AB_DUMP` `FR_SPLIT_AUDIT` · crash:
`FR_CRASH_TEST|FULLDUMP|VERIFY` `FR_NO_CRASH` · plus `FRUSTRACER_STAB` (inter-frame
stability readout) and `FRUSTRACER_HUD_STATS`.

## Layout

```
src/               117 .rs, ~150k lines  (main.rs alone is 39k — session loop, gates, modes)
  cli.rs           the 220-flag parser; parse_from is PURE (no globals) so --check can gate it
  render.rs shade.rs bvh.rs frustum.rs ftree.rs   the tracer core; shade.rs is the shading spec
  hemi.rs sphcell.rs sky.rs sh.rs clouds.rs       bounce integrator + the one sky
  scene.rs scene_cache.rs world.rs matclass.rs texture.rs gltf_loader.rs   assets and loading
  temporal.rs replay.rs                           cross-frame reuse
  gpu/           D3D12: wavefront tracer, DXR pipeline, upscaler/denoiser/FG wiring
  vk/            Vulkan backend (unix)          mtl/   Metal backend (macOS)
  gfx/           backend-neutral core: shader assembly, FrameCb, denoiser vocabulary
  hud/           Slint software-rendered HUD and pause menu
  shaders/       43 HLSL files — the corpus all three backends compile
  bin/frqa.rs    the --qa socket driver
shim/            15 C++/ObjC++ shims (DLSS-D/G, FFX FSR3 D3D12/VK/Metal, NGX)
SDKs/            gitignored; install-prerequisites.{sh,bat} fetches/builds them
scenes/          git LFS; .obj.zst + lossless WebP
tools/           pre-commit hook, dump-hlsl.ps1, win-cross-check.sh, media-encode.py
docs/history/    the campaign archive, 15 files + an index (see tier 2 above)
docs/            design notes, the Intel B70 RT brief, media stills, papers
```

`build.rs` compiles the shims, builds FidelityFX 1.1.4 from source off Windows, transpiles
SPIR-V→MSL for Metal, and **hard-fails** on a missing NRD artifact (NRD is the default
denoiser, so a tree that cannot produce it renders undenoised in silence).
