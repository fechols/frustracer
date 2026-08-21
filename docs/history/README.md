# docs/history — the design notebook's campaign records

The engineering archive: campaign write-ups, measurement tables, the reasoning behind every
default, and the bug each gate was written for. It is **reference, not contract** — the rules
you must not break live in `CLAUDE.md` at the repository root.

None of this is auto-loaded. These files were extracted **verbatim** from
`CLAUDE_Historical.md`, which keeps a stub for every entry pointing here, so its `## Commands`
block still works as a complete flag index.

## The files, in document order

| file | size | holds |
|---|---:|---|
| [getting-started.md](getting-started.md) | 13 KB | build, THE WORLD, loading a model, `--cam` |
| [sky-lighting-clouds.md](sky-lighting-clouds.md) | 62 KB | `--tod`, fireflies, emissive lights, volumetric clouds, `--sky-lod` |
| [upscalers-and-framegen.md](upscalers-and-framegen.md) | 76 KB | the upscaler chain, OIDN, XeSS, FSR/FSR4/FSR3, all three FG legs |
| [denoisers.md](denoisers.md) | 158 KB | NPPD, NRD/ReBLUR, FRD, and the AI QA lab (`--frd-lab`, `--qa`) |
| [exposure-camera-session.md](exposure-camera-session.md) | 75 KB | temporal levers, bloom, auto-exposure, `--move-ease`, crash handling |
| [materials-and-textures.md](materials-and-textures.md) | 54 KB | mips, anisotropy, heightfield relief, detail texturing, tinted shadows |
| [gi-and-acceleration.md](gi-and-acceleration.md) | 67 KB | the `--rtgi-bounces` GI ladder, BVH builders, `--blas-split`, `--dual-gpu` |
| [tracing-scheduling.md](tracing-scheduling.md) | 30 KB | cut-seeded rays, the frustum tree, wide levels, `--spp`, `--lock-res` |
| [gpu-and-dxr.md](gpu-and-dxr.md) | 30 KB | `--gpu`, `--check-gpu`, `--dxr`, `--dxr-inline`, `--dxr-sbt` |
| [shader-toolchains.md](shader-toolchains.md) | 29 KB | `--check-spirv`, `--check-msl` — the corpus's second and third code generators |
| [vulkan-backend.md](vulkan-backend.md) | 226 KB | `--check-vk`, stages V0–V20. The largest entry in the notebook |
| [metal-backend.md](metal-backend.md) | 60 KB | `--check-fsr3`, `--check-metalfx`, `--check-mtl` |
| [tooling-and-capture.md](tooling-and-capture.md) | 30 KB | Tracy, `--quinlight`, `--spin`, `--cinematic`, settings, HDR, GPU timing |
| [profiling.md](profiling.md) | 41 KB | the `## Profiling` section — Tracy, PIX, timestamp queries, the perf campaigns |
| [intel-arc-xe2.md](intel-arc-xe2.md) | 38 KB | the `## Intel Arc / Xe2` section — what the hardware does and does not offer |
| [web-backend.md](web-backend.md) | 9 KB | the browser port (WASM + WebGPU): the wasm compile guard, `--check-wgsl`, naga's measured verdict. NOT an extraction — written live as the campaign runs |

Two entries sit in a topically adjacent file rather than their own, because the extraction
preserved the original document order exactly: **`--no-audio`** is in `sky-lighting-clouds.md`
and **`--move-ease` / `--no-move-ease`** in `exposure-camera-session.md`.

## How to search

```
grep -rn 'CACHE_VERSION' docs/history/ CLAUDE_Historical.md   # a topic, anywhere
grep -rn -- '--check-vk' docs/history/                        # a flag's story
grep -n  -- '--check-vk' CLAUDE_Historical.md                 # which file holds it
grep -rn 'the sky_sh precedent' docs/history/                 # a named precedent
```

The notebook refers to itself almost entirely **by name** — `the sky_sh precedent`, `the
spp_defs idiom`, `the --fsr4 doctrine`, `the probe-reach class` — rather than by position.
Those names are unique tokens, so grep lands you in whichever file now holds them. That
property is why this archive could be split at all.

## Gate-ID namespaces

Each gate suite numbers its own checks, and the prefixes are reused across suites. This maps a
prefix to the suite that owns it and the file where its story lives:

| IDs | suite | file |
|---|---|---|
| `V0`–`V20` | `--check-vk` | vulkan-backend.md |
| `S0`–`S3` | `--check-spirv` | shader-toolchains.md |
| `M0`–`M5` | `--check-msl` | shader-toolchains.md |
| `M1`–`M13` | `--check-gpu` | gpu-and-dxr.md |
| `T1`–`T4` | `--check-dxr` | gpu-and-dxr.md |
| `N0`–`N11` | NRD gates (`--check-nrd`, `--check-gpu`) | denoisers.md |
| `F0`–`F10` | FRD gates (`--check-gpu`) | denoisers.md |
| `U0`–`U4` | `--check-fsr3` | metal-backend.md |
| `X0`–`X8` | `--check-metalfx` (X7–X8 are D5's Metal 4 arm and are M1-only — the CI runner has no Metal 4, so they SKIP there) | metal-backend.md |
| `K0`–`K11` | `--check-mtl` (K5 is the VERDICT and stays last-numbered-first — K6–K8, K9–K10 and K11 all came later, and renumbering it would strand this table's own references) | metal-backend.md |
| `G1`–`G14` | `clouds::self_test` | sky-lighting-clouds.md |
| `W0`–`W7` | `--check-wgsl` (the browser corpus; W5–W7 not yet built) | web-backend.md |
| `J0`–`J3` | `--check-wgpu` (the WebGPU host; more stages join at Stage C2 — `U*` being taken is why it is not that) | web-backend.md |

**Two hazards.** The `M` prefix is genuinely ambiguous — `--check-gpu` and `--check-msl` both
use it, so `M1` means different things in `gpu-and-dxr.md` and `shader-toolchains.md`. And a
bare grep for a short ID also hits ordinary prose: `M1` is Apple silicon, `F0` is the Fresnel
term, `N` is a count (`N=32`), `V` and `F11` are keyboard keys, and `B70` is an Intel GPU. The
notebook qualifies about a third of its cross-suite references itself (`` `--check-gpu`'s N4 ``);
the rest are bare, so prefer searching for the surrounding phrase over the ID alone.

## Provenance

Extracted 2026-08-12 from `CLAUDE_Historical.md` (1,328,823 bytes) by a scripted, contiguous,
byte-exact split: reassembling every file's body in document order reproduces the original
file's SHA-256 (`487e5a2f…eccab1`) exactly. No text was rewritten, reordered, or reformatted.
