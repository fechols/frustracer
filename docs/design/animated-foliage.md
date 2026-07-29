# Animated foliage via tetrahedral cages — design doc

**Status: v0.1, DEFAULT ON (2026-07-28)** — src/foliage.rs: the leaves-only /
translation-per-cell / DXR-only cut described under "Phase 1", minus tets and
clipping (leaves are disconnected cutout geometry, so nothing can tear — the
clipping machinery is deferred with the per-tet affine).
`--no-foliage-sway` is the kill lever (bit-identical off); `--foliage-sway`
spells the default. The default flip (same day, after the user's visual
verdict on v0.1) is safe on three structural grounds: a scene with no
foliage-classed materials leaves the BLAS plan bit-identical (`split_plan`
returns None — the apply_tod-unreachable shape, so procedural/stress/glTF
sessions are the pre-feature renderer); every headless path (`--check*`,
`--spin`, cinematic) pins `sway_time: None`, so no benchmark or gate has
geometry move under it and `--spin` numbers stay rest-pose comparable; and
the armed structural change (the re-partitioned chunk remap + the animated
ring beside the static TLAS) was already gate-proven at v0 (exact-zero
counters 0, same-seed image A/B exactly 0.00e0). `--check-gpu`/`--check-dxr`
on foliage scenes therefore now gate the ARMED config by default — which is
the shipping config, the heightfield lesson in reverse. Gates green at
landing: `--check` (foliage self-test synthetic + mixed-mask arms; check.png
byte-identical), `san-miguel-low-poly --check`, and `--check-dxr` +
`--check-gpu` on San Miguel (1.4M leaf tris → 294 cells). The rest of this
document is the full epic the prototype was cut from.

**v0.1 (same day, after the first visual smoke)** — two user-reported
problems, both fixed:

- *"~100× too much motion"*: `SWAY_AMP_K` 0.010 → 0.0003 and `SWAY_BOB_K`
  0.0015 → 0.00005 (v0's constants were ~25 cm of rigid per-cell translation
  plus ~4 cm of flutter jitter on San Miguel — an earthquake, not a breeze;
  now ~1 cm + mm). `--foliage-amp <x>` (0..=8, default 1) is the taste
  multiplier, applied at BAKE time only so the cell partition stays
  knob-independent; the self-test sweeps the displacement bound at several
  mults through the internal `translation_with`.
- *"Only one scene sways"*: the v0 leaf mask re-derived the foliage class
  from the TEXTURE STEM, and the matclass vocabulary was San Miguel's.
  Per-scene root cause: bistro's stems/names say "leaves"/"foliage" (absent
  from the table); rungholt/vokselia carry the signal ONLY on the material
  NAME (`Leaves`, `Sapling` — one shared alpha atlas texture), which
  classify consults at load but nothing retained. Fix: (a) the foliage class
  gained `leaves/foliage/sapling/plants` tokens and stone gained
  `pavement/cobblestone/cobble` (so bistro's leaf-LITTER pavement classifies
  by its stem before the "leaves" name token fires); deliberately NO "grass"
  — Minecraft's `Grass` is the GROUND block. (b) `Material` retains the
  classify verdict as a `class: u8` (DiskMat + CACHE_VERSION 10 → 11), and
  `leaf_materials` reads the byte — the design's "class byte on Material"
  option, which also deleted the stem re-derivation hack. Coverage after:
  san-miguel-lp 1,401,749 leaf tris → 294 cells (identical to v0 — the byte
  reproduces the stem mask there), bistro exterior 764,492 → 99 (7 of its
  10 foliage materials sway; trunks/branches excluded by opaque bark),
  rungholt 2,226,024 → 423; vokselia rides the same MTL vocabulary; THE
  WORLD inherits all of them through the material merge. Sponza-khronos /
  helmet / powerplant have no foliage (verified) and stay static. Gates
  re-run green: `--check` (+ `--stress 5000`) with check PNGs
  byte-identical, cold SM-lp/rungholt/bistro `--check`,
  `--check-dxr --foliage-sway` on all three (class-mismatch 0, radiance A/B
  0.002–0.024%), `--check-gpu --foliage-sway` on SM-lp (exact-zero counters
  0, same-seed 0.00e0). Known wart, accepted: a handful of rungholt/bistro
  materials shift class (leaf-litter pavement → stone, Minecraft Leaves →
  foliage translucency 0.3) — a shading improvement, absorbed by the
  statistical gates.

**Original status: SCOPED, NOT BUILT.** This is the design record for the one
animated-geometry feature this renderer could plausibly want — wind-swaying
foliage in THE WORLD — and the honest bill for it. Nothing in this document
ships anything; if the epic is built, this content graduates into a CLAUDE.md
`##` section plus the new module's `//!` header (the repo's convention for
shipped features), and this file becomes the archaeology.

The enabling prior work is Gruen, Benthin, Kern & McAllister, **Ray Tracing
Massive Amounts of Animated Geometry**, Proc. ACM Comput. Graph. Interact.
Tech. 9(4), July 2026 — https://doi.org/10.1145/3820014. The PDF is
deliberately NOT committed to `docs/papers/`: its front page grants "personal
use" and says "Not for redistribution" explicitly, which is *stronger* than
the no-explicit-grant posture the two committed papers carry (the intel-sponza
precedent — either reason alone would be sufficient). Cite by DOI.

## What the paper contributes, in one paragraph

Tetrahedral cages decouple animation cost from triangle density. Preprocess:
voxelize the rest pose, split occupied voxels into 6 tetrahedra (Freudenthal),
clip triangles against each tet (1.3–2.3× triangle inflation, deduped shared
vertices), build one small **static** BLAS per tet. Per frame: animate ONLY
the cage vertices (2–3 orders of magnitude fewer than triangles), express each
tet's deformation as a DXR **instance transform** — the rest-tet basis M⁻¹
composed with the animated-tet basis A (paper §4.1) — and rebuild only the
TLAS over tets. Measured on an RX 9070 XT: 585M uniquely animated triangles at
60 fps; 2.8M tets → 9.66 ms TLAS rebuild (`PREFER_FAST_BUILD`). The honest
catch: the only real-time variant is **not watertight** (mitigated by
ε-growing tets before clipping; measured zero escapes on a sphere probe, not
guaranteed), and the watertight variant (4D barycentric encoding + software
4D BVH) is 19–80× slower — kept by the authors purely as a validation
reference, which is this repo's own `--check` discipline in different clothes.

## Why this composes with frustracer at all

Frustracer has **no animated geometry anywhere** — every moving thing (clouds,
fireflies, ripples, TOD) is deliberately shading-only, and every visibility
structure (frustum claims, temporal ring, structure replay, `.fcache`) is
premised on a static world. The paper's own Fig. 10e concedes render time is
slightly *worse* than a standard BLAS — it wins on update time and memory,
two costs a static scene never pays. So there is no static-scene optimization
to lift, and this epic is the only consumer.

The deep composition is a clean split of labor:

- **Animated-exact inner structure (the paper's spine).** Tet cages over
  foliage-tagged geometry; one small static BLAS per tet, built once and
  compacted; per frame, animate cage vertices, recompute per-tet 3×4 instance
  transforms, rebuild the TLAS. At our scale (~1e3–1e4 tets against the
  paper's 2.8M → 9.66 ms) the per-frame TLAS cost is well under a
  millisecond — but Phase 0 measures it rather than asserting it.
- **Static-conservative outer structure (frustracer's spine).** The frustum
  quadtree, temporal claims, and node cuts never see per-frame motion. The
  wind is the existing curl field — `clouds::curl_offset` (clouds.rs:632):
  time-independent, zero rng draws, soft-normalized |v| < 1, and
  `CLOUD_CURL_AMP_K · diag` is an **exact** displacement bound (gate G10b
  pins it). The consumption template is `fireflies::pose` (fireflies.rs:224):
  a static field, a time-shifted lookup point, and an amplitude constant that
  IS the bound — motion bounded **by construction**, no clamp, no runtime
  test.
- **The soundness composition.** The paper's Appendix A proves (interval
  arithmetic over 4D barycentric bounds) that deformed geometry stays inside
  its animated tet hull under ANY vertex motion — even inversion. Our motion
  is bounded, so a tet AABB swept ONCE at build by the exact bound contains
  every animated hull the session can ever produce. Therefore every frustum
  claim built on swept boxes — inherited `t_start`, cuts, empty-tile proofs,
  temporal-ring entries — remains a conservative lower bound under animation,
  with **zero per-frame frustum work**. Static outer proof, animated inner
  exactness: each system does the thing it is sound at.

## Design decisions

### 1. Pad at TET granularity, never per-triangle

`bvh::grow_height_sweep` (bvh.rs:1219) is the build-time pad idiom, and its
measured lesson is the hazard: an all-axis pad at pad ≫ triangle size wrecked
BVH quality 4× on DamagedHelmet (596 vs 146 ms/frame with the feature OFF).
Foliage leaf triangles are tiny against a sway amplitude — the exact
pathological regime. The escape is structural: the frustum bound query never
touches a triangle (a leaf sets `best` to the BOX distance), so the foliage
regions of the software BVH hold **swept tet-hull AABBs as proxy leaves**
instead of per-triangle boxes. The pad/box ratio is healthy at tet scale
(tet ≈ tree/N per axis vs sway ≈ a small fraction of the tree), and
non-foliage geometry keeps its exact unpadded boxes.

### 2. Foliage tagging

matclass classes are load-time-only `usize` indices and are NOT retained on
`Material` (scene.rs:1798–1806 folds the classification into
roughness/translucency and keeps only a histogram; `tri_mat` is the only
per-triangle metadata in `Scene`). Two enabling options, either small:
export the private `matclass::keyword_class` (matclass.rs:223) and re-run it
over `Texture::source` stems via `tri_mat → MatKind::Textured{tex} →
textures[tex].source`, or store a class byte on `Material` (a
`CACHE_VERSION` bump — the on-disk-repr rule). No tri→island map survives
`world::merge_scenes` (world.rs:301–328 — the per-part bases are locals), but
islands are spatially disjoint by construction, so `Island{center, radius,
height}` boxes give an unambiguous spatial scope for per-island cage builds.

### 3. GPU-first scope; the CPU reference is the epic's hardest question

The CPU renderer is the reference for everything. Two options, presented
honestly:

- **(a) True parity**: software two-level traversal — the CPU BVH gains
  tet-instance leaves and every ray crossing one applies the per-tet basis
  change (the paper's §4.1 in software). Correct, large, and it drags the
  whole verify/temporal/hemi surface with it.
- **(b) The fireflies shape (RECOMMENDED for v1)**: the feature is
  world/interactive-only; the CPU arm renders the rest pose with one loud
  line; gates run as GPU pairs (wavefront vs DXR, on-vs-off A/Bs) plus a
  DLL-free `self_test` for the pure math. Precedent: fireflies and audio are
  already features whose full experience is interactive-session-only, and
  `--check*` never loads the world, so no structural gate moves.

Option (b) is the difference between an epic and an unshippable one. The doc
records (a) as the follow-on if the feature earns it.

### 4. Per-frame GPU plumbing — all greenfield today

The TLAS is build-once: no `ALLOW_UPDATE` anywhere in the tree, `NumDescs`
baked, and the instance-desc upload buffer is dropped immediately after the
build (trace.rs:2360, 2374, 2997). Needed:

- Keep instance descs resident in a `FRAMES_IN_FLIGHT`-sliced `UploadBuffer`
  ring (the `frame_cb` shape, trace.rs:4155), rewritten per frame for
  foliage instances only.
- Per-frame `BuildRaytracingAccelerationStructure` recorded on the frame's
  own list. The paper rebuilds (`PREFER_FAST_BUILD`) rather than refits its
  tetLAS every frame; follow that (a refit degrades under large sway, and
  rebuild cost at our instance counts is the thing Phase 0 prices).
- The CB is NOT the vehicle for transforms: `CB_STRIDE = 2560` with a
  compile-time size assert, and 64 firefly rows already spend 1 KB of it.
  Per-tet 3×4 matrices ride their own SRV/upload ring. The firefly rows are
  the precedent for the *bake* (CPU computes poses once per frame, both
  renderers read bit-equal data), not for the capacity.

### 5. Non-identity transforms reach the hit sites

`tri_of(inst, prim)` (trace_common.hlsli:1400) is transform-agnostic — the id
remap does not care where the geometry is. But every hit fetch today reads
`positions[]` as object == world space. Foliage instances need
`ObjectToWorld3x4()` folded in at the hit-attribute sites (normals via the
inverse transpose — paper §6), or equivalently: shade from rest-pose
attributes and transform the results, which is the paper's own shading model
(they fetch non-deformed vertex normals and skin them). Either way this is a
per-site audit of `rt.hlsli` / `rt_dxr.hlsli` / `dxr.hlsl` / `shade.hlsli`
consumers, gated by the existing statistical GPU-vs-CPU A/Bs on a
sway-frozen pose.

### 6. Structure replay SURVIVES on the GPU — and why

Replay soundness is "the terminal structure is a pure function of
(scene, BVH, basis, rw, rh)" (replay.rs:6–15). Under this design the quadtree
structure depends only on the **swept-static** software BVH — the animated
TLAS is consumed exclusively by leaf rays, which are fresh every dispatch. So
a still frame's replay stays valid while the foliage sways through it, the
same way spp/jitter/frame/clouds already ride the CB across a replay. The
temporal ring in GPU/DXR modes is already dropped every frame
(main.rs:14009, 14986), so no new invalidation is needed there. One clock
rule delivers converging stills: plain accumulation advances `cloud_time`
only at frame 0 (the clouds/fireflies precedent), so an accumulating still
integrates ONE frozen pose — mid-flight-frozen fireflies are the accepted
look, mid-sway-frozen foliage is the same accept.

CPU-side replay/temporal need nothing in v1 under decision 3(b): the CPU arm
renders the rest pose, whose structures are exactly today's.

### 7. Clock and determinism

The sway animates on `cloud_time` (upscaler/denoiser frames advance every
frame; plain accumulation only at frame 0; `--spin` = idx·CLOUD_SPIN_DT;
every `--check*` pins CLOUD_CHECK_TIME). Zero rng draws anywhere in cage
animation, transform baking, or traversal — poses and matrices are pure
functions of (position, time), so every same-seed / replay / VisCtl-burn
contract holds structurally, not by luck.

### 8. Watertightness policy — the one honest concession

Adopt the paper's non-watertight variant with ε-grown tets (grow the four
planes outward before clipping). It is the only real-time option — their
watertight 4D-barycentric variant costs 19–80×. A watertightness escape is
exactly this repo's `false-sky` class, so state the concession plainly: rays
whose path crosses **animated** geometry get a bounded-count allowance in the
gates (the AMD TMin-re-origining precedent — a bounded count, never a
widened tolerance), while everything static keeps exact-zero. Two mitigating
facts: the ε-grow errs toward duplicate/nearer hits, which the soundness
spine tolerates (extra hits only pull t nearer — conservative); and the
paper's measured escape rate is zero on their sphere probe even before
duplicates. Cutout foliage keeps working unchanged — µBLASes are ordinary
triangle BLASes, so the any-hit/candidate cutout path applies verbatim; the
care point is UV/attribute interpolation through clipped triangles (clipping
must interpolate texcoords exactly or cutout masks crawl at tet boundaries).

### 9. Motion vectors — worse than the clouds accept, and bounded

Sway is a VISIBILITY change with no MVs, which is a harder ask of the
upscalers than cloud drift (shading-only). v1 accepts ghosting at a bounded
amplitude — the sway constant is authored small (a foliage flutter, not a
storm), and the amplitude constant is simultaneously the BVH sweep pad, so
taste and cost push the same direction. The follow-on is nearly free by
construction: per-tet MVs fall out of the prev/cur instance transform pair
the animation already computes (project the rest-space hit through both).

### 10. Preprocess and caching

Voxelize the foliage-tagged subset per island → 6-tet Freudenthal split →
clip triangles per tet (dedup shared vertices; expect the paper's 1.3–2.3×
triangle inflation on the foliage subset only) → per-tet BLAS at scene
upload. The cage (tets, clipped mesh, rest-basis matrices) rides the world
sidecar (`scenes/world.fcache`): `WORLD_VERSION` + `CACHE_VERSION` bumps, and
the feature lever keys `lever_word` so an A/B can never serve a stale
sidecar.

### 11. The lever

`--foliage-sway` / `--no-foliage-sway` in cli.rs (one `Opts` field + one line
in main's lever block — cli.rs:21–22 is explicit that a setter in the parse
loop un-gates the parser). The off arm builds no cages, keeps identity
transforms and the build-once TLAS, and is **bit-identical by construction**
(the apply_tod-unreachable / fireflies-count-0 class), which is what keeps
every existing gate untouched. Default OFF until the kill criterion below is
met; flip the default only with the measured numbers in hand.

## Phasing

- **Phase 0 — price the per-frame TLAS (cheap, standalone value).** An
  oracle.rs-style env-armed spike (`FR_TLAS_REBUILD=1`; behavior-free —
  identical identity transforms, so the image cannot move) that rebuilds the
  TLAS every frame at the current ~890 chunks and at synthetic 10k/100k
  instance counts, measured via `--gpu-timing` on the 4090 and the B70. This
  prices the enabling cost before any cage code exists, and doubles as the
  measured verdict on the cut-driven-TLAS idea (`BlasPlan::chunk_node`,
  blas_split.rs:77–80, is the waiting hook; prediction: too small to build —
  890 instances is nothing to an RT core — but the culture here is to
  measure, and the negative result is worth recording either way).
- **Phase 1 — one island, DXR mode only.** Cages over foliage-classified
  geometry in one island (bistro's trees or san-miguel's foliage —
  bistro's 38 live normal maps make it the visual showcase), curl-driven
  sway, per-frame TLAS rebuild, swept proxy boxes in the software BVH, and a
  DLL-free `self_test` for the pure math: cage build determinism, the
  clipping bounding property, the basis-change round-trip
  (M⁻¹ then A reproduces the paper's Fig. 5 identity), and motion bound vs
  sweep pad (the G10b shape).
- **Phase 2 — wavefront arm + gates.** `--check-gpu`/`--check-dxr` arms with
  the sway LIVE (the `--tod 2` pattern: run every existing gate with the
  feature exercising), the bounded-count allowance for animated-geometry
  rays, on-vs-off same-seed A/Bs, and the three-pose screenshot check —
  the gates prove soundness, never looks, and a sway bug of the
  wedding-cake class will be invisible to every counter.

## Kill criterion

Measured frame cost on both vendors (the interactive world session, the
`--gpu-timing` window medians) plus the visual smoke. If the sway is not
visibly worth its milliseconds — or the ghosting at MV-less amplitude reads
worse than stillness — the epic records the negative result and stops, per
the repo's measured culture. The paper's economics say the per-frame cost
should be near-nil at our scale; the risk is not the TLAS, it is the
BVH-quality tax of the swept proxy boxes and the upscaler ghosting.

## Known-accepts (v1)

- CPU arm renders the rest pose (world/interactive-only feature, loud line).
- A converging still freezes the sway mid-gust (the fireflies accept).
- No MVs on sway → bounded upscaler ghosting (follow-on: per-tet MVs).
- Bounded-count watertightness allowance on rays crossing animated geometry
  (static geometry keeps exact-zero).
- Foliage triangle count inflates 1.3–2.3× after clipping (foliage subset
  only; memory and BLAS build time follow).
- Hemi/GI gathers see the rest pose in v1 (bounce lighting is
  still-frame-only anyway, and the frozen-still rule already pins one pose).

## Open questions

- Cage resolution per island (the paper's medium-vs-high sweep says both
  look right; ours should start coarse — the sway is a flutter, not
  skinning).
- Whether the swept proxy boxes live as a new leaf kind in the one flat BVH
  or as a per-island sub-root — the flat-BVH assumption is load-bearing
  everywhere, so the leaf-kind route is strongly preferred.
- ε for the tet grow (paper: 2.5e-6, scene-dependent; ours should be
  eps-relative — the `Scene::diag` discipline).
- Whether Phase 1's transform ring wants the DXR pipeline only or both GPU
  arms from day one (DXR-only halves the Phase 1 shader surface; the
  wavefront consumes the same TLAS through RayQuery, so the port is
  mechanical).
