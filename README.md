# frustracer

A **frustum-tracer**: a hybrid between beam tracing and ray tracing.

The screen is a **quadtree**. One frustum covers the whole screen; it is traced
forward until it *could* hit geometry, that conservative distance is recorded,
and the tile splits into 4 quadrants. Each child frustum inherits the parent's
proven-empty distance and starts there — empty space is skipped once per tile
instead of once per pixel. Recursion bottoms out at 8×8 tiles, which trace
ordinary per-pixel rays with `tmin` set to the inherited distance. A tile whose
frustum intersects nothing is filled with sky immediately, **zero rays traced**.

Children also inherit a **node cut**: the parent's list of surviving BVH nodes,
re-culled and refined level by level (`frustum.rs::refine_cut`), so a tile's
frustum query descends the parent's frontier instead of the BVH root, and leaf
pixel rays seed their traversal from the cut (`Bvh::intersect_multi`).

Related prior art: cone marching, and the beam optimization in Laine & Karras'
sparse-voxel-octree paper.

## Why a BVH for the scene?

The frustum query needs two things from the spatial structure: a cheap
"is this region outside the frustum?" test, and a cheap conservative
"nearest possible distance" bound. A **BVH of AABBs** gives both — the
classic positive-vertex plane test for culling, and point-to-box distance
for the bound. BSP/KD-trees are worse here: their splitting planes don't
yield a cheap conservative nearest-hit distance along the beam.

## The correctness crux

The distance recorded by a parent is the Euclidean distance from the camera
origin, so the region it proves empty is *frustum ∩ ball(origin, t)* — a
**spherical** bound, not a planar near clip. The query
(`src/frustum.rs::nearest_geometry_distance`) therefore culls a node as
"already proven empty" only when its whole box lies inside that ball, and
clamps candidates up to the inherited distance. Mixing in a flat near plane
would over-cull near frustum corners and let child rays start past real
geometry. Secondary rays (shadow / AO / reflection) never see the inherited
`tmin` — it is a primary-frustum property only.

The node cut obeys the same discipline: `refine_cut` may drop a node only if
it is fully outside the tile frustum or fully inside the proven-empty ball —
**never** by distance-to-best pruning (a far node can still be the nearest
thing in a sibling's frustum; pruning it would surface as false sky). When the
`MAX_CUT` budget runs out, internal nodes are emitted coarsely, never dropped.

`frustracer --check` verifies this directly: it renders one hybrid frame,
re-traces every pixel with a `tmin = 0` reference ray, and requires the
**false-sky** and **tmin-overshoot** counters to be exactly zero — at full
depth and again through the depth-capped driver (coarse flat-filled pixels are
excluded from the comparison but must exist, so the capped path is provably
exercised).

## Hemisphere bounces: the same idea, aimed at the light integral

Secondary lighting is an integral of incoming light over the hemisphere above
each shading point — and the same divide-and-conquer that drives the screen
quadtree can dispatch that search (**H** cycles it: off → AO → GI →
GI + shadow shafts; still frames only).

The hemisphere is a quadtree too, but built from **spherical triangles**
instead of squares: the root is the tangent half-space (a 1-plane frustum),
level 1 is 4 spherical octants, and deeper levels split each triangle through
its great-circle edge midpoints. Great-circle edges are exactly planes through
the apex, so every cell *is* a `TileFrustum` (3 planes + unused slots), the
midpoint children exactly partition their parent (which is what makes
inherited `tmin` + node cuts sound, the same argument as pixel quadrants), and
each cell has a closed-form cosine-weighted area (Lambert's formula — an
octant is exactly π/4). The apex is the shading point; the hemisphere runs its
own tmin chain from its own apex — the primary tile's `tmin` never leaks in.

Each cell runs the familiar step: bound query over the inherited cut, then

- **proven empty** → the cell's contribution is *analytic*: exact projected
  solid angle for AO, `sky() ×` exact PSA for GI (with pure-math refinement
  near the sun's glow lobe — an empty parent proves all children empty, so
  refining costs sky evaluations only). Zero rays, zero variance.
- **query cutoff reached** → one stratified ray per sub-cell (uniform inside
  the spherical triangle, Arvo '95), seeded from the inherited cut with the
  inherited `tmin` — the hemisphere analog of `LEAF_TILE`: one bound query
  amortizes over 4 rays, because an occlusion ray is ~10 node visits and a
  bound query on a dense cut costs more.
- otherwise → refine the cut, split 4-way, recurse (blocked cells subdivide,
  never stop — same rule as the screen).

AO additionally clamps every query to the AO radius (`None` then means "open
within the radius", never "sky") and drops cut nodes entirely beyond it —
sound only because every consumer ray's `tmax` is clamped the same way.

Verification mirrors the primary gates: every claimed-empty cell is re-tested
with reference rays through its interior (**false-empty = 0**), every leaf
ray is re-traced with `tmin = 0` (**tmin-overshoot = 0**), cut-seeded
traversal must agree with the full tree, and the accounted projected solid
angle per point must total π. On top of that, `--check` A/Bs the integrator
against high-sample cosine references: AO within 0.02 mean absolute error and
bias-free (the estimator is unbiased — the leaf samples are uniform in their
cells); GI within 5% mean relative against a reference implementing the same
one-bounce policy.

The measured economics are honest rather than triumphant: a hemi-GI frame
costs ~40-60× a plain hybrid frame at 64 cells/point on the dense default
scene — but it is *converged* immediately where the sampled path needs
hundreds of accumulation frames, and open scenes adapt (most of the
hemisphere resolves analytically at octant scale).

**Shadow shafts** apply the same machinery to the area light: a frustum from
the shading point through the light's corners (clipped by the tangent plane —
without that 5th plane the own surface's AABB hugs the apex and nothing is
ever proven lit), subdivided once on ambiguity. Samples landing in a subrect
proven empty skip their occlusion ray outright — same sampling, same
estimator, identical image; 75% of shadow rays vanish on the default scene.
The candid result: at 2–4 shadow samples/point the culling query costs more
than the rays it saves (~3× net slower), so shafts are off by default — the
technique needs cross-point claim sharing (the temporal cache's δ-subtraction
transfer, future work) before it pays.

## Parallelism

Rayon's work-stealing pool is the "task list + pool of listener threads":
each quadrant split is a recursive `rayon::join` (tiles ≤ 32 px recurse
sequentially for granularity). The framebuffer is a `Vec<AtomicU32>` of
f32-bit linear RGB with relaxed stores — tile writes are disjoint, so this is
the idiomatic safe shared buffer.

## Running

```
cargo run --release                   # procedural scene (boxes, spheres, marble bunny, gold teapot);
                                      # the DXR DispatchRays pipeline + the first supported upscaler
cargo run --release -- model.obj      # load an OBJ (auto-fitted onto the ground)
cargo run --release -- --stress 5000  # perf test: field of n objects (boxes/spheres/bunnies/teapots)
cargo run --release -- --check        # headless: verify vs reference, benchmark, write check.png
cargo run --release -- --check-dlss   # headless: DLSS G-buffer MV/depth/matrix self-test
cargo run --release -- --cpu          # the CPU frustum-tracer as the render mode (opts out of --dxr/--gpu)
cargo run --release -- --no-dlss      # skip the DLSS-RR level of the always-on upscaler chain
                                      # (DLSS-RR -> FSR4-RR -> XeSS -> FSR3; the first supported
                                      # level wins, and --<x> force-starts the chain at level x)
cargo run --release -- --no-upscale   # plain presentation: no temporal upscaler at all
cargo run --release -- --nppd         # NPPD neural denoising before an XeSS upscale (needs
                                      # onnxruntime.dll + an exported model — see
                                      # tools/nppd-export/README.md; --no-xess = standalone)
cargo run --release -- --gpu --nppd   # the same composition GPU-resident: ONNX Runtime executes
                                      # on the tracer's own D3D12 queue, zero per-frame CPU traffic
```

Debug builds are ~10× too slow to judge anything — always use `--release`.

### Build times, and the `quick` profile

`.cargo/config.toml` links with **`rust-lld`** (LLVM's LLD in its `link.exe`-compatible
COFF mode). It ships inside the rustup toolchain, so there is nothing to install
and no new prerequisite. Measured on a one-line touch of `main.rs`: ~26 s → ~21 s
per link, and the variance collapses (±0.2 s vs ±2.8 s).

(**mold is not an option on Windows** — it is an ELF linker and cannot produce a PE
binary. Windows support is an aspirational goal for a hypothetical mold 3.0. The
question keeps coming up; `rust-lld` is the answer.)

The linker is not the bottleneck, though — the `release` profile's `lto = "thin"` +
`codegen-units = 1` whole-program pass is. The same one-line touch costs **~123 s**
under `release` and **~25 s** under `quick`:

```
cargo build --profile quick
cargo run --profile quick -- --check
```

> **Never benchmark under `quick`.** Every performance number this project reports —
> `--check`'s A/B bench rows, the hemi-share kill criterion, the adopt on/off
> regression guards — is only meaningful under `release`'s `lto`/`codegen-units`
> settings. Use `quick` to find compile errors and to exercise the exact-zero
> correctness gates (which are perf-independent); use `--release` for anything that
> prints a number.

One local step this repo cannot make for you: **exclude `target/` from Windows
Defender**. Every link writes an 18 MB exe and a 33 MB PDB past the real-time
scanner, and the cost is invisible in `cargo --timings` — it just looks like slow
linking.

```powershell
Add-MpPreference -ExclusionPath '<repo>\target'   # elevated; narrows AV coverage on build output
```

### Runtime SDKs (optional features)

Building never needs any SDK: the MIT headers the shims compile against are
vendored, and every SDK below is `LoadLibrary`'d at runtime, so a bare checkout
builds and passes every DLL-free `--check*` gate. The interactive features want
runtime DLLs, which are license-restricted and therefore not committed —
`install-prerequisites.bat` downloads them from each vendor's own release page
into the directories the defaults already point at:

```
install-prerequisites.bat              # everything below (~700 MB)
install-prerequisites.bat dxc xess     # just those; /force re-installs, /clean drops the cache
```

| Component | Enables | Lands in |
|---|---|---|
| `dxc` | `--dxr` (the **default** render mode) and `--gpu` | `SDKs\dxc\bin\x64` |
| `dlss` | DLSS-RR (`G`) | `SDKs\streamline-sdk\bin\x64` |
| `fsr` | FSR4-RR / FSR 3.1 (`K`) | `SDKs\FidelityFX-Samples-prebuilt\...` |
| `xess` | XeSS-SR (`X`) | `SDKs\XeSS-SDK\bin` |
| `nppd` | NPPD neural denoising (`J`) | `SDKs\onnxruntime\bin` |
| `oidn` | OIDN denoising (`N`) | `SDKs\oidn.x64.windows\bin` |
| `pix` | `--pix-markers` | `SDKs\pix\bin\x64` |

Each is also overridable with the matching `--*-path` flag / `FRUSTRACER_*_PATH`
env var. The one thing the script cannot fetch is the NPPD **model weights**
(the upstream checkpoint carries no license grant) — export those yourself with
`tools/nppd-export/export.py --fp16`; it prints the command.

### DLSS Ray Reconstruction setup (optional)

The renderer can hand its 1-spp frames to NVIDIA's DLSS Ray Reconstruction
denoiser (RTX GPU required). Only the MIT-licensed Streamline headers and
docs are vendored in `SDKs/streamline-sdk` — the runtime DLLs are
license-restricted and are **not** in this repo. To enable DLSS, download
the Streamline SDK 2.12.0 release zip from
<https://github.com/NVIDIA-RTX/Streamline/releases> and extract it over
`SDKs/streamline-sdk` so that `SDKs/streamline-sdk/bin/x64/sl.interposer.dll`
exists (or point `--sl-path` / `FRUSTRACER_SL_PATH` at any directory holding
the DLLs). Without the DLLs the app logs a note and runs with native D3D12
presentation; building never needs them.

### NPPD neural denoising setup (optional)

The renderer can also run NPPD (*Neural Partitioning Pyramids for Denoising
Monte Carlo Renderings*, Bálint et al., SIGGRAPH 2023) as a vendor-neutral
neural denoiser through ONNX Runtime + DirectML (any D3D12 GPU — NVIDIA, AMD,
or Intel; CPU fallback). Nothing ships in this repo: drop `onnxruntime.dll`
from the [Microsoft.ML.OnnxRuntime.DirectML NuGet](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML)
(≥ 1.22) and `DirectML.dll` from [Microsoft.AI.DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML)
(≥ 1.15) into `SDKs/onnxruntime/bin`, then export the pretrained model with
`tools/nppd-export/export.py --fp16` (see its README — the upstream weights
carry no explicit license, so neither the checkpoint nor the exported `.onnx`
may be committed). Run with `--nppd` or toggle with **J**. By default `--nppd`
composes with XeSS: the frame is traced at 2/3 resolution, NPPD denoises at
that resolution, and XeSS upscales to the window; under `--gpu` the whole
stage is GPU-resident (ONNX Runtime executes on the tracer's queue with the
staging buffers bound directly as tensors — no per-frame CPU traffic).

### Controls

| Input | Action |
|---|---|
| WASD / QE / Space | fly (Shift = fast) |
| hold left mouse | look around |
| **R** | toggle hybrid frustum-tracer vs plain per-pixel (A/B benchmark) |
| **T** | toggle dynamic resolution vs fixed half-res while moving |
| **O** | quadtree debug overlay: subdivision-depth heatmap + tile borders |
| **G** | toggle DLSS Ray Reconstruction (when available) |
| **J** | toggle NPPD neural denoising (in XeSS mode: the pre-upscale slot) |
| **H** | hemisphere frustum bounces: off → AO → GI → GI + shadow shafts (still frames) |
| **B** | toggle GPU vs CPU tonemap (non-DLSS mode) |
| **1 / 2 / 3** | quality presets (shadow/AO samples, reflections) |
| **C** | verify current view against the reference (prints counters) |
| **P** | screenshot (in DLSS mode: reads the denoised output back from the GPU) |
| Esc | quit |

## Dynamic resolution (60 FPS target)

While the camera moves, each frame targets a time budget (~15 ms render +
resolve/present headroom ≈ 60 FPS). The budget is not a per-tile deadline:
a controller converts the *previous* frame's measured time into a **uniform
quadtree depth cap** for the next frame, and the same depth-first recursion as
the normal driver runs everywhere to that cap. Tiles reaching it unresolved
become one flat quad — the color of a single representative ray through their
center, starting at the tile's inherited distance. A uniform cap means no
screen corner gets refined at another's expense (the failure mode that a
wall-clock deadline on a depth-first traversal would have), so the recursion
keeps its cache-friendly shape: node cuts live on the recursion stack, hot in
cache, instead of a materialized breadth-first frontier of 256-byte tiles.
Zero clock reads in the driver — a frame is deterministic for a given cap.

The controller works in log space: cost roughly quadruples per level, so
`log4(budget / elapsed)` reads "levels of headroom" directly. A fractional
depth accumulator moves by `0.6 ×` that error, clamped to creep up slowly
(≤ 0.4/frame) but drop more than a full level after a blown frame, within
`[2, depth_full]`; a deadband stops it climbing when a frame already uses
> 60 % of the budget (the next level costs ~4×). At the top of the range the
cap reaches leaf tiles and a "budget" frame is simply a full hybrid frame.
The trade-off of having no in-frame deadline: a hard cut to much denser
geometry can blow one frame past budget before the controller reacts on the
next. Resolution floats with scene and shading cost instead of frame time:
crank the quality preset and tiles get coarser, not slower. The overlay tints
these depth-capped quads orange — uniform quad size is the visible signature
of the cap. When the camera stops, jittered samples accumulate (up to 1024
spp) so soft shadows, AO, and antialiasing converge at full resolution. **T**
falls back to the old fixed half-res moving mode for comparison.

Per-second stats on stderr show the frustum counters: queries, node visits
(frustum + ray), mean node-cut length, sky pixels resolved with zero rays,
coarse pixels, and mean `t_start / t_hit` (how much empty space the quadtree
skipped before each pixel ray even started).

## Deferred material-sorted shading (--defer-shade): a candid negative result

The idea: the quadtree already visits the screen depth-first, so let leaf
tiles trace their pixels but *defer* shading, merge same-material runs on the
way back up (whole-segment pointer moves, capped at a 64×64 maximum shading
tile for load balance), and flush each macro-tile with its segments
stable-sorted by material — a "tiled shader" where one material's textures
stay cache-hot for a whole burst instead of being evicted at every 8×8
boundary. The plumbing is exact: each record carries its pixel's RNG state,
`dir`/`ray` are recomputed bit-identically from the frame camera, and the
flush replays `shade()` verbatim — `--check` on any textured scene gates
defer-off vs defer-on **bit-identity** of color/t/info (0 px differ), so the
reorder provably changes nothing but timing.

Measured (7950X3D, 1080p `--spin path`, 200 frames): default procedural scene
22.7 → 22.7 ms (untextured materials shade inline — the machinery
structurally disengages), Vokselia 22.5 → 23.0, San Miguel interior 7.74 →
7.75 with 12% of pixels deferring in mean-400 px bursts, Intel Sponza (2.7 GB
of 4K textures, 23% of pixels deferring) 36.9 → **41.2 ms** — the more
texture traffic, the more the staging costs, and nowhere a win. Two earlier
implementation lessons are baked into the code: merging by `Vec::append`
re-copied every record per level (~GB/frame, 3× frame time) — segments fixed
that; and flushing a 4096-px bucket sequentially made every macro-tile an
end-of-frame straggler on one core (+29% on Sponza) — the flush now re-splits
across rayon at the fused path's 1024-px grain.

Why it can't win: at 1 spp, adjacent pixels' bilinear footprints are
essentially disjoint, so each texture cache line is touched about once per
frame *regardless of shading order* — material sorting optimizes the order of
a stream that has no reuse to exploit (and a 128 MB V-Cache absorbs much of
what little there is). Mips were the obvious suspect for creating that reuse,
so they were built next (below) and the A/B was rerun: **still no win** — San
Miguel interior 7.49 → 8.89 ms, Intel Sponza 32.6 → 38.8. Mips shrink the
*working set*, which helps the fused path just as much; they don't manufacture
inter-pixel reuse that a reorder could newly exploit. The feature stays
off-by-default behind `--defer-shade`, gates and `defer:` counters intact.

## Mip-mapping, trilinear, and 16× anisotropic filtering

Textures are sampled trilinear with a **ray-cone LOD** (Möller 2019,
curvature-free), on all three renderers — CPU, the `--gpu` wavefront, and
`--dxr`. Cone width grows along the ray (`w0 + t·spread`, primary spread = one
pixel's angular size); the LOD adds the triangle's texel density
(`0.5·log2(uv_area/world_area)`, computed on the fly from the hit's vertices —
a cached per-tri array would cost ~400 MB at 100M-tri tiling scale), the map's
dimensions, and a grazing-angle term. Reflection and glass continuations
inherit the parent hit's cone width; hemisphere-GI bounce hits read a fixed
broad footprint (over-blurred bounce albedo is variance reduction, never
error).

The chain is generated once on the CPU (2×2 box filter to 1×1) and uploaded to
the GPU verbatim, so the renderers' long-standing parity axiom is *upgraded*
rather than broken: identical texels at identical LODs, and the GPU-vs-CPU
albedo gate (mean |Δ| ≤ 0.02/channel) now compares trilinear against trilinear
— measuring 0.0000/channel on San Miguel, with no tolerance widened. Two
details are load-bearing. The filter runs in **linear space** (sRGB texels
decode, average, re-encode) — a gamma-space box filter darkens mid-tones, and
the self-test's 2×2-checker case rejects it. And **alpha cutout never sees
mips**: it stays nearest-texel at level 0 on every path, because that test is
*visibility*, and CPU/RayQuery/DXR agreeing on it bit-for-bit is a correctness
contract.

`lod ≤ 0` reproduces the old bilinear sampler bit-exactly, so magnified views
— and every existing tolerance gate — are unmoved by the feature. `--no-mips`
is the A/B lever. Measured (1080p `--spin path`, 7950X3D): mips are *faster*,
Intel Sponza 34.7 → 32.6 ms and San Miguel interior 7.7 → 7.5 (fewer
DRAM-miss taps under minification more than pays for the extra tap), at +33%
texture memory. Aliasing goes with them: `FRUSTRACER_STAB=1` on a still XeSS
Sponza view reads 0.65 → 0.57 /255 of inter-frame shimmer.

### Anisotropy (`--aniso N`, default 16)

A ray cone is a *circle*, but the surface it lands on sees an *ellipse*:
projected along the ray, the footprint is `cone_w` across the direction of
travel and `cone_w / |n·d|` along it. One scalar LOD can only describe a
circle, so the formula above covers the **major** axis and blurs the minor one
with it — its `− log2(max(|n·d|, 0.05))` term *is* that compromise, and it is
why trilinear mushes distant floors and long walls. Anisotropy is therefore not
a rival filter here but the same footprint kept honest: `tri_grads` returns both
axes as normalized-UV gradients (Cramer's rule against the triangle's UV basis
— the on-the-fly `∂P/∂u, ∂P/∂v` the normal-mapping tangent frame already
derived, now shared), the CPU averages up to 16 trilinear taps along the major
axis at the *minor* axis's LOD, and the GPU hands the identical gradients to
hardware `SampleGrad` on an anisotropic sampler. That the two paths are one
formula is gated, not asserted: on a conformal UV map the major axis in texels
must equal the old isotropic LOD to 1e-4.

Worth knowing if you go looking for this in the code: **`SampleLevel` cannot be
anisotropic.** It hands the TMU one scalar LOD and no gradients, so switching
the sampler to `ANISOTROPIC` while still calling `SampleLevel` would have been
a silent no-op — the gradients are the whole feature. Hemisphere-GI bounce rays
deliberately stay isotropic (their cone is octant-coarse by design; 16 taps
would buy nothing).

`--aniso 1` (= `--no-aniso`) runs the isotropic path *verbatim*, so it is
bit-identical to the pre-anisotropy renderer by construction — which makes the
whole check suite re-run under it the regression proof. Hardware aniso and the
CPU's N-tap approximation use different tap distributions, so they were never
going to agree exactly; the GPU-vs-CPU albedo gate measures **0.0001/channel
against its unchanged 0.02 limit**. Cost is pose-dependent and real on the CPU
(16 taps where the footprint is 16:1): Intel Sponza 24.1 → 26.3 ms (+8.7%), San
Miguel interior 16.0 → 16.1 (+1.1%); `--aniso 4|8` buys that back. On the GPU
it disappears under the bench row's own noise.

## Future work

Cut-aware leaf ordering (sort the cut by distance once per leaf tile so all 64
rays shrink `tmax` early), and adapting the frame budget from measured
resolve/present cost. A GPU compute BC7 encoder (the ispc encode is ~20 s on
Intel Sponza and runs every load — there is no disk cache).
