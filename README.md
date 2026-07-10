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
cargo run --release                   # procedural scene (boxes, spheres, marble bunny, gold teapot)
cargo run --release -- model.obj      # load an OBJ (auto-fitted onto the ground)
cargo run --release -- --stress 5000  # perf test: field of n objects (boxes/spheres/bunnies/teapots)
cargo run --release -- --check        # headless: verify vs reference, benchmark, write check.png
cargo run --release -- --check-dlss   # headless: DLSS G-buffer MV/depth/matrix self-test
cargo run --release -- --no-dlss      # skip Streamline; native D3D12 presentation
```

Debug builds are ~10× too slow to judge anything — always use `--release`.

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

### Controls

| Input | Action |
|---|---|
| WASD / QE / Space | fly (Shift = fast) |
| hold left mouse | look around |
| **R** | toggle hybrid frustum-tracer vs plain per-pixel (A/B benchmark) |
| **T** | toggle dynamic resolution vs fixed half-res while moving |
| **O** | quadtree debug overlay: subdivision-depth heatmap + tile borders |
| **G** | toggle DLSS Ray Reconstruction (when available) |
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

## Future work

Cut-aware leaf ordering (sort the cut by distance once per leaf tile so all 64
rays shrink `tmax` early), and adapting the frame budget from measured
resolve/present cost.
