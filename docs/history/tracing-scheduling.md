# Ray and tree scheduling, sampling, resolution

Cut-seeded rays, the software-ray lever, the 8-wide frustum tree, wide level kernels, `--spp` multi-sampling, and `--lock-res`.

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
cargo run --release -- --no-cut-rays    # A/B lever: cut-SEEDED rays (primary leaf-tile rays)
                                        # traverse from the BVH root instead; the inherited
                                        # t_start is a scalar and survives. Isolates what the
                                        # CUT itself is worth to the ray path (~10% procedural,
                                        # ~2.5% San Miguel after the root-order fix)
cargo run --release -- --cut-hemi       # re-enable hemi leaf rays seeding from their bounce cut
                                        # (the pre-M2 behavior): 64 scattered cut roots measured
                                        # 3-10% SLOWER than one coherent root descent on every
                                        # scene/tree tried, so root-first is the DEFAULT; the
                                        # bound queries still consume the cut either way, and
                                        # --check's hemi probe gates force seeding ON so the
                                        # cut-miss gate keeps exercising the cut machinery
cargo run --release -- --gpu --continuation-rays  # A/B measurement lever (default OFF;
                                        # --sw-rays is the technical alias, --no-sw-rays the
                                        # kill): the WAVEFRONT tracer's rays traverse the
                                        # SOFTWARE BVH — bvh.rs's loops ported to
                                        # rt_sw.hlsli, pasted IN PLACE of rt.hlsli's RayQuery
                                        # bodies (same three primitive signatures; off arm =
                                        # the exact pre-lever source lists) — so leaf
                                        # PRIMARIES seed traversal from the tile's node cut.
                                        # IT IS FRAMED AS A SEMANTIC PROTOTYPE OF A HARDWARE
                                        # SEAM THAT DOES NOT EXIST: the terminal beam
                                        # publishes ONE opaque TraversalFrontier
                                        # (shaders/continuation.hlsli — a cookie-tagged
                                        # uint2, v1 packing slot<<6 | len-1; concatenated
                                        # ahead of queues.hlsli at the ONE QUEUES_HLSLI site,
                                        # so no unit can compile a different producer/consumer
                                        # contract) and every ray AND spp sample in that leaf
                                        # record reuses it through trace_closest_frontier
                                        # (= intersect_multi's semantics, v1 pool-order roots
                                        # + running-tmax prune, behind the opaque seam). The
                                        # leaf shader CANNOT read a node id, pool slot, or
                                        # length — a native provider could swap the two words
                                        # for driver-owned traversal state without touching
                                        # LeafRec or the call site. An invalid cookie, an
                                        # out-of-domain token, an exhausted arena, and an
                                        # explicit root ALL degrade conservatively to root
                                        # traversal — never an out-of-bounds arena read,
                                        # never a dropped candidate. t_start is deliberately
                                        # NOT in the token: the empty-space proof stays valid
                                        # when a frontier coarsens to an ancestor or the
                                        # root. The
                                        # REFERENCE kernel swaps too (one intersector both
                                        # sides — and the wavefront-vs-reference same-seed A/B
                                        # then reads EXACT 0.00e0 / hot 0 on NVIDIA *and* AMD:
                                        # the TMin-re-origin ulp class disappears because no
                                        # re-origining exists). LeafRec grew 16→24 B (the
                                        # frontier's two words, written always, read only
                                        # armed; trace::LEAF_REC_BYTES ↔ queues.hlsli ↔
                                        # main.rs readback in lockstep, and --check-gpu audits
                                        # every record's cookie + token domain CPU-side before
                                        # trusting the consumer); under FTREE the slot-ref cut
                                        # is translated to binary node ids at leaf EMISSION
                                        # via the lever-only ft_bnode map (QFNode still drops
                                        # bnode for everyone else) into a second pool slot
                                        # (cap_cut ×2, overflow stays gated 0; exhaustion =
                                        # root fallback, counted). Stacks are per-lane SCRATCH
                                        # (96 = bvh::TRAV_STACK, injected; groupshared at
                                        # 32×96×8 B would LDS-cap a zero-LDS kernel — the
                                        # documented sweep). Composes: --no-cut-rays = software
                                        # from the root (SW_RAYS_LEAF compiles out, the CPU's
                                        # short-circuit); --no-ftree = binary cuts, no
                                        # translation. Secondaries/hemi rays go software from
                                        # the root (a primary cut is apex-specific — inheriting
                                        # it would light-leak; hemi cuts still drive bound
                                        # queries only). CTR_FRONTIER_HANDLES / _RAYS /
                                        # _ENTRIES (per leaf RECORD, never per ray — the
                                        # per-ray atomic would tax the very path the lever
                                        # exists to measure) are the --check-gpu must-fires:
                                        # non-root handles > 0, rays > handles (reuse IS the
                                        # claim), 1 <= entries/handle <= 64. They count
                                        # frontiers CONSUMED, which is why frontier_record_
                                        # reuse zeroes its flag on !SW_RAYS_LEAF while still
                                        # executing all three atomics: the root control pays
                                        # identical telemetry cost AND reports zero BY
                                        # CONSTRUCTION, which is what lets the off-lever gate
                                        # demand exact 0. Do not key that flag on the token
                                        # alone — a MIXED split (one child <= LEAF_TILE while
                                        # a sibling is not, reachable at a parent extent of
                                        # exactly 2*LEAF_TILE+1) mints a real frontier in
                                        # EVERY arm, so the gate would become a property of
                                        # its own resolution's split ladder. alpha/relief/
                                        # tint counters ride candidate_reject unchanged. THE
                                        # VERDICT, measured (--spin path 1080p wall, 600f, rep-2
                                        # warm per the Arc compile trap): hardware RayQuery WINS
                                        # EVERYWHERE — 4090 spp=1: hw 0.87 / sw 1.13; B70 spp=1:
                                        # hw 1.76 / sw 2.54 / sw--no-cut-rays 2.57 (the cut seed
                                        # recovers ~1% of a 44% gap); B70 spp=16: hw 13.54 / sw
                                        # 26.35 (~2× even amortized; sw marginal ≈ 3.3 vs hw
                                        # 1.0-1.6 ms/sample). So even on the vendor whose RT
                                        # cores are weakest, driver traversal beats this
                                        # software walk ~2×, and cut-seeding cannot close it —
                                        # the empty-space proof stays the quadtree's whole GPU
                                        # value (the leaf.hlsl t_start-ablation conclusion,
                                        # now proven from the other side). 2026-08-01 ABBA
                                        # re-run (identical protocol, B70, 1600+600, fresh
                                        # process per run, root/frontier/frontier/root): root
                                        # vs frontier now IDENTICAL — ±0.004 ms per 120-frame
                                        # window across the whole lap — while both arms run
                                        # 7-11% faster than the 07-26 recording (wave-atomics
                                        # + leaf-kernel restructurings landed in between), so
                                        # the README's 6.5%-leaf/3.2%-frame frontier margin is
                                        # RETIRED; the frontier is proven LIVE the same day
                                        # (check-gpu: 768/768 non-root handles, 468.8
                                        # rays/handle, 0 root fallbacks) — the machinery works
                                        # and buys ~0 time on this workload. Known-accepts v1:
                                        # BLAS/TLAS still build (SPACE→DXR works; AS-skip +
                                        # dropping the RT-1.1 requirement — running on non-RT
                                        # GPUs — is the documented follow-on), require_caps
                                        # unchanged, no scene-cache key (GPU-only, the
                                        # blas-split class). THE CONTROL ARM is
                                        # --continuation-rays --no-cut-rays: same intersector,
                                        # shading, and inherited t_start, rays from the root.
                                        # It is NOT the same quadtree — SW_RAYS_LEAF also
                                        # gates the terminal-cut skip (see the wavefront queue
                                        # treatments below), so the control refines strictly
                                        # FEWER cuts (800x600 gate frame: 65 vs 449) and any
                                        # measured continuation win is a CONSERVATIVE bound.
                                        # Follow-ons: FR_SW_SORT
                                        # group-cooperative front-to-back root order,
                                        # groupshared-stack sweep, --cut-hemi re-measure on
                                        # GPU (HemiCellRec already carries cut_slot/cut_len),
                                        # hybrid sw-primary/hw-secondary. Gates: --check-gpu
                                        # [--sw-rays [--no-ftree|--no-cut-rays|--stress|
                                        # san-miguel-lp|--heightfield]] all PASS (exact-zero +
                                        # bit-identical A/B), --check-dxr untouched (dxr.rs
                                        # pastes neither queues nor frustum). Pre-existing,
                                        # NOT this lever: the AMD iGPU fails check-gpu's spp
                                        # readback with and without --sw-rays (environment)
cargo run --release -- --no-ftree       # A/B lever: hemi bound queries back on the binary BVH.
                                        # Default is the 8-wide frustum tree (src/ftree.rs) —
                                        # lazily collapsed from the ray BVH on the first hemi
                                        # query (only fb sessions pay its build/memory), returns
                                        # BIT-IDENTICAL bounds (self-test-pinned), measured
                                        # -15/-17% hemi-ao and -4/-8% hemi-gi ms/frame; cuts are
                                        # slot-refs, translated by Accel::ray_roots iff a ray
                                        # seeds from them (--cut-hemi)
cargo run --release -- --ftree-tiles    # A/B lever: the CPU tile recursion on the wide tree too
                                        # (tile_step/adopt_step; leaf tiles translate their cut
                                        # to binary ray roots once). Default OFF — unlike the GPU
                                        # tile kernels (-23%), CPU tiles measured wall-NEUTRAL on
                                        # San Miguel and ~10% slower on --stress no-temporal
                                        # (fat singleton-entry cuts, short descents — the
                                        # short-query regime again) despite -21..45% counted
                                        # frustum nodes; --check's `wide-tiles` gate verifies the
                                        # wired path every run so the lever can't rot
cargo run --release -- --no-wide-levels # A/B lever (GPU): every quadtree level runs one THREAD per
                                        # tile (the pre-cooperative ladder). Default ON = the shallow
                                        # levels (d < trace::WIDE_LEVELS) give one TILE a whole 32-lane
                                        # group sharing a breadth-first frontier (wavefront.hlsl::
                                        # bound_query_wave / cs_level_wide) — the ladder was under-
                                        # occupied (level 0 is one lane descending the whole BVH). A
                                        # BFS, so node counts differ, but `best` is an order-independent
                                        # min, so the same-seed image A/B comes back to the digit (a
                                        # pure perf A/B). Works on both frustum structures (binary +
                                        # ftree), unlike the old FTREE-only draft. See the Profiling
                                        # section for the measured -7..30% and the WIDE_LEVELS crossover
cargo run --release -- --spp 4         # multi-sampling: N primary samples per pixel per frame (1..128,
                                       # default 1; U doubles live), averaged into ONE splat
                                       # before the frame reaches the upscaler/denoiser. All three
                                       # render modes. Sample 0 is the frame's REPORTED sample (same
                                       # position rule, same rng seed, the only one that writes
                                       # tbuf/info/G-buffers/MVs — so spp=1 is bit-identical to a
                                       # single-sample frame and the upscaler's jitter contract stays
                                       # literally true); samples 1.. take dlss::jitter_for_sample
                                       # (the same Halton sequence at a phase-coprime stride, so the
                                       # reported 72-phase coverage is untouched) and contribute
                                       # color only. SOUND because every sample lands inside the same
                                       # pixel, hence inside the tile frustum: it consumes the SAME
                                       # inherited t_start/cut as sample 0 (the leaf-tile argument —
                                       # gated per sample, see --check). Pinned to 1 on fb (H) frames.
                                       # --defer-shade defers to the fused path at spp > 1 (a deferred
                                       # leaf stages ONE Traced per pixel — deferring a multi-sampled
                                       # tile would drop every sample but the first; coarser, never
                                       # wrong, and the two levers compose).
                                       # UNDER FSR RAY REGENERATION the presented color is
                                       # RECONSTRUCTED from the signal planes (dd⊗kd + ds⊗f0 +
                                       # residual), never from accum — so the residual must be the
                                       # exact remainder against the AVERAGE, not sample 0's color, or
                                       # --spp would be a costly no-op there. The GPU feed kernel gets
                                       # this for free (it subtracts from averaged accum); the CPU path
                                       # rewrites the sig after the average (render.rs::write_fsr — the
                                       # ONE fsr_buf write site, called again by shade_pixel at
                                       # spp > 1). Known accept, both feeds: the DENOISED lobes
                                       # (dd/ds) are the probe sample's, so the other N−1 samples'
                                       # direct light rides the un-denoised residual — --spp buys RR
                                       # less than it buys RR-less upscalers. Averaging dd/ds would
                                       # need the DXR pack write hoisted out of chs_shade (the
                                       # PrimSurf would have to ride the payload).
                                       # The 128 cap is NOT a math limit: it is the size of the
                                       # jitter table in FrameCb (MAX_SPP × 8 B, must fit CB_STRIDE) —
                                       # raise those two in lockstep (the HLSL cbuffer's row count is
                                       # INJECTED from MAX_SPP by trace::spp_defs, so it follows).
                                       # The extra samples' Halton index
                                       # runs FREE (not mod JITTER_PHASE, which bounds only the
                                       # sequence the UPSCALER sees): a wrap would alias sample 72
                                       # onto sample 0, so --check gates 128/128 distinct positions
                                       # and re-verifies the LAST sample at spp=128 (CPU and GPU).
                                       #
                                       # WHERE THE RETURNS STOP (measured; both benches print the fit).
                                       # Frame time is affine in the sample count: ms(n) = F + m·n,
                                       # F = the once-per-frame quadtree, m = one sample's rays+shading.
                                       # So amortization(n) = ms(n)/(n·ms(1)) = m/(F+m) + F/((F+m)·n):
                                       # an asymptote plus a 1/n term — HALF the fixed cost is diluted
                                       # away by spp 2, 90% by spp 10, 99% by spp 100. The amortization
                                       # is therefore spent by ~8-16 spp; past that every sample pays
                                       # the full marginal price m, while QUALITY improves only as
                                       # 1/√n. spp 128 is honest supersampling, not a free lunch.
                                       #   HISTORICAL GPU wavefront table (1080p, interleaved medians).
                                       #     It predates the shipping (LEAF_TILE, LEAF_GROUP)=(32,256)
                                       #     frontier and the gputime async-compile-bias fix. Retained as
                                       #     experiment provenance only; do NOT cite its ratios/crossovers
                                       #     as current. The t_start ablation below remains the durable result.
                                       #     These rows were post the earlier wave64 leaf-lane repair:
                                       #     4070 Ti: hybrid = 1.32 ms fixed + 0.464 ms/sample (floor 0.26×)
                                       #              plain  = 0.15 ms fixed + 0.420 ms/sample (floor 0.74×)
                                       #     R9700:   hybrid = 1.50 ms fixed + 0.690 ms/sample (floor 0.32×)
                                       #              plain  = 0.15 ms fixed + 0.544 ms/sample (floor 0.78×)
                                       #     hybrid/plain floor 1.11× (NV) / 1.27× (AMD) — on those two
                                       #     vendors the quadtree does not win primary visibility (its
                                       #     marginal sample stays dearer than an RT-core root traversal,
                                       #     and only the marginal cost survives at high spp) — but the
                                       #     margin is small, where it used to read 1.33× on a 4090 and
                                       #     2.56× on AMD. Most of that gap was the wave64 lane waste,
                                       #     not the algorithm.
                                       #   ON INTEL IT WINS, AND THAT IS THE FIRST GPU WHERE IT DOES.
                                       #     Arc Pro B70, post cs_sky + cs_level_wide:
                                       #       default: hybrid 1.28 fixed + 1.046 ms/sample
                                       #                plain  0.18 fixed + 1.134 ms/sample -> 0.92×
                                       #       stress : hybrid 1.51 fixed + 0.732 ms/sample
                                       #                plain  0.21 fixed + 0.842 ms/sample -> 0.87×
                                       #       powerplant 12.8M tris:
                                       #                hybrid 1.33 fixed + 0.869 ms/sample
                                       #                plain  0.19 fixed + 0.934 ms/sample -> 0.93×
                                       #     i.e. the quadtree makes each SAMPLE 7-13% cheaper than an
                                       #     RT-core root traversal there, and at spp=16 the hybrid beats
                                       #     the control outright (18.01 vs 18.32 default, 13.22 vs 13.69
                                       #     stress). Same asymptote is 1.37×/1.31×/1.36× on a 4090 — so
                                       #     this is a property of the HARDWARE BALANCE, not of the scene:
                                       #     the quadtree trades RT-core work for shader-core work, and
                                       #     Intel's RT is weak relative to its shader cores (plain
                                       #     reference 1.31 ms vs the 4090's 0.36 — 3.6×, far wider than
                                       #     the gap in shading-bound work), so there is more traversal to
                                       #     save and it is worth more.
                                       #     THREE THINGS THAT ARE **NOT** TRUE, EACH MEASURED:
                                       #     (1) It does not scale with scene complexity. The ratio is
                                       #         FLAT (0.87-0.93 Intel, 1.31-1.37 NV) across 80k -> 12.8M
                                       #         tris, a 160× range. What little variation there is tracks
                                       #         SPARSITY, not size — --stress 5000's 5000 separate objects
                                       #         (0.87) beat one dense powerplant mesh (0.93), which is
                                       #         what you would expect of a structure whose product is
                                       #         proving space empty.
                                       #     (2) It is not the INHERITED DISTANCE BOUND doing the work.
                                       #         Ablating t_start to 0 while keeping the quadtree costs
                                       #         only +1.7/+1.1/+1.7% on the B70 (and straddles zero on a
                                       #         4090: -3.5/-7.1/+5.2%, i.e. free and worth nothing, the
                                       #         verdict AMD already had). That is 7-21% of the advantage;
                                       #         the other ~80-93% is TILES PROVEN EMPTY TRACING NO RAYS —
                                       #         a conservative screen-space occupancy mask would buy most
                                       #         of it, with no quadtree at all. See leaf.hlsl.
                                       #     (3) It does not help the config that ships. At spp=1, the
                                       #         interactive default, the quadtree still LOSES on Intel
                                       #         (1.77× default, 2.12× stress, 1.96× powerplant); the win
                                       #         needs the once-per-frame quadtree cost amortized and
                                       #         crosses over at ~spp 16.
                                       #     Both halves moved when cs_sky was fixed: the B70 asymptote was
                                       #     1.29× before it, because the sky fill's per-sample cloud
                                       #     marching (--spp averages sample positions in cs_sky) was
                                       #     inflating the MARGINAL cost, not just the fixed one.
                                       #   CPU (default scene): hybrid = 0.9 ms fixed + 9.6 ms/sample,
                                       #     plain = ~1 ms + 10.6 — the mirror image: almost no fixed
                                       #     cost to amortize (floor 0.91×), but the quadtree makes
                                       #     each SAMPLE ~10% cheaper, and that discount does NOT decay
                                       #     with spp. (--spin path SM low-poly: 17.1 → 61.5 ms at 4 spp.)
                                       # Noise: 1.9-2.0× quieter at 4 spp (--check's stability gate).
                                       # --dxr traces from the TLAS root: no claim to inherit, so
                                       # there it is plain supersampling (quality only)
cargo run --release -- --lock-res dynamic  # step-wise dynamic render resolution (DLSS-RR and XeSS)
cargo run --release -- --lock-res 0.75     # lock the render res to a fixed scale of the window; the
                                           # default is `native` (100%, xess::DEFAULT_LOCK_SCALE —
                                           # DLAA-shaped: the wired upscaler runs as a pure
                                           # antialiaser/denoiser at window res; `--lock-res quality`
                                           # = the 2/3 arm) for
                                           # EVERY render mode — CPU, --gpu and --dxr alike, so F/SPACE
                                           # cycling arms never moves the render res. HISTORY, four
                                           # moves: until 2026-07-26 the GPU arms defaulted to native
                                           # through a second `Opts::gpu_lock_scale` (that field and
                                           # the split are gone); `quality` (2/3) until 2026-07-31;
                                           # `native` (100%, DLAA-shaped) until 2026-08-08; quality
                                           # again for part of that day; native again since (the
                                           # user's call). So numbers recorded at "the flagless
                                           # default" carry their era's scale — 0.444x the PIXELS in
                                           # the quality windows (2/3 is a
                                           # LINEAR scale, 1920x1080 -> 1280x720, so pixel-proportional
                                           # costs scale by 0.444 and per-tile ones (the level ladder)
                                           # barely move at all), full-res in the native ones.
                                           # `--lock-res quality` spells the
                                           # 2/3 arm explicitly (the preset vocabulary is decoupled
                                           # from the default constant — see xess::lock_scale). Headless
                                           # `--spin` stays at native unless --lock-res
                                           # is passed — benchmarks must not have defaults move under
                                           # them, the vendor-default rule (the two currently
                                           # coincide, but the rule predates and outlives that).
                                           # The 2/3-era accepts are CLOSED again: G/X/K toggling the
                                           # upscaler OFF presents at window res (no DWM-stretched
                                           # blit unless an explicit sub-native --lock-res asks for
                                           # one), and vendor_defaults' Intel entry IS measured at
                                           # the res a flagless session traces (its res-basis caveat
                                           # is re-closed — the RES-BASIS DEBT paragraph there).
                                           # Also takes quality (2/3)
                                           # |balanced|performance|ultra-performance or a ratio in (0, 1]
```
