# The GPU wavefront tracer and the DXR pipeline

`--gpu`, `--check-gpu`, `--dxr`, the `--dxr-inline` ablation ladder, and the `--dxr-sbt` material-sorted SBT experiment.

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
cargo run --release -- --gpu          # GPU-resident tracing: the whole quadtree + shading in D3D12
                                      # compute with DXR RayQuery rays (needs the DXC DLLs + RT tier 1.1;
                                      # falls back to the CPU renderer with the reason on stderr).
                                      # Composes with the chain's wired level (GPU-born G-buffers,
                                      # zero CPU readback — RR/XeSS/FSR4-RR/FSR3 all GPU-fed);
                                      # --no-upscale = plain. Wins over the --dxr default
cargo run --release -- --gpu --xess   # GPU tracer -> XeSS-SR composition (implies --no-dlss); the
                                      # render res is LOCKED per session (--lock-res, default native
                                      # 100% — `--lock-res dynamic` is not honorable under --gpu, it
                                      # locks at that same default with a loud line)
cargo run --release -- --gpu --nppd   # GPU tracer -> GPU-RESIDENT NPPD -> XeSS: ONNX Runtime executes
                                      # on the tracer's own queue (DML1) with the staging buffers bound
                                      # as tensors — zero per-frame CPU traffic (pack/warp/crop are
                                      # nppd.hlsl kernels). J toggles; XeSS-only (--no-xess forces it off)
cargo run --release -- --check-gpu    # GPU tracer gate suite + bench (needs a real GPU + the DXC DLLs;
                                      # composes with --stress; exit 2 = environment, 1 = a gate failed)
cargo run --release -- --dxr          # the by-the-book DXR pipeline (RTPSO + SBT + DispatchRays with
                                      # raygen/closest-hit/miss shaders) — the DEFAULT render mode
                                      # on NVIDIA/AMD (--cpu opts out). ON AN INTEL ADAPTER the
                                      # flagless default is the WAVEFRONT tracer instead
                                      # (main::vendor_defaults — measured 2.6-5.1x, see the vendor-
                                      # aware defaults paragraph; DXR stays armed as the automatic
                                      # fallback if the wavefront init fails), so --dxr is no
                                      # longer a pure no-op: it PINS the DXR start against that
                                      # policy (mode_explicit). F toggles CPU <->
                                      # DXR live. COMPOSES with the chain's wired upscaler: DXR-fed
                                      # DLSS-RR / FSR4-RR / XeSS / FSR3 — tracing at the LOCKED
                                      # --lock-res scale (default native 100% — the ONE
                                      # session default, every mode; window-res when plain
                                      # either way). Needs the
                                      # DXC DLLs + RT tier 1.0; falls back to the
                                      # CPU renderer with a loud line (the chain's upscaler stays
                                      # wired). SPACE cycles all three render modes live (see the
                                      # interactive-keys paragraph); the CLI flags pick the mode a
                                      # session STARTS in. This pipeline's rays ride --dxr-inline
                                      # (below): DEFAULT 1 = inline RayQuery secondaries — 2 on an
                                      # INTEL adapter (main::vendor_defaults, 2026-08-01)
cargo run --release -- --dxr-inline 0 # A/B lever: the DXR pipeline back on ALL-TraceRay dispatch —
                                      # the pre-W2 by-the-book build, bit-identical library. The
                                      # CROSS-VENDOR DEFAULT is 1: primary TraceRay -> chs_shade,
                                      # every secondary an inline RayQuery inside the hit shader
                                      # (MaxTraceRecursionDepth 1) — promoted because it strictly
                                      # DOMINATES 0 at every measured point on both vendors
                                      # (spp=1 tracer: B70 9.05 -> 2.35 ms, 4090 1.34 -> 0.26;
                                      # never slower at any spp). 2 = everything inline in raygen
                                      # (DispatchRays as a bare launch grid) — the measurement arm
                                      # that proved launch overhead ~ 0, and THE INTEL DEFAULT
                                      # since 2026-08-01 (main::vendor_defaults: mode 2 beats 1 on
                                      # the B70 at every measured point — 1.41/1.22/1.29 vs
                                      # 2.35/1.64/1.94 spp=1, world span 4.77 vs 5.36 — while the
                                      # 4090 prefers 1; and mode 1's fat hit shader pays occupancy
                                      # per sample, B70 marginal 2.2 ms/sample vs mode 2's 1.11,
                                      # so high spp widens it). ANY explicit --dxr-inline N — 1
                                      # included — sets dxr_inline_explicit, the policy's veto
                                      # (presence-not-value, the --spin-frames doctrine), and a
                                      # settings-file value vetoes too (the renderer.mode
                                      # precedent: the menu writes it). Armed modes need tier
                                      # 1.1/SM 6.5 (lib_6_5); lesser hardware degrades to 0 with
                                      # one loud line. The cross-vendor default stays quiet; 0/2
                                      # print (on Intel the mode-2 line names the vendor route +
                                      # the opt-out); an illegal value exits 2 (CLI) / warns
                                      # (settings file). Headless (--check*/--spin) never runs
                                      # the vendor policy — gates stay a pure function of the
                                      # command line. See the DXR section's ablation table.
                                      # 3 = THIN CHS + DEFERRED COMPUTE SHADE (2026-08-03, built on
                                      # the mechanism campaign's finding that Arc executes a fat
                                      # shader hosted in a raygen/CHS stage at 3-4.5x its compute
                                      # cost — FR_ABL=nosec collapsed mode-1 DXR 2.395 -> 0.478 ms,
                                      # BELOW the compute reference's 0.604, with the component
                                      # ablations sub-additive and `noglass` "saving" 0.28 on a
                                      # glassless scene: an occupancy/spill tax, not ray work; the
                                      # inherited t_start measured EXACTLY 0.000 via the new
                                      # FR_ABL=tzero lever). Raygen fires ONLY the bare-hit primary
                                      # (HgHit — cutout any-hit + relief re-march inherited) and
                                      # writes a 20 B record at u7 (the wavefront's dead qleaf
                                      # register, the cloud-cache u5/u6 precedent); dxr_shade.hlsl
                                      # (cs_6_5) shades from the record with rt.hlsli's inline
                                      # secondaries; one sample per pass pair (index in the b1 push
                                      # constants), cross-pass sum at u8, one store-or-add splat on
                                      # the last pass. MEASURED (spin path 1080p spp=1, dxr core,
                                      # default/stress/SM-lp): B70 1.39/1.56/1.56 vs mode 1's
                                      # 2.51/1.67/2.20 — THE BEST DXR ARM ON ARC, and the thin
                                      # dispatch is finally cheap (dxr-rays 0.23-0.35; THE WORLD
                                      # 0.54 vs mode 1's 2.87). NOT promoted, two measured reasons:
                                      # 4090 mode 1 still edges it (0.224 vs 0.243; spp=16 mode 2
                                      # 2.31 vs 3.53 — 2N RTPSO rebinds), and on Arc the DEFERRED
                                      # KERNEL now pays the codegen tax the CHS used to (dxr-shade
                                      # 1.124 vs the reference kernel's 0.603 for strictly MORE
                                      # work there — Arc compute codegen is knife-edged; the D2
                                      # lottery read 1.41/1.77/2.46 across identical builds), so
                                      # the wavefront still wins every Arc point (0.745 spin /
                                      # 3.25 world vs D3 1.39/4.73) and the Intel vendor default
                                      # STANDS (see vendor_defaults' 2026-08-03 paragraph).
                                      # Follow-on that would change that: split the deferred
                                      # kernel (hit/sky — the wavefront's own leaf+sky lesson) or
                                      # find its register cliff; dxr-shade < reference is the bar.
                                      # KNOWN REFUSAL: mode 3 + --heightfield on Intel driver
                                      # 32.0.101.8805 hangs the device (DEVICE_HUNG, GBV silent,
                                      # 4090 passes the identical suite) — DxrGpu::new degrades
                                      # the combo to mode 1 with a loud line; re-test on newer
                                      # drivers. COMPARISON-TARGET NOTE: with mode 2 now the Intel
                                      # DXR default, mode 2 is D3's Arc bar — JUDGED PER BINARY
                                      # 2026-08-04 (merged tree, B70, same binary, ABBA ±0.01):
                                      # D3 wins ONLY the default scene (span 1.40 vs 1.80, −22%);
                                      # D2 wins stress (1.40 vs 1.44), SM-lp (1.94 vs 2.20), THE
                                      # WORLD parked (4.93 vs 5.08), and every spp=16 point by
                                      # 26-85%. The thin half works everywhere (dxr-rays
                                      # 0.25-0.52, world 2.70 -> 0.52); the deferred kernel is
                                      # the whole loss (dxr-shade 1.12/1.10/1.81/2.35 vs the
                                      # ~0.60 reference class) — NO promotion, mode 2 keeps the
                                      # Intel default, dxr-shade < reference stays the bar; both
                                      # arms are lottery-prone, re-judge per binary.
                                      # Gates: --check-dxr --dxr-inline 3 green on
                                      # default/smlp/stress (B70) and smlp+relief (4090); 4 new
                                      # cargo-test source pins (miss-sentinel-before-consumers,
                                      # no-TraceRay-in-the-cs-unit, rt_dxr guards intact + inst
                                      # guarded, thin-raygen-writes-only-the-record)
cargo run --release -- --dxr-sbt 1    # EXPERIMENT lever (default 0 = off): the many-record,
                                      # MATERIAL-SORTED SBT ladder — the Intel-brief Q4
                                      # counterfactual (the TSU sorts by shader RECORD; our SBT
                                      # had effectively one). 8 field-derived shading classes
                                      # (src/shadeclass.rs — the STRIPS table IS the soundness
                                      # argument: a class may strip a shade arm only when its
                                      # membership predicate forces that arm's guard data-false,
                                      # self-tested in --check + re-verified on the LIVE scene
                                      # at upload; anything not provably strippable lands in
                                      # uber) partition every blas-split chunk into per-class
                                      # SUB-CHUNK instances (blas_split::refine_by_class —
                                      # INSTANCE-keyed, never multi-geometry: PrimitiveIndex()
                                      # restarts per GEOMETRY, which would break tri_of on BOTH
                                      # pipelines and drag GeometryIndex()'s SM 6.5 floor into
                                      # the lib_6_3 mode-0 path; instance-keying keeps the remap
                                      # contract with ZERO shader edits, and the wavefront
                                      # ignores hit groups entirely so the grown TLAS is
                                      # transparent to it). Each instance carries
                                      # InstanceContributionToHitGroupIndex = class*3 into a
                                      # class-major [HgShade_ck, HgHit, HgOcclude]x8 SBT — every
                                      # TraceRay call site untouched (multipliers stay literal
                                      # 0). The sway tail is RELABELED, never split (the
                                      # cells-parallel contract); sub-chunks stay under the cap
                                      # by construction; windows/stream/FR_SPLIT_AUDIT derive
                                      # from the mutated plan and need no changes. MODE 1 =
                                      # ALIAS records: 8 ExportToRename aliases of the ONE
                                      # chs_shade — identical code, distinct sort keys, zero new
                                      # compiles — isolating the PURE record-sort/repack effect
                                      # (plus the sibling sub-chunk AABB overlap cost, the
                                      # structural price of instance-keying). MODE 2 =
                                      # SPECIALIZED records: one extra DXIL library per class
                                      # PRESENT in the scene (k != uber), compiled with
                                      # shadeclass::strip_defines(k) prepended — shade.hlsli
                                      # gained SHADE_MAT_* macro seams over every material-
                                      # feature guard whose #ifndef defaults ARE the verbatim
                                      # expressions (all five pasting units stay semantically
                                      # identical unarmed — the same-seed wavefront-vs-reference
                                      # bit A/B is the drift tooth; REFL's seam carries the MIS
                                      # coupling: refl_ray feeds the VNDF block AND the w_l
                                      # reweight, so a strip keeps w_l=1 and light sampling
                                      # delivers the whole sun specular, rng pair inside the
                                      # gate so streams never need a burn). Each specialized
                                      # library exports exactly {chs_shade_ck <- chs_shade};
                                      # lib 0 aliases only uber + ABSENT classes (exported
                                      # names are state-object-unique; ah_*/misses resolve
                                      # cross-library). The identifier audit's REQUIRED set
                                      # narrows to specialized ∪ uber, and a dedupe there fails
                                      # HARD on every vendor (different libraries folding is a
                                      # defect, not a quirk) — MEASURED 2026-08-04: NVIDIA
                                      # mints DISTINCT identifiers for specialized libraries
                                      # (3/3 default scene), so it genuinely joins the ladder
                                      # at this rung; mode-2-vs-mode-1 accum drift is the
                                      # predicted DXC-rescheduling class (default scene: ~1% of
                                      # channels at max |d| 7.6e-6 — noise-scale, which is why
                                      # that compare is REPORT-ONLY and the statistical suite
                                      # is specialization's gate: a mis-routed class strips a
                                      # LIVE arm and blows T2's 2% loudly). MODE 3 = RECURSIVE
                                      # class dispatch: rung 2's records dispatched the way
                                      # production titles feed the TSU — every reflection/
                                      # glass continuation is a REAL TraceRay at
                                      # RayContribution 0, so the hit instance's class*3
                                      # contribution lands it in the hit surface's OWN
                                      # specialized closest-hit (routing = SBT arithmetic,
                                      # zero shader-side dispatch; rt_dxr.hlsli::trace_shade).
                                      # shade_split's DXR_SBT_RECURSE arm collapses the lap
                                      # loop to one iteration — the hardware ray stack
                                      # replaces the stash; Beer–Lambert multiplies the
                                      # RETURNED radiance (the CPU's own association); ind_s
                                      # becomes the literal rtput*child_color; rng round-trips
                                      # through the payload so the stream keeps the CPU DFS
                                      # draw order; depth+cone ride the repurposed sp lanes
                                      # (no payload growth past the 32 B config). HYBRID:
                                      # shadow/AO occlusion stays inline RayQuery (rt.hlsli
                                      # rides along; lib_6_5 + tier 1.1 or degrade to 2),
                                      # which is what caps MaxTraceRecursionDepth at 5
                                      # (primary 1 + refl 1 + the depth<TRANS_MAX_DEPTH=4
                                      # chain's 3 — the pipe_cfg derivation; exceeding a
                                      # declared depth is device removal, so the bound is
                                      # soundness, not tuning). Continuation misses take the
                                      # miss_rec SENTINEL (miss index 3 — the 4th record
                                      # fills the SBT's [64,192) miss gap to the byte): t=INF
                                      # and NO sky, because a reflection miss needs the
                                      # PARENT lobe's MIS weight — the parent keeps its own
                                      # miss arms. Arms only at --dxr-inline 0 (inline modes
                                      # have no TraceRay continuations to redirect; asked-for
                                      # anyway degrades to 2 loudly). Unbuilt/unarmed rungs
                                      # degrade loudly at DxrGpu construction. A dev MEASUREMENT
                                      # lever, the --sw-rays class: no vendor policy, no
                                      # settings row, loud on every armed mode, off-state
                                      # byte-identical (source AND instance descs). Must be set
                                      # at parse — the SceneGpu core bakes contributions at
                                      # UPLOAD (a partition-free core degrades the pipeline to
                                      # the one-record SBT with one loud line); --dxr-inline 2
                                      # composition is VACUOUS (zero TraceRay dispatches no
                                      # record) and --dxr-inline 3 nearly so (only the thin
                                      # bare-hit record dispatches — the sorted SHADING records
                                      # never run), both said loudly. Gates: shadeclass::self_test (the
                                      # strip-soundness must-fire + all-8 anti-vacuity) and
                                      # blas_split's refine spec-replay + grow must-fire in
                                      # --check; `--check-dxr --dxr-sbt 1` adds T1d — the
                                      # construction audit (>=2 live classes; PAIRWISE-DISTINCT
                                      # alias identifiers — MEASURED 2026-08-04: NVIDIA DEDUPES
                                      # the 8 aliases to ONE identifier on every scene while
                                      # the Intel B70 mints all 8 distinct, so rung 1 is an
                                      # INTEL-ONLY instrument (the vendor the TSU experiment
                                      # exists for) and NVIDIA joins at rung 2 where genuinely
                                      # different libraries cannot dedupe; the gate is HARD on
                                      # Intel, a recorded loud note elsewhere) and the
                                      # alias-vs-off same-seed A/B through a
                                      # SECOND SceneGpu core (the partition changes the plan, so
                                      # the T1c one-core flip is insufficient): BIT-identical
                                      # accum/tbuf/info on tint-free scenes — aliases run
                                      # identical code — with transmissive scenes printed
                                      # ungated (any-hit tint order is hardware-arbitrary by
                                      # contract, and the partition moves exact-t ties). NOTE
                                      # the routing wiring's real teeth arrive with rung 2:
                                      # under aliasing a mis-routed class is image-neutral by
                                      # construction; specialized records make it fail T2.
                                      # `--check-dxr --dxr-sbt 2` swaps T1d's image arms (the
                                      # off-core bit A/B cannot hold under rescheduling): (a)
                                      # rebuild DETERMINISM, bit-exact and HARD — a second
                                      # armed pipeline on the SAME core (partition identical
                                      # across armed modes; only RTPSO/SBT differ) must
                                      # reproduce accum/tbuf/info to the byte — and (b) the
                                      # mode-1 comparison, report-only (above). Both suites
                                      # pass armed on default/stress/SM-lp, NVIDIA + B70.
                                      # `--check-dxr --dxr-inline 0 --dxr-sbt 3` gates mode 3
                                      # the same way (armed rows REQUIRE the explicit
                                      # --dxr-inline 0 — the parse default is 1 and headless
                                      # never runs the vendor policy, so without it the row
                                      # silently gates rung 2) — green on default/stress/
                                      # SM-lp × NVIDIA + B70 + the GBV run. DEEP-CHAIN
                                      # LIVENESS is pose-bound (the committed poses recurse
                                      # only to depth 2 — the SM-lp default frame's drift
                                      # report is bit-equal mode 2's, the tell): the glassware
                                      # close-up (--cam 0.71,1.55,0.45,0.71,1.25,-0.35,
                                      # SM-lp) is the depth-proof pose — radiance A/B 0.031%
                                      # NV / 0.054% B70 with 444800 hit px of glass chains
                                      # live and no depth violation; its exit 1 is ENTIRELY
                                      # the documented mv_selftest close-up caveat (median
                                      # 3.156 vs 0.17 limit, vendor-independent, pre-existing
                                      # — read the log, not the exit code, at that pose).
                                      # MODE 2'S KNOWN APPROXIMATION, measured at that same
                                      # pose: a specialized record's lap loop also shades its
                                      # CONTINUATION surfaces, which can be a different class
                                      # (a tex-opaque parent's strips drop a glass child's
                                      # transmission) — mode-2-vs-mode-1 drift max |d| 9.61e-3
                                      # (~3% of channels) vs the 5.96e-8 fp floor; real,
                                      # bounded under T2, the documented price of an occupancy
                                      # instrument. Mode 3 closes it BY CONSTRUCTION (every
                                      # surface shades in its own class record) — the second
                                      # reason that rung exists.
                                      # THE LADDER, MEASURED (2026-08-04, --spin path 1080p,
                                      # min of 2 reps forward/reversed, spans in ms at
                                      # default/stress/SM-lp; CSVs + protocol in the session
                                      # scratchpad's matrix1): at --dxr-inline 0 — the
                                      # by-the-book all-TraceRay pipeline, the TSU's regime —
                                      # B70 spp=1 reads sbt0 8.02/5.31/6.79 → sbt1
                                      # 8.60/5.39/6.67 (FLAT with 8 genuinely distinct sort
                                      # keys) → sbt2 2.51/2.38/2.17 (−55..−69%) → sbt3
                                      # 1.49/1.78/1.29 (−66..−81%); 4090 1.17/0.83/1.15 →
                                      # flat → 0.25/0.34/0.24 → 0.19/0.27/0.20. Per-sample
                                      # marginals (spp16−spp1)/15: B70 7.02/4.54/5.92 → sbt3
                                      # 0.93/1.03/0.81 (5-7x); 4090 1.26/1.48/1.00 → sbt3
                                      # 0.15/0.29/0.14. At --dxr-inline 1, sbt2 is −20..−30%
                                      # on the 4090 (0.186/0.228/0.202) and −50..−60% on the
                                      # B70 (1.05/0.87/0.97) — which BEATS the same-day
                                      # inline-2 (1.80/1.40/1.94) and inline-3 (1.40/1.44/
                                      # 2.20) bars: `--dxr-inline 1 --dxr-sbt 2` is the
                                      # fastest DXR configuration measured on Arc (the
                                      # wavefront still wins outright — 0.64/0.78 recorded).
                                      # FOUR READINGS: (1) sort keys ALONE buy ~0 on both
                                      # vendors — mode 1 is flat even where the TSU has 8
                                      # distinct keys, because sorting identical fat shaders
                                      # has nothing to gain; (2) SPECIALIZATION is the prize —
                                      # thin per-class hit shaders recover 55-80% of the
                                      # by-the-book pipeline's cost, refining the launch-tax
                                      # story: most of the tax was the FAT UBER SHADER hosted
                                      # in RT pipeline stages, not TraceRay itself; (3) the
                                      # recursion rung lands the textbook pipeline at parity
                                      # with the inline hybrids (B70 sbt3 1.49 vs the same-day
                                      # inline-3 1.40; 4090 0.19 vs inline-1-sbt-2's 0.186) —
                                      # noting sbt3-vs-sbt2-at-inline-0 confounds recursion
                                      # with inline occlusion; (4) specialization also
                                      # STABILIZES Arc codegen — the >15%-spread rows in the
                                      # rep-trust check are all fat-shader configs (mode 0/1
                                      # SM-lp), the specialized rows repeat tight.
                                      # ENVIRONMENT (2026-08-04): the AMD iGPU ("Radeon(TM)
                                      # Graphics", driver 32.0.21018.14) AVs 0xC0000005 inside
                                      # CreateStateObject/identifier query on ANY armed mode —
                                      # mode 1 (Commit A code, single library) crashes
                                      # identically, mode 0 passes, the SAME run passes under
                                      # --gpu-debug (the debug layer masks it), and NVIDIA's
                                      # debug layer validates the identical descs clean — so
                                      # the driver chokes on ExportToRename itself, the
                                      # pre-existing-iGPU-environment class (the spp-readback
                                      # precedent). Deterministic (2/2 reps). The vendor
                                      # rename triptych: NVIDIA dedupes, Intel mints distinct,
                                      # this AMD iGPU crashes. Not coded around — the ladder
                                      # is a dev lever and the iGPU is not a measurement
                                      # target; re-probe when an RDNA4 discrete card returns.
cargo run --release -- --check-dxr    # DXR pipeline gate suite (needs a real RT GPU + the DXC DLLs;
                                      # composes with --stress; exit 2 = environment, 1 = a gate failed)
```
