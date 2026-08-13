# Intel Arc / Xe2 (Battlemage): what the hardware actually offers

The `## Intel Arc / Xe2` section: what the vendor extensions do and do not offer, the zero-LDS rule for RT kernels, the thread-sorting-unit findings, and the register-cliff and DispatchRays-regime campaigns.

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

## Intel Arc / Xe2 (Battlemage): what the hardware actually offers

Researched 2026-07-26 against primary sources (Intel's *Arc Graphics Developer Guide for
Real-time Ray Tracing in Games* v4, the `igdext.h` public header, Microsoft's DXR 1.2 / SER /
OMM / work-graph specs, and 1438 real `d3d12info` capability dumps including two Arc Pro B70).
**The vendor-specific surface is narrower than it looks, and most of it this tree already
satisfies.** Recorded so it is not re-researched.

**Ruled out — do not spend time here again.**
- **Intel Extensions for DirectX (`igdext`) cannot control SIMD/wave width.** The only `SIMD`
  token in the 1326-line header is a read-only `SIMD16Required` query;
  `INTC_D3D12_CreateComputePipelineState` is a shader-BYPASS path (CM/SPIR-V/ESIMD instead of
  DXIL) with no width field. Its D3D12 HLSL surface is 9 functions, all 64-bit atomics. There is
  no public SDK repo any more (headers survive vendored in `intel/gits`, v4.20.5).
- **Its 64-bit typed-atomics extension is obsolete on Battlemage**:
  `AtomicInt64OnTypedResourceSupported` is false on A770 but **true** on B-series, so the
  standard SM 6.6 path covers it. (Groupshared 64-bit atomics remain unsupported on both.)
- **Opacity Micromaps: not supported on Intel** ("actively evaluating" — Microsoft). Our
  `ALPHA_CUTOUT` candidate loops stay the answer, and `FR_ABL=noalpha,notrans` already measured
  that whole ceiling at −2.3%, so this costs nothing.
- **SER gates on Shader Model 6.9, not `RaytracingTier`.** Measured here: the B70 reports SM 6.8
  and the 4090 SM 6.8, so neither can run it today — and SER accelerates *many divergent hit
  shaders*, which a one-shader-record SBT does not have.
- **XMX is dead weight** without a neural stage of our own (XeSS already uses it through Intel's
  DLL). `WaveMMA` never shipped in any Shader Model; the D3D12 door is Cooperative Vectors, which
  Microsoft has already deprecated in favour of an SM 6.10 redesign.

**Already satisfied — three facts that explain existing decisions, and one rule not to break.**
- **THE RULE: groupshared memory is allocated out of the same L1 that services the RT unit**
  (RT guide v4, p.26), so LDS in a ray-tracing kernel degrades ray throughput — "even if the
  groupshared memory is running on a different queue (e.g. an Async compute queue)". This tree is
  compliant by construction and that is worth keeping: the two kernels that trace rays
  (`cs_leaf`, `cs_hemi_leaf`) have **zero** groupshared — neither pastes `frustum.hlsli`, see
  `trace.rs`'s `leaf_of`/`hemi_leaf_src` — while the LDS-carrying kernels (`cs_level*`,
  `cs_hemi_root/cell`) trace no rays; and with one DIRECT queue and barriers between the ladder
  and the terminal fills, LDS-heavy and RT-heavy work never overlap. `leaf.hlsl`'s header carries
  the warning. Intel's prescribed alternative to LDS traffic — **wave intrinsics** — is what
  `ctr.hlsli`'s `ctr_add`/`ctr_bump` and `wavefront.hlsl`'s `gw_alloc`/`gw_min_if` now use.
  (Caveat: Intel's statement is Xe-HPG-era. Xe2 unified L1/SLM further, 192 → 256 KB, so the
  structural premise is if anything more true — but the penalty is not re-confirmed for Xe2.)
- **The Thread Sorting Unit sorts by shader RECORD, not shader function**, and it is disabled by
  RayQuery. That is why `--dxr-inline 1` beat recursive TraceRay 4–6× here rather than losing to
  it: our SBT has effectively one record ("identifier-only SBT records, no local root
  signatures"), so the DXR 1.0 path paid the full repacking cost — live state spilled to the ray
  stack, thread terminated, continuation dispatched, payload through memory, 256 B/ray minimum —
  for **zero** coherence benefit. Intel's own document predicts that outcome, and the guidance
  would only flip if we grew many materially different hit records.
- **Xe2 made `ExecuteIndirect` a hardware block** (Intel quote: up to 12.5× vs Alchemist's
  software emulation), which retroactively justifies the wavefront ladder on Intel and matches
  our own measured ~11 µs of ladder dispatch overhead across 8 levels. Intel's "avoid
  ExecuteIndirect on buffers of size 0 or 1" rule is **A-series-era guidance for the emulated
  path** — do not act on it for Xe2 without measuring.

**Measured caps on this box** (printed by `--check-gpu`; `query_caps` now walks the shader-model
seed down from 6.9, having previously reported 6.7 purely because 6.7 was what it asked about):

| | Arc Pro B70 | RTX 4090 |
|---|---|---|
| RT tier | 1.1 | **1.2** |
| shader model | 6.8 | 6.8 |
| wave lane count | **16..32** (A-series: 8..32) | 32..32 |
| total lanes | 8192 | 16384 |
| work graphs | **Tier 1.0** | Tier 1.0 |
| `WaveGetLaneCount()` @ group 32 / 64 / 256 | **32 / 32 / 32** | 32 / 32 / 32 |

Two things follow. The lane count is a **range the driver picks inside per shader**, so the caps
never predict it — `trace::wave_probe` asks a kernel of each shipping group width and
`--check-gpu` prints the table (it also FAILS on an inconsistent report, since aggregation that
reasoned about the wrong partition would be silently wrong). That consistency check is a
**ceiling**, `waves == ceil(group / lanes)`, never exact division: a group NARROWER than the wave
is one PARTIAL wave, which is exactly what a 32-thread group is on wave64 hardware — and 32 is a
shipping width (`cs_level`, `cs_level_wide`, `cs_hemi_*`), so an exact-division predicate would
fail the suite on AMD RDNA for no defect at all. And Xe2 dropped SIMD8, which is the
8 → 16 minimum above. **`LEAF_GROUP = 256` was never a wave-matching result** — at 32 lanes it is
8 waves — it is an occupancy/dispatch-shape result, exactly as its own doc comment concluded.

**Wave-aggregated atomics — SHIPPED, and MEASURED NEUTRAL. Do not re-derive this.** Intel's
guide prescribes wave intrinsics over shared-memory traffic, and the tree used ZERO of them, so
the three atomic hot spots were converted: `bound_query_wave`'s `gw_min_if`/`gw_alloc` (the FTREE
round was issuing up to 8 LDS atomics per lane per node — ~256 per group iteration to one
address; it is now one wave reduction and one atomic per node, via a per-lane bitmask compacted
by popcount rank), `level_finish`'s counter bumps through `ctr.hlsli`'s `ctr_add`/`ctr_bump`
(layered ON TOP of the HOMOGENEOUS-BATCH quadrant folding: 4 x 32 becomes 1), and `leaf.hlsl`'s
per-pixel `CTR_HEMI_PT`. Revert arm: `FR_ABL=nowave` (`nobatch,nowave` is the full pre-wave-pass
queue code).

Measured `--spin path` 1080p, interleaved, rep 1 discarded, median of 3, windowed `gpu frame
span`: **B70 default 0.780 vs 0.780 (0.0%), B70 stress 1.045 vs 1.045 (0.0%), 4090 default 0.246
vs 0.246 (0.0%), 4090 stress 0.505 vs 0.510 (-1.0%)**. The B70 repeats to ±0.001 ms, so those
zeros are real, not noise. **The atomics were simply never the bottleneck** — the same conclusion
the ladder's own comment already reached about prep dispatches and barriers (0.011 ms across 8
levels): the cost is the level KERNEL descending the BVH, not the bookkeeping around it. Kept
because it is strictly less shared-memory traffic, is what Intel documents, and costs nothing;
but do not expect it to buy anything, and do not spend further effort on atomic contention in
this ladder without new evidence. CAVEAT ON THAT TABLE (2026-08-01): those A/Bs were taken while
`nowave` reached only the ctr.hlsli half — wavefront.hlsl's `gw_*` frontier aggregation stayed
ARMED in the "revert" arm (the probe-reach trap, instance 3; the arm is dual-homed now). The
leaf/sky/counter halves really were neutral as measured; the LADDER half of the feature was never
actually A/B'd until the repair. RE-MEASURED 2026-08-01 with both halves armed (`--spin path`
1080p, current tree; B70 2 reps ±0.002, 4090 3 reps), and THE VERDICT INVERTS: `FR_ABL=nowave`
BEATS the shipping code on BOTH vendors — B70 span 0.637 → 0.628 default / 0.775 → 0.752 stress
(ladder 0.110 → 0.102 / 0.200 → 0.180, −7%/−10%), 4090 span 0.255 → 0.252 / 0.350 → 0.345
(ladder −5%). `nobatch,nowave` lands BETWEEN baseline and nowave (B70 stress 0.759), which
decomposes the pair cleanly: the HOMOGENEOUS-BATCH half is a keeper, and the `gw_*` frontier
aggregation is the whole regression — small (~1-3% span, 5-10% ladder), cross-vendor, and
invisible for a month because the revert arm never reached it. The "costs nothing" premise above
is retired; flipping the gw half back to plain atomics (keeping ctr.hlsli's, which the old
half-armed A/B measured correctly as neutral) is the open follow-on — PAID 2026-08-09, see the
Battlemage-guide campaign below.

**THE 2026-08-09 BATTLEMAGE-GUIDE CAMPAIGN — the B70 optimization guide
(`C:\Docs\Intel\B70\B70_OPTIMIZATION.md`, the oneAPI guide translated to graphics vocabulary)
cross-referenced against this tree; the untaken items built and MEASURED, most of them to a
NO.** One measurement trap first: a CONCURRENT session was benchmarking the B70 during the
first pass — caught mid-campaign (a live foreign frustracer PID), every B70 verdict re-taken
on a quiet box, and the contaminated pair read nearly identical to the clean one (the
contender was parked/idle — a CONSTANT background load preserves deltas), but only the
re-measure could prove that. Check `Get-Process frustracer` before believing any interactive
A/B on this box. The verdicts, each behind its lever:

- **gw_* frontier aggregation → plain atomics, SHIPPED** (the follow-on above; user-approved).
  Only wavefront.hlsl's gw_alloc/gw_min_bits flipped — ctr.hlsli's global-counter halves keep
  their wave forms (measured neutral) and the homogeneous batch stays. `FR_ABL=wavegw` re-arms
  the aggregation (tile-unit-only; a simultaneous `nowave` wins); pixel-identical both ways.
  Clean-box flip-day numbers: B70 default span 0.854/0.854 plain vs 0.857/0.858 wavegw (plain
  ahead in every interleaved rep), stress dead even; 4090 default 0.420 vs 0.430. The 08-01
  ladder magnitudes did not reproduce (tree drift — leaf grew RTGI/spec-aa); the default
  stands on plain-never-loses + both vendors' small edges + simpler code.
- **g_stack DEPTH-MAJOR transpose (guide §6.3), SHIPPED AS A WASH** — the one genuinely new
  find: the serial DFS stack's `lane * LANE_STACK + sp` indexing strode lanes 64 B apart, the
  exact 16-banks×4-B pathology, at EVERY legal FR_LSTACK (all powers of two). The transpose
  (`GS_AT` in frustum.hlsli, covering the binary AND ftree bodies; `FR_STACK_LAYOUT=lane`
  restores v1) measured a WASH everywhere the serial path runs — --no-wide-levels ladders
  bit-repeatable at ±0.002 (ladder-sum 0.385 vs 0.385 default, 1.592 vs 1.594 stress), hemi-gi
  bench ±0.2% — because every stack op sits beside a BvhNode/FtNode GLOBAL fetch and the SLM
  serialization hides entirely under it. Ships anyway as the no-downside conflict-free form
  (bit-identical by construction — a pure address remap, unlike the gw lesson there is no
  behavior to regress); the shipping 1080p config barely touches the path at all (wide levels
  alias the slab flat, conflict-free).
- **Cloud coarse-march `[unroll]` → `[loop]`, SHIPPED** (trace_common.hlsli:~1011): the 6
  inlined bodies sat in spill-knee territory; clean-box B70 span 0.865/0.868 → 0.850/0.854
  default and 1.258 → 1.242 stress (leaf 0.649 → 0.638), 4090 a wash, FR_WIDTH unmoved
  (SIMD16 — spill traffic, not width). One token, no lever.
- **FR_DXR_LEAN → Intel VENDOR DEFAULT, SHIPPED** (the third vendor_defaults-family entry,
  applied at the call-site re-store block in run_window beside set_inline_mode): Intel +
  resolved mode 2 + sbt 0 arms the raygen-only RTPSO (the finding-1 audit's measured 12-18%
  of dxr-rays back on real scenes; 4090 nil). `FR_DXR_LEAN=1|0` is the explicit force/veto
  (lean_env), headless never runs the policy — --check-dxr gates both arms off the
  environment alone. Smoke-verified live on the B70 (announce + veto lines).
- **Record-load scalarization (guide §7.3): NO** — WaveReadLaneFirst on the per-group
  LeafRec/SkyRec/TileRec loads measured a wash (B70 leaf 0.645 → 0.643) and was dropped;
  ~2.4k record loads/frame is noise next to the ray work.
- **NRD/ReBLUR per-resource barrier narrowing (guide §11.2): NO** — built (per-resource UAV
  barriers on exactly the untransitioned STORAGE bindings each DispatchDesc names), gated
  green (N-suite + GBV clean, both vendors), and measured a WASH on clean-box parked B70 NRD
  sessions (span 6.145/6.144 global vs 6.153/6.132 narrow; nrd region 1.19 both): ReBLUR's
  ~31 dispatches are FULL-SCREEN — each fills the machine, so there is no cross-dispatch
  overlap for narrowing to unlock. Global stays the default; `FR_NRD_BARRIER=narrow` keeps
  the arm compiled for a future driver.
- **f16 register-pressure attack (guide §12.1): CLOSED WITHOUT CODE** — the DFS stash the
  strip sweep indicted is ~16 dwords of which 13 are GEOMETRY (ray origin/dir, hit t/uv)
  that cannot f16-quantize without breaking the eps-offset discipline (f16 granularity at
  world scale ±70 is ~0.03 vs eps ≈ 7e-3); the packable 2-3 registers are irrelevant below
  the knee (the ballast campaign measured ~56-60 floats of headroom in the reference class)
  and orders too small above it. Scoped `half` arithmetic (`-enable-16bit-types`) was gated
  on this showing headroom and is NOT taken: the u32-exact/precise/near-f16-max exclusion
  list carves out most of the shade path, and FR_WIDTH shows SIMD16 is sticky for RayQuery
  kernels regardless. The measured route to the spill knee remains kernel SPLITTING (the
  mode-3 follow-on), not packing.

Guide items already fought and won here, so nobody re-derives them: group sizing/dispatch
rounds (the leaf-frontier and cs_sky campaigns), zero-LDS RT kernels (the L1 rule), wave
intrinsics (the ctr/gw split verdict above), spills (FR_BALLAST/FR_WIDTH), ExecuteIndirect
(the ladder), readback rings (gputime/autoexp), 32-KB-safe SLM (2 KB slab). Deliberately NOT
taken, with reasons: async compute (user-deferred; concurrency unverified on B70, and the
LDS/RT-overlap discipline argues against casual pairing), `[WaveSize]` (documented — forces
the spill the compiler avoids by narrowing), ReBAR GPU_UPLOAD for the 4.5 KB FrameCb
(negligible traffic), the hemi batch's ~2500 barriers (opt-in still-frame mode).

**THE 2026-08-01 PRESSURE/OCCUPANCY CAMPAIGN — the remaining questions answered behaviorally.**
(1) DEAD-ARM REGISTER PRESSURE (the LEAF_NO_FB class) is real but MARGINAL on Xe2:
`FR_ABL=noffcode,noelcode` (compile the firefly + emissive code OUT — a day frame executes
identically either way, so the A/B isolates pure allocation) reads leaf −2.1%/−3.2%
(default/stress spin — ~9 µs absolute) and **~0 on THE WORLD** (leaf 1.011 baseline vs 1.01-1.03
across arms — the world's leaf is ray-bound, not allocation-bound). So the 2×2 leaf-PSO ship —
plus the sky/reference/DXR variants the firefly axis would drag in, each paying Arc's
async-compile warm-up — is NOT taken; the probe arms stay as documented instruments. (2) The
IGC ISA route is BLOCKED on driver 8805: `IGC_ShaderDumpEnable`/`IGC_DumpToCustomDir` (plus the
EnableAll/PidDisable variants) produce ZERO files from the D3D12 UMD, no default dump dir
appears — and the registry route is NOW ALSO PROVEN DEAD (2026-08-04, elevation obtained): the
same value names verified present under BOTH `HKLM\SOFTWARE\INTEL\IGFX\IGC` and the adapter's
class key (`...\Control\Class\{4d36e968-*}\0002\IGC`), a GENUINELY FRESH kernel variant forced
past the D3DSCache (`FR_BALLAST=7` — the width report proved it compiled and ran), zero files
anywhere. ISA dumps are unavailable on 8805 by every documented route; the occupancy question
is answered behaviorally instead — see the FR_WIDTH paragraph below, which supersedes
"finding 1: cs_leaf is not allocation-crippled" with the direct width readings. (3) Reference
points on the current tree for future diffs
(2 reps, ±0.002): procedural spin span **0.636 ms** (leaf 0.419, ladder 0.110), stress **0.775**
(0.462/0.200); parked WORLD XeSS session span **3.23** = leaf 1.011 + sky/caches ~0.15 + feed
0.231 + xess-eval 0.522 inside the replay bracket — reconciling exactly with the recorded 3.30
baseline.

**THE 2026-08-04 REGISTER-CLIFF CAMPAIGN — the pressure story MEASURED, not inferred**
(`FR_WIDTH=1` + `FR_BALLAST=N`, both default-off, unarmed sessions untouched — the tzero
class; gates green armed and unarmed on B70 + 4090, check.png byte-identical). **FR_WIDTH**
arms a WIDTH_PROBE epilogue in every real kernel (counter slots ≥ CTR_COUNT — never zeroed,
never gated, by construction; DXR gets a dedicated `width_buf` at its otherwise-unbound u3):
each kernel reports its COMPILED `WaveGetLaneCount()` — the per-shader SIMD width IGC picks
from register pressure, printed at the spin accounting site, both check suites, and the C-key
verify. THE TABLE (B70, driver 8805): **leaf=16 hemi=16 reference=16 dxr-raygen=16
dxr-shade=16 vs sky=32 level=32**; the 4090 control reads 32 everywhere. Three readings:
(1) every RayQuery-carrying kernel compiles SIMD16 — ctr.hlsli's old "32 at every group
width" note was the TRIVIAL wave_probe talking (that probe measures group shape, not the real
kernels; note amended); (2) the DXR raygen reads 16 even THIN (mode 3's bare-hit raygen) —
the RT launch regime itself narrows, independent of footprint (the brief's hypothesis 1,
half-confirmed from inside); (3) reference=16 == dxr-shade=16 means the 1.9× deferred-kernel
penalty is NOT width — it is SPILL AT SIMD16. **FR_BALLAST=N** proves that directly: N
synthetic live floats (loop recurrence on the traced t — not dead, not rematerializable,
[unroll] register-resident; folded under a never-true `spp == 0xdead` branch so the image is
bit-identical) injected into cs_reference. THE KNEE: per-float cost runs ~1.5-2 µs to N=48
(occupancy dilution), breaks 3× between 56 and 60 (0.704 → 0.785 ms — a +0.08 step where
4-float steps cost ~0.01), and accelerates past it (N=160 = 1.613 ms, 2.6× baseline) — **the
reference kernel's own live state sits ~56-60 floats below IGC's spill edge**, and dxr-shade's
1.9× is bracketed by reference + O(100) ballast floats. THE STRIP SWEEP (FR_ABL × FR_WIDTH on
dxr-shade, base 1.238 ms): nosec 0.463 (−0.775), norefl 0.750, **noglass 0.749 — −0.489 on a
scene with ZERO transmissive geometry** — and the single-strip savings SUM to 1.22 vs nosec's
joint 0.78: **cost near the cliff is a THRESHOLD, not a per-feature sum** — removing EITHER
big arm's live state clears the same spill edge (the CHS campaign's sub-additivity, reproduced
in plain compute; noffcode −0.05 = lottery noise). No strip flips 16→32 — SIMD16 is sticky for
RayQuery kernels; the lever is spill traffic at 16, not width. CONSEQUENCE for the mode-3
follow-on: `dxr-shade < reference` is reachable by getting the deferred kernel's live state
under the knee — the norefl/noglass overlap says the reflection+glass DFS stash is the hog, so
splitting the reflection lap (or hit/sky) out of the kernel is the measured route, not a guess.
Sweep discipline: every N and every strip is a NEW kernel variant (maiden discard), width read
from the report line, ms from the LAST gputime table.

*Measurement trap this campaign re-learned the hard way:* `--gpu-timing` prints a table every 120
frames AND at exit, and a parser that takes the FIRST match reads frames 0-119 — the coldest
window there is, where `win ms` == `mean ms` by construction. That alone manufactured ±20%
"noise" and an apparent 19% NVIDIA win, in a region (`leaf+sky`) the change cannot even touch.
Always parse the LAST table.

**THE 2026-08-04 DISPATCHRAYS-REGIME CAMPAIGN — the launch tax's mechanism, closed.** The
register-cliff campaign left one anomaly: IDENTICAL code (mode-2 raygen == cs_reference, same
compiled SIMD16) paid 2-2.4× under DispatchRays on fat scenes and parity on thin ones. Three
instruments answered it the same day. (1) **`dxr stack:` line** at every DxrGpu construction —
`GetShaderStackSize` per export (hit-group members by the qualified `group::stage` spelling;
0xFFFFFFFF prints `-`) + the driver's default `GetPipelineStackSize`, off the SAME
`ID3D12StateObjectProperties` the SBT identifiers already cast; `FR_DXR_STACK=min|<bytes>`
overrides via `SetPipelineStackSize` (`min` = the call-graph bound from the driver's own
numbers — mode 2 has no TraceRay, so its true bound is the raygen frame alone; undershooting
real usage is device removal by spec, so never guess). VERDICT: **stack reservation REFUTED**
— B70 defaults are tiny and honest (mode 1 = 112 B, mode 2 = 64-192 B even on SM-lp; the
formula is visible: mode 0 = 80 + 2×1056 = 2192), so there is nothing to reclaim and
FR_DXR_STACK is a probe, not a lever. The one gem: mode-0's uber CHS reports **1056 B ≈ 264
floats of TraceRay-live state** (4090: 544 B) — the driver itself printing why mode 0 was
catastrophic; NVIDIA reports near-zeros everywhere else (mode 1: raygen=32, rest 0). (2)
**`FR_BALLAST=dxr:N`** (the reserved prefix, implemented — reference.hlsl's three ballast
blocks mirrored into dxr.hlsl's mode-2 arm under a compound `BALLAST_N && DXR_INLINE_SEC==2`
guard; dxr.rs pushes the define only at mode 2 and refuses other modes loudly, since a seed
whose update compiled out would "measure" a flat curve). THE KNEE-VS-KNEE (default procedural
`--spin path`, same binary same day, maiden discard, last-table 600-frame-lap mean, ms):

| N | B70 ref | B70 raygen | 4090 ref | 4090 raygen |
|---|---|---|---|---|
| 0 | 0.610 | 1.646 | 0.270 | 0.260 |
| 16 | 0.638 | 2.097 | 0.270 | 0.427 |
| 32 | 0.660 | 2.799 | 0.263 | 0.594 |
| 56/64 | 0.704/0.780 | 4.038/4.400 | — /0.433 | — /0.952 |
| 160 | 2.078 | 10.675 | ~1.2 | 5.122 |

THE READING: **the raygen has NO knee — it prices live state from float ZERO** at ~20-45
µs/float on the B70 (accelerating; compute pays 1.7 µs/float to its +56 knee), and the N=0
gap (1.646 − 0.610 ≈ 1.0 ms) ≈ the kernel's own ~50 live floats at that rate — **the entire
baseline DispatchRays tax IS live state × the RT launch regime's spill rate**. The 4090
control shows the same SHAPE at lower severity: raygen prices immediately (~10 µs/float,
compute free to +32) but its budget still covers the kernel's own state (baseline parity
0.260/0.270), with a second cliff at 128→160 (2.03 → 5.12). Width never flips (16/32
throughout) — spill traffic, not width; cross-vendor in shape, Intel-specific in severity.
(3) **The host×strip cross** (SM-lp, B70, `FR_ABL` × {compute reference, mode-2 raygen}):
base 0.710/1.908, nosec 0.534/**0.696** — stripping the secondaries brings DispatchRays to
COMPUTE PARITY, i.e. the whole real-scene host gap is the secondary machinery's live state —
norefl 0.687/1.131 (the same code removal saves 0.023 in compute and 0.777 in the raygen,
**34× amplification**), noglass 0.706/1.195 (glassless scene — the threshold again),
noalpha,notrans 0.547/1.136; and the DXR column repeats the non-additivity (singles sum
−2.26 vs joint nosec −1.21). CONSEQUENCES: the world's 2-2.4× tax and the ~65% hybrid margin
are fully mechanistic — fat-shader live state at the regime rate — so the only real levers
are the THIN-RECORD family (mode 3's bare-hit raygen, sbt-2 specialization: both already the
measured winners, now with their mechanism), and there is no stack/width/scheduling fix left
to find locally. The remaining WHY — what the RT launch regime does to the register budget
that compute hosting doesn't (128-vs-256 GRF mode? ray-state co-residency in the GRF?) — is
IGC/driver internals, i.e. the brief's territory; the knee-vs-knee table is the repro to
hand over. Discipline notes: the exit gputime table can be a DEGENERATE 1-frame window —
parse the LAST row match's `mean ms` column, which stays valid there; and an explicit
`--spin-frames` under 1600 exits 2 on Arc (the warmup guard) — pass `--spin-warmup 0` for
construction-only reads like the stack line.

**THE 2026-08-05 FINDING-1 AUDIT — the DispatchRays-tax claim adversarially re-measured, and
it survives as the FLOOR of a band.** Before the brief's spearhead claim ("DispatchRays costs
~2x on register-fat shaders with zero TraceRay") shipped to Intel, it was re-proven at every
layer, same driver 8805. (1) **Zero TraceRay is now an ARTIFACT proof**: FR_DUMP_HLSL the
mode-2 library, `dxc -T lib_6_5 -HV 2021 -O3 -Fc`, grep the disassembly — `dx.op.traceRay` =
0 on both the procedural and SM-lp configurations, `rayQuery_TraceRayInline` = 16 each (the
anti-vacuity; the reference-unit control reads 0/9). Grep the SOURCE and you get 6 hits, all
preprocessor-dead — the DXIL is the artifact; a cargo pin
(`mode2_raygen_tracerays_are_preprocessor_dead`, a DEPTH-TRACKED #else scan — the mode-2 arm
nests a SKY_LOD #if/#else, and anchoring on the first textual #else would false-pass a
mode-2-live TraceRay) guards the guards. (2) **THE BRACKET-ASYMMETRY TRAP, a reusable
class**: the `reference` gputime region contained bind_common + BOTH cloud-cache fills + the
PSO set + the trailing barrier, while `dxr-rays` wraps the bare DispatchRays — every recorded
reference-vs-dxr-rays compare was two DIFFERENT bracket shapes (~3% in the DXR arm's favor
here; the general lesson: two regions compared as flat must first prove the same nesting).
`record_reference` now carries a nested **`reference-kernel`** row mirroring `dxr-rays`'
shape exactly — the like-for-like column for every future compare. (3) **Re-measured
like-for-like** (min-of-2, spin last-table mean / world WinMs): B70 procedural 0.569 vs
1.655 = 2.91x, SM-lp 0.674 vs 2.221 = 3.30x, SM-lp --tile 2 0.649 vs 1.957 = 3.02x; THE
WORLD (FR_REF sessions, --tod 11, boot pose) 1.178 vs 2.808 = **2.38x** (mode 1 3.11 =
2.64x; the recorded 1.28/2.59/3.13 replicate as outer-bracket 1.215 / 2.81 / 3.11 — mode 1
within 1%). 4090: 1.12x/1.36x/1.24x — cross-vendor in shape, Intel in severity. (4) **The
build lottery is ONE-SIDED**: three comment-touch rebuilds of trace_common.hlsli moved the
mode-2 raygen 1.18-1.66 ms (procedural) / 1.81-2.22 (SM-lp) while the compute reference
repeated to ±0.001 across ALL variants — the codegen instability is entirely the DXR door's.
Ratio bands: procedural 2.07-2.91x, SM-lp 2.68-3.30x — **"~2x" is the BAND FLOOR, never
crossed below by any draw, scene, or arm**. (5) **FR_DXR_LEAN=1 — the dead-exports control**
(dxr.rs; mode-2-only by soundness: raygen-only export list on the SAME DXIL blob, zero hit
groups, MaxTraceRecursionDepth 0, null miss/hit SBT ranges; default-off byte-identical;
check-dxr green armed on both vendors): the tax PERSISTS raygen-only (world 1.93x, SM-lp
2.85x, procedural 2.92x) — finding 5's intercept attribution confirmed by its strongest
control — AND removing the dead-but-exported fat entries recovers a dose-responsive **0%
(procedural) / 12% (SM-lp, 2.193→1.922) / 18% (world, 2.81→2.28, 0.53 ms)** of dxr-rays on
the B70 (4090 nil): the driver pays real per-dispatch cost provisioning exports no ray can
reach. (6) Width parity at the exact configuration: FR_WIDTH on SM-lp reads reference=16 AND
mode-2 raygen=16 — the previously-unrecorded cell. **FR_REF=1** (main.rs) starts an
interactive session in the reference arm (first-entry default only; resize re-entries keep
Persist) — the world reference arm was previously reachable only by a manual R keypress no
scripted protocol can fire. Verdict shipped in the brief v1.7: heading re-scoped to "2-3x on
the same code", the "register-fat" qualifier retired (finding 5: the kernel's own ~50 live
floats price from float zero — slim scenes pay too), and the audit block added to finding 1.

**WAVEVIZ (2026-08-05) — wave footprints made visible, and the launch-packing question
closed.** `--waveviz [chs]` (CLI, the user-facing spelling since the funnel promotion;
`FR_WAVEVIZ=1|chs` stays as the env alias, CLI wins — main's lever block owns the
precedence via `trace::set_waveviz`) arms the wave-ticket overlay: each covered kernel's
wave takes a POSITION-KEYED ID — `WaveReadLaneFirst(first lane's pixel/thread index)`,
minted converged, per wave never per stride iteration (leaf/sky use group×lane math + a
per-kernel salt) — unique per wave within a frame (a pixel belongs to exactly one wave)
and IDENTICAL across frames whenever the packing is identical, so a parked view is
color-STABLE and residual shimmer is REAL repacking. THE LESSON (shipped wrong twice
before landing here): an arrival-order ATOMIC ticket strobes at frame rate because wave
scheduling order is nondeterministic per frame, and a per-frame counter reset cannot fix
it — order, not magnitude, was the noise; both are retired (CTR_TOTAL back to 30, no
counter, no per-frame clears). Every pixel stores its wave's ID as its
LAST tbuf touch (`asfloat` bit-cast — tbuf has no live-frame consumer), and the overlay is
COMPOSITED AT THE PRESENT FUNNEL (waveviz.hlsl — its own PS + PSO on the tonemap root
signature, the HUD's exact shape: premultiplied blend, drawn after the tonemap draw inside
`fullscreen_to_backbuffer`), which is what makes it work under EVERY upscaler sub-mode —
RR/XeSS/FSR/quinlight included (feeding hash colors INTO a temporal model was rejected: it
would smear per-frame tickets; the resolve-stage colorizer that shipped first was
plain-arm-only and is deleted). Plumbing: tickets reach the PS as a t2 ROOT SRV (root
param 3, bound by VA — no descriptor), nearest window→render mapping off the draw's own
8-DWORD b0 layout (root constants are per-draw; tonemap/hud untouched), tbuf bracketed
UA↔PIXEL_SHADER_RESOURCE around the draw (the bloom bracket), and `GpuContext::waveviz_src`
(a Cell stamped by every presenter: Trace/Dxr/None) names whose tbuf to read — the None
stamp on CPU-fed presenters is what stops a SPACE back to CPU compositing a stale GPU
tbuf, and `present_again` inherits the Cell like `last_present`. Spin's `waves=` line
counts distinct IDs in the LAST frame's tbuf — still the per-frame wave-execution count
(fragment shards have distinct first active lanes), so the B70 fragmentation numbers
stay comparable across the ID redesign. **I** toggles it live in GPU
arms — TWO handler copies by structure: the gpu_trace arm `continue`s before the shared
toggle block, so it carries its own (the quality/spp pattern; a shared-block-only handler
shipped first and was dead code in --gpu sessions). The toggle sets `frame = 0` (a
CONVERGED still frame re-presents without tracing — no trace, no tickets — so plain
accumulation restarts; every history untouched; C-verify stands down while live — tbuf
holds tickets). Known-accept: P screenshots under upscalers read the upscaler output
PRE-funnel, so they exclude the overlay (the HUD's accept); headless `--spin` runs arm it for the whole run and
dump `waveviz-<arm>.png` + a compactness line (waves, px/wave, bbox stats — main.rs's
`waveviz_dump`, whose Rust hash mirrors resolve.hlsl's `wv_hash_color` term for term).
Covered: reference, leaf+sky (full wavefront coverage), the DXR raygen at inline 1/2
(mode 0 = lib_6_3 no wave ops, mode 3's thin raygen is pinned to write no tbuf — both
refuse loudly); `FR_WAVEVIZ=chs` (mode 1 only) tickets `chs_shade` instead, with the raygen
sentineling misses 0xFFFFFFFE (rendered dark). Compute units bump `counters[CTR_WV_TICKET]`
(= 30; CTR_TOTAL 31 — the never-zeroed ≥ CTR_COUNT class); the DXR pipeline uses
`width_buf` slot 2 (created for either probe now). Unarmed sessions byte-identical
(conditional defs pushes; `waveviz_blocks_are_guarded` + the widened dxr guard test pin it);
both check suites green unarmed on the 4090. THE ANSWER (60-frame parked spins, 1080p,
means over all waves): **launch packing is screen-tiled and FULL on BOTH vendors** —
reference / mode-2 raygen / mode-1 raygen-end all read bbox exactly 32 (4090, 8×4 at
SIMD32) or 16 (B70, 8×2 at SIMD16) at 100% compact with every lane live, so the
DispatchRays grid is packed exactly like the compute grid and the folklore is now a
measured row — **and the BTD-scatters-waves hypothesis is REFUTED, corroborating the
regime-pricing attribution** (the mode-2 tax cannot be packing). THE NEW FINDING: **Intel
fragments waves at TraceRay boundaries; NVIDIA does not.** B70 mode-1 raygen-end reads
195,203 wave executions for 129,600 waves' worth of pixels at **10.6/16 live lanes** (mean
bbox 132, 96.9% compact — mostly tiled, ~3% scattered shards), and the chs arm reads
163,284 hit waves at **8.9/16** — while the 4090's continuation and hit waves stay full
(31.9-32.0/32) and perfectly tiled. So mode 1 on Arc pays a SECOND mechanism beside the
live-state pricing: ~1/3 of post-TraceRay lanes are dead, i.e. TraceRay-heavy pipelines
lose wave occupancy at every stage boundary — consistent with finding 2's mode-0 column
and worth a brief row. The wavefront arm reads 1.5% compact BY DESIGN (our own grid-stride
spreads each wave across its whole ~540-px tile — the instrument documenting our dispatch
shape, not the driver's; do not read that row as a driver finding).

**Work graphs (`FR_WORKGRAPH=1`) — the ladder as a D3D12 work graph.** The one genuinely NEW
Xe2 capability, and the queue records were already "work-graph-shaped". `src/shaders/
workgraph.hlsl` replaces `cs_seed` + depth_full x (`cs_prep` + ExecuteIndirect) with ONE
`DispatchGraph`: a broadcasting node per shallow tile (the `bound_query_wave` frontier) handing
off to a coalescing node for deep levels (32 tiles per group — the `WIDE_LEVELS` split, which
work graphs express natively). Leaf and sky records keep their UAV queues, so every terminal
gate is untouched; only the ping-pong tile queues go, because they assume levels SERIALISE and
that is exactly what a graph breaks. `level_finish` is NOT forked — `#if defined(WORKGRAPH)`
swaps its child emission from `qout` to an `out TileRec[4] + mask` the node compacts by
popcount rank.

**Status: correct, and blocked on Intel's driver — re-confirmed on 32.0.101.8805 (2026-08-01):
the refusal arm was deleted locally per its own instruction and the IDENTICAL 0xC0000005 landed
at the first graph dispatch, backing ask still 517.62 MB, state object still building happily;
the arm now records both driver versions.** On the 4090 the whole `--check-gpu` suite
passes with the graph armed and the result is **bit-identical** to the ladder (`leaves 768 |
sky-tiles 4 | splits 257 | blocked 256 | cuts 65 | overflow 0`, same-seed image `mean |d| 0.00e0
max 0.00e0`). That took a fix worth recording, because the gate that caught it only fires at
half the resolutions: `cs_seed` used to enqueue a root TileRec unconditionally, and the graph
takes ITS root as CPU input, so `CTR_TILE_A` sat at 1 for the whole frame with no ladder level
to consume it. The depth-accounting gate reads A or B **by `depth_full` parity** (the last
level's INPUT counter is legitimately non-zero, which is why it cannot just check both), so at
800x600 with `LEAF_TILE` 32 — `depth_full` 5, odd — it read B and passed, while `FR_LEAF=16`
(`depth_full` 6) failed it outright with `1 tile records left`. `cs_seed` now takes `push0 = 1`
to skip the enqueue. **A parity-selected gate is only half a gate: prove an arm against BOTH
parities before believing it.** On the B70 the state object builds and reports `WorkGraphsTier 1.0`, then
`DispatchGraph` takes an access violation with the debug layer and GPU-based validation both
silent — so `FR_WORKGRAPH=1` is REFUSED on Intel with a loud line (`trace.rs`, the first real
caller of `adapter::picked_vendor()`; delete that arm and re-test on a driver newer than
32.0.101.8515). The corroborating tell is the backing-memory ask for the identical graph:
**517 MB on Arc against 82 MB on NVIDIA**.

**Performance: a WASH on NVIDIA, and scene-dependent** (`--spin path` 1080p, same discipline,
windowed span): **default 0.262 graph vs 0.245 ladder (+6.9%), `--stress 5000` 0.486 vs 0.505
(-3.8%)**; `leaf+sky` is unmoved (0.0% / +3.3%), so the delta really is the ladder. That fits
the mechanism: the graph deletes ~6 prep dispatches and lets levels OVERLAP, which pays on a
deep many-tile stress field and loses on the shallow sky-heavy default scene where the ladder's
own dispatch overhead was already only ~11 µs and Xe2/Ada both accelerate `ExecuteIndirect` in
hardware. **It is therefore an env lever and never a default** — and note it is worth exactly
ZERO on a resting frame either way, because structure replay skips the ladder entirely.

Three spec rules that shaped the design and are easy to get wrong:
- **Output allocation must be THREAD-GROUP uniform, and "varying includes threads exiting."**
  `cs_level_wide`'s `if (gtid.x != 0) return;` before `level_finish` would therefore be
  undefined behaviour in a node. The wide node keeps all 32 lanes alive: lane 0 computes the
  split into groupshared, a barrier publishes it, then every lane calls
  `GetThreadNodeOutputRecords` with a per-thread count of 0 or 1 (a varying COUNT is explicitly
  allowed; varying control flow is not). `OutputComplete()` is mandatory even for a 0-record thread.
- **Only self-recursion exists** (a node may not target an ancestor), the longest chain is 32
  with recursion levels counted individually, and `[NodeMaxRecursionDepth(0)]` means "not
  recursive" — which would make a self-output an illegal cycle, hence the `.max(1)` clamps.
- **Exceeding `[MaxRecords]` or the declared recursion depth is memory corruption or device
  removal, NOT a caught error.** So the deepest node counts any child it must drop into
  `CTR_OVERFLOW`, which every suite already gates at exactly 0 — a silent drop would be a hole
  in the image, i.e. the false-sky class.

**`[WaveSize]` is deliberately unused.** It is a *validated constraint* that fails PSO creation
out of range, it is compute-only (so it can never reach `dxr.hlsl`), and forcing 32 on a
register-heavy kernel converts a compiler-avoided spill into a mandatory one — Intel's compiler
picks the narrower width precisely to avoid spilling. If it is ever wanted, `[WaveSize(16,32,32)]`
(SM 6.8, range form) is the portable spelling; a bare `[WaveSize(32)]` fails on AMD wave64 parts.

