//! The by-the-book DXR pipeline (`--dxr`, the F key): an RTPSO state object,
//! a shader binding table, and DispatchRays with raygen / closest-hit / miss
//! shaders — hardware ray tracing dispatched the way the DXR spec draws it,
//! next to trace.rs's compute-wavefront + inline-RayQuery flavor. The CPU
//! renderer stays the reference. Shading parity is inherited, not re-ported:
//! the DXR library pastes the SAME trace_common.hlsli + shade.hlsli the
//! compute tracer runs, with rt_dxr.hlsli swapping the two trace primitives
//! from RayQuery to TraceRay — so the F toggle on a converged frame is an
//! intersector/dispatch A/B, not a shading A/B. Scene buffers, the BLAS/TLAS
//! (SceneGpu), the compute root signature (as the DXR GLOBAL root
//! signature — same registers), the FrameCb layout, and the resolve kernel
//! are all shared with trace.rs.
//!
//! SBT layout (rt_dxr.hlsli mirrors the indices — keep in lockstep):
//!   raygen @ 0    | miss @ 64: [radiance, shadow, hit_info]
//!   hit groups @ 192: [HgShade, HgHit, null (occlusion; the any-hit-only
//!   HgOcclude instead on alpha-masked scenes — see ALPHA_CUTOUT)]
//!
//! FR_DXR_INLINE (dxr_inline_mode below) is the W2 experiment lever: the
//! same pipeline with its rays moved onto inline RayQuery, one stage at a
//! time — the layout above is unchanged in every mode.

use super::d3d12::{self, committed_buffer, transition, uav_barrier, Result};
use super::dxc::Dxc;
use super::trace::{
    self, FrameCb, FrameParams, SceneGpu, CB_STRIDE, RP_FRAME_CBV, RP_GBUF, RP_GBUF_EXT, RP_PUSH,
    RP_SCENE_TEX, RP_SRV0, RP_TEX, RP_UAV0, SRV_INDICES, SRV_MATERIALS, SRV_NORMALS,
    SRV_POSITIONS, SRV_TLAS, SRV_TRI_MAT, TEX_HEAP_BASE, TEX_TABLE_BUFS, UAV_ACCUM, UAV_COUNTERS,
    UAV_INFO, UAV_QIN, UAV_QLEAF, UAV_QOUT, UAV_QSKY, UAV_TBUF,
};
use crate::scene::Scene;
use windows::core::{Interface, PCWSTR};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;

const RT_DXR_HLSLI: &str = include_str!("shaders/rt_dxr.hlsli");
const DXR_HLSL: &str = include_str!("shaders/dxr.hlsl");
const DXR_SHADE_HLSL: &str = include_str!("shaders/dxr_shade.hlsl");

const IDENT: usize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES as usize; // 32
/// Table starts are 64-aligned (D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
/// identifier-only records make the stride the 32-byte identifier itself.
/// The hit-table RECORD COUNT is runtime since --dxr-sbt (3 off-lever,
/// class-major 8×3 on the armed rungs) — `DxrGpu::sbt_hit_records` snapshots
/// it so the fill and DispatchRays cannot disagree; SBT_HIT itself never
/// moves (192 stays 64-aligned, and the miss table fits beneath it at 96
/// bytes — or at EXACTLY 128 under --dxr-sbt 3, whose 4th record, miss_rec,
/// fills the [64, 192) gap to the byte).
const SBT_MISS: usize = 64;
const SBT_HIT: usize = 192;

/// What DispatchRays needs, queried once. Tier 1.0 suffices (the compute
/// tracer's RayQuery needs 1.1; this pipeline predates it) and the library
/// compiles as lib_6_3. Missing caps are a clean "stay on the CPU" story.
pub fn require_caps(device: &ID3D12Device) -> Result<()> {
    let caps = trace::query_caps(device)?;
    let mut missing = Vec::new();
    if caps.rt_tier < D3D12_RAYTRACING_TIER_1_0.0 {
        missing.push(format!(
            "DXR raytracing tier 1.0 (DispatchRays) — device reports tier {}",
            caps.rt_tier
        ));
    }
    // 0x63 == D3D_SHADER_MODEL_6_3 (absent from the windows 0.62 bindings).
    if caps.shader_model < 0x63 {
        missing.push(format!("shader model 6.3 — device reports 0x{:x}", caps.shader_model));
    }
    if device.cast::<ID3D12Device5>().is_err() {
        missing.push("ID3D12Device5 (CreateStateObject/DispatchRays)".into());
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!("DXR pipeline unsupported here: {}", missing.join("; ")))
    }
}

/// `--dxr-inline` (cross-vendor default **1** — the W2 promotion; **2 on an
/// Intel adapter** via `main::vendor_defaults`, the B70-campaign promotion,
/// unless `dxr_inline_explicit` vetoes): which of this pipeline's rays ride
/// recursive TraceRay vs inline RayQuery, without leaving DispatchRays.
///   0 — all TraceRay: the original by-the-book pipeline, kept as the A/B
///       escape (bit-identical library to the pre-lever build).
///   1 — THE DEFAULT: primary TraceRay -> chs_shade, every secondary
///       shade.hlsli fires (shadow/AO/reflection/transmission/translucency)
///       an inline RayQuery inside the hit shader (rt.hlsli's bodies compile
///       in place of rt_dxr.hlsli's TraceRay flavors);
///       MaxTraceRecursionDepth 1. Promoted because it strictly DOMINATES
///       mode 0 at every measured point on both vendors — never slower,
///       −68 to −81% at the shipping spp=1 — while the payload/closest-hit/
///       SBT machinery keeps doing its real job for the primary.
///   2 — everything inline in raygen (dxr.hlsl's DXR_INLINE_SEC == 2 arm):
///       no TraceRay anywhere, DispatchRays as a bare launch grid over the
///       reference loop. The measurement arm that proved launch overhead is
///       ≈ zero — and since 2026-08-01 the INTEL DEFAULT (it beats mode 1 on
///       the B70 at every measured point: the spp=1 table below, world span
///       4.77 vs 5.36, and mode 1's fat hit shader pays occupancy per sample
///       — B70 marginal 2.2 ms/sample vs mode 2's 1.11 — so high spp widens
///       the gap; it was "the manual Intel pick" until vendor_defaults
///       automated it).
///   3 — THIN CHS + deferred compute shade (the 2026-08 Intel-campaign
///       finding: Arc executes a fat shader hosted in a raygen/closest-hit
///       stage at 3-4.5x its compute cost — an occupancy/spill tax nearly
///       independent of the rays cast; FR_ABL=nosec collapsed mode 1 from
///       2.395 to 0.478 ms, BELOW the compute reference's 0.604 — the
///       existence proof). Raygen fires ONLY the bare-hit primary (HgHit —
///       cutout any-hit + relief re-march inherited) and writes a 20 B
///       record at u7; dxr_shade.hlsl (cs_6_5, this file's `Mode3`) shades
///       from the record with rt.hlsli's inline secondaries. One sample per
///       pass pair, index in the b1 push constants; cross-pass sum at u8.
///       MEASURED (2026-08-03, spin path 1080p spp=1, dxr core row,
///       default/stress/SM-lp): B70 1.39/1.56/1.56 vs mode 1's 2.51/1.67/2.20
///       — THE BEST DXR ARM ON ARC (−45%/−7%/−29%), and the thin dispatch
///       itself is finally cheap (dxr-rays 0.23-0.35; THE WORLD 0.54 vs mode
///       1's 2.87). NOT the default, two measured reasons: on the 4090 mode 1
///       still edges it (0.224 vs 0.243 default; at spp=16 mode 2's in-shader
///       loop crushes the pass pairs, 2.31 vs 3.53 — 2N RTPSO rebinds), and
///       on Arc the deferred kernel ITSELF now pays the codegen tax the CHS
///       used to (dxr-shade 1.124 vs the reference kernel's 0.603 for
///       strictly MORE work there) — so the wavefront keeps winning
///       (0.745 spin / 3.25 world vs D3's 1.39 / 4.73) and neither promotion
///       bar cleared. The identified follow-on: split the deferred kernel
///       (hit/sky, the wavefront's own leaf+sky lesson) or hunt its register
///       cliff — dxr-shade < reference is the target that would make DXR-3
///       the first DXR arm to threaten the wavefront on Arc.
///       THE CLIFF WAS HUNTED AND FOUND (2026-08-04, FR_WIDTH + FR_BALLAST
///       — CLAUDE.md's register-cliff campaign block): reference AND
///       dxr-shade both compile SIMD16 on the B70 (width is NOT the gap),
///       the reference kernel sits ~56-60 live floats below IGC's spill
///       knee (per-float cost breaks 3× there, N=160 = 2.6× baseline —
///       bracketing this kernel's 1.9×), and the FR_ABL×FR_WIDTH strip
///       sweep on THIS kernel reads norefl −0.49 / noglass −0.49 (on a
///       glassless scene) / nosec −0.78 with the singles summing to 1.22 —
///       a THRESHOLD, not a sum: shedding EITHER the reflection or the
///       glass DFS live state clears the same spill edge. So the follow-on
///       has a mechanism: split the reflection/glass lap (or hit/sky) out
///       of this kernel and its live state drops under the knee.
///       KNOWN REFUSAL: mode 3 + HEIGHTFIELD on Intel driver 32.0.101.8805
///       hangs the device (GBV silent, 4090 clean) — degraded to mode 1 with
///       a loud line below; re-test on a newer driver.
///       COMPARISON-TARGET NOTE: vendor_defaults has since made mode 2 the
///       Intel DXR default, so mode 2 is D3's Arc bar — and it was JUDGED
///       PER BINARY on 2026-08-04 (the merged tree, B70, same binary both
///       arms, ABBA reps repeating to ±0.01): D3 wins ONLY the default
///       scene (span 1.40 vs 1.80, −22%); D2 wins stress (1.40 vs 1.44),
///       SM-lp (1.94 vs 2.20), THE WORLD parked at the boot pose (4.93 vs
///       5.08), and every spp=16 point by 26-85% (wall 13.45 vs 17.08
///       default, 10.67 vs 19.74 stress — the per-sample pass pair vs the
///       in-shader loop). The thin half works AS DESIGNED everywhere
///       (dxr-rays 0.25/0.31/0.35/0.52 incl. the world's 2.70 → 0.52); the
///       deferred kernel is the WHOLE loss (dxr-shade 1.12/1.10/1.81/2.35
///       across default/stress/smlp/world vs the ~0.60 reference-kernel
///       class) — so NO promotion, mode 2 keeps the Intel default, and
///       `dxr-shade < reference` stays the bar. Note BOTH arms sit on
///       codegen cliffs (this binary's D2 default drew 1.80 where earlier
///       builds drew 1.41-2.46), so the per-scene ordering can flip with
///       any rebuild — re-judge on the binary in hand, never from tables.
/// Measured (--spin path 1080p spp=1, GPU frame span ms, default/stress/
/// SM-lp): B70 mode 0 9.05/5.30/6.75 -> mode 1 2.35/1.64/1.94 -> mode 2
/// 1.41/1.22/1.29; 4090 1.34/0.79/1.18 -> 0.26/0.25/0.34 -> 0.29/0.27/0.34.
/// (Mode-2 caveat: Arc's RT-stage codegen is build-unstable — the same mode-2
/// program later read 2.46 and 1.77 across semantically identical builds;
/// compare same-binary deltas only.)
/// Armed modes compile lib_6_5 and need RT tier 1.1 (the wavefront's own
/// floor); lesser hardware degrades to 0 with one loud line — the default is
/// a preference, never a requirement (NOT the --fsr4 shape). The RTPSO/SBT
/// layout is identical in every mode: unreached hit groups and misses stay
/// exported (identifier-only records, no cost). Set from main's parse via
/// `set_inline_mode` (the texture::set_aniso knob-before-anything idiom),
/// then RE-STORED in run_window right after `main::vendor_defaults` (which
/// moves an Intel session to 2 unless `dxr_inline_explicit` vetoes — every
/// interactive DxrGpu::new sits below that re-store; the headless harnesses
/// never run the policy and keep the parse-time store, so gates stay a pure
/// function of the command line). Legal values 0..=3: the CLI exits 2 on an
/// illegal value; the settings file warns instead and sets the explicit veto
/// on legal ones.
static INLINE_MODE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

pub fn set_inline_mode(n: u32) {
    INLINE_MODE.store(n, std::sync::atomic::Ordering::Relaxed);
}

/// FR_DXR_STACK=min|<bytes> — override the RTPSO's pipeline stack via
/// SetPipelineStackSize (the DispatchRays hunt: the driver's DEFAULT is the
/// spec's conservative formula over its own per-shader GetShaderStackSize
/// numbers plus whatever it pads on top — the work-graph 517-vs-82 MB
/// backing tell says Intel pads RT regimes generously). `min` recomputes
/// from those same driver numbers against OUR known call graph — mode 2 has
/// no TraceRay at all, so its true bound is the raygen frame alone, a fact
/// the spec formula cannot see; other modes take raygen + declared-depth ×
/// the largest callee frame. `<bytes>` is the raw bisection arm. SAFETY:
/// undershooting real usage is DEVICE REMOVAL by spec, not an error — `min`
/// derives from the driver's own numbers, never guesses; use `<bytes>` only
/// to bisect a suspected over-reservation. Loud on arm AND on an illegal
/// value (the FR_WIDE rule); default off = the driver default, untouched.
#[derive(Clone, Copy, PartialEq)]
enum StackOverride {
    Off,
    Min,
    Bytes(u64),
}

fn stack_override() -> StackOverride {
    static OV: std::sync::OnceLock<StackOverride> = std::sync::OnceLock::new();
    *OV.get_or_init(|| match std::env::var("FR_DXR_STACK") {
        Err(_) => StackOverride::Off,
        Ok(v) if v == "min" => {
            eprintln!(
                "gpu: FR_DXR_STACK=min — pipeline stack clamped to the \
                 call-graph bound (SetPipelineStackSize)"
            );
            StackOverride::Min
        }
        Ok(v) => match v.parse::<u64>() {
            Ok(n) if n > 0 => {
                eprintln!(
                    "gpu: FR_DXR_STACK={n} — raw pipeline stack override \
                     (undershooting real usage is device removal by spec)"
                );
                StackOverride::Bytes(n)
            }
            _ => {
                eprintln!("gpu: FR_DXR_STACK={v:?} illegal (min or bytes > 0) — off");
                StackOverride::Off
            }
        },
    })
}

pub(crate) fn dxr_inline_mode() -> u32 {
    INLINE_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

/// FR_DXR_LEAN=1 — the DEAD-EXPORTS CONTROL for the DispatchRays-regime
/// finding (finding 1's "zero TraceRay" arm). At `--dxr-inline 2` the RTPSO
/// still compiles and EXPORTS the fat `chs_shade` + misses + any-hits as
/// dead code (rt_dxr.hlsli's own note: "dead-but-exported and no ray can
/// reach them"), so "the raygen's 2× is its own hosting" and "the driver
/// provisions for the fat unreached exports" were observationally
/// confounded. Armed, the state object takes the RAYGEN-ONLY export list
/// (same DXIL blob — the front-end input is unchanged; the driver is merely
/// told only raygen participates), zero hit-group subobjects, declared
/// recursion depth 0 (legal: nothing traces — the micro-variant the
/// pipe_cfg comment deferred), and null miss/hit SBT ranges on the
/// DispatchRays desc. Image provably identical (nothing reachable changed).
/// If the mode-2 tax PERSISTS lean, the raygen's own hosting carries it —
/// the ballast dose-response's intercept attribution confirmed; if it
/// COLLAPSES, the tax was export provisioning and finding 1 gets rewritten.
/// Mode-2-only BY SOUNDNESS: any other mode has real TraceRay consumers
/// whose miss/hit lookups against a null table are undefined — asked-for
/// anywhere else it refuses loudly and stays off. Default off =
/// byte-identical descs (the FR_ABL class).
fn lean_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FR_DXR_LEAN") {
        Err(_) => false,
        Ok(v) if v == "1" => true, // announced at DxrGpu::new, where the mode is known
        Ok(v) => {
            eprintln!("gpu: FR_DXR_LEAN={v:?} unrecognized (legal: 1) — off");
            false
        }
    })
}

/// `--dxr-sbt 0|1|2|3` (default **0** = off, today's one-record SBT): the
/// many-record, MATERIAL-SORTED SBT ladder — the counterfactual the Intel
/// brief's Q4 promised ("the TSU sorts by shader RECORD… the guidance would
/// only flip if we grew many materially different hit records"). A DEV
/// MEASUREMENT lever, the `--sw-rays` class: no vendor policy, no settings
/// exposure, loud on every armed mode, off-state assembles the byte-identical
/// mode-0 source and the byte-identical instance descs.
///   1 — ALIAS records: the 8 shading classes (shadeclass.rs) get 8 export
///       ALIASES of the ONE `chs_shade` (ExportToRename — zero new compiles,
///       IDENTICAL code), the SBT grows to class-major
///       [HgShade_ck, HgHit, HgOcclude] × 8, and each TLAS instance carries
///       `InstanceContributionToHitGroupIndex = class * 3` (baked at upload —
///       the lever must be set BEFORE the SceneGpu core uploads, the
///       knob-before-anything rule; a core without the partition degrades
///       this to 0 with one loud line). Isolates the PURE record-sort /
///       repacking effect (± the sibling sub-chunk AABB overlap cost, the
///       structural price of instance-keyed sorting) — radiance-identical to
///       mode 0 by construction, gated bit-exact on default/stress.
///   2 — SPECIALIZED records: one extra DXIL library per class PRESENT in
///       the scene (k ≠ uber), compiled with `shadeclass::strip_defines(k)`
///       prepended — the SHADE_MAT_* seams in shade.hlsli fold that class's
///       provably-dead arms out (the STRIPS table is the soundness argument
///       FOR THE RECORD'S OWN SURFACE; `verify_strips` re-proved it on this
///       scene's materials at upload). KNOWN APPROXIMATION, measured: the
///       flattened lap loop also shades the record's CONTINUATION surfaces
///       (reflected/refracted children), which can be a DIFFERENT class —
///       a specialized parent then shades a child with the parent's strips
///       (a tex-opaque record's glass child loses its transmission). At the
///       SM-lp glassware pose the mode-2-vs-mode-1 drift reads max |d|
///       9.61e-3 (~3% of channels) against the 5.96e-8 fp-noise floor —
///       real, bounded under T2's statistical gates, and the documented
///       price of an occupancy INSTRUMENT. Mode 3 closes it BY CONSTRUCTION
///       (every surface shades in its own class's record) — an unplanned
///       second reason that rung exists.
///       Each library exports exactly `{chs_shade_ck ← chs_shade}`; absent
///       classes + uber stay lib-0 aliases (absent records are never
///       addressed, uber IS the unstripped fallback). Adds primary-hit
///       register/occupancy relief on top of rung 1's sort keys — and it is
///       where NVIDIA joins the ladder (genuinely different libraries cannot
///       dedupe; the recorded finding is that it folds rung 1's renames).
///       NOT bit-comparable to modes 0/1 (the stripped code is provably dead
///       per class, but DXC reschedules the survivors — fp association
///       drift, tracked by report, gated statistically); rebuild-vs-rebuild
///       IS bit-exact and gated (T1d's determinism arm).
///   3 — RECURSIVE class dispatch: rung 2's records, dispatched the way the
///       TSU's sorter is fed in production titles — every reflection/glass
///       continuation is a REAL `TraceRay` at RayContribution 0, so the hit
///       instance's `class * 3` contribution lands it in the hit surface's
///       OWN specialized closest-hit (the routing is SBT arithmetic, zero
///       shader-side dispatch). The flattened DFS lap loop collapses to one
///       iteration; the hardware ray stack replaces the stash
///       (shade.hlsli's DXR_SBT_RECURSE arm — Beer–Lambert multiplies the
///       RETURNED radiance, the CPU's own association; ind_s becomes the
///       literal `rtput * child_color`; rng round-trips through the payload
///       so the stream keeps the CPU DFS draw order; depth + cone ride the
///       repurposed sp lanes, no payload growth). The HYBRID half: shadow/AO
///       occlusion stays inline RayQuery (rt.hlsli rides along, lib_6_5 +
///       tier 1.1 required — lesser devices degrade to 2), which caps
///       MaxTraceRecursionDepth at 5 (see the pipe_cfg derivation). Misses
///       of continuation rays hit the miss_rec SENTINEL (index 3, t = INF,
///       no sky) because a reflection miss needs the PARENT lobe's MIS
///       weight — the parent keeps its own miss arms. Arms only at
///       --dxr-inline 0 (inline modes have no TraceRay continuations to
///       redirect — asked-for anyway degrades to 2, loudly). A
///       mode-0-descendant measurement arm, never a default candidate.
/// Composition: the contribution only matters where hit-group records
/// DISPATCH — every mode-0 TraceRay and the inline-0/1 primary; under
/// `--dxr-inline 2` (zero TraceRay) the arm is VACUOUS, and under
/// `--dxr-inline 3` nearly so (only the thin bare-hit record dispatches —
/// the SHADING records this ladder sorts never run) — both say so loudly
/// (kept runnable for A/B matrix hygiene). Set from main's parse via
/// `set_sbt_mode` (the set_inline_mode idiom; headless keeps the parse-time
/// store).
static SBT_MODE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn set_sbt_mode(n: u32) {
    SBT_MODE.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn dxr_sbt_mode() -> u32 {
    SBT_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

/// `--dxr-inline 3`'s deferred half: the cs_6_5 shade kernel plus the two
/// pass-pair buffers — hit records at u7 and the cross-pass color sum at u8
/// (the wavefront's qleaf/qsky registers, undeclared in every other DXR
/// unit, so no root-signature change — the cloud-cache u5/u6 precedent).
struct Mode3 {
    pso_shade: ID3D12PipelineState,
    hitrec: ID3D12Resource,
    csum: ID3D12Resource,
}

pub struct DxrGpu {
    root_sig: ID3D12RootSignature,
    state: ID3D12StateObject,
    sbt: d3d12::UploadBuffer,
    /// Hit-table record count (3 off-lever; classes × 3 on the --dxr-sbt
    /// alias arm) — the fill and DispatchRays read this ONE snapshot.
    sbt_hit_records: u32,
    /// FR_DXR_LEAN snapshot (mode-2 raygen-only RTPSO): DispatchRays carries
    /// NULL miss/hit table ranges — the tables were never filled and the
    /// state object holds nothing they could name.
    lean: bool,
    /// The armed --dxr-sbt rung this pipeline was BUILT at (post-degrade),
    /// how many pairwise-distinct per-class shader identifiers the driver
    /// actually minted, and how many the rung REQUIRES distinct (mode 1: all
    /// 8 — the rename experiment; mode 2: the specialized set ∪ uber —
    /// absent classes alias uber and legitimately dedupe on drivers that
    /// fold renames). distinct != required is the dedupe finding. All read
    /// by --check-dxr's construction audit.
    pub sbt_mode: u32,
    pub sbt_distinct_idents: u32,
    pub sbt_required_idents: u32,
    pso_resolve: ID3D12PipelineState,
    /// The cloud-cache fill kernels + their buffers (u5/u6), armed per the
    /// snapshotted levers. `None` when the respective cache is off. bound before
    /// DispatchRays and left resident for the whole ray dispatch.
    pso_sky_lod: Option<ID3D12PipelineState>,
    pso_cloud_shadow: Option<ID3D12PipelineState>,
    cloud_lod: ID3D12Resource,
    cloud_shadow: ID3D12Resource,
    /// Snapshotted cloud-cache levers (see the assembly comment) + the scene
    /// AABB the slab-space shadow grid spans.
    sky_lod_k: u32,
    cloud_shadow_n: u32,
    scene_aabb: ([f32; 3], [f32; 3]),
    /// The shared scene core — the SAME Rc the wavefront tracer holds (cached
    /// in GpuContext), so a session running both pays the scene VRAM once.
    pub scene: std::rc::Rc<SceneGpu>,
    /// Per-pixel planes, CPU-layout parity (accum = 3 f32/px, tbuf = f32/px,
    /// info = u32/px) — the same readback-compare shape as the compute tracer.
    pub accum: ID3D12Resource,
    pub tbuf: ID3D12Resource,
    pub info: ID3D12Resource,
    /// The G-buffer pack's CORE half at RP_GBUF: `GBufCore`, 16 B/px when the
    /// session composes with an upscaler (`gbuf_full`), a stride-sized dummy
    /// otherwise — FLAG_GBUF is clear then, but root-descriptor UAVs have no
    /// bounds check, so the plain-mode dummy is memory safety, not an
    /// optimization (the trace.rs precedent).
    pub gbuf: ID3D12Resource,
    /// The pack's guide/signal half at RP_GBUF_EXT (`GBufExt`, 72 B/px),
    /// written only under FLAG_GBUF_EXT. Same allocation rule as `gbuf`.
    pub gbuf_ext: ID3D12Resource,
    /// Test hook — the twin of `TraceGpu::force_gbuf_ext`; see it for why the
    /// pack gates need it (their consumer is a CPU readback, and they trace
    /// before any feed is wired).
    force_gbuf_ext: std::cell::Cell<bool>,
    /// The frame's sway clock, stashed by `write_cb` (which sees FrameParams)
    /// for `record_frame`/`bind_common` (which don't): Some = rebuild + bind
    /// the animated-TLAS ring slot, None = the static rest-pose TLAS. Only
    /// meaningful when `SceneGpu::sway` exists (--foliage-sway with leaf
    /// cells) — the pair is checked together at both consumers.
    sway_t: std::cell::Cell<Option<f32>>,
    /// The frame's sway-MV clock pair (`trace::sway_mv_pair`), stashed by
    /// `write_cb` beside `sway_t` for `record_frame`'s dmv-ring fill — one
    /// predicate site, so the CB flag and the slot's rows cannot disagree.
    /// None on every camera-only frame (headless gates, frozen stills).
    sway_mv_t: std::cell::Cell<Option<(f32, f32)>>,
    /// Whether SWAY_MV compiled into this library (`trace::sway_defs` — ring
    /// armed; DXR has no --sw-rays carve-out). The arming gate.
    sway_mv_on: bool,
    /// Mode-3 (thin CHS) state — Some iff the DEGRADED-LOCAL inline mode was
    /// 3 at construction, i.e. the library really compiled the thin raygen.
    /// `record_frame` branches on THIS, never on `dxr_inline_mode()` (the
    /// tier-1.0 degrade path and the check harness's between-builds static
    /// flips would otherwise run the pass loop against a non-thin library).
    mode3: Option<Mode3>,
    /// FR_WIDTH: the DXR pipeline's width-report sink (slot 0 = raygen,
    /// slot 1 = cs_dxr_shade). Some iff the lever was armed at construction;
    /// bound at the u3 root param — which this pipeline otherwise leaves
    /// UNBOUND (the wavefront's counters register) — so unarmed sessions
    /// bind exactly what they bind today. Read back by the spin/check width
    /// prints via `width_buf()`.
    width_buf: Option<ID3D12Resource>,
    /// The frame's spp, stashed by `write_cb` (which sees FrameParams) for
    /// `record_frame`'s mode-3 pass-pair loop (which doesn't) — the `sway_t`
    /// idiom; every caller including all --check-dxr sites writes the CB
    /// first.
    spp: std::cell::Cell<u32>,
    gbuf_full: bool,
    /// RGBA16F resolve target; rests in PIXEL_SHADER_RESOURCE between frames
    /// (the tonemap PS reads it via SRV_SLOT_DXR).
    pub hdr: ID3D12Resource,
    uav_heap: ID3D12DescriptorHeap,
    /// GPU handle of the RP_SCENE_TEX table (heap slot TEX_HEAP_BASE).
    tex_table: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// Upscaler feed kernels (compiled only when `gbuf_full`; every kind so
    /// --check-dxr can rewire between them like --check-gpu does) and the
    /// wired planes record_feed barriers over.
    pso_feed_xess: Option<ID3D12PipelineState>,
    pso_feed_rr: Option<ID3D12PipelineState>,
    pso_feed_fsr_rr: Option<ID3D12PipelineState>,
    /// One entry per wired engine; the index IS its descriptor set (see
    /// trace::FEED_SETS). Normally one — several under --quinlight.
    feed: Vec<(trace::FeedKind, Vec<ID3D12Resource>)>,
    /// For the per-set descriptor-table handles record_feed computes.
    device: ID3D12Device,
    frame_cb: d3d12::UploadBuffer,
    cb_base: FrameCb,
    pub rw: u32,
    pub rh: u32,
}

impl DxrGpu {
    pub fn new(
        device: &ID3D12Device,
        dxc: &Dxc,
        scene: &Scene,
        scene_gpu: std::rc::Rc<SceneGpu>,
        rw: u32,
        rh: u32,
        gbuf_full: bool,
        debug: bool,
    ) -> Result<Self> {
        require_caps(device)?;
        trace::abl_announce();
        let device5: ID3D12Device5 =
            device.cast().map_err(|e| format!("ID3D12Device5: {e}"))?;
        let root_sig = trace::create_root_signature(device)?;

        // --dxr-inline (see dxr_inline_mode): armed modes compile RayQuery
        // into the library, which needs the wavefront's caps floor, not this
        // pipeline's — gate here so a tier-1.0 box degrades to the TraceRay
        // path with one loud line instead of a DXC error. The cross-vendor
        // default (1) stays QUIET; 0 and 2 print — and on Intel, 2 is an
        // ARRIVAL (the vendor default), not a departure, so its line names
        // both routes and the opt-out rather than claiming "the default".
        let inline_mode = {
            let m = dxr_inline_mode();
            if m > 0 {
                let caps = trace::query_caps(device)?;
                if caps.rt_tier < D3D12_RAYTRACING_TIER_1_1.0 || caps.shader_model < 0x65 {
                    eprintln!(
                        "dxr: --dxr-inline {m} unavailable — inline RayQuery needs RT tier 1.1 \
                         + SM 6.5 (device: tier {}, SM 0x{:x}); running the all-TraceRay \
                         pipeline",
                        caps.rt_tier, caps.shader_model
                    );
                    0
                } else {
                    if m == 2 {
                        eprintln!(
                            "dxr: --dxr-inline 2 — everything inline in raygen (DispatchRays \
                             as a bare launch grid; cross-vendor default 1, Intel sessions \
                             default here — --dxr-inline 1 opts out)"
                        );
                    }
                    if m == 3 {
                        eprintln!(
                            "dxr: --dxr-inline 3 — thin closest-hit + deferred compute shade \
                             (bare-hit DispatchRays writes records, cs_dxr_shade shades; the \
                             default is 1, inline secondaries)"
                        );
                    }
                    m
                }
            } else {
                eprintln!(
                    "dxr: --dxr-inline 0 — all-TraceRay dispatch (the pre-lever pipeline; \
                     the cross-vendor default is 1, inline RayQuery secondaries — Intel \
                     defaults to 2)"
                );
                0
            }
        };
        // MEASURED REFUSAL, the FR_WORKGRAPH-on-Intel class (trace.rs:4580's
        // war story, same shape): mode 3 + a HEIGHTFIELD-armed scene hangs the
        // Arc driver — DXGI_ERROR_DEVICE_HUNG at the --check-dxr spp gate,
        // deterministic, with the debug layer AND GPU-based validation silent
        // (Intel driver 32.0.101.8805). The IDENTICAL suite passes on the
        // 4090 (all four spp probes, rel-t <= 2.1e-4), and Arc itself runs
        // modes 0/1 + relief AND mode 3 without relief clean — so this is the
        // driver's (non-opaque-anyhit RTPSO x thin-raygen pass pairs) combo,
        // not the shader. Degrade to mode 1 (the proven-with-relief default),
        // never 0; keyed on the PICKED adapter (a fact), the vendor_defaults
        // rule. Re-test on a newer driver and delete this arm if it passes.
        let inline_mode = if inline_mode == 3
            && !trace::height_defs(scene).is_empty()
            && super::adapter::picked_vendor() == super::adapter::Vendor::Intel
        {
            eprintln!(
                "dxr: --dxr-inline 3 + --heightfield hangs this Intel driver \
                 (DXGI_ERROR_DEVICE_HUNG, 32.0.101.8805; GBV silent, 4090 clean) — \
                 degrading to --dxr-inline 1 for this session"
            );
            1
        } else {
            inline_mode
        };

        // --dxr-sbt snapshot (see the SBT_MODE doc). Degrades: to 0 when the
        // shared core carries no class partition (the lever must precede the
        // upload — --no-blas-split, or a core cached before arming); mode 3
        // to 2 when the inline mode isn't 0 (recursive dispatch is the
        // all-TraceRay pipeline's descendant — inline modes have no TraceRay
        // continuations to redirect) or the device lacks tier 1.1/SM 6.5
        // (the hybrid's inline occlusion needs RayQuery in a lib target).
        // The --dxr-inline 2 composition is VACUOUS (zero TraceRay ⇒ no
        // record ever dispatches), and --dxr-inline 3 nearly so (only the
        // thin bare-hit record dispatches — the SHADING records this ladder
        // sorts never run; per-class HgHit copies share one identifier, one
        // sort key) — both runnable for matrix hygiene, said loudly.
        // Snapshotted once like inline_mode — reading the LOCAL, post-degrade
        // binding (the heightfield-Intel arm above can move 3 → 1) — so a
        // mid-process A/B can never desync the RTPSO from its SBT.
        let sbt_mode = {
            let m = dxr_sbt_mode();
            if m == 0 {
                0
            } else if scene_gpu.sbt_class.is_none() {
                eprintln!(
                    "dxr-sbt: mode {m} armed but the scene core carries no class partition \
                     (--no-blas-split, or the core uploaded before the lever) — running the \
                     one-record SBT"
                );
                0
            } else {
                let mut built = m.min(3);
                if built == 3 && inline_mode != 0 {
                    eprintln!(
                        "dxr-sbt: mode 3 (recursive class dispatch) descends from the \
                         all-TraceRay pipeline — --dxr-inline {inline_mode} has no TraceRay \
                         continuations to redirect; running mode 2 (specialized records). \
                         Pass --dxr-inline 0 to arm it"
                    );
                    built = 2;
                }
                if built == 3 {
                    let caps = trace::query_caps(device)?;
                    if caps.rt_tier < D3D12_RAYTRACING_TIER_1_1.0 || caps.shader_model < 0x65 {
                        eprintln!(
                            "dxr-sbt: mode 3's inline-occlusion hybrid needs RT tier 1.1 + \
                             SM 6.5 (device: tier {}, SM 0x{:x}) — running mode 2",
                            caps.rt_tier, caps.shader_model
                        );
                        built = 2;
                    }
                }
                if inline_mode == 2 {
                    eprintln!(
                        "dxr-sbt: NOTE --dxr-inline 2 dispatches no hit-group records at all \
                         — the sorted SBT is VACUOUS in this composition (kept runnable for \
                         A/B matrix hygiene)"
                    );
                } else if inline_mode == 3 {
                    eprintln!(
                        "dxr-sbt: NOTE --dxr-inline 3 dispatches only the thin bare-hit \
                         record (shading is the deferred compute pass) — the sorted SHADING \
                         records never run, so the ladder is VACUOUS in this composition \
                         (kept runnable for A/B matrix hygiene)"
                    );
                }
                if built == 1 {
                    eprintln!(
                        "dxr-sbt: mode 1 — {} class records (alias arm: 8 renames of one \
                         chs_shade; identical code, distinct sort keys)",
                        crate::shadeclass::N_CLASSES * 3
                    );
                } else {
                    let live = scene_gpu.sbt_class.as_ref().map_or(0, |c| {
                        (0..crate::shadeclass::N_CLASSES)
                            .filter(|&k| {
                                k != crate::shadeclass::CK_UBER as usize && c.histo[k] > 0
                            })
                            .count()
                    });
                    eprintln!(
                        "dxr-sbt: mode {built} — {} class records, {live} specialized \
                         librar{} + the uber fallback (strip-defines folded per class{})",
                        crate::shadeclass::N_CLASSES * 3,
                        if live == 1 { "y" } else { "ies" },
                        if built == 3 {
                            "; continuations TraceRay into their class's CHS, occlusion \
                             inline, MaxTraceRecursionDepth 5"
                        } else {
                            ""
                        }
                    );
                }
                built
            }
        };

        // Alpha-masked and height-carrying scenes compile the ah_* any-hit
        // shaders + non-opaque ray flags in (trace.rs::alpha_defs /
        // height_defs — the same per-scene predicates that drop OPAQUE from
        // the BLAS); scenes with neither compile verbatim.
        // Derived from the three per-scene predicates rather than re-deriving
        // `scene.any_*`, so this arm's AS flag and its any-hit shaders are one
        // decision — and an FR_ABL neutralization moves both together.
        let non_opaque = trace::non_opaque(scene);
        // The cbuffer's --spp jitter-table size, injected like alpha_defs.
        let sd = trace::spp_defs();
        let sd = sd.as_str();
        // The detail strength knobs (shade.hlsli's DETAIL_STR seams).
        let dd = trace::detail_defs();
        let dd = dd.as_str();
        let sway_def = trace::sway_defs(&scene_gpu);
        let defs = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}",
            // SCENE_EMPTY is INERT in this pipeline and is carried only so the
            // two arms' define sets stay comparable: the guards it gates live in
            // frustum.hlsli and rt_sw.hlsli, neither of which this library
            // pastes, and DXR rays go through the TLAS (an empty TLAS is the
            // driver's problem, not ours). It becomes load-bearing the moment
            // this pipeline ever pastes a software-BVH consumer.
            trace::empty_defs(scene),
            trace::alpha_defs(scene),
            trace::height_defs(scene),
            trace::trans_defs(scene),
            trace::blas_defs(),
            // SWAY_MV: the prev-pose MV correction in gbuf_write_hit, off the
            // uploaded ring's existence (trace::sway_defs — no --sw-rays
            // carve-out here, DXR never software-rays).
            sway_def,
            // FR_ABL, shared with the wavefront — without it every cloud cost
            // attribution was wavefront-only and silently incomparable here.
            trace::abl_defs()
        );
        // The cloud shading caches, snapshotted at construction like TraceGpu:
        // the library is compiled against these, the buffers sized against them,
        // and record_frame's fills / write_cb's grid all read the fields, so a
        // mid-process A/B (the --check-dxr on/off gate flips the static between
        // two DxrGpu builds) can never desync a shader from its fill dispatch.
        let sky_lod_k = trace::sky_lod();
        let cloud_shadow_n = trace::cloud_shadow_n();
        // The two cache defines arm trace_common's cached cloud_sun_transmittance
        // (u6) + skylod.hlsli's sky_radiance_lod (u5) for EVERY shade path in the
        // library — parity inherited, not re-ported. u5/u6 are unbound in the DXR
        // root signature today (the wavefront's tile queues), so binding dedicated
        // buffers there needs no root-signature change.
        let cloud_defs = format!(
            "#define CLOUD_SHADOW_N {cloud_shadow_n}\n#define SKY_LOD {sky_lod_k}\n#define SKY_LOD_LOG {}",
            sky_lod_k.trailing_zeros()
        );
        // Mode 0 assembles EXACTLY the shipping sequence (the lever's
        // off-state is byte-identical source, not merely equivalent); armed
        // modes prepend the define and paste rt.hlsli's RayQuery primitives
        // ahead of rt_dxr.hlsli, whose TraceRay flavors + tlas/HitInfo
        // compile out under DXR_INLINE_SEC. The two cache defines ride in every
        // mode (the shipping sequence now carries them; mode 0 stays
        // byte-identical ACROSS inline modes, the lever's actual contract).
        // --dxr-sbt 3 (recurse — arms only at inline 0, the snapshot's
        // degrade) is the HYBRID paste: rt.hlsli rides along for INLINE
        // occlusion (shadow/AO rays never re-enter the pipeline, which is
        // what caps the recursion at 5) while DXR_SBT_RECURSE reshapes
        // rt_dxr.hlsli (tlas/HitInfo/TraceRay primitives out, trace_shade in)
        // and shade_split's continuations (recursive TraceRay in place of the
        // lap loop). An unarmed sbt mode leaves every push identical — the
        // mode-0 byte-identity contract holds across the WHOLE ladder.
        let recurse = sbt_mode == 3;
        // FR_DXR_LEAN — the dead-exports control (doc at lean_on). Qualifies
        // ONLY at mode 2 with the sbt ladder off: every other config has real
        // TraceRay consumers (or sorted records under audit) that a
        // raygen-only state object would break — a miss/hit lookup against a
        // null table is undefined, so the guard is soundness, not tidiness.
        // The DXIL blob is deliberately UNCHANGED (same source, same
        // compile): the lever moves only what the state object EXPORTS, so a
        // cost delta is attributable to export provisioning and nothing else.
        let lean = if lean_on() {
            if inline_mode == 2 && !recurse && sbt_mode == 0 {
                eprintln!(
                    "dxr: FR_DXR_LEAN=1 — raygen-only RTPSO (exports: raygen; \
                     recursion 0; null miss/hit SBT ranges) — the dead-exports \
                     control"
                );
                true
            } else {
                eprintln!(
                    "gpu: FR_DXR_LEAN=1 needs --dxr-inline 2 with --dxr-sbt 0 \
                     (this session is inline {inline_mode}, sbt {sbt_mode}) — off"
                );
                false
            }
        } else {
            false
        };
        let inline_def = format!("#define DXR_INLINE_SEC {inline_mode}");
        let mut parts = vec![defs.as_str(), sd, dd, cloud_defs.as_str()];
        if inline_mode > 0 {
            parts.push(inline_def.as_str());
        }
        if recurse {
            parts.push("#define DXR_SBT_RECURSE 1");
        }
        // FR_WIDTH: the raygen wave-width report (dxr_width[0]) — armed only
        // where the library compiles at lib_6_5 (the inline modes' floor;
        // wave ops in a 6_3 library are off the table, and mode 0's raygen
        // is not the lottery victim anyway). The deferred-shade unit below
        // carries its own WIDTH_PROBE define.
        if trace::width_probe_on() && (inline_mode > 0 || recurse) {
            parts.push("#define WIDTH_PROBE_RAYGEN 1");
        }
        // FR_BALLAST=dxr:N — the raygen ballast (the knee-vs-knee host
        // comparison; reference.hlsl's liveness argument, mirrored in
        // dxr.hlsl's mode-2 arm). The blocks' compound guard already
        // confines them to DXR_INLINE_SEC == 2, but pushing the define into
        // a mode whose arm compiles out would leave the SEED live and the
        // update dead — a ballast that "measures" a flat curve — so any
        // other mode refuses loudly instead (the FR_WIDE rule).
        let bd = trace::ballast_dxr_defs();
        if !bd.is_empty() {
            if inline_mode == 2 {
                parts.push(bd.as_str());
            } else {
                eprintln!(
                    "gpu: FR_BALLAST=dxr:N needs --dxr-inline 2 (this session is \
                     mode {inline_mode}) — off"
                );
            }
        }
        parts.push(trace::TRACE_COMMON_HLSLI);
        // skylod.hlsli after trace_common (needs sky_compose/sky_backdrop/rw); no
        // SKY_UNIT — this unit pastes no queues.hlsli, so u5 is free anyway.
        parts.push(trace::SKYLOD_HLSLI);
        if inline_mode > 0 || recurse {
            parts.push(trace::RT_HLSLI);
        }
        parts.extend([
            RT_DXR_HLSLI,
            trace::RIPPLE_HLSLI,
            trace::SHADE_HLSLI,
            DXR_HLSL,
        ]);
        let lib_src = parts.join("\n");
        // rt.hlsli's RayQuery in a library target needs SM 6.5 — the inline
        // modes' floor, and mode 3's (the snapshot degraded to 2 if the
        // device lacks it).
        let lib_target = if inline_mode > 0 || recurse { "lib_6_5" } else { "lib_6_3" };
        let dxil = dxc.compile(&lib_src, "", lib_target, "dxr library", debug)?;
        // --dxr-sbt 2: one SPECIALIZED library per PRESENT class (k ≠ uber) —
        // shadeclass::strip_defines(k) prepended to the IDENTICAL source, so
        // that class's chs_shade compiles with its provably-dead arms folded
        // out at the SHADE_MAT_* seams (the STRIPS table is the soundness
        // argument; verify_strips re-proved every obligation on this scene's
        // own materials at upload). The k enumeration is 0..N in order —
        // deterministic library set, which the T1d determinism arm gates
        // bit-exact. Distinct `what` per class: FR_DUMP_HLSL keys its dump
        // files on it, and eight libraries sharing "dxr library" would
        // overwrite each other's dumps.
        let spec_classes: Vec<usize> = if sbt_mode >= 2 {
            let histo = &scene_gpu.sbt_class.as_ref().unwrap().histo;
            (0..crate::shadeclass::N_CLASSES)
                .filter(|&k| k != crate::shadeclass::CK_UBER as usize && histo[k] > 0)
                .collect()
        } else {
            Vec::new()
        };
        let class_dxil: Vec<Vec<u8>> = spec_classes
            .iter()
            .map(|&k| {
                let src_k = format!("{}{}", crate::shadeclass::strip_defines(k as u8), lib_src);
                dxc.compile(
                    &src_k,
                    "",
                    lib_target,
                    &format!("dxr library c{k} ({})", crate::shadeclass::NAMES[k]),
                    debug,
                )
            })
            .collect::<Result<_>>()?;
        let resolve_src = [sd, trace::TRACE_COMMON_HLSLI, trace::RESOLVE_HLSL].join("\n");
        let pso_resolve = trace::compute_pso(
            device,
            &root_sig,
            &dxc.compile(&resolve_src, "cs_resolve", "cs_6_3", "dxr resolve", debug)?,
            "dxr resolve",
        )?;
        // The cloud-cache FILL kernels (cs_sky_lod / cs_cloud_shadow), from the
        // SHARED sky_unit_src so they cannot drift from the wavefront's — plain
        // compute (no rays), so cs_6_3 like the resolve/feed kernels. Compiled
        // only when the respective cache is armed; None otherwise.
        let (pso_sky_lod, pso_cloud_shadow) = if sky_lod_k > 1 || cloud_shadow_n > 0 {
            let sky_unit = trace::sky_unit_src(sky_lod_k, cloud_shadow_n);
            let mk = |entry: &str, name: &str| -> Result<Option<ID3D12PipelineState>> {
                Ok(Some(trace::compute_pso(
                    device,
                    &root_sig,
                    &dxc.compile(&sky_unit, entry, "cs_6_3", name, debug)?,
                    name,
                )?))
            };
            (
                if sky_lod_k > 1 { mk("cs_sky_lod", "dxr sky_lod")? } else { None },
                if cloud_shadow_n > 0 { mk("cs_cloud_shadow", "dxr cloud_shadow")? } else { None },
            )
        } else {
            (None, None)
        };
        // Upscaler sessions: the same feed kernels the wavefront runs, at
        // this pipeline's cs_6_3 cap floor (feed.hlsl needs nothing newer).
        let (pso_feed_xess, pso_feed_rr, pso_feed_fsr_rr) = if gbuf_full {
            // abl_defs FIRST so a feed ablation is not silently inert — the
            // library's `defs` above already carries it, but this unit did
            // not, so an `FR_ABL=nopack` probe under --dxr compared identical
            // code against itself (feed.hlsl consumes ABL_NOPACK; trace.rs's
            // feed_src learned the same lesson — "an ablation that cannot
            // reach its target answers confidently"). Pushed CONDITIONALLY,
            // unlike trace.rs's unconditional first element: this unit's
            // unarmed baseline has no leading blank line, and an empty first
            // segment + join("\n") would prepend one — the unarmed source
            // stays byte-identical. Armed, both pipelines' feed units
            // assemble identical leading text (abl_defs ends in '\n').
            let feed_abl = trace::abl_defs();
            let mut feed_parts: Vec<&str> = Vec::new();
            if !feed_abl.is_empty() {
                feed_parts.push(feed_abl.as_str());
            }
            feed_parts.extend([
                sd,
                trace::TRACE_COMMON_HLSLI,
                trace::FSR_WIRE_HLSLI,
                trace::FEED_HLSL,
            ]);
            let feed_src = feed_parts.join("\n");
            let pso = |entry: &str, name: &str| -> Result<ID3D12PipelineState> {
                trace::compute_pso(
                    device,
                    &root_sig,
                    &dxc.compile(&feed_src, entry, "cs_6_3", name, debug)?,
                    name,
                )
            };
            (
                Some(pso("cs_feed_xess", "dxr feed_xess")?),
                Some(pso("cs_feed_rr", "dxr feed_rr")?),
                Some(pso("cs_feed_fsr_rr", "dxr feed_fsr_rr")?),
            )
        } else {
            (None, None, None)
        };
        // --dxr-inline 3: the deferred-shade kernel. cs_6_5 — it fires
        // rt.hlsli's inline RayQuery secondaries, the same SM 6.5 / tier 1.1
        // floor the armed-mode gate above already enforced. Sources mirror
        // the library's minus its DispatchRays halves: NO rt_dxr.hlsli and NO
        // dxr.hlsl — this unit must never contain a TraceRay (cargo test pins
        // the kernel source for one).
        let pso_dxr_shade = if inline_mode == 3 {
            // FR_WIDTH pushed conditionally (the trace.rs unit rule): arms
            // dxr_shade.hlsl's guarded u3 view + width write (slot 1).
            let wd = trace::width_defs();
            let mut shade_parts: Vec<&str> = Vec::new();
            if !wd.is_empty() {
                shade_parts.push(wd.as_str());
            }
            shade_parts.extend([
                defs.as_str(),
                sd,
                dd,
                cloud_defs.as_str(),
                inline_def.as_str(),
                trace::TRACE_COMMON_HLSLI,
                trace::SKYLOD_HLSLI,
                trace::RT_HLSLI,
                trace::RIPPLE_HLSLI,
                trace::SHADE_HLSLI,
                DXR_SHADE_HLSL,
            ]);
            let shade_src = shade_parts.join("\n");
            Some(trace::compute_pso(
                device,
                &root_sig,
                &dxc.compile(&shade_src, "cs_dxr_shade", "cs_6_5", "dxr deferred shade", debug)?,
                "dxr deferred shade",
            )?)
        } else {
            None
        };

        // --- RTPSO. Every pDesc payload (and every name string) lives in a
        // local that outlives CreateStateObject.
        let wname = |s: &str| -> Vec<u16> { s.encode_utf16().chain(std::iter::once(0)).collect() };
        let hg_shade_name = wname("HgShade");
        let hg_hit_name = wname("HgHit");
        let hg_occlude_name = wname("HgOcclude");
        let chs_shade_name = wname("chs_shade");
        let chs_hit_name = wname("chs_hit");
        let ah_shade_name = wname("ah_shade");
        let ah_hit_name = wname("ah_hit");
        let ah_shadow_name = wname("ah_shadow");

        // --dxr-sbt name storage: per-class hit-group + alias export names.
        // Vec<Vec<u16>>: the OUTER vec may move, the inner heap buffers the
        // PCWSTRs point into do not — but the desc vecs below are still
        // fully built before any pointer is taken (the :400-402 rule).
        let n_classes = crate::shadeclass::N_CLASSES;
        let hg_class_names: Vec<Vec<u16>> =
            (0..n_classes).map(|k| wname(&format!("HgShade_c{k}"))).collect();
        let chs_alias_names: Vec<Vec<u16>> =
            (0..n_classes).map(|k| wname(&format!("chs_shade_c{k}"))).collect();

        // Alias arm: the library takes an EXPLICIT export list — every base
        // [shader(...)] entry re-exported under its own name (NumExports 0
        // means "export all", so naming any means naming ALL that the SBT or
        // the hit groups will look up), plus 8 ExportToRename ALIASES of the
        // one chs_shade. 8 renames of one function = 8 distinct shader
        // identifiers = the TSU's sort keys, zero new compiles — the whole
        // point of ladder rung 1. Off (sbt_mode 0): NumExports stays 0, the
        // pre-lever library subobject verbatim.
        let base_export_names: Vec<Vec<u16>> = {
            let mut v: Vec<&str> =
                vec!["raygen", "chs_shade", "chs_hit", "miss_radiance", "miss_shadow", "miss_hit"];
            if non_opaque {
                v.extend(["ah_shade", "ah_hit", "ah_shadow"]);
            }
            // Mode 3's continuation-miss sentinel (miss index 3) — the shader
            // only exists in source under DXR_SBT_RECURSE, so exporting it in
            // any other mode would fail CreateStateObject.
            if recurse {
                v.push("miss_rec");
            }
            v.into_iter().map(wname).collect()
        };
        // FR_DXR_LEAN: the raygen-only export list needs a name that outlives
        // CreateStateObject (the desc stores a raw pointer — the RTPSO
        // lifetime rule).
        let raygen_export_name = wname("raygen");
        let mut exports: Vec<D3D12_EXPORT_DESC> = Vec::new();
        if lean {
            // Same DXIL blob; the state object is told ONLY raygen
            // participates (sbt_mode is 0 under the arming guard, so this is
            // the exports Vec's only writer).
            exports.push(D3D12_EXPORT_DESC {
                Name: PCWSTR(raygen_export_name.as_ptr()),
                ExportToRename: PCWSTR::null(),
                Flags: D3D12_EXPORT_FLAG_NONE,
            });
        }
        if sbt_mode >= 1 {
            exports.reserve_exact(base_export_names.len() + n_classes);
            for n in &base_export_names {
                exports.push(D3D12_EXPORT_DESC {
                    Name: PCWSTR(n.as_ptr()),
                    ExportToRename: PCWSTR::null(),
                    Flags: D3D12_EXPORT_FLAG_NONE,
                });
            }
            // Mode 1: all 8 chs_shade_ck alias lib 0's chs_shade. Mode 2: a
            // class with its own specialized library exports its name THERE
            // (exported names are unique state-object-wide), so lib 0
            // aliases only uber + the absent classes.
            for k in 0..n_classes {
                if spec_classes.contains(&k) {
                    continue;
                }
                exports.push(D3D12_EXPORT_DESC {
                    Name: PCWSTR(chs_alias_names[k].as_ptr()),
                    ExportToRename: PCWSTR(chs_shade_name.as_ptr()),
                    Flags: D3D12_EXPORT_FLAG_NONE,
                });
            }
        }
        // Per-specialized-library export lists (mode 2): exactly ONE entry —
        // chs_shade_ck renamed from that library's own (stripped) chs_shade.
        // Nothing else exports: raygen/misses/chs_hit/ah_* must exist ONCE in
        // the state object and resolve from lib 0's pool (hit groups compose
        // imports across libraries). Built FULLY before any lib desc takes a
        // pointer; the outer Vec never grows after collect, so the inline
        // [_; 1] buffers are stable.
        let class_exports: Vec<[D3D12_EXPORT_DESC; 1]> = spec_classes
            .iter()
            .map(|&k| {
                [D3D12_EXPORT_DESC {
                    Name: PCWSTR(chs_alias_names[k].as_ptr()),
                    ExportToRename: PCWSTR(chs_shade_name.as_ptr()),
                    Flags: D3D12_EXPORT_FLAG_NONE,
                }]
            })
            .collect();
        // The library descs as ONE pre-sized Vec (lib 0 + the per-class
        // specializations), frozen before the subobject array stores raw
        // pointers into it — the hit-group lifetime rule again.
        let mut lib_descs: Vec<D3D12_DXIL_LIBRARY_DESC> =
            Vec::with_capacity(1 + class_dxil.len());
        lib_descs.push(D3D12_DXIL_LIBRARY_DESC {
            DXILLibrary: D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr() as *const _,
                BytecodeLength: dxil.len(),
            },
            // No export list off-lever: every [shader("...")] entry exports.
            NumExports: exports.len() as u32,
            pExports: if exports.is_empty() {
                std::ptr::null_mut()
            } else {
                exports.as_ptr() as *mut _
            },
        });
        for (i, blob) in class_dxil.iter().enumerate() {
            lib_descs.push(D3D12_DXIL_LIBRARY_DESC {
                DXILLibrary: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: blob.as_ptr() as *const _,
                    BytecodeLength: blob.len(),
                },
                NumExports: 1,
                pExports: class_exports[i].as_ptr() as *mut _,
            });
        }
        // ALPHA_CUTOUT scenes attach the cutout any-hit to every hit group;
        // HgOcclude carries ONLY an any-hit (legal — SKIP_CLOSEST_HIT skips
        // just that stage, any-hit still runs during traversal: the standard
        // alpha-tested-shadow pattern, and the untouched-payload = occluded
        // convention holds: all-rejected => miss_shadow writes 0).
        let ahs = |name: &Vec<u16>| {
            if non_opaque { PCWSTR(name.as_ptr()) } else { PCWSTR::null() }
        };
        let hit_group = |export: &Vec<u16>, chs: PCWSTR, ah: PCWSTR| D3D12_HIT_GROUP_DESC {
            HitGroupExport: PCWSTR(export.as_ptr()),
            Type: D3D12_HIT_GROUP_TYPE_TRIANGLES,
            AnyHitShaderImport: ah,
            ClosestHitShaderImport: chs,
            IntersectionShaderImport: PCWSTR::null(),
        };
        // Hit groups as ONE pre-sized Vec: the subobject array stores raw
        // pointers INTO it, so it is fully populated here and NEVER pushed
        // to again (a realloc would invalidate every stored pDesc — the
        // documented RTPSO-lifetime footgun). Off-lever the contents are
        // exactly the legacy three; the alias arm swaps the single HgShade
        // for 8 per-class groups whose chs imports are the renamed aliases
        // (every class shares ah_shade under non_opaque — the payload-type
        // pairing is per group KIND, so the per-class surface is only the
        // chs column).
        let mut hit_groups: Vec<D3D12_HIT_GROUP_DESC> = Vec::with_capacity(n_classes + 2);
        // FR_DXR_LEAN: zero hit groups — a hit group's imports would drag the
        // dead chs/ah exports back into the state object, which is exactly
        // what the control removes.
        if !lean {
            if sbt_mode >= 1 {
                for k in 0..n_classes {
                    hit_groups.push(hit_group(
                        &hg_class_names[k],
                        PCWSTR(chs_alias_names[k].as_ptr()),
                        ahs(&ah_shade_name),
                    ));
                }
            } else {
                hit_groups.push(hit_group(
                    &hg_shade_name,
                    PCWSTR(chs_shade_name.as_ptr()),
                    ahs(&ah_shade_name),
                ));
            }
            hit_groups
                .push(hit_group(&hg_hit_name, PCWSTR(chs_hit_name.as_ptr()), ahs(&ah_hit_name)));
            if non_opaque {
                hit_groups.push(hit_group(
                    &hg_occlude_name,
                    PCWSTR::null(),
                    PCWSTR(ah_shadow_name.as_ptr()),
                ));
            }
        }
        // RayPayload {float3 + float + uint + float2 + uint} = 32 B is the
        // largest payload (the float2/uint tail is --spp: the sample's own
        // position, and prim = `(sample << 1) | probe_bit` — the index rides
        // the high bits so the miss shader can key the per-sample cloud march
        // phase without growing the payload). HitPayload and ShadowPayload are
        // each 24 B after adding the relief ray's logical tmin/tmax, so they
        // remain below that ceiling; triangle barycentrics = 8 B.
        let shader_cfg = D3D12_RAYTRACING_SHADER_CONFIG {
            MaxPayloadSizeInBytes: 32,
            MaxAttributeSizeInBytes: 8,
        };
        // raygen -> chs_shade (1); its shadow/AO/reflection rays (2); chs_hit
        // and the misses fire nothing. The CPU's depth-1 recursion is the
        // flattened lap loop inside chs_shade, not payload recursion. Under
        // FR_DXR_INLINE the secondaries are inline RayQuery, so the deepest
        // TraceRay is raygen's primary (mode 1) or none at all (mode 2 —
        // depth 1 stays declared: 0 is legal but is a separate micro-variant,
        // not worth a DispatchRays-validation seam until mode 2 shows a win).
        // --dxr-sbt 3 makes the continuations REAL payload recursion, and the
        // depth derives from the CPU policy the lap arithmetic encodes:
        // raygen's primary (1) → the depth-0 reflection (2) → that surface's
        // transmission chain, which fires while depth < TRANS_MAX_DEPTH = 4,
        // i.e. from surfaces at depths 1, 2, 3 (+3) = 5; the root's own chain
        // reaches the same 5 without the reflection hop. Occlusion is inline
        // RayQuery (the hybrid), so shadow/AO rays add NOTHING — exceeding
        // the declared depth is device removal, not an error, so this
        // arithmetic is a soundness bound, not a tuning knob.
        let pipe_cfg = D3D12_RAYTRACING_PIPELINE_CONFIG {
            // FR_DXR_LEAN takes the depth-0 micro-variant the comment above
            // deferred (legal: the lean guard proved nothing traces).
            MaxTraceRecursionDepth: if lean {
                0
            } else if recurse {
                5
            } else if inline_mode > 0 {
                1
            } else {
                2
            },
        };
        let grs = D3D12_GLOBAL_ROOT_SIGNATURE {
            pGlobalRootSignature: unsafe { std::mem::transmute_copy(&root_sig) },
        };
        let sub = |t: D3D12_STATE_SUBOBJECT_TYPE, p: *const std::ffi::c_void| {
            D3D12_STATE_SUBOBJECT { Type: t, pDesc: p }
        };
        let mut subobjects = vec![
            sub(
                D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG,
                &shader_cfg as *const _ as *const _,
            ),
            sub(
                D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG,
                &pipe_cfg as *const _ as *const _,
            ),
            sub(D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &grs as *const _ as *const _),
        ];
        // Every DXIL library — lib 0 + the mode-2 per-class specializations —
        // by reference into the frozen Vec (off/mode-1: exactly one entry).
        for ld in &lib_descs {
            subobjects
                .push(sub(D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, ld as *const _ as *const _));
        }
        // Hit groups by reference into the (now frozen) Vec — one per class
        // + HgHit (+ HgOcclude, which imports ah_shadow and only exports
        // under ALPHA_CUTOUT/HEIGHTFIELD/TRANS_SHADOW, so its group exists
        // exactly when the library exports it). Subobject ORDER is not
        // significant to CreateStateObject.
        for hg in &hit_groups {
            subobjects.push(sub(D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, hg as *const _ as *const _));
        }
        let so_desc = D3D12_STATE_OBJECT_DESC {
            Type: D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
            NumSubobjects: subobjects.len() as u32,
            pSubobjects: subobjects.as_ptr(),
        };
        let state: ID3D12StateObject = unsafe { device5.CreateStateObject(&so_desc) }
            .map_err(|e| format!("CreateStateObject(DXR pipeline): {e}"))?;

        // --- SBT: bare 32-byte identifiers (the global root signature
        // carries every binding, so records need no local root arguments).
        let props: ID3D12StateObjectProperties =
            state.cast().map_err(|e| format!("ID3D12StateObjectProperties: {e}"))?;
        let ident = |name: &str| -> Result<[u8; IDENT]> {
            let wn = wname(name);
            let p = unsafe { props.GetShaderIdentifier(PCWSTR(wn.as_ptr())) };
            if p.is_null() {
                return Err(format!("GetShaderIdentifier({name}): not found in the RTPSO"));
            }
            Ok(unsafe { *(p as *const [u8; IDENT]) })
        };
        // Hit-table records: 3 off-lever (the pre-feature layout verbatim),
        // class-major [HgShade_ck, HgHit, HgOcclude] × 8 on the alias arm —
        // repeating HgHit/HgOcclude per class slot is free at identifier-only
        // stride and is what keeps RayContribution's {0,1,2} meaning inside
        // every class triplet with ZERO TraceRay call-site changes.
        let sbt_hit_records: u32 =
            if sbt_mode >= 1 { (n_classes * 3) as u32 } else { 3 };
        let sbt_size = SBT_HIT + sbt_hit_records as usize * IDENT;
        let sbt = d3d12::UploadBuffer::new(device, sbt_size)?;
        unsafe { std::ptr::write_bytes(sbt.ptr, 0, sbt_size) };
        let put = |off: usize, id: [u8; IDENT]| unsafe {
            std::ptr::copy_nonoverlapping(id.as_ptr(), sbt.ptr.add(off), IDENT);
        };
        put(0, ident("raygen")?);
        // FR_DXR_LEAN: the miss/hit exports don't exist in the state object —
        // ident() on them would fail, and the dispatch desc carries null
        // ranges, so the (allocated, zeroed) table regions are never read.
        if !lean {
            put(SBT_MISS, ident("miss_radiance")?);
            put(SBT_MISS + IDENT, ident("miss_shadow")?);
            put(SBT_MISS + 2 * IDENT, ident("miss_hit")?);
            // Mode 3's continuation sentinel at miss index 3: 64 + 4·32 = 192 —
            // the fourth record fills the miss region to EXACTLY SBT_HIT, no
            // layout move (the const's own doc says the table fits beneath it).
            if recurse {
                put(SBT_MISS + 3 * IDENT, ident("miss_rec")?);
            }
        }
        let mut sbt_distinct_idents: u32 = 1;
        let mut sbt_required_idents: u32 = 1;
        if sbt_mode >= 1 {
            // Per-class identifiers. PAIRWISE DISTINCTNESS over the mode's
            // REQUIRED set is the anti-vacuity — mode 1: all 8 (the rename
            // experiment IS "does the driver mint distinct keys for
            // renames"); mode 2: the specialized set ∪ uber (absent classes
            // alias uber by construction and legitimately dedupe on drivers
            // that fold renames — the recorded NVIDIA finding — while a
            // SPECIALIZED library colliding with anything is a defect on any
            // vendor). Loud here (a real Q4 data point), GATED in
            // --check-dxr's construction audit via the two counts.
            let ids: Vec<[u8; IDENT]> = (0..n_classes)
                .map(|k| ident(&format!("HgShade_c{k}")))
                .collect::<Result<_>>()?;
            let req: Vec<usize> = if sbt_mode >= 2 {
                let mut v = spec_classes.clone();
                v.push(crate::shadeclass::CK_UBER as usize);
                v
            } else {
                (0..n_classes).collect()
            };
            sbt_required_idents = req.len() as u32;
            let mut uniq: Vec<&[u8; IDENT]> = Vec::new();
            for &k in &req {
                if !uniq.contains(&&ids[k]) {
                    uniq.push(&ids[k]);
                }
            }
            sbt_distinct_idents = uniq.len() as u32;
            if sbt_distinct_idents != sbt_required_idents {
                eprintln!(
                    "dxr-sbt: DRIVER DEDUPED shading records — {} distinct of {} required \
                     (the TSU has that many sort keys{})",
                    sbt_distinct_idents,
                    sbt_required_idents,
                    if sbt_mode >= 2 {
                        "; two DIFFERENT libraries folded — a finding on any vendor"
                    } else {
                        "; the alias rung is vacuous here, which is itself a finding"
                    }
                );
            }
            let hit_id = ident("HgHit")?;
            let occl_id = if non_opaque { Some(ident("HgOcclude")?) } else { None };
            for (k, id) in ids.iter().enumerate() {
                put(SBT_HIT + (3 * k) * IDENT, *id);
                put(SBT_HIT + (3 * k + 1) * IDENT, hit_id);
                // Slot 3k+2: the zeroed null record on opaque scenes (the
                // legacy convention, repeated per class); HgOcclude armed.
                if let Some(o) = occl_id {
                    put(SBT_HIT + (3 * k + 2) * IDENT, o);
                }
            }
        } else if !lean {
            put(SBT_HIT, ident("HgShade")?);
            put(SBT_HIT + IDENT, ident("HgHit")?);
            // Hit group 2 (occlusion rays): the zeroed null record on opaque
            // scenes (SKIP_CLOSEST_HIT + FORCE_OPAQUE never run a shader from
            // it); the any-hit-only HgOcclude on alpha-masked/height scenes.
            if non_opaque {
                put(SBT_HIT + 2 * IDENT, ident("HgOcclude")?);
            }
        }

        // --- Stack introspection (the DispatchRays hunt): the ONE
        // register-adjacent number D3D12 exposes for RT pipelines. Per-shader
        // frames via GetShaderStackSize (hit-group members take the spec's
        // qualified "group::stage" spelling; 0xFFFFFFFF = not queryable,
        // printed '-') plus the driver's default GetPipelineStackSize — the
        // reservation the work-graph backing tell (B70 517 MB vs NVIDIA
        // 82 MB, identical graph) suggests Intel pads. Always-on one-liner
        // (the vram-line precedent): a fat shader's spill lives in this
        // stack on drivers that scratch-back it, so the B70-vs-4090 diff of
        // this line is the DXR-side twin of the FR_WIDTH report.
        let sstack = |name: &str| -> u64 {
            let wn = wname(name);
            unsafe { props.GetShaderStackSize(PCWSTR(wn.as_ptr())) }
        };
        const STACK_NA: u64 = 0xFFFFFFFF;
        let sfmt =
            |v: u64| if v >= STACK_NA { "-".to_string() } else { v.to_string() };
        let pipe_default = unsafe { props.GetPipelineStackSize() };
        let mut stack_rows: Vec<(String, u64)> = vec![("raygen".into(), sstack("raygen"))];
        // FR_DXR_LEAN: raygen is the only export — querying the others would
        // just print '-' rows for shaders the state object doesn't hold.
        if !lean {
            stack_rows.push(("miss".into(), sstack("miss_radiance")));
            stack_rows.push(("miss_sh".into(), sstack("miss_shadow")));
            stack_rows.push(("miss_hit".into(), sstack("miss_hit")));
            if recurse {
                stack_rows.push(("miss_rec".into(), sstack("miss_rec")));
            }
            let shade_group = if sbt_mode >= 1 { "HgShade_c0" } else { "HgShade" };
            if sbt_mode >= 1 {
                // Per-class frames — under sbt 2 a specialized thin CHS should
                // report a smaller frame than uber's, a real per-class datum.
                for k in 0..n_classes {
                    stack_rows
                        .push((format!("c{k}"), sstack(&format!("HgShade_c{k}::closesthit"))));
                }
            } else {
                stack_rows.push(("chs".into(), sstack("HgShade::closesthit")));
            }
            stack_rows.push(("hit".into(), sstack("HgHit::closesthit")));
            if non_opaque {
                stack_rows.push(("ah".into(), sstack(&format!("{shade_group}::anyhit"))));
                stack_rows.push(("occl".into(), sstack("HgOcclude::anyhit")));
            }
        }
        eprintln!(
            "dxr stack: pipeline-default={pipe_default} | {}",
            stack_rows
                .iter()
                .map(|(l, v)| format!("{l}={}", sfmt(*v)))
                .collect::<Vec<_>>()
                .join(" ")
        );
        match stack_override() {
            StackOverride::Off => {}
            ov => {
                let rg = stack_rows[0].1;
                if rg >= STACK_NA {
                    eprintln!("dxr stack: raygen frame not queryable — FR_DXR_STACK skipped");
                } else {
                    // Callee frame = max over every non-raygen shader the
                    // graph can invoke; raygen + declared-depth × that is
                    // sufficient for every mode (mode 2 invokes NOTHING
                    // below raygen — its bound is the raygen frame alone).
                    let callee_max = stack_rows[1..]
                        .iter()
                        .map(|&(_, v)| v)
                        .filter(|&v| v < STACK_NA)
                        .max()
                        .unwrap_or(0);
                    let v = match ov {
                        StackOverride::Bytes(n) => n,
                        _ if inline_mode == 2 => rg,
                        _ => rg + u64::from(pipe_cfg.MaxTraceRecursionDepth) * callee_max,
                    };
                    unsafe { props.SetPipelineStackSize(v) };
                    eprintln!(
                        "dxr stack: SetPipelineStackSize({v}) — FR_DXR_STACK armed \
                         (driver default {pipe_default})"
                    );
                }
            }
        }

        // The shared core arrived pre-uploaded (Rc from GpuContext's cache).
        // The DXR pipeline never binds the software BVH (see bind_common —
        // t0/t1 stay unset), and those trees now live OUTSIDE the core
        // (trace::SwTreesGpu, per-TraceGpu), so a DXR session structurally
        // never pays their upload (~2.3 GB at 100M tris).
        let uaf = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let px = rw as u64 * rh as u64;
        let accum = committed_buffer(device, px * 12, uaf, ua)?;
        let tbuf = committed_buffer(device, px * 4, uaf, ua)?;
        let info = committed_buffer(device, px * 4, uaf, ua)?;
        let gbuf = committed_buffer(
            device,
            if gbuf_full { px * trace::GBUF_STRIDE } else { trace::GBUF_STRIDE },
            uaf,
            ua,
        )?;
        let gbuf_ext = committed_buffer(
            device,
            if gbuf_full { px * trace::GBUF_EXT_STRIDE } else { trace::GBUF_EXT_STRIDE },
            uaf,
            ua,
        )?;
        let hdr = d3d12::committed_tex(
            device,
            rw,
            rh,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            uaf,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        // The cloud caches (sized exactly as TraceGpu's). One float4 per lattice
        // point + a one-point border; N*N scalar transmittances at the cap.
        let lw = (rw >> sky_lod_k.trailing_zeros()) as u64 + 2;
        let lh = (rh >> sky_lod_k.trailing_zeros()) as u64 + 2;
        let cloud_lod = committed_buffer(device, (lw * lh).max(1) * 16, uaf, ua)?;
        let csn_n =
            if cloud_shadow_n > 0 { crate::clouds::CLOUD_SHADOW_MAX as u64 } else { 1 };
        let cloud_shadow = committed_buffer(device, csn_n * csn_n * 4, uaf, ua)?;
        // Mode-3 pass-pair buffers: 20 B/px hit records + 12 B/px cross-pass
        // color sum (~66 MB at 1080p), only when the thin arm compiled.
        let mode3 = match pso_dxr_shade {
            Some(pso_shade) => Some(Mode3 {
                pso_shade,
                hitrec: committed_buffer(device, px * 20, uaf, ua)?,
                csum: committed_buffer(device, px * 12, uaf, ua)?,
            }),
            None => None,
        };
        let scene_aabb = trace::scene_shadow_aabb(scene);
        // FEED_SETS copies of the RP_TEX table (hdr resolve target at each set's
        // offset 0, then that set's feed planes — wired later), then slots
        // TEX_HEAP_BASE.. = the RP_SCENE_TEX scene table — the tracer's heap
        // layout exactly.
        let uav_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: TEX_HEAP_BASE
                    + TEX_TABLE_BUFS
                    + scene_gpu.textures.len() as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("CreateDescriptorHeap(dxr UAV): {e}"))?;
        trace::write_resolve_uavs(device, &uav_heap, &hdr);
        scene_gpu.write_scene_descriptors(device, &uav_heap, TEX_HEAP_BASE);
        let tex_table = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: unsafe { uav_heap.GetGPUDescriptorHandleForHeapStart() }.ptr
                + TEX_HEAP_BASE as u64
                    * unsafe {
                        device.GetDescriptorHandleIncrementSize(
                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        )
                    } as u64,
        };
        let frame_cb = d3d12::UploadBuffer::new(device, CB_STRIDE * d3d12::FRAMES_IN_FLIGHT)?;

        let name = |res: &ID3D12Resource, n: &str| {
            let w: Vec<u16> = n.encode_utf16().chain(std::iter::once(0)).collect();
            let _ = unsafe { res.SetName(PCWSTR(w.as_ptr())) };
        };
        name(&accum, "dxr.accum");
        name(&tbuf, "dxr.tbuf");
        name(&info, "dxr.info");
        name(&hdr, "dxr.hdr");
        name(&gbuf, if gbuf_full { "dxr.gbuf" } else { "dxr.gbuf_dummy" });
        name(&gbuf_ext, if gbuf_full { "dxr.gbuf_ext" } else { "dxr.gbuf_ext_dummy" });

        name(&cloud_lod, "dxr.cloud_lod");
        name(&cloud_shadow, "dxr.cloud_shadow");

        // The DXR twin of the wavefront's construction vram line (trace.rs):
        // in a SPACE-cycled session this pipeline's planes land beside a live
        // TraceGpu and the shared scene core, and WDDM demotes over-budget
        // commits silently — print where the commit landed.
        if let Some((usage, budget)) = super::adapter::vram_info(device) {
            eprintln!(
                "gpu tracer: dxr planes committed | vram {} / {} MB",
                usage >> 20,
                budget >> 20
            );
        }

        Ok(Self {
            root_sig,
            state,
            sbt,
            sbt_hit_records,
            lean,
            sbt_mode,
            sbt_distinct_idents,
            sbt_required_idents,
            pso_resolve,
            pso_sky_lod,
            pso_cloud_shadow,
            cloud_lod,
            cloud_shadow,
            sky_lod_k,
            cloud_shadow_n,
            scene_aabb,
            scene: scene_gpu,
            accum,
            tbuf,
            info,
            gbuf,
            gbuf_ext,
            force_gbuf_ext: std::cell::Cell::new(false),
            sway_t: std::cell::Cell::new(None),
            sway_mv_t: std::cell::Cell::new(None),
            sway_mv_on: !sway_def.is_empty(),
            mode3,
            width_buf: if trace::width_probe_on() {
                Some(committed_buffer(device, 16, uaf, ua)?)
            } else {
                None
            },
            spp: std::cell::Cell::new(1),
            gbuf_full,
            hdr,
            uav_heap,
            tex_table,
            pso_feed_xess,
            pso_feed_rr,
            pso_feed_fsr_rr,
            feed: Vec::new(),
            device: device.clone(),
            frame_cb,
            cb_base: FrameCb::base(scene, rw, rh),
            rw,
            rh,
        })
    }

    /// See `TraceGpu::force_gbuf_ext` — gates only.
    pub fn force_gbuf_ext(&self, on: bool) {
        self.force_gbuf_ext.set(on);
    }

    /// FR_WIDTH: the DXR width sink (slot 0 = raygen, slot 1 = cs_dxr_shade),
    /// Some iff the lever was armed at construction. Readback-only consumers.
    pub fn width_buf(&self) -> Option<&ID3D12Resource> {
        self.width_buf.as_ref()
    }

    pub fn write_cb(&self, slot: usize, p: &FrameParams) {
        // Stash the sway clock for record_frame/bind_common — write_cb is the
        // one per-frame site that sees FrameParams, and every present chain
        // calls it before record_frame (the --check-dxr paths pass None).
        self.sway_t.set(p.sway_time);
        // ...and the frame's spp for the mode-3 pass-pair loop (same idiom;
        // U-key live changes and every check path route through here).
        self.spp.set(p.spp.max(1));
        // One FSR4-RR subscriber among the wired engines is enough to arm the
        // pack's signal lanes (--quinlight can wire several).
        let fsr_sig = self.feed.iter().any(|(k, _)| matches!(k, trace::FeedKind::FsrRr));
        // ...and one RR/FSR-RR subscriber is enough to require the pack's
        // guide/signal half. DXR has no NPPD arm, so unlike TraceGpu's
        // `gbuf_ext_needed` there is no nppd term here.
        let gbuf_ext = self.force_gbuf_ext.get()
            || self
                .feed
                .iter()
                .any(|(k, _)| matches!(k, trace::FeedKind::Rr | trace::FeedKind::FsrRr));
        let mut cb = self.cb_base.with_frame(p, self.gbuf_full, fsr_sig, gbuf_ext);
        // Sway MVs: arm the flag + the ring-slot base, and stash the clock
        // pair for record_frame's dmv fill — one predicate (sway_mv_pair +
        // the compile-in) drives both, the TraceGpu discipline.
        let pair = if self.sway_mv_on { trace::sway_mv_pair(p) } else { None };
        self.sway_mv_t.set(pair);
        if pair.is_some() {
            if let Some(sw) = &self.scene.sway {
                cb.arm_sway_mv(slot as u32 * sw.n_inst());
            }
        }
        // The per-frame slab-space shadow grid (the shared shadow_grid_row —
        // one source of truth with TraceGpu); zero when the cache is off or
        // clouds are disabled (cloud_sun_transmittance then takes its exact arm).
        if self.cloud_shadow_n > 0 && p.clouds.enabled {
            cb.cloud_grid = crate::clouds::shadow_grid_row(
                self.cb_base.sun,
                self.scene_aabb,
                p.clouds.diag,
                self.cloud_shadow_n,
            );
        }
        cb.store(unsafe { self.frame_cb.ptr.add(slot * CB_STRIDE) });
    }

    /// Re-derive the base CB's sun/sky rows after a TOD change —
    /// `TraceGpu::refresh_sky`'s twin (`FrameCb::refresh_sky_rows`).
    pub fn refresh_sky(&mut self, scene: &Scene) {
        self.cb_base.refresh_sky_rows(scene, self.rw, self.rh);
    }

    /// The DXR twin of TraceGpu::wire_feed — same heap layout, same typed-store
    /// gate, same semantics: REPLACES the wiring with this one engine (so
    /// --check-dxr can rewire from one feed kind to the next).
    pub fn wire_feed(
        &mut self,
        device: &ID3D12Device,
        kind: trace::FeedKind,
        targets: &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
    ) -> Result<()> {
        self.feed.clear();
        self.wire_feed_add(device, kind, targets)
    }

    /// APPENDS one engine, claiming the next descriptor set (--quinlight).
    pub fn wire_feed_add(
        &mut self,
        device: &ID3D12Device,
        kind: trace::FeedKind,
        targets: &[(u32, &ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)],
    ) -> Result<()> {
        let set = self.feed.len() as u32;
        let planes = trace::wire_feed_targets(device, &self.uav_heap, set, targets)?;
        self.feed.push((kind, planes));
        Ok(())
    }

    /// Fan the pack + accum out into the wired upscaler input planes — one
    /// dispatch per wired engine. Record AFTER record_frame on the same list
    /// (its trailing global UAV barrier fences the pack/accum writes).
    pub fn record_feed(&self, list: &ID3D12GraphicsCommandList, slot: usize) -> Result<()> {
        if self.feed.is_empty() {
            return Err("feed targets not wired".into());
        }
        let mut feeds: Vec<(&ID3D12PipelineState, u32, &[ID3D12Resource])> = Vec::new();
        for (set, (kind, planes)) in self.feed.iter().enumerate() {
            let pso = trace::feed_pso(
                *kind,
                None,
                self.pso_feed_xess.as_ref(),
                self.pso_feed_rr.as_ref(),
                self.pso_feed_fsr_rr.as_ref(),
            )
            .ok_or("feed PSO missing (DxrGpu built without gbuf)")?;
            feeds.push((pso, set as u32, planes.as_slice()));
        }
        trace::record_feed_dispatch(
            list,
            &self.device,
            &self.uav_heap,
            &feeds,
            None,
            self.rw,
            self.rh,
            &|| unsafe { self.bind_common(list, slot) },
        );
        Ok(())
    }

    /// The DXR subset of the shared root layout. t0/t1 (software BVH) and the
    /// wavefront queue UAVs stay unbound — no shader in this library touches
    /// them, and unaccessed root descriptors are legal to leave unset.
    unsafe fn bind_common(&self, list: &ID3D12GraphicsCommandList, slot: usize) {
        unsafe {
            list.SetComputeRootSignature(&self.root_sig);
            list.SetComputeRootConstantBufferView(
                RP_FRAME_CBV,
                self.frame_cb.resource.GetGPUVirtualAddress() + (slot * CB_STRIDE) as u64,
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_ACCUM,
                self.accum.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_TBUF,
                self.tbuf.GetGPUVirtualAddress(),
            );
            list.SetComputeRootUnorderedAccessView(
                RP_UAV0 + UAV_INFO,
                self.info.GetGPUVirtualAddress(),
            );
            // FR_WIDTH: the width sink at the otherwise-unbound u3 root param
            // (writing through an unset root descriptor is UB — the buffer
            // exists exactly when the shaders that write it compiled in).
            if let Some(wb) = &self.width_buf {
                list.SetComputeRootUnorderedAccessView(
                    RP_UAV0 + UAV_COUNTERS,
                    wb.GetGPUVirtualAddress(),
                );
            }
            list.SetComputeRootUnorderedAccessView(RP_GBUF, self.gbuf.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(
                RP_GBUF_EXT,
                self.gbuf_ext.GetGPUVirtualAddress(),
            );
            let s = &self.scene;
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_POSITIONS,
                s.positions.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_NORMALS,
                s.normals.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_INDICES,
                s.indices.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_TRI_MAT,
                s.tri_mat.GetGPUVirtualAddress(),
            );
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_MATERIALS,
                s.materials.GetGPUVirtualAddress(),
            );
            // --foliage-sway: an animated frame traces the ring slot's TLAS;
            // everything else (plain sessions, gates, None-clock frames) the
            // static rest pose. record_frame rebuilt the slot BEFORE this
            // bind takes effect at DispatchRays.
            list.SetComputeRootShaderResourceView(
                RP_SRV0 + SRV_TLAS,
                match (self.sway_t.get(), s.sway.as_ref()) {
                    (Some(_), Some(sw)) => sw.tlas_va(slot),
                    _ => s.tlas.GetGPUVirtualAddress(),
                },
            );
            // The scene-texture table (t0..t3 + texs[] in space1) — heap
            // before table, same as the tracer's bind_common.
            list.SetDescriptorHeaps(&[Some(self.uav_heap.clone())]);
            list.SetComputeRootDescriptorTable(RP_SCENE_TEX, self.tex_table);
        }
    }

    /// One DispatchRays over the full target. Ends with a global UAV barrier
    /// so the resolve (or a readback) sees the splats.
    pub fn record_frame(&self, list: &ID3D12GraphicsCommandList, slot: usize) -> Result<()> {
        let list4: ID3D12GraphicsCommandList4 =
            list.cast().map_err(|e| format!("ID3D12GraphicsCommandList4: {e}"))?;
        let _ev = super::pix::scope(list, c"dxr");
        // --foliage-sway: rebuild this slot's animated TLAS at the frame's
        // clock BEFORE anything binds it (bind_common below picks the ring
        // slot when the clock is Some). A bit-equal clock records nothing —
        // the converging-still fast path.
        if let (Some(t), Some(sw)) = (self.sway_t.get(), self.scene.sway.as_ref()) {
            let _sv = super::pix::scope(list, c"dxr-sway-tlas");
            sw.record_rebuild(list, slot, t)?;
            // Sway MVs: fill the slot's prev−cur rows under write_cb's own
            // stashed predicate (the TraceGpu::record_sway shape).
            if let Some((tc, tp)) = self.sway_mv_t.get() {
                sw.write_mv_rows(slot, tc, tp);
            }
        }
        unsafe {
            self.bind_common(list, slot);
            // The cloud caches, ahead of the ray dispatch and left bound at
            // u6/u5 for the whole DispatchRays. record_frame is the ONE dispatch
            // site (session chains AND --check-dxr route through it), so the
            // "every compiling path must fill or the shader reads garbage as
            // float4 = device hang" contract (trace.rs::record_cloud_shadow's
            // war story) holds structurally. The root-descriptor bindings
            // persist on the list, so the DispatchRays below reads them.
            if let Some(pso) = &self.pso_cloud_shadow {
                let _e = super::pix::scope(list, c"dxr-cloud-shadow");
                list.SetComputeRootUnorderedAccessView(
                    RP_UAV0 + UAV_QOUT,
                    self.cloud_shadow.GetGPUVirtualAddress(),
                );
                let groups = (crate::clouds::CLOUD_SHADOW_MAX * crate::clouds::CLOUD_SHADOW_MAX)
                    .div_ceil(64);
                list.SetPipelineState(pso);
                list.Dispatch(groups.min(32768), groups.div_ceil(32768), 1);
                list.ResourceBarrier(&[uav_barrier(None)]);
            }
            if let Some(pso) = &self.pso_sky_lod {
                let _e = super::pix::scope(list, c"dxr-sky-lod");
                list.SetComputeRootUnorderedAccessView(
                    RP_UAV0 + UAV_QIN,
                    self.cloud_lod.GetGPUVirtualAddress(),
                );
                let k = self.sky_lod_k;
                let pts = ((self.rw / k) + 2) * ((self.rh / k) + 2);
                let groups = pts.div_ceil(64);
                list.SetPipelineState(pso);
                list.Dispatch(groups.min(32768), groups.div_ceil(32768), 1);
                list.ResourceBarrier(&[uav_barrier(None)]);
            }
            let va = self.sbt.resource.GetGPUVirtualAddress();
            let desc = D3D12_DISPATCH_RAYS_DESC {
                RayGenerationShaderRecord: D3D12_GPU_VIRTUAL_ADDRESS_RANGE {
                    StartAddress: va,
                    SizeInBytes: IDENT as u64,
                },
                // FR_DXR_LEAN: null ranges — the raygen-only state object has
                // no miss/hit exports, the tables were never filled, and
                // nothing traces (declared recursion 0), so no lookup exists
                // to dereference them.
                MissShaderTable: if self.lean {
                    Default::default()
                } else {
                    D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
                        StartAddress: va + SBT_MISS as u64,
                        // 4 records under --dxr-sbt 3 (miss_rec, the continuation
                        // sentinel at index 3 — trace_shade names it); 3 otherwise.
                        SizeInBytes: (if self.sbt_mode == 3 { 4 } else { 3 } * IDENT) as u64,
                        StrideInBytes: IDENT as u64,
                    }
                },
                HitGroupTable: if self.lean {
                    Default::default()
                } else {
                    D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
                        StartAddress: va + SBT_HIT as u64,
                        // Runtime record count (--dxr-sbt) — snapshotted at
                        // construction beside the fill, so the two cannot
                        // disagree.
                        SizeInBytes: (self.sbt_hit_records as usize * IDENT) as u64,
                        StrideInBytes: IDENT as u64,
                    }
                },
                CallableShaderTable: Default::default(),
                Width: self.rw,
                Height: self.rh,
                Depth: 1,
            };
            if let Some(m3) = &self.mode3 {
                // Mode 3: thin pass pairs — for each sample, a bare-hit
                // DispatchRays (records only) fenced against a cs_dxr_shade
                // dispatch that does the fat shading in COMPUTE, which is the
                // whole point (see the mode doc at the top). The sample index
                // rides the b1 push constants; root arguments persist across
                // SetPipelineState/SetPipelineState1, so bind_common above
                // covers both halves — and must NOT be re-issued per pass (a
                // root-signature re-set would wipe the pushed constant). The
                // per-pass gputime scopes overflow MAX_TS at spp ≳ 30 —
                // graceful (dropped counter), and spp=1 is the regime the
                // mode exists for. The final barrier is the same trailing
                // fence record_feed/record_resolve have always relied on.
                let spp = self.spp.get().max(1);
                list.SetComputeRootUnorderedAccessView(
                    RP_UAV0 + UAV_QLEAF,
                    m3.hitrec.GetGPUVirtualAddress(),
                );
                list.SetComputeRootUnorderedAccessView(
                    RP_UAV0 + UAV_QSKY,
                    m3.csum.GetGPUVirtualAddress(),
                );
                for s in 0..spp {
                    list.SetComputeRoot32BitConstants(
                        RP_PUSH,
                        1,
                        &s as *const u32 as *const _,
                        0,
                    );
                    list4.SetPipelineState1(&self.state);
                    {
                        let _e = super::pix::scope(list, c"dxr-rays");
                        list4.DispatchRays(&desc);
                    }
                    list.ResourceBarrier(&[uav_barrier(None)]);
                    {
                        let _e = super::pix::scope(list, c"dxr-shade");
                        list.SetPipelineState(&m3.pso_shade);
                        list.Dispatch(self.rw.div_ceil(8), self.rh.div_ceil(8), 1);
                    }
                    list.ResourceBarrier(&[uav_barrier(None)]);
                }
            } else {
                list4.SetPipelineState1(&self.state);
                // The ray dispatch alone. Without this bracket `dxr` conflates
                // it with bind_common + SetPipelineState1 (a real cost on Arc),
                // and the decision this instrument exists for — optimize the
                // wavefront or flip the Intel world default to DXR — turns on
                // whether the DXR arm's time is rays or setup. The residual
                // `dxr - dxr-rays - the two cache fills` is now exactly that
                // setup.
                {
                    let _e = super::pix::scope(list, c"dxr-rays");
                    list4.DispatchRays(&desc);
                }
                list.ResourceBarrier(&[uav_barrier(None)]);
            }
        }
        Ok(())
    }

    /// accum -> hdr at 1/samples (trace.rs's resolve kernel + curve); hdr
    /// ends in PIXEL_SHADER_RESOURCE for the tonemap blit.
    pub fn record_resolve(&self, list: &ID3D12GraphicsCommandList, slot: usize, samples: u32) {
        let _ev = super::pix::scope(list, c"dxr-resolve");
        unsafe {
            list.ResourceBarrier(&[transition(
                &self.hdr,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            self.bind_common(list, slot);
            list.SetDescriptorHeaps(&[Some(self.uav_heap.clone())]);
            list.SetComputeRootDescriptorTable(
                RP_TEX,
                self.uav_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            let push = [1.0f32 / samples.max(1) as f32, 0.0, 0.0, 0.0];
            list.SetComputeRoot32BitConstants(RP_PUSH, 4, push.as_ptr() as *const _, 0);
            list.SetPipelineState(&self.pso_resolve);
            list.Dispatch(self.rw.div_ceil(8), self.rh.div_ceil(8), 1);
            list.ResourceBarrier(&[transition(
                &self.hdr,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
    }
}

#[cfg(test)]
mod mode3_shader_source_tests {
    use super::{DXR_HLSL, DXR_SHADE_HLSL, RT_DXR_HLSLI};

    /// FR_WIDTH's DXR halves stay behind their guards: an unguarded
    /// `dxr_width` would declare/write u3 in every session — a register this
    /// pipeline deliberately leaves unbound unless the lever armed a sink
    /// (writing through an unset root descriptor is UB).
    #[test]
    fn width_writes_are_guarded() {
        for (name, src, guard) in [
            ("dxr.hlsl", DXR_HLSL, "#ifdef WIDTH_PROBE_RAYGEN"),
            ("dxr_shade.hlsl", DXR_SHADE_HLSL, "#ifdef WIDTH_PROBE"),
        ] {
            for (i, _) in src.match_indices("dxr_width") {
                let last_open = src[..i]
                    .rfind(guard)
                    .unwrap_or_else(|| panic!("{name}: dxr_width used before any {guard}"));
                assert!(
                    !src[last_open..i].contains("#endif"),
                    "{name}: dxr_width use at byte {i} escaped its guard"
                );
            }
        }
    }

    /// FR_BALLAST=dxr:N (the knee-vs-knee host comparison): dxr.hlsl must
    /// carry exactly reference.hlsl's three ballast blocks — seed, loop
    /// recurrence, dead fold — each under the COMPOUND guard that confines
    /// them to the mode-2 arm (a seed compiled without its update would
    /// "measure" a flat curve), every `ballast` use inside a guard, and the
    /// fold branch-dead on the spp sentinel.
    #[test]
    fn dxr_ballast_confined_and_fold_branch_dead() {
        const GUARD: &str = "#if defined(BALLAST_N) && (BALLAST_N > 0) && \
                             defined(DXR_INLINE_SEC) && (DXR_INLINE_SEC == 2)";
        // The literal above is wrapped for rustfmt; rebuild the one-line form
        // the shader actually carries.
        let guard = GUARD.split_whitespace().collect::<Vec<_>>().join(" ");
        let n_guards = DXR_HLSL.matches(guard.as_str()).count();
        assert_eq!(n_guards, 3, "dxr.hlsl must carry exactly the 3 ballast blocks");
        for (i, _) in DXR_HLSL.match_indices("ballast") {
            let g = DXR_HLSL[..i]
                .rfind(guard.as_str())
                .expect("ballast text before any compound BALLAST_N guard");
            assert!(
                !DXR_HLSL[g..i].contains("#endif"),
                "ballast use at byte {i} escaped its guard"
            );
        }
        assert!(
            DXR_HLSL.contains("if (spp == 0xdeadu)"),
            "the dead fold must branch on the spp sentinel"
        );
    }

    /// The record's miss convention is `t < 0` (miss_hit's wire format);
    /// tbuf's is INF, and ff_glow/T1 classify on it. The deferred kernel must
    /// convert BEFORE any consumer — a raw -1 reaching tbuf flips every sky
    /// pixel to "hit". Source-ordering pin, the house style: the live HLSL
    /// stays the executable specification.
    #[test]
    fn deferred_kernel_converts_miss_sentinel_before_consumers() {
        let src = DXR_SHADE_HLSL;
        let sentinel = src.find("rec.t < 0.0").expect("miss-sentinel branch missing");
        let rebuild = src.find("h.tri = rec.tri").expect("HitInfo rebuild missing");
        let tb = src.find("tbuf[pi]").expect("tbuf write missing");
        assert!(
            sentinel < rebuild && sentinel < tb,
            "the miss-sentinel branch must precede the HitInfo rebuild and the tbuf write"
        );
    }

    /// The deferred kernel is a COMPUTE unit: a TraceRay in it would fail PSO
    /// creation at best and silently miscompile at worst. Its rays are
    /// rt.hlsli's inline RayQuery only.
    #[test]
    fn deferred_kernel_never_traces() {
        assert!(
            !DXR_SHADE_HLSL.contains("TraceRay("),
            "dxr_shade.hlsl must never contain a TraceRay — it is a cs_6_5 unit"
        );
    }

    /// Modes 0-2 must preprocess to today's bytes: rt_dxr.hlsli keeps exactly
    /// its two tlas/primitive guards (changing them double-defines the trace
    /// primitives against rt.hlsli — the review finding). The guard form is
    /// `!defined(DXR_INLINE_SEC) && !defined(DXR_SBT_RECURSE)` since the
    /// --dxr-sbt 3 hybrid ALSO pastes rt.hlsli (inline occlusion) — the same
    /// double-define hazard from a second define, so both must key both.
    /// The HitPayload's mode-3 `inst` field sits behind the mode-3 guard.
    #[test]
    fn rt_dxr_guards_intact_and_inst_guarded() {
        let src = RT_DXR_HLSLI;
        assert_eq!(
            src.matches("#if !defined(DXR_INLINE_SEC) && !defined(DXR_SBT_RECURSE)").count(),
            2,
            "rt_dxr.hlsli's tlas/primitive guards must key BOTH rt.hlsli-pasting defines"
        );
        let hp = src.find("struct HitPayload").expect("HitPayload missing");
        let hp_end = hp + src[hp..].find("};").expect("HitPayload unterminated");
        let body = &src[hp..hp_end];
        let guard = body
            .find("DXR_INLINE_SEC == 3")
            .expect("HitPayload's inst field must be mode-3-guarded");
        let inst = body.find("uint inst;").expect("HitPayload inst field missing");
        assert!(guard < inst, "the mode-3 guard must precede the inst field");
    }

    /// The finding-1 audit's source half of the zero-TraceRay proof (the
    /// artifact half is the offline DXIL disassembly — `dx.op.traceRay`
    /// absent from the mode-2 library): dxr.hlsl carries exactly TWO TraceRay
    /// sites, and BOTH are preprocessor-dead at DXR_INLINE_SEC == 2 — the
    /// thin one sits in the `== 3` guard's then-arm (no intervening
    /// #else/#endif between guard and call), the payload one in the `== 2`
    /// guard's #else arm. rt_dxr.hlsli's four sites are covered by the guard
    /// pins above/below (the two-define primitive guards + trace_shade under
    /// DXR_SBT_RECURSE). A third site, or one of these migrating out of its
    /// guard, fails here before any GPU sees it.
    #[test]
    fn mode2_raygen_tracerays_are_preprocessor_dead() {
        let src = DXR_HLSL;
        let sites: Vec<usize> = src.match_indices("TraceRay(").map(|(i, _)| i).collect();
        assert_eq!(sites.len(), 2, "dxr.hlsl must carry exactly two TraceRay sites");
        // Site 1 — the mode-3 thin arm: the nearest preceding `== 3` guard
        // with no #else/#endif between it and the call puts the call in that
        // guard's then-arm, dead at mode 2.
        let g3 = src[..sites[0]]
            .rfind("#if defined(DXR_INLINE_SEC) && DXR_INLINE_SEC == 3")
            .expect("thin TraceRay must follow a mode-3 guard");
        let between = &src[g3..sites[0]];
        assert!(
            !between.contains("#else") && !between.contains("#endif"),
            "the thin TraceRay must sit DIRECTLY in the mode-3 guard's then-arm"
        );
        // Site 2 — the payload primary: inside the mode-2 guard's OWN #else
        // arm (the guard spelling is unique; the ballast compound guards
        // parenthesize differently). The then-arm nests a SKY_LOD #if/#else,
        // so the arm boundary needs a DEPTH-tracked scan — anchoring on the
        // first textual #else would land on the nested one and let a
        // mode-2-LIVE TraceRay after it pass (the false-pass shape this pin
        // exists to prevent).
        let g2 = src
            .find("#if defined(DXR_INLINE_SEC) && DXR_INLINE_SEC == 2")
            .expect("mode-2 raygen guard missing");
        let (mut depth, mut arm_else, mut end2) = (0i32, None, None);
        for (off, line) in src[g2..].lines().map(|l| {
            let off = l.as_ptr() as usize - src.as_ptr() as usize;
            (off, l.trim_start())
        }) {
            if line.starts_with("#if") {
                depth += 1;
            } else if line.starts_with("#endif") {
                depth -= 1;
                if depth == 0 {
                    end2 = Some(off);
                    break;
                }
            } else if line.starts_with("#else") && depth == 1 {
                arm_else = Some(off);
            }
        }
        let arm_else = arm_else.expect("mode-2 guard has no depth-1 #else arm");
        let end2 = end2.expect("mode-2 guard unterminated");
        assert!(
            src[end2..].starts_with("#endif // DXR_INLINE_SEC == 2"),
            "the mode-2 guard's closing #endif lost its label — re-anchor this pin"
        );
        assert!(
            arm_else < sites[1] && sites[1] < end2,
            "the payload TraceRay must sit in the mode-2 guard's OWN #else arm"
        );
    }

    /// The thin raygen writes ONLY the hit record — accum/tbuf/info belong to
    /// the deferred kernel (two writers would break the one-splat-per-frame
    /// contract). Slice the mode-3 arm out of raygen by its landmarks.
    #[test]
    fn thin_raygen_writes_no_shading_outputs() {
        let src = DXR_HLSL;
        let start = src.find("The THIN arm:").expect("mode-3 raygen arm missing");
        let end = start + src[start..].find("#else").expect("mode-3 arm unterminated");
        let arm = &src[start..end];
        for w in ["accum[", "tbuf[", "info["] {
            assert!(
                !arm.contains(w),
                "the thin raygen arm must not write {w} — the deferred kernel owns it"
            );
        }
        assert!(arm.contains("hitrec[pi]"), "the thin arm must write the hit record");
    }
}

#[cfg(test)]
mod sbt_recurse_shader_source_tests {
    use super::{trace, DXR_HLSL, RT_DXR_HLSLI};

    /// The recursion's whole routing contract is ONE TraceRay line:
    /// RayContribution 0 (the class-major triplet's SHADING slot — the
    /// instance's `class * 3` contribution does the per-class dispatch, no
    /// shader-side routing exists to get wrong) and MissShaderIndex 3 (the
    /// miss_rec sentinel — a continuation must NEVER take miss_radiance's
    /// display sky: a reflection miss needs the PARENT lobe's MIS weight).
    #[test]
    fn trace_shade_routes_contribution_zero_and_sentinel_miss() {
        let src = RT_DXR_HLSLI;
        let f = src.find("float3 trace_shade(").expect("trace_shade missing");
        let body = &src[f..f + src[f..].find("\n}").expect("trace_shade unterminated")];
        assert!(
            body.contains("TraceRay(tlas, OPAQUE_RF, 0xffu, 0u, 0u, 3u, r, p)"),
            "trace_shade must fire at RayContribution 0 with MissShaderIndex 3"
        );
        assert!(
            body.contains("p.prim = 0x80000000u"),
            "trace_shade must tag the payload as a recursion continuation (probe bit 0)"
        );
    }

    /// BOTH continuation branches must recurse with depth + 1 — the
    /// probe-reach lesson's shape: a define that rewires only one branch
    /// leaves the other on the flattened lap loop, whose nx_* feeds are dead
    /// under the single-lap recursion arm, i.e. that branch's radiance
    /// silently vanishes. And the increment IS the recursion bound: the
    /// depth-gated chain (depth < TRANS_MAX_DEPTH) is what keeps the
    /// declared MaxTraceRecursionDepth = 5 sound — a dropped increment
    /// recurses flat past it, which is device removal, not an error.
    #[test]
    fn both_continuations_recurse_with_incremented_depth() {
        let src = trace::SHADE_HLSLI;
        assert!(
            src.contains("trace_shade(p, rdir, rng, depth + 1u, cone_w, rec_t)"),
            "the reflection continuation must recurse at depth + 1"
        );
        assert!(
            src.contains("trace_shade(torig, tdir, rng, depth + 1u, cone_w, trec_t)"),
            "the transmission continuation must recurse at depth + 1"
        );
    }

    /// miss_rec is a SENTINEL: t = INF and NOTHING else. A color write there
    /// would double-count against the parent's own miss arms (the MIS-
    /// weighted reflection sky / the fixed-phase glass sky).
    #[test]
    fn miss_rec_writes_only_the_sentinel() {
        let src = DXR_HLSL;
        let f = src.find("void miss_rec(").expect("miss_rec missing");
        let body = &src[f..f + src[f..].find('}').expect("miss_rec unterminated")];
        assert!(body.contains("p.t = INF"), "miss_rec must write the INF sentinel");
        assert!(!body.contains("p.color"), "miss_rec must never write color");
    }

    /// chs_shade's recursion branch must be decided BEFORE the primary-path
    /// shade_full call and must consume the payload's depth (sp.x, bit-punned)
    /// — a branch after shade_full would shade continuations twice, and one
    /// ignoring the depth would give every recursive surface the root policy
    /// (reflection re-arming at every level: unbounded recursion again).
    #[test]
    fn chs_recursion_branch_precedes_primary_and_carries_depth() {
        let src = DXR_HLSL;
        let rec = src.find("p.prim & 0x80000000u").expect("chs recursion tag test missing");
        let full = src.find("p.color = shade_full(").expect("chs primary shade_full missing");
        assert!(rec < full, "the recursion branch must precede the primary shade_full path");
        let arm = &src[rec..full];
        assert!(
            arm.contains("asuint(p.sp.x)"),
            "the recursion branch must pass the payload's depth into shade_split"
        );
    }
}
