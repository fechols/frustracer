//! Kernel assembly: the HLSL corpus, the `#define` generators that specialize
//! it per session, and the tuning knobs those defines carry.
//!
//! THERE IS NO `#include` ANYWHERE in `src/shaders/`. Every compile unit is a
//! CONCATENATION — a generated prelude of `#define`s, then the `.hlsli` files
//! that unit needs in a load-bearing order, then the kernel — so no file on
//! disk is what a shader compiler ever sees. That is the reason this module
//! exists as its own thing rather than as a corner of a backend: the assembly
//! is pure `String` work over `&'static str`, it decides what the shaders MEAN
//! (which is a renderer question, not an API one), and both backends must feed
//! their compilers byte-identical sources or every cross-backend A/B is
//! comparing two different programs.
//!
//! It moved out of `gpu/trace.rs` in the Vulkan port's M1, and the move is what
//! makes `cargo test` run the shader-source gates at the bottom of this file on
//! every platform. They had been stranded: the tests only ever asserted things
//! about `&'static str`s, but they lived inside a `#[cfg(windows)]` module, so
//! a Linux `cargo test` reported "0 tests" and the 21 soundness pins that no
//! GPU-free gate can otherwise reach were simply absent. A test that cannot run
//! where the developer works is a test that will rot.
//!
//! WHAT IS DELIBERATELY NOT HERE. Recording — pipeline-state objects, root
//! signatures, descriptor tables, dispatches — stays in the backend, because
//! that is genuinely per-API. The line falls where a type would have to be
//! named: two generators used to take backend types and both were reduced to
//! the FACT they were reading, not moved along with their arguments —
//! `cand_defs` takes a `gfx::vocab::Vendor` (which is a PCI id either API
//! reports identically), and `sway_defs` takes the bool `scene_gpu.sway
//! .is_some()` rather than the `SceneGpu`. Neither reduction loses anything:
//! the callers still derive the argument from the one place that owns it, so
//! the define and the resource it describes remain one decision.
//!
//! THE KNOBS LIVE HERE TOO, and that is not filing convenience. `LEAF_GROUP`,
//! `WIDE_LEVELS`, `LANE_STACK`, `SKY_GROUP`/`SKY_SPLIT` and the rest are
//! constants that reach the GPU ONLY as `#define`s in these strings — they have
//! no other representation — and every one of them carries a measured
//! justification in its doc comment that a second backend must not silently
//! re-decide. A Vulkan tracer that picked its own group widths would be a
//! different renderer wearing the same name, and the cross-backend image A/B
//! that is supposed to catch porting defects would instead be measuring the
//! tuning difference. Their `FR_*` levers move with them for the same reason.
//!
//! ONE THING TO WATCH when adding a define: it must reach EVERY compile unit
//! that can consume it. An ablation or lever that misses one unit does not
//! fail loudly — it answers CONFIDENTLY, comparing identical code against
//! itself. That has shipped four times here (`nogbuf` never reaching the sky
//! unit, `nopack` never reaching feed, `nowave` never reaching the tile unit,
//! and the DXR pipeline's feed unit pasting no ablations at all), and each time
//! the wrong conclusion survived for weeks because the probe reported "no
//! effect" and that read as a finding. `abl_announce`'s "matched GPU arms:
//! (none)" line exists as the alarm for exactly this.

// Off Windows this module's only non-test consumers do not exist yet: the
// backend that records with these constants is `#[cfg(windows)]`, and `vk/`
// has not landed. So a Linux `cargo check` sees a module whose items are used
// exclusively by the gates below — which `cargo test` DOES compile and run,
// and which is the whole reason the module is portable. The allow is scoped to
// the platform where the reason holds, so Windows keeps full dead-code
// analysis over every one of these; it retires when the second backend arrives.
// (Deliberately module-level rather than 60 per-item attributes: unlike the
// grab-bag this replaced in main.rs, every item here is unused for the ONE
// reason stated above.)
#![cfg_attr(not(windows), allow(dead_code))]

use crate::gfx::vocab::Vendor;
use crate::scene::Scene;

// counters[] slots — mirror of ctr.hlsli.
pub const CTR_TILE_A: u32 = 0;
pub const CTR_TILE_B: u32 = 1;
pub const CTR_LEAF: u32 = 2;
pub const CTR_SKY: u32 = 3;
pub const CTR_CUT: u32 = 4;
pub const CTR_OVERFLOW: u32 = 5;
pub const CTR_CUT_FALLBACK: u32 = 6;
pub const CTR_SPLIT: u32 = 7;
pub const CTR_BLOCKED: u32 = 8;
pub const CTR_HEMI_PT: u32 = 9;
pub const CTR_HEMI_A: u32 = 10;
pub const CTR_HEMI_B: u32 = 11;
pub const CTR_HEMI_LEAF: u32 = 12;
// Never read back on the CPU (the hemi cut pool is transient per batch), but
// the HLSL twin — ctr.hlsli slot 13 — is live (bumped in hemi_wave, zeroed in
// cs_seed), so the const stays: this table is the shared counter LAYOUT, and a
// hole in it would invite reusing 13 for something else.
#[allow(dead_code)]
pub const CTR_HEMI_CUT: u32 = 13;
pub const CTR_HEMI_EMPTY: u32 = 14;
pub const CTR_HEMI_RAYS: u32 = 15;
pub const CTR_V_FALSE_EMPTY: u32 = 16;
pub const CTR_V_TMIN: u32 = 17;
pub const CTR_ALPHA_REJ: u32 = 18;
pub const CTR_HEIGHT_REJ: u32 = 19;
/// Tinted-shadow candidate passes (TRANS_SHADOW scenes) — the anti-vacuity
/// stat proving occlusion rays really crossed transmissive surfaces.
pub const CTR_TRANS_PASS: u32 = 20;
/// Opaque software-continuation telemetry. All three are accumulated once per
/// non-root leaf record, so their cost does not scale with pixel count/SPP.
pub const CTR_FRONTIER_HANDLES: u32 = 21;
pub const CTR_FRONTIER_RAYS: u32 = 22;
pub const CTR_FRONTIER_ENTRIES: u32 = 23;
/// Pixels inside proven-empty sky rects (zero rays traced) — the empty-space
/// proof's product as a per-frame pixel count. Part of the TERMINAL structure
/// (the qsky rect-area sum), so `cs_seed_replay` preserves it with
/// CTR_LEAF/CTR_SKY/CTR_CUT.
pub const CTR_SKY_PX: u32 = 24;
/// Real-time-GI bounce rays (shade_full's RTGI block; wavefront leaf units
/// only — the reference kernel and the DXR library carry no counters). The
/// `--check-gpu` must-fire on armed sessions, exactly 0 under `--no-rtgi`.
pub const CTR_RTGI_RAYS: u32 = 25;
pub const CTR_COUNT: u32 = 26;
// WIDTH_PROBE slots (FR_WIDTH=1 — `width_defs`): each kernel reports its
// COMPILED wave width (WaveGetLaneCount()). DELIBERATELY >= CTR_COUNT: every
// zero loop runs `i < CTR_COUNT` and every gate readback reads CTR_COUNT*4
// bytes, so these slots are never zeroed and never gated by construction.
// LOCKSTEP with ctr.hlsli's block AND the counters buffer size (CTR_TOTAL*4).
pub const CTR_W_LEAF: u32 = 26;
pub const CTR_W_SKY: u32 = 27;
pub const CTR_W_LEVEL: u32 = 28;
pub const CTR_W_HEMI: u32 = 29;
pub const CTR_W_REFERENCE: u32 = 30;
pub const CTR_TOTAL: u32 = 31;
const _: () = assert!(CTR_W_LEAF >= CTR_COUNT && CTR_TOTAL == CTR_W_REFERENCE + 1);

/// queues.hlsli::LeafRec's stride — the qleaf allocation and main.rs's
/// check-gpu readback move in lockstep with the HLSL struct through this one
/// value (xy0 | xy1 | t_start | depth | TraversalFrontier::opaque.xy).
pub const LEAF_REC_BYTES: u64 = 24;

/// continuation.hlsli's software-provider wire values. The ray call site
/// treats both words as opaque; these mirrors exist only so --check-gpu can
/// reject a malformed producer record before trusting the consumer.
pub const FRONTIER_COOKIE_V1: u32 = 0x4652_4301;
pub const FRONTIER_ROOT_TOKEN: u32 = 0xffff_ffff;

/// The leaf kernel's thread-group width — ONE WAVE on both vendors, and that
/// is the whole point.
///
/// A leaf tile is not 8x8. `depth_full` is driven by the WIDER screen axis, so
/// at 1920x1080 a leaf rect is 1920/2^8 = 7.5 by 1080/2^8 = 4.2 — about **32
/// pixels**, never 64. The kernel used to dispatch 64 lanes per tile and let
/// the surplus half return immediately, which is nearly free on a wave32 GPU
/// (the all-idle second wave retires at once) and expensive on wave64, where
/// those lanes sit in the SAME wave and waste half its RT throughput. That one
/// mismatch was most of the AMD-vs-NVIDIA gap: per extra sample the leaf kernel
/// cost 2.27x its own reference kernel on RDNA but only 1.24x on Ada, for
/// identical work.
///
/// leaf.hlsl grid-strides over the tile's pixels, so this is a free knob.
/// Measured (--gpu-timing, leaf+sky, 1080p; 64 -> 32):
///   spp=1   AMD 1.63 -> 1.01 ms (-38%)   NVIDIA 2.24 -> 1.38 ms (-38%)
///   spp=16  AMD 19.7 -> 11.4 ms (-42%)   NVIDIA 10.2 ->  7.6 ms (-25%)
/// i.e. a win on BOTH vendors, not an AMD-specific hack — a 64-thread group
/// reserves registers for 64 threads on Ada too, so halving it doubles the
/// blocks in flight.
///
/// 32 is a floor, not a tuning parameter: RDNA's wave is 32 lanes MINIMUM, so
/// a 16-wide group is a half-empty wave again (measured worse — 1.31 ms AMD).
/// And it never loses at other resolutions: a tile larger than 32 px simply
/// takes a second full lap, which is the same lane utilization a 64-wide group
/// would have had.
///
/// **THE GROUP WIDTH IS PAIRED WITH `render::LEAF_TILE` AND MUST MOVE WITH
/// IT.** Everything above is the reasoning at the OLD 8-px frontier, where a
/// leaf rect was ~32 px and one wave covered it. The frontier is now 32 px
/// (`render::LEAF_TILE`, whose doc carries the world measurement), a leaf rect
/// is ~540 px, and it genuinely feeds 256 lanes.
///
/// The pairing is not a preference, it is the whole effect. Measured at the
/// OLD LEAF_TILE=8 (interleaved medians, rep 1 discarded, `--spin path` 1080p
/// GPU frame span vs group 32):
/// ```text
///            g16     g64    g128    g256
///   B70 default   +0.1%   +0.9%   +6.3%  +21.1%
///   B70 stress    +1.3%   -0.4%   +1.6%   +9.5%
///   4090 stress     --    +7.6%  +23.2%     --
/// ```
/// i.e. 256 lanes at an 8-px frontier is a 21% REGRESSION — wider groups only
/// idle lanes when the tile cannot fill them, on every vendor. Take one
/// constant without the other and you get the worst of both. Intel's SIMD16
/// does not rescue g16 either, and there is still no group-only vendor default
/// to take (`main::vendor_defaults` carries mode + dxr-inline, never a group
/// width — the two leaf constants must move TOGETHER or not at all).
pub const LEAF_GROUP: u32 = 256;

/// R&D lever (FR_LGROUP): the leaf kernel's group width. Swept together with
/// FR_LEAF because the two INTERACT — a leaf rect is ~(rw*rh)/4^depth_full
/// pixels, so shrinking LEAF_TILE shrinks the tile below the group width and
/// idles lanes, which is the wave-utilization trap this constant exists to
/// document. Neither axis can be read alone. Loud on departure AND on an
/// illegal value (the FR_WIDE rule, 2026-08-01 — this lever used to revert
/// silently, so a mistyped sweep cell measured the shipping config while
/// believing it measured the lever).
pub fn leaf_group() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var("FR_LGROUP") {
        Err(_) => LEAF_GROUP,
        Ok(v) => match v.parse::<u32>() {
            Ok(n) if n.is_power_of_two() && (8..=256).contains(&n) => {
                eprintln!("gpu: FR_LGROUP={n} (default {LEAF_GROUP}) — leaf kernel group width");
                n
            }
            // Never fall back silently: a sweep that measures the shipping
            // config while believing it measured the lever is the exact
            // failure mode these levers exist to prevent.
            _ => {
                eprintln!(
                    "gpu: FR_LGROUP={v:?} is not a power of two in 8..=256 — using {LEAF_GROUP}"
                );
                LEAF_GROUP
            }
        },
    })
}

/// `--cloud-shadow N` (default ON at 16; 0 = `--no-cloud-shadow`): CELLS PER
/// CLOUD WAVELENGTH for the slab-space cloud-shadow grid.
///
/// It is deliberately NOT a grid side. The resolution the field needs is set by
/// `l0 = CLOUD_SCALE_K * diag` (cloud_cover's coarsest octave; its finest is
/// l0/2), while the AREA the grid must span is set by the shadow footprint —
/// and those are independent. Fixing the side and deriving the cell silently
/// aliases whenever the footprint grows past what the side can resolve (a low
/// sun spreads the projection); capping the cell without growing the side
/// breaks COVERAGE instead, so points fall outside and get edge-clamped. So the
/// cell is pinned to l0/this and the side is derived per frame from the
/// footprint, capped at CLOUD_SHADOW_MAX. `cloud_sun_transmittance` is EXACTLY a
/// function of the shading point's shadow-projection onto the cloud slab (see
/// trace_common.hlsli), so caching it there has no depth discontinuity to
/// filter — unlike a screen-space cache. Measured share of the cloud bill on
/// the B70: sun_transmittance 65%, sky march 26%, along_rough 5%; the cache
/// buys -21%/sample there. The domain reduction to F(M.x, M.z) is EXACT — only
/// the bilinear interpolation approximates, and the field's finest feature is
/// wider than any scene, so the shipped error is ~0. Set once at parse (the
/// `set_inline_mode` knob idiom); TraceGpu/DxrGpu SNAPSHOT it in `new()` so a
/// mid-process A/B (the gate flips the static between two constructions) can
/// never dispatch a fill for a cache its kernels don't compile.
pub static CLOUD_SHADOW: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(16);

/// Set the cloud-shadow cache resolution (cells per cloud wavelength; 0 = off).
pub fn set_cloud_shadow(n: u32) {
    CLOUD_SHADOW.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub fn cloud_shadow_n() -> u32 {
    CLOUD_SHADOW.load(std::sync::atomic::Ordering::Relaxed)
}

/// The sky fill's thread-group width, and how many groups share one SkyRec.
///
/// `cs_sky` is not the leaf kernel's twin, because a sky RECT is not tile-sized:
/// the quadtree emits it at whatever depth it proved empty, so its area is
/// (rw*rh)/4^d — a depth-2 sky tile at 1080p is 480x270 = **129,600 pixels**.
/// One group per record therefore meant ~2,025 serial grid-stride laps for that
/// one group while the rest of the machine idled. Harmless while a sky pixel was
/// a dome+disc evaluation; catastrophic once the volumetric cloud march (default
/// ON) made each sky pixel ~100x dearer.
///
/// Measured (`--spin path`, 1080p, GPU time via --gpu-timing, cloud march
/// neutralized in cs_sky ONLY so the attribution is unambiguous):
///   Arc Pro B70  default 8.95 -> 2.58 ms   stress 5000  13.07 -> 3.87 ms
///   RTX 4090     default 4.99 -> 1.65 ms   stress 5000   7.32 -> 2.84 ms
/// So ~70% of this tracer's frame time was one group serializing one rect. The
/// DXR pipeline never had the bug (DispatchRays gives every sky pixel a thread),
/// which is precisely why `--dxr` measured FASTER than `--gpu` on Arc with
/// clouds on while `--no-clouds` measured the reverse — the anomaly that found
/// this.
///
/// **LEAF_GROUP's reasoning does NOT transfer here, and the sweep says so.**
/// A leaf tile is ~32 px so one wave covers it; a sky rect is thousands of
/// pixels, so more lanes are more parallelism, not more idle lanes. Measured at
/// SKY_SPLIT 1 on the B70 default scene, `leaf+sky`: group 32 = 14.12 ms vs
/// group 64 = 6.39 — narrowing the group to "one wave" would have made this
/// kernel 2.2x WORSE. Do not "unify" the two widths.
///
/// The governing variable is the PRODUCT (pixels retired per lap); returns
/// flatten past ~4096. Full sweep, `leaf+sky` ms, `--spin path` 1080p:
/// ```text
///                    B70 default   B70 stress   NV default   NV stress
///   group  32 x   1     14.12         15.76         --           --
///   group  64 x   1      6.39          8.26         --           --
///   group 128 x   1      5.43          4.79         --           --
///   group  32 x  32       1.31          1.25        0.90         0.64
///   group  64 x  32       1.27          1.08        0.71         0.53
///   group 128 x  32       1.14          0.89        0.51         0.46
///   group  64 x 128       1.04          0.85        0.55         0.40
///   group 128 x 128       0.99          0.84        0.51         0.38
/// ```
/// 64 x 128 is taken over the nominal winner 128 x 128 (within 5-8% on every
/// row) because the two knobs fail differently: a bigger SPLIT only ever costs
/// empty groups, which retire on their first bound test and measured free,
/// while a bigger GROUP reserves registers for every lane of a kernel that
/// inlines the whole cloud march — the occupancy trap LEAF_NO_FB documents. 64
/// is also exactly one wave64 and two wave32s, so it is the safe width on every
/// vendor. The split is where the parallelism should come from.
///
/// The defines are built from these two at kernel-assembly time, and SKY_SPLIT
/// is ALSO the multiplying prep's push constant — one number, never a pair that
/// can drift.
pub const SKY_GROUP: u32 = 64;
pub const SKY_SPLIT: u32 = 128;


/// `--sky-lod K` (power of two, default ON at 4; 1 = `--no-sky-lod`): the pixel
/// pitch of the amortized cloud lattice. See sky.hlsl — the sharp half of the
/// sky integral (sun limb, stars) stays per-pixel; only the march is evaluated
/// at 1/K^2 rate and interpolated (measured 0.14% mean sky error, -9.8% frame
/// at spp=16 / -1.0% at spp=1). Snapshotted in `new()` for the same A/B reason
/// as CLOUD_SHADOW.
pub static SKY_LOD: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(4);

/// Set the sky-march lattice pitch (power of two; 1 = off).
pub fn set_sky_lod(k: u32) {
    SKY_LOD.store(k, std::sync::atomic::Ordering::Relaxed);
}

pub fn sky_lod() -> u32 {
    SKY_LOD.load(std::sync::atomic::Ordering::Relaxed)
}


/// How many of the shallowest quadtree levels run the WAVE-COOPERATIVE level
/// kernel (`cs_level_wide`, one group per tile) instead of one thread per tile.
///
/// There is a real crossover, and it is about how many lanes have work versus
/// how much work each tile is. Level d holds at most 4^d tiles, so levels 0-4
/// are <= 256 threads under the per-thread kernel — a rounding error of a
/// modern GPU's width — while each of those tiles does the MOST work of any
/// level, because a shallow frustum covers a large fraction of the screen and
/// its inherited cut has barely been refined. Deep levels are the mirror image:
/// thousands of tiles, each with a tight cut and a short descent, where a
/// private DFS wins and a whole group per tile would waste 31 lanes.
///
/// Swept, interleaved, 3 reps, medians (`--spin path` 1080p, GPU frame span).
/// The alternative shapes are in the table because both failure modes are real:
/// WIDE 0 leaves the shallow levels serial, WIDE 8 gives every deep level's
/// thousands of tiles a whole group each and collapses on Intel (B70 stress
/// 2.71, San Miguel 2.92 — worse than not doing it at all).
/// ```text
///                     WIDE 0   WIDE 6
///   B70 default        1.739    1.611    -7.4%
///   B70 stress 5000    2.897    2.017   -30.4%
///   B70 san-miguel     1.852    1.808    -2.4%
///   4090 default       1.751    1.725    -1.5%
///   4090 stress 5000   2.539    2.256   -11.1%
///   4090 san-miguel    1.802    1.385   -23.1%
/// ```
/// Never read a single sample of these rows: the B70 repeats within 0.002 ms
/// but the 4090 spread is 1.42-1.98 for one unchanged config, and a naive
/// single-shot sweep "showed" a 9-16% NVIDIA REGRESSION that three interleaved
/// reps erase completely. Same lesson the spp bench row already carries.
///
/// 6 -> 7 (2026-07-31). The old value was UNFALSIFIABLE at the resolution it
/// was measured at, which is the part worth not repeating: at 1920x1080 on the
/// shipping `LEAF_TILE` = 32, `depth_full` is 6, so the ladder runs levels 0..5
/// and `d < 6`, `d < 7`, `d < 99` are the SAME PREDICATE. Every value >= 6 is
/// one config there. The constant only bites at 4K (`depth_full` 7) and 8K (8),
/// where level 6 — and only level 6 — falls off the cooperative kernel.
///
/// THE MEASUREMENT ROUTE, because `--spin` cannot reach 4K (`W`/`H` are consts
/// and `run_spin_gpu` clamps `--lock-res` to <= 1.0): move the FRONTIER instead
/// of the resolution. `depth_full` is the smallest D with
/// max(rw,rh)/2^D <= leaf_tile(), so the two enter ONLY through that ratio, and
/// a level-d tile's frustum is a pure function of (d, camera, aspect) — its
/// rect spans rw/2^d and `ray_dir` divides by rw — with a count of <= 4^d. So
/// `FR_LEAF=16` at 1080p reproduces the 4K LADDER exactly (same levels, same
/// tile counts, same frustums) and `FR_LEAF=8` the 8K one; only the leaf/sky
/// pixel work below differs, and that is a separate timing region held fixed
/// across the A/B. Confirmed empirically: levels 0..4 measure BYTE-IDENTICAL
/// across all three frontiers, and levels 0..5 across the whole `FR_WIDE`
/// sweep — the lever moves only the level it names.
///
/// Ladder ms, 3 reps interleaved, medians (R9700 repeats within 0.3%):
/// ```text
///   FR_LEAF=16 (== 4K, levels 0..6)      WIDE 6   WIDE 7
///     R9700 default                       0.147    0.133    -9.5%
///     R9700 stress 5000                   0.229    0.178   -22.3%
///     R9700 san-miguel-lp                 0.119    0.104   -12.6%
///     4070 Ti default                     0.212    0.195    -8.0%
///     4070 Ti stress 5000                 0.362    0.276   -23.8%
///     4070 Ti san-miguel-lp               0.162    0.145   -10.5%
/// ```
/// Level 6 ALONE goes 0.029 -> 0.015 / 0.067 -> 0.016 / 0.029 -> 0.014 on the
/// R9700 (2-4x), with `leaf` unmoved to within 0.6% — the built-in control that
/// pins the win to the ladder. Frame span follows at -1.7/-4.7/-2.3%.
///
/// WHY NOT 8, measured on the same route (`FR_LEAF=8`, levels 0..7): level 7 is
/// SCENE-DEPENDENT where level 6 is not. R9700 level 7 serial -> wide reads
/// 0.119 -> 0.032 on stress but 0.018 -> 0.031 on default and 0.011 -> 0.028 on
/// san-miguel — wide LOSES on two of three. That is the crossover this constant
/// exists to name, and the mechanism is the one the doc above argues: level 7
/// holds up to 4^7 = 16384 tiles, which already fills the machine one-thread-per
/// -tile, so a whole group each only pays where the per-tile descent stays long
/// (`--stress`'s 5000 sparse objects). It is also exactly where the B70
/// collapse recorded above lives — that row was necessarily taken at the old
/// `LEAF_TILE` = 8 frontier, since 1080p/32 cannot tell 6 from 8 apart.
///
/// SO THE RISK LEDGER: at 1080p this is a PROVABLE no-op on every vendor (same
/// predicate; measured identical ladder, span within 0.5%), and it deliberately
/// leaves level 7 serial. Unmeasured: Intel at `depth_full` >= 7 — no B70 in
/// the box at the time — so a 4K Arc session is the one arm running on the
/// cross-vendor level-6 result rather than its own. Re-sweep with `FR_LEAF=16`
/// if one is available; that is the whole experiment.
pub const WIDE_LEVELS: u32 = 7;

/// R&D lever (`FR_WIDE`): the crossover LEVEL itself, not just on/off.
///
/// `--no-wide-levels` can only answer "is the cooperative kernel worth it at
/// all"; it cannot find the boundary, and the boundary is the whole design —
/// the doc above argues a crossover EXISTS and pins it with a two-point sweep
/// (0 vs 6) that never tested the levels either side of the value it shipped.
///
/// **It has now been swept, and the reason it needed to be is instructive.**
/// The 0-vs-6 table above, and an RDNA4 sweep done on top of it (which found 7
/// better than 6 by 5.6-10.1% on an R9700, by halving what was then the
/// ladder's most expensive level), were both measured when `LEAF_TILE` was 8 —
/// i.e. `depth_full` = 8 at 1080p, so levels 6 and 7 existed and were the first
/// two to fall off the wide kernel. At today's `LEAF_TILE` = 32,
/// `depth_full(1920, 1080)` is **6**: the levels are 0..5, and every value >= 6
/// is one indistinguishable config. So the shipped value was a ceiling nothing
/// reached at 1080p and the crossover under a coarse frontier was unmeasured.
///
/// That old sweep turns out to have been RIGHT and merely inapplicable: this
/// lever composed with `FR_LEAF` reproduces its exact configuration
/// (`FR_LEAF=8`), and level 6 there still measures 0.095 -> 0.041 ms wide on
/// R9700 stress — the halving it reported. What had changed was not the
/// hardware's preference but which levels the shipping frontier creates. A perf
/// constant is only as good as the tree it was measured on, and the cheapest
/// guard is to make the lever able to RECREATE the old tree: `FR_LEAF` +
/// `FR_WIDE` together span every ladder depth the renderer can produce, at one
/// resolution, with no 4K path required. See `WIDE_LEVELS` for the route and
/// the resulting 6 -> 7.
///
/// The crossover is an ABSOLUTE level, not `depth_full - 1`, and that much did
/// survive: what decides a level is its TILE COUNT (a level holds at most 4^d
/// tiles, and the private DFS starts winning once the serial kernel has enough
/// tiles to fill the machine with short descents), not its distance from the
/// leaf frontier. Deriving it from `depth_full` would make the deepest level
/// serial at every resolution, which measured -15.8% WORSE at 960x540.
///
/// 0 == `--no-wide-levels`; a value past `depth_full` makes every level wide.
/// Read by both consumers — the ExecuteIndirect ladder and the work graph's
/// `WG_WIDE_LEVELS` — so the two arms cannot disagree about the crossover.
pub fn wide_levels() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var("FR_WIDE") {
        Err(_) => WIDE_LEVELS,
        Ok(v) => match v.parse::<u32>() {
            Ok(n) => {
                eprintln!("gpu: FR_WIDE={n} (default {WIDE_LEVELS}) — wide levels 0..{n}");
                n
            }
            // Never fall back silently: a sweep that measures the shipping
            // config while believing it measured the lever is the exact
            // failure mode these levers exist to prevent.
            Err(e) => {
                eprintln!("gpu: FR_WIDE={v:?} is not a level ({e}) — using {WIDE_LEVELS}");
                WIDE_LEVELS
            }
        },
    })
}

/// The per-lane frustum traversal-stack depth (`LANE_STACK` in frustum.hlsli),
/// and with it the tracer's ONLY groupshared allocation:
/// `groupshared uint g_stack[32 * LANE_STACK]` — 8 KB/group at 64, 2 KB at 16.
///
/// **16, was 64 (2026-07-31).** RGA on gfx1201 says the kernels carrying the
/// slab are nowhere near VGPR-limited (`cs_level_wide` 54 VGPR, `cs_level` 65,
/// against `cs_leaf`'s 216), so LDS — this slab — is what caps their resident
/// groups. Quartering it roughly quadruples resident groups and the ladder
/// responds almost linearly. `--spin path` 1080p spp=1, GPU frame span, R9700
/// medians of 2 at 600 frames (spread ±0.2%), 4070 Ti medians of 3 at 1200
/// frames INTERLEAVED (spread 15-25% otherwise — never read a single sample):
/// ```text
///                        64      32      16       8      64 -> 16
///   R9700 default      0.728   0.672   0.637   0.616      -12.5%
///   R9700 stress 5000  1.066   0.974   0.903   0.861      -15.3%
///   R9700 san-miguel   0.750   0.700   0.662   0.651      -11.7%
///   R9700 powerplant   0.725   0.665   0.629   0.618      -13.2%
///   4070Ti default     0.982   0.936   0.891   0.877       -9.3%
///   4070Ti stress      1.149   1.039   0.906   0.805      -21.1%
///   4070Ti san-miguel    --      --    0.990   0.938         --
///   4070Ti powerplant  0.921     --    0.774   0.802      -16.0%
/// ```
/// The R9700 san-miguel row is a RE-MEASUREMENT: the first pass read
/// 1.429/1.388/1.347/1.333 because the AMD candidate-loop TMin defect
/// (`rt.hlsli::cand_tmin`) was still live and cost that scene ~2x. Every AMD
/// number taken on a TEXTURED scene before that fix is contaminated the same
/// way; the untextured rows are unaffected (no candidate loop compiles).
/// The `leaf` region is the built-in control — it pastes neither frustum.hlsli
/// nor the slab — and on the R9700 it does not move at any setting (default
/// 0.501/0.506/0.500/0.500), which is what pins the win to LDS occupancy and
/// not to some second effect.
///
/// WHY 16 AND NOT 8, WHICH IS FASTER ON 5 OF THE 8 CELLS ABOVE. A shorter stack
/// coarsens the bound (the pressure arm folds the node in as `best = d`), so
/// t_start shrinks and leaf rays traverse more. On the R9700 that costs
/// ~nothing — AMD re-origins the ray at TMin, so the inherited bound is already
/// "free and worth nothing" there — but on the 4070 Ti TMin really prunes, and
/// on POWERPLANT (12.8M tris, the case where the bound has the most to prune)
/// 8 sends `leaf` 0.579 -> 0.637 (+10.0%), which outweighs the extra 0.020 ms
/// of ladder and makes 8 a net 3.6% REGRESSION there. 16 is the largest step
/// that never regresses any measured (vendor, scene) pair. The occupancy-vs-
/// pruning trade is vendor-asymmetric *because the value of the inherited bound
/// is*, and the crossover moves with the leaf frontier: on the pre-`LEAF_TILE`
/// 8 -> 32 tree the same wall sat between 32 and 16 (16 measured +44% leaf on a
/// 4070 Ti); the coarse frontier made the leaf kernel less TMin-sensitive and
/// moved it down one notch. Expect it to move again — always sweep BOTH vendors
/// and score `leaf` as well as the ladder.
///
/// THE B70 ROW (2026-08-01, `--spin path` 1080p, 2 reps repeating to ±0.002):
/// Intel wants 8 OUTRIGHT — monotone 8 < 16 < 32 on span AND ladder across
/// all four scenes, with NO leaf regression anywhere:
/// ```text
///                 span@8   span@16  span@32   ladder@8/16   leaf@8/16
///   default        0.599    0.635    0.694    0.074/0.111   0.418/0.418
///   stress 5000    0.701    0.774    0.889    0.129/0.200   0.461/0.462
///   san-miguel-lp  0.770    0.783    0.831    0.064/0.076   0.613/0.612
///   powerplant     0.611    0.643    0.696    0.064/0.083   0.449/0.461
/// ```
/// (powerplant's leaf IMPROVES at 8 — the inherited bound is near-worthless
/// on Arc, the t_start-ablation verdict again, so the occupancy side wins
/// with nothing to trade.) The 16 default therefore survives on the 4070
/// Ti's powerplant regression ALONE; an Intel-keyed 8 (the `cand_defs`
/// vendor-compile precedent) is the measured, unshipped follow-on.
///
/// R&D lever (`FR_LSTACK`), a power of two in 8..=64. **64 is a hard ceiling**:
/// refine_cut emits its surviving cut into one 64-u32 `cut_pool` slot, and its
/// `olen + wlen <= LANE_STACK` invariant is what keeps that write in bounds.
/// Going lower is always SOUND — a smaller t_start can only trace MORE rays,
/// never miss a hit — and gated: `--check-gpu` passes at 8 and 16 on both
/// vendors with every exact-zero counter at 0, and NVIDIA stays BIT-EXACT
/// (`max rel t err 0.00e0`, same-seed image mean 0.00e0). It is not
/// image-neutral on AMD, which re-origins at TMin — a 1-2 ulp shift in reported
/// t can flip a grazing occlusion bit, the documented class. It also shrinks
/// the wide kernel's frontier, `WQ_CAP = 16 * LANE_STACK`, which keeps its 16x
/// headroom over `cut_len <= LANE_STACK` at any setting — that ratio is why the
/// seed loop's silent-drop arm stays unreachable, so scale the two together if
/// either ever moves.
pub fn lane_stack() -> u32 {
    const DEFAULT: u32 = 16;
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var("FR_LSTACK") {
        Err(_) => DEFAULT,
        Ok(v) => match v.parse::<u32>() {
            Ok(n) if n.is_power_of_two() && (8..=64).contains(&n) => {
                eprintln!("gpu: FR_LSTACK={n} (default {DEFAULT}) — {} B/group", 32 * n * 4);
                n
            }
            _ => {
                eprintln!(
                    "gpu: FR_LSTACK={v:?} is not a power of two in 8..=64 — using {DEFAULT}"
                );
                DEFAULT
            }
        },
    })
}

/// The serial traversal-stack LAYOUT lever (`FR_STACK_LAYOUT=lane|depth`).
/// DEFAULT = DEPTH-major (2026-08-09): `g_stack[sp * 32 + lane]` puts a
/// wave's simultaneous accesses at one depth in consecutive SLM words —
/// conflict-free on Xe2's 16 banks x 4 B — where the v1 lane-major layout
/// (`lane * LANE_STACK + sp`) strode lanes LANE_STACK*4 B apart, a multiple
/// of 64 B at every legal FR_LSTACK: the textbook 16-way bank serialization
/// on every push/pop (Intel oneAPI guide, SLM banking). MEASURED A WASH on
/// an Arc Pro B70 everywhere the serial path runs — `--no-wide-levels`
/// serial ladders (default/stress spans within ±0.006) and the hemi-gi
/// bench (±0.2%) — because these kernels are global-memory-latency-bound:
/// every stack op sits beside a BvhNode/FtNode fetch, and the LDS
/// serialization hides entirely under it. Depth-major ships anyway as the
/// no-downside conflict-free form (bit-identical by construction — a pure
/// address remap — so unlike the gw_* lesson there is no behavior to
/// regress); `lane` restores v1 for the A/B. Loud on departure AND on an
/// illegal value (the FR_WIDE rule).
pub fn stack_layout_def() -> &'static str {
    static S: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    *S.get_or_init(|| match std::env::var("FR_STACK_LAYOUT") {
        Err(_) => "",
        Ok(v) if v == "depth" => "",
        Ok(v) if v == "lane" => {
            eprintln!(
                "gpu: FR_STACK_LAYOUT=lane — v1 lane-major g_stack (16-way SLM bank \
                 conflicts on Xe2; the A/B arm)"
            );
            "\n#define STACK_LANE_MAJOR 1"
        }
        Ok(v) => {
            eprintln!("gpu: FR_STACK_LAYOUT={v:?} unrecognized (legal: lane, depth) — using depth");
            ""
        }
    })
}

/// The `--no-wide-levels` A/B lever. When false, every quadtree level runs the
/// one-thread-per-tile `cs_level` (the pre-cooperative ladder), so the feature
/// can be measured against its own absence — the codebase's standard perf A/B.
/// Read per level at dispatch; `pso_level_wide` is still built (cheap, and it
/// keeps the lever a runtime toggle rather than a rebuild). The wave-cooperative
/// kernel is bit-identical to the serial one, so this is a pure-perf lever with
/// no correctness gate of its own.
pub static WIDE_LEVELS_ON: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Quadtree depth to the leaf frontier: smallest D with
/// max(rw, rh) / 2^D <= LEAF_TILE (temporal.rs uses the same formula).
pub fn depth_full(rw: u32, rh: u32) -> u32 {
    let m = rw.max(rh) as u64;
    let mut d = 0;
    let mut s = crate::render::leaf_tile() as u64;
    while s < m {
        s *= 2;
        d += 1;
    }
    d
}

/// --continuation-rays / --sw-rays: the wavefront tracer's rays traverse the
/// SOFTWARE BVH
/// (bvh.rs's loops, ported to rt_sw.hlsli) instead of DXR inline RayQuery,
/// so primary leaf rays can seed from the tile's inherited node cut — the
/// one product of the frustum recursion the RayQuery API structurally cannot
/// accept. Leaf records expose only an opaque TraversalFrontier token; the
/// software provider is solely responsible for decoding it. Default OFF; the
/// off arm assembles the exact pre-lever kernel sources (the --dxr-inline 0
/// pattern, modulo the blank lines empty define segments join as — the defs
/// block's existing tolerance). Wavefront only: the DXR pipeline never reads
/// it. Set once at parse (the `set_inline_mode` knob idiom).
pub static SW_RAYS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn sw_rays() -> bool {
    SW_RAYS.load(std::sync::atomic::Ordering::Relaxed)
}

/// The leaf CUT consumption arm: --sw-rays composed with the cut-rays lever
/// exactly as the CPU's `intersect_multi` short-circuit is (`--no-cut-rays`
/// keeps software traversal but seeds from the root). One predicate, four
/// consumers in lockstep: the SW_RAYS_LEAF define (leaf + wavefront units),
/// the ft_bnode upload, its ladder t1 binding, and the cut-pool headroom.
pub fn sw_rays_leaf() -> bool {
    sw_rays() && crate::bvh::CUT_SEED_RAYS.load(std::sync::atomic::Ordering::Relaxed)
}


pub const SMOKE_HLSL: &str = include_str!("../shaders/smoke.hlsl");
/// Order-2 SH irradiance, standalone (no cbuffer of its own — the coefficients
/// are a parameter). pub(crate) because gpu/ffx_rr.rs prepends it to the FSR
/// composite pass, which needs the SAME evaluator but binds its own 9 rows.
pub const SH_HLSLI: &str = include_str!("../shaders/sh.hlsli");
/// The FSR plane wire encodings (octahedral normals + their 10-bit quantum).
/// pub(crate) for the same reason: feed.hlsl writes those planes and
/// fsr_composite.hlsl reads them back, and the composite identity is exactly
/// the claim that the two agree — so they share one copy.
pub const FSR_WIRE_HLSLI: &str = include_str!("../shaders/fsr_wire.hlsli");
// pub(crate): gpu/dxr.rs pastes the same prelude/shading/resolve sources
// into its DXR library (the kernels are single-sourced on disk).
//
// sh.hlsli leads: trace_common's `sh_irradiance` is just the frame's cbuffer
// bound to it. Folding it in here rather than at each of the dozen concat sites
// keeps the prelude one name.
pub const TRACE_COMMON_HLSLI: &str = concat!(
    include_str!("../shaders/sh.hlsli"),
    "\n",
    include_str!("../shaders/trace_common.hlsli")
);
pub const CTR_HLSLI: &str = include_str!("../shaders/ctr.hlsli");
#[cfg(test)]
pub const CONTINUATION_HLSLI: &str = include_str!("../shaders/continuation.hlsli");
// Every queue consumer gets the opaque-handle ABI before LeafRec declares
// TraversalFrontier. Keeping this one concatenation site prevents a shader
// unit from accidentally compiling a different producer/consumer contract.
pub const QUEUES_HLSLI: &str = concat!(
    include_str!("../shaders/continuation.hlsli"),
    "\n",
    include_str!("../shaders/queues.hlsli")
);
pub const FRUSTUM_HLSLI: &str = include_str!("../shaders/frustum.hlsli");
// The 8-wide frustum tree's bound_query/refine_cut, `#ifdef FTREE`-guarded —
// pasted right after FRUSTUM_HLSLI (whose binary halves are `#ifndef FTREE`);
// the ftree_defs prelude picks the structure per session.
pub const FTREE_HLSLI: &str = include_str!("../shaders/ftree.hlsli");
/// pub for gpu/dxr.rs: FR_DXR_INLINE pastes the inline-RayQuery primitives
/// into the DXR library in place of rt_dxr.hlsli's TraceRay flavors.
pub const RT_HLSLI: &str = include_str!("../shaders/rt.hlsli");
/// bvh.rs's traversal loops in HLSL — pasted IN PLACE of RT_HLSLI into every
/// ray-shooting wavefront unit when --sw-rays is armed (same three primitive
/// signatures + the opaque trace_closest_frontier), so rays traverse the
/// software BVH and leaf primaries seed from the tile's node cut.
pub const RT_SW_HLSLI: &str = include_str!("../shaders/rt_sw.hlsli");
/// The water ripple FIELD, pasted immediately ahead of SHADE_HLSLI at every
/// site (shade.hlsli's `ripple_normal` calls into it). Its own file because
/// the fxc frame-generation guide kernel pastes it too — see ngxfg_guides.rs.
pub const RIPPLE_HLSLI: &str = include_str!("../shaders/ripple.hlsli");
pub const SHADE_HLSLI: &str = include_str!("../shaders/shade.hlsli");
pub const HEMI_HLSLI: &str = include_str!("../shaders/hemi.hlsli");
pub const REFERENCE_HLSL: &str = include_str!("../shaders/reference.hlsl");
pub const RESOLVE_HLSL: &str = include_str!("../shaders/resolve.hlsl");
pub const WAVEFRONT_HLSL: &str = include_str!("../shaders/wavefront.hlsl");
pub const SKY_HLSL: &str = include_str!("../shaders/sky.hlsl");
pub const SKYLOD_HLSLI: &str = include_str!("../shaders/skylod.hlsli");
pub const LEAF_HLSL: &str = include_str!("../shaders/leaf.hlsl");
pub const HEMI_WAVE_HLSL: &str = include_str!("../shaders/hemi_wave.hlsl");
pub const HEMI_LEAF_HLSL: &str = include_str!("../shaders/hemi_leaf.hlsl");
pub const COMPOSE_HLSL: &str = include_str!("../shaders/compose.hlsl");
pub const FEED_HLSL: &str = include_str!("../shaders/feed.hlsl");
pub const NPPD_HLSL: &str = include_str!("../shaders/nppd.hlsl");
pub const NRD_BRIDGE_HLSL: &str = include_str!("../shaders/nrd_bridge.hlsl");
pub const WAVEPROBE_HLSL: &str = include_str!("../shaders/waveprobe.hlsl");
pub const WORKGRAPH_HLSL: &str = include_str!("../shaders/workgraph.hlsl");

// The DXR pipeline's own three units. They live here with the rest of the
// corpus because the shader-source gates below pin rt.hlsli and rt_dxr.hlsli
// AGAINST EACH OTHER — the relief interval contract has to hold in both
// intersectors — and a pin that can only see one of its two subjects is not
// a pin. `dxr.rs` reads them through `trace::`, as it always did.
pub const RT_DXR_HLSLI: &str = include_str!("../shaders/rt_dxr.hlsli");
pub const DXR_HLSL: &str = include_str!("../shaders/dxr.hlsl");
pub const DXR_SHADE_HLSL: &str = include_str!("../shaders/dxr_shade.hlsl");

/// Per-scene HLSL prelude: alpha-masked scenes compile the cutout candidate
/// loops / any-hit shaders in; opaque scenes compile byte-identical sources
/// to the pre-cutout tracer. Shared with dxr.rs (the DXR library concat).
pub fn alpha_defs(scene: &Scene) -> &'static str {
    if scene.any_alpha && !abl_has("noalpha") { "#define ALPHA_CUTOUT 1" } else { "" }
}

/// Empty-scene compile-time guard for every shader unit that can consume the
/// binary software BVH. The CPU keeps a count-zero sentinel root so buffers
/// remain non-empty, but that shape is indistinguishable from an internal
/// node to HLSL; these consumers must return their clear-space identities
/// before reading it.
pub fn empty_defs(scene: &Scene) -> &'static str {
    if scene.indices.is_empty() { "#define SCENE_EMPTY 1" } else { "" }
}

/// The relief twin of `alpha_defs`: height-carrying scenes compile the march
/// + candidate loops / any-hit shaders in (runtime-gated by FLAG_HEIGHT —
/// the V toggle); scenes without height data compile byte-identical sources
/// to the pre-relief tracer. Shared with dxr.rs.
pub fn height_defs(scene: &Scene) -> &'static str {
    if scene.any_height && crate::bvh::height_armed() && !abl_has("noheight") {
        "#define HEIGHTFIELD 1"
    } else {
        ""
    }
}

/// The tinted-shadows twin: transmissive scenes compile `transmit_q`'s
/// candidate loop / the ah_shadow tint arm in (`Scene::any_transmissive`
/// already folds the `--no-tinted-shadows` lever); scenes without
/// transmissive materials compile byte-identical sources to the binary
/// occlusion tracer. Shared with dxr.rs.
pub fn trans_defs(scene: &Scene) -> &'static str {
    if scene.any_transmissive && !abl_has("notrans") { "#define TRANS_SHADOW 1" } else { "" }
}

/// `#define RTGI` — compiles shade_full's real-time-GI bounce block in (the
/// trans_defs session-lever pattern). `--no-rtgi` omits it, so the unarmed
/// assembly's shade_full is the verbatim pre-RTGI call — the bit-identity
/// arm. Reaches every unit that pastes SHADE_HLSLI on BOTH pipelines (the
/// probe-reach rule); the per-frame fb stand-down rides FLAG_RTGI on top.
pub fn rtgi_defs() -> &'static str {
    if crate::shade::rtgi_enabled() { "#define RTGI 1" } else { "" }
}

/// The AMD candidate-loop TMin workaround — see `rt.hlsli::cand_tmin` for the
/// bug, the evidence, and why the fix is shaped this way. This is the gate.
///
/// Vendor-keyed, and deliberately NOT routed through `main::vendor_defaults`:
/// that table is for PREFERENCES (which render mode a vendor starts in) and its
/// bar is "measure it or leave it out". This is a correctness workaround for a
/// driver defect; it keys off the PICKED adapter rather than a `--prefer-*`
/// request that can fall back, and it costs nothing to arm on a scene that
/// cannot hit the bug — the helper it feeds is `height_tmin` verbatim unarmed,
/// and the FORCE_OPAQUE arms it never reaches compile byte-identically.
///
/// `FR_ABL=nocandtmin` disarms it, which reproduces the bug on demand: that is
/// the A/B that produced the +TMin evidence, and it is the first thing to run
/// if a future driver claims to have fixed this.
///
/// SCOPE, and why the DXR pipeline deliberately does NOT take it (dxr.rs builds
/// its own define list and omits this one). The error is +TMin, so it only
/// matters where TMin is materially large — and the ONLY such rays in the tree
/// are the wavefront's leaf primaries, which pass the tile's inherited t_start.
/// Every other candidate-loop ray (shadow, AO, the reflection/glass
/// continuations, DXR's inline secondaries) passes an eps-scale tmin, so the
/// offset is eps-scale too, and the hardware still culls below TMin correctly,
/// so nothing is mis-classified. Measured, not assumed: `--check-dxr
/// --prefer-amd` on san-miguel-low-poly reads `max rel t err 8.70e-6` — clean —
/// against the wavefront's 8.70e-1 on the same scene and adapter. Arming DXR
/// would make every secondary enumerate candidates from 0 instead of eps, a
/// real cost on AMD's DEFAULT render mode, to fix nothing measurable.
///
/// Takes the vendor of the device the kernels are being built FOR, never the
/// process-global `picked_vendor()`: under `--dual-gpu` two devices are live and
/// only one of them may be the AMD one. Arming this on the wrong device is not a
/// missed optimization — it is the `tmin-overshoot` defect restored on every
/// leaf primary of the device that does not need it.
pub fn cand_defs(vendor: Vendor) -> &'static str {
    if vendor == Vendor::Amd && !abl_has("nocandtmin") {
        "#define CAND_TMIN0 1"
    } else {
        ""
    }
}

/// Does the BLAS have to drop `GEOMETRY_FLAG_OPAQUE`?
///
/// Exactly when some conditional-hit feature compiled in — so it is DERIVED
/// from the three predicates above rather than re-deriving `scene.any_*`. Both
/// tracers used to spell this out separately (trace.rs's AS sizing and
/// dxr.rs's library defs), which meant the flag and the shaders could drift,
/// and an `FR_ABL` neutralization would have compiled the candidate loops out
/// while still building a non-opaque AS — measuring nothing.
///
/// It also states the real invariant: the AS flag and the shader arms are two
/// halves of one decision. The `bc7::should_compress` discipline — two
/// predicates that MUST stay identical are better written once.
pub fn non_opaque(scene: &Scene) -> bool {
    !alpha_defs(scene).is_empty()
        || !height_defs(scene).is_empty()
        || !trans_defs(scene).is_empty()
}

/// Is `tag` present in `FR_ABL`? Read once — the ablations are session
/// constants, and a per-call `env::var` inside a predicate that runs per
/// chunk would be a syscall per BLAS.
pub fn abl_has(tag: &str) -> bool {
    static ABL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ABL.get_or_init(|| std::env::var("FR_ABL").unwrap_or_default()).contains(tag)
}

/// Every FR_ABL tag any GPU-side `abl_has` consumer recognizes — the announce
/// line's vocabulary. LOCKSTEP: an `abl_has("x")` call added anywhere in the
/// GPU pipelines without appending "x" here makes the announce report
/// "matched GPU arms: (none)" for it — a false ALARM, deliberately the loud
/// direction (the operator investigates a working arm instead of trusting a
/// dead one), but keep the list current.
pub const ABL_GPU_TAGS: &[&str] = &[
    "sunt", "rough", "nogbuf", "nopack", "nowave", "wavegw", "noelcull", "noffcode", "noelcode",
    "oldcut", "nobatch", "nocandtmin", "noalpha", "noheight", "notrans", "tzero", "noshadow",
    "noao", "norefl", "noglass", "nogi", "nosec",
];

/// One loud line per process when FR_ABL is set at all — the GPU twin of
/// `shade::abl`'s and `emissive::cull_abl`'s announces. The GPU side printed
/// NOTHING until 2026-08-01: an unmatched tag (typo, or an arm that reaches
/// no unit) ran the shipping config while the operator believed otherwise —
/// the silent-A/B failure the CPU announces already guard against, and
/// "matched GPU arms: (none)" on a non-empty FR_ABL IS the probe-reach alarm.
/// Called from TraceGpu::new and DxrGpu::new right after require_caps —
/// deliberately NOT from inside abl_has's OnceLock init (calling abl_has from
/// there re-enters the lock, and the first abl_has call today comes from the
/// BLAS predicates with non-obvious timing). Substring semantics untouched —
/// no tokenizer, matching `abl_has` exactly.
pub fn abl_announce() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let raw = std::env::var("FR_ABL").unwrap_or_default();
        if raw.is_empty() {
            return;
        }
        let matched: Vec<&str> = ABL_GPU_TAGS.iter().copied().filter(|t| abl_has(t)).collect();
        let list =
            if matched.is_empty() { "(none)".to_string() } else { matched.join(",") };
        eprintln!("FR_ABL (gpu): {raw:?} — matched GPU arms: {list}");
    });
}

/// FR_WIDTH=1 — arm the in-kernel wave-width report: each real kernel writes
/// its COMPILED WaveGetLaneCount() to a WIDTH_PROBE counter slot (>=
/// CTR_COUNT, so no zero loop or gate ever touches it). This exists because
/// `wave_probe` deliberately measures a TRIVIAL kernel per group width, while
/// the driver picks SIMD width PER SHADER from register pressure — the real
/// kernels' widths are the visible half of the occupancy story (Xe2: 16 vs
/// 32). Loud on arm AND on an unrecognized value (the FR_WIDE rule — a
/// silent no-op walk is the failure mode env levers exist to prevent).
pub fn width_probe_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FR_WIDTH") {
        Err(_) => false,
        Ok(v) if v == "1" => {
            eprintln!(
                "gpu: FR_WIDTH=1 — in-kernel wave-width report armed (writes \
                 counter slots >= CTR_COUNT only; no gate reads them)"
            );
            true
        }
        Ok(v) => {
            eprintln!("gpu: FR_WIDTH={v:?} unrecognized (legal: 1) — off");
            false
        }
    })
}

/// The WIDTH_PROBE defines, "" when off (callers push CONDITIONALLY — an
/// empty join element would prepend a blank line and break the unarmed
/// sources' byte-identity, the dxr.rs feed-unit lesson). Carries the
/// CTR_W_REFERENCE literal because the reference unit pastes no ctr.hlsli;
/// units that DO paste it see an identical object-like redefinition, legal
/// HLSL (the noelcode precedent).
pub fn width_defs() -> String {
    if width_probe_on() {
        format!("#define WIDTH_PROBE 1\n#define CTR_W_REFERENCE {CTR_W_REFERENCE}u")
    } else {
        String::new()
    }
}

/// FR_BALLAST=N | dxr:N — inject N synthetic LIVE floats into cs_reference
/// (bare N) or the mode-2 DXR raygen (`dxr:N` — dxr.hlsl's DXR_INLINE_SEC==2
/// arm, needs --dxr-inline 2; dxr.rs refuses other modes loudly). The
/// register-cliff bisection lever; see reference.hlsl's liveness argument,
/// mirrored term-for-term in the raygen arm. The two targets carry the SAME
/// code at the SAME compiled width (FR_WIDTH: both SIMD16 on the B70), so
/// sweeping both knees on one scene measures the RT launch regime's
/// confiscated live state IN FLOATS — the knee-vs-knee host comparison.
/// Legal 1..=256, one target per run; anything else is loud + off. Image
/// bit-identical at every real spp (the fold is branch-dead at runtime); a
/// PROBE, never a lever — pair with FR_WIDTH=1 and sweep N to find the
/// width flip / ms step.
pub fn ballast_parsed() -> (u32, u32) {
    static N: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let Ok(v) = std::env::var("FR_BALLAST") else { return (0, 0) };
        let (target, num) = match v.strip_prefix("dxr:") {
            Some(rest) => ("the mode-2 DXR raygen", rest),
            None => ("cs_reference", v.as_str()),
        };
        match num.parse::<u32>() {
            Ok(n) if (1..=256).contains(&n) => {
                eprintln!(
                    "gpu: FR_BALLAST={v} — {n} synthetic live floats in \
                     {target} (register-cliff probe; image bit-identical)"
                );
                if target == "cs_reference" { (n, 0) } else { (0, n) }
            }
            _ => {
                eprintln!("gpu: FR_BALLAST={v:?} illegal (N or dxr:N, N in 1..=256) — off");
                (0, 0)
            }
        }
    })
}

/// The bare-N arm: ballast in cs_reference (0 under `dxr:N`).
pub fn ballast_ref_n() -> u32 {
    ballast_parsed().0
}

/// The `dxr:N` arm: ballast in the mode-2 raygen (0 under bare N).
pub fn ballast_dxr_n() -> u32 {
    ballast_parsed().1
}

/// The BALLAST_N define, "" when off (pushed conditionally, like width_defs).
pub fn ballast_defs() -> String {
    match ballast_ref_n() {
        0 => String::new(),
        n => format!("#define BALLAST_N {n}u"),
    }
}

/// The raygen twin — consumed only by dxr.rs's library assembly.
pub fn ballast_dxr_defs() -> String {
    match ballast_dxr_n() {
        0 => String::new(),
        n => format!("#define BALLAST_N {n}u"),
    }
}

/// FR_WAVEVIZ=1|chs — arm the wave-footprint visualization: every wave takes
/// a TICKET (one atomic bump per wave, broadcast to its lanes), each covered
/// kernel stores its wave's ticket per pixel (hijacking tbuf as asfloat bits —
/// nothing consumes tbuf in a live GPU frame), and the resolve stage hashes
/// ticket→color under FLAG_WAVEVIZ, making wave footprints literally visible
/// (compact tiles vs scattered confetti — the DispatchRays launch-packing
/// question). `chs` = the mode-1 closest-hit variant: chs_shade takes the
/// ticket instead of the raygen, so the picture shows whether the driver
/// REPACKS waves between launch and hit shading (the TSU acting). Armed
/// sessions toggle the overlay live with the I key (display-stage only — no
/// resets); headless --spin runs dump waveviz-<arm>.png + compactness stats.
/// Unarmed sessions stay byte-identical (conditional defs pushes). Loud on
/// arm and on an unrecognized value (the FR_WIDE rule).
pub fn waveviz_on() -> bool {
    WAVEVIZ_MODE.load(std::sync::atomic::Ordering::Relaxed) != 0
}

/// The `--waveviz chs` sub-mode (implies `waveviz_on`).
pub fn waveviz_chs() -> bool {
    WAVEVIZ_MODE.load(std::sync::atomic::Ordering::Relaxed) == 2
}

/// 0 = off, 1 = on, 2 = chs. Written ONCE by main's lever block (the CLI
/// `--waveviz [chs]` wins over the `FR_WAVEVIZ` env alias) BEFORE any GPU
/// construction — Passes::new, both tracers' kernel assemblies, and the
/// spin arms all read it through the accessors above.
pub static WAVEVIZ_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn set_waveviz(mode: u8) {
    WAVEVIZ_MODE.store(mode, std::sync::atomic::Ordering::Relaxed);
    match mode {
        1 => eprintln!(
            "gpu: waveviz — wave-ticket overlay armed (I toggles live in GPU \
             arms, every upscaler included; tbuf carries tickets while ON, so \
             C-verify stands down)"
        ),
        2 => eprintln!(
            "gpu: waveviz chs — closest-hit wave tickets armed (mode-1 DXR \
             only; shows the driver's hit-stage packing)"
        ),
        _ => {}
    }
}

/// The `FR_WAVEVIZ` env alias (scripts): parsed by main's lever block when
/// the CLI flag is absent. Loud on an illegal value (the FR_WIDE rule).
pub fn waveviz_env() -> u8 {
    match std::env::var("FR_WAVEVIZ") {
        Err(_) => 0,
        Ok(v) if v == "1" => 1,
        Ok(v) if v == "chs" => 2,
        Ok(v) => {
            eprintln!("gpu: FR_WAVEVIZ={v:?} unrecognized (legal: 1 | chs) — off");
            0
        }
    }
}

/// The live overlay switch inside an armed session (the I key; headless spin
/// arms it unconditionally). Read at CB-build time like the V toggle.
pub static WAVEVIZ_LIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn waveviz_live() -> bool {
    WAVEVIZ_LIVE.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_waveviz_live(on: bool) {
    WAVEVIZ_LIVE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The WAVEVIZ define, "" when off (pushed conditionally — the width_defs
/// rule: never an empty join element). WAVEVIZ_CHS rides along only for the
/// DXR library (dxr.rs pushes it per mode); compute units take WAVEVIZ alone.
/// No injected literals: the wave IDs are position-keyed pure math.
pub fn waveviz_defs() -> String {
    if waveviz_on() {
        "#define WAVEVIZ 1".to_string()
    } else {
        String::new()
    }
}

/// One line from a CTR_TOTAL-sized counter readback. 0 = that kernel never
/// ran this session (e.g. hemi without an H cycle, level under replay).
pub fn format_width_report(c: &[u32]) -> String {
    format!(
        "width (gpu): leaf={} sky={} level={} hemi={} reference={}",
        c[CTR_W_LEAF as usize],
        c[CTR_W_SKY as usize],
        c[CTR_W_LEVEL as usize],
        c[CTR_W_HEMI as usize],
        c[CTR_W_REFERENCE as usize]
    )
}

/// The `--blas-split` twin: armed sessions compile `tri_of()` as the chunk
/// remap (`blas_tri[chunk_base[inst] + prim]`), unarmed ones as the identity,
/// so every unarmed kernel is byte-identical to the pre-feature source. A
/// SESSION constant (the lever is read once, before any tracer is built), not
/// a per-scene one — hence no `scene` argument.
pub fn blas_defs() -> &'static str {
    if crate::blas_split::max_prims().is_some() { "#define BLAS_SPLIT 1" } else { "" }
}

/// The sway-MV twin of `blas_defs`: sessions whose UPLOADED scene armed the
/// animated-TLAS ring compile the prev-pose MV correction into
/// `gbuf_write_hit` (+ `HitInfo::inst`); every other session — no foliage,
/// `--no-foliage-sway`, `--no-blas-split` — compiles byte-identical sources
/// to the pre-feature tracer. `armed` is the UPLOADED ring's existence — both
/// call sites pass `scene_gpu.sway.is_some()` and neither re-derives the
/// partition/lever/split chain, because the define and the ring it reads have
/// to be ONE decision (the `non_opaque` discipline). It takes the bool rather
/// than the `SceneGpu` only so this module stays portable; the fact is still
/// read from the single place that owns it. The wavefront additionally
/// suppresses it under `--sw-rays` (the software rays render the REST pose —
/// sway MVs would describe motion that is not on screen); DXR never
/// software-rays and takes the predicate verbatim.
pub fn sway_defs(armed: bool) -> &'static str {
    if armed { "#define SWAY_MV 1" } else { "" }
}

/// Dev cost-attribution ablations (`FR_ABL=sunt|rough|...`): neutralize one
/// consumer at a time to find where a layer's cost actually lands. NOT shipping
/// levers — every one of them changes the image, which is the point (you are
/// measuring a feature against its own absence, the `--no-wide-levels` idiom).
///
/// Shared with dxr.rs deliberately. This used to be read inline in
/// `TraceGpu::new` only, so the DXR library — which builds its own defs — never
/// saw an ablation at all, and any cost attribution was silently
/// non-comparable across the two arms. That comparison is exactly what an
/// Intel campaign turns on (the wavefront-vs-DXR default), so the string has
/// one home now.
/// Tags that neutralize a cloud consumer. The `no*` tags are NOT here — they
/// suppress a `#define` rather than adding one, so they live in the three
/// per-scene predicates above (and, through `non_opaque`, in the BLAS flag).
pub fn abl_defs() -> String {
    let mut out = String::new();
    for (tag, def) in [
        ("sunt", "ABL_SUNT"),
        ("rough", "ABL_ROUGH"),
        // The G-buffer pack's two halves, priced separately. `nogbuf`
        // neutralizes the WRITE (gbuf_write_hit/_sky store nothing, so leaf
        // and sky stop moving 88 B/px); `nopack` neutralizes the READ
        // (cs_feed_xess skips the load and writes constants). Both leave the
        // dispatch shape untouched, so the delta is memory traffic and
        // nothing else. Under `nogbuf` the feed reads uninitialized memory —
        // the image is garbage BY DESIGN; these are cost probes, never levers.
        ("nogbuf", "ABL_NOGBUF"),
        ("nopack", "ABL_NOPACK"),
        // Everything back to one LDS/global atomic per lane per candidate —
        // since the 2026-08-09 gw flip this means ctr.hlsli's ctr_add/
        // ctr_bump only (the still-wave-aggregated half; the frontier's
        // gw_alloc/gw_min_bits are plain-atomic BY DEFAULT now, so for them
        // nowave is a no-op that also OVERRIDES a simultaneous wavegw).
        // `FR_ABL=nobatch,nowave` remains the full pre-wave-pass queue code.
        //
        // DUAL-HOMED (2026-08-01): this row reaches ctr.hlsli's consumers
        // (leaf/sky/hemi/reference, via `defs`), and wavefront_ablation_defs
        // emits the SAME define for the tile unit (which also pastes
        // ctr.hlsli, and whose gw guard reads it as the override above). The
        // consumers straddle the shader-cache split, so nowave is the one arm
        // that legitimately churns BOTH caches when flipped. Until the
        // dual-homing the tile unit never saw the define at all, and every
        // recorded nowave A/B — the "wave aggregation measured neutral"
        // verdict included — compared a HALF-ARMED configuration (leaf-side
        // neutralized, tile-side still wave-cooperative): the probe-reach
        // trap's third instance, from inside the very comment that warned
        // about it. Fully armed, the A/B REVERSED the gw half's verdict —
        // which is why the default flipped; see wavefront.hlsl's gw block.
        // Unlike its neighbours this is a PERF arm, not a cost probe — both
        // sides publish identical counter totals and an identical `best`, so
        // the image is unmoved.
        ("nowave", "ABL_NO_WAVE_OPS"),
        // The leaf kernel's per-tile emissive light cull back to the full
        // mask (leaf.hlsl's el_mask block; emissive::cull_abl is the CPU
        // twin of the same FR_ABL tag). BIT-IDENTICAL by construction — a
        // culled light fails every pixel's own d2 >= r_infl2 test — so this
        // is a pure cost probe like nowave, never image-changing. Lives HERE
        // because it reaches leaf.hlsl (the probe-reach rule).
        ("noelcull", "ABL_NO_EL_CULL"),
        // REGISTER-PRESSURE PROBES (2026-08-01, the B70 campaign): compile
        // the CODE out, not just the execution. Fireflies/emissive are
        // runtime-flag branches present in every shading kernel; a day or
        // unarmed frame executes none of it, but register allocation is the
        // MAX over both arms (the LEAF_NO_FB lesson — splitting ONE such
        // branch measured -11% AMD / -16% NVIDIA). The A/B against "flag
        // off, code present" therefore isolates pure occupancy cost: same
        // execution, different allocation. Every guarded block draws ZERO
        // rng, so armed same-seed A/Bs stay exact — both kernels lose
        // identical code. Image-changing wherever the feature is LIVE
        // (night fireflies, armed emissive) — cost probes, never levers.
        ("noffcode", "ABL_NO_FF_CODE"),
        ("noelcode", "ABL_NO_EL_CODE"),
        // noelcode SUBSUMES noelcull: leaf.hlsl's cull hoist already
        // compiles out under ABL_NO_EL_CULL, so emit that too rather than
        // teaching the shader a compound guard. `noelcull,noelcode` together
        // emits ABL_NO_EL_CULL twice — an identical object-like
        // redefinition, legal HLSL, probe-only. (The CPU cull twin
        // `emissive::cull_abl` matches only the literal "noelcull" —
        // correct: noelcode is a GPU code-presence probe, not a cull-policy
        // flip.)
        ("noelcode", "ABL_NO_EL_CULL"),
        // The inherited-t_start lever (leaf.hlsl's long-standing source-edit
        // ablation, made repeatable): leaf primaries trace from 0. Sound —
        // t_start lower-bounds the nearest hit, so the same hit is found and
        // only traversal is paid; --check-gpu passes with it armed.
        ("tzero", "ABL_TZERO"),
        // Secondary-ray cost probes, the GPU twins of shade.rs::Abl and the
        // ONLY primary-vs-secondary scalpel inside the DXR pipeline's one
        // opaque DispatchRays region: shade.hlsli neutralizes the TRAVERSAL
        // at each consumer while keeping every rng draw and all control flow,
        // so the delta prices the rays and nothing else. `nosec` arms all
        // five (the OR lives in shade.hlsli's ABL_* block). Image changes
        // by design — cost probes, never levers (the nogbuf class).
        ("noshadow", "ABL_NOSHADOW"),
        ("noao", "ABL_NOAO"),
        ("norefl", "ABL_NOREFL"),
        ("noglass", "ABL_NOGLASS"),
        ("nogi", "ABL_NOGI"),
        ("nosec", "ABL_NOSEC"),
    ] {
        if abl_has(tag) {
            out.push_str(&format!("#define {def} 1\n"));
        }
    }
    out
}

/// Pixel-identical wavefront performance ablations. Unlike `abl_defs`, these
/// are pasted only into the tile-recursion unit, so toggling `oldcut`/`nobatch`
/// cannot perturb the reference/leaf shader cache. `FR_ABL=oldcut,nobatch`
/// reconstructs the pre-B70-pass queue code for an executable A/B without
/// editing HLSL. `nowave` is the deliberate exception — emitted BOTH here and
/// in `abl_defs`, because its consumers straddle the cache split (the tile
/// unit pastes ctr.hlsli too, and its gw guard reads the define as the
/// wavegw override): an arm that reaches only one side measures half a
/// feature while reporting the whole one, which is exactly what happened
/// until 2026-08-01 (see the abl_defs row's comment — the fully-armed A/B is
/// what flipped the gw default to plain atomics on 2026-08-09). `wavegw`
/// (the gw re-arm) is tile-unit-only and lives here alone.
pub fn wavefront_ablation_defs() -> String {
    let mut out = String::new();
    if abl_has("oldcut") {
        out.push_str("#define ABL_KEEP_TERMINAL_CUT 1\n");
    }
    if abl_has("nobatch") {
        out.push_str("#define ABL_NO_QUEUE_BATCH 1\n");
    }
    // The tile-unit half of `nowave` — see the fn doc. abl_defs carries the
    // leaf/sky/hemi/reference half; no compile unit pastes both fns, so the
    // define is never emitted twice into one source.
    if abl_has("nowave") {
        out.push_str("#define ABL_NO_WAVE_OPS 1\n");
    }
    // Re-arm the frontier's wave-aggregated gw_alloc/gw_min_bits (plain
    // atomics are the DEFAULT since the 2026-08-09 verdict flip — see
    // wavefront.hlsl's gw block). Tile-unit-only by construction: the gw
    // helpers live nowhere else, so this arm needs no abl_defs twin. A
    // simultaneous `nowave` wins (the guard is
    // `defined(ABL_WAVE_GW) && !defined(ABL_NO_WAVE_OPS)`).
    if abl_has("wavegw") {
        out.push_str("#define ABL_WAVE_GW 1\n");
    }
    out
}

/// HLSL prelude every compile unit takes: the `--spp` jitter table's row count
/// (`FrameCb::jitters`, hand-mirrored in trace_common.hlsli's cbuffer). The
/// The detail strength knobs (`--detail-strength` / `--detail-ao-strength`)
/// as compile-time defines for shade.hlsli's DETAIL_STR/DETAIL_AO_STR seams
/// (which default to 1.0 by #ifndef, so the file stands alone). Session
/// constants (restart tier — kernels compile at construction), injected into
/// every unit that pastes SHADE_HLSLI on BOTH pipelines: the probe-reach
/// rule — a define that misses one shade unit silently splits the arms.
/// `{:.9e}` = past-f32 significant digits, so HLSL parses back the exact
/// bits the CPU statics hold (the spp_defs SKY_J idiom).
pub fn detail_defs() -> String {
    format!(
        "#define DETAIL_STR {:.9e}\n#define DETAIL_AO_STR {:.9e}",
        crate::scene::detail_strength(),
        crate::scene::detail_ao_strength()
    )
}

/// SIZE is derived from `dlss::MAX_SPP` rather than written twice — a literal
/// there would be a third constant to raise in lockstep, and a shader reading
/// past a too-small array is silent (no gate can see it). Injected like
/// ALPHA_CUTOUT / FTREE.
pub fn spp_defs() -> String {
    // The sky-fill's extra-sample offsets (cs_sky under FLAG_CLOUDS at
    // spp > 1): PHASE-0 Halton, deliberately frame-INDEPENDENT — a proven-
    // empty tile antialiases a static function, and per-frame offsets put
    // inter-frame dither on cloud edges that the spp stability gate (rightly)
    // rejects at night. Injected as literals ({:.9e} — 10 significant digits,
    // past f32's 9 — so the HLSL parses back the exact bits) because the CB
    // jitter table carries the FRAME's phase and the CB has no room for a
    // second one; the frame-0 gates still match the reference kernel exactly
    // (its jitters[] ARE jitter_for_sample(0, s) there). The CPU twin is
    // fill_sky_rows' jitter_for_sample(0, k).
    let sky_j: String = (0..crate::dlss::MAX_SPP)
        .map(|k| {
            let (x, y) = crate::dlss::jitter_for_sample(0, k);
            format!("float2({x:.9e}, {y:.9e})")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "#define MAX_SPP {}u\n#define JITTER_ROWS {}\n#define MAX_FIREFLIES {}\n#define MAX_EMISSIVE_LIGHTS {}\nstatic const float2 SKY_J[MAX_SPP] = {{ {} }};",
        crate::dlss::MAX_SPP,
        crate::dlss::MAX_SPP / 2,
        // The firefly pose-row count (fireflies.rs::MAX_FIREFLIES), injected
        // for the same reason as JITTER_ROWS: a hand-mirrored literal would
        // be a second constant to raise in lockstep, and a shader reading
        // past a too-small cbuffer array is silent.
        crate::fireflies::MAX_FIREFLIES,
        // The emissive cluster-row count (emissive.rs) — same argument.
        crate::emissive::MAX_EMISSIVE_LIGHTS,
        sky_j
    )
}

/// The sky-fill compile unit (`cs_sky_lod` + `cs_cloud_shadow`), assembled once
/// so the wavefront tracer AND the DXR pipeline compile byte-identical fill
/// kernels — the two cannot drift. `k` = the SKY_LOD pitch (1 = lattice
/// compiled out), `n` = the CLOUD_SHADOW cells/wavelength (0 = cache compiled
/// out). SKY_UNIT gives `cloud_lod` register u5 (queues.hlsli's tile queues
/// suppressed); it traces nothing, so no frustum/rt machinery.
pub fn sky_unit_src(k: u32, n: u32) -> String {
    // WIDTH_PROBE pushed conditionally (cs_sky's width slot); the DXR
    // pipeline compiles only the fill entries from this unit, whose armed
    // epilogue lives in cs_sky alone, so DXR's u3-unbound state is untouched.
    let mut parts: Vec<String> = Vec::new();
    let wd = width_defs();
    if !wd.is_empty() {
        parts.push(wd);
    }
    // WAVEVIZ rides the same conditional-push rule. Like WIDTH_PROBE, its
    // armed block lives in cs_sky alone — the DXR pipeline compiles only the
    // fill entries from this unit, so DXR's u3 stays untouched here.
    let wv = waveviz_defs();
    if !wv.is_empty() {
        parts.push(wv);
    }
    parts.extend([
        // Ablations must reach THIS unit too — see the feed_src note. A
        // `nogbuf` probe once measured `sky` as unchanged and it was read as
        // "the sky pack write is free"; the define had simply never arrived,
        // and cs_sky kept storing all 88 B/px. Real number: 0.736 -> 0.099 ms.
        abl_defs(),
        format!(
            "#define SKY_UNIT 1\n#define SKY_LOD {k}\n#define SKY_LOD_LOG {}",
            k.trailing_zeros()
        ),
        format!("#define CLOUD_SHADOW_N {n}"),
        format!(
            "#define SKY_GROUP {SKY_GROUP}\n#define SKY_SPLIT {SKY_SPLIT}\n#define LEAF_TILE {}",
            crate::render::leaf_tile()
        ),
        spp_defs(),
        TRACE_COMMON_HLSLI.to_string(),
        CTR_HLSLI.to_string(),
        QUEUES_HLSLI.to_string(),
        SKYLOD_HLSLI.to_string(),
        SKY_HLSL.to_string(),
    ]);
    parts.join("\n")
}

/// What the wavefront tracer's kernel sources depend on that this module
/// cannot read for itself: two scene-derived facts and one device fact.
///
/// Everything else the assembly needs — the levers, the `--spp` table, the
/// detail knobs, the frustum structure — is a process-global read at
/// construction time, deliberately: those are SESSION constants, and threading
/// them through a parameter list would invite two tracers in one process
/// compiling against different values of the same knob.
pub struct TraceKeys<'a> {
    /// Drives `alpha_defs`/`height_defs`/`trans_defs`/`empty_defs` — which
    /// conditional-hit machinery compiles in at all.
    pub scene: &'a Scene,
    /// THIS device's vendor, never a process-global "picked" one: under
    /// `--dual-gpu` two devices are live, and `cand_defs` arming on the wrong
    /// one restores a `tmin-overshoot` defect on every leaf primary of the
    /// device that does not need it.
    pub vendor: Vendor,
    /// `scene_gpu.sway.is_some()` — the UPLOADED animated-TLAS ring's
    /// existence. See `sway_defs`.
    pub sway_armed: bool,
}

/// Every compile unit the wavefront tracer needs, assembled.
///
/// The strings are the product; the three snapshots below are the reason this
/// returns a struct rather than a tuple of sources. `CLOUD_SHADOW`, `SKY_LOD`
/// and `FTREE_ENABLED` are process statics that a mid-process A/B (a gate
/// flipping one between two constructions) can move, and the kernels are
/// COMPILED against them while the buffers are SIZED against them and the
/// per-frame record paths dispatch against them. Reading each static once here
/// and handing the caller what was read makes "a kernel can never desync from
/// its own fill dispatch" structural instead of a rule to remember — that
/// desync is the device-hang class `record_cloud_shadow` documents.
pub struct TraceSources {
    pub reference: String,
    pub resolve: String,
    pub wavefront: String,
    pub sky: String,
    /// `LEAF_NO_FB` — the hemi arm compiled OUT. See the assembly for why the
    /// leaf kernel ships as two PSOs from one source.
    pub leaf: String,
    pub leaf_fb: String,
    pub hemi_wave: String,
    pub hemi_leaf: String,
    pub compose: String,
    pub feed: String,
    /// The `--spp` prelude, retained because the two CONDITIONAL units below
    /// are built only when their feature is armed and would otherwise have to
    /// rebuild it (`spp_defs` materializes the whole `SKY_J` table).
    pub spp: String,
    /// The snapshots — see the struct doc. Callers must size and record
    /// against THESE, never a fresh read of the statics.
    pub cloud_shadow_n: u32,
    pub sky_lod: u32,
    pub ftree_on: bool,
    /// Did `SWAY_MV` actually compile in? Not the same question as "is the
    /// ring armed": `--sw-rays` suppresses the define while the ring still
    /// exists. The per-frame CB flag and the dmv-ring fill both key off this,
    /// so it is reported rather than re-derived — a tracer whose flag says
    /// "sway MVs" to a kernel compiled without them writes garbage motion.
    pub sway_mv_on: bool,
}

impl TraceSources {
    /// The NRD bridge unit (`--nrd` sessions + the check-gpu bridge gates).
    /// Conditional, so it is a method rather than a field: `TRACE_COMMON_HLSLI`
    /// is most of a megabyte and an unarmed session should not pay to join it.
    ///
    /// `abl_defs` FIRST — the probe-reach rule; see `feed`'s assembly for the
    /// measurement that established it.
    pub fn nrd_bridge(&self) -> String {
        [abl_defs().as_str(), &self.spp, TRACE_COMMON_HLSLI, FSR_WIRE_HLSLI, NRD_BRIDGE_HLSL]
            .join("\n")
    }

    /// The NPPD staging unit (`--gpu --nppd` XeSS sessions only). Conditional
    /// for the same reason as `nrd_bridge`.
    pub fn nppd(&self) -> String {
        [&self.spp, TRACE_COMMON_HLSLI, NPPD_HLSL].join("\n")
    }
}

/// Assemble every wavefront compile unit.
///
/// THE OUTPUT IS THE CONTRACT. Both backends hand their compiler these exact
/// strings — a Vulkan tracer that assembled its own would be running different
/// programs, and the cross-backend image A/B that is supposed to catch porting
/// defects would instead be measuring the difference between two assemblies.
/// So this function is the one place a `#define` may be decided, and adding one
/// means adding it to every unit that can consume it (see the module header's
/// note on ablations that answer confidently because they never arrived).
pub fn trace_sources(k: &TraceKeys) -> TraceSources {
    let scene = k.scene;
    // Alpha-masked scenes compile the cutout candidate loops into the trace
    // primitives (rt.hlsli); height-carrying scenes likewise compile the relief
    // march in (runtime-gated by FLAG_HEIGHT — the V toggle); transmissive
    // scenes compile transmit_q's tinted candidate loop in (TRANS_SHADOW).
    // Scenes with none compile the FORCE_OPAQUE originals verbatim (modulo
    // leading blank lines) — procedural/stress sessions are structurally
    // untouched, and the bit gates rely on that. Dev cost-attribution ablations
    // ride `abl_defs()`, which the DXR assembly pastes too so the two arms stay
    // comparable.
    let empty_def = empty_defs(scene);
    // SWAY_MV: suppressed under --sw-rays — the software rays render the REST
    // pose, so sway MVs would describe motion that is not on screen (sway_defs'
    // doc). DXR takes the same predicate verbatim.
    let sway_def = if sw_rays() { "" } else { sway_defs(k.sway_armed) };
    let defs = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        empty_def,
        alpha_defs(scene),
        height_defs(scene),
        trans_defs(scene),
        cand_defs(k.vendor),
        blas_defs(),
        sway_def,
        abl_defs(),
        rtgi_defs()
    );
    let defs = defs.as_str();
    // The session's frustum structure: `#define FTREE` swaps frustum.hlsli's
    // binary bound_query/refine_cut for ftree.hlsli's wide bodies (same
    // signatures — the call sites don't know), and the FNode array uploads at
    // t0 in place of the binary nodes. --no-ftree keeps the binary path.
    let ftree_on = crate::ftree::FTREE_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
    let ft_defs = if ftree_on { "#define FTREE 1" } else { "" };
    // The per-lane frustum stack depth — the tracer's ONLY groupshared, and
    // therefore the only thing that can cap resident GROUPS in the level and
    // hemi kernels, which RGA shows are nowhere near VGPR-limited
    // (cs_level_wide 54 VGPR against cs_leaf's 216). Injected into exactly the
    // units that paste frustum.hlsli; the work graph inherits it through the
    // wavefront unit. See lane_stack() before sweeping it.
    let ls_defs = format!("#define LANE_STACK {}u{}", lane_stack(), stack_layout_def());
    let ls_defs = ls_defs.as_str();
    // The cbuffer's jitter-table size (--spp) — every unit sees the cbuffer.
    let sd_owned = spp_defs();
    let sd = sd_owned.as_str();
    // The detail strength knobs — every unit that pastes SHADE_HLSLI.
    let dd = detail_defs();
    let dd = dd.as_str();
    // --sw-rays: rt_sw.hlsli pastes in place of rt.hlsli in every ray-shooting
    // unit (leaf x2, reference, hemi_wave, hemi_leaf); the wavefront unit gets
    // the SW_RAYS define too (level_finish's leaf-cut translation arm).
    // SW_TRAV_STACK rides in from bvh::TRAV_STACK so the HLSL stacks stay in
    // lockstep with the build's max_depth assert.
    let sw_on = sw_rays();
    let rt_src: &str = if sw_on { RT_SW_HLSLI } else { RT_HLSLI };
    let sw_defs = if sw_on {
        format!("#define SW_RAYS 1\n#define SW_TRAV_STACK {}u", crate::bvh::TRAV_STACK)
    } else {
        String::new()
    };
    let sw_defs = sw_defs.as_str();
    // The leaf unit's cut consumption composes with --no-cut-rays exactly as
    // the CPU's intersect_multi short-circuit does: software traversal from the
    // root, the scalar t_start kept. The wavefront unit shares the define
    // (level_finish's leaf-cut translation compiles only when the leaf actually
    // consumes it).
    let sw_leaf_defs = if sw_rays_leaf() { "#define SW_RAYS_LEAF 1" } else { "" };
    // Snapshot the cloud-cache levers ONCE — see `TraceSources`' doc.
    let cloud_shadow_v = cloud_shadow_n();
    let sky_lod_v = sky_lod();
    // The cloud-shadow cache is compiled into every unit that shades (leaf,
    // reference) plus the unit that fills it (sky). wavefront/hemi get 0 and
    // keep the exact per-pixel expression — they must not declare u6, which is
    // the tile queue there. (DXR compiles it in through its own assembly.)
    let csn = format!("#define CLOUD_SHADOW_N {cloud_shadow_v}");
    // The sky-lod lattice defines, shared by the sky-fill unit, both leaf
    // kernels, AND the reference kernel — so reference's sky pixels compose
    // through the identical `sky_radiance_lod` and the exact-zero
    // wavefront-vs-reference image A/B stays bit-identical at the default-ON K.
    // SKY_UNIT is a harmless no-op in a unit that pastes no queues.hlsli
    // (reference), where u5 is free anyway.
    let sky_lod_defs = format!(
        "#define SKY_UNIT 1\n#define SKY_LOD {sky_lod_v}\n#define SKY_LOD_LOG {}",
        sky_lod_v.trailing_zeros()
    );
    // The reference kernel swaps to rt_sw with the wavefront: the exact-zero
    // wavefront-vs-reference gates require ONE intersector on both sides (the
    // "same intersector, same seeds" contract). It also reads the cloud lattice
    // (SKYLOD_HLSLI at u5, filled by record_sky_lod) and the cloud-shadow cache
    // (csn, filled by record_cloud_shadow), so it shades sky exactly as the
    // leaf kernel does.
    // WIDTH_PROBE defines, pushed CONDITIONALLY into every unit below — never
    // as an empty join element (the feed-unit byte-identity rule): unarmed
    // assemblies must be byte-identical to the pre-lever sources.
    let wd = width_defs();
    let bd = ballast_defs();
    let wv = waveviz_defs();
    let mut reference_parts: Vec<&str> = Vec::new();
    if !wd.is_empty() {
        reference_parts.push(wd.as_str());
    }
    if !bd.is_empty() {
        reference_parts.push(bd.as_str());
    }
    if !wv.is_empty() {
        reference_parts.push(wv.as_str());
    }
    reference_parts.extend([
        csn.as_str(),
        sky_lod_defs.as_str(),
        defs,
        sw_defs,
        sd,
        dd,
        TRACE_COMMON_HLSLI,
        SKYLOD_HLSLI,
        rt_src,
        RIPPLE_HLSLI,
        SHADE_HLSLI,
        REFERENCE_HLSL,
    ]);
    let reference = reference_parts.join("\n");
    // The waveviz overlay deliberately does NOT live in this resolve unit any
    // more: it composites at the present funnel (tonemap.rs's waveviz PSO),
    // which is what makes it work under every upscaler — the resolve runs only
    // on the plain arms.
    let resolve = [sd, TRACE_COMMON_HLSLI, RESOLVE_HLSL].join("\n");
    let (sky_group, sky_split) = (SKY_GROUP, SKY_SPLIT);
    let sky_defs = format!(
        "#define SKY_GROUP {sky_group}\n#define SKY_SPLIT {sky_split}\n#define LEAF_TILE {}",
        crate::render::leaf_tile()
    );
    let wavefront_ablation_defs = wavefront_ablation_defs();
    let mut wavefront_parts: Vec<&str> = Vec::new();
    if !wd.is_empty() {
        wavefront_parts.push(wd.as_str());
    }
    wavefront_parts.extend([
        sky_defs.as_str(),
        wavefront_ablation_defs.as_str(),
        empty_def,
        ft_defs,
        ls_defs,
        sw_defs,
        sw_leaf_defs,
        sd,
        TRACE_COMMON_HLSLI,
        CTR_HLSLI,
        QUEUES_HLSLI,
        FRUSTUM_HLSLI,
        FTREE_HLSLI,
        WAVEFRONT_HLSL,
    ]);
    let wavefront = wavefront_parts.join("\n");
    // The sky fill is its own unit so `cloud_lod` can take u5 (SKY_UNIT
    // suppresses queues.hlsli's tile-queue declarations there). Assembled by
    // the shared `sky_unit_src` so the DXR pipeline's fill kernels cannot drift
    // from these.
    let sky = sky_unit_src(sky_lod_v, cloud_shadow_v);
    // Two leaf kernels from the one source. `fb_mode` is a cbuffer value, so
    // leaving the hemi arm as a runtime branch inlines shade_split at both call
    // sites and the kernel's register allocation is the MAX of the two — which
    // on RDNA costs occupancy (and therefore latency hiding) in every fb-OFF
    // frame, i.e. essentially all of them. `LEAF_NO_FB` compiles that arm out;
    // record_wavefront picks per frame. The leaf kernel shades the sky pixels
    // inside leaf tiles, so it reads the same lattice: SKY_UNIT yields u5 (it
    // never touches the tile queues) and skylod.hlsli supplies the accessors.
    let leaf_of = |extra: &str| {
        let lg = format!("#define LEAF_GROUP {}", leaf_group());
        let mut parts: Vec<&str> = Vec::new();
        if !wd.is_empty() {
            parts.push(wd.as_str());
        }
        if !wv.is_empty() {
            parts.push(wv.as_str());
        }
        parts.extend([
            lg.as_str(),
            extra,
            csn.as_str(),
            sky_lod_defs.as_str(),
            defs,
            sw_defs,
            sw_leaf_defs,
            sd,
            dd,
            TRACE_COMMON_HLSLI,
            CTR_HLSLI,
            QUEUES_HLSLI,
            SKYLOD_HLSLI,
            rt_src,
            RIPPLE_HLSLI,
            SHADE_HLSLI,
            LEAF_HLSL,
        ]);
        parts.join("\n")
    };
    let leaf = leaf_of("#define LEAF_NO_FB 1");
    let leaf_fb = leaf_of("");
    // Hemi kernels stay on the BINARY tree deliberately (no ft_defs): hemi
    // bound queries terminate in ~10 visits, where a wide pop's unconditional 8
    // slot tests lose to the binary pop's 1 — measured +35% ms on the hemi-gi
    // bench with the wide tree, against -54% on the tile path. record_hemi
    // rebinds the binary buffer at t0.
    let hemi_wave = [
        defs,
        ls_defs,
        sw_defs,
        sd,
        TRACE_COMMON_HLSLI,
        CTR_HLSLI,
        HEMI_HLSLI,
        FRUSTUM_HLSLI,
        rt_src,
        HEMI_WAVE_HLSL,
    ]
    .join("\n");
    let mut hemi_leaf_parts: Vec<&str> = Vec::new();
    if !wd.is_empty() {
        hemi_leaf_parts.push(wd.as_str());
    }
    hemi_leaf_parts.extend([
        defs,
        sw_defs,
        sd,
        dd,
        TRACE_COMMON_HLSLI,
        CTR_HLSLI,
        HEMI_HLSLI,
        rt_src,
        RIPPLE_HLSLI,
        SHADE_HLSLI,
        HEMI_LEAF_HLSL,
    ]);
    let hemi_leaf = hemi_leaf_parts.join("\n");
    let compose = [sd, TRACE_COMMON_HLSLI, CTR_HLSLI, QUEUES_HLSLI, COMPOSE_HLSL].join("\n");
    // abl_defs FIRST so a feed ablation is not silently inert. It was: an
    // `FR_ABL=nopack` probe reported `feed` unchanged and that was read as "the
    // pack read is free" — but the define never reached this unit, so the probe
    // compared identical code against itself. The shipping split then measured
    // feed 0.544 -> 0.231 ms, i.e. the read very much is not free. An ablation
    // that cannot reach its target is worse than no ablation, because it
    // answers confidently.
    let feed = [abl_defs().as_str(), sd, TRACE_COMMON_HLSLI, FSR_WIRE_HLSLI, FEED_HLSL].join("\n");
    TraceSources {
        reference,
        resolve,
        wavefront,
        sky,
        leaf,
        leaf_fb,
        hemi_wave,
        hemi_leaf,
        compose,
        feed,
        spp: sd_owned,
        cloud_shadow_n: cloud_shadow_v,
        sky_lod: sky_lod_v,
        ftree_on,
        sway_mv_on: !sway_def.is_empty(),
    }
}

/// What the DXR pipeline's compile units depend on beyond this module's own
/// session globals.
///
/// The two mode fields are POST-DEGRADE SNAPSHOTS, never the raw levers.
/// `DxrGpu::new` resolves `--dxr-inline` against the device's RT tier and the
/// Intel heightfield refusal, and `--dxr-sbt` against the scene core's class
/// partition, before anything is assembled; re-reading the levers here would
/// let a refused mode reach the shaders anyway, and those refusals exist
/// because the alternative is a DXC error or a hung device.
///
/// `FR_DXR_LEAN` is deliberately ABSENT. It moves only the state object's
/// EXPORT list and not one byte of source — that identity is exactly what makes
/// a lean cost delta attributable to export provisioning and nothing else — so
/// it is purely the backend's business.
///
/// There is no `vendor` here either, unlike `TraceKeys`: `cand_defs`' AMD
/// candidate-TMin workaround has nothing to arm in this pipeline, whose library
/// rays pass a literal `TMin` 0.
pub struct DxrKeys<'a> {
    /// Drives the same four per-scene predicates the wavefront reads.
    pub scene: &'a Scene,
    /// `scene_gpu.sway.is_some()`. Taken UNCONDITIONALLY, unlike the
    /// wavefront's: `sway_defs`' `--sw-rays` carve-out cannot arise in a
    /// pipeline that never software-rays.
    pub sway_armed: bool,
    /// `--dxr-inline`, after the caps gate and the Intel-heightfield degrade.
    pub inline_mode: u32,
    /// `--dxr-sbt 3` (recursive class dispatch) — the one sbt mode the SOURCE
    /// can see: it pastes rt.hlsli for inline occlusion and reshapes both
    /// rt_dxr.hlsli and shade_split's continuations. Modes 1/2 change only what
    /// the SBT and the export list contain, so the assembly is blind to them.
    pub recurse: bool,
    /// Upscaler session — compile the feed kernels.
    pub gbuf_full: bool,
    /// `--nrd` — compile the bridge kernels (needs `gbuf_full` as well).
    pub nrd: bool,
}

/// Every compile unit the DXR pipeline needs, assembled.
///
/// Four of the units are `Option`: this pipeline compiles the cloud-cache
/// fills, the feed kernels, the NRD bridge and the deferred shade only when the
/// session asks for them, and each would otherwise pay to join
/// `TRACE_COMMON_HLSLI` for nothing.
///
/// THE MODE-0 BYTE-IDENTITY CONTRACT lives here: at `--dxr-inline 0` with no
/// sbt ladder, the library assembles EXACTLY the pre-lever sequence — the
/// lever's off-state is byte-identical source, not merely equivalent source —
/// and it stays so ACROSS inline modes for every OTHER unit. Adding a define
/// means pushing it conditionally, never as an empty join element (see `feed`).
pub struct DxrSources {
    /// The one DXIL library: raygen, misses, closest-hits, any-hits.
    pub library: String,
    /// `lib_6_5` where the library pastes rt.hlsli's RayQuery (the inline modes
    /// and the recursive-SBT hybrid), `lib_6_3` otherwise. This pipeline's caps
    /// floor is deliberately below the wavefront's, and mode 0 keeps it — which
    /// is the whole reason DXR runs on strictly more hardware.
    pub lib_target: &'static str,
    pub resolve: String,
    /// The cloud-cache fill kernels' unit — `sky_unit_src`, shared VERBATIM
    /// with the wavefront so the two pipelines' caches cannot drift. `Some`
    /// when either cache is armed; which of its two entries to compile is
    /// `sky_lod` / `cloud_shadow_n` below.
    pub sky: Option<String>,
    pub feed: Option<String>,
    pub nrd_bridge: Option<String>,
    /// `--dxr-inline 3`'s deferred compute shade. Mirrors the library's sources
    /// MINUS its DispatchRays halves — no rt_dxr.hlsli, no dxr.hlsl — because
    /// this unit must never contain a TraceRay, which a gate below pins.
    pub dxr_shade: Option<String>,
    /// Snapshots; see `TraceSources`' doc for why they are reported rather than
    /// re-read at the record sites.
    pub cloud_shadow_n: u32,
    pub sky_lod: u32,
    pub sway_mv_on: bool,
}

/// Assemble every DXR compile unit.
///
/// SHARED SOURCE IS THE POINT: shading parity with the wavefront is inherited
/// rather than re-ported, because both pipelines paste the same
/// `trace_common.hlsli` + `shade.hlsli`. The trace primitives are the only swap
/// — rt_dxr.hlsli's `TraceRay` flavors, or rt.hlsli's inline RayQuery bodies
/// ahead of them, under `DXR_INLINE_SEC`.
///
/// The three refusal lines below travel WITH the assembly deliberately. Each
/// says "this lever needs that mode, so it is off", which is a statement about
/// what compiled — the `abl_announce` category — and a lever that silently
/// declined to arm is precisely the probe-reach failure the module header
/// warns about.
pub fn dxr_sources(k: &DxrKeys) -> DxrSources {
    let scene = k.scene;
    let inline_mode = k.inline_mode;
    let recurse = k.recurse;
    // The cbuffer's --spp jitter-table size, injected like alpha_defs.
    let sd_owned = spp_defs();
    let sd = sd_owned.as_str();
    // The detail strength knobs (shade.hlsli's DETAIL_STR seams).
    let dd = detail_defs();
    let dd = dd.as_str();
    let sway_def = sway_defs(k.sway_armed);
    let defs = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        // SCENE_EMPTY is INERT in this pipeline and is carried only so the
        // two arms' define sets stay comparable: the guards it gates live in
        // frustum.hlsli and rt_sw.hlsli, neither of which this library
        // pastes, and DXR rays go through the TLAS (an empty TLAS is the
        // driver's problem, not ours). It becomes load-bearing the moment
        // this pipeline ever pastes a software-BVH consumer.
        empty_defs(scene),
        alpha_defs(scene),
        height_defs(scene),
        trans_defs(scene),
        blas_defs(),
        // SWAY_MV: the prev-pose MV correction in gbuf_write_hit, off the
        // uploaded ring's existence (sway_defs — no --sw-rays carve-out here,
        // DXR never software-rays).
        sway_def,
        // FR_ABL, shared with the wavefront — without it every cloud cost
        // attribution was wavefront-only and silently incomparable here.
        abl_defs(),
        // Real-time GI (--no-rtgi omits): shade_full's bounce block — shared
        // with the wavefront so the two pipelines' shading stays one source
        // (the probe-reach rule).
        rtgi_defs()
    );
    let defs = defs.as_str();
    // Snapshot the cloud-cache levers ONCE — see `TraceSources`' doc.
    let sky_lod_k = sky_lod();
    let cloud_shadow_v = cloud_shadow_n();
    // The two cache defines arm trace_common's cached cloud_sun_transmittance
    // (u6) + skylod.hlsli's sky_radiance_lod (u5) for EVERY shade path in the
    // library — parity inherited, not re-ported. u5/u6 are unbound in the DXR
    // root signature today (the wavefront's tile queues), so binding dedicated
    // buffers there needs no root-signature change.
    let cloud_defs = format!(
        "#define CLOUD_SHADOW_N {cloud_shadow_v}\n#define SKY_LOD {sky_lod_k}\n#define SKY_LOD_LOG {}",
        sky_lod_k.trailing_zeros()
    );
    // Mode 0 assembles EXACTLY the shipping sequence (the lever's off-state is
    // byte-identical source, not merely equivalent); armed modes prepend the
    // define and paste rt.hlsli's RayQuery primitives ahead of rt_dxr.hlsli,
    // whose TraceRay flavors + tlas/HitInfo compile out under DXR_INLINE_SEC.
    // The two cache defines ride in every mode (the shipping sequence now
    // carries them; mode 0 stays byte-identical ACROSS inline modes, the
    // lever's actual contract).
    // --dxr-sbt 3 (recurse — arms only at inline 0, the caller's degrade) is
    // the HYBRID paste: rt.hlsli rides along for INLINE occlusion (shadow/AO
    // rays never re-enter the pipeline, which is what caps the recursion at 5)
    // while DXR_SBT_RECURSE reshapes rt_dxr.hlsli (tlas/HitInfo/TraceRay
    // primitives out, trace_shade in) and shade_split's continuations
    // (recursive TraceRay in place of the lap loop). An unarmed sbt mode leaves
    // every push identical — the mode-0 byte-identity contract holds across the
    // WHOLE ladder.
    let inline_def = format!("#define DXR_INLINE_SEC {inline_mode}");
    let mut parts = vec![defs, sd, dd, cloud_defs.as_str()];
    if inline_mode > 0 {
        parts.push(inline_def.as_str());
    }
    if recurse {
        parts.push("#define DXR_SBT_RECURSE 1");
    }
    // FR_WIDTH: the raygen wave-width report (dxr_width[0]) — armed only
    // where the library compiles at lib_6_5 (the inline modes' floor; wave ops
    // in a 6_3 library are off the table, and mode 0's raygen is not the
    // lottery victim anyway). The deferred-shade unit below carries its own
    // WIDTH_PROBE define.
    if width_probe_on() && (inline_mode > 0 || recurse) {
        parts.push("#define WIDTH_PROBE_RAYGEN 1");
    }
    // FR_BALLAST=dxr:N — the raygen ballast (the knee-vs-knee host comparison;
    // reference.hlsl's liveness argument, mirrored in dxr.hlsl's mode-2 arm).
    // The blocks' compound guard already confines them to DXR_INLINE_SEC == 2,
    // but pushing the define into a mode whose arm compiles out would leave the
    // SEED live and the update dead — a ballast that "measures" a flat curve —
    // so any other mode refuses loudly instead (the FR_WIDE rule).
    let bd = ballast_dxr_defs();
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
    // FR_WAVEVIZ — the wave-ticket overlay's DXR half. Modes 1/2 only: mode 0's
    // raygen is lib_6_3 (no wave ops) and mode 3's thin raygen is PINNED to
    // write no tbuf (the deferred kernel is plain compute — its packing is the
    // dispatch grid's, nothing to discover). The `chs` sub-mode additionally
    // needs mode 1 (the one mode whose primary runs a closest-hit).
    let wv = waveviz_defs();
    if !wv.is_empty() {
        if inline_mode == 1 || inline_mode == 2 {
            parts.push(wv.as_str());
            if waveviz_chs() {
                if inline_mode == 1 {
                    parts.push("#define WAVEVIZ_CHS 1");
                } else {
                    eprintln!(
                        "gpu: FR_WAVEVIZ=chs needs --dxr-inline 1 (this session \
                         is mode {inline_mode}) — raygen tickets instead"
                    );
                }
            }
        } else {
            eprintln!(
                "gpu: FR_WAVEVIZ needs --dxr-inline 1|2 (this session is mode \
                 {inline_mode}) — off on this pipeline"
            );
        }
    }
    parts.push(TRACE_COMMON_HLSLI);
    // skylod.hlsli after trace_common (needs sky_compose/sky_backdrop/rw); no
    // SKY_UNIT — this unit pastes no queues.hlsli, so u5 is free anyway.
    parts.push(SKYLOD_HLSLI);
    if inline_mode > 0 || recurse {
        parts.push(RT_HLSLI);
    }
    parts.extend([RT_DXR_HLSLI, RIPPLE_HLSLI, SHADE_HLSLI, DXR_HLSL]);
    let library = parts.join("\n");
    // rt.hlsli's RayQuery in a library target needs SM 6.5 — the inline modes'
    // floor, and mode 3's (the caller degraded to 2 if the device lacks it).
    let lib_target = if inline_mode > 0 || recurse { "lib_6_5" } else { "lib_6_3" };
    // The waveviz overlay composites at the present funnel, not here — this
    // resolve runs only on the plain arm and stays lever-free.
    let resolve = [sd, TRACE_COMMON_HLSLI, RESOLVE_HLSL].join("\n");
    // The cloud-cache FILL kernels (cs_sky_lod / cs_cloud_shadow), from the
    // SHARED sky_unit_src so they cannot drift from the wavefront's — plain
    // compute (no rays), so cs_6_3 like the resolve/feed kernels.
    let sky = (sky_lod_k > 1 || cloud_shadow_v > 0).then(|| sky_unit_src(sky_lod_k, cloud_shadow_v));
    // Upscaler sessions: the same feed kernels the wavefront runs, at this
    // pipeline's cs_6_3 cap floor (feed.hlsl needs nothing newer).
    //
    // abl_defs FIRST so a feed ablation is not silently inert — the library's
    // `defs` above already carries it, but this unit did not, so an
    // `FR_ABL=nopack` probe under --dxr compared identical code against itself
    // (feed.hlsl consumes ABL_NOPACK; the wavefront's `feed` learned the same
    // lesson — "an ablation that cannot reach its target answers
    // confidently"). Pushed CONDITIONALLY, unlike the wavefront's
    // unconditional first element: this unit's unarmed baseline has no leading
    // blank line, and an empty first segment + join("\n") would prepend one —
    // the unarmed source stays byte-identical. Armed, both pipelines' feed
    // units assemble identical leading text (abl_defs ends in '\n').
    let abl_first = |tail: [&'static str; 3]| -> String {
        let abl = abl_defs();
        let mut p: Vec<&str> = Vec::new();
        if !abl.is_empty() {
            p.push(abl.as_str());
        }
        p.push(sd);
        p.extend(tail);
        p.join("\n")
    };
    let feed =
        k.gbuf_full.then(|| abl_first([TRACE_COMMON_HLSLI, FSR_WIRE_HLSLI, FEED_HLSL]));
    // --nrd bridge kernels: the same nrd_bridge.hlsl unit at this pipeline's
    // cs_6_3 floor (conditional abl_defs push — the feed unit's byte-identity
    // rule above).
    let nrd_bridge = (k.gbuf_full && k.nrd)
        .then(|| abl_first([TRACE_COMMON_HLSLI, FSR_WIRE_HLSLI, NRD_BRIDGE_HLSL]));
    // --dxr-inline 3: the deferred-shade kernel. cs_6_5 — it fires rt.hlsli's
    // inline RayQuery secondaries, the same SM 6.5 / tier 1.1 floor the
    // armed-mode gate already enforced.
    let dxr_shade = (inline_mode == 3).then(|| {
        // FR_WIDTH pushed conditionally (the unit rule): arms dxr_shade.hlsl's
        // guarded u3 view + width write (slot 1).
        let wd = width_defs();
        let mut shade_parts: Vec<&str> = Vec::new();
        if !wd.is_empty() {
            shade_parts.push(wd.as_str());
        }
        shade_parts.extend([
            defs,
            sd,
            dd,
            cloud_defs.as_str(),
            inline_def.as_str(),
            TRACE_COMMON_HLSLI,
            SKYLOD_HLSLI,
            RT_HLSLI,
            RIPPLE_HLSLI,
            SHADE_HLSLI,
            DXR_SHADE_HLSL,
        ]);
        shade_parts.join("\n")
    });
    DxrSources {
        library,
        lib_target,
        resolve,
        sky,
        feed,
        nrd_bridge,
        dxr_shade,
        cloud_shadow_n: cloud_shadow_v,
        sky_lod: sky_lod_k,
        sway_mv_on: !sway_def.is_empty(),
    }
}

/// The work-graph ladder's source (`FR_WORKGRAPH=1`): the SAME wavefront unit
/// plus its node shaders, with `WORKGRAPH` switching `level_finish`'s child
/// emission from `qout` to out-params. The tile logic is deliberately NOT
/// forked — one implementation, two dispatch shapes.
///
/// `wide`/`deep` are RECURSION depths, i.e. one less than the level counts they
/// run, and both are clamped to >= 1 by the caller because 0 means "not
/// recursive at all" to the compiler, which would make a node's self-output an
/// illegal cycle. Over-declaring `deep` only reserves more backing memory;
/// UNDER-declaring makes the deepest node drop children, which the shader
/// counts into CTR_OVERFLOW — a failed gate rather than a corrupted image, but
/// still not worth courting.
pub fn workgraph_src(wavefront: &str, wide: u32, deep: u32) -> String {
    format!(
        "#define WORKGRAPH 1\n#define WG_WIDE_LEVELS {wide}\n#define WG_DEEP_LEVELS {deep}\n{}\n{}",
        wavefront, WORKGRAPH_HLSL
    )
}

// --- Shader-source gates -----------------------------------------------
//
// The HLSL below is the executable specification; these tests pin the few
// invariants in it that are load-bearing for soundness but that no CPU-only
// gate can reach (`--check-gpu`/`--check-dxr` need a real adapter). They are
// deliberately narrow: each asserts an ordering or a monotonicity statement,
// never formatting.

#[cfg(test)]
mod width_ballast_shader_source_tests {
    use super::{CTR_COUNT, CTR_HLSLI, CTR_TOTAL, CTR_W_LEAF, LEAF_HLSL, WAVEFRONT_HLSL};
    const REFERENCE: &str = super::REFERENCE_HLSL;
    const SKY: &str = super::SKY_HLSL;
    const HEMI_LEAF: &str = super::HEMI_LEAF_HLSL;

    /// Every WaveGetLaneCount() the FR_WIDTH probe added must sit inside a
    /// WIDTH_PROBE guard — an unguarded one would ship in every session and
    /// break the unarmed byte-identity contract. (ctr.hlsli's wave-aggregated
    /// bumps legitimately call it unguarded; the pin covers the PROBE files.)
    #[test]
    fn width_writes_are_guarded_and_slots_survive_zeroing() {
        for (name, src, slot) in [
            ("leaf", LEAF_HLSL, "CTR_W_LEAF"),
            ("sky", SKY, "CTR_W_SKY"),
            ("wavefront", WAVEFRONT_HLSL, "CTR_W_LEVEL"),
            ("hemi_leaf", HEMI_LEAF, "CTR_W_HEMI"),
            ("reference", REFERENCE, "CTR_W_REFERENCE"),
        ] {
            for (i, _) in src.match_indices(slot) {
                // Every use must sit between a guard-open and its `#endif`:
                // the nearest preceding open of ANY legal spelling must
                // exist, with no intervening close. (reference.hlsl's u3
                // DECL guard widened to `WIDTH_PROBE || WAVEVIZ` for the
                // waveviz ticket — the compound spelling is legal there.)
                let last_open = ["#ifdef WIDTH_PROBE", "#if defined(WIDTH_PROBE) || defined(WAVEVIZ)"]
                    .iter()
                    .filter_map(|g| src[..i].rfind(g))
                    .max()
                    .unwrap_or_else(|| panic!("{name}: {slot} used before any WIDTH_PROBE guard"));
                assert!(
                    !src[last_open..i].contains("#endif"),
                    "{name}: {slot} use at byte {i} escaped its WIDTH_PROBE guard"
                );
            }
        }
        // The survival-by-construction premise: the seed kernels still zero
        // exactly `i < CTR_COUNT`, so the width tail is never wiped.
        assert!(WAVEFRONT_HLSL.contains("i < CTR_COUNT"));
        assert!(!WAVEFRONT_HLSL.contains("i < CTR_TOTAL"));
        assert!(CTR_W_LEAF >= CTR_COUNT && CTR_TOTAL > CTR_W_LEAF);
        // ctr.hlsli's literal block mirrors the Rust consts.
        assert!(CTR_HLSLI.contains("#define CTR_W_LEAF      26u"));
        assert!(CTR_HLSLI.contains("#define CTR_TOTAL       31u"));
    }

    /// --waveviz: every ID-mint touch (the wv_t locals and their
    /// WaveReadLaneFirst mints) must sit inside a WAVEVIZ guard — an
    /// unguarded one ships in every session and breaks the unarmed
    /// byte-identity contract (the width pin's shape). Note the IDs are
    /// POSITION-keyed pure math, never an arrival-order atomic — the atomic
    /// ticket strobed (scheduling order is nondeterministic per frame).
    #[test]
    fn waveviz_blocks_are_guarded() {
        let guards: &[&str] = &["#ifdef WAVEVIZ", "#if defined(WIDTH_PROBE) || defined(WAVEVIZ)"];
        for (name, src, needles) in [
            ("leaf", LEAF_HLSL, &["wv_t"][..]),
            ("sky", SKY, &["wv_t"][..]),
            ("reference", REFERENCE, &["wv_t"][..]),
        ] {
            for needle in needles {
                let mut found = 0u32;
                for (i, _) in src.match_indices(needle) {
                    found += 1;
                    let last_open = guards
                        .iter()
                        .filter_map(|g| src[..i].rfind(g))
                        .max()
                        .unwrap_or_else(|| {
                            panic!("{name}: {needle} used before any WAVEVIZ guard")
                        });
                    assert!(
                        !src[last_open..i].contains("#endif"),
                        "{name}: {needle} use at byte {i} escaped its WAVEVIZ guard"
                    );
                }
                assert!(found > 0, "{name}: expected at least one {needle} (anti-vacuity)");
            }
        }
        // The overlay's colorizer lives at the present funnel (waveviz.hlsl),
        // not in resolve — resolve must stay lever-free (plain arms only).
        assert!(!super::RESOLVE_HLSL.contains("WAVEVIZ"));
    }

    /// The FR_BALLAST array must be confined to its guarded blocks and its
    /// fold must hide behind the never-true `spp == 0xdeadu` branch — that
    /// branch is the whole image-bit-identity argument.
    #[test]
    fn ballast_confined_and_fold_branch_dead() {
        let n_guards = REFERENCE.matches("#if defined(BALLAST_N) && (BALLAST_N > 0)").count();
        assert_eq!(n_guards, 3, "reference.hlsl must carry exactly the 3 ballast blocks");
        for (i, _) in REFERENCE.match_indices("ballast") {
            let before = &REFERENCE[..i];
            let last_open = before
                .rfind("#if defined(BALLAST_N)")
                .expect("ballast text before any BALLAST_N guard");
            assert!(
                !REFERENCE[last_open..i].contains("#endif"),
                "ballast use at byte {i} escaped its guard"
            );
        }
        let fold = REFERENCE.find("bacc").expect("the fold accumulator must exist");
        let branch = REFERENCE.find("if (spp == 0xdeadu)").expect("the dead branch must exist");
        assert!(branch < fold, "the fold must sit under the never-true spp branch");
    }
}

#[cfg(test)]
mod ftree_shader_source_tests {
    use super::FTREE_HLSLI;

    /// The serial wide-node loop caches internal-slot distances before all
    /// terminal slots have tightened `best`. Its overflow fallback must first
    /// reject those stale distances and must only ever lower the bound. This
    /// source-level gate is deliberately hardware-free: the live HLSL remains
    /// the executable specification, while `cargo test` pins the two ordering
    /// and monotonicity statements on which conservative empty-space proof
    /// depends.
    #[test]
    fn overflow_fallback_cannot_raise_best() {
        let expand = FTREE_HLSLI
            .split_once("void ft_expand")
            .expect("ft_expand must remain in the FTree shader")
            .1
            .split_once("// frustum.rs::nearest_geometry_distance")
            .expect("ft_expand must end before bound_query")
            .0;
        let stale_check = expand
            .find("if (pd >= best) continue;")
            .expect("cached distances must be rechecked after terminals tighten best");
        let overflow = expand
            .find("if (sp + 1 > LANE_STACK)")
            .expect("ft_expand must retain a conservative stack-overflow fallback");
        let coarse_min = expand
            .find("best = min(best, pd);")
            .expect("stack overflow must lower best with min");

        assert!(stale_check < overflow && overflow < coarse_min);
        assert!(!expand.contains("best = pd;"));
    }
}

#[cfg(test)]
mod continuation_shader_source_tests {
    use super::{
        CONTINUATION_HLSLI, LEAF_HLSL, QUEUES_HLSLI, RT_SW_HLSLI, WAVEFRONT_HLSL,
    };

    /// Pin the public producer/consumer seam: terminal records carry only an
    /// opaque token, leaf rays pass it through untouched, and the software
    /// provider validates it before reading its private cut arena.
    #[test]
    fn leaf_frontier_is_opaque_and_invalid_tokens_fall_back_to_root() {
        let leaf_rec = QUEUES_HLSLI
            .split_once("struct LeafRec")
            .expect("LeafRec must exist")
            .1
            .split_once("struct SkyRec")
            .expect("LeafRec must precede SkyRec")
            .0;
        assert!(leaf_rec.contains("TraversalFrontier frontier;"));
        assert!(!leaf_rec.contains("cut_slot"));
        assert!(!leaf_rec.contains("cut_len"));

        assert!(CONTINUATION_HLSLI.contains("struct TraversalFrontier"));
        assert!(CONTINUATION_HLSLI.contains("FRONTIER_COOKIE_V1"));
        assert!(CONTINUATION_HLSLI.contains("(slot << 6u) | (len - 1u)"));
        assert!(CONTINUATION_HLSLI.contains("(h.opaque.x >> 6u) >= cap_cut"));
        assert!(WAVEFRONT_HLSL.contains(
            "lf.frontier = frontier_from_binary_cut(leaf_slot, leaf_len);"
        ));
        assert!(LEAF_HLSL.contains(
            "trace_closest_frontier(rec.frontier, cam_origin.xyz, dir,"
        ));
        assert!(!LEAF_HLSL.contains("rec.cut_slot"));
        assert!(!LEAF_HLSL.contains("rec.cut_len"));

        let provider = RT_SW_HLSLI
            .split_once("bool trace_closest_frontier(")
            .expect("software frontier provider must exist")
            .1;
        let validate = provider
            .find("frontier_backend_is_root(frontier)")
            .expect("provider must validate/fallback the opaque handle");
        let arena_read = provider
            .find("cut_pool[")
            .expect("software provider must consume the private cut arena");
        assert!(validate < arena_read);
        assert!(provider[validate..arena_read].contains("trace_closest("));
    }
}

#[cfg(test)]
mod empty_bvh_shader_source_tests {
    use super::{empty_defs, FRUSTUM_HLSLI, RT_SW_HLSLI};

    /// Empty binary BVHs use a count-zero sentinel root on the CPU. GPU
    /// binary consumers must take their identities before interpreting that
    /// sentinel as an internal node; unlike the wide tree, it has no child
    /// occupancy mask to stop a descent.
    #[test]
    fn empty_binary_consumers_short_circuit_before_node_reads() {
        let empty = crate::scene::SceneBuilder::new().finish(crate::scene::default_sun());
        assert_eq!(empty_defs(&empty), "#define SCENE_EMPTY 1");

        let bound = FRUSTUM_HLSLI
            .split_once("float bound_query")
            .expect("binary bound_query must exist")
            .1
            .split_once("// frustum.rs::refine_cut")
            .expect("bound_query must precede refine_cut")
            .0;
        assert!(bound.contains("#ifdef SCENE_EMPTY"));
        assert!(bound.contains("return t_limit;"));
        assert!(
            bound.find("return t_limit;").unwrap()
                < bound.find("BvhNode node = bvh_nodes[idx];").unwrap()
        );

        let refine = FRUSTUM_HLSLI
            .split_once("uint refine_cut")
            .expect("binary refine_cut must exist")
            .1
            .split_once("#endif // !FTREE")
            .expect("binary refine_cut must end before the FTREE guard")
            .0;
        assert!(refine.contains("#ifdef SCENE_EMPTY"));
        assert!(refine.contains("return 0u;"));
        assert!(
            refine.find("return 0u;").unwrap()
                < refine.find("BvhNode node = bvh_nodes[idx];").unwrap()
        );

        for (signature, identity) in [
            ("bool trace_closest(", "return false;"),
            ("bool occluded_q(", "return false;"),
            ("float3 transmit_q(", "return float3(1.0, 1.0, 1.0);"),
            ("bool trace_closest_frontier(", "return false;"),
        ] {
            let body = RT_SW_HLSLI
                .split_once(signature)
                .unwrap_or_else(|| panic!("{signature} must exist"))
                .1;
            let empty_guard = body
                .find("#ifdef SCENE_EMPTY")
                .unwrap_or_else(|| panic!("{signature} must guard an empty scene"));
            let identity_at = body[empty_guard..]
                .find(identity)
                .unwrap_or_else(|| panic!("{signature} must return its empty identity"));
            let first_node = body.find("bvh_nodes[").unwrap_or(usize::MAX);
            assert!(empty_guard + identity_at < first_node);
        }
    }
}

#[cfg(test)]
mod height_interval_shader_source_tests {
    // These two used to be re-`include_str!`ed here, because the module that
    // owned them was in the other backend file and this one could not see it.
    // Both units live in this module now, so the pin reads the SAME `&'static
    // str` the assembly pastes — a second copy of a source is a second thing
    // that can go stale.
    use super::{DXR_HLSL, RT_DXR_HLSLI, RT_HLSLI, TRACE_COMMON_HLSLI};

    /// Relief displacement is bounded in world-normal distance, not in ray t.
    /// Pin the shader-side half of the grazing-interval regression without
    /// requiring an RT-capable adapter: hardware enumeration must cover the
    /// full positive base-triangle ray, and both intersectors must retain the
    /// caller's logical bounds for the post-march test.
    #[test]
    fn relief_candidates_use_full_hardware_interval_and_logical_marched_bounds() {
        let helpers = TRACE_COMMON_HLSLI
            .split_once("float height_tmin")
            .expect("height_tmin helper must exist")
            .1
            .split_once("uint pack_info")
            .expect("height interval helpers must precede pack_info")
            .0;
        assert!(helpers.contains("if (flags & FLAG_HEIGHT) return 0.0;"));
        assert!(helpers.contains("if (flags & FLAG_HEIGHT) return FLT_MAX;"));

        for src in [RT_HLSLI, RT_DXR_HLSLI] {
            assert!(!src.contains("tmin - height_max"));
            assert!(!src.contains("tmax + height_max"));
            assert!(src.contains("height_tmax(tmax)"));
        }
        // rt.hlsli reaches the relief interval through `cand_tmin`, which the
        // AMD candidate-loop workaround also hooks. The relief arm must survive
        // that indirection: the unarmed fallback has to BE height_tmin, or a
        // non-AMD build silently loses the widened interval and relief regresses
        // at grazing incidence with every gate still green.
        assert!(RT_HLSLI.contains("r.TMin = cand_tmin(tmin);"));
        let cand = RT_HLSLI
            .split_once("float cand_tmin(float logical_tmin) {")
            .expect("cand_tmin helper must exist")
            .1
            .split_once('}')
            .expect("cand_tmin must have a body")
            .0;
        assert!(cand.contains("#ifdef CAND_TMIN0"));
        assert!(cand.contains("return 0.0;"));
        assert!(cand.contains("return height_tmin(logical_tmin);"));
        assert!(RT_DXR_HLSLI.contains("height_tmin(tmin)"));

        // SCOPE, not merely existence — and this one shipped broken. The three
        // callers do NOT share a preprocessor condition: trace_closest and
        // occluded_q live inside `#if defined(ALPHA_CUTOUT) || defined(HEIGHTFIELD)`,
        // but transmit_q is under the INDEPENDENT `#ifdef TRANS_SHADOW`. Defining
        // the helper inside the first guard therefore compiles fine for every
        // scene that carries both arms or neither — which is every gated scene
        // (san-miguel and THE WORLD have foliage; procedural/stress/powerplant
        // have no transmissive material) — and fails on the ordinary
        // glass-without-cutout scene with `undeclared identifier 'cand_tmin'`,
        // taking out every ray-shooting kernel at TraceGpu init.
        // `FR_ABL=noalpha` on san-miguel is the repro.
        let def = RT_HLSLI
            .find("float cand_tmin(float logical_tmin) {")
            .expect("cand_tmin helper must exist");
        let guard = RT_HLSLI
            .find("#if defined(ALPHA_CUTOUT) || defined(HEIGHTFIELD)")
            .expect("the cutout/relief guard must exist");
        assert!(
            def < guard,
            "cand_tmin must be defined AHEAD of the cutout/relief guard — \
             transmit_q calls it from the independent TRANS_SHADOW arm"
        );
        let calls: Vec<usize> = RT_HLSLI
            .match_indices("r.TMin = cand_tmin(tmin);")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            calls.len(),
            3,
            "trace_closest, occluded_q and transmit_q must all route TMin through cand_tmin"
        );
        assert!(calls[0] > def, "every cand_tmin call must follow its definition");

        assert!(RT_HLSLI.contains("ct > tmin && ct < tmax"));
        assert!(RT_DXR_HLSLI.contains("p.tmin = tmin"));
        assert!(RT_DXR_HLSLI.contains("p.tmax = tmax"));
        assert!(DXR_HLSL.contains("t <= tmin || t >= tmax"));
        assert!(DXR_HLSL.contains("t <= p.tmin || t >= p.tmax"));
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
        // The waveviz ID mints are position-keyed pure math (no dxr_width /
        // counter touch) — pin every wave-op mint inside a WAVEVIZ guard.
        for (i, _) in DXR_HLSL.match_indices("WaveReadLaneFirst") {
            let last_open = ["#if defined(WAVEVIZ) && defined(WAVEVIZ_CHS)",
                             "#if defined(WAVEVIZ) && !defined(WAVEVIZ_CHS)"]
                .iter()
                .filter_map(|g| DXR_HLSL[..i].rfind(g))
                .max()
                .unwrap_or_else(|| panic!("dxr.hlsl: WaveReadLaneFirst outside a WAVEVIZ guard"));
            assert!(
                !DXR_HLSL[last_open..i].contains("#endif"),
                "dxr.hlsl: WaveReadLaneFirst at byte {i} escaped its WAVEVIZ guard"
            );
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
    use super::{DXR_HLSL, DXR_SHADE_HLSL, RT_DXR_HLSLI, SHADE_HLSLI};

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
        let src = SHADE_HLSLI;
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

    /// Strip `//`-to-end-of-line comments, PRESERVING byte offsets (comment
    /// bytes become spaces) so a failure can name a real position.
    ///
    /// Mandatory for the scan below, not tidiness: `dxr.hlsl` carries the
    /// literal `DispatchRaysIndex()` in TWO comments (the `band_id` header and
    /// the miss shader's sky-LOD note), so a naive scan fails on CORRECT code.
    /// The pre-commit hook's own first draft "passed" by matching nothing for
    /// the mirror-image reason; a scanner has to be shown its own blind spot.
    fn strip_line_comments(src: &str) -> String {
        let b = src.as_bytes();
        let mut out = Vec::with_capacity(b.len());
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
                while i < b.len() && b[i] != b'\n' {
                    out.push(b' ');
                    i += 1;
                }
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        String::from_utf8(out).expect("ascii-preserving strip")
    }

    /// `--dual-gpu`: a banded DispatchRays shrinks the GRID, so every
    /// `DispatchRaysIndex()` must be lifted back to absolute by `band_id` or it
    /// addresses the wrong pixel — and in inline modes 0-2 there is no bounds
    /// check anywhere, so an unlifted index WRITES PAST THE PLANE.
    ///
    /// Five sites carry the lift today (raygen, the waveviz CHS mint,
    /// chs_shade's gbuf write, the miss shader's sky-LOD and its cloud dither).
    /// A sixth added without it is the hole class, and on a single-adapter box
    /// no GPU gate can see it — this one runs in `cargo test`.
    ///
    /// Deliberately NOT pinned to an exact site count: new lifted sites are
    /// expected, and the loop already requires each one to be lifted. It
    /// asserts nothing about formatting, line numbers, or `band_id`'s body
    /// beyond its dependence on the band's own root constant.
    #[test]
    fn every_dispatch_rays_index_is_band_lifted() {
        let src = strip_line_comments(DXR_HLSL);
        let mut n = 0;
        for (i, _) in src.match_indices("DispatchRaysIndex()") {
            assert!(
                src[..i].trim_end().ends_with("band_id("),
                "dxr.hlsl: a DispatchRaysIndex() at byte {i} is not wrapped in band_id() — a \
                 banded --dual-gpu dispatch would address the wrong pixel there, and in inline \
                 modes 0-2 write past the plane"
            );
            n += 1;
        }
        // Anti-vacuity: a rename or a refactor that removed every occurrence
        // would otherwise pass this trivially.
        assert!(n >= 5, "expected at least the 5 known lifted sites, found {n}");
        assert!(
            src.contains("uint2 band_id(uint2 id)"),
            "band_id must exist for the scan above to mean anything"
        );
        assert!(
            src[src.find("uint2 band_id(uint2 id)").unwrap()..]
                .lines()
                .take(4)
                .any(|l| l.contains("push1")),
            "band_id must lift by the band's own root constant (push1)"
        );
    }

    /// Mode 3's deferred compute half has no `DispatchRaysIndex` — it lifts by
    /// hand — and it must do so BEFORE it derives a pixel index. A full-screen
    /// compute grid beside a banded dispatch shades rows this device never
    /// wrote, from a `hitrec` that is stale or uninitialised VRAM.
    #[test]
    fn deferred_shade_lifts_and_clamps_before_indexing() {
        let src = strip_line_comments(DXR_SHADE_HLSL);
        let lift = src.find("id.y += push1").expect("dxr_shade.hlsl: no band lift");
        let clamp = src.find("id.y >= push2").expect("dxr_shade.hlsl: no band clamp");
        let pi = src.find("uint pi =").expect("dxr_shade.hlsl: no pixel index");
        assert!(lift < pi && clamp < pi, "the band lift and clamp must precede the pixel index");
    }
}
