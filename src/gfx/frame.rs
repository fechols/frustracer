//! The per-frame constant buffer, and the tile-ownership arithmetic that
//! rides in it.
//!
//! `FrameCb` is the byte-sensitive heart of the renderer's GPU interface: a
//! hand-written `#[repr(C)]` mirror of `cbuffer Frame`, which is declared in
//! SEVEN concatenated compile units, so a size drift corrupts every field
//! after the drift point in all of them at once. Two `const _: () = assert!`s
//! police it from both directions — the exact composed size, and that the
//! result still fits a ring slot — and they are the reason a new `MAX_*` cap
//! cannot be raised without the stride following.
//!
//! It is here rather than in a backend because it is EXACTLY the structure the
//! Vulkan port must keep byte-compatible. `dxc -spirv` is invoked with
//! `-fvk-use-dx-layout` for this struct and nothing else: without it SPIR-V
//! applies its own std140-ish rules and every offset past the first array
//! moves. One packer, two APIs — a second `with_frame` that "just mirrors the
//! fields" is the drift this module exists to make impossible.
//!
//! `TileSplit` lives here for the same reason it lives in the cbuffer: the
//! `--dual-gpu` band is a 64-bit mask plus a depth in `split.xy`/`split.z`,
//! and the arithmetic that produces those bits, the arithmetic that answers
//! `owns_px` in `cs_compose`, and the arithmetic that turns a band into the
//! transfer's byte range must all agree. `split_self_test` is what pins that
//! they do — and moving it here is what lets `--check` RUN it off Windows,
//! which it could not while the module was `#[cfg(windows)]` (the M0 skip
//! line said so, and this retires it).

#![cfg_attr(not(windows), allow(dead_code))]

use crate::camera::CamBasis;
use crate::shade::Quality;
use crate::gfx::shaders::{waveviz_live, waveviz_on};
use crate::scene::Scene;
use glam::Vec3A;

// Root-CBV alignment (256 B). FrameCb is 4576 bytes — 288 of struct plus the
// MAX_SPP-entry jitter table (--spp), the SH sky rows, the MAX_FIREFLIES
// pose rows, and the MAX_EMISSIVE_LIGHTS cluster-light row pairs, which are
// what set the size (raise the stride in lockstep with any cap; the const
// asserts below police both directions).
pub const CB_STRIDE: usize = 4608;

/// Hemisphere points per batch: bounds the transient hemi queue/pool memory
/// (queues are sized to batch x 4^(depth-1) — bounded, cannot overflow;
/// ~300 MB at this size). Bigger batches amortize the barrier-serialized
/// per-batch drains — 4096 measured 294 ms/frame for 1080p GI, 16384 is the
/// sweet spot on a 24 GB card.
pub const HEMI_BATCH: u32 = 16384;
/// Max fb.depth the hemi queue sizing supports (presets top out at 4).
pub const HEMI_MAX_DEPTH: u32 = 4;

/// 0 = off, 1 = AO, 2 = GI (GI subsumes AO, mirroring shade.rs's tiering).
pub fn fb_mode_of(q: &Quality) -> u32 {
    if q.fb.gi {
        2
    } else if q.fb.ao {
        1
    } else {
        0
    }
}

pub const FLAG_ACCUM: u32 = 1;
pub const FLAG_JITTER: u32 = 2;
pub const FLAG_FRAME_JITTER: u32 = 4;
pub const FLAG_VERIFY: u32 = 8;
/// G-buffer pack writes on. Set ONLY when the pack is full-size (upscaler
/// sessions) — root UAVs have no bounds check and the plain-session pack is
/// a GBUF_STRIDE-byte dummy, so this flag is memory safety, not an
/// optimization.
pub const FLAG_GBUF: u32 = 16;
pub const FLAG_HAS_PREV: u32 = 32;
/// FSR-RR sessions: the pack additionally carries the demodulated
/// direct-light signals (GBufExt.sig) and the prev-camera view-Z
/// (GBufCore.core.w);
/// zeros under every other wiring — RR/XeSS packs stay byte-identical.
pub const FLAG_FSR_SIG: u32 = 64;
/// Anisotropic texture filtering on (the session's `--aniso` > 1). A session
/// constant, not a per-frame decision — set from `texture::max_aniso()`, the
/// same source the static aniso sampler's MaxAnisotropy and the CPU's
/// `Cone::aniso` read, so all three renderers filter the same footprint.
/// Which *rays* use it is decided per call site, not by this flag
/// (`shade_split`'s `aniso` arg — hemi bounce laps pass false).
pub const FLAG_ANISO: u32 = 128;
/// Volumetric clouds on (`--no-clouds` clears it). The cloud state rides two
/// otherwise-zero cam-row w lanes — `cam_right.w` = scene diag, `cam_up.w` =
/// the animation clock (`SCENE_DIAG`/`CLOUD_TIME` in trace_common.hlsli, the
/// SCENE_EPS/AO_RADIUS alias pattern) — so no CB offset moves.
pub const FLAG_CLOUDS: u32 = 256;

/// Heightfield relief march on — the V toggle × any_height × the
/// --no-heightfield lever; per-frame runtime gate over the per-scene
/// HEIGHTFIELD compile-in (trace_common.hlsli mirror).
pub const FLAG_HEIGHT: u32 = 512;

/// Firefly point lights live this frame (src/fireflies.rs — count > 0, which
/// already folds in the session enable and the night fade: a day session
/// never sets it, so day kernels are bit-identical by construction). Poses
/// ride the CB's `ff` rows, CPU-baked — the HLSL re-derives nothing.
pub const FLAG_FIREFLIES: u32 = 1024;

/// Beer–Lambert depth tint over the transmission chain's interior segments
/// (`--no-depth-tint` clears it; shade.hlsli branches inside the
/// transmission arm, which non-transmissive scenes never enter — no compile
/// define needed).
pub const FLAG_DEPTH_TINT: u32 = 2048;

/// The pack's guide/signal half is stored this frame — set when any WIRED feed
/// kind consumes it (RR, FSR-RR) or GPU-resident NPPD is live. XeSS and
/// FSR 3.1 read only the core (mv + view_z), so their sessions skip 72 of the
/// old 88 B/px: measured 0.411 ms of pure store cost in `leaf` on the world,
/// of which this recovers ~0.34. Derived per frame like `fsr_sig` (one
/// subscriber is enough — `--quinlight` wires several kinds at once), never a
/// construction-time constant: the tracer is built BEFORE `wire_feed` runs.
///
/// FLAG_FSR_SIG implies this — FSR-RR reads the sig lanes, which live in ext.
pub const FLAG_GBUF_EXT: u32 = 4096;

/// Foliage-sway MVs live this frame: the pack's MV/prev-Z reproject each
/// hit's PREV-POSE point (`p + du·(a + b·p.y)` off the sway_dmv table)
/// instead of the current one. Armed only when the SWAY_MV compile-in is
/// present AND `sway_mv_pair` holds (prev clock present, bit-different, prev
/// camera present) AND the frame's `write_mv_rows` filled the slot — every
/// pinned-clock gate and frozen still runs the flag-off branch, which is
/// today's expressions verbatim (a branch, never an add-zero: −0.0 + 0.0 =
/// +0.0, the fireflies lesson).
pub const FLAG_SWAY_MV: u32 = 8192;

/// Emissive cluster lights live this frame (src/emissive.rs): the scene
/// derived clusters AND the session lever is on AND the frame is not GI
/// (`fb_mode != 2` — the GI gather already delivers emissive transport
/// exactly, so GI frames keep the gather and drop the cluster NEE; the
/// inverted once-per-path rule). An emissive-free scene never sets the bit,
/// so its kernels are bit-identical by construction (the FLAG_FIREFLIES
/// shape).
pub const FLAG_EMISSIVE: u32 = 16384;
/// FR_WAVEVIZ live overlay (armed session AND the I key / headless spin):
/// covered kernels overwrite tbuf with their wave ticket and the resolve
/// stage blends the ticket hash over the scene. Runtime half of the WAVEVIZ
/// compile-in — an armed-but-OFF frame runs the normal tbuf writes, so
/// toggling off recovers the clean frame immediately.
// RENUMBERED at the 2026-08-06 merge: both parallel sessions claimed bit
// 32768 (the sibling branched before FLAG_DETAIL/FLAG_DETAIL_AO/
// FLAG_AMB_BUMP landed); detail keeps 32768..131072, waveviz takes the
// next free bit. Lockstep with trace_common.hlsli's FLAG_WAVEVIZ.
pub const FLAG_WAVEVIZ: u32 = 262144;

/// Real-time GI live this frame (`--no-rtgi` clears the session lever;
/// `with_frame` additionally clears the bit on fb frames — the hemi tiers
/// take precedence — so shade_full's bounce block can key on the bit alone).
/// Lockstep with trace_common.hlsli's FLAG_RTGI.
pub const FLAG_RTGI: u32 = 524288;

/// Spec-AA (`--no-spec-aa` clears it): the slope-variance → roughness fold —
/// mip-averaged normal-map detail (the variance companion, gated per
/// material on Mat.normal_var_tex) and the detail field's faded octaves
/// (detail_var) widen the GGX lobe instead of vanishing with distance. The
/// FLAG_DETAIL runtime-lever shape; lockstep with trace_common.hlsli's
/// FLAG_SPEC_AA.
pub const FLAG_SPEC_AA: u32 = 1048576;

/// An NRD (ReBLUR) bridge is wired this frame, so shade_full's RTGI bounce
/// folds into the `prim.direct_d` capture (+ the bounce ray's t into `ao_t`)
/// — the bridge's diffuse input carries the GI instead of the un-denoised
/// residual. Runtime like FLAG_RTGI (NRD arms at session start and sheds
/// mid-session), and DISTINCT from FLAG_FSR_SIG: FSR-RR sessions arm that
/// bit too, and their dd must stay pure direct diffuse — it is AMD's own
/// denoiser's input. Lockstep with trace_common.hlsli's FLAG_NRD_GI.
pub const FLAG_NRD_GI: u32 = 2097152;

/// Sky pixels SKIP the GBufExt store (bit 22 — the B70 NRD-cost recovery,
/// 2026-08-09): armed only when NRD is the SOLE ext subscriber, because at a
/// sky texel the bridge needs nothing from ext — cs_nrd_pack takes its own
/// canonical-constant sky branch (never reading the possibly-stale bytes) and
/// cs_nrd_out returns at its 0.999·CAM_FAR predicate before the ext load.
/// Every OTHER ext consumer (cs_feed_rr, cs_feed_fsr_rr, nppd.hlsl, the pack
/// readback gates) reads ext full-screen INCLUDING sky, which is why the
/// derivation vetoes on any of them. Measured: the sky ext store was
/// +0.33–0.51 ms/frame on the B70 at native 1080p. Lockstep with
/// trace_common.hlsli's FLAG_SKY_EXT_SKIP.
pub const FLAG_SKY_EXT_SKIP: u32 = 4194304;

/// `cs_nrd_out` applies NVIDIA's `NRD_SG_ReJitter` Jacobian to the denoiser
/// DELTA (bit 23 — the SG campaign, 2026-08-10). Runtime rather than a compile
/// define ON PURPOSE: the N8 gate must A/B both arms inside one process, and
/// an env-keyed `OnceLock` define cannot flip (the `force_sky_ext_skip`
/// lesson). Armed when an NRD bridge is wired, the engine behind it really is
/// NRD (see `TraceGpu::nrd_rejitter` — FRD shares the bridge and must not
/// inherit this), and `FR_NRD_REJITTER` is not `off`; consumed by the bridge
/// unit alone. Lockstep with trace_common.hlsli's FLAG_NRD_REJITTER.
pub const FLAG_NRD_REJITTER: u32 = 8388608;

/// `FR_NRD_REJITTER=off` — the A/B arm for the ReJitter micro-detail
/// restoration (loud on departure, the FR_NRD_BARRIER idiom). Default ON: the
/// pass exists to put back exactly the texel-scale contrast a converged
/// history's narrow blur removes, which is the parked-camera class the user
/// reported. `on` spells the default.
pub fn nrd_rejitter_lever() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FR_NRD_REJITTER") {
        Err(_) => true,
        Ok(v) if v == "off" => {
            eprintln!("nrd: FR_NRD_REJITTER=off — no SG re-jitter (the A/B arm)");
            false
        }
        Ok(v) if v == "on" => true, // the default, spelled explicitly
        Ok(v) => {
            eprintln!("nrd: FR_NRD_REJITTER={v:?} unrecognized (legal: on, off) — on");
            true
        }
    })
}

/// EXACT REMODULATION (bit 24 — 2026-08-10): the lap-0 sig captures are
/// PRE-SCALED by the post-capture factors shade applies to those same lobes, so
/// the bridge's `kd`/`f0` become the EXACT remodulation divisors and the
/// denoiser's delta lands at its true physical weight.
///
/// WHY IT EXISTS. The delta form preserves `base` exactly, but it remodulates
/// the denoiser's CORRECTION at the wire `kd = albedo·(1−metallic)·(1−trans)`
/// while shade multiplied the same lobes by more: `sk = 1−0.157·sheen` (the
/// Charlie energy term), the translucency split, `detail_sun_shadow` on the
/// direct diffuse, and `detail_cavity` on the bounce and on `direct_s`. So the
/// correction was applied at `1/m` its physical weight — on a `dcav = 0.3` pit
/// under FLAG_NRD_GI the bounce correction landed at 3.3x, and the leftover
/// fraction of every bright 1-spp bounce spike stayed in `base` RAW and
/// UN-DENOISED. That is a firefly source, and nrd_bridge.hlsl's own comment
/// used to assert it could not be ("never the recomposed color") — true only
/// while `D_out == D_in`.
///
/// The FLAG_NRD_GI shape exactly: it cannot arm without the sig capture, and it
/// cannot arm in an FSR-RR session (whose composite identity owns those lanes).
/// Unlike FLAG_NRD_REJITTER it is NOT engine-gated — the mismatch is our
/// arithmetic, not NVIDIA's, and FRD's A/B-oracle role is not compromised by
/// its inputs being correctly scaled. Lockstep with trace_common.hlsli's
/// FLAG_REMOD_EXACT.
pub const FLAG_REMOD_EXACT: u32 = 16777216;

/// `FR_NRD_REMOD=off` — the A/B arm for exact remodulation (loud on departure,
/// the FR_NRD_REJITTER idiom). Default ON: this is a bug fix, not a quality
/// trade, and `off` restores the pre-2026-08-10 arithmetic bit-for-bit. `on`
/// spells the default.
pub fn nrd_remod_lever() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FR_NRD_REMOD") {
        Err(_) => true,
        Ok(v) if v == "off" => {
            eprintln!(
                "nrd: FR_NRD_REMOD=off — the pre-fix remodulation (spikes leak past the \
                 denoiser at 1/m weight; the A/B arm)"
            );
            false
        }
        Ok(v) if v == "on" => true, // the default, spelled explicitly
        Ok(v) => {
            eprintln!("nrd: FR_NRD_REMOD={v:?} unrecognized (legal: on, off) — on");
            true
        }
    })
}

/// THE RESIDUAL SPIKE CAP (bit 25 — 2026-08-10). `cs_nrd_out` reconstructs the
/// RESIDUAL `R = base − D_in·kd·m_d − S_in·f0` — everything in the frame the
/// denoiser never saw — and soft-caps its luma against the 8-neighbour ring.
///
/// WHY THE RESIDUAL IS WHERE FIREFLIES HIDE. The recompose is algebraically
/// `col = R + D_out·kd·m_d + S_out·f0`: the two folded channels are denoised,
/// and R is passed through RAW by construction. With FLAG_NRD_GI live the
/// bounce rides the diffuse fold, so R's remaining stochastic term is the root
/// GLASS/TRANSMISSION chain — each interior lap shades with its own sampled
/// sun-shadow pairs and sampled AO, and nothing filters any of it. Neither
/// ReBLUR's own anti-firefly (relative, and applied inside the denoiser) nor
/// FRD's ring pre-clamp can reach it: both see only the folded channels.
///
/// ENGINE-BLIND, with no `nrd_engine` clause — unlike FLAG_NRD_REJITTER, which
/// carries one because the Jacobian is NVIDIA's and FRD is the oracle it is
/// judged against. The residual is the same residual for both engines and both
/// are scored against the same `base`, so a term that fixes it belongs to the
/// shared bridge.
pub const FLAG_NRD_RCLAMP: u32 = 33554432;

/// The `hard` arm of the residual cap (bit 26): swaps the ring-mean multiplier
/// for `NRD_RCLAMP_K_HARD`, which is 1.0 — i.e. "clamp anything brighter than
/// its own surround". A DIAGNOSTIC, not a quality setting: it paints every
/// pixel the cap could ever touch, which is how you see at a glance whether the
/// feature is aimed at glass or at emissive detail. Also the N10 gate's
/// scene-independent teeth (at K = 1.0 the mechanism must fire somewhere, so a
/// check scene with no transmissive geometry still proves the wiring).
pub const FLAG_NRD_RCLAMP_HARD: u32 = 67108864;

/// `FR_NRD_RCLAMP=off|on|hard` — the residual spike cap's lever. DEFAULT OFF,
/// deliberately unlike `FR_NRD_REJITTER`/`FR_NRD_REMOD`: those are restoration
/// and a bug fix respectively, while this is a QUALITY TRADE with a known
/// accept (a single-texel hard-edged emissive on a black background can be
/// attenuated). It defaults on only once the firefly-count measurement earns
/// it. Returns (armed, hard).
pub fn nrd_rclamp_lever() -> (bool, bool) {
    static V: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("FR_NRD_RCLAMP") {
        Err(_) => (false, false),
        Ok(v) if v == "off" => (false, false), // the default, spelled explicitly
        Ok(v) if v == "on" => {
            eprintln!("nrd: FR_NRD_RCLAMP=on — residual spike cap armed");
            (true, false)
        }
        Ok(v) if v == "hard" => {
            eprintln!(
                "nrd: FR_NRD_RCLAMP=hard — residual spike cap at K=1 (the DIAGNOSTIC arm: it \
                 paints everything the cap could touch, not a quality setting)"
            );
            (true, true)
        }
        Ok(v) => {
            eprintln!("nrd: FR_NRD_RCLAMP={v:?} unrecognized (legal: off, on, hard) — off");
            (false, false)
        }
    })
}

/// Unreal-1 detail texturing (`--no-detail-tex` clears it): procedural
/// close-up albedo grain + micro-bump on MAGNIFIED hits — textured AND
/// untextured since the untextured arm (shade.hlsli's post-match detail
/// block: textured materials window off their albedo texture's lod,
/// untextured off the cone footprint in synthetic texel-equivalents,
/// Mat.detail_scale > 0 either way — the FLAG_DEPTH_TINT shape, no compile
/// define needed).
pub const FLAG_DETAIL: u32 = 32768;

/// Detail cavity AO (`--no-detail-ao` clears it): the detail field's pits
/// darken ambient + direct specular (shade.hlsli branches behind `dh < 0`,
/// which only the fired field sets — the FLAG_DETAIL runtime-lever shape).
pub const FLAG_DETAIL_AO: u32 = 65536;

/// Ambient bump response (`--no-amb-bump` clears it): shade.hlsli's
/// `amb_irradiance` amplifies the SH ambient's response to the n_g → n_s
/// deviation (normal maps + detail bump + ripple) — flat-shaded geometry
/// (n_s == n) takes the plain expression verbatim, the runtime-lever shape.
pub const FLAG_AMB_BUMP: u32 = 131072;

/// `GBufCore` stride in bytes — lockstep with trace_common.hlsli (one float4:
/// mv.xy | view_z | prev_z).
pub const GBUF_STRIDE: u64 = 16;

/// `GBufExt` stride in bytes — lockstep with trace_common.hlsli
/// (nr | alb | spec | sig | sig2 = 3 float4 + 1 uint4 + 1 uint2).
pub const GBUF_EXT_STRIDE: u64 = 72;

/// Mirror of `cbuffer Frame` in trace_common.hlsli (304 bytes, 16-aligned
/// rows — float3s ride in float4 slots with scalars packed in .w).
/// pub(crate): gpu/dxr.rs shares the layout (its lib pastes the same
/// trace_common.hlsli); fields stay module-private — outside constructors go
/// through `FrameCb::base`/`with_frame`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FrameCb {
    cam_origin: [f32; 4],
    cam_forward: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    /// The sun (sky::Sun). Replaces the old five rows (sun + the rect light's
    /// center/u/v/color): a disc at infinity needs a direction, a cone, and two
    /// radiometric values. `scene.eps` / `ao_radius` used to ride in
    /// `light_center.w` / `light_u.w`; they are rehomed onto these rows' w slots.
    pub sun: [f32; 4], // xyz = unit dir; w = cos(angular radius)
    sun_e: [f32; 4], // xyz = irradiance/π (the direct loop's multiplier); w = scene eps
    sun_l: [f32; 4], // xyz = DISC radiance (what an escaping ray sees); w = ao_radius
    rw: u32,
    rh: u32,
    frame: u32,
    flags: u32,
    shadow_samples: u32,
    ao_samples: u32,
    reflections: u32,
    /// The fireflies' CONTENT-diagonal scale (`Fireflies::scale` — every FF_*
    /// length multiplies it; deliberately NOT SCENE_DIAG, which the ground
    /// quad inflates ~17× on the procedural scenes). Rode what was `_pad0`.
    ff_scale: f32,
    frame_jitter: [f32; 2],
    /// Primary ray-cone spread (CamBasis::pixel_cone — the CPU value
    /// verbatim, single source for the trilinear LOD parity).
    pixel_cone: f32,
    /// Time-of-day dome brightness (`Scene::sky_scale` — exactly 1.0 in an
    /// untouched session; `x * 1.0` is bit-preserving, so the day sky gates
    /// are unmoved). Rides what was `_pad2`, so no offset moves.
    sky_scale: f32,
    cap_tile: u32,
    cap_leaf: u32,
    cap_sky: u32,
    cap_cut: u32,
    fb_mode: u32,
    fb_depth: u32,
    hemi_batch: u32,
    cap_hemi_pt: u32,
    cap_hemi_cell: u32,
    cap_hemi_leaf: u32,
    cap_hemi_cut: u32,
    /// Live firefly count (rode what was `_pad3`, so no offset moves) —
    /// 0 in every day/`--no-fireflies` session; FLAG_FIREFLIES mirrors it.
    ff_count: u32,
    // Previous frame's camera basis for G-buffer MVs; near/far ride the w
    // slots of the last two rows (scene-static, from dlss::near_far).
    prev_origin: [f32; 4],  // xyz; w = prev inv_w
    prev_forward: [f32; 4], // xyz; w = prev inv_h
    prev_right: [f32; 4],   // xyz; w = near
    prev_up: [f32; 4],      // xyz; w = far
    // --spp: samples per pixel this frame, and which one writes the per-pixel
    // side channels (tbuf/info/pack). See trace_common.hlsli.
    spp: u32,
    probe_sample: u32,
    /// Star visibility (`Scene::night` — exactly 0.0 in an untouched session;
    /// the HLSL star branch is guarded on it, so day kernels are bit-identical
    /// by construction). Rides what was `_pad4`.
    night: f32,
    /// Scene-wide max relief depth in world units (`bvh::height_max_world` —
    /// 0.0 = no height data, which is also how FLAG_HEIGHT's `with_frame`
    /// predicate reads `any_height`). The wavefront TMin widening constant.
    height_max: f32,
    /// Sample offsets from `dlss::jitter_for_sample` (the ONE Halton source —
    /// no radical-inverse port in HLSL), two per 16-byte row.
    jitters: [[f32; 4]; (crate::dlss::MAX_SPP as usize) / 2],
    /// The sky dome in order-2 SH (`scene.sky_sh`, `sh::N` = 9 RGB rows, .w
    /// unused) — the GPU's copy of the analytic ambient the CPU reads through
    /// `Sh9::irradiance`. Appended after every scalar so no offset above moves.
    sky_sh: [[f32; 4]; crate::sh::N],
    /// Firefly poses (src/fireflies.rs — xyz = world position, w =
    /// brightness), the CPU's baked f32s verbatim so both renderers light
    /// from bit-equal positions. Appended LAST (the sky_sh precedent); rows
    /// past `ff_count` are zero and never read (the HLSL loops on the count).
    ff: [[f32; 4]; crate::fireflies::MAX_FIREFLIES],
    /// Cloud shadow cache transform: [origin.x, origin.z, 1/cell, side].
    /// Appended LAST (the sky_sh / ff precedent) so no offset above moves.
    /// pub(crate) so DxrGpu::write_cb can fill it (the shared shadow_grid_row).
    pub(crate) cloud_grid: [f32; 4],
    /// Sway-MV delta table base: x = the frame's ring-slot offset in float4
    /// elements (`slot · n_inst` — the shader reads `sway_dmv[x +
    /// InstanceID]`), yzw unused. Appended LAST (the cloud_grid precedent).
    /// Set through `arm_sway_mv` beside FLAG_SWAY_MV so the pair cannot
    /// split.
    sway_mv_base: [u32; 4],
    /// Emissive cluster lights (src/emissive.rs): x = count, y = the SCENE
    /// LIGHT GAIN's f32 bits (`Scene::light_gain`, exactly 1.0 outside
    /// `--autoexp-mode lights`), zw unused. The gain rode a free lane rather
    /// than moving `CB_STRIDE` — the `ff_count`/`_pad3` idiom — and the HLSL
    /// reads it back through `scene_light_gain()`.
    /// Scene-static — filled by `base` from `Scene::emissive`; FLAG_EMISSIVE
    /// mirrors it per frame (× the live lever × fb_mode != 2).
    el_meta: [u32; 4],
    /// Cluster row a: xyz = power-weighted centroid, w = rc² (source
    /// radius²) — the CPU's derived f32s VERBATIM, so both renderers light
    /// from bit-equal clusters (parity BY DATA, the ff precedent). Rows past
    /// the count are zero and never read (the HLSL loops on the count).
    el_a: [[f32; 4]; crate::emissive::MAX_EMISSIVE_LIGHTS],
    /// Cluster row b: xyz = C/π (radiance·area over π), w = r_infl²
    /// (the window's exact zero). Appended LAST so no offset above moves.
    el_b: [[f32; 4]; crate::emissive::MAX_EMISSIVE_LIGHTS],
    /// Dual-GPU tile ownership (`--dual-gpu`): xy = the bitmask of
    /// level-`z` quadtree tiles THIS device renders (x = tiles 0..31,
    /// y = 32..63), z = the split depth, w unused. Appended LAST so no
    /// offset above moves.
    ///
    /// `z == 0` is the unsplit session — `level_finish` branches around the
    /// test entirely, so every single-GPU frame is bit-identical to the
    /// pre-feature renderer by construction (the `apply_tod`/`night`
    /// precedent). 64 bits caps the split depth at `MAX_SPLIT_DEPTH` = 3.
    split: [u32; 4],
}

/// Deepest quadtree level a `--dual-gpu` split may be assigned at: 4^3 = 64
/// tiles, exactly the 64 bits of the CB's `split.xy` mask. Also the point of
/// diminishing returns — 1/64 of the screen is finer than the balancer can
/// usefully act on, since every reassignment invalidates that device's
/// structure replay.
pub(crate) const MAX_SPLIT_DEPTH: u32 = 3;

/// Which level-`depth` quadtree tiles THIS device renders (`--dual-gpu`).
///
/// A per-DEVICE property that changes only when the balancer reassigns tiles,
/// which is why it lives on `TraceGpu` rather than in `FrameParams` beside the
/// per-frame jitter: the tracer already owns the other things fixed for a
/// device (its resolution, its queues), and the split belongs with them.
///
/// `depth == 0` is the whole screen — `ALL`, the unsplit default, in which
/// `level_finish` branches around the ownership test entirely and the frame is
/// bit-identical to the pre-feature renderer.
///
/// Equality is part of the structure-replay key: a device whose assignment
/// changed must NOT replay the terminal queues it recorded for the old one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TileSplit {
    /// Bit `i` = the level-`depth` tile whose quadtree path is `i`. Unused
    /// above `4^depth`.
    pub mask: u64,
    /// The level `mask` indexes; 0 = unsplit.
    pub depth: u32,
}

impl Default for TileSplit {
    fn default() -> Self {
        Self::ALL
    }
}

impl TileSplit {
    /// The whole screen — one device, no split, every ownership test skipped.
    pub const ALL: TileSplit = TileSplit { mask: u64::MAX, depth: 0 };

    /// Tiles at `depth`, as a count.
    pub fn tiles_at(depth: u32) -> u32 {
        1u32 << (2 * depth)
    }

    /// A CONTIGUOUS band of whole tile ROWS at `depth`: rows `[row0, row1)` of
    /// the `2^depth × 2^depth` grid.
    ///
    /// This is the shape mixed-mode dual-GPU requires — a DXR partner renders a
    /// rectangle, so the wavefront side's tiles must form one too (an
    /// interleaved mask cannot be a single `DispatchRays`). It is also the
    /// cheapest cross-adapter transfer: one contiguous row range, one copy.
    ///
    /// A tile's row is the interleaved "bottom" bit of its path — bit 1 of each
    /// level's 2-bit code (TL=0 TR=1 BL=2 BR=3), most significant level first.
    pub fn rows(depth: u32, row0: u32, row1: u32) -> TileSplit {
        let mut mask = 0u64;
        for path in 0..Self::tiles_at(depth) {
            let mut row = 0u32;
            for lvl in 0..depth {
                // Level `lvl` contributes its B bit; the FIRST level split is
                // the most significant row bit.
                let shift = 2 * (depth - 1 - lvl);
                row = (row << 1) | ((path >> (shift + 1)) & 1);
            }
            if row >= row0 && row < row1 {
                mask |= 1u64 << path;
            }
        }
        TileSplit { mask, depth }
    }

    /// The complement within `depth` — the partner device's assignment. Their
    /// union must be every tile and their intersection empty, which is what
    /// makes the two halves partition the screen exactly.
    pub fn complement(&self) -> TileSplit {
        if self.depth == 0 {
            return TileSplit { mask: 0, depth: 0 };
        }
        let all = if Self::tiles_at(self.depth) >= 64 {
            u64::MAX
        } else {
            (1u64 << Self::tiles_at(self.depth)) - 1
        };
        TileSplit { mask: !self.mask & all, depth: self.depth }
    }

    /// Does this assignment own the level-`depth` tile CONTAINING pixel
    /// `(x, y)` of an `rw x rh` screen?
    ///
    /// The twin of `trace_common.hlsli`'s `split_owns_px`, written as the same
    /// forward midpoint recursion so the two cannot drift. Two consumers: the
    /// shader side bands `cs_compose` (the one per-pixel pass in the tracer),
    /// and the CPU side derives the cross-adapter transfer's row ranges from it.
    ///
    /// `depth == 0` is the unsplit whole screen and answers true without
    /// touching the mask, matching the branch every other consumer takes.
    pub fn owns_px(&self, x: u32, y: u32, rw: u32, rh: u32) -> bool {
        if self.depth == 0 {
            return true;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (0u32, 0u32, rw, rh);
        let mut path = 0u32;
        for _ in 0..self.depth {
            let xm = x0 + (x1 - x0) / 2;
            let ym = y0 + (y1 - y0) / 2;
            let cx = u32::from(x >= xm);
            let cy = u32::from(y >= ym);
            path = (path << 2) | (cy * 2 + cx);
            if cx == 1 {
                x0 = xm;
            } else {
                x1 = xm;
            }
            if cy == 1 {
                y0 = ym;
            } else {
                y1 = ym;
            }
        }
        // The shader's conservative arm, mirrored: an out-of-range path renders
        // a tile twice rather than dropping it. Unreachable at
        // `depth <= MAX_SPLIT_DEPTH`, kept so the twins stay textually equal.
        if path >= 64 {
            return true;
        }
        (self.mask >> path) & 1 != 0
    }

    /// The two devices' assignments for a secondary share of `rows` out of
    /// `2^depth` tile rows: `(primary, secondary)`.
    ///
    /// **A share of ZERO returns `(ALL, None)`, and that is the safety
    /// property the whole feature rests on.** Not `rows(depth, 0, side)` — a
    /// full mask at a nonzero depth is functionally the same but structurally
    /// different: `SPLIT_DEPTH != 0` makes `level_finish` run the ownership
    /// test and `cs_compose` run `split_owns_px` per pixel. `ALL` is depth 0,
    /// which every consumer branches AROUND, so share 0 is the pre-feature
    /// renderer instruction-for-instruction.
    ///
    /// That is what makes it honest to ship a feature whose correct answer on
    /// a bandwidth-starved box is "give the secondary nothing": arming it
    /// costs exactly nothing when the balancer converges to zero. The `None`
    /// is the other half — the caller must skip the secondary's submit and
    /// the transfer outright, not hand it an empty mask and pay the schedule.
    ///
    /// A share at or above `side` would leave the PRIMARY with nothing, which
    /// no consumer expects (it still presents), so it is clamped to `side-1`.
    pub fn for_share(rows: u32, depth: u32) -> (TileSplit, Option<TileSplit>) {
        if rows == 0 || depth == 0 {
            return (TileSplit::ALL, None);
        }
        let side = 1u32 << depth;
        let rows = rows.min(side - 1);
        let prim = TileSplit::rows(depth, 0, side - rows);
        (prim, Some(prim.complement()))
    }

    /// The CONTIGUOUS pixel row range `[y0, y1)` this assignment owns, or
    /// `None` if it is not a whole-tile-row band.
    ///
    /// This is what makes the cross-adapter transfer one `CopyBufferRegion`
    /// per buffer instead of a scatter: every per-pixel plane is indexed
    /// `y*rw + x`, so a row band is a contiguous BYTE range at
    /// `y0*rw*stride` for `(y1-y0)*rw*stride` bytes. `TileSplit::rows` is
    /// built to satisfy this; an interleaved mask deliberately answers `None`
    /// so a caller cannot silently copy the wrong bytes for one.
    ///
    /// Returns `None` rather than a bounding box on purpose — a bounding box
    /// would be a plausible-looking answer that copies pixels the partner
    /// owns, which is exactly the overlap the whole design forbids.
    pub fn row_range(&self, rw: u32, rh: u32) -> Option<(u32, u32)> {
        if self.depth == 0 {
            return Some((0, rh));
        }
        let side = 1u32 << self.depth;
        // Per grid row: how many of its tiles are owned. A band must own all
        // of them or none — a partially-owned row is not a row band.
        let mut owned = [0u32; 8];
        debug_assert!(side as usize <= owned.len());
        for path in 0..Self::tiles_at(self.depth) {
            if (self.mask >> path) & 1 == 0 {
                continue;
            }
            // The tile's grid row: the interleaved "bottom" bit of each
            // level's 2-bit code, most significant level first — the same
            // extraction `rows()` builds the mask from.
            let mut row = 0u32;
            for lvl in 0..self.depth {
                let shift = 2 * (self.depth - 1 - lvl);
                row = (row << 1) | ((path >> (shift + 1)) & 1);
            }
            owned[row as usize] += 1;
        }
        let mut first = None;
        let mut last = 0u32;
        for r in 0..side {
            match owned[r as usize] {
                0 => {
                    // A gap AFTER the band started means two disjoint bands.
                    if first.is_some() && r <= last {
                        return None;
                    }
                }
                n if n == side => {
                    if first.is_none() {
                        first = Some(r);
                    } else if r != last + 1 {
                        return None; // non-contiguous
                    }
                    last = r;
                }
                _ => return None, // partially-owned row
            }
        }
        let first = first?;
        // Every tile in a grid row shares that row's y extent (the y half of
        // the midpoint recursion depends only on the row bits), so any tile of
        // the row gives the band's edges.
        let top = Self::first_path_of_row(self.depth, first);
        let bot = Self::first_path_of_row(self.depth, last);
        let (_, y0, _, _) = rect_for_path(self.depth, top, rw, rh);
        let (_, _, _, y1) = rect_for_path(self.depth, bot, rw, rh);
        Some((y0, y1))
    }

    /// The lowest path index whose grid row is `row` — the inverse of the row
    /// extraction above, taking every x bit as 0.
    fn first_path_of_row(depth: u32, row: u32) -> u32 {
        let mut path = 0u32;
        for lvl in 0..depth {
            let bit = (row >> (depth - 1 - lvl)) & 1;
            path = (path << 2) | (bit << 1);
        }
        path
    }

    /// The CB row: xy = the mask, z = depth, w unused.
    fn cb_row(&self) -> [u32; 4] {
        [self.mask as u32, (self.mask >> 32) as u32, self.depth, 0]
    }
}

/// The screen rect of level-`depth` tile `path`, replaying `trace_tile` /
/// `level_finish`'s integer midpoint splits exactly (`xm = x0 + (x1-x0)/2`,
/// TL=0 TR=1 BL=2 BR=3).
///
/// Test-side only: it exists so `split_self_test` can check the ownership
/// mask's BIT math against the actual tile GEOMETRY, which is the pair that
/// can drift. The renderer never calls it — the shader derives child rects as
/// it descends.
fn rect_for_path(depth: u32, path: u32, rw: u32, rh: u32) -> (u32, u32, u32, u32) {
    let (mut x0, mut y0, mut x1, mut y1) = (0u32, 0u32, rw, rh);
    for lvl in 0..depth {
        let code = (path >> (2 * (depth - 1 - lvl))) & 3;
        let xm = x0 + (x1 - x0) / 2;
        let ym = y0 + (y1 - y0) / 2;
        if code & 1 == 0 {
            x1 = xm;
        } else {
            x0 = xm;
        }
        if (code >> 1) & 1 == 0 {
            y1 = ym;
        } else {
            y0 = ym;
        }
    }
    (x0, y0, x1, y1)
}

/// Pure-math gates for the `--dual-gpu` tile split. DLL- and GPU-free, and run
/// by every `--check` regardless of the lever — the blas-split rule, so the
/// machinery cannot rot while the feature is off.
///
/// What this actually protects: `TileSplit::rows` derives a tile's ROW from
/// interleaved path bits, while the renderer derives a tile's RECT by
/// recursive midpoint splits. Those are two independent derivations of the same
/// thing, and if they disagree a device renders tiles it does not own (wasted
/// work) or, worse, neither device renders a tile (a hole in the image). The
/// geometry cross-check below is the gate that ties them together.
pub fn split_self_test() -> std::result::Result<(), String> {
    // The unsplit default must be exactly the state every consumer branches
    // around — depth 0. If this drifts, single-GPU frames stop being
    // bit-identical and every existing gate silently changes meaning.
    if TileSplit::ALL.depth != 0 {
        return Err("TileSplit::ALL must be depth 0 (the branched-around unsplit state)".into());
    }

    // The documented level-1 claim: a horizontal half-split IS the level-1
    // quadrant boundary, top band = TL | TR = paths 0 and 1.
    let top = TileSplit::rows(1, 0, 1);
    if top.mask != 0b0011 {
        return Err(format!(
            "rows(1,0,1) must be TL|TR = 0b0011, got {:#06b} — the top band is not the \
             level-1 quadrant pair, so a half-split is no longer a quadtree subtree",
            top.mask
        ));
    }

    for depth in 1..=MAX_SPLIT_DEPTH {
        let n = TileSplit::tiles_at(depth);
        let side = 1u32 << depth;
        let full = TileSplit::rows(depth, 0, side);
        let all_bits = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
        if full.mask != all_bits {
            return Err(format!(
                "rows({depth},0,{side}) must cover all {n} tiles, got {:#x}",
                full.mask
            ));
        }

        for r in 0..=side {
            let a = TileSplit::rows(depth, 0, r);
            let b = TileSplit::rows(depth, r, side);

            // PARTITION: disjoint, and together every tile. This is what makes
            // the two devices' work cover the screen exactly once — the
            // property `--check-gpu`'s exactly-once coverage gate asserts on
            // the GPU, checked here in closed form at every split position.
            if a.mask & b.mask != 0 {
                return Err(format!(
                    "depth {depth} row {r}: the two bands overlap ({:#x}) — those tiles \
                     would be rendered twice",
                    a.mask & b.mask
                ));
            }
            if a.mask | b.mask != all_bits {
                return Err(format!(
                    "depth {depth} row {r}: the two bands leave {:#x} unrendered — a hole \
                     in the image (the false-sky class)",
                    all_bits & !(a.mask | b.mask)
                ));
            }
            // The partner's assignment must be derivable as the complement,
            // since that is how the second device is actually configured.
            if a.complement().mask != b.mask || a.complement().depth != depth {
                return Err(format!(
                    "depth {depth} row {r}: complement() disagrees with rows() — \
                     {:#x} vs {:#x}",
                    a.complement().mask,
                    b.mask
                ));
            }

            // GEOMETRY: the mask's bit math vs the renderer's rect recursion,
            // at a power-of-two resolution AND an odd one (where the integer
            // midpoint rounds and the bands are NOT equal height).
            for &(rw, rh) in &[(1920u32, 1080u32), (533, 400)] {
                // The seam: the largest y1 among the top band's tiles must be
                // the smallest y0 among the bottom band's. Anything else is a
                // gap or an overlap in SCREEN space even if the masks
                // partition in INDEX space.
                let mut top_max_y1 = 0u32;
                let mut bot_min_y0 = u32::MAX;
                for path in 0..n {
                    let (_, y0, _, y1) = rect_for_path(depth, path, rw, rh);
                    let in_a = (a.mask >> path) & 1 == 1;
                    if in_a {
                        top_max_y1 = top_max_y1.max(y1);
                    } else {
                        bot_min_y0 = bot_min_y0.min(y0);
                    }
                }
                if r > 0 && r < side && top_max_y1 != bot_min_y0 {
                    return Err(format!(
                        "depth {depth} row {r} at {rw}x{rh}: band seam disagrees — top ends \
                         at y={top_max_y1}, bottom starts at y={bot_min_y0}. The mask's row \
                         bits and the midpoint-split rects have drifted apart."
                    ));
                }
            }
        }

        // OWNS_PX: the FORWARD pixel->path recursion against the BACKWARD
        // path->rect one — the same two-independent-derivations check the
        // seam test above applies to `rows`.
        //
        // It matters because `cs_compose` is a flat per-pixel dispatch that
        // bands itself with the shader twin of `owns_px`, while the tiles
        // themselves descend through the rect recursion. A drift between the
        // two blanks or double-writes a band on fb frames only — and no image
        // gate can see it, since fb-off frames never dispatch compose at all.
        for &(rw, rh) in &[(1920u32, 1080u32), (533, 400)] {
            // A deliberately MIXED assignment (every other tile): a uniform
            // mask would pass a recursion that always answered the same way.
            let s = TileSplit { mask: 0x5555_5555_5555_5555u64 & all_bits, depth };
            for path in 0..n {
                let (x0, y0, x1, y1) = rect_for_path(depth, path, rw, rh);
                if x0 >= x1 || y0 >= y1 {
                    continue; // degenerate at this resolution — owns no pixels
                }
                let want = (s.mask >> path) & 1 == 1;
                for &(px, py) in &[
                    (x0, y0),
                    (x1 - 1, y0),
                    (x0, y1 - 1),
                    (x1 - 1, y1 - 1),
                    (x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2),
                ] {
                    let got = s.owns_px(px, py, rw, rh);
                    if got != want {
                        return Err(format!(
                            "depth {depth} at {rw}x{rh}: owns_px({px},{py}) = {got}, but that \
                             pixel lies in tile path {path}, whose mask bit is {want}. The \
                             pixel->path recursion and the path->rect one have drifted — \
                             cs_compose would band on a different grid than the tiles do."
                        ));
                    }
                }
            }
        }
    }

    // ROW_RANGE: the transfer's byte range. Two bands' pixel rows must
    // partition [0, rh) exactly and meet at the same seam the tile rects do —
    // an off-by-one here copies a row twice or leaves one stale, which is the
    // hole/overlap class again, one level down in the stack.
    for depth in 1..=MAX_SPLIT_DEPTH {
        let side = 1u32 << depth;
        for &(rw, rh) in &[(1920u32, 1080u32), (533, 400)] {
            for r in 1..side {
                let a = TileSplit::rows(depth, 0, r);
                let b = a.complement();
                let (ay0, ay1) = a.row_range(rw, rh).ok_or_else(|| {
                    format!("depth {depth} row {r} at {rw}x{rh}: rows() produced a mask row_range calls non-contiguous")
                })?;
                let (by0, by1) = b.row_range(rw, rh).ok_or_else(|| {
                    format!("depth {depth} row {r} at {rw}x{rh}: the complement of a row band must also be one")
                })?;
                if ay0 != 0 || by1 != rh || ay1 != by0 {
                    return Err(format!(
                        "depth {depth} row {r} at {rw}x{rh}: bands [{ay0},{ay1}) and [{by0},{by1}) \
                         do not partition [0,{rh}) — the transfer would copy a row twice or leave \
                         one stale"
                    ));
                }
                // And the seam must be where the TILES say it is, not merely
                // self-consistent: row_range and rect_for_path are again two
                // derivations of one number.
                let top = TileSplit::first_path_of_row(depth, r);
                let (_, ty0, _, _) = rect_for_path(depth, top, rw, rh);
                if ay1 != ty0 {
                    return Err(format!(
                        "depth {depth} row {r} at {rw}x{rh}: band seam {ay1} disagrees with the \
                         tile rect's y0 {ty0}"
                    ));
                }
            }
        }
    }

    // An INTERLEAVED mask must refuse rather than answer with a bounding box:
    // a plausible-looking range there would copy pixels the partner owns.
    let inter = TileSplit { mask: 0b0101, depth: 1 };
    if inter.row_range(1920, 1080).is_some() {
        return Err(
            "an interleaved (non-row-band) mask must return None from row_range — a bounding \
             box would silently copy the partner's pixels"
                .into(),
        );
    }
    // A partially-owned row must refuse for the same reason.
    let partial = TileSplit { mask: 0b0001, depth: 1 };
    if partial.row_range(1920, 1080).is_some() {
        return Err("a partially-owned tile row must return None from row_range".into());
    }

    // THE ZERO-SHARE IDENTITY — the safety property the whole feature rests
    // on, and the reason it is honest to ship a split whose correct answer on
    // a bandwidth-starved box is "give the secondary nothing".
    //
    // Share 0 must return the DEPTH-0 unsplit state, not a full mask at a
    // nonzero depth. The two render the same image, but only depth 0 is the
    // state every consumer branches AROUND: at any nonzero depth
    // `level_finish` runs the ownership test and `cs_compose` runs
    // `split_owns_px` for every pixel. Arming the feature must cost exactly
    // nothing when the balancer converges to zero.
    for d in 0..=MAX_SPLIT_DEPTH {
        let (p, s) = TileSplit::for_share(0, d);
        if p != TileSplit::ALL || s.is_some() {
            return Err(format!(
                "for_share(0, {d}) must be (ALL, None) — a share of zero has to take the \
                 pre-feature path, not a full mask at depth {d} that still runs the ownership \
                 test on every tile and every compose pixel"
            ));
        }
    }
    for d in 1..=MAX_SPLIT_DEPTH {
        let side = 1u32 << d;
        for rows in 1..side {
            let (p, s) = TileSplit::for_share(rows, d);
            let s = s.ok_or_else(|| format!("for_share({rows}, {d}) dropped the secondary"))?;
            // The pair must still partition, and the SECONDARY must be the one
            // that grows with the share — invert this and every safety
            // property inverts with it (the balancer's "down is safe"
            // direction would then hand the slow device MORE work).
            if p.complement() != s {
                return Err(format!("for_share({rows}, {d}) is not a complementary pair"));
            }
            let (y0, y1) = s
                .row_range(1920, 1080)
                .ok_or_else(|| format!("for_share({rows}, {d}) secondary is not a row band"))?;
            let got = ((y1 - y0) as f32 / 1080.0 * side as f32).round() as u32;
            if got != rows {
                return Err(format!(
                    "for_share({rows}, {d}): the secondary got {got}/{side} rows, not {rows} — \
                     the share is oriented at the wrong device"
                ));
            }
        }
        // Asking for the whole screen must leave the primary something: it is
        // the device that presents.
        let (p, _) = TileSplit::for_share(side, d);
        if p.mask == 0 {
            return Err(format!("for_share({side}, {d}) starved the primary"));
        }
    }

    // The unsplit default must answer true everywhere without consulting the
    // mask: that is the branch `cs_compose` short-circuits on, and if it ever
    // returned false a single-GPU fb frame would come back black.
    for &(x, y) in &[(0u32, 0u32), (1919, 1079), (960, 540)] {
        if !TileSplit::ALL.owns_px(x, y, 1920, 1080) {
            return Err(format!(
                "TileSplit::ALL must own every pixel; ({x},{y}) came back unowned — an \
                 unsplit fb frame would compose nothing there"
            ));
        }
    }

    // A depth past the mask's width must be refused rather than silently
    // truncated — the CB carries 64 bits and nothing else.
    if TileSplit::tiles_at(MAX_SPLIT_DEPTH) != 64 {
        return Err(format!(
            "MAX_SPLIT_DEPTH={MAX_SPLIT_DEPTH} implies {} tiles, but the CB mask holds 64",
            TileSplit::tiles_at(MAX_SPLIT_DEPTH)
        ));
    }
    Ok(())
}
// The HLSL cbuffer is hand-mirrored across 7 concatenated compile units —
// a size drift here corrupts every field after the drift point.
// 304 (the pre-sun size) − 32 (two rect-light rows dropped) + 16 (the spp
// block) + 8·MAX_SPP (the jitter table) + 16·9 (the SH sky) +
// 16·MAX_FIREFLIES (the firefly pose rows) + 16 + 32·MAX_EMISSIVE_LIGHTS
// (the emissive cluster meta + row pairs).
const _: () = assert!(
    std::mem::size_of::<FrameCb>()
        == 320 - 32
            + 8 * crate::dlss::MAX_SPP as usize
            + 16 * crate::sh::N
            + 16 * crate::fireflies::MAX_FIREFLIES
            + 16 // cloud_grid
            + 16 // sway_mv_base
            + 16 // el_meta
            + 32 * crate::emissive::MAX_EMISSIVE_LIGHTS
            + 16 // split (dual-GPU tile ownership)
);
// ...and the whole thing must still fit a CB ring slot.
const _: () = assert!(std::mem::size_of::<FrameCb>() <= CB_STRIDE);

impl FrameCb {
    /// The scene-static base: sun/light/eps/ao_radius, near/far riding the
    /// prev rows' w slots, rw/rh. Queue capacities zero — the wavefront
    /// tracer overwrites its own; the DXR pipeline never reads them.
    pub fn base(scene: &Scene, rw: u32, rh: u32) -> FrameCb {
        let sun = crate::render::sun_dir(scene);
        let (near, far) = crate::dlss::near_far(scene.diag);
        let v4 = |v: Vec3A, w: f32| [v.x, v.y, v.z, w];
        let mut sky_sh = [[0.0f32; 4]; crate::sh::N];
        for (dst, c) in sky_sh.iter_mut().zip(scene.sky_sh.c.iter()) {
            *dst = [c.x, c.y, c.z, 0.0];
        }
        // Emissive cluster rows: the CPU's derived f32s verbatim — both
        // renderers light from bit-equal clusters (parity BY DATA, the ff
        // precedent). Scene-static, so they ride the base.
        let mut el_a = [[0.0f32; 4]; crate::emissive::MAX_EMISSIVE_LIGHTS];
        let mut el_b = [[0.0f32; 4]; crate::emissive::MAX_EMISSIVE_LIGHTS];
        for i in 0..scene.emissive.count as usize {
            let l = &scene.emissive.lights[i];
            el_a[i] = [l.pos[0], l.pos[1], l.pos[2], l.rc2];
            el_b[i] = [l.color[0], l.color[1], l.color[2], l.r_infl2];
        }
        FrameCb {
            sky_sh,
            cam_origin: [0.0; 4],
            cam_forward: [0.0; 4],
            cam_right: [0.0; 4],
            cam_up: [0.0; 4],
            sun: v4(sun, scene.sun.cos_radius),
            sun_e: v4(scene.sun.e_over_pi, scene.eps),
            sun_l: v4(scene.sun.radiance, scene.ao_radius),
            rw,
            rh,
            frame: 0,
            flags: 0,
            shadow_samples: 0,
            ao_samples: 0,
            reflections: 0,
            ff_scale: 1.0,
            frame_jitter: [0.0, 0.0],
            pixel_cone: 0.0,
            sky_scale: scene.sky_scale,
            cap_tile: 0,
            cap_leaf: 0,
            cap_sky: 0,
            cap_cut: 0,
            fb_mode: 0,
            fb_depth: 2,
            hemi_batch: HEMI_BATCH,
            cap_hemi_pt: rw * rh,
            cap_hemi_cell: 0,
            cap_hemi_leaf: 0,
            cap_hemi_cut: 0,
            ff_count: 0,
            ff: [[0.0; 4]; crate::fireflies::MAX_FIREFLIES],
            cloud_grid: [0.0; 4],
            // Unsplit: depth 0 means every consumer branches around the
            // ownership test, so the whole feature is off by default.
            split: [0; 4],
            sway_mv_base: [0; 4],
            el_meta: [scene.emissive.count, scene.light_gain.to_bits(), 0, 0],
            el_a,
            el_b,
            prev_origin: [0.0; 4],
            prev_forward: [0.0; 4],
            prev_right: [0.0, 0.0, 0.0, near],
            prev_up: [0.0, 0.0, 0.0, far],
            spp: 1,
            probe_sample: 0,
            night: scene.night,
            height_max: crate::bvh::height_max_world(scene),
            jitters: [[0.0; 4]; (crate::dlss::MAX_SPP as usize) / 2],
        }
    }

    /// This tracer's queue capacities, set once at construction.
    ///
    /// A SETTER rather than public fields because the fields are deliberately
    /// module-private (the struct doc's rule: outside constructors go through
    /// `base`/`with_frame`) — and because these seven are the wavefront's
    /// alone. `base` leaves them zero and the DXR pipeline never reads them,
    /// so a backend that forgot to call this would get a tracer whose every
    /// queue is full at the first push, which the overflow counter catches;
    /// one entry point is cheaper than relying on that.
    #[allow(clippy::too_many_arguments)]
    pub fn set_caps(
        &mut self,
        tile: u32,
        leaf: u32,
        sky: u32,
        cut: u32,
        hemi_cell: u32,
        hemi_leaf: u32,
        hemi_cut: u32,
    ) {
        self.cap_tile = tile;
        self.cap_leaf = leaf;
        self.cap_sky = sky;
        self.cap_cut = cut;
        self.cap_hemi_cell = hemi_cell;
        self.cap_hemi_leaf = hemi_leaf;
        self.cap_hemi_cut = hemi_cut;
    }

    /// The `--dual-gpu` band this frame renders, as the cbuffer's 64-bit mask
    /// + depth. Goes through `TileSplit::cb_row` so the packing stays private
    /// to the type that owns the arithmetic `cs_compose` mirrors.
    pub fn set_split(&mut self, split: TileSplit) {
        self.split = split.cb_row();
    }

    /// Re-derive the sun/sky rows from the scene after a TOD change
    /// (`scene::apply_tod`) — the shared body of `TraceGpu::refresh_sky` /
    /// `DxrGpu::refresh_sky`. Whole rows are copied from a fresh base, so the
    /// rehomed w slots (sun_e.w = eps, sun_l.w = ao_radius) are preserved by
    /// construction; every other field (queue caps included) is untouched.
    pub fn refresh_sky_rows(&mut self, scene: &Scene, rw: u32, rh: u32) {
        let fresh = FrameCb::base(scene, rw, rh);
        self.sun = fresh.sun;
        self.sun_e = fresh.sun_e;
        self.sun_l = fresh.sun_l;
        self.sky_sh = fresh.sky_sh;
        self.sky_scale = fresh.sky_scale;
        self.night = fresh.night;
        // The emissive rows too, which used to ride the static base alone.
        // They are scene-static under a fixed aperture, but `--autoexp-mode
        // lights` scales `EmissiveLight::color` per gain move, and this is the
        // one path that pushes a re-derived scene to the GPU — leaving them
        // behind would brighten every emitter's DISPLAY add (which reads the
        // gain from el_meta.y) while its cluster NEE stayed dark.
        self.el_meta = fresh.el_meta;
        self.el_a = fresh.el_a;
        self.el_b = fresh.el_b;
    }

    /// The per-frame fields folded onto the static base — the single source
    /// for the FrameParams -> cbuffer mapping (both dispatch flavors).
    pub fn with_frame(
        &self,
        p: &FrameParams,
        gbuf_full: bool,
        fsr_sig: bool,
        gbuf_ext: bool,
        nrd_sig: bool,
        sky_ext_skip: bool,
        nrd_rejitter: bool,
        remod_exact: bool,
        rclamp: (bool, bool),
    ) -> FrameCb {
        let (origin, forward, right, up, inv_w, inv_h) = p.cam.gpu_fields();
        let mut cb = *self;
        cb.cam_origin = [origin.x, origin.y, origin.z, inv_w];
        cb.cam_forward = [forward.x, forward.y, forward.z, inv_h];
        // The cloud state rides the cam rows' free w lanes (SCENE_DIAG /
        // CLOUD_TIME in the HLSL) — per-frame values on per-frame rows.
        cb.cam_right = [right.x, right.y, right.z, p.clouds.diag];
        cb.cam_up = [up.x, up.y, up.z, p.clouds.time];
        cb.frame = p.frame;
        cb.flags = (p.accumulate as u32 * FLAG_ACCUM)
            | (p.jitter as u32 * FLAG_JITTER)
            | (p.frame_jitter.is_some() as u32 * FLAG_FRAME_JITTER)
            | (p.verify as u32 * FLAG_VERIFY)
            | (gbuf_full as u32 * FLAG_GBUF)
            | (p.prev_cam.is_some() as u32 * FLAG_HAS_PREV)
            | ((gbuf_full && fsr_sig) as u32 * FLAG_FSR_SIG)
            // FSR-RR reads the sig lanes, which live in ext — so the sig flag
            // implies the ext flag by construction, not by convention.
            | ((gbuf_full && (gbuf_ext || fsr_sig)) as u32 * FLAG_GBUF_EXT)
            // The NRD RTGI fold rides the sig capture (it edits the lanes the
            // sig store writes), so it requires the sig flag by construction.
            | ((gbuf_full && fsr_sig && nrd_sig) as u32 * FLAG_NRD_GI)
            // Sky ext-store skip: only meaningful when the ext store runs at
            // all, so it requires the GBUF flag by construction (the branch
            // sits behind gbuf_write_sky's own FLAG_GBUF/FLAG_GBUF_EXT gates).
            | ((gbuf_full && sky_ext_skip) as u32 * FLAG_SKY_EXT_SKIP)
            // ReJitter reads the ext plane's normals (center + 4 neighbours),
            // so it requires the ext flag by construction — the same shape
            // the sig/GI terms above use.
            | ((gbuf_full && (gbuf_ext || fsr_sig) && nrd_rejitter) as u32 * FLAG_NRD_REJITTER)
            // Exact remodulation PRE-SCALES the sig lanes shade captures, so it
            // requires the sig flag by construction — the FLAG_NRD_GI shape,
            // and the same clause is what keeps it out of an FSR-RR session
            // (nrd_sig is false there, and that path's composite identity owns
            // these lanes).
            | ((gbuf_full && fsr_sig && nrd_sig && remod_exact) as u32 * FLAG_REMOD_EXACT)
            // The residual cap reconstructs R from the ext plane's kd/f0, so it
            // requires the ext flag by construction — the ReJitter shape. The
            // `hard` bit additionally requires its own parent: a HARD arm with
            // the cap disarmed would be a bit nothing reads.
            | ((gbuf_full && (gbuf_ext || fsr_sig) && rclamp.0) as u32 * FLAG_NRD_RCLAMP)
            | ((gbuf_full && (gbuf_ext || fsr_sig) && rclamp.0 && rclamp.1) as u32
                * FLAG_NRD_RCLAMP_HARD)
            | ((crate::texture::max_aniso() > 1.0) as u32 * FLAG_ANISO)
            | (p.clouds.enabled as u32 * FLAG_CLOUDS)
            // count > 0 already folds in the session enable + the night fade
            // (fireflies.rs::new) — a day session never sets the bit, so day
            // kernels are bit-identical by construction.
            | ((p.fireflies.count > 0) as u32 * FLAG_FIREFLIES)
            // The V toggle read at CB-build time (height_max > 0 encodes
            // any_height from base()) — no FrameParams plumbing needed, and
            // the HEIGHTFIELD compile-in stays per-scene.
            | ((crate::bvh::height_on() && self.height_max > 0.0) as u32 * FLAG_HEIGHT)
            // The --no-depth-tint lever, read at CB-build time like the V
            // toggle — the branch lives inside shade.hlsli's transmission
            // arm, which non-transmissive scenes never enter.
            | (crate::scene::depth_tint() as u32 * FLAG_DEPTH_TINT)
            // Emissive cluster NEE: the scene derived clusters (el rows ride
            // the base) × the live lever × NOT a GI frame — under fb.gi the
            // hemi gather already delivers emissive transport exactly, so
            // the cluster tier stands down (the inverted once-per-path
            // rule). NEE STAYS LIVE under RTGI (the NEE-keep rule): the
            // bounce's emissive display-add suppresses on this very bit
            // instead (shade.hlsli's `cam_lights || !FLAG_EMISSIVE` gate),
            // so exactly one mechanism delivers per frame. Emissive-free
            // scenes never set the bit.
            | ((self.el_meta[0] > 0
                && crate::emissive::enabled()
                && fb_mode_of(&p.q) != 2) as u32
                * FLAG_EMISSIVE)
            // Real-time GI: the session lever (baked as the RTGI compile
            // define; this runtime bit covers the fb stand-down) × NOT a
            // hemi frame — the still-frame tiers take precedence, so
            // shade_full's bounce block keys on the bit alone.
            | ((p.q.rtgi && fb_mode_of(&p.q) == 0) as u32 * FLAG_RTGI)
            // The --no-detail-tex lever, read at CB-build time (the
            // depth-tint shape) — shade.hlsli's post-match detail block,
            // gated per material on Mat.detail_scale > 0 (untextured
            // materials carry the synthetic scale since the untextured arm).
            | (crate::scene::detail_tex() as u32 * FLAG_DETAIL)
            | (crate::scene::detail_ao() as u32 * FLAG_DETAIL_AO)
            | (crate::scene::spec_aa() as u32 * FLAG_SPEC_AA)
            | (crate::scene::amb_bump() as u32 * FLAG_AMB_BUMP)
            // FR_WAVEVIZ live toggle, read at CB-build time like the V
            // toggle — unarmed sessions compile no WAVEVIZ block, so the
            // bit is only ever consumed where the code exists.
            | ((waveviz_on() && waveviz_live()) as u32 * FLAG_WAVEVIZ);
        cb.shadow_samples = p.q.shadow_samples;
        cb.ao_samples = p.q.ao_samples;
        cb.reflections = p.q.reflections as u32;
        cb.frame_jitter = match p.frame_jitter {
            Some((x, y)) => [x, y],
            None => [0.0, 0.0],
        };
        cb.pixel_cone = p.cam.pixel_cone();
        // Firefly poses: the CPU's baked f32 rows verbatim (CPU↔GPU positions
        // bit-equal by DATA — the HLSL re-derives nothing). Rows past the
        // count stay the base's zeros.
        cb.ff_count = p.fireflies.count;
        cb.ff_scale = p.fireflies.scale;
        for i in 0..p.fireflies.count as usize {
            cb.ff[i] = p.fireflies.pos[i];
        }
        cb.fb_mode = fb_mode_of(&p.q);
        cb.fb_depth = p.q.fb.depth.clamp(1, HEMI_MAX_DEPTH);
        // --spp. Pinned to 1 on fb frames, exactly like FrameCtx::spp(): the
        // leaf pass appends one hemi point per PIXEL (cap_hemi_pt = rw*rh),
        // and N hemispheres per pixel is the wrong way to converge a bounce.
        cb.spp = if cb.fb_mode > 0 { 1 } else { p.spp.clamp(1, crate::dlss::MAX_SPP) };
        cb.probe_sample = p.probe_sample.min(cb.spp - 1);
        for k in 0..cb.spp {
            let (x, y) = crate::dlss::jitter_for_sample(p.frame, k);
            let (row, half) = ((k / 2) as usize, (k % 2) as usize * 2);
            cb.jitters[row][half] = x;
            cb.jitters[row][half + 1] = y;
        }
        if let Some(pc) = &p.prev_cam {
            // The near/far riding the w slots of the last two rows come from
            // the base and must survive the overwrite.
            let (po, pf, pr, pu, piw, pih) = pc.gpu_fields();
            cb.prev_origin = [po.x, po.y, po.z, piw];
            cb.prev_forward = [pf.x, pf.y, pf.z, pih];
            cb.prev_right = [pr.x, pr.y, pr.z, cb.prev_right[3]];
            cb.prev_up = [pu.x, pu.y, pu.z, cb.prev_up[3]];
        }
        cb
    }

    /// Arm the sway-MV correction for this frame: FLAG_SWAY_MV + the frame's
    /// dmv-ring slot base, one call so the pair cannot split (a flag without
    /// its base indexes slot 0's stale rows). Callers (both tracers'
    /// write_cb) gate on `sway_mv_pair` + the session's SWAY_MV compile-in +
    /// the slot fill having run.
    pub(crate) fn arm_sway_mv(&mut self, base: u32) {
        self.flags |= FLAG_SWAY_MV;
        self.sway_mv_base = [base, 0, 0, 0];
    }

    /// Copy into a persistently-mapped CB ring slot.
    /// The packed cbuffer as bytes — the same image `store` writes, for a
    /// backend whose upload takes a slice instead of a mapped pointer. ONE
    /// packing serves both, which is the whole point of `-fvk-use-dx-layout`.
    pub fn bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const FrameCb as *const u8,
                std::mem::size_of::<FrameCb>(),
            )
        }
    }

    /// The same image, into a mapped CB ring slot.
    pub(crate) fn store(&self, ptr: *mut u8) {
        let b = self.bytes();
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), ptr, b.len()) };
    }
}

/// Everything that varies per frame, CPU-side.
pub struct FrameParams {
    pub cam: CamBasis,
    pub frame: u32,
    pub accumulate: bool,
    pub jitter: bool,
    pub frame_jitter: Option<(f32, f32)>,
    /// Previous frame's camera basis for G-buffer motion vectors (upscaler
    /// sessions; None = mv (0,0), consumed as disocclusion).
    pub prev_cam: Option<CamBasis>,
    pub q: Quality,
    /// Check builds: hemi claim re-validation + PSA accounting on the GPU.
    pub verify: bool,
    /// --spp: primary samples per pixel this frame (1..=dlss::MAX_SPP; the CB
    /// pins it to 1 when fb is on). The samples share the tile's inherited
    /// t_start and average into one partial write — accum semantics unchanged.
    pub spp: u32,
    /// Which sample writes tbuf/info/the G-buffer pack. 0 in every real frame;
    /// the check suites sweep it 0..spp so every sample's ray is gated.
    pub probe_sample: u32,
    /// Per-frame cloud state (src/clouds.rs) — enable bit + clock + diag,
    /// mapped onto FLAG_CLOUDS and the cam rows' w lanes by `with_frame`.
    pub clouds: crate::clouds::Clouds,
    /// Per-frame firefly state (src/fireflies.rs) — CPU-baked poses, mapped
    /// onto FLAG_FIREFLIES + the `ff`/`ff_count` CB rows by `with_frame`
    /// (count 0 — every day session — writes neither).
    pub fireflies: crate::fireflies::Fireflies,
    /// --foliage-sway clock for THIS frame (the shared cloud_time), or None =
    /// trace the static rest-pose TLAS. Consumed by BOTH ray pipelines since
    /// v0.2 (each rebuilds `SceneGpu::sway`'s ring TLAS on its list and
    /// binds it — DxrGpu via its sway_t stash, TraceGpu via `record_sway`);
    /// every headless gate/bench passes None — which, plus `sway: None` on
    /// unarmed uploads, is the structural off-state (src/foliage.rs).
    pub sway_time: Option<f32>,
    /// The PREVIOUS frame's sway clock, paired with `prev_cam`'s frame by
    /// main.rs (the PrevPose rule — set beside the camera after a successful
    /// present, cleared with it, so the pair cannot desync). Some + bit-
    /// different from `sway_time` + `prev_cam` Some arms the sway-MV
    /// correction (`sway_mv_pair`); None — every headless gate/bench/spin
    /// site — is the structural camera-only arm.
    pub sway_prev_time: Option<f32>,
    /// Structure-replay enable (opts.replay). When true AND this frame's basis
    /// bit-equals the previous producing frame's, `record_frame` re-dispatches
    /// the persisted terminal queues instead of re-running seed + the ladder
    /// (the GPU mirror of src/replay.rs). Replay frames re-shade fresh — the
    /// leaf shader's MV write included — so the sway-MV fill and CB arming
    /// run on them like any producing frame. NOT a global atomic: every
    /// headless gate/bench sets it false so nothing silently switches paths
    /// under a measurement.
    pub replay: bool,
}
