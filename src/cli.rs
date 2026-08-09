//! The command line: `Opts` (what the flags said) plus the parser that
//! produces it, lifted out of a 17k-line `main.rs` so the CLI is one readable
//! unit and — the part that is load-bearing — so it can be GATED.
//!
//! # `parse_from` is pure, and that is the whole point
//!
//! It reads no process state and writes none. Nineteen flag arms used to store
//! straight into the "knob before scene load" globals from inside the parse
//! loop (`texture::set_mips`, `bvh::set_height_armed`, `clouds::set_enabled`,
//! `gpu::trace::set_cloud_shadow`, ...), and `settings::apply_to_opts` wrote
//! those SAME statics just before it — two writers whose layering held by call
//! ordering alone. They are ordinary `Opts` fields now, applied to the globals
//! exactly once by `main`, in the block that already prints the lever lines.
//!
//! Two things fall out. Precedence stops being an ordering accident and becomes
//! a data flow you can read: `defaults()` → `settings::apply_to_opts` seeds the
//! fields → `parse_from` overwrites them. And `self_test` can run inside
//! `--check`: parsing an argv from within a gate would otherwise stomp the very
//! texture/BVH/effect state that same `--check` run is using.
//!
//! **So a new flag adds a field here plus one line in `main`'s lever block. A
//! setter call in the parse loop silently un-gates the parser.**
//!
//! Diagnostics follow the same rule: notices accumulate into `Cli::notes` and
//! `main` prints them, so parsing inside a gate stays quiet. Hard errors still
//! `exit(2)` where they happen — the gates only ever feed this valid input, and
//! an invalid command line has nowhere useful to return to.

use crate::camera::Camera;
use crate::{
    bc7, blas_split, bloom, bvh, cinematic, clouds, dlss, emissive, fireflies, frd, fsr, gpu,
    nrd, oidn, scene, settings, texture, tone, upchain, xess,
};
use glam::Vec3A;

/// CLI options beyond the OBJ path / --check.
///
/// `Clone` exists for exactly one caller: `run_window` takes a private copy so
/// `vendor_defaults` can move a default the user left alone, once the adapter
/// is known. Parsing stays the single source of the user's intent; the copy is
/// where the HARDWARE's opinion is folded in.
#[derive(Clone)]
pub struct Opts {
    /// The temporal-upscaler fallback chain: which levels of
    /// DLSS-RR → FSR4-RR → XeSS → FSR3 may be probed (first supported level
    /// wins; upscaling is always on unless --no-upscale empties the chain).
    /// `--<x>` force-starts the chain at that level, `--no-<x>` skips it.
    pub chain: upchain::UpChain,
    /// D3D12 debug layer + GPU-based validation.
    pub gpu_debug: bool,
    /// Block-compress the OPAQUE scene textures to BC7 on upload (8 bpp vs
    /// 32), GPU upload only — the CPU renderer keeps sampling the exact RGBA8
    /// texels, so this moves the GPU-vs-CPU statistical gates (albedo A/B,
    /// T2 radiance) and nothing else. Alpha-masked cutout textures are never
    /// compressed (src/bc7.rs). ON BY DEFAULT via the GPU compute encoder
    /// (`Gpu(Fast)` — gpu/bc7gpu.rs; only GPU/DXR sessions ever read this,
    /// so CPU-only sessions are structurally unaffected): `--no-bc7` kills,
    /// `--bc7-cpu` is the ispc A/B arm (the old ~20 s-per-Sponza encode),
    /// `--bc7-quality` keys the current arm. There is still deliberately no
    /// BC7 disk cache — the GPU encode is what made per-load affordable.
    pub bc7: bc7::Bc7Mode,
    /// Audio ambience (biome loops + speed-scaled wind; default on —
    /// interactive sessions only, headless paths never initialize audio).
    /// --no-audio is the kill lever: the subsystem is never constructed.
    pub audio: bool,
    /// Start with OIDN denoising on (N toggles at runtime; default off —
    /// DLSS-RR stays the primary denoiser).
    pub oidn: bool,
    /// Directory holding OpenImageDenoise.dll + its core/device DLLs.
    pub oidn_path: String,
    /// OIDN device type (oidn.h OIDNDeviceType; 0 = auto-pick fastest).
    pub oidn_device: i32,
    /// OIDN RT-filter quality (`oidn::QUALITY_*`; default balanced — HIGH is
    /// documented for final frames, the flag lets stills opt in).
    pub oidn_quality: i32,
    /// Declare the OIDN albedo/normal guides noise-free (default on — they
    /// are deterministic primary-hit values; --oidn-no-clean-aux is the
    /// empirical escape hatch, same policy as the sign/flag constants).
    pub oidn_clean_aux: bool,
    /// OIDN temporal reprojection history (M toggles at runtime; default on —
    /// off means the plain accumulation-average mode that shimmers while
    /// moving).
    pub oidn_temporal: bool,
    /// Start with NPPD neural denoising on (J toggles; mutually exclusive
    /// with DLSS/OIDN/XeSS — NPPD is its own recurrent temporal integrator).
    pub nppd: bool,
    /// Directory holding onnxruntime.dll (+ DirectML.dll).
    pub nppd_path: String,
    /// The exported NPPD ONNX graph (tools/nppd-export/export.py).
    pub nppd_model: String,
    /// NPPD execution provider: None = DirectML then CPU fallback,
    /// Some(-1) = CPU forced, Some(n) = DirectML adapter n forced.
    pub nppd_device: Option<i32>,
    /// NRD (ReBLUR) pre-upscale denoising — ON BY DEFAULT for the sessions
    /// that can arm it (GPU tracers × XeSS/FSR3; `--no-nrd` is the kill
    /// lever, `--nrd` spells the default explicitly). The hand-crafted
    /// (non-neural) temporal denoiser that cleans the 1-spp signal at render
    /// res before the TAA-upscaler runs. DLSS-RR / FSR4-RR sessions never
    /// arm it (they already denoise); a missing NRD.dll sheds loudly to
    /// plain upscaling. Excl. --nppd (both claim the pre-upscale color
    /// slot): the EXPLICIT pair exits 2, a merely-defaulted nrd disarms with
    /// a loud line instead — the fg_explicit pattern (a default must never
    /// make another flag fatal). Mirrored in settings.rs (upscaler.nrd — the
    /// file sets this field only, never nrd_explicit, per that same rule).
    pub nrd: bool,
    /// True only when --nrd was NAMED on the command line — what makes the
    /// --nppd conflict fatal vs a loud disarm, and what gates the
    /// "not armed" session notes (a default shouldn't nag DLSS sessions).
    pub nrd_explicit: bool,
    /// Directory holding NRD.dll (install-prerequisites.bat nrd builds it).
    pub nrd_path: String,
    /// Load the REBLUR_PERFORMANCE_MODE build of NRD.dll (`<nrd_path>\perf`,
    /// the install script's second cmake tree) — cheaper ReBLUR internals
    /// (6-tap Poisson, frame rotators, no Catmull-Rom history), same dispatch
    /// count, lower quality. Perf mode is a COMPILE-TIME NRD option (no
    /// ReblurSettings field exists for it in 4.17.3), hence a second DLL.
    /// A missing perf DLL falls back to the standard one with a loud line.
    pub nrd_perf: bool,
    /// Runtime ReblurSettings overrides (--nrd-max-stabilized-frames,
    /// --nrd-prepass-radius, --nrd-no-anti-firefly, --nrd-max-accum-frames) —
    /// the fsr_tune shape: all None = the settings the session always sent.
    pub nrd_tune: nrd::ReblurTuning,
    /// FRD — the from-scratch clean-room pre-upscale denoiser (src/frd.rs;
    /// same arming surface as NRD: GPU tracers × XeSS/FSR3). OPT-IN until it
    /// reaches NRD parity (the plan's Phase-E default flip): `--frd` takes
    /// the one denoiser slot. Only EXPLICIT pairs are fatal (main's lever
    /// block): explicit --frd + explicit --nrd or + --nppd exits 2; an
    /// explicit --frd silently disarms the defaulted nrd (opting into FRD is
    /// opting out of the default NRD); a FILE-defaulted frd (the settings
    /// row seeds this field WITHOUT frd_explicit) yields loudly to an
    /// explicit --nrd/--nppd instead — a default never makes another flag
    /// fatal.
    pub frd: bool,
    /// True only when --frd was NAMED (the nrd_explicit pattern) — gates the
    /// "not armed" session notes.
    pub frd_explicit: bool,
    /// Runtime FRD tuning (--frd-* levers) — the nrd_tune shape: all None =
    /// the compiled constants in frd.rs.
    pub frd_tune: frd::FrdTuning,
    /// Directory holding libxess.dll.
    pub xess_path: String,
    /// Directory holding amd_fidelityfx_loader_dx12.dll + the provider DLLs.
    pub ffx_path: String,
    /// Frame generation for the session's wired upscaler family — ON BY
    /// DEFAULT (`--no-fg` is the kill lever; `--fg` spells the default
    /// explicitly). DLSS sessions run raw-NGX DLSS-G (SDK builds); FSR
    /// sessions (FSR4-RR / FSR3) wrap the swapchain with the FidelityFX
    /// frame-interpolation proxy; Intel XeSS sessions XeSS-FG — one
    /// generated frame per rendered frame; unsupported pairings fall
    /// through with a loud line. Interactive sessions only — headless
    /// paths never consult it.
    pub fg: bool,
    /// Whether `--fg` was PASSED rather than defaulted (the `mode_explicit`
    /// pattern). Currently informational — its last consumer (the
    /// --quinlight exit-2) died when frame generation learned to compose
    /// with the fuse; kept because the passed-vs-defaulted fact is cheap to
    /// carry and expensive to reconstruct.
    pub fg_explicit: bool,
    /// Directory holding amd_fidelityfx_framegeneration_dx12.dll (`--fg-path`;
    /// the prebuilt drop ships it in the FSR sample dir, not next to the
    /// loader).
    pub fg_path: String,
    /// `--fsr4`: the FSR4 + Ray Regeneration level is REQUIRED, not merely
    /// force-started. `--fsr` falls through to XeSS/FSR3 when the Ray
    /// Regeneration provider is absent (no RDNA4, wrong adapter); this makes
    /// that a hard error instead — the one place in the codebase where an
    /// unsupported feature is not a loud-line fallback, because the flag's
    /// whole point is to be told.
    pub fsr4_required: bool,
    /// Ray Regeneration tuning overrides (`--fsr-max-radiance` &c). All-None
    /// by default: a flagless session configures nothing and runs the
    /// provider's own constants. A/B levers for dialing in cleanliness —
    /// `max_radiance` is the firefly clamp.
    pub fsr_tune: fsr::DenoiseTuning,
    /// `--quinlight`: wire EVERY supported chain level at once, run them all
    /// over the same traced frame, and present the registered-consensus fuse of
    /// their outputs (gpu/quin.rs — LK registration + warp + winsorized mean;
    /// a port of quinlight-player's consensus_registered.comp). GPU-fed only
    /// (--dxr or --gpu): there is no CPU-fed arm.
    pub quin: bool,
    /// `--quin-anchor N`: which engine is the fuse's anchor (never warped;
    /// defines the spatial frame). None = engine 0 = the highest wired chain
    /// level, which is a DENOISING engine wherever the box has one.
    pub quin_anchor: Option<u32>,
    /// Explicit GPU adapter vendor preference (--prefer-nvidia /
    /// --prefer-intel / --prefer-amd). None = the mode default (AMD for
    /// --fsr, NVIDIA otherwise). A preference, not a requirement: the
    /// feature-support probes still gate, so e.g. DLSS on a non-NVIDIA pick
    /// logs and falls back rather than erroring.
    pub prefer: Option<gpu::adapter::Prefer>,
    /// Start XeSS mode with the OIDN denoise placed AFTER the upscale
    /// (requires --xess; N cycles placement at runtime).
    pub oidn_post: bool,
    /// XeSS internal autoexposure (XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE;
    /// default off — A/B lever, init-time only).
    pub xess_autoexposure: bool,
    /// Adaptive shading rate on XeSS frames (default on; --no-adaptive forces
    /// uniform per-pixel shading — visibility is per-pixel either way, only
    /// the 2×2-cell shadow/AO sharing and HOT top-ups are disabled).
    pub adaptive: bool,
    /// Lock the DLSS/XeSS render resolution to this fixed scale of the
    /// window (default `xess::DEFAULT_LOCK_SCALE` = native 100%;
    /// `--lock-res dynamic` -> None = the step-wise dynamic-resolution
    /// controller). CLI-only, no runtime toggle — T prints the locked note.
    /// ONE scale for every render mode: the CPU tracer, `--gpu` and `--dxr`
    /// all trace at it, so F/SPACE cycling arms never moves the render res.
    /// (The GPU arms used to default to native 100% through a second
    /// `gpu_lock_scale` field; that split is gone.) `--lock-res dynamic` is
    /// not honorable on the GPU arms, which lock at the default instead with
    /// a loud line.
    pub lock_scale: Option<f32>,
    /// Primary samples per pixel per frame (--spp, 1..=dlss::MAX_SPP; U cycles
    /// it live). N jittered samples inside each pixel share the tile's
    /// inherited t_start/cut (the quadtree cost amortizes over N× the rays)
    /// and are averaged into ONE splat, so the upscalers/denoisers get a
    /// ~1/N-variance frame at unchanged accum semantics. 1 = today's renderer,
    /// bit-identically.
    pub spp: u32,
    /// Master A/B lever (--no-temporal): disable ALL previous-frame quadtree
    /// reuse — no temporal cache produced or consumed, no claim ring, no cut
    /// store, no structure replay. Every frame proves its empty space from
    /// scratch. Default on.
    pub temporal: bool,
    /// A/B lever (--no-replay): keep temporal seeding but disable the
    /// static-frame structure replay (and its recording). Default on.
    pub replay: bool,
    /// A/B lever (--no-adopt): keep temporal seeding but disable the
    /// query skip / cut adoption (and CutStore production). Default on.
    pub adopt: bool,
    /// A/B/C lever (--discard-seeds): run the whole temporal pipeline —
    /// lookups, ring retries, cache + cut-store production — but consume
    /// nothing, so every frame traces exactly like --no-temporal while
    /// paying the machinery's full cost. With --spin this isolates cost
    /// from benefit as wall-clock differences: (this − --no-temporal) =
    /// pure cost, (default − this) = gross benefit. Default off.
    pub discard_seeds: bool,
    /// Deferred material-sorted shading (--defer-shade): plain-path leaf
    /// tiles trace but defer shading; same-material runs merge up the
    /// quadtree (≤ 64×64 px) and flush as single cache-coherent bursts.
    pub defer_shade: bool,
    /// A/B lever (--no-hemi-share): disable the shared hemisphere capture
    /// in fb (H) frames — every shading point runs its own bounce tree.
    /// Default on.
    pub hemi_share: bool,
    /// A/B lever (--no-cut-rays): cut-SEEDED rays (primary leaf-tile, hemi
    /// leaf, shaft) traverse from the BVH root instead of from their node
    /// cut. The inherited t_start is UNAFFECTED — it is a scalar, not a node
    /// reference — so this isolates what the CUT is worth to the ray path
    /// from what the inherited distance bound is worth. That is the question
    /// that decides whether the frustum structure and the ray BVH can be
    /// separate trees (a cut over one tree cannot index another). On the GPU
    /// the cut already seeds nothing, so the answer there is already zero.
    /// Default on.
    pub cut_rays: bool,
    /// A/B lever (--continuation-rays; --sw-rays is the technical alias): the
    /// GPU WAVEFRONT tracer's rays traverse the software BVH instead of DXR
    /// inline RayQuery. A terminal beam query mints an opaque TraversalFrontier
    /// that every leaf ray/sample reuses — the input current RayQuery cannot
    /// accept. Composes with --no-cut-rays (same software traversal from the
    /// root) and --no-ftree (binary cuts, no translation). Wavefront only;
    /// --dxr and the CPU renderer are untouched. Default off.
    pub sw_rays: bool,
    /// A/B lever (--cut-hemi re-enables, --no-cut-hemi is the default): HEMI
    /// leaf rays traverse from the root instead of from their bounce cut.
    /// Split out from --no-cut-rays because the two consumers disagree: the
    /// primary path's cuts are short (~18) and seeding pays ~10%, while a hemi
    /// cut sits pinned at HEMI_CUT (64) and seeding from it measured 3-10%
    /// SLOWER than a root descent on both the historical and the M2 tree. The
    /// cut still drives the bound QUERIES either way, and --check's probe
    /// gates force seeding ON so cut-miss keeps exercising the machinery.
    /// Default off.
    pub cut_hemi: bool,
    /// SAH traversal cost as a ratio to the intersection cost (--bvh-ctrav;
    /// C_isect is fixed at 1, only the ratio means anything). 0.0 reproduces
    /// the historical COST FUNCTION: with no traversal charge the leaf test can
    /// never fire (split cost is unconditionally <= leaf cost) and every subtree
    /// recurses to the count<=2 floor, which is why the trees came out at ~1.2
    /// nodes/tri. It does NOT reproduce the historical TREE, because the sweep
    /// also bins all three axes where the original binned only the widest.
    /// Default 0.0 pending the sweep; see bvh::C_TRAV_BITS.
    pub c_trav: f32,
    /// Hard ceiling on triangles per leaf (--bvh-maxleaf). Above this a node
    /// splits regardless of what SAH says — phase 1 of the parallel build
    /// relies on a too-big node always splitting.
    pub max_leaf: usize,
    /// Axes the binned SAH searches (--bvh-axes 1|3). 1 = the historical build
    /// (widest centroid axis only); 3 = all axes, global best. A knob rather
    /// than a hardcode so the axis change and the C_trav change stay separately
    /// attributable — they landed together.
    pub split_axes: usize,
    /// The M7 bake-off lever (--bvh-builder sah|lbvh|ploc|som): which
    /// algorithm builds the ray BVH. Every builder produces the same Bvh
    /// (consumers/gates/.fcache unchanged); score on measured counters.
    pub bvh_builder: String,
    /// Triangles per BLAS: cut the ray BVH into maximal subtrees of at most N
    /// triangles and build ONE BLAS per subtree, instanced identity into the
    /// TLAS (blas_split.rs). DEFAULT ON at `DEFAULT_MAX_PRIMS`; `None` =
    /// --no-blas-split = one BLAS over the whole scene (the pre-feature build).
    ///
    /// It is the default for ROBUSTNESS, not throughput — on NVIDIA it is
    /// neutral (measured within +-1% on four world poses). BLAS scratch is
    /// sized by the largest single geometry, so one 34.4M-triangle BLAS made
    /// Intel's driver ask for 1891 MB of it and REMOVE THE DEVICE mid-boot on
    /// THE WORLD (NVIDIA asked 276 MB for the same build and survived); at a
    /// 64k cap the scratch is a function of one chunk — 3 MB — and the same
    /// session runs. GPU-only and derived from the built BVH, so it keys
    /// nothing in the scene cache; the CPU tracer, the software BVH and the
    /// frustum cut never see it.
    pub blas_split: Option<u32>,
    /// `--dual-gpu N`: split the frame across two adapters, giving the
    /// SECONDARY `N` of the 8 level-3 tile rows (1..=7). `None` = off, the
    /// default and structurally the pre-feature renderer.
    ///
    /// Expressed as a share of eighths rather than a boolean because the
    /// optimal share is NOT the compute-balanced one and cannot be guessed: it
    /// minimises `max(T(1-s), r*T*s + s*K)` over the payload, the link speed
    /// AND the tracer cost, and on this box the secondary's link is 4.6x
    /// slower than the primary's (a consumer board's second x16-length slot is
    /// electrically x4). So the knob exists to be SWEPT — `--dual-gpu 1..7`
    /// walks the curve and finds the minimum empirically, which is what the
    /// eventual balancer has to converge to anyway.
    ///
    /// Level 3 is `MAX_SPLIT_DEPTH`: 8 rows is the finest the CB's 64-bit mask
    /// expresses, and finer than the balancer could usefully act on.
    pub dual_gpu: Option<u32>,
    /// `--dual-gpu-auto`: hand the share to the balancer instead of pinning
    /// it. `dual_gpu`'s value is then the STARTING share, not a fixed one.
    /// Implies arming, since a balancer with nothing to balance is a no-op.
    pub dual_gpu_auto: bool,
    /// `--dual-gpu-depth K`: the quadtree level the split is assigned at,
    /// 1..=`MAX_SPLIT_DEPTH`. Trades balance granularity against DUPLICATED
    /// LADDER WORK: levels `0..K` run identically on both devices, so K=3
    /// gives 8 rows of granularity but duplicates 21 of the tree's 1365 tiles
    /// — and they are the most expensive ones, being the shallowest. K=1
    /// duplicates a single tile and offers only a half-and-half split.
    pub dual_gpu_depth: u32,
    /// `--dual-gpu-arm wave|dxr`: force the SECONDARY's pipeline instead of
    /// letting its own adapter's vendor choose it (`gpu::dual::arm_for`).
    /// `None` = the vendor policy, which is the shipping default.
    ///
    /// It names the SECONDARY ONLY. The primary's pipeline is the session's,
    /// picked by --gpu/--dxr/SPACE, and this flag never touches it — the two
    /// are independent and all four combinations are legal.
    ///
    /// A secondary that cannot host a `DxrGpu` (no RT tier 1.0) degrades to
    /// the wavefront with a loud line even when this asks for DXR: that floor
    /// is soundness, not policy.
    pub dual_gpu_arm: Option<crate::gpu::dual::Arm>,
    /// A/B lever (--no-ftree disables): route ALL bound queries through the
    /// 8-wide frustum tree lazily collapsed from the ray BVH (ftree.rs) — the
    /// two-tree split. Rays always stay on the binary BVH.
    /// Default on: -15/-17% hemi-ao, -4/-8% hemi-gi (default scene + San
    /// Miguel), bounds bit-identical per the self-test.
    pub ftree: bool,
    /// Per-consumer A/B lever (--ftree-tiles): the primary tile recursion
    /// (tile_step/adopt_step bound queries + refines) on the wide tree, cuts
    /// translated to binary ray roots once per leaf tile. Default OFF —
    /// measured wall-neutral on San Miguel, ~10% slower on stress no-temporal
    /// (see ftree::FTREE_TILES). Hemi keeps its own wiring; --no-ftree kills
    /// both; --check verifies the wired path regardless (the wide-tiles gate).
    pub ftree_tiles: bool,
    /// A/B lever (--no-wide-levels disables): run the SHALLOW GPU quadtree
    /// levels on the wave-cooperative kernel (one 32-lane group per tile,
    /// sharing a breadth-first frontier) instead of one thread per tile.
    /// Default on — the shallow ladder is under-occupied (level 0 is a single
    /// lane descending the whole BVH). Same-seed image A/B unchanged to the
    /// digit (BFS min is order-independent — see trace::WIDE_LEVELS); GPU only.
    pub wide_levels: bool,
    /// GPU-resident tracing (--gpu): the whole quadtree + shading runs in
    /// D3D12 compute with DXR RayQuery rays. Requires the DXC DLL drop and
    /// RT tier 1.1; falls back to the CPU path with a loud line otherwise.
    pub gpu: bool,
    /// The DXR DispatchRays pipeline as the session's render mode (default
    /// ON — the F key toggles it live against the CPU tracer; --cpu opts
    /// back into the CPU renderer). Requires the DXC DLL drop and RT tier
    /// 1.0; falls back to the CPU path with a loud line.
    pub dxr: bool,
    /// Did the user name a render mode at all (`--cpu` / `--gpu` / `--dxr`)?
    /// `dxr` defaults to true, so "is DXR on" cannot answer "did they ASK for
    /// DXR" — and two places need that distinction: `--spin` (which drives the
    /// CPU renderer unless a GPU arm was requested) and the vendor-default
    /// policy in `run_window`, which may only move a default the user left
    /// alone. `dxr_explicit` already existed as a parse-local for the
    /// OIDN/NPPD opt-out; this is the same idea, promoted so it survives parse.
    pub mode_explicit: bool,
    /// Did the user pass `--lock-res`? Same reason: `lock_scale` has a
    /// non-None default, so its value cannot report whether it was chosen.
    pub lock_res_explicit: bool,
    /// Directory holding dxcompiler.dll + dxil.dll.
    pub dxc_path: String,
    /// PIX Begin/End events on the D3D12 command lists (--pix-markers;
    /// default off so unprofiled sessions stay byte-identical). Needs
    /// WinPixEventRuntime.dll under --pix-path; missing DLL = loud line,
    /// markers stay off.
    pub pix_markers: bool,
    /// Directory holding WinPixEventRuntime.dll.
    pub pix_path: String,
    /// D3D12 timestamp queries around the PIX marker brackets, printed as a
    /// per-region GPU-ms table (--gpu-timing; default off, zero-cost when
    /// off — no query heap, no name allocation, byte-identical lists). The
    /// vendor-neutral profiler, and vendor-neutrality is the whole point:
    /// PIX's capture ANALYSIS only replays on AMD/NVIDIA, so on an Intel
    /// adapter this is the only way to get per-pass GPU numbers at all — and
    /// the per-pass AMD-vs-NVIDIA diff it makes possible is what found the
    /// leaf kernel's wave64 bug.
    pub gpu_timing: bool,
    /// V-sync'd presentation (default on). `--no-vsync` presents at sync
    /// interval 0 on a tearing swapchain so interactive frame times measure
    /// the renderer, not the monitor refresh.
    pub vsync: bool,
    /// Present through the 10-bit swapchain (R10G10B10A2, 4 B/px — the ONE
    /// format; PQ on an HDR-on display, gamma-2.2 "deep colour" on an HDR-off
    /// one, the probe decides). **On by default**: half the bytes per present
    /// of the retired scRGB f16 chain — which is the whole frame budget when
    /// the display hangs off a different GPU than the renderer and every
    /// present is a DWM copy — while 10-bit gamma keeps the no-banding
    /// quality f16 bought on SDR panels. `--no-hdr` forces the legacy 8-bit
    /// swapchain — the A/B lever, and the ladders' last rung.
    pub hdr: bool,
    /// `--hdr10`: force the PQ (G2084) declaration in any session — the A/B
    /// lever. Override-wins like `--hdr-peak` (it fires even where the
    /// display probe says HDR is off). Only meaningful with `hdr` (the parse
    /// arms keep the trio consistent — later flags win).
    pub hdr10: bool,
    /// `--no-hdr10`: force the 10-bit gamma-2.2 (Sdr10) arm — "10-bit, but
    /// NOT PQ" — even on an HDR-on display. PQ is the default there, so Sdr10
    /// needs its own spelling or the arm the gates rest on would be
    /// unreachable from the command line on an HDR box.
    /// `hdr && !hdr10 && !sdr10` = auto-pick by the display probe.
    pub sdr10: bool,
    /// `--hdr-paper-white <nits>`: where linear 1.0 lands. The scene is authored
    /// so 1.0 ≈ diffuse white; 200 is the usual desktop-HDR reference.
    pub hdr_paper_white: f32,
    /// `--hdr-peak <nits>`: override the display's reported peak. None = trust
    /// what the monitor says (`gpu::display::probe`).
    pub hdr_peak: Option<f32>,
    /// `--tod <hour>`: start time-of-day (float hours, wrapped into 0..24 —
    /// e.g. 17.5 = sunset light). None = the default sun, bit-identical to
    /// the pre-TOD renderer; the `,`/`.` keys and D-pad still scrub live.
    /// Applied AFTER the scene cache load/store so a `--tod` session can
    /// never poison the `.fcache` with a non-default sun.
    pub tod: Option<f32>,

    // ---- The "knob before scene load" levers --------------------------------
    //
    // Every field below is applied to its process-global static EXACTLY ONCE,
    // by `main`, in the lever block that also prints the departure lines. The
    // parse loop must never call a setter — see the module header for why.
    /// `--no-mips` clears: no texture mip chains, so every trilinear sample
    /// degenerates to the pre-mip bilinear (`texture::set_mips`).
    pub mips: bool,
    /// `--aniso N`; `--no-aniso` = 1. Max anisotropy in 1..=MAX_ANISO_CAP
    /// (`texture::set_aniso`). Applied AFTER `mips`, and that order is the
    /// CLI's own contract rather than a style choice: `set_mips(false)` forces
    /// aniso to 1 and `set_aniso` re-reads the mips switch, so mips-then-aniso
    /// is the only order in which `--no-mips` still implies `--no-aniso`.
    pub aniso: u32,
    /// `--no-h2n` clears: grayscale bump maps are dropped instead of being
    /// Sobel-converted into normal maps at load (`texture::set_h2n`). Keys the
    /// scene cache.
    pub h2n: bool,
    /// `--no-n2h` clears: no Frankot-Chellappa heightfield derived from normal
    /// maps at load (`texture::set_n2h`). Keys the scene cache.
    pub n2h: bool,
    /// `--no-slope-mips` clears (`texture::set_slope_mips`): normal-map mip
    /// chains fall back to the legacy raw-byte box filter, which under-tilts
    /// (the "normal maps flatten with distance" behavior). Default on: mips
    /// of normal-role textures average SLOPES. Derived-only (mips are never
    /// persisted), so it does NOT key the scene cache — the --no-mips class.
    pub slope_mips: bool,
    /// `--no-spec-aa` clears (`scene::set_spec_aa`): no slope-variance →
    /// roughness fold — mip-averaged normal-map detail and faded detail-field
    /// octaves vanish with distance instead of widening the GGX lobe (the
    /// pre-feature behavior, bit-identical). Default on. Derived-only (the
    /// variance companions are never persisted), so it does NOT key the
    /// scene cache — the --no-mips class. `--no-slope-mips`/`--no-mips` kill
    /// the map half automatically; the detail-field half is independent.
    pub spec_aa: bool,
    /// `--normal-strength K`: session multiplier on every material's
    /// `normal_scale`, applied post-cache in `load_scene` (the --tod
    /// placement class — never baked into a sidecar, and relief's
    /// `height_amp` stays unscaled). 1.0 = bit-identical off arm; 0.0 =
    /// normals fully off (the A/B floor).
    pub normal_strength: f32,
    /// `--no-tinted-shadows` clears (`scene::set_tinted_shadows`).
    pub tinted_shadows: bool,
    /// `--no-spray` clears (`scene::set_spray`). Keys the cache lever word.
    pub spray: bool,
    /// `--no-depth-tint` clears (`scene::set_depth_tint`).
    pub depth_tint: bool,
    /// `--no-detail-tex` clears (`scene::set_detail_tex`): Unreal-1 style
    /// procedural close-up detail (albedo grain + micro-bump on magnified
    /// textured hits). Runtime shading lever — the depth-tint class.
    pub detail_tex: bool,
    /// `--no-detail-ao` clears (`scene::set_detail_ao`): the detail field's
    /// pits darken ambient + direct specular (texel-scale sky-visibility
    /// contrast on flat sun-facing surfaces). Runtime shading lever — the
    /// depth-tint class; a no-op wherever detail-tex never fires.
    pub detail_ao: bool,
    /// `--detail-strength K` (0.0..=4.0, default 0.5 — the 2026-08-06
    /// feel-test calibration; 1.0 spells the original full-strength field,
    /// and ×1.0 is the bit-identical arm): session multiplier on the detail
    /// GRAIN family's amplitudes (albedo grain + micro-bump —
    /// `scene::set_detail_strength`; the GPU twin is the injected DETAIL_STR
    /// define). 0 = grain off, the A/B floor.
    pub detail_strength: f32,
    /// `--detail-ao-strength K` (0.0..=4.0, default 0.125 — the same
    /// feel-test; 1.0 = the original amplitudes): session multiplier on the
    /// detail AO family's amplitudes (pools + cavity + marched sun shadows,
    /// whose early-exit bound scales in lockstep —
    /// `scene::set_detail_ao_strength` / the DETAIL_AO_STR define).
    pub detail_ao_strength: f32,
    /// `--detail-untex-scale K` (0.0..=4.0, default 1.0): multiplier on the
    /// synthetic texel-equivalent scale UNTEXTURED materials get
    /// (`scene::DETAIL_UNTEX_K` × content diag) so albedo-map-free scenes
    /// (powerplant) carry the detail field too. 0 = untextured detail off,
    /// the bitwise pre-untextured-arm A/B (`scene::set_detail_untex_scale`;
    /// read at scale DERIVATION, so restart tier — no GPU define, the scale
    /// rides the per-material lane).
    pub detail_untex_scale: f32,
    /// `--no-amb-bump` clears (`scene::set_amb_bump`): the sampled/SH ambient
    /// amplifies its irradiance response to the shading normal's deviation
    /// from the geometric normal (normal maps + detail bump + ripple read
    /// under sky light). Runtime shading lever — the depth-tint class; a
    /// no-op on flat-shaded geometry (n_s == n_g).
    pub amb_bump: bool,
    /// `--no-rtgi` clears (`shade::set_rtgi`): real-time GI OFF — the ambient
    /// tier goes back to flat `SH sky irradiance × AO`, bit-identical to the
    /// pre-RTGI renderer (the GPU compiles the bounce block out). DEFAULT ON:
    /// one cosine-sampled bounce ray per pixel per frame replaces the ambient
    /// term, shaded at the hemi BOUNCE_Q policy, integrated by the temporal
    /// denoisers / accumulation. Still-frame hemi tiers (H) take precedence.
    pub rtgi: bool,
    /// `--auto-exposure` ARMS (`autoexp::set_enabled`): a display-stage
    /// controller eases a clamped EV toward what the presented frame's mean
    /// log2-luminance asks for (src/autoexp.rs) — enclosures open up,
    /// exteriors hold at ~0 EV. DEFAULT OFF (2026-08-08, the user's call —
    /// the same-day flip of the one-day default-ON: with RTGI on by default
    /// enclosures light themselves, so the aperture holds at exactly 1.0
    /// plus any bias); `--no-auto-exposure` spells the default. The default
    /// is DUPLICATED in autoexp.rs's ENABLED initializer and settings.rs's
    /// menu-row `Toggle { default }` — flip all three in lockstep.
    /// Headless paths never run the controller either way, so every
    /// gate/benchmark sees exposure 1.0 regardless.
    pub autoexp: bool,
    /// `--exposure-bias EV` (stops, -8..=8, default 0 — the cinematic
    /// `-exposure` range): a manual aperture offset composed ON TOP of the
    /// controller's EV, and live even under `--no-auto-exposure` (the manual
    /// exposure lever). Interactive-only by construction — the bias reaches
    /// the screen through the session controller's `set_exposure`, which
    /// headless paths never tick.
    pub exposure_bias: f32,
    /// `--no-water` clears (`scene::set_water`). Keys the cache lever word.
    pub water: bool,
    /// `--no-coincident-cull` clears (`scene::set_coincident_cull`): keep
    /// transmissive faces exactly coincident with opaque faces (the
    /// pre-cull z-fight, where CPU and GPU break the tie differently —
    /// rungholt's water bottoms over the seabed). Keys the cache lever word.
    pub coincident_cull: bool,
    /// `--heightfield` arms relief rendering and starts it ON; the default is
    /// UNARMED. ONE field, TWO statics: `main` stores `set_height_armed` AND
    /// `set_height_on` together, which is what keeps `--no-heightfield
    /// --heightfield` a true arm under later-flags-win. Keys the scene cache.
    pub heightfield: bool,
    /// `--no-bloom` clears (`bloom::set_enabled`) — a display-stage lever.
    pub bloom: bool,
    /// `--no-clouds` clears (`clouds::set_enabled`).
    pub clouds: bool,
    /// `--cloud-shadow N` in 2..=64; 0 = `--no-cloud-shadow`
    /// (`gpu::trace::set_cloud_shadow`).
    pub cloud_shadow: u32,
    /// `--sky-lod K`, a power of two in 2..=32; 1 = `--no-sky-lod`
    /// (`gpu::trace::set_sky_lod`).
    pub sky_lod: u32,
    /// `--no-fireflies` clears (`fireflies::set_enabled`).
    pub fireflies: bool,
    /// `--fireflies N`. `fireflies::set_count` is what CLAMPS to MAX_FIREFLIES;
    /// the parse only NOTES the clamp, so this field carries the raw request.
    pub fireflies_count: u32,
    /// `--emissive-lights [N]` ARMS the direct-tier NEE for emissive
    /// surfaces (src/emissive.rs) — DEFAULT OFF, the heightfield arming
    /// shape (the user's call, third round 2026-08-06: with EL_BOOST the
    /// look earns it, but only the bistro island carries emissive maps and
    /// the CPU cost is per-session); `--no-emissive-lights` spells the
    /// default (later flags win). The default is DUPLICATED in emissive.rs's
    /// ENABLED initializer — flip in lockstep. NOTE the compiled default is
    /// only half the story: `main::upscaler_defaults` arms it in sessions
    /// whose WIRED upscaler is TAA-class (XeSS/FSR3) unless
    /// `emissive_lights_explicit` vetoes — see that field. (That auto-arm
    /// was retired for part of 2026-08-08 on the premise that default-ON
    /// NRD/ReBLUR pre-upscale denoising integrates the RTGI bounce's
    /// stochastic emissive ahead of the TAA clamp; the user's same-day
    /// feel-test found NRD NOT sufficient — the pools still vanish — so the
    /// policy is RE-INSTATED.)
    pub emissive_lights: bool,
    /// Did the user pick an emissive-lights state at all (flag or settings
    /// file)? OFF is a real default, so the value cannot report whether it
    /// was chosen — and the upscaler-class default (`main::
    /// upscaler_defaults`: XeSS/FSR3 sessions arm NEE because a TAA-class
    /// neighborhood clamp rejects the RTGI bounce's sparse stochastic
    /// emissive, and NRD's pre-upscale integration measured insufficient)
    /// may only move a default the user left alone. BOTH spellings
    /// set it — presence, not value, is the signal (the
    /// `dxr_inline_explicit` doctrine), which makes `--no-emissive-lights`
    /// the spelled opt-out in XeSS/FSR3 sessions. The settings file sets it
    /// too — the `dxr_inline` precedent, NOT the fg one: the menu writes
    /// `effects.emissive_lights`, and a saved preference must veto the
    /// policy.
    pub emissive_lights_explicit: bool,
    /// The `--emissive-lights` budget (bare flag keeps the default).
    /// `emissive::set_budget` owns the clamp to MAX_EMISSIVE_LIGHTS; the
    /// parse only NOTES it (the fireflies shape).
    pub emissive_lights_count: u32,
    /// `--el-cluster grid|som` — the emitter-placement A/B lever (the
    /// `--bvh-builder` bake-off shape). Validated in main's lever block
    /// (illegal value exits 2 there — the parse stays pure); grid is the
    /// shipped clusterer, bit-identical.
    pub el_cluster: String,
    /// `--dxr-inline 0|1|2` (`gpu::dxr::set_inline_mode`).
    pub dxr_inline: u32,
    /// Did the user pick a `--dxr-inline` mode at all (flag or settings
    /// file)? 1 is a real default, so the value cannot report whether it was
    /// chosen — and the Intel vendor default (mode 2, `main::vendor_defaults`)
    /// may only move a default the user left alone. Any LEGAL `--dxr-inline
    /// N` sets it, including 1: the flag's presence, not its value, is the
    /// signal (the `spin_frames_explicit` doctrine), which makes
    /// `--dxr-inline 1` the spelled opt-out on Intel. The settings file sets
    /// it too — the `renderer.mode`/`lock_res` precedent, NOT the fg one: the
    /// menu writes `advanced.dxr_inline`, and a saved preference must veto
    /// the policy.
    pub dxr_inline_explicit: bool,
    /// `--waveviz [chs]` (`gpu::trace::set_waveviz`) — the wave-footprint
    /// overlay instrument: 0 off, 1 on, 2 = the mode-1 closest-hit variant.
    /// The optional `chs` token is consumed only when it matches exactly
    /// (the `--blas-split` optional-value idiom — a following scene path is
    /// safe). `FR_WAVEVIZ` stays as the env alias; the CLI wins (main's
    /// lever block owns the precedence).
    pub waveviz: u8,
    /// `--dxr-sbt 0|1|2|3` (`gpu::dxr::set_sbt_mode`) — the many-record,
    /// material-sorted SBT ladder (the Intel-brief Q4 counterfactual). A dev
    /// MEASUREMENT lever, the `--sw-rays` class: default 0 = off, no vendor
    /// policy, no settings exposure; ladder rungs not yet built degrade
    /// loudly at DxrGpu construction. See gpu/dxr.rs's SBT_MODE doc.
    pub dxr_sbt: u32,
    /// `--no-foliage-sway` clears (`foliage::set_armed`) — leaf sway, the
    /// prototype of the tetrahedral-cage epic (docs/design/animated-foliage.md):
    /// leaf triangles (foliage-classified + alpha-masked) bucket into per-cell
    /// chunks that TRANSLATE on the cloud clock — in ALL THREE render modes
    /// since v0.2 (CPU rays displace at the intersector, wavefront + DXR bind
    /// the animated TLAS ring; soundness from build-time swept leaf AABBs,
    /// `bvh::grow_sway_sweep`). DEFAULT ON since 2026-07-28 (`--foliage-sway`
    /// spells the default; safe because a scene with no leaf-classified
    /// materials is STRUCTURALLY untouched — no partition attaches, the tree
    /// and plan stay bit-identical — and every headless `--check*`/`--spin`
    /// path stays at the rest pose, so benchmarks and gates never have
    /// geometry move under them). Off is bit-identical by construction.
    /// Needs --blas-split (the default — the attach predicate,
    /// `foliage::sweep_armed`).
    pub foliage_sway: bool,
    /// `--foliage-amp <x>` (`foliage::set_amp_mult`) — taste multiplier on
    /// the sway amplitude (both curl and flutter halves), default 1.0;
    /// finite, 0.0..=8.0 (0 = armed machinery, zero motion — the null A/B).
    pub foliage_amp: f32,
}


/// Everything the parse loop produces besides `Opts` — the mode selectors and
/// scene sources that were `main()` locals before the extraction.
pub struct Cli {
    pub opts: Opts,
    /// The positional argument: an `.obj` / `.gltf` / `.glb` path.
    pub obj: Option<String>,
    pub check: bool,
    pub check_dlss: bool,
    pub dlss_dump: bool,
    pub check_oidn: bool,
    pub oidn_dump: bool,
    pub check_xess: bool,
    pub xess_dump: bool,
    pub check_fsr: bool,
    pub check_nppd: bool,
    pub nppd_dump: bool,
    pub check_nrd: bool,
    pub check_gpu: bool,
    pub check_dxr: bool,
    /// `--no-xess` was passed (distinct from the chain simply never reaching
    /// that level).
    pub no_xess_explicit: bool,
    /// `--fsr`/`--fsr3`/`--fsr4` passed: flips the default adapter preference
    /// to AMD. A settings file can force the same level, so `main` ORs
    /// `settings::AppliedFx::fsr_forced` into this after the parse.
    pub fsr_forced: bool,
    /// `--dxr` passed explicitly (it is the default): gates the --gpu notice.
    pub dxr_explicit: bool,
    /// `--no-upscale` passed: distinguishes the quiet explicit plain path from
    /// the loud chain-exhausted fallback.
    pub no_upscale: bool,
    pub stress: Option<usize>,
    pub tile: Option<(u32, u32)>,
    pub cam_override: Option<Camera>,
    pub spin: Option<String>,
    pub spin_frames: u32,
    /// Whether `--spin-frames` was actually typed. A DEFAULTED count is
    /// extended to cover a whole camera lap past the resolved warm-up (which
    /// is vendor-derived, so the runner cannot know it here); an explicit one
    /// is obeyed verbatim, because a benchmark's frame count must never move
    /// under the person who set it.
    pub spin_frames_explicit: bool,
    /// Which software/wavefront tracing arm `--spin` executes. `true` is the
    /// quadtree hybrid; `false` is the per-pixel root-traversal reference.
    /// Later `--spin-hybrid` / `--spin-plain` flags win.
    pub spin_hybrid: bool,
    /// Explicit number of leading `--spin` frames to exclude. `None` lets the
    /// runner choose a device-appropriate default (longer on Intel because its
    /// optimized shader can replace the initial variant asynchronously).
    pub spin_warmup: Option<u32>,
    /// `--cinematic`: `None` = not asked for, `Some(sel)` = a preset name or a
    /// shot-list path.
    pub cinematic: Option<String>,
    pub cine: cinematic::CineOpts,
    /// World mode: `None` = the default (flagless interactive boots the world),
    /// `Some(true)` = explicit `--world` (exclusivity ERRORS rather than
    /// silently resolving), `Some(false)` = `--no-world`. Later flags win.
    pub world_flag: Option<bool>,
    /// `--help`/`-h` was given: `main` prints `usage()` and returns. The parser
    /// deliberately does not print it itself — that keeps `parse_from` free of
    /// output, which is what lets a gate parse an argv silently.
    pub helped: bool,
    /// Diagnostics the parser wants said, in order. Collected rather than
    /// printed so `parse_from` stays quiet inside a gate; `main` prints them
    /// verbatim.
    pub notes: Vec<String>,
}

/// The code defaults, before any settings file or flag is applied — the `Opts`
/// literal `main()` used to open with.
pub fn defaults() -> Opts {
    Opts {

        chain: upchain::UpChain::ALL,
        gpu_debug: false,
        bc7: bc7::Bc7Mode::Gpu(bc7::Quality::Fast),
        audio: true,
        oidn: false,
        oidn_path: std::env::var("FRUSTRACER_OIDN_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\oidn.x64.windows\bin").to_string()
        }),
        oidn_device: 0,
        oidn_quality: oidn::QUALITY_BALANCED,
        oidn_clean_aux: true,
        oidn_temporal: true,
        nppd: false,
        nppd_path: std::env::var("FRUSTRACER_ORT_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\onnxruntime\bin").to_string()
        }),
        nppd_model: std::env::var("FRUSTRACER_NPPD_MODEL").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\nppd\nppd_small.onnx").to_string()
        }),
        nppd_device: None,
        nrd: true,
        nrd_explicit: false,
        nrd_path: std::env::var("FRUSTRACER_NRD_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\NRD\bin").to_string()
        }),
        nrd_perf: false,
        nrd_tune: Default::default(),
        frd: false,
        frd_explicit: false,
        frd_tune: Default::default(),
        xess_path: std::env::var("FRUSTRACER_XESS_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\XeSS-SDK\bin").to_string()
        }),
        ffx_path: std::env::var("FRUSTRACER_FFX_PATH").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                r"\SDKs\FidelityFX-Samples-prebuilt\Samples\Denoisers\FidelityFX_Denoiser\dx12\x64\Release"
            )
            .to_string()
        }),
        fg: true,
        fg_explicit: false,
        fg_path: std::env::var("FRUSTRACER_FG_PATH").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                r"\SDKs\FidelityFX-Samples-prebuilt\Samples\Upscalers\FidelityFX_FSR\dx12\x64\Release"
            )
            .to_string()
        }),
        fsr4_required: false,
        fsr_tune: fsr::DenoiseTuning::default(),
        quin: false,
        quin_anchor: None,
        prefer: None,
        oidn_post: false,
        xess_autoexposure: false,
        adaptive: true,
        lock_scale: Some(xess::DEFAULT_LOCK_SCALE),
        spp: 1,
        temporal: true,
        replay: true,
        adopt: true,
        discard_seeds: false,
        defer_shade: false,
        hemi_share: true,
        cut_rays: true,
        sw_rays: false,
        cut_hemi: false,
        // M2 defaults, measured (spin path, 250 frames, vs the historical
        // axes=1 / c_trav=0 build):
        //   San Miguel 22.39 -> 18.45 ms (-17.6%), 328 -> 133 MB (-59%)
        //   stress 5000 34.07 -> 29.0  ms (-15%),  220 ->  99 MB (-55%)
        //   default     flat (its 4 MB tree fits in cache either way)
        // 3-axis carries the SPEED (it is a -33% ray-node win); c_trav carries
        // the MEMORY (speed-neutral). Set --bvh-axes 1 --bvh-ctrav 0 to recover
        // the historical build.
        c_trav: 3.0,
        max_leaf: 8,
        split_axes: 3,
        bvh_builder: "sah".to_string(),
        blas_split: Some(blas_split::DEFAULT_MAX_PRIMS),
        dual_gpu: None,
        dual_gpu_auto: false,
        dual_gpu_depth: 3,
        dual_gpu_arm: None,
        ftree: true,
        ftree_tiles: false,
        wide_levels: true,
        gpu: false,
        dxr: true,
        mode_explicit: false,
        lock_res_explicit: false,
        dxc_path: std::env::var("FRUSTRACER_DXC_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\dxc\bin\x64").to_string()
        }),
        pix_markers: false,
        pix_path: std::env::var("FRUSTRACER_PIX_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\pix\bin\x64").to_string()
        }),
        gpu_timing: false,
        vsync: true,
        // The 10-bit swapchain is the default: see Opts::hdr (the probe picks
        // PQ vs gamma). The 8-bit path survives as --no-hdr and as the FG
        // wrap-failure fallback.
        hdr: true,
        hdr10: false,
        sdr10: false,
        hdr_paper_white: tone::DEFAULT_PAPER_WHITE,
        hdr_peak: None,
        tod: None,
        mips: true,
        aniso: texture::MAX_ANISO_CAP,
        h2n: true,
        n2h: true,
        slope_mips: true,
        spec_aa: true,
        normal_strength: 1.0,
        tinted_shadows: true,
        spray: true,
        depth_tint: true,
        detail_tex: true,
        detail_ao: true,
        detail_strength: 0.5,
        detail_ao_strength: 0.125,
        detail_untex_scale: 1.0,
        amb_bump: true,
        rtgi: true,
        autoexp: false,
        exposure_bias: 0.0,
        water: true,
        coincident_cull: true,
        heightfield: false,
        bloom: true,
        clouds: true,
        cloud_shadow: 16,
        sky_lod: 4,
        fireflies: true,
        fireflies_count: fireflies::DEFAULT_COUNT,
        emissive_lights: false,
        emissive_lights_explicit: false,
        emissive_lights_count: emissive::EL_DEFAULT,
        el_cluster: "grid".to_string(),
        dxr_inline: 1,
        dxr_inline_explicit: false,
        waveviz: 0,
        dxr_sbt: 0,
        foliage_sway: true,
        foliage_amp: 1.0,
    }
}

/// Parse `args` over `base`, which is the precedence seam: `main` hands in
/// `defaults()` already overwritten by the settings file, so a flag arm below
/// simply stores over the file's value and "CLI wins" needs no extra
/// machinery.
///
/// Pure — see the module header. Hard errors `exit(2)` in place; everything
/// else lands in the returned `Cli`.
pub fn parse_from(base: Opts, args: impl Iterator<Item = String>) -> Cli {
    let mut opts = base;
    let mut obj: Option<String> = None;
    let mut check = false;
    let mut check_dlss = false;
    let mut dlss_dump = false;
    let mut check_oidn = false;
    let mut oidn_dump = false;
    let mut check_xess = false;
    let mut xess_dump = false;
    let mut check_fsr = false;
    let mut check_nppd = false;
    let mut nppd_dump = false;
    let mut check_nrd = false;
    let mut no_xess_explicit = false;
    let mut fsr_forced = false;
    let mut dxr_explicit = false;
    let mut no_upscale = false;
    let mut check_gpu = false;
    let mut check_dxr = false;
    let mut stress: Option<usize> = None;
    let mut tile: Option<(u32, u32)> = None;
    let mut cam_override: Option<Camera> = None;
    let mut spin: Option<String> = None;
    let mut spin_frames = 2000u32;
    let mut spin_frames_explicit = false;
    let mut spin_hybrid = true;
    let mut spin_warmup: Option<u32> = None;
    let mut cinematic: Option<String> = None;
    let mut cine = cinematic::CineOpts::default();
    let mut world_flag: Option<bool> = None;
    let mut helped = false;
    let mut notes: Vec<String> = Vec::new();
    // Peekable so a flag can take an OPTIONAL value (--blas-split [N],
    // --cinematic [preset]) without swallowing the next flag.
    let mut args = args.peekable();
    while let Some(a) = args.next() {
        match a.as_str() {

            "--check" => check = true,
            "--check-dlss" => check_dlss = true,
            "--dlss-dump" => {
                check_dlss = true;
                dlss_dump = true;
            }
            "--dlss" => opts.chain.force(upchain::UpLevel::Dlss),
            "--no-dlss" => opts.chain.skip(upchain::UpLevel::Dlss),
            "--no-upscale" => {
                opts.chain = upchain::UpChain::NONE;
                no_upscale = true;
            }
            "--check-oidn" => check_oidn = true,
            "--oidn-dump" => {
                check_oidn = true;
                oidn_dump = true;
            }
            "--oidn" => opts.oidn = true,
            "--no-oidn" => opts.oidn = false,
            "--no-audio" => opts.audio = false,
            "--audio" => opts.audio = true,
            "--oidn-no-temporal" => opts.oidn_temporal = false,
            "--check-nppd" => check_nppd = true,
            "--nppd-dump" => {
                check_nppd = true;
                nppd_dump = true;
            }
            "--check-nrd" => check_nrd = true,
            "--nrd" => {
                opts.nrd = true;
                opts.nrd_explicit = true;
            }
            "--no-nrd" => opts.nrd = false,
            "--nrd-path" => {
                opts.nrd_path = args.next().unwrap_or_else(|| {
                    eprintln!("--nrd-path needs a directory argument");
                    std::process::exit(2);
                })
            }
            "--nrd-perf" => opts.nrd_perf = true,
            "--no-nrd-perf" => opts.nrd_perf = false,
            // ReBLUR runtime tuning overrides (the fsr-tune shape). Absent =
            // the settings the session always sent, so a flagless session is
            // unchanged; each is an A/B lever on the ReBLUR cost/quality
            // trade (max-stabilized-frames 0 drops the TemporalStabilization
            // pass outright; prepass-radius 0 disables both prepasses).
            "--nrd-max-stabilized-frames" | "--nrd-max-accum-frames" => {
                let v: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("{a} needs a non-negative integer argument");
                    std::process::exit(2);
                });
                if a == "--nrd-max-stabilized-frames" {
                    opts.nrd_tune.max_stabilized_frames = Some(v);
                } else {
                    opts.nrd_tune.max_accum_frames = Some(v);
                }
            }
            "--nrd-prepass-radius" => {
                opts.nrd_tune.prepass_radius = args
                    .next()
                    .and_then(|s| s.parse::<f32>().ok())
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .map(Some)
                    .unwrap_or_else(|| {
                        eprintln!(
                            "--nrd-prepass-radius needs a non-negative float (px; 0 disables \
                             the prepasses)"
                        );
                        std::process::exit(2);
                    });
            }
            "--nrd-no-anti-firefly" => opts.nrd_tune.anti_firefly = Some(false),
            "--nrd-anti-firefly" => opts.nrd_tune.anti_firefly = Some(true),
            "--frd" => {
                opts.frd = true;
                opts.frd_explicit = true;
            }
            "--no-frd" => opts.frd = false,
            // FRD runtime tuning (the --nrd-* shape: absent = the compiled
            // frd.rs constants, so a flagless session is unchanged).
            "--frd-max-accum-frames" | "--frd-fast-frames" | "--frd-max-stab-frames" => {
                let v: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("{a} needs a non-negative integer argument");
                    std::process::exit(2);
                });
                match a.as_str() {
                    "--frd-max-accum-frames" => {
                        // The meta plane stores n/63 (frd::META_N_MAX), so a
                        // larger cap truncates at the wire regardless of the
                        // CB — note it (the --fireflies clamp shape; the
                        // field carries the raw request, frd_gpu's cb()
                        // clamps).
                        if v as f32 > frd::META_N_MAX {
                            notes.push(format!(
                                "frd: max-accum-frames {v} clamped to {} (the meta plane's \
                                 n/63 wire cap)",
                                frd::META_N_MAX
                            ));
                        }
                        opts.frd_tune.max_accum_frames = Some(v);
                    }
                    "--frd-fast-frames" => opts.frd_tune.fast_frames = Some(v),
                    _ => opts.frd_tune.max_stab_frames = Some(v),
                }
            }
            "--frd-blur-radius" | "--frd-clamp-sigma" => {
                let v = args
                    .next()
                    .and_then(|s| s.parse::<f32>().ok())
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .unwrap_or_else(|| {
                        eprintln!("{a} needs a non-negative float argument");
                        std::process::exit(2);
                    });
                if a == "--frd-blur-radius" {
                    opts.frd_tune.blur_radius = Some(v);
                } else {
                    opts.frd_tune.clamp_sigma = Some(v);
                }
            }
            "--frd-no-anti-firefly" => opts.frd_tune.anti_firefly = Some(false),
            "--frd-anti-firefly" => opts.frd_tune.anti_firefly = Some(true),
            "--frd-no-fp16" => opts.frd_tune.no_fp16 = true,
            "--nppd" => opts.nppd = true,
            "--no-nppd" => opts.nppd = false,
            "--nppd-path" => {
                opts.nppd_path = args.next().unwrap_or_else(|| {
                    eprintln!("--nppd-path needs a directory argument");
                    std::process::exit(2);
                })
            }
            "--nppd-model" => {
                opts.nppd_model = args.next().unwrap_or_else(|| {
                    eprintln!("--nppd-model needs an .onnx path argument");
                    std::process::exit(2);
                })
            }
            "--nppd-device" => {
                opts.nppd_device = match args.next().as_deref() {
                    Some("auto") => None,
                    Some("cpu") => Some(-1),
                    Some(s) if s == "dml" => Some(0),
                    Some(s) if s.starts_with("dml:") => match s[4..].parse::<i32>() {
                        Ok(n) if n >= 0 => Some(n),
                        _ => {
                            eprintln!("--nppd-device dml:<n> needs a non-negative adapter index");
                            std::process::exit(2);
                        }
                    },
                    _ => {
                        eprintln!("--nppd-device needs one of: auto cpu dml dml:<n>");
                        std::process::exit(2);
                    }
                }
            }
            "--check-xess" => check_xess = true,
            "--xess-dump" => {
                check_xess = true;
                xess_dump = true;
            }
            "--xess" => opts.chain.force(upchain::UpLevel::Xess),
            "--no-xess" => {
                opts.chain.skip(upchain::UpLevel::Xess);
                no_xess_explicit = true;
            }
            "--check-fsr" => check_fsr = true,
            "--fsr" => {
                opts.chain.force(upchain::UpLevel::Fsr4);
                fsr_forced = true;
            }
            // --fsr4 IS --fsr, minus the fall-through: the level becomes a
            // REQUIREMENT. Everywhere else in this codebase an unsupported
            // feature is a loud line + a working fallback; here the user
            // asked to be told, so a failed FSR4 probe exits(2) with the two
            // things worth trying (--fsr3, --prefer-amd). Enforced in
            // run_window against the session's actual wiring, so it can never
            // disagree with what got wired.
            "--fsr4" => {
                opts.chain.force(upchain::UpLevel::Fsr4);
                fsr_forced = true;
                opts.fsr4_required = true;
            }
            "--fsr3" => {
                opts.chain.force(upchain::UpLevel::Fsr3);
                fsr_forced = true;
            }
            "--no-fsr" => {
                opts.chain.skip(upchain::UpLevel::Fsr4);
                opts.chain.skip(upchain::UpLevel::Fsr3);
            }
            // --quinlight suspends the chain's first-hit-wins rule: every level
            // it leaves enabled is WIRED, and the fuse consumes them all. The
            // chain flags still compose (--no-xess &c drop an engine), so the
            // engine set stays the user's to shape.
            "--quinlight" => opts.quin = true,
            "--quin-anchor" => {
                opts.quin_anchor = args.next().and_then(|s| s.parse::<u32>().ok()).or_else(|| {
                    eprintln!("--quin-anchor needs a non-negative engine index");
                    std::process::exit(2);
                })
            }
            "--ffx-path" => {
                opts.ffx_path = args.next().unwrap_or_else(|| {
                    eprintln!("--ffx-path needs a directory argument");
                    std::process::exit(2);
                })
            }
            "--fg" => {
                opts.fg = true;
                opts.fg_explicit = true;
            }
            "--no-fg" => {
                opts.fg = false;
                opts.fg_explicit = false;
            }
            "--fg-path" => {
                opts.fg_path = args.next().unwrap_or_else(|| {
                    eprintln!("--fg-path needs a directory argument");
                    std::process::exit(2);
                })
            }
            "--oidn-post" => opts.oidn_post = true,
            "--xess-autoexposure" => opts.xess_autoexposure = true,
            "--no-adaptive" => opts.adaptive = false,
            "--no-temporal" => opts.temporal = false,
            "--no-replay" => opts.replay = false,
            "--no-adopt" => opts.adopt = false,
            "--discard-seeds" => opts.discard_seeds = true,
            "--defer-shade" => opts.defer_shade = true,
            // Every arm from here down stores a FIELD; `main` applies the
            // whole set to the process globals once, before the scene loads
            // (see the module header). --no-mips: no chains are built, so
            // every trilinear sample falls back to mip-0 bilinear — the
            // pre-mip renderer exactly.
            "--no-mips" => opts.mips = false,
            // Load-time converter kill levers, same "knob before scene load"
            // pattern: --no-h2n restores the pre-conversion behavior exactly
            // (grayscale bump maps are dropped, normal_tex stays NO_TEX);
            // --no-n2h leaves real normal maps' alpha at 255 and height_amp
            // at 0.0 — relief has no field to march, structurally off.
            "--no-h2n" => opts.h2n = false,
            "--no-n2h" => opts.n2h = false,
            // Slope-space normal-map mips A/B lever (derived-only — never
            // keys the cache) and the session normal-strength multiplier
            // (per-load data, the --tod class: no process global, so no
            // lever-block line — load_scene reads it off SceneRequest).
            "--no-slope-mips" => opts.slope_mips = false,
            // Spec-AA A/B lever (derived-only — never keys the cache): the
            // slope-variance → roughness fold that keeps detail maps in the
            // rendering equation at every distance.
            "--no-spec-aa" => opts.spec_aa = false,
            "--normal-strength" => {
                let k: f32 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&k: &f32| k.is_finite() && (0.0..=8.0).contains(&k))
                    .unwrap_or_else(|| {
                        eprintln!("--normal-strength needs a value in 0.0..=8.0 (1 = default, 0 = normals off)");
                        std::process::exit(2);
                    });
                opts.normal_strength = k;
            }
            // Tinted-shadows kill lever, same "knob before scene load"
            // pattern: finalize_scalars never arms any_transmissive, so
            // every light-occlusion query binary-blocks on glass — the
            // pre-feature renderer bit-identically (occlusion rays through
            // transmissive surfaces otherwise carry a transmission×albedo
            // tint per interface).
            "--no-tinted-shadows" => opts.tinted_shadows = false,
            // Water-look levers: --no-spray keeps tiny transmissive islands
            // (fountain droplets) as clear glass instead of retagging them
            // white-scatter at load (keys the cache lever word);
            // --no-depth-tint drops the Beer–Lambert attenuation over the
            // transmission chain's interior segments (runtime shading lever).
            "--no-spray" => opts.spray = false,
            "--no-depth-tint" => opts.depth_tint = false,
            // --no-detail-tex: no Unreal-1 close-up detail field (albedo
            // grain + micro-bump on magnified textured hits) — runtime
            // shading lever, the depth-tint class.
            "--no-detail-tex" => opts.detail_tex = false,
            // --no-detail-ao: the detail field's pits stop darkening
            // ambient/specular (the cavity term) — runtime shading lever,
            // the depth-tint class.
            "--no-detail-ao" => opts.detail_ao = false,
            // Detail strength multipliers — the --normal-strength arm's
            // shape, but process-global levers (main's lever block stores
            // them; the GPU twins are injected #defines at kernel compile).
            "--detail-strength" => {
                let k: f32 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&k: &f32| k.is_finite() && (0.0..=4.0).contains(&k))
                    .unwrap_or_else(|| {
                        eprintln!("--detail-strength needs a value in 0.0..=4.0 (1 = default, 0 = grain off)");
                        std::process::exit(2);
                    });
                opts.detail_strength = k;
            }
            "--detail-ao-strength" => {
                let k: f32 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&k: &f32| k.is_finite() && (0.0..=4.0).contains(&k))
                    .unwrap_or_else(|| {
                        eprintln!("--detail-ao-strength needs a value in 0.0..=4.0 (1 = default, 0 = pools/shadows off)");
                        std::process::exit(2);
                    });
                opts.detail_ao_strength = k;
            }
            // --detail-untex-scale: the untextured materials' synthetic
            // detail texel scale, as a multiplier on DETAIL_UNTEX_K ×
            // content diag (0 = untextured detail off, the bitwise A/B —
            // read at derivation time, not per frame).
            "--detail-untex-scale" => {
                let k: f32 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&k: &f32| k.is_finite() && (0.0..=4.0).contains(&k))
                    .unwrap_or_else(|| {
                        eprintln!("--detail-untex-scale needs a value in 0.0..=4.0 (1 = default, 0 = untextured detail off)");
                        std::process::exit(2);
                    });
                opts.detail_untex_scale = k;
            }
            // --no-amb-bump: the SH ambient stops amplifying its response to
            // the shading normal's deviation (normal maps/detail bump go
            // back to the plain order-2 irradiance) — runtime shading
            // lever, the depth-tint class.
            "--no-amb-bump" => opts.amb_bump = false,
            // --no-rtgi: real-time GI off — the ambient tier reverts to flat
            // SH×AO (bit-identical pre-RTGI arm; the GPU kernels compile the
            // bounce block out). --rtgi spells the default (later flags win).
            "--no-rtgi" => opts.rtgi = false,
            "--rtgi" => opts.rtgi = true,
            // --auto-exposure ARMS the display-stage aperture controller
            // (DEFAULT OFF — RTGI lights enclosures for real, so the
            // aperture holds at a fixed 1.0, plus any --exposure-bias).
            // --no-auto-exposure spells the default (later flags win).
            // Display-stage — no gate contact.
            "--no-auto-exposure" => opts.autoexp = false,
            "--auto-exposure" => opts.autoexp = true,
            "--exposure-bias" => {
                let ev: f32 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&ev: &f32| ev.is_finite() && (-8.0..=8.0).contains(&ev))
                    .unwrap_or_else(|| {
                        eprintln!(
                            "--exposure-bias needs a value in stops, -8..=8 (e.g. --exposure-bias 1.5)"
                        );
                        std::process::exit(2);
                    });
                opts.exposure_bias = ev;
            }
            // --no-water: the fountain classifies as generic glassware (the
            // pre-water-class look) instead of the water refinement (blue-green
            // tint, IOR 1.33, ripple normals); keys the cache lever word.
            "--no-water" => opts.water = false,
            // --no-coincident-cull: keep transmissive faces coincident with
            // opaque faces (the pre-cull z-fight); keys the cache lever word.
            "--no-coincident-cull" => opts.coincident_cull = false,
            // Relief rendering session levers. The DEFAULT is DISARMED —
            // structurally the pre-relief renderer (no AABB sweep at BVH
            // build, no march anywhere; the sweep's all-axis edge pad
            // measurably wrecks BVH quality on all-tris-carry-height scenes
            // even with relief off — see bvh.rs's HEIGHT_ARMED header).
            // --heightfield ARMS the session and starts relief ON (V then
            // toggles relief live within the armed session); --no-heightfield
            // is the explicit spelling of the default, kept as the
            // later-flags-win override. Both set BOTH statics so
            // `--no-heightfield --heightfield` is a true arm — headless
            // --check* paths read height_on() directly, with no session()
            // to re-seed it. Armed state keys the scene cache.
            "--heightfield" => opts.heightfield = true,
            "--no-heightfield" => opts.heightfield = false,
            // A/B lever: no glare. A display-stage pass, so this is a pure
            // presentation change — accum, the temporal cache, every upscaler
            // guide and every radiance gate are untouched either way, and the
            // off path keeps the original alloc-free tonemap loop verbatim.
            "--no-bloom" => opts.bloom = false,
            // A/B kill lever for the volumetric cloud layer (default ON).
            // Same "session constant before scene load" pattern: off takes
            // guarded early returns everywhere — bit-identical to the
            // pre-cloud renderer (clouds::self_test pins it).
            "--no-clouds" => opts.clouds = false,
            // The slab-space cloud-shadow cache (default ON at 16 cells/λ) and
            // the amortized sky-march lattice (default ON at K=4). Both are GPU
            // shading-cache levers; off is bit-identical (guarded arms). Same
            // "knob before scene load" pattern — TraceGpu/DxrGpu snapshot them
            // in new(). 0 / 1 = the respective off spellings.
            "--no-cloud-shadow" => opts.cloud_shadow = 0,
            "--cloud-shadow" => {
                let n: u32 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&n| n >= 2 && n <= 64)
                    .unwrap_or_else(|| {
                        eprintln!("--cloud-shadow needs cells/wavelength in 2..=64 (--no-cloud-shadow = off)");
                        std::process::exit(2);
                    });
                opts.cloud_shadow = n;
            }
            "--no-sky-lod" => opts.sky_lod = 1,
            "--sky-lod" => {
                let k: u32 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&k: &u32| k.is_power_of_two() && (1..=32).contains(&k))
                    .unwrap_or_else(|| {
                        eprintln!("--sky-lod needs a power of two in 2..=32 (1 or --no-sky-lod = off)");
                        std::process::exit(2);
                    });
                opts.sky_lod = k;
            }
            // Firefly point lights (default ON, but they only exist after
            // dusk — a day session snapshots count = 0 and is bit-identical
            // structurally). Same "session constant before scene load"
            // pattern; the count clamps to the CB row cap loudly.
            "--no-fireflies" => opts.fireflies = false,
            "--fireflies" => {
                let n: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!(
                        "--fireflies needs an integer count (1..={})",
                        fireflies::MAX_FIREFLIES
                    );
                    std::process::exit(2);
                });
                if n > fireflies::MAX_FIREFLIES as u32 {
                    notes.push(format!(
                        "fireflies: count {n} clamped to {} (the CB row cap)",
                        fireflies::MAX_FIREFLIES
                    ));
                }
                opts.fireflies_count = n;
            }
            // Emissive cluster lights — DEFAULT OFF (the heightfield arming
            // shape; the user's call, third round 2026-08-06: EL_BOOST fixed
            // the faint pools, but the CPU shadow-ray cost is real and only
            // the bistro island carries emissive maps — see emissive.rs's
            // header).
            // Optional value, the --blas-split idiom: the next token is
            // consumed only when it is all digits, so `--emissive-lights
            // model.obj` leaves the scene path alone — but a numeric token
            // that is not a legal budget (0) is a typo and exits rather than
            // arming at the default and landing in the positional arm.
            "--no-emissive-lights" => {
                opts.emissive_lights = false;
                opts.emissive_lights_explicit = true;
            }
            "--emissive-lights" => {
                opts.emissive_lights = true;
                opts.emissive_lights_explicit = true;
                let numeric = args.peek().is_some_and(|v| {
                    !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit())
                });
                if numeric {
                    let v = args.next().unwrap();
                    let n = v.parse::<u32>().ok().filter(|n| *n >= 1).unwrap_or_else(|| {
                        eprintln!(
                            "--emissive-lights: '{v}' is not a budget (1..={})",
                            emissive::MAX_EMISSIVE_LIGHTS
                        );
                        std::process::exit(2);
                    });
                    if n > emissive::MAX_EMISSIVE_LIGHTS as u32 {
                        notes.push(format!(
                            "emissive-lights: budget {n} clamped to {} (the CB row cap)",
                            emissive::MAX_EMISSIVE_LIGHTS
                        ));
                    }
                    opts.emissive_lights_count = n;
                }
            }
            // Emitter-placement A/B lever (the --bvh-builder bake-off shape):
            // required value, last-wins by plain store; the VOCABULARY is
            // validated in main's lever block via emissive::set_cluster_mode
            // (illegal exits 2 there), which keeps the parse pure. Mirrored in
            // settings.rs (advanced.el_cluster — validated at APPLY through
            // emissive::parse_cluster, so a file value can't reach the exit).
            "--el-cluster" => {
                opts.el_cluster = args.next().unwrap_or_else(|| {
                    eprintln!("--el-cluster needs one of: grid | som");
                    std::process::exit(2);
                });
            }
            // Same "knob before scene load" pattern: the GPU reads it for the
            // static sampler's MaxAnisotropy, the CPU for Cone::aniso. 1 = off
            // ⇒ the isotropic ray-cone lod path runs verbatim (bit-identical
            // to the pre-aniso renderer). --no-mips forces it to 1 — mips are
            // the prerequisite (texture::set_aniso).
            "--no-aniso" => opts.aniso = 1,
            "--aniso" => {
                let n: u32 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&n| n >= 1 && n <= texture::MAX_ANISO_CAP)
                    .unwrap_or_else(|| {
                        eprintln!(
                            "--aniso needs an integer in 1..={} (1 = off)",
                            texture::MAX_ANISO_CAP
                        );
                        std::process::exit(2);
                    });
                opts.aniso = n;
            }
            "--no-hemi-share" => opts.hemi_share = false,
            "--no-cut-rays" => opts.cut_rays = false,
            "--continuation-rays" => {
                opts.sw_rays = true;
                opts.cut_rays = true;
            }
            "--sw-rays" => opts.sw_rays = true,
            "--no-sw-rays" => opts.sw_rays = false,
            "--no-cut-hemi" => opts.cut_hemi = false,
            "--cut-hemi" => opts.cut_hemi = true,
            "--bvh-ctrav" => {
                opts.c_trav = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--bvh-ctrav needs a number (SAH traversal/intersection cost ratio)");
                        std::process::exit(2);
                    });
            }
            "--bvh-maxleaf" => {
                opts.max_leaf = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&n: &usize| n >= 2)
                    .unwrap_or_else(|| {
                        eprintln!("--bvh-maxleaf needs an integer >= 2");
                        std::process::exit(2);
                    });
            }
            // Leaf sway (src/foliage.rs) — DEFAULT ON; --no-foliage-sway is
            // the kill lever, the clouds/fireflies shape; later flags win.
            // Mirrored in settings.rs (effects.foliage_sway / foliage_amp).
            "--foliage-sway" => opts.foliage_sway = true,
            "--no-foliage-sway" => opts.foliage_sway = false,
            "--foliage-amp" => {
                opts.foliage_amp = args
                    .next()
                    .and_then(|s| s.parse::<f32>().ok())
                    .filter(|v| v.is_finite() && (0.0..=8.0).contains(v))
                    .unwrap_or_else(|| {
                        eprintln!("--foliage-amp needs a multiplier in 0..=8 (default 1)");
                        std::process::exit(2);
                    });
            }
            "--ftree" => opts.ftree = true,
            "--no-ftree" => opts.ftree = false,
            "--ftree-tiles" => opts.ftree_tiles = true,
            "--no-ftree-tiles" => opts.ftree_tiles = false,
            "--wide-levels" => opts.wide_levels = true,
            "--no-wide-levels" => opts.wide_levels = false,
            "--bvh-axes" => {
                opts.split_axes = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&n: &usize| n == 1 || n == 3)
                    .unwrap_or_else(|| {
                        eprintln!("--bvh-axes needs 1 (widest centroid axis) or 3 (all axes)");
                        std::process::exit(2);
                    });
            }
            "--bvh-builder" => {
                opts.bvh_builder = args.next().unwrap_or_else(|| {
                    eprintln!("--bvh-builder needs one of: sah | lbvh | ploc | som");
                    std::process::exit(2);
                });
            }
            "--waveviz" => {
                // Optional value: bare = the overlay, `chs` = the mode-1
                // closest-hit variant. Consumed only on an exact match, so a
                // following scene path is safe (the --blas-split idiom).
                // Mirrored in settings.rs (advanced.waveviz — off/on/chs).
                opts.waveviz = if args.peek().is_some_and(|v| v == "chs") {
                    args.next();
                    2
                } else {
                    1
                };
            }
            "--blas-split" => {
                // Optional value: a bare flag takes the conventional-band cap,
                // an explicit N (e.g. 64) reaches the tiny-BLAS regime. The
                // next token is only CONSUMED when it is all digits — so
                // `--blas-split model.obj` and `--blas-split --stress 5000`
                // leave their arguments alone — but a numeric token that is
                // not a legal cap (0, or past u32) is a typo, not a scene
                // path, and exits rather than silently arming at the default
                // and landing in the positional arm as an OBJ named "0".
                let numeric = args.peek().is_some_and(|v| {
                    !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit())
                });
                let n = if numeric {
                    let v = args.next().unwrap();
                    v.parse::<u32>().ok().filter(|n| *n >= 1).unwrap_or_else(|| {
                        eprintln!(
                            "--blas-split: '{v}' is not a triangle cap (1..={})",
                            u32::MAX
                        );
                        std::process::exit(2);
                    })
                } else {
                    blas_split::DEFAULT_MAX_PRIMS
                };
                opts.blas_split = Some(n);
            }
            "--no-blas-split" => opts.blas_split = None,
            // --dual-gpu [N]: the secondary's share in eighths. Optional value,
            // the --blas-split idiom, so a following scene path is safe.
            "--dual-gpu" => {
                let numeric = args
                    .peek()
                    .is_some_and(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()));
                let n = if numeric {
                    let v = args.next().unwrap();
                    v.parse::<u32>().ok().filter(|n| (1..=7).contains(n)).unwrap_or_else(|| {
                        eprintln!(
                            "--dual-gpu: '{v}' is not a secondary share in eighths (1..=7)"
                        );
                        std::process::exit(2);
                    })
                } else {
                    2
                };
                opts.dual_gpu = Some(n);
            }
            "--no-dual-gpu" => {
                opts.dual_gpu = None;
                opts.dual_gpu_auto = false;
                opts.dual_gpu_arm = None;
            }
            // Arms if it wasn't already: a balancer with nothing to balance
            // is a no-op, and silently doing nothing is the failure mode the
            // loud-lever rule exists to prevent.
            "--dual-gpu-auto" => {
                opts.dual_gpu_auto = true;
                if opts.dual_gpu.is_none() {
                    opts.dual_gpu = Some(2);
                }
            }
            "--dual-gpu-depth" => {
                let v = args.next().unwrap_or_default();
                opts.dual_gpu_depth = v
                    .parse::<u32>()
                    .ok()
                    .filter(|k| (1..=3).contains(k))
                    .unwrap_or_else(|| {
                        eprintln!("--dual-gpu-depth: '{v}' is not a split level (1..=3)");
                        std::process::exit(2);
                    });
            }
            // Which pipeline the SECONDARY runs. Names the secondary only —
            // the primary's arm is the session's.
            //
            // ARMS IF IT WASN'T ALREADY, the `--dual-gpu-auto` rule: forcing an
            // arm on a device that was never opened is a silent no-op, and
            // silently doing nothing is the failure mode the loud-lever rule
            // exists to prevent. Mirrored in settings.rs (advanced.dual_gpu_arm
            // — same arming, same explicit dual_gpu:0 veto).
            "--dual-gpu-arm" => {
                let v = args.next().unwrap_or_default();
                if opts.dual_gpu.is_none() {
                    opts.dual_gpu = Some(2);
                }
                opts.dual_gpu_arm = Some(match v.as_str() {
                    "wave" | "wavefront" | "gpu" => crate::gpu::dual::Arm::Wave,
                    "dxr" => crate::gpu::dual::Arm::Dxr,
                    _ => {
                        eprintln!("--dual-gpu-arm: '{v}' is not wave|dxr");
                        std::process::exit(2);
                    }
                });
            }
            // The DXR pipeline's ray-dispatch mode (applied through
            // gpu::dxr::set_inline_mode). DEFAULT 1 = primary TraceRay +
            // inline RayQuery secondaries, which strictly dominates the
            // all-TraceRay pipeline (0) at every measured point on both
            // vendors; 2 = everything inline in raygen — and the INTEL
            // vendor default (`main::vendor_defaults`; any legal value here
            // sets `dxr_inline_explicit`, the policy's veto, so
            // `--dxr-inline 1` pins the cross-vendor default on Arc);
            // 3 = thin closest-hit + deferred compute shade (the 2026-08
            // Intel fat-CHS-hosting finding — bare-hit DispatchRays writes
            // records, cs_dxr_shade shades in compute). See the DXR
            // section's ablation table in CLAUDE.md.
            "--dxr-inline" => {
                let n: u32 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&n| n <= 3)
                    .unwrap_or_else(|| {
                        eprintln!(
                            "--dxr-inline needs 0 (all TraceRay), 1 (inline secondaries — \
                             the cross-vendor default), 2 (everything inline in raygen — \
                             the Intel default), or 3 (thin closest-hit + deferred compute \
                             shade)"
                        );
                        std::process::exit(2);
                    });
                opts.dxr_inline = n;
                opts.dxr_inline_explicit = true;
            }
            // The many-record material-sorted SBT ladder (gpu/dxr.rs's
            // SBT_MODE doc): 0 = off (today's one-record SBT), 1 = alias
            // records (identical code, distinct sort keys), 2 = class-
            // specialized, 3 = recursive class dispatch. A measurement
            // lever, never a default — no vendor policy, no settings row.
            "--dxr-sbt" => {
                let n: u32 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&n| n <= 3)
                    .unwrap_or_else(|| {
                        eprintln!(
                            "--dxr-sbt needs 0 (off — one shading record), 1 (alias records), \
                             2 (class-specialized), or 3 (recursive class dispatch)"
                        );
                        std::process::exit(2);
                    });
                opts.dxr_sbt = n;
            }
            "--spin" => {
                spin = Some(args.next().unwrap_or_else(|| {
                    eprintln!("--spin needs a workload: still | path");
                    std::process::exit(2);
                }));
            }
            "--spin-frames" => {
                spin_frames = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--spin-frames needs a frame count");
                        std::process::exit(2);
                    });
                spin_frames_explicit = true;
            }
            "--spin-hybrid" => spin_hybrid = true,
            "--spin-plain" => spin_hybrid = false,
            "--spin-warmup" => {
                spin_warmup = Some(
                    args.next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("--spin-warmup needs a frame count");
                            std::process::exit(2);
                        }),
                );
            }
            // --cinematic takes an OPTIONAL value (the peek idiom the parse
            // loop is Peekable for): a bare --cinematic is the `hero` still, so
            // the mode always does something useful and self-describing rather
            // than erroring or starting a 20-minute render.
            "--cinematic" => {
                let sel = match args.peek() {
                    Some(v) if !v.starts_with("--") => args.next(),
                    _ => None,
                };
                cinematic = Some(sel.unwrap_or_else(|| "hero".to_string()));
            }
            "--cinematic-out" => {
                cine.out = args.next().unwrap_or_else(|| {
                    eprintln!("--cinematic-out needs a directory");
                    std::process::exit(2);
                });
            }
            "--cinematic-res" => {
                let v = args.next().unwrap_or_else(|| {
                    eprintln!("--cinematic-res needs WxH, e.g. 1920x1080");
                    std::process::exit(2);
                });
                let parts: Vec<&str> = v.split(['x', 'X']).collect();
                let wh = if parts.len() == 2 {
                    match (parts[0].trim().parse::<usize>(), parts[1].trim().parse::<usize>()) {
                        (Ok(w), Ok(h)) if w >= 2 && h >= 2 && w <= 16384 && h <= 16384 => {
                            Some((w, h))
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                cine.res = Some(wh.unwrap_or_else(|| {
                    eprintln!("--cinematic-res: bad size '{v}' (want WxH, 2..16384 each)");
                    std::process::exit(2);
                }));
            }
            "--cinematic-samples" => {
                cine.samples = Some(
                    args.next()
                        .and_then(|v| v.parse::<u32>().ok())
                        .filter(|n| (1..=4096).contains(n))
                        .unwrap_or_else(|| {
                            eprintln!("--cinematic-samples needs 1..=4096");
                            std::process::exit(2);
                        }),
                );
            }
            "--cinematic-frames" => {
                cine.frames = Some(
                    args.next()
                        .and_then(|v| v.parse::<u32>().ok())
                        .filter(|n| (1..=100_000).contains(n))
                        .unwrap_or_else(|| {
                            eprintln!("--cinematic-frames needs 1..=100000");
                            std::process::exit(2);
                        }),
                );
            }
            "--cinematic-fps" => {
                cine.fps = args
                    .next()
                    .and_then(|v| v.parse::<u32>().ok())
                    .filter(|n| (1..=240).contains(n))
                    .unwrap_or_else(|| {
                        eprintln!("--cinematic-fps needs 1..=240");
                        std::process::exit(2);
                    });
            }
            "--cinematic-island" => {
                cine.island = Some(args.next().unwrap_or_else(|| {
                    eprintln!("--cinematic-island needs an island name");
                    std::process::exit(2);
                }));
            }
            "--cinematic-gi" => cine.gi = Some(true),
            "--no-cinematic-gi" => cine.gi = Some(false),
            "--cinematic-overlay" => cine.overlay = true,
            // off | hud | menu | settings:<Group>
            "--cinematic-hud" => {
                let spec = args.next().unwrap_or_else(|| "hud".to_string());
                cine.hud = match spec.as_str() {
                    "off" => None,
                    "hud" => Some(None),
                    "menu" => Some(Some(String::new())),
                    s if s.starts_with("settings:") => {
                        Some(Some(s["settings:".len()..].to_string()))
                    }
                    "settings" => Some(Some(settings::GROUPS[0].to_string())),
                    other => {
                        eprintln!(
                            "--cinematic-hud: unknown '{other}' \
                             (want off | hud | menu | settings | settings:<Group>)"
                        );
                        std::process::exit(2);
                    }
                };
            }
            // Stops, signed. Bounded at +/-8 because past that the tonemap is
            // either a white field or black — a typo, not an intent.
            "--cinematic-exposure" => {
                let v = args.next().unwrap_or_default();
                match v.parse::<f32>() {
                    Ok(ev) if ev.is_finite() && ev.abs() <= 8.0 => cine.exposure = Some(ev),
                    _ => {
                        eprintln!("--cinematic-exposure needs stops in -8..=8 (got '{v}')");
                        std::process::exit(2);
                    }
                }
            }
            "--cinematic-encode" => cine.encode = true,
            "--cinematic-dry-run" => cine.dry_run = true,
            "--cinematic-hdr" => cine.hdr = true,
            "--no-cinematic-hdr" => cine.hdr = false,
            "--no-vsync" => opts.vsync = false,
            "--vsync" => opts.vsync = true,
            // The swapchain flags are a three-way choice (8-bit SDR | Sdr10 |
            // HDR10 — one 10-bit format, two curves) spelled as two toggles,
            // so each arm writes ALL the fields — that is what makes
            // later-flags-win hold across the pairs (`--no-hdr --hdr10` = PQ,
            // `--hdr10 --no-hdr` = 8-bit SDR). Mirrored in settings.rs
            // (display.hdr / display.hdr10).
            "--hdr" => {
                opts.hdr = true;
                opts.hdr10 = false;
                opts.sdr10 = false;
            }
            "--no-hdr" => {
                opts.hdr = false;
                opts.hdr10 = false;
                opts.sdr10 = false;
            }
            "--hdr10" => {
                opts.hdr = true;
                opts.hdr10 = true;
                opts.sdr10 = false;
            }
            // "10-bit, but NOT PQ" — the explicit spelling of Sdr10 (gamma-2.2
            // deep colour), which is not reachable as "neither flag" on an
            // HDR-on display where PQ is the default. Implies the 10-bit
            // swapchain (the --hdr10 arm's shape).
            "--no-hdr10" => {
                opts.hdr = true;
                opts.hdr10 = false;
                opts.sdr10 = true;
            }
            "--hdr-paper-white" => {
                opts.hdr_paper_white = args
                    .next()
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|v| *v >= 1.0)
                    .unwrap_or_else(|| {
                        eprintln!("--hdr-paper-white needs a luminance in nits (e.g. 200)");
                        std::process::exit(2);
                    });
            }
            "--hdr-peak" => {
                opts.hdr_peak = Some(
                    args.next()
                        .and_then(|v| v.parse::<f32>().ok())
                        .filter(|v| *v >= 1.0)
                        .unwrap_or_else(|| {
                            eprintln!("--hdr-peak needs a luminance in nits (e.g. 1000)");
                            std::process::exit(2);
                        }),
                );
            }
            "--pix-markers" => opts.pix_markers = true,
            "--gpu-timing" => opts.gpu_timing = true,
            "--pix-path" => {
                opts.pix_path = args.next().unwrap_or_else(|| {
                    eprintln!("--pix-path needs a directory argument");
                    std::process::exit(2);
                });
            }
            "--xess-path" => {
                opts.xess_path = args.next().unwrap_or_else(|| {
                    eprintln!("--xess-path needs a directory argument");
                    std::process::exit(2);
                })
            }
            "--oidn-path" => {
                opts.oidn_path = args.next().unwrap_or_else(|| {
                    eprintln!("--oidn-path needs a directory argument");
                    std::process::exit(2);
                })
            }
            "--oidn-device" => {
                // Names map to oidn.h OIDNDeviceType values.
                opts.oidn_device = match args.next().as_deref() {
                    Some("default") => 0,
                    Some("cpu") => 1,
                    Some("sycl") => 2,
                    Some("cuda") => 3,
                    Some("hip") => 4,
                    _ => {
                        eprintln!("--oidn-device needs one of: default cpu sycl cuda hip");
                        std::process::exit(2);
                    }
                }
            }
            "--oidn-quality" => {
                opts.oidn_quality = match args.next().as_deref() {
                    Some("fast") => oidn::QUALITY_FAST,
                    Some("balanced") => oidn::QUALITY_BALANCED,
                    Some("high") => oidn::QUALITY_HIGH,
                    _ => {
                        eprintln!("--oidn-quality needs one of: fast balanced high");
                        std::process::exit(2);
                    }
                }
            }
            "--lock-res" => {
                // One flag, one scale, every arm — the per-mode defaults
                // (CPU quality, GPU native) are gone.
                opts.lock_scale = match args.next().as_deref() {
                    Some("dynamic") => None,
                    Some(s) => match xess::lock_scale(s) {
                        Some(r) => Some(r),
                        None => {
                            eprintln!("--lock-res needs quality|balanced|performance|ultra-performance|native|dynamic or a ratio in (0, 1]");
                            std::process::exit(2);
                        }
                    },
                    None => {
                        eprintln!("--lock-res needs quality|balanced|performance|ultra-performance|native|dynamic or a ratio in (0, 1]");
                        std::process::exit(2);
                    }
                };
                opts.lock_res_explicit = true;
            }
            // Ray Regeneration tuning overrides (FfxApiConfigureDenoiserKey).
            // Absent = the provider's own default, so a flagless session is
            // unchanged; each is an A/B lever on denoiser cleanliness.
            "--fsr-max-radiance"
            | "--fsr-stability-bias"
            | "--fsr-radiance-clip-k"
            | "--fsr-disocclusion-threshold"
            | "--fsr-normal-strength"
            | "--fsr-kernel-relaxation" => {
                let v: f32 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("{a} needs a float argument");
                        std::process::exit(2);
                    });
                let t = &mut opts.fsr_tune;
                match a.as_str() {
                    "--fsr-max-radiance" => t.max_radiance = Some(v),
                    "--fsr-stability-bias" => t.stability_bias = Some(v),
                    "--fsr-radiance-clip-k" => t.radiance_clip_k = Some(v),
                    "--fsr-disocclusion-threshold" => t.disocclusion_threshold = Some(v),
                    "--fsr-normal-strength" => t.normal_strength = Some(v),
                    _ => t.kernel_relaxation = Some(v),
                }
            }
            "--spp" => {
                opts.spp = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&n: &u32| n >= 1 && n <= dlss::MAX_SPP)
                    .unwrap_or_else(|| {
                        eprintln!("--spp needs an integer in 1..={}", dlss::MAX_SPP);
                        std::process::exit(2);
                    });
            }
            "--tod" => {
                opts.tod = args
                    .next()
                    .and_then(|s| s.parse::<f32>().ok())
                    .filter(|v| v.is_finite())
                    .map(|v| v.rem_euclid(24.0))
                    .or_else(|| {
                        eprintln!("--tod needs an hour, e.g. 17.5 (wraps into 0..24)");
                        std::process::exit(2);
                    });
            }
            "--oidn-no-clean-aux" => opts.oidn_clean_aux = false,
            "--gpu" => {
                opts.gpu = true;
                opts.mode_explicit = true;
            }
            "--check-gpu" => check_gpu = true,
            "--dxr" => {
                opts.dxr = true;
                dxr_explicit = true;
                opts.mode_explicit = true;
            }
            // The CPU frustum-tracer as the render mode: it is what both GPU
            // modes are peers of, so --cpu clears both. Later flags win (a
            // trailing --gpu/--dxr re-selects), matching the chain's algebra.
            "--cpu" => {
                opts.dxr = false;
                opts.gpu = false;
                opts.mode_explicit = true;
            }
            "--check-dxr" => check_dxr = true,
            "--dxc-path" => {
                opts.dxc_path = args.next().unwrap_or_else(|| {
                    eprintln!("--dxc-path needs a directory argument");
                    std::process::exit(2);
                })
            }
            "--prefer-nvidia" => opts.prefer = Some(gpu::adapter::Prefer::Nvidia),
            "--prefer-intel" => opts.prefer = Some(gpu::adapter::Prefer::Intel),
            "--prefer-amd" => opts.prefer = Some(gpu::adapter::Prefer::Amd),
            "--gpu-debug" => opts.gpu_debug = true,
            // BC7 is ON by default (the GPU arm at `fast`). --bc7 re-arms
            // after a --no-bc7 (later flags win, never switching an explicit
            // CPU arm); --bc7-cpu picks the ispc A/B arm; --bc7-quality keys
            // the CURRENT arm (and arms the GPU default, so order between it
            // and --bc7 doesn't matter).
            "--bc7" => opts.bc7 = opts.bc7.armed_or_default(),
            "--no-bc7" => opts.bc7 = bc7::Bc7Mode::Off,
            "--bc7-cpu" => {
                opts.bc7 = bc7::Bc7Mode::Cpu(opts.bc7.quality().unwrap_or(bc7::Quality::Fast))
            }
            "--bc7-quality" => {
                let q = args.next().unwrap_or_else(|| {
                    eprintln!("--bc7-quality needs ultrafast|fast|basic|slow");
                    std::process::exit(2);
                });
                let q = bc7::Quality::parse(&q).unwrap_or_else(|| {
                    eprintln!("--bc7-quality: unknown profile '{q}' (ultrafast|fast|basic|slow)");
                    std::process::exit(2);
                });
                opts.bc7 = opts.bc7.with_quality(q);
            }
            // Consumed by the pre-parse settings scan (settings::headless_args)
            // — this arm only keeps the token out of the positional fallback,
            // which would read it as a scene path.
            "--no-settings" => {}
            // Same shape: consumed by the pre-parse scan in main
            // (crash::disabled_by_args), because the crash handler installs
            // before this parser runs. The arm exists only so the positional
            // fallback does not read the flag as a scene path.
            "--no-crash-handler" => {}
            "--world" => world_flag = Some(true),
            "--no-world" => world_flag = Some(false),
            "--stress" => {
                stress = Some(
                    args.next()
                        .and_then(|s| s.parse::<usize>().ok())
                        .filter(|&n| n > 0)
                        .unwrap_or_else(|| {
                            eprintln!("--stress needs an object count, e.g. --stress 5000");
                            std::process::exit(2);
                        }),
                )
            }
            "--tile" => {
                let spec = args.next().unwrap_or_default();
                let (x, z) = match spec.split_once('x') {
                    Some((a, b)) => (a.trim().parse().ok(), b.trim().parse().ok()),
                    None => {
                        let n: Option<u32> = spec.trim().parse().ok();
                        (n, n)
                    }
                };
                tile = match (x, z) {
                    (Some(x), Some(z)) if x >= 1 && z >= 1 => Some((x, z)),
                    _ => {
                        eprintln!("--tile needs a grid, e.g. --tile 3 or --tile 4x2");
                        std::process::exit(2);
                    }
                };
            }
            "--cam" => {
                let parts: Vec<f32> = args
                    .next()
                    .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
                    .unwrap_or_default();
                if parts.len() != 6 {
                    eprintln!("--cam needs ex,ey,ez,tx,ty,tz (6 numbers), e.g. --cam 4,2,4,0,1,0");
                    std::process::exit(2);
                }
                cam_override = Some(Camera::look_at(
                    Vec3A::new(parts[0], parts[1], parts[2]),
                    Vec3A::new(parts[3], parts[4], parts[5]),
                    55f32.to_radians(),
                ));
            }
            // Printing is main's job, not the parser's (see `Cli::helped`).
            "--help" | "-h" => {
                helped = true;
                break;
            }

            // A mistyped long flag must not fall through to the positional
            // scene argument: `--preferIntel` used to reach the OBJ loader and
            // panic there ("failed to load OBJ '--preferIntel'"), which reads
            // as a renderer crash rather than a typo. Scene paths never start
            // with `--`, so this arm costs nothing legitimate.
            a if a.starts_with("--") => {
                eprintln!("unknown flag '{a}' — run --help for the flag list");
                std::process::exit(2);
            }
            _ => obj = Some(a),
        }
    }
    Cli {
        opts,
        obj,
        check,
        check_dlss,
        dlss_dump,
        check_oidn,
        oidn_dump,
        check_xess,
        xess_dump,
        check_fsr,
        check_nppd,
        nppd_dump,
        check_nrd,
        check_gpu,
        check_dxr,
        no_xess_explicit,
        fsr_forced,
        dxr_explicit,
        no_upscale,
        stress,
        tile,
        cam_override,
        spin,
        spin_frames,
        spin_frames_explicit,
        spin_hybrid,
        spin_warmup,
        cinematic,
        cine,
        world_flag,
        helped,
        notes,
    }
}

/// The `--help` text, printed by `main` when `Cli::helped` is set.
pub fn usage() {
                eprintln!("usage: frustracer [model.obj|.gltf|.glb] [--stress <n>] [--check] [--check-dlss] [--dlss-dump] [--no-dlss] [--check-oidn] [--oidn-dump] [--oidn] [--check-xess] [--xess-dump] [--xess] [--lock-res <r>] [--gpu-debug] [--oidn-path <dir>] [--oidn-device <d>] [--xess-path <dir>]");
                eprintln!("  --world       boot the curated multi-scene world (the flagless interactive default;");
                eprintln!("                exclusive with a scene arg, --stress, --tile, --spin, and --check*)");
                eprintln!("  --no-world    flagless boot uses the procedural default scene instead");
                eprintln!("  --stress <n>  procedural stress field of n objects (perf test; composes with --check)");
                eprintln!("  --tile <n|nxm>  replicate a loaded OBJ scene into an n×n (or n×m) grid of copies —");
                eprintln!("                flattened geometry, shared materials/textures (composes with --check)");
                eprintln!("  --cam <e,t>   start camera: ex,ey,ez,tx,ty,tz (reproducible benchmark viewpoints)");
                eprintln!("  --spin <still|path>  deterministic headless tracer benchmark");
                eprintln!("    --spin-frames N    total frames (default 2000)");
                eprintln!("    --spin-warmup N    excluded leading frames (Intel default 1600; otherwise 20)");
                eprintln!("    --spin-hybrid | --spin-plain  quadtree vs root-traversal A/B");
                eprintln!("                                  (CPU/--gpu only; --dxr has one DXR arm)");
                eprintln!("    --continuation-rays  software prototype: beam-produced opaque frontier");
                eprintln!("                           reused by leaf rays (--no-cut-rays = SW root control)");
                eprintln!("  --waveviz [chs]  wave-footprint overlay: every pixel wears its wave's hash");
                eprintln!("                color (I toggles live in GPU arms, every upscaler included;");
                eprintln!("                --spin runs dump waveviz-<arm>.png + compactness stats;");
                eprintln!("                chs = mode-1 closest-hit tickets, the hit-stage packing view)");
                eprintln!("  --cinematic [p]  MEDIA MODE: render stills / camera-spline video sequences");
                eprintln!("                headlessly. Presets: hero (default, one still) | islands |");
                eprintln!("                tour (the island lap, dawn -> moonlit night) | orbit | hud |");
                eprintln!("                list. Anything else is read as a JSON shot list. Writes a PNG");
                eprintln!("                sequence + manifest.json and PRINTS the ffmpeg commands");
                eprintln!("                (--cinematic-encode runs them). Loads the world by default.");
                eprintln!("                The GPU arms capture through the upscaler chain BY DEFAULT:");
                eprintln!("                DLSS-RR -> FSR4-RR -> XeSS -> FSR3 probed at 100% render scale");
                eprintln!("                (DLAA-grade), every sub-frame a fresh jittered frame with real");
                eprintln!("                MVs, and the frame written is the model's RECONSTRUCTED output.");
                eprintln!("                The chain flags steer it (--no-dlss, --fsr, ...); --no-upscale,");
                eprintln!("                a GI shot, --cpu, or an exhausted chain fall back to plain");
                eprintln!("                sub-frame accumulation, loudly.");
                eprintln!("    --cinematic-res WxH        default 1920x1080 (odd dims round down: yuv420p)");
                eprintln!("    --cinematic-samples N      sub-frames per OUTPUT frame (default 256 still /");
                eprintln!("                               32 sequence): upscaled capture feeds each one to");
                eprintln!("                               the temporal model (warm/converge passes, history");
                eprintln!("                               carried across output frames); accumulation arms");
                eprintln!("                               average them instead.");
                eprintln!("                               Composes with --spp, which is a different axis:");
                eprintln!("                               --spp shares the tile's inherited cut, so");
                eprintln!("                               `--spp 4 --cinematic-samples 16` is cheaper than");
                eprintln!("                               `--cinematic-samples 64` for the same 64 samples");
                eprintln!("                               (on --cpu/--gpu; on --dxr it is a wash).");
                eprintln!("    --cinematic-frames N       output frames for a sequence (default 900)");
                eprintln!("    --cinematic-fps N          default 30 — drives the cloud clock AND the");
                eprintln!("                               printed ffmpeg commands");
                eprintln!("    --cinematic-island <name>  orbit target");
                eprintln!("    --cinematic-gi             hemisphere-bounce GI (--cpu/--gpu only; GI shots");
                eprintln!("                               always render on the accumulation path — the");
                eprintln!("                               bounce integrator needs accumulating stills;");
                eprintln!("                               --no-cinematic-gi spells the default off)");
                eprintln!("    --cinematic-overlay        the quadtree debug overlay");
                eprintln!("    --cinematic-hud <spec>     off | hud | menu | settings | settings:<Group>");
                eprintln!("    --cinematic-out <dir>      default capture/");
                eprintln!("    --cinematic-exposure <ev>  exposure compensation in stops (-8..=8). The");
                eprintln!("                               enclosures — Sponza's atrium, San Miguel's patio,");
                eprintln!("                               Bistro's street — are correctly in shadow and want");
                eprintln!("                               +2 to +3, exactly as a camera would");
                eprintln!("    --cinematic-hdr            HDR output: sequences become 16-bit PQ/Rec.2020");
                eprintln!("                               frames encoded to HDR10 HEVC; stills also write a");
                eprintln!("                               linear OpenEXR master + a PQ PNG (and the ffmpeg");
                eprintln!("                               line for a viewable HDR AVIF). The SDR PNG is");
                eprintln!("                               always written as well. --no-cinematic-hdr spells");
                eprintln!("                               the default off.");
                eprintln!("    --cinematic-encode         run ffmpeg (default: just print the commands)");
                eprintln!("    --cinematic-dry-run        resolve + print the plan, render nothing");
                eprintln!("  --check       headless: verify hybrid vs reference, benchmark, write check.png");
                eprintln!("  --check-dlss  headless: G-buffer MV/depth/matrix self-test (no GPU needed)");
                eprintln!("  --dlss-dump   --check-dlss plus G-buffer PNG dumps (albedo/spec_albedo/normal/misc/mv)");
                eprintln!("  --no-dlss     skip the DLSS-RR level of the upscaler chain (falls to FSR4/XeSS/FSR3);");
                eprintln!("                the chain DLSS-RR -> FSR4-RR -> XeSS -> FSR3 is always on — the first");
                eprintln!("                supported level wins; --<x> force-starts the chain at that level");
                eprintln!("  --no-upscale  plain presentation: no temporal upscaler at all (benchmark escape)");
                eprintln!("  --no-bc7      upload scene textures as raw RGBA8 — BC7 block compression is ON by");
                eprintln!("                default (8 bpp vs 32; GPU-compute encode at upload; alpha-masked cutout");
                eprintln!("                textures stay exact RGBA8 always). --bc7-cpu forces the CPU ispc encode");
                eprintln!("                (the A/B arm; slow). --bc7-quality <p>: ultrafast|fast|basic|slow");
                eprintln!("                (default fast; on the GPU arm basic|slow deepen the mode-1 search).");
                eprintln!("                --bc7 re-arms the default after a --no-bc7 (later flags win)");
                eprintln!("  --check-oidn  headless: OIDN denoise self-test (needs the OIDN DLLs)");
                eprintln!("  --oidn-dump   --check-oidn plus before/after/G-buffer PNG dumps");
                eprintln!("  --oidn        start with OIDN denoising on (N toggles; implies DLSS off;");
                eprintln!("                --no-oidn spells the default off)");
                eprintln!("  --oidn-no-temporal  start OIDN without the temporal reprojection history (M toggles)");
                eprintln!("  --oidn-path   OIDN DLL directory (default: SDKs\\oidn.x64.windows\\bin)");
                eprintln!("  --oidn-device OIDN device: default|cpu|sycl|cuda|hip");
                eprintln!("  --oidn-quality OIDN RT-filter quality: fast|balanced|high (default balanced)");
                eprintln!("  --oidn-no-clean-aux  don't declare the albedo/normal guides noise-free (A/B lever)");
                eprintln!("  --nppd        NPPD neural denoising (J toggles; needs onnxruntime.dll + an exported");
                eprintln!("                model — see tools/nppd-export). Implies --xess: NPPD denoises at the");
                eprintln!("                render res before the upscale; --no-xess keeps the standalone window-res");
                eprintln!("                mode; --no-nppd spells the default off");
                eprintln!("  --check-nppd  headless: NPPD denoise self-test (needs onnxruntime.dll + the model)");
                eprintln!("  --nppd-dump   --check-nppd plus before/after PNG dumps");
                eprintln!("  --nppd-path   ONNX Runtime DLL directory (default: SDKs\\onnxruntime\\bin)");
                eprintln!("  --nppd-model  exported NPPD .onnx (default: SDKs\\nppd\\nppd_small.onnx)");
                eprintln!("  --nrd         NRD (ReBLUR) pre-upscale denoising — ON BY DEFAULT for XeSS/FSR3");
                eprintln!("                sessions (this flag spells the default; --no-nrd is the kill lever).");
                eprintln!("                The non-neural temporal denoiser cleaning the 1-spp signal at render");
                eprintln!("                res before the TAA-upscaler runs. GPU tracers only; DLSS-RR/FSR4-RR");
                eprintln!("                never arm it; excl. --nppd (explicit pair exits 2, defaulted disarms);");
                eprintln!("                missing SDKs\\NRD\\bin\\NRD.dll sheds loudly — install-prerequisites.bat");
                eprintln!("                nrd builds it");
                eprintln!("  --no-nrd      kill lever: plain (undenoised) XeSS/FSR3 — the pre-NRD baseline");
                eprintln!("  --nrd-path    NRD.dll directory (default: SDKs\\NRD\\bin)");
                eprintln!("  --nrd-perf    load the REBLUR_PERFORMANCE_MODE build (<nrd-path>\\perf\\NRD.dll —");
                eprintln!("                install-prerequisites.bat nrd builds both): cheaper ReBLUR internals,");
                eprintln!("                lower quality; missing perf DLL = loud line + the standard DLL");
                eprintln!("                (--no-nrd-perf spells the default)");
                eprintln!("  --nrd-max-stabilized-frames N  ReBLUR tuning (unset = library default 63): 0 drops");
                eprintln!("                the TemporalStabilization pass outright — the dispatch-level lever");
                eprintln!("  --nrd-prepass-radius X   ReBLUR pre-pass blur radius in px, both lobes (default");
                eprintln!("                30/50; 0 disables the prepasses)");
                eprintln!("  --nrd-max-accum-frames N ReBLUR history length (default 30; lower = more reactive,");
                eprintln!("                noisier)");
                eprintln!("  --nrd-no-anti-firefly    drop ReBLUR's anti-firefly filter (--nrd-anti-firefly");
                eprintln!("                spells the default)");
                eprintln!("  --check-nrd   headless: NRD math gates (DLL-free) + instance/dispatch contract (DLL)");
                eprintln!("  --frd         FRD — the from-scratch clean-room pre-upscale denoiser (opt-in until");
                eprintln!("                NRD parity; same arming surface: GPU tracers x XeSS/FSR3). Takes the");
                eprintln!("                one denoiser slot: an explicit --nrd beside it exits 2, the defaulted");
                eprintln!("                nrd disarms silently; --nppd beside it exits 2. No DLL, no install");
                eprintln!("                step — the kernels compile like every other unit");
                eprintln!("  --no-frd      the default, spelled explicitly");
                eprintln!("  --frd-max-accum-frames N (clamped loudly to 63 — the meta plane's n/63 wire cap)");
                eprintln!("                | --frd-fast-frames N | --frd-blur-radius X | --frd-clamp-sigma X");
                eprintln!("                | --frd-no-fp16 (force the fp32 shader arm)   FRD tuning (unset = the");
                eprintln!("                compiled frd.rs constants). --frd-max-stab-frames N and");
                eprintln!("                --frd-[no-]anti-firefly parse but are NOT YET WIRED (the stabilization");
                eprintln!("                sub-step and firefly pre-clamp are unbuilt phase-C/D items)");
                eprintln!("  --nppd-device NPPD execution provider: auto|cpu|dml|dml:<n> (default auto = DML then CPU)");
                eprintln!("  --check-xess  headless: XeSS dynamic-res contract self-test (no GPU or DLL needed)");
                eprintln!("  --xess-dump   --check-xess plus G-buffer PNG dumps");
                eprintln!("  --check-fsr   headless: FSR signal-split/encoding/MV contract self-test (no GPU or DLL)");
                eprintln!("  --fsr         force-start the upscaler chain at FSR4 + Ray Regeneration (K toggles;");
                eprintln!("                RDNA4 only — elsewhere the chain falls to XeSS then FSR 3.1)");
                eprintln!("  --fsr4        --fsr, but REQUIRED: exit(2) instead of falling through when FSR4 +");
                eprintln!("                Ray Regeneration is unavailable (suggests --fsr3 / --prefer-amd)");
                eprintln!("  --fsr3        force-start the chain at the FSR 3.1 upscale-only level (A/B lever)");
                eprintln!("  --no-fsr      skip both FSR levels of the chain (FSR4-RR and FSR 3.1)");
                eprintln!("  --fsr-max-radiance F   Ray Regeneration denoiser tuning, applied at denoiser");
                eprintln!("                creation — the firefly clamp, the highest-value knob for a 1-spp");
                eprintln!("                path tracer. Siblings, same shape: --fsr-stability-bias F,");
                eprintln!("                --fsr-radiance-clip-k F, --fsr-disocclusion-threshold F,");
                eprintln!("                --fsr-normal-strength F, --fsr-kernel-relaxation F. Each unset =");
                eprintln!("                configure nothing = the provider's own default");
                eprintln!("  --ffx-path    FidelityFX DLL directory (default: the FidelityFX-Samples-prebuilt drop)");
                eprintln!("  --fg          FRAME GENERATION for the wired upscaler family — ON BY DEFAULT");
                eprintln!("                (--no-fg disables): FSR sessions wrap the swapchain with the FidelityFX");
                eprintln!("                frame-interpolation proxy, DLSS sessions run raw-NGX DLSS-G (SDK builds),");
                eprintln!("                Intel XeSS sessions XeSS-FG; one generated frame per rendered frame;");
                eprintln!("                unsupported pairings fall through loudly");
                eprintln!("  --no-fg       kill lever: no frame generation");
                eprintln!("  --fg-path     directory holding amd_fidelityfx_framegeneration_dx12.dll (default: the");
                eprintln!("                drop's FSR sample dir — the FG provider does NOT ship next to the loader)");
                eprintln!("  --quinlight   REGISTERED CONSENSUS: suspend the chain's first-hit-wins rule, wire");
                eprintln!("                EVERY supported level at once (DLSS-RR + FSR4-RR + XeSS + FSR 3.1),");
                eprintln!("                run them all over the SAME traced frame, and present the LK-registered");
                eprintln!("                winsorized consensus of their outputs. GPU-fed only (--dxr/--gpu)");
                eprintln!("  --quin-anchor <n>  which engine defines the fuse's spatial frame (never warped);");
                eprintln!("                default = the denoising engine (DLSS-RR, else FSR4-RR), else engine 0");
                eprintln!("  --gpu         GPU-resident tracing: quadtree + shading in D3D12 compute with DXR");
                eprintln!("                RayQuery rays (needs the DXC DLLs and RT tier 1.1; falls back to CPU)");
                eprintln!("  --check-gpu   headless: GPU tracer gate suite (needs a D3D12 GPU + the DXC DLLs)");
                eprintln!("  --dxr         the DXR DispatchRays pipeline — the DEFAULT render mode (F toggles it");
                eprintln!("                against the CPU tracer live; needs the DXC DLLs and RT tier 1.0;");
                eprintln!("                falls back to CPU with a loud line)");
                eprintln!("  --cpu         the CPU frustum-tracer as the render mode (opts out of --dxr/--gpu)");
                eprintln!("  --check-dxr   headless: DXR pipeline gate suite (needs a D3D12 RT GPU + the DXC DLLs)");
                eprintln!("  --dxc-path    DXC DLL directory (default: SDKs\\dxc\\bin\\x64)");
                eprintln!("  --dxr-inline M  DXR ray-dispatch mode: 0 = all recursive TraceRay (the by-the-book");
                eprintln!("                pipeline), 1 = inline RayQuery secondaries (the cross-vendor default),");
                eprintln!("                2 = everything inline in raygen (the Intel default), 3 = thin");
                eprintln!("                closest-hit + deferred compute shade (experiment). Passing ANY value,");
                eprintln!("                1 included, pins your choice against the vendor policy");
                eprintln!("  --dxr-sbt M   the material-sorted SBT ladder (dev lever, default 0 = off): 1 = alias");
                eprintln!("                records (sort keys only), 2 = per-class specialized hit shaders,");
                eprintln!("                3 = recursive per-class dispatch (needs --dxr-inline 0)");
                eprintln!("  --xess        force-start the upscaler chain at XeSS-SR (X toggles;");
                eprintln!("                N cycles the OIDN denoise: off -> pre-upscale -> post-upscale)");
                eprintln!("  --oidn-post   start XeSS mode with OIDN placed AFTER the upscale (requires --xess)");
                eprintln!("  --xess-autoexposure  let XeSS compute exposure internally (A/B lever)");
                eprintln!("  --no-adaptive disable the adaptive shading rate in XeSS mode (uniform per-pixel shading;");
                eprintln!("                visibility is per-pixel either way)");
                eprintln!("  --no-hemi-share  disable the shared hemisphere capture in fb (H) frames — every");
                eprintln!("                shading point runs its own bounce tree (A/B lever)");
                eprintln!("  --no-temporal disable ALL previous-frame quadtree reuse (temporal cache, claim");
                eprintln!("                ring, query skip, structure replay) — every frame proves its empty");
                eprintln!("                space from scratch (A/B lever)");
                eprintln!("  --no-replay   keep temporal seeding; disable static-frame structure replay (A/B)");
                eprintln!("  --no-adopt    keep temporal seeding; disable the query skip / cut adoption (A/B)");
                eprintln!("  --discard-seeds  run the whole temporal pipeline but consume nothing — with");
                eprintln!("                --spin, (this - --no-temporal) = pure cost, (default - this) = benefit");
                eprintln!("  --no-cut-rays cut-seeded rays traverse from the BVH root instead; the inherited");
                eprintln!("                t_start survives (A/B lever)");
                eprintln!("  --sw-rays     the technical spelling behind --continuation-rays (--no-sw-rays =");
                eprintln!("                off): the software intersector without forcing cut seeding");
                eprintln!("  --cut-hemi    hemi leaf rays seed from their bounce cut (measured slower than one");
                eprintln!("                coherent root descent — default off, kept as the A/B)");
                eprintln!("  --no-ftree    hemi bound queries back on the binary BVH instead of the 8-wide");
                eprintln!("                frustum tree (A/B lever)");
                eprintln!("  --ftree-tiles the CPU tile recursion on the wide tree too (measured wall-neutral;");
                eprintln!("                default off)");
                eprintln!("  --no-wide-levels  every GPU quadtree level runs one thread per tile — no");
                eprintln!("                wave-cooperative shallow levels (A/B lever)");
                eprintln!("  --defer-shade material-sorted deferred shading on plain-path leaf tiles (a candid");
                eprintln!("                measured no-win, kept as the experiment's record)");
                eprintln!("  --bvh-ctrav F | --bvh-axes 1|3 | --bvh-maxleaf N");
                eprintln!("                the binned-SAH build knobs at defaults 3 / 3 / 8 — the memory lever,");
                eprintln!("                the speed lever, the leaf-size cap; build params key the scene");
                eprintln!("                cache, so sweeps never collide with a stale sidecar");
                eprintln!("  --bvh-builder sah|lbvh|ploc|som  swap the ray-BVH builder (bake-off lever;");
                eprintln!("                sah wins and is the default)");
                eprintln!("  --blas-split [N]  triangles per BLAS chunk (GPU; default 65536 — the split IS the");
                eprintln!("                default, a single huge BLAS removes the device on Intel);");
                eprintln!("                --no-blas-split = one BLAS over the whole scene");
                eprintln!("  --no-clouds   disable the drifting volumetric cloud layer (on by default: raymarched");
                eprintln!("                FBM slab — sky, reflections, glass, and cloud shadows on the direct sun;");
                eprintln!("                off is bit-identical to the pre-cloud renderer)");
                eprintln!("  --cloud-shadow N  cells/wavelength for the slab-space cloud-shadow cache (default 16;");
                eprintln!("                GPU sun-transmittance cache, ~-21%/sample of the cloud bill; the domain");
                eprintln!("                reduction is EXACT, only bilinear interp approximates). --no-cloud-shadow = off");
                eprintln!("  --sky-lod K   pixel pitch of the amortized cloud-march lattice, power of two (default 4;");
                eprintln!("                sharp half — sun/stars — stays per-pixel; ~0.14% sky error). --no-sky-lod = off");
                eprintln!("  --no-fireflies  disable the firefly point lights (on by default AFTER DUSK — under");
                eprintln!("                --tod they fade in with the stars: curl-noise drift, real 1/d² light");
                eprintln!("                with hard shadow rays + a depth-tested glow; a day session has zero");
                eprintln!("                fireflies and is bit-identical structurally)");
                eprintln!("  --fireflies N   firefly count (default {}, max {})", fireflies::DEFAULT_COUNT, fireflies::MAX_FIREFLIES);
                eprintln!("  --tod H       start time-of-day: float hours wrapped into 0..24 along the sun arc");
                eprintln!("                (06:00 east horizon -> 12:00 zenith -> 18:00 west; after sunset the");
                eprintln!("                antipodal full moon is the one light and the star field fades in).");
                eprintln!("                Hold , / . to scrub live at 1 game-hour per second. Flagless = the");
                eprintln!("                default sun, bit-identical to the pre-TOD renderer");
                eprintln!("  --emissive-lights [N]  ARM emissive surfaces lighting the scene (OFF by default —");
                eprintln!("                the CPU shadow-ray cost is real, measured bistro +5.5 ms at N=32, and");
                eprintln!("                only emissive-mapped scenes like bistro benefit):");
                eprintln!("                Ke/map_Ke/glTF-emissive triangles cluster into <= {} virtual disc", emissive::MAX_EMISSIVE_LIGHTS);
                eprintln!("                lights sampled in the direct tier — windowed falloff, one hard shadow");
                eprintln!("                ray each, zero rng, x{} artistic boost; bare flag = budget {}, N", emissive::EL_BOOST, emissive::EL_DEFAULT);
                eprintln!("                overrides; GI (H) frames keep the exact gather and skip the clusters");
                eprintln!("  --no-emissive-lights  the default, spelled explicitly (later flags win)");
                eprintln!("  --el-cluster M  emitter clustering: grid (default, the shipped clusterer) | som");
                eprintln!("                (deterministic batch-SOM/weighted-Lloyd placement refinement — A/B lever)");
                eprintln!("  --no-foliage-sway  disable wind-swayed foliage (ON by default: alpha-cutout leaves");
                eprintln!("                bucket into per-cell chunks translated on the cloud clock — ALL render");
                eprintln!("                modes (CPU/GPU/DXR; swept leaf AABBs keep every claim sound); a scene");
                eprintln!("                with no foliage-classified materials is structurally untouched; off is");
                eprintln!("                bit-identical; --foliage-sway spells the default)");
                eprintln!("  --foliage-amp X sway amplitude multiplier, 0..=8 (default 1; >1 = one cold BVH rebuild)");
                eprintln!("  --no-mips     no texture mip chains; every trilinear sample degenerates to the");
                eprintln!("                pre-mip bilinear (A/B lever; mips are on by default — implies --no-aniso)");
                eprintln!("  --no-h2n      don't Sobel-convert grayscale bump maps into normal maps (they are");
                eprintln!("                dropped, the pre-conversion behavior)");
                eprintln!("  --no-n2h      don't derive heightfields from normal maps (Frankot–Chellappa) — no");
                eprintln!("                alpha-channel height, height_amp stays 0");
                eprintln!("  --no-slope-mips  normal-map mips back on the raw-byte box filter (A/B lever; the");
                eprintln!("                default slope-space filter preserves mean tilt, so normal maps stop");
                eprintln!("                flattening with distance)");
                eprintln!("  --no-spec-aa  no slope-variance -> roughness fold (A/B lever; the default keeps");
                eprintln!("                detail maps in the rendering equation at every distance — what a mip");
                eprintln!("                averages away widens the GGX lobe instead of vanishing, so distant");
                eprintln!("                bumpy surfaces shade matte instead of mirror-flat)");
                eprintln!("  --normal-strength K  multiply every material's normal-map strength (0.0..=8.0,");
                eprintln!("                default 1 = bit-identical; 0 = normals off). Post-cache, so relief's");
                eprintln!("                height_amp stays unscaled — decode slopes and --heightfield relief");
                eprintln!("                deliberately decouple at K != 1");
                eprintln!("  --no-tinted-shadows  shadow/AO rays binary-block on transmissive surfaces (the");
                eprintln!("                pre-feature behavior; default: they pass with a transmission×albedo tint)");
                eprintln!("  --no-spray    keep tiny transmissive islands (fountain droplets) as clear glass");
                eprintln!("                instead of white-scatter spray (load-time retag; keys the scene cache)");
                eprintln!("  --no-depth-tint  no Beer–Lambert attenuation over the transmission chain's interior");
                eprintln!("                segments (water loses its depth-graded tint)");
                eprintln!("  --no-detail-tex  no Unreal-1 style detail texturing: procedural close-up albedo");
                eprintln!("                grain + micro-bump on magnified textured hits (runtime shading lever;");
                eprintln!("                default on — only fires where the base texture blurs, lod < 0)");
                eprintln!("  --no-detail-ao  no detail surface AO/shadows: the detail height's pits stop");
                eprintln!("                darkening ambient + specular (cavity), the horizon-marched sun");
                eprintln!("                shadows stop (the closed-form heightfield trace toward the sun),");
                eprintln!("                and the coarse occlusion pools + their relief rims go flat");
                eprintln!("                (a no-op wherever detail-tex never fires)");
                eprintln!("  --detail-strength K  detail GRAIN strength multiplier, 0.0..=4.0 (default 0.5;");
                eprintln!("                1.0 = the original full-strength field; scales the close-up");
                eprintln!("                albedo grain + micro-bump)");
                eprintln!("  --detail-ao-strength K  detail AO strength multiplier, 0.0..=4.0 (default 0.125;");
                eprintln!("                1.0 = original; scales the occlusion pools, cavity, and marched");
                eprintln!("                sun shadows)");
                eprintln!("  --detail-untex-scale K  UNTEXTURED materials' synthetic detail texel scale,");
                eprintln!("                0.0..=4.0 (default 1.0; a multiplier on DETAIL_UNTEX_K x the");
                eprintln!("                content diagonal — what puts the detail field on albedo-map-free");
                eprintln!("                scenes like powerplant; 0 = untextured detail off, the bitwise A/B)");
                eprintln!("  --no-amb-bump  no ambient bump response: the SH ambient stops amplifying its");
                eprintln!("                irradiance response to the shading normal's deviation (normal");
                eprintln!("                maps + detail bump read flat under sky light again; a no-op on");
                eprintln!("                flat-shaded geometry)");
                eprintln!("  --no-rtgi     real-time GI off: the ambient tier reverts to flat SH-sky x AO");
                eprintln!("                (the pre-RTGI renderer bit-exactly). Default ON: one cosine bounce");
                eprintln!("                ray per pixel per frame IS the ambient — real one-bounce GI the");
                eprintln!("                temporal denoisers/accumulation integrate (--rtgi spells the default)");
                eprintln!("  --auto-exposure  ARM the display-stage aperture controller (OFF by default —");
                eprintln!("                RTGI lights enclosures for real, so the aperture holds at 1.0; armed,");
                eprintln!("                exposure eases toward mid-grey; headless paths never adapt either way)");
                eprintln!("  --no-auto-exposure  the default, spelled explicitly (later flags win)");
                eprintln!("  --exposure-bias EV  manual aperture offset in stops (-8..=8, default 0; composes");
                eprintln!("                with auto-exposure and still applies with it off — the manual lever)");
                eprintln!("  --no-water    classify the fountain as generic glassware, not the water class");
                eprintln!("                (no blue-green tint / IOR 1.33 / ripple normals; keys the scene cache)");
                eprintln!("  --no-coincident-cull  keep transmissive faces exactly coincident with opaque faces");
                eprintln!("                (the pre-cull z-fight: CPU and GPU break the tie differently, and the");
                eprintln!("                transmission chain can tunnel past the opaque face; keys the scene cache)");
                eprintln!("  --heightfield ARM relief rendering and start it ON where the scene carries height");
                eprintln!("                data (V toggles relief vs normal-mapping live in the armed session).");
                eprintln!("                The DEFAULT is unarmed: no swept AABBs, no march — the pre-relief");
                eprintln!("                renderer bit-exactly (the swept tree costs real BVH quality)");
                eprintln!("  --no-heightfield  the default, spelled explicitly (later flags win)");
                eprintln!("  --aniso N     max anisotropy for texture filtering, 1..=16 (default 16; 1 = off, i.e.");
                eprintln!("                the isotropic ray-cone trilinear path verbatim). --no-aniso = --aniso 1");
                eprintln!("  --spp <n>     primary samples per pixel per frame (1..=128, default 1; U doubles live).");
                eprintln!("                The N jittered samples share the tile's inherited t_start/node cut, so");
                eprintln!("                the quadtree's per-tile cost amortizes over N× the rays (--cpu/--gpu;");
                eprintln!("                --dxr traces from the TLAS root, so there it is plain supersampling).");
                eprintln!("                They average into ONE per-pixel value — a ~1/N-variance frame for the");
                eprintln!("                upscaler/denoiser. Pinned to 1 on hemisphere-bounce (H) frames");
                eprintln!("  --lock-res    DLSS/XeSS render res: quality|balanced|performance|ultra-performance|native,");
                eprintln!("                a ratio in (0, 1], or dynamic (the step-wise DRS controller); default native (100%)");
                eprintln!("  --hdr | --no-hdr  10-bit swapchain with the curve picked by the display probe");
                eprintln!("                (the default) | the legacy 8-bit SDR swapchain (A/B lever, and the");
                eprintln!("                frame-generation wrap-failure fallback)");
                eprintln!("  --hdr10 | --no-hdr10  force the PQ (HDR10) declaration | force 10-bit gamma-2.2");
                eprintln!("                deep colour (10-bit but NOT PQ); later flags win across the three-way");
                eprintln!("  --hdr-paper-white N  where linear 1.0 lands, in nits (default 200; lower buys");
                eprintln!("                more highlight headroom above white)");
                eprintln!("  --hdr-peak N  override the display's reported peak, in nits (wins over the probe,");
                eprintln!("                including over an \"HDR off\" verdict)");
                eprintln!("  --no-vsync    uncapped presentation (sync interval 0 on a tearing swapchain) so");
                eprintln!("                frame times measure the renderer, not the monitor (--vsync = default)");
                eprintln!("  --xess-path   XeSS DLL directory (default: SDKs\\XeSS-SDK\\bin)");
                eprintln!("  --prefer-nvidia | --prefer-intel | --prefer-amd");
                eprintln!("                pick that vendor's adapter for the D3D12 device (default NVIDIA, or AMD");
                eprintln!("                with --fsr; a preference, not a requirement — features the picked GPU");
                eprintln!("                can't support fall back with a log line)");
                eprintln!("  --dual-gpu [N]  split the frame across two adapters, giving the SECONDARY N of the");
                eprintln!("                8 level-3 tile rows (1..=7, default 2; --no-dual-gpu = off, the default).");
                eprintln!("                Works with either render mode; the secondary's own pipeline follows its");
                eprintln!("                adapter's vendor (Intel -> wavefront, NVIDIA/AMD -> DXR).");
                eprintln!("                MEASURED: it LOSES on a box whose second slot is electrically x4 — the");
                eprintln!("                band is the whole cost. --dual-gpu-auto is how you find that out.");
                eprintln!("    --dual-gpu-auto   hand the share to the balancer (N is then the STARTING share);");
                eprintln!("                      it converges to 0 with a stated verdict where splitting cannot pay");
                eprintln!("    --dual-gpu-depth K  the quadtree level the split is assigned at (1..=3, default 3 =");
                eprintln!("                      eighths; K=1 is halves, with less duplicated ladder work)");
                eprintln!("    --dual-gpu-arm wave|dxr  force the SECONDARY's pipeline instead of the vendor policy");
                eprintln!("                      (names the secondary only; arms --dual-gpu at its default if it");
                eprintln!("                      wasn't already; an adapter without RT tier 1.0 still degrades to");
                eprintln!("                      the wavefront, loudly)");
                eprintln!("  --gpu-debug   D3D12 debug layer + GPU-based validation");
                eprintln!("  --gpu-timing  D3D12 timestamp queries: a per-region GPU-ms table every 120 frames");
                eprintln!("                and at exit — vendor-neutral, and the only per-pass GPU profiler");
                eprintln!("                that works on Intel Arc");
                eprintln!("  --pix-markers PIX Begin/End events on the D3D12 command lists (needs");
                eprintln!("                WinPixEventRuntime.dll; a missing DLL is one loud line, never fatal)");
                eprintln!("  --pix-path    WinPixEventRuntime.dll directory (default: SDKs\\pix\\bin\\x64)");
                eprintln!("  --no-audio    no audio (ON by default in interactive sessions: per-island CC0");
                eprintln!("                ambience crossfaded by camera proximity, plus a procedural wind");
                eprintln!("                swish tracking real camera speed; --audio spells the default;");
                eprintln!("                display-only — headless runs never construct the device)");
                eprintln!("  --no-settings ignore {} for this run (the pause menu's", settings::FILE_NAME);
                eprintln!("                saved settings, read as defaults that CLI flags override;");
                eprintln!("                headless --check*/--spin runs always ignore it)");
                eprintln!("  --no-crash-handler");
                eprintln!("                don't install the crash handler (default ON: on a fault it");
                eprintln!("                prints a symbolized Rust+C++ stack and writes");
                eprintln!("                frustracer-crash-<pid>.txt/.dmp next to the exe;");
                eprintln!("                FR_NO_CRASH=1 does the same, FR_CRASH_FULLDUMP=1 dumps");
                eprintln!("                full memory, FR_CRASH_VERIFY=1 reports after main whether");
                eprintln!("                the filter is still ours, and");
                eprintln!("                FR_CRASH_TEST=deref|cpp|panic|overflow|atexit faults");
                eprintln!("                on purpose to exercise it)");
                eprintln!("  --help | -h   this text (stops the parse — flags after it are not applied)");
}

// ---------------------------------------------------------------------------
// Gate (run by --check)
// ---------------------------------------------------------------------------

/// Every process global the lever fields feed, as one comparable string. The
/// purity gate's instrument: `parse_from` must not move ANY of it.
///
/// `max_aniso` covers the mips↔aniso coupling as well as aniso itself
/// (`set_aniso` clamps to 1 while mips are off), which is why the pair is one
/// contract and gets applied in one order.
fn lever_snapshot() -> String {
    format!(
        "mips={} aniso={} h2n={} n2h={} smips={} saa={} tint={} spray={} depth={} detail={} dao={} \
         dstr={} daostr={} duntex={} \
         ambb={} rtgi={} aexp={} ebias={} water={} ccull={} harm={} hon={} bloom={} clouds={} ff={} ffn={} el={} eln={} \
         elcluster={} cshadow={} skylod={} dxrinline={} dxrsbt={} fsway={} famp={}",
        texture::mips_enabled(),
        texture::max_aniso(),
        texture::h2n_enabled(),
        texture::n2h_enabled(),
        texture::slope_mips_enabled(),
        scene::spec_aa(),
        scene::tinted_shadows(),
        scene::spray_enabled(),
        scene::depth_tint(),
        scene::detail_tex(),
        scene::detail_ao(),
        scene::detail_strength(),
        scene::detail_ao_strength(),
        scene::detail_untex_scale(),
        scene::amb_bump(),
        crate::shade::rtgi_enabled(),
        crate::autoexp::enabled(),
        crate::autoexp::bias(),
        scene::water_enabled(),
        scene::coincident_cull_enabled(),
        bvh::height_armed(),
        bvh::height_on(),
        bloom::enabled(),
        clouds::enabled(),
        fireflies::enabled(),
        fireflies::count(),
        emissive::enabled(),
        emissive::budget(),
        emissive::cluster_mode_name(),
        gpu::trace::cloud_shadow_n(),
        gpu::trace::sky_lod(),
        gpu::dxr::dxr_inline_mode(),
        gpu::dxr::dxr_sbt_mode(),
        crate::foliage::armed(),
        crate::foliage::amp_mult(),
    )
}

fn parse_argv(args: &[&str]) -> Cli {
    parse_from(defaults(), args.iter().map(|s| s.to_string()))
}

/// Pins the parser's contracts. Run by `--check`.
///
/// What it deliberately does NOT cover: the arms that `exit(2)` on malformed
/// input (a bad `--aniso`, `--tile`, `--lock-res`, ...). Those terminate the
/// process by design — there is nowhere useful to return to from a broken
/// command line — so they are unreachable from in-process gates, and every
/// argv below is valid.
pub fn self_test() -> Result<(), String> {
    // ---- 1. PURITY --------------------------------------------------------
    // The gate this module exists for, and the reason it can live inside
    // `--check` at all: parse an argv that moves EVERY lever off its default
    // and require the process globals to come back untouched. A regression
    // here means a `--check` run is being corrupted by its own CLI gate.
    let before = lever_snapshot();
    let c = parse_argv(&[
        "--no-mips",
        "--aniso",
        "4",
        "--no-h2n",
        "--no-n2h",
        "--no-slope-mips",
        "--no-spec-aa",
        "--normal-strength",
        "2.5",
        "--no-tinted-shadows",
        "--no-spray",
        "--no-depth-tint",
        "--no-detail-tex",
        "--no-detail-ao",
        "--detail-strength",
        "2",
        "--detail-ao-strength",
        "0.5",
        "--detail-untex-scale",
        "0.25",
        "--no-amb-bump",
        "--no-rtgi",
        "--auto-exposure",
        "--exposure-bias",
        "1.5",
        "--no-water",
        "--no-coincident-cull",
        "--heightfield",
        "--no-bloom",
        "--no-clouds",
        "--no-cloud-shadow",
        "--no-sky-lod",
        "--no-fireflies",
        "--fireflies",
        "7",
        "--no-emissive-lights",
        "--emissive-lights",
        "9",
        "--el-cluster",
        "som",
        "--dxr-inline",
        "2",
        "--dxr-sbt",
        "3",
        "--waveviz",
        "chs",
        "--no-foliage-sway",
        "--foliage-amp",
        "2",
        "--dual-gpu",
        "3",
        "--dual-gpu-depth",
        "2",
        "--dual-gpu-arm",
        "dxr",
        "--frd",
        "--frd-max-accum-frames",
        "12",
        "--frd-no-fp16",
    ]);
    let after = lever_snapshot();
    if after != before {
        return Err(format!(
            "parse_from wrote a process global — the parser is not pure.\n  before: {before}\n  after:  {after}"
        ));
    }
    // ...and the fields DID move. Without this the check above passes
    // vacuously for a parser that simply ignores every flag.
    let o = &c.opts;
    for (name, took) in [
        ("dual_gpu", o.dual_gpu == Some(3)),
        ("dual_gpu_depth", o.dual_gpu_depth == 2),
        ("dual_gpu_arm", o.dual_gpu_arm == Some(crate::gpu::dual::Arm::Dxr)),
        ("frd", o.frd && o.frd_explicit),
        ("frd_tune", o.frd_tune.max_accum_frames == Some(12) && o.frd_tune.no_fp16),
        ("mips", !o.mips),
        ("aniso", o.aniso == 4),
        ("h2n", !o.h2n),
        ("n2h", !o.n2h),
        ("slope_mips", !o.slope_mips),
        ("spec_aa", !o.spec_aa),
        ("normal_strength", o.normal_strength == 2.5),
        ("tinted_shadows", !o.tinted_shadows),
        ("spray", !o.spray),
        ("depth_tint", !o.depth_tint),
        ("detail_tex", !o.detail_tex),
        ("detail_ao", !o.detail_ao),
        ("detail_strength", o.detail_strength == 2.0),
        ("detail_ao_strength", o.detail_ao_strength == 0.5),
        ("detail_untex_scale", o.detail_untex_scale == 0.25),
        ("amb_bump", !o.amb_bump),
        ("rtgi", !o.rtgi),
        // Default OFF: "moved" means ARMED (the argv passes --auto-exposure).
        ("autoexp", o.autoexp),
        ("exposure_bias", o.exposure_bias == 1.5),
        ("water", !o.water),
        ("coincident_cull", !o.coincident_cull),
        ("heightfield", o.heightfield),
        ("bloom", !o.bloom),
        ("clouds", !o.clouds),
        ("cloud_shadow", o.cloud_shadow == 0),
        ("sky_lod", o.sky_lod == 1),
        ("fireflies", !o.fireflies),
        ("fireflies_count", o.fireflies_count == 7),
        // Default OFF: "moved" means ARMED (the trailing --emissive-lights 9
        // wins over the earlier --no-emissive-lights in the argv above).
        ("emissive_lights", o.emissive_lights),
        // Either spelling sets it (this argv carries both) — the
        // upscaler-default veto, pinned properly in section 2 below.
        ("emissive_lights_explicit", o.emissive_lights_explicit),
        ("emissive_lights_count", o.emissive_lights_count == 9),
        // A field, not a process global — lever_snapshot's elcluster entry
        // additionally proves the parse never called set_cluster_mode.
        ("el_cluster", o.el_cluster == "som"),
        ("dxr_inline", o.dxr_inline == 2),
        ("dxr_inline_explicit", o.dxr_inline_explicit),
        // A field, not a process global — the purity gate's lever_snapshot
        // additionally proves the parse never called set_sbt_mode.
        ("dxr_sbt", o.dxr_sbt == 3),
        ("waveviz", o.waveviz == 2),
        ("foliage_sway", !o.foliage_sway),
        ("foliage_amp", o.foliage_amp == 2.0),
    ] {
        if !took {
            return Err(format!("lever field `{name}` did not take its flag"));
        }
    }

    // ---- 2. later flags win ------------------------------------------------
    // The codebase's parse rule, on the paired arms that spell it. Both
    // orders, because a pair that only works one way is the bug.
    if !parse_argv(&["--no-heightfield", "--heightfield"]).opts.heightfield {
        return Err("--no-heightfield --heightfield must ARM (later flags win)".into());
    }
    if parse_argv(&["--heightfield", "--no-heightfield"]).opts.heightfield {
        return Err("--heightfield --no-heightfield must disarm".into());
    }
    if !parse_argv(&["--no-emissive-lights", "--emissive-lights"]).opts.emissive_lights {
        return Err("--no-emissive-lights --emissive-lights must re-arm (later flags win)".into());
    }
    if parse_argv(&["--emissive-lights", "--no-emissive-lights"]).opts.emissive_lights {
        return Err("--emissive-lights --no-emissive-lights must disarm".into());
    }
    if parse_argv(&["--no-aniso", "--aniso", "8"]).opts.aniso != 8 {
        return Err("--no-aniso --aniso 8 must land on 8".into());
    }
    if parse_argv(&["--dual-gpu", "3", "--no-dual-gpu"]).opts.dual_gpu.is_some() {
        return Err("--dual-gpu 3 --no-dual-gpu must disarm".into());
    }
    if parse_argv(&["--dual-gpu-auto", "--no-dual-gpu"]).opts.dual_gpu_auto {
        return Err("--no-dual-gpu must clear the balancer too, not just the share".into());
    }
    // ...and the secondary's arm with them: --no-dual-gpu spells "no second
    // device at all", so leaving a forced arm behind would have it apply to
    // whatever a later --dual-gpu re-arms.
    if parse_argv(&["--dual-gpu-arm", "dxr", "--no-dual-gpu"]).opts.dual_gpu_arm.is_some() {
        return Err("--no-dual-gpu must clear the forced secondary arm too".into());
    }
    // The default is the VENDOR POLICY, not an arm — `None` is what lets
    // `arm_for` decide, and a default of Some(_) would pin every box to one
    // pipeline while looking like it had chosen.
    if parse_argv(&[]).opts.dual_gpu_arm.is_some() {
        return Err("--dual-gpu-arm must default to None (the vendor policy decides)".into());
    }
    if parse_argv(&["--dual-gpu-arm", "wave"]).opts.dual_gpu_arm
        != Some(crate::gpu::dual::Arm::Wave)
    {
        return Err("--dual-gpu-arm wave must select the wavefront".into());
    }
    // --dual-gpu-auto ARMS: a balancer with nothing to balance is a silent
    // no-op, which is the failure the loud-lever rule exists to prevent.
    if parse_argv(&["--dual-gpu-auto"]).opts.dual_gpu.is_none() {
        return Err("--dual-gpu-auto must arm the split, not sit inert".into());
    }
    // --dual-gpu-arm ARMS for the same reason: an arm forced on a device that
    // was never opened is the same silent no-op.
    if parse_argv(&["--dual-gpu-arm", "dxr"]).opts.dual_gpu.is_none() {
        return Err("--dual-gpu-arm must arm the split, not sit inert".into());
    }
    if parse_argv(&["--dual-gpu", "5", "--dual-gpu-arm", "wave"]).opts.dual_gpu != Some(5) {
        return Err("--dual-gpu-arm must not overwrite an explicit starting share".into());
    }
    // ...and it must not overwrite an explicitly chosen starting share.
    if parse_argv(&["--dual-gpu", "5", "--dual-gpu-auto"]).opts.dual_gpu != Some(5) {
        return Err("--dual-gpu-auto must keep an explicit starting share".into());
    }
    if parse_argv(&[]).opts.normal_strength != 1.0 {
        return Err("normal_strength must default to 1.0 (the bit-identical off arm)".into());
    }
    if parse_argv(&["--normal-strength", "2", "--normal-strength", "0.5"]).opts.normal_strength
        != 0.5
    {
        return Err("--normal-strength must be last-wins".into());
    }
    if parse_argv(&["--aniso", "8", "--no-aniso"]).opts.aniso != 1 {
        return Err("--aniso 8 --no-aniso must land on 1".into());
    }
    if parse_argv(&["--no-cloud-shadow", "--cloud-shadow", "32"]).opts.cloud_shadow != 32 {
        return Err("--no-cloud-shadow --cloud-shadow 32 must land on 32".into());
    }
    if parse_argv(&["--no-sky-lod", "--sky-lod", "8"]).opts.sky_lod != 8 {
        return Err("--no-sky-lod --sky-lod 8 must land on 8".into());
    }
    if !parse_argv(&["--no-ftree", "--ftree"]).opts.ftree {
        return Err("--no-ftree --ftree must re-enable".into());
    }
    if !parse_argv(&["--no-foliage-sway", "--foliage-sway"]).opts.foliage_sway
        || parse_argv(&["--foliage-sway", "--no-foliage-sway"]).opts.foliage_sway
        || !parse_argv(&[]).opts.foliage_sway
    {
        return Err("foliage sway must default ON and obey later-flags-win".into());
    }
    if parse_argv(&["--foliage-amp", "0.5"]).opts.foliage_amp != 0.5
        || parse_argv(&[]).opts.foliage_amp != 1.0
    {
        return Err("--foliage-amp must parse its value and default to 1.0".into());
    }
    if parse_argv(&["--sw-rays", "--no-sw-rays"]).opts.sw_rays {
        return Err("--sw-rays --no-sw-rays must disable".into());
    }
    let continuation = parse_argv(&["--no-cut-rays", "--continuation-rays"]).opts;
    if !continuation.sw_rays || !continuation.cut_rays {
        return Err("--continuation-rays must arm software rays and frontier consumption".into());
    }
    if parse_argv(&["--continuation-rays", "--no-cut-rays"])
        .opts
        .cut_rays
    {
        return Err("--no-cut-rays must remain the continuation-vs-root control".into());
    }
    if parse_argv(&["--fg", "--no-fg"]).opts.fg {
        return Err("--fg --no-fg must disable frame generation".into());
    }
    if !parse_argv(&["--no-fg", "--fg"]).opts.fg_explicit {
        return Err("a trailing --fg must set fg_explicit".into());
    }
    if parse_argv(&[]).opts.dxr_inline_explicit {
        return Err("--dxr-inline was not passed; explicit must stay false".into());
    }
    // The load-bearing veto pin: `--dxr-inline 1` is a real CHOICE even
    // though 1 is also the compiled default — presence, not value, is the
    // signal (the spin_frames doctrine), and it is what lets an Intel user
    // pin the cross-vendor mode against the vendor default.
    let di = parse_argv(&["--dxr-inline", "1"]).opts;
    if di.dxr_inline != 1 || !di.dxr_inline_explicit {
        return Err("--dxr-inline 1 must set dxr_inline_explicit (the vendor-default veto)".into());
    }
    if parse_argv(&[]).opts.emissive_lights_explicit {
        return Err("--emissive-lights was not passed; explicit must stay false".into());
    }
    // The upscaler-default veto pin: `--no-emissive-lights` is a real CHOICE
    // even though OFF is also the compiled default — presence, not value, is
    // the signal (the dxr_inline doctrine), and it is what lets a XeSS/FSR3
    // user pin OFF against main::upscaler_defaults.
    let el = parse_argv(&["--no-emissive-lights"]).opts;
    if el.emissive_lights || !el.emissive_lights_explicit {
        return Err(
            "--no-emissive-lights must disarm AND set emissive_lights_explicit (the upscaler-default veto)".into(),
        );
    }
    if !parse_argv(&["--emissive-lights"]).opts.emissive_lights_explicit {
        return Err("--emissive-lights must set emissive_lights_explicit".into());
    }
    if parse_argv(&["--world", "--no-world"]).world_flag != Some(false) {
        return Err("--world --no-world must resolve to Some(false)".into());
    }

    // ---- 3. the swapchain three-way ---------------------------------------
    // 8-bit SDR | Sdr10 | HDR10 (one 10-bit format, two curves) spelled as
    // two toggles: each arm writes ALL the fields, which is the only way
    // later-flags-win holds ACROSS the pair. CLAUDE.md states both of these
    // outcomes explicitly.
    let pq = parse_argv(&["--no-hdr", "--hdr10"]).opts;
    if !(pq.hdr && pq.hdr10) {
        return Err("--no-hdr --hdr10 must select the PQ swapchain".into());
    }
    let sdr = parse_argv(&["--hdr10", "--no-hdr"]).opts;
    if sdr.hdr || sdr.hdr10 || sdr.sdr10 {
        return Err("--hdr10 --no-hdr must select the 8-bit SDR swapchain".into());
    }
    // Sdr10 must stay REACHABLE on an HDR-on box (where PQ is the default) —
    // `--no-hdr10` is its spelling, and the three-way stays a three-way.
    // Without this the gamma-through-10-bit arm the gates rest on
    // (tone::self_test's Sdr10 range pin, M12's sdr10 wire comparison) would
    // be un-selectable from the command line there.
    let wide = parse_argv(&["--no-hdr10"]).opts;
    if !(wide.hdr && !wide.hdr10 && wide.sdr10) {
        return Err("--no-hdr10 must select Sdr10 (10-bit gamma, not PQ)".into());
    }
    // ...and each of the three must still win from any predecessor.
    let back_to_pq = parse_argv(&["--no-hdr10", "--hdr10"]).opts;
    if !(back_to_pq.hdr10 && !back_to_pq.sdr10) {
        return Err("--no-hdr10 --hdr10 must select PQ".into());
    }
    let auto = parse_argv(&["--hdr10", "--hdr"]).opts;
    if !(auto.hdr && !auto.hdr10 && !auto.sdr10) {
        return Err("--hdr10 --hdr must return to the display-probed default".into());
    }

    // ---- 4. the precedence seam -------------------------------------------
    // `base` is what a settings file already wrote. A flag must overwrite it;
    // silence must leave it standing. (The file -> Opts half is
    // settings::self_test's; this pins the half that lives here.)
    let mut base = defaults();
    base.aniso = 2;
    base.bloom = false;
    let quiet = parse_from(base.clone(), std::iter::empty());
    if quiet.opts.aniso != 2 || quiet.opts.bloom {
        return Err("an empty argv must leave the settings-seeded fields standing".into());
    }
    let loud = parse_from(base, ["--aniso".to_string(), "8".to_string()].into_iter());
    if loud.opts.aniso != 8 {
        return Err("a CLI flag must overwrite the settings-seeded field".into());
    }
    if loud.opts.bloom {
        return Err("a CLI flag must not disturb settings fields it does not name".into());
    }

    // ---- 5. the selectors reach Cli ---------------------------------------
    let c = parse_argv(&[
        "model.obj",
        "--tile",
        "4x2",
        "--spp",
        "4",
        "--spin",
        "path",
        "--spin-plain",
        "--spin-warmup",
        "1600",
    ]);
    if c.obj.as_deref() != Some("model.obj") {
        return Err("the positional scene argument did not reach Cli::obj".into());
    }
    if c.tile != Some((4, 2)) {
        return Err("--tile 4x2 did not reach Cli::tile".into());
    }
    if c.opts.spp != 4 {
        return Err("--spp 4 did not reach Opts::spp".into());
    }
    if c.spin.as_deref() != Some("path") {
        return Err("--spin path did not reach Cli::spin".into());
    }
    if c.spin_hybrid || c.spin_warmup != Some(1600) {
        return Err("--spin-plain/--spin-warmup did not reach Cli".into());
    }
    // A typed frame count is obeyed verbatim; a defaulted one the runner may
    // extend to cover a whole lap past a vendor-derived warm-up. Conflating
    // the two would let the extension rewrite a count somebody set up an A/B
    // around, so the flag's presence — not its value — is the signal.
    if c.spin_frames_explicit {
        return Err("--spin-frames was not passed; explicit must stay false".into());
    }
    let sf = parse_argv(&["--spin", "path", "--spin-frames", "2200"]);
    if sf.spin_frames != 2200 || !sf.spin_frames_explicit {
        return Err("--spin-frames 2200 did not reach Cli as an explicit count".into());
    }
    if !parse_argv(&["--spin-plain", "--spin-hybrid"]).spin_hybrid {
        return Err("--spin-plain --spin-hybrid must select hybrid (later flags win)".into());
    }
    if !parse_argv(&["--stress", "5000"]).stress.is_some_and(|n| n == 5000) {
        return Err("--stress 5000 did not reach Cli::stress".into());
    }
    // --blas-split's optional value: a numeric token is consumed, a scene path
    // is NOT (that arm is why the parser is Peekable at all).
    if parse_argv(&["--blas-split", "64"]).opts.blas_split != Some(64) {
        return Err("--blas-split 64 must take the explicit cap".into());
    }
    let bare = parse_argv(&["--blas-split", "model.obj"]);
    if bare.opts.blas_split != Some(blas_split::DEFAULT_MAX_PRIMS)
        || bare.obj.as_deref() != Some("model.obj")
    {
        return Err("--blas-split must not swallow a following scene path".into());
    }
    // Same optional-value contract for --waveviz: `chs` is consumed on exact
    // match only, a scene path is not.
    let wv = parse_argv(&["--waveviz", "chs"]);
    if wv.opts.waveviz != 2 {
        return Err("--waveviz chs must take the closest-hit mode".into());
    }
    let wv_bare = parse_argv(&["--waveviz", "model.obj"]);
    if wv_bare.opts.waveviz != 1 || wv_bare.obj.as_deref() != Some("model.obj") {
        return Err("--waveviz (bare) must arm without swallowing a scene path".into());
    }
    // Same optional-value contract for the emissive arming flag: bare arms
    // at the default budget without eating a scene path.
    let el_bare = parse_argv(&["--emissive-lights", "model.obj"]);
    if !el_bare.opts.emissive_lights
        || el_bare.opts.emissive_lights_count != emissive::EL_DEFAULT
        || el_bare.obj.as_deref() != Some("model.obj")
    {
        return Err("--emissive-lights (bare) must arm at the default budget without swallowing a scene path".into());
    }

    // ---- 6. --help stops the parse ----------------------------------------
    // It sets a flag rather than printing, so a throwaway parse stays silent;
    // and it BREAKS, which is what reproduces the old `return` out of main.
    let h = parse_argv(&["--help", "--no-bloom"]);
    if !h.helped {
        return Err("--help must set Cli::helped".into());
    }
    if !h.opts.bloom {
        return Err("--help must stop the parse (flags after it are not applied)".into());
    }

    // ---- 7. notes are collected, never printed ----------------------------
    let before = lever_snapshot();
    let clamped = parse_argv(&["--fireflies", "999"]);
    if clamped.notes.len() != 1 || !clamped.notes[0].contains("clamped") {
        return Err("an over-cap --fireflies must leave exactly one note".into());
    }
    if clamped.opts.fireflies_count != 999 {
        return Err("the parser must carry the RAW count — set_count owns the clamp".into());
    }
    if lever_snapshot() != before {
        return Err("the --fireflies note path touched a process global".into());
    }

    Ok(())
}
