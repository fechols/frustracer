// BC7 block compression of OPAQUE scene textures (--bc7) — GPU upload only;
// the CPU renderer keeps sampling the exact RGBA8 texels. Alpha-masked cutout
// textures are deliberately excluded (see the module note).
mod bc7;
mod bvh;
mod camera;
mod dlss;
// Signal split, wire encoders, and demodulation/composite math for FSR Ray
// Regeneration — pure CPU, feeds --check-fsr; the GPU seam is gpu/ffx*.
mod builders;
mod fsr;
mod frustum;
mod ftree;
mod gltf_loader;
mod hemi;
// The presentation stack (D3D12 + Streamline) is Windows-only; everything
// headless (--check, --check-dlss) stays cross-platform.
#[cfg(windows)]
mod gpu;
#[cfg(windows)]
mod input;
// 500 Hz wall-clock input integrator thread (keyboard/mouse/XInput -> the
// shared camera); Win32-only by nature, like the window it serves.
#[cfg(windows)]
mod flycam;
mod matclass;
// OIDN loads its DLLs through the Win32 loader; the denoiser itself is
// CPU/GPU-agnostic but the SDK drop and load path here are Windows-only.
#[cfg(windows)]
mod oidn;
// The ORT loader half is Windows-only (LoadLibrary + DirectML); the padding,
// NCHW packing, and temporal-warp math are pure and feed --check.
mod nppd;
mod overlay;
mod render;
mod reproject;
mod scene;
mod scene_cache;
// Glare: the optics between the scene and the sensor. A display-stage pass, so
// it never touches accum, the temporal cache, or any upscaler guide.
mod bloom;
mod shade;
// Order-2 SH sky irradiance — the smooth half of the one-sky model (the sharp
// half, the sun disc, is an explicit light: SH cannot be shadow-rayed).
mod sh;
// The one sky: a scattering dome + a sun disc at infinity. Read its header —
// the "the disc appears exactly once per light path" invariant governs every
// sky call site in the renderer.
mod sky;
mod sphcell;
mod stats;
mod prof;
mod replay;
mod temporal;
mod texture;
// Pure chain-resolution data for the always-on temporal-upscaler fallback
// (DLSS-RR → FSR4-RR → XeSS → FSR3); the real probes live in GpuContext::new.
mod upchain;
// The loader half is Windows-only (LoadLibrary); the FFI structs, depth
// encoding, and the dynamic-res controller are pure and feed --check-xess.
mod xess;

use camera::Camera;
use glam::Vec3A;
use rayon::prelude::*;
use render::FrameCtx;
use shade::Quality;
use stats::Stats;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering::Relaxed};
use std::time::{Duration, Instant};

const W: usize = 1920;
const H: usize = 1080;
const MAX_SAMPLES: u32 = 1024;
/// Frame budget for dynamic-resolution mode: 60 FPS minus resolve/present
/// headroom. Not a per-tile deadline: a log4-proportional controller turns the
/// previous frame's time against this target into a uniform quadtree depth cap
/// for the next frame; tiles reaching the cap unresolved are sparse-filled
/// (render::sparse_fill). Cost roughly quadruples per level, so
/// log4(budget/elapsed) reads "levels of headroom" directly.
const RENDER_BUDGET: Duration = Duration::from_millis(15);
/// Controller gain on the log4 error.
const DEPTH_GAIN: f32 = 0.6;
/// Max upward step per frame — creep up (>= 3 frames per level)...
const STEP_UP_MAX: f32 = 0.4;
/// ...but drop more than a full level in one step after a blown frame.
const STEP_DOWN_MAX: f32 = 1.5;

/// CLI options beyond the OBJ path / --check.
pub struct Opts {
    /// The temporal-upscaler fallback chain: which levels of
    /// DLSS-RR → FSR4-RR → XeSS → FSR3 may be probed (first supported level
    /// wins; upscaling is always on unless --no-upscale empties the chain).
    /// `--<x>` force-starts the chain at that level, `--no-<x>` skips it.
    pub chain: upchain::UpChain,
    /// D3D12 debug layer + Streamline verbose logging.
    pub gpu_debug: bool,
    /// Block-compress the OPAQUE scene textures to BC7 at load (8 bpp vs 32),
    /// GPU upload only — the CPU renderer keeps sampling the exact RGBA8
    /// texels, so this moves the GPU-vs-CPU statistical gates (albedo A/B,
    /// T2 radiance) and nothing else. Alpha-masked cutout textures are never
    /// compressed (src/bc7.rs). Off by default: an A/B lever, and the encode
    /// is paid on every load (there is deliberately no BC7 disk cache).
    pub bc7: Option<bc7::Quality>,
    /// Directory holding sl.interposer.dll + plugins (M3+).
    pub sl_path: String,
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
    /// Directory holding libxess.dll.
    pub xess_path: String,
    /// Directory holding amd_fidelityfx_loader_dx12.dll + the provider DLLs.
    pub ffx_path: String,
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
    /// window (default quality 2/3; `--lock-res dynamic` -> None = the
    /// step-wise dynamic-resolution controller). CLI-only, no runtime
    /// toggle — T prints the locked note.
    pub lock_scale: Option<f32>,
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
    /// GPU-resident tracing (--gpu): the whole quadtree + shading runs in
    /// D3D12 compute with DXR RayQuery rays. Requires the DXC DLL drop and
    /// RT tier 1.1; falls back to the CPU path with a loud line otherwise.
    pub gpu: bool,
    /// The DXR DispatchRays pipeline as the session's render mode (default
    /// ON — the F key toggles it live against the CPU tracer; --cpu opts
    /// back into the CPU renderer). Requires the DXC DLL drop and RT tier
    /// 1.0; falls back to the CPU path with a loud line.
    pub dxr: bool,
    /// Directory holding dxcompiler.dll + dxil.dll.
    pub dxc_path: String,
    /// PIX Begin/End events on the D3D12 command lists (--pix-markers;
    /// default off so unprofiled sessions stay byte-identical). Needs
    /// WinPixEventRuntime.dll under --pix-path; missing DLL = loud line,
    /// markers stay off.
    pub pix_markers: bool,
    /// Directory holding WinPixEventRuntime.dll.
    pub pix_path: String,
    /// V-sync'd presentation (default on). `--no-vsync` presents at sync
    /// interval 0 on a tearing swapchain so interactive frame times measure
    /// the renderer, not the monitor refresh.
    pub vsync: bool,
}

/// OIDN placement in XeSS mode (N cycles off → pre → post). Pre denoises the
/// 1-spp frame at the dynamic render res before upscaling — the recommended
/// default (guides match, subpixel jitter detail preserved, cheaper). Post
/// denoises the upscaled window-res frame — the A/B experiment; costs a
/// synchronous GPU readback plus a window-res denoise every frame and
/// presents through the CPU blit path.
#[derive(Clone, Copy, PartialEq)]
enum XessOidn {
    Off,
    Pre,
    Post,
}

/// The presentation/denoise mode a frame renders for, resolved ONCE per frame
/// from the toggle flags (precedence dlss > xess > oidn > plain). Every
/// mode-dependent FrameCtx field reads this single value — the per-field
/// if/else chains it replaced each re-encoded the precedence by ordering and
/// could silently disagree when a toggle handler missed a reset.
#[derive(Clone, Copy, PartialEq)]
enum RenderMode {
    Dlss,
    Xess,
    /// FSR Ray Regeneration + FSR4: the XeSS frame contract (fresh jittered
    /// 1-spp full-depth traces at a dynamic render res, no CPU accumulation,
    /// never idle) with Ray Regeneration denoising the two direct-light
    /// signals on the GPU and FSR4 upscaling the remodulated composite.
    Fsr,
    /// Plain-mode OIDN; `temporal` is the reprojected-history sub-mode
    /// (fresh 1-spp frames on a free-running rng index). Both sub-modes
    /// fill the window-res OIDN G-buffers.
    Oidn { temporal: bool },
    /// NPPD neural denoising: fresh 1-spp full-res frames on a free-running
    /// rng index; the network's own recurrent state (warped in Rust) is the
    /// sole temporal integrator — no CPU accumulation, no History, no budget
    /// frames. Fills its own window-res G-buffers with `prev_cam` set (the
    /// state warp consumes real motion vectors).
    Nppd,
    Plain,
}

fn main() {
    prof::init(); // before any zone can fire; inert without --features tracy
    let mut obj: Option<String> = None;
    let mut check = false;
    let mut opts = Opts {
        chain: upchain::UpChain::ALL,
        gpu_debug: false,
        bc7: None,
        sl_path: std::env::var("FRUSTRACER_SL_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\streamline-sdk\bin\x64").to_string()
        }),
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
        prefer: None,
        oidn_post: false,
        xess_autoexposure: false,
        adaptive: true,
        lock_scale: xess::lock_scale("quality"),
        temporal: true,
        replay: true,
        adopt: true,
        discard_seeds: false,
        defer_shade: false,
        hemi_share: true,
        cut_rays: true,
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
        ftree: true,
        ftree_tiles: false,
        gpu: false,
        dxr: true,
        dxc_path: std::env::var("FRUSTRACER_DXC_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\dxc\bin\x64").to_string()
        }),
        pix_markers: false,
        pix_path: std::env::var("FRUSTRACER_PIX_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\pix\bin\x64").to_string()
        }),
        vsync: true,
    };
    let mut check_dlss = false;
    let mut dlss_dump = false;
    let mut check_oidn = false;
    let mut oidn_dump = false;
    let mut check_xess = false;
    let mut xess_dump = false;
    let mut check_fsr = false;
    let mut check_nppd = false;
    let mut nppd_dump = false;
    let mut no_xess_explicit = false;
    // --fsr/--fsr3 passed: flips the default adapter preference to AMD.
    let mut fsr_forced = false;
    // --dxr passed explicitly (it is the default): gates the --gpu notice.
    let mut dxr_explicit = false;
    // --no-upscale passed: distinguishes the quiet explicit plain path from
    // the loud chain-exhausted fallback.
    let mut no_upscale = false;
    let mut check_gpu = false;
    let mut check_dxr = false;
    let mut stress: Option<usize> = None;
    let mut tile: Option<(u32, u32)> = None;
    let mut cam_override: Option<Camera> = None;
    let mut spin: Option<String> = None;
    let mut spin_frames = 2000u32;
    let mut args = std::env::args().skip(1);
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
            "--oidn-no-temporal" => opts.oidn_temporal = false,
            "--check-nppd" => check_nppd = true,
            "--nppd-dump" => {
                check_nppd = true;
                nppd_dump = true;
            }
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
            "--fsr3" => {
                opts.chain.force(upchain::UpLevel::Fsr3);
                fsr_forced = true;
            }
            "--no-fsr" => {
                opts.chain.skip(upchain::UpLevel::Fsr4);
                opts.chain.skip(upchain::UpLevel::Fsr3);
            }
            "--ffx-path" => {
                opts.ffx_path = args.next().unwrap_or_else(|| {
                    eprintln!("--ffx-path needs a directory argument");
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
            // Before scene load by construction (the --bvh-ctrav knob
            // pattern): no chains are built, every trilinear sample falls
            // back to mip-0 bilinear — the pre-mip renderer exactly.
            "--no-mips" => texture::set_mips(false),
            // A/B lever: no glare. A display-stage pass, so this is a pure
            // presentation change — accum, the temporal cache, every upscaler
            // guide and every radiance gate are untouched either way, and the
            // off path keeps the original alloc-free tonemap loop verbatim.
            "--no-bloom" => bloom::set_enabled(false),
            // Same "knob before scene load" pattern: the GPU reads it for the
            // static sampler's MaxAnisotropy, the CPU for Cone::aniso. 1 = off
            // ⇒ the isotropic ray-cone lod path runs verbatim (bit-identical
            // to the pre-aniso renderer). --no-mips forces it to 1 — mips are
            // the prerequisite (texture::set_aniso).
            "--no-aniso" => texture::set_aniso(1),
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
                texture::set_aniso(n);
            }
            "--no-hemi-share" => opts.hemi_share = false,
            "--no-cut-rays" => opts.cut_rays = false,
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
            "--ftree" => opts.ftree = true,
            "--no-ftree" => opts.ftree = false,
            "--ftree-tiles" => opts.ftree_tiles = true,
            "--no-ftree-tiles" => opts.ftree_tiles = false,
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
            }
            "--no-vsync" => opts.vsync = false,
            "--vsync" => opts.vsync = true,
            "--pix-markers" => opts.pix_markers = true,
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
                }
            }
            "--oidn-no-clean-aux" => opts.oidn_clean_aux = false,
            "--gpu" => opts.gpu = true,
            "--check-gpu" => check_gpu = true,
            "--dxr" => {
                opts.dxr = true;
                dxr_explicit = true;
            }
            // The CPU frustum-tracer as the render mode: it is what both GPU
            // modes are peers of, so --cpu clears both. Later flags win (a
            // trailing --gpu/--dxr re-selects), matching the chain's algebra.
            "--cpu" => {
                opts.dxr = false;
                opts.gpu = false;
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
            // --bc7 alone = the `fast` profile; --bc7-quality overrides (and
            // implies --bc7, so the order of the two flags doesn't matter).
            "--bc7" => opts.bc7 = opts.bc7.or(Some(bc7::Quality::Fast)),
            "--bc7-quality" => {
                let q = args.next().unwrap_or_else(|| {
                    eprintln!("--bc7-quality needs ultrafast|fast|basic|slow");
                    std::process::exit(2);
                });
                opts.bc7 = Some(bc7::Quality::parse(&q).unwrap_or_else(|| {
                    eprintln!("--bc7-quality: unknown profile '{q}' (ultrafast|fast|basic|slow)");
                    std::process::exit(2);
                }));
            }
            "--sl-path" => {
                opts.sl_path = args.next().unwrap_or_else(|| {
                    eprintln!("--sl-path needs a directory argument");
                    std::process::exit(2);
                })
            }
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
            "--help" | "-h" => {
                eprintln!("usage: frustracer [model.obj|.gltf|.glb] [--stress <n>] [--check] [--check-dlss] [--dlss-dump] [--no-dlss] [--check-oidn] [--oidn-dump] [--oidn] [--check-xess] [--xess-dump] [--xess] [--lock-res <r>] [--gpu-debug] [--sl-path <dir>] [--oidn-path <dir>] [--oidn-device <d>] [--xess-path <dir>]");
                eprintln!("  --stress <n>  procedural stress field of n objects (perf test; composes with --check)");
                eprintln!("  --tile <n|nxm>  replicate a loaded OBJ scene into an n×n (or n×m) grid of copies —");
                eprintln!("                flattened geometry, shared materials/textures (composes with --check)");
                eprintln!("  --cam <e,t>   start camera: ex,ey,ez,tx,ty,tz (reproducible benchmark viewpoints)");
                eprintln!("  --check       headless: verify hybrid vs reference, benchmark, write check.png");
                eprintln!("  --check-dlss  headless: G-buffer MV/depth/matrix self-test (no GPU needed)");
                eprintln!("  --dlss-dump   --check-dlss plus G-buffer PNG dumps (albedo/spec_albedo/normal/misc/mv)");
                eprintln!("  --no-dlss     skip the DLSS-RR level of the upscaler chain (falls to FSR4/XeSS/FSR3);");
                eprintln!("                the chain DLSS-RR -> FSR4-RR -> XeSS -> FSR3 is always on — the first");
                eprintln!("                supported level wins; --<x> force-starts the chain at that level");
                eprintln!("  --no-upscale  plain presentation: no temporal upscaler at all (benchmark escape)");
                eprintln!("  --bc7         block-compress the OPAQUE scene textures to BC7 at load (8 bpp vs 32;");
                eprintln!("                GPU upload only — alpha-masked cutout textures stay exact RGBA8).");
                eprintln!("                No disk cache: the encode runs every load. --bc7-quality <p> sets the");
                eprintln!("                ISPC profile: ultrafast|fast|basic|slow (default fast)");
                eprintln!("  --check-oidn  headless: OIDN denoise self-test (needs the OIDN DLLs)");
                eprintln!("  --oidn-dump   --check-oidn plus before/after/G-buffer PNG dumps");
                eprintln!("  --oidn        start with OIDN denoising on (N toggles; implies DLSS off)");
                eprintln!("  --oidn-no-temporal  start OIDN without the temporal reprojection history (M toggles)");
                eprintln!("  --oidn-path   OIDN DLL directory (default: SDKs\\oidn.x64.windows\\bin)");
                eprintln!("  --oidn-device OIDN device: default|cpu|sycl|cuda|hip");
                eprintln!("  --oidn-quality OIDN RT-filter quality: fast|balanced|high (default balanced)");
                eprintln!("  --oidn-no-clean-aux  don't declare the albedo/normal guides noise-free (A/B lever)");
                eprintln!("  --nppd        NPPD neural denoising (J toggles; needs onnxruntime.dll + an exported");
                eprintln!("                model — see tools/nppd-export). Implies --xess: NPPD denoises at the");
                eprintln!("                render res before the upscale; --no-xess keeps the standalone window-res mode");
                eprintln!("  --check-nppd  headless: NPPD denoise self-test (needs onnxruntime.dll + the model)");
                eprintln!("  --nppd-dump   --check-nppd plus before/after PNG dumps");
                eprintln!("  --nppd-path   ONNX Runtime DLL directory (default: SDKs\\onnxruntime\\bin)");
                eprintln!("  --nppd-model  exported NPPD .onnx (default: SDKs\\nppd\\nppd_small.onnx)");
                eprintln!("  --nppd-device NPPD execution provider: auto|cpu|dml|dml:<n> (default auto = DML then CPU)");
                eprintln!("  --check-xess  headless: XeSS dynamic-res contract self-test (no GPU or DLL needed)");
                eprintln!("  --xess-dump   --check-xess plus G-buffer PNG dumps");
                eprintln!("  --check-fsr   headless: FSR signal-split/encoding/MV contract self-test (no GPU or DLL)");
                eprintln!("  --fsr         force-start the upscaler chain at FSR4 + Ray Regeneration (K toggles;");
                eprintln!("                RDNA4 only — elsewhere the chain falls to XeSS then FSR 3.1)");
                eprintln!("  --fsr3        force-start the chain at the FSR 3.1 upscale-only level (A/B lever)");
                eprintln!("  --ffx-path    FidelityFX DLL directory (default: the FidelityFX-Samples-prebuilt drop)");
                eprintln!("  --gpu         GPU-resident tracing: quadtree + shading in D3D12 compute with DXR");
                eprintln!("                RayQuery rays (needs the DXC DLLs and RT tier 1.1; falls back to CPU)");
                eprintln!("  --check-gpu   headless: GPU tracer gate suite (needs a D3D12 GPU + the DXC DLLs)");
                eprintln!("  --dxr         the DXR DispatchRays pipeline — the DEFAULT render mode (F toggles it");
                eprintln!("                against the CPU tracer live; needs the DXC DLLs and RT tier 1.0;");
                eprintln!("                falls back to CPU with a loud line)");
                eprintln!("  --cpu         the CPU frustum-tracer as the render mode (opts out of --dxr/--gpu)");
                eprintln!("  --check-dxr   headless: DXR pipeline gate suite (needs a D3D12 RT GPU + the DXC DLLs)");
                eprintln!("  --dxc-path    DXC DLL directory (default: SDKs\\dxc\\bin\\x64)");
                eprintln!("  --xess        force-start the upscaler chain at XeSS-SR (X toggles;");
                eprintln!("                N cycles the OIDN denoise: off -> pre-upscale -> post-upscale)");
                eprintln!("  --oidn-post   start XeSS mode with OIDN placed AFTER the upscale (requires --xess)");
                eprintln!("  --xess-autoexposure  let XeSS compute exposure internally (A/B lever)");
                eprintln!("  --no-adaptive disable the adaptive shading rate in XeSS mode (uniform per-pixel shading;");
                eprintln!("                visibility is per-pixel either way)");
                eprintln!("  --no-hemi-share  disable the shared hemisphere capture in fb (H) frames — every");
                eprintln!("                shading point runs its own bounce tree (A/B lever)");
                eprintln!("  --no-mips     no texture mip chains; every trilinear sample degenerates to the");
                eprintln!("                pre-mip bilinear (A/B lever; mips are on by default — implies --no-aniso)");
                eprintln!("  --aniso N     max anisotropy for texture filtering, 1..=16 (default 16; 1 = off, i.e.");
                eprintln!("                the isotropic ray-cone trilinear path verbatim). --no-aniso = --aniso 1");
                eprintln!("  --lock-res    DLSS/XeSS render res: quality|balanced|performance|ultra-performance|native,");
                eprintln!("                a ratio in (0, 1], or dynamic (the step-wise DRS controller); default quality (2/3)");
                eprintln!("  --xess-path   XeSS DLL directory (default: SDKs\\XeSS-SDK\\bin)");
                eprintln!("  --prefer-nvidia | --prefer-intel | --prefer-amd");
                eprintln!("                pick that vendor's adapter for the D3D12 device (default NVIDIA, or AMD");
                eprintln!("                with --fsr; a preference, not a requirement — features the picked GPU");
                eprintln!("                can't support fall back with a log line)");
                eprintln!("  --gpu-debug   D3D12 debug layer + verbose Streamline logging");
                eprintln!("  --sl-path     Streamline DLL directory (default: SDKs\\streamline-sdk\\bin\\x64)");
                return;
            }
            _ => obj = Some(a),
        }
    }

    if obj.is_some() && stress.is_some() {
        eprintln!("--stress and an OBJ path are mutually exclusive — pick one scene source");
        std::process::exit(2);
    }
    if tile.is_some() && obj.is_none() {
        eprintln!("--tile needs a loaded model to replicate — pass an OBJ path");
        std::process::exit(2);
    }
    if opts.nppd && !no_xess_explicit && !no_upscale && (opts.chain.dlss || opts.chain.fsr4) {
        // The default NPPD experience is the XeSS composition: trace at the
        // --lock-res scale (default quality 2/3), NPPD denoises at that
        // render res, XeSS upscales to the window. Standalone window-res
        // NPPD remains the automatic fallback when the XeSS DLL is missing,
        // or explicitly via --nppd --no-xess.
        opts.chain.force(upchain::UpLevel::Xess);
        eprintln!("nppd: --nppd starts the upscaler chain at XeSS (pre-upscale denoise at the render res; --no-xess opts out)");
    }
    if fsr_forced && (opts.nppd || opts.oidn || opts.oidn_post) {
        eprintln!("fsr: the FSR present arm owns the frame; ignoring OIDN/NPPD flags");
        opts.nppd = false;
        opts.oidn = false;
        opts.oidn_post = false;
    }
    // Adapter preference default: AMD when FSR was explicitly requested
    // (--fsr/--fsr3), NVIDIA otherwise. An explicit --prefer-* always wins;
    // the chain then probes whatever adapter got picked.
    if opts.prefer.is_none() && fsr_forced {
        opts.prefer = Some(gpu::adapter::Prefer::Amd);
    }
    if no_upscale {
        eprintln!("upscale: OFF (--no-upscale; plain presentation)");
    }
    if opts.gpu {
        // DLSS-RR and XeSS compose with --gpu (the tracer feeds them GPU-born
        // G-buffers; SL's proxy queue carries the tracer's workload). OIDN
        // stays a CPU-renderer feature.
        if opts.oidn || opts.oidn_post {
            eprintln!("gpu: OIDN is unavailable with --gpu; ignoring");
        }
        opts.oidn = false;
        opts.oidn_post = false;
        if opts.nppd && !opts.chain.xess {
            // The GPU NPPD stage rides the XeSS composition only (RR is
            // itself a denoiser — the same exclusion as the CPU paths).
            // Phase-C's implication normally forces XeSS; only an explicit
            // --no-xess lands here.
            eprintln!("gpu: NPPD under --gpu needs the XeSS composition; ignoring");
            opts.nppd = false;
        }
        if opts.dxr {
            // The DXR pipeline is a CPU-session peer mode; --gpu is its own
            // self-contained session. --dxr is the default, so only an
            // explicit request earns the notice.
            if dxr_explicit {
                eprintln!("gpu: --dxr is a CPU-session mode, unavailable under --gpu; ignoring");
            }
            opts.dxr = false;
        }
    }
    if opts.dxr && !dxr_explicit && !opts.gpu && (opts.oidn || opts.oidn_post || opts.nppd) {
        // OIDN and (CPU-side) NPPD denoise frames the CPU renderer produces —
        // they cannot run under the DXR arm, which would silently switch them
        // back off at init. Asking for one opts out of the DXR default; an
        // explicit --dxr still wins (and F toggles either way live).
        eprintln!("dxr: OIDN/NPPD are CPU-renderer denoisers — staying on the CPU tracer (--dxr forces the pipeline; F toggles live)");
        opts.dxr = false;
    }

    // PIX command-list markers: opt-in, runtime-loaded; inert otherwise.
    gpu::pix::init(&opts.pix_path, opts.pix_markers);

    // A/B lever: --no-cut-rays sends cut-seeded rays down the root traversal
    // instead. Every ray consumer funnels through intersect_multi/occluded_multi,
    // so setting the flag once here covers primary, hemi and shaft at a stroke.
    if !opts.cut_rays {
        bvh::CUT_SEED_RAYS.store(false, std::sync::atomic::Ordering::Relaxed);
        eprintln!("bvh: --no-cut-rays — cut-seeded rays traverse from the root (inherited t_start unchanged)");
    }
    // Hemi leaf rays go root-first by default (the M1/M2 measurement); the
    // static's own default already matches, so only the opt-in stores.
    if opts.cut_hemi {
        bvh::CUT_SEED_HEMI.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!("bvh: --cut-hemi — hemi leaf rays seed from their bounce cut (the pre-M2 behavior)");
    }
    // Must land before ANY Bvh::build (the loader's cold-miss build included)
    // and before the scene cache is probed — build_key() is part of its key.
    bvh::set_c_trav(opts.c_trav);
    bvh::set_max_leaf(opts.max_leaf);
    bvh::set_split_axes(opts.split_axes);
    if bvh::set_builder(&opts.bvh_builder).is_none() {
        eprintln!("--bvh-builder: unknown builder '{}' (sah | lbvh | ploc | som)", opts.bvh_builder);
        std::process::exit(2);
    }
    if !opts.ftree {
        ftree::FTREE_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
        eprintln!("ftree: --no-ftree — all bound queries stay on the binary BVH");
    }
    // Tile recursion on the wide tree is opt-in (default off — the static's
    // own default already matches, so only the opt-in stores).
    if opts.ftree_tiles {
        ftree::FTREE_TILES.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!("ftree: --ftree-tiles — the tile recursion runs on the 8-wide frustum tree");
    }

    eprintln!("frustracer — loading scene...");
    let mut tile_fh: Option<f32> = None;
    // A cache hit hands back the untiled scene's BVH too; tiling replicates
    // the geometry, so the tiled BVH is always a fresh (parallel) build.
    let mut prebuilt: Option<bvh::Bvh> = None;
    let scene = match (&obj, stress) {
        (Some(p), _) => {
            let resolved = scene::resolve_scene_path(p);
            // Extension sniff: .gltf/.glb take the glTF loader, everything
            // else the OBJ path — reusing the positional arg inherits the
            // exclusivity checks, default camera, structural-skip predicate,
            // and --tile. glTF scenes SKIP the sidecar cache: its texture
            // table stores re-decodable file paths, and glTF images live
            // inside GLB buffer views / data URIs (a cache hit would come
            // back with 1x1 white textures).
            let lower = resolved.to_ascii_lowercase();
            let is_gltf = lower.ends_with(".gltf") || lower.ends_with(".glb");
            let cached = if is_gltf { None } else { scene_cache::try_load(&resolved) };
            let (s, b) = match cached {
                Some((s, b)) => (s, Some(b)),
                None => {
                    let s = if is_gltf {
                        gltf_loader::load_gltf_scene(&resolved)
                    } else {
                        scene::load_obj_scene(&resolved)
                    };
                    // Under --tile the untiled build's ONLY use is feeding
                    // the cache store (the tiled BVH is always a fresh
                    // build) — and glTF skips the cache, so a tiled glTF
                    // run skips this build instead of discarding it.
                    let b = if is_gltf && tile.is_some() {
                        None
                    } else {
                        let b = bvh::Bvh::build(&s);
                        if !is_gltf {
                            scene_cache::store(&resolved, &s, &b);
                        }
                        Some(b)
                    };
                    (s, b)
                }
            };
            match tile {
                Some((tx, tz)) => {
                    let (s, fh) = scene::tile_scene(s, tx, tz);
                    tile_fh = Some(fh);
                    s
                }
                None => {
                    // b is always Some here: the None arm above requires
                    // tile.is_some().
                    prebuilt = b;
                    s
                }
            }
        }
        (None, Some(n)) => scene::stress_scene(n),
        (None, None) => scene::procedural_scene(),
    };
    // The stress field (and a tiled OBJ field) keeps the default look
    // direction but pulls the camera back/up to overlook the field; /8 (not
    // the field half-extent itself) trades the nearest rows off the bottom of
    // the frame for less sky.
    let cam0 = cam_override.unwrap_or_else(|| match (stress, tile_fh) {
        (Some(n), _) => scaled_camera((scene::stress_field_half(n) / 8.0).max(1.0)),
        (None, Some(fh)) => scaled_camera((fh / 8.0).max(1.0)),
        (None, None) => default_camera(),
    });
    let t0 = Instant::now();
    let bvh = prebuilt.unwrap_or_else(|| bvh::Bvh::build(&scene));
    eprintln!(
        "scene: {} tris | BVH: {} nodes ready in {:.0} ms",
        scene.tri_count(),
        bvh.nodes.len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    // SAH is always scored at the fixed reference C_trav so trees built at
    // different C_trav stay comparable on one scale.
    eprintln!(
        "bvh quality (builder {} c_trav {} axes {} maxleaf {}): {}",
        opts.bvh_builder,
        opts.c_trav,
        opts.split_axes,
        opts.max_leaf,
        bvh.quality(bvh::SAH_REF_C_TRAV).line(scene.tri_count())
    );
    if check {
        // Determinism gate for the two-phase parallel build: a rebuild must
        // be byte-identical (node order pinned across runs and thread
        // counts — the property the scene cache and cross-run benchmarks
        // lean on). On a cache hit this doubles as the cache-integrity gate:
        // the loaded BVH must equal one built fresh from the loaded scene.
        assert!(
            bvh.identical(&bvh::Bvh::build(&scene)),
            "BVH build is not deterministic (or the scene cache is corrupt)"
        );
        eprintln!("bvh: deterministic rebuild verified");
    }

    if let Some(mode) = &spin {
        let code = run_spin(&scene, &bvh, cam0, mode, spin_frames, &opts);
        std::process::exit(code);
    }
    // Must-fire structural gates are tuned to the default scene's topology —
    // skipped for --stress AND loaded OBJ scenes (a real scene can lack the
    // required features outright: a skyless view can't fire sky-tiles, and a
    // dense one legitimately overflows the replay recording arena). The
    // scene-agnostic zero-counter gates always run.
    let structural = stress.is_none() && obj.is_none();
    if check {
        let code = run_check(&scene, &bvh, cam0, structural);
        std::process::exit(code);
    }
    if check_dlss {
        let code = run_check_dlss(&scene, &bvh, cam0, dlss_dump);
        std::process::exit(code);
    }
    if check_xess {
        let code = run_check_xess(&scene, &bvh, cam0, xess_dump, structural);
        std::process::exit(code);
    }
    if check_fsr {
        let code = run_check_fsr(&scene, &bvh, cam0);
        std::process::exit(code);
    }
    if check_gpu {
        #[cfg(windows)]
        {
            let code = run_check_gpu(&scene, &bvh, cam0, &opts, structural);
            std::process::exit(code);
        }
        #[cfg(not(windows))]
        {
            eprintln!("--check-gpu requires Windows (D3D12)");
            std::process::exit(2);
        }
    }
    if check_dxr {
        #[cfg(windows)]
        {
            let code = run_check_dxr(&scene, &bvh, cam0, &opts, stress.is_none());
            std::process::exit(code);
        }
        #[cfg(not(windows))]
        {
            eprintln!("--check-dxr requires Windows (D3D12)");
            std::process::exit(2);
        }
    }
    if check_nppd {
        #[cfg(windows)]
        {
            let code = run_check_nppd(&scene, &bvh, cam0, &opts, nppd_dump, structural);
            std::process::exit(code);
        }
        #[cfg(not(windows))]
        {
            let _ = nppd_dump;
            eprintln!("--check-nppd requires Windows (the ONNX Runtime drop is Win64-only here)");
            std::process::exit(2);
        }
    }
    if check_oidn {
        #[cfg(windows)]
        {
            let code = run_check_oidn(&scene, &bvh, cam0, &opts, oidn_dump, structural);
            std::process::exit(code);
        }
        #[cfg(not(windows))]
        {
            let _ = oidn_dump;
            eprintln!("--check-oidn requires Windows (the OIDN SDK drop is Win64-only here)");
            std::process::exit(2);
        }
    }
    #[cfg(windows)]
    run_window(&scene, &bvh, &opts, cam0);
    #[cfg(not(windows))]
    {
        let _ = (&opts, cam0);
        eprintln!("the interactive window requires Windows (D3D12 + DLSS); use --check / --check-dlss");
        std::process::exit(2);
    }
}

/// Headless DXR-pipeline gate suite (--check-dxr). Needs real hardware (RT
/// tier 1.0) + the DXC DLLs, like --check-gpu. The DXR library pastes the
/// SAME shade.hlsli as the compute tracer, so the gates mirror the
/// --check-gpu reference gates: primary visibility vs the CPU plain
/// reference (statistical — hardware watertight intersection differs from
/// moller_trumbore at edges, and the RNG streams differ by design), a
/// 64-frame converged radiance A/B, and the resolve link the tonemap reads.
/// Exit codes: 0 = pass, 1 = a gate failed, 2 = environment.
#[cfg(windows)]
fn run_check_dxr(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    opts: &Opts,
    must_fire: bool,
) -> i32 {
    let dxc = match gpu::dxc::Dxc::load(&opts.dxc_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("check-dxr: {e}");
            return 2;
        }
    };
    let mut hg = match gpu::trace::HeadlessGpu::new(
        opts.gpu_debug,
        opts.prefer.unwrap_or(gpu::adapter::Prefer::Nvidia),
    ) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("check-dxr: device creation failed: {e}");
            return 2;
        }
    };
    eprintln!("check-dxr: adapter \"{}\"", hg.adapter_name);
    if let Err(e) = gpu::dxr::require_caps(&hg.device) {
        eprintln!("check-dxr: {e}");
        return 2;
    }
    let caps = gpu::trace::query_caps(&hg.device).unwrap();
    eprintln!(
        "check-dxr: RT tier {}.{}, shader model {}.{}",
        caps.rt_tier / 10,
        caps.rt_tier % 10,
        caps.shader_model >> 4,
        caps.shader_model & 0xf
    );

    let (gw, gh) = (800usize, 600usize);
    let dev = hg.device.clone();
    let dg = match gpu::dxr::DxrGpu::new(
        &dev,
        &dxc,
        scene,
        gw as u32,
        gh as u32,
        false,
        opts.gpu_debug,
        opts.bc7,
        &mut hg,
    ) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("check-dxr: FAIL DxrGpu init (RTPSO/SBT/scene upload): {e}");
            return 1;
        }
    };
    eprintln!(
        "check-dxr: RTPSO + SBT built, scene uploaded, BLAS/TLAS built ({} tris)",
        scene.tri_count()
    );

    let q = Quality::preset(2);
    let basis = cam0.basis(gw, gh);
    let ua = windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let read_f32 = |hg: &mut gpu::trace::HeadlessGpu, res, n: usize| -> Result<Vec<f32>, String> {
        let b = hg.read_buffer(res, ua, n * 4)?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    };

    // CPU counterpart: the plain per-pixel reference (hybrid = false).
    let stats = Stats::default();
    let accum: Vec<AtomicU32> = (0..gw * gh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..gw * gh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..gw * gh).map(|_| AtomicU32::new(0)).collect();
    let cpu_frame = |frame: u32| {
        let ctx = FrameCtx {
            scene,
            bvh,
            cam: basis,
            q,
            frame,
            jitter: frame > 0,
            rw: gw,
            rh: gh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: &[],
            accumulate: true,
            gbuf: None,
            fsr_buf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
            hemi_share: false,
            replay_rec: None,
            cut_cur: None,
            cut_prev: None,
            discard_seeds: false,
            defer_shade: false,
        };
        render::render_frame(&ctx, false);
    };
    let gpu_frame = |hg: &mut gpu::trace::HeadlessGpu,
                     dg: &gpu::dxr::DxrGpu,
                     frame: u32|
     -> Result<(), String> {
        dg.write_cb(
            0,
            &gpu::trace::FrameParams {
                cam: basis,
                frame,
                accumulate: true,
                jitter: frame > 0,
                frame_jitter: None,
                prev_cam: None,
                q,
                verify: false,
            },
        );
        let mut rec = Ok(());
        hg.run(|l| rec = dg.record_frame(l, 0))?;
        rec
    };

    // T1: one unjittered DispatchRays frame vs the CPU reference — primary
    // visibility (t + hit/sky classification), plus the must-fire halves.
    if let Err(e) = gpu_frame(&mut hg, &dg, 0) {
        eprintln!("check-dxr: FAIL DispatchRays: {e}");
        return 1;
    }
    let gpu_t = match read_f32(&mut hg, &dg.tbuf, gw * gh) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("check-dxr: FAIL tbuf readback: {e}");
            return 1;
        }
    };
    cpu_frame(0);
    let px = gw * gh;
    let mut class_mismatch = 0usize;
    let mut t_viol = 0usize;
    let mut max_rel = 0.0f32;
    let (mut n_hit, mut n_sky) = (0usize, 0usize);
    for i in 0..px {
        let ct = f32::from_bits(tbuf[i].load(Relaxed));
        let gt = gpu_t[i];
        if gt.is_finite() {
            n_hit += 1;
        } else {
            n_sky += 1;
        }
        match (ct.is_finite(), gt.is_finite()) {
            (true, true) => {
                let rel = (ct - gt).abs() / ct.max(1e-6);
                max_rel = max_rel.max(rel);
                if rel > 1e-3 {
                    t_viol += 1;
                }
            }
            (false, false) => {}
            _ => class_mismatch += 1,
        }
    }
    let mut ok = true;
    eprintln!(
        "check-dxr: primary visibility ({px} px): hit {n_hit} | sky {n_sky} | class-mismatch {class_mismatch} | rel-t > 1e-3: {t_viol} | max rel t err {max_rel:.2e}"
    );
    if class_mismatch as f64 > px as f64 * 5e-4 {
        eprintln!("check-dxr: FAIL hit/sky classification mismatch above 0.05%");
        ok = false;
    }
    if t_viol as f64 > px as f64 * 1e-4 {
        eprintln!("check-dxr: FAIL primary-t disagreement above 0.01% of pixels");
        ok = false;
    }
    if must_fire && (n_hit == 0 || n_sky == 0) {
        eprintln!("check-dxr: FAIL must-fire: the default view sees both geometry and sky");
        ok = false;
    }

    // T2: 64-frame jittered accumulation both sides — converged radiance
    // A/B (different RNG streams; only the means are comparable). Also the
    // finiteness gate over the raw HDR sums.
    const AB_FRAMES: u32 = 64;
    for f in 0..AB_FRAMES {
        if let Err(e) = gpu_frame(&mut hg, &dg, f) {
            eprintln!("check-dxr: FAIL accumulation frame {f}: {e}");
            return 1;
        }
    }
    for f in 0..AB_FRAMES {
        cpu_frame(f);
    }
    let gpu_acc = match read_f32(&mut hg, &dg.accum, px * 3) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("check-dxr: FAIL accum readback: {e}");
            return 1;
        }
    };
    let inv = 1.0 / AB_FRAMES as f32;
    let mut sum_c = [0.0f64; 3];
    let mut sum_g = [0.0f64; 3];
    let mut sum_abs = 0.0f64;
    let mut nonfinite = 0usize;
    for i in 0..px * 3 {
        let c = f32::from_bits(accum[i].load(Relaxed)) * inv;
        let g = gpu_acc[i] * inv;
        if !g.is_finite() || g < 0.0 {
            nonfinite += 1;
        }
        sum_c[i % 3] += c as f64;
        sum_g[i % 3] += g as f64;
        sum_abs += (c - g).abs() as f64;
    }
    let mut mean_rel = 0.0f64;
    for ch in 0..3 {
        let rel = (sum_c[ch] - sum_g[ch]).abs() / sum_c[ch].max(1e-9);
        mean_rel = mean_rel.max(rel);
    }
    eprintln!(
        "check-dxr: radiance A/B over {AB_FRAMES} frames: per-channel mean rel diff {:.3}% | mean abs px diff {:.4} | non-finite {nonfinite}",
        mean_rel * 100.0,
        sum_abs / (px * 3) as f64
    );
    if nonfinite > 0 {
        eprintln!("check-dxr: FAIL non-finite or negative HDR samples");
        ok = false;
    }
    if mean_rel > 0.02 {
        eprintln!("check-dxr: FAIL converged radiance means differ by more than 2%");
        ok = false;
    }

    // T3: the resolve pass (accum -> RGBA16F, the tonemap PS's input) —
    // texel == accum/samples within f16 precision; the present chain's only
    // compute link, verified headlessly.
    {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        };
        if let Err(e) = hg.run(|l| dg.record_resolve(l, 0, AB_FRAMES)) {
            eprintln!("check-dxr: FAIL resolve dispatch: {e}");
            return 1;
        }
        let pitch = gpu::d3d12::aligned_pitch(gw * 8);
        let rb = match gpu::d3d12::ReadbackBuffer::new(&hg.device, pitch * gh) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("check-dxr: FAIL readback alloc: {e}");
                return 1;
            }
        };
        let fp = gpu::d3d12::footprint(
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            gw as u32,
            gh as u32,
            8,
            0,
        );
        let hdr = dg.hdr.clone();
        if let Err(e) = hg.run(|l| unsafe {
            l.ResourceBarrier(&[gpu::d3d12::transition(
                &hdr,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyTextureRegion(
                &gpu::d3d12::loc_footprint(&rb.resource, fp),
                0,
                0,
                0,
                &gpu::d3d12::loc_subresource(&hdr),
                None,
            );
            l.ResourceBarrier(&[gpu::d3d12::transition(
                &hdr,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }) {
            eprintln!("check-dxr: FAIL hdr readback: {e}");
            return 1;
        }
        let mut ptr = std::ptr::null_mut();
        if let Err(e) = unsafe { rb.resource.Map(0, None, Some(&mut ptr)) } {
            eprintln!("check-dxr: FAIL hdr Map: {e}");
            return 1;
        }
        let mut resolve_viol = 0usize;
        for y in 0..gh {
            let row: &[[half::f16; 4]] = unsafe {
                std::slice::from_raw_parts((ptr as *const u8).add(y * pitch) as *const _, gw)
            };
            for (x, px_v) in row.iter().enumerate() {
                let i3 = (y * gw + x) * 3;
                for ch in 0..3 {
                    let want = gpu_acc[i3 + ch] * inv;
                    let got = f32::from(px_v[ch]);
                    if (want - got).abs() > want.abs().max(1.0) * 2e-3 {
                        resolve_viol += 1;
                    }
                }
            }
        }
        unsafe { rb.resource.Unmap(0, None) };
        eprintln!("check-dxr: resolve pass: {resolve_viol} texels off accum/samples (f16 tolerance)");
        if resolve_viol > 0 {
            eprintln!("check-dxr: FAIL resolve output disagrees with the accumulation");
            ok = false;
        }
    }

    // --- T4: the G-buffer pack under the upscaler contract ---
    // A second pipeline with the full pack (gbuf_full), at odd dims like
    // --check-gpu's M7: frame A (no prev), a 0.02*diag forward dolly, frame B
    // (prev = basis A) — both fresh 1-spp DispatchRays frames with zero
    // frame-uniform jitter. The pack is read back, unpacked into CPU GBufs,
    // and gated by the EXACT existing dlss::mv_selftest — zero new
    // tolerances — plus structural coverage (every px view-z > 0, sky depth
    // far BIT-EQUAL, sky must-fire).
    {
        let (pw, ph) = (533usize, 400usize);
        let dev = hg.device.clone();
        let mut dg2 = match gpu::dxr::DxrGpu::new(
            &dev,
            &dxc,
            scene,
            pw as u32,
            ph as u32,
            true,
            opts.gpu_debug,
            opts.bc7,
            &mut hg,
        ) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("check-dxr: FAIL gbuf DxrGpu init: {e}");
                return 1;
            }
        };
        let ua = windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let (near, far) = dlss::near_far(scene.diag);
        let uq = Quality::upscaler_1spp();
        let dxr_gbuf_frame = |hg: &mut gpu::trace::HeadlessGpu,
                              dg2: &gpu::dxr::DxrGpu,
                              basis: camera::CamBasis,
                              prev: Option<camera::CamBasis>,
                              frame: u32|
         -> Result<(dlss::GBufs, Vec<f32>), String> {
            let p = gpu::trace::FrameParams {
                cam: basis,
                frame,
                accumulate: false,
                jitter: false,
                frame_jitter: Some((0.0, 0.0)),
                prev_cam: prev,
                q: uq,
                verify: false,
            };
            dg2.write_cb(0, &p);
            let mut rec = Ok(());
            hg.run(|l| rec = dg2.record_frame(l, 0))?;
            rec?;
            let bytes = hg.read_buffer(&dg2.gbuf, ua, pw * ph * gpu::trace::GBUF_STRIDE as usize)?;
            let tb = hg.read_buffer(&dg2.tbuf, ua, pw * ph * 4)?;
            let t: Vec<f32> =
                tb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            Ok((unpack_gbuf_bytes(&bytes, pw, ph), t))
        };
        let basis_a = cam0.basis(pw, ph);
        let mut cam_b = cam0;
        cam_b.pos += cam0.forward() * (0.02 * scene.diag);
        let basis_b = cam_b.basis(pw, ph);
        let (ga, ta) = match dxr_gbuf_frame(&mut hg, &dg2, basis_a, None, 0) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("check-dxr: FAIL gbuf frame A: {e}");
                return 1;
            }
        };
        let (gb2, tb2) = match dxr_gbuf_frame(&mut hg, &dg2, basis_b, Some(basis_a), 1) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("check-dxr: FAIL gbuf frame B: {e}");
                return 1;
            }
        };
        let loadz = |g: &dlss::GBufs, i: usize| {
            f32::from_bits(g.depth[i].load(std::sync::atomic::Ordering::Relaxed))
        };
        let (mut bad_z, mut sky_off, mut skies) = (0usize, 0usize, 0usize);
        for i in 0..pw * ph {
            let z = loadz(&ga, i);
            if !(z > 0.0) {
                bad_z += 1;
            }
            if !ta[i].is_finite() {
                skies += 1;
                if z.to_bits() != far.to_bits() {
                    sky_off += 1;
                }
            }
        }
        let mv_ok = dlss::mv_selftest(
            &ga,
            &basis_a,
            &gb2,
            &basis_b,
            &dlss::cam_matrices(&cam_b, pw, ph, near, far),
            scene.diag,
            far,
        );
        eprintln!(
            "check-dxr: gbuf pack ({pw}x{ph}): view-z<=0 {bad_z} | sky-depth-off {sky_off} (sky px {skies}) | mv/depth/matrix {}",
            if mv_ok { "OK" } else { "FAIL" },
        );
        if !mv_ok || bad_z != 0 || sky_off != 0 {
            eprintln!("check-dxr: FAIL DXR G-buffer pack gates");
            ok = false;
        }
        if must_fire && skies == 0 {
            eprintln!("check-dxr: FAIL gbuf sky gate vacuous (no sky pixels on the default scene)");
            ok = false;
        }
        // Textured scenes: the pack's albedo plane vs a CPU render (the
        // guide-chain proof; the check-gpu M7 twin).
        if !albedo_ab_check(scene, bvh, cam0, &ga, &ta, pw, ph, "dxr") {
            ok = false;
        }

        // --- T5: the XeSS feed over the DXR pack (frame B, still resident) ---
        {
            let xres = match gpu::xr::XessResources::new(
                &hg.device,
                pw as u32,
                ph as u32,
                pw as u32,
                ph as u32,
            ) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("check-dxr: FAIL feed XessResources: {e}");
                    return 1;
                }
            };
            let pl = xres.plane_resources();
            if let Err(e) = dg2.wire_feed(
                &hg.device,
                gpu::trace::FeedKind::Xess,
                &[
                    (gpu::trace::FEED_COLOR, pl[0].0, pl[0].1),
                    (gpu::trace::FEED_MVEC, pl[1].0, pl[1].1),
                    (gpu::trace::FEED_DEPTH, pl[2].0, pl[2].1),
                ],
            ) {
                eprintln!("check-dxr: FAIL feed wiring: {e}");
                return 1;
            }
            let mut feed_rec = Ok(());
            if let Err(e) = hg.run(|l| feed_rec = dg2.record_feed(l, 0)) {
                eprintln!("check-dxr: FAIL feed dispatch submit: {e}");
                return 1;
            }
            if let Err(e) = feed_rec {
                eprintln!("check-dxr: FAIL feed dispatch: {e}");
                return 1;
            }
            let (depth_bytes, mvec_bytes, color_bytes) = match (
                read_feed_tex(&mut hg, pl[2].0, pl[2].1, 4, pw, ph),
                read_feed_tex(&mut hg, pl[1].0, pl[1].1, 4, pw, ph),
                read_feed_tex(&mut hg, pl[0].0, pl[0].1, 8, pw, ph),
            ) {
                (Ok(d), Ok(m), Ok(c)) => (d, m, c),
                _ => {
                    eprintln!("check-dxr: FAIL feed plane readback");
                    return 1;
                }
            };
            let accum_bytes = match hg.read_buffer(&dg2.accum, ua, pw * ph * 12) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("check-dxr: FAIL accum readback: {e}");
                    return 1;
                }
            };
            if !gate_xess_feed(
                "check-dxr",
                pw,
                ph,
                &depth_bytes,
                &mvec_bytes,
                &color_bytes,
                &accum_bytes,
                &gb2,
                &tb2,
                near,
                far,
                must_fire,
            ) {
                ok = false;
            }

            // --- T5b: the FSR3 feed over the same pack (FeedKind::Fsr3 ->
            // cs_feed_xess into the FSR 3.1 planes), gated identically.
            {
                let fres = match gpu::ffx_up::Fsr3Resources::new(
                    &hg.device,
                    pw as u32,
                    ph as u32,
                    pw as u32,
                    ph as u32,
                ) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("check-dxr: FAIL feed Fsr3Resources: {e}");
                        return 1;
                    }
                };
                let fpl = fres.plane_resources();
                if let Err(e) = dg2.wire_feed(
                    &hg.device,
                    gpu::trace::FeedKind::Fsr3,
                    &[
                        (gpu::trace::FEED_COLOR, fpl[0].0, fpl[0].1),
                        (gpu::trace::FEED_MVEC, fpl[1].0, fpl[1].1),
                        (gpu::trace::FEED_DEPTH, fpl[2].0, fpl[2].1),
                    ],
                ) {
                    eprintln!("check-dxr: FAIL FSR3 feed wiring: {e}");
                    return 1;
                }
                let mut f_rec = Ok(());
                if let Err(e) = hg.run(|l| f_rec = dg2.record_feed(l, 0)) {
                    eprintln!("check-dxr: FAIL FSR3 feed dispatch submit: {e}");
                    return 1;
                }
                if let Err(e) = f_rec {
                    eprintln!("check-dxr: FAIL FSR3 feed dispatch: {e}");
                    return 1;
                }
                let (f_depth, f_mvec, f_color) = match (
                    read_feed_tex(&mut hg, fpl[2].0, fpl[2].1, 4, pw, ph),
                    read_feed_tex(&mut hg, fpl[1].0, fpl[1].1, 4, pw, ph),
                    read_feed_tex(&mut hg, fpl[0].0, fpl[0].1, 8, pw, ph),
                ) {
                    (Ok(d), Ok(m), Ok(c)) => (d, m, c),
                    _ => {
                        eprintln!("check-dxr: FAIL FSR3 feed plane readback");
                        return 1;
                    }
                };
                if !gate_xess_feed(
                    "check-dxr fsr3",
                    pw,
                    ph,
                    &f_depth,
                    &f_mvec,
                    &f_color,
                    &accum_bytes,
                    &gb2,
                    &tb2,
                    near,
                    far,
                    must_fire,
                ) {
                    ok = false;
                }
            }

            // --- T6: the RR feed over the same pack, every plane gated ---
            let rres = match gpu::rr::RrResources::new(
                &hg.device,
                (pw as u32, ph as u32),
                (pw as u32, ph as u32),
                (pw as u32, ph as u32),
                pw as u32,
                ph as u32,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("check-dxr: FAIL feed RrResources: {e}");
                    return 1;
                }
            };
            let rpl = rres.plane_resources();
            if let Err(e) = dg2.wire_feed(
                &hg.device,
                gpu::trace::FeedKind::Rr,
                &[
                    (gpu::trace::FEED_COLOR, rpl[0].0, rpl[0].1),
                    (gpu::trace::FEED_NR, rpl[1].0, rpl[1].1),
                    (gpu::trace::FEED_DEPTH, rpl[2].0, rpl[2].1),
                    (gpu::trace::FEED_MVEC, rpl[3].0, rpl[3].1),
                    (gpu::trace::FEED_ALB, rpl[4].0, rpl[4].1),
                    (gpu::trace::FEED_SPEC, rpl[5].0, rpl[5].1),
                    (gpu::trace::FEED_SPECHIT, rpl[6].0, rpl[6].1),
                ],
            ) {
                eprintln!("check-dxr: FAIL RR feed wiring: {e}");
                return 1;
            }
            let mut rr_rec = Ok(());
            if let Err(e) = hg.run(|l| rr_rec = dg2.record_feed(l, 0)) {
                eprintln!("check-dxr: FAIL RR feed dispatch submit: {e}");
                return 1;
            }
            if let Err(e) = rr_rec {
                eprintln!("check-dxr: FAIL RR feed dispatch: {e}");
                return 1;
            }
            let mut read_plane = |idx: usize, bpp: usize, what: &str| -> Option<Vec<u8>> {
                match read_feed_tex(&mut hg, rpl[idx].0, rpl[idx].1, bpp, pw, ph) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        eprintln!("check-dxr: FAIL RR {what} plane readback: {e}");
                        None
                    }
                }
            };
            let (Some(rr_color), Some(rr_nr), Some(rr_depth), Some(rr_mvec)) = (
                read_plane(0, 8, "color"),
                read_plane(1, 8, "normal_rough"),
                read_plane(2, 4, "depth"),
                read_plane(3, 4, "mvec"),
            ) else {
                return 1;
            };
            let (Some(rr_alb), Some(rr_spec), Some(rr_spechit)) = (
                read_plane(4, 4, "albedo"),
                read_plane(5, 4, "spec_albedo"),
                read_plane(6, 2, "spec_hit"),
            ) else {
                return 1;
            };
            // --- T6b: the FSR4-RR feed. Wiring FeedKind::FsrRr arms
            // FLAG_FSR_SIG, so frame B is RE-TRACED with the same params:
            // accum must come back BIT-IDENTICAL, and the nine planes gate
            // against oracles from the armed pack's readback.
            {
                let fres = match gpu::ffx_rr::FsrResources::new(
                    &hg.device,
                    pw as u32,
                    ph as u32,
                    pw as u32,
                    ph as u32,
                ) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("check-dxr: FAIL feed FsrResources: {e}");
                        return 1;
                    }
                };
                let fpl = fres.plane_resources();
                if let Err(e) = dg2.wire_feed(
                    &hg.device,
                    gpu::trace::FeedKind::FsrRr,
                    &[
                        (gpu::trace::FEED_SPECHIT, fpl[0].0, fpl[0].1),
                        (gpu::trace::FEED_DEPTH, fpl[1].0, fpl[1].1),
                        (gpu::trace::FEED_FSR_MVEC, fpl[2].0, fpl[2].1),
                        (gpu::trace::FEED_NR, fpl[3].0, fpl[3].1),
                        (gpu::trace::FEED_ALB, fpl[4].0, fpl[4].1),
                        (gpu::trace::FEED_SPEC, fpl[5].0, fpl[5].1),
                        (gpu::trace::FEED_FSR_DD, fpl[6].0, fpl[6].1),
                        (gpu::trace::FEED_FSR_DS, fpl[7].0, fpl[7].1),
                        (gpu::trace::FEED_COLOR, fpl[8].0, fpl[8].1),
                    ],
                ) {
                    eprintln!("check-dxr: FAIL FSR4-RR feed wiring: {e}");
                    return 1;
                }
                let p = gpu::trace::FrameParams {
                    cam: basis_b,
                    frame: 1,
                    accumulate: false,
                    jitter: false,
                    frame_jitter: Some((0.0, 0.0)),
                    prev_cam: Some(basis_a),
                    q: uq,
                    verify: false,
                };
                dg2.write_cb(0, &p);
                let mut rec = Ok(());
                if hg.run(|l| rec = dg2.record_frame(l, 0)).is_err() || rec.is_err() {
                    eprintln!("check-dxr: FAIL FSR4-RR re-trace");
                    return 1;
                }
                let (pack2, accum2) = match (
                    hg.read_buffer(&dg2.gbuf, ua, pw * ph * gpu::trace::GBUF_STRIDE as usize),
                    hg.read_buffer(&dg2.accum, ua, pw * ph * 12),
                ) {
                    (Ok(a), Ok(b)) => (a, b),
                    _ => {
                        eprintln!("check-dxr: FAIL FSR4-RR pack/accum readback");
                        return 1;
                    }
                };
                if accum2 != accum_bytes {
                    eprintln!(
                        "check-dxr: FAIL FSR-sig on/off accum not bit-identical (the sig capture changed shading)"
                    );
                    ok = false;
                }
                let mut f_rec = Ok(());
                if let Err(e) = hg.run(|l| f_rec = dg2.record_feed(l, 0)) {
                    eprintln!("check-dxr: FAIL FSR4-RR feed dispatch submit: {e}");
                    return 1;
                }
                if let Err(e) = f_rec {
                    eprintln!("check-dxr: FAIL FSR4-RR feed dispatch: {e}");
                    return 1;
                }
                let mut read_plane = |idx: usize, bpp: usize, what: &str| -> Option<Vec<u8>> {
                    match read_feed_tex(&mut hg, fpl[idx].0, fpl[idx].1, bpp, pw, ph) {
                        Ok(v) => Some(v),
                        Err(e) => {
                            eprintln!("check-dxr: FAIL FSR4-RR {what} plane readback: {e}");
                            None
                        }
                    }
                };
                let (Some(f_dlin), Some(f_dclip), Some(f_mvec), Some(f_nrm)) = (
                    read_plane(0, 4, "depth_lin"),
                    read_plane(1, 4, "depth_clip"),
                    read_plane(2, 8, "mvec"),
                    read_plane(3, 4, "normals"),
                ) else {
                    return 1;
                };
                let (Some(f_alb), Some(f_spec), Some(f_dd), Some(f_ds), Some(f_res)) = (
                    read_plane(4, 4, "diff_alb"),
                    read_plane(5, 4, "spec_alb"),
                    read_plane(6, 8, "dd"),
                    read_plane(7, 8, "ds"),
                    read_plane(8, 8, "residual"),
                ) else {
                    return 1;
                };
                if !gate_fsr_rr_feed(
                    "check-dxr",
                    pw,
                    ph,
                    &f_dlin,
                    &f_dclip,
                    &f_mvec,
                    &f_nrm,
                    &f_alb,
                    &f_spec,
                    &f_dd,
                    &f_ds,
                    &f_res,
                    &pack2,
                    &accum2,
                    near,
                    far,
                    must_fire,
                ) {
                    ok = false;
                }
            }

            if !gate_rr_feed(
                "check-dxr",
                pw,
                ph,
                &rr_color,
                &rr_nr,
                &rr_depth,
                &rr_mvec,
                &rr_alb,
                &rr_spec,
                &rr_spechit,
                &accum_bytes,
                &gb2,
            ) {
                ok = false;
            }
        }
    }

    if ok {
        println!("check-dxr: PASS (DispatchRays pipeline vs the CPU plain reference)");
        0
    } else {
        1
    }
}

/// Unpack a GPU `GBufPx` pack readback (GBUF_STRIDE = 80 B/px, 20 lanes:
/// nr | alb+z | spec | mv | sig) into a CPU `dlss::GBufs` so the existing
/// CPU gates (`dlss::mv_selftest`) consume it unmodified — the four sig
/// lanes and mv.zw are FSR-RR extras the GBufs shape doesn't carry (their
/// own gates read the raw bytes). Shared by --check-gpu (M7) and
/// --check-dxr (T4).
#[cfg(windows)]
fn unpack_gbuf_bytes(bytes: &[u8], pw: usize, ph: usize) -> dlss::GBufs {
    let f = |i: usize| f32::from_le_bytes(bytes[i * 4..][..4].try_into().unwrap());
    let g = dlss::GBufs::new(pw, ph);
    let lanes = gpu::trace::GBUF_STRIDE as usize / 4;
    for i in 0..pw * ph {
        let b = i * lanes;
        g.write(
            i % pw,
            i / pw,
            &dlss::GPixel {
                normal: Vec3A::new(f(b), f(b + 1), f(b + 2)),
                rough: f(b + 3),
                diff_alb: Vec3A::new(f(b + 4), f(b + 5), f(b + 6)),
                view_z: f(b + 7),
                spec_alb: Vec3A::new(f(b + 8), f(b + 9), f(b + 10)),
                spec_hit_t: f(b + 11),
                mv: (f(b + 12), f(b + 13)),
            },
        );
    }
    g
}

/// Row-packed feed-plane readback (footprint pitch is 256-aligned); the
/// plane rests in NON_PIXEL_SHADER_RESOURCE around the copy.
#[cfg(windows)]
fn read_feed_tex(
    hg: &mut gpu::trace::HeadlessGpu,
    tex: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    fmt: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    bpp: usize,
    pw: usize,
    ph: usize,
) -> Result<Vec<u8>, String> {
    use gpu::d3d12::{aligned_pitch, footprint, loc_footprint, loc_subresource, transition, ReadbackBuffer};
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    };
    let pitch = aligned_pitch(pw * bpp);
    let rb = ReadbackBuffer::new(&hg.device, pitch * ph)?;
    hg.run(|l| unsafe {
        let npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        l.ResourceBarrier(&[transition(tex, npsr, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let fp = footprint(fmt, pw as u32, ph as u32, bpp, 0);
        l.CopyTextureRegion(
            &loc_footprint(&rb.resource, fp),
            0,
            0,
            0,
            &loc_subresource(tex),
            None,
        );
        l.ResourceBarrier(&[transition(tex, D3D12_RESOURCE_STATE_COPY_SOURCE, npsr)]);
    })?;
    let mut ptr = std::ptr::null_mut();
    unsafe { rb.resource.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("Map: {e}"))?;
    let mut out = vec![0u8; pw * bpp * ph];
    for y in 0..ph {
        unsafe {
            std::ptr::copy_nonoverlapping(
                (ptr as *const u8).add(y * pitch),
                out.as_mut_ptr().add(y * pw * bpp),
                pw * bpp,
            );
        }
    }
    unsafe { rb.resource.Unmap(0, None) };
    Ok(out)
}

/// f16 bit pattern -> monotone integer (ulp distances work across the sign,
/// unlike raw bits).
#[cfg(windows)]
fn mono16(b: u16) -> i32 {
    if b & 0x8000 != 0 { -((b & 0x7fff) as i32) } else { b as i32 }
}

/// The XeSS feed gates: the depth plane vs `xess::view_z_to_clip_depth` of
/// the pack's view-Z (hits <= 4 f32 ulp — D3D's fp32 divide is ~2.5 ulp even
/// under `precise`; sky BIT-EQUAL 0.0 + anti-vacuity), the mvec plane vs the
/// pack within 1 f16 ulp, the color plane vs the 1-spp accum store within
/// 1 f16 ulp (alpha a constant 1.0). `tag` prefixes the report lines.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn gate_xess_feed(
    tag: &str,
    pw: usize,
    ph: usize,
    depth_bytes: &[u8],
    mvec_bytes: &[u8],
    color_bytes: &[u8],
    accum_bytes: &[u8],
    gb2: &dlss::GBufs,
    tb2: &[f32],
    near: f32,
    far: f32,
    must_fire: bool,
) -> bool {
    let accf = |i: usize| f32::from_le_bytes(accum_bytes[i * 4..][..4].try_into().unwrap());
    let mut ok = true;
    let (mut d_ulp_bad, mut d_sky_bad, mut mv_ulp_bad) = (0usize, 0usize, 0usize);
    let mut c_ulp_bad = 0usize;
    let mut d_sky_n = 0usize;
    let mut max_d_ulp = 0u32;
    for i in 0..pw * ph {
        let z = f32::from_bits(gb2.depth[i].load(Relaxed));
        let got = f32::from_le_bytes(depth_bytes[i * 4..][..4].try_into().unwrap());
        if !tb2[i].is_finite() {
            // Sky: the encode must land EXACTLY on 0.0.
            d_sky_n += 1;
            if got.to_bits() != 0 {
                d_sky_bad += 1;
            }
        } else {
            let expect = xess::view_z_to_clip_depth(z, near, far);
            let d = got.to_bits().abs_diff(expect.to_bits());
            max_d_ulp = max_d_ulp.max(d);
            // 4 ulp: division is the one op HLSL doesn't round like IEEE.
            // Formula drift, wrong near/far, or a lost `precise` all blow
            // past this by orders of magnitude; sky stays gated bit-equal.
            if d > 4 {
                d_ulp_bad += 1;
            }
        }
        for ch in 0..2usize {
            let got16 = u16::from_le_bytes(mvec_bytes[i * 4 + ch * 2..][..2].try_into().unwrap());
            // MV storage is f16 bits — the expected texture value IS the
            // stored word (both sides rounded the same pack f32 once);
            // 1 ulp of typed-store latitude stays.
            let expect16 = gb2.mvec[i * 2 + ch].load(Relaxed);
            if (mono16(got16) - mono16(expect16)).unsigned_abs() > 1 {
                mv_ulp_bad += 1;
            }
        }
        // Color: accum's 1-spp store -> RGBA16F (typed-store RTNE matches
        // half's, 1 ulp headroom like mvec; alpha is a constant 1.0).
        for ch in 0..3usize {
            let got16 = u16::from_le_bytes(color_bytes[i * 8 + ch * 2..][..2].try_into().unwrap());
            let expect16 = half::f16::from_f32(accf(i * 3 + ch)).to_bits();
            if (mono16(got16) - mono16(expect16)).unsigned_abs() > 1 {
                c_ulp_bad += 1;
            }
        }
        if u16::from_le_bytes(color_bytes[i * 8 + 6..][..2].try_into().unwrap())
            != half::f16::ONE.to_bits()
        {
            c_ulp_bad += 1;
        }
    }
    eprintln!(
        "{tag}: xess feed ({pw}x{ph}): depth-ulp>4 {d_ulp_bad} (max {max_d_ulp}) | sky-not-0.0 {d_sky_bad} (sky px {d_sky_n}) | mvec-ulp>1 {mv_ulp_bad} | color-ulp>1 {c_ulp_bad}"
    );
    if d_ulp_bad != 0 || d_sky_bad != 0 || mv_ulp_bad != 0 || c_ulp_bad != 0 {
        eprintln!("{tag}: FAIL XeSS feed gates");
        ok = false;
    }
    // Anti-vacuity for the exact-zero sky encode (--stress skips).
    if must_fire && d_sky_n == 0 {
        eprintln!("{tag}: FAIL xess feed sky gate vacuous (no sky pixels on the default scene)");
        ok = false;
    }
    ok
}

/// The RR feed gates: the linear-depth plane BIT-EQUAL to the pack's view-Z
/// (R32F passthrough — no arithmetic, so no ulp allowance), every other
/// plane at its storage tolerance (f16 ulp / <= 2 UNORM LSB) so a
/// plane-order swap in either wiring table cannot pass the suite. Plane
/// byte slices in RR plane order (color, nr, depth, mvec, alb, spec,
/// spechit). `tag` prefixes the report lines.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn gate_rr_feed(
    tag: &str,
    pw: usize,
    ph: usize,
    rr_color: &[u8],
    rr_nr: &[u8],
    rr_depth: &[u8],
    rr_mvec: &[u8],
    rr_alb: &[u8],
    rr_spec: &[u8],
    rr_spechit: &[u8],
    accum_bytes: &[u8],
    gb2: &dlss::GBufs,
) -> bool {
    let accf = |i: usize| f32::from_le_bytes(accum_bytes[i * 4..][..4].try_into().unwrap());
    // The CPU upload's UNORM encode; the hardware typed store rounds RTNE
    // with the spec's 0.6-ULP latitude — gate at <= 1 LSB (+1 for the f16
    // hop, below).
    let to_unorm8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let (mut rd_bad, mut rmv_bad) = (0usize, 0usize);
    let (mut rc_bad, mut rnr_bad, mut ralb_bad, mut rsh_bad) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..pw * ph {
        let got = u32::from_le_bytes(rr_depth[i * 4..][..4].try_into().unwrap());
        if got != gb2.depth[i].load(Relaxed) {
            rd_bad += 1;
        }
        for ch in 0..2usize {
            let got16 = u16::from_le_bytes(rr_mvec[i * 4 + ch * 2..][..2].try_into().unwrap());
            // f16 storage: the expected bits ARE the stored word.
            let expect16 = gb2.mvec[i * 2 + ch].load(Relaxed);
            if (mono16(got16) - mono16(expect16)).unsigned_abs() > 1 {
                rmv_bad += 1;
            }
        }
        for ch in 0..3usize {
            let got16 = u16::from_le_bytes(rr_color[i * 8 + ch * 2..][..2].try_into().unwrap());
            let expect16 = half::f16::from_f32(accf(i * 3 + ch)).to_bits();
            if (mono16(got16) - mono16(expect16)).unsigned_abs() > 1 {
                rc_bad += 1;
            }
            // RGBA8 albedo/spec-albedo vs the CPU encode. <= 2 LSB: the CPU
            // reference is double-rounded (pack f32 -> f16 storage ->
            // unorm8) while the GPU feed converts the pack f32 to unorm8
            // directly — the f16 hop can move a value across a *.5 rounding
            // boundary (1 LSB), on top of the typed store's own 0.6-ULP
            // latitude (1 LSB).
            let a = rr_alb[i * 4 + ch];
            let ea = to_unorm8(dlss::ld16(&gb2.diff_alb[i * 3 + ch]));
            let s = rr_spec[i * 4 + ch];
            let es = to_unorm8(dlss::ld16(&gb2.spec_alb[i * 3 + ch]));
            if a.abs_diff(ea) > 2 || s.abs_diff(es) > 2 {
                ralb_bad += 1;
            }
        }
        for ch in 0..4usize {
            let got16 = u16::from_le_bytes(rr_nr[i * 8 + ch * 2..][..2].try_into().unwrap());
            let expect16 = gb2.normal_rough[i * 4 + ch].load(Relaxed);
            if (mono16(got16) - mono16(expect16)).unsigned_abs() > 1 {
                rnr_bad += 1;
            }
        }
        let got16 = u16::from_le_bytes(rr_spechit[i * 2..][..2].try_into().unwrap());
        let expect16 = gb2.spec_hit_t[i].load(Relaxed);
        if (mono16(got16) - mono16(expect16)).unsigned_abs() > 1 {
            rsh_bad += 1;
        }
    }
    eprintln!(
        "{tag}: rr feed ({pw}x{ph}): depth-not-bit-equal {rd_bad} | mvec-ulp>1 {rmv_bad} | color-ulp>1 {rc_bad} | nr-ulp>1 {rnr_bad} | alb-lsb>2 {ralb_bad} | spechit-ulp>1 {rsh_bad}"
    );
    if rd_bad != 0 || rmv_bad != 0 || rc_bad != 0 || rnr_bad != 0 || ralb_bad != 0 || rsh_bad != 0 {
        eprintln!("{tag}: FAIL RR feed gates");
        return false;
    }
    true
}

/// The FSR4-RR feed gates over a pack traced WITH the FsrRr wiring
/// (FLAG_FSR_SIG armed): all nine planes vs CPU oracles computed from the
/// raw 80-B pack readback + accum. dd/ds are gated BIT-EQUAL to the pack's
/// sig f16 halves (pure widen), the linear-depth plane bit-equal to view-Z
/// (DEPTH_SIGN = 1 passthrough), clip depth at the XeSS 4-ulp bound with
/// sky bit-equal 0.0, the albedos at <= 1 LSB against the single explicit
/// sqrt quantization (`fsr::sqrt_encode8` of the pack f32 — no f16 hop
/// here, unlike gate_rr_feed's 2-LSB double-rounding allowance), the
/// residual against the exact f32 remainder, and the sky contract (sig == 0,
/// prev-Z == far bit-equal => mvec B exactly 0).
///
/// The residual is the one plane whose tolerance is NOT an ulp count of its
/// own value: it is a near-**cancellation** (`color − dd⊗kd − ds⊗f0` — the
/// two products are the same magnitude as the color), so the f32 arithmetic
/// slop is bounded by the CANCELLED terms, not by the tiny result. Gating it
/// at 1 f16 ulp of the remainder makes the limit shrink toward zero exactly
/// where the error doesn't — a lit pixel whose residual lands near 0 then
/// fails on ordinary f32 rounding. The bound below is absolute: a few f32
/// ulps of the largest cancelled term plus the residual's own f16 storage
/// step. (Both sides evaluate the SAME expression on the SAME inputs — the
/// wire factors come from the stored plane bytes — so this only forgives
/// rounding, never a wiring or formula error, which moves the residual by
/// the size of a whole term.)
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn gate_fsr_rr_feed(
    tag: &str,
    pw: usize,
    ph: usize,
    depth_lin: &[u8],
    depth_clip: &[u8],
    mvec: &[u8],
    normals: &[u8],
    alb: &[u8],
    spec: &[u8],
    dd: &[u8],
    ds: &[u8],
    residual: &[u8],
    pack: &[u8],
    accum_bytes: &[u8],
    near: f32,
    far: f32,
    must_fire: bool,
) -> bool {
    let lanes = gpu::trace::GBUF_STRIDE as usize / 4;
    let lane_f = |i: usize, l: usize| {
        f32::from_le_bytes(pack[(i * lanes + l) * 4..][..4].try_into().unwrap())
    };
    let lane_u = |i: usize, l: usize| {
        u32::from_le_bytes(pack[(i * lanes + l) * 4..][..4].try_into().unwrap())
    };
    let accf = |i: usize| f32::from_le_bytes(accum_bytes[i * 4..][..4].try_into().unwrap());
    let h16 = |bytes: &[u8], off: usize| u16::from_le_bytes(bytes[off..][..2].try_into().unwrap());
    let (mut dl_bad, mut dc_bad, mut dc_sky_bad, mut dc_sky_n) = (0usize, 0usize, 0usize, 0usize);
    let (mut mv_bad, mut nrm_bad, mut alb_bad, mut sig_bad) = (0usize, 0usize, 0usize, 0usize);
    let (mut res_bad, mut sky_sig_bad, mut sig_fired) = (0usize, 0usize, 0usize);
    let mut max_dc_ulp = 0u32;
    // Worst margin among the channels that needed the cancellation escape
    // (negative = inside tolerance); -inf when none did.
    let mut worst_res = f32::NEG_INFINITY;
    for i in 0..pw * ph {
        let view_z = lane_f(i, 7);
        let sky = view_z.to_bits() == far.to_bits();
        // Linear depth: R32F passthrough (DEPTH_SIGN = 1), bit-equal.
        if u32::from_le_bytes(depth_lin[i * 4..][..4].try_into().unwrap()) != view_z.to_bits() {
            dl_bad += 1;
        }
        // Clip depth: the XeSS encode bounds.
        let got_dc = f32::from_le_bytes(depth_clip[i * 4..][..4].try_into().unwrap());
        if sky {
            dc_sky_n += 1;
            if got_dc.to_bits() != 0 {
                dc_sky_bad += 1;
            }
        } else {
            let d = got_dc.to_bits().abs_diff(xess::view_z_to_clip_depth(view_z, near, far).to_bits());
            max_dc_ulp = max_dc_ulp.max(d);
            if d > 4 {
                dc_bad += 1;
            }
        }
        // MVs: RG = pixel MV / render dims, B = prev_z - view_z, all f16.
        let (mvx, mvy, prev_z) = (lane_f(i, 12), lane_f(i, 13), lane_f(i, 14));
        let expect = [mvx / pw as f32, mvy / ph as f32, prev_z - view_z];
        for (ch, e) in expect.iter().enumerate() {
            let got16 = h16(mvec, i * 8 + ch * 2);
            let expect16 = half::f16::from_f32(*e).to_bits();
            if (mono16(got16) - mono16(expect16)).unsigned_abs() > 1 {
                mv_bad += 1;
            }
        }
        if sky && h16(mvec, i * 8 + 4) != 0 {
            // Sky prev-Z is far bit-equal, so the depth delta is exactly 0.
            mv_bad += 1;
        }
        // Normals: RGB10A2 — oct RG + rough B at <= 1 LSB, A == 0.
        let got_n = u32::from_le_bytes(normals[i * 4..][..4].try_into().unwrap());
        let n = Vec3A::new(lane_f(i, 0), lane_f(i, 1), lane_f(i, 2));
        let (eu, ev) = fsr::oct_encode(n);
        let q10 = |v: f32| (v.clamp(0.0, 1.0) * 1023.0 + 0.5) as u32;
        let l10 = |w: u32, s: u32| (w >> s) & 0x3ff;
        if l10(got_n, 0).abs_diff(q10(eu)) > 1
            || l10(got_n, 10).abs_diff(q10(ev)) > 1
            || l10(got_n, 20).abs_diff(q10(lane_f(i, 3))) > 1
            || (got_n >> 30) != 0
        {
            nrm_bad += 1;
        }
        // Signals: the planes are pure widens of the pack's sig f16 halves.
        let sig = [lane_u(i, 16), lane_u(i, 17), lane_u(i, 18)];
        let dd16 = [sig[0] as u16, (sig[0] >> 16) as u16, sig[1] as u16];
        let ds16 = [(sig[1] >> 16) as u16, sig[2] as u16, (sig[2] >> 16) as u16];
        for ch in 0..3 {
            if h16(dd, i * 8 + ch * 2) != dd16[ch] || h16(ds, i * 8 + ch * 2) != ds16[ch] {
                sig_bad += 1;
            }
        }
        if sky {
            if sig != [0, 0, 0] || prev_z.to_bits() != far.to_bits() {
                sky_sig_bad += 1;
            }
        } else if sig != [0, 0, 0] {
            sig_fired += 1;
        }
        // Albedos: ONE explicit sqrt quantization of the pack f32 — <= 1 LSB
        // vs the CPU encode (GPU sqrt has 1-ulp latitude, unlike Rust's
        // correctly-rounded sqrt). The residual oracle below therefore takes
        // its wire factors from the PLANE BYTES the GPU actually stored
        // ((n/255)^2 — bit-identical to the kernel's own enc*enc), not from
        // a recompute that could land on the other side of a rounding
        // boundary.
        let mut wire = [Vec3A::ZERO; 2];
        for (k, (plane, base)) in [(alb, 4usize), (spec, 8usize)].into_iter().enumerate() {
            for ch in 0..3 {
                let v = lane_f(i, base + ch);
                let b = plane[i * 4 + ch];
                if b.abs_diff(fsr::sqrt_encode8(v)) > 1 {
                    alb_bad += 1;
                }
                let enc = b as f32 / 255.0;
                wire[k][ch] = enc * enc;
            }
        }
        // Residual: the exact f32 remainder (same expression, same order as
        // the kernel) through the RGBA16F store.
        let ddf = Vec3A::new(
            half::f16::from_bits(dd16[0]).to_f32(),
            half::f16::from_bits(dd16[1]).to_f32(),
            half::f16::from_bits(dd16[2]).to_f32(),
        );
        let dsf = Vec3A::new(
            half::f16::from_bits(ds16[0]).to_f32(),
            half::f16::from_bits(ds16[1]).to_f32(),
            half::f16::from_bits(ds16[2]).to_f32(),
        );
        for ch in 0..3 {
            let color = accf(i * 3 + ch);
            let (t_dd, t_ds) = (ddf[ch] * wire[0][ch], dsf[ch] * wire[1][ch]);
            let e = color - t_dd - t_ds;
            let got16 = h16(residual, i * 8 + ch * 2);
            if (mono16(got16) - mono16(fsr::f16_sat(e).to_bits())).unsigned_abs() <= 1 {
                continue; // the ordinary storage tolerance, like every other plane
            }
            // Beyond 1 ulp: forgive ONLY what the cancellation explains. The
            // GPU's f32 remainder may differ from `e` by a few ulps of the
            // largest CANCELLED term (a magnitude the tiny result knows
            // nothing about), and its own f16 store rounds by half an ulp of
            // what it holds. Anything past that is a real defect — a wiring
            // or formula error moves the residual by the size of a whole term.
            let got = half::f16::from_bits(got16).to_f32();
            let cancelled = color.abs().max(t_dd.abs()).max(t_ds.abs());
            let tol = 8.0 * f32::EPSILON * cancelled + (got.abs() * 4.9e-4).max(3.0e-8);
            let err = (got - e).abs();
            worst_res = worst_res.max(err - tol);
            if err > tol {
                res_bad += 1;
            }
        }
    }
    eprintln!(
        "{tag}: fsr-rr feed ({pw}x{ph}): depth-lin-not-bit-equal {dl_bad} | clip-ulp>4 {dc_bad} (max {max_dc_ulp}) | sky-not-0.0 {dc_sky_bad} (sky px {dc_sky_n}) | mvec-ulp>1 {mv_bad} | normals-lsb>1 {nrm_bad} | alb-lsb>1 {alb_bad} | sig-not-bit-equal {sig_bad} | residual-over-tol {res_bad} (worst margin {worst_res:.2e}) | sky-sig-bad {sky_sig_bad} | sig-fired {sig_fired}"
    );
    if dl_bad != 0
        || dc_bad != 0
        || dc_sky_bad != 0
        || mv_bad != 0
        || nrm_bad != 0
        || alb_bad != 0
        || sig_bad != 0
        || res_bad != 0
        || sky_sig_bad != 0
    {
        eprintln!("{tag}: FAIL FSR4-RR feed gates");
        return false;
    }
    if must_fire && (dc_sky_n == 0 || sig_fired == 0) {
        eprintln!("{tag}: FAIL FSR4-RR feed gates vacuous (no sky / no armed sig on the default scene)");
        return false;
    }
    true
}

/// The locked per-session render resolution for a GPU-driven upscaler
/// composition (`--gpu` and `--dxr` — no DRS on either path): quantize the
/// resolved `--lock-res` scale into the wired upscaler's queried input
/// range. `fallback` answers a missing range — and, for RR
/// (`quantize_degenerate = false`), a degenerate opt == min == max range
/// too (the SDK's "DRS off" signal; the fallback is the optimal/DLAA res).
/// XeSS's queried range is always real, so its arm quantizes whenever the
/// query succeeded (`quantize_degenerate = true`).
#[cfg(windows)]
fn locked_render_res(
    lock: f32,
    range: Option<((u32, u32), (u32, u32), (u32, u32))>,
    win: (usize, usize),
    fallback: (usize, usize),
    quantize_degenerate: bool,
) -> (usize, usize) {
    match range {
        Some((_, min, max)) if quantize_degenerate || min != max => xess::quantize_res(
            lock,
            win,
            (min.0 as usize, min.1 as usize),
            (max.0 as usize, max.1 as usize),
        ),
        _ => fallback,
    }
}

/// Headless GPU-tracer gate suite (M1: toolchain + dispatch plumbing).
/// Unlike --check/--check-dlss/--check-xess this needs real hardware: a
/// D3D12 device with RT tier 1.1 and the DXC DLL drop. Exit codes: 0 = all
/// gates pass, 1 = a gate failed, 2 = environment (no DLLs / no support).
#[cfg(windows)]
fn run_check_gpu(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    opts: &Opts,
    must_fire: bool,
) -> i32 {
    let dxc = match gpu::dxc::Dxc::load(&opts.dxc_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("check-gpu: {e}");
            return 2;
        }
    };
    let mut hg = match gpu::trace::HeadlessGpu::new(
        opts.gpu_debug,
        opts.prefer.unwrap_or(gpu::adapter::Prefer::Nvidia),
    ) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("check-gpu: device creation failed: {e}");
            return 2;
        }
    };
    eprintln!("check-gpu: adapter \"{}\"", hg.adapter_name);
    let caps = match gpu::trace::require_caps(&hg.device) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("check-gpu: {e}");
            return 2;
        }
    };
    eprintln!(
        "check-gpu: RT tier {}.{}, shader model {}.{}",
        caps.rt_tier / 10,
        caps.rt_tier % 10,
        caps.shader_model >> 4,
        caps.shader_model & 0xf
    );
    if let Err(e) = gpu::trace::smoke_test(&mut hg, &dxc, opts.gpu_debug) {
        eprintln!("check-gpu: FAIL {e}");
        return 1;
    }
    println!("check-gpu: dispatch plumbing OK (seed -> prep-args -> ExecuteIndirect -> readback)");

    // --- M2: the vanilla GPU reference tracer vs the CPU plain reference ---
    // Exact-zero gates are GPU-vs-GPU only (M3+); CPU-vs-GPU is statistical —
    // hardware watertight triangle intersection differs from moller_trumbore
    // at edges/grazing, and the RNG streams differ by design.
    let (gw, gh) = (800usize, 600usize);
    let dev = hg.device.clone();
    let tg = match gpu::trace::TraceGpu::new(
        &dev,
        &dxc,
        scene,
        bvh,
        gw as u32,
        gh as u32,
        false, // no pack: the M7-M9 gbuf/feed gates build their own tracer
        false,
        opts.gpu_debug,
        opts.bc7,
        &mut hg,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("check-gpu: FAIL TraceGpu init: {e}");
            return 1;
        }
    };
    eprintln!("check-gpu: scene uploaded, BLAS/TLAS built ({} tris)", scene.tri_count());

    let q = Quality::preset(2);
    let basis = cam0.basis(gw, gh);
    let ua = windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let read_f32 = |hg: &mut gpu::trace::HeadlessGpu, res, n: usize| -> Result<Vec<f32>, String> {
        let b = hg.read_buffer(res, ua, n * 4)?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    };

    // CPU counterpart: the plain per-pixel reference (hybrid = false).
    let stats = Stats::default();
    let accum: Vec<AtomicU32> = (0..gw * gh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..gw * gh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..gw * gh).map(|_| AtomicU32::new(0)).collect();
    let cpu_frame = |frame: u32| {
        let ctx = FrameCtx {
            scene,
            bvh,
            cam: basis,
            q,
            frame,
            jitter: frame > 0,
            rw: gw,
            rh: gh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: &[],
            accumulate: true,
            gbuf: None,
            fsr_buf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
            hemi_share: false,
            replay_rec: None,
            cut_cur: None,
            cut_prev: None,
            discard_seeds: false,
            defer_shade: false,
        };
        render::render_frame(&ctx, false);
    };
    let gpu_frame = |hg: &mut gpu::trace::HeadlessGpu,
                     tg: &gpu::trace::TraceGpu,
                     frame: u32|
     -> Result<(), String> {
        tg.write_cb(0, &gpu::trace::FrameParams {
            cam: basis,
            frame,
            accumulate: true,
            jitter: frame > 0,
            frame_jitter: None,
            prev_cam: None,
            q,
            verify: false,
        });
        hg.run(|l| tg.record_reference(l, 0))
    };

    // T1: one unjittered frame each — primary-visibility compare (t + kind).
    if let Err(e) = gpu_frame(&mut hg, &tg, 0) {
        eprintln!("check-gpu: FAIL reference dispatch: {e}");
        return 1;
    }
    let gpu_t = match read_f32(&mut hg, &tg.tbuf, gw * gh) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("check-gpu: FAIL tbuf readback: {e}");
            return 1;
        }
    };
    cpu_frame(0);
    let px = gw * gh;
    let mut class_mismatch = 0usize;
    let mut t_viol = 0usize;
    let mut max_rel = 0.0f32;
    for i in 0..px {
        let ct = f32::from_bits(tbuf[i].load(Relaxed));
        let gt = gpu_t[i];
        match (ct.is_finite(), gt.is_finite()) {
            (true, true) => {
                let rel = (ct - gt).abs() / ct.max(1e-6);
                max_rel = max_rel.max(rel);
                if rel > 1e-3 {
                    t_viol += 1;
                }
            }
            (false, false) => {}
            _ => class_mismatch += 1,
        }
    }
    let mut ok = true;
    eprintln!(
        "check-gpu: reference visibility ({px} px): class-mismatch {class_mismatch} | rel-t > 1e-3: {t_viol} | max rel t err {max_rel:.2e}"
    );
    if class_mismatch as f64 > px as f64 * 5e-4 {
        eprintln!("check-gpu: FAIL hit/sky classification mismatch above 0.05% (two-intersector edge disagreement should be far rarer)");
        ok = false;
    }
    if t_viol as f64 > px as f64 * 1e-4 {
        eprintln!("check-gpu: FAIL primary-t disagreement above 0.01% of pixels");
        ok = false;
    }

    // T2: 64-frame jittered accumulation both sides — radiance A/B.
    // Different RNG streams; only the converged means are comparable.
    const AB_FRAMES: u32 = 64;
    for f in 0..AB_FRAMES {
        if let Err(e) = gpu_frame(&mut hg, &tg, f) {
            eprintln!("check-gpu: FAIL accumulation frame {f}: {e}");
            return 1;
        }
    }
    for f in 0..AB_FRAMES {
        cpu_frame(f);
    }
    let gpu_acc = match read_f32(&mut hg, &tg.accum, px * 3) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("check-gpu: FAIL accum readback: {e}");
            return 1;
        }
    };
    let inv = 1.0 / AB_FRAMES as f32;
    let mut sum_c = [0.0f64; 3];
    let mut sum_g = [0.0f64; 3];
    let mut sum_abs = 0.0f64;
    for i in 0..px * 3 {
        let c = f32::from_bits(accum[i].load(Relaxed)) * inv;
        let g = gpu_acc[i] * inv;
        sum_c[i % 3] += c as f64;
        sum_g[i % 3] += g as f64;
        sum_abs += (c - g).abs() as f64;
    }
    let mut mean_rel = 0.0f64;
    for ch in 0..3 {
        let rel = (sum_c[ch] - sum_g[ch]).abs() / sum_c[ch].max(1e-9);
        mean_rel = mean_rel.max(rel);
    }
    eprintln!(
        "check-gpu: radiance A/B over {AB_FRAMES} frames: per-channel mean rel diff {:.3}% | mean abs px diff {:.4}",
        mean_rel * 100.0,
        sum_abs / (px * 3) as f64
    );
    if mean_rel > 0.02 {
        eprintln!("check-gpu: FAIL converged radiance means differ by more than 2%");
        ok = false;
    }

    // T3: the resolve pass (accum -> RGBA16F, the tonemap PS's input) —
    // texel == accum/samples within f16 precision. This is the present
    // chain's only compute link, verified headlessly.
    {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        };
        if let Err(e) = hg.run(|l| tg.record_resolve(l, 0, AB_FRAMES)) {
            eprintln!("check-gpu: FAIL resolve dispatch: {e}");
            return 1;
        }
        let pitch = gpu::d3d12::aligned_pitch(gw * 8);
        let rb = match gpu::d3d12::ReadbackBuffer::new(&hg.device, pitch * gh) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("check-gpu: FAIL readback alloc: {e}");
                return 1;
            }
        };
        let fp = gpu::d3d12::footprint(
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
            gw as u32,
            gh as u32,
            8,
            0,
        );
        let hdr = tg.hdr.clone();
        if let Err(e) = hg.run(|l| unsafe {
            l.ResourceBarrier(&[gpu::d3d12::transition(
                &hdr,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyTextureRegion(
                &gpu::d3d12::loc_footprint(&rb.resource, fp),
                0,
                0,
                0,
                &gpu::d3d12::loc_subresource(&hdr),
                None,
            );
            l.ResourceBarrier(&[gpu::d3d12::transition(
                &hdr,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }) {
            eprintln!("check-gpu: FAIL hdr readback: {e}");
            return 1;
        }
        let mut ptr = std::ptr::null_mut();
        if let Err(e) = unsafe { rb.resource.Map(0, None, Some(&mut ptr)) } {
            eprintln!("check-gpu: FAIL hdr Map: {e}");
            return 1;
        }
        let mut resolve_viol = 0usize;
        for y in 0..gh {
            let row: &[[half::f16; 4]] = unsafe {
                std::slice::from_raw_parts((ptr as *const u8).add(y * pitch) as *const _, gw)
            };
            for (x, px_v) in row.iter().enumerate() {
                let i3 = (y * gw + x) * 3;
                for ch in 0..3 {
                    let want = gpu_acc[i3 + ch] * inv;
                    let got = f32::from(px_v[ch]);
                    // f16 has ~3 decimal digits; the divide adds one ulp.
                    if (want - got).abs() > want.abs().max(1.0) * 2e-3 {
                        resolve_viol += 1;
                    }
                }
            }
        }
        unsafe { rb.resource.Unmap(0, None) };
        eprintln!("check-gpu: resolve pass: {resolve_viol} texels off accum/samples (f16 tolerance)");
        if resolve_viol > 0 {
            eprintln!("check-gpu: FAIL resolve output disagrees with the accumulation");
            ok = false;
        }
    }

    // --- M3/M4: the wavefront quadtree vs the on-GPU reference -------------
    // Same intersector on both sides, same seeds, same shading code — these
    // are the transplanted exact-zero gates from the CPU --check.
    let read_u32 = |hg: &mut gpu::trace::HeadlessGpu, res, n: usize| -> Result<Vec<u32>, String> {
        let b = hg.read_buffer(res, ua, n * 4)?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    };

    // One unjittered wavefront frame with the coverage sentinel flooded.
    let wf_params = gpu::trace::FrameParams {
        cam: basis,
        frame: 0,
        accumulate: true,
        jitter: false,
        frame_jitter: None,
        prev_cam: None,
        q,
        verify: false,
    };
    tg.write_cb(0, &wf_params);
    if let Err(e) = hg.run(|l| tg.record_wavefront(l, 0, &wf_params, true)) {
        eprintln!("check-gpu: FAIL wavefront dispatch: {e}");
        return 1;
    }
    let (wave_t, wave_info, wave_acc, ctrs) = match (
        read_f32(&mut hg, &tg.tbuf, px),
        read_u32(&mut hg, &tg.info, px),
        read_f32(&mut hg, &tg.accum, px * 3),
        read_u32(&mut hg, &tg.counters, gpu::trace::CTR_COUNT as usize),
    ) {
        (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
        _ => {
            eprintln!("check-gpu: FAIL wavefront readback");
            return 1;
        }
    };
    let (n_leaf, n_sky) =
        (ctrs[gpu::trace::CTR_LEAF as usize] as usize, ctrs[gpu::trace::CTR_SKY as usize] as usize);

    // Queue accounting: leaf + sky rects partition the screen exactly, both
    // tile queues drained, zero overflows.
    let rect_px = |xy0: u32, xy1: u32| -> u64 {
        let (x0, y0) = (xy0 & 0xffff, xy0 >> 16);
        let (x1, y1) = (xy1 & 0xffff, xy1 >> 16);
        (x1 - x0) as u64 * (y1 - y0) as u64
    };
    let leaf_recs = match read_u32(&mut hg, &tg.qleaf, n_leaf.min(tg.cap_leaf as usize) * 4) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("check-gpu: FAIL leaf queue readback: {e}");
            return 1;
        }
    };
    let sky_recs = match read_u32(&mut hg, &tg.qsky, n_sky.min(tg.cap_sky as usize) * 4) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("check-gpu: FAIL sky queue readback: {e}");
            return 1;
        }
    };
    // Clamped bounds: on overflow the counters keep incrementing past the
    // record writes, and the point is to reach the CTR_OVERFLOW FAIL line
    // below with a diagnostic, not to die on an out-of-bounds index here.
    let mut covered: u64 = 0;
    for r in 0..n_leaf.min(leaf_recs.len() / 4) {
        covered += rect_px(leaf_recs[r * 4], leaf_recs[r * 4 + 1]);
    }
    for r in 0..n_sky.min(sky_recs.len() / 4) {
        covered += rect_px(sky_recs[r * 4], sky_recs[r * 4 + 1]);
    }
    let sentinels = wave_info.iter().filter(|&&i| i == 0xffff_ffff).count();
    let tiles_left = ctrs[gpu::trace::CTR_TILE_A as usize].max(ctrs[gpu::trace::CTR_TILE_B as usize]);
    // The last level consumed one tile queue and must have appended nothing.
    let dangling = if tg.depth_full % 2 == 0 {
        ctrs[gpu::trace::CTR_TILE_A as usize]
    } else {
        ctrs[gpu::trace::CTR_TILE_B as usize]
    };
    let _ = tiles_left;
    eprintln!(
        "check-gpu: wavefront frame: leaves {n_leaf} | sky-tiles {n_sky} | splits {} | blocked {} | cuts {} (fallback {}) | overflow {}",
        ctrs[gpu::trace::CTR_SPLIT as usize],
        ctrs[gpu::trace::CTR_BLOCKED as usize],
        ctrs[gpu::trace::CTR_CUT as usize],
        ctrs[gpu::trace::CTR_CUT_FALLBACK as usize],
        ctrs[gpu::trace::CTR_OVERFLOW as usize],
    );
    if ctrs[gpu::trace::CTR_OVERFLOW as usize] != 0 {
        eprintln!("check-gpu: FAIL queue overflow (queues are sized to the structural worst case)");
        ok = false;
    }
    // Alpha-cutout anti-vacuity: an alpha-masked scene's wavefront frame
    // must actually reject candidates, or the cutout path is dead code
    // (scene-derived, independent of `structural`). Caveat: a custom --cam
    // pose whose view contains no masked geometry would trip this — the
    // CLAUDE.md canopy-caveat class; the default OBJ poses see foliage.
    let alpha_rej = ctrs[gpu::trace::CTR_ALPHA_REJ as usize];
    if scene.any_alpha {
        eprintln!("check-gpu: alpha-cutout rejections: {alpha_rej}");
        if alpha_rej == 0 {
            eprintln!("check-gpu: FAIL alpha-masked scene rejected 0 candidates (cutout must fire)");
            ok = false;
        }
    } else if alpha_rej != 0 {
        eprintln!(
            "check-gpu: FAIL {alpha_rej} alpha rejections on an opaque scene (ALPHA_CUTOUT must be compiled out)"
        );
        ok = false;
    }
    if dangling != 0 {
        eprintln!("check-gpu: FAIL {dangling} tile records left after the last level (depth accounting)");
        ok = false;
    }
    if covered != px as u64 {
        eprintln!("check-gpu: FAIL leaf+sky rects cover {covered} px, screen has {px}");
        ok = false;
    }
    if sentinels != 0 {
        eprintln!("check-gpu: FAIL {sentinels} px never written (exactly-once coverage)");
        ok = false;
    }

    // Reference frame with identical constants -> exact-zero pixel gates.
    if let Err(e) = gpu_frame(&mut hg, &tg, 0) {
        eprintln!("check-gpu: FAIL reference re-run: {e}");
        return 1;
    }
    let (ref_t, ref_acc) =
        match (read_f32(&mut hg, &tg.tbuf, px), read_f32(&mut hg, &tg.accum, px * 3)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => {
                eprintln!("check-gpu: FAIL reference readback");
                return 1;
            }
        };
    let mut false_sky = 0usize;
    let mut overshoot = 0usize;
    let mut hybrid_extra = 0usize;
    let mut max_rel_t = 0.0f32;
    for i in 0..px {
        let (rt, wt) = (ref_t[i], wave_t[i]);
        match (rt.is_finite(), wt.is_finite()) {
            (true, true) => {
                let rel = (wt - rt) / rt.max(1e-6);
                max_rel_t = max_rel_t.max(rel.abs());
                if rel > 1e-4 {
                    overshoot += 1; // wavefront started past real geometry
                }
            }
            (true, false) => false_sky += 1,
            (false, true) => hybrid_extra += 1,
            (false, false) => {}
        }
    }
    // Same-seed same-shading image A/B: identical hits => identical RNG
    // streams => near-identical color (cross-kernel compilation fp only).
    let mut img_sum = 0.0f64;
    let mut img_max = 0.0f32;
    for i in 0..px * 3 {
        let d = (wave_acc[i] - ref_acc[i]).abs();
        img_sum += d as f64;
        img_max = img_max.max(d);
    }
    let img_mean = img_sum / (px * 3) as f64;
    eprintln!(
        "check-gpu: wavefront vs reference ({px} px): false-sky {false_sky} | tmin-overshoot {overshoot} | hybrid-extra {hybrid_extra} | max rel t err {max_rel_t:.2e} | same-seed image mean |d| {img_mean:.2e} max {img_max:.2e}"
    );
    if false_sky != 0 || overshoot != 0 || hybrid_extra != 0 {
        eprintln!("check-gpu: FAIL wavefront visibility gates (the inherited-tmin bug class)");
        ok = false;
    }
    if img_mean > 1e-5 || img_max > 1e-2 {
        eprintln!("check-gpu: FAIL same-seed wavefront/reference images diverge");
        ok = false;
    }
    if must_fire {
        // Structural must-fires, default scene only (mirrors --check).
        if n_sky == 0 || n_leaf == 0 || ctrs[gpu::trace::CTR_BLOCKED as usize] == 0 {
            eprintln!("check-gpu: FAIL structural must-fires (sky/leaf/blocked all expected > 0 on the default scene)");
            ok = false;
        }
    }

    // --- M5: the hemisphere AO/GI wavefront on a deterministic probe set ---
    // Probes are CPU-generated (center rays, surface_point) so both sides
    // integrate at the exact same (o, n). The exact-zero claim gates run on
    // the GPU (FLAG_VERIFY: false-empty / tmin-overshoot re-validated with
    // RayQuery reference rays, PSA accounting in H.w, sampled cut-bound);
    // the A/Bs compare against CPU cosine-sampled references (statistical —
    // different RNG streams by design).
    let mut probes: Vec<(Vec3A, Vec3A)> = Vec::new();
    {
        let mut vls = stats::LocalStats::default();
        let mut y = 7usize;
        while y < gh && probes.len() < 512 {
            let mut x = 11usize;
            while x < gw && probes.len() < 512 {
                let ray = bvh::Ray::new(basis.origin, basis.ray_dir(x as f32 + 0.5, y as f32 + 0.5));
                if let Some(h) = bvh.intersect(scene, &ray, 0.0, f32::INFINITY, &mut vls.ray_nodes) {
                    probes.push(shade::surface_point(scene, &ray, &h));
                }
                x += 53;
            }
            y += 41;
        }
    }
    eprintln!("check-gpu: hemi probe set: {} points", probes.len());
    let mut hemi_ok = true;
    for (mode_name, fb) in [
        ("AO", shade::FrustumBounce { ao: true, gi: false, depth: 3 }),
        ("GI", shade::FrustumBounce { ao: false, gi: true, depth: 3 }),
    ] {
        // Multi-seed estimate (the CB frame seeds the Arvo draws), matching
        // the CPU suite's A/B. The verify/stat counters ACCUMULATE across
        // seeds (cs_seed_probes keeps them on the clear=false passes), so
        // the exact-zero gates observe every seed's rays, not just the
        // last seed's; PSA totals SEEDS·pi.
        const SEEDS: u32 = 8;
        let hq = Quality { fb, ..q };
        for s in 0..SEEDS {
            tg.write_cb(0, &gpu::trace::FrameParams {
                cam: basis,
                frame: s,
                accumulate: true,
                jitter: false,
                frame_jitter: None,
                prev_cam: None,
                q: hq,
                verify: true,
            });
            if let Err(e) = tg.run_hemi_probes(&mut hg, 0, &probes, fb.depth, s == 0) {
                eprintln!("check-gpu: FAIL hemi {mode_name} probes: {e}");
                return 1;
            }
        }
        let h = match read_u32(&mut hg, &tg.hbuf, probes.len() * 4) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("check-gpu: FAIL hbuf readback: {e}");
                return 1;
            }
        };
        let vctrs = match read_u32(&mut hg, &tg.counters, gpu::trace::CTR_COUNT as usize) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("check-gpu: FAIL counters readback: {e}");
                return 1;
            }
        };
        const FIXED: f64 = 262144.0;
        let seeds_f = SEEDS as f64;
        let mut psa_viol = 0usize;
        let mut max_psa_err = 0.0f64;
        for i in 0..probes.len() {
            let acc = h[i * 4 + 3] as f64 / FIXED / seeds_f;
            let err = (acc - std::f64::consts::PI).abs();
            max_psa_err = max_psa_err.max(err);
            if err > 1e-3 {
                psa_viol += 1;
            }
        }
        let (fe, tm) = (
            vctrs[gpu::trace::CTR_V_FALSE_EMPTY as usize],
            vctrs[gpu::trace::CTR_V_TMIN as usize],
        );
        eprintln!(
            "check-gpu: hemi {mode_name} gates ({} probes): psa-viol {psa_viol} (max err {max_psa_err:.2e}) | false-empty {fe} | tmin-overshoot {tm} | empty-cells {} | leaf-rays {}",
            probes.len(),
            vctrs[gpu::trace::CTR_HEMI_EMPTY as usize],
            vctrs[gpu::trace::CTR_HEMI_RAYS as usize],
        );
        if psa_viol != 0 || fe != 0 || tm != 0 {
            eprintln!("check-gpu: FAIL hemi {mode_name} exact-zero gates");
            hemi_ok = false;
        }
        if must_fire
            && (vctrs[gpu::trace::CTR_HEMI_EMPTY as usize] == 0
                || vctrs[gpu::trace::CTR_HEMI_RAYS as usize] == 0)
        {
            eprintln!("check-gpu: FAIL hemi {mode_name} must-fires (empty cells and leaf rays both expected > 0)");
            hemi_ok = false;
        }

        // A/B vs a CPU cosine-sampled reference at the same points. AO also
        // gates the SIGNED mean (the estimator is unbiased); GI runs the
        // same depth-1 BOUNCE_Q policy as the GPU leaf rays.
        const REF_N: u32 = 4096;
        let mut rng = fastrand::Rng::with_seed(0x1234_5678);
        let mut vls = stats::LocalStats::default();
        if mode_name == "AO" {
            let mut sum_abs = 0.0f64;
            let mut sum_signed = 0.0f64;
            for (i, &(o, n)) in probes.iter().enumerate() {
                let gpu_ao = h[i * 4] as f64 / FIXED / seeds_f / std::f64::consts::PI;
                let (t1, t2) = shade::onb(n);
                let mut open = 0u32;
                for _ in 0..REF_N {
                    let d = shade::cosine_dir(n, t1, t2, rng.f32(), rng.f32());
                    if !bvh.occluded(scene, &bvh::Ray::new(o, d), 0.0, scene.ao_radius, &mut vls.ray_nodes) {
                        open += 1;
                    }
                }
                let cpu_ao = open as f64 / REF_N as f64;
                sum_abs += (gpu_ao - cpu_ao).abs();
                sum_signed += gpu_ao - cpu_ao;
            }
            let mean_abs = sum_abs / probes.len() as f64;
            let mean_signed = sum_signed / probes.len() as f64;
            eprintln!(
                "check-gpu: hemi AO A/B vs {REF_N}-sample cosine reference: mean |d| {mean_abs:.4} | signed mean {mean_signed:+.4}"
            );
            if mean_abs >= 0.02 || mean_signed.abs() >= 0.005 {
                eprintln!("check-gpu: FAIL hemi AO A/B (bias or error above the CPU-suite tolerances)");
                hemi_ok = false;
            }
        } else {
            let mut sum_rel = 0.0f64;
            let sun = render::sun_dir(scene);
            for (i, &(o, n)) in probes.iter().enumerate() {
                let gpu_gi = Vec3A::new(
                    h[i * 4] as f32 / FIXED as f32,
                    h[i * 4 + 1] as f32 / FIXED as f32,
                    h[i * 4 + 2] as f32 / FIXED as f32,
                ) / SEEDS as f32
                    / std::f32::consts::PI;
                let (t1, t2) = shade::onb(n);
                let mut sum = Vec3A::ZERO;
                for _ in 0..REF_N {
                    let d = shade::cosine_dir(n, t1, t2, rng.f32(), rng.f32());
                    let ray = bvh::Ray::new(o, d);
                    sum += match bvh.intersect(scene, &ray, 0.0, f32::INFINITY, &mut vls.ray_nodes) {
                        // dome, NOT radiance — this reference must integrate the
                        // same sky hemi.rs does (a GATHER path), or the GI A/B
                        // below is comparing two different functions.
                        None => crate::sky::dome(d, sun),
                        Some(hh) => shade::shade(
                            scene,
                            bvh,
                            &ray,
                            &hh,
                            None,
                            &hemi::BOUNCE_Q,
                            &mut rng,
                            sun,
                            // Same bounce cone as hemi.rs's leaf shades — the
                            // A/B needs both arms sampling textures alike.
                            shade::Cone::bounce(),
                            1,
                            &mut vls,
                            None,
                            shade::VisCtl::Off,
                            None,
                        ),
                    };
                }
                let cpu_gi = sum / REF_N as f32;
                let rel = (gpu_gi - cpu_gi).length() as f64 / cpu_gi.length().max(1e-6) as f64;
                sum_rel += rel;
            }
            let mean_rel = sum_rel / probes.len() as f64;
            eprintln!(
                "check-gpu: hemi GI A/B vs {REF_N}-sample BOUNCE_Q reference: mean rel {:.2}%",
                mean_rel * 100.0
            );
            if mean_rel >= 0.05 {
                eprintln!("check-gpu: FAIL hemi GI A/B (above the CPU-suite 5% tolerance)");
                hemi_ok = false;
            }
        }
    }
    if !hemi_ok {
        ok = false;
    }

    // Frame-level hemi: one full wavefront frame with GI on — the leaf pass
    // must append exactly one point per hit pixel, the batch loop must drain
    // them, and compose must produce finite radiance everywhere.
    {
        let hq = Quality {
            fb: shade::FrustumBounce { ao: false, gi: true, depth: 2 },
            ..q
        };
        let p = gpu::trace::FrameParams {
            cam: basis,
            frame: 0,
            accumulate: true,
            jitter: false,
            frame_jitter: None,
            prev_cam: None,
            q: hq,
            verify: false,
        };
        tg.write_cb(0, &p);
        if let Err(e) = hg.run(|l| tg.record_wavefront(l, 0, &p, false)) {
            eprintln!("check-gpu: FAIL hemi frame dispatch: {e}");
            return 1;
        }
        let (acc, t_w, ctrs2) = match (
            read_f32(&mut hg, &tg.accum, px * 3),
            read_f32(&mut hg, &tg.tbuf, px),
            read_u32(&mut hg, &tg.counters, gpu::trace::CTR_COUNT as usize),
        ) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            _ => {
                eprintln!("check-gpu: FAIL hemi frame readback");
                return 1;
            }
        };
        let hits = t_w.iter().filter(|t| t.is_finite()).count();
        let pts = ctrs2[gpu::trace::CTR_HEMI_PT as usize] as usize;
        let nonfinite = acc.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "check-gpu: hemi GI frame: hit-px {hits} | hemi-points {pts} | hemi-rays {} | overflow {} | non-finite {nonfinite}",
            ctrs2[gpu::trace::CTR_HEMI_RAYS as usize],
            ctrs2[gpu::trace::CTR_OVERFLOW as usize],
        );
        if pts != hits {
            eprintln!("check-gpu: FAIL hemi point count != hit pixels (leaf-pass append accounting)");
            ok = false;
        }
        if ctrs2[gpu::trace::CTR_OVERFLOW as usize] != 0 || nonfinite != 0 {
            eprintln!("check-gpu: FAIL hemi frame overflow/non-finite");
            ok = false;
        }
    }

    // --- M7: GPU-born G-buffers — MV/depth/matrix gates on the pack ---
    // The GPU twin of mv_check_at: frame A (no prev), a 0.02*diag forward
    // dolly, frame B (prev = basis A). The pack is read back, unpacked into
    // CPU GBufs, and gated by the EXACT existing dlss::mv_selftest — zero new
    // tolerances. Odd render dims mirror --check-xess's odd-dimension sweep.
    {
        let (pw, ph) = (533usize, 400usize);
        let dev = hg.device.clone();
        let mut ptg = match gpu::trace::TraceGpu::new(
            &dev,
            &dxc,
            scene,
            bvh,
            pw as u32,
            ph as u32,
            true,
            true, // + the NPPD staging buffers/kernels (M10 gates them)
            opts.gpu_debug,
            opts.bc7,
            &mut hg,
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("check-gpu: FAIL gbuf TraceGpu init: {e}");
                return 1;
            }
        };
        let (near, far) = dlss::near_far(scene.diag);
        let uq = Quality::upscaler_1spp();
        // One fresh 1-spp wavefront frame at the upscaler contract
        // (accumulate off, frame-uniform zero jitter), pack read back into a
        // CPU GBufs so the CPU gate consumes it unmodified. Returns the pack
        // and tbuf (for sky classification).
        let gpu_gbuf_frame = |hg: &mut gpu::trace::HeadlessGpu,
                              basis: camera::CamBasis,
                              prev: Option<camera::CamBasis>,
                              frame: u32|
         -> Result<(dlss::GBufs, Vec<f32>), String> {
            let p = gpu::trace::FrameParams {
                cam: basis,
                frame,
                accumulate: false,
                jitter: false,
                frame_jitter: Some((0.0, 0.0)),
                prev_cam: prev,
                q: uq,
                verify: false,
            };
            ptg.write_cb(0, &p);
            hg.run(|l| ptg.record_wavefront(l, 0, &p, false))?;
            let bytes = hg.read_buffer(&ptg.gbuf, ua, pw * ph * gpu::trace::GBUF_STRIDE as usize)?;
            let tb = hg.read_buffer(&ptg.tbuf, ua, pw * ph * 4)?;
            let t: Vec<f32> =
                tb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            Ok((unpack_gbuf_bytes(&bytes, pw, ph), t))
        };
        let basis_a = cam0.basis(pw, ph);
        let mut cam_b = cam0;
        cam_b.pos += cam0.forward() * (0.02 * scene.diag);
        let basis_b = cam_b.basis(pw, ph);
        let (ga, ta) = match gpu_gbuf_frame(&mut hg, basis_a, None, 0) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("check-gpu: FAIL gbuf frame A: {e}");
                return 1;
            }
        };
        let (gb2, tb2) = match gpu_gbuf_frame(&mut hg, basis_b, Some(basis_a), 1) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("check-gpu: FAIL gbuf frame B: {e}");
                return 1;
            }
        };
        // Structural pack coverage: every pixel carries a positive view-Z
        // (exactly-once coverage rides the sentinel gates; this proves the
        // gbuf writes ran everywhere), and sky depth is far BIT-EQUAL (the
        // write helper stores the CB constant untouched).
        let loadz = |g: &dlss::GBufs, i: usize| {
            f32::from_bits(g.depth[i].load(std::sync::atomic::Ordering::Relaxed))
        };
        let (mut bad_z, mut sky_off, mut skies) = (0usize, 0usize, 0usize);
        for i in 0..pw * ph {
            let z = loadz(&ga, i);
            if !(z > 0.0) {
                bad_z += 1;
            }
            if !ta[i].is_finite() {
                skies += 1;
                if z.to_bits() != far.to_bits() {
                    sky_off += 1;
                }
            }
        }
        let mv_ok =
            dlss::mv_selftest(&ga, &basis_a, &gb2, &basis_b, &dlss::cam_matrices(&cam_b, pw, ph, near, far), scene.diag, far);
        eprintln!(
            "check-gpu: gbuf pack ({pw}x{ph}): view-z<=0 {bad_z} | sky-depth-off {sky_off} (sky px {skies}) | mv/depth/matrix {}",
            if mv_ok { "OK" } else { "FAIL" },
        );
        if !mv_ok || bad_z != 0 || sky_off != 0 {
            eprintln!("check-gpu: FAIL GPU-born G-buffer gates");
            ok = false;
        }
        // Sky gate anti-vacuity: with zero sky pixels the bit-equal gate
        // proves nothing — the default view must contain sky (mirrors M3's
        // n_sky must-fire; --stress skips).
        if must_fire && skies == 0 {
            eprintln!("check-gpu: FAIL gbuf sky gate vacuous (no sky pixels on the default scene)");
            ok = false;
        }
        // Textured scenes: the pack's albedo plane vs a CPU render (the
        // guide-chain proof — RR/XeSS/FSR/NPPD all read this plane).
        if !albedo_ab_check(scene, bvh, cam0, &ga, &ta, pw, ph, "gpu") {
            ok = false;
        }

        // --- M11: --bc7 encode fidelity (GPU decode vs the CPU RGBA8 source).
        // Only meaningful with the flag on: it measures the ENCODER, per
        // texel, through the spec-bit-exact hardware decoder — the number the
        // statistical albedo/radiance gates can't give. The 25 dB limit is a
        // WIRING gate, not a quality bar: a pitch/footprint/format error
        // lands ~10-20 dB, while the worst honest texture measured 31.7 dB at
        // `fast` (San Miguel) — 25 separates the two without false-failing a
        // hard texture at `ultrafast`.
        if let Some(bq) = opts.bc7 {
            match gpu::trace::bc7_fidelity(scene, bq, &mut hg) {
                Ok(Some(f)) => {
                    eprintln!(
                        "check-gpu: bc7 fidelity: {} textures | mean |d| {:.3} LSB | max {} | worst PSNR {:.1} dB (limit 25)",
                        f.textures, f.mean_abs, f.max_abs, f.worst_psnr
                    );
                    if f.worst_psnr < 25.0 {
                        eprintln!("check-gpu: FAIL bc7 fidelity below 25 dB (encode/upload wiring?)");
                        ok = false;
                    }
                }
                Ok(None) => eprintln!("check-gpu: bc7 fidelity skipped (no compressible textures)"),
                Err(e) => {
                    eprintln!("check-gpu: FAIL bc7 fidelity: {e}");
                    ok = false;
                }
            }
        }

        // --- M8: the XeSS feed kernel — depth-encode + mvec plumbing gates ---
        // Wire a headless XessResources' planes as feed targets, run the feed
        // over frame B's pack (still resident on the GPU), read the planes
        // back, and gate against the CPU contracts: the depth plane vs
        // xess::view_z_to_clip_depth of the pack's view_z (hits <= 4 f32 ulp —
        // D3D's divide tolerance; sky BIT-EQUAL 0.0, the `precise` encode's
        // exact-zero numerator), the mvec plane vs the pack's mv within
        // 1 f16 ulp (plumbing + f16-rounding proof).
        {
            let xres = match gpu::xr::XessResources::new(
                &hg.device,
                pw as u32,
                ph as u32,
                pw as u32,
                ph as u32,
            ) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("check-gpu: FAIL feed XessResources: {e}");
                    return 1;
                }
            };
            let pl = xres.plane_resources();
            if let Err(e) = ptg.wire_feed(
                &hg.device,
                gpu::trace::FeedKind::Xess,
                &[
                    (gpu::trace::FEED_COLOR, pl[0].0, pl[0].1),
                    (gpu::trace::FEED_MVEC, pl[1].0, pl[1].1),
                    (gpu::trace::FEED_DEPTH, pl[2].0, pl[2].1),
                ],
            ) {
                eprintln!("check-gpu: FAIL feed wiring: {e}");
                return 1;
            }
            let mut feed_rec = Ok(());
            if let Err(e) = hg.run(|l| feed_rec = ptg.record_feed(l, 0, false)) {
                eprintln!("check-gpu: FAIL feed dispatch submit: {e}");
                return 1;
            }
            if let Err(e) = feed_rec {
                eprintln!("check-gpu: FAIL feed dispatch: {e}");
                return 1;
            }
            let depth_bytes = match read_feed_tex(&mut hg, pl[2].0, pl[2].1, 4, pw, ph) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("check-gpu: FAIL depth plane readback: {e}");
                    return 1;
                }
            };
            let mvec_bytes = match read_feed_tex(&mut hg, pl[1].0, pl[1].1, 4, pw, ph) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("check-gpu: FAIL mvec plane readback: {e}");
                    return 1;
                }
            };
            let color_bytes = match read_feed_tex(&mut hg, pl[0].0, pl[0].1, 8, pw, ph) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("check-gpu: FAIL color plane readback: {e}");
                    return 1;
                }
            };
            // Frame B's 1-spp store — the feed's color source, read back once
            // for the color-plane gates here and in M9 (a plane-order swap in
            // the wiring tables would otherwise pass the depth/mvec gates and
            // surface only as image garbage).
            let accum_bytes = match hg.read_buffer(&ptg.accum, ua, pw * ph * 12) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("check-gpu: FAIL accum readback: {e}");
                    return 1;
                }
            };
            if !gate_xess_feed(
                "check-gpu",
                pw,
                ph,
                &depth_bytes,
                &mvec_bytes,
                &color_bytes,
                &accum_bytes,
                &gb2,
                &tb2,
                near,
                far,
                must_fire,
            ) {
                ok = false;
            }

            // --- M8b: the FSR3 feed — the same kernel/encodes as the XeSS
            // trio (FeedKind::Fsr3 -> cs_feed_xess), rewired over the FSR
            // 3.1 flavor's UAV-capable planes and gated identically, so a
            // wiring-order swap in the FSR arm cannot pass the suite.
            {
                let fres = match gpu::ffx_up::Fsr3Resources::new(
                    &hg.device,
                    pw as u32,
                    ph as u32,
                    pw as u32,
                    ph as u32,
                ) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("check-gpu: FAIL feed Fsr3Resources: {e}");
                        return 1;
                    }
                };
                let fpl = fres.plane_resources();
                if let Err(e) = ptg.wire_feed(
                    &hg.device,
                    gpu::trace::FeedKind::Fsr3,
                    &[
                        (gpu::trace::FEED_COLOR, fpl[0].0, fpl[0].1),
                        (gpu::trace::FEED_MVEC, fpl[1].0, fpl[1].1),
                        (gpu::trace::FEED_DEPTH, fpl[2].0, fpl[2].1),
                    ],
                ) {
                    eprintln!("check-gpu: FAIL FSR3 feed wiring: {e}");
                    return 1;
                }
                let mut f_rec = Ok(());
                if let Err(e) = hg.run(|l| f_rec = ptg.record_feed(l, 0, false)) {
                    eprintln!("check-gpu: FAIL FSR3 feed dispatch submit: {e}");
                    return 1;
                }
                if let Err(e) = f_rec {
                    eprintln!("check-gpu: FAIL FSR3 feed dispatch: {e}");
                    return 1;
                }
                let (f_depth, f_mvec, f_color) = match (
                    read_feed_tex(&mut hg, fpl[2].0, fpl[2].1, 4, pw, ph),
                    read_feed_tex(&mut hg, fpl[1].0, fpl[1].1, 4, pw, ph),
                    read_feed_tex(&mut hg, fpl[0].0, fpl[0].1, 8, pw, ph),
                ) {
                    (Ok(d), Ok(m), Ok(c)) => (d, m, c),
                    _ => {
                        eprintln!("check-gpu: FAIL FSR3 feed plane readback");
                        return 1;
                    }
                };
                if !gate_xess_feed(
                    "check-gpu fsr3",
                    pw,
                    ph,
                    &f_depth,
                    &f_mvec,
                    &f_color,
                    &accum_bytes,
                    &gb2,
                    &tb2,
                    near,
                    far,
                    must_fire,
                ) {
                    ok = false;
                }
            }

            // --- M9: the RR feed kernel — depth plumbing gate ---
            // Rewire the same tracer to a headless RrResources' 7 planes,
            // re-run the feed, and gate the linear-depth plane BIT-EQUAL to
            // the pack's view_z (R32F passthrough — no arithmetic, so no
            // ulp allowance), plus the RR mvec plane at the same f16 bound.
            let rres = match gpu::rr::RrResources::new(
                &hg.device,
                (pw as u32, ph as u32),
                (pw as u32, ph as u32),
                (pw as u32, ph as u32),
                pw as u32,
                ph as u32,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("check-gpu: FAIL feed RrResources: {e}");
                    return 1;
                }
            };
            let rpl = rres.plane_resources();
            if let Err(e) = ptg.wire_feed(
                &hg.device,
                gpu::trace::FeedKind::Rr,
                &[
                    (gpu::trace::FEED_COLOR, rpl[0].0, rpl[0].1),
                    (gpu::trace::FEED_NR, rpl[1].0, rpl[1].1),
                    (gpu::trace::FEED_DEPTH, rpl[2].0, rpl[2].1),
                    (gpu::trace::FEED_MVEC, rpl[3].0, rpl[3].1),
                    (gpu::trace::FEED_ALB, rpl[4].0, rpl[4].1),
                    (gpu::trace::FEED_SPEC, rpl[5].0, rpl[5].1),
                    (gpu::trace::FEED_SPECHIT, rpl[6].0, rpl[6].1),
                ],
            ) {
                eprintln!("check-gpu: FAIL RR feed wiring: {e}");
                return 1;
            }
            let mut rr_rec = Ok(());
            if let Err(e) = hg.run(|l| rr_rec = ptg.record_feed(l, 0, false)) {
                eprintln!("check-gpu: FAIL RR feed dispatch submit: {e}");
                return 1;
            }
            if let Err(e) = rr_rec {
                eprintln!("check-gpu: FAIL RR feed dispatch: {e}");
                return 1;
            }
            // RR plane order: color(0), normal_rough(1), depth(2), mvec(3),
            // albedo(4), spec_albedo(5), spec_hit(6) — every plane gated, so
            // an order swap in either wiring table cannot pass the suite.
            let mut read_plane = |idx: usize, bpp: usize, what: &str| -> Option<Vec<u8>> {
                match read_feed_tex(&mut hg, rpl[idx].0, rpl[idx].1, bpp, pw, ph) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        eprintln!("check-gpu: FAIL RR {what} plane readback: {e}");
                        None
                    }
                }
            };
            let (Some(rr_color), Some(rr_nr), Some(rr_depth), Some(rr_mvec)) = (
                read_plane(0, 8, "color"),
                read_plane(1, 8, "normal_rough"),
                read_plane(2, 4, "depth"),
                read_plane(3, 4, "mvec"),
            ) else {
                return 1;
            };
            let (Some(rr_alb), Some(rr_spec), Some(rr_spechit)) = (
                read_plane(4, 4, "albedo"),
                read_plane(5, 4, "spec_albedo"),
                read_plane(6, 2, "spec_hit"),
            ) else {
                return 1;
            };
            if !gate_rr_feed(
                "check-gpu",
                pw,
                ph,
                &rr_color,
                &rr_nr,
                &rr_depth,
                &rr_mvec,
                &rr_alb,
                &rr_spec,
                &rr_spechit,
                &accum_bytes,
                &gb2,
            ) {
                ok = false;
            }

            // --- M9b: the FSR4-RR feed. Wiring FeedKind::FsrRr arms
            // FLAG_FSR_SIG, so frame B is RE-TRACED with the same params:
            // accum must come back BIT-IDENTICAL (the sig capture is
            // assignment-only, zero rng draws), and the nine planes gate
            // against oracles from the armed pack's readback.
            {
                let fres = match gpu::ffx_rr::FsrResources::new(
                    &hg.device,
                    pw as u32,
                    ph as u32,
                    pw as u32,
                    ph as u32,
                ) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("check-gpu: FAIL feed FsrResources: {e}");
                        return 1;
                    }
                };
                let fpl = fres.plane_resources();
                if let Err(e) = ptg.wire_feed(
                    &hg.device,
                    gpu::trace::FeedKind::FsrRr,
                    &[
                        (gpu::trace::FEED_SPECHIT, fpl[0].0, fpl[0].1),
                        (gpu::trace::FEED_DEPTH, fpl[1].0, fpl[1].1),
                        (gpu::trace::FEED_FSR_MVEC, fpl[2].0, fpl[2].1),
                        (gpu::trace::FEED_NR, fpl[3].0, fpl[3].1),
                        (gpu::trace::FEED_ALB, fpl[4].0, fpl[4].1),
                        (gpu::trace::FEED_SPEC, fpl[5].0, fpl[5].1),
                        (gpu::trace::FEED_FSR_DD, fpl[6].0, fpl[6].1),
                        (gpu::trace::FEED_FSR_DS, fpl[7].0, fpl[7].1),
                        (gpu::trace::FEED_COLOR, fpl[8].0, fpl[8].1),
                    ],
                ) {
                    eprintln!("check-gpu: FAIL FSR4-RR feed wiring: {e}");
                    return 1;
                }
                let p = gpu::trace::FrameParams {
                    cam: basis_b,
                    frame: 1,
                    accumulate: false,
                    jitter: false,
                    frame_jitter: Some((0.0, 0.0)),
                    prev_cam: Some(basis_a),
                    q: uq,
                    verify: false,
                };
                ptg.write_cb(0, &p);
                if let Err(e) = hg.run(|l| ptg.record_wavefront(l, 0, &p, false)) {
                    eprintln!("check-gpu: FAIL FSR4-RR re-trace: {e}");
                    return 1;
                }
                let (pack2, accum2) = match (
                    hg.read_buffer(&ptg.gbuf, ua, pw * ph * gpu::trace::GBUF_STRIDE as usize),
                    hg.read_buffer(&ptg.accum, ua, pw * ph * 12),
                ) {
                    (Ok(a), Ok(b)) => (a, b),
                    _ => {
                        eprintln!("check-gpu: FAIL FSR4-RR pack/accum readback");
                        return 1;
                    }
                };
                if accum2 != accum_bytes {
                    eprintln!(
                        "check-gpu: FAIL FSR-sig on/off accum not bit-identical (the sig capture changed shading)"
                    );
                    ok = false;
                }
                let mut f_rec = Ok(());
                if let Err(e) = hg.run(|l| f_rec = ptg.record_feed(l, 0, false)) {
                    eprintln!("check-gpu: FAIL FSR4-RR feed dispatch submit: {e}");
                    return 1;
                }
                if let Err(e) = f_rec {
                    eprintln!("check-gpu: FAIL FSR4-RR feed dispatch: {e}");
                    return 1;
                }
                let mut read_plane = |idx: usize, bpp: usize, what: &str| -> Option<Vec<u8>> {
                    match read_feed_tex(&mut hg, fpl[idx].0, fpl[idx].1, bpp, pw, ph) {
                        Ok(v) => Some(v),
                        Err(e) => {
                            eprintln!("check-gpu: FAIL FSR4-RR {what} plane readback: {e}");
                            None
                        }
                    }
                };
                let (Some(f_dlin), Some(f_dclip), Some(f_mvec), Some(f_nrm)) = (
                    read_plane(0, 4, "depth_lin"),
                    read_plane(1, 4, "depth_clip"),
                    read_plane(2, 8, "mvec"),
                    read_plane(3, 4, "normals"),
                ) else {
                    return 1;
                };
                let (Some(f_alb), Some(f_spec), Some(f_dd), Some(f_ds), Some(f_res)) = (
                    read_plane(4, 4, "diff_alb"),
                    read_plane(5, 4, "spec_alb"),
                    read_plane(6, 8, "dd"),
                    read_plane(7, 8, "ds"),
                    read_plane(8, 8, "residual"),
                ) else {
                    return 1;
                };
                if !gate_fsr_rr_feed(
                    "check-gpu",
                    pw,
                    ph,
                    &f_dlin,
                    &f_dclip,
                    &f_mvec,
                    &f_nrm,
                    &f_alb,
                    &f_spec,
                    &f_dd,
                    &f_ds,
                    &f_res,
                    &pack2,
                    &accum2,
                    near,
                    far,
                    must_fire,
                ) {
                    ok = false;
                }
            }

            // --- M10: the NPPD GPU staging kernels + DML interop ---
            // The pack/warp kernels are term-for-term ports of
            // nppd::pack_inputs / nppd::warp_temporal — gate them against the
            // CPU oracles running on the SAME readback inputs (gb2 = frame
            // B's pack, accum_bytes = its 1-spp store). DLL-free: the kernels
            // are plain compute. The end-to-end interop gate (ORT executing
            // on hg's queue over these buffers) runs only when the runtime
            // DLLs + the exported model exist.
            {
                use gpu::d3d12::transition;
                let (npw, nph) = nppd::pad_dims(pw, ph);
                let npp = npw * nph;
                let accum_at: Vec<AtomicU32> = accum_bytes
                    .chunks_exact(4)
                    .map(|c| AtomicU32::new(u32::from_le_bytes(c.try_into().unwrap())))
                    .collect();
                let nres = ptg.nppd.as_ref().expect("M7 tracer built with nppd");
                let (nfr, nst, nwp, nout) = (
                    nres.frame.clone(),
                    nres.state.clone(),
                    nres.warped.clone(),
                    nres.out.clone(),
                );

                // Pack + zero (state_valid = false). CB slot 0 still holds
                // frame B's constants.
                let mut pre = Ok(());
                if hg.run(|l| pre = ptg.record_nppd_pre(l, 0, false)).is_err() || pre.is_err() {
                    eprintln!("check-gpu: FAIL nppd staging dispatch: {pre:?}");
                    return 1;
                }
                let readf = |hg: &mut gpu::trace::HeadlessGpu,
                             res: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
                             n: usize|
                 -> Result<Vec<f32>, String> {
                    let b = hg.read_buffer(res, ua, n * 4)?;
                    Ok(b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect())
                };
                let gpu_frame = match readf(&mut hg, &nfr, nppd::CH_FRAME * npp) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("check-gpu: FAIL nppd frame readback: {e}");
                        return 1;
                    }
                };
                let gpu_warped0 = match readf(&mut hg, &nwp, nppd::C_T * npp) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("check-gpu: FAIL nppd warped readback: {e}");
                        return 1;
                    }
                };
                let mut cpu_frame = vec![0.0f32; nppd::CH_FRAME * npp];
                nppd::pack_inputs(&accum_at, &gb2, &basis_b, far, npw, nph, &mut cpu_frame);
                // ch0 (log-depth: ray_dir + divide + log — transcendental
                // slop) and ch1-3 (normal rotation: normalize slop) get small
                // absolute tolerances; ch4-9 are pure copies of the same
                // readback values — BIT-equal. Sky ch0 must be exactly 0 on
                // both sides.
                let (mut p_d, mut p_n, mut p_c, mut p_sky, mut sky_n) =
                    (0usize, 0usize, 0usize, 0usize, 0usize);
                let mut max_d = 0.0f32;
                for i in 0..npp {
                    let (x, y) = (i % npw, i / npw);
                    let si = y.min(ph - 1) * pw + x.min(pw - 1);
                    let d = (gpu_frame[i] - cpu_frame[i]).abs();
                    max_d = max_d.max(d);
                    if !tb2[si].is_finite() {
                        sky_n += 1;
                        if gpu_frame[i].to_bits() != 0 || cpu_frame[i].to_bits() != 0 {
                            p_sky += 1;
                        }
                    } else if d > 1e-4 {
                        p_d += 1;
                    }
                    for c in 1..4 {
                        if (gpu_frame[c * npp + i] - cpu_frame[c * npp + i]).abs() > 1e-5 {
                            p_n += 1;
                        }
                    }
                    for c in 4..10 {
                        if gpu_frame[c * npp + i].to_bits() != cpu_frame[c * npp + i].to_bits() {
                            p_c += 1;
                        }
                    }
                }
                let zero_bad = gpu_warped0.iter().filter(|v| v.to_bits() != 0).count();
                eprintln!(
                    "check-gpu: nppd pack ({pw}x{ph} -> {npw}x{nph}): depth>1e-4 {p_d} (max {max_d:.2e}) | sky-not-0 {p_sky} (sky px {sky_n}) | normal>1e-5 {p_n} | copy-not-bit-equal {p_c} | zeroed-warped-nonzero {zero_bad}"
                );
                if p_d != 0 || p_sky != 0 || p_n != 0 || p_c != 0 || zero_bad != 0 {
                    eprintln!("check-gpu: FAIL nppd pack/zero gates");
                    ok = false;
                }
                if must_fire && sky_n == 0 {
                    eprintln!("check-gpu: FAIL nppd sky gate vacuous");
                    ok = false;
                }

                // Warp gate: a deterministic synthetic state uploaded into
                // the state buffer, warped by frame B's REAL motion vectors,
                // vs the CPU warp on identical inputs. Interior <= 1e-6
                // (bilinear fp order is mirrored); padded border bit-zero.
                let mut cpu_state = vec![0.0f32; nppd::C_T * npp];
                for c in 0..nppd::C_T {
                    for y in 0..nph {
                        for x in 0..npw {
                            cpu_state[c * npp + y * npw + x] =
                                ((x * 7 + y * 13 + c * 29) % 101) as f32 / 101.0;
                        }
                    }
                }
                let up = match gpu::d3d12::UploadBuffer::new(&hg.device, cpu_state.len() * 4) {
                    Ok(u) => u,
                    Err(e) => {
                        eprintln!("check-gpu: FAIL nppd state upload alloc: {e}");
                        return 1;
                    }
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        cpu_state.as_ptr() as *const u8,
                        up.ptr,
                        cpu_state.len() * 4,
                    );
                }
                let upload_ok = hg.run(|l| unsafe {
                    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST;
                    l.ResourceBarrier(&[transition(&nst, ua, D3D12_RESOURCE_STATE_COPY_DEST)]);
                    l.CopyBufferRegion(&nst, 0, &up.resource, 0, (cpu_state.len() * 4) as u64);
                    l.ResourceBarrier(&[transition(&nst, D3D12_RESOURCE_STATE_COPY_DEST, ua)]);
                });
                if upload_ok.is_err() {
                    eprintln!("check-gpu: FAIL nppd state upload");
                    return 1;
                }
                let mut pre = Ok(());
                if hg.run(|l| pre = ptg.record_nppd_pre(l, 0, true)).is_err() || pre.is_err() {
                    eprintln!("check-gpu: FAIL nppd warp dispatch: {pre:?}");
                    return 1;
                }
                let gpu_warped = match readf(&mut hg, &nwp, nppd::C_T * npp) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("check-gpu: FAIL nppd warped readback: {e}");
                        return 1;
                    }
                };
                let mut cpu_warped = vec![0.0f32; nppd::C_T * npp];
                nppd::warp_temporal(
                    &cpu_state,
                    &gb2.mvec,
                    nppd::C_T,
                    pw,
                    ph,
                    npw,
                    nph,
                    &mut cpu_warped,
                );
                let (mut w_bad, mut w_border) = (0usize, 0usize);
                let mut w_max = 0.0f32;
                for c in 0..nppd::C_T {
                    for y in 0..nph {
                        for x in 0..npw {
                            let i = c * npp + y * npw + x;
                            if x >= pw || y >= ph {
                                if gpu_warped[i].to_bits() != 0 {
                                    w_border += 1;
                                }
                            } else {
                                let d = (gpu_warped[i] - cpu_warped[i]).abs();
                                w_max = w_max.max(d);
                                if d > 1e-6 {
                                    w_bad += 1;
                                }
                            }
                        }
                    }
                }
                eprintln!(
                    "check-gpu: nppd warp: interior>1e-6 {w_bad} (max {w_max:.2e}) | border-nonzero {w_border}"
                );
                if w_bad != 0 || w_border != 0 {
                    eprintln!("check-gpu: FAIL nppd warp gates");
                    ok = false;
                }

                // End-to-end DML interop: ORT on hg's queue over these exact
                // buffers vs the CPU-staged NppdContext on the same model —
                // identical logical inputs (the pack gates above bound the
                // input drift), so the outputs must agree closely. Runs only
                // when the DLLs + model exist; one loud skip line otherwise.
                let have_ort =
                    std::path::Path::new(&opts.nppd_path).join("onnxruntime.dll").exists();
                let have_model = std::path::Path::new(&opts.nppd_model).exists();
                if have_ort && have_model {
                    // Back to the reset path (zeroed warped input) so both
                    // sides run the same one-step-from-reset contract.
                    let mut pre = Ok(());
                    if hg.run(|l| pre = ptg.record_nppd_pre(l, 0, false)).is_err() || pre.is_err()
                    {
                        eprintln!("check-gpu: FAIL nppd e2e staging: {pre:?}");
                        return 1;
                    }
                    use windows::core::Interface;
                    let ng = nppd::NppdGpu::new(
                        &opts.nppd_path,
                        &opts.nppd_model,
                        hg.device.as_raw(),
                        hg.queue.as_raw(),
                        pw,
                        ph,
                        nfr.as_raw(),
                        nwp.as_raw(),
                        nout.as_raw(),
                        nst.as_raw(),
                    );
                    let e2e = ng.and_then(|mut ng| {
                        let t0 = Instant::now();
                        ng.run()?;
                        ng.sync_outputs()?;
                        let first_ms = t0.elapsed().as_secs_f64() * 1e3;
                        let t0 = Instant::now();
                        let n_time = 5;
                        for _ in 0..n_time {
                            ng.run()?;
                            ng.sync_outputs()?;
                        }
                        let per_ms = t0.elapsed().as_secs_f64() * 1e3 / n_time as f64;
                        Ok((ng, first_ms, per_ms))
                    });
                    match e2e {
                        Ok((_ng, first_ms, per_ms)) => {
                            let gpu_out = match readf(&mut hg, &nout, 3 * npp) {
                                Ok(v) => v,
                                Err(e) => {
                                    eprintln!("check-gpu: FAIL nppd out readback: {e}");
                                    return 1;
                                }
                            };
                            let gpu_state = match readf(&mut hg, &nst, nppd::C_T * npp) {
                                Ok(v) => v,
                                Err(e) => {
                                    eprintln!("check-gpu: FAIL nppd state readback: {e}");
                                    return 1;
                                }
                            };
                            let mut gpu_inter = vec![0.0f32; pw * ph * 3];
                            nppd::crop_to_interleaved(&gpu_out, npw, nph, pw, ph, &mut gpu_inter);
                            let cpu_ref = nppd::NppdContext::new(
                                &opts.nppd_path,
                                &opts.nppd_model,
                                pw,
                                ph,
                                nppd::NppdDevice::Auto,
                            )
                            .and_then(|mut c| {
                                c.denoise(&accum_at, &gb2, &basis_b, far).map(<[f32]>::to_vec)
                            });
                            match cpu_ref {
                                Ok(cpu_out) => {
                                    let mut num = 0.0f64;
                                    let mut den = 0.0f64;
                                    let mut nonfinite = 0usize;
                                    for (a, b) in gpu_inter.iter().zip(&cpu_out) {
                                        if !a.is_finite() {
                                            nonfinite += 1;
                                        }
                                        num += (a - b).abs() as f64;
                                        den += b.abs() as f64;
                                    }
                                    let rel = num / den.max(1e-12);
                                    let state_nz =
                                        gpu_state.iter().any(|v| *v != 0.0 && v.is_finite());
                                    eprintln!(
                                        "check-gpu: nppd e2e (DML interop): mean rel vs CPU-staged {rel:.2e} | non-finite {nonfinite} | state-advanced {state_nz} | {per_ms:.1} ms/run (first {first_ms:.0} ms)"
                                    );
                                    if rel > 1e-2 || nonfinite != 0 || !state_nz {
                                        eprintln!("check-gpu: FAIL nppd e2e interop gates");
                                        ok = false;
                                    }
                                }
                                Err(e) => {
                                    eprintln!("check-gpu: FAIL nppd CPU reference: {e}");
                                    ok = false;
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("check-gpu: FAIL nppd e2e interop init/run: {e}");
                            ok = false;
                        }
                    }
                } else {
                    eprintln!(
                        "check-gpu: nppd e2e interop SKIPPED (onnxruntime.dll present: {have_ort}, model present: {have_model})"
                    );
                }
            }
        }
    }

    // M13: the glare pyramid, GPU vs CPU. bloom.hlsl was the one CPU/GPU mirror
    // in the renderer with no numeric gate — it PRESENTS, so a swapped octave
    // weight or a bad barrier just looks slightly wrong and no suite objects.
    // This scores the halo (the whole pyramid's product) against `bloom::Bloom`.
    // Structure-free: it runs on a synthetic probe image, so `--stress` and
    // loaded scenes gate it exactly like the default one.
    drop(tg);
    if let Err(e) = gpu::bloom::self_test_gpu(&mut hg) {
        eprintln!("check-gpu: FAIL {e}");
        ok = false;
    }

    if !ok {
        eprintln!("GPU CHECK FAILED");
        return 1;
    }

    // --- Bench: full 1920x1080, GPU hybrid vs GPU vanilla vs CPU hybrid ---
    // Wall-clock around synchronous submits (includes per-frame sync — the
    // interactive loop pays the same). Correctness gates above are the
    // point; this is the speedometer. (`tg` was already dropped above, before
    // the bloom gate borrowed `hg`.)
    let (bw, bh) = (1920usize, 1080usize);
    let dev = hg.device.clone();
    let btg = match gpu::trace::TraceGpu::new(
        &dev,
        &dxc,
        scene,
        bvh,
        bw as u32,
        bh as u32,
        false, // bench frames don't consume the pack
        false,
        opts.gpu_debug,
        opts.bc7,
        &mut hg,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("check-gpu: bench TraceGpu init failed ({e}); skipping bench");
            println!("check-gpu: wavefront quadtree + hemi AO/GI OK");
            return 0;
        }
    };
    let bbasis = cam0.basis(bw, bh);
    let mut bench = |label: &str, hybrid: bool, fb: shade::FrustumBounce| {
        let bq = Quality { fb, ..q };
        let n = 60u32;
        // Warm once (PSO/cache effects), then time.
        for warm in 0..2u32 {
            let p = gpu::trace::FrameParams {
                cam: bbasis,
                frame: warm,
                accumulate: true,
                jitter: warm > 0,
                frame_jitter: None,
                prev_cam: None,
                q: bq,
                verify: false,
            };
            btg.write_cb(0, &p);
            let _ = hg.run(|l| btg.record_frame(l, 0, &p, hybrid));
        }
        let t0 = Instant::now();
        for f in 0..n {
            let p = gpu::trace::FrameParams {
                cam: bbasis,
                frame: f,
                accumulate: true,
                jitter: f > 0,
                frame_jitter: None,
                prev_cam: None,
                q: bq,
                verify: false,
            };
            btg.write_cb(0, &p);
            let _ = hg.run(|l| btg.record_frame(l, 0, &p, hybrid));
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        eprintln!("check-gpu: bench {bw}x{bh} {label}: {ms:6.2} ms/frame");
    };
    let fb_off = shade::FrustumBounce::OFF;
    bench("gpu hybrid          ", true, fb_off);
    bench("gpu plain reference ", false, fb_off);
    bench(
        "gpu hybrid + hemi-gi",
        true,
        shade::FrustumBounce { ao: false, gi: true, depth: 3 },
    );
    {
        // CPU hybrid at the same resolution/quality for scale.
        let stats2 = Stats::default();
        let accum2: Vec<AtomicU32> = (0..bw * bh * 3).map(|_| AtomicU32::new(0)).collect();
        let info2: Vec<AtomicU32> = (0..bw * bh).map(|_| AtomicU32::new(0)).collect();
        let tbuf2: Vec<AtomicU32> = (0..bw * bh).map(|_| AtomicU32::new(0)).collect();
        let mut cpu_ms = 0.0;
        for f in 0..8u32 {
            let ctx = FrameCtx {
                scene,
                bvh,
                cam: bbasis,
                q,
                frame: f,
                jitter: f > 0,
                rw: bw,
                rh: bh,
                accum: &accum2,
                info: &info2,
                tbuf: &tbuf2,
                stats: &stats2,
                sun: render::sun_dir(scene),
                tcache_cur: None,
                tcache_prev: &[],
                accumulate: true,
                gbuf: None,
                fsr_buf: None,
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
                hemi_share: false,
                replay_rec: None,
                cut_cur: None,
                cut_prev: None,
                discard_seeds: false,
                defer_shade: false,
            };
            let t0 = Instant::now();
            render::render_frame(&ctx, true);
            cpu_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }
        eprintln!("check-gpu: bench {bw}x{bh} cpu hybrid          : {:6.2} ms/frame", cpu_ms / 8.0);
    }

    println!("check-gpu: wavefront quadtree + hemi AO/GI OK");
    0
}

/// Headless DLSS G-buffer verification: renders two DLSS-style frames (a
/// small forward dolly apart — the same move `--check` T2 uses), then checks
/// motion vectors, depth, and the camera matrices jointly by reconstructing
/// world positions through both frames. No GPU or Streamline involved — this
/// validates the CPU capture before/without the denoiser.
/// Headless FSR verification — DLL- and GPU-free (the pure half of fsr.rs
/// plus the same G-buffer machinery --check-dlss gates): proves the signal
/// split, the wire encodings, and the MV depth-delta contract before the ffx
/// runtime ever runs. Gates: the octahedral and sqrt-albedo encoders
/// roundtrip through their wire quantization, the composite identity holds on
/// random inputs and per-pixel on rendered frames (recomputed purely from the
/// stored planes), sky pixels carry (0, 0, sky, far) exactly, the captured
/// previous-frame depth agrees with frame A's own depth buffer through the
/// motion vectors (the B channel's convention, jointly with prev_cam), the
/// dynamic-range fallbacks match the documented FSR ratios, FsrBufs
/// reinterprets in place under set_res, and turning the capture on leaves the
/// rendered image bit-identical.
fn run_check_fsr(scene: &scene::Scene, bvh: &bvh::Bvh, cam0: Camera) -> i32 {
    let (_, far) = dlss::near_far(scene.diag);
    let mut all_ok = true;

    // 1. Octahedral normals through the R10G10B10A2 wire: decode(quant10(
    // encode(n))) within 0.5° of n (10-bit octahedral worst case is ~0.2°).
    {
        let mut pass = true;
        let mut rng = fastrand::Rng::with_seed(7);
        let mut worst = 0.0f32;
        let mut samples: Vec<Vec3A> = vec![
            Vec3A::X, Vec3A::NEG_X, Vec3A::Y, Vec3A::NEG_Y, Vec3A::Z, Vec3A::NEG_Z,
        ];
        while samples.len() < 4096 {
            let v = Vec3A::new(rng.f32() * 2.0 - 1.0, rng.f32() * 2.0 - 1.0, rng.f32() * 2.0 - 1.0);
            let l = v.length();
            if l > 0.1 && l <= 1.0 {
                samples.push(v / l);
            }
        }
        for n in samples {
            let (u, v) = fsr::oct_encode(n);
            if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
                eprintln!("oct_encode({n}) = ({u},{v}) escapes [0,1]");
                pass = false;
            }
            let d = fsr::oct_decode(fsr::quant_unorm(u, 10), fsr::quant_unorm(v, 10));
            let ang = n.dot(d).clamp(-1.0, 1.0).acos().to_degrees();
            worst = worst.max(ang);
        }
        if worst > 0.5 {
            eprintln!("octahedral worst-case error {worst:.4}° exceeds 0.5°");
            pass = false;
        }
        eprintln!("octahedral encode (worst {worst:.4}°): {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 2. sqrt-albedo wire: bounded roundtrip over [0,1], the sqrt advantage
    // in the darks, and idempotence (albedo_wire of an albedo_wire value is
    // itself — what lets the frame gate below recompute wire albedos from
    // the f16-stored G-buffer planes).
    {
        let mut pass = true;
        for i in 0..=4096 {
            let v = i as f32 / 4096.0;
            let w = fsr::albedo_wire(v);
            let err = (w - v).abs();
            if err > 1.2e-2 || (v < 0.01 && err > 8e-4) {
                eprintln!("albedo_wire({v}) = {w} (err {err})");
                pass = false;
            }
            if fsr::albedo_wire(w) != w {
                eprintln!("albedo_wire not idempotent at {v}");
                pass = false;
            }
        }
        eprintln!("sqrt-albedo wire: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 2b. The GPU pack's wire twin (`sqrt_wire`, no leading f16 rounding —
    // the pack stores f32): same bounds, idempotence, exact endpoints, and
    // agreement with `albedo_wire` on already-f16 values (the CPU oracle for
    // the GPU FSR-RR feed gates relies on both properties).
    {
        let mut pass = true;
        for i in 0..=4096 {
            let v = i as f32 / 4096.0;
            let w = fsr::sqrt_wire(v);
            if (w - v).abs() > 1.2e-2 || (v < 0.01 && (w - v).abs() > 8e-4) {
                eprintln!("sqrt_wire({v}) = {w}");
                pass = false;
            }
            if fsr::sqrt_wire(w) != w {
                eprintln!("sqrt_wire not idempotent at {v}");
                pass = false;
            }
            if fsr::albedo_wire(fsr::q16(v)) != fsr::sqrt_wire(fsr::q16(v)) {
                eprintln!("sqrt_wire disagrees with albedo_wire on the f16 value of {v}");
                pass = false;
            }
        }
        if fsr::sqrt_wire(0.0) != 0.0 || fsr::sqrt_wire(1.0) != 1.0 {
            eprintln!("sqrt_wire endpoints off");
            pass = false;
        }
        eprintln!("sqrt-wire (GPU pack twin): {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 3. Composite identity on random values (split then remodulate must
    // reproduce the color to f32 re-association noise, independent of any
    // renderer state).
    {
        let mut pass = true;
        let mut rng = fastrand::Rng::with_seed(11);
        for _ in 0..4096 {
            let mut v3 = |scale: f32| Vec3A::new(rng.f32(), rng.f32(), rng.f32()) * scale;
            let color = v3(4.0);
            let (direct_d, direct_s) = (v3(3.0), v3(2.0));
            let albedo = v3(1.0);
            let metallic = rng.f32();
            let kd = albedo * (1.0 - metallic);
            let f0 = Vec3A::splat(0.04).lerp(albedo, metallic);
            let sig = fsr::split_signals(color, direct_d, direct_s, kd, f0);
            let re = fsr::composite(&sig, kd, f0);
            let err = (re - color).abs().max_element();
            if !re.is_finite() || err > 1e-5 * color.abs().max_element().max(1.0) {
                eprintln!("composite identity err {err} at color {color}");
                pass = false;
            }
        }
        // The zero-wire-F0 regression: a saturated colored metal (albedo
        // channel 0 at metallic 1 -> wire F0 channel 0) under a hot specular
        // spike hits the MIN_SPEC_ALB floor with direct_s/1e-4 ≫ f16::MAX;
        // the saturating q16 must keep ds finite (an inf ds makes the
        // residual inf·0 = NaN) and the identity must still hold exactly
        // (residual is the remainder of the clamped wire products).
        {
            let albedo = Vec3A::new(1.0, 0.4, 0.0);
            let (kd, f0) = (Vec3A::ZERO, albedo); // metallic = 1
            let direct_s = Vec3A::splat(300.0);
            let color = direct_s * fsr::albedo_wire3(f0) + Vec3A::splat(0.05);
            let sig = fsr::split_signals(color, Vec3A::ZERO, direct_s, kd, f0);
            let re = fsr::composite(&sig, kd, f0);
            if !sig.ds.is_finite() || !sig.residual.is_finite() || !re.is_finite() {
                eprintln!("zero-wire-F0 split not finite: ds {} residual {}", sig.ds, sig.residual);
                pass = false;
            }
            if (re - color).abs().max_element() > 1e-5 * color.max_element() {
                eprintln!("zero-wire-F0 composite identity broken: {re} vs {color}");
                pass = false;
            }
        }
        eprintln!("composite identity (random): {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 4. Dynamic-range fallbacks match the documented FSR quality ratios.
    {
        let mut pass = true;
        if fsr::fallback_render_res((1920, 1080), fsr::RATIO_QUALITY) != (1280, 720) {
            eprintln!("quality fallback at 1080p wrong");
            pass = false;
        }
        if fsr::fallback_render_res((1920, 1080), fsr::RATIO_ULTRA_PERFORMANCE) != (640, 360) {
            eprintln!("ultra-performance fallback at 1080p wrong");
            pass = false;
        }
        eprintln!("range fallbacks: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 5. FsrBufs set_res reinterpretation: write/read at a smaller logical
    // res inside the construction capacity.
    {
        let mut pass = true;
        let mut f = fsr::FsrBufs::new(64, 64);
        f.set_res(20, 12);
        let sig = fsr::Signals {
            dd: Vec3A::new(0.5, 0.25, 0.125),
            ds: Vec3A::new(1.0, 2.0, 4.0),
            residual: Vec3A::new(0.1, -0.2, 0.3),
        };
        f.write(19, 11, &sig, 42.0);
        let r = f.read(19, 11);
        if r.dd != sig.dd || r.ds != sig.ds || r.residual != sig.residual {
            eprintln!("FsrBufs set_res write/read mismatch");
            pass = false;
        }
        eprintln!("FsrBufs::set_res: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 6. Provider pick (pure): the fsr::pick_version rules that decide the
    // session flavor at init — the FSR4-default / FSR3.1-fallback / --fsr3
    // force triangle, plus name-parse robustness. Ids are arbitrary but
    // distinct; only names carry meaning (matching the live enumeration).
    {
        use fsr::Flavor::{Fsr3, Fsr4Rr};
        let mut pass = true;
        let list = |names: &[&str]| -> Vec<(u64, String)> {
            names.iter().enumerate().map(|(i, n)| (0x100 + i as u64, n.to_string())).collect()
        };
        let all3 = list(&["FSR 4.1.1", "FSR 3.1.5", "FSR 2.3.4"]);
        let id_315 = all3[1].0;
        let mut case = |desc: &str, got: Option<(u64, fsr::Flavor)>, want: Option<(u64, fsr::Flavor)>| {
            if got != want {
                eprintln!("pick_version {desc}: got {got:?}, want {want:?}");
                pass = false;
            }
        };
        // RR available takes the FSR4 default (id 0 = no override — the
        // original create path bit-for-bit) REGARDLESS of what the upscaler
        // names parse to: the RR provider is itself the RDNA4/FSR4 signal
        // and the display names are not a contract (a renamed FSR4 provider
        // must not silently downgrade the session). No RR or --fsr3 takes
        // the 3.1.5 provider.
        case("all, rr", fsr::pick_version(&all3, true, false), Some((0, Fsr4Rr)));
        case("all, rr, forced", fsr::pick_version(&all3, true, true), Some((id_315, Fsr3)));
        case("all, no rr", fsr::pick_version(&all3, false, false), Some((id_315, Fsr3)));
        // Only 3.1 listed: highest patch wins whenever the pick reaches the
        // 3.x scan; with RR available (and not forced) the RR signal still
        // wins even though no 4.x name parses — see above.
        let only3 = list(&["FSR 3.1.4", "FSR 3.1.5"]);
        let id_hi = only3[1].0;
        for force in [false, true] {
            case(
                &format!("only-3.1 no-rr force={force}"),
                fsr::pick_version(&only3, false, force),
                Some((id_hi, Fsr3)),
            );
        }
        case("only-3.1, rr", fsr::pick_version(&only3, true, false), Some((0, Fsr4Rr)));
        case("only-3.1, rr, forced", fsr::pick_version(&only3, true, true), Some((id_hi, Fsr3)));
        case("unparseable, rr", fsr::pick_version(&list(&["FSR4"]), true, false), Some((0, Fsr4Rr)));
        // No pickable provider: only-4.x without RR, FSR2-only, empty, and
        // forced-but-absent must all yield None (init fails loudly — the
        // fallback is never FSR2 and a forced --fsr3 never silently
        // un-forces).
        let only4 = list(&["FSR 4.1.1"]);
        case("only-4.x, no rr", fsr::pick_version(&only4, false, false), None);
        case("only-4.x, forced", fsr::pick_version(&only4, true, true), None);
        case("fsr2-only", fsr::pick_version(&list(&["FSR 2.3.4"]), false, false), None);
        case("empty, no rr", fsr::pick_version(&[], false, false), None);
        // Name-parse robustness: prefixed and bare forms, non-version names
        // ignored, and a second embedded version (SDK/driver build) must not
        // shadow the FSR triple.
        let odd = list(&["FidelityFX FSR 3.1.5", "3.1.4", "experimental"]);
        case("odd names", fsr::pick_version(&odd, false, false), Some((odd[0].0, Fsr3)));
        let multi = list(&["AMD 24.10.1 FSR 3.1.5"]);
        case("multi-triple", fsr::pick_version(&multi, false, false), Some((multi[0].0, Fsr3)));
        if fsr::parse_provider_versions("FidelityFX FSR 3.1.5") != vec![(3, 1, 5)] {
            eprintln!("parse_provider_versions prefixed form wrong");
            pass = false;
        }
        if fsr::parse_provider_versions("AMD 24.10.1 FSR 3.1.5") != vec![(24, 10, 1), (3, 1, 5)] {
            eprintln!("parse_provider_versions multi-triple form wrong");
            pass = false;
        }
        if !fsr::parse_provider_versions("no digits here").is_empty() {
            eprintln!("parse_provider_versions accepted a non-version");
            pass = false;
        }
        eprintln!("provider pick: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 7. Rendered-frame gates at a native and an odd dynamic res (odd dims
    // also exercise the quadtree's odd splits with the capture on).
    for (rw, rh) in [(800usize, 600usize), (531, 399)] {
        all_ok &= fsr_frame_check(scene, bvh, cam0, rw, rh, far);
    }

    if all_ok {
        eprintln!("FSR CHECK PASSED");
        0
    } else {
        eprintln!("FSR CHECK FAILED");
        1
    }
}

/// The rendered-frame half of --check-fsr at one resolution: frame A at
/// `cam0`, frame B a 0.02·diag dolly later with A as its previous frame, both
/// with the G-buffer AND signal capture on. Zero jitter (reconstructions
/// assume pixel centers, as in mv_check_at).
fn fsr_frame_check(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    rw: usize,
    rh: usize,
    far: f32,
) -> bool {
    let q = Quality {
        shadow_samples: 1,
        ao_samples: 1,
        reflections: true,
        fb: shade::FrustumBounce::OFF,
    };
    let stats = Stats::default();
    eprintln!("FSR frame gates at {rw}x{rh}:");
    let alloc32 = |n: usize| -> Vec<AtomicU32> { (0..n).map(|_| AtomicU32::new(0)).collect() };
    let accum = alloc32(rw * rh * 3);
    let info = alloc32(rw * rh);
    let tbuf = alloc32(rw * rh);
    let basis_a = cam0.basis(rw, rh);
    let ga = dlss::GBufs::new(rw, rh);
    let gb = dlss::GBufs::new(rw, rh);
    let fb = fsr::FsrBufs::new(rw, rh);
    let render = |g: &dlss::GBufs,
                  f: Option<&fsr::FsrBufs>,
                  accum: &[AtomicU32],
                  basis: camera::CamBasis,
                  prev: Option<camera::CamBasis>,
                  frame: u32| {
        let ctx = FrameCtx {
            scene,
            bvh,
            cam: basis,
            q,
            frame,
            jitter: false,
            rw,
            rh,
            accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: &[],
            accumulate: false,
            gbuf: Some(g),
            fsr_buf: f,
            prev_cam: prev,
            frame_jitter: Some((0.0, 0.0)),
            adaptive: false,
            hemi_share: false,
            replay_rec: None,
            cut_cur: None,
            cut_prev: None,
            discard_seeds: false,
            defer_shade: false,
        };
        render::render_frame(&ctx, true);
    };

    // Off-path bit-identity: the same frame with the signal capture off must
    // produce a bit-identical image (the capture writes only its own planes).
    let mut ok = true;
    {
        let accum_off = alloc32(rw * rh * 3);
        render(&ga, None, &accum_off, basis_a, None, 0);
        render(&ga, Some(&fb), &accum, basis_a, None, 0);
        let same = (0..rw * rh * 3)
            .all(|i| accum[i].load(Relaxed) == accum_off[i].load(Relaxed));
        if !same {
            eprintln!("  capture on/off bit-identity: FAIL");
            ok = false;
        } else {
            eprintln!("  capture on/off bit-identity: OK");
        }
    }

    // Composite identity per pixel, recomputed purely from the stored planes
    // (dd/ds f16, residual f32, wire albedos from the f16 G-buffer planes) —
    // exactly what the GPU composite pass reads. Sky pixels must carry
    // (0, 0, sky color, prev_z = far) exactly.
    {
        let mut worst = 0.0f32;
        let mut sky_ok = true;
        for y in 0..rh {
            for x in 0..rw {
                let i = y * rw + x;
                let sig = fb.read(x, y);
                let c = Vec3A::new(
                    f32::from_bits(accum[i * 3].load(Relaxed)),
                    f32::from_bits(accum[i * 3 + 1].load(Relaxed)),
                    f32::from_bits(accum[i * 3 + 2].load(Relaxed)),
                );
                let is_sky = f32::from_bits(tbuf[i].load(Relaxed)).is_infinite();
                if is_sky {
                    let prev_z = f32::from_bits(fb.prev_z[i].load(Relaxed));
                    if sig.dd != Vec3A::ZERO
                        || sig.ds != Vec3A::ZERO
                        || sig.residual != c
                        || prev_z != far
                    {
                        sky_ok = false;
                    }
                    continue;
                }
                let l3 = |buf: &[std::sync::atomic::AtomicU16], j: usize| {
                    Vec3A::new(
                        dlss::ld16(&buf[j * 3]),
                        dlss::ld16(&buf[j * 3 + 1]),
                        dlss::ld16(&buf[j * 3 + 2]),
                    )
                };
                let re = fsr::composite(&sig, l3(&ga.diff_alb, i), l3(&ga.spec_alb, i));
                // NaN/inf must fail loudly — f32::max would silently discard
                // a NaN err, hiding an overflowed signal plane.
                if !re.is_finite() {
                    worst = f32::INFINITY;
                    continue;
                }
                let err = (re - c).abs().max_element() / c.abs().max_element().max(1.0);
                worst = worst.max(err);
            }
        }
        // Re-association of the remainder sum costs at most a few ulps.
        if worst > 1e-5 {
            eprintln!("  composite identity (rendered): FAIL (worst rel err {worst:.3e})");
            ok = false;
        } else {
            eprintln!("  composite identity (rendered): OK (worst rel err {worst:.3e})");
        }
        if !sky_ok {
            eprintln!("  sky signal contract: FAIL");
            ok = false;
        } else {
            eprintln!("  sky signal contract: OK");
        }
    }

    // Frame B: dolly forward with A as previous. The captured prev_z must
    // agree with frame A's own depth buffer at the pixel the motion vector
    // lands on (median/p90 gates — edges and disocclusions legitimately
    // disagree, as in dlss::mv_selftest).
    {
        let mut cam_b = cam0;
        cam_b.pos += cam0.forward() * (0.02 * scene.diag);
        let basis_b = cam_b.basis(rw, rh);
        render(&gb, Some(&fb), &accum, basis_b, Some(basis_a), 1);

        let mut errs: Vec<f32> = Vec::new();
        for y in 0..rh {
            for x in 0..rw {
                let i = y * rw + x;
                let zb = f32::from_bits(gb.depth[i].load(Relaxed));
                if !(zb > 0.0) || zb >= far * 0.99 {
                    continue; // sky
                }
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let (mx, my) = (dlss::ld16(&gb.mvec[i * 2]), dlss::ld16(&gb.mvec[i * 2 + 1]));
                let (px, py) = (fx + mx, fy + my);
                let (ax, ay) = (px as usize, py as usize);
                if px < 0.5 || py < 0.5 || ax + 1 >= rw || ay + 1 >= rh {
                    continue;
                }
                let za = f32::from_bits(ga.depth[ay * rw + ax].load(Relaxed));
                if za >= far * 0.99 {
                    continue; // disocclusion
                }
                let prev_z = f32::from_bits(fb.prev_z[i].load(Relaxed));
                errs.push((prev_z - za).abs());
            }
        }
        if errs.is_empty() {
            eprintln!("  prev-depth agreement: FAIL (no comparable pixels)");
            ok = false;
        } else {
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = errs[errs.len() / 2];
            let p90 = errs[errs.len() * 9 / 10];
            let pass = median < 1e-3 * scene.diag && p90 < 1e-2 * scene.diag;
            eprintln!(
                "  prev-depth agreement: {} px | median {:.3e} | p90 {:.3e} -> {}",
                errs.len(),
                median,
                p90,
                if pass { "OK" } else { "FAIL" }
            );
            ok &= pass;
        }
    }
    ok
}

fn run_check_dlss(scene: &scene::Scene, bvh: &bvh::Bvh, cam0: Camera, dump: bool) -> i32 {
    // Halton structural checks: known prefixes, offsets in [-0.5, 0.5).
    let mut halton_ok = true;
    for (i, base, want) in [(1u32, 2u32, 0.5f32), (2, 2, 0.25), (3, 2, 0.75), (1, 3, 1.0 / 3.0), (2, 3, 2.0 / 3.0)] {
        let got = dlss::halton(i, base);
        if (got - want).abs() > 1e-6 {
            eprintln!("halton({i},{base}) = {got}, want {want}");
            halton_ok = false;
        }
    }
    for idx in 0..64 {
        let (ox, oy) = dlss::jitter_for(idx);
        if !(-0.5..0.5).contains(&ox) || !(-0.5..0.5).contains(&oy) {
            eprintln!("jitter_for({idx}) = ({ox},{oy}) out of [-0.5,0.5)");
            halton_ok = false;
        }
    }
    eprintln!("halton/jitter checks: {}", if halton_ok { "OK" } else { "FAIL" });

    // Frame A at the given camera; frame B a 0.02·diag forward dolly later,
    // with A as its previous frame. Run once at the native test res and once
    // at the Quality-mode render res stand-in (odd width — also exercises
    // odd-dim quadtree splits), since the interactive DLSS path now traces
    // at a sub-native resolution.
    let mv_native_ok = mv_check_at(scene, bvh, cam0, 800, 600, if dump { Some("dlss_gbuf") } else { None });
    let (qw, qh) = dlss::headless_render_res(800, 600);
    let mv_quality_ok = mv_check_at(scene, bvh, cam0, qw, qh, None);
    // Step-wise DRS: an arbitrary quantized size inside a typical RR range —
    // the varying-res MV/depth/matrix contract (the extent tagging itself is
    // SL-side and validated interactively; headless stays SL-free).
    let (dw, dh) = xess::quantize_res(0.55, (800, 600), (266, 200), (800, 600));
    let mv_drs_ok = mv_check_at(scene, bvh, cam0, dw, dh, None);

    if halton_ok && mv_native_ok && mv_quality_ok && mv_drs_ok {
        eprintln!("DLSS CHECK PASSED");
        0
    } else {
        eprintln!("DLSS CHECK FAILED");
        1
    }
}

/// One MV/depth/matrix pass at an arbitrary render resolution: frame A at
/// `cam0`, frame B a 0.02·diag forward dolly later with A as its previous
/// frame, gated by `dlss::mv_selftest`. Zero jitter so samples sit on pixel
/// centers — the reconstruction in the self-test assumes centers. Shared by
/// --check-dlss (the fixed DLSS-style resolutions) and --check-xess (a sweep
/// of dynamic render resolutions).
/// Textured-albedo A/B (check-gpu M7 / check-dxr T4, textured scenes only):
/// the GPU pack's diffuse-albedo plane vs a CPU `GBufs` render at the same
/// pose/contract, compared over class-matched hit pixels. Gates: mean |d|
/// per channel <= 0.02 (hardware sRGB decode + bilinear filter vs the CPU
/// LUT — precision slack, not a semantic gap) and > 64 distinct GPU albedo
/// values (a flat-Kd regression collapses to one value per material and
/// cannot pass). Also prints the pose's transmissive-hit pixel count (a
/// center-ray re-trace) — the per-pose glass anti-vacuity signal, unGated
/// because it is pose-dependent. Returns gate pass; opaque untextured
/// scenes return true silently.
#[cfg(windows)]
fn albedo_ab_check(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    gpu_g: &dlss::GBufs,
    gpu_t: &[f32],
    pw: usize,
    ph: usize,
    tag: &str,
) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if scene.textures.is_empty() {
        return true;
    }
    let q = Quality::upscaler_1spp();
    let stats = Stats::default();
    let accum: Vec<AtomicU32> = (0..pw * ph * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..pw * ph).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..pw * ph).map(|_| AtomicU32::new(0)).collect();
    let basis = cam0.basis(pw, ph);
    let cg = dlss::GBufs::new(pw, ph);
    let ctx = FrameCtx {
        scene,
        bvh,
        cam: basis,
        q,
        frame: 0,
        jitter: false,
        rw: pw,
        rh: ph,
        accum: &accum,
        info: &info,
        tbuf: &tbuf,
        stats: &stats,
        sun: render::sun_dir(scene),
        tcache_cur: None,
        tcache_prev: &[],
        accumulate: false,
        gbuf: Some(&cg),
        fsr_buf: None,
        prev_cam: None,
        frame_jitter: Some((0.0, 0.0)),
        adaptive: false,
        hemi_share: false,
        replay_rec: None,
        cut_cur: None,
        cut_prev: None,
        discard_seeds: false,
        defer_shade: false,
    };
    render::render_frame(&ctx, true);

    let mut n = 0usize;
    let mut sum = [0.0f64; 3];
    let mut distinct = std::collections::HashSet::new();
    for i in 0..pw * ph {
        let cpu_hit = f32::from_bits(tbuf[i].load(Relaxed)).is_finite();
        if !cpu_hit || !gpu_t[i].is_finite() {
            continue;
        }
        n += 1;
        let mut key = [0u16; 3];
        for c in 0..3 {
            let a = dlss::ld16(&gpu_g.diff_alb[i * 3 + c]);
            let b = dlss::ld16(&cg.diff_alb[i * 3 + c]);
            sum[c] += (a - b).abs() as f64;
            key[c] = gpu_g.diff_alb[i * 3 + c].load(Relaxed);
        }
        distinct.insert(key);
    }
    // Per-pose glass presence: center-ray re-trace (the GBufs don't store
    // the hit tri). Printed, not gated — a pose can legitimately see none.
    let glass_px: usize = (0..ph)
        .into_par_iter()
        .map(|y| {
            let mut stats = stats::LocalStats::default();
            let mut count = 0usize;
            for x in 0..pw {
                let ray = bvh::Ray::new(
                    basis.origin,
                    basis.ray_dir(x as f32 + 0.5, y as f32 + 0.5),
                );
                if let Some(h) = bvh.intersect(scene, &ray, 0.0, f32::INFINITY, &mut stats.ray_nodes)
                {
                    if scene.materials[scene.tri_mat[h.tri as usize] as usize].transmission > 0.0 {
                        count += 1;
                    }
                }
            }
            count
        })
        .sum();
    let mean = |c: usize| if n > 0 { sum[c] / n as f64 } else { 0.0 };
    eprintln!(
        "check-{tag}: albedo A/B ({pw}x{ph}): mean |d| {:.4}/{:.4}/{:.4} over {n} px | distinct GPU albedos {} | glass px {glass_px}",
        mean(0), mean(1), mean(2), distinct.len(),
    );
    let mut pass = true;
    if mean(0) > 0.02 || mean(1) > 0.02 || mean(2) > 0.02 {
        eprintln!("check-{tag}: FAIL textured albedo off the CPU render (flat-Kd regression?)");
        pass = false;
    }
    if distinct.len() <= 64 {
        eprintln!("check-{tag}: FAIL <= 64 distinct GPU albedo values (textures not sampled)");
        pass = false;
    }
    pass
}

fn mv_check_at(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    rw: usize,
    rh: usize,
    dump_prefix: Option<&str>,
) -> bool {
    let (near, far) = dlss::near_far(scene.diag);
    let q = Quality {
        shadow_samples: 1,
        ao_samples: 1,
        reflections: true,
        fb: shade::FrustumBounce::OFF,
    };
    let stats = Stats::default();
    eprintln!("MV/depth/matrix self-test at {rw}x{rh}:");
    let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let basis_a = cam0.basis(rw, rh);
    let ga = dlss::GBufs::new(rw, rh);
    let gb = dlss::GBufs::new(rw, rh);
    let render_dlss_frame =
        |g: &dlss::GBufs, basis: camera::CamBasis, prev: Option<camera::CamBasis>, frame: u32| {
            let ctx = FrameCtx {
                scene,
                bvh,
                cam: basis,
                q,
                frame,
                jitter: false,
                rw,
                rh,
                accum: &accum,
                info: &info,
                tbuf: &tbuf,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: None,
                tcache_prev: &[],
                accumulate: false,
                gbuf: Some(g),
                fsr_buf: None,
                prev_cam: prev,
                frame_jitter: Some((0.0, 0.0)),
                adaptive: false,
                hemi_share: false,
                replay_rec: None,
                cut_cur: None,
                cut_prev: None,
                discard_seeds: false,
                defer_shade: false,
            };
            render::render_frame(&ctx, true);
        };
    render_dlss_frame(&ga, basis_a, None, 0);

    let mut cam_b = cam0;
    cam_b.pos += cam0.forward() * (0.02 * scene.diag);
    let basis_b = cam_b.basis(rw, rh);
    render_dlss_frame(&gb, basis_b, Some(basis_a), 1);

    let mats_b = dlss::cam_matrices(&cam_b, rw, rh, near, far);
    let ok = dlss::mv_selftest(&ga, &basis_a, &gb, &basis_b, &mats_b, scene.diag, far);

    if let Some(prefix) = dump_prefix {
        dlss::dump_gbufs(&gb, prefix, far);
    }
    ok
}

/// Headless XeSS verification — DLL- and GPU-free (the pure half of xess.rs
/// plus the same G-buffer machinery --check-dlss gates): proves the
/// dynamic-resolution contract before the SDK ever runs. Gates: the
/// view-Z → clip-depth encoding roundtrips and hits its endpoints exactly
/// (sky = far must land on 1.0), `quantize_res` respects the range clamps /
/// height quantum / window aspect, the scale controller clamps, sheds fast,
/// creeps slowly and respects the deadband on a scripted frame-time
/// sequence, and the MV/depth/matrix self-test passes at a sweep of dynamic
/// render resolutions (quantized 16:9 steps plus an odd-dimension literal).
fn run_check_xess(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    dump: bool,
    structural: bool,
) -> i32 {
    let (near, far) = dlss::near_far(scene.diag);
    let mut all_ok = true;

    // 1. Depth encoding (reversed-Z): monotone decreasing, roundtrips,
    // exact endpoints.
    {
        let mut pass = true;
        let mut prev_d = 2.0f32;
        for i in 0..=256 {
            let z = near + (far - near) * (i as f32 / 256.0);
            let d = xess::view_z_to_clip_depth(z, near, far);
            let z2 = xess::clip_depth_to_view_z(d, near, far);
            if d > prev_d {
                eprintln!("depth encoding not monotone decreasing at z={z}");
                pass = false;
            }
            if (z2 - z).abs() > 1e-3 * z {
                eprintln!("depth roundtrip z={z} -> d={d} -> {z2}");
                pass = false;
            }
            prev_d = d;
        }
        if xess::view_z_to_clip_depth(near, near, far) != 1.0 {
            eprintln!("depth(near) != 1");
            pass = false;
        }
        if xess::view_z_to_clip_depth(far, near, far) != 0.0 {
            eprintln!("depth(far) != 0 (sky sentinel would drift)");
            pass = false;
        }
        eprintln!("depth encoding: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 2. quantize_res: height quantum, window aspect, hard range clamps.
    {
        let mut pass = true;
        let out = (1920usize, 1080usize);
        let min = (640usize, 360usize);
        let max = (1920usize, 1080usize);
        for i in 0..=40 {
            let s = 0.2 + 0.9 * (i as f32 / 40.0); // sweeps below min and above max
            let (rw, rh) = xess::quantize_res(s, out, min, max);
            if rw < min.0 || rh < min.1 || rw > max.0 || rh > max.1 {
                eprintln!("quantize_res({s}) = {rw}x{rh} escapes range");
                pass = false;
            }
            let aspect = rw as f32 / rh as f32;
            let want = out.0 as f32 / out.1 as f32;
            if (aspect - want).abs() > 0.02 {
                eprintln!("quantize_res({s}) = {rw}x{rh} aspect {aspect} vs {want}");
                pass = false;
            }
            if rh % xess::RES_STEP != 0 && rh != min.1 && rh != max.1 {
                eprintln!("quantize_res({s}) height {rh} off the quantum");
                pass = false;
            }
        }
        eprintln!("quantize_res: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 2b. --lock-res argument mapping (presets, bare ratios, rejections)
    // and the quality preset's quantization at 1080p (exact 2/3).
    {
        let mut pass = true;
        for (a, want) in [
            ("quality", Some(2.0f32 / 3.0)),
            ("balanced", Some(0.58)),
            ("performance", Some(0.5)),
            ("ultra-performance", Some(1.0 / 3.0)),
            ("native", Some(1.0)),
            ("0.75", Some(0.75)),
            ("0", None),
            ("1.5", None),
            ("-0.5", None),
            ("NaN", None),
            ("bogus", None),
        ] {
            if xess::lock_scale(a) != want {
                eprintln!("lock_scale({a}) != {want:?}");
                pass = false;
            }
        }
        let q = xess::quantize_res(2.0 / 3.0, (1920, 1080), (640, 360), (1920, 1080));
        if q != (1280, 720) {
            eprintln!("quality lock at 1080p = {q:?}, want (1280, 720)");
            pass = false;
        }
        eprintln!("lock-res map: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 3. Scale controller on a scripted frame-time sequence (no wall clock).
    {
        let mut pass = true;
        let budget = 15.0f32;
        let mut ctl = xess::ScaleCtl::new(720, 360, 1080, 1080);
        let s0 = ctl.scale();
        ctl.update(6.0 * budget, budget); // blown frame
        if ctl.scale() >= s0 {
            eprintln!("controller did not shed after a blown frame");
            pass = false;
        }
        let shed = ctl.scale();
        ctl.update(0.2 * budget, budget); // one cheap frame
        if ctl.scale() > shed * 1.05 {
            eprintln!("controller climbed more than the slow-up bound in one frame");
            pass = false;
        }
        for _ in 0..2000 {
            ctl.update(0.2 * budget, budget); // sustained cheap: creep to the top
        }
        if (ctl.scale() - 1.0).abs() > 1e-3 {
            eprintln!("controller failed to reach the max clamp ({})", ctl.scale());
            pass = false;
        }
        for _ in 0..100 {
            ctl.update(100.0 * budget, budget); // sustained blown: floor clamp
        }
        if (ctl.scale() - 360.0 / 1080.0).abs() > 1e-3 {
            eprintln!("controller failed to hold the min clamp ({})", ctl.scale());
            pass = false;
        }
        let mut parked = xess::ScaleCtl::new(720, 360, 1080, 1080);
        let sp = parked.scale();
        for _ in 0..50 {
            parked.update(0.75 * budget, budget); // >60% of budget: deadband
        }
        if parked.scale() != sp {
            eprintln!("controller climbed inside the deadband");
            pass = false;
        }
        // Step limiter, snap parity (ramp = 0 is the pre-ramp behavior):
        // first target adopts; the dwell holds against both growth and
        // non-emergency sheds; an emergency may bypass only to SHED; dwell
        // expiry adopts the current target in one decision.
        let out = (1920usize, 1080usize);
        let (rmin, rmax) = ((640usize, 360usize), (1920usize, 1080usize));
        let mut lim = xess::StepLimiter::new(0);
        if lim.apply((1280, 720), false, out, rmin, rmax) != (1280, 720) {
            eprintln!("step limiter: first apply did not adopt");
            pass = false;
        }
        if lim.apply((1216, 684), false, out, rmin, rmax) != (1280, 720) {
            eprintln!("step limiter: dwell did not hold a shed");
            pass = false;
        }
        if lim.apply((1408, 792), true, out, rmin, rmax) != (1280, 720) {
            eprintln!("step limiter: emergency bypassed for GROWTH");
            pass = false;
        }
        if lim.apply((960, 540), true, out, rmin, rmax) != (960, 540) {
            eprintln!("step limiter: emergency shed did not bypass");
            pass = false;
        }
        let mut adopted = (0, 0);
        for _ in 0..=xess::STEP_DWELL {
            adopted = lim.apply((1152, 648), false, out, rmin, rmax);
        }
        if adopted != (1152, 648) {
            eprintln!("step limiter: dwell expiry did not adopt");
            pass = false;
        }
        // Ramped limiter: an adoption starts a lerp from the previous
        // endpoint — intermediates are weakly monotone in height, in-range,
        // exact-aspect via width_for_height, and land exactly on the
        // endpoint; the dwell is not re-armed mid-ramp; an emergency shed
        // compares against the ramp's CURRENT output and snaps.
        let mut lim = xess::StepLimiter::new(xess::RAMP_FRAMES);
        lim.apply((1280, 720), false, out, rmin, rmax);
        if lim.ramping() {
            eprintln!("ramp: first adoption started a ramp");
            pass = false;
        }
        let mut cur = (0usize, 0usize);
        for _ in 0..xess::STEP_DWELL {
            cur = lim.apply((960, 540), false, out, rmin, rmax);
        }
        if cur != (1280, 720) || !lim.ramping() || lim.endpoint() != (960, 540) {
            eprintln!("ramp: adoption frame did not hold the old endpoint (got {cur:?})");
            pass = false;
        }
        // Walk the down-ramp while feeding a DIFFERENT non-emergency target
        // every frame — none may adopt (the dwell holds mid-ramp).
        let mut last_h = 720usize;
        let mut end = (0usize, 0usize);
        for i in 1..=xess::RAMP_FRAMES {
            let (rw2, rh2) = lim.apply((1152, 648), false, out, rmin, rmax);
            if rh2 > last_h {
                eprintln!("ramp: height not monotone at frame {i} ({last_h} -> {rh2})");
                pass = false;
            }
            last_h = rh2;
            if rh2 < rmin.1 || rh2 > rmax.1 || rw2 < rmin.0 || rw2 > rmax.0 {
                eprintln!("ramp: intermediate {rw2}x{rh2} out of range");
                pass = false;
            }
            if rw2 != xess::width_for_height(rh2, out, rmin.0, rmax.0) {
                eprintln!("ramp: intermediate width {rw2} off the exact aspect at h {rh2}");
                pass = false;
            }
            end = (rw2, rh2);
        }
        if end != (960, 540) {
            eprintln!("ramp: did not land exactly on the endpoint (got {end:?})");
            pass = false;
        }
        if lim.endpoint() != (960, 540) {
            eprintln!("ramp: a mid-ramp target re-armed the dwell / adopted");
            pass = false;
        }
        // Dwell re-arms at adoption: the same target adopts exactly at
        // STEP_DWELL frames since the last adoption, and the new ramp
        // departs from the PREVIOUS endpoint (from = completed endpoint).
        let mut held = true;
        for _ in 0..(xess::STEP_DWELL - xess::RAMP_FRAMES) {
            if lim.apply((1152, 648), false, out, rmin, rmax) != (960, 540) {
                held = false;
            }
        }
        if !held || lim.endpoint() != (1152, 648) {
            eprintln!(
                "ramp: post-ramp dwell wrong (held {held}, endpoint {:?})",
                lim.endpoint()
            );
            pass = false;
        }
        let (_, h1) = lim.apply((1152, 648), false, out, rmin, rmax);
        if !(540 < h1 && h1 < 648) {
            eprintln!("ramp: new ramp did not depart the previous endpoint (h {h1})");
            pass = false;
        }
        // Emergency growth guard vs the CURRENT output: mid up-ramp, a shed
        // target below the endpoint but above the ramp's current output must
        // NOT adopt (it would grow resolution on a blown frame). The warm-up
        // is derived from RAMP_FRAMES (aim ~t = 1/3) and the premise is
        // asserted explicitly so a constant change fails loudly with the
        // real reason instead of a misleading gate failure.
        let warm = (xess::RAMP_FRAMES / 3).max(1);
        let mut cur = (0usize, 0usize);
        for _ in 0..warm {
            cur = lim.apply((1152, 648), false, out, rmin, rmax);
        }
        let step = (648 - 540) / xess::RAMP_FRAMES as usize; // px height / frame
        if !lim.ramping() || cur.1 + step >= 612 {
            eprintln!(
                "ramp: growth-guard premise broken (cur {cur:?}, ramping {}) — retune the gate",
                lim.ramping()
            );
            pass = false;
        }
        let r = lim.apply((1116, 612), true, out, rmin, rmax);
        if lim.endpoint() != (1152, 648) || r.1 >= 612 {
            eprintln!(
                "ramp: emergency grew above the current output (r {r:?}, endpoint {:?})",
                lim.endpoint()
            );
            pass = false;
        }
        // Emergency shed below the current output snaps, cancelling the ramp.
        let r = lim.apply((768, 432), true, out, rmin, rmax);
        if r != (768, 432) || lim.ramping() {
            eprintln!("ramp: emergency shed mid-ramp did not snap (r {r:?})");
            pass = false;
        }
        if lim.apply((768, 432), false, out, rmin, rmax) != (768, 432) {
            eprintln!("ramp: post-snap output unstable");
            pass = false;
        }
        // Emergency with the target already the in-flight down-ramp endpoint
        // must fast-forward the ramp (snap to the endpoint) instead of
        // descending through up to RAMP_FRAMES more blown frames.
        let mut lim = xess::StepLimiter::new(xess::RAMP_FRAMES);
        lim.apply((1280, 720), false, out, rmin, rmax);
        for _ in 0..xess::STEP_DWELL {
            lim.apply((960, 540), false, out, rmin, rmax); // last apply adopts
        }
        if !lim.ramping() || lim.endpoint() != (960, 540) {
            eprintln!("ramp: fast-forward setup did not start a down-ramp");
            pass = false;
        }
        let r = lim.apply((960, 540), true, out, rmin, rmax);
        if r != (960, 540) || lim.ramping() {
            eprintln!("ramp: emergency at the down-ramp endpoint did not fast-forward (r {r:?})");
            pass = false;
        }
        // The mirror case must NOT snap: mid UP-ramp, an emergency whose
        // target equals the (higher) endpoint would grow resolution.
        for _ in 0..xess::STEP_DWELL {
            lim.apply((1280, 720), false, out, rmin, rmax); // last apply adopts
        }
        let up = lim.apply((1280, 720), true, out, rmin, rmax);
        if xess::RAMP_FRAMES > 1 && up.1 >= 720 {
            eprintln!("ramp: emergency at an up-ramp endpoint snapped upward (r {up:?})");
            pass = false;
        }
        eprintln!("scale controller + step limiter: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 3b. Guide nearest-upscale (the post-OIDN placement feeds window-res
    // OIDN with albedo/normal guides pulled from the render-res G-buffers):
    // every destination texel of the three copied planes must bit-equal its
    // nearest source texel — indexing/stride/plane-mix-up guard. Checked at
    // identity, integer, and non-integer ratios.
    {
        let mut pass = true;
        let (sw, sh) = (64usize, 36usize);
        let src = dlss::GBufs::new(sw, sh);
        // Distinct u16 bit patterns per plane (the guides store f16 words):
        // diff_alb 0..6912, spec_alb 0x4000+.. tops out at 23295 < 0x7000,
        // normal_rough 0x7000+.. (max index 9215) tops out at 37887 — the
        // ranges are disjoint, so every element is unique across planes and
        // a cross-plane wiring swap cannot alias.
        for i in 0..sw * sh {
            for k in 0..3 {
                src.diff_alb[i * 3 + k].store((i * 3 + k) as u16, Relaxed);
            }
            for k in 0..3 {
                src.spec_alb[i * 3 + k].store(0x4000 + (i * 3 + k) as u16, Relaxed);
            }
            for k in 0..4 {
                src.normal_rough[i * 4 + k].store(0x7000 + (i * 4 + k) as u16, Relaxed);
            }
        }
        let check = |dst: &dlss::GBufs| -> bool {
            let (dw, dh) = (dst.rw, dst.rh);
            for y in 0..dh {
                let sy = (y * sh / dh).min(sh - 1);
                for x in 0..dw {
                    let sx = (x * sw / dw).min(sw - 1);
                    let (si, di) = (sy * sw + sx, y * dw + x);
                    for k in 0..3 {
                        if dst.diff_alb[di * 3 + k].load(Relaxed)
                            != src.diff_alb[si * 3 + k].load(Relaxed)
                        {
                            return false;
                        }
                    }
                    for k in 0..3 {
                        if dst.spec_alb[di * 3 + k].load(Relaxed)
                            != src.spec_alb[si * 3 + k].load(Relaxed)
                        {
                            return false;
                        }
                    }
                    for k in 0..4 {
                        if dst.normal_rough[di * 4 + k].load(Relaxed)
                            != src.normal_rough[si * 4 + k].load(Relaxed)
                        {
                            return false;
                        }
                    }
                }
            }
            true
        };
        for (dw, dh) in [(sw, sh), (2 * sw, 2 * sh), (100, 54)] {
            let dst = dlss::GBufs::new(dw, dh);
            dst.upscale_guides_from(&src);
            if !check(&dst) {
                eprintln!("guide upscale {sw}x{sh} -> {dw}x{dh}: texel mismatch");
                pass = false;
            }
        }
        eprintln!("guide nearest-upscale: {}", if pass { "OK" } else { "FAIL" });
        all_ok &= pass;
    }

    // 4. Adaptive shading rate: the same frame twice (identical seed, res,
    // frame-uniform jitter), BASE vs ADAPTIVE. Visibility must be
    // bit-identical — adaptivity may never touch it (the transplanted
    // bug-class gate); radiance may differ only by the shared-visibility
    // approximation in coherent cells; the counters must fire (default
    // scene) and account for every leaf-shaded pixel.
    {
        let (rw, rh) = (768usize, 432usize);
        // shadow_samples = 2 so the uniformity test has teeth: the penumbra
        // self-declassifier can only fire with >= 2 light samples (a single
        // sample is trivially uniform). The interactive XeSS preset is 1/1 —
        // there, penumbra correlation at cell scale is laundered temporally.
        let q = Quality {
            shadow_samples: 2,
            ao_samples: 2,
            reflections: true,
            fb: shade::FrustumBounce::OFF,
        };
        // Accumulate several jittered frames per side: two single 1-spp
        // frames differ only by the shared-visibility approximation (Apply
        // pixels reuse the rep's occlusion; the rng stream stays aligned via
        // burned draws), which is noise, not bias — the averages expose the
        // actual approximation error, and the SIGNED mean is the bias
        // detector (noise cancels, systematic error doesn't). Rep rotation
        // across frames is part of the contract under test.
        const AB_FRAMES: u32 = 16;
        let render_avg = |adaptive: bool, stats: &Stats| -> (Vec<f32>, Vec<u32>, dlss::GBufs) {
            let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
            let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
            let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
            let g = dlss::GBufs::new(rw, rh);
            for f in 0..AB_FRAMES {
                let ctx = FrameCtx {
                    scene,
                    bvh,
                    cam: cam0.basis(rw, rh),
                    q,
                    frame: f,
                    jitter: false,
                    rw,
                    rh,
                    accum: &accum,
                    info: &info,
                    tbuf: &tbuf,
                    stats,
                    sun: render::sun_dir(scene),
                    tcache_cur: None,
                    tcache_prev: &[],
                    accumulate: true,
                    gbuf: Some(&g),
                    fsr_buf: None,
                    prev_cam: None,
                    frame_jitter: Some(dlss::jitter_for(f)),
                    adaptive,
                    hemi_share: false,
                    replay_rec: None,
                    cut_cur: None,
                    cut_prev: None,
                    discard_seeds: false,
                    defer_shade: false,
                };
                render::render_frame(&ctx, true);
            }
            let inv = 1.0 / AB_FRAMES as f32;
            (
                accum.iter().map(|a| f32::from_bits(a.load(Relaxed)) * inv).collect(),
                tbuf.iter().map(|a| a.load(Relaxed)).collect(),
                g,
            )
        };
        let stats_b = Stats::default();
        let (col_b, t_b, g_b) = render_avg(false, &stats_b);
        let stats_a = Stats::default();
        let (col_a, t_a, g_a) = render_avg(true, &stats_a);

        let mism = t_a.iter().zip(&t_b).filter(|(a, b)| a != b).count();
        let vis_ok = mism == 0;
        eprintln!(
            "adaptive visibility bit-identity: {} px differ -> {}",
            mism,
            if vis_ok { "OK" } else { "FAIL (adaptivity touched visibility)" }
        );
        all_ok &= vis_ok;

        // Every G-buffer guide plane must be bit-identical too — including
        // spec_hit_t, the only rng-dependent plane (Apply burns the draws it
        // skips precisely so the GGX reflection sample stays aligned). This
        // holds per frame; the last frame's buffers are what's compared.
        // (Identical f32 inputs narrow to identical f16 bits, so the gate's
        // meaning is unchanged by the f16 storage; depth is the one f32
        // plane and gets the same compare on u32 words.)
        let planes16: [(&str, &[AtomicU16], &[AtomicU16]); 5] = [
            ("normal_rough", &g_a.normal_rough, &g_b.normal_rough),
            ("diff_alb", &g_a.diff_alb, &g_b.diff_alb),
            ("spec_alb", &g_a.spec_alb, &g_b.spec_alb),
            ("mvec", &g_a.mvec, &g_b.mvec),
            ("spec_hit_t", &g_a.spec_hit_t, &g_b.spec_hit_t),
        ];
        for (name, pa, pb) in planes16 {
            let gm = pa
                .iter()
                .zip(pb)
                .filter(|(a, b)| a.load(Relaxed) != b.load(Relaxed))
                .count();
            if gm != 0 {
                eprintln!("adaptive G-buffer bit-identity: {name}: {gm} texels differ -> FAIL");
                all_ok = false;
            }
        }
        {
            let gm = g_a
                .depth
                .iter()
                .zip(&g_b.depth)
                .filter(|(a, b)| a.load(Relaxed) != b.load(Relaxed))
                .count();
            if gm != 0 {
                eprintln!("adaptive G-buffer bit-identity: depth: {gm} texels differ -> FAIL");
                all_ok = false;
            }
        }
        eprintln!("adaptive G-buffer bit-identity (6 planes): checked");

        let lum =
            |c: &[f32], i: usize| 0.2126 * c[i * 3] + 0.7152 * c[i * 3 + 1] + 0.0722 * c[i * 3 + 2];
        let mut dsum = 0.0f64;
        let mut ssum = 0.0f64;
        let mut bsum = 0.0f64;
        for i in 0..rw * rh {
            let d = (lum(&col_a, i) - lum(&col_b, i)) as f64;
            dsum += d.abs();
            ssum += d;
            bsum += lum(&col_b, i).abs() as f64;
        }
        let rel = dsum / bsum.max(1e-9);
        let signed = ssum / bsum.max(1e-9);
        // |Δ| bounds residual noise + local approximation; the signed mean is
        // the bias gate (shared visibility/AO must not brighten or darken).
        let rad_ok = rel < 0.02 && signed.abs() < 0.005;
        eprintln!(
            "adaptive radiance A/B ({AB_FRAMES} frames): mean |Δ| {rel:.4} (limit 0.02) | signed {signed:+.4} (limit ±0.005) -> {}",
            if rad_ok { "OK" } else { "FAIL" }
        );
        all_ok &= rad_ok;

        let ld = |c: &std::sync::atomic::AtomicU64| c.load(Relaxed);
        let (coarse, base, hot) =
            (ld(&stats_a.adapt_coarse), ld(&stats_a.adapt_base), ld(&stats_a.adapt_hot));
        let partial = ld(&stats_a.adapt_partial_px);
        let topup = ld(&stats_a.adapt_topup);
        let prim = ld(&stats_a.primary_rays);
        let acct_ok = 4 * (coarse + base + hot) + partial + topup == prim;
        eprintln!(
            "adaptive accounting: {coarse}c/{base}b/{hot}h cells + {partial} edge-px + {topup} topup vs {prim} primaries -> {}",
            if acct_ok { "OK" } else { "FAIL" }
        );
        all_ok &= acct_ok;
        if structural {
            let pen = ld(&stats_a.adapt_penumbra);
            let saved = ld(&stats_a.adapt_rays_saved);
            let fired = coarse > 0 && hot > 0 && pen > 0 && saved > 0;
            eprintln!(
                "adaptive structural: coarse {coarse} hot {hot} penumbra {pen} rays-saved {saved} -> {}",
                if fired { "OK" } else { "FAIL (must-fire counters missing)" }
            );
            all_ok &= fired;
        }
        let base_clean =
            ld(&stats_b.adapt_coarse) + ld(&stats_b.adapt_base) + ld(&stats_b.adapt_hot) == 0;
        if !base_clean {
            eprintln!("adaptive counters fired on a non-adaptive frame -> FAIL");
        }
        all_ok &= base_clean;
    }

    // 5. MV/depth/matrix contract at a sweep of dynamic render resolutions:
    // two quantized 16:9 steps the controller would actually pick, plus an
    // odd-dimension literal (any res inside the range is legal).
    let out = (768usize, 432usize);
    let min = (out.0 / 3, out.1 / 3);
    for (rw, rh) in [
        xess::quantize_res(0.5, out, min, out),
        xess::quantize_res(0.8, out, min, out),
        (515, 289),
    ] {
        all_ok &= mv_check_at(scene, bvh, cam0, rw, rh, if dump { Some("xess_gbuf") } else { None });
    }

    if all_ok {
        eprintln!("XESS CHECK PASSED");
        0
    } else {
        eprintln!("XESS CHECK FAILED");
        1
    }
}

/// Headless OIDN verification: accumulate a few jittered hybrid frames with
/// G-buffer capture (the exact interactive OIDN-mode contract), denoise, and
/// gate on structural properties of the output — finite everywhere, actually
/// changed, measurably smoother (mean |Laplacian| of luminance must drop),
/// mean value preserved within 2×. A second denoise after one more
/// accumulated frame proves the commit-once/execute-many filter reuse.
/// Unlike --check/--check-dlss this needs the (license-clean, gitignored)
/// OIDN runtime DLLs on disk.
#[cfg(windows)]
/// The NPPD gate suite (--check-nppd): needs onnxruntime.dll and an exported
/// model on disk — the only NPPD check with external dependencies (the pure
/// staging math runs under --check via nppd::self_test, repeated here as G0).
/// Renders fresh 1-spp frames through the exact interactive NPPD contract
/// (accumulate = false, jitter = true, free-running seq, prev_cam set from
/// the previous frame) and gates one recurrent step at a time: frame 0 from
/// a reset state (output finite, ≠ input, smoother than input, mean
/// preserved, state populated), frame 1 static (recurrence engaged: the
/// identity-warped state must not roughen the output — structural, default
/// scene only), frame 2 under a small dolly with real motion vectors (state
/// advances, gates hold), then a reset_temporal + re-denoise (the reset path).
/// A --random-init plumbing export passes session/run wiring but fails the
/// quality gates by design — gate against the real pretrained export.
#[cfg(windows)]
fn run_check_nppd(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    opts: &Opts,
    dump: bool,
    structural: bool,
) -> i32 {
    let (rw, rh) = (800usize, 600usize);
    let mut ok = true;

    // G0: the closed-form staging self-test also runs here so --check-nppd
    // is self-contained (it is the check that owns nppd.rs).
    match nppd::self_test() {
        Ok(()) => eprintln!("nppd self-test: OK"),
        Err(e) => {
            eprintln!("nppd self-test: FAIL — {e}");
            ok = false;
        }
    }

    let dev = match opts.nppd_device {
        None => nppd::NppdDevice::Auto,
        Some(-1) => nppd::NppdDevice::Cpu,
        Some(n) => nppd::NppdDevice::Dml(n),
    };
    let mut nctx = match nppd::NppdContext::new(&opts.nppd_path, &opts.nppd_model, rw, rh, dev) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            eprintln!(
                "onnxruntime.dll expected at {} (--nppd-path / FRUSTRACER_ORT_PATH); \
                 model at {} (--nppd-model / FRUSTRACER_NPPD_MODEL — \
                 tools/nppd-export/export.py produces it)",
                opts.nppd_path, opts.nppd_model
            );
            return 1;
        }
    };

    // The interactive NPPD quality contract: fixed cheap 1-spp preset.
    let q = Quality {
        shadow_samples: 1,
        ao_samples: 1,
        reflections: true,
        fb: shade::FrustumBounce::OFF,
    };
    let stats = Stats::default();
    let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let g = dlss::GBufs::new(rw, rh);
    let (_, far) = dlss::near_far(scene.diag);
    let render_fresh = |basis: &camera::CamBasis, seq: u32, prev: Option<camera::CamBasis>| {
        let ctx = FrameCtx {
            scene,
            bvh,
            cam: *basis,
            q,
            frame: seq,
            jitter: true,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: &[],
            accumulate: false,
            gbuf: Some(&g),
            fsr_buf: None,
            prev_cam: prev,
            frame_jitter: None,
            adaptive: false,
            hemi_share: false, // inert: the NPPD contract pins fb OFF
            replay_rec: None,
            cut_cur: None,
            cut_prev: None,
            discard_seeds: false,
            defer_shade: false,
        };
        render::render_frame(&ctx, true);
    };
    let load = |a: &AtomicU32| f32::from_bits(a.load(Relaxed));
    let lum = |b: &[f32], x: usize, y: usize| {
        let i = (y * rw + x) * 3;
        0.2126 * b[i] + 0.7152 * b[i + 1] + 0.0722 * b[i + 2]
    };
    let roughness = |b: &[f32]| -> f64 {
        let mut s = 0.0f64;
        for y in 1..rh - 1 {
            for x in 1..rw - 1 {
                let l = 4.0 * lum(b, x, y)
                    - lum(b, x - 1, y)
                    - lum(b, x + 1, y)
                    - lum(b, x, y - 1)
                    - lum(b, x, y + 1);
                s += l.abs() as f64;
            }
        }
        s / ((rw - 2) * (rh - 2)) as f64
    };
    let mean = |b: &[f32]| b.iter().map(|&v| v as f64).sum::<f64>() / b.len() as f64;
    // Shared output gates: finite, ≠ input, smoother than the 1-spp input,
    // mean-value preserved within 2×.
    let gate_output = |out: &[f32], input: &[f32], what: &str, ms: f64, ok: &mut bool| -> (f64, f64) {
        if !out.iter().all(|v| v.is_finite()) {
            eprintln!("nppd {what}: output contains non-finite values");
            *ok = false;
        }
        let max_diff = input.iter().zip(out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if max_diff <= 1e-6 {
            eprintln!("nppd {what}: output identical to input (max diff {max_diff:.2e})");
            *ok = false;
        }
        let (r_in, r_out) = (roughness(input), roughness(out));
        let ratio = mean(out) / mean(input).max(1e-9);
        eprintln!(
            "nppd {what}: mean |laplacian| {r_in:.4} -> {r_out:.4} | mean ratio {ratio:.3} | {ms:.1} ms"
        );
        if r_out >= r_in {
            eprintln!("nppd {what}: output not smoother than input");
            *ok = false;
        }
        if !(0.5..=2.0).contains(&ratio) {
            eprintln!("nppd {what}: mean value ratio {ratio:.3} outside [0.5, 2.0]");
            *ok = false;
        }
        (r_in, r_out)
    };

    // G1: frame 0 from a reset state.
    let basis = cam0.basis(rw, rh);
    render_fresh(&basis, 0, None);
    let input0: Vec<f32> = accum.iter().map(load).collect();
    if nctx.temporal_valid() {
        eprintln!("nppd: state valid before the first denoise");
        ok = false;
    }
    let out0 = match nctx.denoise(&accum, &g, &basis, far) {
        Ok(o) => o.to_vec(),
        Err(e) => {
            eprintln!("{e}");
            eprintln!("NPPD CHECK FAILED");
            return 1;
        }
    };
    let (_, r_out0) = gate_output(&out0, &input0, "frame 0 (reset)", nctx.last_ms, &mut ok);
    if !nctx.temporal_valid() {
        eprintln!("nppd: state not marked valid after a denoise");
        ok = false;
    }
    if nctx.state().iter().all(|&v| v == 0.0) {
        eprintln!("nppd: recurrent state is all-zero after a denoise");
        ok = false;
    }

    // G2: frame 1 static — the identity-warped state feeds the second step;
    // recurrence must not roughen the output (structural: on the default
    // scene the temporal blend demonstrably engages and smooths further).
    render_fresh(&basis, 1, Some(basis));
    let input1: Vec<f32> = accum.iter().map(load).collect();
    let state0: Vec<f32> = nctx.state().to_vec();
    let out1 = match nctx.denoise(&accum, &g, &basis, far) {
        Ok(o) => o.to_vec(),
        Err(e) => {
            eprintln!("{e}");
            eprintln!("NPPD CHECK FAILED");
            return 1;
        }
    };
    let (_, r_out1) = gate_output(&out1, &input1, "frame 1 (static)", nctx.last_ms, &mut ok);
    if nctx.state() == &state0[..] {
        eprintln!("nppd: recurrent state did not advance across a step");
        ok = false;
    }
    if structural && r_out1 > r_out0 * 1.05 {
        eprintln!(
            "nppd static recurrence: frame-1 |laplacian| {r_out1:.4} > frame-0 {r_out0:.4} — temporal accumulation didn't engage"
        );
        ok = false;
    }

    // G3: frame 2 under a small forward dolly (the --check constant) with
    // prev_cam set — real motion vectors drive the state warp.
    let mut cam_b = cam0;
    cam_b.pos += cam0.forward() * (0.02 * scene.diag);
    let basis_b = cam_b.basis(rw, rh);
    render_fresh(&basis_b, 2, Some(basis));
    let input2: Vec<f32> = accum.iter().map(load).collect();
    let out2 = match nctx.denoise(&accum, &g, &basis_b, far) {
        Ok(o) => o.to_vec(),
        Err(e) => {
            eprintln!("{e}");
            eprintln!("NPPD CHECK FAILED");
            return 1;
        }
    };
    gate_output(&out2, &input2, "frame 2 (dolly)", nctx.last_ms, &mut ok);

    // G4: reset + re-denoise — the reset path is exercised end to end.
    nctx.reset_temporal();
    if nctx.temporal_valid() {
        eprintln!("nppd: reset_temporal did not invalidate the state");
        ok = false;
    }
    render_fresh(&basis_b, 3, None);
    match nctx.denoise(&accum, &g, &basis_b, far) {
        Ok(o) => {
            if !o.iter().all(|v| v.is_finite()) {
                eprintln!("nppd: post-reset denoise produced non-finite values");
                ok = false;
            }
            eprintln!("nppd: post-reset denoise {:.1} ms", nctx.last_ms);
        }
        Err(e) => {
            eprintln!("{e}");
            ok = false;
        }
    }

    if dump {
        let mut present = vec![0u32; rw * rh];
        for (name, data) in [("nppd_before", &input0), ("nppd_after", &out0), ("nppd_dolly", &out2)]
        {
            render::resolve_hdr(data, &info, false, &mut present, rw, rh, rw, rh);
            save_png(&format!("{name}.png"), &present, rw, rh);
            eprintln!("wrote {name}.png");
        }
    }

    if ok {
        eprintln!("NPPD CHECK PASSED");
        0
    } else {
        eprintln!("NPPD CHECK FAILED");
        1
    }
}

fn run_check_oidn(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    opts: &Opts,
    dump: bool,
    structural: bool,
) -> i32 {
    let (rw, rh) = (800usize, 600usize);
    let mut octx = match oidn::OidnContext::new(
        &opts.oidn_path,
        rw,
        rh,
        opts.oidn_device,
        opts.oidn_quality,
        opts.oidn_clean_aux,
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            eprintln!(
                "OIDN DLLs expected at {} (override with --oidn-path or FRUSTRACER_OIDN_PATH)",
                opts.oidn_path
            );
            return 1;
        }
    };
    eprintln!("oidn: device {}", octx.device_desc);

    let q = Quality::preset(2);
    let stats = Stats::default();
    let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let g = dlss::GBufs::new(rw, rh);
    let basis = cam0.basis(rw, rh);
    let render_accum_frame = |frame: u32| {
        let ctx = FrameCtx {
            scene,
            bvh,
            cam: basis,
            q,
            frame,
            jitter: frame > 0,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: &[],
            accumulate: true,
            gbuf: Some(&g),
            fsr_buf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
            hemi_share: false,
            replay_rec: None,
            cut_cur: None,
            cut_prev: None,
            discard_seeds: false,
            defer_shade: false,
        };
        render::render_frame(&ctx, true);
    };
    const WARM: u32 = 4;
    for f in 0..WARM {
        render_accum_frame(f);
    }

    let inv = 1.0 / WARM as f32;
    let input: Vec<f32> =
        accum.iter().map(|a| f32::from_bits(a.load(Relaxed)) * inv).collect();
    let out = match octx.denoise(&accum, WARM, &g) {
        Ok(o) => o.to_vec(),
        Err(e) => {
            eprintln!("{e}");
            eprintln!("OIDN CHECK FAILED");
            return 1;
        }
    };
    eprintln!("oidn: denoised {rw}x{rh} in {:.1} ms", octx.last_ms);

    let mut ok = true;
    if !out.iter().all(|v| v.is_finite()) {
        eprintln!("oidn: output contains non-finite values");
        ok = false;
    }
    let max_diff =
        input.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    if max_diff <= 1e-6 {
        eprintln!("oidn: output identical to input (max diff {max_diff:.2e}) — filter did nothing");
        ok = false;
    }

    // Noise metric on interior pixels: denoising must strictly reduce the
    // mean |Laplacian| of luminance while roughly preserving the mean value.
    let lum = |b: &[f32], x: usize, y: usize| {
        let i = (y * rw + x) * 3;
        0.2126 * b[i] + 0.7152 * b[i + 1] + 0.0722 * b[i + 2]
    };
    let roughness = |b: &[f32]| -> f64 {
        let mut s = 0.0f64;
        for y in 1..rh - 1 {
            for x in 1..rw - 1 {
                let l = 4.0 * lum(b, x, y)
                    - lum(b, x - 1, y)
                    - lum(b, x + 1, y)
                    - lum(b, x, y - 1)
                    - lum(b, x, y + 1);
                s += l.abs() as f64;
            }
        }
        s / ((rw - 2) * (rh - 2)) as f64
    };
    let (r_in, r_out) = (roughness(&input), roughness(&out));
    let mean = |b: &[f32]| b.iter().map(|&v| v as f64).sum::<f64>() / b.len() as f64;
    let ratio = mean(&out) / mean(&input).max(1e-9);
    eprintln!("oidn: mean |laplacian| {r_in:.4} -> {r_out:.4} | mean value ratio {ratio:.3}");
    if r_out >= r_in {
        eprintln!("oidn: output not smoother than input");
        ok = false;
    }
    if !(0.5..=2.0).contains(&ratio) {
        eprintln!("oidn: mean value ratio {ratio:.3} outside [0.5, 2.0]");
        ok = false;
    }

    // One more accumulated frame + re-denoise: the per-frame reuse contract
    // (images bound and filter committed once; only buffer contents change).
    render_accum_frame(WARM);
    match octx.denoise(&accum, WARM + 1, &g) {
        Ok(o) => {
            if !o.iter().all(|v| v.is_finite()) {
                eprintln!("oidn: second denoise produced non-finite values");
                ok = false;
            }
            eprintln!("oidn: second denoise (filter reuse) {:.1} ms", octx.last_ms);
        }
        Err(e) => {
            eprintln!("{e}");
            ok = false;
        }
    }

    // ---- Temporal reprojection gates (G0-G5): scene-level validation of
    // reproject::History against the exact interactive temporal contract
    // (fresh 1-spp frames: accumulate=false, jitter=true, free-running seq).
    // Deterministic — the per-pixel RNG is seeded from (x, y, frame) only.
    // Camera moves reuse the --check constants (0.02·diag dolly, +0.05 yaw,
    // cap d=4). Pure CPU math on the shared buffers; the OIDN DLLs above are
    // the only external dependency of this check.
    let (_, far) = dlss::near_far(scene.diag);
    // G0: the closed-form self-test also runs here so --check-oidn is
    // self-contained (it is the check that owns reproject.rs).
    match reproject::self_test() {
        Ok(()) => eprintln!("reproject self-test: OK"),
        Err(e) => {
            eprintln!("reproject self-test: FAIL — {e}");
            ok = false;
        }
    }
    // G2 thresholds, tuned on the default scene (structural-gated):
    // forward dolly keeps the screen edges on-screen, so rejections are
    // silhouette disocclusions + sky/geom transitions — small but nonzero.
    const REJ_FRAC_MAX: f64 = 0.10;
    // World-point agreement is pure geometry (shading-noise-immune): the
    // point the previous frame stored at the fetched texel, projected back
    // into the CURRENT screen, must land within ~a texel of the consuming
    // pixel. Screen-space pixels make the gate scale-invariant — a 3D
    // distance gate has an error floor of one texel's world footprint, which
    // grows with depth/grazing angle and broke on the stress scene. The
    // error floor by construction: fresh frames jitter per-pixel, so both
    // depths belong to random points inside their pixels (~0.7 px each) plus
    // 0.5 px nearest-tap rounding — median ~1 px is geometry agreeing; a
    // real reprojection defect (sign, basis) measures tens to hundreds.
    const WP_MEDIAN_PX: f32 = 1.5;
    const WP_P90_PX: f32 = 4.0;
    // An 8-deep history cuts 1-spp luminance error ~3x vs a converged
    // reference; 0.7 leaves margin for bilinear blur and view-dependence.
    const CONV_GAIN_MAX: f64 = 0.7;
    const CONV_ABS_MAX: f64 = 0.2;

    let render_fresh = |basis: &camera::CamBasis, seq: u32, cap: Option<u32>| {
        let ctx = FrameCtx {
            scene,
            bvh,
            cam: *basis,
            q,
            frame: seq,
            jitter: true,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: &[],
            accumulate: false,
            gbuf: Some(&g),
            fsr_buf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
            hemi_share: false,
            replay_rec: None,
            cut_cur: None,
            cut_prev: None,
            discard_seeds: false,
            defer_shade: false,
        };
        match cap {
            Some(d) => render::render_frame_capped(&ctx, d),
            None => render::render_frame(&ctx, true),
        }
    };
    let load = |a: &AtomicU32| f32::from_bits(a.load(Relaxed));
    // G5: resets are the only writers of L == 1 (coarse_kept may legitimately
    // carry an inherited L of 1, hence the upper slack).
    let l_accounting = |hist: &reproject::History, st: &reproject::UpdateStats, what: &str| {
        let ones = hist.sample_counts().iter().filter(|&&l| l == 1.0).count() as u64;
        let lo = st.rejected + st.coarse_reset;
        let hi = lo + st.coarse_kept;
        if ones < lo || ones > hi {
            eprintln!("reproject {what}: #(L==1) = {ones} outside [{lo}, {hi}]");
            return false;
        }
        true
    };
    let finite = |hist: &reproject::History, what: &str| {
        if hist.color().iter().all(|v| v.is_finite()) {
            true
        } else {
            eprintln!("reproject {what}: history contains non-finite values");
            false
        }
    };

    // G1: static replay. Fresh frames at a fixed basis take the identity
    // path and the history must be the exact running mean (fp tolerance).
    let mut hist = reproject::History::new(rw, rh);
    let mut sum = vec![0f64; rw * rh * 3];
    const T_WARM: u32 = 8;
    for f in 0..T_WARM {
        render_fresh(&basis, f, None);
        for (s, a) in sum.iter_mut().zip(accum.iter()) {
            *s += load(a) as f64;
        }
        let st = hist.update(&basis, &accum, &g, &info, far, MAX_SAMPLES as f32);
        let good = if f == 0 {
            st.rejected == (rw * rh) as u64 && !st.identity
        } else {
            st.identity && st.rejected == 0 && st.coarse_kept + st.coarse_reset == 0
        };
        if !good {
            eprintln!(
                "reproject static f{f}: identity {} rejected {} coarse {}/{}",
                st.identity, st.rejected, st.coarse_kept, st.coarse_reset
            );
            ok = false;
        }
        if f == 3 {
            // Snapshot gates at frame 4 (the plan's G1 point): L uniform,
            // history == sum/4 within fp-rounding of a running mean.
            if st.len_min != 4.0 || st.len_max != 4.0 {
                eprintln!("reproject static: L range [{}, {}], want [4, 4]", st.len_min, st.len_max);
                ok = false;
            }
            let mut max_rel = 0f64;
            for (i, &v) in hist.color().iter().enumerate() {
                let m = sum[i] / 4.0;
                max_rel = max_rel.max((v as f64 - m).abs() / m.abs().max(1.0));
            }
            if max_rel >= 1e-4 {
                eprintln!("reproject static: max |hist - mean| = {max_rel:.2e} (limit 1e-4)");
                ok = false;
            }
        }
        ok &= l_accounting(&hist, &st, &format!("static f{f}"));
    }
    ok &= finite(&hist, "static");

    // G2: forward dolly against an 8-deep history.
    let prev_depth: Vec<f32> = g.depth.iter().map(load).collect();
    let mut cam_b = cam0;
    cam_b.pos += cam0.forward() * (0.02 * scene.diag);
    let basis_b = cam_b.basis(rw, rh);
    render_fresh(&basis_b, T_WARM, None);
    let fresh_b: Vec<f32> = accum.iter().map(load).collect();
    let t_upd = Instant::now();
    let st_b = hist.update(&basis_b, &accum, &g, &info, far, MAX_SAMPLES as f32);
    // Print-only perf diagnostic — the one wall-clock read in this check.
    // Never gate on it: everything gated must stay deterministic.
    let upd_ms = t_upd.elapsed().as_secs_f64() * 1000.0;
    // Dump snapshot here (post-dolly, pre-G4): the G4 budget frame legitimately
    // floods the history with flat quads wherever it was disoccluded.
    let hist_dolly: Vec<f32> = if dump { hist.color().to_vec() } else { Vec::new() };
    let depth_b: Vec<f32> = g.depth.iter().map(load).collect();
    let rough_b: Vec<f32> =
        (0..rw * rh).map(|i| dlss::ld16(&g.normal_rough[i * 4 + 3])).collect();
    let rej_frac = st_b.rejected as f64 / (rw * rh) as f64;
    eprintln!(
        "reproject dolly: accepted {} ({} sky) rejected {} ({:.2}%) | L {:.0}..{:.0} | update {upd_ms:.2} ms",
        st_b.accepted,
        st_b.sky_accepted,
        st_b.rejected,
        rej_frac * 100.0,
        st_b.len_min,
        st_b.len_max
    );
    if rej_frac >= REJ_FRAC_MAX {
        eprintln!("reproject dolly: rejection fraction {rej_frac:.3} >= {REJ_FRAC_MAX}");
        ok = false;
    }
    if structural && st_b.rejected == 0 {
        eprintln!("reproject dolly: expected silhouette disocclusions > 0 — rejection never fired");
        ok = false;
    }
    ok &= l_accounting(&hist, &st_b, "dolly");
    ok &= finite(&hist, "dolly");
    // World-point agreement on a sparse deterministic subset of accepted
    // geometry pixels: the point this frame saw, reprojected, must be the
    // point the previous frame stored at that texel (both reconstructed from
    // depth — immune to shading noise).
    {
        let sky_z = 0.99 * far;
        let mut errs: Vec<f32> = Vec::new();
        for y in 0..rh {
            for x in 0..rw {
                let i = y * rw + x;
                if (x * 7 + y * 13) % 97 != 0 || hist.mask()[i] == 0 || depth_b[i] >= sky_z {
                    continue;
                }
                let dir = basis_b.ray_dir(x as f32 + 0.5, y as f32 + 0.5);
                let p_cur = basis_b.origin + dir * (depth_b[i] / dir.dot(basis_b.forward()));
                let Some((px, py)) = basis.project(p_cur - basis.origin) else { continue };
                let (tx, ty) = (px.round() as i64, py.round() as i64);
                if tx < 0 || ty < 0 || tx >= rw as i64 || ty >= rh as i64 {
                    continue;
                }
                let ti = ty as usize * rw + tx as usize;
                let pz = prev_depth[ti];
                if pz >= sky_z {
                    continue;
                }
                let pdir = basis.ray_dir(tx as f32 + 0.5, ty as f32 + 0.5);
                let p_prev = basis.origin + pdir * (pz / pdir.dot(basis.forward()));
                let Some((qx, qy)) = basis_b.project(p_prev - basis_b.origin) else { continue };
                errs.push(((qx - (x as f32 + 0.5)).powi(2) + (qy - (y as f32 + 0.5)).powi(2)).sqrt());
            }
        }
        errs.sort_by(|a, b| a.total_cmp(b));
        if errs.is_empty() {
            eprintln!("reproject dolly: world-point sample set is empty");
            ok = false;
        } else {
            let med = errs[errs.len() / 2];
            let p90 = errs[errs.len() * 9 / 10];
            eprintln!(
                "reproject dolly world-point agreement ({} samples): median {med:.3} px p90 {p90:.3} px",
                errs.len(),
            );
            if med >= WP_MEDIAN_PX || p90 >= WP_P90_PX {
                eprintln!("reproject dolly: world-point agreement out of bounds");
                ok = false;
            }
        }
    }
    // Convergence-beats-1-spp: on accepted diffuse geometry pixels the
    // history must track a 16-frame converged reference strictly better than
    // the fresh 1-spp frame does (self-normalizing against scene brightness).
    if structural {
        for f in 0..16u32 {
            let ctx = FrameCtx {
                scene,
                bvh,
                cam: basis_b,
                q,
                frame: f,
                jitter: f > 0,
                rw,
                rh,
                accum: &accum,
                info: &info,
                tbuf: &tbuf,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: None,
                tcache_prev: &[],
                accumulate: true,
                gbuf: Some(&g),
                fsr_buf: None,
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
                hemi_share: false,
                replay_rec: None,
                cut_cur: None,
                cut_prev: None,
                discard_seeds: false,
                defer_shade: false,
            };
            render::render_frame(&ctx, true);
        }
        let ref16: Vec<f32> = accum.iter().map(|a| load(a) / 16.0).collect();
        let lum3 = |b: &[f32], i: usize| {
            0.2126 * b[i * 3] as f64 + 0.7152 * b[i * 3 + 1] as f64 + 0.0722 * b[i * 3 + 2] as f64
        };
        let sky_z = 0.99 * far;
        let (mut e_hist, mut e_fresh, mut n) = (0f64, 0f64, 0u64);
        for i in 0..rw * rh {
            if hist.mask()[i] == 0 || depth_b[i] >= sky_z || rough_b[i] < 0.5 {
                continue;
            }
            e_hist += (lum3(hist.color(), i) - lum3(&ref16, i)).abs();
            e_fresh += (lum3(&fresh_b, i) - lum3(&ref16, i)).abs();
            n += 1;
        }
        if n == 0 {
            eprintln!("reproject dolly: convergence sample set is empty");
            ok = false;
        } else {
            e_hist /= n as f64;
            e_fresh /= n as f64;
            eprintln!(
                "reproject dolly vs 16-frame reference ({n} diffuse px): hist err {e_hist:.4} | 1-spp err {e_fresh:.4} (gain limit {CONV_GAIN_MAX})"
            );
            if e_hist >= CONV_GAIN_MAX * e_fresh || e_hist >= CONV_ABS_MAX {
                eprintln!("reproject dolly: history does not beat 1-spp against the reference");
                ok = false;
            }
        }
    }

    // G3: pure yaw. Sky reprojects exactly under rotation (direction-only),
    // and the newly-exposed edge band reprojects off the old screen.
    {
        let mut h3 = reproject::History::new(rw, rh);
        for f in 0..2u32 {
            render_fresh(&basis, 100 + f, None);
            h3.update(&basis, &accum, &g, &info, far, MAX_SAMPLES as f32);
        }
        let mut cam_y = cam0;
        cam_y.yaw += 0.05;
        let basis_y = cam_y.basis(rw, rh);
        render_fresh(&basis_y, 102, None);
        let st = h3.update(&basis_y, &accum, &g, &info, far, MAX_SAMPLES as f32);
        eprintln!(
            "reproject yaw: accepted {} ({} sky) rejected {}",
            st.accepted, st.sky_accepted, st.rejected
        );
        if structural && st.sky_accepted == 0 {
            eprintln!("reproject yaw: expected sky reprojection > 0 — the sky path didn't fire");
            ok = false;
        }
        if structural && st.rejected == 0 {
            eprintln!("reproject yaw: expected the exposed edge band to reject");
            ok = false;
        }
        ok &= l_accounting(&h3, &st, "yaw");
        ok &= finite(&h3, "yaw");
    }

    // G4: a depth-capped budget frame while moving — coarse quads over
    // previously-visible geometry must keep the history (blend weight 0),
    // and the moving-frame length cap must hold. Reuses the dolly history
    // (prev state = frame B); same d=4 cap as --check.
    {
        let mut cam_c = cam_b;
        cam_c.pos += cam0.forward() * (0.02 * scene.diag);
        let basis_c = cam_c.basis(rw, rh);
        let smp0 = stats.coarse_samples.load(Relaxed);
        render_fresh(&basis_c, 200, Some(4));
        let smp_c = stats.coarse_samples.load(Relaxed) - smp0;
        let coarse_px = info
            .iter()
            .filter(|i| overlay::info_kind(i.load(Relaxed)) == overlay::KIND_COARSE)
            .count();
        let st = hist.update(&basis_c, &accum, &g, &info, far, MAX_SAMPLES as f32);
        eprintln!(
            "reproject capped d=4: coarse px {} samples {} | kept {} reset {} | L {:.0}..{:.0}",
            coarse_px, smp_c, st.coarse_kept, st.coarse_reset, st.len_min, st.len_max
        );
        if coarse_px == 0 {
            eprintln!("reproject capped: no coarse pixels — the capped path didn't run");
            ok = false;
        }
        // Coarse pixels imply per-cell point samples on any scene: the
        // samples must have gone through the normal (non-coarse) blend path.
        if coarse_px > 0 && (smp_c == 0 || st.accepted + st.rejected == 0) {
            eprintln!(
                "reproject capped: samples {} accepted {} rejected {} — sparse samples didn't blend",
                smp_c, st.accepted, st.rejected
            );
            ok = false;
        }
        if structural && st.coarse_kept == 0 {
            eprintln!("reproject capped: expected coarse-kept > 0 — the mask rule didn't fire");
            ok = false;
        }
        if st.len_max > reproject::L_MAX {
            eprintln!("reproject capped: L max {} exceeds the moving cap {}", st.len_max, reproject::L_MAX);
            ok = false;
        }
        ok &= l_accounting(&hist, &st, "capped");
        ok &= finite(&hist, "capped");
    }

    if dump {
        let mut present = vec![0u32; rw * rh];
        render::resolve_hdr(&input, &info, false, &mut present, rw, rh, rw, rh);
        save_png("oidn_before.png", &present, rw, rh);
        render::resolve_hdr(&out, &info, false, &mut present, rw, rh, rw, rh);
        save_png("oidn_after.png", &present, rw, rh);
        render::resolve_hdr(&hist_dolly, &info, false, &mut present, rw, rh, rw, rh);
        save_png("oidn_hist.png", &present, rw, rh);
        dlss::dump_gbufs(&g, "oidn_gbuf", far);
        eprintln!("wrote oidn_before.png / oidn_after.png / oidn_hist.png / oidn_gbuf_*.png");
    }

    if ok {
        eprintln!("OIDN CHECK PASSED");
        0
    } else {
        eprintln!("OIDN CHECK FAILED");
        1
    }
}

fn default_camera() -> Camera {
    scaled_camera(1.0)
}

/// The default view pushed back along its own look direction: eye and target
/// scale by `k`, so a `k`-times-wider scene gets the same framing.
fn scaled_camera(k: f32) -> Camera {
    Camera::look_at(
        Vec3A::new(11.0, 6.5, 13.0) * k,
        Vec3A::new(0.0, 1.2 * k, 0.0),
        55f32.to_radians(),
    )
}

/// One deterministic probe for the bounce-integrator check sections: a
/// primary-hit surface point (with its shading normal) on the check camera.
struct Probe {
    x: usize,
    y: usize,
    p: Vec3A,
    n: Vec3A,
}

/// The deterministic probe sweep shared by the hemi-AO, hemi-GI, and shaft
/// check sections — every section MUST run on this exact set (the A/B gates
/// compare like-for-like only because integrator and reference share points).
fn collect_probes(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam: &camera::CamBasis,
    rw: usize,
    rh: usize,
) -> Vec<Probe> {
    let mut vis = 0u64;
    let mut probes = Vec::new();
    for y in 0..rh {
        for x in 0..rw {
            if (x * 7 + y * 13) % 397 != 0 {
                continue;
            }
            let dir = cam.ray_dir(x as f32 + 0.5, y as f32 + 0.5);
            let ray = bvh::Ray::new(cam.origin, dir);
            let Some(hit) = bvh.intersect(scene, &ray, 0.0, f32::INFINITY, &mut vis) else {
                continue;
            };
            let (p, n) = shade::surface_point(scene, &ray, &hit);
            probes.push(Probe { x, y, p, n });
        }
    }
    probes
}

/// Per-pixel deterministic RNG seed for the probe sweeps (`s` salts the
/// stratified repeats and separates integrator from reference streams).
fn px_seed(x: usize, y: usize, s: u64) -> u64 {
    (x as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F))
        .wrapping_add(s)
}

/// Centripetal-free (uniform) Catmull-Rom through p1..p2 at t ∈ [0, 1).
fn catmull_rom(p0: Vec3A, p1: Vec3A, p2: Vec3A, p3: Vec3A, t: f32) -> Vec3A {
    ((p1 * 2.0)
        + (p2 - p0) * t
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (t * t)
        + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * (t * t * t))
        * 0.5
}

/// Frames per full lap of the benchmark path.
const SPIN_LAP: f32 = 600.0;

/// Deterministic benchmark camera: a CLOSED-loop Catmull-Rom spline keyed
/// relative to cam0 (offsets in units of scene.diag along cam0's
/// forward/right/world-up, plus yaw/pitch offsets). The pose is a pure
/// function of the frame index — no wall clock — so runs are bit-repeatable
/// and A/B-comparable. The loop mixes dolly, strafe, climb, and yaw sweeps
/// and returns near its start each lap, so every temporal regime (seeds,
/// query skips, off-screen ring retries, pan-back) is exercised per lap.
fn spin_path_pose(cam0: &Camera, diag: f32, frame: u32) -> Camera {
    // (fwd, right, up) offsets /diag, yaw offset, pitch offset.
    const KEYS: [([f32; 3], f32, f32); 6] = [
        ([0.00, 0.00, 0.00], 0.00, 0.00),
        ([0.08, 0.03, 0.01], 0.10, -0.03),
        ([0.12, -0.02, 0.03], -0.15, 0.02),
        ([0.05, -0.09, 0.02], -0.35, 0.05),
        ([-0.02, -0.04, 0.04], -0.10, 0.08),
        ([-0.01, 0.02, 0.01], 0.05, 0.02),
    ];
    let n = KEYS.len() as isize;
    let fwd = cam0.forward();
    let right = fwd.cross(Vec3A::Y).normalize();
    let t = frame as f32 * (KEYS.len() as f32 / SPIN_LAP);
    let seg = t.floor() as isize;
    let u = t.fract();
    let key = |i: isize| {
        let (o, yw, pt) = KEYS[((seg + i).rem_euclid(n)) as usize];
        (
            (fwd * o[0] + right * o[1] + Vec3A::Y * o[2]) * diag,
            Vec3A::new(yw, pt, 0.0),
        )
    };
    let (p0, a0) = key(-1);
    let (p1, a1) = key(0);
    let (p2, a2) = key(1);
    let (p3, a3) = key(2);
    let pos = catmull_rom(p0, p1, p2, p3, u);
    let ang = catmull_rom(a0, a1, a2, a3, u);
    Camera {
        pos: cam0.pos + pos,
        yaw: cam0.yaw + ang.x,
        pitch: cam0.pitch + ang.y,
        fov_y: cam0.fov_y,
    }
}

/// Ring depth: the last TRING producing frames' caches stay consultable.
const TRING: usize = 3;

/// The temporal claim ring's per-frame state machine — the last `TRING`
/// producing frames' caches (+ the basis each traced with, newest first;
/// older entries answer regions that panned off the newest screen and back —
/// a claim never goes stale in a static scene, only wrong-basis/wrong-res
/// pairing could hurt, so each entry carries its basis and the whole ring
/// drops on a res change), plus the cut stores double-buffered in lockstep:
/// cur is produced this frame, prev pairs with ring[0] — the SAME frame,
/// same basis, by construction (producing frames update both, replay frames
/// freeze both, res changes / non-participating frames drop both).
///
/// Shared by the interactive loop and --spin so the benchmark can never
/// drift from the pipeline it claims to measure: `begin` before the render
/// hands out the borrows FrameCtx needs, `end` after it rotates (producing
/// frame) or drops (non-participating frame) the ring. Replay frames call
/// neither — the ring freezes, per the temporal contract.
struct TemporalRing {
    /// TRING + 1 buffers so a victim always exists outside the ring.
    caches: Vec<temporal::TemporalCache>,
    cutstores: [temporal::CutStore; 2],
    ring: Vec<(usize, camera::CamBasis)>,
    /// Res the ring's claims were traced at.
    res: (usize, usize),
    cut_cur_i: usize,
    cut_prev_ok: bool,
    victim: usize,
}

impl TemporalRing {
    fn new(rw: usize, rh: usize) -> Self {
        TemporalRing {
            caches: (0..TRING + 1).map(|_| temporal::TemporalCache::new(rw, rh)).collect(),
            cutstores: [temporal::CutStore::new(rw, rh), temporal::CutStore::new(rw, rh)],
            ring: Vec::new(),
            res: (0, 0),
            cut_cur_i: 0,
            cut_prev_ok: false,
            victim: usize::MAX,
        }
    }

    /// Producing-frame setup: drop all entries on a res change (claims are
    /// consumed at the res they were traced at — the documented contract),
    /// pick a victim buffer outside the ring and clear it, clear the current
    /// cut store, and hand out the borrows FrameCtx needs — the current
    /// cache, the ring (newest first), and the (cur, prev) cut pair (prev
    /// only when the last frame produced one and the ring is nonempty).
    #[allow(clippy::type_complexity)]
    fn begin(
        &mut self,
        temporal_on: bool,
        adopt: bool,
        rw: usize,
        rh: usize,
    ) -> (
        Option<&temporal::TemporalCache>,
        Vec<(&temporal::TemporalCache, camera::CamBasis)>,
        Option<&temporal::CutStore>,
        Option<&temporal::CutStore>,
    ) {
        if !temporal_on {
            return (None, Vec::new(), None, None);
        }
        zone!("temporal-admin"); // MB-scale cache clear + ring build
        if self.res != (rw, rh) {
            self.ring.clear();
            self.cut_prev_ok = false;
            self.res = (rw, rh);
        }
        self.victim =
            (0..self.caches.len()).find(|i| !self.ring.iter().any(|(j, _)| j == i)).unwrap();
        self.caches[self.victim].clear();
        let (cut_cur, cut_prev) = if adopt {
            zone!("cutstore-clear");
            self.cutstores[self.cut_cur_i].clear();
            (
                Some(&self.cutstores[self.cut_cur_i]),
                if self.cut_prev_ok && !self.ring.is_empty() {
                    Some(&self.cutstores[self.cut_cur_i ^ 1])
                } else {
                    None
                },
            )
        } else {
            (None, None)
        };
        let mut tprev = Vec::with_capacity(self.ring.len());
        for &(i, b) in &self.ring {
            tprev.push((&self.caches[i], b));
        }
        (Some(&self.caches[self.victim]), tprev, cut_cur, cut_prev)
    }

    /// Post-render bookkeeping for a NON-replay frame. Producing frame: the
    /// victim becomes the newest ring entry (the oldest rotates out and
    /// becomes the next frame's victim) and the cut store flips in lockstep
    /// so prev always pairs with ring[0] — only a produced store may pair.
    /// Non-participating frame (plain / half-res): the ring drops wholesale
    /// — the old `tprev_ok = false` contract.
    fn end(&mut self, temporal_on: bool, adopt: bool, basis: camera::CamBasis) {
        if temporal_on {
            self.ring.insert(0, (self.victim, basis));
            self.ring.truncate(TRING);
            self.cut_prev_ok = adopt;
            self.cut_cur_i ^= 1;
        } else {
            self.ring.clear();
            self.cut_prev_ok = false;
        }
    }
}

/// Headless deterministic workload driver (--spin still|path): replicates
/// the interactive frame contract — the replay/trace arm split, temporal
/// ring rotation, cut-store pairing, recording gate — at the interactive
/// native res with the 1-spp upscaler quality and free-running Halton
/// jitter, but with no window, no denoiser, and no wall-clock dependence in
/// the workload. Composes with --no-temporal/--no-replay/--no-adopt, which
/// makes it both the profiling target (attach Tracy or a sampler) and the
/// reproducible benchmark for caching A/Bs. Prints a mean-ms summary
/// (warmup excluded) and the per-phase counters.
fn run_spin(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    cam0: Camera,
    mode: &str,
    frames: u32,
    opts: &Opts,
) -> i32 {
    let moving = match mode {
        "still" => false,
        "path" => true,
        _ => {
            eprintln!("--spin: unknown workload {mode} (use still | path)");
            return 2;
        }
    };
    if opts.gpu {
        eprintln!("--spin: drives the CPU renderer only; ignoring --gpu");
    }
    let (rw, rh) = (W, H);
    let q = Quality::upscaler_1spp();
    let stats = Stats::default();
    let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let mut tr = TemporalRing::new(rw, rh);
    let replay_cache = replay::ReplayCache::new(rw, rh);
    let mut replay_key: Option<camera::CamBasis> = None;

    eprintln!(
        "spin {mode}: {frames} frames at {rw}x{rh}, 1-spp quality | temporal {} replay {} adopt {} discard {} | pid {}",
        opts.temporal, opts.replay, opts.adopt, opts.discard_seeds, std::process::id()
    );
    const WARMUP: u32 = 20;
    let mut total_ms = 0.0f64;
    let mut peak_ms = 0.0f64;
    let mut replay_frames = 0u64;
    let mut window = Instant::now();
    for idx in 0..frames {
        let cam = if moving { spin_path_pose(&cam0, scene.diag, idx) } else { cam0 };
        let basis = cam.basis(rw, rh);
        let can_replay =
            opts.temporal && opts.replay && replay_key.as_ref().is_some_and(|b| *b == basis);
        let record = opts.temporal && opts.replay && !can_replay;
        if record {
            replay_cache.begin(rw, rh);
        }
        let temporal_on = opts.temporal && !can_replay;
        let (tcache_cur, tprev_vec, cut_cur, cut_prev) =
            tr.begin(temporal_on, opts.adopt, rw, rh);
        let ctx = FrameCtx {
            scene,
            bvh,
            cam: basis,
            q,
            frame: idx,
            jitter: false,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur,
            tcache_prev: &tprev_vec,
            accumulate: false,
            gbuf: None,
            fsr_buf: None,
            prev_cam: None,
            frame_jitter: Some(dlss::jitter_for(idx)),
            adaptive: false,
            hemi_share: false,
            replay_rec: if record { Some(&replay_cache) } else { None },
            cut_cur,
            cut_prev,
            discard_seeds: opts.discard_seeds,
            defer_shade: opts.defer_shade,
        };
        let t = Instant::now();
        if can_replay {
            render::render_frame_replay(&ctx, &replay_cache);
            replay_frames += 1;
        } else {
            render::render_frame(&ctx, true);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        prof::frame_mark();
        if !can_replay {
            tr.end(temporal_on, opts.adopt, basis);
            replay_key = if record && replay_cache.valid() { Some(basis) } else { None };
        }
        if idx >= WARMUP {
            total_ms += ms;
            peak_ms = peak_ms.max(ms);
        }
        if (idx + 1) % 200 == 0 {
            eprintln!(
                "spin {mode} [{:4}]: {:6.2} ms/frame over the last 200 | {}",
                idx + 1,
                window.elapsed().as_secs_f64() * 1000.0 / 200.0,
                stats.summary_line()
            );
            stats.clear();
            window = Instant::now();
        }
    }
    let timed = frames.saturating_sub(WARMUP).max(1) as f64;
    eprintln!(
        "spin {mode} summary: {:.2} ms/frame mean (peak {:.2}) over {} timed frames | replay frames {replay_frames}",
        total_ms / timed,
        peak_ms,
        timed as u64,
    );
    0
}

/// Headless end-to-end check: correctness counters (must be 0), an A/B
/// benchmark of hybrid vs plain, and a rendered check.png. `structural`
/// additionally gates the scene-topology assertions (coarse pixels at fixed
/// caps, temporal seeds/sky-tiles firing) — they are tuned to the default
/// procedural scene; a `--stress` scene keeps only the scene-agnostic
/// zero-counter invariants.
fn run_check(scene: &scene::Scene, bvh: &bvh::Bvh, cam0: Camera, structural: bool) -> i32 {
    let (rw, rh) = (800usize, 600usize);
    let cam = cam0.basis(rw, rh);
    let q = Quality::preset(2);
    let stats = Stats::default();

    // Mip-chain / trilinear / aniso gates: chain shape, linear-space
    // filtering, sRGB roundtrip, level/lerp mechanics, the anisotropic tap
    // mechanics, and the lod ≤ 0 bit-compat contract (magnified views
    // identical to the pre-mip renderer).
    let tex_ok = texture::self_test();

    // SH sky irradiance — basis orthonormality (pins every constant), the
    // uniform-sky convention pin (radiance L in, exactly L out — what makes it
    // a drop-in for the old AMBIENT · ao), accuracy vs a brute-force
    // cosine-weighted reference, and projection determinism.
    let sh_ok = match sh::self_test() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("sh self-test: FAIL — {e}");
            false
        }
    };

    // Glare: normalized octave weights, ENERGY CONSERVATION (a uniform image
    // must come back unchanged — glare redistributes light, never creates it),
    // total-energy preservation on a point source, and a monotone heavy tail.
    let bloom_ok = match bloom::self_test() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("bloom self-test: FAIL — {e}");
            false
        }
    };

    // The one sky: the disc's radiance/irradiance round-trip (the classic 4π
    // slip), cone sampling inside-and-covering the disc, the disc test agreeing
    // with the cone the sampler draws from, the DOME carrying no disc (the
    // invariant the hemi fixed-point accumulator depends on), and the resulting
    // ambient landing in a physically sane, blue-dominant band.
    let sky_ok = match sky::self_test() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("sky self-test: FAIL — {e}");
            false
        }
    };

    // Spherical-cell math self-test — closed-form identities the hemisphere
    // bounce integrator is built on (Ω/PSA anchors, exact partition,
    // in-cell sampling).
    let sph_ok = match sphcell::self_test() {
        Ok(()) => {
            eprintln!("sphcell self-test: OK");
            true
        }
        Err(e) => {
            eprintln!("sphcell self-test: FAIL — {e}");
            false
        }
    };

    // Wide frustum tree: structural audit (every slot box == its binary node's
    // box) + bound-equivalence sweep vs the binary query + cut translation.
    // Runs on every --check regardless of --ftree — it builds its own tree —
    // so the structure can't rot while the lever is off.
    let ftree_ok = match ftree::self_test(scene, bvh) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("ftree self-test: FAIL — {e}");
            false
        }
    };

    // Reprojection-history math self-test — closed-form gates the OIDN
    // temporal mode is built on (projection roundtrip, static replay = exact
    // running mean, analytic strafe, behind-plane/depth rejection, coarse
    // keep/reset, L-accounting).
    let reproj_ok = match reproject::self_test() {
        Ok(()) => {
            eprintln!("reproject self-test: OK");
            true
        }
        Err(e) => {
            eprintln!("reproject self-test: FAIL — {e}");
            false
        }
    };

    // NPPD staging math self-test — closed-form gates the neural-denoiser
    // path is built on (pad table, NCHW pack/crop bit-identity, warp
    // identity/shift/midpoint/zeros-outside, MV-sign convention).
    let nppd_ok = match nppd::self_test() {
        Ok(()) => {
            eprintln!("nppd self-test: OK");
            true
        }
        Err(e) => {
            eprintln!("nppd self-test: FAIL — {e}");
            false
        }
    };

    // Material-classifier self-test — deterministic spot checks over the
    // real San Miguel naming patterns (keyword precedence, whole-token
    // safety, the name/Ns/illum fallback tiers).
    let matclass_ok = match matclass::self_test() {
        Ok(()) => {
            eprintln!("matclass self-test: OK");
            true
        }
        Err(e) => {
            eprintln!("matclass self-test: FAIL — {e}");
            false
        }
    };

    // Tangent-frame / normal-map decode self-test — analytic directions,
    // green-channel sign pin, mirrored-UV handedness, degenerate skip.
    let tangent_ok = match shade::tangent_self_test() {
        Ok(()) => {
            eprintln!("tangent self-test: OK");
            true
        }
        Err(e) => {
            eprintln!("tangent self-test: FAIL — {e}");
            false
        }
    };

    // Upscaler-chain self-test — the DLSS→FSR4→XeSS→FSR3 resolution order and
    // the force/skip flag algebra, with availability injected (DLL-free).
    let upchain_ok = match upchain::self_test() {
        Ok(()) => {
            eprintln!("upchain self-test: OK");
            true
        }
        Err(e) => {
            eprintln!("upchain self-test: FAIL — {e}");
            false
        }
    };

    // glTF loader self-test — an in-code GLB exercises node flattening, the
    // mirrored-winding flip, u16 index widening, and the factor mapping.
    let gltf_ok = match gltf_loader::self_test() {
        Ok(()) => {
            eprintln!("gltf self-test: OK");
            true
        }
        Err(e) => {
            eprintln!("gltf self-test: FAIL — {e}");
            false
        }
    };

    // BC7 self-test — the STRUCTURAL half of --bc7 (block counts, the
    // mandatory edge-replicate pad, the cutout carve-out predicate, encode
    // determinism). Fidelity needs a decoder we don't have on the CPU, so it
    // is measured on the GPU by --check-gpu's M11 instead.
    let bc7_ok = match bc7::self_test() {
        Ok(()) => {
            eprintln!("bc7 self-test: OK");
            true
        }
        Err(e) => {
            eprintln!("bc7 self-test: FAIL — {e}");
            false
        }
    };

    let rep = render::verify(scene, bvh, &cam, q, rw, rh, &stats, None, &[], None);
    eprintln!(
        "verify full-depth ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e}",
        rep.pixels, rep.false_sky, rep.overshoot, rep.hybrid_extra, rep.max_rel_err
    );
    // The wide-TILE wiring (tile_step/adopt_step dispatch, wide root_cut
    // seeding, the once-per-leaf-tile slot-ref -> binary ray-root translation)
    // defaults OFF — measured wall-neutral on San Miguel and ~10% SLOWER on
    // the stress field's fat-cut/short-descent regime (the GPU-hemi lesson on
    // CPU tiles; --ftree-tiles re-enables, the quantized-box work re-measures).
    // So --check forces it ON for one full verify pass: the reference re-trace
    // gates false-sky/tmin-overshoot through the whole translated path, which
    // would otherwise rot while the lever is off. Restored so every later
    // gate and bench row measures the session default.
    let tiles_session = ftree::FTREE_TILES.load(Relaxed);
    ftree::FTREE_TILES.store(true, Relaxed);
    let rep_wt = render::verify(scene, bvh, &cam, q, rw, rh, &stats, None, &[], None);
    ftree::FTREE_TILES.store(tiles_session, Relaxed);
    eprintln!(
        "verify wide-tiles ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e}",
        rep_wt.pixels, rep_wt.false_sky, rep_wt.overshoot, rep_wt.hybrid_extra, rep_wt.max_rel_err
    );
    // The capped driver is the uncapped one with an extra depth check (a cap
    // past the leaf depth is bit-identical by construction), so verify it at a
    // cap that actually sparse-fills: every non-coarse pixel — including the
    // per-cell point samples, which are KIND_LEAF and thus inside the gates —
    // must still match the reference exactly, and both coarse pixels and
    // samples must exist (deterministic — no wall clock involved).
    let smp0 = stats.coarse_samples.load(Relaxed);
    let rep_c = render::verify(scene, bvh, &cam, q, rw, rh, &stats, Some(4), &[], None);
    let smp_c = stats.coarse_samples.load(Relaxed) - smp0;
    eprintln!(
        "verify capped d=4 ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e} | coarse px {} samples {}",
        rep_c.pixels, rep_c.false_sky, rep_c.overshoot, rep_c.hybrid_extra, rep_c.max_rel_err, rep_c.coarse, smp_c
    );
    let mut capped_ok = rep_c.ok() && (!structural || rep_c.coarse > 0);
    if structural && rep_c.coarse == 0 {
        eprintln!("verify capped d=4: expected coarse pixels, found none — capped path not exercised");
    }
    // Coarse tiles exist above, so their per-cell samples must too — this is
    // sound on any scene, not just the default topology (no structural gate).
    if rep_c.coarse > 0 && smp_c == 0 {
        eprintln!("verify capped d=4: coarse pixels without point samples — sparse fill didn't fire");
        capped_ok = false;
    }

    // Hemisphere frustum AO: soundness gates (reference rays re-validate
    // every empty-cell claim and leaf-ray tmin on a deterministic probe set —
    // the false-sky / tmin-overshoot analogs) plus an A/B error measurement
    // against high-sample cosine AO at the same surface points. The
    // integrator is unbiased, so the signed mean is a bias detector.
    // The hemi probe gates exist to validate the CUT MACHINERY — cut-miss
    // re-traces every cut-seeded leaf ray against a root-traversal reference,
    // which is vacuous if the leaf rays themselves go root-first. So the probe
    // gates force cut-seeded rays regardless of the session default (root-first
    // since the M2 re-measure: 3-10% faster in every fb mode, both trees).
    // Restored before the bench table, which must measure the defaults.
    let cut_hemi_session = bvh::CUT_SEED_HEMI.load(Relaxed);
    bvh::CUT_SEED_HEMI.store(true, Relaxed);
    let probes = collect_probes(scene, bvh, &cam, rw, rh);
    let hemi_ok = {
        const SEEDS: u64 = 8; // hemi AO averaged over stratified frames
        const REF_SAMPLES: u32 = 1024;
        // Per-probe work is independent (per-pixel seeds); results are
        // collected in probe order and folded sequentially, so the sweep is
        // deterministic and parallel.
        let results: Vec<_> = probes
            .par_iter()
            .map(|pr| {
                let mut hv = hemi::VerifyCounters::default();
                let mut ls = stats::LocalStats::default();
                let mut vis = 0u64;
                let (t1, t2) = shade::onb(pr.n);
                let mut ao_h = 0.0;
                for s in 0..SEEDS {
                    let mut rng = fastrand::Rng::with_seed(px_seed(pr.x, pr.y, s));
                    ao_h += hemi::ao(
                        scene,
                        ftree::Accel::of(bvh),
                        pr.p,
                        pr.n,
                        t1,
                        t2,
                        q.fb.depth,
                        scene.ao_radius,
                        None,
                        &mut rng,
                        if s == 0 { Some(&mut hv) } else { None },
                        &mut ls,
                    );
                }
                let ao_h = ao_h / SEEDS as f32;
                // Reference: cosine-sampled AO, the same construction shade()
                // uses, from the same eps-offset point.
                let mut rng = fastrand::Rng::with_seed(px_seed(pr.x, pr.y, 0xA0));
                let mut open = 0u32;
                for _ in 0..REF_SAMPLES {
                    let r1 = rng.f32();
                    let r2 = rng.f32();
                    let d = shade::cosine_dir(pr.n, t1, t2, r1, r2);
                    if !bvh.occluded(scene, &bvh::Ray::new(pr.p, d), 0.0, scene.ao_radius, &mut vis)
                    {
                        open += 1;
                    }
                }
                let ao_r = open as f32 / REF_SAMPLES as f32;
                ((ao_h - ao_r) as f64, hv, ls)
            })
            .collect();
        let mut hv = hemi::VerifyCounters::default();
        let mut ls = stats::LocalStats::default();
        let (mut sum_abs, mut sum_signed, mut worst) = (0f64, 0f64, 0f32);
        for (d, phv, pls) in &results {
            hv.merge(phv);
            ls.merge(pls);
            sum_abs += d.abs();
            sum_signed += *d;
            worst = worst.max(d.abs() as f32);
        }
        let nprobes = results.len() as u64;
        let mean_abs = sum_abs / nprobes.max(1) as f64;
        let mean_signed = sum_signed / nprobes.max(1) as f64;
        eprintln!(
            "hemi AO ({nprobes} probes, depth {}): psa-viol {} | false-empty {} | tmin-overshoot {} | cut-miss {} | max psa err {:.2e}",
            q.fb.depth, hv.psa_violations, hv.false_empty, hv.tmin_overshoot, hv.cut_miss, hv.max_psa_err
        );
        eprintln!(
            "hemi AO vs {REF_SAMPLES}-sample cosine: mean |Δ| {mean_abs:.4} (limit 0.02) | mean Δ {mean_signed:+.4} (limit ±0.005) | worst {worst:.3} | per point: {:.1} queries, {:.1} rays, {:.1} cells empty",
            ls.hemi_queries as f64 / ls.hemi_points.max(1) as f64,
            ls.hemi_leaf_rays as f64 / ls.hemi_points.max(1) as f64,
            ls.hemi_cells_empty as f64 / ls.hemi_points.max(1) as f64,
        );
        let mut ok = hv.ok() && mean_abs < 0.02 && mean_signed.abs() < 0.005;
        if structural && (ls.hemi_cells_empty == 0 || ls.hemi_leaf_rays == 0) {
            eprintln!("hemi AO: expected both empty cells and leaf rays > 0 — a path didn't fire");
            ok = false;
        }
        ok
    };

    // Hemisphere frustum GI: the same soundness gates (with t_limit = ∞, so
    // empty-cell claims are true sky claims) plus an A/B against a
    // cosine-sampled reference implementing the SAME depth-1 bounce policy
    // (hemi::BOUNCE_Q) — the comparison isolates integrator error from policy
    // error. Luminance-relative gate; the signed mean detects bias.
    let gi_ok = {
        let sun = render::sun_dir(scene);
        const SEEDS: u64 = 8;
        const REF_SAMPLES: u32 = 512;
        let lum = |c: Vec3A| c.dot(Vec3A::new(0.2126, 0.7152, 0.0722));
        // Same probe set and parallel/sequential-fold structure as hemi AO.
        let results: Vec<_> = probes
            .par_iter()
            .map(|pr| {
                let mut hv = hemi::VerifyCounters::default();
                let mut ls = stats::LocalStats::default();
                let mut vis = 0u64;
                let (t1, t2) = shade::onb(pr.n);
                let mut e_h = Vec3A::ZERO;
                for s in 0..SEEDS {
                    let mut rng = fastrand::Rng::with_seed(px_seed(pr.x, pr.y, s));
                    e_h += hemi::gi(
                        scene,
                        ftree::Accel::of(bvh),
                        pr.p,
                        pr.n,
                        t1,
                        t2,
                        q.fb.depth,
                        sun,
                        0,
                        None,
                        &mut rng,
                        if s == 0 { Some(&mut hv) } else { None },
                        &mut ls,
                    );
                }
                let e_h = e_h / SEEDS as f32;
                // Reference: cosine-sampled hemisphere, identical depth-1
                // policy (miss → sky, hit → shade at BOUNCE_Q, depth 1).
                let mut rng = fastrand::Rng::with_seed(px_seed(pr.x, pr.y, 0xE0));
                let mut lsr = stats::LocalStats::default();
                let mut e_r = Vec3A::ZERO;
                for _ in 0..REF_SAMPLES {
                    let r1 = rng.f32();
                    let r2 = rng.f32();
                    let d = shade::cosine_dir(pr.n, t1, t2, r1, r2);
                    let bray = bvh::Ray::new(pr.p, d);
                    e_r += match bvh.intersect(scene, &bray, 0.0, f32::INFINITY, &mut vis) {
                        // dome, NOT radiance — a GATHER path, mirroring
                        // hemi.rs's leaf-ray miss exactly (see sky.rs).
                        None => crate::sky::dome(d, sun),
                        Some(h) => shade::shade(
                            scene,
                            bvh,
                            &bray,
                            &h,
                            None,
                            &hemi::BOUNCE_Q,
                            &mut rng,
                            sun,
                            shade::Cone::bounce(),
                            1,
                            &mut lsr,
                            None,
                            shade::VisCtl::Off,
                            None,
                        ),
                    };
                }
                let e_r = e_r / REF_SAMPLES as f32;
                let rel = ((lum(e_h) - lum(e_r)) / lum(e_r).max(0.05)) as f64;
                (rel, hv, ls)
            })
            .collect();
        let mut hv = hemi::VerifyCounters::default();
        let mut ls = stats::LocalStats::default();
        let mut worst = 0f32;
        for (rel, phv, pls) in &results {
            hv.merge(phv);
            ls.merge(pls);
            worst = worst.max(rel.abs() as f32);
        }
        // Firefly-robust statistic — the documented unpaired-A/B fix (the
        // CLAUDE.md canopy-caveat lesson): relative error is bounded −1
        // below but UNBOUNDED above (a hemi leaf ray that catches a bright
        // emitter or sun-glow lobe the 512-sample reference undersamples is
        // a +NX outlier — DamagedHelmet's emissive visor trips it on real
        // content), so the gate trims 2% from each tail before the means.
        // The exact-zero soundness gates above stay untouched — this only
        // robustifies the estimator comparison; `worst` still prints raw.
        let mut rels: Vec<f64> = results.iter().map(|(rel, _, _)| *rel).collect();
        // total_cmp: a NaN rel (NaN radiance) must FAIL the gate (it sorts
        // past +inf into the kept slice and poisons the mean), not panic the
        // sort's partial_cmp unwrap.
        rels.sort_unstable_by(|a, b| a.total_cmp(b));
        let trim = rels.len() / 50;
        let kept = &rels[trim..rels.len() - trim];
        let mean_rel =
            kept.iter().map(|r| r.abs()).sum::<f64>() / kept.len().max(1) as f64;
        let mean_signed = kept.iter().sum::<f64>() / kept.len().max(1) as f64;
        let nprobes = results.len() as u64;
        eprintln!(
            "hemi GI ({nprobes} probes, depth {}): psa-viol {} | false-empty {} | tmin-overshoot {} | cut-miss {} | max psa err {:.2e}",
            q.fb.depth, hv.psa_violations, hv.false_empty, hv.tmin_overshoot, hv.cut_miss, hv.max_psa_err
        );
        eprintln!(
            "hemi GI vs {REF_SAMPLES}-sample cosine (same depth-1 policy, 2% trimmed): mean rel {mean_rel:.4} (limit 0.05) | signed {mean_signed:+.4} (limit ±0.01) | worst raw {worst:.3} | per point: {:.1} queries, {:.1} rays, {:.1} cells empty",
            ls.hemi_queries as f64 / ls.hemi_points.max(1) as f64,
            ls.hemi_leaf_rays as f64 / ls.hemi_points.max(1) as f64,
            ls.hemi_cells_empty as f64 / ls.hemi_points.max(1) as f64,
        );
        let mut ok = hv.ok() && mean_rel < 0.05 && mean_signed.abs() < 0.01;
        if structural && (ls.hemi_cells_empty == 0 || ls.hemi_leaf_rays == 0) {
            eprintln!("hemi GI: expected both empty cells and leaf rays > 0 — a path didn't fire");
            ok = false;
        }
        ok
    };

    // Hemi sharing: one padded tree capture per coherent 2×2 group, consumed
    // by every member from its own apex (see hemi::share_capture). Gates: the
    // transplanted exact-zero soundness counters run PER MEMBER — a member
    // re-validates the rep's Open claim, every leaf-ray tmin, and every cut
    // traversal at ITS OWN origin (false-empty / tmin-overshoot / cut-miss /
    // PSA) — plus shared-vs-unshared A/Bs on independent seed sets: both arms
    // estimate the same integral, so a nonzero mean difference is exactly the
    // bias a sharing soundness hole would introduce.
    let mut hemi_share_ok = {
        const SEEDS: u64 = 8;
        let sun = render::sun_dir(scene);
        let lum = |c: Vec3A| 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
        // 2×2 groups anchored at each probe pixel, kept under the renderer's
        // exact predicate: same triangle, bit-equal shading normal, measured
        // η/δ qualifiers (hemi::SHARE_*). Center rays, like collect_probes.
        struct GProbe {
            x: usize,
            y: usize,
            pts: [(Vec3A, Vec3A); 4],
            delta: f32,
            eta: f32,
        }
        let mut groups: Vec<GProbe> = Vec::new();
        // Probes that pass same-tri + bit-equal-normal but fail the η/δ
        // qualifiers — must-fired below (default scene: 61, from grazing-angle
        // ground-plane probes) so the qualifier branch can't rot into dead
        // code; skipped under --stress like every topology-tuned assertion.
        let mut qual_reject = 0u64;
        {
            let mut vis = 0u64;
            for pr in &probes {
                if pr.x + 1 >= rw || pr.y + 1 >= rh {
                    continue;
                }
                let mut pts = [(Vec3A::ZERO, Vec3A::ZERO); 4];
                let mut tri = [u32::MAX; 4];
                let mut all_hit = true;
                for (i, (dx, dy)) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)].into_iter().enumerate()
                {
                    let dir = cam.ray_dir((pr.x + dx) as f32 + 0.5, (pr.y + dy) as f32 + 0.5);
                    let ray = bvh::Ray::new(cam.origin, dir);
                    match bvh.intersect(scene, &ray, 0.0, f32::INFINITY, &mut vis) {
                        Some(h) => {
                            pts[i] = shade::surface_point(scene, &ray, &h);
                            tri[i] = h.tri;
                        }
                        None => {
                            all_hit = false;
                            break;
                        }
                    }
                }
                if !all_hit
                    || tri.iter().any(|&t| t != tri[0])
                    || pts.iter().any(|&(_, n)| n != pts[0].1)
                {
                    continue;
                }
                let (rp, rn) = pts[0];
                let (mut delta, mut eta) = (0.0f32, 0.0f32);
                for &(p, _) in &pts[1..] {
                    let d = p - rp;
                    delta = delta.max(d.length());
                    eta = eta.max(rn.dot(d).abs());
                }
                if eta > scene.eps * hemi::SHARE_ETA_FRAC
                    || delta > scene.ao_radius * hemi::SHARE_DELTA_FRAC
                {
                    qual_reject += 1;
                    continue;
                }
                groups.push(GProbe { x: pr.x, y: pr.y, pts, delta, eta });
            }
        }
        let results: Vec<_> = groups
            .par_iter()
            .map(|g| {
                let mut hv = hemi::VerifyCounters::default();
                let mut ls = stats::LocalStats::default();
                let (rp, rn) = g.pts[0];
                let (t1r, t2r) = shade::onb(rn);
                // record_empties: Apply's verify arm re-validates every
                // folded empty claim from EACH member's apex.
                let mut ao_share = hemi::HemiShare::new();
                ao_share.record_empties = true;
                hemi::share_capture(
                    scene,
                    ftree::Accel::of(bvh),
                    rp,
                    rn,
                    t1r,
                    t2r,
                    q.fb.depth,
                    scene.ao_radius,
                    g.delta,
                    g.eta,
                    None,
                    &mut ao_share,
                    &mut ls,
                );
                let mut gi_share = hemi::HemiShare::new();
                gi_share.record_empties = true;
                hemi::share_capture(
                    scene,
                    ftree::Accel::of(bvh),
                    rp,
                    rn,
                    t1r,
                    t2r,
                    q.fb.depth,
                    f32::INFINITY,
                    g.delta,
                    g.eta,
                    Some(sun),
                    &mut gi_share,
                    &mut ls,
                );
                // Poison at a preset depth is a record-capacity bug — report
                // it through the gate (a poisoned record must never be
                // consumed, so this group's A/B is skipped).
                if ao_share.poisoned || gi_share.poisoned {
                    return (0.0, 0.0, hv, ls, true);
                }
                let (mut d_ao, mut d_gi) = (0f64, 0f64);
                for (i, &(p, n)) in g.pts.iter().enumerate() {
                    let (t1, t2) = shade::onb(n);
                    let (x, y) = (g.x + (i & 1), g.y + (i >> 1));
                    // PAIRED same-seed arms: both draw the identical rng
                    // stream, so every ray the two trees have in common
                    // cancels exactly in the difference — GI's heavy-tailed
                    // fireflies included (an unpaired construction was tried
                    // and measured the baseline estimator's skew, not the
                    // sharing: identical +2.4% with sharing disabled in both
                    // arms). The residual is purely the sharing-induced
                    // delta, and an unbiased sharing keeps its mean at 0.
                    let (mut aos, mut aou) = (0.0f32, 0.0f32);
                    let (mut gis, mut giu) = (Vec3A::ZERO, Vec3A::ZERO);
                    for s in 0..SEEDS {
                        let mut rng = fastrand::Rng::with_seed(px_seed(x, y, s));
                        aos += hemi::ao(
                            scene,
                            ftree::Accel::of(bvh),
                            p,
                            n,
                            t1,
                            t2,
                            q.fb.depth,
                            scene.ao_radius,
                            Some(&ao_share),
                            &mut rng,
                            if s == 0 { Some(&mut hv) } else { None },
                            &mut ls,
                        );
                        let mut rng = fastrand::Rng::with_seed(px_seed(x, y, s));
                        aou += hemi::ao(
                            scene,
                            ftree::Accel::of(bvh),
                            p,
                            n,
                            t1,
                            t2,
                            q.fb.depth,
                            scene.ao_radius,
                            None,
                            &mut rng,
                            None,
                            &mut ls,
                        );
                        let mut rng = fastrand::Rng::with_seed(px_seed(x, y, 0x80 + s));
                        gis += hemi::gi(
                            scene,
                            ftree::Accel::of(bvh),
                            p,
                            n,
                            t1,
                            t2,
                            q.fb.depth,
                            sun,
                            0,
                            Some(&gi_share),
                            &mut rng,
                            if s == 0 { Some(&mut hv) } else { None },
                            &mut ls,
                        );
                        let mut rng = fastrand::Rng::with_seed(px_seed(x, y, 0x80 + s));
                        giu += hemi::gi(
                            scene,
                            ftree::Accel::of(bvh),
                            p,
                            n,
                            t1,
                            t2,
                            q.fb.depth,
                            sun,
                            0,
                            None,
                            &mut rng,
                            None,
                            &mut ls,
                        );
                    }
                    d_ao += ((aos - aou) / SEEDS as f32) as f64;
                    let (l_s, l_u) = (lum(gis / SEEDS as f32), lum(giu / SEEDS as f32));
                    d_gi += ((l_s - l_u) / l_u.max(0.05)) as f64;
                }
                (d_ao / 4.0, d_gi / 4.0, hv, ls, false)
            })
            .collect();
        let mut hv = hemi::VerifyCounters::default();
        let mut ls = stats::LocalStats::default();
        let (mut sum_ao, mut sum_gi, mut worst) = (0f64, 0f64, 0f64);
        let mut n_poisoned = 0u64;
        for (da, dg, phv, pls, poisoned) in &results {
            hv.merge(phv);
            ls.merge(pls);
            sum_ao += da;
            sum_gi += dg;
            worst = worst.max(da.abs()).max(dg.abs());
            n_poisoned += *poisoned as u64;
        }
        let ng = results.len() as f64;
        let (mean_ao, mean_gi) = (sum_ao / ng.max(1.0), sum_gi / ng.max(1.0));
        eprintln!(
            "hemi share ({} groups of 4, {} qual-rejected): psa-viol {} | false-empty {} | tmin-overshoot {} | cut-miss {} | max psa err {:.2e}",
            results.len(), qual_reject, hv.psa_violations, hv.false_empty, hv.tmin_overshoot, hv.cut_miss, hv.max_psa_err
        );
        eprintln!(
            "hemi share paired A/B vs unshared (same seeds, per member): AO Δ {mean_ao:+.4} (limit ±0.005) | GI rel Δ {mean_gi:+.4} (limit ±0.01) | worst {worst:.3} | shared pts {} | share q/pt {:.2}",
            ls.hemi_share_points,
            ls.hemi_queries as f64 / ls.hemi_points.max(1) as f64,
        );
        let mut ok = hv.ok() && mean_ao.abs() < 0.005 && mean_gi.abs() < 0.01;
        if n_poisoned > 0 {
            eprintln!("hemi share: {n_poisoned} captures poisoned at preset depth (record capacity bug)");
            ok = false;
        }
        if structural && (results.is_empty() || ls.hemi_share_points == 0) {
            eprintln!("hemi share: expected qualifying probe groups and shared points > 0 — the path didn't fire");
            ok = false;
        }
        if structural && qual_reject == 0 {
            eprintln!("hemi share: expected η/δ qualifier rejections > 0 — the qualifier branch didn't fire");
            ok = false;
        }
        // Guard must-fire: a capture past FB_DEPTH_CAP must poison (and so
        // fall back per-pixel). At any legal depth the capacity math makes
        // record overflow unreachable (≤ 4^(FB_DEPTH_CAP−1) leaves, ≤ 21 cut
        // slots), so the depth guard is the one live poison producer — this
        // keeps it from rotting into dead code.
        if let Some(g) = groups.first() {
            let (rp, rn) = g.pts[0];
            let (t1, t2) = shade::onb(rn);
            let mut deep = hemi::HemiShare::new();
            let mut dls = stats::LocalStats::default();
            hemi::share_capture(
                scene,
                ftree::Accel::of(bvh),
                rp,
                rn,
                t1,
                t2,
                hemi::FB_DEPTH_CAP + 1,
                scene.ao_radius,
                g.delta,
                g.eta,
                None,
                &mut deep,
                &mut dls,
            );
            if !deep.poisoned {
                eprintln!(
                    "hemi share: a depth-{} capture did not poison — the FB_DEPTH_CAP guard is dead",
                    hemi::FB_DEPTH_CAP + 1
                );
                ok = false;
            }
        }
        ok
    };

    // Probe gates done — the bench rows below measure the session defaults.
    bvh::CUT_SEED_HEMI.store(cut_hemi_session, Relaxed);

    let accum: Vec<AtomicU32> = (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..rw * rh).map(|_| AtomicU32::new(0)).collect();

    const BENCH_FRAMES: u32 = 8;
    // (hemi_queries, share groups, share fallback) per row, for the
    // hemi-share must-fires below (share-on must run strictly fewer queries).
    let mut share_rows: Vec<(&str, bool, u64, u64, u64)> = Vec::new();
    for (label, hybrid, hemi_ao, hemi_gi, share) in [
        ("hybrid ", true, false, false, false),
        ("hemi-ao (share off)", true, true, false, false),
        ("hemi-ao (share on) ", true, true, false, true),
        ("hemi-gi (share off)", true, false, true, false),
        ("hemi-gi (share on) ", true, false, true, true),
        ("plain  ", false, false, false, false),
    ] {
        stats.clear();
        let mut bq = q;
        bq.fb.ao = hemi_ao;
        bq.fb.gi = hemi_gi;
        let ctx = FrameCtx {
            scene,
            bvh,
            cam,
            q: bq,
            frame: 0,
            jitter: false,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: &[],
            accumulate: true,
            gbuf: None,
            fsr_buf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
            hemi_share: share,
            replay_rec: None,
            cut_cur: None,
            cut_prev: None,
            discard_seeds: false,
            defer_shade: false,
        };
        let t = Instant::now();
        for _ in 0..BENCH_FRAMES {
            render::render_frame(&ctx, hybrid);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / BENCH_FRAMES as f64;
        eprintln!("{label}: {ms:6.1} ms/frame | per {BENCH_FRAMES} frames: {}", stats.summary_line());
        if hemi_ao || hemi_gi {
            share_rows.push((
                label,
                share,
                stats.hemi_queries.load(Relaxed),
                stats.hemi_share_groups.load(Relaxed),
                stats.hemi_share_fallback.load(Relaxed),
            ));
        }
        // Save the plain-hybrid and hemi-GI images while their buffers are
        // fresh (frame stays 0, so accum holds exactly the last frame).
        if hybrid && !hemi_ao && !hemi_gi {
            let mut present = vec![0u32; rw * rh];
            render::resolve(&accum, &info, 1, false, &mut present, rw, rh, rw, rh);
            save_png("check.png", &present, rw, rh);
        } else if hemi_gi && !share {
            let mut present = vec![0u32; rw * rh];
            render::resolve(&accum, &info, 1, false, &mut present, rw, rh, rw, rh);
            save_png("check_gi.png", &present, rw, rh);
        }
    }
    eprintln!("wrote check.png + check_gi.png");
    // Hemi-share frame must-fires (KILL CRITERION rides the printed ms above,
    // the shafts/adopt precedent: if share-on is not measurably faster on
    // both the default scene and --stress 5000, the feature does not merge):
    // share-on frames must form groups, must still fall back somewhere
    // (curved geometry — proof the predicate rejects), and must run strictly
    // fewer hemi queries than their share-off twin.
    for pair in share_rows.chunks(2) {
        let [(l0, s0, q0, _, _), (l1, s1, q1, g1, f1)] = pair else { continue };
        debug_assert!(!s0 && *s1, "share rows must alternate off/on");
        if structural {
            if *g1 == 0 || *f1 == 0 {
                eprintln!("hemi share bench {l1}: expected groups and fallbacks > 0 (groups {g1}, fallback {f1})");
                hemi_share_ok = false;
            }
            if q1 >= q0 {
                eprintln!("hemi share bench: {l1} ran {q1} hemi queries, not fewer than {l0}'s {q0}");
                hemi_share_ok = false;
            }
        }
    }
    // Hemi-share frame gates: (a) determinism — two same-seed share-on fb
    // frames must be bit-identical (group formation and capture are pure
    // functions of the frame inputs); (b) structure-replay coverage — the
    // existing replay bit-identity gates run fb-OFF and prove nothing about
    // the share cell loop, so one fb-ao trace/replay pair is gated here, with
    // groups > 0 on both arms as the anti-vacuity check.
    {
        let snap = || -> (Vec<u32>, Vec<u32>, Vec<u32>) {
            (
                accum.iter().map(|a| a.load(Relaxed)).collect(),
                info.iter().map(|a| a.load(Relaxed)).collect(),
                tbuf.iter().map(|a| a.load(Relaxed)).collect(),
            )
        };
        let mut bq = q;
        bq.fb.ao = true;
        let rcache = replay::ReplayCache::new(rw, rh);
        rcache.begin(rw, rh);
        let ctx_rec = FrameCtx {
            scene,
            bvh,
            cam,
            q: bq,
            frame: 0,
            jitter: false,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: &[],
            accumulate: true,
            gbuf: None,
            fsr_buf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
            hemi_share: true,
            replay_rec: Some(&rcache),
            cut_cur: None,
            cut_prev: None,
            discard_seeds: false,
            defer_shade: false,
        };
        stats.clear();
        render::render_frame(&ctx_rec, true);
        let groups_a = stats.hemi_share_groups.load(Relaxed);
        let a = snap();
        let ctx = FrameCtx { replay_rec: None, ..ctx_rec };
        stats.clear();
        render::render_frame(&ctx, true);
        let b = snap();
        let mut ok = true;
        if a != b {
            eprintln!("hemi share: two same-seed share-on frames differ — nondeterministic grouping");
            ok = false;
        }
        if !rcache.valid() {
            // The contract's lost-replay state (arena overflow): production
            // never replays a poisoned frame. Only the default scene must
            // record validly — a dense OBJ scene overflows legitimately.
            if structural {
                eprintln!("hemi share: fb-ao recording frame poisoned — replay coverage not provable");
                ok = false;
            } else {
                eprintln!("hemi share: fb-ao recording poisoned (arena overflow) — replay gate skipped");
            }
        } else {
            stats.clear();
            render::render_frame_replay(&ctx, &rcache);
            let groups_c = stats.hemi_share_groups.load(Relaxed);
            let c = snap();
            if a != c {
                eprintln!("hemi share: fb-ao replay is not bit-identical to its trace");
                ok = false;
            }
            if structural && (groups_a == 0 || groups_c == 0) {
                eprintln!("hemi share: replay gate vacuous (trace groups {groups_a}, replay groups {groups_c})");
                ok = false;
            }
        }
        eprintln!(
            "hemi share frame gates: determinism {} | fb-ao replay bit-identity {} (groups {groups_a})",
            if a == b { "OK" } else { "FAIL" },
            if ok { "OK" } else { "FAIL" },
        );
        hemi_share_ok &= ok;
    }

    // Smoke-test dynamic resolution: one depth-capped frame per plausible cap
    // must complete; coarse coverage shrinks monotonically-ish as the cap
    // deepens until the leaf depth makes it a normal hybrid frame.
    for cap in [3u32, 5, 7] {
        stats.clear();
        let ctx = FrameCtx {
            scene,
            bvh,
            cam,
            q,
            frame: 0,
            jitter: false,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: &[],
            accumulate: true,
            gbuf: None,
            fsr_buf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
            hemi_share: false,
            replay_rec: None,
            cut_cur: None,
            cut_prev: None,
            discard_seeds: false,
            defer_shade: false,
        };
        let t = Instant::now();
        render::render_frame_capped(&ctx, cap);
        eprintln!(
            "dynamic (depth cap {cap}): {:5.1} ms | coarse tiles {} covering {} px",
            t.elapsed().as_secs_f64() * 1000.0,
            stats.coarse_tiles.load(Relaxed),
            stats.coarse_pixels.load(Relaxed),
        );
    }

    // Temporal cache: warm it with one full-depth frame at camera A, then
    // verify frames that consume it. Every pixel still gets the tmin=0
    // reference-ray treatment, so a bad seed surfaces as false-sky/overshoot.
    // All deterministic — no wall clock.
    let tcache = temporal::TemporalCache::new(rw, rh);
    let cuts_a = temporal::CutStore::new(rw, rh);
    {
        let ctx = FrameCtx {
            scene,
            bvh,
            cam,
            q,
            frame: 0,
            jitter: false,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: Some(&tcache),
            tcache_prev: &[],
            // Produced in lockstep with the claim cache: the T passes below
            // consume it with adoption on, so every adopted cut faces the
            // per-pixel reference re-trace.
            accumulate: true,
            gbuf: None,
            fsr_buf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
            hemi_share: false,
            replay_rec: None,
            cut_cur: Some(&cuts_a),
            cut_prev: None,
            discard_seeds: false,
            defer_shade: false,
        };
        render::render_frame(&ctx, true);
    }
    let mut temporal_ok = true;
    let mut temporal_pass = |label: &str, basis: &camera::CamBasis, max_depth: Option<u32>, want_seeds: bool, want_sky: bool, want_adopts: bool| {
        stats.clear();
        let rep = render::verify(scene, bvh, basis, q, rw, rh, &stats, max_depth, &[(&tcache, cam)], Some(&cuts_a));
        let seeds = stats.temporal_seeds.load(Relaxed);
        let sky = stats.temporal_sky_tiles.load(Relaxed);
        let tests = stats.temporal_tests.load(Relaxed);
        let adopts = stats.temporal_cut_adopts.load(Relaxed);
        eprintln!(
            "verify temporal {label} ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e} | seeds {seeds} sky-tiles {sky} cells {tests} adopts {adopts} coarse px {}",
            rep.pixels, rep.false_sky, rep.overshoot, rep.hybrid_extra, rep.max_rel_err, rep.coarse
        );
        let mut ok = rep.ok();
        if structural && want_seeds && seeds == 0 {
            eprintln!("verify temporal {label}: expected temporal seeds > 0 — the path didn't fire");
            ok = false;
        }
        if structural && want_sky && sky == 0 {
            eprintln!("verify temporal {label}: expected temporal sky-tiles > 0 — the sky path didn't fire");
            ok = false;
        }
        if structural && want_adopts && adopts == 0 {
            eprintln!("verify temporal {label}: expected cut adoptions > 0 — the adoption path didn't fire");
            ok = false;
        }
        if structural && max_depth.is_some() && rep.coarse == 0 {
            eprintln!("verify temporal {label}: expected coarse pixels, found none — capped path not exercised");
            ok = false;
        }
        temporal_ok &= ok;
    };
    // T1: identical basis — the static-accumulation fast path. Every sky tile
    // must come from the cache, at least one node must seed, and the verbatim
    // cut-adoption arm must fire (identical cone ⇒ own old cut valid).
    temporal_pass("static", &cam, None, true, true, true);
    // T2: pure forward dolly. Seeds must fire (the root, at minimum: its
    // extreme dirs are the old corners ± fp on the old screen boundary plus
    // the focus of expansion). Sky is NOT asserted: at this δ the λ_max tilt
    // drags every sky tile's query box toward the FOE, across the sky
    // boundary into finite cells — expected, not a regression.
    let mut cam_b = cam0;
    cam_b.pos += cam0.forward() * (0.02 * scene.diag);
    let basis_b = cam_b.basis(rw, rh);
    // Under motion adoption must also fire (the blocked-regime cut-only
    // adoption: tilt-widened hulls land in ancestor cells).
    temporal_pass("dolly", &basis_b, None, true, false, true);
    // T3: the same dolly through the depth-capped driver (the root's seed is
    // cap-independent).
    temporal_pass("dolly capped d=4", &basis_b, Some(4), true, false, false);
    // T4: translate + rotate — the root leaves the old screen and this
    // scene's finite bound landscape is single-valued (the ground AABB blocks
    // everything at one distance), so only correctness is asserted.
    let mut cam_c = cam_b;
    cam_c.yaw += 0.05;
    temporal_pass("dolly+yaw", &cam_c.basis(rw, rh), None, false, false, false);
    // T5: pure rotation — the region-min query's structural win: δ = 0, the
    // old proven-empty balls are unchanged in world space, and panned-into
    // sky tiles overlap only old sky cells → free. Seeds are NOT asserted:
    // with a single-valued finite landscape, min == inherited everywhere.
    let mut cam_y = cam0;
    cam_y.yaw += 0.05;
    let basis_y = cam_y.basis(rw, rh);
    temporal_pass("yaw", &basis_y, None, false, true, false);

    // Informational A/B: static (the accumulation-frame path) and pure yaw
    // (the rotation path), each cold vs seeded. Not gated — the win is
    // scene-dependent.
    let warm = [(&tcache, cam)];
    for (label, basis, prev) in [
        ("static cold", cam, &warm[..0]),
        ("static warm", cam, &warm[..]),
        ("yaw cold   ", basis_y, &warm[..0]),
        ("yaw warm   ", basis_y, &warm[..]),
    ] {
        stats.clear();
        let ctx = FrameCtx {
            scene,
            bvh,
            cam: basis,
            q,
            frame: 0,
            jitter: false,
            rw,
            rh,
            accum: &accum,
            info: &info,
            tbuf: &tbuf,
            stats: &stats,
            sun: render::sun_dir(scene),
            tcache_cur: None,
            tcache_prev: prev,
            accumulate: true,
            gbuf: None,
            fsr_buf: None,
            prev_cam: None,
            frame_jitter: None,
            adaptive: false,
            hemi_share: false,
            replay_rec: None,
            cut_cur: None,
            cut_prev: None,
            discard_seeds: false,
            defer_shade: false,
        };
        let t = Instant::now();
        render::render_frame(&ctx, true);
        eprintln!(
            "temporal A/B {label}: {:5.1} ms | frustum nodes {} | queries {} | temporal: seeds {} sky {} cells {}",
            t.elapsed().as_secs_f64() * 1000.0,
            stats.frustum_nodes.load(Relaxed),
            stats.frustum_queries.load(Relaxed),
            stats.temporal_seeds.load(Relaxed),
            stats.temporal_sky_tiles.load(Relaxed),
            stats.temporal_tests.load(Relaxed),
        );
    }

    // Structure replay (replay.rs): record one producing frame, then prove a
    // replay is indistinguishable from a fresh trace — tbuf, info, AND accum
    // bit-identical — first on the recording frame itself (frame 0,
    // unjittered) and then on a warm jittered frame 1 through the interactive
    // wiring shape (which also proves the terminal structure is stable under
    // a warm identical-basis re-trace, the property the interactive replay
    // relies on). Plus exact terminal pixel accounting, must-fire counts, and
    // a post-replay dolly verify against the untouched producer cache (the
    // frozen-prev contract). All deterministic.
    let mut replay_ok = true;
    {
        let rcache = replay::ReplayCache::new(rw, rh);
        let tcache_p = temporal::TemporalCache::new(rw, rh);
        let warm_p = [(&tcache_p, cam)];
        let alloc3 = || (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();
        let alloc1 = || (0..rw * rh).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();
        let bits_differ = |a: &[AtomicU32], b: &[AtomicU32]| {
            a.iter().zip(b).filter(|(x, y)| x.load(Relaxed) != y.load(Relaxed)).count()
        };
        let (accum_p, info_p, tbuf_p) = (alloc3(), alloc1(), alloc1());
        rcache.begin(rw, rh);
        {
            let ctx = FrameCtx {
                scene,
                bvh,
                cam,
                q,
                frame: 0,
                jitter: false,
                rw,
                rh,
                accum: &accum_p,
                info: &info_p,
                tbuf: &tbuf_p,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: Some(&tcache_p),
                tcache_prev: &[],
                accumulate: true,
                gbuf: None,
                fsr_buf: None,
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
                hemi_share: false,
                replay_rec: Some(&rcache),
                cut_cur: None,
                cut_prev: None,
                discard_seeds: false,
                defer_shade: false,
            };
            render::render_frame(&ctx, true);
        }
        let (nl, ns) = rcache.counts();
        let (leaf_px, sky_px) = rcache.accounting();
        eprintln!(
            "replay record: valid {} | leaves {nl} ({leaf_px} px) + sky {ns} ({sky_px} px) = {} px (screen {} px)",
            rcache.valid(),
            leaf_px + sky_px,
            (rw * rh) as u64,
        );
        if rcache.valid() && leaf_px + sky_px != (rw * rh) as u64 {
            eprintln!("replay record: terminals don't partition the screen");
            replay_ok = false;
        }
        if !rcache.valid() {
            // A poisoned recording (arena overflow) is the contract's
            // lost-replay state: production clears `replay_key` via
            // `valid()` and traces fresh, so there is nothing to
            // replay-gate. The default scene must record validly; a dense
            // OBJ scene can overflow legitimately.
            if structural {
                eprintln!("replay record: recording poisoned on the default scene");
                replay_ok = false;
            } else {
                eprintln!("replay record: poisoned (arena overflow) — replay gates skipped (production traces fresh)");
            }
        }
        if structural && (nl == 0 || ns == 0) {
            eprintln!("replay record: expected both leaves and sky terminals > 0 — a path didn't fire");
            replay_ok = false;
        }
        // Same-seed replay of frame 0 vs the recording frame itself.
        let (accum_r, info_r, tbuf_r) = (alloc3(), alloc1(), alloc1());
        if rcache.valid() {
            let ctx = FrameCtx {
                scene,
                bvh,
                cam,
                q,
                frame: 0,
                jitter: false,
                rw,
                rh,
                accum: &accum_r,
                info: &info_r,
                tbuf: &tbuf_r,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: None,
                tcache_prev: &[],
                accumulate: true,
                gbuf: None,
                fsr_buf: None,
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
                hemi_share: false,
                replay_rec: None,
                cut_cur: None,
                cut_prev: None,
                discard_seeds: false,
                defer_shade: false,
            };
            render::render_frame_replay(&ctx, &rcache);
            let (dt, di, da) = (
                bits_differ(&tbuf_p, &tbuf_r),
                bits_differ(&info_p, &info_r),
                bits_differ(&accum_p, &accum_r),
            );
            eprintln!("replay bit-identity (frame 0): tbuf {dt} | info {di} | accum {da} px differ");
            if dt + di + da > 0 {
                replay_ok = false;
            }
        }
        // Warm frame 1 (the interactive shape: jittered, consuming the
        // producer cache) traced fresh vs replayed from frame 0's structure.
        // Both accumulate onto zeroed buffers, so accum stays comparable.
        let (accum_f, info_f, tbuf_f) = (alloc3(), alloc1(), alloc1());
        let (accum_r1, info_r1, tbuf_r1) = (alloc3(), alloc1(), alloc1());
        for (bufs, fresh) in [((&accum_f, &info_f, &tbuf_f), true), ((&accum_r1, &info_r1, &tbuf_r1), false)] {
            if !fresh && !rcache.valid() {
                continue;
            }
            let ctx = FrameCtx {
                scene,
                bvh,
                cam,
                q,
                frame: 1,
                jitter: true,
                rw,
                rh,
                accum: bufs.0,
                info: bufs.1,
                tbuf: bufs.2,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: None,
                tcache_prev: if fresh { &warm_p[..] } else { &warm_p[..0] },
                accumulate: true,
                gbuf: None,
                fsr_buf: None,
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
                hemi_share: false,
                replay_rec: None,
                cut_cur: None,
                cut_prev: None,
                discard_seeds: false,
                defer_shade: false,
            };
            if fresh {
                render::render_frame(&ctx, true);
            } else {
                render::render_frame_replay(&ctx, &rcache);
            }
        }
        if rcache.valid() {
            let (dt1, di1, da1) = (
                bits_differ(&tbuf_f, &tbuf_r1),
                bits_differ(&info_f, &info_r1),
                bits_differ(&accum_f, &accum_r1),
            );
            eprintln!("replay bit-identity (warm frame 1): tbuf {dt1} | info {di1} | accum {da1} px differ");
            if dt1 + di1 + da1 > 0 {
                replay_ok = false;
            }
        }
        // Frozen-prev contract: replays wrote nothing, so the producer cache
        // must still seed a moving consumer exactly as if they hadn't run.
        stats.clear();
        let rep_d = render::verify(scene, bvh, &basis_b, q, rw, rh, &stats, None, &[(&tcache_p, cam)], None);
        let seeds_d = stats.temporal_seeds.load(Relaxed);
        eprintln!(
            "replay frozen-prev dolly: false-sky {} | tmin-overshoot {} | hybrid-extra {} | seeds {seeds_d}",
            rep_d.false_sky, rep_d.overshoot, rep_d.hybrid_extra,
        );
        if !rep_d.ok() || (structural && seeds_d == 0) {
            eprintln!("replay frozen-prev dolly: gates failed — replay perturbed the temporal contract");
            replay_ok = false;
        }
        // A/B: what a still frame costs cold, cold while recording (the
        // producer overhead), warm (seeded queries), and replayed (zero
        // queries). Informational — the win is the point.
        for (label, warm, do_replay, do_record) in [
            ("static cold    ", false, false, false),
            ("static cold+rec", false, false, true),
            ("static warm    ", true, false, false),
            ("static replay  ", false, true, false),
        ] {
            if do_replay && !rcache.valid() {
                continue; // nothing to replay — production traces fresh
            }
            stats.clear();
            let ctx = FrameCtx {
                scene,
                bvh,
                cam,
                q,
                frame: 0,
                jitter: false,
                rw,
                rh,
                accum: &accum_p,
                info: &info_p,
                tbuf: &tbuf_p,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: None,
                tcache_prev: if warm { &warm_p[..] } else { &warm_p[..0] },
                accumulate: true,
                gbuf: None,
                fsr_buf: None,
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
                hemi_share: false,
                replay_rec: if do_record { Some(&rcache) } else { None },
                cut_cur: None,
                cut_prev: None,
                discard_seeds: false,
                defer_shade: false,
            };
            let t = Instant::now();
            for _ in 0..BENCH_FRAMES {
                if do_replay {
                    render::render_frame_replay(&ctx, &rcache);
                } else {
                    if do_record {
                        rcache.begin(rw, rh); // per-frame reset is part of the cost
                    }
                    render::render_frame(&ctx, true);
                }
            }
            eprintln!(
                "replay A/B {label}: {:5.1} ms | frustum queries {} nodes {} | replay: leaves {} sky {}",
                t.elapsed().as_secs_f64() * 1000.0 / BENCH_FRAMES as f64,
                stats.frustum_queries.load(Relaxed),
                stats.frustum_nodes.load(Relaxed),
                stats.replay_leaf_tiles.load(Relaxed),
                stats.replay_sky_tiles.load(Relaxed),
            );
        }
    }

    // Temporal claim ring: claims are standalone world-space facts, so a
    // region that pans off the newest cache's screen and back is answered by
    // an older ring entry. Produce cache B far off to the side (consuming
    // [A], which also exercises production on a warm ring), then verify near
    // A's pose with ring [B, A]: on the pan-back side the query extremes
    // project off B's screen (an OffScreen miss) and A answers — and every
    // consumed claim still passes the per-pixel reference re-trace.
    let mut ring_ok = true;
    {
        let mut cam_far = cam0;
        cam_far.yaw += 0.35;
        let basis_far = cam_far.basis(rw, rh);
        let tc_b = temporal::TemporalCache::new(rw, rh);
        {
            let warm_a = [(&tcache, cam)];
            let ctx = FrameCtx {
                scene,
                bvh,
                cam: basis_far,
                q,
                frame: 0,
                jitter: false,
                rw,
                rh,
                accum: &accum,
                info: &info,
                tbuf: &tbuf,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: Some(&tc_b),
                tcache_prev: &warm_a,
                accumulate: true,
                gbuf: None,
                fsr_buf: None,
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
                hemi_share: false,
                replay_rec: None,
                cut_cur: None,
                cut_prev: None,
                discard_seeds: false,
                defer_shade: false,
            };
            render::render_frame(&ctx, true);
        }
        // Exact pan-back (bit-equal to A's basis): tiles that project off
        // B's screen retry A and take its verbatim identical-basis path —
        // the structural must-fire. The near-pose pass below (not bit-equal)
        // exercises the general-query retry and is gated on correctness
        // only: whether an older general query beats "not useful" is
        // scene-landscape-dependent, exactly like T5's seed count.
        let ring = [(&tc_b, basis_far), (&tcache, cam)];
        stats.clear();
        let rep_r = render::verify(scene, bvh, &cam, q, rw, rh, &stats, None, &ring, None);
        let hits = stats.temporal_ring_hits.load(Relaxed);
        let seeds = stats.temporal_seeds.load(Relaxed);
        let sky = stats.temporal_sky_tiles.load(Relaxed);
        eprintln!(
            "verify temporal ring pan-back exact: false-sky {} | tmin-overshoot {} | hybrid-extra {} | ring hits {hits} | seeds {seeds} sky-tiles {sky}",
            rep_r.false_sky, rep_r.overshoot, rep_r.hybrid_extra,
        );
        ring_ok &= rep_r.ok();
        if structural && hits == 0 {
            eprintln!("verify temporal ring pan-back exact: expected ring hits > 0 — the ring path didn't fire");
            ring_ok = false;
        }
        let mut cam_back = cam0;
        cam_back.yaw += 0.002;
        stats.clear();
        let rep_n = render::verify(scene, bvh, &cam_back.basis(rw, rh), q, rw, rh, &stats, None, &ring, None);
        eprintln!(
            "verify temporal ring pan-back near: false-sky {} | tmin-overshoot {} | hybrid-extra {} | ring hits {} | seeds {} sky-tiles {}",
            rep_n.false_sky,
            rep_n.overshoot,
            rep_n.hybrid_extra,
            stats.temporal_ring_hits.load(Relaxed),
            stats.temporal_seeds.load(Relaxed),
            stats.temporal_sky_tiles.load(Relaxed),
        );
        ring_ok &= rep_n.ok();
    }

    // Cut-adoption multi-hop chain: 4 consecutive dolly steps, each frame
    // producing (claims, cuts) while consuming the previous pair with
    // adoption on, each consumer re-verified against the pair it consumed
    // (per-pixel reference rays — a stale or too-small chained cut surfaces
    // as overshoot/false-sky). MAX_ADOPT_AGE = 3, so by step 4 age-capped
    // nodes must appear and force requeries — the decay control must-fire.
    let mut adopt_ok = true;
    {
        let chain_tc = [temporal::TemporalCache::new(rw, rh), temporal::TemporalCache::new(rw, rh)];
        let chain_cs = [temporal::CutStore::new(rw, rh), temporal::CutStore::new(rw, rh)];
        let mut chain_cam = cam0;
        let mut prev_basis = cam;
        {
            // Step 0: cold producer at cam0.
            let ctx = FrameCtx {
                scene,
                bvh,
                cam,
                q,
                frame: 0,
                jitter: false,
                rw,
                rh,
                accum: &accum,
                info: &info,
                tbuf: &tbuf,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: Some(&chain_tc[0]),
                tcache_prev: &[],
                accumulate: true,
                gbuf: None,
                fsr_buf: None,
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
                hemi_share: false,
                replay_rec: None,
                cut_cur: Some(&chain_cs[0]),
                cut_prev: None,
                discard_seeds: false,
                defer_shade: false,
            };
            render::render_frame(&ctx, true);
        }
        let (mut adopts_total, mut requery_total) = (0u64, 0u64);
        for step in 1..=4usize {
            let (pi, ci) = ((step + 1) & 1, step & 1);
            chain_cam.pos += chain_cam.forward() * (0.01 * scene.diag);
            let basis_s = chain_cam.basis(rw, rh);
            chain_tc[ci].clear();
            chain_cs[ci].clear();
            stats.clear();
            {
                let prev_ring = [(&chain_tc[pi], prev_basis)];
                let ctx = FrameCtx {
                    scene,
                    bvh,
                    cam: basis_s,
                    q,
                    frame: 0,
                    jitter: false,
                    rw,
                    rh,
                    accum: &accum,
                    info: &info,
                    tbuf: &tbuf,
                    stats: &stats,
                    sun: render::sun_dir(scene),
                    tcache_cur: Some(&chain_tc[ci]),
                    tcache_prev: &prev_ring,
                    accumulate: true,
                    gbuf: None,
                    fsr_buf: None,
                    prev_cam: None,
                    frame_jitter: None,
                    adaptive: false,
                    hemi_share: false,
                    replay_rec: None,
                    cut_cur: Some(&chain_cs[ci]),
                    cut_prev: Some(&chain_cs[pi]),
                    discard_seeds: false,
                    defer_shade: false,
                };
                render::render_frame(&ctx, true);
            }
            let a = stats.temporal_cut_adopts.load(Relaxed);
            let r = stats.temporal_adopt_requery.load(Relaxed);
            let af = stats.temporal_cut_arena_full.load(Relaxed);
            adopts_total += a;
            requery_total += r;
            stats.clear();
            let prev_ring = [(&chain_tc[pi], prev_basis)];
            let rep_s = render::verify(scene, bvh, &basis_s, q, rw, rh, &stats, None, &prev_ring, Some(&chain_cs[pi]));
            eprintln!(
                "adopt chain step {step}: adopts {a} requery {r} arena-full {af} | verify false-sky {} | tmin-overshoot {} | hybrid-extra {}",
                rep_s.false_sky, rep_s.overshoot, rep_s.hybrid_extra,
            );
            adopt_ok &= rep_s.ok();
            prev_basis = basis_s;
        }
        if structural && adopts_total == 0 {
            eprintln!("adopt chain: expected adoptions > 0 across the chain — the path didn't fire");
            adopt_ok = false;
        }
        if structural && requery_total == 0 {
            eprintln!("adopt chain: expected age-capped requeries > 0 by step 4 — decay control didn't fire");
            adopt_ok = false;
        }

        // A/B + KILL CRITERION (the shafts / specular-cone precedent): warm
        // dolly frames with adoption off vs on. If adopt-on is not
        // measurably faster on BOTH the default scene and --stress 5000, C
        // does not merge — skipped bound queries must beat the containment
        // walk plus the (possibly coarser) adopted-cut refine.
        // The 1-spp rows are the workload the skip targets: DLSS/XeSS motion
        // frames trace full-depth at 1 shadow / 1 AO sample, where the
        // quadtree is a large share of the frame; the preset-q rows show the
        // heavy-shading dilution for context.
        let q1 = Quality { shadow_samples: 1, ao_samples: 1, reflections: true, fb: shade::FrustumBounce::OFF };
        for (label, bq, cuts_opt) in [
            ("dolly warm q2 (adopt off) ", q, None),
            ("dolly warm q2 (adopt on)  ", q, Some(&cuts_a)),
            ("dolly warm 1spp (adopt off)", q1, None),
            ("dolly warm 1spp (adopt on) ", q1, Some(&cuts_a)),
        ] {
            stats.clear();
            let ctx = FrameCtx {
                scene,
                bvh,
                cam: basis_b,
                q: bq,
                frame: 0,
                jitter: false,
                rw,
                rh,
                accum: &accum,
                info: &info,
                tbuf: &tbuf,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: None,
                tcache_prev: &warm,
                accumulate: true,
                gbuf: None,
                fsr_buf: None,
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
                hemi_share: false,
                replay_rec: None,
                cut_cur: None,
                cut_prev: cuts_opt,
                discard_seeds: false,
                defer_shade: false,
            };
            // More frames than the other benches: the effect competes with
            // run-to-run noise at this frame time, and the kill criterion
            // hangs on the delta.
            const ADOPT_BENCH_FRAMES: u32 = 24;
            let t = Instant::now();
            for _ in 0..ADOPT_BENCH_FRAMES {
                render::render_frame(&ctx, true);
            }
            eprintln!(
                "adopt A/B {label}: {:5.2} ms | frustum queries {} (blocked {}) nodes {} | ray nodes {} | adopts {} sky {}",
                t.elapsed().as_secs_f64() * 1000.0 / ADOPT_BENCH_FRAMES as f64,
                stats.frustum_queries.load(Relaxed) / ADOPT_BENCH_FRAMES as u64,
                stats.blocked_queries.load(Relaxed) / ADOPT_BENCH_FRAMES as u64,
                stats.frustum_nodes.load(Relaxed) / ADOPT_BENCH_FRAMES as u64,
                stats.ray_nodes.load(Relaxed) / ADOPT_BENCH_FRAMES as u64,
                stats.temporal_cut_adopts.load(Relaxed) / ADOPT_BENCH_FRAMES as u64,
                stats.temporal_adopt_sky.load(Relaxed) / ADOPT_BENCH_FRAMES as u64,
            );
        }
    }

    // Deferred material-sorted shading (--defer-shade): a same-seed frame
    // with deferral off vs on must be BIT-IDENTICAL — the records carry each
    // pixel's rng stream and every accum/tbuf/info/G-buffer write is
    // single-writer, so reordering shading may not change one bit. The
    // off/on rows are the kill-criterion bench (the shaft precedent: the
    // tiled shader earns a default only by measuring faster on a real
    // textured scene). Scenes with no textured material skip — deferral
    // structurally never engages there (the leaf probe shades inline).
    let mut defer_ok = true;
    if !scene.materials.iter().any(|m| m.any_tex()) {
        eprintln!("defer gates: no textured materials — deferral never engages (skipped)");
    } else {
        let alloc3 = || (0..rw * rh * 3).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();
        let alloc1 = || (0..rw * rh).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();
        let bits_differ = |a: &[AtomicU32], b: &[AtomicU32]| {
            a.iter().zip(b).filter(|(x, y)| x.load(Relaxed) != y.load(Relaxed)).count()
        };
        let (accum_a, info_a, tbuf_a) = (alloc3(), alloc1(), alloc1());
        let (accum_b, info_b, tbuf_b) = (alloc3(), alloc1(), alloc1());
        for (bufs, defer) in [
            ((&accum_a, &info_a, &tbuf_a), false),
            ((&accum_b, &info_b, &tbuf_b), true),
        ] {
            stats.clear();
            let ctx = FrameCtx {
                scene,
                bvh,
                cam,
                q,
                frame: 0,
                jitter: false,
                rw,
                rh,
                accum: bufs.0,
                info: bufs.1,
                tbuf: bufs.2,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: None,
                tcache_prev: &[],
                accumulate: true,
                gbuf: None,
                fsr_buf: None,
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
                hemi_share: false,
                replay_rec: None,
                cut_cur: None,
                cut_prev: None,
                discard_seeds: false,
                defer_shade: defer,
            };
            render::render_frame(&ctx, true);
        }
        let dpx = stats.defer_px.load(Relaxed);
        let (dt, di, da) = (
            bits_differ(&tbuf_a, &tbuf_b),
            bits_differ(&info_a, &info_b),
            bits_differ(&accum_a, &accum_b),
        );
        eprintln!("defer bit-identity: tbuf {dt} | info {di} | accum {da} px differ | deferred px {dpx}");
        if dt + di + da > 0 {
            defer_ok = false;
        }
        // A must-fire, so `structural` gates it like every other one: the
        // scene having textured materials does not mean THIS VIEW contains
        // any (a loaded OBJ at a custom --cam can legitimately see only the
        // ground quad), and the bit-identity half above is the soundness
        // signal that runs everywhere.
        if structural && dpx == 0 {
            eprintln!("defer: textured scene deferred 0 px — the path didn't fire");
            defer_ok = false;
        }
        for (label, defer) in [("defer A/B (off)", false), ("defer A/B (on) ", true)] {
            stats.clear();
            let ctx = FrameCtx {
                scene,
                bvh,
                cam,
                q,
                frame: 0,
                jitter: false,
                rw,
                rh,
                accum: &accum_a,
                info: &info_a,
                tbuf: &tbuf_a,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur: None,
                tcache_prev: &[],
                accumulate: true,
                gbuf: None,
                fsr_buf: None,
                prev_cam: None,
                frame_jitter: None,
                adaptive: false,
                hemi_share: false,
                replay_rec: None,
                cut_cur: None,
                cut_prev: None,
                discard_seeds: false,
                defer_shade: defer,
            };
            const DEFER_BENCH_FRAMES: u32 = 24;
            let t = Instant::now();
            for _ in 0..DEFER_BENCH_FRAMES {
                render::render_frame(&ctx, true);
            }
            eprintln!(
                "{label}: {:5.2} ms | defer px {} flushes {} mixed {}",
                t.elapsed().as_secs_f64() * 1000.0 / DEFER_BENCH_FRAMES as f64,
                stats.defer_px.load(Relaxed) / DEFER_BENCH_FRAMES as u64,
                stats.defer_flushes.load(Relaxed) / DEFER_BENCH_FRAMES as u64,
                stats.defer_mixed.load(Relaxed) / DEFER_BENCH_FRAMES as u64,
            );
        }
    }

    let gates = [
        ("texture", tex_ok),
        ("sh", sh_ok),
        ("sky", sky_ok),
        ("bloom", bloom_ok),
        ("sphcell", sph_ok),
        ("ftree", ftree_ok),
        ("reproject", reproj_ok),
        ("nppd", nppd_ok),
        ("matclass", matclass_ok),
        ("tangent", tangent_ok),
        ("upchain", upchain_ok),
        ("gltf", gltf_ok),
        ("bc7", bc7_ok),
        ("hemi-ao", hemi_ok),
        ("hemi-gi", gi_ok),
        ("hemi-share", hemi_share_ok),
        ("verify", rep.ok()),
        ("wide-tiles", rep_wt.ok()),
        ("capped", capped_ok),
        ("temporal", temporal_ok),
        ("replay", replay_ok),
        ("ring", ring_ok),
        ("adopt", adopt_ok),
        ("defer", defer_ok),
    ];
    let failed: Vec<&str> = gates.iter().filter(|(_, ok)| !ok).map(|(n, _)| *n).collect();
    if failed.is_empty() {
        eprintln!("CHECK PASSED");
        0
    } else {
        eprintln!("CHECK FAILED ({})", failed.join(", "));
        1
    }
}

/// Extract the Win32 HWND from the SDL2 window for swapchain creation.
#[cfg(windows)]
fn sdl_hwnd(window: &sdl2::video::Window) -> windows::Win32::Foundation::HWND {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let handle = window.window_handle().expect("window handle").as_raw();
    match handle {
        RawWindowHandle::Win32(h) => {
            windows::Win32::Foundation::HWND(h.hwnd.get() as *mut core::ffi::c_void)
        }
        _ => unreachable!("non-Win32 window handle on Windows"),
    }
}

/// Adoption-only DRS logging: one line per adopted endpoint (ramp
/// intermediates land silently), tracked via the caller's last-logged
/// endpoint. `(0, 0)` marks "nothing logged yet" — the first adoption of a
/// session is silent.
#[cfg(windows)]
fn log_drs_adoption(path: &str, lim: &xess::StepLimiter, last_ep: &mut (usize, usize)) {
    let ep = lim.endpoint();
    if ep == *last_ep {
        return;
    }
    if last_ep.0 != 0 {
        if lim.ramping() {
            eprintln!(
                "{path}: drs {}x{} -> {}x{} (ramp {} frames)",
                last_ep.0,
                last_ep.1,
                ep.0,
                ep.1,
                xess::RAMP_FRAMES
            );
        } else {
            eprintln!("{path}: drs shed -> {}x{}", ep.0, ep.1);
        }
    }
    *last_ep = ep;
}

/// The GPU arm's upscaler sub-mode (--gpu and the DXR pipeline share it):
/// which wired upscaler composes on the tracer, or plain presentation.
/// `Fsr4` = Ray Regeneration + FSR4 fed on-GPU (the nine-plane feed);
/// `Fsr3` = the FSR 3.1 upscale-only chain level. K toggles either against
/// plain, mirroring G/X.
#[cfg(windows)]
#[derive(Clone, Copy, PartialEq)]
enum GpuUp {
    Plain,
    Rr,
    Xess,
    Fsr4,
    Fsr3,
}

/// A session's exit reason: quit the app, or rebuild everything at a new
/// window client size (maximize / F11 fullscreen / drag-resize settled).
#[cfg(windows)]
enum SessionEnd {
    Quit,
    Resize(u32, u32),
}

/// User-visible state that survives a window-resize session restart: mode
/// toggles, denoiser intents, counters. The camera pose is NOT here — the
/// flycam integrator thread owns it for the whole app lifetime (it keeps
/// flying through the rebuild) and the next session just snapshots it.
/// Everything else — every buffer, controller, history, and upscaler
/// context — intentionally rebuilds at the new size by re-entering the
/// session init code (which is the same code that already handles every
/// mode/fallback at any (w, h)).
#[cfg(windows)]
#[derive(Clone, Copy)]
struct Persist {
    hybrid: bool,
    dynamic: bool,
    overlay_on: bool,
    gpu_tonemap: bool,
    bounce_mode: u32,
    preset: u32,
    dlss_on: bool,
    xess_on: bool,
    fsr_on: bool,
    oidn_on: bool,
    oidn_temporal: bool,
    xess_oidn: XessOidn,
    nppd_on: bool,
    xess_nppd: bool,
    dxr_on: bool,
    /// The user toggled the DXR / --gpu arm to plain presentation (G/X
    /// inside the arm); the wired upscaler itself is re-derived at re-entry.
    dxr_up_plain: bool,
    gpu_up_plain: bool,
    gpu_nppd_on: bool,
    /// Failed-init latches: a missing DLL stays missing across a resize —
    /// don't retry per re-entry.
    oidn_failed: bool,
    nppd_failed: bool,
    dxr_failed: bool,
    shot: u32,
    depth_est: f32,
}

#[cfg(windows)]
fn run_window(scene: &scene::Scene, bvh: &bvh::Bvh, opts: &Opts, cam0: Camera) {
    // Opt into per-monitor DPI awareness so Windows doesn't stretch the window
    // on scaled displays — W×H stays W×H physical pixels, matching the swapchain.
    sdl2::hint::set("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2");
    let sdl = sdl2::init().expect("SDL init failed");
    let video = sdl.video().expect("SDL video failed");
    let mut window = video
        .window(
            "frustracer — R: hybrid/plain  T: dynamic-res  O: overlay  B: gpu-tonemap  G: dlss  X: xess  N: oidn  H: hemi-bounce  1-3: quality  C: verify  P: screenshot",
            W as u32,
            H as u32,
        )
        .position_centered()
        .resizable() // enables the maximize button + drag borders
        .build()
        .expect("failed to open window");
    let mut inp = input::Input::new(&sdl).expect("SDL event pump failed");
    // Hoisted so GpuContext::new and every resize_output share one value.
    let gopts = gpu::GpuOptions {
        chain: opts.chain,
        sl_dir: opts.sl_path.clone(),
        xess_dir: opts.xess_path.clone(),
        xess_autoexposure: opts.xess_autoexposure,
        ffx_dir: opts.ffx_path.clone(),
        prefer: opts.prefer,
        debug: opts.gpu_debug,
        vsync: opts.vsync,
    };
    let mut gpu = gpu::GpuContext::new(sdl_hwnd(&window), W as u32, H as u32, &gopts)
        .expect("GPU init failed");
    // The 500 Hz integrator owns the camera pose for the whole app lifetime
    // (spawned here, not per session, so the pose survives resize rebuilds —
    // re-entry continuity is automatic). Sessions only snapshot. It spawns
    // PAUSED and each session resumes it once its frame loop is live: a long
    // FRAME must integrate (the whole point), but a rebuild presents nothing,
    // and flying through a seconds-long BLAS build with W held is flying
    // blind.
    let fly =
        flycam::FlyCam::spawn(sdl_hwnd(&window).0 as isize, cam0, scene.diag);

    // A window resize exits the session and re-enters it at the new client
    // size: the session init code IS the rebuild path (every buffer,
    // controller, history, and tracer pipeline re-derives from (w, h) —
    // including the upscalers' render-res ranges, so a --lock-res scale
    // stays the same PERCENTAGE of the new window). The GpuContext itself
    // survives via resize_output; user-visible state crosses over in
    // Persist.
    let (mut w, mut h) = (W, H);
    let mut persist: Option<Persist> = None;
    loop {
        match session(scene, bvh, opts, &fly, &mut window, &mut inp, &mut gpu, &mut persist, w, h)
        {
            SessionEnd::Quit => break,
            SessionEnd::Resize(nw, nh) => {
                // No frames from here until the next session's loop starts.
                fly.pause();
                if let Err(e) = gpu.resize_output(nw, nh, &gopts) {
                    eprintln!("resize: rebuild at {nw}x{nh} failed ({e}); exiting");
                    break;
                }
                (w, h) = (nw as usize, nh as usize);
                eprintln!("window: resized to {nw}x{nh}");
            }
        }
    }
}

/// One renderer session at a fixed window size (w, h): everything sized
/// from the window lives in here and rebuilds when a resize re-enters.
/// Seeds user-visible state from `persist` (subsequent entries) or the CLI
/// flags (first entry), and writes it back on every exit.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn session(
    scene: &scene::Scene,
    bvh: &bvh::Bvh,
    opts: &Opts,
    fly: &flycam::FlyCam,
    window: &mut sdl2::video::Window,
    inp: &mut input::Input,
    gpu: &mut gpu::GpuContext,
    persist: &mut Option<Persist>,
    w: usize,
    h: usize,
) -> SessionEnd {
    let p0: Option<Persist> = *persist;
    // Denoiser/pipeline intents: first entry from the CLI flags, re-entries
    // from the persisted runtime state (the CLI branches below double as the
    // restore path — they rebuild the contexts at the new size).
    let want_oidn = p0.map_or(opts.oidn, |p| p.oidn_on || p.xess_oidn == XessOidn::Pre);
    let want_oidn_post = p0.map_or(opts.oidn_post, |p| p.xess_oidn == XessOidn::Post);
    let want_nppd = p0.map_or(opts.nppd, |p| p.nppd_on || p.xess_nppd);
    let want_dxr = p0.map_or(opts.dxr, |p| p.dxr_on);

    // The integrator thread owns the pose (it kept flying through any resize
    // rebuild); the session works exclusively on per-iteration snapshots.
    let mut cam = fly.snapshot();
    let mut prev_snap = cam;
    let accum: Vec<AtomicU32> = (0..w * h * 3).map(|_| AtomicU32::new(0)).collect();
    let info: Vec<AtomicU32> = (0..w * h).map(|_| AtomicU32::new(0)).collect();
    let tbuf: Vec<AtomicU32> = (0..w * h).map(|_| AtomicU32::new(0)).collect();
    let mut present = vec![0u32; w * h];
    let stats = Stats::default();
    // Temporal claim ring + lockstep cut stores: see TemporalRing. Half-res
    // and plain frames don't participate and drop the ring — a cache from
    // any older frame or another resolution is never consulted.
    let mut tr = TemporalRing::new(w, h);
    // Static-frame structure replay (replay.rs): `replay_key` describes the
    // immediately preceding rendered frame's recorded terminal structure, or
    // None. A frame either records and sets it, replays and leaves it, or
    // does neither and clears it; idle (converged) frames don't touch it.
    let replay_cache = replay::ReplayCache::new(w, h);
    let mut replay_key: Option<(camera::CamBasis, (usize, usize))> = None;

    let mut frame: u32 = 0;
    let mut hybrid = p0.map_or(true, |p| p.hybrid);
    let mut dynamic = p0.map_or(true, |p| p.dynamic);
    let mut overlay_on = p0.map_or(false, |p| p.overlay_on);
    let mut gpu_tonemap = p0.map_or(false, |p| p.gpu_tonemap);
    // Hemisphere frustum bounces (H cycles off → AO → GI): still-frame
    // quality — moving/DLSS frames keep the sampled path.
    let mut bounce_mode = p0.map_or(0u32, |p| p.bounce_mode);
    let mut preset = p0.map_or(2u32, |p| p.preset);
    let mut prev_rw = w;

    // DLSS Ray Reconstruction state. In DLSS mode every frame is a fresh
    // 1-spp hybrid frame at RR's Quality-mode render resolution (RR upscales
    // + denoises to the window size) and RR is the only temporal integrator:
    // no CPU accumulation, no half-res moving mode, no depth-cap budget.
    let mut dlss_on = p0.map_or(gpu.dlss_ready(), |p| p.dlss_on && gpu.dlss_ready());
    // Step-wise DRS (shares xess::ScaleCtl / quantize_res — pure controller
    // math common to both upscalers). Steps are made RARE (the quantization
    // plus the StepLimiter dwell is the hysteresis) because RR re-initializes
    // its internal denoiser on an input-res change — but a step is a scale
    // change, not a scene change: the res-step block below does NOT reset
    // (no dlss_reset, no prev drop; history survives via the extent tags).
    // A degenerate reported range (min == max) means the driver offers no
    // DRS — fixed res, no controller. --lock-res (default quality = 2/3)
    // pins a fixed res inside the range instead; `--lock-res dynamic` opts
    // back into the controller.
    let dlss_range = gpu.rr_res_range();
    let (drw, drh) = match (opts.lock_scale, dlss_range) {
        (Some(r), Some((_, min, max))) if min != max => xess::quantize_res(
            r,
            (w, h),
            (min.0 as usize, min.1 as usize),
            (max.0 as usize, max.1 as usize),
        ),
        // No usable range: the lock can't be honored — keep the SDK
        // optimal / DLAA fallback (warned about at the startup print).
        _ => gpu.rr_render_res().map(|(a, b)| (a as usize, b as usize)).unwrap_or((w, h)),
    };
    let dlss_drs = opts.lock_scale.is_none()
        && dlss_range.map(|(_, min, max)| min != max).unwrap_or(false);
    let mut dlss_ctl = dlss_range.filter(|_| dlss_drs).map(|(_, min, max)| {
        let start_h = (h * 2 / 3).clamp(min.1 as usize, max.1 as usize);
        xess::ScaleCtl::new(start_h, min.1 as usize, max.1 as usize, h)
    });
    // G-buffer capacity = the range max; reinterpreted per step via set_res.
    let mut gbufs = {
        let (gw, gh) = dlss_range
            .map(|(_, _, max)| (max.0 as usize, max.1 as usize))
            .unwrap_or((drw, drh));
        dlss::GBufs::new(gw.max(drw), gh.max(drh))
    };
    gbufs.set_res(drw, drh); // fixed-res default until the controller speaks
    let mut dlss_idx: u32 = 0;
    let mut dlss_prev: Option<dlss::DlssPrev> = None;
    let mut dlss_reset = true;
    // Applied-step rate limiters, one per DRS path: bound how often the
    // upscalers take a history hit, and ramp each adopted step over
    // RAMP_FRAMES instead of snapping (see xess::StepLimiter).
    let mut dlss_lim = xess::StepLimiter::new(xess::RAMP_FRAMES);
    let mut xess_lim = xess::StepLimiter::new(xess::RAMP_FRAMES);
    // Last logged endpoints — adoption-only DRS logging (a ramp would
    // otherwise print every intermediate res).
    let mut dlss_ep = (0usize, 0usize);
    let mut xess_ep = (0usize, 0usize);
    // FRUSTRACER_STAB=1: numeric stability meter — every 15th upscaled frame
    // is read back and diffed against the previous capture; a static camera
    // on a converged pipeline trends to ~0, temporal instability ("dancing")
    // shows as a persistently high mean.
    let stab_on = std::env::var("FRUSTRACER_STAB").is_ok();
    let mut stab_prev: Option<Vec<u32>> = None;
    let mut stab_n = 0u32;
    let (dlss_near, dlss_far) = dlss::near_far(scene.diag);
    if dlss_on {
        eprintln!(
            "dlss: Ray Reconstruction ON (G toggles), dynamic resolution {}",
            if dlss_drs && opts.gpu {
                // The GPU arm locks the render res; the `gpu:` line that
                // follows states the resolution actually chosen.
                "requested (--lock-res dynamic) — locked under --gpu, see the gpu: line".to_string()
            } else if dlss_drs {
                "ON (step-wise; history survives steps)".to_string()
            } else if opts.lock_scale.is_some() {
                if dlss_range.map(|(_, min, max)| min != max).unwrap_or(false) {
                    format!(
                        "LOCKED at {}x{} ({}%, --lock-res{})",
                        drw,
                        drh,
                        (drh * 100 + h / 2) / h,
                        // No DRS under --gpu — don't advertise re-enabling it.
                        if opts.gpu { "" } else { "; `--lock-res dynamic` re-enables" }
                    )
                } else {
                    format!("unavailable (no render-res range) — --lock-res not honorable, keeping {}x{}", drw, drh)
                }
            } else {
                "unavailable (degenerate range) — fixed render res".to_string()
            }
        );
    }

    // XeSS-SR state (--xess sessions; X toggles). The all-in "a pixel is a
    // sample" mode: every frame is a fresh jittered 1-spp full-depth hybrid
    // trace at a dynamic render resolution picked by the scale controller
    // (quantized inside the SDK's queried input range), and XeSS's temporal
    // accumulation of the jittered sample stream is the ONLY spatial
    // reconstruction — the depth-cap/quad-fill budget path, the half-res
    // moving mode, and CPU accumulation never run. N composes an OIDN
    // pre-denoise at the same dynamic render res (XeSS-SR is a TAA-upscaler,
    // not a denoiser); the render-res G-buffers are reinterpreted in place
    // on a res step (GBufs::set_res), and `xess_prev` is its own MV-basis
    // contract — dropped on any res step, since the previous frame's pixel
    // grid no longer matches.
    let mut xess_on = p0.map_or(gpu.xess_ready(), |p| p.xess_on && gpu.xess_ready());
    let xess_range = gpu.xess_res_range(); // (optimal, min, max)
    // --lock-res (default quality): one fixed render res for the whole
    // session — the ScaleCtl/StepLimiter pair is never built/consulted.
    // quantize_res clamps the requested scale into the SDK range.
    let xess_lock = opts.lock_scale.and_then(|r| {
        xess_range.map(|(_, min, max)| {
            xess::quantize_res(
                r,
                (w, h),
                (min.0 as usize, min.1 as usize),
                (max.0 as usize, max.1 as usize),
            )
        })
    });
    let mut xess_ctl = xess_range.filter(|_| xess_lock.is_none()).map(|(_, min, max)| {
        // Start at ~2/3 scale (the DLSS-Quality neighborhood), not the SDK's
        // "optimal" — with the ULTRA_PERFORMANCE init that widens the range,
        // optimal is the 1/3-scale floor and would open blurry. The
        // controller corrects from here either way.
        let start_h = (h * 2 / 3).clamp(min.1 as usize, max.1 as usize);
        xess::ScaleCtl::new(start_h, min.1 as usize, max.1 as usize, h)
    });
    let mut xess_gbufs: Option<dlss::GBufs> = None; // capacity = range max, lazily allocated
    let mut xess_idx: u32 = 0;
    // The previous frame's CAMERA (not basis): the MV basis is derived at
    // each frame's own resolution, so it stays correct across DRS steps
    // without dropping history.
    let mut xess_prev: Option<Camera> = None;
    let mut xess_reset = true;
    // OIDN placement in XeSS mode (independent of the plain-mode `oidn_on`);
    // xess_hdr is the post-placement's window-res readback staging.
    let mut xess_oidn = XessOidn::Off;
    let mut xess_hdr: Vec<f32> = Vec::new();
    if xess_on {
        if let Some((lw, lh)) = xess_lock {
            eprintln!(
                "xess: super-resolution ON, render res LOCKED at {}x{} ({}%, --lock-res; X toggles{})",
                lw,
                lh,
                (lh * 100 + h / 2) / h,
                // No DRS and no OIDN composition under --gpu.
                if opts.gpu {
                    ""
                } else {
                    "; `--lock-res dynamic` re-enables DRS; N cycles OIDN off/pre/post"
                }
            );
        } else if opts.gpu {
            // The GPU arm locks the render res; the `gpu:` line that follows
            // states the resolution actually chosen.
            eprintln!("xess: super-resolution ON (X toggles) — dynamic res locked under --gpu, see the gpu: line");
        } else {
            eprintln!("xess: dynamic super-resolution ON (X toggles; N cycles OIDN off/pre/post)");
        }
        if !opts.adaptive {
            eprintln!("xess: adaptive shading rate OFF (--no-adaptive; uniform per-pixel shading)");
        }
    }

    // FSR state (--fsr sessions; K toggles — F belongs to DXR): the XeSS
    // dynamic-res frame contract with Ray Regeneration as the denoiser. Same
    // controller /
    // quantizer / step-limiter machinery; `fsr_prev` is its own prev-camera
    // contract (basis re-derived at each frame's res, so MVs survive DRS
    // steps); `fsr_bufs` carries the demodulated signals + residual and is
    // reinterpreted in place on a step exactly like the G-buffers.
    let mut fsr_on = p0.map_or(gpu.fsr_ready(), |p| p.fsr_on && gpu.fsr_ready());
    // Which FSR pipeline initialized: FSR4 + Ray Regeneration (RDNA4) or
    // FSR 3.1 upscale-only (everything else / --fsr3). The flavor is fixed
    // for the session — K toggles it against plain, never across flavors.
    // Display strings come from fsr::Flavor::label/hud, never re-derived
    // per site (a hardcoded title-bar "FSR4" shipped wrong once).
    let fsr_flavor = gpu.fsr_flavor();
    let fsr_rr = fsr_flavor == Some(fsr::Flavor::Fsr4Rr);
    let fsr_label = fsr_flavor.map_or("FSR", fsr::Flavor::label);
    let (fsr_hud, fsr_hud_sfx) = fsr_flavor.map_or(("FSR", ""), fsr::Flavor::hud);
    let fsr_range = gpu.fsr_res_range(); // (seed, min, max)
    let fsr_lock = opts.lock_scale.and_then(|r| {
        fsr_range.map(|(_, min, max)| {
            xess::quantize_res(
                r,
                (w, h),
                (min.0 as usize, min.1 as usize),
                (max.0 as usize, max.1 as usize),
            )
        })
    });
    let mut fsr_ctl = fsr_range.filter(|_| fsr_lock.is_none()).map(|(opt, min, max)| {
        // Seed from the Quality-mode query (the 2/3-scale neighborhood) —
        // unlike XeSS there is no widened-range "optimal" pitfall, so the
        // queried seed is used as-is. The controller corrects from here.
        let start_h = (opt.1 as usize).clamp(min.1 as usize, max.1 as usize);
        xess::ScaleCtl::new(start_h, min.1 as usize, max.1 as usize, h)
    });
    let mut fsr_lim = xess::StepLimiter::new(xess::RAMP_FRAMES);
    let mut fsr_ep = (0usize, 0usize);
    let mut fsr_gbufs: Option<dlss::GBufs> = None; // capacity = range max, lazily allocated
    let mut fsr_bufs: Option<fsr::FsrBufs> = None; // ditto (signal planes)
    let mut fsr_idx: u32 = 0;
    let mut fsr_prev: Option<Camera> = None;
    let mut fsr_reset = true;
    if fsr_on {
        if let Some((lw, lh)) = fsr_lock {
            eprintln!(
                "fsr: {} ON, render res LOCKED at {}x{} ({}%, --lock-res; K toggles; `--lock-res dynamic` re-enables DRS)",
                fsr_label,
                lw,
                lh,
                (lh * 100 + h / 2) / h,
            );
        } else {
            eprintln!("fsr: {fsr_label} dynamic super-resolution ON (K toggles)");
        }
    }

    // --gpu: bring up the GPU-resident tracer (DXC compile, scene upload,
    // BLAS/TLAS, and — in upscaler sessions — the feed wiring). The session
    // sub-mode mirrors the CPU defaults: DLSS-RR when supported, XeSS with
    // --xess, plain with --no-dlss. The render resolution is LOCKED for the
    // session (from --lock-res, default quality = 2/3, quantized into the
    // upscaler's range) — the tracer's buffers are sized to it once; there
    // is no DRS on the GPU path. Any init failure falls back to the CPU
    // renderer with the reason on stderr. With Streamline live the tracer's
    // whole workload (ExecuteIndirect + DXR + AS builds) executes on the SL
    // PROXY queue — validated in M7.
    let mut gpu_up = GpuUp::Plain;
    let mut gpu_trace = false;
    let (mut grw, mut grh) = (w, h);
    if opts.gpu {
        // Session sub-mode + locked trace res. `--lock-res dynamic` can't be
        // honored here (no DRS on the GPU path): lock at the quality default.
        let lock = opts.lock_scale.unwrap_or_else(|| {
            if dlss_on || xess_on || fsr_on {
                eprintln!("gpu: dynamic render res is unsupported under --gpu; locking at quality (2/3)");
            }
            2.0 / 3.0
        });
        if dlss_on {
            gpu_up = GpuUp::Rr;
            // Degenerate range: the SDK optimal/DLAA res.
            (grw, grh) = locked_render_res(lock, dlss_range, (w, h), (drw, drh), false);
        } else if xess_on {
            gpu_up = GpuUp::Xess;
            (grw, grh) = locked_render_res(lock, xess_range, (w, h), (w, h), true);
        } else if fsr_on {
            // Both FSR chain levels compose on-GPU: FSR4-RR via the
            // nine-plane feed, FSR3 via the XeSS trio.
            gpu_up = match gpu.fsr_flavor() {
                Some(fsr::Flavor::Fsr4Rr) => GpuUp::Fsr4,
                _ => GpuUp::Fsr3,
            };
            (grw, grh) = locked_render_res(lock, fsr_range, (w, h), (w, h), true);
        }
        gpu_trace = match gpu::dxc::Dxc::load(&opts.dxc_path).and_then(|dxc| {
            gpu.init_trace(
                &dxc,
                scene,
                bvh,
                grw as u32,
                grh as u32,
                gpu_up != GpuUp::Plain,
                (gpu_up == GpuUp::Xess && opts.nppd)
                    .then_some((opts.nppd_path.as_str(), opts.nppd_model.as_str())),
                opts.gpu_debug,
                opts.bc7,
            )
        }) {
            Ok(()) => {
                eprintln!(
                    "gpu: GPU-resident tracer active ({grw}x{grh}, DXR RayQuery{})",
                    match gpu_up {
                        GpuUp::Plain => "".to_string(),
                        GpuUp::Rr => format!(" -> DLSS-RR {w}x{h}"),
                        GpuUp::Xess => format!(" -> XeSS-SR {w}x{h}"),
                        GpuUp::Fsr4 => format!(" -> FSR4-RR {w}x{h}"),
                        GpuUp::Fsr3 => format!(" -> FSR3 {w}x{h}"),
                    }
                );
                true
            }
            Err(e) => {
                eprintln!("gpu: falling back to CPU tracing — {e}");
                gpu_up = GpuUp::Plain;
                (grw, grh) = (w, h);
                false
            }
        };
        if gpu_trace && gpu_up == GpuUp::Rr {
            eprintln!("gpu: DLSS-RR ON (G toggles), render res LOCKED at {grw}x{grh}");
        }
    }
    // Upscaler sub-mode per-frame state: free-running frame index (the RNG /
    // jitter phase must advance even though accumulate is off), the previous
    // frame's camera (the MV contract — its own state, like dlss_prev), and
    // the history-reset latch (set on discontinuities, never on motion).
    let mut gpu_up_idx: u32 = 0;
    let mut gpu_prev_cam: Option<Camera> = None;
    let mut gpu_reset = true;
    // Whether this session wired an upscaler feed at init (the G/X/K toggles
    // can only move between Plain and a WIRED upscaler — wiring is
    // init-time).
    let gpu_xess_avail = gpu_trace && gpu_up == GpuUp::Xess;
    let gpu_rr_avail = gpu_trace && gpu_up == GpuUp::Rr;
    // The wired FSR kind (captured before the plain-toggle restore below —
    // the K toggle moves between Plain and exactly this).
    let gpu_fsr_kind = (gpu_trace && matches!(gpu_up, GpuUp::Fsr4 | GpuUp::Fsr3)).then_some(gpu_up);
    // GPU-resident NPPD (wired at init in --gpu --nppd XeSS sessions; J
    // toggles the pre-upscale slot, mirroring the CPU xess_nppd contract).
    let gpu_nppd_avail = gpu_trace && gpu.nppd_gpu_ready();
    let mut gpu_nppd_on = p0.map_or(gpu_nppd_avail, |p| p.gpu_nppd_on && gpu_nppd_avail);
    if gpu_nppd_avail {
        eprintln!("gpu: NPPD pre-upscale denoise ON (J toggles)");
    }
    // Restore the user's plain-presentation toggle (G/X inside the --gpu
    // arm) — AFTER the avail flags, so the session's wiring is untouched.
    if p0.map_or(false, |p| p.gpu_up_plain) && gpu_trace {
        gpu_up = GpuUp::Plain;
    }

    // OIDN state — the secondary denoiser (N toggles, mutually exclusive
    // with DLSS). It keeps the normal render loop (temporal cache, budget
    // frames, hemi bounces) at forced full-res (the half-res moving mode
    // writes a half-res prefix that would misalign the full-res G-buffers)
    // and denoises each rendered frame. Two sub-modes (M toggles): temporal
    // (default) renders fresh 1-spp frames and folds them into a reprojected
    // EMA history (reproject.rs) that is the sole accumulator and denoiser
    // input; plain denoises the accumulation average and shimmers while
    // moving. Context, w×h G-buffers (~116 MB) and history (~77 MB) are
    // lazily created on first enable; a failed init is remembered and not
    // retried per keypress.
    let mut oidn_on = false;
    let mut oidn_temporal = p0.map_or(opts.oidn_temporal, |p| p.oidn_temporal);
    let mut oidn_ctx: Option<oidn::OidnContext> = None;
    let mut oidn_gbufs: Option<dlss::GBufs> = None;
    let mut oidn_hist: Option<reproject::History> = None;
    // Free-running frame index for the temporal mode's per-pixel RNG seed —
    // the dlss_idx pattern: `frame` is pinned to 0 while moving, which would
    // freeze the noise pattern and defeat the history average.
    let mut oidn_seq: u32 = 0;
    let mut last_hist = reproject::UpdateStats::default();
    let mut hist_ms = 0.0f64;
    // Previous frame's XeSS pre-denoise cost (set_res + history + filter):
    // it scales with the render area just like the trace, so the resolution
    // controller must see it or a slow OIDN device (e.g. CPU) lets the scale
    // creep to the range max while total frame time blows the budget.
    let mut pre_ms = 0.0f64;
    let mut oidn_failed = p0.map_or(false, |p| p.oidn_failed);
    // "XeSS session" for the SYCL-avoidance pick below: XeSS is the chain
    // level that got wired (libxess.dll is live in-process).
    let xess_wired = gpu.xess_ready();
    let oidn_try_enable =|oidn_ctx: &mut Option<oidn::OidnContext>,
                           oidn_gbufs: &mut Option<dlss::GBufs>,
                           oidn_hist: &mut Option<reproject::History>| {
        if oidn_ctx.is_none() {
            // XeSS sessions must not let OIDN auto-pick its SYCL device: the
            // SYCL runtime and libxess.dll drag conflicting Intel compute
            // stacks into one process and abort() natively at first use
            // (observed: OIDN 2.5 SYCL + XeSS SDK 2.0.2). Auto in a XeSS
            // session means CUDA then CPU; an explicit --oidn-device is
            // honored as given.
            let devices: &[i32] = if xess_wired && opts.oidn_device == 0 {
                &[3, 1] // cuda, cpu
            } else {
                &[opts.oidn_device]
            };
            for &d in devices {
                match oidn::OidnContext::new(
                    &opts.oidn_path,
                    w,
                    h,
                    d,
                    opts.oidn_quality,
                    opts.oidn_clean_aux,
                ) {
                    Ok(c) => {
                        eprintln!("oidn: ready on {} device", c.device_desc);
                        *oidn_ctx = Some(c);
                        break;
                    }
                    Err(e) => eprintln!("{e}"),
                }
            }
            if oidn_ctx.is_none() {
                eprintln!(
                    "oidn: DLLs expected at {} (--oidn-path / FRUSTRACER_OIDN_PATH)",
                    opts.oidn_path
                );
            }
        }
        if oidn_ctx.is_some() && oidn_gbufs.is_none() {
            *oidn_gbufs = Some(dlss::GBufs::new(w, h));
        }
        if oidn_ctx.is_some() && oidn_hist.is_none() {
            *oidn_hist = Some(reproject::History::new(w, h));
        }
        oidn_ctx.is_some()
    };
    if (want_oidn || want_oidn_post) && !oidn_failed {
        if oidn_try_enable(&mut oidn_ctx, &mut oidn_gbufs, &mut oidn_hist) {
            if xess_on {
                // XeSS sessions: --oidn = pre-upscale placement, --oidn-post
                // = post-upscale; the plain-mode oidn_on stays independent.
                xess_oidn = if want_oidn_post { XessOidn::Post } else { XessOidn::Pre };
                eprintln!(
                    "oidn: {}-upscale denoise ON (N cycles off/pre/post)",
                    if want_oidn_post { "POST" } else { "PRE" }
                );
            } else if want_oidn_post {
                eprintln!("oidn: --oidn-post requires a live --xess session; ignoring");
            } else {
                oidn_on = true;
                if dlss_on {
                    dlss_on = false;
                    eprintln!("dlss: Ray Reconstruction OFF (--oidn; G re-enables)");
                }
                if fsr_on {
                    // The chain can wire FSR in an --oidn session; the FSR
                    // present arm owns the frame, so it yields (K re-enables).
                    fsr_on = false;
                    fsr_prev = None;
                    eprintln!("fsr: OFF (--oidn; K re-enables)");
                }
                eprintln!(
                    "oidn: denoising ON, temporal reprojection {} (N / M toggle)",
                    if oidn_temporal { "ON" } else { "OFF" }
                );
            }
        } else {
            oidn_failed = true;
        }
    }

    // NPPD state — the neural denoiser (J toggles, mutually exclusive with
    // DLSS/OIDN/XeSS). It keeps the normal render loop at forced full-res
    // with fresh 1-spp frames (accumulate off, free-running nppd_seq) and
    // runs the recurrent network per frame: the state is backward-warped by
    // this frame's motion vectors (so prev_cam is set — the first
    // CPU-denoiser mode that needs it), the 10-channel stack packed from
    // accum + the G-buffers, and the ONNX session executed via DirectML (CPU
    // EP fallback). Context (~800 MB staging at 1080p — the 38-channel
    // recurrent state dominates) and w×h G-buffers are lazily created on
    // first enable; a failed init is remembered and not retried per keypress.
    // nppd_prev is its own prev-camera contract (dlss_prev/xess_prev
    // precedent), cleared on toggle, set only after a rendered NPPD frame.
    let mut nppd_on = false;
    // XeSS composition: NPPD as the pre-upscale denoiser at the (locked)
    // render res — the slot OIDN's Pre placement occupies, mutually
    // exclusive with it (J toggles; N cycling to pre/post turns this off).
    // Under --lock-res dynamic every DRS step invalidates the recurrent
    // state and re-specializes the DML graph — noted loudly, not forbidden.
    let mut xess_nppd = false;
    let mut nppd_ctx: Option<nppd::NppdContext> = None;
    let mut nppd_gbufs: Option<dlss::GBufs> = None;
    let mut nppd_prev: Option<Camera> = None;
    let mut nppd_seq: u32 = 0;
    let mut nppd_failed = p0.map_or(false, |p| p.nppd_failed);
    let nppd_drs_note = |lock: Option<f32>| {
        if lock.is_none() {
            eprintln!(
                "nppd: --lock-res dynamic steps the render res — each step resets the \
                 recurrent state and re-specializes the graph (a fixed --lock-res avoids it)"
            );
        }
    };
    // `want_gbufs`: the standalone mode fills its own window-res G-buffers;
    // the XeSS composition reads xess_gbufs at render res and must not
    // commit the ~116 MB window-res set. `(iw, ih)` is the res the FIRST
    // denoise will run at — the session's frozen dims (opening anywhere else
    // costs an immediate DML session rebuild, hundreds of ms); capacity is
    // always window-res, so a later set_res (X-off standalone J, dynamic-DRS
    // steps) stays within bounds.
    let nppd_try_enable = |nppd_ctx: &mut Option<nppd::NppdContext>,
                           nppd_gbufs: &mut Option<dlss::GBufs>,
                           want_gbufs: bool,
                           (iw, ih): (usize, usize)| {
        if nppd_ctx.is_none() {
            let dev = match opts.nppd_device {
                None => nppd::NppdDevice::Auto,
                Some(-1) => nppd::NppdDevice::Cpu,
                Some(n) => nppd::NppdDevice::Dml(n),
            };
            match nppd::NppdContext::with_capacity(
                &opts.nppd_path,
                &opts.nppd_model,
                iw,
                ih,
                w,
                h,
                dev,
            ) {
                Ok(c) => *nppd_ctx = Some(c),
                Err(e) => {
                    eprintln!("{e}");
                    eprintln!(
                        "nppd: DLLs expected at {} (--nppd-path / FRUSTRACER_ORT_PATH), \
                         model at {} (--nppd-model / FRUSTRACER_NPPD_MODEL — \
                         tools/nppd-export/export.py produces it)",
                        opts.nppd_path, opts.nppd_model
                    );
                }
            }
        }
        if want_gbufs && nppd_ctx.is_some() && nppd_gbufs.is_none() {
            *nppd_gbufs = Some(dlss::GBufs::new(w, h));
        }
        nppd_ctx.is_some()
    };
    if want_nppd && !nppd_failed && !gpu_trace {
        // (Under --gpu the NPPD stage is GPU-resident — nppd_gpu in
        // GpuContext, wired by init_trace; the CPU context never builds.)
        if xess_on {
            // XeSS session: --nppd lands the pre-upscale placement (the
            // NPPD context works at the render res; the window-res gbufs
            // stay unallocated — XeSS mode fills xess_gbufs).
            if nppd_try_enable(&mut nppd_ctx, &mut nppd_gbufs, false, xess_lock.unwrap_or((w, h)))
            {
                xess_nppd = true;
                if xess_oidn != XessOidn::Off {
                    xess_oidn = XessOidn::Off;
                    eprintln!("oidn: OFF (--nppd takes the XeSS pre-denoise slot)");
                }
                nppd_drs_note(opts.lock_scale);
                eprintln!("nppd: PRE-upscale denoise ON (J toggles)");
            } else {
                nppd_failed = true;
            }
        } else if nppd_try_enable(&mut nppd_ctx, &mut nppd_gbufs, true, (w, h)) {
            nppd_on = true;
            if dlss_on {
                dlss_on = false;
                eprintln!("dlss: Ray Reconstruction OFF (--nppd; G re-enables)");
            }
            if fsr_on {
                // Same yield as OIDN's: the chain can wire FSR here.
                fsr_on = false;
                fsr_prev = None;
                eprintln!("fsr: OFF (--nppd; K re-enables)");
            }
            if oidn_on {
                oidn_on = false;
                eprintln!("oidn: OFF (--nppd; N re-enables)");
            }
            eprintln!("nppd: neural denoising ON (J toggles)");
        } else {
            nppd_failed = true;
        }
    }
    // DXR pipeline state — the by-the-book DispatchRays mode (F toggles it
    // against the CPU tracer live; unavailable under --gpu, which is its
    // own session). By default it COMPOSES with the session's wired
    // upscaler exactly like --gpu: DXR-fed DLSS-RR in an SL session,
    // DXR-fed XeSS in a XeSS session, plain otherwise (--no-dlss / FSR / no
    // support). Inside the mode G/X toggle wired-upscaler <-> plain; the
    // CPU-side denoisers (OIDN/NPPD) and the rival FSR presenter stay
    // mutually exclusive. Lazily built on first enable — DXC load, RTPSO
    // compile, scene + BLAS/TLAS upload, feed wiring, all once; a failed
    // init is remembered and not retried per keypress. Plain sub-mode frame
    // semantics mirror the --gpu plain sub-mode.
    let mut dxr_on = false;
    let mut dxr_failed = p0.map_or(false, |p| p.dxr_failed);
    // Availability = the session's WIRING (an SL session's queue is the
    // proxy queue; a XeSS session's context lives on the native device) —
    // the --gpu rule: you can't toggle into an upscaler the session didn't
    // wire. dlss_on/xess_on are the CPU arm's live flags and may be toggled
    // off without unwiring the session.
    let dxr_rr_avail = gpu.dlss_ready();
    let dxr_xess_avail = gpu.xess_ready();
    // Both FSR kinds compose on the DXR pipeline: FSR3 via the XeSS feed
    // trio, FSR4-RR via the nine-plane feed.
    let dxr_fsr3_avail = gpu.fsr_flavor() == Some(fsr::Flavor::Fsr3);
    let dxr_fsr4_avail = gpu.fsr_flavor() == Some(fsr::Flavor::Fsr4Rr);
    // The DXR trace res: locked into the wired upscaler's range (the --gpu
    // contract — DxrGpu's buffers are sized once, no DRS), window-res when
    // plain. Computed once so the eager --dxr init and the lazy F init
    // build the pipeline at the same res.
    let (dxw, dxh) = if dxr_rr_avail {
        locked_render_res(opts.lock_scale.unwrap_or(2.0 / 3.0), dlss_range, (w, h), (drw, drh), false)
    } else if dxr_xess_avail {
        locked_render_res(opts.lock_scale.unwrap_or(2.0 / 3.0), xess_range, (w, h), (w, h), true)
    } else if dxr_fsr3_avail || dxr_fsr4_avail {
        locked_render_res(opts.lock_scale.unwrap_or(2.0 / 3.0), fsr_range, (w, h), (w, h), true)
    } else {
        (w, h)
    };
    // Upscaler sub-mode per-frame state, the gpu_up_* contract: free-running
    // frame index, the previous frame's camera (its own MV state, like
    // dlss_prev/gpu_prev_cam), the history-reset latch (discontinuities,
    // never motion), and the sub-mode itself (set at each F enable).
    let mut dxr_up = GpuUp::Plain;
    let mut dxr_up_idx: u32 = 0;
    let mut dxr_prev_cam: Option<Camera> = None;
    let mut dxr_reset = true;
    if want_dxr && !dxr_failed && !gpu_trace {
        let compose = dxr_rr_avail || dxr_xess_avail || dxr_fsr3_avail || dxr_fsr4_avail;
        if compose && opts.lock_scale.is_none() {
            eprintln!("dxr: dynamic render res is unsupported on the DXR pipeline; locking at quality (2/3)");
        }
        match gpu::dxc::Dxc::load(&opts.dxc_path).and_then(|dxc| {
            gpu.init_dxr(&dxc, scene, bvh, dxw as u32, dxh as u32, compose, opts.gpu_debug, opts.bc7)
        }) {
            Ok(()) => {
                dxr_on = true;
                dxr_up = if dxr_rr_avail {
                    GpuUp::Rr
                } else if dxr_xess_avail {
                    GpuUp::Xess
                } else if dxr_fsr4_avail {
                    GpuUp::Fsr4
                } else if dxr_fsr3_avail {
                    GpuUp::Fsr3
                } else {
                    GpuUp::Plain
                };
                // Restore the user's plain toggle (G/X/K inside the DXR arm).
                if p0.map_or(false, |p| p.dxr_up_plain) {
                    dxr_up = GpuUp::Plain;
                }
                // The CPU-side denoisers can't run under the DXR arm; the
                // CPU upscalers stay WIRED (fsr_on included — every FSR kind
                // composes) — the DXR arm presents through the same session
                // contexts, and F-off resumes them intact.
                if oidn_on {
                    oidn_on = false;
                    eprintln!("oidn: OFF (--dxr; N re-enables)");
                }
                if nppd_on {
                    nppd_on = false;
                    nppd_prev = None;
                    eprintln!("nppd: OFF (--dxr; J re-enables)");
                }
                eprintln!(
                    "dxr: DispatchRays pipeline ON at {dxw}x{dxh}{} (F toggles CPU <-> DXR)",
                    match dxr_up {
                        GpuUp::Rr => format!(" -> DLSS-RR {w}x{h} (G toggles plain)"),
                        GpuUp::Xess => format!(" -> XeSS-SR {w}x{h} (X toggles plain)"),
                        GpuUp::Fsr4 => format!(" -> FSR4-RR {w}x{h} (K toggles plain)"),
                        GpuUp::Fsr3 => format!(" -> FSR3 {w}x{h} (K toggles plain)"),
                        GpuUp::Plain => String::new(),
                    }
                );
            }
            Err(e) => {
                eprintln!("dxr: falling back to CPU tracing — {e}");
                dxr_failed = true;
            }
        }
    }
    let mut prev_budget = false;
    // Depth cap that fully resolves the screen (tiles reach LEAF_TILE): 7 at 1024.
    let depth_full: f32 = ((w.max(h) as f32) / render::LEAF_TILE as f32).log2().ceil();
    // Fractional depth-cap estimate for budget frames. Mid-range prior: one
    // slightly-coarse first frame beats a hitch. Deliberately not reset when
    // the camera stops — the last value is the best prior for the same
    // neighborhood, and the controller corrects within a frame on resume.
    // A resize re-entry keeps the last estimate — cost scales with area,
    // but the controller corrects within a frame either way.
    let mut depth_est: f32 = p0.map_or(4.0, |p| p.depth_est);
    let mut last_title = Instant::now();
    let mut last_stats = Instant::now();
    // Presented-frames-per-second, recomputed once per second (frame ms in
    // the title is render time only; this is the actual present rate).
    let mut fps = 0.0f64;
    let mut fps_frames = 0u32;
    let mut fps_t = Instant::now();
    let mut last_ms = 0.0f64;
    let mut shot = p0.map_or(0u32, |p| p.shot);
    // Debounced window-resize commit: SizeChanged events (a stream during a
    // drag, one for maximize/F11) arm the timer; the session exits for a
    // rebuild only once the size has been quiet for RESIZE_SETTLE_MS.
    // Until then frames keep presenting into the old-size swapchain and
    // DWM stretches them (DXGI_SCALING_STRETCH) — momentarily soft, never
    // broken. A minimized window reports a 0 dimension and never commits.
    const RESIZE_SETTLE_MS: u128 = 250;
    let mut pending_resize: Option<Instant> = None;

    // Init is done and frames start now, so the integrator may fly again (it
    // spawned paused, and a resize re-entry paused it). Everything above this
    // line — kernel compile, scene upload, BLAS build — presented nothing.
    // No baseline re-sync is needed: a paused span integrates nothing, so the
    // pose `prev_snap` captured at init is still the shared camera's pose.
    fly.resume();

    let end = loop {
        let now = Instant::now();

        let edges = inp.poll();
        if edges.quit {
            break SessionEnd::Quit;
        }
        if edges.toggle_fullscreen {
            // Borderless desktop fullscreen (F11) — the resulting
            // SizeChanged flows through the same debounce below.
            let to = if window.fullscreen_state() == sdl2::video::FullscreenType::Off {
                sdl2::video::FullscreenType::Desktop
            } else {
                sdl2::video::FullscreenType::Off
            };
            if let Err(e) = window.set_fullscreen(to) {
                eprintln!("fullscreen: {e}");
            }
        }
        if edges.size_changed.is_some() {
            pending_resize = Some(now);
        }
        if let Some(t0) = pending_resize {
            if (now - t0).as_millis() >= RESIZE_SETTLE_MS {
                pending_resize = None;
                // drawable_size at settle time is authoritative (physical
                // pixels — the permonitorv2 DPI hint makes logical equal
                // physical).
                let (dw, dh) = window.drawable_size();
                if dw > 0 && dh > 0 && (dw as usize, dh as usize) != (w, h) {
                    break SessionEnd::Resize(dw, dh);
                }
            }
        }
        // One pose snapshot per iteration: everything this frame does (trace,
        // MVs, prev-camera captures, verify) reads this copy, so trace pose ==
        // MV pose == prev-capture pose even while the integrator keeps
        // flying. `moved` = the integrator wrote between frames (bit compare;
        // taps shorter than a frame land as exactly one moved frame).
        cam = fly.snapshot();
        let moved = cam != prev_snap;
        prev_snap = cam;

        // GPU-resident tracing (--gpu): a self-contained arm — every frame is
        // traced (and in upscaler sub-modes fed + upscaled), tonemapped, and
        // presented on the GPU; the CPU's only jobs are input, the frame
        // counter, and ~300 bytes of constants. The CPU mode machinery below
        // (budget frames, OIDN/DLSS/XeSS, overlay) deliberately doesn't run.
        if gpu_trace {
            if let Some(p) = edges.quality {
                preset = p;
                frame = 0;
                gpu_reset = true; // noise statistics change across presets
                if gpu_up == GpuUp::Plain {
                    eprintln!("quality preset {preset}");
                } else {
                    eprintln!("quality preset {preset} (upscaler sub-mode traces 1-spp; presets apply in plain)");
                }
            }
            if edges.toggle_hybrid {
                hybrid = !hybrid;
                frame = 0;
                gpu_reset = true;
                eprintln!("gpu: {}", if hybrid { "hybrid (wavefront quadtree)" } else { "plain (per-pixel reference)" });
            }
            if edges.toggle_xess {
                if gpu_xess_avail {
                    gpu_up = if gpu_up == GpuUp::Xess { GpuUp::Plain } else { GpuUp::Xess };
                    frame = 0;
                    gpu_reset = true;
                    gpu_prev_cam = None;
                    eprintln!(
                        "gpu: XeSS-SR {}",
                        if gpu_up == GpuUp::Xess { "ON" } else { "OFF (plain present)" }
                    );
                } else {
                    eprintln!("gpu: XeSS not wired in this session (start with --gpu --xess)");
                }
            }
            if edges.toggle_dlss {
                if gpu_rr_avail {
                    gpu_up = if gpu_up == GpuUp::Rr { GpuUp::Plain } else { GpuUp::Rr };
                    frame = 0;
                    gpu_reset = true;
                    gpu_prev_cam = None;
                    eprintln!(
                        "gpu: DLSS-RR {}",
                        if gpu_up == GpuUp::Rr { "ON" } else { "OFF (plain present)" }
                    );
                } else {
                    eprintln!("gpu: DLSS-RR not wired in this session");
                }
            }
            if edges.toggle_fsr {
                if let Some(kind) = gpu_fsr_kind {
                    gpu_up = if gpu_up == kind { GpuUp::Plain } else { kind };
                    frame = 0;
                    gpu_reset = true;
                    gpu_prev_cam = None;
                    eprintln!(
                        "gpu: {} {}",
                        if kind == GpuUp::Fsr4 { "FSR4-RR" } else { "FSR3" },
                        if gpu_up == kind { "ON" } else { "OFF (plain present)" }
                    );
                } else {
                    eprintln!("gpu: FSR not wired in this session");
                }
            }
            if edges.toggle_bounce {
                if gpu_up != GpuUp::Plain {
                    eprintln!("gpu: hemi bounces unavailable in the upscaler sub-mode (still-frame feature)");
                } else {
                    bounce_mode = (bounce_mode + 1) % 3;
                    frame = 0;
                    eprintln!(
                        "gpu: hemisphere frustum bounces: {}",
                        ["OFF", "AO (still frames)", "GI (still frames)"][bounce_mode as usize]
                    );
                }
            }
            if edges.toggle_nppd {
                if gpu_nppd_avail {
                    if gpu_up == GpuUp::Xess {
                        gpu_nppd_on = !gpu_nppd_on;
                        frame = 0;
                        // Noise statistics change; the recurrent state resets
                        // via nppd_state_valid (J-off frames clear it).
                        gpu_reset = true;
                        eprintln!(
                            "gpu: NPPD pre-upscale denoise {}",
                            if gpu_nppd_on { "ON" } else { "OFF" }
                        );
                    } else {
                        eprintln!("gpu: NPPD rides the XeSS composition (X back on first)");
                    }
                } else {
                    eprintln!("gpu: NPPD not wired in this session (start with --gpu --nppd)");
                }
            }
            // CPU-renderer-only keys: consume the edges with a note instead
            // of silently swallowing them (CLAUDE.md: "T/O/N/M print notes").
            if edges.toggle_dynamic {
                eprintln!("gpu: dynamic resolution is a CPU-renderer feature; the GPU render res is locked per session (--lock-res)");
            }
            if edges.toggle_overlay {
                eprintln!("gpu: the quadtree overlay is CPU-only");
            }
            if edges.toggle_oidn {
                eprintln!("gpu: OIDN denoising is CPU-only; unavailable under --gpu");
            }
            if edges.toggle_temporal {
                eprintln!("gpu: the OIDN reprojection history is CPU-only; unavailable under --gpu");
            }
            if edges.toggle_dxr {
                eprintln!("gpu: the DXR DispatchRays pipeline is a CPU-session mode; unavailable under --gpu");
            }
            if moved {
                frame = 0;
            }
            let base_q = Quality::preset(preset);
            // C key first: verify clobbers accum/tbuf/info, so the frame
            // presented below must be a frame-0 store — handling it after
            // the present would let the next frame ADD onto the verify image.
            // The basis is the SESSION render res (the tracer's buffers are
            // sized to it), not the window.
            if edges.verify {
                eprintln!("gpu: verifying current view (wavefront vs reference, on-GPU)...");
                match gpu.verify_trace(&cam.basis(grw, grh), base_q) {
                    Ok(report) => eprintln!("{report}"),
                    Err(e) => eprintln!("gpu: verify failed to run: {e}"),
                }
                frame = 0;
                gpu_reset = true;
            }
            let t = Instant::now();
            if gpu_up != GpuUp::Plain {
                // The upscaler contract (the CPU arms'): every frame is a
                // fresh jittered 1-spp full-depth frame at the locked render
                // res; the upscaler is the only temporal integrator. One
                // camera pose feeds every consumer — the shader MVs
                // (prev_cam basis) and, in RR mode, fc's prev matrices — so
                // they can never disagree.
                let jit = dlss::jitter_for(gpu_up_idx);
                let p = gpu::trace::FrameParams {
                    cam: cam.basis(grw, grh),
                    frame: gpu_up_idx,
                    accumulate: false,
                    jitter: false,
                    frame_jitter: Some(jit),
                    prev_cam: gpu_prev_cam.map(|c| c.basis(grw, grh)),
                    q: Quality::upscaler_1spp(),
                    verify: false,
                };
                let presented = if gpu_up == GpuUp::Xess {
                    gpu.present_trace_xess(&p, hybrid, jit, gpu_reset, gpu_nppd_on)
                } else {
                    // The prev matrices are recomputed from the stored
                    // camera (pure math, fixed res: identical to last
                    // frame's).
                    let mats = dlss::cam_matrices(&cam, grw, grh, dlss_near, dlss_far);
                    let prev_mats = gpu_prev_cam
                        .map(|c| dlss::cam_matrices(&c, grw, grh, dlss_near, dlss_far));
                    let fc = dlss::frame_constants(
                        &cam,
                        &mats,
                        prev_mats.as_ref(),
                        jit,
                        gpu_reset,
                        dlss_near,
                        dlss_far,
                        grw,
                        grh,
                    );
                    match gpu_up {
                        // frameTimeDelta is the PREVIOUS frame's render time
                        // (the DXR arm's contract); the desc clamps it into
                        // [0.1, 200].
                        GpuUp::Fsr3 => gpu.present_trace_fsr3(&p, hybrid, &fc, last_ms as f32),
                        GpuUp::Fsr4 => gpu.present_trace_fsr_rr(
                            &p,
                            hybrid,
                            &fc,
                            gpu_prev_cam.map(|c| c.pos),
                            gpu_up_idx,
                            last_ms as f32,
                        ),
                        _ => gpu.present_trace_rr(&p, hybrid, &fc, gpu_up_idx),
                    }
                };
                match presented {
                    Ok(()) => {
                        last_ms = t.elapsed().as_secs_f64() * 1000.0;
                        gpu_prev_cam = Some(cam);
                        gpu_reset = false;
                        gpu_up_idx = gpu_up_idx.wrapping_add(1);
                    }
                    Err(e) => {
                        if gpu_up == GpuUp::Xess && gpu_nppd_on {
                            // Shed the NPPD stage first — XeSS itself may be
                            // fine (the run/split path is NPPD-only).
                            eprintln!("gpu: NPPD present failed ({e}); NPPD OFF (J to retry)");
                            gpu_nppd_on = false;
                            gpu_reset = true;
                        } else {
                            let (name, key) = match gpu_up {
                                GpuUp::Xess => ("XeSS", 'X'),
                                GpuUp::Fsr3 => ("FSR3", 'K'),
                                GpuUp::Fsr4 => ("FSR4-RR", 'K'),
                                _ => ("DLSS-RR", 'G'),
                            };
                            eprintln!("gpu: {name} present failed ({e}); presenting plain ({key} to retry)");
                            gpu_up = GpuUp::Plain;
                        }
                        frame = 0;
                    }
                }
            } else {
                let mut q = if moved { base_q.while_moving() } else { base_q };
                if !moved && hybrid {
                    // Hemi tiers are a still-frame, wavefront-path feature (the
                    // reference kernel keeps the sampled-ambient path).
                    q.fb.ao = bounce_mode == 1;
                    q.fb.gi = bounce_mode == 2;
                }
                let p = gpu::trace::FrameParams {
                    cam: cam.basis(grw, grh),
                    frame,
                    accumulate: true,
                    jitter: frame > 0,
                    frame_jitter: None,
                    prev_cam: None,
                    q,
                    verify: false,
                };
                if frame < MAX_SAMPLES {
                    if let Err(e) = gpu.present_trace(&p, frame + 1, hybrid) {
                        eprintln!("gpu: present failed: {e}");
                    }
                    // Moving frames stay at frame 0: every one is a fresh
                    // store, and the first still frame then re-stores at full
                    // quality instead of adding onto a while_moving()-quality
                    // sample #0.
                    if !moved {
                        frame += 1;
                    }
                } else if let Err(e) = gpu.present_hold() {
                    // Converged: re-present the resolved image without
                    // tracing. Re-adding the pinned-seed sample while resolve
                    // divides by the pinned count would brighten the image
                    // without bound.
                    eprintln!("gpu: present failed: {e}");
                }
            }
            // Stability meter (FRUSTRACER_STAB=1): the numeric dancing
            // detector — hold the camera still and a healthy upscaled output
            // trends toward ~0 (a wrong jitter sign or MV polarity holds a
            // high mean). Same meter as the CPU upscaler arms.
            if stab_on && gpu_up != GpuUp::Plain {
                stab_n = stab_n.wrapping_add(1);
                if stab_n % 15 == 0 {
                    let cap = match gpu_up {
                        GpuUp::Xess => gpu.read_xess_output(),
                        GpuUp::Fsr3 | GpuUp::Fsr4 => gpu.read_fsr_output(),
                        _ => gpu.read_rr_output(),
                    };
                    if let Ok(px) = cap {
                        if let Some(prev) = &stab_prev {
                            if prev.len() == px.len() {
                                let sum: u64 = px
                                    .iter()
                                    .zip(prev)
                                    .map(|(a, b)| {
                                        let d = |s: u32| {
                                            ((a >> s) & 0xff).abs_diff((b >> s) & 0xff) as u64
                                        };
                                        d(16) + d(8) + d(0)
                                    })
                                    .sum();
                                eprintln!(
                                    "stab: mean |Δ| {:.2}/255 over 15 frames (window-res output; render {grw}x{grh})",
                                    sum as f64 / (px.len() * 3) as f64,
                                );
                            }
                        }
                        stab_prev = Some(px);
                    }
                }
            }
            if edges.screenshot {
                // Upscaler sub-modes: the presented image is the upscaler's
                // window-res output; plain: the tracer's hdr (render-res in
                // an upscaler SESSION — dims come back with the pixels).
                let cap = match gpu_up {
                    GpuUp::Xess => gpu.read_xess_output().map(|px| (px, w, h)),
                    GpuUp::Rr => gpu.read_rr_output().map(|px| (px, w, h)),
                    GpuUp::Fsr3 | GpuUp::Fsr4 => gpu.read_fsr_output().map(|px| (px, w, h)),
                    GpuUp::Plain => gpu.read_trace_output(),
                };
                match cap {
                    Ok((px, sw, sh)) => {
                        let name = format!("screenshot_{shot}.png");
                        save_png(&name, &px, sw, sh);
                        eprintln!("saved {name}");
                        shot += 1;
                    }
                    Err(e) => eprintln!("screenshot: GPU readback failed ({e})"),
                }
            }
            fps_frames += 1;
            if (now - fps_t).as_secs_f64() >= 0.5 {
                fps = fps_frames as f64 / (now - fps_t).as_secs_f64();
                fps_frames = 0;
                fps_t = now;
                let mode = match gpu_up {
                    GpuUp::Plain => format!("{}x{}", grw, grh),
                    GpuUp::Rr => format!("RR {}x{} -> {}x{}", grw, grh, w, h),
                    GpuUp::Fsr4 => format!("FSR4-RR {}x{} -> {}x{}", grw, grh, w, h),
                    GpuUp::Fsr3 => format!("FSR3 {}x{} -> {}x{}", grw, grh, w, h),
                    GpuUp::Xess => format!(
                        "XeSS{} {}x{} -> {}x{}",
                        if gpu_nppd_on { "+NPPD(pre)" } else { "" },
                        grw,
                        grh,
                        w,
                        h
                    ),
                };
                let spp = if gpu_up == GpuUp::Plain {
                    format!(
                        " | {} spp{}",
                        frame.min(MAX_SAMPLES),
                        if frame >= MAX_SAMPLES { " | converged" } else { "" }
                    )
                } else {
                    " | 1 spp".to_string()
                };
                let _ = window.set_title(&format!(
                    "frustracer | {:.0} fps | GPU {} | {} | quality {}{}",
                    fps,
                    if hybrid { "hybrid" } else { "plain" },
                    mode,
                    preset,
                    spp,
                ));
            }
            continue;
        }
        if edges.toggle_hybrid {
            hybrid = !hybrid;
            frame = 0;
            dlss_reset = true; // noise statistics change across the toggle
        }
        if edges.toggle_dynamic {
            if (dlss_on || fsr_on || xess_on) && opts.lock_scale.is_some() {
                let (lw, lh) = if dlss_on {
                    (drw, drh)
                } else if fsr_on {
                    fsr_lock.unwrap_or((drw, drh))
                } else {
                    xess_lock.unwrap_or((drw, drh))
                };
                eprintln!(
                    "render res locked at {}x{} by --lock-res (CLI-only; restart with `--lock-res dynamic` for DRS)",
                    lw, lh
                );
            } else if dlss_on {
                eprintln!(
                    "dynamic-res in DLSS mode is {}",
                    if dlss_drs {
                        "always on (the scale controller drives it, step-wise)"
                    } else {
                        "unavailable (driver reported no DRS range)"
                    }
                );
            } else if fsr_on {
                eprintln!("dynamic-res is always on in FSR mode (the scale controller drives it)");
            } else if xess_on {
                eprintln!("dynamic-res is always on in XeSS mode (the scale controller drives it)");
            } else {
                dynamic = !dynamic;
                frame = 0;
            }
        }
        if edges.toggle_overlay {
            if dlss_on || xess_on {
                eprintln!("overlay unavailable in DLSS/XeSS mode (lives in the CPU resolve)");
            } else {
                overlay_on = !overlay_on;
            }
        }
        if edges.toggle_gpu_tone {
            gpu_tonemap = !gpu_tonemap;
            let note = if oidn_on { " (no effect in OIDN mode — presents via the CPU resolve)" } else { "" };
            eprintln!("tonemap: {}{note}", if gpu_tonemap { "GPU" } else { "CPU" });
        }
        if edges.toggle_bounce {
            bounce_mode = (bounce_mode + 1) % 3;
            frame = 0;
            eprintln!(
                "hemisphere frustum bounces: {}",
                ["OFF", "AO (still frames)", "GI (still frames)"][bounce_mode as usize]
            );
        }
        if edges.toggle_dlss {
            if dxr_on {
                // Inside DXR mode G toggles the wired upscaler vs plain
                // DXR — the --gpu G semantics; the CPU DLSS state is
                // untouched (F-off resumes it).
                if dxr_rr_avail {
                    dxr_up = if dxr_up == GpuUp::Rr { GpuUp::Plain } else { GpuUp::Rr };
                    frame = 0;
                    dxr_reset = true;
                    dxr_prev_cam = None;
                    eprintln!(
                        "dxr: DLSS-RR {}",
                        if dxr_up == GpuUp::Rr { "ON" } else { "OFF (plain present)" }
                    );
                } else {
                    eprintln!("dxr: DLSS-RR not wired in this session");
                }
            } else if gpu.dlss_ready() {
                dlss_on = !dlss_on;
                frame = 0;
                dlss_reset = true;
                dlss_prev = None;
                // Fresh limiter: a re-enabled session adopts the controller's
                // target immediately instead of dwelling on the stale res.
                dlss_lim = xess::StepLimiter::new(xess::RAMP_FRAMES);
                dlss_ep = (0, 0);
                if dlss_on && oidn_on {
                    oidn_on = false;
                    eprintln!("oidn: OFF (DLSS enabled)");
                }
                if dlss_on && nppd_on {
                    nppd_on = false;
                    nppd_prev = None;
                    eprintln!("nppd: OFF (DLSS enabled)");
                }
                if dlss_on && xess_on {
                    // Structurally unreachable (XeSS sessions never init SL),
                    // kept for the day both live on one pipeline.
                    xess_on = false;
                    xess_prev = None;
                    eprintln!("xess: OFF (DLSS enabled)");
                }
                eprintln!("dlss: Ray Reconstruction {}", if dlss_on { "ON" } else { "OFF" });
            } else {
                eprintln!("dlss: not wired in this session (the chain selected another level; restart with --dlss)");
            }
        }
        if edges.toggle_xess {
            if dxr_on {
                // The X twin of the DXR-mode G toggle.
                if dxr_xess_avail {
                    dxr_up = if dxr_up == GpuUp::Xess { GpuUp::Plain } else { GpuUp::Xess };
                    frame = 0;
                    dxr_reset = true;
                    dxr_prev_cam = None;
                    eprintln!(
                        "dxr: XeSS {}",
                        if dxr_up == GpuUp::Xess { "ON" } else { "OFF (plain present)" }
                    );
                } else {
                    eprintln!("dxr: XeSS not wired in this session");
                }
            } else if gpu.xess_ready() {
                xess_on = !xess_on;
                frame = 0;
                xess_reset = true;
                xess_prev = None;
                // Fresh limiter: a re-enabled session adopts the controller's
                // target immediately instead of dwelling on the stale res.
                xess_lim = xess::StepLimiter::new(xess::RAMP_FRAMES);
                xess_ep = (0, 0);
                if xess_on && dlss_on {
                    dlss_on = false;
                    dlss_prev = None;
                    dlss_reset = true;
                    eprintln!("dlss: Ray Reconstruction OFF (XeSS enabled)");
                }
                if xess_on && oidn_on {
                    // Plain-mode OIDN is unreachable while XeSS presents (the
                    // xess_on present arm wins); clear it so M and the stats
                    // line don't keep acting on a denoiser that isn't running.
                    oidn_on = false;
                    eprintln!("oidn: OFF (XeSS enabled; N cycles the XeSS placement)");
                }
                if xess_on && nppd_on {
                    nppd_on = false;
                    nppd_prev = None;
                    eprintln!("nppd: OFF (XeSS enabled)");
                }
                eprintln!(
                    "xess: {}",
                    if xess_on {
                        match xess_oidn {
                            XessOidn::Off => "ON",
                            XessOidn::Pre => "ON (OIDN pre-denoise)",
                            XessOidn::Post => "ON (OIDN post-denoise)",
                        }
                    } else {
                        "OFF"
                    }
                );
            } else {
                eprintln!("xess: not wired in this session (restart with --xess and the SDK DLL on disk)");
            }
        }
        if edges.toggle_fsr {
            if dxr_on {
                // The K twin of the DXR-mode G/X toggles: wired-FSR <-> plain.
                let kind = if dxr_fsr4_avail {
                    Some(GpuUp::Fsr4)
                } else if dxr_fsr3_avail {
                    Some(GpuUp::Fsr3)
                } else {
                    None
                };
                if let Some(kind) = kind {
                    dxr_up = if dxr_up == kind { GpuUp::Plain } else { kind };
                    frame = 0;
                    dxr_reset = true;
                    dxr_prev_cam = None;
                    eprintln!(
                        "dxr: {} {}",
                        if kind == GpuUp::Fsr4 { "FSR4-RR" } else { "FSR3" },
                        if dxr_up == kind { "ON" } else { "OFF (plain present)" }
                    );
                } else {
                    eprintln!("dxr: FSR not wired in this session");
                }
            } else if gpu.fsr_ready() {
                fsr_on = !fsr_on;
                frame = 0;
                fsr_reset = true;
                fsr_prev = None;
                // Fresh limiter: a re-enabled session adopts the controller's
                // target immediately instead of dwelling on the stale res.
                fsr_lim = xess::StepLimiter::new(xess::RAMP_FRAMES);
                fsr_ep = (0, 0);
                // DLSS/XeSS never coexist with a live FSR session (one wired
                // upscaler per session) — no cross-disable needed. OIDN/NPPD
                // and DXR are toggleable live, so they do need it (the chain
                // can wire FSR in an --oidn/--nppd session where the startup
                // yield ran the other way).
                if fsr_on && dxr_on {
                    dxr_on = false;
                    eprintln!("dxr: OFF (FSR enabled)");
                }
                if fsr_on && oidn_on {
                    oidn_on = false;
                    eprintln!("oidn: OFF (FSR enabled; N re-enables)");
                }
                if fsr_on && nppd_on {
                    nppd_on = false;
                    nppd_prev = None;
                    eprintln!("nppd: OFF (FSR enabled; J re-enables)");
                }
                eprintln!("fsr: {} {}", fsr_label, if fsr_on { "ON" } else { "OFF" });
            } else {
                eprintln!("fsr: not wired in this session (restart with --fsr/--fsr3 and the FidelityFX DLLs on disk)");
            }
        }
        if edges.toggle_oidn {
            if dxr_on {
                // CPU-side denoisers never run under the DXR arm — refuse
                // instead of silently mutating latent CPU-mode state.
                eprintln!("oidn: a CPU-mode denoiser — unavailable under the DXR pipeline (F toggles DXR off first)");
            } else if fsr_on {
                // Enabling plain-mode OIDN under a live FSR present arm would
                // only allocate its window-res G-buffers and lie in the logs
                // (the FSR arm wins the mode arbitration) — refuse instead.
                eprintln!("oidn: the FSR present arm owns the frame (K toggles FSR off first)");
            } else if xess_on {
                // XeSS mode: N cycles the OIDN placement (off → pre → post).
                let next = match xess_oidn {
                    XessOidn::Off => XessOidn::Pre,
                    XessOidn::Pre => XessOidn::Post,
                    XessOidn::Post => XessOidn::Off,
                };
                if next == XessOidn::Off {
                    xess_oidn = next;
                    frame = 0;
                    eprintln!("oidn: OFF (raw XeSS)");
                } else if oidn_failed {
                    eprintln!("oidn: unavailable (earlier init failed; restart with --oidn-path to retry)");
                } else if oidn_try_enable(&mut oidn_ctx, &mut oidn_gbufs, &mut oidn_hist) {
                    xess_oidn = next;
                    frame = 0;
                    if xess_nppd {
                        xess_nppd = false;
                        eprintln!("nppd: OFF (OIDN takes the XeSS pre-denoise slot)");
                    }
                    eprintln!(
                        "oidn: {} the XeSS upscale (N cycles off → pre → post)",
                        if next == XessOidn::Pre {
                            "PRE-denoise at render res, before"
                        } else {
                            "POST-denoise at window res, after"
                        }
                    );
                } else {
                    oidn_failed = true;
                }
            } else if oidn_on {
                oidn_on = false;
                frame = 0;
                eprintln!("oidn: OFF");
            } else if oidn_failed {
                eprintln!("oidn: unavailable (earlier init failed; restart with --oidn-path to retry)");
            } else if oidn_try_enable(&mut oidn_ctx, &mut oidn_gbufs, &mut oidn_hist) {
                oidn_on = true;
                frame = 0;
                if dlss_on {
                    dlss_on = false;
                    dlss_prev = None;
                    dlss_reset = true;
                    eprintln!("dlss: Ray Reconstruction OFF (OIDN enabled)");
                }
                if nppd_on {
                    nppd_on = false;
                    nppd_prev = None;
                    eprintln!("nppd: OFF (OIDN enabled)");
                }
                eprintln!(
                    "oidn: ON{}, temporal reprojection {} (M toggles)",
                    if xess_on { " (XeSS pre-denoise at the dynamic render res)" } else { "" },
                    if oidn_temporal { "ON" } else { "OFF" }
                );
            } else {
                oidn_failed = true;
            }
        }
        if edges.toggle_nppd {
            if dxr_on {
                // Same refusal as N: CPU-mode denoisers don't run here.
                eprintln!("nppd: a CPU-mode denoiser — unavailable under the DXR pipeline (F toggles DXR off first)");
            } else if fsr_on {
                // Same refusal as N: the FSR arm wins the mode arbitration
                // (flavor-neutral — the 3.1 flavor has no denoiser at all).
                eprintln!("nppd: the FSR present arm owns the frame (K toggles FSR off first)");
            } else if xess_on {
                // XeSS mode: J toggles the NPPD pre-upscale placement
                // (mutually exclusive with the OIDN N-cycle's pre/post).
                if xess_nppd {
                    xess_nppd = false;
                    frame = 0;
                    eprintln!("nppd: OFF (raw XeSS input)");
                } else if nppd_failed {
                    eprintln!(
                        "nppd: unavailable (earlier init failed; restart with --nppd-path/--nppd-model to retry)"
                    );
                } else if nppd_try_enable(
                    &mut nppd_ctx,
                    &mut nppd_gbufs,
                    false,
                    xess_lock.unwrap_or((w, h)),
                ) {
                    xess_nppd = true;
                    frame = 0;
                    if xess_oidn != XessOidn::Off {
                        xess_oidn = XessOidn::Off;
                        eprintln!("oidn: OFF (NPPD takes the XeSS pre-denoise slot)");
                    }
                    nppd_drs_note(opts.lock_scale);
                    let c = nppd_ctx.as_ref().unwrap();
                    eprintln!(
                        "nppd: PRE-upscale denoise ON ({} | onnxruntime {})",
                        c.device_desc, c.ort_version
                    );
                } else {
                    nppd_failed = true;
                }
            } else if nppd_on {
                nppd_on = false;
                nppd_prev = None;
                frame = 0;
                eprintln!("nppd: OFF");
            } else if nppd_failed {
                eprintln!(
                    "nppd: unavailable (earlier init failed; restart with --nppd-path/--nppd-model to retry)"
                );
            } else if nppd_try_enable(&mut nppd_ctx, &mut nppd_gbufs, true, (w, h)) {
                nppd_on = true;
                nppd_prev = None;
                frame = 0;
                if dlss_on {
                    dlss_on = false;
                    dlss_prev = None;
                    dlss_reset = true;
                    eprintln!("dlss: Ray Reconstruction OFF (NPPD enabled)");
                }
                if xess_on {
                    xess_on = false;
                    xess_prev = None;
                    xess_reset = true;
                    eprintln!("xess: OFF (NPPD enabled)");
                }
                if oidn_on {
                    oidn_on = false;
                    eprintln!("oidn: OFF (NPPD enabled)");
                }
                let c = nppd_ctx.as_ref().unwrap();
                eprintln!("nppd: neural denoising ON ({} | onnxruntime {})", c.device_desc, c.ort_version);
            } else {
                nppd_failed = true;
            }
        }
        if edges.toggle_dxr {
            if dxr_on {
                dxr_on = false;
                frame = 0;
                // The CPU upscalers resume with their own contracts; their
                // histories just watched DXR frames at a different res —
                // declare the discontinuity (harmless when they're off).
                dlss_prev = None;
                dlss_reset = true;
                xess_prev = None;
                xess_reset = true;
                fsr_prev = None;
                fsr_reset = true;
                eprintln!("dxr: OFF (CPU tracing)");
            } else if dxr_failed {
                eprintln!("dxr: unavailable (earlier init failed; restart with --dxc-path to retry)");
            } else {
                let compose = dxr_rr_avail || dxr_xess_avail || dxr_fsr3_avail || dxr_fsr4_avail;
                if compose && opts.lock_scale.is_none() {
                    eprintln!("dxr: dynamic render res is unsupported on the DXR pipeline; locking at quality (2/3)");
                }
                match gpu::dxc::Dxc::load(&opts.dxc_path).and_then(|dxc| {
                    gpu.init_dxr(&dxc, scene, bvh, dxw as u32, dxh as u32, compose, opts.gpu_debug, opts.bc7)
                }) {
                    Ok(()) => {
                        dxr_on = true;
                        frame = 0;
                        // Every F enable restores the session default
                        // sub-mode (a G/X/K plain-toggle doesn't outlive it).
                        dxr_up = if dxr_rr_avail {
                            GpuUp::Rr
                        } else if dxr_xess_avail {
                            GpuUp::Xess
                        } else if dxr_fsr4_avail {
                            GpuUp::Fsr4
                        } else if dxr_fsr3_avail {
                            GpuUp::Fsr3
                        } else {
                            GpuUp::Plain
                        };
                        dxr_reset = true;
                        dxr_prev_cam = None;
                        // CPU-side denoisers can't run under the DXR arm;
                        // the CPU upscalers stay wired (fsr_on included —
                        // every FSR kind composes) — the arm presents
                        // through the same session contexts, and F-off
                        // resumes them intact.
                        if oidn_on {
                            oidn_on = false;
                            eprintln!("oidn: OFF (DXR enabled)");
                        }
                        if nppd_on {
                            nppd_on = false;
                            nppd_prev = None;
                            eprintln!("nppd: OFF (DXR enabled)");
                        }
                        eprintln!(
                            "dxr: DispatchRays pipeline ON at {dxw}x{dxh}{} (F toggles CPU <-> DXR)",
                            match dxr_up {
                                GpuUp::Rr => format!(" -> DLSS-RR {w}x{h} (G toggles plain)"),
                                GpuUp::Xess => format!(" -> XeSS-SR {w}x{h} (X toggles plain)"),
                                GpuUp::Fsr4 => format!(" -> FSR4-RR {w}x{h} (K toggles plain)"),
                                GpuUp::Fsr3 => format!(" -> FSR3 {w}x{h} (K toggles plain)"),
                                GpuUp::Plain => String::new(),
                            }
                        );
                    }
                    Err(e) => {
                        eprintln!("dxr: unavailable — {e}");
                        dxr_failed = true;
                    }
                }
            }
        }
        // True only when M actually flipped oidn_temporal (not the Post/no-op
        // arms) — the single source for the reset predicates below, so they
        // can't drift from the handler's own condition.
        let mut temporal_flipped = false;
        if edges.toggle_temporal {
            if xess_on && xess_oidn == XessOidn::Post {
                eprintln!("oidn: no temporal history in POST placement (XeSS itself is the temporal integrator there)");
            } else if oidn_on || (xess_on && xess_oidn == XessOidn::Pre) {
                oidn_temporal = !oidn_temporal;
                temporal_flipped = true;
                // Required in BOTH directions: accum semantics flip between
                // "last 1-spp frame" (temporal) and "pure sum" (plain) — a
                // stale sample count would divide a 1-spp frame to near-black.
                frame = 0;
                eprintln!(
                    "oidn: temporal reprojection {}",
                    if oidn_temporal { "ON" } else { "OFF (plain accumulation)" }
                );
            } else {
                eprintln!("oidn: temporal toggle is OIDN-only (N enables OIDN, then M toggles)");
            }
        }
        if let Some(p) = edges.quality {
            preset = p;
            frame = 0;
            dlss_reset = true;
        }
        // Any XeSS frame after this point sends reset_history = 1: the
        // upscaler's accumulated history mixes shading statistics, so every
        // predicate that resets `frame`/the OIDN history also resets it —
        // EXCEPT camera motion, which is exactly what the temporal upscaler
        // exists to survive.
        if edges.toggle_hybrid
            || edges.toggle_bounce
            || edges.quality.is_some()
            || edges.toggle_xess
            || edges.toggle_oidn
            || edges.toggle_nppd
            || edges.toggle_dxr
            || temporal_flipped
        {
            xess_reset = true;
        }
        // The same reset contract for the FSR histories (Ray Regeneration's
        // temporal accumulation + FSR4's): every shading-semantics change,
        // never camera motion.
        if edges.toggle_hybrid
            || edges.toggle_bounce
            || edges.quality.is_some()
            || edges.toggle_fsr
            || temporal_flipped
        {
            fsr_reset = true;
        }
        if moved {
            frame = 0;
        }
        // Reprojection-history invalidation: any setting change that alters
        // shading or mode semantics drops the history; camera motion and the
        // budget↔normal transition deliberately do NOT (surviving motion is
        // the history's whole purpose; coarse budget pixels are handled
        // per-pixel by the KIND_COARSE rule). Over-invalidating on no-op
        // edges (e.g. T in DLSS mode) is accepted for the simple predicate.
        let hist_stale = edges.toggle_hybrid
            || edges.toggle_dynamic
            || edges.toggle_bounce
            || edges.quality.is_some()
            || edges.toggle_oidn
            || edges.toggle_dlss
            || edges.toggle_xess
            || edges.toggle_fsr
            || edges.toggle_nppd
            || edges.toggle_dxr
            || temporal_flipped;
        if hist_stale {
            if let Some(h) = &mut oidn_hist {
                h.invalidate();
            }
            // The NPPD recurrent state follows the same staleness predicate —
            // any shading/mode-semantics change, never camera motion
            // (surviving motion is exactly what the warped state is for).
            if let Some(c) = &mut nppd_ctx {
                c.reset_temporal();
            }
        }

        // DXR mode: a self-contained arm — the frame is traced by
        // DispatchRays and presented on the GPU; the CPU render machinery
        // below (budget frames, temporal ring, the denoiser modes)
        // deliberately doesn't run. In an upscaler sub-mode (the session
        // default when RR/XeSS is wired) the frame follows the --gpu
        // upscaler contract: a fresh jittered 1-spp DispatchRays frame at
        // the locked render res, fed to the upscaler on the same list, ONE
        // camera pose for every consumer, never idle. Plain sub-mode
        // mirrors the --gpu plain sub-mode: while_moving quality with
        // `frame` pinned at 0, accumulate + converge when still, re-present
        // the resolved image when converged.
        if dxr_on {
            let basis = cam.basis(w, h);
            // The temporal ring must not survive non-participating frames:
            // F-off resumes CPU tracing against whatever it still holds.
            tr.end(false, false, basis);
            if edges.verify {
                eprintln!("dxr: C verify is a CPU-tracer feature; --check-dxr gates this pipeline");
            }
            let t = Instant::now();
            if dxr_up != GpuUp::Plain {
                if edges.quality.is_some() {
                    // The generic handler already reset `frame`; the noise
                    // statistics change, so declare the discontinuity.
                    dxr_reset = true;
                    eprintln!("dxr: quality pinned at the 1-spp upscaler preset (plain DXR honors 1-3)");
                }
                let jit = dlss::jitter_for(dxr_up_idx);
                let p = gpu::trace::FrameParams {
                    cam: cam.basis(dxw, dxh),
                    frame: dxr_up_idx,
                    accumulate: false,
                    jitter: false,
                    frame_jitter: Some(jit),
                    prev_cam: dxr_prev_cam.map(|c| c.basis(dxw, dxh)),
                    q: Quality::upscaler_1spp(),
                    verify: false,
                };
                let presented = if dxr_up == GpuUp::Xess {
                    gpu.present_dxr_xess(&p, jit, dxr_reset)
                } else {
                    // The prev matrices are recomputed from the stored
                    // camera (pure math, fixed res: identical to last
                    // frame's).
                    let mats = dlss::cam_matrices(&cam, dxw, dxh, dlss_near, dlss_far);
                    let prev_mats =
                        dxr_prev_cam.map(|c| dlss::cam_matrices(&c, dxw, dxh, dlss_near, dlss_far));
                    let fc = dlss::frame_constants(
                        &cam,
                        &mats,
                        prev_mats.as_ref(),
                        jit,
                        dxr_reset,
                        dlss_near,
                        dlss_far,
                        dxw,
                        dxh,
                    );
                    match dxr_up {
                        GpuUp::Fsr3 => gpu.present_dxr_fsr3(&p, &fc, last_ms as f32),
                        GpuUp::Fsr4 => gpu.present_dxr_fsr_rr(
                            &p,
                            &fc,
                            dxr_prev_cam.map(|c| c.pos),
                            dxr_up_idx,
                            last_ms as f32,
                        ),
                        _ => gpu.present_dxr_rr(&p, &fc, dxr_up_idx),
                    }
                };
                match presented {
                    Ok(()) => {
                        last_ms = t.elapsed().as_secs_f64() * 1000.0;
                        dxr_prev_cam = Some(cam);
                        dxr_reset = false;
                        dxr_up_idx = dxr_up_idx.wrapping_add(1);
                    }
                    Err(e) => {
                        let (name, key) = match dxr_up {
                            GpuUp::Xess => ("XeSS", 'X'),
                            GpuUp::Fsr3 => ("FSR3", 'K'),
                            GpuUp::Fsr4 => ("FSR4-RR", 'K'),
                            _ => ("DLSS-RR", 'G'),
                        };
                        eprintln!("dxr: {name} present failed ({e}); presenting plain ({key} to retry)");
                        dxr_up = GpuUp::Plain;
                        frame = 0;
                    }
                }
            } else {
                let base_q = Quality::preset(preset);
                // Hemi (H) tiers are CPU/wavefront features; the DXR
                // closest-hit keeps the sampled-ambient path, so fb stays
                // OFF here.
                let q = if moved { base_q.while_moving() } else { base_q };
                // The basis is the SESSION trace res (DxrGpu's buffers and
                // DispatchRays dims are sized to it) — dxw x dxh, NOT the
                // window: a composed session toggled plain via G/X still
                // traces at the locked render res.
                let p = gpu::trace::FrameParams {
                    cam: cam.basis(dxw, dxh),
                    frame,
                    accumulate: true,
                    jitter: frame > 0,
                    frame_jitter: None,
                    prev_cam: None,
                    q,
                    verify: false,
                };
                if frame < MAX_SAMPLES {
                    match gpu.present_dxr(&p, frame + 1) {
                        Ok(()) => {
                            last_ms = t.elapsed().as_secs_f64() * 1000.0;
                            // Moving frames stay at frame 0: every one is a
                            // fresh store, and the first still frame
                            // re-stores at full quality instead of adding
                            // onto a while_moving() sample #0.
                            if !moved {
                                frame += 1;
                            }
                        }
                        Err(e) => {
                            eprintln!("dxr: present failed ({e}); DXR OFF (F to retry)");
                            dxr_on = false;
                            frame = 0;
                        }
                    }
                } else if let Err(e) = gpu.present_dxr_hold() {
                    eprintln!("dxr: present failed: {e}");
                }
            }
            // Stability meter (FRUSTRACER_STAB=1): the numeric dancing
            // detector — same meter as the --gpu and CPU upscaler arms;
            // healthy statics match their baselines (RR ≈ 0.14/255,
            // XeSS ≈ 1.0/255).
            if stab_on && dxr_up != GpuUp::Plain {
                stab_n = stab_n.wrapping_add(1);
                if stab_n % 15 == 0 {
                    let cap = match dxr_up {
                        GpuUp::Xess => gpu.read_xess_output(),
                        GpuUp::Fsr3 | GpuUp::Fsr4 => gpu.read_fsr_output(),
                        _ => gpu.read_rr_output(),
                    };
                    if let Ok(px) = cap {
                        if let Some(prev) = &stab_prev {
                            if prev.len() == px.len() {
                                let sum: u64 = px
                                    .iter()
                                    .zip(prev)
                                    .map(|(a, b)| {
                                        let d = |s: u32| {
                                            ((a >> s) & 0xff).abs_diff((b >> s) & 0xff) as u64
                                        };
                                        d(16) + d(8) + d(0)
                                    })
                                    .sum();
                                eprintln!(
                                    "stab: mean |Δ| {:.2}/255 over 15 frames (window-res output; render {dxw}x{dxh})",
                                    sum as f64 / (px.len() * 3) as f64,
                                );
                            }
                        }
                        stab_prev = Some(px);
                    }
                }
            }
            if edges.screenshot {
                // Upscaler sub-modes save the window-res upscaled output;
                // plain saves the render-res hdr at its own dims — the
                // --gpu P behavior.
                let grab = match dxr_up {
                    GpuUp::Rr => gpu.read_rr_output().map(|px| (px, w, h)),
                    GpuUp::Xess => gpu.read_xess_output().map(|px| (px, w, h)),
                    GpuUp::Fsr3 | GpuUp::Fsr4 => gpu.read_fsr_output().map(|px| (px, w, h)),
                    GpuUp::Plain => gpu.read_dxr_output(),
                };
                match grab {
                    Ok((px, sw, sh)) => {
                        let name = format!("screenshot_{shot}.png");
                        save_png(&name, &px, sw, sh);
                        eprintln!("saved {name}");
                        shot += 1;
                    }
                    Err(e) => eprintln!("screenshot: GPU readback failed ({e})"),
                }
            }
            fps_frames += 1;
            if (now - fps_t).as_secs_f64() >= 0.5 {
                fps = fps_frames as f64 / (now - fps_t).as_secs_f64();
                fps_frames = 0;
                fps_t = now;
                let sub = match dxr_up {
                    GpuUp::Rr => format!("DXR {dxw}x{dxh} -> DLSS-RR {w}x{h}"),
                    GpuUp::Xess => format!("DXR {dxw}x{dxh} -> XeSS {w}x{h}"),
                    GpuUp::Fsr4 => format!("DXR {dxw}x{dxh} -> FSR4-RR {w}x{h}"),
                    GpuUp::Fsr3 => format!("DXR {dxw}x{dxh} -> FSR3 {w}x{h}"),
                    GpuUp::Plain => format!(
                        "DXR {}x{} | quality {} | {} spp{}",
                        dxw,
                        dxh,
                        preset,
                        frame.min(MAX_SAMPLES),
                        if frame >= MAX_SAMPLES { " | converged" } else { "" },
                    ),
                };
                let _ =
                    window.set_title(&format!("frustracer | {fps:.0} fps | {last_ms:.1} ms | {sub}"));
            }
            continue;
        }

        // All toggle handlers have run: resolve this frame's mode once.
        // Everything below reads `mode`, never the flag soup.
        let mode = if dlss_on {
            RenderMode::Dlss
        } else if fsr_on {
            RenderMode::Fsr
        } else if xess_on {
            RenderMode::Xess
        } else if oidn_on {
            RenderMode::Oidn { temporal: oidn_temporal }
        } else if nppd_on {
            RenderMode::Nppd
        } else {
            RenderMode::Plain
        };
        // DLSS/FSR/XeSS: an upscaler owns temporal integration — fresh 1-spp
        // frames, fixed cheap preset, no budget path, no CPU accumulation.
        let upscaled = matches!(mode, RenderMode::Dlss | RenderMode::Fsr | RenderMode::Xess);
        // `neural` extends the same frame contract to NPPD (a denoiser, not
        // an upscaler — it presents through the CPU path at window res, but
        // its recurrent network owns temporal integration exactly like RR:
        // fresh 1-spp frames, fixed cheap preset, no budget frames, no CPU
        // accumulation, never idle).
        let neural = upscaled || mode == RenderMode::Nppd;

        // Cheap while moving, converge while still. Dynamic-res mode keeps
        // full resolution buffers and full quality — the estimated depth cap
        // floats the effective resolution instead. DLSS mode traces every
        // frame uncapped at RR's fixed render resolution (RR requires clean
        // per-pixel G-buffers) with frame-stationary quality.
        let use_budget = moved && hybrid && dynamic && !neural;
        // Emergency-shed predicate for the step limiters: a badly blown
        // previous frame may bypass the dwell (shed only, never grow).
        let blown = last_ms as f32 > 1.5 * RENDER_BUDGET.as_secs_f32() * 1000.0;
        let (rw, rh) = match mode {
            // Step-wise DRS when the driver reported a range; the fixed
            // optimal res otherwise. Same controller math as XeSS, with the
            // step limiter bounding how often RR takes a history hit.
            RenderMode::Dlss if dlss_drs => {
                let (_, min, max) = dlss_range.unwrap();
                let (mn, mx) = ((min.0 as usize, min.1 as usize), (max.0 as usize, max.1 as usize));
                let target =
                    xess::quantize_res(dlss_ctl.as_ref().unwrap().scale(), (w, h), mn, mx);
                let r = dlss_lim.apply(target, blown, (w, h), mn, mx);
                log_drs_adoption("dlss", &dlss_lim, &mut dlss_ep);
                r
            }
            RenderMode::Dlss => (drw, drh),
            // --lock-res: one fixed res, controller/limiter bypassed.
            RenderMode::Fsr if fsr_lock.is_some() => fsr_lock.unwrap(),
            RenderMode::Fsr => {
                // Same dynamic-resolution choreography as XeSS: the scale
                // controller's estimate quantized into [min, max], rate-
                // limited/ramped by the step limiter. Both ffx dispatches
                // take a per-dispatch renderSize, so a step costs nothing on
                // the GPU side.
                let (_, min, max) = fsr_range.unwrap();
                let (mn, mx) = ((min.0 as usize, min.1 as usize), (max.0 as usize, max.1 as usize));
                let target =
                    xess::quantize_res(fsr_ctl.as_ref().unwrap().scale(), (w, h), mn, mx);
                let r = fsr_lim.apply(target, blown, (w, h), mn, mx);
                log_drs_adoption("fsr", &fsr_lim, &mut fsr_ep);
                r
            }
            RenderMode::Xess if xess_lock.is_some() => xess_lock.unwrap(),
            RenderMode::Xess => {
                // XeSS mode: dynamic resolution, no block filling — the scale
                // controller's estimate quantized into the SDK's input range.
                // Every frame is a full-depth per-pixel trace at this size.
                let (_, min, max) = xess_range.unwrap();
                let (mn, mx) = ((min.0 as usize, min.1 as usize), (max.0 as usize, max.1 as usize));
                let target =
                    xess::quantize_res(xess_ctl.as_ref().unwrap().scale(), (w, h), mn, mx);
                let r = xess_lim.apply(target, blown, (w, h), mn, mx);
                log_drs_adoption("xess", &xess_lim, &mut xess_ep);
                r
            }
            // OIDN mode never drops to half-res: the G-buffers are full-res
            // and a half-res frame renders into a prefix with a different
            // stride. Budget (dynamic-res) frames are full-res and fine.
            RenderMode::Oidn { .. } => (w, h),
            // NPPD: fixed window res — the session's staging and the
            // recurrent state are laid out for it (no budget frames either).
            RenderMode::Nppd => (w, h),
            RenderMode::Plain if moved && !use_budget => (w / 2, h / 2),
            RenderMode::Plain => (w, h),
        };
        if rw != prev_rw {
            // Fires on every distinct-res ramp frame by design — harmless in
            // the upscaler modes (accumulate = false, free-running RNG index,
            // no CPU accumulation runs).
            frame = 0;
            prev_rw = rw;
        }
        if xess_on {
            // Keep the render-res G-buffers on this frame's resolution. On a
            // res change (per-frame during a ramp, otherwise rare) the
            // buffers are reinterpreted in place; the prev camera, the MV
            // contract, and XeSS's accumulation all survive the change (see
            // the inner comment) — only the temporal cache drops, via
            // tprev_res.
            if xess_gbufs.is_none() {
                let (_, _, max) = xess_range.unwrap();
                xess_gbufs = Some(dlss::GBufs::new(max.0 as usize, max.1 as usize));
            }
            let g = xess_gbufs.as_mut().unwrap();
            if (g.rw, g.rh) != (rw, rh) {
                // A step is a scale change, not a scene change: no
                // reset_history, no prev drop — the MV basis is derived from
                // the prev CAMERA at each frame's own res, so it stays
                // correct across the step, and XeSS's DRS carries its
                // accumulation across extent changes by design. (Resetting
                // here was the "dancing": every step wiped the history and
                // the image re-converged patchily.) Adoptions are logged at
                // the limiter; ramp intermediates land here silently.
                g.set_res(rw, rh);
            }
        }
        if fsr_on {
            // Same in-place reinterpretation for the FSR buffers (G-buffers
            // + the signal planes) — the XeSS step contract verbatim: no
            // reset, no prev drop; both ffx histories survive a renderSize
            // change by design, and the temporal cache drops via tprev_res.
            if fsr_gbufs.is_none() {
                let (_, _, max) = fsr_range.unwrap();
                // The 3.1 flavor consumes only mvec + depth (ffx_up uploads
                // exactly those; no denoiser to feed) — the slim variant
                // skips the guide planes' allocation AND their per-pixel
                // encodes at the fill sites. Signal planes are likewise
                // RR-only (~52 B/px skipped).
                fsr_gbufs = Some(if fsr_rr {
                    dlss::GBufs::new(max.0 as usize, max.1 as usize)
                } else {
                    dlss::GBufs::new_slim(max.0 as usize, max.1 as usize)
                });
                if fsr_rr {
                    fsr_bufs = Some(fsr::FsrBufs::new(max.0 as usize, max.1 as usize));
                }
            }
            let g = fsr_gbufs.as_mut().unwrap();
            if (g.rw, g.rh) != (rw, rh) {
                g.set_res(rw, rh);
                if let Some(f) = fsr_bufs.as_mut() {
                    f.set_res(rw, rh);
                }
            }
        }
        if mode == RenderMode::Dlss && (gbufs.rw, gbufs.rh) != (rw, rh) {
            // Same contract for RR: rebuild the previous frame's basis and
            // matrices at the NEW resolution (same pose, new pixel mapping)
            // so MVs land in current-res pixels; history survives via the
            // extents. The temporal cache still drops itself via tprev_res —
            // that one is a correctness contract, not a quality choice.
            gbufs.set_res(rw, rh);
            if let Some(p) = &mut dlss_prev {
                p.basis = p.cam.basis(rw, rh);
                p.mats = dlss::cam_matrices(&p.cam, rw, rh, dlss_near, dlss_far);
            }
        }
        if use_budget != prev_budget {
            frame = 0; // budget frames hold coarse fills — never accumulate onto them
            prev_budget = use_budget;
        }
        let base_q = Quality::preset(preset);
        let mut q = if neural {
            // Fixed cheap preset: the temporal denoisers/upscalers want
            // frame-stationary noise statistics.
            Quality {
                shadow_samples: 1,
                ao_samples: 1,
                reflections: true,
                fb: shade::FrustumBounce::OFF,
            }
        } else if moved && !use_budget {
            base_q.while_moving()
        } else {
            base_q
        };
        if !neural && !moved {
            q.fb.ao = bounce_mode == 1;
            q.fb.gi = bounce_mode >= 2;
        }

        // Temporal-OIDN mode renders fresh 1-spp frames (the DLSS pattern);
        // the reprojected history in the present chain is the accumulator.
        let oidn_t = mode == RenderMode::Oidn { temporal: true };
        // DLSS, XeSS, and NPPD modes never idle: fresh jittered 1-spp frames
        // are what their temporal accumulators integrate — super-resolution
        // in XeSS's case, which converges while "still" instead of on the
        // CPU; NPPD's recurrent state wants a steady stream.
        let rendered = neural || frame < MAX_SAMPLES;
        // Hoisted out of the render arm: the OIDN present branch needs the
        // exact basis this frame traced with for the history update.
        let basis = cam.basis(rw, rh);
        if rendered {
            stats.clear();
            // Static-frame structure replay: bit-equal basis at the same res,
            // and the previous rendered frame recorded a full-depth uncapped
            // hybrid structure — re-shade it with zero frustum queries
            // (replay.rs). Camera motion, res steps, budget frames, and plain
            // mode all miss or clear the key; quality/denoiser toggles do NOT
            // (the structure is a function of scene/BVH/basis/res only —
            // shading params come from this frame's ctx). Everything temporal
            // is FROZEN across replay frames: no cache clear, no rotation, so
            // tcache_prev stays the last PRODUCING frame and motion resume
            // consults exactly the last traced quadtree.
            let can_replay = opts.temporal
                && opts.replay
                && hybrid
                && !use_budget
                && replay_key.is_some_and(|(b, r)| b == basis && r == (rw, rh));
            // Record whenever this frame's structure could seed a replay next
            // frame (full-depth uncapped hybrid at a recordable res).
            let record = opts.temporal
                && opts.replay
                && !can_replay
                && hybrid
                && !use_budget
                && rw <= w
                && rh <= h;
            if record {
                replay_cache.begin(rw, rh);
            }
            // Budget (moving) frames are full-res, so the cache stays live
            // through motion and across the moving→static transition. In
            // DLSS mode "full participation res" is RR's render resolution;
            // the tprev_res check drops the cache whenever the resolution
            // changes (e.g. the G toggle), per the temporal invariant.
            // XeSS mode participates at whatever res this frame traces:
            // every frame is a full-depth hybrid frame at one fixed res, so
            // the producer/consumer contract holds; the tprev_res check
            // below drops the prev cache across any res step.
            let temporal_on = opts.temporal
                && !can_replay
                && hybrid
                && (upscaled || (rw, rh) == (w, h));
            let (tcache_cur, tprev_vec, cut_cur, cut_prev) =
                tr.begin(temporal_on, opts.adopt, rw, rh);
            let ctx = FrameCtx {
                scene,
                bvh,
                cam: basis,
                q,
                frame: match mode {
                    RenderMode::Dlss => dlss_idx,
                    RenderMode::Fsr => fsr_idx, // free-running, the dlss_idx pattern
                    RenderMode::Xess => xess_idx, // free-running, the dlss_idx pattern
                    // free-running: decorrelates the RNG while `frame` is pinned
                    RenderMode::Oidn { temporal: true } => oidn_seq,
                    RenderMode::Nppd => nppd_seq,
                    _ => frame,
                },
                // DLSS/XeSS ignore `jitter` (frame_jitter wins in
                // trace_primary); temporal OIDN and NPPD always jitter their
                // fresh 1-spp frames; the accumulating modes jitter after
                // the first (pilot) sample.
                jitter: oidn_t || mode == RenderMode::Nppd || (!upscaled && frame > 0),
                rw,
                rh,
                accum: &accum,
                info: &info,
                tbuf: &tbuf,
                stats: &stats,
                sun: render::sun_dir(scene),
                tcache_cur,
                tcache_prev: &tprev_vec,
                accumulate: !neural && !oidn_t,
                gbuf: match mode {
                    RenderMode::Dlss => Some(&gbufs),
                    RenderMode::Fsr => fsr_gbufs.as_ref(),
                    RenderMode::Xess => xess_gbufs.as_ref(),
                    // Both OIDN sub-modes fill the window-res G-buffers.
                    RenderMode::Oidn { .. } => oidn_gbufs.as_ref(),
                    RenderMode::Nppd => nppd_gbufs.as_ref(),
                    RenderMode::Plain => None,
                },
                fsr_buf: match mode {
                    RenderMode::Fsr => fsr_bufs.as_ref(),
                    _ => None,
                },
                prev_cam: match mode {
                    RenderMode::Dlss => dlss_prev.as_ref().map(|p| p.basis),
                    // Basis derived at THIS frame's res — correct across
                    // DRS steps by construction.
                    RenderMode::Fsr => fsr_prev.map(|c| c.basis(rw, rh)),
                    RenderMode::Xess => xess_prev.map(|c| c.basis(rw, rh)),
                    // Window res always — the state warp consumes these MVs.
                    RenderMode::Nppd => nppd_prev.map(|c| c.basis(rw, rh)),
                    _ => None,
                },
                frame_jitter: match mode {
                    RenderMode::Dlss => Some(dlss::jitter_for(dlss_idx)),
                    RenderMode::Fsr => Some(dlss::jitter_for(fsr_idx)),
                    RenderMode::Xess => Some(dlss::jitter_for(xess_idx)),
                    _ => None,
                },
                // Adaptive shading rate: XeSS only — its temporal
                // accumulation launders the spatially varying sampling. On
                // RR the 2×2-correlated shadow noise and per-frame cell
                // reclassification presented as patchy "dancing" (RR's
                // network preserves block-correlated noise as structure
                // instead of integrating it). Revisit with an RR-friendly
                // classifier before re-enabling. --no-adaptive forces
                // uniform per-pixel shading (visibility is per-pixel
                // either way).
                adaptive: mode == RenderMode::Xess && opts.adaptive,
                // Hemi sharing only ever fires on fb frames (still,
                // non-upscaled — the shade_tile branch checks q.fb);
                // --no-hemi-share is the per-session kill switch.
                hemi_share: opts.hemi_share,
                replay_rec: if record { Some(&replay_cache) } else { None },
                cut_cur,
                cut_prev,
                discard_seeds: opts.discard_seeds,
                defer_shade: opts.defer_shade,
            };
            let t = Instant::now();
            if can_replay {
                render::render_frame_replay(&ctx, &replay_cache);
            } else if use_budget {
                render::render_frame_capped(&ctx, depth_est.floor() as u32);
            } else {
                render::render_frame(&ctx, hybrid);
            }
            last_ms = t.elapsed().as_secs_f64() * 1000.0;
            if !can_replay {
                tr.end(temporal_on, opts.adopt, basis);
                // A recorded, unpoisoned structure seeds next frame's replay;
                // anything else (plain, budget, overflow) clears the key.
                replay_key = if record && replay_cache.valid() {
                    Some((basis, (rw, rh)))
                } else {
                    None
                };
            }
            if use_budget {
                // Update the cap estimate from this frame only — non-budget
                // frames (half-res, plain, converging) have incomparable cost.
                let target = RENDER_BUDGET.as_secs_f32() * 1000.0;
                let err = (target / (last_ms as f32).max(0.1)).log2() * 0.5; // log4
                let step = (DEPTH_GAIN * err).clamp(-STEP_DOWN_MAX, STEP_UP_MAX);
                // Deadband: don't climb while already using >60% of the budget
                // (the next level costs ~4x) — parks at the deepest cap that
                // fits instead of flapping across the boundary.
                if step < 0.0 || (last_ms as f32) < 0.6 * target {
                    depth_est = (depth_est + step).clamp(render::MIN_BUDGET_DEPTH as f32, depth_full);
                }
            }
            if xess_on {
                // The scale controller only ever sees XeSS frames — a
                // comparable cost model (full-depth trace, cost ~ area).
                // While still, the temporal cache makes frames cheaper and
                // the scale creeps toward the range max: super-resolution.
                // The previous frame's pre-denoise cost rides along: it is
                // area-proportional work the chosen resolution buys, and a
                // controller blind to it would creep past the budget on a
                // slow OIDN device.
                if let Some(ctl) = &mut xess_ctl {
                    ctl.update(
                        (last_ms + pre_ms) as f32,
                        RENDER_BUDGET.as_secs_f32() * 1000.0,
                    );
                }
            }
            if mode == RenderMode::Dlss {
                // Same controller, DLSS flavor. RR has no CPU pre-pass, so
                // the trace time alone is the area-proportional cost.
                if let Some(ctl) = &mut dlss_ctl {
                    ctl.update(last_ms as f32, RENDER_BUDGET.as_secs_f32() * 1000.0);
                }
            }
            if mode == RenderMode::Fsr {
                // Same controller, FSR flavor: no CPU pre-pass, the trace
                // time alone is the area-proportional cost.
                if let Some(ctl) = &mut fsr_ctl {
                    ctl.update(last_ms as f32, RENDER_BUDGET.as_secs_f32() * 1000.0);
                }
            }
            frame += 1;
            oidn_seq = oidn_seq.wrapping_add(1);
            nppd_seq = nppd_seq.wrapping_add(1);
        } else {
            std::thread::sleep(Duration::from_millis(8)); // converged — idle
        }

        // GPU tonemap consumes the raw HDR accumulation directly, but only
        // for full-res frames without the overlay — half-res upscale and the
        // overlay composite live in the CPU resolve. OIDN mode presents its
        // denoised output through the CPU path, so it excludes the GPU tonemap.
        let use_gpu_tone =
            gpu_tonemap && !oidn_on && !nppd_on && rw == w && !(overlay_on && hybrid);
        if dlss_on {
            // DLSS-RR: hand the 1-spp radiance + G-buffers to the denoiser;
            // it outputs denoised HDR which the GPU tonemap presents.
            // Everything SL sees — matrices, jitter, MVs, mvec_scale — lives
            // in render-res pixel space; only the RR output is window-sized.
            let mats = dlss::cam_matrices(&cam, rw, rh, dlss_near, dlss_far);
            let fc = dlss::frame_constants(
                &cam,
                &mats,
                dlss_prev.as_ref().map(|p| &p.mats),
                dlss::jitter_for(dlss_idx),
                dlss_reset,
                dlss_near,
                dlss_far,
                rw,
                rh,
            );
            match gpu.present_rr(&accum, &gbufs, &fc, dlss_idx) {
                Ok(()) => {
                    dlss_prev = Some(dlss::DlssPrev { basis: cam.basis(rw, rh), mats, cam });
                    dlss_reset = false;
                    dlss_idx = dlss_idx.wrapping_add(1);
                }
                Err(e) => {
                    // A Streamline failure (e.g. out of VRAM) shouldn't kill
                    // the app: the aborted frame never reached the GPU, so
                    // fall back to the CPU pipeline; the next loop iteration
                    // presents normally. G retries.
                    eprintln!("dlss: present failed ({e}); Ray Reconstruction disabled (G to retry)");
                    dlss_on = false;
                    dlss_prev = None;
                    dlss_reset = true;
                    frame = 0;
                }
            }
        } else if fsr_on {
            // FSR4+RR flavor: hand the G-buffers + demodulated signals to
            // Ray Regeneration (denoise) -> composite (remodulate) -> FSR4
            // (upscale to the window). FSR 3.1 flavor: the frame's 1-spp
            // HDR shade (accum) + MVs + depth -> one FSR 3.1 upscale
            // dispatch, no denoiser anywhere. Either way everything the ffx
            // dispatches see is in render-res space; only the upscaled
            // output is window-sized, and it is never written back into
            // accum.
            let fg = fsr_gbufs.as_ref().expect("fsr_on without gbufs");
            let mats = dlss::cam_matrices(&cam, rw, rh, dlss_near, dlss_far);
            let prev_mats = fsr_prev.map(|c| dlss::cam_matrices(&c, rw, rh, dlss_near, dlss_far));
            let fc = dlss::frame_constants(
                &cam,
                &mats,
                prev_mats.as_ref(),
                dlss::jitter_for(fsr_idx),
                fsr_reset,
                dlss_near,
                dlss_far,
                rw,
                rh,
            );
            let presented = if fsr_rr {
                let fb = fsr_bufs.as_ref().expect("fsr_on without signal bufs");
                gpu.present_fsr(fg, fb, &fc, fsr_prev.map(|c| c.pos), fsr_idx, last_ms as f32)
            } else {
                gpu.present_fsr3(&accum, fg, &fc, last_ms as f32)
            };
            match presented {
                Ok(()) => {
                    fsr_prev = Some(cam);
                    fsr_reset = false;
                    fsr_idx = fsr_idx.wrapping_add(1);
                }
                Err(e) => {
                    // An ffx failure shouldn't kill the app: the aborted
                    // frame never reached the GPU — fall back to the plain
                    // pipeline; K retries.
                    eprintln!("fsr: present failed ({e}); FSR disabled (K to retry)");
                    fsr_on = false;
                    fsr_prev = None;
                    fsr_reset = true;
                    frame = 0;
                }
            }
        } else if xess_on {
            // XeSS-SR: hand the fresh 1-spp frame (optionally OIDN-denoised
            // first, at the same dynamic render res) + MV/depth to the
            // upscaler; it accumulates the jittered sample stream into the
            // window-sized image. Everything XeSS sees is in input-res pixel
            // space; only its output is window-sized. The denoised/upscaled
            // output is never written back into accum or the history.
            let xg = xess_gbufs.as_ref().expect("xess_on without gbufs");
            let n = rw * rh * 3;
            let jit = dlss::jitter_for(xess_idx);
            if xess_oidn == XessOidn::Post {
                // POST placement (the A/B experiment): raw 1-spp → XeSS
                // upscale → readback → OIDN at window res with
                // nearest-upscaled guides → CPU tonemap → present_cpu (the
                // frame's single Present). Costs a synchronous readback and
                // a window-res denoise; no pre-EMA history in this ordering.
                // Post's denoise is window-res (constant, not area-
                // proportional) — shedding render resolution wouldn't reduce
                // it, so the scale controller must not see it.
                pre_ms = 0.0;
                if xess_hdr.len() != w * h * 3 {
                    xess_hdr.resize(w * h * 3, 0.0);
                }
                match gpu.upscale_xess_to_cpu(
                    &gpu::xr::ColorSrc::Accum(&accum[..n]),
                    xg,
                    rw,
                    rh,
                    jit,
                    xess_reset,
                    dlss_near,
                    dlss_far,
                    &mut xess_hdr,
                ) {
                    Ok(()) => {
                        xess_prev = Some(cam);
                        xess_reset = false;
                        xess_idx = xess_idx.wrapping_add(1);
                        let octx = oidn_ctx.as_mut().expect("xess_oidn without context");
                        let og = oidn_gbufs.as_ref().expect("xess_oidn without gbufs");
                        og.upscale_guides_from(xg);
                        let result = octx
                            .set_res(w, h)
                            .and_then(|()| octx.denoise_hdr(&xess_hdr, og).map(drop));
                        match result {
                            Ok(()) => render::resolve_hdr(
                                octx.last_output(),
                                &info,
                                false,
                                &mut present,
                                w,
                                h,
                                w,
                                h,
                            ),
                            Err(e) => {
                                eprintln!(
                                    "oidn: post-denoise failed ({e}); presenting the raw upscale (N to retry)"
                                );
                                xess_oidn = XessOidn::Off;
                                render::resolve_hdr(
                                    &xess_hdr, &info, false, &mut present, w, h, w, h,
                                );
                            }
                        }
                        gpu.present_cpu(&present).expect("GPU present failed");
                    }
                    Err(e) => {
                        eprintln!("xess: upscale failed ({e}); XeSS disabled (X to retry)");
                        xess_on = false;
                        xess_prev = None;
                        xess_reset = true;
                        frame = 0;
                    }
                }
            } else {
                // OFF / PRE placements: the GPU tonemap present path. The
                // pre-pass is resolution-agile — set_res rebinds the filter
                // on a res step (cheap; the weights stay loaded), and the
                // reprojected history reinterprets in place within its
                // window-res capacity, invalidating itself (a fresh history
                // is an invalidated one; XeSS's own accumulation hides the
                // blip). Both run per distinct-res frame during a DRS ramp.
                // Which pre-denoiser fed this frame, if any (borrow-friendly
                // tag; the actual output slices are re-borrowed below).
                enum Pre {
                    Oidn,
                    Nppd,
                }
                let denoised = if xess_nppd {
                    // NPPD pre-denoise at the render res: one recurrent step
                    // on the 1-spp frame — the state warp reads xess_gbufs'
                    // motion vectors (the same xess_prev contract the
                    // upscaler consumes). set_res is a no-op at a locked res;
                    // under dynamic DRS each step invalidates the state (the
                    // startup note).
                    let nctx = nppd_ctx.as_mut().expect("xess_nppd without context");
                    let t_pre = Instant::now();
                    let result = nctx
                        .set_res(rw, rh)
                        .and_then(|()| nctx.denoise(&accum[..n], xg, &basis, dlss_far).map(drop));
                    pre_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
                    match result {
                        Ok(()) => Some(Pre::Nppd),
                        Err(e) => {
                            eprintln!("nppd: pre-denoise failed ({e}); raw XeSS input (J to retry)");
                            xess_nppd = false;
                            pre_ms = 0.0;
                            // Statistics flip denoised→raw mid-stream: the
                            // upscaler must not integrate across it.
                            xess_reset = true;
                            None
                        }
                    }
                } else if xess_oidn == XessOidn::Pre {
                    let octx = oidn_ctx.as_mut().expect("xess_oidn without context");
                    let t_pre = Instant::now();
                    let result = octx.set_res(rw, rh).and_then(|()| {
                        if oidn_temporal {
                            let hist = oidn_hist.as_mut().expect("xess_oidn without history");
                            hist.set_res(rw, rh);
                            let t0 = Instant::now();
                            last_hist = hist.update(
                                &basis,
                                &accum[..n],
                                xg,
                                &info[..rw * rh],
                                dlss_far,
                                MAX_SAMPLES as f32,
                            );
                            hist_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            octx.denoise_hdr(hist.color(), xg).map(drop)
                        } else {
                            // accum holds exactly the last 1-spp frame (store
                            // semantics in XeSS mode) — denoise it per-frame.
                            octx.denoise(&accum[..n], 1, xg).map(drop)
                        }
                    });
                    pre_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
                    match result {
                        Ok(()) => Some(Pre::Oidn),
                        Err(e) => {
                            eprintln!("oidn: pre-denoise failed ({e}); raw XeSS input (N to retry)");
                            xess_oidn = XessOidn::Off;
                            pre_ms = 0.0;
                            // The color source flips denoised→raw this same
                            // frame; the upscaler must not integrate across
                            // the statistics flip (mirrors the keyboard
                            // Pre→Off transition, which resets via the
                            // toggle_oidn edge).
                            xess_reset = true;
                            None
                        }
                    }
                } else {
                    pre_ms = 0.0;
                    None
                };
                let color = match denoised {
                    Some(Pre::Oidn) => {
                        gpu::xr::ColorSrc::Hdr(oidn_ctx.as_ref().unwrap().last_output())
                    }
                    Some(Pre::Nppd) => {
                        gpu::xr::ColorSrc::Hdr(nppd_ctx.as_ref().unwrap().last_output())
                    }
                    None => gpu::xr::ColorSrc::Accum(&accum[..n]),
                };
                match gpu.present_xess(&color, xg, rw, rh, jit, xess_reset, dlss_near, dlss_far) {
                    Ok(()) => {
                        xess_prev = Some(cam);
                        xess_reset = false;
                        xess_idx = xess_idx.wrapping_add(1);
                    }
                    Err(e) => {
                        // Nothing reached the GPU (abort_frame) — fall back to
                        // the CPU pipeline next iteration; X retries.
                        eprintln!("xess: present failed ({e}); XeSS disabled (X to retry)");
                        xess_on = false;
                        xess_prev = None;
                        xess_reset = true;
                        frame = 0;
                    }
                }
            }
        } else if oidn_on {
            // OIDN: denoise on the CPU-resolve path — temporal mode folds the
            // fresh 1-spp frame into the reprojected history and denoises
            // that; plain mode denoises the accumulation average. The
            // denoised HDR is never written back into accum or the history.
            // On an idle (converged) iteration the cached present buffer is
            // re-presented without re-denoising; an overlay toggle while idle
            // re-composites the retained denoised HDR instead.
            let octx = oidn_ctx.as_mut().expect("oidn_on without context");
            if rendered {
                let og = oidn_gbufs.as_ref().expect("oidn_on without gbufs");
                // Returning from XeSS mode leaves the filter/history bound at
                // a render res — rebind at window res (no-op otherwise).
                let result = octx.set_res(w, h).and_then(|()| {
                    if oidn_t {
                        let hist = oidn_hist.as_mut().expect("oidn_on without history");
                        hist.set_res(w, h);
                        let t0 = Instant::now();
                        last_hist =
                            hist.update(&basis, &accum, og, &info, dlss_far, MAX_SAMPLES as f32);
                        hist_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        octx.denoise_hdr(hist.color(), og).map(drop)
                    } else {
                        octx.denoise(&accum, frame.max(1), og).map(drop)
                    }
                });
                match result {
                    Ok(()) => render::resolve_hdr(
                        octx.last_output(),
                        &info,
                        overlay_on && hybrid,
                        &mut present,
                        w,
                        h,
                        w,
                        h,
                    ),
                    Err(e) => {
                        eprintln!("oidn: denoise failed ({e}); OFF (N to retry)");
                        oidn_on = false;
                        if oidn_t {
                            // accum holds one 1-spp frame, not a sum: resolve
                            // it as 1 sample and restart accumulation cleanly.
                            frame = 0;
                            if let Some(h) = &mut oidn_hist {
                                h.invalidate();
                            }
                        }
                        render::resolve(
                            &accum,
                            &info,
                            if oidn_t { 1 } else { frame.max(1) },
                            overlay_on && hybrid,
                            &mut present,
                            rw,
                            rh,
                            w,
                            h,
                        );
                    }
                }
            } else if edges.toggle_overlay {
                render::resolve_hdr(
                    octx.last_output(),
                    &info,
                    overlay_on && hybrid,
                    &mut present,
                    w,
                    h,
                    w,
                    h,
                );
            }
            gpu.present_cpu(&present).expect("GPU present failed");
        } else if nppd_on {
            // NPPD: one recurrent network step on the CPU-resolve path — the
            // state is warped by this frame's motion vectors, the 1-spp frame
            // + G-buffers packed, the graph run (DirectML or CPU EP), and the
            // denoised HDR tonemapped into the present buffer. The output is
            // never written back into accum; the mode never idles (`rendered`
            // is unconditional), so no re-present arm exists.
            let nctx = nppd_ctx.as_mut().expect("nppd_on without context");
            if rendered {
                let ng = nppd_gbufs.as_ref().expect("nppd_on without gbufs");
                let result = nctx
                    .set_res(w, h)
                    .and_then(|()| nctx.denoise(&accum, ng, &basis, dlss_far).map(drop));
                match result {
                    Ok(()) => {
                        render::resolve_hdr(
                            nctx.last_output(),
                            &info,
                            overlay_on && hybrid,
                            &mut present,
                            w,
                            h,
                            w,
                            h,
                        );
                        // The MV contract: prev is the camera of the last
                        // frame the recurrent state actually saw.
                        nppd_prev = Some(cam);
                    }
                    Err(e) => {
                        eprintln!("nppd: denoise failed ({e}); OFF (J to retry)");
                        nppd_on = false;
                        nppd_prev = None;
                        // accum holds one 1-spp frame, not a sum: resolve it
                        // as 1 sample and restart accumulation cleanly.
                        frame = 0;
                        render::resolve(
                            &accum,
                            &info,
                            1,
                            overlay_on && hybrid,
                            &mut present,
                            rw,
                            rh,
                            w,
                            h,
                        );
                    }
                }
            }
            gpu.present_cpu(&present).expect("GPU present failed");
        } else if use_gpu_tone {
            gpu.present_hdr(&accum, frame.max(1)).expect("GPU present failed");
        } else {
            render::resolve(
                &accum,
                &info,
                frame.max(1),
                overlay_on && hybrid,
                &mut present,
                rw,
                rh,
                w,
                h,
            );
            gpu.present_cpu(&present).expect("GPU present failed");
        }
        // Stability meter (FRUSTRACER_STAB=1): quantifies temporal
        // instability of the upscaled output — hold the camera still and a
        // healthy pipeline trends toward ~0; "dancing" holds a high mean.
        // Reads back the GPU output synchronously; diagnostics only.
        if stab_on && upscaled && rendered {
            stab_n = stab_n.wrapping_add(1);
            if stab_n % 15 == 0 {
                let cap = if dlss_on {
                    gpu.read_rr_output()
                } else if fsr_on {
                    gpu.read_fsr_output()
                } else {
                    gpu.read_xess_output()
                };
                if let Ok(px) = cap {
                    if let Some(prev) = &stab_prev {
                        if prev.len() == px.len() {
                            let sum: u64 = px
                                .iter()
                                .zip(prev)
                                .map(|(a, b)| {
                                    let d = |s: u32| {
                                        ((a >> s) & 0xff).abs_diff((b >> s) & 0xff) as u64
                                    };
                                    d(16) + d(8) + d(0)
                                })
                                .sum();
                            eprintln!(
                                "stab: mean |Δ| {:.2}/255 over 15 frames (window-res output; render {}x{})",
                                sum as f64 / (px.len() * 3) as f64,
                                rw,
                                rh
                            );
                        }
                    }
                    stab_prev = Some(px);
                }
            }
        }
        // Tracy frame boundary + per-frame plots (all presents are done).
        prof::frame_mark();
        plot!("frame ms", last_ms);
        plot!("fr-queries", stats.frustum_queries.load(Relaxed));
        plot!("adopts", stats.temporal_cut_adopts.load(Relaxed));
        plot!("replay leaves", stats.replay_leaf_tiles.load(Relaxed));
        plot!("render h", rh);
        fps_frames += 1;
        if (now - fps_t).as_secs_f64() >= 1.0 {
            fps = fps_frames as f64 / (now - fps_t).as_secs_f64();
            fps_frames = 0;
            fps_t = now;
        }

        if edges.screenshot {
            if dlss_on {
                // The denoised image exists only on the GPU — read the RR
                // output back and tonemap it (same curve, 1 spp). On failure
                // fall back to a fresh CPU resolve of the noisy input.
                match gpu.read_rr_output() {
                    Ok(px) => present.copy_from_slice(&px),
                    Err(e) => {
                        eprintln!("screenshot: RR readback failed ({e}); saving noisy 1-spp resolve");
                        render::resolve(&accum, &info, 1, false, &mut present, rw, rh, w, h);
                    }
                }
            } else if fsr_on {
                // Same story as DLSS: the denoised+upscaled image lives only
                // on the GPU.
                match gpu.read_fsr_output() {
                    Ok(px) => present.copy_from_slice(&px),
                    Err(e) => {
                        eprintln!("screenshot: FSR readback failed ({e}); saving noisy 1-spp resolve");
                        render::resolve(&accum, &info, 1, false, &mut present, rw, rh, w, h);
                    }
                }
            } else if xess_on && xess_oidn != XessOidn::Post {
                // Same story as DLSS: the upscaled image lives only on the
                // GPU. (POST placement presents via the CPU path, so its
                // present buffer is already current — plain save.)
                match gpu.read_xess_output() {
                    Ok(px) => present.copy_from_slice(&px),
                    Err(e) => {
                        eprintln!("screenshot: XeSS readback failed ({e}); saving noisy 1-spp resolve");
                        render::resolve(&accum, &info, 1, false, &mut present, rw, rh, w, h);
                    }
                }
            } else if use_gpu_tone {
                // The present buffer is stale in GPU-tonemap mode; resolve
                // fresh for the screenshot.
                render::resolve(&accum, &info, frame.max(1), false, &mut present, rw, rh, w, h);
            }
            let name = format!("screenshot_{shot}.png");
            save_png(&name, &present, w, h);
            eprintln!("saved {name}");
            shot += 1;
        }
        if edges.verify {
            eprintln!("verifying current view...");
            let vstats = Stats::default();
            // Cache-free on purpose: an independent ground-truth oracle.
            let rep = render::verify(scene, bvh, &cam.basis(w, h), base_q, w, h, &vstats, None, &[], None);
            eprintln!(
                "verify ({} px): false-sky {} | tmin-overshoot {} | hybrid-extra {} | max rel t err {:.2e} -> {}",
                rep.pixels,
                rep.false_sky,
                rep.overshoot,
                rep.hybrid_extra,
                rep.max_rel_err,
                if rep.ok() { "OK" } else { "MISMATCH" }
            );
        }

        if (now - last_title).as_secs_f64() > 0.25 {
            last_title = now;
            let coarse_px = stats.coarse_pixels.load(Relaxed);
            let coarse = if use_budget {
                format!(" | cap {} coarse {}%", depth_est.floor() as u32, coarse_px * 100 / (rw * rh) as u64)
            } else {
                String::new()
            };
            let _ = window.set_title(&format!(
                "frustracer | {:.0} fps | {}{}{} | {}x{} | {:.1} ms | {} | quality {}{}{}{}{}{}",
                fps,
                if hybrid { "hybrid" } else { "plain" },
                if !dlss_on && !fsr_on && !xess_on && hybrid && dynamic { "+dyn" } else { "" },
                if dlss_on {
                    if dlss_drs {
                        format!(" | DLSS: dyn {}% + RR", rh * 100 / h)
                    } else if opts.lock_scale.is_some() && (drw, drh) != (w, h) {
                        format!(" | DLSS: lock {}% + RR", rh * 100 / h)
                    } else if (drw, drh) == (w, h) {
                        // Native render res means the optimal-settings query
                        // fell back to DLAA (or a native lock); anything
                        // smaller is Quality mode.
                        " | DLSS: DLAA + RR".to_string()
                    } else {
                        " | DLSS: Quality + RR".to_string()
                    }
                } else if fsr_on {
                    format!(
                        " | {}: {} {}%{}",
                        fsr_hud,
                        if fsr_lock.is_some() { "lock" } else { "dyn" },
                        rh * 100 / h,
                        fsr_hud_sfx,
                    )
                } else if xess_on {
                    format!(
                        " | XeSS: {} {}%{}",
                        if xess_lock.is_some() { "lock" } else { "dyn" },
                        rh * 100 / h,
                        if xess_nppd {
                            match nppd_ctx.as_ref() {
                                Some(c) =>
                                    format!(" +NPPD(pre) {} {:.1} ms", c.device_desc, c.last_ms),
                                None => String::new(),
                            }
                        } else {
                            match (xess_oidn, oidn_ctx.as_ref()) {
                                (XessOidn::Pre, Some(c)) =>
                                    format!(" +OIDN(pre) {} {:.1} ms", c.device_desc, c.last_ms),
                                (XessOidn::Post, Some(c)) =>
                                    format!(" +OIDN(post) {} {:.1} ms", c.device_desc, c.last_ms),
                                _ => String::new(),
                            }
                        }
                    )
                } else if oidn_on {
                    match oidn_ctx.as_ref() {
                        Some(c) => format!(
                            " | OIDN{}: {} {:.1} ms",
                            if oidn_temporal { "+T" } else { "" },
                            c.device_desc,
                            c.last_ms
                        ),
                        None => " | OIDN".to_string(),
                    }
                } else if nppd_on {
                    match nppd_ctx.as_ref() {
                        Some(c) => format!(" | NPPD: {} {:.1} ms", c.device_desc, c.last_ms),
                        None => " | NPPD".to_string(),
                    }
                } else {
                    " | DLSS: off".to_string()
                },
                rw,
                rh,
                last_ms,
                if dlss_on || fsr_on || xess_on || nppd_on {
                    "1 spp".to_string()
                } else {
                    format!("{} spp", frame.min(MAX_SAMPLES))
                },
                preset,
                coarse,
                if dlss_on || fsr_on || xess_on || nppd_on {
                    ""
                } else {
                    ["", " | hemi-AO", " | hemi-GI"][bounce_mode as usize]
                },
                if use_gpu_tone && !dlss_on && !fsr_on && !xess_on { " | gpu-tone" } else { "" },
                if overlay_on { " | overlay" } else { "" },
                if !dlss_on && !fsr_on && !xess_on && !nppd_on && frame >= MAX_SAMPLES {
                    " | converged"
                } else {
                    ""
                },
            ));
        }
        if (now - last_stats).as_secs_f64() > 1.0 && frame <= MAX_SAMPLES && frame > 0 {
            last_stats = now;
            eprintln!(
                "[{}] {:.1} ms | {}{}{}",
                if hybrid { "hybrid" } else { "plain" },
                last_ms,
                stats.summary_line(),
                if xess_on || fsr_on || (dlss_on && (dlss_drs || opts.lock_scale.is_some())) {
                    format!(
                        " | {} {}x{} ({}%)",
                        if xess_on {
                            "xess"
                        } else if fsr_on {
                            "fsr"
                        } else if dlss_drs {
                            "dlss-drs"
                        } else {
                            "dlss-lock"
                        },
                        rw,
                        rh,
                        rh * 100 / h
                    )
                } else {
                    String::new()
                },
                if (oidn_on || (xess_on && xess_oidn == XessOidn::Pre)) && oidn_temporal {
                    format!(
                        " | hist {:.1} ms acc {} rej {} coarse {}/{} L {:.0}..{:.0}",
                        hist_ms,
                        last_hist.accepted,
                        last_hist.rejected,
                        last_hist.coarse_kept,
                        last_hist.coarse_reset,
                        last_hist.len_min,
                        last_hist.len_max
                    )
                } else {
                    String::new()
                }
            );
        }
    };

    // Carry user-visible state into the next session (resize re-entry) or
    // just record it on quit — cheap either way.
    *persist = Some(Persist {
        hybrid,
        dynamic,
        overlay_on,
        gpu_tonemap,
        bounce_mode,
        preset,
        dlss_on,
        xess_on,
        fsr_on,
        oidn_on,
        oidn_temporal,
        xess_oidn,
        nppd_on,
        xess_nppd,
        dxr_on,
        dxr_up_plain: dxr_up == GpuUp::Plain,
        gpu_up_plain: gpu_up == GpuUp::Plain,
        gpu_nppd_on,
        oidn_failed,
        nppd_failed,
        dxr_failed,
        shot,
        depth_est,
    });
    end
}

fn save_png(name: &str, present: &[u32], w: usize, h: usize) {
    let mut rgb = Vec::with_capacity(w * h * 3);
    for px in present {
        rgb.push((px >> 16) as u8);
        rgb.push((px >> 8) as u8);
        rgb.push(*px as u8);
    }
    if let Err(e) = image::save_buffer(name, &rgb, w as u32, h as u32, image::ColorType::Rgb8) {
        eprintln!("failed to save {name}: {e}");
    }
}
