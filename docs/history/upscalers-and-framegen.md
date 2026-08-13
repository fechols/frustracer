# Upscalers and frame generation

Scene levers (`--stress`, `--tile`, `--no-bc7`), the gate basics, the always-on upscaler chain, OIDN, XeSS, FSR/FSR4/FSR3, and all three frame-generation legs (ffx FI, raw-NGX DLSS-G, XeSS-FG).

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
cargo run --release -- --stress 5000  # perf test: n-object procedural field (composes with --check*)
cargo run --release -- model.obj --tile 3       # replicate a loaded OBJ into a 3x3 grid (also NxM,
                                                # e.g. --tile 4x2): flattened copies, shared
                                                # materials/textures, stress-style camera/light
                                                # framing — the 100M-triangle path (see Big scenes;
                                                # composes with --check* as the loaded-scene gate class)
cargo run --release -- model.obj --no-bc7  # A/B lever: upload scene textures as raw RGBA8. BC7
                                        # block compression is ON BY DEFAULT (8 bpp vs 32 — GPU
                                        # upload only; on BOTH backends since 2026-08-11, see
                                        # src/vk/bc7.rs and the M3j block in --check-vk: one
                                        # kernel, two hosts, and `should_compress` called verbatim; measured live set incl. mips: Intel Sponza
                                        # 4608 -> 3072 MB, Bistro 2310 -> 1725; the CPU samplers
                                        # keep exact RGBA8 and alpha-masked cutout / height-
                                        # carrying textures NEVER compress — see Real scenes). The
                                        # DEFAULT arm is a GPU COMPUTE encoder (src/gpu/bc7gpu.rs
                                        # + shaders/bc7enc.hlsl: mode-6 PCA/LS fit + a 2-means-
                                        # ranked mode-1 two-subset arm; fxc cs_5_0, the bloom
                                        # no-DXC precedent) dispatched per band inside the scene
                                        # upload — what made default-on affordable: measured at
                                        # fast, SM-lp 117 ms / Bistro 229 ms (2.1 Gtexel/s) /
                                        # Intel Sponza 282 ms (3.8 Gtexel/s; rates count every
                                        # encoded level, mips incl.) vs the ispc CPU
                                        # arm's 0.8 / 9 / 20 s. THOSE FIGURES PREDATE THE BATCHED
                                        # STAGING RING (2026-08-12 — the loading-screen pump; see
                                        # "Loading screen" below): the bands' encode+copy-out pairs
                                        # used to be one BLOCKING submit each, so the rate was
                                        # round-trip-bound rather than kernel-bound. Same blocks,
                                        # ~2x the throughput — THE WORLD's 306 BC7 textures read
                                        # 1091 -> 558 ms (859 -> 1680 Mtexel/s, 4090).
                                        # --bc7-cpu keeps that ispc arm as
                                        # the A/B lever + independent cross-check (M11 worst on
                                        # SM-lp: cpu 33.0 dB vs gpu 32.0); --bc7-quality
                                        # ultrafast|fast|basic|slow = GPU effort tiers (0 = mode-6
                                        # no-refit, 26.3 dB; 1 = +2 LS rounds + CONDITIONAL
                                        # mode-1 top-4; 2/3 = mode-1 always, top-8/16). Encoder
                                        # construction failure = LOUD line + uncompressed RGBA8,
                                        # never an implicit CPU stall. Still no disk cache — the
                                        # per-load encode is now cheap by construction. Every
                                        # --check-gpu runs the bc7-gpu structural gate (flat
                                        # bit-exact + stride + ramp + two-cluster mode-1 proof,
                                        # synthetic — fires even on the untextured procedural
                                        # scene) and M11 runs whenever BC7 is armed on a
                                        # compressible-textured scene
cargo run --release -- --check        # headless: verify + benchmark + write check.png (the
                                      # DEFAULT scene's frame — a scene-keyed run writes
                                      # check-<tag>.png instead and leaves the tracked goldens
                                      # alone; see "Otherwise --check is the test suite")
cargo run --release -- --check-dlss   # headless: DLSS G-buffer MV/depth/matrix self-test (no GPU)
cargo run --release -- --dlss-dump    # --check-dlss + G-buffer PNG dumps
cargo run --release -- --no-dlss      # skip the DLSS-RR level of the ALWAYS-ON upscaler chain:
                                      # every session probes DLSS-RR -> FSR4-RR -> XeSS -> FSR3 in
                                      # that order and wires the FIRST supported level (--<x>
                                      # force-starts the chain at level x, --no-<x> skips a level;
                                      # upchain::self_test in --check gates the resolution algebra;
                                      # chain exhausted = one LOUD line + plain presentation)
cargo run --release -- --no-upscale   # plain presentation: no temporal upscaler at all — the
                                      # benchmark escape and the ONLY spelling of the old --no-dlss
                                      # plain path
cargo run --release -- --cpu          # the CPU frustum-tracer as the render mode — clears BOTH GPU
                                      # modes (--dxr is the DEFAULT session on NVIDIA/AMD: the eager
                                      # DispatchRays pipeline feeding the chain's wired upscaler;
                                      # --gpu is the wavefront tracer, and the FLAGLESS default on an
                                      # INTEL adapter — see the vendor-defaults paragraph in the DXR
                                      # section). Later flags win: `--cpu --gpu` = --gpu
cargo run --release -- --check-oidn   # headless: OIDN denoise self-test (needs the OIDN DLLs on disk)
cargo run --release -- --oidn-dump    # --check-oidn + before/after/G-buffer PNG dumps
cargo run --release -- --oidn         # start with OIDN denoising on (N toggles; DLSS off)
cargo run --release -- --oidn --oidn-no-temporal  # OIDN without the reprojection history (M toggles)
cargo run --release -- --oidn --oidn-quality high # OIDN RT-filter quality: fast|balanced|high (default balanced)
cargo run --release -- --oidn --oidn-no-clean-aux # don't declare the OIDN guides noise-free (A/B lever)
cargo run --release -- --check-xess   # headless: XeSS dynamic-res contract self-test (no GPU, no DLL)
cargo run --release -- --xess-dump    # --check-xess + G-buffer PNG dumps
cargo run --release -- --xess         # XeSS-SR dynamic super-resolution (X toggles; force-starts
                                      # the upscaler chain at XeSS — missing libxess.dll falls to FSR3)
cargo run --release -- --xess --oidn  # + OIDN pre-denoise at the dynamic render res (N cycles off/pre/post)
cargo run --release -- --xess --oidn-post  # + OIDN post-denoise on the upscaled window-res frame (A/B lever)
cargo run --release -- --xess --no-adaptive  # XeSS without the adaptive shading rate (uniform per-pixel shading)
cargo run --release -- --xess --xess-autoexposure  # XeSS computes exposure internally (A/B lever)
cargo run --release -- --check-fsr    # headless: FSR signal-split/encoding/MV/provider-pick contract self-test (no GPU, no DLL)
                                      # — plus, since 2026-08-11, gate 8: the SECOND FidelityFX
                                      # generation (SDK 1.1.4, src/ffx_fsr3.rs). Two FFX
                                      # generations coexist and do NOT compete: everything else in
                                      # this file is ffx-api v2.3.0 (signed prebuilt DX12 provider
                                      # DLLs — FSR4, Ray Regeneration, frame generation), which
                                      # cannot reach the Vulkan or Metal backends because it has NO
                                      # Vulkan backend at all (its own readme lists "Vulkan is
                                      # currently not supported in SDK" under known issues) and v2
                                      # REMOVED the FfxInterface custom-backend seam, so a Metal
                                      # implementation is not merely unwritten but unexpressible.
                                      # v1.1.4 is the last MIT-source generation with a stock
                                      # first-party ffx_vk AND a seam. Windows keeps v2.3.0.
                                      # THAT SEAM IS NOT THEORETICAL — it is what
                                      # shim/ffx_fsr3_metal.mm fills with a hand-written
                                      # `FfxInterface` against Metal, so macOS upscales through
                                      # the same SDK with no Vulkan and no vendor backend at all
                                      # (--check-fsr3). Keeping a generation that HAS the seam is
                                      # what bought a third platform.
                                      # UNLIKE EVERY OTHER SDK HERE IT IS COMPILED FROM SOURCE and
                                      # statically linked — no DLL, no fn-pointer table, no
                                      # runtime shed; the degrade is at BUILD time
                                      # (cfg(ffx_fsr3_src), set by build.rs only when BOTH halves
                                      # are present). Warn-and-skip, deliberately NOT require_nrd()'s
                                      # hard fail: NRD is the default denoiser so a tree that cannot
                                      # produce it renders undenoised silently, while FSR3 here is
                                      # opt-in and its absence costs a feature, not correctness.
                                      # TWO HALVES, ONLY ONE FETCHABLE: the SDK source
                                      # (`install-prerequisites.sh fsr3src` — 4 paths of a 189 MB
                                      # tarball, ~19 MB, gitignored) and the SPIR-V shader
                                      # permutations, which are COMMITTED (200 files / ~17 MiB,
                                      # plain git, SDKs/FidelityFX-SDK-prebuilt/) because the SDK
                                      # compiles them with FidelityFX_SC.exe, a WINDOWS-ONLY tool,
                                      # and upstream ships no prebuilt SPIR-V — vendoring the output
                                      # is what makes FSR3 buildable off Windows at all. They serve
                                      # BOTH backends: Metal transpiles the same bytes to .metallib
                                      # at build time, so no Metal shader artifact is committed.
                                      # shim/ffx_msvc_compat.h is force-included into every SDK TU
                                      # (`-include`, never a patch — the sources are FETCHED and a
                                      # patch would silently un-apply on the next fetch): it
                                      # supplies _countof/wcscpy_s/strcpy_s/sprintf_s/swprintf_s
                                      # and, load-bearing, RAISES FFX_SDK_DEFAULT_CONTEXT_SIZE from
                                      # the upstream 128 KB to 1 MB, because that constant is sized
                                      # against a 2-byte wchar_t and every non-MSVC platform's
                                      # 4-byte one bloats the private FSR3 context past it.
                                      # MEASURED: all 7 backend-neutral TUs compile unmodified
                                      # under clang on macOS with only that header; the blob
                                      # accessor links 3.4 MB of SPIR-V; libffx_fsr3.a is 4.4 MB.
                                      # Gate 8 is a PIN PAIR check in nrd.rs's GetLibraryDesc shape
                                      # — it reads the version out of the headers the objects were
                                      # COMPILED against and compares it to ffx_fsr3::PIN, so
                                      # fetching a different FFX_SRC_TAG without moving the PIN
                                      # fails loudly instead of surfacing as unattributed behaviour
                                      # — plus a read-back of the context size THROUGH the SDK's own
                                      # headers, which is what proves the force-include actually
                                      # reached them rather than trusting that a build flag is still
                                      # being passed. The pure half (the (major<<22)|(minor<<12)|patch
                                      # round-trip + field-bleed teeth) runs on EVERY platform
                                      # including Windows, which never builds the SDK: the packing
                                      # is a fact about ffx_interface.h, not about the host, and a
                                      # gate that only ran where the feature is built would stop
                                      # covering the transcription the moment someone develops on
                                      # the other OS. Touch shim/ffx_fsr3.* / shim/ffx_msvc_compat.h
                                      # / build.rs's build_ffx_fsr3 / src/ffx_fsr3.rs → run
                                      # --check-fsr with the SDK present AND with it moved aside
                                      # (the degrade is the half nothing else covers)
cargo run --release -- --fsr          # force-start the upscaler chain at FSR4 + Ray Regeneration
                                      # (K toggles; RDNA4 only — elsewhere the chain falls through
                                      # XeSS to FSR 3.1 upscale-only, cross-vendor; also flips the
                                      # default adapter preference to AMD)
cargo run --release -- --fsr4         # --fsr, but the level is REQUIRED, not merely force-started: a
                                      # chain fall-through (no RDNA4 / no Ray Regeneration provider /
                                      # wrong adapter) is a HARD ERROR — exit 2 with the probe's reason
                                      # and the flags worth trying (--fsr3, --prefer-amd, --fsr). The
                                      # one non-fallback in the codebase; being told IS the feature
cargo run --release -- --fsr3         # force-start the chain at the FSR 3.1 upscale-only level even
                                      # where FSR4+RR exists (A/B lever; no 3.1 provider = loud line
                                      # + plain, never a silent un-force)
cargo run --release -- --fsr3 --fg    # FRAME GENERATION, ffx family (W4 leg 1 of 3; legs 2-3 =
                                      # raw-NGX DLSS-G for DLSS sessions, XeSS-FG+XeLL for Intel
                                      # XeSS sessions — all three live). DEFAULT ON since
                                      # 2026-07-24 (--no-fg is the kill lever; --fg spells the
                                      # default; a defaulted fg under --quinlight disarms with a
                                      # loud line instead of the explicit pair's exit 2 — the
                                      # Opts::fg_explicit / mode_explicit pattern).
                                      # Exposed in the settings menu as the Upscaler page's
                                      # frame-generation row (restart-tier; the file drives the
                                      # DEFAULT arm only — never fg_explicit, so a menu click
                                      # can't make --quinlight fatal).
                                      # In an
                                      # FSR session (FSR4-RR or FSR3 wired) the swapchain is
                                      # WRAPPED by the FidelityFX frame-interpolation proxy at
                                      # creation (d3d12::SwapWrap — the wrap sits between
                                      # colour-space negotiation and RTV creation because
                                      # GetBuffer on the proxy returns the PROXY's backbuffers;
                                      # a PQ declare is re-asserted through the proxy, which ffx
                                      # supports — the FI swapchain reads the transfer function
                                      # off the chain's declared colour space, and the display
                                      # probe's real min/max nits ride the dispatch under
                                      # Hdr10 only) and ONE generated frame
                                      # is inserted per rendered frame (measured 4090, THE
                                      # WORLD, DXR->FSR3 vsync: 96 rendered -> ~195 presented
                                      # fps, the exact-halving pacing signature; same stack live
                                      # on the B70 and the AMD iGPU — AMD interpolation on Intel
                                      # silicon is the cross-vendor demo). The FG provider ships
                                      # in a DIFFERENT sample dir than the loader
                                      # (--fg-path/FRUSTRACER_FG_PATH; ffxshim_preload_dir skips
                                      # basenames already in the module list, so the primary
                                      # --ffx-path stays authoritative). fsr::pick_fg_version:
                                      # an FSR4 session prefers the 4.x ML frame generation,
                                      # everything else the 3.1 interpolation, other major =
                                      # fallback (the enumeration is device-filtered; the 4.0.1
                                      # provider DLL enumerates only "3.1.6" on non-RDNA4 —
                                      # RDNA4-gating observed live), never id 0 — gated in
                                      # --check-fsr. Per-frame contract: frame_id advances by
                                      # EXACTLY 1 (any other delta resets interpolation history
                                      # by ffx contract); the six FSR present arms record a
                                      # PrepareV2 dispatch (reversed-Z clip depth + the MV plane
                                      # with the SAME mv_scale their upscale dispatch uses — trio
                                      # = pixels (1,1), RR plane = UV-deltas (rw,rh); one MV
                                      # convention per session by construction) and configure the
                                      # FI swapchain live; fullscreen_to_backbuffer's HANDSHAKE
                                      # covers everything else — any frame presented WITHOUT a
                                      # prepare (plain arms, SPACE mode switches, the pause-menu
                                      # present_again hold) finds `prepared` unset and configures
                                      # the proxy DISABLED first (idempotent via `live`), so
                                      # pacing never runs against stale motion.
                                      # THE MODE-SWITCH STRADDLE (2026-07-31 — the AMD
                                      # mode-cycle-slowdown fix): carrying the prepare stream
                                      # SEAMLESSLY across a SPACE/F render-mode switch — a
                                      # reset=1 prepare + the depth/MV resource-set swap (each
                                      # arm feeds FG its OWN planes: CPU-upload vs
                                      # wavefront-pack vs DXR-pack) + a frame-time cadence jump
                                      # (66 -> ~5 ms in the trace that caught it), generation
                                      # enabled throughout — wedges the AMD provider's pacing
                                      # into a MASSIVE persistent slowdown after a few SPACE
                                      # laps (R9700, THE WORLD; NVIDIA/Intel never — they run
                                      # different FG families). NOT VRAM: measured 8.4/31.7 GB
                                      # while slow (the mode: vram line exists from this hunt).
                                      # Diagnosed by elimination, each arm user-measured:
                                      # --no-fg clean, F11 resize CURES (context rebuild), K
                                      # plain-toggle round trip CURES (disable/enable configures
                                      # with NO rebuild), FR_FG_CYCLE=recreate prevents. So
                                      # GpuContext::fg_mode_switch (fired from main.rs's
                                      # landed-switch hook only — a refused press straddles
                                      # nothing) DEFAULTS to the cheapest cure as prevention:
                                      # skip the next prepare, so the funnel hands the FI proxy
                                      # exactly ONE disabled passthrough present at the seam
                                      # (the K sequence compressed to a frame; frame_id
                                      # deliberately does not advance — the disable configure
                                      # reuses the last id, bit-identical to the K path).
                                      # FR_FG_CYCLE=off restores the carry-across repro arm,
                                      # =recreate is the heavy A/B (effect-context rebuild, the
                                      # resize straddle — also proven curative). Instruments
                                      # from the hunt, all shipped: the always-on
                                      # `fg: interpolation paused/resumed` transition lines
                                      # (pause counts + frame_id), FR_FG_TRACE=1 (reset
                                      # prepares + resource-set-swap lines), `mode: vram` per
                                      # switch, and per-tracer construction vram lines. NGX and
                                      # XeSS-FG deliberately keep their existing seam handling —
                                      # the wedge is ffx-family-only as measured. Resize: pending
                                      # paced presents retire, the display-size-bound FG effect
                                      # context rebuilds, the swapchain context survives
                                      # (ResizeBuffers forwards). Teardown: GpuContext::drop
                                      # waits presents with the queue live; `fg` is declared
                                      # AFTER `d3d` so the proxy refs release before the
                                      # swapchain context destroys the proxy. THE HUD RIDES THE
                                      # PROXY'S UI COMPOSITION since 2026-08-08 (the v1 baked-
                                      # pre-present known-accept — HUD warped by scene motion on
                                      # every generated frame, visibly JUMPING because this HUD
                                      # is not static — is retired): on interpolating presents
                                      # the funnel renders the HUD through hud.hlsl into a
                                      # display-space premul target (hud::HudFi — RGBA16F under
                                      # Hdr10, RGBA8 under SDR/Sdr10; hud.hlsl stays the single
                                      # encoder) and registers it with the FI swapchain
                                      # (FgSwapchain::register_ui — premul + INTERNAL_UI_DOUBLE_
                                      # BUFFERING, the pacing-thread race answer), which
                                      # composites it onto BOTH pair halves AFTER interpolation;
                                      # the baked backbuffer draw is skipped only while the
                                      # registration is LIVE (ui_reg — a failed configure sheds
                                      # loudly once and the same frame still bakes, zero HUD-less
                                      # frames by construction), and every disable path (K
                                      # toggle, holds, mode-switch straddle, resize — which nulls
                                      # the registration BEFORE the window-sized target drops,
                                      # the surviving-swapchain-context dangling-pointer rule; a
                                      # FAILED unregister there LEAKS the old target rather than
                                      # dangle it — and teardown via fg_disable_now, which
                                      # unregisters ahead of its ctx/live early-returns: the
                                      # registration lives on the swapchain context, which
                                      # outlives the effect ctx) drives the registration
                                      # to null. FR_FG_TRACE=1 prints the register/unregister
                                      # transitions. --waveviz stays baked (interpolated with the
                                      # scene — the remaining overlay accept). Known-accepts v1:
                                      # XeSS/plain
                                      # sessions wrap but present passthrough (their FG families
                                      # are legs 2-3); latency untouched (W5). FG COMPOSES
                                      # WITH --quinlight (2026-08-01): the family follows the
                                      # session, the quin present arms carry the per-frame
                                      # contract (gpu/mod.rs::quin_fg_tail — NGX interpolates
                                      # the FUSED image via ngxfg_target, ffx FI prepares from
                                      # the planes actually FED — the FSR4-RR pair when wired,
                                      # else the XeSS trio, NEVER a shared FSR3's own stale
                                      # planes — and XeSS-FG tags per present; VERIFIED live
                                      # 4090: NGX pair-present over [dlss-rr+fsr3+xess], ffx
                                      # 3.1.6 generating over [fsr3+xess] under --no-dlss).
                                      # Headless (--check*/--spin)
                                      # never consults it. Vendored: the v2.3.0 framegeneration
                                      # headers (FG kit 4.0.1 + FI swapchain 3.1.7) under
                                      # SDKs/fidelityfx-sdk/framegeneration/. Touch the shim FG
                                      # block/ffx.rs FG wrappers/fg_prepare/the wrap hook -> run
                                      # --check, --check-fsr, --check-gpu, then the interactive
                                      # smoke on 4090 + B70 (fg lines + the cadence-halving test)
cargo run --release -- --fg           # FRAME GENERATION, DLSS family (W4 leg 2): in a DLSS
                                      # session (the flagless NVIDIA default) fg — ON BY DEFAULT
                                      # — arms RAW-NGX DLSS-G, the ONE DLSS FG backend since the
                                      # Streamline retirement (the SL DLSS-G fallback — the
                                      # declines-to-insert open issue that also rejected scRGB —
                                      # is DELETED with the interposer; a build without the DLSS
                                      # SDK has no DLSS at all, RR included: one loud line, the
                                      # chain falls to FSR4/XeSS/FSR3):
                                      # RAW NGX (the DLSS SDK present at build —
                                      # FRUSTRACER_DLSS_SDK, default
                                      # ..\quinlight-player\SDKs\DLSS-SDK; never committed,
                                      # build.rs cfg(dlss_ngx) + stages nvngx_dlssg.dll AND
                                      # nvngx_dlssd.dll — one SDK, one gate, both features) —
                                      # VERIFIED GENERATING on the 4090: shim/dlssg_shim.cpp
                                      # (the quinlight-player blueprint, adapted with REAL
                                      # camera data) drives NVSDK_NGX_Feature_FrameGeneration
                                      # directly; the feature retains the previous rr.output
                                      # internally, one evaluate per frame writes the
                                      # in-between frame into fg_n.out, and ngxfg_tail
                                      # PAIR-PRESENTS: tonemap(interp) -> present_mid (Close+
                                      # Execute+Present+Reset on the same slot allocator, the
                                      # split_frame legality) -> tonemap(real) -> end_frame.
                                      # Under vsync the two presents land a vblank apart =
                                      # the pacing (measured: rendered 186 -> 93 fps while
                                      # presents hold ~174/s — the exact-halving signature).
                                      # NO handshake needed (nothing generates behind our
                                      # back), DLSS-RR runs in the SAME session (both raw NGX,
                                      # one refcounted init — shim/ngx_shared), and the
                                      # swapchain format is IRRELEVANT (no swapchain policing —
                                      # there is no swapchain hook; NGX sees only internal fp16
                                      # textures). THREE TRAPS, all
                                      # measured: (1) [SL-era, now moot — the init is
                                      # unconditionally refcounted through ngx_shared since
                                      # both consumers are ours] NGX could already be
                                      # initialized in-process by Streamline; two differently-
                                      # keyed inits on one device silently break each other;
                                      # (2) a null app-data
                                      # path fails init with 0xBAD0000F FAIL_UnableToWrite-
                                      # ToAppDataPath — pass %LOCALAPPDATA%\frustracer\ngx;
                                      # (3) motionVectorsInvalidValue must be FLT_MAX, not 0
                                      # (0 tags every static pixel invalid — the quinlight
                                      # lesson). THREE MORE, found chasing the DamagedHelmet
                                      # sky-reflection swim (generated frames only) — the
                                      # common root: quinlight's inputs were ZERO MVs, zero
                                      # jitter, and a synthetic [0,1] luma-depth, so NOTHING
                                      # motion-dependent in the blueprint was ever validated;
                                      # treat every "quinlight-settled" constant that way.
                                      # THAT ROOT BIT AGAIN on 2026-07-26 — TRAP 9, JITTER
                                      # SIGN: the evaluate was handed the NEGATED sample
                                      # offset, reasoned by analogy from Streamline's RR
                                      # (which does want it negated) on "same NGX family, one
                                      # sign". RAW NGX WANTS IT AS IS. quinlight's jitter was
                                      # (0,0), so every sign is identical there and the
                                      # blueprint could never disagree. A sign error misplaces
                                      # content by TWICE the jitter (~1 px) — invisible on
                                      # diffuse geometry, BLATANT on a small ultra-bright
                                      # specular highlight (the sun off DamagedHelmet's metal:
                                      # ~44,000 radiance against a ~1.0 scene turns 1 px into
                                      # a strobe). It predates the c66417d "swim FIXED"
                                      # commit — that binary reproduces it — so it was never a
                                      # regression, just never caught: the swim fix addressed
                                      # reflections DRAGGING, this is the highlight JUMPING.
                                      # Now `raw` by default; FR_NGXFG_JITTER=neg restores it.
                                      # HOW IT WAS FOUND, because the method transfers: every
                                      # environmental variable was eliminated by measurement
                                      # (resolution, frame rate, the resize path, the
                                      # virtual-image MVs, PAIR_BACKBUFFERS, the scene, and a
                                      # c66417d-era build), and then the SAME FRAME through
                                      # the ffx FI interpolator came back CLEAN — which
                                      # localized it to our NGX inputs and left the FR_NGXFG_*
                                      # levers to walk it down in two runs. A cross-vendor
                                      # A/B beats another mechanism hypothesis: three
                                      # plausible ones (DestroyParameters, the virtual-MV
                                      # blend, the sky-reflection distance) each explained the
                                      # symptom and each measured wrong.
                                      # (4) DEPTH: the snippet's Depth slot has DLSS-SR's
                                      # contract — a [0,1] buffer CONSISTENT WITH THE SUPPLIED
                                      # MATRICES — while RR's plane holds unbounded linear
                                      # view-Z (RR reads it via the LINEAR-depth tag, a
                                      # different contract). (5) MVEC SCALE: DLSSG.MvecScale
                                      # converts stored MVs to PIXELS — settled from
                                      # dlssg-to-fsr3, which hands it STRAIGHT to FSR3's
                                      # motionVectorScale across shipped SL titles; the SDK
                                      # header's "[-1,1]" comment is stale, and the
                                      # quinlight-era {1/rend} starved the snippet of geometry
                                      # motion ~2000× (why the depth fix alone changed nothing
                                      # visible). Our MV plane stores pixels ⇒ mv_scale {1,1}.
                                      # (6) REFLECTION MVs: surface MVs describe the SURFACE,
                                      # but a mirror pixel's CONTENT is the reflection — a
                                      # VIRTUAL IMAGE at path depth t_surf + t_refl (planar
                                      # unfold along the primary ray; a MISSED reflection is
                                      # the SKY, i.e. a virtual image at INFINITY with EXACTLY
                                      # zero translation parallax — the "reflection drifts
                                      # opposite the surface" strafe observation), so
                                      # warping with surface MVs drags the reflection with the
                                      # helmet on every generated frame. Both conversions run
                                      # in ONE fused pass, gpu/ngxfg_guides.rs (fxc cs_5_0,
                                      # the bloom no-DXC precedent — records inside
                                      # ngxfg_dispatch, the one site all three RR arms share):
                                      # clip depth d = A + B/z (the EXACT perspective_lh
                                      # z-mapping — deliberately NOT xess::view_z_to_clip_
                                      # depth, which is REVERSED-Z and inconsistent with the
                                      # matrices NGX gets) + an FG-ONLY MV plane
                                      # lerp(mv_surface, mv_virtual, w), w = lum(spec_alb)/
                                      # (lum(diff_alb)+lum(spec_alb)) damped over roughness
                                      # ROUGH_LO..ROUGH_HI (metal helmet ⇒ w≈1; RR's own MV
                                      # plane untouched — RR is trained for surface MVs + the
                                      # spec-hit guide; spec_hit_t is the reflection distance
                                      # source, 0 = no ray ⇒ passthrough). ONE LANE, TWO JOBS:
                                      # the pack clamps a MISSED reflection to CAM_FAR because
                                      # that lane's OTHER consumer is RR's depth delta, which
                                      # wants far — but "far" is a LIE as a reflection
                                      # distance (2*diag ≈ 138 world units is not infinity),
                                      # and feeding it the point form gave the sky real
                                      # parallax. The kernel now takes the analytic LIMIT for
                                      # t_r >= cam_far: as t_r → ∞ the virtual point becomes a
                                      # DIRECTION, so it projects with the translation column
                                      # dropped (w = 0) ⇒ rotation-only, exactly right.
                                      # True RADIANCE-
                                      # weighted w needs a dd/ds/ind_s-style capture in DLSS
                                      # sessions (the FLAG_FSR_SIG precedent) — the follow-on
                                      # if albedo-weighting leaves residue. ROUND 3 of the same
                                      # pass (the night-swarm strobe fix): fireflies move every
                                      # rendered frame with NO MVs anywhere (the glow is a
                                      # color-only add after the G-buffer capture), so FG warped
                                      # the bright blobs with the BACKGROUND's MV — and on
                                      # smooth/metal pixels the round-2 material-driven blend
                                      # confidently handed it the virtual-reflection MV at
                                      # exactly those pixels. Poses are closed-form, so the CPU
                                      # bakes per-firefly SCREEN-SPACE splat rows (ff_guide_rows:
                                      # cur px, prev px through the same world->prev-clip matrix,
                                      # view-Z, sigma-px, center lum — prev poses = the LAST
                                      # SUCCESSFULLY EVALUATED frame's swarm, retained beside
                                      # `primed`; a count mismatch reprojects the current pose,
                                      # camera-motion-only, never wrong-signed) and the kernel
                                      # lerps toward mv_i = prev_px - cur_px where glow luminance
                                      # dominates (w = S/(S+FF_MV_L_REF); analytic weight, never
                                      # an accum read — a 1-spp denominator would flicker the MV
                                      # plane; the exp-reject rides the fireflies +34 ms lesson
                                      # with a 1e-4 skirt so the weight is continuous at the
                                      # cut). MV constant across a splat (rigid translation —
                                      # per-pixel reprojection would contract the blob). The
                                      # table rides a root CBV (b1) on a FRAMES_IN_FLIGHT upload
                                      # ring; ffc=0 (day / --no-fireflies / lever-off) executes
                                      # the pre-round-3 kernel stream bit-identically.
                                      # FR_NGXFG_FFMV=off is the A/B (strobe returns on demand).
                                      # ROUND 4 of the same pass (the PARKED-CAMERA WATER
                                      # strobe) is round 3's shape again, not the swim's: a
                                      # MISSING MV, not a wrong one. Water's mirror normal
                                      # MOVES — ripple_normal tilts it every rendered frame
                                      # on the cloud clock (~14 deg of tilt ⇒ ~28 deg of
                                      # reflected swing, 1-3 deg per frame at 60 fps) — so the
                                      # reflected skyline slides across a surface whose
                                      # GEOMETRY is still, and every MV plane (camera motion
                                      # only) reports zero. Water was also the class most
                                      # exposed already: roughness 0.05 is below ROUGH_LO, so
                                      # it took 15-45% of the virtual-reflection MV with NO
                                      # roughness damping. The still-mirror unfold is
                                      # normal-FREE only because mirroring the reflected point
                                      # across the surface plane sends it back down the
                                      # primary ray — true while the normal holds still. The
                                      # field is closed-form, so the PREVIOUS normal is
                                      # computable, and reflecting the current content
                                      # direction off it collapses BOTH branches into one
                                      # expression: d = reflect(reflect(du, n_c), n_p).
                                      # reflect is an involution, so n_p == n_c gives d == du
                                      # EXACTLY — the finite arm reduces to org+du*(ray_t+t_r)
                                      # and the sky arm to du, i.e. the round-2/3 kernel
                                      # bit-for-bit. That identity IS the safety argument.
                                      # n_p is first order (the ripple SUBTRACTS the in-plane
                                      # gradient, so stepping the gradient back steps the
                                      # normal back); exact at dt=0, second-order otherwise
                                      # (<= ~0.8 deg), and a degenerate/horizon-crossing
                                      # reconstruction falls back to n_c = the exact
                                      # pre-round-4 unfold (coarser, never wrong-signed).
                                      # PLUS FRESNEL, scoped to ripple pixels so everything
                                      # else stays bit-identical: ls/(ld+ls) is an F0 proxy
                                      # that sits flat at ~0.15-0.45 on water while real
                                      # reflectance runs ~2% face-on to ~100% grazing, so
                                      # without it the fix applies a FRACTION of the correct
                                      # MV exactly where the sliding skyline is and HALVES the
                                      # strobe. Face-on stays low deliberately — the refracted
                                      # basin dominates there and has no MV at all, so the
                                      # near-zero surface MV is the honest answer.
                                      # PLUMBING: ripple_amp rides GBufExt.alb.w (the ONE
                                      # documented-unused lane ⇒ no stride change, and the ext
                                      # gates that skip lane 7 keep passing), reaching the
                                      # guide pass on an 8th RrResources plane delivered
                                      # through FEED_FSR_AO (u26 — same RWTexture2D<float>/
                                      # R16F, and an RR session never runs the FSR-RR kernel
                                      # that owns it; NOT an RR input, RR is never tagged with
                                      # it). The clock pairs with prev_ff beside `primed` (set
                                      # only on a SUCCESSFUL evaluate) and t_prev defaults to
                                      # t_cur, NEVER 0.0 — a session minutes into its clock
                                      # would otherwise inject a huge bogus delta on the first
                                      # armed frame, the confident-wrong-MV failure this whole
                                      # pass exists to avoid.
                                      # FR_NGXFG_RIPPLEMV=off is the A/B.
                                      # THE dt-CONFIDENCE FADE (2026-08-05 — the 8K
                                      # water-glitch fix): the ripple clock advances by REAL
                                      # WALL TIME — at 8K's ~6-10 fps each frame steps the
                                      # field 100-165 ms. MEASURED (the ripple_probe cargo
                                      # test, world scale): the field stays COHERENT there
                                      # (normal swing mean 1.4 / max 3.5 deg at 150 ms — the
                                      # waves are slow and wide, the user's observation) and
                                      # the first-order reconstruction stays near-EXACT (err
                                      # <= 0.06 deg out to 250 ms) — the first-draft
                                      # "value noise decorrelates" story was WRONG. What
                                      # explodes is the TRUE motion's MAGNITUDE: the
                                      # reflected image moves 200-550 px/frame at 8K pixel
                                      # density, water's rough 0.05 sits below ROUGH_LO
                                      # (zero damping) with the grazing Fresnel driving wgt
                                      # to ~1, NO clamp anywhere — and handing NGX accurate
                                      # MVs of that size measured as severe glitching
                                      # (game MVs are normally tens of px, and the ripple
                                      # field also STRETCHES, which a warp at that scale
                                      # tears on); invisible at 1080p/high fps (dt ~25x
                                      # smaller, pixels 4x coarser). OPEN: that glitchy
                                      # unfaded arm was only ever observed WITH the f16
                                      # sky-compare bug below live — FR_NGXFG_RIPPLEDT=off
                                      # at 8K on the fixed build is the pending A/B that
                                      # decides whether the fade can narrow (e.g. become
                                      # magnitude-aware). The fade scales the GRADIENT DELTA by
                                      # 1 - smoothstep(RIPPLE_DT_LO=1/30, RIPPLE_DT_HI=0.1,
                                      # dt), so past HI the path collapses BITWISE onto the
                                      # still-mirror unfold (gd = 0 IS the involution
                                      # identity) — coarser, never wrong-signed: better a
                                      # half-cadence shimmer than a torn 500-px warp.
                                      # Multiplying by the
                                      # in-window w == 1.0 is the bitwise identity, so the
                                      # validated 60 fps regime is structurally untouched. It
                                      # also covers the stale-prev_clock seam (an FG res-move
                                      # skip window leaves a multi-frame delta on the first
                                      # paired frame). FR_NGXFG_RIPPLEDT=off is the unfaded
                                      # repro arm (kernel-side only; the Rust twins always
                                      # fade). ALSO FIXED THE SAME DAY, found on the way: the
                                      # kernel's sky test `t_r >= cam_far` compared the
                                      # f16-STORED spec_hit plane against the EXACT f32 far —
                                      # on THE WORLD (2*diag ~ 138.56, which f16 rounds DOWN
                                      # to 138.5) the sky branch could never fire, so every
                                      # water sky-reflection took the finite-point branch and
                                      # got false translation parallax (the round-2 class,
                                      # reopened by the wire format). GuideParams.cam_far now
                                      # carries far's f16 FLOOR (ngxfg_guides::wire_cam_far —
                                      # the floor, not RNE, because a typed-UAV R16F store
                                      # has round-toward-zero latitude the CPU arm's st16
                                      # RNE does not); known-accept: a genuine hit within one
                                      # f16 quantum of far classifies as sky (~zero parallax
                                      # at 2*diag anyway). Rungholt STANDALONE never had this
                                      # bug (diag 10 -> far 20, f16-exact).
                                      # ROUND 5 of the same pass (2026-08-10, v1.5.4 — the
                                      # MAX-STRAFE curved-mirror SHRED, the user's "smearing
                                      # at max strafing speed past the helmet"; diagnosed
                                      # end-to-end by the AI QA Lab with DESKTOP screen
                                      # captures, since P/frqa screenshots read rr.output and
                                      # never see a generated frame): the round-2 unfold is
                                      # FLAT-mirror — on the helmet's convex dome the sun
                                      # glint's true motion rides the SURFACE (the sun images
                                      # at ~R/2 behind the shell — FRD's v1.5.1 insight, never
                                      # ported here), so the sky arm handed NGX a rotation-
                                      # only ~zero-translation MV that at max strafe was wrong
                                      # by HUNDREDS of px/frame, and the smooth-metal blend
                                      # weight (w≈1) applied it at full strength: every
                                      # generated frame warped into mosaic tears
                                      # (FR_NGXFG_SHOW=interp captures; =real clean at the
                                      # same speed = the frame-alternating band the user
                                      # screenshots at fg x2). Fix: FRD's v1.5.1/.3 curvature
                                      # ported — κ from de-biased res-scaled central diffs of
                                      # the nrough plane (fg_vm_curv_at/fg_vm_axis, literals
                                      # in lockstep with frd_common's twins; offsets ±s/±2s,
                                      # s = round(h/540)) and t_v = t_r/(1+2κ·t_r) BEFORE the
                                      # sky test and the unfold — a curved dome's "sky"
                                      # reflection re-routes to a near virtual image whose MV
                                      # rides the surface; Δn ≡ 0 keeps the flat path bitwise
                                      # (planar mirrors/still water untouched). The Rust twin
                                      # is ngxfg_guides::curved_unfold_dist — a thin
                                      # composition over frd::oracle::{vm_kappa,virtual_dist},
                                      # so the two engines share ONE κ family (the unfold
                                      # cross-pin closed from the κ side). FR_NGXFG_CURV=off
                                      # is the repro arm (the shred returns on demand — the
                                      # curv lane rides GuideParams' ex-_pad2). Gated in the
                                      # ngxfg-guides self-test: the (d2) composed pin — under
                                      # the same strafe the curved unfold must land within
                                      # 0.15·mv_surf of the SURFACE reprojection while the
                                      # flat arm's ~0-px answer (correct on planes, the teeth)
                                      # sits a whole surface-MV away, heavy-κ t_v < 0.25·z,
                                      # constant-field t_r bitwise.
                                      # Gate teeth: a
                                      # parked-camera probe scans dt INSIDE the fade window
                                      # across 8 base phases until >= 2 px (loud if never —
                                      # anti-vacuity; the old single-phase scan only reached
                                      # its 5 px bar at deltas the fade now rejects; fires at
                                      # dt=0.033s, 2.82 px vs still-mirror 0.0000), the
                                      # oracle is an INVERSE ROUND-TRIP rather than a
                                      # re-derivation, and the pre-fix answer must FAIL the
                                      # bound. Three gate families joined with the fade
                                      # (self_test r-h/r-i/r-j): the collapse pins (dt >= HI
                                      # bit-equals the still-mirror twin at 6 fps/4 fps/1 s,
                                      # weight anchors exact at both edges + midpoint 0.5 +
                                      # monotone, no pop crossing either edge, mid-window
                                      # anti-vacuity), the RECONSTRUCTION-FIDELITY pin that
                                      # never existed — ripple_prev_normal vs
                                      # shade::ripple_normal actually evaluated at t_prev
                                      # (the function it claims to reconstruct), <= 2 deg
                                      # over a grid x phases x 120/60/30 fps deltas — and the
                                      # f16 sky-compare pins (floor <= every possible stored
                                      # sentinel at both RNE-down and RNE-up fars; strafe
                                      # behavioral: fixed threshold translation-invariant,
                                      # exact-f32 threshold must FAIL). The pack lane is gated in
                                      # --check-gpu/--check-dxr and PROVEN NON-VACUOUS on
                                      # rungholt (water px 1552) — the default and san-miguel
                                      # probe poses see NO water, so that gate alone would
                                      # have passed while proving nothing.
                                      # FG-ONLY: RR's MV plane, ffx FI and XeSS-FG unchanged
                                      # (their zero-MV glow drag is a documented accept); firefly
                                      # SPECULAR highlights still ride surface/virtual MVs
                                      # (half-vector geometry, out of scope). Gated in --check as
                                      # `ngxfg-guides` (clip-depth matrix-consistency sweep;
                                      # virtual-MV: static-camera zero, t_r=0 continuity vs
                                      # CamBasis::project itself, the strafe reflected-sky
                                      # collapse, weight anchors; round 3: off arms exact +
                                      # empty-table blend bit-identity, the moving-firefly gate
                                      # with anti-vacuity and TEETH pins — the pass-through
                                      # surface MV must FAIL the bound — occlusion/behind-camera
                                      # drops, the weight-continuity skirt pin, and the
                                      # projection-route/sigma-lum anchors vs CamBasis::project
                                      # and the shipped fireflies::glow). THE STRAFE GATE'S OWN
                                      # LESSON: it was RELATIVE (`mv_virt <= 0.05 * mv_surf`)
                                      # where the correct answer is EXACTLY ZERO, so any
                                      # percentage of a large surface MV passed; and it ran at
                                      # far = 5000 while the renderer ships far = 2*diag —
                                      # 36x more distant, making "reflection at far"
                                      # impersonate infinity far better in the gate than in
                                      # the product. It is now ABSOLUTE, sweeps
                                      # production-scale far values, and carries a TEETH pin
                                      # (the pre-fix point form must blow the bound). Note
                                      # this is the mirror of the --spp image A/B lesson,
                                      # where an ABSOLUTE limit was the wrong shape: pick the
                                      # form from what the true value is, not by habit. A SEVENTH trap, structural:
                                      # pair-present consumes TWO backbuffers per frame, so at
                                      # the shipped BACKBUFFERS=3 a buffer came back around
                                      # 1.5 frames later — under vsync with the DXGI present
                                      # queue full that re-renders into a buffer still queued
                                      # for scanout (stale-frame flicker; a timing race no
                                      # debug layer flags). Raw-NGX sessions now create the
                                      # swapchain at d3d12::PAIR_BACKBUFFERS=6, restoring the
                                      # exact 3-buffers-per-present ratio every other session
                                      # has (quinlight's pair-present had its own fence ring —
                                      # PAIR_PRESENT_FENCES — which the port had dropped).
                                      # THE INPUT CURVE (ngxfg_guides::TonePass, DEFAULT ON as
                                      # `reinhard`, 2026-07-31 — the sun-strobe fix): NGX's
                                      # flow estimator needs a DISPLAY-CURVE-shaped input, not
                                      # scene-referred linear radiance (sun disc ~810 vs scene
                                      # ~1) — diagnosed by elimination (camera/matrix/jitter/
                                      # depth/MV/HDR-declaration/magnitude/clouds/bloom/DRS all
                                      # measured out) with the ffx FI interpolator CLEAN on
                                      # identical content, then confirmed by the arms: `scale`
                                      # (ratios preserved) double-ghosts under rotation, `log`
                                      # (bounded but midtones crushed to 0.06) ghosts + bands,
                                      # `reinhard` (v/(1+v), a real tonemap operator putting
                                      # 1.0 at 0.5) correct parked AND rotating. So every --fg
                                      # session compresses rr.output into a scratch, hands THAT
                                      # to NGX, and expands the interpolated output in place —
                                      # presentation untouched, the REAL pair half bit-identical
                                      # in every mode. KNOWN-ACCEPT: the inverse is
                                      # ill-conditioned near the ceiling (one f16 quantum ≈ 490
                                      # radiance), which can band the outer bloom edge on
                                      # GENERATED frames only; the tracked follow-on (hand FG a
                                      # genuinely display-referred image + present through a
                                      # matching curve — no inverse pass) is blocked on
                                      # fullscreen_to_backbuffer render-target parameterization
                                      # and the bloom pyramid running on linear input. A second
                                      # SURVIVING artifact is NOT ours: straight piecewise-
                                      # linear banding in the smooth aureole under camera
                                      # motion, settling parked — NGX's own optical flow hitting
                                      # the aperture problem in a textureless gradient (per-
                                      # block flow drifts independently; would band along
                                      # curved iso-radiance contours if it were our round-trip
                                      # quantization, and it doesn't).
                                      # Empirical-settling + ELIMINATION env levers (the
                                      # FR_ABL read-only-probe idiom, loud on departure):
                                      # FR_NGXFG_TONEMAP=off|scale|reinhard|log (off = raw
                                      # linear to the evaluate — the sun strobe returns on
                                      # demand; scale/log = the diagnostic arms above),
                                      # FR_NGXFG_DEPTH=linear, FR_NGXFG_RMV=off (surface MVs —
                                      # brings the reflection swim back on demand),
                                      # FR_NGXFG_JITTER=0|raw, FR_NGXFG_MV=norm|neg|normneg
                                      # (scale/polarity walks), FR_NGXFG_FFMV=off (surface MVs
                                      # at firefly glow pixels — the night-swarm strobe A/B,
                                      # see round 3 above), FR_NGXFG_RIPPLEMV=off (a
                                      # STILL-mirror unfold on water — the parked-camera
                                      # water strobe A/B, see round 4 below),
                                      # FR_NGXFG_CAM=identity
                                      # (quinlight's proven identity-camera block — isolates
                                      # our matrix plumbing), FR_NGXFG_MAT=col (column-major
                                      # matrices — the majority was never validated: quinlight's
                                      # identities are transpose-invariant), FR_NGXFG_SHOW=
                                      # interp|real (present ONE side for both halves of the
                                      # pair: interp = inspect generated frames at full rate —
                                      # non-generating frames fall back to the real frame, so
                                      # a failed/skipped evaluate never re-presents a stale
                                      # out-texture; real = nothing NGX-made on screen, the
                                      # present-path null test — pacing identical in all
                                      # modes), FR_NGXFG_PACE=1 (per-frame pacing probe:
                                      # backbuffer indices of both pair halves + DXGI frame
                                      # statistics per rendered frame — diff pacing between
                                      # arms from a log; FOREGROUND window only, DWM retires
                                      # an occluded window's presents unthrottled). An
                                      # unrecognized lever value is LOUD and takes
                                      # the default (a silent no-op A/B walk is the failure
                                      # mode the levers exist to prevent).
                                      # Reset frames evaluate (to seed history) but
                                      # present real-only (`primed`); the feature is fixed-res
                                      # by creation (lazy-created at the frame's render res)
                                      # but FOLLOWS a render-res MOVE: a moved res that HOLDS
                                      # FG_RECREATE_STABLE=8 consecutive dispatches drains the
                                      # queue and FEATURE-SCOPE-recreates at it
                                      # (shim frdlssg_recreate: ReleaseFeature + CreateFeature
                                      # ONLY, then the guide planes re-ensure) — so FG survives
                                      # SPACE/F mode cycles (the CPU arm's quality-2/3 res vs
                                      # the GPU arms' native; the CPU renderer fills the same
                                      # MV/depth/guide planes, so CPU-rendered frames generate
                                      # too — the XeSS-FG composes-with---cpu precedent).
                                      # TRAP 8 [SL-era mechanism, discipline KEPT]: the
                                      # recreate must NEVER route through frdlssg_destroy
                                      # mid-session — destroy tore at the GetCapability-
                                      # Parameters map the in-process Streamline SHARED and
                                      # every subsequent RR evaluate failed 0xBAD00004
                                      # FeatureNotFound. The sharer today is our OWN DLSSD
                                      # session (same NGX-owned map), so the never-Destroy-
                                      # Parameters / feature-scoped-recreate rules still hold,
                                      # now by ownership discipline instead of SL archaeology.
                                      # A --lock-res dynamic RAMP changes res per
                                      # frame, never qualifies, and skips with a note (the
                                      # recreate-storm guard; a completed DRS step holds the
                                      # 90-frame dwell = one recreate per adoption).
                                      # RESIZE KEEPS THE FEATURE ALIVE and lets the
                                      # res-follow recreate adopt the new size — it does NOT
                                      # destroy (2026-07-26; FR_FG_RESIZE_DESTROY=1 restores
                                      # the old path for A/B). The SL-era crash that settled
                                      # this (destroy on an 8K resize -> the shared NGX state
                                      # torn under Streamline's live RR -> an AV inside _nvngx
                                      # surfacing as Present E_ABORT -> the session shed RR,
                                      # shed DXR, panicked at the plain present; NOT VRAM —
                                      # reproduced at 418 MB as readily as 5.8 GB) is
                                      # structurally impossible since the retirement, but the
                                      # cheap keep-alive shape stays — the DLSSD session
                                      # shares the same map. frdlssg_recreate therefore takes
                                      # DISPLAY dims as well as render dims — a window resize
                                      # moves both, and rend-only rebuilt the feature as old
                                      # display x new render, which NGX rejects at evaluate
                                      # with 0xBAD00005 FAIL_InvalidParameter.
                                      # Known-accepts: latency +~half frame
                                      # (the interpolation cost, W5 owns measurement). NOTE
                                      # the HUD is NOT among them on THIS path (it is on the
                                      # ffx one): NGX is handed `color: rr.output` — linear,
                                      # pre-tonemap, PRE-HUD — and both pair halves composite
                                      # the HUD themselves inside fullscreen_to_backbuffer, so
                                      # the UI is never interpolated and pHudless/pUI being
                                      # null costs nothing here.
                                      # RETIRED WITH STREAMLINE: the SL DLSS-G fallback
                                      # backend (contract-complete — Reflex/PCL markers, the
                                      # funnel mode-off handshake, the type-0 depth dual-tag —
                                      # yet SL's closed dlfg present layer DECLINED TO INSERT
                                      # on the dev box with every verifiable element green;
                                      # the elimination record, the FR_DLSSG_NO_RR isolate
                                      # lever, and the never-resolved open issue live in git
                                      # history at the SL-retirement commits). Its deletion is
                                      # half of why SL retired at all: the interposer's only
                                      # other job was evaluating RR, which the raw DLSSD shim
                                      # now does. Gates: --check, --check-dlss, --check-gpu
cargo run --release -- --prefer-intel # FRAME GENERATION, XeSS family (W4 leg 3 —
                                      # VERIFIED GENERATING on the B70; fg is ON BY DEFAULT, so
                                      # a flagless Arc session takes this leg — --no-fg opts
                                      # out): an Intel XeSS session
                                      # (the flagless Arc default) wraps its swapchain with the
                                      # XeSS-FG proxy (src/xess_fg.rs — libxess_fg.dll +
                                      # libxell.dll from the xess_path dir, the xess.rs
                                      # fn-table loader idiom; xefgSwapChainD3D12InitFromSwap-
                                      # Chain + GetSwapChainPtr at the same d3d12::SwapWrap
                                      # hook the ffx family uses). XeLL is created, sleep-moded
                                      # low-latency, and LINKED at wrap (a hard xefg
                                      # requirement); the three XeSS present arms tag depth +
                                      # MV (the XeSS trio planes, NPSR state) + row-major
                                      # frame constants per presentId (+1 per prepared frame),
                                      # fire all six XeLL markers (sleep + sim/renderSubmit at
                                      # prepare, present pair around Execute+Present), and the
                                      # funnel handshake disables generation on any unprepared
                                      # present (READ-not-consume, the DLSS-G shape). THE
                                      # OWNERSHIP TRAP (measured as a silent native crash):
                                      # unlike the ffx wrap, which CONSUMES the app swapchain,
                                      # the xefg proxy DELEGATES to it — the app-side ref must
                                      # stay alive until xefgSwapChainDestroy (XefgSwapchain
                                      # holds it; released LAST in Drop). XeSS-FG REJECTED the
                                      # old scRGB fp16 swapchain (InitFromSwapChain INVALID_
                                      # ARGUMENT — measured; no HDR flag exists in its API) but
                                      # ACCEPTS 10-bit (VERIFIED on the B70: R10G10B10A2 +
                                      # G2084 wraps and generates, gen result SUCCESS x2). Both
                                      # session defaults are now that same R10G10B10A2 format
                                      # (PQ on HDR-on, Sdr10 gamma on HDR-off — the Sdr10 wrap
                                      # is a byte-identical desc, not yet B70-smoke-verified),
                                      # so the old wrapper-forces-PQ-or-8-bit special case is
                                      # gone. If the wrap ever rejects the 10-bit chain,
                                      # D3d::with_queue rebuilds at 8-bit SDR and wraps AGAIN —
                                      # FG is why the session exists, so SDR with FG beats
                                      # 10-bit without it.
                                      # Verified on the B70 (SDK 1.2.2 + XeLL 1.2.1):
                                      # the library's own GetLastPresentStatus reports 2
                                      # frames presented per present / gen result SUCCESS, and
                                      # PresentMon shows ~174 presents/s over ~87 rendered.
                                      # Composes with --cpu (the CPU-fed XeSS arm carries the
                                      # prepare too — the biggest visual win, low source fps).
                                      # A status poll auto-disables on a negative gen result.
                                      # THE HUD RIDES A RES_UI TAG since 2026-08-08 (the ffx
                                      # family's UI-composition fix, leg-3 flavor): xefg_prepare
                                      # tags the hud::HudFi display-space premul texture (xefg's
                                      # DEFAULT alpha convention — the NOT_PREMUL init flag is
                                      # the opt-out we don't take) per present, and UI_MODE_AUTO
                                      # resolves to BACKBUFFER_UITEXTURE: the baked HUD draw
                                      # STAYS (unlike ffx, whose proxy composites both halves
                                      # from the registration) and the proxy REFINES the UI
                                      # region on generated frames from the tag — strictly
                                      # additive, a tag failure sheds the tag alone (ui_shed,
                                      # loud once — the shed also skips the HudFi pre-pass, and
                                      # a resize re-arms it: the tag is window-sized, the ffx
                                      # ui_shed rule), never FG. NOT yet B70-smoke-verified (the
                                      # owed leg-3 smoke).
                                      # Gates: --check, --check-xess, --check-gpu
                                      # --prefer-intel. Touch xess_fg.rs / the xefg_* helpers /
                                      # the XeSS arms -> run those three + the interactive B70
                                      # smoke (fg lines + last-present status x2)
cargo run --release -- --fsr --fsr-max-radiance 10  # Ray Regeneration tuning (FfxApiConfigureDenoiserKey,
                                      # applied at denoiser creation): --fsr-max-radiance (the firefly
                                      # clamp — the highest-value knob for a 1-spp path tracer),
                                      # --fsr-stability-bias, --fsr-radiance-clip-k,
                                      # --fsr-disocclusion-threshold, --fsr-normal-strength,
                                      # --fsr-kernel-relaxation. Each unset = configure nothing = the
                                      # provider's own default, so a flagless session is unchanged
```
