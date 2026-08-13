# Denoisers — NPPD, NRD, FRD, and the AI QA lab

The one pre-upscale denoiser slot: NPPD, NVIDIA NRD/ReBLUR, the from-scratch FRD, and the two halves of the QA lab (`--frd-lab` batch, `--qa` live socket) built to measure them.

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
cargo run --release -- --check-nppd   # headless: NPPD neural-denoise self-test (needs onnxruntime.dll
                                      # + the exported model; the staging math is gated DLL-free by --check)
cargo run --release -- --nppd-dump    # --check-nppd + before/after PNG dumps
cargo run --release -- --nppd         # NPPD neural denoising (J toggles; mutually exclusive with G/N;
                                      # needs SDKs\onnxruntime\bin + SDKs\nppd\nppd_small.onnx — see
                                      # tools/nppd-export, export with --fp16; --nppd-device auto|cpu|dml[:n]).
                                      # IMPLIES --xess: trace at --lock-res (default native 100%), NPPD
                                      # pre-denoises at that render res, XeSS upscales; --no-xess keeps
                                      # the standalone window-res mode (also the automatic fallback when
                                      # libxess.dll is missing)
cargo run --release -- --xess --nppd  # same session spelled explicitly; J toggles the pre-upscale slot
                                      # (takes the slot OIDN's N-cycle pre placement uses)
cargo run --release -- --gpu --xess --nrd  # NRD (ReBLUR) pre-upscale denoising — the HAND-CRAFTED
                                      # (non-neural) temporal ray reconstruction that makes the
                                      # TAA-upscalers peers of the RR engines (the quinlight
                                      # "pre-denoise their SHARED input" follow-on, built).
                                      # THE DEFAULT DENOISER (d1b315f — it briefly was not: the
                                      # 2026-08-09 Phase-E flip handed the slot to FRD and the
                                      # same week's emissive-integration campaign handed it
                                      # back, NRD measuring ahead where this content lives; the
                                      # --frd entry carries the numbers and the full order).
                                      # So --nrd holds the one denoiser slot, --frd CLAIMS it,
                                      # and --no-nrd --no-frd is the plain undenoised baseline.
                                      # This is also why build.rs's require_nrd() hard-FAILS
                                      # rather than degrading: a tree that cannot produce NRD is
                                      # a tree whose DEFAULT session silently runs undenoised.
                                      # Opts::nrd_explicit is the fg_explicit pattern —
                                      # a FILE-defaulted nrd under --nppd disarms with a loud
                                      # line while the EXPLICIT pair exits 2 (a default must
                                      # never make another flag fatal), and the "not armed"
                                      # session notes fire only for the explicit flag (the
                                      # default must not nag DLSS sessions). A missing NRD.dll
                                      # sheds loudly per session with the install hint.
                                      # THE SOURCE IS A GIT SUBMODULE and the build REQUIRES it
                                      # (2026-08-10): SDKs/NRD-src -> NVIDIA-RTX/NRD, which
                                      # vendors nothing (a URL + a SHA; each clone fetches from
                                      # NVIDIA, exactly as the retired tag-zip download did, so
                                      # the object-code-only grant is never engaged). build.rs's
                                      # require_nrd() hard-FAILS — not the DLSS block's
                                      # cargo:warning degrade — on a missing submodule (all
                                      # platforms) or a missing ARTIFACT. Rationale: NRD is the
                                      # DEFAULT denoiser, so a tree that cannot produce it is a
                                      # tree whose default session silently runs undenoised.
                                      # TWO ARTIFACTS SINCE 2026-08-11 (the Vulkan port's B4b-i),
                                      # and the owed arm this comment used to promise is PAID:
                                      # SDKs\NRD\bin\NRD.dll carrying DXIL for D3D12, and
                                      # SDKs/NRD/bin/libNRD.so carrying SPIR-V for the Vulkan
                                      # backend, both from the same submodule at the same tag with
                                      # the same encoding pins — only the shader arm differs, and
                                      # off WIN32 that is not even a choice (NRD's own
                                      # cmake_dependent_option forces DXIL/DXBC OFF, and
                                      # NRD_EMBEDS_SPIRV_SHADERS already defaults ON everywhere).
                                      # `install-prerequisites.sh nrd` builds BOTH Linux arms
                                      # (standard + perf) in ~18 s; the DLL half still is not
                                      # producible off Windows and the blocker is dxil.dll, the
                                      # Windows-only DXIL SIGNER, not CMake or MSVC.
                                      # THE CHECK IS KEYED ON THE TARGET, NOT THE HOST — the
                                      # `cfg!(windows)`-describes-the-HOST defect build_ffx_fsr3
                                      # documents, which require_nrd had too and only
                                      # accidentally: it made cross-compiling to Windows skip the
                                      # DLL check. BUT THE ARTIFACT HALF FIRES ONLY ON A NATIVE
                                      # BUILD (`HOST == TARGET`), and that is a statement rather
                                      # than an escape: the panic exists to stop a SESSION
                                      # rendering undenoised, and `cargo check --target
                                      # x86_64-pc-windows-msvc` — what tools/win-cross-check.sh
                                      # runs on a Linux box to type-check the cfg(windows) half of
                                      # this tree — produces no session and cannot produce an
                                      # NRD.dll either. Target-keying WITHOUT that guard turns
                                      # today's accidental pass into a hard panic and takes the one
                                      # tool covering the Windows half every commit; cross-builds
                                      # get a cargo:warning naming the target's artifact and its
                                      # installer (verified: the Linux->Windows cross-check prints
                                      # the NRD.dll line and still exits 0). Consequence of the
                                      # hard fail, accepted rather than discovered: a bare Linux
                                      # clone must run the installer — network plus a CMake build —
                                      # before `cargo build` works at all, CPU-tracer-only users
                                      # included, so do_nrd's missing-cmake degrade is an
                                      # UNCONDITIONAL fail (a named-only skip would leave a tree
                                      # that no longer compiles; the two move together).
                                      # Consequences, all deliberate:
                                      # .gitignore needs the `!/SDKs/NRD-src` negation (the
                                      # blanket /SDKs/* otherwise makes `git submodule add`
                                      # refuse — and NO trailing slash, which only matches an
                                      # existing directory); CI runs submodules: true + the real
                                      # installer component with the DLLs cached on the submodule
                                      # SHA; and the `nrd` component's informational-skip degrade
                                      # is GONE (a skip now means the tree does not build).
                                      # THE CLEAN-ROOM RULE LOST ITS PHYSICAL ENFORCEMENT with
                                      # the submodule — Shaders/NRD.hlsli now sits in the tree
                                      # beside our shader concat, where its absence used to make
                                      # a paste impossible, and N0/N2 would still pass with
                                      # pasted math (they compare against the oracle, the very
                                      # thing a paste replaces).
                                      # gfx::shaders::nrd_clean_room_tests
                                      # is the replacement: a cargo test scanning our four
                                      # assembled shader units for NVIDIA's distinctive entry
                                      # names, COMMENTS STRIPPED FIRST (the DispatchRaysIndex
                                      # gate's lesson — nrd_bridge.hlsl:77 legitimately NAMES
                                      # NRD_FrontEnd_PackNormalAndRoughness to say which
                                      # semantics it reimplements, and citing what you
                                      # reimplement must stay legal), with teeth pinning that the
                                      # stripper neither guts the sources nor over-strips.
                                      # NVIDIA's NRD v4.17.3, PINNED in FIVE places that must
                                      # move together — and the submodule SHA is the one that
                                      # drifts SILENTLY (a `git submodule update --remote` moves
                                      # it with no diff in any pinned constant):
                                      # install-prerequisites.bat's NRD_TAG (now only the
                                      # human-readable label the loud lines print — the SOURCE is
                                      # the submodule; the `nrd` component CMake-builds it
                                      # locally, NVIDIA shipping no prebuilt binaries, so
                                      # SDKs\NRD\bin\NRD.dll is gitignored + LoadLibrary'd at
                                      # runtime, the xess.rs footprint; the ONE component
                                      # needing CMake+VS), install-prerequisites.sh's NRD_TAG,
                                      # the SUBMODULE SHA itself,
                                      # src/nrd.rs (repr(C) transcription against MSVC-sizer
                                      # ground truth + the Nrd::new GetLibraryDesc gate: version
                                      # 4.17 AND normalEncoding 2 AND roughnessEncoding 1, else
                                      # loud shed), and nrd_bridge.hlsl's reimplemented packing
                                      # math (YCoCg, the enc-2 L1-oct normal, normHitDist —
                                      # NEVER paste the licensed NRD.hlsli; nrd::oracle's CPU
                                      # twins + the N0/N2 gates keep the three in lockstep).
                                      # GPU tracers only (wavefront AND DXR), XeSS or FSR3
                                      # sessions (RR/FSR4-RR already denoise; --nppd excluded —
                                      # both claim the pre-upscale color slot, exit 2; quinlight
                                      # excluded). ONE denoiser, TWO EXACT-LINEAR FOLDS into
                                      # REBLUR_DIFFUSE_SPECULAR: diffuse = dd + ao·sh_irr(n)
                                      # (shared kd factor), specular = ds + is (shared un-floored
                                      # wire F0 — the pack's own demodulation divisors), so the
                                      # DELTA-form recompose color' = accum + (D'−D)·kd +
                                      # (S'−S)·f0 is exact, the residual is untouched by
                                      # construction, and a PASSTHROUGH denoiser reproduces
                                      # cs_feed_xess's color plane BYTE-identically (the N3
                                      # control arm). THE RTGI BOUNCE RIDES THE DIFFUSE FOLD
                                      # since 2026-08-08 (FLAG_NRD_GI, bit 21 — the "NRD looks
                                      # like it isn't on" fix): under default-ON RTGI the split
                                      # arm leaves prim.ao at 0, so the old diffuse input
                                      # collapsed to DIRECT-ONLY while the 1-spp bounce — the
                                      # dominant noise — rode the un-denoised residual, and
                                      # ReBLUR's diffuse hit-dist guide was 0 on EVERY pixel (no
                                      # AO ray exists at depth 0 under RTGI). Now shade_full's
                                      # RTGI block adds the bounce radiance into prim.direct_d
                                      # and the bounce ray's own t into ao_t (miss = CAM_FAR)
                                      # when the flag is set — capture-only, zero rng, accum
                                      # bit-identical across the toggle; cs_nrd_out reads D back
                                      # from the packed plane, so the delta recompose stays
                                      # exact with ZERO bridge/oracle changes. The flag is
                                      # nrd-WIRING-derived (nrd_sig(), the fsr_sig shape;
                                      # force_nrd_sig = the dual-GPU mirror + the gate hook,
                                      # mirrored per frame in record_split beside force_fsr_sig)
                                      # and DISTINCT from FLAG_FSR_SIG — FSR-RR sessions keep dd
                                      # = pure direct diffuse for AMD's own denoiser. Accepted
                                      # mismatch (the ao-fold precedent, nrd_bridge.hlsl:25-29):
                                      # the bounce enters accum via kd_full·kt·dcav, the
                                      # recompose remodulates at wire kd — the delta stays in
                                      # the residual; a sig3 lane (stride 72→80) is the fallback
                                      # if a feel-test ever objects. Gate: check-gpu N6 (armed
                                      # fold frame vs disarmed — ao_t fires on every hit px,
                                      # accum bit-identical, dd-differs anti-vacuity; stands
                                      # down under --no-rtgi where the block never compiled).
                                      # arm_nrd_for also prints the ONE success line now
                                      # ("nrd: armed — ReBLUR pre-upscale denoising at WxH" —
                                      # every other nrd: line is a failure path, and an armed
                                      # session used to be indistinguishable from an unarmed
                                      # one), and the DXR-init-fail → CPU fallback says
                                      # "nrd: not armed — the CPU renderer has no NRD path".
                                      # Hit-dist guides ride the pack's ex-free
                                      # sig.w lane (f16x2(ao_t, shadow_t), FLAG_FSR_SIG,
                                      # captured by transmit_q_t twins in rt.hlsli —
                                      # assignment-only off the EXISTING queries, zero rng, M9b
                                      # accum-bit-identity preserved; rt_sw/--dxr-inline-0 arms
                                      # report tmax = no capture, documented known-accepts —
                                      # ReBLUR's AREA_3X3 hit-dist reconstruction covers them;
                                      # shadow_t = SIGMA_SHADOW's NoL<=0-is-0 convention,
                                      # captured now for the Phase-2 sun-shadow denoiser).
                                      # Bridge kernels (nrd_bridge.hlsl, cs_6_3-clean, both
                                      # pipelines) ride descriptor set NRD_FEED_SET=3
                                      # (FEED_SETS 3→4, heap arithmetic only, root sig stays
                                      # 64/64); NRD's own ~14 pipelines/31 dispatches run from
                                      # NrdGpu's private heap (gpu/nrd_gpu.rs — the bloom
                                      # pattern at the DXC tier; CommonSettings matrices are
                                      # glam col-major VERBATIM, no transpose — the anti-SL
                                      # convention, proven by N4's 8× temporal-delta shrink);
                                      # all on the ONE list, no split_frame; an NRD frame runs
                                      # NO engine feed dispatch at all since THE FOLD
                                      # (2026-08-09, the B70 cost recovery): cs_nrd_pack itself
                                      # writes the engine's mvec/depth guides (cs_feed_xess_dm's
                                      # stores verbatim; record_feed_nrd is DELETED, the shared
                                      # view_z_to_clip_depth moved to trace_common.hlsli so the
                                      # precise-sensitive encode has ONE copy), which moved
                                      # NRD's linear view-Z u18 → u26 and wired the engine
                                      # depth/mvec at 18/19 of the NRD set (arm_nrd_for /
                                      # nrd_bridge.hlsl's register map, lockstep) — the
                                      # nrd_guides NPSR brackets ride record_nrd_pack. AND THE
                                      # SKY EXT-STORE SKIP (same day, FLAG_SKY_EXT_SKIP bit 22):
                                      # when NRD is the SOLE ext subscriber (derived; RR/FsrRr
                                      # feeds, NPPD, force_gbuf_ext all veto — they read sky ext
                                      # full-screen) sky pixels elide the whole 72 B GBufExt
                                      # store and cs_nrd_pack's sky branch (the out kernel's
                                      # exact 0.999·CAM_FAR predicate) writes canonical
                                      # constants without reading a byte of ext — proven by N7's
                                      # NaN-sentinel gate in BOTH GPU suites (fill ext with
                                      # 0xFF, force the skip, assert sky bytes stay sentinel +
                                      # hit bytes byte-equal + accum bit-identical + pack planes
                                      # clean constants; force_sky_ext_skip is the Option
                                      # override, the force_nrd_sig shape). cs_nrd_out brackets
                                      # the engine color plane NPSR↔UA itself (the plane RESTS
                                      # in NON_PIXEL_SHADER_RESOURCE — the upscaler-eval
                                      # contract; gate stand-ins must rest NPSR too), and its
                                      # sky pass-through predicate is 0.999·CAM_FAR — the EXACT
                                      # denoisingRange bound (a plain-CAM_FAR predicate shipped
                                      # once and recomposed the [0.999·far, far) hit band from
                                      # OUT texels NRD never wrote). fsr_sig() AND
                                      # gbuf_ext_needed() arm on nrd_WIRED (never PSO presence
                                      # — M9b's baseline teeth; the ext term is what carries
                                      # GBufExt across the dual-gpu band for the full-screen
                                      # bridge reads), so dual-gpu works by construction (the
                                      # per-frame force mirrors + fed_planes cover every bridge
                                      # input). ONE NrdGpu per session, shared by both GPU arms
                                      # via gpu::arm_nrd_for (both trace at the locked res;
                                      # a second instance would strand the memoized arm's
                                      # NRD_FEED_SET descriptors on dropped pools). Reset =
                                      # the presenters' gpu_reset → AccumulationMode::RESTART
                                      # (never camera motion); any frame error sheds NRD for
                                      # the session, loudly, frame continues plain — the shed
                                      # is TWO-PHASE (flag now, wait_idle-then-drop at the next
                                      # presenter entry + clear_nrd_wired on both tracers):
                                      # D3D12 lists don't refcount, so an immediate drop frees
                                      # heaps/PSOs/pools that in-flight lists still reference
                                      # (device removal), and the wiring clear is what releases
                                      # the bridge planes and disarms the pack's sig stores
                                      # after the shed. MEASURED
                                      # (4090, procedural still, native 1080p): FRUSTRACER_STAB
                                      # XeSS-alone 0.42/255 → NRD 0.07–0.10 (BELOW DLSS-RR's
                                      # ~0.12 reference); AMD iGPU FSR3 0.38 → 0.19 still
                                      # ramping (vendor neutrality live); DXR+XeSS → 0.10.
                                      # COST, re-measured 2026-08-09 ALL-IN (span delta vs
                                      # --no-nrd, parked native 1080p — the recorded "~1.18 ms"
                                      # counted only the three nrd regions and missed the
                                      # leaf/sky ext-store delta): B70 procedural +3.65 ms
                                      # BEFORE the fold+sky-skip (2.28 → 5.93 span, 2.6×!! — the
                                      # user-reported "NRD makes the B70 2x slower", decomposed
                                      # as ReBLUR 1.97 + pack 0.57 + out 0.48 + leaf ext +0.43 +
                                      # sky ext +0.33 − feed 0.14), B70 world +2.64 (1.62×),
                                      # 4090 procedural +1.72 (2.8× — same class, NOT
                                      # Intel-specific: the tax is fixed per frame and the
                                      # frames are tiny). AFTER: B70 5.93 → 5.44 procedural /
                                      # 6.89 → 6.15 world, 4090 2.68 → 2.31 — the engine-side
                                      # recovery, quality-neutral by construction. The
                                      # remaining tax is mostly ReBLUR itself + the hit-pixel
                                      # ext stores; the OPT-IN levers go further (quality
                                      # trades, feel-test before defaulting): --nrd-perf loads
                                      # the REBLUR_PERFORMANCE_MODE DLL from <nrd-path>\perf
                                      # (install-prerequisites.bat builds BOTH — perf mode is
                                      # COMPILE-TIME in 4.17.3, no ReblurSettings field exists;
                                      # the armed line prints the DLL dir since LibraryDesc
                                      # cannot tell the variants apart; B70 nrd 1.99 → 1.74),
                                      # and the --nrd-* runtime tuning family (nrd::ReblurTuning,
                                      # the fsr-tune all-Option shape, applied in
                                      # nrd_gpu::reblur_settings): --nrd-max-stabilized-frames 0
                                      # DROPS the TemporalStabilization pass (nrd 1.74 → 1.55
                                      # stacked on perf), --nrd-prepass-radius 0 disables the
                                      # prepasses, --nrd-no-anti-firefly, --nrd-max-accum-frames.
                                      # THE COMPLETENESS SWEEP (2026-08-10, the migration's
                                      # Tier A — 22 of 28 ReblurSettings fields and 16 of 31
                                      # CommonSettings fields had NO plumbing at all): the three
                                      # ReBLUR SUB-STRUCTS are reachable as comma-tuples (the
                                      # --cam idiom; nrd_floats exits 2 on a wrong arity rather
                                      # than half-applying, since a partially-landed tuple makes
                                      # an A/B report the wrong arm) — **--nrd-convergence S,B,P**
                                      # is the one that matters for the parked-camera darkening
                                      # class: ReBLUR drives denoising by f = 1/(1 + k*N) with
                                      # k = S*lerp(B, 1, N/(1 + P*maxAccum)) since v4.17, and
                                      # B < 1 (default 0.2) deliberately means "blur MORE on a
                                      # short history" — the SAME shape the FRD campaign measured
                                      # (a moving frame flattered by young-history wide blur
                                      # bleeding light outward, a parked frame converging onto a
                                      # genuinely dimmer truth), so raising B toward 1 is the
                                      # lever that makes the two regimes agree;
                                      # **--nrd-responsive R[,N]** is NVIDIA's animated-water
                                      # lever (below roughness R the history scales WITH
                                      # roughness — this scene's ripple-normal water is
                                      # roughness 0.05, exactly the case; the optional N floors
                                      # the frames kept at roughness 0, and the one-arity form
                                      # must leave N unset — cli::self_test pins that);
                                      # **--nrd-antilag SIGMA,SENS**. Plus the CommonSettings
                                      # half (nrd::CommonTuning, the ReblurTuning shape one
                                      # level up): **--nrd-split X** is ReBLUR's OWN
                                      # noisy-vs-denoised wipe — the left X of the frame takes
                                      # the SPLIT_SCREEN pass, which copies IN to OUT, so our
                                      # delta-form recompose collapses to `col = base` there
                                      # (the raw 1-spp accum) with no second capture to line up.
                                      # MEASURED liveness at X=1.0 (full passthrough, the arm
                                      # Reblur.cpp early-returns on): +50% Laplacian vs the
                                      # denoised arm, 23% of px past 4/255. CAVEAT worth not
                                      # re-deriving: a FRACTIONAL split is nearly invisible in a
                                      # parked screenshot because XeSS's own temporal
                                      # accumulation launders the undenoised half — read it
                                      # under motion, or at X=1.
                                      # THREE MORE Tier-A repairs, none of them a new flag:
                                      # (a) timeDeltaBetweenFrames is now passed EXPLICITLY (ms)
                                      # instead of left at the header's "0 = tracked internally"
                                      # default — NRD's own timer measures wall clock BETWEEN
                                      # SetCommonSettings CALLS (InstanceImpl.cpp's m_Timer),
                                      # which in a headless gate is the gate's readbacks and
                                      # oracle loops, not a frame, and it is not cosmetic:
                                      # m_FrameRateScale = max(33.333/dt, 1) reaches ReBLUR's
                                      # antilag scale, its accumulation-speed curve, and the
                                      # specular virtual-motion tap stride, so an
                                      # internally-timed gate is a gate whose denoiser settings
                                      # DRIFT WITH MACHINE LOAD. Interactive presenters hand
                                      # over their real frame_ms (clamped 1..200 ms — 0 is the
                                      # sentinel that silently re-enables the internal timer);
                                      # the N4 gate and cinematic capture hand over
                                      # nrd_gpu::NOMINAL_DT_MS, a fixed 60 Hz constant, because
                                      # both are deterministic by contract (a capture's
                                      # sub-frames are convergence passes at ONE pose, so
                                      # neither the frame clock nor --cinematic-fps describes
                                      # them). (b) resourceSizePrev/rectSizePrev received the
                                      # CURRENT size; NrdGpu::prev_size now tracks the real one
                                      # (equal today — the res is locked at construction and a
                                      # resize rebuilds the instance — so this is bookkeeping
                                      # against a future DRS path, kept because the failure it
                                      # prevents is silent: reprojection through the wrong prev
                                      # rect just denoises a slightly wrong history).
                                      # (c) THE cameraJitter POLARITY QUESTION IS SETTLED, and
                                      # the answer is "structurally inert here" — do not
                                      # re-litigate it with an A/B. We feed the RAW offset while
                                      # every sibling SDK site applies a sign constant
                                      # (xess/fsr::JITTER_SIGN), which looked like an
                                      # unverified asymmetry; in the v4.17.3 source the value
                                      # reaches exactly two places, REBLUR_Validation's overlay
                                      # UV and m_JitterDelta = max(|dx|, |dy|) over the cur/prev
                                      # pair (which feeds only the CHECKERBOARD resolve speed,
                                      # and is sign-symmetric anyway), and we never run
                                      # checkerboard. What is NOT inert is the RANGE: NRD
                                      # asserts [-0.5, 0.5], which is exactly
                                      # FrameCtx::frame_jitter's own interval.
                                      # OUT_VALIDATION IS READABLE AT LAST (the plane was
                                      # allocated, bound, and read by nothing):
                                      # FR_NRD_DEBUG=<frame> arms enable_validation AND dumps
                                      # that frame's overlay to nrd-validation.png — a readback
                                      # copy recorded at the target frame and mapped
                                      # RING_FRAMES later (the gputime retirement argument; no
                                      # wait_idle, which would perturb the very frame pacing
                                      # that now feeds dt_ms), with the buffer committed ONLY
                                      # under the lever. A DUMP rather than the live overlay the
                                      # old comment promised, deliberately: compositing would
                                      # have to happen in cs_nrd_out, the bridge unit BOTH
                                      # engines compile, so it would need an engine-conditional
                                      # binding, an RGBA8 dummy plane on the FRD side, and a
                                      # fifth wire site — real surface across the engine-blind
                                      # DnGpu boundary for a view that only runs under an env
                                      # var, and one frame answers its questions just as well
                                      # (and works headlessly). READING IT: the FRAMES panels
                                      # are f = 1 - frames/maxAccum through ColorizeZucconi, so
                                      # BLACK = FULLY ACCUMULATED and bright = a fresh history
                                      # (the inverse of the natural guess). Baseline read
                                      # (helmet, parked, frame 100): normals/roughness sane, Z
                                      # red exactly where the sky is (out of denoisingRange), MV
                                      # black (a parked camera HAS no motion), both FRAMES
                                      # panels black = both histories saturated — i.e. the
                                      # wiring is healthy. The same dump with
                                      # --nrd-responsive 0.3 live shows the specular history
                                      # deliberately SHORT on the low-roughness visor and long
                                      # on the rough shell, which is that lever doing exactly
                                      # what it claims.
                                      # THE SG / DIRECTIONAL-RADIANCE CAMPAIGN
                                      # (2026-08-10) — and its headline is a
                                      # NEGATIVE result that should stop anyone
                                      # re-attempting the obvious version.
                                      # NVIDIA's NRD.hlsli is now READABLE by
                                      # one compile unit: gfx::shaders::
                                      # nrd_bridge_tail `include_str!`s it
                                      # STRAIGHT OUT OF THE SUBMODULE and joins
                                      # it ahead of nrd_bridge.hlsl (both
                                      # assemblies — the wavefront's
                                      # TraceSources::nrd_bridge and DxrGpu's
                                      # own cs_6_3 build — go through that ONE
                                      # tail; a wavefront-only first draft
                                      # compiled clean and broke --check-dxr on
                                      # an undeclared identifier, which is why
                                      # the cargo pin now reads dxr.rs too).
                                      # THIS IS NOT A PASTE: nothing of
                                      # NVIDIA's is committed here, the header
                                      # arrives from NVIDIA's own repository in
                                      # each checkout, and the ONE line we
                                      # rewrite is `#include "NRDConfig.hlsli"`
                                      # — mandatory, because this tree has no
                                      # #include machinery at all AND that file
                                      # is CMake-GENERATED (install-
                                      # prerequisites configures twice, the
                                      # perf arm rewriting it, so including
                                      # whatever the last configure left would
                                      # make our encoding a function of
                                      # installer ordering). We state the pair
                                      # ourselves and they are the exact ones
                                      # nrd.rs's GetLibraryDesc gate REFUSES to
                                      # run without (2 / 1), so a mismatched
                                      # DLL sheds loudly instead of silently
                                      # disagreeing. The reimplemented pack
                                      # math STAYS ours and must not be
                                      # "unified" with the header — N0/N2 score
                                      # the shader against nrd::oracle's CPU
                                      # twins, and swapping the shader side to
                                      # NVIDIA's would leave those gates
                                      # comparing NVIDIA's code to itself.
                                      # WHAT WE DID NOT BUILD, with the source
                                      # evidence: (1) REBLUR_DIFFUSE_SPECULAR
                                      # _SH's radiance output is IDENTICAL to
                                      # the non-SH denoiser's. The SH1 plane is
                                      # a strict PASSENGER in every pass that
                                      # touches it — TemporalAccumulation lerps
                                      # it by the SAME diff/specNonLinear
                                      # AccumSpeed and scales it by the SAME
                                      # GetLumaScale the RADIANCE channel
                                      # derived, HistoryFix and the spatial
                                      # filter accumulate it under the SAME w,
                                      # TemporalStabilization the same again.
                                      # NOTHING in REBLUR derives a weight FROM
                                      # SH. So the mode switch costs 4 in + 4
                                      # out planes and bandwidth in 5 passes
                                      # and changes the denoised radiance by
                                      # exactly nothing; its whole value is
                                      # what the APPLICATION does with the
                                      # direction at the back end. (2) And the
                                      # back end we cannot use is the RESOLVE:
                                      # NRD_SG_ResolveDiffuse/Specular take
                                      # INCIDENT radiance from the dominant
                                      # direction and INTEGRATE a BRDF
                                      # (_NRD_DiffuseTerm / _NRD_GeometryTerm
                                      # are applied inside them), while our
                                      # lanes are already BRDF-integrated,
                                      # albedo-demodulated OUTGOING lobes — so
                                      # resolving would apply the BRDF twice.
                                      # Using it means re-cutting the pack to
                                      # carry incident radiance and dropping
                                      # the delta-form spine with it: a
                                      # G-buffer project (the Phase-4 class),
                                      # not a back-end swap. CONSEQUENCE: SH
                                      # mode is worth wiring only once REAL
                                      # per-ray directions exist to put in it
                                      # (the RTGI bounce direction weighted by
                                      # bounce radiance, the reflection ray
                                      # direction) — which is the same
                                      # G-buffer change, so the two land
                                      # together or not at all.
                                      # WHAT SHIPPED: the RE-JITTER
                                      # (FLAG_NRD_REJITTER, bit 23, default ON,
                                      # FR_NRD_REJITTER=off is the A/B arm) —
                                      # NVIDIA's own NRD_SG_ReJitter, which
                                      # needs NO extra planes and NO mode
                                      # change, returning a float2 Jacobian
                                      # clamped to [1/A, A] = the ratio of the
                                      # BRDF at this pixel's normal to the mean
                                      # over its 4 neighbours, i.e. exactly the
                                      # texel-scale variation a spatial blur
                                      # averaged away. THE DIRECTIONS ARE
                                      # ANALYTIC and that bounds it: N for the
                                      # cosine lobe, NVIDIA's own
                                      # _NRD_GetSpecularDominantDirection for
                                      # GGX — which is what the Jacobian wants
                                      # (it measures the BRDF's response to the
                                      # NORMAL varying across the
                                      # neighbourhood) but means the specular
                                      # term cannot know a particular pixel's
                                      # reflection ray went elsewhere. AND THE
                                      # SUBSTITUTION IS NOT NEUTRAL — the
                                      # DIFFUSE half is ONE-SIDED:
                                      # _NRD_ComputeBrdfs weights by
                                      # saturate(dot(N_tap, Ld)), so Ld = the
                                      # CENTER normal evaluates the center at
                                      # dot(N,N) = 1, the maximum, while every
                                      # neighbour sits at dot(N_nb, N_c) <= 1.
                                      # The diffuse Jacobian is therefore
                                      # biased ABOVE 1 wherever normals vary
                                      # (specular is anchored the same way, Ls
                                      # being derived FROM the center N, though
                                      # ReJitter's lerp(V, Ls, roughness) step
                                      # muddies it): it SHARPENS and
                                      # essentially never attenuates, up to the
                                      # 2.0 clamp. A true dominant LIGHT
                                      # direction is generally not N, which is
                                      # what makes NVIDIA's own Jacobian
                                      # two-sided — so read this as "amplify
                                      # the delta where normals vary", and read
                                      # the +5.2% Laplacian / +0.2% mean below
                                      # the same way (contrast AND level moving
                                      # together is the signature of a
                                      # one-sided multiplier, not of a
                                      # symmetric restoration). FOLLOW-ON if
                                      # the default is kept: the sun direction
                                      # is already in the CB and is a far
                                      # better diffuse-dominant proxy for this
                                      # content than N, restoring two-sidedness
                                      # for free — measure before adopting, the
                                      # analytic pair's virtue being that it
                                      # cannot be wrong about a direction it
                                      # never claimed to know. THE
                                      # JACOBIAN MULTIPLIES THE DELTA, not the
                                      # output as NVIDIA's own Composition
                                      # example does, and the reason is that
                                      # OUR recompose is additive against a raw
                                      # base: `base` already carries this
                                      # pixel's own shading at its own normal,
                                      # so the only denoiser-produced quantity
                                      # — the only thing that got smeared — is
                                      # the delta. It also makes N3 hold BY
                                      # CONSTRUCTION rather than by tolerance
                                      # (a passthrough's exact-0.0 delta times
                                      # a finite clamped Jacobian is 0.0).
                                      # Neighbour taps reject on viewZ BEFORE
                                      # reading a neighbour normal — under
                                      # FLAG_SKY_EXT_SKIP an out-of-range
                                      # neighbour's ext record is UNWRITTEN, and
                                      # NRD's own Z test would disarm those
                                      # pixels anyway, so pre-testing changes no
                                      # result and keeps stale bytes out of an
                                      # arithmetic expression (the armed
                                      # NaN-sentinel contract). IT IS NRD-ONLY
                                      # BY AN EXPLICIT ENGINE TERM, and that is
                                      # the one place the arming does NOT
                                      # follow nrd_sig's wiring-derived shape:
                                      # `nrd_wired` is non-empty under FRD too
                                      # (one bridge, two engines —
                                      # arm_denoiser_for wires BOTH arms
                                      # through wire_nrd_feed and cs_nrd_out is
                                      # shared), so a wiring-only predicate
                                      # handed FRD's deltas NVIDIA's Jacobian
                                      # with nothing in the code, the lever
                                      # name, or these notes saying so — it
                                      # shipped that way in the first draft and
                                      # the review caught it. Two costs, either
                                      # sufficient: the F4-F7 bands recorded
                                      # here were measured without it, and FRD
                                      # is the A/B ORACLE this whole migration
                                      # is judged against — an oracle sharing a
                                      # post-process with the arm under test is
                                      # not one (N8 measures the perturbation
                                      # at 18209/213200 px, so the
                                      # contamination was not academic).
                                      # wire_nrd_feed therefore takes a
                                      # `dn_is_nrd` PARAMETER on both tracers
                                      # (10 call sites, each declaring; nothing
                                      # inside can derive it — NrdRes is two
                                      # PSOs and the bridge is deliberately
                                      # engine-blind). Whether FRD WANTS the
                                      # same correction is a real and separate
                                      # question — the Jacobian is
                                      # engine-agnostic in nature, any spatial
                                      # denoiser averages the same local BRDF
                                      # variation away — and it needs its own
                                      # measurement and its own name, never
                                      # inheritance. MEASURED at the
                                      # bistro terrace pose, PARKED and
                                      # pose-registered (one pose, two arms —
                                      # unlike the still-vs-moving harness,
                                      # whose `drive` displaces the camera and
                                      # whose crops are therefore NOT
                                      # comparable): Laplacian +5.2% (0.6774 ->
                                      # 0.7125 — detail restored, the intended
                                      # direction), 0.51% of px past 2/255, and
                                      # the luma distribution UNMOVED at every
                                      # percentile (mean 10.356 -> 10.377, p10/
                                      # p50/p90 identical to 2 dp). SO IT IS NOT
                                      # THE FIX FOR THE PARKED-CAMERA DARKENING
                                      # — it restores contrast, not level; the
                                      # darkening remains open. Single-run cost
                                      # observation: parked frame-to-frame
                                      # mean|d| 0.172 -> 0.192, small in
                                      # absolute terms (both ~0.2/255) but a
                                      # real trade, so the default wants a feel
                                      # test.
                                      # EXACT REMODULATION (FLAG_REMOD_EXACT,
                                      # bit 24, DEFAULT ON, FR_NRD_REMOD=off is
                                      # the A/B arm — 2026-08-10, the firefly
                                      # campaign's first of two leaks): the
                                      # delta form recomposes
                                      # `col = base + (D_out−D_in)·j·kd +
                                      # (S_out−S_in)·j·f0`, but it remodulated
                                      # the denoiser's CORRECTION at the wire
                                      # `kd = albedo·(1−metallic)·(1−trans)`
                                      # while shade had multiplied those same
                                      # lobes by MORE: `sk = 1−0.157·sheen` (the
                                      # Charlie energy term, folded into shade's
                                      # own kd), the translucency split,
                                      # `detail_sun_shadow` on the direct
                                      # diffuse, and `detail_cavity` on the
                                      # bounce and on direct_s. So the
                                      # correction landed at 1/m its physical
                                      # weight — on a dcav = 0.3 pit under
                                      # FLAG_NRD_GI the bounce correction was
                                      # 3.3x — and the leftover raw fraction of
                                      # every bright 1-spp spike stayed in
                                      # `base` UN-DENOISED. A firefly source
                                      # that nrd_bridge.hlsl's OWN COMMENT
                                      # asserted impossible ("the mismatch only
                                      # shifts what the denoiser sees between
                                      # signal and residual, never the
                                      # recomposed color") — true only while
                                      # D_out == D_in, i.e. of a passthrough and
                                      # of nothing else; correcting that
                                      # sentence was part of the change.
                                      # THE FIX IS A RE-CAPTURE, NOT A NEW LANE:
                                      # shade.hlsli re-assigns the two lap-0 sig
                                      # captures AFTER every post-capture factor
                                      # has landed, so the bridge's kd/f0 become
                                      # the EXACT divisors. A REASSIGNMENT (the
                                      # originals stay put) so the flag-clear
                                      # arm is textually and numerically today's
                                      # code — the guarded-never-`* 1.0`
                                      # discipline, and what keeps the off arm's
                                      # expression DAG bit-identical for the
                                      # M9b/N6 accum gates. It captures
                                      # `diffuse_d` and NOT `direct_d·(1−tl)`
                                      # deliberately: diffuse_d already carries
                                      # the translucency BACK-RAY term, which
                                      # had no wire lane at all and was
                                      # therefore 100% un-denoised stochastic
                                      # residual — the one place this fix
                                      # REMOVES noise rather than merely
                                      # reweighting it.
                                      # THE SPECULAR LANE NEEDS NO MULTIPLIER,
                                      # and that is luck landing on the right
                                      # side: ds's exact factor is dcav (folded
                                      # at the cavity block) and is's is exactly
                                      # 1, so folding dcav into ds leaves BOTH
                                      # sub-terms remodulating at f0 — which
                                      # puts the free side of the split on `is`,
                                      # the reflection bounce, the NOISY one.
                                      # m_s stays 1. THE DIFFUSE LANE CANNOT do
                                      # the same: its two sub-terms disagree
                                      # (diffuse_d by sk, the ambient/RTGI
                                      # bounce by sk·dcav), so both factors ride
                                      # PrimSurf (`m_d` / `amb_k`) and are
                                      # blended by ENERGY at the one site both
                                      # are known — shade_full's GI fold under
                                      # RTGI, the sampled-ambient block
                                      # otherwise (`split_ambient == rtgi`, so
                                      # the two blend sites are mutually
                                      # exclusive by construction).
                                      # AND THE BLEND IS EXACT, not a
                                      # compromise, for the correction a
                                      # denoiser can actually deliver. Write the
                                      # channel as A + B with factors k_a, k_b;
                                      # accum holds kd·(k_a·A + k_b·B). For any
                                      # UNIFORM scaling D_out = L·D_in the delta
                                      # form gives base + (L−1)(A+B)·kd·m, and
                                      # at m = (k_a·A + k_b·B)/(A+B) that
                                      # collapses ALGEBRAICALLY to
                                      # L·kd·(k_a·A + k_b·B) — the right answer.
                                      # It is approximate only where the
                                      # denoiser REDISTRIBUTES between the two
                                      # sub-terms, which it cannot: they were
                                      # summed before it saw them. One lane is
                                      # not a thrifty approximation of two, it
                                      # is the whole information the wire admits
                                      # (exact when the sub-terms share a hue,
                                      # bounded by the convex combination
                                      # otherwise — m ∈ [min(k_a,k_b),
                                      # max(k_a,k_b)], so it can never amplify).
                                      # WHY NOT the obvious `m_d = dcav` with
                                      # the signal DIVIDED by it: that injects
                                      # texel-frequency 1/dcav INTO the denoiser
                                      # input, which the spatial filter then
                                      # averages — blur(S/dcav)·dcav is not S,
                                      # so the direct diffuse, which carries no
                                      # cavity in shade at all, comes back with
                                      # a spurious cavity RIPPLE. The energy
                                      # blend keeps the input clean and puts
                                      # every crisp factor on the DELTA, where
                                      # `base` already holds the exact per-pixel
                                      # value and only the correction needs
                                      # weighting.
                                      # THE WIRE: m_d rides sig.w's HIGH half,
                                      # ON LOAN from `shadow_t` — written today
                                      # and decoded by NOBODY (nrd_bridge masks
                                      # `sig.w & 0xffff` for ao_t; FSR-RR never
                                      # reads sig.w at all), captured ahead of a
                                      # SIGMA shadow denoiser that does not
                                      # exist. Gated, so the flag-clear arm
                                      # packs shadow_t verbatim and N5's
                                      # sh_fired must-fire runs THERE (N5
                                      # asserts its own precondition now rather
                                      # than trusting a comment: an armed m_d is
                                      # positive on every hit pixel, so the
                                      # must-fire would PASS while scoring the
                                      # wrong lane — silent-vacuous, the worst
                                      # shape). ZERO stride change; the exit
                                      # when SIGMA lands is written down: stride
                                      # 72 → 80 is 16-ALIGNED (72 is not), at
                                      # ~+11% on a 0.34 ms store, plus
                                      # GBUF_EXT_STRIDE, dual.rs's fed_strides /
                                      # MAX_FED_STRIDES, and the hardcoded `72`
                                      # literals in the N5/N6/N7/N9/N11 gates.
                                      # APPEND, never insert — every gate
                                      # decodes lanes by index.
                                      # Derivation is the FLAG_NRD_GI shape
                                      # (gbuf_full && fsr_sig && nrd_sig), so it
                                      # cannot arm without the sig capture and
                                      # cannot arm in an FSR-RR session (whose
                                      # composite identity owns these lanes).
                                      # NOT engine-gated, the one place it
                                      # departs from FLAG_NRD_REJITTER: that
                                      # term exists because the Jacobian is
                                      # NVIDIA's and FRD is the oracle it is
                                      # judged against; THIS is our arithmetic
                                      # being wrong, and an oracle fed
                                      # mis-scaled inputs is not a better oracle
                                      # for being fed them consistently. Both
                                      # engines take the fix. DUAL-GPU MIRRORS
                                      # IT (gpu/mod.rs, beside force_nrd_sig):
                                      # the two arms pre-scale the sig lanes
                                      # differently, so an unmirrored band hands
                                      # the bridge lanes scaled by one rule and
                                      # remodulates them by the other — a
                                      # visible brightness STEP at the band seam
                                      # on every sheened/translucent/cavity
                                      # pixel, worse than merely a different
                                      # denoiser input. Zero rng draws; accum is
                                      # BIT-IDENTICAL across the toggle
                                      # (assignment-only — N6b asserts it), so
                                      # every same-seed A/B and both check PNGs
                                      # are untouched.
                                      # COVERAGE, SAID PLAINLY BECAUSE A GREEN
                                      # SUITE DOES NOT IMPLY IT: the procedural
                                      # check scene has no sheen, no
                                      # translucency, no cavity pits and no
                                      # detail sun-shadow, so every factor is
                                      # exactly 1.0, m_d pins at [1.0000,
                                      # 1.0000] and the whole feature is INERT
                                      # there — N6b prints a NOTE saying so, and
                                      # a green run proves it HARMLESS, not
                                      # correct. san-miguel-lp exercises it
                                      # (m_d to 0.9087 over 1514 px). The
                                      # scene-independent proof is --check's
                                      # `remod` sweep (N9-CPU).
                                      # THE RESIDUAL SPIKE CAP (FLAG_NRD_RCLAMP
                                      # bit 25 / _HARD bit 26,
                                      # FR_NRD_RCLAMP=off|on|hard, DEFAULT OFF —
                                      # the campaign's second leak): the
                                      # recompose is algebraically
                                      # `col = R + D_out·kd·m_d + S_out·f0` with
                                      # `R = base − D_in·kd·m_d − S_in·f0`, so
                                      # the two folded channels are denoised and
                                      # R is passed through RAW BY
                                      # CONSTRUCTION. With FLAG_NRD_GI live the
                                      # bounce rides the diffuse fold, so R's
                                      # remaining stochastic term is the root
                                      # GLASS/TRANSMISSION chain — each interior
                                      # lap shades with its own sampled
                                      # sun-shadow pairs and sampled AO and
                                      # nothing filters any of it. Neither
                                      # ReBLUR's own anti-firefly nor FRD's ring
                                      # pre-clamp can reach it: both see only
                                      # the folded channels. cs_nrd_out
                                      # reconstructs R and soft-caps it against
                                      # its 8-neighbour ring.
                                      # THE CAP IS A CONJUNCTION, which is what
                                      # makes it safe:
                                      # `cap = max(K_MEAN·ring_mean,
                                      # K_MAX·ring_max)` at K_MEAN = 8.0 (=
                                      # FRD's FIREFLY_K, deliberately, so the
                                      # two clamps speak one number) and K_MAX =
                                      # 3.0 — a pixel is capped only when it
                                      # beats BOTH 8x its ring's mean AND 3x its
                                      # ring's BRIGHTEST member. A lone
                                      # one-sample spike has mean ≈ max so the
                                      # mean term binds; a RESOLVED feature (a
                                      # bright emissive texel, a caustic edge)
                                      # has a bright neighbour so max >> mean
                                      # and the max term protects it. Both
                                      # reductions come out of one loop, so the
                                      # discrimination is free.
                                      # THE TEST IS ON `base`, THE CORRECTION
                                      # LANDS ON `r`, and mixing those two up is
                                      # the trap this shipped wrong once to
                                      # find: the ring holds BASE luma, so a cap
                                      # built from it is in base units, while
                                      # the residual is only ever a FRACTION of
                                      # base — testing `rl > cap` puts the
                                      # threshold orders of magnitude above the
                                      # quantity and the clamp provably never
                                      # fires (measured: the K=1 diagnostic arm
                                      # left N3 and N4 byte-identical, and a
                                      # feature that cannot fire also cannot be
                                      # gated). The consistent formulation needs
                                      # no neighbour residuals at all: model a
                                      # neighbour's residual as its base times
                                      # this pixel's residual FRACTION f = rl/bl
                                      # and `rl > K·mean(ring_R)` reduces
                                      # EXACTLY to `bl > K·mean(ring_base)` — f
                                      # cancels — so the outlier test is a
                                      # statement about base, which we have a
                                      # ring for, while the scale cap/bl applies
                                      # to r, the only part the denoiser did not
                                      # clean. THE HARD ARM RELAXES BOTH
                                      # MULTIPLIERS for the same class of
                                      # reason: `cap` is a max and mx >= mean
                                      # always, so leaving K_MAX at 3.0 would
                                      # leave 3·ring_max binding and "hard"
                                      # would be barely harder than the default
                                      # (measured: byte-identical N3/N4 — it
                                      # gated nothing at all). At K_HARD = 1.0
                                      # on both, cap == ring_max and the test
                                      # becomes "is this pixel a strict local
                                      # maximum", which is the diagnostic's
                                      # actual claim.
                                      # THE HALO AND ITS BARRIER ARE HOISTED
                                      # ABOVE BOTH EARLY RETURNS, and that is a
                                      # correctness requirement:
                                      # GroupMemoryBarrierWithGroupSync in a
                                      # PARTIALLY-RETURNED group is undefined
                                      # and partial groups are REAL (the QA lab
                                      # runs 960x540; 540/8 = 67.5). Indexed by
                                      # TILE SLOT (thread-linear over 100 slots
                                      # by 64 threads), never by the centre
                                      # pixel, so a returned thread's slot still
                                      # gets filled. `flags` is a cbuffer value
                                      # and therefore group-uniform, so gating
                                      # the barrier on it is legal and keeps the
                                      # disarmed arm free of both the LDS
                                      # traffic and the sync. The halo holds
                                      # BASE luma, not the residual, and that is
                                      # load-bearing twice: gathering R at eight
                                      # neighbours would need neighbour
                                      # gbuf_ext, which under FLAG_SKY_EXT_SKIP
                                      # is UNWRITTEN at sky texels — so the base
                                      # ring keeps N7's "stale ext bytes stay
                                      # unread BY CONSTRUCTION" contract without
                                      # nrd_tap's pre-test — and it is 12 B per
                                      # halo texel instead of ~52. RING ONLY,
                                      # CENTRE EXCLUDED (frd.rs's recorded
                                      # teeth: with the centre in, a lone
                                      # outlier dominates its own cap and the
                                      # clamp asymptotes to a fixed 1−K/9 trim
                                      # however extreme the spike);
                                      # out-of-range neighbours EXCLUDED, not
                                      # zeroed (sky `base` is full sky radiance
                                      # and would inflate the cap into vacuity
                                      # along every horizon); nv < 4 ⇒ skip, a
                                      # 3-neighbour estimate is not a
                                      # neighbourhood. The correction is an
                                      # `if`/`-=` and NOT a third additive term
                                      # inside the `precise` expression:
                                      # `col + (R_capped − R)` would evaluate
                                      # `+ 0.0` on every unfired pixel and
                                      # `x + 0.0 == x` bitwise for every finite
                                      # x EXCEPT −0.0 (frd_temporal.hlsl's own
                                      # "the skip is a BRANCH, never a computed
                                      # 1.0"). IT ATTENUATES IN LUMA, NOT PER
                                      # CHANNEL — r can be negative in a channel
                                      # (the deterministic-factor remainders
                                      # leave base carrying less diffuse than
                                      # D_in·kd·m_d at some pixels), and
                                      # subtracting a positive multiple of a
                                      # negative channel RAISES it, so N10
                                      # asserts luma monotonicity and
                                      # deliberately not the componentwise
                                      # version (N8's own first-draft failure,
                                      # in a different currency).
                                      # N3/F3 SURVIVE BECAUSE THE GATES PIN THE
                                      # BIT CLEAR, not by construction — worth
                                      # stating plainly, because the natural
                                      # assumption is the opposite. A real
                                      # frame's R is a real noisy quantity, so
                                      # an ARMED cap fires and legitimately
                                      # breaks a byte compare; that is the
                                      # feature working, and N10 is its own gate
                                      # exactly as N8 is ReJitter's beside N3.
                                      # The pin covers N3/F3 (whose subject is
                                      # the recompose) AND N4/F4/N8 (whose
                                      # subject is the denoiser and the
                                      # Jacobian: a gate whose subject shares a
                                      # post-process with the thing under test
                                      # is not scoring the thing under test).
                                      # ENGINE-BLIND, with no nrd_engine clause
                                      # — the residual is the same residual for
                                      # both engines and both are scored against
                                      # the same `base`, so a term that fixes it
                                      # belongs to the shared bridge (F10 is
                                      # N10's FRD twin and proves exactly that).
                                      # DEFAULT OFF, deliberately unlike
                                      # FR_NRD_REJITTER/FR_NRD_REMOD: those are
                                      # restoration and a bug fix, this is a
                                      # QUALITY TRADE with a known-accept — a
                                      # single-texel hard-edged emissive on a
                                      # black background can be attenuated
                                      # (K_MAX is the mitigation; the exemption
                                      # tag would need the stride move, since
                                      # sig.w's high half is now m_d's). THE
                                      # SPARSE-SIGNAL RISK is the K >= 1/p
                                      # lesson (frd.rs), and the answer differs
                                      # per term: the glass/transmission chain's
                                      # refraction direction is DETERMINISTIC
                                      # (glass_snell), so it fires at EVERY
                                      # glass pixel, p ≈ 1, bounded
                                      # multiplicative jitter on an every-frame
                                      # term — safe, and the target; sun through
                                      # refraction is coherent on flat glass but
                                      # genuinely sparse on RIPPLED WATER, which
                                      # is where to measure; the emissive
                                      # display-add is not stochastic at all but
                                      # can be 1-2 px wide. Unlike FRD's
                                      # GI-fold case, the dominant noisy term
                                      # here has p ≈ 1, so a fixed K is not
                                      # structurally doomed. COVERAGE GAP,
                                      # stated: --check-dxr never dispatches
                                      # cs_nrd_out AT ALL (every bridge-out gate
                                      # lives in check-gpu), so an armed cap on
                                      # the DXR pipeline is exercised by no gate
                                      # — the accepted "one text, two
                                      # root-signature objects" argument, but if
                                      # the clamp ever grows engine- or
                                      # pipeline-dependent behaviour check-dxr
                                      # needs a bridge-out arm BEFORE that
                                      # lands. THE MEASUREMENT THAT DECIDES THE
                                      # DEFAULT (and whether a third denoised
                                      # channel is built at all): a firefly
                                      # count = px whose luma exceeds 16x the
                                      # local 5x5 MEDIAN, over --frd-lab on
                                      # san-miguel (glass) and the world
                                      # (water) — watching `keep` for the
                                      # suppression signature — plus a live
                                      # --qa/frqa A/B across
                                      # FR_NRD_RCLAMP=off|on|hard on bistro lamp
                                      # poses, a --gpu-timing cost read on the
                                      # nrd-out region (BUDGET +0.10 ms; +0.30
                                      # is not acceptable on a 0.48 ms pass),
                                      # and FRUSTRACER_STAB staying in the
                                      # 0.06-0.11 band. If the count drops and
                                      # the artifact goes, default it on and
                                      # build no third channel; if the count
                                      # barely moves while glass still shimmers,
                                      # the residual's noise is BROADBAND, a
                                      # spike cap is the wrong instrument, and
                                      # the third channel is justified — built
                                      # in FRD first, behind its own bit, with
                                      # NRD carrying it clear.
                                      # Gates: --check-nrd (N0
                                      # DLL-free math twins + N1 instance/dispatch contract).
                                      # N1 RUNS ON EVERY PLATFORM since 2026-08-11 (B4b-i) — it
                                      # was Windows-only, printing a SKIP line here — and its
                                      # absent/told split changed WITH that, on BOTH platforms:
                                      # a MISSING artifact still SKIPs at exit 0 (an environment
                                      # fact), but a PRESENT one the version/encoding gate REFUSES
                                      # is now a FAIL. Until then any Nrd::new error SKIPped, so a
                                      # library built without the cmake encoding pins — the exact
                                      # drift that gate exists to catch — exited 0 having gated
                                      # nothing, and the hole is far wider on Linux where the
                                      # artifact is built locally by a script anyone can mis-flag
                                      # (TOOTH FIRED: a -DNRD_NORMAL_ENCODING=0 build reads
                                      # "encodings (normal 0, roughness 1) != pinned (2, 1)" and
                                      # exits 1). MEASURED on Linux, and these are the numbers
                                      # B4b-ii's recorder is designed against: 14 pipelines / 31
                                      # dispatches / pool perm 13 trans 8 / cb-max 864 B /
                                      # samplers 2 / entry point "main" / spaces resources 0,
                                      # cb+samplers 1.
                                      # N1'S NEW ASSERTIONS, all read from the library, no
                                      # literals but the pins: (a) spirv_binding_offsets, PRINTED
                                      # FIELD BY NAME and pinned at {sampler 0, texture 20,
                                      # cbuffer 2, storage 3} — a field NOTHING in this tree had
                                      # ever read, and a prerequisite INPUT to the Vulkan
                                      # recorder's descriptor layout. UN-cfg'd deliberately:
                                      # g_NrdLibraryDesc is constexpr and the offsets reach it as
                                      # compile definitions regardless of which shader arm was
                                      # embedded, so a Windows DLL reports the same four and the
                                      # Windows gate thereby protects a value only Vulkan
                                      # consumes. THE TRAP it pins: NRD's CMakeLists sets them as
                                      # (S=0, B=2, U=3, T=20) plain set()s no -D can move, and
                                      # Source/Wrapper.cpp REORDERS them into the struct as
                                      # {sampler, texture, constantBuffer, storageTextureAndBuffer}
                                      # — so a recorder reading the CMake order binds every
                                      # resource at the wrong register (TOOTH FIRED: pinning
                                      # 0/2/3/20 fails). Naming each field in the line is what
                                      # makes that visible instead of four bare integers.
                                      # (b) every pipeline carries THIS platform's blob, through
                                      # ONE selector (PipelineDesc::shader) the recorder shares,
                                      # plus its container MAGIC — non-null-and-nonzero passes on
                                      # garbage. (c) THE DESCRIPTOR LAYOUT IS REALISABLE: the
                                      # binding windows a recorder would build, computed from read
                                      # values and required DISJOINT — measured samplers [0,2),
                                      # cbuffer [2,3), uav [3,10), srv [20,38), which is what makes
                                      # TREG=20 a hand-packed binding map rather than a magic
                                      # number, and it is non-vacuous (it fires the moment a
                                      # denoiser wants >17 storage images or >2 samplers; TOOTH
                                      # FIRED by swapping one offset). (d) pool coherence against
                                      # each pipeline's own resource_ranges. (e) the summed blob
                                      # bytes — the FIRST thing that can tell the perf artifact
                                      # from the standard one at gate time (LibraryDesc carries no
                                      # perf bit, which is why the --nrd-perf pick must be loud):
                                      # measured 523132 B standard vs 452988 B perf.
                                      # THE FINDING, and the reason this gate precedes the recorder
                                      # rather than shipping with it: 9 of 14 SPIR-V blobs sit at a
                                      # NON-4-BYTE-ALIGNED address. That is legal — NRD packs them
                                      # back to back and promises nothing — but vkCreateShaderModule
                                      # takes *const u32, so B4b-ii must COPY each blob into a
                                      # Vec<u32> and must never cast the pointer in place. Reported
                                      # as a NOTE, not a failure; the half that IS a defect (a size
                                      # that is not a whole number of words — SPIR-V is a word
                                      # stream) fails, and reads 0.
                                      # --check-gpu N2
                                      # (pack-vs-oracle, 0 bad px), N3 (passthrough byte-equal),
                                      # N4 (real ReBLUR: Laplacian −70%, energy +0.6%, temporal
                                      # 8× shrink, RESTART departs — N4 RESTORES frame B's
                                      # buffers after, the M10-independence lesson), N5 (sig.w
                                      # sky-0/ranges/must-fires, both suites; each must-fire
                                      # guards on its own no-capture lever — check-dxr on
                                      # dxr_inline_mode()!=0, check-gpu on !--sw-rays, since
                                      # the rt_sw/rt_dxr twins report tmax by design and
                                      # ao_occluded is structurally 0 there), N6 (the RTGI
                                      # fold A/B), N7 (the sky-ext-skip NaN-sentinel proof,
                                      # both GPU suites — plus N2's own sky-constant arm with
                                      # its n2-sky must-fire), N8 (the RE-JITTER
                                      # arms — DXR has none, deliberately: the
                                      # bridge is one TEXT compiled against two
                                      # root-signature objects, so a second
                                      # arm would re-measure identical math,
                                      # and DxrGpu therefore carries no
                                      # force_nrd_rejitter hook at all rather
                                      # than dead code claiming to be one;
                                      # the N2/N3 wiring passes dn_is_nrd=true
                                      # so N3's byte-diff-0 EXERCISES the
                                      # armed re-jitter path — that is where
                                      # the delta form's `0.0 * j == 0.0` is
                                      # checked against real bytes:
                                      # cs_nrd_out re-run TWICE over the
                                      # converged frame's planes with
                                      # force_nrd_rejitter flipped — no
                                      # re-trace, no second ReBLUR pass, so the
                                      # arms differ in one CB bit and the
                                      # verdict is attributable. THE
                                      # ANTI-VACUITY IS THE POINT: N3's
                                      # byte-diff 0 is satisfied BOTH by "the
                                      # delta form correctly multiplies an exact
                                      # zero" AND by "the block never ran", and
                                      # a wiring bug sits between them —
                                      # measured 18209/213200 px moved. TWO
                                      # ASSERTIONS WERE DELIBERATELY NOT MADE,
                                      # because neither is implied: the obvious
                                      # amplitude bound (the re-jittered
                                      # departure from base being at most A×
                                      # the plain one) FAILED on 542 px in its
                                      # first draft and was RIGHT to — the
                                      # composite is jx·D + jy·S over two
                                      # independently-corrected lobes, so where
                                      # D and S nearly cancel the ratio is
                                      # unbounded even though each TERM is
                                      # clamped; and the zero-delta implication
                                      # falls to the same cancellation (a==base
                                      # can mean D=−S). Bounding either honestly
                                      # needs the four NRD planes plus kd/f0 read
                                      # back to reconstruct D and S separately —
                                      # a per-term gate, checking a clamp that
                                      # lives inside NVIDIA's own function.
                                      # Both are REPORTED), and the exact-remod
                                      # / residual-cap family:
                                      # N9-CPU (`remod` in --check —
                                      # nrd::remod::self_test, DLL-free so it
                                      # runs on every platform and owes nothing
                                      # to scene content, which is the point:
                                      # sweeps sheen x tl x dcav x dsun x trans
                                      # x L directly, 432 rows, and asserts BOTH
                                      # directions — the exact arm holds the
                                      # identity to one f16 capture quantum
                                      # (measured 1.40e-4 worst) and the PRE-FIX
                                      # arm PROVABLY FAILS it on every live row
                                      # (4.54e0 worst). Its teeth are a CLOSED
                                      # FORM, not a threshold: the mildest live
                                      # row is sheen-only, whose pre-fix error
                                      # is exactly (L−1)(1−sk)/(L·sk) = 4.26e-2
                                      # at sheen 0.5, L = 2 — asserting that
                                      # identity pins WHY the old arithmetic is
                                      # wrong and by how much, and cannot be
                                      # satisfied by a sweep that merely
                                      # produces large numbers; a companion
                                      # assertion stops any live row falling
                                      # BELOW that anchor, which is what would
                                      # catch a factor list quietly edited down
                                      # to near-unity values. Chromatic rows get
                                      # the convexity BOUND plus a strict
                                      # aggregate improvement, since a scalar
                                      # lane cannot be exact there);
                                      # N6b (check-gpu — the capture-invariance
                                      # arm: two traces across the remod toggle,
                                      # accum BIT-IDENTICAL, m_d ∈ (0,1], and
                                      # the sig must-fire. Reports `lobes`
                                      # SEPARATELY from `sig-differs` on
                                      # purpose: sig.w's high half differs on
                                      # every hit pixel by construction, so the
                                      # whole-sig count is satisfied without the
                                      # dd/ds re-capture having moved anything —
                                      # `lobes` is the re-capture's own,
                                      # scene-dependent liveness, and a
                                      # [1.0000, 1.0000] m_d range prints the
                                      # INERT note);
                                      # N9-GPU (check-gpu — the CAPTURE scored
                                      # on the real shader: two arms, and the
                                      # claim is RELATIVE rather than absolute
                                      # because `base` carries terms the
                                      # reconstruction does not model, so R is
                                      # negative on a large minority of pixels
                                      # even armed (measured base 0.413 vs
                                      # folded 0.376, 3.07e3 over 63755 px,
                                      # IDENTICAL in both arms — the tell that
                                      # it is not this feature's). Soundness:
                                      # the fix may only ever REDUCE
                                      # over-subtraction. Teeth: on the pixels
                                      # whose LOBE lanes actually differ, the
                                      # signed correction must be positive and
                                      # at least 0.1% of their folded energy —
                                      # scored on that SUBSET because a
                                      # frame-wide compare buries it (2.4614e3
                                      # vs 2.4616e3 on san-miguel, a 4-pixel
                                      # margin, teeth in name only). RUNS UNDER
                                      # RTGI BY NECESSITY: off that path the
                                      # bridge rebuilds the ambient as
                                      # ao·sh_irr(n_s) while shade used the
                                      # AMB_BUMP-amplified amb_irradiance(n,n_s)
                                      # — a signed per-channel RGB ratio needing
                                      # the geometric normal, which is not on
                                      # the wire — so a sign gate there measures
                                      # a documented known-approximate instead
                                      # of the remodulation; under RTGI prim.ao
                                      # is 0 and the term is PROVABLY gone
                                      # (`ao-nz 0` in the report). And the teeth
                                      # gate on the MEASURED `live` count, never
                                      # on must_fire, which is FALSE on loaded
                                      # OBJ scenes and TRUE on the procedural
                                      # one — exactly inverted from where the
                                      # feature is exercised, so teeth behind it
                                      # could never fire anywhere, and didn't);
                                      # N11 (check-gpu — the bridge's USE of the
                                      # lane, which N9 provably CANNOT gate: at
                                      # its site OUT equals IN byte for byte, so
                                      # the delta is zero and m_d multiplies
                                      # nothing — planted `m_d = 1.0` inside
                                      # cs_nrd_out and not one byte moved, which
                                      # is how the attempt was caught. N11 runs
                                      # on N4's CONVERGED planes, the only state
                                      # in the suite with a live denoise, and
                                      # FORCES the lane to 1.0/0.5/0.25 with the
                                      # flag ARMED throughout (the N7 ext-fill
                                      # idiom), so the arms differ only in the
                                      # BYTES OF THE LANE and the test is pure
                                      # LINEARITY — col(0.25)−col(1.0) must be
                                      # exactly 1.5x col(0.5)−col(1.0) per
                                      # channel, needing neither D nor kd.
                                      # SCENE-INDEPENDENT by construction, which
                                      # matters because the procedural scene
                                      # derives m_d = 1.0 everywhere. Teeth
                                      # verified BOTH ways: a lane-ignoring
                                      # plant reads `moved 0 px`, a mis-applied
                                      # one (m_d²) reads 248801/250149
                                      # non-linear. Measured green: procedural
                                      # moved 145703 px / live 224927 ch /
                                      # non-linear 0, smlp 150891 / 42180 / 0.
                                      # It also records an UNEXPLAINED
                                      # observation worth resolving before any
                                      # gate trusts the OUT planes at that site:
                                      # the IN and OUT plane readbacks come back
                                      # BYTE-EQUAL while the shader demonstrably
                                      # sees them differ);
                                      # N10 / F10 (check-gpu — the residual cap
                                      # on N4's converged planes, three arms
                                      # differing in ONE CB bit with no re-trace
                                      # and no second denoise, so the verdict is
                                      # attributable; F10 is the same body at
                                      # the FRD site and is what proves the term
                                      # engine-blind. Asserts: the off arm
                                      # byte-identical to N4's frame 8, finite,
                                      # out-of-range untouched, LUMA
                                      # monotonicity (never componentwise —
                                      # see above), agreement with
                                      # nrd::oracle::rclamp_scale pixel for
                                      # pixel (which pins halo geometry, centre
                                      # exclusion, out-of-range exclusion and
                                      # both K constants across shader and CPU
                                      # at once), the 5%-of-mean energy budget
                                      # (the 90%-of-the-pool lesson as a
                                      # number), fired_hard >= fired_on (the
                                      # arms must NEST — equality is the exact
                                      # defect the K_MAX relaxation fixed), a
                                      # DEAD-HALO upper bound (a zero ring makes
                                      # cap ≈ 0 and removes everything: the most
                                      # likely implementation bug in the pass,
                                      # named), and a hard-arm must-fire. WHAT
                                      # IT DOES NOT PROVE, printed as a NOTE
                                      # when both arms read ~0% drop: the trim
                                      # is 1−cap/bl, so a pixel that only just
                                      # clears its cap loses only just more than
                                      # nothing, and on CONVERGED DENOISED
                                      # planes nearly every firing pixel is
                                      # exactly that. A green N10 holds the
                                      # WIRING and the SOUNDNESS, not that the
                                      # cap catches fireflies — that needs a
                                      # noisy 1-spp frame with real glass, i.e.
                                      # the measurement campaign above).
                                      # Two cargo pins ride
                                      # along: NRD.hlsli is byte-equal to the
                                      # submodule file save the one substituted
                                      # line (we INCLUDE, never transcribe), and
                                      # nrd_header is joined at exactly ONE site
                                      # — nrd_bridge_tail — with dxr.rs checked
                                      # for going through it. A THIRD pin came
                                      # with the cap: cs_nrd_out's halo is
                                      # indexed from NRD_OUT_G while the
                                      # hardware launches [numthreads], and HLSL
                                      # gives no way to read that attribute
                                      # back, so `nrd_out_group_shape_is_derived`
                                      # parses both out of the source and
                                      # asserts they agree — a group shape
                                      # changed at the attribute alone would
                                      # leave the ring reading a neighbourhood
                                      # that is not this pixel's, WRONG QUIETLY
                                      # rather than out of bounds (the halo's
                                      # own bounds stay valid because they are
                                      # derived from the same wrong constant).
                                      # Touch nrd.rs /
                                      # nrd_gpu.rs / nrd_bridge.hlsl / the transmit_q_t twins /
                                      # the sig.w pack lane / the remod_blend
                                      # sites or PrimSurf's m_d/amb_k /
                                      # record_*_nrd / gbuf_write_sky's
                                      # skip branch / the four presenter
                                      # branches → run --check, --check-nrd, --check-gpu,
                                      # --check-dxr, --check-fsr, then the STAB smoke on a still
                                      # XeSS view (the 0.42-baseline table above)
                                      # — and, for anything touching the
                                      # remodulation or the cap, the OFF arms
                                      # too (FR_NRD_REMOD=off and
                                      # FR_NRD_RCLAMP=off|on|hard across --check
                                      # and --check-gpu), plus a byte-compare of
                                      # check.png / check_gi.png: they come off
                                      # the CPU path, which never runs the
                                      # bridge, so ANY movement there means a
                                      # capture edit leaked into shading
cargo run --release -- --gpu --fsr3 --nrd  # the same denoiser under the FSR 3.1 chain (and
                                      # --dxr --xess/--fsr3 --nrd for the DXR pipeline — all
                                      # four chain combinations share one nrd_frame_step)
cargo run --release -- --gpu --xess --frd  # FRD (src/frd.rs + gpu/frd_gpu.rs, 2026-08-09) — the
                                      # FROM-SCRATCH, REDISTRIBUTABLE, pure-Rust+HLSL denoiser
                                      # being built to REPLACE NRD in the pre-upscale slot
                                      # (coexist-then-retire: NRD stays the A/B oracle until FRD
                                      # reaches its STAB 0.07-0.10 band, then the whole NRD stack
                                      # — nrd.rs, nrd_gpu.rs, the install-prerequisites nrd
                                      # component (the ONE CMake+VS dependency), the version-pin
                                      # gate — dies in one commit). CLEAN-ROOM RULE, load-bearing:
                                      # the NRD source tree is NEVER read/quoted/transcribed
                                      # (extends the nrd::oracle never-paste rule; do not open
                                      # %TEMP%\frustracer-prereqs); the design comes from the
                                      # published literature (RTG II ch. 49 ReBLUR, the GDC
                                      # self-stabilizing-recurrent-blurs talk, SVGF) — plan doc:
                                      # the ReBLUR-class recurrent 3-dispatch shape (temporal /
                                      # blur / post+feedback) at fp16, wave-ops, narrow barriers,
                                      # B70-first (XMX ruled out: bilateral weights are not a
                                      # GEMM). NOT THE DEFAULT — NRD IS. This constant moved
                                      # TWICE and the second move is the one that sticks, so
                                      # read the order: FRD took the slot at the PHASE-E FLIP
                                      # (2026-08-09 — the parity bar was met, quality band held
                                      # on both boxes at 2.5x NRD's speed), and d1b315f
                                      # REVERSED it, because the emissive-integration campaign
                                      # measured NRD ahead where this content lives (68% vs 42%
                                      # of DLSS-RR's pool delivery, still-frame stability 0.29
                                      # vs 0.47) and the parked-camera darkening report is the
                                      # visible half of that gap. THE CODE IS THE ARBITER when
                                      # this entry and the --nrd one disagree: cli.rs's
                                      # defaults() reads `frd: false`, and the shipping --help
                                      # says "--no-frd spells the default (NRD holds the slot)".
                                      # The DELETION half of Phase E is retired with the flip;
                                      # FRD stays compiled and reachable as --frd on its own
                                      # merits — 2.5x faster in the denoiser region, and the A/B
                                      # oracle that isolated every bug in that campaign. It
                                      # takes the ONE
                                      # denoiser
                                      # slot (enum DnGpu in gpu/mod.rs — arm_denoiser_for /
                                      # nrd_frame_step / the shed machinery are engine-blind, the
                                      # bridge kernels + NRD_FEED_SET + nrd_sig/sky-ext-skip
                                      # wiring UNCHANGED: FrdGpu carries NrdGpu's exact plane
                                      # contract); only EXPLICIT pairs are fatal — explicit
                                      # --frd + explicit --nrd exits 2, explicit --frd + --nppd
                                      # exits 2, --frd CLAIMS the slot from the nrd default, and
                                      # a bare --nppd disarms a FILE-SAVED frd loudly (a default
                                      # never makes another flag fatal — the fg_explicit
                                      # doctrine); the not-armed session notes
                                      # key on frd_explicit/nrd_explicit, so a default never
                                      # nags; --no-frd spells the default (NRD holds the slot)
                                      # and --no-nrd --no-frd is the plain undenoised
                                      # XeSS/FSR3 baseline; settings rows `frd`
                                      # (Toggle, default OFF) + `nrd` (default ON)
                                      # drive the default arms only (the fg-row rule).
                                      # --frd-max-accum-frames/-fast-frames/-max-stab-frames/
                                      # -blur-radius/-clamp-sigma/-[no-]anti-firefly/-no-fp16 =
                                      # the tuning family (frd::FrdTuning, all-None = compiled
                                      # constants; --frd-no-fp16 forces the fp32 shader arm).
                                      # PHASE STATUS: A+B+C SHIPPED (2026-08-09) — the plan's
                                      # 3-dispatch recurrent shape is LIVE: cs_frd_temporal
                                      # (reproject off the wire 2.5D MV, per-foot disocclusion
                                      # with the grazing-relaxed relative-Z test, slow/fast
                                      # accumulation, Welford variance) -> cs_frd_blur
                                      # (8-tap Vogel-disk bilateral at a hit-dist-driven,
                                      # accumulation-scaled radius; history fix folded in as the
                                      # n<N_FIX radius boost) -> cs_frd_post (1.7x disk,
                                      # fast-history clamp, writes OUT + the RECURRENT slow
                                      # feedback — pass 3's output IS next frame's history, the
                                      # self-stabilizing compounding that makes a 30-frame cap
                                      # converge; slow is single-buffered, pass 1 reads before
                                      # pass 3 rewrites). Tap rotation = a pure integer hash of
                                      # (pixel, frame, salt) — zero rng draws; tuning rides root
                                      # constants (a lever never recompiles); per-pass
                                      # frd/frd-temporal/frd-blur/frd-post pix::scope regions
                                      # feed --gpu-timing from day one. force_passthrough is the
                                      # F3 control arm (the spatial passes fire on reset frames
                                      # BY DESIGN — that wide blur IS the history fix).
                                      # MEASURED (4090, procedural still, native 1080p,
                                      # FRUSTRACER_STAB — taken while other sessions ran
                                      # benchmarks, so re-confirm solo before recording as
                                      # canonical): XeSS-alone 0.79 -> FRD 0.09-0.11 (the NRD
                                      # 0.07-0.10 band; Phase B temporal-only read 0.27); F4
                                      # Laplacian 0.1264 -> 0.0436, mean drift 0.9%, temporal
                                      # shrink 5.8x; B70 solo re-confirm 2026-08-09: FRD 0.06 vs
                                      # NRD 0.04-0.05 on the converged still — band held.
                                      # THE PHASE-D PERF CAMPAIGN (2026-08-09, same day —
                                      # triggered by a user world test reading "NRD 36 FPS vs
                                      # FRD 31", which decomposed to a MISLABELED BASELINE: that
                                      # checkout had no NRD.dll, so the "NRD" arm shed loudly to
                                      # PLAIN upscaling and the 4.5 ms delta was FRD-vs-NOTHING,
                                      # not FRD-vs-NRD — check the `nrd: armed` vs `nrd:
                                      # unavailable` line before believing any denoiser A/B; the
                                      # DLL is now built on this checkout and N4/F8 run live).
                                      # THE REAL HEAD-TO-HEAD (B70 world parked, native 1080p,
                                      # same protocol both arms, foreign-PID check per d2d5f6a):
                                      # FRD region 0.488 ms vs NRD 1.204 — 2.5x FASTER, whole
                                      # frame span 5.42 vs 6.12; 4090 FRD region 0.226. Three
                                      # moves got it from the first-light 0.543: (1) the
                                      # TAP-LOOP DIET — the fused log2-domain tap weight
                                      # (oracle::tap_exp2 + frd_tap_exp2: Gaussian/z/normal/
                                      # hit-dist exponents SUM under ONE exp2; the Gaussian's
                                      # r_i^2 = (i+0.5)/TAPS is a per-index constant so the
                                      # per-tap length/sqrt died with it) + the literal Vogel
                                      # table rotated by one per-pixel sincos (VOGEL8/vogel_rot,
                                      # F0-pinned to the analytic generator) + pass-1 feet
                                      # testing b==0 and z-validity BEFORE the prev_nr
                                      # load+decode — measured only −2% on the B70 (0.543 ->
                                      # 0.531: the passes are LATENCY-bound, not EM-pipe-bound,
                                      # the d2d5f6a guide-campaign shape again) but ships as
                                      # strictly-less-work with F4 statistically identical;
                                      # (2) the GROUP-SHAPE SWEEP (FR_FRD_GROUP=TXxTY,BXxBY —
                                      # loud on departure, loud+defaults on illegal; every cell
                                      # a new Arc variant, maiden discard): temporal wants 8x8
                                      # (0.233 -> 0.224) while blur/post want 16x16 (0.294 ->
                                      # 0.260 combined; 8x8 was WORST there — the two pass
                                      # families genuinely prefer opposite shapes, so per-pass
                                      # constants, never one), adopted as compiled defaults and
                                      # re-verified a 4090 WIN too (0.260 -> 0.226); (3) the
                                      # fp16 arm is DEFERRED WITH REASONING — the diet's null
                                      # result is the proxy measurement (ALU-rate wins don't
                                      # move latency-bound passes), the group sweep already
                                      # banked the occupancy half, and 0.488 sits 2.5x under
                                      # the bar; the FRD_FP16 typedefs + compile_args hook +
                                      # OPTIONS4 probe stay ready if a future res/scene regime
                                      # re-opens it. THE BRIGHT-SPECULAR SMEAR FIX (2026-08-09,
                                      # three commits — the user's "sun glints trail" report,
                                      # confirmed repro: a PARKED camera with the sun moving,
                                      # where NO motion vector of any kind can express the
                                      # glint's motion since prev camera == cur camera makes
                                      # every reprojection the identity): (A) the FIREFLY
                                      # PRE-CLAMP is BUILT — pass 1 soft-clamps each input's
                                      # luma to FIREFLY_K × its 8-NEIGHBOR RING mean (center
                                      # EXCLUDED — center-in makes a lone outlier's own
                                      # contribution dominate its cap, an 11% trim however
                                      # extreme; the ring crushes it to K× its surround while a
                                      # real multi-pixel glint survives proportionally) BEFORE
                                      # accumulation, so an F0-demodulated outlier can neither
                                      # seed the slow history nor inflate the Welford m2 that
                                      # sizes its OWN clamp box; --frd-[no-]anti-firefly is
                                      # LIVE, default ON. The ANTI-LAG BRAKE is BUILT: pass 3
                                      # records the clamp excess as g = 1/max(e,1) into a new
                                      # single-buffered R8G8_UNORM antilag plane (SRVS 12→13),
                                      # pass 1 cuts the reprojected age by it — TWO robustness
                                      # rules the F5/F6 gates found live, do not remove: the
                                      # cut FLOORS at the history-fix window (frd_antilag_apply
                                      # — cutting below N_FIX re-arms the wide history-fix
                                      # blur, which wipes sharp features from the feedback and
                                      # re-fires the clamp, a PERMANENT limit cycle on
                                      # converged sharp features), and the excess denominator
                                      # FLOORS RELATIVELY (frd_antilag_excess, ANTILAG_E_FLOOR
                                      # = 1% of fast luma — a converged signal's box is exactly
                                      # zero-width while blur fp residue is ulp-scale-of-
                                      # RADIANCE, so an absolute floor mints junk gains on
                                      # perfectly converged pixels). And the specular parallax
                                      # gains the LIGHT-MOTION term: refresh_sky captures the
                                      # sun direction, nrd_frame_step retains the prev, and 2×
                                      # the per-frame angular delta (the mirror-swing factor,
                                      # oracle::light_parallax — bit-equal dirs are EXACTLY 0)
                                      # SUMS with the camera term into frd_spec_max_frames —
                                      # the cap is the mechanism content motion needs.
                                      # FR_FRD_SUNPAR=off|<gain> / FR_FRD_ANTILAG=off are the
                                      # repro arms. (B) v1.5 SPECULAR VIRTUAL MOTION is BUILT:
                                      # the spec history fetch takes a second reprojection
                                      # point — the reflection's virtual image at t_surf + t_r
                                      # (t_r off the wire's normalized hit-dist via
                                      # frd_hitdist_denorm_factor, exact below saturation)
                                      # unfolded along the primary ray and projected through
                                      # the prev world→clip in the DELTA form (surface AND
                                      # virtual through the SAME matrix, only the difference
                                      # added to the measured-MV q: jitter cancels first-order,
                                      # t_r = 0 exactly zero, a parked camera's fp residue
                                      # snaps to 0 under VM_DEADZONE2, behind-camera takes the
                                      # humble arm); t_r ≥ VM_FAR_K(0.99)·far is the rotation-
                                      # only sky limit (project the DIRECTION, w = 0 — the
                                      # wire_cam_far f16 lesson); roughness fades the
                                      # COORDINATE over VM_ROUGH_LO..HI (one fetch, never two);
                                      # the displaced fetch runs its own foot loop with the
                                      # relative-Z test slack-widened by the applied offset
                                      # (VM_Z_SLACK — grazing planar mirrors vary reflector
                                      # depth along the plane); the unfold's t_r reads the
                                      # ACCUMULATED hit-dist at the surface-reprojected texel
                                      # (raw 1-spp .w jitters the fetch position — measured
                                      # live as strafe smear), and the parallax cap STAYS the
                                      # crude cam_step/z + light_par (the measured divergence
                                      # is ALWAYS ≤ cam_step/z, so capping by it LENGTHENS
                                      # history exactly in proportion to trust in the planar-
                                      # still unfold — rippled water/curved reflectors broke
                                      # that trust, the 2026-08-10 strafe-smear regression;
                                      # v1.5 = corrected FETCH under the SAME conservative
                                      # cap, relaxing it is v2's confidence term). v1.5.1
                                      # (2026-08-10, the helmet sun-streak fix — the user's
                                      # exact diagnosis): the FLAT-mirror sky arm is
                                      # ANTI-correct on a CURVED mirror — a convex mirror
                                      # images an infinite object at its focal point ~R/2
                                      # BEHIND the surface (the paraxial mirror equation), so
                                      # a helmet glint's true screen motion rides the SURFACE,
                                      # and the screen-pinned sky fetch re-painted every
                                      # vacated pixel with its own stale bright history (an
                                      # ACCEPTED stale self-fetch: the per-frame normal drift
                                      # at |mv| px sits UNDER the spec ladder, so the stale
                                      # history keeps being accepted). The fix is the mirror
                                      # equation on the unfold distance BEFORE the sky test
                                      # and the construction — t_v = t_r/(1 + 2κ·t_r)
                                      # (frd_virtual_dist: exact at ALL object distances;
                                      # κ = 0 is a BRANCH returning t_r verbatim, the bitwise
                                      # flat path; κ·t_r ≫ 1 ⇒ t_v → 1/(2κ) ≈ R/2 ⇒ the fetch
                                      # collapses to the surface reprojection; for κ ≥ 0,
                                      # t_v ∈ (0, t_r] monotone — κ errors INTERPOLATE between
                                      # the two shipped behaviors, never extrapolate) — with κ
                                      # from dead-zoned central differences of the DECODED
                                      # wire normals (VM_DN_DZ = 5e-3
                                      # SUBTRACTIVE — continuous through the boundary, sized
                                      # so 10-bit-oct quantization reads exactly 0;
                                      # magnitude-only, concave reads convex = the humble
                                      # direction; a constant field is ONE encoded word ⇒
                                      # Δn ≡ 0 bitwise ⇒ still water/F7's mirror untouched by
                                      # construction). v1.5.2 (same day — the CLOSE-UP
                                      # BIDIRECTIONAL-streak feel-test, lever-confirmed:
                                      # leading streak = κ OVER-read from normal-map detail
                                      # riding the wire's SHADING normals ⇒ the fetch rides
                                      # the surface FASTER than the true virtual image, which
                                      # moves z/(z+R/2) slower; trailing = κ under-read, the
                                      # close-up dome's macro signal at the dead-zone):
                                      # frd_vm_kappa runs FOUR estimates — 2px AND 4px
                                      # baselines per axis (8 loads + decodes, vm-gated; the
                                      # same absolute DZ at both scales means the 4px read
                                      # RESCUES the close-up macro signal the 2px read
                                      # dead-zones away) — per-axis MIN across scales (bump
                                      # noise decorrelates across scales, macro curvature
                                      # doesn't: the min kills single-scale spikes, the
                                      # leading-streak suppressor), MAX across axes
                                      # (cylinders track their curved axis), and the
                                      # κ_lo/κ_hi BRACKET's projected fetch spread charges
                                      # the specular history cap (vm_unc on the parallax
                                      # line, only ever TIGHTENING it): where the estimator
                                      # disagrees with itself (bumpy visor, dead-zone-
                                      # straddling close-ups, cylinders' genuine ambiguity)
                                      # history collapses and NEITHER streak can accumulate —
                                      # short-history noise the spatial passes cover, never a
                                      # confident wrong fetch. The v2 confidence term, cheap
                                      # form. Rippled water's κ fires and that is
                                      # CORRECT (crests are genuine convex micro-mirrors —
                                      # the flat unfold there was the d23ecc3 regression).
                                      # Residual known-accept: bumps aligning at BOTH scales
                                      # and axes still over-read with a narrow bracket (v2's
                                      # content-based confidence is the real answer). NOTE
                                      # the beyond-the-path brightness the same feel-test
                                      # raised: under pure translation the sky-arm fetch is
                                      # the pixel's OWN old position (decay-in-place —
                                      # transport is impossible, and F7 pins the delta's
                                      # sign/magnitude to the oracle), so brightness past the
                                      # glint's historical path is the RECURRENT SPATIAL BLUR
                                      # halo of a long-lived bright trail (pass 3's blurred
                                      # output IS next frame's history — the halo compounds
                                      # while the trail lives); v1.5.2 removes the trail
                                      # fuel, and the identified follow-up if residue
                                      # survives is a luma-ratio tap guard in frd_disk (the
                                      # firefly ring clamp's spatial sibling).
                                      # v1.5.3 (2026-08-10 — THE FIRST CLOSED-LOOP AI-QA-LAB
                                      # CAMPAIGN: the residual "faster = wider" strafe smear
                                      # the user reported, diagnosed and fixed with ZERO human
                                      # feel-tests — live frqa strafe passes with FWHM
                                      # measurement on the screenshots, the NRD oracle A/B,
                                      # the per-lever live sweep, and the batch lab as the
                                      # regression pin; the whole method is the --frd-lab/--qa
                                      # entries' reason to exist). TWO κ-estimator defects,
                                      # both invisible at the 540p lab/gate resolutions:
                                      # (a) RESOLUTION DEPENDENCE — the 2px/4px baselines are
                                      # TEXEL counts while VM_DN_DZ is absolute per-sample
                                      # quantization noise, so at 1080p each baseline spans
                                      # HALF the 540p world distance, |Δn| halves against the
                                      # same DZ, κ under-reads, and the fetch slides toward
                                      # the flat/sky arm — the v1.5.1 trailing streak REOPENED
                                      # at exactly the res users play (measured live: glint
                                      # FWHM 47 still → 141 px at the fastest strafe while
                                      # the 540p lab read the same world pass clean; NRD and
                                      # --no-frd stayed ~50 at every speed — the two-arm A/B
                                      # that pinned it on FRD; FR_FRD_VMOTION=off BEATING the
                                      # shipping vm arm was the tell that κ under-read, since
                                      # a close-up dome's correct behavior ≈ the surface
                                      # fetch). Fix: oracle::vm_baseline_scale = max(1,
                                      # round(rh/VM_BASE_RH=540)) scales the sample OFFSETS
                                      # (±s/±2s texels — the same WORLD footprint at every
                                      # res, DZ ratio restored; s is integer, session-fixed,
                                      # rides the CB's ex-pad dword 17, loud when ≠1); every
                                      # gate res (533x400/800x600) and the lab's 540p keep
                                      # s=1 BITWISE. FR_FRD_VMSCALE=off|<n> is the repro/
                                      # force lever (measured: off returns 96 of the 140 px).
                                      # (b) THE MIN-ACROSS-SCALES KEPT THE WORST-BITTEN READ —
                                      # on a clean field k4 = k2 + skew EXACTLY (skew =
                                      # DZ·proj/(2·b1·z), the self-test closed form), so the
                                      # per-axis min ALWAYS returned the short baseline, the
                                      # one the subtractive DZ ate most of (~37% of true κ on
                                      # the live dome ⇒ t_v overshoots ~2.7× ⇒ the fetch lags
                                      # the content ⇒ velocity-proportional trailing residue)
                                      # — and the documented "4px rescue" NEVER FIRED
                                      # (min(0, k4) = 0 discards exactly the close-up macro
                                      # signal it was written for). Fix: the DE-BIASED combine
                                      # — the clean-field identity true κ = k2 + 2·skew = k4
                                      # + skew makes min(k2+2·skew, k4+skew) EXACTLY unbiased
                                      # on clean fields and spike-bounded both ways (either
                                      # single-scale spike leaves the other arm's clean-field
                                      # extrapolation; both-spiked stays the v2 known-accept);
                                      # extrapolation is GATED on both reads clearing the DZ —
                                      # a constant field returns exactly (0,0,0) (the bitwise
                                      # still-water/F7 contract, no skew leaks), a
                                      # 2px-dead-zoned close-up takes the BARE k4 (the real
                                      # rescue, humble), short-only signal stays 0 (bump
                                      # noise); κ_hi additionally covers the applied κ so the
                                      # vm_unc bracket only widens. MEASURED LIVE (1080p
                                      # deflection sweep, glint FWHM px vs the ~50 px clean
                                      # band): still 47 | 0.015 → 72-83 pre / 61-76 post |
                                      # 0.03 → 86-89 / 57-62 | 0.06 → 140-142 / 57-88 —
                                      # INSIDE NRD's own 53-81 at every speed, aspect back to
                                      # round; the velocity term is gone. F0's vm_kappa
                                      # family REWRITTEN to the new closed forms (unbiased
                                      # clean-field anchor + de-bias teeth, spike suppression
                                      # to k4+skew, the REAL rescue pin, the exact-(0,0,0)
                                      # flat pin, short-only-spike-is-0, bracket = 2·skew,
                                      # the baseline-scale rule table, and the BITWISE
                                      # res-invariance identity — 2× proj with 2× scale must
                                      # reproduce κ exactly, powers of two); F7C green with
                                      # kap 1.17 → 1.38 (the de-biased read), gapF/gapC
                                      # unmoved (0.049/0.004 — repro arm and collapse intact).
                                      # LAB LESSON, load-bearing: the batch lab at 960x540
                                      # measured v1.5.2 "clean" while shipping keep sat at
                                      # 0.15-0.27 — a RELATIVE teeth verdict (flat vs
                                      # shipping) can't flag the shipping arm degrading, and
                                      # an instrument at the wrong RESOLUTION can't see a
                                      # res-dependent estimator at all; the live half of the
                                      # QA lab (real render res, real upscaler) is what
                                      # caught both.
                                      # v1.5.5 (2026-08-10 — the B70 1-spp MAX-strafe BLOTCH,
                                      # the third closed-loop lab campaign in one day: the
                                      # user's session shape was XeSS+fg×2 on the B70 — where
                                      # XeSS-FG generates — but the blotch survived --no-fg,
                                      # --no-frd was clean, and the live lever sweep pinned
                                      # it: FR_FRD_VMOTION=off clean, --frd-blur-radius 0
                                      # still dirty ⇒ the vm FETCH, not the blur). MECHANISM:
                                      # at ~0.2·rh of per-frame glint motion the uncapped
                                      # vm_slack (1 + 0.05·|appl|) widened the z-test 16-40×
                                      # — vacuous — so displaced fetches accepted bright
                                      # history ANYWHERE on the dome (every position passes
                                      # the slowly-varying normal/z ladders there), and the
                                      # rate-only parallax cap still allowed 5-8 frames ⇒
                                      # frames × px/frame = a dome-wide swath. FIVE layers,
                                      # each only ever TIGHTENING, each with F0 pins:
                                      # (a) VM_SLACK_CAP=60 px — beyond it the widening
                                      # stops (below-cap bitwise; F7's 38 px untouched);
                                      # (b) VM_APPL_UNC_K=0.1 — |appl|/proj joins the
                                      # history cap (fetch error is MULTIPLICATIVE in the
                                      # offset); (c) the MAGNITUDE FADE (VM_FADE_LO/HI =
                                      # 0.08/0.20·rh, res-invariant): past HI the fetch
                                      # collapses onto the SURFACE fetch (measured clean at
                                      # max strafe) — the FG dt-fade shape on the fetch
                                      # position; (d) the SMEAR BUDGET (SPEC_PX_BUDGET =
                                      # 0.38·rh): n_smax is ALSO capped by budget/(per-frame
                                      # glint motion in px) — the rate cap bounds frames-per-
                                      # radian while the ARTIFACT is frames × px/frame;
                                      # restart-class at max strafe, inert wherever the rate
                                      # cap already binds (parked, TOD scrubs, slow strafes;
                                      # oracle::spec_budget_frames, floored at 1);
                                      # (e) the LUMA-RATIO TAP GUARD in frd_disk
                                      # (TAP_LUMA_K=4 — v1.5.2's documented follow-up,
                                      # finally forced: with the budget keeping history young
                                      # under sustained fast motion the history-fix radius
                                      # stays wide, and an 800×-mean glint tap averaged into
                                      # dim neighbors was the residual soft BLOOM; within-K
                                      # taps ride a multiply-by-exactly-1.0 — ordinary
                                      # fields bitwise, the F3/F4 contract). VERIFIED LIVE
                                      # (B70 max-deflection passes, desktop captures): blob →
                                      # compact glint with a small motion tail, at the
                                      # --no-frd baseline; 4090 normal-speed regression table
                                      # unmoved (0.03→59 px, 0.06→59 px — the v1.5.3 band's
                                      # good end); F5/F6/F7/F7C green with F7 gap 0.184 vs
                                      # pred 0.182; the 540p lab canonical keep 0.29 with
                                      # teeth firing. LESSON: throttling fps via spp to match
                                      # a user's per-frame step changes INPUT QUALITY too —
                                      # the first campaign's spp-8 throttle hid this 1-spp
                                      # failure; match spp as well as step.
                                      # FR_FRD_CURV=off is the flat-mirror repro arm (flags
                                      # bit 4, force_curv the gate hook). Pure
                                      # rotation needs none of this (both
                                      # points share the ray through the unmoved origin — the
                                      # surface MV is already correct there). CB_DWORDS 17→47
                                      # (prev world→clip + CamBasis org/rgt/up, the ngxfg
                                      # conventions verbatim; the oracle is CROSS-PINNED
                                      # against ngxfg_guides::virtual_prev_px — one unfold, two
                                      # engines, one pin). FR_FRD_VMOTION=off is the repro arm.
                                      # FrdFrame is record()'s params struct; force_fire/
                                      # force_antilag/force_vmotion are the gate hooks (the
                                      # force_sky_ext_skip shape — OnceLock levers can't flip
                                      # in-process). STILL PENDING in C: hit-dist
                                      # reconstruction, the 3x3 valid-foot fallback
                                      # (shader-header notes); F8 as a DEDICATED gate (N4 + F4
                                      # now run live on the same protocol/inputs — the
                                      # report-only comparison — but the promoted must-fire
                                      # form is unbuilt). Gates: F0 (`frd` in
                                      # --check —
                                      # frd::oracle::self_test: reprojection convention,
                                      # disocclusion anchors + grazing relaxation, the
                                      # running-mean accumulation identity, Welford variance,
                                      # clamp idempotence + box widths, firefly cap, the
                                      # antilag gain/apply/excess families, light-parallax
                                      # anchors (bit-equal ⇒ ABSOLUTE exact 0 — the strafe-gate
                                      # lesson), the vm family (parked exact-0 snap, t_r=0
                                      # exact, moved-camera must-fire, sky-vs-finite must-
                                      # differ, the ngxfg cross-pin, slack-1.0 identity, the
                                      # v1.5.1/.2/.3 curvature family — virtual_dist κ<=0
                                      # bitwise identity/R-over-2 limit/monotone,
                                      # vm_curvature_at dead-zone exact-0 + closed-form
                                      # anchor, the v1.5.3 vm_kappa forms (UNBIASED
                                      # clean-field anchor + de-bias teeth, spike suppression
                                      # to k4+skew, the real 4px-rescue pin, exact-(0,0,0)
                                      # flat, short-only-spike-0, axis symmetry,
                                      # honest-cylinder-κ_lo-0, bracket = 2·skew, the
                                      # baseline-scale rule table + the BITWISE
                                      # res-invariance identity), and the composed
                                      # heavy-κ snap-to-surface pin),
                                      # bilateral
                                      # weight shapes, radius endpoints, Vogel-disk spread, and
                                      # the wire re-export pin vs nrd::oracle); F1 (check-gpu:
                                      # instance contract + the fp16 probe); F3 (check-gpu, the
                                      # structurally-critical one: pack -> FrdGpu.record ->
                                      # out must reproduce cs_feed_xess's color BYTE-identically,
                                      # with a measured dirty-pass anti-vacuity — 629341 channels
                                      # provably dirtied, then byte-diff 0); F4 (check-gpu,
                                      # DLL-free — the N4 protocol at FrdGpu, scored on the
                                      # CONVERGED frame 7 since FRD's reset frame restarts
                                      # history — one spatial-only wide blur, not the recurrent
                                      # loop: finite, differs, Laplacian drops, mean
                                      # <=25%, temporal shrink, RESTART departs, frame-B
                                      # restore); F5 (check-gpu, HAND-BUILT planes — the N2
                                      # style, write_tex_at is the upload twin of read_tex_at:
                                      # the firefly A/B with teeth both ways — a lone 400-luma
                                      # outlier must survive bright OFF and crush to ~the ring
                                      # cap ON, measured 20.4 vs 0.27 — plus the CONVERGED-
                                      # INERT antilag pin: a converged uniform field's recorded
                                      # gains must be EXACTLY 255 everywhere, the teeth against
                                      # the junk-gain class); F6 (the parked-camera moving-
                                      # glint probe, the user's repro shape: a converged 3x3
                                      # glint's input moves +24 px, three arms — light_par 0.5
                                      # must dump the stale history (0.102), the brake arm must
                                      # lead its off arm on the real ghost with a recorded
                                      # sub-1 gain (g-min 0.016 — sampled DURING convergence,
                                      # where the history-fix-vs-clamp fight actually mints
                                      # gains), and the all-off pre-fix arm must FAIL the bound
                                      # (1.114, the teeth)); F7 (virtual-motion tracking: a
                                      # converged x-gradient mirror history + ONE strafe frame
                                      # with real matrices and the exact surface-MV plane — the
                                      # vmotion-on/off arms' OUT gap must land within ±50% of
                                      # slope·|oracle d.x|, measured 0.189 vs 0.182 predicted
                                      # at |d.x| = 38 px; plus the curv-on/off BYTE-identity
                                      # replay on the flat probe — the κ=0-branch contract on
                                      # the wire, at a reset frame_index so the tap-rotation
                                      # hash matches); F7C (the CURVED-mirror variant, the
                                      # helmet-streak repro: a 1°/px world-anchored normal
                                      # field — the probe frame samples it at each pixel's
                                      # PREV screen position, an unshifted field disoccludes
                                      # every arm and the gate goes vacuous — far = 50 puts
                                      # the flat t_r into the REAL sky arm, strafe 0.06 keeps
                                      # the stale self-fetch's ~4.9° drift UNDER the 6.6°
                                      # ladder so the streak mechanism actually runs; three
                                      # arms — the flat arm must TRACK the sky oracle, gapF
                                      # 0.049 vs pred 0.048, the bug repro with teeth; the
                                      # curv arm must collapse to ≤ 0.35x of it, measured
                                      # 0.004 = 12x reduction; anti-vacuity: quantized-round-
                                      # trip dn ≥ 2·DN_DZ, |d_sky| ≥ 2.5 px, oracle d_curv ≤
                                      # 0.15·|d_sky|). dxc::compile_args is the per-unit
                                      # -enable-16bit-types hook the phase-D fp16 arm uses.
                                      # Next: C-completion (hit-dist-recon/3x3-fallback)
                                      # + F8, further D tuning (fp16 arm, wave ops,
                                      # plane-distance bilateral upgrade, stabilization,
                                      # specular virtual-motion v2 — model CONFIDENCE to
                                      # relax the translation cap, signed/concave curvature),
                                      # then
                                      # E's
                                      # remaining half — the NRD deletion, which is NOT
                                      # PENDING BUT RETIRED: E's flip half shipped 2026-08-09
                                      # and was REVERSED by d1b315f the same week, so NRD is
                                      # the default and deleting it is off the table until FRD
                                      # wins the quality argument it lost (see the precedence
                                      # block above for the numbers).
                                      # THE SMEAR-FIX INTERACTIVE PROTOCOL (the feel-test):
                                      # (1) moving sun, parked camera — `--gpu --xess --frd` on
                                      # rungholt water or a bistro glossy floor, park, hold `.`
                                      # to scrub TOD; arms: fixed default → FR_FRD_SUNPAR=off →
                                      # --frd-no-anti-firefly → FR_FRD_ANTILAG=off (each
                                      # returns its slice of the trail) → --nrd (the oracle A/B
                                      # — check the `nrd: armed` line first, the mislabeled-
                                      # baseline lesson). (2) camera translation — strafe past
                                      # water/helmet sky glints with --no-fg; default vs
                                      # FR_FRD_CURV=off (the helmet streak's repro arm) vs
                                      # FR_FRD_VMOTION=off vs --nrd; still water's sky
                                      # reflection must stay PINNED (constant normal word ⇒
                                      # the bitwise flat path). (3) regression —
                                      # FRUSTRACER_STAB still in the 0.06-0.11 band,
                                      # --gpu-timing frd-temporal before/after.
                                      # Touch frd.rs / frd_gpu.rs / the DnGpu enum /
                                      # arm_denoiser_for -> run --check, --check-gpu, --check-dxr,
                                      # --check-nrd (must stay untouched while NRD lives),
                                      # --check-fsr, then the --gpu --xess --frd armed-line smoke
cargo run --release -- scenes/damaged-helmet/DamagedHelmet.glb --frd-lab strafe --frd-lab-speed 8 \
                        --cam 3.4,2.5,2.3,0,2.4,0 --tod 8
                                      # THE AI QA LAB, batch half (main.rs::run_frd_lab,
                                      # 2026-08-10) — the AGENT'S OWN FEEL-TEST: a headless dev
                                      # INSTRUMENT (the --spin class, exit 0 always, never a
                                      # gate) that reproduces and MEASURES the bright-specular
                                      # streak family with the real wavefront tracer feeding the
                                      # real FrdGpu, so streak iteration no longer round-trips
                                      # through a human. RUN IT BEFORE ASKING THE USER TO STRAFE.
                                      # The line above is the CANONICAL HELMET REPRO (glint peak
                                      # 141.8 = 800x scene mean on the curved shell, dome + the
                                      # v1.5.2 hex pattern in frame). Structure: the F4 harness
                                      # (HeadlessGpu -> TraceGpu(gbuf_full,nrd) -> wire_nrd_feed
                                      # -> per-frame trace/pack/FrdGpu/out) with REAL prev
                                      # threading (the nrd_frame_step Frd-arm math verbatim —
                                      # the presenter contract F4's parked pose never runs),
                                      # over converge(32f) -> motion at --frd-lab-speed px/frame
                                      # (surface speed at the probed center depth; travel
                                      # clamped to 45% of width — unclamped, the subject
                                      # marched out of frame and the corridor measured leftover
                                      # sky) -> FRESH-history converge at the final pose (the
                                      # ground truth; parking on the trail would contaminate
                                      # it). Kinds: strafe (the repro) | dolly | orbit + tod
                                      # (negative controls — orbit reads ~0 on all arms, the
                                      # user-confirmed rotation-never-repros; tod residue is
                                      # legitimate but must be ARM-UNIFORM, the shared
                                      # light-par cap regime). THREE A/B ARMS per run via the
                                      # force Cells (no env restarts): v152 = shipping | flat =
                                      # force_curv off (the helmet-streak repro) | novm =
                                      # force_vmotion off (the surface-MV arm). METRICS, all
                                      # corridor-disciplined around ONE tracked glint (shared
                                      # cross-arm lock; per-frame local argmax + windowed
                                      # centroid — whole-frame centroids merged unrelated
                                      # blobs and measured a 216px phantom lag; analytic -x
                                      # axis on strafe; the GLINT-LIFETIME gate stops metrics
                                      # when the input's local peak fades under 5% of initial
                                      # or the track jumps — a strafed mirror point slides off
                                      # the shell and the re-lock onto a DIFFERENT glint
                                      # measured a phantom 83px trail): lag, trail/lead px past
                                      # the converged half-extent, beyond-path energy (the
                                      # "past where the sun ever was" question, as a number),
                                      # keep = out local peak / conv peak (a streak SPREADS
                                      # energy — at low speeds the trail sits under T_streak
                                      # while keep collapses: flat reads 0.04 vs shipping
                                      # 0.21-0.27 at speeds 4-8, the suppression signature;
                                      # teeth = trail OR keep), end_err vs truth, a 4px-bin
                                      # profile, and tone+log frd-lab-*.png dumps the agent
                                      # READS itself. MEASURED at the canonical line: teeth
                                      # fire at speeds 4-12 (flat trail 99px vs shipping 14 at
                                      # speed 12), novm shows the predicted LEADING signature,
                                      # shipping v1.5.2 reads clean at every speed. Sub-flags:
                                      # --frd-lab-speed (default 4) / -frames (24) / -res
                                      # (960x540); exclusive with --spin/--check*/--cinematic,
                                      # never loads the world; the --frd-lab prefix family
                                      # rides one headless_args clause; cli::self_test pins the
                                      # parse (known-kind optional value — a scene path is
                                      # never swallowed, an unknown kind exits 2)
cargo run --release -- scenes/damaged-helmet/DamagedHelmet.glb --qa --cam 3.4,2.5,2.3,0,2.4,0 --tod 8
                                      # THE AI QA LAB, live half (src/qa.rs + session()'s drain,
                                      # 2026-08-10) — the LIVE QA CONTROL SOCKET: a localhost
                                      # TCP line protocol (--qa [port], default 4599, 127.0.0.1
                                      # only — the bind IS the security boundary, like the dev
                                      # console it mirrors) that lets a scripted driver run the
                                      # REAL interactive session, every mode/upscaler/FG family
                                      # included — the half the batch lab can't see (it measures
                                      # FRD's OUT plane; this measures the screen). Ported from
                                      # diamondmine's QA socket (commit 14683e6 there) with ONE
                                      # Windows change: TcpListener for the Unix socket — plus
                                      # THE WINDOWS ACCEPT TRAP found live: an accepted socket
                                      # INHERITS the listener's nonblocking mode (Linux resets
                                      # it), so without set_nonblocking(false) per connection
                                      # the first read races the client's write and drops it as
                                      # WouldBlock — an intermittent 1-in-4 "write failed".
                                      # Protocol: one verb line in, exactly ONE JSON line back
                                      # ({"ok":bool,"lines":[[level,text]..],"ms":f}; ok false
                                      # iff any level-3 line — qa::self_test pins it in
                                      # --check). VERBS (dispatched in session()'s drain, once
                                      # per loop iteration BEFORE the menu hold, so a held menu
                                      # still answers and can be quit): pos (camera/tod/mode/
                                      # wired upscaler/res/fps/frame/fg_mult as JSON) | tp x y z
                                      # [yaw pitch] + look yaw pitch (FlyCam::set — the
                                      # write-through built for exactly this) | tod H
                                      # (fly.set_tod — the MenuFx path; the session's own
                                      # sun_moved detector applies it) | drive x y z ticks /
                                      # drive stop (synthetic ANALOG flight in the flycam
                                      # thread: axes ride the pad path's deflection-scales-
                                      # speed math for N 500Hz ticks; EXEMPT from the focus
                                      # gate — a driver's window is normally unfocused; OS
                                      # keys/mouse/pad are zeroed while unfocused so background
                                      # typing can't leak — never exempt from pause. USAGE:
                                      # deflection scales diag*0.1875 u/s, and diag is
                                      # ground-quad-inflated, so subject-scale passes want
                                      # 0.02-0.05, NOT 1.0 — full deflection overshot a 4-unit-
                                      # distant helmet in 0.26 s) | key <name> (synthesized
                                      # Edges — the settings::MenuFx precedent, so reset
                                      # semantics cannot drift from the keys; refused while the
                                      # menu is open) | screenshot <path.png> (a PENDING verb:
                                      # the three P arms consume the qa_shot path and resolve
                                      # ok/err AFTER the write — the diamondmine typed-result
                                      # lesson; captures whatever the session presents, RR/
                                      # XeSS/FSR/quin included) | sync N (N loop iterations —
                                      # the settle primitive: tp -> sync 120 -> screenshot) |
                                      # quit (clean SessionEnd::Quit). Pendings carry a 30s
                                      # backstop; disconnects are non-events; a session exit
                                      # answers in-flight requests "session ended" via the
                                      # dropped sender. The listener lives in run_window BESIDE
                                      # the FlyCam (connections survive resize re-entries);
                                      # headless paths never construct it, no settings row.
                                      # DRIVE IT WITH `frqa` (src/bin/frqa.rs, the diamondmine
                                      # qactl twin — pure std, builds standalone): `frqa [-p
                                      # port] <verb...>` prints the JSON reply raw, exit 0 on
                                      # ok:true / 1 on ok:false / 2 on connect-usage; raw TCP
                                      # works too. VERIFIED LIVE (2026-08-10, DXR->DLSS-RR
                                      # session): 12/12 rapid-fire, concurrent connections
                                      # interleave, and the agent's own mid-strafe helmet-glint
                                      # screenshot via tp -> tod -> sync -> drive -0.04 ->
                                      # screenshot — the end-to-end loop the lab exists for.
                                      # An FRD live session is `--qa --gpu --xess` (FRD is the
                                      # XeSS/FSR3 sessions' denoiser). MCP: deliberately NOT in
                                      # the game — a thin external wrapper over this socket is
                                      # the recorded follow-on if typed tool schemas ever earn
                                      # their keep; Bash + frqa carries the capability today
```
