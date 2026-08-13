# Tooling, capture, and presentation

SDK paths, Tracy, `--quinlight` registered consensus, `--spin`, `--cinematic` media capture, the settings file, vsync, the HDR output quartet, and GPU timing/PIX markers.

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
cargo run --release -- --dxc-path <d> # DXC DLL directory (default SDKs\dxc\bin\x64; or FRUSTRACER_DXC_PATH)
cargo run --release -- --prefer-intel # pick that vendor's adapter for the D3D12 device (also
                                      # --prefer-nvidia / --prefer-amd; default NVIDIA, or AMD under
                                      # --fsr). A preference, not a requirement: features the picked
                                      # GPU can't support (DLSS/FSR/RT tiers) fall back with a log
                                      # line, per the existing probes. Applies to --check-gpu /
                                      # --check-dxr too
cargo run --release --features tracy  # Tracy CPU profiling (see the Profiling section; off = zero-cost)
cargo run --release -- --quinlight    # REGISTERED CONSENSUS (a port of quinlight-player's
                                      # consensus_registered.comp): suspend the chain's
                                      # first-hit-wins rule, wire EVERY supported level at once
                                      # (DLSS-RR + FSR4-RR + XeSS + FSR 3.1), run them all over the
                                      # SAME traced frame, and present the LK-registered winsorized
                                      # consensus of their outputs. GPU-fed only (--dxr/--gpu).
                                      # --quin-anchor N picks the engine that defines the spatial
                                      # frame (default 0 = the highest wired level). The chain flags
                                      # still compose: --quinlight --no-dlss fuses XeSS + FSR3
cargo run --release -- --spin path    # headless deterministic benchmark/profiling workload: the
                                      # interactive frame contract on a closed-loop Catmull-Rom
                                      # camera (still | path; --spin-frames n, default 2000 — a
                                      # DEFAULTED count is extended so the timed span covers a
                                      # whole SPIN_LAP=600-frame lap past the warm-up, an
                                      # EXPLICIT one is obeyed verbatim; --spin-warmup n
                                      # excludes leading frames, default 20 but 1600 on an
                                      # INTEL adapter, where the driver's async shader
                                      # recompile lands ~600-1500 frames in — see the Arc
                                      # measurement trap in Profiling, and note that every
                                      # Intel --spin number recorded in this file predates the
                                      # warm-up and was taken at WARMUP=20; --spin-hybrid /
                                      # --spin-plain pick the quadtree or the per-pixel
                                      # root-traversal reference arm for the CPU and --gpu
                                      # runners, and print a note under --dxr, which has only
                                      # its one DispatchRays arm; pose is
                                      # a pure function of the frame index — bit-repeatable A/Bs;
                                      # composes with --no-temporal / --no-replay / --no-adopt).
                                      # Drives the GPU arms too: `--gpu` (wavefront) or an EXPLICIT
                                      # `--dxr` runs the same pose loop through HeadlessGpu
                                      # (record -> execute -> block, no swapchain), at the
                                      # `--lock-res` scale — NATIVE unless --lock-res is passed,
                                      # deliberately independent of the interactive default (a
                                      # benchmark's res must not move under recorded numbers; the
                                      # interactive default is native again since 2026-08-08 —
                                      # the two coincide today, but the rule predates and
                                      # outlives that) — with the SAME
                                      # per-frame contract as
                                      # the CPU arm (1-spp upscaler quality, accumulate off,
                                      # frame-uniform Halton) so --cpu/--gpu/--dxr rows compare
                                      # directly. `opts.dxr` defaults ON, so it takes an explicit
                                      # --dxr — a bare --spin still drives the CPU renderer. This is
                                      # the deterministic GPU benchmark the tree lacked: the
                                      # `gpu hybrid` bench row is warm-clock noisy (its own spp
                                      # sweep interleaves and takes medians for that reason) and an
                                      # interactive --gpu-timing table depends on where the camera
                                      # happened to be. It measures the TRACER (no G-buffer pack,
                                      # no feed/upscale — those need a swapchain and are constant
                                      # across tracer changes); pair with --gpu-timing, whose
                                      # per-pass table prints every 120 frames and at exit
cargo run --release -- --cinematic tour --cinematic-res 3840x2160 --cinematic-fps 60 \
                        --cinematic-frames 1200 --cinematic-hdr
                                      # MEDIA MODE (see the Cinematic capture section): headless,
                                      # deterministic stills and camera-spline sequences for the
                                      # README/release. Presets hero|islands|tour|orbit|foliage|hud|list,
                                      # or a JSON shot-list path; bare --cinematic = hero + the
                                      # catalogue. Writes a numbered PNG sequence + manifest and
                                      # PRINTS the exact ffmpeg commands (--cinematic-encode runs
                                      # them). The GPU arms capture through the upscaler chain BY
                                      # DEFAULT (DLSS-RR -> FSR4-RR -> XeSS -> FSR3 at 100%
                                      # render scale — DLAA-grade; the frame written is the
                                      # model's RECONSTRUCTED output; chain flags steer it,
                                      # --no-upscale / GI shots / --cpu / chain exhausted fall
                                      # back to accumulation loudly — see the cinematic section).
                                      # XeSS/FSR3 captures ALSO run the session's pre-upscale
                                      # denoiser since 2026-08-10 (gpu::CineDn — the interactive
                                      # FRD/NRD fold's cinematic twin: pack -> denoise -> out
                                      # replaces the engine feed per sub-frame; measured 3.1x
                                      # less high-frequency noise on a bistro still; --no-frd
                                      # opts out and restores the NEE emissive auto-arm; not
                                      # under --dual-gpu — said loudly).
                                      # Sub-flags: -res WxH (odd dims round down — yuv420p),
                                      # -samples N (sub-frames per OUTPUT frame: reconstruction
                                      # warm/converge passes, or plain accumulation on the
                                      # fallback arms — composes with --spp, a different axis
                                      # that amortizes the
                                      # quadtree), -frames, -fps (drives the cloud clock AND the
                                      # encode), -island, -gi (forces the wavefront: DXR has no
                                      # hemi stage), -overlay, -hud off|hud|menu|settings:<Group>,
                                      # -hdr (16-bit PQ frames + HDR10 HEVC; stills also get a
                                      # linear EXR master + a PQ-tagged AVIF), -out, -encode,
                                      # -exposure EV (STOPS, -8..=8, applied to linear radiance at
                                      # the ONE write site so the SDR PNG / PQ frames / EXR master
                                      # are one exposure by construction; 0.0 returns EXACTLY 1.0
                                      # and the call site branches around the copy, so every
                                      # pre-exposure capture is bit-identical. It exists because the
                                      # tonemap is anchored at a fixed paper white while the
                                      # interesting parts of these scenes are ENCLOSURES whose sun
                                      # is occluded by construction: a physically correct San Miguel
                                      # patio at 15:30 is 2-3 stops under a lit exterior — correct,
                                      # and unpublishable. Brightening the sky or the curve would be
                                      # a lie about the lighting; opening the aperture is what a
                                      # photographer does),
                                      # -dry-run. LOADS THE WORLD by default (it is a media mode,
                                      # not a gate — and it is exclusive with --spin/--check*,
                                      # which keep their own scenes so no must-fire gate moves).
                                      # The only path that can render a moving camera WITH hemi GI,
                                      # because every output frame is a static accumulating pose.
                                      # THREE ARMS, ONE PER BACKEND (B5b, 2026-08-12 — the Vulkan
                                      # one is what finally makes this backend produce a PICTURE;
                                      # sixteen --check-vk stages had scored it and every one ends
                                      # in a number). It needs no window, no swapchain, no surface
                                      # and no new dependency, because everything below
                                      # `cine_write_frame` was already portable and already ran
                                      # here for the CPU arm — which is also the doctrine it
                                      # inherits: every arm hands a LINEAR f32 image to the one
                                      # `resolve_hdr -> ToneParams::SDR -> save_png` path, so the
                                      # arms cannot drift onto different tone curves. MEASURED at
                                      # 480x270x8: Vulkan mean level 138.5 vs the CPU arm's 139.0
                                      # — the shared curve; bit-identity was never available
                                      # (different tracer, different reconstruction, different rng),
                                      # so "the same curve by construction" is the honest claim.
                                      # THE ARM PICK IS PURE DATA (`cinematic::pick_arm`), and it
                                      # exists because the LABEL and the DISPATCH had drifted:
                                      # `opts.dxr` DEFAULTS TO TRUE and the label was derived on
                                      # every platform while the dispatch sat behind
                                      # `#[cfg(windows)]`, so a bare `--cinematic` off Windows
                                      # announced `[dxr]`, fell past the block, and rendered every
                                      # frame on the CPU labelling it "CPU". Three bools in, an
                                      # enum out, ONE call site; gated in `cinematic::self_test`,
                                      # so the table is pinned on a box with no GPU. `--gpu` is the
                                      # right spelling for the Vulkan arm and no `--vk` flag exists
                                      # — the flag names an ARM (the GPU-resident wavefront tracer)
                                      # and Vulkan implements exactly that, so one command line
                                      # means one thing on both OSes. `--dxr` there is a
                                      # SUBSTITUTION with one loud line, not an exit 2: there is
                                      # exactly one GPU arm to pick, and the --fsr4 doctrine is
                                      # about being TOLD something impossible rather than about a
                                      # default. The line prints beside the LABEL so
                                      # `--cinematic-dry-run` explains itself.
                                      # TWO ARMS PER SHOT, mirroring D3D12: RECONSTRUCTION (FSR3 at
                                      # 1:1 — DLAA-shaped, the temporal model as antialiaser and
                                      # integrator — fed the COMPOSED frame when NRD is armed, so
                                      # pack -> ReBLUR -> recompose -> FFX with NO feed dispatch)
                                      # and ACCUMULATION (the fallback, and what a GI shot always
                                      # takes, because the hemisphere integrator is a still-frame
                                      # accumulation contract). The sub-frame contract is copied
                                      # verbatim: free-running `seq` so the Halton phase never
                                      # restarts, `reset` only at seq 0, and the OUTPUT-FRAME-0
                                      # WARM-UP of `JITTER_PHASE - samples` emitting passes —
                                      # without which frame 0 is reconstructed from under half a
                                      # phase AND sampled on a biased lattice, a discontinuity that
                                      # shows once per lap in a looping clip. Self-limiting (a
                                      # 256-sample still is already several phases) and
                                      # deliberately NOT on the accumulation arm. Replay covers
                                      # sub-frames 1..N-1 there: this backend asks the CALLER to
                                      # prove the bit-equality rather than keeping a `last_struct`
                                      # cache, and "one `basis`, computed once per output frame and
                                      # reused" is the narrowest possible proof of it.
                                      # THE TRACER IS CACHED BY RESOLUTION, not rebuilt per shot —
                                      # constructing one compiles the corpus through DXC (~20 s)
                                      # and the `islands` preset is seven shots at one res — and
                                      # `CineVk::destroy` is a struct rather than D3D12's
                                      # positional tuple for a reason only this backend has:
                                      # `VkTracer`/`Fsr3`/`VkNrd` have `destroy(&hg)` and NO `Drop`,
                                      # so a forgotten teardown leaks device memory across a
                                      # seven-shot preset.
                                      # THE TEMPORAL STATE, FIXED (B5c, 2026-08-12). B5b shipped
                                      # FOUR defects, all in the RECONSTRUCTION arm, all invisible
                                      # to every gate — and the sharpest framing is that the
                                      # capture arm's two arms DISAGREED: `accumulate` replayed
                                      # and carried a comment saying why it was a pure win, while
                                      # `output_frame` — the DEFAULT arm, the one that produced
                                      # every shipped picture — did not. (1) It computed `basis`
                                      # once and passed it as BOTH `cam` and `prev_cam`, so every
                                      # moving shot's motion vectors were ZERO; this hit FSR3
                                      # whether or not NRD was armed, i.e. `--no-nrd` too.
                                      # (2) `common_settings(&mats, &mats, ..)` told NRD the
                                      # camera had not moved, with `reset` only at seq 0, so a
                                      # tour accumulated under ACCUM_CONTINUE against an identity
                                      # reprojection for its whole length. (3) `replay: false`
                                      # always, so a 256-sample still re-ran the whole quadtree
                                      # ladder 256 times at one bit-identical pose. (4) `CineVk`
                                      # is CACHED BY RESOLUTION across shots while `seq` only ever
                                      # incremented, so ONE `reset` served an entire run — the
                                      # `islands` preset is seven shots at one resolution, and
                                      # shots 2-7 reconstructed their first frames out of shot 1's
                                      # island. D3D12 declares its `seq` INSIDE the per-shot loop
                                      # and says "reset fires once per shot"; hoisting the Vulkan
                                      # equivalent into the cached struct is what broke it.
                                      # THE FIX IS A DERIVATION, not a patch to two literals:
                                      # `cinematic::Temporal` remembers the previous SUB-frame and
                                      # `replay` becomes the FACT `render_wavefront_replay`'s own
                                      # doc demands — the basis bit-equals the previous producing
                                      # frame's — instead of the index heuristic `k > 0`. Per
                                      # SUB-frame rather than per output frame is what makes it
                                      # right at both seams: inside an output frame the pose is
                                      # identical, so prev == cur AND replay is legal; at a frame
                                      # boundary prev is the frame before and the structure must
                                      # be re-traced. `step`/`advance` are separate calls for
                                      # `nrd_frame_step`'s reason — a failed sub-frame must not
                                      # become the pose the next one reprojects from. `begin_shot`
                                      # drops it per shot. The tracer cache key gained `recon`,
                                      # since pack/feed/NRD are BUILD-time options and a
                                      # size-only key froze the first shot's GI answer for every
                                      # later shot at that size.
                                      # GATED IN `--check` (`cinematic::self_test`, pure, GPU-free,
                                      # every platform): three sub-frames at pose A then two at B,
                                      # asserting the prev sequence, the replay sequence, and that
                                      # after the first sub-frame prev is never THIS one's own
                                      # pose+jitter — plus a purity arm proving a failed frame
                                      # cannot advance. TOOTH FIRED: planting the shipped defect
                                      # reads "temporal step 1: prev jitter equals this
                                      # sub-frame's own — the arm is reprojecting from itself".
                                      # NOTE WHY THE DEVICE GATES COULD NOT HAVE CAUGHT IT: V15
                                      # and V16 each render ONE pose repeatedly, where passing the
                                      # same value twice is CORRECT, so they share the exact call
                                      # shape that was wrong here — and nothing in the type system
                                      # separates `common_settings`' two `&CamMatrices`.
                                      # MEASURED, and one of them is a NEGATIVE worth keeping:
                                      # replay fires on 71 of 72 sub-frames (verified by probe —
                                      # `trace k=0` once per output frame, `replay k=1..71`) and
                                      # buys NOTHING measurable (1080p 32-sample still, interleaved:
                                      # 1.34/1.37 with vs 1.36/1.35 without — the arms overlap).
                                      # A composed capture sub-frame also runs the pack, ~31 ReBLUR
                                      # dispatches, the recompose and FFX, and `VkHeadless::run` is
                                      # submit-and-wait, so the ladder is a far smaller share here
                                      # than in the interactive D3D12 frame where -43% was
                                      # measured. It ships because it is strictly less work and
                                      # provably bit-identical (V17), NOT because it is faster.
                                      # `accumulate` was deliberately NOT converted: its `k > 0`
                                      # already IS the bit-equality its doc claims, it has no MV
                                      # consumer, and it takes `&self` — the two arms disagreed
                                      # because one was WRONG, not because they lacked a shared
                                      # mechanism.
                                      # WORKS ON DAY ONE: stills, sequences, GI, overlay (the
                                      # `info` plane exists, so the quadtree overlay draws real
                                      # subdivision), --cinematic-hdr's PQ/EXR, --cinematic-encode,
                                      # exposure/res/samples/frames/fps/island/out/dry-run, TOD
                                      # attractors, BC7, --spp, --replay, --sw-rays. LOUD DEGRADES,
                                      # each with its reason: DLSS/FSR4-RR/XeSS have no Linux
                                      # artifact (the chain falls through to FSR 3.1); --fsr4 is
                                      # the one that stays FATAL, being a requirement rather than a
                                      # preference; --dual-gpu and --frd are D3D12-only; the HUD is
                                      # (`slint` is cfg(windows)); and FOLIAGE SWAY has no animated
                                      # TLAS on this backend, so leaves render at their REST POSE —
                                      # named explicitly because the `foliage` preset's whole
                                      # subject is the wind, and N identical PNGs with no error is
                                      # worse than a missing feature.
                                      # MEASURED (RADV, procedural, 480x270): the denoised capture
                                      # carries 40% less high-frequency content than `--no-nrd` at
                                      # 2 samples/frame — the composed frame doing real work in the
                                      # media path, not only in V16.
                                      # WHAT IS NOT ASSERTABLE, and the plan says so rather than
                                      # manufacturing a gate: whether the picture is GOOD.
                                      # `--check-vk` writes no files and is a pure function of the
                                      # command line; this mode's product IS a file. What gates is
                                      # the arm-pick table (in --check, GPU-free) and V16's
                                      # composed frame; the rest is a human look, and this arm is
                                      # what finally makes that look possible on Linux
cargo run --release -- --no-settings  # ignore frustracer-settings.json for this run (the pause
                                      # menu's saved settings — loaded as DEFAULTS the CLI flags
                                      # override; auto-saved on every menu edit; headless
                                      # --check*/--spin runs always ignore it. ESC opens the menu,
                                      # F1 toggles the HUD — see the HUD/pause-menu section)
cargo run --release -- --no-vsync     # uncapped presentation (Present sync interval 0 on a tearing
                                      # swapchain when DXGI supports it) so interactive frame times
                                      # measure the renderer, not the monitor refresh; composes with
                                      # every mode/upscaler
                                      # (the 10-bit swapchain is ON BY DEFAULT — see the HDR section)
cargo run --release -- --no-hdr       # A/B lever: force the legacy 8-bit B8G8R8A8 swapchain (also
                                      # the FG wrap-failure fallback; the swapchain is otherwise
                                      # ALWAYS 10-bit R10G10B10A2 — the scRGB f16 chain is deleted)
cargo run --release -- --hdr10        # force the PQ declaration (R10G10B10A2 +
                                      # G2084, tone::ToneMode::Pq — 709->2020 matrix + ST 2084 at
                                      # the end of the one curve) in ANY session — which only ADDS
                                      # the HDR-off case, because PQ is the DEFAULT on an HDR-ON
                                      # display; on an HDR-OFF display the same buffer defaults to
                                      # its gamma-2.2 reading instead (Sdr10 — DXGI's default
                                      # interpretation of an undeclared UNORM chain; see the HDR
                                      # section for the bytes-per-present measurement).
                                      # Override-wins like --hdr-peak, including over an "HDR off"
                                      # probe verdict; a REFUSED G2084 declare relabels the session
                                      # Sdr10 on the same swapchain (no rebuild — the buffer's
                                      # default reading IS Sdr10). The swapchain flags are a
                                      # THREE-way (8-bit SDR | Sdr10 | HDR10 — one 10-bit format,
                                      # two curves) spelled as toggles, later flags win across the
                                      # pairs: `--no-hdr --hdr10` = PQ, `--hdr10 --no-hdr` = 8-bit,
                                      # and `--no-hdr10` = Sdr10 — "10-bit but NOT PQ", needed
                                      # because PQ is the HDR-display default so the gamma arm
                                      # isn't reachable as "neither flag" there (cli::self_test
                                      # pins the three-way and the fact that each arm still wins
                                      # from any predecessor). `--hdr` returns to the
                                      # display-probed default. Exposed in the settings menu as the
                                      # Display page's hdr10 row (restart-tier; its OFF state
                                      # means Sdr10, mirroring --no-hdr10; files written before the
                                      # scRGB retirement stored false meaning scRGB — it now reads
                                      # as Sdr10, deliberately unmigrated).
                                      # The wrapper-FG families need no format special-case any
                                      # more (XeSS-FG rejected scRGB fp16 but takes 10-bit —
                                      # VERIFIED GENERATING at HDR10 on the B70; the Sdr10 wrap is
                                      # the same desc and falls to 8-bit + rewrap if refused).
                                      # See the HDR section
cargo run --release -- --hdr-paper-white 120  # where linear 1.0 lands, in nits (default 200);
                                              # LOWER = more highlight headroom above white
cargo run --release -- --hdr-peak 1000        # override the display's reported peak (A/B lever).
                                              # WINS over the probe, including over an "HDR off"
                                              # verdict — the probe can be wrong, and an override
                                              # that no-op'd in exactly that case would be no
                                              # escape hatch at all
cargo run --release -- --check-gpu --gpu-timing  # the same timestamps over the DETERMINISTIC workload: a
                                            # per-pass table under every bench row (drained per row, so the
                                            # reference kernel's frames can't dilute the wavefront's mean),
                                            # plus a spp=1-vs-spp=16 pair — which is what separates a pass's
                                            # FIXED cost from its per-sample MARGINAL cost. The per-pass
                                            # AMD-vs-NVIDIA diff this makes possible is what found the
                                            # LEAF_GROUP wave64 bug
cargo run --release -- --gpu --pix-markers  # PIX events on the D3D12 lists (needs WinPixEventRuntime.dll,
                                            # --pix-path / FRUSTRACER_PIX_PATH, default SDKs\pix\bin\x64)
cargo run --release -- --gpu --gpu-timing   # D3D12 timestamp queries around the SAME marker brackets,
                                            # printed as a per-region GPU-ms table every 120 frames. No
                                            # DLL, every vendor — and the ONLY per-pass GPU numbers
                                            # available on Intel, whose captures PIX cannot analyze at
                                            # all (see Profiling)
```
