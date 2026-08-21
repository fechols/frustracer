# The Metal backend

`--check-fsr3` (FidelityFX FSR3 over a hand-written Metal `FfxInterface`) and `--check-metalfx` (MetalFX temporal upscaling, denoising, and frame interpolation), plus `--check-mtl` (the backend binding and dispatching the corpus's own kernels -- C2).

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
cargo run --release -- --check-fsr3   # FSR3 UPSCALING ON METAL, gated (macOS; src/mtl/ +
                                      # shim/ffx_fsr3_metal.mm + build.rs's generate_fsr3_metallibs
                                      # — the Metal port's B2, 2026-08-12). Headless and CPU-FED:
                                      # the CPU tracer makes the G-buffer (dlss::GBufs, which runs
                                      # on every platform), FSR 3.1 upscales it, and the result is
                                      # SCORED. There is no Metal tracer, no presentation stage and
                                      # no UpChain integration — an interactive macOS session is
                                      # explicitly not what this gates. Wrong OS = exit 2 (an
                                      # explicitly requested gate on a platform that cannot host it
                                      # is an ENVIRONMENT error, the --check-vk-on-Windows
                                      # convention); a missing toolchain on the RIGHT OS SKIPs.
                                      # THE ASYMMETRY WITH --check-vk's V11 IS THE WHOLE COST, and
                                      # it is why there is one gate per BACKEND rather than one
                                      # flag with two arms: FidelityFX ships `ffx_vk` and
                                      # `ffx_dx12` and NOTHING ELSE, so the Vulkan arm is a thin C
                                      # ABI over a STOCK backend on a device that suite already
                                      # opens, while this one must ALSO CARRY THE BACKEND — a
                                      # complete `FfxInterface` (23 callbacks) against Metal,
                                      # ~1300 lines of non-ARC Objective-C++ this tree owns with no
                                      # upstream. That file is the single largest maintenance
                                      # surface the Metal arm has and the one piece with no
                                      # fallback.
                                      # THE SHADER ROUTE, measured before any of it was written
                                      # (M1, Apple M1 / macOS 26.5 / Xcode 26.6): all 160 committed
                                      # SPIR-V blobs transpile with `spirv-cross --msl --msl-version
                                      # 30000 --msl-decoration-binding`, and then **112 of 160
                                      # FAILED `xcrun metal -c` with ONE error** — `'sampler'
                                      # attribute parameter is out of bounds: must be between 0 and
                                      # 15` — because the corpus's only sampler is FFX's
                                      # `s_LinearClamp` at VK binding 1001. After the `binding -
                                      # 1000` remap: 160/160 compile and package. The shader half
                                      # was solved by that one substitution; everything else here
                                      # is a port with a proven reference.
                                      # EXACTLY 80 OF THE 200 COMMITTED FILES ARE TRANSPILED, and
                                      # a plain subtraction gets it wrong because the two skipped
                                      # sets OVERLAP: 40 are permutation INDEX headers (consumed by
                                      # C++, not shader blobs) of which 20 are themselves wave64,
                                      # leaving 160 blobs of which 80 are wave64. 200 - 40 - 80 =
                                      # 80. Both fp32 and fp16 are kept — a pass picks at runtime.
                                      # The per-PASS breakdown is a MEASURED table, not a product
                                      # (accumulate alone is 24 of the 40 per precision): it lives
                                      # at `mtl::fsr3::EXPECTED_PERMUTATIONS`, and if that number
                                      # moves, RE-COUNT rather than multiply.
                                      # FOUR CONTRACTS BIND build.rs AND THE SHIM, none of which
                                      # either side can change alone — there is no other handshake
                                      # between the transpiled table and the loader:
                                      # (1) THE KEY IS FNV-1a-64 OF THE SPIR-V BYTES. build.rs
                                      # hashes the blob it transpiles; `CreatePipelineMetal` hashes
                                      # `FfxShaderBlob.data` — the same bytes, from the same
                                      # `ffxGetPermutationBlobByIndex` accessor the Vulkan backend
                                      # uses — and linear-scans for the match. SPIR-V is the
                                      # interchange format and no Metal artifact is committed.
                                      # (2) A 12-BYTE LE THREADGROUP HEADER precedes each metallib.
                                      # Metal needs the workgroup size HOST-side at dispatch, where
                                      # Vulkan and DXIL both reflect it out of the bytecode, so
                                      # build.rs parses `OpExecutionMode LocalSize` and prepends
                                      # (x,y,z); the shim strips the same 12 bytes back off.
                                      # (3) THE SAMPLER REMAP IS `binding - 1000` ON BOTH SIDES —
                                      # build.rs::remap_ffx_samplers rewrites `[[sampler(N)]]` for
                                      # N >= 1000, and CreatePipelineMetal binds static sampler j
                                      # at index j on the same assumption. Disagree and every
                                      # sampler lands on the wrong slot.
                                      # (4) `--msl-decoration-binding` MAKES THE METAL ARGUMENT
                                      # INDEX EQUAL THE FFX/VK BINDING NUMBER, which is what lets
                                      # the backend bind discretely (setTexture/setBuffer/setBytes
                                      # at slotIndex) instead of building descriptor sets. Drop the
                                      # flag and every binding silently renumbers.
                                      # A FIFTH IS A PAIR RATHER THAN A CONTRACT: the caps hardcode
                                      # in `GetDeviceCapabilitiesMetal` (SM 6.6, waveLaneCountMin =
                                      # Max = 32, fp16Supported = true) and build.rs's WAVE64 SKIP.
                                      # Apple GPUs are SIMD-32, so reporting a 32-lane wave is what
                                      # makes FFX request exactly the set that was transpiled;
                                      # widen the caps and FFX asks for a hash that was never
                                      # emitted (and the wave64 blobs would mis-execute at width 32
                                      # anyway), narrow the transpile and the same happens from the
                                      # other side.
                                      # THE FOUR VALIDATION-ONLY BUGS, carried verbatim from the
                                      # reference rather than re-derived, and marked GOTCHA 1-4 in
                                      # the .mm. EVERY ONE was found only under `MTL_DEBUG_LAYER=1
                                      # MTL_SHADER_VALIDATION=1`, and each MASKS THE NEXT (the
                                      # layer aborts on the first error per command buffer), so a
                                      # green run without them proves considerably less than it
                                      # looks: (1) constant buffers must be copied into a 16-BYTE-
                                      # PADDED temp — spirv-cross rounds an MSL cbuffer struct up
                                      # (cbFSR3Upscaler is 148 B, the MSL struct 160), and FFX's
                                      # `data` pointer backs only the 148, so binding the larger
                                      # length over-reads it (UB) while binding the smaller is
                                      # REJECTED as too small; (2) R32_UINT textures must be
                                      # BUFFER-BACKED — spirv-cross emulates image atomics with an
                                      # aliased MTLBuffer bound at the SAME slot number in buffer
                                      # space, addressed `alignedWidth*y + x`, so function constant
                                      # 65535 (`spvLinearTextureAlignmentOverride`) must be set
                                      # from `minimumLinearTextureAlignmentForPixelFormat:` on
                                      # EVERY pipeline or the atomic argument is left unbound;
                                      # (3) `FFX_GPU_JOB_CLEAR_FLOAT` is ILLEGAL on a buffer-backed
                                      # texture (Metal forbids RenderTarget usage there), so those
                                      # clear by blit-filling the backing buffer — exact, since FFX
                                      # only ever zero-clears them and every byte of a 0.0f fill is
                                      # zero (a nonzero request is now LOUD rather than silently
                                      # serviced as zero); (4) shim-owned temporals must be created
                                      # WITH RenderTarget usage, or the render-pass load-clear path
                                      # aborts. TWO MORE that are not optional:
                                      # `newFunctionWithName:constantValues:` ALWAYS (some
                                      # permutations declare the optional spirv-cross function
                                      # constant and Metal then REFUSES the plain variant), and the
                                      # wide-char binding `name` is REQUIRED — FSR3's
                                      # `patchResourceBindings()` resolves each slot by `wcscmp`
                                      # and returns FFX_ERROR_INVALID_ARGUMENT without it, which
                                      # nothing in the public headers says.
                                      # THE STAGES. U0 the input trio's staging math (fsr::
                                      # stage_self_test — the same pure functions --check-fsr runs
                                      # on every platform and the D3D12 arm's record_upload uses,
                                      # repeated here so the gate is self-contained); U1 the
                                      # metallib table (count vs build.rs's ENUMERATED denominator,
                                      # strictly-ascending unique hashes, the 12-byte header
                                      # parsed back, `MTLB` magic — pure, no device, no SDK); U2 a
                                      # real device plus a texture round-trip through the SHIPPING
                                      # staging encoder, which is what keeps "the upscaler produced
                                      # nothing" and "our texture plumbing never carried the bytes"
                                      # separable; U3 the FSR3 CONTEXT, which is the stage carrying
                                      # the whole risk — creation drives fpCreatePipeline across
                                      # every pass, so a mis-keyed hash, a spirv-cross output Metal
                                      # will not accept, or a caps/wave64 divergence fails THERE
                                      # rather than at whichever later dispatch needed the missing
                                      # permutation; U4 a real upscale of two real rendered frames,
                                      # scored.
                                      # THE COVERAGE LIMIT IS MEASURED AND IS NOT WHAT U1 PROVES.
                                      # One context creation builds ELEVEN pipelines of the 80
                                      # metallibs offered — one per FSR3 pass at the option word
                                      # its flags select — and a sweep of all EIGHT context-flag
                                      # combinations reaches only FOURTEEN distinct blobs, because
                                      # most passes ignore most option bits and the 40 fp32
                                      # variants are never requested at all (the caps report fp16).
                                      # So U1 proves all 80 are well-formed metallibs and U3 proves
                                      # eleven of them become pipeline states; NOTHING here proves
                                      # the other 69 ever run. The count is returned through
                                      # `out_pipelines` and floored at 10 rather than left implicit
                                      # because "context created OK" and "the metallibs work" are
                                      # different claims that no return code separates — a backend
                                      # answering every fpCreatePipeline with an empty success
                                      # would create a context and build nothing.
                                      # U4 IS WRITTEN AROUND THE FACT THAT AN UPSCALER IS EASY TO
                                      # GATE VACUOUSLY: a pass-through copy, a bilinear stretch and
                                      # a working temporal reconstruction all produce a plausible,
                                      # finite, correctly-sized image, so every assertion is paired
                                      # with something it must be DIFFERENT from. Two real frames
                                      # at 321x181 -> 642x362 with REAL Halton jitter (unlike
                                      # `mv_check_at`/`fsr_frame_check`, which pin it to (0,0)
                                      # because they reconstruct world positions — here jitter is a
                                      # genuine FSR3 INPUT and pinning it would leave the one input
                                      # this renderer feeds and the reference never did untested)
                                      # and the same 0.02*diag dolly `fsr_frame_check` uses, so the
                                      # motion vectors are the ones that gate already proves.
                                      # Asserts: finite and non-negative; energy within [0.5, 2]x
                                      # of the input (an upscaler RESAMPLES, it does not expose —
                                      # a colour-space or exposure mistake moves this by orders of
                                      # magnitude); a warmed history must DIFFER from a reset one
                                      # (else FFX is not accumulating and every temporal claim is
                                      # false); and THE ASSERTION THAT CARRIES THE STAGE — the
                                      # output must differ measurably from a bilinear stretch of
                                      # its own input, or nothing proves a reconstruction ran.
                                      # MEASURED energy 1.000x, history 2.197e-2, vs-bilinear
                                      # 2.962e-2 — and that last BOUND is deliberately SIX TIMES
                                      # under its measurement (5e-3) rather than snug: the failure
                                      # it rejects lands at ~0, so a low bound loses no
                                      # discrimination while a snug one fails on any scene or pose
                                      # that reconstructs less. Plus three PLUMBING probes, each
                                      # re-running the sequence with exactly ONE input of frame B
                                      # changed so a null result attributes to that input alone:
                                      # jitter 2.592e-2, depth 1.218e-2, motion 3.503e-2.
                                      # THE PROBES MUST BASELINE ON THE ACCUMULATING FRAME, and the
                                      # first draft used the RESET one — which reported depth at
                                      # EXACTLY 0.00e0 and read like a plumbing bug in the backend.
                                      # It is not: a reset frame has no history, so the
                                      # depth-derived disocclusion mask has nothing to reject and
                                      # depth genuinely cannot change the output. A probe that
                                      # cannot move its target fails whatever the code does — the
                                      # mirror of one that cannot fail — and depth and motion
                                      # vectors are BOTH in that class.
                                      # DETERMINISM SPLITS IN TWO, AND THE OBVIOUS SINGLE BITWISE
                                      # CLAIM IS WRONG in a way that took the measurement to see.
                                      # Two fresh contexts, identical inputs: a RESET-ONLY frame is
                                      # EXACTLY bit-identical every time (4/4, and with the output
                                      # plane deliberately left uncleared, which also proves FSR3
                                      # writes every output texel), while an ACCUMULATING frame
                                      # differs in 100-2800 of 929616 channels, each by EXACTLY ONE
                                      # f16 ULP, the count varying run to run. THAT IS NOT A DEFECT
                                      # AND NOT GPU NON-DETERMINISM: FFX declares essentially every
                                      # internal resource — the three shared temporals included —
                                      # as FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED, and
                                      # ffx_vk.cpp skips its init copy for exactly that type, so
                                      # texels a reset frame never wrote hold per-allocation
                                      # residue BY CONTRACT; a following frame reads them at an
                                      # accumulation weight near zero, which is why the effect can
                                      # only ever tip a rounding boundary. ZEROING THE TEMPORALS
                                      # ANYWAY WAS TRIED AND MEASURED WORSE (median 388 -> 1050
                                      # differing channels, 8 samples each) — no benefit, inside
                                      # the same wide band, for three blocking blits per context
                                      # creation — so the residue is not the whole story either; it
                                      # is simply not ours to control. CORROBORATED FROM THE OTHER
                                      # SIDE, which is what settles it: under
                                      # MTL_SHADER_VALIDATION=1 the accumulating count drops to
                                      # EXACTLY ZERO, because that layer zero-fills allocations —
                                      # execution-order non-determinism would be untouched by it,
                                      # per-allocation residue is removed by it entirely. So the
                                      # VALIDATED run is also the STRICTEST run, a reason to prefer
                                      # it beyond the four gotchas. The gate therefore makes the
                                      # exact claim where it holds and bounds the other (<= 1%
                                      # relative, <= 2% of channels), and both halves have teeth:
                                      # garbage propagating at full magnitude fails the ULP bound,
                                      # a genuine backend race (the image-atomic aliasing class)
                                      # fails the exact one.
                                      # THE BUILD IS CACHED AND THE CACHE IS KEYED ON THE RECIPE AS
                                      # WELL AS THE INPUT. A metallib's file name is FNV-1a of its
                                      # SPIR-V, so an existing one is current with respect to its
                                      # INPUT by construction — but not to spirv-cross's version
                                      # (whose MSL emission a Homebrew bump changes while every key
                                      # stays identical) nor to our own recipe, so both ride a
                                      # `.toolstamp` and a change wipes the directory. SPIRV_CROSS_
                                      # ARGS is on it verbatim; `RECIPE` is the hand-bumped half
                                      # covering the two rules that alter the bytes without
                                      # altering a key (the 12-byte header format, the -1000
                                      # sampler remap). Without the cache every build re-runs 240
                                      # subprocesses (~55 s). `SPIRV_CROSS` overrides the binary.
                                      # ONE CFG PER ARTIFACT: `ffx_fsr3_metal` means the metallib
                                      # table AND the backend built, decided off ONE boolean in
                                      # build.rs, because a `ffx_metal` backend with no table can
                                      # do exactly one thing — fail at the first fpCreatePipeline —
                                      # so compiling it without one would ship a linked, reachable,
                                      # guaranteed-to-fail arm (the distinction the `ffx_fsr3_vk`
                                      # repair was about). Gated on the TARGET, not the host.
                                      # Missing SDK source, missing committed SPIR-V, or absent
                                      # spirv-cross/Xcode each degrade to a warn-and-skip with the
                                      # build printing which half was absent, and U1 then SKIPs by
                                      # name — "it did not build" is the expected state on a fresh
                                      # clone and must not read as a defect.
                                      # Deps: objc2 / objc2-metal / objc2-foundation (macOS only);
                                      # Metal and Foundation are LINKED (unlike `src/vk/`'s dlopen
                                      # policy — the backend calls Metal directly, and there is no
                                      # dlopen equivalent), scoped to want_metal so a bare checkout
                                      # still links nothing.
                                      # Touch shim/ffx_fsr3_metal.{mm,h} / src/mtl/ / build.rs's
                                      # generate_fsr3_metallibs or transpile_ffx_metallib -> run
                                      # --check-fsr3, then AGAIN under `MTL_DEBUG_LAYER=1
                                      # MTL_SHADER_VALIDATION=1` (the four gotchas are invisible
                                      # otherwise, and the gate says so on an unvalidated run),
                                      # then --check-fsr (the shared staging math), --check,
                                      # cargo test, and FR_FSR3_DUMP=1 for the half no magnitude
                                      # can score — a mirrored jitter or MV sign reads as doubled
                                      # wobble or a directional smear, and both move the numbers
                                      # by about as much as being right does
cargo run --release -- --check-metalfx# MetalFX TEMPORAL UPSCALING, gated (macOS; src/mtl/mfx.rs +
                                      # src/mtl/planes.rs — the Metal port's B3, 2026-08-12). Apple's
                                      # `MTLFXTemporalScaler` over the SAME CPU-rendered G-buffer
                                      # --check-fsr3 feeds FidelityFX, headless and scored. No Metal
                                      # tracer, no presentation stage — the B2 scope, unchanged.
                                      # A SEPARATE GATE rather than a stage of --check-fsr3, on the
                                      # one-gate-per-SDK convention (--check-oidn/-xess/-nrd): the skip
                                      # stories are unrelated (device support and a macOS floor here;
                                      # the FidelityFX SDK source plus a build-time transpile there),
                                      # and a flag named after FidelityFX that also ran Apple's
                                      # upscaler would mislead every later reader. Note --check-vk's
                                      # V11 is the OTHER precedent and points the same way: one gate
                                      # per backend there, one per SDK here, and in both cases the
                                      # thing that decides is what a reader would expect the flag to
                                      # cover.
                                      # IT COSTS ALMOST NOTHING NEXT TO THE FSR3 ARM, and the asymmetry
                                      # is structural. FidelityFX ships ffx_vk and ffx_dx12 and nothing
                                      # else, so Metal needed a hand-written FfxInterface (~1350 lines
                                      # of non-ARC ObjC++ this tree OWNS, with no upstream) plus a
                                      # build-time SPIR-V -> MSL transpile of 80 permutations. MetalFX
                                      # is a system framework: no shim, no metallib table, no SDK
                                      # fetch. So this arm WORKS ON A BARE CLONE and carries no cfg
                                      # beyond macOS, where mtl::fsr3 is gated on ffx_fsr3_metal.
                                      # THE POINT IS NOT A SECOND UPSCALER, IT IS AN INSTRUMENT. B2's
                                      # own risk record says real depth and real jitter were the
                                      # untested path (the reference it was ported from upscaled VIDEO:
                                      # flat depth, zero jitter, hardcoded camera) and that a single
                                      # arm's teeth "catch gross wiring, not polarity — that needs the
                                      # dump images". Two independent consumers of byte-identical
                                      # inputs can do what one cannot: correlate.
                                      # THE CONVENTIONS, and which are DERIVED. (a) MOTION VECTORS,
                                      # derived: Apple documents motionVectorScaleX at 1.0 as "the
                                      # motion vectors for an object that moves down and to the right
                                      # in the colorTexture by 10 pixels would be (-10,-10)"; GBufs::
                                      # mvec stores prev_px - cur_px y-down, and such an object has
                                      # prev - cur = (-10,-10). Exact match, so the scale is the bare
                                      # fsr::UPSCALE_MV_SIGN — the same value both FSR3 arms pass.
                                      # (b) DEPTH, derived: fsr::stage_depth writes reversed-Z with sky
                                      # at exactly 0.0, so setDepthReversed(true), the twin of FSR3's
                                      # FLAG_DEPTH_INVERTED. (c) JITTER, NOT documented by Apple — and
                                      # MEASURED here rather than left a coin flip, which is the
                                      # headline result. mtl::mfx::JITTER_SIGN started seeded at FFX's
                                      # value on the reasoning that both take "the subpixel jitter
                                      # offset applied to the camera" in pixels; X3's cross-check then
                                      # SETTLED it: mirroring this arm alone drops the FSR3 correlation
                                      # 0.655 -> 0.479 and fails the gate, so +1 is the measurement.
                                      # FR_MFX_JITTER=raw|neg is the lever (the FR_VK_FSR3_JITTER
                                      # shape).
                                      # THE ONE HARD API CONSTRAINT, and it reshaped the harness:
                                      # MTLFXTemporalScaler.h:222 says of outputTexture "You are
                                      # responsible for providing a texture with a private storageMode"
                                      # — the one texture in this tree the CPU may not touch. So
                                      # Mtl::texture GAINED a storage-mode parameter (it hardcoded
                                      # Shared), and Mtl gained blit-based read_private/clear_private,
                                      # because getBytes and replaceRegion are both illegal there.
                                      # KEEPING THE CLEAR mattered more than it looks: it is what
                                      # separates "the scaler wrote nothing" from "the scaler wrote a
                                      # dark image" in the readback, and dropping it on the grounds
                                      # that replaceRegion no longer applies would have made X3's
                                      # non-zero assertion silently untestable — the highest-value
                                      # vacuity available in this gate, arriving as a side effect of a
                                      # mechanical fix.
                                      # TWO SETTINGS ARE DETERMINISM PRECONDITIONS, not preferences.
                                      # requiresSynchronousInitialization(true): Apple's default is
                                      # FALSE, which returns a scaler immediately and "compile[s] a
                                      # faster upscaler in the BACKGROUND" — and the gate builds
                                      # several scalers and compares pairs, so a compile landing
                                      # between two runs would be comparing two implementations and
                                      # calling the difference residue. (Apple says quality is
                                      # "consistent" across the two, which is a quality claim, not a
                                      # bit-identity one, and this gate asserts bit-identity.) Cost is
                                      # real and reported per run: X2's FIRST scaler measures 266-627 ms
                                      # (cold — it carries one-time framework init), after which X3's
                                      # four build in ~720 ms total, i.e. ~180 ms each.
                                      # And autoExposureEnabled(FALSE) plus a 1x1 R16Float
                                      # exposure texture holding 1.0 — DELIBERATELY UNLIKE the Metal
                                      # FSR3 arm, which sets FLAG_AUTO_EXPOSURE: auto-exposure has
                                      # MetalFX compute a per-frame gain that multiplies the input
                                      # colour with nothing documenting it being un-applied on output,
                                      # which would make X3's energy assertion a measurement of Apple's
                                      # heuristic rather than of our wiring — and the temptation on
                                      # failure would be to widen the one bound that catches
                                      # colour-space mistakes.
                                      # DETERMINISM IS STRICTER HERE THAN FOR FSR3, and that is a
                                      # measurement, not an assumption inherited from the sibling. The
                                      # claim was written STRICT (both regimes exactly reproducible)
                                      # precisely because MetalFX's history is internal and opaque, so
                                      # no cause was available in advance to justify FSR3's two-regime
                                      # split — and it HELD: reset-only 0 channels AND accumulating 0
                                      # channels, across fresh scalers, unchanged under
                                      # MTL_SHADER_VALIDATION=1. FSR3 needs its split because
                                      # FidelityFX declares internal resources UNINITIALIZED; nothing
                                      # of the kind surfaces here. The comparison is an INTEGER ULP
                                      # distance over the f16 bit patterns (read_output_bits), not a
                                      # relative bound on widened floats: an ULP distance cannot be
                                      # answered by moving a threshold, and --check-fsr3's own comment
                                      # already records "EXACTLY ONE f16 ULP" while its code asserts a
                                      # tunable 1%.
                                      # THE CROSS-CHECK (X3, compiled only where FSR3 is also built).
                                      # Both arms render from mtl::planes::Trio — ONE allocation and
                                      # ONE staging site, which is what makes "identical inputs"
                                      # structural rather than two call sites agreeing — and the gate
                                      # correlates their deviations from a common bilinear reference.
                                      # THE OBVIOUS FORMULATION IS UNSOUND AND WAS MEASURED TO BE, so
                                      # do not re-derive it: the tempting d(mfx,fsr3) < d(mfx,bilinear)
                                      # ("two reconstructions must resemble each other more than either
                                      # resembles a resampler") is implied by nothing. Bilinear is the
                                      # SMOOTH image; both arms deviate by ADDING detail, and where
                                      # those deviations are not identical they add rather than cancel,
                                      # so the mutual distance can legitimately EXCEED each arm's
                                      # distance from the smooth reference — approaching sqrt(2) times
                                      # it for independent deviations. Measured: mutual 2.999e-2
                                      # against 2.752e-2, ratio 1.09, i.e. strongly correlated AND the
                                      # inequality false. The correlation is what that 1.09 actually
                                      # says.
                                      # MEASURED (M1, macOS 26.5, procedural scene, 321x181 -> 642x362):
                                      # energy 0.988x, history 1.140e-2, vs-bilinear 2.752e-2, FSR3
                                      # correlation 0.655, probes jitter 2.882e-2 / depth 3.938e-3 /
                                      # motion 1.671e-2, scale range 1.00x..3.00x, required usage
                                      # color/depth/motion 0x1 and output 0x7 (all inside the harness's
                                      # 0x7). MFX_BILINEAR_MIN is 4.5e-3 — this arm's OWN measurement
                                      # divided by six, never FSR3's 5e-3 inherited, since that number
                                      # describes FidelityFX's reconstruction on this scene and not
                                      # Apple's.
                                      # MFX_FSR3_CORR_MIN IS NOT MIDWAY, and the asymmetry is chosen:
                                      # 0.5 sits 0.024 above the MIRRORED reading and 0.155 below the
                                      # correct one, buying margin against false failures at the cost
                                      # of margin against a blunt tooth — the right direction for a
                                      # gate, since a flaky red costs every future run while a tooth
                                      # that stops biting costs only the case it was aimed at. The
                                      # consequence to know: a scene or driver that moves the MIRRORED
                                      # reading up 5% silently stops this discriminating, and the
                                      # symptom is a GREEN run, not a noisy one — re-measure both arms
                                      # (FR_MFX_JITTER=neg) before trusting it on new content.
                                      # WHAT IT STILL CANNOT SCORE, said rather than implied: the depth
                                      # POLARITY. Flattening the depth plane moves the output whichever
                                      # way setDepthReversed points, so that probe fires either way —
                                      # the same class the jitter sign was in before the cross-check,
                                      # and the reason FR_MFX_DUMP=1 exists.
                                      # KNOWN-ACCEPTS. The macOS floor rises to 13.0 for the WHOLE
                                      # binary: objc2-metal-fx carries an unconditional #[link(name =
                                      # "MetalFX", kind = "framework")], so MetalFX.framework is an
                                      # LC_LOAD_DYLIB of every macOS build including bare checkouts
                                      # that never run this gate, and on an older macOS the process
                                      # does not launch AT ALL (not a graceful skip). Accepted — and
                                      # note the related claim this repaired: build.rs said scoping the
                                      # Metal/Foundation links to want_metal meant "a bare checkout
                                      # still links nothing", which was FALSE (objc2-metal and objc2
                                      # carry their own #[link] attributes; both frameworks were always
                                      # in the load commands). Fixed-size allocation, not the FSR3
                                      # arm's allocate-at-max-and-sub-rect: MetalFX validates textures
                                      # against its descriptor's dimensions and its dynamic-resolution
                                      # equivalent is descriptor-level (inputContentPropertiesEnabled
                                      # plus a device scale range), i.e. a different mechanism, and an
                                      # untested dynamic-res path inside a gate that exists to prove
                                      # the static one is scope creep. Enabling MTLFXTemporalScaler
                                      # additively pulls objc2-metal/MTL4Compiler and MTLFence into the
                                      # graph, so Cargo.toml's deliberately-short one-feature-per-header
                                      # list is no longer the whole truth (said there).
                                      # Touch src/mtl/mfx.rs / planes.rs / device.rs's texture storage
                                      # or blit helpers / run_check_metalfx -> run --check-metalfx,
                                      # then AGAIN under MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 (the
                                      # storage-mode class is invisible otherwise and the gate says so
                                      # on an unvalidated run), then --check-fsr3 (planes::Trio is
                                      # shared, so its U4 numbers are the before/after fingerprint —
                                      # energy/history/vs-bilinear must not move), --check-fsr, --check,
                                      # and cargo test. FR_MFX_JITTER=neg must FAIL the cross-check;
                                      # if it stops failing, the tooth has gone blunt.
                                      # X4-X6 EXTEND THE SAME GATE (B4, 2026-08-12) — one SDK,
                                      # one flag, three more subjects, each SKIPping on its own
                                      # terms because MTLFXTemporalDenoisedScaler and
                                      # MTLFXFrameInterpolator are API_AVAILABLE(macos(26.0))
                                      # where the plain scaler is 13.0. THE FLOOR DOES NOT RISE:
                                      # rustc's deployment target is 11.0, so those classes
                                      # weak-import and the binary still launches on older macOS
                                      # — but objc2's extern_class! PANICS on a missing class, so
                                      # every typed entry point goes through an
                                      # AnyClass::get(c"...") probe first (mtl::mfxdn::available).
                                      # A cargo feature or a build.rs cfg was rejected: build.rs
                                      # keys on the BUILD host, the cfg(windows)-describes-the-HOST
                                      # defect class this file already records.
                                      # X4 = the denoised scaler creates, its nine required
                                      # usages are a subset of what Mtl::texture hands out, and it
                                      # destroys. X5 = a real denoise, scored. X6 = frame
                                      # interpolation, REPORTED (below).
                                      # THE DENOISER IS THE FIRST THING ON THIS PLATFORM THAT CAN
                                      # BE SCORED FOR QUALITY, and that is structural rather than
                                      # lucky: an upscaler has no directional claim (--check-fsr3
                                      # says plainly that its own quality comparison is
                                      # report-only, because mean-|d| against a converged
                                      # reference rewards blur and INVERTS between scenes), while
                                      # a denoiser's claim is "noise must go DOWN". THE CONTROL IS
                                      # THE PLAIN SCALER, not a resampler — both arms consume
                                      # byte-identical planes at identical extents through
                                      # planes::Trio, both upscale 2x, and only one denoises, so
                                      # the mean-|Laplacian| comparison has no scene-dependent
                                      # bias to argue about and doubles as the anti-vacuity (two
                                      # arms that agreed would mean the denoise did nothing). A
                                      # bilinear reference would have confounded the denoise with
                                      # the upscale, since a 2x upscale reduces per-pixel
                                      # Laplacian by itself. MEASURED 33.8% less high-frequency
                                      # content; MFXDN_LAPLACIAN_DROP_MIN sits six times under
                                      # that, because the failure it rejects (guides unbound, no
                                      # denoise at all) lands at or BELOW zero.
                                      # THAT METRIC SETTLED THE ONE CONVENTION APPLE DOES NOT
                                      # DOCUMENT. normalTexture's doc is "The normal texture this
                                      # scaler evaluates" and nothing else — no space, no
                                      # encoding, no range. World space was the ARGUMENT (our
                                      # plane, DLSS-RR's own input, and a denoiser wanting
                                      # view-space normals would not need a worldToViewMatrix to
                                      # be told how to get there), and B3's jitter sign is the
                                      # standing warning that a good argument about an
                                      # undocumented convention is still a coin flip. MEASURED:
                                      # world 33.8% vs view 28.8%, so world is the answer.
                                      # FR_MFXDN_NORMALS=world|view keeps the lever, because that
                                      # is a per-scene measurement rather than a proof; its view
                                      # arm is a SECOND pure function (fsr::stage_normal_view),
                                      # never a parameter on stage_normal — planes.rs's ratchet
                                      # rule, applied to a diagnostic. THAT ARM IS GATED LIKE THE
                                      # REST and shipped for one review cycle without being: it
                                      # is the only stager here that does arithmetic rather than
                                      # a bit copy, so the world-vs-view measurement above rested
                                      # on untested code — and, because `mod fsr` carries
                                      # #[cfg_attr(not(windows), allow(dead_code))], its one
                                      # macOS-only caller also made it a dead_code WARNING on the
                                      # Windows target, invisible from this box. One fix closed
                                      # both. The probe rotation is an exact axis PERMUTATION
                                      # (columns (0,0,-1)/(0,1,0)/(1,0,0) = a 90 degree yaw, so
                                      # (x,y,z) -> (z,y,-x)) precisely so the assertion can be
                                      # BITWISE: every coefficient is 0 or +-1, so the products
                                      # and sums are exact in f32 and each output lane is a
                                      # source lane re-encoded from a value that came out of f16
                                      # — deliberately NOT from_rotation_y(FRAC_PI_2), whose
                                      # cosine is -4.4e-8 and would turn the whole thing into an
                                      # uncalibratable threshold. Two teeth, both fired: a LARGE
                                      # translation column (100,-50,25) catches the obvious wrong
                                      # spelling (Mat4 * Vec4(n, 1.0) instead of the rotation
                                      # alone — a normal is a direction), measured 0x5648 vs
                                      # 0x3800 on the first pixel; and an IDENTITY arm requires
                                      # byte-for-byte agreement with stage_normal, which pins the
                                      # f16 round trip AND stops FR_MFXDN_NORMALS from A/Bing two
                                      # differences at once (an un-zeroed alpha fails it by
                                      # name).
                                      # THE FOUR NEW STAGERS INTRODUCE NO ENCODING CONVENTION, and
                                      # that is the design: fsr::stage_normal / stage_roughness /
                                      # stage_albedo / stage_hit_dist are bit copies or lane
                                      # extracts (GBufs::normal_rough is ALREADY an RGBA16Float
                                      # layout, and its w lane IS roughness), so the f16 narrowing
                                      # that already happened at GBufs::write is the only one.
                                      # They ride fsr::stage_self_test, which --check-fsr runs on
                                      # Windows and Linux too. Teeth exercised BOTH ways: a
                                      # stage_roughness reading lane 0 fails by name, and a
                                      # stage_normal SMUGGLING roughness into alpha fails by name
                                      # — the probe is built so the w lane is always positive and
                                      # the x lane always negative, since either check is vacuous
                                      # on a probe whose lanes agree.
                                      # TWO FINDINGS ABOUT APPLE'S IMPLEMENTATION, both measured
                                      # and both left as REPORTS rather than softened assertions.
                                      # (1) THE TWO CAMERA MATRICES ARE NOT SET AT ALL.
                                      # worldToViewMatrix/viewToClipMatrix are absent from
                                      # objc2-metal-fx (its header-translator drops
                                      # simd_float4x4), so reaching them means objc_msgSend
                                      # transmuted to a hand-written AArch64 HVA signature. Built,
                                      # then removed, on two measurements: the round trip does not
                                      # work (viewToClip read back EXACTLY while worldToView read
                                      # back as pointer-shaped garbage — and setting ONLY
                                      # worldToView then reading viewToClip returned the WORLD
                                      # matrix, i.e. the "successful" reads were register residue
                                      # from the preceding setter), and setting them changes
                                      # NOTHING (every X5 number byte-identical with both setters
                                      # removed; a 90-degree rotation of world_to_view moves the
                                      # output by exactly 0). So: no measurable benefit and an
                                      # unverifiable 64-byte write. AN UNVERIFIABLE WRITE IS WORSE
                                      # THAN AN OMISSION — "unset" is a defined default a future
                                      # driver would read sanely, "written through an ABI we have
                                      # evidence against" is not. The way in if they ever matter
                                      # is an ObjC++ shim taking const float*, where the compiler
                                      # generates the ABI. (2) THE SPECULAR-HIT PLANE IS ACCEPTED,
                                      # ADVERTISED AND UNUSED: the descriptor takes
                                      # setSpecularHitDistanceTextureEnabled(true) and reads it
                                      # back true (mfxdn::new FAILS if it does not), the scaler
                                      # reports wanting ShaderRead on it, we bind it — and zeroing
                                      # it leaves the output BIT-IDENTICAL at a pose where 12.7%
                                      # of the plane is non-zero. Wiring stays (correct, one
                                      # texture, free the day a driver reads it); the assertion
                                      # does not. The other four guides ARE asserted, and the
                                      # probe carries its own anti-vacuity: a plane that is
                                      # already all-zero SKIPs, because zeroing it would score the
                                      # scene rather than the wiring.
                                      # X6 IS A REPORT, NOT AN ASSERTION, and the plan's intended
                                      # claim is retired by measurement. Frame generation's
                                      # product is presented CADENCE and this harness has no
                                      # presentation, so the design was to use the ground truth an
                                      # in-between frame does have — render the midpoint pose,
                                      # require the interpolation of A->B to land closer to M than
                                      # a 50/50 blend of A and B. IT DOES NOT WORK: the
                                      # interpolator returns the CURRENT frame (5.3e-5 from B
                                      # against an A-to-B distance of 8.0e-2). THREE
                                      # configurations give byte-identical results — a single
                                      # reset dispatch, a primed pair, and a driven
                                      # MTLFXTemporalScaler attached through the descriptor's
                                      # `scaler` property — and the plant that settles it is
                                      # handing it frame B as the PREVIOUS frame, which changes
                                      # nothing: prevColorTexture is not read. So X6 asserts the
                                      # wiring (creates, accepts our formats and usages,
                                      # dispatches, writes a finite non-empty frame) and prints
                                      # the quality numbers with the reason they are not
                                      # assertions. A green X6 means ready for a presentation
                                      # stage, not that generation works.
                                      # AND X6 SKIPS UNDER MTL_SHADER_VALIDATION=1, for a reason
                                      # that is not ours: one of MetalFX's own interpolation
                                      # kernels dispatches a 32x32 threadgroup while the validated
                                      # device limit drops to 832 threads, and Metal ABORTS THE
                                      # PROCESS on the assertion rather than failing the encode
                                      # (_validateThreadsPerThreadgroup:1310, "1024 must be <=
                                      # 832"). We author no compute kernels on this path. Skipping
                                      # is what keeps the VALIDATED run — the stricter one, the
                                      # one this gate's own NOTE tells people to prefer —
                                      # runnable at all.
                                      # --cinematic RECONSTRUCTS THROUGH IT on macOS (B4 phase 4),
                                      # which is the first thing on this platform that produces a
                                      # PICTURE rather than a number. A STAGE ON THE CPU ARM, NOT
                                      # A NEW ARM: cinematic::pick_arm is pure and gated in
                                      # --check, and the tracer really is still the CPU one, so
                                      # the label gains a suffix (cpu+mfx-dn) and pick_arm gains
                                      # no case. Ladder: denoised -> plain -> accumulation, each
                                      # degrading loudly. THE MID-SHOT SHED FALLS THROUGH RATHER
                                      # THAN SKIPPING, and that is the one place the obvious
                                      # spelling is wrong: a reconstruction failure at frame f
                                      # clears the reconstructor AND lets frame f render by
                                      # accumulation, because an Err arm that `continue`d to the
                                      # next f would leave a HOLE in the numbered sequence — and
                                      # cine_encode feeds ffmpeg an image2 pattern, whose input
                                      # stops at the first missing index, so one failed frame
                                      # would truncate the whole clip rather than roughen one
                                      # frame of it. Demonstrated with a planted failure: pre-fix
                                      # a 3-frame shot wrote f_00001/f_00002 and lost f_00000
                                      # (the plant fires during frame 0's own 64-pass warm-up),
                                      # post-fix all three are present.
                                      # --no-upscale selects accumulation, and
                                      # GI shots always take it (the hemisphere integrator is a
                                      # still-frame accumulation contract, so a reconstructor fed
                                      # one bounce sample per sub-frame would be reconstructing
                                      # from an estimator that never converged — the GPU arm makes
                                      # the same exclusion). 1:1, DLAA-shaped, matching
                                      # gpu::CineUp's own "100% render scale": a capture has no
                                      # frame budget to buy back. The sub-frame contract is
                                      # run_cinematic_gpu's verbatim — free-running seq, reset
                                      # only at seq 0, and the output-frame-0 warm-up of
                                      # JITTER_PHASE - samples emitting passes, without which
                                      # frame 0 is reconstructed from under half a jitter phase on
                                      # a biased lattice (a discontinuity that shows once per lap
                                      # in a looping clip). MEASURED at 640x360x16: Laplacian 2.03
                                      # vs accumulation's 6.16 — 3.0x less high-frequency content
                                      # — with the mean level unmoved (138.73 vs 139.00), which is
                                      # the shared tone curve holding (cine_write_frame owns it,
                                      # so all arms tonemap identically). KNOWN COST: structure
                                      # replay is NOT used on this path, because each sub-frame
                                      # needs its own G-buffer and its own jitter while
                                      # render_frame_replay re-shades from a fresh ctx without
                                      # re-deriving the G-buffer writes the reconstructor depends
                                      # on — a documented follow-on worth roughly the
                                      # frustum-query share of a sub-frame.
                                      # Touch src/mtl/mfxdn.rs / mfxfi.rs / planes.rs's Guides /
                                      # the four fsr::stage_* guides / CineRecon /
                                      # cine_reconstruct -> run --check-fsr (the pure half, and it
                                      # runs on Windows and Linux), --check-metalfx plain AND
                                      # under MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1,
                                      # FR_MFXDN_NORMALS=view and FR_MFXDN_JITTER=neg (both must
                                      # still PASS — they are measurement levers, not teeth),
                                      # --check-fsr3 (planes::Trio is shared, so its U4 numbers
                                      # are the before/after fingerprint), --check, --check-dlss,
                                      # --check-xess, --check-spirv, cargo test, and
                                      # `--cinematic hero --no-world --cinematic-res 640x360` with
                                      # and without --no-upscale. Restore check.png/check_gi.png:
                                      # they are tracked WINDOWS goldens.
cargo run --release -- --check-mtl    # THE METAL BACKEND ACTUALLY RUNNING SOMETHING (macOS;
                                      # src/mtl/bind.rs + src/mtl/smoke.rs, 2026-08-12 — the Metal
                                      # port's C2, "how do we bind it", which msl.rs:7-8 names).
                                      # --check-msl proves the corpus COMPILES to .metallib; this
                                      # proves a DEVICE consumes one — the same pair --check-spirv
                                      # and --check-vk already make, and a metallib can be perfectly
                                      # well-formed and still read the wrong resource. THE SUBJECT IS
                                      # smoke.hlsl, already in the shared corpus (main.rs's
                                      # corpus_units), and already what D3D12 (gpu::trace::smoke_test)
                                      # and Vulkan (vk::headless::smoke_test, V3) each run as their
                                      # first dispatch: three backends, one kernel, comparable line
                                      # for line. It proves the wavefront tracer's own machinery in
                                      # miniature — constants reaching a kernel, a storage buffer
                                      # written, a GPU-WRITTEN COUNTER turned into dispatch arguments,
                                      # and a third kernel launched INDIRECTLY from them with the CPU
                                      # never seeing the count.
                                      # THE MAP IS DERIVED AND IT CANNOT BE A TABLE, which is the
                                      # milestone's whole finding. Under the shipping msl::CROSS_ARGS
                                      # (argument buffers on, --msl-decoration-binding deliberately
                                      # off) spirv-cross moves each resource inside a per-set struct
                                      # as [[id(n)]] — and those ids are NOT the SPIR-V bindings.
                                      # They are dense, sequential, and assigned in ascending binding
                                      # order over ONLY the resources that entry point references.
                                      # MEASURED on smoke.hlsl, whose three kernels disagree:
                                      #   cs_seed  id0 Push (b1->1)      id1 counters (u0->2000)
                                      #   cs_prep  id0 counters          id1 args     (u1->2001)
                                      #   cs_fill  id0 counters          id1 outbuf   (u2->2002)
                                      # `counters` is id(1) in one kernel and id(0) in the other two,
                                      # and `args` is absent from cs_fill entirely. So a hand-written
                                      # table is not merely a liability the way create_root_signature
                                      # is — it is UNREPRESENTABLE, and one argument buffer shared
                                      # across the three pipelines would bind the wrong pointers with
                                      # no error anywhere. (This also corrects msl.rs:38-47's recorded
                                      # [[id(0)]] [[id(1000)]] [[id(2000)]] [[id(3000)]] — that is the
                                      # layout when --msl-decoration-binding IS passed.)
                                      # TWO DERIVATIONS, CROSS-CHECKED, because one cannot check
                                      # itself and Metal has no validation layer armed by default (CI
                                      # runs the Metal job with them explicitly OFF): mtl::bind::derive
                                      # reads the map off the COMPILED MTLFunction, vk::reflect reads
                                      # the same module's SPIR-V independently (it already runs on
                                      # macOS — --check-spirv's S0), and cross_check requires them to
                                      # agree. The join key is the resource NAME and that it works is
                                      # MEASURED: spirv-cross's MSL member names and the SPIR-V
                                      # OpNames are byte-identical (Push/counters/args/outbuf), so no
                                      # normalizer. cross_check also PINS the dense-ascending-binding
                                      # rule — not by hardcoding an id, but so a change in
                                      # spirv-cross's assignment is a loud finding instead of a
                                      # silently different binding.
                                      # THE DEPRECATED REFLECTION SPELLING IS DELIBERATE and the
                                      # reason is objc2, not taste: the modern route is
                                      # MTLComputePipelineReflection::bindings() ->
                                      # MTLBufferBinding::bufferStructType(), but MTLBufferBinding is
                                      # an extern_protocol! and objc2 0.6 implements DowncastTarget
                                      # for classes only, so a ProtocolObject<dyn MTLBinding> cannot
                                      # be narrowed to it without hand-rolled msg_send!.
                                      # newArgumentEncoderWithBufferIndex:reflection: hands back the
                                      # concrete MTLArgument class. The ENCODER also writes each
                                      # member at the offset the compiled function declares, so
                                      # nothing here computes id*8 or assumes tier-2 argument buffers
                                      # are raw pointers.
                                      # THE GROUP SIZE COMES FROM THE HOST, which is Metal-specific:
                                      # MSL carries no [numthreads] and both dispatchThreadgroups: and
                                      # the indirect form take it as an argument. crate::spirv::
                                      # local_size recovers it from OpExecutionMode LocalSize and
                                      # ERRS rather than defaulting to [1,1,1] — a value
                                      # indistinguishable from a real 1x1x1 kernel that would run
                                      # cs_fill at 1/64 rate with no error anywhere (build.rs's own
                                      # argument, whose spirv_local_size is the documented TWIN; a
                                      # build script is its own compilation unit and cannot `use
                                      # crate::spirv`, the fnv1a64 situation).
                                      # RESIDENCY IS STRUCTURAL, NOT CHECKED. A resource reached
                                      # THROUGH an argument buffer is neither made resident nor
                                      # hazard-tracked, so useResource: is mandatory and has no Vulkan
                                      # or D3D12 analogue. It is also MEASURED UNOBSERVABLE here —
                                      # FR_MTL_NO_RESIDENCY changes nothing, plain AND under
                                      # MTL_DEBUG_LAYER=1 AND MTL_SHADER_VALIDATION=1, because on
                                      # unified memory a non-heap MTLBuffer is already page-backed. So
                                      # ArgBuf exposes ONE mutator, set_buffer, which records the
                                      # resource as it binds it: "forgot useResource" is not
                                      # EXPRESSIBLE rather than caught (planes.rs's move). It turns
                                      # fatal the day these buffers move into an MTLHeap.
                                      # THE ENCODER IS SHARED, AND THE RETARGET IS PER WRITE — the
                                      # one invariant here that ordinary use would never exercise. An
                                      # ArgBuf holds a `Retained::clone` of its Kernel's
                                      # MTLArgumentEncoder, which is a RETAIN and not a copy, and
                                      # setArgumentBuffer:offset: is what selects the destination. So
                                      # pointing it once at construction makes a LATER arg_buf on the
                                      # same kernel silently redirect the first one's remaining
                                      # writes into the new buffer — while bind() keeps using the
                                      # right buffer, so nothing looks wrong. It shipped that way for
                                      # a review cycle, correct only because two statements happened
                                      # to be in the right order (the seed-map plant is the sole path
                                      # that creates two, and it fails structurally first).
                                      # set_buffer re-points per write instead, which costs one
                                      # message send and makes each ArgBuf self-contained. GATED, and
                                      # the gate had to be built for it: K3's isolation arm creates
                                      # two ArgBufs from ONE kernel, writes the SECOND, snapshots its
                                      # bytes, then writes the FIRST and requires the second's bytes
                                      # UNCHANGED. Scored as a before/after on one buffer because
                                      # "is the first still empty" would rest on Metal
                                      # zero-initialising a fresh allocation — true in practice,
                                      # promised nowhere — and the order matters (creating the second
                                      # is what steals the encoder, so the first's write must come
                                      # after). Anti-vacuity: the second's own write must have landed
                                      # first, else the compare is between two empty reads. TOOTH
                                      # FIRED: the pre-fix code reads "writing to one ArgBuf changed
                                      # another's bytes ([23040, 21, 0, 0] -> [22784, 21, 0, 0])" —
                                      # the two probe buffers' addresses, one overwriting the other.
                                      # THE POISON IS PER BUFFER, and that is a SAFETY property the
                                      # plants make real: vk/headless.rs poisons outbuf alone (its
                                      # validation layer catches the rest), but `args` is read by the
                                      # HARDWARE as a threadgroup grid, so SENTINEL there is ~5e10
                                      # threadgroups — a wedged WindowServer on a dev Mac; and
                                      # counters[1] is cs_fill's bounds guard, so a poison past
                                      # outbuf's length is an out-of-bounds WRITE. Hence GRID_POISON
                                      # (cube = 27) and COUNT_POISON (const-asserted inside the
                                      # allocation).
                                      # THE SAME HAZARD BOUNDS AN ALLOCATION, not just a poison:
                                      # `counters` is FOUR words though smoke.hlsl reads two, because
                                      # spirv-cross emits `packed_uint3 _m0[1]` for
                                      # RWStructuredBuffer<uint3> (measured), so cs_prep's
                                      # `args[0] = uint3(...)` is a 12-byte store — and swap_members
                                      # deliberately points the `args` MEMBER at that buffer. At two
                                      # words the plant ran 4 bytes past the allocation: benign
                                      # (Metal rounds up) and unreported by either validation layer,
                                      # but a plant that corrupts memory instead of failing an
                                      # assertion is not a tooth.
                                      # Stages: K0 pure (no device, no toolchain) | K1 the device,
                                      # SKIP on absent AND on argument-buffer tier 1 (an environment
                                      # fact — CROSS_ARGS asks spirv-cross for tier 2) | K2 the
                                      # toolchain | K3 the derived map + cross-check + the group-size
                                      # assertion + the per-ArgBuf isolation arm above | K4 the
                                      # chain at 555 and at 0. K3 and K4 are both
                                      # needed: a map that is self-consistently wrong passes K3, a
                                      # binding that happens to land right passes K4. K3 asserts
                                      # local_size DIRECTLY because cs_fill's own id.x < counters[1]
                                      # guard makes an OVER-large threadgroup produce a byte-identical
                                      # readback — invisible to everything K4 has.
                                      # TEETH, all five fired and their observables recorded (M1):
                                      # FR_MTL_SEED_MAP (bind cs_prep through cs_seed's map — exactly
                                      # what a hand-written table produces) reads "no argument-buffer
                                      # member named `args`" and fails STRUCTURALLY before any
                                      # dispatch; FR_MTL_ARGBUF_INDEX and FR_MTL_SWAP_MEMBERS both
                                      # read "indirect args [3, 3, 3], expected [9, 1, 1]" (the grid
                                      # poison — cs_prep never wrote it); FR_MTL_TG_ONE reads
                                      # "outbuf[9] = 0xeeeeeeee ... (never written)" = 9 groups x 1
                                      # thread. FR_MTL_NO_RESIDENCY is a MEASUREMENT rather than a
                                      # tooth (the Expect::ToolDefect asymmetry): it is REPORTED, and
                                      # demanding it fail would fail the gate on every Apple-silicon
                                      # Mac. FR_MTL_MAP=1 prints the derived map (the FR_VK_MAP idiom).
                                      # A VALIDATION-LAYER NARROWING, free: --check-mtl PASSES on an
                                      # M1 under MTL_DEBUG_LAYER=1 alone, MTL_SHADER_VALIDATION=1
                                      # alone, and neither — so the recorded --check-fsr3 U4
                                      # "uniformly zero under validation" is NOT a general property of
                                      # Metal compute under those layers. Still confounded (that was
                                      # the PARAVIRTUAL runner, this is real silicon), so it is a
                                      # narrowing and not the answer; running --check-mtl under both
                                      # layers ON THAT RUNNER is the cheap next step, since this is a
                                      # 40-line chain with exact-integer observables where FSR3 is
                                      # eleven pipelines and an image.
                                      # SCOPE, stated because a green run does not imply it:
                                      # smoke.hlsl declares no t, no s and nothing in space1, so C2
                                      # exercises the set-0 BUFFER half of CROSS_ARGS and NEITHER
                                      # --msl-device-argument-buffer 1 NOR the tier-2 unbounded texs[]
                                      # array those flags argue hardest for. Textures, samplers and a
                                      # second set are C3's; so is the [loop]-for-[unroll] hemi_wave
                                      # workaround, which needs --check-gpu (unreachable from CI at
                                      # all) and --check-vk re-run.
                                      # C3, FIRST MEASUREMENT (2026-08-14, Apple M1, spirv-cross
                                      # 2026-07-06). Before any host code: src/shaders/mtlbind.hlsl, an
                                      # argument-buffer probe in the SHARED corpus (waveprobe's
                                      # precedent) with three entry points -- cs_tex (set 0 only),
                                      # cs_arr (both), cs_set1 (set 1 only). --check-spirv 80 -> 83
                                      # modules, S3 all validated. It was added to settle one unknown
                                      # and it found a second, larger one.
                                      # THE ID RULE IS DECLARATION ORDER, CONFIRMED ON A SUBJECT THAT
                                      # DISAGREES. The paragraph above -- "assigned in ascending
                                      # binding order" -- is FALSE, and passed only because smoke.hlsl
                                      # declares b1,u0,u1,u2 where the two orders coincide. bind.rs's
                                      # pin was corrected in e1b0abf off FFX SPIR-V; the probe is the
                                      # independent second subject, declared t,s,s,u on purpose:
                                      #   cs_tex   id0 src (t0->1000)    id1 samp_clamp  (s0->3000)
                                      #            id2 samp_repeat       id3 outbuf      (u0->2000)
                                      # Binding order would have been 0,3,1,2. The governing ordinal is
                                      # the SPIR-V OpVariable stream (vk::reflect::Desc::decl), which
                                      # DXC picks and which need not equal HLSL source order -- it does
                                      # here, and one experiment below shows it need not in general.
                                      # STEP 0's THREE-WAY QUESTION ANSWERS "SHAPE A": the unsized
                                      # array stays a MEMBER of the set struct, so MTLArgumentEncoder
                                      # writes everything and bind.rs:55-67's "the encoder writes the
                                      # layout" survives C3 intact. Verbatim:
                                      #     struct spvDescriptorSetBuffer1 {
                                      #         spvDescriptor<texture2d<float>> texs [[id(0)]][1] /* unsized array hack */;
                                      #         sampler samp_aniso [[id(1)]];
                                      #     };
                                      #     kernel void cs_set1(device spvDescriptorSetBuffer1& spvDescriptorSet1 [[buffer(1)]])
                                      # encodedLength therefore budgets exactly ONE element: the host
                                      # over-allocates and writes elements >= 1 raw at offset +
                                      # i*stride, stride from MTLArrayType, never a literal 8.
                                      # cs_set1's signature is also the C2 refutation the probe was
                                      # built for -- one argument buffer, at [[buffer(1)]]. C2's
                                      # `const BUFFER_INDEX: usize = 0` fails here and nowhere else in
                                      # the corpus, and so does a scan written as a dense 0..n loop.
                                      # THE FINDING, AND IT IS NOT A BUG -- though it was recorded as
                                      # one first, and the correction is below. spirv-cross cannot
                                      # lay out a descriptor set holding an unsized array ALONGSIDE
                                      # anything else. It assigns two members the same [[id]], drops
                                      # one behind an "// Overlapping binding:" comment, and rewrites
                                      # that member's uses as a reinterpret_cast of the survivor's
                                      # storage. It COMPILES. It reaches AIR. It reads texture
                                      # descriptor bytes as a sampler. The casualty is samp_lin, the
                                      # trilinear sampler every colour / normal / rough-metal read goes
                                      # through, and the blast radius was NINETEEN MODULES -- leaf,
                                      # leaf_fb, reference and hemi_leaf across every vendor and sway arm,
                                      # every one of them counted as a --check-msl PASS. Two were found by
                                      # reading the generated code; the other seventeen only by writing the
                                      # detector, which is the argument for writing it:
                                      #     spvDescriptor<texture2d<float>> texs [[id(4)]][1] /* unsized array hack */;
                                      #     // Overlapping binding: sampler samp_lin [[id(4)]];
                                      #     sampler samp_aniso [[id(5)]];
                                      #     const device auto &samp_lin = reinterpret_cast<const device sampler &>(spvDescriptorSet1.texs);
                                      # NO ORDERING INSIDE ONE SET AVOIDS IT, measured by moving the
                                      # array through the block: array FIRST drops the member after it;
                                      # array in the MIDDLE or LAST drops the array itself; the array
                                      # ALONE in a set is clean. (The middle case is also where DXC's
                                      # OpVariable stream visibly reordered against source order.) THE
                                      # ARRANGEMENT THAT WORKS is the array alone in its OWN set, both
                                      # sets marked device -- samplers survive, both sample sites bind
                                      # correctly, no overlap anywhere:
                                      #     struct spvDescriptorSetBuffer1 { sampler samp_lin [[id(0)]]; sampler samp_aniso [[id(1)]]; ... };
                                      #     struct spvDescriptorSetBuffer2 { spvDescriptor<texture2d<float>> texs [[id(0)]][1]; };
                                      # TAKEN, after the framing above was corrected. It is NOT a tool bug:
                                      # SPIRV-Cross PR #2292 ("MSL: Add support for overlapping bindings",
                                      # merged 2024) added the drop-and-cast deliberately, and the README
                                      # states the rule it follows -- "arrays of resources consume multiple
                                      # ids, where Vulkan does not... This can be worked around either from
                                      # shader authoring stage or remapping bindings as needed to avoid the
                                      # overlap." An unsized array cannot reserve the ids it will consume,
                                      # so its setmate collides. There is no fix to wait for and nothing to
                                      # report; the installed 1.4.357.0 is already brew's current stable.
                                      # WHY BOUNDING THE ARRAY LOST, since it is the obvious alternative and
                                      # it is smaller: both existing backends deliberately chose scene-exact,
                                      # UNCAPPED sizing -- gpu/trace.rs's range is NumDescriptors u32::MAX
                                      # with the heap slice cut at init, and vk/tracer.rs sizes the layout to
                                      # scene.textures.len(). A fixed N forces a global texture cap plus
                                      # N - textures.len() null-descriptor padding into two working backends
                                      # to accommodate a third that does not exist yet. A scene-derived N is
                                      # worse: it makes shader SOURCE vary with texture count, so every asset
                                      # edit is a fresh compile and corpus_units' source-hash dedupe stops
                                      # meaning anything. The space move keeps both backends' design intact.
                                      # AND IT COST THEM NOTHING, which is the part worth carrying: VULKAN
                                      # NEEDED NO CODE CHANGE AT ALL. vk::layout::Layouts::build is generic
                                      # over sets and derived from reflection, and vk::tracer::bind_textures
                                      # finds the array BY KIND across every set -- find(SampledImage &&
                                      # count == 0) -- then binds into self.sets[set]. The derived-layout
                                      # decision (this file's own M3a argument) paid for itself here. D3D12
                                      # is one descriptor range: RegisterSpace 2 and BaseShaderRegister 0,
                                      # since numbering restarts per space. One table may span two spaces,
                                      # and OffsetInDescriptorsFromTableStart is a HEAP fact rather than a
                                      # register one, so the heap slice, its sizing and write_scene_descriptors
                                      # are all untouched. TEX_TABLE_BUFS stops doubling as the array's base
                                      # register and goes back to one meaning. Metal is one CROSS_ARGS flag,
                                      # --msl-device-argument-buffer 2, derived from UNBOUNDED_ARRAY_SET.
                                      # THE GATE LEARNED TO SEE IT: msl::overlap_check refuses either marker
                                      # (the struct comment and the cast, INDEPENDENTLY -- a dropped member
                                      # that is never used emits only the first). Teeth both ways, and the
                                      # full revert is the honest arm: reverting the shaders ALONE fails 19
                                      # modules on "Runtime sized variables must be in device storage
                                      # argument buffers" -- loud, but that is CROSS_ARGS catching it, not
                                      # the detector. Reverting the shaders AND the flag reproduces the
                                      # original and fires the detector on all 19. A revert test that moves
                                      # only half the fix proves the wrong half.
                                      # THE PROBE-REACH TRAP FIRED ON THE VERY MEASUREMENT THAT FOUND
                                      # THE FIX, exactly as FR_ABL's does. The first run of the
                                      # own-set experiment grepped 0 "Overlapping binding" lines and
                                      # read as CLEAN; spirv-cross had in fact thrown "Runtime sized
                                      # variables must be in device storage argument buffers", because
                                      # --msl-device-argument-buffer names ONE set and space2 was not
                                      # named. A count of zero over an error message is not a null
                                      # result. Re-run with the flag passed for both sets.
                                      # THE TIER LITERAL IS NOT OUT OF RANGE, correcting the reading
                                      # that --help-msl's "0 = Tier1, 1 = Tier2" invites: tier 0 THROWS
                                      # "Unsized array of descriptors requires argument buffer tier 2",
                                      # and 1 and 2 are byte-identical -- the check is >= Tier2, so
                                      # CROSS_ARGS's literal 2 is undocumented spelling and not a bug.
                                      # It is the only value that was ever exercised by texs[]-carrying
                                      # modules, which is why they reached AIR at all. The SET the companion
                                      # --msl-device-argument-buffer names moved 1 -> 2 with the array; it is
                                      # derived from UNBOUNDED_ARRAY_SET rather than written twice, because a
                                      # literal left behind would not have failed loudly -- spirv-cross throws
                                      # "Runtime sized variables must be in device storage argument buffers",
                                      # which reads as a capability problem rather than as a stale constant.
                                      # CPU-ONLY TEETH, since --check-mtl needs a Metal device and is
                                      # not in CI: gfx/shaders.rs pins the probe's space1 block
                                      # BYTE-IDENTICAL to trace_common.hlsli's and pins its set-0
                                      # declaration order as DISAGREEING with binding order -- the one
                                      # property that keeps the corrected id pin non-vacuous. Both
                                      # proven to fail on a planted edit. The first draft of the second
                                      # pin searched for bare identifiers and PASSED on a planted
                                      # reorder, because `outbuf` also occurs in cs_tex's body; it
                                      # anchors on ": register(" now.
                                      # Touch src/mtl/bind.rs / src/mtl/smoke.rs / device.rs's buffer
                                      # + compute half / msl.rs's compile_lib / spirv::local_size /
                                      # src/shaders/mtlbind.hlsl / trace_common.hlsli's space1
                                      # block ->
                                      # run --check-mtl clean AND each of the five levers (four must
                                      # exit 1; the residency one is reported), --check-msl on the
                                      # procedural scene AND san-miguel-low-poly AND --sw-rays,
                                      # --check-spirv, --check-fsr3 and --check-metalfx (they share
                                      # the objc2-metal graph and mtl::device), --check + cargo test,
                                      # then restore the Windows goldens
                                      # C3 STEPS 4-6 -- THE HOST HALF, K6/K7/K8 (2026-08-14, Apple M1,
                                      # macOS 26.5, spirv-cross 2026-07-06). mtlbind.hlsl had been in
                                      # the shared corpus since ef359d6 and EXECUTED BY NOTHING; this
                                      # is what runs it. src/mtl/texprobe.rs is new, mtl::bind goes
                                      # multi-set, device.rs grows samplers, Cargo.toml gains exactly
                                      # one feature (MTLSampler). New stages: K6 cs_tex (a texture
                                      # through two samplers and through .Load, set 0 only), K7 cs_arr
                                      # (the unbounded array walked dynamically across three sets), K8
                                      # cs_set1 (argument buffers at 1 and 2, NOTHING at 0). K5 stays
                                      # the verdict and stays numbered 5.
                                      # THE ID RULE NEEDED CORRECTING A SECOND TIME, and it is the
                                      # same shape as e1b0abf's: the numbering is PER SET. The
                                      # ordering is module-wide (Desc::decl is the OpVariable ordinal
                                      # over the whole module) but each set struct restarts at
                                      # [[id(0)]]. Measured, and mtlbind.hlsl's declaration order is
                                      # texs(space2), samp_lin, samp_aniso, set1_out(space1):
                                      #   cs_tex   set0 { src 0, samp_clamp 1, samp_repeat 2, outbuf 3 }
                                      #   cs_arr   set0 { outbuf 0 } set1 { samp_lin 0, samp_aniso 1 }
                                      #            set2 { texs 0 }
                                      #   cs_set1  set1 { samp_lin 0, set1_out 1 } set2 { texs 0 }
                                      # The pre-C3 cross_check sorted ALL descriptors by decl and
                                      # expected one dense 0..n, so it would have fired on every one
                                      # of those rows -- correct on the one-set subject it was written
                                      # against, wrong the moment the subject gained a second set.
                                      # Caught the same way the first correction was: on a second
                                      # subject that disagrees.
                                      # [[buffer(n)]] == set n, CONFIRMED on all three, including the
                                      # cs_arr case that has set 0 AND higher sets. cs_set1 has
                                      # nothing at buffer(0), which is what makes it the only unit in
                                      # the corpus that can refute BUFFER_INDEX = 0.
                                      # PLACING AN ARRAY ELEMENT TOOK THREE ANSWERS AND THE FIRST TWO
                                      # BOTH LOOKED RIGHT. This is the milestone's finding.
                                      #  (1) MTLArrayType::stride() is 0. The obvious reading of the
                                      #      generated MSL says otherwise -- spvDescriptorArray's
                                      #      operator[] is ptr[i] over &ptr_->value, a contiguous byte
                                      #      walk -- so re-pointing the encoder base by i*stride() is
                                      #      the natural move. Metal does not address an argument-
                                      #      buffer texture entry by byte offset at all and has no
                                      #      byte stride to report. Caught by a structural refusal of
                                      #      stride 0 (`every element would land on element 0`),
                                      #      which was written before the number was known.
                                      #  (2) setTexture:atIndex:(id + i * argumentIndexStride) is the
                                      #      next obvious reading and it is WRONG SILENTLY. The
                                      #      reflected arrayLength is 1 -- the unsized-array hack --
                                      #      so every index past slot.id is past what the encoder was
                                      #      told it owns. All five writes aliased onto element 0 and
                                      #      the readback was the LAST texture written, five times:
                                      #      outbuf[0] = 0xc30004 where 0xc30000 was expected. It does
                                      #      not error. It answers.
                                      #  (3) WHAT WORKS: re-point the encoder's base by one element's
                                      #      BYTES and write at slot.id every time, so the encoder
                                      #      only ever does the one thing it agreed to. The element
                                      #      size is encodedLength (8 B here) -- and it IS one
                                      #      element's size only because texs[] is ALONE in its set,
                                      #      which is the arrangement a19e385 was forced into by the
                                      #      overlapping-binding rule. That fix paid for itself twice.
                                      #      K7 asserts the aloneness directly, so arg_buf_n's sizing
                                      #      rests on a checked premise rather than an assumed one.
                                      # ASKING METAL ABOUT AN UNUSED BUFFER INDEX ABORTS THE PROCESS,
                                      # and this one deserves to be read before anyone writes a scan:
                                      #   -[_MTLFunction newArgumentEncoderWithBufferIndex:reflection:
                                      #     functionReflection:]:11540: failed assertion
                                      #     `bufferIndex 0 does not identify an argument buffer'
                                      # SIGABRT, exit 134. It does NOT return nil. So the obvious
                                      # "scan 0..30 and keep whatever answers" is not merely imprecise
                                      # -- it is a crash on the first unused index, and no amount of
                                      # nil-checking saves it. derive() therefore asks only for the
                                      # sets the SPIR-V declares, and bind.rs's ok_or arm is recorded
                                      # as MEASURED UNREACHABLE rather than left reading as the guard
                                      # it is not. It also reclassifies C2's own constant: on cs_set1,
                                      # `BUFFER_INDEX = 0` would not have mis-derived, it would have
                                      # aborted -- a crash waiting for the corpus to grow.
                                      # WHICH IS WHY FR_MTL_BUFFER0 SIMULATES A NIL METAL DOES NOT
                                      # RETURN. The first version made the real call and produced a
                                      # SIGABRT, and a plant that cannot be told from a crash is not a
                                      # tooth. The host defect is reproduced faithfully; only the
                                      # crash is not.
                                      # SAMPLERS ARE NOT MTLResourceS, so set_sampler records no
                                      # residency -- a different answer, not a missing one. usage_for
                                      # returns Option for exactly this, and K6 asserts 2 resident for
                                      # a kernel with 4 members, so the asymmetry is stated in the
                                      # gate's own output rather than only in prose. MTLResourceUsage
                                      # ::Sample is deprecated and folded into Read; the texture arms
                                      # use Read.
                                      # TEETH: SIX BITE AND TWO ARE MEASUREMENTS, each proven on this
                                      # M1 with its observable recorded, because a tooth whose
                                      # observable was not written down is one nobody can tell has
                                      # gone vacuous:
                                      #   FR_MTL_SWAP_SAMP   K6  outbuf[0]=0xc30000 exp 0xc30001; [2] UNMOVED
                                      #   FR_MTL_TEX_DECOY   K6  outbuf[0]=0xc300ff -- all three move
                                      #   FR_MTL_ARR_ROTATE  K7  outbuf[0]=0xc30001 exp 0xc30000 (wrong INDEX)
                                      #   FR_MTL_ARR_SHORT   K7  outbuf[1]=0x0
                                      #   FR_MTL_ARR_STRIDE  K7  outbuf[1]=0x0 (doubled element size)
                                      #   FR_MTL_BUFFER0     K6  structural, before any dispatch
                                      #   FR_MTL_NO_TEX      --  MEASURED: an unwritten texture slot
                                      #                          reads 0 and is OBSERVABLE; it does
                                      #                          not fault and does not assert
                                      #   FR_MTL_ARR_NORESIDENT  MEASURED: UNOBSERVABLE, matching the
                                      #                          buffer case -- unified memory again
                                      # The first two are a designed PAIR: cs_tex's .Load deliberately
                                      # reads the SAME texel the clamp sample resolves to, so a
                                      # sampler swap moves words 0 and 1 and leaves word 2 alone while
                                      # an unbound texture moves all three. The two failures are
                                      # distinguishable from the readback with no extra instrument,
                                      # and a cargo test pins the coordinate and the .Load texel that
                                      # property rests on.
                                      # FR_MTL_SET_SWAP WAS DESIGNED AND NOT TAKEN: binding set 1's
                                      # buffer at set 2's index makes cs_set1 read a texture array as
                                      # a sampler-plus-pointer struct, which is a plausible device
                                      # hang rather than an error. A plant that can require a reboot
                                      # is not a tooth, and BUFFER0 already covers K8 structurally.
                                      # A VALIDATION NARROWING, and it is the second one --check-mtl
                                      # has contributed: K6/K7/K8 PASS with byte-identical readbacks
                                      # under MTL_DEBUG_LAYER=1 alone and under
                                      # MTL_SHADER_VALIDATION=1 alone. C2 narrowed --check-fsr3's U4
                                      # "uniformly zero under validation" for buffer compute; this
                                      # extends it to TEXTURES, SAMPLERS and an over-allocated
                                      # argument buffer -- the classes all four ffx_fsr3_metal.mm
                                      # GOTCHAs live in. Still confounded with the paravirtual runner,
                                      # so running --check-mtl under both layers THERE remains the
                                      # cheap next step.
                                      # CPU-ONLY TEETH, since --check-mtl needs a device and is not in
                                      # CI: four new cargo tests, each proven to fail on a planted
                                      # edit and then restored -- TEX_N against the host constant, the
                                      # sample coordinate AND the .Load texel K6's diagnostic pair
                                      # rests on, cs_set1 referencing nothing in space0 (K8's entire
                                      # premise), and cs_arr's walk staying dynamic. cargo test 34 ->
                                      # 38. bind::self_test gains the multi-set cases: the measured
                                      # cs_set1 layout ACCEPTED, ids numbered densely across sets
                                      # REJECTED, a buffer at an undeclared set REJECTED, and the
                                      # array's two descriptions disagreeing in both directions.
                                      # Touch src/mtl/texprobe.rs / bind.rs's multi-set half /
                                      # device.rs's sampler half / src/shaders/mtlbind.hlsl /
                                      # gfx::shaders::MTLBIND_* ->
                                      # run --check-mtl clean AND each of the eight levers (six must
                                      # exit 1; NO_TEX and ARR_NORESIDENT are reported), --check-msl
                                      # on the procedural scene AND san-miguel-low-poly AND --sw-rays,
                                      # --check-spirv, --check-fsr3 and --check-metalfx (they share
                                      # the objc2-metal graph, which C3 widened by MTLSampler),
                                      # --check + cargo test, then restore the Windows goldens; plus
                                      # --check-mtl once under each validation layer SEPARATELY
                                      # C3 REVIEW PASS (2026-08-19, same M1, macOS 26.5.1) --
                                      # reviewing steps 4-6 rather than extending them. Two of four
                                      # findings were bookkeeping; two were not.
                                      # FR_MTL_ARR_STRIDE WAS AN OUT-OF-BOUNDS WRITE, AND UNDER
                                      # VALIDATION IT ABORTED RATHER THAN BITING. The plant doubles
                                      # the measured element size but the buffer was still sized for
                                      # the real one -- 5 elements at 16 B into a 40 B allocation, so
                                      # elements 3 and 4 ran past the end. Plain that is invisible and
                                      # the recorded exit-1 observable holds; under MTL_DEBUG_LAYER=1
                                      # it is
                                      #   -[MTLDebugArgumentEncoder setArgumentBuffer:startOffset:
                                      #     elementIndex:]:409: failed assertion `Argument Buffer
                                      #     Validation offset (48) + encodedLength (8) should be
                                      #     smaller or equal to the buffer length (40)'   exit 134
                                      # -- newArgumentEncoderWithBufferIndex:'s lesson a second time:
                                      # the API answers a placement it cannot honour by killing the
                                      # process. It also made two of this milestone's own instructions
                                      # contradict each other, since texprobe's run-list asks for each
                                      # lever AND for a run under each layer, and their intersection
                                      # was a SIGABRT. A plant that cannot be told from a crash is not
                                      # a tooth. Fixed on both sides: texprobe::array_span sizes the
                                      # allocation for the PLANTED stride, so the defect stays a wrong
                                      # PLACEMENT rather than becoming a wrong LENGTH; and
                                      # bind::set_texture_array now refuses a span past the buffer's
                                      # own length(). Proven by reverting the sizing, which turns the
                                      # abort into "FAIL K7 cs_arr: `texs`: 5 element(s) at 16 B each
                                      # need 72 B, but set 2's argument buffer is 40 B -- allocate it
                                      # with `arg_buf_n(m, 9)`", identically plain and under the
                                      # layer. The refusal is the durable half: a future caller who
                                      # forgets arg_buf_n gets a sentence instead of a signal.
                                      # THE RESIDENCY MEASUREMENT REVERSED, ON BOTH RESOURCE CLASSES.
                                      # C2 recorded FR_MTL_NO_RESIDENCY as "bit-identical to a clean
                                      # run, plain AND under MTL_DEBUG_LAYER=1 AND
                                      # MTL_SHADER_VALIDATION=1", and C3 wrote its texture twin
                                      # expecting to EXTEND that. Re-measured, 3/3 each:
                                      #   lever                  plain  DEBUG_LAYER  SHADER_VALIDATION
                                      #   FR_MTL_NO_RESIDENCY    clean  clean        FAIL K4, args
                                      #                                              poison intact
                                      #   FR_MTL_ARR_NORESIDENT  clean  clean        FAIL K7, outbuf
                                      #                                              never written
                                      # So the LAYER decides, not the resource class, and "unobservable
                                      # everywhere" no longer holds. Which measurement was wrong is not
                                      # recoverable -- same machine, one OS point release later -- and
                                      # the rule survives either way: the omission is not reliably
                                      # unobservable, so residency stays STRUCTURAL rather than
                                      # checked. The interesting half is HOW: neither arm produces a
                                      # command-buffer error (device::run checks cb.error() and gets
                                      # None both times), the writes simply do not land. The layer
                                      # diagnoses nothing; the poison-and-read instrument is what
                                      # catches it -- smoke.rs's per-buffer-poison argument holding on
                                      # a class it was not written for. The previous entry's
                                      # validation narrowing stands as written: K6/K7/K8 still pass
                                      # byte-identically under both layers CLEAN. What changed is that
                                      # the LEVERS no longer do.
                                      # ALSO FIXED, none of them observable: bind_opts refuses an index
                                      # override on a multi-set ArgBuf (one index cannot name three
                                      # bind points, and Metal accepts the collapse silently);
                                      # set_sampler gained set_texture's array refusal; the shared
                                      # cross-check loop labels each kernel's failures with its own
                                      # stage rather than printing cs_arr's as K6; and K7's residency
                                      # assertion dropped a guard on arr_noresident that could not fire
                                      # -- resident_count is the list the binder BUILT, which that
                                      # lever does not touch.
                                      # AND ONE PRE-EXISTING FINDING THE WIDENED SWEEP TURNED UP:
                                      # C2's FR_MTL_ARGBUF_INDEX aborts under MTL_DEBUG_LAYER=1 --
                                      #   validateComputeFunctionArguments:1038: failed assertion
                                      #     `Compute Function(cs_seed): missing Buffer binding at
                                      #     index 0 for spvDescriptorSet0[0].'          exit 134
                                      # Same shape as ARR_STRIDE, opposite conclusion. ARR_STRIDE
                                      # aborted because the plant carried a SECOND, gratuitous defect
                                      # beyond the one it meant to plant, so it was fixed. This one
                                      # is inherent: the defect BEING planted is "nothing is bound at
                                      # the index the function reads", which is exactly what the layer
                                      # exists to catch, and its way of reporting is to abort. A
                                      # single-set kernel has no valid-but-wrong index to move to. So
                                      # it is recorded in smoke::Plant rather than fixed, and the
                                      # run-list means each lever, and SEPARATELY the layers on a
                                      # clean run -- not the cross product.

                                      # D2 -- THE IMAGE-ATOMIC EMULATION DIES (2026-08-19, Apple M1,
                                      # macOS 26.5.1, Xcode 26.6, spirv-cross 2026-07-06). The one rung
                                      # of the Metal 4 ladder that needs no Metal 4 API at all, and its
                                      # product is a DELETION: build.rs::MSL_VERSION 30000 -> 30100 makes
                                      # spirv-cross emit NATIVE Metal image atomics, which retires GOTCHA
                                      # 2 and GOTCHA 3 of the four in ffx_fsr3_metal.mm along with the
                                      # buffer-backed emulation itself.
                                      #
                                      # 83 LINES OF CODE, COUNTED RATHER THAN ESTIMATED. The plan said
                                      # "~150-200 lines" and that was an estimate that would have
                                      # shipped as a measurement. ffx_fsr3_metal.mm is -143/+95, but 58
                                      # of the deletions and 93 of the additions are comment or blank:
                                      # 85 lines of code out, 2 in. mtl::device::linear_tex_align adds 7
                                      # more. The comment count going UP while the code count goes down
                                      # is the intended shape -- every deleted site keeps a note saying
                                      # what left, because GOTCHA numbering is cited from smoke.rs,
                                      # main.rs and this file.
                                      #
                                      # MEASURED FIRST, over all 80 committed permutations, because the
                                      # plan this rung was written from got two things wrong.
                                      #
                                      #   30000 -> 30100  20 of 80 permutations change
                                      #                   binding names REMOVED : 20, all *_atomic BUFFERs
                                      #                   binding names ADDED   : 0
                                      #                   indices MOVED         : 0
                                      #
                                      # THE CHANGE IS PURELY SUBTRACTIVE ON THE BINDING SURFACE, and that
                                      # is what made it safe for ffx_metal's discrete binds: the companion
                                      # `device atomic_uint*` disappears and every surviving resource keeps
                                      # its index. Two FFX resources are involved --
                                      # rw_reconstructed_previous_nearest_depth (16 permutations) and
                                      # rw_spd_global_atomic (4, the SPD passes).
                                      #
                                      # THE PLAN SAID 30100/30200/40000 WERE BYTE-IDENTICAL. They are not.
                                      # 3.1 is the exact threshold where the emulation dies; 4.0 ALSO
                                      # rewrites nine barrier sites in the four SPD permutations --
                                      # threadgroup_barrier(mem_device|mem_threadgroup|mem_texture) becomes
                                      # atomic_thread_fence(..., seq_cst, thread_scope_threadgroup) plus a
                                      # narrow threadgroup_barrier -- and adds memory_coherence_device. SPD
                                      # is a CROSS-THREADGROUP reduction and U4's quality comparison is
                                      # report-only, so a subtly wrong reduction there is the
                                      # silent-failure class. The pin is the threshold, and 4.0 became
                                      # FR_FFX_MSL=40000, a measurement rather than a bundled assumption.
                                      #
                                      # THE PLAN ALSO SAID RECIPE MUST BE BUMPED. It must not.
                                      # SPIRV_CROSS_ARGS rides the .toolstamp VERBATIM, so editing the
                                      # version token wipes all 80 metallibs by construction (confirmed:
                                      # the first build after the change reported "0 cached"). RECIPE is
                                      # for changes to build.rs's own Rust that alter emitted bytes
                                      # INVISIBLY -- the 12-byte header format and the -1000 sampler remap.
                                      # Bumping it anyway would have taught the next reader the wrong rule.
                                      #
                                      # EACH EMISSION IMPLIES A -std FLOOR, which matters because
                                      # transpile_ffx_metallib passes NO -std and takes the SDK default
                                      # (__METAL_VERSION__ 400 on SDK 26):
                                      #
                                      #   emit 30000   compiles at -std=metal3.0 and up
                                      #   emit 30100   needs 3.1  "no member named 'atomic_fetch_add'"
                                      #   emit 40000   needs 3.2  "undeclared identifier
                                      #                            'memory_coherence_device'"
                                      #
                                      # So 30100 raises the implicit floor one step and stays independent
                                      # of D1's coming -std pin; 40000 would couple them.
                                      #
                                      # GOTCHA 4 GOT WIDER, AND THAT IS THE ONLY NON-DELETION HERE. R32_UINT
                                      # surfaces could not carry MTLTextureUsageRenderTarget while they were
                                      # buffer-backed (Metal forbids the combination -- which is what GOTCHA
                                      # 3 existed to work around), so they were the standing exception to
                                      # GOTCHA 4. As ordinary textures they take the usage bit like
                                      # everything else, and FFX's reset-frame CLEAR_FLOAT on
                                      # reconstructed-prev-depth goes through the render pass rather than a
                                      # blit fill of a backing buffer. This was the rung's predicted risk
                                      # and it is invisible without the validation layers.
                                      #
                                      # THE C2 DEBT IS DISCHARGED, and the answer retired the hypothesis
                                      # this rung was written on. The recorded --check-fsr3 U4 verdict
                                      # "uniformly zero under validation" (above) was the reason to suspect
                                      # GOTCHA 2 of causing it. Measured BEFORE any change, on this M1:
                                      #
                                      #   config                    exit  energy  history    vs-bilinear
                                      #   plain                     0     1.000x  2.197e-2   2.962e-2
                                      #   MTL_DEBUG_LAYER=1         0     1.000x  2.197e-2   2.962e-2
                                      #   MTL_SHADER_VALIDATION=1   0     1.000x  2.197e-2   2.962e-2
                                      #
                                      # --check-fsr3 PASSES under both layers on real silicon, U4 producing
                                      # identical numbers -- it does NOT read "the dispatch wrote nothing".
                                      # That verdict is PARAVIRTUAL-RUNNER-SPECIFIC, not a property of the
                                      # FSR3 arm, which resolves the confound C2 recorded and C3 narrowed.
                                      # The layers' only visible effect is on the accumulating-frame f16 ULP
                                      # count: 1800 ch plain, 132 under DEBUG_LAYER, 0 under
                                      # SHADER_VALIDATION -- monotone, and the last one is exactly the
                                      # "drops to EXACTLY ZERO" corroboration that
                                      # main.rs::fsr3_upscale_check already records (grep the phrase --
                                      # cited by NAME because this rung's own edits above it moved the
                                      # line number by 60 between writing this entry and reviewing it).
                                      # --check-mtl also passes under each layer separately.
                                      #
                                      # AFTER, same box, same session (the metallib table shrank 1148 KiB
                                      # -> 1121 KiB): plain / DEBUG_LAYER / SHADER_VALIDATION all PASSED,
                                      # every U4 number identical to four significant figures -- energy
                                      # 1.000x, history 2.197e-2, vs-bilinear 2.962e-2, jitter 2.592e-2,
                                      # depth 1.218e-2, motion 3.502e-2, 11 pipelines.
                                      #
                                      # AND THE FINDING THE TOOTH TURNED UP, which is why U1 gained an
                                      # assertion rather than the rung trusting U4. Armed at FR_FFX_MSL=30000
                                      # -- the emulation back in the shaders, gone from the shim -- U4 STILL
                                      # PASSES ALL NINE OF ITS ASSERTIONS:
                                      #
                                      #   metric        correct (30100)   mismatched (30000)   U4 bound
                                      #   energy        1.000x            0.999x               0.5..2.0
                                      #   history       2.197e-2          2.579e-2             >= 1e-3
                                      #   vs-bilinear   2.962e-2          3.090e-2             >= 5e-3
                                      #   jitter        2.592e-2          2.303e-2             >= 1e-3
                                      #   depth         1.218e-2          9.688e-3             >= 1e-3
                                      #
                                      # An unbound atomic argument does not crash and does not zero the
                                      # output; it DEGRADES it, by 17% on history and 20% on depth, well
                                      # inside every bound U4 has. This is the invariants-that-fail-silently
                                      # class exactly, and the only thing that catches it is U1's
                                      # emulation-site count -- which is why that count is asserted and why
                                      # CI greps for it as a third fsr3 guard.
                                      #
                                      # FR_FFX_MSL IS THIS TREE'S FIRST BUILD-TIME LEVER and needs two
                                      # things a runtime one does not. cargo:rerun-if-env-changed=FR_FFX_MSL
                                      # (build.rs had exactly one such line before, Windows-only, for
                                      # FRUSTRACER_DLSS_SDK) or cargo never re-runs the script and the lever
                                      # is inert -- the FR_ABL probe-reach trap in build-script currency. And
                                      # cargo:rustc-env, because a gate cannot read the environment to
                                      # discover what the BUILD did; build.rs states FR_FFX_MSL_BUILT and
                                      # FR_FFX_ATOMIC_SITES, and U1 reads them back with option_env!.
                                      #
                                      # It is a TWO-STEP tooth -- rebuild, then run -- and the run-list says
                                      # so. The vacuity hazard specific to the class is why the tooth is a
                                      # VERSION rather than a failure: a lever that made spirv-cross fail
                                      # outright would empty the table, unset cfg(ffx_fsr3_metal), and U1
                                      # would SKIP and exit 0. At 30000 the transpile still succeeds, the
                                      # table stays full, and the assertion fires.
                                      #
                                      #   lever              class        observable
                                      #   FR_FFX_MSL=30000   TOOTH        20 sites; FAIL U1; exit 1
                                      #   FR_FFX_MSL=40000   MEASUREMENT  0 sites; PASSED; every U4 number
                                      #                                   IDENTICAL to 30100 (table 1122 KiB
                                      #                                   vs 1121). Evidence for a future
                                      #                                   promotion, not a reason to promote:
                                      #                                   one box, one scene, and the SPD
                                      #                                   barrier rewrite is what it would buy.
                                      #   FR_FFX_MSL=31337   (illegal)    loud, reverts to 30100, serves cache
                                      #
                                      # FR_FFX_BUFBACK was DESIGNED AND NOT TAKEN -- the FR_MTL_SET_SWAP
                                      # precedent. Keeping the emulation alive behind a branch forfeits the
                                      # deletion, which is the whole product, and FR_FFX_MSL=30000 already
                                      # covers the claim structurally. FR_FFX_FCV65535 was dropped too: the
                                      # plan predicted "pipeline creation must fail", and the emitted corpus
                                      # at 30100 declares ZERO function constants, so setting one Metal
                                      # ignores proves nothing. The specialized
                                      # newFunctionWithName:constantValues: form STAYS, with an EMPTY value
                                      # set -- Metal refuses the plain variant on any function that declares
                                      # a constant, and empty values fail LOUDLY if spirv-cross reintroduces
                                      # one instead of silently specializing it to a stale number.
                                      #
                                      # WHAT REVIEWING THIS RUNG FOUND, recorded because three of the four
                                      # are the failure classes this file spends its length on and they
                                      # were shipped BY the person who wrote those warnings down.
                                      #
                                      #   * THE CI GUARD WAS VACUOUS ON ITS OWN TEST CASE. `grep -q "0
                                      #     image-atomic emulation sites"` matches "20 image-atomic
                                      #     emulation sites" as a SUBSTRING, and 20 is precisely what
                                      #     FR_FFX_MSL=30000 produces. U1 prints its table line whether or
                                      #     not the assertion fired, so the guard went green on the exact
                                      #     input it was written for. The leading ", " is now load-bearing
                                      #     and both ends say so.
                                      #   * THE GATE FAILED OPEN ON A MISSING COUNT. `option_env!(...)
                                      #     .unwrap_or(0)` reads an absent FR_FFX_ATOMIC_SITES as a clean
                                      #     zero. Unreachable today -- cfg(ffx_fsr3_metal) is only set on
                                      #     the path that emits it -- and unreachable is not a reason to
                                      #     pick the wrong default. Absent is now a FAIL.
                                      #   * MSL_VERSION HAD AN UNPINNED SECOND COPY. U1 compared `built !=
                                      #     "30100"` against a literal, so promoting the pin in D1 would
                                      #     have left every clean run announcing a PLANT that says "this
                                      #     run MUST fail". build.rs now publishes FR_FFX_MSL_DEFAULT and
                                      #     the gate compares against that.
                                      #   * A LINE CITATION ROTTED BEFORE THE COMMIT. This entry cited
                                      #     main.rs:22715; the same rung's edits to that file had already
                                      #     moved the text to :22775. Cited by name now -- the identical
                                      #     repair this rung made to smoke.rs's GOTCHA 1 reference.
                                      #
                                      # And one over-claim removed rather than fixed: the CLEAR_FLOAT arm
                                      # said a nonzero clear "would now be serviced correctly". It would be
                                      # serviced as an INTEGER conversion of the float, which is not
                                      # obviously the same thing, and FFX issues no such request to measure
                                      # against. Zero -- the only value asked for -- is exact either way.
                                      #
                                      # Touch build.rs::MSL_VERSION / spirv_cross_args / count_atomic_emulation
                                      # / shim/ffx_fsr3_metal.mm's GOTCHA sites / mtl::device::line ->
                                      # run --check-fsr3 clean, then FR_FFX_MSL=30000 REBUILD + run (must
                                      # exit 1 on U1), then FR_FFX_MSL=40000 rebuild + run (reported, not
                                      # asserted), then rebuild at the default -- the lever is a BUILD-time
                                      # one and a stale binary measures the wrong arm; --check-fsr3 AND
                                      # --check-mtl once under MTL_DEBUG_LAYER=1 and once under
                                      # MTL_SHADER_VALIDATION=1 SEPARATELY (GOTCHA 4 widened and the layers
                                      # are the only thing that sees it); --check-msl on the procedural scene
                                      # AND san-miguel-low-poly AND --sw-rays; --check-spirv; --check-metalfx;
                                      # --check + cargo test; then restore the Windows goldens.
                                      #
                                      # D4 -- THE METAL 4 SUBMISSION PATH (2026-08-19, Apple M1,
                                      # macOS 26.5.1, Xcode 26.6, SDK 26.5). The rung where the
                                      # campaign question flips: `grep -rn MTL4 src/ shim/` was EMPTY
                                      # through C3 and D2, and `src/mtl/mtl4.rs` is the answer. The same
                                      # smoke.hlsl the other three backends run, submitted through
                                      # MTL4CommandQueue / MTL4CommandBuffer / MTL4CommandAllocator /
                                      # MTL4ArgumentTable / MTLResidencySet. --check-mtl K9 and K10.
                                      #
                                      # THE CLAIM IS THE CROSS-COMPARE, NOT THE COMPILE: one shader, one
                                      # PIPELINE OBJECT, two submission APIs, 619 words identical,
                                      # verified by smoke::verify UNCHANGED. That last part is the
                                      # anti-vacuity half -- a second verifier written the same day by
                                      # the same hand agreeing with the first proves nothing, so mtl4::
                                      # pass returns a smoke::Pass and is held to the function the Metal
                                      # 3 path already answers to.
                                      #
                                      #   check-mtl: K9 MTL4 queue + allocator + shared event OK |
                                      #              metal4 family true | pipelines SHARED with the
                                      #              Metal 3 path (MTL4Compiler not needed)
                                      #   check-mtl: K10 MTL4 seed -> prep -> indirect fill OK at 555
                                      #              and at 0 | 7 allocations resident | argument table
                                      #              by GPU ADDRESS
                                      #   check-mtl: K10 cross-API compare OK -- 619 words, indirect
                                      #              args and counters IDENTICAL
                                      #
                                      # FOUR THINGS WERE MEASURED BEFORE ANY CODE, AND TWO OF THEM
                                      # CHANGED THE LADDER. All read-only, off the objc2-metal 0.3.2
                                      # bindings and the SDK headers.
                                      #
                                      # 1. MTL4 HAS NO useResource: ON ANY ENCODER. Checked against
                                      # every MTL4*.rs binding in the crate, not inferred from prose.
                                      # Residency is expressible only as MTLResidencySet, and
                                      # MTL4ArgumentTable::setAddress:atIndex: takes a RAW MTLGPUAddress
                                      # rather than a buffer -- which is WHY. So D3, planned as its own
                                      # rung ("residency sets"), has no rung of its own: residency is not
                                      # something you add to MTL4, it is something MTL4 refuses to work
                                      # without. D3 dissolved into D4.
                                      #
                                      # 2. MTL4Compiler IS NOT NEEDED, and the ladder assumed it was.
                                      # MTL4ComputeCommandEncoder::setComputePipelineState: takes a plain
                                      # MTLComputePipelineState -- the object bind::Kernel already builds.
                                      # That SHRINKS the rung and STRENGTHENS its claim: the two paths
                                      # share the pipeline, so what differs is submission, binding and
                                      # residency and nothing else. A rung that also swapped the compiler
                                      # could not say that.
                                      #
                                      # 3. NO DEPENDENCY BUMP. objc2-metal 0.3.2 already ships every
                                      # MTL4* binding plus MTLResidencySet, feature-gated. The whole rung
                                      # is a feature-list edit plus one module. (Note the cost: a feature
                                      # edit changes the package metadata hash, so OUT_DIR moves and the
                                      # 80 FSR3 metallibs transpile cold -- ~55 s, twice here.)
                                      #
                                      # 4. probe-metal4 HAD NEVER RUN. `gh run list` shows zero
                                      # workflow_dispatch runs, so the job answered nothing it was
                                      # written for. RETIRED in this rung -- K1 prints `metal4 family`
                                      # and `arg-buffers tier` and K9 builds the objects, on EVERY run.
                                      # A probe that reports is strictly worse than a gate that runs.
                                      #
                                      # THE FOUR THINGS MTL4 TOOK AWAY. Three cost a line of code here
                                      # and carry a claim the Metal 3 gate cannot make; the fourth
                                      # deliberately costs none:
                                      #   * hazard tracking -- MTLDispatchTypeSerial had the driver order
                                      #     and hazard-track the three dispatches, which is why smoke.rs
                                      #     says vk/headless.rs's barrier apparatus "has no analogue here
                                      #     rather than being omitted". Under MTL4 it has one.
                                      #   * implicit residency -- see 1 above.
                                      #   * waitUntilCompleted -- the queue signals an MTLSharedEvent and
                                      #     the CPU blocks with a 2 s timeout, so a submission that never
                                      #     lands FAILS the gate instead of hanging it. The allocator is
                                      #     reset only after a wait that actually RETURNED: the timeout
                                      #     branch is the one case where the GPU provably still owns the
                                      #     encoded commands, so it leaks rather than resetting on the
                                      #     strength of a wait that did not happen. Same for the
                                      #     residency set, which mtl4::pass releases only on success.
                                      #   * the command buffer's `error`, AND THIS ONE IS NOT REPLACED.
                                      #     MTL4 has no synchronous error channel anywhere -- not on the
                                      #     queue, the command buffer or the event. The whole channel is
                                      #     MTL4CommitFeedback::error, an async BLOCK through
                                      #     MTL4CommitOptions::addFeedbackHandler:. It is NOT wired,
                                      #     because reading a captured error after the event wait would
                                      #     report "none" both when there was none and when the handler
                                      #     had not run yet -- the FR_ABL probe-reach trap. Doing it
                                      #     properly means making the FEEDBACK the completion signal
                                      #     instead of the event, a different design for the wait rather
                                      #     than a check bolted on. What it costs today is diagnosis and
                                      #     not coverage: smoke::verify reads all 619 words, so a faulted
                                      #     submission still fails the gate -- on its DATA, where Metal 3
                                      #     would have named the command buffer. A rung extending this
                                      #     path beyond the smoke chain should wire it FIRST.
                                      #
                                      # THE LEVERS, CLASSIFIED BY MEASUREMENT AND NOT BY PREDICTION --
                                      # the D2 lesson (FR_FFX_FCV65535 was planned as a tooth, measured
                                      # as a no-op, and dropped). The plan predicted the split the other
                                      # way round and was wrong on both halves:
                                      #
                                      #   FR_MTL4_TABLE_INDEX   TOOTH  K10 "indirect args [3, 3, 3]" --
                                      #                                GRID_POISON intact, exactly its
                                      #                                Metal 3 twin FR_MTL_ARGBUF_INDEX.
                                      #   FR_MTL4_NO_BARRIER    TOOTH  K10 "indirect args [4, 1, 1]",
                                      #                                7/7 identical, plain AND under
                                      #                                both validation layers.
                                      #   FR_MTL4_NO_RESIDENCY  MEAS.  UNOBSERVABLE plain and under
                                      #                                MTL_DEBUG_LAYER=1; observable
                                      #                                under MTL_SHADER_VALIDATION=1.
                                      #   FR_MTL4_OFF           --     forces the SKIP branch on a box
                                      #                                that HAS MTL4. Exits 0 by design.
                                      #
                                      # [4, 1, 1] IS THE WHOLE BARRIER ARGUMENT IN ONE NUMBER, and it is
                                      # not the poison. It is COUNT_POISON.div_ceil(64) -- 238 rounded up
                                      # over the group size. cs_prep RAN and read `counters` before
                                      # cs_seed had written it: a plain read-before-write, deterministic,
                                      # and reported by NEITHER validation layer. This is the one claim
                                      # this backend gained by moving to MTL4 rather than re-proving.
                                      #
                                      # THE RESIDENCY PREDICTION WAS WRONG, and that is the other
                                      # finding. The rung was planned expecting binding-by-raw-address to
                                      # make an omitted residency declaration fail outright, promoting
                                      # Metal 3's MEASUREMENT into a TOOTH. It does not.
                                      # FR_MTL4_NO_RESIDENCY behaves EXACTLY like FR_MTL_NO_RESIDENCY --
                                      # invisible until GPU validation is armed, at which point cs_prep
                                      # never writes `args` and the grid poison survives as [3, 3, 3]. On
                                      # unified memory a non-heap buffer is already page-backed, and the
                                      # argument table taking an address rather than an object does not
                                      # change that. So must_fail() does NOT name it, exactly as
                                      # smoke::Plant::must_fail does not name its twin.
                                      #
                                      # ITS FIRST MEASUREMENT WAS CIRCULAR AND HAD TO BE RETAKEN, which
                                      # is the methodological finding of the rung. K10 asserted the
                                      # residency set's allocationCount, and this plant zeroes that BY
                                      # CONSTRUCTION -- so the gate failed on having noticed its own
                                      # plant, exited 1, and printed "observable" having learned nothing
                                      # about the GPU at all. The assertion is now skipped when the lever
                                      # is armed, which is what leaves only the DATA able to fail. A
                                      # plant that trips the gate's own bookkeeping is not a measurement
                                      # of the platform.
                                      #
                                      # AND THE ANTI-VACUITY COUNT ITSELF WAS WRONG ON THE FIRST RUN,
                                      # caught by the check rather than by a reader. K10 expected 9
                                      # allocations (6 declared resources + 3 argument buffers) and
                                      # MTLResidencySet reported 7. It DEDUPLICATES: `counters` is
                                      # reached by all three kernels, so the nine addAllocation: calls
                                      # name seven distinct allocations. mtl4::pass now dedups by
                                      # identity before adding and the check is exact equality --
                                      # a count compared against the number of CALLS would have failed
                                      # on correct code forever. Note 7 != K4's 6 either: the Metal 3
                                      # path never declares the argument buffers, because it binds them
                                      # as OBJECTS and MTL4 cannot.
                                      #
                                      # AND THE REVIEW CAUGHT A THIRD DEFECT THAT NO GATE ON THIS BOX
                                      # COULD HAVE. run_mtl4 was inserted into main.rs between
                                      # run_mtl_texprobe's doc comment and its #[cfg(target_os =
                                      # "macos")], so the NEW function took the attribute and
                                      # run_mtl_texprobe was left with none -- and `mod mtl` is macOS-
                                      # only, so that function stops compiling on Linux and Windows
                                      # while all six Metal gates pass on a Mac. The class is invisible
                                      # from here by construction: a macOS-only insertion into a file
                                      # whose macOS-only items are gated ONE AT A TIME rather than by a
                                      # region. Each fn now carries its own doc AND its own cfg, and the
                                      # check is an audit over every top-level fn in main.rs -- does the
                                      # body name a platform-gated module, and does its attribute block
                                      # gate it? -- which reports clean (the three other hits are two
                                      # comments and #[cfg(ffx_fsr3_metal)], which implies macOS).
                                      #
                                      # CI: the K9 guard matches BOTH `K9` and `SKIP K9`, and K10 is
                                      # deliberately NOT guarded. This runner is an Apple Paravirtual
                                      # device that nothing has ever probed for Metal 4 -- the job that
                                      # was meant to answer that never ran -- so the guard asserts only
                                      # that the STAGE RAN. Read the first green run's K9 line, which
                                      # prints `metal4 family` either way, and tighten in a FOLLOW-UP.
                                      # Tightening here would be the --check-metalfx mistake ci.yml
                                      # already records: a guard on a line never proven to run.
                                      #
                                      # Touch mtl::mtl4 / mtl::device::Mtl4 / bind::set_binds /
                                      # bind::resident_allocations / the objc2-metal feature list ->
                                      # run --check-mtl clean; then EACH of the four levers SEPARATELY
                                      # (the two teeth must exit 1, and FR_MTL4_NO_BARRIER x5 because a
                                      # race that is flaky is not a tooth); then --check-mtl under
                                      # MTL_DEBUG_LAYER=1 and under MTL_SHADER_VALIDATION=1 SEPARATELY;
                                      # then --check-msl AND --check-fsr3 AND --check-metalfx AND
                                      # --check-spirv, because a Cargo.toml feature edit has whole-graph
                                      # reach and those four share it; then --check + cargo test LAST
                                      # and restore the Windows goldens.
                                      #
                                      # THE wasm32 STEP IS NOT RUNNABLE ON A MAC, and the run-list says
                                      # so rather than listing a step that always fails. `cargo check
                                      # --target wasm32-unknown-unknown` dies in zstd-sys, which hands
                                      # Apple clang an x86-64 .S file (huf_decompress_amd64.S) with
                                      # --target=wasm32 -- reproducible on a clean target dir, and
                                      # nothing to do with this tree. CI's check-wasm runs on
                                      # ubuntu-latest and is green; that job IS the coverage. What made
                                      # it worth attempting at all is that a Cargo.toml feature edit
                                      # has whole-graph reach -- but this one is nine strings inside
                                      # the `cfg(target_os = "macos")` objc2-metal block, so it cannot
                                      # reach a wasm build by construction.

cargo run --release -- --check-mtl     # D4b -- THE MTL4 ERROR CHANNEL, and it turned out to be
                                      # the wait as well (macOS; src/mtl/device.rs +
                                      # src/mtl/mtl4.rs). K11.
                                      #
                                      # D4 SHIPPED THE MTL4 PATH WITH NO ERROR CHECK AT ALL, and said so
                                      # at length rather than hiding it. Metal 3 next door reads
                                      # `cb.error()` and its comment gives the reason -- "the only channel a
                                      # committed buffer has; dropping it silently is how a failed dispatch
                                      # reads as a black image". MTL4 has no synchronous equivalent
                                      # ANYWHERE: not on the queue, not on the command buffer, not on the
                                      # allocator (checked against all 32 generated MTL4* files). The whole
                                      # channel is MTL4CommitFeedback::error, delivered as a block.
                                      #
                                      # D4 declined to bolt it on, and the refusal was correct: a handler
                                      # ordered against nothing the wait observes reports "no error"
                                      # identically when there was none and when it has simply not run yet.
                                      # That is the FR_ABL probe-reach trap, which this project has fallen
                                      # into four times. D4 named the fix in its own doc -- "making the
                                      # FEEDBACK the completion signal instead of the event" -- and left it
                                      # for the rung that first extended the path beyond the smoke chain.
                                      #
                                      # THE TWO REMOVALS ARE ONE REMOVAL, and that is the finding that made
                                      # the rung small. MTL4 took away waitUntilCompleted AND the error
                                      # accessor, and MTL4CommitFeedback is the replacement for BOTH -- it
                                      # is the only thing that reports completion without polling and the
                                      # only thing that carries an error. So the wait is not "an event plus
                                      # a check": it is one handler whose ARRIVAL is completion and whose
                                      # error() is the diagnosis. MTLSharedEvent, the monotone `signalled`
                                      # counter and MTLEvent in Cargo.toml all went with it.
                                      #
                                      # REACH IS NOW STRUCTURAL RATHER THAN ASSERTED, which is the whole
                                      # point. Nothing but the handler can unblock Mtl4::submit, so there is
                                      # no state in which it did not run and we return Ok anyway. Non-reach
                                      # is a 2000 ms timeout and a red gate, not a confident green line.
                                      #
                                      # MEASURED, Apple M1, macOS 26.5.1, quick profile:
                                      #
                                      #   check-mtl: K11 commit feedback OK -- 3 commits, 3 handlers,
                                      #              GPU 0.015 ms on the last one; 0 errors reported
                                      #
                                      #   clean runs                     10/10 quick, then 5/5 release after
                                      #                                  the counter fix below -- 3/3 every run
                                      #   FR_MTL4_NO_FEEDBACK            exit 1, ~3 s: K10 "did not arrive
                                      #                                  within 2000 ms", K11 "1 commits but
                                      #                                  0 feedback handlers"
                                      #   FR_MTL4_TABLE_INDEX            exit 1, [3, 3, 3] -- SIGNATURE UNCHANGED
                                      #   FR_MTL4_NO_BARRIER             exit 1, [4, 1, 1] -- SIGNATURE UNCHANGED
                                      #   FR_MTL4_NO_RESIDENCY           exit 0            -- SIGNATURE UNCHANGED
                                      #   FR_MTL4_OFF                    exit 0, SKIP K9   -- SIGNATURE UNCHANGED
                                      #   MTL_DEBUG_LAYER=1              exit 0
                                      #   MTL_SHADER_VALIDATION=1        exit 0
                                      #   --check-msl/-spirv/-metalfx/-fsr3   all exit 0
                                      #
                                      # THE 10x CLEAN RUN IS NOT PADDING. It is the only instrument for the
                                      # question the bindings do not settle: is feedback delivered strictly
                                      # AFTER the work completes, so that the readback after the wait is
                                      # race-free? smoke::verify reads every one of 619 words, so an early
                                      # wakeup would flake. Ten clean readbacks is evidence; one is not.
                                      # Apple's own wording ("after completing a workload", GPUEndTime = "the
                                      # host time when the GPU finishes execution") agrees, but it is prose.
                                      #
                                      # THE TIMESTAMPS ARE THE OTHER HALF OF THE REACH PROOF, and they cost
                                      # nothing: objc2-core-foundation was already in the build graph. We
                                      # cannot fabricate them. A block handed a stub, or one for a workload
                                      # that never executed, yields 0.0/0.0; nonzero and correctly ordered is
                                      # evidence the handler arrived carrying real data about OUR submission.
                                      # K11 also bounds GPUEndTime-GPUStartTime by the wall time of the wait
                                      # that contained it -- a DURATION comparison across two clocks, never
                                      # an epoch one, because the GPU execution must be a subinterval of the
                                      # wait and that holds without knowing what zero means on either clock.
                                      # Comparing absolute values would need a timebase unreachable without a
                                      # new dependency, and a mis-derived epoch fails a CORRECT run.
                                      #
                                      # A ZERO READING IS A PLATFORM FACT AND NOT A FAILURE. K11 prints NOT
                                      # POPULATED and says the reach then rests on the structure alone. This
                                      # box populates them; the CI runner never reaches K11 at all.
                                      #
                                      # "3 HANDLERS" IS COUNTED IN THE HANDLER, AND THE FIRST DRAFT OF THIS
                                      # RUNG DID NOT DO THAT -- caught in review of the uncommitted change,
                                      # recorded because the mistake is more instructive than the fix. The
                                      # count started life on the WAITING thread, ticked once per successful
                                      # wait. That is once per commit BY CONSTRUCTION: tracing every path
                                      # between the two increments, the only exit that skips the second is the
                                      # timeout, so "handled != commits" could mean nothing except "a wait
                                      # timed out" -- which FR_MTL4_NO_FEEDBACK already reports one line up.
                                      # The field's doc meanwhile claimed it caught two undocumented
                                      # behaviours, per-buffer-vs-per-commit delivery and a double fire, and
                                      # it could see NEITHER. A counter that restates its own denominator,
                                      # wearing a second name and a coverage claim.
                                      #
                                      # THE SHAPE IS THE FR_ABL PROBE-REACH TRAP ARRIVING IN THE COUNTER
                                      # instead of in the check, one layer below where this rung was watching
                                      # for it -- the rung whose entire subject is not trusting a green line
                                      # from a probe that may not have run. Fixed by moving the tick INSIDE
                                      # the block (Arc<AtomicU64>, SeqCst, ordered ahead of the mutex release
                                      # that publishes the payload), so it counts invocations rather than
                                      # observations. K11 reads it after every wait has returned, so a stray
                                      # second fire on an EARLIER submission is still visible.
                                      #
                                      # AND ONLY NOW IS 3/3 A MEASUREMENT. 5/5 clean release runs report 3
                                      # commits and 3 handlers, which is the first evidence in this campaign
                                      # that MTL4 delivers feedback exactly once per commit rather than per
                                      # command buffer -- a thing the bindings state nowhere. Under the old
                                      # counter those same five runs would have printed 3/3 with the handler
                                      # stubbed out entirely.
                                      #
                                      # ONE ORDERING BUG CAME OUT OF THE SAME READ: the wall-clock bound
                                      # returned its Err ahead of `inner?`, so a binding error could be masked
                                      # by a complaint about the platform's clock. submit's own doc states the
                                      # opposite order and Mtl::compute next door keeps it -- a binding error
                                      # is reported before the command buffer's own, being usually its cause
                                      # rather than its symptom. `inner?` now precedes both checks. Unreachable
                                      # in practice; it is a stated contract in a file that runs on them.
                                      #
                                      # THE CHANNEL IS PROVEN DELIVERED AND NOT PROVEN TO REPORT FAULTS, and
                                      # the entry says so rather than implying coverage. Both of the other
                                      # teeth produce WRONG BYTES through a command buffer that reports no
                                      # error -- K11 prints "1 commits, 1 handlers, 0 errors reported" under
                                      # FR_MTL4_TABLE_INDEX and under FR_MTL4_NO_BARRIER alike. smoke.rs
                                      # records Metal 3 answering identically for a residency violation under
                                      # MTL_SHADER_VALIDATION=1: "The command buffer reports no error ... the
                                      # writes simply do not land."
                                      #
                                      # INDUCING A REAL FAULT WAS CONSIDERED AND REJECTED, listed here so the
                                      # next reader does not spend the afternoon re-deriving it:
                                      #   * out-of-bounds shader write -- measured silent next door (above)
                                      #   * a hang -> MTL4CommandQueueError::Timeout -- would work, and is the
                                      #     class smoke.rs::GRID_POISON exists to PREVENT: on a dev Mac it
                                      #     takes the WindowServer with it, in CI it is a 45-minute timeout
                                      #   * an enormous grid -- same class, same refusal
                                      #   * OutOfMemory -- newBufferWithLength returns nil before submission,
                                      #     so the channel is never reached
                                      #   * an unbacked GPU address -- already measured silent, twice
                                      #   * a misaligned indirect-dispatch address -- undefined and
                                      #     non-deterministic; a plant that may not repeat is not an instrument
                                      #
                                      # dispatch2 IS NOT NEEDED, and the binding settles it rather than a
                                      # guess: MTL4CommandQueueDescriptor::feedbackQueue says nil is "the
                                      # default, Metal allocates an internal dispatch queue to service
                                      # feedback notifications". We set nothing and take that queue, which
                                      # also dodges the binding's own warning that the queue "is not retained
                                      # internally". Verified after the edit: cargo tree still shows zero
                                      # dispatch2 in the macOS graph.
                                      #
                                      # WHAT IT DOES COST IS A THREAD, and every capture is owned because of
                                      # it. The handler runs on that Metal-allocated serial queue, so the
                                      # NSError is stringified INSIDE the block -- Retained<T> is not Send --
                                      # and the rendezvous is Arc<Mutex<Option<Feedback>>> + Condvar rather
                                      # than the Cell the event wait could use. The budget is TOTAL, not
                                      # per-wakeup: a spurious wakeup re-derives what is left from the start
                                      # instant, because passing MTL4_WAIT_MS again would make the constant a
                                      # per-wakeup bound instead of the total bound its doc claims.
                                      #
                                      # block2 IS A DIRECT DEPENDENCY NOW, AND IT COSTS NO COMPILE. objc2-
                                      # metal does not re-export it (its lib.rs re-exports only its own
                                      # modules), so RcBlock::new is reachable only by naming the crate --
                                      # which is why the "objc2-foundation is deliberately NOT a direct
                                      # dependency" rule does not extend to it: that rule works because we
                                      # only ever RECEIVE an NSError, and here we must CONSTRUCT a block.
                                      # cargo tree already showed block2 0.6 twice in this build (objc2-metal
                                      # enables it for objc2-metal-fx's MTLFX* features; wgpu-hal enables it
                                      # independently), so naming it converts an accident into an intention --
                                      # the same lesson the MTL4Compiler note in Cargo.toml already records.
                                      #
                                      # AND ONE STALE COMMENT FELL OUT. msl.rs said newLibraryWithData: "is
                                      # gated behind objc2-metal's dispatch2 feature and would pull block2 in
                                      # with it". The block2 half is now false and was corrected in place, not
                                      # deleted -- the wrong half is the one a reader weighing the two routes
                                      # reaches first. The dispatch2 half stands, and so does the file route.
                                      #
                                      # THE THIRD FAILURE MODE. mtl4::pass skipped its residency teardown on
                                      # failure and enumerated exactly two reasons; there are now three, since
                                      # a submission can also fail having REPORTED an error, where the GPU is
                                      # provably done and teardown would be safe. The skip still covers all
                                      # three -- distinguishing them there would put a second copy of submit's
                                      # ending-classification in a second file, where the two could drift.
                                      # submit's own timeout branch grew the matching rule: it leaks the
                                      # RcBlock and the options as well as the allocator, because a wait that
                                      # timed out has not established that the handler will NEVER run -- only
                                      # that it has not run yet, and freeing a callback Metal may still invoke
                                      # is the use-after-free arriving through the cleanup path.
                                      #
                                      # Mtl4::compute IS NOW A WRAPPER over Mtl4::submit, which owns the
                                      # command buffer, the commit and the wait. compute keeps only what is
                                      # about COMPUTE -- making the encoder and reporting a nil one as ours.
                                      # Done here rather than in the MetalFX rung that needs submit, because
                                      # this rung was rewriting the function anyway and one wait with two
                                      # callers cannot drift the way two copies can.
                                      #
                                      # CI STILL CANNOT SEE ANY OF THIS, and the ci.yml comment was rewritten
                                      # from a hedge into the measurement. Run 32309810963 answered D4's open
                                      # question NO: macos-latest is an Apple Paravirtual device reporting
                                      # `metal4 family false`, newMTL4CommandQueue returns nil, SKIP K9 every
                                      # run. --check-fsr3 U2 and --check-metalfx X1 print the same from the
                                      # same device, so it is the device and not a gate-local quirk. The guard
                                      # cannot be tightened until GitHub's image moves to macOS 26, and a
                                      # green check-metal covers the skip branch and nothing else of MTL4.
                                      #
                                      # Touch mtl::device::Mtl4::submit / Mtl4::compute / Mtl4::tally /
                                      # mtl4::Plant / the objc2-metal feature list / the block2 dependency ->
                                      # run --check-mtl clean, and 10x for the delivery-ordering claim; then
                                      # EACH of the five levers SEPARATELY (three teeth must exit 1); then
                                      # --check-mtl under MTL_DEBUG_LAYER=1 and under MTL_SHADER_VALIDATION=1
                                      # SEPARATELY; then --check-msl AND --check-fsr3 AND --check-metalfx AND
                                      # --check-spirv, because a Cargo.toml feature edit has whole-graph reach
                                      # and those four share it; then cargo tree to confirm dispatch2 stayed
                                      # out; then --check + cargo test LAST and restore the Windows goldens.
```
