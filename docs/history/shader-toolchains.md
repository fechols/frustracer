# Shader toolchains — SPIR-V and MSL

`--check-spirv` (the corpus compiled to SPIR-V and validated) and `--check-msl` (SPIR-V to MSL to metallib, the third code generator).

Extracted verbatim from `CLAUDE_Historical.md`, which keeps a stub pointing here. Nothing in this file was rewritten.

```
cargo run --release -- --check-spirv  # THE CORPUS'S SHADER TOOLCHAIN, gated (any OS with a DXC drop;
                                      # unix-only until 2026-08-14; src/spirv.rs
                                      # + run_check_spirv in main.rs, 2026-08-10 — the Vulkan port's M2a).
                                      # NOTE the module MOVED out of `src/vk/` on 2026-08-12 (it names
                                      # no `ash` type and never did — it is the corpus's second CODE
                                      # GENERATOR, and since --check-msl consumes the same SPIR-V through
                                      # spirv-cross, leaving it under vk/ would make the Metal gate
                                      # import the Vulkan backend). `vk::spirv` re-exports it, so every
                                      # call site is unchanged; the -fvk-*-shift constants stay a VULKAN
                                      # choice and Metal deliberately does not reuse them (see
                                      # --check-msl). The unit ENUMERATION also moved, into
                                      # main.rs::corpus_units, shared with --check-msl — the two gates'
                                      # premise is "one corpus, two code generators", which two `push!`
                                      # lists could quietly falsify while both stayed green.
                                      # Assembles the SHIPPING corpus through gfx::shaders — the same
                                      # functions a session calls, which is what M1 moving the assembly
                                      # into the shared core BOUGHT — and compiles every unit to SPIR-V,
                                      # then spirv-val's each module. NOT a spike transcription: a
                                      # hand-mirrored concat would be gating itself.
                                      # S0 pure (the binding scheme + its INVERSE + blob checks + the
                                      # descriptor reflector's own walk; runs with no compiler on disk) |
                                      # S1 loads libdxcompiler.so | S2 assembles+compiles |
                                      # S3 validates. S2 and S3 answer DIFFERENT questions and neither
                                      # substitutes: DXC will happily emit a module no driver accepts,
                                      # and spirv-val validates a MODULE, so it can say nothing about two
                                      # registers landing on one binding (S0's injectivity sweep is that
                                      # guard). MEASURED: 47 units -> 78 modules, 78 validated, 0 failed
                                      # — all four --dxr-inline libraries (mode 0's lib_6_3 all-TraceRay
                                      # build included) plus --dxr-sbt 3's recursive arm, the wavefront
                                      # ladder, FRD's kernels, the FSR composite, quin, and the display
                                      # stage — with ZERO edits to any .hlsl. The corpus is one body of
                                      # source with two code generators; it is not "D3D12 shaders we
                                      # will have to port".
                                      # THE FLAG SET, each entry load-bearing (vk::spirv::spirv_args):
                                      # -spirv; -fspv-target-env=vulkan1.3 (DXC's SPIR-V DEFAULT is
                                      # vulkan1.0, which cannot express what these kernels use — wave
                                      # intrinsics need SPIR-V 1.3 subgroup ops and RayQuery needs
                                      # SPV_KHR_ray_query); -fvk-use-dx-layout (what lets ONE Rust
                                      # packer serve both backends — gfx::frame::FrameCb's 4608 bytes
                                      # stay byte-compatible instead of needing a std140 twin; the price
                                      # is scalarBlockLayout, core in Vulkan 1.2, which the device probe
                                      # must REQUIRE not prefer); -fvk-{b,t,u,s}-shift N all; -HV 2021
                                      # and -O3 identical to the DXIL side (differing optimization
                                      # levels would make the two backends different shaders).
                                      # REGISTERS -> DESCRIPTORS: space becomes the SET, register number
                                      # becomes the binding shifted by TYPE so b0/t0/u0/s0 — all of
                                      # which exist here — cannot collide (b +0, t +1000, u +2000,
                                      # s +3000; largest register in the corpus is u32, so three orders
                                      # of headroom). vk::spirv::binding_of IS that rule and the DXC
                                      # flags are GENERATED from the same constants, so the compiler and
                                      # the descriptor-set layouts cannot disagree — never hardcode a
                                      # binding. Verified in the emitted SPIR-V, not assumed: accum
                                      # (u0 space0) -> set 0 binding 2000, uv_buf (t0 space1) -> set 1
                                      # binding 1000, samp_lin (s0 space1) -> set 1 binding 3000.
                                      # GATE TEETH, all exercised: a planted undeclared identifier in
                                      # smoke.hlsl fails 3 modules and exits 1; a unit yielding no
                                      # [numthreads] entry FAILS rather than silently compiling nothing;
                                      # a corpus that assembled zero units FAILS; and the vendor arms
                                      # must assemble DIFFERENTLY or the run says so — the source-dedupe
                                      # that keeps the module count honest is also the one thing that
                                      # could silently HALVE reach (if cand_defs stopped distinguishing
                                      # them, the AMD CAND_TMIN0 arm would collapse into the NVIDIA one
                                      # and the run would still say PASSED).
                                      # ENTRY POINTS ARE SCANNED OUT OF THE ASSEMBLED SOURCE, never
                                      # listed: a hardcoded table goes stale the first time a kernel
                                      # gains an entry, and silently — the gate would pass having
                                      # compiled less. Scanning [numthreads] cannot, and it over-covers.
                                      # SCENE-KEYED like every other suite (--check-spirv
                                      # san-miguel-low-poly.obj arms ALPHA_CUTOUT/TRANS_SHADOW, which
                                      # the procedural scene cannot reach) — and the summary reports
                                      # ASSEMBLED BYTES because the unit and module COUNTS cannot tell
                                      # two scenes apart: both give 47/78. Measured 7797887 B procedural
                                      # vs 7799031 B san-miguel, with FR_ABL=noalpha,notrans returning
                                      # it EXACTLY to the procedural count — the three-way proof the
                                      # keying reaches. (Refreshed 2026-08-11, TWICE in one day, and
                                      # the second time is the evidence for the caveat: the ABSOLUTE
                                      # figures drift with ANY shader edit — origin's auto-exposure
                                      # light-gain change alone moved them +62315 B — so treat them as
                                      # a snapshot; the +1144 B DELTA between the two scenes is the
                                      # load-bearing part and did not move by a byte across it.) (A first draft printed 2-decimal MB and read
                                      # identical across all three: an instrument at the wrong
                                      # RESOLUTION cannot see the effect it was built for — the v1.5.3
                                      # lab lesson, in a different currency.) FR_SPIRV_LIST=1 names
                                      # every unit, which is how "what did this run really cover" gets
                                      # answered; the dedupe is visible there too (dxr-resolve and
                                      # dxr-sky collapse into the wavefront's — evidence the sources
                                      # really are shared, not merely claimed to be).
                                      # WHAT ONE PROCESS CANNOT VARY: the OnceLock levers (--sw-rays,
                                      # --no-ftree, --heightfield, the FR_* family) — re-run the gate
                                      # under them, the rule tools/dump-hlsl.ps1 states for the same
                                      # reason. DELIBERATELY ABSENT: nppd.hlsl (its stage is ONNX
                                      # Runtime + the DirectML EP, which has no Vulkan execution
                                      # provider — out of the port's scope by decision) and
                                      # workgraph.hlsl (VK_AMDX_shader_enqueue is a vendor provisional,
                                      # and the file is a default-off lever measured as a wash).
                                      # Nothing else in src/shaders/ is skipped.
                                      # SDKs: SDKs/dxc-linux (install-prerequisites.sh dxc — which
                                      # fetches the Linux tarball and the Windows zip from the SAME
                                      # release tag, so the compiler emitting DXIL and the one emitting
                                      # SPIR-V cannot drift to different HLSL front ends) + SDKs/
                                      # spirv-tools (install-prerequisites.sh spirv). ON macOS THE DXC
                                      # HALF IS A SOURCE BUILD, and the pin is why, not effort: DXC_TAG
                                      # publishes a Windows zip, a linux x86_64 tarball and a PDB zip —
                                      # no macOS build, no arm64 build of anything — so a community
                                      # binary would not be from the tag, breaking the same-front-end
                                      # invariant above, which is exactly the drift no gate can see. The
                                      # route that keeps it is a source build at DXC_TAG (the shape
                                      # `nrd` already uses), and `install-prerequisites.sh dxc` now DOES
                                      # that on macOS (do_dxc_macos) instead of standing the half down:
                                      # clone at the tag, cmake+ninja, install to SDKs/dxc-macos/{bin,lib}
                                      # — ~10 min on an M1, and the binary carries
                                      # LC_RPATH=@executable_path/../lib so it finds its dylib with
                                      # nothing added to the environment, exactly as the Linux drop's
                                      # DT_RPATH does. THREE BUILD FLAGS, each a measured workaround, all
                                      # of them compiler/CMake levers rather than patches to the fork (a
                                      # patch would silently un-apply on the next fetch AND would break
                                      # what the pin means): -DCMAKE_POLICY_VERSION_MINIMUM=3.5, because
                                      # DXC is an old LLVM fork and CMake 4 removed compatibility with
                                      # cmake_minimum_required(VERSION <3.5); -DCMAKE_CXX_FLAGS=
                                      # -Wno-invalid-specialization, because llvm/ADT/StringRef.h
                                      # specializes std::is_nothrow_constructible and Xcode 26's libc++
                                      # marks that entity __no_specializations__, which clang now
                                      # diagnoses as an ERROR (three lines in one header, 15 errors, the
                                      # whole build); and -DLLVM_PARALLEL_LINK_JOBS=1, because LLVM link
                                      # steps are the memory-hungry part and 8 GB Apple silicon is a real
                                      # configuration. spirv DOES have a macOS prebuilt,
                                      # x86_64-only, so it runs under Rosetta on Apple silicon and the
                                      # installer checks that it EXECUTES rather than assuming.
                                      # MEASURED on macOS 26.5 / M1 the first time the corpus went
                                      # through a macOS DXC (2026-08-12): 47 units -> 78 modules, 78
                                      # validated, 0 failed — the SAME counts Linux records — with
                                      # assembled bytes 7797887 procedural and 7799031 san-miguel, i.e.
                                      # BOTH absolutes matching the recorded Linux figures and the
                                      # load-bearing +1144 B delta reproduced exactly. The corpus
                                      # assembly is platform-independent, confirmed rather than assumed.
                                      # --check-vk still SKIPs on macOS (no Vulkan loader). Either
                                      # missing =
                                      # loud SKIP + exit 0, the bare-checkout degrade; the path lever is
                                      # FRUSTRACER_DXC_SPIRV_PATH and NOT FRUSTRACER_DXC_PATH, which
                                      # names the Windows drop (two artifacts, two directories — one
                                      # path variable cannot name both). Nothing links the compiler:
                                      # libdxcompiler.so is dlopen'd and DxcCreateInstance resolved by
                                      # symbol, the LoadLibraryExW+GetProcAddress policy spelled
                                      # portably, so every other --check* stays DLL-free.
                                      # WAS unix-only for ONE reason: DXC's WinAdapter typedefs LPCWSTR
                                      # as `const wchar_t*`, 4 bytes there against Windows' 2, so the
                                      # argument array is UTF-32 off Windows and UTF-16 on it. Both arms
                                      # landed 2026-08-14 (the browser port's Stage 0a): WChar, wide(),
                                      # LIB_NAME and default_dir carry two lines each and NOTHING else
                                      # does — the vtables, CLSIDs, flags and binding scheme were always
                                      # neutral, extern "C" is already right on x86_64 Windows (there is
                                      # only one calling convention there), and no dxil.dll is needed
                                      # because SPIR-V has no signing step. Measured on Windows the day
                                      # it landed: 47 units -> 80 modules, 37 -> 68 under --sw-rays.
                                      # `src/vk/reflect.rs` hoisted to `src/reflect.rs` in the same
                                      # change and for the same reason — S0 is device-free and now runs
                                      # where `vk/` does not build; `vk::reflect` re-exports it.
cargo run --release -- --check-msl    # THE METAL SHADER TOOLCHAIN, gated (macOS; src/mtl/msl.rs +
                                      # run_check_msl in main.rs, 2026-08-12 — the Metal port's C1, and
                                      # the first rung of a Metal TRACER rather than another consumer
                                      # of the CPU G-buffer). Takes --check-spirv's corpus one generator
                                      # further: SPIR-V -> MSL (spirv-cross) -> AIR -> .metallib (xcrun
                                      # metal). It renders nothing, binds nothing and dispatches
                                      # nothing — what reaches a metallib and what refuses IS the
                                      # product, the shape M2a had for Vulkan, where it found the one
                                      # blocker nobody predicted. M0 pure arg set + classifier | M1 the
                                      # tools (absent -> SKIP) | M2 assemble + DXC | M3 spirv-cross |
                                      # M4 metal/metallib | M5 the verdict. Wrong OS = exit 2 (the
                                      # --check-fsr3 convention). MEASURED: 47 units -> 78 SPIR-V -> 65
                                      # metallib (2852234 B) on the procedural scene; 73 on
                                      # san-miguel-low-poly; 61 of 66 under --sw-rays.
                                      # M1'S PROBE CHECKS THE EXIT STATUS, NOT THE LAUNCH, and the two
                                      # differ for a reason that is easy to get backwards: the process
                                      # being launched is `xcrun`, which exists on any box with the
                                      # Command Line Tools, so a missing Metal toolchain shows up as
                                      # `xcrun -sdk macosx metal --version` LAUNCHING FINE and exiting
                                      # 72 (measured). A launch-only probe therefore cannot see it at
                                      # all — and the Metal toolchain is a separate MobileAsset cryptex
                                      # the Xcode manifest does not list, so "Xcode present, `metal`
                                      # absent" is exactly the shape of a box that lacks it. It shipped
                                      # that way for a day: the gate let such a box past M1 and turned
                                      # an environment fact into ~65 M4 failures, inverting the
                                      # absent-is-a-SKIP contract the module is written around. The
                                      # SPIRV-CROSS PROBE ABOVE IT IS DELIBERATELY THE OTHER WAY —
                                      # spirv-cross treats `--version` as a usage error and exits
                                      # non-zero on a HEALTHY install, so launch is the only signal
                                      # there, while `metal` and `metallib` both exit 0 (measured). Two
                                      # probes, two rules, per tool. TEETH: `METAL=/usr/bin/false`
                                      # (launchable, cannot answer) and `METALLIB=/nonexistent` (cannot
                                      # launch) must BOTH read `SKIP M1` at exit 0.
                                      # THE ROUTE, settled on evidence: metal-shaderconverter (Apple's
                                      # own DXIL -> metallib tool) is OUT — not installed with Xcode,
                                      # not an xcrun tool, and it consumes SIGNED DXIL, whose signer
                                      # dxil.dll is Windows-only (the same fact the NRD entry records).
                                      # spirv-cross is the route build.rs already runs for FidelityFX
                                      # and CI already installs.
                                      # THE ARG SET, every entry measured (mtl::msl::CROSS_ARGS):
                                      # --msl --msl-version 30000 --msl-argument-buffers
                                      # --msl-argument-buffer-tier 2 --msl-device-argument-buffer 1.
                                      # The device-argument-buffer SET is a derivation, not a literal:
                                      # texs[] is register(t10, space1), the register SPACE becomes the
                                      # descriptor SET, and spirv-cross requires runtime-sized arrays in
                                      # DEVICE storage argument buffers ("Runtime sized variables must
                                      # be in device storage argument buffers" is the exact refusal
                                      # without it); self_test pins the constant and the flag agree.
                                      # --msl-decoration-binding is ABSENT, and the reasoning was
                                      # CORRECTED mid-milestone rather than merely stated: WITHOUT
                                      # argument buffers it is fatal (measured, `bloom` alone emits
                                      # [[texture(1000)]], [[texture(2000)]], [[sampler(3000)]] against
                                      # Metal's 0-127 / 0-15 — the FFX precedent, which lost 112 of 160
                                      # permutations to ONE sampler at 1001 and is fixed by
                                      # build.rs::remap_ffx_samplers' single subtraction; ours would
                                      # need three). WITH them it is merely MOOT — resources move
                                      # inside an argument-buffer struct as [[id(n)]], measured
                                      # [[id(0)]] [[id(1000)]] [[id(2000)]] [[id(3000)]] at
                                      # [[buffer(0)]], where no such ceiling applies, and the same 65
                                      # compile either way. So self_test pins the IMPLICATION that is
                                      # true (asking for it REQUIRES argument buffers) rather than its
                                      # absence, which would pin a preference. Either way the milestone
                                      # boundary holds: the Metal argument indices are spirv-cross's
                                      # business, so NOTHING may hardcode one — C2 derives the map (from
                                      # Metal reflection or spirv-cross's own output), exactly as
                                      # vk::reflect derives the Vulkan one.
                                      # -ffp-contract=off is NOT needed and that was a surprise:
                                      # spirv-cross PRESERVES NoContraction, emitting
                                      # `[[clang::optnone]] T spvFMul(T l, T r){return fma(l,r,T(0));}`
                                      # plus MSL's precise:: namespace — so the corpus's `precise`
                                      # discipline (ftree.hlsli's decoded boxes must CONTAIN the true
                                      # ones or every prune stops being conservative) survives the
                                      # crossing. __METAL_FAST_MATH__ is 0 by default too.
                                      # THE FINDING THAT OVERTURNED THE PLAN: spirv-cross LOWERS
                                      # RayQuery to Metal. leaf/reference/leaf_fb/hemi_leaf all reach
                                      # AIR with hardware ray tracing intact, as
                                      # raytracing::acceleration_structure<raytracing::instancing> and
                                      # raytracing::intersection_query — so a Metal tracer does NOT need
                                      # --sw-rays, which was the milestone's founding premise. What
                                      # Metal has no analogue for is the DXR PIPELINE shape (raygen/
                                      # closest-hit/miss/SBT), i.e. the 5 dxr-lib modules, which the
                                      # port does not need: what is being ported is the wavefront tracer.
                                      # THE VERDICT IS ASYMMETRIC BY CLASS (mtl::msl::Expect), and the
                                      # asymmetry is the milestone's own lesson: NoAnalogue is a
                                      # CAPABILITY claim, so a dxr-lib that ever COMPILES is a hard FAIL
                                      # ("Metal grew an analogue... move it to Metallib") and the class
                                      # must still be REACHED (dxr-lib is enumerated on every scene under
                                      # every lever, so demanding it is safe). ToolDefect is a BUG-
                                      # PRESENCE claim and is REPORTED, never required — because bug
                                      # presence is configuration-dependent: the 8 hemi_wave failures
                                      # need the OPAQUE occluded_q arm, so an ALPHA_CUTOUT/TRANS_SHADOW
                                      # scene compiles them (73/78) and --sw-rays has no RayQuery to
                                      # mis-scope at all (61/66). A first draft demanded both classes
                                      # fire and FAILED on exactly the configurations where the corpus
                                      # does BETTER — caught by running the scene-keyed arm, which is
                                      # why the verification list has one. A zero prints a NOTE rather
                                      # than a silent 0.
                                      # THE DEFECT ITSELF (upstream, not ours): check_empty_cell calls
                                      # occluded_q six times under [unroll]; each declares its own
                                      # RayQuery, but SPIR-V requires every OpVariable in a function's
                                      # FIRST block, so the six unrolled bodies share one function-scope
                                      # variable — and spirv-cross declares it inside a do{}while(false)
                                      # and references it after the block closes ("use of undeclared
                                      # identifier '_194'"). MEASURED IDENTICAL on spirv-cross 1.4.350.1
                                      # and 1.4.357.0 (this box was upgraded to match CI, which installs
                                      # unpinned — and the build.rs toolstamp correctly invalidated the
                                      # FFX metallib cache and re-transpiled all 80, with --check-fsr3's
                                      # numbers unmoved to the digit). WORKAROUND MEASURED AND NOT
                                      # SHIPPED: [loop] instead of [unroll] takes the corpus to 73/78 —
                                      # it is a codegen change to a path D3D12 and Vulkan both compile
                                      # and NEITHER is verifiable from a macOS box, for code that only
                                      # runs under a verify probe, so it belongs to C2 with those two
                                      # suites re-run.
                                      # FR_MSL_LIST=1 names every module that reached a metallib, with
                                      # its size (the FR_SPIRV_LIST idiom). Its sibling landed on the
                                      # SPIR-V side and is what made all of the above measurement rather
                                      # than argument: FR_SPIRV_DUMP=<dir> on --check-spirv keeps every
                                      # compiled module instead of deleting it — the analogue of
                                      # gpu/dxc.rs's Windows-only FR_DUMP_HLSL, and deliberately
                                      # INDEPENDENT of S3 so it works without spirv-tools installed.
                                      # CI: in the check-metal job, and it is the coverage
                                      # --check-metalfx CANNOT give — it needs the shader TOOLCHAIN and
                                      # no GPU, so the paravirtual device that makes MetalFX skip is
                                      # irrelevant to it. Guarded on the M3/M4 line as well as PASSED
                                      # (the gate SKIPs on an absent toolchain, so PASSED alone would go
                                      # green on an image that lost spirv-cross or the Metal cryptex).
                                      # Needs SDKs/dxc-macos, a ~10-min SOURCE build at DXC_TAG (upstream
                                      # publishes no macOS binary, and a community build would break the
                                      # one-release-tag invariant that keeps the DXIL and SPIR-V front
                                      # ends identical), cached on the tag.
                                      # Touch mtl/msl.rs / CROSS_ARGS / Expect / corpus_units /
                                      # corpus_jobs -> run --check-msl on the procedural scene AND
                                      # san-miguel-low-poly AND --sw-rays (the three configurations
                                      # disagree, which is the point), --check-spirv both arms (the
                                      # shared enumeration), --check-fsr3 and --check-metalfx (build.rs
                                      # shares the spirv-cross toolstamp), --check + cargo test, then
                                      # restore the Windows goldens
```
