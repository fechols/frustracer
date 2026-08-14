/// NRD is a HARD build requirement (2026-08-10): it is the default pre-upscale
/// denoiser for XeSS/FSR3 sessions, so a tree that cannot produce it is a tree
/// whose default session silently runs undenoised — the failure this gate
/// exists to make impossible. Deliberately NOT the DLSS block's
/// `cargo:warning` degrade.
///
/// TWO checks, because they fail for different reasons and want different
/// fixes: the SOURCE is the `SDKs/NRD-src` submodule (platform-neutral — a
/// fresh clone without `--recurse-submodules` has an empty directory), and the
/// ARTIFACT is the built DLL, which only `install-prerequisites.bat nrd` can
/// produce. Neither shells out to CMake: NRD's configure FetchContent's
/// ShaderMake and MathLib as URL zips, and putting the network inside every
/// `cargo build` is not a trade worth making.
///
/// The artifact is per TARGET: `NRD.dll` carrying DXIL for D3D12, `libNRD.so`
/// carrying SPIR-V for the Vulkan backend. Both arms are hard requirements —
/// the Linux one since the installer learned to build it — and both are keyed
/// on `CARGO_CFG_TARGET_OS`, never `cfg!(windows)`, which describes the HOST
/// (the defect `build_ffx_fsr3` documents below; `require_nrd` had it too, and
/// only accidentally: it made cross-compiling to Windows skip the DLL check).
///
/// BUT THE ARTIFACT HALF FIRES ONLY ON A NATIVE BUILD, and that is a statement
/// rather than an escape. The panic exists to stop a SESSION rendering
/// undenoised without saying so; `cargo check --target x86_64-pc-windows-msvc`
/// — what `tools/win-cross-check.sh` runs on a Linux box to type-check the
/// `#[cfg(windows)]` half of this tree — produces no session and cannot
/// produce an `NRD.dll` either. Keying the artifact on the target WITHOUT this
/// guard would turn today's accidental pass into a hard panic and take the one
/// tool that covers the Windows half every commit. Cross-builds get a warning.
fn require_nrd() {
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed=SDKs/NRD-src/CMakeLists.txt");
    if !manifest.join("SDKs/NRD-src/CMakeLists.txt").exists() {
        panic!(
            "\n\nNRD source is missing: SDKs/NRD-src is empty (the submodule was not \
             checked out).\n    git submodule update --init SDKs/NRD-src\n\nNRD is the \
             default denoiser and is required to build.\n"
        );
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // `HOST`/`TARGET` are triples cargo always sets for a build script.
    let native = std::env::var("HOST").ok() == std::env::var("TARGET").ok();
    let (rel, installer, deps) = match target_os.as_str() {
        "windows" => (
            "SDKs/NRD/bin/NRD.dll",
            "install-prerequisites.bat nrd",
            "CMake (3.22...3.30) + VS 2022 C++ tools",
        ),
        "linux" => (
            "SDKs/NRD/bin/libNRD.so",
            "./install-prerequisites.sh nrd",
            "CMake + a C++17 compiler",
        ),
        // macOS and anything else: no NRD consumer exists there (NRD emits
        // DXBC/DXIL/SPIR-V and has no Metal output at all), so requiring an
        // artifact nothing loads would be a gate with no subject.
        _ => return,
    };
    println!("cargo:rerun-if-changed={rel}");
    if manifest.join(rel).exists() {
        return;
    }
    if !native {
        println!(
            "cargo:warning={rel} is missing, and this is a cross-build ({} -> {}) — the \
             artifact check stands down, so this checks types only and produces nothing \
             runnable. Build it with `{installer}` on the target platform.",
            std::env::var("HOST").unwrap_or_default(),
            std::env::var("TARGET").unwrap_or_default(),
        );
        return;
    }
    panic!(
        "\n\nThe NRD library is missing: {rel} has not been built.\n    {installer}\n\n\
         Needs {deps}; NVIDIA ships no prebuilt binaries, so it compiles locally from the \
         submodule. NRD is the default denoiser, so a tree that cannot produce it is a tree \
         whose default session silently runs undenoised.\n"
    );
}

/// The FidelityFX SDK 1.1.4 core — the backend-neutral half of FSR3 for the
/// Vulkan and Metal backends. See shim/ffx_fsr3.cpp for why a second, older
/// FidelityFX generation exists beside the Windows ffx-api v2.3.0 path.
///
/// WARN AND SKIP, NEVER PANIC. quinlight-player (where this integration comes
/// from) panics when the SDK is absent; this tree must not, because a bare
/// checkout has to build — install-prerequisites.sh's own header states it
/// ("Building NEVER needs any of this"), and every gate that runs in CI runs
/// without a single vendor SDK on disk. So this follows the DLSS block's
/// degrade: one `cargo:warning` naming the missing half, no cfg, and a build
/// that simply has no FSR3. It is deliberately NOT the `require_nrd()` shape —
/// NRD hard-fails because it is the DEFAULT denoiser and a tree that cannot
/// produce it renders undenoised without saying so; FSR3 here is opt-in and its
/// absence costs a feature, not correctness.
///
/// TWO HALVES, BOTH REQUIRED, ONLY ONE FETCHABLE: the SDK source
/// (`install-prerequisites.sh fsr3src`) and the pre-built SPIR-V permutations,
/// which are COMMITTED because the SDK compiles them with a Windows-only tool.
/// They are reported separately so a missing one names the right fix — a
/// re-fetch or a re-checkout, never both.
#[cfg(not(windows))]
fn build_ffx_fsr3() {
    // THE GUARD MUST BE ON THE *TARGET*, AND `cfg!` IS NOT. Inside a build
    // script `cfg!(windows)` describes the HOST the script was compiled for, so
    // the `#[cfg(not(windows))]` on this function only says "not built ON
    // Windows" — cross-compiling TO Windows from Linux still runs it, and then
    // tries to build the FSR3 SDK with clang-cl for a platform that upscales
    // through ffx-api v2.3.0 and wants none of this.
    //
    // A latent defect since B0 that could only surface once the SDK was
    // actually fetched: with `SDKs/FidelityFX-SDK` absent, the sentinel check
    // below returned first and the cross-compile never got here.
    // `CARGO_CFG_TARGET_OS` is the target, which is the question being asked.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        return;
    }

    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let sdk = manifest.join("SDKs/FidelityFX-SDK/sdk");
    let shaders = manifest.join("SDKs/FidelityFX-SDK-prebuilt/shaders/vk");

    // The two sentinels build.rs and the installer agree on — one fact, one
    // place, so a partial install cannot read as complete on either side.
    let sdk_stamp = sdk.join("CMakeLists.txt");
    let shader_stamp = shaders.join("ffx_fsr3upscaler_accumulate_pass_permutations.h");
    println!("cargo:rerun-if-changed={}", sdk_stamp.display());
    println!("cargo:rerun-if-changed={}", shader_stamp.display());
    println!("cargo:rerun-if-changed=shim/ffx_fsr3.cpp");
    println!("cargo:rerun-if-changed=shim/ffx_fsr3.h");
    println!("cargo:rerun-if-changed=shim/ffx_msvc_compat.h");

    if !sdk_stamp.exists() {
        println!(
            "cargo:warning=FidelityFX SDK 1.1.4 source not found at {} — FSR3 disabled \
             for the Vulkan/Metal backends (./install-prerequisites.sh fsr3src)",
            sdk.display()
        );
        return;
    }
    if !shader_stamp.exists() {
        println!(
            "cargo:warning=the committed FSR3 SPIR-V permutations are missing from {} — \
             FSR3 disabled. They ship in this repository (their compiler is Windows-only); \
             re-check out that directory.",
            shaders.display()
        );
        return;
    }

    // THE VULKAN BACKEND IS ONLY BUILT WHERE ITS HEADERS ARE. `ffx_vk.cpp`
    // includes <vulkan/vulkan.h>, which comes from the distro's vulkan-headers
    // package rather than from anything this repository fetches — so it gets
    // the same warn-and-skip treatment as the two halves above rather than a
    // hard failure, and the whole FSR3 arm stands down instead of half-building.
    // (macOS will take the hand-written Metal `FfxInterface` here instead.)
    let vk_header = std::path::Path::new("/usr/include/vulkan/vulkan.h");
    let want_vk = target_os == "linux" && vk_header.exists();
    if target_os == "linux" && !want_vk {
        println!(
            "cargo:warning=<vulkan/vulkan.h> not found at {} — FSR3 disabled for the \
             Vulkan backend (install your distribution's vulkan-headers package)",
            vk_header.display()
        );
        return;
    }

    // The backend-NEUTRAL translation units. The per-backend half — `ffx_vk.cpp`
    // for Vulkan, a hand-written Metal `FfxInterface` later — lands beside them
    // below, because each needs an API's headers the other platform lacks.
    let src = sdk.join("src");
    let units = [
        "components/fsr3upscaler/ffx_fsr3upscaler.cpp",
        "shared/ffx_assert.cpp",
        "shared/ffx_message.cpp",
        "shared/ffx_breadcrumbs_list.cpp",
        "shared/ffx_object_management.cpp",
        "backends/shared/ffx_shader_blobs.cpp",
        "backends/shared/blob_accessors/ffx_fsr3upscaler_shaderblobs.cpp",
    ];

    let mut b = cc::Build::new();
    b.cpp(true)
        .std("c++17")
        // Third-party source we do not edit, so its warnings are noise we
        // cannot act on — and `cc` surfaces them as `cargo:warning`, i.e. on
        // EVERY build, where they would bury the two warnings this function
        // actually wants read (the missing-half degrades above). The SDK
        // carries MSVC `#pragma warning` directives clang does not know and
        // aggregate initializers that trip -Wextra; neither is a defect.
        .warnings(false)
        // Force-included rather than patched into the sources: the SDK is
        // FETCHED, so a local patch would silently un-apply on the next fetch.
        .flag("-include")
        .flag(manifest.join("shim/ffx_msvc_compat.h").to_str().unwrap())
        .define("FFX_FSR3UPSCALER", None)
        .include(sdk.join("include"))
        .include(&src)
        // The SDK's own sources include each other by bare name from these
        // three directories, and the blob accessor includes the permutation
        // headers by bare name too — which is what points the last include at
        // the COMMITTED shaders rather than anywhere under the fetched tree.
        .include(src.join("shared"))
        .include(src.join("backends/shared"))
        .include(src.join("components"))
        .include(&shaders);
    for u in units {
        b.file(src.join(u));
    }
    b.file(manifest.join("shim/ffx_fsr3.cpp"));

    // THE METAL HALF IS DECIDED BEFORE THE COMPILE, because the shaders and the
    // backend are two artifacts that are only useful TOGETHER. The transpile is
    // independent of `cc` and could run anywhere, but a `ffx_metal` backend with
    // no metallib table can do exactly one thing — fail at the first
    // fpCreatePipeline — so compiling it without one would ship a linked,
    // reachable, guaranteed-to-fail arm. Deciding both off one boolean is what
    // makes `cfg(ffx_fsr3_metal)` mean "the Metal FSR3 arm is fully built"
    // rather than "half of it is", which is the distinction the `ffx_fsr3_vk`
    // repair was about.
    //
    // Gated on the TARGET, not the host — `cfg(not(windows))` above is a HOST
    // test (the M0 link-flag fix records that trap). A Linux host
    // cross-compiling to macOS still gets the metallibs; it will NOT get the
    // .mm, since neither Metal.framework nor an ObjC++ runtime is there to
    // build it against, and `generate_fsr3_metallibs` needs Xcode anyway — so
    // that configuration lands on the same warn-and-skip degrade as a Mac
    // without spirv-cross.
    let want_metal = target_os == "macos" && generate_fsr3_metallibs(&shaders) > 0;

    if want_vk {
        // `ffx_vk.cpp`'s own sources include each other by bare name from this
        // directory, exactly as the neutral units do from theirs.
        b.include(src.join("backends/vk"));
        b.file(src.join("backends/vk/ffx_vk.cpp"));
        b.file(manifest.join("shim/ffx_fsr3_vk.cpp"));

        // AN ALIGNMENT UB IN THE SDK, and the one thing here that is a crash
        // rather than a compile error. `CreateBackendContextVK` zeroes a run of
        // fields in an `alignas(32)` EffectContext carved out of a 16-byte-aligned
        // scratch buffer; at -O2+ the compiler folds those writes into an aligned
        // 128-bit store on a misaligned address and takes a #GP. Disabling
        // SLP/loop vectorization keeps them scalar. Perf impact is nil — these are
        // init and record paths, not arithmetic kernels — and `flag_if_supported`
        // is what keeps clang and MSVC out of it, so no compiler test is needed.
        // Do not remove either flag without reproducing the segfault first.
        b.flag_if_supported("-fno-tree-slp-vectorize");
        b.flag_if_supported("-fno-tree-vectorize");
    }

    if want_metal {
        // clang derives Objective-C++ from the `.mm` extension, and `cc` is
        // already driving the C++ compiler here, so no `-x` and no extra
        // standard flag are needed. ARC is OPT-IN (`-fobjc-arc`), which is why
        // not passing it is what gives the manual reference counting the file's
        // header depends on — every `alloc`/`new*` in there is a +1 some line
        // must release.
        b.file(manifest.join("shim/ffx_fsr3_metal.mm"));
    }

    b.compile("ffx_fsr3");

    if want_vk {
        // THE TREE'S FIRST *LINKED* VULKAN DEPENDENCY, and it is worth saying so:
        // `src/vk/` deliberately links nothing (`libvulkan.so.1` is dlopen'd
        // through `ash` and every entry point resolved by symbol, which is what
        // keeps every `--check*` DLL-free). `ffx_vk.cpp` resolves MOST of Vulkan
        // through the `vkGetDeviceProcAddr` we hand it, but not all of it —
        // measured, `nm` on the object leaves nine undefined:
        //
        //   vkCreateBuffer                        vkGetPhysicalDeviceFeatures
        //   vkEnumerateDeviceExtensionProperties  vkGetPhysicalDeviceFeatures2
        //   vkGetDeviceProcAddr                   vkGetPhysicalDeviceMemoryProperties
        //   vkGetPhysicalDeviceProperties         vkGetPhysicalDeviceProperties2
        //   (+ ffxSetFrameGenerationConfigToSwapchainVK, stubbed in our shim)
        //
        // so the loader is needed at LINK time. Scoped to `cfg(ffx_fsr3_src)` —
        // a bare checkout, which is what that policy actually protects, still
        // links nothing.
        //
        // DO NOT "CLEAN THIS UP" ON THE EVIDENCE OF `ldd`: until something in
        // Rust actually calls `frshim_fsr3vk_*`, `--gc-sections` drops these
        // objects and `--as-needed` then drops libvulkan from DT_NEEDED, so the
        // link looks superfluous while being load-bearing the moment the shim
        // gains its first caller.
        println!("cargo:rustc-link-lib=dylib=vulkan");
        println!("cargo:rerun-if-changed=shim/ffx_fsr3_vk.cpp");
        println!("cargo:rerun-if-changed=shim/ffx_fsr3_vk.h");

        // ONE CFG PER ARTIFACT. `ffx_fsr3_src` says the backend-NEUTRAL units
        // built; it does NOT say `ffx_vk.cpp` and our shim did, and on macOS
        // they provably did not — `want_vk` requires Linux, because that
        // platform takes the Metal `FfxInterface` instead. `src/vk/fsr3.rs`
        // declared its `frshim_fsr3vk_*` externs under the broader cfg, so any
        // box with the FFX SDK source and no `want_vk` — every Mac — failed to
        // LINK with three undefined symbols, while build.rs's own warning
        // promised the arm had "stood down". The Linux-without-headers case was
        // fine only by accident: its early `return` also skips `ffx_fsr3_src`.
        println!("cargo:rustc-cfg=ffx_fsr3_vk");
    }

    println!("cargo:rustc-cfg=ffx_fsr3_src");

    if want_metal {
        // The `ffx_metal` backend calls Metal DIRECTLY — unlike `ffx_vk`, which
        // resolves most of its API through a caller-supplied
        // vkGetDeviceProcAddr — so the frameworks are a link-time dependency
        // with no dlopen equivalent. `src/mtl/` reaches Metal through `objc2`,
        // which links the same frameworks, so this adds no NEW dependency to a
        // macOS build; it is what the shim's own objects need.
        //
        // Scoped to `want_metal` because that is when the SHIM's objects need
        // them — and note what this scoping does NOT buy, since the comment
        // here used to claim it and was wrong. A bare checkout links Metal and
        // Foundation anyway: `mod mtl` is unconditional on macOS,
        // `objc2-metal` carries `#[link(name = "Metal", kind = "framework")]`
        // in its own generated root and `objc2` does the same for Foundation,
        // so both are in the binary's load commands with or without these two
        // lines. (`objc2-metal-fx` adds MetalFX on the same terms — see
        // Cargo.toml, where the macOS floor that implies is spelled out.)
        // These lines are load-bearing for `shim/ffx_fsr3_metal.mm`, which
        // calls Metal directly from C++, and decorative for the Rust side.
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rerun-if-changed=shim/ffx_fsr3_metal.mm");
        println!("cargo:rerun-if-changed=shim/ffx_fsr3_metal.h");

        // ONE CFG PER ARTIFACT — see `want_metal`'s declaration for why the
        // metallib table and the backend are ONE artifact and not two.
        println!("cargo:rustc-cfg=ffx_fsr3_metal");
    }
}

/// FSR3-on-Metal shaders: transpile the committed FidelityFX SPIR-V permutations
/// to `.metallib` for the hand-written `ffx_metal` backend. FidelityFX ships only
/// `ffx_vk`/`ffx_dx12`, so Metal reuses the very blobs the Vulkan shaderblob
/// accessor returns — SPIR-V is the interchange format, and no Metal shader
/// artifact is committed.
///
/// ONLY THE NON-WAVE64 PERMUTATIONS. Apple GPUs are SIMD-32, so `ffx_metal`
/// reports `waveLaneCountMax = 32` and FFX never requests the wave64 blobs (which
/// would also mis-execute at width 32). Both fp32 and fp16 (`_16bit`) are kept
/// because a pass picks between them at runtime.
///
/// Expect 80 of the 200 committed files, and the two skipped sets OVERLAP — a
/// plain subtraction gets this wrong: 40 are permutation INDEX headers (consumed
/// by C++, not shader blobs) of which 20 are themselves wave64, leaving 160
/// shader blobs of which 80 are wave64. 200 - 40 - 80 = 80.
///
/// Each metallib is content-addressed by FNV-1a-64 of its SPIR-V bytes; at
/// pipeline-create time the backend hashes `FfxShaderBlob.data` — the same bytes,
/// from the same accessor — and loads the match. Emitted into `$OUT_DIR`:
///   - `ffx_fsr3/<hash>.metallib` — `[12-byte LE threadgroup header][metallib]`
///   - `ffx_fsr3_metallibs.rs`    — the `&[(u64, &[u8])]` table the Rust side `include!`s
///
/// THE DENOMINATOR IS EMITTED, not just warned about. A partial transpile —
/// 79 of 80 — is otherwise invisible until a specific FSR3 pass asks for the one
/// missing hash at pipeline-create time, which is a rare, scene-dependent runtime
/// failure. `FFX_FSR3_PERMUTATIONS_FOUND` lets `--check-fsr3` compare enumerated
/// against emitted and fail at gate time instead.
///
/// BEST-EFFORT, like the two degrades above: a failed permutation is counted and
/// skipped, and an absent `spirv-cross` or Xcode leaves the table empty, which
/// the Rust side reads as "FSR3-Metal unavailable" rather than a link error. The
/// failures are reported as ONE aggregated line naming the first — 80
/// `cargo:warning`s would bury the two degrade warnings above, which this
/// function's own header protects. Returns the number emitted.
#[cfg(not(windows))]
fn generate_fsr3_metallibs(prebuilt: &std::path::Path) -> usize {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let lib_dir = out_dir.join("ffx_fsr3");

    // CONTENT-ADDRESSED CACHE, KEYED ON THE TOOLS AND THE RECIPE AS WELL AS THE
    // INPUT. The file name is FNV-1a of the SPIR-V, so an existing
    // `<hash>.metallib` is current with respect to its INPUT by construction —
    // but not with respect to anything else that shapes the output, and there
    // are two such things. spirv-cross, whose MSL emission is exactly what a
    // Homebrew bump changes while every key stays identical; and OUR OWN
    // recipe, which lives in this file. So both ride the stamp and a change
    // wipes the directory rather than serving pre-change metallibs forever.
    // `SPIRV_CROSS_ARGS` is on it verbatim, so editing the flag list
    // invalidates BY CONSTRUCTION; `RECIPE` is the hand-bumped half, named at
    // its declaration beside the two rules it covers. Without the cache every
    // build re-runs 240 subprocesses (measured ~55 s).
    let stamp_want = format!(
        "v2 {} {RECIPE}\n{}\n{}\n",
        SPIRV_CROSS_ARGS.join(" "),
        tool_version(&mut std::process::Command::new(
            std::env::var("SPIRV_CROSS").unwrap_or_else(|_| "spirv-cross".into())
        )),
        tool_version(&mut xcrun_tool("metal"))
    );
    let stamp_path = lib_dir.join(".toolstamp");
    if std::fs::read_to_string(&stamp_path).ok().as_deref() != Some(stamp_want.as_str()) {
        let _ = std::fs::remove_dir_all(&lib_dir);
    }
    if let Err(e) = std::fs::create_dir_all(&lib_dir) {
        println!("cargo:warning=FSR3-Metal: cannot create {}: {e}", lib_dir.display());
        return 0;
    }
    let entries = match std::fs::read_dir(prebuilt) {
        Ok(rd) => rd,
        Err(e) => {
            println!("cargo:warning=FSR3-Metal: read_dir {}: {e}", prebuilt.display());
            return 0;
        }
    };

    let mut emitted: Vec<u64> = Vec::new();
    let (mut found, mut cached) = (0usize, 0usize);
    let mut failures: Vec<String> = Vec::new();
    for ent in entries.flatten() {
        let path = ent.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()).map(str::to_owned) else {
            continue;
        };
        if !name.ends_with(".h") || name.ends_with("_permutations.h") || name.contains("wave64") {
            continue;
        }
        // Counted BEFORE the extraction, so the denominator is what was
        // ENUMERATED rather than what parsed. Counting after made a header that
        // lost its array shrink `found` alongside `emitted`, printing the
        // self-contradictory `79/79 ... (1 failed)` — and, worse, hiding the
        // loss from the gate's `libs.len() != found` bound, leaving only the
        // `EXPECTED_PERMUTATIONS` one to catch it.
        found += 1;
        let spirv = match extract_spirv_from_header(&path) {
            Some(b) if !b.is_empty() => b,
            _ => {
                failures.push(format!("{name}: no SPIR-V _data[] array"));
                continue;
            }
        };
        let hash = fnv1a64(&spirv);
        let stem = lib_dir.join(format!("{hash:016x}"));
        let lib = stem.with_extension("metallib");
        // THE CACHE HIT VALIDATES THE SHAPE, not just the size. A bare
        // `len() > 12` accepts a file left behind by a `metallib` that failed
        // or was interrupted — truncated, or (the likelier window) complete but
        // written before the 12-byte prefix landed, since that prefix is a
        // second write. Such a file is >12 bytes forever, so it would be served
        // as cached on every subsequent build and its container magic would be
        // read as a threadgroup size. `table_self_test` does catch it, but at
        // gate time and with no hint that the remedy is to delete OUT_DIR.
        // Four bytes read here instead. (`transpile_ffx_metallib` additionally
        // writes the combined file atomically now, so the window is closed at
        // both ends.)
        if std::fs::read(&lib)
            .map(|b| b.len() > 16 && &b[12..16] == b"MTLB")
            .unwrap_or(false)
        {
            emitted.push(hash);
            cached += 1;
            continue;
        }
        let spv = format!("{}.spv", stem.display());
        if let Err(e) = std::fs::write(&spv, &spirv) {
            failures.push(format!("{name}: write {spv}: {e}"));
            continue;
        }
        match transpile_ffx_metallib(&spv, &stem.display().to_string(), &spirv) {
            Ok(()) => emitted.push(hash),
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    let _ = std::fs::write(&stamp_path, &stamp_want);

    // Sorted, because read_dir is unordered and the table is compiled into the
    // binary — an unstable order would make two builds of one tree differ. The
    // Rust side additionally gates strict-ascending, which is what makes the
    // C side's linear scan for an equal hash unambiguous.
    emitted.sort_unstable();
    let mut rs = String::with_capacity(256 + emitted.len() * 96);
    rs.push_str("// @generated by build.rs::generate_fsr3_metallibs — do not edit.\n");
    rs.push_str("// FidelityFX FSR3 SPIR-V permutations transpiled to Metal, keyed by\n");
    rs.push_str("// FNV-1a-64 of the SPIR-V bytes. shim/ffx_fsr3_metal.mm hashes\n");
    rs.push_str("// FfxShaderBlob.data with the byte-identical hash to find its metallib.\n");
    rs.push_str("pub static FFX_FSR3_METALLIBS: &[(u64, &[u8])] = &[\n");
    for hash in &emitted {
        // include_bytes! resolves relative to this generated file, i.e. OUT_DIR.
        rs.push_str(&format!(
            "    (0x{hash:016x}u64, include_bytes!(\"ffx_fsr3/{hash:016x}.metallib\")),\n"
        ));
    }
    rs.push_str("];\n\n");
    rs.push_str("/// Non-wave64, non-index permutation headers ENUMERATED — the denominator.\n");
    rs.push_str("/// A gap between this and the table's length is a partial transpile.\n");
    rs.push_str(&format!("pub const FFX_FSR3_PERMUTATIONS_FOUND: usize = {found};\n"));
    let rs_path = out_dir.join("ffx_fsr3_metallibs.rs");
    if let Err(e) = std::fs::write(&rs_path, rs) {
        println!("cargo:warning=FSR3-Metal: write {}: {e}", rs_path.display());
        return 0;
    }

    // The tool version rides the success line deliberately: a spirv-cross bump
    // that changes MSL emission is the most likely future breakage here, and
    // this is the only forensic trail a past build leaves.
    println!(
        "cargo:warning=FSR3-Metal: {}/{found} non-wave64 permutations -> metallib \
         ({cached} cached, {} failed) [{}]",
        emitted.len(),
        failures.len(),
        stamp_want.lines().nth(1).unwrap_or("?")
    );
    if let Some(first) = failures.first() {
        println!(
            "cargo:warning=FSR3-Metal: {} permutation(s) failed, first: {first}",
            failures.len()
        );
    }
    emitted.len()
}

/// One line of a tool's `--version`, for the cache stamp. Absent tool =>
/// "missing", which still makes a usable stamp (and the transpile then fails
/// loudly per permutation rather than here).
#[cfg(not(windows))]
fn tool_version(cmd: &mut std::process::Command) -> String {
    match cmd.arg("--version").output() {
        Ok(o) => {
            let s = String::from_utf8_lossy(if o.stdout.is_empty() { &o.stderr } else { &o.stdout });
            s.lines().next().unwrap_or("?").trim().to_string()
        }
        Err(_) => "missing".to_string(),
    }
}

/// Pull the SPIR-V module out of a FidelityFX_SC-generated header — the
/// `static const unsigned char g_..._data[] = { 0x.., ... };` array. One array
/// per header (verified across all 200).
#[cfg(not(windows))]
fn extract_spirv_from_header(path: &std::path::Path) -> Option<Vec<u8>> {
    let txt = std::fs::read_to_string(path).ok()?;
    let key = txt.find("_data[]")?;
    let open = key + txt[key..].find('{')?;
    let close = open + txt[open..].find('}')?;
    let mut out = Vec::with_capacity((close - open) / 5);
    for tok in txt[open + 1..close].split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        let hex = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X"))?;
        out.push(u8::from_str_radix(hex, 16).ok()?);
    }
    Some(out)
}

/// 64-bit FNV-1a. `shim/ffx_fsr3_metal.mm` implements the byte-identical hash — it is
/// the ONLY thing the two sides agree on to find a permutation's metallib, so
/// neither may be "cleaned up" without the other.
#[cfg(not(windows))]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Recover a compute kernel's workgroup size from `OpExecutionMode <entry>
/// LocalSize x y z`. Metal needs it HOST-side at dispatch (Vulkan and DXIL both
/// reflect it out of the bytecode), so it is prepended to the metallib rather
/// than looked up later.
///
/// `None` ON A PARSE MISS, DELIBERATELY, rather than a `[1,1,1]` fallback: that
/// value is indistinguishable from a real 1x1x1 kernel — nonzero, product under
/// Metal's 1024 ceiling, so the Rust-side header gate accepts it — and the
/// dispatch would then run one thread per threadgroup, which is correct-looking
/// output at ~1/64 the rate with no error anywhere. Every FFX compute kernel
/// has a real group size (MEASURED over this corpus: 74x (8,8,1), 4x (256,1,1),
/// 2x (64,1,1), and no [1,1,1] at all), so a miss is always a defect and is
/// worth failing the permutation for. The realistic trigger is a blob using
/// `OpExecutionModeId LocalSizeId`, which this does not decode.
///
/// TWIN: `src/spirv.rs::local_size` is the same walk for OUR corpus, which
/// needs it for the same reason (Metal takes the group shape from the host).
/// It is not shared because a build script is its own compilation unit and
/// cannot `use crate::spirv` — the `fnv1a64` situation exactly, and handled the
/// same way. Change one, change both. The differences are deliberate and local:
/// this side takes BYTES and tolerates a big-endian module because it reads
/// whatever `extract_spirv_from_header` produced, while that side takes WORDS
/// from `to_words` (which has already rejected a byte-swapped blob) and refuses
/// disagreeing entry points, which cannot arise in FFX's one-entry permutations.
#[cfg(not(windows))]
fn spirv_local_size(spv: &[u8]) -> Option<[u32; 3]> {
    const MAGIC: u32 = 0x0723_0203;
    const OP_EXECUTION_MODE: u16 = 16;
    const EXEC_MODE_LOCAL_SIZE: u32 = 17;
    if spv.len() < 20 || spv.len() % 4 != 0 {
        return None;
    }
    let le = u32::from_le_bytes([spv[0], spv[1], spv[2], spv[3]]) == MAGIC;
    let word = |i: usize| -> u32 {
        let b = [spv[i * 4], spv[i * 4 + 1], spv[i * 4 + 2], spv[i * 4 + 3]];
        if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) }
    };
    let n = spv.len() / 4;
    let mut i = 5; // past the 5-word header
    while i < n {
        let op = word(i);
        let count = (op >> 16) as usize;
        if count == 0 || i + count > n {
            break;
        }
        if (op & 0xffff) as u16 == OP_EXECUTION_MODE
            && count >= 6
            && word(i + 2) == EXEC_MODE_LOCAL_SIZE
        {
            return Some([word(i + 3), word(i + 4), word(i + 5)]);
        }
        i += count;
    }
    None
}

/// The spirv-cross invocation, in ONE place because it rides the cache stamp
/// verbatim — editing it invalidates every cached metallib by construction,
/// which is not true of anything expressed only at the call site.
///
/// `--msl-decoration-binding` is LOAD-BEARING: it makes the Metal argument index
/// equal FFX's own VK binding number, which is what lets `ffx_metal` bind
/// discretely (`setTexture:atIndex:` straight from the shaderblob reflection
/// tables) instead of building descriptor sets. No push-constant relocation —
/// FFX uses a real constant buffer, never `[[buffer(0)]]`.
#[cfg(not(windows))]
const SPIRV_CROSS_ARGS: &[&str] = &["--msl", "--msl-version", "30000", "--msl-decoration-binding"];

/// The half of the recipe that is not a flag list, and therefore has to be
/// bumped BY HAND when either of the two rules it names changes:
///   - the 12-byte little-endian threadgroup prefix on each metallib, whose
///     format `shim/ffx_fsr3_metal.mm` strips back off;
///   - the `binding - 1000` sampler remap in `remap_ffx_samplers`.
/// Both change the emitted bytes while leaving every cache key identical, so
/// without this the cache would serve pre-change output indefinitely.
#[cfg(not(windows))]
const RECIPE: &str = "hdr=le-u32x3 sampler=-1000";

/// SPIR-V -> MSL -> AIR -> metallib for one FFX permutation.
#[cfg(not(windows))]
fn transpile_ffx_metallib(spv: &str, stem: &str, spv_bytes: &[u8]) -> Result<(), String> {
    let spirv_cross =
        std::env::var("SPIRV_CROSS").unwrap_or_else(|_| "spirv-cross".to_string());
    let msl = format!("{stem}.metal");
    let air = format!("{stem}.air");
    let lib = format!("{stem}.metallib");

    // Parsed FIRST, so a blob we cannot dispatch fails before three subprocesses
    // are spent compiling it.
    let [lx, ly, lz] = spirv_local_size(spv_bytes)
        .ok_or("no OpExecutionMode LocalSize (a threadgroup size we cannot dispatch)")?;

    let st = std::process::Command::new(&spirv_cross)
        .args(SPIRV_CROSS_ARGS)
        .arg("--output")
        .arg(&msl)
        .arg(spv)
        .status()
        .map_err(|e| format!("launch spirv-cross (`{spirv_cross}`): {e}"))?;
    if !st.success() {
        return Err(format!("spirv-cross failed ({st})"));
    }

    remap_ffx_samplers(std::path::Path::new(&msl))?;

    for (tool, args) in [("metal", vec!["-c", &msl, "-o", &air]), ("metallib", vec![&air, "-o", &lib])]
    {
        let mut c = xcrun_tool(tool);
        let st = c
            .args(&args)
            .status()
            .map_err(|e| format!("launch {tool}: {e}"))?;
        if !st.success() {
            return Err(format!("{tool} failed ({st})"));
        }
    }

    // Prepend the workgroup size. `ffx_metal` strips these 12 bytes back off.
    //
    // WRITTEN VIA A TEMP AND RENAMED, because `<hash>.metallib` existing IS the
    // cache key: an in-place rewrite leaves a window in which the file is a
    // complete but header-LESS metallib, and a build interrupted there would
    // hand every later build a poisoned entry. `rename` within one directory is
    // atomic on every filesystem this runs on, so the name only ever appears
    // carrying both halves.
    let lib_bytes = std::fs::read(&lib).map_err(|e| format!("read {lib}: {e}"))?;
    let mut combined = Vec::with_capacity(12 + lib_bytes.len());
    for v in [lx, ly, lz] {
        combined.extend_from_slice(&v.to_le_bytes());
    }
    combined.extend_from_slice(&lib_bytes);
    let tmp = format!("{lib}.tmp");
    std::fs::write(&tmp, &combined).map_err(|e| format!("write {tmp}: {e}"))?;
    std::fs::rename(&tmp, &lib).map_err(|e| format!("rename {tmp} -> {lib}: {e}"))
}

/// `$METAL` / `$METALLIB` (whitespace-split, overrides the whole command) else
/// `xcrun -sdk macosx <tool>` — the active-Xcode default.
#[cfg(not(windows))]
fn xcrun_tool(tool: &str) -> std::process::Command {
    if let Ok(p) = std::env::var(tool.to_uppercase()) {
        let mut parts = p.split_whitespace();
        if let Some(prog) = parts.next() {
            let mut c = std::process::Command::new(prog);
            c.args(parts);
            return c;
        }
    }
    let mut c = std::process::Command::new("xcrun");
    c.args(["-sdk", "macosx", tool]);
    c
}

/// Rewrite FFX's static-sampler indices into Metal's valid 0..15 range.
///
/// FFX binds its immutable samplers at VK binding 1000+ (`s_LinearClamp` = 1001,
/// and MEASURED across this corpus that is the only one used at all); spirv-cross
/// faithfully emits `[[sampler(1001)]]`, which `metal` rejects with "must be
/// between 0 and 15". Without this, 112 of 160 permutations fail to compile and
/// every one of them fails on exactly that line.
///
/// `ffx_metal` applies the identical `binding - 1000` rule when it binds the
/// sampler — THE TWO SIDES MUST AGREE, so neither may be changed alone. Only
/// `[[sampler(N)]]` with `N >= 1000` is touched; texture and buffer indices are
/// left exactly as `--msl-decoration-binding` emitted them.
#[cfg(not(windows))]
fn remap_ffx_samplers(msl: &std::path::Path) -> Result<(), String> {
    let src = std::fs::read_to_string(msl).map_err(|e| format!("read {}: {e}", msl.display()))?;
    const NEEDLE: &str = "[[sampler(";
    let mut out = String::with_capacity(src.len());
    let (mut last, mut from) = (0usize, 0usize);
    while let Some(rel) = src[from..].find(NEEDLE) {
        let num_start = from + rel + NEEDLE.len();
        let Some(p) = src[num_start..].find(')') else { break };
        let num_end = num_start + p;
        if let Ok(n) = src[num_start..num_end].trim().parse::<u32>() {
            if n >= 1000 {
                out.push_str(&src[last..num_start]);
                out.push_str(&(n - 1000).to_string());
                last = num_end;
            }
        }
        from = num_end;
    }
    out.push_str(&src[last..]);
    std::fs::write(msl, out).map_err(|e| format!("write {}: {e}", msl.display()))
}

fn main() {
    require_nrd();
    // Declared on every platform, not just where it can be set: Windows never
    // builds this (it upscales through ffx-api v2.3.0), and an undeclared cfg
    // is a warning wherever the attribute is written.
    println!("cargo:rustc-check-cfg=cfg(ffx_fsr3_src)");
    // The Metal arm of the same feature: set when at least one SPIR-V
    // permutation reached `.metallib`. (The ObjC++ `ffx_metal` backend joins
    // this condition when it exists; today the cfg gates only the transpiled
    // shader table.) Declared everywhere for the same reason as the line above.
    println!("cargo:rustc-check-cfg=cfg(ffx_fsr3_metal)");
    // The Vulkan arm of the same feature: set when `ffx_vk.cpp` and
    // shim/ffx_fsr3_vk.cpp actually compiled, which needs the distro's
    // vulkan-headers and is Linux-only. Distinct from `ffx_fsr3_src` because
    // the neutral units build on macOS too — see the note at the cfg's site.
    println!("cargo:rustc-check-cfg=cfg(ffx_fsr3_vk)");
    #[cfg(windows)]
    {
        // FidelityFX FFI shim: the ffx-api structs (pNext chains,
        // FfxApiResource descriptions) are built inside one C++ TU against
        // the vendored MIT headers; the loader DLL is LoadLibraryExW'd at
        // runtime, so nothing links FFX and headless `--check*` runs never
        // touch it. ffx_shim.cpp reaches the vendored headers by relative
        // path, so no .include() is needed beyond rerun tracking.
        println!("cargo:rerun-if-changed=shim/ffx_shim.cpp");
        println!("cargo:rerun-if-changed=shim/ffx_shim.h");
        println!("cargo:rerun-if-changed=SDKs/fidelityfx-sdk");
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .flag("/EHsc")
            .file("shim/ffx_shim.cpp")
            .compile("ffx_shim");

        // Raw-NGX DLSS shim family (frame generation + ray reconstruction) —
        // BUILD-OPTIONAL: the NDA-tier DLSS SDK is not redistributable and
        // never committed, so these compile only when FRUSTRACER_DLSS_SDK
        // (default: the sibling quinlight-player's vendored copy) points at
        // one. Without it the build proceeds and the Rust side stubs BOTH
        // features to "unavailable" — no DLSS at all, the chain falls to
        // FSR4/XeSS/FSR3. One SDK provides both features, one cfg gates both
        // (dlss_ngx). Links nvsdk_ngx_d.lib (the /MD import stub bound to the
        // driver's _nvngx.dll — no CRT conflict; the static _s variants are
        // /MT) and stages nvngx_dlssg.dll + nvngx_dlssd.dll next to the
        // binary (NGX resolves feature snippets from the exe directory).
        println!("cargo:rustc-check-cfg=cfg(dlss_ngx)");
        println!("cargo:rerun-if-changed=shim/ngx_shared.cpp");
        println!("cargo:rerun-if-changed=shim/ngx_shared.h");
        println!("cargo:rerun-if-changed=shim/dlssg_shim.cpp");
        println!("cargo:rerun-if-changed=shim/dlssg_shim.h");
        println!("cargo:rerun-if-changed=shim/dlssd_shim.cpp");
        println!("cargo:rerun-if-changed=shim/dlssd_shim.h");
        println!("cargo:rerun-if-env-changed=FRUSTRACER_DLSS_SDK");
        let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        let sdk = std::env::var("FRUSTRACER_DLSS_SDK")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| manifest.join(r"..\quinlight-player\SDKs\DLSS-SDK"));
        let dlssg_header = sdk.join(r"include\nvsdk_ngx_helpers_dlssg.h");
        let dlssd_header = sdk.join(r"include\nvsdk_ngx_helpers_dlssd.h");
        let import_lib_dir = sdk.join(r"lib\Windows_x86_64\x64");
        if dlssg_header.exists()
            && dlssd_header.exists()
            && import_lib_dir.join("nvsdk_ngx_d.lib").exists()
        {
            cc::Build::new()
                .cpp(true)
                .std("c++17")
                .flag("/EHsc")
                .include(sdk.join("include"))
                .file("shim/ngx_shared.cpp")
                .file("shim/dlssg_shim.cpp")
                .file("shim/dlssd_shim.cpp")
                .compile("dlss_shim");
            println!("cargo:rustc-link-search=native={}", import_lib_dir.display());
            println!("cargo:rustc-link-lib=static=nvsdk_ngx_d");
            // NGX's import-side helpers touch registry + UI APIs.
            println!("cargo:rustc-link-lib=dylib=advapi32");
            println!("cargo:rustc-link-lib=dylib=user32");
            println!("cargo:rustc-cfg=dlss_ngx");
            // Stage the snippet DLLs next to the binary for every profile dir
            // this build might land in (target/<profile>/).
            if let Ok(out) = std::env::var("OUT_DIR") {
                // OUT_DIR = target/<profile>/build/<pkg>-<hash>/out
                let profile_dir = std::path::PathBuf::from(out)
                    .ancestors()
                    .nth(3)
                    .map(|p| p.to_path_buf());
                if let Some(dir) = profile_dir {
                    for dll in ["nvngx_dlssg.dll", "nvngx_dlssd.dll"] {
                        let src = sdk.join(r"lib\Windows_x86_64\rel").join(dll);
                        if src.exists() {
                            let _ = std::fs::copy(&src, dir.join(dll));
                        }
                    }
                }
            }
        } else {
            // Say WHICH file is missing: "not found" for an SDK that exists but
            // predates the DLSSD headers sends someone checking the path
            // instead of the SDK version (and that shape disables FG too).
            let why = if !sdk.exists() {
                format!("DLSS SDK not found at {}", sdk.display())
            } else {
                let missing = [&dlssg_header, &dlssd_header, &import_lib_dir.join("nvsdk_ngx_d.lib")]
                    .into_iter()
                    .find(|p| !p.exists())
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();
                format!(
                    "DLSS SDK at {} is missing {} (an SDK predating the DLSSD \
                     ray-reconstruction headers? both features gate on one SDK)",
                    sdk.display(),
                    missing
                )
            };
            println!(
                "cargo:warning={why} — raw-NGX DLSS (ray reconstruction + frame \
                 generation) disabled (set FRUSTRACER_DLSS_SDK to enable)"
            );
        }
    }
    #[cfg(not(windows))]
    {
        println!("cargo:rustc-check-cfg=cfg(dlss_ngx)");
        build_ffx_fsr3();

        // `intel_tex_2` (the ispc BC7 encoder — the `--bc7-cpu` A/B arm) ships
        // PRECOMPILED objects whose C++ exception tables reference
        // `__gxx_personality_v0`, and its build script never asks for the C++
        // runtime that defines it. On Windows the MSVC CRT supplies the
        // equivalent implicitly, which is why this has never been visible.
        //
        // `cargo build --release` links anyway — `--gc-sections` collects the
        // `DW.ref` away once the encoder is inlined out — so the failure is
        // DEBUG-ONLY, which is exactly why it went unnoticed: M0's Linux
        // verification ran `--check` from a release build. `cargo test` builds
        // at the test profile, so without this the shader-source gates could
        // not link on Linux no matter which module they lived in.
        //
        // THE RUNTIME IS NAMED PER PLATFORM, and on macOS the difference is
        // not cosmetic: there is no libstdc++ in the SDK at all (only
        // `libc++.tbd` / `libc++abi.tbd`), so `-lstdc++` fails at LIBRARY
        // RESOLUTION — before any dead-stripping can collect the reference
        // away. That makes it fatal in EVERY profile here, release included,
        // and it is the whole reason the tree did not build on macOS. The
        // `intel_tex_2` archives shipped for `aarch64-apple-darwin` are clang
        // builds, so the personality symbol they want comes from libc++abi,
        // which `-lc++` pulls in.
        //
        // NOTE the mixed predicates, which is pre-existing and only safe
        // because this tree is built natively: the enclosing `cfg(not(windows))`
        // is a HOST test (a build script compiles for the host) while
        // CARGO_CFG_TARGET_OS names the TARGET, and a link-lib directive is a
        // property of the target. They agree for every supported build. If
        // cross-compilation is ever wanted, the whole of `main` wants moving
        // onto CARGO_CFG_TARGET_OS, not just this line.
        let cxx = if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
            "c++"
        } else {
            "stdc++"
        };
        println!("cargo:rustc-link-lib=dylib={cxx}");
    }
}
