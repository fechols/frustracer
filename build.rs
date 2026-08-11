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
/// The artifact half is Windows-only today because the artifact itself is —
/// a Linux build emits libNRD.so carrying SPIR-V, which is right for a Vulkan
/// backend and unloadable by the D3D12 sessions this tree currently renders
/// with. When the Vulkan backend lands, add its arm here beside the DLL.
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
    #[cfg(windows)]
    {
        println!("cargo:rerun-if-changed=SDKs/NRD/bin/NRD.dll");
        if !manifest.join("SDKs/NRD/bin/NRD.dll").exists() {
            panic!(
                "\n\nNRD.dll is missing: SDKs\\NRD\\bin\\NRD.dll has not been built.\n    \
                 install-prerequisites.bat nrd\n\nNeeds CMake (3.22...3.30) + VS 2022 C++ \
                 tools; NVIDIA ships no prebuilt binaries, so it compiles locally from the \
                 submodule.\n"
            );
        }
    }
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

    // The backend-NEUTRAL translation units only. `ffx_vk.cpp` (Vulkan) and the
    // hand-written Metal `FfxInterface` land beside these when their backends
    // do, because each needs an API's headers that the other platform lacks —
    // and ffx_vk.cpp additionally needs GCC's vectorizer disabled (an alignment
    // UB in CreateBackendContextVK that miscompiles into a segfault at -O2+).
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
    b.compile("ffx_fsr3");

    println!("cargo:rustc-cfg=ffx_fsr3_src");
}

fn main() {
    require_nrd();
    // Declared on every platform, not just where it can be set: Windows never
    // builds this (it upscales through ffx-api v2.3.0), and an undeclared cfg
    // is a warning wherever the attribute is written.
    println!("cargo:rustc-check-cfg=cfg(ffx_fsr3_src)");
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
