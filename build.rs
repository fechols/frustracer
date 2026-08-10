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

fn main() {
    require_nrd();
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
    }
}
