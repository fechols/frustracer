fn main() {
    // Streamline FFI shim: a small C++ TU compiled against the real SL
    // headers (layouts/GUIDs/struct versions come from the SDK, not from
    // hand-mirrored Rust). It dynamically loads sl.interposer.dll at runtime,
    // so nothing links against Streamline import libraries and headless
    // `--check` runs never touch SL.
    #[cfg(windows)]
    {
        println!("cargo:rerun-if-changed=shim/sl_shim.cpp");
        println!("cargo:rerun-if-changed=shim/sl_shim.h");
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .flag("/EHsc")
            .include("SDKs/streamline-sdk/include")
            .file("shim/sl_shim.cpp")
            .compile("sl_shim");

        // FidelityFX FFI shim: same doctrine as the SL shim — the ffx-api
        // structs (pNext chains, FfxApiResource descriptions) are built inside
        // one C++ TU against the vendored MIT headers; the loader DLL is
        // LoadLibraryExW'd at runtime, so nothing links FFX and headless
        // `--check*` runs never touch it. ffx_shim.cpp reaches the vendored
        // headers by relative path, so no .include() is needed beyond rerun
        // tracking.
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
            println!(
                "cargo:warning=DLSS SDK not found at {} — raw-NGX DLSS (ray \
                 reconstruction + frame generation) disabled (set \
                 FRUSTRACER_DLSS_SDK to enable)",
                sdk.display()
            );
        }
    }
    #[cfg(not(windows))]
    {
        println!("cargo:rustc-check-cfg=cfg(dlss_ngx)");
    }
}
