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
    }
}
