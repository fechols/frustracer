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
    }
}
