//! FidelityFX SDK 1.1.4 — the Rust side of frustracer's second FSR3 arm.
//!
//! **Two FidelityFX generations live in this tree and they do not compete.**
//! `src/fsr.rs` + `src/gpu/ffx*.rs` + `shim/ffx_shim.cpp` are **ffx-api v2.3.0**:
//! signed prebuilt DX12 provider DLLs, `ffxCreateContext`/`ffxDispatch`, and with
//! them FSR4, Ray Regeneration and frame generation. That generation cannot reach
//! the Vulkan or Metal backends — it has no Vulkan backend at all (its own readme
//! lists *"Vulkan is currently not supported in SDK"* under known issues) and v2
//! **removed the `FfxInterface` custom-backend seam**, so a Metal implementation
//! is not merely unwritten but unexpressible. v1.1.4 is the last generation that
//! is MIT source with a stock first-party `ffx_vk` and a seam. Windows keeps
//! v2.3.0; this serves Vulkan and Metal.
//!
//! **Nothing here is loaded at runtime.** Unlike every other vendor SDK in this
//! tree (DXC, OIDN, XeSS, NRD, ffx-api), FidelityFX 1.1.x is *compiled from
//! source and statically linked* — so there is no DLL to find, no fn-pointer
//! table, and no absent-library shed. The degrade is at BUILD time: `build.rs`
//! sets `cfg(ffx_fsr3_src)` only when both halves are present (the fetched SDK
//! source and the committed SPIR-V permutations), warns naming the missing one
//! otherwise, and a build without it simply has no FSR3.
//!
//! The pure half below compiles and gates on **every** platform, Windows
//! included, which is deliberate: the version arithmetic and the pin are facts
//! about the SDK, not about the host, and `--check-fsr` is DLL-free everywhere.

/// The SDK version this tree is written against, as (major, minor, patch).
///
/// A PIN PAIR with `install-prerequisites.sh`'s `FFX_SRC_TAG` — move them
/// together. The gate below is what makes a drift loud: fetch a different tag
/// without editing this and the compiled headers stop agreeing with it. Same
/// shape as `NRD_TAG` ↔ `src/nrd.rs`'s `GetLibraryDesc` check, and for the same
/// reason — a silently mismatched SDK is a class of bug no image gate can see.
pub const PIN: (u32, u32, u32) = (1, 1, 4);

/// `FFX_SDK_MAKE_VERSION` — `(major << 22) | (minor << 12) | patch`
/// (`ffx_interface.h:65`). Transcribed rather than called so the decoder below
/// has something independent to be checked against.
pub const fn make_version(major: u32, minor: u32, patch: u32) -> u32 {
    (major << 22) | (minor << 12) | patch
}

/// The inverse. Field widths follow the packing: patch is 12 bits, minor 10.
pub const fn split_version(v: u32) -> (u32, u32, u32) {
    (v >> 22, (v >> 12) & 0x3ff, v & 0xfff)
}

#[cfg(ffx_fsr3_src)]
unsafe extern "C" {
    fn frshim_ffx_sdk_version() -> u32;
    fn frshim_ffx_context_size() -> u32;
}

/// Was the SDK compiled into this binary?
pub const fn built() -> bool {
    cfg!(ffx_fsr3_src)
}

/// The version of the headers the linked objects were actually compiled
/// against — not a string this side could hold its own stale copy of.
pub fn sdk_version() -> Option<(u32, u32, u32)> {
    #[cfg(ffx_fsr3_src)]
    {
        Some(split_version(unsafe { frshim_ffx_sdk_version() }))
    }
    #[cfg(not(ffx_fsr3_src))]
    {
        None
    }
}

/// `FFX_SDK_DEFAULT_CONTEXT_SIZE` as the SDK's headers saw it, after
/// `shim/ffx_msvc_compat.h` raised it. Upstream hard-codes 128 KB against a
/// 2-byte `wchar_t`; every platform that is not MSVC has a 4-byte one, which
/// bloats the private FSR3 context past that bound — so the override is a
/// correctness requirement, and reading it back is how the gate proves the
/// force-include actually reached the SDK rather than trusting a build flag.
pub fn context_size() -> Option<u32> {
    #[cfg(ffx_fsr3_src)]
    {
        Some(unsafe { frshim_ffx_context_size() })
    }
    #[cfg(not(ffx_fsr3_src))]
    {
        None
    }
}

/// The minimum context size the 4-byte-`wchar_t` platforms need. Upstream's
/// 128 KB is what we must be ABOVE; the compat header sets 1 MB.
pub const MIN_CONTEXT_SIZE: u32 = 128 * 1024;

/// Pure gates — run everywhere, no SDK required.
pub fn self_test() -> Result<(), String> {
    // Round-trip the packing over the whole representable field, including the
    // boundaries where a wrong shift width would alias one field into another.
    for &(ma, mi, pa) in &[
        (0, 0, 0),
        PIN,
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (3, 1023, 4095), // every field at its maximum
        (2, 512, 2048),
    ] {
        let (a, b, c) = split_version(make_version(ma, mi, pa));
        if (a, b, c) != (ma, mi, pa) {
            return Err(format!(
                "ffx version round-trip: ({ma},{mi},{pa}) -> {:#x} -> ({a},{b},{c})",
                make_version(ma, mi, pa)
            ));
        }
    }

    // TEETH: the fields must not bleed. A patch at its maximum must leave minor
    // and major at zero — this is what catches a mis-transcribed shift width,
    // which the round-trip above would happily accept if BOTH directions were
    // wrong in the same way.
    if split_version(make_version(0, 0, 4095)) != (0, 0, 4095) {
        return Err("ffx version: patch overflowed its field".into());
    }
    if make_version(0, 1, 0) != 1 << 12 || make_version(1, 0, 0) != 1 << 22 {
        return Err("ffx version: field offsets do not match ffx_interface.h:65".into());
    }

    // The pin, spelled as the number the C side returns, so the two encodings
    // are checked against each other rather than only against themselves.
    if make_version(PIN.0, PIN.1, PIN.2) != (1 << 22) | (1 << 12) | 4 {
        return Err("ffx version: PIN does not encode to v1.1.4".into());
    }
    Ok(())
}
