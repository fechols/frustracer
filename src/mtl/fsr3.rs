//! FidelityFX FSR3 1.1.4 on Metal — the transpiled shader table and its gates.
//!
//! FidelityFX ships only `ffx_vk` and `ffx_dx12` backends, so Metal reuses the
//! very SPIR-V blobs the Vulkan shaderblob accessor returns: `build.rs::
//! generate_fsr3_metallibs` transpiles them (spirv-cross -> `metal` -> `metallib`)
//! and emits the table this module `include!`s. The key is FNV-1a-64 of the
//! SPIR-V bytes, and `shim/ffx_metal.mm` computes the byte-identical hash over
//! `FfxShaderBlob.data` — the same bytes from the same accessor — to find a
//! permutation's metallib at pipeline-create time. That hash is the ONLY thing
//! the two sides agree on, so neither may be changed alone.

/// The permutation count this tree expects. It is a MEASURED total, not a
/// product — the options-per-pass count is wildly non-uniform, and an earlier
/// comment here read "10 passes x {fp32, fp16} x 4 options = 80", which is
/// arithmetically 80 and factually wrong about every term but the first. What
/// the committed corpus actually holds, per precision:
///
/// | pass                  | permutations |
/// |-----------------------|--------------|
/// | accumulate            | 24           |
/// | prepare_inputs        | 8            |
/// | autogen_reactive      | 1            |
/// | debug_view            | 1            |
/// | luma_instability      | 1            |
/// | luma_pyramid          | 1            |
/// | prepare_reactivity    | 1            |
/// | rcas                  | 1            |
/// | shading_change        | 1            |
/// | shading_change_pyramid| 1            |
///
/// `(24 + 8 + 8x1) x {fp32, fp16} = 40 x 2 = 80`. The distinction matters
/// because the false factorization predicts +8 for an eleventh pass when the
/// truth is anywhere from +2 to +48 — so if this number moves, RE-COUNT rather
/// than multiply.
///
/// The wave64 half of the 160 committed blobs is deliberately NOT transpiled:
/// Apple GPUs are SIMD-32, so `GetDeviceCapabilitiesMetal` reports
/// `waveLaneCountMax = 32` and FFX never requests them (they would also
/// mis-execute at width 32). **That skip and that caps hardcode are a PAIR** —
/// change one and FFX asks for a hash that was never emitted.
pub const EXPECTED_PERMUTATIONS: usize = 80;

#[cfg(ffx_fsr3_metal)]
mod table {
    // FFX_FSR3_METALLIBS: &[(u64, &[u8])] sorted by hash, and
    // FFX_FSR3_PERMUTATIONS_FOUND: the count build.rs ENUMERATED (the
    // denominator — see the gate below for why it is emitted at all).
    include!(concat!(env!("OUT_DIR"), "/ffx_fsr3_metallibs.rs"));
}

/// The transpiled permutations, keyed by FNV-1a-64 of their SPIR-V.
#[cfg(ffx_fsr3_metal)]
pub fn metallibs() -> &'static [(u64, &'static [u8])] {
    table::FFX_FSR3_METALLIBS
}

/// How many non-wave64 permutation headers build.rs enumerated. A gap between
/// this and `metallibs().len()` is a partial transpile.
#[cfg(ffx_fsr3_metal)]
pub fn permutations_found() -> usize {
    table::FFX_FSR3_PERMUTATIONS_FOUND
}

/// The metallib table's gates. Pure — no device, no GPU, no FFX.
///
/// These exist because every failure they catch is otherwise a RARE, RUNTIME,
/// SCENE-DEPENDENT pipeline-create failure: a truncated table only bites when
/// some particular FSR3 pass asks for the one hash that is missing. Catching it
/// here makes a `spirv-cross` upgrade that changes MSL emission a gate failure
/// instead of a mystery.
#[cfg(ffx_fsr3_metal)]
pub fn table_self_test() -> Result<(), String> {
    let libs = metallibs();
    let found = permutations_found();

    // TEETH: an empty or truncated table cannot pass. Both bounds matter —
    // `found` catches "build.rs enumerated 80 and emitted 60" (a transpile
    // regression), `EXPECTED_PERMUTATIONS` catches "the committed shader
    // directory itself lost files", which `found` alone would happily accept.
    if libs.len() != found {
        return Err(format!(
            "ffx-metal: {} metallibs but {found} permutations enumerated — a partial \
             transpile (re-read the build's FSR3-Metal warning for the first failure)",
            libs.len()
        ));
    }
    if found != EXPECTED_PERMUTATIONS {
        return Err(format!(
            "ffx-metal: {found} non-wave64 permutations enumerated, expected \
             {EXPECTED_PERMUTATIONS} ((24 + 8 + 8x1) x {{fp32,fp16}}; see the constant's \
             per-pass table) — the committed SPIR-V under SDKs/FidelityFX-SDK-prebuilt/ \
             has changed"
        ));
    }

    let mut prev: Option<u64> = None;
    for &(hash, blob) in libs {
        // Strictly ascending, hence unique: the C side linear-scans for an
        // equal hash, so a duplicate key makes the selection ambiguous rather
        // than wrong-and-loud.
        if let Some(p) = prev {
            if hash <= p {
                return Err(format!(
                    "ffx-metal: table is not strictly ascending at {hash:#018x} (after \
                     {p:#018x}) — build.rs must sort, and duplicates make lookup ambiguous"
                ));
            }
        }
        prev = Some(hash);

        // `[12-byte LE threadgroup header][metallib]`. Metal needs the
        // workgroup size HOST-side at dispatch (Vulkan and DXIL both reflect it
        // out of the bytecode), so build.rs parses `OpExecutionMode LocalSize`
        // and prepends it; the shim strips the same 12 bytes back off.
        if blob.len() < 16 {
            return Err(format!("ffx-metal: {hash:#018x} blob is {} B — truncated", blob.len()));
        }
        let g = |i: usize| u32::from_le_bytes([blob[i], blob[i + 1], blob[i + 2], blob[i + 3]]);
        let (gx, gy, gz) = (g(0), g(4), g(8));
        // A mis-parsed OpExecutionMode reads as a nonsense group. Metal's own
        // ceiling is 1024 threads per threadgroup, so this is the real bound
        // and not a guess.
        if gx == 0 || gy == 0 || gz == 0 || (gx as u64) * (gy as u64) * (gz as u64) > 1024 {
            return Err(format!(
                "ffx-metal: {hash:#018x} threadgroup header is [{gx},{gy},{gz}] — \
                 OpExecutionMode LocalSize was mis-parsed"
            ));
        }
        // `metallib`'s container magic, so a garbage entry fails HERE rather
        // than inside newLibraryWithData at pipeline-create time.
        if &blob[12..16] != b"MTLB" {
            return Err(format!(
                "ffx-metal: {hash:#018x} is not a metallib past the header (got {:02x?})",
                &blob[12..16]
            ));
        }
    }
    Ok(())
}
