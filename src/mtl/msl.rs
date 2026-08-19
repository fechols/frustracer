//! HLSL -> SPIR-V -> **MSL** -> AIR -> `.metallib`: the corpus's third code
//! generator, and the first rung of a Metal tracer.
//!
//! `crate::spirv` already compiles the shipping corpus to SPIR-V on this
//! platform (47 units -> 78 modules, 0 failed). This module is what happens
//! next: `spirv-cross --msl`, then `xcrun metal`. Nothing here renders, binds
//! or dispatches — `--check-msl` answers "does it compile", and C2 answers
//! "how do we bind it".
//!
//! # The route, and why it is this one
//!
//! `metal-shaderconverter` (Apple's own DXIL -> metallib tool, built for D3D12
//! ports) was rejected on evidence rather than taste: it is not installed with
//! Xcode, is not an `xcrun` tool, and consumes **signed** DXIL — and `dxil.dll`
//! is the Windows-only signer this tree already documents for NRD. spirv-cross
//! is the route `build.rs` already runs for FidelityFX and CI already installs.
//!
//! # The arg set, every entry measured
//!
//! ```text
//! --msl --msl-version 30000
//! --msl-argument-buffers --msl-argument-buffer-tier 2
//! --msl-device-argument-buffer 1
//! ```
//!
//! **`--msl-decoration-binding` is ABSENT**, and it is the one flag `build.rs`
//! passes for FFX. It makes the Metal argument index EQUAL the SPIR-V binding
//! — and ours are `SHIFT_T=1000`, `SHIFT_U=2000`, `SHIFT_S=3000`
//! (`crate::spirv`) while Metal allows textures 0-127 and samplers 0-15.
//!
//! WITHOUT ARGUMENT BUFFERS THAT IS FATAL AND NOT A CORNER CASE: measured, the
//! simplest unit in the corpus (`bloom`) emits `[[texture(1000)]]`,
//! `[[texture(2000)]]` and `[[sampler(3000)]]` and is rejected on all three.
//! FFX hit the same wall with a SINGLE sampler at 1001 and lost 112 of 160
//! permutations to it; `build.rs::remap_ffx_samplers` is the one-subtraction
//! fix that case admits, and ours would need three.
//!
//! WITH THEM IT IS MERELY MOOT, and that correction is worth recording because
//! the first draft of this header got it wrong by generalising from the
//! no-argument-buffer sweep. Resources move INSIDE an argument-buffer struct
//! as `[[id(n)]]` — measured `[[id(0)]] [[id(1000)]] [[id(2000)]]
//! [[id(3000)]]` at `[[buffer(0)]]` — and argument-buffer ids carry no such
//! ceiling, so the same 65 modules compile either way. The flag is omitted
//! because it buys nothing here, not because it would break this
//! configuration. `self_test` therefore pins the IMPLICATION that is actually
//! true (asking for it REQUIRES argument buffers) rather than its absence,
//! which would be pinning a preference.
//!
//! Either way the milestone boundary holds: the Metal argument indices are
//! spirv-cross's business, so **nothing may hardcode one**. C2 derives the map
//! — from Metal reflection or from spirv-cross's own output — exactly as
//! `vk::reflect` derives the Vulkan one rather than transcribing it. The shift
//! constants in `crate::spirv` stay a VULKAN choice and are free to stay one.
//!
//! **`--msl-device-argument-buffer 1` is derived, not magic.** `texs[]` is
//! `register(t10, space1)`, the register SPACE becomes the descriptor SET, and
//! spirv-cross requires runtime-sized arrays to live in *device* storage
//! argument buffers ("Runtime sized variables must be in device storage
//! argument buffers" is the exact refusal without it). Tier 2 is required for
//! the unsized array at all and is supported on every Apple silicon GPU.
//!
//! **`-ffp-contract=off` is NOT needed, and that surprised the plan.**
//! spirv-cross preserves `NoContraction`: it emits
//! `[[clang::optnone]] T spvFMul(T l, T r) { return fma(l, r, T(0)); }` and
//! MSL's `precise::` namespace. So the corpus's `precise` discipline — the
//! thing `ftree.hlsli` relies on so a fused multiply-add cannot round a
//! frustum box inward and break every prune's conservativeness — survives the
//! crossing intact. `__METAL_FAST_MATH__` is also already 0 by default.
//!
//! # What compiles, measured
//!
//! **78 of 83 modules reach AIR under one uniform recipe**, no per-unit
//! conditionals. The 5 that do not are one class — `dxr-lib`, `Expect::
//! NoAnalogue` — and `Expect`'s doc says why a gate that merely skipped them
//! would be worse than useless. 83 − 78 = 5 is the whole remaining gap, so the
//! count is self-checking against that class's size.
//!
//! The denominators below are pre-C3 and stay as written: `mtlbind.hlsl`'s
//! three entry points took every arm up by 3 (80 → 83, 68 → 71) without
//! touching the gap, which is itself the check that the probe added coverage
//! and not exceptions.
//!
//! AND "REACHES AIR" IS NOT "IS CORRECT", which C3 learned the hard way and
//! which `overlap_check` now covers. For eight months 19 of these modules —
//! `leaf`, `leaf_fb`, `reference`, `hemi_leaf` across every vendor and sway
//! arm — were counted as PASSES while `samp_lin` was not bound at all:
//! spirv-cross cannot lay out a descriptor set holding an unsized array beside
//! anything else, so it dropped the sampler behind an `// Overlapping binding:`
//! comment and reinterpret_cast the texture array's storage in its place. That
//! COMPILES. Nothing in a compile gate can see it.
//!
//! Two things came out of that and both are load-bearing. `texs[]` now lives
//! ALONE in `space2` (`trace_common.hlsli` carries the rule), and
//! `overlap_check` refuses the pattern outright, so the gate can no longer
//! count a silently-wrong module. The general lesson is the one worth keeping:
//! `--check-msl` asks "did it compile" and `--check-mtl` asks "is it bound",
//! and the distance between those two questions is exactly where this hid.
//! See `docs/history/metal-backend.md`'s C3 record.
//!
//! # The spirv-cross scoping defect, and how it was retired (C3)
//!
//! Until C3 a second class, `Expect::ToolDefect`, held exactly one unit. The
//! record is kept here because the variant is gone:
//!
//! `hemi_wave.hlsl::check_empty_cell` called `occluded_q` six times under
//! `[unroll]`. Each call declares its own `RayQuery`, but SPIR-V requires every
//! `OpVariable` in a function's FIRST block, so the six unrolled bodies shared
//! one function-scope variable — and spirv-cross then declared it inside a
//! `do{}while(false)` and referenced it after that block closed, giving `use of
//! undeclared identifier '_194'`. **Eight modules.** It needed the OPAQUE
//! `occluded_q` (`rt.hlsli`'s `#else` arm), so it was configuration-dependent —
//! which is why the class was REPORTED rather than required:
//!
//! * procedural scene, hardware rays: 8 modules failed (67/80)
//! * a scene arming `ALPHA_CUTOUT`/`TRANS_SHADOW` (san-miguel): the
//!   candidate-loop arm is structured differently and compiled (75/80)
//! * `--sw-rays`: `rt_sw.hlsli` has no `RayQuery`, so nothing to mis-scope (63/68)
//!
//! MEASURED IDENTICAL on spirv-cross 1.4.350.1 and 1.4.357.0 — not a stale-tool
//! artifact. The fix is `[loop]` for `[unroll]` plus a `switch`-returning weight
//! helper, shipped in the SHARED corpus (a per-backend arm would mean this gate
//! stops covering the program D3D12 and Vulkan actually ship). The procedural
//! arm then lands on 75/80 — **exactly san-miguel's existing number**, which is
//! the check that it changed what it meant to and nothing else.
//!
//! It cost the two backends that were already fine nothing, and the way that was
//! established is worth carrying: in DXIL `dx.op.allocateRayQuery` goes 7 → 2 —
//! the layer spirv-cross consumes, and why this works at all — but on RDNA4 the
//! `image_bvh*_intersect_ray` sites stay 6 and 12 in BOTH arms, because the AMD
//! backend re-rolls to its own taste regardless of the attribute. **An HLSL loop
//! attribute is a statement to the FRONT end.** VGPR/waves/scratch unmoved; see
//! `docs/history/profiling.md`. `cargo test`'s rolled-grid pin is what keeps a
//! revert to `[unroll]` — which reads faster and looks better — from landing
//! green on the two platforms that cannot run `--check-msl`.
//!
//! # The finding that overturned the plan: hardware ray tracing WORKS
//!
//! The milestone was planned on the premise that `RayQuery` has no
//! spirv-cross -> MSL lowering, so a Metal tracer would need `--sw-rays` and
//! its own software BVH walk. **That is false.** `leaf`, `reference`,
//! `leaf_fb` and `hemi_leaf` all reach AIR with ray tracing intact, lowered to
//! `raytracing::acceleration_structure<raytracing::instancing>` and
//! `raytracing::intersection_query`. A Metal tracer can use the hardware, and
//! `--sw-rays` is a measurement lever here exactly as it is everywhere else
//! rather than a required path.
//!
//! What Metal has no analogue for is the DXR **pipeline** shape — raygen,
//! closest-hit, miss and an SBT — which is what the five `dxr-lib` modules
//! are, and which the port does not need: the thing being ported is the
//! wavefront tracer.

use std::path::{Path, PathBuf};

/// `absent` = an environment fact (SKIP, exit 0); anything else is a failure.
/// `mtl::device::MtlError`'s split, for its reason: a box without the Metal
/// toolchain has not failed a gate, and a box that HAS it and refuses the
/// corpus has.
pub struct MslError {
    pub absent: bool,
    pub msg: String,
}

impl MslError {
    fn absent(msg: impl Into<String>) -> MslError {
        MslError { absent: true, msg: msg.into() }
    }
}

/// The spirv-cross arg set. Every entry is justified in the module header;
/// none of them is a preference.
pub const CROSS_ARGS: &[&str] = &[
    "--msl",
    "--msl-version",
    "30000",
    "--msl-argument-buffers",
    "--msl-argument-buffer-tier",
    "2",
    "--msl-device-argument-buffer",
    "2",
];

/// spirv-cross's announcement that it dropped an argument-buffer member.
const OVERLAP_MARK: &str = "// Overlapping binding:";
/// The cast it emits in the dropped member's place. Named separately because
/// the two can appear apart: the comment is emitted at the struct, the cast at
/// each use, and a member that is dropped but never used produces only the
/// first. Either one alone is the failure.
const OVERLAP_CAST: &str = "reinterpret_cast<const device sampler";

/// Refuse generated MSL in which spirv-cross resolved an argument-buffer id
/// collision by DROPPING a member and casting another in its place.
///
/// THIS IS NOT A TOOL BUG, and the message has to say so or the next reader
/// will go looking for one. SPIRV-Cross PR #2292 ("MSL: Add support for
/// overlapping bindings") added the behaviour deliberately, and the README
/// states the rule it follows: *"arrays of resources consume multiple ids,
/// where Vulkan does not… This can be worked around either from shader
/// authoring stage or remapping bindings as needed to avoid the overlap."* An
/// UNSIZED array cannot reserve the ids it will consume, so whatever shares its
/// set collides, and the overlapping-bindings feature resolves the collision by
/// reinterpret_cast. The output COMPILES and reaches AIR reading texture
/// descriptor bytes as a sampler.
///
/// WHICH IS WHY THIS EXISTS AT ALL. `--check-msl` asks "did it compile", and
/// for eight months the answer for `leaf` and `reference` was yes while
/// `samp_lin` — the trilinear sampler every colour / normal / rough-metal read
/// goes through — was not bound at all. A gate that counts a miscompile as a
/// pass is the vacuity class this suite is built to rule out, and the C3 probe
/// found it only because a human read the generated code.
///
/// Kept a free function over `&str` rather than a method so the pure `self_test`
/// can exercise it with no spirv-cross, no device and no macOS.
pub fn overlap_check(msl: &str) -> Result<(), String> {
    let hit = msl.lines().find(|l| l.contains(OVERLAP_MARK) || l.contains(OVERLAP_CAST));
    let Some(line) = hit else { return Ok(()) };
    Err(format!(
        "spirv-cross dropped an argument-buffer member to resolve an [[id]] \
         collision: `{}`. This is DELIBERATE (SPIRV-Cross PR #2292) and not a \
         bug to report: an unsized descriptor array consumes ids it cannot \
         reserve, so anything sharing its descriptor set collides and is \
         replaced by a reinterpret_cast of the array's storage — which compiles, \
         reaches AIR, and reads the wrong descriptor. The fix is at the shader \
         authoring stage, which the tool's own README names: give the unsized \
         array its own register space, alone. Do NOT silence this by relaxing \
         the check.",
        line.trim()
    ))
}

/// The descriptor set `--msl-device-argument-buffer` names, as a derivation
/// rather than a literal: `texs[]` is the corpus's one unbounded array, it
/// lives in `space2`, and the register space IS the descriptor set under the
/// `-fvk-*-shift` scheme. `self_test` pins that the arg set carries this.
///
/// IT NAMES ONE SET, WHICH IS THE WHOLE REASON THIS IS DERIVED. The flag is
/// per-set and repeatable, and `texs[]` moved from space1 to space2 the day C3
/// found it could not share a set (`trace_common.hlsli` carries the rule). A
/// literal `1` left behind would not have failed loudly — spirv-cross throws
/// "Runtime sized variables must be in device storage argument buffers", which
/// reads as a capability problem rather than as a stale constant. The gate's
/// FIRST run of that experiment counted zero overlaps in that exception's
/// output and read it as clean; see the archive.
pub const UNBOUNDED_ARRAY_SET: u32 = 2;

/// What a unit is expected to do, and why.
///
/// A gate that only checked "did the expected-to-work units work" would go
/// green while the known-failing list silently rotted — a workaround could
/// land, or spirv-cross could fix the bug, and nothing would say so. So the
/// failing class carries teeth against staleness as well.
///
/// * `NoAnalogue` is a CAPABILITY claim, so it is hard in BOTH directions: it
///   must still be reached, and one compiling is a FAIL with an instruction in
///   it. Metal either has an analogue for a DXIL ray-tracing library or it
///   does not, and that answer cannot vary with the scene.
///
/// **There were two failing classes, and `ToolDefect` was RETIRED in C3 rather
/// than left with no members** — an unreachable variant makes the sentence above
/// false, which is the staleness this enum exists to prevent. It held exactly
/// one unit, `hemi_wave`, and the finding is kept in the module header rather
/// than deleted with the code.
///
/// Read `run_check_msl`'s M5 block with this: the verdict is per class, and
/// the only unconditional failure is an `Expect::Metallib` unit that refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Expect {
    /// Must reach a `.metallib`.
    Metallib,
    /// The DXR **pipeline** shape — raygen/closest-hit/miss/SBT. Metal
    /// expresses ray tracing as intersection queries and (separately)
    /// intersection functions, but has no analogue for a DXIL ray-tracing
    /// LIBRARY, and spirv-cross emits `unknown type name 'unknown'` for it.
    /// This is a capability gap and a permanent one — but it costs the port
    /// nothing, because what is being ported is the wavefront tracer, and the
    /// wavefront's own ray-shooting kernels compile WITH hardware ray tracing.
    NoAnalogue,
}

/// Classify a unit by the name `--check-spirv` gives it (`hemi_wave[Nvidia]`,
/// `dxr-lib[0]`, …). Prefix-matched on the unit half, so vendor/sway/inline
/// arms all follow their family — a per-arm table would be four times the size
/// and could disagree with itself.
pub fn expect_of(unit: &str) -> Expect {
    let base = unit.split('[').next().unwrap_or(unit);
    match base {
        "dxr-lib" => Expect::NoAnalogue,
        _ => Expect::Metallib,
    }
}

/// The Metal shader toolchain: spirv-cross plus Xcode's `metal`/`metallib`.
pub struct Msl {
    cross: PathBuf,
    scratch: PathBuf,
}

impl Msl {
    /// Locate the tools. `$SPIRV_CROSS` overrides the transpiler, matching
    /// `build.rs`; `$METAL`/`$METALLIB` override the Apple tools through
    /// `xcrun_tool`, likewise.
    ///
    /// Every absence is `absent: true`. A macOS box without the Metal
    /// toolchain cryptex or without spirv-cross is an environment, not a
    /// defect — and CI's own probe step exists because that cryptex is a
    /// MobileAsset that the Xcode manifest does not list.
    pub fn find(scratch: PathBuf) -> Result<Msl, MslError> {
        let cross = PathBuf::from(
            std::env::var("SPIRV_CROSS").unwrap_or_else(|_| "spirv-cross".to_string()),
        );
        // spirv-cross exits non-zero on `--version` (it prints its banner and
        // treats the flag as a usage error), so LAUNCHING it is the probe.
        if std::process::Command::new(&cross).arg("--version").output().is_err() {
            return Err(MslError::absent(format!(
                "spirv-cross not found (`{}`; brew install spirv-cross, or set \
                 $SPIRV_CROSS)",
                cross.display()
            )));
        }
        for tool in ["metal", "metallib"] {
            // THE STATUS IS THE PROBE HERE, not the launch, and the difference
            // is the whole point: the process being launched is `xcrun`, which
            // exists on any box with the Command Line Tools. MEASURED —
            // `xcrun -sdk macosx <absent-tool> --version` LAUNCHES and exits
            // 72, so `output()` returns `Ok` and a launch-only probe cannot
            // see a missing tool at all. That is not a corner case: the Metal
            // toolchain is a separate MobileAsset cryptex the Xcode manifest
            // does not list, so "Xcode present, `metal` absent" is the
            // expected shape of a box that lacks it — and letting it past
            // turns an environment fact into ~65 M4 failures, the exact
            // inversion of this function's contract.
            //
            // Checking the status is safe HERE and would be wrong for
            // spirv-cross above, which treats `--version` as a usage error;
            // `metal` and `metallib` both exit 0 on it (measured). A launch
            // failure still lands in the same arm, which covers no Xcode at
            // all and a `$METAL`/`$METALLIB` override naming nothing.
            match xcrun_tool(tool).arg("--version").output() {
                Ok(o) if o.status.success() => {}
                _ => {
                    return Err(MslError::absent(format!(
                        "`xcrun -sdk macosx {tool}` did not answer --version — the \
                         Metal toolchain is a separate MobileAsset cryptex on recent \
                         Xcode, and is absent or broken here"
                    )))
                }
            }
        }
        std::fs::create_dir_all(&scratch)
            .map_err(|e| MslError { absent: false, msg: format!("scratch dir: {e}") })?;
        Ok(Msl { cross, scratch })
    }

    /// Tool identity, for the gate's line. Versions rather than paths because
    /// the `hemi_wave` defect above is version-sensitive in principle even
    /// though it is measured stable across two.
    pub fn line(&self) -> String {
        let v = |mut c: std::process::Command| -> String {
            c.output()
                .ok()
                .map(|o| {
                    let s = String::from_utf8_lossy(&o.stdout).to_string()
                        + &String::from_utf8_lossy(&o.stderr);
                    s.lines().next().unwrap_or("").trim().to_string()
                })
                .unwrap_or_else(|| "?".into())
        };
        let mut c = std::process::Command::new(&self.cross);
        c.arg("--version");
        format!("spirv-cross [{}] | metal [{}]", v(c), v({
            let mut m = xcrun_tool("metal");
            m.arg("--version");
            m
        }))
    }

    /// SPIR-V words -> MSL source. `stem` names the scratch files, so it must
    /// already be filesystem-safe.
    pub fn transpile(&self, words: &[u32], stem: &str) -> Result<String, String> {
        let spv = self.scratch.join(format!("{stem}.spv"));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&spv, &bytes).map_err(|e| format!("write {}: {e}", spv.display()))?;
        let out = std::process::Command::new(&self.cross)
            .args(CROSS_ARGS)
            .arg(&spv)
            .output()
            .map_err(|e| format!("launch spirv-cross: {e}"))?;
        let _ = std::fs::remove_file(&spv);
        if !out.status.success() {
            // spirv-cross reports refusals on stderr as
            // "SPIRV-Cross threw an exception: <reason>" — keep the reason and
            // drop the prefix, since the gate prints one line per failure.
            let e = String::from_utf8_lossy(&out.stderr);
            let first = e.lines().next().unwrap_or("").trim();
            return Err(format!(
                "spirv-cross: {}",
                first.strip_prefix("SPIRV-Cross threw an exception: ").unwrap_or(first)
            ));
        }
        let msl = String::from_utf8(out.stdout)
            .map_err(|e| format!("spirv-cross emitted non-UTF-8: {e}"))?;
        overlap_check(&msl)?;
        Ok(msl)
    }

    /// MSL -> AIR -> `.metallib`, returning the library bytes.
    ///
    /// NO 12-BYTE THREADGROUP HEADER, unlike `build.rs`'s FFX path. That header
    /// exists because the FFX shim must dispatch without reflecting; C2's
    /// dispatcher holds the SPIR-V words and reads the group size out of them
    /// (`crate::spirv::local_size`), so adding one here would still be inventing
    /// a wire format for a consumer that does not exist.
    pub fn compile(&self, msl: &str, stem: &str) -> Result<Vec<u8>, String> {
        let lib = self.compile_lib(msl, stem)?;
        let bytes = std::fs::read(&lib).map_err(|e| format!("read {}: {e}", lib.display()));
        let _ = std::fs::remove_file(&lib);
        bytes
    }

    /// The same recipe, leaving the `.metallib` ON DISK and returning its path.
    ///
    /// `compile` is this plus a read and an unlink — ONE recipe, two consumers,
    /// the `corpus_units` rule. `--check-msl` wants the bytes and nothing else;
    /// `mtl::bind` wants a file, because `newLibraryWithURL:` is the route that
    /// costs no dependency (`newLibraryWithData:` is gated behind objc2-metal's
    /// `dispatch2` feature and would pull `block2` in with it).
    ///
    /// The caller owns the file. Every caller here writes into `Msl::scratch`,
    /// which the gate removes wholesale when it finishes.
    pub fn compile_lib(&self, msl: &str, stem: &str) -> Result<PathBuf, String> {
        let m = self.scratch.join(format!("{stem}.metal"));
        let air = self.scratch.join(format!("{stem}.air"));
        let lib = self.scratch.join(format!("{stem}.metallib"));
        std::fs::write(&m, msl).map_err(|e| format!("write {}: {e}", m.display()))?;
        let steps: [(&str, Vec<&Path>); 2] = [
            ("metal", vec![Path::new("-c"), &m, Path::new("-o"), &air]),
            ("metallib", vec![&air, Path::new("-o"), &lib]),
        ];
        let mut err = None;
        for (tool, args) in steps {
            let out = xcrun_tool(tool)
                .args(&args)
                .output()
                .map_err(|e| format!("launch {tool}: {e}"))?;
            if !out.status.success() {
                let e = String::from_utf8_lossy(&out.stderr);
                // The first `error:` line, without clang's file:line prefix —
                // the same shape the failure table wants for every tool.
                let first = e
                    .lines()
                    .find(|l| l.contains("error:"))
                    .and_then(|l| l.split("error: ").nth(1))
                    .unwrap_or_else(|| e.lines().next().unwrap_or("").trim());
                err = Some(format!("{tool}: {first}"));
                break;
            }
        }
        // The intermediates go either way; the library survives only on
        // success, so a stale one from a previous stem can never be mistaken
        // for this compile's output.
        for p in [&m, &air] {
            let _ = std::fs::remove_file(p);
        }
        match err {
            Some(e) => {
                let _ = std::fs::remove_file(&lib);
                Err(e)
            }
            None => Ok(lib),
        }
    }
}

/// `$METAL` / `$METALLIB` (whitespace-split, overriding the whole command)
/// else `xcrun -sdk macosx <tool>`. `build.rs::xcrun_tool`'s twin — the two
/// are the same rule because a developer overriding the toolchain for a build
/// means to override it for the gate as well.
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

/// The pure half — no tools, no device. Runs as M0.
pub fn self_test() -> Result<(), String> {
    // The arg set must carry the three decisions the header argues for, and
    // must NOT carry the one it argues against. Checked by CONTENT rather than
    // by a full-list compare so the test says which decision broke.
    let has = |a: &str| CROSS_ARGS.contains(&a);
    // A flag's VALUE is the token after it. Positional, never a bare
    // `contains`: a value check that merely asks whether the string appears
    // anywhere passes on any OTHER flag's argument that happens to match, so
    // it can go vacuous without moving a line of this test.
    let val = |flag: &str| -> Option<&'static str> {
        CROSS_ARGS.iter().position(|a| *a == flag).and_then(|i| CROSS_ARGS.get(i + 1)).copied()
    };
    if !has("--msl") {
        return Err("CROSS_ARGS does not ask for MSL".into());
    }
    if !has("--msl-argument-buffers") || !has("--msl-argument-buffer-tier") {
        return Err("CROSS_ARGS lost argument buffers — `texs[]` cannot lower without them".into());
    }
    match val("--msl-argument-buffer-tier") {
        Some("2") => {}
        v => {
            return Err(format!(
                "--msl-argument-buffer-tier names {v:?}, but `texs[]` is a runtime-sized \
                 array and those are tier 2 only"
            ))
        }
    }
    // The IMPLICATION, not the absence — see the header. Asking for the
    // identity mapping is legal here only because the resources sit in an
    // argument-buffer struct, where `[[id(1000)]]` carries no ceiling; drop
    // the argument buffers and the same flag makes `[[texture(1000)]]`, which
    // Metal rejects. So this pins the pairing rather than a preference.
    if has("--msl-decoration-binding") && !has("--msl-argument-buffers") {
        return Err(
            "CROSS_ARGS asks for --msl-decoration-binding WITHOUT argument buffers, \
             which makes the Metal argument index equal the SPIR-V binding — and ours \
             are 1000/2000/3000, all out of Metal's range (textures 0-127, samplers \
             0-15). Measured: `bloom` alone fails on three of them."
                .into(),
        );
    }
    // The device-argument-buffer set is a DERIVATION, so prove the constant
    // and the flag agree rather than trusting two literals to stay in step.
    let want = UNBOUNDED_ARRAY_SET.to_string();
    if !has("--msl-device-argument-buffer") {
        return Err("CROSS_ARGS lost --msl-device-argument-buffer".into());
    }
    match val("--msl-device-argument-buffer") {
        Some(v) if v == want => {}
        v => {
            return Err(format!(
                "--msl-device-argument-buffer names {v:?}, but `texs[]` lives in \
                 space{UNBOUNDED_ARRAY_SET} so the set must be {want}"
            ))
        }
    }

    // `overlap_check` both ways, on VERBATIM spirv-cross output rather than on
    // a paraphrase — a detector tested against a string this file also wrote is
    // testing its own spelling. The positive case is the shipping `leaf`
    // set-1 struct as it was emitted before `texs[]` moved to its own space;
    // the negative is the same struct after.
    const DROPPED: &str = "\
struct spvDescriptorSetBuffer1
{
    device type_StructuredBuffer_uint* chunk_base [[id(3)]];
    spvDescriptor<texture2d<float>> texs [[id(4)]][1] /* unsized array hack */;
    // Overlapping binding: sampler samp_lin [[id(4)]];
    sampler samp_aniso [[id(5)]];
};
    const device auto &samp_lin = reinterpret_cast<const device sampler &>(spvDescriptorSet1.texs);";
    const CLEAN: &str = "\
struct spvDescriptorSetBuffer1
{
    device type_StructuredBuffer_uint* chunk_base [[id(3)]];
    sampler samp_lin [[id(4)]];
    sampler samp_aniso [[id(5)]];
};
struct spvDescriptorSetBuffer2
{
    spvDescriptor<texture2d<float>> texs [[id(0)]][1] /* unsized array hack */;
};";
    match overlap_check(DROPPED) {
        Err(e) if e.contains("samp_lin") => {}
        Err(e) => return Err(format!("overlap_check fired but did not name the member: {e}")),
        Ok(()) => {
            return Err("overlap_check PASSED the dropped-member MSL — the marker it \
                        scans for no longer matches what spirv-cross emits, so the \
                        check is silent on the exact failure it was written for"
                .into())
        }
    }
    // The two markers must be independently sufficient: a dropped member that
    // is never USED emits the comment and no cast, and a check that demanded
    // both would miss it.
    for (only, what) in [(OVERLAP_MARK, "the struct comment"), (OVERLAP_CAST, "the cast")] {
        if overlap_check(&format!("    {only} sampler x [[id(0)]];")).is_ok() {
            return Err(format!("overlap_check needs BOTH markers — {what} alone is silent"));
        }
    }
    if let Err(e) = overlap_check(CLEAN) {
        return Err(format!(
            "overlap_check rejected MSL with the array in its OWN set, which is the \
             arrangement the corpus now uses — it would fail every run: {e}"
        ));
    }

    // The classification. Both non-Metallib arms must be reachable, or the
    // gate's both-directions teeth are scoring an empty set.
    let cases: [(&str, Expect); 6] = [
        ("dxr-lib[0]", Expect::NoAnalogue),
        ("dxr-lib[3]", Expect::NoAnalogue),
        // POSITIVE pins, and they are the retired class's replacement. Under
        // `ToolDefect` a regression was merely REPORTED; as `Metallib` a revert
        // to `[unroll]` — or a spirv-cross change that reintroduces the
        // mis-scope — is a hard `FAIL M4` on every arm. Both vendor spellings,
        // because `expect_of` prefix-matches and a change to that split would
        // otherwise reclassify one arm silently.
        ("hemi_wave[Nvidia]", Expect::Metallib),
        ("hemi_wave[Amd+sway]", Expect::Metallib),
        ("leaf[Nvidia]", Expect::Metallib),
        ("bloom", Expect::Metallib),
    ];
    for (unit, want) in cases {
        let got = expect_of(unit);
        if got != want {
            return Err(format!("expect_of({unit}) = {got:?}, want {want:?}"));
        }
    }
    // `dxr-shade` and `dxr-feed` are ORDINARY COMPUTE units that happen to
    // carry the dxr prefix, and they compile — so the classifier must key on
    // the whole unit name and not on a `dxr` substring. This case is the one
    // that would have caught a `starts_with("dxr")` shortcut.
    for unit in ["dxr-shade[3]", "dxr-feed[0]", "dxr-nrd[0]", "dxr-resolve[0]"] {
        if expect_of(unit) != Expect::Metallib {
            return Err(format!("{unit} is ordinary compute and must be expected to compile"));
        }
    }
    Ok(())
}
