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
//! **65 of 78 modules reach AIR under one uniform recipe**, no per-unit
//! conditionals. The 13 that do not are exactly two classes, and `Expect`
//! below names both — see its doc for why a gate that merely skipped them
//! would be worse than useless.
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
    "1",
];

/// The descriptor set `--msl-device-argument-buffer` names, as a derivation
/// rather than a literal: `texs[]` is the corpus's one unbounded array, it
/// lives in `space1`, and the register space IS the descriptor set under the
/// `-fvk-*-shift` scheme. `self_test` pins that the arg set carries this.
pub const UNBOUNDED_ARRAY_SET: u32 = 1;

/// What a unit is expected to do, and why.
///
/// A gate that only checked "did the expected-to-work units work" would go
/// green while the known-failing list silently rotted — a workaround could
/// land, or spirv-cross could fix the bug, and nothing would say so. So the
/// two failing classes carry teeth against staleness as well; **what differs
/// is how hard, and the asymmetry is deliberate rather than an oversight.**
///
/// * `NoAnalogue` is a CAPABILITY claim, so it is hard in BOTH directions: it
///   must still be reached, and one compiling is a FAIL with an instruction in
///   it. Metal either has an analogue for a DXIL ray-tracing library or it
///   does not, and that answer cannot vary with the scene.
/// * `ToolDefect` is a BUG-PRESENCE claim, which is a property of the
///   configuration and the tool version — so it is REPORTED, never required.
///   Demanding it fire would fail the gate on exactly the configurations where
///   the corpus does BETTER; see the variant's own doc for the three measured
///   arms.
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
    /// An upstream spirv-cross scoping defect, not a corpus problem.
    ///
    /// `hemi_wave.hlsl::check_empty_cell` calls `occluded_q` six times under
    /// `[unroll]`. Each call declares its own `RayQuery`, but SPIR-V requires
    /// every `OpVariable` in a function's FIRST block, so the six unrolled
    /// bodies share one function-scope variable — and spirv-cross then
    /// declares it inside a `do{}while(false)` block and references it after
    /// that block closes, giving `use of undeclared identifier '_194'`.
    ///
    /// IT IS CONFIGURATION-DEPENDENT, and that is why this class is REPORTED
    /// rather than required (see `run_check_msl`'s verdict). The defect needs
    /// the OPAQUE `occluded_q` — `rt.hlsli`'s `#else` arm — so:
    ///
    /// * procedural scene, hardware rays: 8 modules fail (65/78)
    /// * a scene arming `ALPHA_CUTOUT`/`TRANS_SHADOW` (san-miguel): the
    ///   candidate-loop arm is structured differently and compiles (73/78)
    /// * `--sw-rays`: `rt_sw.hlsli` has no `RayQuery` at all, so there is no
    ///   function-scope query variable to mis-scope (61/66)
    ///
    /// A bug's PRESENCE is a property of the configuration and the tool
    /// version, unlike `NoAnalogue`'s capability claim — so demanding that it
    /// still reproduce would make the gate fail on exactly the scenes where
    /// the corpus does BETTER.
    ///
    /// MEASURED IDENTICAL on spirv-cross 1.4.350.1 and 1.4.357.0, so it is not
    /// a stale-tool artifact. A workaround is measured and NOT shipped:
    /// `[loop]` instead of `[unroll]` takes the corpus to 73/78. It is a
    /// codegen change to a path D3D12 and Vulkan both compile and neither is
    /// verifiable from a macOS box, for code that only runs under a verify
    /// probe — so it belongs to the milestone that needs `hemi_wave` to
    /// actually RUN, with those two suites re-run, rather than to a gate whose
    /// product is a measurement.
    ToolDefect,
}

/// Classify a unit by the name `--check-spirv` gives it (`hemi_wave[Nvidia]`,
/// `dxr-lib[0]`, …). Prefix-matched on the unit half, so vendor/sway/inline
/// arms all follow their family — a per-arm table would be four times the size
/// and could disagree with itself.
pub fn expect_of(unit: &str) -> Expect {
    let base = unit.split('[').next().unwrap_or(unit);
    match base {
        "dxr-lib" => Expect::NoAnalogue,
        "hemi_wave" => Expect::ToolDefect,
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
        String::from_utf8(out.stdout).map_err(|e| format!("spirv-cross emitted non-UTF-8: {e}"))
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

    // The classification. Both non-Metallib arms must be reachable, or the
    // gate's both-directions teeth are scoring an empty set.
    let cases: [(&str, Expect); 6] = [
        ("dxr-lib[0]", Expect::NoAnalogue),
        ("dxr-lib[3]", Expect::NoAnalogue),
        ("hemi_wave[Nvidia]", Expect::ToolDefect),
        ("hemi_wave[Amd+sway]", Expect::ToolDefect),
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
