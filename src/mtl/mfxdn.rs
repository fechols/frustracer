//! MetalFX temporal DENOISED upscaling — Apple's `MTLFXTemporalDenoisedScaler`
//! over the same G-buffer `mtl::fsr3` and `mtl::mfx` feed.
//!
//! **This is the platform-native answer to "NRD on macOS", not a substitute for
//! one.** The original ask named NRD for pre-upscale denoising; transpiling
//! NVIDIA's SPIR-V would mean a macOS CMake arm, a `.dylib` loader and a third
//! recorder, and at least one unknown (four of NRD's fourteen kernels declare
//! `SPV_KHR_compute_shader_derivatives`, whose survival through an MSL
//! transpile is unchecked). Apple ships a ray-reconstruction-class denoiser
//! that takes EXACTLY the planes `dlss::GBufs` already fills — colour, depth,
//! motion, diffuse albedo, specular albedo, normal, roughness, and optionally
//! specular hit distance — every one of them written unconditionally at
//! `render.rs`'s two fill sites on every platform.
//!
//! **It is also the first thing on macOS that can be scored for QUALITY rather
//! than wiring.** An upscaler has no directional claim — `--check-fsr3`'s own
//! notes record that its quality comparison is report-only, because mean-|d|
//! against a converged reference rewards blur and inverts between scenes. A
//! denoiser's claim is directional: noise must go DOWN. That makes
//! `--check-oidn`'s "mean |Laplacian| strictly drops" available here, and it is
//! what settles the one convention Apple does not document (below).
//!
//! # What it shares with `mtl::mfx`, and what it does not
//!
//! Shared, and deliberately NOT re-derived — see that module's header for the
//! arguments: the MV convention (its `motionVectorScaleX` doc is verbatim
//! identical to the plain scaler's, so the derivation transfers unchanged and
//! the scale is the bare `fsr::UPSCALE_MV_SIGN`); reversed-Z depth; the two
//! determinism preconditions (`requiresSynchronousInitialization(true)`,
//! `autoExposureEnabled(false)` plus a 1x1 exposure texture at 1.0); the
//! PRIVATE output with its blit-based read and pre-dispatch clear; and
//! allocating the planes before the scaler.
//!
//! Four things differ, each with a reason:
//!
//! * **`setShouldResetHistory`, not `setReset`.** The property is RENAMED on
//!   the denoised protocol. The old selector does not respond, so a copy-paste
//!   from `mfx.rs` would leave history reset wired to nothing — silent, and in
//!   the direction that looks like a working denoiser with a permanent ghost.
//! * **No `setInputContentWidth/Height`.** They are dropped from
//!   `MTLFXTemporalDenoisedScalerBase` entirely, so this path has no
//!   dynamic-resolution mechanism at all and fixed-size allocation is
//!   STRUCTURAL here rather than the scoping choice it is in `mfx.rs`.
//! * **`setSpecularHitDistanceTextureEnabled(true)`.** Optional, and we have
//!   the plane — it is what DLSS-RR uses it for, and enabling it costs one
//!   texture.
//! * **Two camera matrices**, which the plain scaler does not take at all —
//!   and which this module deliberately does NOT set. See the block above
//!   `impl MfxDn` for the measurements that decided that.
//!
//! # macOS 26.0, and why that is a runtime probe rather than a build gate
//!
//! `MTLFXTemporalDenoisedScaler` is `API_AVAILABLE(macos(26.0))` against the
//! plain scaler's 13.0. That is NOT handled with a cargo feature or a build.rs
//! cfg, because a build.rs check keys on the BUILD host — precisely the
//! `cfg(windows)`-describes-the-HOST defect class this tree records elsewhere,
//! and a cross-compile or a CI image would silently decide it for a machine it
//! knows nothing about.
//!
//! It does not need one either. rustc's deployment target is 11.0, so the class
//! weak-imports and the binary still LAUNCHES on older macOS — the floor does
//! not rise. What does bite is that objc2's `extern_class!` PANICS on a missing
//! class, so `available()` below probes with `AnyClass::get` and every typed
//! entry point goes through it first. Absent = an environment fact = a loud
//! SKIP, exactly as an absent SDK gets everywhere else.
//!
//! # The one convention Apple does not document: the normal SPACE
//!
//! `normalTexture`'s doc says only "The normal texture this scaler evaluates"
//! plus the format/usage boilerplate. No space, no encoding, no range — and the
//! same is true of roughness and both albedos, though those three have no space
//! to be wrong about.
//!
//! We ship WORLD space, and it is MEASURED rather than argued. The argument
//! existed first — it is our plane, it matches DLSS-RR's own normal input, and
//! a denoiser wanting view-space normals would not need a `worldToViewMatrix`
//! to be told how to get there — but B3's jitter sign is the standing warning
//! that a good argument about an undocumented convention is still a coin flip.
//!
//! `FR_MFXDN_NORMALS=world|view` walks it, and the DIRECTIONAL metric settles
//! it where no upscaler arm could: on the default scene the denoised output
//! carries **33.8% less high-frequency content than the plain scaler with world
//! normals, against 28.8% with view**. World denoises better, so world is the
//! answer, and the lever stays because that is a per-scene measurement rather
//! than a proof.

use objc2::rc::Retained;
use objc2::runtime::{AnyClass, ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLPixelFormat, MTLStorageMode, MTLTexture, MTLTextureUsage};
use objc2_metal_fx::{
    MTLFXTemporalDenoisedScaler, MTLFXTemporalDenoisedScalerBase,
    MTLFXTemporalDenoisedScalerDescriptor,
};
use std::sync::atomic::AtomicU32;

use super::device::Mtl;

/// Sign applied to the renderer's sample offset before it reaches
/// `setJitterOffsetX/Y`.
///
/// A LITERAL, deliberately, and not `= mfx::JITTER_SIGN` — which is what it was
/// until the review that wrote this paragraph.
///
/// The claim being made is that these are two different scalers with two
/// independent and equally undocumented jitter conventions, which happen to
/// agree today as a MEASUREMENT rather than by definition. An alias contradicts
/// that claim in the one way that matters: `mfx::JITTER_SIGN` is itself
/// `= fsr::JITTER_SIGN`, so the chain bottoms out at the FidelityFX wire's
/// polarity — a genuinely unrelated subject — and settling FFX's sign to -1.0
/// would silently flip BOTH MetalFX arms with no edit to either and no gate
/// that would notice.
///
/// So the value is written out, and `FR_MFXDN_JITTER` is what walks it. That it
/// equals `mfx::JITTER_SIGN` and `fsr::JITTER_SIGN` today is a fact about three
/// separate measurements agreeing, which is the thing the doc wanted to say.
pub const JITTER_SIGN: f32 = 1.0;

/// Which space the normal plane is handed to the scaler in. See the header.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Normals {
    World,
    View,
}

/// Per-dispatch state.
///
/// `jitter` is ALREADY multiplied by `JITTER_SIGN` — the same contract
/// `mtl::mfx::MfxParams` and `mtl::fsr3::DispatchParams` state, so no arm can
/// disagree about whose job the sign is.
pub struct MfxDnParams {
    pub jitter: (f32, f32),
    pub reset: bool,
}

pub struct MfxDn {
    scaler: Retained<ProtocolObject<dyn MTLFXTemporalDenoisedScaler>>,
    trio: super::planes::Trio,
    guides: super::planes::Guides,
    /// PRIVATE storage — `outputTexture` is the one property Apple documents a
    /// storage requirement for, on this protocol as on the plain one.
    output: Retained<ProtocolObject<dyn MTLTexture>>,
    exposure: Retained<ProtocolObject<dyn MTLTexture>>,
    render: (usize, usize),
    upscale: (usize, usize),
    normals: Normals,
}

// ---------------------------------------------------------------------------
// THE TWO CAMERA MATRICES ARE DELIBERATELY NOT SET, and this block is why —
// because "a correct integration sets every documented input" is the obvious
// reading and it is the wrong one here.
//
// `MTLFXTemporalDenoisedScalerBase` declares `worldToViewMatrix` and
// `viewToClipMatrix` as readwrite `simd_float4x4`. Both are ABSENT from
// objc2-metal-fx (its header-translator drops that type), so reaching them
// means `objc_msgSend` transmuted to a hand-written signature — and on AArch64
// a struct of four 128-bit vectors is a Homogeneous Vector Aggregate passed in
// v0..v3, which a `#[repr(C)] struct([f32; 16])` is NOT. That was built, and
// then measured, and the measurements are what removed it:
//
//   1. THE ROUND TRIP DOES NOT WORK. `viewToClipMatrix` read back EXACTLY while
//      `worldToViewMatrix` read back as pointer-shaped garbage — same function,
//      same struct, same transmute. Setting ONLY `worldToViewMatrix` and then
//      reading `viewToClipMatrix` returned the WORLD matrix's contents, which
//      is the diagnosis: the "successful" reads were register residue from the
//      immediately preceding setter, not storage. The RETURN side of a 64-byte
//      HVA is not classified the way ObjC returns it.
//   2. SETTING THEM CHANGES NOTHING. With both setters removed entirely, every
//      number `--check-metalfx`'s X5 reports is byte-for-byte identical —
//      energy, history, the Laplacian drop, and all four guide probes. A 90
//      degree rotation of `world_to_view` moves the output by exactly 0.
//
// So: no measurable benefit, and an unverifiable 64-byte write through a
// transmuted function pointer. AN UNVERIFIABLE WRITE IS WORSE THAN AN OMISSION,
// and that is the whole argument: "unset" is a defined default that a future
// macOS would interpret sanely, while "written through an ABI we have evidence
// against" is 64 bytes of we-do-not-know-what sitting in the object waiting for
// a driver that starts reading it. Leaving them alone is the conservative
// action, not the lazy one.
//
// IF THEY EVER MATTER — a driver that starts using them, or a configuration
// this harness does not exercise (reactive masks, a different scale) — the way
// in is an ObjC++ shim function taking `const float*`, where the compiler
// generates the ABI instead of us guessing it. That is a few lines in a `.mm`
// file; what it costs is that this module stops being pure Rust and gains a
// build-time dependency it currently does not have, which is exactly the price
// worth paying THEN and not now.
// ---------------------------------------------------------------------------

impl MfxDn {
    /// Whether this macOS is new enough to HAVE the denoised scaler at all.
    ///
    /// Checked with a raw class lookup rather than by calling anything typed,
    /// because objc2's generated class reference panics when the class is
    /// missing — see the header. Every other entry point here goes through this
    /// first, so a macOS 15 box gets a SKIP and never a panic.
    pub fn available() -> bool {
        AnyClass::get(c"MTLFXTemporalDenoisedScalerDescriptor").is_some()
    }

    /// Whether this device supports it. False on a macOS that predates the API,
    /// which is what keeps the two questions one answer at the call site.
    pub fn supported(mtl: &Mtl) -> bool {
        Self::available()
            && unsafe { MTLFXTemporalDenoisedScalerDescriptor::supportsDevice(mtl.device()) }
    }

    /// The input-scale range this device supports, as `(min, max)`. `(0, 0)`
    /// when the API is absent — the caller has already SKIPped by then.
    pub fn scale_range(mtl: &Mtl) -> (f32, f32) {
        if !Self::available() {
            return (0.0, 0.0);
        }
        unsafe {
            (
                MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMinScaleForDevice(
                    mtl.device(),
                ),
                MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMaxScaleForDevice(
                    mtl.device(),
                ),
            )
        }
    }

    /// The normal-space lever, read once per construction.
    ///
    /// Loud on departure and loud+default on an illegal value — the FR_LEAF
    /// rule, which exists because a mistyped sweep cell that silently takes the
    /// default measures the shipping arm while believing it measured the lever.
    fn normals_mode() -> Normals {
        // ANNOUNCED ONCE PER PROCESS, not once per construction: the gate builds
        // several scalers per run, and eight identical lever lines read as a
        // malfunction rather than as a departure from the default.
        static SAID: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        let say = |m: String| {
            if SAID.set(()).is_ok() {
                eprintln!("{m}");
            }
        };
        match std::env::var("FR_MFXDN_NORMALS").as_deref() {
            Ok("world") => {
                say("mfxdn: FR_MFXDN_NORMALS=world (the default, spelled)".into());
                Normals::World
            }
            Ok("view") => {
                say("mfxdn: FR_MFXDN_NORMALS=view — normals transformed to view space".into());
                Normals::View
            }
            Ok(v) => {
                say(format!("mfxdn: FR_MFXDN_NORMALS={v:?} unrecognized — using world"));
                Normals::World
            }
            Err(_) => Normals::World,
        }
    }

    /// `render` is the EXACT input extent. Not a scoping choice here: the
    /// denoised protocol has no `inputContentWidth`, so there is no sub-rect
    /// mechanism to decline.
    pub fn new(
        mtl: &Mtl,
        render: (usize, usize),
        upscale: (usize, usize),
    ) -> Result<MfxDn, String> {
        let (rw, rh) = render;
        let (uw, uh) = upscale;
        if rw == 0 || rh == 0 || uw == 0 || uh == 0 {
            return Err(format!("zero extent: {rw}x{rh} -> {uw}x{uh}"));
        }
        if !Self::available() {
            return Err(
                "MTLFXTemporalDenoisedScaler is macOS 26.0+ and this system predates it".into()
            );
        }

        let desc = unsafe { MTLFXTemporalDenoisedScalerDescriptor::new() };
        unsafe {
            // Every format from `planes`, the same consts the allocation reads.
            desc.setColorTextureFormat(super::planes::COLOR);
            desc.setDepthTextureFormat(super::planes::DEPTH);
            desc.setMotionTextureFormat(super::planes::MOTION);
            desc.setNormalTextureFormat(super::planes::NORMAL);
            desc.setRoughnessTextureFormat(super::planes::ROUGHNESS);
            desc.setDiffuseAlbedoTextureFormat(super::planes::ALBEDO);
            desc.setSpecularAlbedoTextureFormat(super::planes::ALBEDO);
            desc.setSpecularHitDistanceTextureFormat(super::planes::HIT_DIST);
            desc.setOutputTextureFormat(super::planes::OUTPUT);
            desc.setInputWidth(rw);
            desc.setInputHeight(rh);
            desc.setOutputWidth(uw);
            desc.setOutputHeight(uh);
            // Optional input, and we have the plane — DLSS-RR's own use for it.
            desc.setSpecularHitDistanceTextureEnabled(true);
            // The two determinism preconditions — see `mtl::mfx`'s header.
            desc.setAutoExposureEnabled(false);
            desc.setRequiresSynchronousInitialization(true);
        }
        // READ THE OPTIONAL-INPUT FLAG BACK. It is the one descriptor property
        // here whose effect is invisible downstream: a scaler built with it off
        // still accepts `setSpecularHitDistanceTexture`, still reports
        // `ShaderRead` for that plane, and simply never reads it — so without
        // this the gate's own plumbing probe would be reporting on a request
        // that silently did not take, rather than on the wiring.
        if !unsafe { desc.isSpecularHitDistanceTextureEnabled() } {
            return Err(
                "the descriptor refused specularHitDistanceTextureEnabled — the optional \
                 specular-hit input cannot be wired on this system"
                    .into(),
            );
        }

        // Planes before the scaler, the `mfx.rs` ordering.
        let trio = super::planes::Trio::new(mtl, render)?;
        let guides = super::planes::Guides::new(mtl, render)?;
        let output = mtl
            .texture(uw, uh, super::planes::OUTPUT, MTLStorageMode::Private)
            .map_err(|e| format!("output plane: {e}"))?;
        let exposure = mtl
            .texture(1, 1, MTLPixelFormat::R16Float, MTLStorageMode::Shared)
            .map_err(|e| format!("exposure texture: {e}"))?;
        // f16 1.0 is 0x3C00; little-endian on every target this builds for.
        mtl.upload(&exposure, 1, 1, 2, &[0x00, 0x3C]);

        let scaler = unsafe { desc.newTemporalDenoisedScalerWithDevice(mtl.device()) }
            .ok_or_else(|| {
                format!(
                    "newTemporalDenoisedScalerWithDevice returned nil for {rw}x{rh} -> {uw}x{uh} \
                     — if `supported()` passed, then the DESCRIPTOR is what it rejected (a \
                     format or a scale ratio), not the device"
                )
            })?;

        Ok(MfxDn {
            scaler,
            trio,
            guides,
            output,
            exposure,
            render,
            upscale,
            normals: Self::normals_mode(),
        })
    }

    /// What the scaler reports it needs of each texture we bind, read off the
    /// created object rather than assumed (`--check-nrd`'s N1 discipline).
    pub fn required_usage(&self) -> [(&'static str, MTLTextureUsage); 9] {
        unsafe {
            [
                ("color", self.scaler.colorTextureUsage()),
                ("depth", self.scaler.depthTextureUsage()),
                ("motion", self.scaler.motionTextureUsage()),
                ("normal", self.scaler.normalTextureUsage()),
                ("roughness", self.scaler.roughnessTextureUsage()),
                ("diffuse-albedo", self.scaler.diffuseAlbedoTextureUsage()),
                ("specular-albedo", self.scaler.specularAlbedoTextureUsage()),
                ("specular-hit", self.scaler.specularHitDistanceTextureUsage()),
                ("output", self.scaler.outputTextureUsage()),
            ]
        }
    }

    /// Stage every input plane. `g` must be a FULL `GBufs` — see
    /// `planes::Guides::stage`.
    pub fn stage(
        &self,
        mtl: &Mtl,
        accum: &[AtomicU32],
        g: &crate::dlss::GBufs,
        near: f32,
        far: f32,
        world_to_view: glam::Mat4,
    ) {
        self.trio.stage(mtl, accum, g, self.render, near, far);
        self.guides.stage(mtl, g, self.render);
        if self.normals == Normals::View {
            // The lever's arm: re-stage ONLY the normal plane, transformed.
            // A second write rather than a parameter on `Guides::stage`,
            // because `planes.rs`'s ratchet rule is that shared staging must
            // not acquire a conditional — and because this is a diagnostic
            // path, where the cost of one extra upload is irrelevant and the
            // cost of complicating the shipping path is not.
            let (rw, rh) = self.render;
            let mut buf = vec![0u8; rw * rh * 8];
            crate::fsr::stage_normal_view(&mut buf, rw * 8, &g.normal_rough, rw, rh, world_to_view);
            mtl.upload(&self.guides.normal, rw, rh, rw * 8, &buf);
        }
    }

    /// Record one denoise+upscale, submit it, and block. Blocking for the same
    /// reason `mfx::Mfx::dispatch` blocks.
    pub fn dispatch(&self, mtl: &Mtl, p: &MfxDnParams) -> Result<(), String> {
        // Cleared so "the scaler wrote nothing" reads as zeros rather than as
        // whatever the allocator handed us; through a blit, because Private.
        mtl.clear_private(&self.output, self.upscale.0, self.upscale.1, 8, super::planes::OUTPUT)?;

        unsafe {
            self.scaler.setColorTexture(Some(&self.trio.color));
            self.scaler.setDepthTexture(Some(&self.trio.depth));
            self.scaler.setMotionTexture(Some(&self.trio.motion));
            self.scaler.setNormalTexture(Some(&self.guides.normal));
            self.scaler.setRoughnessTexture(Some(&self.guides.roughness));
            self.scaler.setDiffuseAlbedoTexture(Some(&self.guides.diff_alb));
            self.scaler.setSpecularAlbedoTexture(Some(&self.guides.spec_alb));
            self.scaler.setSpecularHitDistanceTexture(Some(&self.guides.hit_dist));
            self.scaler.setOutputTexture(Some(&self.output));
            self.scaler.setExposureTexture(Some(&self.exposure));
            self.scaler.setDepthReversed(true);
            self.scaler.setJitterOffsetX(p.jitter.0);
            self.scaler.setJitterOffsetY(p.jitter.1);
            self.scaler.setMotionVectorScaleX(crate::fsr::UPSCALE_MV_SIGN.0);
            self.scaler.setMotionVectorScaleY(crate::fsr::UPSCALE_MV_SIGN.1);
            // RENAMED from the plain scaler's `setReset`, and written every
            // frame in both directions for the reason `mfx.rs` states: it is
            // persistent state, so a stuck `true` never accumulates and a stuck
            // `false` reads uninitialized history on frame 0.
            self.scaler.setShouldResetHistory(p.reset);
        }

        mtl.run(|cb: &ProtocolObject<dyn MTLCommandBuffer>| unsafe {
            self.scaler.encodeToCommandBuffer(cb);
        })
    }

    /// The reconstructed frame as linear f32 RGBA, row-major at the upscale
    /// extent.
    pub fn read_output(&self, mtl: &Mtl) -> Result<Vec<f32>, String> {
        let (w, h) = self.upscale;
        let bytes = mtl.read_private(&self.output, w, h, 8, super::planes::OUTPUT)?;
        Ok((0..w * h * 4)
            .map(|i| f32::from(half::f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])))
            .collect())
    }

    /// The raw f16 bit patterns, for the determinism comparison — an INTEGER
    /// ULP distance, never a tunable relative bound (`mfx.rs`'s rule).
    pub fn read_output_bits(&self, mtl: &Mtl) -> Result<Vec<u16>, String> {
        let (w, h) = self.upscale;
        let bytes = mtl.read_private(&self.output, w, h, 8, super::planes::OUTPUT)?;
        Ok((0..w * h * 4).map(|i| u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])).collect())
    }
}
