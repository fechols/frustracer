//! MetalFX temporal upscaling — Apple's `MTLFXTemporalScaler` over the same
//! G-buffer `mtl::fsr3` feeds FidelityFX.
//!
//! **Why a second upscaler exists at all.** B2's own risk record says real depth
//! and real jitter were the untested path — the reference implementation this
//! tree ported upscaled VIDEO (flat depth, zero jitter, hardcoded camera) — and
//! that a gate's teeth "catch gross wiring, not polarity." A second,
//! independent consumer of the identical planes is stronger evidence than
//! looking at dump images: if both reconstruct and both respond to the
//! jitter/depth/motion probes, the conventions are almost certainly right, and
//! if one smears while the other does not, the disagreement localises to a
//! PLANE rather than to a renderer. `super::planes::Trio` is what makes
//! "identical" structural — see its header.
//!
//! **It also costs far less than the FSR3 arm, and for a structural reason.**
//! FidelityFX ships `ffx_vk` and `ffx_dx12` and nothing else, so Metal needed a
//! hand-written `FfxInterface` (`shim/ffx_fsr3_metal.mm`, ~1350 lines of
//! non-ARC ObjC++ this tree owns with no upstream) plus a build-time SPIR-V ->
//! MSL transpile of 80 shader permutations. MetalFX is a system framework:
//! no shim, no metallib table, no SDK fetch, and therefore **this module works
//! on a bare clone** where `mtl::fsr3` needs `install-prerequisites.sh fsr3src`
//! plus spirv-cross and Xcode at build time. That is why it carries no `cfg`
//! beyond macOS while `mtl::fsr3` is gated on `ffx_fsr3_metal`.
//!
//! # The conventions, and which of them are DERIVED
//!
//! * **Motion vectors — derived, not guessed.** Apple documents
//!   `motionVectorScaleX` at 1.0 as: "the motion vectors for an object that
//!   moves down and to the right in the `colorTexture` by 10 pixels would be
//!   `(-10,-10)`". `GBufs::mvec` stores `prev_px - cur_px` y-down, and an
//!   object moving down-right by 10 has `prev - cur = (-10,-10)`. Exact match,
//!   so the scale is the bare `fsr::UPSCALE_MV_SIGN` — the same value both FSR3
//!   arms pass, for the same reason.
//! * **Depth — derived.** `fsr::stage_depth` writes
//!   `xess::view_z_to_clip_depth`'s reversed-Z, where sky lands on exactly 0.0,
//!   so `setDepthReversed(true)` ("whether the depth texture uses zero to
//!   represent the farthest distance"). This is the exact twin of FSR3's
//!   `FLAG_DEPTH_INVERTED`, and getting it wrong inverts the whole disocclusion
//!   test. NOTE the gate cannot score it: the depth probe flattens the plane
//!   and asserts the output moved, which it does whichever way the flag points.
//! * **Jitter — not DOCUMENTED, but MEASURED.** Apple states the MV convention
//!   and says nothing about the jitter sign, so `JITTER_SIGN` below started as
//!   FFX's value on the reasoning that both take "the subpixel jitter offset
//!   applied to the camera" in pixels — a guess, with `FR_MFX_JITTER=raw|neg`
//!   to walk it, exactly as `FR_VK_FSR3_JITTER` does for the Vulkan arm.
//!   IT IS NO LONGER A GUESS. `--check-metalfx`'s X3 cross-check correlates
//!   this arm's deviation-from-bilinear against FSR3's, whose sign is already
//!   settled, and mirroring THIS one drops that correlation from 0.655 to
//!   0.479. So +1 is the measured answer, and the cross-check is the only
//!   assertion in either Metal gate that scores a polarity at all — the thing
//!   `--check-fsr3` says plainly that no magnitude of its own can do.
//!
//! # Two settings that are determinism preconditions, not preferences
//!
//! * `requiresSynchronousInitialization(true)`. Apple's default is FALSE, which
//!   "quickly create[s] and return[s]" a scaler and then "compile[s] a faster
//!   upscaler in the background". The gate creates several scalers in sequence
//!   and compares pairs of them; a background compile landing between two runs
//!   would mean comparing two different upscalers and calling the difference
//!   residue. Apple says image quality is "consistent" across the two — which
//!   is a quality claim, not a bit-identity one, and this gate asserts
//!   bit-identity.
//! * `autoExposureEnabled(false)` plus a 1x1 exposure texture holding 1.0.
//!   Auto-exposure would have MetalFX compute a per-frame gain that multiplies
//!   the input colour, with nothing documenting that it is un-applied on
//!   output — which would make the gate's energy assertion (output mean within
//!   [0.5, 2] of input mean) a measurement of Apple's exposure heuristic rather
//!   than of our wiring, and the temptation on failure would be to widen the
//!   one bound that catches colour-space mistakes. The FSR3 arm sets
//!   `FLAG_AUTO_EXPOSURE` and inherits exactly that ambiguity; this one does
//!   not, and the asymmetry is deliberate.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLPixelFormat, MTLStorageMode, MTLTexture, MTLTextureUsage,
};
use objc2_metal_fx::{
    MTLFXTemporalScaler, MTLFXTemporalScalerBase, MTLFXTemporalScalerDescriptor,
};
use std::sync::atomic::AtomicU32;

use super::device::Mtl;

/// Sign applied to the renderer's sample offset before it reaches
/// `setJitterOffsetX/Y`.
///
/// IT LIVES HERE AND NOT IN `src/fsr.rs`, deliberately. That module's header
/// says its polarity constants "live HERE and nowhere else" — but its subject
/// is the FFX WIRE, and MetalFX is a different vendor's wire with its own
/// conventions and its own documentation. Putting an Apple constant in the
/// FidelityFX module would break that module's stated scope; leaving this one
/// implicit would break the rule that a polarity has exactly one home. So it is
/// here, seeded at FFX's value because both take "the subpixel jitter offset
/// applied to the camera" in pixels, and walked by `FR_MFX_JITTER`.
pub const JITTER_SIGN: f32 = crate::fsr::JITTER_SIGN;

/// What the scaler says it needs of each texture, read back off the created
/// object rather than assumed. See `Mfx::required_usage`.
pub struct Usage {
    pub color: MTLTextureUsage,
    pub depth: MTLTextureUsage,
    pub motion: MTLTextureUsage,
    pub output: MTLTextureUsage,
}

/// Per-dispatch state. `jitter` is ALREADY multiplied by `JITTER_SIGN` — the
/// same contract `mtl::fsr3::DispatchParams` states, so the two arms cannot
/// disagree about whose job the sign is.
pub struct MfxParams {
    pub jitter: (f32, f32),
    pub reset: bool,
}

pub struct Mfx {
    scaler: Retained<ProtocolObject<dyn MTLFXTemporalScaler>>,
    trio: super::planes::Trio,
    /// PRIVATE storage, unlike everything else in this harness: Apple documents
    /// "You are responsible for providing a texture with a private
    /// `storageMode` to this property" for `outputTexture` alone. That is why
    /// `read_output` blits and why the pre-dispatch clear does too.
    output: Retained<ProtocolObject<dyn MTLTexture>>,
    /// 1x1 R16Float holding 1.0 — see the header on auto-exposure. Held for the
    /// scaler's whole life because the property is set per frame.
    exposure: Retained<ProtocolObject<dyn MTLTexture>>,
    render: (usize, usize),
    upscale: (usize, usize),
}

impl Mfx {
    /// `render` is the EXACT input extent, not a maximum.
    ///
    /// The FSR3 arm allocates at its maximum and dispatches sub-rects, because
    /// FFX takes a `renderSize` per dispatch. MetalFX has no such per-dispatch
    /// knob: it validates textures against the descriptor's dimensions, and its
    /// dynamic-resolution equivalent is a descriptor-level mechanism
    /// (`inputContentPropertiesEnabled` plus a scale range that must come from
    /// `supportedInputContentMinScaleForDevice`). Putting that untested path
    /// inside a gate whose job is to prove the static one works would be scope
    /// creep, so this arm is fixed-size and says so.
    pub fn new(
        mtl: &Mtl,
        render: (usize, usize),
        upscale: (usize, usize),
    ) -> Result<Mfx, String> {
        let (rw, rh) = render;
        let (uw, uh) = upscale;
        if rw == 0 || rh == 0 || uw == 0 || uh == 0 {
            return Err(format!("zero extent: {rw}x{rh} -> {uw}x{uh}"));
        }

        let desc = unsafe { MTLFXTemporalScalerDescriptor::new() };
        unsafe {
            // The formats come from `planes`, the same consts the allocation
            // reads, so declaration and allocation cannot drift.
            desc.setColorTextureFormat(super::planes::COLOR);
            desc.setDepthTextureFormat(super::planes::DEPTH);
            desc.setMotionTextureFormat(super::planes::MOTION);
            desc.setOutputTextureFormat(super::planes::OUTPUT);
            desc.setInputWidth(rw);
            desc.setInputHeight(rh);
            desc.setOutputWidth(uw);
            desc.setOutputHeight(uh);
            // Both of these are determinism preconditions — see the header.
            desc.setAutoExposureEnabled(false);
            desc.setRequiresSynchronousInitialization(true);
        }

        // The planes FIRST, so a scaler that creates successfully is never
        // stranded by a later allocation failure. (`Mfx` has no `Drop` beyond
        // the automatic `Retained` releases, so this is ordering hygiene rather
        // than a leak fix — unlike `Fsr3::new`, whose FFX handle really would
        // leak.)
        let trio = super::planes::Trio::new(mtl, render)?;
        let output = mtl
            .texture(uw, uh, super::planes::OUTPUT, MTLStorageMode::Private)
            .map_err(|e| format!("output plane: {e}"))?;
        let exposure = mtl
            .texture(1, 1, MTLPixelFormat::R16Float, MTLStorageMode::Shared)
            .map_err(|e| format!("exposure texture: {e}"))?;
        // f16 1.0 is 0x3C00; little-endian on every target this builds for.
        mtl.upload(&exposure, 1, 1, 2, &[0x00, 0x3C]);

        let scaler = unsafe { desc.newTemporalScalerWithDevice(mtl.device()) }.ok_or_else(|| {
            format!(
                "newTemporalScalerWithDevice returned nil for {rw}x{rh} -> {uw}x{uh} \
                 ({:?}/{:?}/{:?} -> {:?}) — if `supported()` passed, then the DESCRIPTOR is \
                 what it rejected (a format or a scale ratio), not the device",
                super::planes::COLOR,
                super::planes::DEPTH,
                super::planes::MOTION,
                super::planes::OUTPUT,
            )
        })?;

        Ok(Mfx { scaler, trio, output, exposure, render, upscale })
    }

    /// Whether this device has MetalFX temporal scaling at all.
    ///
    /// An environment fact, so a gate SKIPs on false rather than failing — and
    /// checking it BEFORE `new` is what lets `new`'s error say "the descriptor
    /// is what it rejected" instead of conflating the two.
    pub fn supported(mtl: &Mtl) -> bool {
        unsafe { MTLFXTemporalScalerDescriptor::supportsDevice(mtl.device()) }
    }

    /// The input-scale range this device supports, as `(min, max)`.
    ///
    /// Printed and asserted rather than discovered as an opaque nil from
    /// `newTemporalScalerWithDevice`: a ratio outside it is an environment
    /// limit with a number attached, not a mystery.
    pub fn scale_range(mtl: &Mtl) -> (f32, f32) {
        unsafe {
            (
                MTLFXTemporalScalerDescriptor::supportedInputContentMinScaleForDevice(mtl.device()),
                MTLFXTemporalScalerDescriptor::supportedInputContentMaxScaleForDevice(mtl.device()),
            )
        }
    }

    /// What the scaler reports it needs of each texture.
    ///
    /// Read off the created object rather than assumed — the discipline
    /// `--check-nrd`'s N1 applies to `spirv_binding_offsets`. `Mtl::texture`
    /// hands out `ShaderRead|ShaderWrite|RenderTarget`, and the gate asserts
    /// that is a SUPERSET of these. Note what such a check can and cannot see:
    /// it is about USAGE, and the requirement that actually bites on this API
    /// is the output's STORAGE MODE, which no usage comparison can reach.
    pub fn required_usage(&self) -> Usage {
        unsafe {
            Usage {
                color: self.scaler.colorTextureUsage(),
                depth: self.scaler.depthTextureUsage(),
                motion: self.scaler.motionTextureUsage(),
                output: self.scaler.outputTextureUsage(),
            }
        }
    }

    pub fn stage(
        &self,
        mtl: &Mtl,
        accum: &[AtomicU32],
        g: &crate::dlss::GBufs,
        near: f32,
        far: f32,
    ) {
        self.trio.stage(mtl, accum, g, self.render, near, far);
    }

    /// Record one upscale, submit it, and block until the GPU is done.
    ///
    /// Blocking for the same reason `Fsr3::dispatch` blocks: headless, so there
    /// is no next frame to overlap with and a fence ring would be apparatus
    /// with no consumer. It is also why `setFence` is left alone — these are
    /// default hazard-tracked textures in one command buffer, which is the case
    /// Apple's fence property explicitly does NOT cover ("your app's untracked
    /// resources").
    pub fn dispatch(&self, mtl: &Mtl, p: &MfxParams) -> Result<(), String> {
        // Cleared so that "the scaler wrote nothing" reads as zeros rather than
        // as whatever the allocator handed us — the assertion `--check-metalfx`
        // makes on the output mean. It goes through a blit because the output
        // is Private; dropping it on the grounds that `replaceRegion` no longer
        // applies would make that assertion silently untestable.
        mtl.clear_private(&self.output, self.upscale.0, self.upscale.1, 8, super::planes::OUTPUT)?;

        unsafe {
            self.scaler.setColorTexture(Some(&self.trio.color));
            self.scaler.setDepthTexture(Some(&self.trio.depth));
            self.scaler.setMotionTexture(Some(&self.trio.motion));
            self.scaler.setOutputTexture(Some(&self.output));
            self.scaler.setExposureTexture(Some(&self.exposure));
            // Equal to the input extent by construction here — this arm does not
            // enable `inputContentPropertiesEnabled`, so there is no smaller
            // content rect to describe. Set anyway, because Apple's documented
            // per-frame sequence names them and a future dynamic-res arm would
            // change only these two lines.
            self.scaler.setInputContentWidth(self.render.0);
            self.scaler.setInputContentHeight(self.render.1);
            self.scaler.setDepthReversed(true);
            self.scaler.setJitterOffsetX(p.jitter.0);
            self.scaler.setJitterOffsetY(p.jitter.1);
            // THE MV CONVENTION, derived in this module's header from Apple's
            // own documentation and matching `GBufs::mvec` exactly, so the scale
            // is the bare sign — the same value both FSR3 arms pass.
            self.scaler.setMotionVectorScaleX(crate::fsr::UPSCALE_MV_SIGN.0);
            self.scaler.setMotionVectorScaleY(crate::fsr::UPSCALE_MV_SIGN.1);
            // PERSISTENT STATE, not a per-dispatch argument: `reset` is a
            // property on the scaler, so it must be written EVERY frame in both
            // directions. Leaving it true silently makes every frame a reset
            // (history never accumulates); leaving it false on the first frame
            // has the scaler read its own uninitialized history, which is
            // silent in a different and worse way.
            self.scaler.setReset(p.reset);
        }

        mtl.run(|cb: &ProtocolObject<dyn MTLCommandBuffer>| unsafe {
            self.scaler.encodeToCommandBuffer(cb);
        })
    }

    /// The upscaled frame as linear f32 RGBA, row-major at the upscale extent.
    pub fn read_output(&self, mtl: &Mtl) -> Result<Vec<f32>, String> {
        let (w, h) = self.upscale;
        let bytes = mtl.read_private(&self.output, w, h, 8, super::planes::OUTPUT)?;
        Ok((0..w * h * 4)
            .map(|i| f32::from(half::f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])))
            .collect())
    }

    /// The raw f16 bit patterns of the output, for the determinism comparison.
    ///
    /// Separate from `read_output` because the honest determinism claim is an
    /// INTEGER one — "these two runs differ by at most one ULP" — and a
    /// relative bound on widened floats is a tunable number where an ULP
    /// distance is not.
    pub fn read_output_bits(&self, mtl: &Mtl) -> Result<Vec<u16>, String> {
        let (w, h) = self.upscale;
        let bytes = mtl.read_private(&self.output, w, h, 8, super::planes::OUTPUT)?;
        Ok((0..w * h * 4).map(|i| u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])).collect())
    }
}
