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
//!
//! # D5 — the same scaler through the Metal 4 submission path
//!
//! **THE FORK IS TWO LINES, AND THAT IS A MEASUREMENT OFF THE BINDINGS.**
//! `MTLFXTemporalScalerBase` carries all 40 members this module touches — every
//! texture setter, the jitter, the MV scale, `reset`, `depthReversed`, the
//! `*TextureUsage` getters — and `MTLFXTemporalScaler` and
//! `MTL4FXTemporalScaler` each add exactly `encodeToCommandBuffer:` and nothing
//! else. There is no `MTL4FXTemporalScalerDescriptor` either: one descriptor
//! class, and the factory selector decides which world you get. So `describe`
//! and `configure` are shared VERBATIM, and the arms differ only in
//! `newTemporalScalerWithDevice:` vs `…:compiler:` and which command-buffer
//! type the encode takes. `--check-metalfx` X8's byte-compare is what that
//! buys: 929616 channels IDENTICAL across the two APIs, because there is no
//! second copy of our configuration for a difference to hide in.
//!
//! **RESIDENCY IS REQUIRED, LOUDLY, AND THAT REVERSED THE PREDICTION.** MTL4
//! has no `useResource:`, so the five textures go into an `MTLResidencySet`.
//! The rung was planned expecting a residency mistake to present as WRONG
//! PIXELS — raw-address binding makes bound-but-not-resident a use-after-free,
//! and `--check-mtl`'s `FR_MTL4_NO_RESIDENCY` is a MEASUREMENT precisely
//! because unified memory hides it there. Here it is a TOOTH: armed,
//! `FR_MFX4_NO_RESIDENCY` makes MetalFX write NOTHING, the output keeps its
//! pre-dispatch clear, and X8 fails on the data with no validation layer
//! involved. MetalFX evidently checks residency in a way our hand-built
//! argument table cannot, which is a fact about MetalFX and should not be
//! generalised to the next MTL4 consumer.
//!
//! **METAL API VALIDATION ABORTS ON APPLE'S SIDE OF IT, so `--check-metalfx`
//! X7-X8 SKIP under `MTL_DEBUG_LAYER=1`.** MetalFX's own Metal 4 effect fails
//! an internal assertion — `Metal4FXTemporalScalingEffectV4.mm:561:
//! _outputTextureBarrierStages not set` — and the macOS 26 SDK exposes no
//! property that could satisfy it (40 properties across the base protocol and
//! the descriptor; not one names a barrier or a stage). `MTL_SHADER_VALIDATION
//! =1` is unaffected and runs the whole arm, so this path keeps the layer that
//! checks code we author; what it loses is the one that would catch a
//! texture-usage or storage-mode mistake of ours. That is a real hole and is
//! said rather than hidden. Same shape as X6's threadgroup-limit skip.
//!
//! **SCOPED TO THIS MODULE ON PURPOSE.** `mfxdn` and `mfxfi` keep their Metal 3
//! route until a rung measures them — this backend's stated rule that two
//! unproven things in one rung is not how it got built. `Cargo.toml` names
//! exactly one MTL4FX feature for the same reason.
//!
//! **AND IT RUNS ON EVERY APPLE SILICON MAC, OR SKIPS SAYING WHY.**
//! `supportsMetal4FX:` and `Mtl::mtl4()` are separate runtime probes with
//! separate SKIP messages, so a Mac without Metal 4 — including every CI
//! runner today — takes a loud skip rather than a failure. The bit-identity
//! claim above was measured on an M1 only, so X8 prints `Mtl::line()` on both
//! the pass and the failure: a green run on an M3 or M4 extends the claim, and
//! a red one is a per-generation finding rather than a bound to widen.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLCommandBuffer, MTLPixelFormat, MTLResidencySet, MTLStorageMode, MTLTexture,
    MTLTextureUsage,
};
use objc2_metal_fx::{
    MTL4FXTemporalScaler, MTLFXTemporalScaler, MTLFXTemporalScalerBase,
    MTLFXTemporalScalerDescriptor,
};
use std::sync::atomic::AtomicU32;

use super::device::{Mtl, Mtl4};

/// Which submission API a scaler was built for.
///
/// **NOT A QUALITY SETTING AND NOT A PREFERENCE** — it is which of two
/// factories made the object, and therefore which `encodeToCommandBuffer:`
/// overload it answers to. Everything between those two points is shared.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Api {
    /// `newTemporalScalerWithDevice:` — `MTLCommandBuffer`, `Mtl::run`.
    Three,
    /// `newTemporalScalerWithDevice:compiler:` — `MTL4CommandBuffer`,
    /// `Mtl4::submit`, and an explicit residency set.
    Four,
}

/// The scaler itself, which is the ONLY thing D5 forks.
///
/// **THE TWO PROTOCOLS SHARE EVERY MEMBER BUT ONE.** Reading the bindings
/// rather than assuming: `MTLFXTemporalScalerBase` carries all 40 of them —
/// every texture setter, `jitterOffsetX/Y`, `motionVectorScaleX/Y`, `reset`,
/// `depthReversed`, `inputContentWidth/Height`, the four `*TextureUsage`
/// getters, and `setFence` — while `MTLFXTemporalScaler` and
/// `MTL4FXTemporalScaler` each add exactly `encodeToCommandBuffer:` and
/// nothing else. So `configure` below is written ONCE against the base
/// protocol and is not a shared-looking copy that could drift; the fork is two
/// factories and two encode calls, and `base()` is what keeps it that small.
enum Scaler {
    Three(Retained<ProtocolObject<dyn MTLFXTemporalScaler>>),
    Four(Retained<ProtocolObject<dyn MTL4FXTemporalScaler>>),
}

impl Scaler {
    /// The half of the object both APIs agree about.
    ///
    /// An upcast, not a conversion — `ProtocolObject::from_ref` on a
    /// supertrait, the same move `as_interpolatable` has always made.
    fn base(&self) -> &ProtocolObject<dyn MTLFXTemporalScalerBase> {
        match self {
            Scaler::Three(s) => ProtocolObject::from_ref(&**s),
            Scaler::Four(s) => ProtocolObject::from_ref(&**s),
        }
    }
}

/// Levers over the Metal 4 MetalFX arm — default off, loud when armed, and
/// bit-identical to the shipping path when unarmed.
///
/// **CLASSIFICATION IS DEFERRED TO MEASUREMENT, per the D2 lesson**: D4
/// predicted both of its splits the wrong way round, so nothing here is
/// written down as a TOOTH until `--check-metalfx` has been watched failing
/// with it armed. See the D5 entry in `docs/history/metal-backend.md` for the
/// signatures as measured.
#[derive(Clone, Copy, Default)]
pub struct Plant {
    /// Report the Metal 4 MetalFX path as absent on a box that has it, forcing
    /// the SKIP branch. The `FR_MTL4_OFF` idiom, and worth the same as that
    /// one: the skip is the only branch CI can ever take, so it is the only
    /// branch CI can regress, and nothing else exercises it.
    pub off: bool,
    /// Build the scaler but attach NO residency set.
    ///
    /// The direct test of the claim that drove this rung's design: MTL4 has no
    /// `useResource:` on any encoder, so a texture MetalFX reads must be made
    /// resident some other way or not at all. Whether MetalFX manages its own
    /// arguments — it builds internal resources we never see — is undocumented
    /// either way, which is why this is a lever and not an assumption.
    ///
    /// **A TOOTH, AND MEASURED TO BE ONE, WHICH INVERTS THE D4 RESULT.** Armed,
    /// the MTL4 dispatch writes NOTHING: the output keeps its pre-dispatch
    /// clear, X8's energy ratio reads 0.000x, and the gate fails — plain, with
    /// no validation layer, on the DATA. Its `--check-mtl` namesake
    /// `FR_MTL4_NO_RESIDENCY` behaves the opposite way and is classified a
    /// MEASUREMENT for it: on the smoke chain a missing set is unobservable
    /// except under `MTL_SHADER_VALIDATION=1`, because unified memory leaves
    /// the pages readable anyway.
    ///
    /// The difference is the answer to this rung's stated primary risk, and it
    /// is the reverse of the prediction: residency was expected to fail
    /// SILENTLY, as wrong pixels. MetalFX declines instead. That is a fact
    /// about MetalFX's own binding — it evidently checks, where our hand-built
    /// argument table cannot — and not about MTL4, so it should not be
    /// generalised to the next MTL4 consumer.
    pub no_residency: bool,
    /// Configure everything, then never call `encodeToCommandBuffer:`.
    ///
    /// The anti-vacuity floor for the whole MTL4 arm. The output is cleared
    /// before every dispatch, so an arm that encodes nothing leaves zeros and
    /// the gate's energy assertion must fail. If it does NOT fail, then
    /// something other than this encode is writing the output and the stage is
    /// scoring the wrong thing.
    pub no_encode: bool,
}

impl Plant {
    pub fn from_env() -> Plant {
        let on = |k: &str| std::env::var(k).is_ok_and(|v| v != "0");
        Plant {
            off: on("FR_MFX4_OFF"),
            no_residency: on("FR_MFX4_NO_RESIDENCY"),
            no_encode: on("FR_MFX4_NO_ENCODE"),
        }
    }

    pub fn any(&self) -> bool {
        self.off || self.no_residency || self.no_encode
    }

    /// The levers that MUST make the gate fail, which the verdict enforces.
    ///
    /// `off` is excluded for the reason `mtl4::Plant` gives for its own: its
    /// whole point is that the gate stays GREEN down the skip branch. The
    /// other two are TEETH and both were measured biting before being named
    /// here — `no_residency` against the prediction, see its field doc.
    pub fn must_fail(&self) -> bool {
        self.no_encode || self.no_residency
    }

    pub fn line(&self) -> String {
        let names = [
            (self.off, "FR_MFX4_OFF"),
            (self.no_residency, "FR_MFX4_NO_RESIDENCY"),
            (self.no_encode, "FR_MFX4_NO_ENCODE"),
        ];
        names.iter().filter(|(on, _)| *on).map(|(_, n)| *n).collect::<Vec<_>>().join(" ")
    }
}

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
    scaler: Scaler,
    /// The five textures made resident, on the MTL4 arm only.
    ///
    /// `None` on the Metal 3 arm, and structurally rather than by policy:
    /// Metal 3 hazard-tracks and infers residency from what was bound, so
    /// there is nothing for a set to do. `None` on the MTL4 arm too when
    /// `FR_MFX4_NO_RESIDENCY` is armed, which is the lever that measures
    /// whether the set was load-bearing.
    residency: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
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

        let desc = Mfx::describe(render, upscale);
        let (trio, output, exposure) = Mfx::planes(mtl, render, upscale)?;

        let scaler = unsafe { desc.newTemporalScalerWithDevice(mtl.device()) }
            .ok_or_else(|| Mfx::nil_scaler("newTemporalScalerWithDevice", render, upscale))?;

        Ok(Mfx {
            scaler: Scaler::Three(scaler),
            residency: None,
            trio,
            output,
            exposure,
            render,
            upscale,
        })
    }

    /// The same scaler, built to encode into a **Metal 4** command buffer.
    ///
    /// **THE FORK IS THIS FUNCTION AND THE ENCODE CALL, AND NOTHING ELSE** —
    /// which is a measurement off the bindings rather than a hope. The
    /// descriptor is the SAME CLASS with the same properties (there is no
    /// `MTL4FXTemporalScalerDescriptor`), so `describe` is shared verbatim and
    /// the two arms cannot disagree about a format, an extent, auto-exposure
    /// or synchronous initialization. What differs is the factory selector,
    /// which takes an `MTL4Compiler`, and what it hands back: a protocol whose
    /// only added member is an `encodeToCommandBuffer:` over the other buffer
    /// type. See `Scaler`.
    ///
    /// # Residency, which is the part the bindings do NOT settle
    ///
    /// MTL4 removed `useResource:` from every encoder, so nothing infers
    /// residency from what was bound — `mtl4.rs` is built entirely around that.
    /// The five textures here are OURS, so they get a residency set. But
    /// MetalFX also allocates internal resources we never see and cannot name,
    /// and no binding says who makes those resident. So the set covers what we
    /// can cover, `FR_MFX4_NO_RESIDENCY` measures whether it was load-bearing,
    /// and neither this comment nor the gate claims more than that.
    pub fn new_mtl4(
        mtl: &Mtl,
        g: &Mtl4,
        render: (usize, usize),
        upscale: (usize, usize),
        plant: Plant,
    ) -> Result<Mfx, String> {
        let (rw, rh) = render;
        let (uw, uh) = upscale;
        if rw == 0 || rh == 0 || uw == 0 || uh == 0 {
            return Err(format!("zero extent: {rw}x{rh} -> {uw}x{uh}"));
        }

        let desc = Mfx::describe(render, upscale);
        let (trio, output, exposure) = Mfx::planes(mtl, render, upscale)?;

        // THE ONE LINE THAT IS NOT THE METAL 3 ARM'S. `MTL4Compiler` lives on
        // the `Mtl4` handle because that is the only place it can be built —
        // see `Mtl::mtl4`, which explains why touching its descriptor earlier
        // would panic on a Mac without Metal 4.
        let scaler = unsafe {
            desc.newTemporalScalerWithDevice_compiler(mtl.device(), g.compiler())
        }
        .ok_or_else(|| Mfx::nil_scaler("newTemporalScalerWithDevice:compiler:", render, upscale))?;

        // ALL FIVE, and the exposure texture is not an afterthought: the
        // scaler reads it every dispatch (auto-exposure is off, so the 1x1
        // holding 1.0 is a real input, not a placeholder). Listing four and
        // calling it done is the shape of bug this whole rung is about.
        let allocs: Vec<&ProtocolObject<dyn MTLAllocation>> = vec![
            ProtocolObject::from_ref(&*trio.color),
            ProtocolObject::from_ref(&*trio.depth),
            ProtocolObject::from_ref(&*trio.motion),
            ProtocolObject::from_ref(&*output),
            ProtocolObject::from_ref(&*exposure),
        ];
        let residency = if plant.no_residency {
            None
        } else {
            let (set, n) = g.residency(mtl, &allocs)?;
            // EXACT, not a floor, and for the reason `mtl4::pass` learned the
            // hard way: `MTLResidencySet` DEDUPLICATES, so a count compared
            // against the number of calls fails on correct code. These five
            // are distinct textures, so the two numbers agree — and asserting
            // it here is what keeps a set that silently came back empty from
            // reading like one that covered everything.
            if n != allocs.len() {
                return Err(format!(
                    "the residency set took {n} of {} allocations — the MTL4 arm binds by raw \
                     address, so a texture that is bound but not resident is a use-after-free \
                     that presents as wrong pixels rather than as an error",
                    allocs.len()
                ));
            }
            Some(set)
        };

        Ok(Mfx {
            scaler: Scaler::Four(scaler),
            residency,
            trio,
            output,
            exposure,
            render,
            upscale,
        })
    }

    /// The descriptor, shared VERBATIM by both arms.
    ///
    /// There is no `MTL4FXTemporalScalerDescriptor` — Apple uses one
    /// descriptor class and decides the world at the factory — so this is not
    /// a deduplication but a structural guarantee that the two arms describe
    /// the same scaler. Every property here is load-bearing and documented
    /// where it is set.
    fn describe(
        render: (usize, usize),
        upscale: (usize, usize),
    ) -> Retained<MTLFXTemporalScalerDescriptor> {
        let desc = unsafe { MTLFXTemporalScalerDescriptor::new() };
        unsafe {
            // The formats come from `planes`, the same consts the allocation
            // reads, so declaration and allocation cannot drift.
            desc.setColorTextureFormat(super::planes::COLOR);
            desc.setDepthTextureFormat(super::planes::DEPTH);
            desc.setMotionTextureFormat(super::planes::MOTION);
            desc.setOutputTextureFormat(super::planes::OUTPUT);
            desc.setInputWidth(render.0);
            desc.setInputHeight(render.1);
            desc.setOutputWidth(upscale.0);
            desc.setOutputHeight(upscale.1);
            // Both of these are determinism preconditions — see the header.
            desc.setAutoExposureEnabled(false);
            desc.setRequiresSynchronousInitialization(true);
        }
        desc
    }

    /// The five textures, allocated BEFORE the scaler so that a scaler which
    /// creates successfully is never stranded by a later allocation failure.
    ///
    /// (`Mfx` has no `Drop` beyond the automatic `Retained` releases, so this
    /// is ordering hygiene rather than a leak fix — unlike `Fsr3::new`, whose
    /// FFX handle really would leak. It matters more on the MTL4 arm, where a
    /// half-built `Mfx` would also strand a residency set attached to a queue.)
    #[allow(clippy::type_complexity)]
    fn planes(
        mtl: &Mtl,
        render: (usize, usize),
        upscale: (usize, usize),
    ) -> Result<
        (
            super::planes::Trio,
            Retained<ProtocolObject<dyn MTLTexture>>,
            Retained<ProtocolObject<dyn MTLTexture>>,
        ),
        String,
    > {
        let trio = super::planes::Trio::new(mtl, render)?;
        let output = mtl
            .texture(upscale.0, upscale.1, super::planes::OUTPUT, MTLStorageMode::Private)
            .map_err(|e| format!("output plane: {e}"))?;
        let exposure = mtl
            .texture(1, 1, MTLPixelFormat::R16Float, MTLStorageMode::Shared)
            .map_err(|e| format!("exposure texture: {e}"))?;
        // f16 1.0 is 0x3C00; little-endian on every target this builds for.
        mtl.upload(&exposure, 1, 1, 2, &[0x00, 0x3C]);
        Ok((trio, output, exposure))
    }

    /// One message for both factories returning nil, which is the same
    /// diagnosis either way: `supported()` already cleared the DEVICE, so what
    /// was rejected is the descriptor.
    fn nil_scaler(what: &str, render: (usize, usize), upscale: (usize, usize)) -> String {
        format!(
            "{what} returned nil for {}x{} -> {}x{} ({:?}/{:?}/{:?} -> {:?}) — if `supported()` \
             passed, then the DESCRIPTOR is what it rejected (a format or a scale ratio), not \
             the device",
            render.0,
            render.1,
            upscale.0,
            upscale.1,
            super::planes::COLOR,
            super::planes::DEPTH,
            super::planes::MOTION,
            super::planes::OUTPUT,
        )
    }

    /// Whether this device has MetalFX temporal scaling at all.
    ///
    /// An environment fact, so a gate SKIPs on false rather than failing — and
    /// checking it BEFORE `new` is what lets `new`'s error say "the descriptor
    /// is what it rejected" instead of conflating the two.
    pub fn supported(mtl: &Mtl) -> bool {
        unsafe { MTLFXTemporalScalerDescriptor::supportsDevice(mtl.device()) }
    }

    /// Whether this device has MetalFX temporal scaling **through Metal 4**.
    ///
    /// A SECOND question, not a rephrasing of the first. `supportsDevice:` and
    /// `supportsMetal4FX:` are separate selectors and Apple documents them
    /// separately ("temporal scaling compatible with Metal 4"), so a device
    /// could answer yes to one and no to the other — and this gate must SKIP on
    /// that rather than fail, exactly as `Mtl::mtl4` returning `None` SKIPs.
    /// Nothing in this tree called it before D5.
    ///
    /// `FR_MFX4_OFF` forces a `false` here, which is the only way to exercise
    /// the SKIP branch on a box that has Metal 4 — and the branch CI takes
    /// every run, since `macos-latest` has no Metal 4 at all.
    pub fn supported_mtl4(mtl: &Mtl, plant: Plant) -> bool {
        if plant.off {
            return false;
        }
        unsafe { MTLFXTemporalScalerDescriptor::supportsMetal4FX(mtl.device()) }
    }

    /// Which API this scaler was built against.
    pub fn api(&self) -> Api {
        match self.scaler {
            Scaler::Three(_) => Api::Three,
            Scaler::Four(_) => Api::Four,
        }
    }

    /// How many allocations this arm made resident. `None` on Metal 3, where
    /// the question does not arise; `Some(0)` is a set that did not take.
    pub fn resident(&self) -> Option<usize> {
        self.residency.as_ref().map(|s| s.allocationCount())
    }

    /// Release the residency set, which only the MTL4 arm has.
    ///
    /// **ONLY WHERE THE CALLER CAN PROVE THE GPU IS DONE**, which is the rule
    /// `Mtl4::drop_residency` states and `mtl4::pass` follows: these textures
    /// are reachable by raw address, so revoking their backing while work is in
    /// flight is a use-after-free rather than a tidy-up. Every `dispatch4`
    /// blocks on commit feedback, so a caller that has returned from one has
    /// that proof; a caller that has not must leave the set alone.
    pub fn drop_residency(&self, g: &Mtl4) {
        if let Some(s) = &self.residency {
            g.drop_residency(s);
        }
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
        let b = self.scaler.base();
        unsafe {
            Usage {
                color: b.colorTextureUsage(),
                depth: b.depthTextureUsage(),
                motion: b.motionTextureUsage(),
                output: b.outputTextureUsage(),
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

        // CHECKED BEFORE ANY WORK, and reported rather than ignored. The two
        // protocols are distinct types, so this cannot be a silent no-op —
        // encoding nothing would leave the cleared output and read as "the
        // scaler produced black", the one failure this module's energy
        // assertion is least able to explain. The mismatch is a caller bug,
        // and its symptom should name itself. `dispatch4` says the mirror
        // image for the same reason.
        let Scaler::Three(scaler) = &self.scaler else {
            return Err(
                "Mfx::dispatch was called on a scaler built for Metal 4 — an \
                 MTL4FXTemporalScaler encodes only into an MTL4CommandBuffer; use \
                 Mfx::dispatch4"
                    .to_string(),
            );
        };

        self.configure(p);

        mtl.run(|cb: &ProtocolObject<dyn MTLCommandBuffer>| unsafe {
            scaler.encodeToCommandBuffer(cb);
        })
    }

    /// The same upscale, submitted through **Metal 4**.
    ///
    /// **EVERY PROPERTY IS WRITTEN BY THE SAME `configure` THE METAL 3 ARM
    /// USES**, against the shared base protocol, so this function is the
    /// encode and the submission and nothing else. That is the whole claim of
    /// the rung: if the two arms produce different pixels, the difference is
    /// in Apple's two encode paths, because there is no second copy of our
    /// configuration for it to be in.
    ///
    /// It blocks, like its twin, and for a better reason than the twin has:
    /// `Mtl4::submit` blocks on the commit feedback, which is also the only
    /// error channel MTL4 has (D4b). So a faulted upscale here reports what
    /// Metal said, where `Mtl::run` next door reads `cb.error()` and this
    /// path's D4 ancestor read nothing at all.
    pub fn dispatch4(&self, mtl: &Mtl, g: &Mtl4, p: &MfxParams, plant: Plant) -> Result<(), String> {
        let Scaler::Four(scaler) = &self.scaler else {
            return Err(
                "Mfx::dispatch4 was called on a scaler built for Metal 3 — it encodes only into \
                 an MTLCommandBuffer and cannot reach an MTL4CommandBuffer; use Mfx::dispatch"
                    .to_string(),
            );
        };

        // The Metal 3 blit, deliberately. The subject of this arm is the
        // SCALER's submission path, and clearing through the route the other
        // arm uses keeps the two dispatches starting from byte-identical
        // state — a cleared-by-MTL4 output would be one more difference to
        // rule out when the cross-API compare disagrees.
        mtl.clear_private(&self.output, self.upscale.0, self.upscale.1, 8, super::planes::OUTPUT)?;

        self.configure(p);

        g.submit(mtl, super::device::SubmitPlant::default(), |cb| {
            // A BRANCH, NOT A SUPPRESSED CALL. The armed arm encodes nothing
            // into a command buffer that is still begun, committed and waited
            // on, so what it measures is exactly "did this encode do the
            // work?" — with the submission machinery held constant. The
            // unarmed arm is one unconditional call, which is what keeps the
            // shipping path free of the lever.
            if !plant.no_encode {
                unsafe { scaler.encodeToCommandBuffer(cb) };
            }
            Ok(())
        })
        .map(|_| ())
    }

    /// Every per-dispatch property, written ONCE against the base protocol.
    ///
    /// **THE ONE COPY IS THE POINT.** All 40 of these live on
    /// `MTLFXTemporalScalerBase`, which both `MTLFXTemporalScaler` and
    /// `MTL4FXTemporalScaler` inherit, so the two submission arms are
    /// configured by the same code rather than by two copies that agree today.
    /// A property added here reaches both arms; a property added to one arm's
    /// dispatch would be a fork, and there is nowhere to put one.
    fn configure(&self, p: &MfxParams) {
        let s = self.scaler.base();
        unsafe {
            s.setColorTexture(Some(&self.trio.color));
            s.setDepthTexture(Some(&self.trio.depth));
            s.setMotionTexture(Some(&self.trio.motion));
            s.setOutputTexture(Some(&self.output));
            s.setExposureTexture(Some(&self.exposure));
            // Equal to the input extent by construction here — this arm does not
            // enable `inputContentPropertiesEnabled`, so there is no smaller
            // content rect to describe. Set anyway, because Apple's documented
            // per-frame sequence names them and a future dynamic-res arm would
            // change only these two lines.
            s.setInputContentWidth(self.render.0);
            s.setInputContentHeight(self.render.1);
            s.setDepthReversed(true);
            s.setJitterOffsetX(p.jitter.0);
            s.setJitterOffsetY(p.jitter.1);
            // THE MV CONVENTION, derived in this module's header from Apple's
            // own documentation and matching `GBufs::mvec` exactly, so the scale
            // is the bare sign — the same value both FSR3 arms pass.
            s.setMotionVectorScaleX(crate::fsr::UPSCALE_MV_SIGN.0);
            s.setMotionVectorScaleY(crate::fsr::UPSCALE_MV_SIGN.1);
            // PERSISTENT STATE, not a per-dispatch argument: `reset` is a
            // property on the scaler, so it must be written EVERY frame in both
            // directions. Leaving it true silently makes every frame a reset
            // (history never accumulates); leaving it false on the first frame
            // has the scaler read its own uninitialized history, which is
            // silent in a different and worse way.
            s.setReset(p.reset);
            // `setFence` IS STILL NOT SET, AND THAT NEEDED RE-CHECKING RATHER
            // THAN INHERITING. It lives on the BASE protocol, so an MTL4FX
            // object exposes a Metal 3 `MTLFence` setter — and the Metal 3
            // reason for leaving it alone (default hazard-tracked textures in
            // one command buffer, the case Apple's fence property explicitly
            // does not cover) is an argument about a world MTL4 removed. What
            // replaces it is not a fence: it is that `dispatch4` submits ONE
            // command buffer and blocks on its completion before anything
            // reads the output, so there is no second submission for a fence
            // to order against. If this arm ever pipelines, that sentence
            // stops being true and this is the line to revisit.
        }
    }

    /// This scaler as an `MTLFXFrameInterpolatableScaler`, for
    /// `MTLFXFrameInterpolatorDescriptor::setScaler`.
    ///
    /// The interpolator NEEDS one: standalone it is a passthrough — measured,
    /// see `mtl::mfxfi`'s header — and its descriptor's `scaler` property is
    /// how a real session chains generation onto reconstruction. Handing out a
    /// borrow rather than a clone keeps the lifetime obvious: the interpolator
    /// must not outlive the scaler it was built against.
    pub fn as_interpolatable(
        &self,
    ) -> &ProtocolObject<dyn objc2_metal_fx::MTLFXFrameInterpolatableScaler> {
        ProtocolObject::from_ref(self.scaler.base())
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
