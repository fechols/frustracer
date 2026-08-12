//! MetalFX frame interpolation — `MTLFXFrameInterpolator`, the fourth thing the
//! Windows path has and this one did not.
//!
//! Same framework, same crate and the same macOS 26.0 floor as
//! `mtl::mfxdn`, so the availability probe and its reasoning are that module's
//! — see its header on why this is a runtime `AnyClass::get` and not a build
//! gate.
//!
//! # What it takes that nothing else here does
//!
//! `prevColorTexture` beside `colorTexture` — it interpolates BETWEEN two
//! rendered frames rather than reconstructing one — plus a real camera
//! description: `deltaTime`, `nearPlane`, `farPlane`, `fieldOfView`,
//! `aspectRatio`. Those five are ordinary scalars with documented meanings, so
//! unlike the denoised scaler's two `simd_float4x4` properties there is no ABI
//! question to answer and no finding to record.
//!
//! Its descriptor also accepts a `scaler` (`MTLFXFrameInterpolatableScaler`,
//! which both temporal scalers conform to). That is NOT optional in practice —
//! see below — so one is always attached.
//!
//! # What a headless harness can prove about it, which is less than planned
//!
//! Frame generation's product is presented CADENCE, and this harness has no
//! presentation. The plan was to get around that with a ground truth an
//! in-between frame does have — render the midpoint pose, and require the
//! interpolation of A->B to land closer to M than a 50/50 blend of A and B
//! does, which is the vs-bilinear anti-vacuity in another currency.
//!
//! IT DOES NOT WORK, AND THE MEASUREMENT IS WHY. On this driver the
//! interpolator returns the CURRENT frame: `--check-metalfx`'s X6 measures the
//! output at 5.3e-5 from frame B against an A-to-B distance of 8.0e-2. Three
//! configurations give byte-identical results — a single reset dispatch, a
//! primed pair, and a driven `MTLFXTemporalScaler` attached through `scaler` —
//! and the plant that settles it is handing the interpolator frame B as its
//! PREVIOUS frame, which changes nothing: `prevColorTexture` is not read.
//!
//! So interpolation appears to be tied to the presentation path, and X6 asserts
//! the wiring (it creates, accepts our formats and usages, dispatches, and
//! writes a finite non-empty frame) while REPORTING the quality numbers with
//! the reason they are not assertions. A green X6 means this is ready for a
//! presentation stage, not that generation works — and the day the output stops
//! being a copy of its input, the midpoint claim is sitting there to promote.

use objc2::rc::Retained;
use objc2::runtime::{AnyClass, ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLStorageMode, MTLTexture, MTLTextureUsage};
use objc2_metal_fx::{
    MTLFXFrameInterpolator, MTLFXFrameInterpolatorBase, MTLFXFrameInterpolatorDescriptor,
};
use std::sync::atomic::AtomicU32;

use super::device::Mtl;

/// Per-dispatch state. `jitter` is ALREADY signed, the contract every other
/// Metal driver here states.
pub struct FiParams {
    pub jitter: (f32, f32),
    pub reset: bool,
    /// Seconds between the two rendered frames. A real number rather than a
    /// nominal one would make a gate's settings depend on machine load, so the
    /// caller passes a fixed value — `nrd_gpu::NOMINAL_DT_MS`' reasoning.
    pub delta_time: f32,
    pub near: f32,
    pub far: f32,
    pub fov_y: f32,
}

pub struct Mfxfi {
    interp: Retained<ProtocolObject<dyn MTLFXFrameInterpolator>>,
    trio: super::planes::Trio,
    /// The PREVIOUS frame's colour. Its own texture rather than a second
    /// `Trio`, because only the colour plane has a previous — the depth and
    /// motion the interpolator reads are the CURRENT frame's.
    prev_color: Retained<ProtocolObject<dyn MTLTexture>>,
    output: Retained<ProtocolObject<dyn MTLTexture>>,
    render: (usize, usize),
    upscale: (usize, usize),
}

impl Mfxfi {
    /// Whether this macOS has the interpolator at all — `mtl::mfxdn`'s probe,
    /// for its reason (objc2's class reference panics when the class is
    /// missing).
    pub fn available() -> bool {
        AnyClass::get(c"MTLFXFrameInterpolatorDescriptor").is_some()
    }

    pub fn supported(mtl: &Mtl) -> bool {
        Self::available()
            && unsafe { MTLFXFrameInterpolatorDescriptor::supportsDevice(mtl.device()) }
    }

    pub fn new(
        mtl: &Mtl,
        render: (usize, usize),
        upscale: (usize, usize),
        scaler: Option<&ProtocolObject<dyn objc2_metal_fx::MTLFXFrameInterpolatableScaler>>,
    ) -> Result<Mfxfi, String> {
        let (rw, rh) = render;
        let (uw, uh) = upscale;
        if rw == 0 || rh == 0 || uw == 0 || uh == 0 {
            return Err(format!("zero extent: {rw}x{rh} -> {uw}x{uh}"));
        }
        if !Self::available() {
            return Err("MTLFXFrameInterpolator is macOS 26.0+ and this system predates it".into());
        }

        let desc = unsafe { MTLFXFrameInterpolatorDescriptor::new() };
        unsafe {
            desc.setColorTextureFormat(super::planes::COLOR);
            desc.setDepthTextureFormat(super::planes::DEPTH);
            desc.setMotionTextureFormat(super::planes::MOTION);
            desc.setOutputTextureFormat(super::planes::OUTPUT);
            desc.setInputWidth(rw);
            desc.setInputHeight(rh);
            desc.setOutputWidth(uw);
            desc.setOutputHeight(uh);
            // NOT OPTIONAL IN PRACTICE — see the header. Without it the
            // interpolator ignores `prevColorTexture` entirely and returns the
            // current colour.
            desc.setScaler(scaler);
        }

        let trio = super::planes::Trio::new(mtl, render)?;
        let prev_color = mtl
            .texture(rw, rh, super::planes::COLOR, MTLStorageMode::Shared)
            .map_err(|e| format!("prev colour plane: {e}"))?;
        // PRIVATE, the storage rule every MetalFX output carries.
        let output = mtl
            .texture(uw, uh, super::planes::OUTPUT, MTLStorageMode::Private)
            .map_err(|e| format!("output plane: {e}"))?;

        let interp = unsafe { desc.newFrameInterpolatorWithDevice(mtl.device()) }.ok_or_else(
            || {
                format!(
                    "newFrameInterpolatorWithDevice returned nil for {rw}x{rh} -> {uw}x{uh} — if \
                     `supported()` passed, the DESCRIPTOR is what it rejected"
                )
            },
        )?;

        Ok(Mfxfi { interp, trio, prev_color, output, render, upscale })
    }

    /// What the interpolator says it needs of each texture we bind — read off
    /// the object, the `--check-nrd` N1 discipline.
    pub fn required_usage(&self) -> [(&'static str, MTLTextureUsage); 5] {
        unsafe {
            [
                ("color", self.interp.colorTextureUsage()),
                ("depth", self.interp.depthTextureUsage()),
                ("motion", self.interp.motionTextureUsage()),
                ("ui", self.interp.uiTextureUsage()),
                ("output", self.interp.outputTextureUsage()),
            ]
        }
    }

    /// Stage the PREVIOUS frame's colour alone.
    pub fn stage_prev(&self, mtl: &Mtl, accum: &[AtomicU32]) {
        let (rw, rh) = self.render;
        let mut buf = vec![0u8; rw * rh * 8];
        crate::fsr::stage_color(&mut buf, rw * 8, accum, rw, rh);
        mtl.upload(&self.prev_color, rw, rh, rw * 8, &buf);
    }

    /// Stage the CURRENT frame's colour, depth and motion.
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

    /// Record one interpolation, submit it, and block — the reason every other
    /// Metal driver here blocks.
    pub fn dispatch(&self, mtl: &Mtl, p: &FiParams) -> Result<(), String> {
        mtl.clear_private(&self.output, self.upscale.0, self.upscale.1, 8, super::planes::OUTPUT)?;
        let (uw, uh) = self.upscale;
        unsafe {
            self.interp.setColorTexture(Some(&self.trio.color));
            self.interp.setPrevColorTexture(Some(&self.prev_color));
            self.interp.setDepthTexture(Some(&self.trio.depth));
            self.interp.setMotionTexture(Some(&self.trio.motion));
            self.interp.setOutputTexture(Some(&self.output));
            self.interp.setDepthReversed(true);
            self.interp.setJitterOffsetX(p.jitter.0);
            self.interp.setJitterOffsetY(p.jitter.1);
            self.interp.setMotionVectorScaleX(crate::fsr::UPSCALE_MV_SIGN.0);
            self.interp.setMotionVectorScaleY(crate::fsr::UPSCALE_MV_SIGN.1);
            self.interp.setDeltaTime(p.delta_time);
            self.interp.setNearPlane(p.near);
            self.interp.setFarPlane(p.far);
            self.interp.setFieldOfView(p.fov_y);
            self.interp.setAspectRatio(uw as f32 / uh as f32);
            // Persistent state, written every frame in both directions — the
            // rule `mtl::mfx` states for `setReset`.
            self.interp.setShouldResetHistory(p.reset);
        }
        mtl.run(|cb: &ProtocolObject<dyn MTLCommandBuffer>| unsafe {
            self.interp.encodeToCommandBuffer(cb);
        })
    }

    /// The interpolated frame as linear f32 RGBA at the output extent.
    pub fn read_output(&self, mtl: &Mtl) -> Result<Vec<f32>, String> {
        let (w, h) = self.upscale;
        let bytes = mtl.read_private(&self.output, w, h, 8, super::planes::OUTPUT)?;
        Ok((0..w * h * 4)
            .map(|i| f32::from(half::f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])))
            .collect())
    }
}
