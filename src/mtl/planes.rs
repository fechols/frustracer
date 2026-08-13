//! The input planes the Metal upscalers consume — allocated, formatted and
//! staged in one place. `Trio` is the three every one of them reads;
//! `Guides` is the five more the DENOISED scaler reads, and it is a separate
//! struct for a reason its own doc gives.
//!
//! **This is a dedup, not an abstraction, and the distinction is what makes it
//! allowed.** `src/mtl/mod.rs` states the rule: share MATH and VOCABULARY,
//! duplicate the RECORDING *per API*, and two `Fsr3` implementations is the
//! correct shape where a `trait Upscaler` is not. That rule's unit is the API —
//! "the D3D12 recording (`gpu/ffx_up.rs::record_upload`) and the Metal one here
//! stay separate" — and `mtl::fsr3` and `mtl::mfx` are the SAME API, recording
//! the same `replaceRegion` calls against the same texture type from the same
//! `fsr::stage_*` encoders.
//!
//! But the rule exists because of a RATCHET rather than because of API
//! boundaries: shared code acquires parameters, and a `Trio` that grows an
//! `Option<reactive_mask>` or a flags word IS the forbidden trait spelled as a
//! struct. So: **if this ever needs a conditional, delete it and inline both
//! copies.** That sentence is the only thing keeping the rule honest.
//!
//! WHY IT OWNS THE TEXTURES AND NOT JUST THE STAGING. The two upscalers exist
//! to be read against each other — if both reconstruct and both respond to the
//! jitter/depth/motion probes, the G-buffer's conventions are almost certainly
//! right, and if one smears while the other does not, the disagreement
//! localises to a plane rather than to a renderer. That argument is worth
//! something only if the two see the SAME BYTES, and a first draft of this
//! module took the three textures as ARGUMENTS — which leaves format and extent
//! at two independent call sites agreeing with each other, i.e. exactly the
//! drift the sharing was supposed to remove. Owning the allocation makes the
//! invariant structural instead of aspirational.
//!
//! Nothing about the three ENCODINGS is decided here either. They are
//! `fsr::stage_color` / `stage_mvec` / `stage_depth` — the pure functions
//! `--check-fsr` gates on every platform, and the ones the D3D12 arm reuses the
//! leaves of.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLPixelFormat, MTLStorageMode, MTLTexture};
use std::sync::atomic::AtomicU32;

use super::device::Mtl;

/// THE WIRE FORMATS, and the reason they are constants rather than literals at
/// each call site: every consumer must DECLARE them to its SDK as well as
/// allocate them. `shim/ffx_fsr3_metal.mm` describes them to FFX, which
/// rebuilds its own views from that description and ignores what we hold; the
/// MetalFX descriptor names them in `setColorTextureFormat` and friends. A
/// mismatch between what is allocated and what is declared is read as garbage
/// rather than rejected — so allocation and declaration read the same const.
pub const COLOR: MTLPixelFormat = MTLPixelFormat::RGBA16Float;
pub const DEPTH: MTLPixelFormat = MTLPixelFormat::R32Float;
pub const MOTION: MTLPixelFormat = MTLPixelFormat::RG16Float;
/// Not part of the trio — the upscalers own their own output — but it is the
/// same wire question and belongs beside the others.
pub const OUTPUT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;

/// The three input planes at one extent, in `Shared` storage.
///
/// `Shared` is right for all three: MetalFX documents a storage requirement for
/// its OUTPUT alone (`device.rs::texture`), and FFX has none.
pub struct Trio {
    pub color: Retained<ProtocolObject<dyn MTLTexture>>,
    pub depth: Retained<ProtocolObject<dyn MTLTexture>>,
    pub motion: Retained<ProtocolObject<dyn MTLTexture>>,
    extent: (usize, usize),
}

impl Trio {
    /// Allocate at `extent`, which is the MAXIMUM the planes will hold.
    ///
    /// `mtl::fsr3` passes its max render extent and dispatches smaller
    /// sub-rects into the same planes (the discipline `gpu/ffx_up.rs` states
    /// for D3D12); `mtl::mfx` passes its exact extent, because MetalFX
    /// validates textures against its descriptor's dimensions and its
    /// sub-rect equivalent is a different, descriptor-level mechanism
    /// (`inputContentPropertiesEnabled`). Both work against `stage` below.
    pub fn new(mtl: &Mtl, extent: (usize, usize)) -> Result<Trio, String> {
        let (w, h) = extent;
        let t = |f: MTLPixelFormat, what: &str| {
            mtl.texture(w, h, f, MTLStorageMode::Shared).map_err(|e| format!("{what}: {e}"))
        };
        Ok(Trio {
            color: t(COLOR, "color plane")?,
            depth: t(DEPTH, "depth plane")?,
            motion: t(MOTION, "motion plane")?,
            extent,
        })
    }

    /// Stage the input trio into the planes' top-left `rw x rh` sub-rect.
    ///
    /// The pitches are tight and the DESTINATION region is what makes this a
    /// sub-rect, not the source packing: `replaceRegion` takes an arbitrary
    /// `bytesPerRow`, so there is no 256-byte rule to honour as in D3D12.
    pub fn stage(
        &self,
        mtl: &Mtl,
        accum: &[AtomicU32],
        g: &crate::dlss::GBufs,
        render: (usize, usize),
        near: f32,
        far: f32,
    ) {
        let (rw, rh) = render;

        // BOTH PRECONDITIONS BELOW ARE REAL ASSERTS, and the reason is the same
        // one: `debug_assert` IS DEAD IN THIS TREE. It is compiled out of
        // `[profile.release]` and of `[profile.quick]`, which inherits it, so a
        // `debug_assert` here has never executed in any build this project runs,
        // CI included. Two integer tests per frame against a staging copy that
        // touches every pixel is not a cost worth reasoning about; a guard that
        // provably never runs is not a guard.
        //
        // EXTENT: the planes are allocated once at a maximum and every caller
        // dispatches a sub-rect into them, so `render` larger than `extent`
        // hands `replaceRegion` an out-of-bounds destination — a validation
        // abort under `MTL_DEBUG_LAYER=1` and undefined behaviour without it.
        // Both call sites are correct today; this is what keeps a third one
        // from being wrong quietly.
        assert!(
            rw <= self.extent.0 && rh <= self.extent.1,
            "stage extent {rw}x{rh} exceeds the planes' {}x{}",
            self.extent.0,
            self.extent.1,
        );

        // One scratch, reused: colour is the widest at 8 B/px, so the two
        // 4 B/px planes take a prefix of it.
        let mut buf = vec![0u8; rw * rh * 8];

        // ALIGNMENT: `fsr::stage_*` casts each row to `[f16; N]` / `[f32]` and
        // debug-asserts the pointer and pitch are aligned to that element —
        // i.e. its own guard is one of the dead ones above. A `Vec<u8>` promises
        // alignment 1; the system allocator's real answer for these sizes is 16,
        // which is why the casts are sound today. That is luck the type system
        // does not express, and this is the one place a future "pack all three
        // planes into one buffer at offsets" refactor would break it.
        assert!(
            buf.as_ptr() as usize % 4 == 0,
            "staging scratch is {}-aligned; fsr::stage_* casts rows to f32/f16",
            1 << (buf.as_ptr() as usize).trailing_zeros().min(8),
        );

        crate::fsr::stage_color(&mut buf, rw * 8, accum, rw, rh);
        mtl.upload(&self.color, rw, rh, rw * 8, &buf);

        crate::fsr::stage_mvec(&mut buf[..rw * rh * 4], rw * 4, &g.mvec, rw, rh);
        mtl.upload(&self.motion, rw, rh, rw * 4, &buf[..rw * rh * 4]);

        crate::fsr::stage_depth(&mut buf[..rw * rh * 4], rw * 4, &g.depth, rw, rh, near, far);
        mtl.upload(&self.depth, rw, rh, rw * 4, &buf[..rw * rh * 4]);
    }
}

/// The DENOISER guide formats. Same rule as the trio's above — allocated and
/// declared from one const, because a mismatch reads as garbage.
///
/// `NORMAL` and `ALBEDO` are `RGBA16Float` rather than a 3-component format
/// because Metal has no RGB16Float: 3-channel pixel formats do not exist in the
/// API at all. The fourth channel is therefore not a choice we are making, and
/// `fsr::stage_normal`/`stage_albedo` zero it rather than leave it undefined.
pub const NORMAL: MTLPixelFormat = MTLPixelFormat::RGBA16Float;
pub const ROUGHNESS: MTLPixelFormat = MTLPixelFormat::R16Float;
pub const ALBEDO: MTLPixelFormat = MTLPixelFormat::RGBA16Float;
pub const HIT_DIST: MTLPixelFormat = MTLPixelFormat::R16Float;

/// The five guide planes `MTLFXTemporalDenoisedScaler` reads beyond the trio,
/// at one extent, in `Shared` storage.
///
/// **A SEPARATE STRUCT, AND THAT IS THE MODULE RULE BEING OBEYED RATHER THAN
/// SIDESTEPPED.** The obvious shape is five `Option` fields on `Trio` — and
/// this file's own header forbids exactly that: "shared code acquires
/// parameters, and a `Trio` that grows an `Option<reactive_mask>` or a flags
/// word IS the forbidden trait spelled as a struct. So: if this ever needs a
/// conditional, delete it and inline both copies."
///
/// It is also what protects the thing `Trio` exists for. Its value is that the
/// two upscalers provably see the SAME BYTES, which is what makes their
/// cross-check evidence; an `Option` on `Trio` would mean one consumer's
/// allocation depended on which consumer it was, and the "identical by
/// construction" claim would quietly become "identical for the three fields
/// that happen not to be optional".
///
/// So `Trio` is untouched, `mtl::mfxdn` composes the two, and the two upscalers
/// keep their guarantee. The cost is one more extent to keep in step, which is
/// what the assert in `stage` is for.
pub struct Guides {
    pub normal: Retained<ProtocolObject<dyn MTLTexture>>,
    pub roughness: Retained<ProtocolObject<dyn MTLTexture>>,
    pub diff_alb: Retained<ProtocolObject<dyn MTLTexture>>,
    pub spec_alb: Retained<ProtocolObject<dyn MTLTexture>>,
    pub hit_dist: Retained<ProtocolObject<dyn MTLTexture>>,
    extent: (usize, usize),
}

impl Guides {
    /// Allocate at `extent`. Unlike `Trio` this is always the EXACT input
    /// extent, because its only consumer is the denoised scaler and that path
    /// has no sub-rect mechanism at all — `inputContentWidth`/`Height` are
    /// dropped from `MTLFXTemporalDenoisedScalerBase`, so the scaler always
    /// consumes the full `inputWidth x inputHeight`. The assert in `stage` is
    /// therefore an equality in practice and an inequality by contract.
    pub fn new(mtl: &Mtl, extent: (usize, usize)) -> Result<Guides, String> {
        let (w, h) = extent;
        let t = |f: MTLPixelFormat, what: &str| {
            mtl.texture(w, h, f, MTLStorageMode::Shared).map_err(|e| format!("{what}: {e}"))
        };
        Ok(Guides {
            normal: t(NORMAL, "normal plane")?,
            roughness: t(ROUGHNESS, "roughness plane")?,
            diff_alb: t(ALBEDO, "diffuse albedo plane")?,
            spec_alb: t(ALBEDO, "specular albedo plane")?,
            hit_dist: t(HIT_DIST, "specular hit distance plane")?,
            extent,
        })
    }

    /// Stage the five guides into the planes' top-left `rw x rh` sub-rect.
    ///
    /// PRECONDITION ON `g`: a FULL `GBufs`, never `GBufs::new_slim`. The slim
    /// variant leaves every plane read here zero-length and `GBufs::write`
    /// skips their stores, so a slim buffer would panic on the first index
    /// rather than merely produce a black guide — which is the good failure,
    /// and is why this is stated rather than defended against.
    pub fn stage(
        &self,
        mtl: &Mtl,
        g: &crate::dlss::GBufs,
        render: (usize, usize),
    ) {
        let (rw, rh) = render;

        // The same two real asserts as `Trio::stage`, for the same two reasons
        // spelled out there — `debug_assert` is dead in this tree, and the row
        // casts in `fsr::stage_*` carry an alignment precondition a `Vec<u8>`
        // does not promise.
        assert!(
            rw <= self.extent.0 && rh <= self.extent.1,
            "guide stage extent {rw}x{rh} exceeds the planes' {}x{}",
            self.extent.0,
            self.extent.1,
        );

        // One scratch at the widest plane's 8 B/px; the 2 B/px planes take a
        // prefix, exactly as the trio's 4 B/px ones do.
        let mut buf = vec![0u8; rw * rh * 8];
        assert!(
            buf.as_ptr() as usize % 4 == 0,
            "staging scratch is {}-aligned; fsr::stage_* casts rows to f32/f16",
            1 << (buf.as_ptr() as usize).trailing_zeros().min(8),
        );

        crate::fsr::stage_normal(&mut buf, rw * 8, &g.normal_rough, rw, rh);
        mtl.upload(&self.normal, rw, rh, rw * 8, &buf);

        crate::fsr::stage_albedo(&mut buf, rw * 8, &g.diff_alb, rw, rh);
        mtl.upload(&self.diff_alb, rw, rh, rw * 8, &buf);

        crate::fsr::stage_albedo(&mut buf, rw * 8, &g.spec_alb, rw, rh);
        mtl.upload(&self.spec_alb, rw, rh, rw * 8, &buf);

        crate::fsr::stage_roughness(&mut buf[..rw * rh * 2], rw * 2, &g.normal_rough, rw, rh);
        mtl.upload(&self.roughness, rw, rh, rw * 2, &buf[..rw * rh * 2]);

        crate::fsr::stage_hit_dist(&mut buf[..rw * rh * 2], rw * 2, &g.spec_hit_t, rw, rh);
        mtl.upload(&self.hit_dist, rw, rh, rw * 2, &buf[..rw * rh * 2]);
    }
}
