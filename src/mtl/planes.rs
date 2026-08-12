//! The three input planes every Metal upscaler consumes — allocated, formatted
//! and staged in one place.
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
