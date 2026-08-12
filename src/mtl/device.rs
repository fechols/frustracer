//! The minimal headless Metal harness — the `vk/headless.rs` + `vk/device.rs`
//! analogue, and deliberately a fraction of their size.
//!
//! Metal deletes most of what those two carry: there is no instance, no
//! physical-device enumeration, no queue family, no descriptor set, no memory
//! type index, and no explicit fence ring. Unified memory deletes the rest —
//! on Apple silicon a `Shared` texture is GPU-writable AND host-readable with
//! no staging copy, which is the entire `UploadBuffer` / readback apparatus
//! that `vk/stage.rs` and `gpu/d3d12.rs` exist to provide.
//!
//! What survives from the Vulkan side is the part that is about DISCIPLINE
//! rather than about the API: the `absent` flag on the error type. A box with
//! no Metal device — or a Mac whose GPU has no unified memory, see `Mtl::new`
//! — is an environment fact and the gate SKIPs; anything else, such as a
//! device that refuses a command queue, is a failure. Reporting "no GPU here"
//! for a broken lever exits 0 on a run that gated nothing, which is the shape
//! `vk::device::VkError` was given to prevent.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};
use std::ffi::c_void;

/// `absent` distinguishes "this machine has no Metal" (SKIP, exit 0) from
/// "Metal is here and something went wrong" (FAIL). See the module header.
pub struct MtlError {
    pub absent: bool,
    pub msg: String,
}

impl std::fmt::Display for MtlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.msg)
    }
}

pub struct Mtl {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl Mtl {
    pub fn new() -> Result<Mtl, MtlError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MtlError {
            absent: true,
            msg: "no Metal device (MTLCreateSystemDefaultDevice returned nil)".into(),
        })?;
        // UNIFIED MEMORY IS A PRECONDITION, NOT A PROPERTY WE MERELY REPORT.
        // Every texture here is `Shared` storage AND asks for `RenderTarget`
        // usage — a combination that is only universally available where the
        // CPU and GPU address one pool; on a discrete Mac GPU the correct mode
        // is `Managed` with explicit `didModifyRange:` / `synchronizeResource:`,
        // which this harness does not do and does not need to. Refusing here,
        // as ABSENT, is what keeps that an environment fact: without it the
        // first `newTextureWithDescriptor` returns nil and the gate reports a
        // FAILURE on a machine that is merely a different Mac — precisely the
        // inversion `MtlError::absent` exists to prevent.
        if !device.hasUnifiedMemory() {
            return Err(MtlError {
                absent: true,
                msg: format!(
                    "{} has no unified memory — this harness is Apple-silicon-only (it uses \
                     Shared-storage render targets with no Managed-mode synchronization)",
                    device.name()
                ),
            });
        }
        let queue = device.newCommandQueue().ok_or_else(|| MtlError {
            absent: false,
            msg: "the Metal device refused a command queue".into(),
        })?;
        Ok(Mtl { device, queue })
    }

    /// The `vk::device`-style identity line. `linear_tex_align` is on it
    /// deliberately: it is the number the buffer-backed image-atomic textures
    /// depend on (spirv-cross emulates Metal image atomics with a buffer
    /// aliased over the texture's memory, addressed `alignedWidth*y + x`), so
    /// a gate log that omits it cannot explain a failure that involves it.
    pub fn line(&self) -> String {
        format!(
            "{} | unified {} | linear-tex-align {} B",
            self.device.name(),
            self.device.hasUnifiedMemory(),
            self.linear_tex_align(),
        )
    }

    /// Row alignment for a buffer-backed `R32Uint` texture — the format FFX
    /// uses for its image-atomic surfaces (`reconstructedPrevNearestDepth`,
    /// the SPD global atomic). Fed to every pipeline as spirv-cross's
    /// `spvLinearTextureAlignmentOverride` function constant.
    pub fn linear_tex_align(&self) -> u32 {
        self.device
            .minimumLinearTextureAlignmentForPixelFormat(MTLPixelFormat::R32Uint)
            as u32
    }

    /// A raw `id<MTLDevice>` for the ObjC++ shim. The `Retained` keeps it
    /// alive for as long as `self`, and the shim never retains it.
    // Consumed by shim/ffx_metal.mm's `create`, which does not exist yet.
    #[allow(dead_code)]
    pub fn device_ptr(&self) -> *mut c_void {
        Retained::as_ptr(&self.device) as *mut c_void
    }

    /// A texture in `Shared` storage — CPU-writable, CPU-readable, and usable
    /// by the GPU with no staging copy or blit on Apple silicon. `Private`
    /// would need both, for nothing: this harness is headless, so no texture
    /// here is bandwidth-critical.
    pub fn texture(
        &self,
        w: usize,
        h: usize,
        format: MTLPixelFormat,
    ) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
        let d = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format, w, h, false,
            )
        };
        d.setStorageMode(MTLStorageMode::Shared);
        // ShaderRead|ShaderWrite|RenderTarget is the superset FFX can ask for:
        // its reset dispatch issues FFX_GPU_JOB_CLEAR_FLOAT on shared
        // temporals, and that path is a render-pass load-clear, which Metal
        // rejects without RenderTarget usage.
        d.setUsage(
            MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite | MTLTextureUsage::RenderTarget,
        );
        self.device
            .newTextureWithDescriptor(&d)
            .ok_or_else(|| format!("newTextureWithDescriptor {w}x{h} {format:?} returned nil"))
    }

    /// Upload a tightly- or loosely-packed image into a texture's `(0,0,w,h)`
    /// sub-rect. Metal takes an arbitrary `bytesPerRow`, so there is no
    /// 256-byte pitch rule to honour and no intermediate upload heap — the one
    /// difference between this and `gpu/ffx_up.rs::record_upload`'s recording.
    pub fn upload(
        &self,
        tex: &ProtocolObject<dyn MTLTexture>,
        w: usize,
        h: usize,
        pitch: usize,
        bytes: &[u8],
    ) {
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: w, height: h, depth: 1 },
        };
        unsafe {
            tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                std::ptr::NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                pitch,
            );
        }
    }

    /// Read a texture's `(0,0,w,h)` sub-rect back, tightly packed.
    pub fn read(
        &self,
        tex: &ProtocolObject<dyn MTLTexture>,
        w: usize,
        h: usize,
        bpp: usize,
    ) -> Vec<u8> {
        let mut out = vec![0u8; w * h * bpp];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: w, height: h, depth: 1 },
        };
        unsafe {
            tex.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::new(out.as_mut_ptr() as *mut c_void).unwrap(),
                w * bpp,
                region,
                0,
            );
        }
        out
    }

    /// Record through one command buffer and block until the GPU is done.
    /// Headless: there is no frame pacing to overlap with, so a fence ring
    /// would be apparatus with no consumer.
    pub fn run<F: FnOnce(&ProtocolObject<dyn MTLCommandBuffer>)>(&self, f: F) -> Result<(), String> {
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| "commandBuffer() returned nil".to_string())?;
        f(&cb);
        cb.commit();
        cb.waitUntilCompleted();
        // `error` is the only channel a committed buffer has; dropping it
        // silently is how a failed dispatch reads as a black image.
        match cb.error() {
            None => Ok(()),
            Some(e) => Err(format!("command buffer failed: {}", e.localizedDescription())),
        }
    }

    /// Fill a texture with zeros. Used on the output surface before a dispatch,
    /// so "the shim wrote nothing" stays distinguishable from "the shim wrote
    /// zeros" — without it a no-op dispatch and a black upscale are the same
    /// readback.
    pub fn clear(&self, tex: &ProtocolObject<dyn MTLTexture>, w: usize, h: usize, bpp: usize) {
        let zeros = vec![0u8; w * h * bpp];
        self.upload(tex, w, h, w * bpp, &zeros);
    }
}

/// The harness's own gate: a real texture round-trip through a real device.
///
/// Worth its own stage because it separates two failures that otherwise arrive
/// together and look identical — "the FSR3 shim produced nothing" and "our
/// texture plumbing never carried the bytes". It runs the SHIPPING staging
/// encoder (`fsr::stage_color`) rather than a synthetic pattern, so what it
/// proves is the actual upload path the dispatch will use, at a loose pitch
/// (the sub-rect discipline) and against a cleared surface.
pub fn self_test(m: &Mtl) -> Result<(), String> {
    use objc2_metal::MTLPixelFormat;
    use std::sync::atomic::AtomicU32;

    let (w, h) = (37usize, 29usize); // odd on purpose — see fsr::stage_self_test
    let tex = m.texture(w, h, MTLPixelFormat::RGBA16Float)?;

    // Cleared first, so a failed upload reads as zeros rather than as whatever
    // the allocator happened to hand us.
    m.clear(&tex, w, h, 8);
    if m.read(&tex, w, h, 8).iter().any(|&b| b != 0) {
        return Err("clear+read did not come back zero — Shared storage is not coherent".into());
    }

    let accum: Vec<AtomicU32> = (0..w * h * 3)
        .map(|i| AtomicU32::new((i as f32 * 0.25 - 8.0).to_bits()))
        .collect();
    let pitch = w * 8 + 16; // loose, to prove `bytesPerRow` is honoured
    let mut staged = vec![0u8; pitch * h];
    crate::fsr::stage_color(&mut staged, pitch, &accum, w, h);
    m.upload(&tex, w, h, pitch, &staged);

    let back = m.read(&tex, w, h, 8);
    for y in 0..h {
        for x in 0..w {
            for k in 0..4 {
                let got = &back[(y * w + x) * 8 + k * 2..][..2];
                let want = &staged[y * pitch + x * 8 + k * 2..][..2];
                if got != want {
                    return Err(format!(
                        "texture round-trip differs at px({x},{y}).{k}: {got:02x?} != {want:02x?} \
                         (a loose bytesPerRow was not honoured, or Shared storage is stale)"
                    ));
                }
            }
        }
    }

    // Command submission, separately: the round-trip above is pure CPU access
    // to a Shared texture and would pass on a device that cannot execute
    // anything at all.
    m.run(|cb| {
        if let Some(e) = cb.blitCommandEncoder() {
            e.endEncoding();
        }
    })?;
    Ok(())
}
