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
    MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandQueue,
    MTL4ComputeCommandEncoder, MTLArgumentBuffersTier, MTLBlitCommandEncoder, MTLBuffer,
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLCreateSystemDefaultDevice, MTLDevice,
    MTLGPUFamily, MTLOrigin, MTLPixelFormat, MTLRegion, MTLResidencySet, MTLResourceOptions,
    MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter,
    MTLSamplerState, MTLSharedEvent, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage,
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

    /// The `vk::device`-style identity line.
    ///
    /// **`linear-tex-align` came OFF this line in D2, and its removal is the
    /// point.** It was here deliberately — the row alignment the buffer-backed
    /// image-atomic textures depended on, so a gate log could explain a failure
    /// that involved it. `build.rs::MSL_VERSION` 30100 makes spirv-cross emit
    /// native image atomics, so nothing is buffer-backed, no pipeline is
    /// specialized on the alignment, and a number no code consumes on an
    /// identity line is worse than absent: it reads as a live constraint.
    ///
    /// **`metal4 family` joined it in D4 for the reason the alignment left.**
    /// It is what separates "this machine cannot do Metal 4" from "this machine
    /// says it can and something else went wrong", and K9 SKIPs on the second
    /// half of that sentence rather than the first — see `metal4_family`, which
    /// is reported and never gates.
    pub fn line(&self) -> String {
        format!(
            "{} | unified {} | arg-buffers tier {} | metal4 family {}",
            self.device.name(),
            self.device.hasUnifiedMemory(),
            match self.arg_buffers_tier() {
                MTLArgumentBuffersTier::Tier1 => "1",
                MTLArgumentBuffersTier::Tier2 => "2",
                _ => "?",
            },
            self.metal4_family(),
        )
    }

    /// A raw `id<MTLDevice>` for the ObjC++ shim. The `Retained` keeps it
    /// alive for as long as `self`, and the shim never retains it — which is
    /// what makes `Mtl` the sole owner and its lifetime the binding one.
    ///
    /// Consumed by `shim/ffx_fsr3_metal.mm`'s `create`. The `allow` stays
    /// because that call site is behind `cfg(ffx_fsr3_metal)`: a checkout
    /// without the FFX SDK builds this harness and nothing that uses it.
    #[allow(dead_code)]
    pub fn device_ptr(&self) -> *mut c_void {
        Retained::as_ptr(&self.device) as *mut c_void
    }

    /// The device as a typed object, for Rust-side framework calls.
    ///
    /// The sibling of `device_ptr`, which exists only because the ObjC++ shim
    /// takes a `void*`. MetalFX is called from Rust through `objc2`, so it
    /// wants the real thing.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Does the device claim the `Metal4` GPU family?
    ///
    /// REPORTED, never used as the gate on anything, and the two are not the
    /// same question. `mtl4()` below decides availability by asking for the
    /// object it actually needs; this is the capability bit Apple publishes,
    /// and it is on the identity line so a run where the two DISAGREE is
    /// legible instead of mysterious. The CI runner is an Apple Paravirtual
    /// device that nothing has ever probed for Metal 4, so that disagreement is
    /// a live possibility rather than a hypothetical.
    pub fn metal4_family(&self) -> bool {
        self.device.supportsFamily(MTLGPUFamily::Metal4)
    }

    /// The Metal 4 submission objects, or `None` on a system without them.
    ///
    /// **`None` IS AN ENVIRONMENT FACT AND A FAILURE IS NOT**, the same split
    /// `MtlError::absent` draws one level up. `newMTL4CommandQueue` returning
    /// nil means this macOS predates Metal 4 or this device does not implement
    /// it — a loud SKIP. A device that HANDS OVER a Metal 4 queue and then
    /// refuses an allocator or a shared event is broken, and that is an `Err`.
    ///
    /// Probed at RUNTIME rather than gated at build time, following the reason
    /// `mfxdn` states at length: a build.rs check keys on the BUILD host, which
    /// is the `cfg(windows)`-describes-the-HOST defect class, and a CI image or
    /// a cross-compile would decide it for a machine it knows nothing about.
    /// Unlike `mfxdn` this needs no `AnyClass::get` dance — every MTL4 type
    /// here is a PROTOCOL reached through a method that returns nil, never an
    /// `extern_class!` whose absence would panic.
    pub fn mtl4(&self) -> Result<Option<Mtl4>, String> {
        let Some(queue) = self.device.newMTL4CommandQueue() else { return Ok(None) };
        let allocator = self
            .device
            .newCommandAllocator()
            .ok_or("the device gave an MTL4 command queue and then refused a command allocator")?;
        // THE ONLY WAY THE CPU CAN WAIT. MTL4 has no `waitUntilCompleted` —
        // `MTL4CommandQueue` offers `commit:count:` and nothing that blocks —
        // so the queue signals this event and `Mtl4::compute` blocks on it.
        let event = self
            .device
            .newSharedEvent()
            .ok_or("the device gave an MTL4 command queue and then refused a shared event")?;
        Ok(Some(Mtl4 { queue, allocator, event, signalled: std::cell::Cell::new(0) }))
    }

    /// Which argument-buffer tier the device supports.
    ///
    /// `--msl-argument-buffer-tier 2` (`mtl::msl::CROSS_ARGS`) is a spirv-cross
    /// CODEGEN promise; this is the device's side of it, and the two are
    /// unrelated until something checks. Every Apple silicon GPU is Tier 2, but
    /// nothing in this tree had ever read the property, so a Tier-1 device would
    /// have surfaced as a pipeline that builds and then reads the wrong
    /// resource. `line()` prints it for the reason `linear-tex-align` USED to be
    /// there: a gate log that omits a number a failure depends on cannot explain
    /// that failure. The tier outlived the alignment — D2 retired the latter
    /// with the image-atomic emulation, and this one still gates `texs[]`.
    pub fn arg_buffers_tier(&self) -> MTLArgumentBuffersTier {
        self.device.argumentBuffersSupport()
    }

    /// Does the device honour what `msl::CROSS_ARGS` asks spirv-cross for?
    ///
    /// A predicate rather than a raw tier at the call site, so `main.rs` does
    /// not have to name an `objc2_metal` type to ask — the same reason
    /// `PresentSpace::format` lives in `gpu/d3d12.rs`.
    pub fn is_tier2(&self) -> bool {
        self.arg_buffers_tier() == MTLArgumentBuffersTier::Tier2
    }

    /// A `Shared`-storage buffer, host-writable and host-readable with no
    /// staging.
    ///
    /// Deliberately NOT parameterised by storage mode, unlike `texture` — that
    /// one takes the mode because MetalFX's output surface is REQUIRED to be
    /// `Private`, and no such consumer exists for buffers. `Shared` here is the
    /// unified-memory precondition `Mtl::new` already enforces as ABSENT, so
    /// `contents()` is a plain host pointer on every device this harness will
    /// ever see, and the whole `vk/stage.rs` upload-ring apparatus has nothing
    /// to do.
    pub fn buffer(&self, len: usize) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        // A zero-length buffer is nil rather than an empty allocation, which
        // would surface later as an unrelated nil-deref inside an encoder.
        if len == 0 {
            return Err("buffer(0) — Metal has no zero-length allocation".into());
        }
        self.device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| format!("newBufferWithLength({len}) returned nil"))
    }

    /// Fill a buffer with one repeated `u32`, host-side.
    ///
    /// The poison. `vk/headless.rs` does this with `vkCmdFillBuffer` because a
    /// device-local Vulkan buffer has no host pointer; `Shared` storage has one,
    /// so this is a plain write and needs no blit encoder, no barrier, and no
    /// command buffer at all. A `Private` buffer could not be filled this way —
    /// which is the other half of why `buffer` does not take a mode.
    ///
    /// WORDS RATHER THAN BYTES, and that is a safety property, not a
    /// convenience. A byte fill can only produce byte-uniform patterns, so
    /// every buffer would have to share one sentinel — and `smoke.hlsl`'s
    /// `cs_prep` turns whatever it reads from `counters` into a THREADGROUP
    /// COUNT (`(n + 63) / 64`). Poisoning `counters` with the natural
    /// `0xEEEE_EEEE` means a mis-bound run asks for ~62 million threadgroups,
    /// which on a dev Mac can take the WindowServer down with it and on CI is a
    /// 45-minute timeout. So the caller picks a per-buffer poison, and the ones
    /// a dispatch count is derived from get a small distinctive value.
    pub fn fill_words(&self, buf: &ProtocolObject<dyn MTLBuffer>, word: u32) {
        let n = buf.length() / 4;
        // SAFETY: `Shared` storage on a unified-memory device (the `Mtl::new`
        // precondition), `length()` is the allocation's own size, and the count
        // is derived from it. Nothing is in flight: every submit here blocks in
        // `run`.
        unsafe {
            std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut u32, n).fill(word)
        };
    }

    /// Write `u32` words at the start of a buffer.
    ///
    /// Here rather than at the call site so every raw host-pointer access in
    /// this backend lives in the module that owns the precondition making one
    /// legal (`Shared` storage on unified memory, enforced as ABSENT in
    /// `Mtl::new`). A caller doing its own `from_raw_parts_mut` would be
    /// relying on that precondition without being able to see it.
    ///
    /// Silently writes nothing past the allocation rather than trusting the
    /// caller's length.
    pub fn write_words(&self, buf: &ProtocolObject<dyn MTLBuffer>, words: &[u32]) {
        let n = words.len().min(buf.length() / 4);
        // SAFETY: as `fill_words` — Shared storage, host pointer, GPU idle,
        // count clamped to the allocation.
        unsafe {
            std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut u32, n)
                .copy_from_slice(&words[..n]);
        }
    }

    /// Read a buffer back as `u32` words.
    ///
    /// MUST be called after the submit that wrote it has completed — `run`
    /// blocks, so "outside the closure" is the whole rule. Reading inside would
    /// race the GPU with no diagnostic: unified memory means the stale value is
    /// a perfectly valid read of the wrong moment.
    pub fn read_words(&self, buf: &ProtocolObject<dyn MTLBuffer>, n: usize) -> Vec<u32> {
        let want = n * 4;
        let have = buf.length();
        let n = if want <= have { n } else { have / 4 };
        // SAFETY: as `fill` — Shared storage, host pointer, GPU idle. The count
        // is clamped to the allocation above rather than trusted.
        unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u32, n).to_vec() }
    }

    /// Record one COMPUTE pass and block until it completes.
    ///
    /// `run`'s sibling, and it exists to carry two disciplines rather than to
    /// save typing. First, `blit`'s nil-encoder rule: a nil
    /// `computeCommandEncoder()` must be an ERROR, not an empty command buffer
    /// that commits cleanly and reads exactly like a pass that ran and wrote
    /// nothing. Second, ONE encoder for the whole chain — Metal's default
    /// dispatch type is `MTLDispatchTypeSerial`, so consecutive dispatches in
    /// one encoder are ordered and hazard-tracked by the driver. That is why
    /// this has no barrier apparatus at all where `vk/headless.rs` needs
    /// `compute_barrier` between every pass; if that ordering ever proves not
    /// to hold for an indirect ARGUMENT read, the fallback is one encoder per
    /// dispatch (Metal orders those absolutely), not scattered barriers.
    pub fn compute<F>(&self, f: F) -> Result<(), String>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>) -> Result<(), String>,
    {
        let mut inner: Result<(), String> = Ok(());
        let outer = self.run(|cb| {
            let enc = match cb.computeCommandEncoder() {
                Some(e) => e,
                None => {
                    inner = Err("computeCommandEncoder() returned nil".into());
                    return;
                }
            };
            inner = f(&enc);
            enc.endEncoding();
        });
        // The inner error first: a bad bind is the interesting failure, and the
        // command buffer's own error is often its downstream consequence.
        inner.and(outer)
    }

    /// A texture, in the storage mode the CALLER names.
    ///
    /// `Shared` is what this harness wants nearly everywhere — CPU-writable,
    /// CPU-readable, and GPU-usable with no staging copy or blit on Apple
    /// silicon, so `upload`/`read`/`clear` below are plain memory access.
    /// `Private` would need all three, for nothing: headless, so no texture
    /// here is bandwidth-critical.
    ///
    /// THE MODE IS A PARAMETER BECAUSE ONE CONSUMER HAS NO CHOICE. MetalFX's
    /// `MTLFXTemporalScalerBase::outputTexture` is documented "You are
    /// responsible for providing a texture with a private `storageMode`"
    /// (`MTLFXTemporalScaler.h:222` in the macOS 26.5 SDK) — the one texture in
    /// this tree the CPU may not touch directly, which is why `read_private`
    /// and `clear_private` exist beside their plain siblings. Hardcoding
    /// `Shared` here and hoping the requirement is advisory would put the whole
    /// question under `MTL_SHADER_VALIDATION=1`, where an assert aborts the
    /// process rather than failing a gate.
    pub fn texture(
        &self,
        w: usize,
        h: usize,
        format: MTLPixelFormat,
        storage: MTLStorageMode,
    ) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
        let d = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format, w, h, false,
            )
        };
        d.setStorageMode(storage);
        // ShaderRead|ShaderWrite|RenderTarget is the superset FFX can ask for:
        // its reset dispatch issues FFX_GPU_JOB_CLEAR_FLOAT on shared
        // temporals, and that path is a render-pass load-clear, which Metal
        // rejects without RenderTarget usage.
        d.setUsage(
            MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite | MTLTextureUsage::RenderTarget,
        );
        self.device
            .newTextureWithDescriptor(&d)
            .ok_or_else(|| {
                format!("newTextureWithDescriptor {w}x{h} {format:?} {storage:?} returned nil")
            })
    }

    /// An `MTLSamplerState`, the Metal spelling of an HLSL `SamplerState`.
    ///
    /// `supportArgumentBuffers(true)` IS A PRECONDITION, NOT A HINT, and it is
    /// unconditional here because every sampler this backend creates is reached
    /// through an argument buffer — `msl::CROSS_ARGS` has no other mode. A
    /// sampler created without it cannot be encoded into one.
    ///
    /// THE PARAMETERS ARE THE OTHER TWO BACKENDS', not free choices. C3's
    /// subject copies `trace_common.hlsli`'s `samp_lin` / `samp_aniso`
    /// verbatim, so Metal's pair must be what those already are:
    /// `vk/tracer.rs:822-849` builds LINEAR min/mag/mip, REPEAT in every axis,
    /// `max_anisotropy(aniso)`, unclamped LOD; `gpu/trace.rs:507-548` spells the
    /// same pair as D3D12 STATIC samplers (`MIN_MAG_MIP_LINEAR` / `ANISOTROPIC`,
    /// `WRAP`). Static is a root-signature spelling with no Metal analogue —
    /// creating objects here is the faithful port, not a divergence.
    ///
    /// `nearest` exists for the probe's sake and is the property its exactness
    /// rests on: `mtlbind.hlsl`'s two space0 samplers differ ONLY in address
    /// mode, so one coordinate outside [0,1] resolves to a different TEXEL with
    /// no filter weights — no tolerance, no undefined tie-break.
    pub fn sampler(
        &self,
        nearest: bool,
        repeat: bool,
        aniso: u32,
    ) -> Result<Retained<ProtocolObject<dyn MTLSamplerState>>, String> {
        let d = unsafe { MTLSamplerDescriptor::new() };
        let f = if nearest { MTLSamplerMinMagFilter::Nearest } else { MTLSamplerMinMagFilter::Linear };
        d.setMinFilter(f);
        d.setMagFilter(f);
        // NotMipmapped for the nearest arm on purpose: the probe's textures have
        // no mips, and a Linear mip filter over a single level is a weight
        // computation the exact-integer readback does not need to depend on.
        d.setMipFilter(if nearest {
            MTLSamplerMipFilter::NotMipmapped
        } else {
            MTLSamplerMipFilter::Linear
        });
        let addr = if repeat {
            MTLSamplerAddressMode::Repeat
        } else {
            MTLSamplerAddressMode::ClampToEdge
        };
        d.setSAddressMode(addr);
        d.setTAddressMode(addr);
        d.setRAddressMode(addr);
        // Metal rejects 0 and clamps above 16; the callers pass 1 or
        // `texture::max_aniso()`, which is the same value Vulkan and D3D12 feed
        // their own `samp_aniso`.
        d.setMaxAnisotropy(aniso.clamp(1, 16) as usize);
        d.setSupportArgumentBuffers(true);
        self.device
            .newSamplerStateWithDescriptor(&d)
            .ok_or_else(|| format!("newSamplerStateWithDescriptor(nearest {nearest}, repeat {repeat}, aniso {aniso}) returned nil"))
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

    /// Copy `w x h` from one texture's `(0,0)` to another's, on the GPU.
    ///
    /// The only way to move bytes in or out of a `Private` texture: `getBytes`
    /// and `replaceRegion` are both illegal there, which is what `read_private`
    /// and `clear_private` use this for. Same blocking submit as `run`.
    ///
    /// A NIL ENCODER IS AN ERROR, not a skipped copy, and that is the whole
    /// reason `encoded` exists: `run`'s closure returns nothing, so without it a
    /// nil encoder would submit an EMPTY command buffer, `cb.error()` would be
    /// `None`, and this would return `Ok` having moved no bytes. `read_private`
    /// would then hand back a freshly-created staging texture's UNDEFINED
    /// contents as if they were data, and `clear_private` would leave its target
    /// untouched — which is precisely the "silently untestable" shape
    /// `clear_private`'s own comment argues against. `blitCommandEncoder()`
    /// realistically never returns nil; the point is that if it did, nothing
    /// downstream could tell.
    fn blit(
        &self,
        src: &ProtocolObject<dyn MTLTexture>,
        dst: &ProtocolObject<dyn MTLTexture>,
        w: usize,
        h: usize,
    ) -> Result<(), String> {
        let mut encoded = false;
        self.run(|cb| {
            if let Some(e) = cb.blitCommandEncoder() {
                unsafe {
                    e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        src,
                        0,
                        0,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                        MTLSize { width: w, height: h, depth: 1 },
                        dst,
                        0,
                        0,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                    );
                }
                e.endEncoding();
                encoded = true;
            }
        })?;
        if !encoded {
            return Err(format!("blitCommandEncoder() returned nil for a {w}x{h} copy"));
        }
        Ok(())
    }

    /// `read`, for a `Private` texture: blit into a Shared staging texture and
    /// read THAT. The staging texture is created per call — this is a headless
    /// gate reading one frame, not a present path.
    pub fn read_private(
        &self,
        tex: &ProtocolObject<dyn MTLTexture>,
        w: usize,
        h: usize,
        bpp: usize,
        format: MTLPixelFormat,
    ) -> Result<Vec<u8>, String> {
        let staging = self.texture(w, h, format, MTLStorageMode::Shared)?;
        self.blit(tex, &staging, w, h)?;
        Ok(self.read(&staging, w, h, bpp))
    }

    /// `clear`, for a `Private` texture: zero a Shared texture and blit it over.
    ///
    /// Preserving the clear across the storage-mode change is the point, not an
    /// incidental convenience — it is what keeps "the scaler wrote nothing"
    /// distinguishable from "the scaler wrote a dark image" in the readback,
    /// and dropping it because `replaceRegion` no longer applies would make a
    /// gate's non-zero assertion silently untestable.
    pub fn clear_private(
        &self,
        tex: &ProtocolObject<dyn MTLTexture>,
        w: usize,
        h: usize,
        bpp: usize,
        format: MTLPixelFormat,
    ) -> Result<(), String> {
        let staging = self.texture(w, h, format, MTLStorageMode::Shared)?;
        self.clear(&staging, w, h, bpp);
        self.blit(&staging, tex, w, h)
    }
}

/// The Metal 4 submission objects — `Mtl`'s peer, not its replacement.
///
/// `Mtl` keeps the device, the buffers and the `MTLCommandQueue`; this owns
/// only what MTL4 adds, so both paths allocate from ONE device and dispatch ONE
/// set of pipelines. That is what makes `mtl4`'s byte-compare against `smoke`
/// a comparison of SUBMISSION and nothing else.
///
/// # Three things MTL4 removed, and each one costs a line here
///
/// * **No implicit hazard tracking.** `MTLComputeCommandEncoder` defaults to
///   `MTLDispatchTypeSerial` and the driver orders and hazard-tracks the
///   dispatches — `smoke::pass` says so, and says `vk/headless.rs`'s barrier
///   apparatus therefore "has no analogue here". Under MTL4 it has one:
///   `mtl4::pass` must issue explicit barriers between the three dispatches.
/// * **No implicit residency.** There is no `useResource:` on any MTL4 encoder
///   (checked against every binding in the crate), and the argument table takes
///   a raw GPU ADDRESS, so nothing infers residency from what was bound.
/// * **No `waitUntilCompleted`.** Hence `event` and `signalled` below.
///
/// # And a fourth it removed that costs NO line here, deliberately
///
/// **There is no `error` on an MTL4 command buffer, and this path therefore
/// checks none.** `Mtl::run` next door checks `cb.error()` and says why —
/// *"the only channel a committed buffer has; dropping it silently is how a
/// failed dispatch reads as a black image"* — and MTL4 has no synchronous
/// equivalent anywhere: not on the queue, not on the command buffer, not on
/// the event. The whole channel is `MTL4CommitFeedback::error`, delivered as a
/// BLOCK through `MTL4CommitOptions::addFeedbackHandler:` and
/// `commit:count:options:`, on a dispatch queue.
///
/// It is not wired, and the reason is that wiring it naively would be a probe
/// that has not been shown to REACH its target — the `FR_ABL` trap this
/// project has fallen into four times. The callback is asynchronous and
/// ordered against nothing this function waits on, so reading a captured error
/// after `waitUntilSignaledValue:` returns would report "no error" both when
/// there was none and when the handler had simply not run yet, and the two are
/// indistinguishable from here. Doing it properly means making the FEEDBACK
/// the completion signal instead of the event, which is a different design for
/// the wait rather than a check bolted onto this one.
///
/// What that costs today is diagnosis, not coverage: `smoke::verify` reads
/// every word of the result, so a faulted submission fails the gate on its
/// DATA. It fails with "out[0] is 0xDEADBEEF" where Metal 3 would have said
/// which command buffer died and why. **A rung that extends this path beyond
/// the smoke chain should wire the handler first**, because the further the
/// work gets from a fully-verified 619-word readback, the more of that
/// distinction stops being cosmetic.
///
/// # `signalled` is a monotone counter, not a fence index
///
/// Each `compute` takes the next value, signals it AFTER the commit, and blocks
/// until the event reaches it. Reusing one value across passes would make the
/// second wait return instantly on the first pass's signal — a readback racing
/// a live GPU on unified memory, which is exactly the failure `Mtl::compute`'s
/// "read outside the closure" comment describes, with no diagnostic.
pub struct Mtl4 {
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    signalled: std::cell::Cell<u64>,
}

/// How long `compute` will wait for a submission before calling it a failure.
///
/// A NUMBER RATHER THAN `u64::MAX`, because the failure this bounds is a gate
/// that never returns. The smoke chain is three dispatches over 619 words and
/// completes in single-digit milliseconds on an M1; two seconds is three orders
/// of magnitude of headroom and still far inside CI's 10-minute step budget.
/// `smoke.rs`'s `GRID_POISON` exists to prevent the OTHER version of this — a
/// dispatch that runs 5e10 threadgroups — and this covers the case where the
/// work never starts at all.
const MTL4_WAIT_MS: u64 = 2_000;

impl Mtl4 {
    /// `Mtl::compute`'s twin: encode one compute pass, submit it, and block.
    ///
    /// The shape is deliberately identical — the closure returns a `Result` so
    /// a binding error is reported before the command buffer's own, which is
    /// usually its downstream consequence — and the differences are all MTL4's:
    /// an allocator to begin against, an explicit `endCommandBuffer`, and a
    /// signal-and-wait instead of `waitUntilCompleted`.
    ///
    /// **THE ALLOCATOR IS RESET AFTER THE WAIT, NEVER BEFORE — AND NOT AT ALL
    /// IF THE WAIT TIMES OUT.** It owns the memory the encoded commands live
    /// in, so resetting it while the GPU is still reading them is a
    /// use-after-free that would present as corrupted output rather than as an
    /// error — the same class as the readback race above, and invisible for the
    /// same reason. The timeout branch is the one place where the GPU is
    /// PROVABLY still reading, so it returns the error and leaks the allocator
    /// rather than resetting on the strength of a wait that did not happen.
    pub fn compute<F>(&self, m: &Mtl, f: F) -> Result<(), String>
    where
        F: FnOnce(&ProtocolObject<dyn MTL4ComputeCommandEncoder>) -> Result<(), String>,
    {
        let cb = m.device().newCommandBuffer().ok_or("newCommandBuffer() returned nil")?;
        cb.beginCommandBufferWithAllocator(&self.allocator);
        let inner = {
            let enc = cb
                .computeCommandEncoder()
                .ok_or("MTL4CommandBuffer::computeCommandEncoder() returned nil")?;
            let r = f(&enc);
            enc.endEncoding();
            r
        };
        cb.endCommandBuffer();

        // `commit:count:` takes a C ARRAY of command buffers, so even one needs
        // a slot to point at. Built here rather than at the call site because
        // the lifetime is the delicate part: `cb` must outlive the call, and it
        // does — `one` borrows it and both live to the end of this function.
        let mut one = [std::ptr::NonNull::from(&*cb)];
        // SAFETY: `count` is 1 and `one` holds exactly one valid, non-null
        // pointer to a command buffer that outlives the call.
        unsafe { self.queue.commit_count(std::ptr::NonNull::from(&mut one[0]), 1) };

        let want = self.signalled.get() + 1;
        self.signalled.set(want);
        self.queue.signalEvent_value(ProtocolObject::from_ref(&*self.event), want);
        if !self.event.waitUntilSignaledValue_timeoutMS(want, MTL4_WAIT_MS) {
            // RETURNED WITHOUT RESETTING, and that is the whole point of the
            // paragraph above. A wait that timed out is the ONE case where the
            // GPU is provably still holding the commands this allocator owns,
            // so resetting on the way out would be precisely the use-after-free
            // the success path is careful to avoid — committed on the strength
            // of a wait that, by hypothesis, did not happen. Leaking the
            // allocator instead ends the gate with a message a reader can act
            // on; the process is failing either way, and only one of the two
            // endings is defined behaviour.
            return Err(format!(
                "the MTL4 submission did not signal {want} within {MTL4_WAIT_MS} ms — the \
                 command buffer never completed"
            ));
        }
        self.allocator.reset();
        inner
    }

    /// A residency set holding `allocations`, committed, requested and attached
    /// to the queue.
    ///
    /// **ALL FOUR STEPS, and none of them is optional.** `addAllocation:`
    /// alone leaves the set uncommitted (its own doc says so); `commit` without
    /// `requestResidency` describes a set nothing has been asked to page in;
    /// and a set the QUEUE does not hold applies to no submission. Each
    /// omission fails the same way — the resource is simply not there when the
    /// shader reads through its address — which is why they are one function
    /// rather than four calls at a call site.
    ///
    /// Returns the set, which the caller must keep alive for the submission,
    /// and its `allocationCount` — `smoke::Pass::resident`'s value on this
    /// path, and the anti-vacuity number for a set that silently came back
    /// empty.
    pub fn residency(
        &self,
        m: &Mtl,
        allocations: &[&ProtocolObject<dyn objc2_metal::MTLAllocation>],
    ) -> Result<(Retained<ProtocolObject<dyn MTLResidencySet>>, usize), String> {
        let desc = objc2_metal::MTLResidencySetDescriptor::new();
        let set = m
            .device()
            .newResidencySetWithDescriptor_error(&desc)
            .map_err(|e| format!("newResidencySetWithDescriptor: {e}"))?;
        for a in allocations {
            set.addAllocation(a);
        }
        set.commit();
        set.requestResidency();
        self.queue.addResidencySet(&set);
        let n = set.allocationCount();
        Ok((set, n))
    }

    /// Detach a set the queue no longer needs.
    ///
    /// Paired with `residency` rather than left to `Drop`: the queue holds a
    /// strong reference, so a set that is never removed keeps its allocations
    /// resident for the life of the queue. Harmless for one gate run and wrong
    /// as a pattern — and `--check-mtl` runs the chain twice, so a leak here
    /// would be a second set covering the first's buffers, which is precisely
    /// the state that would make a residency TOOTH stop biting.
    ///
    /// **CALL IT ONLY WHERE THE SUBMISSION IS KNOWN TO HAVE COMPLETED.** It
    /// revokes the backing for every allocation in the set, so calling it while
    /// work is still in flight — after `compute` reports a TIMEOUT, say — is
    /// the very use-after-free the set exists to prevent, arriving through the
    /// cleanup path. `mtl4::pass` therefore skips it on failure and says so;
    /// an over-long residency in a process that is failing anyway costs
    /// nothing, and the other ending is undefined.
    pub fn drop_residency(&self, set: &ProtocolObject<dyn MTLResidencySet>) {
        self.queue.removeResidencySet(set);
        set.endResidency();
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
    let tex = m.texture(w, h, MTLPixelFormat::RGBA16Float, MTLStorageMode::Shared)?;

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
