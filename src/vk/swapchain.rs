//! A swapchain over `VK_EXT_headless_surface` — the present path, with
//! nothing to scan out.
//!
//! WHY A SURFACE NOBODY LOOKS AT. Rung 1 gave this backend a graphics pipeline
//! drawing into an image it owns; that proves rasterisation and the tone curve
//! and says nothing about presentation. A real surface needs a window, which
//! needs a windowing crate and an input design (B6b) — so the present PATH
//! would have arrived untested underneath the first thing that could not run
//! headlessly. `VK_EXT_headless_surface` splits that: acquire, render into an
//! image the presentation engine owns, present, and let the engine recycle it,
//! all with no display attached and therefore all runnable on a CI box with no
//! GPU at all. What it cannot prove is PACING — there is no vblank and no
//! scanout — and that is B6b's to measure, not this module's to claim.
//!
//! WHAT A GATE CAN AND CANNOT ASSERT HERE, because the obvious assertion is
//! weaker than it looks. You cannot read an image back after presenting it:
//! ownership returns to the presentation engine. So the claim is made one step
//! earlier — the same pipeline drawing into a swapchain image must produce the
//! same bytes as into an offscreen image OF THE SAME FORMAT — which decouples
//! from format negotiation entirely and stays a BYTE identity rather than a
//! tolerance. And `VK_SUCCESS` out of `vkQueuePresentKHR` is a statement about
//! a function call, not about a frame reaching anything, which is why the gate
//! proves the present BY EXHAUSTION instead: run one more cycle than there are
//! images, and the last acquire can only succeed if the engine RELEASED one.
//!
//! THE FORMAT IS NEGOTIATED, NEVER ASSUMED, and this module refuses two
//! classes rather than guessing. `_SRGB` is refused because the pixel shader
//! already applies its own `pow(1/2.2)` and the hardware would encode it a
//! second time — a wrong image that still looks plausible, which is the worst
//! shape a defect can take here. And a format `display::rgb_offsets` does not
//! know is refused because decoding it would mean assuming a byte order: the
//! surface's own first choice on both ICDs measured here is `R8G8B8A8_UNORM`,
//! the opposite order from the one rung 1 renders, so "take `formats[0]` and
//! decode it the way we always have" would have swapped R and B in 255 of
//! every 256 texels of a hashed pattern while every other assertion stayed
//! green.

use ash::vk;

use crate::vk::device::Vk;
use crate::vk::display;
use crate::vk::headless::VkHeadless;

/// How long to wait for an image before calling it a failure.
///
/// FINITE, and that is the whole reason the exhaustion proof works. A present
/// that did nothing starves the pool, and `u64::MAX` would turn that into a
/// hang — a gate that never reports rather than one that reports a failure.
/// Five seconds is far beyond any legitimate acquire on a surface with no
/// scanout to wait for.
const ACQUIRE_TIMEOUT_NS: u64 = 5_000_000_000;

/// The usage a swapchain image must support for the gate's claim to be
/// reachable: draw into it, then copy it out.
///
/// MEASURED on both ICDs on the development box as part of
/// `supportedUsageFlags = 0x8009F`, so this is a check rather than a hope. If
/// a future ICD reports less, the SKIP names which flag was missing — see
/// `new`'s comment for the two documented fallbacks, deliberately NOT built,
/// because an untested path reachable only on hardware nobody here has is how
/// silent breakage gets in.
const NEED_USAGE: vk::ImageUsageFlags =
    vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::TRANSFER_SRC.as_raw(),
    );

/// A headless surface, its swapchain, and the synchronisation one present
/// cycle needs.
pub struct Swapchain {
    surface_fn: ash::khr::surface::Instance,
    swapchain_fn: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,

    /// The NEGOTIATED format. Every consumer reads it from here rather than
    /// assuming: the second `Passes` is built at it, and `display::decode` is
    /// handed it.
    pub fmt: vk::Format,
    pub w: u32,
    pub h: u32,

    /// The presentation engine's images. It owns them — no `VkDeviceMemory`
    /// here, which is exactly why `Passes::record_to` takes four fields rather
    /// than a `display::Image`.
    pub images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,

    acquire: vk::Semaphore,

    /// ONE RENDER-FINISHED SEMAPHORE PER IMAGE, and the count is not a
    /// preference — a single shared one is a validation ERROR that the layer
    /// named exactly when this shipped that way:
    ///
    /// ```text
    /// vkQueueSubmit(): pSubmits[0].pSignalSemaphores[0] is being signaled by
    /// VkQueue ..., but it may still be in use by VkSwapchainKHR ...
    /// ```
    ///
    /// A binary semaphore may not be re-signalled while a wait on it is
    /// outstanding, and `vkQueuePresentKHR`'s wait has no CPU-visible
    /// completion — the fence this harness signals covers the SUBMIT, not the
    /// present. Per-image is what makes the reuse provably safe rather than
    /// probably safe: acquiring image N means the presentation engine RELEASED
    /// image N, which means the present that waited on `render_done[N]` has
    /// completed. So the index that selects the image selects a semaphore that
    /// is free by the same argument.
    ///
    /// Note the ACQUIRE semaphore above needs no such treatment, and for a
    /// different reason rather than by luck: `wait_submit` blocks before the
    /// next acquire, and the submit it waits for is the one that waited on the
    /// acquire semaphore — so that wait is provably retired.
    render_done: Vec<vk::Semaphore>,
}

impl Swapchain {
    /// Create a headless surface and a swapchain on it, or say why not.
    ///
    /// `Ok(None)` is "this box cannot present" and is a SKIP; `Err` is a real
    /// failure of something that should have worked.
    pub fn new(hg: &VkHeadless, w: u32, h: u32) -> Result<Option<Swapchain>, String> {
        let vkd = &hg.vk;

        // THE THREE INDEPENDENT ENVIRONMENT FACTS, checked separately so the
        // skip line names which one is missing. They fail for unrelated
        // reasons: the surface pair is a loader/ICD property, the swapchain
        // extension is a device property, and a graphics queue is a property
        // of the family the pick chose.
        if !vkd.headless_surface {
            return Ok(None);
        }
        if !vkd.info.swapchain {
            return Ok(None);
        }
        if !vkd.info.graphics_queue {
            return Ok(None);
        }

        // A SOFTWARE DEVICE IS STOOD DOWN BEFORE THE CALL, NOT AFTER IT, and
        // that asymmetry with V11/V13's software skips is the whole point:
        // those skip on an ERROR the API returned, which is a thing a gate can
        // catch. This one cannot be caught at all.
        //
        // MEASURED on Mesa 25.x lavapipe (`libvulkan_lvp.so`): it advertises
        // `VK_KHR_swapchain`, accepts a headless surface, and answers EVERY
        // capability query — present support true, a non-empty format list,
        // FIFO present mode, `supportedUsageFlags` containing everything asked
        // for — and then `vkCreateSwapchainKHR` JUMPS TO ADDRESS ZERO inside
        // its own frames. Backtrace: `#0 0x0 / #1..#4 libvulkan_lvp.so / #5
        // libvulkan.so.1 / #8 ash create_swapchain`. It reproduces identically
        // with the validation layer disabled, so the layer is not implicated,
        // and RADV runs this exact code path clean with validation armed. So
        // it is lavapipe advertising support it does not have.
        //
        // A segfault is not a failure mode a gate can report — it takes the
        // process down mid-suite and every stage after it goes unrun — and CI
        // runs `--check-vk` on llvmpipe on every push, so making that call is
        // not an option. Refusing to make it is.
        //
        // `FR_VK_PRESENT_SOFTWARE=1` forces the attempt anyway, so the day a
        // Mesa release fixes this the re-test is one variable rather than a
        // patch: the escape is RECORDED rather than pre-applied, and if it
        // comes back clean this whole block is deleted and V19 joins CI's
        // forbidden-skip list.
        let force = std::env::var("FR_VK_PRESENT_SOFTWARE").is_ok_and(|v| v != "0");
        if vkd.info.kind == vk::PhysicalDeviceType::CPU && !force {
            return Ok(None);
        }

        let surface_fn = ash::khr::surface::Instance::new(&vkd.entry, &vkd.instance);
        let headless_fn = ash::ext::headless_surface::Instance::new(&vkd.entry, &vkd.instance);

        let surface = unsafe {
            headless_fn.create_headless_surface(&vk::HeadlessSurfaceCreateInfoEXT::default(), None)
        }
        .map_err(|e| format!("vkCreateHeadlessSurfaceEXT: {e}"))?;

        // From here on a failure must destroy the surface before returning, so
        // the body is a closure and the surface is freed on every path out.
        // Leaking an instance-level handle would outlive the device teardown
        // below it and surface (pun intended) as a validation error in an
        // unrelated later stage.
        let built = Self::build(hg, &surface_fn, surface, w, h);
        match built {
            Ok(Some(mut sc)) => {
                sc.surface_fn = surface_fn;
                sc.surface = surface;
                Ok(Some(sc))
            }
            Ok(None) => {
                unsafe { surface_fn.destroy_surface(surface, None) };
                Ok(None)
            }
            Err(e) => {
                unsafe { surface_fn.destroy_surface(surface, None) };
                Err(e)
            }
        }
    }

    fn build(
        hg: &VkHeadless,
        surface_fn: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        w: u32,
        h: u32,
    ) -> Result<Option<Swapchain>, String> {
        let vkd = &hg.vk;

        // PRESENT SUPPORT IS PER (device, queue family, surface) and is its own
        // question: a device can advertise `VK_KHR_swapchain` while the family
        // we picked for compute cannot present on this surface.
        let supported = unsafe {
            surface_fn.get_physical_device_surface_support(vkd.phys, vkd.qfam, surface)
        }
        .map_err(|e| format!("vkGetPhysicalDeviceSurfaceSupportKHR: {e}"))?;
        if !supported {
            return Ok(None);
        }

        let caps = unsafe {
            surface_fn.get_physical_device_surface_capabilities(vkd.phys, surface)
        }
        .map_err(|e| format!("vkGetPhysicalDeviceSurfaceCapabilitiesKHR: {e}"))?;

        if !caps.supported_usage_flags.contains(NEED_USAGE) {
            // TWO FALLBACKS ARE DOCUMENTED AND NOT BUILT, deliberately. Tier 2
            // (SAMPLED but no TRANSFER_SRC) would bind the swapchain image as
            // `t0` and blit it into an offscreen one, recovering the byte
            // identity one indirection removed; tier 3 (neither) could only
            // assert structurally — acquire/render/present completes with no
            // validation error and no timeout — and its line would have to
            // read differently so nobody mistakes it for tier 1's claim.
            // Neither is written because both ICDs here report 0x8009F, so
            // shipping them would mean shipping code no gate can reach.
            return Ok(None);
        }

        let formats = unsafe { surface_fn.get_physical_device_surface_formats(vkd.phys, surface) }
            .map_err(|e| format!("vkGetPhysicalDeviceSurfaceFormatsKHR: {e}"))?;
        let Some(sf) = pick_format(&formats) else {
            return Ok(None);
        };

        // FIFO is the one present mode the spec guarantees, and with no
        // scanout there is nothing MAILBOX would buy — the exhaustion proof
        // wants images returned, not latency.
        let modes = unsafe {
            surface_fn.get_physical_device_surface_present_modes(vkd.phys, surface)
        }
        .map_err(|e| format!("vkGetPhysicalDeviceSurfacePresentModesKHR: {e}"))?;
        if !modes.contains(&vk::PresentModeKHR::FIFO) {
            return Ok(None);
        }

        // `currentExtent` of 0xFFFFFFFF means the swapchain chooses — measured
        // on both ICDs here, and it is what lets the gate keep rung 1's
        // fixture size with no resize logic. Clamp anyway: a surface that does
        // pin an extent must be obeyed, and an out-of-range request is an
        // error rather than a negotiation.
        let extent = if caps.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: w.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: h.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        } else {
            caps.current_extent
        };

        // `max_image_count == 0` means unbounded. Asking for the minimum is
        // right here: the exhaustion proof wants the SMALLEST pool the surface
        // will give, since that is what makes "one more cycle than there are
        // images" a short loop rather than a long one.
        let mut count = caps.min_image_count;
        if caps.max_image_count != 0 {
            count = count.min(caps.max_image_count);
        }

        let swapchain_fn = ash::khr::swapchain::Device::new(&vkd.instance, &vkd.device);
        let sci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(count)
            .image_format(sf.format)
            .image_color_space(sf.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(NEED_USAGE)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);
        let swapchain = unsafe { swapchain_fn.create_swapchain(&sci, None) }
            .map_err(|e| format!("vkCreateSwapchainKHR: {e}"))?;

        let images = unsafe { swapchain_fn.get_swapchain_images(swapchain) }
            .map_err(|e| format!("vkGetSwapchainImagesKHR: {e}"))?;

        let mut views = Vec::with_capacity(images.len());
        for &img in &images {
            let view = unsafe {
                vkd.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(img)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(sf.format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
            }
            .map_err(|e| format!("vkCreateImageView(swapchain): {e}"))?;
            views.push(view);
        }

        let sem = || unsafe {
            vkd.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        };
        let acquire = sem().map_err(|e| format!("vkCreateSemaphore(acquire): {e}"))?;
        let mut render_done = Vec::with_capacity(images.len());
        for _ in 0..images.len() {
            render_done.push(sem().map_err(|e| format!("vkCreateSemaphore(render): {e}"))?);
        }

        Ok(Some(Swapchain {
            // Filled by `new` — `build` does not own the surface, so that it
            // can be freed on every early return above.
            surface_fn: ash::khr::surface::Instance::new(&vkd.entry, &vkd.instance),
            swapchain_fn,
            surface: vk::SurfaceKHR::null(),
            swapchain,
            fmt: sf.format,
            w: extent.width,
            h: extent.height,
            images,
            views,
            acquire,
            render_done,
        }))
    }

    /// The image count, which is what the exhaustion proof counts cycles
    /// against.
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Acquire the next image, or say which way it failed.
    ///
    /// The timeout is FINITE by design (see `ACQUIRE_TIMEOUT_NS`): a present
    /// that returned success and did nothing shows up here as a timeout on the
    /// cycle after the pool is exhausted, which is the whole exhaustion proof.
    pub fn acquire(&self) -> Result<usize, String> {
        let (idx, suboptimal) = unsafe {
            self.swapchain_fn.acquire_next_image(
                self.swapchain,
                ACQUIRE_TIMEOUT_NS,
                self.acquire,
                vk::Fence::null(),
            )
        }
        .map_err(|e| match e {
            vk::Result::TIMEOUT | vk::Result::NOT_READY => format!(
                "vkAcquireNextImageKHR timed out after {} s with {} image(s) in the pool — the \
                 presentation engine released none, i.e. a present that reported success did \
                 nothing",
                ACQUIRE_TIMEOUT_NS / 1_000_000_000,
                self.images.len()
            ),
            other => format!("vkAcquireNextImageKHR: {other}"),
        })?;
        // Reported, never failed on: a headless surface has no scanout to be
        // suboptimal FOR, and treating it as an error would be inventing a
        // contract the spec does not state.
        let _ = suboptimal;
        Ok(idx as usize)
    }

    /// The view for an acquired index — what `Passes::record_to` draws through.
    pub fn view(&self, idx: usize) -> vk::ImageView {
        self.views[idx]
    }

    /// The acquire semaphore and the stage that must wait on it.
    ///
    /// `COLOR_ATTACHMENT_OUTPUT` rather than `TOP_OF_PIPE`: the acquire only
    /// has to complete before anything WRITES the image, so waiting later lets
    /// the earlier part of the command buffer overlap the acquire. Correctness
    /// is identical; this is the shape B6b wants.
    pub fn wait_pair(&self) -> ([vk::Semaphore; 1], [vk::PipelineStageFlags; 1]) {
        ([self.acquire], [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
    }

    /// The render-finished semaphore FOR THIS IMAGE, signalled by the submit
    /// and waited on by the present.
    ///
    /// Indexed by the acquired image, never a single shared handle — see the
    /// field's own doc for the validation error that is.
    pub fn signal(&self, idx: usize) -> [vk::Semaphore; 1] {
        [self.render_done[idx]]
    }

    /// Present an acquired image.
    ///
    /// Waits on the render-finished semaphore, so it must follow a
    /// `run_present` that signalled it.
    pub fn present(&self, vkd: &Vk, idx: usize) -> Result<(), String> {
        let sems = [self.render_done[idx]];
        let chains = [self.swapchain];
        let indices = [idx as u32];
        let pi = vk::PresentInfoKHR::default()
            .wait_semaphores(&sems)
            .swapchains(&chains)
            .image_indices(&indices);
        unsafe { self.swapchain_fn.queue_present(vkd.queue, &pi) }
            .map(|_suboptimal| ())
            .map_err(|e| format!("vkQueuePresentKHR: {e}"))
    }

    /// Record the barrier a presented image needs after the readback copy.
    ///
    /// `Passes::record_to` leaves the target in `TRANSFER_SRC_OPTIMAL` — which
    /// is exactly where the copy wants it — so this is the ONE barrier rung 2
    /// appends rather than a re-parameterisation of that function.
    pub fn to_present_layout(&self, d: &ash::Device, cmd: vk::CommandBuffer, idx: usize) {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.images[idx])
                    .subresource_range(range)
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)],
            );
        }
    }

    /// Flood an acquired image with a byte pattern, leaving it in
    /// `TRANSFER_SRC_OPTIMAL` so the caller can copy it straight out.
    ///
    /// THE SENTINEL, and the reason it is written with a draw-less
    /// `LOAD_OP_CLEAR` rather than `vkCmdClearColorImage` is a real constraint
    /// rather than taste: the latter needs `TRANSFER_DST` usage, which the
    /// offscreen twin does not carry — so using it would put a DIFFERENCE
    /// between the two images whose equality the whole gate exists to assert.
    /// A render pass that clears and does not draw needs only the usage both
    /// already have.
    ///
    /// It separates THREE outcomes rather than the usual two, which is free
    /// and strictly better: the pattern means everything ran; the CLEAR colour
    /// means the render pass ran and the DRAW did not; the sentinel itself
    /// means the draw went to a DIFFERENT image — a real possibility with a
    /// multi-image pool and an index threaded through render, copy and
    /// present.
    pub fn clear_to(
        &self,
        d: &ash::Device,
        cmd: vk::CommandBuffer,
        idx: usize,
        colour: [f32; 4],
    ) {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.images[idx])
                    .subresource_range(range)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)],
            );

            let att = [vk::RenderingAttachmentInfo::default()
                .image_view(self.views[idx])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: colour },
                })];
            let ri = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: self.w, height: self.h },
                })
                .layer_count(1)
                .color_attachments(&att);
            d.cmd_begin_rendering(cmd, &ri);
            d.cmd_end_rendering(cmd);

            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.images[idx])
                    .subresource_range(range)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
            );
        }
    }

    /// Destroy everything, in the ONE order that is legal.
    ///
    /// Views before the swapchain (they reference its images), the swapchain
    /// before the surface, and the surface before `Vk` drops — a
    /// `VkSurfaceKHR` is INSTANCE-level while `Vk::drop` goes device ->
    /// messenger -> instance, so a surface still alive at that point outlives
    /// nothing and is destroyed after its instance.
    pub fn destroy(&self, vkd: &Vk) {
        unsafe {
            for &v in &self.views {
                vkd.device.destroy_image_view(v, None);
            }
            vkd.device.destroy_semaphore(self.acquire, None);
            for &s in &self.render_done {
                vkd.device.destroy_semaphore(s, None);
            }
            self.swapchain_fn.destroy_swapchain(self.swapchain, None);
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_fn.destroy_surface(self.surface, None);
            }
        }
    }
}

/// Choose a surface format this backend can both RENDER to and DECODE, or
/// `None`.
///
/// Takes the surface's own preference order and returns its first acceptable
/// entry, so the gate exercises what a real application would negotiate rather
/// than a format chosen to be convenient. On both ICDs measured here that is
/// `R8G8B8A8_UNORM` — deliberately the order rung 1 never renders, which is
/// what makes `display::rgb_offsets` load-bearing rather than decorative.
///
/// Two refusals, each for a stated reason:
///
/// - **`_SRGB` is refused.** `tonemap.hlsl` applies its own transfer function,
///   so an sRGB view would encode it twice. The result is too dark in a way
///   that still looks like a picture, and no tolerance in the gate would call
///   it out.
/// - **An unknown byte order is refused.** `display::rgb_offsets` is the one
///   statement of 8-bit channel order; a format missing from it cannot be
///   decoded without guessing, and a guess here is a silent swizzle.
fn pick_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
    formats.iter().copied().find(|f| decodable(f.format))
}

/// Can `display::decode` read this format correctly?
fn decodable(fmt: vk::Format) -> bool {
    // The 10-bit wire is unpacked arithmetically rather than by byte offset,
    // so it is listed here rather than in `rgb_offsets`.
    fmt == vk::Format::A2B10G10R10_UNORM_PACK32 || display::rgb_offsets(fmt).is_some()
}

/// A human-readable reason the present path stood down, for the SKIP line.
///
/// Recomputed rather than threaded out of `new`, because a SKIP is rare and a
/// precise message is worth more than the duplication: "no headless surface"
/// and "the queue family cannot present" send a reader to completely different
/// places.
pub fn skip_reason(hg: &VkHeadless) -> String {
    let vkd = &hg.vk;
    if !vkd.headless_surface {
        return "the instance has no VK_KHR_surface + VK_EXT_headless_surface pair".into();
    }
    if !vkd.info.swapchain {
        return format!("{} does not support VK_KHR_swapchain", vkd.info.name);
    }
    if !vkd.info.graphics_queue {
        return format!(
            "the queue family picked on {} is compute-only, so it cannot draw or present",
            vkd.info.name
        );
    }
    if vkd.info.kind == vk::PhysicalDeviceType::CPU {
        return format!(
            "{} is a software device, and lavapipe answers every surface query and then \
             segfaults inside vkCreateSwapchainKHR — see swapchain.rs for the backtrace; \
             FR_VK_PRESENT_SOFTWARE=1 forces the attempt when a Mesa release fixes it",
            vkd.info.name
        );
    }
    format!(
        "the surface on {} offers no usable format, present mode, or usage set",
        vkd.info.name
    )
}
