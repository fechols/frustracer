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

/// Why a swapchain could not be built — split by what the CALLER should do
/// about it, which is the same distinction `device::VkError`'s `absent` draws
/// one level up.
///
/// IT EXISTS BECAUSE ONE BODY NOW SERVES TWO CALLERS WITH OPPOSITE DUTIES.
/// `new` is a gate's constructor: a box whose surface cannot present, or whose
/// format nothing here can decode, is an environment fact and V19 must SKIP.
/// `from_surface` is a window's constructor: the user asked for a window, so
/// the identical fact is a REFUSAL that has to name itself and exit, the
/// `--fsr4` being-told doctrine. Returning a bare `Option` from `build` could
/// express the first and not the second — the caller got "no" with no way to
/// say why, so a window would have had to print "no swapchain" and leave the
/// reason on the floor.
///
/// `From<String>` is what keeps the ~10 existing `.map_err(|e| format!(..))?`
/// sites in `build` textually unchanged: they still produce a `String` and `?`
/// lifts it into `Err`.
enum Refusal {
    /// A fact about this box. A gate skips; a window refuses and says this.
    Env(String),
    /// An API call failed. Both callers propagate it.
    Err(String),
}

impl From<String> for Refusal {
    fn from(s: String) -> Refusal {
        Refusal::Err(s)
    }
}

impl Refusal {
    /// The message either way — a window has no use for the distinction, only
    /// for the sentence.
    fn text(self) -> String {
        match self {
            Refusal::Env(s) | Refusal::Err(s) => s,
        }
    }
}

/// Why an acquire or a present did not go through.
///
/// `Stale` IS NOT A FAILURE, and separating it is the whole reason this type
/// exists. `VK_ERROR_OUT_OF_DATE_KHR` says the surface no longer matches the
/// swapchain — a resize, a mode change, a monitor move — and the spec's answer
/// is to rebuild, not to give up. Folded into a string it reads as a crash, and
/// that is exactly what a window did with it before B6b rung 1: a compositor
/// resizing the window ended the session with a raw
/// `vkQueuePresentKHR: ERROR_OUT_OF_DATE_KHR` and exit 2.
///
/// IT IS SEPARATE FROM `SUBOPTIMAL_KHR`, which stays ignored below. Suboptimal
/// means the swapchain still works and the presentation engine would rather it
/// were built differently; out-of-date means it does not work at all. MEASURED
/// on RADV under XWayland: a resize from 1280x720 to 320x240 reports SUBOPTIMAL
/// and keeps presenting (the compositor scales), so the two really are
/// different answers on real hardware and a driver that reports the other one
/// is equally conformant.
///
/// `Display`, so the ~4 existing `{e}` call sites in `--check-vk`'s V19 are
/// textually unchanged by the type moving under them.
pub enum Lost {
    /// The surface and the swapchain disagree. Rebuild, or quit cleanly.
    Stale,
    /// Anything else — a real failure.
    Fatal(String),
}

impl std::fmt::Display for Lost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Named as the CAUSE rather than as the error code, since every
            // consumer that prints this has no rebuild path and is about to
            // stop.
            Lost::Stale => write!(f, "the surface no longer matches the swapchain (resized?)"),
            Lost::Fatal(s) => write!(f, "{s}"),
        }
    }
}

/// A surface, its swapchain, and the synchronisation one present cycle needs.
///
/// The surface is a HEADLESS one under `new` and a window's under
/// `from_surface`; nothing below that line can tell the difference, which is
/// what rung 2 built and what rung 1 of B6b consumes.
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

    /// Set when a `rebuild` called `vkCreateSwapchainKHR` naming this chain as
    /// `oldSwapchain` and the call did not go through. The spec retires the old
    /// chain on the CALL, not on its success — so after such a failure this
    /// handle still exists and is still ours to destroy, but every acquire and
    /// present on it answers `OUT_OF_DATE`, and naming it as `oldSwapchain`
    /// again is VUID-VkSwapchainCreateInfoKHR-oldSwapchain-01933. The next
    /// `rebuild` reads this and passes null instead, which is what turns a
    /// failed resize into a `Stale` the caller's own arm already handles rather
    /// than into a chain that can never present again.
    retired: bool,
}

/// What `Swapchain::negotiate` settled with the surface — the inputs to the one
/// call with a side effect, kept apart from it so `rebuild` can tell a refusal
/// before that call from one after it.
struct Negotiated {
    sf: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    count: u32,
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
        let built = Self::build(hg, &surface_fn, surface, w, h, vk::SwapchainKHR::null());
        match built {
            Ok(mut sc) => {
                sc.surface_fn = surface_fn;
                sc.surface = surface;
                Ok(Some(sc))
            }
            // A gate SKIPS on an environment fact and FAILS on a broken call —
            // the split `Refusal` exists to preserve now that `from_surface`
            // reads the same body the other way.
            Err(Refusal::Env(_)) => {
                unsafe { surface_fn.destroy_surface(surface, None) };
                Ok(None)
            }
            Err(Refusal::Err(e)) => {
                unsafe { surface_fn.destroy_surface(surface, None) };
                Err(e)
            }
        }
    }

    /// Build a swapchain over a surface SOMEBODY ELSE created — the window's
    /// constructor, and the one thing B6b rung 1 adds to this module.
    ///
    /// EVERY REFUSAL IS AN ERROR HERE, and that is the whole difference from
    /// `new`. A gate that finds a box which cannot present has learned an
    /// environment fact and should skip; a caller that was TOLD to open a
    /// window and cannot has to say which fact stopped it and exit non-zero
    /// (`--fsr4`'s doctrine — being told IS the feature). Silently presenting
    /// nothing, or falling back to something that is not a window, would be the
    /// worst of the three.
    ///
    /// THE SOFTWARE-DEVICE STAND-DOWN IS KEPT, and it is not inherited
    /// laziness: lavapipe advertises `VK_KHR_swapchain`, answers every
    /// capability query, and then jumps to address zero inside
    /// `vkCreateSwapchainKHR`. A window on a software device would take the
    /// process down rather than report, so this refuses BEFORE the call for the
    /// same reason `new` does, and names the device so the message is
    /// actionable.
    ///
    /// OWNERSHIP: the surface passes to us. SDL creates it and does NOT destroy
    /// it when the window drops — `vkDestroySurfaceKHR` is the application's
    /// job — and `destroy` already frees a non-null surface, so handing it over
    /// here is what makes the lifetimes come out even.
    pub fn from_surface(
        hg: &VkHeadless,
        surface: vk::SurfaceKHR,
        w: u32,
        h: u32,
    ) -> Result<Swapchain, String> {
        let vkd = &hg.vk;

        // Two of `new`'s three environment checks apply; `headless_surface`
        // deliberately does not, because the surface arrived from SDL and
        // `VK_EXT_headless_surface` has nothing to do with it. The instance
        // extension that DID matter (`VK_KHR_surface`, plus SDL's platform
        // one) was unioned in at `Vk::new` before the instance existed.
        if !vkd.info.swapchain {
            return Err(format!(
                "{} does not support VK_KHR_swapchain — this device cannot present",
                vkd.info.name
            ));
        }
        if !vkd.info.graphics_queue {
            return Err(format!(
                "the queue family picked on {} has no GRAPHICS bit — it can trace but not draw",
                vkd.info.name
            ));
        }
        let force = std::env::var("FR_VK_PRESENT_SOFTWARE").is_ok_and(|v| v != "0");
        if vkd.info.kind == vk::PhysicalDeviceType::CPU && !force {
            return Err(format!(
                "{} is a software device — lavapipe segfaults inside vkCreateSwapchainKHR, so \
                 this refuses the call rather than taking the process down. Pick a real GPU with \
                 FR_VK_DEVICE=<name>, or FR_VK_PRESENT_SOFTWARE=1 to try anyway",
                vkd.info.name
            ));
        }

        let surface_fn = ash::khr::surface::Instance::new(&vkd.entry, &vkd.instance);
        match Self::build(hg, &surface_fn, surface, w, h, vk::SwapchainKHR::null()) {
            Ok(mut sc) => {
                sc.surface_fn = surface_fn;
                sc.surface = surface;
                Ok(sc)
            }
            // The surface is NOT destroyed here: it came from the caller, who
            // still holds the window it belongs to and is about to exit. `new`
            // frees it on this path because `new` is also the one that made it.
            Err(r) => Err(r.text()),
        }
    }

    /// `old` is the swapchain being replaced, or `null` for a first build — see
    /// `rebuild`, the one caller that passes a live handle.
    ///
    /// TWO HALVES, `negotiate` then `create`, and the seam is where the spec
    /// puts a side effect: everything up to `vkCreateSwapchainKHR` is a QUERY
    /// and leaves the world as it found it, while that call retires `old`
    /// whether or not it succeeds. `rebuild` calls the halves separately so it
    /// can tell which side of the seam a refusal came from.
    fn build(
        hg: &VkHeadless,
        surface_fn: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        w: u32,
        h: u32,
        old: vk::SwapchainKHR,
    ) -> Result<Swapchain, Refusal> {
        let neg = Self::negotiate(hg, surface_fn, surface, w, h)?;
        Self::create(hg, surface, &neg, old).map_err(Refusal::Err)
    }

    /// The query half of `build`: present support, usage, format, present
    /// mode, extent and image count — every one of them a refusal BEFORE any
    /// call with a side effect, so a `rebuild` that stops here still holds a
    /// live, presentable chain.
    fn negotiate(
        hg: &VkHeadless,
        surface_fn: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        w: u32,
        h: u32,
    ) -> Result<Negotiated, Refusal> {
        let vkd = &hg.vk;

        // PRESENT SUPPORT IS PER (device, queue family, surface) and is its own
        // question: a device can advertise `VK_KHR_swapchain` while the family
        // we picked for compute cannot present on this surface.
        let supported = unsafe {
            surface_fn.get_physical_device_surface_support(vkd.phys, vkd.qfam, surface)
        }
        .map_err(|e| format!("vkGetPhysicalDeviceSurfaceSupportKHR: {e}"))?;
        if !supported {
            return Err(Refusal::Env(format!(
                "queue family {} on {} cannot present to this surface",
                vkd.qfam, vkd.info.name
            )));
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
            return Err(Refusal::Env(format!(
                "this surface supports image usage {:?}, which is missing part of the required \
                 {:?} (COLOR_ATTACHMENT to draw into it, TRANSFER_SRC to copy it out)",
                caps.supported_usage_flags, NEED_USAGE
            )));
        }

        let formats = unsafe { surface_fn.get_physical_device_surface_formats(vkd.phys, surface) }
            .map_err(|e| format!("vkGetPhysicalDeviceSurfaceFormatsKHR: {e}"))?;
        let Some(sf) = pick_format(&formats) else {
            return Err(Refusal::Env(format!(
                "none of this surface's {} format(s) is usable — see `pick_format` for the two \
                 refusals (_SRGB would double-encode the shader's own transfer function; an \
                 unknown byte order cannot be decoded)",
                formats.len()
            )));
        };

        // FIFO is the one present mode the spec guarantees, and with no
        // scanout there is nothing MAILBOX would buy — the exhaustion proof
        // wants images returned, not latency.
        let modes = unsafe {
            surface_fn.get_physical_device_surface_present_modes(vkd.phys, surface)
        }
        .map_err(|e| format!("vkGetPhysicalDeviceSurfacePresentModesKHR: {e}"))?;
        if !modes.contains(&vk::PresentModeKHR::FIFO) {
            // FIFO is the one mode the spec GUARANTEES, so this cannot fire on
            // a conformant driver — which is exactly why it stays a check
            // rather than an assumption.
            return Err(Refusal::Env(
                "this surface does not offer FIFO, the one present mode the spec guarantees"
                    .into(),
            ));
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

        Ok(Negotiated { sf, extent, count })
    }

    /// The half of `build` with side effects: `vkCreateSwapchainKHR`, then the
    /// images, their views and the semaphores one present cycle needs.
    ///
    /// UNWINDS ON EVERY ARM AFTER THE CREATE. A view or a semaphore that fails
    /// mid-loop used to leave the new chain and everything made so far alive
    /// with no handle to free them by — under a window's `rebuild` that is a
    /// chain still attached to the surface at `vkDestroySurfaceKHR`
    /// (VUID-vkDestroySurfaceKHR-surface-01266), and under `new` it is a leak
    /// the layer reports at instance teardown against an unrelated stage.
    fn create(
        hg: &VkHeadless,
        surface: vk::SurfaceKHR,
        neg: &Negotiated,
        old: vk::SwapchainKHR,
    ) -> Result<Swapchain, String> {
        let vkd = &hg.vk;
        let (sf, extent, count) = (neg.sf, neg.extent, neg.count);
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
            .clipped(true)
            // THE SPEC'S OWN RESIZE PATH, and it is a hand-off rather than a
            // hint: naming the dying chain lets the driver reuse its images
            // and keeps the surface continuously owned, which is what stops a
            // rebuild from flashing an unowned surface on compositors that
            // notice. It does NOT free the old one — a retired swapchain is
            // still the application's to destroy, which `rebuild` does after
            // this call returns. And it retires `old` EVEN IF THIS CALL FAILS,
            // which is why `rebuild` marks it so on that arm.
            .old_swapchain(old);
        let swapchain = unsafe { swapchain_fn.create_swapchain(&sci, None) }
            .map_err(|e| format!("vkCreateSwapchainKHR: {e}"))?;

        // Everything after the create is fallible and owns handles, so it runs
        // in a body whose failure frees what IT made, and then this frees the
        // chain — one unwind per level, in reverse creation order.
        let attach = || -> Result<(Vec<vk::Image>, Vec<vk::ImageView>, vk::Semaphore, Vec<vk::Semaphore>), String> {
            let images = unsafe { swapchain_fn.get_swapchain_images(swapchain) }
                .map_err(|e| format!("vkGetSwapchainImagesKHR: {e}"))?;

            let mut views: Vec<vk::ImageView> = Vec::with_capacity(images.len());
            let unwind_views = |views: &[vk::ImageView]| unsafe {
                for &v in views {
                    vkd.device.destroy_image_view(v, None);
                }
            };
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
                };
                match view {
                    Ok(v) => views.push(v),
                    Err(e) => {
                        unwind_views(&views);
                        return Err(format!("vkCreateImageView(swapchain): {e}"));
                    }
                }
            }

            let sem = || unsafe {
                vkd.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            };
            let acquire = match sem() {
                Ok(s) => s,
                Err(e) => {
                    unwind_views(&views);
                    return Err(format!("vkCreateSemaphore(acquire): {e}"));
                }
            };
            let mut render_done: Vec<vk::Semaphore> = Vec::with_capacity(images.len());
            for _ in 0..images.len() {
                match sem() {
                    Ok(s) => render_done.push(s),
                    Err(e) => {
                        unsafe {
                            for &s in &render_done {
                                vkd.device.destroy_semaphore(s, None);
                            }
                            vkd.device.destroy_semaphore(acquire, None);
                        }
                        unwind_views(&views);
                        return Err(format!("vkCreateSemaphore(render): {e}"));
                    }
                }
            }
            Ok((images, views, acquire, render_done))
        };
        let (images, views, acquire, render_done) = match attach() {
            Ok(t) => t,
            Err(e) => {
                unsafe { swapchain_fn.destroy_swapchain(swapchain, None) };
                return Err(e);
            }
        };

        Ok(Swapchain {
            // Filled by the CALLER — `build` does not own the surface, so that
            // `new` can free it on every early return above and `from_surface`
            // can leave it with the window it belongs to.
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
            retired: false,
        })
    }

    /// Replace this swapchain with one at a new extent, KEEPING THE SURFACE —
    /// B6b rung 3, and the one thing `destroy` cannot express.
    ///
    /// `destroy` frees the surface along with everything above it, which is
    /// right for both of its callers (a gate that made its own headless
    /// surface, and a window shutting down) and wrong for a resize: SDL's
    /// surface belongs to the WINDOW and must outlive every swapchain built on
    /// it. Destroying and re-creating it would also mean re-running
    /// `SDL_Vulkan_CreateSurface` from a thread that does not own the window.
    ///
    /// `vkDeviceWaitIdle` FIRST, and it is load-bearing rather than defensive.
    /// `Presenter::present` ends in `wait_submit`, so the SUBMIT is retired —
    /// but `vkQueuePresentKHR`'s wait on `render_done[idx]` has no CPU-visible
    /// completion at all. That is the very argument the `render_done` field's
    /// own doc makes for why there is one semaphore per image; destroying that
    /// array while a present may still be waiting inside it is a
    /// use-after-free, and the only thing standing between here and it is this
    /// wait. V20 perturbs it away to prove the layer notices.
    ///
    /// THE NEW CHAIN IS BUILT BEFORE THE OLD ONE IS TORN DOWN, so a refusal
    /// leaves `self` intact and still destroyable — and, up to a point the
    /// spec fixes, still PRESENTABLE. `negotiate` is queries only, so a
    /// refusal from it changes nothing. `create` names `self.swapchain` as
    /// `oldSwapchain`, and the spec retires it on the call whether or not the
    /// call succeeds: after a refusal from that half the chain still exists and
    /// `destroy` still frees it, but every acquire and present answers
    /// `OUT_OF_DATE`, and naming it as `oldSwapchain` again would be
    /// VUID-VkSwapchainCreateInfoKHR-oldSwapchain-01933. So that arm sets
    /// `retired`, the next `rebuild` passes null in its place, and a caller
    /// with a `Stale` arm (the window has one) recovers on its own rather than
    /// holding a chain that can never present again. The reverse order — tear
    /// down, then build — reads more naturally and would leave a half-freed
    /// swapchain that the caller's own cleanup then double-frees on the way
    /// out.
    pub fn rebuild(&mut self, hg: &VkHeadless, w: u32, h: u32) -> Result<(), String> {
        let vkd = &hg.vk;
        unsafe { vkd.device.device_wait_idle() }
            .map_err(|e| format!("vkDeviceWaitIdle before a swapchain rebuild: {e}"))?;

        // The query half first: nothing here has happened yet if it refuses.
        let neg = Self::negotiate(hg, &self.surface_fn, self.surface, w, h)
            .map_err(|r| r.text())?;

        // `old_swapchain` is the handle we are replacing: the driver may reuse
        // its images, and it is RETIRED rather than freed by the call. A chain
        // an earlier failed rebuild already retired may not be named again.
        let old = if self.retired { vk::SwapchainKHR::null() } else { self.swapchain };
        let new = match Self::create(hg, self.surface, &neg, old) {
            Ok(n) => n,
            Err(e) => {
                // The call was made, so `old` is retired whatever it returned.
                self.retired = true;
                return Err(e);
            }
        };

        unsafe {
            for &v in &self.views {
                vkd.device.destroy_image_view(v, None);
            }
            vkd.device.destroy_semaphore(self.acquire, None);
            for &s in &self.render_done {
                vkd.device.destroy_semaphore(s, None);
            }
            // The retired chain, freed through the OLD dispatch table — the new
            // one below has not been installed yet, and they are equivalent
            // anyway (both are loaded from the same instance + device).
            self.swapchain_fn.destroy_swapchain(self.swapchain, None);
        }

        // The surface and its loader are DELIBERATELY not touched: `build`
        // hands back a null surface and a fresh `surface_fn` of its own (see
        // its tail), and taking those would drop the window's surface on the
        // floor — the leak `from_surface`'s ownership comment exists to
        // prevent, arriving from the other direction.
        self.swapchain_fn = new.swapchain_fn;
        self.swapchain = new.swapchain;
        self.fmt = new.fmt;
        self.w = new.w;
        self.h = new.h;
        self.images = new.images;
        self.views = new.views;
        self.acquire = new.acquire;
        self.render_done = new.render_done;
        self.retired = false;
        Ok(())
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
    pub fn acquire(&self) -> Result<usize, Lost> {
        let (idx, suboptimal) = unsafe {
            self.swapchain_fn.acquire_next_image(
                self.swapchain,
                ACQUIRE_TIMEOUT_NS,
                self.acquire,
                vk::Fence::null(),
            )
        }
        .map_err(|e| match e {
            vk::Result::ERROR_OUT_OF_DATE_KHR => Lost::Stale,
            vk::Result::TIMEOUT | vk::Result::NOT_READY => Lost::Fatal(format!(
                "vkAcquireNextImageKHR timed out after {} s with {} image(s) in the pool — the \
                 presentation engine released none, i.e. a present that reported success did \
                 nothing",
                ACQUIRE_TIMEOUT_NS / 1_000_000_000,
                self.images.len()
            )),
            other => Lost::Fatal(format!("vkAcquireNextImageKHR: {other}")),
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
    pub fn present(&self, vkd: &Vk, idx: usize) -> Result<(), Lost> {
        let sems = [self.render_done[idx]];
        let chains = [self.swapchain];
        let indices = [idx as u32];
        let pi = vk::PresentInfoKHR::default()
            .wait_semaphores(&sems)
            .swapchains(&chains)
            .image_indices(&indices);
        unsafe { self.swapchain_fn.queue_present(vkd.queue, &pi) }
            // Suboptimal is a SUCCESS code and stays ignored — see `Lost`.
            .map(|_suboptimal| ())
            .map_err(|e| match e {
                vk::Result::ERROR_OUT_OF_DATE_KHR => Lost::Stale,
                other => Lost::Fatal(format!("vkQueuePresentKHR: {other}")),
            })
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
