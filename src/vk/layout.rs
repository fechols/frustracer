//! Descriptor-set layouts built from the corpus's own declarations, and
//! compute pipelines created against them.
//!
//! This is the Vulkan answer to `gpu/trace.rs::create_root_signature`, and it
//! is deliberately a different SHAPE than that function rather than a
//! translation of it. D3D12's root signature is written by hand — ~150 lines
//! of parameters kept in step with `src/shaders/` by care alone — and the one
//! thing that makes that survivable is that a disagreement fails PSO creation
//! loudly. Vulkan gives no such guarantee: a layout that omits a binding a
//! module uses is caught by a validation layer that may not be installed
//! (M2c's planted binding shift PASSED unvalidated, because the driver
//! resolved it to the only slot in the set). So the layout here is not
//! written at all — `vk::reflect` reads it back out of the compiled modules,
//! and this module turns that reading into `VkDescriptorSetLayout`s.
//!
//! The consequence worth stating: **adding a resource to a shader adds it to
//! the layout automatically**, and a resource two units disagree about is a
//! hard error at build time rather than a wrong-resource read at run time.
//! The register map has exactly one statement of itself, and it is the
//! `register(...)` declarations both backends already compile.
//!
//! ONE PIPELINE LAYOUT FOR EVERY KERNEL, matching D3D12, where one root
//! signature serves the whole tracer. That is not thrift: the wavefront ladder
//! binds once and dispatches a dozen kernels against it, and per-kernel
//! layouts would mean re-binding descriptor sets between dispatches that were
//! deliberately arranged not to need it.

use ash::vk;

use crate::vk::device::Vk;
use crate::vk::reflect::{DescKind, Map};

/// How many descriptors an UNBOUNDED array gets in the layout.
///
/// SPIR-V says `texs[]` has no length; Vulkan needs one. The real backend
/// sizes it per scene (D3D12's own table is `NumDescriptors: u32::MAX`, an
/// unbounded range whose heap slice is cut at init), so this is a ceiling for
/// the layout, clamped to what the device admits. Deliberately a parameter
/// rather than a constant: the gate wants the largest the device allows —
/// which is the number that says whether a real scene's texture table fits —
/// while a session wants its own scene's count.
pub const DEFAULT_TEX_CAP: u32 = 4096;

/// The descriptor-set layouts for one pipeline layout, indexed BY SET NUMBER.
///
/// Sets are contiguous from 0 because Vulkan requires it: `set_layouts[i]` is
/// set `i`, so a corpus using spaces 0 and 2 would need an empty layout at 1.
/// That is handled rather than assumed — an empty `VkDescriptorSetLayout` is
/// legal and costs nothing, and the alternative (renumbering) would break the
/// space-is-the-set rule the DXC flags encode.
pub struct Layouts {
    pub sets: Vec<vk::DescriptorSetLayout>,
    pub pipeline: vk::PipelineLayout,
}

/// Our descriptor vocabulary in Vulkan's. The mapping is the only place these
/// two enums meet, which is what keeps `vk::reflect` pure (and testable on a
/// box with no Vulkan at all).
pub fn desc_type(k: DescKind) -> vk::DescriptorType {
    match k {
        DescKind::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        DescKind::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        DescKind::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
        DescKind::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        DescKind::Sampler => vk::DescriptorType::SAMPLER,
        DescKind::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        DescKind::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        DescKind::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        DescKind::AccelStruct => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
    }
}

impl Layouts {
    /// Build the layouts a map describes.
    ///
    /// `tex_cap` sizes every unbounded binding. `drop_binding`, when set,
    /// OMITS one `(set, binding)` — the gate's teeth, and the only reason
    /// this takes an option it never uses in anger: a layout builder that is
    /// derived from the shaders cannot be tested by feeding it a wrong layout
    /// unless something is allowed to make one wrong.
    pub fn build(
        vkd: &Vk,
        map: &Map,
        tex_cap: u32,
        drop_binding: Option<(u32, u32)>,
    ) -> Result<Layouts, String> {
        let d = &vkd.device;
        let max_set = map.entries.keys().map(|&(s, _)| s).max().unwrap_or(0);
        let mut bindings: Vec<Vec<vk::DescriptorSetLayoutBinding<'static>>> =
            vec![Vec::new(); max_set as usize + 1];
        for (&(set, binding), e) in &map.entries {
            if drop_binding == Some((set, binding)) {
                continue;
            }
            let count = if e.count == 0 { tex_cap } else { e.count };
            bindings[set as usize].push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(desc_type(e.kind))
                    .descriptor_count(count)
                    // ALL rather than COMPUTE: the same map covers the display
                    // stage's vertex/fragment units, and D3D12's own root
                    // signature is `SHADER_VISIBILITY_ALL` for exactly this
                    // reason. Narrowing per set would be a second decision to
                    // keep in step with which units bind what.
                    .stage_flags(vk::ShaderStageFlags::ALL),
            );
        }

        let mut sets = Vec::with_capacity(bindings.len());
        for b in &bindings {
            // PARTIALLY_BOUND on every binding, so a set may be bound with
            // slots the dispatched kernel never touches left unwritten. That
            // is the normal case here by construction: one layout serves every
            // kernel, and no kernel uses all of it. Without it, binding a set
            // for the sky fill would require valid descriptors for the hemi
            // queues, the G-buffer pack and the whole texture table.
            let flags: Vec<vk::DescriptorBindingFlags> =
                vec![vk::DescriptorBindingFlags::PARTIALLY_BOUND; b.len()];
            let mut bf =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
            let ci = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(b)
                .push_next(&mut bf);
            let sl = unsafe { d.create_descriptor_set_layout(&ci, None) }
                .map_err(|e| format!("vkCreateDescriptorSetLayout: {e}"))?;
            sets.push(sl);
        }

        let pipeline =
            unsafe { d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sets), None) }
                .map_err(|e| format!("vkCreatePipelineLayout: {e}"))?;
        Ok(Layouts { sets, pipeline })
    }

    pub fn destroy(&self, vkd: &Vk) {
        unsafe {
            vkd.device.destroy_pipeline_layout(self.pipeline, None);
            for &s in &self.sets {
                vkd.device.destroy_descriptor_set_layout(s, None);
            }
        }
    }
}

/// Create a compute pipeline for one entry point of one module.
///
/// The module is destroyed before returning — Vulkan copies what it needs at
/// pipeline creation, so keeping it alive across a corpus-wide sweep would
/// hold dozens of megabytes for nothing.
///
/// THIS IS THE CHECK, not just a step toward one. `vkCreateComputePipelines`
/// is where a module meets a layout, so a binding the module uses and the
/// layout omits, a type that disagrees, or a SPIR-V capability the device
/// does not have all surface here — which is why the gate creates a pipeline
/// for every kernel and does nothing with it.
pub fn compute_pipeline(
    vkd: &Vk,
    layouts: &Layouts,
    words: &[u32],
    entry: &str,
) -> Result<vk::Pipeline, String> {
    let d = &vkd.device;
    let module = unsafe {
        d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
    }
    .map_err(|e| format!("vkCreateShaderModule: {e}"))?;
    let name = std::ffi::CString::new(entry).map_err(|_| "entry name has a NUL".to_string())?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&name);
    let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layouts.pipeline);
    let r = unsafe { d.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) };
    unsafe { d.destroy_shader_module(module, None) };
    match r {
        Ok(p) => Ok(p[0]),
        Err((_, e)) => Err(format!("vkCreateComputePipelines: {e}")),
    }
}

/// Create a graphics pipeline for one vertex/fragment pair, drawing into a
/// single colour attachment of `fmt`.
///
/// `compute_pipeline`'s sibling, and its doc's claim — that creation IS the
/// check — reaches FURTHER here: a graphics pipeline validates two stages
/// against one layout, so a binding the fragment shader uses and the layout
/// omits surfaces at creation rather than as a wrong-resource read at draw
/// time.
///
/// DYNAMIC RENDERING, so there is no `VkRenderPass` and no `VkFramebuffer`:
/// `VkPipelineRenderingCreateInfo` names the attachment format and the caller
/// brackets its draw with `cmd_begin_rendering`/`cmd_end_rendering`. That is
/// core in Vulkan 1.3 — this backend's floor (`device::REQ_API`) — but core
/// means MANDATED, not enabled, and `vkCreateGraphicsPipelines` enforces the
/// difference by rejecting a null `renderPass` unless the feature was switched
/// on at device creation. `DeviceInfo::dynamic_rendering` records it and
/// `open_device` enables it; a caller that reaches here on a device without it
/// gets a validation error rather than a silent wrong result.
///
/// EVERY FIXED-FUNCTION CHOICE BELOW IS FORCED BY THE CORPUS, not preferred:
///
/// * **No vertex input.** Both display shaders build a fullscreen triangle out
///   of `SV_VertexID` (`tonemap.hlsl`'s `vsmain`, `blit.hlsl`'s), so the draw is
///   three vertices against an empty binding/attribute set.
/// * **`cullMode = NONE`, and this one is load-bearing rather than lazy.** D3D
///   and Vulkan disagree about which way NDC y points, so the IDENTICAL
///   `SV_VertexID` triangle has opposite screen-space winding under the two
///   APIs. Its coverage is unaffected — it contains the whole [-1,1] square
///   either way, which is the point of the trick — but a cull mode that is
///   correct on D3D12 would discard it outright here. `gpu/tonemap.rs` sets
///   `D3D12_CULL_MODE_NONE` for these same shaders, so this is agreement with
///   the other backend, not a workaround for this one.
/// * **No depth, no stencil, no blend.** The display stage overwrites its
///   target. The one D3D12 pass that blends is the HUD's premultiplied arm,
///   which is not part of this stage.
/// * **Viewport and scissor DYNAMIC**, so one pipeline serves every target
///   size — which is what lets a gate sweep resolutions without recompiling.
///
/// TWO ash DEFAULTS ARE INVALID AND ARE SET EXPLICITLY. `Default` here is a
/// zeroed struct with only `sType` filled, so `line_width` would be 0.0 (the
/// spec requires exactly 1.0 unless `wideLines` is enabled) and
/// `rasterization_samples` would be 0 (not a legal `VkSampleCountFlagBits`).
/// Both are the kind of thing a validation layer catches and a driver without
/// one may not.
pub fn graphics_pipeline(
    vkd: &Vk,
    layouts: &Layouts,
    vs: &[u32],
    vs_entry: &str,
    ps: &[u32],
    ps_entry: &str,
    fmt: vk::Format,
) -> Result<vk::Pipeline, String> {
    let d = &vkd.device;
    let mk = |words: &[u32]| unsafe {
        d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
    };
    let vsm = mk(vs).map_err(|e| format!("vkCreateShaderModule(vs): {e}"))?;
    let psm = match mk(ps) {
        Ok(m) => m,
        Err(e) => {
            // The vertex module is already live; leaking it because the second
            // create failed would be a leak on the error path nothing cleans.
            unsafe { d.destroy_shader_module(vsm, None) };
            return Err(format!("vkCreateShaderModule(ps): {e}"));
        }
    };

    let vsn = std::ffi::CString::new(vs_entry).map_err(|_| "vs entry has a NUL".to_string())?;
    let psn = std::ffi::CString::new(ps_entry).map_err(|_| "ps entry has a NUL".to_string())?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vsm)
            .name(&vsn),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(psm)
            .name(&psn),
    ];

    let vin = vk::PipelineVertexInputStateCreateInfo::default();
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    // Counts only — the values are dynamic, so the pointers stay null.
    let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let rs = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let ms = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let atts = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&atts);
    let dynstate = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dy = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynstate);

    let fmts = [fmt];
    let mut rend = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&fmts);
    let ci = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vin)
        .input_assembly_state(&ia)
        .viewport_state(&vp)
        .rasterization_state(&rs)
        .multisample_state(&ms)
        .color_blend_state(&cb)
        .dynamic_state(&dy)
        .layout(layouts.pipeline)
        // No render pass, no subpass — the whole point of dynamic rendering.
        .render_pass(vk::RenderPass::null())
        .push_next(&mut rend);
    let r = unsafe { d.create_graphics_pipelines(vk::PipelineCache::null(), &[ci], None) };
    unsafe {
        d.destroy_shader_module(vsm, None);
        d.destroy_shader_module(psm, None);
    }
    match r {
        Ok(p) => Ok(p[0]),
        Err((_, e)) => Err(format!("vkCreateGraphicsPipelines: {e}")),
    }
}
