//! Bind-group layouts and compute pipelines, derived PER ENTRY POINT from
//! the very module wgpu is about to compile.
//!
//! The Vulkan answer to this is `vk/layout.rs`, and the shape differs in the
//! one way that matters: **there is one layout per ENTRY POINT here, not one
//! for the whole tracer.** That is not a preference, and it is not the
//! WebGPU spec's "no PARTIALLY_BOUND" clause alone — C1 MEASURED it (see
//! `headless.rs`'s header): a dispatch's usage scope is computed from the
//! pipeline LAYOUT, not from the shader's static use, so a shared layout
//! carrying `args` as read-write storage makes the indirect fill dispatch
//! conflict with itself. One layout per entry, carrying exactly what that
//! entry declares, is the only shape that validates.
//!
//! The consequence for the tracer above is real and worth stating: D3D12
//! binds one root signature and dispatches a dozen kernels against it, and
//! Vulkan binds one descriptor set per PARITY. Here every dispatch sets its
//! own bind group. The bind groups are still built ONCE at construction —
//! one per (entry, variant) pair — so the per-dispatch cost is a
//! `set_bind_group` call, not a descriptor write.
//!
//! WHAT IS DERIVED AND FROM WHAT. `vk/layout.rs` reads `crate::reflect`,
//! which parses the SPIR-V. This reads `wgsl::bindings`, which walks the
//! naga IR — the same IR the WGSL text is emitted from, one step further
//! down the chain. Two reasons, both load-bearing:
//!
//! - **WebGPU distinguishes read-only from read-write storage in the layout
//!   ENTRY**, and Vulkan does not (it is a SPIR-V decoration there, which is
//!   why `reflect::DescKind` has one `StorageBuffer` and says so). The access
//!   mode a WGSL global carries is what the browser's compiler reads; taking
//!   it from the same module the text came from makes the two unable to
//!   disagree.
//! - **`reflect` is `cfg(any(unix, windows))`** — it reads SPIR-V, which the
//!   shipped page will not carry. `wgsl::bindings` is pure and compiles on
//!   wasm, so Stage E can BAKE its output and the browser can build the same
//!   layouts from data. A derivation that only runs natively would leave the
//!   browser's layouts hand-written, which is the liability this whole
//!   derivation exists to remove.

use crate::wgsl::{self, BindDecl, BindKind, StorageFmt};

use super::device::Wgpu;

/// One entry point's layouts, plus the table they were derived from.
///
/// `groups` is indexed BY GROUP NUMBER, with `None` for a group this entry
/// declares nothing in — WebGPU's `bind_group_layouts` slice is positional
/// exactly as Vulkan's `set_layouts` is, and the corpus genuinely skips
/// group 0 in some units (a texture-only unit declares only `space1`).
pub struct EntryLayout {
    pub decls: Vec<BindDecl>,
    pub groups: Vec<Option<wgpu::BindGroupLayout>>,
    pub pipeline: wgpu::PipelineLayout,
}

impl EntryLayout {
    /// The declarations of one group, in binding order — what a bind group
    /// for that group must supply, in the order the resolver walks.
    pub fn group_decls(&self, group: u32) -> impl Iterator<Item = &BindDecl> {
        self.decls.iter().filter(move |d| d.group == group)
    }
}

/// One binding's layout entry.
///
/// `dynamic` names the bindings that take a DYNAMIC OFFSET — the tracer's
/// `b1` push block, and nothing else. WebGPU has no `vkCmdUpdateBuffer` and
/// no push constants in core, so the per-level push the wavefront ladder
/// rewrites twice per level becomes one uniform RING written before the
/// submit and selected per dispatch by offset. That makes exactly one
/// binding special in a derived layout, which `vk/layout.rs` explicitly
/// judged the worse trade for Vulkan — where an inline transfer update
/// exists. Here it is the only shape available, so the special case is
/// named, passed in by the caller, and asserted to be a uniform buffer.
fn entry_of(d: &BindDecl, dynamic: bool) -> Result<wgpu::BindGroupLayoutEntry, String> {
    use wgpu::{BindingType as B, BufferBindingType as Bb};
    let ty = match d.kind {
        BindKind::Uniform => B::Buffer {
            ty: Bb::Uniform,
            has_dynamic_offset: dynamic,
            // `None` rather than the struct's span: naga's own layout
            // validation already proved the module and the text agree, and a
            // size pinned here would be a SECOND statement of the cbuffer's
            // size with nothing keeping it in step (W5's `frame_stride` row
            // is where that pin lives, against `gfx::frame::CB_STRIDE`).
            min_binding_size: None,
        },
        BindKind::Storage { read_only } => B::Buffer {
            ty: Bb::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindKind::Texture { arrayed } => B::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: if arrayed {
                wgpu::TextureViewDimension::D2Array
            } else {
                wgpu::TextureViewDimension::D2
            },
            multisampled: false,
        },
        BindKind::StorageTexture { fmt, read, write } => B::StorageTexture {
            access: match (read, write) {
                (false, true) => wgpu::StorageTextureAccess::WriteOnly,
                (true, false) => wgpu::StorageTextureAccess::ReadOnly,
                (true, true) => wgpu::StorageTextureAccess::ReadWrite,
                // A storage texture the module neither reads nor writes is
                // not something naga emits; flag rather than pick one.
                (false, false) => {
                    return Err(format!("{:?}: a storage texture with no access", d.name));
                }
            },
            format: texture_format(fmt),
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        BindKind::Sampler { comparison } => B::Sampler(if comparison {
            wgpu::SamplerBindingType::Comparison
        } else {
            wgpu::SamplerBindingType::Filtering
        }),
    };
    if dynamic && !matches!(d.kind, BindKind::Uniform) {
        return Err(format!("{:?}: a dynamic offset was asked for a non-uniform binding", d.name));
    }
    Ok(wgpu::BindGroupLayoutEntry {
        binding: d.binding,
        // ALL the compute stage: every unit this backend builds is compute
        // (the display stage's vertex/fragment pair joins at Stage C2's
        // display half and widens this then, deliberately, rather than
        // pre-emptively).
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        // `None` — WebGPU has no unbounded binding arrays, which is the whole
        // reason `texs[]` became `gfx::texweb`'s buckets.
        count: None,
    })
}

/// The one place [`StorageFmt`] meets a wgpu type.
fn texture_format(f: StorageFmt) -> wgpu::TextureFormat {
    match f {
        StorageFmt::R32Float => wgpu::TextureFormat::R32Float,
        StorageFmt::Rg16Float => wgpu::TextureFormat::Rg16Float,
        StorageFmt::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        StorageFmt::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        StorageFmt::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
    }
}

/// A compiled entry point: the WGSL text, its module, and its layouts.
pub struct Unit {
    pub layout: EntryLayout,
    pub pipeline: wgpu::ComputePipeline,
    /// Emitted WGSL bytes — the "the translator ran" number gates print.
    pub wgsl_bytes: usize,
}

/// HLSL -> DXC -> normalize -> naga -> (bindings, WGSL) -> layouts ->
/// pipeline. The browser's whole chain, in the order the browser runs it.
///
/// `dynamic` is the `(group, binding)` set that takes dynamic offsets (see
/// [`entry_of`]). The vulkan1.1 target env is `--check-wgsl` W2's arm and
/// the C1 smoke's, for the reason recorded there: SPIR-V >= 1.4 emits
/// `OpCopyLogical`, which naga refuses.
pub fn build_unit(
    dev: &Wgpu,
    sp: &crate::spirv::Spirv,
    src: &str,
    entry: &str,
    tag: &str,
    dynamic: &[(u32, u32)],
) -> Result<Unit, String> {
    let words = sp.compile_args(src, entry, "cs_6_5", tag, false, &["-fspv-target-env=vulkan1.1"])?;
    let words = wgsl::normalize(&words);
    let module = wgsl::parse_spv(&words).map_err(|e| format!("{tag}: {e}"))?;
    let info = wgsl::validate(&module).map_err(|e| format!("{tag}: {e}"))?;
    let decls = wgsl::bindings(&module).map_err(|e| format!("{tag}: {e}"))?;
    let text = wgsl::emit_wgsl(&module, &info).map_err(|e| format!("{tag}: {e}"))?;
    let wgsl_bytes = text.len();

    let layout = build_layout(dev, tag, decls, dynamic)?;
    let shader = dev.scoped(&format!("{tag}: create_shader_module"), || {
        dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(tag),
            source: wgpu::ShaderSource::Wgsl(text.into()),
        })
    })?;
    let pipeline = dev.scoped(&format!("{tag}: create_compute_pipeline"), || {
        dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(tag),
            layout: Some(&layout.pipeline),
            module: &shader,
            // The sole @compute entry of a per-entry module — and immune to
            // naga renaming the entry on the way through.
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    })?;
    Ok(Unit { layout, pipeline, wgsl_bytes })
}

/// The layout half alone — separated so a caller with a table but no shader
/// (Stage E's baked manifest, the browser) builds layouts the same way.
pub fn build_layout(
    dev: &Wgpu,
    tag: &str,
    decls: Vec<BindDecl>,
    dynamic: &[(u32, u32)],
) -> Result<EntryLayout, String> {
    let max_group = decls.iter().map(|d| d.group).max();
    let mut groups: Vec<Option<wgpu::BindGroupLayout>> = Vec::new();
    if let Some(mg) = max_group {
        for g in 0..=mg {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = decls
                .iter()
                .filter(|d| d.group == g)
                .map(|d| entry_of(d, dynamic.contains(&(d.group, d.binding))))
                .collect::<Result<_, _>>()
                .map_err(|e| format!("{tag}: {e}"))?;
            // A group this entry declares nothing in stays `None`: an empty
            // BindGroupLayout is legal, but a hole is what the corpus means
            // and it costs nothing to say so.
            groups.push(if entries.is_empty() {
                None
            } else {
                Some(dev.scoped(&format!("{tag}: bind group layout {g}"), || {
                    dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(&format!("{tag}.g{g}")),
                        entries: &entries,
                    })
                })?)
            });
        }
    }
    let refs: Vec<Option<&wgpu::BindGroupLayout>> = groups.iter().map(|g| g.as_ref()).collect();
    let pipeline = dev.scoped(&format!("{tag}: pipeline layout"), || {
        dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(tag),
            bind_group_layouts: &refs,
            // Push constants are "immediates" in wgpu 30, and this corpus
            // declares none: `b1` is a cbuffer, and DXC has no flag to
            // promote one (the Vulkan twin's own finding).
            immediate_size: 0,
        })
    })?;
    Ok(EntryLayout { decls, groups, pipeline })
}
