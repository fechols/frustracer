//! The scene on a WebGPU device: the streams every shading path reads, and
//! the browser texture plan bound the way the browser will bind it.
//!
//! The peer of `vk::scene::VkScene`, and SMALLER than it in exactly one
//! structural way: **there is no acceleration structure here at all.**
//! That is not an omission to be filled in later — WebGPU has no ray
//! tracing, which is why the browser path is `--sw-rays` and why the whole
//! corpus traverses `rt_sw.hlsli` over OUR binary BVH. `VkScene` still
//! builds a BLAS/TLAS nothing reads under the lever (its own header says
//! so); this does not, and cannot.
//!
//! **Textures are `gfx::texweb`'s plan, not a bindless table.** WebGPU
//! cannot express `texs[]` — an unbounded binding array — so the browser
//! corpus samples exact-size `Texture2DArray` buckets keyed `(w, h, mips,
//! srgb)` through a 16-byte meta row per texture, plus a mip-0 texel buffer
//! for the `.Load`-only cutout and height paths. All of that already exists
//! and is already PROVEN: `--check-gpu` M15 renders the bucket path against
//! the bindless one on native D3D12 and byte-compares. What is new here is
//! only the WebGPU spelling of the upload, which is the easy half — WebGPU's
//! `write_texture` takes the row pitch as a parameter, so the band loop and
//! the 256-byte alignment discipline D3D12 needs both disappear.
//!
//! `mapped_at_creation` rather than a staging ring, and the trade is stated
//! rather than hidden: it costs ONE mapped copy per stream and no second
//! `Vec`, which is the same peak the Vulkan ring achieves, but it maps the
//! WHOLE buffer at once instead of a chunk. At the sizes a browser can hold
//! (Stage H's four-island ring, ~9.8M triangles) that is fine; a bundle that
//! outgrows it would need the chunked path, and Stage E is where that
//! decision belongs because it is the one that knows the bundle's size.

use crate::gfx::scene as gs;
use crate::gfx::texweb::{self, WebTexPlan};
use crate::scene::Scene;

use super::device::Wgpu;

/// Every stream the corpus binds, plus what must merely stay ALIVE.
pub struct WgpuScene {
    pub positions: wgpu::Buffer,
    pub normals: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub tri_mat: wgpu::Buffer,
    pub materials: wgpu::Buffer,
    pub uv_buf: wgpu::Buffer,
    pub mat_cutout: wgpu::Buffer,
    pub mat_height: wgpu::Buffer,
    pub mat_shadow: wgpu::Buffer,
    /// `--blas-split`'s chunk remap. Uploaded whether or not any kernel
    /// reads it: `BLAS_SPLIT` is a SESSION define (`gfx::shaders::blas_defs`)
    /// and the software intersector needs no remap, so whether these survive
    /// DXC's dead-resource strip is a property of the assembled unit — and a
    /// stream that is cheap to have and catastrophic to lack (the Vulkan
    /// suite's own `blas_tri`-on-the-dummy bug shaded a whole frame as
    /// triangle 0) is not one to make conditional.
    pub blas_tri: wgpu::Buffer,
    pub chunk_base: wgpu::Buffer,
    /// The stand-in for a slot the layout declares and this session has no
    /// resource for. 16 bytes — big enough for one row of anything, small
    /// enough that an out-of-bounds read is a WebGPU bounds-clamp rather
    /// than a wrong answer (which is the one thing WebGPU gives here that
    /// neither native backend does).
    ///
    /// The ONE buffer carrying `UNIFORM` as well as `STORAGE`: a fall-through
    /// has to serve whichever kind the slot declared, and a usage a buffer
    /// does not carry is a validation error rather than the graceful stand-in
    /// this is for. Every other buffer here is storage-only, which is what a
    /// usage set should be — this one is the deliberate exception.
    pub dummy: wgpu::Buffer,
    /// The browser texture plan: the meta/texel buffers and one texture per
    /// bucket. Always present — a texture-free scene plans zero buckets and
    /// still needs the two buffers bound (the codegen's documented
    /// dummy-bind case).
    pub tex: WebTextures,
    /// Bytes uploaded, for the gate's own report — a run that uploaded the
    /// streams and one that uploaded stubs allocate identically, and only a
    /// byte total tells them apart.
    pub bytes: u64,
}

/// `gfx::texweb`'s plan, resident.
pub struct WebTextures {
    pub plan: WebTexPlan,
    pub meta: wgpu::Buffer,
    pub texels: wgpu::Buffer,
    pub buckets: Vec<wgpu::TextureView>,
    /// The textures the views are of — views borrow, so these must outlive
    /// them, and Rust will not let them be dropped while `buckets` lives.
    _tex: Vec<wgpu::Texture>,
    pub samp_lin: wgpu::Sampler,
    pub samp_aniso: wgpu::Sampler,
    pub bytes: u64,
}

const SB: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC).union(wgpu::BufferUsages::COPY_DST);

/// One stream, repacked on the way into the mapped range.
///
/// `f` is the wire repack — `Scene::positions` is `Vec3A` (16 B) and the
/// shaders declare `StructuredBuffer<float3>` (12 B), so the conversion
/// happens HERE rather than by building a second `Vec` first. Identical in
/// intent to `vk::stage::Stage::stream`, minus the ring.
pub fn stream<T, U: Copy, F: Fn(&T) -> U>(
    dev: &Wgpu,
    label: &str,
    src: &[T],
    f: F,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let stride = std::mem::size_of::<U>();
    // A zero-length buffer is illegal, and a stream with no elements is
    // normal (a scene with no texcoords, a BVH with no tri_idx under
    // hardware rays). One element's worth of zeros keeps the binding legal
    // and reads as the dummy it is.
    let n = src.len().max(1);
    let buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (n * stride) as u64,
        usage,
        mapped_at_creation: true,
    });
    {
        let mut view = buf
            .slice(..)
            .get_mapped_range_mut()
            .expect("a buffer mapped at creation is mappable by construction");
        // WRITE-ONLY, ELEMENT BY ELEMENT. wgpu 30 hands back a `WriteOnly`
        // rather than a `&mut [u8]` deliberately: mapped memory may be
        // WRITE-COMBINING, where a read (or an atomic) does not behave the
        // way `&mut [u8]` promises. So the repacked value is built on the
        // stack and copied in — no read of the mapping, ever.
        for (i, e) in src.iter().enumerate() {
            let w = f(e);
            // SAFETY: `U` is `Copy` and `w` lives on the stack for the whole
            // borrow; `stride` is its exact size. This is a POD-to-bytes
            // view of one value, the same shape every backend here uses to
            // put a wire struct into mapped memory.
            let bytes =
                unsafe { std::slice::from_raw_parts((&w as *const U).cast::<u8>(), stride) };
            view.slice(i * stride..(i + 1) * stride).copy_from_slice(bytes);
        }
    }
    buf.unmap();
    buf
}

/// A zero-filled buffer of `bytes` — WebGPU zero-initializes by spec, so
/// this is `create_buffer` and nothing else.
fn zeros(dev: &Wgpu, label: &str, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage,
        mapped_at_creation: false,
    })
}

/// The largest single buffer this session will bind, and its bucket count —
/// the two scene rows of the device ask ([`super::device::Ask`]), computed
/// with no device in hand because they must be known BEFORE one is created.
///
/// The BVH is normally the largest; it is passed rather than derived because
/// the tracer owns the trees (the Vulkan split, verbatim: `vk::scene` has the
/// streams, `vk::tracer` has the trees).
///
/// **THE RESOLUTION IS AN INPUT**, and that is not bookkeeping: the tracer's
/// own buffers are resolution-derived (`accum` is 12 bytes per pixel, the cut
/// pool 256 per split slot), and at a large enough frame one of THOSE is the
/// largest buffer rather than the scene's BVH — `accum` alone passes the
/// 128 MB default `max_storage_buffer_binding_size` at 8K. An ask that
/// covered only the scene would open a device the tracer then cannot
/// allocate on. The browser has the same ordering problem for the same
/// reason: it must know its canvas size before `requestDevice`.
pub fn ask_for(scene: &Scene, bvh: &crate::bvh::Bvh, rw: u32, rh: u32) -> super::device::Ask {
    let plan = texweb::plan(scene);
    let node = std::mem::size_of::<gs::GpuBvhNode>() as u64;
    let ft = if crate::ftree::FTREE_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        // The wide tree is the big one, and its node count is ~1/7 of the
        // binary tree's — sized here from the binary count rather than by
        // building it, because this runs before anything is uploaded and
        // building an FTree to measure it would double the cost of asking.
        // The per-node size is `size_of`, never a transcribed number: a
        // `QFNode` that grew a field would otherwise under-ask in silence.
        (bvh.nodes.len() as u64 / 6 + 1) * std::mem::size_of::<crate::ftree::QFNode>() as u64
    } else {
        0
    };
    // The ladder's caps, from the SAME derivation the tracer allocates with.
    // `doubled_cut: true` unconditionally — `--sw-rays` may not be armed yet
    // at ask time (the gate arms it later), and the ask is an upper bound.
    let caps = super::tracer::caps_for(rw, rh, true);
    let px = u64::from(rw) * u64::from(rh);
    let big = [
        bvh.nodes.len() as u64 * node,
        ft,
        bvh.tri_idx.len() as u64 * 4,
        scene.positions.len() as u64 * 12,
        scene.normals.len() as u64 * 12,
        scene.indices.len() as u64 * 12,
        scene.texcoords.len() as u64 * 8,
        plan.texels.len() as u64 * 4,
        // The tracer's own, in `WgpuTracer::new`'s order.
        px * 12,                                              // accum
        caps.cut * 256,                                       // the cut pool
        caps.leaf * crate::gfx::shaders::LEAF_REC_BYTES,      // the leaf queue
        caps.tile * 24,                                       // one tile queue
    ];
    super::device::Ask {
        buckets: plan.buckets.len() as u32,
        buffer_bytes: big.into_iter().max().unwrap_or(0),
    }
}

impl WgpuScene {
    pub fn new(dev: &Wgpu, scene: &Scene, bvh: &crate::bvh::Bvh) -> Result<WgpuScene, String> {
        let mut bytes = 0u64;
        let mut track = |b: wgpu::Buffer| {
            bytes += b.size();
            b
        };
        let positions =
            track(stream(dev, "positions", &scene.positions, |p| [p.x, p.y, p.z], SB));
        let normals = track(stream(dev, "normals", &scene.normals, |n| [n.x, n.y, n.z], SB));
        let indices = track(stream(dev, "indices", &scene.indices, |t| *t, SB));
        let tri_mat = track(stream(dev, "tri_mat", &scene.tri_mat, |t| *t, SB));
        let mats = gs::gpu_materials(scene);
        let materials = track(stream(dev, "materials", &mats, |m| *m, SB));
        let uv_buf = track(stream(dev, "uv", &scene.texcoords, |t| [t.x, t.y], SB));
        let cut = gs::mat_cutout(scene);
        let mat_cutout = track(stream(dev, "mat_cutout", &cut, |m| *m, SB));
        let hgt = gs::mat_height(scene);
        let mat_height = track(stream(dev, "mat_height", &hgt, |m| *m, SB));
        let shd = gs::mat_shadow(scene);
        let mat_shadow = track(stream(dev, "mat_shadow", &shd, |m| *m, SB));

        // The chunk remap. Under `--no-blas-split` there is one chunk and the
        // remap IS the identity, which is what the unarmed HLSL compiles to —
        // so the identity is what gets uploaded, and the two agree by
        // construction rather than by a branch here.
        let (packed, base) = match crate::blas_split::max_prims() {
            Some(cap) => {
                let p = crate::blas_split::plan(bvh, cap);
                (p.packed_tris, p.chunk_base)
            }
            None => ((0..scene.indices.len() as u32).collect(), vec![0u32]),
        };
        let blas_tri = track(stream(dev, "blas_tri", &packed, |t| *t, SB));
        let chunk_base = track(stream(dev, "chunk_base", &base, |t| *t, SB));
        let dummy = zeros(dev, "dummy", 16, SB | wgpu::BufferUsages::UNIFORM);

        let tex = WebTextures::new(dev, scene)?;
        bytes += tex.bytes;
        Ok(WgpuScene {
            positions,
            normals,
            indices,
            tri_mat,
            materials,
            uv_buf,
            mat_cutout,
            mat_height,
            mat_shadow,
            blas_tri,
            chunk_base,
            dummy,
            tex,
            bytes,
        })
    }
}

impl WebTextures {
    pub fn new(dev: &Wgpu, scene: &Scene) -> Result<WebTextures, String> {
        let plan = texweb::plan(scene);
        let mut bytes = 0u64;
        // The meta buffer's element is the 16-byte row the generated HLSL
        // declares; an empty plan still needs one legal element behind the
        // binding, which `stream`'s `max(1)` gives.
        let meta = stream(dev, "web_meta", &plan.meta, |m| [m.w, m.h, m.bl, m.ofs], SB);
        let texels = stream(dev, "web_texels", &plan.texels, |t| *t, SB);
        bytes += meta.size() + texels.size();

        let mut tex = Vec::with_capacity(plan.buckets.len());
        let mut views = Vec::with_capacity(plan.buckets.len());
        for (bi, b) in plan.buckets.iter().enumerate() {
            let t = dev.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("web_bucket_{bi}")),
                size: wgpu::Extent3d {
                    width: b.w,
                    height: b.h,
                    depth_or_array_layers: b.layers.len() as u32,
                },
                mip_level_count: b.levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // A bucket is keyed on `srgb`, so one bucket is one format —
                // which is also why a bucket can never reproduce BC7 (the
                // `should_compress` predicate varies WITHIN a key). The
                // browser plan is RGBA8 throughout; `texture-compression-bc`
                // is Stage L's 4× download win, not this rung's.
                format: if b.srgb {
                    wgpu::TextureFormat::Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                },
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            for (layer, &ti) in b.layers.iter().enumerate() {
                let src = &scene.textures[ti as usize];
                for mip in 0..b.levels {
                    let (mw, mh, texels): (u32, u32, &[[u8; 4]]) = if mip == 0 {
                        (src.w, src.h, &src.texels)
                    } else {
                        let m = &src.mips[mip as usize - 1];
                        (m.w, m.h, &m.texels)
                    };
                    // The plan is EXACT-SIZE by construction (no resample, no
                    // padding — `texweb::plan`'s key includes w/h/levels), so
                    // a disagreement here is a plan bug and must be loud
                    // rather than a clamped copy.
                    let want = (b.w >> mip).max(1);
                    let wanth = (b.h >> mip).max(1);
                    if mw != want || mh != wanth {
                        return Err(format!(
                            "web bucket {bi} layer {layer} mip {mip}: texture is {mw}x{mh}, \
                             the bucket's level is {want}x{wanth}"
                        ));
                    }
                    let raw = texels.as_flattened();
                    bytes += raw.len() as u64;
                    dev.queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &t,
                            mip_level: mip,
                            origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        raw,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            // No 256-byte row alignment: `write_texture`
                            // takes the pitch, which is the whole reason the
                            // D3D12 band loop has no peer here.
                            bytes_per_row: Some(mw * 4),
                            rows_per_image: Some(mh),
                        },
                        wgpu::Extent3d { width: mw, height: mh, depth_or_array_layers: 1 },
                    );
                }
            }
            views.push(t.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("web_bucket_{bi}")),
                // ALWAYS D2Array, even for a one-layer bucket: the generated
                // HLSL declares `Texture2DArray` unconditionally, so the view
                // dimension is a property of the corpus, not of the layer
                // count.
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }));
            tex.push(t);
        }

        // TWO samplers, and the split is `trace_common.hlsli`'s, verbatim
        // from the Vulkan twin: `samp_lin` is trilinear and takes an explicit
        // ray-cone lod through `SampleLevel`, `samp_aniso` is hardware
        // anisotropic and is fed the elliptical footprint through
        // `SampleGrad`. `--aniso 1` makes the second a copy of the first —
        // the isotropic path verbatim, bit-identical by construction.
        //
        // WebGPU caps `anisotropy_clamp` at 16 and requires linear filtering
        // in all three of min/mag/mip when it is > 1, which this pair already
        // is.
        let samp = |aniso: u16, label: &str| {
            dev.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: aniso,
                ..Default::default()
            })
        };
        let want = crate::texture::max_aniso().clamp(1.0, 16.0) as u16;
        Ok(WebTextures {
            plan,
            meta,
            texels,
            buckets: views,
            _tex: tex,
            samp_lin: samp(1, "samp_lin"),
            samp_aniso: samp(want.max(1), "samp_aniso"),
            bytes,
        })
    }
}
