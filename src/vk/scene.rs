//! The scene on the GPU: the streams every shading path reads, plus a
//! BLAS/TLAS built through `VK_KHR_acceleration_structure`.
//!
//! The peer of `gpu::trace::SceneGpu`, and deliberately a much smaller one:
//! everything that decides WHAT to upload — the material wire format, the
//! per-material predicate maps, the geometry's opacity — already lives in
//! `gfx::scene` and `gfx::shaders`, read by both backends. What is left here
//! is the part D3D12 and Vulkan genuinely spell differently: buffers, device
//! addresses, and the acceleration-structure build.
//!
//! THREE THINGS THIS DOES NOT DO YET, each a deliberate M3b boundary rather
//! than an oversight:
//!
//! - **No staging.** Every stream is HOST_VISIBLE and written through a map.
//!   Correct everywhere and right on the UMA part this runs on; a discrete GPU
//!   would want the `STAGE_CHUNK` ring `SceneGpu` carries, which is a
//!   throughput question and not a correctness one.
//! - **No textures.** `texs[]` is an unbounded array with no descriptors
//!   written, which `descriptorBindingPartiallyBound` makes legal for a scene
//!   that fetches none. A textured scene needs images, and images need the
//!   sampler/format/mip machinery that is its own slice.
//! - **One BLAS, not `--blas-split`'s many.** One acceleration structure over
//!   the whole index stream in its original order, and one identity instance.
//!   Note that this is NOT the `--no-blas-split` arm: the split is ON by
//!   default, so `BLAS_SPLIT` is compiled in and `tri_of` goes through the
//!   remap — which is uploaded here at its single-chunk values (see
//!   `blas_tri`). The real split exists because one 34M-triangle BLAS removes
//!   the device on some drivers, so it returns with the scenes that need it.

use ash::vk;

use crate::gfx::scene as gs;
use crate::scene::Scene;
use crate::vk::device::Buffer;
use crate::vk::headless::VkHeadless;

/// Everything the tracer binds, plus what must merely stay ALIVE.
pub struct VkScene {
    pub positions: Buffer,
    pub normals: Buffer,
    pub indices: Buffer,
    pub tri_mat: Buffer,
    pub materials: Buffer,
    pub uv_buf: Buffer,
    pub mat_cutout: Buffer,
    pub mat_height: Buffer,
    pub mat_shadow: Buffer,
    /// The chunk remap `tri_of(InstanceID(), PrimitiveIndex())` reads, at its
    /// ONE-CHUNK values: `blas_tri[i] = i` and `chunk_base = [0]`.
    ///
    /// NOT optional, and the reason is a bug this gate caught on its first
    /// run. `--blas-split` is ON BY DEFAULT, so `blas_defs` arms `BLAS_SPLIT`
    /// and every intersector site goes through the remap — with these bound to
    /// a zero dummy, `tri_of` returned 0 for every hit and the whole frame
    /// shaded as triangle 0's material. It looked like a lighting bug, and the
    /// visibility gate could not see it at all: `t` comes from the ray query,
    /// while the triangle id is what indexes positions/normals/tri_mat. One
    /// BLAS over the index stream in its original order makes the identity the
    /// CORRECT remap here, so this is the single-chunk case of the real thing
    /// rather than a stand-in.
    pub blas_tri: Buffer,
    pub chunk_base: Buffer,
    /// One 16-byte buffer standing in for every declared-but-unread binding.
    /// `PARTIALLY_BOUND` would let those slots go unwritten, but a bound dummy
    /// is strictly safer and free: it turns "the shader touched a slot nobody
    /// expected" from undefined behaviour into a read of zeros.
    pub dummy: Buffer,
    pub tlas: vk::AccelerationStructureKHR,
    /// The AS backing stores. Never read through, and never droppable: an
    /// acceleration structure is a VIEW of buffer memory, and the TLAS's
    /// instance bakes the BLAS's device address — the `SceneGpu::blas`
    /// hold-it-or-lose-it rule, one API over.
    blas_mem: Buffer,
    tlas_mem: Buffer,
    blas: vk::AccelerationStructureKHR,
    accel: ash::khr::acceleration_structure::Device,
}

/// Bytes of a `#[repr(C)]` slice, for the host-visible writes below.
fn bytes_of<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

impl VkScene {
    pub fn new(hg: &VkHeadless, scene: &Scene) -> Result<VkScene, String> {
        let vkd = &hg.vk;
        if !vkd.info.ray_query || !vkd.info.accel_struct {
            return Err("the device has no ray query / acceleration structures".into());
        }
        let accel = ash::khr::acceleration_structure::Device::new(&vkd.instance, &vkd.device);

        // Usage vocabulary, named once. `AS_IN` is what the build reads
        // through a device address; `SB` is what a shader reads as a
        // StructuredBuffer.
        let sb = vk::BufferUsageFlags::STORAGE_BUFFER;
        let as_in = sb
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        let up = |v: &[u8], usage: vk::BufferUsageFlags| -> Result<Buffer, String> {
            let b = vkd.buffer(v.len().max(4) as u64, usage, true)?;
            vkd.write(&b, v)?;
            Ok(b)
        };

        // `Scene::positions` is Vec3A — 16 B each — and the shaders declare
        // `StructuredBuffer<float3>`, so the stream is repacked to 12 B on the
        // way out. That is also what the BLAS reads, at `R32G32B32_SFLOAT`
        // stride 12: one buffer, two consumers, exactly as on D3D12.
        let pos: Vec<[f32; 3]> = scene.positions.iter().map(|p| [p.x, p.y, p.z]).collect();
        let nrm: Vec<[f32; 3]> = scene.normals.iter().map(|n| [n.x, n.y, n.z]).collect();
        let uv: Vec<[f32; 2]> = scene.texcoords.iter().map(|t| [t.x, t.y]).collect();

        let positions = up(bytes_of(&pos), as_in)?;
        let indices = up(bytes_of(&scene.indices), as_in)?;
        let normals = up(bytes_of(&nrm), sb)?;
        let tri_mat = up(bytes_of(&scene.tri_mat), sb)?;
        let materials = up(bytes_of(&gs::gpu_materials(scene)), sb)?;
        let uv_buf = up(bytes_of(&uv), sb)?;
        let mat_cutout = up(bytes_of(&gs::mat_cutout(scene)), sb)?;
        let mat_height = up(bytes_of(&gs::mat_height(scene)), sb)?;
        let mat_shadow = up(bytes_of(&gs::mat_shadow(scene)), sb)?;
        let dummy = vkd.buffer(16, sb, false)?;

        let n_tris = scene.indices.len() as u32;
        let remap: Vec<u32> = (0..n_tris).collect();
        let blas_tri = up(bytes_of(&remap), sb)?;
        let chunk_base = up(bytes_of(&[0u32]), sb)?;

        let n_verts = scene.positions.len() as u32;

        // Geometry flags, term for term with `geometry_desc` on the D3D12 side
        // (see its comment for why NO_DUPLICATE_ANY_HIT is transmissive-only:
        // the cutout/relief rejects are idempotent and the tint MULTIPLY is
        // not). The predicate itself is `gfx::shaders::non_opaque`, so the
        // two backends cannot disagree about which scenes are fast-path.
        let gflags = if crate::gfx::shaders::non_opaque(scene) {
            if scene.any_transmissive {
                vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION
            } else {
                vk::GeometryFlagsKHR::empty()
            }
        } else {
            vk::GeometryFlagsKHR::OPAQUE
        };

        let tri = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vkd.buffer_device_address(&positions),
            })
            .vertex_stride(12)
            .max_vertex(n_verts.saturating_sub(1))
            .index_type(vk::IndexType::UINT32)
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vkd.buffer_device_address(&indices),
            });
        let geo = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .flags(gflags)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: tri });

        let (blas, blas_mem) = build_one(
            hg,
            &accel,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            &[geo],
            &[n_tris],
        )?;

        // ONE identity instance. `instance_custom_index` is InstanceID(), which
        // `tri_of` reads as the chunk index — 0 is the identity remap the
        // `--no-blas-split` arm compiles.
        let blas_addr = unsafe {
            accel.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(blas),
            )
        };
        let inst = vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                // The kernels' intersector is two-sided (moller_trumbore is,
                // and the CPU reference it is scored against is), so culling
                // must be disabled on the instance — D3D12 gets this from
                // never setting a cull flag on the ray.
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas_addr,
            },
        };
        let instances = up(bytes_of(std::slice::from_ref(&inst)), as_in)?;
        let inst_geo = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default().data(
                    vk::DeviceOrHostAddressConstKHR {
                        device_address: vkd.buffer_device_address(&instances),
                    },
                ),
            });
        let (tlas, tlas_mem) = build_one(
            hg,
            &accel,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            &[inst_geo],
            &[1],
        )?;
        // The instance buffer is build INPUT only — the built TLAS is
        // self-contained, exactly as on D3D12, so this is the one AS-adjacent
        // allocation that may be freed here.
        vkd.free_buffer(&instances);

        Ok(VkScene {
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
            blas,
            blas_mem,
            tlas,
            tlas_mem,
            accel,
        })
    }

    pub fn destroy(&self, hg: &VkHeadless) {
        let vkd = &hg.vk;
        unsafe {
            self.accel.destroy_acceleration_structure(self.tlas, None);
            self.accel.destroy_acceleration_structure(self.blas, None);
        }
        for b in [
            &self.positions,
            &self.normals,
            &self.indices,
            &self.tri_mat,
            &self.materials,
            &self.uv_buf,
            &self.mat_cutout,
            &self.mat_height,
            &self.mat_shadow,
            &self.blas_tri,
            &self.chunk_base,
            &self.dummy,
            &self.blas_mem,
            &self.tlas_mem,
        ] {
            vkd.free_buffer(b);
        }
    }
}

/// Size, allocate, create and BUILD one acceleration structure, returning it
/// with the buffer it is a view of.
///
/// The scratch alignment is queried rather than assumed
/// (`minAccelerationStructureScratchOffsetAlignment`): a fresh allocation's
/// base is generously aligned in practice on every driver tried, which is
/// exactly what makes an assumption here survive testing and fail somewhere
/// else. Over-allocate by the alignment and round the ADDRESS up.
fn build_one(
    hg: &VkHeadless,
    accel: &ash::khr::acceleration_structure::Device,
    ty: vk::AccelerationStructureTypeKHR,
    geos: &[vk::AccelerationStructureGeometryKHR],
    prim_counts: &[u32],
) -> Result<(vk::AccelerationStructureKHR, Buffer), String> {
    let vkd = &hg.vk;
    let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ty)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(geos);

    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &info,
            prim_counts,
            &mut sizes,
        )
    };

    let store = vkd.buffer(
        sizes.acceleration_structure_size.max(4),
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        false,
    )?;
    let handle = unsafe {
        accel.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(store.buf)
                .size(sizes.acceleration_structure_size)
                .ty(ty),
            None,
        )
    }
    .map_err(|e| {
        vkd.free_buffer(&store);
        format!("vkCreateAccelerationStructureKHR: {e}")
    })?;

    let align = scratch_align(vkd) as u64;
    let scratch = match vkd.buffer(
        sizes.build_scratch_size.max(4) + align,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        false,
    ) {
        Ok(s) => s,
        Err(e) => {
            unsafe { accel.destroy_acceleration_structure(handle, None) };
            vkd.free_buffer(&store);
            return Err(e);
        }
    };
    let scratch_addr = vkd.buffer_device_address(&scratch).div_ceil(align) * align;

    info = info
        .dst_acceleration_structure(handle)
        .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_addr });
    let ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> = prim_counts
        .iter()
        .map(|&n| vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(n))
        .collect();

    let r = hg.run(|d, cmd| unsafe {
        accel.cmd_build_acceleration_structures(cmd, &[info], &[&ranges]);
        // The build WRITES the structure; the TLAS build and every RayQuery
        // READ it. Vulkan orders those only through this edge — there is no
        // implicit AS visibility the way a D3D12 UAV barrier gives.
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::SHADER_READ,
            );
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[mb],
            &[],
            &[],
        );
    });
    vkd.free_buffer(&scratch);
    match r {
        Ok(()) => Ok((handle, store)),
        Err(e) => {
            unsafe { accel.destroy_acceleration_structure(handle, None) };
            vkd.free_buffer(&store);
            Err(e)
        }
    }
}

/// `minAccelerationStructureScratchOffsetAlignment`, read off the physical
/// device we actually opened — which is what `Vk::phys` is held for.
fn scratch_align(vkd: &crate::vk::device::Vk) -> u32 {
    let mut asp = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut asp);
    unsafe { vkd.instance.get_physical_device_properties2(vkd.phys, &mut p2) };
    asp.min_acceleration_structure_scratch_offset_alignment.max(1)
}
