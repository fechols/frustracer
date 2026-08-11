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
//! Every stream here is DEVICE_LOCAL and written through `vk::stage`'s ring —
//! the M3b "no staging" boundary, closed. What that buys is in that module's
//! header; what it costs here is that the streams which needed a repack
//! (`Vec3A` -> `float3`) or had no source slice at all (`blas_tri`'s identity
//! remap) are now GENERATED into the ring a chunk at a time rather than
//! collected first, so the `Vec`s they used to build are gone with them.
//!
//! **`--blas-split` is REAL here** (the M3b "one BLAS" boundary, closed): one
//! acceleration structure per maximal BVH subtree under the cap, each instanced
//! identity into the TLAS with `InstanceID` = the chunk index, built worst-case
//! with `ALLOW_COMPACTION` and compacted into an exact-size arena. The planner,
//! the vertex windowing, and every structural contract they satisfy are
//! `blas_split`'s — shared with D3D12 and gated purely in `--check`, so what is
//! new below is only the Vulkan spelling: arena sub-allocation, a compacted-size
//! query pool, and the per-chunk geometry's device addresses.
//!
//! TWO THINGS IT STILL DOES NOT DO, both deliberate, and both about a consumer
//! that does not exist on this backend yet:
//!
//! - **`--dxr-sbt`'s class refinement.** `refine_by_class` cuts each chunk into
//!   per-shading-class sub-chunks so an instance can carry its class as
//!   `InstanceContributionToHitGroupIndex`. That index means something only to
//!   a ray-tracing PIPELINE's shader binding table; this backend has RayQuery
//!   and nothing else, so refining here would multiply the instance count to
//!   feed a table no ray consults.
//! - **`--foliage-sway`'s animated TLAS ring.** D3D12 pulls leaf triangles into
//!   per-cell chunks and rebuilds a TLAS per frame from sheared instances — but
//!   its STATIC TLAS still holds every chunk at identity, and that rest pose is
//!   what every headless gate traces. There is no per-frame driver here, so
//!   this builds exactly that static rest-pose TLAS and never calls
//!   `foliage::split_plan`. The consequence is worth stating precisely: the two
//!   backends' PARTITIONS differ chunk for chunk on a swaying scene, and their
//!   IMAGES do not, because `tri_of` follows whatever partition it was handed.

use ash::vk;

use crate::blas_split::ChunkWindow;
use crate::gfx::scene as gs;
use crate::scene::Scene;
use crate::vk::device::Buffer;
use crate::vk::headless::VkHeadless;
use crate::vk::stage::Stage;

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
    /// The chunk remap `tri_of(InstanceID(), PrimitiveIndex())` reads:
    /// `blas_tri` is the plan's chunk-major `packed_tris`, `chunk_base` its
    /// per-chunk start (with the trailing sentinel). Under `--no-blas-split`
    /// the shaders compile `tri_of` as the identity and never read either, but
    /// they are still filled with the values that WOULD be correct — see the
    /// unarmed arm in `new`.
    ///
    /// NOT optional, and the reason is a bug this gate caught on its first run.
    /// `--blas-split` is ON BY DEFAULT, so `blas_defs` arms `BLAS_SPLIT` and
    /// every intersector site goes through the remap — with these bound to a
    /// zero dummy, `tri_of` returned 0 for every hit and the whole frame shaded
    /// as triangle 0's material. It looked like a lighting bug, and the
    /// visibility gate could not see it at all: `t` comes from the ray query,
    /// while the triangle id is what indexes positions/normals/tri_mat.
    pub blas_tri: Buffer,
    pub chunk_base: Buffer,
    /// One 16-byte buffer standing in for every declared-but-unread binding.
    /// `PARTIALLY_BOUND` would let those slots go unwritten, but a bound dummy
    /// is strictly safer and free: it turns "the shader touched a slot nobody
    /// expected" from undefined behaviour into a read of zeros — which became
    /// literally true with the staging ring, since it can zero-fill (before,
    /// this was device-local memory nobody had written).
    pub dummy: Buffer,
    /// Bytes staged and submits spent, for the report line.
    pub staged: (u64, u32),
    /// Chunks the plan cut, and the arena/scratch sizes it cost. The count is
    /// what tells an armed run that exercised the remap from one that produced
    /// a single chunk and exercised nothing — `--check-vk`'s anti-vacuity.
    pub n_chunks: u32,
    pub blas_report: String,
    pub tlas: vk::AccelerationStructureKHR,
    /// The AS backing stores. Never read through, and never droppable: an
    /// acceleration structure is a VIEW of buffer memory, and the TLAS's
    /// instances bake the BLASes' device addresses — the `SceneGpu::blas`
    /// hold-it-or-lose-it rule, one API over. `blas_mem` is the ONE arena every
    /// chunk structure is sub-allocated from.
    blas_mem: Buffer,
    tlas_mem: Buffer,
    blas: Vec<vk::AccelerationStructureKHR>,
    accel: ash::khr::acceleration_structure::Device,
}

impl VkScene {
    /// `bvh` is the RAY BVH, and it is here for one reason: `--blas-split`
    /// cuts the acceleration structure along that tree's own subtrees, which
    /// is what makes each chunk a spatially coherent group and what a
    /// cut-driven TLAS rebuild would address chunks by.
    pub fn new(hg: &VkHeadless, scene: &Scene, bvh: &crate::bvh::Bvh) -> Result<VkScene, String> {
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

        // One ring for the whole set, sized to it (so a small scene commits
        // kilobytes, not the cap) and freed before this returns — the D3D12
        // discipline that keeps peak commit at steady-state plus one chunk
        // rather than twice steady-state. `atom` is one element: every wire
        // type here is tens of bytes, so the 4 KB floor covers it, but it is
        // passed rather than assumed because `Stage` treats it as a
        // correctness floor and not a hint.
        let mut stage = Stage::new(vkd, gs::stream_bytes(scene) as u64, 256)?;

        // The whole upload runs inside one closure so a failure anywhere in it
        // still frees the ring. Everything ELSE it allocated leaks on that
        // path, exactly as before — a scene that cannot upload ends the run,
        // and a per-stream unwind would be machinery serving nothing.
        let built = (|| -> Result<VkScene, String> {
            // `Scene::positions` is Vec3A — 16 B each — and the shaders declare
            // `StructuredBuffer<float3>`, so the stream is repacked to 12 B on
            // the way out. That repack used to build a whole `Vec` first; now
            // it happens INSIDE the ring, one chunk at a time. The same buffer
            // is what the BLAS reads, at `R32G32B32_SFLOAT` stride 12: one
            // buffer, two consumers, exactly as on D3D12.
            let positions = stage.stream(hg, &scene.positions, |p| [p.x, p.y, p.z], as_in)?;
            let indices = stage.stream(hg, &scene.indices, |t| *t, as_in)?;
            let normals = stage.stream(hg, &scene.normals, |n| [n.x, n.y, n.z], sb)?;
            let tri_mat = stage.stream(hg, &scene.tri_mat, |t| *t, sb)?;
            let mats = gs::gpu_materials(scene);
            let materials = stage.stream(hg, &mats, |m| *m, sb)?;
            let uv_buf = stage.stream(hg, &scene.texcoords, |t| [t.x, t.y], sb)?;
            let cut = gs::mat_cutout(scene);
            let mat_cutout = stage.stream(hg, &cut, |m| *m, sb)?;
            let hgt = gs::mat_height(scene);
            let mat_height = stage.stream(hg, &hgt, |m| *m, sb)?;
            let shd = gs::mat_shadow(scene);
            let mat_shadow = stage.stream(hg, &shd, |m| *m, sb)?;
            let dummy = stage.zeros(hg, 16, sb)?;

            let n_tris = scene.indices.len();
            let n_verts = scene.positions.len() as u32;

            // Geometry flags, term for term with `geometry_desc` on the D3D12
            // side (see its comment for why NO_DUPLICATE_ANY_HIT is
            // transmissive-only: the cutout/relief rejects are idempotent and
            // the tint MULTIPLY is not). The predicate itself is
            // `gfx::shaders::non_opaque`, so the two backends cannot disagree
            // about which scenes are fast-path.
            let gflags = if crate::gfx::shaders::non_opaque(scene) {
                if scene.any_transmissive {
                    vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION
                } else {
                    vk::GeometryFlagsKHR::empty()
                }
            } else {
                vk::GeometryFlagsKHR::OPAQUE
            };

            let (blas, blas_mem, blas_tri, chunk_base, n_chunks, blas_report) =
                match crate::blas_split::max_prims() {
                    Some(cap) => {
                        let plan = crate::blas_split::plan(bvh, cap);
                        if plan.chunks() == 0 {
                            return Err("--blas-split: the scene has no triangles to chunk".into());
                        }
                        // The chunk index rides InstanceID's 24 bits, and a
                        // silent wrap would remap every triangle in the
                        // overflowing chunks — the D3D12 check, verbatim,
                        // because the ceiling is the API's on both.
                        if plan.chunks() > (1 << 24) {
                            return Err(format!(
                                "--blas-split {cap}: {} chunks exceeds the 2^24 InstanceID \
                                 ceiling (raise the cap)",
                                plan.chunks()
                            ));
                        }
                        // PER-CHUNK VERTEX WINDOWING — the RDNA4 index-value
                        // workaround (`blas_split::plan_windows` carries the
                        // defect write-up and `self_test` pins the rule). It
                        // runs here rather than being D3D12-only for two
                        // reasons: it is what keeps a chunk's index VALUES
                        // small, which is a property of the geometry desc and
                        // not of any one API, and the defect was found on AMD
                        // hardware this backend also runs on.
                        let no_rebase =
                            std::env::var("FR_SPLIT_NOREBASE").is_ok_and(|v| v == "1");
                        if no_rebase {
                            eprintln!("blas-split: FR_SPLIT_NOREBASE=1 — absolute BLAS index values");
                        }
                        let wins = crate::blas_split::plan_windows(
                            &plan,
                            &scene.indices,
                            |v| {
                                let p = scene.positions[v as usize];
                                [p.x, p.y, p.z]
                            },
                            no_rebase,
                        );
                        if wins.gathered() > 0 {
                            eprintln!(
                                "blas-split: {} chunk(s) vertex-gathered ({} KB side buffer) — id \
                                 range over the {} ceiling (the RDNA4 index-value workaround)",
                                wins.gathered(),
                                (wins.aux.len() * 12) >> 10,
                                crate::blas_split::SPLIT_INDEX_CEILING,
                            );
                        }
                        let aux = if wins.aux.is_empty() {
                            None
                        } else {
                            Some(stage.stream(hg, &wins.aux, |v| *v, as_in)?)
                        };
                        // The REORDERED index stream: chunk-major, windowed.
                        // Generated into the ring rather than collected — 12
                        // B/tri is 413 MB on a tiled scene, and it feeds only
                        // the builds (a built AS is self-contained), so it is
                        // freed below.
                        //
                        // `chunk_of` walks a cursor because `stream_gen` calls
                        // `make` in element order, and FALLS BACK to a binary
                        // search when it is called out of order — which never
                        // happens today, and which is exactly the assumption a
                        // future parallel ring would break silently.
                        let cursor = std::cell::Cell::new(0usize);
                        let chunk_of = |i: usize| {
                            let mut c = cursor.get();
                            if i < plan.chunk_base[c] as usize {
                                c = plan.chunk_base.partition_point(|&b| b as usize <= i) - 1;
                            }
                            while i >= plan.chunk_base[c + 1] as usize {
                                c += 1;
                            }
                            cursor.set(c);
                            c
                        };
                        let blas_indices = stage.stream_gen(
                            hg,
                            plan.packed_tris.len(),
                            |i| {
                                let c = chunk_of(i);
                                wins.tri(c, scene.indices[plan.packed_tris[i] as usize])
                            },
                            as_in,
                        )?;
                        let blas_tri = stage.stream(hg, &plan.packed_tris, |t| *t, sb)?;
                        let chunk_base = stage.stream(hg, &plan.chunk_base, |b| *b, sb)?;
                        if std::env::var("FR_SPLIT_AUDIT").is_ok_and(|v| v == "1") {
                            audit(hg, &stage, &blas_tri, &plan.packed_tris, "blas_tri")?;
                            audit(hg, &stage, &chunk_base, &plan.chunk_base, "chunk_base")?;
                            let expected: Vec<u32> = (0..plan.chunks())
                                .flat_map(|i| {
                                    plan.tris(i)
                                        .iter()
                                        .flat_map(|&t| wins.tri(i, scene.indices[t as usize]))
                                        .collect::<Vec<u32>>()
                                })
                                .collect();
                            audit(hg, &stage, &blas_indices, &expected, "blas_indices")?;
                        }
                        // Per-chunk geometry. The vertex WINDOW is
                        // `plan_windows`' decision, and both of its arms are
                        // just an address plus a count: `Rebase` slides into
                        // the SHARED position buffer (nothing duplicates),
                        // `Gather` points at this chunk's slice of the side
                        // buffer. Both offsets are multiples of 12, hence of 4
                        // — the alignment `R32G32B32_SFLOAT` requires of
                        // `vertexData` and `UINT32` of `indexData`.
                        let pos_addr = vkd.buffer_device_address(&positions);
                        let idx_addr = vkd.buffer_device_address(&blas_indices);
                        let aux_addr = aux.as_ref().map(|b| vkd.buffer_device_address(b));
                        let geos: Vec<vk::AccelerationStructureGeometryKHR> = (0..plan.chunks())
                            .map(|i| {
                                let (vcount, vaddr) = match wins.win[i] {
                                    ChunkWindow::Rebase(base) => {
                                        (n_verts - base, pos_addr + base as u64 * 12)
                                    }
                                    ChunkWindow::Gather { base, count } => (
                                        count,
                                        aux_addr.expect("gather chunks imply an aux buffer")
                                            + base as u64 * 12,
                                    ),
                                };
                                vk::AccelerationStructureGeometryKHR::default()
                                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                                    .flags(gflags)
                                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                                        triangles: triangles(
                                            vaddr,
                                            vcount,
                                            idx_addr + plan.chunk_base[i] as u64 * 12,
                                        ),
                                    })
                            })
                            .collect();
                        let prims: Vec<u32> = (0..plan.chunks()).map(|i| plan.prims(i)).collect();
                        let split = build_blas_set(hg, &accel, &geos, &prims);
                        // The reordered index stream and the gathered side
                        // buffer fed only the builds — a built acceleration
                        // structure is self-contained — so they go back now
                        // rather than resting for the session. `indices`
                        // (original order) stays: that one is what SHADING
                        // reads.
                        vkd.free_buffer(&blas_indices);
                        if let Some(a) = &aux {
                            vkd.free_buffer(a);
                        }
                        let (handles, arena, report) = split?;
                        let (lo, mean, hi) = plan.stats();
                        (
                            handles,
                            arena,
                            blas_tri,
                            chunk_base,
                            plan.chunks() as u32,
                            format!(
                                "{} chunks (prims min {lo} mean {mean:.0} max {hi}, cap {cap}), \
                                 {report}",
                                plan.chunks()
                            ),
                        )
                    }
                    None => {
                        // `--no-blas-split`: ONE structure over the index
                        // stream in its ORIGINAL order, so the shaders'
                        // compiled-out `tri_of` (the identity) is the correct
                        // remap. The remap buffers are filled with the values
                        // that would agree with it — GENERATED, never
                        // collected, since at 34.4M triangles the identity is
                        // 138 MB of `i`.
                        let blas_tri = stage.stream_gen(hg, n_tris, |i| i as u32, sb)?;
                        let chunk_base = stage.stream(hg, &[0u32, n_tris as u32], |v| *v, sb)?;
                        let tri = triangles(
                            vkd.buffer_device_address(&positions),
                            n_verts,
                            vkd.buffer_device_address(&indices),
                        );
                        let geo = vk::AccelerationStructureGeometryKHR::default()
                            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                            .flags(gflags)
                            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: tri });
                        let (blas, mem, report) =
                            build_blas_set(hg, &accel, &[geo], &[n_tris as u32])?;
                        (
                            blas,
                            mem,
                            blas_tri,
                            chunk_base,
                            1,
                            format!("one BLAS over the whole scene (--no-blas-split), {report}"),
                        )
                    }
                };

            // Identity instances, `instance_custom_index` = the chunk index —
            // which is InstanceID(), `tri_of`'s first coordinate.
            let insts: Vec<vk::AccelerationStructureInstanceKHR> = blas
                .iter()
                .enumerate()
                .map(|(i, &h)| {
                    let addr = unsafe {
                        accel.get_acceleration_structure_device_address(
                            &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                                .acceleration_structure(h),
                        )
                    };
                    vk::AccelerationStructureInstanceKHR {
                        transform: vk::TransformMatrixKHR {
                            matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        },
                        instance_custom_index_and_mask: vk::Packed24_8::new(i as u32, 0xff),
                        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                            0,
                            // The kernels' intersector is two-sided
                            // (moller_trumbore is, and the CPU reference it is
                            // scored against is), so culling must be disabled
                            // on the instance — D3D12 gets this from never
                            // setting a cull flag on the ray.
                            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw()
                                as u8,
                        ),
                        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                            device_handle: addr,
                        },
                    }
                })
                .collect();
            let instances = stage.stream(hg, &insts, |i| *i, as_in)?;
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
            let tlas_built = build_one(
                hg,
                &accel,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                &[inst_geo],
                &[insts.len() as u32],
            );
            // The instance buffer is build INPUT only — the built TLAS is
            // self-contained, exactly as on D3D12, so this is the one
            // AS-adjacent allocation that may be freed here.
            vkd.free_buffer(&instances);
            let (tlas, tlas_mem) = tlas_built?;

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
                staged: (stage.bytes(), stage.chunks()),
                n_chunks,
                blas_report,
                blas,
                blas_mem,
                tlas,
                tlas_mem,
                accel: accel.clone(),
            })
        })();
        stage.free(vkd);
        built
    }

    pub fn destroy(&self, hg: &VkHeadless) {
        let vkd = &hg.vk;
        unsafe {
            self.accel.destroy_acceleration_structure(self.tlas, None);
            for &h in &self.blas {
                self.accel.destroy_acceleration_structure(h, None);
            }
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

/// `VkAccelerationStructureCreateInfoKHR::offset` must be a multiple of 256 —
/// the same number D3D12 spells
/// `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT`, which is why the
/// arena arithmetic below is `build_split_blas`'s line for line.
const AS_ALIGN: u64 = 256;

fn align_as(x: u64) -> u64 {
    x.div_ceil(AS_ALIGN) * AS_ALIGN
}

/// One chunk's triangle geometry: a window into the SHARED vertex buffer and a
/// slice of an index buffer. The whole-scene case is the same call with the
/// window opened at 0 and the original index stream — which is what keeps the
/// `--no-blas-split` arm from being a second spelling of the same descriptor.
fn triangles(
    vertex_addr: vk::DeviceAddress,
    n_verts: u32,
    index_addr: vk::DeviceAddress,
) -> vk::AccelerationStructureGeometryTrianglesDataKHR<'static> {
    vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vertex_addr })
        .vertex_stride(12)
        .max_vertex(n_verts.saturating_sub(1))
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR { device_address: index_addr })
}

/// Build one BLAS per geometry and COMPACT them all into an exact-size arena.
/// Returns the per-structure handles, the arena they are views of, and the size
/// report.
///
/// Both arms go through this — `--blas-split`'s N chunk geometries and
/// `--no-blas-split`'s single whole-scene one — which is deliberate rather than
/// incidental: with the unarmed arm on a separate uncompacted path, its size
/// report would be a worst-case number beside the split's compacted one, and
/// the A/B those two lines invite would read a compaction win as a split win.
///
/// Shape mirrors `gpu::trace::build_split_blas` — build worst-case with
/// `ALLOW_COMPACTION`, query the compacted sizes, copy into an exact arena —
/// for its reasons: compaction is 40-50% of BLAS memory, so an uncompacted
/// split would COST memory against the single build rather than saving it, and
/// at a few hundred structures the extra queries are noise. Builds run SERIALLY
/// through one shared scratch buffer with a barrier between them; that barrier
/// IS the serialization the sharing requires, not an optimization to remove.
///
/// ONE DELIBERATE ABSENCE vs D3D12: no VRAM pre-flight. That check exists there
/// because WDDM silently DEMOTES an over-budget commit to system memory and
/// renders at a tenth the speed; here `vkAllocateMemory` returns
/// `ERROR_OUT_OF_DEVICE_MEMORY` and `Vk::buffer` turns it into a loud failure,
/// so the quiet-wrong outcome the pre-flight guards against does not exist.
/// `VK_EXT_memory_budget` is the instrument if a predictive one is ever wanted.
fn build_blas_set(
    hg: &VkHeadless,
    accel: &ash::khr::acceleration_structure::Device,
    geos: &[vk::AccelerationStructureGeometryKHR],
    prims: &[u32],
) -> Result<(Vec<vk::AccelerationStructureKHR>, Buffer, String), String> {
    let vkd = &hg.vk;
    let d = &vkd.device;
    let n = geos.len();

    let bflags = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION;
    let info_for = |i: usize| {
        vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(bflags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geos[i..i + 1])
    };

    // Sizing pass: worst-case arena offsets + the scratch high-water mark.
    let mut build_off = Vec::with_capacity(n);
    let mut build_size = Vec::with_capacity(n);
    let mut total_build = 0u64;
    let mut scratch_max = 0u64;
    for i in 0..n {
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info_for(i),
                &[prims[i]],
                &mut sizes,
            )
        };
        build_off.push(total_build);
        build_size.push(sizes.acceleration_structure_size);
        total_build = align_as(total_build + sizes.acceleration_structure_size);
        scratch_max = scratch_max.max(sizes.build_scratch_size);
    }

    let store_usage = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    let build_arena = vkd.buffer(total_build.max(4), store_usage, false)?;
    let align = scratch_align(vkd) as u64;
    let scratch = vkd.buffer(
        scratch_max.max(4) + align,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        false,
    )?;
    let scratch_addr = vkd.buffer_device_address(&scratch).div_ceil(align) * align;

    // Everything from here can fail with structures already created, so the
    // whole build runs in a closure and one cleanup path frees what exists.
    let mut built: Vec<vk::AccelerationStructureKHR> = Vec::with_capacity(n);
    let mut final_as: Vec<vk::AccelerationStructureKHR> = Vec::new();
    let mut arena: Option<Buffer> = None;
    let mut pool = vk::QueryPool::null();
    let r = (|| -> Result<String, String> {
        for i in 0..n {
            built.push(create_as(
                accel,
                &build_arena,
                build_off[i],
                build_size[i],
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            )?);
        }
        pool = unsafe {
            d.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
                    .query_count(n as u32),
                None,
            )
        }
        .map_err(|e| format!("vkCreateQueryPool(compacted size, {n}): {e}"))?;

        // Submit 1: every chunk build, then the compacted-size query. The
        // barrier after EACH build does two jobs at once — it serializes the
        // shared scratch, and it makes the last build's writes visible to the
        // property write that follows the loop.
        hg.run(|d, cmd| unsafe {
            d.cmd_reset_query_pool(cmd, pool, 0, n as u32);
            for i in 0..n {
                let info = info_for(i)
                    .dst_acceleration_structure(built[i])
                    .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_addr });
                let range = [vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(prims[i])];
                accel.cmd_build_acceleration_structures(cmd, &[info], &[&range]);
                as_barrier(d, cmd);
            }
            accel.cmd_write_acceleration_structures_properties(
                cmd,
                &built,
                vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                pool,
                0,
            );
        })?;

        let mut csizes = vec![0u64; n];
        unsafe {
            d.get_query_pool_results(
                pool,
                0,
                &mut csizes,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(|e| format!("vkGetQueryPoolResults(compacted size): {e}"))?;

        // Compacted layout. A degenerate reported size (0, or no smaller than
        // the build) keeps that chunk uncompacted — never wrong, just bigger.
        let mut final_off = Vec::with_capacity(n);
        let mut total_final = 0u64;
        let mut compact = Vec::with_capacity(n);
        for i in 0..n {
            let c = csizes[i] > 0 && csizes[i] < build_size[i];
            compact.push(c);
            final_off.push(total_final);
            total_final = align_as(total_final + if c { csizes[i] } else { build_size[i] });
        }
        let a = vkd.buffer(total_final.max(4), store_usage, false)?;
        for i in 0..n {
            let h = create_as(
                accel,
                &a,
                final_off[i],
                if compact[i] { csizes[i] } else { build_size[i] },
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            );
            match h {
                Ok(h) => final_as.push(h),
                Err(e) => {
                    arena = Some(a);
                    return Err(e);
                }
            }
        }
        arena = Some(a);

        // Submit 2: compact (or clone) every chunk into the exact-size arena.
        hg.run(|d, cmd| unsafe {
            for i in 0..n {
                accel.cmd_copy_acceleration_structure(
                    cmd,
                    &vk::CopyAccelerationStructureInfoKHR::default()
                        .src(built[i])
                        .dst(final_as[i])
                        .mode(if compact[i] {
                            vk::CopyAccelerationStructureModeKHR::COMPACT
                        } else {
                            vk::CopyAccelerationStructureModeKHR::CLONE
                        }),
                );
            }
            as_barrier(d, cmd);
        })?;

        // SCRATCH IS REPORTED SEPARATELY FROM THE REST OF THE TRANSIENT, and
        // that is the whole reason this feature exists rather than a
        // presentation choice: the build arena scales with the SCENE either
        // way, while scratch is sized by the LARGEST SINGLE GEOMETRY and so is
        // the number the split actually moves (measured san-miguel-low-poly:
        // 665 MB as one BLAS, 7 MB as 198 chunks, for the same compacted
        // result). A line that folded them together would hide it.
        Ok(format!(
            "blas {} MB (compacted from {}, both live across the copy) | scratch {} MB (freed)",
            total_final >> 20,
            total_build >> 20,
            scratch_max >> 20,
        ))
    })();

    // The build arena and its structures existed only to be compacted out of.
    for h in built {
        unsafe { accel.destroy_acceleration_structure(h, None) };
    }
    if pool != vk::QueryPool::null() {
        unsafe { d.destroy_query_pool(pool, None) };
    }
    vkd.free_buffer(&scratch);
    vkd.free_buffer(&build_arena);
    match r {
        Ok(report) => Ok((final_as, arena.expect("the arena exists on success"), report)),
        Err(e) => {
            for h in final_as {
                unsafe { accel.destroy_acceleration_structure(h, None) };
            }
            if let Some(a) = arena {
                vkd.free_buffer(&a);
            }
            Err(e)
        }
    }
}

/// One acceleration structure as a VIEW of `buf` at `offset`.
fn create_as(
    accel: &ash::khr::acceleration_structure::Device,
    buf: &Buffer,
    offset: u64,
    size: u64,
    ty: vk::AccelerationStructureTypeKHR,
) -> Result<vk::AccelerationStructureKHR, String> {
    unsafe {
        accel.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(buf.buf)
                .offset(offset)
                .size(size)
                .ty(ty),
            None,
        )
    }
    .map_err(|e| format!("vkCreateAccelerationStructureKHR(offset {offset}, {size} B): {e}"))
}

/// The build WRITES a structure; the next build's scratch, the compacted-size
/// query, the compaction copy, the TLAS build and every RayQuery READ one.
/// Vulkan orders those only through this edge — there is no implicit
/// acceleration-structure visibility the way a D3D12 UAV barrier gives.
fn as_barrier(d: &ash::Device, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_access_mask(
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
                | vk::AccessFlags::SHADER_READ,
        );
    unsafe {
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[mb],
            &[],
            &[],
        )
    };
}

/// `FR_SPLIT_AUDIT=1` — the D3D12 diagnostic, ported: read a streamed u32
/// buffer straight back and memcmp it against the CPU plan, so "the GPU sees
/// wrong remap DATA" can be ruled in or out without touching a shader. Loud
/// either way. It exists on both backends because the question it answers is
/// about the STREAM, and the two backends stream differently.
fn audit(
    hg: &VkHeadless,
    stage: &Stage,
    buf: &Buffer,
    expect: &[u32],
    name: &str,
) -> Result<(), String> {
    let vkd = &hg.vk;
    let bytes = std::mem::size_of_val(expect) as u64;
    // Reuses the staging ring's own size as the readback chunk: it is already
    // sized to this scene, and a second cap would be a second thing to keep in
    // step with `FR_VK_STAGE`.
    let chunk = stage.size().min(bytes.max(4));
    let rb = vkd.buffer(chunk, vk::BufferUsageFlags::TRANSFER_DST, true)?;
    let mut bad = 0usize;
    let mut off = 0u64;
    while off < bytes {
        let len = chunk.min(bytes - off);
        let r = hg.run(|d, cmd| unsafe {
            let region = vk::BufferCopy::default().src_offset(off).size(len);
            d.cmd_copy_buffer(cmd, buf.buf, rb.buf, &[region]);
        });
        if let Err(e) = r {
            vkd.free_buffer(&rb);
            return Err(e);
        }
        let ptr = match unsafe {
            vkd.device.map_memory(rb.mem, 0, len, vk::MemoryMapFlags::empty())
        } {
            Ok(p) => p as *const u32,
            Err(e) => {
                vkd.free_buffer(&rb);
                return Err(format!("audit map: {e}"));
            }
        };
        let first = (off / 4) as usize;
        for j in 0..(len / 4) as usize {
            if unsafe { ptr.add(j).read() } != expect[first + j] {
                bad += 1;
            }
        }
        unsafe { vkd.device.unmap_memory(rb.mem) };
        off += len;
    }
    vkd.free_buffer(&rb);
    if bad > 0 {
        return Err(format!(
            "FR_SPLIT_AUDIT: {name} differs from the CPU plan in {bad} of {} words",
            expect.len()
        ));
    }
    eprintln!("blas-split: FR_SPLIT_AUDIT {name} OK — {} words match", expect.len());
    Ok(())
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
