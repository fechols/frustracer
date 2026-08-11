//! The upload ring: one reusable host-visible chunk that every immutable
//! device-local stream is written through.
//!
//! `SceneGpu::new_uploaded`'s `STAGE_CHUNK` ring, in the other API — and the
//! last of `vk/scene.rs`'s three declared M3b boundaries. Until this landed,
//! every stream on this backend was `HOST_VISIBLE` and written through a map,
//! which is correct everywhere and costs two things that only get worse with
//! scene size:
//!
//! - **Peak host memory.** A mapped upload is a second full copy of the
//!   stream, and the repack that feeds it (`Vec3A` -> `float3`, `BvhNode` ->
//!   `GpuBvhNode`, `FNode` -> `QFNode`) was a THIRD. At 34.4M triangles the
//!   frustum tree alone is ~480 MB and the binary tree ~960 MB, so the
//!   materialized-`Vec`-plus-mapped-buffer shape cost ~2.9 GB of host RAM to
//!   move ~1.4 GB. Streaming through a bounded ring makes the transient one
//!   chunk, and the generator form below makes the repack disappear entirely.
//! - **Read bandwidth, forever.** A host-visible buffer a shader reads is
//!   host memory a shader reads. Invisible on the UMA part this was developed
//!   on and a per-access PCIe round trip on a discrete GPU — which is the
//!   half that cannot be measured here at all, and so the half worth getting
//!   structurally right rather than empirically.
//!
//! WHAT STAYS MAPPED, deliberately: `frame_cb` (rewritten every frame),
//! `push` (rewritten twice per ladder level), and the hemi probe points
//! (regenerated per probe run). Those are the dynamic class — D3D12 keeps its
//! own equivalents in an upload heap for the same reason — and staging them
//! would trade a host write for a host write plus a copy plus a barrier.
//!
//! THE ONE THING TO GET RIGHT is that the ring must hold at least one whole
//! element: the chunk size is `(ring / size_of::<U>()).max(1)`, and that
//! `.max(1)` is a silent buffer overrun the moment an element is bigger than
//! the ring. `Stage::new` takes the largest indivisible piece as a floor and
//! `stream_gen` refuses rather than trusting it.

use ash::vk;

use crate::vk::device::{Buffer, Vk};
use crate::vk::headless::VkHeadless;

/// Ring cap. Smaller than D3D12's 256 MB on purpose: every chunk here is its
/// own submit-and-fence-wait (`VkHeadless::run` is blocking by construction),
/// so the ring buys fewer waits rather than deeper pipelining, and 64 MB is
/// already ~1 wait per 64 MB — negligible against a load that decodes
/// textures. It is also the size the texture path has been batching at since
/// M3d, measured on a 511 MB scene, so this is one number where there was
/// already a working value to adopt.
const RING_CAP: u64 = 64 << 20;

/// `FR_VK_STAGE=<bytes>` (a `k` or `m` suffix scales) — the ring cap, levered.
///
/// It exists because the multi-chunk path is otherwise reachable only on a
/// scene big enough to need it, which makes the gate's coverage a property of
/// which scene somebody happened to run: at 64 MB the procedural check scene
/// is one chunk per stream and every off-by-one in the loop is invisible,
/// while san-miguel's index stream is 67 MB and chunks by accident. A small
/// ring turns every stream on any scene into a chunked one and the results
/// must not move — the `FR_LEAF` discipline, where a lever's job is to let a
/// gate recreate a configuration it could not otherwise reach.
///
/// BYTES rather than the MiB the first draft took, because the unit has to be
/// finer than the smallest thing worth chunking: the procedural scene's
/// largest stream is under 1 MB, so a 1 MiB floor "armed" the lever and
/// chunked nothing (measured: 13 submits before and after) — an instrument at
/// the wrong RESOLUTION cannot see what it was built for.
///
/// LOUD on departure and loud-plus-default on an illegal value, because a
/// mistyped sweep cell that silently measures the default is the failure mode
/// this lever family exists to prevent. 0 is illegal rather than "unbounded":
/// `Stage::new` floors at 4096 anyway, so it would read as a working request
/// that did nothing.
fn ring_cap() -> u64 {
    let raw = match std::env::var("FR_VK_STAGE") {
        Err(_) => return RING_CAP,
        Ok(v) => v,
    };
    let t = raw.trim();
    let (num, scale) = match t.as_bytes().last() {
        Some(b'k') | Some(b'K') => (&t[..t.len() - 1], 1024),
        Some(b'm') | Some(b'M') => (&t[..t.len() - 1], 1024 * 1024),
        _ => (t, 1),
    };
    match num.parse::<u64>() {
        Ok(n) if n > 0 => {
            let b = n.saturating_mul(scale);
            println!("FR_VK_STAGE={raw}: {b} B staging ring (default {RING_CAP} B)");
            b
        }
        _ => {
            println!("FR_VK_STAGE={raw}: not a positive size — using the default {RING_CAP} B");
            RING_CAP
        }
    }
}

/// A reusable host-visible staging buffer, persistently mapped.
///
/// Persistent mapping rather than map/unmap per chunk: the memory is
/// `HOST_COHERENT` (see `Vk::buffer`), so there is no flush either way, and
/// mapping is the one part of an upload that has no reason to happen more
/// than once. The raw pointer is why this type is neither `Send` nor `Sync`,
/// which is correct — scene load is single-threaded here.
pub struct Stage {
    buf: Buffer,
    ptr: *mut u8,
    /// Bytes actually staged, and submits spent doing it. Reported rather than
    /// asserted, for the reason the texture line reports bytes: a run that
    /// staged nothing and a run that staged everything have the same buffer
    /// count, and only a byte total tells them apart.
    bytes: u64,
    chunks: u32,
}

impl Stage {
    /// `total` = everything this ring is expected to carry (so a small scene
    /// does not commit the cap), `atom` = the largest single piece that must
    /// fit WHOLE — a texture's entire mip chain for the texture path, one
    /// element for a stream. Both are floors on correctness, not hints:
    /// undersizing `atom` is the overrun the module header names.
    pub fn new(vkd: &Vk, total: u64, atom: u64) -> Result<Stage, String> {
        // `atom` survives the cap on purpose — it is a correctness floor, and
        // an `FR_VK_STAGE=1` ring that could not hold one texture chain would
        // fail the run rather than exercise the chunking it was set to
        // exercise. So the lever shrinks the ring exactly as far as the
        // indivisible pieces allow, and no further.
        let size = ring_cap().min(total.max(4096)).max(atom).max(4096);
        // `STORAGE_BUFFER` because the GPU BC7 encoder READS THE RING IN
        // PLACE — `vk::bc7` binds it at `t0` and moves a byte window in its
        // constants, which is what D3D12's arm does too (its `src` is the
        // upload heap read straight as a root SRV). Unconditional rather than
        // asked of the texture path alone, for the `TRANSFER_SRC` reason
        // below: a bit that only some callers set makes a consumer's
        // availability a property of who built the ring, and the bit is free.
        let buf = vkd.buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            true,
        )?;
        let ptr = match unsafe {
            vkd.device.map_memory(buf.mem, 0, buf.size, vk::MemoryMapFlags::empty())
        } {
            Ok(p) => p as *mut u8,
            Err(e) => {
                vkd.free_buffer(&buf);
                return Err(format!("vkMapMemory(stage, {size} B): {e}"));
            }
        };
        Ok(Stage { buf, ptr, bytes: 0, chunks: 0 })
    }

    pub fn size(&self) -> u64 {
        self.buf.size
    }

    /// The ring itself, for a caller that records its own copy command — the
    /// texture path, whose destination is an image and whose regions are its
    /// own business.
    pub fn buf(&self) -> &Buffer {
        &self.buf
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn chunks(&self) -> u32 {
        self.chunks
    }

    /// Copy `bytes` into the ring, ready for a caller-recorded transfer. Does
    /// not submit; the caller's own `hg.run` follows it one-for-one, which is
    /// why this still counts a chunk.
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() as u64 > self.buf.size {
            return Err(format!(
                "stage write {} B into a {} B ring",
                bytes.len(),
                self.buf.size
            ));
        }
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr, bytes.len()) };
        self.bytes += bytes.len() as u64;
        self.chunks += 1;
        Ok(())
    }

    /// A DEVICE_LOCAL buffer of `n` elements, filled from `make` through the
    /// ring — `gpu::trace::stream_buffer`'s twin.
    ///
    /// GENERATOR rather than slice because the two streams that most need
    /// this have no source slice to point at: the `--blas-split` identity
    /// remap is 138 MB of `0..n` at 34.4M triangles, and the frustum tree's
    /// wire nodes are computed. `stream` below is the slice form, expressed
    /// over this one so there is a single chunking loop.
    ///
    /// `TRANSFER_DST` is added here rather than asked of the caller, for the
    /// reason `Vk::buffer` derives its allocate flag from the usage: a caller
    /// who lists four usage bits and forgets the fifth gets a clean
    /// `vkCreateBuffer` and a validation error at copy time.
    ///
    /// `TRANSFER_SRC` rides along for a different reason — every staged stream
    /// is READABLE BACK. That is what an audit against the CPU-side source
    /// needs (`FR_SPLIT_AUDIT`), and it is deliberately unconditional rather
    /// than armed with the lever: a diagnostic that only runs against a
    /// specially-built buffer set is a diagnostic whose subject is not the
    /// shipping configuration, so it has to be trusted instead of run. The bit
    /// is free — `TRANSFER_SRC` on a buffer is universally supported and does
    /// not move the memory requirements.
    pub fn stream_gen<U: Copy>(
        &mut self,
        hg: &VkHeadless,
        n: usize,
        make: impl Fn(usize) -> U,
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer, String> {
        let vkd = &hg.vk;
        let esz = std::mem::size_of::<U>();
        let dst = vkd.buffer(
            (esz * n).max(4) as u64,
            usage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            false,
        )?;
        if n == 0 {
            // A zero-length stream still gets a real 4-byte buffer (zero-size
            // is illegal), and it gets ZEROED rather than left as whatever the
            // heap held: nothing should read it, but "nothing should" and
            // "reads a defined value if something does" are different
            // guarantees, and only one of them survives a future bug.
            return self.finish(hg, dst, |d, cmd, b| unsafe {
                d.cmd_fill_buffer(cmd, b.buf, 0, vk::WHOLE_SIZE, 0);
            });
        }
        if esz as u64 > self.buf.size {
            vkd.free_buffer(&dst);
            return Err(format!(
                "a {esz} B element does not fit a {} B staging ring",
                self.buf.size
            ));
        }
        // `vkMapMemory` returns a pointer aligned to `minMemoryMapAlignment`,
        // which the spec floors at 64 — every wire struct here has alignment
        // 16 or less, so the cast below is aligned by construction.
        //
        // The chunking goes through `plan` rather than repeating its four
        // lines, which is what makes `self_test` a gate on THIS loop instead
        // of on a parallel description of it.
        let chunks = plan(n, esz, self.buf.size as usize);
        // `i + 1 == n_chunks` rather than a precomputed `len() - 1`: `n > 0`
        // does guarantee a chunk here, but a subtraction that underflows on an
        // empty plan is a panic reachable only by editing the plan, and the
        // total form costs nothing.
        let n_chunks = chunks.len();
        for (i, (first, cnt, off)) in chunks.into_iter().enumerate() {
            unsafe {
                let out = std::slice::from_raw_parts_mut(self.ptr as *mut U, cnt);
                for (j, o) in out.iter_mut().enumerate() {
                    *o = make(first + j);
                }
            }
            let len = (cnt * esz) as u64;
            let r = hg.run(|d, cmd| unsafe {
                let region = vk::BufferCopy::default().dst_offset(off as u64).size(len);
                d.cmd_copy_buffer(cmd, self.buf.buf, dst.buf, &[region]);
                if i + 1 == n_chunks {
                    barrier(d, cmd);
                }
            });
            if let Err(err) = r {
                vkd.free_buffer(&dst);
                return Err(err);
            }
            self.bytes += len;
            self.chunks += 1;
        }
        Ok(dst)
    }

    /// The slice form: `src` mapped element-wise into the wire type.
    pub fn stream<T, U: Copy>(
        &mut self,
        hg: &VkHeadless,
        src: &[T],
        map: impl Fn(&T) -> U,
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer, String> {
        self.stream_gen(hg, src.len(), |i| map(&src[i]), usage)
    }

    /// A DEVICE_LOCAL buffer of `size` bytes, zero-filled. For the slots that
    /// are declared and never read — a bound dummy turns "the shader touched a
    /// binding nobody expected" from undefined content into a read of zeros,
    /// which is only true if somebody writes the zeros.
    pub fn zeros(
        &mut self,
        hg: &VkHeadless,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer, String> {
        let dst =
            hg.vk.buffer(
                size.max(4),
                usage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
                false,
            )?;
        self.finish(hg, dst, |d, cmd, b| unsafe {
            d.cmd_fill_buffer(cmd, b.buf, 0, vk::WHOLE_SIZE, 0);
        })
    }

    /// Run one command that writes `dst`, then the visibility barrier, and
    /// hand `dst` back — the tail every path above shares.
    fn finish(
        &mut self,
        hg: &VkHeadless,
        dst: Buffer,
        f: impl FnOnce(&ash::Device, vk::CommandBuffer, &Buffer),
    ) -> Result<Buffer, String> {
        let r = hg.run(|d, cmd| {
            f(d, cmd, &dst);
            barrier(d, cmd);
        });
        self.chunks += 1;
        match r {
            Ok(()) => Ok(dst),
            Err(e) => {
                hg.vk.free_buffer(&dst);
                Err(e)
            }
        }
    }

    pub fn free(self, vkd: &Vk) {
        unsafe { vkd.device.unmap_memory(self.buf.mem) };
        vkd.free_buffer(&self.buf);
    }
}

/// Make every transfer write issued so far visible to shader reads and to the
/// acceleration-structure build.
///
/// EMITTED ON THE LAST CHUNK ONLY, and that is a property of Vulkan's
/// synchronization scopes rather than an optimization: a pipeline barrier's
/// first scope is "all commands that occur earlier in SUBMISSION ORDER", which
/// spans earlier submits on the same queue — so one barrier after the final
/// write covers every chunk before it, in whatever command buffer it landed.
/// The chunks themselves need no ordering against each other: they write
/// disjoint ranges and nothing reads the buffer in between.
///
/// The destination mask covers `ACCELERATION_STRUCTURE_READ` as well as
/// `SHADER_READ` because `positions` and `indices` are read by the BLAS build
/// through a device address, not by a shader. A shader-only mask would be
/// right for nine of the eleven streams and silently wrong for the two whose
/// corruption is hardest to attribute.
fn barrier(d: &ash::Device, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        );
    unsafe {
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            &[mb],
            &[],
            &[],
        )
    };
}

// ---------------------------------------------------------------------------

/// The chunking arithmetic, gated without a device.
///
/// The loop above is four lines and every one of them is an off-by-one
/// waiting to happen — a dropped last chunk, an offset in elements instead of
/// bytes, a ring rounded up instead of down. None of that is visible in an
/// image gate unless a committed scene is big enough to chunk at all, and at
/// a 64 MB ring the procedural scene never is. So the plan is stated as a
/// pure function and checked here, and the device half is left to prove that
/// a copy lands where the plan says.
pub fn plan(n: usize, esz: usize, ring: usize) -> Vec<(usize, usize, usize)> {
    let per = (ring / esz.max(1)).max(1);
    let mut out = Vec::new();
    let mut e = 0usize;
    while e < n {
        let cnt = per.min(n - e);
        out.push((e, cnt, e * esz));
        e += cnt;
    }
    out
}

pub fn self_test() -> Result<(), String> {
    // A stream that fits in one chunk is one chunk, at offset 0.
    let one = plan(100, 12, 4096);
    if one != vec![(0, 100, 0)] {
        return Err(format!("a fitting stream did not plan as one chunk: {one:?}"));
    }
    // ...and a stream that does not is exactly-covered, in byte offsets.
    let many = plan(1000, 12, 1200);
    let total: usize = many.iter().map(|c| c.1).sum();
    if total != 1000 {
        return Err(format!("plan covered {total} of 1000 elements"));
    }
    let mut next = 0usize;
    for (first, cnt, off) in &many {
        if *first != next {
            return Err(format!("plan chunk starts at {first}, expected {next}"));
        }
        if *off != first * 12 {
            return Err(format!("chunk offset {off} is not {first} * 12 — elements, not bytes"));
        }
        if cnt * 12 > 1200 {
            return Err(format!("chunk of {cnt} x 12 B overruns a 1200 B ring"));
        }
        next += cnt;
    }
    // 1200 / 12 = 100 per chunk, so the LAST chunk is a full one — the shape
    // where a `< n` loop condition and a `<= n` one still agree. The teeth are
    // the ragged case beside it.
    if many.len() != 10 {
        return Err(format!("expected 10 chunks of 100, got {}", many.len()));
    }
    let ragged = plan(1005, 12, 1200);
    if ragged.len() != 11 || ragged[10] != (1000, 5, 12000) {
        return Err(format!("ragged tail planned as {:?}", ragged.last()));
    }

    // The empty stream plans NO copy at all — `stream_gen` takes its own
    // zero-fill path there, and a phantom chunk would copy from an unwritten
    // ring.
    if !plan(0, 12, 4096).is_empty() {
        return Err("an empty stream planned a chunk".into());
    }

    // THE OVERRUN CASE. `.max(1)` keeps the loop making progress when an
    // element is larger than the ring — which is the right answer for
    // termination and the WRONG one for memory, so `stream_gen` refuses that
    // configuration outright rather than relying on this. Pinned so the
    // refusal cannot be deleted as redundant: without it, this plan is a
    // 256 B write into a 100 B mapping.
    let over = plan(3, 256, 100);
    if over.len() != 3 || over.iter().any(|c| c.1 != 1) {
        return Err(format!("oversized elements did not plan one at a time: {over:?}"));
    }
    if over[1].2 != 256 {
        return Err("oversized-element offsets are not byte offsets".into());
    }

    // A ring that is not a multiple of the element size rounds DOWN — the
    // whole point of the division. Rounding up is the overrun again, one
    // element at a time.
    let odd = plan(10, 12, 40);
    if odd[0].1 != 3 {
        return Err(format!("a 40 B ring took {} x 12 B elements", odd[0].1));
    }
    Ok(())
}
