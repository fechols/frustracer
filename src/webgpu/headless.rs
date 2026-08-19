//! Headless one-shot execution and the smoke test — `--check-wgpu` J2/J3.
//!
//! What the smoke proves is the same list `gpu::smoke_test` proves on
//! D3D12, `vk::headless::smoke_test` on Vulkan and `mtl::smoke` on Metal,
//! because it is the same shader (`gfx::shaders::SMOKE_HLSL`): constants
//! reaching a kernel, storage buffers bound and written, a GPU-WRITTEN
//! counter turned into dispatch arguments by a second kernel, and a third
//! kernel launched from those arguments without the CPU ever seeing the
//! count. That last one is the whole wavefront design; if indirect dispatch
//! does not work, nothing above it can.
//!
//! Two things are different here, and both are the point:
//!
//! - The kernel arrives through the BROWSER's shader chain — DXC ->
//!   SPIR-V (vulkan1.1) -> `wgsl::normalize` -> `wgsl::spv_to_wgsl` ->
//!   `create_shader_module(ShaderSource::Wgsl)`. This file is
//!   `spv_to_wgsl`'s first live consumer: the gate executes exactly the
//!   text a `--bake-web` page will hand the browser's compiler.
//! - There are NO barriers. WebGPU's per-dispatch usage scopes are the
//!   synchronization — the three explicit vkCmdPipelineBarriers of the
//!   Vulkan twin (transfer->compute, compute->compute, compute->indirect)
//!   are the implementation's job here, and that this works is part of what
//!   J3 proves.
//!
//! And one thing this gate MEASURED on its first run (2026-08-19), against
//! the design's expectation: a dispatch's usage scope is computed from the
//! pipeline LAYOUT, not the shader's static use. The Vulkan twin's shape —
//! one descriptor-set layout carrying all four bindings, shared by all
//! three pipelines — is ILLEGAL here: the shared layout carries `args` as
//! read-write storage, and STORAGE_READ_WRITE is exclusive, so the fill
//! dispatch's scope (layout bindings + the INDIRECT buffer) conflicts with
//! itself. Layouts are therefore PER PIPELINE, each carrying only the
//! bindings its entry statically uses — the same conclusion Stage C2's
//! tracer design reached on paper (no PARTIALLY_BOUND in WebGPU), now
//! confirmed by a validation error rather than an assumption.

use super::device::Wgpu;
use crate::spirv::{Reg, Spirv, binding_of};

/// A device wrapped for one-shot gate work. wgpu needs no pool, command
/// buffer or fence — the encoder is per-run and the queue owns pacing.
pub struct WgpuHeadless {
    pub dev: Wgpu,
}

impl WgpuHeadless {
    pub fn new() -> Result<WgpuHeadless, super::device::WgpuError> {
        Ok(WgpuHeadless { dev: Wgpu::new()? })
    }

    /// Record with `f`, submit, and block until the GPU is done. The poll is
    /// for ERROR LOCALITY, not correctness — same-queue submissions execute
    /// in submission order, so a later readback would see these writes
    /// without it; blocking here just makes a device loss or an uncaptured
    /// error attribute to the stage that caused it (the VkHeadless::run
    /// contract). Only `map_async` truly NEEDS a poll.
    pub fn run<F: FnOnce(&mut wgpu::CommandEncoder)>(&self, f: F) -> Result<(), String> {
        let mut enc = self
            .dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("headless") });
        f(&mut enc);
        self.dev.queue.submit([enc.finish()]);
        self.dev
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device.poll: {e}"))?;
        Ok(())
    }

    /// Device-local -> host, via a MAP_READ staging buffer and its own
    /// submit — the vk read_buffer shape; RAII frees the staging buffer.
    pub fn read_buffer(&self, src: &wgpu::Buffer, len: usize) -> Result<Vec<u8>, String> {
        // An Err, not an assert: a gate must die with a FAIL line, never a
        // panic (every current caller passes static, aligned sizes — this
        // guards the future caller who does not).
        if len as u64 % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(format!(
                "read_buffer: len {len} is not a multiple of {}",
                wgpu::COPY_BUFFER_ALIGNMENT
            ));
        }
        let staging = self.dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: len as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.run(|enc| enc.copy_buffer_to_buffer(src, 0, &staging, 0, len as u64))?;

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // THE poll that is actually required: map_async only completes
        // inside poll/submit, and wait_indefinitely guarantees the callback
        // has fired before recv.
        self.dev
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device.poll (map): {e}"))?;
        rx.recv()
            .map_err(|_| "map_async callback never fired".to_string())?
            .map_err(|e| format!("map_async: {e}"))?;
        let out = {
            let view = slice.get_mapped_range().map_err(|e| format!("get_mapped_range: {e}"))?;
            view.to_vec()
        };
        staging.unmap();
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// The smoke test. Constants and verdicts are the Vulkan twin's, verbatim.

const FILL_N: u32 = 555; // deliberately not a multiple of 64
const TAIL: u32 = 64; // guard words past the fill, checked untouched
const SENTINEL: u32 = 0xEEEE_EEEE;

/// Bindings, derived from the ONE rule (`spirv::binding_of`). `smoke.hlsl`
/// declares `b1` and `u0..u2` in the default space, so these are group 0 at
/// the shifted bindings — computed rather than written down, because a
/// literal is exactly how a layout drifts away from the `-fvk-*-shift`
/// flags the module was compiled with. That `u0` lands at 2000 is not
/// incidental: it is what drags the layout past WebGPU's default
/// binding-number ceiling and makes J3 exercise the raised limit J1 asked
/// for.
fn smoke_bindings() -> [(u32, wgpu::BufferBindingType); 4] {
    use wgpu::BufferBindingType as B;
    [
        (binding_of(Reg::B, 1), B::Uniform),
        (binding_of(Reg::U, 0), B::Storage { read_only: false }),
        (binding_of(Reg::U, 1), B::Storage { read_only: false }),
        (binding_of(Reg::U, 2), B::Storage { read_only: false }),
    ]
}

/// Which of the four `smoke_bindings()` slots each entry statically uses —
/// the per-pipeline layouts are cut from this ONE table, and so are the
/// per-pipeline bind groups, so the two cannot disagree. Spelled by index
/// into `smoke_bindings()` rather than by register so the shift rule stays
/// the single binding-number authority.
const SMOKE_USES: [(&str, &[usize]); 3] = [
    ("cs_seed", &[0, 1]), // push, counters
    ("cs_prep", &[1, 2]), // counters, args
    ("cs_fill", &[1, 3]), // counters, outbuf — NOT args (it arrives as INDIRECT)
];

pub struct SmokePipes {
    /// One (layout, pipeline) pair per entry — the module header says why
    /// a shared layout is illegal here.
    bgls: Vec<wgpu::BindGroupLayout>,
    pipes: Vec<wgpu::ComputePipeline>,
    /// Total WGSL bytes emitted — J2's proof the translator ran.
    pub wgsl_bytes: usize,
}

/// The J2 seam: HLSL -> DXC -> normalize -> WGSL -> modules -> pipelines,
/// one layout per pipeline (the module header's measured rule).
pub fn build_pipes(dev: &Wgpu, sp: &Spirv) -> Result<SmokePipes, String> {
    let binds = smoke_bindings();
    let mut bgls = Vec::new();
    let mut pipes = Vec::new();
    let mut wgsl_bytes = 0usize;
    for (e, uses) in SMOKE_USES {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = uses
            .iter()
            .map(|&u| {
                let (b, ty) = binds[u];
                wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            })
            .collect();
        let (bgl, layout) = dev.scoped(&format!("smoke {e}: layout"), || {
            let bgl = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(e),
                entries: &entries,
            });
            let layout = dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(e),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            (bgl, layout)
        })?;
        bgls.push(bgl);
        // The browser's exact diet: vulkan1.1 (the --check-wgsl W2 arm —
        // SPIR-V >= 1.4 emits OpCopyLogical, which naga refuses), then the
        // same normalize the shipped blobs get, then the full
        // parse/validate/emit/re-parse chain.
        let words = sp.compile_args(
            crate::gfx::shaders::SMOKE_HLSL,
            e,
            "cs_6_5",
            &format!("smoke {e}"),
            false,
            &["-fspv-target-env=vulkan1.1"],
        )?;
        let words = crate::wgsl::normalize(&words);
        let text = crate::wgsl::spv_to_wgsl(&words).map_err(|err| format!("smoke {e}: {err}"))?;
        wgsl_bytes += text.len();
        let module = dev.scoped(&format!("smoke {e}: create_shader_module"), || {
            dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(e),
                source: wgpu::ShaderSource::Wgsl(text.into()),
            })
        })?;
        let pipe = dev.scoped(&format!("smoke {e}: create_compute_pipeline"), || {
            dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&layout),
                module: &module,
                // The sole @compute entry in a per-entry module — and
                // immune to naga renaming the entry on the way through.
                entry_point: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        })?;
        pipes.push(pipe);
    }
    Ok(SmokePipes { bgls, pipes, wgsl_bytes })
}

/// One pass of the seed -> prep -> indirect-fill chain at a chosen count.
/// Returns `(outbuf words, indirect args, counters)`.
#[allow(clippy::type_complexity)]
fn smoke_pass(
    hg: &WgpuHeadless,
    pipes: &SmokePipes,
    n: u32,
) -> Result<(Vec<u32>, [u32; 3], [u32; 2]), String> {
    let d = &hg.dev;
    let words = (FILL_N + TAIL) as u64;

    let buf = |label: &str, size: u64, usage: wgpu::BufferUsages| {
        d.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    use wgpu::BufferUsages as U;
    let push = buf("smoke push", 16, U::UNIFORM | U::COPY_DST);
    // WebGPU zero-initializes buffers by spec — counters needs no seed.
    let counters = buf("smoke counters", 8, U::STORAGE | U::COPY_SRC);
    let args = buf("smoke args", crate::gfx::shaders::ARG_STRIDE, U::STORAGE | U::INDIRECT | U::COPY_SRC);
    let outbuf = buf("smoke out", words * 4, U::STORAGE | U::COPY_SRC | U::COPY_DST);

    let push_bytes: Vec<u8> = [n, 0u32, 0, 0].iter().flat_map(|w| w.to_le_bytes()).collect();
    d.queue.write_buffer(&push, 0, &push_bytes);
    // Poison the output, so a never-dispatched fill reads back as the
    // sentinel and fails loudly at word 0 instead of reading back as zeros
    // and looking like a shader that ran and wrote nothing. write_buffer,
    // not clear_buffer (which can only write zeros — the opposite of a
    // poison); it is ordered before the submit, so it is also the fill
    // barrier the Vulkan twin spells out.
    let poison: Vec<u8> =
        std::iter::repeat_n(SENTINEL, words as usize).flat_map(|w| w.to_le_bytes()).collect();
    d.queue.write_buffer(&outbuf, 0, &poison);

    // One bind group per pipeline, cut from the same SMOKE_USES table as
    // the layouts — so cs_fill's group carries no `args`, and the indirect
    // read is the only use of that buffer in the fill dispatch's scope.
    let binds = smoke_bindings();
    let resources = [&push, &counters, &args, &outbuf];
    let mut bgs = Vec::new();
    for (i, (e, uses)) in SMOKE_USES.iter().enumerate() {
        let entries: Vec<wgpu::BindGroupEntry> = uses
            .iter()
            .map(|&u| wgpu::BindGroupEntry {
                binding: binds[u].0,
                resource: resources[u].as_entire_binding(),
            })
            .collect();
        bgs.push(d.scoped(&format!("smoke {e}: bind group"), || {
            d.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(e),
                layout: &pipes.bgls[i],
                entries: &entries,
            })
        })?);
    }

    hg.run(|enc| {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        p.set_pipeline(&pipes.pipes[0]);
        p.set_bind_group(0, &bgs[0], &[]);
        p.dispatch_workgroups(1, 1, 1); // seed: counters[0] = n
        p.set_pipeline(&pipes.pipes[1]);
        p.set_bind_group(0, &bgs[1], &[]);
        p.dispatch_workgroups(1, 1, 1); // prep: counters -> args
        p.set_pipeline(&pipes.pipes[2]);
        p.set_bind_group(0, &bgs[2], &[]);
        p.dispatch_workgroups_indirect(&args, 0); // the whole wavefront design
    })?;

    let le_words = |b: Vec<u8>| -> Vec<u32> {
        b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let out = le_words(hg.read_buffer(&outbuf, words as usize * 4)?);
    let a = le_words(hg.read_buffer(&args, crate::gfx::shaders::ARG_STRIDE as usize)?);
    let c = le_words(hg.read_buffer(&counters, 8)?);
    Ok((out, [a[0], a[1], a[2]], [c[0], c[1]]))
}

/// The J3 seam. Two passes — the real count and zero — with the Vulkan
/// twin's verdicts verbatim, then the uncaptured-error sweep.
pub fn smoke_run(hg: &WgpuHeadless, pipes: &SmokePipes) -> Result<(), String> {
    let (out, args, counters) = smoke_pass(hg, pipes, FILL_N)?;

    let want_args = [FILL_N.div_ceil(64), 1, 1];
    if args != want_args {
        return Err(format!("smoke: indirect args {args:?}, expected {want_args:?}"));
    }
    if counters != [FILL_N, FILL_N] {
        return Err(format!("smoke: counters {counters:?}, expected [{FILL_N}, {FILL_N}]"));
    }
    for i in 0..FILL_N {
        let want = i ^ 0x00C0_FFEE;
        if out[i as usize] != want {
            return Err(format!(
                "smoke: outbuf[{i}] = {:#x}, expected {want:#x}{}",
                out[i as usize],
                if out[i as usize] == SENTINEL { " (never written)" } else { "" }
            ));
        }
    }
    // The guard's own tooth: the last group covers words 512..576, so words
    // 555..576 were dispatched and MUST have been rejected by
    // `id.x < counters[1]`.
    for i in FILL_N..FILL_N + TAIL {
        if out[i as usize] != SENTINEL {
            return Err(format!(
                "smoke: outbuf[{i}] = {:#x} past the fill count — the bounds guard did not hold",
                out[i as usize]
            ));
        }
    }

    // Zero work: `prep` clamps y to 0, so the indirect dispatch launches
    // nothing — the empty-queue case every level of the real ladder hits,
    // and the one an emulated indirect path gets wrong.
    let (out0, args0, counters0) = smoke_pass(hg, pipes, 0)?;
    if args0 != [0, 0, 1] {
        return Err(format!("smoke(0): indirect args {args0:?}, expected [0, 0, 1]"));
    }
    if counters0 != [0, 0] {
        return Err(format!("smoke(0): counters {counters0:?}, expected [0, 0]"));
    }
    if let Some(i) = out0.iter().position(|&w| w != SENTINEL) {
        return Err(format!(
            "smoke(0): outbuf[{i}] = {:#x} — an empty indirect dispatch ran work",
            out0[i]
        ));
    }

    // The side-channel sweep (the vk validation_errors twin): anything the
    // creation scopes did not own — submit-time validation, device loss —
    // landed on the uncaptured counter instead of panicking the gate.
    let e = hg.dev.errors();
    if e > 0 {
        return Err(format!("smoke: {e} uncaptured wgpu error(s) — the run was not clean"));
    }
    Ok(())
}
