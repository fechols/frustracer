//! A headless compute harness — the Vulkan peer of `gpu::HeadlessGpu`: a
//! device, one command buffer, one fence, `run(|cmd| ...)`, and staging
//! copies. No surface, no swapchain, no window.
//!
//! `VK_EXT_headless_surface` is deliberately NOT used here, and the reason is
//! worth stating because the plan named it: that extension exists to run the
//! PRESENTATION path without a window (a swapchain with nowhere to scan out).
//! Compute needs no surface at all, so the harness every gate runs on has one
//! less moving part than the plan assumed. The extension becomes relevant
//! when the display stage wants gating, not before.
//!
//! What the smoke test at the bottom proves is the same list `gpu::smoke_test`
//! proves on D3D12, because it is the same shader: constants reaching a
//! kernel, storage buffers bound and written, a GPU-WRITTEN counter turned
//! into dispatch arguments by a second kernel, and a third kernel launched
//! from those arguments without the CPU ever seeing the count. That last one
//! is the whole wavefront design; if indirect dispatch does not work, nothing
//! above it can.

use ash::vk;
use std::ffi::CString;

use super::device::{Buffer, Vk, VkError};
use super::spirv::{self, Reg, Spirv};

pub struct VkHeadless {
    pub vk: Vk,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

impl VkHeadless {
    /// The error carries `absent` (see `device::VkError`): a box with no
    /// Vulkan is a SKIP, a lever that named nothing is a FAIL.
    pub fn new(validation: bool) -> Result<VkHeadless, VkError> {
        let vk = Vk::new(validation)?;
        let pci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(vk.qfam)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool = unsafe { vk.device.create_command_pool(&pci, None) }
            .map_err(|e| format!("vkCreateCommandPool: {e}"))?;
        let ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { vk.device.allocate_command_buffers(&ai) }
            .map_err(|e| format!("vkAllocateCommandBuffers: {e}"))?[0];
        let fence = unsafe { vk.device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| format!("vkCreateFence: {e}"))?;
        Ok(VkHeadless { vk, pool, cmd, fence })
    }

    /// Record, submit, and block until the GPU is done — the `HeadlessGpu::run`
    /// contract: no swapchain, no frames in flight, one fence.
    pub fn run<F: FnOnce(&ash::Device, vk::CommandBuffer)>(&self, f: F) -> Result<(), String> {
        let d = &self.vk.device;
        unsafe {
            d.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(|e| format!("vkResetCommandPool: {e}"))?;
            let bi = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            d.begin_command_buffer(self.cmd, &bi)
                .map_err(|e| format!("vkBeginCommandBuffer: {e}"))?;
            f(d, self.cmd);
            d.end_command_buffer(self.cmd).map_err(|e| format!("vkEndCommandBuffer: {e}"))?;

            d.reset_fences(&[self.fence]).map_err(|e| format!("vkResetFences: {e}"))?;
            let cmds = [self.cmd];
            let si = vk::SubmitInfo::default().command_buffers(&cmds);
            d.queue_submit(self.vk.queue, &[si], self.fence)
                .map_err(|e| format!("vkQueueSubmit: {e}"))?;
            d.wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("vkWaitForFences: {e}"))?;
        }
        Ok(())
    }

    /// Device-local -> host, through a staging buffer and its own submit.
    pub fn read_buffer(&self, src: &Buffer, len: usize) -> Result<Vec<u8>, String> {
        let len = len.min(src.size as usize);
        let stage = self.vk.buffer(len as u64, vk::BufferUsageFlags::TRANSFER_DST, true)?;
        let r = self.run(|d, cmd| unsafe {
            let region = vk::BufferCopy::default().size(len as u64);
            d.cmd_copy_buffer(cmd, src.buf, stage.buf, &[region]);
        });
        let out = r.and_then(|_| self.vk.read(&stage, len));
        self.vk.free_buffer(&stage);
        out
    }
}

impl Drop for VkHeadless {
    fn drop(&mut self) {
        unsafe {
            let _ = self.vk.device.device_wait_idle();
            self.vk.device.destroy_fence(self.fence, None);
            self.vk.device.destroy_command_pool(self.pool, None);
        }
    }
}

/// A whole-pipeline memory barrier between two compute dispatches. Coarse on
/// purpose: this harness runs gates, where a missing barrier is a wrong answer
/// and a redundant one is nothing. The real tracer narrows them.
fn compute_barrier(d: &ash::Device, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    unsafe {
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[mb],
            &[],
            &[],
        );
    }
}

// ---------------------------------------------------------------------------
// The smoke test.

const FILL_N: u32 = 555; // deliberately not a multiple of 64
const TAIL: u32 = 64; // guard words past the fill, checked untouched
const SENTINEL: u32 = 0xEEEE_EEEE;

/// Bindings, derived from the ONE rule. `smoke.hlsl` declares `b1` and
/// `u0..u2`, all in the default space, so these are set 0 at the shifted
/// bindings — and they are computed here rather than written down, because a
/// literal is exactly how a descriptor-set layout drifts away from the
/// `-fvk-*-shift` flags the module was compiled with.
fn smoke_bindings() -> [(u32, vk::DescriptorType); 4] {
    [
        (spirv::binding_of(Reg::B, 1), vk::DescriptorType::UNIFORM_BUFFER),
        (spirv::binding_of(Reg::U, 0), vk::DescriptorType::STORAGE_BUFFER),
        (spirv::binding_of(Reg::U, 1), vk::DescriptorType::STORAGE_BUFFER),
        (spirv::binding_of(Reg::U, 2), vk::DescriptorType::STORAGE_BUFFER),
    ]
}

struct SmokePipes {
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    modules: Vec<vk::ShaderModule>,
    pipes: Vec<vk::Pipeline>,
}

impl SmokePipes {
    fn destroy(&self, d: &ash::Device) {
        unsafe {
            for p in &self.pipes {
                d.destroy_pipeline(*p, None);
            }
            for m in &self.modules {
                d.destroy_shader_module(*m, None);
            }
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

fn build_pipes(d: &ash::Device, sp: &Spirv, entries: &[&str]) -> Result<SmokePipes, String> {
    let binds: Vec<vk::DescriptorSetLayoutBinding> = smoke_bindings()
        .iter()
        .map(|&(b, t)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(t)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let sl = unsafe {
        d.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
            None,
        )
    }
    .map_err(|e| format!("vkCreateDescriptorSetLayout: {e}"))?;
    let sls = [sl];
    let layout = unsafe {
        d.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&sls),
            None,
        )
    }
    .map_err(|e| format!("vkCreatePipelineLayout: {e}"))?;

    let mut modules = Vec::new();
    let mut pipes = Vec::new();
    for e in entries {
        let words = sp.compile(crate::gfx::shaders::SMOKE_HLSL, e, "cs_6_5", &format!("smoke {e}"), false)?;
        let m = unsafe {
            d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|err| format!("vkCreateShaderModule({e}): {err}"))?;
        modules.push(m);
        let name = CString::new(*e).unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(m)
            .name(&name);
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let p = unsafe { d.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) }
            .map_err(|(_, err)| format!("vkCreateComputePipelines({e}): {err}"))?;
        pipes.push(p[0]);
    }
    Ok(SmokePipes { layout, set_layout: sl, modules, pipes })
}

/// One pass of the seed -> prep -> indirect-fill chain at a chosen count.
/// Returns `(outbuf words, indirect args, counters)`.
#[allow(clippy::type_complexity)]
fn smoke_pass(
    hg: &VkHeadless,
    pipes: &SmokePipes,
    n: u32,
) -> Result<(Vec<u32>, [u32; 3], [u32; 2]), String> {
    let push_bytes: Vec<u8> = [n, 0u32, 0, 0].iter().flat_map(|w| w.to_le_bytes()).collect();
    let vkd = &hg.vk;
    let d = &vkd.device;
    let words = (FILL_N + TAIL) as u64;

    let ub = vk::BufferUsageFlags::UNIFORM_BUFFER;
    let sb = vk::BufferUsageFlags::STORAGE_BUFFER;
    let ind = vk::BufferUsageFlags::INDIRECT_BUFFER;
    let src = vk::BufferUsageFlags::TRANSFER_SRC;
    let dst = vk::BufferUsageFlags::TRANSFER_DST;

    let push = vkd.buffer(16, ub, true)?;
    let counters = vkd.buffer(8, sb | src | dst, false)?;
    let args = vkd.buffer(12, sb | ind | src, false)?;
    let outbuf = vkd.buffer(words * 4, sb | src | dst, false)?;

    let free = |vkd: &Vk| {
        vkd.free_buffer(&push);
        vkd.free_buffer(&counters);
        vkd.free_buffer(&args);
        vkd.free_buffer(&outbuf);
    };

    let body = (|| -> Result<(Vec<u32>, [u32; 3], [u32; 2]), String> {
        vkd.write(&push, &push_bytes)?;

        let sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3),
        ];
        let pool = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| format!("vkCreateDescriptorPool: {e}"))?;
        let sls = [pipes.set_layout];
        let set = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sls),
            )
        }
        .map_err(|e| format!("vkAllocateDescriptorSets: {e}"))?[0];

        let infos = [
            [vk::DescriptorBufferInfo::default().buffer(push.buf).range(vk::WHOLE_SIZE)],
            [vk::DescriptorBufferInfo::default().buffer(counters.buf).range(vk::WHOLE_SIZE)],
            [vk::DescriptorBufferInfo::default().buffer(args.buf).range(vk::WHOLE_SIZE)],
            [vk::DescriptorBufferInfo::default().buffer(outbuf.buf).range(vk::WHOLE_SIZE)],
        ];
        let writes: Vec<vk::WriteDescriptorSet> = smoke_bindings()
            .iter()
            .zip(infos.iter())
            .map(|(&(b, t), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b)
                    .descriptor_type(t)
                    .buffer_info(info)
            })
            .collect();
        unsafe { d.update_descriptor_sets(&writes, &[]) };

        hg.run(|d, cmd| unsafe {
            // Poison the output. A never-dispatched fill then reads back as
            // the sentinel and fails loudly at word 0, instead of reading back
            // as zeros and looking like a shader that ran and wrote nothing.
            d.cmd_fill_buffer(cmd, outbuf.buf, 0, vk::WHOLE_SIZE, SENTINEL);
            let mb = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mb],
                &[],
                &[],
            );

            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipes.layout,
                0,
                &[set],
                &[],
            );

            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipes.pipes[0]);
            d.cmd_dispatch(cmd, 1, 1, 1);
            compute_barrier(d, cmd);

            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipes.pipes[1]);
            d.cmd_dispatch(cmd, 1, 1, 1);

            // The args barrier is the one that matters: the writer is a
            // compute shader, the reader is the command processor, and
            // INDIRECT_COMMAND_READ at DRAW_INDIRECT is the only edge that
            // orders them.
            let mb = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::INDIRECT_COMMAND_READ | vk::AccessFlags::SHADER_READ,
                );
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mb],
                &[],
                &[],
            );

            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipes.pipes[2]);
            d.cmd_dispatch_indirect(cmd, args.buf, 0);
            compute_barrier(d, cmd);
        })?;

        unsafe { d.destroy_descriptor_pool(pool, None) };

        let ob = hg.read_buffer(&outbuf, words as usize * 4)?;
        let out: Vec<u32> =
            ob.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let ab = hg.read_buffer(&args, 12)?;
        let a: Vec<u32> =
            ab.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let cb = hg.read_buffer(&counters, 8)?;
        let c: Vec<u32> =
            cb.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok((out, [a[0], a[1], a[2]], [c[0], c[1]]))
    })();

    free(vkd);
    body
}

/// The gate. Two passes: the real count, and zero.
pub fn smoke_test(hg: &VkHeadless, sp: &Spirv) -> Result<(), String> {
    let pipes = build_pipes(&hg.vk.device, sp, &["cs_seed", "cs_prep", "cs_fill"])?;
    let r = smoke_run(hg, &pipes);
    pipes.destroy(&hg.vk.device);
    r
}

fn smoke_run(hg: &VkHeadless, pipes: &SmokePipes) -> Result<(), String> {
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
    // `id.x < counters[1]`. A guard that let them through would write past
    // the logical end of a real queue.
    for i in FILL_N..FILL_N + TAIL {
        if out[i as usize] != SENTINEL {
            return Err(format!(
                "smoke: outbuf[{i}] = {:#x} past the fill count — the bounds guard did not hold",
                out[i as usize]
            ));
        }
    }

    // Zero work. `prep` clamps y to 0, so the indirect dispatch launches
    // nothing at all — the empty-queue case every level of the real ladder
    // hits, and the one an emulated indirect path gets wrong.
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
    Ok(())
}
