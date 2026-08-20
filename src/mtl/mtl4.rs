//! The **Metal 4** submission path: the same smoke chain, through MTL4.
//!
//! `smoke.rs`'s peer, not its successor. It runs `smoke.hlsl` — the kernel
//! D3D12 runs at `gpu::trace::smoke_test`, Vulkan at `vk/headless.rs` and
//! Metal 3 next door — through `MTL4CommandQueue` / `MTL4CommandBuffer` /
//! `MTL4CommandAllocator` / `MTL4ArgumentTable` / `MTLResidencySet`, and holds
//! the result to the byte contract the Metal 3 path already meets.
//!
//! **THIS FILE IS THE ANSWER TO "ARE WE USING THE METAL 4 API?".** Before D4,
//! `grep -rn MTL4 src/ shim/` was empty — `MTL4Compiler` was in the dependency
//! graph, pulled in transitively by `objc2-metal-fx`, with no caller of ours.
//! Cargo.toml's own note records both halves of that and is now dated rather
//! than rewritten.
//!
//! # The claim, and why it is stronger than "MTL4 works"
//!
//! **One shader, one pipeline object, two submission APIs, identical bytes,
//! verified by one function.**
//!
//! `smoke::verify` is reused VERBATIM — it reads `out` / `args` / `counters`
//! off a `smoke::Pass` and knows nothing about how they got there, so nothing
//! here may weaken it. `pass` below returns a `smoke::Pass` for exactly that
//! reason, and `--check-mtl` K10 then compares the two paths' `Pass`es to each
//! other as well.
//!
//! The pipeline is SHARED, and that is a measurement rather than a shortcut:
//! `MTL4ComputeCommandEncoder::setComputePipelineState:` takes a plain
//! `MTLComputePipelineState`, the object `bind::Kernel` already builds through
//! `newComputePipelineStateWithFunction:`. So `MTL4Compiler` is not needed for
//! this rung at all, and the byte-compare has one fewer variable: what differs
//! between the two paths is submission, binding and residency, and nothing
//! else. A rung that also swapped the compiler could not say that.
//!
//! # The three things MTL4 took away
//!
//! Each is a line of code here and a claim the Metal 3 gate cannot make.
//!
//! **1. Hazard tracking.** `smoke::pass` states that Metal's default
//! `MTLDispatchTypeSerial` has the driver order and hazard-track the three
//! dispatches, so `vk/headless.rs`'s `compute_barrier` apparatus "has no
//! analogue here rather than being omitted". Under MTL4 it has one — see
//! `BARRIER` below. **This was the predicted risk and it is real:**
//! `FR_MTL4_NO_BARRIER` deterministically produces `indirect args [4, 1, 1]`,
//! which is `cs_prep` reading `counters` before `cs_seed` wrote it, and NEITHER
//! validation layer sees it. It is the one claim this backend gained by moving
//! to MTL4 rather than merely re-proving.
//!
//! **2. Residency.** There is no `useResource:` on ANY MTL4 encoder — checked
//! against every `MTL4*` binding in objc2-metal 0.3.2, not inferred from the
//! documentation. `MTL4ArgumentTable::setAddress:atIndex:` takes a raw
//! `MTLGPUAddress`, so Metal cannot infer residency from the bound object
//! because there is no bound object. **And omitting it is STILL unobservable
//! on Apple silicon without GPU validation** — measured, against the
//! expectation this rung was planned on. The API contract is real; the unified
//! memory model hides its violation exactly as it hides Metal 3's.
//!
//! **3. `waitUntilCompleted`.** `Mtl4::compute` signals a shared event and
//! blocks on it with a timeout, so a submission that never lands fails the gate
//! instead of hanging it.
//!
//! **4. The command buffer's `error`, and this one is NOT replaced.** MTL4 has
//! no synchronous error channel at all — only `MTL4CommitFeedback::error`
//! through an asynchronous block — so a faulted submission is caught here by
//! `smoke::verify` reading every word of the result rather than by Metal saying
//! so. `Mtl4`'s own doc carries the argument for why wiring the handler naively
//! would be a probe that cannot be shown to have reached its target, and why
//! the next rung to extend this path should wire it properly first.
//!
//! # What this file deliberately does NOT do
//!
//! It re-derives nothing. It takes `&smoke::Pipes`, builds its `ArgBuf`s
//! through the same `Kernel::arg_buf`, and reads the bind points back out of
//! `ArgBuf::set_binds` — so the two paths cannot disagree about where a set
//! lives, because neither of them writes the number down. `bind.rs` grew two
//! accessors for this and no logic.
//!
//! It also owns no textures. `texprobe`'s three passes stay on the Metal 3 path
//! this rung: `MTL4ArgumentTable` has `setTexture:atIndex:` and
//! `setSamplerState:atIndex:` alongside `setAddress:`, so the extension is
//! mechanical — but two unproven things in one rung is not how this backend
//! got built.

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandEncoder, MTL4ComputeCommandEncoder,
    MTL4VisibilityOptions, MTLAllocation, MTLBuffer, MTLDevice, MTLSize, MTLStages,
};

use crate::mtl::bind::{ArgBuf, Kernel};
use crate::mtl::device::{Mtl, Mtl4};
use crate::mtl::smoke::{self, COUNT_POISON, FILL_N, GRID_POISON, SENTINEL, TAIL};

/// The barrier between the three dispatches, and the reason it is a named
/// constant rather than three literal calls.
///
/// `MTLStages::Dispatch` before and after, with `MTL4VisibilityOptions::Device`
/// — an execution barrier AND a cache flush to the device coherence point.
/// `MTL4VisibilityOptions::None` would give the execution half only, which is
/// documented in the header as turning the barrier into an execution barrier
/// and is NOT enough here: `cs_prep` writes `args` and the command processor
/// reads it as a dispatch grid, which is the one hazard in this chain that is
/// not a plain shader-to-shader read.
///
/// **Named because the two halves must not drift apart.** A future reader
/// weakening this to `None` for one dispatch and not the other would produce a
/// chain that works on the box it was tested on.
const BARRIER: (MTLStages, MTLStages, MTL4VisibilityOptions) =
    (MTLStages::Dispatch, MTLStages::Dispatch, MTL4VisibilityOptions::Device);

/// Levers, the `FR_ABL` read-only-probe idiom — default off, loud when armed.
///
/// **TWO TEETH AND ONE MEASUREMENT, and the split was MEASURED rather than
/// predicted — the plan predicted it the other way round.** Apple M1, macOS
/// 26.5.1, 2026-08-19:
///
/// ```text
///  FR_MTL4_TABLE_INDEX    TOOTH   K10 "indirect args [3, 3, 3], expected [9, 1, 1]"
///                                 — GRID_POISON intact, its Metal 3 twin's signature
///  FR_MTL4_NO_BARRIER     TOOTH   K10 "indirect args [4, 1, 1], expected [9, 1, 1]"
///                                 — 7/7 identical, plain and under BOTH layers
///  FR_MTL4_NO_RESIDENCY   MEAS.   UNOBSERVABLE plain and under MTL_DEBUG_LAYER=1;
///                                 observable under MTL_SHADER_VALIDATION=1 only
/// ```
///
/// **`[4, 1, 1]` IS THE WHOLE BARRIER ARGUMENT IN ONE NUMBER.** It is not the
/// poison — it is `COUNT_POISON.div_ceil(64)`, i.e. 238 rounded up over the
/// group size. `cs_prep` RAN, and read `counters` before `cs_seed` had written
/// it. A plain read-before-write, deterministic, and invisible to both
/// validation layers. Metal 3's gate cannot make this mistake because
/// `MTLDispatchTypeSerial` has the driver order and hazard-track the
/// dispatches; MTL4 hands that back to us, and this is what it looks like when
/// it is dropped.
///
/// **THE RESIDENCY PREDICTION WAS WRONG, and that is the other finding.** The
/// rung was planned expecting binding-by-raw-address to make an omitted
/// residency declaration fail outright, promoting Metal 3's MEASUREMENT into a
/// TOOTH. It does not: `FR_MTL4_NO_RESIDENCY` behaves EXACTLY like
/// `FR_MTL_NO_RESIDENCY` — invisible until GPU validation is armed, at which
/// point `cs_prep` never writes `args` and the grid poison survives. On unified
/// memory a non-heap buffer is already page-backed, and the argument table
/// taking an address rather than an object does not change that. Recorded as
/// measured, not quietly reclassified: `must_fail` below therefore does NOT
/// name it, exactly as `smoke::Plant::must_fail` does not name its twin.
///
/// (Its first measurement was CIRCULAR and had to be retaken. K10 asserted the
/// residency-set allocation count, which this plant zeroes by construction, so
/// the gate failed on having noticed its own plant and reported "observable"
/// having learned nothing. The assertion is now skipped when the lever is
/// armed, which is what left only the DATA able to fail.)
#[derive(Clone, Copy, Default)]
pub struct Plant {
    /// Drop the residency set entirely. **MEASURED to behave exactly like
    /// Metal 3's `FR_MTL_NO_RESIDENCY`** — see the struct doc for why that was
    /// the surprise and not the expectation.
    pub no_residency: bool,
    /// Drop the barriers between the three dispatches. **A TOOTH, and the one
    /// claim this backend could not make before D4:** Metal 3's driver does the
    /// hazard tracking, so its gate can neither make this mistake nor catch it.
    pub no_barrier: bool,
    /// Bind every argument buffer at table index 1 instead of its derived
    /// index. `FR_MTL_ARGBUF_INDEX`'s twin, which is MEASURED to produce
    /// `indirect args [3, 3, 3], expected [9, 1, 1]` — the `GRID_POISON`
    /// intact, because `cs_prep` never ran against the right pointers.
    pub table_index: bool,
    /// Report MTL4 as absent on a machine that has it, so the SKIP branch is
    /// exercised somewhere it can be observed. Not a tooth: its whole point is
    /// that the gate stays green.
    pub off: bool,
}

impl Plant {
    pub fn from_env() -> Plant {
        let on = |k: &str| std::env::var(k).is_ok_and(|v| v != "0");
        Plant {
            no_residency: on("FR_MTL4_NO_RESIDENCY"),
            no_barrier: on("FR_MTL4_NO_BARRIER"),
            table_index: on("FR_MTL4_TABLE_INDEX"),
            off: on("FR_MTL4_OFF"),
        }
    }

    pub fn any(&self) -> bool {
        self.no_residency || self.no_barrier || self.table_index || self.off
    }

    /// The levers that MUST make the gate fail, which K5's verdict enforces.
    ///
    /// `off` is absent because it suppresses the whole stage. `no_residency` is
    /// absent because it was MEASURED to be unobservable without GPU
    /// validation, the same answer its Metal 3 twin gives — demanding that it
    /// fail would make the gate fail on a correct machine, which is the reason
    /// `smoke::Plant` gives for the identical exclusion. The other two are here
    /// and both were measured biting: see the struct doc for their signatures.
    pub fn must_fail(&self) -> bool {
        self.table_index || self.no_barrier
    }

    /// The armed levers, for the gate's PLANT line.
    pub fn line(&self) -> String {
        let names = [
            (self.no_residency, "FR_MTL4_NO_RESIDENCY"),
            (self.no_barrier, "FR_MTL4_NO_BARRIER"),
            (self.table_index, "FR_MTL4_TABLE_INDEX"),
            (self.off, "FR_MTL4_OFF"),
        ];
        names.iter().filter(|(on, _)| *on).map(|(_, n)| *n).collect::<Vec<_>>().join(" ")
    }
}

/// One pass of the chain through MTL4, at a given fill count.
///
/// The buffer set-up is `smoke::pass`'s verbatim — the same four allocations,
/// the same three poisons with the same two bounds — because a difference there
/// would show up in the byte-compare as a difference in SUBMISSION. Read
/// `smoke.rs`'s header for why the poisons are per buffer and why two of them
/// are bounded rather than `SENTINEL`.
pub fn pass(
    m: &Mtl,
    g: &Mtl4,
    p: &smoke::Pipes,
    n: u32,
    plant: Plant,
) -> Result<smoke::Pass, String> {
    let words = (FILL_N + TAIL) as usize;

    let push = m.buffer(16)?;
    let counters = m.buffer(4 * 4)?;
    let args = m.buffer(16)?;
    let outbuf = m.buffer(words * 4)?;

    m.fill_words(&outbuf, SENTINEL);
    m.fill_words(&counters, COUNT_POISON);
    m.fill_words(&args, GRID_POISON);
    m.write_words(&push, &[n, 0, 0, 0]);

    // One argument buffer per kernel, for `bind.rs`'s measured reason: the
    // `[[id(n)]]` assignment is PER ENTRY POINT, so sharing one would bind the
    // wrong pointers with no error anywhere.
    let mut ab_seed = p.seed.arg_buf(m)?;
    ab_seed.set_buffer("Push", &push)?;
    ab_seed.set_buffer("counters", &counters)?;

    let mut ab_prep = p.prep.arg_buf(m)?;
    ab_prep.set_buffer("counters", &counters)?;
    ab_prep.set_buffer("args", &args)?;

    let mut ab_fill = p.fill.arg_buf(m)?;
    ab_fill.set_buffer("counters", &counters)?;
    ab_fill.set_buffer("outbuf", &outbuf)?;

    // ---- residency ------------------------------------------------------
    //
    // EVERY ALLOCATION THE THREE PASSES REACH, IN ONE SET. `ArgBuf::
    // resident_allocations` returns what the Metal 3 path declares through
    // `useResource:` PLUS the argument buffers themselves, which that path gets
    // for free by binding them as objects and this one cannot — its doc carries
    // the argument. The four data buffers appear through the ArgBufs that point
    // at them, so nothing is added here by hand: a resource this file forgot to
    // put in an argument buffer would be missing from the shader anyway.
    //
    // DEDUPLICATED BY IDENTITY, AND THE COUNT IS THE MEASUREMENT THAT FORCED
    // IT. The three ArgBufs name nine entries between them but only SEVEN
    // distinct allocations — `counters` is reached by all three kernels,
    // `push`/`args`/`outbuf` by one each, plus the three argument buffers —
    // and `MTLResidencySet::allocationCount` reported 7 for nine
    // `addAllocation:` calls on the first run. So the set deduplicates, which
    // means a count compared against the number of CALLS is a gate that fails
    // on correct code. Deduplicating here rather than tolerating the gap keeps
    // the anti-vacuity check EXACT: `allocationCount` must equal what we asked
    // for, and any other number is a set that did not take.
    let mut allocs: Vec<&ProtocolObject<dyn MTLAllocation>> = Vec::new();
    for ab in [&ab_seed, &ab_prep, &ab_fill] {
        for a in ab.resident_allocations() {
            let id = std::ptr::from_ref(a).cast::<()>();
            if !allocs.iter().any(|e| std::ptr::from_ref(*e).cast::<()>() == id) {
                allocs.push(a);
            }
        }
    }
    let (set, resident) = if plant.no_residency {
        (None, 0)
    } else {
        let (s, n) = g.residency(m, &allocs)?;
        (Some(s), n)
    };
    // ANTI-VACUITY, and it is not the same number as the Metal 3 path's. A set
    // that silently came back empty would otherwise be indistinguishable from
    // one that covered everything — `smoke::Pass::resident`'s whole reason for
    // existing — and here the failure mode is worse, because an empty set on
    // this path means every address is unbacked.
    if !plant.no_residency && resident != allocs.len() {
        return Err(format!(
            "the residency set holds {resident} of the {} DISTINCT allocations it was \
             given — addAllocation: or commit did not take",
            allocs.len()
        ));
    }

    let one = MTLSize { width: 1, height: 1, depth: 1 };
    let tg = |k: &Kernel| MTLSize {
        width: k.local_size[0] as usize,
        height: k.local_size[1] as usize,
        depth: k.local_size[2] as usize,
    };

    // ---- the argument table ---------------------------------------------
    //
    // ONE TABLE FOR THE WHOLE CHAIN, re-pointed between dispatches, which is
    // the MTL4 analogue of `setBuffer:offset:atIndex:` per kernel rather than a
    // departure from it. `maxBufferBindCount` has to cover the highest index
    // any set is bound at, +1 — a table too small is an ERROR at
    // `setAddress:atIndex:` by the descriptor's own documentation, not a
    // silently ignored bind.
    let highest = [&ab_seed, &ab_prep, &ab_fill]
        .iter()
        .flat_map(|ab| ab.set_binds().into_iter().map(|(i, _)| i))
        .max()
        .unwrap_or(0);
    let desc = MTL4ArgumentTableDescriptor::new();
    // The plant's index 1 must fit too, or it would fail as a table-size error
    // rather than as the mis-bind it is meant to be — a plant that fails for
    // the wrong reason proves nothing.
    desc.setMaxBufferBindCount(highest.max(1) as usize + 1);
    let table = m
        .device()
        .newArgumentTableWithDescriptor_error(&desc)
        .map_err(|e| format!("newArgumentTableWithDescriptor: {e}"))?;

    let bind = |ab: &ArgBuf| {
        for (index, buf) in ab.set_binds() {
            let at = if plant.table_index { 1 } else { index } as usize;
            // SAFETY: `at` is below `maxBufferBindCount` by the descriptor
            // above, which covers both the derived indices and the plant's 1.
            unsafe { table.setAddress_atIndex(buf.gpuAddress(), at) };
        }
    };

    g.compute(m, |enc| {
        // The set is attached to the QUEUE by `Mtl4::residency`, and also
        // declared on the command buffer through the encoder's owner — see
        // `Mtl4::residency`'s doc for why all of it is one function.
        let barrier = |enc: &ProtocolObject<dyn MTL4ComputeCommandEncoder>| {
            if !plant.no_barrier {
                let (after, before, vis) = BARRIER;
                enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    after, before, vis,
                );
            }
        };

        enc.setArgumentTable(Some(&table));

        enc.setComputePipelineState(p.seed.pipeline());
        bind(&ab_seed);
        enc.dispatchThreadgroups_threadsPerThreadgroup(one, tg(&p.seed));
        barrier(enc);

        enc.setComputePipelineState(p.prep.pipeline());
        bind(&ab_prep);
        enc.dispatchThreadgroups_threadsPerThreadgroup(one, tg(&p.prep));
        // THE BARRIER THAT IS NOT LIKE THE OTHERS. `cs_prep` writes `args` and
        // the next line hands `args` to the COMMAND PROCESSOR as a threadgroup
        // grid, so the consumer is not another shader. `MTL4VisibilityOptions::
        // Device` is what makes the write visible at the device coherence point
        // rather than merely ordering the two dispatches.
        barrier(enc);

        enc.setComputePipelineState(p.fill.pipeline());
        bind(&ab_fill);
        // THE INDIRECT DISPATCH, AND IT DOES NOT TAKE A BUFFER. Metal 3's form
        // is `(buffer, offset, threadsPerThreadgroup)`; MTL4's is
        // `(MTLGPUAddress, threadsPerThreadgroup)` — the offset is GONE,
        // folded into the address the caller computes. `smoke::pass` passes
        // offset 0, so `gpuAddress()` alone is the same dispatch; a non-zero
        // offset would become `gpuAddress() + off` here, and Apple's own doc
        // requires 4-byte alignment of the result.
        //
        // Apple's documentation for this selector says, in as many words, "Use
        // an instance of MTLResidencySet to mark residency of the indirect
        // buffer" — the command processor reads `args` through the same raw
        // address the shaders reach their data by, and nothing else keeps it
        // backed.
        //
        // SAFETY: `args` is 16 B, 4-byte aligned by allocation, and the
        // dispatch reads the 12 that `MTLDispatchThreadgroupsIndirectArguments`
        // describes; it is resident for this submission through the set above.
        unsafe {
            enc.dispatchThreadgroupsWithIndirectBuffer_threadsPerThreadgroup(
                args.gpuAddress(),
                tg(&p.fill),
            );
        }
        Ok(())
    })?;

    // Outside the closure, for `smoke::pass`'s reason: `compute` blocks, and a
    // read inside would race the GPU with no diagnostic — unified memory makes
    // a stale read a perfectly valid read of the wrong moment.
    let c = m.read_words(&counters, 2);
    let a = m.read_words(&args, 3);
    let out = m.read_words(&outbuf, words);

    // AFTER the readback: the set is what keeps the addresses backed, and
    // dropping it before the data is read is the failure this whole module is
    // about.
    //
    // AND ONLY ON THE SUCCESS PATH, WHICH IS DELIBERATE AND NOT A LEAK LEFT
    // LYING. The `?` on `compute` above skips this, so a failed submission
    // leaves the set attached to the queue — and that is the SAFE ending, not
    // the untidy one. `compute` fails either before anything was submitted or
    // because the wait TIMED OUT, and in the second case the GPU may still be
    // reading through exactly these addresses; `endResidency` there would
    // revoke the backing out from under live work, which is worse than an
    // over-long residency in a process that is already failing the gate. The
    // set is released when the run that can prove the GPU is done with it says
    // so, and at no other time. See `Mtl4::drop_residency`.
    if let Some(s) = &set {
        g.drop_residency(s);
    }

    Ok(smoke::Pass { out, args: [a[0], a[1], a[2]], counters: [c[0], c[1]], resident })
}

/// Both passes, verified by `smoke::verify` — the real count and zero.
///
/// **`smoke::verify` IS THE POINT AND IS NOT REIMPLEMENTED HERE.** A second
/// verifier that agreed with the first would prove that two functions written
/// by the same hand on the same day agree; reusing the one the Metal 3 path is
/// already held to is what makes "identical bytes" mean anything. The zero-work
/// pass matters more on this path than on that one, because MTL4's indirect
/// dispatch of a zero grid is documented nowhere either.
pub fn run(m: &Mtl, g: &Mtl4, p: &smoke::Pipes, plant: Plant) -> Result<(usize, usize), String> {
    let a = pass(m, g, p, FILL_N, plant)?;
    smoke::verify(&a, FILL_N)?;
    let z = pass(m, g, p, 0, plant)?;
    smoke::verify(&z, 0)?;
    Ok((a.resident, z.resident))
}

/// The cross-API comparison: two submission paths, one shader, one verifier.
///
/// Both `Pass`es have already been through `smoke::verify` when this runs, so
/// what it adds is not "are they correct" but "are they the SAME" — and the
/// distinction is real, because `verify` checks `out[i] == i ^ 0x00C0FFEE` for
/// `i < n` and `SENTINEL` past it, which is every word. The comparison is
/// therefore expected to be redundant TODAY and is here for the day `verify`
/// grows a tolerance, or a field, that it is not.
///
/// **`resident` is deliberately NOT compared.** The two paths' counts differ by
/// the set count by construction — `ArgBuf::resident_allocations` includes the
/// argument buffers and `resident_count` does not, for the reason that
/// function's doc gives. Comparing them would encode an off-by-three that a
/// future set-count change would silently break.
pub fn cross_check(mtl3: &smoke::Pass, mtl4: &smoke::Pass) -> Result<(), String> {
    if mtl3.args != mtl4.args {
        return Err(format!(
            "indirect args differ: Metal 3 {:?}, MTL4 {:?}",
            mtl3.args, mtl4.args
        ));
    }
    if mtl3.counters != mtl4.counters {
        return Err(format!(
            "counters differ: Metal 3 {:?}, MTL4 {:?}",
            mtl3.counters, mtl4.counters
        ));
    }
    if mtl3.out.len() != mtl4.out.len() {
        return Err(format!(
            "outbuf lengths differ: Metal 3 {}, MTL4 {}",
            mtl3.out.len(),
            mtl4.out.len()
        ));
    }
    if let Some(i) = (0..mtl3.out.len()).find(|&i| mtl3.out[i] != mtl4.out[i]) {
        let diff = (0..mtl3.out.len()).filter(|&i| mtl3.out[i] != mtl4.out[i]).count();
        return Err(format!(
            "outbuf differs in {diff} of {} words, first at [{i}]: Metal 3 {:#x}, MTL4 {:#x}",
            mtl3.out.len(),
            mtl3.out[i],
            mtl4.out[i]
        ));
    }
    Ok(())
}
