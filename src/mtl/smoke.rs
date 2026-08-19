//! The Metal smoke chain: seed -> prep -> **indirect** fill.
//!
//! `src/vk/headless.rs:212-500`'s twin, over the same `smoke.hlsl`
//! (`gfx::shaders::SMOKE_HLSL`) that D3D12 runs at `gpu::trace::smoke_test`.
//! Three backends, one kernel — which is what makes the three results
//! comparable line for line, and what makes this milestone's claim "Metal does
//! what the other two do" rather than "Metal does something".
//!
//! What it proves is the wavefront tracer's own machinery, in miniature:
//! constants reach a kernel, a storage buffer is written, **a GPU-written
//! counter becomes dispatch arguments**, and a third kernel launches from those
//! arguments with the CPU never seeing the count.
//!
//! # THE POISON IS PER BUFFER, and that is not tidiness
//!
//! `vk/headless.rs:379` poisons `outbuf` alone, and is right to: Vulkan runs
//! with a validation layer that catches a mis-bound descriptor directly. Metal
//! has no layer armed by default and CI runs the Metal job with them explicitly
//! off (`ci.yml:637-664`), so **the data is the only instrument** — an
//! unpoisoned `counters` reads as zeros, and "never bound" becomes
//! indistinguishable from "correctly bound, count 0", which is exactly the
//! observable the count-0 arm depends on. So all four buffers are poisoned.
//!
//! And they are poisoned with DIFFERENT values, which is a SAFETY property
//! rather than tidiness — the plants below are what make it one, and the two
//! bounds are not obvious:
//!
//! * **`args` is read by the hardware as a threadgroup GRID.** It is passed to
//!   `dispatchThreadgroupsWithIndirectBuffer:` directly, so any plant that
//!   stops `cs_prep` writing it leaves the poison AS the grid. `SENTINEL` there
//!   is `0xEEEE_EEEE` cubed — about 5e10 threadgroups, which on a dev Mac takes
//!   the WindowServer with it and in CI is a 45-minute timeout. Hence
//!   `GRID_POISON`, whose cube is 27.
//! * **`counters[1]` is `cs_fill`'s bounds guard.** The kernel writes
//!   `outbuf[id.x]` for every `id.x < counters[1]`, so a poison larger than
//!   `outbuf` is an out-of-bounds write — heap corruption, not a failed gate.
//!   Hence `COUNT_POISON`, asserted below to be inside the allocation.
//!
//! Neither value is producible by `smoke.hlsl`, so both still read as "never
//! written" in a failure message.
//!
//! # The constants are DUPLICATED from `vk/headless.rs`, deliberately
//!
//! `FILL_N` / `TAIL` / `SENTINEL` and the `[FILL_N.div_ceil(64), 1, 1]`
//! expectation are properties of `smoke.hlsl`, so their doctrinally right home
//! is beside `gfx::shaders::SMOKE_HLSL`. Hoisting them there would change
//! `gpu::trace::smoke_test`, which no macOS box can re-run — the argument
//! `msl.rs:186-192` makes about the `[loop]` workaround, in the same shape. So:
//! duplicated now, and **the hoist belongs to whoever next touches
//! `smoke.hlsl` from a Windows box, with `--check-gpu` re-run.**

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::mtl::bind::{self, Kernel};
use crate::mtl::device::Mtl;
use crate::mtl::msl::Msl;
use crate::spirv::Spirv;

/// Deliberately not a multiple of 64 — `vk/headless.rs:214`.
pub const FILL_N: u32 = 555;
/// Guard words past the fill, checked untouched.
pub const TAIL: u32 = 64;
/// `vk/headless.rs:216`, byte-uniform so the two gates poison identically.
pub const SENTINEL: u32 = 0xEEEE_EEEE;
/// The poison for `counters`. Bounded by `outbuf`'s length, because
/// `cs_fill`'s guard is `id.x < counters[1]` and a larger value is an
/// out-of-bounds write rather than a failed assertion. `(238 + 63) / 64 = 4`
/// groups if it reaches `cs_prep`'s arithmetic, and it is plainly not 555.
pub const COUNT_POISON: u32 = 0x0000_00EE;
/// The poison for `args`, which the hardware reads as a threadgroup GRID.
/// Bounded because it IS the grid: `3 * 3 * 3 = 27` groups, against
/// `SENTINEL`'s ~5e10. Plainly not `[9, 1, 1]` and not `[0, 0, 1]`.
pub const GRID_POISON: u32 = 3;

// `cs_fill` writes `outbuf[id.x]` for every `id.x < counters[1]`. If a plant
// leaves `counters[1]` poisoned, that bound is COUNT_POISON — so the allocation
// has to cover it, or a deliberately-wrong bind corrupts the heap instead of
// failing the gate.
const _: () = assert!(COUNT_POISON <= FILL_N + TAIL);

/// What one pass read back.
pub struct Pass {
    pub out: Vec<u32>,
    pub args: [u32; 3],
    pub counters: [u32; 2],
    /// Resources declared through `useResource:` — the anti-vacuity number for
    /// a residency list that silently came back empty.
    pub resident: usize,
}

/// Levers, the `FR_ABL` read-only-probe idiom: default off, loud when armed.
///
/// **Four are TEETH and one is a MEASUREMENT, and the asymmetry is the
/// `Expect::NoAnalogue` / `Expect::ToolDefect` split (`msl.rs:129-150`) in
/// another currency.** A tooth makes a claim about OUR code, so it must fire; a
/// measurement makes a claim about the platform, so it is reported. Demanding
/// that `no_residency` fail would make the gate fail on a machine where the
/// omission is genuinely unobservable — which is every Apple-silicon Mac.
///
/// Measured on an Apple M1, and each observable is worth keeping because it is
/// what a future regression would stop producing:
///
/// ```text
///  FR_MTL_SEED_MAP       K3/K4  "no argument-buffer member named `args`"
///  FR_MTL_ARGBUF_INDEX   K4     "indirect args [3, 3, 3], expected [9, 1, 1]"
///                               — but ABORTS under MTL_DEBUG_LAYER=1, below
///  FR_MTL_SWAP_MEMBERS   K4     "indirect args [3, 3, 3], expected [9, 1, 1]"
///  FR_MTL_TG_ONE         K4     "outbuf[9] = 0xeeeeeeee ... (never written)"
///  FR_MTL_NO_RESIDENCY   —      bit-identical plain and under
///                               MTL_DEBUG_LAYER=1; OBSERVABLE under
///                               MTL_SHADER_VALIDATION=1 (re-measured, below)
/// ```
///
/// **THE RESIDENCY LINE WAS RE-MEASURED AND CHANGED**, 2026-08-19, Apple M1,
/// macOS 26.5.1, 3/3 reproducible. It used to read *"bit-identical to a clean
/// run, plain AND under MTL_DEBUG_LAYER=1 AND MTL_SHADER_VALIDATION=1"*. Under
/// `MTL_SHADER_VALIDATION=1` it is not: `cs_prep` never writes `args`, so K4
/// reports `indirect args [3, 3, 3], expected [9, 1, 1]` — the poison intact.
///
/// Two things about HOW it is observable matter more than the fact:
///
/// * **The command buffer reports no error.** `device::run` checks `cb.error()`
///   and gets `None`; the writes simply do not land. So the layer does not
///   diagnose this — the DATA does, which is the argument this whole gate is
///   built on, arriving from a direction it did not expect.
/// * `mtl::texprobe` measured the same reversal for TEXTURES on the same day,
///   so this is a property of the layer and not of the resource class.
///
/// Which measurement was wrong is not recoverable — C2's was taken on this
/// machine and this OS is a point release newer. The rule the tree keeps is
/// the one that survives either way: **the omission is not reliably
/// unobservable, so residency stays structural** rather than checked. The
/// `Expect`-style asymmetry below is unaffected — this lever still makes a
/// claim about the PLATFORM, and a platform whose answer moved is exactly why
/// it is reported rather than asserted.
///
/// **`ARGBUF_INDEX` IS THE ONE LEVER HERE THAT CANNOT BE RUN UNDER
/// `MTL_DEBUG_LAYER=1`**, measured 2026-08-19 alongside the residency
/// re-measurement:
///
/// ```text
/// validateComputeFunctionArguments:1038: failed assertion `Compute
///   Function(cs_seed): missing Buffer binding at index 0 for
///   spvDescriptorSet0[0].'                                      exit 134
/// ```
///
/// Unlike `texprobe`'s `ARR_STRIDE` — whose first draft aborted because the
/// plant carried a SECOND, gratuitous defect (an out-of-bounds write) beyond
/// the one it meant to plant, and was fixed — this one is inherent. The defect
/// being planted IS "nothing is bound at the index the function reads", which
/// is precisely what that layer exists to catch, and its way of reporting is to
/// abort. There is no valid-but-wrong index for a single-set kernel to move to.
///
/// So it is recorded rather than fixed, and the run-list means "each lever, and
/// separately the layers on a CLEAN run" — not the cross product. A future
/// reader who arms this one under the layer and gets a SIGABRT is seeing the
/// documented behaviour, not a regression.
///
/// `SEED_MAP` failing in K3/K4 rather than in a readback is the strongest of
/// the four: binding `cs_prep`'s resources through `cs_seed`'s map is what a
/// hand-written table produces, and it is caught STRUCTURALLY — the member does
/// not exist — before a dispatch happens at all.
#[derive(Clone, Copy, Default)]
pub struct Plant {
    /// Bind `cs_prep` through `cs_seed`'s argument-buffer map — precisely what
    /// a hand-written table produces, and the measured per-entry-point ids are
    /// why it is wrong.
    pub seed_map_on_prep: bool,
    /// Bind the argument buffer at `[[buffer(1)]]` instead of the derived index.
    pub argbuf_index: bool,
    /// Write `outbuf`'s address into `counters`' slot and vice versa.
    pub swap_members: bool,
    /// Dispatch `cs_fill` at `threadsPerThreadgroup = 1` instead of its derived
    /// group size — the Metal-specific tooth, because no other backend takes
    /// this parameter from the host at all.
    pub threadgroup_one: bool,
    /// Skip the `useResource:` replay. Expected to be INVISIBLE on unified
    /// memory; the gate reports which it was rather than assuming.
    pub no_residency: bool,
}

impl Plant {
    pub fn from_env() -> Plant {
        let on = |k: &str| std::env::var(k).is_ok_and(|v| v != "0");
        Plant {
            seed_map_on_prep: on("FR_MTL_SEED_MAP"),
            argbuf_index: on("FR_MTL_ARGBUF_INDEX"),
            swap_members: on("FR_MTL_SWAP_MEMBERS"),
            threadgroup_one: on("FR_MTL_TG_ONE"),
            no_residency: on("FR_MTL_NO_RESIDENCY"),
        }
    }

    pub fn any(&self) -> bool {
        self.must_fail() || self.no_residency
    }

    /// The four that are TEETH: each asserts something about this module's own
    /// code, so an armed run that PASSES is a defect in the gate.
    ///
    /// `no_residency` is deliberately absent — see the struct doc. It probes a
    /// PLATFORM behaviour, and the measured answer is that the omission is
    /// unobservable on unified memory even with both validation layers armed.
    pub fn must_fail(&self) -> bool {
        self.seed_map_on_prep || self.argbuf_index || self.swap_members || self.threadgroup_one
    }

    pub fn line(&self) -> String {
        let mut v = Vec::new();
        for (on, name) in [
            (self.seed_map_on_prep, "FR_MTL_SEED_MAP"),
            (self.argbuf_index, "FR_MTL_ARGBUF_INDEX"),
            (self.swap_members, "FR_MTL_SWAP_MEMBERS"),
            (self.threadgroup_one, "FR_MTL_TG_ONE"),
            (self.no_residency, "FR_MTL_NO_RESIDENCY"),
        ] {
            if on {
                v.push(name);
            }
        }
        v.join(",")
    }
}

/// The three compiled entry points.
pub struct Pipes {
    pub seed: Kernel,
    pub prep: Kernel,
    pub fill: Kernel,
}

impl Pipes {
    /// The three, in dispatch order, paired with the entry name the gate
    /// reports them under.
    pub fn all(&self) -> [(&'static str, &Kernel); 3] {
        [("cs_seed", &self.seed), ("cs_prep", &self.prep), ("cs_fill", &self.fill)]
    }
}

/// HLSL -> SPIR-V -> MSL -> metallib -> library -> pipeline, for all three
/// entry points, with the argument-buffer map derived off each.
pub fn build(m: &Mtl, msl: &Msl, sp: &Spirv) -> Result<Pipes, String> {
    let src = crate::gfx::shaders::SMOKE_HLSL;
    let mut built = Vec::new();
    for entry in ["cs_seed", "cs_prep", "cs_fill"] {
        let words = sp.compile(src, entry, "cs_6_5", &format!("smoke {entry}"), false)?;
        let msl_src = msl.transpile(&words, entry)?;
        let lib = msl.compile_lib(&msl_src, entry)?;
        // `load` reflects the SPIR-V itself, so the map and the descriptors it
        // is cross-checked against provably come from one module.
        built.push(bind::load(m, &lib, &words).map_err(|e| format!("{entry}: {e}"))?);
    }
    let mut it = built.into_iter();
    Ok(Pipes { seed: it.next().unwrap(), prep: it.next().unwrap(), fill: it.next().unwrap() })
}

/// One pass of the chain at a given fill count.
pub fn pass(m: &Mtl, p: &Pipes, n: u32, plant: Plant) -> Result<Pass, String> {
    let words = (FILL_N + TAIL) as usize;

    // `Push` is 4 uints = 16 B. GOTCHA 1 (`ffx_fsr3_metal.mm:862-878`):
    // spirv-cross rounds an MSL constant-buffer struct up to 16-byte alignment,
    // so a bind must cover the PADDED size. 16 is already a multiple of 16 here
    // — the rule is inert for this kernel and stated because `FrameCb` is
    // 5616 B and arrives at the next rung.
    let push = m.buffer(16)?;
    // FOUR words, not the two `smoke.hlsl` reads, and the extra pair is a
    // SAFETY margin for the swap plant rather than slack. spirv-cross emits
    // `packed_uint3 _m0[1]` for `RWStructuredBuffer<uint3>` (measured — see
    // `args` below), so `cs_prep`'s `args[0] = uint3(...)` is a 12-byte store —
    // and `swap_members` deliberately points the `args` MEMBER at THIS buffer.
    // At two words that store runs 4 bytes past the allocation. Benign in
    // practice (Metal rounds allocations up, and neither validation layer
    // reports it — measured), but it is the same hazard class the
    // `COUNT_POISON` assert above exists to prevent, and a plant that corrupts
    // memory instead of failing an assertion is not a tooth. The readback is
    // still `read_words(&counters, 2)`.
    let counters = m.buffer(4 * 4)?;
    // 16, not 12, and the reason is measured rather than folklore: spirv-cross
    // emits `packed_uint3 _m0[1]` for `RWStructuredBuffer<uint3>`, which is 12 B
    // — the plain MSL `uint3` that would force 16 is NOT what it used. The extra
    // word costs nothing and keeps the allocation right if that ever changes.
    let args = m.buffer(16)?;
    let outbuf = m.buffer(words * 4)?;

    // `outbuf` is neither a grid nor a bound, so it keeps the full sentinel and
    // stays byte-identical to the Vulkan gate's poison.
    m.fill_words(&outbuf, SENTINEL);
    // These two are both, and each has its own bound — see the module header.
    m.fill_words(&counters, COUNT_POISON);
    m.fill_words(&args, GRID_POISON);
    // The push constants are the one buffer the host legitimately writes.
    // `cs_seed` reads only `fill_count`, but the whole struct is written so an
    // uninitialised tail can never be what a future kernel reads.
    m.write_words(&push, &[n, 0, 0, 0]);

    // One argument buffer PER KERNEL. The measured per-entry-point ids
    // (`bind.rs`'s header) are the whole reason; sharing one would bind the
    // wrong pointers with no error anywhere.
    let mut ab_seed = p.seed.arg_buf(m)?;
    ab_seed.set_buffer("Push", &push)?;
    ab_seed.set_buffer("counters", &counters)?;

    // The plant binds `cs_prep`'s resources through `cs_seed`'s layout — the
    // hand-written-table failure, made concrete.
    let prep_src = if plant.seed_map_on_prep { &p.seed } else { &p.prep };
    let mut ab_prep = prep_src.arg_buf(m)?;
    if plant.swap_members {
        // Deliberately crossed: the `args` BUFFER into the `counters` MEMBER.
        // `set_buffer` already takes the member name and the buffer separately,
        // so the plant needs no special API — which is the shape a plant should
        // have: it exercises the real path with wrong arguments.
        ab_prep.set_buffer("counters", &args)?;
        ab_prep.set_buffer("args", &counters)?;
    } else {
        ab_prep.set_buffer("counters", &counters)?;
        ab_prep.set_buffer("args", &args)?;
    }

    let mut ab_fill = p.fill.arg_buf(m)?;
    ab_fill.set_buffer("counters", &counters)?;
    ab_fill.set_buffer("outbuf", &outbuf)?;

    let resident = ab_seed.resident_count() + ab_prep.resident_count() + ab_fill.resident_count();

    let one = MTLSize { width: 1, height: 1, depth: 1 };
    let tg = |k: &Kernel| MTLSize {
        width: k.local_size[0] as usize,
        height: k.local_size[1] as usize,
        depth: k.local_size[2] as usize,
    };
    // The derived group size, or the plant's 1. `cs_fill`'s own
    // `id.x < counters[1]` guard means an OVER-large group is invisible in the
    // readback, which is why the gate also asserts `local_size` directly.
    let fill_tg = if plant.threadgroup_one { one } else { tg(&p.fill) };

    // ONE encoder for the whole chain. Metal's default dispatch type is
    // `MTLDispatchTypeSerial`, so the three dispatches are ordered and
    // hazard-tracked by the driver — `vk/headless.rs`'s `compute_barrier`
    // apparatus has no analogue here rather than being omitted.
    m.compute(|enc| {
        let bind_ab = |enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
                       ab: &crate::mtl::bind::ArgBuf| {
            // The UNPLANTED run goes through `bind`, the shipping entry point,
            // rather than through the option form with neutral arguments —
            // otherwise the gate would exercise a path no caller uses.
            // `bind_opts` refuses an index override on a multi-set ArgBuf
            // since C3 — every kernel here is single-set (`smoke.hlsl` uses
            // only space0), so the `?` is the refusal travelling rather than a
            // path this gate can take.
            if plant.argbuf_index || plant.no_residency {
                let index = plant.argbuf_index.then_some(1);
                ab.bind_opts(enc, index, !plant.no_residency)
            } else {
                ab.bind(enc);
                Ok(())
            }
        };

        enc.setComputePipelineState(p.seed.pipeline());
        bind_ab(enc, &ab_seed)?;
        enc.dispatchThreadgroups_threadsPerThreadgroup(one, tg(&p.seed));

        enc.setComputePipelineState(p.prep.pipeline());
        bind_ab(enc, &ab_prep)?;
        enc.dispatchThreadgroups_threadsPerThreadgroup(one, tg(&p.prep));

        enc.setComputePipelineState(p.fill.pipeline());
        bind_ab(enc, &ab_fill)?;
        // THE INDIRECT DISPATCH — the whole point of the chain. The grid comes
        // from the buffer `cs_prep` just wrote; `threadsPerThreadgroup` still
        // comes from the host, which is the Metal-specific half.
        unsafe {
            enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &args, 0, fill_tg,
            );
        }
        Ok(())
    })?;

    // Outside the closure: `compute` blocks, and reading inside would race the
    // GPU with no diagnostic — unified memory makes a stale read a perfectly
    // valid read of the wrong moment.
    let c = m.read_words(&counters, 2);
    let a = m.read_words(&args, 3);
    Ok(Pass {
        out: m.read_words(&outbuf, words),
        args: [a[0], a[1], a[2]],
        counters: [c[0], c[1]],
        resident,
    })
}

/// The gate's own assertions — `vk/headless.rs:459-510`, verbatim in substance.
pub fn verify(p: &Pass, n: u32) -> Result<(), String> {
    let want_args = [n.div_ceil(64), u32::from(n > 0), 1];
    if p.args != want_args {
        return Err(format!("indirect args {:?}, expected {want_args:?}", p.args));
    }
    if p.counters != [n, n] {
        return Err(format!("counters {:?}, expected [{n}, {n}]", p.counters));
    }
    for i in 0..n {
        let want = i ^ 0x00C0_FFEE;
        if p.out[i as usize] != want {
            return Err(format!(
                "outbuf[{i}] = {:#x}, expected {want:#x}{}",
                p.out[i as usize],
                if p.out[i as usize] == SENTINEL { " (never written)" } else { "" }
            ));
        }
    }
    // The guard's own tooth: the last group covers words past the fill count,
    // so those threads ran and MUST have been rejected by `id.x < counters[1]`.
    for i in n..FILL_N + TAIL {
        if p.out[i as usize] != SENTINEL {
            return Err(format!(
                "outbuf[{i}] = {:#x} past the fill count — the bounds guard did not hold",
                p.out[i as usize]
            ));
        }
    }
    Ok(())
}

/// Both passes: the real count, and zero.
pub fn run(m: &Mtl, p: &Pipes, plant: Plant) -> Result<(usize, usize), String> {
    let a = pass(m, p, FILL_N, plant)?;
    verify(&a, FILL_N)?;

    // Zero work. `prep` clamps y to 0, so the indirect dispatch launches
    // nothing at all — the empty-queue case every level of the real ladder
    // hits, and the one an emulated indirect path gets wrong. Metal's
    // behaviour for a zero-group INDIRECT dispatch is not clearly documented,
    // so this is a measurement as much as a gate.
    let z = pass(m, p, 0, plant)?;
    verify(&z, 0)?;
    Ok((a.resident, z.resident))
}
