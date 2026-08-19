//! The Metal texture/sampler/multi-set probe: `cs_tex`, `cs_arr`, `cs_set1`.
//!
//! `smoke.rs`'s sibling and C3's proof, over `src/shaders/mtlbind.hlsl` — which
//! has been in the shared corpus since `ef359d6`, compiled by `--check-spirv`
//! and `--check-msl`, and **executed by nothing** until this module. That gap
//! is the one C3 exists to close: `--check-msl` asks "did it compile", and for
//! eight months the answer for `leaf` and `reference` was yes while `samp_lin`
//! was not bound at all (`msl::overlap_check`'s subject). A compile gate cannot
//! see that. This can.
//!
//! What it proves that `smoke.rs` cannot, one per entry point:
//!
//! * `cs_tex` — a TEXTURE and two SAMPLERS in set 0, and that the
//!   argument-buffer ids follow DECLARATION order on a subject where binding
//!   order would give a different answer.
//! * `cs_arr` — the tier-2 UNBOUNDED array, walked dynamically, alongside a
//!   second and third descriptor set. This is the arrangement
//!   `--msl-device-argument-buffer` and `--msl-argument-buffer-tier 2` exist
//!   for, and the one C2 explicitly did not reach.
//! * `cs_set1` — a kernel that references NOTHING in space0, so its argument
//!   buffers land at `[[buffer(1)]]` and `[[buffer(2)]]` with nothing at 0.
//!   C2's `const BUFFER_INDEX: usize = 0` fails here and nowhere else in the
//!   corpus, and so does any replacement written as a dense `0..n` scan.
//!
//! # The poisons, and the two bounds that DO NOT apply here
//!
//! `smoke.rs` poisons its four buffers with four different values, and its
//! header is careful that this is a SAFETY property: `args` is read by the
//! hardware as a threadgroup grid, and `counters[1]` is a bounds guard, so a
//! byte-uniform poison in either is a wedged WindowServer or an out-of-bounds
//! write rather than a failed assertion.
//!
//! **Neither hazard exists in this file, and that was checked rather than
//! assumed.** `outbuf` and `set1_out` are pure outputs; no kernel here reads a
//! count, no dispatch is indirect, and all three are `[1,1,1]` groups of
//! `[1,1,1]`. So one byte-uniform `SENTINEL` is correct for both buffers — and
//! it is deliberately the same value `smoke.rs` and `vk/headless.rs` use, so a
//! "never written" line reads identically across all three gates.
//!
//! The TEXTURES are where the equivalent care goes. Every texel is uploaded
//! before the dispatch, so an element that was never written reads whatever the
//! allocator handed back — which is why `DECOY` exists as a distinguishable
//! payload rather than the gate leaning on unwritten memory having a value.
//!
//! # Touch this →
//!
//! run `--check-mtl` clean AND each `FR_MTL_*` lever (the teeth must exit 1,
//! the two residency levers are reported), `--check-msl` on the procedural
//! scene AND san-miguel-low-poly AND `--sw-rays`, `--check-spirv`,
//! `--check-fsr3` and `--check-metalfx` (they share `mtl::device` and the
//! `objc2-metal` feature graph, which C3 widened by `MTLSampler`), `--check` +
//! `cargo test`, then restore the Windows goldens. Also run `--check-mtl` once
//! under `MTL_DEBUG_LAYER=1` and once under `MTL_SHADER_VALIDATION=1`,
//! SEPARATELY: C3 is the first Metal work here with textures, samplers and an
//! over-allocated argument buffer, and those are exactly what the validation
//! layers report and the data cannot.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLComputeCommandEncoder, MTLPixelFormat, MTLSamplerState, MTLSize, MTLStorageMode, MTLTexture,
};

use crate::gfx::shaders::{MTLBIND_PAYLOAD, MTLBIND_SRC_W, MTLBIND_TEX_N};
use crate::mtl::bind::{self, Kernel, LoadOpts};
use crate::mtl::device::Mtl;
use crate::mtl::msl::Msl;
use crate::spirv::Spirv;

/// `smoke::SENTINEL` and `vk/headless.rs:216`'s, byte-uniform, so all three
/// gates poison identically and a "never written" line reads the same.
pub const SENTINEL: u32 = 0xEEEE_EEEE;
/// The wrong-texture plant's payload. Plainly not any `MTLBIND_PAYLOAD | i` for
/// `i < MTLBIND_TEX_N`, and plainly not the sentinel — so a decoy read names
/// itself in the failure message instead of looking like an unwritten slot.
pub const DECOY: u32 = MTLBIND_PAYLOAD | 0xFF;

// The decoy has to be distinguishable from every real payload, or the K6
// texture tooth's observable is ambiguous with a correct read.
const _: () = assert!((DECOY & 0xFF) as usize >= MTLBIND_TEX_N);
// `cs_arr` writes `outbuf[0..TEX_N+2]` and `cs_tex` writes `outbuf[0..3]`; the
// allocation below must cover the larger.
const _: () = assert!(MTLBIND_TEX_N + 2 >= 3);

/// Words allocated for `outbuf` / `set1_out`, with room past what any kernel
/// writes so an over-running dispatch is observable rather than silent.
pub const OUT_WORDS: usize = MTLBIND_TEX_N + 2 + 8;

/// What `FR_MTL_ARR_STRIDE` multiplies the measured element size by.
///
/// A factor and not a literal byte count, so the plant stays wrong on the day
/// the real stride changes — a plant that hardcodes a value goes vacuous the
/// moment that value becomes correct.
///
/// **It is also what the argument buffer is SIZED for when the plant is
/// armed**, and that is the whole point of naming it once. A wrong stride into
/// a right-sized buffer is a wrong PLACEMENT — which is the defect being
/// planted — while a wrong stride into a buffer sized for the real one is an
/// out-of-bounds write, and Metal answers those by aborting under validation
/// instead of failing. See `Plant`.
const STRIDE_PLANT_MUL: usize = 2;

/// Levers, the `FR_ABL` read-only-probe idiom: default off, loud when armed.
///
/// **Six are TEETH and two are MEASUREMENTS**, the same asymmetry
/// `smoke::Plant` sets out: a tooth makes a claim about OUR code, so it must
/// fire; a measurement makes a claim about the PLATFORM, so it is reported.
/// Demanding that an unobservable omission fail would fail the gate on every
/// Apple-silicon Mac.
///
/// ```text
///  FR_MTL_SWAP_SAMP     K6  outbuf[0] and [1] swap, [2] UNMOVED
///  FR_MTL_TEX_DECOY     K6  all three words become 0xc300ff
///  FR_MTL_ARR_ROTATE    K7  texs[i] reads index (i+4)%5 — a wrong INDEX
///  FR_MTL_ARR_SHORT     K7  elements >= 1 never written
///  FR_MTL_ARR_STRIDE    K7  a computed stride instead of the reflected one
///  FR_MTL_BUFFER0       K8  structural: no argument buffer at [[buffer(0)]]
///  FR_MTL_NO_TEX        —   an unwritten texture descriptor: REPORTED
///  FR_MTL_ARR_NORESIDENT —  useResource: dropped for cs_arr: REPORTED
/// ```
///
/// **Every tooth must bite the same way with the validation layers ON**, and
/// `ARR_STRIDE` is why that is written down. Its first draft placed elements
/// with the doubled stride into a buffer sized for the real one, so the last
/// two writes ran past the allocation — invisible plain, and under
/// `MTL_DEBUG_LAYER=1` an ABORT (`setArgumentBuffer:startOffset:elementIndex:`,
/// exit 134) rather than the exit-1 the table above records. That is
/// `FR_MTL_BUFFER0`'s lesson arriving from the other direction: a plant that
/// cannot be told from a crash is not a tooth, and this module's run-list asks
/// for both the levers and the layers, so the two instructions have to be able
/// to hold at once. The buffer is sized for the planted stride below; the
/// refusal that would have caught it lives in `bind::set_texture_array`.
///
/// `SWAP_SAMP` and `TEX_DECOY` are a designed PAIR, and `mtlbind.hlsl:98-104`
/// is what makes them one: `cs_tex` reads one texture three ways, and the
/// `.Load` deliberately hits the SAME texel the clamp sample resolves to. So a
/// sampler swap moves words 0 and 1 and leaves word 2 alone, while a texture
/// that never bound moves all three — the two failures are distinguishable from
/// the readback, with no extra instrument.
///
/// `BUFFER0` is the strongest of the six for `smoke::Plant::seed_map_on_prep`'s
/// reason: it fails STRUCTURALLY, in the derivation, before any dispatch
/// happens at all.
///
/// **`FR_MTL_SET_SWAP` was designed and NOT taken.** Binding set 1's buffer at
/// set 2's index makes `cs_set1` read a texture-descriptor array as a
/// sampler-plus-pointer struct; that is a plausible device hang rather than a
/// command-buffer error, and a plant that can require a reboot is not a tooth.
/// `BUFFER0` already covers K8's claim, and covers it structurally.
#[derive(Clone, Copy, Default)]
pub struct Plant {
    /// Bind `samp_clamp`'s sampler into `samp_repeat`'s member and vice versa —
    /// what a binder routing by POSITION instead of by name produces.
    pub swap_samp: bool,
    /// Put the decoy texture in `src`'s slot.
    pub tex_decoy: bool,
    /// Never write `src` at all. PLATFORM behaviour, reported: an unwritten
    /// texture descriptor may read zero, may fault the command buffer, or may
    /// assert under shader validation, and which it is has not been measured.
    pub no_tex: bool,
    /// Write array element `i` at slot `(i + 1) % TEX_N`.
    pub arr_rotate: bool,
    /// Write only element 0 of the array.
    pub arr_short: bool,
    /// Place elements with a COMPUTED stride instead of the reflected one. The
    /// plant doubles the real stride rather than hardcoding 8, because a plant
    /// that hardcodes a value is vacuous on the day that value is correct.
    pub arr_stride: bool,
    /// Reinstate C2's `const BUFFER_INDEX: usize = 0`.
    pub buffer0: bool,
    /// Drop the `useResource:` replay for the whole `cs_arr` pass — **not for
    /// the array's elements alone**, which the name suggests and the one
    /// suppression point this module offers cannot do.
    ///
    /// `ArgBuf` deliberately has no per-resource residency switch: binding and
    /// declaring are one call, so "declared this one and not that one" is not
    /// expressible, and `bind_opts`'s flag drops the replay entire. Narrowing
    /// this lever would mean adding exactly the API that rule exists to
    /// withhold, which is a poor trade for a measurement.
    ///
    /// So it is `FR_MTL_NO_RESIDENCY` on a different subject rather than a
    /// finer one, and that is still worth having: `cs_arr` is the only pass
    /// that reaches TEXTURES through an argument buffer, and textures — not
    /// buffers — are the class the four `ffx_fsr3_metal.mm` GOTCHAs live in.
    /// The buffer answer does not carry over on its own.
    ///
    /// **MEASURED, and it changed the buffer answer too.** Apple M1, macOS
    /// 26.5.1, 2026-08-19, 3/3 reproducible each:
    ///
    /// ```text
    ///  plain                    unobservable — CHECK-MTL PASSED
    ///  MTL_DEBUG_LAYER=1        unobservable — CHECK-MTL PASSED
    ///  MTL_SHADER_VALIDATION=1  OBSERVABLE — K7 "outbuf[0] = 0xeeeeeeee
    ///                           (never written)": the dispatch wrote NOTHING
    /// ```
    ///
    /// Run against `FR_MTL_NO_RESIDENCY` the same day, the two arms agree —
    /// the buffer-only chain fails K4 under the same layer with its poison
    /// intact. So the layer, not the resource class, is what decides, and
    /// C2's recorded "bit-identical under every layer" is stale; `smoke::Plant`
    /// carries that correction.
    ///
    /// **Neither arm produces a command-buffer error.** `device::run` checks
    /// `cb.error()` and gets `None` both times; the writes are simply dropped.
    /// A gate that trusted the layer to say something would see nothing, and
    /// only the poison-and-read instrument catches it — which is the argument
    /// `smoke.rs`'s per-buffer poison makes, holding on a class it was not
    /// written for.
    pub arr_noresident: bool,
}

impl Plant {
    pub fn from_env() -> Plant {
        let on = |k: &str| std::env::var(k).is_ok_and(|v| v != "0");
        Plant {
            swap_samp: on("FR_MTL_SWAP_SAMP"),
            tex_decoy: on("FR_MTL_TEX_DECOY"),
            no_tex: on("FR_MTL_NO_TEX"),
            arr_rotate: on("FR_MTL_ARR_ROTATE"),
            arr_short: on("FR_MTL_ARR_SHORT"),
            arr_stride: on("FR_MTL_ARR_STRIDE"),
            buffer0: on("FR_MTL_BUFFER0"),
            arr_noresident: on("FR_MTL_ARR_NORESIDENT"),
        }
    }

    pub fn any(&self) -> bool {
        self.must_fail() || self.no_tex || self.arr_noresident
    }

    /// The six that are TEETH: each asserts something about this module's own
    /// code, so an armed run that PASSES is a defect in the gate.
    pub fn must_fail(&self) -> bool {
        self.swap_samp
            || self.tex_decoy
            || self.arr_rotate
            || self.arr_short
            || self.arr_stride
            || self.buffer0
    }

    pub fn line(&self) -> String {
        let mut v = Vec::new();
        for (on, name) in [
            (self.swap_samp, "FR_MTL_SWAP_SAMP"),
            (self.tex_decoy, "FR_MTL_TEX_DECOY"),
            (self.no_tex, "FR_MTL_NO_TEX"),
            (self.arr_rotate, "FR_MTL_ARR_ROTATE"),
            (self.arr_short, "FR_MTL_ARR_SHORT"),
            (self.arr_stride, "FR_MTL_ARR_STRIDE"),
            (self.buffer0, "FR_MTL_BUFFER0"),
            (self.arr_noresident, "FR_MTL_ARR_NORESIDENT"),
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
    pub tex: Kernel,
    pub arr: Kernel,
    pub set1: Kernel,
}

impl Pipes {
    pub fn all(&self) -> [(&'static str, &Kernel); 3] {
        [("cs_tex", &self.tex), ("cs_arr", &self.arr), ("cs_set1", &self.set1)]
    }
}

/// HLSL -> SPIR-V -> MSL -> metallib -> library -> pipeline, for all three
/// entry points, with one argument-buffer map derived per descriptor SET.
///
/// `plant.buffer0` is threaded through `load_opts` rather than read from the
/// environment here, so `bind.rs` stays free of `std::env` — the same division
/// `smoke.rs` keeps.
pub fn build(m: &Mtl, msl: &Msl, sp: &Spirv, plant: Plant) -> Result<Pipes, String> {
    let src = crate::gfx::shaders::MTLBIND_HLSL;
    let opts = LoadOpts { only_buffer_zero: plant.buffer0 };
    let mut built = Vec::new();
    for entry in ["cs_tex", "cs_arr", "cs_set1"] {
        let words = sp.compile(src, entry, "cs_6_5", &format!("mtlbind {entry}"), false)?;
        let msl_src = msl.transpile(&words, entry)?;
        let lib = msl.compile_lib(&msl_src, entry)?;
        built.push(bind::load_opts(m, &lib, &words, opts).map_err(|e| format!("{entry}: {e}"))?);
    }
    let mut it = built.into_iter();
    Ok(Pipes { tex: it.next().unwrap(), arr: it.next().unwrap(), set1: it.next().unwrap() })
}

/// The textures and samplers, built once and shared by all three passes.
///
/// Shared deliberately, `planes::Trio`'s argument in miniature: the three
/// entry points are only comparable if they provably saw the SAME bytes, and
/// two allocation sites agreeing is not a proof.
pub struct Res {
    /// `MTLBIND_SRC_W` x 1, texels `MTLBIND_PAYLOAD | x`.
    pub src: Retained<ProtocolObject<dyn MTLTexture>>,
    /// `MTLBIND_TEX_N` 1x1 textures, texel `MTLBIND_PAYLOAD | i`. Each is a
    /// single texel, so the trilinear and anisotropic samplers below are exact
    /// by construction — a lone texel has no neighbours to weight.
    pub texs: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    /// 1x1, texel `DECOY`. The wrong-texture plant's subject.
    pub decoy: Retained<ProtocolObject<dyn MTLTexture>>,
    pub samp_clamp: Retained<ProtocolObject<dyn MTLSamplerState>>,
    pub samp_repeat: Retained<ProtocolObject<dyn MTLSamplerState>>,
    pub samp_lin: Retained<ProtocolObject<dyn MTLSamplerState>>,
    pub samp_aniso: Retained<ProtocolObject<dyn MTLSamplerState>>,
}

/// One RGBA32Float texture whose `.x` carries each supplied payload.
///
/// RGBA32Float and not a uint format on purpose: the HLSL declares
/// `Texture2D<float4>`, and every payload is below 2^24, so the value survives
/// float storage and the `(uint)` cast in the shader EXACTLY. A readback
/// mismatch is a binding bug, never a precision one.
fn payload_tex(
    m: &Mtl,
    payloads: &[u32],
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let w = payloads.len();
    let t = m.texture(w, 1, MTLPixelFormat::RGBA32Float, MTLStorageMode::Shared)?;
    let mut bytes = Vec::with_capacity(w * 16);
    for &p in payloads {
        // `.x` is the payload; the other three channels are 0 and unread. The
        // cast is the inverse of the shader's `(uint)`, and both are exact
        // below 2^24.
        for v in [p as f32, 0.0, 0.0, 0.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    m.upload(&t, w, 1, w * 16, &bytes);
    Ok(t)
}

pub fn resources(m: &Mtl) -> Result<Res, String> {
    let src_payloads: Vec<u32> = (0..MTLBIND_SRC_W as u32).map(|x| MTLBIND_PAYLOAD | x).collect();
    let mut texs = Vec::with_capacity(MTLBIND_TEX_N);
    for i in 0..MTLBIND_TEX_N as u32 {
        texs.push(payload_tex(m, &[MTLBIND_PAYLOAD | i])?);
    }
    Ok(Res {
        src: payload_tex(m, &src_payloads)?,
        texs,
        decoy: payload_tex(m, &[DECOY])?,
        // NEAREST for the space0 pair — `mtlbind.hlsl`'s property 2 rests on
        // it. They differ ONLY in address mode.
        samp_clamp: m.sampler(true, false, 1)?,
        samp_repeat: m.sampler(true, true, 1)?,
        // The space1 pair is `trace_common.hlsli`'s, so it is the other two
        // backends' — trilinear, and hardware anisotropic at the same value
        // `vk/tracer.rs` and `gpu/trace.rs` feed theirs.
        samp_lin: m.sampler(false, true, 1)?,
        // `as u32` is `gpu/trace.rs:539`'s spelling, not a rounding choice of
        // this file's: `texture::max_aniso()` is an f32 and both D3D12 and
        // Vulkan take the same truncation to their own integer/float cap.
        samp_aniso: m.sampler(false, true, crate::texture::max_aniso() as u32)?,
    })
}

/// What one pass read back, plus the numbers the gate reports rather than
/// recomputes.
pub struct Pass {
    pub out: Vec<u32>,
    /// Resources declared through `useResource:`. Samplers are NOT counted —
    /// they are not `MTLResource`s (`bind::set_sampler`).
    pub resident: usize,
    /// The array member's reflected ARGUMENT-INDEX stride (ids per element),
    /// when this pass bound one. Not a byte stride — Metal reports 0 for that.
    pub stride: Option<usize>,
    /// The set-2 argument buffer's `encodedLength`, when this pass has one.
    pub encoded_len: Option<usize>,
}

/// `cs_tex` — set 0 only: one texture reached through two samplers and through
/// `.Load`.
pub fn pass_tex(m: &Mtl, p: &Pipes, r: &Res, plant: Plant) -> Result<Pass, String> {
    let outbuf = m.buffer(OUT_WORDS * 4)?;
    m.fill_words(&outbuf, SENTINEL);

    let mut ab = p.tex.arg_buf(m)?;
    ab.set_buffer("outbuf", &outbuf)?;
    if !plant.no_tex {
        ab.set_texture("src", if plant.tex_decoy { &r.decoy } else { &r.src })?;
    }
    // The plant crosses the two samplers. It needs no special API — it calls
    // the real path with the arguments swapped, which is the shape a plant
    // should have.
    let (c, rp) = if plant.swap_samp {
        (&r.samp_repeat, &r.samp_clamp)
    } else {
        (&r.samp_clamp, &r.samp_repeat)
    };
    ab.set_sampler("samp_clamp", c)?;
    ab.set_sampler("samp_repeat", rp)?;

    let resident = ab.resident_count();
    dispatch(m, &p.tex, &ab)?;
    Ok(Pass { out: m.read_words(&outbuf, OUT_WORDS), resident, stride: None, encoded_len: None })
}

/// How many elements' worth of argument buffer to allocate.
///
/// `MTLBIND_TEX_N` normally, and `STRIDE_PLANT_MUL` times that when the stride
/// plant is armed — so the planted placement lands INSIDE the allocation and
/// the tooth's observable is a wrong readback rather than an abort. Shared by
/// both array-bearing passes so the sizing and the override cannot drift, which
/// is how the first draft got it wrong: the override was computed per pass and
/// the allocation was not.
fn array_span(plant: Plant) -> usize {
    MTLBIND_TEX_N * if plant.arr_stride { STRIDE_PLANT_MUL } else { 1 }
}

/// The element size to place with: `None` for the real (reflected) one, or the
/// plant's multiple of it.
///
/// `None` and not `Some(encoded_len)` on the clean path deliberately — the
/// unplanted run goes through `set_texture_array`'s own measured default rather
/// than through a value this file recomputed, so the gate exercises the path a
/// caller takes. `smoke::pass`'s rule, applied to an argument instead of a
/// method.
fn stride_override(k: &Kernel, plant: Plant) -> Option<usize> {
    if !plant.arr_stride {
        return None;
    }
    k.map(2).ok().map(|m| m.encoded_len * STRIDE_PLANT_MUL)
}

/// `cs_arr` — all three sets, and the unbounded array walked dynamically.
pub fn pass_arr(m: &Mtl, p: &Pipes, r: &Res, plant: Plant) -> Result<Pass, String> {
    let outbuf = m.buffer(OUT_WORDS * 4)?;
    m.fill_words(&outbuf, SENTINEL);

    let mut ab = p.arr.arg_buf_n(m, array_span(plant))?;
    ab.set_buffer("outbuf", &outbuf)?;
    ab.set_sampler("samp_lin", &r.samp_lin)?;
    ab.set_sampler("samp_aniso", &r.samp_aniso)?;

    let (_, slot) = p.arr.find("texs")?;
    let stride = slot.array_ids;
    let over = stride_override(&p.arr, plant);
    let texs: Vec<_> = if plant.arr_rotate {
        // Element i written at slot (i+1) % N, so the readback is a WRONG
        // INDEX rather than garbage — the payload's low byte says which.
        (0..MTLBIND_TEX_N).map(|i| r.texs[(i + 1) % MTLBIND_TEX_N].clone()).collect()
    } else if plant.arr_short {
        r.texs[..1].to_vec()
    } else {
        r.texs.clone()
    };
    ab.set_texture_array("texs", &texs, over)?;

    let resident = ab.resident_count();
    let encoded_len = p.arr.map(2).ok().map(|m| m.encoded_len);
    dispatch_opts(m, &p.arr, &ab, !plant.arr_noresident)?;
    Ok(Pass { out: m.read_words(&outbuf, OUT_WORDS), resident, stride, encoded_len })
}

/// `cs_set1` — references nothing in space0, so its argument buffers are at
/// `[[buffer(1)]]` and `[[buffer(2)]]`.
pub fn pass_set1(m: &Mtl, p: &Pipes, r: &Res, plant: Plant) -> Result<Pass, String> {
    let out = m.buffer(OUT_WORDS * 4)?;
    m.fill_words(&out, SENTINEL);

    let mut ab = p.set1.arg_buf_n(m, array_span(plant))?;
    ab.set_buffer("set1_out", &out)?;
    ab.set_sampler("samp_lin", &r.samp_lin)?;
    let (_, slot) = p.set1.find("texs")?;
    let stride = slot.array_ids;
    let over = stride_override(&p.set1, plant);
    ab.set_texture_array("texs", &r.texs, over)?;

    let resident = ab.resident_count();
    let encoded_len = p.set1.map(2).ok().map(|m| m.encoded_len);
    dispatch(m, &p.set1, &ab)?;
    Ok(Pass { out: m.read_words(&out, OUT_WORDS), resident, stride, encoded_len })
}

fn dispatch(m: &Mtl, k: &Kernel, ab: &bind::ArgBuf) -> Result<(), String> {
    dispatch_opts(m, k, ab, true)
}

fn dispatch_opts(m: &Mtl, k: &Kernel, ab: &bind::ArgBuf, residency: bool) -> Result<(), String> {
    let one = MTLSize { width: 1, height: 1, depth: 1 };
    let tg = MTLSize {
        width: k.local_size[0] as usize,
        height: k.local_size[1] as usize,
        depth: k.local_size[2] as usize,
    };
    m.compute(|enc: &ProtocolObject<dyn MTLComputeCommandEncoder>| {
        enc.setComputePipelineState(k.pipeline());
        // The UNPLANTED run goes through `bind`, the shipping entry point,
        // rather than the option form with neutral arguments — otherwise the
        // gate would exercise a path no caller uses. `smoke::pass`'s rule.
        if residency {
            ab.bind(enc);
        } else {
            // `None` for the index: these ArgBufs are multi-set, and
            // `bind_opts` refuses an override on one — a single index names one
            // place to put three buffers. The residency flag is the only half
            // that means anything here.
            ab.bind_opts(enc, None, false)?;
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one, tg);
        Ok(())
    })
}

/// `outbuf[i]`'s expected value, or a message naming what it actually was.
///
/// Shared by the three verifiers so "never written" and "the decoy" are
/// diagnosed in one place — the payload's low byte is the index, so a wrong
/// element says which one it was.
fn want(out: &[u32], i: usize, want: u32, what: &str) -> Result<(), String> {
    let got = out[i];
    if got == want {
        return Ok(());
    }
    let why = if got == SENTINEL {
        " (never written)".to_string()
    } else if got == DECOY {
        " (the DECOY texture — a texture was bound into the wrong member)".to_string()
    } else if got & !0xFF == MTLBIND_PAYLOAD {
        format!(" (payload index {}, expected {})", got & 0xFF, want & 0xFF)
    } else {
        String::new()
    };
    Err(format!("{what}[{i}] = {got:#x}, expected {want:#x}{why}"))
}

/// `u = MTLBIND_SRC_U` on a width-`MTLBIND_SRC_W` texture: clamp resolves to
/// the last texel, repeat wraps to `frac(u) * w`.
///
/// DERIVED rather than written as literals, so the two constants and this
/// expectation cannot drift apart — and so a change to either is a compile-time
/// relationship rather than a gate that quietly checks the wrong texel.
fn tex_expect() -> (u32, u32) {
    let w = MTLBIND_SRC_W as f32;
    let clamp = (MTLBIND_SRC_W - 1) as u32;
    let repeat = (crate::gfx::shaders::MTLBIND_SRC_U.fract() * w) as u32;
    (MTLBIND_PAYLOAD | clamp, MTLBIND_PAYLOAD | repeat)
}

pub fn verify_tex(p: &Pass) -> Result<(), String> {
    let (clamp, repeat) = tex_expect();
    want(&p.out, 0, clamp, "outbuf")?;
    want(&p.out, 1, repeat, "outbuf")?;
    // The `.Load` reads the SAME texel the clamp sample resolves to,
    // deliberately: a sampler swap moves word 0 and leaves this one alone,
    // while a texture that never bound moves both.
    want(&p.out, 2, clamp, "outbuf")?;
    tail(&p.out, 3, "outbuf")
}

pub fn verify_arr(p: &Pass) -> Result<(), String> {
    for i in 0..MTLBIND_TEX_N {
        want(&p.out, i, MTLBIND_PAYLOAD | i as u32, "outbuf")?;
    }
    want(&p.out, MTLBIND_TEX_N, MTLBIND_PAYLOAD, "outbuf")?;
    want(&p.out, MTLBIND_TEX_N + 1, MTLBIND_PAYLOAD | 1, "outbuf")?;
    tail(&p.out, MTLBIND_TEX_N + 2, "outbuf")
}

pub fn verify_set1(p: &Pass) -> Result<(), String> {
    // `cs_set1` reads `texs[3]` twice — through `.Load` and through `samp_lin`.
    // Index 3 is inside the array and is not 0, so a short write, a rotation
    // and a stride error each move it.
    want(&p.out, 0, MTLBIND_PAYLOAD | 3, "set1_out")?;
    want(&p.out, 1, MTLBIND_PAYLOAD | 3, "set1_out")?;
    tail(&p.out, 2, "set1_out")
}

/// The words past what the kernel writes must still hold the poison. Its own
/// tooth: a dispatch that ran wider than its declared group, or a buffer bound
/// with the wrong length, shows up here and nowhere else.
fn tail(out: &[u32], from: usize, what: &str) -> Result<(), String> {
    for (i, &w) in out.iter().enumerate().skip(from) {
        if w != SENTINEL {
            return Err(format!(
                "{what}[{i}] = {w:#x} past what the kernel writes — the dispatch or the \
                 allocation overran"
            ));
        }
    }
    Ok(())
}
