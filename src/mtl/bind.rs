//! What a `.metallib` ASKS FOR, read back off the compiled function — and the
//! argument buffer that answers it.
//!
//! `msl.rs` answers "does the corpus compile"; this answers **"how do we bind
//! it"**, which its header (`msl.rs:7-8`) names as C2's whole subject.
//!
//! # Why this is derived, and why it CANNOT be a table
//!
//! Under the shipping `msl::CROSS_ARGS` — argument buffers on,
//! `--msl-decoration-binding` deliberately off — spirv-cross moves every
//! resource inside a per-descriptor-set struct as `[[id(n)]]`. **Those ids are
//! not the SPIR-V bindings.** They are dense, sequential, and assigned in
//! **DECLARATION** order — the module's `OpVariable` stream — over *only the
//! resources that entry point references*. MEASURED on `smoke.hlsl`, whose three
//! kernels share one file and disagree with each other:
//!
//! ```text
//!            [[id(0)]]              [[id(1)]]
//!  cs_seed   Push     (b1 -> 1)     counters (u0 -> 2000)
//!  cs_prep   counters (u0 -> 2000)  args     (u1 -> 2001)
//!  cs_fill   counters (u0 -> 2000)  outbuf   (u2 -> 2002)
//! ```
//!
//! `counters` is `id(1)` in one kernel and `id(0)` in the other two, and `args`
//! does not appear in `cs_fill` at all. So a hand-written table is not merely
//! bad practice the way `gpu/trace.rs::create_root_signature` is a liability —
//! it is **unrepresentable**: the map is per-ENTRY-POINT, and one argument
//! buffer shared across the three pipelines would bind the wrong pointers with
//! no error anywhere.
//!
//! (This corrects the measurement recorded at `msl.rs:38-47`. That
//! `[[id(0)]] [[id(1000)]] [[id(2000)]] [[id(3000)]]` layout is what you get
//! when `--msl-decoration-binding` IS passed. It is not the shipping arg set.)
//!
//! **C2 SAID "ASCENDING BINDING ORDER" AND IT WAS WRONG — corrected in C3.**
//! The table above is consistent with BOTH rules, because `smoke.hlsl` declares
//! `b1`, `u0`, `u1`, `u2` in that order: declaration order and binding order
//! coincide, and it was the only unit C2 bound. Measured on an FFX module whose
//! twelve resources are dense `[[id(0..11)]]`, cross-read with `spirv-dis`:
//!
//! ```text
//!   [[id]]   0   1   2   3   4   5   6     7   8   9  10  11
//!   binding  12  8   3   9  10   5  1001   7  11   2   0   1
//! ```
//!
//! Under the old rule `[[id(0)]]` would have to be binding 0. It is binding 12,
//! and binding 0 sits at `[[id(10)]]`. **Every unit that declares a `t` before a
//! `u` disagrees**, since `SHIFT_T` < `SHIFT_U` — so the old pin fired on correct
//! code the moment the subject stopped being `smoke.hlsl`. `reflect::Desc::decl`
//! carries the order the `(set, binding)` sort destroys; `cross_check` is its
//! only consumer.
//!
//! **AND THE NUMBERING IS PER SET — the same rule corrected twice.** The
//! ORDERING is module-wide (`decl` is the `OpVariable` ordinal over the whole
//! module) but each set struct restarts at `[[id(0)]]`. Measured on
//! `mtlbind.hlsl`, whose declaration order is `texs`(space2), `samp_lin`,
//! `samp_aniso`, `set1_out`(space1):
//!
//! ```text
//!  cs_tex   set0 { src 0, samp_clamp 1, samp_repeat 2, outbuf 3 }
//!  cs_arr   set0 { outbuf 0 }  set1 { samp_lin 0, samp_aniso 1 }  set2 { texs 0 }
//!  cs_set1                     set1 { samp_lin 0, set1_out 1 }    set2 { texs 0 }
//! ```
//!
//! A pin written over all descriptors at once expects one dense `0..n` and
//! fires on every one of those rows — correct on the one-set subject it was
//! written against, wrong the moment the subject gained a second set. That is
//! the ascending-binding mistake in a second currency, and it is why
//! `cross_check` filters to one set before ranking.
//!
//! # Two derivations, cross-checked — the M2c lesson in Metal's idiom
//!
//! `vk/reflect.rs:14-16` records that a planted binding shift PASSED
//! unvalidated, because the driver resolved it to the only slot there was.
//! Metal is worse off: there is no validation layer armed by default, and CI
//! runs the Metal job with them explicitly OFF (`ci.yml:637-664`). **So the DATA
//! is the only instrument**, and one derivation cannot check itself.
//!
//! * `derive` reads the map off the **compiled `MTLFunction`** — Metal's own
//!   answer, via `newArgumentEncoderWithBufferIndex:reflection:`.
//! * `crate::vk::reflect::reflect` reads the **same module's SPIR-V**
//!   independently, and already runs on macOS today (`--check-spirv`'s S0).
//! * `cross_check` requires them to agree.
//!
//! The join key is the resource NAME, and that it works is MEASURED rather than
//! hoped: spirv-cross's MSL member names are `Push` / `counters` / `args` /
//! `outbuf`, and the SPIR-V `OpName`s on the same variables are byte-identical.
//! No normalizer, and if that ever stops being true the gate says so instead of
//! quietly matching nothing.
//!
//! # The encoder writes the layout; nothing here assumes one
//!
//! `MTLArgumentEncoder::setBuffer:offset:atIndex:` places a pointer at whatever
//! offset the compiled function declares, so this module never computes
//! `id * 8` and never depends on Tier-2 argument buffers being raw pointers.
//! The deprecated `:reflection:` spelling is used ON PURPOSE and the reason is
//! objc2, not taste: the modern route is
//! `MTLComputePipelineReflection::bindings()` ->
//! `MTLBufferBinding::bufferStructType()`, but `MTLBufferBinding` is an
//! `extern_protocol!` and objc2 0.6 implements `DowncastTarget` for classes
//! only — a `ProtocolObject<dyn MTLBinding>` cannot be narrowed to it without
//! hand-rolled `msg_send!`. `newArgumentEncoderWithBufferIndex:reflection:`
//! hands back the concrete `MTLArgument` class and needs no downcast.
//!
//! **THE ONE PLACE THAT RULE HAS TO BEND IS THE UNBOUNDED ARRAY**, and C3
//! measured its way there through two wrong answers. `set_texture_array`'s doc
//! carries the detail; the shape is: `MTLArrayType::stride()` is **0** (a
//! texture entry has no byte stride), `setTexture:atIndex:(id + i)` silently
//! aliases every element onto element 0 (the reflected `arrayLength` is 1), and
//! what works is re-pointing the encoder's base by one element's BYTES — taken
//! from `encodedLength`, which is one element's worth only because `texs[]` is
//! alone in its set. Neither wrong answer errors; both answer.
//!
//! **AND ASKING METAL ABOUT AN INDEX IT DOES NOT USE ABORTS THE PROCESS.**
//! `newArgumentEncoderWithBufferIndex:reflection:` does not return nil for an
//! unused index — it fails an assertion and the process dies (measured, exit
//! 134). So `derive`'s scan is driven by the SPIR-V's set numbers and asks for
//! nothing else; the obvious "scan `0..30` and keep what answers" is not merely
//! imprecise, it is a crash. `smoke.hlsl` never exposed this because it uses
//! set 0 and the constant happened to be right.
//!
//! # Residency is STRUCTURAL, because nothing else could catch it
//!
//! A resource reached *through* an argument buffer is neither made resident nor
//! hazard-tracked by the encoder: `useResource:usage:` is mandatory and has no
//! Vulkan or D3D12 analogue, so no habit from either backend reminds you. It is
//! also the rule whose omission this gate cannot observe **on a plain run** —
//! on unified memory a non-heap `MTLBuffer` is already page-backed, so the
//! omission tends to pass, and the validation layers are off in CI.
//!
//! **It IS observable under `MTL_SHADER_VALIDATION=1`, on buffers and on
//! textures alike** — re-measured 2026-08-19 (M1, macOS 26.5.1, 3/3), which
//! overturns C2's recorded "bit-identical under every layer"; `smoke::Plant`
//! carries the correction and `texprobe::Plant` the texture arm. The
//! observable is that the dispatch's writes never land **with no
//! command-buffer error**, so it is the data that catches it and not the
//! layer's diagnostics. That makes the rule more dangerous rather than less:
//! it costs correctness only where nobody is looking, and it is silent where
//! everyone is. So `ArgBuf` exposes one mutator PER RESOURCE CLASS —
//! `set_buffer`, `set_texture`, `set_texture_array` — each of which records the
//! resource as it binds it: **"forgot `useResource`" is not expressible**,
//! rather than checked. That is `planes.rs`'s move — own the allocation and the
//! invariant stops being aspirational — applied to a rule that will turn fatal
//! the day these buffers move into an `MTLHeap`.
//!
//! **`set_sampler` is the one exception, and it is a different answer rather
//! than a missing one.** `MTLSamplerState` is not an `MTLResource`; it has no
//! backing allocation, so `useResource:` neither accepts it nor applies. That
//! is why `usage_for` returns `Option` and why `resident_count` is not the
//! member count — a reader tallying one against the other will otherwise read
//! the difference as exactly the hole this design closes.
//!
//! # What this deliberately does NOT cover
//!
//! Buffers, textures, samplers and the unbounded texture array, across as many
//! descriptor sets as a kernel declares. What is left is `DescKind::AccelStruct`
//! — C4's, and `usage_for` refuses it by name rather than defaulting
//! permissively, exactly as it refused a texture through C2.
//!
//! The C2 boundary this replaces read: *"`smoke.hlsl` declares no `t`, no `s`,
//! and nothing in `space1`, so C2 exercises the set-0 buffer half of
//! `CROSS_ARGS` and neither `--msl-device-argument-buffer` nor the tier-2
//! unbounded `texs[]` array those flags argue hardest for."* All four are
//! covered now, by `mtlbind.hlsl` under `--check-mtl`'s K6/K7/K8 — and the
//! value of that subject over a real tracer kernel is unchanged from C2's
//! argument: every failure has exactly one cause.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
// `MTLArgument` and the `:reflection:` encoder spelling carry Apple's
// `#[deprecated]`, pointing at `MTLBinding` / `newArgumentEncoderWithBufferBinding:`.
// That replacement is UNREACHABLE from objc2 0.6 — see the module header: the
// modern route ends at `MTLBufferBinding`, an `extern_protocol!`, and objc2
// implements `DowncastTarget` for classes only, so `ProtocolObject<dyn MTLBinding>`
// cannot be narrowed to it without hand-rolled `msg_send!`. The allow is
// therefore recording a binding-crate limitation, not a preference, and it
// should be revisited when objc2 grows protocol downcasting.
#[allow(deprecated)]
use objc2_metal::MTLArgument;
use objc2_metal::{
    MTLArgumentEncoder, MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice,
    MTLFunction, MTLLibrary, MTLResourceUsage, MTLSamplerState, MTLTexture,
};
use std::collections::BTreeMap;
use std::path::Path;

use crate::mtl::device::Mtl;
use crate::spirv::Reg;
use crate::vk::reflect::{self, DescKind};

/// One member of an argument-buffer struct, as the compiled function describes
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    /// The `[[id(n)]]`. Dense and per-entry-point — see the module header.
    pub id: u32,
    /// The MSL member name, which is also the SPIR-V `OpName` (measured).
    pub name: String,
    /// Byte offset inside the argument buffer. Reported, never computed.
    pub offset: usize,
    /// `Some(bytes)` when the member is an ARRAY, from
    /// `MTLStructMember::arrayType()` -> `MTLArrayType::stride()`.
    ///
    /// C3's whole reason for existing. spirv-cross lowers HLSL's unsized
    /// `texs[]` to `spvDescriptor<texture2d<float>> texs [[id(0)]][1]` — an
    /// array of ONE, because an unsized array cannot reserve the ids it will
    /// consume (`msl::overlap_check`'s subject). `encodedLength` therefore
    /// budgets one element, and the host must over-allocate and place element
    /// *i* itself. The MEASURED shader side is a plain contiguous walk:
    /// `spvDescriptorArray::operator[]` is `ptr[i]` over `&ptr_->value`. So the
    /// stride is a fact to be read off the compiled function, **never the
    /// literal 8** a `Retained` pointer's size would suggest.
    ///
    /// MEASURED: for `spvDescriptor<texture2d<float>>` this is **0**. Metal
    /// does not address an argument-buffer texture entry by byte offset, so
    /// `MTLArrayType::stride()` has nothing to report and says so. It is kept
    /// because it is the number that would matter for a POD array, and because
    /// a nonzero value here on a texture array would mean the layout changed
    /// under us — but `array_ids` is what places an element.
    pub array_stride: Option<usize>,
    /// `MTLArrayType::argumentIndexStride()` — how many `[[id]]`s one element
    /// consumes, which is how an argument-buffer array is ACTUALLY addressed.
    ///
    /// THIS IS THE NUMBER THAT PLACES ELEMENT *i*, at `id + i * array_ids`, and
    /// it is also spirv-cross's whole problem restated: its README says "arrays
    /// of resources consume multiple ids, where Vulkan does not", and an
    /// unsized array cannot reserve the ids it will consume — which is why
    /// `texs[]` had to move alone into space2 (`msl::overlap_check`). The
    /// reflected `arrayLength` is 1 (the unsized-array hack), so this is the
    /// stride of an array Metal believes has one element; the host
    /// over-allocates and keeps walking.
    pub array_ids: Option<usize>,
}

/// The argument-buffer layout of ONE (entry point, descriptor SET) pair.
///
/// Per-set, not per-kernel, since C3: a kernel referencing three spaces gets
/// three of these. Measured on `mtlbind.hlsl` — `cs_arr` takes
/// `spvDescriptorSetBuffer0/1/2` at `[[buffer(0/1/2)]]`, and `cs_set1` takes
/// only sets 1 and 2, with **nothing at `[[buffer(0)]]`**.
#[derive(Clone, Debug)]
pub struct ArgMap {
    /// The `[[buffer(n)]]` the argument buffer itself binds at — derived, and
    /// no longer the literal 0 it used to equal: `cs_set1` has none at 0.
    pub buffer_index: u32,
    /// Members, ascending by `id`. Ids are dense **within this set**, over the
    /// resources of this set that the entry point references.
    pub slots: Vec<Slot>,
    /// `MTLArgumentEncoder::encodedLength` — the argument buffer's own size.
    /// For a set holding only an unsized array this is ONE element's worth.
    pub encoded_len: usize,
}

impl ArgMap {
    pub fn slot(&self, name: &str) -> Option<&Slot> {
        self.slots.iter().find(|s| s.name == name)
    }

    /// One line per member, for `FR_MTL_MAP=1`. The `FR_VK_MAP` idiom: "what
    /// does this kernel actually bind" answerable without reading the shaders.
    pub fn lines(&self) -> Vec<String> {
        let mut v = vec![format!(
            "argument buffer at [[buffer({})]], {} B, {} slot(s)",
            self.buffer_index,
            self.encoded_len,
            self.slots.len()
        )];
        v.extend(self.slots.iter().map(|s| {
            format!(
                "  [[id({})]] @{:<4} {}{}",
                s.id,
                s.offset,
                s.name,
                match s.array_ids {
                    Some(n) => format!(" [] {n} id(s)/element, byte stride {:?}", s.array_stride),
                    None => String::new(),
                }
            )
        }));
        v
    }
}

/// A compiled entry point: the pipeline, the encoder that knows its layout, the
/// derived map, and the group size the host must supply at dispatch.
pub struct Kernel {
    pub name: String,
    /// One map per descriptor SET the entry point references, keyed by set.
    ///
    /// A `BTreeMap` and not a `Vec`, because the sets are neither dense nor
    /// zero-based: `cs_set1` has 1 and 2 and nothing at 0. A `Vec` indexed by
    /// position is the same mistake as `const BUFFER_INDEX: usize = 0` in a
    /// different spelling — it reads correct until the corpus has a kernel that
    /// skips a space, which is exactly the kernel the C3 probe added.
    pub maps: BTreeMap<u32, ArgMap>,
    /// What the module's own SPIR-V declares — the SECOND derivation, kept
    /// beside the first so `cross_check` and `usage_for` read one source and a
    /// caller cannot pair a map with another kernel's descriptors.
    pub descs: Vec<reflect::Desc>,
    /// `[numthreads]`, recovered from the SPIR-V. **Metal takes this from the
    /// CPU** — MSL carries no `[numthreads]`, and both `dispatchThreadgroups:`
    /// and the indirect form take it as an argument, so this is not a
    /// convenience but the only way the shape survives the crossing.
    pub local_size: [u32; 3],
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// One encoder per set, keyed the same way as `maps`. Each writes the
    /// layout ITS set declares, so the two must never be paired across sets.
    argencs: BTreeMap<u32, Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
}

impl Kernel {
    pub fn pipeline(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.pipeline
    }

    /// The map for one set, or a message naming the sets this kernel HAS —
    /// which is the diagnosis for the `cs_set1` shape, where asking for set 0
    /// is the bug rather than a lookup miss.
    pub fn map(&self, set: u32) -> Result<&ArgMap, String> {
        self.maps.get(&set).ok_or_else(|| {
            let have: Vec<u32> = self.maps.keys().copied().collect();
            format!("{} declares no descriptor set {set} — it has {have:?}", self.name)
        })
    }

    /// Every member of every set. The gate's anti-vacuity number: a reflection
    /// that came back empty builds empty argument buffers and reports clean.
    pub fn slots_total(&self) -> usize {
        self.maps.values().map(|m| m.slots.len()).sum()
    }

    /// The (set, slot) a member name lives at, searched across every set.
    ///
    /// Names are unique across the corpus's sets by construction — they are
    /// HLSL identifiers in one translation unit — so a hit is unambiguous, and
    /// a SECOND hit is a finding rather than a tie to break. Checked, not
    /// assumed: two sets declaring the same name would silently route half the
    /// writes to the wrong buffer.
    pub fn find(&self, name: &str) -> Result<(u32, &Slot), String> {
        let mut hits = self.maps.iter().filter_map(|(&set, m)| m.slot(name).map(|s| (set, s)));
        let first = hits.next().ok_or_else(|| {
            let all: Vec<&str> = self
                .maps
                .values()
                .flat_map(|m| m.slots.iter().map(|s| s.name.as_str()))
                .collect();
            format!("no argument-buffer member named `{name}` in {all:?}")
        })?;
        match hits.next() {
            None => Ok(first),
            Some((set2, _)) => Err(format!(
                "`{name}` is a member of BOTH set {} and set {set2} — the name join \
                 `cross_check` and this router both rely on is no longer unique",
                first.0
            )),
        }
    }

    /// The device's own ceiling for this pipeline. A `local_size` product above
    /// it cannot be dispatched at all, so it is worth saying which of the two
    /// is wrong.
    pub fn max_threads(&self) -> usize {
        self.pipeline.maxTotalThreadsPerThreadgroup()
    }

    /// A fresh argument buffer sized and laid out by THIS kernel's encoder.
    ///
    /// One per kernel in ordinary use — the measured per-entry-point ids in the
    /// module header are the whole reason — but calling this twice is legal and
    /// must stay so, because that is exactly what the `seed_map_on_prep` plant
    /// does. Note what is deliberately NOT here: the encoder is not pointed at
    /// the new buffer. `ArgBuf` shares ONE `MTLArgumentEncoder` with its kernel
    /// (`Retained::clone` is a retain, not a copy), so a retarget here would
    /// last only until the next `arg_buf` call and would make correctness a
    /// property of statement ORDER. `set_buffer` owns the retarget instead.
    pub fn arg_buf(&self, m: &Mtl) -> Result<ArgBuf, String> {
        self.arg_buf_n(m, 1)
    }

    /// `arg_buf`, sized for an unbounded array of `array_len` elements.
    ///
    /// THE OVER-ALLOCATION IS THE POINT, and it is C3's. spirv-cross lowers
    /// `texs[]` to an array of ONE (`Slot::array_stride`'s doc says why), so
    /// `encodedLength` budgets one element and a buffer sized by it alone would
    /// have the other four elements land past its end. The extra is
    /// `(array_len - 1) * stride` for every array member the set holds, with the
    /// stride READ OFF the compiled function.
    ///
    /// A separate entry point rather than a parameter on `arg_buf`, so
    /// `smoke.rs` — which has no arrays and must stay byte-comparable with the
    /// D3D12 and Vulkan smoke tests — is untouched.
    pub fn arg_buf_n(&self, m: &Mtl, array_len: usize) -> Result<ArgBuf, String> {
        let mut sets = BTreeMap::new();
        for (&set, map) in &self.maps {
            // THE EXTRA IS IN BYTES AND THE STRIDE IS IN IDS, so the byte size
            // of one element cannot come from `array_ids`. It comes from
            // `encodedLength` itself: `texs[]` is ALONE in space2 (the fix
            // `a19e385` took), so the whole set IS one element, and `n` of them
            // is `n * encoded_len`. K7 asserts that aloneness directly, so this
            // is a checked premise rather than an assumed one — and a set that
            // gained a second member would fail there before it under-allocated
            // here.
            let extra: usize = map
                .slots
                .iter()
                .filter(|s| s.array_ids.is_some())
                .map(|_| map.encoded_len * array_len.saturating_sub(1))
                .sum();
            let buf = m.buffer(map.encoded_len.max(1) + extra)?;
            let argenc = self
                .argencs
                .get(&set)
                .ok_or_else(|| format!("no argument encoder for set {set}"))?
                .clone();
            sets.insert(set, SetBuf { buf, argenc, map: map.clone() });
        }
        Ok(ArgBuf { sets, descs: self.descs.clone(), resident: Vec::new() })
    }
}

/// One set's buffer, the encoder that knows its layout, and that layout.
struct SetBuf {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    argenc: Retained<ProtocolObject<dyn MTLArgumentEncoder>>,
    map: ArgMap,
}

/// A resource reached THROUGH an argument buffer, which Metal will not make
/// resident on its own.
///
/// An enum and not a `dyn MTLResource`, because `useResource:` takes the
/// protocol object and the two source types coerce to it differently — and
/// because a third variant arriving (an acceleration structure, C4's) should be
/// a compile error here rather than a silently un-declared resource.
enum Resident {
    Buffer(Retained<ProtocolObject<dyn MTLBuffer>>),
    Texture(Retained<ProtocolObject<dyn MTLTexture>>),
}

/// Every set's filled argument buffer, and the ONE residency list they built as
/// they filled.
///
/// Multi-set since C3, and the sets live INSIDE one `ArgBuf` rather than the
/// caller holding a `Vec<ArgBuf>`: residency is a property of the dispatch, not
/// of a set, so one list is the honest shape — and it is what keeps `smoke.rs`'s
/// call sites and the K3 isolation arm reading exactly as they did.
pub struct ArgBuf {
    sets: BTreeMap<u32, SetBuf>,
    descs: Vec<reflect::Desc>,
    resident: Vec<(Resident, MTLResourceUsage)>,
}

impl ArgBuf {
    /// The (set, slot) a member name lives at — `Kernel::find`'s twin over the
    /// buffers actually allocated, with the same uniqueness requirement.
    fn find(&self, name: &str) -> Result<(u32, Slot), String> {
        let mut hits =
            self.sets.iter().filter_map(|(&set, sb)| sb.map.slot(name).map(|s| (set, s.clone())));
        let first = hits.next().ok_or_else(|| {
            // WORDING IS LOAD-BEARING: `FR_MTL_SEED_MAP`'s recorded observable
            // is this exact sentence (`smoke.rs:105`), so it may gain context
            // but must not lose its head.
            let all: Vec<&str> = self
                .sets
                .values()
                .flat_map(|sb| sb.map.slots.iter().map(|s| s.name.as_str()))
                .collect();
            format!("no argument-buffer member named `{name}` in {all:?}")
        })?;
        match hits.next() {
            None => Ok(first),
            Some((set2, _)) => {
                Err(format!("`{name}` is a member of BOTH set {} and set {set2}", first.0))
            }
        }
    }
    /// Write a buffer into the member NAMED, and record it as resident.
    ///
    /// The only mutator, deliberately. Binding and declaring residency are one
    /// call because they are one obligation, and separating them would make
    /// "bound but not resident" expressible — a state that is fatal on a heap,
    /// invisible here, and reported by no instrument this gate has.
    ///
    /// The usage is DERIVED from the register class rather than passed: a `b#`
    /// is a constant buffer (`Read`), a `u#` is an unordered-access buffer
    /// (`Read | Write`). A per-call-site usage argument is a per-call-site
    /// opportunity to under-declare one.
    pub fn set_buffer(
        &mut self,
        name: &str,
        buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    ) -> Result<(), String> {
        let (set, slot) = self.find(name)?;
        let usage = usage_for(name, &self.descs)?.ok_or_else(|| {
            format!("`{name}` is a sampler, so it must be bound with set_sampler")
        })?;
        let sb = &self.sets[&set];
        // THE RETARGET IS PER WRITE, and that is what makes an `ArgBuf`
        // self-contained rather than order-dependent. The encoder is shared
        // with the kernel and with every other `ArgBuf` it created, and
        // `setArgumentBuffer:offset:` is what selects the destination — so
        // pointing it once at construction means a LATER `arg_buf` on the same
        // kernel silently redirects THIS one's remaining writes into the new
        // buffer, while `bind` keeps using the right one and nothing looks
        // wrong. `pass`'s seed-map plant creates two, and only the order of two
        // statements kept that correct. Re-pointing here is the documented
        // per-buffer usage pattern and costs one message send.
        unsafe {
            sb.argenc.setArgumentBuffer_offset(Some(&sb.buf), 0);
            sb.argenc.setBuffer_offset_atIndex(Some(buf), 0, slot.id as usize);
        }
        self.resident.push((Resident::Buffer(buf.clone()), usage));
        Ok(())
    }

    /// Write a TEXTURE into the member named, and record it as resident.
    ///
    /// `set_buffer`'s twin in every respect that matters — one call for one
    /// obligation, usage derived from the descriptor kind and never passed —
    /// and separate only because `MTLArgumentEncoder` has a separate setter.
    pub fn set_texture(
        &mut self,
        name: &str,
        tex: &Retained<ProtocolObject<dyn MTLTexture>>,
    ) -> Result<(), String> {
        let (set, slot) = self.find(name)?;
        let usage = usage_for(name, &self.descs)?
            .ok_or_else(|| format!("`{name}` is a sampler, not a texture"))?;
        if slot.array_stride.is_some() {
            return Err(format!(
                "`{name}` is an ARRAY member — use set_texture_array, or only element 0 \
                 would ever be written"
            ));
        }
        let sb = &self.sets[&set];
        unsafe {
            sb.argenc.setArgumentBuffer_offset(Some(&sb.buf), 0);
            sb.argenc.setTexture_atIndex(Some(tex), slot.id as usize);
        }
        self.resident.push((Resident::Texture(tex.clone()), usage));
        Ok(())
    }

    /// Write a SAMPLER into the member named. **Nothing is made resident, and
    /// that is not an omission.**
    ///
    /// `MTLSamplerState` is not an `MTLResource` — it has no backing allocation
    /// to page in — so `useResource:` neither accepts it nor applies to it. This
    /// is the one place the module's "binding and declaring residency are one
    /// call" rule has a different answer rather than a missing one, and it is
    /// stated here because a reader counting residency entries against members
    /// will otherwise read the difference as exactly the hole that rule exists
    /// to close. `usage_for` returns `None` for a sampler for the same reason,
    /// rather than a `Read` that would be a lie about what was declared.
    pub fn set_sampler(
        &mut self,
        name: &str,
        samp: &Retained<ProtocolObject<dyn MTLSamplerState>>,
    ) -> Result<(), String> {
        let (set, slot) = self.find(name)?;
        if usage_for(name, &self.descs)?.is_some() {
            return Err(format!("`{name}` is not a sampler in the SPIR-V"));
        }
        // The same refusal `set_texture` makes, for the same reason and not
        // because the corpus has a sampler array — it has none. An array
        // member written through the scalar setter takes element 0's write and
        // silently leaves the rest unbound, and the point of stating that in
        // `set_texture` was the shape of the mistake, not the resource class.
        // A sampler array arriving with only this guard missing would be C3's
        // texture-array lesson relearned on the one class that skipped it.
        if slot.array_ids.is_some() {
            return Err(format!(
                "`{name}` is an ARRAY member — only element 0 would be written, and there \
                 is no set_sampler_array (no sampler array exists in the corpus; adding \
                 one means porting set_texture_array's measured placement)"
            ));
        }
        let sb = &self.sets[&set];
        unsafe {
            sb.argenc.setArgumentBuffer_offset(Some(&sb.buf), 0);
            sb.argenc.setSamplerState_atIndex(Some(samp), slot.id as usize);
        }
        Ok(())
    }

    /// Write EVERY element of an unbounded texture array, and record each one
    /// resident.
    ///
    /// THE ONLY MUTATOR THAT PLACES MORE THAN ONE THING, and the stride is
    /// measured rather than assumed — it is also the one number this milestone
    /// got wrong on its first run, which is why the wrong one is named here.
    ///
    /// **It is an ARGUMENT-INDEX stride, not a byte stride.** The obvious
    /// reading of spirv-cross's output says otherwise: it emits
    /// `spvDescriptor<texture2d<float>> texs [[id(0)]][1]` plus an
    /// `spvDescriptorArray` whose `operator[]` is `ptr[i]` over `&ptr_->value`,
    /// which looks like a contiguous byte walk one should follow by re-pointing
    /// the encoder's base by `i * MTLArrayType::stride()`. **`stride()` is 0.**
    /// Metal does not address an argument-buffer texture entry by byte offset
    /// at all, so it has no byte stride to report, and a host that trusted the
    /// obvious reading would place all five elements on element 0 and read back
    /// "the last one wins" — a wrong ANSWER, not a failure.
    ///
    /// `MTLArrayType::argumentIndexStride()` is the number that exists: element
    /// *i* is at `[[id(slot.id + i * n)]]`. That is spirv-cross's own README
    /// rule seen from the other side — "arrays of resources consume multiple
    /// ids, where Vulkan does not" — and it is why an UNSIZED array, which
    /// cannot reserve the ids it will consume, had to move alone into space2.
    ///
    /// **AND THE ARGUMENT INDEX IS NOT THE ANSWER EITHER — MEASURED.** Writing
    /// element *i* at `[[id(slot.id + i * n)]]` was the obvious next reading
    /// once `stride()` turned out to be 0, and it is what this file did on its
    /// second run. The encoder's reflected `arrayLength` is **1**, so every
    /// index past `slot.id` is past what it was told it owns: all five writes
    /// aliased onto element 0 and the readback was the LAST texture written,
    /// five times (`outbuf[0] = 0xc30004`). It does not error — it answers
    /// wrongly, which is the class this whole module is built against.
    ///
    /// WHAT WORKS is re-pointing the encoder's BASE by one element's BYTES and
    /// writing at `slot.id` every time, so the encoder only ever does the one
    /// thing it has agreed to do. The element size is `encodedLength` — not a
    /// literal, and not `MTLArrayType::stride()`, which is 0 — and it is that
    /// only because `texs[]` is **ALONE in its set**, the arrangement
    /// `a19e385` was forced into by the overlapping-binding rule. A set with a
    /// second member has no element size to offer, so this REFUSES rather than
    /// guessing; K7 asserts the aloneness directly, so the premise is checked
    /// on every run rather than assumed here.
    ///
    /// **AND THE LAST ELEMENT HAS TO FIT IN THE ALLOCATION, which is checked
    /// here rather than left to Metal — because Metal's answer is an abort.**
    /// The body says what was measured. `arg_buf_n` is what sizes the buffer,
    /// and a `stride_override` is exactly the way to hand this method a
    /// placement the allocation was never sized for.
    pub fn set_texture_array(
        &mut self,
        name: &str,
        texs: &[Retained<ProtocolObject<dyn MTLTexture>>],
        stride_override: Option<usize>,
    ) -> Result<(), String> {
        let (set, slot) = self.find(name)?;
        let usage = usage_for(name, &self.descs)?
            .ok_or_else(|| format!("`{name}` is a sampler, not a texture array"))?;
        let ids = slot.array_ids.ok_or_else(|| {
            format!(
                "`{name}` is not an array member — Metal describes it as a scalar, so there \
                 is no stride to place elements with"
            )
        })?;
        // A zero stride would land every element on element 0. Refused
        // structurally, the `COUNT_POISON` discipline in another currency —
        // and this is not hypothetical: `MTLArrayType::stride()` reports
        // exactly 0 here, which is what sent the first draft down a wrong road.
        if ids == 0 {
            return Err(format!("`{name}` consumes 0 argument-buffer ids per element"));
        }
        let sb = &self.sets[&set];
        if sb.map.slots.len() != 1 {
            return Err(format!(
                "`{name}` shares set {set} with {} other member(s), so `encodedLength` is not \
                 one element's size and there is no measured way to place element i. An \
                 unsized array must be ALONE in its space — see msl::overlap_check",
                sb.map.slots.len() - 1
            ));
        }
        let elem = stride_override.unwrap_or(sb.map.encoded_len);
        if elem == 0 {
            return Err(format!("`{name}`'s set encodes 0 B — no element size"));
        }
        // THE LAST ELEMENT MUST FIT, AND THIS IS A REFUSAL RATHER THAN A
        // METAL ASSERTION — measured, on this file's own plant. Metal validates
        // `setArgumentBuffer:offset:` against the buffer's length and ABORTS
        // when it does not fit:
        //
        //   -[MTLDebugArgumentEncoder setArgumentBuffer:startOffset:
        //     elementIndex:]:409: failed assertion `Argument Buffer Validation
        //     offset (48) + encodedLength (8) should be smaller or equal to
        //     the buffer length (40)'
        //
        // — which is `newArgumentEncoderWithBufferIndex:`'s lesson a second
        // time: the API answers a placement it cannot honour by killing the
        // process, so the host has to be the one that says no. It also only
        // fires under MTL_DEBUG_LAYER; with the layer off the same write
        // scribbles past the requested length and the run looks fine, which is
        // the worse half. A caller reaching here has either forgotten
        // `arg_buf_n` or passed a `stride_override` the allocation was not
        // sized for, and both are worth naming.
        let span = (texs.len().saturating_sub(1)) * elem + sb.map.encoded_len;
        let have = sb.buf.length();
        if span > have {
            let need = span.div_ceil(sb.map.encoded_len.max(1));
            return Err(format!(
                "`{name}`: {} element(s) at {elem} B each need {span} B, but set {set}'s \
                 argument buffer is {have} B — allocate it with `arg_buf_n(m, {need})`. \
                 Metal aborts on this under validation and writes past the allocation \
                 without it, so it is refused here",
                texs.len()
            ));
        }
        for (i, t) in texs.iter().enumerate() {
            unsafe {
                sb.argenc.setArgumentBuffer_offset(Some(&sb.buf), i * elem);
                sb.argenc.setTexture_atIndex(Some(t), slot.id as usize);
            }
            self.resident.push((Resident::Texture(t.clone()), usage));
        }
        Ok(())
    }

    /// Bind every set's argument buffer at its derived index and declare every
    /// resource reached through them.
    pub fn bind(&self, enc: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        self.bind_at(enc, None, true);
    }

    /// `bind`, with the two things a plant needs to get wrong.
    ///
    /// `index` overrides the DERIVED bind point; `residency` drops the
    /// `useResource:` replay. Both exist so the gate can make itself fail on
    /// demand — the `FR_ABL` read-only-probe idiom — and neither is reachable
    /// from `bind`, so an ordinary caller cannot take either by accident.
    ///
    /// **`index` NAMES ONE BIND POINT, so it is refused on a multi-set
    /// `ArgBuf`** rather than applied to each set in turn. Applying it would
    /// stack every set's buffer on the same index, so the last one written
    /// wins and the other sets are simply absent — a plant that mis-binds
    /// three buffers in a way no real defect produces, and worse, one whose
    /// failure would be read as evidence about the override it was testing.
    /// Metal will not object: `setBuffer:offset:atIndex:` at a repeated index
    /// is a legal overwrite. This is the only caller-facing way to get the
    /// sets and their bind points out of step, so it is the place to say no.
    pub fn bind_opts(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        index: Option<u32>,
        residency: bool,
    ) -> Result<(), String> {
        if let Some(at) = index {
            if self.sets.len() != 1 {
                let sets: Vec<u32> = self.sets.keys().copied().collect();
                return Err(format!(
                    "bind_opts cannot override the bind point of a multi-set ArgBuf — this \
                     one holds sets {sets:?} and index {at} names one place to put them all"
                ));
            }
        }
        self.bind_at(enc, index, residency);
        Ok(())
    }

    /// The binding itself, infallible, so `bind` — which passes no override —
    /// does not have to carry a `Result` it can never produce.
    fn bind_at(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        index: Option<u32>,
        residency: bool,
    ) {
        for sb in self.sets.values() {
            let at = index.unwrap_or(sb.map.buffer_index) as usize;
            unsafe { enc.setBuffer_offset_atIndex(Some(&sb.buf), 0, at) };
        }
        if residency {
            for (r, usage) in &self.resident {
                let res = match r {
                    Resident::Buffer(b) => ProtocolObject::from_ref(&**b),
                    Resident::Texture(t) => ProtocolObject::from_ref(&**t),
                };
                enc.useResource_usage(res, *usage);
            }
        }
    }

    /// The residency list, for the gate's anti-vacuity check — a bind that
    /// declared nothing would otherwise look identical to one that declared
    /// everything.
    ///
    /// **Samplers do not appear here**, so this is not the member count. See
    /// `set_sampler`.
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// The argument buffer's own encoded words.
    ///
    /// Exists for `resident_count`'s reason: an invariant nothing can observe
    /// is an invariant that regresses silently. The one here is that two
    /// `ArgBuf`s from ONE kernel encode into their OWN buffers — which holds
    /// only because `set_buffer` re-points the shared encoder per write, and
    /// which ordinary use would never exercise, since the only path that
    /// creates two is a plant that fails structurally first.
    ///
    /// Deliberately words rather than a decoded address: this module does not
    /// assume tier-2 argument buffers are raw pointers, and a gate that
    /// compared `gpuAddress()` against these bytes would be asserting exactly
    /// that.
    ///
    /// Set-keyed since C3, since there is a buffer per set to read.
    pub fn encoded_words(&self, m: &Mtl, set: u32) -> Vec<u32> {
        match self.sets.get(&set) {
            Some(sb) => m.read_words(&sb.buf, sb.map.encoded_len / 4),
            None => Vec::new(),
        }
    }
}

/// How a resource reached through an argument buffer must be declared to
/// `useResource:` — or `None` when it is not an `MTLResource` at all.
///
/// DERIVED from the descriptor kind the SPIR-V declares, never from the
/// resource's name — a first draft matched `"Push"` and worked only because
/// `smoke.hlsl`'s constant buffer happens to be called that, which is the same
/// class of mistake as hardcoding an `[[id(n)]]`.
///
/// Not a caller-supplied argument either: a per-call-site usage is a
/// per-call-site opportunity to under-declare one, and an under-declared
/// resource is exactly the failure `useResource:` exists to prevent.
///
/// **`None` is a distinct answer, not "no usage".** A sampler has no backing
/// allocation, so `useResource:` neither accepts it nor means anything for it.
/// Spelling that as `Read` would put a lie in the residency list and make
/// `resident_count` — the gate's anti-vacuity number — count a thing that was
/// never declared.
///
/// **`Sample` is deliberately absent from the texture arms.** Apple deprecated
/// `MTLResourceUsageSample` (it carries `#[deprecated]` in `objc2-metal`) and
/// folded sampling into `Read`; asking for it would be an under-documented flag
/// on every textured dispatch for no coverage.
fn usage_for(name: &str, descs: &[reflect::Desc]) -> Result<Option<MTLResourceUsage>, String> {
    let d = descs
        .iter()
        .find(|d| d.name == name)
        .ok_or_else(|| format!("`{name}` is in the argument buffer but not in the SPIR-V"))?;
    match d.kind {
        DescKind::UniformBuffer => Ok(Some(MTLResourceUsage::Read)),
        DescKind::StorageBuffer => Ok(Some(MTLResourceUsage::Read | MTLResourceUsage::Write)),
        DescKind::SampledImage => Ok(Some(MTLResourceUsage::Read)),
        DescKind::StorageImage => Ok(Some(MTLResourceUsage::Read | MTLResourceUsage::Write)),
        DescKind::Sampler => Ok(None),
        // Still exhaustive-by-refusal rather than a permissive default. C3
        // binds buffers, textures and samplers; an acceleration structure
        // arriving here means C4's work has started without its residency
        // question being answered, exactly as a texture meant C3's had.
        k => Err(format!(
            "`{name}` is a {k:?}, which mtl::bind does not bind yet \
             (C3 is buffers, textures and samplers)"
        )),
    }
}

/// Load a `.metallib`, build the pipeline for its ONE entry point, and derive
/// the argument-buffer map.
///
/// `words` is the SPIR-V the metallib was built from — the second derivation's
/// input, and where the group size comes from.
pub fn load(m: &Mtl, lib_path: &Path, words: &[u32]) -> Result<Kernel, String> {
    load_opts(m, lib_path, words, LoadOpts::default())
}

/// `load`, with the one thing a plant needs to get wrong.
///
/// `only_buffer_zero` reinstates C2's `const BUFFER_INDEX: usize = 0` — look
/// for the argument buffer at index 0 and nowhere else. On `cs_set1`, whose
/// buffers are at 1 and 2, that fails STRUCTURALLY before any dispatch, which
/// is the strongest shape a tooth here can have (`smoke.rs`'s `SEED_MAP`
/// argument). It is not reachable from `load`, so an ordinary caller cannot
/// take it by accident.
///
/// **IT SIMULATES A NIL METAL DOES NOT RETURN, and that is deliberate.** See
/// `derive`: the real call ABORTS the process on an index that is not an
/// argument buffer, so a plant that made it would produce a SIGABRT rather
/// than a gate failure — and a plant that cannot be told from a crash is not a
/// tooth. The defect being planted is the host's ("it assumes set 0"), and its
/// consequence is faithfully reproduced; only the crash is not.
#[derive(Clone, Copy, Default)]
pub struct LoadOpts {
    pub only_buffer_zero: bool,
}

pub fn load_opts(
    m: &Mtl,
    lib_path: &Path,
    words: &[u32],
    opts: LoadOpts,
) -> Result<Kernel, String> {
    let dev = m.device();

    let path = lib_path.to_str().ok_or("metallib path is not UTF-8")?;
    let url = NSURL::fileURLWithPath(&NSString::from_str(path));
    let lib = dev
        .newLibraryWithURL_error(&url)
        .map_err(|e| format!("newLibraryWithURL({path}): {}", e.localizedDescription()))?;

    // THE ENTRY NAME IS DERIVED. Hardcoding it is the same mistake as
    // hardcoding an id in a different register — and `ffx_fsr3_metal.mm:630`
    // hardcodes `main0`, which is right for FFX (whose SPIR-V entry is `main`)
    // and would be wrong here. Exactly one, because `corpus_jobs` compiles one
    // entry per module; more would mean this is not the module we think.
    let names = lib.functionNames();
    if names.len() != 1 {
        let all: Vec<String> = names.iter().map(|n| n.to_string()).collect();
        return Err(format!(
            "metallib declares {} entry points {all:?}, expected exactly 1",
            names.len()
        ));
    }
    let fname = names.objectAtIndex(0);
    let func = lib
        .newFunctionWithName(&fname)
        .ok_or_else(|| format!("newFunctionWithName({fname}) returned nil"))?;

    // Pipeline first: the argument encoder is a property of the FUNCTION, but a
    // function that cannot become a pipeline is a failure worth reporting
    // before a map that would describe it.
    let pipeline = dev
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| format!("pipeline for {fname}: {}", e.localizedDescription()))?;

    let local_size = crate::spirv::local_size(words)?;
    // Reflected BEFORE `derive` since C3, and no longer only so the map and the
    // descriptors provably come from one module: the SPIR-V's set numbers are
    // what the scan is cross-checked against.
    let descs = reflect::reflect(words).map_err(|e| format!("SPIR-V reflection: {e}"))?;
    // One encoder per set, each carrying both halves of its derivation.
    let (maps, argencs) = derive(&func, &descs, opts)?;

    Ok(Kernel { name: fname.to_string(), maps, descs, local_size, pipeline, argencs })
}

/// Read the argument-buffer layout off the compiled function — the layout AND
/// the encoder that writes it, out of ONE object.
///
/// `newArgumentEncoderWithBufferIndex:reflection:` hands back a writer together
/// with a description of what it writes, so these are one call rather than
/// three. A first draft made three — a probe, a describer, and a writer, each
/// minting an encoder and two of them dropping it — which was both wasteful and
/// three chances for the map and the writer to come from different objects.
///
/// WHICH `[[buffer(n)]]`, and this is C3's correction to C2. It used to be
/// `const BUFFER_INDEX: usize = 0`, with a note that "a set-1 corpus turns this
/// into a scan". It is that scan now, and the constant was not merely
/// unambitious — it was WRONG for a shape the corpus now contains: `cs_set1`
/// references nothing in `space0`, so its argument buffers are at
/// `[[buffer(1)]]` and `[[buffer(2)]]` and index 0 reflects **nil**.
///
/// THE SCAN IS DRIVEN BY THE SPIR-V'S SETS, not by counting up until something
/// answers. A dense `0..n` loop finds two buffers for `cs_set1` — the ones at 1
/// and 2 — and would look perfectly correct while proving nothing about which
/// index goes with which set. Asking Metal for exactly the set numbers the
/// module declares, and requiring each to answer, is the same two-derivations
/// discipline `cross_check` embodies:
///
/// * the SPIR-V says which sets exist,
/// * Metal says what is at each of those indices,
/// * and a set that reflects nil is a FAIL that names the set.
///
/// The reverse direction — an index Metal answers at that the SPIR-V never
/// declared — is `cross_check`'s `d.set != metal.buffer_index` arm, which C3
/// promotes from an incidental check to the pin behind the whole `[[buffer(n)]]
/// == set n` claim.
#[allow(deprecated)]
fn derive(
    func: &ProtocolObject<dyn MTLFunction>,
    descs: &[reflect::Desc],
    opts: LoadOpts,
) -> Result<
    (BTreeMap<u32, ArgMap>, BTreeMap<u32, Retained<ProtocolObject<dyn MTLArgumentEncoder>>>),
    String,
> {
    // The sets the module itself declares — deduplicated and ascending.
    let mut want: Vec<u32> = descs.iter().map(|d| d.set).collect();
    want.sort_unstable();
    want.dedup();
    if want.is_empty() {
        return Err("the SPIR-V declares no descriptors at all — nothing to derive".into());
    }
    // The plant: look only at 0, exactly as C2 did. Intersected with the sets
    // that EXIST rather than replacing them, because the aborting call is not
    // usable as a tooth — see `LoadOpts`.
    if opts.only_buffer_zero {
        if !want.contains(&0) {
            return Err(format!(
                "the function declares no argument buffer at [[buffer(0)]] — its SPIR-V \
                 declares sets {want:?}. This is C2's `const BUFFER_INDEX: usize = 0` on a \
                 kernel that references nothing in space0, and the real call would not \
                 report it: it ASSERTS"
            ));
        }
        want = vec![0];
    }

    let mut maps = BTreeMap::new();
    let mut encs = BTreeMap::new();
    for &set in &want {
        let mut refl: Option<Retained<MTLArgument>> = None;
        let enc = unsafe {
            #[allow(deprecated)]
            func.newArgumentEncoderWithBufferIndex_reflection(set as usize, Some(&mut refl))
        };
        // MEASURED UNREACHABLE, and kept anyway. Metal does not return nil for
        // an index that is not an argument buffer — it ASSERTS:
        //
        //   -[_MTLFunction newArgumentEncoderWithBufferIndex:reflection:
        //     functionReflection:]:11540: failed assertion
        //     `bufferIndex 0 does not identify an argument buffer'
        //
        // (`FR_MTL_BUFFER0` on `cs_set1`, Apple M1, macOS 26.5 — SIGABRT, exit
        // 134.) So this arm is a belt, not the guard it reads as, and the real
        // guard is the loop bound above: **the scan asks only for sets the
        // SPIR-V declares.** A scan written as `0..MAX_BUFFER_INDEX` — the
        // obvious way to "find them all" — aborts the process on its first
        // unused index instead of skipping it, and no amount of nil-checking
        // saves it. That is also what C2's constant would have done the moment
        // the corpus gained `cs_set1`: not a mis-derivation, a crash.
        let arg = refl.ok_or_else(|| {
            format!(
                "the function declares no argument buffer at [[buffer({set})]], but its \
                 SPIR-V declares descriptor set {set} — either CROSS_ARGS' \
                 --msl-argument-buffers did not take, or spirv-cross stopped putting set n \
                 at buffer n"
            )
        })?;
        let st = arg.bufferStructType().ok_or_else(|| {
            format!("the argument buffer at [[buffer({set})]] has no struct type")
        })?;

        // Copy out plain values immediately. The reflection objects are +0
        // autoreleased (the type is literally `MTLAutoreleasedArgument`), so
        // holding `Retained<MTLStructMember>` past this point is a lifetime bug
        // waiting for a pool drain.
        let mut slots: Vec<Slot> = st
            .members()
            .iter()
            .map(|mem| Slot {
                id: mem.argumentIndex() as u32,
                name: mem.name().to_string(),
                offset: mem.offset(),
                // The ARRAY stride, when Metal describes this member as an
                // array. `None` for every scalar member, which is all of them
                // outside `texs[]`.
                array_stride: mem.arrayType().map(|a| a.stride()),
                array_ids: mem.arrayType().map(|a| a.argumentIndexStride()),
            })
            .collect();
        slots.sort_by_key(|s| s.id);

        maps.insert(set, ArgMap { buffer_index: set, slots, encoded_len: enc.encodedLength() });
        encs.insert(set, enc);
    }
    Ok((maps, encs))
}

/// The two derivations must agree. Every disagreement is a FAIL that names both
/// sides — never a patch, and never a fallback to the SPIR-V answer alone,
/// which would make the whole cross-check decorative.
pub fn cross_check(maps: &BTreeMap<u32, ArgMap>, spv: &[reflect::Desc]) -> Vec<String> {
    let mut bad = Vec::new();

    // THE SETS THEMSELVES MUST AGREE, and this is the pin behind C3's
    // `[[buffer(n)]] == set n` claim. `derive` asks Metal for exactly the sets
    // the SPIR-V declares and fails on a nil, so this direction is already
    // covered there; what it adds is that nothing EXTRA appeared, which a
    // future scan written as `0..n` would violate silently.
    let mut want: Vec<u32> = spv.iter().map(|d| d.set).collect();
    want.sort_unstable();
    want.dedup();
    let have: Vec<u32> = maps.keys().copied().collect();
    if have != want {
        bad.push(format!(
            "metal reports argument buffers at {have:?}, the SPIR-V declares sets {want:?} — \
             the buffer-index == descriptor-set identity no longer holds"
        ));
    }

    let total: usize = maps.values().map(|m| m.slots.len()).sum();
    if total != spv.len() {
        bad.push(format!(
            "metal reports {total} member(s) across {} set(s), the SPIR-V declares {} \
             descriptor(s)",
            maps.len(),
            spv.len()
        ));
    }

    for (&set, metal) in maps {
        cross_check_set(set, metal, spv, &mut bad);
    }

    bad
}

/// One set's half of `cross_check`.
fn cross_check_set(set: u32, metal: &ArgMap, spv: &[reflect::Desc], bad: &mut Vec<String>) {
    for s in &metal.slots {
        let Some(d) = spv.iter().find(|d| d.name == s.name) else {
            let names: Vec<&str> = spv.iter().map(|d| d.name.as_str()).collect();
            bad.push(format!(
                "argument-buffer member `{}` is in no SPIR-V descriptor {names:?} — \
                 spirv-cross's member naming changed and the join key must be re-measured",
                s.name
            ));
            continue;
        };
        // The register SPACE is the descriptor SET, and spirv-cross emits one
        // argument buffer per set — so a member from another set landing in
        // this buffer is a mis-derivation, not a curiosity.
        if d.set != metal.buffer_index {
            bad.push(format!(
                "`{}` is in set {} but rides the argument buffer at [[buffer({})]]",
                s.name, d.set, metal.buffer_index
            ));
        }
        // A `u#` landing where a `b#` is expected would bind an unordered-access
        // buffer as constants — `Map::class_violations`' check, reused.
        if let Some((reg, _)) = crate::spirv::reg_of_binding(d.binding) {
            if !reflect::class_allows(reg, d.kind) {
                bad.push(format!(
                    "`{}` is a {:?} at binding {} ({:?}), which its register class does not allow",
                    s.name, d.kind, d.binding, reg
                ));
            }
        } else {
            bad.push(format!("`{}` binding {} is outside every register range", s.name, d.binding));
        }
        // THE ARRAY'S TWO DESCRIPTIONS MUST AGREE. `vk::reflect` reports an
        // unbounded array as `count == 0` (its own sentinel — no legal fixed
        // array has that length); Metal describes the same member as an array
        // with a stride. Either one alone is not enough to place element i, and
        // the mismatch is silent: a member Metal thinks is scalar takes only
        // element 0's write, so `texs[1..]` reads whatever the allocation had.
        let unbounded = d.count == 0;
        match (unbounded, s.array_ids) {
            (true, None) => bad.push(format!(
                "`{}` is an UNBOUNDED array in the SPIR-V (count 0) but Metal describes a \
                 scalar member — there is no stride to place elements past 0 with",
                s.name
            )),
            (false, Some(n)) => bad.push(format!(
                "`{}` is a scalar in the SPIR-V (count {}) but Metal describes an array \
                 consuming {n} id(s) per element",
                s.name, d.count
            )),
            (true, Some(0)) => bad.push(format!(
                "`{}` consumes 0 argument-buffer ids per element — every element would land \
                 on element 0",
                s.name
            )),
            _ => {}
        }
    }

    // THE PIN THAT MAKES THE MEASUREMENT A CLAIM. spirv-cross assigns ids
    // densely, in **DECLARATION** order — the module's `OpVariable` stream —
    // over the referenced resources only. This does not hardcode an id (the map
    // is still read off the function); it asserts the RULE, so a change in
    // spirv-cross's assignment is a loud finding rather than a silently
    // different binding.
    //
    // C2 SAID "ASCENDING BINDING ORDER" AND THAT WAS WRONG. It passed because
    // `smoke.hlsl` declares b1, u0, u1, u2 — declaration order and binding order
    // coincide there by accident, and it is the only unit C2 bound. Measured on
    // FFX (one module, ids dense 0..11, cross-read with spirv-dis):
    //
    //   [[id]]   0   1   2   3   4   5   6     7   8   9  10  11
    //   binding  12  8   3   9  10   5  1001   7  11   2   0   1
    //
    // Under the old rule `[[id(0)]]` would have to be binding 0; it is binding
    // 12, and binding 0 sits at `[[id(10)]]`. Any unit whose declaration order
    // disagrees with its binding order — which is every unit that declares a
    // `t` before a `u`, since SHIFT_T < SHIFT_U — fired the old pin on CORRECT
    // code. `reflect::Desc::decl` exists to carry the order the `(set, binding)`
    // sort destroys.
    //
    // AND C3 CORRECTS IT A SECOND TIME: THE NUMBERING IS PER SET. The ordering
    // is module-wide (`decl` is the `OpVariable` ordinal over the whole module),
    // but each set struct restarts at `[[id(0)]]`. Measured on `mtlbind.hlsl`,
    // whose declaration order is `texs`(space2), `samp_lin`, `samp_aniso`,
    // `set1_out`(space1) — and whose emitted MSL is:
    //
    //   cs_arr   set0 { outbuf 0 }  set1 { samp_lin 0, samp_aniso 1 }  set2 { texs 0 }
    //   cs_set1  set1 { samp_lin 0, set1_out 1 }                       set2 { texs 0 }
    //
    // A pin written over ALL descriptors at once expects a single dense `0..n`
    // and fires on every one of those rows. That is the same failure mode as
    // the ascending-binding rule this comment already records: correct on the
    // one-set subject it was written against, wrong the moment the subject
    // gained a second set. Filter to THIS set, then rank by `decl` within it.
    let mut by_decl: Vec<(u32, &str)> =
        spv.iter().filter(|d| d.set == set).map(|d| (d.decl, d.name.as_str())).collect();
    by_decl.sort();
    for (want_id, (_, name)) in by_decl.iter().enumerate() {
        if let Some(s) = metal.slot(name) {
            if s.id as usize != want_id {
                bad.push(format!(
                    "`{name}` is at [[id({})]]; declaration order within set {set} puts it at \
                     {want_id} — spirv-cross's id assignment changed, so the pin must be \
                     re-measured (do NOT patch it with a constant)",
                    s.id
                ));
            }
        }
    }
}

/// Pure gate: the parts that need no device and no toolchain.
pub fn self_test() -> Result<(), String> {
    // A map shaped like the measured `cs_prep` one.
    let map = ArgMap {
        buffer_index: 0,
        slots: vec![
            Slot { id: 0, name: "counters".into(), offset: 0, array_stride: None, array_ids: None },
            Slot { id: 1, name: "args".into(), offset: 8, array_stride: None, array_ids: None },
        ],
        encoded_len: 16,
    };
    let desc = |name: &str, binding: u32, decl: u32| reflect::Desc {
        set: 0,
        binding,
        kind: DescKind::StorageBuffer,
        count: 1,
        name: name.into(),
        decl,
    };
    // `smoke.hlsl` declares `counters` (u0) before `args` (u1), so here
    // declaration order and binding order AGREE. That agreement is exactly why
    // C2 could measure the id rule wrong and still pass — see `cross_check`.
    let spv = vec![
        desc("counters", crate::spirv::binding_of(Reg::U, 0), 0),
        desc("args", crate::spirv::binding_of(Reg::U, 1), 1),
    ];
    // `cross_check` is set-keyed since C3; every one-set case below goes
    // through this, so the pre-C3 cases read exactly as they did.
    let one = |m: &ArgMap| BTreeMap::from([(m.buffer_index, m.clone())]);

    let bad = cross_check(&one(&map), &spv);
    if !bad.is_empty() {
        return Err(format!("cross_check rejected an agreeing pair: {bad:?}"));
    }

    // Teeth, one per way the two derivations can disagree.

    // A member Metal reports that the SPIR-V does not declare.
    let mut renamed = map.clone();
    renamed.slots[1].name = "argz".into();
    if cross_check(&one(&renamed), &spv).is_empty() {
        return Err("cross_check accepted a member with no SPIR-V descriptor".into());
    }

    // The ids no longer follow declaration order. This is the pin that turns the
    // measurement into a claim, and it is the one a future spirv-cross could
    // break.
    let mut swapped = map.clone();
    swapped.slots[0].id = 1;
    swapped.slots[1].id = 0;
    let bad = cross_check(&one(&swapped), &spv);
    if !bad.iter().any(|b| b.contains("id assignment changed")) {
        return Err(format!("cross_check accepted ids out of declaration order: {bad:?}"));
    }

    // THE TOOTH FOR THE CORRECTION ITSELF, and it is the one case the old rule
    // got wrong rather than merely stated oddly. Declare the resources in the
    // order u1, u0 — so declaration order and binding order DISAGREE — and give
    // the ids in declaration order, which is what spirv-cross actually emits.
    // The corrected pin must ACCEPT this; the old ascending-binding pin rejected
    // it, and that is the whole bug.
    let mut rev = vec![
        desc("args", crate::spirv::binding_of(Reg::U, 1), 0),
        desc("counters", crate::spirv::binding_of(Reg::U, 0), 1),
    ];
    let rev_map = ArgMap {
        buffer_index: 0,
        slots: vec![
            Slot { id: 0, name: "args".into(), offset: 0, array_stride: None, array_ids: None },
            Slot { id: 1, name: "counters".into(), offset: 8, array_stride: None, array_ids: None },
        ],
        encoded_len: 16,
    };
    let bad = cross_check(&one(&rev_map), &rev);
    if !bad.is_empty() {
        return Err(format!(
            "cross_check REJECTED declaration-ordered ids whose bindings descend — this is \
             exactly the case the C2 ascending-binding pin got wrong, so a failure here means \
             the correction was reverted: {bad:?}"
        ));
    }
    // …and it must still bite when the ids disagree with declaration order on
    // that same disagreeing layout, or the acceptance above proves nothing.
    rev.swap(0, 1);
    rev[0].decl = 0;
    rev[1].decl = 1;
    let bad = cross_check(&one(&rev_map), &rev);
    if !bad.iter().any(|b| b.contains("id assignment changed")) {
        return Err(format!(
            "cross_check accepted ids that contradict declaration order on a layout whose \
             bindings descend — the corrected pin is vacuous: {bad:?}"
        ));
    }

    // A member from another descriptor set riding this buffer.
    let mut other_set = spv.clone();
    other_set[1].set = 1;
    if cross_check(&one(&map), &other_set).is_empty() {
        return Err("cross_check accepted a member from a different set".into());
    }

    // Counts that disagree at all.
    if cross_check(&one(&map), &spv[..1]).is_empty() {
        return Err("cross_check accepted a member count mismatch".into());
    }

    // A binding outside every register range cannot be attributed.
    let mut wild = spv.clone();
    wild[0].binding = crate::spirv::SHIFT_S + crate::spirv::SHIFT_STRIDE;
    if cross_check(&one(&map), &wild).is_empty() {
        return Err("cross_check accepted a binding outside every register range".into());
    }

    // `usage_for` derives from the descriptor KIND, so a uniform buffer must
    // never be handed write access however it is named. Naming this one
    // `counters` is the point: a first draft keyed off the name `Push` and
    // would pass a test that used the expected name.
    let mut kinds = spv.clone();
    kinds[0].kind = DescKind::UniformBuffer;
    if usage_for("counters", &kinds)? != Some(MTLResourceUsage::Read) {
        return Err("a uniform buffer was given write usage".into());
    }
    if usage_for("counters", &spv)? != Some(MTLResourceUsage::Read | MTLResourceUsage::Write) {
        return Err("a storage buffer was not given read|write usage".into());
    }
    if usage_for("nope", &spv).is_ok() {
        return Err("usage_for accepted a name with no SPIR-V descriptor".into());
    }
    // C3's kinds, each asserted for the thing it is easy to get wrong.
    let kind_of = |k: DescKind| {
        let mut v = spv.clone();
        v[0].kind = k;
        usage_for("counters", &v)
    };
    // A texture is READ, never Read|Write — a sampled image handed write access
    // is the mirror of the uniform-buffer case above.
    if kind_of(DescKind::SampledImage)? != Some(MTLResourceUsage::Read) {
        return Err("a sampled image was not given read-only usage".into());
    }
    if kind_of(DescKind::StorageImage)? != Some(MTLResourceUsage::Read | MTLResourceUsage::Write) {
        return Err("a storage image was not given read|write usage".into());
    }
    // `None` MEANS "not an MTLResource", and it must not be reachable by any
    // other kind — a stray `None` would drop a real resource out of the
    // residency list silently, which is the one failure `set_buffer`'s
    // bind-and-declare-together design exists to make inexpressible.
    if kind_of(DescKind::Sampler)? != None {
        return Err("a sampler was given a residency usage — it is not an MTLResource".into());
    }
    // An acceleration structure reaching here is C4's work started early, the
    // same refusal a texture used to get.
    if kind_of(DescKind::AccelStruct).is_ok() {
        return Err("usage_for accepted an acceleration structure".into());
    }

    // ---- C3: the multi-set cases. Pure, and the shape `--check-mtl` cannot
    // reach on a box with no Metal — which is every CI runner this suite has.

    // The MEASURED `cs_set1` layout: sets 1 and 2, NOTHING at 0, ids dense
    // WITHIN each set, and the module-wide declaration order interleaving them
    // (`texs` is declared first but lands at [[id(0)]] of its own set, while
    // `samp_lin` — declared second — is [[id(0)]] of set 1).
    let texs = reflect::Desc {
        set: 2,
        binding: crate::spirv::binding_of(Reg::T, 0),
        kind: DescKind::SampledImage,
        count: 0, // UNBOUNDED — vk::reflect's sentinel
        name: "texs".into(),
        decl: 0,
    };
    let samp_lin = reflect::Desc {
        set: 1,
        binding: crate::spirv::binding_of(Reg::S, 0),
        kind: DescKind::Sampler,
        count: 1,
        name: "samp_lin".into(),
        decl: 1,
    };
    let set1_out = reflect::Desc {
        set: 1,
        binding: crate::spirv::binding_of(Reg::U, 0),
        kind: DescKind::StorageBuffer,
        count: 1,
        name: "set1_out".into(),
        decl: 3,
    };
    let set1_spv = vec![texs.clone(), samp_lin.clone(), set1_out.clone()];
    let set1_maps = BTreeMap::from([
        (
            1,
            ArgMap {
                buffer_index: 1,
                slots: vec![
                    Slot { id: 0, name: "samp_lin".into(), offset: 0, array_stride: None, array_ids: None },
                    Slot { id: 1, name: "set1_out".into(), offset: 8, array_stride: None, array_ids: None },
                ],
                encoded_len: 16,
            },
        ),
        (
            2,
            ArgMap {
                buffer_index: 2,
                slots: vec![Slot {
                    id: 0,
                    name: "texs".into(),
                    offset: 0,
                    // MEASURED 0 on an M1: a texture entry has no byte stride.
                    array_stride: Some(0),
                    array_ids: Some(1),
                }],
                encoded_len: 8,
            },
        ),
    ]);
    let bad = cross_check(&set1_maps, &set1_spv);
    if !bad.is_empty() {
        return Err(format!(
            "cross_check REJECTED the measured cs_set1 layout — sets 1 and 2 with nothing at \
             0, ids dense within each set. This is what spirv-cross emits; a failure here \
             means the per-set correction was reverted: {bad:?}"
        ));
    }

    // THE TOOTH FOR THAT CORRECTION, and it is the one a careless multi-set
    // port produces: ids numbered densely ACROSS the sets in declaration order
    // — `texs` 0, `samp_lin` 1, `set1_out` 2 — which is what the pre-C3
    // `cross_check` expected and what Metal provably does not emit.
    let mut global = set1_maps.clone();
    global.get_mut(&1).unwrap().slots[0].id = 1;
    global.get_mut(&1).unwrap().slots[1].id = 2;
    let bad = cross_check(&global, &set1_spv);
    if !bad.iter().any(|b| b.contains("id assignment changed")) {
        return Err(format!(
            "cross_check accepted ids numbered densely ACROSS sets — the per-set pin is \
             vacuous: {bad:?}"
        ));
    }

    // Metal answering at an index the SPIR-V never declared. A scan written as
    // a dense `0..n` loop produces exactly this on cs_set1.
    let mut extra = set1_maps.clone();
    extra.insert(0, ArgMap { buffer_index: 0, slots: vec![], encoded_len: 0 });
    if !cross_check(&extra, &set1_spv).iter().any(|b| b.contains("identity no longer holds")) {
        return Err("cross_check accepted an argument buffer at a set the SPIR-V never \
                    declared — the buffer-index == set pin is vacuous"
            .into());
    }

    // The array's two descriptions disagreeing, both directions. Either one
    // alone leaves the host with no way to place element i, and neither is
    // observable in a readback that only ever writes element 0.
    let mut scalar = set1_maps.clone();
    scalar.get_mut(&2).unwrap().slots[0].array_ids = None;
    if !cross_check(&scalar, &set1_spv).iter().any(|b| b.contains("UNBOUNDED array")) {
        return Err("cross_check accepted an unbounded SPIR-V array that Metal calls a \
                    scalar member"
            .into());
    }
    let mut bounded = set1_spv.clone();
    bounded[0].count = 1;
    if !cross_check(&set1_maps, &bounded).iter().any(|b| b.contains("scalar in the SPIR-V")) {
        return Err("cross_check accepted a Metal array whose SPIR-V descriptor is a scalar".into());
    }
    let mut zero = set1_maps.clone();
    zero.get_mut(&2).unwrap().slots[0].array_ids = Some(0);
    if !cross_check(&zero, &set1_spv).iter().any(|b| b.contains("0 argument-buffer ids")) {
        return Err("cross_check accepted array stride 0 — every element would land on \
                    element 0"
            .into());
    }

    // The summed count across sets. Without this the multi-set port of the
    // count check is untested until a device runs it.
    if !cross_check(&set1_maps, &set1_spv[..2]).iter().any(|b| b.contains("member(s) across")) {
        return Err("cross_check accepted a member count mismatch spread over two sets".into());
    }

    Ok(())
}
