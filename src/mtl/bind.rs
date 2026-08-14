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
//! ascending binding order over *only the resources that entry point
//! references*. MEASURED on `smoke.hlsl`, whose three kernels share one file and
//! disagree with each other:
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
//! # Residency is STRUCTURAL, because nothing else could catch it
//!
//! A resource reached *through* an argument buffer is neither made resident nor
//! hazard-tracked by the encoder: `useResource:usage:` is mandatory and has no
//! Vulkan or D3D12 analogue, so no habit from either backend reminds you. It is
//! also the one rule here whose omission this gate probably CANNOT observe — on
//! unified memory a non-heap `MTLBuffer` is already page-backed, so the
//! omission tends to pass, and Metal API validation (which does report it) is
//! off in CI. So `ArgBuf` exposes exactly one mutator, `set_buffer`, which
//! records the resource as it binds it: **"forgot `useResource`" is not
//! expressible**, rather than checked. That is `planes.rs`'s move — own the
//! allocation and the invariant stops being aspirational — applied to a rule
//! that will turn fatal the day these buffers move into an `MTLHeap`.
//!
//! # What this deliberately does NOT cover
//!
//! `smoke.hlsl` declares no `t`, no `s`, and nothing in `space1`. So C2
//! exercises the set-0 buffer half of `CROSS_ARGS` and **neither**
//! `--msl-device-argument-buffer 1` **nor** the tier-2 unbounded `texs[]` array
//! those flags argue hardest for (`msl.rs:55-60`). Textures, samplers and a
//! second descriptor set are C3's; each adds a distinct failure mode, and the
//! value of one argument buffer over four buffers in one space is that every
//! failure has exactly one cause.

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
    MTLFunction, MTLLibrary, MTLResourceUsage,
};
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
}

/// The argument-buffer layout of ONE entry point.
#[derive(Clone, Debug)]
pub struct ArgMap {
    /// The `[[buffer(n)]]` the argument buffer itself binds at — derived, not
    /// the literal 0 it happens to equal.
    pub buffer_index: u32,
    /// Members, ascending by `id`.
    pub slots: Vec<Slot>,
    /// `MTLArgumentEncoder::encodedLength` — the argument buffer's own size.
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
        v.extend(
            self.slots
                .iter()
                .map(|s| format!("  [[id({})]] @{:<4} {}", s.id, s.offset, s.name)),
        );
        v
    }
}

/// A compiled entry point: the pipeline, the encoder that knows its layout, the
/// derived map, and the group size the host must supply at dispatch.
pub struct Kernel {
    pub name: String,
    pub map: ArgMap,
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
    argenc: Retained<ProtocolObject<dyn MTLArgumentEncoder>>,
}

impl Kernel {
    pub fn pipeline(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.pipeline
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
        let len = self.map.encoded_len.max(1);
        let buf = m.buffer(len)?;
        Ok(ArgBuf {
            buf,
            argenc: self.argenc.clone(),
            map: self.map.clone(),
            descs: self.descs.clone(),
            resident: Vec::new(),
        })
    }
}

/// A filled argument buffer, and the residency list it built as it filled.
pub struct ArgBuf {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    argenc: Retained<ProtocolObject<dyn MTLArgumentEncoder>>,
    map: ArgMap,
    descs: Vec<reflect::Desc>,
    resident: Vec<(Retained<ProtocolObject<dyn MTLBuffer>>, MTLResourceUsage)>,
}

impl ArgBuf {
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
        let slot = self
            .map
            .slot(name)
            .ok_or_else(|| format!("no argument-buffer member named `{name}`"))?;
        let usage = usage_for(name, &self.descs)?;
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
            self.argenc.setArgumentBuffer_offset(Some(&self.buf), 0);
            self.argenc.setBuffer_offset_atIndex(Some(buf), 0, slot.id as usize);
        }
        self.resident.push((buf.clone(), usage));
        Ok(())
    }

    /// Bind the argument buffer at its derived index and declare every resource
    /// reached through it.
    pub fn bind(&self, enc: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        self.bind_opts(enc, None, true);
    }

    /// `bind`, with the two things a plant needs to get wrong.
    ///
    /// `index` overrides the DERIVED bind point; `residency` drops the
    /// `useResource:` replay. Both exist so the gate can make itself fail on
    /// demand — the `FR_ABL` read-only-probe idiom — and neither is reachable
    /// from `bind`, so an ordinary caller cannot take either by accident.
    pub fn bind_opts(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        index: Option<u32>,
        residency: bool,
    ) {
        let at = index.unwrap_or(self.map.buffer_index) as usize;
        unsafe { enc.setBuffer_offset_atIndex(Some(&self.buf), 0, at) };
        if residency {
            for (r, usage) in &self.resident {
                let res = ProtocolObject::from_ref(&**r);
                enc.useResource_usage(res, *usage);
            }
        }
    }

    /// The residency list, for the gate's anti-vacuity check — a bind that
    /// declared nothing would otherwise look identical to one that declared
    /// everything.
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
    pub fn encoded_words(&self, m: &Mtl) -> Vec<u32> {
        m.read_words(&self.buf, self.map.encoded_len / 4)
    }
}

/// `Read` for a uniform buffer, `Read | Write` for a storage buffer.
///
/// DERIVED from the descriptor kind the SPIR-V declares, never from the
/// resource's name — a first draft matched `"Push"` and worked only because
/// `smoke.hlsl`'s constant buffer happens to be called that, which is the same
/// class of mistake as hardcoding an `[[id(n)]]`.
///
/// Not a caller-supplied argument either: a per-call-site usage is a
/// per-call-site opportunity to under-declare one, and an under-declared
/// resource is exactly the failure `useResource:` exists to prevent.
fn usage_for(name: &str, descs: &[reflect::Desc]) -> Result<MTLResourceUsage, String> {
    let d = descs
        .iter()
        .find(|d| d.name == name)
        .ok_or_else(|| format!("`{name}` is in the argument buffer but not in the SPIR-V"))?;
    match d.kind {
        DescKind::UniformBuffer => Ok(MTLResourceUsage::Read),
        DescKind::StorageBuffer => Ok(MTLResourceUsage::Read | MTLResourceUsage::Write),
        // Deliberately exhaustive-by-refusal rather than a permissive default:
        // C2 binds buffers only, and a texture or sampler arriving here means
        // C3's work has started without its residency question being answered.
        k => Err(format!("`{name}` is a {k:?}, which mtl::bind does not bind yet (C2 is buffers only)")),
    }
}

/// Load a `.metallib`, build the pipeline for its ONE entry point, and derive
/// the argument-buffer map.
///
/// `words` is the SPIR-V the metallib was built from — the second derivation's
/// input, and where the group size comes from.
pub fn load(m: &Mtl, lib_path: &Path, words: &[u32]) -> Result<Kernel, String> {
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

    // ONE encoder, carrying both halves of the derivation — see `derive`.
    let (map, argenc) = derive(&func)?;
    let local_size = crate::spirv::local_size(words)?;
    // Reflected HERE rather than by the caller, so the map and the descriptors
    // it is checked against provably come from the same module.
    let descs = reflect::reflect(words).map_err(|e| format!("SPIR-V reflection: {e}"))?;

    Ok(Kernel { name: fname.to_string(), map, descs, local_size, pipeline, argenc })
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
/// WHICH `[[buffer(n)]]`: measured as 0, and 0 is exactly the kind of literal
/// this milestone exists not to write — so it is the reflection's PRESENCE at
/// that index that establishes it, and its absence is a diagnosis rather than a
/// nil deref somewhere later. spirv-cross emits one argument buffer per
/// descriptor SET and `smoke.hlsl` uses one space; a set-1 corpus (C3) turns
/// this into a scan, which is why the index is carried in `ArgMap` rather than
/// assumed at the call site.
#[allow(deprecated)]
fn derive(
    func: &ProtocolObject<dyn MTLFunction>,
) -> Result<(ArgMap, Retained<ProtocolObject<dyn MTLArgumentEncoder>>), String> {
    const BUFFER_INDEX: usize = 0;

    let mut refl: Option<Retained<MTLArgument>> = None;
    let enc = unsafe {
        #[allow(deprecated)]
        func.newArgumentEncoderWithBufferIndex_reflection(BUFFER_INDEX, Some(&mut refl))
    };
    let arg = refl.ok_or(
        "the function declares no argument buffer at [[buffer(0)]] — \
         CROSS_ARGS' --msl-argument-buffers did not take",
    )?;
    let st = arg
        .bufferStructType()
        .ok_or("the argument buffer has no struct type — nothing to derive")?;

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
        })
        .collect();
    slots.sort_by_key(|s| s.id);

    let map =
        ArgMap { buffer_index: BUFFER_INDEX as u32, slots, encoded_len: enc.encodedLength() };
    Ok((map, enc))
}

/// The two derivations must agree. Every disagreement is a FAIL that names both
/// sides — never a patch, and never a fallback to the SPIR-V answer alone,
/// which would make the whole cross-check decorative.
pub fn cross_check(metal: &ArgMap, spv: &[reflect::Desc]) -> Vec<String> {
    let mut bad = Vec::new();

    if metal.slots.len() != spv.len() {
        bad.push(format!(
            "metal reports {} member(s), the SPIR-V declares {} descriptor(s)",
            metal.slots.len(),
            spv.len()
        ));
    }

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
    }

    // THE PIN THAT MAKES THE MEASUREMENT A CLAIM. spirv-cross assigns ids
    // densely, in ascending binding order, over the referenced resources only.
    // This does not hardcode an id — the map is still read off the function —
    // it asserts the RULE the map was measured to follow, so a change in
    // spirv-cross's assignment is a loud finding instead of a silently
    // different binding.
    let mut by_binding: Vec<(u32, &str)> =
        spv.iter().map(|d| (d.binding, d.name.as_str())).collect();
    by_binding.sort();
    for (want_id, (_, name)) in by_binding.iter().enumerate() {
        if let Some(s) = metal.slot(name) {
            if s.id as usize != want_id {
                bad.push(format!(
                    "`{name}` is at [[id({})]]; ascending-binding order puts it at {want_id} — \
                     spirv-cross's id assignment changed, so the pin must be re-measured \
                     (do NOT patch it with a constant)",
                    s.id
                ));
            }
        }
    }

    bad
}

/// Pure gate: the parts that need no device and no toolchain.
pub fn self_test() -> Result<(), String> {
    // A map shaped like the measured `cs_prep` one.
    let map = ArgMap {
        buffer_index: 0,
        slots: vec![
            Slot { id: 0, name: "counters".into(), offset: 0 },
            Slot { id: 1, name: "args".into(), offset: 8 },
        ],
        encoded_len: 16,
    };
    let desc = |name: &str, binding: u32| reflect::Desc {
        set: 0,
        binding,
        kind: DescKind::StorageBuffer,
        count: 1,
        name: name.into(),
    };
    let spv = vec![
        desc("counters", crate::spirv::binding_of(Reg::U, 0)),
        desc("args", crate::spirv::binding_of(Reg::U, 1)),
    ];
    let bad = cross_check(&map, &spv);
    if !bad.is_empty() {
        return Err(format!("cross_check rejected an agreeing pair: {bad:?}"));
    }

    // Teeth, one per way the two derivations can disagree.

    // A member Metal reports that the SPIR-V does not declare.
    let mut renamed = map.clone();
    renamed.slots[1].name = "argz".into();
    if cross_check(&renamed, &spv).is_empty() {
        return Err("cross_check accepted a member with no SPIR-V descriptor".into());
    }

    // The ids no longer follow ascending binding order. This is the pin that
    // turns the measurement into a claim, and it is the one a future
    // spirv-cross could break.
    let mut swapped = map.clone();
    swapped.slots[0].id = 1;
    swapped.slots[1].id = 0;
    let bad = cross_check(&swapped, &spv);
    if !bad.iter().any(|b| b.contains("id assignment changed")) {
        return Err(format!("cross_check accepted ids out of binding order: {bad:?}"));
    }

    // A member from another descriptor set riding this buffer.
    let mut other_set = spv.clone();
    other_set[1].set = 1;
    if cross_check(&map, &other_set).is_empty() {
        return Err("cross_check accepted a member from a different set".into());
    }

    // Counts that disagree at all.
    if cross_check(&map, &spv[..1]).is_empty() {
        return Err("cross_check accepted a member count mismatch".into());
    }

    // A binding outside every register range cannot be attributed.
    let mut wild = spv.clone();
    wild[0].binding = crate::spirv::SHIFT_S + crate::spirv::SHIFT_STRIDE;
    if cross_check(&map, &wild).is_empty() {
        return Err("cross_check accepted a binding outside every register range".into());
    }

    // `usage_for` derives from the descriptor KIND, so a uniform buffer must
    // never be handed write access however it is named. Naming this one
    // `counters` is the point: a first draft keyed off the name `Push` and
    // would pass a test that used the expected name.
    let mut kinds = spv.clone();
    kinds[0].kind = DescKind::UniformBuffer;
    if usage_for("counters", &kinds)? != MTLResourceUsage::Read {
        return Err("a uniform buffer was given write usage".into());
    }
    if usage_for("counters", &spv)? != (MTLResourceUsage::Read | MTLResourceUsage::Write) {
        return Err("a storage buffer was not given read|write usage".into());
    }
    if usage_for("nope", &spv).is_ok() {
        return Err("usage_for accepted a name with no SPIR-V descriptor".into());
    }
    // A texture reaching the buffer binder is C3's work started early, not a
    // thing to default permissively.
    let mut tex = spv.clone();
    tex[0].kind = DescKind::SampledImage;
    if usage_for("counters", &tex).is_ok() {
        return Err("usage_for accepted a non-buffer descriptor".into());
    }
    Ok(())
}
