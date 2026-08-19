//! What a SPIR-V module ASKS FOR: the descriptor bindings it declares, read
//! back out of the compiled words.
//!
//! # Why this exists rather than a table
//!
//! Vulkan needs a `VkDescriptorSetLayout` per set before anything can be
//! bound, and that layout must agree with every module bound against it — in
//! set, in binding number, in descriptor TYPE, and in array length. D3D12 gets
//! the same agreement from `create_root_signature`, which is 150 lines of
//! hand-written parameters kept in step with `src/shaders/` by nothing but
//! care. Writing that table a second time for Vulkan would be writing the same
//! liability twice, in a language where the failure is quieter: a root
//! signature that disagrees with a shader fails PSO creation loudly, while a
//! descriptor-set layout that disagrees is caught only by a validation layer
//! that may not be installed (M2c's planted binding shift PASSED unvalidated —
//! the driver resolved it to the only slot there was).
//!
//! So the layout is DERIVED. The corpus already states the register map, in
//! `register(...)` declarations that are the single source of truth for both
//! backends; `spirv.rs`'s `-fvk-*-shift` flags translate those into
//! (set, binding) pairs; and this module reads the answer back out of the
//! module DXC produced. Nothing is transcribed, so nothing can drift — and
//! because the reflection covers the whole corpus, a binding that two units
//! declare with two different types is a hard error here instead of a
//! wrong-resource read at run time.
//!
//! # Scope
//!
//! Deliberately narrow: descriptor variables only. Not a general SPIR-V
//! reader, not a validator (`spirv-val` is that, and `--check-spirv` S3 runs
//! it), and not an interface reflector — push constants, inputs/outputs,
//! specialization constants and execution modes are all uninteresting to a
//! backend whose only stage is compute against one pipeline layout. The parse
//! is a linear walk of the instruction stream; every unknown opcode is
//! skipped by its own word count, which is what keeps it total over SPIR-V
//! versions it has never seen.
//!
//! # Why this is `crate::reflect` and not `vk::reflect`
//!
//! Verbatim the argument `crate::spirv`'s header makes about itself, and the
//! same fix: it names no `ash` type and never did. What it reads is SPIR-V,
//! which this tree now has THREE consumers for — Vulkan binds against it,
//! `mtl::bind` cross-checks spirv-cross's Metal argument numbering against it,
//! and `--check-spirv`'s S0 runs its `self_test` with no device and no backend
//! at all. That last one is what forced the move: S0 is a pure gate, and once
//! `crate::spirv` compiled on Windows, leaving the reflector under `vk/` would
//! have made a GPU-free corpus gate depend on a backend that platform does not
//! build. `vk::reflect` re-exports it, so no call site moved.
//!
//! It is also the derivation source for a WebGPU bind-group layout, which is
//! the same shape a third time: `(set, binding)` read back out of the module
//! rather than transcribed. `vk/layout.rs` remains the ONE place `DescKind`
//! meets a Vulkan type; keep it that way, and keep this module free of every
//! backend's vocabulary.

use std::collections::{BTreeMap, BTreeSet};

use crate::spirv::{self, Reg};

// ---------------------------------------------------------------------------
// The SPIR-V words this module knows. Values from the SPIR-V 1.6 spec; the
// stream is self-describing (word count in the high half of the opcode word),
// so anything absent here is stepped over rather than misread.
// ---------------------------------------------------------------------------

const MAGIC: u32 = 0x0723_0203;
const HEADER_WORDS: usize = 5;

const OP_NAME: u16 = 5;
const OP_DECORATE: u16 = 71;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLER: u16 = 26;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_CONSTANT: u16 = 43;
const OP_VARIABLE: u16 = 59;
const OP_TYPE_ACCEL_STRUCT: u16 = 5341;

const DEC_BLOCK: u32 = 2;
const DEC_BUFFER_BLOCK: u32 = 3;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;

const SC_UNIFORM_CONSTANT: u32 = 0;
const SC_UNIFORM: u32 = 2;
const SC_STORAGE_BUFFER: u32 = 12;

/// Image `Dim` operand values this module distinguishes.
const DIM_BUFFER: u32 = 5;

// ---------------------------------------------------------------------------
// The reflected shape
// ---------------------------------------------------------------------------

/// A descriptor type, in the vocabulary a `VkDescriptorSetLayoutBinding`
/// needs. Deliberately NOT `ash::vk::DescriptorType`: this module is pure and
/// its `self_test` runs where no Vulkan exists, and the mapping to ash lives
/// at the one site that builds a layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DescKind {
    /// HLSL `cbuffer` — `Uniform` storage class over a `Block` struct.
    UniformBuffer,
    /// HLSL `StructuredBuffer` / `RWStructuredBuffer` / `ByteAddressBuffer`.
    /// Both `t` and `u` registers land here, which is expected: read-only is a
    /// `NonWritable` decoration in SPIR-V, not a different descriptor type.
    StorageBuffer,
    /// HLSL `Texture2D` and friends.
    SampledImage,
    /// HLSL `RWTexture2D` and friends.
    StorageImage,
    /// HLSL `SamplerState`.
    Sampler,
    /// A combined image+sampler. HLSL never declares one, but a future
    /// `[[vk::combinedImageSampler]]` or a hand-written module could, so the
    /// classifier names it rather than failing.
    CombinedImageSampler,
    /// HLSL `Buffer<T>` — a texel buffer, not a structured one.
    UniformTexelBuffer,
    /// HLSL `RWBuffer<T>`.
    StorageTexelBuffer,
    /// HLSL `RaytracingAccelerationStructure`.
    AccelStruct,
}

impl DescKind {
    pub fn name(self) -> &'static str {
        match self {
            DescKind::UniformBuffer => "uniform-buffer",
            DescKind::StorageBuffer => "storage-buffer",
            DescKind::SampledImage => "sampled-image",
            DescKind::StorageImage => "storage-image",
            DescKind::Sampler => "sampler",
            DescKind::CombinedImageSampler => "combined-image-sampler",
            DescKind::UniformTexelBuffer => "uniform-texel-buffer",
            DescKind::StorageTexelBuffer => "storage-texel-buffer",
            DescKind::AccelStruct => "acceleration-structure",
        }
    }
}

/// One descriptor a module declares.
#[derive(Clone, Debug)]
pub struct Desc {
    pub set: u32,
    pub binding: u32,
    pub kind: DescKind,
    /// Array length. `1` for a scalar declaration, `n` for `res[n]`, and
    /// **0 for an UNBOUNDED array** (`texs[]` — SPIR-V `OpTypeRuntimeArray`),
    /// which is a device requirement rather than a size: indexing one needs
    /// `runtimeDescriptorArray`, and the layout entry needs a real ceiling
    /// chosen by the backend. Zero is the sentinel precisely because no legal
    /// fixed array has that length, so it cannot be mistaken for one.
    pub count: u32,
    /// The `OpName`, when the module carries one. Debug-only — DXC emits the
    /// HLSL identifier here, which is what makes a conflict message legible
    /// ("t7 is `tlas` in one unit and `sw_tri` in another") rather than a pair
    /// of numbers.
    pub name: String,
    /// This descriptor's ordinal in the module's **`OpVariable` stream** — the
    /// order the variables are DECLARED in, which the `(set, binding)` sort
    /// below deliberately destroys.
    ///
    /// NOT debug-only, and it is here because a Metal consumer cannot do its
    /// job without it. spirv-cross assigns argument-buffer `[[id(n)]]` densely
    /// **in declaration order**, not in binding order — measured on FFX, where
    /// `[[id(0)]]` is binding 12 and binding 0 lands at `[[id(10)]]`. Nothing
    /// on the Vulkan side reads this; `mtl::bind::cross_check` is the consumer,
    /// and without it the second derivation cannot express the rule at all.
    ///
    /// It is the ordinal over DESCRIPTORS, not over all variables: the pass
    /// below skips every variable lacking both decorations, and those are not
    /// resources spirv-cross would number.
    pub decl: u32,
}

/// The union of what a set of modules declares — i.e. the pipeline layout the
/// backend must build, derived rather than written.
#[derive(Clone, Debug, Default)]
pub struct Map {
    pub entries: BTreeMap<(u32, u32), Entry>,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub kind: DescKind,
    /// The widest array seen. `0` (unbounded) is absorbing: one unit declaring
    /// `texs[]` makes the binding unbounded for the layout even if another
    /// unit happens to see it as a fixed array.
    pub count: u32,
    /// Every `OpName` seen at this slot, across units.
    pub names: BTreeSet<String>,
    /// How many modules declared it — the reach signal a bare list cannot
    /// give ("this binding is in one unit" reads very differently from "in
    /// forty").
    pub units: u32,
}

impl Map {
    /// Fold one module's descriptors in.
    ///
    /// Returns the conflicts found, rather than failing: a caller gating the
    /// whole corpus wants EVERY disagreement in one run, not the first.
    pub fn add(&mut self, unit: &str, descs: &[Desc]) -> Vec<String> {
        let mut bad = Vec::new();
        for d in descs {
            match self.entries.get_mut(&(d.set, d.binding)) {
                None => {
                    let mut names = BTreeSet::new();
                    if !d.name.is_empty() {
                        names.insert(d.name.clone());
                    }
                    self.entries.insert(
                        (d.set, d.binding),
                        Entry { kind: d.kind, count: d.count, names, units: 1 },
                    );
                }
                Some(e) => {
                    if e.kind != d.kind {
                        bad.push(format!(
                            "set {} binding {} is {} in {unit} ({}) but {} elsewhere ({})",
                            d.set,
                            d.binding,
                            d.kind.name(),
                            if d.name.is_empty() { "?" } else { &d.name },
                            e.kind.name(),
                            e.names.iter().cloned().collect::<Vec<_>>().join("/"),
                        ));
                    }
                    // Unbounded is absorbing; otherwise take the widest.
                    e.count = if e.count == 0 || d.count == 0 { 0 } else { e.count.max(d.count) };
                    if !d.name.is_empty() {
                        e.names.insert(d.name.clone());
                    }
                    e.units += 1;
                }
            }
        }
        bad
    }

    /// Every slot whose reflected TYPE contradicts the register class its
    /// binding number encodes.
    ///
    /// This is the cross-check between the two halves of the binding scheme:
    /// the `-fvk-*-shift` flags DXC was given, and `binding_of`, which the
    /// layout is built from. They are generated from one set of constants, so
    /// they cannot disagree by construction — but "cannot by construction" is
    /// exactly the claim worth making checkable, because the failure mode is
    /// silent (a `u0` landing at binding 0 would be read as `b0` and the
    /// module would sample garbage, with `spirv-val` perfectly happy).
    pub fn class_violations(&self) -> Vec<String> {
        let mut bad = Vec::new();
        for (&(set, binding), e) in &self.entries {
            let Some((reg, n)) = spirv::reg_of_binding(binding) else {
                bad.push(format!(
                    "set {set} binding {binding} ({}) is outside every register range",
                    e.kind.name()
                ));
                continue;
            };
            if !class_allows(reg, e.kind) {
                bad.push(format!(
                    "set {set} binding {binding} reflects as {} but its number says {:?}{n} ({})",
                    e.kind.name(),
                    reg,
                    e.names.iter().cloned().collect::<Vec<_>>().join("/"),
                ));
            }
        }
        bad
    }

    /// One line per slot, in binding order.
    ///
    /// The MIRROR IMAGE of `main.rs`'s `#[cfg_attr(not(windows), ...)]` pair,
    /// and annotated rather than deleted for the same reason they are: both
    /// callers are inside `--check-vk` (`run_check_vk_layout` and the pipeline
    /// dump), so on Windows this is unused today and `dead_code` is right to
    /// say so. It is the human-readable half of the reflection — the
    /// thing a layout gate prints when it fails — and a `--check-wgsl` bind-
    /// group report wants exactly it. Keep the warning silenced NARROWLY, per
    /// target rather than per module, so a genuinely orphaned method still
    /// surfaces.
    #[cfg_attr(not(unix), allow(dead_code))]
    pub fn lines(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(|(&(set, binding), e)| {
                let (reg, n) = spirv::reg_of_binding(binding)
                    .map_or(("?".to_string(), 0), |(r, n)| (format!("{r:?}").to_lowercase(), n));
                let reg = match e.count {
                    0 => format!("{reg}{n}[]"),
                    1 => format!("{reg}{n}"),
                    c => format!("{reg}{n}[{c}]"),
                };
                format!(
                    "set {set} binding {binding:<5} {reg:<8} {:<22} x{:<3} {}",
                    e.kind.name(),
                    e.units,
                    e.names.iter().cloned().collect::<Vec<_>>().join("/"),
                )
            })
            .collect()
    }
}

/// Which descriptor types may legally appear at a register of this class.
///
/// `t` and `u` both admit `StorageBuffer` because HLSL's read-only/read-write
/// distinction is a decoration in SPIR-V, not a type. What they do NOT share
/// is the image half — a `t` is sampled, a `u` is storage — and that is the
/// half a shift mismatch would break.
pub fn class_allows(reg: Reg, kind: DescKind) -> bool {
    match reg {
        Reg::B => kind == DescKind::UniformBuffer,
        Reg::S => kind == DescKind::Sampler,
        Reg::T => matches!(
            kind,
            DescKind::StorageBuffer
                | DescKind::SampledImage
                | DescKind::CombinedImageSampler
                | DescKind::UniformTexelBuffer
                | DescKind::AccelStruct
        ),
        Reg::U => matches!(
            kind,
            DescKind::StorageBuffer | DescKind::StorageImage | DescKind::StorageTexelBuffer
        ),
    }
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

/// Read every descriptor a compiled module declares.
///
/// Total over malformed input: a truncated or nonsensical instruction is an
/// `Err`, never a panic and never a silently short list — this feeds a gate,
/// and a reflector that returns "no descriptors" on a stream it failed to
/// understand would turn a broken module into a passing one.
pub fn reflect(words: &[u32]) -> Result<Vec<Desc>, String> {
    if words.len() < HEADER_WORDS {
        return Err(format!("module is {} words — shorter than a header", words.len()));
    }
    if words[0] != MAGIC {
        return Err(format!("bad magic {:#010x} (expected {MAGIC:#010x})", words[0]));
    }

    // Pass 1: everything a variable's classification can depend on. The
    // stream is not ordered for our convenience (types precede variables, but
    // decorations precede both), so one collecting pass then one deciding
    // pass is simpler than trying to do it in flight.
    let mut set_of: BTreeMap<u32, u32> = BTreeMap::new();
    let mut binding_of_id: BTreeMap<u32, u32> = BTreeMap::new();
    let mut name_of: BTreeMap<u32, String> = BTreeMap::new();
    let mut buffer_block: BTreeSet<u32> = BTreeSet::new();
    let mut block: BTreeSet<u32> = BTreeSet::new();
    /// A type id's shape, insofar as classification cares.
    enum Ty {
        Image { sampled: u32, dim: u32 },
        Sampler,
        SampledImage,
        AccelStruct,
        Struct,
        Array(u32, u32),
        RuntimeArray(u32),
        Pointer(u32, u32),
        Other,
    }
    let mut types: BTreeMap<u32, Ty> = BTreeMap::new();
    let mut consts: BTreeMap<u32, u32> = BTreeMap::new();
    let mut vars: Vec<(u32, u32, u32)> = Vec::new(); // (result, ptr type, storage class)

    let mut i = HEADER_WORDS;
    while i < words.len() {
        let w = words[i];
        let op = (w & 0xffff) as u16;
        let len = (w >> 16) as usize;
        if len == 0 {
            return Err(format!("zero-length instruction at word {i}"));
        }
        if i + len > words.len() {
            return Err(format!(
                "instruction at word {i} claims {len} words, {} remain",
                words.len() - i
            ));
        }
        let a = &words[i + 1..i + len]; // operands
        match op {
            OP_NAME if a.len() >= 2 => {
                name_of.insert(a[0], decode_string(&a[1..]));
            }
            OP_DECORATE if a.len() >= 2 => match a[1] {
                DEC_DESCRIPTOR_SET if a.len() >= 3 => {
                    set_of.insert(a[0], a[2]);
                }
                DEC_BINDING if a.len() >= 3 => {
                    binding_of_id.insert(a[0], a[2]);
                }
                DEC_BUFFER_BLOCK => {
                    buffer_block.insert(a[0]);
                }
                DEC_BLOCK => {
                    block.insert(a[0]);
                }
                _ => {}
            },
            // result, sampled type, dim, depth, arrayed, MS, sampled, format
            OP_TYPE_IMAGE if a.len() >= 7 => {
                types.insert(a[0], Ty::Image { sampled: a[6], dim: a[2] });
            }
            OP_TYPE_SAMPLER if !a.is_empty() => {
                types.insert(a[0], Ty::Sampler);
            }
            OP_TYPE_SAMPLED_IMAGE if !a.is_empty() => {
                types.insert(a[0], Ty::SampledImage);
            }
            OP_TYPE_ACCEL_STRUCT if !a.is_empty() => {
                types.insert(a[0], Ty::AccelStruct);
            }
            OP_TYPE_STRUCT if !a.is_empty() => {
                types.insert(a[0], Ty::Struct);
            }
            OP_TYPE_ARRAY if a.len() >= 3 => {
                types.insert(a[0], Ty::Array(a[1], a[2]));
            }
            OP_TYPE_RUNTIME_ARRAY if a.len() >= 2 => {
                types.insert(a[0], Ty::RuntimeArray(a[1]));
            }
            OP_TYPE_POINTER if a.len() >= 3 => {
                types.insert(a[0], Ty::Pointer(a[1], a[2]));
            }
            // result type, result, value... — only the 32-bit case is a legal
            // array length here, and a wider one would be a length no layout
            // could express anyway.
            OP_CONSTANT if a.len() >= 3 => {
                consts.insert(a[1], a[2]);
            }
            OP_VARIABLE if a.len() >= 3 => {
                vars.push((a[1], a[0], a[2]));
            }
            _ => {}
        }
        i += len;
    }

    // Pass 2: decide. Only variables carrying BOTH decorations are
    // descriptors — a `Uniform` variable without a set/binding is a push
    // constant block or an interface, neither of which belongs in a layout.
    let mut out = Vec::new();
    for (id, ptr_ty, _sc) in vars {
        let (Some(&set), Some(&binding)) = (set_of.get(&id), binding_of_id.get(&id)) else {
            continue;
        };
        let Some(Ty::Pointer(sc, pointee)) = types.get(&ptr_ty).map(|t| match *t {
            Ty::Pointer(a, b) => Ty::Pointer(a, b),
            _ => Ty::Other,
        }) else {
            return Err(format!("descriptor %{id} has a non-pointer type %{ptr_ty}"));
        };

        // Unwrap the array, if any. One level is all HLSL can produce here.
        let (elem, count) = match types.get(&pointee) {
            Some(Ty::Array(e, len_id)) => {
                let n = *consts.get(len_id).ok_or_else(|| {
                    format!("descriptor %{id}: array length %{len_id} is not an OpConstant")
                })?;
                if n == 0 {
                    return Err(format!("descriptor %{id}: array length 0"));
                }
                (*e, n)
            }
            Some(Ty::RuntimeArray(e)) => (*e, 0),
            _ => (pointee, 1),
        };

        let kind = match types.get(&elem) {
            Some(Ty::Image { sampled, dim }) => match (*sampled, *dim == DIM_BUFFER) {
                (1, false) => DescKind::SampledImage,
                (1, true) => DescKind::UniformTexelBuffer,
                (2, false) => DescKind::StorageImage,
                (2, true) => DescKind::StorageTexelBuffer,
                // `Sampled = 0` means "known only at use". HLSL never emits it
                // and a layout cannot guess, so say so rather than pick.
                (s, _) => {
                    return Err(format!(
                        "descriptor %{id} (set {set} binding {binding}) is an image with \
                         Sampled={s} — neither sampled nor storage"
                    ))
                }
            },
            Some(Ty::Sampler) => DescKind::Sampler,
            Some(Ty::SampledImage) => DescKind::CombinedImageSampler,
            Some(Ty::AccelStruct) => DescKind::AccelStruct,
            Some(Ty::Struct) => {
                // Since SPIR-V 1.3 a storage buffer is `StorageBuffer` +
                // `Block`; the older spelling is `Uniform` + `BufferBlock`,
                // which DXC can still emit at a lower target env. Accept both
                // rather than assume the target env this module was built
                // with — the classification is what matters, not the era.
                if sc == SC_STORAGE_BUFFER || (sc == SC_UNIFORM && buffer_block.contains(&elem)) {
                    DescKind::StorageBuffer
                } else if sc == SC_UNIFORM && block.contains(&elem) {
                    DescKind::UniformBuffer
                } else {
                    return Err(format!(
                        "descriptor %{id} (set {set} binding {binding}) is a struct in storage \
                         class {sc} with neither Block nor BufferBlock"
                    ));
                }
            }
            _ if sc == SC_UNIFORM_CONSTANT => {
                return Err(format!(
                    "descriptor %{id} (set {set} binding {binding}) has an opaque type this \
                     reflector does not know"
                ))
            }
            _ => {
                return Err(format!(
                    "descriptor %{id} (set {set} binding {binding}) is neither an opaque type \
                     nor a block"
                ))
            }
        };

        // `vars` was filled in `OP_VARIABLE` order by pass 1 and this loop walks
        // it in that order, so `out.len()` IS the declaration ordinal — taken
        // BEFORE the push, and before the sort below throws the order away.
        let decl = out.len() as u32;
        out.push(Desc {
            set,
            binding,
            kind,
            count,
            name: name_of.get(&id).cloned().unwrap_or_default(),
            decl,
        });
    }
    // Sorted by (set, binding) because every Vulkan consumer wants layout order.
    // `decl` is what survives it — see its doc: the Metal argument-buffer id
    // rule is declaration-ordered, and this sort is exactly what would hide it.
    out.sort_by_key(|d| (d.set, d.binding));
    Ok(out)
}

/// A SPIR-V literal string: UTF-8, NUL-terminated, four bytes per word,
/// little-endian.
fn decode_string(words: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    'outer: for w in words {
        for b in w.to_le_bytes() {
            if b == 0 {
                break 'outer;
            }
            bytes.push(b);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

/// The pure half. Runs anywhere — no Vulkan, no compiler, no GPU.
///
/// The walk is exercised against a hand-built word stream rather than a real
/// module, deliberately: a real module would make this test a function of
/// whichever DXC happens to be on disk, and the property under test is that
/// the PARSER reads a stream correctly, which a synthetic stream states far
/// more sharply (every id and length is chosen to make one behaviour
/// observable).
pub fn self_test() -> Result<(), String> {
    // --- (a) the class table, in both directions ---
    if class_allows(Reg::B, DescKind::StorageBuffer) {
        return Err("a b# register must not admit a storage buffer".into());
    }
    if class_allows(Reg::T, DescKind::StorageImage) {
        return Err("a t# register must not admit a storage image".into());
    }
    if class_allows(Reg::U, DescKind::SampledImage) {
        return Err("a u# register must not admit a sampled image".into());
    }
    if !class_allows(Reg::T, DescKind::AccelStruct) || !class_allows(Reg::T, DescKind::StorageBuffer)
    {
        return Err("a t# register must admit both an AS and a storage buffer".into());
    }
    if !class_allows(Reg::U, DescKind::StorageBuffer)
        || !class_allows(Reg::U, DescKind::StorageImage)
    {
        return Err("a u# register must admit both a storage buffer and a storage image".into());
    }

    // --- (b) the walk, over a synthetic module ---
    //
    // Three descriptors chosen to cover the three shapes that behave
    // differently: a scalar storage buffer, a RUNTIME-array sampled image
    // (the `texs[]` case, whose count-0 sentinel the layout depends on), and
    // a sampler. Ids are arbitrary; the assertions are about what comes back.
    let mut m = Module::new();
    // types
    m.op(OP_TYPE_STRUCT, &[10]); // %10 = struct
    m.op(OP_TYPE_POINTER, &[11, SC_STORAGE_BUFFER, 10]); // %11 = ptr SB %10
    m.op(OP_TYPE_IMAGE, &[20, 1, 1, 0, 0, 0, 1, 0]); // %20 = 2D sampled image
    m.op(OP_TYPE_RUNTIME_ARRAY, &[21, 20]); // %21 = %20[]
    m.op(OP_TYPE_POINTER, &[22, SC_UNIFORM_CONSTANT, 21]);
    m.op(OP_TYPE_SAMPLER, &[30]);
    m.op(OP_TYPE_POINTER, &[31, SC_UNIFORM_CONSTANT, 30]);
    // decorations
    m.op(OP_DECORATE, &[10, DEC_BLOCK]);
    m.op(OP_DECORATE, &[100, DEC_DESCRIPTOR_SET, 0]);
    m.op(OP_DECORATE, &[100, DEC_BINDING, spirv::binding_of(Reg::U, 0)]);
    m.op(OP_DECORATE, &[101, DEC_DESCRIPTOR_SET, 1]);
    m.op(OP_DECORATE, &[101, DEC_BINDING, spirv::binding_of(Reg::T, 10)]);
    m.op(OP_DECORATE, &[102, DEC_DESCRIPTOR_SET, 1]);
    m.op(OP_DECORATE, &[102, DEC_BINDING, spirv::binding_of(Reg::S, 0)]);
    // names
    m.name(100, "accum");
    m.name(101, "texs");
    m.name(102, "samp_lin");
    // variables
    m.op(OP_VARIABLE, &[11, 100, SC_STORAGE_BUFFER]);
    m.op(OP_VARIABLE, &[22, 101, SC_UNIFORM_CONSTANT]);
    m.op(OP_VARIABLE, &[31, 102, SC_UNIFORM_CONSTANT]);
    // A variable with NO set/binding — an interface, not a descriptor. It
    // must be ignored rather than counted or rejected.
    m.op(OP_VARIABLE, &[11, 103, SC_STORAGE_BUFFER]);
    // An instruction this reflector has never heard of, mid-stream, to prove
    // the skip-by-length walk really steps over it.
    m.op(4242, &[1, 2, 3, 4, 5]);

    let d = reflect(&m.words)?;
    if d.len() != 3 {
        return Err(format!("expected 3 descriptors, got {}", d.len()));
    }
    let by = |set: u32, reg: Reg, n: u32| -> &Desc {
        d.iter()
            .find(|x| x.set == set && x.binding == spirv::binding_of(reg, n))
            .expect("descriptor")
    };
    let accum = by(0, Reg::U, 0);
    if accum.kind != DescKind::StorageBuffer || accum.count != 1 || accum.name != "accum" {
        return Err(format!("u0 reflected as {:?}", accum));
    }
    let texs = by(1, Reg::T, 10);
    if texs.kind != DescKind::SampledImage {
        return Err(format!("texs[] reflected as {}", texs.kind.name()));
    }
    // The whole reason count is a u32 with a 0 sentinel rather than an
    // Option: an unbounded array is a DEVICE REQUIREMENT, and a reflector
    // that reported it as 1 would build a layout that silently truncates the
    // scene's texture table to its first entry.
    if texs.count != 0 {
        return Err(format!("texs[] must reflect as unbounded (count 0), got {}", texs.count));
    }
    if by(1, Reg::S, 0).kind != DescKind::Sampler {
        return Err("s0 did not reflect as a sampler".into());
    }

    // --- (c) the map: union, conflict, class check ---
    let mut map = Map::default();
    if !map.add("a", &d).is_empty() {
        return Err("a module cannot conflict with an empty map".into());
    }
    if !map.add("b", &d).is_empty() {
        return Err("a module cannot conflict with itself".into());
    }
    if map.entries.len() != 3 {
        return Err(format!("union of a module with itself has {} slots", map.entries.len()));
    }
    if map.entries[&(0, spirv::binding_of(Reg::U, 0))].units != 2 {
        return Err("the unit count did not accumulate".into());
    }
    if !map.class_violations().is_empty() {
        return Err(format!("a well-formed map reported {:?}", map.class_violations()));
    }
    // Teeth for the conflict path: the SAME slot with a different type must
    // be reported, not silently overwritten by the last writer.
    let clash = vec![Desc {
        set: 0,
        binding: spirv::binding_of(Reg::U, 0),
        kind: DescKind::StorageImage,
        count: 1,
        name: "hdr".into(),
        decl: 0,
    }];
    if map.add("c", &clash).is_empty() {
        return Err("a type conflict at one slot went unreported".into());
    }
    // Teeth for the class check: a storage buffer at a `b#` binding is the
    // shape a shift mismatch takes, and it must not pass.
    let mut bad = Map::default();
    bad.add(
        "x",
        &[Desc {
            set: 0,
            binding: spirv::binding_of(Reg::B, 0),
            kind: DescKind::StorageBuffer,
            count: 1,
            name: "wrong".into(),
            decl: 0,
        }],
    );
    if bad.class_violations().is_empty() {
        return Err("a storage buffer at a b# binding passed the class check".into());
    }

    // --- (d) totality over malformed input ---
    if reflect(&[]).is_ok() {
        return Err("an empty module reflected successfully".into());
    }
    if reflect(&[1, 2, 3, 4, 5]).is_ok() {
        return Err("a module with bad magic reflected successfully".into());
    }
    // A truncated instruction must ERROR. The dangerous failure is not a
    // panic but a short read that looks like "this module declares nothing".
    let mut trunc = m.words.clone();
    trunc.truncate(trunc.len() - 2);
    match reflect(&trunc) {
        Ok(v) => return Err(format!("a truncated module reflected {} descriptors", v.len())),
        Err(e) if e.contains("claims") => {}
        Err(e) => return Err(format!("a truncated module failed for the wrong reason: {e}")),
    }
    // A zero-length instruction would spin the walk forever if it were not
    // rejected — the one malformed input with a non-obvious consequence.
    let mut zero = m.words.clone();
    zero.push(0);
    if reflect(&zero).is_ok() {
        return Err("a zero-length instruction was accepted".into());
    }
    Ok(())
}

/// A word-stream builder for the synthetic module above. Test-only, and
/// deliberately dumb: it emits exactly what it is told, so a test can state a
/// malformed stream as easily as a well-formed one.
struct Module {
    words: Vec<u32>,
}

impl Module {
    fn new() -> Self {
        Module { words: vec![MAGIC, 0x0001_0600, 0, 1000, 0] }
    }
    fn op(&mut self, op: u16, operands: &[u32]) {
        let len = (operands.len() + 1) as u32;
        self.words.push((len << 16) | u32::from(op));
        self.words.extend_from_slice(operands);
    }
    fn name(&mut self, id: u32, s: &str) {
        let mut ops = vec![id];
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        for c in bytes.chunks(4) {
            ops.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        self.op(OP_NAME, &ops);
    }
}
