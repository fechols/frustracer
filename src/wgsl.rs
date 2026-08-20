//! SPIR-V -> WGSL: the browser port's code generator, and the third consumer
//! of the corpus's SPIR-V arm (Vulkan eats it directly, Metal through
//! spirv-cross, the browser through naga — this module).
//!
//! UNCFG'D ON PURPOSE, unlike `crate::spirv` (`cfg(any(unix, windows))`):
//! naga is pure Rust with no loader and no DLL, and the browser session runs
//! THIS code at page load — `--bake-web` ships SPIR-V blobs, and the wasm
//! build translates them with the same naga the `--check-wgsl` gate validated
//! them with. One translator, lockfile-pinned, both sides; the gate and the
//! page cannot drift onto different WGSL.
//!
//! naga rides in as a BASE dependency, not a cargo feature — the repo's
//! recorded rule that a gate's availability must not depend on build flags
//! (Cargo.toml's MetalFX note). It compiles everywhere `--check` does and
//! loads nothing, so the DLL-free bar for every `--check*` is untouched.
//!
//! # Where the teeth are
//!
//! The wave-ops story is the example to keep in mind: the web corpus is
//! assembled under `ABL_NO_WAVE_OPS` (ctr.hlsli's plain-atomic arm — the
//! `FR_ABL=nowave` fallback promoted to the web default). If that define ever
//! failed to reach a unit, the SPIR-V would carry subgroup ops — and
//! `validate()` below runs with capabilities that do NOT include SUBGROUP, so
//! the leak fails `--check-wgsl`'s W4 loudly instead of surfacing as a Tint
//! error in one browser three months later. The capability set is therefore
//! part of the contract, not a permissiveness knob: widen it only when the
//! BROWSER floor actually widens, never to make a stage pass.

/// Capabilities the validator grants — the WebGPU CORE floor, deliberately
/// narrow (see the module note). naga's default() is empty-ish already; this
/// is spelled as a function so the one place to widen it is greppable.
fn capabilities() -> naga::valid::Capabilities {
    naga::valid::Capabilities::empty()
}

/// Flatten an error's SOURCE CHAIN into one line. naga's `Display`s are
/// deliberately shallow ("Global variable [7] 'FrdCb' is invalid") with the
/// actionable detail one or two `source()` links down; a gate that prints
/// only the top line sends its reader into a debugger for information the
/// error already carried.
fn err_chain(e: &dyn std::error::Error) -> String {
    let mut s = e.to_string();
    let mut cur = e.source();
    while let Some(src) = cur {
        s.push_str(": ");
        s.push_str(&src.to_string());
        cur = src.source();
    }
    s
}

/// De-CSE access chains: clone every OpAccessChain so a copy sits in the
/// SAME basic block as each of its users. Runs between DXC and naga — at the
/// gate AND at `--bake-web` (the shipped blobs are post-pass; the page's
/// naga eats exactly what this validated).
///
/// WHY: DXC's legalization CSEs a chain to one site and loads it from many
/// blocks. naga's spv-in spills any expression used outside its defining
/// structured body into a synthesized LocalVariable (front/spv/mod.rs,
/// `get_expr_handle`) — and when the expression is a POINTER, that local is
/// a pointer-typed local, which WGSL cannot hold ("Local variable has a
/// type that can't be stored", cs_sky/cs_sky_lod, measured 2026-08-18). The
/// same spill is what breaks naga's atomic-upgrade walk ("expected to find
/// a global variable", cs_hemi_root/cs_hemi_cell): the atomic's pointer no
/// longer traces to its global. Neither `spirv-opt -O` nor DXC -O1 removes
/// the shape (measured), and naga 30.0.0 is current — so the artifact is
/// normalized instead.
///
/// SOUNDNESS: a clone before a use is always legal SPIR-V — the original
/// chain dominates the use (SSA), so the chain's operands dominate the use
/// too; cloning changes no value. W2 runs spirv-val on the POST-pass module,
/// so the rewrite itself stays under the reference validator.
///
/// OpPhi operands are never rewritten (their values live in predecessor
/// blocks); a phi over a pointer would already be un-WGSLable and fails
/// downstream loudly. Ids: clones take fresh ids above the header bound.
pub fn split_chains(words: &[u32]) -> Vec<u32> {
    const OP_ACCESS_CHAIN: u16 = 65;
    const OP_IN_BOUNDS_ACCESS_CHAIN: u16 = 66;
    const OP_LABEL: u16 = 248;
    const OP_PHI: u16 = 245;
    let is_chain = |op: u16| op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN;

    if words.len() < 5 {
        return words.to_vec();
    }
    // Pass 1: every chain's instruction words + defining block.
    let mut chains: std::collections::HashMap<u32, (Vec<u32>, u32)> = Default::default();
    {
        let (mut i, mut cur_block) = (5usize, 0u32);
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            let op = (words[i] & 0xffff) as u16;
            if wc == 0 || i + wc > words.len() {
                return words.to_vec(); // malformed: hand it to naga untouched
            }
            if op == OP_LABEL {
                cur_block = words[i + 1];
            } else if is_chain(op) && cur_block != 0 {
                chains.insert(words[i + 2], (words[i..i + wc].to_vec(), cur_block));
            }
            i += wc;
        }
    }
    // Recursively clone a chain (and any chain its base rides on) into the
    // use block, appending to `out` just before the caller emits the user.
    fn clone_into(
        def: &[u32],
        chains: &std::collections::HashMap<u32, (Vec<u32>, u32)>,
        use_block: u32,
        bound: &mut u32,
        out: &mut Vec<u32>,
        clones: &mut usize,
    ) -> u32 {
        let mut inst = def.to_vec();
        if inst.len() > 3 {
            if let Some((bdef, bblock)) = chains.get(&inst[3]) {
                if *bblock != use_block {
                    inst[3] = clone_into(&bdef.clone(), chains, use_block, bound, out, clones);
                }
            }
        }
        let id = *bound;
        *bound += 1;
        inst[2] = id;
        out.extend_from_slice(&inst);
        *clones += 1;
        id
    }
    // Pass 2: emit, planting clones before out-of-block users.
    let mut out: Vec<u32> = words[..5].to_vec();
    let mut bound = words[3];
    let (mut i, mut cur_block) = (5usize, 0u32);
    let mut clones = 0usize;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        let op = (words[i] & 0xffff) as u16;
        let mut inst = words[i..i + wc].to_vec();
        if op == OP_LABEL {
            cur_block = words[i + 1];
        } else if op != OP_PHI && cur_block != 0 {
            // ONLY the pointer-operand slots of the pointer-consuming
            // opcodes are rewritten. A blanket id sweep is NOT safe: literal
            // operands (OpExtInst's instruction number, composite indices)
            // can numerically collide with a chain id — spirv-val caught
            // exactly that on the first corpus run (cs_leaf, "invalid
            // extended instruction number"). Pointers are only ever
            // CONSUMED by this closed set, so the narrow rewrite is also
            // the complete one.
            let slots: &[usize] = match op {
                61 => &[3],          // OpLoad             ptr
                62 => &[1],          // OpStore            ptr
                63 => &[1, 2],       // OpCopyMemory       target, source
                60 => &[3],          // OpImageTexelPointer image
                65 | 66 => &[3],     // Op(InBounds)AccessChain base
                83 => &[3],          // OpCopyObject       operand
                227 => &[3],         // OpAtomicLoad       ptr
                228 => &[1],         // OpAtomicStore      ptr
                229..=242 => &[3],   // OpAtomicExchange..OpAtomicXor ptr
                _ => &[],
            };
            for &oi in slots {
                if oi >= wc {
                    continue;
                }
                let id = inst[oi];
                if let Some((def, def_block)) = chains.get(&id) {
                    if *def_block != cur_block {
                        let new_id = clone_into(
                            &def.clone(),
                            &chains,
                            cur_block,
                            &mut bound,
                            &mut out,
                            &mut clones,
                        );
                        inst[oi] = new_id;
                    }
                }
            }
        }
        out.extend_from_slice(&inst);
        i += wc;
    }
    out[3] = bound;
    out
}

/// De-SSA cross-block VALUES: every value defined in one basic block and
/// consumed in another is stored to a Function-storage variable right after
/// its definition and re-loaded beside each remote use; cross-edge OpPhi
/// incomings load in their predecessor, and cross-block phi RESULTS spill
/// the same way (stores deferred past the phi group, which SPIR-V requires
/// to stay contiguous at block start).
///
/// WHY: naga's spv-in models cross-body values by spilling them into locals
/// and CACHES the spill-load for reuse — and that cached load can be reused
/// from a body its own block does not dominate ("Expression used by a
/// statement before it was introduced into the scope by any of the
/// dominating blocks", cs_level/cs_level_wide, measured 2026-08-18; no
/// spirv-opt pass and no DXC -O level removes the triggering shape). After
/// this pass no value crosses a block at all, so naga never spills — the
/// entire bug class is unreachable. The browser driver's compiler
/// re-promotes the memory traffic to SSA, so the runtime cost is nil; the
/// WGSL text just gets more `var`s.
///
/// SAFETY RULES the implementation is built around:
/// - UNDER-rewriting is always safe (the original SSA def stays in place,
///   so an un-rewritten use still reads a valid id). Only operand slots
///   whose grammar is certain are touched — a blanket id sweep once
///   corrupted an OpExtInst literal (split_chains' history).
/// - Only types storable in a Function variable participate: pointers,
///   images, samplers, runtime arrays and any composite carrying one are
///   excluded (pointers are split_chains' job; a runtime-array-bearing
///   VALUE cannot exist in legal SPIR-V, but the pass must not lean on its
///   input being well-formed — the closure has a self_test tooth).
/// - Variables are injected PER FUNCTION: each entry block receives only
///   its own crossers' declarations. DXC's output is fully inlined (one
///   function) today, but module-wide injection would duplicate the
///   OpVariable ids the moment a module carried two — self_test pins it.
/// - OpSelectionMerge/OpLoopMerge must immediately precede their branch, so
///   re-loads feeding a terminator are emitted before the stashed merge.
/// - Emission is byte-deterministic (BTree containers) — bake outputs and
///   the W7 corpus golden depend on it.
///
/// The tooth: with this pass deleted, `--check-wgsl` fails W4 on both level
/// kernels — the gate itself is the both-ways proof, and the W2 line's
/// normalization counter is the reach proof.
pub fn spill_values(words: &[u32]) -> Vec<u32> {
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    const OP_LABEL: u16 = 248;
    const OP_FUNCTION: u16 = 54;
    const OP_TYPE_POINTER: u16 = 32;
    const OP_VARIABLE: u16 = 59;
    const OP_PHI: u16 = 245;

    // Value-producing opcodes with the (type@1, result@2) layout that can
    // legally live in a Function variable. Anything not listed is simply
    // left alone (the under-rewriting rule).
    fn has_value_result(op: u16) -> bool {
        matches!(op,
            12 | 61 | 79 | 80 | 81 | 82 | 83 | 95 | 98 | 111 | 169 | 227
            | 109..=124 | 126..=152 | 154..=168 | 170..=191 | 194..=205 | 229..=242)
    }
    // The operand slots that are KNOWN to be value ids for each opcode.
    // Literal-carrying ops (ExtInst, CompositeExtract, shuffles, image
    // operand masks) enumerate only their id slots explicitly.
    fn id_slots(op: u16, wc: usize) -> Vec<usize> {
        match op {
            12 => (5..wc).collect(),
            61 => vec![3],
            62 => vec![1, 2],
            79 => vec![3, 4],
            81 => vec![3],
            82 => vec![3, 4],
            95 | 98 => vec![3, 4],
            245 | 246 | 247 | 248 | 249 => vec![],
            250 => vec![1],
            251 => vec![1],
            254 => vec![1],
            63 => vec![1, 2],
            60 | 65 | 66 | 80 | 83 | 169 => (3..wc).collect(),
            227 | 229..=242 => (3..wc).collect(),
            228 => (1..wc).collect(),
            _ if has_value_result(op) => (3..wc).collect(),
            _ => vec![],
        }
    }

    if words.len() < 5 {
        return words.to_vec();
    }
    // Scan A: unstorable types, existing Function-storage pointer types, and
    // the type/global section's end (the insertion point for new types).
    let mut bad_ty: BTreeSet<u32> = Default::default();
    let mut ptr_fn_ty: BTreeMap<u32, u32> = Default::default();
    let mut i = 5usize;
    let mut fn_start = 5usize;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        let op = (words[i] & 0xffff) as u16;
        if wc == 0 || i + wc > words.len() {
            return words.to_vec(); // malformed: hand it to naga untouched
        }
        match op {
            // Opaque handles and pointers can appear as VALUES (loads and
            // copies of them are legal) but cannot live in a Function
            // variable. OpTypeRuntimeArray (29) joins for closure — no legal
            // op even yields such a value, but the pass must not lean on the
            // input being well-formed.
            25 | 26 | 27 | 29 | 32 => {
                bad_ty.insert(words[i + 1]);
            }
            // OpTypeArray (28) / OpTypeStruct (30): a composite is only
            // Function-storable if every member is — types are declared
            // before use, so one forward pass settles the closure. (28's
            // trailing word is a length CONSTANT id, never a type id, so
            // the over-wide member scan is inert for it.)
            28 | 30 => {
                if words[i + 2..i + wc].iter().any(|m| bad_ty.contains(m)) {
                    bad_ty.insert(words[i + 1]);
                }
            }
            OP_FUNCTION => break,
            _ => {}
        }
        if op == OP_TYPE_POINTER && words[i + 2] == 7 {
            ptr_fn_ty.insert(words[i + 3], words[i + 1]);
        }
        i += wc;
        fn_start = i;
    }
    // Scan B: defining block, type and OWNING FUNCTION of every candidate
    // value (phis too). The function index keys the variable injection: each
    // entry block gets only its own crossers' declarations (see the doc
    // comment's per-function rule).
    let mut def: HashMap<u32, (u32, u32, u32)> = Default::default();
    let (mut j, mut cur_block, mut fn_idx) = (fn_start, 0u32, 0u32);
    while j < words.len() {
        let wc = (words[j] >> 16) as usize;
        let op = (words[j] & 0xffff) as u16;
        if op == OP_FUNCTION {
            fn_idx += 1;
        } else if op == OP_LABEL {
            cur_block = words[j + 1];
        } else if (has_value_result(op) || op == OP_PHI) && cur_block != 0 && wc > 2 {
            let ty = words[j + 1];
            if !bad_ty.contains(&ty) {
                def.insert(words[j + 2], (cur_block, ty, fn_idx));
            }
        }
        j += wc;
    }
    // Scan C: which values actually cross blocks, and which phi edges need
    // a predecessor-side load.
    let mut crossers: BTreeSet<u32> = Default::default();
    let mut phi_needs: BTreeSet<(u32, u32)> = Default::default();
    let (mut j, mut cur_block) = (fn_start, 0u32);
    while j < words.len() {
        let wc = (words[j] >> 16) as usize;
        let op = (words[j] & 0xffff) as u16;
        if op == OP_LABEL {
            cur_block = words[j + 1];
        }
        if op == OP_PHI {
            let mut oi = 3;
            while oi + 1 < wc {
                let (val, pred) = (words[j + oi], words[j + oi + 1]);
                if let Some((b, _, _)) = def.get(&val) {
                    if *b != pred {
                        crossers.insert(val);
                        phi_needs.insert((pred, val));
                    }
                }
                oi += 2;
            }
        }
        for oi in id_slots(op, wc) {
            if oi >= wc {
                continue;
            }
            if let Some((b, _, _)) = def.get(&words[j + oi]) {
                if *b != cur_block {
                    crossers.insert(words[j + oi]);
                }
            }
        }
        j += wc;
    }
    if crossers.is_empty() {
        return words.to_vec();
    }
    // Allocate one Function variable per crosser (+ any missing pointer
    // types), and preassign phi-edge load ids — a loop back-edge's phi is
    // emitted before its predecessor, so the ids must exist up front.
    let mut bound = words[3];
    let mut new_types: Vec<u32> = Vec::new();
    let mut var_of: BTreeMap<u32, u32> = Default::default();
    let mut var_decls: BTreeMap<u32, Vec<u32>> = Default::default();
    for &id in &crossers {
        let (_, ty, f) = def[&id];
        let pty = *ptr_fn_ty.entry(ty).or_insert_with(|| {
            let t = bound;
            bound += 1;
            new_types.extend_from_slice(&[(4 << 16) | 32, t, 7, ty]);
            t
        });
        let v = bound;
        bound += 1;
        var_decls
            .entry(f)
            .or_default()
            .extend_from_slice(&[(4 << 16) | OP_VARIABLE as u32, pty, v, 7]);
        var_of.insert(id, v);
    }
    let mut phi_load: BTreeMap<(u32, u32), u32> = Default::default();
    for &(pred, val) in &phi_needs {
        let l = bound;
        bound += 1;
        phi_load.insert((pred, val), l);
    }
    let mut pred_loads: BTreeMap<u32, Vec<u32>> = Default::default();
    for (&(pred, val), &l) in &phi_load {
        let (_, ty, _) = def[&val];
        pred_loads
            .entry(pred)
            .or_default()
            .extend_from_slice(&[(4 << 16) | 61, ty, l, var_of[&val]]);
    }
    // Emission.
    let mut out: Vec<u32> = words[..5].to_vec();
    out.extend_from_slice(&words[5..fn_start]);
    out.extend_from_slice(&new_types);
    let (mut j, mut cur_block, mut cur_fn) = (fn_start, 0u32, 0u32);
    let mut in_entry_block = false;
    let mut pending_vars = true;
    let mut fn_seen = false;
    let mut pending_merge: Vec<u32> = Vec::new();
    let mut pending_phi_stores: Vec<u32> = Vec::new();
    while j < words.len() {
        let wc = (words[j] >> 16) as usize;
        let op = (words[j] & 0xffff) as u16;
        let mut inst = words[j..j + wc].to_vec();
        if op == OP_FUNCTION {
            fn_seen = true;
            pending_vars = true;
            in_entry_block = false;
            cur_fn += 1; // the same count Scan B keyed def by
        }
        if op == OP_LABEL {
            cur_block = words[j + 1];
            in_entry_block = fn_seen && pending_vars;
            out.extend_from_slice(&inst);
            j += wc;
            continue;
        }
        // New OpVariables join the entry block's variable group — THIS
        // function's only (the per-function rule in the doc comment).
        if in_entry_block && op != OP_VARIABLE {
            if let Some(decls) = var_decls.get(&cur_fn) {
                out.extend_from_slice(decls);
            }
            in_entry_block = false;
            pending_vars = false;
        }
        // Merges stash so the branch's re-loads can precede them.
        if op == 247 || op == 246 {
            pending_merge = inst;
            j += wc;
            continue;
        }
        if op == OP_PHI {
            let mut oi = 3;
            while oi + 1 < wc {
                if let Some(&l) = phi_load.get(&(inst[oi + 1], inst[oi])) {
                    inst[oi] = l;
                }
                oi += 2;
            }
            if let Some(&var) = var_of.get(&inst[2]) {
                pending_phi_stores.extend_from_slice(&[(3u32 << 16) | 62, var, inst[2]]);
            }
        } else if cur_block != 0 {
            if !pending_phi_stores.is_empty() {
                out.extend_from_slice(&pending_phi_stores);
                pending_phi_stores = Vec::new();
            }
            for oi in id_slots(op, wc) {
                if oi >= wc {
                    continue;
                }
                let id = inst[oi];
                if let Some(&var) = var_of.get(&id) {
                    let (db, ty, _) = def[&id];
                    if db != cur_block {
                        let l = bound;
                        bound += 1;
                        out.extend_from_slice(&[(4 << 16) | 61, ty, l, var]);
                        inst[oi] = l;
                    }
                }
            }
        }
        if matches!(op, 249 | 250 | 251) {
            if let Some(loads) = pred_loads.remove(&cur_block) {
                out.extend_from_slice(&loads);
            }
        }
        if !pending_merge.is_empty() {
            out.extend_from_slice(&pending_merge);
            pending_merge = Vec::new();
        }
        out.extend_from_slice(&inst);
        if has_value_result(op) && wc > 2 {
            if let Some(&var) = var_of.get(&words[j + 2]) {
                out.extend_from_slice(&[(3 << 16) | 62, var, words[j + 2]]);
            }
        }
        j += wc;
    }
    out[3] = bound;
    out
}

/// Decorate every storage image the module never READS as `NonReadable`.
///
/// WHY: WebGPU allows `access: "read-write"` on a storage texture for
/// exactly three formats — `r32uint`, `r32sint`, `r32float`. Everything else
/// must be write-only (or read-only). DXC emits no `NonReadable` for an
/// `RWTexture2D` it only ever stores to, so naga reads the access as
/// `LOAD | STORE`, the derived bind-group layout says `ReadWrite`, and
/// `createBindGroupLayout` refuses it — MEASURED on `cs_resolve`'s `hdr`
/// (rgba16f) the first time Stage C2 built a layout, which is what this
/// pass was written for.
///
/// The decoration is not a hint and not a workaround: it states a fact
/// about the shader that HLSL has no syntax for. `resolve.hlsl` stores to
/// `hdr` and never loads it; SPIR-V has a word for that, and DXC simply does
/// not emit it.
///
/// SOUNDNESS, and the shape of the argument matters because a WRONG
/// decoration is undefined behaviour rather than an error — spirv-val does
/// not check the claim and naga simply believes it, so nothing downstream
/// would catch one.
///
/// A storage image can only be read through four opcodes: `OpImageRead`,
/// `OpImageSparseRead`, `OpImageFetch`, and `OpImageTexelPointer` (the
/// atomic path). So the pass does not ask "did I see a read" — an
/// ENUMERATED question, which is only as good as the propagation shapes it
/// happens to model. It asks the TOTAL one: **is every occurrence of those
/// four opcodes in this module attributable to a variable I know?** Image
/// values are followed through the laundering opcodes (`OpLoad`,
/// `OpCopyObject`, `OpCopyLogical`, `OpSelect`, `OpPhi`) to a fixed point;
/// an image handed to `OpFunctionCall` or stored through `OpStore` escapes
/// analysis and marks its variable READ; and if any of the four read
/// opcodes names an image this walk cannot resolve, the pass **decorates
/// nothing at all** and hands the module on untouched. Refusing to make a
/// claim is always available and always safe: the cost is the same
/// `createBindGroupLayout` refusal that motivated the pass, which is loud.
///
/// A module that genuinely reads and writes one is left alone, and will
/// still be refused downstream — correctly, because a browser cannot run
/// it, and hiding that would move the failure to the page.
///
/// The decoration is appended at the END of the annotations section, which
/// is where SPIR-V's logical layout requires it.
pub fn mark_write_only(words: &[u32]) -> Vec<u32> {
    const OP_TYPE_IMAGE: u16 = 25;
    const OP_TYPE_POINTER: u16 = 32;
    const OP_FUNCTION_CALL: u16 = 57;
    const OP_VARIABLE: u16 = 59;
    const OP_IMAGE_TEXEL_POINTER: u16 = 60;
    const OP_LOAD: u16 = 61;
    const OP_STORE: u16 = 62;
    const OP_COPY_OBJECT: u16 = 83;
    const OP_IMAGE_FETCH: u16 = 95;
    const OP_IMAGE_READ: u16 = 98;
    const OP_SELECT: u16 = 169;
    const OP_PHI: u16 = 245;
    const OP_IMAGE_SPARSE_READ: u16 = 320;
    const OP_COPY_LOGICAL: u16 = 400;
    const OP_DECORATE: u16 = 71;
    const DECORATION_NON_READABLE: u32 = 25;
    /// `Sampled = 2` in `OpTypeImage` — "used with load/store", i.e. a
    /// storage image. (`1` is a sampled texture, `0` unknown.)
    const SAMPLED_STORAGE: u32 = 2;
    // The annotations section, by opcode: OpDecorate, OpMemberDecorate,
    // OpDecorationGroup, OpGroupDecorate, OpGroupMemberDecorate,
    // OpDecorateId, and the two string forms.
    let is_annotation =
        |op: u16| matches!(op, 71..=75 | 332 | 5632 | 5633);

    if words.len() < 5 {
        return words.to_vec();
    }
    let mut storage_img_types: std::collections::HashSet<u32> = Default::default();
    let mut storage_ptr_types: std::collections::HashSet<u32> = Default::default();
    let mut storage_vars: std::collections::HashSet<u32> = Default::default();
    let mut order: Vec<u32> = Vec::new();
    // EVERY image, not just the storage ones. A read whose image resolves to
    // a SAMPLED variable is simply not our business; a read that resolves to
    // NOTHING is the alarm, and telling those two apart is what keeps the
    // bail-out below from firing on ordinary texture work.
    let mut img_types: std::collections::HashSet<u32> = Default::default();
    let mut img_ptr_types: std::collections::HashSet<u32> = Default::default();
    let mut img_vars: std::collections::HashSet<u32> = Default::default();
    // Instruction offsets, so the propagation below can re-walk without
    // re-parsing the header each time.
    let mut insts: Vec<(u16, usize, usize)> = Vec::new();
    // Where the annotations section ends — the insertion point.
    let mut ann_end = 0usize;
    let mut saw_annotation = false;

    let mut i = 5usize;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        let op = (words[i] & 0xffff) as u16;
        if wc == 0 || i + wc > words.len() {
            return words.to_vec(); // malformed: hand it to naga untouched
        }
        let inst = &words[i..i + wc];
        if is_annotation(op) {
            saw_annotation = true;
            ann_end = i + wc;
        } else if !saw_annotation && ann_end == 0 && (19..=59).contains(&op) {
            // The FIRST type/constant/global instruction reached with no
            // annotation seen: an undecorated module's annotations section is
            // empty and ends exactly here. `ann_end == 0` is what makes this
            // first-only — advancing it per instruction would splice the
            // decoration into the middle of the module (opcode 59 is
            // OpVariable, which also appears inside function bodies). DXC
            // always decorates — every resource carries DescriptorSet and
            // Binding — so this is the total-function arm, not the expected
            // one.
            ann_end = i;
        }
        match op {
            OP_TYPE_IMAGE if wc >= 9 => {
                img_types.insert(inst[1]);
                if inst[7] == SAMPLED_STORAGE {
                    storage_img_types.insert(inst[1]);
                }
            }
            OP_TYPE_POINTER if wc >= 4 && img_types.contains(&inst[3]) => {
                img_ptr_types.insert(inst[1]);
                if storage_img_types.contains(&inst[3]) {
                    storage_ptr_types.insert(inst[1]);
                }
            }
            OP_VARIABLE if wc >= 4 && img_ptr_types.contains(&inst[1]) => {
                img_vars.insert(inst[2]);
                if storage_ptr_types.contains(&inst[1]) && storage_vars.insert(inst[2]) {
                    order.push(inst[2]);
                }
            }
            _ => {}
        }
        insts.push((op, i, wc));
        i += wc;
    }
    if ann_end == 0 {
        ann_end = words.len();
    }
    if storage_vars.is_empty() {
        return words.to_vec();
    }

    // ---- follow image VALUES to a fixed point ----
    //
    // `val_var[v]` = the storage-image variable the value `v` came from. A
    // fixed point rather than one forward pass because `OpPhi` may name a
    // value defined later in the module (a back edge), so a single pass can
    // see the use before the definition.
    let mut val_var: std::collections::HashMap<u32, u32> = Default::default();
    let src_of = |val_var: &std::collections::HashMap<u32, u32>, id: u32| -> Option<u32> {
        if img_vars.contains(&id) { Some(id) } else { val_var.get(&id).copied() }
    };
    loop {
        let before = val_var.len();
        for &(op, at, wc) in &insts {
            let inst = &words[at..at + wc];
            // (result, operands that could carry an image value)
            let (res, from): (u32, &[u32]) = match op {
                OP_LOAD | OP_COPY_OBJECT | OP_COPY_LOGICAL if wc >= 4 => (inst[2], &inst[3..4]),
                // OpSelect: result, type, condition, obj1, obj2.
                OP_SELECT if wc >= 6 => (inst[2], &inst[4..6]),
                // OpPhi: result, type, then (value, parent) pairs — the
                // values are every other word from operand 3.
                OP_PHI if wc >= 5 => (inst[2], &inst[3..]),
                _ => continue,
            };
            let step = if op == OP_PHI { 2 } else { 1 };
            for (k, &id) in from.iter().enumerate() {
                if k % step != 0 {
                    continue;
                }
                if let Some(v) = src_of(&val_var, id) {
                    val_var.insert(res, v);
                    break;
                }
            }
        }
        if val_var.len() == before {
            break;
        }
    }

    // ---- attribute every read, or decline to claim anything ----
    let mut read: std::collections::HashSet<u32> = Default::default();
    for &(op, at, wc) in &insts {
        let inst = &words[at..at + wc];
        match op {
            // The four read opcodes — the ONLY way a storage image can be
            // read. Each must resolve to an image variable this walk knows.
            // A read that resolves to a SAMPLED variable is not our business;
            // one that resolves to NOTHING means there is a path through the
            // module the analysis cannot see (an access chain into an image
            // array, a function parameter, an opcode added to a later SPIR-V
            // version), and the honest answer to that is to make no claim
            // about this module at all.
            OP_IMAGE_READ | OP_IMAGE_SPARSE_READ | OP_IMAGE_FETCH | OP_IMAGE_TEXEL_POINTER
                if wc >= 4 =>
            {
                match src_of(&val_var, inst[3]) {
                    Some(v) => {
                        read.insert(v);
                    }
                    None => return words.to_vec(),
                }
            }
            // An image that escapes into a call or into memory is no longer
            // followable — the callee may read it, so its variable is READ.
            OP_FUNCTION_CALL if wc >= 4 => {
                for &a in &inst[4..] {
                    if let Some(v) = src_of(&val_var, a) {
                        read.insert(v);
                    }
                }
            }
            OP_STORE if wc >= 3 => {
                if let Some(v) = src_of(&val_var, inst[2]) {
                    read.insert(v);
                }
            }
            _ => {}
        }
    }

    let mut add: Vec<u32> = Vec::new();
    for v in order {
        if !read.contains(&v) {
            add.extend_from_slice(&[(3 << 16) | u32::from(OP_DECORATE), v, DECORATION_NON_READABLE]);
        }
    }
    if add.is_empty() {
        return words.to_vec();
    }
    let mut out = Vec::with_capacity(words.len() + add.len());
    out.extend_from_slice(&words[..ann_end]);
    out.extend_from_slice(&add);
    out.extend_from_slice(&words[ann_end..]);
    out
}

/// The whole naga-normalization: pointers first (split_chains — a clone's
/// index operands may themselves be cross-block values), then values, then
/// the write-only decoration (which reads the instruction stream and must
/// see its final shape).
/// Applied by `--check-wgsl` before spirv-val AND by `--bake-web` before
/// writing blobs — the page's naga eats exactly what the gate validated.
pub fn normalize(words: &[u32]) -> Vec<u32> {
    mark_write_only(&spill_values(&split_chains(words)))
}

/// Parse a SPIR-V module (DXC's output).
pub fn parse_spv(words: &[u32]) -> Result<naga::Module, String> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let options = naga::front::spv::Options::default();
    naga::front::spv::parse_u8_slice(&bytes, &options)
        .map_err(|e| format!("spv-in: {}", err_chain(&e)))
}

/// Validate against the WebGPU-core capability floor.
pub fn validate(module: &naga::Module) -> Result<naga::valid::ModuleInfo, String> {
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities())
        .validate(module)
        .map_err(|e| format!("validate: {}", err_chain(&e)))
}

/// Emit WGSL text from a validated module.
pub fn emit_wgsl(module: &naga::Module, info: &naga::valid::ModuleInfo) -> Result<String, String> {
    naga::back::wgsl::write_string(module, info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|e| format!("wgsl-out: {e}"))
}

/// Parse WGSL text (the round-trip's second half, and the page's first).
pub fn parse_wgsl(src: &str) -> Result<naga::Module, String> {
    naga::front::wgsl::parse_str(src).map_err(|e| format!("wgsl-in: {e}"))
}

/// The whole chain: SPIR-V -> parse -> validate -> WGSL text -> RE-PARSE the
/// emitted text -> re-validate. The re-parse is not paranoia — the emitted
/// text is what the browser's own compiler eats, and naga's WGSL writer and
/// reader are separate code; a writer bug that emits text the reader rejects
/// is exactly the class of defect this round-trip exists to catch before a
/// browser does.
pub fn spv_to_wgsl(words: &[u32]) -> Result<String, String> {
    let module = parse_spv(words)?;
    let info = validate(&module)?;
    let text = emit_wgsl(&module, &info)?;
    let reparsed = parse_wgsl(&text).map_err(|e| format!("round-trip {e}"))?;
    validate(&reparsed).map_err(|e| format!("round-trip {e}"))?;
    Ok(text)
}

/// FNV-1a 64 — the src-side twin of `build.rs`'s `fnv1a64` (which content-
/// addresses FSR3 metallibs; `shim/ffx_fsr3_metal.mm` carries the third
/// byte-identical copy — `src/spirv.rs` documents the twin convention). W7's
/// corpus golden hashes each unit's assembled HLSL with it: dependency-free
/// and platform-stable — PROVIDED the caller strips `\r` first (checkout
/// bytes are CRLF on Windows, LF on unix CI). `self_test` pins the constants
/// so the twins cannot drift silently.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// One entry-point module's resource footprint — what the W5 audit scores
/// against [`BUDGET`] and the W7 golden records per line. Counted from naga
/// IR DECLARATIONS (`binding.is_some()`), not `GlobalUse`: WebGPU's
/// per-stage limits bind on the layout entries Stage C2 must declare, and
/// DXC strips resources an entry never reads (the trace.rs invariant), so
/// declared ≈ used — and the declared count is the one a `BindGroupLayout`
/// pays for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    pub workgroup: [u32; 3],
    pub uniform_bufs: u32,
    pub storage_bufs: u32,
    /// Sampled textures TOTAL; [`Self::buckets`] is the scene-keyed subset.
    pub sampled: u32,
    /// The `web_bucket_*` subset of `sampled` (gfx::texweb's codegen names,
    /// carried through DXC as OpName) — audited against the scene's plan,
    /// never against the fixed budget: the default scene plans 0 (it is
    /// texture-free), bistro 21, san-miguel 165, and all must stay green.
    pub buckets: u32,
    pub storage_tex: u32,
    pub samplers: u32,
    /// Σ byte sizes of `var<workgroup>` globals (HLSL groupshared).
    pub groupshared: u32,
    /// Byte span of the `Frame` uniform (trace_common.hlsli's FrameCb twin)
    /// when this module declares it — the cross-language layout pin.
    pub frame_span: Option<u32>,
    /// IR-level hostile types (binding arrays, acceleration structures, ray
    /// queries) — W6's IR half; always empty on a healthy corpus.
    pub hostile: Vec<String>,
}

/// Profile one compiled module. Asserts the corpus invariant that one
/// compiled module IS one entry point (corpus_jobs compiles per entry) — a
/// multi-entry module would silently attribute another entry's bindings.
pub fn profile(module: &naga::Module) -> Result<Profile, String> {
    if module.entry_points.len() != 1 {
        return Err(format!(
            "profile: {} entry points, want exactly 1 (one compiled module is one entry)",
            module.entry_points.len()
        ));
    }
    let gctx = module.to_ctx();
    let size_of = |ty: naga::Handle<naga::Type>| -> Result<u32, String> {
        module.types[ty]
            .inner
            .try_size(gctx)
            .ok_or_else(|| "profile: unsizable type".to_string())
    };
    let mut p = Profile {
        workgroup: module.entry_points[0].workgroup_size,
        uniform_bufs: 0,
        storage_bufs: 0,
        sampled: 0,
        buckets: 0,
        storage_tex: 0,
        samplers: 0,
        groupshared: 0,
        frame_span: None,
        hostile: Vec::new(),
    };
    // The types arena first — a hostile TYPE is hostile whether or not a
    // global carries it (naga only arenas types the module references).
    for (_, ty) in module.types.iter() {
        let bad = match &ty.inner {
            naga::TypeInner::BindingArray { .. } => Some("binding_array"),
            naga::TypeInner::AccelerationStructure { .. } => Some("acceleration_structure"),
            naga::TypeInner::RayQuery { .. } => Some("ray_query"),
            _ => None,
        };
        if let Some(b) = bad {
            p.hostile.push(format!("type {:?}: {b}", ty.name));
        }
    }
    for (_, gv) in module.global_variables.iter() {
        match gv.space {
            naga::AddressSpace::WorkGroup => p.groupshared += size_of(gv.ty)?,
            naga::AddressSpace::Uniform if gv.binding.is_some() => {
                p.uniform_bufs += 1;
                if gv.name.as_deref() == Some("Frame") {
                    p.frame_span = Some(size_of(gv.ty)?);
                }
            }
            naga::AddressSpace::Storage { .. } if gv.binding.is_some() => p.storage_bufs += 1,
            naga::AddressSpace::Handle if gv.binding.is_some() => {
                match &module.types[gv.ty].inner {
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Storage { .. }, ..
                    } => p.storage_tex += 1,
                    naga::TypeInner::Image { .. } => {
                        p.sampled += 1;
                        if gv.name.as_deref().is_some_and(|n| n.starts_with("web_bucket_")) {
                            p.buckets += 1;
                        }
                    }
                    naga::TypeInner::Sampler { .. } => p.samplers += 1,
                    // BindingArray/AccelStruct globals were already flagged
                    // by the type walk; anything else in Handle space is a
                    // construct this audit does not know — flag, never skip.
                    naga::TypeInner::BindingArray { .. }
                    | naga::TypeInner::AccelerationStructure { .. } => {}
                    other => p.hostile.push(format!(
                        "global {:?}: unclassified handle {other:?}",
                        gv.name
                    )),
                }
            }
            _ => {}
        }
    }
    Ok(p)
}

// ---------------------------------------------------------------------------
// The layout table (Stage C2's input)
//
// `profile` above COUNTS what a module declares; this NAMES it. Same walk,
// same arena, one more consumer — and deliberately the same function's
// neighbour rather than a second traversal somewhere else, because the two
// must never disagree about what a module asks for.

/// The storage-texture formats this corpus declares, in a spelling neither
/// naga's nor wgpu's.
///
/// A THIRD enum rather than either of theirs, for the reason every other
/// vocabulary type in this tree is its own (`reflect::DescKind`'s argument,
/// verbatim): this module is pure, compiles on wasm, and its output is what
/// Stage E will BAKE into the browser bundle — so it may not carry a type
/// from a backend crate, and it may not carry naga's own, which is an
/// implementation detail of the translator rather than a contract. The list
/// is deliberately short: a format the corpus does not declare is an ERROR
/// here, not a silent pass-through, because an unmapped format reaching a
/// backend is exactly the class of drift this indirection exists to catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageFmt {
    R32Float,
    Rg16Float,
    Rgba8Unorm,
    Rgba16Float,
    Rgba32Float,
}

/// What one binding IS, in the vocabulary a WebGPU bind-group-layout entry
/// needs. Every field here is a distinction WebGPU makes and Vulkan does not
/// — which is why `reflect::DescKind` cannot serve and this is not a
/// duplicate of it:
///
/// * **`Storage { read_only }`.** Vulkan has one `STORAGE_BUFFER` descriptor
///   type and spells read-only as a `NonWritable` decoration the layout never
///   sees; WebGPU makes it part of the binding TYPE, and a mismatch is a
///   pipeline-creation error. `reflect` says so in `DescKind::StorageBuffer`'s
///   own doc — it is a decoration, not a type, THERE.
/// * **`StorageTexture { fmt, .. }`.** WebGPU's layout entry names the format;
///   Vulkan's does not.
///
/// Both come from the naga IR rather than from the register class, and that
/// is the load-bearing choice: the access mode a WGSL global carries is what
/// the browser's compiler will read out of the emitted text, so deriving the
/// layout from the same module the text came from makes the two unable to
/// disagree. Deriving it from `t` vs `u` would be a SECOND statement of the
/// same fact — the register-shift rule — with nothing pinning them together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindKind {
    Uniform,
    Storage { read_only: bool },
    /// `Texture2D` / `Texture2DArray` (the WEB_TEX buckets).
    Texture { arrayed: bool },
    StorageTexture { fmt: StorageFmt, read: bool, write: bool },
    Sampler { comparison: bool },
}

/// One declared binding of one entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindDecl {
    /// The HLSL `space` — DXC maps it to the SPIR-V descriptor set, and
    /// WebGPU calls it the bind GROUP.
    pub group: u32,
    /// The shifted register number (`spirv::binding_of`'s output).
    pub binding: u32,
    /// The `OpName` naga carried through. Debug and bucket classification
    /// only — never the identity of a binding, which is `(group, binding)`.
    pub name: String,
    pub kind: BindKind,
}

/// The bindings one entry-point module declares, sorted by `(group, binding)`
/// — the table `webgpu::layout` turns into `BindGroupLayout`s and Stage E
/// bakes for the browser.
///
/// PURE, and that is the point: it names no backend type, so the browser gets
/// its layouts from data rather than from a reflector it cannot run
/// (`crate::reflect` is `cfg(any(unix, windows))` — it reads SPIR-V, which
/// the shipped page will not carry).
pub fn bindings(module: &naga::Module) -> Result<Vec<BindDecl>, String> {
    let mut out: Vec<BindDecl> = Vec::new();
    for (_, gv) in module.global_variables.iter() {
        let Some(b) = gv.binding else { continue };
        let name = gv.name.clone().unwrap_or_default();
        let what = format!("{name:?} at group {} binding {}", b.group, b.binding);
        let kind = match gv.space {
            naga::AddressSpace::Uniform => BindKind::Uniform,
            naga::AddressSpace::Storage { access } => BindKind::Storage {
                // LOAD-only is WebGPU's `read`; anything that can store (or
                // atomically update — ATOMIC implies both) is `read_write`.
                read_only: !access.intersects(naga::StorageAccess::STORE | naga::StorageAccess::ATOMIC),
            },
            naga::AddressSpace::Handle => match &module.types[gv.ty].inner {
                naga::TypeInner::Image { dim, arrayed, class } => {
                    if *dim != naga::ImageDimension::D2 {
                        return Err(format!("bindings: {what} is a {dim:?} image — the corpus is 2D only"));
                    }
                    match class {
                        naga::ImageClass::Sampled { kind, multi } => {
                            if *multi {
                                return Err(format!("bindings: {what} is multi-sampled"));
                            }
                            if *kind != naga::ScalarKind::Float {
                                return Err(format!("bindings: {what} samples {kind:?}, not Float"));
                            }
                            BindKind::Texture { arrayed: *arrayed }
                        }
                        naga::ImageClass::Storage { format, access } => BindKind::StorageTexture {
                            fmt: storage_fmt(*format).ok_or_else(|| {
                                format!("bindings: {what} declares {format:?}, which this table does not map")
                            })?,
                            read: access.contains(naga::StorageAccess::LOAD),
                            write: access.contains(naga::StorageAccess::STORE),
                        },
                        other => {
                            return Err(format!("bindings: {what} is an unsupported image class {other:?}"));
                        }
                    }
                }
                naga::TypeInner::Sampler { comparison } => BindKind::Sampler { comparison: *comparison },
                other => {
                    return Err(format!("bindings: {what} is an unbindable handle type {other:?}"));
                }
            },
            // A bound global in any other address space is not a thing SPIR-V
            // can produce — flag rather than skip (the profile walk's rule).
            other => return Err(format!("bindings: {what} is in address space {other:?}")),
        };
        out.push(BindDecl { group: b.group, binding: b.binding, name, kind });
    }
    out.sort_by_key(|d| (d.group, d.binding));
    // One module declaring a slot twice would make the layout depend on
    // iteration order. It cannot happen through DXC — but a layout builder
    // that silently took the last one is precisely the quiet-wrong-resource
    // class this whole derivation exists to remove.
    for w in out.windows(2) {
        if (w[0].group, w[0].binding) == (w[1].group, w[1].binding) {
            return Err(format!(
                "bindings: group {} binding {} declared twice ({:?} and {:?})",
                w[0].group, w[0].binding, w[0].name, w[1].name
            ));
        }
    }
    Ok(out)
}

fn storage_fmt(f: naga::StorageFormat) -> Option<StorageFmt> {
    Some(match f {
        naga::StorageFormat::R32Float => StorageFmt::R32Float,
        naga::StorageFormat::Rg16Float => StorageFmt::Rg16Float,
        naga::StorageFormat::Rgba8Unorm => StorageFmt::Rgba8Unorm,
        naga::StorageFormat::Rgba16Float => StorageFmt::Rgba16Float,
        naga::StorageFormat::Rgba32Float => StorageFmt::Rgba32Float,
        _ => return None,
    })
}

/// The browser session's resource-budget contract — the limits Stage C2's
/// wgpu device will ASK for via `required_limits`, which is what the W5
/// audit pins the corpus under. NOT the WebGPU defaults: storage buffers
/// (32 vs the default 8) and storage textures (12 vs 4) deliberately exceed
/// them — C1 measured every native adapter granting far more, and whether
/// BROWSERS grant the ask is C2's go/no-go. Rows move only with a measured
/// W5 print in hand, and C2's `ask_limits` must move in lockstep (the
/// duplicated-constants rule).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budget {
    pub storage_bufs: u32,
    pub storage_tex: u32,
    /// Sampled textures EXCLUDING `web_bucket_*` — the bucket count is
    /// scene-keyed and audited against the scene's own plan instead.
    pub sampled_fixed: u32,
    pub samplers: u32,
    pub uniform_bufs: u32,
    /// Bytes of `var<workgroup>` storage — the WebGPU default (16384) on
    /// purpose: exceeding it would make the row a new C2 ask.
    pub groupshared: u32,
    /// x·y·z per workgroup, and the per-dimension caps (WebGPU defaults).
    pub invocations: u32,
    pub workgroup_dim: [u32; 3],
    /// `Frame`'s reflected span rounded up to the 256-B CB ring stride must
    /// EQUAL this — the HLSL cbuffer twin pinned against `gfx::frame`'s
    /// ring constant (5616 B struct → 5632 B stride).
    pub frame_stride: u32,
}

// Measured worsts (procedural corpus, W5's first print, 2026-08-20): sb 22,
// st 9 (frd_temporal), fixed sampled 15 (frd_temporal), samplers 1, ub 2,
// groupshared 2060 B, Frame 5616 -> stride 5632. Every row is measured +
// headroom; the bucket count rides on top of sampled_fixed per scene.
pub const BUDGET: Budget = Budget {
    storage_bufs: 32,
    storage_tex: 12,
    sampled_fixed: 16,
    samplers: 4,
    uniform_bufs: 3,
    groupshared: 16 * 1024,
    invocations: 256,
    workgroup_dim: [256, 256, 64],
    frame_stride: crate::gfx::frame::CB_STRIDE as u32,
};

/// Score one profile against a budget; every string is a violation. Pure —
/// `self_test` plants a lowered row AND a violating count (teeth both ways).
pub fn audit_with(what: &str, p: &Profile, b: &Budget) -> Vec<String> {
    let mut v = Vec::new();
    let mut chk = |name: &str, got: u32, cap: u32| {
        if got > cap {
            v.push(format!("{what}: {name} {got} exceeds the budget {cap}"));
        }
    };
    chk("storage buffers", p.storage_bufs, b.storage_bufs);
    chk("storage textures", p.storage_tex, b.storage_tex);
    chk("fixed sampled textures", p.sampled - p.buckets, b.sampled_fixed);
    chk("samplers", p.samplers, b.samplers);
    chk("uniform buffers", p.uniform_bufs, b.uniform_bufs);
    chk("groupshared bytes", p.groupshared, b.groupshared);
    chk(
        "workgroup invocations",
        p.workgroup[0] * p.workgroup[1] * p.workgroup[2],
        b.invocations,
    );
    for i in 0..3 {
        chk("workgroup dimension", p.workgroup[i], b.workgroup_dim[i]);
    }
    drop(chk);
    if let Some(span) = p.frame_span {
        let stride = span.div_ceil(256) * 256;
        if stride != b.frame_stride {
            v.push(format!(
                "{what}: Frame span {span} B -> stride {stride}, want {} (gfx::frame::CB_STRIDE \
                 — the HLSL twin moved without the Rust side, or vice versa)",
                b.frame_stride
            ));
        }
    }
    for h in &p.hostile {
        v.push(format!("{what}: hostile construct in IR — {h}"));
    }
    v
}

pub fn audit(what: &str, p: &Profile) -> Vec<String> {
    audit_with(what, p, &BUDGET)
}

/// W6's text half: scan EMITTED WGSL for constructs outside the browser
/// core floor. Belt to `validate()`'s braces (`Capabilities::empty()`
/// refuses SUBGROUP/RAY_QUERY structurally) — this survives a naga bump
/// that widens a default. Tokens verified against naga 30's WGSL writer:
/// `enable f16;` / `enable wgpu_binding_array;` directives,
/// `binding_array<`, `ray_query`, `acceleration_structure`, and the
/// `subgroup*` builtin family. Tokenization is identifier-exact, so
/// `foo_f16` and `workgroupBarrier` never false-positive; an identifier
/// literally starting `subgroup` would flag — loud beats silent.
pub fn scan_wgsl(text: &str) -> Vec<String> {
    const EXACT: [&str; 5] =
        ["binding_array", "wgpu_binding_array", "f16", "ray_query", "acceleration_structure"];
    let mut hits = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        for tok in line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
            if !tok.is_empty() && (EXACT.contains(&tok) || tok.starts_with("subgroup")) {
                hits.push(format!("line {}: {tok}", ln + 1));
            }
        }
    }
    hits
}

/// One W7 golden entry line — byte-stable by construction (integers only,
/// no paths, no floats, no timestamps); `self_test` pins the exact text so
/// format drift is a caught defect, not silent golden churn.
pub fn golden_entry_line(entry: &str, target: &str, p: &Profile) -> String {
    format!(
        "  entry {entry} target={target} wg={}x{}x{} gs={} ub={} sb={} st={} tex={}+{} samp={} \
         frame={}",
        p.workgroup[0],
        p.workgroup[1],
        p.workgroup[2],
        p.groupshared,
        p.uniform_bufs,
        p.storage_bufs,
        p.storage_tex,
        p.sampled - p.buckets,
        p.buckets,
        p.samplers,
        p.frame_span.map(|s| s.to_string()).unwrap_or_else(|| "-".into()),
    )
}

/// W0's pure half — no DXC, no GPU, no file on disk; runs on any OS and on
/// wasm itself. Teeth both ways: the good module must survive the full
/// round-trip, AND two differently-broken modules must fail — a validator
/// that accepts everything scores nothing (the anti-vacuity rule).
pub fn self_test() -> Result<(), String> {
    // A miniature of the real corpus's shape: a storage buffer of atomics
    // (the counter idiom), a plain storage array (the queue idiom), and a
    // workgroup-sized compute entry.
    const GOOD: &str = r#"
        struct Ctr { n: array<atomic<u32>, 4> }
        @group(0) @binding(2000) var<storage, read_write> ctr: Ctr;
        @group(0) @binding(2001) var<storage, read_write> q: array<u32>;
        @compute @workgroup_size(32)
        fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
            let slot = atomicAdd(&ctr.n[0], 1u);
            if (slot < arrayLength(&q)) { q[slot] = id.x; }
        }
    "#;
    let module = parse_wgsl(GOOD).map_err(|e| format!("wgsl: good module refused: {e}"))?;
    let info = validate(&module).map_err(|e| format!("wgsl: good module invalid: {e}"))?;
    let text = emit_wgsl(&module, &info)?;
    let re = parse_wgsl(&text).map_err(|e| format!("wgsl: emitted text refused: {e}"))?;
    validate(&re).map_err(|e| format!("wgsl: emitted text invalid: {e}"))?;

    // Anti-vacuity 1: a syntax-broken module must fail the PARSER.
    const BAD_PARSE: &str = "@compute fn { this is not wgsl }";
    if parse_wgsl(BAD_PARSE).is_ok() {
        return Err("wgsl: parser accepted garbage — the front end proves nothing".into());
    }

    // Anti-vacuity 2: a module that PARSES but breaks a rule the validator
    // owns (a write through a read-only storage binding) must fail
    // VALIDATION — this is the stage the wave-ops tooth relies on, so it must
    // be shown to bite. Some naga versions reject this in the front end
    // instead; either refusal keeps the tooth, but a PASS is vacuity.
    const BAD_VALIDATE: &str = r#"
        @group(0) @binding(0) var<storage, read> ro: array<u32>;
        @compute @workgroup_size(1)
        fn cs_main() { ro[0] = 1u; }
    "#;
    match parse_wgsl(BAD_VALIDATE) {
        Err(_) => {} // refused at the front end — the tooth still bit
        Ok(m) => {
            if validate(&m).is_ok() {
                return Err(
                    "wgsl: validator accepted a write through var<storage, read> — \
                     validation proves nothing"
                        .into(),
                );
            }
        }
    }

    // split_chains, teeth both ways on a hand-assembled SPIR-V module (1.3,
    // BufferBlock-era classes — what the vulkan1.1 web arm emits). Shape: an
    // OpAccessChain defined in a selection arm whose other arm RETURNS, so
    // the arm dominates the merge (legal SSA) while naga places arm and
    // merge in different bodies — the exact spill that synthesizes a
    // pointer-typed local. RAW must FAIL validation (if it passes, naga
    // learned cross-body chains and the pass may be obsolete — re-measure);
    // SPLIT must round-trip clean.
    #[rustfmt::skip]
    let crafted: Vec<u32> = vec![
        0x0723_0203, 0x0001_0300, 0, 18, 0,
        (2 << 16) | 17, 1,                       // OpCapability Shader
        (3 << 16) | 14, 0, 1,                    // OpMemoryModel Logical GLSL450
        (5 << 16) | 15, 5, 11, 0x6e69_616d, 0,   // OpEntryPoint GLCompute %11 "main"
        (6 << 16) | 16, 11, 17, 1, 1, 1,         // OpExecutionMode LocalSize 1 1 1
        (3 << 16) | 71, 5, 3,                    // OpDecorate %5 BufferBlock
        (5 << 16) | 72, 5, 0, 35, 0,             // OpMemberDecorate %5 0 Offset 0
        (4 << 16) | 71, 7, 34, 0,                // OpDecorate %7 DescriptorSet 0
        (4 << 16) | 71, 7, 33, 0,                // OpDecorate %7 Binding 0
        (2 << 16) | 19, 1,                       // %1 = OpTypeVoid
        (3 << 16) | 33, 2, 1,                    // %2 = OpTypeFunction %1
        (4 << 16) | 21, 3, 32, 0,                // %3 = OpTypeInt 32 0
        (2 << 16) | 20, 4,                       // %4 = OpTypeBool
        (3 << 16) | 30, 5, 3,                    // %5 = OpTypeStruct %3
        (4 << 16) | 32, 6, 2, 5,                 // %6 = OpTypePointer Uniform %5
        (4 << 16) | 59, 6, 7, 2,                 // %7 = OpVariable %6 Uniform
        (4 << 16) | 32, 8, 2, 3,                 // %8 = OpTypePointer Uniform %3
        (4 << 16) | 43, 3, 9, 0,                 // %9 = OpConstant %3 0
        (3 << 16) | 41, 4, 10,                   // %10 = OpConstantTrue %4
        (5 << 16) | 54, 1, 11, 0, 2,             // %11 = OpFunction %1 None %2
        (2 << 16) | 248, 12,                     //   %12 = OpLabel (entry)
        (3 << 16) | 247, 15, 0,                  //   OpSelectionMerge %15 None
        (4 << 16) | 250, 10, 13, 14,             //   OpBranchConditional %10 %13 %14
        (2 << 16) | 248, 13,                     //   %13 = OpLabel (arm A)
        (5 << 16) | 65, 8, 16, 7, 9,             //     %16 = OpAccessChain %8 %7 %9
        (2 << 16) | 249, 15,                     //     OpBranch %15
        (2 << 16) | 248, 14,                     //   %14 = OpLabel (arm R)
        (1 << 16) | 253,                         //     OpReturn
        (2 << 16) | 248, 15,                     //   %15 = OpLabel (merge M)
        (4 << 16) | 61, 3, 17, 16,               //     %17 = OpLoad %3 %16
        (3 << 16) | 62, 16, 17,                  //     OpStore %16 %17
        (1 << 16) | 253,                         //     OpReturn
        (1 << 16) | 56,                          // OpFunctionEnd
    ];
    let raw = parse_spv(&crafted).map_err(|e| format!("wgsl: crafted module refused: {e}"))?;
    if validate(&raw).is_ok() {
        return Err("wgsl: RAW cross-body chain validated — naga may have learned the spill \
                    shape; re-measure whether split_chains is still needed"
            .into());
    }
    let split = split_chains(&crafted);
    if split[3] != crafted[3] + 2 {
        return Err(format!(
            "wgsl: split_chains cloned {} chains on the crafted module, want 2 (load + store)",
            split[3] - crafted[3]
        ));
    }
    let m = parse_spv(&split).map_err(|e| format!("wgsl: split module refused: {e}"))?;
    let info =
        validate(&m).map_err(|e| format!("wgsl: split module still invalid — the pass \
                                          did not heal the spill: {e}"))?;
    emit_wgsl(&m, &info).map_err(|e| format!("wgsl: split module wgsl-out: {e}"))?;

    // The full normalize() (split_chains + spill_values) must also leave the
    // crafted module healthy — spill_values sees the load's result cross no
    // block here, so it must change NOTHING on this input (bound stable
    // proves it does not rewrite blindly). Its firing tooth is the gate
    // itself: deleting the pass fails W4 on both level kernels.
    let norm = normalize(&crafted);
    if norm[3] != split[3] {
        return Err(format!(
            "wgsl: spill_values touched a module with no cross-block values \
             (bound {} -> {})",
            split[3], norm[3]
        ));
    }
    let m = parse_spv(&norm).map_err(|e| format!("wgsl: normalized module refused: {e}"))?;
    let info = validate(&m).map_err(|e| format!("wgsl: normalized module invalid: {e}"))?;
    emit_wgsl(&m, &info).map_err(|e| format!("wgsl: normalized module wgsl-out: {e}"))?;

    // spill_values, teeth on its two exclusion/placement rules.
    //
    // (a) PER-FUNCTION variable injection: two functions, each with one
    // value crossing an (unconditional) block edge. The pass must plant each
    // crosser's variable in ITS OWN function's entry block — module-wide
    // injection (the pre-fix behavior) duplicates the OpVariable ids, and
    // the structural count below catches that with no dependence on how
    // tolerant a downstream parser is of duplicate ids.
    #[rustfmt::skip]
    let two_fns: Vec<u32> = vec![
        0x0723_0203, 0x0001_0300, 0, 15, 0,
        (2 << 16) | 17, 1,                       // OpCapability Shader
        (3 << 16) | 14, 0, 1,                    // OpMemoryModel Logical GLSL450
        (5 << 16) | 15, 5, 5, 0x6e69_616d, 0,    // OpEntryPoint GLCompute %5 "main"
        (6 << 16) | 16, 5, 17, 1, 1, 1,          // OpExecutionMode LocalSize 1 1 1
        (2 << 16) | 19, 1,                       // %1 = OpTypeVoid
        (3 << 16) | 33, 2, 1,                    // %2 = OpTypeFunction %1
        (4 << 16) | 21, 3, 32, 0,                // %3 = OpTypeInt 32 0
        (4 << 16) | 43, 3, 4, 0,                 // %4 = OpConstant %3 0
        (5 << 16) | 54, 1, 5, 0, 2,              // %5 = OpFunction (the entry)
        (2 << 16) | 248, 6,                      //   %6 = OpLabel
        (5 << 16) | 128, 3, 7, 4, 4,             //     %7 = OpIAdd %3 %4 %4
        (2 << 16) | 249, 8,                      //     OpBranch %8
        (2 << 16) | 248, 8,                      //   %8 = OpLabel
        (5 << 16) | 128, 3, 9, 7, 4,             //     %9 = OpIAdd %3 %7 %4 (%7 crosses;
                                                 //     one crossing SLOT — reloads are
                                                 //     per operand slot, so the +5 below
                                                 //     stays exact)
        (1 << 16) | 253,                         //     OpReturn
        (1 << 16) | 56,                          // OpFunctionEnd
        (5 << 16) | 54, 1, 10, 0, 2,             // %10 = OpFunction (plain, uncalled)
        (2 << 16) | 248, 11,                     //   %11 = OpLabel
        (5 << 16) | 128, 3, 12, 4, 4,            //     %12 = OpIAdd %3 %4 %4
        (2 << 16) | 249, 13,                     //     OpBranch %13
        (2 << 16) | 248, 13,                     //   %13 = OpLabel
        (5 << 16) | 128, 3, 14, 12, 4,           //     %14 = OpIAdd %3 %12 %4 (%12 crosses)
        (1 << 16) | 253,                         //     OpReturn
        (1 << 16) | 56,                          // OpFunctionEnd
    ];
    let spilled = spill_values(&two_fns);
    // One shared pointer type + one variable and one reload per crosser.
    if spilled[3] != two_fns[3] + 5 {
        return Err(format!(
            "wgsl: spill_values on the two-function module: bound +{}, want +5",
            spilled[3] - two_fns[3]
        ));
    }
    let count_fn_vars = |m: &[u32]| {
        let (mut i, mut n) = (5usize, 0usize);
        while i < m.len() {
            let wc = (m[i] >> 16) as usize;
            if (m[i] & 0xffff) == 59 && wc == 4 && m[i + 3] == 7 {
                n += 1;
            }
            i += wc;
        }
        n
    };
    if count_fn_vars(&spilled) != 2 {
        return Err(format!(
            "wgsl: spill_values planted {} Function OpVariables on the two-function module, \
             want 2 — module-wide injection duplicates ids",
            count_fn_vars(&spilled)
        ));
    }
    let m = parse_spv(&spilled).map_err(|e| format!("wgsl: two-function module refused: {e}"))?;
    let info = validate(&m).map_err(|e| format!("wgsl: two-function module invalid: {e}"))?;
    emit_wgsl(&m, &info).map_err(|e| format!("wgsl: two-function module wgsl-out: {e}"))?;

    // (b) The Function-storability CLOSURE: a struct carrying a runtime
    // array crosses a block. Constructible only in ILLEGAL SPIR-V (no legal
    // op yields such a value) — which is exactly why the exclusion must not
    // lean on the input being well-formed. The pass must refuse it storage:
    // the module comes back UNTOUCHED. Without the composite propagation it
    // gains a pointer type + variable + reload, and post-pass spirv-val
    // would be the only thing between the bad variable and a shipped blob.
    // (Never handed to naga — the input is deliberately invalid.)
    #[rustfmt::skip]
    let rt_cross: Vec<u32> = vec![
        0x0723_0203, 0x0001_0300, 0, 12, 0,
        (2 << 16) | 19, 1,                       // %1 = OpTypeVoid
        (3 << 16) | 33, 2, 1,                    // %2 = OpTypeFunction %1
        (4 << 16) | 21, 3, 32, 0,                // %3 = OpTypeInt 32 0
        (3 << 16) | 29, 4, 3,                    // %4 = OpTypeRuntimeArray %3
        (3 << 16) | 30, 5, 4,                    // %5 = OpTypeStruct %4
        (5 << 16) | 54, 1, 6, 0, 2,              // %6 = OpFunction
        (2 << 16) | 248, 7,                      //   %7 = OpLabel
        (3 << 16) | 1, 4, 8,                     //     %8 = OpUndef %4
        (4 << 16) | 80, 5, 9, 8,                 //     %9 = OpCompositeConstruct %5 %8
        (2 << 16) | 249, 10,                     //     OpBranch %10
        (2 << 16) | 248, 10,                     //   %10 = OpLabel
        (6 << 16) | 81, 3, 11, 9, 0, 0,          //     %11 = OpCompositeExtract %3 %9 0 0
        (1 << 16) | 253,                         //     OpReturn
        (1 << 16) | 56,                          // OpFunctionEnd
    ];
    if spill_values(&rt_cross) != rt_cross {
        return Err("wgsl: spill_values allocated Function storage for a runtime-array-carrying \
                    struct — the storability closure did not fire"
            .into());
    }

    // fnv1a64 — pin the constants against build.rs's twin (and the shim's
    // third copy): the offset basis alone, then one absorbed byte.
    if fnv1a64(b"") != 0xcbf2_9ce4_8422_2325 || fnv1a64(b"a") != 0xaf63_dc4c_8601_ec8c {
        return Err("wgsl: fnv1a64 drifted from the FNV-1a-64 constants — the build.rs/shim \
                    twins no longer agree with this one"
            .into());
    }

    // W5 profile() — on the GOOD module above (already parsed + validated):
    // the counts are structural facts of its source, so any drift here is a
    // classification bug, not churn.
    let p = profile(&module).map_err(|e| format!("wgsl: profile(good): {e}"))?;
    if p.storage_bufs != 2
        || p.workgroup != [32, 1, 1]
        || p.groupshared != 0
        || (p.sampled, p.storage_tex, p.samplers, p.uniform_bufs) != (0, 0, 0, 0)
        || p.frame_span.is_some()
        || !p.hostile.is_empty()
    {
        return Err(format!("wgsl: profile(good) misclassified: {p:?}"));
    }
    // Groupshared math: 64 × vec4<f32> = 1024 B, and a Frame uniform whose
    // span the profile must report.
    const SHARED: &str = r#"
        struct FrameLike { rows: array<vec4<f32>, 351> }
        @group(0) @binding(1) var<uniform> Frame: FrameLike;
        var<workgroup> sh: array<vec4<f32>, 64>;
        @compute @workgroup_size(8, 8, 1)
        fn cs_main(@builtin(local_invocation_index) i: u32) {
            sh[i % 64u] = Frame.rows[0];
            workgroupBarrier();
        }
    "#;
    let m = parse_wgsl(SHARED).map_err(|e| format!("wgsl: shared module refused: {e}"))?;
    validate(&m).map_err(|e| format!("wgsl: shared module invalid: {e}"))?;
    let ps = profile(&m).map_err(|e| format!("wgsl: profile(shared): {e}"))?;
    if ps.groupshared != 1024 || ps.frame_span != Some(351 * 16) || ps.uniform_bufs != 1 {
        return Err(format!("wgsl: profile(shared) misclassified: {ps:?}"));
    }
    // The bucket classifier, pinned HERE because the default scene is
    // texture-free (0 buckets), so a bare CI run never exercises it via the
    // corpus — only a textured dev-box run does. This pins the name-prefix
    // half every run; whether DXC PRESERVES the OpName is what the runner's
    // scene-keyed probe (buckets > 0 ⇒ some module classified one) proves
    // on bistro per the run-list.
    const BUCKETED: &str = r#"
        @group(1) @binding(12) var web_bucket_0: texture_2d_array<f32>;
        @group(1) @binding(13) var not_a_bucket: texture_2d<f32>;
        @group(1) @binding(20) var samp: sampler;
        @group(1) @binding(30) var outv: texture_storage_2d<rgba8unorm, write>;
        @compute @workgroup_size(1)
        fn cs_main() {
            let c = textureSampleLevel(web_bucket_0, samp, vec2f(0.0), 0i, 0.0)
                  + textureSampleLevel(not_a_bucket, samp, vec2f(0.0), 0.0);
            textureStore(outv, vec2u(0u), c);
        }
    "#;
    let m = parse_wgsl(BUCKETED).map_err(|e| format!("wgsl: bucketed module refused: {e}"))?;
    validate(&m).map_err(|e| format!("wgsl: bucketed module invalid: {e}"))?;
    let pb = profile(&m).map_err(|e| format!("wgsl: profile(bucketed): {e}"))?;
    if pb.sampled != 2 || pb.buckets != 1 || pb.samplers != 1 || pb.storage_tex != 1 {
        return Err(format!("wgsl: profile(bucketed) misclassified: {pb:?}"));
    }
    // `bindings` over the same three fixtures — the layout half of the walk.
    // Sorted by (group, binding), one entry per declared global, and every
    // WebGPU-only distinction (arrayed, storage format, write-only) named.
    let bb = bindings(&m).map_err(|e| format!("wgsl: bindings(bucketed): {e}"))?;
    let want = [
        (1u32, 12u32, BindKind::Texture { arrayed: true }),
        (1, 13, BindKind::Texture { arrayed: false }),
        (1, 20, BindKind::Sampler { comparison: false }),
        (1, 30, BindKind::StorageTexture { fmt: StorageFmt::Rgba8Unorm, read: false, write: true }),
    ];
    if bb.len() != want.len() {
        return Err(format!("wgsl: bindings(bucketed) found {} decls, want 4: {bb:?}", bb.len()));
    }
    for (got, (g, b, k)) in bb.iter().zip(want) {
        if (got.group, got.binding, got.kind) != (g, b, k) {
            return Err(format!("wgsl: bindings(bucketed) mismatch at {g}/{b}: {got:?}"));
        }
    }
    // read_only is the distinction Vulkan's layout does not make and
    // WebGPU's pipeline creation rejects — pinned both ways off two modules
    // that genuinely differ, not off a constructed value.
    let bg = bindings(&module).map_err(|e| format!("wgsl: bindings(good): {e}"))?;
    if bg.len() != 2 || !bg.iter().all(|d| d.kind == BindKind::Storage { read_only: false }) {
        return Err(format!("wgsl: bindings(good) did not read as read_write storage: {bg:?}"));
    }
    const READONLY: &str = r#"
        @group(0) @binding(1002) var<storage, read> pos: array<vec4<f32>>;
        @group(0) @binding(2000) var<storage, read_write> out_: array<vec4<f32>>;
        @compute @workgroup_size(1)
        fn cs_main() { out_[0] = pos[0]; }
    "#;
    let mr = parse_wgsl(READONLY).map_err(|e| format!("wgsl: readonly module refused: {e}"))?;
    let br = bindings(&mr).map_err(|e| format!("wgsl: bindings(readonly): {e}"))?;
    if br[0].kind != (BindKind::Storage { read_only: true })
        || br[1].kind != (BindKind::Storage { read_only: false })
    {
        return Err(format!("wgsl: read vs read_write storage was not distinguished: {br:?}"));
    }
    // The uniform arm, and the ORDER: group-major, then binding.
    let bs = bindings(&parse_wgsl(SHARED).map_err(|e| e.to_string())?)
        .map_err(|e| format!("wgsl: bindings(shared): {e}"))?;
    if bs.len() != 1 || bs[0].kind != BindKind::Uniform || bs[0].name != "Frame" {
        return Err(format!("wgsl: bindings(shared) misclassified: {bs:?}"));
    }
    // `mark_write_only`, teeth both ways, over a SYNTHETIC word stream.
    //
    // Word-level rather than module-level on purpose: the pass is a linear
    // opcode walk, and a hand-built stream exercises exactly that walk with
    // no DXC, no naga and no GPU — so this tooth runs inside plain `--check`
    // on every OS and on wasm. The END-TO-END proof is elsewhere and is
    // structural: if the pass stops firing, `--check-wgpu` J6 cannot create
    // `cs_resolve`'s bind-group layout at all (a ReadWrite rgba16f storage
    // texture is not a thing WebGPU has), which is how the pass was found to
    // be necessary in the first place.
    let synth = |read: bool| {
        let mut w: Vec<u32> = vec![0x0723_0203, 0x0001_0000, 0, 20, 0];
        // OpDecorate %10 DescriptorSet 0 — the annotations section, so the
        // insertion point is after THIS and not at the top of the module.
        w.extend_from_slice(&[(4 << 16) | 71, 10, 34, 0]);
        // OpTypeImage %8 = %7, Dim2D, sampled=2 (a STORAGE image)
        w.extend_from_slice(&[(9 << 16) | 25, 8, 7, 1, 0, 0, 0, 2, 1]);
        // OpTypePointer %9 = UniformConstant %8 ; OpVariable %10 : %9
        w.extend_from_slice(&[(4 << 16) | 32, 9, 0, 8]);
        w.extend_from_slice(&[(4 << 16) | 59, 9, 10, 0]);
        if read {
            // OpLoad %11 = %10 ; OpImageRead %12 = %11
            w.extend_from_slice(&[(4 << 16) | 61, 8, 11, 10]);
            w.extend_from_slice(&[(5 << 16) | 98, 7, 12, 11, 0]);
        }
        w
    };
    let marked = mark_write_only(&synth(false));
    // The decoration lands right after the annotations section, not appended
    // at the end — SPIR-V's logical layout requires it there.
    let want: [u32; 3] = [(3 << 16) | 71, 10, 25]; // OpDecorate %10 NonReadable
    if marked.len() != synth(false).len() + 3 || marked[9..12] != want {
        return Err(format!(
            "wgsl: mark_write_only did not decorate a never-read storage image (got {:?})",
            &marked[9..marked.len().min(15)]
        ));
    }
    // And the other way: a storage image the module READS must be left
    // exactly alone. A pass that decorated it would turn a legal read into
    // undefined behaviour, which is worse than the error it exists to avoid.
    let read_arm = synth(true);
    if mark_write_only(&read_arm) != read_arm {
        return Err("wgsl: mark_write_only decorated an image the module reads".into());
    }
    // A module with no storage image at all is returned untouched (the
    // no-op arm — most of the corpus).
    let plain: Vec<u32> = vec![0x0723_0203, 0x0001_0000, 0, 20, 0, (4 << 16) | 71, 10, 34, 0];
    if mark_write_only(&plain) != plain {
        return Err("wgsl: mark_write_only touched a module with no storage image".into());
    }
    // ---- the TOTALITY arms: a read the walk cannot follow directly ----
    //
    // The pass claims something stronger than "I saw no read": that every
    // occurrence of the four read opcodes is attributable. These four
    // fixtures are that claim's teeth, and each one is a way the enumerated
    // version of this pass would have decorated an image that IS read —
    // which is undefined behaviour nothing downstream would catch.
    let tail = |t: &[u32]| {
        let mut w = synth(false);
        w.extend_from_slice(t);
        w
    };
    let decorated = |w: &[u32]| w.windows(3).any(|s| s == [(3u32 << 16) | 71, 10, 25]);
    // (a) LAUNDERED: OpLoad %11 <- %10 ; OpCopyObject %13 <- %11 ;
    //     OpImageRead %12 <- %13. The read names a copy, not the load.
    let laundered = tail(&[
        (4 << 16) | 61, 8, 11, 10,
        (4 << 16) | 83, 8, 13, 11,
        (5 << 16) | 98, 7, 12, 13, 0,
    ]);
    if decorated(&mark_write_only(&laundered)) {
        return Err("wgsl: mark_write_only decorated an image read through OpCopyObject".into());
    }
    // (b) ESCAPED: the image is handed to a function, whose body this pass
    //     does not walk. The callee may read it, so the variable is READ.
    let called = tail(&[(4 << 16) | 61, 8, 11, 10, (5 << 16) | 57, 7, 14, 15, 11]);
    if decorated(&mark_write_only(&called)) {
        return Err("wgsl: mark_write_only decorated an image passed to OpFunctionCall".into());
    }
    // (c) UNRESOLVABLE: a read whose image operand traces to nothing this
    //     walk defined (an access chain into an image array, a function
    //     parameter, an opcode from a later SPIR-V version). The honest
    //     answer is to make NO claim about the module — including about the
    //     never-read %10, which the enumerated version would have decorated.
    let opaque = tail(&[(5 << 16) | 98, 7, 12, 99, 0]);
    if mark_write_only(&opaque) != opaque {
        return Err("wgsl: mark_write_only claimed something about a module holding a read \
                    it could not attribute"
            .into());
    }
    // (d) …and the precision that keeps (c) from firing on ordinary work: a
    //     fetch from a SAMPLED image resolves, is not our business, and must
    //     NOT suppress the storage image's decoration. Without this the bail
    //     would disable the pass on every unit that reads a texture.
    let sampled = tail(&[
        // OpTypeImage %18 (sampled=1) ; OpTypePointer %19 ; OpVariable %20
        (9 << 16) | 25, 18, 7, 1, 0, 0, 0, 1, 1,
        (4 << 16) | 32, 19, 0, 18,
        (4 << 16) | 59, 19, 20, 0,
        // OpLoad %21 <- %20 ; OpImageFetch %22 <- %21
        (4 << 16) | 61, 18, 21, 20,
        (5 << 16) | 95, 7, 22, 21, 0,
    ]);
    if !decorated(&mark_write_only(&sampled)) {
        return Err("wgsl: a SAMPLED image's fetch suppressed the storage image's decoration \
                    — the bail-out is firing on ordinary texture work"
            .into());
    }

    // A format this table does not map must be an ERROR, never a silent
    // pass-through — an unmapped format reaching a backend is the drift the
    // third enum exists to catch.
    const ODDFMT: &str = r#"
        @group(0) @binding(0) var outv: texture_storage_2d<rgba8uint, write>;
        @compute @workgroup_size(1)
        fn cs_main() { textureStore(outv, vec2u(0u), vec4u(1u)); }
    "#;
    match bindings(&parse_wgsl(ODDFMT).map_err(|e| e.to_string())?) {
        Err(e) if e.contains("does not map") => {}
        Err(e) => return Err(format!("wgsl: an unmapped storage format failed wrongly: {e}")),
        Ok(v) => return Err(format!("wgsl: an unmapped storage format was accepted: {v:?}")),
    }
    // The audit, teeth both ways. The good profile passes the shipped table;
    // a PLANTED LOWERED ROW must flag it (a budget that cannot fire proves
    // nothing); a PLANTED VIOLATING COUNT must fail the shipped table.
    if !audit("good", &p).is_empty() {
        return Err(format!("wgsl: audit flagged the good profile: {:?}", audit("good", &p)));
    }
    if audit_with("good", &p, &Budget { storage_bufs: 1, ..BUDGET }).is_empty() {
        return Err("wgsl: a lowered budget row did not flag — the audit cannot fire".into());
    }
    let mut pbad = p.clone();
    pbad.storage_bufs = BUDGET.storage_bufs + 1;
    if audit("bad", &pbad).is_empty() {
        return Err("wgsl: a count over the shipped budget did not flag".into());
    }
    // The Frame pin, both ways: 5616 rounds to the ring stride (clean); one
    // byte over the stride rounds past it (flagged).
    let mut pf = p.clone();
    pf.frame_span = Some(BUDGET.frame_stride - 16);
    if !audit("frame-ok", &pf).is_empty() {
        return Err("wgsl: the in-stride Frame span flagged".into());
    }
    pf.frame_span = Some(BUDGET.frame_stride + 1);
    if audit("frame-bad", &pf).is_empty() {
        return Err("wgsl: an over-stride Frame span did not flag".into());
    }
    // A hostile IR entry must be a violation, not a footnote.
    let mut ph = p.clone();
    ph.hostile.push("type Some(\"x\"): binding_array".into());
    if audit("hostile", &ph).is_empty() {
        return Err("wgsl: an IR hostile construct did not flag".into());
    }

    // W6 scan_wgsl — the planted hostile text must flag every family; the
    // GOOD module's own emitted text and the near-misses must stay clean.
    const HOSTILE: &str = "enable f16;\nenable wgpu_binding_array;\n\
                           var x: binding_array<f32>;\nvar rq: ray_query;\n\
                           var a: acceleration_structure;\nlet s = subgroupAdd(1u);";
    if scan_wgsl(HOSTILE).len() < 5 {
        return Err(format!(
            "wgsl: the hostile text scored {} hits, want >= 5 — the scan cannot fire",
            scan_wgsl(HOSTILE).len()
        ));
    }
    if !scan_wgsl(&text).is_empty() {
        return Err(format!("wgsl: the good emitted text flagged: {:?}", scan_wgsl(&text)));
    }
    const NEAR: &str = "let foo_f16 = 1u; workgroupBarrier(); let sub_group = 2u;";
    if !scan_wgsl(NEAR).is_empty() {
        return Err(format!("wgsl: near-miss identifiers flagged: {:?}", scan_wgsl(NEAR)));
    }

    // W7 golden_entry_line — the exact text is the golden's file format;
    // drift here would churn the tracked golden silently.
    let want = "  entry cs_main target=cs_6_5 wg=32x1x1 gs=0 ub=0 sb=2 st=0 tex=0+0 samp=0 \
                frame=-";
    let got = golden_entry_line("cs_main", "cs_6_5", &p);
    if got != want {
        return Err(format!("wgsl: golden_entry_line drifted:\n  got  {got}\n  want {want}"));
    }

    Ok(())
}
