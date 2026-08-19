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

/// The whole naga-normalization: pointers first (split_chains — a clone's
/// index operands may themselves be cross-block values), then values.
/// Applied by `--check-wgsl` before spirv-val AND by `--bake-web` before
/// writing blobs — the page's naga eats exactly what the gate validated.
pub fn normalize(words: &[u32]) -> Vec<u32> {
    spill_values(&split_chains(words))
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

    Ok(())
}
