//! HLSL -> SPIR-V: the twin of `gpu/dxc.rs`, compiling the IDENTICAL
//! concatenated source that backend compiles to DXIL. Consumed by the Vulkan
//! backend directly and by the Metal one through `spirv-cross` — see "Two
//! consumers" below.
//!
//! Same footprint policy every SDK in this tree states: nothing links the
//! compiler. `libdxcompiler.so` is `dlopen`'d at run time from a gitignored
//! drop (default `SDKs/dxc-linux/lib`, fetched by
//! `install-prerequisites.sh dxc`), and the single entry point
//! `DxcCreateInstance` is resolved by symbol — the `LoadLibraryExW` +
//! `GetProcAddress` shape, spelled portably. `--check`-class gates stay
//! DLL-free because none of them come through here.
//!
//! ONE PIN, BOTH SIDES. The installer fetches the Windows zip and the Linux
//! tarball from the SAME DXC release tag, deliberately: if the compiler that
//! emits DXIL and the compiler that emits SPIR-V ever drifted to different
//! HLSL front ends, the result would be a divergence no gate in this tree
//! could see — each side green against its own compiler.
//!
//! # The flag set, and why it is exactly this
//!
//! * `-spirv` — the codegen backend. Present in the official Linux drop; no
//!   source build and no Vulkan SDK needed.
//! * `-fspv-target-env=vulkan1.3` — NOT cosmetic. DXC's SPIR-V default is
//!   vulkan1.0, which cannot express what these kernels use: wave intrinsics
//!   need SPIR-V 1.3 subgroup ops (Vulkan 1.1) and `RayQuery` needs
//!   `SPV_KHR_ray_query` (Vulkan 1.2). A too-low target env is a compile
//!   error, not a silent downgrade, but naming the real floor keeps the
//!   failure legible.
//! * `-fvk-use-dx-layout` — the reason ONE Rust packer serves both backends.
//!   `gfx::frame::FrameCb` is 5616 bytes of D3D12 constant-buffer layout;
//!   this makes Vulkan read it with the same rules instead of std140/std430,
//!   so the byte-identical-CB claim holds by construction rather than by a
//!   second packer that must be kept in sync. The price is a device feature —
//!   `scalarBlockLayout`, core in Vulkan 1.2 — which the device probe must
//!   REQUIRE rather than prefer.
//! * `-fvk-{b,t,u,s}-shift N all` — the binding scheme below.
//! * `-HV 2021` and `-O3`/`-Od -Zi` — identical to the DXIL side, because the
//!   kernels are written against HLSL 2021 (`frustum.hlsli`'s `select()`) and
//!   because differing optimization levels would make the two backends
//!   different shaders.
//!
//! # Registers -> descriptor sets
//!
//! HLSL register SPACE becomes the descriptor SET (space0 -> set 0, space1 ->
//! set 1 — `texs[]`, the scene tables and the two static samplers), and the
//! register NUMBER becomes the binding, shifted by type so that `b0`, `t0`,
//! `u0` and `s0` — all of which exist in this corpus — do not collide inside
//! one set:
//!
//! ```text
//!     b# -> SHIFT_B + #        t# -> SHIFT_T + #
//!     u# -> SHIFT_U + #        s# -> SHIFT_S + #
//! ```
//!
//! `binding_of` is that rule, and it is the ONE statement of it: the flags
//! handed to DXC are generated from the same constants the descriptor-set
//! layouts will be built from, so a shift that changed on one side and not
//! the other is unrepresentable. Do not hardcode a binding number anywhere.
//!
//! The shifts are 1000 apart because the largest register number in the
//! corpus is `u32` (`RP_GBUF_EXT`) — three orders of headroom. Raising one
//! is free; making them collide is not, and nothing would catch it in the
//! module (`spirv-val` validates a module, not a pipeline layout), only as
//! garbage reads at run time. That is why `self_test` proves the mapping is
//! injective rather than assuming it.
//!
//! # Measured
//!
//! The whole shipping corpus — **47 units → 80 modules**, assembled by
//! `gfx::shaders` exactly as a session assembles them, including all four
//! `--dxr-inline` libraries with their RayQuery/any-hit machinery, the
//! wavefront ladder, FRD's three kernels and the FSR composite — compiles
//! under this flag set and passes `spirv-val`, with zero edits to any
//! `.hlsl`. `--check-spirv` is that measurement, wired as a gate. The
//! `--sw-rays` arm reads 37 → 68, which is also the anti-vacuity proof that
//! the lever reaches the assembly (8 213 479 B vs 5 634 944 B).
//!
//! Treat those as a SNAPSHOT: the counts move with the corpus, and the header
//! said "40 units, 84 entry points" for long enough that two other places in
//! the tree disagreed with it and with each other. The gate prints the live
//! numbers; this paragraph is orientation, not a pin.
//!
//! The snapshot above replaced "47 → 78 / 37 → 66" on 2026-08-14, and the
//! delta is worth reading once because it is the shape these numbers move in:
//! ONE entry point was added (`cs_rr_emis_readd`, in `feed.hlsl`) and the count
//! rose by TWO, because `feed.hlsl` is assembled as two units — `feed[Nvidia]`
//! and `dxr-feed[0]` — while the UNIT counts, 47 and 37, did not move at all.
//! A module count is entries × the arms that paste them, so it is not a
//! triangulation of anything on its own; read it beside the unit count.
//!
//! # The count is now cross-checkable, which it was not before
//!
//! Until this module grew its Windows arm, exactly one platform ever ran the
//! gate, so "the corpus is one corpus everywhere" was an unfalsifiable claim.
//! It is now a comparison: the assembly path is free of `#[cfg]` (nothing in
//! `gfx::shaders` outside its `#[cfg(test)]` blocks) and the vendor arms are
//! ENUMERATED rather than detected (`corpus_units` walks `[Nvidia, Amd]`), so
//! the same numbers must print on every platform — and if they ever do not,
//! that is the front-end divergence this module's ONE PIN exists to prevent,
//! finally visible to something.
//!
//! VISIBLE, not yet ASSERTED: no gate compares the two platforms' counts, so
//! this is a thing a reader can check across two CI logs and not a thing that
//! fails. Wiring the count into a cross-platform pin needs a golden the two
//! jobs share, which is the same shape `--check-wgsl`'s W7 will need for the
//! browser corpus. Do not read the paragraph above as a guard that exists.
//!
//! # Two consumers, not one
//!
//! Vulkan eats this SPIR-V directly. **Metal eats it through `spirv-cross`**
//! (`mtl::msl`, gated by `--check-msl`), which is why this module is
//! `crate::spirv` rather than `vk::spirv` — it names no backend type and
//! never did. `vk::spirv` re-exports it so no call site moved.
//!
//! One consequence worth knowing before changing the shift constants below:
//! the Metal route CANNOT use them as argument indices. `--msl-decoration-
//! binding` makes the Metal index equal the SPIR-V binding, and Metal allows
//! textures 0-127 and samplers 0-15 — so `SHIFT_T`/`SHIFT_U`/`SHIFT_S` are
//! all out of bounds there. `mtl::msl` therefore drops that flag and lets
//! spirv-cross renumber per namespace; see its header for the measurement.
//! These constants remain a VULKAN choice, and are free to stay one.
//!
//! # The one platform fact, and the two arms it needs
//!
//! DXC's `WinAdapter.h` typedefs `LPCWSTR` as `const wchar_t*`, and `wchar_t`
//! is **4 bytes** off Windows against Windows' 2 — so the argument array is
//! UTF-32 there and UTF-16 here. That is the entire difference, and it is the
//! only reason this module was `cfg(unix)` until now: `WChar`, `wide()`,
//! `LIB_NAME` and `default_dir()` each carry a two-line arm and nothing else
//! does. The vtables, the CLSIDs, the flags and the binding scheme were always
//! platform-neutral.
//!
//! Two corollaries worth stating, because both would be silent if wrong:
//!
//! * **The calling convention is already right.** `STDMETHODCALLTYPE` expands
//!   to nothing off Windows, and on **x86_64** Windows `__stdcall` IS the one
//!   and only convention — so `extern "C"` is correct on both arms. It would
//!   not be on 32-bit Windows, which this tree does not target and whose DXC
//!   drop we do not fetch.
//! * **No `dxil.dll`.** `gpu/dxc.rs` loads the validator/signer because
//!   unsigned DXIL is rejected by the runtime. SPIR-V has no such step, so the
//!   Windows arm here loads exactly one library, and a tree with
//!   `dxcompiler.dll` but no `dxil.dll` still gates.
//!
//! `--check-spirv` therefore runs on Windows, which matters beyond Vulkan:
//! it is a corpus gate, so it belongs on the platform the corpus's OTHER code
//! generator (`gpu/dxc.rs`) lives on, where a divergence between the two is a
//! same-box comparison rather than a cross-CI one.

use std::ffi::c_void;

pub type Result<T> = std::result::Result<T, String>;

// ---------------------------------------------------------------------------
// The binding scheme — one statement, two consumers (the DXC flags below and
// the descriptor-set layouts the backend will build).
// ---------------------------------------------------------------------------

/// Register kinds, in HLSL's own spelling.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reg {
    /// `b#` — constant buffers.
    B,
    /// `t#` — SRVs.
    T,
    /// `u#` — UAVs.
    U,
    /// `s#` — samplers.
    S,
}

pub const SHIFT_B: u32 = 0;
pub const SHIFT_T: u32 = 1000;
pub const SHIFT_U: u32 = 2000;
pub const SHIFT_S: u32 = 3000;

/// The Vulkan binding for an HLSL register. The descriptor SET is the
/// register space and needs no mapping.
pub fn binding_of(kind: Reg, reg: u32) -> u32 {
    let shift = match kind {
        Reg::B => SHIFT_B,
        Reg::T => SHIFT_T,
        Reg::U => SHIFT_U,
        Reg::S => SHIFT_S,
    };
    shift + reg
}

/// The inverse: which register a Vulkan binding number came from.
///
/// It exists because the layout is DERIVED from compiled modules rather than
/// transcribed (see `vk::reflect`), so the only thing a reflected binding
/// carries is its number — and reading the register class back out of it is
/// what lets a gate say "this slot reflects as a storage image but its number
/// says it is a `t`". Written against the same constants as `binding_of`, and
/// `self_test` pins the round trip in both directions rather than trusting
/// that two `match`es stayed in step.
///
/// `None` for a binding outside every range, which is itself a finding: with
/// three orders of magnitude of headroom above the corpus's largest register,
/// a number out there means a shift moved.
pub fn reg_of_binding(binding: u32) -> Option<(Reg, u32)> {
    // Order matters only in that each range must be tested against its own
    // ceiling; the shifts are 1000 apart and `self_test` proves they stay so.
    for (kind, shift) in
        [(Reg::B, SHIFT_B), (Reg::T, SHIFT_T), (Reg::U, SHIFT_U), (Reg::S, SHIFT_S)]
    {
        if binding >= shift && binding - shift < SHIFT_STRIDE {
            return Some((kind, binding - shift));
        }
    }
    None
}

/// The gap between adjacent shifts — the largest register number any one
/// class may hold. The corpus's largest is `u32`, so this is three orders of
/// headroom; it is a named constant because `reg_of_binding` needs a ceiling
/// and picking one per call site is how the two halves drift.
pub const SHIFT_STRIDE: u32 = 1000;

/// The `-fvk-*-shift` flags, generated from the constants above so the
/// compiler and the layout builder cannot disagree. `all` applies the shift
/// in every space, which is what makes space0 and space1 follow one rule.
fn shift_args() -> Vec<String> {
    let mut a = Vec::new();
    for (flag, shift) in [
        ("-fvk-b-shift", SHIFT_B),
        ("-fvk-t-shift", SHIFT_T),
        ("-fvk-u-shift", SHIFT_U),
        ("-fvk-s-shift", SHIFT_S),
    ] {
        a.push(flag.to_string());
        a.push(shift.to_string());
        a.push("all".to_string());
    }
    a
}

/// Every SPIR-V-specific argument, in one place. See the header for why each
/// one is here.
pub fn spirv_args() -> Vec<String> {
    let mut a = vec![
        "-spirv".to_string(),
        "-fspv-target-env=vulkan1.3".to_string(),
        "-fvk-use-dx-layout".to_string(),
    ];
    a.extend(shift_args());
    a
}

// ---------------------------------------------------------------------------
// The DXC COM surface, hand-declared.
//
// Four vtables, taken from SDKs/dxc-linux/include/dxc/{dxcapi,WinAdapter}.h.
// `IUnknown` there declares NO virtual destructor, so its three slots sit at
// 0..2 exactly as on Windows and every derived interface appends after them
// in declaration order. Deriving these from the header rather than from
// memory is deliberate: a slot off by one is a call into the wrong function
// with plausible-looking arguments, which is not a crash you can read.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct Guid {
    d1: u32,
    d2: u16,
    d3: u16,
    d4: [u8; 8],
}

// {73E22D93-E6CE-47F3-B5BF-F0664F39C1B0}
const CLSID_DXC_COMPILER: Guid = Guid {
    d1: 0x73e2_2d93,
    d2: 0xe6ce,
    d3: 0x47f3,
    d4: [0xb5, 0xbf, 0xf0, 0x66, 0x4f, 0x39, 0xc1, 0xb0],
};
// {228B4687-5A6A-4730-900C-9702B2203F54}
const IID_IDXC_COMPILER3: Guid = Guid {
    d1: 0x228b_4687,
    d2: 0x5a6a,
    d3: 0x4730,
    d4: [0x90, 0x0c, 0x97, 0x02, 0xb2, 0x20, 0x3f, 0x54],
};
// {58346CDA-DDE7-4497-9461-6F87AF5E0659}
const IID_IDXC_RESULT: Guid = Guid {
    d1: 0x5834_6cda,
    d2: 0xdde7,
    d3: 0x4497,
    d4: [0x94, 0x61, 0x6f, 0x87, 0xaf, 0x5e, 0x06, 0x59],
};

type Hresult = i32;

#[repr(C)]
struct DxcBuffer {
    ptr: *const c_void,
    size: usize,
    encoding: u32,
}
const DXC_CP_UTF8: u32 = 65001;

#[repr(C)]
struct IUnknownVtbl {
    query_interface:
        unsafe extern "C" fn(*mut c_void, *const Guid, *mut *mut c_void) -> Hresult,
    add_ref: unsafe extern "C" fn(*mut c_void) -> u32,
    release: unsafe extern "C" fn(*mut c_void) -> u32,
}

#[repr(C)]
struct ICompiler3Vtbl {
    base: IUnknownVtbl,
    #[allow(clippy::type_complexity)]
    compile: unsafe extern "C" fn(
        *mut c_void,
        *const DxcBuffer,
        *const *const WChar,
        u32,
        *mut c_void, // IDxcIncludeHandler* — always null; there are no #includes
        *const Guid,
        *mut *mut c_void,
    ) -> Hresult,
    // Disassemble follows; unused.
}

/// `IDxcResult` derives `IDxcOperationResult`, so these three are slots 3..5
/// and the `IDxcResult`-only methods sit after them. Only the pre-`GetOutput`
/// API is used, exactly as the DXIL twin does.
#[repr(C)]
struct IResultVtbl {
    base: IUnknownVtbl,
    get_status: unsafe extern "C" fn(*mut c_void, *mut Hresult) -> Hresult,
    get_result: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> Hresult,
    get_error_buffer: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> Hresult,
}

/// `IDxcBlobEncoding` derives `IDxcBlob`, so an error blob answers these two
/// at the same slots a result blob does.
#[repr(C)]
struct IBlobVtbl {
    base: IUnknownVtbl,
    get_buffer_pointer: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    get_buffer_size: unsafe extern "C" fn(*mut c_void) -> usize,
}

/// `wchar_t` as DXC sees it. See the header — this, `LIB_NAME` and
/// `default_dir` are the whole platform delta.
#[allow(non_camel_case_types)]
#[cfg(windows)]
type WChar = u16;
#[allow(non_camel_case_types)]
#[cfg(not(windows))]
type WChar = u32;

/// The two arms are two DIFFERENT ENCODINGS, not one cast at two widths.
/// `c as u16` would silently truncate anything past the BMP into a wrong
/// character rather than the surrogate pair Windows expects; every argument we
/// pass today is ASCII, so the bug would sit unfired until the first non-ASCII
/// path reached it. Spell each encoding correctly instead.
#[cfg(windows)]
fn wide(s: &str) -> Vec<WChar> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(not(windows))]
fn wide(s: &str) -> Vec<WChar> {
    s.chars().map(|c| c as WChar).chain(std::iter::once(0)).collect()
}

/// Releases on drop. Manual refcounting across ~400 compiles in one gate run
/// is exactly the kind of leak that reads as "the compiler uses a lot of
/// memory".
struct Com(*mut c_void);

impl Com {
    fn vtbl<T>(&self) -> &T {
        unsafe { &**(self.0 as *const *const T) }
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        if !self.0.is_null() {
            let vt: &IUnknownVtbl = self.vtbl();
            unsafe { (vt.release)(self.0) };
        }
    }
}

fn blob_to_string(blob: &Com) -> String {
    let vt: &IBlobVtbl = blob.vtbl();
    unsafe {
        let ptr = (vt.get_buffer_pointer)(blob.0) as *const u8;
        let len = (vt.get_buffer_size)(blob.0);
        if ptr.is_null() || len == 0 {
            return String::new();
        }
        String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len)).into_owned()
    }
}

type CreateFn = unsafe extern "C" fn(*const Guid, *const Guid, *mut *mut c_void) -> Hresult;

// ---------------------------------------------------------------------------

/// The in-process compile memo — the "SPIR-V memo" B6b rungs 3 and 4 both
/// recommended. The compiled words depend on nothing but what reaches DXC
/// (`compile_args` builds the whole argument vector from its parameters; the
/// source arrives as an in-memory buffer with no include handler, no
/// filenames, no timestamps), and `gs::TraceKeys` carries no resolution — so
/// a window resize used to recompile 24 units to the SAME words, ~7.3 s of a
/// ~7.8 s rebuild. This map returns them instead.
///
/// KEYED ON EVERYTHING THAT REACHES DXC: (source, entry, target, debug).
/// `what` is diagnostic-only and deliberately excluded; a call carrying
/// `extra` args goes through `compile_args` directly and bypasses the memo —
/// which is also the gates' fresh-compile handle, so the bypass is a feature
/// with two names rather than a hole.
///
/// The buckets hash the full key but COMPARE the full key: a hash collision
/// that served the wrong kernel would be a wrong-resource read behind a valid
/// module — the exact failure class `self_test`'s injectivity sweep exists
/// for, reachable here through a different door. Storage is bounded by the
/// corpus (~11 distinct sources per session) and dies with the process, so
/// there is no version to bump and no lever word to carry.
///
/// `FR_SPIRV_NOMEMO=1` kills it (loud once, the `FR_*` convention) for A/B-ing
/// a suspected stale serve — though `--check-vk`'s end-of-suite hit assert and
/// `--check-spirv`'s determinism arm exist so that suspicion has a gate to
/// fall on first.
#[derive(Default)]
struct Memo {
    map: std::cell::RefCell<std::collections::HashMap<u64, Vec<(MemoKey, Vec<u32>)>>>,
    hits: std::cell::Cell<u32>,
    misses: std::cell::Cell<u32>,
}

/// (source, entry, target, debug) — owned only once stored; lookups hash and
/// compare borrowed halves so a memo HIT allocates nothing but the returned
/// words.
type MemoKey = (String, String, String, bool);

impl Memo {
    fn hash_key(src: &str, entry: &str, target: &str, debug: bool) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        src.hash(&mut h);
        entry.hash(&mut h);
        target.hash(&mut h);
        debug.hash(&mut h);
        h.finish()
    }

    fn get(&self, src: &str, entry: &str, target: &str, debug: bool) -> Option<Vec<u32>> {
        let map = self.map.borrow();
        let bucket = map.get(&Self::hash_key(src, entry, target, debug))?;
        let words = bucket
            .iter()
            .find(|((s, e, t, d), _)| s == src && e == entry && t == target && *d == debug)
            .map(|(_, w)| w.clone())?;
        self.hits.set(self.hits.get().saturating_add(1));
        Some(words)
    }

    fn put(&self, src: &str, entry: &str, target: &str, debug: bool, words: &[u32]) {
        let h = Self::hash_key(src, entry, target, debug);
        self.map
            .borrow_mut()
            .entry(h)
            .or_default()
            .push(((src.into(), entry.into(), target.into(), debug), words.to_vec()));
        self.misses.set(self.misses.get().saturating_add(1));
    }

    /// (hits, misses) — a hit is a compile that never reached DXC, a miss is
    /// a fresh compile this memo stored.
    fn stats(&self) -> (u32, u32) {
        (self.hits.get(), self.misses.get())
    }
}

/// Whether the compile memo is live — the ONE statement of the
/// `FR_SPIRV_NOMEMO` predicate, public so the gates that assert "the memo
/// fired" can exempt an armed kill lever without a second copy of the parse.
pub fn memo_enabled() -> bool {
    !memo_off()
}

/// `FR_SPIRV_NOMEMO=1` — every `compile` reaches DXC. Read once, loud once.
fn memo_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        let armed = std::env::var("FR_SPIRV_NOMEMO").is_ok_and(|v| v != "0");
        if armed {
            eprintln!("vk: FR_SPIRV_NOMEMO — the SPIR-V compile memo is OFF, every compile is fresh");
        }
        armed
    })
}

/// The SPIR-V compiler. One per process is plenty; `Compile` is reentrant but
/// this holds the library alive and must outlive every blob it produced.
pub struct Spirv {
    // Dropped last (field order): the compiler object lives inside this
    // library's image, so releasing it after an unload would be a call into
    // unmapped memory. The memo between them is plain data and order-blind.
    compiler: Com,
    memo: Memo,
    _lib: libloading::Library,
}

/// The host-native DXC library's filename, which is a fact about the TARGET
/// and not about the host: a Linux drop is `libdxcompiler.so`, a macOS build
/// `libdxcompiler.dylib`, a Windows one `dxcompiler.dll`. One const, read by
/// both the path join and the advice text, so a message can never name a file
/// the loader did not try.
pub const LIB_NAME: &str = if cfg!(windows) {
    "dxcompiler.dll"
} else if cfg!(target_os = "macos") {
    "libdxcompiler.dylib"
} else {
    "libdxcompiler.so"
};

/// Where the host-native drop lives, overridable with
/// `FRUSTRACER_DXC_SPIRV_PATH`.
///
/// Off Windows this is deliberately NOT `--dxc-path`/`FRUSTRACER_DXC_PATH`,
/// which names the WINDOWS drop (`dxcompiler.dll` + `dxil.dll`). The installer
/// fetches both from one release tag precisely so the two compilers cannot
/// drift, but they are two artifacts in two directories and one path variable
/// cannot name both — pointing the D3D12 lever at a `.so` would be the kind of
/// quiet mis-wiring that surfaces as "SPIR-V is unavailable" on a tree that has
/// it.
///
/// TWO DIRECTORIES, ONE PIN, because the two arms are ACQUIRED differently and
/// the pin is what makes that safe. Linux has an upstream tarball at `DXC_TAG`;
/// macOS has nothing published at all (`DXC_TAG` ships a Windows zip, a Linux
/// x86_64 tarball and a PDB zip — no macOS build, no arm64 build of anything),
/// so the drop there is a SOURCE BUILD at the same tag. A community binary
/// would not be from the tag, and the invariant this module's header states —
/// the two backends compile identical concatenated source, so a front-end
/// difference is invisible to every gate — is exactly what that would break.
///
/// ON WINDOWS THAT WHOLE ARGUMENT COLLAPSES, and the code has to say so rather
/// than inherit a caution that no longer applies: there is ONE drop, and the
/// `dxcompiler.dll` that emits DXIL for `gpu/dxc.rs` is the same file that
/// emits SPIR-V here. Two directories was never the point — ONE PIN was, and
/// on this platform the pin is trivially held because it is one artifact. So
/// the `FRUSTRACER_DXC_PATH` ENV VAR is honoured as the second-choice source:
/// pointing the D3D12 lever at a Windows DXC drop and having SPIR-V not find it
/// would be the mis-wiring, in the opposite direction.
///
/// HALF a lever, and deliberately so for now: `--dxc-path` is a FLAG, and
/// `cli.rs` only reads the env var to seed its DEFAULT — the flag never writes
/// the env var back, so a session started with `--dxc-path D:\dxc` still finds
/// SPIR-V at the built-in path. `default_dir()` takes no `Opts` (its three
/// gate call sites have none to give it), so closing that needs the dir
/// threaded from `run_check_spirv`'s caller rather than a wider `var()` here.
/// The advice text in `load` says which of the two actually works; do not
/// widen this doc to promise the flag until the plumbing exists.
///
/// `FRUSTRACER_DXC_SPIRV_PATH` still wins where set, so the escape hatch for
/// "compile SPIR-V with a different DXC than the one signing my DXIL" survives
/// on every platform — which is what a front-end-divergence bisect would need.
pub fn default_dir() -> String {
    if let Ok(d) = std::env::var("FRUSTRACER_DXC_SPIRV_PATH") {
        return d;
    }
    #[cfg(windows)]
    {
        std::env::var("FRUSTRACER_DXC_PATH")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), r"\SDKs\dxc\bin\x64").to_string())
    }
    #[cfg(not(windows))]
    {
        let sub = if cfg!(target_os = "macos") { "dxc-macos" } else { "dxc-linux" };
        format!("{}/SDKs/{sub}/lib", env!("CARGO_MANIFEST_DIR"))
    }
}

impl Spirv {
    /// Load `LIB_NAME` from `dir` and create the compiler. A missing drop is a
    /// normal condition (the SDK tree is gitignored) — the error names the fix,
    /// the same way the DXIL loader's does.
    pub fn load(dir: &str) -> Result<Self> {
        let path = std::path::Path::new(dir).join(LIB_NAME);
        let advice = if cfg!(windows) {
            format!(
                "The corpus's SPIR-V arm needs the SAME {LIB_NAME} the DXIL arm already\n\
                 uses — one file, one drop, no second download. Run\n\
                 install-prerequisites.bat dxc, or point FRUSTRACER_DXC_SPIRV_PATH\n\
                 (or the FRUSTRACER_DXC_PATH env var, which this falls back to — the\n\
                 --dxc-path FLAG does NOT reach here) at a directory holding it.\n\
                 dxil.dll is NOT needed here: it signs DXIL, and SPIR-V has no\n\
                 signing step."
            )
        } else if cfg!(target_os = "macos") {
            format!(
                "The Vulkan backend needs the DirectX Shader Compiler, and upstream\n\
                 publishes no macOS build at the pinned tag — so it is built from\n\
                 SOURCE at that tag. Run `./install-prerequisites.sh dxc`, which does\n\
                 the whole build (needs cmake + ninja + git and Xcode's clang; ~10 min\n\
                 on an M1), or point FRUSTRACER_DXC_SPIRV_PATH at a directory holding\n\
                 {LIB_NAME} (NOT FRUSTRACER_DXC_PATH — that one names the Windows drop)."
            )
        } else {
            format!(
                "The Vulkan backend needs the DirectX Shader Compiler's Linux build.\n\
                 Run `./install-prerequisites.sh dxc` (it fetches the Linux tarball and\n\
                 the Windows zip from the SAME release tag), or point\n\
                 FRUSTRACER_DXC_SPIRV_PATH at a directory holding {LIB_NAME}\n\
                 (NOT FRUSTRACER_DXC_PATH — that one names the Windows drop)."
            )
        };
        // SAFETY: loading a shared object runs its initializers. This one is a
        // compiler with no global side effects of interest.
        let lib = unsafe { libloading::Library::new(&path) }
            .map_err(|e| format!("failed to load {}: {e}\n{advice}", path.display()))?;
        let create: libloading::Symbol<CreateFn> =
            unsafe { lib.get(b"DxcCreateInstance\0") }
                .map_err(|e| format!("{}: missing export DxcCreateInstance: {e}", path.display()))?;
        let create = *create;
        let mut raw: *mut c_void = std::ptr::null_mut();
        let hr = unsafe { create(&CLSID_DXC_COMPILER, &IID_IDXC_COMPILER3, &mut raw) };
        if hr < 0 || raw.is_null() {
            return Err(format!("DxcCreateInstance(DxcCompiler) failed: 0x{hr:08x}"));
        }
        Ok(Self { compiler: Com(raw), memo: Memo::default(), _lib: lib })
    }

    /// (hits, misses) since load — the instrument `--check-vk`'s end-of-suite
    /// assert and the window's rebuild split line read. A hit is a compile
    /// that never reached DXC; a miss is a fresh compile the memo stored.
    pub fn memo_stats(&self) -> (u32, u32) {
        self.memo.stats()
    }

    /// Compile HLSL to SPIR-V, MEMOIZED on (src, entry, target, debug) — see
    /// `Memo` for why that key is complete and what `FR_SPIRV_NOMEMO` does.
    /// A caller that needs a provably fresh compile calls `compile_args` with
    /// `&[]`, which never touches the memo — that is what `--check-spirv`'s
    /// determinism arm and `--check-vk`'s hit-fidelity check do.
    ///
    /// `what` names the unit in errors; an empty `entry` omits `-E`, which a
    /// `lib_*` target requires (a DXR library exports every `[shader(...)]`
    /// entry, and naming one is a compile error).
    ///
    /// Returns SPIR-V WORDS, not bytes — `vkCreateShaderModule` takes
    /// `*const u32` and requires 4-byte alignment, which a `Vec<u8>` does not
    /// guarantee.
    pub fn compile(
        &self,
        src: &str,
        entry: &str,
        target: &str,
        what: &str,
        debug: bool,
    ) -> Result<Vec<u32>> {
        if memo_off() {
            return self.compile_args(src, entry, target, what, debug, &[]);
        }
        if let Some(words) = self.memo.get(src, entry, target, debug) {
            return Ok(words);
        }
        let words = self.compile_args(src, entry, target, what, debug, &[])?;
        self.memo.put(src, entry, target, debug, &words);
        Ok(words)
    }

    /// `compile` with per-unit extra arguments appended after the shared set —
    /// the FRD kernels' `-enable-16bit-types`, matching the DXIL twin. NEVER
    /// memoized (the `extra` args are not in the memo key, so caching here
    /// would serve an `-enable-16bit-types` module to a plain request), which
    /// doubles as the fresh-compile handle the gates need.
    pub fn compile_args(
        &self,
        src: &str,
        entry: &str,
        target: &str,
        what: &str,
        debug: bool,
        extra: &[&str],
    ) -> Result<Vec<u32>> {
        let mut args: Vec<String> = vec!["-T".into(), target.into(), "-HV".into(), "2021".into()];
        if !entry.is_empty() {
            args.extend(["-E".into(), entry.into()]);
        }
        if debug {
            args.extend(["-Od".into(), "-Zi".into()]);
        } else {
            args.push("-O3".into());
        }
        args.extend(spirv_args());
        args.extend(extra.iter().map(|a| a.to_string()));

        let wide_args: Vec<Vec<WChar>> = args.iter().map(|a| wide(a)).collect();
        let ptrs: Vec<*const WChar> = wide_args.iter().map(|w| w.as_ptr()).collect();

        let buf = DxcBuffer {
            ptr: src.as_ptr() as *const c_void,
            size: src.len(),
            encoding: DXC_CP_UTF8,
        };

        let vt: &ICompiler3Vtbl = self.compiler.vtbl();
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `buf` and `ptrs` outlive the call; DXC copies what it keeps.
        let hr = unsafe {
            (vt.compile)(
                self.compiler.0,
                &buf,
                ptrs.as_ptr(),
                ptrs.len() as u32,
                std::ptr::null_mut(),
                &IID_IDXC_RESULT,
                &mut raw,
            )
        };
        if hr < 0 || raw.is_null() {
            return Err(format!("DXC Compile({what}) failed: 0x{hr:08x}"));
        }
        let result = Com(raw);
        let rvt: &IResultVtbl = result.vtbl();

        let mut status: Hresult = 0;
        unsafe { (rvt.get_status)(result.0, &mut status) };

        let mut eraw: *mut c_void = std::ptr::null_mut();
        unsafe { (rvt.get_error_buffer)(result.0, &mut eraw) };
        let errors = if eraw.is_null() { String::new() } else { blob_to_string(&Com(eraw)) };

        if status < 0 {
            return Err(format!("DXC({what}) failed:\n{errors}"));
        }
        if !errors.trim().is_empty() {
            eprintln!("vk: DXC({what}) warnings:\n{errors}");
        }

        let mut braw: *mut c_void = std::ptr::null_mut();
        unsafe { (rvt.get_result)(result.0, &mut braw) };
        if braw.is_null() {
            return Err(format!("DXC({what}): no output blob"));
        }
        let blob = Com(braw);
        let bvt: &IBlobVtbl = blob.vtbl();
        let (ptr, len) = unsafe {
            ((bvt.get_buffer_pointer)(blob.0) as *const u8, (bvt.get_buffer_size)(blob.0))
        };
        if ptr.is_null() || len == 0 {
            return Err(format!("DXC({what}): empty output blob"));
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        to_words(bytes).map_err(|e| format!("DXC({what}): {e}"))
    }
}

/// SPIR-V's magic number, little-endian as every toolchain in this tree emits
/// it.
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Byte blob -> SPIR-V words, structurally checked on the way through.
///
/// This is a WIRING check, not a validator (`spirv-val` is that, and
/// `--check-spirv` runs it when the drop is present): what it catches is a
/// truncated, empty, or byte-swapped blob — the failure modes that would
/// otherwise surface inside `vkCreateShaderModule` as a driver-specific
/// message, or worse, as undefined behaviour on a length the loader trusted.
pub fn to_words(bytes: &[u8]) -> std::result::Result<Vec<u32>, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!("SPIR-V blob is {} bytes, not a whole number of words", bytes.len()));
    }
    if bytes.len() < 20 {
        return Err(format!("SPIR-V blob is {} bytes; the header alone is 20", bytes.len()));
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words[0] != SPIRV_MAGIC {
        // A byte-swapped module is a REAL possibility the spec allows and a
        // driver rejects, so name it rather than printing a bare mismatch.
        let swapped = words[0].swap_bytes() == SPIRV_MAGIC;
        return Err(format!(
            "bad SPIR-V magic 0x{:08x} (expected 0x{SPIRV_MAGIC:08x}{})",
            words[0],
            if swapped { "; the blob is byte-swapped" } else { "" }
        ));
    }
    Ok(words)
}

/// A compute kernel's workgroup size, recovered from `OpExecutionMode <entry>
/// LocalSize x y z`.
///
/// # Why a HOST-side parse exists at all
///
/// D3D12 and Vulkan both take the group size from the bytecode: `[numthreads]`
/// rides the DXIL, and `vkCmdDispatch` counts GROUPS whose shape the module
/// already declared. **Metal does not.** `dispatchThreadgroups:
/// threadsPerThreadgroup:` — and the indirect form with it — take that shape as
/// a CPU argument, because MSL has no `[numthreads]` to carry it. So a Metal
/// dispatch of a corpus kernel has to be told, and this is where it is told
/// from.
///
/// # `Err` on a miss, never a `[1, 1, 1]` fallback
///
/// That value is indistinguishable from a real 1x1x1 kernel — nonzero, product
/// well under Metal's 1024 ceiling, so every downstream sanity check accepts it
/// — and dispatching with it runs one thread per threadgroup: correct-looking
/// output at 1/64 the rate for `cs_fill`, with no error anywhere. A miss is
/// always a defect, and the realistic trigger has a name, so the message says
/// it: `OpExecutionModeId` + `LocalSizeId` (a specialization-constant group
/// size), which this deliberately does not decode because nothing in this
/// corpus emits one.
///
/// # Disagreement is a finding, not a first match
///
/// Every compute unit here is compiled one entry point at a time
/// (`main.rs::corpus_jobs` yields one job per entry), so a well-formed module
/// declares exactly one `LocalSize`. Returning the first and walking on would
/// silently pick an arbitrary kernel's shape out of a module that somehow
/// carried two; requiring them to AGREE costs one comparison and turns that
/// into a diagnosis.
///
/// # The twin
///
/// `build.rs::spirv_local_size` is a byte-for-byte equivalent of this walk, and
/// is deliberately NOT a call to it: a build script is its own compilation unit
/// and cannot `use crate::spirv`. The tree already accepts documented twins
/// where the language forbids sharing — `build.rs::fnv1a64` names its own in
/// `shim/ffx_fsr3_metal.mm` — so both sides carry a comment naming the other.
/// Change one, change both.
pub fn local_size(words: &[u32]) -> std::result::Result<[u32; 3], String> {
    /// `OpExecutionMode`. `OpExecutionModeId` (331) is deliberately not read;
    /// see the module doc.
    const OP_EXECUTION_MODE: u16 = 16;
    const EXEC_MODE_LOCAL_SIZE: u32 = 17;

    if words.len() < 5 || words[0] != SPIRV_MAGIC {
        return Err("local_size: not a SPIR-V module (call to_words first)".into());
    }
    let mut found: Option<[u32; 3]> = None;
    // Past the 5-word header, stepping every instruction by its own declared
    // word count — which is what keeps this total over SPIR-V versions and
    // opcodes it has never seen (`vk::reflect`'s discipline).
    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        // A zero count cannot advance, so it is a malformed stream rather than
        // an unknown opcode: stop instead of looping forever.
        if count == 0 || i + count > words.len() {
            break;
        }
        if (words[i] & 0xffff) as u16 == OP_EXECUTION_MODE
            && count >= 6
            && words[i + 2] == EXEC_MODE_LOCAL_SIZE
        {
            let ls = [words[i + 3], words[i + 4], words[i + 5]];
            match found {
                Some(prev) if prev != ls => {
                    return Err(format!(
                        "local_size: module declares two different LocalSize modes, \
                         {prev:?} and {ls:?} — it carries more than one entry point"
                    ));
                }
                _ => found = Some(ls),
            }
        }
        i += count;
    }
    found.ok_or_else(|| {
        "local_size: no OpExecutionMode LocalSize (a LocalSizeId group size is not decoded)"
            .to_string()
    })
}

/// Pure gate: the binding scheme, with the property that actually matters.
///
/// `spirv-val` validates a MODULE and knows nothing about pipeline layouts, so
/// a shift table that mapped two different registers onto one binding would
/// pass every compile in the corpus and then read the wrong resource at run
/// time. The injectivity sweep is the tooth against that.
pub fn self_test() -> std::result::Result<(), String> {
    // Injective over the whole register range this corpus can reach, with an
    // order of magnitude of headroom past `u32`, the largest register in use.
    let kinds = [Reg::B, Reg::T, Reg::U, Reg::S];
    let mut seen = std::collections::HashMap::new();
    for k in kinds {
        for r in 0..512u32 {
            let b = binding_of(k, r);
            if let Some(prev) = seen.insert(b, (k, r)) {
                return Err(format!("binding {b} collides: {prev:?} and {:?}", (k, r)));
            }
        }
    }

    // The inverse must undo the map EXACTLY, over the same range. Two
    // independent `match`es over four constants is precisely the shape that
    // stays correct until someone edits one of them, and the failure is
    // silent: a layout built from a mis-inverted binding is a wrong-resource
    // read with a valid module and a happy `spirv-val`.
    for k in kinds {
        for r in 0..512u32 {
            match reg_of_binding(binding_of(k, r)) {
                Some((k2, r2)) if k2 == k && r2 == r => {}
                other => {
                    return Err(format!("reg_of_binding(binding_of({k:?},{r})) = {other:?}"))
                }
            }
        }
    }
    // The shifts must stay a stride apart, or the ranges overlap and the
    // inverse silently attributes a register to the wrong class.
    for (a, b) in [(SHIFT_B, SHIFT_T), (SHIFT_T, SHIFT_U), (SHIFT_U, SHIFT_S)] {
        if b - a != SHIFT_STRIDE {
            return Err(format!("shifts {a} and {b} are not SHIFT_STRIDE apart"));
        }
    }
    // Above the last range there is nothing to attribute a binding to, and
    // saying so is what makes a moved shift a finding rather than a misread.
    if reg_of_binding(SHIFT_S + SHIFT_STRIDE).is_some() {
        return Err("a binding past every range was attributed to a register".into());
    }

    // The flags must be GENERATED from the same constants — a hand-typed
    // shift in the argument list is the drift this exists to prevent.
    let args = spirv_args();
    for (flag, shift) in [
        ("-fvk-b-shift", SHIFT_B),
        ("-fvk-t-shift", SHIFT_T),
        ("-fvk-u-shift", SHIFT_U),
        ("-fvk-s-shift", SHIFT_S),
    ] {
        let i = args
            .iter()
            .position(|a| a == flag)
            .ok_or_else(|| format!("spirv_args is missing {flag}"))?;
        if args.get(i + 1).map(String::as_str) != Some(shift.to_string().as_str()) {
            return Err(format!("{flag} does not carry its constant {shift}"));
        }
        if args.get(i + 2).map(String::as_str) != Some("all") {
            return Err(format!("{flag} must apply to every space, not one"));
        }
    }
    for want in ["-spirv", "-fvk-use-dx-layout"] {
        if !args.iter().any(|a| a == want) {
            return Err(format!("spirv_args is missing {want}"));
        }
    }
    // The target env is a floor, not a preference: below 1.2 RayQuery cannot
    // be expressed at all.
    if !args.iter().any(|a| a.starts_with("-fspv-target-env=vulkan1.")) {
        return Err("spirv_args must pin a Vulkan target env".into());
    }

    // to_words' teeth, both directions.
    let mut good = SPIRV_MAGIC.to_le_bytes().to_vec();
    good.extend(std::iter::repeat_n(0u8, 16));
    if to_words(&good)?.len() != 5 {
        return Err("to_words dropped words from a well-formed header".into());
    }
    if to_words(&good[..good.len() - 1]).is_ok() {
        return Err("to_words accepted a partial word".into());
    }
    if to_words(&good[..16]).is_ok() {
        return Err("to_words accepted a blob shorter than the header".into());
    }
    let swapped: Vec<u8> = SPIRV_MAGIC
        .to_be_bytes()
        .iter()
        .copied()
        .chain(std::iter::repeat_n(0u8, 16))
        .collect();
    match to_words(&swapped) {
        Ok(_) => return Err("to_words accepted a byte-swapped module".into()),
        Err(e) if !e.contains("byte-swapped") => {
            return Err(format!("to_words did not diagnose the byte swap: {e}"))
        }
        Err(_) => {}
    }

    // local_size, on synthetic modules. The REAL corpus is parsed by
    // `--check-mtl`'s K-stages, which have a compiler; what belongs here is the
    // walk's own behaviour, including the two ways it must refuse.
    let hdr = |body: &[u32]| -> Vec<u32> {
        let mut w = vec![SPIRV_MAGIC, 0x0001_0600, 0, 1, 0];
        w.extend_from_slice(body);
        w
    };
    // OpExecutionMode <entry> LocalSize x y z — 6 words including the opcode.
    let exec_mode = |x: u32, y: u32, z: u32| -> [u32; 6] { [(6 << 16) | 16, 1, 17, x, y, z] };

    match local_size(&hdr(&exec_mode(64, 1, 1))) {
        Ok([64, 1, 1]) => {}
        other => return Err(format!("local_size on a plain module = {other:?}")),
    }
    // An unknown opcode must be stepped over by its OWN word count, not by a
    // guess — the property that keeps this total over instructions it has never
    // seen. A 4-word filler whose payload deliberately contains the LocalSize
    // pattern would be misread by a scan that walked word by word.
    let mut hidden = vec![(4u32 << 16) | 999, 1, 17, 8];
    hidden.extend_from_slice(&exec_mode(64, 1, 1));
    match local_size(&hdr(&hidden)) {
        Ok([64, 1, 1]) => {}
        other => return Err(format!("local_size did not step over an unknown opcode: {other:?}")),
    }
    // A miss must NAME the realistic cause rather than reporting a bare absence.
    match local_size(&hdr(&[])) {
        Err(e) if e.contains("LocalSizeId") => {}
        other => return Err(format!("local_size on a module with no LocalSize = {other:?}")),
    }
    // Two entry points that disagree is a diagnosis, not a first match.
    let mut two = exec_mode(64, 1, 1).to_vec();
    two.extend_from_slice(&exec_mode(8, 8, 1));
    match local_size(&hdr(&two)) {
        Err(e) if e.contains("two different LocalSize") => {}
        other => return Err(format!("local_size accepted disagreeing LocalSize modes: {other:?}")),
    }
    // Two that AGREE are not an error — the refusal is about ambiguity, not
    // about the count.
    let mut same = exec_mode(64, 1, 1).to_vec();
    same.extend_from_slice(&exec_mode(64, 1, 1));
    if local_size(&hdr(&same)) != Ok([64, 1, 1]) {
        return Err("local_size refused two agreeing LocalSize modes".into());
    }
    // A zero word count cannot advance the cursor. This must TERMINATE — a
    // walk that trusted the count would hang here, and a hung gate reads
    // exactly like a slow one.
    if local_size(&hdr(&[0, 0, 0])).is_ok() {
        return Err("local_size read a size out of a malformed instruction stream".into());
    }
    if local_size(&[SPIRV_MAGIC]).is_ok() {
        return Err("local_size accepted a blob with no header".into());
    }

    // The memo, on synthetic data — the DXC-free half of its claim. Hit
    // fidelity (the words that come back are the words that went in), the
    // accounting the gates read, and FULL-KEY discrimination: a memo that
    // keyed on less than every element serves kernel A to a request for
    // kernel B — a wrong-resource read behind a valid module, the same
    // failure class the injectivity sweep above guards through another door.
    // The DXC half (a hit byte-equals a fresh compile) needs a compiler and
    // lives in --check-spirv's determinism arm and --check-vk's end-of-suite
    // assert.
    let m = Memo::default();
    if m.get("src", "e", "cs_6_5", false).is_some() {
        return Err("memo hit on an empty map".into());
    }
    m.put("src", "e", "cs_6_5", false, &[SPIRV_MAGIC, 1, 2, 3]);
    match m.get("src", "e", "cs_6_5", false) {
        Some(w) if w == [SPIRV_MAGIC, 1, 2, 3] => {}
        other => return Err(format!("memo returned {other:?} for a stored key")),
    }
    for (s, e, t, d) in [
        ("src2", "e", "cs_6_5", false),
        ("src", "e2", "cs_6_5", false),
        ("src", "e", "cs_6_0", false),
        ("src", "e", "cs_6_5", true),
    ] {
        if m.get(s, e, t, d).is_some() {
            return Err(format!(
                "memo served ({s:?}, {e:?}, {t:?}, debug={d}) from a key differing in one element"
            ));
        }
    }
    // Two entries under one map do not shadow each other.
    m.put("src2", "e", "cs_6_5", false, &[9]);
    if m.get("src2", "e", "cs_6_5", false) != Some(vec![9])
        || m.get("src", "e", "cs_6_5", false) != Some(vec![SPIRV_MAGIC, 1, 2, 3])
    {
        return Err("memo entries shadow each other".into());
    }
    // hits: the three successful gets above; misses: the two puts. The four
    // discrimination probes bump NEITHER — a miss is a stored compile, not a
    // failed lookup, so the pair reconciles against work done rather than
    // questions asked.
    if m.stats() != (3, 2) {
        return Err(format!("memo stats {:?}, want (3, 2)", m.stats()));
    }
    Ok(())
}
