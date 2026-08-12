//! HLSL -> SPIR-V for the Vulkan backend: the twin of `gpu/dxc.rs`, compiling
//! the IDENTICAL concatenated source that backend compiles to DXIL.
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
//! The whole shipping corpus — 40 units, 84 entry points, assembled by
//! `gfx::shaders` exactly as a session assembles them, including all four
//! `--dxr-inline` libraries with their RayQuery/any-hit machinery, the
//! wavefront ladder, FRD's three kernels and the FSR composite — compiles
//! under this flag set and passes `spirv-val`, with zero edits to any
//! `.hlsl`. `--check-spirv` is that measurement, wired as a gate.
//!
//! # The one platform fact that makes this module unix-only
//!
//! DXC's `WinAdapter.h` typedefs `LPCWSTR` as `const wchar_t*`, and `wchar_t`
//! is **4 bytes** here against Windows' 2 — so the argument array is UTF-32,
//! not UTF-16. That is the entire difference. Vulkan-on-Windows needs
//! `wide()` to emit `u16`, the library name flipped to `dxcompiler.dll`, and
//! the `cfg(unix)` on the module and the `libloading` dependency lifted;
//! everything else here — the vtables, the CLSIDs, the flags, the binding
//! scheme — is already platform-neutral. `STDMETHODCALLTYPE` expands to
//! nothing on this platform, so the methods are plain `extern "C"`.

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

/// `wchar_t` on this platform. See the header — this is the one difference
/// from the Windows arm.
#[allow(non_camel_case_types)]
type WChar = u32;

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

/// The SPIR-V compiler. One per process is plenty; `Compile` is reentrant but
/// this holds the library alive and must outlive every blob it produced.
pub struct Spirv {
    // Dropped last (field order): the compiler object lives inside this
    // library's image, so releasing it after an unload would be a call into
    // unmapped memory.
    compiler: Com,
    _lib: libloading::Library,
}

/// Where `install-prerequisites.sh dxc` puts the Linux drop, overridable with
/// `FRUSTRACER_DXC_SPIRV_PATH`.
///
/// Deliberately NOT `--dxc-path`/`FRUSTRACER_DXC_PATH`, which names the
/// WINDOWS drop (`dxcompiler.dll` + `dxil.dll`). The installer fetches both
/// from one release tag precisely so the two compilers cannot drift, but they
/// are two artifacts in two directories and one path variable cannot name
/// both — pointing the D3D12 lever at a `.so` would be the kind of quiet
/// mis-wiring that surfaces as "SPIR-V is unavailable" on a tree that has it.
pub fn default_dir() -> String {
    std::env::var("FRUSTRACER_DXC_SPIRV_PATH")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/SDKs/dxc-linux/lib").to_string())
}

impl Spirv {
    /// Load `libdxcompiler.so` from `dir` and create the compiler. A missing
    /// drop is a normal condition (the SDK tree is gitignored) — the error
    /// names the fix, the same way the DXIL loader's does.
    pub fn load(dir: &str) -> Result<Self> {
        let path = std::path::Path::new(dir).join("libdxcompiler.so");
        let advice = format!(
            "The Vulkan backend needs the DirectX Shader Compiler's Linux build.\n\
             Run `./install-prerequisites.sh dxc` (it fetches the Linux tarball and\n\
             the Windows zip from the SAME release tag), or point\n\
             FRUSTRACER_DXC_SPIRV_PATH at a directory holding libdxcompiler.so\n\
             (NOT FRUSTRACER_DXC_PATH — that one names the Windows drop)."
        );
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
        Ok(Self { compiler: Com(raw), _lib: lib })
    }

    /// Compile HLSL to SPIR-V. `what` names the unit in errors; an empty
    /// `entry` omits `-E`, which a `lib_*` target requires (a DXR library
    /// exports every `[shader(...)]` entry, and naming one is a compile
    /// error).
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
        self.compile_args(src, entry, target, what, debug, &[])
    }

    /// `compile` with per-unit extra arguments appended after the shared set —
    /// the FRD kernels' `-enable-16bit-types`, matching the DXIL twin.
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
    Ok(())
}
