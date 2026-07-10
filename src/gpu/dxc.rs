//! Runtime DXC loader — the SM 6.x compiler for the GPU tracer's compute
//! kernels (RayQuery needs cs_6_5; FXC tops out at SM 5.0). Same footprint
//! policy as Streamline/OIDN/XeSS: nothing links the compiler —
//! `dxcompiler.dll` (and `dxil.dll`, the DXIL validator/signer it lazily
//! loads) are `LoadLibraryExW`'d at runtime from a gitignored SDK drop
//! (default `SDKs/dxc/bin/x64`), and the single entry point
//! `DxcCreateInstance` is resolved with GetProcAddress. The windows-rs
//! `Win32_Graphics_Direct3D_Dxc` feature supplies only interface types and
//! CLSIDs (pure data) — calling the crate's linked `DxcCreateInstance` would
//! put dxcompiler.dll in the import table and break DLL-free builds.
//!
//! There is no include handler: kernels are assembled by string concatenation
//! in Rust (trace.rs pastes the shared .hlsli sources ahead of each kernel),
//! which keeps the include_str!-and-rerun workflow of the FXC passes and
//! avoids implementing a COM interface for a virtual filesystem.

use std::ffi::c_void;
use windows::core::{Interface, GUID, HRESULT, PCSTR, PCWSTR};
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::Dxc::{
    CLSID_DxcCompiler, DxcBuffer, IDxcBlobEncoding, IDxcCompiler3, IDxcResult, DXC_CP_UTF8,
};
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LoadLibraryExW, LOAD_WITH_ALTERED_SEARCH_PATH,
};

pub type Result<T> = std::result::Result<T, String>;

fn load_dll(dir: &str, name: &str) -> Result<HMODULE> {
    let path = format!("{dir}\\{name}");
    let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
    // ALTERED_SEARCH_PATH so each DLL's own imports resolve next to it.
    unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, LOAD_WITH_ALTERED_SEARCH_PATH) }
        .map_err(|e| format!("failed to load {path}: {e}"))
}

type CreateFn = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

pub struct Dxc {
    /// Load order matters and both stay resident: dxil.dll first (dxcompiler
    /// finds the already-loaded validator in the module list — unsigned DXIL
    /// is rejected by CreateComputePipelineState outside developer mode).
    _dxil: HMODULE,
    _dxcompiler: HMODULE,
    compiler: IDxcCompiler3,
}

impl Dxc {
    /// Load dxcompiler.dll + dxil.dll from `dir` and create the compiler.
    /// A missing drop is a normal condition (the SDK is gitignored) — the
    /// error names the fix.
    pub fn load(dir: &str) -> Result<Self> {
        let advice = || {
            format!(
                "GPU mode needs the DirectX Shader Compiler DLLs.\n\
                 Drop dxcompiler.dll AND dxil.dll into {dir}\\ (from the dxc_*.zip at\n\
                 https://github.com/microsoft/DirectXShaderCompiler/releases, bin\\x64),\n\
                 or point --dxc-path / FRUSTRACER_DXC_PATH at a directory holding them."
            )
        };
        // LOAD_WITH_ALTERED_SEARCH_PATH is only defined for ABSOLUTE paths
        // (with a relative one, imports fall back to the normal search
        // order); the default dir is the relative SDKs\dxc\bin\x64.
        let abs = std::fs::canonicalize(dir)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| dir.to_string());
        let dxil = load_dll(&abs, "dxil.dll").map_err(|e| format!("{e}\n{}", advice()))?;
        let dxcompiler =
            load_dll(&abs, "dxcompiler.dll").map_err(|e| format!("{e}\n{}", advice()))?;
        let sym = unsafe { GetProcAddress(dxcompiler, PCSTR(c"DxcCreateInstance".as_ptr() as _)) }
            .ok_or("dxcompiler.dll: missing export DxcCreateInstance")?;
        let create: CreateFn = unsafe { std::mem::transmute(sym) };
        let mut raw: *mut c_void = std::ptr::null_mut();
        unsafe { create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut raw) }
            .ok()
            .map_err(|e| format!("DxcCreateInstance(DxcCompiler): {e}"))?;
        let compiler = unsafe { IDxcCompiler3::from_raw(raw) };
        Ok(Self { _dxil: dxil, _dxcompiler: dxcompiler, compiler })
    }

    /// Compile HLSL source to DXIL. `what` names the kernel in errors.
    pub fn compile(
        &self,
        src: &str,
        entry: &str,
        target: &str,
        what: &str,
        debug: bool,
    ) -> Result<Vec<u8>> {
        let mut args: Vec<String> = vec![
            "-E".into(),
            entry.into(),
            "-T".into(),
            target.into(),
            // HLSL 2021 semantics; kernels are written against it.
            "-HV".into(),
            "2021".into(),
        ];
        if debug {
            // Unoptimized + embedded debug info for PIX shader debugging.
            args.extend(["-Od".into(), "-Zi".into(), "-Qembed_debug".into()]);
        } else {
            args.push("-O3".into());
        }
        let wide: Vec<Vec<u16>> = args
            .iter()
            .map(|a| a.encode_utf16().chain(std::iter::once(0)).collect())
            .collect();
        let ptrs: Vec<PCWSTR> = wide.iter().map(|w| PCWSTR(w.as_ptr())).collect();

        let buf = DxcBuffer {
            Ptr: src.as_ptr() as *const c_void,
            Size: src.len(),
            Encoding: DXC_CP_UTF8.0,
        };
        let result: IDxcResult = unsafe { self.compiler.Compile(&buf, Some(&ptrs), None) }
            .map_err(|e| format!("DXC Compile({what}): {e}"))?;

        // IDxcResult derives from IDxcOperationResult — GetStatus/GetResult/
        // GetErrorBuffer are the stable pre-GetOutput API and are all we need.
        let status = unsafe { result.GetStatus() }
            .map_err(|e| format!("DXC GetStatus({what}): {e}"))?;
        let errors = unsafe { result.GetErrorBuffer() }
            .ok()
            .map(|b: IDxcBlobEncoding| blob_to_string(&b))
            .unwrap_or_default();
        if status.is_err() {
            return Err(format!("DXC({what}) failed:\n{errors}"));
        }
        if !errors.trim().is_empty() {
            eprintln!("gpu: DXC({what}) warnings:\n{errors}");
        }
        let blob = unsafe { result.GetResult() }
            .map_err(|e| format!("DXC GetResult({what}): {e}"))?;
        let bytes = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())
        };
        Ok(bytes.to_vec())
    }
}

fn blob_to_string<T: Interface>(blob: &T) -> String {
    // Both IDxcBlobEncoding and IDxcBlobUtf8 start with the IDxcBlob vtable;
    // go through the common interface.
    let b: windows::Win32::Graphics::Direct3D::Dxc::IDxcBlob = match blob.cast() {
        Ok(b) => b,
        Err(_) => return String::new(),
    };
    unsafe {
        let ptr = b.GetBufferPointer() as *const u8;
        let len = b.GetBufferSize();
        if ptr.is_null() || len == 0 {
            return String::new();
        }
        String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len)).into_owned()
    }
}
