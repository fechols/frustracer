//! Intel Open Image Denoise (OIDN 2.x) — the secondary denoiser next to DLSS
//! Ray Reconstruction. Same footprint policy as Streamline: nothing links the
//! SDK; `OpenImageDenoise.dll` is loaded at runtime from the SDK bin directory
//! (default `SDKs/oidn.x64.windows/bin`), so builds and DLL-free machines are
//! unaffected and every entry point is resolved into a fn-pointer table.
//! Unlike SL there is no C++ shim — OIDN is a plain, unversioned C API, safe
//! to mirror directly in Rust.
//!
//! Data path: the RT filter consumes the *resolved accumulation average*
//! (linear HDR color) plus first-hit albedo and world-space normal, both
//! already captured in `dlss::GBufs` by the primary-path fill sites in
//! render.rs. All images go through OIDN buffers (`oidnNewBuffer` +
//! write/read) rather than shared host pointers — host memory is not
//! guaranteed device-accessible on the GPU backends, and the staging pass
//! exists anyway (color needs the sample divide, albedo the diffuse+specular
//! combine, normal the float4→float3 repack). Images are bound and the filter
//! committed once at construction; per frame only buffer contents change,
//! which needs no recommit.

use crate::dlss::GBufs;
use half::f16;
use rayon::prelude::*;
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::time::Instant;
use windows::core::{PCSTR, PCWSTR};
use windows::Win32::Foundation::HMODULE;
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LoadLibraryExW, LOAD_WITH_ALTERED_SEARCH_PATH,
};

type OIDNDevice = *mut c_void;
type OIDNBuffer = *mut c_void;
type OIDNFilter = *mut c_void;

// oidn.h enum values (C ints). Device type 0 = auto; the `--oidn-device`
// parser in main.rs maps names to these same values.
const DEVICE_CPU: i32 = 1;
const DEVICE_SYCL: i32 = 2;
const DEVICE_CUDA: i32 = 3;
const DEVICE_HIP: i32 = 4;
// All four filter images are half-precision (SDKs oidn.h: OIDN_FORMAT_HALF =
// 257, HALF3 = 259): the color input is 1-spp noise or an EMA average, the
// guides come from f16 G-buffer storage, and the network runs in reduced
// precision internally anyway — halves the staging and device-transfer bytes.
const FORMAT_HALF3: i32 = 259;

/// Narrow HDR radiance to f16 for the color image. The clamp matters: a
/// linear-RGB value above f16::MAX (65504) would narrow to +Inf, which can
/// propagate through the filter to a non-finite output (the --check-oidn
/// finite gate) and the tonemap. Guides ([0,1] albedo, unit normals) never
/// need it. Deliberately a `>` compare, not `f32::min` — min(NaN, c)
/// returns c, which would launder an upstream NaN into 65504 and hide it
/// from the finite gate; NaN fails the compare and passes through.
#[inline(always)]
fn narrow_hdr(v: f32) -> f16 {
    let max = f16::MAX.to_f32_const();
    f16::from_f32(if v > max { max } else { v })
}
pub const QUALITY_FAST: i32 = 4;
pub const QUALITY_BALANCED: i32 = 5;
pub const QUALITY_HIGH: i32 = 6;

/// The resolved OIDN C entry points (all cdecl, undecorated x64 exports).
struct Api {
    new_device: unsafe extern "C" fn(i32) -> OIDNDevice,
    commit_device: unsafe extern "C" fn(OIDNDevice),
    get_device_error: unsafe extern "C" fn(OIDNDevice, *mut *const c_char) -> i32,
    get_device_int: unsafe extern "C" fn(OIDNDevice, *const c_char) -> i32,
    release_device: unsafe extern "C" fn(OIDNDevice),
    new_buffer: unsafe extern "C" fn(OIDNDevice, usize) -> OIDNBuffer,
    write_buffer: unsafe extern "C" fn(OIDNBuffer, usize, usize, *const c_void),
    read_buffer: unsafe extern "C" fn(OIDNBuffer, usize, usize, *mut c_void),
    release_buffer: unsafe extern "C" fn(OIDNBuffer),
    new_filter: unsafe extern "C" fn(OIDNDevice, *const c_char) -> OIDNFilter,
    set_filter_image: unsafe extern "C" fn(
        OIDNFilter,
        *const c_char,
        OIDNBuffer,
        i32,   // OIDNFormat
        usize, // width
        usize, // height
        usize, // byteOffset
        usize, // pixelByteStride (0 = auto)
        usize, // rowByteStride (0 = auto)
    ),
    set_filter_bool: unsafe extern "C" fn(OIDNFilter, *const c_char, bool),
    set_filter_int: unsafe extern "C" fn(OIDNFilter, *const c_char, i32),
    commit_filter: unsafe extern "C" fn(OIDNFilter),
    execute_filter: unsafe extern "C" fn(OIDNFilter),
    release_filter: unsafe extern "C" fn(OIDNFilter),
}

fn load_dll(dir: &str, name: &str) -> Result<HMODULE, String> {
    let path = format!("{dir}\\{name}");
    let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
    // ALTERED_SEARCH_PATH so each DLL's own imports resolve next to it.
    unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, LOAD_WITH_ALTERED_SEARCH_PATH) }
        .map_err(|e| format!("failed to load {path}: {e}"))
}

/// `GetProcAddress` + transmute to the field's fn type (inferred at the use
/// site from the `Api` struct literal).
macro_rules! resolve {
    ($h:expr, $name:literal) => {{
        let sym = unsafe { GetProcAddress($h, PCSTR(concat!($name, "\0").as_ptr())) }
            .ok_or_else(|| format!("OpenImageDenoise.dll: missing export {}", $name))?;
        // Untyped on purpose: the fn type is inferred from the `Api` field
        // being initialized — annotating here would hand-mirror every
        // signature a second time.
        #[allow(clippy::missing_transmute_annotations)]
        let f = unsafe { std::mem::transmute(sym) };
        f
    }};
}

pub struct OidnContext {
    api: Api,
    dev: OIDNDevice,
    filter: OIDNFilter,
    buf_color: OIDNBuffer,
    buf_albedo: OIDNBuffer,
    buf_normal: OIDNBuffer,
    buf_out: OIDNBuffer,
    /// Buffer/staging capacity — the construction resolution. `set_res` may
    /// rebind the filter at any resolution up to this (XeSS mode denoises at
    /// the dynamic render res; the buffers never reallocate).
    max_w: usize,
    max_h: usize,
    w: usize,
    h: usize,
    staging_color: Vec<f16>,
    staging_albedo: Vec<f16>,
    staging_normal: Vec<f16>,
    staging_out: Vec<f16>,
    /// The denoised output widened back to f32 — the seam that keeps
    /// `render::resolve_hdr` (shared with the f32 producers) unchanged.
    out_f32: Vec<f32>,
    /// "cpu" / "cuda" / ... — the device OIDN actually picked.
    pub device_desc: &'static str,
    /// Wall time of the last `denoise` (staging + transfer + filter).
    pub last_ms: f64,
}

fn device_error(api: &Api, dev: OIDNDevice, what: &str) -> Result<(), String> {
    let mut msg: *const c_char = std::ptr::null();
    let err = unsafe { (api.get_device_error)(dev, &mut msg) };
    if err == 0 {
        return Ok(());
    }
    let m = if msg.is_null() {
        String::new()
    } else {
        // The message pointer is only valid until the next OIDN call — copy now.
        unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned()
    };
    Err(format!("oidn {what}: error {err}: {m}"))
}

impl OidnContext {
    /// Load the DLLs from `dll_dir`, create + commit a device of
    /// `device_type` (`DEVICE_DEFAULT` auto-picks the fastest), and build the
    /// RT filter with color/albedo/normal inputs at `w`×`h` — which is also
    /// the buffer/staging capacity; `set_res` can later rebind the filter at
    /// any resolution that fits (XeSS mode's dynamic render res). The filter
    /// commit loads the network weights — the expensive one-time step.
    /// `quality` is one of the `QUALITY_*` consts; `clean_aux` declares the
    /// albedo/normal guides noise-free (true here by construction — they are
    /// deterministic primary-hit values, not Monte Carlo estimates).
    pub fn new(
        dll_dir: &str,
        w: usize,
        h: usize,
        device_type: i32,
        quality: i32,
        clean_aux: bool,
    ) -> Result<Self, String> {
        // LOAD_WITH_ALTERED_SEARCH_PATH only alters the search for absolute
        // paths — make a relative --oidn-path absolute (best-effort: a
        // missing directory keeps the original string so the load error
        // stays readable).
        let dir = std::fs::canonicalize(dll_dir)
            .map(|p| p.to_string_lossy().trim_start_matches(r"\\?\").to_string())
            .unwrap_or_else(|_| dll_dir.to_string());
        let dll_dir = dir.as_str();
        // Preload the dependency chain by absolute path so OIDN's own lazy
        // loads (core pulls in a device runtime at commit) resolve from the
        // already-loaded module list, wherever the process cwd is. tbb/core
        // failures are deferred — the main DLL load gives the clearer error —
        // and the device runtimes are best-effort (cuda legitimately fails
        // without an NVIDIA driver; cpu is the fallback).
        for dep in ["tbb12.dll", "OpenImageDenoise_core.dll"] {
            let _ = load_dll(dll_dir, dep);
        }
        for dev_dll in ["OpenImageDenoise_device_cpu.dll", "OpenImageDenoise_device_cuda.dll"] {
            let _ = load_dll(dll_dir, dev_dll);
        }
        let h_dll = load_dll(dll_dir, "OpenImageDenoise.dll")?;
        let api = Api {
            new_device: resolve!(h_dll, "oidnNewDevice"),
            commit_device: resolve!(h_dll, "oidnCommitDevice"),
            get_device_error: resolve!(h_dll, "oidnGetDeviceError"),
            get_device_int: resolve!(h_dll, "oidnGetDeviceInt"),
            release_device: resolve!(h_dll, "oidnReleaseDevice"),
            new_buffer: resolve!(h_dll, "oidnNewBuffer"),
            write_buffer: resolve!(h_dll, "oidnWriteBuffer"),
            read_buffer: resolve!(h_dll, "oidnReadBuffer"),
            release_buffer: resolve!(h_dll, "oidnReleaseBuffer"),
            new_filter: resolve!(h_dll, "oidnNewFilter"),
            set_filter_image: resolve!(h_dll, "oidnSetFilterImage"),
            set_filter_bool: resolve!(h_dll, "oidnSetFilterBool"),
            set_filter_int: resolve!(h_dll, "oidnSetFilterInt"),
            commit_filter: resolve!(h_dll, "oidnCommitFilter"),
            execute_filter: resolve!(h_dll, "oidnExecuteFilter"),
            release_filter: resolve!(h_dll, "oidnReleaseFilter"),
        };
        // The HMODULE is deliberately never freed (same policy as the SL shim).

        let dev = unsafe { (api.new_device)(device_type) };
        if dev.is_null() {
            // A null device is queried through the null handle by design.
            device_error(&api, std::ptr::null_mut(), "device creation")?;
            return Err("oidn: device creation failed with no error message".into());
        }
        unsafe { (api.commit_device)(dev) };
        if let Err(e) = device_error(&api, dev, "device commit") {
            unsafe { (api.release_device)(dev) };
            return Err(e);
        }
        let device_desc = match unsafe { (api.get_device_int)(dev, c"type".as_ptr()) } {
            DEVICE_CPU => "cpu",
            DEVICE_SYCL => "sycl",
            DEVICE_CUDA => "cuda",
            DEVICE_HIP => "hip",
            5 => "metal",
            _ => "unknown",
        };

        let bytes = w * h * 3 * size_of::<f16>();
        let cleanup = |api: &Api, bufs: &[OIDNBuffer], filter: OIDNFilter| {
            for &b in bufs {
                if !b.is_null() {
                    unsafe { (api.release_buffer)(b) };
                }
            }
            if !filter.is_null() {
                unsafe { (api.release_filter)(filter) };
            }
            unsafe { (api.release_device)(dev) };
        };
        let filter = unsafe { (api.new_filter)(dev, c"RT".as_ptr()) };
        if filter.is_null() {
            let e = device_error(&api, dev, "filter creation");
            cleanup(&api, &[], filter);
            return Err(e.err().unwrap_or_else(|| "oidn: RT filter creation failed".into()));
        }
        let mut bufs = [std::ptr::null_mut(); 4];
        for (i, name) in [c"color", c"albedo", c"normal", c"output"].iter().enumerate() {
            let b = unsafe { (api.new_buffer)(dev, bytes) };
            if b.is_null() {
                let e = device_error(&api, dev, "buffer allocation");
                cleanup(&api, &bufs, filter);
                return Err(e.err().unwrap_or_else(|| "oidn: buffer allocation failed".into()));
            }
            bufs[i] = b;
            unsafe { (api.set_filter_image)(filter, name.as_ptr(), b, FORMAT_HALF3, w, h, 0, 0, 0) };
        }
        unsafe {
            (api.set_filter_bool)(filter, c"hdr".as_ptr(), true);
            (api.set_filter_bool)(filter, c"cleanAux".as_ptr(), clean_aux);
            (api.set_filter_int)(filter, c"quality".as_ptr(), quality);
            (api.commit_filter)(filter);
        }
        if let Err(e) = device_error(&api, dev, "filter commit") {
            cleanup(&api, &bufs, filter);
            return Err(e);
        }

        Ok(Self {
            api,
            dev,
            filter,
            buf_color: bufs[0],
            buf_albedo: bufs[1],
            buf_normal: bufs[2],
            buf_out: bufs[3],
            max_w: w,
            max_h: h,
            w,
            h,
            staging_color: vec![f16::ZERO; w * h * 3],
            staging_albedo: vec![f16::ZERO; w * h * 3],
            staging_normal: vec![f16::ZERO; w * h * 3],
            staging_out: vec![f16::ZERO; w * h * 3],
            out_f32: vec![0.0; w * h * 3],
            device_desc,
            last_ms: 0.0,
        })
    }

    /// Rebind the filter images at a new resolution (≤ the construction
    /// resolution) and recommit. The buffers and network weights stay put —
    /// this is the cheap path that makes OIDN tolerate XeSS mode's dynamic
    /// render resolution; res steps are already rare by quantization, and
    /// each costs one filter commit, not a device rebuild.
    pub fn set_res(&mut self, w: usize, h: usize) -> Result<(), String> {
        if (w, h) == (self.w, self.h) {
            return Ok(());
        }
        if w * h > self.max_w * self.max_h || w == 0 || h == 0 {
            return Err(format!(
                "oidn: set_res {w}x{h} exceeds buffer capacity {}x{}",
                self.max_w, self.max_h
            ));
        }
        for (name, buf) in [
            (c"color", self.buf_color),
            (c"albedo", self.buf_albedo),
            (c"normal", self.buf_normal),
            (c"output", self.buf_out),
        ] {
            unsafe {
                (self.api.set_filter_image)(
                    self.filter,
                    name.as_ptr(),
                    buf,
                    FORMAT_HALF3,
                    w,
                    h,
                    0,
                    0,
                    0,
                )
            };
        }
        unsafe { (self.api.commit_filter)(self.filter) };
        // Cache the new res only after a clean commit. On failure the images
        // are already rebound at the new res, so poison the cache — (0, 0)
        // never matches a request (zero dims are rejected above) — or the
        // equal-res early return would forever skip the rebind back and every
        // later denoise would run mis-strided.
        if let Err(e) = device_error(&self.api, self.dev, "set_res commit") {
            self.w = 0;
            self.h = 0;
            return Err(e);
        }
        self.w = w;
        self.h = h;
        Ok(())
    }

    /// Denoise the accumulation average. `accum` is the render loop's raw
    /// linear-RGB sum buffer at exactly `w`×`h` (in XeSS mode: the `rw*rh*3`
    /// prefix slice); `g` holds the matching-frame G-buffers. Returns the
    /// denoised linear HDR image (3 floats/px).
    pub fn denoise(
        &mut self,
        accum: &[AtomicU32],
        samples: u32,
        g: &GBufs,
    ) -> Result<&[f32], String> {
        let n = self.w * self.h * 3;
        assert_eq!(accum.len(), n);
        let t0 = Instant::now();
        let inv = 1.0 / samples.max(1) as f32;
        let w = self.w;
        self.staging_color[..n].par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            for (k, v) in row.iter_mut().enumerate() {
                *v = narrow_hdr(f32::from_bits(accum[y * w * 3 + k].load(Relaxed)) * inv);
            }
        });
        self.run_filter(None, g, t0)
    }

    /// Denoise an already-averaged linear-HDR image (3 floats/px) — the
    /// temporal-reprojection history path, where the EMA fold has replaced
    /// the accumulation divide.
    pub fn denoise_hdr(&mut self, color: &[f32], g: &GBufs) -> Result<&[f32], String> {
        let t0 = Instant::now();
        self.run_filter(Some(color), g, t0)
    }

    /// Shared tail of the `denoise*` entry points: stage albedo/normal from
    /// `g`, write the three input buffers, execute the filter, read back.
    /// `color` is an f32 HDR input to narrow into `staging_color`; `None`
    /// means the caller just filled it (the accumulation-divide path).
    fn run_filter(&mut self, color: Option<&[f32]>, g: &GBufs, t0: Instant) -> Result<&[f32], String> {
        crate::zone!("oidn-filter");
        assert_eq!((g.rw, g.rh), (self.w, self.h), "gbuf/filter resolution mismatch");
        let w = self.w;
        let n = self.w * self.h * 3;
        if let Some(c) = color {
            assert_eq!(c.len(), n);
            self.staging_color[..n]
                .par_iter_mut()
                .zip(c.par_iter())
                .for_each(|(d, &v)| *d = narrow_hdr(v));
        }
        // First-hit albedo per the OIDN guidance: diffuse color plus the
        // (RGB F0) specular reflectivity, clamped to [0,1]. Sky pixels
        // already carry diff_alb = 1. Recomputed in f32 (the add + clamp
        // can't be a bit copy), then narrowed — values are in [0,1].
        let load = crate::dlss::ld16; // the guide planes store f16 bits
        self.staging_albedo[..n].par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            for (x, px) in row.chunks_exact_mut(3).enumerate() {
                let i = y * w + x;
                for k in 0..3 {
                    px[k] = f16::from_f32(
                        (load(&g.diff_alb[i * 3 + k]) + load(&g.spec_alb[i * 3 + k]))
                            .clamp(0.0, 1.0),
                    );
                }
            }
        });
        // World-space normal (OIDN accepts any frame as long as it is
        // consistent); repack float4 (xyz + roughness) → float3 — a raw
        // bit copy now that the plane stores f16.
        self.staging_normal[..n].par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            for (x, px) in row.chunks_exact_mut(3).enumerate() {
                let i = y * w + x;
                for k in 0..3 {
                    px[k] = f16::from_bits(g.normal_rough[i * 4 + k].load(Relaxed));
                }
            }
        });

        let bytes = n * size_of::<f16>();
        unsafe {
            (self.api.write_buffer)(
                self.buf_color,
                0,
                bytes,
                self.staging_color.as_ptr().cast(),
            );
            (self.api.write_buffer)(self.buf_albedo, 0, bytes, self.staging_albedo.as_ptr().cast());
            (self.api.write_buffer)(self.buf_normal, 0, bytes, self.staging_normal.as_ptr().cast());
            // Blocks until the filter (and the writes queued before it) finish.
            (self.api.execute_filter)(self.filter);
        }
        device_error(&self.api, self.dev, "execute")?;
        unsafe {
            (self.api.read_buffer)(self.buf_out, 0, bytes, self.staging_out.as_mut_ptr().cast());
        }
        device_error(&self.api, self.dev, "readback")?;
        // Widen for the f32 consumers (resolve_hdr, the history-free XeSS
        // post path) — the seam that keeps their signatures unchanged.
        self.out_f32[..n]
            .par_iter_mut()
            .zip(self.staging_out[..n].par_iter())
            .for_each(|(d, &v)| *d = v.to_f32());
        self.last_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(&self.out_f32[..n])
    }

    /// The most recent denoised output (valid after any successful
    /// `denoise` at the current resolution) — lets the caller re-composite
    /// the overlay without re-running the filter.
    pub fn last_output(&self) -> &[f32] {
        &self.out_f32[..self.w * self.h * 3]
    }
}

impl Drop for OidnContext {
    fn drop(&mut self) {
        unsafe {
            (self.api.release_filter)(self.filter);
            for b in [self.buf_color, self.buf_albedo, self.buf_normal, self.buf_out] {
                (self.api.release_buffer)(b);
            }
            (self.api.release_device)(self.dev);
        }
    }
}
