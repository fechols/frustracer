//! Registered multi-engine consensus (`--quinlight`) — the fuse pass.
//!
//! A port of quinlight-player's `consensus_registered.comp`: run several
//! upscalers over the SAME traced frame, register each one to an anchor engine
//! with a per-pixel Lucas-Kanade solve, warp it into the anchor's frame, and
//! reduce the stack with a per-channel winsorized mean. The engines are the
//! chain's own levels (DLSS-RR / FSR4-RR / XeSS / FSR 3.1) — all window-res
//! RGBA16F, all fed from the same G-buffer pack, so they are N views of one
//! frame and the fuse is a purely SPATIAL, stateless pass.
//!
//! This is a STANDALONE pass with its OWN root signature: 4 root constants + a
//! descriptor table (MAX_ENGINES SRVs + 1 UAV) + one STATIC sampler (which
//! costs 0 root DWORDs and gives the warp hardware bilinear). It deliberately
//! never touches the tracer's root signature, which sits at 62 of its 64
//! DWORDs.
//!
//! The reduce math is mirrored in Rust below (`winsorized_mean`) and gated by
//! `self_test` in `--check`; the wired kernel is gated by `gate` in
//! `--check-gpu`.

use super::d3d12::{transition, Result};
use super::dxc::Dxc;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::gfx::shaders::QUIN_HLSL;

/// Must equal `MAX_ENGINES` in quin.hlsl — the SRV range width and the reduce
/// stack bound. 4 covers every engine the chain can wire (DLSS-RR, FSR4-RR,
/// XeSS, FSR 3.1).
pub const MAX_ENGINES: usize = 4;

/// Winsorization window, mirrored from quin.hlsl: tol = TAU*median + ABS.
const WINSOR_TAU: f32 = 0.1;
const WINSOR_ABS: f32 = 0.02;

/// The DENOISING engines. Both are ray-reconstruction upscalers: they take the
/// raw 1-spp trace and denoise it, where XeSS and FSR 3.1 are TAA upscalers that
/// do not. Every other engine name here is a plain upscaler.
///
/// This distinction is the single most consequential thing about a fuse in this
/// renderer — see `Engines::default_anchor`.
pub const DENOISING: [&str; 2] = ["dlss-rr", "fsr4-rr"];

/// The engine list of a quinlight session, in fuse order: index 0.. are the
/// SRVs the kernel reads, and `anchor` indexes into this list. Named so the
/// banner and `--quin-anchor` speak the same language the shader does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Engines(pub Vec<&'static str>);

impl Engines {
    pub fn names(&self) -> String {
        self.0.join(" + ")
    }

    /// The anchor a session picks when `--quin-anchor` says nothing: **the first
    /// DENOISING engine if the box wired one**, else engine 0.
    ///
    /// The anchor is the engine that is never warped — it defines the spatial
    /// frame every other engine is registered INTO, so it is the one whose
    /// geometry and sharpness survive the fuse intact. A ray-reconstruction
    /// engine (DLSS-RR, FSR4-RR) is a strictly better-conditioned reference than
    /// a TAA upscaler here for two reasons that compound:
    ///
    ///   * It is the CLEANEST image in the stack, and the LK solve registers
    ///     everything else against it. The structure tensor and the
    ///     brightness-constancy residual are both built from the anchor alone, so
    ///     anchoring on a noisy engine feeds per-frame shading noise straight
    ///     into the displacement solve — the registration would be chasing the
    ///     noise rather than the geometry.
    ///   * It is the image a viewer would rather keep. The reduce can only pull
    ///     the anchor TOWARD the others (see the measured quality table in
    ///     CLAUDE.md); making the best engine the anchor is what keeps its
    ///     spatial frame, even though the winsorized mean still moves its values.
    ///
    /// Today this reproduces the old "index 0" default exactly — the chain order
    /// already puts DLSS-RR first and FSR4-RR next — but it now says so on
    /// purpose, rather than depending on the order of a list that is free to
    /// change. `--quin-anchor N` overrides it.
    pub fn default_anchor(&self) -> u32 {
        self.0
            .iter()
            .position(|n| DENOISING.contains(n))
            .unwrap_or(0) as u32
    }

    /// Whether the anchor denoises but some other engine does not — the mixed
    /// stack whose measured behavior is the fuse's big caveat (the clean anchor
    /// gets pulled toward the noisier median). Worth saying out loud at startup.
    pub fn mixed_noise(&self, anchor: u32) -> bool {
        self.0.get(anchor as usize).is_some_and(|a| DENOISING.contains(a))
            && self.0.iter().any(|n| !DENOISING.contains(n))
    }
}

pub struct Quin {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// MAX_ENGINES SRVs then 1 UAV — the table's two ranges, in order.
    heap: ID3D12DescriptorHeap,
    /// The fused window-res image. Rests in PIXEL_SHADER_RESOURCE (the tonemap
    /// reads it at `tonemap::SRV_SLOT_QUIN`).
    pub output: ID3D12Resource,
    /// The engine outputs, kept for the record-side barriers.
    engines: Vec<ID3D12Resource>,
    pub names: Engines,
    anchor: u32,
    w: u32,
    h: u32,
}

impl Quin {
    /// `engines` are the wired upscalers' window-res RGBA16F outputs, in fuse
    /// order; `anchor` indexes them (clamped by the shader too).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ID3D12Device,
        dxc: &Dxc,
        engines: &[&ID3D12Resource],
        names: Engines,
        anchor: u32,
        w: u32,
        h: u32,
        debug: bool,
    ) -> Result<Self> {
        Self::from_source(device, dxc, QUIN_HLSL, "quin", engines, names, anchor, w, h, debug)
    }

    /// `new`, but over an arbitrary kernel source — the one seam `gate`'s
    /// QUIN_ITERS=0 control needs. Everything else (root signature, descriptor
    /// heap, output texture, `record`) is the SHIPPING pass, which is the point:
    /// a control that rebuilt those by hand would stop being a control the
    /// moment `Quin` grew a field.
    #[allow(clippy::too_many_arguments)]
    fn from_source(
        device: &ID3D12Device,
        dxc: &Dxc,
        src: &str,
        tag: &str,
        engines: &[&ID3D12Resource],
        names: Engines,
        anchor: u32,
        w: u32,
        h: u32,
        debug: bool,
    ) -> Result<Self> {
        if engines.is_empty() || engines.len() > MAX_ENGINES {
            return Err(format!("quin: {} engines (want 1..={MAX_ENGINES})", engines.len()));
        }
        let root_sig = create_root_signature(device)?;
        let dxil = dxc.compile(src, "cs_quin", "cs_6_0", tag, debug)?;
        let pso = super::trace::compute_pso(device, &root_sig, &dxil, tag)?;

        let output = super::d3d12::committed_tex(
            device,
            w,
            h,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;

        let heap = write_descriptors(device, engines, &output)?;
        Ok(Self {
            root_sig,
            pso,
            heap,
            output,
            engines: engines.iter().map(|e| (*e).clone()).collect(),
            names,
            anchor,
            w,
            h,
        })
    }

    pub fn n_engines(&self) -> u32 {
        self.engines.len() as u32
    }

    /// Fuse the engine outputs into `self.output`. Record AFTER every engine's
    /// upscale dispatch on the same list — the barriers below are the
    /// write->read sync.
    ///
    /// The engine outputs rest in PIXEL_SHADER_RESOURCE (that is the state
    /// their own upscale passes leave them in, and the state the tonemap would
    /// read them in); a COMPUTE SRV read needs NON_PIXEL_SHADER_RESOURCE, so
    /// they round-trip and come back. Batched — one ResourceBarrier per side.
    pub fn record(&self, list: &ID3D12GraphicsCommandList) {
        let _ev = super::pix::scope(list, c"quin-fuse");
        let psr = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let ua = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        unsafe {
            let mut to_read: Vec<_> =
                self.engines.iter().map(|e| transition(e, psr, npsr)).collect();
            to_read.push(transition(&self.output, psr, ua));
            list.ResourceBarrier(&to_read);

            list.SetComputeRootSignature(&self.root_sig);
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootDescriptorTable(1, self.heap.GetGPUDescriptorHandleForHeapStart());
            let cb = [self.w, self.h, self.n_engines(), self.anchor];
            list.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const _, 0);
            list.SetPipelineState(&self.pso);
            list.Dispatch(self.w.div_ceil(16), self.h.div_ceil(16), 1);

            let mut back: Vec<_> = self.engines.iter().map(|e| transition(e, npsr, psr)).collect();
            back.push(transition(&self.output, ua, psr));
            list.ResourceBarrier(&back);
        }
        // The next pass (fullscreen_to_backbuffer) re-binds its own root
        // signature, heap and PSO from scratch, so the heap swap above needs no
        // restore — the same discipline every upscaler dispatch already relies
        // on.
    }
}

/// 4 root constants (b0: w, h, n, anchor) + one table (t0..t3 SRV, u0 UAV) +
/// one static linear clamp-to-edge sampler (s0). ~6 DWORDs, and the sampler is
/// free.
fn create_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature> {
    let ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: MAX_ENGINES as u32,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: MAX_ENGINES as u32,
        },
    ];
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: ranges.len() as u32,
                    pDescriptorRanges: ranges.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    // Clamp-to-edge linear: the warp samples outside the image at the border,
    // and a wrapped tap there would register against the opposite edge.
    let sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxLOD: f32::MAX,
        ShaderRegister: 0,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        ..Default::default()
    };
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 1,
        pStaticSamplers: &sampler,
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let mut blob = None;
    let mut errb = None;
    unsafe {
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut errb))
    }
    .map_err(|e| format!("D3D12SerializeRootSignature(quin): {e}"))?;
    let blob = blob.unwrap();
    unsafe {
        device.CreateRootSignature(
            0,
            std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()),
        )
    }
    .map_err(|e| format!("CreateRootSignature(quin): {e}"))
}

/// MAX_ENGINES SRV slots then the output UAV. Slots past the wired engine count
/// get engine 0's descriptor again: the shader never reads them (`n` bounds the
/// loop), but a descriptor table must not contain a stale/null descriptor in a
/// declared range.
fn write_descriptors(
    device: &ID3D12Device,
    engines: &[&ID3D12Resource],
    output: &ID3D12Resource,
) -> Result<ID3D12DescriptorHeap> {
    let heap: ID3D12DescriptorHeap = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: MAX_ENGINES as u32 + 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
    }
    .map_err(|e| format!("quin descriptor heap: {e}"))?;
    let inc =
        unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) }
            as usize;
    let base = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    for slot in 0..MAX_ENGINES {
        let res = engines[slot.min(engines.len() - 1)];
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
            },
        };
        unsafe {
            device.CreateShaderResourceView(
                res,
                Some(&srv),
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + slot * inc },
            )
        };
    }
    let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    unsafe {
        device.CreateUnorderedAccessView(
            output,
            None,
            Some(&uav),
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + MAX_ENGINES * inc },
        )
    };
    Ok(heap)
}

// ---------------------------------------------------------------------------
// The reduce math, in Rust — the twin of quin.hlsl's winsorized_mean. Gated by
// self_test below, which is what pins the shader's identities (they are the
// same 12 lines; a drift in either is a drift in both).

/// A non-finite engine sample is MISSING, not a 0 addend. The exact predicate
/// the HLSL bit-tests (`((asuint(x) >> 23) & 0xFF) != 0xFF`).
pub fn finite1(x: f32) -> bool {
    (x.to_bits() >> 23) & 0xFF != 0xFF
}

/// Robust mean of `a`: clamp each sample to median +/- (TAU*median + ABS), then
/// average. m == 1 is a passthrough; m == 2 is EXACTLY the plain mean (both
/// samples are symmetric about their own median, so the clamp cannot move
/// either) — winsorization only starts REJECTING at m >= 3.
pub fn winsorized_mean(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    if a.len() == 1 {
        return a[0];
    }
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mid = s.len() / 2;
    let center =
        if s.len() % 2 == 1 { s[mid] } else { 0.5 * (s[mid - 1] + s[mid]) };
    let tol = WINSOR_TAU * center.max(0.0) + WINSOR_ABS;
    let (lo, hi) = (center - tol, center + tol);
    a.iter().map(|v| v.clamp(lo, hi)).sum::<f32>() / a.len() as f32
}

/// Pure, DLL-free, GPU-free — run by `--check`.
pub fn self_test() -> Result<()> {
    let close = |a: f32, b: f32, eps: f32, what: &str| -> Result<()> {
        if (a - b).abs() <= eps {
            Ok(())
        } else {
            Err(format!("quin::self_test: {what}: {a} != {b}"))
        }
    };

    // m == 0 / m == 1: the degenerate arms.
    close(winsorized_mean(&[]), 0.0, 0.0, "empty")?;
    for x in [0.0, 1e-6, 1.0, 44000.0] {
        close(winsorized_mean(&[x]), x, 0.0, "m==1 passthrough")?;
    }

    // m == 2 IS the plain mean, for ANY pair — the identity that makes a
    // two-engine session's fuse a PROVABLE registered mean rather than an
    // accident of the tolerance. Swept across the scene's whole dynamic range
    // (near-black to a sun disc).
    for &a in &[0.0f32, 0.001, 0.5, 1.0, 200.0, 44000.0] {
        for &b in &[0.0f32, 0.001, 0.5, 1.0, 200.0, 44000.0] {
            let want = 0.5 * (a + b);
            close(winsorized_mean(&[a, b]), want, want.abs() * 1e-6 + 1e-9, "m==2 plain mean")?;
        }
    }

    // m == 3: winsorization actually bites. median 1.0 => tol = 0.1*1 + 0.02,
    // so the 100.0 outlier is clamped to 1.12 instead of dragging the mean to 34.
    let w3 = winsorized_mean(&[1.0, 1.0, 100.0]);
    close(w3, (1.0 + 1.0 + 1.12) / 3.0, 1e-6, "m==3 winsorized")?;
    if w3 >= 2.0 {
        return Err(format!("quin::self_test: m==3 outlier not rejected ({w3})"));
    }

    // The median is order-independent (the reduce must not depend on which
    // engine happened to be wired first).
    let base = winsorized_mean(&[0.2, 5.0, 0.25, 0.22]);
    for perm in [[5.0, 0.2, 0.25, 0.22], [0.22, 0.25, 0.2, 5.0], [0.25, 0.22, 5.0, 0.2]] {
        close(winsorized_mean(&perm), base, 1e-6, "order independence")?;
    }

    // The default anchor: a DENOISING engine whenever one is wired, whatever
    // order the list happens to be in. These are the real engine sets each box
    // produces — including the RDNA4 one, which the dev NVIDIA box cannot run,
    // so this is the only thing standing between that path and a silent
    // regression.
    let anchor_of = |v: Vec<&'static str>| -> &'static str {
        let e = Engines(v);
        e.0[e.default_anchor() as usize]
    };
    for (set, want) in [
        // NVIDIA (measured: this exact set + anchor).
        (vec!["dlss-rr", "fsr3", "xess"], "dlss-rr"),
        // RDNA4: no DLSS, so FSR4-RR is the denoiser and must anchor.
        (vec!["fsr4-rr", "fsr3", "xess"], "fsr4-rr"),
        // Both RR flavors somehow live: DLSS-RR is first in chain order.
        (vec!["dlss-rr", "fsr4-rr", "fsr3", "xess"], "dlss-rr"),
        // A denoiser that is NOT first in the list must still win the anchor —
        // the property is "denoising", not "index 0".
        (vec!["xess", "fsr4-rr"], "fsr4-rr"),
        (vec!["fsr3", "xess", "dlss-rr"], "dlss-rr"),
        // No denoiser wired: fall back to engine 0.
        (vec!["fsr3", "xess"], "fsr3"),
        (vec!["xess", "fsr3"], "xess"),
    ] {
        let got = anchor_of(set.clone());
        if got != want {
            return Err(format!(
                "quin::self_test: default anchor for {set:?} = {got}, want {want}"
            ));
        }
    }
    // The mixed-noise warning fires exactly when the anchor denoises and some
    // engine does not.
    if !Engines(vec!["dlss-rr", "xess"]).mixed_noise(0) {
        return Err("quin::self_test: mixed_noise missed a denoiser+TAA stack".into());
    }
    if Engines(vec!["fsr3", "xess"]).mixed_noise(0) {
        return Err("quin::self_test: mixed_noise fired on an all-TAA stack".into());
    }
    if Engines(vec!["dlss-rr", "fsr4-rr"]).mixed_noise(0) {
        return Err("quin::self_test: mixed_noise fired on an all-denoiser stack".into());
    }

    // Non-finite is MISSING: the exact 8-bit-exponent predicate the shader uses.
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        if finite1(x) {
            return Err(format!("quin::self_test: finite1({x}) accepted a non-finite"));
        }
    }
    for x in [0.0, -0.0, 1e-45, f32::MAX, -44000.0] {
        if !finite1(x) {
            return Err(format!("quin::self_test: finite1({x}) rejected a finite"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The GPU gate (--check-gpu): the REAL kernel through the REAL root signature
// and descriptor table, over synthetic engine images.

/// A high-frequency test image: a checker times a smooth gradient, so the
/// structure tensor is well conditioned everywhere (a flat image would solve to
/// zero displacement no matter what the residual said, and would pass the shift
/// gate vacuously).
fn test_image(w: u32, h: u32, dx: i32) -> Vec<[f32; 3]> {
    let mut v = vec![[0.0f32; 3]; (w * h) as usize];
    for y in 0..w.max(h) as i32 {
        for x in 0..w as i32 {
            if y >= h as i32 {
                break;
            }
            // Sample the SOURCE at (x - dx): the result is the source shifted
            // by +dx px, which is exactly what the LK must solve for.
            let sx = x - dx;
            let checker = if ((sx.rem_euclid(8)) < 4) ^ ((y.rem_euclid(8)) < 4) { 1.0 } else { 0.3 };
            let ramp = 0.2 + 0.8 * (sx.rem_euclid(64) as f32 / 64.0);
            let c = checker * ramp;
            v[(y as u32 * w + x as u32) as usize] = [c, c * 0.75, c * 0.5];
        }
    }
    v
}

/// Upload an RGB f32 image as an RGBA16F Texture2D resting in
/// PIXEL_SHADER_RESOURCE (the state `Quin::record` expects its engines in).
fn upload_engine(
    hg: &mut super::trace::HeadlessGpu,
    img: &[[f32; 3]],
    w: u32,
    h: u32,
) -> Result<ID3D12Resource> {
    use super::d3d12::{
        aligned_pitch, committed_tex, footprint, loc_footprint, loc_subresource, UploadBuffer,
    };
    use half::f16;
    let tex = committed_tex(
        &hg.device,
        w,
        h,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let pitch = aligned_pitch(w as usize * 8);
    let up = UploadBuffer::new(&hg.device, pitch * h as usize)?;
    for y in 0..h as usize {
        let row: &mut [[f16; 4]] =
            unsafe { std::slice::from_raw_parts_mut(up.ptr.add(y * pitch) as *mut [f16; 4], w as usize) };
        for (x, px) in row.iter_mut().enumerate() {
            let c = img[y * w as usize + x];
            *px = [
                f16::from_f32(c[0]),
                f16::from_f32(c[1]),
                f16::from_f32(c[2]),
                f16::from_f32(1.0),
            ];
        }
    }
    let fp = footprint(DXGI_FORMAT_R16G16B16A16_FLOAT, w, h, 8, 0);
    hg.run(|list| unsafe {
        list.CopyTextureRegion(&loc_subresource(&tex), 0, 0, 0, &loc_footprint(&up.resource, fp), None);
        list.ResourceBarrier(&[transition(
            &tex,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )]);
    })?;
    Ok(tex)
}

/// Read the fuse output back as RGB f32.
fn read_output(
    hg: &mut super::trace::HeadlessGpu,
    out: &ID3D12Resource,
    w: u32,
    h: u32,
) -> Result<Vec<[f32; 3]>> {
    use super::d3d12::{aligned_pitch, footprint, loc_footprint, loc_subresource, ReadbackBuffer};
    use half::f16;
    let pitch = aligned_pitch(w as usize * 8);
    let rb = ReadbackBuffer::new(&hg.device, pitch * h as usize)?;
    let fp = footprint(DXGI_FORMAT_R16G16B16A16_FLOAT, w, h, 8, 0);
    hg.run(|list| unsafe {
        list.ResourceBarrier(&[transition(
            out,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyTextureRegion(&loc_footprint(&rb.resource, fp), 0, 0, 0, &loc_subresource(out), None);
        list.ResourceBarrier(&[transition(
            out,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )]);
    })?;
    let mut ptr = std::ptr::null_mut();
    unsafe { rb.resource.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("quin readback Map: {e}"))?;
    let mut v = vec![[0.0f32; 3]; (w * h) as usize];
    for y in 0..h as usize {
        let row = unsafe { (ptr as *const u8).add(y * pitch) };
        for x in 0..w as usize {
            let px: &[f16; 4] = unsafe { &*(row.add(x * 8) as *const [f16; 4]) };
            v[y * w as usize + x] = [px[0].into(), px[1].into(), px[2].into()];
        }
    }
    unsafe { rb.resource.Unmap(0, None) };
    Ok(v)
}

/// The engine images live in RGBA16F, so what the kernel reads — and therefore
/// the most a bit-exact gate can demand back — is the f16-rounded source, not
/// the f32 one it was built from. Charging the port for the WIRE's rounding
/// would gate the upload, not the fuse (the `tonemap::selftest` precedent).
fn round_f16(img: &[[f32; 3]]) -> Vec<[f32; 3]> {
    use half::f16;
    img.iter()
        .map(|c| {
            [
                f16::from_f32(c[0]).to_f32(),
                f16::from_f32(c[1]).to_f32(),
                f16::from_f32(c[2]).to_f32(),
            ]
        })
        .collect()
}

/// Mean |Δ| against a reference, over the interior only — the border is where
/// the warp clamps and the solve window runs off the image, and upstream does
/// not claim registration there either.
fn interior_err(a: &[[f32; 3]], b: &[[f32; 3]], w: u32, h: u32, margin: u32) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for y in margin..h - margin {
        for x in margin..w - margin {
            let i = (y * w + x) as usize;
            for c in 0..3 {
                sum += (a[i][c] - b[i][c]).abs() as f64;
                n += 1;
            }
        }
    }
    (sum / n.max(1) as f64) as f32
}

/// `--check-gpu`: the wired fuse over synthetic engines. Three gates, each
/// aimed at a distinct way the port can be wrong.
pub fn gate(hg: &mut super::trace::HeadlessGpu, dxc: &Dxc, debug: bool) -> Result<()> {
    let (w, h) = (128u32, 96u32);
    let anchor_img = test_image(w, h, 0);
    // The oracle every gate below compares against: the source as the kernel
    // actually sees it, through the RGBA16F wire.
    let anchor_ref = round_f16(&anchor_img);

    let run = |hg: &mut super::trace::HeadlessGpu,
               engines: &[&ID3D12Resource],
               n_names: Engines,
               anchor: u32|
     -> Result<Vec<[f32; 3]>> {
        let q = Quin::new(&hg.device, dxc, engines, n_names, anchor, w, h, debug)?;
        hg.run(|list| q.record(list))?;
        read_output(hg, &q.output, w, h)
    };

    let e0 = upload_engine(hg, &anchor_img, w, h)?;

    // (1) N == 1: a lone engine passes through bit-exactly. Nothing to
    // register, nothing to average — this pins the degenerate arm and the
    // descriptor/UAV wiring in isolation.
    let out = run(hg, &[&e0], Engines(vec!["e0"]), 0)?;
    let err = interior_err(&out, &anchor_ref, w, h, 0);
    if err != 0.0 {
        return Err(format!("quin gate: N==1 passthrough not bit-exact (mean |d| {err:.6})"));
    }

    // (2) IDENTITY: two bit-identical engines. The LK must solve (0, 0) and the
    // m==2 reduce is the mean of two equal samples, so the output must come back
    // bit-equal to the input. This is the gate that catches the sampler's +0.5
    // texel-centre convention, the groupshared HALO indexing, a sign flip in the
    // residual, and an inverted tensor solve — every one of those perturbs a
    // pixel here.
    let out = run(hg, &[&e0, &e0], Engines(vec!["e0", "e0"]), 0)?;
    let err = interior_err(&out, &anchor_ref, w, h, 0);
    if err != 0.0 {
        return Err(format!("quin gate: identity fuse not bit-exact (mean |d| {err:.6})"));
    }

    // (3) SHIFT: engine 1 is the anchor shifted by exactly (+1, 0) px. A correct
    // registration solves uu = +1, warps engine 1 back onto the anchor's frame,
    // and the fuse reproduces the anchor. The CONTROL is the same fuse with the
    // registration disabled (QUIN_ITERS 0 — the shader then folds engine 1 in
    // unwarped, i.e. the plain unregistered mean): the registered error must be
    // a large factor better, which is the only thing that proves the LK is doing
    // real work rather than solving zero and getting away with it.
    let shifted = test_image(w, h, 1);
    let e1 = upload_engine(hg, &shifted, w, h)?;
    let out = run(hg, &[&e0, &e1], Engines(vec!["e0", "e1"]), 0)?;
    let margin = 8; // HALO + MAX_DISP + slack: the border the warp clamps in
    let reg_err = interior_err(&out, &anchor_ref, w, h, margin);

    let control = {
        let src = QUIN_HLSL.replace("#define QUIN_ITERS 2", "#define QUIN_ITERS 0");
        if src == QUIN_HLSL {
            return Err("quin gate: QUIN_ITERS control substitution failed".into());
        }
        let q = Quin::from_source(
            &hg.device,
            dxc,
            &src,
            "quin-noreg",
            &[&e0, &e1],
            Engines(vec!["e0", "e1"]),
            0,
            w,
            h,
            debug,
        )?;
        hg.run(|list| q.record(list))?;
        read_output(hg, &q.output, w, h)?
    };
    let raw_err = interior_err(&control, &anchor_ref, w, h, margin);

    if !(reg_err < raw_err * 0.25) {
        return Err(format!(
            "quin gate: registration did not recover the (+1,0) shift \
             (registered mean |d| {reg_err:.5} vs unregistered {raw_err:.5}; want < 25% of it)"
        ));
    }
    eprintln!(
        "  quin  : N==1 bit-exact, identity bit-exact, (+1,0) shift recovered \
         (mean |d| {reg_err:.5} vs {raw_err:.5} unregistered, {:.1}x better)",
        raw_err / reg_err.max(1e-9)
    );
    Ok(())
}
