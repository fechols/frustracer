//! FrdGpu — the D3D12 host for FRD, the from-scratch clean-room denoiser
//! (`--frd`; src/frd.rs carries the math + the provenance rule). The NrdGpu
//! slot-mate in GpuContext's ONE denoiser slot (`DnGpu`), drastically
//! simpler: no DLL, no repr(C) transcription, no foreign dispatch list — we
//! own the pass sequence statically, compiled through the session DXC on our
//! own root signature (the bloom private-heap pattern at the DXC tier).
//!
//! PLANE CONTRACT: the seven app planes carry the SAME names, formats, and
//! rest state as NrdGpu's (`plane_in_mv/nr/viewz/diff/spec`,
//! `plane_out_diff/spec`, all resting in UNORDERED_ACCESS — the NppdRes
//! doctrine), which is what lets `arm_denoiser_for`'s wire-register block
//! serve either engine verbatim: the bridge kernels (nrd_bridge.hlsl's
//! cs_nrd_pack / cs_nrd_out) neither know nor care which engine sits between
//! them. FRD never writes an IN plane.
//!
//! PHASE C (current): the plan's 3-dispatch recurrent shape —
//!   1. cs_frd_temporal (frd_temporal.hlsl): reproject off the wire's 2.5D
//!      MV, per-foot disocclusion, slow/fast accumulation → transient accum.
//!   2. cs_frd_blur (frd_blur.hlsl): Vogel-disk bilateral blur at a
//!      hit-dist-driven, accumulation-scaled radius; history fix folded in
//!      as the n < N_FIX radius boost.
//!   3. cs_frd_post: second disk at 1.7×, fast-history clamp, then writes
//!      OUT **and the recurrent slow feedback** — pass 3's output is the
//!      history pass 1 reprojects next frame (slow is single-buffered:
//!      pass 1 reads it, pass 3 rewrites it, the inter-pass barrier orders).
//! `force_passthrough` byte-copies IN→OUT instead — the F3 gate's control
//! arm (the plumbing proof, kept alive for the engine's whole life).
//! Phase D: the fp16 arm (`fp16` is the OPTIONS4 probe), LDS tile,
//! stabilization. The firefly pre-clamp + anti-lag brake are BUILT
//! (2026-08-09, the bright-specular-smear fix): pass 1 clamps the input
//! against its 3x3 mean (flags bit 1, --frd-[no-]anti-firefly) and cuts
//! reprojected age by pass 3's recorded clamp excess (flags bit 3,
//! FR_FRD_ANTILAG=off — the `antilag` R8G8_UNORM plane is the feedback
//! path, single-buffered like `slow`); the specular parallax input sums
//! the camera term with `light_par` (the moving-sun case — see
//! frd::oracle::light_parallax and nrd_frame_step's Frd arm).

#![allow(dead_code)]

use super::{d3d12, dxc};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

type Result<T> = std::result::Result<T, String>;

const FRD_COMMON_HLSLI: &str = include_str!("shaders/frd_common.hlsli");
const FRD_TEMPORAL_HLSL: &str = include_str!("shaders/frd_temporal.hlsl");
const FRD_BLUR_HLSL: &str = include_str!("shaders/frd_blur.hlsl");

/// One texture + its tracked state (rests in UNORDERED_ACCESS).
struct Reg {
    res: ID3D12Resource,
    state: std::cell::Cell<D3D12_RESOURCE_STATES>,
}

// Wire-plane indices (the NrdGpu ordering, minus the validation plane).
const P_IN_MV: usize = 0;
const P_IN_NR: usize = 1;
const P_IN_VIEWZ: usize = 2;
const P_IN_DIFF: usize = 3;
const P_IN_SPEC: usize = 4;
const P_OUT_DIFF: usize = 5;
const P_OUT_SPEC: usize = 6;

// Per-parity history roles.
const H_FAST_D: usize = 0;
const H_FAST_S: usize = 1;
const H_META: usize = 2;
const H_NR: usize = 3;
const H_VZ: usize = 4;
const H_COUNT: usize = 5;

// Transients.
const T_ACCUM_D: usize = 0;
const T_ACCUM_S: usize = 1;
const T_BLUR_D: usize = 2;
const T_BLUR_S: usize = 3;

const SRVS: usize = 13;
const UAVS: usize = 9;
const SET_STRIDE: usize = SRVS + UAVS;
// Heap sets: [pass1 P0, pass1 P1, pass2 P0, pass2 P1, pass3 P0, pass3 P1].
const SETS: usize = 6;
const CB_DWORDS: u32 = 17;

// Compiled group shapes (temporal / blur+post) — the shipping defaults,
// adopted from the 2026-08-09 B70 sweep (world parked, per-axis cells then
// the combined verify): temporal 16x16 → 8x8 read 0.233 → 0.224 ms and
// blur+post 16x8 → 16x16 read 0.294 → 0.260, landing the frd region at
// 0.488 from the pre-sweep 0.531 (32x16 blur tied 16x16 — the simpler
// square ships; 8x8 blur was WORST, so the two passes genuinely want
// different shapes). `FR_FRD_GROUP=TXxTY,BXxBY` is the sweep lever (the
// FR_LEAF discipline: loud on departure, loud + defaults on an illegal
// value; injected as the FRD_GTX/GTY/GBX/GBY defines, so every value is a
// new kernel variant — maiden-run discard on Arc applies).
const GROUP_T: (u32, u32) = (8, 8);
const GROUP_B: (u32, u32) = (16, 16);

fn group_lever() -> ((u32, u32), (u32, u32)) {
    let Ok(v) = std::env::var("FR_FRD_GROUP") else {
        return (GROUP_T, GROUP_B);
    };
    let parse_pair = |s: &str| -> Option<(u32, u32)> {
        let (x, y) = s.split_once('x')?;
        let (x, y) = (x.trim().parse::<u32>().ok()?, y.trim().parse::<u32>().ok()?);
        ((1..=1024).contains(&x) && (1..=1024).contains(&y) && x * y <= 1024).then_some((x, y))
    };
    let parsed = v
        .split_once(',')
        .and_then(|(t, b)| Some((parse_pair(t)?, parse_pair(b)?)));
    match parsed {
        Some((t, b)) => {
            eprintln!("frd: FR_FRD_GROUP={v} — temporal {}x{}, blur {}x{}", t.0, t.1, b.0, b.1);
            (t, b)
        }
        None => {
            eprintln!(
                "frd: FR_FRD_GROUP='{v}' is not TXxTY,BXxBY with x*y <= 1024 — using the \
                 shipping {}x{},{}x{}",
                GROUP_T.0, GROUP_T.1, GROUP_B.0, GROUP_B.1
            );
            (GROUP_T, GROUP_B)
        }
    }
}

/// Per-frame camera/light facts for `record` — one struct so the Stage-B
/// virtual-motion additions extend fields instead of churning the signature
/// (three call sites: nrd_frame_step's Frd arm + the F3/F4 harnesses).
#[derive(Clone, Copy)]
pub struct FrdFrame {
    /// Restart history (the caller's reset || !hist_valid).
    pub reset: bool,
    /// CAM_FAR — the bridge's 0.999·far range predicate (LOCKSTEP).
    pub far: f32,
    /// world→pixel projection scale (view_to_clip m11 · rh/2 — what turns a
    /// world blur radius into pixels).
    pub proj: f32,
    /// |O_cur − O_prev| world units — the camera-translation half of the
    /// specular parallax.
    pub cam_step: f32,
    /// World-space camera forward (the n·v proxy).
    pub cam_fwd: [f32; 3],
    /// 2× the light's per-frame angular delta in radians
    /// (frd::oracle::light_parallax × the FR_FRD_SUNPAR gain) — the
    /// moving-light half of the specular parallax; the two halves SUM in
    /// the kernel. 0.0 whenever the sun is static (bit-equal directions).
    pub light_par: f32,
}

pub struct FrdGpu {
    planes: Vec<Reg>,
    /// The recurrent slow history (single-buffered — pass 3 writes what
    /// pass 1 read this frame).
    slow: [Reg; 2], // [diff, spec]
    /// The anti-lag feedback plane (R8G8_UNORM g_diff/g_spec, single-
    /// buffered like `slow`: pass 1 reads it, pass 3 rewrites it — the
    /// brake's path around pass 3's missing meta UAV).
    antilag: Reg,
    /// hist[parity][role] — the fast/meta/geometry ping-pong.
    hist: [Vec<Reg>; 2],
    trans: Vec<Reg>,
    root_sig: ID3D12RootSignature,
    pso_temporal: ID3D12PipelineState,
    pso_blur: ID3D12PipelineState,
    pso_post: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    inc: u32,
    frame_idx: std::cell::Cell<u32>,
    /// Group shapes the PSOs compiled with (temporal / blur+post) — the
    /// dispatch dims must divide by THESE, never the consts.
    gt: (u32, u32),
    gb: (u32, u32),
    /// The F3 control arm: byte-copy IN→OUT instead of the passes.
    pub force_passthrough: std::cell::Cell<bool>,
    /// Native 16-bit shader ops (OPTIONS4) — the phase-D fp16 arm's probe;
    /// recorded now so the armed line names it. Phases B/C compile fp32.
    pub fp16: bool,
    pub rw: u32,
    pub rh: u32,
}

impl FrdGpu {
    /// `debug` = the session's --gpu-debug: FRD kernels compile the debug
    /// arm with the rest of the tree (dxc::compile's -Zi path), never a
    /// hard-coded -O3 that would leave these the one unsymbolized unit.
    pub fn new(
        device: &ID3D12Device,
        dxc: &dxc::Dxc,
        rw: u32,
        rh: u32,
        debug: bool,
    ) -> Result<Self> {
        let mut o4 = D3D12_FEATURE_DATA_D3D12_OPTIONS4::default();
        let fp16 = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS4,
                &mut o4 as *mut _ as *mut _,
                std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS4>() as u32,
            )
        }
        .is_ok()
            && o4.Native16BitShaderOpsSupported.as_bool()
            && !crate::frd::tuning().no_fp16;

        let ust = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let uaf = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        let mk = |f: DXGI_FORMAT| -> Result<Reg> {
            let res = d3d12::committed_tex(device, rw.max(1), rh.max(1), f, uaf, ust)?;
            Ok(Reg { res, state: std::cell::Cell::new(ust) })
        };
        let f16x4 = DXGI_FORMAT_R16G16B16A16_FLOAT;
        // Wire planes (lockstep with NrdGpu's plane_fmts — the bridge
        // kernels' typed UAV declarations are the third copy of this table).
        let wire_fmts = [
            f16x4,                         // IN_MV (2.5D)
            DXGI_FORMAT_R10G10B10A2_UNORM, // IN_NORMAL_ROUGHNESS (enc-2)
            DXGI_FORMAT_R32_FLOAT,         // IN_VIEWZ
            f16x4,                         // IN_DIFF (YCoCg + normHitDist)
            f16x4,                         // IN_SPEC
            f16x4,                         // OUT_DIFF
            f16x4,                         // OUT_SPEC
        ];
        let planes: Result<Vec<Reg>> = wire_fmts.iter().map(|&f| mk(f)).collect();
        let planes = planes?;
        let slow = [mk(f16x4)?, mk(f16x4)?];
        let antilag = mk(DXGI_FORMAT_R8G8_UNORM)?;
        let hist_fmts = [
            f16x4,                         // fast diff (YCoCg + Welford m2)
            f16x4,                         // fast spec
            // meta: n/frd::META_N_MAX(=63) per lane. NOT exact — n/63 only
            // lands on the UNORM8 1/255 grid at multiples of 21, so a
            // round-trip reads ~0.988 per accumulated frame (accepted:
            // fractional n exists anyway via confidence decay); the CAP is
            // what the encode enforces, and cb() clamps the lever to it.
            DXGI_FORMAT_R8G8_UNORM,
            DXGI_FORMAT_R10G10B10A2_UNORM, // prev nr snapshot (wire packing)
            DXGI_FORMAT_R32_FLOAT,         // prev view-Z snapshot
        ];
        let mk_set = || -> Result<Vec<Reg>> { hist_fmts.iter().map(|&f| mk(f)).collect() };
        let hist = [mk_set()?, mk_set()?];
        let trans: Result<Vec<Reg>> = [f16x4, f16x4, f16x4, f16x4].iter().map(|&f| mk(f)).collect();
        let trans = trans?;

        // --- Root signature (the bloom shape): SRV table + UAV table + root
        // constants. No samplers — every fetch is an explicit Load (per-foot
        // validity / per-tap bilateral tests forbid hardware bilinear).
        let ranges = [
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: SRVS as u32,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: 0,
            },
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                NumDescriptors: UAVS as u32,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: 0,
            },
        ];
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &ranges[0],
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &ranges[1],
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: CB_DWORDS,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            ..Default::default()
        };
        let mut blob: Option<ID3DBlob> = None;
        unsafe { D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None) }
            .map_err(|e| format!("frd: serialize root sig: {e}"))?;
        let blob = blob.unwrap();
        let root_sig: ID3D12RootSignature = unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()),
            )
        }
        .map_err(|e| format!("frd: create root sig: {e}"))?;

        // --- Kernels: [defines, frd_common.hlsli, pass source] at cs_6_5.
        // Phases B/C compile the fp32 arm unconditionally; phase D flips
        // FRD_FP16 + -enable-16bit-types off the probe (compile_args is the
        // hook). Tuning rides the CB, not defines — a lever must not
        // recompile.
        let (gt, gb) = group_lever();
        let pso_of = |src_body: &str, entry: &str| -> Result<ID3D12PipelineState> {
            let src = format!(
                "#define FRD_FP16 0\n#define FRD_GTX {}\n#define FRD_GTY {}\n\
                 #define FRD_GBX {}\n#define FRD_GBY {}\n{FRD_COMMON_HLSLI}\n{src_body}",
                gt.0, gt.1, gb.0, gb.1
            );
            let cs = dxc.compile_args(&src, entry, "cs_6_5", &format!("frd-{entry}"), debug, &[])?;
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: unsafe { std::mem::transmute_copy(&root_sig) },
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: cs.as_ptr() as *const _,
                    BytecodeLength: cs.len(),
                },
                ..Default::default()
            };
            unsafe { device.CreateComputePipelineState(&desc) }
                .map_err(|e| format!("frd: {entry} PSO: {e}"))
        };
        let pso_temporal = pso_of(FRD_TEMPORAL_HLSL, "cs_frd_temporal")?;
        let pso_blur = pso_of(FRD_BLUR_HLSL, "cs_frd_blur")?;
        let pso_post = pso_of(FRD_BLUR_HLSL, "cs_frd_post")?;

        // --- Shader-visible heap: 6 baked sets (3 passes × 2 parities),
        // each a full [SRVS | UAVS] window; slots a pass doesn't declare are
        // filled with a benign valid descriptor (never accessed).
        let inc = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: (SETS * SET_STRIDE) as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("frd: heap: {e}"))?;
        let base = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let at = |slot: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base.ptr + slot * inc as usize };
        let bake = |set: usize, srvs: &[&Reg], uavs: &[&Reg]| {
            let s0 = set * SET_STRIDE;
            for i in 0..SRVS {
                let r = srvs.get(i).copied().unwrap_or(&planes[P_IN_VIEWZ]);
                unsafe { device.CreateShaderResourceView(&r.res, None, at(s0 + i)) };
            }
            for i in 0..UAVS {
                let r = uavs.get(i).copied().unwrap_or(&planes[P_OUT_DIFF]);
                unsafe { device.CreateUnorderedAccessView(&r.res, None, None, at(s0 + SRVS + i)) };
            }
        };
        for parity in 0..2usize {
            // Pass 1 (frd_temporal.hlsl's t0..t12 / u0..u6).
            bake(
                parity,
                &[
                    &planes[P_IN_MV],
                    &planes[P_IN_NR],
                    &planes[P_IN_VIEWZ],
                    &planes[P_IN_DIFF],
                    &planes[P_IN_SPEC],
                    &slow[0],
                    &slow[1],
                    &hist[1 - parity][H_FAST_D],
                    &hist[1 - parity][H_FAST_S],
                    &hist[1 - parity][H_META],
                    &hist[1 - parity][H_NR],
                    &hist[1 - parity][H_VZ],
                    &antilag,
                ],
                &[
                    &trans[T_ACCUM_D],
                    &trans[T_ACCUM_S],
                    &hist[parity][H_FAST_D],
                    &hist[parity][H_FAST_S],
                    &hist[parity][H_META],
                    &hist[parity][H_NR],
                    &hist[parity][H_VZ],
                ],
            );
            // Pass 2 (frd_blur.hlsl's t0..t4 / u0..u1).
            bake(
                2 + parity,
                &[
                    &trans[T_ACCUM_D],
                    &trans[T_ACCUM_S],
                    &planes[P_IN_NR],
                    &planes[P_IN_VIEWZ],
                    &hist[parity][H_META],
                ],
                &[&trans[T_BLUR_D], &trans[T_BLUR_S]],
            );
            // Pass 3 (t0..t6 / u0..u4).
            bake(
                4 + parity,
                &[
                    &trans[T_BLUR_D],
                    &trans[T_BLUR_S],
                    &planes[P_IN_NR],
                    &planes[P_IN_VIEWZ],
                    &hist[parity][H_META],
                    &hist[parity][H_FAST_D],
                    &hist[parity][H_FAST_S],
                ],
                &[&planes[P_OUT_DIFF], &planes[P_OUT_SPEC], &slow[0], &slow[1], &antilag],
            );
        }

        Ok(Self {
            planes,
            slow,
            antilag,
            hist,
            trans,
            root_sig,
            pso_temporal,
            pso_blur,
            pso_post,
            heap,
            inc,
            frame_idx: std::cell::Cell::new(0),
            gt,
            gb,
            force_passthrough: std::cell::Cell::new(false),
            fp16,
            rw,
            rh,
        })
    }

    pub fn plane_in_mv(&self) -> &ID3D12Resource {
        &self.planes[P_IN_MV].res
    }
    pub fn plane_in_nr(&self) -> &ID3D12Resource {
        &self.planes[P_IN_NR].res
    }
    pub fn plane_in_viewz(&self) -> &ID3D12Resource {
        &self.planes[P_IN_VIEWZ].res
    }
    pub fn plane_in_diff(&self) -> &ID3D12Resource {
        &self.planes[P_IN_DIFF].res
    }
    pub fn plane_in_spec(&self) -> &ID3D12Resource {
        &self.planes[P_IN_SPEC].res
    }
    pub fn plane_out_diff(&self) -> &ID3D12Resource {
        &self.planes[P_OUT_DIFF].res
    }
    pub fn plane_out_spec(&self) -> &ID3D12Resource {
        &self.planes[P_OUT_SPEC].res
    }

    fn to_state(reg: &Reg, want: D3D12_RESOURCE_STATES, b: &mut Vec<D3D12_RESOURCE_BARRIER>) {
        let cur = reg.state.get();
        if cur != want {
            reg.state.set(want);
            b.push(d3d12::transition(&reg.res, cur, want));
        }
    }

    fn uav_barrier(reg: &Reg, b: &mut Vec<D3D12_RESOURCE_BARRIER>) {
        b.push(D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(reg.res.clone())),
                }),
            },
        });
    }

    /// Record one frame's FRD passes between the bridge's pack and out
    /// dispatches (the nrd_frame_step ordering); `f` carries the per-frame
    /// camera/light facts (see FrdFrame). Every touched resource is
    /// restored to UNORDERED_ACCESS.
    pub fn record(
        &self,
        list: &ID3D12GraphicsCommandList,
        _slot: usize,
        f: &FrdFrame,
    ) -> Result<()> {
        let _ev = super::pix::scope(list, c"frd");
        let npsr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let ust = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        if self.force_passthrough.get() {
            // The F3 control arm: OUT = byte-copy of IN — the bridge's delta
            // recompose is then exactly 0.0 (the plumbing proof).
            let mut b = Vec::new();
            Self::to_state(&self.planes[P_IN_DIFF], D3D12_RESOURCE_STATE_COPY_SOURCE, &mut b);
            Self::to_state(&self.planes[P_IN_SPEC], D3D12_RESOURCE_STATE_COPY_SOURCE, &mut b);
            Self::to_state(&self.planes[P_OUT_DIFF], D3D12_RESOURCE_STATE_COPY_DEST, &mut b);
            Self::to_state(&self.planes[P_OUT_SPEC], D3D12_RESOURCE_STATE_COPY_DEST, &mut b);
            unsafe { list.ResourceBarrier(&b) };
            unsafe {
                list.CopyResource(&self.planes[P_OUT_DIFF].res, &self.planes[P_IN_DIFF].res);
                list.CopyResource(&self.planes[P_OUT_SPEC].res, &self.planes[P_IN_SPEC].res);
            }
            let mut b = Vec::new();
            for i in [P_IN_DIFF, P_IN_SPEC, P_OUT_DIFF, P_OUT_SPEC] {
                Self::to_state(&self.planes[i], ust, &mut b);
            }
            unsafe { list.ResourceBarrier(&b) };
            return Ok(());
        }

        let parity = (self.frame_idx.get() & 1) as usize;
        let t = crate::frd::tuning();
        // flags: bit 0 = reset | bit 1 = firefly pre-clamp (the
        // --frd-[no-]anti-firefly lever, DEFAULT ON — it is the bright-
        // trail killer) | bit 3 = anti-lag (FR_FRD_ANTILAG=off disarms).
        let flags = (f.reset as u32)
            | ((t.anti_firefly.unwrap_or(true) as u32) << 1)
            | ((crate::frd::antilag_enabled() as u32) << 3);
        let cb = |pass_scale: f32, salt: u32| -> [u32; CB_DWORDS as usize] {
            [
                self.rw,
                self.rh,
                flags,
                self.frame_idx.get(),
                f.far.to_bits(),
                // Clamped to the meta wire cap: a larger value would truncate
                // at the n/63 UNORM encode anyway — clamp HERE so the CB and
                // the wire agree (the parse already noted it).
                (t.max_accum_frames
                    .map_or(crate::frd::MAX_ACCUM_FRAMES, |v| {
                        (v as f32).min(crate::frd::META_N_MAX)
                    }))
                .to_bits(),
                (t.fast_frames.map_or(crate::frd::FAST_FRAMES, |v| v as f32)).to_bits(),
                (t.clamp_sigma.unwrap_or(crate::frd::CLAMP_SIGMA)).to_bits(),
                f.cam_step.to_bits(),
                f.cam_fwd[0].to_bits(),
                f.cam_fwd[1].to_bits(),
                f.cam_fwd[2].to_bits(),
                f.proj.to_bits(),
                (t.blur_radius.unwrap_or(crate::frd::R_MAX)).to_bits(),
                pass_scale.to_bits(),
                salt,
                f.light_par.to_bits(),
            ]
        };
        let gpu0 = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        let bind_set = |set: usize| unsafe {
            let s = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu0.ptr + (set * SET_STRIDE) as u64 * self.inc as u64,
            };
            list.SetComputeRootDescriptorTable(0, s);
            list.SetComputeRootDescriptorTable(
                1,
                D3D12_GPU_DESCRIPTOR_HANDLE { ptr: s.ptr + SRVS as u64 * self.inc as u64 },
            );
        };
        let (tgx, tgy) = (self.rw.div_ceil(self.gt.0), self.rh.div_ceil(self.gt.1));
        let (bgx, bgy) = (self.rw.div_ceil(self.gb.0), self.rh.div_ceil(self.gb.1));
        unsafe {
            list.SetComputeRootSignature(&self.root_sig);
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
        }

        // Pass 1 — temporal. SRV inputs: 5 wire IN (the transitions double
        // as the pack→read hazard sync), prev parity set, slow (recurrent).
        {
            let _p = super::pix::scope(list, c"frd-temporal");
            let mut b = Vec::new();
            for i in [P_IN_MV, P_IN_NR, P_IN_VIEWZ, P_IN_DIFF, P_IN_SPEC] {
                Self::to_state(&self.planes[i], npsr, &mut b);
            }
            for r in &self.hist[1 - parity] {
                Self::to_state(r, npsr, &mut b);
            }
            for r in &self.slow {
                Self::to_state(r, npsr, &mut b);
            }
            Self::to_state(&self.antilag, npsr, &mut b);
            unsafe { list.ResourceBarrier(&b) };
            let c = cb(1.0, 0);
            unsafe {
                list.SetPipelineState(&self.pso_temporal);
                bind_set(parity);
                list.SetComputeRoot32BitConstants(2, CB_DWORDS, c.as_ptr() as *const _, 0);
                list.Dispatch(tgx, tgy, 1);
            }
        }

        // Pass 2 — blur. accum + this frame's meta become SRVs (the
        // transitions fence pass 1's UAV writes).
        {
            let _p = super::pix::scope(list, c"frd-blur");
            let mut b = Vec::new();
            Self::to_state(&self.trans[T_ACCUM_D], npsr, &mut b);
            Self::to_state(&self.trans[T_ACCUM_S], npsr, &mut b);
            Self::to_state(&self.hist[parity][H_META], npsr, &mut b);
            unsafe { list.ResourceBarrier(&b) };
            let c = cb(1.0, 17);
            unsafe {
                list.SetPipelineState(&self.pso_blur);
                bind_set(2 + parity);
                list.SetComputeRoot32BitConstants(2, CB_DWORDS, c.as_ptr() as *const _, 0);
                list.Dispatch(bgx, bgy, 1);
            }
        }

        // Pass 3 — post + clamp + the recurrent feedback. blur + this
        // frame's fast become SRVs; slow returns to UA for the write.
        {
            let _p = super::pix::scope(list, c"frd-post");
            let mut b = Vec::new();
            Self::to_state(&self.trans[T_BLUR_D], npsr, &mut b);
            Self::to_state(&self.trans[T_BLUR_S], npsr, &mut b);
            Self::to_state(&self.hist[parity][H_FAST_D], npsr, &mut b);
            Self::to_state(&self.hist[parity][H_FAST_S], npsr, &mut b);
            for r in &self.slow {
                Self::to_state(r, ust, &mut b);
            }
            Self::to_state(&self.antilag, ust, &mut b);
            unsafe { list.ResourceBarrier(&b) };
            let c = cb(1.7, 41);
            unsafe {
                list.SetPipelineState(&self.pso_post);
                bind_set(4 + parity);
                list.SetComputeRoot32BitConstants(2, CB_DWORDS, c.as_ptr() as *const _, 0);
                list.Dispatch(bgx, bgy, 1);
            }
        }

        // Restore rest state; the OUT planes never left UA, so cs_nrd_out
        // needs explicit UAV barriers to see pass 3's writes.
        let mut b = Vec::new();
        for i in [P_IN_MV, P_IN_NR, P_IN_VIEWZ, P_IN_DIFF, P_IN_SPEC] {
            Self::to_state(&self.planes[i], ust, &mut b);
        }
        for r in self.hist.iter().flatten() {
            Self::to_state(r, ust, &mut b);
        }
        for r in &self.trans {
            Self::to_state(r, ust, &mut b);
        }
        Self::uav_barrier(&self.planes[P_OUT_DIFF], &mut b);
        Self::uav_barrier(&self.planes[P_OUT_SPEC], &mut b);
        unsafe { list.ResourceBarrier(&b) };
        self.frame_idx.set(self.frame_idx.get().wrapping_add(1));
        Ok(())
    }
}
