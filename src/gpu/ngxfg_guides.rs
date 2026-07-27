//! FG guide conversion for the raw-NGX DLSS-G evaluate (`--fg` DLSS sessions,
//! `gpu/mod.rs::ngxfg_dispatch`): ONE fused compute pass producing the two
//! planes the FG snippet needs but the RR plane set does not carry —
//!
//! 1. **Clip depth** (round 1): DLSS-RR consumes the depth plane through the
//!    RR-specific LINEAR-depth tag, so it holds unbounded linear view-Z
//!    (0..far). The FG snippet's plain `Depth` input has DLSS-SR's contract —
//!    a [0,1] depth buffer CONSISTENT WITH THE SUPPLIED CAMERA MATRICES
//!    (ProgrammingGuideDLSS_G.md §5.1) — and dlssg-to-fsr3 confirms real SL
//!    titles feed a projection-matrix-consistent depth buffer (it derives
//!    near/far back out of the matrix). The mapping here is the EXACT
//!    z-mapping of the `Mat4::perspective_lh` matrix riding the same
//!    dispatch: `d = A + B/z`, `A = far/(far−near)`, `B = −near·far/(far−near)`
//!    — near → 0, sky (`z = far`) → 1, `depthInverted` 0. Deliberately NOT
//!    `xess::view_z_to_clip_depth` (REVERSED-Z, inconsistent with these
//!    matrices).
//!
//! 2. **Reflection-aware motion vectors** (round 2 — the DamagedHelmet
//!    sky-reflection swim): the MV plane describes SURFACE motion, but a
//!    mirror pixel's CONTENT is the reflection — a VIRTUAL IMAGE at path
//!    depth `t_surface + t_reflection` (planar unfold: the image lies along
//!    the primary ray, beyond the surface; reflected sky ⇒ `t_v ≈ far` ⇒
//!    near-zero translation parallax — exactly the "reflection drifts
//!    opposite the surface" the user observed strafing). Warping with the
//!    surface MV drags the reflection with the helmet on every generated
//!    frame; real frames snap it back = swimming at half the present rate.
//!    We can compute the virtual-image MV EXACTLY because the reflection
//!    ray's hit distance is already captured per pixel (`spec_hit_t`, the RR
//!    guide plane; 0 = no reflection ray, `far` = reflected sky — pinned in
//!    `shade.rs::PrimarySurface::spec_t`). The output is
//!    `lerp(mv_surface, mv_virtual, w)` with `w` = how much of the pixel's
//!    LOOK is the reflection: `lum(spec_alb)/(lum(diff_alb)+lum(spec_alb))`
//!    damped above `ROUGH_LO..ROUGH_HI` roughness — on the metal helmet
//!    diffuse ≈ 0 so w ≈ 1. (True RADIANCE-weighted w — the user's exact
//!    formulation, which would also catch a blinding glint on a glossy
//!    dielectric — needs a dd/ds/ind_s-style capture armed in DLSS sessions,
//!    the FLAG_FSR_SIG precedent: documented follow-on, not v1.) The blended
//!    plane is FG-ONLY — RR keeps its own MV plane untouched (RR is trained
//!    for surface MVs + the spec-hit guide).
//!
//! Compiled with D3DCompile (fxc, cs_5_0) like `gpu/bloom.rs` — a CPU-fed RR
//! session never loads DXC, and this pass must run in all three RR arms
//! (CPU-fed, wavefront-fed, DXR-fed): it records inside `ngxfg_dispatch`,
//! the one site they share. `self_test` (run by `--check`, DLL- and
//! GPU-free) pins the Rust mirrors; the HLSL is their literal twin — change
//! both together (the clouds-wind idiom).

use super::d3d12::{self, Result};
use super::tonemap::compile;
use glam::{Mat4, Vec3A, Vec4};
use windows::core::s;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// Roughness band over which the virtual-image blend fades out: past
/// ROUGH_HI the one-VNDF-sample `spec_hit_t` is too stochastic to steer a
/// warp (and the blurred reflection hides the surface-MV error anyway).
/// Mirrored as HLSL literals below — change all four together.
pub const ROUGH_LO: f32 = 0.25;
pub const ROUGH_HI: f32 = 0.6;

// The Rust twins are `clip_depth` / `virtual_prev_px` / `rmv_weight` below —
// change both together.
const HLSL: &str = r#"
Texture2D<float>  zsrc   : register(t0); // linear view-Z (P_DEPTH)
Texture2D<float2> mvsrc  : register(t1); // surface MVs, prev - cur, pixels (P_MVEC)
Texture2D<float>  hitsrc : register(t2); // reflection-ray hit distance (P_SPEC_HIT)
Texture2D<float4> dalb   : register(t3); // diffuse albedo, linear (P_ALBEDO)
Texture2D<float4> salb   : register(t4); // specular albedo / F0, linear (P_SPEC_ALBEDO)
Texture2D<float4> nrough : register(t5); // normal.xyz + roughness.w (P_NORMAL_ROUGH)
RWTexture2D<float>  zdst  : register(u0); // [0,1] clip depth out
RWTexture2D<float2> mvdst : register(u1); // FG motion vectors out
cbuffer C : register(b0) {
    uint   w; uint h; float A; float B; // clip depth d = A + B/z
    float4 m0; float4 m1; float4 m2; float4 m3; // world -> PREV clip (glam columns)
    float4 org;  // camera origin
    float4 fwd;  // unit forward
    float4 rgt;  // right * tan(fov/2) * aspect (CamBasis pre-scaling)
    float4 upv;  // up * tan(fov/2)
    uint   rmv; float cam_far; float2 _pad;
}
[numthreads(8, 8, 1)]
void cs_guides(uint3 id : SV_DispatchThreadID) {
    if (id.x >= w || id.y >= h) return;
    float z = zsrc[id.xy];
    zdst[id.xy] = saturate(A + B / max(z, 1e-6f));
    float2 mv = mvsrc[id.xy];
    float t_r = hitsrc[id.xy];
    if (rmv != 0 && t_r > 0.0f && z > 0.0f) {
        // Virtual-image reprojection — the Rust twin is virtual_prev_px.
        // Pixel center; the surface MV is anchored at the jittered sample
        // position instead, a sub-pixel mismatch the blend absorbs.
        float2 c = float2(id.xy) + 0.5f;
        float ndx = c.x * (2.0f / w) - 1.0f;
        float ndy = 1.0f - c.y * (2.0f / h);
        float3 du = normalize(fwd.xyz + rgt.xyz * ndx + upv.xyz * ndy);
        float ray_t = z / dot(du, fwd.xyz); // view-Z -> Euclidean ray t
        // A MISSED reflection is the SKY — a virtual image at INFINITY, whose
        // translation parallax is exactly zero (only rotation moves it). The
        // pack cannot say "infinity" though: it clamps a missed reflection to
        // CAM_FAR because that lane's OTHER consumer is the depth delta, which
        // wants far. Feeding that finite distance into the point form below
        // gives the sky real parallax it does not have — at far = 2*diag (~138
        // world units) and a 3841 px render width that is ~28 px of motion
        // error per world unit of camera translation, which warps the sun's
        // specular highlight tens of pixels on every generated frame and snaps
        // it back on the next real one. Measured on the world's DamagedHelmet:
        // strafing is far worse than rotating, which is the signature.
        //
        // So take the analytic LIMIT of the same formula instead of a finite
        // stand-in: as t_r -> inf, V -> org + du*inf, i.e. a pure DIRECTION.
        // Projecting a direction is the point projection with the translation
        // column dropped (a w = 0 homogeneous point), which yields exactly the
        // rotation-only reprojection the sky deserves.
        bool sky_refl = t_r >= cam_far;
        float3 V = org.xyz + du * (ray_t + t_r); // planar-unfold virtual point
        float4 pc = sky_refl ? (m0 * du.x + m1 * du.y + m2 * du.z)
                             : (m0 * V.x + m1 * V.y + m2 * V.z + m3);
        if (pc.w > 1e-6f) {
            float2 pndc = pc.xy / pc.w;
            float2 ppx = float2((pndc.x + 1.0f) * 0.5f * w,
                                (1.0f - pndc.y) * 0.5f * h);
            float2 mv_v = ppx - c; // current -> previous, pixels (the plane's convention)
            const float3 LUM = float3(0.2126f, 0.7152f, 0.0722f);
            float ld = dot(dalb[id.xy].rgb, LUM);
            float ls = dot(salb[id.xy].rgb, LUM);
            float rough = nrough[id.xy].w;
            float wgt = saturate(ls / (ld + ls + 1e-4f))
                      * (1.0f - smoothstep(0.25f, 0.6f, rough)); // ROUGH_LO..ROUGH_HI
            mv = lerp(mv, mv_v, wgt);
        }
    }
    mvdst[id.xy] = mv;
}
"#;

/// The CPU mirror of `cs_guides`' depth half — the value
/// `perspective_lh(_, _, near, far)` produces as `clip.z / clip.w` at view
/// depth `view_z` (see `self_test` for the matrix-consistency pin).
pub fn clip_depth(view_z: f32, near: f32, far: f32) -> f32 {
    let a = far / (far - near);
    let b = -near * far / (far - near);
    (a + b / view_z.max(1e-6)).clamp(0.0, 1.0)
}

/// The CPU mirror of `cs_guides`' virtual-image reprojection: the PREVIOUS
/// frame's pixel position of the virtual point behind pixel center (cx, cy),
/// or None when it lands behind the previous image plane. `right_s`/`up_s`
/// carry the CamBasis pre-scaling (tan(fov/2)·aspect / tan(fov/2));
/// `m` = world → previous clip (glam column-vector convention).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn virtual_prev_px(
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    view_z: f32,
    t_r: f32,
    cam_far: f32,
    origin: Vec3A,
    fwd: Vec3A,
    right_s: Vec3A,
    up_s: Vec3A,
    m: &Mat4,
) -> Option<(f32, f32)> {
    let ndx = cx * (2.0 / w) - 1.0;
    let ndy = 1.0 - cy * (2.0 / h);
    let du = (fwd + right_s * ndx + up_s * ndy).normalize();
    let ray_t = view_z / du.dot(fwd);
    let v = origin + du * (ray_t + t_r);
    // t_r >= cam_far IS the pack's "reflection missed" encoding: the sky is at
    // infinity, so project the DIRECTION (w = 0 — the translation column drops
    // out) for rotation-only parallax. See the cs_guides twin for why a finite
    // stand-in warps the sun's highlight.
    let pc = if t_r >= cam_far {
        *m * Vec4::new(du.x, du.y, du.z, 0.0)
    } else {
        *m * Vec4::new(v.x, v.y, v.z, 1.0)
    };
    if pc.w <= 1e-6 {
        return None;
    }
    let (px, py) = (pc.x / pc.w, pc.y / pc.w);
    Some(((px + 1.0) * 0.5 * w, (1.0 - py) * 0.5 * h))
}

/// The CPU mirror of `cs_guides`' blend weight: how much of the pixel's look
/// is the reflection — specular vs diffuse albedo luminance, damped over
/// [ROUGH_LO, ROUGH_HI].
pub fn rmv_weight(lum_diff: f32, lum_spec: f32, rough: f32) -> f32 {
    let t = ((rough - ROUGH_LO) / (ROUGH_HI - ROUGH_LO)).clamp(0.0, 1.0);
    let smooth = 1.0 - t * t * (3.0 - 2.0 * t);
    (lum_spec / (lum_diff + lum_spec + 1e-4)).clamp(0.0, 1.0) * smooth
}

/// Root constants for `cs_guides` — field order IS the cbuffer layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuideParams {
    pub w: u32,
    pub h: u32,
    pub a: f32,
    pub b: f32,
    /// world → PREVIOUS clip, glam columns.
    pub m: [[f32; 4]; 4],
    pub org: [f32; 4],
    pub fwd: [f32; 4],
    pub rgt: [f32; 4],
    pub upv: [f32; 4],
    pub rmv: u32,
    /// The far plane the pack clamps a MISSED reflection to (`spec_hit_t`'s
    /// "reflected sky" value). The kernel needs it to tell a genuine hit at
    /// distance from a miss, because one lane carries both — see the sky-miss
    /// branch in `cs_guides`. Rides the old `_pad`, so the layout is unmoved.
    pub cam_far: f32,
    pub _pad: [f32; 2],
}
const PARAM_DWORDS: u32 = (std::mem::size_of::<GuideParams>() / 4) as u32;
const _: () = assert!(std::mem::size_of::<GuideParams>() == 40 * 4);

/// Kernel-order indices into the source-plane array `ensure` takes — must
/// match `rr::RrResources::guide_inputs` (which returns them in this order).
pub const SRC_PLANES: usize = 6;
const SRC_FORMATS: [DXGI_FORMAT; SRC_PLANES] = [
    DXGI_FORMAT_R32_FLOAT,          // linear view-Z
    DXGI_FORMAT_R16G16_FLOAT,       // surface MVs
    DXGI_FORMAT_R16_FLOAT,          // spec hit distance
    DXGI_FORMAT_R8G8B8A8_UNORM,     // diffuse albedo
    DXGI_FORMAT_R8G8B8A8_UNORM,     // specular albedo
    DXGI_FORMAT_R16G16B16A16_FLOAT, // normal + roughness
];

/// One dispatch: read six RR planes, write the [0,1] clip-depth plane and
/// the reflection-blended MV plane the NGX evaluate consumes. Planes +
/// descriptors are (re)built by `ensure` at feature-creation time (first
/// dispatch / after a resize), so no descriptor is ever rewritten under an
/// in-flight frame.
pub struct GuidePass {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// [0..6] = SRVs of the RR planes (kernel order), [6] = clip UAV,
    /// [7] = mv UAV.
    heap: ID3D12DescriptorHeap,
    inc: u32,
    /// Render-res R32F / RG16F; rest UNORDERED_ACCESS between frames
    /// (`record` hands them to the evaluate as NON_PIXEL_SHADER_RESOURCE,
    /// `restore` puts them back).
    clip: Option<ID3D12Resource>,
    mv: Option<ID3D12Resource>,
    w: u32,
    h: u32,
}

impl GuidePass {
    pub fn new(device: &ID3D12Device) -> Result<GuidePass> {
        // Root signature: [0] SRV table (t0..t5), [1] UAV table (u0..u1),
        // [2] the GuideParams root constants (b0). The bloom shape, minus
        // its sampler.
        let srv_range = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: SRC_PLANES as u32,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        }];
        let uav_range = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: 2,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        }];
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: srv_range.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: uav_range.as_ptr(),
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
                        Num32BitValues: PARAM_DWORDS,
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
        let mut errb: Option<ID3DBlob> = None;
        unsafe {
            D3D12SerializeRootSignature(
                &rs_desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                Some(&mut errb),
            )
        }
        .map_err(|e| format!("ngxfg-guides: D3D12SerializeRootSignature: {e}"))?;
        let blob = blob.unwrap();
        let root_sig: ID3D12RootSignature = unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    blob.GetBufferPointer() as *const u8,
                    blob.GetBufferSize(),
                ),
            )
        }
        .map_err(|e| format!("ngxfg-guides: CreateRootSignature: {e}"))?;

        let cs = compile(HLSL, s!("cs_guides"), s!("cs_5_0"), "ngxfg-guides cs_guides")?;
        let pso: ID3D12PipelineState = unsafe {
            device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root_sig),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: cs.GetBufferPointer(),
                    BytecodeLength: cs.GetBufferSize(),
                },
                ..Default::default()
            })
        }
        .map_err(|e| format!("ngxfg-guides: CreateComputePipelineState: {e}"))?;

        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: SRC_PLANES as u32 + 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .map_err(|e| format!("ngxfg-guides: CreateDescriptorHeap: {e}"))?;
        let inc = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };

        Ok(GuidePass { root_sig, pso, heap, inc, clip: None, mv: None, w: 0, h: 0 })
    }

    /// (Re)build the output planes at the feature's render res and rewrite
    /// ALL descriptors — the SRVs must be rewritten even at unchanged dims,
    /// because a resize rebuilt `RrResources` and `srcs` are new resources.
    /// Only called at NGX-feature creation (first dispatch ever, or the
    /// first after a resize's `wait_idle`), so no in-flight frame references
    /// these descriptors.
    pub fn ensure(
        &mut self,
        device: &ID3D12Device,
        w: u32,
        h: u32,
        srcs: [&ID3D12Resource; SRC_PLANES],
    ) -> Result<()> {
        if self.clip.is_none() || self.w != w || self.h != h {
            self.clip = Some(d3d12::committed_tex(
                device,
                w,
                h,
                DXGI_FORMAT_R32_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?);
            // RG16F typed UAV STORE is an optional D3D12 cap (the mandatory
            // set is RGBA32/RGBA16/RGBA8/R32/R16/R8 — two-channel formats
            // need CheckFormatSupport, which trace::wire_feed's format gate
            // performs for the feed kernels). Deliberately unchecked here:
            // this pass exists only in raw-NGX DLSS-G sessions, i.e. NVIDIA
            // with the NDA SDK, where support is universal. Port the pass to
            // a cross-vendor path and it needs the wire_feed gate.
            self.mv = Some(d3d12::committed_tex(
                device,
                w,
                h,
                DXGI_FORMAT_R16G16_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?);
            self.w = w;
            self.h = h;
        }
        let base = unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() };
        let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + (i * self.inc) as usize,
        };
        for (i, (src, fmt)) in srcs.iter().zip(SRC_FORMATS).enumerate() {
            unsafe {
                device.CreateShaderResourceView(
                    *src,
                    Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: fmt,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                        },
                    }),
                    at(i as u32),
                );
            }
        }
        let uav = |res: &ID3D12Resource, fmt, slot: u32| unsafe {
            device.CreateUnorderedAccessView(
                res,
                None,
                Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: fmt,
                    ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                    ..Default::default()
                }),
                at(slot),
            );
        };
        uav(self.clip.as_ref().unwrap(), DXGI_FORMAT_R32_FLOAT, SRC_PLANES as u32);
        uav(self.mv.as_ref().unwrap(), DXGI_FORMAT_R16G16_FLOAT, SRC_PLANES as u32 + 1);
        Ok(())
    }

    /// Record the conversion. The source planes already rest
    /// NON_PIXEL_SHADER_RESOURCE (the state a compute SRV read wants); both
    /// output planes go UAV → NON_PIXEL_SHADER_RESOURCE for the evaluate
    /// that follows. Pair with `restore` after the evaluate.
    pub fn record(&self, list: &ID3D12GraphicsCommandList, p: &GuideParams) {
        let gpu0 = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        let uavs = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu0.ptr + (SRC_PLANES as u32 * self.inc) as u64,
        };
        unsafe {
            list.SetComputeRootSignature(&self.root_sig);
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetPipelineState(&self.pso);
            list.SetComputeRootDescriptorTable(0, gpu0);
            list.SetComputeRootDescriptorTable(1, uavs);
            list.SetComputeRoot32BitConstants(2, PARAM_DWORDS, p as *const _ as *const _, 0);
            list.Dispatch(self.w.div_ceil(8), self.h.div_ceil(8), 1);
            list.ResourceBarrier(&[
                d3d12::transition(
                    self.clip.as_ref().unwrap(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
                d3d12::transition(
                    self.mv.as_ref().unwrap(),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
            ]);
        }
    }

    /// Back to the UAV rest state after the evaluate consumed the planes.
    pub fn restore(&self, list: &ID3D12GraphicsCommandList) {
        unsafe {
            list.ResourceBarrier(&[
                d3d12::transition(
                    self.clip.as_ref().unwrap(),
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
                d3d12::transition(
                    self.mv.as_ref().unwrap(),
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
            ]);
        }
    }

    pub fn clip(&self) -> &ID3D12Resource {
        self.clip.as_ref().expect("ensure before clip")
    }
    pub fn mv(&self) -> &ID3D12Resource {
        self.mv.as_ref().expect("ensure before mv")
    }
}

/// Pure-math gates (run by `--check`, DLL- and GPU-free).
///
/// Depth half: `clip_depth` IS the z-mapping of the `perspective_lh` matrix
/// the NGX dispatch carries (matrix-consistency sweep + the
/// near/far/harmonic-midpoint anchors).
///
/// MV half: the virtual-image reprojection against `CamBasis::project`
/// itself — (b) a static camera reprojects every virtual point to its own
/// pixel (zero MV at any reflection depth); (c) `t_r = 0` degenerates to the
/// surface reprojection `render.rs::write_gbuf_hit` computes, pinned against
/// the REAL `CamBasis::project` on a dollied+yawed pose (the continuity pin
/// that proves the mirrored matrix math matches the plane's convention);
/// (d) lateral strafe with the reflection at `far` moves the virtual pixel
/// far less than the surface pixel — the user's strafe observation, as a
/// gate; (e) the blend-weight anchors.
pub fn self_test() -> std::result::Result<(), String> {
    use crate::camera::Camera;
    // ---- depth half ----
    let cases = [(0.0421f32, 421.0f32), (0.1, 10_000.0), (1.0, 100.0)];
    for &(near, far) in &cases {
        let d_near = clip_depth(near, near, far);
        let d_far = clip_depth(far, near, far);
        if d_near.abs() > 1e-6 {
            return Err(format!("clip_depth(near) = {d_near}, want 0 (near={near} far={far})"));
        }
        if (d_far - 1.0).abs() > 1e-6 {
            return Err(format!("clip_depth(far) = {d_far}, want 1 (near={near} far={far})"));
        }
        let zm = 2.0 * near * far / (near + far);
        let d_mid = clip_depth(zm, near, far);
        if (d_mid - 0.5).abs() > 1e-5 {
            return Err(format!("clip_depth(harmonic mid) = {d_mid}, want 0.5"));
        }
        let m = Mat4::perspective_lh(1.0, 16.0 / 9.0, near, far);
        let mut prev = -1.0f32;
        for i in 0..64 {
            let t = i as f32 / 63.0;
            let z = near * (far / near).powf(t);
            let clip = m * Vec4::new(0.0, 0.0, z, 1.0);
            let want = clip.z / clip.w;
            let got = clip_depth(z, near, far);
            if (got - want).abs() > 1e-5 {
                return Err(format!(
                    "clip_depth({z}) = {got} but perspective_lh gives {want} \
                     (near={near} far={far})"
                ));
            }
            if got < prev {
                return Err(format!("clip_depth not monotone at z={z}"));
            }
            prev = got;
        }
    }
    if clip_depth(1e30, 0.1, 10_000.0) != 1.0 {
        return Err("clip_depth beyond far must saturate to 1.0".into());
    }

    // ---- MV half ----
    let (rw, rh) = (432usize, 324usize);
    let (near, far) = (0.05f32, 5000.0f32);
    let cam = Camera::look_at(Vec3A::new(4.0, 2.0, 4.0), Vec3A::new(0.0, 1.0, 0.0), 0.9);
    // The dispatch's basis derivation (fc.right/up are unit; pre-scale like
    // CamBasis).
    let basis_fields = |c: &Camera| {
        let f = c.forward();
        let r = f.cross(Vec3A::Y).normalize();
        let u = r.cross(f);
        let tanh = (c.fov_y * 0.5).tan();
        (c.pos, f, r * (tanh * rw as f32 / rh as f32), u * tanh)
    };
    let (org, fwd, rgt, upv) = basis_fields(&cam);
    let mats = crate::dlss::cam_matrices(&cam, rw, rh, near, far);
    let world_to_clip = mats.view_to_clip * mats.world_to_view;

    // (b) static camera: prev == current ⇒ the virtual point reprojects to
    // its own pixel at ANY reflection depth (a still frame's reflection has
    // zero MV — the model's sanity anchor).
    for &(cx, cy, t_r) in
        &[(216.5f32, 162.5f32, 0.0f32), (40.5, 300.5, 3.0), (400.5, 20.5, far)]
    {
        let (px, py) = virtual_prev_px(
            cx, cy, rw as f32, rh as f32, 7.0, t_r, far, org, fwd, rgt, upv, &world_to_clip,
        )
        .ok_or("static virtual point behind its own camera")?;
        if (px - cx).abs() > 2e-2 || (py - cy).abs() > 2e-2 {
            return Err(format!(
                "static camera: virtual pixel drifted ({cx},{cy}) -> ({px},{py}) at t_r={t_r}"
            ));
        }
    }

    // A genuinely moved previous pose for (c) and (d').
    let prev_cam = Camera {
        pos: cam.pos + Vec3A::new(0.12, 0.02, -0.05),
        yaw: cam.yaw + 0.01,
        pitch: cam.pitch - 0.004,
        fov_y: cam.fov_y,
    };
    let prev_basis = prev_cam.basis(rw, rh);
    let prev_mats = crate::dlss::cam_matrices(&prev_cam, rw, rh, near, far);
    let world_to_prev_clip = prev_mats.view_to_clip * prev_mats.world_to_view;

    // (c) t_r = 0 degenerates to the surface reprojection — pinned against
    // CamBasis::project (the exact function the MV fill sites use), which
    // proves the matrix route and the basis route agree in the plane's own
    // convention.
    for &(cx, cy, view_z) in &[(100.5f32, 80.5f32, 3.0f32), (300.5, 250.5, 11.0), (16.5, 16.5, 0.8)]
    {
        let ndx = cx * (2.0 / rw as f32) - 1.0;
        let ndy = 1.0 - cy * (2.0 / rh as f32);
        let du = (fwd + rgt * ndx + upv * ndy).normalize();
        let p = org + du * (view_z / du.dot(fwd));
        let (ex, ey) = prev_basis
            .project(p - prev_basis.origin)
            .ok_or("surface point behind prev camera")?;
        let (px, py) = virtual_prev_px(
            cx, cy, rw as f32, rh as f32, view_z, 0.0, far, org, fwd, rgt, upv,
            &world_to_prev_clip,
        )
        .ok_or("virtual point behind prev camera")?;
        if (px - ex).abs() > 0.05 || (py - ey).abs() > 0.05 {
            return Err(format!(
                "t_r=0 continuity: matrix route ({px},{py}) vs CamBasis::project ({ex},{ey})"
            ));
        }
    }

    // (d) PURE LATERAL STRAFE with a MISSED reflection (the sky). The camera
    // only translates, and the sky is at infinity, so the correct virtual MV
    // is EXACTLY ZERO — hence an ABSOLUTE bound, not a fraction of the surface
    // MV. That distinction is the whole gate: the old `mv_virt <= 0.05 *
    // mv_surf` form passed while production visibly warped, because zero is
    // the right answer and any percentage of a large surface MV clears a
    // percentage bar. It also ran at far = 5000 while the renderer ships
    // far = 2*diag (~138 on the world) — 36x more distant, so "reflection at
    // far" impersonated infinity in the gate far better than it ever did in
    // the product. Both halves are fixed here: the bound is absolute, and the
    // sweep includes a PRODUCTION-scale far.
    //
    // The sky-miss branch makes this exact rather than merely small: a
    // direction reprojected under a pure translation is unchanged.
    for &far_probe in &[far, 2.0 * 69.0, 2.0 * 10.0] {
        let strafe_cam = Camera { pos: cam.pos + rgt.normalize() * 0.3, ..cam };
        let strafe_mats = crate::dlss::cam_matrices(&strafe_cam, rw, rh, near, far_probe);
        let world_to_strafe_clip = strafe_mats.view_to_clip * strafe_mats.world_to_view;
        let strafe_basis = strafe_cam.basis(rw, rh);
        let (cx, cy, view_z) = (216.5f32, 162.5f32, 2.5f32);
        let ndx = cx * (2.0 / rw as f32) - 1.0;
        let ndy = 1.0 - cy * (2.0 / rh as f32);
        let du = (fwd + rgt * ndx + upv * ndy).normalize();
        let p = org + du * (view_z / du.dot(fwd));
        let (sx, sy) = strafe_basis
            .project(p - strafe_basis.origin)
            .ok_or("strafe: surface point behind camera")?;
        let mv_surf = ((sx - cx).powi(2) + (sy - cy).powi(2)).sqrt();
        // t_r == far_probe IS the miss encoding the pack produces.
        let (vx, vy) = virtual_prev_px(
            cx, cy, rw as f32, rh as f32, view_z, far_probe, far_probe, org, fwd, rgt, upv,
            &world_to_strafe_clip,
        )
        .ok_or("strafe: virtual point behind camera")?;
        let mv_virt = ((vx - cx).powi(2) + (vy - cy).powi(2)).sqrt();
        if mv_surf < 5.0 {
            return Err(format!(
                "strafe gate is vacuous — surface MV only {mv_surf} px (probe too gentle)"
            ));
        }
        if mv_virt > 0.05 {
            return Err(format!(
                "strafe (far={far_probe}): reflected-sky virtual MV {mv_virt} px, want ~0 \
                 (surface MV {mv_surf} px) — a missed reflection is at INFINITY and must \
                 not translate; this is the sun-highlight jump on generated frames"
            ));
        }
        // GATE TEETH. The pre-fix behaviour is still reachable: cam_far =
        // INFINITY makes the miss branch unreachable, leaving the old point
        // form that placed the sky at a finite `far`. At a PRODUCTION-scale
        // far that must BLOW this bound — if it does not, the probe is too
        // gentle to have caught the bug that shipped, and a future regression
        // would pass unnoticed. (At the old test's far = 5000 it legitimately
        // does not fire, which is precisely why that value hid the defect.)
        if far_probe < 1000.0 {
            let (ox, oy) = virtual_prev_px(
                cx, cy, rw as f32, rh as f32, view_z, far_probe, f32::INFINITY, org, fwd, rgt,
                upv, &world_to_strafe_clip,
            )
            .ok_or("strafe: pre-fix virtual point behind camera")?;
            let mv_old = ((ox - cx).powi(2) + (oy - cy).powi(2)).sqrt();
            if mv_old <= 0.05 {
                return Err(format!(
                    "strafe (far={far_probe}) gate is TOOTHLESS: the pre-fix point form \
                     also lands at {mv_old} px, so this probe could never have caught the \
                     sky-reflection warp"
                ));
            }
        }
    }

    // (e) blend-weight anchors: metal (diffuse ~0) ⇒ ~1; white dielectric ⇒
    // ~F0 fraction; past ROUGH_HI ⇒ exactly 0.
    let w_metal = rmv_weight(0.0, 0.55, 0.2);
    if w_metal < 0.95 {
        return Err(format!("rmv_weight(metal) = {w_metal}, want ~1"));
    }
    let w_diel = rmv_weight(0.8, 0.04, 0.2);
    if !(0.01..=0.1).contains(&w_diel) {
        return Err(format!("rmv_weight(white dielectric) = {w_diel}, want ~0.05"));
    }
    if rmv_weight(0.0, 0.55, ROUGH_HI + 0.01) != 0.0 {
        return Err("rmv_weight past ROUGH_HI must be exactly 0".into());
    }
    eprintln!("ngxfg-guides self-test: OK");
    Ok(())
}
