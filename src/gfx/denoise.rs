//! The backend-neutral half of the NRD host: the plane vocabulary, the
//! per-frame `CommonSettings`, and the session's `ReblurSettings`.
//!
//! `src/nrd.rs` is the library binding (portable since B4b-i); `gpu/nrd_gpu.rs`
//! and `vk/nrd.rs` are the two RECORDERS. What sits between them is a small
//! body of settled doctrine that names no API type at all — matrix convention,
//! MV scale, the denoising-range lockstep with `cs_nrd_out`, the `dt_ms` clamp
//! band, the jitter-polarity question — and copying it per backend would be
//! copying the ARGUMENTS, not just the code. The transpose is the sharpest
//! example: it is a one-line difference that no gate can see directly (a wrong
//! one denoises a slightly wrong history), and its justification is fifty lines
//! long, so a second copy is a second place to get it wrong quietly.
//!
//! WHAT DELIBERATELY STAYS PER-BACKEND: anything that names a format, a
//! descriptor, or a barrier. `Plane::nrd_format` returns NRD's OWN `Format`
//! enum value — a `u32` from `NRDDescs.h`, which is as neutral as the resource
//! types beside it — and each backend maps that through the table it already
//! has (`dxgi_format` / `vk_format`). So the fact "IN_VIEWZ is single-channel
//! f32" is stated once, in NRD's vocabulary, and neither backend's table grows
//! a second opinion about it.

/// NRD `Format` values (NRDDescs.h enum order — the same ordinals
/// `dxgi_format`/`vk_format` switch on, and the reason those two tables can be
/// checked against each other by eye).
pub const FMT_R8G8B8A8_UNORM: u32 = 8;
pub const FMT_R32_FLOAT: u32 = 30;
pub const FMT_R16G16B16A16_FLOAT: u32 = 27;
pub const FMT_R10G10B10A2_UNORM: u32 = 40;

/// The app-owned planes an NRD instance reads and writes, in the order both
/// recorders index them.
///
/// This exists because it was NOT written down: `vk/tracer.rs`'s bridge comment
/// cited `P_*` constants in `nrd_gpu.rs` that never existed — those arrays are
/// anonymous, so B4a had to re-invent the names and B4b-ii would have been the
/// third naming of one thing. The ordering is load-bearing in both recorders
/// (it indexes `plane_idx` / `NRD_REGS`), so it is a type rather than a
/// convention.
///
/// `OutValidation` is LAST and is the only optional member: NRD writes it only
/// when `CommonSettings::enable_validation` is set, but both recorders allocate
/// it unconditionally — it is one RGBA8 screen, and making it conditional would
/// mean a reallocation to turn debugging on.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Plane {
    InMv,
    InNormalRoughness,
    InViewZ,
    InDiffRadianceHitDist,
    InSpecRadianceHitDist,
    OutDiffRadianceHitDist,
    OutSpecRadianceHitDist,
    OutValidation,
}

impl Plane {
    pub const COUNT: usize = 8;

    /// Every plane, in index order. `ALL[p.index()] == p` — pinned by
    /// `self_test`, since the two recorders build their plane arrays by
    /// iterating this and index them by `index()`.
    pub const ALL: [Plane; Self::COUNT] = [
        Plane::InMv,
        Plane::InNormalRoughness,
        Plane::InViewZ,
        Plane::InDiffRadianceHitDist,
        Plane::InSpecRadianceHitDist,
        Plane::OutDiffRadianceHitDist,
        Plane::OutSpecRadianceHitDist,
        Plane::OutValidation,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    /// The `ResourceType` NRD names this plane by in a `DispatchDesc`.
    pub fn resource_type(self) -> u32 {
        use crate::nrd::*;
        match self {
            Plane::InMv => RES_IN_MV,
            Plane::InNormalRoughness => RES_IN_NORMAL_ROUGHNESS,
            Plane::InViewZ => RES_IN_VIEWZ,
            Plane::InDiffRadianceHitDist => RES_IN_DIFF_RADIANCE_HITDIST,
            Plane::InSpecRadianceHitDist => RES_IN_SPEC_RADIANCE_HITDIST,
            Plane::OutDiffRadianceHitDist => RES_OUT_DIFF_RADIANCE_HITDIST,
            Plane::OutSpecRadianceHitDist => RES_OUT_SPEC_RADIANCE_HITDIST,
            Plane::OutValidation => RES_OUT_VALIDATION,
        }
    }

    /// The inverse — `None` for a pool resource or for a signal this
    /// REBLUR_DIFFUSE_SPECULAR wiring does not carry. Both recorders route
    /// their `reg_for` through it, so an unmapped type is one error message.
    pub fn from_resource_type(t: u32) -> Option<Plane> {
        Plane::ALL.into_iter().find(|p| p.resource_type() == t)
    }

    /// The wire format, in NRD's own `Format` vocabulary.
    ///
    /// These are the formats `nrd_bridge.hlsl` packs into and recomposes from,
    /// so they are a CONTRACT with the shader and not a recorder's choice — the
    /// bridge is engine-blind (it neither knows nor cares which engine sits
    /// between pack and out), which is exactly why a plane whose format drifted
    /// on one backend would be a plane a real NRD instance could not be handed.
    pub fn nrd_format(self) -> u32 {
        match self {
            // NRD enc-2 packs the normal's L1-octahedral pair plus roughness
            // into 10:10:10:2 — this one is not RGBA16F and never was.
            Plane::InNormalRoughness => FMT_R10G10B10A2_UNORM,
            Plane::InViewZ => FMT_R32_FLOAT,
            Plane::OutValidation => FMT_R8G8B8A8_UNORM,
            _ => FMT_R16G16B16A16_FLOAT,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Plane::InMv => "in_mv",
            Plane::InNormalRoughness => "in_nr",
            Plane::InViewZ => "in_viewz",
            Plane::InDiffRadianceHitDist => "in_diff",
            Plane::InSpecRadianceHitDist => "in_spec",
            Plane::OutDiffRadianceHitDist => "out_diff",
            Plane::OutSpecRadianceHitDist => "out_spec",
            Plane::OutValidation => "out_validation",
        }
    }
}

/// The `timeDeltaBetweenFrames` (ms) every DETERMINISTIC caller reports — the
/// gates and cinematic capture. Those paths have no meaningful frame clock
/// (a gate frame is a submit-and-wait around readbacks and oracle loops), and
/// letting NRD fall back to its own wall-clock timer would make the settings
/// it derives — the antilag scale, the accumulation-speed curve, the specular
/// tap stride — a function of machine load. A fixed 60 Hz nominal makes two
/// runs of the same gate hand NRD identical settings.
pub const NOMINAL_DT_MS: f32 = 1000.0 / 60.0;

/// The frame's CommonSettings from the tree's own camera facts.
/// `dlss::CamMatrices` is glam — COLUMN-major with COLUMN vectors, which is
/// NRD's stated convention (NRDSettings.h), so `to_cols_array()` IS the wire
/// format: no transpose anywhere (the deliberate contrast with
/// `gpu::row_major`'s SL boundary — verify with FR_NRD_DEBUG's validation
/// overlay if reprojection ever looks dead). MV scale: the pack stores
/// pixel-space prev-cur (+ the 2.5D view-Z delta), NRD wants
/// `pixelUvPrev = pixelUv + mv.xy` in UV units — {1/rw, 1/rh, 1}.
/// denoisingRange: sky stores view_z == far, so anything at/past far*0.999
/// is out of range and NRD leaves those texels alone. cs_nrd_out's
/// pass-accum-through predicate is `view_z >= 0.999 * CAM_FAR` — the SAME
/// bound, LOCKSTEP (a shader predicate of plain CAM_FAR shipped once and
/// recomposed the [0.999*far, far) hit band from OUT texels NRD never wrote).
///
/// `dt_ms` is `timeDeltaBetweenFrames`, and it is passed EXPLICITLY rather
/// than left at the header's "0 = tracked internally" default because NRD's
/// internal timer measures WALL CLOCK BETWEEN `SetCommonSettings` CALLS
/// (InstanceImpl.cpp's `m_Timer`), which for a headless gate is the gate's own
/// CPU work — readbacks, oracle loops, PNG writes — not a frame. It is not
/// cosmetic: `m_FrameRateScale = max(33.333/dt, 1)` reaches ReBLUR's antilag
/// scale, its accumulation-speed curve, and the specular virtual-motion tap
/// stride, so an internally-timed gate is a gate whose denoiser settings drift
/// with machine load. Every caller therefore hands over a value it controls:
/// the real per-frame wall time interactively, a fixed nominal in the gates
/// and in cinematic capture (which is deterministic by contract).
///
/// `cameraJitter`, by contrast, is deliberately NOT sign-corrected the way
/// every sibling SDK site is (`xess::JITTER_SIGN`, `fsr::JITTER_SIGN`), and
/// that asymmetry is settled rather than assumed: in the v4.17.3 source the
/// value reaches exactly two places — `REBLUR_Validation.cs.hlsl`'s overlay UV,
/// and `m_JitterDelta = max(|dx|, |dy|)` over the cur/prev pair, which feeds
/// only the CHECKERBOARD resolve speed. Both are sign-symmetric or
/// debug-only, and we never run checkerboard mode, so polarity is
/// STRUCTURALLY inert here. What is NOT inert is the range: NRD asserts
/// [-0.5, 0.5], which is exactly `FrameCtx::frame_jitter`'s own interval, so
/// the raw offset is passed through.
///
/// `enable_validation` is a PARAMETER rather than an environment read, and
/// that is the one thing this function changed on the way here: a settings
/// builder that reads `FR_NRD_DEBUG` itself is a deterministic gate with a
/// hidden input, and the two recorders disagree about what the flag should do
/// anyway (D3D12 dumps the overlay plane, Vulkan does not yet), so the arming
/// belongs to whoever can honour it.
#[allow(clippy::too_many_arguments)]
pub fn common_settings(
    mats: &crate::dlss::CamMatrices,
    prev_mats: &crate::dlss::CamMatrices,
    jitter: (f32, f32),
    prev_jitter: (f32, f32),
    rw: u32,
    rh: u32,
    prev_rw: u32,
    prev_rh: u32,
    far: f32,
    dt_ms: f32,
    frame_index: u32,
    reset: bool,
    enable_validation: bool,
) -> crate::nrd::CommonSettings {
    let mut cs = crate::nrd::CommonSettings::default();
    cs.view_to_clip_matrix = mats.view_to_clip.to_cols_array();
    cs.view_to_clip_matrix_prev = prev_mats.view_to_clip.to_cols_array();
    cs.world_to_view_matrix = mats.world_to_view.to_cols_array();
    cs.world_to_view_matrix_prev = prev_mats.world_to_view.to_cols_array();
    cs.motion_vector_scale = [1.0 / rw as f32, 1.0 / rh as f32, 1.0];
    cs.camera_jitter = [jitter.0, jitter.1];
    cs.camera_jitter_prev = [prev_jitter.0, prev_jitter.1];
    cs.resource_size = [rw as u16, rh as u16];
    cs.resource_size_prev = [prev_rw as u16, prev_rh as u16];
    cs.rect_size = [rw as u16, rh as u16];
    cs.rect_size_prev = [prev_rw as u16, prev_rh as u16];
    cs.denoising_range = far * 0.999;
    // (ms) — clamped into a sane band rather than trusted: a hitch, a
    // debugger break, or a first frame can hand us 0 or seconds, and 0 is the
    // sentinel that silently re-enables the internal timer this exists to
    // replace. 1 ms .. 200 ms spans 1000 fps down to 5.
    cs.time_delta_between_frames = if dt_ms.is_finite() { dt_ms.clamp(1.0, 200.0) } else { 16.667 };
    cs.frame_index = frame_index;
    cs.accumulation_mode =
        if reset { crate::nrd::ACCUM_RESTART } else { crate::nrd::ACCUM_CONTINUE };
    if let Some(x) = crate::nrd::common_tuning().split_screen {
        cs.split_screen = x;
    }
    if enable_validation {
        cs.enable_validation = 1;
    }
    cs
}

/// The session's ReblurSettings: defaults + the ONE departure the 1-spp path
/// demands (the hit-dist params stay at their {3, 0.1, 20} defaults, which is
/// what keeps them in LOCKSTEP with nrd_bridge.hlsl's literals), then the
/// `--nrd-*` tuning overrides (all-None flagless — bit-identical settings).
pub fn reblur_settings() -> crate::nrd::ReblurSettings {
    let mut rs = crate::nrd::ReblurSettings::default();
    // Pixels whose reflection gate never fired carry hit-dist 0 ("no data")
    // — reconstruction fills them from neighbors (required below ~1 rpp).
    rs.hit_distance_reconstruction_mode = crate::nrd::HITDIST_RECONSTRUCTION_AREA_3X3;
    crate::nrd::tuning().apply(&mut rs);
    rs
}

/// Pure gates for the vocabulary above — run by `--check` on every platform,
/// which is what makes them cover the Windows recorder from a Linux box.
pub fn self_test() -> Result<(), String> {
    // ALL is in index order, and both recorders depend on it: they build their
    // plane arrays by iterating ALL and then index by `index()`, so a reorder
    // of one and not the other is a silent plane swap.
    for (i, p) in Plane::ALL.iter().enumerate() {
        if p.index() != i {
            return Err(format!("Plane::ALL[{i}] is {p:?} whose index() is {}", p.index()));
        }
    }
    // The resource-type map is INJECTIVE and round-trips. A duplicate would
    // make `from_resource_type` return the wrong plane for one of them, which
    // aliases two images in `reg_for` — a wrong picture, never an error.
    for p in Plane::ALL {
        match Plane::from_resource_type(p.resource_type()) {
            Some(q) if q == p => {}
            other => {
                return Err(format!(
                    "{p:?} -> type {} -> {other:?} (expected {p:?})",
                    p.resource_type()
                ))
            }
        }
    }
    // Names are UNIQUE and non-empty. They label the per-plane readbacks the
    // gates report, so a duplicate would print one plane's count under
    // another's name — a wrong DIAGNOSIS, which costs more than no diagnosis.
    for (i, p) in Plane::ALL.iter().enumerate() {
        if p.name().is_empty() {
            return Err(format!("{p:?} has an empty name"));
        }
        for q in &Plane::ALL[i + 1..] {
            if p.name() == q.name() {
                return Err(format!("{p:?} and {q:?} share the name {:?}", p.name()));
            }
        }
    }
    // Pool types must NOT resolve to a plane — `reg_for` tests them first, and
    // a collision here would route a pool texture to an app plane.
    for t in [crate::nrd::RES_PERMANENT_POOL, crate::nrd::RES_TRANSIENT_POOL] {
        if let Some(p) = Plane::from_resource_type(t) {
            return Err(format!("pool ResourceType {t} resolves to {p:?}"));
        }
    }
    // The formats are the bridge's contract. Spelled out rather than derived,
    // so a change to `nrd_format`'s match has to be made twice on purpose.
    let want = [
        (Plane::InMv, FMT_R16G16B16A16_FLOAT),
        (Plane::InNormalRoughness, FMT_R10G10B10A2_UNORM),
        (Plane::InViewZ, FMT_R32_FLOAT),
        (Plane::InDiffRadianceHitDist, FMT_R16G16B16A16_FLOAT),
        (Plane::InSpecRadianceHitDist, FMT_R16G16B16A16_FLOAT),
        (Plane::OutDiffRadianceHitDist, FMT_R16G16B16A16_FLOAT),
        (Plane::OutSpecRadianceHitDist, FMT_R16G16B16A16_FLOAT),
        (Plane::OutValidation, FMT_R8G8B8A8_UNORM),
    ];
    for (p, f) in want {
        if p.nrd_format() != f {
            return Err(format!("{p:?} format {} != {f}", p.nrd_format()));
        }
    }
    // The dt clamp: 0 is NRD's "track it internally" sentinel and must never
    // survive, and a non-finite input must not reach the library at all.
    let m = crate::dlss::CamMatrices {
        world_to_view: glam::Mat4::IDENTITY,
        view_to_clip: glam::Mat4::IDENTITY,
    };
    for (dt, want) in [(0.0f32, 1.0f32), (1e9, 200.0), (16.0, 16.0), (f32::NAN, 16.667)] {
        let cs = common_settings(&m, &m, (0.0, 0.0), (0.0, 0.0), 8, 8, 8, 8, 100.0, dt, 0, false, false);
        if (cs.time_delta_between_frames - want).abs() > 1e-3 {
            return Err(format!(
                "dt {dt} -> {} (expected {want})",
                cs.time_delta_between_frames
            ));
        }
    }
    // denoisingRange is in LOCKSTEP with cs_nrd_out's sky predicate. Stated as
    // the shader's own constant so a change to one fails here.
    let cs = common_settings(&m, &m, (0.0, 0.0), (0.0, 0.0), 4, 2, 4, 2, 1000.0, 16.0, 7, true, false);
    if (cs.denoising_range - 999.0).abs() > 1e-3 {
        return Err(format!("denoising_range {} != 999.0", cs.denoising_range));
    }
    // MV scale is UV-per-pixel: the pack stores pixel-space deltas and NRD
    // wants UV. A square-image test cannot tell {1/rw, 1/rh} from {1/rh, 1/rw},
    // so this one is deliberately 4x2.
    if cs.motion_vector_scale != [0.25, 0.5, 1.0] {
        return Err(format!("motion_vector_scale {:?}", cs.motion_vector_scale));
    }
    if cs.accumulation_mode != crate::nrd::ACCUM_RESTART {
        return Err("reset=true did not select ACCUM_RESTART".into());
    }
    if cs.frame_index != 7 {
        return Err("frame_index did not reach CommonSettings".into());
    }
    // Validation is off unless asked. The recorders allocate the plane either
    // way, so this bit is the only thing that makes NRD write it.
    if cs.enable_validation != 0 {
        return Err("enable_validation set without being asked".into());
    }
    let cv =
        common_settings(&m, &m, (0.0, 0.0), (0.0, 0.0), 4, 2, 4, 2, 1000.0, 16.0, 7, true, true);
    if cv.enable_validation == 0 {
        return Err("enable_validation was asked for and did not arrive".into());
    }
    Ok(())
}
