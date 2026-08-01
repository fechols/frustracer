//! DLSS Ray Reconstruction via RAW NGX (shim/dlssd_shim.cpp) — the
//! Streamline retirement's RR half. One session per GpuContext (the
//! availability probe + the NGX-owned capability map), one feature per
//! render/target dim pair; the evaluate records into the caller's open list
//! exactly where the retired `rr_sl_sequence` did, between the same output
//! PSR->UAV->PSR barriers, over the same `rr::RrResources` planes.
//!
//! CONVENTION LEVERS (the FR_ABL read-only-probe idiom: loud on departure,
//! loud + default on an unrecognized value). The two priors pointed in
//! OPPOSITE directions — SL wanted NEGATED jitter + mvec_scale {1/rw,1/rh},
//! raw NGX frame generation wants RAW jitter + {1,1} pixels — and trap 9's
//! rule (each feature keys its OWN polarity, never reason one from the
//! other) cut BOTH ways here, measured 2026-08-01 on the 4090:
//!   JITTER: raw DLSSD wants the NEGATED offset, like SL and UNLIKE raw-NGX
//!   FG. FRUSTRACER_STAB=1 static view: negated 0.12-0.16/255 (the healthy
//!   RR band), raw 0.26-0.50 (the wrong-polarity wobble). NEGATED IS THE
//!   DEFAULT; `raw` restores the wrong arm for the A/B.
//!   MV: {1,1} — the DLSSD eval's InMVScale converts stored MVs TO PIXEL
//!   SPACE per its own header contract and ours already are pixels (a wrong
//!   arm is a directional smear under strafe, invisible parked; walk
//!   FR_NGXRR_MV if one appears).
//!   FR_NGXRR_JITTER=raw|0  (default: negated)
//!   FR_NGXRR_MV=norm|neg|normneg  (default: {1,1} pixels)
//!   FR_NGXRR_DEPTH=hw  (default: linear view-Z, Depth_Type_Linear — the
//!                       plane rr.rs has always carried)
//!   FR_NGXRR_EXPO=auto (default: no AutoExposure — the SL path set no
//!                       exposure, and the helper defaults pre-exposure 1.0)

use std::ffi::c_void;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;

pub const ERR_UNSUPPORTED: i32 = -3;

/// NVSDK_NGX_DLSS_Feature_Flags — the create-time flags the shim forwards.
/// IsHDR: linear scene-referred RGBA16F color (the retired SL options forced
/// colorBuffersHDR true). MVLowRes: MVs at render res, not target res — the
/// one contract SL expressed implicitly via its resource-tag extents.
pub const FLAG_IS_HDR: i32 = 1 << 0;
pub const FLAG_MV_LOW_RES: i32 = 1 << 1;
pub const FLAG_AUTO_EXPOSURE: i32 = 1 << 6;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FrDlssdOptimal {
    pub opt_w: u32,
    pub opt_h: u32,
    pub min_w: u32,
    pub min_h: u32,
    pub max_w: u32,
    pub max_h: u32,
}

#[repr(C)]
pub struct FrDlssdDispatch {
    pub cmdlist: *mut c_void,
    pub color: *mut c_void,
    pub output: *mut c_void,
    pub depth: *mut c_void,
    pub motion: *mut c_void,
    pub diff_albedo: *mut c_void,
    pub spec_albedo: *mut c_void,
    pub normal_rough: *mut c_void,
    pub spec_hit: *mut c_void,
    pub world_to_view: [f32; 16],
    pub view_to_clip: [f32; 16],
    pub jitter: [f32; 2],
    pub mv_scale: [f32; 2],
    pub rend_w: u32,
    pub rend_h: u32,
    pub reset: i32,
    pub frame_time_ms: f32,
}
// The dlssd_shim.cpp twin asserts the identical literals (the FrDlssgDispatch
// discipline: pin the padding-hole shapes on both sides).
const _: () = assert!(std::mem::offset_of!(FrDlssdDispatch, world_to_view) == 72);
const _: () = assert!(std::mem::size_of::<FrDlssdDispatch>() == 232);

#[cfg(dlss_ngx)]
pub const BUILT: bool = true;
#[cfg(dlss_ngx)]
unsafe extern "C" {
    pub fn frdlssd_open(device: *mut c_void, out_session: *mut *mut c_void) -> i32;
    pub fn frdlssd_optimal(
        session: *mut c_void,
        target_w: u32,
        target_h: u32,
        out: *mut FrDlssdOptimal,
    ) -> i32;
    pub fn frdlssd_create(
        session: *mut c_void,
        rend_w: u32,
        rend_h: u32,
        target_w: u32,
        target_h: u32,
        dlaa: i32,
        depth_hw: u32,
        flags: i32,
        out_feature: *mut *mut c_void,
    ) -> i32;
    pub fn frdlssd_recreate(
        session: *mut c_void,
        feature: *mut *mut c_void,
        rend_w: u32,
        rend_h: u32,
        target_w: u32,
        target_h: u32,
        dlaa: i32,
        depth_hw: u32,
        flags: i32,
    ) -> i32;
    pub fn frdlssd_evaluate(
        session: *mut c_void,
        feature: *mut c_void,
        d: *const FrDlssdDispatch,
    ) -> i32;
    pub fn frdlssd_release_feature(feature: *mut c_void);
    pub fn frdlssd_close(session: *mut c_void);
}

#[cfg(not(dlss_ngx))]
pub const BUILT: bool = false;
#[cfg(not(dlss_ngx))]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn frdlssd_open(_d: *mut c_void, _o: *mut *mut c_void) -> i32 {
    ERR_UNSUPPORTED
}
#[cfg(not(dlss_ngx))]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn frdlssd_optimal(
    _s: *mut c_void,
    _w: u32,
    _h: u32,
    _o: *mut FrDlssdOptimal,
) -> i32 {
    ERR_UNSUPPORTED
}
#[cfg(not(dlss_ngx))]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn frdlssd_create(
    _s: *mut c_void,
    _rw: u32,
    _rh: u32,
    _tw: u32,
    _th: u32,
    _dlaa: i32,
    _hw: u32,
    _fl: i32,
    _o: *mut *mut c_void,
) -> i32 {
    ERR_UNSUPPORTED
}
#[cfg(not(dlss_ngx))]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn frdlssd_recreate(
    _s: *mut c_void,
    _f: *mut *mut c_void,
    _rw: u32,
    _rh: u32,
    _tw: u32,
    _th: u32,
    _dlaa: i32,
    _hw: u32,
    _fl: i32,
) -> i32 {
    ERR_UNSUPPORTED
}
#[cfg(not(dlss_ngx))]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn frdlssd_evaluate(
    _s: *mut c_void,
    _f: *mut c_void,
    _d: *const FrDlssdDispatch,
) -> i32 {
    ERR_UNSUPPORTED
}
#[cfg(not(dlss_ngx))]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn frdlssd_release_feature(_f: *mut c_void) {}
#[cfg(not(dlss_ngx))]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn frdlssd_close(_s: *mut c_void) {}

/// One raw-NGX RR session: the refcounted shared init + the capability map +
/// the availability probe, plus the session's lever-resolved conventions.
/// Holds a device clone so drop order against GpuContext's other fields is
/// free (the shim keeps its own raw device pointer alive through it).
pub struct NgxRr {
    session: *mut c_void,
    _device: ID3D12Device,
    /// Multiplier applied to the frame's sample offset before the evaluate
    /// (-1.0 = negated, the MEASURED default — see the module doc; 1.0 = the
    /// raw A/B arm; 0.0 = the null arm).
    pub jitter_mul: f32,
    /// FR_NGXRR_MV arm: 0 = {1,1} pixels (default), 1 = {1/rw,1/rh} (the SL
    /// scale), 2 = {-1,-1}, 3 = {-1/rw,-1/rh}.
    mv_mode: u8,
    /// Create-time NVSDK_NGX_DLSS_Depth_Type (false = linear view-Z).
    pub depth_hw: bool,
    /// Create-time feature flags (IsHDR | MVLowRes, +AutoExposure on lever).
    pub flags: i32,
}

impl NgxRr {
    /// The chain's DLSS level-1 probe: shared NGX init + capability map +
    /// SuperSamplingDenoising.Available. Err = fall through the chain (the
    /// probe's reason is already on stderr from the shim).
    pub fn open(device: &ID3D12Device) -> Result<Self, String> {
        if !BUILT {
            return Err(
                "built without the DLSS SDK — set FRUSTRACER_DLSS_SDK and rebuild".into(),
            );
        }
        let mut session: *mut c_void = std::ptr::null_mut();
        let r = unsafe { frdlssd_open(device.as_raw(), &mut session) };
        if r == ERR_UNSUPPORTED {
            // The shim's fall-through code: pre-RTX hardware / old driver
            // (details already on stderr) — the ordinary chain fall-through,
            // distinct from a real init error.
            return Err("ray reconstruction not supported on this adapter/driver".into());
        }
        if r != 0 || session.is_null() {
            return Err(format!("raw-NGX RR init failed (shim code {r})"));
        }

        // The FR_ABL lever idiom: loud on departure, loud + default on an
        // unrecognized value (a silent no-op A/B walk is the failure mode
        // the levers exist to prevent).
        let lever = |k: &str, legal: &[&str]| -> u8 {
            let Ok(s) = std::env::var(k) else { return 0 };
            let s = s.to_ascii_lowercase();
            match legal.iter().position(|v| *v == s) {
                Some(i) => i as u8 + 1,
                None => {
                    eprintln!(
                        "dlss-rr: {k}={s} unrecognized (legal: {}) — using the default",
                        legal.join("|")
                    );
                    0
                }
            }
        };
        let jitter_mul = match lever("FR_NGXRR_JITTER", &["raw", "0"]) {
            1 => {
                eprintln!(
                    "dlss-rr: FR_NGXRR_JITTER=raw — the un-negated sample offset (the \
                     measured-wrong polarity: a static view wobbles, STAB ~0.33/255 \
                     vs the negated default's ~0.13)"
                );
                1.0
            }
            2 => {
                eprintln!("dlss-rr: FR_NGXRR_JITTER=0 — zero jitter to the evaluate");
                0.0
            }
            _ => -1.0,
        };
        let mv_mode = lever("FR_NGXRR_MV", &["norm", "neg", "normneg"]);
        if mv_mode != 0 {
            eprintln!(
                "dlss-rr: FR_NGXRR_MV arm {} — walking the MV scale/polarity (wrong \
                 arm = directional smear under strafe)",
                ["norm", "neg", "normneg"][mv_mode as usize - 1]
            );
        }
        let depth_hw = lever("FR_NGXRR_DEPTH", &["hw"]) == 1;
        if depth_hw {
            eprintln!(
                "dlss-rr: FR_NGXRR_DEPTH=hw — Depth_Type_HW at create (default is the \
                 linear view-Z plane rr.rs carries)"
            );
        }
        let mut flags = FLAG_IS_HDR | FLAG_MV_LOW_RES;
        if lever("FR_NGXRR_EXPO", &["auto"]) == 1 {
            eprintln!("dlss-rr: FR_NGXRR_EXPO=auto — AutoExposure create flag armed");
            flags |= FLAG_AUTO_EXPOSURE;
        }

        Ok(Self { session, _device: device.clone(), jitter_mul, mv_mode, depth_hw, flags })
    }

    /// The DLSSD optimal-settings triple for a target resolution:
    /// ((opt_w, opt_h), (min_w, min_h), (max_w, max_h)). The caller owns the
    /// degenerate-collapse-to-DLAA logic (the retired query_rr_res shape).
    pub fn optimal(&self, w: u32, h: u32) -> Result<((u32, u32), (u32, u32), (u32, u32)), String> {
        let mut o = FrDlssdOptimal::default();
        let r = unsafe { frdlssd_optimal(self.session, w, h, &mut o) };
        if r != 0 {
            return Err(format!("DLSSD optimal-settings query failed (shim code {r})"));
        }
        Ok(((o.opt_w, o.opt_h), (o.min_w, o.min_h), (o.max_w, o.max_h)))
    }

    /// CreateFeature at the ALLOCATION dims (the DRS range max) for a target
    /// resolution; per-frame subrect dims name the real render res, so DRS
    /// steps never recreate.
    pub fn create_feature(
        &self,
        rend_max: (u32, u32),
        out: (u32, u32),
        dlaa: bool,
    ) -> Result<RrFeature, String> {
        let mut f: *mut c_void = std::ptr::null_mut();
        let r = unsafe {
            frdlssd_create(
                self.session,
                rend_max.0,
                rend_max.1,
                out.0,
                out.1,
                dlaa as i32,
                self.depth_hw as u32,
                self.flags,
                &mut f,
            )
        };
        if r != 0 || f.is_null() {
            return Err(format!(
                "DLSSD CreateFeature failed at rend {}x{} target {}x{} (shim code {r})",
                rend_max.0, rend_max.1, out.0, out.1
            ));
        }
        Ok(RrFeature { feature: f, rend: rend_max, out, dlaa })
    }

    /// The evaluate's InMVScale for this frame's render res, per the lever.
    pub fn mv_scale(&self, rw: u32, rh: u32) -> [f32; 2] {
        match self.mv_mode {
            1 => [1.0 / rw as f32, 1.0 / rh as f32],
            2 => [-1.0, -1.0],
            3 => [-1.0 / rw as f32, -1.0 / rh as f32],
            _ => [1.0, 1.0],
        }
    }
}

impl Drop for NgxRr {
    fn drop(&mut self) {
        // Refcounted: Shutdown1 only when the last raw-NGX consumer (this or
        // the FG shim) releases its ref.
        unsafe { frdlssd_close(self.session) };
    }
}

/// The created RR feature. Destroy EXPLICITLY with the queue drained
/// (`destroy`, the fg_n pattern); Drop is the backstop for the paths that
/// already idle the device on the way out.
pub struct RrFeature {
    feature: *mut c_void,
    pub rend: (u32, u32),
    pub out: (u32, u32),
    pub dlaa: bool,
}

impl RrFeature {
    /// ReleaseFeature + CreateFeature at new dims (session untouched; caller
    /// drains the queue first — the frdlssg_recreate contract). On failure
    /// the feature is GONE and this returns Err — the caller sheds RR loud.
    pub fn recreate(
        &mut self,
        s: &NgxRr,
        rend_max: (u32, u32),
        out: (u32, u32),
        dlaa: bool,
    ) -> Result<(), String> {
        let r = unsafe {
            frdlssd_recreate(
                s.session,
                &mut self.feature,
                rend_max.0,
                rend_max.1,
                out.0,
                out.1,
                dlaa as i32,
                s.depth_hw as u32,
                s.flags,
            )
        };
        if r != 0 {
            // The shim released the old feature and deleted the handle on a
            // failed re-create; poison so Drop can't double-free.
            self.feature = std::ptr::null_mut();
            return Err(format!(
                "DLSSD recreate failed at rend {}x{} target {}x{} (shim code {r})",
                rend_max.0, rend_max.1, out.0, out.1
            ));
        }
        self.rend = rend_max;
        self.out = out;
        self.dlaa = dlaa;
        Ok(())
    }

    /// One RR evaluate into the caller's open list (inputs resting
    /// NON_PIXEL_SHADER_RESOURCE, output in UNORDERED_ACCESS — the caller's
    /// existing barriers).
    pub fn evaluate(&self, s: &NgxRr, d: &FrDlssdDispatch) -> Result<(), String> {
        if self.feature.is_null() {
            return Err("DLSSD feature is dead (a failed recreate)".into());
        }
        let r = unsafe { frdlssd_evaluate(s.session, self.feature, d) };
        if r != 0 {
            return Err(format!("DLSSD evaluate failed (shim code {r})"));
        }
        Ok(())
    }

    /// Explicit release with the queue drained by the caller.
    pub fn destroy(mut self) {
        if !self.feature.is_null() {
            unsafe { frdlssd_release_feature(self.feature) };
            self.feature = std::ptr::null_mut();
        }
    }
}

impl Drop for RrFeature {
    fn drop(&mut self) {
        if !self.feature.is_null() {
            unsafe { frdlssd_release_feature(self.feature) };
        }
    }
}
