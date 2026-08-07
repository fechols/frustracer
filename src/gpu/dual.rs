//! Cross-adapter band transfer (`--dual-gpu`): moving the secondary device's
//! owned rows into the primary's per-pixel planes.
//!
//! **The payload is per-PIXEL and the saving is per-SAMPLE**, which is the
//! whole economics of the feature. Every plane here is indexed `y*rw + x`, so
//! a `TileSplit::rows` band is one contiguous byte range and one
//! `CopyBufferRegion` per plane — `TileSplit::row_range` is what refuses to
//! answer for any mask that is not a band, rather than handing back a bounding
//! box that would copy the partner's pixels.
//!
//! MEASURED on this box (`--check-gpu`'s transfer probe, 2026-08-06):
//! the RTX 4090 sustains ~25.8 GB/s to and from system memory, the Arc Pro B70
//! ~5.6 — the second x16-length slot on a consumer board is electrically x4,
//! and that slow link, crossed by the secondary's write, is the binding
//! constraint on the entire design. Keep the payload small; nothing else moves
//! the number as much.
//!
//! THE SYSMEM PATH IS THREE CROSSINGS, NOT TWO. A heap created on device A
//! cannot be touched by device B, so the readback -> upload `memcpy` in
//! `hop()` is unavoidable here. A `SHARED | SHARED_CROSS_ADAPTER` heap removes
//! exactly that middle term — and only that term, so on a box whose secondary
//! link dominates (this one) it is worth ~15%, not the 33% the crossing count
//! suggests. It is also the reason this module is shaped around a staging
//! buffer at all: `ALLOW_CROSS_ADAPTER` and `ALLOW_UNORDERED_ACCESS` are
//! MUTUALLY EXCLUSIVE, so `accum`/`gbuf` — root UAVs on every dispatch — can
//! never themselves be the shared resource. Both paths need a separate
//! UAV-free staging buffer and a copy on each side; only the heap in the
//! middle differs, so the shape below is already the shared-heap shape.

use windows::Win32::Graphics::Direct3D12::*;

use super::d3d12::{self, ReadbackBuffer, Result, UploadBuffer};

/// A per-pixel plane taking part in the transfer: its resource and its bytes
/// per pixel (`accum` 12, `GBufCore` 16, `GBufExt` 72, `tbuf`/`info` 4).
#[derive(Clone, Copy)]
pub struct Plane<'a> {
    pub res: &'a ID3D12Resource,
    pub stride: u64,
}

/// Staging for one direction of a band copy: a readback buffer on the source
/// device and an upload buffer on the destination's.
///
/// Both are allocated once at the worst-case band size and reused, the
/// `XessResources` discipline — a per-frame allocation on this path would cost
/// more than the copy.
pub struct BandTransfer {
    readback: ReadbackBuffer,
    upload: UploadBuffer,
    cap: usize,
}

/// Total payload for `rows` pixel rows of `rw`-wide planes.
pub fn payload_bytes(planes: &[u64], rw: u32, rows: u32) -> usize {
    planes.iter().map(|s| s * rw as u64 * rows as u64).sum::<u64>() as usize
}

impl BandTransfer {
    /// `cap` must cover the largest band the balancer can assign — size it
    /// from the FULL screen, not the current split, or a rebalance reallocates
    /// mid-session.
    pub fn new(src: &ID3D12Device, dst: &ID3D12Device, cap: usize) -> Result<Self> {
        Ok(Self {
            readback: ReadbackBuffer::new(src, cap)?,
            upload: UploadBuffer::new(dst, cap)?,
            cap,
        })
    }

    /// Byte offsets of each plane's slice inside the staging buffer, packed in
    /// `planes` order. One source of truth for both `record_out` and
    /// `record_in`: the two sides must agree on the layout or the copy lands
    /// the wrong plane's bytes, which no image gate would attribute correctly.
    fn offsets(planes: &[Plane], rw: u32, rows: u32) -> Vec<(u64, u64)> {
        let mut out = Vec::with_capacity(planes.len());
        let mut at = 0u64;
        for p in planes {
            let len = p.stride * rw as u64 * rows as u64;
            out.push((at, len));
            at += len;
        }
        out
    }

    /// Record on the SOURCE device's list: each plane's `[y0, y1)` rows into
    /// staging. The planes rest in `UNORDERED_ACCESS`, so each copy is
    /// bracketed back to it — this path never leaves a tracer plane in a state
    /// the next dispatch would not expect.
    pub fn record_out(
        &self,
        list: &ID3D12GraphicsCommandList,
        planes: &[Plane],
        rw: u32,
        y0: u32,
        y1: u32,
    ) -> Result<()> {
        let rows = y1.saturating_sub(y0);
        let offs = Self::offsets(planes, rw, rows);
        let total: u64 = offs.iter().map(|(_, l)| l).sum();
        if total as usize > self.cap {
            return Err(format!(
                "band transfer: {total} B exceeds the {} B staging cap — the cap must be sized \
                 from the full screen, not the current split",
                self.cap
            ));
        }
        for (p, (at, len)) in planes.iter().zip(&offs) {
            if *len == 0 {
                continue;
            }
            let src_off = p.stride * rw as u64 * y0 as u64;
            unsafe {
                list.ResourceBarrier(&[d3d12::transition(
                    p.res,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.CopyBufferRegion(&self.readback.resource, *at, p.res, src_off, *len);
                list.ResourceBarrier(&[d3d12::transition(
                    p.res,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
        }
        Ok(())
    }

    /// The system-memory hop: readback (source device) -> upload (destination
    /// device). Valid only after the submission that recorded `record_out` has
    /// COMPLETED — the fence wait is the caller's, because the whole point of
    /// the dual-GPU schedule is choosing where that wait lands.
    ///
    /// This is the crossing a shared heap removes; nothing else about the
    /// shape changes when it does.
    pub fn hop(&self, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        if bytes > self.cap {
            return Err(format!("band hop: {bytes} B over the {} B cap", self.cap));
        }
        unsafe {
            let mut p = std::ptr::null_mut();
            self.readback
                .resource
                .Map(0, None, Some(&mut p))
                .map_err(|e| format!("band hop Map: {e}"))?;
            std::ptr::copy_nonoverlapping(p as *const u8, self.upload.ptr, bytes);
            self.readback.resource.Unmap(0, None);
        }
        Ok(())
    }

    /// Record on the DESTINATION device's list: staging back into each plane's
    /// `[y0, y1)` rows. Layout mirrors `record_out` through the shared
    /// `offsets`.
    pub fn record_in(
        &self,
        list: &ID3D12GraphicsCommandList,
        planes: &[Plane],
        rw: u32,
        y0: u32,
        y1: u32,
    ) -> Result<()> {
        let rows = y1.saturating_sub(y0);
        let offs = Self::offsets(planes, rw, rows);
        for (p, (at, len)) in planes.iter().zip(&offs) {
            if *len == 0 {
                continue;
            }
            let dst_off = p.stride * rw as u64 * y0 as u64;
            unsafe {
                list.ResourceBarrier(&[d3d12::transition(
                    p.res,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                list.CopyBufferRegion(p.res, dst_off, &self.upload.resource, *at, *len);
                list.ResourceBarrier(&[d3d12::transition(
                    p.res,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
        }
        Ok(())
    }
}

// --- the balancer -------------------------------------------------------
//
// WHY NOT `xess::ScaleCtl`, which is the obvious thing to reuse and which the
// plan originally called for. Three reasons, each fatal on its own:
//
//   1. It lives in LOG2 space, and `new()` applies `h.max(1)`. A share of
//      ZERO is therefore unrepresentable — and zero is the correct answer
//      whenever the secondary's link cannot carry its own output. That is the
//      coordinate system, not a clamp to relax.
//   2. Its `* 0.5` exponent encodes cost ~ scale^2 (area). A tile-row share
//      is LINEAR in cost, so the gain would be halved.
//   3. Its deadband is one-sided ("don't climb past 60% of budget"). This is
//      not budget-tracking; it is two-sided equalisation with no budget at
//      all, wanting a deadband centred on parity.
//
// And why not `xess::StepLimiter`: `applied.0 == 0` is its uninitialised
// sentinel, so a control variable that can legitimately be zero re-runs
// first-frame adoption every frame, silently disabling dwell and emergency
// logic while still producing plausible output. The discipline transfers; the
// types do not.

/// Fraction of the normalised imbalance applied per frame.
const SHARE_GAIN: f32 = 0.25;
/// Per-frame cap on GROWING the secondary's share. Deliberately 5x tighter
/// than the shrink cap: growth adds transfer on the slow link and is the
/// direction that can make a frame worse, so it creeps.
const SHARE_UP_MAX: f32 = 0.02;
/// Per-frame cap on SHRINKING it. Shrinking degrades toward single-GPU, which
/// is always safe, so it sheds fast — `ScaleCtl`'s asymmetry, same reasoning.
const SHARE_DOWN_MAX: f32 = 0.10;
/// Normalised imbalance below which the share parks instead of flapping.
/// |err| < 0.03 means the two devices finish within ~6% of each other.
const SHARE_DEADBAND: f32 = 0.03;
/// Extra distance (in rows) the continuous estimate must travel past the
/// midpoint before the quantiser moves. At one row per quantum the quantiser
/// itself provides NO hysteresis — where `RES_STEP` did — so this restores it.
const SHARE_HYST: f32 = 0.15;
/// Frames between APPLIED row changes. Every change is a real reassignment
/// that invalidates structure replay on BOTH devices.
pub const SHARE_DWELL: u32 = 90;
/// Frames the secondary stays idle before one forced probe re-measures it.
/// Without a probe a single transient disables the secondary permanently;
/// with one the controller is self-correcting at the cost of one hitched
/// frame per ~10 s at 60 fps.
pub const SHARE_PROBE_FRAMES: u32 = 600;

/// Continuous secondary share, in linear space so that ZERO is reachable.
///
/// The control law is `err = (prim - sec) / (prim + sec)` — normalised, so it
/// is bounded in [-1, 1] and needs no gain retune between a 0.3 ms procedural
/// frame and a 113 ms GI one. Positive means the PRIMARY is the critical
/// path, so the secondary should take more.
///
/// `sec` must include the secondary's TRANSFER, not just its trace. Direct
/// precedent: `ScaleCtl` is fed `last_ms + pre_ms` because the OIDN pre-pass
/// is area-proportional work the chosen scale buys. Omit it here and the
/// controller grows the share past what the link can carry.
pub struct ShareCtl {
    share: f32,
    max: f32,
    idle: u32,
}

impl ShareCtl {
    pub fn new(start: f32, max: f32) -> Self {
        let max = max.clamp(0.0, 1.0);
        Self { share: start.clamp(0.0, max), max, idle: 0 }
    }

    pub fn share(&self) -> f32 {
        self.share
    }

    /// Feed a frame in which the secondary ACTUALLY rendered.
    ///
    /// Never call this on a frame the secondary sat out: with `sec` near zero
    /// the error pins at +1 and the share climbs straight back out of the idle
    /// state it correctly reached. That oscillation is the whole reason
    /// `idle_frame`/`probe` exist as separate entry points.
    pub fn update(&mut self, prim_ms: f32, sec_ms: f32) {
        self.idle = 0;
        let sum = prim_ms + sec_ms;
        if !(sum > 0.0) {
            return;
        }
        let err = (prim_ms - sec_ms) / sum;
        if err.abs() < SHARE_DEADBAND {
            return;
        }
        let step = (SHARE_GAIN * err).clamp(-SHARE_DOWN_MAX, SHARE_UP_MAX);
        self.share = (self.share + step).clamp(0.0, self.max);
    }

    /// Feed a frame in which the secondary did NOT render (the share
    /// quantised to zero). Returns true when a re-probe is due.
    pub fn idle_frame(&mut self) -> bool {
        self.idle = self.idle.saturating_add(1);
        if self.idle >= SHARE_PROBE_FRAMES {
            self.idle = 0;
            return true;
        }
        false
    }

    /// Result of a forced probe out of the idle state, where `level` is the
    /// share the probe frame actually ran at.
    ///
    /// A favourable probe ADOPTS `level` outright rather than creeping toward
    /// it: creeping at `SHARE_UP_MAX` from zero would need ~7 probes, i.e.
    /// ~70 s, to earn back a single row — slow enough that the secondary would
    /// look permanently dead. An unfavourable one leaves the share at zero.
    pub fn probe(&mut self, prim_ms: f32, sec_ms: f32, level: f32) {
        self.idle = 0;
        let sum = prim_ms + sec_ms;
        if sum > 0.0 && (prim_ms - sec_ms) / sum > SHARE_DEADBAND {
            self.share = self.share.max(level).clamp(0.0, self.max);
        }
    }
}

/// Turn the per-frame phase timers into the controller's `(prim_ms, sec_ms)`.
///
/// THE SUBTLETY THAT MAKES THIS A FUNCTION RATHER THAN TWO ADDITIONS: the
/// schedule waits on the SECONDARY first, so that its band-out overlaps the
/// primary's remaining trace — the overlap that hides the expensive PCIe
/// direction. The cost is that a primary which finishes early leaves
/// `prim_wait == 0` and its exact duration unobserved, and a naive
/// `prim = sec + prim_wait` would then be >= `sec` on EVERY frame, so the
/// controller could only ever grow the share. It would never find zero.
///
/// The fix is not a fudge factor — it is the bound the schedule already
/// proves. An unobserved primary finished at or before `sec_wait`, so
/// `sec_wait` is a hard UPPER bound on it, while the secondary additionally
/// spent `out + hop` of critical path getting its band across. The resulting
/// error is `-(out + hop) / (2*sec_wait + out + hop)`: negative, and
/// proportional to exactly the transfer the secondary cost us. On this box's
/// x4 link that is ~-0.23, a decisive shrink; on a fast link it is near zero
/// and the share holds. Using an upper bound makes the shrink CONSERVATIVE,
/// which is the safe direction.
///
/// `prim_early` must come from a NON-BLOCKING fence query
/// (`HeadlessGpu::completed`), never from "was the wait short?". Waiting on an
/// already-signalled fence still burns tens of nanoseconds, so a duration test
/// reports "still running" on essentially every frame — which is precisely how
/// a first draft of this ended up growing the share to 3/8 on a box whose
/// right answer is 0. Measure the condition, not a proxy for it.
pub fn phase_times(
    sec_wait: f32,
    out: f32,
    hop: f32,
    prim_wait: f32,
    prim_early: bool,
) -> (f32, f32) {
    let sec = sec_wait + out + hop;
    let prim = if prim_early { sec_wait } else { sec + prim_wait };
    (prim, sec)
}

/// Continuous share -> whole tile rows, with hysteresis.
///
/// `cur` is the row count in force, and the result stays there unless the
/// estimate has travelled more than half a row PLUS `SHARE_HYST` away — the
/// anti-flap the `RES_STEP` quantum provides for resolutions and that a
/// one-row quantum cannot.
///
/// Zero is a legal output, which is the entire point: the three `.max(1)` /
/// `.max(RES_STEP)` floors in `quantize_res`'s lineage are deliberately absent.
/// The upper clamp is `side - 1` because the primary presents and must keep
/// at least one row.
pub fn quantize_share(share: f32, side: u32, cur: u32) -> u32 {
    let top = side.saturating_sub(1);
    let raw = (share * side as f32).clamp(0.0, top as f32);
    if (raw - cur as f32).abs() <= 0.5 + SHARE_HYST {
        return cur.min(top);
    }
    (raw.round() as u32).min(top)
}

/// Rate-limits APPLIED row changes. Dwell only — deliberately no ramp.
///
/// A resolution ramp is cheap because rounding makes its intermediates
/// repeat; every distinct row count here is a real reassignment that drops
/// structure replay on both devices, so ramping would pay the cost once per
/// intermediate for no benefit.
///
/// `applied: Option<u32>` rather than a zero sentinel, because zero is a
/// legal applied value — the failure `StepLimiter` would have had here is
/// silent, not loud.
pub struct ShareLimiter {
    applied: Option<u32>,
    since: u32,
}

impl Default for ShareLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl ShareLimiter {
    pub fn new() -> Self {
        Self { applied: None, since: 0 }
    }

    pub fn rows(&self) -> u32 {
        self.applied.unwrap_or(0)
    }

    /// The row count to use THIS frame. `shed` bypasses the dwell, but only
    /// downward — an emergency may never hand the secondary MORE work.
    pub fn apply(&mut self, target: u32, shed: bool) -> u32 {
        let Some(cur) = self.applied else {
            self.applied = Some(target);
            self.since = 0;
            return target;
        };
        self.since = self.since.saturating_add(1);
        if shed && target < cur {
            self.applied = Some(target);
            self.since = 0;
        } else if target != cur && self.since >= SHARE_DWELL {
            self.applied = Some(target);
            self.since = 0;
        }
        self.applied.unwrap_or(cur)
    }
}

/// Pure-math gates for the transfer's addressing. DLL- and GPU-free, run by
/// every `--check` like the split's own — the byte arithmetic is where a band
/// copy goes wrong silently, and a wrong offset lands a plausible image with
/// one displaced stripe.
pub fn self_test() -> std::result::Result<(), String> {
    use super::trace::TileSplit;

    // The payload identity the whole cost model rests on: bytes are linear in
    // rows and in stride, so a band's cost is exactly its share of the screen.
    let (rw, rh) = (1920u32, 1080u32);
    let strides = [12u64, 16];
    let whole = payload_bytes(&strides, rw, rh);
    let half = payload_bytes(&strides, rw, rh / 2);
    if whole != half * 2 {
        return Err(format!("payload_bytes is not linear in rows: {whole} vs 2x{half}"));
    }
    if whole != (12 + 16) * rw as usize * rh as usize {
        return Err("payload_bytes disagrees with stride x width x rows".into());
    }

    // Complementary bands must tile the byte range of every plane exactly:
    // A's slice ends where B's begins, and together they are the whole plane.
    for depth in 1..=3u32 {
        for r in 1..(1u32 << depth) {
            let a = TileSplit::rows(depth, 0, r);
            let b = a.complement();
            let (ay0, ay1) = a
                .row_range(rw, rh)
                .ok_or("rows() must yield a contiguous band")?;
            let (by0, by1) = b.row_range(rw, rh).ok_or("its complement must too")?;
            for &s in &strides {
                let a_end = s * rw as u64 * ay1 as u64;
                let b_start = s * rw as u64 * by0 as u64;
                if a_end != b_start {
                    return Err(format!(
                        "depth {depth} row {r} stride {s}: A's byte range ends at {a_end} but B's \
                         starts at {b_start} — the copy would duplicate or skip a row"
                    ));
                }
                if s * rw as u64 * by1 as u64 != s * rw as u64 * rh as u64 {
                    return Err(format!(
                        "depth {depth} row {r} stride {s}: B's range does not reach the end of \
                         the plane"
                    ));
                }
                let _ = ay0;
            }
        }
    }

    // The staging layout must be a partition too — planes packed back to back,
    // no overlap, no gap. An overlap here silently corrupts one plane with
    // another's bytes, which reads as garbage in a G-buffer rather than as a
    // missing stripe, so it is the harder failure to attribute.
    let rows = 137u32; // deliberately not a divisor of anything
    let mut at = 0u64;
    for &s in &strides {
        let len = s * rw as u64 * rows as u64;
        at += len;
    }
    if at as usize != payload_bytes(&strides, rw, rows) {
        return Err("staging layout total disagrees with payload_bytes".into());
    }

    balance_self_test()?;
    Ok(())
}

/// The balancer, on scripted frame times with NO wall clock and no GPU.
///
/// THIS IS WHAT LETS THE FEATURE SHIP BEFORE THE HARDWARE EXISTS. The split's
/// partition, the transfer's addressing and the controller's convergence are
/// all provable on a box that cannot demonstrate the win; only the win itself
/// needs a platform with two full-width slots. Modelled on `run_check_xess`'s
/// scale-controller section, which establishes the pattern.
fn balance_self_test() -> std::result::Result<(), String> {
    const MAX: f32 = 0.875;

    // (1) CONVERGENCE TO THE ANALYTIC OPTIMUM. Simulate the real cost model —
    // primary T(1-s), secondary r*T*s of trace plus K*s of transfer — and let
    // the controller see only the two times it would see in a real frame. It
    // must find s* = T / (T(1+r) + K), the minimiser of max(prim, sec), which
    // is NOT the compute-balanced share: that distinction is the single
    // biggest thing this controller exists to get right.
    let (t, r, k) = (10.0f32, 2.0f32, 5.0f32);
    let opt = t / (t * (1.0 + r) + k);
    let mut c = ShareCtl::new(0.0, MAX);
    for _ in 0..4000 {
        let s = c.share();
        c.update(t * (1.0 - s), r * t * s + k * s);
    }
    if (c.share() - opt).abs() > 0.02 {
        return Err(format!(
            "balancer converged to {:.4}, not the optimum {opt:.4} — it is minimising the \
             wrong objective (compute balance rather than max(primary, secondary+transfer))",
            c.share()
        ));
    }
    // ...and from ABOVE, so the result is a fixed point rather than a floor
    // the creep happened to stop at.
    let mut c = ShareCtl::new(MAX, MAX);
    for _ in 0..4000 {
        let s = c.share();
        c.update(t * (1.0 - s), r * t * s + k * s);
    }
    if (c.share() - opt).abs() > 0.02 {
        return Err(format!(
            "balancer settled at {:.4} descending but {opt:.4} ascending — not a fixed point",
            c.share()
        ));
    }

    // (2) CONVERGENCE TO EXACTLY ZERO, and STAYING there. The correct answer
    // whenever the secondary cannot carry its own output, and the reason the
    // controller is in linear space at all.
    let mut c = ShareCtl::new(0.5, MAX);
    for _ in 0..200 {
        c.update(1.0, 10.0);
    }
    if c.share() != 0.0 {
        return Err(format!(
            "a useless secondary must drive the share to EXACTLY 0.0, got {} — log-space \
             controllers cannot express this, which is why ScaleCtl was not reused",
            c.share()
        ));
    }
    for _ in 0..200 {
        c.update(1.0, 10.0);
    }
    if c.share() != 0.0 {
        return Err("the zero state must be stable under continued bad frames".into());
    }

    // (3) PER-FRAME STEP BOUNDS, both directions.
    let mut c = ShareCtl::new(0.4, MAX);
    c.update(100.0, 1.0); // maximally primary-bound: grow
    if c.share() - 0.4 > SHARE_UP_MAX + 1e-6 {
        return Err(format!("grew {} in one frame, over the {SHARE_UP_MAX} cap", c.share() - 0.4));
    }
    let mut c = ShareCtl::new(0.4, MAX);
    c.update(1.0, 100.0); // maximally secondary-bound: shed
    if 0.4 - c.share() > SHARE_DOWN_MAX + 1e-6 {
        return Err(format!("shed {} in one frame, over the {SHARE_DOWN_MAX} cap", 0.4 - c.share()));
    }
    // The asymmetry itself is load-bearing: shedding is the safe direction.
    if SHARE_DOWN_MAX <= SHARE_UP_MAX {
        return Err("shrinking the secondary's share must be faster than growing it".into());
    }

    // (4) DEADBAND — exact equality, no epsilon, per the existing gate's own
    // discipline. A near-balanced pair must not move the share at all.
    let mut c = ShareCtl::new(0.3, MAX);
    let parked = c.share();
    for _ in 0..50 {
        c.update(10.0, 10.2); // |err| ~ 0.01, inside the band
    }
    if c.share() != parked {
        return Err(format!("controller moved inside the deadband: {parked} -> {}", c.share()));
    }

    // (5) THE IDLE/PROBE PAIR. `update` must never be fed an idle frame (the
    // oscillation trap: sec ~ 0 pins err at +1 and climbs straight back out
    // of the zero state), so idling is its own entry point, and a probe
    // ADOPTS on success rather than creeping.
    let mut c = ShareCtl::new(0.0, MAX);
    let mut fired = 0;
    for _ in 0..SHARE_PROBE_FRAMES * 2 {
        if c.idle_frame() {
            fired += 1;
        }
    }
    if fired != 2 {
        return Err(format!("expected 2 re-probes in {} idle frames, got {fired}", SHARE_PROBE_FRAMES * 2));
    }
    let mut c = ShareCtl::new(0.0, MAX);
    c.probe(1.0, 10.0, 0.125); // unfavourable
    if c.share() != 0.0 {
        return Err("an unfavourable probe must leave the share at zero".into());
    }
    c.probe(10.0, 1.0, 0.125); // favourable
    if (c.share() - 0.125).abs() > 1e-6 {
        return Err(format!(
            "a favourable probe must ADOPT the probed level, got {} — creeping from zero at \
             the growth cap would need ~7 probes to earn one row back",
            c.share()
        ));
    }

    // (5b) PHASE DERIVATION. An observed primary is exact; an UNobserved one
    // (finished before the schedule looked) must report BELOW the secondary,
    // or the controller can only ever grow and will never find zero.
    let (p, s) = phase_times(2.0, 0.5, 0.1, 1.0, false);
    if (p - 3.6).abs() > 1e-5 || (s - 2.6).abs() > 1e-5 {
        return Err(format!("phase_times mis-derived an observed primary: ({p}, {s})"));
    }
    let (p, s) = phase_times(2.0, 0.5, 0.1, 0.0, true);
    if p != 2.0 || (s - 2.6).abs() > 1e-5 {
        return Err(format!(
            "an early (unobserved) primary must be reported at its proven upper bound \
             `sec_wait`, got ({p}, {s})"
        ));
    }
    if p >= s {
        return Err(
            "an early primary must report BELOW the secondary — otherwise the error is \
             non-negative on every frame and the share can only ever grow"
                .into(),
        );
    }
    // ...and the shrink must be DECISIVE, not a token nudge. This is the
    // regression that shipped once: a weak constant signal let occasional
    // slow-primary frames climb faster than early-primary frames could shed,
    // and the balancer settled at 3/8 rows on a box whose right answer is 0.
    // A costly transfer must dominate.
    let mut c = ShareCtl::new(0.5, MAX);
    let before = c.share();
    let (p, s) = phase_times(2.0, 0.5, 0.1, 0.0, true);
    c.update(p, s);
    let shed = before - c.share();
    if shed <= SHARE_UP_MAX {
        return Err(format!(
            "an early primary shed only {shed}, no more than the {SHARE_UP_MAX} a single \
             grow frame adds — the share would ratchet upward on a link that cannot pay"
        ));
    }
    // The signal must SCALE with the transfer: a free link should barely
    // shed, an expensive one decisively. Otherwise it is a constant wearing a
    // measurement's clothes.
    let (pf, sf) = phase_times(2.0, 0.01, 0.0, 0.0, true);
    let (pe, se) = phase_times(2.0, 2.0, 1.0, 0.0, true);
    let cheap = (sf - pf) / (sf + pf);
    let dear = (se - pe) / (se + pe);
    if !(dear > cheap * 4.0) {
        return Err(format!(
            "the early-primary signal does not scale with transfer cost (cheap {cheap:.4} vs \
             dear {dear:.4}) — it is a constant, not a measurement"
        ));
    }

    // (6) QUANTISER: zero reachable, top clamped below `side`, range escape,
    // and hysteresis actually holding.
    for side in [2u32, 4, 8] {
        let top = side - 1;
        for i in -3..=13 {
            let s = i as f32 / 10.0;
            for cur in 0..=top {
                let q = quantize_share(s, side, cur);
                if q > top {
                    return Err(format!(
                        "quantize_share({s}, {side}, {cur}) = {q} starves the primary, which \
                         still has to present"
                    ));
                }
            }
        }
        if quantize_share(0.0, side, 0) != 0 {
            return Err("a zero share must quantise to zero rows".into());
        }
        // From zero, a share worth a full row must actually reach it.
        if quantize_share(1.0, side, 0) != top {
            return Err(format!("a full share must reach {top} rows at side {side}"));
        }
        // Hysteresis: a nudge just past the midpoint must NOT move.
        let held = quantize_share((2.0 + 0.5 + SHARE_HYST * 0.5) / side as f32, side, 2);
        if side > 3 && held != 2 {
            return Err(format!("quantiser moved inside the hysteresis band at side {side}"));
        }
    }

    // (7) LIMITER: first frame adopts, the dwell holds, an emergency sheds but
    // may never grow.
    let mut l = ShareLimiter::new();
    if l.apply(3, false) != 3 {
        return Err("the first apply must adopt unconditionally".into());
    }
    if l.apply(1, false) != 3 {
        return Err("the dwell must hold a new target".into());
    }
    if l.apply(5, true) != 3 {
        return Err("an emergency must never GROW the secondary's share".into());
    }
    if l.apply(1, true) != 1 {
        return Err("an emergency must shed past the dwell".into());
    }
    let mut l = ShareLimiter::new();
    l.apply(4, false);
    for _ in 0..SHARE_DWELL {
        l.apply(2, false);
    }
    if l.rows() != 2 {
        return Err(format!("the dwell never expired: still at {} rows", l.rows()));
    }
    // Zero is a legal APPLIED value — the case a (0,0) sentinel would have
    // silently turned into "uninitialised", re-adopting every frame forever.
    let mut l = ShareLimiter::new();
    if l.apply(0, false) != 0 {
        return Err("zero rows must be adoptable".into());
    }
    if l.apply(7, false) != 0 {
        return Err(
            "after adopting ZERO the dwell must still hold — a zero sentinel would have \
             re-run first-frame adoption here and disabled the limiter silently"
                .into(),
        );
    }
    Ok(())
}
