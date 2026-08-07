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
/// device and an upload RING on the destination's.
///
/// Both are allocated once at the worst-case band size and reused, the
/// `XessResources` discipline — a per-frame allocation on this path would cost
/// more than the copy.
///
/// THE UPLOAD SIDE IS A RING AND THE READBACK SIDE IS NOT, and the asymmetry
/// is exactly the fencing. `record_out` runs on the secondary through a
/// BLOCKING `run`/`wait`, so the readback is complete before `hop` reads it and
/// nothing else can be in flight against it. `record_in` is recorded onto the
/// PRIMARY's frame list and executes whenever that frame executes — so on the
/// interactive path frame N's copy is still queued when frame N+1's `hop`
/// memcpys, and a single buffer would be rewritten under an executing copy
/// (`UploadBuffer`'s own safety note assumes the begin_frame fence wait, which
/// at `FRAMES_IN_FLIGHT = 2` only retires frame N-2). The symptom would be a
/// horizontal tear inside the band, between two frames' radiance, appearing
/// only under a particular primary/secondary timing — the class that never
/// reproduces. One sub-buffer per in-flight slot, keyed by the caller's own
/// frame slot, makes it structural: the slot the caller writes was last read
/// by frame N-2, whose retirement `D3d::begin_frame` already waited on.
pub struct BandTransfer {
    readback: ReadbackBuffer,
    upload: UploadBuffer,
    cap: usize,
}

/// Total payload for `rows` pixel rows of `rw`-wide planes.
pub fn payload_bytes(planes: &[u64], rw: u32, rows: u32) -> usize {
    planes.iter().map(|s| s * rw as u64 * rows as u64).sum::<u64>() as usize
}

/// The strides a band must carry for a frame that will be FED to an upscaler,
/// in the order `record_out`/`record_in` pack them.
///
/// `accum` alone is only enough for a frame the CPU reads back — an
/// accumulation capture. Anything that runs `record_feed` reads the secondary's
/// rows of the G-BUFFER PACK too, so the pack crosses with it: `GBufCore`
/// always, `GBufExt` only when a guide-consuming feed (RR, FSR4-RR, NPPD) is
/// wired. That is 28 B/px for a XeSS or FSR 3.1 session and 100 for an RR one,
/// against the capture path's 12 — and unlike the capture, a fed frame pays it
/// EVERY frame, because the engine consumes a whole merged image each time.
/// Sizing a staging cap from the largest of these is what lets the wiring
/// change without reallocating.
/// `pack` is `TraceGpu::pack_full()` and MUST be read from the PRIMARY: a
/// plain session's pack buffers are single-element dummies, so a band copied
/// into them is an out-of-bounds `CopyBufferRegion` — which does not fault,
/// it just makes the whole command list fail to Close.
pub fn fed_strides(pack: bool, ext: bool) -> &'static [u64] {
    match (pack, ext) {
        (false, _) => &[12],
        (true, false) => &[12, 16],
        (true, true) => &[12, 16, 72],
    }
}

/// Every stride any arm can ask for — the staging cap's worst case.
pub const MAX_FED_STRIDES: [u64; 3] = [12, 16, 72];

/// Byte base of frame slot `slot`'s sub-buffer inside a `cap`-sized upload
/// ring. Free-standing so the arithmetic is gated without a device — the
/// module's own "a wrong offset lands a plausible image with one displaced
/// stripe" rule, which is exactly what a ring the two halves disagree about
/// would produce.
pub fn ring_base(cap: usize, slot: usize) -> u64 {
    (slot % d3d12::FRAMES_IN_FLIGHT) as u64 * cap as u64
}

impl BandTransfer {
    /// `cap` must cover the largest band the balancer can assign — size it
    /// from the FULL screen, not the current split, or a rebalance reallocates
    /// mid-session.
    ///
    /// The upload allocation is `FRAMES_IN_FLIGHT * cap`: one sub-buffer per
    /// frame slot, see the type's own note.
    pub fn new(src: &ID3D12Device, dst: &ID3D12Device, cap: usize) -> Result<Self> {
        Ok(Self {
            readback: ReadbackBuffer::new(src, cap)?,
            upload: UploadBuffer::new(dst, cap * d3d12::FRAMES_IN_FLIGHT)?,
            cap,
        })
    }

    /// Byte base of one frame slot's upload sub-buffer.
    fn slot_base(&self, slot: usize) -> u64 {
        ring_base(self.cap, slot)
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
    ///
    /// `slot` is the DESTINATION frame's slot — the same one its `record_in`
    /// passes, and the same one `D3d::begin_frame` fence-waited on. A
    /// submit-and-wait driver (`--spin`, `--cinematic`) has one frame in
    /// flight and passes 0.
    pub fn hop(&self, slot: usize, bytes: usize) -> Result<()> {
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
            let dst = self.upload.ptr.add(self.slot_base(slot) as usize);
            std::ptr::copy_nonoverlapping(p as *const u8, dst, bytes);
            self.readback.resource.Unmap(0, None);
        }
        Ok(())
    }

    /// Record on the DESTINATION device's list: staging back into each plane's
    /// `[y0, y1)` rows. Layout mirrors `record_out` through the shared
    /// `offsets`, offset by the frame slot's own sub-buffer.
    ///
    /// `slot` MUST be the one the matching `hop` wrote — they are the two
    /// halves of one ring entry, and disagreeing would read a stale frame's
    /// band with no error anywhere.
    pub fn record_in(
        &self,
        list: &ID3D12GraphicsCommandList,
        planes: &[Plane],
        rw: u32,
        y0: u32,
        y1: u32,
        slot: usize,
    ) -> Result<()> {
        let rows = y1.saturating_sub(y0);
        let offs = Self::offsets(planes, rw, rows);
        let base = self.slot_base(slot);
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
                list.CopyBufferRegion(p.res, dst_off, &self.upload.resource, base + *at, *len);
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
/// Output frames between SOLO probes: one frame forced to zero share, so the
/// controller has a measured no-split baseline to judge the split against.
/// Cheap by construction — a solo frame is a correct frame, and on a box where
/// splitting does not pay it is the FASTER one.
pub const SOLO_PROBE_FRAMES: u32 = 16;
/// How much worse than solo the split must measure before the controller sheds
/// on that ground alone. Small, because the comparison is between EWMAs of
/// nearby frames; it exists to stop churn, not to grant the split leeway.
const SOLO_MARGIN: f32 = 0.02;
/// EWMA weight for both cost estimates.
const COST_EWMA: f32 = 0.25;

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
    /// Ticks since the last solo probe, the running dual cost, and the LATCHED
    /// verdict of the most recent solo comparison.
    since_solo: u32,
    dual_ms: f32,
    split_loses: bool,
    grow_block: bool,
}

impl ShareCtl {
    pub fn new(start: f32, max: f32) -> Self {
        let max = max.clamp(0.0, 1.0);
        Self {
            share: start.clamp(0.0, max),
            max,
            idle: 0,
            // Due soon, but not on the very first tick: the verdict compares
            // the probe against the dual frames just before it, so it needs a
            // few of those to exist.
            since_solo: SOLO_PROBE_FRAMES.saturating_sub(4),
            dual_ms: 0.0,
            split_loses: false,
            grow_block: false,
        }
    }

    pub fn share(&self) -> f32 {
        self.share
    }

    /// True when a SOLO probe is due — one frame forced to zero share, whose
    /// cost `solo` then records.
    ///
    /// WHY THIS EXISTS, and it is the difference between the balancer being
    /// honest and being a regression a user armed on purpose: equalising
    /// `prim` and `sec` finds the best SPLIT, but says nothing about whether
    /// splitting beats not splitting. That proxy is exact only while the two
    /// devices do not interfere. On this box they do — a 4090 that renders the
    /// whole screen in 21 ms takes 16.6 for HALF of it while the secondary
    /// runs — so the balanced point is 1.7x slower than one GPU, and a
    /// balance-only controller reports "settled at 3 rows" as though that were
    /// a success. MEASURED, and it is why the pre-anti-windup code looked
    /// better than it was: it reached zero only by overshooting.
    ///
    /// A solo frame is a correct frame, so the probe costs nothing but the
    /// split's own benefit for one frame out of `SOLO_PROBE_FRAMES`.
    pub fn solo_frame(&mut self) -> bool {
        self.since_solo = self.since_solo.saturating_add(1);
        if self.since_solo >= SOLO_PROBE_FRAMES {
            self.since_solo = 0;
            return true;
        }
        false
    }

    /// Whether the last solo probe found the split LOSING to no split at all.
    ///
    /// This is the limiter's `shed` condition, and the only thing in the design
    /// that legitimately bypasses the dwell: a dwell exists to stop churn
    /// between two workable shares, not to hold a configuration that has been
    /// measured slower than doing nothing. Without it the walk down costs one
    /// dwell per row — 4 x `SHARE_DWELL` = 360 frames on `--spin`, measured at
    /// 2.67 ms/frame mean against a 0.59 ms baseline.
    pub fn losing(&self) -> bool {
        self.split_loses
    }

    /// An emergency shed landed. The verdict is spent — it said "THIS share
    /// loses", not "every share loses", so the next probe must re-test at the
    /// new one; a latch that survived its own shed would run straight to zero
    /// off one measurement, skipping shares that may well pay.
    ///
    /// But clearing it alone is not enough, and this is the part that had to be
    /// measured to find: on an interfering box the BALANCE error is positive at
    /// small shares (the primary is the slow one there), so between probes it
    /// simply grows the share back and the controller parks at the balance
    /// point it was just told is worse than nothing. So growth is also blocked
    /// until a fresh probe re-tests. Shrinking stays free.
    pub fn shed_landed(&mut self) {
        self.split_loses = false;
        self.grow_block = true;
        self.dual_ms = 0.0;
    }

    /// Record what a solo (zero-share) frame cost, AND make the verdict.
    ///
    /// The comparison happens here, against `dual_ms` — an EWMA whose effective
    /// window is a handful of frames — so it is between the probe and the dual
    /// frames IMMEDIATELY BEFORE IT. That locality is the point: on a moving
    /// camera the per-frame cost varies with content, so averaging solo samples
    /// and dual samples over a whole shot and comparing the two means compares
    /// different parts of the scene. MEASURED: whole-shot averaging over a
    /// 48-frame GI tour left the controller oscillating at 3 rows while the
    /// capture ran 1.4x slower than single-GPU.
    ///
    /// The result LATCHES until the next probe, so the shed pressure between
    /// probes is consistent rather than flickering on frame-to-frame noise.
    pub fn solo(&mut self, ms: f32) {
        if !(ms > 0.0 && self.dual_ms > 0.0) {
            return;
        }
        if self.dual_ms > ms * (1.0 + SOLO_MARGIN) {
            self.split_loses = true;
        } else if self.dual_ms < ms * (1.0 - SOLO_MARGIN) {
            // A MEASURED WIN. Only this lifts the growth block — the split has
            // to EARN the right to grow back, not merely fail to lose.
            self.split_loses = false;
            self.grow_block = false;
        } else {
            // PARITY: stop shedding, but do not resume growing. Splitting at
            // parity buys nothing and costs a second scene upload, its VRAM and
            // a second device's power. Treating "not losing" as permission was
            // measured churning 14 adoptions over a 200-frame GI capture (9.8 s
            // against 7.1 s single-GPU): on a moving camera the frame cost
            // varies with content, so a parity verdict flips on noise and the
            // controller walks back up between probes.
            self.split_loses = false;
        }
    }

    /// Feed a frame in which the secondary ACTUALLY rendered.
    ///
    /// Never call this on a frame the secondary sat out: with `sec` near zero
    /// the error pins at +1 and the share climbs straight back out of the idle
    /// state it correctly reached. That oscillation is the whole reason
    /// `idle_frame`/`probe` exist as separate entry points.
    /// `frame_ms` is the tick's MEASURED wall time, the number `solo` records
    /// on a baseline frame — the two must be the same quantity or the
    /// comparison below is between different clocks. Pass 0.0 to fall back to
    /// `max(prim, sec)`, which is the same thing up to submit overhead.
    pub fn update(&mut self, prim_ms: f32, sec_ms: f32, frame_ms: f32) {
        self.idle = 0;
        let sum = prim_ms + sec_ms;
        if !(sum > 0.0) {
            return;
        }
        // The frame cost IS the later of the two — that is the objective, and
        // the balance error below is only a proxy for it.
        let cost = if frame_ms > 0.0 { frame_ms } else { prim_ms.max(sec_ms) };
        self.dual_ms =
            if self.dual_ms > 0.0 { self.dual_ms + (cost - self.dual_ms) * COST_EWMA } else { cost };
        // THE BASELINE VERDICT OVERRIDES THE BALANCE ONE. Shedding a row here
        // is not a guess: the next probe re-measures at the smaller share, so
        // it walks down only while the split is still losing — and stops at
        // whatever share does pay, zero included.
        let err = if self.split_loses { -1.0 } else { (prim_ms - sec_ms) / sum };
        if err.abs() < SHARE_DEADBAND {
            return;
        }
        let mut step = (SHARE_GAIN * err).clamp(-SHARE_DOWN_MAX, SHARE_UP_MAX);
        if self.grow_block {
            step = step.min(0.0);
        }
        self.share = (self.share + step).clamp(0.0, self.max);
    }

    /// ANTI-WINDUP: hold the continuous estimate within one quantum of what is
    /// actually APPLIED. Call once per tick, after `update`.
    ///
    /// Without it the controller keeps integrating in one direction for the
    /// whole dwell while the row count it is judging never changes — so it is
    /// re-counting the same evidence, and by the time the limiter may act the
    /// estimate has overshot far past the balance point. MEASURED: a 48-frame
    /// GI capture at 4/8 rows shed 8 x SHARE_DOWN_MAX = 0.8 during one 8-frame
    /// dwell, clamped at 0.0, and adopted ZERO in a single jump — skipping
    /// straight past the 2/8 the same numbers say is optimal, and then idling
    /// there because nothing re-probes within the shot. One quantum per dwell
    /// is not a rate limit bolted on; it is the largest step whose EFFECT the
    /// next tick can actually observe.
    pub fn anti_windup(&mut self, rows: u32, side: u32) {
        if side == 0 {
            return;
        }
        let q = 1.0 / side as f32;
        let at = rows as f32 * q;
        let lo = (at - q).max(0.0);
        let hi = (at + q).min(self.max).max(lo);
        self.share = self.share.clamp(lo, hi);
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
            // Clear the stale verdict: it was reached at a different share, on
            // a scene that has since moved. A freshly re-armed secondary gets
            // until the next solo probe to prove itself, rather than being shed
            // on the first `update` by a latch nothing has re-tested.
            self.split_loses = false;
            self.dual_ms = 0.0;
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
/// THE UPPER BOUND IS ONLY USABLE WHILE THE TRANSFER IS A LARGE FRACTION OF
/// THE WAIT, and that is a real limit rather than a caveat. The error it
/// produces is proportional to `out + hop`, so on a per-frame schedule like
/// `--spin` (0.4 ms frames, ~0.5 ms of transfer) it is decisive — but a
/// `--cinematic` capture amortises ONE crossing over `shot.samples` sub-frames
/// BY DESIGN, which shrinks it to ~0.2% and drops it inside `SHARE_DEADBAND`.
/// Measured: a 48-frame GI tour with the secondary 62 ms behind and the primary
/// at 0.00 produced |err| 0.002 and ZERO adoptions. So the capture path does not
/// use this function at all — it measures both sides directly with `join_poll`,
/// which it can afford because its calling thread is blocked either way. The
/// amortisation that makes a capture fast is exactly what blinds a controller
/// fed by inference.
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

/// Observe when EACH of two concurrently-launched submissions completes, in ms
/// from the moment polling began. Returns once both are done.
///
/// This is the exact measurement `phase_times` can only bound. Waiting on one
/// fence and inferring the other structurally loses whichever finished first,
/// and the inference's magnitude comes from the transfer — which a capture
/// amortises to nothing (see `phase_times`). Polling recovers both sides with
/// no such coupling.
///
/// It costs no wall clock: the calling thread has nothing to do until both
/// devices are finished, so this spins where it would otherwise block. It does
/// burn a core, so it belongs on batch paths (a capture) and not on a benchmark
/// whose reported number is wall clock. `yield_now` keeps it cooperative rather
/// than starving the driver's own threads.
///
/// The two predicates must be NON-BLOCKING fence queries. Handing this a
/// blocking wait would serialise the very concurrency it exists to measure.
pub fn join_poll(mut a: impl FnMut() -> bool, mut b: impl FnMut() -> bool) -> (f32, f32) {
    let t0 = std::time::Instant::now();
    let (mut ta, mut tb) = (None, None);
    loop {
        if ta.is_none() && a() {
            ta = Some(t0.elapsed());
        }
        if tb.is_none() && b() {
            tb = Some(t0.elapsed());
        }
        match (ta, tb) {
            (Some(x), Some(y)) => {
                return (x.as_secs_f32() * 1e3, y.as_secs_f32() * 1e3);
            }
            _ => std::thread::yield_now(),
        }
    }
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
    dwell: u32,
}

impl Default for ShareLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl ShareLimiter {
    pub fn new() -> Self {
        Self::with_dwell(SHARE_DWELL)
    }

    /// A limiter whose control tick is not a DISPLAY frame.
    ///
    /// The dwell's whole justification is that every applied change drops
    /// structure replay on both devices — so it is priced in replays lost per
    /// unit time, not in ticks. `--cinematic` ticks once per OUTPUT frame,
    /// where the camera has already moved and replay is therefore ALREADY
    /// invalidated at the boundary; the dwell there is pure anti-flap and can
    /// be far shorter. At the display-frame default a 120-frame shot would get
    /// at most one adoption, i.e. a balancer that cannot balance.
    pub fn with_dwell(dwell: u32) -> Self {
        Self { applied: None, since: 0, dwell }
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
        } else if target != cur && self.since >= self.dwell {
            self.applied = Some(target);
            self.since = 0;
        }
        self.applied.unwrap_or(cur)
    }
}

/// What a frame IS to the balancer, which is what decides how its cost is
/// interpreted afterwards.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tick {
    /// An ordinary frame at the share in force: its cost feeds `update`.
    Split,
    /// Forced to ZERO share to measure the no-split baseline (`solo`).
    Solo,
    /// Forced to ONE row to re-measure an idled secondary (`probe`).
    Probe,
    /// The share is zero and this frame is not a probe — the pre-feature path.
    Idle,
}

/// THE balancer: the share, the two controllers behind it, and the ONE
/// per-frame decision every driver makes.
///
/// IT IS ONE TYPE BECAUSE THREE COPIES OF IT DRIFTED. `--spin`, `--cinematic`
/// and the interactive presenters each grew their own fifteen-line version of
/// this tick, and a fix applied to two of them left the third running its solo
/// probe regardless of `--dual-gpu-auto` — where the probe restores from the
/// LIMITER, which a pinned run never applies, so the first one dropped a pinned
/// `--dual-gpu 4` to zero rows for the whole run. It failed silently: the run
/// completed, printed a per-phase table, and reported 0.97 ms/frame against a
/// 0.96 single-GPU baseline with 0.01 ms of transfer — indistinguishable from a
/// free split rather than an absent one. Folding the copies into one is the
/// actual fix; gating THIS is what makes it stay fixed.
///
/// The drivers differ only in what they can observe, never in the policy, so
/// they pass that in (`frozen`) and read the decision back out.
pub struct Balancer {
    ctl: ShareCtl,
    lim: ShareLimiter,
    depth: u32,
    auto: bool,
    rows: u32,
    /// What the LAST UNFROZEN tick actually rendered with — the share a frozen
    /// tick holds. Not the same thing as `rows`, which `observe` rewrites
    /// after a Solo or Probe frame; see `tick`.
    held: u32,
    kind: Tick,
    adopts: u64,
    solos: u64,
    probes: u64,
    idles: u64,
    /// Whether a split frame has ever actually run — so "converged to zero"
    /// can be told from "never armed", which look identical from outside.
    ran: bool,
}

impl Balancer {
    /// `share` is the starting row count, `depth` the split level (side =
    /// 2^depth), `auto` whether the balancer may move it at all.
    pub fn new(share: u32, depth: u32, auto: bool, dwell: u32) -> Self {
        let side = 1u32 << depth;
        let share = share.min(side.saturating_sub(1));
        Self {
            ctl: ShareCtl::new(share as f32 / side as f32, (side - 1) as f32 / side as f32),
            lim: ShareLimiter::with_dwell(dwell),
            depth,
            auto,
            rows: share,
            held: share,
            kind: Tick::Split,
            adopts: 0,
            solos: 0,
            probes: 0,
            idles: 0,
            ran: false,
        }
    }

    pub fn side(&self) -> u32 {
        1u32 << self.depth
    }
    pub fn depth(&self) -> u32 {
        self.depth
    }
    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn kind(&self) -> Tick {
        self.kind
    }
    pub fn ran(&self) -> bool {
        self.ran
    }
    pub fn share(&self) -> f32 {
        self.ctl.share()
    }
    /// `(adoptions, idle frames, re-probes, solo baselines)` for the verdict.
    pub fn counts(&self) -> (u64, u64, u64, u64) {
        (self.adopts, self.idles, self.probes, self.solos)
    }
    /// Mark that a split frame really rendered (the driver knows; the balancer
    /// only knows what it asked for).
    pub fn mark_ran(&mut self) {
        self.ran = true;
    }

    /// THE per-frame decision. Returns the rows the secondary takes THIS frame.
    ///
    /// `frozen` holds the current share — each driver decides what freezing
    /// means for it, because only the driver knows whether its frame is
    /// accumulating (see `GpuContext::record_trace`, where moving the split
    /// mid-accumulation is a correctness bug rather than a preference).
    ///
    /// EVERY probe is gated on `auto`, not just the adoption: a pinned share
    /// means "do not move it", and both probes move it for their frame and
    /// restore from the limiter afterwards — which a pinned run never applied.
    ///
    /// A FROZEN TICK HOLDS `held`, NOT `rows`, and the distinction is the
    /// whole point of the field. `observe` restores `rows` from the limiter
    /// after a Solo or Probe frame, so "keep doing what you were doing" read
    /// off `rows` could hand back a DIFFERENT share than the frame it is
    /// meant to be continuing: a Solo landing on the last moving frame (which
    /// stores at `frame == 0`) would be followed by a frozen frame at the
    /// restored share, and the secondary — which never ran the storing frame —
    /// would ADD its band onto an accumulation it never opened. That is a
    /// ghosted band appearing the instant the camera stops. Holding what the
    /// frame BEFORE the freeze actually rendered makes the store and every
    /// subsequent add agree by construction; the price is that a probe frame's
    /// share persists for that still period, which is one converge at a
    /// measured share rather than a wrong image.
    pub fn tick(&mut self, frozen: bool) -> u32 {
        self.kind = Tick::Split;
        if frozen {
            self.rows = self.held;
        } else if self.auto {
            // The idle re-probe: one forced row so an idled secondary can be
            // re-measured. Without it a single transient disables it forever.
            if self.rows == 0 && self.ctl.idle_frame() {
                self.probes += 1;
                self.kind = Tick::Probe;
                // A probe BYPASSES the limiter — it is one measurement, not an
                // adoption. Routing it through `apply` resets the dwell, which
                // then pins the secondary on for the whole dwell after every
                // probe (measured on --spin at 2x the baseline frame time).
                self.rows = 1;
                return self.commit();
            }
            // A measured loss to the no-split baseline is the emergency the
            // limiter's `shed` exists for — it may only ever shrink, so
            // bypassing the dwell cannot hand the secondary more work.
            let shed = self.ctl.losing();
            let want = quantize_share(self.ctl.share(), self.side(), self.rows);
            let next = self.lim.apply(want, shed);
            if next != self.rows {
                if shed && next < self.rows {
                    self.ctl.shed_landed();
                }
                self.adopts += 1;
                self.rows = next;
            }
            // The SOLO probe is the idle probe's mirror: one frame at ZERO
            // share, so the balancer has a measured no-split baseline to judge
            // the split against. Without it it can only find the best SPLIT,
            // which on an interfering box is still slower than not splitting.
            if self.rows > 0 && self.ctl.solo_frame() {
                self.solos += 1;
                self.kind = Tick::Solo;
                self.rows = 0;
                return self.commit();
            }
        }
        if self.rows == 0 {
            self.idles += 1;
            self.kind = Tick::Idle;
        }
        self.commit()
    }

    /// Latch what this frame renders with, so the next FROZEN tick can hold
    /// it. The one exit every arm of `tick` goes through.
    fn commit(&mut self) -> u32 {
        self.held = self.rows;
        self.rows
    }

    /// Close the loop with what the frame cost. `frame_ms` is its measured
    /// wall total — the same quantity on every kind of tick, which is what
    /// makes the solo comparison meaningful.
    ///
    /// Restores the limiter's own decision after a probe or a solo: both
    /// MEASURED, neither adopted.
    pub fn observe(&mut self, prim_ms: f32, sec_ms: f32, frame_ms: f32) {
        match self.kind {
            Tick::Split => {
                self.ctl.update(prim_ms, sec_ms, frame_ms);
                self.ctl.anti_windup(self.rows, self.side());
            }
            Tick::Solo => {
                self.ctl.solo(frame_ms);
                self.rows = self.lim.rows();
            }
            Tick::Probe => {
                self.ctl.probe(prim_ms, sec_ms, 1.0 / self.side() as f32);
                self.rows = self.lim.rows();
            }
            Tick::Idle => {}
        }
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

    // THE UPLOAD RING. `record_in` executes on the destination's FRAME list, so
    // it is still queued when the next frame's `hop` memcpys — one sub-buffer
    // per in-flight slot is what stops the second overwriting the first's
    // source. Both halves derive their base from `ring_base`, so what has to
    // hold is that the slots are distinct, `cap` apart (a full-size band at
    // slot i cannot reach slot i+1), and that the index WRAPS — the callers
    // pass a frame slot, and a raw frame index must be as safe.
    {
        let n = d3d12::FRAMES_IN_FLIGHT;
        if n < 2 {
            return Err("FRAMES_IN_FLIGHT < 2 makes the band's upload ring a single buffer, \
                        which is the tear this ring exists to prevent"
                .into());
        }
        let cap = payload_bytes(&MAX_FED_STRIDES, rw, rh);
        for i in 0..n {
            for j in 0..n {
                let (a, b) = (ring_base(cap, i), ring_base(cap, j));
                if (i == j) != (a == b) {
                    return Err(format!("ring slots {i} and {j} disagree about being distinct"));
                }
                if i != j && a.abs_diff(b) < cap as u64 {
                    return Err(format!(
                        "ring slots {i} and {j} are {} B apart but a band is up to {cap} B — \
                         one frame's copy would read the next frame's bytes",
                        a.abs_diff(b)
                    ));
                }
            }
        }
        if ring_base(cap, n) != ring_base(cap, 0) {
            return Err("ring_base must wrap: callers pass a frame slot, and a raw frame \
                        index has to land on the same entry"
                .into());
        }
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
        c.update(t * (1.0 - s), r * t * s + k * s, 0.0);
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
        c.update(t * (1.0 - s), r * t * s + k * s, 0.0);
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
        c.update(1.0, 10.0, 0.0);
    }
    if c.share() != 0.0 {
        return Err(format!(
            "a useless secondary must drive the share to EXACTLY 0.0, got {} — log-space \
             controllers cannot express this, which is why ScaleCtl was not reused",
            c.share()
        ));
    }
    for _ in 0..200 {
        c.update(1.0, 10.0, 0.0);
    }
    if c.share() != 0.0 {
        return Err("the zero state must be stable under continued bad frames".into());
    }

    // (2b) THE CLOSED LOOP — quantiser, limiter, anti-windup and the idle/probe
    // pair all in the loop against the same cost model. This is the
    // configuration that actually ships, and it catches what an open-loop
    // controller test structurally cannot: the estimate winding up during the
    // dwell and adopting a row count whose effect it never observed.
    //
    // The target is the DISCRETE argmin of max(prim, sec), brute-forced —
    // deliberately not `round(s* * side)`, which disagrees with it. At
    // (T 10, r 6, K 40) the continuous optimum is 0.73 rows, but one row
    // measures WORSE than none (12.5 vs 10.0), and rounding would have gated
    // the controller against an answer that is simply wrong.
    let side = 8u32;
    // `windup` false is the SHIPPING loop; true reproduces the pre-fix one, so
    // the gate can prove its own teeth rather than asserting them. Horizon and
    // dwell are parameters because THE FAILURE IS HORIZON-DEPENDENT — see the
    // teeth below.
    // `interf` is the fraction by which running BOTH devices slows the primary,
    // which is the effect that makes balance a wrong proxy for the minimum. 0.0
    // is the ideal non-interfering model; 0.58 is this box, fitted to the
    // measured GI capture.
    let closed_loop_i = |t: f32,
                         r: f32,
                         k: f32,
                         interf: f32,
                         solo_probe: bool,
                         windup: bool,
                         ticks: u32,
                         dwell: u32,
                         start: u32|
     -> f64 {
        let mut c = ShareCtl::new(start as f32 / side as f32, MAX);
        let mut l = ShareLimiter::with_dwell(dwell);
        let mut rows = start;
        let (mut sum, mut n) = (0.0f64, 0.0f64);
        for i in 0..ticks {
            let shed = c.losing();
            let next = l.apply(quantize_share(c.share(), side, rows), shed);
            if shed && next < rows {
                c.shed_landed();
            }
            rows = next;
            // A solo probe forces zero share for one tick, exactly like the
            // idle probe forces one row — and bypasses the limiter the same way.
            let solo = solo_probe && rows > 0 && c.solo_frame();
            if solo {
                rows = 0;
            }
            if rows == 0 {
                let cost = t;
                if solo {
                    c.solo(cost);
                } else if c.idle_frame() {
                    let s = 1.0 / side as f32;
                    let prim = t * (1.0 - s) * (1.0 + interf);
                    c.probe(prim, r * t * s + k * s, s);
                }
            } else {
                let s = rows as f32 / side as f32;
                c.update(t * (1.0 - s) * (1.0 + interf), r * t * s + k * s, 0.0);
                if !windup {
                    c.anti_windup(rows, side);
                }
            }
            if i * 4 >= ticks * 3 {
                sum += rows as f64;
                n += 1.0;
            }
        }
        sum / n
    };
    let closed_loop = |t: f32, r: f32, k: f32, windup: bool, ticks: u32, dwell: u32, start: u32| -> f64 {
        let mut c = ShareCtl::new(start as f32 / side as f32, MAX);
        let mut l = ShareLimiter::with_dwell(dwell);
        let mut rows = start;
        let (mut sum, mut n) = (0.0f64, 0.0f64);
        for i in 0..ticks {
            // DECIDE, then render at that decision, then observe — the shipping
            // order. Reversing it lets the limiter's first (unconditional)
            // adoption consume a share the controller has already moved, which
            // hides the windup entirely: the first draft of this gate did
            // exactly that and reported the broken loop converging.
            rows = l.apply(quantize_share(c.share(), side, rows), false);
            if rows == 0 {
                // The secondary sat this tick out, so `update` must not see it
                // (sec ~ 0 pins err at +1). Only a forced probe re-measures.
                if c.idle_frame() {
                    let s = 1.0 / side as f32;
                    c.probe(t * (1.0 - s), r * t * s + k * s, s);
                }
            } else {
                let s = rows as f32 / side as f32;
                c.update(t * (1.0 - s), r * t * s + k * s, 0.0);
                if !windup {
                    c.anti_windup(rows, side);
                }
            }
            if i * 4 >= ticks * 3 {
                sum += rows as f64;
                n += 1.0;
            }
        }
        sum / n
    };
    let discrete_opt = |t: f32, r: f32, k: f32| -> u32 {
        let cost = |rows: u32| {
            let s = rows as f32 / side as f32;
            (t * (1.0 - s)).max(r * t * s + k * s)
        };
        (0..side).min_by(|&a, &b| cost(a).total_cmp(&cost(b))).unwrap()
    };
    for (t, r, k) in [(10.0f32, 2.0f32, 5.0f32), (20.0, 1.0, 1.0), (10.0, 6.0, 40.0)] {
        let want = discrete_opt(t, r, k);
        // A discrete actuator cannot sit ON a continuous optimum, so it dithers
        // between the two row counts bracketing it — which is exactly why
        // growing is 5x slower than shedding, biasing the duty cycle toward the
        // cheaper side. Score the time AVERAGE, not a snapshot.
        let mean = closed_loop(t, r, k, false, 2000, 4, side - 1);
        if (mean - want as f64).abs() > 1.0 {
            return Err(format!(
                "closed loop (T {t}, r {r}, K {k}) averaged {mean:.2} rows, not the discrete \
                 optimum {want} — quantiser/limiter/anti-windup are not converging together"
            ));
        }
    }
    // TEETH, AND THE HORIZON IS PART OF THEM. Over thousands of ticks the
    // wound-up controller ALSO finds the optimum — the probe machinery digs it
    // back out — so a long-horizon teeth check passes on the broken code and
    // proves nothing (it did, on the first attempt at this gate). The failure
    // is a CAPTURE's horizon: 48 output frames, the 8-frame dwell, starting
    // where `--dual-gpu 4` starts, and `SHARE_PROBE_FRAMES` longer than the
    // whole shot. Constants fitted to the measured GI capture (primary 16.6 ms
    // and secondary 37.6 ms per sub-frame at 4/8), where the discrete optimum
    // is 2 rows: the shipping loop must find it and the wound-up one must shed
    // 8 x SHARE_DOWN_MAX in a single dwell, adopt ZERO in one jump, and then
    // idle there for the rest of the shot.
    let (t, r, k) = (33.2f32, 2.26f32, 0.04f32);
    let want = discrete_opt(t, r, k);
    let good = closed_loop(t, r, k, false, 48, 8, 4);
    if (good - want as f64).abs() > 1.0 {
        return Err(format!(
            "over a capture's horizon the loop averaged {good:.2} rows, not the optimum {want}"
        ));
    }
    let bad = closed_loop(t, r, k, true, 48, 8, 4);
    if (bad - want as f64).abs() <= 1.0 {
        return Err(format!(
            "the pre-anti-windup loop also averaged {bad:.2} rows over a capture's horizon — \
             this gate is not testing what it claims to"
        ));
    }

    // (2c) THE INTERFERING BOX — the one this ships on, and the case where
    // BALANCE IS THE WRONG OBJECTIVE. Constants fitted to the measured GI
    // capture: a 4090 that renders the whole screen in 21 ms takes 16.6 for
    // half of it while the B70 runs, and the B70 takes 37.6 for the other
    // half. Every split therefore costs MORE than not splitting (the cheapest,
    // 2 rows, is 24.9 against 21.0), so the only right answer is zero — and a
    // controller that only equalises `prim` and `sec` cannot see that.
    let (t, r, k, interf) = (21.0f32, 3.58f32, 0.04f32, 0.58f32);
    for rows in 1..side {
        let s = rows as f32 / side as f32;
        let cost = (t * (1.0 - s) * (1.0 + interf)).max(r * t * s + k * s);
        if cost <= t {
            return Err(format!(
                "the interference model is not adversarial: {rows} rows costs {cost:.1} against \
                 a solo {t:.1}, so zero is not the right answer and the gate proves nothing"
            ));
        }
    }
    let with_solo = closed_loop_i(t, r, k, interf, true, false, 400, 8, 4);
    if with_solo > 0.05 {
        return Err(format!(
            "on an interfering box the controller averaged {with_solo:.2} rows — every split \
             there is slower than no split, so it must reach and hold ZERO"
        ));
    }
    // TEETH: without the solo baseline the SAME loop must provably settle on a
    // split, because equalising prim and sec is a proxy that this box breaks.
    let no_solo = closed_loop_i(t, r, k, interf, false, false, 400, 8, 4);
    if no_solo <= 0.35 {
        return Err(format!(
            "balance-only also reached {no_solo:.2} rows — the solo probe is not being tested \
             (it is what turns 'the best split' into 'is splitting worth it')"
        ));
    }
    // THE ESCAPE MUST BE FAST, and at the DISPLAY dwell, which is where it
    // matters: `--spin`'s 90-frame dwell would otherwise cost one dwell per row
    // to walk out of a losing configuration (4 x SHARE_DWELL = 360 frames,
    // measured at 2.67 ms/frame against a 0.59 ms baseline). The emergency shed
    // — the limiter's own mechanism, previously never used by anything — must
    // bring that inside a couple of probe intervals while still re-testing at
    // each share on the way down.
    let escaped = closed_loop_i(t, r, k, interf, true, false, 200, SHARE_DWELL, 4);
    if escaped > 0.05 {
        return Err(format!(
            "at the display dwell the controller still averaged {escaped:.2} rows over 200 \
             ticks — the emergency shed is not bypassing the dwell for a measured loss"
        ));
    }
    // ...and on a NON-interfering box the solo probe must not cost anything:
    // the split genuinely wins there, so it must still be found.
    let (t2, r2, k2) = (20.0f32, 1.0f32, 1.0f32);
    let want2 = discrete_opt(t2, r2, k2);
    let ideal = closed_loop_i(t2, r2, k2, 0.0, true, false, 2000, 4, side - 1);
    if (ideal - want2 as f64).abs() > 1.0 {
        return Err(format!(
            "with a solo baseline on a non-interfering box the loop averaged {ideal:.2} rows, \
             not the optimum {want2} — the baseline test must not veto a split that pays"
        ));
    }
    // ANTI-WINDUP ITSELF: one quantum is the largest step whose effect the next
    // tick can observe, so the estimate may never sit further out than that.
    for rows in 0..8u32 {
        let mut c = ShareCtl::new(MAX, MAX);
        for _ in 0..200 {
            c.update(100.0, 1.0, 0.0); // maximally primary-bound: pin the growth
            c.anti_windup(rows, 8);
        }
        if c.share() > (rows as f32 + 1.0) / 8.0 + 1e-6 {
            return Err(format!(
                "the estimate wound up to {} at {rows} applied rows — a whole dwell of \
                 one-sided error must not carry it past one quantum",
                c.share()
            ));
        }
        let mut c = ShareCtl::new(0.0, MAX);
        for _ in 0..200 {
            c.update(1.0, 100.0, 0.0);
            c.anti_windup(rows, 8);
        }
        if c.share() + 1e-6 < (rows as f32 - 1.0).max(0.0) / 8.0 {
            return Err(format!("the estimate wound DOWN past one quantum at {rows} rows"));
        }
    }

    // (3) PER-FRAME STEP BOUNDS, both directions.
    let mut c = ShareCtl::new(0.4, MAX);
    c.update(100.0, 1.0, 0.0); // maximally primary-bound: grow
    if c.share() - 0.4 > SHARE_UP_MAX + 1e-6 {
        return Err(format!("grew {} in one frame, over the {SHARE_UP_MAX} cap", c.share() - 0.4));
    }
    let mut c = ShareCtl::new(0.4, MAX);
    c.update(1.0, 100.0, 0.0); // maximally secondary-bound: shed
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
        c.update(10.0, 10.2, 0.0); // |err| ~ 0.01, inside the band
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
    c.update(p, s, 0.0);
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

    // (5c) THE EXACT ALTERNATIVE. `join_poll` must return only when BOTH are
    // done, must order the two finishes correctly, and must not conflate them —
    // the failure `phase_times` has by construction and that measured as zero
    // adoptions on a 48-frame GI capture. Driven by counters, so it is
    // deterministic in everything except the wall-clock magnitudes.
    for (na, nb) in [(1u32, 40u32), (40, 1), (1, 1)] {
        let (mut ca, mut cb) = (0u32, 0u32);
        let (ta, tb) = join_poll(
            || {
                ca += 1;
                ca >= na
            },
            || {
                cb += 1;
                cb >= nb
            },
        );
        if !(ta.is_finite() && tb.is_finite() && ta >= 0.0 && tb >= 0.0) {
            return Err(format!("join_poll returned nonsense: ({ta}, {tb})"));
        }
        // The one that needed more polls cannot have been observed EARLIER.
        // (Equal counts may tie either way, so only compare when they differ.)
        if na < nb && ta > tb {
            return Err("join_poll reported the earlier finisher as the later one".into());
        }
        if nb < na && tb > ta {
            return Err("join_poll reported the earlier finisher as the later one".into());
        }
    }

    // (5d) THE BALANCER AS ONE OBJECT — the tick every driver now shares, and
    // the gate that exists because three hand-copies of it drifted.
    //
    // THE PIN. A `--dual-gpu N` without `--dual-gpu-auto` must hold N for the
    // whole run, whatever it measures and however hostile: no adoption, no solo
    // probe, no idle probe. That is the bug this file's history is about, and
    // it was invisible in the product — a pinned run reported plausible timings
    // that happened to be single-GPU's.
    for pin in 1..8u32 {
        let mut b = Balancer::new(pin, 3, false, 4);
        for i in 0..1000 {
            if b.tick(false) != pin {
                return Err(format!(
                    "a PINNED balancer moved off {pin} rows at tick {i} (to {}, kind {:?}) — \
                     `--dual-gpu {pin}` without --dual-gpu-auto must never move",
                    b.rows(),
                    b.kind()
                ));
            }
            if b.kind() != Tick::Split {
                return Err(format!(
                    "a PINNED balancer took a {:?} tick — both probes zero or force the share \
                     for their frame and restore from the LIMITER, which a pinned run never \
                     applies, so one is enough to strand it",
                    b.kind()
                ));
            }
            // Feed it the most hostile evidence available, in both directions.
            b.observe(if i % 2 == 0 { 100.0 } else { 1.0 }, 1.0, 50.0);
        }
        let (adopts, idles, probes, solos) = b.counts();
        if (adopts, idles, probes, solos) != (0, 0, 0, 0) {
            return Err(format!(
                "a PINNED balancer counted ({adopts} adoptions, {idles} idle, {probes} probes, \
                 {solos} solos) — it must do nothing at all"
            ));
        }
    }
    // ...and the same balancer with `auto` must provably NOT be inert, or the
    // pin gate above passes on a balancer that never works.
    {
        let mut b = Balancer::new(4, 3, true, 4);
        let (mut saw_solo, mut saw_move) = (false, false);
        for _ in 0..400 {
            let rows = b.tick(false);
            if b.kind() == Tick::Solo {
                saw_solo = true;
            }
            if rows != 4 && b.kind() == Tick::Split {
                saw_move = true;
            }
            // A hopeless secondary: the split always costs more than solo.
            let (prim, sec) = (10.0f32, 40.0f32);
            b.observe(prim, sec, if b.kind() == Tick::Solo { 12.0 } else { 40.0 });
        }
        if !saw_solo || !saw_move {
            return Err(format!(
                "an AUTO balancer was inert (solo probe {saw_solo}, share moved {saw_move}) — \
                 the pin gate above would then prove nothing"
            ));
        }
        if b.rows() != 0 {
            return Err(format!(
                "an auto balancer facing a hopeless secondary settled at {} rows, not 0",
                b.rows()
            ));
        }
    }
    // FROZEN holds the share without consuming a probe: a driver that freezes
    // for correctness (an accumulating frame) must not silently lose its
    // baseline schedule to the freeze.
    {
        let mut b = Balancer::new(3, 3, true, 4);
        for _ in 0..500 {
            if b.tick(true) != 3 {
                return Err("a FROZEN tick must hold the share exactly".into());
            }
            if b.kind() != Tick::Split {
                return Err(format!("a FROZEN tick took a {:?}", b.kind()));
            }
        }
    }
    // A probe and a solo each MEASURE, never adopt: the share afterwards is the
    // limiter's own, not the forced value.
    {
        let mut b = Balancer::new(0, 3, true, 4);
        let mut probed = false;
        for _ in 0..SHARE_PROBE_FRAMES + 8 {
            let rows = b.tick(false);
            if b.kind() == Tick::Probe {
                if rows != 1 {
                    return Err(format!("an idle probe must force exactly 1 row, got {rows}"));
                }
                probed = true;
                b.observe(1.0, 40.0, 40.0); // unfavourable
                if b.rows() != 0 {
                    return Err(format!(
                        "after an unfavourable probe the share must return to the limiter's \
                         own value (0), got {}",
                        b.rows()
                    ));
                }
            } else {
                b.observe(1.0, 1.0, 1.0);
            }
        }
        if !probed {
            return Err("an idled auto balancer never re-probed".into());
        }
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
    // A shortened dwell must be OBEYED and must be the only thing that changed:
    // adopt, hold, and move on exactly the d-th tick after adoption — the same
    // accounting `SHARE_DWELL` gets above. `--cinematic` ticks once per OUTPUT
    // frame and at the display default would get ~1 adoption per shot.
    for d in [0u32, 1, 8] {
        let mut l = ShareLimiter::with_dwell(d);
        l.apply(4, false);
        for i in 1..d {
            if l.apply(2, false) != 4 {
                return Err(format!("dwell {d} released early, at tick {i}"));
            }
        }
        if l.apply(2, false) != 2 {
            return Err(format!("dwell {d} never released"));
        }
    }

    // (8) THE FREEZE HOLDS WHAT THE PREVIOUS FRAME RENDERED, not what
    // `observe` restored afterwards — the accumulation rule in
    // `GpuContext::record_trace`. A SOLO frame is the case that separates the
    // two: it renders at ZERO rows and STORES (it is an unfrozen frame, so
    // `frame == 0`), then `observe` puts the share back from the limiter. If
    // the following frozen frame read `rows` it would re-arm the split, and
    // the secondary — which never ran the storing frame — would ADD its band
    // into an accumulation it never opened: a ghosted band the instant the
    // camera stops.
    {
        let mut b = Balancer::new(3, 3, true, SHARE_DWELL);
        let mut solo = false;
        for _ in 0..4096 {
            let rows = b.tick(false);
            if b.kind() == Tick::Solo {
                if rows != 0 {
                    return Err(format!("a SOLO tick must render at zero rows, got {rows}"));
                }
                // A solo frame that measures SLOWER than the splits keeps the
                // limiter's share alive, which is what makes the restore — and
                // therefore this gate — non-vacuous.
                b.observe(0.0, 0.0, 2.0);
                solo = true;
                break;
            }
            b.observe(1.0, 1.0, 1.0);
        }
        if !solo {
            return Err("no SOLO baseline in 4096 ticks — the freeze gate would be vacuous".into());
        }
        // TEETH: `observe` really did restore a DIFFERENT share, so holding
        // zero below is a distinction rather than a coincidence.
        if b.rows() == 0 {
            return Err("the solo restore left zero rows — the freeze gate proves nothing".into());
        }
        if b.tick(true) != 0 {
            return Err(
                "a frozen tick after a SOLO frame must hold ZERO rows: the solo frame is the \
                 one that STORED, and the secondary did not run it"
                    .into(),
            );
        }
        // ...and the hold is not a latch. The moment the freeze lifts the
        // balancer is free again, or a still frame would disable the secondary
        // for the rest of the session.
        let back = b.tick(false);
        if back == 0 {
            return Err("the freeze must release: an unfrozen tick is free to re-adopt".into());
        }
        // An ordinary split frame's share is held across a freeze unchanged —
        // the general contract, of which the solo case is the sharp corner.
        b.observe(1.0, 1.0, 1.0);
        if b.tick(true) != back {
            return Err("a frozen tick must hold the previous frame's share exactly".into());
        }
    }
    Ok(())
}
