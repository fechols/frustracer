//! GPU timestamp queries — per-region GPU time on the command list, printed
//! as a table. The vendor-neutral half of the profiling story: PIX's GPU
//! captures give a far richer picture but its analysis engine only replays on
//! AMD/NVIDIA, so on an Intel adapter there is no way to get numbers out of a
//! `.wpix` at all. D3D12 timestamps work on every vendor, so this is what
//! answers "where is the frame going" on Arc.
//!
//! **The PIX marker brackets ARE the timing brackets** — `pix::scope(list,
//! c"wavefront")` opens a timestamp region of the same name, so the two
//! instruments can never drift apart and a new marker is automatically timed.
//! Opt-in (`--gpu-timing`), and every call is an inert no-op when off: no
//! query heap is created, no name is allocated, and the `EndQuery` calls are
//! never recorded, so default sessions and every `--check*` stay byte-identical.
//!
//! Correctness of the readback rests on the frame ring: `D3d::begin_frame`
//! has already waited on this slot's fence, so the slot's timestamps are
//! complete before we map them. We therefore report frame N's timings at the
//! start of frame N + FRAMES_IN_FLIGHT, never stalling the pipeline to do it.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use windows::Win32::Graphics::Direct3D12::*;

use super::d3d12::{ReadbackBuffer, FRAMES_IN_FLIGHT};

/// Timestamps (not regions) per frame slot. Regions nest, so this is 2x the
/// region count; the deepest frame here records ~20 regions.
const MAX_TS: u32 = 128;

/// Frames between printed tables.
const REPORT_EVERY: u64 = 120;

static ENABLED: AtomicBool = AtomicBool::new(false);

/// One timer PER DEVICE, keyed by adapter LUID, in first-seen order.
///
/// A query heap belongs to the device that created it and a timestamp frequency
/// is a property of one queue, so resolving device B's timestamps into device
/// A's heap is invalid — which is exactly what a single global did the moment
/// `--dual-gpu` put two devices in one process. Per-device is also what the
/// split's load balancer needs: "how long did EACH card take" is the input, and
/// a merged table cannot answer it.
///
/// A `Vec` rather than a map: there are at most a handful of adapters on a box,
/// entries are only ever pushed (so an index stays valid for the process's
/// life — which is what lets `CURRENT` be an index), and first-seen order is
/// the order the tables should print in.
static TIMERS: Mutex<Vec<Timer>> = Mutex::new(Vec::new());

/// Index into `TIMERS`, PLUS ONE — 0 means "no device has begun a frame yet".
///
/// The recording-side calls (`scope`, `resolve`, the `TimeScope` drop) sit deep
/// inside command-list recording and have no device in hand, so `begin_frame`
/// publishes which device is being recorded and they follow it. Sound because
/// recording is sequential on the main thread.
///
/// IT DOES NEST, exactly once, and that is why `begin_frame` hands back a
/// token. The `--dual-gpu` schedule submits the SECONDARY's whole frame from
/// inside the primary's open one (`GpuContext::record_trace`) — that is the
/// point of the feature, since serialising the two would measure their sum.
/// A plain store would leave `CURRENT` on the secondary for the rest of the
/// primary's frame, and the primary's remaining `EndQuery`/`ResolveQueryData`
/// calls would then name the SECONDARY device's query heap from the PRIMARY's
/// command list: invalid D3D12 (a GBV flood with `--gpu-debug`, a plausible
/// route to DEVICE_REMOVED without it), and — where a driver tolerates it —
/// two silently wrong `--gpu-timing` tables, which is the one measurement the
/// per-device split exists to produce. Save and restore, never assume.
static CURRENT: AtomicUsize = AtomicUsize::new(0);

/// One open/closed region within a frame: the two timestamp indices —
/// SLOT-RELATIVE, since `collect` maps one slot's window of the heap — and the
/// nesting depth at the time it was opened (so the table reads like the
/// command list). The ABSOLUTE index (slot_base + these) is what `EndQuery`
/// takes; it rides the guard, never this struct.
struct Region {
    name: String,
    depth: u32,
    begin: u32,
    end: u32,
}

/// Running per-name accumulation across frames. `order` is first-seen index —
/// the table prints in command-list order, not alphabetically.
struct Acc {
    order: usize,
    depth: u32,
    sum_ms: f64,
    max_ms: f64,
    calls: u64,
    frames: u64,
    /// `sum_ms` as of the last printed table — the windowed column's baseline.
    /// See `Timer::last_frames` for why the window exists at all.
    last_sum_ms: f64,
    /// `max_ms` within the current window, reset at every report.
    win_max_ms: f64,
}

struct Timer {
    /// The adapter LUID this timer's heap belongs to — the map key.
    luid: u64,
    /// Short human label for the table header (the adapter's vendor). Only
    /// printed when more than one device is timed, so single-GPU sessions'
    /// output is byte-identical to before.
    label: String,
    heap: ID3D12QueryHeap,
    readback: ReadbackBuffer,
    /// Ticks per second on this queue.
    freq: u64,
    /// Regions recorded for the slot currently being recorded.
    regions: Vec<Region>,
    /// Timestamps consumed in the slot currently being recorded.
    used: u32,
    /// Nesting depth of the currently open regions.
    depth: u32,
    /// The slot currently being recorded (set by `begin_frame`).
    cur_slot: usize,
    /// Regions awaiting readback, per in-flight slot; None = nothing pending.
    pending: [Option<Vec<Region>>; FRAMES_IN_FLIGHT],
    stats: Vec<(String, Acc)>,
    /// Frames whose timings we have actually collected.
    frames: u64,
    /// `frames` as of the last printed table. Two jobs: it is the windowed
    /// column's baseline, and comparing against it is what schedules the next
    /// report (which is why there is no "bump frames so we don't re-report"
    /// hack — that hack cost the window one frame in every REPORT_EVERY).
    ///
    /// THE WINDOW IS THE POINT. The cumulative mean divides by every frame
    /// since the first, so on Intel it permanently carries the async-compile
    /// fallback: a fresh DXIL variant runs ~4.7x on the span (~7.6x on the
    /// leaf kernel) for the 5-8 s before the driver's background recompile
    /// lands, which at ~190 fps is 1200-1500 frames — +80% over a 6k-frame
    /// run, +7% over 30k, and UNEVEN per region, so it distorts the shape and
    /// not just the total. A per-window column makes each table an independent
    /// sample (median + IQR across ~90 of them per minute) and makes the
    /// fallback self-diagnosing: it reads as a step down to a flat floor
    /// rather than a slow drift no single number can reveal.
    last_frames: u64,
    /// Dropped because a frame blew MAX_TS — reported, never silent.
    dropped: u64,
}

pub fn enable(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
    if on {
        eprintln!(
            "gputime: --gpu-timing on; per-region GPU timestamps every {REPORT_EVERY} frames \
             (regions are the PIX marker brackets)"
        );
    }
}

fn on() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Run `f` against the device whose frame is currently being recorded.
///
/// `None` — and therefore a no-op at every call site — before any `begin_frame`,
/// which is exactly the inert behavior the single `Option<Timer>` used to give
/// when no device had created a timer yet.
fn with_current<R>(f: impl FnOnce(&mut Timer) -> R) -> Option<R> {
    let idx = CURRENT.load(Ordering::Relaxed);
    if idx == 0 {
        return None;
    }
    let mut guard = TIMERS.lock().unwrap();
    guard.get_mut(idx - 1).map(f)
}

/// Pack an adapter LUID into a comparable key.
fn luid_key(device: &ID3D12Device) -> u64 {
    let l = unsafe { device.GetAdapterLuid() };
    ((l.HighPart as u32 as u64) << 32) | l.LowPart as u64
}

impl Timer {
    fn new(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> Result<Self, String> {
        let desc = D3D12_QUERY_HEAP_DESC {
            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
            Count: MAX_TS * FRAMES_IN_FLIGHT as u32,
            NodeMask: 0,
        };
        let mut heap: Option<ID3D12QueryHeap> = None;
        unsafe { device.CreateQueryHeap(&desc, &mut heap) }
            .map_err(|e| format!("CreateQueryHeap(timestamp): {e}"))?;
        let readback =
            ReadbackBuffer::new(device, MAX_TS as usize * FRAMES_IN_FLIGHT * 8)?;
        let freq = unsafe { queue.GetTimestampFrequency() }
            .map_err(|e| format!("GetTimestampFrequency: {e}"))?;
        Ok(Self {
            luid: luid_key(device),
            label: format!("{:?}", super::adapter::vendor_of_device(device)),
            heap: heap.unwrap(),
            readback,
            freq,
            regions: Vec::new(),
            used: 0,
            depth: 0,
            cur_slot: 0,
            pending: Default::default(),
            stats: Vec::new(),
            frames: 0,
            last_frames: 0,
            dropped: 0,
        })
    }

    /// Map this slot's timestamps and fold them into the running stats. The
    /// caller has already waited on the slot's fence.
    fn collect(&mut self, slot: usize) {
        let Some(regions) = self.pending[slot].take() else { return };
        if regions.is_empty() {
            return;
        }
        let base = slot * MAX_TS as usize;
        let byte_range = D3D12_RANGE {
            Begin: base * 8,
            End: (base + MAX_TS as usize) * 8,
        };
        let mut ptr = std::ptr::null_mut();
        if unsafe { self.readback.resource.Map(0, Some(&byte_range), Some(&mut ptr)) }.is_err() {
            return;
        }
        let ticks: &[u64] = unsafe {
            std::slice::from_raw_parts((ptr as *const u64).add(base), MAX_TS as usize)
        };
        let freq = self.freq as f64;
        let to_ms = move |t: u64| (t as f64) * 1000.0 / freq;

        let mut span_lo = u64::MAX;
        let mut span_hi = 0u64;
        for r in &regions {
            let (b, e) = (ticks[r.begin as usize], ticks[r.end as usize]);
            // A GPU can retire timestamps out of order across a reset; a
            // backwards pair is meaningless, not negative time — drop it.
            if e < b {
                continue;
            }
            span_lo = span_lo.min(b);
            span_hi = span_hi.max(e);
            self.add(&r.name, r.depth, to_ms(e) - to_ms(b));
        }
        if span_hi > span_lo {
            self.add("= gpu frame span", 0, to_ms(span_hi) - to_ms(span_lo));
        }
        unsafe { self.readback.resource.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 })) };
        self.frames += 1;
    }

    fn add(&mut self, name: &str, depth: u32, ms: f64) {
        match self.stats.iter_mut().find(|(n, _)| n == name) {
            Some((_, a)) => {
                a.sum_ms += ms;
                a.max_ms = a.max_ms.max(ms);
                a.win_max_ms = a.win_max_ms.max(ms);
                a.calls += 1;
                a.frames += 1;
            }
            None => {
                let order = self.stats.len();
                self.stats.push((
                    name.to_string(),
                    Acc {
                        order,
                        depth,
                        sum_ms: ms,
                        max_ms: ms,
                        calls: 1,
                        frames: 1,
                        last_sum_ms: 0.0,
                        win_max_ms: ms,
                    },
                ));
            }
        }
    }

    /// Print the table and roll the window forward. `&mut` because the window
    /// baselines are snapshotted here — reporting IS what closes a window.
    ///
    /// Both columns divide by the FRAME count (not the region's own call
    /// count), so a region that fires on only some frames reads as its share
    /// of the frame, and the children of a bracket still sum toward it. That
    /// also means a region which stops firing — `wavefront` once structure
    /// replay takes over — correctly windows to 0.000 while its cumulative
    /// column keeps the history.
    /// `multi` adds the adapter label to the header. It is false in a
    /// single-device session, so that output stays byte-identical to the
    /// pre-dual-GPU one and no recorded table is invalidated by this change.
    fn report(&mut self, multi: bool) {
        if self.frames == 0 {
            return;
        }
        let win_frames = self.frames.saturating_sub(self.last_frames);
        let who = if multi { format!(" [{}]", self.label) } else { String::new() };
        eprintln!(
            "\ngputime{}: per-region GPU ms | win = last {} frames, mean = all {} frames",
            who,
            win_frames.max(1),
            self.frames
        );
        eprintln!(
            "  {:<28} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "region", "win ms", "win max", "mean ms", "max ms", "calls/f"
        );
        let mut idx: Vec<usize> = (0..self.stats.len()).collect();
        idx.sort_by_key(|&i| self.stats[i].1.order);
        let wf = win_frames.max(1) as f64;
        for &i in &idx {
            let (name, a) = &self.stats[i];
            let indent = "  ".repeat(a.depth as usize);
            eprintln!(
                "  {:<28} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.1}",
                format!("{indent}{name}"),
                (a.sum_ms - a.last_sum_ms) / wf,
                a.win_max_ms,
                a.sum_ms / self.frames as f64,
                a.max_ms,
                a.calls as f64 / self.frames as f64
            );
        }
        if self.dropped > 0 {
            eprintln!("  (dropped {} regions over MAX_TS={MAX_TS})", self.dropped);
        }
        for (_, a) in self.stats.iter_mut() {
            a.last_sum_ms = a.sum_ms;
            a.win_max_ms = 0.0;
        }
        self.last_frames = self.frames;
    }
}

/// Drain the accumulated per-region means and reset the accumulator, so the
/// caller can attribute them to ONE labelled workload. `--check-gpu`'s bench
/// calls this after every row: the running table `report` prints is exactly
/// wrong there, because its divisor is the FRAME count — fold the reference
/// kernel's frames in with the wavefront's and every region's mean is diluted
/// by the frames that never ran it. Per-row is also what makes the numbers
/// actionable: "AMD is slower per sample" is not, "AMD is slower in the leaf
/// kernel only, while its reference kernel is FASTER" points at the kernel.
///
/// Pending (submitted, not yet collected) frames are DROPPED, not collected:
/// the caller's last frame is still in flight, and folding it into the next
/// row would attribute one frame of the old config to the new one.
pub fn take_regions() -> Vec<(String, u32, f64)> {
    if !on() {
        return Vec::new();
    }
    with_current(|t| {
        let frames = t.frames.max(1) as f64;
        let mut rows: Vec<(String, Acc)> = std::mem::take(&mut t.stats);
        rows.sort_by_key(|(_, a)| a.order);
        t.frames = 0;
        t.last_frames = 0;
        t.pending = Default::default();
        rows.into_iter().map(|(n, a)| (n, a.depth, a.sum_ms / frames)).collect()
    })
    .unwrap_or_default()
}

/// Whichever device's frame was open when a nested one began. Hand it to
/// `restore` once the nested frame is recorded and resolved — see `CURRENT`.
#[must_use = "a nested gputime frame must be restored, or the outer device's \
              timestamps are recorded against this one's query heap"]
pub struct Nested(usize);

/// Republish the device whose frame `begin_frame` interrupted.
pub fn restore(prev: Nested) {
    CURRENT.store(prev.0, Ordering::Relaxed);
}

/// Collect the previous timings for this slot and start recording it afresh.
/// Called from `D3d::begin_frame` (and `HeadlessGpu::submit`/`run`), after the
/// slot's fence wait.
///
/// The returned token restores whatever was open before — an outermost frame
/// discards it (there is nothing to go back to), the dual-GPU schedule's inner
/// one must not.
pub fn begin_frame(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    slot: usize,
) -> Nested {
    let prev = Nested(CURRENT.load(Ordering::Relaxed));
    if !on() {
        return prev;
    }
    let mut guard = TIMERS.lock().unwrap();
    let luid = luid_key(device);
    let idx = match guard.iter().position(|t| t.luid == luid) {
        Some(i) => i,
        None => match Timer::new(device, queue) {
            Ok(t) => {
                guard.push(t);
                guard.len() - 1
            }
            Err(e) => {
                eprintln!("gputime: {e}; timing off");
                ENABLED.store(false, Ordering::Relaxed);
                CURRENT.store(0, Ordering::Relaxed);
                // Timing is off from here on, so the token's only job is to
                // keep the caller's `restore` a no-op that agrees with it.
                return Nested(0);
            }
        },
    };
    // Publish before recording: every `scope` between here and `resolve`
    // follows this, and they run on the same thread.
    CURRENT.store(idx + 1, Ordering::Relaxed);
    // Snapshot before the mutable borrow of the entry below.
    let multi = guard.len() > 1;
    let t = &mut guard[idx];
    t.collect(slot);
    // Scheduled off the last report's frame count, not off a modulus: `report`
    // closes the window, so this can never fire twice on one count and every
    // window is exactly REPORT_EVERY collected frames wide.
    if t.frames >= t.last_frames + REPORT_EVERY {
        t.report(multi);
    }
    t.regions = Vec::new();
    t.used = 0;
    t.depth = 0;
    t.cur_slot = slot;
    prev
}

/// Record the ResolveQueryData for the slot just recorded. Called from
/// `D3d::end_frame` / `submit_and_wait`, immediately before `Close`.
pub fn resolve(list: &ID3D12GraphicsCommandList, slot: usize) {
    if !on() {
        return;
    }
    with_current(|t| {
        if t.used == 0 {
            return;
        }
        let base = slot as u32 * MAX_TS;
        unsafe {
            list.ResolveQueryData(
                &t.heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                base,
                t.used,
                &t.readback.resource,
                base as u64 * 8,
            )
        };
        let regions = std::mem::take(&mut t.regions);
        t.pending[slot] = Some(regions);
    });
}

/// Print the final table (session exit). The window it closes is the tail
/// since the last periodic report, so a session killed on a timer still banks
/// every completed window — losing at most the partial tail.
pub fn report() {
    if !on() {
        return;
    }
    // Every device, not just the one that happened to record last: at session
    // exit both cards' tables are wanted, and under --dual-gpu comparing them
    // is the whole point.
    let mut guard = TIMERS.lock().unwrap();
    let multi = guard.len() > 1;
    for t in guard.iter_mut() {
        t.report(multi);
    }
}

/// RAII timing region. Inert (and allocation-free) when timing is off.
pub struct TimeScope(Option<(ID3D12GraphicsCommandList, usize, u32)>);

impl Drop for TimeScope {
    fn drop(&mut self) {
        let Some((list, _idx, end_q)) = self.0.take() else { return };
        // Closes against the SAME device that opened it: a scope cannot outlive
        // its frame, and frames do not nest across devices (see `CURRENT`).
        with_current(|t| {
            // The region's (relative) end index was set when it opened; the
            // absolute one rode the guard here purely for EndQuery.
            unsafe { list.EndQuery(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, end_q) };
            t.depth = t.depth.saturating_sub(1);
        });
    }
}

/// Open a timing region on `list`. `name` is only materialized when timing is
/// on, so the marker sites pay nothing in a default session.
pub fn scope(list: &ID3D12GraphicsCommandList, name: impl FnOnce() -> String) -> TimeScope {
    if !on() {
        return TimeScope(None);
    }
    with_current(|t| {
        let slot_base = t.cur_slot as u32 * MAX_TS;
        if t.used + 2 > MAX_TS {
            t.dropped += 1;
            return TimeScope(None);
        }
        let rel = t.used; // slot-relative; collect() maps one slot's window
        let begin_q = slot_base + rel;
        let end_q = begin_q + 1;
        t.used += 2;
        let depth = t.depth;
        t.depth += 1;
        unsafe { list.EndQuery(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, begin_q) };
        let idx = t.regions.len();
        t.regions.push(Region { name: name(), depth, begin: rel, end: rel + 1 });
        TimeScope(Some((list.clone(), idx, end_q)))
    })
    .unwrap_or(TimeScope(None))
}
