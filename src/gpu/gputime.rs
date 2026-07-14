//! D3D12 timestamp-query scopes — the GPU-side twin of the PIX markers.
//!
//! `--pix-markers` names the passes for an external capture tool; this names
//! them for the app itself, so a session can print its own per-pass GPU
//! breakdown with no SDK, no DLL, and no vendor tooling installed. That
//! vendor-neutrality is the whole point: the same numbers come back from an
//! NVIDIA and an AMD adapter, so a *per-pass* AMD-vs-NVIDIA diff is possible,
//! which is what turns "AMD is slower" into "AMD is slower *in this kernel*".
//!
//! Scopes bracket exactly what the PIX scopes bracket (wavefront, level {d},
//! leaf+sky, hemi, compose, resolve, reference). Timestamps are written with
//! `EndQuery` at both ends (D3D12 timestamps have no Begin), resolved once at
//! the end of the list, and read back after the caller's own fence wait —
//! so this never adds a sync point of its own.
//!
//! Opt-in (`--gpu-timing`) and thread-local: when off, every call is an
//! `Option` check on a thread-local and the recorded command list is
//! byte-identical to an unprofiled session.
//!
//! Caveat worth knowing when reading the output: a timestamp scope measures
//! wall-clock on the queue between two markers, so *nested* scopes inside a
//! pass that overlaps itself (our levels are barrier-separated, so they do
//! not) would double-count. The levels are serialized by UAV barriers, which
//! is exactly why their spans sum to the parent.

use std::cell::RefCell;
use windows::Win32::Graphics::Direct3D12::*;

use super::d3d12;

type Result<T> = std::result::Result<T, String>;

/// Query slots. Two per scope; ~20 scopes/frame at depth_full 8, so this is
/// ~6x headroom. Overflow drops the scope (counted) rather than corrupting.
const SLOTS: u32 = 256;

struct Span {
    name: String,
    depth: u32,
    begin: u32,
    end: u32,
}

struct Timer {
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    /// Ticks per second on this queue (GetTimestampFrequency).
    freq: u64,
    next: u32,
    depth: u32,
    spans: Vec<Span>,
    dropped: u32,
}

thread_local! {
    static TIMER: RefCell<Option<Timer>> = const { RefCell::new(None) };
}

/// Arm timestamp profiling on this thread. Call once, after the queue exists;
/// `enabled` false leaves every call inert.
pub fn init(device: &ID3D12Device, queue: &ID3D12CommandQueue, enabled: bool) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    let desc = D3D12_QUERY_HEAP_DESC {
        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
        Count: SLOTS,
        NodeMask: 0,
    };
    let mut heap: Option<ID3D12QueryHeap> = None;
    unsafe { device.CreateQueryHeap(&desc, &mut heap) }
        .map_err(|e| format!("CreateQueryHeap(TIMESTAMP): {e}"))?;
    let heap = heap.ok_or("CreateQueryHeap returned null")?;
    let readback = d3d12::ReadbackBuffer::new(device, SLOTS as usize * 8)?.resource;
    let freq = unsafe { queue.GetTimestampFrequency() }
        .map_err(|e| format!("GetTimestampFrequency: {e}"))?;
    TIMER.with(|t| {
        *t.borrow_mut() =
            Some(Timer { heap, readback, freq, next: 0, depth: 0, spans: Vec::new(), dropped: 0 });
    });
    eprintln!("gpu-timing: on (timestamp freq {freq} Hz)");
    Ok(())
}

pub fn enabled() -> bool {
    TIMER.with(|t| t.borrow().is_some())
}

/// Drop any spans recorded but never collected, and rewind the slot cursor.
/// Call at the START of every list you intend to `resolve` — a frame that
/// opens scopes without resolving (a gate frame, a warm-up frame) otherwise
/// leaves its spans behind to be mis-attributed to the next collector, and
/// walks the slot cursor into the SLOTS ceiling where every later scope is
/// silently dropped.
pub fn reset() {
    TIMER.with(|t| {
        let mut b = t.borrow_mut();
        let Some(tm) = b.as_mut() else { return };
        tm.next = 0;
        tm.depth = 0;
        tm.spans.clear();
    });
}

/// RAII scope: writes the begin timestamp now and the end timestamp on drop.
/// Holds a COM clone of the list (a refcount bump), not a borrow — the same
/// reason `pix::PixScope` does.
pub struct TimeScope(Option<(ID3D12GraphicsCommandList, usize)>);

impl Drop for TimeScope {
    fn drop(&mut self) {
        let Some((list, idx)) = self.0.take() else { return };
        TIMER.with(|t| {
            let mut b = t.borrow_mut();
            let Some(tm) = b.as_mut() else { return };
            let slot = tm.next;
            tm.next += 1;
            tm.depth -= 1;
            tm.spans[idx].end = slot;
            unsafe { list.EndQuery(&tm.heap, D3D12_QUERY_TYPE_TIMESTAMP, slot) };
        });
    }
}

pub fn scope(list: &ID3D12GraphicsCommandList, name: impl Into<String>) -> TimeScope {
    TIMER.with(|t| {
        let mut b = t.borrow_mut();
        let Some(tm) = b.as_mut() else { return TimeScope(None) };
        // Two slots per scope; refuse to start one we cannot close.
        if tm.next + 2 > SLOTS {
            tm.dropped += 1;
            return TimeScope(None);
        }
        let slot = tm.next;
        tm.next += 1;
        let idx = tm.spans.len();
        tm.spans.push(Span { name: name.into(), depth: tm.depth, begin: slot, end: slot });
        tm.depth += 1;
        unsafe { list.EndQuery(&tm.heap, D3D12_QUERY_TYPE_TIMESTAMP, slot) };
        TimeScope(Some((list.clone(), idx)))
    })
}

/// Resolve the frame's queries into the readback buffer. Call at the END of
/// the command list, after every scope has closed and before Close().
pub fn resolve(list: &ID3D12GraphicsCommandList) {
    TIMER.with(|t| {
        let b = t.borrow();
        let Some(tm) = b.as_ref() else { return };
        if tm.next == 0 {
            return;
        }
        unsafe {
            list.ResolveQueryData(
                &tm.heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                0,
                tm.next,
                &tm.readback,
                0,
            )
        };
    });
}

/// One pass's measured time, in list order.
pub struct PassTime {
    pub name: String,
    pub depth: u32,
    pub ms: f64,
}

/// Read the resolved timestamps back and clear the frame's scopes. Call only
/// after the submission that recorded them has been waited on — this maps a
/// readback buffer and does no synchronization of its own.
pub fn collect() -> Vec<PassTime> {
    TIMER.with(|t| {
        let mut b = t.borrow_mut();
        let Some(tm) = b.as_mut() else { return Vec::new() };
        let n = tm.next as usize;
        let mut out = Vec::new();
        if n > 0 {
            let mut ptr = std::ptr::null_mut();
            if unsafe { tm.readback.Map(0, None, Some(&mut ptr)) }.is_ok() {
                let ticks = unsafe { std::slice::from_raw_parts(ptr as *const u64, n) };
                let per_ms = tm.freq as f64 / 1000.0;
                for s in &tm.spans {
                    let (a, z) = (ticks[s.begin as usize], ticks[s.end as usize]);
                    // A timestamp pair can invert only if the queue was
                    // reset under us; report 0 rather than a wild negative.
                    let d = z.saturating_sub(a) as f64 / per_ms;
                    out.push(PassTime { name: s.name.clone(), depth: s.depth, ms: d });
                }
                unsafe { tm.readback.Unmap(0, None) };
            }
        }
        tm.next = 0;
        tm.depth = 0;
        tm.spans.clear();
        out
    })
}

/// Median-aggregate N frames' worth of `collect()` output, keyed by
/// (name, depth) in first-seen order. The median is deliberate: a GPU pass's
/// wall time has a long right tail (clocks ramping, other work on the queue),
/// and a mean of 60 frames reports the tail, not the pass.
pub fn median_passes(frames: &[Vec<PassTime>]) -> Vec<(String, u32, f64)> {
    let mut order: Vec<(String, u32)> = Vec::new();
    let mut samples: Vec<Vec<f64>> = Vec::new();
    for f in frames {
        for p in f {
            let key = (p.name.clone(), p.depth);
            let i = match order.iter().position(|k| *k == key) {
                Some(i) => i,
                None => {
                    order.push(key);
                    samples.push(Vec::new());
                    order.len() - 1
                }
            };
            samples[i].push(p.ms);
        }
    }
    order
        .into_iter()
        .zip(samples)
        .map(|((name, depth), mut v)| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = if v.is_empty() { 0.0 } else { v[v.len() / 2] };
            (name, depth, med)
        })
        .collect()
}
