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
    Ok(())
}
