//! The HUD's CPU→GPU wire: `DirtyRect`, `HudFrame`, and the ONE packer that
//! turns a premultiplied-RGBA8 buffer plus a dirty-rect list into the bytes a
//! backend uploads. Consumed by `gpu/hud.rs` (D3D12) and `vk/hud.rs` (Vulkan);
//! produced by `hud::Hud::raster` (the session HUD and the loading page share
//! it) and by `--check-vk` V21, which builds a SYNTHETIC frame with no Slint
//! in the process at all.
//!
//! Lives under `gfx/` and not in `hud/` because it is exactly the shape the
//! module header describes — vocabulary two backends share — and because
//! `hud/` is cfg'd to the platforms with a window (it imports `slint` and
//! `sdl3`) while this must compile wherever `vk::` does, macOS included. A
//! gate that builds its fixture from these types is then a gate that needs no
//! font, no `slint::platform::set_platform` (once per process) and no main
//! thread, which is what lets it run headless on llvmpipe in CI.
//!
//! THE LAYOUT IS THE CONTRACT: each rect's rows tightly packed (`w*4` bytes
//! per row, no pitch), rects concatenated in list order, rects clamped to the
//! buffer BEFORE packing so the byte count and the rect list can never
//! disagree about a row's length. Both backends compute their per-rect source
//! offsets from this rule and nothing else.

/// One changed region of the HUD buffer, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// A frame's changed pixels: each rect's rows tightly packed (`w*4` bytes per
/// row), concatenated in rect order. The backend hud modules consume this
/// layout; `pack_rects` is the one writer.
pub struct HudFrame {
    pub rects: Vec<DirtyRect>,
    pub bytes: Vec<u8>,
}

/// Clamp every rect to a `w`×`h` buffer and drop the empties. Idempotent.
pub fn clamp_rects(rects: &mut Vec<DirtyRect>, w: u32, h: u32) {
    for r in rects.iter_mut() {
        r.x = r.x.min(w);
        r.y = r.y.min(h);
        r.w = r.w.min(w - r.x);
        r.h = r.h.min(h - r.y);
    }
    rects.retain(|r| r.w > 0 && r.h > 0);
}

/// Pack `rects` out of a `w`×`h` RGBA8 buffer (`src.len() == w*h*4`) into a
/// `HudFrame`, or `None` when nothing survives clamping. The ONE copy of the
/// layout rule in the module header.
pub fn pack_rects(src: &[u8], w: u32, h: u32, mut rects: Vec<DirtyRect>) -> Option<HudFrame> {
    debug_assert_eq!(src.len(), (w as usize) * (h as usize) * 4);
    clamp_rects(&mut rects, w, h);
    if rects.is_empty() {
        return None;
    }
    let bytes_len: usize = rects.iter().map(|r| (r.w * r.h * 4) as usize).sum();
    let mut bytes = Vec::with_capacity(bytes_len);
    for r in &rects {
        for row in r.y..r.y + r.h {
            let o = (row as usize * w as usize + r.x as usize) * 4;
            bytes.extend_from_slice(&src[o..o + r.w as usize * 4]);
        }
    }
    debug_assert_eq!(bytes.len(), bytes_len);
    Some(HudFrame { rects, bytes })
}

/// The packer's contract, gated without Slint: byte count, row content by
/// construction, edge-touching exactness, clamp-then-drop, all-empty → None,
/// rect ORDER (a misordered reference must not equal the output — the tooth).
pub fn self_test() -> Result<(), String> {
    let (w, h) = (8u32, 4u32);
    // A hashed buffer: every texel's four bytes differ from every other's,
    // so a one-texel slip anywhere shows up as a byte mismatch.
    let src: Vec<u8> = (0..w * h * 4)
        .map(|i| ((i.wrapping_mul(2654435761) >> 13) & 0xff) as u8)
        .collect();
    let texel = |x: u32, y: u32| -> &[u8] {
        let o = ((y * w + x) * 4) as usize;
        &src[o..o + 4]
    };

    // 1. One interior rect: exact byte count, exact row content.
    let r0 = DirtyRect { x: 2, y: 1, w: 3, h: 2 };
    let f = pack_rects(&src, w, h, vec![r0]).ok_or("interior rect packed to None")?;
    if f.rects != vec![r0] {
        return Err(format!("interior rect list changed: {:?}", f.rects));
    }
    if f.bytes.len() != 3 * 2 * 4 {
        return Err(format!("interior rect bytes {} != 24", f.bytes.len()));
    }
    let mut want = Vec::new();
    for y in 1..3 {
        for x in 2..5 {
            want.extend_from_slice(texel(x, y));
        }
    }
    if f.bytes != want {
        return Err("interior rect row content mismatch".into());
    }

    // 2. Edge-touching (x+w == W, y+h == H) packs exactly, unclamped.
    let r1 = DirtyRect { x: 5, y: 2, w: 3, h: 2 };
    let f = pack_rects(&src, w, h, vec![r1]).ok_or("edge rect packed to None")?;
    if f.rects != vec![r1] || f.bytes.len() != 24 {
        return Err(format!("edge rect changed: {:?} / {} bytes", f.rects, f.bytes.len()));
    }
    if &f.bytes[20..24] != texel(7, 3) {
        return Err("edge rect's last texel is not (W-1,H-1)".into());
    }

    // 3. Over-range clamps (then packs what is left); fully-out drops.
    let f = pack_rects(&src, w, h, vec![DirtyRect { x: 7, y: 3, w: 10, h: 10 }])
        .ok_or("over-range rect packed to None")?;
    if f.rects != vec![DirtyRect { x: 7, y: 3, w: 1, h: 1 }] || f.bytes != texel(7, 3) {
        return Err(format!("over-range rect did not clamp to (7,3,1,1): {:?}", f.rects));
    }
    if pack_rects(&src, w, h, vec![DirtyRect { x: 8, y: 4, w: 2, h: 2 }]).is_some() {
        return Err("fully-out rect survived clamping".into());
    }
    if pack_rects(&src, w, h, vec![DirtyRect { x: 1, y: 1, w: 0, h: 3 }]).is_some() {
        return Err("zero-width rect survived clamping".into());
    }
    if pack_rects(&src, w, h, Vec::new()).is_some() {
        return Err("empty rect list packed to Some".into());
    }

    // 4. Two rects pack in LIST order; the swapped reference differs (teeth).
    let ab = pack_rects(&src, w, h, vec![r0, r1]).ok_or("two rects packed to None")?;
    let ba = pack_rects(&src, w, h, vec![r1, r0]).ok_or("two rects (swapped) packed to None")?;
    if ab.bytes.len() != 48 || ba.bytes.len() != 48 {
        return Err("two-rect byte count != 48".into());
    }
    let a = pack_rects(&src, w, h, vec![r0]).unwrap().bytes;
    let b = pack_rects(&src, w, h, vec![r1]).unwrap().bytes;
    if ab.bytes != [a.as_slice(), b.as_slice()].concat() {
        return Err("two-rect pack is not the concatenation in list order".into());
    }
    if ab.bytes == ba.bytes {
        return Err("swapped rect order packed identically — the order tooth is vacuous".into());
    }
    Ok(())
}
