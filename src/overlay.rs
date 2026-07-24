use glam::Vec3A;

pub const KIND_LEAF: u32 = 0;
pub const KIND_SKY: u32 = 1;
pub const KIND_BLOCKED: u32 = 2;
/// Depth-cap sparse fill, flooded pixel: the tile reached the frame-budget
/// depth cap and this pixel shows its cell's point sample (dynamic
/// resolution). The sample pixels themselves are KIND_LEAF.
pub const KIND_COARSE: u32 = 3;

pub fn pack_info(depth: u32, kind: u32) -> u32 {
    (depth & 0xff) | (kind << 8)
}

pub fn info_depth(info: u32) -> u32 {
    info & 0xff
}

pub fn info_kind(info: u32) -> u32 {
    (info >> 8) & 0xff
}

/// Blue → cyan → green → yellow → red over quadtree depth 0..=8.
pub fn heat(depth: u32) -> Vec3A {
    let t = (depth as f32 / 8.0).clamp(0.0, 1.0);
    let stops = [
        Vec3A::new(0.10, 0.15, 0.85),
        Vec3A::new(0.05, 0.75, 0.85),
        Vec3A::new(0.15, 0.80, 0.20),
        Vec3A::new(0.95, 0.85, 0.10),
        Vec3A::new(0.90, 0.15, 0.10),
    ];
    let f = t * 4.0;
    let i = (f as usize).min(3);
    stops[i].lerp(stops[i + 1], f - i as f32)
}

/// (tint color, blend alpha) for a pixel's quadtree info.
pub fn tint(info: u32) -> (Vec3A, f32) {
    match info_kind(info) {
        KIND_SKY => (Vec3A::new(0.2, 0.4, 1.0), 0.12),
        KIND_BLOCKED => (Vec3A::new(1.0, 0.1, 0.1), 0.30),
        KIND_COARSE => (Vec3A::new(1.0, 0.55, 0.05), 0.35),
        _ => (heat(info_depth(info)), 0.35),
    }
}
