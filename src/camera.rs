use crate::frustum::TileFrustum;
use glam::Vec3A;

#[derive(Clone, Copy)]
pub struct Camera {
    pub pos: Vec3A,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

impl Camera {
    pub fn look_at(pos: Vec3A, target: Vec3A, fov_y: f32) -> Self {
        let d = (target - pos).normalize();
        Camera {
            pos,
            yaw: d.z.atan2(d.x),
            pitch: d.y.asin(),
            fov_y,
        }
    }

    pub fn forward(&self) -> Vec3A {
        Vec3A::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        )
    }

    /// Per-frame derived basis for a given render resolution.
    pub fn basis(&self, w: usize, h: usize) -> CamBasis {
        let forward = self.forward();
        let right = forward.cross(Vec3A::Y).normalize();
        let up = right.cross(forward);
        let tan_half = (self.fov_y * 0.5).tan();
        CamBasis {
            origin: self.pos,
            forward,
            right: right * (tan_half * w as f32 / h as f32),
            up: up * tan_half,
            inv_w: 1.0 / w as f32,
            inv_h: 1.0 / h as f32,
        }
    }
}

// PartialEq: bitwise field equality — an unchanged `Camera` re-derives a
// bit-identical basis (pure f32 code, no NaN possible), and the temporal cache
// uses `==` to detect the static-camera case exactly.
#[derive(Clone, Copy, PartialEq)]
pub struct CamBasis {
    pub origin: Vec3A,
    forward: Vec3A,
    right: Vec3A, // pre-scaled by tan(fov/2) * aspect
    up: Vec3A,    // pre-scaled by tan(fov/2)
    inv_w: f32,
    inv_h: f32,
}

impl CamBasis {
    /// Normalized ray direction through the continuous image point (fx, fy),
    /// fx ∈ [0, w], fy ∈ [0, h] (pixel-grid coordinates, y down).
    #[inline(always)]
    pub fn ray_dir(&self, fx: f32, fy: f32) -> Vec3A {
        let ndx = fx * self.inv_w * 2.0 - 1.0;
        let ndy = 1.0 - fy * self.inv_h * 2.0;
        (self.forward + self.right * ndx + self.up * ndy).normalize()
    }

    /// Inverse of `ray_dir`: the continuous image point a world direction
    /// passes through, or `None` if it points at or behind the image plane.
    /// `forward`, `right`, `up` are mutually orthogonal (up = right × forward
    /// before scaling), so the NDC components separate by projection.
    pub fn project(&self, d: Vec3A) -> Option<(f32, f32)> {
        let df = d.dot(self.forward);
        if df <= 0.0 {
            return None;
        }
        let ndx = d.dot(self.right) / (self.right.length_squared() * df);
        let ndy = d.dot(self.up) / (self.up.length_squared() * df);
        Some(((ndx + 1.0) * 0.5 / self.inv_w, (1.0 - ndy) * 0.5 / self.inv_h))
    }

    /// Frustum through the tile's continuous pixel-grid edges — pixel
    /// *footprints*, not centers, so jittered samples stay inside their tile.
    pub fn tile_frustum(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> TileFrustum {
        let (x0, y0, x1, y1) = (x0 as f32, y0 as f32, x1 as f32, y1 as f32);
        TileFrustum::new(
            self.origin,
            [
                self.ray_dir(x0, y0),
                self.ray_dir(x1, y0),
                self.ray_dir(x1, y1),
                self.ray_dir(x0, y1),
            ],
        )
    }
}
