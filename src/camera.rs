use crate::frustum::TileFrustum;
use glam::Vec3A;

// PartialEq: bitwise field equality, same rationale as CamBasis below — no
// camera math produces NaN, and successive `FlyCam::snapshot`s of an
// unwritten shared camera are bit-identical copies, so `!=` detects "the
// integrator thread wrote something" exactly (the `moved` signal).
#[derive(Clone, Copy, PartialEq)]
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
        let up = up * tan_half;
        let inv_h = 1.0 / h as f32;
        CamBasis {
            origin: self.pos,
            forward,
            right: right * (tan_half * w as f32 / h as f32),
            up,
            inv_w: 1.0 / w as f32,
            inv_h,
            // Derived once per frame, not per pixel: `shade_traced` builds the
            // primary Cone for every shaded pixel and this is a sqrt.
            pixel_cone: 2.0 * up.length() * inv_h,
        }
    }
}

// ---------------------------------------------------------------------------
// Keyboard flight ease (the flycam integrator's inertia)
//
// Lives here rather than in flycam.rs because the contract these functions
// exist to satisfy is documented at the top of THIS file: `Camera: PartialEq`
// is the "the integrator thread wrote something" signal, and an ease that
// never reaches exact rest would keep writing `cam.pos` forever after key
// release, silently defeating plain accumulation, structure replay, and every
// upscaler history on an idle session.
//
// It USED to be here for a second reason as well — `mod flycam` was
// `#[cfg(windows)]`, so a gate hosted there would not have run off Windows.
// B6b rung 2 retired that: the integrator is platform-free now and only its
// two sources are cfg'd, so `flycam::self_test` runs everywhere `--check`
// does. The reason above still stands on its own; the cfg one no longer
// exists, and leaving it written down would send the next reader to look for
// a constraint that is not there.
//
// The analog stick path does NOT go through here: deflection is already the
// throttle there, and `stick()`'s magnitude^2 shaping is its response curve.
// ---------------------------------------------------------------------------

/// Seconds for keyboard flight to ease between rest and full speed — and, by
/// the same ramp, to coast to a stop after the keys are released. 0 disables
/// the ease entirely (the hard-step A/B arm; `--no-move-ease`).
pub const MOVE_EASE_S: f32 = 0.18;

/// Advance the keyboard flight ramp one tick toward `target`.
///
/// `v` and `target` are CAMERA-RELATIVE (x = right, y = up, z = forward) with
/// |target| ∈ {0, 1} — the same basis the analog stick path already uses, so
/// the ramp can be advanced before the pose mutex is taken.
///
/// The step is a LINEAR slew of fixed length `dt / ease_s` toward `target`,
/// **snapping to `target` exactly** when it is within one step. Two properties
/// follow, and both are load-bearing:
///
/// * **Both rest states are reached EXACTLY.** Releasing the keys drives `v`
///   to a bitwise `Vec3A::ZERO` in bounded time, which is what lets the
///   integrator's idle early-out resume and leave the shared state
///   bit-untouched. An exponential smoother (`v += (t - v) * k`) is the
///   tempting wrong answer here: it is asymptotic, so it never stops writing.
/// * **|v| ≤ 1 by construction.** A point stepped along the segment toward a
///   target of length ≤ 1 never leaves the unit ball, so no clamp is needed
///   and full deflection cannot overshoot.
///
/// Because it is one vector rather than a scalar plus a latched direction, a
/// direction change slews: W→S passes through the origin (a momentary stop —
/// the inertia), W→D arcs and dips slightly in speed.
pub fn move_ease(v: Vec3A, target: Vec3A, dt: f32, ease_s: f32) -> Vec3A {
    if !(ease_s > 0.0) {
        return target; // off arm (also the NaN arm — `!(x > 0)` rejects it)
    }
    let d = target - v;
    let step = dt / ease_s;
    if d.length() <= step {
        target
    } else {
        v + d.normalize() * step
    }
}

/// Speed scalar for a ramp position: `smoothstep(|v|)`, the ease-in/ease-out
/// shaping. A SCALAR rather than a shaped vector on purpose — the caller keeps
/// normalizing its direction in WORLD space, so full deflection stays exactly
/// the full flight speed even where the camera basis is not orthonormal (`f`
/// carries a Y component when pitched, so a forward+vertical combination has
/// world length > 1 and coefficient-space normalization would run it fast).
///
/// Exact at both endpoints, which is what makes the ease invisible at rest and
/// at full speed: exactly 0.0 at rest and exactly 1.0 at |v| = 1
/// (`1*1*(3-2)`). `move_ease` already keeps |v| <= 1, so the `.min(1.0)` is a
/// defensive clamp on an unreachable input rather than part of the shaping —
/// without it the polynomial turns back DOWN past 1 and would read as the
/// camera slowing at full speed.
pub fn move_scale(v: Vec3A) -> f32 {
    let t = v.length().min(1.0);
    if t <= 0.0 {
        return 0.0;
    }
    t * t * (3.0 - 2.0 * t)
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
    // A pure function of `up`/`inv_h` (see `pixel_cone`), so including it in
    // the derived PartialEq changes no basis-equality decision.
    pixel_cone: f32,
}

impl CamBasis {
    /// Unit forward axis — needed by the DLSS G-buffer capture to convert
    /// Euclidean hit t to linear view-space Z (t * dot(dir, forward)).
    #[inline(always)]
    pub fn forward(&self) -> Vec3A {
        self.forward
    }

    /// Raw basis fields for the GPU tracer's frame constants: (origin,
    /// forward, right, up, inv_w, inv_h). The HLSL ray_dir in
    /// shaders/trace_common.hlsli mirrors `ray_dir` above term for term —
    /// these are its inputs, pre-scaled exactly as the CPU uses them.
    pub fn gpu_fields(&self) -> (Vec3A, Vec3A, Vec3A, Vec3A, f32, f32) {
        (self.origin, self.forward, self.right, self.up, self.inv_w, self.inv_h)
    }

    /// Angular size of one pixel at the screen center (radians) — the
    /// primary ray-cone spread for texture LOD. `up` is pre-scaled by
    /// tan(fov/2) against a unit forward, so one pixel step moves the
    /// direction by 2·|up|·inv_h. Computed once in `Camera::basis` and stored;
    /// the GPU tracer receives this exact value through FrameCb::pixel_cone
    /// (single source, both samplers agree).
    #[inline(always)]
    pub fn pixel_cone(&self) -> f32 {
        self.pixel_cone
    }

    /// Normalized ray direction through the continuous image point (fx, fy),
    /// fx ∈ [0, w], fy ∈ [0, h] (pixel-grid coordinates, y down).
    #[inline(always)]
    pub fn ray_dir(&self, fx: f32, fy: f32) -> Vec3A {
        self.ray_dir_unnorm(fx, fy).normalize()
    }

    /// `ray_dir` without the normalization, for consumers whose math is
    /// scale-invariant in the direction: `project` is a pure ratio of dots,
    /// and `origin + d·(z / d·forward)` cancels the length. Skips the
    /// per-pixel sqrt — the tracer itself must keep `ray_dir` (distance ==
    /// ray parameter t requires unit directions; see CLAUDE.md invariants).
    #[inline(always)]
    pub fn ray_dir_unnorm(&self, fx: f32, fy: f32) -> Vec3A {
        let ndx = fx * self.inv_w * 2.0 - 1.0;
        let ndy = 1.0 - fy * self.inv_h * 2.0;
        self.forward + self.right * ndx + self.up * ndy
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

/// Pure gate for the keyboard flight ease (`--check`). Pure math, no window,
/// no Win32 — which is why it lives beside the `Camera: PartialEq` contract it
/// defends rather than in `flycam`, whose own `self_test` gates the integrator
/// that calls these.
///
/// The arm that matters most is (1): the flycam integrator's idle early-out,
/// and therefore accumulation / structure replay / every upscaler history on a
/// still camera, rests on the ramp reaching a BITWISE zero. Its teeth are the
/// asymptotic smoother that would break it, run through the same probe.
pub fn self_test() -> Result<(), String> {
    let ease = MOVE_EASE_S;
    let fwd = Vec3A::new(0.0, 0.0, 1.0);
    let right = Vec3A::new(1.0, 0.0, 0.0);
    let up = Vec3A::new(0.0, 1.0, 0.0);
    let diag = Vec3A::new(1.0, 0.0, 1.0).normalize();
    let rates = [1.0f32 / 500.0, 1.0 / 240.0, 1.0 / 60.0, 1.0 / 30.0];

    // (1) EXACT REST, from anywhere, at any tick rate.
    for &dt in &rates {
        for &start in &[fwd, right, diag, -fwd, Vec3A::new(0.0, 0.0, 1e-7)] {
            let ticks = (ease / dt).ceil() as usize + 2;
            let mut v = start;
            for _ in 0..ticks {
                v = move_ease(v, Vec3A::ZERO, dt, ease);
            }
            if v != Vec3A::ZERO {
                return Err(format!(
                    "ease did not reach EXACT rest (dt {dt}, start {start:?}): after {ticks} ticks \
                     v = {v:?} — the integrator's idle early-out never re-arms, so an idle session \
                     re-writes cam.pos forever and nothing converges"
                ));
            }
        }
    }
    if move_scale(Vec3A::ZERO) != 0.0 {
        return Err("move_scale(ZERO) must be exactly 0.0".into());
    }
    // TEETH: the asymptotic smoother this contract exists to exclude must
    // provably FAIL the same probe — otherwise arm (1) passes on anything that
    // merely gets small.
    {
        let dt = 1.0 / 500.0;
        let k = dt / ease;
        let mut v = fwd;
        for _ in 0..((ease / dt).ceil() as usize + 2) {
            v += (Vec3A::ZERO - v) * k;
        }
        if v == Vec3A::ZERO {
            return Err(format!(
                "teeth: an exponential smoother reached bitwise zero ({v:?}) — the exact-rest arm \
                 cannot distinguish it from a snapping slew and proves nothing"
            ));
        }
    }

    // (2) EXACT SATURATION: full speed must be exactly full speed.
    for &dt in &rates {
        for &tgt in &[fwd, right, up, diag] {
            let ticks = (ease / dt).ceil() as usize + 2;
            let mut v = Vec3A::ZERO;
            for _ in 0..ticks {
                v = move_ease(v, tgt, dt, ease);
            }
            if v != tgt {
                return Err(format!(
                    "ease did not arrive EXACTLY on target (dt {dt}, target {tgt:?}): v = {v:?}"
                ));
            }
        }
    }
    for &v in &[fwd, right, up, fwd * 2.0] {
        if move_scale(v) != 1.0 {
            return Err(format!(
                "move_scale must be exactly 1.0 at (or above) full deflection: {v:?} -> {}",
                move_scale(v)
            ));
        }
    }

    // (3) BOUND: |ramp| <= 1 through arbitrary target switching — the property
    // that makes move_scale's `.min(1.0)` unreachable and full speed
    // unexceedable.
    // (Slack, not bitwise: a diagonal `normalize()` is unit only to ~1 ulp.)
    {
        let dirs = [
            Vec3A::ZERO,
            fwd,
            -fwd,
            right,
            -right,
            up,
            -up,
            diag,
            Vec3A::new(-1.0, 1.0, 1.0).normalize(),
        ];
        let mut v = Vec3A::ZERO;
        let mut h: u32 = 0x9e37_79b9;
        for _ in 0..20_000 {
            h = h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let tgt = dirs[(h >> 16) as usize % dirs.len()];
            let dt = rates[((h >> 8) & 3) as usize];
            v = move_ease(v, tgt, dt, ease);
            let len = v.length();
            if !(len <= 1.0 + 1e-6) || !v.is_finite() {
                return Err(format!("ramp left the unit ball: |v| = {len} at {v:?}"));
            }
            let s = move_scale(v);
            if !(0.0..=1.0).contains(&s) {
                return Err(format!("move_scale left [0, 1]: {s} at {v:?}"));
            }
        }
    }

    // (4) SHAPE: smoothstep in the ramp magnitude, monotone, exact endpoints.
    {
        let mut prev = -1.0f32;
        for i in 0..=1000 {
            let t = i as f32 / 1000.0;
            let m = move_scale(Vec3A::new(0.0, 0.0, t));
            let want = t * t * (3.0 - 2.0 * t);
            if (m - want).abs() > 1e-6 {
                return Err(format!("move_scale is not smoothstep at t {t}: {m} vs {want}"));
            }
            if m < prev - 1e-7 {
                return Err(format!("move_scale is not monotone at t {t}"));
            }
            prev = m;
        }
    }

    // (5) dt-INVARIANCE — the "deterministic" half of the claim. The ease takes
    // MOVE_EASE_S of WALL CLOCK in both directions whatever the tick rate,
    // because the integrator steps by measured dt.
    for &dt in &rates {
        let mut v = Vec3A::ZERO;
        let mut n_in = 0;
        while v != fwd && n_in < 1_000_000 {
            v = move_ease(v, fwd, dt, ease);
            n_in += 1;
        }
        let mut n_out = 0;
        while v != Vec3A::ZERO && n_out < 1_000_000 {
            v = move_ease(v, Vec3A::ZERO, dt, ease);
            n_out += 1;
        }
        let (t_in, t_out) = (n_in as f32 * dt, n_out as f32 * dt);
        if (t_in - ease).abs() > dt + 1e-6 || (t_out - ease).abs() > dt + 1e-6 {
            return Err(format!(
                "ease duration depends on the tick rate (dt {dt}): in {t_in} s, out {t_out} s, \
                 want {ease} s within one tick"
            ));
        }
    }
    // ...and so does the DISPLACEMENT over a hold. Bounded rather than exact:
    // the sum is a left-Riemann quadrature of the smoothstep, so it carries an
    // O(dt) transient error (the two transients partially cancel, one rising
    // and one falling). A real dt-dependence bug is order-1, not percent.
    {
        let travel = |dt: f32| {
            let mut v = Vec3A::ZERO;
            let mut d = 0.0f32;
            let hold = (1.0 / dt).round() as usize;
            for i in 0..hold * 3 {
                let tgt = if i < hold { fwd } else { Vec3A::ZERO };
                v = move_ease(v, tgt, dt, ease);
                d += move_scale(v) * dt;
            }
            d
        };
        let base = travel(rates[0]);
        for &dt in &rates[1..] {
            let got = travel(dt);
            if !(((got - base) / base).abs() < 0.05) {
                return Err(format!(
                    "displacement over a 1 s hold depends on the tick rate: {got} at dt {dt} vs \
                     {base} at dt {}",
                    rates[0]
                ));
            }
        }
    }

    // (6) OFF ARM: ease_s <= 0 (and NaN) is the hard step, bitwise.
    for &tgt in &[Vec3A::ZERO, fwd, diag] {
        for &v0 in &[Vec3A::ZERO, fwd, -fwd, diag] {
            for &e in &[0.0, -1.0, f32::NAN] {
                if move_ease(v0, tgt, 1.0 / 500.0, e) != tgt {
                    return Err(format!("move_ease with ease_s {e} must return the target verbatim"));
                }
            }
        }
    }

    // (7) REVERSAL — the inertia itself, and the whole gate's anti-vacuity: a
    // W->S flip must slew THROUGH a momentary stop rather than snapping, and
    // take ~2x the ease (down, then back up).
    {
        let dt = 1.0 / 500.0;
        let mut v = fwd;
        let (mut n, mut min_speed) = (0usize, f32::MAX);
        while v != -fwd && n < 1_000_000 {
            v = move_ease(v, -fwd, dt, ease);
            min_speed = min_speed.min(move_scale(v));
            n += 1;
        }
        if min_speed > 0.02 {
            return Err(format!(
                "a full reversal never passed through a stop (min speed {min_speed}) — the ramp is \
                 not slewing, so there is no inertia to test"
            ));
        }
        let t_rev = n as f32 * dt;
        if (t_rev - 2.0 * ease).abs() > 2.0 * dt + 1e-6 {
            return Err(format!("a full reversal took {t_rev} s, want {} s", 2.0 * ease));
        }
    }

    // (8) DETERMINISM: same inputs, bitwise same outputs.
    for &dt in &rates {
        for &tgt in &[Vec3A::ZERO, fwd, diag] {
            let a = move_ease(diag, tgt, dt, ease);
            let b = move_ease(diag, tgt, dt, ease);
            if a != b || move_scale(a) != move_scale(b) {
                return Err("move_ease/move_scale are not deterministic".into());
            }
        }
    }

    Ok(())
}
