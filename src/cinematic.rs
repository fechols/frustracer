//! `--cinematic`: the offline beauty path — stills and camera-spline video
//! sequences, rendered headlessly and deterministically.
//!
//! This module is the PURE half: the data model, the spline, the presets, and
//! the HUD composite. It touches no GPU, no window and no platform API, so it
//! is covered by `--check` (`self_test`) on every platform, DLL-free. The two
//! drivers that actually render (`run_cinematic` / `run_cinematic_gpu`) live in
//! main.rs beside `run_spin` / `run_spin_gpu`, whose frame contract they mirror.
//!
//! WHY THIS EXISTS AT ALL, given `--spin` already walks a camera path: `--spin`
//! is a benchmark and deliberately writes no pixels, and its path amplitudes are
//! benchmark-sized (<= 0.12 diag). More importantly, the interactive renderer
//! CANNOT produce these frames. Every cinematic output frame is a STATIC pose
//! that accumulates N sub-frames, which buys two things a live session cannot
//! have: proper multi-sample antialiasing without a temporal upscaler, and the
//! hemisphere bounce integrator (`Quality::fb`), which is still-frames-only by
//! construction. Cinematic mode is therefore the only path in the tree that can
//! render a moving camera WITH global illumination.
//!
//! THREE INVARIANTS, in the order they are easiest to get wrong:
//!
//! 1. **The volumetric clock is per OUTPUT frame, never per sub-frame.** Clouds
//!    and fireflies are sampled once per output frame and held fixed across
//!    every accumulated sub-frame. Keying them off the sub-frame index would
//!    average N different skies into one image and smear the clouds *within* a
//!    single frame. The clock runs in REAL SECONDS (`frame / fps`), not the
//!    benchmark's fixed `CLOUD_SPIN_DT`, so a 30 fps film has correctly-paced
//!    drift.
//! 2. **The spline interpolates POSITIONS, never angles.** Keyframes carry an
//!    eye and a look-at target; both are splined as points and the camera is
//!    rebuilt with `Camera::look_at`. `spin_path_pose` interpolates yaw/pitch
//!    offsets instead, which is safe only because its amplitudes are tiny — a
//!    lap of the island ring sweeps yaw through 2*pi, where interpolating the
//!    angle runs backwards across the wrap. Splining points has no wrap to hit.
//! 3. **Time of day is evaluated INSTANTANEOUSLY from the pose**, not eased.
//!    This deliberately differs from the interactive integrator (`flycam.rs`,
//!    which eases at `TOD_RATE` toward `attractor_hour`): easing carries
//!    hysteresis, so the same frame index would render differently depending on
//!    how the camera got there. Sampling the attractor field directly makes the
//!    hour a pure function of the pose, so a film is reproducible and a re-shoot
//!    of frame 700 matches the first take. The ring is hour-ordered, so a lap
//!    still sweeps the day monotonically.

use glam::Vec3A;

use crate::camera::Camera;
use crate::world::World;

/// The preset catalogue. A `--cinematic` argument that is not one of these is
/// treated as a path to a JSON shot list.
///
/// There is deliberately no `ab` preset: an A/B pair differs by a SESSION lever
/// (`--no-clouds`, `--heightfield`, …) which is read once at load, so a pair is
/// two runs of the same preset with different flags — not one run of a special
/// preset. That keeps the levers composable instead of duplicating each one.
pub const PRESETS: &[&str] = &["hero", "tour", "islands", "orbit", "foliage", "hud", "list"];

/// Default output frames for a sequence: 30 s at 30 fps.
pub const DEFAULT_FRAMES: u32 = 900;
pub const DEFAULT_FPS: u32 = 30;
/// Accumulated sub-frames per output frame. Stills can afford to converge; a
/// sequence pays the cost once per output frame and 32 is where the noise stops
/// reading as noise at 30 fps.
pub const DEFAULT_STILL_SAMPLES: u32 = 256;
pub const DEFAULT_SEQ_SAMPLES: u32 = 32;

/// The `foliage` preset's defaults: 4 s at 30 fps, on San Miguel's ficus.
///
/// It is the one shot in the catalogue whose SUBJECT IS MOTION, so it is the
/// one that cannot be a still — leaf sway is a per-frame displacement of real
/// geometry (`src/foliage.rs`), and a still can only ever show one pose of it,
/// indistinguishable from the rest pose. Everything else in the media set is
/// framed to hold still and converge; this one is framed to be watched.
///
/// The camera is LOCKED OFF, deliberately: a moving camera muddies the reading
/// (parallax on a static tree looks much like sway on a static camera, which is
/// exactly the ambiguity the shot exists to remove). A one-key `Shot` is how
/// that is expressed — `pose_at` returns key 0 verbatim at every `u`, with its
/// pinned hour, so no spline runs and the pose is bit-identical every frame
/// while the sway/cloud/firefly clocks advance underneath it.
pub const FOLIAGE_FRAMES: u32 = 120;
pub const FOLIAGE_ISLAND: &str = "san-miguel";

/// One authored camera key. Positions, not angles — see invariant 2.
///
/// `tod` is per-key so an authored path can pin the clock; `None` means "let
/// the world's attractor field decide" (or the session's `--tod`).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Keyframe {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    #[serde(default = "default_fov")]
    pub fov_deg: f32,
    #[serde(default)]
    pub tod: Option<f32>,
}

fn default_fov() -> f32 {
    55.0
}

impl Keyframe {
    pub fn new(eye: Vec3A, target: Vec3A) -> Keyframe {
        Keyframe {
            eye: [eye.x, eye.y, eye.z],
            target: [target.x, target.y, target.z],
            fov_deg: default_fov(),
            tod: None,
        }
    }
    pub fn with_tod(mut self, hour: f32) -> Keyframe {
        self.tod = Some(hour);
        self
    }
    /// Vertical FOV in degrees. Interiors want more than the 55-degree default:
    /// an atrium or a street is framed by how much of the enclosure fits, not
    /// by how large the subject is.
    pub fn with_fov(mut self, deg: f32) -> Keyframe {
        self.fov_deg = deg;
        self
    }
    pub fn eye_v(&self) -> Vec3A {
        Vec3A::from_array(self.eye)
    }
    pub fn target_v(&self) -> Vec3A {
        Vec3A::from_array(self.target)
    }
    pub fn camera(&self) -> Camera {
        Camera::look_at(self.eye_v(), self.target_v(), self.fov_deg.to_radians())
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ShotKind {
    Still,
    Sequence { frames: u32, fps: u32 },
}

impl ShotKind {
    pub fn frames(&self) -> u32 {
        match self {
            ShotKind::Still => 1,
            ShotKind::Sequence { frames, .. } => *frames,
        }
    }
    pub fn fps(&self) -> u32 {
        match self {
            ShotKind::Still => DEFAULT_FPS,
            ShotKind::Sequence { fps, .. } => *fps,
        }
    }
    pub fn is_sequence(&self) -> bool {
        matches!(self, ShotKind::Sequence { .. })
    }
}

/// One rendered thing: a still, or a frame sequence along `keys`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Shot {
    pub name: String,
    pub kind: ShotKind,
    pub keys: Vec<Keyframe>,
    /// Closed loops wrap the spline, so the last frame flows into the first —
    /// which is what an inline README animation wants.
    #[serde(default)]
    pub closed: bool,
    pub res: (usize, usize),
    pub samples: u32,
    /// Hemisphere bounce GI. Supported on the CPU tracer and the `--gpu`
    /// wavefront arm; the DXR pipeline has no hemi stage, so the driver
    /// switches arms with a loud line rather than silently dropping it.
    #[serde(default)]
    pub gi: bool,
    /// The quadtree debug overlay (depth heatmap + tile borders).
    #[serde(default)]
    pub overlay: bool,
    /// Composite the HUD. `Some(None)` = the flight HUD; `Some(Some(group))` =
    /// the pause menu opened on that settings page.
    #[serde(default)]
    pub hud: Option<Option<String>>,
    /// Exposure compensation in STOPS, applied to linear radiance immediately
    /// before the tonemap — the shutter/ISO control a real camera has and the
    /// renderer did not.
    ///
    /// It exists because the tonemap is anchored at a fixed paper white and the
    /// interesting parts of these scenes are ENCLOSURES: Sponza's atrium, San
    /// Miguel's patio, Bistro's street. The sun is occluded there by
    /// construction, so a physically correct render of a courtyard at 15:30 is
    /// two or three stops under a sunlit exterior — correct, and unpublishable.
    /// Brightening the sky or the tonemap instead would be a lie about the
    /// lighting; opening the aperture is what a photographer does.
    ///
    /// 0.0 is EXACTLY unchanged (`scale()` returns 1.0 and the call site
    /// branches around the copy entirely), so every existing capture and the
    /// whole `--check` surface are bit-identical.
    #[serde(default)]
    pub exposure: f32,
}

impl Shot {
    pub fn still(name: &str, key: Keyframe, res: (usize, usize), samples: u32) -> Shot {
        Shot {
            name: name.to_string(),
            kind: ShotKind::Still,
            keys: vec![key],
            closed: false,
            res,
            samples,
            gi: false,
            overlay: false,
            hud: None,
            exposure: 0.0,
        }
    }

    /// Linear multiplier for `exposure` stops. Exactly 1.0 at 0 stops — the
    /// off-state the bit-identity contract above rests on.
    pub fn exposure_scale(&self) -> f32 {
        if self.exposure == 0.0 {
            1.0
        } else {
            self.exposure.exp2()
        }
    }
}

// ---------------------------------------------------------------------------
// The spline
// ---------------------------------------------------------------------------

/// Uniform Catmull-Rom through p1..p2 at t in [0, 1). Duplicated from main.rs's
/// `catmull_rom` deliberately: this module must stay platform-free and
/// self-contained so `self_test` can pin it, and the expression is four lines.
/// The two must agree — `self_test` pins the closed-form value.
#[inline]
pub fn catmull_rom(p0: Vec3A, p1: Vec3A, p2: Vec3A, p3: Vec3A, t: f32) -> Vec3A {
    ((p1 * 2.0)
        + (p2 - p0) * t
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (t * t)
        + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * (t * t * t))
        * 0.5
}

/// Where segment/local-t land for a given normalized path position.
///
/// Open paths have `n - 1` segments and clamp at both ends; closed paths have
/// `n` (the last runs key[n-1] -> key[0]) and wrap. Factored out so `self_test`
/// can pin the seam behaviour directly.
fn segment_at(n: usize, closed: bool, u: f32) -> (isize, f32) {
    let segs = if closed { n } else { n - 1 };
    let t = u.clamp(0.0, 1.0) * segs as f32;
    if closed {
        let s = t.floor();
        (s as isize, t - s)
    } else {
        let s = t.floor().min(segs as f32 - 1.0);
        (s as isize, (t - s).clamp(0.0, 1.0))
    }
}

/// Shortest-arc interpolation on the 24-hour circle, so a key at 23:00 followed
/// by one at 01:00 passes through midnight rather than running backwards
/// through noon (the circular-mean reasoning `attractor_hour` already uses).
fn lerp_hours(a: f32, b: f32, t: f32) -> f32 {
    let mut d = (b - a).rem_euclid(24.0);
    if d > 12.0 {
        d -= 24.0;
    }
    (a + d * t).rem_euclid(24.0)
}

/// The pose at normalized path position `u` in [0, 1], and the authored time of
/// day there if every neighbouring key carries one.
pub fn pose_at(keys: &[Keyframe], closed: bool, u: f32) -> (Camera, Option<f32>) {
    match keys.len() {
        0 => (Camera::look_at(Vec3A::ZERO, Vec3A::Z, 55f32.to_radians()), None),
        1 => (keys[0].camera(), keys[0].tod),
        n => {
            let (seg, local) = segment_at(n, closed, u);
            let at = |i: isize| -> &Keyframe {
                let j = if closed {
                    (seg + i).rem_euclid(n as isize)
                } else {
                    (seg + i).clamp(0, n as isize - 1)
                };
                &keys[j as usize]
            };
            let (k0, k1, k2, k3) = (at(-1), at(0), at(1), at(2));
            let eye = catmull_rom(k0.eye_v(), k1.eye_v(), k2.eye_v(), k3.eye_v(), local);
            let target =
                catmull_rom(k0.target_v(), k1.target_v(), k2.target_v(), k3.target_v(), local);
            // fov rides the x lane of the same evaluator — one spline, no second
            // code path to keep in step.
            let fov = catmull_rom(
                Vec3A::new(k0.fov_deg, 0.0, 0.0),
                Vec3A::new(k1.fov_deg, 0.0, 0.0),
                Vec3A::new(k2.fov_deg, 0.0, 0.0),
                Vec3A::new(k3.fov_deg, 0.0, 0.0),
                local,
            )
            .x;
            let tod = match (k1.tod, k2.tod) {
                (Some(a), Some(b)) => Some(lerp_hours(a, b, local)),
                (Some(a), None) => Some(a),
                _ => None,
            };
            (Camera::look_at(eye, target, fov.to_radians()), tod)
        }
    }
}

/// The time of day for output frame `f` of `shot`: the world's attractor field
/// sampled along the path and band-limited by a symmetric window.
///
/// The window is what replaces the interactive integrator's easing. Sampling
/// `attractor_hour` at a single pose is faithful to the field but not to the
/// experience: the weight is `1/(d² + r²)`, so passing a SMALL island at close
/// range spikes its weight over a couple of frames and the hour lurches (the
/// curated set has a ~20x radius spread, so this is not hypothetical — the
/// self-test measures 2.5 h between adjacent frames without this). The flycam
/// hides that by easing at `TOD_RATE`, but easing makes the clock a function of
/// the entire preceding history, which would cost exactly the property this
/// mode is built on: that any single frame is re-renderable in isolation and
/// reproduces bit-for-bit.
///
/// A symmetric window over NEIGHBOURING POSES keeps both. It is still a pure
/// function of `f` — no history, no state — and it band-limits the swing the
/// same way easing does. Averaged as a circular mean, because hours wrap.
pub fn path_hour(
    shot: &Shot,
    attractors: &[crate::world::TodAttractor],
    f: u32,
) -> Option<f32> {
    if attractors.is_empty() {
        return None;
    }
    use std::f32::consts::TAU;
    let n = shot.kind.frames().max(1);
    // ~1.5% of the lap each side, at least one frame. Wide enough to smooth a
    // close pass, far narrower than the hour structure itself.
    let w = ((n as f32 * 0.015).round() as i64).clamp(1, 64);
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    for j in -w..=w {
        let fi = f as i64 + j;
        // Closed paths wrap (the lap is a loop, so the window must be too);
        // open paths clamp at the ends.
        let fi = if shot.closed {
            fi.rem_euclid(n as i64)
        } else {
            fi.clamp(0, n as i64 - 1)
        };
        let u = if n <= 1 { 0.0 } else { fi as f32 / n as f32 };
        let (cam, _) = pose_at(&shot.keys, shot.closed, u);
        let th = (crate::world::attractor_hour(attractors, cam.pos) / 24.0 * TAU) as f64;
        sx += th.cos();
        sy += th.sin();
    }
    if sx == 0.0 && sy == 0.0 {
        return Some(0.0);
    }
    Some(((sy.atan2(sx) / TAU as f64 * 24.0) as f32).rem_euclid(24.0))
}

// ---------------------------------------------------------------------------
// Presets derived from the world
// ---------------------------------------------------------------------------

/// Outward radial direction of an island from the world origin — the ring is
/// centered there by `world::ring_layout`, so this is "away from the middle".
/// A island sitting exactly at the origin (the single-island world) has no
/// radial direction; fall back to +X so the framing stays defined.
fn outward(c: Vec3A) -> Vec3A {
    let r = Vec3A::new(c.x, 0.0, c.z);
    if r.length_squared() > 1e-6 { r.normalize() } else { Vec3A::X }
}

/// The flagship path: one closed lap of the island ring, hour-ordered, so the
/// time-of-day attractors sweep dawn -> moonlit night over the loop.
///
/// The standoff/height blend the island's OWN radius with the largest radius in
/// the world. Pure per-island scaling frames a tiny island (the helmet) as
/// nicely as a huge one (the powerplant), but the spline between them would
/// then dive and climb by two orders of magnitude; the shared term keeps the
/// lap smooth while the per-island term keeps each subject framed.
pub fn world_lap(w: &World, frames: u32, fps: u32, res: (usize, usize), samples: u32) -> Shot {
    let rmax = w.islands.iter().fold(1.0f32, |a, i| a.max(i.radius));
    let n = w.islands.len();
    // Framing. The eye sits `stand` out horizontally and `up` above the ground,
    // looking at a point just above the island's base, so the depression angle
    // is atan((up - 0.18r) / stand) — roughly 28 degrees at these constants.
    // The first cut used 1.8r/0.55r, which is a ~15 degree near-horizon view:
    // technically correct, and it renders each island as a speck on an empty
    // plain. An aerial three-quarter view is what makes an island read as a
    // PLACE. Standoff stays >= ~1.5r so the lap is still comfortably outside
    // every island (the clearance gate measures it).
    //
    // HEIGHT COMES OFF `height`, NOT `radius`. Both terms used to scale with the
    // footprint, which flies the camera at a fixed multiple of how WIDE an
    // island is and has nothing to do with how tall it is: the lap cleared
    // Sponza (3.4 high) by the same margin it cleared Rungholt (0.8), so a city
    // and a cathedral both read as models on a table. Keyed to height the eye
    // sits a little above each subject's own roofline, which is the altitude at
    // which a place looks like a place.
    let stand = |i: &crate::world::Island| i.radius * 1.90 + i.height * 0.60;
    let up = |i: &crate::world::Island| i.height * 0.75 + i.radius * 0.45;
    // Per-island viewing key, plus a TRANSIT key on the ring arc between each
    // pair. The transit keys are what keep the lap outside the ring: a spline
    // straight from one island's eye to the next chords ACROSS the ring, and at
    // small island counts that chord passes through the islands themselves (at
    // n = 2 it degenerates to a line through both — the clearance gate caught
    // exactly this). Stepping through the angular midpoint at ring radius makes
    // the path an orbit rather than a polygon, which is also what it should
    // look like.
    //
    // Deliberately NO per-key `tod`: the tour's clock comes from the world's
    // own attractor field, sampled per output frame (invariant 3), which is the
    // same signal the interactive flycam eases toward — so a lap sweeps the day
    // exactly as flying it by hand does.
    //
    // THE LAP STAYS OUTSIDE, and that is a decision rather than an oversight.
    // Threading it through the authored interiors was built and measured and is
    // worse: the transits between enclosures have to clear both rooflines, so
    // the camera spends most of its frames high regardless; the climbs in and
    // out clip walls on the way through; and swinging between floor level and
    // above the ring seven times makes the attractor clock lurch, so the day
    // sweep — the whole point of the lap — stops reading as one continuous
    // sunrise. An aerial lap is the honest shape for a ring of dioramas. The
    // interiors are what the `islands` series is for.
    let mut keys: Vec<Keyframe> = Vec::with_capacity(n * 2);
    for (idx, isl) in w.islands.iter().enumerate() {
        let eye = isl.center + outward(isl.center) * stand(isl) + Vec3A::Y * up(isl);
        keys.push(Keyframe::new(eye, isl.center + Vec3A::Y * (isl.height * 0.45)));
        if n < 2 {
            continue;
        }
        let next = &w.islands[(idx + 1) % n];
        // Angular midpoint the SHORT way round, taken on angles rather than by
        // averaging the two centers — averaging opposed centers gives the
        // origin, which has no direction at all.
        let a0 = isl.center.z.atan2(isl.center.x);
        let a1 = next.center.z.atan2(next.center.x);
        let mut d = a1 - a0;
        while d <= 0.0 {
            d += std::f32::consts::TAU;
        }
        while d > std::f32::consts::TAU {
            d -= std::f32::consts::TAU;
        }
        let am = a0 + d * 0.5;
        let r0 = Vec3A::new(isl.center.x, 0.0, isl.center.z).length();
        let r1 = Vec3A::new(next.center.x, 0.0, next.center.z).length();
        // Transits ride WIDER and HIGHER than the island keys they sit between,
        // which is what turns the lap into a series of dips: the camera climbs
        // away over the empty apron, then drops toward the next island so it
        // grows in frame on approach. Pushing the radius past the plain mean
        // also counters the inward scallop a Catmull-Rom takes across a convex
        // ring (the clearance gate measures what survives).
        let rm = 0.5 * (r0 + r1) + 0.62 * (stand(isl) + stand(next));
        let hm = 0.80 * (up(isl) + up(next)) + rmax * 0.10;
        let eye = Vec3A::new(rm * am.cos(), hm, rm * am.sin());
        // Look ahead to where the lap is going, so the camera arrives already
        // framing the next island instead of whipping round on approach.
        keys.push(Keyframe::new(eye, next.center + Vec3A::Y * (next.height * 0.45)));
    }
    Shot {
        name: "tour".to_string(),
        kind: ShotKind::Sequence { frames, fps },
        keys,
        closed: true,
        res,
        samples,
        exposure: 0.0,
        gi: false,
        overlay: false,
        hud: None,
    }
}

/// Authored framing for the curated islands: `(name, eye, target, fov, stops)`,
/// where eye/target are offsets from the island centre in units of its own
/// `radius` (x/z) and `height` (y), so an entry survives a scene being refitted
/// or the ring being re-laid.
///
/// THE RULE THIS EXISTS TO BREAK. The bounding-sphere fit below is correct for
/// a SUBJECT and wrong for an ENCLOSURE. Framing a whole sphere from outside at
/// a 30-degree depression photographs the Damaged Helmet beautifully and
/// photographs Sponza's ROOF — and roofs are what shipped: the most famous
/// atrium in computer graphics as a rectangle of tiles, San Miguel's patio as a
/// grey box, Bistro's street as a smudge two hundred metres up. An enclosure has
/// to be shot from INSIDE, at eye level, which is a fact about the building and
/// not something a formula over a bounding box can recover.
///
/// So the five enclosures are authored and the two subjects are not: Rungholt
/// is a landscape and reads from above, the helmet is an object and the sphere
/// fit is exactly right for it. Anything absent — a user's own scene, an island
/// added later — falls through to the generic rule unchanged.
///
/// The hours are `CURATED`'s own, and each entry shoots what that table already
/// SAYS the island is for: "morning courtyard", "afternoon patio",
/// "golden-hour street", "moonlit night garden".
///
/// `stops` is exposure compensation (see `Shot::exposure`). It belongs here
/// rather than in a flag because it is a property of the subject: a courtyard
/// with the sun behind a wall is genuinely two stops under a lit exterior, and
/// every photographer opens up for the same reason. The two exteriors take 0.
const ISLAND_FRAMING: &[(&str, [f32; 3], [f32; 3], f32, f32)] = &[
    // Down the length of the atrium from the west end, tilted up so the
    // colonnade and the open sky above it carry the frame.
    ("sponza", [-0.425, 0.103, 0.025], [0.575, 0.324, 0.025], 70.0, 2.0),
    // The patio: fountain anchoring the centre, the arcade sweeping up on the
    // right, the ficus canopy framing the top and a gap of sky between them.
    // The eye sits LOW and the target HIGH on purpose — a level camera renders
    // the same courtyard as a furniture catalogue, and tilting up is what makes
    // the arcade tower instead of merely stand there.
    ("san-miguel", [0.261, 0.170, 0.043], [-0.391, 0.380, 0.043], 66.0, 2.0),
    // Street level at the corner, looking across the plaza to the tree and the
    // lamps, with the wet cobbles taking the low sun.
    ("bistro", [0.222, 0.225, 0.222], [0.639, 0.300, 0.556], 65.0, 2.0),
    // Shooting INTO the low sun, so the plant is rim-lit and the cranes and
    // chimney read as silhouette. The obvious sun-behind-camera angle is
    // technically better exposed and looks like nothing: at 06:30 the long
    // optical path makes the Mie haze the brightest thing in frame and the
    // whole image goes milky. -1 stop puts the contrast back.
    ("powerplant", [-0.700, 0.236, -0.656], [-0.378, 0.264, -0.098], 65.0, -1.0),
    // Low in the voxel garden so the fireflies sit against the star field.
    // Night stays night: +2 is legibility at README thumbnail size, not daylight.
    ("vokselia", [0.657, 0.375, 0.0], [-0.486, 0.750, 0.0], 75.0, 2.0),
];

/// One framed still per island, named so they sort in ring/hour order.
///
/// Authored framing wins where `ISLAND_FRAMING` has an entry; everything else
/// takes the bounding-sphere fit below.
pub fn island_shots(w: &World, res: (usize, usize), samples: u32) -> Vec<Shot> {
    w.islands
        .iter()
        .enumerate()
        .map(|(i, isl)| {
            if let Some(&(_, eye, tgt, fov, stops)) =
                ISLAND_FRAMING.iter().find(|f| f.0 == isl.name)
            {
                let at = |o: [f32; 3]| {
                    isl.center + Vec3A::new(o[0] * isl.radius, o[1] * isl.height, o[2] * isl.radius)
                };
                let hh = (isl.theme_hour.floor() as u32).min(23);
                let mm = ((isl.theme_hour.fract() * 60.0).round() as u32).min(59);
                let key = Keyframe::new(at(eye), at(tgt))
                    .with_fov(fov)
                    .with_tod(isl.theme_hour);
                let mut s = Shot::still(
                    &format!("island-{:02}-{}-{:02}{:02}", i + 1, isl.name, hh, mm),
                    key,
                    res,
                    samples,
                );
                s.exposure = stops;
                return s;
            }
            // Frame from INSIDE the ring looking outward, at a ~30 degree
            // depression. Two deliberate choices, both learned from the first
            // 4K pass:
            //
            // - Radial, not a fixed world direction. A constant look direction
            //   (the app's boot offset) points across the ring for some
            //   islands, so a "portrait of Powerplant" also contained Sponza,
            //   Rungholt and Vokselia on the horizon and read as a landscape of
            //   nothing in particular. Standing inside the ring and looking OUT
            //   puts the empty outer apron behind the subject.
            // - Steep enough to drop the horizon. At 30 degrees of depression
            //   with the 55 degree vertical FOV, the top ray still points
            //   ~2 degrees below level, so the horizon — and therefore every
            //   other island, which all sit on it — falls just outside the
            //   frame, leaving the subject and a clean apron.
            // Fit the whole subject: the framing radius is the larger of the
            // x/z half-footprint and the half-HEIGHT. Sizing off the footprint
            // alone frames a flat city correctly and cuts a tall one in half —
            // the Damaged Helmet is as tall as it is wide, and came out
            // cropped at the neck.
            // Distance from a BOUNDING-SPHERE fit, not a hand-picked multiple
            // of the footprint. A sphere of radius R fills the frame at
            // d = R / sin(fov/2); at the 55 degree vertical FOV that is 2.17R,
            // so 2.5R leaves a comfortable margin. The footprint is weighted
            // 0.75 because at a 30 degree depression a flat extent compresses
            // on screen while height does not — without that a flat city gets
            // framed as if it were a tower and sits tiny in the middle.
            let fit = (isl.radius * 0.75).max(isl.height * 0.5).max(1e-3);
            let d = fit * 2.5;
            let mid = isl.height * 0.5;
            let inward = -outward(isl.center);
            // 30 degrees of depression, held exactly, so every island in the
            // set is photographed from the same angle and the seven read as
            // one series rather than seven unrelated snapshots.
            let (c, s) = (30f32.to_radians().cos(), 30f32.to_radians().sin());
            let eye = isl.center + inward * (d * c) + Vec3A::Y * (mid + d * s);
            let target = isl.center + Vec3A::Y * mid;
            let hh = (isl.theme_hour.floor() as u32).min(23);
            let mm = ((isl.theme_hour.fract() * 60.0).round() as u32).min(59);
            let key = Keyframe::new(eye, target).with_tod(isl.theme_hour);
            Shot::still(
                &format!("island-{:02}-{}-{:02}{:02}", i + 1, isl.name, hh, mm),
                key,
                res,
                samples,
            )
        })
        .collect()
}

/// A closed orbit around a point — the preset that works for any scene,
/// world or not (`model.obj --cinematic orbit`).
pub fn orbit_shot(
    center: Vec3A,
    radius: f32,
    frames: u32,
    fps: u32,
    res: (usize, usize),
    samples: u32,
) -> Shot {
    const N: usize = 8;
    let keys: Vec<Keyframe> = (0..N)
        .map(|i| {
            let th = std::f32::consts::TAU * i as f32 / N as f32;
            let eye = center + Vec3A::new(radius * th.cos(), radius * 0.45, radius * th.sin());
            Keyframe::new(eye, center + Vec3A::Y * (radius * 0.06))
        })
        .collect();
    Shot {
        name: "orbit".to_string(),
        kind: ShotKind::Sequence { frames, fps },
        keys,
        closed: true,
        res,
        samples,
        exposure: 0.0,
        gi: false,
        overlay: false,
        hud: None,
    }
}

/// Smallest x/z clearance between any sampled pose on `shot`'s path and any
/// island, as a MULTIPLE of that island's radius. `> 1` means the camera stayed
/// outside every island for the whole flight.
///
/// This is the highest-value gate in the feature: `world_lap` places eyes
/// outside a convex ring and asserts that a Catmull-Rom through them scallops
/// toward each island without cutting through one — and a spline through points
/// on a circle bulges OUTSIDE the polygon, so the claim is plausible but not
/// self-evident. Measuring it is what keeps the release video from flying
/// through the middle of San Miguel.
pub fn min_clearance(shot: &Shot, w: &World) -> f32 {
    let n = shot.kind.frames().max(1);
    let mut worst = f32::INFINITY;
    for f in 0..n {
        let u = if n <= 1 { 0.0 } else { f as f32 / n as f32 };
        let (cam, _) = pose_at(&shot.keys, shot.closed, u);
        for isl in &w.islands {
            let d = cam.pos - isl.center;
            let dxz = (d.x * d.x + d.z * d.z).sqrt();
            worst = worst.min(dxz / isl.radius.max(1e-3));
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// CLI options and shot resolution
// ---------------------------------------------------------------------------

/// The parsed `--cinematic-*` sub-flags. One local in main.rs instead of ten.
#[derive(Clone, Debug)]
pub struct CineOpts {
    pub out: String,
    pub res: Option<(usize, usize)>,
    pub samples: Option<u32>,
    pub frames: Option<u32>,
    pub fps: u32,
    pub gi: Option<bool>,
    pub overlay: bool,
    /// `None` = no HUD. `Some(None)` = the flight HUD. `Some(Some(g))` = the
    /// pause menu on settings group `g`.
    pub hud: Option<Option<String>>,
    pub encode: bool,
    pub dry_run: bool,
    pub island: Option<String>,
    /// HDR output. Sequences write 16-bit PQ / Rec.2020 frames and encode to
    /// HDR10 HEVC; stills additionally write a linear OpenEXR master and a
    /// 16-bit PQ PNG (with the ffmpeg line that makes a viewable HDR AVIF).
    /// The SDR PNG is always written too — a README has to have something
    /// every browser can show.
    pub hdr: bool,
    /// Where linear 1.0 lands, in nits. Lower = more highlight headroom.
    pub paper_white: f32,
    /// Exposure compensation in stops — see `Shot::exposure`. `None` leaves
    /// each shot's own value alone (presets author their own for enclosures).
    pub exposure: Option<f32>,
}

impl Default for CineOpts {
    fn default() -> Self {
        CineOpts {
            out: "capture".to_string(),
            res: None,
            samples: None,
            frames: None,
            fps: DEFAULT_FPS,
            gi: None,
            overlay: false,
            hud: None,
            encode: false,
            dry_run: false,
            island: None,
            hdr: false,
            paper_white: 200.0,
            exposure: None,
        }
    }
}

impl CineOpts {
    fn res_or(&self, d: (usize, usize)) -> (usize, usize) {
        // yuv420p (the mp4 arm) requires even dimensions, so round down and say
        // so rather than letting ffmpeg fail after a long render.
        let (w, h) = self.res.unwrap_or(d);
        ((w & !1).max(2), (h & !1).max(2))
    }
    fn samples_or(&self, seq: bool) -> u32 {
        self.samples
            .unwrap_or(if seq { DEFAULT_SEQ_SAMPLES } else { DEFAULT_STILL_SAMPLES })
            .clamp(1, 4096)
    }
    fn frames_or(&self, d: u32) -> u32 {
        self.frames.unwrap_or(d).clamp(1, 100_000)
    }
}

/// Print the preset catalogue — what a bare `--cinematic` shows before it
/// renders its hero still, and all `--cinematic list` does.
pub fn print_catalogue() {
    eprintln!("cinematic presets:");
    eprintln!("  hero     one still at the boot overview pose (or --cam)          [default]");
    eprintln!("  islands  one framed still per world island, at its own theme hour");
    eprintln!("  tour     THE LAP: closed-loop flight over every island; the");
    eprintln!("           time-of-day attractors sweep dawn -> moonlit night");
    eprintln!("  orbit    a closed orbit of one island (--cinematic-island) or the scene");
    eprintln!("  foliage  a LOCKED-OFF clip at an island's framing — the wind in the");
    eprintln!("           leaves is the subject, so this is the one shot that cannot");
    eprintln!("           be a still (--cinematic-island, default san-miguel)");
    eprintln!("  hud      a hero still with the HUD composited (--cinematic-hud,");
    eprintln!("           --cinematic-island — it is `hero` wearing a hat)");
    eprintln!("  list     print this and exit");
    eprintln!("  <path>   anything else is read as a JSON shot list");
}

/// Turn the selector into the shots to render. `scene_center`/`scene_radius`
/// frame non-world scenes so every preset degrades rather than erroring — a
/// fresh checkout without `git lfs pull` must still produce media.
pub fn resolve_shots(
    sel: &str,
    c: &CineOpts,
    world: Option<&World>,
    cam0: Camera,
    scene_center: Vec3A,
    scene_radius: f32,
) -> Result<Vec<Shot>, String> {
    let still_res = c.res_or((1920, 1080));
    let seq_res = c.res_or((1920, 1080));
    let still_s = c.samples_or(false);
    let seq_s = c.samples_or(true);
    let gi_still = c.gi.unwrap_or(false);

    let hero = |name: &str| {
        let key = Keyframe {
            eye: [cam0.pos.x, cam0.pos.y, cam0.pos.z],
            target: {
                let t = cam0.pos + cam0.forward();
                [t.x, t.y, t.z]
            },
            fov_deg: cam0.fov_y.to_degrees(),
            tod: None,
        };
        let mut s = Shot::still(name, key, still_res, still_s);
        s.gi = gi_still;
        s.overlay = c.overlay;
        s.hud = c.hud.clone();
        s
    };

    // `hero` as a function rather than a match arm, because `hud` IS a hero
    // shot with the HUD forced on and used to be a SECOND, simpler expression
    // of that — one that read `cam0` directly and so silently ignored
    // `--cinematic-island`. `--cinematic hud --cinematic-island bistro` framed
    // the boot overview instead, produced a plausible-looking screenshot of an
    // empty plane, and nothing anywhere said no. One definition, two callers.
    let hero_shot = |sel: &str| -> Result<Shot, String> {
        Ok(match (world, &c.island) {
            (Some(w), Some(name)) => {
                let all = island_shots(w, still_res, still_s);
                let idx = w
                    .islands
                    .iter()
                    .position(|i| i.name.eq_ignore_ascii_case(name))
                    .ok_or_else(|| {
                        format!(
                            "unknown island '{name}' (have: {})",
                            w.islands.iter().map(|i| i.name.as_str()).collect::<Vec<_>>().join(", ")
                        )
                    })?;
                let mut s = all.into_iter().nth(idx).expect("index in range");
                s.name = sel.to_string();
                s.gi = gi_still;
                // A hero shot SHOOTS INTO THE LIGHT. The series framing looks
                // radially outward, which isolates a subject cleanly but points
                // wherever the ring happens to face — and at Bistro's golden
                // hour that meant turning our back on the sunset and rendering
                // a flat, backlit grey. Re-place the eye on the far side of the
                // island from the sun's azimuth, keeping the same distance and
                // depression, so the sun sits beyond the subject: rim light,
                // long shadows toward camera, the disc and its glare in frame.
                //
                // ...unless the island is AUTHORED. The relocation below is a
                // rule for framing a subject from outside, and an authored
                // entry means the subject is an enclosure whose shot was
                // composed by eye from within it — moving that eye out to the
                // island's sunward apron discards the composition and puts the
                // camera back outside the building looking at a wall. Where an
                // author already made the call, honour it (powerplant's entry
                // shoots into the sun on its own terms).
                let isl = &w.islands[idx];
                let authored = ISLAND_FRAMING.iter().any(|f| f.0 == isl.name);
                let sun = crate::scene::sun_dir_for_tod(isl.theme_hour);
                let az = Vec3A::new(sun.x, 0.0, sun.z);
                if !authored && az.length_squared() > 1e-6 {
                    let target = s.keys[0].target_v();
                    let fit = (isl.radius * 0.75).max(isl.height * 0.5).max(1e-3);
                    let d = fit * 2.6;
                    // HERO_DEPRESSION is much shallower than the series' 30
                    // degrees, and that is the whole shot. The series angle
                    // exists to drop the horizon out of frame; a hero shot
                    // needs the opposite — at 12 degrees the top of the frame
                    // reaches ~15 degrees above level, so the sky, the horizon
                    // and the sun DISC are all in it. Shot from 30 degrees the
                    // same pose is a grey plain with a bright smudge on it.
                    let (c, sn) = (12f32.to_radians().cos(), 12f32.to_radians().sin());
                    // `sun_dir_for_tod` points TOWARD the sun, so the eye goes
                    // on the OPPOSITE side of the subject: standing away from
                    // the sun and looking back along its azimuth puts subject
                    // and sun in one frame — rim light, long shadows toward
                    // camera, and the glare the bloom pass exists for.
                    let e = target - az.normalize() * (d * c) + Vec3A::Y * (d * sn);
                    // Keep the key's pinned hour. Re-placing the eye with a
                    // bare `Keyframe::new` drops it, and the clock then falls
                    // back to the attractor field AT THE NEW POSITION — which
                    // moved the camera nearer the night island and rendered
                    // golden-hour Bistro at midnight.
                    let tod = s.keys[0].tod;
                    let fov = s.keys[0].fov_deg;
                    s.keys[0] = Keyframe { fov_deg: fov, tod, ..Keyframe::new(e, target) };
                }
                s
            }
            _ => hero(sel),
        })
    };

    let mut shots = match sel {
        // `--cinematic hero --cinematic-island bistro` frames that island with
        // the same rig the `islands` series uses, so a hero shot is a member of
        // the series rather than a one-off pose that has to be re-found by hand.
        "hero" => vec![hero_shot("hero")?],
        "hud" => {
            let mut s = hero_shot("hud")?;
            // The hud preset implies the HUD even without --cinematic-hud;
            // otherwise the preset name would be a lie.
            s.hud = Some(c.hud.clone().flatten());
            vec![s]
        }
        "islands" => match world {
            Some(w) => island_shots(w, still_res, still_s),
            None => {
                eprintln!(
                    "cinematic: 'islands' needs the world (git lfs pull?) — \
                     falling back to one hero still"
                );
                vec![hero("hero")]
            }
        },
        "tour" => {
            let frames = c.frames_or(DEFAULT_FRAMES);
            let shot = match world {
                Some(w) if !w.islands.is_empty() => world_lap(w, frames, c.fps, seq_res, seq_s),
                _ => {
                    eprintln!(
                        "cinematic: 'tour' needs the world (git lfs pull?) — \
                         falling back to an orbit of this scene"
                    );
                    orbit_shot(scene_center, scene_radius, frames, c.fps, seq_res, seq_s)
                }
            };
            vec![shot]
        }
        "orbit" => {
            let frames = c.frames_or(360);
            let (center, radius) = match (world, &c.island) {
                (Some(w), Some(name)) => {
                    let isl = w
                        .islands
                        .iter()
                        .find(|i| i.name.eq_ignore_ascii_case(name))
                        .ok_or_else(|| {
                            format!(
                                "unknown island '{name}' (have: {})",
                                w.islands
                                    .iter()
                                    .map(|i| i.name.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        })?;
                    (isl.center, isl.radius * 2.4)
                }
                (Some(w), None) => {
                    let isl = w
                        .islands
                        .iter()
                        .max_by(|a, b| a.radius.total_cmp(&b.radius))
                        .expect("non-empty islands");
                    (isl.center, isl.radius * 2.4)
                }
                _ => (scene_center, scene_radius),
            };
            vec![orbit_shot(center, radius, frames, c.fps, seq_res, seq_s)]
        }
        // A LOCKED-OFF sequence at an island's authored framing — see
        // FOLIAGE_FRAMES for why this preset exists and why it does not move.
        // It reuses `island_shots`' pose rather than authoring a second one, so
        // the clip is literally the `islands` still with time running.
        "foliage" => {
            let frames = c.frames_or(FOLIAGE_FRAMES);
            match world {
                Some(w) if !w.islands.is_empty() => {
                    let name = c.island.clone().unwrap_or_else(|| FOLIAGE_ISLAND.to_string());
                    let idx = w
                        .islands
                        .iter()
                        .position(|i| i.name.eq_ignore_ascii_case(&name))
                        .ok_or_else(|| {
                            format!(
                                "unknown island '{name}' (have: {})",
                                w.islands
                                    .iter()
                                    .map(|i| i.name.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        })?;
                    // Sequence res/samples, not the stills': this pays its cost
                    // once per output frame, 120 times over.
                    let mut s = island_shots(w, seq_res, seq_s)
                        .into_iter()
                        .nth(idx)
                        .expect("index in range");
                    s.name = "foliage".to_string();
                    s.kind = ShotKind::Sequence { frames, fps: c.fps };
                    s.keys.truncate(1);
                    vec![s]
                }
                _ => {
                    eprintln!(
                        "cinematic: 'foliage' needs the world (git lfs pull?) — \
                         falling back to a locked-off clip of this scene"
                    );
                    let mut s = hero("foliage");
                    s.kind = ShotKind::Sequence { frames, fps: c.fps };
                    s.res = seq_res;
                    s.samples = seq_s;
                    vec![s]
                }
            }
        }
        other => return Err(format!("unknown preset '{other}'")),
    };

    // Sub-flags override whatever the preset chose.
    for s in &mut shots {
        if let Some(g) = c.gi {
            s.gi = g;
        }
        if c.overlay {
            s.overlay = true;
        }
        if c.hud.is_some() {
            s.hud = c.hud.clone();
        }
        if let Some(ev) = c.exposure {
            s.exposure = ev;
        }
    }
    Ok(shots)
}

// ---------------------------------------------------------------------------
// HDR output
// ---------------------------------------------------------------------------

/// Mastering peak for HDR files, in nits. A file has no display to probe, so
/// this is the mastering-display luminance the content is graded against —
/// 1000 is the HDR10 convention and what consumer displays tone-map from.
pub const HDR_MASTER_NITS: f32 = 1000.0;

/// Encode a linear-RGB image (post-glare, the same signal the SDR PNG is made
/// from) to 16-bit PQ / Rec.2020 — the wire format for an HDR10 still or an
/// HDR10 video frame.
///
/// This is `tone::ToneParams::hdr10` — the SAME curve the renderer presents
/// through on an HDR10 swapchain, so a captured HDR still and the live window
/// agree by construction rather than by a second implementation.
pub fn pq_rgb16(hdr: &[f32], paper_white: f32, peak_nits: f32) -> Vec<u16> {
    let p = crate::tone::ToneParams::hdr10(paper_white, peak_nits);
    let enc = |v: f32| -> u16 { (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16 };
    hdr.chunks_exact(3)
        .flat_map(|c| {
            let v = crate::tone::map(Vec3A::new(c[0], c[1], c[2]), p);
            [enc(v.x), enc(v.y), enc(v.z)]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ffmpeg
// ---------------------------------------------------------------------------

/// The encode commands for a rendered sequence, as (label, argv) pairs.
///
/// WebP leads because an animated WebP is what actually plays inline in a
/// GitHub README; a committed mp4 does not (only assets uploaded to GitHub's
/// own CDN do), so the mp4 is for the Release page.
pub fn ffmpeg_cmds(
    dir: &str,
    name: &str,
    fps: u32,
    width: usize,
    hdr: bool,
) -> Vec<(String, Vec<String>)> {
    let pattern = format!("{dir}/frames/f_%05d.png");
    // The inline loop is downscaled hard: a 4K animation is neither renderable
    // by a browser nor welcome in a git clone.
    let inline_w = width.min(1280) & !1;
    let s = |v: &str| v.to_string();
    let mut out: Vec<(String, Vec<String>)> = Vec::new();

    if hdr {
        // HDR10: PQ (SMPTE 2084) + Rec.2020 primaries at 10 bits. The frames
        // are already 16-bit PQ/Rec.2020, so the encoder must be TOLD that
        // rather than converting — `-color_trc smpte2084` on the input side
        // labels it, and the matching output tags travel in the bitstream so a
        // player knows not to treat it as SDR.
        //
        // `-tag:v hvc1` is not optional in practice: without it QuickTime and
        // Safari refuse the file outright (they reject the default `hev1`).
        out.push((
            "HDR10 HEVC (Rec.2020 / PQ, 10-bit) — the Release asset".to_string(),
            vec![
                s("-y"), s("-framerate"), fps.to_string(), s("-start_number"), s("0"),
                s("-i"), pattern.clone(),
                s("-vf"), s("scale=in_range=full:out_range=limited,format=yuv420p10le"),
                s("-c:v"), s("libx265"), s("-preset"), s("slow"), s("-crf"), s("18"),
                s("-pix_fmt"), s("yuv420p10le"),
                s("-color_primaries"), s("bt2020"), s("-color_trc"), s("smpte2084"),
                s("-colorspace"), s("bt2020nc"),
                s("-x265-params"),
                format!(
                    "hdr10=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:\
                     colormatrix=bt2020nc:master-display=G(8500,39850)B(6550,2300)\
                     R(35400,14600)WP(15635,16450)L({},1)",
                    (HDR_MASTER_NITS as u32) * 10_000
                ),
                s("-tag:v"), s("hvc1"), s("-movflags"), s("+faststart"),
                format!("{dir}/{name}-hdr10.mp4"),
            ],
        ));
        // An SDR sibling, tone-mapped from the PQ frames, so there is something
        // every player and every README can actually show.
        //
        // THE INPUT COLOUR TAGS ARE LOAD-BEARING AND GO BEFORE `-i`. A PNG
        // carries no colour metadata, so a bare `zscale=t=linear` has no input
        // transfer function to convert FROM and the whole filter graph dies
        // with "code 3074: no path between colorspaces" — which is an ffmpeg
        // library error, so it surfaces as a failed encode and a zero-byte mp4
        // rather than anything that names the cause. Declaring the frames as
        // PQ/Rec.2020 on the INPUT side is what fixes it; the same properties
        // written into the filter as `tin=`/`pin=` do NOT (zscale still refuses
        // the RGB-matrix conversion), which is worth knowing before anyone
        // "tidies" this back into the filter string.
        out.push((
            "SDR HEVC (tone-mapped from the HDR frames, for players without HDR)".to_string(),
            vec![
                s("-y"), s("-framerate"), fps.to_string(), s("-start_number"), s("0"),
                s("-color_primaries"), s("bt2020"), s("-color_trc"), s("smpte2084"),
                s("-colorspace"), s("bt2020nc"),
                s("-i"), pattern.clone(),
                s("-vf"),
                format!(
                    "zscale=t=linear:npl={},format=gbrpf32le,zscale=p=bt709,\
                     tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=limited,format=yuv420p",
                    HDR_MASTER_NITS as u32 / 5
                ),
                s("-c:v"), s("libx265"), s("-preset"), s("slow"), s("-crf"), s("20"),
                s("-tag:v"), s("hvc1"), s("-movflags"), s("+faststart"),
                format!("{dir}/{name}.mp4"),
            ],
        ));
    } else {
        out.push((
            "HEVC (Release asset — does NOT play inline in a README)".to_string(),
            vec![
                s("-y"), s("-framerate"), fps.to_string(), s("-start_number"), s("0"),
                s("-i"), pattern.clone(),
                s("-c:v"), s("libx265"), s("-preset"), s("slow"), s("-crf"), s("18"),
                s("-pix_fmt"), s("yuv420p"),
                s("-tag:v"), s("hvc1"), s("-movflags"), s("+faststart"),
                format!("{dir}/{name}.mp4"),
            ],
        ));
        out.push((
            "webp (inline-able in a GitHub README, loops forever)".to_string(),
            vec![
                s("-y"), s("-framerate"), fps.to_string(), s("-start_number"), s("0"),
                s("-i"), pattern,
                s("-vf"), format!("scale={inline_w}:-2:flags=lanczos"),
                s("-c:v"), s("libwebp"), s("-lossless"), s("0"), s("-q:v"), s("72"),
                s("-compression_level"), s("6"), s("-loop"), s("0"), s("-an"),
                format!("{dir}/{name}.webp"),
            ],
        ));
    }
    out
}

/// The command that turns a captured 16-bit PQ still into a VIEWABLE HDR image.
/// EXR is the archival master but no browser shows one; AVIF with PQ/Rec.2020
/// tagging is what a modern viewer actually renders as HDR.
/// THE TAGGING TRAP, measured: the `-color_primaries`/`-color_trc` OUTPUT
/// options that correctly tag an HEVC/mp4 do NOT reach the AVIF `colr` box.
/// Encoding that way writes primaries=2, transfer=2 ("unspecified") with only
/// the matrix surviving, and a viewer then renders a PQ image as if it were
/// sRGB — washed out, and silently so. `-aom-params` is worse (it loses the
/// matrix too). Stamping the frame with `setparams` inside the filter chain is
/// what actually lands 9/16/9 in the box; verify with the colr bytes, not with
/// ffprobe, which does not surface an AVIF container box at all.
pub fn ffmpeg_still_hdr(png16: &str, out_avif: &str) -> Vec<String> {
    let s = |v: &str| v.to_string();
    vec![
        s("-y"), s("-i"), png16.to_string(),
        s("-vf"),
        s("format=yuv420p10le,setparams=color_primaries=bt2020:\
           color_trc=smpte2084:colorspace=bt2020nc"),
        s("-c:v"), s("libaom-av1"), s("-crf"), s("20"), s("-cpu-used"), s("6"),
        s("-frames:v"), s("1"), out_avif.to_string(),
    ]
}

// ---------------------------------------------------------------------------
// HUD compositing
// ---------------------------------------------------------------------------

/// Source-over composite of a PREMULTIPLIED RGBA overlay onto a packed
/// `0x00RRGGBB` present buffer, in DISPLAY space.
///
/// Display space (not linear) is deliberate and matches the shipped GPU path:
/// `hud.hlsl`'s SDR arm blends the texel straight against the gamma-encoded
/// backbuffer. Compositing in linear here would make the captured HUD differ
/// from the one on screen, which is the whole point of capturing it.
///
/// Premultiplied means the source term is already scaled by alpha, so the blend
/// is `dst = src + dst * (1 - a)` with no divide.
/// One pixel of it — the per-texel form `Hud::composite_sdr` calls, mirroring
/// the GPU blend state (`SrcBlend = ONE`, `DestBlend = INV_SRC_ALPHA`).
///
/// Known-accept: the GPU composites in float, this in 8-bit with round-half-up,
/// so the two can differ by 1 LSB. No gate compares them and none is wanted.
#[inline]
pub fn over_sdr(dst: u32, r: u8, g: u8, b: u8, a: u8) -> u32 {
    if a == 0 {
        return dst;
    }
    let inv = 255 - a as u32;
    // +127 rounds the 8-bit multiply instead of truncating it; without it a
    // transparent-to-opaque ramp drifts a level dark.
    let ch = |s: u8, d: u32| -> u32 { (s as u32 + (d * inv + 127) / 255).min(255) };
    (ch(r, (dst >> 16) & 0xff) << 16) | (ch(g, (dst >> 8) & 0xff) << 8) | ch(b, dst & 0xff)
}

pub fn composite_premul(present: &mut [u32], hud: &[[u8; 4]], w: usize, h: usize) {
    debug_assert!(present.len() >= w * h && hud.len() >= w * h);
    for i in 0..w * h {
        let [r, g, b, a] = hud[i];
        present[i] = over_sdr(present[i], r, g, b, a);
    }
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

/// Closed-form gates, run by `--check`. Pure — no rng, no GPU, no DLLs.
pub fn self_test() -> Result<(), String> {
    // ISLAND_FRAMING is keyed by island NAME, and a key that matches nothing
    // fails SILENTLY into the bounding-sphere rule — which is the exact bug the
    // table exists to fix, so a rename would quietly restore roof photos of
    // Sponza with every gate still green. Pin the keys against the curated set.
    {
        for (name, eye, tgt, fov, ev) in ISLAND_FRAMING {
            if !crate::world::CURATED.iter().any(|s| s.name == *name) {
                return Err(format!(
                    "ISLAND_FRAMING references '{name}', which is not a CURATED island"
                ));
            }
            // Offsets are in units of radius/height; past a few island radii the
            // camera is out over the ring and framing something else entirely.
            for v in eye.iter().chain(tgt.iter()) {
                if !v.is_finite() || v.abs() > 4.0 {
                    return Err(format!("ISLAND_FRAMING '{name}': offset {v} out of range"));
                }
            }
            if !(20.0..=110.0).contains(fov) {
                return Err(format!("ISLAND_FRAMING '{name}': fov {fov} out of range"));
            }
            if !ev.is_finite() || ev.abs() > 8.0 {
                return Err(format!("ISLAND_FRAMING '{name}': exposure {ev} out of range"));
            }
            // An eye that coincides with its target has no look direction.
            let d = (0..3).map(|i| (eye[i] - tgt[i]).powi(2)).sum::<f32>();
            if d < 1e-6 {
                return Err(format!("ISLAND_FRAMING '{name}': eye and target coincide"));
            }
        }
        // Exposure must be structurally inert at 0 — the bit-identity contract
        // every pre-exposure capture rests on.
        let mut s = Shot::still("t", Keyframe::new(Vec3A::ZERO, Vec3A::X), (4, 4), 1);
        if s.exposure_scale() != 1.0 {
            return Err("exposure 0 must scale by exactly 1.0".into());
        }
        s.exposure = 1.0;
        if s.exposure_scale() != 2.0 {
            return Err("exposure +1 stop must scale by exactly 2.0".into());
        }
        s.exposure = -1.0;
        if s.exposure_scale() != 0.5 {
            return Err("exposure -1 stop must scale by exactly 0.5".into());
        }
    }
    // The `foliage` preset, whose two claims are both silently breakable.
    {
        // (a) Its default island must be CURATED *and* AUTHORED. A non-curated
        // name errors loudly, but an un-authored one does not: the shot would
        // fall through to the bounding-sphere fit and produce a 120-frame clip
        // of a roof, where the leaves are a few pixels wide and the whole point
        // of the preset is lost. Same silent-failure class as the table above.
        if !crate::world::CURATED.iter().any(|s| s.name == FOLIAGE_ISLAND) {
            return Err(format!("FOLIAGE_ISLAND '{FOLIAGE_ISLAND}' is not a CURATED island"));
        }
        if !ISLAND_FRAMING.iter().any(|f| f.0 == FOLIAGE_ISLAND) {
            return Err(format!(
                "FOLIAGE_ISLAND '{FOLIAGE_ISLAND}' has no ISLAND_FRAMING entry — the clip \
                 would take the bounding-sphere fit and film a roof"
            ));
        }
        // (b) LOCKED OFF. The preset expresses a static camera as a one-key
        // Sequence, which is only a static camera because `pose_at` short-
        // circuits at len 1; a future spline that interpolated a single key
        // against itself would still be static, but one that clamped into a
        // 4-point window would not. Pin the pose AND the pinned hour across u.
        let key = Keyframe { tod: Some(15.5), ..Keyframe::new(Vec3A::new(3.0, 2.0, 1.0), Vec3A::X) };
        let (c0, t0) = pose_at(std::slice::from_ref(&key), true, 0.0);
        for u in [0.25, 0.5, 0.75, 1.0] {
            let (c, t) = pose_at(std::slice::from_ref(&key), true, u);
            if c != c0 || t != t0 {
                return Err(format!("foliage: one-key pose moved at u={u} — camera not locked off"));
            }
        }
        if t0 != Some(15.5) {
            return Err("foliage: one-key pose dropped its pinned hour".into());
        }
    }
    // `hud` and `foliage` must both honour --cinematic-island, because both are
    // `hero` wearing a different hat. `hud` did NOT: it read the boot camera
    // directly, so `--cinematic hud --cinematic-island bistro` rendered the
    // overview pose — a perfectly plausible screenshot of the wrong thing, with
    // no error anywhere. Pin all three against one synthetic world.
    {
        let w = World {
            islands: vec![crate::world::Island {
                name: "probe-a".into(),
                center: Vec3A::new(10.0, 0.0, 0.0),
                radius: 4.0,
                height: 3.0,
                theme_hour: 9.0,
            }],
            field_half: 30.0,
        };
        let c = CineOpts {
            out: String::new(),
            res: Some((64, 64)),
            samples: Some(1),
            frames: Some(2),
            fps: 30,
            gi: None,
            overlay: false,
            hud: None,
            encode: false,
            dry_run: true,
            island: Some("probe-a".into()),
            hdr: false,
            paper_white: 200.0,
            exposure: None,
        };
        let cam = Camera::look_at(Vec3A::ZERO, Vec3A::X, 1.0);
        let key_of = |sel: &str| -> Result<Keyframe, String> {
            let s = resolve_shots(sel, &c, Some(&w), cam, Vec3A::ZERO, 1.0)?;
            let s = s.into_iter().next().ok_or_else(|| format!("{sel}: no shot"))?;
            s.keys.first().copied().ok_or_else(|| format!("{sel}: no keys"))
        };
        let same = |a: &Keyframe, b: &Keyframe| a.eye == b.eye && a.target == b.target;
        // `hud` IS `hero` — including the sunward relocation an un-authored
        // island gets, which is why this compares against hero and not islands.
        let hero_key = key_of("hero")?;
        let hud_key = key_of("hud")?;
        if !same(&hud_key, &hero_key) || hud_key.tod != hero_key.tod {
            return Err(format!(
                "preset 'hud' ignored --cinematic-island: eye {:?} vs hero's {:?}",
                hud_key.eye, hero_key.eye
            ));
        }
        // `foliage` is the ISLANDS framing with the clock running — deliberately
        // NOT hero's, which re-places the eye to shoot into the sun and would
        // discard the composition the clip is framed on.
        let isl_key = *resolve_shots("islands", &c, Some(&w), cam, Vec3A::ZERO, 1.0)?[0]
            .keys
            .first()
            .ok_or("islands: no keys")?;
        let fol_key = key_of("foliage")?;
        if !same(&fol_key, &isl_key) || fol_key.tod != isl_key.tod {
            return Err(format!(
                "preset 'foliage' is not the islands framing: eye {:?} vs {:?}",
                fol_key.eye, isl_key.eye
            ));
        }
        // Anti-vacuity: both comparisons must be able to fail. If the island
        // pose coincided with the no-island fallback, a preset that ignored
        // --cinematic-island entirely would still pass.
        let mut c0 = c.clone();
        c0.island = None;
        let fb = *resolve_shots("hero", &c0, Some(&w), cam, Vec3A::ZERO, 1.0)?[0]
            .keys
            .first()
            .ok_or("hero: no keys")?;
        if same(&fb, &hero_key) || same(&fb, &isl_key) {
            return Err("island framing coincides with the fallback pose — gate is vacuous".into());
        }
    }
    // The spline must agree with main.rs's copy at an arbitrary interior point.
    {
        let (p0, p1, p2, p3) = (
            Vec3A::new(-1.0, 0.0, 0.0),
            Vec3A::new(0.0, 0.0, 0.0),
            Vec3A::new(1.0, 2.0, 0.0),
            Vec3A::new(2.0, 0.0, 0.0),
        );
        let a = catmull_rom(p0, p1, p2, p3, 0.5);
        let b = crate::catmull_rom(p0, p1, p2, p3, 0.5);
        if a != b {
            return Err(format!("catmull_rom diverged from main.rs: {a:?} vs {b:?}"));
        }
        // Catmull-Rom interpolates its control points: t=0 IS p1, t=1 IS p2.
        if catmull_rom(p0, p1, p2, p3, 0.0) != p1 {
            return Err("catmull_rom(0) must be p1".into());
        }
        let e = (catmull_rom(p0, p1, p2, p3, 1.0) - p2).length();
        if e > 1e-6 {
            return Err(format!("catmull_rom(1) must be p2 (err {e})"));
        }
    }

    // A square, as both an open and a closed path.
    let sq: Vec<Keyframe> = [
        (Vec3A::new(10.0, 2.0, 0.0), Vec3A::ZERO),
        (Vec3A::new(0.0, 2.0, 10.0), Vec3A::ZERO),
        (Vec3A::new(-10.0, 2.0, 0.0), Vec3A::ZERO),
        (Vec3A::new(0.0, 2.0, -10.0), Vec3A::ZERO),
    ]
    .iter()
    .map(|(e, t)| Keyframe::new(*e, *t))
    .collect();

    // u = 0 reproduces key 0 exactly (a film's first frame is the pose the
    // author wrote, not an interpolation of it).
    let (c0, _) = pose_at(&sq, true, 0.0);
    if (c0.pos - sq[0].eye_v()).length() > 1e-6 {
        return Err(format!("pose_at(0) must be key 0, got {:?}", c0.pos));
    }

    // CLOSED-LOOP SEAM: u -> 1 must approach u = 0 in both position and
    // direction. A seam discontinuity is invisible in a still and reads as a
    // hard jolt in the loop, which is exactly what an inline README animation
    // shows off. Compare the wrap-around endpoint and a step either side.
    {
        let (c1, _) = pose_at(&sq, true, 1.0);
        if (c1.pos - c0.pos).length() > 1e-4 {
            return Err(format!(
                "closed loop must close: u=1 {:?} vs u=0 {:?}",
                c1.pos, c0.pos
            ));
        }
        // C1 at the seam. A one-sided finite difference carries O(eps)
        // truncation error from the path's own curvature, so a fixed tolerance
        // would measure the probe, not the spline — uniform Catmull-Rom is
        // analytically C1 at its knots (f'(1) = (p3-p1)/2 for one segment is
        // f'(0) = (p2-p0)/2 for the next), and a naive limit here rejected a
        // provably-continuous seam. Test the SCALING instead: for a C1 seam the
        // measured gap falls linearly with eps; a real kink leaves a gap that
        // does not shrink at all.
        let gap = |eps: f32| -> f32 {
            let (before, _) = pose_at(&sq, true, 1.0 - eps);
            let (after, _) = pose_at(&sq, true, eps);
            let v_in = (c0.pos - before.pos) / eps;
            let v_out = (after.pos - c0.pos) / eps;
            (v_in - v_out).length() / v_in.length().max(1e-6)
        };
        let (g1, g4) = (gap(4e-3), gap(1e-3));
        if g1 > 0.2 {
            return Err(format!("closed-loop seam has a large velocity jump: {g1}"));
        }
        // Quartering eps must quarter the gap (allow 0.5x slack for fp noise).
        if g4 > g1 * 0.5 {
            return Err(format!(
                "closed-loop seam is not C1: gap {g4} at eps/4 vs {g1} at eps did not shrink"
            ));
        }
    }

    // OPEN paths clamp instead of wrapping: u=1 is the LAST key, not the first.
    {
        let (c1, _) = pose_at(&sq, false, 1.0);
        if (c1.pos - sq[3].eye_v()).length() > 1e-6 {
            return Err(format!("open path u=1 must be the last key, got {:?}", c1.pos));
        }
    }

    // Determinism: same input, bit-identical pose. The whole reproducibility
    // claim (re-shoot frame 700 and get frame 700) rests on this.
    for &u in &[0.0f32, 0.123, 0.5, 0.77, 1.0] {
        let (a, _) = pose_at(&sq, true, u);
        let (b, _) = pose_at(&sq, true, u);
        if a.pos != b.pos || a.yaw != b.yaw || a.pitch != b.pitch || a.fov_y != b.fov_y {
            return Err(format!("pose_at is not deterministic at u={u}"));
        }
    }

    // look_at round-trip: the camera actually points at the authored target.
    {
        let (c, _) = pose_at(&sq, true, 0.0);
        let want = (sq[0].target_v() - sq[0].eye_v()).normalize();
        if (c.forward() - want).length() > 1e-5 {
            return Err(format!("look_at round-trip: {:?} vs {:?}", c.forward(), want));
        }
    }

    // The hour circle takes the short way round: 23:00 -> 01:00 crosses
    // midnight (the reason `attractor_hour` is a circular mean at all).
    {
        let m = lerp_hours(23.0, 1.0, 0.5);
        if !(m > 23.9 || m < 0.1) {
            return Err(format!("lerp_hours(23,1,0.5) should cross midnight, got {m}"));
        }
        if (lerp_hours(6.0, 10.0, 0.5) - 8.0).abs() > 1e-4 {
            return Err("lerp_hours(6,10,0.5) should be 8".into());
        }
        if (lerp_hours(5.0, 5.0, 0.7) - 5.0).abs() > 1e-6 {
            return Err("lerp_hours of equal hours must be that hour".into());
        }
    }

    // Segment mapping: closed paths get n segments, open ones n-1.
    {
        let (s, l) = segment_at(4, true, 1.0);
        if s != 4 || l.abs() > 1e-6 {
            return Err(format!("closed segment_at(1.0) = ({s}, {l})"));
        }
        let (s, l) = segment_at(4, false, 1.0);
        if s != 2 || (l - 1.0).abs() > 1e-6 {
            return Err(format!("open segment_at(1.0) = ({s}, {l})"));
        }
    }

    // HUD composite identities: alpha 0 leaves the pixel untouched, and an
    // opaque premultiplied source replaces it exactly.
    {
        let mut px = vec![0x00405060u32; 4];
        let clear = vec![[0u8, 0, 0, 0]; 4];
        let before = px.clone();
        composite_premul(&mut px, &clear, 2, 2);
        if px != before {
            return Err("alpha-0 overlay must not change the image".into());
        }
        let opaque = vec![[10u8, 20, 30, 255]; 4];
        composite_premul(&mut px, &opaque, 2, 2);
        if px.iter().any(|&p| p != 0x000A141E) {
            return Err(format!("opaque overlay must replace the pixel, got {:08x}", px[0]));
        }
        // Half-alpha premultiplied grey over black stays exactly the source.
        let mut px2 = vec![0u32; 1];
        composite_premul(&mut px2, &[[64, 64, 64, 128]], 1, 1);
        if px2[0] != 0x00404040 {
            return Err(format!("premultiplied blend over black: {:08x}", px2[0]));
        }
    }

    // THE CLEARANCE GATE. `world_lap` puts the eye outside a convex ring of
    // islands and relies on a Catmull-Rom through those eyes staying outside
    // every island. A spline through points on a circle bulges OUTWARD between
    // them, so that is plausible — but "plausible" is how you end up with a
    // release video that flies through the middle of San Miguel. Measure it, on
    // ring shapes built the way `ring_layout` builds them, including a hostile
    // mix of radii (one huge island next to tiny ones is the case where the
    // blended standoff could pull the path in).
    {
        use crate::world::Island;
        for (n, radii) in [
            (3usize, vec![40.0f32, 40.0, 40.0]),
            (7, vec![60.0, 8.0, 45.0, 3.0, 55.0, 30.0, 20.0]),
            (7, vec![60.0; 7]),
            (2, vec![50.0, 5.0]),
        ] {
            let fmax = radii.iter().cloned().fold(1e-3f32, f32::max);
            // `ring_layout`'s own geometry: pitch = 1.25 * the largest
            // FOOTPRINT (= 2 * radius), ring radius = max(pitch*n/tau, pitch).
            let pitch = 1.25 * (2.0 * fmax);
            let r = (pitch * n as f32 / std::f32::consts::TAU).max(pitch);
            let islands: Vec<Island> = (0..n)
                .map(|i| {
                    let th = std::f32::consts::TAU * i as f32 / n as f32;
                    Island {
                        name: format!("i{i}"),
                        center: Vec3A::new(r * th.cos(), 0.0, r * th.sin()),
                        radius: radii[i],
                        // Taller than wide on the small islands: the aspect
                        // ratio the framing has to survive.
                        height: radii[i] * if i % 2 == 0 { 0.4 } else { 2.5 },
                        // Modelled on the real curated set (06:30 -> 22:00),
                        // deliberately NOT a full 24h sweep: hours spaced a
                        // clean 24/n apart put an even-n ring in EXACT
                        // antipodal opposition, where the circular mean has no
                        // defined value at the midpoint (attractor_hour
                        // documents that cancellation) and swings 12h across
                        // it. That is a property of averaging opposed angles,
                        // not of the lap — and the curated set cannot produce
                        // it, since it spans well under a full circle.
                        theme_hour: 6.5 + 15.5 * i as f32 / n as f32,
                    }
                })
                .collect();
            let w = World { islands, field_half: r + 0.5 * pitch };
            let shot = world_lap(&w, 240, 30, (1920, 1080), 1);
            let clear = min_clearance(&shot, &w);
            if !(clear > 1.05) {
                return Err(format!(
                    "world_lap flies through an island: min clearance {clear:.3}x radius \
                     (n={n}, radii={radii:?})"
                ));
            }
            // The lap must also be a LAP: it visits every island, so the
            // farthest-from-any-island pose can't be absurdly far out either.
            if clear > 40.0 {
                return Err(format!(
                    "world_lap never approaches an island: min clearance {clear:.1}x radius"
                ));
            }
            // And the hour must sweep continuously around the loop: adjacent
            // frames must not jump (the circular mean is what guarantees this,
            // and the seam must close).
            // The clock must be CONTINUOUS along the lap. Test that as a
            // scaling property, not an absolute per-frame bound: a continuous
            // function sampled twice as finely takes steps half the size, while
            // a discontinuity keeps its jump no matter the sampling. An
            // absolute limit would instead measure how hostile this synthetic
            // config is — and it is deliberately far more hostile than reality
            // (a 20x radius spread; the real islands are each auto-fit to a
            // similar footprint before the merge, so their radii are close).
            //
            // Sample the SHIPPED clock (`path_hour`, window and all), not a
            // reimplementation — a gate against a parallel copy proves nothing
            // about what actually renders.
            let attr = crate::world::attractors(&w);
            let worst_at = |frames: u32| -> f32 {
                let s = world_lap(&w, frames, 30, (640, 360), 1);
                let mut prev: Option<f32> = None;
                let mut worst = 0.0f32;
                for f in 0..frames {
                    let h = path_hour(&s, &attr, f).expect("attractors present");
                    if let Some(p) = prev {
                        let mut d: f32 = (h - p).rem_euclid(24.0);
                        if d > 12.0 {
                            d -= 24.0;
                        }
                        worst = worst.max(d.abs());
                    }
                    prev = Some(h);
                }
                worst
            };
            let (j1, j2) = (worst_at(240), worst_at(480));
            if j1 > 4.0 {
                return Err(format!(
                    "attractor hour swings {j1:.2}h between adjacent tour frames \
                     (n={n}, radii={radii:?})"
                ));
            }
            if j2 > j1 * 0.7 {
                return Err(format!(
                    "tour clock is not continuous: {j2:.3}h/frame at 480 frames vs \
                     {j1:.3}h at 240 did not shrink (n={n}, radii={radii:?})"
                ));
            }
        }
    }

    // The preset table and the arg resolution rule must agree: every listed
    // preset resolves as a preset, and something that is not one does not.
    for p in PRESETS {
        if !is_preset(p) {
            return Err(format!("preset {p} not recognized by is_preset"));
        }
    }
    if is_preset("shots/mine.json") {
        return Err("a path must not resolve as a preset".into());
    }

    // ffmpeg command construction: the pattern, the frame rate and the
    // loop flag are what make the encode reproducible from the manifest alone.
    {
        let sdr = ffmpeg_cmds("capture/tour", "tour", 60, 3840, false);
        let hevc = &sdr[0].1;
        let webp = &sdr[1].1;
        if !hevc.iter().any(|a| a.contains("f_%05d.png")) {
            return Err("ffmpeg cmd lost the frame pattern".into());
        }
        if !hevc.windows(2).any(|w| w[0] == "-c:v" && w[1] == "libx265") {
            return Err("video encode must be HEVC".into());
        }
        // Without hvc1, QuickTime and Safari refuse the file outright.
        if !hevc.windows(2).any(|w| w[0] == "-tag:v" && w[1] == "hvc1") {
            return Err("HEVC in mp4 must carry the hvc1 tag".into());
        }
        if !hevc.windows(2).any(|w| w[0] == "-framerate" && w[1] == "60") {
            return Err("ffmpeg cmd lost the frame rate".into());
        }
        if !webp.windows(2).any(|w| w[0] == "-loop" && w[1] == "0") {
            return Err("webp encode must loop forever".into());
        }
        // The inline loop must be downscaled — a 4K animated WebP is neither
        // renderable by a browser nor welcome in a clone.
        if !webp.iter().any(|a| a.starts_with("scale=1280")) {
            return Err("inline webp must downscale a 4K capture".into());
        }

        // HDR10: the transfer/primaries tags are what stop a player treating
        // PQ frames as SDR, and 10 bits is the format's floor.
        let hdr = ffmpeg_cmds("capture/tour", "tour", 60, 3840, true);
        let h10 = &hdr[0].1;
        if !h10.windows(2).any(|w| w[0] == "-color_trc" && w[1] == "smpte2084") {
            return Err("HDR10 encode must tag the PQ transfer".into());
        }
        if !h10.windows(2).any(|w| w[0] == "-color_primaries" && w[1] == "bt2020") {
            return Err("HDR10 encode must tag Rec.2020 primaries".into());
        }
        if !h10.windows(2).any(|w| w[0] == "-pix_fmt" && w[1] == "yuv420p10le") {
            return Err("HDR10 encode must be 10-bit".into());
        }
        if !h10.iter().any(|a| a.contains("-hdr10.mp4")) {
            return Err("HDR10 output must be named distinctly from the SDR sibling".into());
        }
        // The SDR sibling tone-maps FROM those PQ frames, and a PNG carries no
        // colour metadata — so the input transfer/primaries must be declared
        // BEFORE `-i` or zscale has nothing to convert from and the encode dies
        // with "no path between colorspaces", leaving a zero-byte mp4. This
        // shipped broken; the position of the tags is the whole fix, so pin it.
        let sdr = &hdr[1].1;
        let i_at = sdr
            .iter()
            .position(|a| a == "-i")
            .ok_or("SDR sibling has no input")?;
        for tag in ["-color_trc", "-color_primaries"] {
            match sdr.iter().position(|a| a == tag) {
                Some(p) if p < i_at => {}
                Some(_) => return Err(format!("SDR tone-map: {tag} must precede -i")),
                None => return Err(format!("SDR tone-map must declare {tag} on the input")),
            }
        }
        if !sdr.iter().any(|a| a.contains("tonemap=")) {
            return Err("SDR sibling must actually tone-map".into());
        }
        // A still's HDR path must produce a VIEWABLE, PQ-TAGGED file — and the
        // tag must go through `setparams`, since the output-option form writes
        // "unspecified" into the AVIF colr box (see ffmpeg_still_hdr).
        let still = ffmpeg_still_hdr("a-pq.png", "a.avif");
        let vf = still
            .windows(2)
            .find(|w| w[0] == "-vf")
            .map(|w| w[1].clone())
            .unwrap_or_default();
        if !vf.contains("setparams") || !vf.contains("smpte2084") {
            return Err("HDR still must stamp PQ via setparams, not output options".into());
        }
    }

    // PQ encoding anchors. The curve is `tone::ToneParams::hdr10`, which
    // `tone::self_test` already pins in detail; what matters here is that the
    // 16-bit wire is monotone, spans the range, and puts black at zero.
    {
        let black = pq_rgb16(&[0.0, 0.0, 0.0], 200.0, HDR_MASTER_NITS);
        if black != [0u16, 0, 0] {
            return Err(format!("PQ black must encode to 0, got {black:?}"));
        }
        let a = pq_rgb16(&[0.1, 0.1, 0.1], 200.0, HDR_MASTER_NITS)[0];
        let b = pq_rgb16(&[1.0, 1.0, 1.0], 200.0, HDR_MASTER_NITS)[0];
        let c = pq_rgb16(&[100.0, 100.0, 100.0], 200.0, HDR_MASTER_NITS)[0];
        if !(a < b && b < c) {
            return Err(format!("PQ encoding must be monotone, got {a} {b} {c}"));
        }
        // A physical sun (radiance ~44000) must not wrap or clip to black — the
        // rolloff is asymptotic, so it pins near the top of the range.
        let sun = pq_rgb16(&[44000.0, 44000.0, 44000.0], 200.0, HDR_MASTER_NITS)[0];
        if sun < c || sun == 0 {
            return Err(format!("a physical sun must pin near PQ peak, got {sun}"));
        }
    }

    // Even-dimension rounding: yuv420p cannot encode an odd dimension, and
    // discovering that after a 900-frame render is the expensive way to learn.
    {
        let c = CineOpts { res: Some((1921, 1081)), ..CineOpts::default() };
        if c.res_or((1920, 1080)) != (1920, 1080) {
            return Err("odd capture dimensions must round down to even".into());
        }
    }

    Ok(())
}

/// Is `arg` one of the built-in presets? Anything else is a shot-list path.
pub fn is_preset(arg: &str) -> bool {
    PRESETS.contains(&arg)
}
