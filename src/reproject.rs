//! TAA-style reprojected per-pixel EMA history for OIDN mode: while the
//! camera moves, each fresh 1-spp frame is folded into a history reprojected
//! from the previous frame, so the denoiser input converges across time
//! instead of shimmering at 1 spp. Presentation-side only: reads accum, the
//! G-buffer depth, and the info kinds; it never feeds back into `accum`,
//! `tbuf`, the temporal cache, or any tracing decision. (Unrelated to
//! `temporal.rs`, which caches frustum empty-space claims for the tracer.)
//!
//! Reprojection is recomputed from DEPTH here, deliberately not from
//! `GBufs::mvec`: the MV of a budget-frame coarse quad projects the quad's
//! shared center-ray hit point, so an MV-driven fetch would collapse the
//! whole quad's history to one previous texel; and `project() -> None`
//! (behind the old image plane) is stored as mv (0,0), indistinguishable
//! from zero motion. Reconstructing `p = origin + ray_dir(center)·t` from
//! the per-pixel view-Z fixes both and keeps `FrameCtx::prev_cam` at `None`
//! in OIDN mode — the renderer is untouched.

use crate::camera::CamBasis;
use crate::dlss::GBufs;
use crate::overlay::{self, KIND_COARSE};
use glam::Vec3A;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// History-length cap while the camera moves: the EMA weight of a fresh
/// sample floors at 1/(L_MAX+1), so a moving history tracks lighting changes
/// within ~L_MAX frames while cutting 1-spp variance ~L_MAX×. Static frames
/// (identity reprojection) instead grow to the caller's `still_cap`
/// (MAX_SAMPLES) — statistically the plain accumulation, so stopping the
/// camera converges seamlessly with no mode switch.
pub const L_MAX: f32 = 32.0;
/// Relative view-Z tolerance for history validation. Wide enough to absorb
/// the quad-constant depth a previous-frame coarse quad leaves behind at
/// moderate slopes; tight enough that silhouette disocclusions (which differ
/// by far more than 5%) reject. Worst case of a wrong rejection is a local
/// L reset (briefly noisier input) — never ghosting.
pub const DEPTH_TOL: f32 = 0.05;
/// depth >= SKY_DEPTH_FRAC * far  <=>  sky (write_gbuf_sky stores `far`;
/// same convention as dlss::mv_selftest).
pub const SKY_DEPTH_FRAC: f32 = 0.99;
/// Minimum total bilinear validity weight to accept a fetch: below this the
/// sample would be dominated by renormalization of a near-zero tap.
const W_MIN: f32 = 1e-3;

/// Per-update accounting, used by the interactive stats line and the
/// `--check-oidn` temporal gates.
#[derive(Clone, Copy)]
pub struct UpdateStats {
    /// Valid history blended with the fresh sample.
    pub accepted: u64,
    /// Subset of `accepted`: sky→sky direction reprojections.
    pub sky_accepted: u64,
    /// No valid history: hist = cur, L = 1.
    pub rejected: u64,
    /// KIND_COARSE with valid history: history preserved, flat quad ignored.
    pub coarse_kept: u64,
    /// KIND_COARSE without history: flat color taken at L = 1.
    pub coarse_reset: u64,
    /// The static (bit-equal basis) fast path was taken.
    pub identity: bool,
    pub len_min: f32,
    pub len_max: f32,
}

impl Default for UpdateStats {
    fn default() -> Self {
        Self {
            accepted: 0,
            sky_accepted: 0,
            rejected: 0,
            coarse_kept: 0,
            coarse_reset: 0,
            identity: false,
            // Min/max identities so parallel row-stats merge cleanly.
            len_min: f32::INFINITY,
            len_max: 0.0,
        }
    }
}

impl UpdateStats {
    fn merge(a: Self, b: Self) -> Self {
        Self {
            accepted: a.accepted + b.accepted,
            sky_accepted: a.sky_accepted + b.sky_accepted,
            rejected: a.rejected + b.rejected,
            coarse_kept: a.coarse_kept + b.coarse_kept,
            coarse_reset: a.coarse_reset + b.coarse_reset,
            identity: a.identity || b.identity,
            len_min: a.len_min.min(b.len_min),
            len_max: a.len_max.max(b.len_max),
        }
    }
}

pub struct History {
    w: usize,
    h: usize,
    /// Pixel capacity fixed at construction; `set_res` reinterprets within it
    /// (the `GBufs::set_res` pattern — no per-step realloc).
    cap: usize,
    color: Vec<f32>,    // cap*3 EMA history, front (read); current res prefix
    color_bk: Vec<f32>, // back (write) — the update is a gather, so ping-pong
    len: Vec<f32>,      // cap effective sample count in the stored history
    len_bk: Vec<f32>,
    /// 1 where the last update's fetch succeeded (accepted or coarse_kept).
    /// Written every update; consumed by the check gates.
    mask: Vec<u8>,
    /// Linear view-Z of the previous frame (sky = far).
    prev_depth: Vec<f32>,
    /// Basis of the frame the history was last updated with. `None` <=> no
    /// usable history (every pixel resets on the next update). Deliberately
    /// its own state — not `dlss_prev`, not `tprev_basis` (different
    /// contracts).
    prev_basis: Option<CamBasis>,
}

impl History {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            cap: w * h,
            color: vec![0.0; w * h * 3],
            color_bk: vec![0.0; w * h * 3],
            len: vec![0.0; w * h],
            len_bk: vec![0.0; w * h],
            mask: vec![0; w * h],
            prev_depth: vec![0.0; w * h],
            prev_basis: None,
        }
    }

    /// Drop the history: the next `update` resets every pixel. Call on any
    /// setting change that alters shading (quality, hybrid, bounce mode, mode
    /// toggles) — NOT on camera motion, which is the case this exists for.
    pub fn invalidate(&mut self) {
        self.prev_basis = None;
    }

    /// Reinterpret the buffers at a new resolution within capacity, without
    /// reallocating — a res change invalidates by definition (reprojection
    /// across a res change would mix pixel grids), so the stale contents are
    /// never read. Called per distinct-res frame during a DRS ramp in XeSS
    /// mode; a realloc here was a ~77 MB hit per ramp frame.
    pub fn set_res(&mut self, w: usize, h: usize) {
        assert!(w * h <= self.cap, "History::set_res beyond capacity");
        if (w, h) != (self.w, self.h) {
            self.w = w;
            self.h = h;
            self.invalidate();
        }
    }

    /// The current history image (linear HDR, 3 floats/px) — the OIDN input.
    pub fn color(&self) -> &[f32] {
        &self.color[..self.w * self.h * 3]
    }

    /// The resolution the history currently holds (see `set_res`).
    pub fn res(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    /// Effective per-pixel sample counts (check gates).
    pub fn sample_counts(&self) -> &[f32] {
        &self.len[..self.w * self.h]
    }

    /// Last update's per-pixel fetch-succeeded mask (check gates).
    pub fn mask(&self) -> &[u8] {
        &self.mask[..self.w * self.h]
    }

    /// Fold one fresh frame into the history. `accum` holds the frame's
    /// 1-spp linear radiance (f32 bit patterns), `g` the matching G-buffers
    /// (only `depth` is read), `info` the per-pixel quadtree kind, `far` the
    /// sky depth sentinel (`dlss::near_far(scene.diag).1`), `still_cap` the
    /// history-length cap on the static identity path (MAX_SAMPLES).
    pub fn update(
        &mut self,
        cam: &CamBasis,
        accum: &[AtomicU32],
        g: &GBufs,
        info: &[AtomicU32],
        far: f32,
        still_cap: f32,
    ) -> UpdateStats {
        let (w, h) = (self.w, self.h);
        assert_eq!((g.rw, g.rh), (w, h), "gbuf/history resolution mismatch");
        assert_eq!(accum.len(), w * h * 3);
        assert_eq!(info.len(), w * h);

        let have_prev = self.prev_basis.is_some();
        let identity = self.prev_basis.as_ref() == Some(cam);
        let cap = if identity { still_cap } else { L_MAX };
        let prev = self.prev_basis.unwrap_or(*cam); // read only when have_prev
        let fwd = cam.forward();
        let sky_z = SKY_DEPTH_FRAC * far;
        // Current-res prefixes: the vecs are capacity-sized (see set_res).
        let (color, len, prev_depth) =
            (&self.color[..w * h * 3], &self.len[..w * h], &self.prev_depth[..w * h]);

        // Bilinear history fetch at continuous pixel coords with per-tap
        // validity (in-bounds, sky<->sky / geom<->geom, relative view-Z
        // agreement); invalid taps drop out and the rest renormalize, which
        // keeps bilinear from bleeding across depth edges. `z_exp` is the
        // expected previous view-Z of the reprojected point (ignored for
        // sky). Returns None when no usable weight remains.
        let fetch_at = |px: f32, py: f32, sky: bool, z_exp: f32| -> Option<(Vec3A, f32)> {
            let (bx, by) = (px - 0.5, py - 0.5);
            let (x0f, y0f) = (bx.floor(), by.floor());
            let (fx, fy) = (bx - x0f, by - y0f);
            let (x0, y0) = (x0f as i64, y0f as i64);
            let taps = [
                (0i64, 0i64, (1.0 - fx) * (1.0 - fy)),
                (1, 0, fx * (1.0 - fy)),
                (0, 1, (1.0 - fx) * fy),
                (1, 1, fx * fy),
            ];
            let (mut c, mut l, mut sw) = (Vec3A::ZERO, 0.0f32, 0.0f32);
            for (dx, dy, wt) in taps {
                let (tx, ty) = (x0 + dx, y0 + dy);
                if wt <= 0.0 || tx < 0 || ty < 0 || tx >= w as i64 || ty >= h as i64 {
                    continue;
                }
                let ti = ty as usize * w + tx as usize;
                let pz = prev_depth[ti];
                let valid = if sky {
                    pz >= sky_z
                } else {
                    pz < sky_z && (z_exp - pz).abs() <= DEPTH_TOL * z_exp.max(pz)
                };
                if !valid {
                    continue;
                }
                c += Vec3A::new(color[ti * 3], color[ti * 3 + 1], color[ti * 3 + 2]) * wt;
                l += len[ti] * wt;
                sw += wt;
            }
            if sw < W_MIN { None } else { Some((c / sw, l / sw)) }
        };

        let mut stats = self.color_bk[..w * h * 3]
            .par_chunks_mut(w * 3)
            .zip(self.len_bk[..w * h].par_chunks_mut(w))
            .zip(self.mask[..w * h].par_chunks_mut(w))
            .enumerate()
            .map(|(y, ((crow, lrow), mrow))| {
                let mut s = UpdateStats::default();
                for x in 0..w {
                    let i = y * w + x;
                    let cur = Vec3A::new(
                        f32::from_bits(accum[i * 3].load(Relaxed)),
                        f32::from_bits(accum[i * 3 + 1].load(Relaxed)),
                        f32::from_bits(accum[i * 3 + 2].load(Relaxed)),
                    );
                    let kind = overlay::info_kind(info[i].load(Relaxed));
                    let z = f32::from_bits(g.depth[i].load(Relaxed));
                    let is_sky = z >= sky_z;

                    let fetched = if !have_prev {
                        None
                    } else if identity {
                        // Same basis + static scene => same texel, exactly.
                        Some((
                            Vec3A::new(color[i * 3], color[i * 3 + 1], color[i * 3 + 2]),
                            len[i],
                        ))
                    } else {
                        // Unnormalized on purpose: `project` is a ratio of
                        // dots and `origin + dir·(z / dir·fwd)` cancels the
                        // length — identical points, no per-pixel sqrt.
                        let dir = cam.ray_dir_unnorm(x as f32 + 0.5, y as f32 + 0.5);
                        if is_sky {
                            // Environment at infinity: direction-only, exact.
                            prev.project(dir).and_then(|(px, py)| fetch_at(px, py, true, 0.0))
                        } else {
                            let t = z / dir.dot(fwd);
                            let p = cam.origin + dir * t;
                            let d = p - prev.origin;
                            let z_exp = d.dot(prev.forward());
                            if z_exp <= 0.0 {
                                None // behind the old image plane
                            } else {
                                prev.project(d)
                                    .and_then(|(px, py)| fetch_at(px, py, false, z_exp))
                            }
                        }
                    };

                    let (out, l_new) = match fetched {
                        Some((fetch, l_fetch)) if kind == KIND_COARSE => {
                            // A flat-filled budget quad carries strictly less
                            // information than valid reprojected history:
                            // keep the history, blend weight 0.
                            s.coarse_kept += 1;
                            (fetch, l_fetch.min(cap))
                        }
                        Some((fetch, l_fetch)) => {
                            s.accepted += 1;
                            if is_sky {
                                s.sky_accepted += 1;
                            }
                            let l_eff = l_fetch.min(cap - 1.0);
                            ((fetch * l_eff + cur) / (l_eff + 1.0), l_eff + 1.0)
                        }
                        None => {
                            if kind == KIND_COARSE {
                                s.coarse_reset += 1;
                            } else {
                                s.rejected += 1;
                            }
                            (cur, 1.0)
                        }
                    };
                    mrow[x] = fetched.is_some() as u8;
                    crow[x * 3] = out.x;
                    crow[x * 3 + 1] = out.y;
                    crow[x * 3 + 2] = out.z;
                    lrow[x] = l_new;
                    s.len_min = s.len_min.min(l_new);
                    s.len_max = s.len_max.max(l_new);
                }
                s
            })
            .reduce(UpdateStats::default, UpdateStats::merge);
        stats.identity = identity;

        std::mem::swap(&mut self.color, &mut self.color_bk);
        std::mem::swap(&mut self.len, &mut self.len_bk);
        self.prev_depth[..w * h].par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, v) in row.iter_mut().enumerate() {
                *v = f32::from_bits(g.depth[y * w + x].load(Relaxed));
            }
        });
        self.prev_basis = Some(*cam);
        stats
    }
}

/// Closed-form self-test on synthetic 8×8 buffers (the `sphcell::self_test`
/// precedent): projection roundtrip, static replay = exact running mean,
/// analytic integer-pixel strafe, behind-plane rejection, depth-mismatch
/// rejection, the coarse keep/reset rules, and L-accounting. Runs in both
/// `--check` and `--check-oidn`; sub-millisecond, no scene, no wall clock.
pub fn self_test() -> Result<(), String> {
    use crate::camera::Camera;
    const N: usize = 8;
    let far = 100.0f32;
    let z0 = 10.0f32;
    let cam0 = Camera::look_at(Vec3A::ZERO, Vec3A::X, 60f32.to_radians());
    let basis0 = cam0.basis(N, N);

    // 1. Projection roundtrip: project(ray_dir(fx, fy)) == (fx, fy), for
    //    both the normalized and the unnormalized direction (project must be
    //    scale-invariant — that is what licenses the sqrt skip in `update`).
    for iy in 0..=4 {
        for ix in 0..=4 {
            let (fx, fy) = (ix as f32 * 2.0, iy as f32 * 2.0);
            for dir in [basis0.ray_dir(fx, fy), basis0.ray_dir_unnorm(fx, fy)] {
                let (px, py) = basis0
                    .project(dir)
                    .ok_or_else(|| format!("roundtrip: project None at ({fx}, {fy})"))?;
                if (px - fx).abs() > 1e-3 || (py - fy).abs() > 1e-3 {
                    return Err(format!("roundtrip: ({fx}, {fy}) -> ({px}, {py})"));
                }
            }
        }
    }

    let alloc_atomic = |n: usize| -> Vec<AtomicU32> { (0..n).map(|_| AtomicU32::new(0)).collect() };
    let g = GBufs::new(N, N);
    let info = alloc_atomic(N * N);
    let accum = alloc_atomic(N * N * 3);
    let set_depth_all = |z: f32| {
        for d in &g.depth {
            d.store(z.to_bits(), Relaxed);
        }
    };
    // Distinct per-pixel colors: c(i, k) = base + i*3 + k, exactly
    // representable so mean identities hold to fp rounding only.
    let fill_accum = |base: f32| {
        for (i, a) in accum.iter().enumerate() {
            a.store((base + i as f32).to_bits(), Relaxed);
        }
    };
    let check_l_accounting = |hist: &History, s: &UpdateStats, what: &str| -> Result<(), String> {
        let ones = hist.sample_counts().iter().filter(|&&l| l == 1.0).count() as u64;
        let lo = s.rejected + s.coarse_reset;
        let hi = lo + s.coarse_kept; // coarse_kept may legitimately carry L=1
        if ones < lo || ones > hi {
            return Err(format!("{what}: #(L==1) = {ones}, expected in [{lo}, {hi}]"));
        }
        Ok(())
    };

    // 2. Static replay: frame 1 all-reject (cold), frame 2 identity path,
    //    hist == (c0 + c1)/2 exactly (within fp), L == 2 everywhere.
    let mut hist = History::new(N, N);
    set_depth_all(z0);
    fill_accum(1.0);
    let s1 = hist.update(&basis0, &accum, &g, &info, far, 1024.0);
    if s1.rejected != (N * N) as u64 || s1.identity {
        return Err(format!("static f1: rejected {} identity {}", s1.rejected, s1.identity));
    }
    check_l_accounting(&hist, &s1, "static f1")?;
    fill_accum(101.0);
    let s2 = hist.update(&basis0, &accum, &g, &info, far, 1024.0);
    if !s2.identity || s2.rejected != 0 || s2.accepted != (N * N) as u64 {
        return Err(format!("static f2: identity {} rejected {}", s2.identity, s2.rejected));
    }
    if s2.len_min != 2.0 || s2.len_max != 2.0 {
        return Err(format!("static f2: L range [{}, {}], want [2, 2]", s2.len_min, s2.len_max));
    }
    for (i, &v) in hist.color().iter().enumerate() {
        let want = (1.0 + i as f32 + 101.0 + i as f32) * 0.5;
        if (v - want).abs() > 1e-4 * want.abs().max(1.0) {
            return Err(format!("static f2: hist[{i}] = {v}, want {want}"));
        }
    }

    // 3. Analytic strafe: move the camera along its screen-x axis by the
    //    amount that shifts a constant view-Z plane exactly 2 pixels. The
    //    shift is linear in the offset, so one probe calibrates it without
    //    reaching into CamBasis internals.
    let right_world = cam0.forward().cross(Vec3A::Y).normalize();
    // Probe with the roles `update` uses: the CURRENT camera sits one unit
    // right of PREV; the point at pixel (0.5, 0.5), view-Z z0 from current,
    // projects into prev at 0.5 + px_per_unit.
    let dir0 = basis0.ray_dir(0.5, 0.5);
    let p_probe = (basis0.origin + right_world) + dir0 * (z0 / dir0.dot(basis0.forward()));
    let (ppx, _) = basis0
        .project(p_probe - basis0.origin)
        .ok_or("strafe probe: project None")?;
    let px_per_unit = ppx - 0.5;
    let k = 2.0f32;
    let strafe = Camera { pos: cam0.pos + right_world * (k / px_per_unit), ..cam0 };
    let basis_b = strafe.basis(N, N);

    let mut hist = History::new(N, N);
    fill_accum(1.0);
    hist.update(&basis0, &accum, &g, &info, far, 1024.0); // warm at A
    fill_accum(1001.0);
    let s3 = hist.update(&basis_b, &accum, &g, &info, far, 1024.0);
    // Expected split: pixel (x, y) at view-Z z0 seen from B projects into A
    // at x + k (the plane slid k pixels); columns whose source is off-screen
    // reject. Verify counts and one interior pixel's exact blend.
    let mut want_acc = 0u64;
    for x in 0..N {
        let src = x as f32 + k;
        if src < N as f32 {
            want_acc += 1;
        }
    }
    let want_acc = want_acc * N as u64;
    if s3.accepted != want_acc || s3.rejected != (N * N) as u64 - want_acc {
        return Err(format!(
            "strafe: accepted {} rejected {}, want {} / {}",
            s3.accepted,
            s3.rejected,
            want_acc,
            (N * N) as u64 - want_acc
        ));
    }
    check_l_accounting(&hist, &s3, "strafe")?;
    // Interior pixel (2, 3): fetches A's pixel (4, 3) = old color, blends
    // with the fresh sample at L=1 -> mean of the two.
    {
        let (x, y) = (2usize, 3usize);
        let src_i = y * N + x + k as usize;
        let cur_i = y * N + x;
        for ch in 0..3 {
            let old = 1.0 + (src_i * 3 + ch) as f32;
            let new = 1001.0 + (cur_i * 3 + ch) as f32;
            let want = (old + new) * 0.5;
            let got = hist.color()[cur_i * 3 + ch];
            if (got - want).abs() > 1e-3 * want {
                return Err(format!("strafe blend: px({x},{y})[{ch}] = {got}, want {want}"));
            }
        }
    }

    // 4. Behind-plane: dolly the camera backward past the stored geometry —
    //    every reprojected point lands behind the old image plane and must
    //    reject (never fetch).
    let mut hist = History::new(N, N);
    fill_accum(1.0);
    hist.update(&basis0, &accum, &g, &info, far, 1024.0);
    let back = Camera { pos: cam0.pos - cam0.forward() * (z0 + 1.0), ..cam0 };
    let s4 = hist.update(&back.basis(N, N), &accum, &g, &info, far, 1024.0);
    if s4.rejected != (N * N) as u64 {
        return Err(format!("behind-plane: rejected {}, want {}", s4.rejected, N * N));
    }

    // 5. Depth rejection: sub-strafe (non-identity), poke one prev-depth
    //    texel to 2x — exactly its consumer must lose that tap and reject.
    let mut hist = History::new(N, N);
    fill_accum(1.0);
    hist.update(&basis0, &accum, &g, &info, far, 1024.0);
    let poke = 3usize * N + 5; // texel (5, 3)
    hist.prev_depth[poke] = z0 * 2.0;
    let s5 = hist.update(&basis_b, &accum, &g, &info, far, 1024.0);
    // Consumer of (5, 3) under the k-pixel shift is (5 - k, 3).
    let consumer = 3 * N + (5 - k as usize);
    if hist.mask()[consumer] != 0 {
        return Err("depth poke: consumer still accepted".into());
    }
    if s5.rejected <= s3.rejected {
        return Err(format!("depth poke: rejected {} not > baseline {}", s5.rejected, s3.rejected));
    }

    // 6. Coarse rules: with valid history the flat quad is ignored (history
    //    bit-equal, L unchanged); after invalidate() it is taken at L = 1.
    let mut hist = History::new(N, N);
    fill_accum(1.0);
    hist.update(&basis0, &accum, &g, &info, far, 1024.0);
    fill_accum(101.0);
    hist.update(&basis0, &accum, &g, &info, far, 1024.0); // L = 2 everywhere
    let coarse_i = 4 * N + 4;
    info[coarse_i].store(overlay::pack_info(3, KIND_COARSE), Relaxed);
    fill_accum(5001.0);
    let before: [f32; 3] = std::array::from_fn(|ch| hist.color()[coarse_i * 3 + ch]);
    let s6 = hist.update(&basis0, &accum, &g, &info, far, 1024.0);
    if s6.coarse_kept != 1 {
        return Err(format!("coarse: kept {}, want 1", s6.coarse_kept));
    }
    for ch in 0..3 {
        if hist.color()[coarse_i * 3 + ch] != before[ch] {
            return Err("coarse: history changed under a kept flat quad".into());
        }
    }
    if hist.sample_counts()[coarse_i] != 2.0 {
        return Err(format!("coarse: L = {}, want 2 (unchanged)", hist.sample_counts()[coarse_i]));
    }
    check_l_accounting(&hist, &s6, "coarse kept")?;
    hist.invalidate();
    let s7 = hist.update(&basis0, &accum, &g, &info, far, 1024.0);
    if s7.coarse_reset != 1 || s7.rejected != (N * N - 1) as u64 {
        return Err(format!(
            "coarse reset: coarse_reset {} rejected {}",
            s7.coarse_reset, s7.rejected
        ));
    }
    if hist.color()[coarse_i * 3] != 5001.0 + (coarse_i * 3) as f32
        || hist.sample_counts()[coarse_i] != 1.0
    {
        return Err("coarse reset: flat color not taken at L = 1".into());
    }
    check_l_accounting(&hist, &s7, "coarse reset")?;

    // 7. set_res: reinterpret within capacity — the accessor lengths track
    //    the current res (not capacity), any res CHANGE invalidates (the
    //    next update all-resets), and a same-res set_res keeps the history.
    info[coarse_i].store(0, Relaxed); // undo section 6's coarse texel
    let mut hist = History::new(N, N);
    fill_accum(1.0);
    hist.update(&basis0, &accum, &g, &info, far, 1024.0);
    fill_accum(101.0);
    hist.update(&basis0, &accum, &g, &info, far, 1024.0); // L = 2 everywhere
    hist.set_res(N, N); // no-op: same res must NOT invalidate
    let s8 = hist.update(&basis0, &accum, &g, &info, far, 1024.0);
    if !s8.identity || s8.rejected != 0 {
        return Err("set_res: same-res set_res invalidated the history".into());
    }
    let m = N / 2;
    hist.set_res(m, m);
    if hist.res() != (m, m)
        || hist.color().len() != m * m * 3
        || hist.sample_counts().len() != m * m
        || hist.mask().len() != m * m
    {
        return Err("set_res: accessor lengths did not track the res".into());
    }
    let g2 = GBufs::new(m, m);
    for d in &g2.depth {
        d.store(z0.to_bits(), Relaxed);
    }
    let info2 = alloc_atomic(m * m);
    let accum2 = alloc_atomic(m * m * 3);
    for (i, a) in accum2.iter().enumerate() {
        a.store((7.0 + i as f32).to_bits(), Relaxed);
    }
    let s9 = hist.update(&cam0.basis(m, m), &accum2, &g2, &info2, far, 1024.0);
    if s9.identity || s9.rejected != (m * m) as u64 {
        return Err(format!(
            "set_res: shrink did not invalidate (identity {}, rejected {})",
            s9.identity, s9.rejected
        ));
    }
    if hist.sample_counts().iter().any(|&l| l != 1.0) {
        return Err("set_res: post-shrink L != 1".into());
    }
    hist.set_res(N, N); // back up within capacity: also a res change
    let s10 = hist.update(&basis0, &accum, &g, &info, far, 1024.0);
    if s10.identity || s10.rejected != (N * N) as u64 {
        return Err(format!(
            "set_res: grow-back did not invalidate (identity {}, rejected {})",
            s10.identity, s10.rejected
        ));
    }

    Ok(())
}
