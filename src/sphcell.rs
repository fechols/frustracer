//! Spherical-cell math for the hemisphere bounce integrator — the direction
//! sphere analog of the screen quadtree. Cells are spherical triangles whose
//! edges are great-circle arcs, i.e. traces of planes through the apex, so a
//! cell is exactly a 3-plane `TileFrustum` and every frustum invariant applies
//! unchanged. Midpoint subdivision keeps edges great circles and the 4
//! children exactly partition the parent (the containment that makes
//! inherited (tc, cut) sound, mirroring the pixel-quadrant argument).
//!
//! Vertex order convention: counter-clockwise seen from outside the sphere
//! (the octant frame `(n, t1, t2)` with `t1 × t2 = n` is CCW, and `split`
//! preserves the order), so `psa` comes out positive for hemisphere cells.

use glam::Vec3A;
use std::f32::consts::PI;

/// Solid angle of the spherical triangle (a, b, c), unit vectors.
/// Van Oosterom–Strackee: numerically stable for the tiny cells deep
/// subdivision produces, where Girard's angle-sum formula cancels
/// catastrophically. Orientation-free (absolute value).
pub fn solid_angle(a: Vec3A, b: Vec3A, c: Vec3A) -> f32 {
    let num = a.dot(b.cross(c)).abs();
    let den = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    2.0 * num.atan2(den)
}

/// Lambert's formula: the exact projected solid angle ∫ (d·axis) dω of the
/// spherical triangle onto the plane ⊥ `axis` — the cosine-weighted measure
/// the hemisphere integrals are taken in. Per great-circle edge: arc length ×
/// (edge-plane normal · axis), halved. Signed; positive for CCW cells inside
/// the `axis` hemisphere. Anchors: octant → π/4, whole hemisphere → π.
pub fn psa(a: Vec3A, b: Vec3A, c: Vec3A, axis: Vec3A) -> f32 {
    let mut sum = 0.0;
    for (u, v) in [(a, b), (b, c), (c, a)] {
        let cr = u.cross(v);
        let l = cr.length();
        if l > 1e-12 {
            // atan2(|u×v|, u·v) is the arc length, stable at both ends.
            sum += l.atan2(u.dot(v)) * (cr.dot(axis) / l);
        }
    }
    0.5 * sum
}

/// Great-circle edge midpoints — the 4-way split shared by the integrator and
/// the self-tests. Children `(a,mab,mca) (mab,b,mbc) (mca,mbc,c) (mab,mbc,mca)`
/// exactly partition the parent and preserve vertex orientation.
#[inline]
pub fn midpoints(a: Vec3A, b: Vec3A, c: Vec3A) -> (Vec3A, Vec3A, Vec3A) {
    ((a + b).normalize(), (b + c).normalize(), (c + a).normalize())
}

/// Spherical centroid (normalized vertex mean) — strictly inside the cell.
#[inline]
pub fn centroid(a: Vec3A, b: Vec3A, c: Vec3A) -> Vec3A {
    (a + b + c).normalize_or_zero()
}

/// Uniform sample inside the spherical triangle (Arvo '95, the PBRT-v4
/// formulation). The construction stays inside the cell: the returned
/// direction lies on the arc B→C' with C' on the arc A→C, and spherical
/// triangles smaller than a hemisphere are geodesically convex. Boundary fp
/// error (~1e-7) is far below the frustum plane test's inclusive slack
/// (~1e-5), so a boundary sample is still covered by the cell's (tc, cut)
/// claim. Estimator: ∫ f dω ≈ f(sample) · solid_angle(cell).
pub fn sample_tri(a: Vec3A, b: Vec3A, c: Vec3A, u0: f32, u1: f32) -> Vec3A {
    let n_ab = a.cross(b);
    let n_bc = b.cross(c);
    let n_ca = c.cross(a);
    if n_ab.length_squared() < 1e-18
        || n_bc.length_squared() < 1e-18
        || n_ca.length_squared() < 1e-18
    {
        return centroid(a, b, c); // degenerate (repeated vertex)
    }
    let n_ab = n_ab.normalize();
    let n_bc = n_bc.normalize();
    let n_ca = n_ca.normalize();
    // Interior vertex angles; Girard area only steers the sub-area split, so
    // its cancellation for tiny cells degrades stratification, not soundness.
    let alpha = angle_between(n_ab, -n_ca);
    let beta = angle_between(n_bc, -n_ab);
    let gamma = angle_between(n_ca, -n_bc);
    let area = alpha + beta + gamma - PI;
    if area < 1e-7 {
        return centroid(a, b, c);
    }
    // Sub-triangle sharing vertex A and edge AB with area u0·area: solve for
    // its third vertex C' on the arc A→C. The working angle is
    // φ = ap + π − α (the +π is Arvo's parameterization — dropping it mirrors
    // C' through the apex plane and samples the wrong hemisphere).
    let ap = u0 * area;
    let (sin_ap, cos_ap) = ap.sin_cos();
    let (sin_alpha, cos_alpha) = alpha.sin_cos();
    let sin_phi = sin_alpha * cos_ap - sin_ap * cos_alpha; // sin(ap + π − α)
    let cos_phi = -(cos_ap * cos_alpha + sin_ap * sin_alpha); // cos(ap + π − α)
    let k1 = cos_phi + cos_alpha;
    let k2 = sin_phi - sin_alpha * a.dot(b); // a·b = cos(arc AB)
    let mut cos_bp = (k2 + (k2 * cos_phi - k1 * sin_phi) * cos_alpha)
        / ((k2 * sin_phi + k1 * cos_phi) * sin_alpha);
    if !cos_bp.is_finite() {
        cos_bp = 1.0; // degenerate slice → C' = A, still inside
    }
    let cos_bp = cos_bp.clamp(-1.0, 1.0);
    let sin_bp = (1.0 - cos_bp * cos_bp).max(0.0).sqrt();
    let cp = (a * cos_bp + gram_schmidt(c, a) * sin_bp).normalize();
    // Uniform along the arc B→C'.
    let cos_theta = 1.0 - u1 * (1.0 - cp.dot(b));
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    (b * cos_theta + gram_schmidt(cp, b) * sin_theta).normalize()
}

/// v with its w-component removed, normalized (w unit).
#[inline]
fn gram_schmidt(v: Vec3A, w: Vec3A) -> Vec3A {
    (v - w * v.dot(w)).normalize_or_zero()
}

/// Angle between unit vectors, stable near both 0 and π.
#[inline]
fn angle_between(u: Vec3A, v: Vec3A) -> f32 {
    if u.dot(v) < 0.0 {
        PI - 2.0 * (0.5 * (u + v).length()).clamp(-1.0, 1.0).asin()
    } else {
        2.0 * (0.5 * (v - u).length()).clamp(-1.0, 1.0).asin()
    }
}

/// The 4 spherical octants covering the hemisphere around `n`, in the CCW
/// vertex order the psa sign convention requires. `(t1, t2)` is any
/// right-handed tangent frame (`t1 × t2 = n`).
pub fn octants(n: Vec3A, t1: Vec3A, t2: Vec3A) -> [[Vec3A; 3]; 4] {
    [
        [n, t1, t2],
        [n, t2, -t1],
        [n, -t1, -t2],
        [n, -t2, t1],
    ]
}

/// Closed-form identity checks, run by `--check`. Exercises two frames (one
/// axis-aligned, one tilted) so no test passes by cardinal-axis accident.
pub fn self_test() -> Result<(), String> {
    for n in [Vec3A::Y, Vec3A::new(0.3, -0.8, 0.5).normalize()] {
        let (t1, t2) = crate::shade::onb(n);
        // onb must be right-handed for the octant orientation convention.
        if (t1.cross(t2) - n).length() > 1e-5 {
            return Err(format!("onb({n}) is not right-handed"));
        }
        let cells = octants(n, t1, t2);

        // Octant anchors: Ω = π/2, PSA = π/4; hemisphere totals 2π and π.
        let mut om_sum = 0.0;
        let mut psa_sum = 0.0;
        for [a, b, c] in cells {
            let om = solid_angle(a, b, c);
            let pr = psa(a, b, c, n);
            if (om - PI / 2.0).abs() > 1e-5 {
                return Err(format!("octant solid angle {om}, want {}", PI / 2.0));
            }
            if (pr - PI / 4.0).abs() > 1e-5 {
                return Err(format!("octant psa {pr}, want {}", PI / 4.0));
            }
            om_sum += om;
            psa_sum += pr;
        }
        if (om_sum - 2.0 * PI).abs() > 1e-4 || (psa_sum - PI).abs() > 1e-4 {
            return Err(format!("hemisphere Ω {om_sum} (want 2π) / PSA {psa_sum} (want π)"));
        }

        // Subdivision: children exactly partition the parent (Ω and PSA), and
        // the totals survive to depth 3 (4·64 cells): Σ Ω = 2π, Σ PSA = π.
        let mut om_d = 0.0f64;
        let mut psa_d = 0.0f64;
        let mut stack: Vec<([Vec3A; 3], u32)> = cells.iter().map(|c| (*c, 0)).collect();
        while let Some(([a, b, c], depth)) = stack.pop() {
            if depth == 3 {
                om_d += solid_angle(a, b, c) as f64;
                psa_d += psa(a, b, c, n) as f64;
                continue;
            }
            let (mab, mbc, mca) = midpoints(a, b, c);
            let kids = [[a, mab, mca], [mab, b, mbc], [mca, mbc, c], [mab, mbc, mca]];
            let om_kids: f32 = kids.iter().map(|k| solid_angle(k[0], k[1], k[2])).sum();
            let om_par = solid_angle(a, b, c);
            if (om_kids - om_par).abs() > 1e-4 * om_par.max(1e-3) {
                return Err(format!("split Ω mismatch: parent {om_par}, children {om_kids}"));
            }
            for k in kids {
                stack.push((k, depth + 1));
            }
        }
        if (om_d - 2.0 * std::f64::consts::PI).abs() > 1e-4 {
            return Err(format!("depth-3 Σ Ω {om_d}, want 2π"));
        }
        if (psa_d - std::f64::consts::PI).abs() > 1e-4 {
            return Err(format!("depth-3 Σ PSA {psa_d}, want π"));
        }

        // Sampling: unit length, inside the cell, and above the horizon —
        // checked on octants and their children. Containment is tested
        // against the REAL `TileFrustum::tri_cell` the integrator queries
        // with (not a hand-rebuilt copy), so a divergence in tri_cell's
        // orientation fix or normalization fails here.
        let mut check_cells: Vec<[Vec3A; 3]> = cells.to_vec();
        for [a, b, c] in cells {
            let (mab, mbc, mca) = midpoints(a, b, c);
            check_cells.extend([[a, mab, mca], [mab, b, mbc], [mca, mbc, c], [mab, mbc, mca]]);
        }
        for [a, b, c] in check_cells {
            let cell = crate::frustum::TileFrustum::tri_cell(Vec3A::ZERO, a, b, c);
            for u0 in [0.01f32, 0.33, 0.5, 0.77, 0.99] {
                for u1 in [0.02f32, 0.4, 0.6, 0.98] {
                    let d = sample_tri(a, b, c, u0, u1);
                    if (d.length() - 1.0).abs() > 1e-5 {
                        return Err(format!("sample not unit: {d}"));
                    }
                    if d.dot(n) < -1e-6 {
                        return Err(format!("sample below horizon: {d}"));
                    }
                    if !cell.contains_dir(d, 1e-5) {
                        return Err(format!("sample outside cell: d {d}, u ({u0},{u1})"));
                    }
                }
            }
        }
    }
    Ok(())
}
