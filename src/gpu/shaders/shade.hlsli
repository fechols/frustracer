// Scene resources + the shade.rs port. Requires trace_common.hlsli +
// rt.hlsli pasted first (trace.rs does the concatenation).
//
// This is shade.rs::shade with VisCtl::Off — the exact interactive non-bounce
// path — with the depth-0/depth-1 recursion flattened into a two-lap loop
// (the CPU recursion is structurally depth <= 1: one GGX VNDF reflection
// bounce, and the recursive call runs with reflections ineligible). Every ray
// here goes through DXR inline RayQuery against the TLAS; secondary rays use
// tmin = 0 from the eps-offset surface point, exactly like the CPU (the
// quadtree's inherited tmin is a primary-frustum property — it must never
// leak into shading; see CLAUDE.md invariants). Hemi mode (fb_mode > 0)
// splits the PRIMARY surface's ambient out: shade_split returns the
// ambient-free color plus the kd weight and surface point the hemi wavefront
// consumes; the reflected hit (lap 1) always shades with its own sampled
// ambient (the CPU forces fb OFF at depth 1).

// t0 (bvh nodes) and t1 (tri_idx) belong to the frustum kernels.
StructuredBuffer<float3> positions : register(t2);
StructuredBuffer<float3> normals   : register(t3);
StructuredBuffer<uint>   indices   : register(t4); // flat, 3 per tri (also the BLAS index buffer)
StructuredBuffer<uint>   tri_mat   : register(t5);

#define MAT_DIFFUSE 0u
#define MAT_MARBLE  1u

// Mirrors trace.rs::GpuMat field-for-field (48 B) — a stride skew reads
// garbage; the two must move in the same commit.
struct Mat {
    float3 albedo;
    float roughness;
    float metallic;
    float anisotropy;
    uint kind;
    float scale; // marble feature frequency
    float sheen;
    float translucency;
    float transmission; // not consumed: GPU transmission not yet ported
    float _pad;
};
StructuredBuffer<Mat> materials : register(t6);

// --- shade.rs helper ports ----------------------------------------------------

float3 cosine_dir(float3 n, float3 t1, float3 t2, float r1, float r2) {
    float phi = TAU * r1;
    float sq = sqrt(r2);
    return t1 * (cos(phi) * sq) + t2 * (sin(phi) * sq) + n * sqrt(1.0 - r2);
}

void ggx_alphas(float roughness, float anisotropy, out float ax, out float ay) {
    const float MIN_ALPHA = 5e-3;
    float alpha = roughness * roughness;
    float aspect = sqrt(1.0 - 0.9 * anisotropy);
    ax = max(alpha / aspect, MIN_ALPHA);
    ay = max(alpha * aspect, MIN_ALPHA);
}

// Smith lambda for anisotropic GGX; w in tangent space, w.z > 0.
float ggx_lambda(float3 w, float ax, float ay) {
    float t = ((ax * w.x) * (ax * w.x) + (ay * w.y) * (ay * w.y)) / (w.z * w.z);
    return (sqrt(1.0 + t) - 1.0) * 0.5;
}

float ggx_ndf(float3 h, float ax, float ay) {
    float hx = h.x / ax;
    float hy = h.y / ay;
    float d = hx * hx + hy * hy + h.z * h.z;
    return 1.0 / (PI * ax * ay * d * d);
}

float3 schlick(float3 f0, float c) {
    float m = saturate(1.0 - c);
    float m2 = m * m;
    return f0 + (1.0 - f0) * (m2 * m2 * m);
}

// shade.rs::hash3/vnoise/fbm/marble — deterministic world-space veining.
float hash3(int x, int y, int z) {
    uint h = uint(x) * 0x8da6b343u ^ uint(y) * 0xd8163841u ^ uint(z) * 0xcb1ab31fu;
    h ^= h >> 13u;
    h *= 0x9e3779b1u;
    h ^= h >> 16u;
    return float(h & 0x00ffffffu) / 16777216.0;
}

float vnoise(float3 p) {
    float3 f = floor(p);
    int ix = int(f.x), iy = int(f.y), iz = int(f.z);
    float3 t = p - f;
    float3 s = t * t * (3.0 - 2.0 * t);
    float x00 = lerp(hash3(ix, iy, iz),         hash3(ix + 1, iy, iz),         s.x);
    float x10 = lerp(hash3(ix, iy + 1, iz),     hash3(ix + 1, iy + 1, iz),     s.x);
    float x01 = lerp(hash3(ix, iy, iz + 1),     hash3(ix + 1, iy, iz + 1),     s.x);
    float x11 = lerp(hash3(ix, iy + 1, iz + 1), hash3(ix + 1, iy + 1, iz + 1), s.x);
    return lerp(lerp(x00, x10, s.y), lerp(x01, x11, s.y), s.z);
}

float fbm(float3 p) {
    float amp = 0.5;
    float sum = 0.0;
    [unroll] for (int i = 0; i < 5; ++i) {
        sum += amp * vnoise(p);
        p *= 2.02;
        amp *= 0.5;
    }
    return sum;
}

float3 marble(float3 p, float scale) {
    const float3 BASE = float3(0.93, 0.92, 0.90);
    const float3 VEIN = float3(0.10, 0.11, 0.15);
    float3 q = p * scale;
    float s = sin(q.x + 0.7 * q.y + 5.0 * fbm(q));
    float t = saturate((abs(s) - 0.04) / 0.18);
    return lerp(VEIN, BASE, t * t * (3.0 - 2.0 * t));
}

// shade.rs::surface_point: interpolated, face-flipped shading normal and the
// eps-offset origin for secondary rays.
void surface_point(float3 ro, float3 rd, HitInfo hit, out float3 p, out float3 n) {
    uint3 idx = uint3(indices[hit.tri * 3u], indices[hit.tri * 3u + 1u], indices[hit.tri * 3u + 2u]);
    float w = 1.0 - hit.u - hit.v;
    n = normalize_or_zero(normals[idx.x] * w + normals[idx.y] * hit.u + normals[idx.z] * hit.v);
    if (all(n == float3(0.0, 0.0, 0.0))) {
        float3 e1 = positions[idx.y] - positions[idx.x];
        float3 e2 = positions[idx.z] - positions[idx.x];
        n = normalize_or_zero(cross(e1, e2));
    }
    if (dot(n, rd) > 0.0) n = -n;
    p = ro + rd * hit.t + n * SCENE_EPS;
}

// --- The shade() port ----------------------------------------------------------

// Whitted shading of a committed primary hit, reflection bounce included.
// (ro, rd) is the ray that produced `hit`; rd unit-length. RNG draw order
// matches shade.rs exactly per lap: shadow pairs, AO pairs, reflection pair.
// Quality is explicit (n_shadow / n_ao / refl) so the hemi leaf pass can run
// the BOUNCE_Q policy (1/0/false) through the same code.
// `split_ambient` (hemi mode, lap 0 only): the primary ambient term is
// omitted — no AO draws — and amb_w = kd, (amb_o, amb_n) = the surface point
// are returned for the hemi wavefront; the reflected hit (lap 1) always
// shades its own sampled ambient (the CPU forces fb OFF at depth 1).
// `prim` is the PRIMARY surface capture (shade.rs::PrimarySurface, lap 0
// only) for the G-buffer pack — a pure copy of already-computed values, so
// exposing it adds NO rng draws (the same-seed A/B gates rely on that).
float3 shade_split(float3 ro, float3 rd, HitInfo hit, inout uint rng,
                   uint n_shadow, uint n_ao, bool refl, bool split_ambient,
                   out float3 amb_w, out float3 amb_o, out float3 amb_n,
                   out PrimSurf prim) {
    amb_w = 0.0;
    amb_o = 0.0;
    amb_n = 0.0;
    prim = (PrimSurf)0;
    float3 total = 0.0;
    float3 tput = 1.0;
    [loop] for (uint lap = 0u; lap < 2u; ++lap) {
        float3 p, n;
        surface_point(ro, rd, hit, p, n);
        Mat mat = materials[tri_mat[hit.tri]];
        float3 albedo = mat.kind == MAT_MARBLE ? marble(ro + rd * hit.t, mat.scale) : mat.albedo;
        if (lap == 0u) {
            // shade.rs:226-233 — spec_t stays 0 unless the lap-0 reflection
            // below traces (hit t / INF on miss, shade.rs:481/:494).
            prim.n = n;
            prim.rough = mat.roughness;
            prim.albedo = albedo;
            prim.metallic = mat.metallic;
        }

        float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, mat.metallic);
        // 0.157 = Charlie peak directional albedo (shade.rs energy comp).
        float3 kd = albedo * (1.0 - mat.metallic) * (1.0 - 0.157 * mat.sheen);

        // Tangent frame: anisotropic materials brush circumferentially
        // around world-up; onb covers poles and isotropic.
        float3 t1, t2;
        if (mat.anisotropy > 0.0) {
            float3 t = cross(float3(0.0, 1.0, 0.0), n);
            if (dot(t, t) > 1e-8) {
                t1 = normalize(t);
                t2 = cross(n, t1);
            } else {
                onb(n, t1, t2);
            }
        } else {
            onb(n, t1, t2);
        }
        float ax, ay;
        ggx_alphas(mat.roughness, mat.anisotropy, ax, ay);
        float3 v = -rd;
        float3 vl = float3(dot(v, t1), dot(v, t2), max(dot(v, n), 1e-4));
        float lambda_v = ggx_lambda(vl, ax, ay);
        // Charlie-sheen inverse alpha, hoisted like the CPU.
        float sheen_inv_a = 1.0 / clamp(mat.roughness, 0.07, 1.0);

        // Direct light: N area-light samples, Lambert (1/pi omitted by
        // convention) + Cook-Torrance GGX with the compensating pi.
        float3 direct_d = 0.0;
        float3 direct_s = 0.0;
        float3 direct_t = 0.0; // thin-surface back transmission (foliage)
        for (uint si = 0u; si < n_shadow; ++si) {
            float su = rng_next(rng) * 2.0 - 1.0;
            float sv = rng_next(rng) * 2.0 - 1.0;
            float3 lp = light_center.xyz + light_u.xyz * su + light_v.xyz * sv;
            float3 lv = lp - p;
            float dist2 = dot(lv, lv);
            float dist = sqrt(dist2);
            float3 wi = lv / dist;
            float ndl = dot(n, wi);
            if (ndl <= 0.0) {
                // shade.rs translucency arm: a back-lit leaf receives the
                // light through itself; the occlusion ray starts on the
                // transmitted side (p - 2*eps*n = hit - n*eps). The rng
                // draws above already happened — order matches the CPU.
                if (mat.translucency > 0.0 && ndl < 0.0) {
                    if (!occluded_q(p - n * (2.0 * SCENE_EPS), wi, 0.0, dist - SCENE_EPS)) {
                        direct_t += light_color.xyz * ((-ndl) / dist2);
                    }
                }
                continue;
            }
            if (!occluded_q(p, wi, 0.0, dist - SCENE_EPS)) {
                float3 li = light_color.xyz * (ndl / dist2);
                direct_d += li;
                float3 h = normalize_or_zero(wi + v);
                float3 hl = float3(dot(h, t1), dot(h, t2), dot(h, n));
                if (hl.z > 0.0) {
                    float dn = ggx_ndf(hl, ax, ay);
                    float3 wil = float3(dot(wi, t1), dot(wi, t2), dot(wi, n));
                    float g2 = 1.0 / (1.0 + lambda_v + ggx_lambda(wil, ax, ay));
                    float3 f = schlick(f0, max(dot(wi, h), 0.0));
                    direct_s += li * f * (PI * dn * g2 / (4.0 * vl.z * ndl));
                    if (mat.sheen > 0.0) {
                        // Charlie NDF + Ashikhmin visibility (shade.rs).
                        float sin2 = max(1.0 - hl.z * hl.z, 0.0);
                        float d_c = (2.0 + sheen_inv_a) * pow(sin2, sheen_inv_a * 0.5) / TAU;
                        float v_ash = 1.0 / max(4.0 * (ndl + vl.z - ndl * vl.z), 1e-4);
                        direct_s += li * (PI * mat.sheen * d_c * v_ash);
                    }
                }
            }
        }
        if (n_shadow > 0u) {
            direct_d /= float(n_shadow);
            direct_s /= float(n_shadow);
            direct_t /= float(n_shadow);
        }
        // Diffuse budget split, front vs transmitted (ambient front-only —
        // shade.rs composition). GPU transmission is NOT ported: glass keeps
        // its full diffuse here (the SceneGpu init note flags it), so no
        // (1 - transmission) factor.
        float3 diffuse_d = direct_d * (1.0 - mat.translucency) + direct_t * mat.translucency;

        if (split_ambient && lap == 0u) {
            // Hemi mode: the primary ambient is integrated by the hemisphere
            // wavefront (compose applies amb_w * ambient later). No AO draws
            // here — the CPU's sampled loop is skipped the same way.
            total += tput * (kd * diffuse_d + direct_s);
            amb_w = tput * kd;
            amb_o = p;
            amb_n = n;
        } else {
            // Sampled AO modulating the constant ambient.
            float ao = 1.0;
            if (n_ao > 0u) {
                float3 at1, at2;
                onb(n, at1, at2);
                uint open = 0u;
                for (uint ai = 0u; ai < n_ao; ++ai) {
                    float r1 = rng_next(rng);
                    float r2 = rng_next(rng);
                    if (!occluded_q(p, cosine_dir(n, at1, at2, r1, r2), 0.0, AO_RADIUS)) open++;
                }
                ao = float(open) / float(n_ao);
            }
            total += tput * (kd * (diffuse_d + AMBIENT * ao) + direct_s);
        }

        // One specular bounce (lap 0 only): GGX VNDF importance sample
        // (Heitz 2018), throughput F*G2/G1 <= 1.
        if (lap == 1u || !refl || !(mat.metallic > 0.04 || mat.roughness < 0.45)) break;
        float3 vh = normalize(float3(ax * vl.x, ay * vl.y, vl.z));
        float lensq = vh.x * vh.x + vh.y * vh.y;
        float3 b1 = lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) / sqrt(lensq) : float3(1.0, 0.0, 0.0);
        float3 b2 = cross(vh, b1);
        float r = sqrt(rng_next(rng));
        float phi = TAU * rng_next(rng);
        float p1 = r * cos(phi);
        float p2 = r * sin(phi);
        float s = 0.5 * (1.0 + vh.z);
        p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * p2;
        float3 nh = b1 * p1 + b2 * p2 + vh * sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0));
        float3 hl = normalize(float3(ax * nh.x, ay * nh.y, max(nh.z, 1e-6)));
        float3 h = t1 * hl.x + t2 * hl.y + n * hl.z;
        float3 rdir = normalize(2.0 * dot(v, h) * h - v);
        // Below-horizon samples are dropped (slight darkening, no upward bias).
        if (dot(rdir, n) <= 0.0) break;
        float3 rdl = float3(dot(rdir, t1), dot(rdir, t2), dot(rdir, n));
        float g2_over_g1 = (1.0 + lambda_v) / (1.0 + lambda_v + ggx_lambda(rdl, ax, ay));
        tput *= schlick(f0, max(dot(v, h), 0.0)) * g2_over_g1;
        HitInfo rh;
        if (!trace_closest(p, rdir, 0.0, FLT_MAX, rh)) {
            prim.spec_t = INF; // reflection missed (shade.rs:494)
            total += tput * sky_color(rdir);
            break;
        }
        prim.spec_t = rh.t; // lap 0 only — the loop breaks before a lap-1 trace
        ro = p;
        rd = rdir;
        hit = rh;
        // lap 1 shades the reflected hit with reflections ineligible —
        // the CPU's depth-1 recursion.
    }
    return total;
}

// The plain (non-hemi) entry: quality straight from the frame constants.
float3 shade_full(float3 ro, float3 rd, HitInfo hit, inout uint rng, out PrimSurf prim) {
    float3 w, o, n;
    return shade_split(ro, rd, hit, rng, shadow_samples, ao_samples, reflections != 0u,
                       false, w, o, n, prim);
}
