// Scene resources + the shade.rs port. Requires trace_common.hlsli +
// rt.hlsli pasted first (trace.rs does the concatenation).
//
// This is shade.rs::shade with VisCtl::Off — the exact interactive non-bounce
// path — with the recursion flattened into a bounded DFS loop. The CPU
// recursion tree: the VNDF reflection bounce fires at depth 0 only; the
// glass transmission chain fires at any depth < TRANS_MAX_DEPTH — so only
// the ROOT can have two children (reflection + transmission), one stash slot
// is provably sufficient, and the loop runs at most 1 + TRANS_MAX_DEPTH +
// TRANS_MAX_DEPTH = 9 laps in the CPU's exact DFS order (reflection subtree
// first, then the root's transmission chain). Every ray here goes through
// the shared trace primitives (inline RayQuery, or TraceRay in the DXR
// library — transmission continuations route to HgHit, so the whole tree
// stays at TraceRay recursion depth 2); secondary rays use tmin = 0 from the
// eps-offset surface point, exactly like the CPU (the quadtree's inherited
// tmin is a primary-frustum property — it must never leak into shading; see
// CLAUDE.md invariants). Hemi mode (fb_mode > 0) splits the PRIMARY
// surface's ambient out: shade_split returns the ambient-free color plus the
// kd weight and surface point the hemi wavefront consumes; every non-root
// lap shades with its own sampled ambient (the CPU forces fb OFF past depth
// 0).

// t0 (bvh nodes) and t1 (tri_idx) belong to the frustum kernels.
StructuredBuffer<float3> positions : register(t2);
StructuredBuffer<float3> normals   : register(t3);
StructuredBuffer<uint>   indices   : register(t4); // flat, 3 per tri (also the BLAS index buffer)
StructuredBuffer<uint>   tri_mat   : register(t5);

#define MAT_DIFFUSE  0u
#define MAT_MARBLE   1u
#define MAT_TEXTURED 2u

// scene.rs::NO_TEX — "no map" sentinel for the texture-index fields.
#define TEX_NONE 0xffffffffu

// Mirrors trace.rs::GpuMat field-for-field (80 B) — a stride skew reads
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
    float transmission;
    uint tex; // Scene::textures index (MAT_TEXTURED: the space1 texs[] slot)
    float3 emissive;   // Ke; added at every lap, outside the kd*(1-transmission) factor
    uint normal_tex;   // tangent-space normal map (TEX_NONE = none; UNORM SRV)
    uint rough_tex;    // roughness map, samples .g (glTF channel convention)
    uint metal_tex;    // metallic map, samples .b
    uint emissive_tex; // emissive map (sRGB SRV)
    float normal_scale;
};
StructuredBuffer<Mat> materials : register(t6);

// shade.rs's glassware constants: fixed IOR (the classifier's `transmission`
// scalar carries all the variation) and the interface budget for the
// refraction chain (front/back walls of a two-walled tumbler, no TIR
// detour). Past the budget, glass shades opaque.
#define GLASS_IOR 1.5
#define TRANS_MAX_DEPTH 4u

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

// shade.rs::perturb_normal — tangent-space normal mapping with the tangent
// derived on the fly from the triangle's positions + UVs (zero storage),
// Gram-Schmidt vs n, bitangent handedness from the UV winding. Degenerate
// UVs/tangent or a past-horizon perturbation degrade to n. The green channel
// is negated (shade.rs::NORMAL_MAP_Y_SIGN): the loader V-flips at load, so
// OpenGL-convention (+Y up) maps point against our +v rows.
#define NORMAL_MAP_Y_SIGN (-1.0)
float3 perturb_normal(HitInfo hit, float3 n, Mat mat, float2 uv) {
    uint3 idx = uint3(indices[hit.tri * 3u], indices[hit.tri * 3u + 1u],
                      indices[hit.tri * 3u + 2u]);
    float3 p0 = positions[idx.x];
    float3 e1 = positions[idx.y] - p0;
    float3 e2 = positions[idx.z] - p0;
    float2 t0 = uv_buf[idx.x];
    float2 d1 = uv_buf[idx.y] - t0;
    float2 d2 = uv_buf[idx.z] - t0;
    float det = d1.x * d2.y - d2.x * d1.y;
    if (abs(det) < 1e-12) return n;
    float3 t = normalize_or_zero((e1 * d2.y - e2 * d1.y) / det - n * dot(n, (e1 * d2.y - e2 * d1.y) / det));
    if (all(t == float3(0.0, 0.0, 0.0))) return n;
    float3 b_raw = (e2 * d1.x - e1 * d2.x) / det;
    float3 bx = cross(n, t);
    float3 b = bx * (dot(bx, b_raw) >= 0.0 ? 1.0 : -1.0);
    float3 s = texs[NonUniformResourceIndex(mat.normal_tex)].SampleLevel(samp_lin, uv, 0.0).rgb;
    float3 tn = float3((s.x * 2.0 - 1.0) * mat.normal_scale,
                       (s.y * 2.0 - 1.0) * mat.normal_scale * NORMAL_MAP_Y_SIGN,
                       max(s.z * 2.0 - 1.0, 0.05));
    float3 outn = normalize_or_zero(t * tn.x + b * tn.y + n * tn.z);
    if (all(outn == float3(0.0, 0.0, 0.0)) || dot(outn, n) <= 0.0) return n;
    return outn;
}

// --- The shade() port ----------------------------------------------------------

// Whitted shading of a committed primary hit, reflection bounce included.
// (ro, rd) is the ray that produced `hit`; rd unit-length. RNG draw order
// matches shade.rs exactly per lap: shadow pairs, AO pairs, reflection pair.
// Quality is explicit (n_shadow / n_ao / refl) so the hemi leaf pass can run
// the BOUNCE_Q policy (1/0/false) through the same code.
// `split_ambient` (hemi mode, lap 0 only): the primary ambient term is
// omitted — no AO draws — and amb_w = kd*(1-transmission), (amb_o, amb_n) =
// the surface point are returned for the hemi wavefront; every non-root lap
// shades its own sampled ambient (the CPU forces fb OFF past depth 0).
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
    // Flattened-DFS state: `depth` is the CPU recursion depth of the surface
    // this lap shades; the one stash slot holds the root's transmission
    // child while the reflection subtree runs (only the root can have two
    // children — reflection is depth-0-only). Max laps = 1 root +
    // TRANS_MAX_DEPTH reflection-branch nodes + TRANS_MAX_DEPTH root-chain
    // nodes = 9, the CPU's exact DFS order.
    uint depth = 0u;
    bool have_stash = false;
    float3 st_o = 0.0, st_d = 0.0, st_tput = 0.0;
    HitInfo st_hit = (HitInfo)0;
    uint st_depth = 0u;
    // Is THIS lap inside the root's reflection subtree? Everything such a lap
    // adds to `total` is part of the CPU's `tput * rcol` — i.e. the
    // INDIRECT_SPECULAR signal — and the subtree is more than one lap: a
    // reflected ray that hits glass continues its own transmission chain.
    // The stash only ever holds the ROOT's transmission child, which is not.
    bool in_refl = false;
    bool st_in_refl = false;
    [loop] for (uint lap = 0u; lap < 1u + 2u * TRANS_MAX_DEPTH; ++lap) {
        float3 p, n;
        surface_point(ro, rd, hit, p, n);
        Mat mat = materials[tri_mat[hit.tri]];
        // Albedo: the shade.rs match. A texture REPLACES Kd (exporters set
        // Kd = 1 alongside map_Kd); hardware bilinear + the SRGB SRV format
        // reproduce texture.rs::sample_bilinear (decode per texel, then
        // filter — precision-level differences only, absorbed by the
        // statistical CPU-vs-GPU gates).
        float3 albedo =
            mat.kind == MAT_MARBLE   ? marble(ro + rd * hit.t, mat.scale)
          : mat.kind == MAT_TEXTURED ? texs[NonUniformResourceIndex(mat.tex)]
                .SampleLevel(samp_lin, tri_uv(hit.tri, hit.u, hit.v), 0.0).rgb
          : mat.albedo;
        // Map-driven material terms — the shade.rs block, ZERO rng draws
        // (materials with every map at TEX_NONE run bit-identically to the
        // pre-map kernel; the same-seed wavefront-vs-reference gates rely on
        // that). Linear maps ride UNORM SRVs (no sRGB decode).
        float2 map_uv = float2(0.0, 0.0);
        if (mat.normal_tex != TEX_NONE || mat.rough_tex != TEX_NONE ||
            mat.metal_tex != TEX_NONE || mat.emissive_tex != TEX_NONE) {
            map_uv = tri_uv(hit.tri, hit.u, hit.v);
        }
        float rough_eff = mat.roughness;
        float metal_eff = mat.metallic;
        if (mat.rough_tex != TEX_NONE) {
            rough_eff = clamp(rough_eff * texs[NonUniformResourceIndex(mat.rough_tex)]
                                              .SampleLevel(samp_lin, map_uv, 0.0).g,
                              0.02, 1.0);
        }
        if (mat.metal_tex != TEX_NONE) {
            metal_eff = clamp(metal_eff * texs[NonUniformResourceIndex(mat.metal_tex)]
                                              .SampleLevel(samp_lin, map_uv, 0.0).b,
                              0.0, 1.0);
        }
        // Shading normal n_s (n_s == n when unmapped): the BRDF frame, N·L,
        // and the guide use it; n keeps the ray offsets, the translucency
        // back ray, the hemi handoff, and the glass chain — the shade.rs
        // n_g/n_s split.
        float3 n_s = n;
        if (mat.normal_tex != TEX_NONE) n_s = perturb_normal(hit, n, mat, map_uv);
        if (lap == 0u) {
            // shade.rs — spec_t stays 0 unless the lap-0 reflection below
            // traces (hit t / INF on miss).
            prim.n = n_s;
            prim.rough = rough_eff;
            prim.albedo = albedo;
            prim.metallic = metal_eff;
        }

        float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metal_eff);
        // 0.157 = Charlie peak directional albedo (shade.rs energy comp).
        float3 kd = albedo * (1.0 - metal_eff) * (1.0 - 0.157 * mat.sheen);

        // Tangent frame on the SHADING normal: anisotropic materials brush
        // circumferentially around world-up; onb covers poles and isotropic.
        float3 t1, t2;
        if (mat.anisotropy > 0.0) {
            float3 t = cross(float3(0.0, 1.0, 0.0), n_s);
            if (dot(t, t) > 1e-8) {
                t1 = normalize(t);
                t2 = cross(n_s, t1);
            } else {
                onb(n_s, t1, t2);
            }
        } else {
            onb(n_s, t1, t2);
        }
        float ax, ay;
        ggx_alphas(rough_eff, mat.anisotropy, ax, ay);
        float3 v = -rd;
        // The face-flip guarantees n·v >= 0 for the GEOMETRIC normal; a
        // perturbed n_s can dip below — the grazing guard covers both.
        float3 vl = float3(dot(v, t1), dot(v, t2), max(dot(v, n_s), 1e-4));
        float lambda_v = ggx_lambda(vl, ax, ay);
        // Charlie-sheen inverse alpha, hoisted like the CPU.
        float sheen_inv_a = 1.0 / clamp(rough_eff, 0.07, 1.0);

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
            // N·L against the SHADING normal; the shadow/translucency ray
            // geometry stays on the geometric n (shade.rs).
            float ndl = dot(n_s, wi);
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
                float3 hl = float3(dot(h, t1), dot(h, t2), dot(h, n_s));
                if (hl.z > 0.0) {
                    float dn = ggx_ndf(hl, ax, ay);
                    float3 wil = float3(dot(wi, t1), dot(wi, t2), dot(wi, n_s));
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
        if (lap == 0u) {
            // shade.rs's post-average lobe export (the FSR-RR signal split
            // demodulates these at the G-buffer write) — pure copies, zero
            // rng draws.
            prim.direct_d = direct_d;
            prim.direct_s = direct_s;
        }
        // Diffuse budget split, front vs transmitted (ambient front-only —
        // shade.rs composition). Transmissive glass has (almost) no diffuse
        // response — the transmitted scene replaces it (the chain below);
        // the GGX highlight stays unscaled. kt == 1.0 exactly when
        // transmission == 0 (every procedural/stress material), so the
        // multiply is bit-neutral there.
        float3 diffuse_d = direct_d * (1.0 - mat.translucency) + direct_t * mat.translucency;
        float kt = 1.0 - mat.transmission;

        if (split_ambient && lap == 0u) {
            // Hemi mode: the primary ambient is integrated by the hemisphere
            // wavefront (compose applies amb_w * ambient later). No AO draws
            // here — the CPU's sampled loop is skipped the same way. The
            // ambient sits INSIDE the (1 - transmission) factor (shade.rs),
            // so kt folds into the weight.
            float3 c = tput * (kd * kt * diffuse_d + direct_s);
            total += c;
            if (in_refl) prim.ind_s += c;
            amb_w = tput * kd * kt;
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
            // FSR's AO signal — lap 0 only (later laps compute their own AO
            // for their own ambient; the capture is the PRIMARY surface's).
            if (lap == 0u) prim.ao = ao;
            float3 c = tput * (kd * kt * (diffuse_d + AMBIENT * ao) + direct_s);
            total += c;
            if (in_refl) prim.ind_s += c;
        }

        // Emitted radiance — additive per lap, OUTSIDE the kd*kt factor, so
        // emitters appear in reflections and through glass (shade.rs). The
        // guard keeps emissive-free materials bit-identical.
        if (any(mat.emissive != 0.0) || mat.emissive_tex != TEX_NONE) {
            float3 emis = mat.emissive;
            if (mat.emissive_tex != TEX_NONE) {
                emis *= texs[NonUniformResourceIndex(mat.emissive_tex)]
                            .SampleLevel(samp_lin, map_uv, 0.0).rgb;
            }
            total += tput * emis;
            if (in_refl) prim.ind_s += tput * emis;
        }

        // Continuation bookkeeping: at most one of the two branches below
        // becomes `next`; the root's second branch (transmission behind a
        // traced reflection) goes to the stash.
        bool next_set = false;
        float3 nx_o = 0.0, nx_d = 0.0, nx_tput = 0.0;
        HitInfo nx_hit = (HitInfo)0;
        uint nx_depth = 0u;
        // A continuation inherits this lap's subtree unless the reflection
        // branch below sets it — that branch IS the root of the ind_s subtree.
        bool nx_in_refl = in_refl;

        // (a) One specular bounce — ROOT only (shade.rs gates depth == 0):
        // GGX VNDF importance sample (Heitz 2018), throughput F*G2/G1 <= 1.
        // The two rng draws stay conditional on the same gate as the CPU's.
        if (depth == 0u && refl && (mat.metallic > 0.04 || mat.roughness < 0.45)) {
            float3 vh = normalize(float3(ax * vl.x, ay * vl.y, vl.z));
            float lensq = vh.x * vh.x + vh.y * vh.y;
            float3 b1 =
                lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) / sqrt(lensq) : float3(1.0, 0.0, 0.0);
            float3 b2 = cross(vh, b1);
            float r = sqrt(rng_next(rng));
            float phi = TAU * rng_next(rng);
            float p1 = r * cos(phi);
            float p2 = r * sin(phi);
            float s = 0.5 * (1.0 + vh.z);
            p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * p2;
            float3 nh = b1 * p1 + b2 * p2 + vh * sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0));
            float3 hl = normalize(float3(ax * nh.x, ay * nh.y, max(nh.z, 1e-6)));
            float3 h = t1 * hl.x + t2 * hl.y + n_s * hl.z;
            float3 rdir = normalize(2.0 * dot(v, h) * h - v);
            // Below-horizon samples are dropped (slight darkening, no upward
            // bias; spec_t stays 0.0 = "no reflection traced") — but the
            // transmission branch below still runs, like the CPU. BOTH
            // horizons: n_s (the lobe's frame) and the geometric n (a
            // perturbed lobe must not fire a ray that re-enters the surface).
            if (dot(rdir, n_s) > 0.0 && dot(rdir, n) > 0.0) {
                float3 rdl = float3(dot(rdir, t1), dot(rdir, t2), dot(rdir, n_s));
                float g2_over_g1 =
                    (1.0 + lambda_v) / (1.0 + lambda_v + ggx_lambda(rdl, ax, ay));
                float3 rtput = tput * schlick(f0, max(dot(v, h), 0.0)) * g2_over_g1;
                HitInfo rh;
                if (trace_closest(p, rdir, 0.0, FLT_MAX, rh)) {
                    prim.spec_t = rh.t; // depth 0 == the captured surface
                    nx_o = p;
                    nx_d = rdir;
                    nx_hit = rh;
                    nx_tput = rtput;
                    nx_depth = 1u;
                    next_set = true;
                    nx_in_refl = true; // everything below this is ind_s
                } else {
                    prim.spec_t = INF; // reflection missed (shade.rs)
                    // The reflection subtree is just the sky here — still the
                    // ind_s signal (the CPU's tput * rcol with rcol = sky).
                    total += rtput * sky_color(rdir);
                    prim.ind_s += rtput * sky_color(rdir);
                }
            }
        }

        // (b) Glass transmission — the shade.rs Snell chain, shading-only
        // (glass still HITS: frustum bounds / inherited tmin / temporal
        // claims are untouched) and drawing ZERO rng, placed after the
        // reflection draws so the stream never moves. The Fresnel-reflected
        // fraction at the root is the VNDF bounce above; at interior
        // interfaces it is dropped — dimming, never gaining. TIR continues
        // as an internal mirror bounce.
        if (refl && mat.transmission > 0.0 && depth < TRANS_MAX_DEPTH) {
            // Entering or exiting? Re-derive the pre-flip normal orientation
            // (surface_point returns only the viewer-facing normal).
            uint3 idx = uint3(indices[hit.tri * 3u], indices[hit.tri * 3u + 1u],
                              indices[hit.tri * 3u + 2u]);
            float w = 1.0 - hit.u - hit.v;
            float3 n_raw =
                normalize_or_zero(normals[idx.x] * w + normals[idx.y] * hit.u +
                                  normals[idx.z] * hit.v);
            if (all(n_raw == float3(0.0, 0.0, 0.0))) {
                float3 e1 = positions[idx.y] - positions[idx.x];
                float3 e2 = positions[idx.z] - positions[idx.x];
                n_raw = normalize_or_zero(cross(e1, e2));
            }
            bool entering = dot(n_raw, n) >= 0.0; // the viewer-flip didn't fire
            float eta = entering ? 1.0 / GLASS_IOR : GLASS_IOR;
            // The refraction chain stays on the GEOMETRIC normal (shade.rs);
            // v·n with the same grazing guard is bit-identical to the old
            // n-frame vl.z when no normal map is present.
            float cos_i = max(dot(v, n), 1e-4);
            float k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
            float3 hit_p = ro + rd * hit.t;
            float3 tdir, torig;
            float ttw;
            if (k >= 0.0) {
                // Exact unpolarized dielectric Fresnel (not Schlick — it
                // must reach 1 as k -> 0 or the TIR handoff pops).
                float cos_t = sqrt(k);
                float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
                float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
                float fr = 0.5 * (rs * rs + rp * rp);
                tdir = normalize(rd * eta + n * (eta * cos_i - cos_t));
                torig = hit_p - n * SCENE_EPS;
                ttw = mat.transmission * (1.0 - fr);
            } else {
                // TIR: mirror about n, staying on the incident side.
                tdir = rd + n * (2.0 * cos_i);
                torig = hit_p + n * SCENE_EPS;
                ttw = mat.transmission;
            }
            if (ttw > 1e-3) {
                // Tinted by albedo — the classifier lifts dark MTL glass Kd
                // toward white so this doesn't go black.
                float3 t_tput = tput * albedo * ttw;
                HitInfo th;
                if (trace_closest(torig, tdir, 0.0, FLT_MAX, th)) {
                    if (!next_set) {
                        nx_o = torig;
                        nx_d = tdir;
                        nx_hit = th;
                        nx_tput = t_tput;
                        nx_depth = depth + 1u;
                        next_set = true;
                    } else {
                        // Root only: park the transmission child while the
                        // reflection subtree runs (CPU DFS order).
                        st_o = torig;
                        st_d = tdir;
                        st_hit = th;
                        st_tput = t_tput;
                        st_depth = depth + 1u;
                        st_in_refl = in_refl; // root's own chain: not ind_s
                        have_stash = true;
                    }
                } else {
                    total += t_tput * sky_color(tdir);
                    if (in_refl) prim.ind_s += t_tput * sky_color(tdir);
                }
            }
        }

        if (!next_set) {
            if (!have_stash) break;
            nx_o = st_o;
            nx_d = st_d;
            nx_hit = st_hit;
            nx_tput = st_tput;
            nx_depth = st_depth;
            nx_in_refl = st_in_refl;
            have_stash = false;
        }
        // Continuation laps shade with reflections ineligible past the root
        // (the depth gate) and with fb OFF (the split_ambient lap-0 gate) —
        // the CPU's recursive `shade(depth + 1)` policy.
        ro = nx_o;
        rd = nx_d;
        hit = nx_hit;
        tput = nx_tput;
        depth = nx_depth;
        in_refl = nx_in_refl;
    }
    return total;
}

// The plain (non-hemi) entry: quality straight from the frame constants.
float3 shade_full(float3 ro, float3 rd, HitInfo hit, inout uint rng, out PrimSurf prim) {
    float3 w, o, n;
    return shade_split(ro, rd, hit, rng, shadow_samples, ao_samples, reflections != 0u,
                       false, w, o, n, prim);
}
