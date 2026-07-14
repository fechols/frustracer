// Hemisphere-bounce wavefront: sphcell.rs ports (van Oosterom–Strackee solid
// angle, Lambert PSA, midpoint splits, Arvo triangle sampling) + the hemi
// queue records. Requires trace_common.hlsli + ctr.hlsli pasted first.
// Hemi units bind the SAME physical buffers the primary units see at
// u5/u6/u7/u9/u12/u13 — trace.rs rebinds the root descriptors per pass, so
// the register meanings here are the hemi ones.
//
// Soundness contract carried over from hemi.rs verbatim: the apex is
// hit + n·eps with root t_start = 0 (NOT eps — ball(o, eps) is a false claim
// at concave corners); the root cut is the BVH root [0] (a primary tile's
// cut is invalid at a different apex); midpoint children exactly partition
// the parent so inherited (tc, cut) is sound; blocked cells still subdivide;
// AO clamps every query to ao_radius and a `None` is only ever "open within
// the radius", never sky.

// One spherical-triangle cell (also the leaf-cell record — same shape).
// 64 bytes.
struct HemiCellRec {
    float3 a;
    float t_start; // inherited tc (leaf recs: the tc its rays consume as TMin)
    float3 b;
    uint point_id; // index into hemi_pts
    float3 c;
    uint meta;     // depth | (path << 8): path = octant + 2 bits/level
    uint cut_slot; // hemi cut pool slot; 0xffffffff = the root cut [0]
    uint cut_len;
    uint _pad1;
    uint _pad2;
};

struct HemiPointRec {
    float3 o;
    uint pixel;
    float3 n;
    uint _pad;
};

RWStructuredBuffer<HemiCellRec>  hqin     : register(u5);
RWStructuredBuffer<HemiCellRec>  hqout    : register(u6);
RWStructuredBuffer<HemiCellRec>  hqleaf   : register(u7);
// Named `cut_pool` so frustum.hlsli's cut_node/refine_cut bind to the hemi
// pool in hemi units (trace.rs binds the hemi pool at u9 for these passes).
RWStructuredBuffer<uint>         cut_pool : register(u9);
RWStructuredBuffer<uint>         hbuf     : register(u12);
RWStructuredBuffer<HemiPointRec> hemi_pts : register(u13);

#define ROOT_CUT_SLOT 0xffffffffu
#define HEMI_FIXED 262144.0
#define HEMI_SALT 0x48454d49u

// AO clamps to the radius; GI's limit is "infinite" (FLT_MAX sentinel).
float hemi_t_limit() { return fb_mode == 1u ? AO_RADIUS : FLT_MAX; }

void hemi_add(uint pixel, uint ch, float v) {
    // Clamp into the accumulator's representable range: ftou is UNDEFINED
    // for negatives (fp-noise psa3 of degenerate cells) and above 2^32-1
    // (a GI firefly grazing the light rect, li ~ ndl/dist^2) — NVIDIA
    // saturates, other hardware may wrap to 0. The CPU's f32 accumulator
    // just carries the firefly; here 16383 is the u32's whole dynamic range.
    uint dummy;
    float c = clamp(v, 0.0, 16383.0);
    InterlockedAdd(hbuf[pixel * 4u + ch], uint(c * HEMI_FIXED + 0.5), dummy);
}

void hemi_add3(uint pixel, float3 v) {
    hemi_add(pixel, 0u, v.x);
    hemi_add(pixel, 1u, v.y);
    hemi_add(pixel, 2u, v.z);
}

// --- sphcell.rs ports ---------------------------------------------------------

// Van Oosterom–Strackee (orientation-free).
float solid_angle3(float3 a, float3 b, float3 c) {
    float num = abs(dot(a, cross(b, c)));
    float den = 1.0 + dot(a, b) + dot(b, c) + dot(c, a);
    return 2.0 * atan2(num, den);
}

// Lambert's exact projected solid angle onto the plane perpendicular to axis.
float psa3(float3 a, float3 b, float3 c, float3 axis) {
    precise float sum = 0.0;
    float3 uu[3] = { a, b, c };
    float3 vv[3] = { b, c, a };
    [unroll] for (uint i = 0; i < 3; ++i) {
        float3 cr = cross(uu[i], vv[i]);
        float l = length(cr);
        if (l > 1e-12) {
            sum += atan2(l, dot(uu[i], vv[i])) * (dot(cr, axis) / l);
        }
    }
    return 0.5 * sum;
}

void midpoints3(float3 a, float3 b, float3 c, out float3 mab, out float3 mbc, out float3 mca) {
    mab = normalize(a + b);
    mbc = normalize(b + c);
    mca = normalize(c + a);
}

float3 centroid3(float3 a, float3 b, float3 c) {
    return normalize_or_zero(a + b + c);
}

// Angle between unit vectors, stable near both 0 and pi (sphcell.rs).
float angle_between(float3 u, float3 v) {
    if (dot(u, v) < 0.0) {
        return PI - 2.0 * asin(clamp(0.5 * length(u + v), -1.0, 1.0));
    }
    return 2.0 * asin(clamp(0.5 * length(v - u), -1.0, 1.0));
}

float3 gram_schmidt(float3 v, float3 w) {
    return normalize_or_zero(v - w * dot(v, w));
}

// Uniform sample inside the spherical triangle (Arvo '95, PBRT-v4 form,
// including the +pi parameterization — see sphcell.rs, whose self-test
// caught the sign mirror once already).
float3 sample_tri(float3 a, float3 b, float3 c, float u0, float u1) {
    float3 n_ab = cross(a, b);
    float3 n_bc = cross(b, c);
    float3 n_ca = cross(c, a);
    if (dot(n_ab, n_ab) < 1e-18 || dot(n_bc, n_bc) < 1e-18 || dot(n_ca, n_ca) < 1e-18) {
        return centroid3(a, b, c);
    }
    n_ab = normalize(n_ab);
    n_bc = normalize(n_bc);
    n_ca = normalize(n_ca);
    float alpha = angle_between(n_ab, -n_ca);
    float beta = angle_between(n_bc, -n_ab);
    float gamma = angle_between(n_ca, -n_bc);
    float area = alpha + beta + gamma - PI;
    if (area < 1e-7) {
        return centroid3(a, b, c);
    }
    float ap = u0 * area;
    float sin_ap = sin(ap);
    float cos_ap = cos(ap);
    float sin_alpha = sin(alpha);
    float cos_alpha = cos(alpha);
    float sin_phi = sin_alpha * cos_ap - sin_ap * cos_alpha; // sin(ap + pi - alpha)
    float cos_phi = -(cos_ap * cos_alpha + sin_ap * sin_alpha); // cos(ap + pi - alpha)
    float k1 = cos_phi + cos_alpha;
    float k2 = sin_phi - sin_alpha * dot(a, b);
    float cos_bp = (k2 + (k2 * cos_phi - k1 * sin_phi) * cos_alpha)
        / ((k2 * sin_phi + k1 * cos_phi) * sin_alpha);
    if (!(cos_bp == cos_bp) || abs(cos_bp) > 1e30) {
        cos_bp = 1.0; // degenerate slice -> C' = A, still inside
    }
    cos_bp = clamp(cos_bp, -1.0, 1.0);
    float sin_bp = sqrt(max(1.0 - cos_bp * cos_bp, 0.0));
    float3 cp = normalize(a * cos_bp + gram_schmidt(c, a) * sin_bp);
    float cos_theta = 1.0 - u1 * (1.0 - dot(cp, b));
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    return normalize(b * cos_theta + gram_schmidt(cp, b) * sin_theta);
}

// The 4 CCW octants around n; (t1, t2) right-handed (t1 x t2 = n).
void octant(uint i, float3 n, float3 t1, float3 t2, out float3 a, out float3 b, out float3 c) {
    a = n;
    b = i == 0u ? t1 : (i == 1u ? t2 : (i == 2u ? -t1 : -t2));
    c = i == 0u ? t2 : (i == 1u ? -t1 : (i == 2u ? -t2 : t1));
}

// hemi.rs::sky_cell, iteratively (HLSL has no recursion): centroid radiance x
// exact PSA with midpoint refinement to ~12 deg (and ~6 deg near the sun-glow
// lobe). Empty parents prove children empty — pure math, no BVH work. Max
// live stack: 4-way DFS, depth <= 5 => 3*5 + 1 = 16 entries.
float3 sky_cell_sum(float3 n, float3 a0, float3 b0, float3 c0, uint levels0) {
    float3 sa[16];
    float3 sb[16];
    float3 sc[16];
    uint sl[16];
    sa[0] = a0; sb[0] = b0; sc[0] = c0; sl[0] = levels0;
    uint sp = 1;
    float3 sum = 0.0;
    [loop] while (sp > 0) {
        --sp;
        float3 a = sa[sp], b = sb[sp], c = sc[sp];
        uint lv = sl[sp];
        float3 cen = centroid3(a, b, c);
        bool refine = false;
        if (lv > 0) {
            float cos_r = clamp(min(dot(cen, a), min(dot(cen, b), dot(cen, c))), -1.0, 1.0);
            bool coarse = cos_r < 0.978;
            // The dome's sharpest surviving feature is the MIE AUREOLE (the
            // forward HG lobe at g = 0.76). Refine to ~6 deg within a
            // conservative 30 deg cone of the sun: cos(ang) > cos(r + 30),
            // expanded. hemi.rs::sky_cell — keep in lockstep.
            bool near_aureole = cos_r < 0.995;
            if (near_aureole) {
                float sin_r = sqrt(max(1.0 - cos_r * cos_r, 0.0));
                near_aureole = dot(cen, sun.xyz) > cos_r * 0.866 - sin_r * 0.5;
            }
            refine = coarse || near_aureole;
        }
        if (refine) {
            float3 mab, mbc, mca;
            midpoints3(a, b, c, mab, mbc, mca);
            sa[sp] = a;   sb[sp] = mab; sc[sp] = mca; sl[sp] = lv - 1; ++sp;
            sa[sp] = mab; sb[sp] = b;   sc[sp] = mbc; sl[sp] = lv - 1; ++sp;
            sa[sp] = mca; sb[sp] = mbc; sc[sp] = c;   sl[sp] = lv - 1; ++sp;
            sa[sp] = mab; sb[sp] = mbc; sc[sp] = mca; sl[sp] = lv - 1; ++sp;
        } else {
            // The DOME — never the disc. Centroid point-sampling a cell coarser
            // than the sun would alias it catastrophically (see hemi.rs).
            sum += sky_dome(cen) * psa3(a, b, c, n);
        }
    }
    return sum;
}
