// The amortized cloud lattice, shared by every kernel that shades sky:
// `cs_sky` (proven-empty rects) and `cs_leaf`'s miss branch (sky pixels inside
// leaf tiles, near silhouettes). Requires trace_common.hlsli pasted first, and
// SKY_UNIT defined so queues.hlsli yields u5.
//
// WHY SCREEN-SPACE RATHER THAN PER-RECT. The first version filled the lattice
// only from the proven-empty rects, which is the version the quadtree uniquely
// enables. Measured on the B70: it removed 0.077 of the 0.287 ms/sample the
// cloud march costs — **27%** — and the other 73% turned out to live outside
// those rects, mostly in leaf-tile sky misses. So the lattice is filled over
// the whole screen and both consumers read it. That trades elegance for the
// milliseconds: a per-pixel renderer could do the same thing, and the
// empty-space proof is no longer what licenses it.
//
// The lattice still costs almost nothing (0.03 ms for the rect version), so the
// waste on geometry pixels is affordable — but it does move the optimum K
// upward: a full-screen lattice at K=2 computes about as many marches as the
// sky pixels it serves, and only pays from K=4 on.
//
// `cloud_lod` takes u5 — the tile queue's register, dead by the time anything
// shades. Each compile unit re-declares what its registers mean in that phase
// (the `record_hemi` idiom); the alternative was a 15th root UAV against a
// root signature already at 62/64 DWORDs.

RWStructuredBuffer<float4> cloud_lod : register(u5); // (scatter.rgb, transmittance)

// Point (lx, ly) samples pixel CENTRE ((lx<<L)+0.5, (ly<<L)+0.5), so a pixel
// landing on a lattice point interpolates to it with weight 1 — no drift. One
// point of border past the far edge so every pixel has all four cell corners.
uint sky_lod_w() { return (rw >> SKY_LOD_LOG) + 2u; }
uint sky_lod_h() { return (rh >> SKY_LOD_LOG) + 2u; }
uint sky_lod_idx(uint lx, uint ly) { return ly * sky_lod_w() + lx; }

// Bilinear fetch of the cloud half. Weights are in pixel units over the lattice
// pitch, so w == 0 exactly on a lattice column/row.
float4 sky_cloud_lerp(uint x, uint y) {
    uint lx = x >> SKY_LOD_LOG;
    uint ly = y >> SKY_LOD_LOG;
    float fx = float(x - (lx << SKY_LOD_LOG)) * (1.0 / float(SKY_LOD));
    float fy = float(y - (ly << SKY_LOD_LOG)) * (1.0 / float(SKY_LOD));
    float4 c00 = cloud_lod[sky_lod_idx(lx, ly)];
    float4 c10 = cloud_lod[sky_lod_idx(lx + 1u, ly)];
    float4 c01 = cloud_lod[sky_lod_idx(lx, ly + 1u)];
    float4 c11 = cloud_lod[sky_lod_idx(lx + 1u, ly + 1u)];
    return lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy);
}

// The whole sky integral at a pixel, cloud half interpolated. `d` is the
// pixel's OWN (possibly jittered) direction, so the sharp half — sun limb,
// stars — stays exact per pixel and per sample; only the march is shared.
float3 sky_radiance_lod(float3 d, uint x, uint y) {
    return sky_compose(sky_backdrop(d, pixel_cone * 0.5, frame), sky_cloud_lerp(x, y));
}
