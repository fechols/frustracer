// Shared prelude for every GPU-tracer kernel. There is no #include — trace.rs
// concatenates the .hlsli sources ahead of each kernel before DXC sees them,
// so this file must stay self-contained and order-independent apart from
// coming first.
//
// Contract notes (mirrors of the CPU renderer, keep in lockstep):
// - ray_dir == camera.rs::CamBasis::ray_dir (normalized — distance == ray t).
// - sky_dome / sky_radiance == src/sky.rs (read its header: WHICH one a ray
//   calls is a correctness decision — the sun disc appears exactly once per
//   light path).
// - pack_info == overlay.rs::pack_info; KIND_* == overlay.rs.
// - The RNG is a per-pixel counter-based PCG stream seeded from
//   (x, y, frame, salt) — deliberately NOT the CPU's WyRand: sequences
//   differ, means match, and no exact-zero gate depends on the sequence.
//   Draw ORDER within a pixel mirrors the CPU exactly (jitter, then shadow
//   pairs, then AO pairs, then the two reflection draws).

#define PI  3.14159265358979
#define TAU 6.28318530717959
#define FLT_MAX 3.402823466e38
#define INF (asfloat(0x7f800000u))

#define KIND_LEAF    0u
#define KIND_SKY     1u
#define KIND_BLOCKED 2u
#define KIND_COARSE  3u

#define FLAG_ACCUM        1u   // frame > 0 adds instead of stores
#define FLAG_JITTER       2u   // per-pixel rng jitter (legacy accumulation)
#define FLAG_FRAME_JITTER 4u   // frame-uniform jitter from frame_jitter (DLSS/XeSS)
#define FLAG_VERIFY       8u   // check builds: hemi claim re-validation + PSA accounting
#define FLAG_GBUF         16u  // G-buffer pack writes on (upscaler sessions ONLY —
                               // the pack buffer is a 64-byte dummy otherwise and
                               // root UAVs have no bounds check: this gate is
                               // memory safety, not an optimization)
#define FLAG_HAS_PREV     32u  // the prev_* rows carry last frame's camera basis
#define FLAG_FSR_SIG      64u  // FSR-RR sessions: demodulated direct-light
                               // signals in GBufPx.sig + the prev-camera
                               // view-Z in mv.z (zeros otherwise — RR/XeSS
                               // sessions keep their pack bytes unchanged)
#define FLAG_ANISO        128u // anisotropic filtering on (--aniso > 1; a
                               // SESSION constant from texture::max_aniso()).
                               // WHICH laps use it is a call-site decision
                               // (shade_split's `aniso` arg — hemi bounce
                               // laps pass false), not this flag alone.
#define FLAG_CLOUDS       256u // volumetric cloud layer on (--no-clouds
                               // clears it); state rides SCENE_DIAG/CLOUD_TIME

cbuffer Frame : register(b0) {
    float4 cam_origin;   // xyz; w = inv_w
    float4 cam_forward;  // xyz (unit); w = inv_h
    float4 cam_right;    // xyz pre-scaled by tan(fov/2)*aspect
    float4 cam_up;       // xyz pre-scaled by tan(fov/2)
    // The sun (sky::Sun) — a DISC AT INFINITY, not the old rect area light.
    // Its three rows replace the old five; scene eps / ao_radius were rehomed
    // out of the dead light rows' w slots onto sun_e.w / sun_l.w.
    float4 sun;   // xyz (unit dir); w = cos(angular radius)
    float4 sun_e; // xyz = irradiance/PI (the direct loop's multiplier); w = scene eps
    float4 sun_l; // xyz = DISC radiance (what an escaping ray sees); w = ao_radius
    uint rw; uint rh; uint frame; uint flags;
    uint shadow_samples; uint ao_samples; uint reflections; uint _pad0;
    // pixel_cone: primary ray-cone spread (CamBasis::pixel_cone verbatim,
    // the trilinear texture LOD's single source — shade.hlsli::tex_lod_base).
    // sky_scale: time-of-day dome brightness (Scene::sky_scale — exactly 1.0
    // in an untouched session; x * 1.0 is bit-preserving).
    float2 frame_jitter; float pixel_cone; float sky_scale;
    // Wavefront queue capacities (resolution-derived, trace.rs computes them;
    // structural worst cases — the overflow counter is gated == 0).
    uint cap_tile; uint cap_leaf; uint cap_sky; uint cap_cut;
    // Hemisphere bounce state: fb_mode 0 = off, 1 = AO, 2 = GI; fb_depth =
    // the subdivision budget (Quality presets 2/3/4); hemi_batch = points
    // per batch (bounds transient queue memory); cap_hemi_pt = point-queue
    // capacity (rw*rh).
    uint fb_mode; uint fb_depth; uint hemi_batch; uint cap_hemi_pt;
    // Hemi per-batch queue capacities (batch-bounded, cannot overflow).
    uint cap_hemi_cell; uint cap_hemi_leaf; uint cap_hemi_cut; uint _pad3;
    // Previous frame's camera basis for G-buffer motion vectors (upscaler
    // sessions; zeroed with FLAG_HAS_PREV clear when there is no prev frame).
    // The scene-static near/far planes (dlss::near_far) ride the w slots.
    float4 prev_origin;  // xyz; w = prev inv_w
    float4 prev_forward; // xyz (unit); w = prev inv_h
    float4 prev_right;   // xyz pre-scaled; w = NEAR
    float4 prev_up;      // xyz pre-scaled; w = FAR
    // --spp: primary samples per pixel this frame (1..=MAX_SPP; the CPU pins
    // it to 1 on fb frames — one hemi point per pixel). probe_sample names the
    // sample that writes tbuf/info/the G-buffer pack: 0 in every real frame,
    // swept by --check-gpu/--check-dxr so EVERY sample's ray gets gated.
    // night: star visibility (Scene::night — exactly 0.0 in an untouched
    // session; sky_stars' guard is a BRANCH on it, so day kernels are
    // bit-identical by construction).
    uint spp; uint probe_sample; float night; uint _pad5;
    // Sub-pixel offsets for samples 1.. (dlss::jitter_for_sample, computed on
    // the CPU — one Halton source of truth), packed two per row. The row count
    // is INJECTED (trace::spp_defs, the ALPHA_CUTOUT/FTREE pattern) so it is
    // dlss::MAX_SPP / 2 by construction — a hand-mirrored literal here would
    // be a third constant to keep in lockstep, and reading past it is silent.
    // Slot 0 holds sample 0's offset, which the FLAG_FRAME_JITTER branch
    // already knows (sample_pos never reads it).
    float4 jitters[JITTER_ROWS];
    // The sky dome in order-2 SH (scene.sky_sh / sh.rs::Sh9) — the analytic
    // ambient irradiance, one RGB row per coefficient (.w unused). Appended
    // LAST so every offset above is unmoved.
    float4 sky_sh[9];
}

#define SCENE_EPS  (sun_e.w)
#define AO_RADIUS  (sun_l.w)
#define CAM_NEAR   (prev_right.w)
#define CAM_FAR    (prev_up.w)
// The cloud layer's state rides the cam rows' otherwise-zero w lanes
// (trace.rs::with_frame) — scene diag + the animation clock in seconds.
#define SCENE_DIAG (cam_right.w)
#define CLOUD_TIME (cam_up.w)

uint pack_info(uint depth, uint kind) { return (depth & 0xffu) | (kind << 8); }

// --- RNG: pcg-style hash stream ---------------------------------------------

uint pcg_mix(uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

uint rng_init(uint x, uint y, uint frame_idx, uint salt) {
    return pcg_mix(x * 0x9E3779B9u ^ y * 0xC2B2AE3Du ^ frame_idx * 0x27D4EB2Fu ^ salt);
}

float rng_next(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    w = (w >> 22u) ^ w;
    return float(w >> 8u) * (1.0 / 16777216.0);
}

// --- Camera ------------------------------------------------------------------

// camera.rs::ray_dir: pixel-grid coords (y down), normalized result.
float3 ray_dir(float fx, float fy) {
    float ndx = fx * cam_origin.w * 2.0 - 1.0;
    float ndy = 1.0 - fy * cam_forward.w * 2.0;
    return normalize(cam_forward.xyz + cam_right.xyz * ndx + cam_up.xyz * ndy);
}

// --- Multi-sampling (--spp) ---------------------------------------------------

// Sample k's continuous position inside pixel (x, y) — render.rs's
// trace_primary, term for term. k == 0 is the frame's REPORTED sample (the
// jitter policy a single-sample frame has always used, so spp == 1 stays
// bit-identical); k > 0 takes the deterministic Halton offset the CPU packed
// into `jitters`. Every sample stays inside the pixel, hence inside the tile
// frustum — which is what lets it consume the tile's inherited t_start/cut.
float2 sample_pos(uint x, uint y, uint k, inout uint rng) {
    float jx = 0.5, jy = 0.5;
    if (k > 0u) {
        float4 r = jitters[k >> 1u];
        float2 o = (k & 1u) ? r.zw : r.xy;
        jx = 0.5 + o.x;
        jy = 0.5 + o.y;
    } else if (flags & FLAG_FRAME_JITTER) {
        jx = 0.5 + frame_jitter.x;
        jy = 0.5 + frame_jitter.y;
    } else if (flags & FLAG_JITTER) {
        jx = rng_next(rng);
        jy = rng_next(rng);
    }
    return float2(float(x) + jx, float(y) + jy);
}

// --- The one sky (src/sky.rs) -------------------------------------------------
//
// Term-for-term with sky.rs; change both together. THE INVARIANT (see that
// file's header): the sun disc is delivered exactly once per light path. A ray
// sees the disc only if no light-sampling strategy already covers the sun along
// that path.
//
//   sky_dome(d)      GATHER paths: hemi cells, GI leaf misses. NO disc — the
//                    disc would double-count direct_d AND saturate the 2^18
//                    fixed-point hemi accumulator outright.
//   sky_radiance(o, d)  DISPLAY paths: the camera's own miss, and glass —
//                    the backdrop through the CLOUD layer along the ray
//                    (FLAG_CLOUDS; the cloud block below), plus its scatter.
//
// The specular reflection ray is the one path both strategies can reach, so it
// takes the dome plus a MIS-weighted disc (see shade.hlsli).

static const float  SKY_MIE_G   = 0.76;
static const float3 SKY_BETA_R  = float3(0.038, 0.096, 0.244); // Rayleigh, 1/lambda^4
static const float3 SKY_BETA_M  = float3(0.021, 0.021, 0.021); // Mie, wavelength-flat
static const float  SKY_GROUND_ALBEDO = 0.28;
static const float  SKY_DOME_SCALE    = 14.8;

// Single-scatter, kept LINEAR in beta with extinction as separate view/sun
// transmittances. Dividing the in-scatter by the (blue-heavy) combined beta
// would invert the grey Mie aureole into a red one — see sky.rs.
//
// `t_sun` is passed IN, not recomputed: the sun's path down to the scattering
// volume depends only on its elevation, so it is one value for the whole frame
// (sky.rs hoists it the same way — this is the hemi integrator's inner loop).
float3 sky_scatter(float3 dir, float mu, float3 t_sun) {
    float ph_r = 0.0596831 * (1.0 + mu * mu);
    float g2 = SKY_MIE_G * SKY_MIE_G;
    float den = max(1.0 + g2 - 2.0 * SKY_MIE_G * mu, 1e-4);
    float ph_m = 0.0795775 * (1.0 - g2) / (den * sqrt(den));

    float m_view = 1.0 / (abs(dir.y) + 0.15);
    float3 t_view = exp(-(SKY_BETA_R + SKY_BETA_M) * m_view);

    return (SKY_BETA_R * ph_r + SKY_BETA_M * ph_m) * m_view * t_view * t_sun * SKY_DOME_SCALE;
}

// The smooth scattering dome — NO sun disc. Defined over the FULL sphere (the
// SH projection integrates all of it); below the horizon is the ground bouncing
// the dome back, blended over a band so order-2 SH doesn't ring on a step.
//
// The blend band is narrow, so each end returns EARLY rather than evaluating
// both scatters and lerping one away at weight 0 or 1 (sky.rs, same shape).
// sky_scale is the time-of-day brightness (cbuffer; sky.rs::dome's `scale`):
// exactly 1.0 untouched (bit-preserving multiply), the MOON_DOME_FRAC floor at
// night — when `sun` is the MOON and this same model renders the moonlit sky.
float3 sky_dome(float3 d) {
    float3 t_sun = exp(-(SKY_BETA_R + SKY_BETA_M) / (max(sun.y, 0.0) + 0.15));
    float t = saturate((d.y + 0.05) / 0.10);
    if (t >= 1.0) return sky_scatter(d, clamp(dot(d, sun.xyz), -1.0, 1.0), t_sun) * sky_scale;
    float3 dm = float3(d.x, abs(d.y), d.z);
    float3 ground =
        sky_scatter(dm, clamp(dot(dm, sun.xyz), -1.0, 1.0), t_sun) * SKY_GROUND_ALBEDO;
    if (t <= 0.0) return ground * sky_scale;
    float3 sky = sky_scatter(d, clamp(dot(d, sun.xyz), -1.0, 1.0), t_sun);
    return lerp(ground, sky, t) * sky_scale;
}

// The disc, ANTIALIASED against the ray's angular footprint (sky::disc). The
// limb really is a hard step, but a RAY has a footprint — a pixel that half
// covers the sun gets half its radiance. Without this the edge is a binary
// per-ray test: jagged still, crawling under motion at 1 spp. half_angle = 0
// reproduces the hard step exactly.
//
// The angular radius is DERIVED from sun.w (= cos of it), never re-declared as
// a literal here: sky::SUN_ANGULAR_RADIUS is a knob its own doc invites you to
// turn (narrowing it sharpens shadows and brightens the disc without moving
// exposure), and a second copy on this side would leave the AA band ramping
// around the old angle while the hard-step test tracked the new one — a bright
// or dark annulus on the GPU sun that no gate compares against the CPU's.
float3 sky_disc(float3 d, float half_angle) {
    float c = clamp(dot(d, sun.xyz), -1.0, 1.0);
    if (half_angle <= 1e-7) return c >= sun.w ? sun_l.xyz : float3(0.0, 0.0, 0.0);
    float radius = acos(clamp(sun.w, -1.0, 1.0));
    float theta = acos(c);
    float cov = saturate((radius + half_angle - theta) / (2.0 * half_angle));
    return sun_l.xyz * cov;
}

// --- Stars (sky.rs::stars — term-for-term; change both together) -------------
//
// Procedural twinkling stars, DISPLAY paths only like the disc (never
// sky_dome/gather). Deterministic, zero rng draws; `twinkle` is the frame
// index on primary misses, 0 on secondary paths (matching the CPU, which has
// no frame in scope there). Gated by the cbuffer `night` — exactly 0.0 by
// day, and the guard is a BRANCH, so day kernels are bit-identical.
static const float STAR_ANGULAR_RADIUS = 0.03 * 3.14159265 / 180.0;
static const float STAR_E     = 2.0e-6;
static const uint  STAR_CELLS = 64u;
static const float STAR_L_MAX = 4096.0;

float star_hash01(uint h) { return float(h >> 8u) * (1.0 / 16777216.0); }
float star_rise(float y, float lo, float hi) {
    float t = saturate((y - lo) / (hi - lo));
    return t * t * (3.0 - 2.0 * t);
}

float3 sky_stars(float3 d, float half_angle, uint twinkle) {
    if (night <= 0.0 || d.y <= 0.0) return float3(0.0, 0.0, 0.0);
    // Dominant-axis cube face + in-face coords: hash cells over the sphere.
    float3 ad = abs(d);
    uint face; float b1; float b2; float ma;
    if (ad.x >= ad.y && ad.x >= ad.z) { face = d.x > 0.0 ? 0u : 1u; b1 = d.y; b2 = d.z; ma = ad.x; }
    else if (ad.y >= ad.z)            { face = d.y > 0.0 ? 2u : 3u; b1 = d.x; b2 = d.z; ma = ad.y; }
    else                              { face = d.z > 0.0 ? 4u : 5u; b1 = d.x; b2 = d.y; ma = ad.z; }
    float u = (b1 / ma) * 0.5 + 0.5;
    float v = (b2 / ma) * 0.5 + 0.5;
    uint n = STAR_CELLS;
    uint cx = min(uint(u * float(n)), n - 1u);
    uint cy = min(uint(v * float(n)), n - 1u);
    uint seed = face * n * n + cy * n + cx;
    uint h0 = pcg_mix(seed);
    if ((h0 & 0xffu) >= 102u) return float3(0.0, 0.0, 0.0); // ~40% occupancy
    uint h1 = pcg_mix(h0);
    uint h2 = pcg_mix(h1);
    uint h3 = pcg_mix(h2);
    // Inset sub-position (inner 80% — single-cell lookup, no neighbor scan),
    // mapped back through the face parameterization.
    float su = (float(cx) + 0.1 + 0.8 * star_hash01(h1)) / float(n) * 2.0 - 1.0;
    float sv = (float(cy) + 0.1 + 0.8 * star_hash01(h2)) / float(n) * 2.0 - 1.0;
    float3 sdir;
    if      (face == 0u) sdir = float3( 1.0, su, sv);
    else if (face == 1u) sdir = float3(-1.0, su, sv);
    else if (face == 2u) sdir = float3(su,  1.0, sv);
    else if (face == 3u) sdir = float3(su, -1.0, sv);
    else if (face == 4u) sdir = float3(su, sv,  1.0);
    else                 sdir = float3(su, sv, -1.0);
    sdir = normalize(sdir);
    // Energy-conserving Gaussian splat over angle (theta^2 = 2(1-cos), exact
    // enough at star scales): irradiance-authored, so a star delivers the
    // same energy whatever the footprint — no crawl, no resolution dimming.
    float sigma = max(half_angle * 0.5, STAR_ANGULAR_RADIUS);
    float theta2 = max(2.0 * (1.0 - dot(d, sdir)), 0.0);
    float g = exp(-theta2 / (2.0 * sigma * sigma));
    if (g < 1e-4) return float3(0.0, 0.0, 0.0);
    float tier = 0.25 * float(1u << (h3 & 3u));
    float warm = star_hash01(pcg_mix(h3));
    float3 tint = lerp(float3(0.75, 0.85, 1.0), float3(1.0, 0.85, 0.7), warm);
    float tw = 0.75 + 0.25 * star_hash01(pcg_mix(seed ^ pcg_mix(twinkle >> 3u)));
    float l = min(STAR_E * tier / (6.2831853 * sigma * sigma), STAR_L_MAX);
    return tint * (l * g * tw * night * star_rise(d.y, 0.0, 0.05));
}

// --- Volumetric clouds (src/clouds.rs — term-for-term; change both together) --
//
// A drifting 2.5D coverage slab carved by 3D erosion, marched TWO-PHASE with
// a per-(pixel, frame, sample) DITHERED phase (cloud_dither_k — a pure
// integer hash + spp stratification, zero rng draws like everything in the
// sky). Display paths see the whole infinity backdrop (dome + disc + stars)
// extinguished through the layer plus its sun-lit scatter; the direct loop
// multiplies its sun by cloud_sun_transmittance (shade.hlsli).
// sky_dome/gather paths NEVER see clouds — the SH ambient and the hemi
// integrators stay cloud-free by design. Guards are BRANCHES on FLAG_CLOUDS /
// the per-ray miss, so a --no-clouds session (and every cloud-free ray) is
// bit-identical by construction.

static const float CLOUD_BASE_K       = 1.6;
static const float CLOUD_THICK_K      = 0.8;
static const float CLOUD_MIN_DY       = 0.05;
static const float CLOUD_FADE_BAND    = 0.10;
// Sun elevations at/below this cast no cloud shadow; the shadow eases out
// over CLOUD_FADE_BAND above it (clouds.rs::CLOUD_SUN_MIN_Y).
static const float CLOUD_SUN_MIN_Y    = 0.05;
static const float CLOUD_SCALE_K      = 2.6;
static const float CLOUD_WIND_SPEED_K = 0.02;
// Coverage octave-1 amplitude; the retired detail octaves ride as their mean
// (CLOUD_REST_MEAN). Coverage is 2D (where clouds ARE); the 3D EROSION
// octaves are what shape they are (clouds.rs).
static const float CLOUD_AMP1         = 0.3;
static const float CLOUD_REST_MEAN    = 0.1;
static const float CLOUD_EROSION      = 0.4;
// xz lean per unit altitude above base — the strata lean downwind.
static const float CLOUD_SHEAR        = 0.5;
static const float CLOUD_THRESH       = 0.60;
static const float CLOUD_SOFT         = 0.14;
static const float CLOUD_TAU          = 3.5;
static const uint  CLOUD_STEPS        = 6u;  // coarse occupancy probes
static const uint  CLOUD_FINE         = 3u;  // fine sub-steps per occupied one
static const float CLOUD_MAX_STEP_K   = 3.0;
static const float CLOUD_SUN_STEP_K   = 0.25;
static const float CLOUD_ALBEDO       = 0.92;
static const float CLOUD_AMB_K        = 1.0;
static const float CLOUD_MS           = 0.06;   // multi-scatter floor (clouds.rs)
static const float CLOUD_G_FWD        = 0.60;
static const float CLOUD_G_BACK       = -0.15;
static const float CLOUD_LOBE_MIX     = 0.7;
// The static 3D curl wind field (clouds.rs) — wavelength, amplitude, and the
// vertical fraction of the displacement.
static const float CLOUD_CURL_SCALE_K = 6.5;
static const float CLOUD_CURL_AMP_K   = 0.8;
static const float CLOUD_CURL_YSCALE  = 0.3;

// Gates ONLY the frame term of the march-phase dither — clouds.rs's
// CLOUD_TEMPORAL_DITHER, keep in lockstep (false = the static fallback).
static const bool CLOUD_TEMPORAL_DITHER = true;

// The march-phase dither seed (clouds::dither_j): a PURE integer hash of
// (pixel, frame) — u32-exact CPU<->GPU, consuming NOTHING from any shading
// rng stream (n = 0 is the XOR identity: bit-identical to the pre-temporal
// dither). The frame term turns march grain into ordinary temporal noise
// that accumulation/RR/XeSS integrate; it is NOT the SKY_J lesson's case —
// that gate rejected the sky-tile fill's DIRECTION set changing per frame,
// while the phase rides the same directions and applies to every sample
// symmetrically. Do NOT "clean up" the dither back to a fixed 0.5: with a
// fixed phase the sample altitudes are ray-independent and every smooth
// field renders as N nested step-entry contours (the wedding-cake bug,
// twice shipped).
float cloud_dither(uint2 px, uint n) {
    n = CLOUD_TEMPORAL_DITHER ? n : 0u;
    return star_hash01(pcg_mix(px.x * 0x9E3779B9u ^ px.y * 0x85EBCA6Bu
        ^ n * 0x3C6EF372u ^ 0x68E31DA4u));
}

// The per-(pixel, frame, SAMPLE) phase (clouds::dither_jk): hashed per
// (pixel, frame), STRATIFIED across the frame's spp samples — N evenly
// spaced phases integrate the march near-exactly, which is what makes --spp
// soften the clouds. k = 0 adds an exact 0.0 (bitwise cloud_dither); the
// wrap is a conditional subtract, exact for s < 2.
float cloud_dither_k(uint2 px, uint n, uint k, uint nspp) {
    float s = cloud_dither(px, n) + float(k) / float(nspp);
    return s >= 1.0 ? s - 1.0 : s;
}

// Lattice hash: pcg_mix over a corner mix — int -> uint is bit-preserving,
// matching the CPU's wrapping `as u32` casts exactly (u32-exact pipeline).
float cloud_cell_hash(int i, int j, uint oct) {
    return star_hash01(pcg_mix(uint(i) * 0x9E3779B9u ^ uint(j) * 0x85EBCA6Bu ^ oct * 0xC2B2AE3Du));
}

// 3D lattice corner hash — the 2D mix plus a third axis constant.
float cloud_cell_hash3(int i, int j, int k, uint oct) {
    return star_hash01(pcg_mix(uint(i) * 0x9E3779B9u ^ uint(j) * 0x85EBCA6Bu
        ^ uint(k) * 0x27D4EB2Fu ^ oct * 0xC2B2AE3Du));
}

// 2D value noise: floor() then cast (NEVER a bare int() truncation — negative
// coordinates would mirror), smoothstep fade, bilerp of 4 corner hashes.
float cloud_vnoise(float2 q, uint oct) {
    float fx = floor(q.x);
    float fy = floor(q.y);
    int i = int(fx);
    int j = int(fy);
    float tx = q.x - fx;
    float ty = q.y - fy;
    float ux = tx * tx * (3.0 - 2.0 * tx);
    float uy = ty * ty * (3.0 - 2.0 * ty);
    float h00 = cloud_cell_hash(i, j, oct);
    float h10 = cloud_cell_hash(i + 1, j, oct);
    float h01 = cloud_cell_hash(i, j + 1, oct);
    float h11 = cloud_cell_hash(i + 1, j + 1, oct);
    float a = h00 + (h10 - h00) * ux;
    float b = h01 + (h11 - h01) * ux;
    return a + (b - a) * uy;
}

// 3D value noise (clouds.rs::vnoise3): smoothstep-faded trilerp of 8 corner
// hashes — the erosion field's genuinely-3D noise.
float cloud_vnoise3(float3 q, uint oct) {
    float fx = floor(q.x);
    float fy = floor(q.y);
    float fz = floor(q.z);
    int i = int(fx);
    int j = int(fy);
    int k = int(fz);
    float tx = q.x - fx;
    float ty = q.y - fy;
    float tz = q.z - fz;
    float ux = tx * tx * (3.0 - 2.0 * tx);
    float uy = ty * ty * (3.0 - 2.0 * ty);
    float uz = tz * tz * (3.0 - 2.0 * tz);
    float h000 = cloud_cell_hash3(i, j, k, oct);
    float h100 = cloud_cell_hash3(i + 1, j, k, oct);
    float h010 = cloud_cell_hash3(i, j + 1, k, oct);
    float h110 = cloud_cell_hash3(i + 1, j + 1, k, oct);
    float h001 = cloud_cell_hash3(i, j, k + 1, oct);
    float h101 = cloud_cell_hash3(i + 1, j, k + 1, oct);
    float h011 = cloud_cell_hash3(i, j + 1, k + 1, oct);
    float h111 = cloud_cell_hash3(i + 1, j + 1, k + 1, oct);
    float a0 = h000 + (h100 - h000) * ux;
    float b0 = h010 + (h110 - h010) * ux;
    float c0 = a0 + (b0 - a0) * uy;
    float a1 = h001 + (h101 - h001) * ux;
    float b1 = h011 + (h111 - h011) * ux;
    float c1 = a1 + (b1 - a1) * uy;
    return c0 + (c1 - c0) * uz;
}

// ANALYTIC gradient of cloud_vnoise3 w.r.t. its unit-cell coordinate
// (clouds.rs::vnoise3_grad): same 8 corner hashes, the Hermite fade's own
// derivative, per-axis bilerps of the corner differences. C1 across cell
// edges (the fade's derivative vanishes there), so the curl field it feeds
// has no creases.
float3 cloud_vnoise3_grad(float3 q, uint oct) {
    float fx = floor(q.x);
    float fy = floor(q.y);
    float fz = floor(q.z);
    int i = int(fx);
    int j = int(fy);
    int k = int(fz);
    float tx = q.x - fx;
    float ty = q.y - fy;
    float tz = q.z - fz;
    float ux = tx * tx * (3.0 - 2.0 * tx);
    float uy = ty * ty * (3.0 - 2.0 * ty);
    float uz = tz * tz * (3.0 - 2.0 * tz);
    float dux = 6.0 * tx * (1.0 - tx);
    float duy = 6.0 * ty * (1.0 - ty);
    float duz = 6.0 * tz * (1.0 - tz);
    float h000 = cloud_cell_hash3(i, j, k, oct);
    float h100 = cloud_cell_hash3(i + 1, j, k, oct);
    float h010 = cloud_cell_hash3(i, j + 1, k, oct);
    float h110 = cloud_cell_hash3(i + 1, j + 1, k, oct);
    float h001 = cloud_cell_hash3(i, j, k + 1, oct);
    float h101 = cloud_cell_hash3(i + 1, j, k + 1, oct);
    float h011 = cloud_cell_hash3(i, j + 1, k + 1, oct);
    float h111 = cloud_cell_hash3(i + 1, j + 1, k + 1, oct);
    float xa = (h100 - h000) + ((h110 - h010) - (h100 - h000)) * uy;
    float xb = (h101 - h001) + ((h111 - h011) - (h101 - h001)) * uy;
    float ya = (h010 - h000) + ((h110 - h100) - (h010 - h000)) * ux;
    float yb = (h011 - h001) + ((h111 - h101) - (h011 - h001)) * ux;
    float za = (h001 - h000) + ((h101 - h100) - (h001 - h000)) * ux;
    float zb = (h011 - h010) + ((h111 - h110) - (h011 - h010)) * ux;
    return float3(
        dux * (xa + (xb - xa) * uz),
        duy * (ya + (yb - ya) * uz),
        duz * (za + (zb - za) * uy));
}

// The low-frequency 3D curl wind field as a STATIC sampling displacement
// (clouds.rs::curl_offset): v = grad(psi1) x grad(psi2), soft-normalized to
// |v| < 1 (the hard bound the march's skip margin leans on), y scaled by
// CLOUD_CURL_YSCALE. Time-independent, sampled at raw world coordinates —
// clouds deform/wander/billow as the wind carries them through it.
float3 cloud_curl_offset(float3 p) {
    float lc = CLOUD_CURL_SCALE_K * SCENE_DIAG;
    float3 q = p * (1.0 / lc);
    float3 v = cross(cloud_vnoise3_grad(q, 6u), cloud_vnoise3_grad(q, 7u));
    v *= 1.0 / (1.0 + length(v));
    return float3(v.x, CLOUD_CURL_YSCALE * v.y, v.z) * (CLOUD_CURL_AMP_K * SCENE_DIAG);
}

// Per-octave anti-alias attenuation (clouds.rs::oct_t): full detail while
// the octave's wavelength is resolved by the sampling footprint w (the
// march's step length), collapsing to the octave MEAN once w >= l —
// point-sampling unresolvable detail renders each grazing march step as its
// own separated bead. Fully-attenuated octaves skip their noise evals.
float cloud_oct_t(float w, float l) { return saturate(2.0 - 2.0 * w / l); }

// THE advection expression (clouds.rs::advect), factored so cover and
// erosion share ONE copy — the CPU's advection-identity gate premise. The
// wind term stays the LAST subtraction.
float2 cloud_advect(float3 p) {
    float base = CLOUD_BASE_K * SCENE_DIAG;
    float2 wind = float2(0.932, 0.362) * (CLOUD_WIND_SPEED_K * SCENE_DIAG);
    float lean = CLOUD_SHEAR * (p.y - base);
    return float2(p.x + 0.932 * lean, p.z + 0.362 * lean) - wind * CLOUD_TIME;
}

// THE 2D coverage field (clouds.rs::cloud_cover) — where clouds ARE. Shared
// verbatim by the lo (shadow/lighting) and visible (erosion-carved) fields,
// which is what makes density <= density_lo a bitwise theorem (G8). The
// staged cutoff is the clear-sky fast path AND exact: octave 0's partial sum
// plus the full remaining amplitude bounds the sum, so the early 0.0 is the
// value the remap would produce (value-continuous).
float cloud_cover(float3 p, float w) {
    float l0 = CLOUD_SCALE_K * SCENE_DIAG;
    float2 q = cloud_advect(p) * (1.0 / l0);
    float t0 = cloud_oct_t(w, l0);
    float n0 = 0.5;
    if (t0 >= 1.0) n0 = cloud_vnoise(q, 0u);
    else if (t0 > 0.0) n0 = 0.5 + (cloud_vnoise(q, 0u) - 0.5) * t0;
    float c0 = 0.5 * n0;
    if (c0 + CLOUD_AMP1 + CLOUD_REST_MEAN <= CLOUD_THRESH) return 0.0;
    float t1 = cloud_oct_t(w, l0 * 0.5);
    float n1 = 0.5;
    if (t1 >= 1.0) n1 = cloud_vnoise(q * 2.0, 1u);
    else if (t1 > 0.0) n1 = 0.5 + (cloud_vnoise(q * 2.0, 1u) - 0.5) * t1;
    float c1 = c0 + CLOUD_AMP1 * n1;
    return saturate((c1 + CLOUD_REST_MEAN - CLOUD_THRESH) / CLOUD_SOFT);
}

// The column's local top (clouds.rs::cloud_top) — shared by prof and the
// march's interval-window skip.
float cloud_top(float cover, float base, float thick) {
    return base + thick * (0.30 + 0.70 * cover);
}

// The coverage-driven vertical window (clouds.rs::cloud_prof): fast rise off
// the base, taper to the column's OWN top (taller where denser).
float cloud_prof(float py, float cover, float base, float thick) {
    float top_l = cloud_top(cover, base, thick);
    return saturate((py - base) / (0.20 * thick))
        * saturate((top_l - py) / (0.30 * thick));
}

// The 3D erosion factor (clouds.rs::erosion3): ONE octave of genuinely-3D
// value noise at l0/4 (a second at l0/8 was tried and rejected on cost —
// clouds.rs) — what breaks the nested-level-set structure a 2D field is
// stuck with. xz rides the SAME advection expression (the erosion drifts and
// shears with its cloud); y raw over the same wavelength.
float cloud_erosion3(float3 p, float w) {
    float l0 = CLOUD_SCALE_K * SCENE_DIAG;
    float le = l0 * 0.25;
    float2 uv = cloud_advect(p);
    float3 qe = float3(uv.x, p.y, uv.y) * (1.0 / le);
    float te0 = cloud_oct_t(w, le);
    if (te0 >= 1.0) return cloud_vnoise3(qe, 5u);
    if (te0 > 0.0) return 0.5 + (cloud_vnoise3(qe, 5u) - 0.5) * te0;
    return 0.5;
}

// The VISIBLE density at an ALREADY-WARPED point (clouds.rs::density_at):
// shared 2D coverage carved by the 3D erosion inside the coverage-driven
// window. vnoise3 runs only where cover*prof > 0. The public _f wrapper
// applies the exact per-point curl warp; the march passes a warp HOISTED
// per coarse interval.
float cloud_density_at(float3 pw, float w) {
    float base = CLOUD_BASE_K * SCENE_DIAG;
    float thick = CLOUD_THICK_K * SCENE_DIAG;
    float cover = cloud_cover(pw, w);
    if (cover <= 0.0) return 0.0;
    float prof = cloud_prof(pw.y, cover, base, thick);
    if (prof <= 0.0) return 0.0;
    float s3 = cloud_erosion3(pw, w);
    float eroded = saturate(cover - CLOUD_EROSION * (1.0 - s3));
    return eroded * prof;
}
float cloud_density_f(float3 p, float w) {
    return cloud_density_at(p + cloud_curl_offset(p), w);
}
float cloud_density(float3 p) { return cloud_density_f(p, 0.0); }

// The 2D SHADOW/LIGHTING field (cover*prof, no erosion) — the hot one:
// sun_transmittance, the march's sun probes, the rough reflection march.
// Sees the SAME curl warp as the visible field (the shadow must track the
// cloud that casts it; G8 requires the shared domain).
float cloud_density_lo_at(float3 pw, float w) {
    float base = CLOUD_BASE_K * SCENE_DIAG;
    float thick = CLOUD_THICK_K * SCENE_DIAG;
    float cover = cloud_cover(pw, w);
    if (cover <= 0.0) return 0.0;
    return cover * cloud_prof(pw.y, cover, base, thick);
}
float cloud_density_lo_f(float3 p, float w) {
    return cloud_density_lo_at(p + cloud_curl_offset(p), w);
}
float cloud_density_lo(float3 p) { return cloud_density_lo_f(p, 0.0); }

float cloud_hg(float mu, float g) {
    float g2 = g * g;
    float den = max(1.0 + g2 - 2.0 * g * mu, 1e-4);
    return (1.0 - g2) / (4.0 * 3.14159265 * den * sqrt(den));
}

// The two-phase adaptive march (clouds::along_k). Returns false when the ray
// met no cloud — the caller's backdrop must pass through UNTOUCHED (the
// bit-identity arm). `j` in [0,1) is the dither phase
// (cloud_dither_k(px, frame, s, spp) where a pixel exists — per (pixel,
// frame, sample); 0.5 on pixel-less paths).
// Phase A: CLOUD_STEPS coarse DITHERED probes of the 2D cover only (a point
// test of prof would re-quantize the top surface — the ring bug). Phase B:
// occupied coarse intervals get CLOUD_FINE fine sub-steps of the full 3D
// density at the FINE footprint, plus ONE shared sun probe per coarse step.
// `amb` is dome(d) * CLOUD_AMB_K; `sun` may be the MOON at night.
bool clouds_along(float3 o, float3 d, float3 amb, float j, out float t_out, out float3 sc_out) {
    t_out = 1.0;
    sc_out = float3(0.0, 0.0, 0.0);
    if ((flags & FLAG_CLOUDS) == 0u || d.y <= CLOUD_MIN_DY) return false;
    float base = CLOUD_BASE_K * SCENE_DIAG;
    float thick = CLOUD_THICK_K * SCENE_DIAG;
    if (o.y >= base) return false; // 2.5D: modeled from below only
    float sigma_t = CLOUD_TAU / thick;
    float t0 = (base - o.y) / d.y;
    float dt_c = min(thick / (float(CLOUD_STEPS) * d.y), CLOUD_MAX_STEP_K * thick);
    float dt_f = dt_c / float(CLOUD_FINE);
    float mu = clamp(dot(d, sun.xyz), -1.0, 1.0);
    float phase = CLOUD_LOBE_MIX * cloud_hg(mu, CLOUD_G_FWD)
        + (1.0 - CLOUD_LOBE_MIX) * cloud_hg(mu, CLOUD_G_BACK);
    // Sunlight at cloud altitude: e_over_pi rebuilt to irradiance, tinted by
    // the dome's own sun-path transmittance (sky.rs::t_sun_path) — clouds
    // redden at sunset in lockstep with the sky.
    float3 sun_col = sun_e.xyz * 3.14159265
        * exp(-(SKY_BETA_R + SKY_BETA_M) / (max(sun.y, 0.0) + 0.15));
    float l_sun = CLOUD_SUN_STEP_K * thick / max(sun.y, 0.35);
    float t_acc = 1.0;
    bool opaque = false;
    // The curl warp, HOISTED per RAY and folded into a WARPED RAY ORIGIN
    // (clouds.rs::along_k — per-coarse-step warps measured +21 ms CPU and
    // ~2x the wavefront's per-sample cost): every sample position below is
    // ow + d*t, zero extra inner-loop arithmetic, and the skip's
    // field-space altitude is exact. The slab geometry (t0, dt_c) stays a
    // function of the REAL o.
    float3 ow = o + cloud_curl_offset(o + d * (t0 + 0.5 * float(CLOUD_STEPS) * dt_c));
    [unroll]
    for (uint i = 0u; i < CLOUD_STEPS; i++) {
        if (opaque) break;
        // Phase A: DITHERED coarse occupancy probe — 2D cover only, sampled
        // through the ray's hoisted curl warp.
        float tc = t0 + (float(i) + j) * dt_c;
        float3 pc = ow + d * tc;
        float cov = cloud_cover(pc, dt_c);
        if (cov <= 0.0) continue;
        // Interval-window skip (clouds.rs): fine-marching the empty air
        // above the column's own top was a measured +17 ms CPU regression.
        // The per-RAY hoist makes the warp constant along this chord, so
        // ow.y + d.y*t IS the field-space altitude — exact, no margin.
        float y_lo = ow.y + d.y * (t0 + float(i) * dt_c);
        if (y_lo >= cloud_top(cov, base, thick) + 0.1 * thick) continue;
        // One sun probe AND one lighting transmittance per occupied coarse
        // step, shared by its sub-steps (clouds.rs — a per-fine-sample exp
        // pair was a measured cost driver; the coarse cover is the local-
        // density proxy). The probe rides the ray's hoisted warp.
        float rho_sun = cloud_density_lo_at(pc + sun.xyz * l_sun, 0.0);
        float t_sun = exp(-((rho_sun + 0.5 * cov) * sigma_t * l_sun));
        float3 s = (sun_col * ((phase + CLOUD_MS) * t_sun) + amb) * CLOUD_ALBEDO;
        // Phase B: fine sub-steps, full 3D density, same dither phase.
        // [loop], not [unroll]: 18 inlined density bodies would bloat every
        // kernel that pastes this file (the LEAF_NO_FB VGPR lesson).
        [loop]
        for (uint m = 0u; m < CLOUD_FINE; m++) {
            // The COARSE cover is reused across the sub-interval (clouds.rs:
            // re-evaluating the smooth 2D placement per fine sample was a
            // measured cost with no visible gain); only erosion + the
            // vertical window run at fine resolution.
            float tf = t0 + float(i) * dt_c + (float(m) + j) * dt_f;
            float3 p = ow + d * tf;
            float prof = cloud_prof(p.y, cov, base, thick);
            if (prof <= 0.0) continue;
            float s3 = cloud_erosion3(p, dt_f);
            float rho = saturate(cov - CLOUD_EROSION * (1.0 - s3)) * prof;
            if (rho <= 0.0) continue;
            float a = exp(-(rho * sigma_t * dt_f));
            sc_out += s * (t_acc * (1.0 - a));
            t_acc *= a;
            // Opaque-core break (clouds.rs): < 0.5% left to add.
            if (t_acc < 0.005) { opaque = true; break; }
        }
    }
    if (t_acc >= 1.0) return false; // every sample empty — bit-clean fallthrough
    float fade = saturate((d.y - CLOUD_MIN_DY) / CLOUD_FADE_BAND);
    t_out = 1.0 + (t_acc - 1.0) * fade;
    sc_out *= fade;
    return true;
}

// The ROUGH-path march (clouds::along_rough): 2 fixed midpoints over the
// 2-octave field, for SECONDARY specular paths — a reflected sky is seen
// through the GGX lobe's blur, and the full march at the reflection-miss
// site was the largest single share of the cloud layer's cost.
static const uint CLOUD_ROUGH_STEPS = 2u;
bool clouds_along_rough(float3 o, float3 d, float3 amb, out float t_out, out float3 sc_out) {
    t_out = 1.0;
    sc_out = float3(0.0, 0.0, 0.0);
    if ((flags & FLAG_CLOUDS) == 0u || d.y <= CLOUD_MIN_DY) return false;
    float base = CLOUD_BASE_K * SCENE_DIAG;
    float thick = CLOUD_THICK_K * SCENE_DIAG;
    if (o.y >= base) return false;
    float sigma_t = CLOUD_TAU / thick;
    float t0 = (base - o.y) / d.y;
    float dt = min(thick / (float(CLOUD_ROUGH_STEPS) * d.y), CLOUD_MAX_STEP_K * thick);
    float mu = clamp(dot(d, sun.xyz), -1.0, 1.0);
    float phase = CLOUD_LOBE_MIX * cloud_hg(mu, CLOUD_G_FWD)
        + (1.0 - CLOUD_LOBE_MIX) * cloud_hg(mu, CLOUD_G_BACK);
    float3 sun_col = sun_e.xyz * 3.14159265
        * exp(-(SKY_BETA_R + SKY_BETA_M) / (max(sun.y, 0.0) + 0.15));
    float l_sun = CLOUD_SUN_STEP_K * thick / max(sun.y, 0.35);
    float t_acc = 1.0;
    // ONE curl warp for the whole rough march, folded into the ray origin
    // (clouds.rs::along_rough): a reflected sky is seen through the GGX
    // lobe's blur.
    float3 ow = o + cloud_curl_offset(o + d * (t0 + 0.5 * dt));
    [unroll]
    for (uint k = 0u; k < CLOUD_ROUGH_STEPS; k++) {
        float tk = t0 + (float(k) + 0.5) * dt;
        float3 p = ow + d * tk;
        float rho = cloud_density_lo_at(p, dt);
        if (rho <= 0.0) continue;
        float a = exp(-(rho * sigma_t * dt));
        float rho_sun = cloud_density_lo_at(p + sun.xyz * l_sun, 0.0);
        float t_sun = exp(-((rho_sun + 0.5 * rho) * sigma_t * l_sun));
        float3 s = (sun_col * ((phase + CLOUD_MS) * t_sun) + amb) * CLOUD_ALBEDO;
        sc_out += s * (t_acc * (1.0 - a));
        t_acc *= a;
    }
    if (t_acc >= 1.0) return false;
    float fade = saturate((d.y - CLOUD_MIN_DY) / CLOUD_FADE_BAND);
    t_out = 1.0 + (t_acc - 1.0) * fade;
    sc_out *= fade;
    return true;
}

// Cloud shadow toward the sun (clouds::sun_transmittance): exactly two
// density evals at the slab's quarter heights, Beer over the slant path.
// The unshadowed arms return an EXACT 1.0 (x * 1.0 is bit-preserving); the
// low-sun band eases the shadow out so a TOD scrub never pops it.
float cloud_sun_transmittance(float3 p) {
    if ((flags & FLAG_CLOUDS) == 0u || sun.y <= CLOUD_SUN_MIN_Y) return 1.0;
    float base = CLOUD_BASE_K * SCENE_DIAG;
    float thick = CLOUD_THICK_K * SCENE_DIAG;
    if (p.y >= base) return 1.0;
    float s_lo = (base + 0.25 * thick - p.y) / sun.y;
    float s_hi = (base + 0.75 * thick - p.y) / sun.y;
    // ONE curl warp at the probes' midpoint, folded into the surface point
    // and shared by both evals — the hot path (two evals per shade on every
    // lit pixel); clouds.rs, term for term.
    float3 pw = p + cloud_curl_offset(p + sun.xyz * (0.5 * (s_lo + s_hi)));
    float rho =
        0.5 * (cloud_density_lo_at(pw + sun.xyz * s_lo, 0.0)
             + cloud_density_lo_at(pw + sun.xyz * s_hi, 0.0));
    if (rho <= 0.0) return 1.0;
    float t = exp(-(rho * CLOUD_TAU / max(sun.y, 0.25)));
    // Low-sun ease-out (clouds.rs, term for term). fade >= 1 returns the
    // Beer factor VERBATIM — 1 + (t-1) is not bit-preserving for small t.
    float fade = saturate((sun.y - CLOUD_SUN_MIN_Y) / CLOUD_FADE_BAND);
    return fade >= 1.0 ? t : 1.0 + (t - 1.0) * fade;
}

// What an escaping ray SEES: the backdrop (dome + disc + stars) through the
// cloud layer, plus the layer's scatter. DISPLAY paths only. `o` is the ray's
// ORIGIN — the slab is finite-altitude, so unlike everything at infinity the
// start matters (sky.rs::radiance, same signature change). `half_angle` is
// the ray's angular footprint (primary: pixel_cone/2); `twinkle` is the frame
// index on primary misses, 0 on secondary paths.
// `j` is the cloud march's dither phase (cloud_dither_k(px, frame, s, spp)
// where a pixel exists — every kernel's miss and the sky fill, per (pixel,
// frame, sample); 0.5 on pixel-less paths like the glass miss inside
// shade.hlsli, which the temporal dither deliberately excludes).
float3 sky_radiance(float3 o, float3 d, float half_angle, uint twinkle, float j) {
    float3 dm = sky_dome(d);
    float3 backdrop = dm + sky_disc(d, half_angle) + sky_stars(d, half_angle, twinkle);
    if ((flags & FLAG_CLOUDS) == 0u) return backdrop;
    float ct;
    float3 cs;
    if (!clouds_along(o, d, dm * CLOUD_AMB_K, j, ct, cs)) return backdrop;
    return backdrop * ct + cs;
}

// Balance heuristic (sky::mis_weight). The sun's specular is reachable both by
// light sampling (direct_s) and by the VNDF reflection ray landing in the disc;
// counting both double-counts it AND fires ~1e3-radiance fireflies into FSR's
// un-denoised residual on rough surfaces.
float sky_mis_weight(float p_bsdf, float p_light) {
    float s = p_bsdf + p_light;
    return s > 0.0 ? p_bsdf / s : 0.0;
}

// The light-sampling pdf: uniform in the sun's cone. Omega = 2pi(1 - cos_r).
float sky_light_pdf() { return 1.0 / (6.28318530718 * (1.0 - sun.w)); }
// sun_sample_dir is defined below, after onb().

// glam normalize_or_zero.
float3 normalize_or_zero(float3 v) {
    float l2 = dot(v, v);
    return l2 > 1e-30 ? v * rsqrt(l2) : float3(0.0, 0.0, 0.0);
}

// --- G-buffer pack (upscaler sessions; dlss.rs::GPixel on the GPU) ------------

// Primary-hit surface capture — the shade.rs::PrimarySurface mirror. spec_t:
// reflection-ray hit t, INF when the reflection missed, 0 when none was traced.
// direct_d/direct_s are the post-average direct-light lobes (lap 0 only; the
// two addends of color = kd*(direct_d + ambient) + direct_s) — assignment-only
// copies, zero rng draws, so the same-seed bit-identity gates hold.
struct PrimSurf {
    float3 n;      // shading normal, post face-flip
    float rough;
    float3 albedo; // raw material albedo (diffuse/specular split at the write)
    float metallic;
    float spec_t;
    float3 direct_d; // albedo-free direct diffuse (kd multiplies later)
    float3 direct_s; // direct specular incl. per-sample Fresnel
    float ao;        // AO open fraction, the `ambient = AMBIENT * ao` factor
    float3 ind_s;    // the reflection bounce's whole contribution to color
};

// dlss::GBufs re-hosted as one interleaved plane; the feed kernels fan it out
// into the upscalers' input textures. 88 B/px (GBUF_STRIDE in trace.rs —
// keep in lockstep).
struct GBufPx {
    float4 nr;    // normal.xyz, roughness
    float4 alb_z; // diff_alb.xyz = albedo*(1-metallic), view_z = t*dot(dir, forward)
    float4 spec;  // spec_alb.xyz = lerp(0.04, albedo, metallic) (RGB F0), spec_hit_t
                  // (also the INDIRECT_SPECULAR signal's ray-hit-distance channel)
    float4 mv;    // xy = motion vector in render-res pixels (y-down, current ->
                  // previous); z = prev-camera linear view-Z of the SAME hit
                  // point (FLAG_FSR_SIG — the denoiser MV's B channel differences
                  // it against alb_z.w; sky stores CAM_FAR so the delta is 0);
                  // else 0. w = 0.
    uint4 sig;    // f16x2 packs (dd.x|dd.y, dd.z|ds.x, ds.y|ds.z, 0) of the
                  // DEMODULATED FSR-RR signals — fsr::split_signals' twin,
                  // f16 IS the wire precision. FLAG_FSR_SIG; else 0.
    uint2 sig2;   // the other two FSR-RR signals, same f16x2 packing:
                  // (ao|is.x, is.y|is.z). FLAG_FSR_SIG; else 0.
};
RWStructuredBuffer<GBufPx> gbuf : register(u15);

// f32 -> f16 bits with round-to-nearest-even — NOT the f32tof16 intrinsic
// (the legacy DXIL op truncates). The CPU twin is half::f16::from_f32;
// nppd.hlsl's q16 round-trip and the sig packing below both build on it.
uint f16bits_rtne(float v) {
    uint x = asuint(v);
    uint s = (x >> 16u) & 0x8000u;
    x &= 0x7FFFFFFFu;
    uint h;
    if (x >= 0x47800000u) {            // >= 2^16: overflow -> inf; keep nan
        h = x > 0x7F800000u ? 0x7E00u : 0x7C00u;
    } else if (x >= 0x38800000u) {     // f16 normal range [2^-14, 2^16)
        h = (((x >> 23u) - 112u) << 10u) | ((x >> 13u) & 0x3FFu);
        uint rem = x & 0x1FFFu;
        // RTNE; a mantissa carry may bump the exponent, [65520, 65536)
        // correctly lands on inf.
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h += 1u;
    } else if (x >= 0x33000000u) {     // f16 subnormal range [2^-25, 2^-14)
        uint shift = 126u - (x >> 23u); // 14..24
        uint m = (x & 0x7FFFFFu) | 0x800000u;
        h = m >> shift;
        uint rem = m & ((1u << shift) - 1u);
        uint halfb = 1u << (shift - 1u);
        if (rem > halfb || (rem == halfb && (h & 1u) != 0u)) h += 1u;
    } else {                           // < 2^-25 underflows to signed zero
        h = 0u;
    }
    return s | h;
}

// fsr::f16_sat's twin: saturate into the finite f16 range (an inf on a signal
// plane turns the residual remainder into inf*0 = NaN downstream).
uint f16bits_sat(float v) { return f16bits_rtne(clamp(v, -65504.0, 65504.0)); }

uint pack_h2(float lo, float hi) { return f16bits_sat(lo) | (f16bits_sat(hi) << 16u); }

// fsr's GPU wire chain for the sqrt-encoded RGBA8 albedo planes: one explicit
// 8-bit quantization (fsr::sqrt_wire — the GPU pack stores f32, so unlike the
// CPU's albedo_wire there is no leading f16 rounding). Used identically at
// the sig demodulation here, the feed's residual, and the composite decode —
// consistency of those three sites IS the composite identity.
float sqrt_enc8(float v) { return floor(sqrt(saturate(v)) * 255.0 + 0.5) / 255.0; }
float sqrt_wire(float v) {
    float enc = sqrt_enc8(v);
    return enc * enc;
}
float3 sqrt_wire3(float3 v) { return float3(sqrt_wire(v.x), sqrt_wire(v.y), sqrt_wire(v.z)); }

// camera.rs::CamBasis::project against the PREVIOUS basis: the continuous
// image point a world direction passes through, y-down pixels. ok = false
// means at/behind the old image plane — consumed as mv (0,0) (disocclusion).
float2 project_prev(float3 d, out bool ok) {
    float df = dot(d, prev_forward.xyz);
    ok = df > 0.0;
    if (!ok) return float2(0.0, 0.0);
    float ndx = dot(d, prev_right.xyz) / (dot(prev_right.xyz, prev_right.xyz) * df);
    float ndy = dot(d, prev_up.xyz) / (dot(prev_up.xyz, prev_up.xyz) * df);
    return float2((ndx + 1.0) * 0.5 / prev_origin.w, (1.0 - ndy) * 0.5 / prev_forward.w);
}

float2 gbuf_mv(float3 d, float fx, float fy) {
    if ((flags & FLAG_HAS_PREV) == 0u) return float2(0.0, 0.0);
    bool ok;
    float2 p = project_prev(d, ok);
    return ok ? p - float2(fx, fy) : float2(0.0, 0.0);
}

// render.rs::write_gbuf_hit. (fx, fy) is the jittered sample position — the
// MV is unjittered by construction (both bases are jitter-free and the hit
// point lies on the jittered ray).
void gbuf_write_hit(uint pi, float fx, float fy, float3 dir, float t, PrimSurf ps) {
    if ((flags & FLAG_GBUF) == 0u) return;
    GBufPx g;
    g.nr = float4(ps.n, ps.rough);
    g.alb_z = float4(ps.albedo * (1.0 - ps.metallic), t * dot(dir, cam_forward.xyz));
    float3 spec_alb = lerp(float3(0.04, 0.04, 0.04), ps.albedo, ps.metallic);
    g.spec = float4(spec_alb, isinf(ps.spec_t) ? CAM_FAR : ps.spec_t);
    float3 hit_rel = cam_origin.xyz + dir * t - prev_origin.xyz;
    g.mv = float4(gbuf_mv(hit_rel, fx, fy), 0.0, 0.0);
    g.sig = uint4(0u, 0u, 0u, 0u);
    g.sig2 = uint2(0u, 0u);
    if (flags & FLAG_FSR_SIG) {
        // render.rs's fsr_buf fill: prev-camera linear view-Z of the SAME
        // hit point (no prev camera degrades to "no depth motion"); the
        // demodulation divides direct_s / ind_s by the un-floored WIRE F0 —
        // fsr::split_signals with sqrt_wire in place of albedo_wire (the
        // pack stores f32; the wire quantization happens exactly once).
        g.mv.z = (flags & FLAG_HAS_PREV) ? dot(hit_rel, prev_forward.xyz) : g.alb_z.w;
        float3 sf0w = sqrt_wire3(spec_alb);
        float3 f0_floor = max(sf0w, float3(1e-4, 1e-4, 1e-4));
        float3 dd = ps.direct_d;
        float3 ds = ps.direct_s / f0_floor;
        float3 is = ps.ind_s / f0_floor;
        g.sig = uint4(pack_h2(dd.x, dd.y), pack_h2(dd.z, ds.x), pack_h2(ds.y, ds.z), 0u);
        g.sig2 = uint2(pack_h2(ps.ao, is.x), pack_h2(is.y, is.z));
    }
    gbuf[pi] = g;
}

// render.rs::write_gbuf_sky: depth = far (finite, f16-safe); MV = direction-
// only reprojection (exact for an environment at infinity). FSR sessions:
// sig = 0 (residual = the sky color itself downstream) and prev-Z = far so
// the depth delta is exactly 0 — the sky contract --check-fsr pins.
void gbuf_write_sky(uint pi, float fx, float fy, float3 dir) {
    if ((flags & FLAG_GBUF) == 0u) return;
    GBufPx g;
    g.nr = float4(-dir, 1.0);
    g.alb_z = float4(1.0, 1.0, 1.0, CAM_FAR);
    g.spec = float4(0.0, 0.0, 0.0, 0.0);
    g.mv = float4(gbuf_mv(dir, fx, fy), 0.0, 0.0);
    g.sig = uint4(0u, 0u, 0u, 0u);
    g.sig2 = uint2(0u, 0u);
    if (flags & FLAG_FSR_SIG) {
        g.mv.z = CAM_FAR;
    }
    gbuf[pi] = g;
}

// sh.rs::Sh9::irradiance — cosine-weighted sky irradiance at `n`, DIVIDED BY PI
// (the renderer convention: a uniform sky of radiance L returns exactly L, so
// this drops straight into the slot the old flat AMBIENT constant occupied).
// Ramamoorthi & Hanrahan 2001's closed form: the clamped-cosine kernel is a
// zonal harmonic (A0 = pi, A1 = 2pi/3, A2 = pi/4) and folding those into the
// basis constants collapses the hemisphere integral to these five numbers.
// Clamped at zero: a truncated series of a sky with a horizon step rings, and
// negative irradiance is unphysical. Term-for-term with the Rust; change both
// together.
// The frame's sky, bound to the shared evaluator (sh.hlsli — the same function
// the FSR composite pass calls against its own copy of the coefficients).
float3 sh_irradiance(float3 n) { return sh_irr(sky_sh, n); }

// Duff et al. orthonormal basis; right-handed (t1 x t2 = n) — the hemisphere
// octant orientation relies on it (sphcell::self_test asserts the CPU twin).
void onb(float3 n, out float3 t1, out float3 t2) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    t1 = float3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    t2 = float3(b, s + n.y * n.y * a, -n.y);
}

// Uniform-in-cone sample toward the sun disc (sky::Sun::sample_dir) — the exact
// replacement for the old rect sample, consuming the SAME two draws in the same
// order, which is what keeps the same-seed bit-identity contracts intact. Needs
// onb(), hence its position here rather than up with the rest of the sky.
float3 sun_sample_dir(float r1, float r2) {
    float cos_t = 1.0 - r1 * (1.0 - sun.w);
    float sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
    float phi = 6.28318530718 * r2;
    float3 st1, st2;
    onb(sun.xyz, st1, st2);
    return st1 * (cos(phi) * sin_t) + st2 * (sin(phi) * sin_t) + sun.xyz * cos_t;
}

// --- Scene textures + UV stream (space1; the RP_SCENE_TEX table) --------------
// Self-contained here (NOT on shade.hlsli's declarations) because the cutout
// test runs inside the trace primitives — hemi_wave.hlsl calls occluded_q
// without pasting shade.hlsli. uv_indices/uv_tri_mat are second descriptors
// over the same indices/tri_mat resources as the t4/t5 space0 root SRVs.

StructuredBuffer<float2> uv_buf     : register(t0, space1); // Scene::texcoords
StructuredBuffer<uint>   uv_indices : register(t1, space1); // flat, 3 per tri
StructuredBuffer<uint>   uv_tri_mat : register(t2, space1); // material per tri
// Per-material cutout map: tex + 1 when textured AND alpha_masked, else 0
// (the bvh.rs::moller_trumbore gate chain folded to one fetch).
StructuredBuffer<uint>   mat_cutout : register(t3, space1);
// R8G8B8A8 (_SRGB for color maps, _UNORM for linear-data normal/rough-metal
// maps — Texture::srgb), carrying the FULL CPU-generated mip chain (built in
// texture.rs::build_mips, uploaded verbatim — CPU-trilinear parity: both
// samplers read identical texels at identical ray-cone lods).
Texture2D<float4>        texs[]     : register(t4, space1);
// Static: trilinear, repeat wrap — texture.rs::sample_trilinear in hardware
// (sRGB decode per texel via the SRGB SRV format, texel centers at i + 0.5;
// every sample passes an explicit ray-cone lod to SampleLevel, and lod <= 0
// reads level 0 only = the old bilinear). The cutout test below deliberately
// uses .Load instead — visibility never touches mips.
SamplerState             samp_lin   : register(s0, space1);
// Static: hardware anisotropic (MaxAnisotropy = the session's --aniso), fed
// the elliptical footprint through SampleGrad — texture.rs::sample_aniso in
// hardware. SampleLevel hands the TMU one scalar lod and no gradients, so an
// aniso filter THERE would be a silent no-op: the gradients are the feature.
SamplerState             samp_aniso : register(s1, space1);

// texture.rs::wrap — repeat into [0,1], non-finite collapses to 0 (fp can
// round c - floor(c) up to exactly 1.0 for c just below an integer; the
// consumer's min(w-1) clamp absorbs it, same as the CPU).
float uv_wrap(float c) {
    float f = c - floor(c);
    return isfinite(f) ? f : 0.0;
}

// scene.rs::tri_uv — barycentric UV interpolation at a hit.
float2 tri_uv(uint tri, float u, float v) {
    uint i0 = uv_indices[tri * 3u];
    uint i1 = uv_indices[tri * 3u + 1u];
    uint i2 = uv_indices[tri * 3u + 2u];
    return uv_buf[i0] * (1.0 - u - v) + uv_buf[i1] * u + uv_buf[i2] * v;
}

// bvh.rs::moller_trumbore's alpha-cutout test: true = reject the candidate.
// texture.rs::alpha_nearest mirrored in ALU over a .Load (not a sampler), so
// the RayQuery candidate loop and the DXR any-hit agree bit-for-bit with each
// other; `.a` of an SRGB SRV is linear per D3D spec, and alpha*255 < 127.5
// reproduces the CPU's `u8 < 128` through the UNORM n/255 representation.
bool alpha_cutout(uint tri, float u, float v) {
    uint cm = mat_cutout[uv_tri_mat[tri]];
    if (cm == 0u) return false;
    float2 uv = tri_uv(tri, u, v);
    uint ti = NonUniformResourceIndex(cm - 1u);
    uint w, h;
    texs[ti].GetDimensions(w, h);
    uint x = min(uint(uv_wrap(uv.x) * float(w)), w - 1u);
    uint y = min(uint(uv_wrap(uv.y) * float(h)), h - 1u);
    return texs[ti].Load(int3(int(x), int(y), 0)).a * 255.0 < 127.5;
}
