// FR_WAVEVIZ overlay composite (--waveviz): the wave-ticket hash, blended
// over the PRESENTED image as a second fullscreen draw inside
// `fullscreen_to_backbuffer` — the HUD's exact shape (separate PS + PSO on
// the tonemap root signature, premultiplied ONE / INV_SRC_ALPHA blend), which
// is what makes the overlay work under EVERY present arm: plain, DLSS-RR,
// XeSS, FSR, quinlight. The tickets live in the tracer's render-res tbuf
// (asfloat bits — the covered kernels' LAST tbuf touch while the overlay is
// live), read here as a ROOT SRV (t2, bound by GPU VA — no descriptor) and
// nearest-mapped from window to render res (identity at the native lock).
//
// The cbuffer REUSES register b0 through the shared root signature but with
// THIS pass's own layout — root constants are per-draw state, so the
// tonemap/hud draws are untouched (their 8 DWORDs arrive in their layout,
// ours in this one; record_waveviz writes it).
StructuredBuffer<float> tickets : register(t2);

cbuffer Params : register(b0) {
    uint rw;      // render res (the tickets' pitch)
    uint rh;
    uint ww;      // window res (the RTV this draw covers)
    uint wh;
    float scale;  // ToneParams.scale — paper_white/10000 under PQ
    float mode;   // 1 = gamma 2.2 (SDR/Sdr10), 2 = HDR10 PQ (hud.hlsl's split)
    float _wvp0;
    float _wvp1;
}

// tone.rs twins (see tonemap.hlsl / hud.hlsl — same literals, change all
// together).
float3 m709_to_2020(float3 v) {
    return float3(
        0.627404 * v.r + 0.329283 * v.g + 0.043313 * v.b,
        0.069097 * v.r + 0.919540 * v.g + 0.011362 * v.b,
        0.016391 * v.r + 0.088013 * v.g + 0.895595 * v.b);
}

float3 pq_encode(float3 y) {
    float3 yp = pow(saturate(y), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * yp) / (1.0 + 18.6875 * yp), 78.84375);
}

// resolve.hlsl's retired colorizer, verbatim (main.rs's waveviz_dump mirrors
// it term for term — the live overlay and the headless PNG must agree
// color-for-color).
float3 wv_hash_color(uint t) {
    uint h = t * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    float3 c = float3(float(h & 1023u), float((h >> 10u) & 1023u),
                      float((h >> 20u) & 1023u)) * (1.0 / 1023.0);
    return 0.25 + 0.75 * c;
}

float4 vsmain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 psmain(float4 pos : SV_Position) : SV_Target {
    // Nearest window->render mapping (integer, exact; identity when rw == ww).
    uint2 rp = uint2(uint(pos.x) * rw / ww, uint(pos.y) * rh / wh);
    rp = min(rp, uint2(rw - 1u, rh - 1u));
    uint t = asuint(tickets[rp.y * rw + rp.x]);
    // 0xFFFFFFFE = the WAVEVIZ_CHS miss sentinel (no hit shader ran) —
    // darken instead of hashing, so miss regions read as "no data".
    float3 c;
    float a;
    if (t == 0xFFFFFFFEu) {
        c = 0.0;
        a = 0.85;
    } else {
        c = wv_hash_color(t); // display-space [0.25, 1] — the O-overlay rule
        a = 0.65;
    }
    if (mode > 1.5) {
        // HDR10: decode the display-space color into paper-white-relative
        // light, then the PQ chain (hud.hlsl's arm, same compromise: blend
        // premultiplied values at the backbuffer's own encoding).
        c = pq_encode(m709_to_2020(pow(max(c, 0.0), 2.2) * scale));
    }
    // mode == 1 (SDR/Sdr10): the backbuffer is display space — as authored.
    return float4(c * a, a);
}
