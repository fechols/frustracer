// GPU BC7 encoder (src/gpu/bc7gpu.rs is the host; fxc cs_5_0 — the bloom
// no-DXC precedent, so the encode works before any tracer kernel exists).
// One thread = one 4x4 block.
//
// Two modes, chosen per block by measured SSE:
// - MODE 6: single subset, RGBA 7.7.7.7 endpoints + one P-bit per endpoint,
//   4-bit indices — the highest-precision opaque mode, always fitted first.
// - MODE 1: two subsets (64 spec partitions), RGB 6.6.6 endpoints + one
//   shared P-bit per subset, 3-bit indices — what multi-hue blocks need
//   (mode-6-only measured 26.4 dB worst on San Miguel's patterned plate vs
//   the ispc encoder's 33.0; the gap is the second subset, not the fit).
//   Candidate partitions are ranked by a cheap 2-means SSE over all 64,
//   then the top few get the full fit. `effort` 1 (fast) tries mode 1 only
//   when the mode-6 SSE is already noticeable; 2/3 (basic/slow) always try,
//   with a deeper candidate list.
//
// Alpha: mode 6 pins its alpha endpoints to 127/P (recon 254-255); mode 1
// carries no alpha at all (decodes 255) — the shader twin of the ispc
// `opaque_*` / `channels: 3` carve-out, and the error metric is RGB-only
// (nothing ever samples a compressed texture's alpha — see src/bc7.rs).
//
// SOLID blocks: a P-bit is shared across an endpoint's channels, so an
// arbitrary flat color lands within 1 LSB (mixed channel parity cannot be an
// exact endpoint), but a flat color whose channels AGREE in parity encodes
// bit-exactly via e0 == e1 — the property the --check-gpu structural gate
// pins. Algorithmic reference: the standard PCA + least-squares refinement
// shape every fast encoder uses (bc7e/Betsy class); no code copied. The
// weight / partition / anchor tables are BC7 spec data.
//
// OUTPUT LAYOUT IS THE ONE WIRING TRAP: blocks store at
// (by * row_blocks + bx) * 16 where row_blocks is the 256-BYTE-ALIGNED
// d3d12::block_pitch in blocks — NOT tight-packed. footprint_block declares
// that pitch to CopyTextureRegion; a tight-packed store shears every row and
// M11 lands at ~10-20 dB.

ByteAddressBuffer src : register(t0);   // staged RGBA8 texel rows (upload ring)
RWByteAddressBuffer dst : register(u0); // BC7 blocks, block_pitch-strided rows

cbuffer C : register(b0) {
    uint mw;          // mip width in texels
    uint staged_rows; // texel rows staged in this band (clamp = edge replicate)
    uint src_pitch;   // bytes per staged texel row (256-aligned)
    uint bw;          // block columns = ceil(mw/4)
    uint band_rows;   // block rows in this band
    uint row_blocks;  // dst row stride in BLOCKS (= block_pitch/16)
    uint effort;      // 0 ultrafast | 1 fast | 2 basic | 3 slow
    // Byte offsets of THIS dispatch's window within src/dst. D3D12 passes 0
    // for both and slides the root SRV/UAV virtual addresses instead; Vulkan
    // binds one descriptor per buffer for a whole batch of levels and moves
    // the window here, because a per-dispatch descriptor rebind is a
    // descriptor-set allocation per level and a dynamic-offset binding would
    // need the derived layout to special-case one slot. Integer address
    // arithmetic with a 0 offset is bit-identical to not having it.
    uint src_off;
    uint dst_off;
}

// BC7 spec interpolation weights.
static const uint W4[16] = {0,4,9,13,17,21,26,30,34,38,43,47,51,55,60,64};
static const uint W3[8]  = {0,9,18,27,37,46,55,64};

// The 64 two-subset partitions as bitmasks (bit i = texel i's subset) and
// their subset-1 anchor indices — BC7 spec tables.
static const uint PART2[64] = {
    0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80,
    0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000,
    0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
    0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
    0xAAAA, 0xF0F0, 0x5A5A, 0x33CC, 0x3C3C, 0x55AA, 0x9696, 0xA55A,
    0x73CE, 0x13C8, 0x324C, 0x3BDC, 0x6996, 0xC33C, 0x9966, 0x0660,
    0x0272, 0x04E4, 0x4E40, 0x2720, 0xC936, 0x936C, 0x39C6, 0x639C,
    0x9336, 0x9CC6, 0x817E, 0xE718, 0xCCF0, 0x0FCC, 0x7744, 0xEE22
};
static const uint ANCHOR2[64] = {
    15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,
    15, 2, 8, 2, 2, 8, 8,15,
     2, 8, 2, 2, 8, 8, 2, 2,
    15,15, 6, 8, 2, 8,15,15,
     2, 8, 2, 2, 2,15,15, 6,
     6, 2, 6, 8,15,15, 2, 2,
    15,15,15,15,15, 2, 2,15
};

// The mode-1 trigger at effort 1: try the two-subset fit only when mode 6
// left more than ~2.5 LSB RMS over the block's 48 RGB samples.
static const float M1_TRIGGER = 300.0;

// The block being encoded (per-thread).
static float3 PX[16];

// LSB-first 128-bit bitstream over a local uint[4].
void put_bits(inout uint blk[4], inout uint pos, uint v, uint n) {
    uint w = pos >> 5u, b = pos & 31u;
    blk[w] |= v << b;
    if (b + n > 32u && w < 3u)
        blk[w + 1u] |= v >> (32u - b);
    pos += n;
}

uint sub_of(uint mask, uint i) {
    return (mask >> i) & 1u;
}

// ---- subset fitting (mode 6 = the mask-0 whole-block "subset") ----

float3 subset_mean(uint mask, uint s, out uint n) {
    float3 acc = 0;
    n = 0;
    [loop]
    for (uint i = 0; i < 16; i++)
        if (sub_of(mask, i) == s) {
            acc += PX[i];
            n++;
        }
    return acc / (float)max(n, 1u);
}

// Principal axis by covariance power iteration. The seed is the covariance
// COLUMN with the largest diagonal — never a fixed vector: (1,1,1) is
// exactly perpendicular to an anti-correlated axis like (1,-1,0), and a
// red<->green block would collapse to a near-solid encode. A column seed is
// nonzero whenever that channel carries any variance at all.
float3 subset_axis(uint mask, uint s, float3 mu) {
    float cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
    [loop]
    for (uint i = 0; i < 16; i++)
        if (sub_of(mask, i) == s) {
            float3 d = PX[i] - mu;
            cxx += d.x * d.x; cxy += d.x * d.y; cxz += d.x * d.z;
            cyy += d.y * d.y; cyz += d.y * d.z; czz += d.z * d.z;
        }
    float3 axis = float3(cxx, cxy, cxz);
    if (cyy > cxx && cyy >= czz)
        axis = float3(cxy, cyy, cyz);
    else if (czz > cxx && czz > cyy)
        axis = float3(cxz, cyz, czz);
    [unroll]
    for (uint r = 0; r < 4; r++) {
        float3 v = float3(
            cxx * axis.x + cxy * axis.y + cxz * axis.z,
            cxy * axis.x + cyy * axis.y + cyz * axis.z,
            cxz * axis.x + cyz * axis.y + czz * axis.z);
        float len = length(v);
        if (len > 1e-8)
            axis = v / len;
    }
    float alen = length(axis);
    return alen > 1e-8 ? axis / alen : float3(0.57735, 0.57735, 0.57735);
}

void subset_extremes(uint mask, uint s, float3 mu, float3 axis, out float3 A, out float3 B) {
    float tmin = 1e30, tmax = -1e30;
    [loop]
    for (uint i = 0; i < 16; i++)
        if (sub_of(mask, i) == s) {
            float tt = dot(PX[i] - mu, axis);
            tmin = min(tmin, tt);
            tmax = max(tmax, tt);
        }
    if (tmin > tmax) {
        tmin = 0;
        tmax = 0;
    }
    A = clamp(mu + axis * tmin, 0.0, 255.0);
    B = clamp(mu + axis * tmax, 0.0, 255.0);
}

// Best index (nearest interpolant) for p against reconstructed endpoints;
// projection + the two neighbors — the weights are near-uniform.
uint pick_index(float3 p, float3 EA, float3 EB, bool four_bit, out float err) {
    float3 d = EB - EA;
    float len2 = dot(d, d);
    if (len2 < 1e-6) {
        float3 e = p - EA;
        err = dot(e, e);
        return 0;
    }
    int hi = four_bit ? 15 : 7;
    float tt = saturate(dot(p - EA, d) / len2);
    int k0 = (int)round(tt * (float)hi);
    uint bk = 0;
    float be = 1e30;
    [unroll]
    for (int n = -1; n <= 1; n++) {
        int k = clamp(k0 + n, 0, hi);
        uint w = four_bit ? W4[k] : W3[k];
        float3 rc = EA + d * ((float)w / 64.0);
        float3 e = p - rc;
        float er = dot(e, e);
        if (er < be) {
            be = er;
            bk = (uint)k;
        }
    }
    err = be;
    return bk;
}

// Least-squares endpoint refit of one subset from its chosen indices.
void ls_refit(uint mask, uint s, uint idx[16], bool four_bit, inout float3 A, inout float3 B) {
    float saa = 0, sab = 0, sbb = 0;
    float3 sap = 0, sbp = 0;
    [loop]
    for (uint i = 0; i < 16; i++)
        if (sub_of(mask, i) == s) {
            uint w = four_bit ? W4[idx[i]] : W3[idx[i]];
            float b = (float)w / 64.0;
            float a = 1.0 - b;
            saa += a * a;
            sab += a * b;
            sbb += b * b;
            sap += a * PX[i];
            sbp += b * PX[i];
        }
    float det = saa * sbb - sab * sab;
    if (abs(det) > 1e-4) {
        A = clamp((sap * sbb - sbp * sab) / det, 0.0, 255.0);
        B = clamp((sbp * saa - sap * sab) / det, 0.0, 255.0);
    }
}

// ---- mode-6 endpoint quantization: 7 bits + per-endpoint P, recon 2q+p ----

uint quant7(float c, uint p) {
    int q = (int)round((c - (float)p) * 0.5);
    return (uint)clamp(q, 0, 127);
}

// ---- mode-1 endpoint quantization: 6 bits + shared-subset P ----
// recon e7 = (q<<1)|p, e8 = (e7<<1)|(e7>>6) = 4q + 2p + (q>=32).

uint dequant6(uint q, uint p) {
    uint e7 = (q << 1u) | p;
    return (e7 << 1u) | (e7 >> 6u);
}

uint quant6(float c, uint p) {
    int q0 = (int)round((c - 2.0 * (float)p) * 0.25);
    uint bq = 0;
    float be = 1e30;
    [unroll]
    for (int n = -1; n <= 1; n++) {
        uint q = (uint)clamp(q0 + n, 0, 63);
        float e = abs(c - (float)dequant6(q, p));
        if (e < be) {
            be = e;
            bq = q;
        }
    }
    return bq;
}

// Full mode-1 fit of one partition: per subset PCA -> quantize at the best
// shared P -> indices -> one LS refit -> final quantize + indices. Returns
// the block SSE; eq[e] = endpoint e's quantized RGB (e = s0e0,s0e1,s1e0,s1e1).
float fit_mode1(uint part, out uint3 eq[4], out uint2 pb, out uint idx[16]) {
    uint mask = PART2[part];
    float sse = 0;
    // Init (fxc wants every out lane written on all paths).
    [unroll]
    for (uint z = 0; z < 16; z++)
        idx[z] = 0;
    pb = uint2(0, 0);
    [loop]
    for (uint s = 0; s < 2; s++) {
        uint n;
        float3 mu = subset_mean(mask, s, n);
        float3 axis = subset_axis(mask, s, mu);
        float3 A, B;
        subset_extremes(mask, s, mu, axis, A, B);
        uint p = 0, qa3[3], qb3[3];
        float3 EA = A, EB = B;
        [loop]
        for (uint r = 0; r < 2; r++) {
            // Shared-subset P: pick by endpoint reconstruction error.
            float e0 = 0, e1 = 0;
            [unroll]
            for (uint ch = 0; ch < 3; ch++) {
                float da0 = A[ch] - (float)dequant6(quant6(A[ch], 0), 0);
                float db0 = B[ch] - (float)dequant6(quant6(B[ch], 0), 0);
                float da1 = A[ch] - (float)dequant6(quant6(A[ch], 1), 1);
                float db1 = B[ch] - (float)dequant6(quant6(B[ch], 1), 1);
                e0 += da0 * da0 + db0 * db0;
                e1 += da1 * da1 + db1 * db1;
            }
            p = e1 < e0 ? 1u : 0u;
            [unroll]
            for (uint ch = 0; ch < 3; ch++) {
                qa3[ch] = quant6(A[ch], p);
                qb3[ch] = quant6(B[ch], p);
                EA[ch] = (float)dequant6(qa3[ch], p);
                EB[ch] = (float)dequant6(qb3[ch], p);
            }
            // Indices for this subset's texels.
            [loop]
            for (uint i = 0; i < 16; i++)
                if (sub_of(mask, i) == s) {
                    float err;
                    idx[i] = pick_index(PX[i], EA, EB, false, err);
                    if (r == 1)
                        sse += err;
                }
            if (r == 0)
                ls_refit(mask, s, idx, false, A, B);
        }
        if (s == 0) {
            eq[0] = uint3(qa3[0], qa3[1], qa3[2]);
            eq[1] = uint3(qb3[0], qb3[1], qb3[2]);
            pb.x = p;
        } else {
            eq[2] = uint3(qa3[0], qa3[1], qa3[2]);
            eq[3] = uint3(qb3[0], qb3[1], qb3[2]);
            pb.y = p;
        }
    }
    return sse;
}

[numthreads(8, 8, 1)]
void cs_bc7_encode(uint3 id : SV_DispatchThreadID) {
    uint bx = id.x, by = id.y;
    if (bx >= bw || by >= band_rows)
        return;

    // 16 texels, edge-replicate clamped (the CPU encode_level's pad
    // semantics: right edge clamps to mw-1, bottom to the last staged row —
    // the band containing a mip's bottom edge always stages it).
    uint3 ipx[16];
    bool solid = true;
    [unroll]
    for (uint t = 0; t < 16; t++) {
        uint sx = min(bx * 4u + (t & 3u), mw - 1u);
        uint sy = min(by * 4u + (t >> 2u), staged_rows - 1u);
        uint p = src.Load(src_off + sy * src_pitch + sx * 4u);
        uint3 c = uint3(p & 255u, (p >> 8u) & 255u, (p >> 16u) & 255u);
        ipx[t] = c;
        PX[t] = (float3)c;
        solid = solid && all(c == ipx[0]);
    }

    uint qa[4], qb[4]; // mode-6 7-bit endpoint channels, RGBA
    uint pa = 0, pb6 = 0;
    uint idx[16];
    float m6_sse = 0;

    if (solid) {
        // e0 == e1 at the majority channel parity: exact when all channels
        // agree in parity, else off by 1 LSB on the minority channels — the
        // best a shared P-bit allows for equal endpoints.
        uint3 c = ipx[0];
        uint odd = (c.r & 1u) + (c.g & 1u) + (c.b & 1u);
        pa = odd >= 2u ? 1u : 0u;
        pb6 = pa;
        [unroll]
        for (uint ch = 0; ch < 3; ch++) {
            uint q = quant7((float)c[ch], pa);
            qa[ch] = q;
            qb[ch] = q;
        }
        qa[3] = 127u;
        qb[3] = 127u;
        [unroll]
        for (uint i = 0; i < 16; i++)
            idx[i] = 0u;
    } else {
        // --- mode-6 fit: PCA extremes + `effort` refinement rounds ---
        uint n16;
        float3 mu = subset_mean(0u, 0u, n16);
        float3 axis = subset_axis(0u, 0u, mu);
        float3 A, B;
        subset_extremes(0u, 0u, mu, axis, A, B);
        uint rounds = effort == 0u ? 0u : 2u;
        float3 EA = A, EB = B;
        [loop]
        for (uint r = 0; r <= rounds; r++) {
            // Quantize A/B at the best (pa, pb) combo by endpoint recon err.
            float best = 1e30;
            [unroll]
            for (uint pc = 0; pc < 4; pc++) {
                uint p0 = pc & 1u, p1 = pc >> 1u;
                float err = 0;
                [unroll]
                for (uint ch = 0; ch < 3; ch++) {
                    float ra = (float)(quant7(A[ch], p0) * 2u + p0);
                    float rb = (float)(quant7(B[ch], p1) * 2u + p1);
                    err += (A[ch] - ra) * (A[ch] - ra) + (B[ch] - rb) * (B[ch] - rb);
                }
                if (err < best) {
                    best = err;
                    pa = p0;
                    pb6 = p1;
                }
            }
            [unroll]
            for (uint ch = 0; ch < 3; ch++) {
                qa[ch] = quant7(A[ch], pa);
                qb[ch] = quant7(B[ch], pb6);
                EA[ch] = (float)(qa[ch] * 2u + pa);
                EB[ch] = (float)(qb[ch] * 2u + pb6);
            }
            qa[3] = 127u;
            qb[3] = 127u;
            m6_sse = 0;
            [loop]
            for (uint i = 0; i < 16; i++) {
                float err;
                idx[i] = pick_index(PX[i], EA, EB, true, err);
                m6_sse += err;
            }
            if (r == rounds)
                break;
            ls_refit(0u, 0u, idx, true, A, B);
        }
    }

    // --- mode-1 (two subsets): rank the 64 partitions by 2-means SSE, full
    // fit the top candidates, keep the better mode. effort 1 = only when
    // mode 6 left visible error; effort 2/3 = always, deeper list.
    bool use_m1 = false;
    uint m1_part = 0;
    uint3 m1_eq[4];
    uint2 m1_pb = uint2(0, 0);
    uint m1_idx[16];
    [unroll]
    for (uint z = 0; z < 16; z++)
        m1_idx[z] = 0;
    bool try_m1 =
        !solid && effort >= 1u && (effort >= 2u || m6_sse > M1_TRIGGER);
    if (try_m1) {
        // 2-means SSE per partition via SSE = total - |sum_s|^2/n_s.
        float total = 0;
        [unroll]
        for (uint i = 0; i < 16; i++)
            total += dot(PX[i], PX[i]);
        float psse[64];
        [loop]
        for (uint part = 0; part < 64; part++) {
            uint mask = PART2[part];
            float3 s0 = 0, s1 = 0;
            uint n1 = 0;
            [loop]
            for (uint i = 0; i < 16; i++)
                if (sub_of(mask, i) == 1u) {
                    s1 += PX[i];
                    n1++;
                } else {
                    s0 += PX[i];
                }
            uint n0 = 16u - n1;
            psse[part] =
                total - dot(s0, s0) / (float)max(n0, 1u) - dot(s1, s1) / (float)max(n1, 1u);
        }
        uint cands = effort >= 3u ? 16u : (effort == 2u ? 8u : 4u);
        float best_sse = m6_sse;
        [loop]
        for (uint k = 0; k < cands; k++) {
            // Extract the next-best partition.
            uint part = 0;
            float pm = 1e30;
            [loop]
            for (uint pp = 0; pp < 64; pp++)
                if (psse[pp] < pm) {
                    pm = psse[pp];
                    part = pp;
                }
            psse[part] = 1e30;
            // NOTE: pm (the 2-means SSE) is an UPPER-bound predictor of the
            // line fit — a subset's best line only removes error vs its mean
            // — so it ranks candidates but must never prune them.
            uint3 eq[4];
            uint2 pbk;
            uint idxk[16];
            float sse = fit_mode1(part, eq, pbk, idxk);
            if (sse < best_sse) {
                best_sse = sse;
                use_m1 = true;
                m1_part = part;
                [unroll]
                for (uint e = 0; e < 4; e++)
                    m1_eq[e] = eq[e];
                m1_pb = pbk;
                [unroll]
                for (uint i = 0; i < 16; i++)
                    m1_idx[i] = idxk[i];
            }
        }
    }

    uint blk[4] = {0, 0, 0, 0};
    uint pos = 0;
    if (use_m1) {
        // Anchor rule per subset: texel 0 anchors subset 0, ANCHOR2[part]
        // anchors subset 1; an anchor whose index MSB is set swaps that
        // subset's endpoints (shared P survives) and inverts its indices.
        uint mask = PART2[m1_part];
        uint anchor = ANCHOR2[m1_part];
        if (m1_idx[0] & 4u) {
            uint3 tq = m1_eq[0];
            m1_eq[0] = m1_eq[1];
            m1_eq[1] = tq;
            [unroll]
            for (uint i = 0; i < 16; i++)
                if (sub_of(mask, i) == 0u)
                    m1_idx[i] = 7u - m1_idx[i];
        }
        if (m1_idx[anchor] & 4u) {
            uint3 tq = m1_eq[2];
            m1_eq[2] = m1_eq[3];
            m1_eq[3] = tq;
            [unroll]
            for (uint i = 0; i < 16; i++)
                if (sub_of(mask, i) == 1u)
                    m1_idx[i] = 7u - m1_idx[i];
        }
        // Pack (LSB-first): mode 1 = one 0 then a 1 (value 2 in 2 bits),
        // partition (6), R0..R3 G0..G3 B0..B3 (6 each), P0, P1, indices —
        // anchors get 2 bits, the rest 3. Total 128.
        put_bits(blk, pos, 2u, 2u);
        put_bits(blk, pos, m1_part, 6u);
        [unroll]
        for (uint ch = 0; ch < 3; ch++) {
            [unroll]
            for (uint e = 0; e < 4; e++)
                put_bits(blk, pos, m1_eq[e][ch], 6u);
        }
        put_bits(blk, pos, m1_pb.x, 1u);
        put_bits(blk, pos, m1_pb.y, 1u);
        [unroll]
        for (uint i = 0; i < 16; i++)
            put_bits(blk, pos, m1_idx[i], (i == 0u || i == anchor) ? 2u : 3u);
    } else {
        // Anchor rule: index 0's MSB must be 0; swap endpoints + invert.
        if (idx[0] & 8u) {
            [unroll]
            for (uint ch = 0; ch < 4; ch++) {
                uint tq = qa[ch];
                qa[ch] = qb[ch];
                qb[ch] = tq;
            }
            uint tp = pa;
            pa = pb6;
            pb6 = tp;
            [unroll]
            for (uint i = 0; i < 16; i++)
                idx[i] = 15u - idx[i];
        }
        // Pack (LSB-first): mode 6 = six 0s then a 1 (value 64 in 7 bits),
        // then R0 R1 G0 G1 B0 B1 A0 A1 (7 each), P0, P1, then indices —
        // texel 0 gets 3 bits (anchor MSB dropped), texels 1..15 get 4.
        put_bits(blk, pos, 64u, 7u);
        [unroll]
        for (uint ch = 0; ch < 4; ch++) {
            put_bits(blk, pos, qa[ch], 7u);
            put_bits(blk, pos, qb[ch], 7u);
        }
        put_bits(blk, pos, pa, 1u);
        put_bits(blk, pos, pb6, 1u);
        put_bits(blk, pos, idx[0], 3u);
        [unroll]
        for (uint i = 1; i < 16; i++)
            put_bits(blk, pos, idx[i], 4u);
    }

    dst.Store4(dst_off + (by * row_blocks + bx) * 16u, uint4(blk[0], blk[1], blk[2], blk[3]));
}
