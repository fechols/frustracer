// Glare (bloom) on the GPU — the term-for-term mirror of src/bloom.rs. Read that
// file's header first: this is a DISPLAY-stage pass (it runs on the tonemap's
// source image, never on accum), and its composite is energy-conserving, so a
// uniformly lit frame comes back unchanged.
//
// Two kernels:
//   cs_down  a 2x2 box downsample (one bilinear tap at the 2x2 center IS the
//            box average, so the hardware filter does the work for free).
//   cs_up    tent-upsample the coarser level and add it into this one, weighted.
//            The tent is not cosmetic: a 2x2 box has a SQUARE footprint, and
//            reconstructing it with a single bilinear tap leaves the glare core
//            looking like a rounded rectangle instead of a round halo — very
//            visible on the sun, which is the one thing this pass exists for.
//
// The final tent-sample back to full res, and the composite, live in the tonemap
// PS (tonemap.hlsl) — one fewer pass and one fewer full-res texture.

// The WIDE downsample arm needs BOTH halves to fire: this compile define and
// the FLAG_WIDE bit in `flags`. A compile-only lever would fire inside gates
// that pinned the feature off; a runtime-only one would leave the shipped box
// arm carrying the wide path's registers. `bloom::kernel_defs` emits the define,
// `gpu::bloom::record` sets the bit, and the #ifndef below is the fallback that
// keeps this file assembling standalone under --check-spirv / --check-msl, which
// compile the corpus with no injector in front of it.
#ifndef BLOOM_ALLOW_WIDE
#define BLOOM_ALLOW_WIDE 0
#endif

#define BLOOM_FLAG_WIDE 1u

Texture2D<float4>   src : register(t0);
RWTexture2D<float4> dst : register(u0);
SamplerState        samp_lin : register(s0); // linear, clamp

cbuffer BloomParams : register(b0) {
    uint2  dst_size;
    uint2  src_size;
    float  weight;   // this level's octave weight (cs_up only)
    uint   flags;    // BLOOM_FLAG_* — the runtime half of each compile arm
    float2 _pad;     // offset 24, ends at 32: does NOT straddle the 16B row
};

[numthreads(8, 8, 1)]
void cs_down(uint3 id : SV_DispatchThreadID) {
    if (id.x >= dst_size.x || id.y >= dst_size.y) return;
    // The center of the 2x2 source footprint, in source UV. One linear tap there
    // averages exactly those four texels — the box filter, for free.
    float2 uv = (float2(id.xy) + 0.5) / float2(dst_size);
#if BLOOM_ALLOW_WIDE
    if (flags & BLOOM_FLAG_WIDE) {
        // Five OVERLAPPING 2x2 boxes (Jimenez/COD): the centre box at 0.5, plus
        // the same box shifted one SOURCE texel diagonally, at 0.125 each. Five
        // bilinear taps, because each box is still one hardware fetch — and the
        // weights sum to 1, so energy conservation is untouched.
        //
        // The overlap is the whole point. The box arm's support is EXACTLY one
        // destination texel, so a source pixel contributes 100% to one mip texel
        // and 0% to its neighbour; a light translating across that boundary makes
        // the halo jump rather than slide, which is what reads as flicker. Here
        // adjacent destination texels share half their footprint, so the energy
        // hands over continuously. Measured with --bloom-lab: on a sun-bright
        // disc at 4 px/frame the worst pixel's per-frame swing falls 41.3 -> 1.7
        // /255 and the halo's centroid slide 6.86 -> 0.06 px.
        //
        // This is the ONE path that reads src_size — see the note in
        // gpu::bloom::record about why the box arm is still passed zeros.
        float2 t = 1.0 / float2(src_size);
        float3 s = src.SampleLevel(samp_lin, uv, 0).rgb * 0.5;
        s += src.SampleLevel(samp_lin, uv + float2(-t.x, -t.y), 0).rgb * 0.125;
        s += src.SampleLevel(samp_lin, uv + float2( t.x, -t.y), 0).rgb * 0.125;
        s += src.SampleLevel(samp_lin, uv + float2(-t.x,  t.y), 0).rgb * 0.125;
        s += src.SampleLevel(samp_lin, uv + float2( t.x,  t.y), 0).rgb * 0.125;
        dst[id.xy] = float4(s * weight, 1.0);
        return;
    }
#endif
    // `weight` is 1.0 for every level except the COARSEST, which is pre-scaled by
    // its octave weight here. bloom.rs pre-scales the last level the same way,
    // and that is what seeds the upsample recurrence
    //     acc_i = w_i * L_i + tent(acc_{i+1}),  acc_last = w_last * L_last.
    // Folding it here avoids a third kernel.
    dst[id.xy] = float4(src.SampleLevel(samp_lin, uv, 0).rgb * weight, 1.0);
}

// 3x3 tent (1,2,1 / 2,4,2 / 1,2,1 over 16) at one SOURCE-texel spacing.
// Weights sum to 1, so energy conservation is untouched.
float3 tent(float2 uv, float2 texel) {
    float3 s = 0.0;
    s += src.SampleLevel(samp_lin, uv + float2(-texel.x, -texel.y), 0).rgb * 1.0;
    s += src.SampleLevel(samp_lin, uv + float2( 0.0,     -texel.y), 0).rgb * 2.0;
    s += src.SampleLevel(samp_lin, uv + float2( texel.x, -texel.y), 0).rgb * 1.0;
    s += src.SampleLevel(samp_lin, uv + float2(-texel.x,  0.0),     0).rgb * 2.0;
    s += src.SampleLevel(samp_lin, uv,                              0).rgb * 4.0;
    s += src.SampleLevel(samp_lin, uv + float2( texel.x,  0.0),     0).rgb * 2.0;
    s += src.SampleLevel(samp_lin, uv + float2(-texel.x,  texel.y), 0).rgb * 1.0;
    s += src.SampleLevel(samp_lin, uv + float2( 0.0,      texel.y), 0).rgb * 2.0;
    s += src.SampleLevel(samp_lin, uv + float2( texel.x,  texel.y), 0).rgb * 1.0;
    return s * (1.0 / 16.0);
}

[numthreads(8, 8, 1)]
void cs_up(uint3 id : SV_DispatchThreadID) {
    if (id.x >= dst_size.x || id.y >= dst_size.y) return;
    float2 uv = (float2(id.xy) + 0.5) / float2(dst_size);
    float2 texel = 1.0 / float2(src_size);
    // Read-modify-write: this level already holds its own downsampled image, and
    // we fold the (wider) level above it in. bloom.rs does the same, in the same
    // order: dst = dst * w + tent(coarser).
    float3 cur = dst[id.xy].rgb;
    dst[id.xy] = float4(cur * weight + tent(uv, texel), 1.0);
}
