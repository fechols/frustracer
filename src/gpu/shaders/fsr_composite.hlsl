// FSR-mode remodulation: out = dd*decode(diff_alb) + ds*decode(spec_alb)
// + ao*AMBIENT*decode(diff_alb) + is*decode(spec_alb) + residual, over the
// rw x rh dynamic sub-rect (planes are allocated at the range max). With a
// pass-through denoiser (dd_out == dd_in, ...) this reproduces the traced
// frame — the GPU twin of fsr::composite, gated on the CPU by --check-fsr;
// the albedo decode must mirror fsr.rs's sqrt wire encoding, and `ambient`
// must be the same constant the split subtracted (shade::AMBIENT, riding in
// as a root constant — this compile unit has no shade.hlsli prelude).

Texture2D<float4> dd_tex : register(t0);   // denoised direct diffuse
Texture2D<float4> ds_tex : register(t1);   // denoised direct specular
Texture2D<float4> da_tex : register(t2);   // sqrt-encoded diffuse albedo (RGBA8)
Texture2D<float4> sa_tex : register(t3);   // sqrt-encoded specular albedo (RGBA8)
Texture2D<float4> res_tex : register(t4);  // pass-through residual
Texture2D<float> ao_tex : register(t5);    // denoised ambient occlusion (R16F)
Texture2D<float4> is_tex : register(t6);   // denoised indirect specular (A = hit t)
RWTexture2D<float4> outp : register(u0);

// Field ORDER is load-bearing: HLSL never lets a float3 straddle a 16-byte
// boundary, so `float3 ambient` after two uints would be bumped from offset 8
// to offset 16 — i.e. root-constant DWORDs 4..6, not 2..4, silently reading
// past the 5 the root signature declares. Leading the float3 packs the two
// scalars into its trailing slot and the block is exactly 5 contiguous DWORDs
// (ambient.xyz | rw | rh), which is what ffx_rr.rs::record_composite writes.
cbuffer C : register(b0) { float3 ambient; uint rw; uint rh; }

// Inverse of fsr::sqrt_encode8 (the hardware UNORM read gives s = byte/255).
float3 decode_albedo(float3 s) { return s * s; }

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint2 p = id.xy;
    float3 kd = decode_albedo(da_tex[p].rgb);
    float3 f0 = decode_albedo(sa_tex[p].rgb);
    float3 c = dd_tex[p].rgb * kd
             + ds_tex[p].rgb * f0
             + ao_tex[p] * ambient * kd
             + is_tex[p].rgb * f0
             + res_tex[p].rgb;
    outp[p] = float4(c, 1.0);
}
