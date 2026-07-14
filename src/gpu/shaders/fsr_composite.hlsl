// FSR-mode remodulation: out = dd*decode(diff_alb) + ds*decode(spec_alb)
// + ao*amb(n)*decode(diff_alb) + is*decode(spec_alb) + residual, over the
// rw x rh dynamic sub-rect (planes are allocated at the range max). With a
// pass-through denoiser (dd_out == dd_in, ...) this reproduces the traced
// frame — the GPU twin of fsr::composite, gated on the CPU by --check-fsr and
// on the GPU by the `fsr composite identity` gate in --check-gpu/--check-dxr.
// The albedo decode must mirror fsr.rs's sqrt wire encoding.
//
// `amb` — the factor the denoised AO open fraction is remodulated by — used to
// be the flat shade::AMBIENT constant riding in as a root constant. The one sky
// (sky.rs) makes it DIRECTIONAL: the sky's own order-2 SH irradiance at the
// pixel's normal. So this pass now reads the normals plane and evaluates the
// same sh_irr() the tracer does (sh.hlsli, prepended by ffx_rr.rs — this compile
// unit has no shade.hlsli prelude).
//
// It must decode that normal from the PLANE BYTES, which is the whole reason
// fsr::wire_normal exists: this pass has no other source for it, so the split
// site subtracted a factor built from the same 10-bit oct normal. Reading the
// full-precision shading normal on one side and the quantized one here would
// leave a quantization-sized hole in the composite identity.

Texture2D<float4> dd_tex : register(t0);   // denoised direct diffuse
Texture2D<float4> ds_tex : register(t1);   // denoised direct specular
Texture2D<float4> da_tex : register(t2);   // sqrt-encoded diffuse albedo (RGBA8)
Texture2D<float4> sa_tex : register(t3);   // sqrt-encoded specular albedo (RGBA8)
Texture2D<float4> res_tex : register(t4);  // pass-through residual
Texture2D<float> ao_tex : register(t5);    // denoised ambient occlusion (R16F)
Texture2D<float4> is_tex : register(t6);   // denoised indirect specular (A = hit t)
Texture2D<float4> nm_tex : register(t7);   // RG = oct normal (RGB10A2), B = rough
RWTexture2D<float4> outp : register(u0);

// Field ORDER is load-bearing: HLSL never lets a float3/float4 array straddle
// awkwardly, and a scalar placed before one would be padded out to a 16-byte
// boundary — silently shifting every row and reading past the constants the
// root signature declares. (That exact bug shipped once, with a `float3 ambient`
// after two uints: DXC bumped it from offset 8 to 16 and the pass read one
// channel shifted plus two undeclared DWORDs.) The float4 array LEADS, so the
// block is 9*4 + 2 = 38 contiguous DWORDs (sky_sh[0..9] | rw | rh), which is
// what ffx_rr.rs::record_composite writes.
cbuffer C : register(b0) { float4 sky_sh[9]; uint rw; uint rh; }

// Inverse of fsr::sqrt_encode8 (the hardware UNORM read gives s = byte/255).
float3 decode_albedo(float3 s) { return s * s; }

// oct_decode comes from fsr_wire.hlsli — the same one feed.hlsl encoded with.

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint2 p = id.xy;
    float3 kd = decode_albedo(da_tex[p].rgb);
    float3 f0 = decode_albedo(sa_tex[p].rgb);
    float3 amb = sh_irr(sky_sh, oct_decode(nm_tex[p].rg));
    float3 c = dd_tex[p].rgb * kd
             + ds_tex[p].rgb * f0
             + ao_tex[p] * amb * kd
             + is_tex[p].rgb * f0
             + res_tex[p].rgb;
    outp[p] = float4(c, 1.0);
}
