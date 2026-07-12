// FSR-mode remodulation: out = dd*decode(diff_alb) + ds*decode(spec_alb)
// + residual over the rw x rh dynamic sub-rect (planes are allocated at the
// range max). With a pass-through denoiser (dd_out == dd_in) this reproduces
// the traced frame — the GPU twin of fsr::composite, gated on the CPU by
// --check-fsr; the albedo decode must mirror fsr.rs's sqrt wire encoding.

Texture2D<float4> dd_tex : register(t0);   // denoised direct diffuse
Texture2D<float4> ds_tex : register(t1);   // denoised direct specular
Texture2D<float4> da_tex : register(t2);   // sqrt-encoded diffuse albedo (RGBA8)
Texture2D<float4> sa_tex : register(t3);   // sqrt-encoded specular albedo (RGBA8)
Texture2D<float4> res_tex : register(t4);  // pass-through residual
RWTexture2D<float4> outp : register(u0);

cbuffer C : register(b0) { uint rw; uint rh; }

// Inverse of fsr::sqrt_encode8 (the hardware UNORM read gives s = byte/255).
float3 decode_albedo(float3 s) { return s * s; }

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint2 p = id.xy;
    float3 c = dd_tex[p].rgb * decode_albedo(da_tex[p].rgb)
             + ds_tex[p].rgb * decode_albedo(sa_tex[p].rgb)
             + res_tex[p].rgb;
    outp[p] = float4(c, 1.0);
}
