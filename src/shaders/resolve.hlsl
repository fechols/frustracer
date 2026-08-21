// accum -> RGBA16F HDR texture at 1/samples. The divide happens HERE, before
// the f16 texture, so long accumulations can't overflow half range; the
// tonemap PS then runs with inv_samples pinned to 1.0. Requires
// trace_common.hlsli pasted first.

RWStructuredBuffer<float> accum : register(u0);
#ifdef WEB
// WebGPU's bind-group layout entry NAMES the storage texture's format and
// requires it to equal the texture's own. HLSL has no format syntax for a
// UAV, so DXC's SPIR-V picks `rgba32f` for an unannotated `float4` — which
// is not what this image is, and a browser refuses the mismatch rather than
// reinterpreting it. `[[vk::image_format]]` is the one place the shader can
// say what the comment below has always said.
//
// SPIR-V-ONLY AND WEB-ONLY. DXIL has no such concept, and the native SPIR-V
// corpus (`--check-spirv`, `--check-vk`) is deliberately left byte-identical
// — Vulkan's layout carries no format, so the annotation would buy it
// nothing and would move recorded numbers for stages that never wanted it.
[[vk::image_format("rgba16f")]]
#endif
RWTexture2D<float4> hdr : register(u14); // typed UAV -> descriptor table (base = NUM_UAVS)

cbuffer Push : register(b1) { float inv_samples; uint _pp0; uint _pp1; uint _pp2; }

[numthreads(8, 8, 1)]
void cs_resolve(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint i3 = (id.y * rw + id.x) * 3u;
    float3 c = float3(accum[i3], accum[i3 + 1u], accum[i3 + 2u]) * inv_samples;
    hdr[id.xy] = float4(c, 1.0);
}
