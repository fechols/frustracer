// C3 infrastructure probe: the argument-buffer shapes `mtl::bind` has to
// derive, in the smallest subject that still contains all of them — a texture,
// two samplers, a SECOND descriptor set, and an unbounded texture array.
//
// Why a probe and not a real unit. No corpus unit references `space1` without
// also carrying a ray intersector: `texs[]` lives in `trace_common.hlsli` and
// every consumer of it is a tracer kernel, so a real unit cannot READ the
// second set without geometry, a BVH and a scene upload — none of which exist
// on the Metal side yet. C3 without a dispatch would derive a map and never
// execute it, which is exactly the "K3 without K4" gap `run_check_mtl` warns
// about. `waveprobe.hlsl` is the precedent: an infrastructure probe from one
// backend's milestone, living in the SHARED corpus, adopted by a second
// backend later. Living here rather than as a Metal-only Rust `&str` costs
// Windows nothing (`corpus_units` is where both gates read it) and buys
// `--check-spirv` + `spirv-val` on Linux CI for free.
//
// THREE LOAD-BEARING PROPERTIES. Each is a decision, not an accident, and the
// gate asserts each of them — change one and the stage it feeds goes vacuous.
//
// 1. DECLARATION ORDER DELIBERATELY DISAGREES WITH BINDING ORDER. spirv-cross
//    assigns argument-buffer `[[id(n)]]` densely in the module's OpVariable
//    (declaration) order, NOT in binding order — measured on FFX, recorded in
//    `mtl::bind::cross_check`. C2's pin said "binding order" and passed only
//    because `smoke.hlsl` declares b1,u0,u1,u2, where the two orders coincide.
//    So this file declares t, s, s, u — bindings 1000, 3000, 3001, 2000 after
//    the `-fvk-*-shift` mapping (spirv.rs) — and the two orders CANNOT
//    coincide. K6 asserts the disagreement itself, so a well-meaning tidy-up
//    that sorted these declarations would fail loudly rather than quietly
//    turning the corrected pin back into the accident it was.
//    Confirmed on the emitted MSL: `cs_tex` gets src[[id(0)]],
//    samp_clamp[[id(1)]], samp_repeat[[id(2)]], outbuf[[id(3)]] — binding
//    order would have been 0, 3, 1, 2. The ordinal that governs is the
//    SPIR-V OpVariable stream (`vk::reflect::Desc::decl`), not this file's
//    text; the two agree here, and DO NOT assume they always will — DXC picks
//    the stream order, and it has been seen to differ from source order.
//
// 2. EVERY OBSERVABLE IS AN EXACT INTEGER. The textures are RGBA32Float
//    carrying `0xC30000 | i` in .x — all below 2^24, so exactly representable
//    in fp32 and exactly recoverable through a uint cast. The two space0
//    samplers differ ONLY in address mode (clamp vs repeat, both NEAREST), so
//    one SampleLevel coordinate outside [0,1] resolves to a different TEXEL
//    with no filter weights involved: no tolerance, no undefined tie-break.
//    A readback mismatch is a binding bug, never a precision one.
//
// 3. THE space1 BLOCK IS BYTE-IDENTICAL TO `trace_common.hlsli`. The three
//    declarations below are copied verbatim from trace_common.hlsli's
//    RP_SCENE_TEX table, and a `cargo test` in `gfx/shaders.rs` pins them
//    against it. The probe cannot drift from the declaration it stands in for,
//    and touching the corpus's `texs[]` now trips a CPU-only test rather than
//    nothing.
//
// WHAT IT FOUND, FIRST RUN — and property 3 is why it mattered. spirv-cross
// cannot lay out a descriptor set holding an unsized array alongside anything
// else. It drops one member with an `// Overlapping binding:` comment and
// rewrites its uses as a `reinterpret_cast` of the array's storage: it
// COMPILES, reaches AIR, and reads texture descriptor bytes as a sampler. The
// casualty was `samp_lin` — the trilinear sampler every colour/normal/
// rough-metal read goes through — in 19 modules across `leaf`, `leaf_fb`,
// `reference` and `hemi_leaf`, all of which `--check-msl` had been counting as
// PASSES. No ordering inside one set avoids it (array first drops the next
// member, array last or middle drops the array); the array ALONE in its own set
// is clean.
//
// IT IS NOT A TOOL BUG: SPIRV-Cross PR #2292 added the behaviour deliberately,
// and its README states the rule — an array consumes one id per ELEMENT, so an
// unsized one cannot reserve them. The fix was taken at the shader-authoring
// stage the README names: `texs[]` is alone in space2, here and in
// trace_common.hlsli, and `mtl::msl::overlap_check` refuses the pattern so it
// cannot come back quietly. `docs/history/metal-backend.md` carries the
// verbatim output, the arrangements that fail, and why bounding the array lost.

// ---- set 0 -----------------------------------------------------------------
// Declared t, s, s, u ON PURPOSE. See property 1 above; do not sort.
Texture2D<float4>        src         : register(t0);  // 2x1, .x = 0xC30000 | x
SamplerState             samp_clamp  : register(s0);  // nearest + clamp
SamplerState             samp_repeat : register(s1);  // nearest + repeat
RWStructuredBuffer<uint> outbuf      : register(u0);

// ---- sets 1 and 2 ----------------------------------------------------------
// The next three lines are trace_common.hlsli's, verbatim. Property 3 — and
// the SPLIT is the thing being copied, not just the declarations: `texs[]` is
// alone in space2 because an unsized array cannot share a Metal argument
// buffer, which is what this probe found. See trace_common.hlsli for the rule.
Texture2D<float4>        texs[]     : register(t0, space2);
SamplerState             samp_lin   : register(s0, space1);
SamplerState             samp_aniso : register(s1, space1);
// NOT from trace_common — the real table is read-only, but `cs_set1` has to
// have somewhere to write that is not set 0, or it would reference set 0 and
// stop proving that a kernel's argument buffers are found rather than assumed.
RWStructuredBuffer<uint> set1_out   : register(u0, space1);

// The texel payload is `0xC30000 | i`, and it is the HOST's constant, stated
// here only as prose: nothing in this file computes it, so there is no second
// copy to drift. 0xC30000 is arbitrary and non-zero; the low byte is the
// index, so a rotated or short-written descriptor array reads back as a WRONG
// INDEX rather than as garbage — the failure names its own cause.

// ---- set 0 only ------------------------------------------------------------
// One texture reached three ways: through each sampler, and through .Load with
// no sampler at all. u = 1.25 is outside [0,1] on a 2x1, so clamp resolves to
// texel 1 and repeat wraps to 0.25 -> texel 0. The .Load reads texel 1 — the
// SAME texel the clamp sample resolves to, deliberately: a sampler swap moves
// word 0 and leaves word 2 alone, while a texture that never bound moves both.
// The two failures are then distinguishable from the readback alone.
[numthreads(1, 1, 1)]
void cs_tex(uint3 id : SV_DispatchThreadID)
{
    outbuf[0] = (uint)src.SampleLevel(samp_clamp,  float2(1.25, 0.5), 0).x;
    outbuf[1] = (uint)src.SampleLevel(samp_repeat, float2(1.25, 0.5), 0).x;
    outbuf[2] = (uint)src.Load(int3(1, 0, 0)).x;
}

// ---- both sets -------------------------------------------------------------
// The unbounded array, indexed dynamically so the compiler cannot fold the
// walk into five distinct descriptors. Five entries, deliberately not a power
// of two — a stride or rotation bug that happened to be a multiple of the
// count would alias back to correct on 4 or 8.
//
// Each `texs[i]` is 1x1, so the two space1 samplers below are exact by
// construction (a lone texel has no neighbours to weight). They are sampled
// rather than merely declared because spirv-cross assigns ids over the
// REFERENCED resources only: an unreferenced sampler would vanish from the
// module and the set-1 map would be a smaller thing than the one C4 needs.
#define TEX_N 5
[numthreads(1, 1, 1)]
void cs_arr(uint3 id : SV_DispatchThreadID)
{
    for (uint i = 0; i < TEX_N; ++i)
        outbuf[i] = (uint)texs[i].Load(int3(0, 0, 0)).x;
    outbuf[TEX_N + 0] = (uint)texs[0].SampleLevel(samp_lin,   float2(0.5, 0.5), 0).x;
    outbuf[TEX_N + 1] = (uint)texs[1].SampleLevel(samp_aniso, float2(0.5, 0.5), 0).x;
}

// ---- no set 0 --------------------------------------------------------------
// References nothing in space0, so this kernel's argument buffers land at
// [[buffer(1)]] and [[buffer(2)]] and there is NOTHING at index 0. C2's
// `const BUFFER_INDEX: usize = 0` fails here and nowhere else in the corpus —
// and so does a replacement written as a dense 0..n loop over the buffers a
// kernel happens to have. The index has to come out of reflection.
//
// It read one buffer at [[buffer(1)]] until `texs[]` moved to its own space;
// two non-contiguous-from-zero indices is the same property, proven harder.
[numthreads(1, 1, 1)]
void cs_set1(uint3 id : SV_DispatchThreadID)
{
    set1_out[0] = (uint)texs[3].Load(int3(0, 0, 0)).x;
    set1_out[1] = (uint)texs[3].SampleLevel(samp_lin, float2(0.5, 0.5), 0).x;
}
