// Feed pass: fan the tracer's G-buffer pack (gbuf, u15) + the 1-spp radiance
// (accum) out into an upscaler's input textures — the GPU-resident
// replacement for rr.rs/xr.rs::record_upload's CPU convert-and-copy. One
// thread per pixel; the target textures are typed UAVs reached through the
// second range of the u14 descriptor table (registers u16..u22, heap slots
// 1..7 — trace.rs::wire_feed builds the descriptors per session).
// Requires trace_common.hlsli pasted first.
//
// Register/type layout is shared by both kernels (same compile unit); each
// kernel writes only its planes and DXC strips the rest. Where both write
// the same register the TYPE matches; the VALUE may differ (u18 depth is
// reversed-Z clip depth for XeSS but raw linear view-Z for RR).

RWStructuredBuffer<float> accum : register(u0); // rw*rh*3 linear HDR (1-spp store)

RWTexture2D<float4> feed_color   : register(u16); // RGBA16F  (RR/XeSS); RGBA16F residual (FSR-RR)
RWTexture2D<float4> feed_nr      : register(u17); // RGBA16F  normal.xyz + rough (RR); RGB10A2 oct-normals (FSR-RR)
RWTexture2D<float>  feed_depth   : register(u18); // R32F     (all; encoding differs per kernel)
RWTexture2D<float2> feed_mvec    : register(u19); // RG16F    (RR/XeSS)
RWTexture2D<float4> feed_alb     : register(u20); // RGBA8    diffuse albedo (RR/FSR-RR, sqrt-encoded there)
RWTexture2D<float4> feed_spec    : register(u21); // RGBA8    specular albedo F0 (RR/FSR-RR)
RWTexture2D<float>  feed_spechit : register(u22); // R16F     spec hit distance (RR); R32F linear depth (FSR-RR)
RWTexture2D<float4> feed_aux0    : register(u23); // RGBA16F  FSR-RR mvec (UV-delta RG + depth-delta B)
RWTexture2D<float4> feed_aux1    : register(u24); // RGBA16F  FSR-RR demodulated direct diffuse
RWTexture2D<float4> feed_aux2    : register(u25); // RGBA16F  FSR-RR demodulated direct specular
RWTexture2D<float>  feed_aux3    : register(u26); // R16F     FSR-RR ambient-occlusion open fraction;
                                                  //          RR: the FG guide pass's ripple tag.
                                                  //          Same HLSL type + format, and the two
                                                  //          sessions are mutually exclusive per
                                                  //          descriptor set — the register's VALUE
                                                  //          may differ per kernel, its TYPE may not.
RWTexture2D<float4> feed_aux4    : register(u27); // RGBA16F  FSR-RR indirect specular (A = ray hit t)

// view_z_to_clip_depth moved to trace_common.hlsli (pasted ahead of this
// unit) — the NRD pack fold made it a two-unit function, and a precise-
// sensitive encode must have exactly one copy.

[numthreads(8, 8, 1)]
void cs_feed_xess(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint pi = id.y * rw + id.x;
    uint i3 = pi * 3u;
#ifdef ABL_NOPACK
    // Cost probe (see trace::abl_defs): skip the pack load entirely and write
    // constants, so the delta against the real kernel is exactly the 88 B/px
    // read. The accum load and all three stores stay, so only the pack's
    // traffic moves. Image is wrong by design.
    feed_color[id.xy] = float4(accum[i3], accum[i3 + 1u], accum[i3 + 2u], 1.0);
    feed_mvec[id.xy] = float2(0.0, 0.0);
    feed_depth[id.xy] = 0.5;
#else
    // Core only — this kernel is the reason the pack is split (XeSS and
    // FSR 3.1 read 12 of the old 88 B/px).
    GBufCore g = gbuf[pi];
    feed_color[id.xy] = float4(accum[i3], accum[i3 + 1u], accum[i3 + 2u], 1.0);
    feed_mvec[id.xy] = g.core.xy;
    feed_depth[id.xy] = view_z_to_clip_depth(g.core.z, CAM_NEAR, CAM_FAR);
#endif
}

// NPPD pre-denoise frames (--gpu --nppd): the color plane comes from
// cs_nppd_out (nppd.hlsl) instead — this writes only the guide planes.
[numthreads(8, 8, 1)]
void cs_feed_xess_dm(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    GBufCore g = gbuf[id.y * rw + id.x];
    feed_mvec[id.xy] = g.core.xy;
    feed_depth[id.xy] = view_z_to_clip_depth(g.core.z, CAM_NEAR, CAM_FAR);
}

[numthreads(8, 8, 1)]
void cs_feed_rr(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint pi = id.y * rw + id.x;
    uint i3 = pi * 3u;
    GBufCore c = gbuf[pi];
    GBufExt g = gbuf_ext[pi]; // FLAG_GBUF_EXT is set whenever this kernel runs
    // EMISSIVE DEMODULATION (FLAG_EMIS_DEMOD): hand RR the radiance WITHOUT
    // the primary hit's emission. The guides beside this plane describe
    // REFLECTANCE, which explains none of an emitter's emission, so an
    // undemodulated ~1e3-radiance texel gets redistributed across RR's
    // receptive field — measured at +0.488 stops of whole-frame lift on an
    // emitter-free region (gfx::frame::FLAG_EMIS_DEMOD carries the numbers).
    // `cs_rr_emis_readd` puts it back after RR resolves, decoding the SAME
    // lane, so the round trip cancels exactly and only what RR SAW changed.
    // Structurally off: a BRANCH, never a `- 0.0`.
    float3 lit = float3(accum[i3], accum[i3 + 1u], accum[i3 + 2u]);
    if (flags & FLAG_EMIS_DEMOD) {
        lit = max(lit - emis_unpack(g.sig.w), 0.0);
    }
    feed_color[id.xy] = float4(lit, 1.0);
    feed_nr[id.xy] = g.nr;
    feed_depth[id.xy] = c.core.z; // linear view-Z, RR's LINEAR_DEPTH contract
    feed_mvec[id.xy] = c.core.xy;
    // UNORM stores clamp to [0,1] in hardware (the CPU path's to_unorm8).
    feed_alb[id.xy] = float4(g.alb.xyz, 1.0);
    feed_spec[id.xy] = float4(g.spec.xyz, 1.0);
    feed_spechit[id.xy] = g.spec.w;
    // Pack lane 7 (alb.w) -> the guide pass's ripple plane. Not an RR input:
    // RR is never tagged with it. Zero on every non-water pixel, which is the
    // guide kernel's structural "nothing moves here" arm.
    feed_aux3[id.xy] = g.alb.w;
}

// --- FSR4 + Ray Regeneration (ffx_rr.rs::record_upload's GPU replacement) ---
// Mirror of fsr.rs's polarity/type constants — fsr.rs is the source of truth;
// keep in lockstep (they are "settle on RDNA4 hardware" knobs, default 1/0).
#define FSR_DEPTH_SIGN 1.0
#define FSR_MAT_TYPE   0.0

// oct_encode / oct_decode / oct_quant10 / wire_normal come from fsr_wire.hlsli,
// which fsr_composite.hlsl pastes too — the two sides of the composite identity
// must not each own a copy of the wire encoding.

// The nine Ray Regeneration + FSR4 planes from pack + accum. The demodulated
// signals ride the pack's f16 sig lanes (bit-preserved onto RGBA16F); the
// albedo planes store the EXPLICITLY quantized sqrt encode, and the residual
// multiplies the square of that same enc — plane byte and wire factor derive
// from one quantization, which IS the composite identity (fsr.rs's
// split_signals/composite contract; fsr_composite.hlsl decodes b*b).
[numthreads(8, 8, 1)]
void cs_feed_fsr_rr(uint3 id : SV_DispatchThreadID) {
    if (id.x >= rw || id.y >= rh) return;
    uint pi = id.y * rw + id.x;
    uint i3 = pi * 3u;
    GBufCore c = gbuf[pi];
    GBufExt g = gbuf_ext[pi];
    // Demodulated signals from the pack's f16x2 lanes (exact widen).
    float3 dd = float3(f16tof32(g.sig.x & 0xffffu), f16tof32(g.sig.x >> 16u),
                       f16tof32(g.sig.y & 0xffffu));
    float3 ds = float3(f16tof32(g.sig.y >> 16u), f16tof32(g.sig.z & 0xffffu),
                       f16tof32(g.sig.z >> 16u));
    float ao = f16tof32(g.sig2.x & 0xffffu);
    float3 is = float3(f16tof32(g.sig2.x >> 16u), f16tof32(g.sig2.y & 0xffffu),
                       f16tof32(g.sig2.y >> 16u));
    feed_aux1[id.xy] = float4(dd, 0.0);
    feed_aux2[id.xy] = float4(ds, 0.0);
    feed_aux3[id.xy] = ao;
    // Indirect specular carries its ray hit distance in A (the signal's own
    // channel layout) — the pack's spec_hit_t lane, same value RR's own
    // spec-hit plane gets.
    feed_aux4[id.xy] = float4(is, g.spec.w);
    // sqrt-encoded albedos; enc is exactly representable as n/255, so the
    // UNORM8 store round-trips to the same byte the wire factors assume.
    float3 kd_enc = float3(sqrt_enc8(g.alb.x), sqrt_enc8(g.alb.y), sqrt_enc8(g.alb.z));
    float3 f0_enc = float3(sqrt_enc8(g.spec.x), sqrt_enc8(g.spec.y), sqrt_enc8(g.spec.z));
    feed_alb[id.xy] = float4(kd_enc, 1.0);
    feed_spec[id.xy] = float4(f0_enc, 1.0);
    // Exact remainder — `precise` (no FMA re-association; the subtraction is
    // a near-cancellation of same-magnitude terms, so a fused multiply-add
    // would move the result by far more than its own size) and f16-saturated
    // (an inf on the wire turns the composite into NaN). Every remodulated
    // term comes off here, in fsr_composite.hlsl's exact set — the three
    // sites' agreement IS the composite identity.
    precise float3 kd_wire = kd_enc * kd_enc;
    precise float3 f0_wire = f0_enc * f0_enc;
    // The AO signal's remodulation factor is the sky's own SH irradiance, not a
    // constant (the one sky — sky.rs), so it is DIRECTIONAL and must be built
    // from the WIRE normal: fsr_composite.hlsl has only the plane bytes to
    // rebuild it from. Quantize first, store exactly what we quantized.
    float2 n_enc = oct_quant10(oct_encode(g.nr.xyz));
    precise float3 amb = sh_irr(sky_sh, oct_decode(n_enc));
    precise float3 res = float3(accum[i3], accum[i3 + 1u], accum[i3 + 2u])
        - dd * kd_wire - ds * f0_wire - ao * amb * kd_wire - is * f0_wire;
    feed_color[id.xy] = float4(clamp(res, -65504.0, 65504.0), 0.0);
    // Octahedral normal + roughness + material type (RGB10A2; A's 2-bit
    // quantum holds FSR_MAT_TYPE's class, 0 = default).
    feed_nr[id.xy] = float4(n_enc, g.nr.w, FSR_MAT_TYPE);
    // Denoiser MVs: UV-delta RG (the pixel-space MV over the render dims),
    // linear-depth delta B from the pack's prev-Z lane (sky: far - far = 0).
    feed_aux0[id.xy] = float4(c.core.x / float(rw), c.core.y / float(rh),
                              c.core.w - c.core.z, 0.0);
    // Both depth encodes: reversed-Z clip for the upscaler, signed linear
    // view-Z for the denoiser.
    feed_depth[id.xy] = view_z_to_clip_depth(c.core.z, CAM_NEAR, CAM_FAR);
    feed_spechit[id.xy] = FSR_DEPTH_SIGN * c.core.z;
}

// --- FLAG_EMIS_DEMOD's other half: put the emission back after RR ----------
//
// Runs between `rr_ngx_sequence` and the tonemap, while `rr.output` is still
// a UAV. Reads the SAME `sig.w` lane cs_feed_rr subtracted, so the pair
// cancels to the pre-fix radiance up to the R11G11B10F quantum — the only
// approximation is in what RR SAW, never in what is presented.
//
// It re-adds BEFORE the tonemap source is finalized rather than inside the
// tonemap draw, because the bloom pyramid reads that source: folding it into
// the tonemap would have silently stopped emitters blooming.
//
// Dispatched over the OUTPUT grid, whose extent comes from the UAV itself
// (no new root constants); the emissive is point-sampled from the render
// grid, which is an exact 1:1 map at DLAA. Emission is a sharp, noise-free
// feature, so nearest is the honest reconstruction — there is nothing to
// interpolate between.
//
// It writes through `feed_color` (u16) rather than a register of its own: the
// re-add rides a spare FEED SET whose slot 0 holds RR's OUTPUT as a UAV
// (GpuContext::wire_rr_feed), so the register is already in the root
// signature and its TYPE matches. A private register would need a root-layout
// change in BOTH pipelines — and the DXR root signature rejected exactly that
// (`CreateComputePipelineState: The parameter is incorrect`). This is the
// FEED_FSR_AO/ripple precedent: the register's VALUE may differ per kernel,
// its type may not.

[numthreads(8, 8, 1)]
void cs_rr_emis_readd(uint3 id : SV_DispatchThreadID) {
    // THE SAME PREDICATE cs_feed_rr subtracted under, and not a weaker one.
    // The dispatch is gated on the RrEmis descriptor SET being wired, which is
    // a strictly coarser condition than the CB bit: `with_frame` also requires
    // the ext pack and, decisively, `!FLAG_FSR_SIG` — because `sig.w` carries
    // `pack_h2(ao_t, shadow_t)` there. A frame that wired the set but cleared
    // the bit would decode two f16 DISTANCES as an emissive triple and add
    // hundreds of nits to every pixel. Reading the bit here makes subtract and
    // re-add the same condition by construction, which is also what makes
    // `--no-rr-emis-demod` a BRANCH rather than a `+ 0.0` (`x + 0.0 != x` for
    // -0.0, and the off arm must be bit-identical to the pre-fix renderer).
    if ((flags & FLAG_EMIS_DEMOD) == 0u) return;
    uint ow, oh;
    feed_color.GetDimensions(ow, oh);
    if (id.x >= ow || id.y >= oh) return;
    uint sx = min((id.x * rw) / ow, rw - 1u);
    uint sy = min((id.y * rh) / oh, rh - 1u);
    float3 e = emis_unpack(gbuf_ext[sy * rw + sx].sig.w);
    float4 c = feed_color[id.xy];
    feed_color[id.xy] = float4(c.rgb + e, c.a);
}
