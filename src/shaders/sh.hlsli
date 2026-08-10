// Order-2 SH irradiance (sh.rs::Sh9::irradiance — Ramamoorthi & Hanrahan
// 2001). Term-for-term with the CPU; change both together.
//
// The coefficients are a PARAMETER, not a global, because this function has two
// callers whose 9 rows live in different cbuffers: the tracer's FrameCb (via
// trace_common.hlsli's sh_irradiance) and the FSR composite pass's own root
// constants (fsr_composite.hlsl, which has no shade.hlsli prelude). One formula,
// two bindings — the alternative was a second copy of it, and the two copies
// would silently disagree the moment either moved.
//
// Convention (pinned by sh::self_test): the result is irradiance / PI, i.e. a
// uniform sky of radiance L returns exactly L, so it drops straight into the
// place the old flat AMBIENT constant sat.
float3 sh_irr(float4 c[9], float3 n) {
    const float C1 = 0.429043, C2 = 0.511664, C3 = 0.743125,
                C4 = 0.886227, C5 = 0.247708;
    float x = n.x, y = n.y, z = n.z;
    float3 e = c[0].xyz * C4 - c[6].xyz * C5
             + c[6].xyz * (C3 * z * z)
             + c[8].xyz * (C1 * (x * x - y * y))
             + (c[4].xyz * (x * y) + c[7].xyz * (x * z)
                + c[5].xyz * (y * z)) * (2.0 * C1)
             + (c[3].xyz * x + c[1].xyz * y + c[2].xyz * z) * (2.0 * C2);
    return max(e * (1.0 / 3.14159265358979), 0.0);
}
