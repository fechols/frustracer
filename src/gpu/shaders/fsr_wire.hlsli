// The FSR plane WIRE encodings — the twins of fsr.rs's oct_encode/oct_decode/
// quant_unorm. Shared by the two compile units that must agree on them
// bit-for-bit: feed.hlsl (which WRITES the planes and subtracts the wire values
// to form the residual) and fsr_composite.hlsl (which READS them back and
// remodulates). Their agreement IS the composite identity; a private copy in
// either would be a second source of truth for it.

float oct_sign_nz(float v) { return v >= 0.0 ? 1.0 : -1.0; }

// Unit normal -> [0,1]^2 octahedral (the RG of the RGB10A2 normals plane).
float2 oct_encode(float3 n) {
    float inv = 1.0 / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-12);
    float u = n.x * inv;
    float v = n.y * inv;
    if (n.z < 0.0) {
        float ou = u;
        float ov = v;
        u = (1.0 - abs(ov)) * oct_sign_nz(ou);
        v = (1.0 - abs(ou)) * oct_sign_nz(ov);
    }
    return float2(u * 0.5 + 0.5, v * 0.5 + 0.5);
}

// ...and back. Algebraically identical to fsr::oct_decode's branch (for z < 0,
// u - sign(u)*(|u|+|v|-1) == (1 - |v|)*sign(u)); the additive form just avoids
// the branch.
float3 oct_decode(float2 e) {
    float2 f = e * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += float2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}

// The 10-bit UNORM quantum of the normals plane, applied explicitly so the
// value a shader COMPUTES from the normal equals the value it will read back
// out of the plane (the sqrt_enc8 precedent: n/1023 round-trips to the same
// code the hardware store would have produced).
float2 oct_quant10(float2 e) { return floor(saturate(e) * 1023.0 + 0.5) / 1023.0; }

// fsr::wire_normal — the normal as the plane actually stores it. The AO
// signal's remodulation factor is the sky's SH irradiance at THIS normal, on
// every side of the wire.
float3 wire_normal(float3 n) { return oct_decode(oct_quant10(oct_encode(n))); }
