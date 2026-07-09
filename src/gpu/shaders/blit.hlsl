// Fullscreen-triangle blit of the CPU-tonemapped B8G8R8A8 frame to the
// backbuffer. Alpha in the source is 0 (the CPU buffer is 0RGB) — force 1.
Texture2D<float4> src : register(t0);

float4 vsmain(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 psmain(float4 pos : SV_Position) : SV_Target {
    return float4(src.Load(int3(pos.xy, 0)).rgb, 1.0);
}
