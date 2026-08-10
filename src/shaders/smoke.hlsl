// M1 infrastructure self-test: exercises the exact machinery every real
// trace pass uses — DXC/SM6.5 compute PSOs, the shared root signature's
// root constants + root UAVs, a GPU-written counter turned into
// DispatchIndirect args by prep, and ExecuteIndirect consuming them.
// Nothing here survives into the real kernels; only the plumbing does.

cbuffer Push : register(b1) { uint fill_count; uint _p0; uint _p1; uint _p2; }

RWStructuredBuffer<uint>  counters : register(u0); // [0]=produced, [1]=consumer bound
RWStructuredBuffer<uint3> args     : register(u1); // DispatchIndirect (x,y,z)
RWStructuredBuffer<uint>  outbuf   : register(u2);

// Stand-in for a producer pass: publish a work count via the counter.
[numthreads(1, 1, 1)]
void cs_seed(uint3 id : SV_DispatchThreadID)
{
    counters[0] = fill_count;
}

// The prep-args pattern used before every indirect dispatch: counter ->
// threadgroup count. 2D group split (x capped at 32768) is exercised by the
// real pipeline's worst case; the smoke test keeps y for the roundup check.
[numthreads(1, 1, 1)]
void cs_prep(uint3 id : SV_DispatchThreadID)
{
    uint n = counters[0];
    args[0] = uint3((n + 63) / 64, n > 0 ? 1 : 0, 1);
    counters[1] = n;
}

// Stand-in for a consumer pass: guarded grid write.
[numthreads(64, 1, 1)]
void cs_fill(uint3 id : SV_DispatchThreadID)
{
    if (id.x < counters[1])
        outbuf[id.x] = id.x ^ 0x00C0FFEE;
}
