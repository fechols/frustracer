// Wave-width probe: what the driver ACTUALLY picked for a kernel of a given
// thread-group width.
//
// `D3D12_OPTIONS1` reports WaveLaneCountMin/Max — a RANGE the driver may pick
// from PER SHADER, not the width any particular kernel got. On Intel Xe2 that
// range is [16, 32] (Xe-HPG was [8, 32] before SIMD8 was dropped), and Intel's
// compiler deliberately picks the narrower width to avoid spilling a
// register-heavy kernel. So the range bounds what `WaveGetLaneCount()` can
// return and never predicts it, and the only way to learn the real width is to
// ask a kernel of the real shape. Compiled once per distinct group width the
// tracer actually dispatches (see `trace::wave_probe`).
//
// Deliberately trivial: no groupshared, no rays, one dispatch of one group.
// The lane count is a property of the compiled kernel's group width, not of
// its body, so a probe that mimicked a real kernel's register pressure would
// measure the probe rather than the kernel it stands in for. Read the numbers
// as "what a kernel of this WIDTH gets", nothing finer.

// [0] = WaveGetLaneCount(), [1] = the group width, [2] = waves per group.
RWStructuredBuffer<uint> out_buf : register(u0);

#ifndef PROBE_GROUP
#define PROBE_GROUP 32
#endif

[numthreads(PROBE_GROUP, 1, 1)]
void cs_wave_probe(uint3 gtid : SV_GroupThreadID)
{
    if (gtid.x == 0) {
        out_buf[0] = WaveGetLaneCount();
        out_buf[1] = PROBE_GROUP;
    }
    // Counts waves independently of the reported width, so a driver whose
    // WaveGetLaneCount() disagreed with how it actually partitioned the group
    // would show up as an inconsistent pair rather than a plausible lie.
    // The buffer is a fresh committed resource (HEAP_FLAG_NONE => D3D12
    // zero-initializes it), so this accumulates from a known 0.
    if (WaveIsFirstLane()) {
        uint prev;
        InterlockedAdd(out_buf[2], 1u, prev);
    }
}
