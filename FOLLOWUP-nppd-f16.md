# FOLLOWUP: adapt the in-flight NPPD work to f16 G-buffer storage

The NPPD neural-denoiser work (`src/nppd.rs` + its `main.rs` arms, developed in a
separate working tree) predates this branch's `GBufs` change: five planes
(`normal_rough`, `diff_alb`, `spec_alb`, `mvec`, `spec_hit_t`) are now
`Vec<AtomicU16>` of **f16 bit patterns** (`dlss::ld16`/`st16` are the conversion
sites); `depth` stays `Vec<AtomicU32>` f32. When the NPPD branch merges onto this
one, it will not compile until:

1. **`nppd::pack_inputs`** — the normal channels (camera-space dots) and
   diff_alb channels must load via `dlss::ld16` instead of
   `f32::from_bits(...load())`. Depth and `accum` reads are unchanged (both
   still f32). The ONNX tensor stays `ELEMENT_F32`.

2. **`nppd::warp_temporal`** — the `mvec: &[AtomicU32]` parameter becomes
   `&[AtomicU16]`, its two component loads become `ld16`. The recurrent-state
   planes are `Vec<f32>` and are unaffected.

3. **`nppd::self_test`** — synthetic plane stores go through `dlss::st16`
   (or raw u16 stores), and every oracle that recomputes an expected value
   from an UN-stored f32 must quantize it through
   `half::f16::from_f32(v).to_f32()` before the 1e-6 compare — the packer now
   reads f16-rounded storage. Component-wise for the normal BEFORE the dot
   (that is what the packer computes). This is mandatory, not cosmetic: the
   synthetic `val()` patterns reach ~7000 where f16 spacing is 4. The hand
   anchors are already f16-exact (ln 2 depth rides the f32 depth plane; the
   +Y-normal components are 0/±1; the warp-test MVs 0/2/1/0.5/100 are exact).

Verify with `--check` (runs `nppd::self_test`) and `--check-nppd`.
Delete this file once applied.
