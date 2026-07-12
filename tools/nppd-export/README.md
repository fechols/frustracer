# NPPD → ONNX export

One-time tooling that turns a pretrained [NPPD](https://github.com/balintio/nppd)
checkpoint (Bálint et al., *Neural Partitioning Pyramids for Denoising Monte
Carlo Renderings*, SIGGRAPH 2023) into the inference ONNX graph frustracer's
`--nppd` backend runs through ONNX Runtime + DirectML.

## Licensing (read before redistributing anything)

* The upstream **code** is Apache-2.0. This script clones it at a pinned
  commit and imports the network classes at export time only — no NPPD code
  ships in frustracer.
* The pretrained **weights** (`nppd_pretrained.zip`, hosted on an MPI server)
  carry **no explicit license**. Do NOT commit the checkpoint or the exported
  `.onnx` (a derivative of the weights) to the repo — they live in the
  gitignored `SDKs/nppd/`, exactly like the Streamline DLLs. If the authors
  confirm the weights are Apache-2.0, that call can be revisited.

## Setup

```
pip install torch onnx onnxscript onnxruntime numpy
```

(CPU torch is fine — export only. On Windows set `PYTHONUTF8=1`; the torch
exporter's progress output is emoji and trips cp1252 consoles.)

No Lightning, no noisebase — the checkpoint is `torch.load`ed directly and
only the (torch-only) network definitions are imported from the pinned clone
(`nppd-upstream/`, cloned automatically on first run at the SHA in
`export.py`; delete the directory to re-clone after bumping the pin).

Download the pretrained models from the link in the upstream README
(`nppd_pretrained.zip`) and unzip somewhere local.

## Export

```
python export.py path\to\small_2_spp\checkpoints\....ckpt ^
    --out ..\..\SDKs\nppd\nppd_small.onnx --spp 1 --fp16
```

* `--spp` bakes the sample-dim size S into the graph (frustracer v1 feeds
  S = 1; the `small_2_spp` checkpoint was trained at 2 spp — the sample dim
  is mean-pooled, so S = 1 runs fine with a mild statistics mismatch).
* `--fp16` converts the graph internals to fp16 AFTER the strict fp32 verify
  (I/O tensors stay fp32 via `keep_io_types` — the Rust packer and staging
  are unchanged either way). The converter is `onnxruntime.transformers.
  float16`, NOT onnxconverter-common (which misses a boundary Cast on this
  graph and writes an invalid model); its own quirk — the same boundary cast
  emitted once per consumer — is deduped by the script. fp16 is for the DML
  EP; the CPU EP is SLOWER on fp16 internals, so keep an fp32 export around
  (`nppd_small_fp32.onnx`) if you use `--nppd-device cpu`. Quality is
  indistinguishable (output mean rel ~2e-4 vs fp32).
* `--network auto` detects 15M (ConvUNet, the `small_*` checkpoints) vs 30M
  (ConvUNeXt, `large_*`) from the state-dict shape; the loader hard-fails on
  any missing/unexpected tensor rather than half-loading.
* `--golden g.npz` additionally dumps a verified input/output pair for
  cross-checking the Rust ORT plumbing (reflects the fp16 graph when --fp16).

The script self-verifies before finishing: the re-implemented (export-safe)
partitioning pyramid against the upstream class at 1e-5, and the exported
ONNX against PyTorch at 1e-4 — at the trace resolution AND a second one, so
the dynamic H/W axes are proven, not just declared. With `--fp16` a second
verify pass gates the converted graph (output mean rel < 2e-2, max < 0.1,
temporal mean rel < 5e-2; measured ~2e-4 / ~3e-3 / ~2e-4 on small_2_spp).

Perf note (measured, RTX 4090, 1280×736): the graph is **launch-bound** —
~1600 tiny Slice/Mul/Gather ops from the pyramid's dynamic-shape math; the
convolutions are ~1 ms. fp16 alone is therefore only ~10% faster through the
CPU-staged path. The wins come from the session, not the graph:
pinning the dynamic h/w dims at session creation (frustracer does this —
`AddFreeDimensionOverrideByName`) folds the shape chains (~25%), and binding
DML device memory for I/O (the `--gpu` path) removes the ~340 MB/run staging
traffic: 94 ms fp32-staged → 26 ms fp16-frozen-bound.

## Matching runtime DLLs

The exported graph is opset 18 / IR version 10 — the `onnxruntime.dll`
dropped in `SDKs/onnxruntime/bin` must be **≥ 1.22** (from the
Microsoft.ML.OnnxRuntime.DirectML NuGet), and `DirectML.dll` must be recent
enough for that runtime (Microsoft.AI.DirectML **≥ 1.15** — an old 1.13
DirectML under ORT 1.24 fails the U-Net's nearest-2× Resize node at run time
with `80070057 The parameter is incorrect`). Verified pairing: ORT 1.24.4 +
DirectML 1.15.4.

## What the graph expects (the src/nppd.rs packer contract)

Inputs are the Noisebase training-time buffers, reproduced by
`nppd::pack_inputs`:

| channel | content |
|---|---|
| 0 | log-depth `ln(1 + 1/d)`, d = **Euclidean** camera distance; sky = 0 |
| 1-3 | **camera-space** normal `(n·forward, n·right, n·up)` |
| 4-6 | diffuse albedo, linear |
| 7-9 | radiance, linear HDR, per sample |

`normalize_radiance`/`clip_logp1` are inside the graph. The recurrent state
(`temporal_warped`, 38 channels = color 3 + output 3 + feature 32) must be
backward-warped by motion vectors BEFORE the graph runs — frustracer does
this in Rust (`nppd::warp_temporal`); the upstream `grid_sample` reprojection
is deliberately excised so DirectML never sees a GridSample op. H and W must
be multiples of 16 (the K=5 pyramid); frustracer pads to /32.
