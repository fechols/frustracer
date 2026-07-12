#!/usr/bin/env python3
"""One-time NPPD -> ONNX export for frustracer's --nppd backend.

Takes a pretrained NPPD Lightning checkpoint (Bálint et al., SIGGRAPH 2023,
https://github.com/balintio/nppd, Apache-2.0) and produces an inference-only
ONNX graph with the recurrent temporal state as explicit I/O:

    frame_input    [1, S, 10, H, W]   depth | normal(3) | diffuse(3) | color(3)
    temporal_warped [1, 38, H, W]     previous state, ALREADY backward-warped
        -> output       [1, 3, H, W]  denoised linear-HDR radiance
        -> temporal_out [1, 38, H, W] state to warp + feed next frame

Deliberate differences from the upstream model.step():
  * The motion-vector reprojection (backproject_pixel_centers + grid_sample)
    is EXCISED — frustracer warps the state in Rust (src/nppd.rs::
    warp_temporal, bilinear + zeros padding, the same semantics). No
    GridSample op ever reaches DirectML.
  * The parameter-free PartitioningPyramid is re-implemented with
    dynamic-shape-safe ops (negative-end slices instead of Python-int shape
    math, pixel_shuffle instead of strided assignment) and verified against
    the upstream implementation to 1e-5 before export.
  * normalize_radiance / clip_logp1 stay INSIDE the graph; the Rust packer
    stages the Noisebase dataloader transforms (log-depth ln(1 + 1/d) of the
    EUCLIDEAN camera distance with sky = 0, CAMERA-space normals
    (n.forward, n.right, n.up), raw linear diffuse + HDR color).

Requires only: torch, onnx, onnxruntime, numpy — no Lightning, no noisebase
(the checkpoint is torch.load-ed directly; the network classes are imported
from a pinned clone of the upstream repo).

Usage:
    python export.py path/to/small_2_spp/....ckpt --out ../../SDKs/nppd/nppd_small.onnx --spp 1
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pinned upstream commit (checked after clone; bump deliberately).
NPPD_REPO = "https://github.com/balintio/nppd"
NPPD_SHA = "3b7ac8452370eb5c74a8f3339c2cd1a843fa33d3"

C_T = 38  # temporal state channels: color 3 + output 3 + feature 32
T_LAMBDA_INDEX = 51
K = 5  # pyramid levels; /16 at the coarsest, frustracer pads to /32

IN_FRAME = "frame_input"
IN_TEMPORAL = "temporal_warped"
OUT_COLOR = "output"
OUT_TEMPORAL = "temporal_out"


def upstream_src(args):
    """Return the upstream src/ directory, cloning at the pinned SHA if needed."""
    if args.nppd_src:
        src = os.path.join(args.nppd_src, "src")
        if not os.path.isdir(src):
            sys.exit(f"--nppd-src: {src} does not exist")
        return src
    clone = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nppd-upstream")
    if not os.path.isdir(clone):
        print(f"cloning {NPPD_REPO} @ {NPPD_SHA[:12]} -> {clone}")
        subprocess.check_call(["git", "clone", "--no-checkout", NPPD_REPO, clone])
        subprocess.check_call(["git", "-C", clone, "checkout", NPPD_SHA])
    head = subprocess.check_output(["git", "-C", clone, "rev-parse", "HEAD"], text=True).strip()
    if head != NPPD_SHA:
        sys.exit(f"nppd-upstream is at {head[:12]}, expected {NPPD_SHA[:12]} — "
                 "delete tools/nppd-export/nppd-upstream or pass --nppd-src")
    return os.path.join(clone, "src")


# ---------------------------------------------------------------------------
# PartitioningPyramid, re-implemented export-friendly. The upstream version
# (src/partitioning_pyramid.py) slices with Python ints derived from
# tensor.shape (bakes H/W into the trace, breaking dynamic axes) and builds
# the 2x upscale by strided assignment (ScatterND in ONNX). This version uses
# constant negative-end slices and pixel_shuffle; `verify_pyramid` gates it
# against the upstream class before every export.
# ---------------------------------------------------------------------------

def splat(img, kernel, size=5):
    """Sum over 5x5 shifted copies weighted by the per-pixel splat kernel."""
    pad = (size - 1) // 2
    img = F.pad(img, [pad] * 4)
    kernel = F.pad(kernel, [pad] * 4)
    total = None
    for i in range(size):
        for j in range(size):
            # img[:, :, i:i+H, j:j+W] on the padded tensor == negative-end
            # slice with static bounds: end = i - 2*pad (or None when 0).
            ei = i - 2 * pad if i - 2 * pad != 0 else None
            ej = j - 2 * pad if j - 2 * pad != 0 else None
            term = img[:, :, i:ei, j:ej] * kernel[:, i * size + j, None, i:ei, j:ej]
            total = term if total is None else total + term
    return total


def upscale_quadrant(img, kernel, indices):
    """2x nearest upscale of img with one of 4 kernel taps per output quadrant
    position — pixel_shuffle of the 4 weighted copies (channel-interleaved),
    replacing upstream's strided assignment."""
    taps = [img * kernel[:, k, None] for k in indices]  # 4 x [B,C,h,w]
    # pixel_shuffle maps input channel c*4 + (i*2 + j) to output (2y+i, 2x+j).
    stacked = torch.stack(taps, 2).flatten(1, 2)  # [B, C*4, h, w]
    return F.pixel_shuffle(stacked, 2)


def upscale(img, kernel):
    img = F.pad(img, (1, 1, 1, 1))
    kernel = F.pad(kernel, (1, 1, 1, 1))
    tl = upscale_quadrant(img, kernel, [0, 1, 4, 5])
    tr = upscale_quadrant(img, kernel, [2, 3, 6, 7])
    bl = upscale_quadrant(img, kernel, [8, 9, 12, 13])
    br = upscale_quadrant(img, kernel, [10, 11, 14, 15])
    return (tl[:, :, 3:-1, 3:-1] + tr[:, :, 3:-1, 1:-3]
            + bl[:, :, 1:-3, 3:-1] + br[:, :, 1:-3, 1:-3])


def pyramid_filter(weights, rendered, previous):
    """The upstream PartitioningPyramid.__call__, ops swapped as above."""
    part_weights = F.softmax(weights[0][:, 52:], 1)
    partitions = part_weights[:, :, None] * rendered[:, None]

    denoised_levels = [
        splat(F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
              F.softmax(weights[i][:, 0:25], 1))
        for i in range(K)
    ]

    denoised = denoised_levels[-1]
    for i in reversed(range(K - 1)):
        denoised = denoised_levels[i] + upscale(denoised, F.softmax(weights[i + 1][:, 25:41], 1) * 4)

    previous = splat(previous, F.softmax(weights[0][:, 25:50], 1))
    t_mu = torch.sigmoid(weights[0][:, 50, None])
    return t_mu * previous + (1 - t_mu) * denoised


def verify_pyramid(upstream_mod):
    """Gate this file's pyramid against the upstream class on random tensors."""
    torch.manual_seed(0)
    up = upstream_mod.PartitioningPyramid()
    h, w = 64, 96
    weights = [torch.randn(1, 57, h, w)] + [
        torch.randn(1, 41, h // 2 ** i, w // 2 ** i) for i in range(1, K)
    ]
    rendered = torch.randn(1, 3, h, w).abs()
    previous = torch.randn(1, 3, h, w).abs()
    with torch.no_grad():
        a = up(weights, rendered, previous)
        b = pyramid_filter(weights, rendered, previous)
    err = (a - b).abs().max().item()
    if err > 1e-5:
        sys.exit(f"pyramid re-implementation mismatch: max |delta| {err:.3e}")
    print(f"pyramid re-implementation verified: max |delta| {err:.3e}")


# ---------------------------------------------------------------------------
# Inference wrapper: upstream Model.step() minus the reprojection.
# ---------------------------------------------------------------------------

def clip_logp1(x):
    return torch.log1p(torch.clamp(x, min=0))


class InferenceModel(nn.Module):
    def __init__(self, encoder, weight_predictor):
        super().__init__()
        self.encoder = encoder
        self.weight_predictor = weight_predictor

    def forward(self, frame_input, temporal_warped):
        prev_color = temporal_warped[:, :3]
        prev_output = temporal_warped[:, 3:6]
        prev_feature = temporal_warped[:, 6:]

        b, s = frame_input.shape[0], frame_input.shape[1]
        raw_color = frame_input[:, :, 7:10]  # [B,S,3,H,W]

        # normalize_radiance: mean over everything but batch (util.py) — here
        # dims (S, C, H, W) of the color stack — then clip_logp1.
        mean = raw_color.mean(dim=(1, 2, 3, 4), keepdim=True) + 1e-8
        norm_color = clip_logp1(raw_color / mean)
        enc_in = torch.cat((frame_input[:, :, 0:7], norm_color), 2)

        feature = self.encoder(enc_in.flatten(0, 1))  # [B*S,32,H,W]
        feature = feature.unflatten(0, (b, s)).mean(1)
        color = raw_color.mean(1)  # [B,3,H,W] raw radiance

        pc = torch.cat((prev_color, color), 1)
        mean2 = pc.mean(dim=(1, 2, 3), keepdim=True) + 1e-8
        wp_in = torch.cat((clip_logp1(pc / mean2), prev_feature, feature), 1)
        weights = self.weight_predictor(wp_in)

        t_lambda = torch.sigmoid(weights[0][:, T_LAMBDA_INDEX, None])
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = pyramid_filter(weights, color, prev_output)
        return output, torch.cat((color, output, feature), 1)


def build_model(ckpt_path, network, src):
    sys.path.insert(0, src)
    import networks.convunet as convunet
    import networks.convnext as convnext
    import partitioning_pyramid as upstream_pyramid

    verify_pyramid(upstream_pyramid)

    inputs = [25 + 25 + 1 + 1 + K] + [41] * (K - 1)  # PartitioningPyramid.inputs
    encoder = nn.Sequential(
        nn.Conv2d(10, 32, 1), nn.LeakyReLU(0.3),
        nn.Conv2d(32, 32, 1), nn.LeakyReLU(0.3),
        nn.Conv2d(32, 32, 1),
    )

    if ckpt_path is None:
        # Plumbing-test mode (--random-init): no checkpoint, random weights.
        # The exported graph is structurally identical (shapes, I/O, ops) but
        # denoises garbage — for exercising the Rust ORT path, never images.
        name = "15M" if network in ("auto", "15M") else "30M"
        wp = (convunet.ConvUNet if name == "15M" else convnext.ConvUNeXt)(70, inputs)
        model = InferenceModel(encoder, wp)
        print(f"RANDOM-INIT {name} model — plumbing test only")
        model.eval()
        return model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    def try_network(name):
        wp = (convunet.ConvUNet if name == "15M" else convnext.ConvUNeXt)(70, inputs)
        model = InferenceModel(encoder, wp)
        want = {k: v for k, v in sd.items()
                if k.startswith("encoder.") or k.startswith("weight_predictor.")}
        missing, unexpected = model.load_state_dict(want, strict=False)
        # Every checkpoint tensor for these submodules must be consumed and
        # every model parameter covered — a silent partial load is a wrong
        # denoiser, not an error message.
        return model, missing, unexpected, want

    candidates = ["15M", "30M"] if network == "auto" else [network]
    last = None
    for name in candidates:
        model, missing, unexpected, want = try_network(name)
        if not missing and not unexpected:
            print(f"network: {name} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
                  f"{len(want)} tensors)")
            model.eval()
            return model
        last = (name, missing, unexpected)
    name, missing, unexpected = last
    sys.exit(f"checkpoint does not match network {name}: "
             f"missing {missing[:3]}... unexpected {unexpected[:3]}...")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?",
                    help="pretrained .ckpt (e.g. small_2_spp); omit with --random-init")
    ap.add_argument("--random-init", action="store_true",
                    help="no checkpoint: export a random-weights graph "
                         "(structure-only plumbing test, never for images)")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--spp", type=int, default=1, help="fixed sample count S baked into the graph")
    ap.add_argument("--network", choices=["auto", "15M", "30M"], default="auto")
    ap.add_argument("--nppd-src", help="existing upstream clone (default: clone pinned SHA here)")
    # 18: the dynamo exporter implements >= 18 and its 18 -> 17 version
    # down-conversion fails on the Resize axes form anyway. Needs an ORT
    # runtime >= 1.22 (IR version 10).
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--golden", help="write a golden input/output .npz for --check-nppd")
    ap.add_argument("--fp16", action="store_true",
                    help="after the fp32 verify, convert the graph internals to fp16 "
                         "(I/O stays fp32 via keep_io_types — the Rust packer and both "
                         "runtime paths are unchanged; ~2x on DML, SLOWER on the CPU EP)")
    args = ap.parse_args()
    if (args.checkpoint is None) == (not args.random_init):
        ap.error("pass a checkpoint OR --random-init (exactly one)")

    model = build_model(args.checkpoint, args.network, upstream_src(args))

    s = args.spp
    h, w = 192, 320  # export-time trace resolution (dynamic axes below)
    torch.manual_seed(1)
    frame = torch.rand(1, s, 10, h, w)
    frame[:, :, 7:10] *= 4.0  # HDR-ish color
    temporal = torch.rand(1, C_T, h, w)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (frame, temporal),
            args.out,
            opset_version=args.opset,
            input_names=[IN_FRAME, IN_TEMPORAL],
            output_names=[OUT_COLOR, OUT_TEMPORAL],
            dynamic_axes={
                IN_FRAME: {3: "h", 4: "w"},
                IN_TEMPORAL: {2: "h", 3: "w"},
                OUT_COLOR: {2: "h", 3: "w"},
                OUT_TEMPORAL: {2: "h", 3: "w"},
            },
        )
    print(f"exported {args.out} (opset {args.opset}, S={s})")

    # Verify ONNX vs PyTorch at the trace res AND a different res (dynamic
    # axes are real, not just declared).
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    for vh, vw in [(h, w), (96, 160)]:
        fi = torch.rand(1, s, 10, vh, vw)
        fi[:, :, 7:10] *= 4.0
        ti = torch.rand(1, C_T, vh, vw)
        with torch.no_grad():
            ref_out, ref_tmp = model(fi, ti)
        got = sess.run([OUT_COLOR, OUT_TEMPORAL],
                       {IN_FRAME: fi.numpy(), IN_TEMPORAL: ti.numpy()})
        e1 = np.abs(got[0] - ref_out.numpy()).max()
        e2 = np.abs(got[1] - ref_tmp.numpy()).max()
        print(f"verify {vw}x{vh}: max |delta| output {e1:.3e}, temporal {e2:.3e}")
        if e1 > 1e-4 or e2 > 1e-4:
            sys.exit("ONNX/PyTorch mismatch above 1e-4")

    if args.fp16:
        # Convert internals to fp16 AFTER the strict fp32 verify passed, so a
        # conversion problem is attributable. keep_io_types leaves the four
        # I/O tensors fp32 (Cast nodes at the boundary) — nothing outside the
        # graph changes. Converter choice is deliberate: onnxconverter-common
        # misses a boundary Cast on this graph (mixed-type Concat, invalid
        # model); onnxruntime.transformers.float16 (ORT's patched fork of the
        # same code) converts correctly but emits the SAME boundary cast once
        # per consumer — bit-identical duplicate nodes ORT then rejects, so we
        # keep the first of each and drop the rest. The default op block list
        # keeps fp16-hostile ops in fp32 automatically; if normalize_radiance's
        # ReduceMean ever overflows on real HDR, add it to node_block_list.
        import onnx
        from onnxruntime.transformers import float16
        m16 = float16.convert_float_to_float16(onnx.load(args.out), keep_io_types=True)
        seen_out = {}
        keep = []
        dropped = 0
        for node in m16.graph.node:
            key = tuple(node.output)
            if key in seen_out:
                if node.SerializeToString() != seen_out[key].SerializeToString():
                    sys.exit(f"duplicate output {key} from non-identical nodes — cannot dedup")
                dropped += 1
                continue
            seen_out[key] = node
            keep.append(node)
        del m16.graph.node[:]
        m16.graph.node.extend(keep)
        if dropped:
            print(f"fp16: dropped {dropped} duplicate boundary casts")
        onnx.save(m16, args.out)
        # onnx.load materialized any external-data weights and the fp16 save
        # inlines them (~31 MB for small_*) — drop the now-stale sidecar. If a
        # future model were big enough to still need it, the re-verify's
        # session load below would fail loudly.
        try:
            os.remove(args.out + ".data")
            print(f"removed stale {args.out}.data (weights now inline)")
        except OSError:
            pass
        print("converted internals to fp16 (I/O stays fp32), re-verifying...")
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        for vh, vw in [(h, w), (96, 160)]:
            fi = torch.rand(1, s, 10, vh, vw)
            fi[:, :, 7:10] *= 4.0
            ti = torch.rand(1, C_T, vh, vw)
            with torch.no_grad():
                ref_out, ref_tmp = model(fi, ti)
            got = sess.run([OUT_COLOR, OUT_TEMPORAL],
                           {IN_FRAME: fi.numpy(), IN_TEMPORAL: ti.numpy()})
            d_out = np.abs(got[0] - ref_out.numpy())
            rel_out = d_out.mean() / max(np.abs(ref_out.numpy()).mean(), 1e-12)
            d_tmp = np.abs(got[1] - ref_tmp.numpy())
            rel_tmp = d_tmp.mean() / max(np.abs(ref_tmp.numpy()).mean(), 1e-12)
            print(f"verify fp16 {vw}x{vh}: output mean rel {rel_out:.3e} "
                  f"max |delta| {d_out.max():.3e}, temporal mean rel {rel_tmp:.3e}")
            if rel_out > 2e-2 or d_out.max() > 0.1 or rel_tmp > 5e-2:
                sys.exit("fp16/PyTorch mismatch above the fp16 gates "
                         "(output mean rel 2e-2 / max 0.1, temporal mean rel 5e-2)")

    if args.golden:
        # (after --fp16 the pair reflects the shipped fp16 graph)
        np.savez(args.golden,
                 frame_input=fi.numpy(), temporal_warped=ti.numpy(),
                 output=got[0], temporal_out=got[1])
        print(f"wrote golden pair {args.golden}")

    print(f"\nRust consts (src/nppd.rs): C_T = {C_T}, io = "
          f"[{IN_FRAME!r}, {IN_TEMPORAL!r}, {OUT_COLOR!r}, {OUT_TEMPORAL!r}], S = {s}")
    print("packer contract: ch0 log-depth ln(1+1/d_euclid) sky=0 | ch1-3 camera-space "
          "normal (n.fwd, n.right, n.up) | ch4-6 linear diffuse albedo | ch7-9 linear HDR color")


if __name__ == "__main__":
    main()
