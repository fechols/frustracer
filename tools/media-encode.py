#!/usr/bin/env python3
"""Turn a `--cinematic` capture into the committed `docs/media/` assets.

WHY THIS EXISTS. `--cinematic` writes PNG frames, and the program prints the
ffmpeg lines for the *sequence* encodes it knows about — but two hops in the
committed media set were recorded nowhere in the repo:

  1. STILLS.  Every still on the README is a WebP; nothing named the tool or
     the setting that made it. They are now LOSSLESS WebP, which is the whole
     point of the page: a ray tracer's output should not be shown through a
     compressor that invents its own detail. Lossless is not a nicety here —
     the grain, cavity shadows and contact shadows the renderer is being
     advertised for are exactly the frequencies a lossy encoder rings on.

  2. THE INLINE TOUR CLIP.  `tour.webp` is NOT what the program prints. The
     printed command encodes every frame at the full width; the committed clip
     selects every third frame (60 fps -> 20 fps), scales to 1100 and crops to
     a 2.39:1 letterbox. That variant lived only in a shell history.

`foliage.webp` IS the program's own printed webp command (its render res is
already 1280x536, so the `scale=1280` is a no-op) — reproduce it with
`--cinematic-encode`, or with `clip` below, which is the same command.

THE EXACT INVOCATIONS FOR THE COMMITTED SET
-------------------------------------------
    # the 7 isles, hero, clouds A/B, UI, overlay pair  (PNG -> lossless WebP)
    python3 tools/media-encode.py still \\
        capture-islands/island-02-sponza-0830/island-02-sponza-0830.png \\
        docs/media/islands/02-sponza.webp

    # the foliage clip: 120 frames at 30 fps, 1280 wide, no crop
    python3 tools/media-encode.py clip capture-foliage/foliage/frames \\
                                       docs/media/foliage.webp --fps 30 --width 1280

    # the lap: 1200 frames at 60 fps -> every 3rd frame at 20 fps, letterboxed
    python3 tools/media-encode.py tour capture-tour/tour/frames docs/media/tour.webp

The B70 hybrid-vs-DXR captures are desktop screen grabs, not `--cinematic`
output (the title-bar fps receipt is not in any render target), and they go
through `still` like everything else.

Requires Pillow for the stills and ffmpeg on PATH for the clips.
"""

import argparse
import os
import subprocess
import sys

# Mirrors src/cinematic.rs::INLINE_WEBP_Q. Kept in sync by hand and by the
# comment there; the Rust self_test pins the floor.
INLINE_WEBP_Q = 95


def human(n):
    return f"{n / 1_048_576:.2f} MB" if n >= 1_048_576 else f"{n / 1024:.0f} KB"


def do_still(args):
    try:
        from PIL import Image
    except ImportError:
        sys.exit("media-encode: Pillow is required for stills (pip install pillow)")

    src, dst = args.src, args.dst
    if not os.path.isfile(src):
        sys.exit(f"media-encode: no such file: {src}")
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)

    im = Image.open(src)
    # RGB, not RGBA: a screen grab arrives with an opaque alpha channel that
    # costs bytes and says nothing. Anything genuinely transparent would be a
    # bug in the capture, not something to preserve.
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    # `exact=True` matters even at lossless: without it libwebp is free to
    # rewrite RGB under fully-transparent pixels. Nothing here is transparent,
    # but the flag costs nothing and the repo already relies on it for scene
    # textures, where it is load-bearing at cutout edges.
    im.save(dst, "WEBP", lossless=True, exact=True, method=6)

    a, b = os.path.getsize(src), os.path.getsize(dst)
    print(f"still  {im.size[0]}x{im.size[1]}  {human(a)} PNG -> {human(b)} lossless WebP  {dst}")


def run_ffmpeg(argv, dst):
    print("  ffmpeg " + " ".join(a if " " not in a else f'"{a}"' for a in argv))
    r = subprocess.run(["ffmpeg"] + argv)
    if r.returncode != 0:
        sys.exit(f"media-encode: ffmpeg failed ({r.returncode})")
    print(f"  -> {dst}  {human(os.path.getsize(dst))}")


# Under --cinematic-hdr a SEQUENCE writes 16-bit PQ / Rec.2020 frames to the
# very path an SDR run would use (main.rs: `if !hdr_frames` skips the SDR PNG,
# and save_png16 takes over the same name). Encoding those as if they were sRGB
# gives a washed-out clip, silently -- so detect the wire format.
#
# READ THE PNG HEADER, NOT PILLOW. Pillow silently truncates a 16-bit PNG to
# 8 bits on load and then reports `mode=RGB, bits=8` for it, so a Pillow-based
# probe fails OPEN -- it says "this is SDR" about the exact frames that are
# not, which is the washed-out clip it was written to prevent. IHDR puts the
# bit depth at byte 24, immediately after the 8-byte signature, the 4-byte
# length, the "IHDR" tag and the 4+4-byte dimensions.
def is_pq16(frames):
    for f in sorted(os.listdir(frames)):
        if not f.endswith(".png"):
            continue
        with open(os.path.join(frames, f), "rb") as fh:
            head = fh.read(26)
        if len(head) < 26 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
            return False
        return head[24] == 16
    return False


# The input colour tags MUST precede -i. A PNG carries no transfer/primaries
# metadata, so a bare zscale=t=linear has nothing to convert FROM and the graph
# dies with "code 3074: no path between colorspaces" -- an ffmpeg library
# error, so it surfaces as a failed encode rather than anything naming the
# cause. Setting tin=/pin= ON zscale does NOT fix it.
PQ_IN = ["-color_primaries", "bt2020", "-color_trc", "smpte2084",
         "-colorspace", "bt2020nc"]

# The same tone-map the program prints for its own SDR sibling encode
# (cinematic.rs), npl = HDR_MASTER_NITS / 5.
TONEMAP = ("zscale=t=linear:npl=200,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=full,format=rgb24")


def do_clip(args):
    """The program's own printed webp command, with the quality constant."""
    pat = os.path.join(args.frames, "f_%05d.png")
    pq = is_pq16(args.frames)
    print(f"clip: {'16-bit PQ frames — tone-mapping to SDR' if pq else '8-bit SDR frames'}")
    chain = [TONEMAP] if pq else []
    chain.append(f"scale={args.width}:-2:flags=lanczos")
    run_ffmpeg(
        (PQ_IN if pq else []) +
        [
            "-y", "-framerate", str(args.fps), "-start_number", "0", "-i", pat,
            "-vf", ",".join(chain),
            "-c:v", "libwebp", "-lossless", "0", "-q:v", str(args.quality),
            "-compression_level", "6", "-loop", "0", "-an",
            args.dst,
        ],
        args.dst,
    )


def do_tour(args):
    """The letterboxed, frame-decimated inline lap.

    Three details that each cost something to rediscover:

    * START AT FRAME 1, not 0. Frame 0 is the only frame rendered from a hard
      history reset on the reconstruction arm, so in a clip that loops forever
      it shows its under-converged self once per lap.
    * `-r` fights `-fps_mode passthrough`, so the retime is `setpts=N/<fps>/TB`
      alongside `-r <fps>` rather than `-r` alone.
    * `select` comes FIRST in the chain, so two thirds of the frames are
      dropped before anything pays for scaling.
    """
    pat = os.path.join(args.frames, "f_%05d.png")
    pq = is_pq16(args.frames)
    print(f"tour: {'16-bit PQ frames — tone-mapping to SDR' if pq else '8-bit SDR frames'}")
    # `select` leads so two thirds of the frames are dropped before anything
    # pays for tone-mapping or scaling.
    parts = [f"select='not(mod(n\\,{args.step}))'", f"setpts=N/{args.fps}/TB"]
    if pq:
        parts.append(TONEMAP)
    parts += [f"scale={args.width}:-2:flags=lanczos", f"crop={args.width}:{args.height}"]
    run_ffmpeg(
        (PQ_IN if pq else []) +
        [
            "-y", "-framerate", str(args.src_fps), "-start_number", "1", "-i", pat,
            "-vf", ",".join(parts), "-r", str(args.fps),
            "-c:v", "libwebp", "-lossless", "0", "-q:v", str(args.quality),
            "-compression_level", "6", "-loop", "0", "-an",
            args.dst,
        ],
        args.dst,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("still", help="PNG -> lossless WebP")
    p.add_argument("src")
    p.add_argument("dst")
    p.set_defaults(fn=do_still)

    p = sub.add_parser("clip", help="frame dir -> inline animated WebP")
    p.add_argument("frames")
    p.add_argument("dst")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--quality", type=int, default=INLINE_WEBP_Q)
    p.set_defaults(fn=do_clip)

    p = sub.add_parser("tour", help="the letterboxed, decimated lap clip")
    p.add_argument("frames")
    p.add_argument("dst")
    p.add_argument("--src-fps", type=int, default=60)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--step", type=int, default=3, help="keep every Nth frame")
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=460, help="letterbox crop height")
    p.add_argument("--quality", type=int, default=INLINE_WEBP_Q)
    p.set_defaults(fn=do_tour)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
