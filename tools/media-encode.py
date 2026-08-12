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
     decimates, scales to 1100 and crops to a 2.39:1 letterbox. That variant
     lived only in a shell history.

`foliage.webp` IS the program's own printed webp command (its render res is
already 1280x536, so the `scale=1280` is a no-op) — reproduce it with
`--cinematic-encode`, or with `clip` below, which is the same command.

EACH SHOT SHIPS TWICE, and the split is the point. The committed **WebP**
auto-loops, so it works for every reader — offline, on a fork, without AV1
decode. The **AV1** is the 60 fps quality asset, LINKED from GitHub Pages (see `do_av1`
for why a README cannot embed a player and why raw/release hosting cannot serve
one). The README puts that link beside each loop.

THE EXACT INVOCATIONS FOR THE COMMITTED SET
-------------------------------------------
    # the 7 isles, hero, clouds A/B, UI, overlay pair  (PNG -> lossless WebP)
    python3 tools/media-encode.py still \\
        capture-islands/island-02-sponza-0830/island-02-sponza-0830.png \\
        docs/media/islands/02-sponza.webp

    # the foliage loop: 30 fps from the 60 fps render (every 2nd frame)
    python3 tools/media-encode.py clip capture-foliage60/foliage/frames \\
        docs/media/foliage.webp --fps 30 --width 1280

    # the lap loop: 60 fps render -> every 2nd frame at 30 fps, letterboxed
    python3 tools/media-encode.py tour capture-tour/tour/frames docs/media/tour.webp

    # the 60 fps AV1s the README links to
    python3 tools/media-encode.py av1 capture-tour/tour/frames      pages/media/tour-av1.mp4
    python3 tools/media-encode.py av1 capture-foliage60/foliage/frames \\
        pages/media/foliage-av1.mp4 --width 1280 --height 0

    # then publish them + verify the serving headers
    python3 tools/media-encode.py pages pages/media/tour-av1.mp4 pages/media/foliage-av1.mp4

30 fps is a FLOOR for anything shipped (`MIN_PRODUCTION_FPS`) and the tool
refuses to go below it rather than quietly obeying — the tour shipped at 20 fps
for months because nothing said no.

The B70 hybrid-vs-DXR captures are desktop screen grabs, not `--cinematic`
output (the title-bar fps receipt is not in any render target), and they go
through `still` like everything else.

Requires Pillow for the stills and ffmpeg on PATH for the clips.
"""

import argparse
import os
import subprocess
import sys

# Mirror src/cinematic.rs. Kept in sync by hand and by the comments there; the
# Rust self_test pins the floors.
INLINE_WEBP_Q = 95
MIN_PRODUCTION_FPS = 30      # a shipped animation may not go below this
INLINE_AV1_CRF = 18          # libsvtav1 CRF; VMAF 97.64 vs a lossless 10-bit ref

# Low-amplitude noise added before the 8-bit conversion, as DITHER.
#
# It is here because ffmpeg's real dither controls cannot reach this conversion,
# which took four attempts to establish: zscale has NO output-pixel-format
# option (its `f` is the scaling FILTER), so the 8-bit reduction is always done
# by an auto-inserted swscale and zscale's own `dither=` never applies -- proven
# by byte-identical md5 across dither=none/ordered/error_diffusion in three
# different chain arrangements. And `-sws_dither error_diffusion` is rejected by
# this build wherever it is placed (encoder open fails -22), while `auto`
# measures identical to `none`.
#
# Noise works, and the numbers are the justification: on the tour's sky the
# widest flat plateau goes 14.5% of the region -> 6.9% at alls=2 (+27% bytes)
# and 5.0% at alls=4 (+110%). 2 is the knee. The AV1 does not need this at all
# -- it is 10-bit and never quantizes to 8.
BAND_NOISE = 2


def require_fps(fps):
    """A shipped animation may not go below the floor. Refuse rather than obey:
    the tour shipped at 20 fps for months because nothing said no."""
    if fps < MIN_PRODUCTION_FPS:
        sys.exit(f"media-encode: {fps} fps is below MIN_PRODUCTION_FPS "
                 f"({MIN_PRODUCTION_FPS}) -- shipped animation may not stutter")


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
# (cinematic.rs), npl = HDR_MASTER_NITS / 5. Ends at 8-bit rgb24 for the WebP,
# which is all libwebp lossy accepts (yuv420p/yuva420p -- there is no lossy
# 4:4:4 mode, so the WebP is chroma-subsampled by construction).
TONEMAP = ("zscale=t=linear:npl=200,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=full,format=rgb24")

# The AV1's tone map stops at 10 BITS and never touches 8 -- that is the sky
# banding fix. The renderer's own gradient is smooth (786 distinct 16-bit levels
# over 972 rows); the 8-bit WebP keeps ~50 over 207. At 10 bits the video
# carries about twice the WebP's gradient precision, and smooth gradients
# compress better at 10 bits, so it is not even a size cost.
TONEMAP10 = ("zscale=t=linear:npl=200,format=gbrpf32le,zscale=p=bt709,"
             "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=limited,"
             "format=yuv420p10le")


def do_clip(args):
    """The program's own printed webp command, with the quality constant."""
    require_fps(args.fps)
    pat = os.path.join(args.frames, "f_%05d.png")
    pq = is_pq16(args.frames)
    print(f"clip: {'16-bit PQ frames — tone-mapping to SDR' if pq else '8-bit SDR frames'}")
    chain = [TONEMAP] if pq else []
    chain.append(f"scale={args.width}:-2:flags=lanczos")
    if args.noise:
        chain.append(f"noise=alls={args.noise}:allf=t+u")   # dither; see BAND_NOISE
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
    require_fps(args.fps)
    pat = os.path.join(args.frames, "f_%05d.png")
    pq = is_pq16(args.frames)
    print(f"tour: {'16-bit PQ frames — tone-mapping to SDR' if pq else '8-bit SDR frames'}")
    # `select` leads so two thirds of the frames are dropped before anything
    # pays for tone-mapping or scaling.
    parts = [f"select='not(mod(n\\,{args.step}))'", f"setpts=N/{args.fps}/TB"]
    if pq:
        parts.append(TONEMAP)
    parts += [f"scale={args.width}:-2:flags=lanczos", f"crop={args.width}:{args.height}"]
    if args.noise:
        parts.append(f"noise=alls={args.noise}:allf=t+u")   # dither; see BAND_NOISE
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


def do_av1(args):
    """The 60 fps quality asset: 10-bit AV1, LINKED from GitHub Pages.

    Two measured reasons it is a link and not an embedded player:

    1. A README cannot embed one. GitHub's README renderer STRIPS <video>;
       verified against the live page through repos/.../readme with
       Accept: application/vnd.github.html, which finds zero <video> elements
       while the <img> beside it survives. THE TRAP: the standalone /markdown
       API is more permissive and returns the element with its src intact, so
       testing there says the opposite of what shipping does. A plain link is
       what works -- the same thing quinlight-audio does for its .m4a clips.
    2. The bytes still have to be playable. raw.githubusercontent.com serves a
       committed file as application/octet-stream WITH
       X-Content-Type-Options: nosniff, which forbids the browser decoding it,
       and a release asset is an `attachment` behind a signed URL that expires
       in an hour. Pages serves a real video/mp4 with Accept-Ranges: bytes on a
       permanent URL, so the link opens in the browser's player and seeks.
    """
    require_fps(args.fps)
    pat = os.path.join(args.frames, "f_%05d.png")
    pq = is_pq16(args.frames)
    print(f"av1: {'16-bit PQ frames — tone-mapping to 10-bit SDR' if pq else '8-bit SDR frames'}")
    parts = []
    if args.step > 1:
        parts += [f"select='not(mod(n\\,{args.step}))'", f"setpts=N/{args.fps}/TB"]
    if pq:
        parts.append(TONEMAP10)
    parts.append(f"scale={args.width}:-2:flags=lanczos")
    if args.height:
        parts.append(f"crop={args.width}:{args.height}")
    argv = (PQ_IN if pq else []) + [
        "-y", "-framerate", str(args.src_fps), "-start_number", "1", "-i", pat,
        "-vf", ",".join(parts),
    ]
    if args.step > 1:
        argv += ["-r", str(args.fps)]
    argv += [
        "-c:v", "libsvtav1", "-preset", str(args.preset), "-crf", str(args.crf),
        "-svtav1-params", "tune=0", "-pix_fmt", "yuv420p10le",
        "-movflags", "+faststart", args.dst,
    ]
    run_ffmpeg(argv, args.dst)


def do_pages(args):
    """Stage the videos onto the orphan gh-pages branch Pages serves.

    An orphan branch carrying ONLY media, deliberately not `/docs` on master:
    Pages publishes everything in its source tree, and docs/ holds 18 MB of
    third-party academic PDFs plus seven revisions of a vendor brief. A branch
    that contains nothing but videos cannot leak them, whatever anyone adds to
    docs/ later.
    """
    files = args.files
    for f in files:
        if not os.path.isfile(f):
            sys.exit(f"media-encode: no such file: {f}")
    print("Run these to publish (review before pushing):\n")
    print("  git switch --orphan gh-pages   # first time only")
    print("  # .mp4 is gitignored unanchored, so force the add:")
    for f in files:
        print(f"  git add -f {f}")
    print("  echo > .nojekyll && git add .nojekyll")
    print("  git commit -m 'pages: media for the README video links'")
    print("  git push -u origin gh-pages")
    print("  git switch master")
    print("\nThen verify the serving headers -- this is the whole reason Pages was")
    print("chosen, so do not assume it:")
    for f in files:
        print(f"  curl -sI https://{args.owner}.github.io/{args.repo}/{os.path.basename(f)}"
              " | grep -iE 'content-type|accept-ranges|nosniff'")
    print("\n  Expect: Content-Type: video/mp4, Accept-Ranges: bytes, and NO nosniff.")


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
    p.add_argument("--noise", type=int, default=BAND_NOISE,
                   help="dither amplitude; 0 disables (see BAND_NOISE)")
    p.set_defaults(fn=do_clip)

    p = sub.add_parser("tour", help="the letterboxed, decimated lap clip")
    p.add_argument("frames")
    p.add_argument("dst")
    p.add_argument("--src-fps", type=int, default=60)
    # 30, not the old 20: MIN_PRODUCTION_FPS. Every 2nd source frame, not 3rd.
    p.add_argument("--fps", type=int, default=MIN_PRODUCTION_FPS)
    p.add_argument("--step", type=int, default=2, help="keep every Nth frame")
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=460, help="letterbox crop height")
    p.add_argument("--quality", type=int, default=INLINE_WEBP_Q)
    p.add_argument("--noise", type=int, default=BAND_NOISE,
                   help="dither amplitude; 0 disables (see BAND_NOISE)")
    p.set_defaults(fn=do_tour)

    p = sub.add_parser("av1", help="10-bit AV1 for the Pages-hosted 60 fps link")
    p.add_argument("frames")
    p.add_argument("dst")
    p.add_argument("--src-fps", type=int, default=60)
    p.add_argument("--fps", type=int, default=60, help="output rate; 60 for the video")
    p.add_argument("--step", type=int, default=1, help="1 = keep every frame")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=804,
                   help="letterbox crop height; 0 to keep the full frame")
    p.add_argument("--crf", type=int, default=INLINE_AV1_CRF)
    p.add_argument("--preset", type=int, default=3, help="libsvtav1 preset; lower = smaller")
    p.set_defaults(fn=do_av1)

    p = sub.add_parser("pages", help="print the gh-pages publish + header-verify steps")
    p.add_argument("files", nargs="+")
    p.add_argument("--owner", default="fechols")
    p.add_argument("--repo", default="frustracer")
    p.set_defaults(fn=do_pages)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
