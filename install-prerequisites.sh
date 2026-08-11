#!/usr/bin/env bash
# ===========================================================================
#  frustracer — runtime SDK installer (the UNIX twin of
#  install-prerequisites.bat: same pins, same destinations, same markers).
#  Runs on Linux and macOS; the split from the .bat is the SHELL LANGUAGE, not
#  the platform, which is why there is no third script — the per-platform
#  surface is three constants and one component arm, and a third copy of the
#  fetch/extract/report machinery would drift exactly as the header below warns
#  the .bat and this one can.
#
#  MOST OF WHAT IT FETCHES IS WINDOWS BINARIES, ON PURPOSE. frustracer is a
#  D3D12 renderer, so those components do not run on Linux. The point is that a
#  checkout that LIVES on Linux — a shared or dual-boot drive, a WSL working
#  copy, a cross-build staging tree, a fileserver clone — can have its SDKs/
#  tree populated from here instead of rebooting to do it. Every file lands in
#  exactly the directory the matching default in src/cli.rs points at, so the
#  Windows session that consumes them needs no flags.
#
#  TWO COMPONENTS ARE HOST-NATIVE and exist for the Vulkan backend port:
#  `dxc` additionally fetches the Linux DXC (SPIR-V codegen — verified present
#  in the official build, which is the port's whole toolchain premise) beside
#  the Windows one at the SAME pin, and `spirv` fetches SPIRV-Tools for
#  validating what DXC emits. Both are no-ops for a Windows session. They are
#  in `all` because a checkout that cannot compile a shader cannot make
#  progress on the port, and finding that out at the first kernel is worse than
#  paying the download here.
#
#  ON macOS THE HOST-NATIVE DXC HALF IS NOT AVAILABLE, and the reason is the
#  pin rather than effort: Microsoft publishes exactly three assets at DXC_TAG
#  (the Windows zip, a linux x86_64 tarball, and a PDB zip) — no macOS build and
#  no arm64 build of any kind. A community binary would be, by construction, not
#  from that tag, which breaks the invariant DXC_LINUX_TGZ's own comment calls
#  out as one no gate could see. So the half skips with a loud line naming the
#  route that preserves the pin (a source build at DXC_TAG, the shape `nrd`
#  already uses on the other side), and `spirv` — which DOES publish a macOS
#  prebuilt — installs normally.
#
#  Building NEVER needs any of this: the MIT headers the shims compile against
#  (FidelityFX) are committed, and every SDK below is LoadLibrary'd at runtime,
#  so `cargo build --release` and every DLL-free `--check*` gate work on a bare
#  checkout.
#
#  DLSS is the one feature no installer can fetch: ray reconstruction and frame
#  generation both ride the raw-NGX shims, which need the (non-redistributable,
#  non-fetchable) DLSS SDK present at BUILD time — FRUSTRACER_DLSS_SDK points at
#  it; build.rs stages the snippet DLLs.
#
#  The binaries are license-restricted (that is why they are gitignored and not
#  committed) — this downloads them from each vendor's own release page onto
#  YOUR machine. Nothing here redistributes them.
#
#  DIFFERENCES FROM THE .bat, each forced by the platform:
#    * nrd is skipped. NRD ITSELF IS NOT WINDOWS-ONLY — it is an API-agnostic
#      library that builds fine on Linux — but the artifact this renderer loads
#      is NRD.dll: a PE with DXIL shaders EMBEDDED in it (the .bat configures
#      NRD_EMBEDS_DXIL_SHADERS=ON, DXBC/SPIRV OFF). A Linux build emits
#      libNRD.so carrying SPIRV, which a D3D12 session cannot load, and a
#      mingw cross-build would still hand D3D12 UNSIGNED DXIL, because the
#      signer is dxil.dll — Windows-only, and the `(validator)` row in the
#      checklist below. So this degrades exactly as the .bat does when it finds
#      no Visual Studio: a note by default, [x] and a failure exit only when nrd
#      was asked for BY NAME. Build it from Windows: `install-prerequisites.bat
#      nrd`.
#    * flags are --force / --clean (the .bat's /force and /clean also work).
#    * unzip (or bsdtar) does the extracting — GNU tar cannot read a zip, which
#      is the mirror image of the .bat's reason for calling System32's bsdtar by
#      absolute path.
#
#  Usage:
#    ./install-prerequisites.sh                 all components
#    ./install-prerequisites.sh dxc fsr         only those
#    ./install-prerequisites.sh all --force     re-download and re-extract
#    ./install-prerequisites.sh --clean         delete the download cache
#
#  Components: dxc fsr xess nppd oidn pix nrd spirv
#  Needs: bash 3.2+, curl (or wget), unzip (or bsdtar). THE BASH FLOOR IS 3.2
#  BECAUSE macOS SHIPS 3.2 (GPL3) and `#!/usr/bin/env bash` finds it, so the
#  bash-4 conveniences are out on purpose: no ${x,,}, no mapfile/readarray, and
#  no GNU-only utility flags (`grep -P`, `readlink -f`) — BSD userland has to
#  run this too. That is not a style preference; ${x,,} under 3.2 aborts the
#  enclosing `case` without matching any branch, and with no `set -e` the arg
#  parser then fell THROUGH to an empty SEL, i.e. "install everything" in answer
#  to `--help`. An arg parser that fails open is worse than one that exits.
#  Measured on a full run:
#  ~510 MB of downloads, 549 MB in SDKs/ after extraction. (The .bat quotes
#  ~2 GB because its nrd component also unpacks and builds an NRD source tree,
#  which this one does not.)
# ===========================================================================
set -uo pipefail

# `readlink -f` is GNU; BSD readlink only grew it in recent macOS, so resolving
# the directory by cd'ing there is the portable spelling and drops a macOS
# version floor the rest of the script does not have. `pwd -P` does the symlink
# resolution the -f was wanted for.
cd "$(dirname "$0")" || exit 2
ROOT="$(pwd -P)"
cd "$ROOT" || exit 2
SDKS="$ROOT/SDKs"
CACHE="${TMPDIR:-/tmp}/frustracer-prereqs"

# --- host ----------------------------------------------------------------
#  The host decides exactly two things: which SPIRV-Tools prebuilt to fetch and
#  whether the host-native DXC half exists at all. Everything else is Windows
#  payload that lands identically either way — which IS the point of the script
#  (see the header): populating SDKs/ from the machine the checkout lives on.
case "$(uname -s)" in
    Darwin) OS=macos ;;
    Linux) OS=linux ;;
    *)
        # Name it and continue rather than exiting: the Windows components are
        # host-agnostic downloads, so a BSD or Solaris checkout still gets a
        # complete SDKs/ tree. Only the two host-native rows stand down.
        OS=other
        echo "[i] unrecognized host \"$(uname -s)\" — fetching the Windows"
        echo "    components only; the host-native dxc/spirv halves stand down."
        ;;
esac

# lower <str> — ${x,,} is bash 4 and macOS ships 3.2. tr is everywhere.
lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

# read_lines <var> — mapfile/readarray are bash 4 too. Reads stdin into the
# named array. The trailing element guard matters: a `find` that matches
# nothing yields one EMPTY line under a plain read loop, which would then look
# like one result to the callers below, both of which count matches.
read_lines() {
    local __n="$1" __line
    eval "$__n=()"
    while IFS= read -r __line; do
        [[ -n $__line ]] || continue
        eval "$__n+=(\"\$__line\")"
    done
}

# --- pinned versions ------------------------------------------------------
#  Keep these in lockstep with install-prerequisites.bat — the two scripts
#  populate the same tree and a drift between them is a version skew nobody
#  would look for. Bump deliberately: the ORT/DirectML pair below is the one
#  CLAUDE.md pins as verified (an old DirectML under a new ORT fails the NPPD
#  U-Net's Resize node at run time), and XeSS stays on the 2.x line the code was
#  written and gated against (3.0.1 exists and is untested here).
OIDN_VER=2.5.0
XESS_VER=2.1.1
DXC_TAG=v1.9.2602.24
DXC_ZIP=dxc_2026_05_27.zip
#  The Linux DXC rides the SAME release tag, so the compiler that emits DXIL for
#  the D3D12 backend and the one that emits SPIR-V for Vulkan can never drift to
#  different HLSL front ends — which would be a difference no gate could see,
#  since the two backends compile the identical concatenated source. NOTE the
#  tarball's own date differs from the zip's by a day; that is upstream's
#  asset naming within one release, not a skew.
DXC_LINUX_TGZ=linux_dxc_2026_05_26.x86_64.tar.gz
#  SPIRV-Tools: the prebuilt install.tgz from the SPIRV-Tools CI bucket, whose
#  latest link is a redirect stub we resolve at run time rather than pinning a
#  build number that expires. `spirv-val` is the gate on everything DXC emits.
#  The bucket publishes a macOS clang build beside the Linux one, so this is the
#  one host-native component that needs nothing but the right URL — with the
#  caveat that the macOS build is x86_64 only (Rosetta on Apple silicon; see
#  do_spirv, which checks rather than assumes).
SPIRV_BADGE_BASE=https://storage.googleapis.com/spirv-tools/badges/build_link
FFX_VER=2.3.0
ORT_VER=1.24.4
DML_VER=1.15.4
PIX_VER=1.0.240308001
#  NRD is pinned BOTH here and in src/nrd.rs (the transcribed structs + runtime
#  GetLibraryDesc gate) — move them together or --nrd sheds loudly. Unused on
#  this platform (see the header), kept so the two installers stay comparable.
NRD_TAG=v4.17.3

# --- tools ---------------------------------------------------------------
#  One downloader and one extractor, chosen once: a minimal container tends to
#  have exactly one of each pair, and picking per call would mean two code paths
#  to keep honest.
DL=
command -v curl >/dev/null 2>&1 && DL=curl
[[ -z $DL ]] && command -v wget >/dev/null 2>&1 && DL=wget
if [[ -z $DL ]]; then
    echo "[x] need curl or wget on PATH"
    exit 2
fi
UNZIP=
command -v unzip >/dev/null 2>&1 && UNZIP=unzip
[[ -z $UNZIP ]] && command -v bsdtar >/dev/null 2>&1 && UNZIP=bsdtar
if [[ -z $UNZIP ]]; then
    echo "[x] need unzip or bsdtar on PATH (GNU tar cannot read a zip)"
    echo "    apt install unzip | dnf install unzip | pacman -S unzip"
    echo "    (macOS ships bsdtar as /usr/bin/tar's sibling — if this fires there,"
    echo "     PATH is unusual)"
    exit 2
fi

# --- args ----------------------------------------------------------------
FORCE=
NRD_EXPLICIT=
SEL=()
ALL=(dxc fsr xess nppd oidn pix nrd spirv)
for a in "$@"; do
    case "$(lower "$a")" in
        --force | /force | -f) FORCE=1 ;;
        --clean | /clean)
            echo "removing $CACHE"
            rm -rf "$CACHE"
            exit 0
            ;;
        --help | -h | /?)
            # print the header block itself — awk to the first non-comment line,
            # never a line range, which drifts silently as the header is edited
            awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
            exit 0
            ;;
        all) SEL=() ;;
        # `dlss` was a valid component in the Streamline era; say why it is gone
        # instead of silently installing nothing.
        dlss)
            echo "[x] dlss is no longer fetchable: DLSS builds against the NDA DLSS"
            echo "    SDK at BUILD time (set FRUSTRACER_DLSS_SDK; see the header)."
            exit 2
            ;;
        dxc | fsr | xess | nppd | oidn | pix | spirv)
            SEL+=("$(lower "$a")")
            ;;
        nrd)
            SEL+=(nrd)
            NRD_EXPLICIT=1
            ;;
        *)
            echo "[x] unknown component \"$a\" (valid: ${ALL[*]} all)"
            exit 2
            ;;
    esac
done
((${#SEL[@]})) || SEL=("${ALL[@]}")

want() {
    local c
    for c in "${SEL[@]}"; do [[ $c == "$1" ]] && return 0; done
    return 1
}

mkdir -p "$CACHE" || exit 2

# Failures are collected and NAMED rather than counted: "one or more downloads
# failed" is a wrong diagnosis for the component that had nothing to download
# (nrd), and a wrong diagnosis costs more than a missing one.
FAILED=
FAILED_LIST=()
CUR=
fail() {
    FAILED=1
    local c
    for c in ${FAILED_LIST[@]+"${FAILED_LIST[@]}"}; do [[ $c == "$CUR" ]] && return 0; done
    FAILED_LIST+=("$CUR")
}

echo
echo "frustracer prerequisites -> $SDKS"
echo "components: ${SEL[*]}"
echo "cache:      $CACHE   (reused across runs; --clean to drop)"
echo

# =========================== helpers ======================================

# skip <marker> <name> — already installed and not --force?
skip() {
    [[ -n $FORCE ]] && return 1
    if [[ -e $1 ]]; then
        echo "[=] $2 already installed"
        return 0
    fi
    return 1
}

# fetch <cachefile> <url>
fetch() {
    local out="$CACHE/$1"
    if [[ -z $FORCE && -s $out ]]; then
        echo "[.] $1 cached"
        return 0
    fi
    echo "[>] downloading $1"
    local rc=0
    if [[ $DL == curl ]]; then
        curl -L --fail --retry 3 --retry-delay 2 --progress-bar -o "$out.part" "$2" || rc=$?
    else
        # -q --show-progress = curl's --progress-bar: the transfer log alone,
        # not wget's full resolve/connect/redirect narration per archive
        wget -q --show-progress --tries=3 --waitretry=2 -O "$out.part" "$2" || rc=$?
    fi
    if ((rc != 0)); then
        echo "    [x] download FAILED: $2"
        rm -f "$out.part"
        fail
        return 1
    fi
    mv -f "$out.part" "$out"
}

# unzip_to <archive> <destdir> — the one extractor call site.
#  DISPATCHES ON THE ARCHIVE'S MAGIC BYTES, never its name. The Windows drops
#  are zips and the Linux ones (DXC, SPIRV-Tools) are gzipped tars, and upstream
#  naming is not a reliable signal either way — this repo already carries
#  .zip-named tarballs. Reading the first two bytes is the only test that cannot
#  be wrong, and getting it wrong costs a confusing "not a zipfile" against a
#  perfectly good download.
#  unzip exits 1 for warnings (skipped/renamed entries) and >=2 for real
#  errors, so only >=2 is a failure; bsdtar and tar are plain 0/non-0.
unzip_to() {
    local rc=0 magic
    magic=$(head -c2 "$1" | od -An -tx1 | tr -d ' \n')
    if [[ $magic == 1f8b ]]; then
        # gzip — GNU tar handles it; bsdtar equally if that is what we have
        tar -xzf "$1" -C "$2" || rc=$?
        return $rc
    fi
    if [[ $UNZIP == unzip ]]; then
        unzip -qq -o "$1" -d "$2" || rc=$?
        ((rc == 1)) && rc=0
    else
        bsdtar -xf "$1" -C "$2" || rc=$?
    fi
    return $rc
}

# extract <cachefile> <destdir> [strip1]
#  strip1 hoists the archive's single top-level directory away, the equivalent
#  of bsdtar's --strip-components=1 (OIDN is the one archive with a
#  version-stamped wrapper dir). It stages and then MERGES with `cp -a src/. dst`
#  rather than `mv`: a plain `mv stage/*/* dest/` fails the moment a
#  destination subdirectory already exists, which is exactly the --force
#  re-extract case.
extract() {
    local zip="$CACHE/$1" dest="$2" strip="${3:-}"
    mkdir -p "$dest"
    echo "    [+] extracting $1 -> ${dest#"$ROOT"/}"
    if [[ -n $strip ]]; then
        local stage="$CACHE/stage-${1%%.*}"
        rm -rf "$stage"
        mkdir -p "$stage"
        if ! unzip_to "$zip" "$stage"; then
            echo "    [x] extract FAILED (corrupt download? rerun with --force)"
            fail
            return 1
        fi
        local inner=()
        read_lines inner < <(find "$stage" -mindepth 1 -maxdepth 1)
        if ((${#inner[@]} != 1)) || [[ ! -d ${inner[0]} ]]; then
            echo "    [x] $1: expected ONE top-level directory to strip, found ${#inner[@]}"
            fail
            return 1
        fi
        cp -a "${inner[0]}/." "$dest/" || {
            fail
            return 1
        }
        rm -rf "$stage"
        return 0
    fi
    if ! unzip_to "$zip" "$dest"; then
        echo "    [x] extract FAILED (corrupt download? rerun with --force)"
        fail
        return 1
    fi
}

# grab <srcdir> <relpath> <destdir> — copy one file out of an extracted
#  archive, tolerating the case the archive actually used. Windows normalizes
#  case and Linux does not, so a literal path out of a Windows-authored zip is
#  a coin flip nobody can see failing on the other platform; this resolves it
#  once and says so when the answer is ambiguous.
grab() {
    local src="$1" rel="$2" dst="$3" hit=()
    if [[ -f $src/$rel ]]; then
        cp -f "$src/$rel" "$dst/" && return 0
        return 1
    fi
    read_lines hit < <(find "$src" -ipath "*/${rel}" -type f)
    if ((${#hit[@]} == 1)); then
        cp -f "${hit[0]}" "$dst/" && return 0
        return 1
    fi
    echo "    [x] $rel: ${#hit[@]} matches under ${src#"$CACHE"/} (expected 1)"
    return 1
}

# check <label> <path>
check() {
    if [[ -e $2 ]]; then
        echo " [ok] $1"
    else
        echo " [--] $1  MISSING"
    fi
}

# =========================== components ===================================

do_dxc() {
    # TWO DROPS, ONE PIN. The Windows DLLs are what a D3D12 session loads; the
    # Linux tarball is what the Vulkan port compiles SPIR-V with. They are
    # fetched together, and deliberately from the SAME release tag — see
    # DXC_LINUX_TGZ. Each half skips independently so a partial tree completes.

    # dxcompiler.dll + dxil.dll — required by the DEFAULT --dxr session and by
    # --gpu; without them both fall back to the CPU tracer with a loud line.
    if ! skip "$SDKS/dxc/bin/x64/dxcompiler.dll" "dxc (windows)"; then
        if fetch dxc.zip "https://github.com/microsoft/DirectXShaderCompiler/releases/download/$DXC_TAG/$DXC_ZIP"; then
            # archive root is bin/ inc/ lib/ — extracts straight over SDKs/dxc
            extract dxc.zip "$SDKS/dxc" || true
        fi
    fi

    # bin/dxc + lib/libdxcompiler.so — the SPIR-V compiler for the Vulkan
    # backend. The official build DOES carry SPIR-V codegen (verified: it emits
    # a valid module for a cs_6_5 kernel), which is the premise the whole port
    # rests on; if a future pin ever drops it, the fallbacks are the Vulkan
    # SDK's DXC or a source build with -DENABLE_SPIRV_CODEGEN=ON.
    #
    # NOTE the runtime needs its own lib/ on the loader path:
    #     LD_LIBRARY_PATH=SDKs/dxc-linux/lib SDKs/dxc-linux/bin/dxc ...
    # since bin/dxc resolves libdxcompiler.so by soname, not by sibling.
    #
    # ONLY LINUX HAS AN UPSTREAM DROP, and that is a fact about the release
    # rather than about this script: DXC_TAG publishes the Windows zip, this
    # tarball, and a PDB zip — nothing for macOS, nothing for arm64. The pin is
    # what makes a substitute unacceptable rather than merely unofficial (see
    # DXC_LINUX_TGZ above: the two backends compile the identical concatenated
    # source, so a front-end difference is invisible to every gate), so the
    # macOS answer is a source build at the SAME tag, not a community binary.
    if [[ $OS != linux ]]; then
        if [[ ! -e $SDKS/dxc-$OS/bin/dxc ]]; then
            echo "[i] dxc (host-native, SPIR-V) skipped — upstream publishes no"
            echo "    $OS build at $DXC_TAG (windows zip + linux x86_64 tarball only)."
            echo "    A community binary would not be from this pin, and the pin is"
            echo "    what keeps the DXIL and SPIR-V front ends from drifting."
            echo "    Build it from source at the tag, then point"
            echo "    FRUSTRACER_DXC_SPIRV_PATH at the result."
        fi
        return 0
    fi
    if ! skip "$SDKS/dxc-linux/bin/dxc" "dxc (linux, SPIR-V)"; then
        if fetch dxc-linux.tar.gz "https://github.com/microsoft/DirectXShaderCompiler/releases/download/$DXC_TAG/$DXC_LINUX_TGZ"; then
            extract dxc-linux.tar.gz "$SDKS/dxc-linux" || return 0
            chmod +x "$SDKS/dxc-linux/bin/"* 2>/dev/null
        fi
    fi
}

do_spirv() {
    # spirv-val is the gate on everything DXC emits: DXC will happily produce a
    # module that no driver accepts, and "it compiled" is not the claim the port
    # needs. spirv-dis comes along and is what turns a validation failure into a
    # readable diagnosis.
    #
    # The archive is ~180 MB because it carries headers and static libs as well
    # as the binaries; only bin/ is used here. There is no versioned release to
    # pin (upstream ships SPIRV-Tools through the Vulkan SDK), so we resolve the
    # CI bucket's "latest" redirect stub at run time rather than pin a build
    # number that will 404 within weeks.
    if [[ $OS == other ]]; then
        echo "    [i] spirv skipped — the CI bucket publishes linux and macos"
        echo "        builds only. apt install spirv-tools, or the LunarG Vulkan"
        echo "        SDK, which carries spirv-val and a DXC too."
        return 0
    fi
    skip "$SDKS/spirv-tools/bin/spirv-val" spirv && return 0
    local url
    # `grep -oP` is GNU-only — BSD grep rejects -P outright ("invalid option"),
    # which on macOS made this resolve to empty and report the bucket as
    # unreachable. sed's BRE does the same job everywhere.
    url=$(curl -sS --max-time 60 "${SPIRV_BADGE_BASE}_${OS}_clang_release.html" 2>/dev/null |
        sed -n 's/.*url=\([^"]*\)".*/\1/p' | head -1)
    if [[ -z $url ]]; then
        echo "    [x] could not resolve the SPIRV-Tools latest-build link"
        echo "        alternative: apt install spirv-tools / brew install"
        echo "        spirv-tools, or the LunarG Vulkan SDK, which carries"
        echo "        spirv-val and a DXC too"
        fail
        return 0
    fi
    fetch spirv-tools.tgz "$url" || return 0
    # archive root is a single install/ dir holding bin/ include/ lib/
    extract spirv-tools.tgz "$SDKS/spirv-tools" strip1 || return 0
    chmod +x "$SDKS/spirv-tools/bin/"* 2>/dev/null
    # THE macOS PREBUILT IS x86_64 — the bucket publishes no arm64 build — so on
    # Apple silicon it runs under Rosetta, and on a machine without Rosetta it
    # fails as "Bad CPU type in executable" rather than as anything mentioning
    # architecture. Say so at install time: a gate that cannot start its
    # validator is otherwise indistinguishable from one whose modules are bad.
    if [[ $OS == macos && $(uname -m) == arm64 ]] &&
        file "$SDKS/spirv-tools/bin/spirv-val" 2>/dev/null | grep -q x86_64; then
        if ! "$SDKS/spirv-tools/bin/spirv-val" --version >/dev/null 2>&1; then
            echo "    [x] spirv-val will not execute: the prebuilt is x86_64 and this"
            echo "        is arm64. Install Rosetta (softwareupdate --install-rosetta)"
            echo "        or: brew install spirv-tools"
            fail
            return 0
        fi
        echo "    [i] note: the macOS prebuilt is x86_64 (no arm64 build upstream),"
        echo "        so it runs under Rosetta here. brew install spirv-tools for"
        echo "        a native one."
    fi
}

do_fsr() {
    # The ffx loader resolves its provider DLLs by NAME at runtime, so the
    # loader and every amd_fidelityfx_*_dx12.dll must sit in ONE directory — the
    # shim preloads them by absolute path from there. The Denoiser sample's
    # Release dir is that directory (loader + denoiser + upscaler providers
    # together), which is why --ffx-path defaults into it; --ffx-fg-path points
    # at the FSR sample's, which carries the frame-generation provider.
    # Extracted whole: the archive is per-sample and the paths are load-bearing.
    skip "$SDKS/FidelityFX-Samples-prebuilt/Samples/Denoisers/FidelityFX_Denoiser/dx12/x64/Release/amd_fidelityfx_loader_dx12.dll" fsr && return 0
    fetch ffx.zip "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/download/v$FFX_VER/FidelityFX-Samples-v$FFX_VER-prebuilt.zip" || return 0
    extract ffx.zip "$SDKS/FidelityFX-Samples-prebuilt" || return 0
}

do_xess() {
    # libxess.dll sits in bin/ (NOT bin/x64) — which is exactly what
    # --xess-path defaults to: SDKs/XeSS-SDK/bin.
    skip "$SDKS/XeSS-SDK/bin/libxess.dll" xess && return 0
    fetch xess.zip "https://github.com/intel/xess/releases/download/v$XESS_VER/XeSS_SDK_$XESS_VER.zip" || return 0
    extract xess.zip "$SDKS/XeSS-SDK" || return 0
}

do_nppd() {
    # Two NuGet packages, one destination dir: nppd.rs loads DirectML.dll FIRST
    # (by absolute path) so onnxruntime.dll's lazy DML EP resolves from the
    # module list. A .nupkg is a plain zip.
    skip "$SDKS/onnxruntime/bin/DirectML.dll" nppd && return 0
    fetch ort.zip "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$ORT_VER" || return 0
    fetch dml.zip "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/$DML_VER" || return 0
    extract ort.zip "$CACHE/stage-ort" || return 0
    extract dml.zip "$CACHE/stage-dml" || return 0
    mkdir -p "$SDKS/onnxruntime/bin"
    grab "$CACHE/stage-ort" "runtimes/win-x64/native/onnxruntime.dll" "$SDKS/onnxruntime/bin" || fail
    # optional: only some ORT builds ship it, and the DML EP does not need it
    grab "$CACHE/stage-ort" "runtimes/win-x64/native/onnxruntime_providers_shared.dll" "$SDKS/onnxruntime/bin" >/dev/null 2>&1
    grab "$CACHE/stage-dml" "bin/x64-win/DirectML.dll" "$SDKS/onnxruntime/bin" || fail
    echo "    [+] onnxruntime.dll + DirectML.dll -> SDKs/onnxruntime/bin"
}

do_oidn() {
    # The only archive with a version-stamped wrapper dir
    # (oidn-2.5.0.x64.windows/). Stripping it drops the contents in place, so
    # bin/OpenImageDenoise.dll lands under the un-stamped name --oidn-path
    # defaults to.
    skip "$SDKS/oidn.x64.windows/bin/OpenImageDenoise.dll" oidn && return 0
    fetch oidn.zip "https://github.com/RenderKit/oidn/releases/download/v$OIDN_VER/oidn-$OIDN_VER.x64.windows.zip" || return 0
    extract oidn.zip "$SDKS/oidn.x64.windows" strip1 || return 0
}

do_pix() {
    skip "$SDKS/pix/bin/x64/WinPixEventRuntime.dll" pix && return 0
    fetch pix.zip "https://www.nuget.org/api/v2/package/WinPixEventRuntime/$PIX_VER" || return 0
    extract pix.zip "$SDKS/pix" || return 0
}

do_nrd() {
    # The .bat COMPILES this component (NVIDIA ships no prebuilt NRD binaries),
    # and the artifact is a PE whose DXIL shaders are embedded by dxc at build
    # time — an MSVC + Windows-SDK job with no Linux equivalent. A Linux CMake
    # run would happily produce libNRD.so with SPIRV shaders, which a D3D12
    # session cannot load; producing that and calling the component done would
    # be worse than skipping it, so this degrades exactly like the .bat does on
    # a machine with no Visual Studio: a note by default, a failure when named.
    if [[ -e $SDKS/NRD/bin/NRD.dll && -e $SDKS/NRD/bin/perf/NRD.dll ]]; then
        echo "[=] nrd already installed"
        return 0
    fi
    local where="Windows"
    [[ -n ${WSL_DISTRO_NAME:-} ]] && where="the Windows side of this WSL install"
    if [[ -n $NRD_EXPLICIT ]]; then
        echo "    [x] nrd: NRD is portable, but the artifact --nrd loads is not —"
        echo "        NRD.dll is a PE with dxc-built DXIL embedded, and NVIDIA ships"
        echo "        no prebuilt binaries. Build it from $where:"
        echo "            install-prerequisites.bat nrd"
        fail
    else
        echo "    [i] nrd skipped — the artifact is a Windows DLL with embedded DXIL"
        echo "        (MSVC + CMake + dxc), not producible here. Run"
        echo "        \`install-prerequisites.bat nrd\` from $where; every other"
        echo "        feature works without it."
    fi
}

# run <component> — the one place CUR is set, so `fail` can name what broke
# without every helper having to be told which component called it.
run() {
    want "$1" || return 0
    CUR="$1"
    "do_$1"
}

run dxc
run fsr
run xess
run nppd
run oidn
run pix
run nrd
run spirv

# =========================== verification =================================
echo
echo "---- installed ----"
check "DXR/GPU tracing (--dxr default, --gpu)" "$SDKS/dxc/bin/x64/dxcompiler.dll"
check "  (validator)" "$SDKS/dxc/bin/x64/dxil.dll"
check "FSR4-RR / FSR3 (--fsr / K)" "$SDKS/FidelityFX-Samples-prebuilt/Samples/Denoisers/FidelityFX_Denoiser/dx12/x64/Release/amd_fidelityfx_loader_dx12.dll"
check "  (frame generation, --fg)" "$SDKS/FidelityFX-Samples-prebuilt/Samples/Upscalers/FidelityFX_FSR/dx12/x64/Release/amd_fidelityfx_framegeneration_dx12.dll"
check "XeSS (--xess / X)" "$SDKS/XeSS-SDK/bin/libxess.dll"
check "  (XeSS-FG + XeLL, --fg)" "$SDKS/XeSS-SDK/bin/libxess_fg.dll"
check "NPPD (--nppd / J)" "$SDKS/onnxruntime/bin/onnxruntime.dll"
check "  (DirectML EP)" "$SDKS/onnxruntime/bin/DirectML.dll"
check "OIDN (--oidn / N)" "$SDKS/oidn.x64.windows/bin/OpenImageDenoise.dll"
check "PIX markers (--pix-markers)" "$SDKS/pix/bin/x64/WinPixEventRuntime.dll"
check "NRD denoiser (--nrd)" "$SDKS/NRD/bin/NRD.dll"
check "  (perf variant, --nrd-perf)" "$SDKS/NRD/bin/perf/NRD.dll"
echo "---- host-native ($OS; vulkan backend port) ----"
if [[ $OS == linux ]]; then
    check "DXC -> SPIR-V" "$SDKS/dxc-linux/bin/dxc"
    check "  (runtime; needs LD_LIBRARY_PATH)" "$SDKS/dxc-linux/lib/libdxcompiler.so"
else
    # Report the ABSENCE with its reason rather than a bare MISSING row: this
    # one is not retryable, so a row that looks like a failed download would
    # send someone rerunning the script forever.
    echo " [--] DXC -> SPIR-V  no upstream $OS build at $DXC_TAG — source build"
    echo "                     at the tag (--check-spirv/--check-vk need it)"
fi
check "SPIR-V validation" "$SDKS/spirv-tools/bin/spirv-val"
check "  (disassembler)" "$SDKS/spirv-tools/bin/spirv-dis"

# DLSS is decided at BUILD time, not here (see the header) — but say so in the
# summary, where someone looking for the missing DLSS-RR row will look. The
# staged snippet DLL is the truthful signal: build.rs copies it next to the
# binary exactly when the SDK was present at `cargo build --release`.
echo
if [[ -e $ROOT/target/release/nvngx_dlssd.dll || -e $ROOT/target/x86_64-pc-windows-msvc/release/nvngx_dlssd.dll ]]; then
    echo " [ok] DLSS (RR+FG)   built in (nvngx_dlssd.dll staged by build.rs)"
else
    echo " [i] DLSS (RR+FG)    build-time only — needs the NDA DLSS SDK at"
    echo "                     FRUSTRACER_DLSS_SDK when \`cargo build\` runs; not"
    echo "                     fetchable here. Without it the chain starts at FSR/XeSS."
fi

# The NPPD weights are the one thing no installer may fetch: the pretrained
# checkpoint carries no license grant (see tools/nppd-export/README.md), so
# neither it nor the exported graph may be redistributed — you export it.
echo
if [[ -e $SDKS/nppd/nppd_small.onnx ]]; then
    echo " [ok] NPPD model     SDKs/nppd/nppd_small.onnx"
else
    echo " [--] NPPD model     MISSING — the weights carry no license grant and"
    echo "                     cannot be downloaded by this script. Export them:"
    echo "                         python3 tools/nppd-export/export.py --fp16"
    echo "                     (--nppd needs it; every other feature above does not)"
fi

echo
if [[ -n $FAILED ]]; then
    echo "failed: ${FAILED_LIST[*]}"
    if [[ ${FAILED_LIST[*]} == nrd ]]; then
        echo "nrd is a Windows BUILD, not a download — there is nothing here to retry."
    else
        echo "rerun, or install those by hand (see README)."
    fi
    exit 1
fi
echo "done. These are Windows runtime DLLs: run \`cargo run --release\` from Windows"
echo "against this checkout and it picks them up automatically."
exit 0
