#!/usr/bin/env bash
# ===========================================================================
#  frustracer — runtime SDK installer (the Linux twin of
#  install-prerequisites.bat: same pins, same destinations, same markers)
#
#  IT FETCHES WINDOWS BINARIES, ON PURPOSE. frustracer is a D3D12 renderer
#  with no second backend, so nothing this script installs runs on Linux. The
#  point is that a checkout that LIVES on Linux — a shared or dual-boot drive,
#  a WSL working copy, a cross-build staging tree, a fileserver clone — can
#  have its SDKs/ tree populated from here instead of rebooting to do it. Every
#  file lands in exactly the directory the matching default in src/cli.rs
#  points at, so the Windows session that consumes them needs no flags.
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
#  Components: dxc fsr xess nppd oidn pix nrd
#  Needs: bash 4+, curl (or wget), unzip (or bsdtar). Measured on a full run:
#  ~510 MB of downloads, 549 MB in SDKs/ after extraction. (The .bat quotes
#  ~2 GB because its nrd component also unpacks and builds an NRD source tree,
#  which this one does not.)
# ===========================================================================
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")" || exit 2
ROOT="$PWD"
SDKS="$ROOT/SDKs"
CACHE="${TMPDIR:-/tmp}/frustracer-prereqs"

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
    echo "    apt install unzip   |   dnf install unzip   |   pacman -S unzip"
    exit 2
fi

# --- args ----------------------------------------------------------------
FORCE=
NRD_EXPLICIT=
SEL=()
ALL=(dxc fsr xess nppd oidn pix nrd)
for a in "$@"; do
    case "${a,,}" in
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
        dxc | fsr | xess | nppd | oidn | pix)
            SEL+=("${a,,}")
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

# unzip_to <zipfile> <destdir> — the one extractor call site.
#  unzip exits 1 for warnings (skipped/renamed entries) and >=2 for real
#  errors, so only >=2 is a failure; bsdtar is plain 0/non-0.
unzip_to() {
    local rc=0
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
        local stage="$CACHE/stage-${1%.zip}"
        rm -rf "$stage"
        mkdir -p "$stage"
        if ! unzip_to "$zip" "$stage"; then
            echo "    [x] extract FAILED (corrupt download? rerun with --force)"
            fail
            return 1
        fi
        local inner=()
        mapfile -t inner < <(find "$stage" -mindepth 1 -maxdepth 1)
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
    mapfile -t hit < <(find "$src" -ipath "*/${rel}" -type f)
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
    # dxcompiler.dll + dxil.dll — required by the DEFAULT --dxr session and by
    # --gpu; without them both fall back to the CPU tracer with a loud line.
    skip "$SDKS/dxc/bin/x64/dxcompiler.dll" dxc && return 0
    fetch dxc.zip "https://github.com/microsoft/DirectXShaderCompiler/releases/download/$DXC_TAG/$DXC_ZIP" || return 0
    # archive root is bin/ inc/ lib/ — extracts straight over SDKs/dxc
    extract dxc.zip "$SDKS/dxc" || return 0
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
