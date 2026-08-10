#!/usr/bin/env bash
# ===========================================================================
#  Type-check the WINDOWS (D3D12) half of the tree from Linux.
#
#  WHY THIS EXISTS. Most of this renderer is `#[cfg(windows)]` — the whole
#  src/gpu/ backend, session(), the present arms — so a Linux `cargo check` is
#  green while saying nothing at all about that code. During the Vulkan port
#  that is not a theoretical gap: M0 shipped a hard E0433 (cli.rs calling
#  `gpu::` after its import was dropped) that was invisible on Linux and would
#  have broken every Windows build, and the M0 cleanup shipped a second, milder
#  one (an unused re-export in oidn.rs). Both were found by this script.
#
#  It needs no Windows machine, no MSVC, and no license beyond Microsoft's
#  redistributable-headers terms that `xwin` prompts for. `cargo check` does
#  not link the final binary, and frustracer's own build.rs gates the C++ shims
#  on the HOST being Windows, so the shims are skipped here — only the two
#  third-party -sys crates actually compile C for the target.
#
#  USAGE
#      tools/win-cross-check.sh                  # check
#      tools/win-cross-check.sh --setup          # one-time deps, then check
#      tools/win-cross-check.sh clippy           # any cargo subcommand
#
#  WHAT --setup INSTALLS (all user-local, nothing touches the repo or /usr):
#      cargo install xwin                        ~/.cargo/bin
#      xwin splat --include-debug-libs           ~/.xwin      (~820 MB)
#      ninja                                     ~/.local/bin
#  It also needs clang-cl/lld-link/llvm-lib/llvm-rc from an LLVM install, and
#  the rustup target x86_64-pc-windows-msvc.
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XW="${XWIN_DIR:-$HOME/.xwin}"
TARGET=x86_64-pc-windows-msvc

# Require the WHOLE toolset from one directory, not just clang-cl. /usr/bin
# commonly carries clang-cl and llvm-rc while lld-link lives only in the
# versioned dir — and picking that one yields a CMake compiler-probe failure
# whose message ("is not able to compile a simple test program") points nowhere
# near the actual cause. Prefer the versioned dirs; fall back to /usr/bin.
LLVM_BIN=""
for d in $(ls -d /usr/lib/llvm-*/bin 2>/dev/null | sort -V -r) /usr/bin; do
    if [[ -x "$d/clang-cl" && -x "$d/lld-link" && -x "$d/llvm-lib" && -x "$d/llvm-rc" ]]; then
        LLVM_BIN="$d"; break
    fi
done
if [[ -z $LLVM_BIN ]]; then
    echo "[x] no directory carries the full set (clang-cl lld-link llvm-lib llvm-rc)"
    echo "    looked in: /usr/lib/llvm-*/bin  /usr/bin"
    echo "    install LLVM/clang/lld, e.g.  apt install clang lld llvm"
    exit 2
fi
export PATH="$LLVM_BIN:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [[ ${1:-} == --setup ]]; then
    shift
    echo "[>] rustup target"
    rustup target add "$TARGET"
    command -v xwin >/dev/null || { echo "[>] cargo install xwin"; cargo install xwin --locked; }
    # --include-debug-libs is REQUIRED, not optional: CMake's compiler probe
    # links with -MDd, so without msvcrtd.lib the sdl3-sys build dies in
    # configure with a misleading "compiler is not able to compile a simple
    # test program".
    [[ -f "$XW/crt/lib/x86_64/msvcrtd.lib" ]] || {
        echo "[>] xwin splat (Microsoft CRT + Windows SDK headers) -> $XW"
        # --cache-dir is not optional hygiene: xwin defaults it to ./.xwin-cache,
        # i.e. 886 MB of Microsoft packages dropped in whatever directory you
        # happened to run from — the repo root, if you ran this the obvious way.
        xwin --accept-license --arch x86_64 --cache-dir "$HOME/.xwin-cache" \
             splat --include-debug-libs --output "$XW"
    }
    command -v ninja >/dev/null || {
        echo "[>] ninja (CMake cannot find a Visual Studio generator here)"
        mkdir -p "$HOME/.local/bin"
        tmp=$(mktemp -d)
        curl -sSL -o "$tmp/n.zip" \
          https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip
        unzip -oq "$tmp/n.zip" -d "$HOME/.local/bin"
        chmod +x "$HOME/.local/bin/ninja"
        rm -rf "$tmp"
    }
fi

if [[ ! -d $XW/crt/include ]]; then
    echo "[x] no Windows CRT/SDK headers at $XW — run: $0 --setup"
    exit 2
fi

# --- the environment, and why each piece is load-bearing -------------------
#  INCLUDE   llvm-rc does NOT read CFLAGS, and SDL compiles a version.rc; the
#            MSVC-standard INCLUDE var is what it honours (-I appeared not to).
#  LIB       lld-link honours LIB the same way link.exe does. The CMake
#            toolchain's *_LINKER_FLAGS_INIT are NOT enough on their own.
#  CFLAGS    the `cmake` crate passes -DCMAKE_C_FLAGS explicitly, which
#            OVERRIDES a toolchain file's CMAKE_C_FLAGS_INIT — so the include
#            paths have to ride the env var, not the toolchain.
#  winrt     SDL's src/core/windows pulls roapi.h.
INC_DIRS=("$XW/crt/include" "$XW/sdk/include/ucrt" "$XW/sdk/include/um"
          "$XW/sdk/include/shared" "$XW/sdk/include/winrt" "$XW/sdk/include/cppwinrt")
LIB_DIRS=("$XW/crt/lib/x86_64" "$XW/sdk/lib/ucrt/x86_64" "$XW/sdk/lib/um/x86_64")

imsvc=""; for d in "${INC_DIRS[@]}"; do imsvc+=" /imsvc$d"; done
INCLUDE=$(IFS=';'; echo "${INC_DIRS[*]}"); export INCLUDE
LIB=$(IFS=';'; echo "${LIB_DIRS[*]}");     export LIB

export CC_x86_64_pc_windows_msvc=clang-cl
export CXX_x86_64_pc_windows_msvc=clang-cl
export AR_x86_64_pc_windows_msvc=llvm-lib
export CFLAGS_x86_64_pc_windows_msvc="-Wno-unused-command-line-argument$imsvc"
export CXXFLAGS_x86_64_pc_windows_msvc="$CFLAGS_x86_64_pc_windows_msvc"
export CMAKE_GENERATOR=Ninja
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=lld-link

libpath=""; for d in "${LIB_DIRS[@]}"; do libpath+=" /libpath:$d"; done
TC="${TMPDIR:-/tmp}/frustracer-xwin-toolchain.cmake"
cat > "$TC" <<EOF
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)
set(CMAKE_C_COMPILER clang-cl)
set(CMAKE_CXX_COMPILER clang-cl)
set(CMAKE_LINKER lld-link)
set(CMAKE_AR llvm-lib)
set(CMAKE_RC_COMPILER llvm-rc)
set(CMAKE_MT llvm-mt)
# Both the LIB env var and these flags are set, deliberately. LIB alone is
# enough in an inherited interactive shell but was observed NOT to reach
# lld-link from a scrubbed environment, and a cross-check that works only when
# you happen to have the right shell is worse than none.
set(CMAKE_EXE_LINKER_FLAGS_INIT    "$libpath")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "$libpath")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "$libpath")
# We link SDL statically (build-from-source-static), so the shared target and
# its version.rc are dead weight here — and the .rc is the one thing in the SDL
# build that needs a resource compiler at all.
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
EOF
export CMAKE_TOOLCHAIN_FILE="$TC"

SUB="${1:-check}"; shift || true
cd "$ROOT"
echo "[>] cargo $SUB --target $TARGET"
exec cargo "$SUB" --target "$TARGET" "$@"
