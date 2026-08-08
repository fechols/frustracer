@echo off
rem ===========================================================================
rem  frustracer — runtime SDK installer
rem
rem  Building NEVER needs any of this: the MIT headers the shims compile
rem  against (FidelityFX) are committed, and every SDK below is
rem  LoadLibrary'd at runtime, so `cargo build --release` and every DLL-free
rem  `--check*` gate work on a bare checkout. This script fetches the runtime
rem  DLLs the *interactive* features need, into the directories the defaults in
rem  src/main.rs already point at (each is also overridable with the matching
rem  --*-path flag / FRUSTRACER_*_PATH env var).
rem
rem  DLSS is the one feature this script CANNOT fetch: ray reconstruction and
rem  frame generation both ride the raw-NGX shims, which need the (non-
rem  redistributable, non-fetchable) DLSS SDK present at BUILD time --
rem  FRUSTRACER_DLSS_SDK points at it; build.rs stages the snippet DLLs.
rem
rem  The binaries are license-restricted (that is why they are gitignored and
rem  not committed) — this downloads them from each vendor's own release page
rem  onto YOUR machine. Nothing here redistributes them.
rem
rem  Usage:
rem    install-prerequisites.bat                 all components
rem    install-prerequisites.bat dxc fsr         only those
rem    install-prerequisites.bat all /force      re-download and re-extract
rem    install-prerequisites.bat /clean          delete the download cache
rem
rem  Components: dxc fsr xess nppd oidn pix nrd
rem  Needs: Windows 10 1803+ (curl.exe + tar.exe are in-box). ~700 MB of
rem  downloads, ~2 GB on disk after extraction.
rem
rem  nrd is the ONE component that compiles locally (VS 2022 + CMake; NVIDIA
rem  publishes no prebuilt NRD binaries — the GitHub releases are source tags).
rem  Building frustracer itself still needs none of this: NRD.dll is
rem  LoadLibrary'd at runtime like every other SDK here, and a default run that
rem  finds no VS just skips nrd with a note ([x] + failure exit only when nrd
rem  was asked for BY NAME).
rem ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "ROOT=%CD%"
set "SDKS=%ROOT%\SDKs"
set "CACHE=%TEMP%\frustracer-prereqs"

rem --- pinned versions ------------------------------------------------------
rem  Bump deliberately: the ORT/DirectML pair below is the one CLAUDE.md pins
rem  as verified (an old DirectML under a new ORT fails the NPPD U-Net's Resize
rem  node at run time), and XeSS stays on the 2.x line the code was written and
rem  gated against (3.0.1 exists and is untested here).
set "OIDN_VER=2.5.0"
set "XESS_VER=2.1.1"
set "DXC_TAG=v1.9.2602.24"
set "DXC_ZIP=dxc_2026_05_27.zip"
set "FFX_VER=2.3.0"
set "ORT_VER=1.24.4"
set "DML_VER=1.15.4"
set "PIX_VER=1.0.240308001"
rem  NRD is pinned BOTH here and in src/nrd.rs (the transcribed structs +
rem  runtime GetLibraryDesc gate) — move the two together or --nrd sheds loudly.
set "NRD_TAG=v4.17.3"

rem --- tools ---------------------------------------------------------------
rem  Absolute System32 paths on purpose, NOT bare names: a PATH with Git for
rem  Windows (or msys/cygwin) on it shadows both with their GNU/MSYS twins, and
rem  GNU tar cannot read a zip and parses "C:\..." as an rsh host:path.
set "CURL=%SystemRoot%\System32\curl.exe"
set "TAR=%SystemRoot%\System32\tar.exe"
if not exist "%CURL%" (echo [x] %CURL% not found ^(needs Windows 10 1803+^)& exit /b 2)
if not exist "%TAR%"  (echo [x] %TAR% not found ^(needs Windows 10 1803+^)& exit /b 2)

rem --- args ----------------------------------------------------------------
set "FORCE="
set "SEL="
set "KNOWN="
for %%A in (%*) do (
    if /i "%%~A"=="/force" (set "FORCE=1") else (
    if /i "%%~A"=="/clean" (
        echo removing %CACHE%
        rmdir /s /q "%CACHE%" 2>nul
        exit /b 0
    ) else (
    if /i "%%~A"=="all" (set "SEL=") else (
    if /i "%%~A"=="dlss" (goto :arg_dlss) else (
    if /i "%%~A"=="dxc" set "KNOWN=1"
    if /i "%%~A"=="fsr" set "KNOWN=1"
    if /i "%%~A"=="xess" set "KNOWN=1"
    if /i "%%~A"=="nppd" set "KNOWN=1"
    if /i "%%~A"=="oidn" set "KNOWN=1"
    if /i "%%~A"=="pix" set "KNOWN=1"
    if /i "%%~A"=="nrd" (set "KNOWN=1" & set "NRD_EXPLICIT=1")
    rem  Reject via goto, not an in-block exit /b: exiting from inside this
    rem  parenthesized loop body terminates the script but does NOT reliably
    rem  propagate the exit code to the caller (measured: cmd /c saw 0).
    if not defined KNOWN (set "BAD=%%~A" & goto :arg_unknown)
    set "KNOWN="
    set "SEL=!SEL! %%~A"
    ))))
)
if not defined SEL (set "SEL= dxc fsr xess nppd oidn pix nrd")
goto :args_done

rem `dlss` was a valid component in the Streamline era; say why it is gone
rem instead of silently installing nothing.
:arg_dlss
echo [x] dlss is no longer fetchable: DLSS builds against the NDA DLSS
echo     SDK at BUILD time ^(set FRUSTRACER_DLSS_SDK; see the header^).
exit /b 2

:arg_unknown
echo [x] unknown component "%BAD%" ^(valid: dxc fsr xess nppd oidn pix all^)
exit /b 2

:args_done

if not exist "%CACHE%" mkdir "%CACHE%"
set "FAILED="

echo.
echo frustracer prerequisites -^> %SDKS%
echo components:%SEL%
echo cache:     %CACHE%   ^(reused across runs; /clean to drop^)
echo.

call :want dxc  && call :do_dxc
call :want fsr  && call :do_fsr
call :want xess && call :do_xess
call :want nppd && call :do_nppd
call :want oidn && call :do_oidn
call :want pix  && call :do_pix
call :want nrd  && call :do_nrd

rem =========================== verification =================================
echo.
echo ---- installed ----
call :check "DXR/GPU tracing (--dxr default, --gpu)" "%SDKS%\dxc\bin\x64\dxcompiler.dll"
call :check "  (validator)"                          "%SDKS%\dxc\bin\x64\dxil.dll"
call :check "FSR4-RR / FSR3 (--fsr / K)"             "%SDKS%\FidelityFX-Samples-prebuilt\Samples\Denoisers\FidelityFX_Denoiser\dx12\x64\Release\amd_fidelityfx_loader_dx12.dll"
call :check "XeSS (--xess / X)"                      "%SDKS%\XeSS-SDK\bin\libxess.dll"
call :check "NPPD (--nppd / J)"                      "%SDKS%\onnxruntime\bin\onnxruntime.dll"
call :check "  (DirectML EP)"                        "%SDKS%\onnxruntime\bin\DirectML.dll"
call :check "OIDN (--oidn / N)"                      "%SDKS%\oidn.x64.windows\bin\OpenImageDenoise.dll"
call :check "PIX markers (--pix-markers)"            "%SDKS%\pix\bin\x64\WinPixEventRuntime.dll"
call :check "NRD denoiser (--nrd)"                   "%SDKS%\NRD\bin\NRD.dll"

rem DLSS is decided at BUILD time, not here (see the header) — but say so in
rem the summary, where someone looking for the missing DLSS-RR row will look.
rem The staged snippet DLL is the truthful signal: build.rs copies it next to
rem the binary exactly when the SDK was present at `cargo build --release`.
echo.
if exist "%ROOT%\target\release\nvngx_dlssd.dll" (
    echo  [ok] DLSS ^(RR+FG^)   built in ^(nvngx_dlssd.dll staged by build.rs^)
) else (
    echo  [i] DLSS ^(RR+FG^)   build-time only — needs the NDA DLSS SDK at
    echo                      FRUSTRACER_DLSS_SDK when `cargo build` runs; not
    echo                      fetchable here. Without it the chain starts at FSR/XeSS.
)

rem The NPPD weights are the one thing no installer may fetch: the pretrained
rem checkpoint carries no license grant (see tools/nppd-export/README.md), so
rem neither it nor the exported graph may be redistributed — you export it.
echo.
if exist "%SDKS%\nppd\nppd_small.onnx" (
    echo  [ok] NPPD model     SDKs\nppd\nppd_small.onnx
) else (
    echo  [--] NPPD model     MISSING — the weights carry no license grant and
    echo                      cannot be downloaded by this script. Export them:
    echo                          python tools\nppd-export\export.py --fp16
    echo                      ^(--nppd needs it; every other feature above does not^)
)

echo.
if defined FAILED (
    echo one or more downloads failed — rerun, or install those by hand ^(see README^).
    exit /b 1
)
echo done. `cargo run --release` now picks these up automatically.
exit /b 0

rem =========================== components ===================================

:do_dxc
rem dxcompiler.dll + dxil.dll — required by the DEFAULT --dxr session and by
rem --gpu; without them both fall back to the CPU tracer with a loud line.
call :skip "%SDKS%\dxc\bin\x64\dxcompiler.dll" dxc && exit /b 0
call :fetch dxc.zip "https://github.com/microsoft/DirectXShaderCompiler/releases/download/%DXC_TAG%/%DXC_ZIP%" || exit /b 0
rem archive root is bin/ inc/ lib/ — extracts straight over SDKs\dxc
call :unzip dxc.zip "%SDKS%\dxc" || exit /b 0
exit /b 0

:do_fsr
rem The ffx loader resolves its provider DLLs by NAME at runtime, so the loader
rem and every amd_fidelityfx_*_dx12.dll must sit in ONE directory — the shim
rem preloads them by absolute path from there. The Denoiser sample's Release
rem dir is that directory (loader + denoiser + upscaler providers together),
rem which is why --ffx-path defaults into it. Extracted whole: the archive is
rem per-sample and the paths are load-bearing.
call :skip "%SDKS%\FidelityFX-Samples-prebuilt\Samples\Denoisers\FidelityFX_Denoiser\dx12\x64\Release\amd_fidelityfx_loader_dx12.dll" fsr && exit /b 0
call :fetch ffx.zip "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/download/v%FFX_VER%/FidelityFX-Samples-v%FFX_VER%-prebuilt.zip" || exit /b 0
call :unzip ffx.zip "%SDKS%\FidelityFX-Samples-prebuilt" || exit /b 0
exit /b 0

:do_xess
rem libxess.dll sits in bin/ (NOT bin/x64) — which is exactly what
rem --xess-path defaults to: SDKs\XeSS-SDK\bin.
call :skip "%SDKS%\XeSS-SDK\bin\libxess.dll" xess && exit /b 0
call :fetch xess.zip "https://github.com/intel/xess/releases/download/v%XESS_VER%/XeSS_SDK_%XESS_VER%.zip" || exit /b 0
call :unzip xess.zip "%SDKS%\XeSS-SDK" || exit /b 0
exit /b 0

:do_nppd
rem Two NuGet packages, one destination dir: nppd.rs loads DirectML.dll FIRST
rem (by absolute path) so onnxruntime.dll's lazy DML EP resolves from the
rem module list. A .nupkg is a plain zip.
call :skip "%SDKS%\onnxruntime\bin\DirectML.dll" nppd && exit /b 0
call :fetch ort.zip "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/%ORT_VER%" || exit /b 0
call :fetch dml.zip "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/%DML_VER%" || exit /b 0
call :unzip ort.zip "%CACHE%\stage-ort" || exit /b 0
call :unzip dml.zip "%CACHE%\stage-dml" || exit /b 0
if not exist "%SDKS%\onnxruntime\bin" mkdir "%SDKS%\onnxruntime\bin"
copy /y "%CACHE%\stage-ort\runtimes\win-x64\native\onnxruntime.dll" "%SDKS%\onnxruntime\bin\" >nul || set "FAILED=1"
copy /y "%CACHE%\stage-ort\runtimes\win-x64\native\onnxruntime_providers_shared.dll" "%SDKS%\onnxruntime\bin\" >nul
copy /y "%CACHE%\stage-dml\bin\x64-win\DirectML.dll" "%SDKS%\onnxruntime\bin\" >nul || set "FAILED=1"
echo     [+] onnxruntime.dll + DirectML.dll -^> SDKs\onnxruntime\bin
exit /b 0

:do_oidn
rem The only archive with a version-stamped wrapper dir (oidn-2.5.0.x64.windows/).
rem --strip-components=1 drops it in place, so bin\OpenImageDenoise.dll lands
rem under the un-stamped name --oidn-path defaults to. (Staging + `move` would
rem NOT do: the cache is on %TEMP% and `move` cannot relocate a directory across
rem volumes — it fails with "Access is denied" whenever the repo is on another
rem drive.)
call :skip "%SDKS%\oidn.x64.windows\bin\OpenImageDenoise.dll" oidn && exit /b 0
call :fetch oidn.zip "https://github.com/RenderKit/oidn/releases/download/v%OIDN_VER%/oidn-%OIDN_VER%.x64.windows.zip" || exit /b 0
rem  the tar arg MUST stay quoted: "=" is a batch token delimiter, so an
rem  unquoted --strip-components=1 arrives at :unzip split across %3 and %4.
call :unzip oidn.zip "%SDKS%\oidn.x64.windows" "--strip-components=1" || exit /b 0
exit /b 0

:do_pix
call :skip "%SDKS%\pix\bin\x64\WinPixEventRuntime.dll" pix && exit /b 0
call :fetch pix.zip "https://www.nuget.org/api/v2/package/WinPixEventRuntime/%PIX_VER%" || exit /b 0
call :unzip pix.zip "%SDKS%\pix" || exit /b 0
exit /b 0

:do_nrd
rem NVIDIA publishes NO prebuilt NRD binaries, so this component COMPILES the
rem pinned tag locally — the one component with a toolchain requirement (CMake +
rem VS 2022 C++ tools; NRD's FetchContent pulls ShaderMake/MathLib as plain zip
rem URLs, so configure needs network but NOT git). Only the resulting NRD.dll is
rem kept: the DXIL shader blobs are EMBEDDED in it and src/nrd.rs loads it at
rem runtime, so building frustracer still needs none of this. A missing
rem toolchain is a hard failure only when nrd was named on the command line —
rem a default `all` run degrades to a note, like the DLSS/NPPD-model rows.
call :skip "%SDKS%\NRD\bin\NRD.dll" nrd && exit /b 0
set "CMAKE="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
where cmake.exe >nul 2>nul && set "CMAKE=cmake.exe"
if not defined CMAKE if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%C in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe 2^>nul`) do set "CMAKE=%%C"
)
set "VSDIR="
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%V in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VSDIR=%%V"
)
if not defined CMAKE goto :nrd_no_toolchain
if not defined VSDIR goto :nrd_no_toolchain
call :fetch nrd-src.zip "https://github.com/NVIDIA-RTX/NRD/archive/refs/tags/%NRD_TAG%.zip" || exit /b 0
call :unzip nrd-src.zip "%CACHE%\nrd-src" || exit /b 0
rem  the tag zip wraps everything in NRD-<ver-without-v>/
set "NRD_SRC=%CACHE%\nrd-src\NRD-%NRD_TAG:~1%"
if not exist "%NRD_SRC%\CMakeLists.txt" (
    echo     [x] nrd source layout unexpected ^(no CMakeLists at %NRD_SRC%^)
    set "FAILED=1"
    exit /b 0
)
rem  DXIL only (our sessions are D3D12; skipping DXBC/SPIRV halves the shader
rem  build); encoding pins 2/1 are the build contract src/nrd.rs gates at
rem  runtime via GetLibraryDesc. SHADERMAKE_DXC_PATH prefers the dxc component's
rem  compiler when installed (script order runs dxc first); absent, ShaderMake
rem  falls back to the Windows-SDK dxc on its own.
set "DXCARG="
if exist "%SDKS%\dxc\bin\x64\dxc.exe" set "DXCARG=-DSHADERMAKE_DXC_PATH=%SDKS%\dxc\bin\x64\dxc.exe"
echo     [+] configuring NRD %NRD_TAG% ^(cmake log: %CACHE%\nrd-cmake.log^)
"%CMAKE%" -S "%NRD_SRC%" -B "%CACHE%\nrd-build" -A x64 -DNRD_STATIC_LIBRARY=OFF -DNRD_NRI=OFF -DNRD_EMBEDS_DXIL_SHADERS=ON -DNRD_EMBEDS_DXBC_SHADERS=OFF -DNRD_EMBEDS_SPIRV_SHADERS=OFF -DNRD_NORMAL_ENCODING=2 -DNRD_ROUGHNESS_ENCODING=1 %DXCARG% > "%CACHE%\nrd-cmake.log" 2>&1
if errorlevel 1 (
    echo     [x] nrd cmake configure FAILED — see %CACHE%\nrd-cmake.log
    set "FAILED=1"
    exit /b 0
)
echo     [+] building NRD ^(a few minutes; log: %CACHE%\nrd-build.log^)
"%CMAKE%" --build "%CACHE%\nrd-build" --config Release --parallel > "%CACHE%\nrd-build.log" 2>&1
if errorlevel 1 (
    echo     [x] nrd build FAILED — see %CACHE%\nrd-build.log
    set "FAILED=1"
    exit /b 0
)
rem  CMAKE_RUNTIME_OUTPUT_DIRECTORY is _Bin under the SOURCE dir; cover both
rem  the flat and the per-config layout across generator versions.
set "NRD_DLL="
if exist "%NRD_SRC%\_Bin\NRD.dll" set "NRD_DLL=%NRD_SRC%\_Bin\NRD.dll"
if exist "%NRD_SRC%\_Bin\Release\NRD.dll" set "NRD_DLL=%NRD_SRC%\_Bin\Release\NRD.dll"
if not defined NRD_DLL (
    echo     [x] nrd build produced no NRD.dll under %NRD_SRC%\_Bin
    set "FAILED=1"
    exit /b 0
)
if not exist "%SDKS%\NRD\bin" mkdir "%SDKS%\NRD\bin"
copy /y "%NRD_DLL%" "%SDKS%\NRD\bin\" >nul || (set "FAILED=1" & exit /b 0)
echo     [+] NRD.dll -^> SDKs\NRD\bin
exit /b 0

:nrd_no_toolchain
if defined NRD_EXPLICIT (
    echo     [x] nrd needs CMake + Visual Studio 2022 C++ tools: NVIDIA ships no
    echo         prebuilt NRD binaries, so it must compile locally.
    set "FAILED=1"
) else (
    echo     [i] nrd skipped — needs CMake + VS 2022 C++ tools to compile
    echo         ^(NVIDIA ships no prebuilt binaries^). Rerun `install-prerequisites.bat nrd`
    echo         once they are installed; every other feature works without it.
)
exit /b 0

rem =========================== helpers ======================================

rem :want <component> — is it in the selection?
:want
echo %SEL% | findstr /i /c:" %~1" >nul
exit /b %errorlevel%

rem :skip <marker> <name> — already installed and not /force?
:skip
if defined FORCE exit /b 1
if exist "%~1" (echo [=] %~2 already installed & exit /b 0)
exit /b 1

rem :fetch <cachefile> <url>
:fetch
if not defined FORCE if exist "%CACHE%\%~1" (
    echo [.] %~1 cached
    exit /b 0
)
echo [^>] downloading %~1
"%CURL%" -L --fail --retry 3 --retry-delay 2 --progress-bar -o "%CACHE%\%~1.part" "%~2"
if errorlevel 1 (
    echo     [x] download FAILED: %~2
    del "%CACHE%\%~1.part" 2>nul
    set "FAILED=1"
    exit /b 1
)
move /y "%CACHE%\%~1.part" "%CACHE%\%~1" >nul
exit /b 0

rem :unzip <cachefile> <destdir> [extra tar args] — in-box bsdtar reads zip. The
rem archive is named RELATIVE from a pushd into the cache: bsdtar parses a
rem "C:\..." -f argument as an rsh host:path. -C is not parsed that way, so the
rem dest stays absolute.
:unzip
if not exist "%~2" mkdir "%~2"
echo     [+] extracting %~1 -^> %~2
rem  TAR_RC, deliberately NOT "RC": RC is the MSVC resource-compiler override,
rem  and leaking RC=0 into the nrd component's cmake child broke its configure
rem  ("Could not find compiler set in environment variable RC: 0").
pushd "%CACHE%"
"%TAR%" -xf "%~1" -C "%~2" %~3
set "TAR_RC=%errorlevel%"
popd
if not "%TAR_RC%"=="0" (
    echo     [x] extract FAILED ^(corrupt download? rerun with /force^)
    set "FAILED=1"
    exit /b 1
)
exit /b 0

rem :check <label> <path>
rem  Deliberately goto, not an if/else block: the labels carry parentheses
rem  ("(--dxr default)") and a `)` inside a parenthesized block is a parse error.
:check
if not exist "%~2" goto :check_miss
echo  [ok] %~1
exit /b 0
:check_miss
echo  [--] %~1  MISSING
exit /b 0
