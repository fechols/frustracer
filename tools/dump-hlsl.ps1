<#
.SYNOPSIS
    Sweep FR_DUMP_HLSL over the kernel-assembly configurations that change what
    compiles, and zip the result.

.DESCRIPTION
    Kernels in this tree are string-CONCATENATED at runtime (gpu/trace.rs holds
    32 include_str! consts, gpu/dxr.rs 3 more, with generated #define blocks in
    front), so NO FILE ON DISK IS WHAT THE COMPILER SEES. FR_DUMP_HLSL writes
    the assembled source of every unit; this script runs it once per interesting
    configuration and collects the lot.

    Written for the Vulkan port's SPIR-V spike -- the dumps are the ground truth
    that a Linux-side reproduction of the assembly is diffed against -- but it
    is equally the input Radeon GPU Analyzer wants for offline ISA work, which
    is the other reason FR_DUMP_HLSL exists.

    ONE PROCESS PER CONFIG IS MANDATORY, not tidiness: FR_DUMP_HLSL is read
    through a OnceLock and cached for the life of the process, so a second
    configuration run in the same process dumps into the FIRST one's directory
    and silently overwrites nothing while appearing to work.

.PARAMETER OutDir
    Where to write. Defaults OUTSIDE the repo (%TEMP%\frustracer-hlsl-dump) so
    a few hundred MB of generated HLSL never lands in the working tree and
    .gitignore needs no entry.

.PARAMETER Amd
    Also dump the AMD arm (adds "#define CAND_TMIN0 1" via cand_defs). Needs an
    AMD adapter present; skipped by default because a box without one would
    produce a dump identical to base and imply coverage it does not have.

.PARAMETER NoZip
    Leave the directories loose instead of producing the .zip.

.EXAMPLE
    .\tools\dump-hlsl.ps1
    .\tools\dump-hlsl.ps1 -Amd -OutDir D:\dumps
#>
[CmdletBinding()]
param(
    [string] $OutDir = (Join-Path $env:TEMP "frustracer-hlsl-dump"),
    [switch] $Amd,
    [switch] $NoZip
)

$ErrorActionPreference = "Stop"

# Repo root = this script's parent's parent, so the script works from any cwd.
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$exe  = Join-Path $repo "target\release\frustracer.exe"

if (-not (Test-Path $exe)) {
    Write-Host "[x] no release binary at $exe" -ForegroundColor Red
    Write-Host "    build it first:  cargo build --release"
    exit 2
}

# The dumps are only meaningful against the tree that produced them -- the whole
# point is comparing ASSEMBLED sources. Report the commit rather than enforcing
# it: a dirty tree is a legitimate thing to dump, it just has to be said.
$commit = (& git -C $repo rev-parse --short HEAD 2>$null)
$dirty  = (& git -C $repo status --porcelain 2>$null)
Write-Host ""
Write-Host "frustracer HLSL dump sweep"
Write-Host "  repo:   $repo"
Write-Host "  commit: $commit$(if ($dirty) { '  (WORKING TREE DIRTY -- say so when you hand these over)' })"
Write-Host "  out:    $OutDir"
Write-Host ""

# A scene with alpha-cutout foliage AND transmissive glass, which is what arms
# ALPHA_CUTOUT + TRANS_SHADOW. Committed via git-lfs, and a pointer-file
# checkout is a real failure mode, so probe rather than assume.
$sm = $null
foreach ($cand in @("scenes\san-miguel\san-miguel-low-poly.obj",
                    "scenes\san-miguel\san-miguel-low-poly.obj.zst",
                    "scenes\san-miguel\san-miguel.obj")) {
    $p = Join-Path $repo $cand
    if ((Test-Path $p) -and ((Get-Item $p).Length -gt 4096)) { $sm = $cand; break }
}
if (-not $sm) {
    Write-Host "[!] no san-miguel scene found (git lfs pull?) -- skipping the" -ForegroundColor Yellow
    Write-Host "    alphatrans and height configs; every other config still runs." -ForegroundColor Yellow
    Write-Host ""
}

# Each entry: the config label, and the argv that produces it. --check-gpu
# builds the whole wavefront kernel set; --check-dxr builds the RTPSO library.
$runs = New-Object System.Collections.ArrayList
[void]$runs.Add(@{ d = "base";     a = @("--check-gpu") })
if ($sm) {
    [void]$runs.Add(@{ d = "alphatrans"; a = @($sm, "--check-gpu") })
    [void]$runs.Add(@{ d = "height";     a = @($sm, "--check-gpu", "--heightfield") })
}
[void]$runs.Add(@{ d = "swrays";   a = @("--check-gpu", "--sw-rays") })
[void]$runs.Add(@{ d = "noftree";  a = @("--check-gpu", "--no-ftree") })
[void]$runs.Add(@{ d = "nocaches"; a = @("--check-gpu", "--no-cloud-shadow", "--no-sky-lod") })
[void]$runs.Add(@{ d = "inline1";  a = @("--check-dxr") })
[void]$runs.Add(@{ d = "inline0";  a = @("--check-dxr", "--dxr-inline", "0") })
[void]$runs.Add(@{ d = "inline2";  a = @("--check-dxr", "--dxr-inline", "2") })
[void]$runs.Add(@{ d = "inline3";  a = @("--check-dxr", "--dxr-inline", "3") })
if ($Amd) { [void]$runs.Add(@{ d = "amd"; a = @("--check-gpu", "--prefer-amd") }) }

# A stale directory must not be merged into (a config that fails to dump would
# inherit the previous run's files and read as healthy), so the sweep starts
# clean -- but -OutDir is user-supplied and this is a recursive force delete, so
# it only ever fires on a path that is plausibly ours: at least two segments
# deep, and either empty, absent, or already holding a dump.
if (Test-Path $OutDir) {
    $full  = (Resolve-Path $OutDir).Path
    $depth = @($full.TrimEnd('\').Split('\') | Where-Object { $_ -ne "" }).Count
    $ours  = @(Get-ChildItem -Path $full -Force -ErrorAction SilentlyContinue)
    $looksLikeOurs = ($ours.Count -eq 0) -or
                     (@($ours | Where-Object { $_.Name -match '\.log$' -or $_.PSIsContainer }).Count -eq $ours.Count)
    if ($depth -lt 2 -or -not $looksLikeOurs) {
        Write-Host "[x] refusing to clear $full" -ForegroundColor Red
        Write-Host "    it is either a drive root or holds files this script did not write."
        Write-Host "    point -OutDir at a fresh directory."
        exit 2
    }
    Remove-Item -Recurse -Force $full
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$results = @()
foreach ($r in $runs) {
    $dir = Join-Path $OutDir $r.d
    $env:FR_DUMP_HLSL = $dir
    Write-Host ("==> {0,-11} {1}" -f $r.d, ($r.a -join " ")) -ForegroundColor Cyan
    # The GATE's verdict is irrelevant here: FR_DUMP_HLSL writes at COMPILE
    # time, before any gate runs, so a failing suite still yields a full dump.
    # Only "did any .hlsl appear" decides whether this config is usable.
    #
    # $ErrorActionPreference MUST drop to Continue across the native call.
    # Under "Stop", PowerShell 5.1 turns a native command's stderr into
    # NativeCommandError and THROWS on the first line -- and this renderer
    # narrates heavily on stderr, so every config would abort within moments of
    # starting while looking like a GPU failure. The log keeps that output for
    # diagnosing a config that dumps nothing.
    $log     = Join-Path $OutDir ("{0}.log" -f $r.d)
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    Push-Location $repo
    try     { & $exe $r.a 2>&1 | Out-File -FilePath $log -Width 200 }
    finally { Pop-Location; $ErrorActionPreference = $prevEAP }
    $n = 0
    if (Test-Path $dir) { $n = @(Get-ChildItem -Path $dir -Filter *.hlsl -ErrorAction SilentlyContinue).Count }
    $results += [pscustomobject]@{ Config = $r.d; Files = $n; Exit = $LASTEXITCODE }
    if ($n -eq 0) {
        Write-Host ("    [x] no dump written (exit {0}) -- see {1}" -f $LASTEXITCODE, $log) -ForegroundColor Red
    } else {
        Write-Host ("    [ok] {0} units" -f $n) -ForegroundColor Green
    }
}
Remove-Item Env:\FR_DUMP_HLSL -ErrorAction SilentlyContinue

Write-Host ""
$results | Format-Table -AutoSize
$total = ($results | Measure-Object -Property Files -Sum).Sum
$empty = @($results | Where-Object { $_.Files -eq 0 }).Count
Write-Host ("{0} units across {1} configs ({2} empty)" -f $total, $results.Count, $empty)

if ($total -eq 0) {
    Write-Host "[x] nothing dumped -- no GPU, or DXC missing (SDKs\dxc\bin\x64)" -ForegroundColor Red
    exit 1
}

if (-not $NoZip) {
    $zip = "$OutDir.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path (Join-Path $OutDir "*") -DestinationPath $zip
    $mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Write-Host ""
    Write-Host ("wrote {0} ({1} MB)" -f $zip, $mb) -ForegroundColor Green
    Write-Host "commit: $commit"
}
