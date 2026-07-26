param(
    [ValidateSet("demo", "check", "bench", "continuation")]
    [string]$Mode = "demo"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$exe = Join-Path $PSScriptRoot "target\release\frustracer.exe"
& cargo build --release
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

switch ($Mode) {
    "demo" {
        & $exe --no-world --gpu --prefer-intel `
            --no-upscale --no-fg --no-hdr --no-settings
        exit $LASTEXITCODE
    }
    "check" {
        & $exe --check-gpu --prefer-intel
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        & $exe --check-dxr --prefer-intel
        exit $LASTEXITCODE
    }
    "bench" {
        $common = @(
            "--no-world",
            "--spin", "path",
            "--gpu",
            "--prefer-intel",
            "--gpu-timing",
            "--spin-frames", "2200",
            "--spin-warmup", "1600"
        )

        # Each executable is a fresh shader-compilation session. Preserve any
        # caller ablations, but clear them for the current/plain rows.
        $savedAbl = [Environment]::GetEnvironmentVariable("FR_ABL", "Process")
        $code = 0
        try {
            Write-Host "=== pre-B70-pass hybrid (oldcut,nobatch) ==="
            [Environment]::SetEnvironmentVariable(
                "FR_ABL", "oldcut,nobatch", "Process"
            )
            & $exe @common --spin-hybrid
            $code = $LASTEXITCODE

            if ($code -eq 0) {
                Write-Host "=== current hybrid ==="
                [Environment]::SetEnvironmentVariable("FR_ABL", $null, "Process")
                & $exe @common --spin-hybrid
                $code = $LASTEXITCODE
            }
            if ($code -eq 0) {
                Write-Host "=== plain hardware RayQuery ==="
                & $exe @common --spin-plain
                $code = $LASTEXITCODE
            }
        }
        finally {
            [Environment]::SetEnvironmentVariable(
                "FR_ABL", $savedAbl, "Process"
            )
        }
        exit $code
    }
    "continuation" {
        Write-Host "=== opaque traversal-frontier correctness ==="
        & $exe --check-gpu --prefer-intel --continuation-rays
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }

        $common = @(
            "--no-world",
            "--spin", "path",
            "--gpu",
            "--prefer-intel",
            "--gpu-timing",
            "--no-replay",
            "--spin-frames", "2200",
            "--spin-warmup", "1600",
            "--spin-hybrid",
            "--continuation-rays"
        )

        Write-Host "=== SW continuation (opaque beam frontier) ==="
        & $exe @common
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }

        Write-Host "=== SW root control (same t_start/intersector/shading) ==="
        & $exe @common --no-cut-rays
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }

        # Complete an ABBA ordering so clock/temperature drift cannot masquerade
        # as a continuation win in a single process-order pair.
        Write-Host "=== SW root control (ABBA repeat) ==="
        & $exe @common --no-cut-rays
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }

        Write-Host "=== SW continuation (ABBA repeat) ==="
        & $exe @common
        exit $LASTEXITCODE
    }
}
