# crenderctl.ps1 - apply/remove CRender in a game folder.
#
# Usage:
#   .\tools\crenderctl.ps1 -Path "F:\Heaven Benchmark 4.0\bin" -Enable
#   .\tools\crenderctl.ps1 -Path "F:\Heaven Benchmark 4.0\bin" -Enable -Neural
#   .\tools\crenderctl.ps1 -Path "F:\Heaven Benchmark 4.0\bin" -Status
#   .\tools\crenderctl.ps1 -Path "F:\Heaven Benchmark 4.0\bin" -Disable
#
# -Enable copies d3d11.dll + retro.json next to the game exe (-Neural adds the
# ONNX Runtime/DirectML runtime + models). -Disable removes exactly what
# -Enable installed (tracked via a .crenderctl.json manifest in the game dir).

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [switch]$Enable,
    [switch]$Disable,
    [switch]$Status,
    [switch]$Neural,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$root     = Split-Path $PSScriptRoot -Parent
$manifest = Join-Path $Path '.crenderctl.json'
$legacy   = Join-Path $Path '.retroctl.json'
if (-not (Test-Path $manifest) -and (Test-Path $legacy)) { $manifest = $legacy }

$coreFiles = @(
    @{ Src = Join-Path $root 'build\win32\Release\d3d11.dll'; Dst = 'd3d11.dll' },
    @{ Src = Join-Path $root 'config\retro.json';           Dst = 'retro.json'  }
)
$neuralFiles = @(
    @{ Src = Join-Path $root 'third_party\onnxruntime\bin\onnxruntime.dll'; Dst = 'onnxruntime.dll' },
    @{ Src = Join-Path $root 'third_party\onnxruntime\bin\DirectML.dll';    Dst = 'DirectML.dll'    },
    @{ Src = Join-Path $root 'config\retro_ai.onnx';          Dst = 'retro_ai.onnx'          },
    @{ Src = Join-Path $root 'config\retro_ai_temporal.onnx'; Dst = 'retro_ai_temporal.onnx' }
)

function Get-ExeArch([string]$exe) {
    # Returns 'x86', 'x64', or $null. Reads the PE machine field.
    try {
        $fs = [IO.File]::OpenRead($exe)
        try {
            $br = New-Object IO.BinaryReader $fs
            $fs.Seek(0x3C, 'Begin') | Out-Null
            $fs.Seek($br.ReadInt32() + 4, 'Begin') | Out-Null
            switch ($br.ReadUInt16()) {
                0x014c { return 'x86' }
                0x8664 { return 'x64' }
            }
        } finally { $fs.Close() }
    } catch {}
    return $null
}

if (-not (Test-Path $Path)) { throw "Game folder not found: $Path" }

if ($Status) {
    if (Test-Path $manifest) {
        $m = Get-Content $manifest -Raw | ConvertFrom-Json
        Write-Host "CRender is ENABLED in $Path"
        $m.files | ForEach-Object {
            $ok = Test-Path (Join-Path $Path $_)
            Write-Host ("  {0} {1}" -f $(if ($ok) {'[ok]'} else {'[MISSING]'}), $_)
        }
    } else {
        Write-Host "CRender is NOT installed in $Path"
    }
    exit 0
}

if ($Enable) {
    $dllSrc = $coreFiles[0].Src
    if (-not (Test-Path $dllSrc)) {
        throw "d3d11.dll not built yet. Run: cmake --build --preset win32-release"
    }

    # Sanity check: warn if the folder's exes are 64-bit (proxy is x86-only).
    Get-ChildItem $Path -Filter *.exe | ForEach-Object {
        $arch = Get-ExeArch $_.FullName
        if ($arch -eq 'x64') {
            Write-Warning "$($_.Name) is 64-bit - the x86 proxy will NOT load in it."
        } elseif ($arch -eq 'x86') {
            Write-Host "$($_.Name) is x86 - OK"
        }
    }

    # Don't stomp a foreign d3d11.dll that isn't ours.
    $dllDst = Join-Path $Path 'd3d11.dll'
    if ((Test-Path $dllDst) -and -not $Force) {
        $same = (Get-FileHash $dllSrc).Hash -eq (Get-FileHash $dllDst).Hash
        if (-not $same) {
            throw "A different d3d11.dll already exists in $Path. Inspect it or rerun with -Force to overwrite."
        }
    }

    $files = @($coreFiles)
    if ($Neural) {
        $files += $neuralFiles | Where-Object { Test-Path $_.Src }
        if (($files | Measure-Object).Count -lt ($coreFiles.Count + $neuralFiles.Count)) {
            Write-Warning "Some Neural files are missing - fetch them with tools/ai/fetch_ort.ps1. Continuing without them."
        }
    }

    $installed = @()
    foreach ($f in $files) {
        if (-not (Test-Path $f.Src)) { Write-Warning "skip (missing): $($f.Src)"; continue }
        Copy-Item $f.Src (Join-Path $Path $f.Dst) -Force
        $installed += $f.Dst
        Write-Host "installed $($f.Dst)"
    }

    @{ files = $installed; when = (Get-Date).ToString('s') } |
        ConvertTo-Json | Set-Content $manifest -Encoding UTF8
    Write-Host "Enabled. Launch the game normally; press F9 in-game to cycle modes."
    exit 0
}

if ($Disable) {
    if (-not (Test-Path $manifest)) {
        Write-Host "Nothing to remove (no .crenderctl.json manifest in $Path)."
        exit 0
    }
    $m = Get-Content $manifest -Raw | ConvertFrom-Json
    foreach ($f in $m.files) {
        $p = Join-Path $Path $f
        if (Test-Path $p) { Remove-Item $p -Force; Write-Host "removed $f" }
    }
    Remove-Item $manifest -Force
    Write-Host "Disabled. The game runs stock again."
    exit 0
}

Write-Host "Pass -Enable, -Disable, or -Status. See header comment for examples."
