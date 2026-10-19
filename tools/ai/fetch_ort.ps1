# fetch_ort.ps1 — pulls ONNX Runtime (DirectML EP) + DirectML redist for Win32/x86
# and x64 into third_party/onnxruntime/{include,bin,bin64}.
#
# Microsoft.ML.OnnxRuntime.DirectML dropped win-x86 binaries in newer releases, so this
# walks versions newest-first (<= -MaxVersion) and keeps the first one that still ships
# runtimes/win-x86/native/onnxruntime.dll. The matching Microsoft.AI.DirectML version is
# read from the package's .nuspec dependency.
#
# Usage:  powershell -ExecutionPolicy Bypass -File tools/ai/fetch_ort.ps1 [-Version 1.20.1]
param(
    [string]$Version = "",
    [string]$MaxVersion = "1.22.99"
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

$root    = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$outDir  = Join-Path $root "third_party\onnxruntime"
$incDir  = Join-Path $outDir "include"
$binDir  = Join-Path $outDir "bin"
$bin64Dir = Join-Path $outDir "bin64"
$tmpDir  = Join-Path $env:TEMP "retro_ort_fetch"
New-Item -ItemType Directory -Force -Path $incDir, $binDir, $bin64Dir, $tmpDir | Out-Null

$feed = "https://api.nuget.org/v3-flatcontainer"

function Get-Nupkg([string]$id, [string]$ver) {
    $lid = $id.ToLowerInvariant()
    $dst = Join-Path $tmpDir "$lid.$ver.nupkg"
    if (-not (Test-Path $dst)) {
        Write-Host "  downloading $id $ver ..."
        Invoke-WebRequest -UseBasicParsing -Uri "$feed/$lid/$ver/$lid.$ver.nupkg" -OutFile $dst
    }
    return $dst
}

function Extract-Entries($zip, [string]$prefix, [string]$dest, [string]$filter = "*") {
    $n = 0
    foreach ($e in $zip.Entries) {
        if ($e.FullName.StartsWith($prefix) -and $e.Name -ne "" -and $e.Name -like $filter) {
            $target = Join-Path $dest $e.Name
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $target, $true)
            $n++
        }
    }
    return $n
}

function Version-Key([string]$v) {
    $core = ($v -split "-")[0]
    $p = $core.Split(".") | ForEach-Object { [int]$_ }
    while ($p.Count -lt 3) { $p += 0 }
    return ($p[0] * 1000000 + $p[1] * 1000 + $p[2])
}

$ortId = "Microsoft.ML.OnnxRuntime.DirectML"
if ($Version -ne "") {
    $candidates = @($Version)
} else {
    $idx = Invoke-RestMethod -UseBasicParsing -Uri "$feed/$($ortId.ToLowerInvariant())/index.json"
    $maxKey = Version-Key $MaxVersion
    $candidates = $idx.versions |
        Where-Object { $_ -notmatch "-" -and (Version-Key $_) -le $maxKey } |
        Sort-Object { Version-Key $_ } -Descending
}

$picked = $null
foreach ($v in $candidates) {
    Write-Host "probing $ortId $v"
    $pkg = Get-Nupkg $ortId $v
    $zip = [System.IO.Compression.ZipFile]::OpenRead($pkg)
    try {
        $hasX86 = $zip.Entries | Where-Object { $_.FullName -eq "runtimes/win-x86/native/onnxruntime.dll" }
        if (-not $hasX86) { Write-Host "  no win-x86 binaries"; continue }
        $picked = $v
        [void](Extract-Entries $zip "runtimes/win-x86/native/" $binDir "*.dll")
        $n64 = Extract-Entries $zip "runtimes/win-x64/native/" $bin64Dir "*.dll"
        if ($n64 -eq 0) { Write-Host "  note: no win-x64 binaries in $v (Terminator ONNX stays x86-only)" }
        [void](Extract-Entries $zip "build/native/include/" $incDir "*.h")
        $nuspec = $zip.Entries | Where-Object { $_.FullName -like "*.nuspec" } | Select-Object -First 1
        $sr = New-Object System.IO.StreamReader($nuspec.Open())
        [xml]$spec = $sr.ReadToEnd(); $sr.Close()
        $dmlDep = $spec.package.metadata.dependencies.SelectNodes("//*[local-name()='dependency']") |
                  Where-Object { $_.id -eq "Microsoft.AI.DirectML" } | Select-Object -First 1
        $dmlVer = ($dmlDep.version -replace "[\[\]\(\) ]", "").Split(",")[0]
    } finally { $zip.Dispose() }
    break
}
if ($null -eq $picked) { throw "no $ortId version <= $MaxVersion ships win-x86 binaries" }
Write-Host "ORT-DML $picked (x86) extracted; DirectML dependency $dmlVer"

$dmlPkg = Get-Nupkg "Microsoft.AI.DirectML" $dmlVer
$zip = [System.IO.Compression.ZipFile]::OpenRead($dmlPkg)
try {
    $n = Extract-Entries $zip "bin/x86-win/" $binDir "DirectML.dll"
    if ($n -eq 0) { throw "Microsoft.AI.DirectML $dmlVer has no bin/x86-win/DirectML.dll" }
    [void](Extract-Entries $zip "bin/x64-win/" $bin64Dir "DirectML.dll")
    [void](Extract-Entries $zip "include/" $incDir "*.h")
} finally { $zip.Dispose() }

Set-Content -Path (Join-Path $outDir "VERSION.txt") -Value "Microsoft.ML.OnnxRuntime.DirectML $picked`nMicrosoft.AI.DirectML $dmlVer"
Write-Host "done -> $outDir"
Get-ChildItem $binDir | Format-Table Name, Length
if (Test-Path $bin64Dir) {
    Write-Host "x64:";
    Get-ChildItem $bin64Dir | Format-Table Name, Length
}
