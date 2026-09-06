$ErrorActionPreference = "Stop"

$rendererRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$extractReport = Join-Path $rendererRoot "preview\out\resources\resource_animation_extract.json"
$rawRoot = Join-Path $rendererRoot "preview\out\resources\raw_animations"
$packRoot = Join-Path $rendererRoot "packs\ResourceNormalized"
$converter = Join-Path $rendererRoot "preview\out\animation_tools\export_civ6_animation.exe"
$civNexus = Join-Path $rendererRoot "third_party\CivNexus6\bin\Release\CivNexus6.exe"
$build = Join-Path $PSScriptRoot "BUILD_CIV6_ANIMATION_CONVERTER.bat"

& cmd.exe /d /c $build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$document = Get-Content -Raw -LiteralPath $extractReport | ConvertFrom-Json
if ($document.schema -ne "c3x.resource_animation_extract.v0") {
    throw "Unsupported resource-animation extraction report"
}

$converted = 0
foreach ($clip in $document.unique_clips) {
    $inputPath = Join-Path $rawRoot ($clip.raw_fgx -replace "/", "\")
    $outputPath = Join-Path $packRoot ($clip.normalized_clip -replace "/", "\")
    $outputDirectory = Split-Path -Parent $outputPath
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    $scale = [Convert]::ToString([double]$clip.translation_scale, [Globalization.CultureInfo]::InvariantCulture)
    & $converter $civNexus $inputPath $outputPath $scale
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $converted += 1
}

Write-Output "Converted $converted resource animation clips."
exit 0
