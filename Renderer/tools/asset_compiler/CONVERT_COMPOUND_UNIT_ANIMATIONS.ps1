param(
    [Parameter(Mandatory = $true)][string]$SourceSets,
    [Parameter(Mandatory = $true)][string]$Pack
)

$ErrorActionPreference = "Stop"
$toolsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rendererRoot = Resolve-Path (Join-Path $toolsDir "..\..")
$converter = Join-Path $rendererRoot "preview\out\animation_tools\export_civ6_animation.exe"
$civNexus = Join-Path $rendererRoot "third_party\CivNexus6\bin\Release\CivNexus6.exe"
$sourceCandidates = @(
    "Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets\Base\Platforms\Windows\BLPs\SHARED_DATA",
    "\\Mac\Home\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets\Base\Platforms\Windows\BLPs\SHARED_DATA"
)
$sourceRoot = $sourceCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $sourceRoot) { throw "The installed Civ VI SHARED_DATA directory is unavailable." }
if (-not (Test-Path $converter)) { throw "Missing animation converter: $converter" }
if (-not (Test-Path $civNexus)) { throw "Missing CivNexus6 executable: $civNexus" }

$document = Get-Content -Raw $SourceSets | ConvertFrom-Json
$converted = 0
foreach ($composition in $document.compositions) {
    $nodeIds = @($composition.parent.id) + @($composition.children | ForEach-Object { $_.id })
    foreach ($actionProperty in $composition.actions.PSObject.Properties) {
        $action = $actionProperty.Name
        $record = $actionProperty.Value
        if ($record.alias) { continue }
        foreach ($nodeId in $nodeIds) {
            $sourceName = $record.nodes.$nodeId
            if (-not $sourceName) { throw "$($composition.slug)/$action does not bind node $nodeId" }
            $input = Join-Path $sourceRoot $sourceName
            $output = Join-Path $Pack "animations\unit\$($composition.slug)\$nodeId\$action.c3anim"
            if (-not (Test-Path $input)) { throw "Missing compound-unit animation source: $input" }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null
            & $converter $civNexus $input $output 0.01
            if ($LASTEXITCODE -ne 0) { throw "Animation conversion failed for $($composition.slug)/$nodeId/$action" }
            $converted += 1
        }
    }
}
Write-Host "Converted $converted compound-unit node/action bindings."
