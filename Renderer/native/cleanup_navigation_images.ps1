param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'build\navigation-evidence-20260909'))
if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw "Evidence folder not found: $root" }
if ((Get-Item -LiteralPath $root).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing a linked evidence folder.' }
$keepRuns = @('region-input-ring4-100', 'tight-natural-bounds-100', 'receiver-shadows-independent-100')
$keepImages = @('zoom.bmp', 'zoom.bmp.resident0.bmp', 'zoom.bmp.resident47.bmp', 'zoom.bmp.resident99.bmp')
$images = @(foreach ($run in Get-ChildItem -LiteralPath $root -Directory) {
    if ($run.Name -in $keepRuns -or ($run.Attributes -band [IO.FileAttributes]::ReparsePoint)) { continue }
    foreach ($file in Get-ChildItem -LiteralPath $run.FullName -File -Filter '*.bmp') {
        if ($file.Name -in $keepImages -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) { continue }
        if (-not $file.FullName.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Image is outside the evidence folder.' }
        $file
    }
})
$logicalBytes = ($images | Measure-Object Length -Sum).Sum
Write-Host ("Selected {0} obsolete generated images ({1:N2} GiB logical size)." -f $images.Count, ($logicalBytes / 1GB))
Write-Host 'Only BMP files are selected. Sources, packs, logs, JSON reports, DLLs and recent reference runs stay.'
if (-not $Apply) { Write-Host 'Preview only. Run this script with -Apply to delete the selected images.'; return }
$driveName = ([IO.Path]::GetPathRoot($root)).Substring(0,1)
$freeBefore = (Get-PSDrive -Name $driveName).Free
$manifest = [ordered]@{
    reason = 'User-requested removal of obsolete generated navigation images'
    timestamp = (Get-Date).ToString('o')
    kept_runs = $keepRuns
    kept_images_per_other_run = $keepImages
    note = 'Existing image checksums and comparison reports remain. Deleted images require regeneration for new pixel inspection.'
    removed_files = @($images | ForEach-Object { [ordered]@{path=$_.FullName.Substring($root.Length+1); bytes=$_.Length} })
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root 'image-cleanup-manifest.json') -Encoding utf8
foreach ($file in $images) { Remove-Item -LiteralPath $file.FullName -Force }
$freeAfter = (Get-PSDrive -Name $driveName).Free
Write-Host ("Done. Deleted {0} files; drive space recovered: {1:N2} GiB; free now: {2:N2} GiB." -f $images.Count, (($freeAfter-$freeBefore)/1GB), ($freeAfter/1GB))
