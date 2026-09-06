$ErrorActionPreference = 'Stop'
$install = $env:C3X_RENDERER_CIV3_CONQUESTS
if (-not $install) { $install = Join-Path ${env:ProgramFiles(x86)} 'GOG Galaxy\Games\Civilization III Complete\Conquests' }
$files = @(Get-ChildItem -LiteralPath $install -Filter test.biq -Recurse -File | Where-Object { $_.FullName -notmatch 'C3X_Shared_Verify|C3X_Districts' })
if ($env:C3X_LAB_TEST_BIQ) { $files = @(Get-Item -LiteralPath $env:C3X_LAB_TEST_BIQ) }
$records = @($files | ForEach-Object { [PSCustomObject]@{name=$_.Name; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower(); file=$_} })
$identities = @($records.sha256 | Select-Object -Unique)
if ($identities.Count -ne 1) { throw "Expected one distinct test.biq identity; found $($identities.Count). Set C3X_LAB_TEST_BIQ to disambiguate." }
$destination = Join-Path $PSScriptRoot '.local\real_map'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
Copy-Item -LiteralPath $records[0].file.FullName -Destination (Join-Path $destination 'test.biq') -Force
$records | Select-Object name,bytes,sha256 | ConvertTo-Json
