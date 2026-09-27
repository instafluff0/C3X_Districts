param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$renderer = Split-Path $PSScriptRoot -Parent
$arm = $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64'
$name = if ($arm) { 'dbgviewcli64a.exe' } else { 'dbgviewcli64.exe' }
$debug = Join-Path $renderer ('native\build\live-tools\DebugView\' + $name)
if (-not (Test-Path -LiteralPath $debug -PathType Leaf)) { throw ('Missing signed DebugView CLI: ' + $debug) }
$signature = Get-AuthenticodeSignature -LiteralPath $debug
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike '*CN=Microsoft Corporation,*') {
    throw 'DebugView signature validation failed.'
}
if ($CheckOnly) { Write-Host 'PASS: signed map-failure capture tool is ready. No game launched.'; exit 0 }
if (Get-Process -Name Civ3Conquests -ErrorAction SilentlyContinue) {
    throw 'Close Civ III before starting the capture, so the first map attempt is recorded.'
}
if (Get-Process -Name dbgviewcli,dbgviewcli64,dbgviewcli64a -ErrorAction SilentlyContinue) {
    throw 'Another DebugView collector is running. Close it before this capture.'
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$session = Join-Path $env:TEMP ('C3XMapFailure\' + $stamp)
New-Item -ItemType Directory -Path $session -Force | Out-Null
$log = Join-Path $session 'renderer.log'
$binaries = foreach ($name in @('C3XRenderer.dll', 'C3XRenderer_x64.dll', 'C3XRendererHelper64.exe')) {
    $binary = Join-Path $renderer ('bin\renderer64\' + $name)
    [ordered]@{ name = $name; sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant() }
}
[ordered]@{ renderer_binaries = @($binaries) } | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $session 'binaries.json') -Encoding UTF8
$arguments = @('--accepteula', '--no-banner', '--no-kernel', '--duration', '180',
    '--max-lines', '50000', '--log', ('"' + $log + '"'), '--log-limit', '16') -join ' '
$collector = Start-Process -FilePath $debug -ArgumentList $arguments -PassThru -WindowStyle Hidden
Start-Sleep -Milliseconds 400
if ($collector.HasExited) { throw 'DebugView stopped before capture began.' }
Write-Host 'Capture started. Launch Civ III normally and show the black map.'
Write-Host 'Wait about 15 seconds, then return here and press Enter.'
[void](Read-Host 'Press Enter to finish')
if (-not $collector.HasExited) {
    & $debug --stop | Out-Null
    if (-not $collector.WaitForExit(5000)) { Stop-Process -Id $collector.Id -ErrorAction SilentlyContinue }
}
Write-Host ('Capture saved: ' + $log)
