# Exercise launcher quoting and automatic collector shutdown without a game.
param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $PSScriptRoot 'capture_game.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$quote = $ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Quote-Arguments'
}, $true)
if ($quote.Count -ne 1) { throw 'Launcher argument encoder is missing or ambiguous' }
# Load only that function. Never evaluate the launcher or its game-start path.
Invoke-Expression $quote[0].Extent.Text
$stop = $ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-ShortCaptureStopReason'
}, $true)
if ($stop.Count -ne 1) { throw 'Short-capture memory guard is missing' }
Invoke-Expression $stop[0].Extent.Text
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Test directory must be new' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$memoryTest = Join-Path $OutputDirectory 'memory.jsonl'
if (Get-ShortCaptureStopReason $memoryTest) { throw 'Missing telemetry requested a stop' }
'{"event":"process_memory","free_bytes":268435456}' | Set-Content -LiteralPath $memoryTest
if (Get-ShortCaptureStopReason $memoryTest) { throw 'Healthy headroom requested a stop' }
'{"event":"process_memory","free_bytes":125829120}' | Add-Content -LiteralPath $memoryTest
if ((Get-ShortCaptureStopReason $memoryTest) -ne 'low-address-space') { throw 'Low headroom failed to request a stop' }
$windowDirectory = Join-Path $OutputDirectory 'window samples'
$renderer = Split-Path $PSScriptRoot -Parent
$helper = Join-Path $renderer 'native\build\window-witness\window_witness.exe'
$process = Start-Process -FilePath $helper -ArgumentList (Quote-Arguments @(
    'self-test', $windowDirectory, '20', '5', 'sampled-window-evidence')) -PassThru `
    -RedirectStandardOutput (Join-Path $OutputDirectory 'window.log') `
    -RedirectStandardError (Join-Path $OutputDirectory 'window-errors.log')
$null = $process.Handle
$deadline = [DateTime]::UtcNow.AddSeconds(15)
while (-not (Test-Path -LiteralPath (Join-Path $windowDirectory 'timeline.jsonl'))) {
    if ($process.HasExited -or [DateTime]::UtcNow -gt $deadline) { throw 'Test observer failed to start' }
    Start-Sleep -Milliseconds 100
}
Start-Sleep -Seconds 3
'stop' | Set-Content -LiteralPath (Join-Path $windowDirectory 'stop.txt')
if (-not $process.WaitForExit(10000) -or $process.ExitCode -ne 0) { throw 'Test observer failed to stop' }
$finished = Get-Content -LiteralPath (Join-Path $windowDirectory 'finished.json') -Raw | ConvertFrom-Json
if (-not $finished.complete -or $finished.reason -ne 'requested' -or $finished.frames -lt 1) {
    throw 'Missing completed window samples'
}
@{ status='pass'; game_launched=$false; launcher_parses=$true; quoted_shared_path=$true;
   stop_reason=$finished.reason; sampled_frames=$finished.frames } | ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $OutputDirectory 'receipt.json')
Write-Host 'PASS launcher parser, shared-path argument encoding, owned window samples and automatic stop. No game launched.'
