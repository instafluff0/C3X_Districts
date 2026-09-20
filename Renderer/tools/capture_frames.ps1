# Elevated collector only. The game and the main launcher remain unelevated.
param(
    [Parameter(Mandatory=$true)][string]$SessionDirectory,
    [Parameter(Mandatory=$true)][ValidatePattern('^C3XCapture-[0-9a-f-]+$')][string]$SessionName,
    [Parameter(Mandatory=$true)][string]$PresentMon,
    [ValidatePattern('^[a-zA-Z0-9_.-]+\.exe$')][string]$TargetProcess = 'Civ3Conquests.exe'
)
$ErrorActionPreference = 'Stop'
$collector = $null
$result = 1
try {
    if ($SessionDirectory -match '["\r\n]') { throw 'Unsupported capture directory' }
    $signature = Get-AuthenticodeSignature -LiteralPath $PresentMon
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike '*CN=Intel Corporation,*') {
        throw 'PresentMon signature could not be verified'
    }
    $output = Join-Path $SessionDirectory 'frames.csv'
    $arguments = '--process_name ' + $TargetProcess + ' --session_name ' + $SessionName +
        ' --output_file "' + $output + '" --v1_metrics --qpc_time --no_track_input' +
        ' --no_console_stats --timed 900 --terminate_after_timed'
    $collector = Start-Process -FilePath $PresentMon -ArgumentList $arguments -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $SessionDirectory 'frames-status.log') `
        -RedirectStandardError (Join-Path $SessionDirectory 'frames-errors.log')
    Start-Sleep -Milliseconds 500
    if ($collector.HasExited) { throw 'PresentMon stopped before capture began; see frames-errors.log' }
    'ready' | Set-Content -LiteralPath (Join-Path $SessionDirectory 'frames-ready.txt')
    while (-not $collector.HasExited -and -not (Test-Path -LiteralPath (Join-Path $SessionDirectory 'stop-frames.txt'))) {
        Start-Sleep -Milliseconds 250
    }
    if ($collector.HasExited -and $collector.ExitCode -ne 0) { throw 'PresentMon failed; see frames-errors.log' }
    $result = 0
} catch {
    $_.Exception.Message | Set-Content -LiteralPath (Join-Path $SessionDirectory 'frames-helper-error.txt')
} finally {
    if ($collector -and -not $collector.HasExited) {
        # Stop only this invocation's named session, and let CSV output flush.
        & $PresentMon --session_name $SessionName --terminate_existing_session | Out-Null
        if (-not $collector.WaitForExit(5000)) { $collector.Kill(); $collector.WaitForExit(); $result = 1 }
    }
    @{ complete = ($result -eq 0); finished_utc = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $SessionDirectory 'frames-done.json')
}
exit $result
