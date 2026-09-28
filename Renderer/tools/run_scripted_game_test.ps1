# An elevated interactive task preserves Civ III's required token and child
# environment. The task exists only for this bounded diagnostic invocation.
param([Parameter(Mandatory=$true)][string]$SaveFile, [ValidateRange(35,120)][int]$Seconds=75, [ValidateSet('scroll','interaction','lifecycle','combat','turn','mouse')][string]$Scenario='scroll', [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$UnitPack='UnitAnimationFidelity', [ValidateSet('melee','victory','retreat','bombard','army','air','capture')][string]$CombatCase='melee', [ValidateRange(1,10)][int]$SampleHz=2, [switch]$ProfileRenderer)
$ErrorActionPreference='Stop'
$user=(Get-CimInstance Win32_ComputerSystem).UserName
if (-not $user) { throw 'An interactive Windows login is required.' }
$renderer=Split-Path $PSScriptRoot -Parent
$name='C3XRendererGameTest-'+[guid]::NewGuid().ToString('N')
$out=Join-Path $renderer ('native\build\'+$name)
New-Item -ItemType Directory -Path $out | Out-Null
$target=Join-Path $PSScriptRoot 'scripted_game_test.ps1'
if ($target.Contains("'") -or $SaveFile.Contains("'") -or $out.Contains("'")) { throw 'Unsupported diagnostic path.' }
$wrapper=Join-Path $out 'run.ps1'
$profileArgument=if ($ProfileRenderer) { '-ProfileRenderer' } else { '' }
@"
Start-Transcript -Path '$out\launcher.log' -Force
try { & '$target' -SaveFile '$SaveFile' -Seconds $Seconds -Scenario $Scenario -CombatCase $CombatCase -UnitPack $UnitPack -SampleHz $SampleHz $profileArgument; exit `$LASTEXITCODE }
finally { Stop-Transcript }
"@ | Set-Content -LiteralPath $wrapper
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -ExecutionPolicy Bypass -File "'+$wrapper+'"')
$principal=New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Seconds ($Seconds+45))
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings | Out-Null
try {
    Start-ScheduledTask -TaskName $name
    Write-Host ('Started diagnostic task: '+$name)
    $deadline=[DateTime]::UtcNow.AddSeconds($Seconds+40)
    do {
        Start-Sleep -Seconds 1
        $info=Get-ScheduledTaskInfo -TaskName $name
        $state=(Get-ScheduledTask -TaskName $name).State
    } while ([DateTime]::UtcNow -lt $deadline -and ($state -eq 'Running' -or $info.LastTaskResult -eq 267011))
    $info | Select-Object LastRunTime,LastTaskResult | ConvertTo-Json | Set-Content (Join-Path $out 'task-result.json')
    if ($state -eq 'Running') { throw 'Diagnostic task did not finish within its bound.' }
    Get-Content -LiteralPath (Join-Path $out 'launcher.log')
    exit $info.LastTaskResult
} finally {
    if ((Get-ScheduledTask -TaskName $name).State -eq 'Running') { Stop-ScheduledTask -TaskName $name }
    Unregister-ScheduledTask -TaskName $name -Confirm:$false
}
