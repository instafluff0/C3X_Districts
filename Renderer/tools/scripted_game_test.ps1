# Bounded real-game renderer diagnostic, enabled only in the child environment.
param([Parameter(Mandatory=$true)][string]$SaveFile, [string]$ConquestsDirectory,
      [ValidateRange(35,120)][int]$Seconds = 75,
      [ValidateSet('scroll','interaction')][string]$Scenario = 'scroll')
$ErrorActionPreference = 'Stop'
$renderer = Split-Path $PSScriptRoot -Parent
if (-not $ConquestsDirectory) { $ConquestsDirectory = $env:C3X_RENDERER_CIV3_CONQUESTS }
if (-not $ConquestsDirectory) { $ConquestsDirectory = Join-Path ${env:ProgramFiles(x86)} 'GOG Galaxy\Games\Civilization III Complete\Conquests' }
$game = Join-Path $ConquestsDirectory 'Civ3Conquests.exe'
$arm = $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64'
$debugName = if ($arm) { 'dbgviewcli64a.exe' } else { 'dbgviewcli64.exe' }
$debug = Join-Path $renderer ('native\build\live-tools\DebugView\' + $debugName)
$witness = Join-Path $renderer 'native\build\window-witness\window_witness.exe'
foreach ($file in @($game,$SaveFile,$debug,$witness)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw ('Missing input: ' + $file) }
}
if (Get-Process Civ3Conquests,dbgviewcli,dbgviewcli64,dbgviewcli64a -ErrorAction SilentlyContinue) { throw 'Close the game and any debug collectors before a scripted test.' }
$session = Join-Path $env:TEMP ('C3XGameTest\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $session | Out-Null
$save = Join-Path $session 'input.SAV'
Copy-Item -LiteralPath $SaveFile -Destination $save
$originalSaveHash = (Get-FileHash -LiteralPath $SaveFile).Hash
$identities = foreach ($name in @('C3XRenderer.dll','C3XRenderer_x64.dll','C3XRendererHelper64.exe')) {
    [ordered]@{ name=$name; sha256=(Get-FileHash -LiteralPath (Join-Path $renderer ('bin\renderer64\'+$name))).Hash }
}
[ordered]@{game_sha256=(Get-FileHash -LiteralPath $game).Hash; save_sha256=$originalSaveHash; binaries=@($identities)} |
    ConvertTo-Json -Depth 4 | Set-Content (Join-Path $session 'inputs.json')
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class RendererGameCommand {
    [StructLayout(LayoutKind.Sequential)] public struct Point { public int X, Y; }
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr window, uint message, IntPtr key, IntPtr detail);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window, out uint process);
    [DllImport("user32.dll")] public static extern bool GetCursorPos(out Point point);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr window, out Rect rect);
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr window, ref Point point);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] public static extern bool WritePrivateProfileString(string section, string key, string value, string path);
}
'@
function Quote-Arguments([string[]]$Values) {
    ($Values | ForEach-Object {
        if ($_ -match '["\r\n]' -or $_.EndsWith('\')) { throw 'Unsupported argument' }
        '"' + $_ + '"'
    }) -join ' '
}
$collector=$null; $child=$null; $observer=$null
$oldEnvironment=$env:C3X_RENDERER_GAME_TEST_SAVE
$oldMode=$env:C3X_RENDERER_GAME_TEST_MODE
$ini=Join-Path $ConquestsDirectory 'conquests.ini'
$iniBytes=[IO.File]::ReadAllBytes($ini)
$cursor=New-Object RendererGameCommand+Point
$restoreCursor=[RendererGameCommand]::GetCursorPos([ref]$cursor)
try {
    # Restore these exact bytes in finally. Native menu/file-picker loading is
    # necessary because C3X reads the renderer setting during scenario load.
    $latestSave=Join-Path (Split-Path $save -Parent) ([IO.Path]::GetFileNameWithoutExtension($save))
    foreach ($item in @(@('Top Menu','3'),@('Latest Save',$latestSave),@('WindowsFileBox','0'),@('PlayIntro','0'))) {
        if (-not [RendererGameCommand]::WritePrivateProfileString('Conquests',$item[0],$item[1],$ini)) { throw 'Cannot prepare native load menu.' }
    }
    $collector=Start-Process $debug -ArgumentList (Quote-Arguments @('--accepteula','--no-banner','--no-kernel','--duration','180','--max-lines','100000','--log',(Join-Path $session 'renderer.log'),'--log-limit','32')) -PassThru -WindowStyle Hidden
    Start-Sleep -Milliseconds 400
    if ($collector.HasExited) { throw 'Diagnostic collector exited before launch.' }
    # Use direct process creation so the diagnostic environment reaches C3X.
    $env:C3X_RENDERER_GAME_TEST_SAVE=$save
    $env:C3X_RENDERER_GAME_TEST_MODE=$Scenario
    $start=New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName=$game; $start.WorkingDirectory=$ConquestsDirectory; $start.UseShellExecute=$false
    $child=[System.Diagnostics.Process]::Start($start)
    $env:C3X_RENDERER_GAME_TEST_SAVE=$oldEnvironment
    $env:C3X_RENDERER_GAME_TEST_MODE=$oldMode
    Write-Host ('Scripted game PID='+$child.Id+' capture='+$session)
    $observer=Start-Process $witness -ArgumentList (Quote-Arguments @([string]$child.Id,(Join-Path $session 'window'),[string]$Seconds,'2','sampled-window-evidence')) -PassThru -WindowStyle Hidden
    $end=[DateTime]::UtcNow.AddSeconds($Seconds)
    $sent=0
    $started=[DateTime]::UtcNow
    $enterCount=0
    $cursorParked=$false
    $interactionIndex=0
    $interaction=@(@(28,13,'close-welcome'),@(36,90,'zoom-192'),@(43,90,'zoom-160'),@(50,90,'zoom-128'),@(57,0x66,'move-east'),@(65,0x70,'advisor'),@(74,27,'close-advisor'))
    while ([DateTime]::UtcNow -lt $end -and -not $child.HasExited) {
        $child.Refresh(); $window=$child.MainWindowHandle
        if ($window -ne [IntPtr]::Zero) {
            [uint32]$owner=0
            [void][RendererGameCommand]::GetWindowThreadProcessId($window,[ref]$owner)
            if ($owner -ne $child.Id) { throw 'Game window owner changed.' }
            if (-not $cursorParked) {
                $rect=New-Object RendererGameCommand+Rect
                $point=New-Object RendererGameCommand+Point
                if ([RendererGameCommand]::GetClientRect($window,[ref]$rect) -and $rect.Right -gt 0) {
                    $point.X=[int]($rect.Right/2); $point.Y=[int]($rect.Bottom/2)
                    if ([RendererGameCommand]::ClientToScreen($window,[ref]$point)) {
                        $cursorParked=[RendererGameCommand]::SetCursorPos($point.X,$point.Y)
                    }
                }
            }
            # Two Enter presses select Load Game and accept the copied save.
            # The opt-in post-load hook dismisses the known welcome popup.
            $elapsed=([DateTime]::UtcNow-$started).TotalSeconds
            $key=if ($Scenario -eq 'scroll') {0x87} else {0}
            if ($enterCount -lt 2 -and $elapsed -ge (5+2*$enterCount)) { $key=13; ++$enterCount }
            if ($Scenario -eq 'interaction' -and $interactionIndex -lt $interaction.Count -and $elapsed -ge $interaction[$interactionIndex][0]) {
                $key=$interaction[$interactionIndex][1]
                Write-Host ('Interaction command: '+$interaction[$interactionIndex][2])
                ++$interactionIndex
            }
            if ($key -eq 0) { Start-Sleep -Milliseconds 1000; continue }
            if (-not [RendererGameCommand]::PostMessage($window,0x100,[IntPtr]$key,[IntPtr]1)) { throw 'Cannot post diagnostic command.' }
            [void][RendererGameCommand]::PostMessage($window,0x101,[IntPtr]$key,[IntPtr]0)
            ++$sent
        }
        Start-Sleep -Milliseconds 1000
    }
    Write-Host ('Posted diagnostic commands: '+$sent)
} finally {
    $env:C3X_RENDERER_GAME_TEST_SAVE=$oldEnvironment
    $env:C3X_RENDERER_GAME_TEST_MODE=$oldMode
    if ($observer -and -not $observer.HasExited) {
        if (Test-Path (Join-Path $session 'window')) { Set-Content -LiteralPath (Join-Path $session 'window\stop.txt') -Value 'stop' }
        if (-not $observer.WaitForExit(5000)) { $observer.Kill(); $observer.WaitForExit() }
    }
    # Terminate only the process created above. The diagnostic never advances a
    # turn or saves gameplay; the input save is a disposable copy.
    if ($child -and -not $child.HasExited) { $child.Kill(); $child.WaitForExit() }
    [IO.File]::WriteAllBytes($ini,$iniBytes)
    if ($restoreCursor) { [void][RendererGameCommand]::SetCursorPos($cursor.X,$cursor.Y) }
    if ($collector -and -not $collector.HasExited) {
        & $debug --stop | Out-Null
        if (-not $collector.WaitForExit(5000)) { $collector.Kill(); $collector.WaitForExit() }
    }
    if ((Get-FileHash -LiteralPath $SaveFile).Hash -ne $originalSaveHash) { throw 'Original input save changed.' }
}
$log=Get-Content -LiteralPath (Join-Path $session 'renderer.log') -Raw
$steps=@([regex]::Matches($log,'stage=scripted-game-scroll step=(\d+)') | ForEach-Object {[int]$_.Groups[1].Value})
$errors=@($log -split "`n" | Where-Object {$_ -match 'stage=native-operation-failed|budget exceeded|stage=visual-failure|stage=worker-error'})
[ordered]@{ scenario=$Scenario; interaction_commands=$interactionIndex; posted_commands=$sent; load_requested=($log -match 'stage=scripted-game-load'); scroll_steps=$steps;
    native_failures=$errors; original_save_unchanged=$true; scope='Real Civ III diagnostic; window samples require visual review; not an FPS benchmark' } |
    ConvertTo-Json -Depth 4 | Set-Content (Join-Path $session 'result.json')
Write-Host ('Scripted diagnostic saved: '+$session)
if (($Scenario -eq 'scroll' -and $steps.Count -ne 32) -or ($Scenario -eq 'interaction' -and $interactionIndex -ne $interaction.Count) -or $errors.Count) { exit 1 }
