# Bounded real-game renderer diagnostic, enabled only in the child environment.
param([Parameter(Mandatory=$true)][string]$SaveFile, [string]$ConquestsDirectory,
      [ValidateRange(35,360)][int]$Seconds = 75,
      [ValidateSet('scroll','interaction','lifecycle','combat','turn','mouse','zoom','hud','city','settler','site-toggle','debug','debug-scroll','newgame','turn-stress')][string]$Scenario = 'scroll',
      [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$UnitPack='UnitAnimationFidelity', [ValidateSet('melee','victory','retreat','bombard','army','air','capture')][string]$CombatCase='melee', [ValidateRange(1,10)][int]$SampleHz = 2,
      [switch]$ProfileRenderer, [switch]$MeasureCadence,
      [ValidateSet(0,1,2)][int]$SceneSamples=0, [ValidateRange(-1,1)][double]$SceneSharpness=-1)
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
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint x, uint y, uint data, UIntPtr extra);
    [DllImport("user32.dll")] public static extern void keybd_event(byte key, byte scan, uint flags, UIntPtr extra);
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
// Read the helper's existing successful-presentation counter. This maps only
// the IPC header read-only; it never requests pixels or submits renderer work.
public sealed class RendererCadenceReader : IDisposable {
    // Mirrored from helper_trial/scene_wire.h; executable test checks this ABI.
    public const int WireVersion = 11, FrameOffset = 228, ZoomOffset = 232;
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] static extern IntPtr OpenFileMapping(uint access, bool inherit, string name);
    [DllImport("kernel32.dll")] static extern IntPtr MapViewOfFile(IntPtr mapping, uint access, uint high, uint low, UIntPtr bytes);
    [DllImport("kernel32.dll")] static extern bool UnmapViewOfFile(IntPtr view);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    IntPtr mapping, view;
    public RendererCadenceReader(string name) {
        mapping=OpenFileMapping(4, false, name);
        if(mapping!=IntPtr.Zero) view=MapViewOfFile(mapping,4,0,0,(UIntPtr)(ZoomOffset+4));
        if(view==IntPtr.Zero) { Dispose(); throw new InvalidOperationException("Cannot read diagnostic helper telemetry"); }
        if(Marshal.ReadInt32(view)!=0x32483343 || Marshal.ReadInt32(view,4)!=WireVersion) {
            Dispose(); throw new InvalidOperationException("Renderer telemetry ABI changed");
        }
    }
    public int Frames { get { return Marshal.ReadInt32(view,FrameOffset); } }
    public int ZoomQ16 { get { return Marshal.ReadInt32(view,ZoomOffset); } }
    public void Dispose() {
        if(view!=IntPtr.Zero) { UnmapViewOfFile(view); view=IntPtr.Zero; }
        if(mapping!=IntPtr.Zero) { CloseHandle(mapping); mapping=IntPtr.Zero; }
    }
}
'@
function Quote-Arguments([string[]]$Values) {
    ($Values | ForEach-Object {
        if ($_ -match '["\r\n]' -or $_.EndsWith('\')) { throw 'Unsupported argument' }
        '"' + $_ + '"'
    }) -join ' '
}
$collector=$null; $child=$null; $observer=$null; $earlyExit=$false; $gameExitCode=$null
$cadence=$null; $cadenceProcess=$null; $cadenceSamples=@(); $cadenceNext=20.0
$oldEnvironment=$env:C3X_RENDERER_GAME_TEST_SAVE
$oldMode=$env:C3X_RENDERER_GAME_TEST_MODE
$oldCombat=$env:C3X_RENDERER_GAME_TEST_COMBAT
$oldPack=$env:C3X_RENDERER_UNIT_PACK
$oldInputTrace=$env:C3X_RENDERER_TRACE_INPUT
$traceEnvironment=@{}
foreach ($key in @('C3X_RENDERER_TRACE','C3X_RENDERER_TRACE_BUFFERED','C3X_RENDERER_TRACE_FILE','C3X_RENDERER_TRACE_MIB','C3X_RENDERER_SCENE_SAMPLES','C3X_RENDERER_SCENE_SHARPNESS')) {
    $traceEnvironment[$key]=[Environment]::GetEnvironmentVariable($key)
}
$ini=Join-Path $ConquestsDirectory 'conquests.ini'
$iniBytes=[IO.File]::ReadAllBytes($ini)
$cursor=New-Object RendererGameCommand+Point
$restoreCursor=[RendererGameCommand]::GetCursorPos([ref]$cursor)
try {
    # Restore these exact bytes in finally. Native menu/file-picker loading is
    # necessary because C3X reads the renderer setting during scenario load.
    $latestSave=Join-Path (Split-Path $save -Parent) ([IO.Path]::GetFileNameWithoutExtension($save))
    foreach ($item in @(@('Top Menu',$(if ($Scenario -eq 'newgame') {'1'} else {'3'})),@('Latest Save',$latestSave),@('WindowsFileBox','0'),@('PlayIntro','0'))) {
        if (-not [RendererGameCommand]::WritePrivateProfileString('Conquests',$item[0],$item[1],$ini)) { throw 'Cannot prepare native load menu.' }
    }
    $collector=Start-Process $debug -ArgumentList (Quote-Arguments @('--accepteula','--no-banner','--no-kernel','--duration',[string]($Seconds+30),'--max-lines','1000000','--log',(Join-Path $session 'renderer.log'),'--log-limit','128')) -PassThru -WindowStyle Hidden
    Start-Sleep -Milliseconds 400
    if ($collector.HasExited) { throw 'Diagnostic collector exited before launch.' }
    # Use direct process creation so the diagnostic environment reaches C3X.
    $env:C3X_RENDERER_GAME_TEST_SAVE=$save
    $env:C3X_RENDERER_GAME_TEST_MODE=$Scenario
    $env:C3X_RENDERER_GAME_TEST_COMBAT=$CombatCase
    $env:C3X_RENDERER_UNIT_PACK=$UnitPack
    if ($SceneSamples -gt 0) { $env:C3X_RENDERER_SCENE_SAMPLES=[string]$SceneSamples }
    if ($SceneSharpness -ge 0) { $env:C3X_RENDERER_SCENE_SHARPNESS=$SceneSharpness.ToString([cultureinfo]::InvariantCulture) }
    if ($Scenario -in @('mouse','zoom','hud','city')) { $env:C3X_RENDERER_TRACE_INPUT='1' }
    if ($MeasureCadence -and -not $ProfileRenderer) { $env:C3X_RENDERER_TRACE='0' }
    if ($ProfileRenderer) {
        $env:C3X_RENDERER_TRACE='2'
        $env:C3X_RENDERER_TRACE_BUFFERED='1'
        if ($Scenario -in @('turn-stress','debug-scroll')) {
            # Preserve failure evidence even when a stalled helper must be killed.
            $env:C3X_RENDERER_TRACE_BUFFERED='0'
            $env:C3X_RENDERER_TRACE_MIB='64'
        }
        $env:C3X_RENDERER_TRACE_FILE=Join-Path $session 'renderer-core.log'
    }
    $start=New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName=$game; $start.WorkingDirectory=$ConquestsDirectory; $start.UseShellExecute=$false
    $child=[System.Diagnostics.Process]::Start($start)
    $env:C3X_RENDERER_GAME_TEST_SAVE=$oldEnvironment
    $env:C3X_RENDERER_GAME_TEST_MODE=$oldMode
    $env:C3X_RENDERER_GAME_TEST_COMBAT=$oldCombat
    $env:C3X_RENDERER_UNIT_PACK=$oldPack
    $env:C3X_RENDERER_TRACE_INPUT=$oldInputTrace
    foreach ($key in $traceEnvironment.Keys) { [Environment]::SetEnvironmentVariable($key,$traceEnvironment[$key]) }
    Write-Host ('Scripted game PID='+$child.Id+' capture='+$session)
    $observer=Start-Process $witness -ArgumentList (Quote-Arguments @([string]$child.Id,(Join-Path $session 'window'),[string]$Seconds,[string]$SampleHz,'sampled-window-evidence')) -PassThru -WindowStyle Hidden -RedirectStandardError (Join-Path $session 'window-errors.log')
    $end=[DateTime]::UtcNow.AddSeconds($Seconds)
    $sent=0
    $started=[DateTime]::UtcNow
    $enterCount=0
    $cursorParked=$false
    $mouseIndex=0; $mouseHeld=$false; $mouseEvents=@()
    $mouseSteps=@(@(30,0,0,0x800,240),@(36,0,0,2),@(38,32,0,0),@(39,64,0,0),@(40,96,0,0),@(41,128,0,0),@(43,128,0,4))
    if ($Scenario -eq 'zoom') {
        $mouseSteps=@(@(30,0,0,0x800,120),@(32,0,0,0x800,120),@(34,0,0,0x800,-120),@(36,0,0,0x800,-120),@(38,0,0,0x800,240),@(38.15,0,0,0x800,-120),@(40,0,0,0x800,-240),@(42,0,0,0x800,40),@(42.2,0,0,0x800,40),@(42.4,0,0,0x800,40))
    }
    if ($Scenario -eq 'city') {
        $mouseSteps=@(@(54,0,0,0x800,240),@(56,-1100,0,0),@(58,0,0,0),@(66,0,0,0x800,-240),@(68,1100,0,0),@(70,0,0,0))
    }
    $interactionIndex=0
    $combatReadyAt=$null
    $combatAttackSent=$false
    $combatNextAttack=0.0
    $combatNextPrepare=36.0
    $failureCheck=20.0
    $interaction=@(@(28,13,'close-welcome'),@(36,90,'zoom-192'),@(39,0x86,'text-192'),@(43,90,'zoom-160'),@(46,0x86,'text-160'),@(50,90,'zoom-128'),@(53,0x86,'text-128'),@(57,0x66,'move-east'),@(65,0x70,'advisor'),@(74,27,'close-advisor'))
    if ($Scenario -in @('mouse','zoom')) {
        $interaction=@()
    }
    if ($Scenario -eq 'settler') { $interaction=@() }
    if ($Scenario -eq 'newgame') {
        $interaction=@(@(12,13,'accept-new-game-rules'),@(35,13,'start-new-game'),@(46,13,'dismiss-new-game-welcome'))
    }
    if ($Scenario -in @('debug','debug-scroll')) {
        # Exercise the existing C3X DEBUG shortcut through normal key events.
        $interaction=@(@(25,68,'debug-D'),@(26,69,'debug-E'),@(27,66,'debug-B'),
            @(28,85,'debug-U'),@(29,71,'debug-G'),@(30,38,'choose-debug-yes'),@(31,13,'confirm-debug'),
            @(35,0x87,'scroll-revealed-map'),@(39,68,'normal-D'),@(40,69,'normal-E'),
            @(41,66,'normal-B'),@(42,85,'normal-U'),@(43,71,'normal-G'),
            @(45,13,'dismiss-debug-off'),@(49,0x87,'scroll-normal-map'))
    }
    if ($Scenario -eq 'debug-scroll') {
        $interaction=$interaction[0..6]
        for ($step=0; $step -lt 32; ++$step) { $interaction+=,@((45+4*$step),0x87,('scroll-revealed-'+$step)) }
    }
    if ($Scenario -eq 'site-toggle') {
        $interaction=@(@(37,76,'open-city-site-picker'),@(39,13,'choose-off'),
            @(43,76,'reopen-city-site-picker'),@(45,40,'choose-player'),@(46,13,'accept-player'))
    }
    if ($Scenario -eq 'hud') {
        $interaction=@(@(31,66,'found-city'),@(34,13,'accept-city-name'),@(44,90,'city-native-zoom-out'),@(48,90,'city-native-zoom-in'),@(53,13,'close-new-city-screen'),@(61,90,'zoom-192'),@(64,0x86,'text-192'),@(68,0x87,'scroll-city-label'),@(72,90,'zoom-160'),@(75,0x86,'text-160'),@(79,0x87,'scroll-city-label-again'),@(83,90,'zoom-128'),@(86,0x86,'text-128'),@(92,0x70,'advisor'),@(101,27,'close-advisor'))
    }
    if ($Scenario -eq 'city') {
        $interaction=@(@(31,66,'found-city'),@(34,13,'accept-city-name'),@(44,90,'city-native-zoom-out'),@(60,90,'city-native-zoom-in'),@(77,13,'close-city-screen'))
    }
    if ($Scenario -eq 'turn') {
        $interaction=@(@(36,32,'skip-first-unit'),@(39,32,'skip-second-unit'),@(42,13,'end-turn'),@(65,32,'skip-first-unit-next-turn'),@(68,32,'skip-second-unit-next-turn'),@(71,13,'end-second-turn'))
    }
    if ($Scenario -eq 'turn-stress') {
        $interaction=@()
        for ($turn=0; $turn -lt 8; ++$turn) {
            $offset=29*$turn
            $interaction+=,@((36+$offset),0x66,'move-east')
            $interaction+=,@((39+$offset),0x64,'move-west')
            $interaction+=,@((42+$offset),0x66,'move-east-again')
            $interaction+=,@((45+$offset),0x64,'move-west-again')
            $interaction+=,@((48+$offset),32,'skip-remaining-first-unit')
            $interaction+=,@((51+$offset),32,'skip-remaining-second-unit')
            $interaction+=,@((54+$offset),13,('end-turn-'+($turn+1)))
        }
    }
    if ($Scenario -eq 'combat') {
        $interaction=,@(28,13,'close-welcome')
    }
    if ($Scenario -eq 'lifecycle') {
        $interaction=@(@(42,81,'return-first-game-to-menu'),@(44,40,'select-quit'),@(45,13,'confirm-quit'),@(55,13,'load-again'),@(58,13,'accept-save'),@(100,0x86,'text-second-game'),@(105,81,'return-second-game-to-menu'),@(107,40,'select-second-quit'),@(108,13,'confirm-second-quit'))
    }
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
            if ($Scenario -in @('turn-stress','debug-scroll') -and $elapsed -ge $failureCheck) {
                $failureCheck=$elapsed+5
                $recent=Get-Content -LiteralPath (Join-Path $session 'renderer.log') -Tail 1000
                if ($recent -match 'stage=async-publication-failed|stage=visual-failure|stage=native-operation-failed') {
                    Write-Host 'Renderer failure detected; preserving evidence and stopping the disposable test.'
                    break
                }
            }
            if ($MeasureCadence -and $elapsed -ge $cadenceNext) {
                $cadenceNext=$elapsed+$(if ($Scenario -eq 'zoom') {0.02} else {1.0})
                if ($cadenceProcess -and $cadenceProcess.HasExited) {
                    $cadence.Dispose(); $cadence=$null; $cadenceProcess=$null
                }
                if (-not $cadence) {
                    # Match the helper's named channel to this disposable game's
                    # PID. Never attach to another user's renderer session.
                    $channelPattern='--child\s+"(Local\\C3XScene_'+$child.Id+'_\d+)"'
                    $helper=Get-CimInstance Win32_Process -Filter "Name='C3XRendererHelper64.exe'" |
                        Where-Object { $_.CommandLine -match $channelPattern } | Select-Object -First 1
                    if ($helper -and $helper.CommandLine -match $channelPattern) {
                        $cadence=New-Object RendererCadenceReader ($Matches[1]+'_map')
                        $cadenceProcess=[System.Diagnostics.Process]::GetProcessById($helper.ProcessId)
                    }
                }
                if ($cadence) {
                    $child.Refresh(); $cadenceProcess.Refresh()
                    $cadenceSamples += [ordered]@{ qpc=[System.Diagnostics.Stopwatch]::GetTimestamp();
                        helper_pid=$cadenceProcess.Id; frames=$cadence.Frames; zoom_q16=$cadence.ZoomQ16;
                        elapsed_seconds=$elapsed;
                        game_private_bytes=$child.PrivateMemorySize64; game_working_bytes=$child.WorkingSet64;
                        helper_private_bytes=$cadenceProcess.PrivateMemorySize64; helper_working_bytes=$cadenceProcess.WorkingSet64;
                        game_cpu_seconds=$child.TotalProcessorTime.TotalSeconds; helper_cpu_seconds=$cadenceProcess.TotalProcessorTime.TotalSeconds;
                        foreground=([RendererGameCommand]::GetForegroundWindow() -eq $window) }
                }
            }
            if ($Scenario -in @('mouse','zoom','city') -and $mouseIndex -lt $mouseSteps.Count -and $elapsed -ge $mouseSteps[$mouseIndex][0]) {
                $step=$mouseSteps[$mouseIndex]
                if ($mouseIndex -eq 0) { [void][RendererGameCommand]::SetForegroundWindow($window) }
                if ([RendererGameCommand]::GetForegroundWindow() -ne $window) { throw 'Diagnostic game lost foreground before mouse input.' }
                $rect=New-Object RendererGameCommand+Rect
                $point=New-Object RendererGameCommand+Point
                [void][RendererGameCommand]::GetClientRect($window,[ref]$rect)
                $point.X=[int]($rect.Right/2)+$step[1]; $point.Y=[int]($rect.Bottom/2)+$step[2]
                [void][RendererGameCommand]::ClientToScreen($window,[ref]$point)
                $inputTicks=[System.Diagnostics.Stopwatch]::GetTimestamp()
                [void][RendererGameCommand]::SetCursorPos($point.X,$point.Y)
                $wheelDelta=if ($step.Count -gt 4) { [int]$step[4] } else { 0 }
                $wheelData=[uint32]([long]$wheelDelta -band 4294967295)
                if ($step[3]) { [RendererGameCommand]::mouse_event($step[3],0,0,$wheelData,[UIntPtr]::Zero) }
                $mouseEvents += [ordered]@{ qpc=$inputTicks; client_x=([int]($rect.Right/2)+$step[1]); client_y=([int]($rect.Bottom/2)+$step[2]); flags=$step[3]; wheel_delta=$wheelDelta }
                if ($step[3] -eq 2) { $mouseHeld=$true }
                if ($step[3] -eq 4) { $mouseHeld=$false }
                Write-Host ('Mouse command: '+($step -join ',')); ++$mouseIndex
            }
            $key=if ($Scenario -eq 'scroll') {0x87} else {0}
            if ($enterCount -lt 2 -and $elapsed -ge (5+2*$enterCount)) { $key=13; ++$enterCount }
            if ($Scenario -ne 'scroll' -and $interactionIndex -lt $interaction.Count -and $elapsed -ge $interaction[$interactionIndex][0]) {
                $key=$interaction[$interactionIndex][1]
                Write-Host ('Interaction command: '+$interaction[$interactionIndex][2])
                ++$interactionIndex
            }
            if ($Scenario -eq 'combat' -and $elapsed -ge 36 -and -not $combatAttackSent) {
                # Preparation is idempotent and can be rejected until the first
                # map is ready. Start the attack only after native confirmation.
                if ($null -eq $combatReadyAt) {
                    $liveLog=Get-Content -LiteralPath (Join-Path $session 'renderer.log') -Raw
                    if ($liveLog -match 'stage=scripted-combat-ready') { $combatReadyAt=$elapsed }
                    elseif ($elapsed -ge $combatNextPrepare) {
                        $key=0x84; $combatNextPrepare=$elapsed+3
                        Write-Host 'Interaction command: prepare-combat'
                    }
                } elseif ($elapsed -ge $combatReadyAt+4) {
                    $liveLog=Get-Content -LiteralPath (Join-Path $session 'renderer.log') -Raw
                    if ($liveLog -match 'stage=scripted-combat-start') { $combatAttackSent=$true }
                    elseif ($elapsed -ge $combatNextAttack) {
                        # The native hook rejects a command during a pending
                        # map handoff. Its start marker, not key delivery,
                        # acknowledges the once-only order.
                        $key=0x85; $combatNextAttack=$elapsed+3
                        Write-Host 'Interaction command: attack-east'
                    }
                }
            }
            if ($key -eq 0) {
                $pollMs=if ($Scenario -eq 'zoom') {20} else {1000}
                Start-Sleep -Milliseconds $pollMs; continue
            }
            if ($Scenario -eq 'lifecycle' -and $key -eq 81) {
                # Native Ctrl+Shift+Q returns to the menu; Escape exits the application.
                # Modifiers must reach the keyboard state used by JGL's handler.
                [void][RendererGameCommand]::SetForegroundWindow($window)
                if ([RendererGameCommand]::GetForegroundWindow() -ne $window) { throw 'Diagnostic game lost foreground before Ctrl+Shift+Q.' }
                try {
                    [RendererGameCommand]::keybd_event(0x11,0,0,[UIntPtr]::Zero)
                    [RendererGameCommand]::keybd_event(0x10,0,0,[UIntPtr]::Zero)
                    [RendererGameCommand]::keybd_event(81,0,0,[UIntPtr]::Zero)
                    Start-Sleep -Milliseconds 100
                } finally {
                    [RendererGameCommand]::keybd_event(81,0,2,[UIntPtr]::Zero)
                    [RendererGameCommand]::keybd_event(0x10,0,2,[UIntPtr]::Zero)
                    [RendererGameCommand]::keybd_event(0x11,0,2,[UIntPtr]::Zero)
                }
            } elseif ($Scenario -in @('debug','debug-scroll') -and $key -in @(68,69,66,85,71)) {
                # C3X recognizes the key-up history. Avoid triggering each
                # letter's native unit order before the full shortcut exists.
                [void][RendererGameCommand]::PostMessage($window,0x101,[IntPtr]$key,[IntPtr](-1073741823))
            } else {
                if (-not [RendererGameCommand]::PostMessage($window,0x100,[IntPtr]$key,[IntPtr]1)) { throw 'Cannot post diagnostic command.' }
                [void][RendererGameCommand]::PostMessage($window,0x101,[IntPtr]$key,[IntPtr](-1073741823))
            }
            ++$sent
        }
        Start-Sleep -Milliseconds 1000
    }
    if ($child.HasExited) { $earlyExit=$true; $gameExitCode=$child.ExitCode }
    Write-Host ('Posted diagnostic commands: '+$sent)
} finally {
    if ($cadence) { $cadence.Dispose(); $cadence=$null }
    if ($mouseHeld) { [RendererGameCommand]::mouse_event(4,0,0,0,[UIntPtr]::Zero) }
    $env:C3X_RENDERER_GAME_TEST_SAVE=$oldEnvironment
    $env:C3X_RENDERER_GAME_TEST_MODE=$oldMode
    $env:C3X_RENDERER_GAME_TEST_COMBAT=$oldCombat
    $env:C3X_RENDERER_UNIT_PACK=$oldPack
    $env:C3X_RENDERER_TRACE_INPUT=$oldInputTrace
    foreach ($key in $traceEnvironment.Keys) { [Environment]::SetEnvironmentVariable($key,$traceEnvironment[$key]) }
    if ($observer -and -not $observer.HasExited) {
        if (Test-Path (Join-Path $session 'window')) { Set-Content -LiteralPath (Join-Path $session 'window\stop.txt') -Value 'stop' }
        if (-not $observer.WaitForExit(5000)) { $observer.Kill(); $observer.WaitForExit() }
    }
    # Terminate only the process created above. The input save is a disposable
    # copy. No scenario saves its resulting gameplay.
    if ($child -and -not $child.HasExited) { $child.Kill(); $child.WaitForExit() }
    [IO.File]::WriteAllBytes($ini,$iniBytes)
    if ($restoreCursor) { [void][RendererGameCommand]::SetCursorPos($cursor.X,$cursor.Y) }
    if ($collector -and -not $collector.HasExited) {
        & $debug --stop | Out-Null
        if (-not $collector.WaitForExit(5000)) { $collector.Kill(); $collector.WaitForExit() }
    }
    if ((Get-FileHash -LiteralPath $SaveFile).Hash -ne $originalSaveHash) { throw 'Original input save changed.' }
}
if ($MeasureCadence) {
    [ordered]@{ qpc_frequency=[System.Diagnostics.Stopwatch]::Frequency; detailed_trace=[bool]$ProfileRenderer;
        meaning='Successful Renderer64 presentations, not physical scanout'; samples=$cadenceSamples } |
        ConvertTo-Json -Depth 4 | Set-Content (Join-Path $session 'cadence.json')
}
$log=Get-Content -LiteralPath (Join-Path $session 'renderer.log') -Raw
if ($Scenario -in @('mouse','zoom')) {
    [ordered]@{ qpc_frequency=[System.Diagnostics.Stopwatch]::Frequency; events=$mouseEvents } |
        ConvertTo-Json -Depth 4 | Set-Content (Join-Path $session 'mouse-events.json')
}
$steps=@([regex]::Matches($log,'stage=scripted-game-scroll step=(\d+)') | ForEach-Object {[int]$_.Groups[1].Value})
$turns=@([regex]::Matches($log,'stage=scripted-turn-end turn=(\d+)') | ForEach-Object {[int]$_.Groups[1].Value})
$moves=[regex]::Matches($log,'stage=motion-start[^\r\n]*result=1').Count
$combatReady=[regex]::Matches($log,'stage=scripted-combat-ready').Count
$combatFinished=[regex]::Matches($log,'stage=scripted-combat-end').Count
$textEvents=[regex]::Matches($log,'stage=scripted-game-map-text').Count
$cityZooms=@([regex]::Matches($log,'stage=city-native-zoom tile_width=(\d+)') | ForEach-Object {[int]$_.Groups[1].Value})
$cityAnchors=@([regex]::Matches($log,'stage=city-native-zoom[^\r\n]*city_anchor=(-?\d+,-?\d+)') | ForEach-Object {$_.Groups[1].Value})
$readyEvents=[regex]::Matches($log,'stage=render-done result=1').Count
$loadingMaps=[regex]::Matches($log,'stage=loading-map-complete result=1').Count
$unloadEvents=[regex]::Matches($log,'stage=scene-unloaded result=1').Count
$errors=@($log -split "`n" | Where-Object {$_ -match 'stage=native-operation-failed|stage=async-publication-failed|budget exceeded|stage=visual-failure|stage=worker-error|stage=unit-publication-failed|stage=unit-animation-failed|stage=motion-start[^\r\n]*result=[0235]|stage=scene-unload-failed|stage=(?:first-map-ready|loading-map-complete) result=[02345]|stage=assets-prepared ready=0'})
$debugReveals=([regex]::Matches($log,'stage=debug-reveal result=1')).Count
$debugHides=([regex]::Matches($log,'stage=debug-hide result=1')).Count
$windowResult=Join-Path $session 'window\finished.json'
$windowEvidence=if (Test-Path -LiteralPath $windowResult) { Get-Content -LiteralPath $windowResult -Raw | ConvertFrom-Json } else { $null }
$windowComplete=$null -ne $windowEvidence -and $windowEvidence.complete -and $windowEvidence.frames -gt 0
[ordered]@{ scenario=$Scenario; debug_reveals=$debugReveals; debug_hides=$debugHides; scene_samples=$SceneSamples; scene_sharpness=$SceneSharpness; combat_case=$CombatCase; unit_pack=$UnitPack; mouse_commands=$mouseIndex; completed_turns=$turns; accepted_moves=$moves; interaction_commands=$interactionIndex; combat_ready=$combatReady; combat_finished=$combatFinished; map_text_events=$textEvents; city_zoom_widths=$cityZooms; city_anchors=$cityAnchors; posted_commands=$sent; load_requested=($log -match 'stage=scripted-game-load'); scroll_steps=$steps;
    game_exited_early=$earlyExit; game_exit_code=$gameExitCode; first_map_ready=$readyEvents; loading_maps_ready=$loadingMaps; scenes_unloaded=$unloadEvents; native_failures=$errors; window_evidence=$windowEvidence; original_save_unchanged=$true; scope='Real Civ III diagnostic; window samples require visual review; not an FPS benchmark' } |
    ConvertTo-Json -Depth 4 | Set-Content (Join-Path $session 'result.json')
Write-Host ('Scripted diagnostic saved: '+$session)
if ($MeasureCadence -and $cadenceSamples.Count -lt 2) { Write-Error 'Insufficient renderer cadence samples; inspect cadence.json and the helper channel.'; exit 1 }
if ($Scenario -eq 'turn-stress' -and $moves -lt 8) { Write-Error 'Insufficient accepted unit movement; inspect the native movement records.'; exit 1 }
if ($MeasureCadence -and $Scenario -in @('turn-stress','debug-scroll')) {
    $tail=@($cadenceSamples | Where-Object { $_.elapsed_seconds -ge $cadenceSamples[-1].elapsed_seconds-10 })
    if ($tail.Count -lt 2 -or $tail[-1].frames -le $tail[0].frames) { Write-Error 'Renderer stopped presenting at the end of the stress run.'; exit 1 }
}
if (-not $windowComplete) { Write-Error 'Window evidence did not complete; inspect window-errors.log and window/finished.json.'; exit 1 }
if (($Scenario -in @('mouse','zoom','city') -and $mouseIndex -ne $mouseSteps.Count) -or ($Scenario -eq 'turn' -and $turns.Count -lt 2) -or ($Scenario -eq 'combat' -and ($combatReady -ne 1 -or $combatFinished -ne 1)) -or $earlyExit -or ($Scenario -eq 'scroll' -and $steps.Count -ne 32) -or ($Scenario -in @('settler','site-toggle','debug','debug-scroll','newgame','turn-stress') -and $readyEvents -lt 1) -or ($Scenario -in @('site-toggle','debug','debug-scroll','newgame','turn-stress') -and $interactionIndex -ne $interaction.Count) -or ($Scenario -in @('interaction','hud') -and ($interactionIndex -ne $interaction.Count -or $textEvents -ne 3)) -or ($Scenario -eq 'hud' -and (($cityZooms -join ',') -ne '64,128' -or $steps.Count -ne 2)) -or ($Scenario -eq 'lifecycle' -and ($interactionIndex -ne $interaction.Count -or $readyEvents -ne 2 -or $unloadEvents -lt 2 -or $textEvents -ne 1)) -or ($Scenario -eq 'city' -and ($interactionIndex -ne $interaction.Count -or ($cityZooms -join ',') -ne '64,128')) -or ($Scenario -in @('hud','city') -and ($cityAnchors.Count -ne 2 -or $cityAnchors[0] -ne $cityAnchors[1])) -or ($Scenario -eq 'debug' -and ($steps.Count -ne 2 -or $debugReveals -ne 1 -or $debugHides -ne 1)) -or ($Scenario -eq 'debug-scroll' -and ($steps.Count -ne 32 -or $debugReveals -ne 1 -or $debugHides -ne 0)) -or ($Scenario -eq 'turn-stress' -and ($turns.Count -lt 8 -or @($turns | Select-Object -Unique).Count -lt 8)) -or $errors.Count) { exit 1 }

# DebugView --stop may leave a nonzero native exit code after successful cleanup.
# Only the explicit diagnostic checks above determine this scenario result.
exit 0
