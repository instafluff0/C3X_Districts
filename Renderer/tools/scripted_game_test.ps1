# Bounded real-game renderer diagnostic, enabled only in the child environment.
param([Parameter(Mandatory=$true)][string]$SaveFile, [string]$ConquestsDirectory,
      [ValidateRange(35,360)][int]$Seconds = 75,
      [ValidateSet('scroll','interaction','lifecycle','combat','turn','mouse','zoom','zoom-out','near','units','volcanoes','resources','hud','city','settler','site-toggle','debug','debug-scroll','newgame','turn-stress','turn-scroll','unit-turn','unit-motion','research-turn','reveal-scroll','route-city','route-zoom','city-builds','navigation','camera','forest-shadow','pedia-hover','foreign-advisor','diplomacy-hover','domestic-slider','screen-sweep')][string]$Scenario = 'scroll',
      [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$UnitPack='UnitAnimationFidelity', [ValidateSet('melee','victory','retreat','bombard','army','air','capture')][string]$CombatCase='melee', [ValidateRange(1,10)][int]$SampleHz = 2,
      [switch]$ProfileRenderer, [switch]$MeasureCadence, [switch]$NativeRendering, [switch]$EnableDistricts,
      [ValidatePattern('^-?[0-9]+,-?[0-9]+$')][string]$SweepLeader='-271,-154',
      [ValidateSet(0,1,2)][int]$SceneSamples=0, [ValidateRange(-1,1)][double]$SceneSharpness=-1,
      # Renderer diagnostics for the game child only, e.g. C3X_SANDBOX_PASS_COUNTS=1;C3X_RENDERER_PROFILE=1
      [ValidatePattern('^$|^C3X_[A-Z0-9_]+=[A-Za-z0-9_.,-]*(;C3X_[A-Z0-9_]+=[A-Za-z0-9_.,-]*)*$')][string]$RendererOptions='',
      # Camera route for F24 steps (x,y,zoom width in native pixels; see parse_custom_renderer_test_route)
      [ValidatePattern('^$|^[0-9]+,[0-9]+,[0-9]+(;[0-9]+,[0-9]+,[0-9]+)*$')][string]$Route='', [ValidateRange(0,20)][int]$LoadWarnings=0)
$ErrorActionPreference = 'Stop'
# 'volcanoes' is the units camera script aimed at the 1498 AD save's volcano groups;
# 'resources' aims it at that save's ivory near rivers, fish and whales, and coastal gems.
$volcanoTargets=$Scenario -eq 'volcanoes'
$resourceTargets=$Scenario -eq 'resources'
if ($volcanoTargets -or $resourceTargets) { $Scenario='units' }
if ($Scenario -eq 'camera' -and $Seconds -lt 100) { throw 'camera requires at least 100 seconds.' }
if ($Scenario -in @('navigation','zoom-out','near','units') -and $Seconds -lt 120) { throw 'navigation requires at least 120 seconds.' }
if ($Scenario -eq 'city-builds' -and $Seconds -lt 180) { throw 'city-builds requires at least 180 seconds.' }
if ($Scenario -eq 'pedia-hover' -and $Seconds -lt 85) { throw 'pedia-hover requires at least 85 seconds.' }
if ($Scenario -eq 'foreign-advisor' -and $Seconds -lt 95) { throw 'foreign-advisor requires at least 95 seconds.' }
if ($Scenario -eq 'diplomacy-hover' -and $Seconds -lt 110) { throw 'diplomacy-hover requires at least 110 seconds.' }
if ($Scenario -eq 'domestic-slider' -and $Seconds -lt 100) { throw 'domestic-slider requires at least 100 seconds.' }
if ($Scenario -eq 'screen-sweep' -and $Seconds -lt 210) { throw 'screen-sweep requires at least 210 seconds.' }
if ($Scenario -eq 'route-city' -and $Seconds -lt 245) { throw 'route-city requires at least 245 seconds.' }
if ($Scenario -eq 'route-zoom' -and $Seconds -lt 120) { throw 'route-zoom requires at least 120 seconds.' }
if ($Scenario -in @('research-turn','reveal-scroll') -and $Seconds -lt 200) { throw 'research-turn requires at least 200 seconds.' }
if ($Scenario -eq 'unit-motion' -and $Seconds -lt 100) { throw 'unit-motion requires at least 100 seconds.' }
if ($Scenario -eq 'unit-turn' -and $Seconds -lt 130) { throw 'unit-turn requires at least 130 seconds.' }
if ($Scenario -eq 'turn-scroll' -and $Seconds -lt 200) { throw 'turn-scroll requires at least 200 seconds.' }
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
# Windows may retain an exited crash record; only live processes own a session.
if (Get-Process Civ3Conquests,dbgviewcli,dbgviewcli64,dbgviewcli64a -ErrorAction SilentlyContinue | Where-Object { -not $_.HasExited }) { throw 'Close the game and any debug collectors before a scripted test.' }
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
// UI-thread responsiveness with or without the renderer: a background thread
// sends WM_NULL every interval; each reply waits until Civ III's thread pumps
// messages, so a blocked or busy UI thread shows as a long round trip.
public static class RendererUiPing {
    [DllImport("user32.dll")] static extern IntPtr SendMessageTimeout(IntPtr window, uint message, IntPtr key, IntPtr detail, uint flags, uint timeout, out IntPtr result);
    static System.Threading.Thread thread; static volatile bool running;
    static readonly System.Collections.Generic.List<long[]> samples = new System.Collections.Generic.List<long[]>();
    public static void Start(IntPtr window, int interval) {
        if (thread != null) return;
        running = true;
        thread = new System.Threading.Thread(() => {
            while (running) {
                long sent = System.Diagnostics.Stopwatch.GetTimestamp(); IntPtr reply;
                bool answered = SendMessageTimeout(window, 0, IntPtr.Zero, IntPtr.Zero, 2, 5000, out reply) != IntPtr.Zero;
                long received = System.Diagnostics.Stopwatch.GetTimestamp();
                lock (samples) samples.Add(new long[] { sent, received, answered ? 1 : 0 });
                System.Threading.Thread.Sleep(interval);
            }
        });
        thread.IsBackground = true; thread.Start();
    }
    public static long[][] Stop() { running = false; if (thread != null) thread.Join(6000); lock (samples) return samples.ToArray(); }
}
// Read the helper's existing successful-presentation counter. This maps only
// the IPC header read-only; it never requests pixels or submits renderer work.
public sealed class RendererCadenceReader : IDisposable {
    // Mirrored from helper_trial/scene_wire.h; executable test checks this ABI.
    public const int WireVersion = 19, FrameOffset = 228, ZoomOffset = 232;
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] static extern IntPtr OpenFileMapping(uint access, bool inherit, string name);
    [DllImport("kernel32.dll")] static extern IntPtr MapViewOfFile(IntPtr mapping, uint access, uint high, uint low, UIntPtr bytes);
    [DllImport("kernel32.dll")] static extern bool UnmapViewOfFile(IntPtr view);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    IntPtr mapping, view;
    public RendererCadenceReader(string name) {
        mapping=OpenFileMapping(4, false, name);
        if(mapping!=IntPtr.Zero) view=MapViewOfFile(mapping,4,0,0,(UIntPtr)(ZoomOffset+4));
        if(view==IntPtr.Zero) { Dispose(); throw new InvalidOperationException("Cannot read diagnostic helper telemetry"); }
        // Later versions add independent mailboxes after these counters; the
        // accepted controls retain this read-only telemetry ABI.
        int version=Marshal.ReadInt32(view,4);
        if(Marshal.ReadInt32(view)!=0x32483343 || (version!=WireVersion && version!=18 && version!=16 && version!=15 && version!=14 && version!=13 && version!=12 && version!=11)) {
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
$optionEnvironment=@{}
foreach ($key in @('C3X_RENDERER_TRACE','C3X_RENDERER_TRACE_BUFFERED','C3X_RENDERER_TRACE_FILE','C3X_RENDERER_TRACE_MIB','C3X_RENDERER_SCENE_SAMPLES','C3X_RENDERER_SCENE_SHARPNESS','C3X_RENDERER_GAME_TEST_ROUTE')) {
    $traceEnvironment[$key]=[Environment]::GetEnvironmentVariable($key)
}
$ini=Join-Path $ConquestsDirectory 'conquests.ini'
$iniBytes=[IO.File]::ReadAllBytes($ini)
# -NativeRendering: the same scenario with Civ III's own 2D rendering, for an
# A/B against Renderer64. enable_custom_rendering is off, and so are the
# day/night and seasonal cycles: vanilla has neither, and without Renderer64
# they load 2D cycle art (absent here; its error dialog stops the load).
# The exact config bytes are restored in finally.
# -EnableDistricts: scenario saves with Wonder Districts (the 1845 AD save)
# need enable_districts = true when their scenario leaves it to this config.
$c3xConfig=Join-Path (Split-Path $renderer -Parent) 'custom.c3x_config.ini'
$c3xConfigBytes=$null
if ($NativeRendering -or $EnableDistricts) {
    $c3xConfigBytes=[IO.File]::ReadAllBytes($c3xConfig)
    $latin=[Text.Encoding]::GetEncoding(28591)
    $configText=$latin.GetString($c3xConfigBytes)
    if ($NativeRendering -and $configText -notmatch '(?m)^enable_custom_rendering\s*=\s*true\s*$') { throw 'NativeRendering needs enable_custom_rendering = true in custom.c3x_config.ini.' }
    if ($EnableDistricts -and $configText -notmatch '(?m)^enable_districts\s*=') { throw 'EnableDistricts needs an enable_districts line in custom.c3x_config.ini.' }
}
$cursor=New-Object RendererGameCommand+Point
$restoreCursor=[RendererGameCommand]::GetCursorPos([ref]$cursor)
try {
    if ($c3xConfigBytes) {
        $runText=$configText
        if ($NativeRendering) { $runText=$runText -replace '(?m)^enable_custom_rendering\s*=\s*true(\s*)$','enable_custom_rendering = false$1' `
            -replace '(?m)^(day_night_cycle_mode|seasonal_cycle_mode)\s*=[^\r\n]*','$1 = off' }
        if ($EnableDistricts) { $runText=$runText -replace '(?m)^enable_districts\s*=[^\r\n]*','enable_districts = true' }
        [IO.File]::WriteAllBytes($c3xConfig,$latin.GetBytes($runText))
    }
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
    if ($Scenario -eq 'navigation') {
        $env:C3X_RENDERER_GAME_TEST_MODE='matched-route'
        $env:C3X_RENDERER_GAME_TEST_ROUTE='2464,0,128;2464,692,128;3760,692,384;2464,692,384'
    }
    if ($Route) {
        $env:C3X_RENDERER_GAME_TEST_MODE='matched-route'
        $env:C3X_RENDERER_GAME_TEST_ROUTE=$Route
    }
    $env:C3X_RENDERER_GAME_TEST_COMBAT=$CombatCase
    $env:C3X_RENDERER_UNIT_PACK=$UnitPack
    if ($SceneSamples -gt 0) { $env:C3X_RENDERER_SCENE_SAMPLES=[string]$SceneSamples }
    if ($SceneSharpness -ge 0) { $env:C3X_RENDERER_SCENE_SHARPNESS=$SceneSharpness.ToString([cultureinfo]::InvariantCulture) }
    if ($Scenario -in @('mouse','zoom','zoom-out','near','units','hud','city','route-city','route-zoom','city-builds','navigation','camera','reveal-scroll','pedia-hover','foreign-advisor','diplomacy-hover','domestic-slider')) { $env:C3X_RENDERER_TRACE_INPUT='1' }
    if ($MeasureCadence -and -not $ProfileRenderer) { $env:C3X_RENDERER_TRACE='0' }
    if ($ProfileRenderer) {
        $env:C3X_RENDERER_TRACE='2'
        $env:C3X_RENDERER_TRACE_BUFFERED='1'
        if ($Scenario -in @('turn-scroll','unit-turn','unit-motion','research-turn','reveal-scroll','route-city','route-zoom','city-builds','zoom-out','near','units','city','pedia-hover','diplomacy-hover','domestic-slider','screen-sweep')) { $env:C3X_RENDERER_TRACE_MIB='64' }
        if ($Scenario -in @('turn-stress','debug-scroll','route-city','city-builds')) {
            # Preserve failure evidence even when a stalled helper must be killed.
            $env:C3X_RENDERER_TRACE_BUFFERED='0'
            $env:C3X_RENDERER_TRACE_MIB='64'
        }
        $env:C3X_RENDERER_TRACE_FILE=Join-Path $session 'renderer-core.log'
    }
    if ($RendererOptions) {
        foreach ($pair in $RendererOptions.Split(';')) {
            $key,$value=$pair.Split('=',2)
            if (-not $optionEnvironment.ContainsKey($key)) { $optionEnvironment[$key]=[Environment]::GetEnvironmentVariable($key) }
            [Environment]::SetEnvironmentVariable($key,$value)
        }
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
    foreach ($key in $optionEnvironment.Keys) { [Environment]::SetEnvironmentVariable($key,$optionEnvironment[$key]) }
    Write-Host ('Scripted game PID='+$child.Id+' capture='+$session)
    $observer=Start-Process $witness -ArgumentList (Quote-Arguments @([string]$child.Id,(Join-Path $session 'window'),[string]$Seconds,[string]$SampleHz,'sampled-window-evidence')) -PassThru -WindowStyle Hidden -RedirectStandardError (Join-Path $session 'window-errors.log')
    $end=[DateTime]::UtcNow.AddSeconds($Seconds)
    $sent=0
    $started=[DateTime]::UtcNow
    $enterCount=0
    $cursorParked=$false
    $navigationReadyAt=$null; $navigationCheck=20.0
    $mouseIndex=0; $mouseHeld=$false; $mouseEvents=@()
    # UI stalls count from the first scripted input: loading a save blocks
    # Civ III's message loop for seconds by design.
    $scriptedInputQpc=$null
    # Posted keys with QPC stamps: input-to-move latency starts here.
    $keyEvents=@()
    $mouseSteps=@(@(30,0,0,0x800,240),@(36,0,0,2),@(38,32,0,0),@(39,64,0,0),@(40,96,0,0),@(41,128,0,0),@(43,128,0,4))
    if ($Scenario -eq 'zoom') {
        $mouseSteps=@(@(30,0,0,0x800,120),@(32,0,0,0x800,120),@(34,0,0,0x800,-120),@(36,0,0,0x800,-120),@(38,0,0,0x800,240),@(38.15,0,0,0x800,-120),@(40,0,0,0x800,-240),@(42,0,0,0x800,40),@(42.2,0,0,0x800,40),@(42.4,0,0,0x800,40))
    }
    if ($Scenario -eq 'zoom-out') {
        # Relative to first-map readiness, through every outward stop, a
        # rapid reversal, both scroll axes and the native Z input path.
        $mouseSteps=@(@(30,0,0,0x800,-120),@(33,0,0,0x800,-120),@(36,0,0,0x800,-120),@(39,0,0,0x800,-120),
            @(43,0,0,0x800,120),@(43.2,0,0,0x800,-120),@(46,0,0,0x800,1200),@(50,0,0,0x800,-720),
            @(54,0,0,0x800,-480),@(58,0,-620,0),@(63,0,0,0),@(67,1118,0,0),@(72,0,0,0),@(76,0,0,0x800,480),@(84,0,0,0x800,120))
    }
    if ($Scenario -eq 'near') {
        # Relative to first-map readiness, at the zooms players use most:
        # 1x idle, 1x edge scroll on both axes, two minimap jumps, notches to
        # 2x and a 2x scroll, 3x scroll and idle, notches back and 1x idle.
        $mouseSteps=@(@(40,1118,0,0),@(46,0,0,0),@(48,0,-620,0),@(53,0,0,0),
            @(56,-1000,434,2),@(56.1,-1000,434,4),@(60,-720,494,2),@(60.1,-720,494,4),@(63,0,0,0),
            @(64,0,0,0x800,120),@(65,0,0,0x800,120),@(66,0,0,0x800,120),@(67,0,0,0x800,120),
            @(70,1118,0,0),@(75,0,0,0),@(77,0,0,0x800,240),@(80,-1118,0,0),@(85,0,0,0),
            @(92,0,0,0x800,-120),@(93,0,0,0x800,-120),@(94,0,0,0x800,-120),@(95,0,0,0x800,-120),
            @(96,0,0,0x800,-120),@(97,0,0,0x800,-120),@(110,0,0,0))
    }
    if ($Scenario -eq 'units') {
        # Unit close-ups on the 1498 AD save (2240-wide client): idle 1x, then
        # minimap jumps (about 18x20 screen pixels per minimap pixel) to the
        # Nagoya Bomber, the two ships off the southern peninsula and the
        # north-eastern harbour, each at 2x and then 3x. Zoom keeps the center.
        $in=@(0..3|%{0.5*$_}); $out=@(0..5|%{0.5*$_})
        # Resource close-ups go two steps further in (the save opens zoomed out).
        if ($resourceTargets) { $in=@(0..5|%{0.5*$_}); $out=@(0..7|%{0.5*$_}) }
        $mouseSteps=@(@(40,0,0,0))
        $t=46
        # Volcanoes: minimap pixel (33+3.5x, 970+1.62y) for map tile (x,y) on that
        # client; the southern group near (95,91), the south-western group near
        # (40,103) and the western pair near (31,40).
        # Resources: ivory at (91,85), fish and whales around (119,94), and gems on
        # mountains at the coast at (29,103) and (49,107).
        $targets=if ($volcanoTargets) {@(@(-754,521),@(-947,541),@(-978,439))} elseif ($resourceTargets) {@(@(-768,512),@(-671,527),@(-985,541),@(-916,547))} else {@(@(-738,498),@(-671,527),@(-661,553),@(-661,505))}
        foreach($target in $targets){
            $mouseSteps+=@(@($t,$target[0],$target[1],2),@(($t+0.1),$target[0],$target[1],4),@(($t+2),0,0,0))
            $mouseSteps+=$in|%{,@(($t+4+$_),0,0,0x800,120)}; $mouseSteps+=,@(($t+12),0,0,0x800,240)
            $mouseSteps+=$out|%{,@(($t+20+$_),0,0,0x800,-120)}
            $t+=26
        }
        $mouseSteps+=,@($t,0,0,0)
    }
    if ($Scenario -eq 'navigation') {
        # 2240x1260 disposable witness: variable edge proximity, both poles,
        # wrapped horizontal motion, and zoom-out re-clamping without input.
        $mouseSteps=@(@(30,0,0,0x800,720),@(33,0,-599,0),@(37,0,-614,0),@(41,0,-629,0),@(46,0,0,0),@(48,0,0,0x800,-720),@(52,0,0,0x800,720),@(54,0,628,0),@(61,0,0,0),@(66,1090,0,0),@(70,1118,0,0),@(74,0,0,0),@(76,0,0,0x800,-720))
    }
    if ($Scenario -eq 'camera') {
        # Relative to the combat-ready marker: pan/zoom/reverse together,
        # then hold the edge and keep zooming throughout native combat.
        $mouseSteps=@(@(.2,0,0,0x800,720),@(.6,1118,0,0),@(1,1118,0,0x800,-240),@(1.3,1118,0,0x800,240),@(1.6,1118,0,0x800,-120),@(1.9,1118,0,0x800,120),@(7.2,0,0,0),@(9,1118,0,0x800,-240),@(10,1118,0,0x800,240),@(11,1118,0,0x800,-120),@(12,1118,0,0x800,120),@(17,0,0,0),@(18,0,0,0x800,-720),@(20,1118,0,0x800,120),@(21,1118,0,0x800,240),@(22,1118,0,0x800,-120),@(23,0,0,0))
    }
    if ($Scenario -eq 'city') {
        $mouseSteps=@(@(54,0,0,0x800,240),@(56,-1100,0,0),@(58,0,0,0),@(66,0,0,0x800,-240),@(68,1100,0,0),@(70,0,0,0))
    }
    if ($Scenario -eq 'pedia-hover') {
        # Circle the pointer over the new city screen's lower-centre production
        # box, where Civ III swaps to its software Civilopedia cursor: every
        # move restores and re-saves the screen under the cursor sprite.
        $mouseSteps=@()
        for($n=0;$n -lt 600;++$n){$phase=2*[Math]::PI*$n/40
            $mouseSteps+=,@((40+0.05*$n),[int](90+28*[Math]::Cos($phase)),[int](255+18*[Math]::Sin($phase)),0)}
    }
    if ($Scenario -eq 'foreign-advisor') {
        # Foreign Advisor (F4) on the user's 3400 BC autosave with known
        # leaders: hover the advisor heads and leader portraits, switch tabs,
        # then click advisor heads (the third opens Military). Its large
        # title text once exceeded the GPU text raster cap; the CPU fallback
        # then needed a readback Renderer64 rejects, freezing every UI update.
        $mouseSteps=@()
        $heads=@(-114,-51,10,73,134,196)
        for($n=0;$n -lt $heads.Count;++$n){$mouseSteps+=,@((44+1.5*$n),-482,$heads[$n],0)}
        $mouseSteps+=@(@(54,-271,-154,0),@(55.5,-92,-226,0))
        foreach($tab in @(@(57,321),@(60,398),@(63,252))){$mouseSteps+=@(@($tab[0],$tab[1],-84,0),@(($tab[0]+0.3),$tab[1],-84,2),@(($tab[0]+0.4),$tab[1],-84,4))}
        foreach($head in @(@(66,-51),@(69,10),@(72,-114))){$mouseSteps+=@(@($head[0],-482,$head[1],0),@(($head[0]+0.3),-482,$head[1],2),@(($head[0]+0.4),-482,$head[1],4))}
        for($n=0;$n -lt 40;++$n){$phase=2*[Math]::PI*$n/20
            $mouseSteps+=,@((75+0.1*$n),[int](120*[Math]::Cos($phase)),[int](60*[Math]::Sin($phase)),0)}
        $mouseSteps+=,@(80,0,0,0)
    }
    if ($Scenario -eq 'diplomacy-hover') {
        # The user's 3400 BC autosave: in the Foreign Advisor, the first click
        # on Bismarck's portrait selects him and a second click opens talks.
        # Choose "We would like to propose a deal...", open our Technology
        # category, then circle its technology links for 30 s (Civ III's software Civilopedia
        # cursor) and plain dialogue text for 10 s (the ordinary cursor).
        # The animated leader head presents its area ~55 times a second.
        $mouseSteps=@(@(44,-271,-154,0),@(44.3,-271,-154,2),@(44.36,-271,-154,4),
            @(44.44,-271,-154,2),@(44.5,-271,-154,4),
            @(50,-700,-89,0),@(50.3,-700,-89,2),@(50.4,-700,-89,4),
            @(52.5,-1036,-345,0),@(52.8,-1036,-345,2),@(52.9,-1036,-345,4))
        for($n=0;$n -lt 600;++$n){$phase=2*[Math]::PI*$n/30
            $mouseSteps+=,@((54+0.05*$n),[int](-1000+30*[Math]::Cos($phase)),[int](-312+17*[Math]::Sin($phase)),0)}
        for($n=0;$n -lt 200;++$n){$phase=2*[Math]::PI*$n/30
            $mouseSteps+=,@((85+0.05*$n),[int](-660+40*[Math]::Cos($phase)),[int](-150+20*[Math]::Sin($phase)),0)}
        $mouseSteps+=,@(95.5,0,0,0)
    }
    if ($Scenario -eq 'domestic-slider') {
        # Domestic Advisor (F1) on the user's 3400 BC autosave: drag the
        # science slider (flask at 50%) for 20 s and the luxury slider
        # (smiley at 0%) for 10 s at 20 moves a second between 0% and the
        # Despotism limit, then leave the screen idle for 14 s.
        $mouseSteps=@(@(44,154,-286,0),@(45,154,-286,2))
        for($n=1;$n -le 400;++$n){$mouseSteps+=,@((45+0.05*$n),[int](115+55*[Math]::Sin(2*[Math]::PI*0.05*$n/4)),-285,0)}
        $mouseSteps+=@(@(65.1,115,-285,4),@(66,73,-238,0),@(67,73,-238,2))
        for($n=1;$n -le 200;++$n){$mouseSteps+=,@((67+0.05*$n),[int](100-40*[Math]::Cos(2*[Math]::PI*0.05*$n/4)),-238,0)}
        $mouseSteps+=@(@(77.1,60,-238,4),@(78,0,0,0))
    }
    if ($Scenario -eq 'screen-sweep') {
        # Every Civ III 2D screen in turn: open it, move the pointer over it
        # for 4 s (hover redraws, tooltips, the software cursor), close it.
        # Each entry: label, opening key (0 for none), then optional clicks:
        # 'click' x,y or 'double' x,y one second after the key, and the
        # number of Esc presses that close it. Fails on GPU refusals, native
        # failures or UI stalls (end of this script).
        $leader=$SweepLeader.Split(',')
        $sweepScreens=@(@('domestic',0x70,'',0,0,1),@('trade',0x71,'',0,0,1),@('military',0x72,'',0,0,1),@('foreign',0x73,'',0,0,1),
            @('cultural',0x74,'',0,0,1),@('science',0x75,'',0,0,1),@('wonders',0x76,'',0,0,1),@('victory',0x77,'',0,0,1),
            @('palace',0x78,'',0,0,1),@('spaceship',0x79,'',0,0,1),@('demographics',0x7A,'',0,0,1),
            @('city',0x70,'click',-366,-61,2),@('diplomacy',0x73,'double',[int]$leader[0],[int]$leader[1],2),
            @('hud-menu',0,'click',-1084,-571,1),@('hud-pedia',0,'click',-1046,-571,1),@('hud-advisors',0,'click',-1009,-571,1),
            @('quit-dialog',27,'',0,0,1))
        $mouseSteps=@();$sweepInteraction=@();$sweepSchedule=@();$at=40
        foreach($screen in $sweepScreens){
            if($screen[1]){$sweepInteraction+=,@($at,$screen[1],('open-'+$screen[0]))}
            $x=$screen[3];$y=$screen[4];$click=if($screen[1]){$at+1}else{$at}
            if($screen[2] -eq 'click'){$mouseSteps+=@(@($click,$x,$y,0),@(($click+0.1),$x,$y,2),@(($click+0.2),$x,$y,4))}
            if($screen[2] -eq 'double'){$mouseSteps+=@(@($click,$x,$y,0),@(($click+0.1),$x,$y,2),@(($click+0.16),$x,$y,4),@(($click+0.24),$x,$y,2),@(($click+0.3),$x,$y,4))}
            $hover=$at+2.5
            for($m=0;$m -lt 80;++$m){$phase=2*[Math]::PI*$m/40
                $mouseSteps+=,@(($hover+0.05*$m),[int](380*[Math]::Cos($phase)),[int](260*[Math]::Sin($phase)),0)}
            for($e=0;$e -lt $screen[5];++$e){$sweepInteraction+=,@(($at+7+1.5*$e),27,('close-'+$screen[0]))}
            $sweepSchedule+=[ordered]@{ screen=$screen[0]; open=$at; hover=$hover; close=$at+7+1.5*($screen[5]-1) }
            $at+=9+1.5*($screen[5]-1)}
        $sweepInteraction+=,@(($at+1),0x87,'scroll-after-sweep')
    }
    if ($Scenario -eq 'forest-shadow') {
        # research-turn at the closest zoom: zoom to 3x, found the city, close
        # its screen (which returns to 1x), zoom back to 3x, move, end the turn
        # and idle. Compare forest shading around the new city.
        $mouseSteps=@(@(52,0,0,0x800,720),@(90,0,0,0x800,720))
    }
    $interactionIndex=0
    $combatReadyAt=$null
    $combatAttackSent=$false
    $combatZoomKeySent=$false
    $combatNextAttack=0.0
    $combatNextPrepare=36.0
    $failureCheck=20.0
    $interaction=@(@(28,13,'close-welcome'),@(36,90,'zoom-192'),@(39,0x86,'text-192'),@(43,90,'zoom-160'),@(46,0x86,'text-160'),@(50,90,'zoom-128'),@(53,0x86,'text-128'),@(57,0x66,'move-east'),@(65,0x70,'advisor'),@(74,27,'close-advisor'))
    if ($Scenario -in @('mouse','zoom','zoom-out','near','units','route-city','route-zoom','city-builds','navigation','camera','reveal-scroll')) {
        $interaction=@()
    }
    if ($Scenario -eq 'navigation') { $interaction=@(@(28,0x87,'north-edge'),@(50,0x87,'south-edge'),@(63,0x87,'wrap-edge'),@(75,0x87,'return-interior')) }
    if ($Scenario -eq 'route-city') {
        # Hold a destination below the selected unit in the 3700 BC witness.
        $mouseSteps=,@(70,-192,355,2)
        for($n=0;$n -lt 600;++$n){
            $phase=2*[Math]::PI*$n/80
            $mouseSteps+=,@((71+0.1*$n),[int](-192+128*[Math]::Cos($phase)),[int](374+64*[Math]::Sin($phase)),0)
        }
        $mouseSteps+=,@(132,-192,355,0)
        $mouseSteps+=,@(134,-192,355,4)
        # Open the observed city, then cycle its production choice.
        $mouseSteps+=,@(145,0,140,2)
        $mouseSteps+=,@(145.08,0,140,4)
        $mouseSteps+=,@(145.18,0,140,2)
        $mouseSteps+=,@(145.26,0,140,4)
        $mouseSteps+=,@(155,450,180,2)
        $mouseSteps+=,@(155.1,450,180,4)
        $interaction=@()
        for($n=0;$n -lt 50;++$n){$interaction+=,@((158+0.4*$n),$(if($n%8 -lt 4){40}else{38}),'cycle-production')}
        $interaction+=,@(220,13,'accept-production')
        $interaction+=,@(230,13,'close-city')
    }
    if ($Scenario -eq 'route-zoom') {
        # The route-city witness at 2x and 3x: hold a pathfinder destination
        # (route lines, target circle and turn count) while the view is zoomed.
        # Zoom keeps the screen center, so the held points stay near the unit.
        # One hold spans 2x, the zoom to 3x and 3x, so the route also redraws
        # during the transition; releasing earlier spent the unit's move.
        $mouseSteps=@(@(62,0,0,0x800,120),@(63,0,0,0x800,120),@(64,0,0,0x800,120),@(65,0,0,0x800,120),@(70,-96,178,2))
        for($n=0;$n -lt 140;++$n){$phase=2*[Math]::PI*$n/30;$scale=$(if($n -lt 70){1.0}else{1.5})
            $mouseSteps+=,@((71+0.1*$n),[int]($scale*(-96+48*[Math]::Cos($phase))),[int]($scale*(178+24*[Math]::Sin($phase))),0)}
        $mouseSteps+=,@(78.05,0,0,0x800,120);$mouseSteps+=,@(78.55,0,0,0x800,120)
        $mouseSteps=@($mouseSteps|Sort-Object {$_[0]})
        $mouseSteps+=,@(86,-144,267,0);$mouseSteps+=,@(87,-144,267,4);$mouseSteps+=,@(100,0,0,0)
        $interaction=@()
    }
    if ($Scenario -eq 'city-builds') {
        # Same small-world capital, but exercise actual pointer hover, wheel,
        # and item clicks: arrow-key delivery is not evidence of item selection.
        $mouseSteps=@(@(70,0,140,2),@(70.08,0,140,4),@(70.18,0,140,2),@(70.26,0,140,4),
            @(80,450,180,2),@(80.08,450,180,4))
        for($n=0;$n -lt 6;++$n){$mouseSteps+=,@((85+0.8*$n),395,(-172+42*$n),0)}
        $mouseSteps+=,@(91,395,-90,0x800,-120)
        $mouseSteps+=,@(92,395,-90,0x800,120)
        $mouseSteps+=,@(94,395,-172,2)
        $mouseSteps+=,@(94.08,395,-172,4)
        for($n=0;$n -lt 20;++$n){
            $t=100+3*$n
            # Selecting an item leaves this chooser open. Clicking the portrait
            # again toggles it closed and would miss every second test item.
            $mouseSteps+=,@($t,395,(-172+42*($n%6)),0)
            $mouseSteps+=,@(($t+0.08),396,(-172+42*($n%6)),0)
            $mouseSteps+=,@(($t+0.7),395,(-172+42*($n%6)),2)
            $mouseSteps+=,@(($t+0.78),395,(-172+42*($n%6)),4)
        }
        $interaction=@(@(166,27,'close-build-chooser'),@(170,13,'close-city'))
    }
    if ($Scenario -eq 'zoom-out') { $interaction=,@(80,90,'world-z-outward') }
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
    if ($Scenario -eq 'pedia-hover') {
        $interaction=@(@(31,66,'found-city'),@(34,13,'accept-city-name'),@(75,13,'close-city-screen'))
    }
    if ($Scenario -eq 'foreign-advisor') {
        $interaction=@(@(40,0x73,'open-foreign-advisor'),@(84,27,'close-foreign-advisor'),@(88,0x87,'scroll-after-advisor'))
    }
    if ($Scenario -eq 'screen-sweep') { $interaction=$sweepInteraction }
    if ($Scenario -eq 'domestic-slider') {
        $interaction=@(@(40,0x70,'open-domestic-advisor'),@(92,27,'close-domestic-advisor'),@(96,0x87,'scroll-after-advisor'))
    }
    if ($Scenario -eq 'diplomacy-hover') {
        $interaction=@(@(40,0x73,'open-foreign-advisor'),@(98,27,'leave-diplomacy'),@(100,27,'close-foreign-advisor'),@(104,0x87,'scroll-after-diplomacy'))
    }
    if ($Scenario -eq 'turn') {
        $interaction=@(@(36,32,'skip-first-unit'),@(39,32,'skip-second-unit'),@(42,13,'end-turn'),@(65,32,'skip-first-unit-next-turn'),@(68,32,'skip-second-unit-next-turn'),@(71,13,'end-second-turn'))
    }
    if ($Scenario -in @('research-turn','reveal-scroll','forest-shadow')) {
        # Reproduce the first-turn modal handoff, then leave all input idle for 70 seconds.
        $interaction=@(@(60,66,'found-city'),@(64,13,'accept-name'),@(85,13,'close-city'),
            @(95,0x68,'worker-north'),@(101,0x68,'scout-north-one'),@(105,0x68,'scout-north-two'),
            @(110,13,'finish-first-turn'),@(115,13,'choose-bronze-working'),@(185,0x87,'scroll-after-idle-turn'))
    }
    if ($Scenario -eq 'reveal-scroll') {
        # Move/reveal at a settled non-native zoom, then drag to the actual
        # client edges. The cursor's 32px save crosses those edges even when
        # camera clamping prevents a scroll step.
        $mouseSteps=@(@(90,0,0,0x800,360),@(135,0,0,2),
            @(137,0,2147483647,0),@(141,0,0,0),@(144,0,0,4),
            @(148,2147483647,0,0),@(152,0,0,0),@(156,0,-2147483648,0),
            @(160,0,0,0),@(164,-2147483648,0,0),@(168,0,0,0))
    }
    if ($Scenario -eq 'unit-motion') {
        # Queue a turn before native animation finishes; never advance the turn.
        $interaction=@(@(75,0x64,'scout-west'),@(75.1,0x68,'scout-north'))
    }
    if ($Scenario -eq 'unit-turn') {
        # Leave the completed turn untouched until the final camera comparison.
        $interaction=@(@(75,0x68,'scout-north-first'),@(78,0x68,'scout-north-second'),
            @(90,32,'skip-remaining'),@(94,13,'finish-turn'),@(120,0x87,'scroll-after-idle-turn'))
    }
    if ($Scenario -eq 'turn-scroll') {
        $interaction=@()
        for($step=0;$step -lt 24;++$step){$interaction+=,@((70+$step),0x87,('scroll-'+$step))}
        $interaction+=,@(112,0x68,'first-scout-move')
        $interaction+=,@(124,0x68,'second-scout-move')
        $interaction+=,@(136,32,'skip-remaining')
        $interaction+=,@(140,32,'skip-second-remaining')
        $interaction+=,@(144,32,'skip-third-remaining')
        $interaction+=,@(148,13,'end-turn-one')
        $interaction+=,@(162,32,'skip-next-unit')
        $interaction+=,@(166,32,'skip-other-unit')
        $interaction+=,@(170,32,'skip-last-unit')
        $interaction+=,@(174,13,'end-turn-two')
        for($step=0;$step -lt 8;++$step){$interaction+=,@((187+$step),0x87,('scroll-after-interturn-'+$step))}
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
        # The opt-in native load hook dismisses this known popup when ready.
        # No timed Enter can accidentally close a later game dialog.
        $interaction=@()
    }
    if ($Scenario -eq 'lifecycle') {
        $interaction=@(@(42,81,'return-first-game-to-menu'),@(44,40,'select-quit'),@(45,13,'confirm-quit'),@(55,13,'load-again'),@(58,13,'accept-save'),@(100,0x86,'text-second-game'),@(105,81,'return-second-game-to-menu'),@(107,40,'select-second-quit'),@(108,13,'confirm-second-quit'))
    }
    while ([DateTime]::UtcNow -lt $end -and -not $child.HasExited) {
        $child.Refresh(); $window=$child.MainWindowHandle
        # Every scenario measures Civ III's message latency: a frozen UI thread
        # shows as unanswered or second-long pings, with or without Renderer64.
        if ($window -ne [IntPtr]::Zero) { [RendererUiPing]::Start($window,20) }
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
            $inputElapsed=$elapsed
            if ($Scenario -eq 'camera') { $inputElapsed=if ($null -eq $combatReadyAt) {-1} else {$elapsed-$combatReadyAt} }
            if ($Scenario -in @('navigation','zoom-out','near','units','city','screen-sweep')) {
                if ($null -eq $navigationReadyAt -and $elapsed -ge $navigationCheck) {
                    $navigationCheck=$elapsed+1
                    # Native rendering has no renderer readiness trace: a fixed 60 s.
                    if (($NativeRendering -and $elapsed -ge 60) -or (Get-Content -LiteralPath (Join-Path $session 'renderer.log') -Raw) -match 'stage=render-done result=1') {
                        $navigationReadyAt=$elapsed
                        Write-Host ('Navigation map ready at '+$navigationReadyAt)
                    }
                }
                $inputElapsed=if ($null -eq $navigationReadyAt) {-1} else {$elapsed-$navigationReadyAt+28}
            }
            if ($Scenario -in @('turn-stress','debug-scroll','turn-scroll','unit-turn','unit-motion','research-turn','reveal-scroll','route-city','route-zoom','city-builds') -and $elapsed -ge $failureCheck) {
                $failureCheck=$elapsed+5
                $recent=Get-Content -LiteralPath (Join-Path $session 'renderer.log') -Tail 1000
                if ($recent -match 'stage=retained-admission-rejected|retained admission:|stage=async-publication-failed|stage=visual-failure|stage=native-operation-failed|stage=required-interturn-preparation-failed') {
                    Write-Host 'Renderer failure detected; preserving evidence and stopping the disposable test.'
                    break
                }
            }
            if ($MeasureCadence -and $elapsed -ge $cadenceNext) {
                $cadenceNext=$elapsed+$(if ($Scenario -in @('zoom','zoom-out','near','units','navigation','camera','reveal-scroll')) {0.02} else {1.0})
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
            if ($Scenario -in @('mouse','zoom','zoom-out','near','units','city','route-city','route-zoom','city-builds','navigation','camera','reveal-scroll','forest-shadow','pedia-hover','foreign-advisor','diplomacy-hover','domestic-slider','screen-sweep') -and $mouseIndex -lt $mouseSteps.Count -and $inputElapsed -ge $mouseSteps[$mouseIndex][0]) {
                $step=$mouseSteps[$mouseIndex]
                if ($mouseIndex -eq 0) { [void][RendererGameCommand]::SetForegroundWindow($window) }
                if ([RendererGameCommand]::GetForegroundWindow() -ne $window) { throw 'Diagnostic game lost foreground before mouse input.' }
                $rect=New-Object RendererGameCommand+Rect
                $point=New-Object RendererGameCommand+Point
                [void][RendererGameCommand]::GetClientRect($window,[ref]$rect)
                $clientX=if($step[1] -eq [int]::MaxValue){$rect.Right-1}elseif($step[1] -eq [int]::MinValue){0}else{[int]($rect.Right/2)+$step[1]}
                $clientY=if($step[2] -eq [int]::MaxValue){$rect.Bottom-1}elseif($step[2] -eq [int]::MinValue){0}else{[int]($rect.Bottom/2)+$step[2]}
                $point.X=$clientX; $point.Y=$clientY
                [void][RendererGameCommand]::ClientToScreen($window,[ref]$point)
                $inputTicks=[System.Diagnostics.Stopwatch]::GetTimestamp()
                [void][RendererGameCommand]::SetCursorPos($point.X,$point.Y)
                $wheelDelta=if ($step.Count -gt 4) { [int]$step[4] } else { 0 }
                $wheelData=[uint32]([long]$wheelDelta -band 4294967295)
                if ($step[3]) { [RendererGameCommand]::mouse_event($step[3],0,0,$wheelData,[UIntPtr]::Zero) }
                $mouseEvents += [ordered]@{ qpc=$inputTicks; client_x=$clientX; client_y=$clientY; flags=$step[3]; wheel_delta=$wheelDelta }
                if ($null -eq $scriptedInputQpc) { $scriptedInputQpc=$inputTicks }
                if ($step[3] -eq 2) { $mouseHeld=$true }
                if ($step[3] -eq 4) { $mouseHeld=$false }
                Write-Host ('Mouse command: '+($step -join ',')); ++$mouseIndex
            }
            $key=if ($Scenario -eq 'scroll') {0x87} else {0}
            # -LoadWarnings: further Enters, four seconds apart, dismiss benign C3X warnings
            # (a scenario's config) shown during the load, never after the first map. The
            # log may not exist yet: an empty read is "not ready", not an empty array.
            if ($enterCount -lt 2+$LoadWarnings -and $elapsed -ge $(if ($enterCount -lt 2) {5+2*$enterCount} else {7+4*($enterCount-1)}) -and
                ($enterCount -lt 2 -or [string](Get-Content -LiteralPath (Join-Path $session 'renderer.log') -Raw -ErrorAction SilentlyContinue) -notmatch 'stage=render-done result=1')) {
                $key=13; ++$enterCount
            }
            $interactionElapsed=if ($Scenario -eq 'camera') {$elapsed} else {$inputElapsed}
            if ($Scenario -ne 'scroll' -and $interactionIndex -lt $interaction.Count -and $interactionElapsed -ge $interaction[$interactionIndex][0]) {
                $key=$interaction[$interactionIndex][1]
                Write-Host ('Interaction command: '+$interaction[$interactionIndex][2])
                if ($null -eq $scriptedInputQpc) { $scriptedInputQpc=[System.Diagnostics.Stopwatch]::GetTimestamp() }
                ++$interactionIndex
            }
            if ($Scenario -in @('combat','camera') -and $elapsed -ge 36 -and -not $combatAttackSent) {
                # Preparation is idempotent and can be rejected until the first
                # map is ready. Start the attack only after native confirmation.
                if ($null -eq $combatReadyAt) {
                    $liveLog=Get-Content -LiteralPath (Join-Path $session 'renderer.log') -Raw
                    if ($liveLog -match 'stage=scripted-combat-ready') { $combatReadyAt=$elapsed }
                    elseif ($elapsed -ge $combatNextPrepare) {
                        $key=0x84; $combatNextPrepare=$elapsed+3
                        Write-Host 'Interaction command: prepare-combat'
                    }
                } elseif ($elapsed -ge $combatReadyAt+$(if ($Scenario -eq 'camera') {8} else {4})) {
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
            if ($Scenario -eq 'camera' -and -not $combatZoomKeySent -and $inputElapsed -ge 12.4) {
                $key=90; $combatZoomKeySent=$true
                Write-Host 'Interaction command: combat-zoom-key'
            }
            if ($key -eq 0) {
                $pollMs=if ($Scenario -in @('zoom','zoom-out','near','units','route-city','route-zoom','city-builds','navigation','camera','reveal-scroll','unit-motion','pedia-hover','foreign-advisor','diplomacy-hover','domestic-slider','screen-sweep')) {20} else {1000}
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
                $keyTicks=[System.Diagnostics.Stopwatch]::GetTimestamp()
                if (-not [RendererGameCommand]::PostMessage($window,0x100,[IntPtr]$key,[IntPtr]1)) { throw 'Cannot post diagnostic command.' }
                [void][RendererGameCommand]::PostMessage($window,0x101,[IntPtr]$key,[IntPtr](-1073741823))
                $keyEvents += [ordered]@{ qpc=$keyTicks; key=$key }
            }
            ++$sent
        }
        # Keep cadence observation running after Z as well as mouse input.
        Start-Sleep -Milliseconds $(if ($Scenario -in @('zoom','zoom-out','near','units','route-city','route-zoom','city-builds','navigation','camera','reveal-scroll','unit-motion','pedia-hover','foreign-advisor','diplomacy-hover','domestic-slider','screen-sweep')) {20} else {1000})
    }
    if ($child.HasExited) { $earlyExit=$true; $gameExitCode=$child.ExitCode }
    Write-Host ('Posted diagnostic commands: '+$sent)
} finally {
    # Before the game is closed: pings to a closing window fail at once.
    $pings=[RendererUiPing]::Stop()
    if ($cadence) { $cadence.Dispose(); $cadence=$null }
    if ($mouseHeld) { [RendererGameCommand]::mouse_event(4,0,0,0,[UIntPtr]::Zero) }
    $env:C3X_RENDERER_GAME_TEST_SAVE=$oldEnvironment
    $env:C3X_RENDERER_GAME_TEST_MODE=$oldMode
    $env:C3X_RENDERER_GAME_TEST_COMBAT=$oldCombat
    $env:C3X_RENDERER_UNIT_PACK=$oldPack
    $env:C3X_RENDERER_TRACE_INPUT=$oldInputTrace
    foreach ($key in $traceEnvironment.Keys) { [Environment]::SetEnvironmentVariable($key,$traceEnvironment[$key]) }
    foreach ($key in $optionEnvironment.Keys) { [Environment]::SetEnvironmentVariable($key,$optionEnvironment[$key]) }
    if ($observer -and -not $observer.HasExited) {
        if (Test-Path (Join-Path $session 'window')) { Set-Content -LiteralPath (Join-Path $session 'window\stop.txt') -Value 'stop' }
        if (-not $observer.WaitForExit(5000)) { $observer.Kill(); $observer.WaitForExit() }
    }
    # Terminate only the process created above. The input save is a disposable
    # copy. No scenario saves its resulting gameplay.
    if ($child -and -not $child.HasExited) { $child.Kill(); $child.WaitForExit() }
    [IO.File]::WriteAllBytes($ini,$iniBytes)
    if ($c3xConfigBytes) { [IO.File]::WriteAllBytes($c3xConfig,$c3xConfigBytes) }
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
if ($Scenario -in @('mouse','zoom','zoom-out','near','units','route-city','route-zoom','city-builds','navigation','camera','reveal-scroll','pedia-hover','foreign-advisor','diplomacy-hover','domestic-slider','screen-sweep')) {
    [ordered]@{ qpc_frequency=[System.Diagnostics.Stopwatch]::Frequency; events=$mouseEvents } |
        ConvertTo-Json -Depth 4 | Set-Content (Join-Path $session 'mouse-events.json')
}
if ($pings.Count) {
    [ordered]@{ qpc_frequency=[System.Diagnostics.Stopwatch]::Frequency; meaning='WM_NULL round trips to the game window: sent, received, answered';
        samples=@($pings | ForEach-Object { ,@($_[0],$_[1],$_[2]) }) } | ConvertTo-Json -Depth 4 -Compress | Set-Content (Join-Path $session 'pings.json')
}
if ($keyEvents.Count) {
    [ordered]@{ qpc_frequency=[System.Diagnostics.Stopwatch]::Frequency; events=$keyEvents } |
        ConvertTo-Json -Depth 4 | Set-Content (Join-Path $session 'key-events.json')
}
$steps=@([regex]::Matches($log,'stage=scripted-game-scroll step=(\d+)') | ForEach-Object {[int]$_.Groups[1].Value})
$turns=@([regex]::Matches($log,'stage=scripted-turn-end turn=(\d+)') | ForEach-Object {[int]$_.Groups[1].Value})
$preparedTurns=[regex]::Matches($log,'stage=required-interturn-preparation-complete result=1').Count
$moves=[regex]::Matches($log,'stage=motion-start[^\r\n]*result=1').Count
$combatReady=[regex]::Matches($log,'stage=scripted-combat-ready').Count
$combatFinished=[regex]::Matches($log,'stage=scripted-combat-end').Count
$textEvents=[regex]::Matches($log,'stage=scripted-game-map-text').Count
$cityZooms=@([regex]::Matches($log,'stage=city-native-zoom tile_width=(\d+)') | ForEach-Object {[int]$_.Groups[1].Value})
$cityAnchors=@([regex]::Matches($log,'stage=city-native-zoom[^\r\n]*city_anchor=(-?\d+,-?\d+)') | ForEach-Object {$_.Groups[1].Value})
$readyEvents=[regex]::Matches($log,'stage=render-done result=1').Count
$loadingMaps=[regex]::Matches($log,'stage=loading-map-complete result=1').Count
$unloadEvents=[regex]::Matches($log,'stage=scene-unloaded result=1').Count
$errors=@($log -split "`n" | Where-Object {$_ -match 'stage=retained-admission-rejected|retained admission:|stage=required-interturn-preparation-failed|stage=native-operation-failed|stage=async-publication-failed|budget exceeded|stage=visual-failure|stage=worker-error|stage=unit-publication-failed|stage=unit-animation-failed|stage=motion-start[^\r\n]*result=[0235]|stage=scene-unload-failed|stage=(?:first-map-ready|loading-map-complete) result=[02345]|stage=assets-prepared ready=0'})
# The helper owns camera/render errors; DebugView may not include its file trace.
$corePath=Join-Path $session 'renderer-core.log.x64'
if(Test-Path $corePath) {
    $errors+=@(Get-Content $corePath | Where-Object {$_ -match 'stage=(fresh-draw-failed|camera-error|gpu-failure|worker-error|visual-failure|async-publication-failed)'})
}
$debugReveals=([regex]::Matches($log,'stage=debug-reveal result=1')).Count
$debugHides=([regex]::Matches($log,'stage=debug-hide result=1')).Count
$scriptedPings=if ($null -ne $scriptedInputQpc) { @($pings | Where-Object { $_[0] -ge $scriptedInputQpc }) } else { @() }
$pingTicks=@($scriptedPings | ForEach-Object { $_[1]-$_[0] })
$pingMaxMs=if ($pingTicks.Count) { [Math]::Round(1000.0*($pingTicks | Measure-Object -Maximum).Maximum/[System.Diagnostics.Stopwatch]::Frequency,1) } else { $null }
$pingStalls=@($scriptedPings | Where-Object { $_[2] -eq 0 -or ($_[1]-$_[0]) -gt [System.Diagnostics.Stopwatch]::Frequency }).Count
$windowResult=Join-Path $session 'window\finished.json'
$windowEvidence=if (Test-Path -LiteralPath $windowResult) { Get-Content -LiteralPath $windowResult -Raw | ConvertFrom-Json } else { $null }
$windowComplete=$null -ne $windowEvidence -and $windowEvidence.complete -and $windowEvidence.frames -gt 0
[ordered]@{ scenario=$Scenario; debug_reveals=$debugReveals; debug_hides=$debugHides; scene_samples=$SceneSamples; scene_sharpness=$SceneSharpness; combat_case=$CombatCase; unit_pack=$UnitPack; mouse_commands=$mouseIndex; completed_turns=$turns; prepared_turns=$preparedTurns; accepted_moves=$moves; interaction_commands=$interactionIndex; combat_ready=$combatReady; combat_finished=$combatFinished; map_text_events=$textEvents; city_zoom_widths=$cityZooms; city_anchors=$cityAnchors; posted_commands=$sent; load_requested=($log -match 'stage=scripted-game-load'); scroll_steps=$steps;
    game_exited_early=$earlyExit; game_exit_code=$gameExitCode; first_map_ready=$readyEvents; loading_maps_ready=$loadingMaps; scenes_unloaded=$unloadEvents; native_failures=$errors; window_evidence=$windowEvidence; native_rendering=[bool]$NativeRendering; ui_ping_max_ms=$pingMaxMs; ui_pings_over_1s=$pingStalls; original_save_unchanged=$true; scope='Real Civ III diagnostic; window samples require visual review; not an FPS benchmark' } |
    ConvertTo-Json -Depth 4 | Set-Content (Join-Path $session 'result.json')
Write-Host ('Scripted diagnostic saved: '+$session)
if ($Scenario -eq 'screen-sweep') {
    $sweepSchedule | ConvertTo-Json -Depth 3 | Set-Content (Join-Path $session 'sweep-screens.json')
    # Pixel, bits and DC access (6-8), drain and reinit (100-101) are CPU
    # ownership by design; every other operation is a refused GPU draw.
    $fallbacks=@([regex]::Matches($log,'stage=native-cpu-fallback operation=(\d+) count=(\d+)') | ForEach-Object {[int]$_.Groups[1].Value})
    $refused=@($fallbacks | Where-Object { $_ -notin @(6,7,8,100,101) } | Sort-Object -Unique)
    $failed=[regex]::Matches($log,'stage=native-operation-failed').Count
    if ($refused.Count -or $failed -or $earlyExit -or $pingStalls) {
        Write-Error ('Screen sweep failed: refused operations ['+($refused -join ',')+'], native failures '+$failed+', UI stalls '+$pingStalls+', early exit '+$earlyExit); exit 1
    }
}
if ($MeasureCadence -and $cadenceSamples.Count -lt 2) { Write-Error 'Insufficient renderer cadence samples; inspect cadence.json and the helper channel.'; exit 1 }
if ($Scenario -eq 'camera') {
    $fight=[regex]::Match($log,'(?s)stage=scripted-combat-start(.*?)stage=scripted-combat-end').Groups[1].Value
    if (-not $fight -or $fight -match 'stage=edge-scroll' -or
        [regex]::Matches($fight,'stage=zoom-target[^\r\n]*combat=1').Count -lt 5 -or
        $log -notmatch 'stage=camera-center[^\r\n]*combat=1') {
        Write-Error 'Camera witness requires native centering, four wheel changes plus Z during combat and no combat edge scrolling.'; exit 1
    }
    if ($MeasureCadence -and $mouseEvents.Count -eq 17) {
        $zoomSamples=@($cadenceSamples | Where-Object { $_.qpc -ge $mouseEvents[7].qpc -and $_.qpc -le $mouseEvents[10].qpc })
        if ($zoomSamples.Count -lt 2 -or $zoomSamples[-1].frames -le $zoomSamples[0].frames -or
            -not ($zoomSamples | Where-Object { $_.zoom_q16 -ge 131072 -and $_.zoom_q16 -le 131730 }) -or
            -not ($zoomSamples | Where-Object { $_.zoom_q16 -ge 195950 -and $_.zoom_q16 -le 196608 })) {
            Write-Error 'Combat zoom did not present both 2x and 3x views.'; exit 1
        }
    }
}
if ($Scenario -eq 'near') {
    $corePath=Join-Path $session 'renderer-core.log.x64'
    if((Test-Path $corePath) -and (Get-Content $corePath -Raw) -match 'stage=(fresh-draw-failed|camera-error)|stage=gpu-failure phase=camera'){
        Write-Error 'Renderer failed a near-zoom camera preparation; inspect renderer-core.log.x64.'; exit 1
    }
    foreach($width in @(160,256,384,128)) {
        if($log -notmatch ('stage=zoom-target[^\r\n]*new_width='+$width+' ')){ Write-Error ('Missing zoom target '+$width); exit 1 }
    }
    if($log -notmatch 'stage=edge-scroll'){ Write-Error 'Incomplete near-zoom input coverage.'; exit 1 }
}
if ($Scenario -eq 'zoom-out') {
    $corePath=Join-Path $session 'renderer-core.log.x64'
    if(Test-Path $corePath){
        $core=Get-Content $corePath -Raw
        if($core -match 'stage=(fresh-draw-failed|camera-error)|stage=gpu-failure phase=camera'){
            Write-Error 'Renderer failed a zoom-out camera preparation; inspect renderer-core.log.x64.'; exit 1
        }
    }
    foreach($width in @(64,80,96,112,128,384)) {
        if($log -notmatch ('stage=zoom-target[^\r\n]*new_width='+$width+' ')){ Write-Error ('Missing zoom target '+$width); exit 1 }
    }
    if($interactionIndex -ne 1 -or $log -notmatch 'stage=edge-scroll'){ Write-Error 'Incomplete zoom-out input coverage.'; exit 1 }
    $handoffs=[regex]::Matches($log,'stage=native-handoff requested=(-?\d+),(-?\d+) displayed=(-?\d+),(-?\d+) valid=(\d+)')
    if(-not $handoffs.Count){ Write-Error 'No completed camera handoff evidence.'; exit 1 }
    $last=$handoffs[$handoffs.Count-1]
    if($last.Groups[1].Value -ne $last.Groups[3].Value -or $last.Groups[2].Value -ne $last.Groups[4].Value -or $last.Groups[5].Value -ne '1') { Write-Error 'Native HUD and renderer camera did not converge after scrolling.'; exit 1 }

    if($MeasureCadence -and -not ($cadenceSamples | Where-Object { $_.frames -gt 0 -and $_.zoom_q16 -ge 32768 -and $_.zoom_q16 -le 32775 })) { Write-Error 'Half-scale view was never presented.'; exit 1 }
}
if ($Scenario -eq 'navigation' -and ($interactionIndex -ne 4 -or [regex]::Matches($log,'stage=scripted-route-accepted').Count -ne 4 -or [regex]::Matches($log,'stage=scripted-route-adopted').Count -ne 4 -or $log -match 'stage=scripted-route-refused')) { Write-Error 'Incomplete navigation coverage.'; exit 1 }
if ($Scenario -eq 'navigation') {
    if ($log -notmatch 'stage=edge-scroll[^\r\n]*camera=\d+,-420 scale=196608' -or
        $log -notmatch 'stage=edge-scroll[^\r\n]*camera=\d+,1112 scale=196608') {
        Write-Error 'Navigation did not reach both zoomed vertical limits.'; exit 1
    }
    $wrapped=$false; $previousX=0
    foreach ($match in [regex]::Matches($log,'stage=edge-scroll[^\r\n]*step=([1-9]\d*),0 camera=(\d+),')) {
        $nextX=[int]$match.Groups[2].Value
        if ($previousX -gt 3700 -and $nextX -lt 100) { $wrapped=$true }
        $previousX=$nextX
    }
    if (-not $wrapped) { Write-Error 'Navigation did not cross the horizontal seam.'; exit 1 }
}

if ($Scenario -eq 'city-builds' -and $interactionIndex -ne $interaction.Count) { Write-Error 'Incomplete city build-item interaction coverage.'; exit 1 }
if ($Scenario -eq 'pedia-hover' -and ($interactionIndex -ne $interaction.Count -or $mouseIndex -ne $mouseSteps.Count)) { Write-Error 'Incomplete Civilopedia-cursor hover coverage.'; exit 1 }
if ($Scenario -eq 'foreign-advisor' -and ($interactionIndex -ne $interaction.Count -or $mouseIndex -ne $mouseSteps.Count -or $steps.Count -ne 1)) { Write-Error 'Incomplete Foreign Advisor coverage.'; exit 1 }
if ($Scenario -eq 'diplomacy-hover' -and ($interactionIndex -ne $interaction.Count -or $mouseIndex -ne $mouseSteps.Count -or $steps.Count -ne 1)) { Write-Error 'Incomplete diplomacy hover coverage.'; exit 1 }
if ($Scenario -eq 'route-city' -and ($interactionIndex -ne $interaction.Count -or $moves -lt 1 -or $log -notmatch 'stage=route-line')) { Write-Error 'Incomplete held-route and city cycling coverage.'; exit 1 }
if ($Scenario -eq 'route-zoom' -and ($log -notmatch 'stage=route-line' -or $log -notmatch 'stage=zoom-target[^\r\n]*new_width=256 ' -or $log -notmatch 'stage=zoom-target[^\r\n]*new_width=384 ')) { Write-Error 'Incomplete zoomed held-route coverage.'; exit 1 }
if ($Scenario -in @('research-turn','reveal-scroll') -and ($turns.Count -ne 1 -or $preparedTurns -ne 1 -or $moves -lt 3 -or $steps.Count -ne 1 -or $readyEvents -ne 1 -or $interactionIndex -ne $interaction.Count)) { Write-Error 'Incomplete first-turn research and idle animation coverage.'; exit 1 }
if ($Scenario -eq 'reveal-scroll' -and ($log -notmatch 'stage=edge-scroll' -or $log -notmatch 'stage=route-line')) { Write-Error 'Missing zoomed reveal / edge drag coverage.'; exit 1 }
if ($Scenario -eq 'unit-motion' -and ($moves -ne 2 -or $turns.Count -ne 0 -or $readyEvents -ne 1 -or $interactionIndex -ne $interaction.Count)) { Write-Error 'Incomplete consecutive unit movement coverage.'; exit 1 }
if ($Scenario -eq 'unit-turn' -and ($turns.Count -lt 1 -or $preparedTurns -ne $turns.Count -or $moves -lt 2 -or $steps.Count -ne 1 -or $readyEvents -ne 1 -or $interactionIndex -ne $interaction.Count)) { Write-Error 'Incomplete unit movement or idle-turn coverage.'; exit 1 }
if ($Scenario -eq 'turn-scroll' -and ($turns.Count -lt 2 -or $preparedTurns -ne $turns.Count -or $moves -lt 2 -or $steps.Count -ne 32 -or $interactionIndex -ne $interaction.Count)) { Write-Error 'Incomplete movement, interturn preparation or scrolling coverage.'; exit 1 }
if ($Scenario -eq 'turn-stress' -and $moves -lt 8) { Write-Error 'Insufficient accepted unit movement; inspect the native movement records.'; exit 1 }
if ($MeasureCadence -and $Scenario -in @('turn-stress','debug-scroll','turn-scroll','unit-turn','unit-motion','research-turn','reveal-scroll','route-city','city-builds')) {
    $tail=@($cadenceSamples | Where-Object { $_.elapsed_seconds -ge $cadenceSamples[-1].elapsed_seconds-10 })
    if ($tail.Count -lt 2 -or $tail[-1].frames -le $tail[0].frames) { Write-Error 'Renderer stopped presenting at the end of the stress run.'; exit 1 }
}
if (-not $windowComplete) { Write-Error 'Window evidence did not complete; inspect window-errors.log and window/finished.json.'; exit 1 }
if (($Scenario -in @('mouse','zoom','zoom-out','near','units','city','route-city','route-zoom','city-builds','navigation','camera','reveal-scroll') -and $mouseIndex -ne $mouseSteps.Count) -or ($Scenario -eq 'turn' -and $turns.Count -lt 2) -or ($Scenario -in @('combat','camera') -and ($combatReady -ne 1 -or $combatFinished -ne 1)) -or $earlyExit -or ($Scenario -eq 'scroll' -and $steps.Count -ne 32) -or ($Scenario -in @('settler','site-toggle','debug','debug-scroll','newgame','turn-stress') -and $readyEvents -lt 1) -or ($Scenario -in @('site-toggle','debug','debug-scroll','newgame','turn-stress') -and $interactionIndex -ne $interaction.Count) -or ($Scenario -in @('interaction','hud') -and ($interactionIndex -ne $interaction.Count -or $textEvents -ne 3)) -or ($Scenario -eq 'hud' -and (($cityZooms -join ',') -ne '64,128' -or $steps.Count -ne 2)) -or ($Scenario -eq 'lifecycle' -and ($interactionIndex -ne $interaction.Count -or $readyEvents -ne 2 -or $unloadEvents -lt 2 -or $textEvents -ne 1)) -or ($Scenario -eq 'city' -and ($interactionIndex -ne $interaction.Count -or ($cityZooms -join ',') -ne '64,128')) -or ($Scenario -in @('hud','city') -and ($cityAnchors.Count -ne 2 -or $cityAnchors[0] -ne $cityAnchors[1])) -or ($Scenario -eq 'debug' -and ($steps.Count -ne 2 -or $debugReveals -ne 1 -or $debugHides -ne 1)) -or ($Scenario -eq 'debug-scroll' -and ($steps.Count -ne 32 -or $debugReveals -ne 1 -or $debugHides -ne 0)) -or ($Scenario -eq 'turn-stress' -and ($turns.Count -lt 8 -or @($turns | Select-Object -Unique).Count -lt 8)) -or $errors.Count) { exit 1 }

# DebugView --stop may leave a nonzero native exit code after successful cleanup.
# Only the explicit diagnostic checks above determine this scenario result.
exit 0
