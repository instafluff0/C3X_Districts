# Start ordinary gameplay with bounded, process-filtered diagnostics.
# Portable Microsoft DebugView and Intel PresentMon live in ignored build data.
param([switch]$CheckOnly, [switch]$NoReplayRecording, [string]$ConquestsDirectory)
$ErrorActionPreference = 'Stop'
$renderer = Split-Path $PSScriptRoot -Parent
$tools = Join-Path $renderer 'native\build\live-tools'
$arm = $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64'
$debugName = if ($arm) { 'dbgviewcli64a.exe' } else { 'dbgviewcli64.exe' }
$debug = Join-Path $tools ('DebugView\' + $debugName)
$present = Join-Path $tools 'PresentMon.exe'
$dll = Join-Path $renderer 'bin\C3XRenderer.dll'
$conquests = $ConquestsDirectory
if (-not $conquests) { $conquests = $env:C3X_RENDERER_CIV3_CONQUESTS }
if (-not $conquests) {
    $conquests = Join-Path ${env:ProgramFiles(x86)} 'GOG Galaxy\Games\Civilization III Complete\Conquests'
}
$game = Join-Path $conquests 'Civ3Conquests.exe'

function Quote-Arguments([string[]]$Values) {
    # Every value here is a flag, a filename or a numeric limit, never shell code.
    # Windows filenames cannot contain quotes; no argument ends with a backslash.
    ($Values | ForEach-Object {
        if ($_ -match '["\r\n]' -or $_.EndsWith('\')) { throw 'Unsupported capture argument' }
        '"' + $_ + '"'
    }) -join ' '
}

function Get-ElevatedPath([string]$Path) {
    # Mapped shares belong to the unelevated token. Keep the same file through
    # its UNC path instead of assuming the elevated token has that drive.
    if ($Path -match '^([A-Za-z]):\\') {
        $drive = Get-PSDrive -Name $Matches[1] -ErrorAction Stop
        if ($drive.DisplayRoot) { return Join-Path $drive.DisplayRoot $Path.Substring(3) }
    }
    return $Path
}

# XP compatibility can elevate Civ III through ShellExecute and discard the
# launching process's environment. Elevate the capture host first, then create
# the game directly with the diagnostic environment. One UAC boundary also
# covers PresentMon; do not change the game's compatibility settings.
$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $CheckOnly -and -not $elevated) {
    Write-Host 'Windows permission is needed for the capture host so game compatibility mode preserves recording settings.'
    # Shell elevation reparses command arguments. Encode this small invocation
    # so shared paths with spaces survive that boundary exactly.
    $script = Get-ElevatedPath $PSCommandPath
    $directory = Get-ElevatedPath $conquests
    $command = "& '" + $script.Replace("'", "''") + "' -ConquestsDirectory '" + $directory.Replace("'", "''") + "'"
    if ($NoReplayRecording) { $command += ' -NoReplayRecording' }
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $arguments = @('-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand',$encoded)
    try {
        $hostProcess = Start-Process powershell.exe -ArgumentList (Quote-Arguments $arguments) -Verb RunAs -PassThru -Wait
        exit $hostProcess.ExitCode
    } catch {
        Write-Error ('Capture host could not start: ' + $_.Exception.Message)
        exit 1
    }
}

$debugProcess = $null
$presentProcess = $null
$gameProcess = $null
$session = $null
$saved = $null
$result = 1
try {
    foreach ($file in @($game, $dll, $debug, $present)) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required file is missing: $file" }
    }
    foreach ($tool in @($debug, $present)) {
        $signature = Get-AuthenticodeSignature -LiteralPath $tool
        $publisher = if ($tool -eq $debug) { 'Microsoft Corporation' } else { 'Intel Corporation' }
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike "*CN=$publisher,*") {
            throw "Capture tool signature could not be verified: $tool"
        }
    }
    if ($CheckOnly) {
        Write-Host 'PASS: game, staged renderer and signed capture tools found. Nothing launched.'
        exit 0
    }
    if (Get-Process -Name Civ3Conquests -ErrorAction SilentlyContinue) {
        throw 'Please close Civ III first, then double-click CAPTURE_GAME.bat again.'
    }
    if (Get-Process -Name dbgview,dbgview64,dbgview64a,dbgviewcli,dbgviewcli64,dbgviewcli64a -ErrorAction SilentlyContinue) {
        throw 'Please close DebugView first so this capture can own and stop its collector.'
    }

    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,6)
    # Write per-frame output to the VM's local disk, not the shared filesystem.
    $session = Join-Path $env:TEMP ('C3XRendererCapture\' + $stamp)
    $saved = Join-Path $renderer ('native\build\live-captures\' + $stamp)
    New-Item -ItemType Directory -Path $session -Force | Out-Null
    $metadata = [ordered]@{
        schema = 1; started_utc = [DateTime]::UtcNow.ToString('o')
        renderer_sha256 = (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash.ToLowerInvariant()
        presentmon_sha256 = (Get-FileHash -LiteralPath $present -Algorithm SHA256).Hash.ToLowerInvariant()
        debugview_sha256 = (Get-FileHash -LiteralPath $debug -Algorithm SHA256).Hash.ToLowerInvariant()
        trace_level = 1; expensive_profiling = $false; limit_seconds = 900
        game_exit_code = $null; result = 'starting'
        composition_recording = -not $NoReplayRecording
        capture_host_elevated = $elevated
        game_launch = 'CreateProcess with inherited diagnostic environment'
        recording_scope = 'native GPU composition; external map/pose pixels; ownership observations'
        recording_max_bytes = 536870912; recording_max_seconds = 180
        recording_duration_anchor = 'first GPU composition session'
        capture_fps_is_performance_baseline = $false
    }
    $metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $session 'session.json') -Encoding UTF8
    $env:C3X_RENDERER_TRACE = '1'
    $env:C3X_RENDERER_PROFILE = '0'
    $env:C3X_RENDERER_TRACE_BUFFERED = '0'
    # Keep renderer evidence even if the external debug collector stops before
    # gameplay begins. The DLL bounds this buffered file independently; it does
    # not require elevation or another process to remain attached.
    $env:C3X_RENDERER_TRACE_FILE = Join-Path $session 'renderer-runtime.log'
    $env:C3X_RENDERER_TRACE_MIB = '64'
    # Recording adds explicit GPU readbacks for external map/pose inputs and
    # selected pixel oracles. Its FPS is diagnostic, never a performance baseline.
    $env:C3X_RENDERER_RECORD_FILE = if ($NoReplayRecording) { '' } else { Join-Path $session 'composition.c3xr' }

    $debugArgs = @('--accepteula','--no-banner','--no-kernel','--process-filter','Civ3Conquests',
        '--duration','900','--max-lines','500000','--log',(Join-Path $session 'renderer.log'),'--log-limit','64')
    $debugProcess = Start-Process -FilePath $debug -ArgumentList (Quote-Arguments $debugArgs) -PassThru -WindowStyle Hidden
    Start-Sleep -Milliseconds 500
    if ($debugProcess.HasExited) { throw 'Debug output collector stopped before the game started.' }

    $presentArgs = @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $PSScriptRoot 'capture_frames.ps1'),
        '-SessionDirectory',$session,'-SessionName',('C3XCapture-' + $stamp),'-PresentMon',$present)
    try {
        $presentProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList (Quote-Arguments $presentArgs) -PassThru -WindowStyle Hidden
        $readyDeadline = [DateTime]::UtcNow.AddSeconds(20)
        while (-not (Test-Path -LiteralPath (Join-Path $session 'frames-ready.txt'))) {
            if ($presentProcess.HasExited -or [DateTime]::UtcNow -gt $readyDeadline) { throw 'FPS collector did not start; see capture diagnostics.' }
            Start-Sleep -Milliseconds 200
        }
        $metadata.fps_collector_started = $true
    } catch {
        $metadata.fps_collector_started = $false
        $_.Exception.Message | Set-Content -LiteralPath (Join-Path $session 'fps-start-error.txt')
        # A delayed elevation response must not leave an orphan collector.
        'stop' | Set-Content -LiteralPath (Join-Path $session 'stop-frames.txt')
        Write-Warning 'FPS collection is unavailable. Continuing with bounded renderer/debug logs.'
    }

    Write-Host ''
    Write-Host 'Capture ready. Play for about 2-3 minutes:'
    if (-not $NoReplayRecording) {
        Write-Host 'Replay recording is ON. Extra capture work can make this run slower.'
        Write-Host 'The recording starts with map composition, then stops at 3 minutes, 512 MiB, or low memory.'
        Write-Host 'Startup ownership events are also captured. Ordinary logs continue afterward.'
    }
    Write-Host '  1. Load your usual save; stay still with animated water/units for 20 seconds.'
    Write-Host '  2. Scroll for 20 seconds, make several distant jumps, and switch selected units.'
    Write-Host '  3. Open/close a city and select a worker. Then quit the game normally.'
    Write-Host 'Leave this window open. Capture stops automatically after 15 minutes.'
    $gameStart = New-Object System.Diagnostics.ProcessStartInfo
    $gameStart.FileName = $game
    $gameStart.WorkingDirectory = $conquests
    $gameStart.UseShellExecute = $false
    $gameProcess = [System.Diagnostics.Process]::Start($gameStart)
    $metadata.game_process_id = $gameProcess.Id
    $gameProcess.WaitForExit()
    $metadata.game_exit_code = $gameProcess.ExitCode
    $metadata.result = 'game-exited'
    $result = 0
} catch {
    Write-Host ('Capture setup: ' + $_.Exception.Message) -ForegroundColor Yellow
    if ($session) { $_.Exception.Message | Set-Content -LiteralPath (Join-Path $session 'capture-error.txt') }
} finally {
    if ($session -and $presentProcess) {
        'stop' | Set-Content -LiteralPath (Join-Path $session 'stop-frames.txt')
        if (-not $presentProcess.WaitForExit(10000)) { Write-Warning 'FPS collector has not finished; its time limit remains active.' }
    }
    if ($debugProcess -and -not $debugProcess.HasExited) {
        & $debug --stop --no-banner | Out-Null
        if (-not $debugProcess.WaitForExit(5000)) { Write-Warning 'Debug collector has not finished; its time limit remains active.' }
    }
    if ($session) {
        $metadata.finished_utc = [DateTime]::UtcNow.ToString('o')
        $metadata.debug_collector_exited = -not $debugProcess -or $debugProcess.HasExited
        $metadata.fps_collector_exited = -not $presentProcess -or $presentProcess.HasExited
        $frameFile = Join-Path $session 'frames.csv'
        $metadata.frame_capture_present = (Test-Path -LiteralPath $frameFile) -and (Get-Item -LiteralPath $frameFile).Length -gt 0
        if ($metadata.frame_capture_present) {
            $metadata.frame_rows = @(Import-Csv -LiteralPath $frameFile).Count
        }
        $recordingFile = Join-Path $session 'composition.c3xr'
        $metadata.recording_present = Test-Path -LiteralPath $recordingFile -PathType Leaf
        if ($metadata.recording_present) {
            $metadata.recording_bytes = (Get-Item -LiteralPath $recordingFile).Length
            $metadata.recording_sha256 = (Get-FileHash -LiteralPath $recordingFile -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        if ($metadata.composition_recording -and -not $metadata.recording_present) {
            $metadata.result = 'recording-missing'
            $result = 1
            Write-Warning 'Replay recording did not start. Logs were saved, but this session cannot be replayed.'
        }
        $metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $session 'session.json') -Encoding UTF8
        New-Item -ItemType Directory -Path $saved -Force | Out-Null
        Copy-Item -Path (Join-Path $session '*') -Destination $saved -Force
        Write-Host ''
        Write-Host ('Capture saved: ' + $saved)
        Write-Host 'Tell Codex: Finished the capture. No upload or copy/paste needed.'
    }
}
exit $result
