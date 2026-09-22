# Start ordinary gameplay with bounded, process-filtered diagnostics.
# Portable Microsoft DebugView and Intel PresentMon live in ignored build data.
param([switch]$CheckOnly, [switch]$NoReplayRecording, [switch]$ShortDiagnostic, [string]$ConquestsDirectory)
$ErrorActionPreference = 'Stop'
if ($ShortDiagnostic -and $NoReplayRecording) { throw 'ShortDiagnostic requires input recording.' }
$renderer = Split-Path $PSScriptRoot -Parent
$tools = Join-Path $renderer 'native\build\live-tools'
$arm = $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64'
$debugName = if ($arm) { 'dbgviewcli64a.exe' } else { 'dbgviewcli64.exe' }
$debug = Join-Path $tools ('DebugView\' + $debugName)
$present = Join-Path $tools 'PresentMon.exe'
$dll = Join-Path $renderer 'bin\C3XRenderer.dll'
$witness = Join-Path $renderer 'native\build\window-witness\window_witness.exe'
$inspect = Join-Path $renderer 'native\build\input-recording\inspect_inputs.exe'
$replay = Join-Path $renderer 'native\build\input-recording\replay_inputs.exe'
$qualification = Join-Path $renderer 'native\build\input-recording\capture-ready.json'
if ($ShortDiagnostic) { $qualification = Join-Path $renderer 'native\build\input-recording\short-capture-ready.json' }
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

function Get-ShortCaptureStopReason([string]$Timeline) {
    if (-not (Test-Path -LiteralPath $Timeline)) { return $null }
    # Reuse the observer's once-per-second memory sample. Never walk the game's
    # address space a second time from this launcher.
    $samples = @(Get-Content -LiteralPath $Timeline -Tail 64 | Where-Object { $_ -match '"event":"process_memory"' })
    if ($samples.Count) {
        try { $sample = $samples[-1] | ConvertFrom-Json } catch { return $null }
        if ($sample.free_bytes -lt 134217728) { return 'low-address-space' }
    }
    return $null
}

function Test-OrphanCaptureCollector($Collector, [string[]]$AllowedPaths, [bool]$ParentExists) {
    # A missing window alone does not imply ownership. Never stop GUI DebugView,
    # another capture host's collector, or a CLI with unknown launch metadata.
    return (-not $ParentExists -and
        $Collector.Name -match '^dbgviewcli(64a?|)\.exe$' -and
        $Collector.ExecutablePath -and $AllowedPaths -contains $Collector.ExecutablePath -and
        $Collector.CommandLine -match '"--process-filter"\s+"Civ3Conquests"' -and
        $Collector.CommandLine -match '"--log"\s+"[^"\r\n]*\\C3XRendererCapture\\\d{8}-\d{6}-[0-9a-f]{6}\\renderer\.log"')
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
    if ($ShortDiagnostic) { $command += ' -ShortDiagnostic' }
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
$witnessProcess = $null
$gameProcess = $null
$session = $null
$saved = $null
$result = 1
try {
    foreach ($file in @($game, $dll, $debug, $present, $witness, $inspect, $replay)) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required file is missing: $file" }
    }
    foreach ($tool in @($debug, $present)) {
        $signature = Get-AuthenticodeSignature -LiteralPath $tool
        $publisher = if ($tool -eq $debug) { 'Microsoft Corporation' } else { 'Intel Corporation' }
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike "*CN=$publisher,*") {
            throw "Capture tool signature could not be verified: $tool"
        }
    }
    $volume = New-Object System.IO.DriveInfo([IO.Path]::GetPathRoot($env:TEMP))
    if (-not $NoReplayRecording -and $volume.AvailableFreeSpace -lt 11811160064) {
        throw 'At least 11 GiB of local free disk space is required for the bounded input and window capture.'
    }
    if (-not $NoReplayRecording) {
        if (-not (Test-Path -LiteralPath $qualification -PathType Leaf)) { throw 'The input recorder has not passed its capture acceptance checks yet. No game was started.' }
        $ready = Get-Content -LiteralPath $qualification -Raw | ConvertFrom-Json
        if ($ShortDiagnostic -and $ready.scope -ne 'short-diagnostic-capture') {
            throw 'This receipt does not qualify the short diagnostic workflow.'
        }
        if ($ShortDiagnostic -and $ready.launcher_sha256 -ne (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()) {
            throw 'The diagnostic launcher changed since validation.'
        }
        if ($ready.status -ne 'pass' -or $ready.dll_sha256 -ne (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash.ToLowerInvariant()) {
            throw 'The staged renderer does not match the qualified capture build. No game was started.'
        }
        foreach ($pair in @(@($witness, $ready.window_witness_sha256), @($inspect, $ready.inspector_sha256), @($replay, $ready.replay_sha256))) {
            if ($pair[1] -ne (Get-FileHash -LiteralPath $pair[0] -Algorithm SHA256).Hash.ToLowerInvariant()) {
                throw 'A capture tool changed since validation. No game was started.'
            }
        }
    }
    if ($CheckOnly) {
        Write-Host 'PASS: game, staged renderer and signed capture tools found. Nothing launched.'
        exit 0
    }
    if (Get-Process -Name Civ3Conquests -ErrorAction SilentlyContinue) {
        throw 'Please close Civ III first, then double-click CAPTURE_GAME.bat again.'
    }
    $modName = Split-Path (Split-Path $renderer -Parent) -Leaf
    $installedDebug = Join-Path $conquests ($modName + '\Renderer\native\build\live-tools\DebugView\' + $debugName)
    foreach ($collector in @(Get-CimInstance Win32_Process -Filter "Name LIKE 'dbgviewcli%.exe'")) {
        $parentExists = $null -ne (Get-Process -Id $collector.ParentProcessId -ErrorAction SilentlyContinue)
        if (Test-OrphanCaptureCollector $collector @($debug, $installedDebug) $parentExists) {
            $owned = Get-Process -Id $collector.ProcessId -ErrorAction SilentlyContinue
            # Check creation time as well: do not target a reused process ID.
            if ($owned -and [Math]::Abs(($owned.StartTime.ToUniversalTime() - $collector.CreationDate.ToUniversalTime()).TotalMilliseconds) -lt 1) {
                Stop-Process -InputObject $owned -ErrorAction Stop
                if (-not $owned.WaitForExit(5000)) { throw 'The leftover capture collector has not stopped.' }
                Write-Host 'Cleared a leftover renderer capture collector. Existing logs were preserved.'
            }
        }
    }
    if (Get-Process -Name dbgview,dbgview64,dbgview64a,dbgviewcli,dbgviewcli64,dbgviewcli64a -ErrorAction SilentlyContinue) {
        throw 'Another DebugView collector is running (possibly hidden). Close its capture console or end its DebugView process in Task Manager, then retry.'
    }

    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,6)
    # Write per-frame output to the VM's local disk, not the shared filesystem.
    $session = Join-Path $env:TEMP ('C3XRendererCapture\' + $stamp)
    $saved = Join-Path $renderer ('native\build\live-captures\' + $stamp)
    New-Item -ItemType Directory -Path $session -Force | Out-Null
    $gameHeader = [IO.File]::ReadAllBytes($game)
    $peOffset = [BitConverter]::ToInt32($gameHeader, 60)
    $largeAddressAware = ([BitConverter]::ToUInt16($gameHeader, $peOffset + 22) -band 32) -ne 0
    $gameHeader = $null
    $metadata = [ordered]@{
        schema = 1; started_utc = [DateTime]::UtcNow.ToString('o')
        renderer_sha256 = (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash.ToLowerInvariant()
        presentmon_sha256 = (Get-FileHash -LiteralPath $present -Algorithm SHA256).Hash.ToLowerInvariant()
        debugview_sha256 = (Get-FileHash -LiteralPath $debug -Algorithm SHA256).Hash.ToLowerInvariant()
        trace_level = 1; expensive_profiling = $false; limit_seconds = 900
        game_exit_code = $null; result = 'starting'
        input_recording = -not $NoReplayRecording; window_recording = -not $NoReplayRecording
        capture_host_elevated = $elevated
        game_launch = 'CreateProcess with inherited diagnostic environment'
        recording_scope = 'production renderer and native bridge inputs; correlated sampled window evidence'
        recording_max_bytes = 8589934592; recording_max_seconds = 600
        recording_duration_anchor = 'first successful GPU presentation'
        short_diagnostic = [bool]$ShortDiagnostic
        capture_stop_requested = $null
        capture_fps_is_performance_baseline = $false
        os_version = [Environment]::OSVersion.Version.ToString()
        process_architecture = $env:PROCESSOR_ARCHITECTURE
        logical_processors = [Environment]::ProcessorCount
        game_large_address_aware = $largeAddressAware
        video_controllers = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM)
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
    # Input and external-window recording compete for CPU/GPU resources.
    # Its observed FPS is diagnostic, never a performance baseline.
    $env:C3X_RENDERER_RECORD_FILE = ''
    $env:C3X_RENDERER_INPUT_RECORD_DIR = if ($NoReplayRecording) { '' } else { Join-Path $session 'inputs' }
    Copy-Item -LiteralPath $dll -Destination (Join-Path $session 'C3XRenderer.dll')
    $replayTools = New-Item -ItemType Directory -Path (Join-Path $session 'replay-tools')
    Copy-Item -LiteralPath $replay, $inspect -Destination $replayTools.FullName
    Copy-Item -LiteralPath $qualification -Destination (Join-Path $session 'capture-build.json') -ErrorAction SilentlyContinue

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
        if ($ShortDiagnostic) { throw 'Short diagnostic capture requires the FPS collector. No game was started.' }
    }

    Write-Host ''
    if ($ShortDiagnostic) {
        Write-Host 'Play for 60-90 seconds, then return to this console and press Enter to save.' -ForegroundColor Cyan
        Write-Host 'Keep the game open until Capture saved appears. Then you may quit it.'
    } elseif ($NoReplayRecording) {
        Write-Host 'Failure diagnostics ready: reproduce the problem, then close the game to save logs.' -ForegroundColor Cyan
        Write-Host 'No replay inputs or window images are recorded. You do not need a ten-minute run.'
    } else { Write-Host 'Capture ready. Play normally for about ten minutes.' }
    if (-not $NoReplayRecording) {
        if (-not $ShortDiagnostic) { Write-Host 'Inputs record for at most ten minutes after the first GPU presentation.' }
        Write-Host 'Window samples and memory are captured too.'
        Write-Host 'Include idle water/units, scrolling, camera jumps, movement, an interturn and opening/closing a city.'
        Write-Host 'Recording overhead is measured separately; this run is diagnostic.'
    }
    Write-Host 'Leave this window open. Collectors stop at the recording boundary or after 15 minutes; the game can remain open.'
    $gameStart = New-Object System.Diagnostics.ProcessStartInfo
    $gameStart.FileName = $game
    $gameStart.WorkingDirectory = $conquests
    $gameStart.UseShellExecute = $false
    $gameProcess = [System.Diagnostics.Process]::Start($gameStart)
    $metadata.game_process_id = $gameProcess.Id
    $null = $gameProcess.Handle
    if (-not $NoReplayRecording) {
        $witnessArgs = @([string]$gameProcess.Id,(Join-Path $session 'window'), '900', '5', 'sampled-window-evidence')
        $witnessProcess = Start-Process -FilePath $witness -ArgumentList (Quote-Arguments $witnessArgs) -PassThru `
            -RedirectStandardOutput (Join-Path $session 'window.log') -RedirectStandardError (Join-Path $session 'window-errors.log')
        $null = $witnessProcess.Handle
    }
    $deadline = [DateTime]::UtcNow.AddSeconds(900)
    while (-not $gameProcess.HasExited -and [DateTime]::UtcNow -lt $deadline) {
        if (-not $NoReplayRecording -and (Test-Path -LiteralPath (Join-Path $session 'inputs\finished.json'))) { break }
        if ($ShortDiagnostic -and -not $metadata.capture_stop_requested) {
            $reason = Get-ShortCaptureStopReason (Join-Path $session 'window\timeline.jsonl')
            if ([Console]::KeyAvailable -and [Console]::ReadKey($true).Key -eq [ConsoleKey]::Enter) { $reason = 'user-finished' }
            if ($reason -and (Test-Path -LiteralPath (Join-Path $session 'inputs\started.json'))) {
                'stop' | Set-Content -LiteralPath (Join-Path $session 'inputs\stop.txt')
                $metadata.capture_stop_requested = $reason
                Write-Host ("Finishing capture ($reason) and saving diagnostics; keep the game open...")
            }
        }
        Start-Sleep -Milliseconds 250
    }
    if ($gameProcess.HasExited) { $gameProcess.WaitForExit(); $metadata.game_exit_code = $gameProcess.ExitCode }
    $metadata.result = if ($gameProcess.HasExited) { 'game-exited' } else { 'recording-ended-game-still-running' }
    $result = 0
} catch {
    Write-Host ('Capture setup: ' + $_.Exception.Message) -ForegroundColor Yellow
    if ($session) { $_.Exception.Message | Set-Content -LiteralPath (Join-Path $session 'capture-error.txt') }
} finally {
    if ($session -and $witnessProcess) {
        $windowDirectory = Join-Path $session 'window'
        if (Test-Path -LiteralPath $windowDirectory) { 'stop' | Set-Content -LiteralPath (Join-Path $windowDirectory 'stop.txt') }
        if (-not $witnessProcess.WaitForExit(10000)) { Write-Warning 'Window collector is still running; its time limit remains active.' }
        else { $metadata.window_collector_exit_code = $witnessProcess.ExitCode }
    }
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
        if ($ShortDiagnostic -and (-not $metadata.frame_capture_present -or $metadata.frame_rows -lt 1)) {
            $metadata.result = 'presentation-evidence-missing'; $result = 1
        }
        $recordingDirectory = Join-Path $session 'inputs'
        $metadata.recording_present = Test-Path -LiteralPath (Join-Path $recordingDirectory 'started.json')
        $metadata.recording_complete = $false
        if ($metadata.recording_present) {
            & $inspect $recordingDirectory (Join-Path $session 'inspection') --allow-prefix > (Join-Path $session 'inspection.log') 2>&1
            $report = Join-Path $session 'inspection\report.json'
            if (Test-Path -LiteralPath $report) {
                $inspection = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
                $metadata.recording_complete = $inspection.complete -and $inspection.verified_prefix
                $metadata.recording_calls = $inspection.calls
                $metadata.recording_presentations = $inspection.accepted_presentations
            }
        }
        if ($metadata.input_recording -and -not $metadata.recording_complete) {
            $metadata.result = 'recording-incomplete'; $result = 1
            Write-Warning 'Capture ended without a complete journal. Verified prefix and diagnostics are preserved.'
        }
        if ($metadata.window_recording) {
            $windowReport = Join-Path $session 'window\finished.json'
            $metadata.window_complete = $false
            if (Test-Path -LiteralPath $windowReport) {
                $windowSummary = Get-Content -LiteralPath $windowReport -Raw | ConvertFrom-Json
                $metadata.window_complete = $windowSummary.complete -and $windowSummary.frames -gt 0
            }
            if (-not $metadata.window_complete) {
                $metadata.result = 'window-evidence-incomplete'; $result = 1
                Write-Warning 'Window evidence is incomplete. Renderer inputs and available diagnostics are preserved.'
            }
        }
        $metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $session 'session.json') -Encoding UTF8
        New-Item -ItemType Directory -Path $saved -Force | Out-Null
        Copy-Item -Path (Join-Path $session '*') -Destination $saved -Recurse -Force
        Write-Host ''
        Write-Host ('Capture saved: ' + $saved)
        Write-Host 'Tell Codex: Finished the capture. No upload or copy/paste needed.'
    }
}
exit $result
