# Sequential Before/After runs: one input timeline, independent production cadence.
param([string]$SessionDirectory, [string]$BaselineDll, [string]$CandidateDll,
      [string]$OutputDirectory, [ValidateRange(0,2048)][int]$ReserveMiB=0, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$renderer = Split-Path $PSScriptRoot -Parent
$local = $null
$saved = $null
$receipt = @{ status='fail'; scope='paced candidate comparison; recorded native consumption points';
    timeline_speed=1; reserved_va_mib=$ReserveMiB; game_launched=$false; physical_scanout_measured=$false; runs=@() }
try {
    if (-not $SessionDirectory) {
        if ($CheckOnly) { throw 'CheckOnly requires a session directory.' }
        Add-Type -AssemblyName System.Windows.Forms
        $picker = New-Object System.Windows.Forms.FolderBrowserDialog
        $picker.Description = 'Choose the input recording to replay Before and After.'
        $picker.SelectedPath = Join-Path $renderer 'native\build\live-captures'
        if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
        $SessionDirectory = $picker.SelectedPath; $picker.Dispose()
    }
    $session = (Resolve-Path -LiteralPath $SessionDirectory).ProviderPath
    if (-not $BaselineDll) { $BaselineDll = Join-Path $session 'C3XRenderer.dll' }
    if (-not $CandidateDll) {
        if ($CheckOnly) { throw 'CheckOnly requires a candidate DLL.' }
        Add-Type -AssemblyName System.Windows.Forms
        $picker = New-Object System.Windows.Forms.OpenFileDialog
        $picker.Title = 'Choose the After renderer DLL'
        $picker.Filter = 'Renderer DLL|*.dll'
        $picker.InitialDirectory = Join-Path $renderer 'native\build\candidate'
        if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
        $CandidateDll = $picker.FileName; $picker.Dispose()
    }
    $before = (Resolve-Path -LiteralPath $BaselineDll).ProviderPath
    $after = (Resolve-Path -LiteralPath $CandidateDll).ProviderPath
    $player = Join-Path $renderer 'native\build\input-recording\replay_inputs.exe'
    $inspector = Join-Path $renderer 'native\build\input-recording\inspect_inputs.exe'
    $inputs = Join-Path $session 'inputs'
    foreach ($file in @($before,$after,$player,$inspector,(Join-Path $inputs 'started.json'),(Join-Path $inputs 'finished.json'))) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing comparison input: $file" }
    }
    $receipt.before_sha256 = (Get-FileHash -LiteralPath $before -Algorithm SHA256).Hash.ToLowerInvariant()
    $receipt.after_sha256 = (Get-FileHash -LiteralPath $after -Algorithm SHA256).Hash.ToLowerInvariant()
    $receipt.player_sha256 = (Get-FileHash -LiteralPath $player -Algorithm SHA256).Hash.ToLowerInvariant()
    $receipt.same_binary_control = $receipt.before_sha256 -eq $receipt.after_sha256
    if ($CheckOnly) { Write-Host 'PASS: comparison files found. Playback API compatibility is checked on launch. Nothing started.'; exit 0 }
    $id = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,6)
    if (-not $OutputDirectory) { $OutputDirectory = Join-Path $session ('comparisons\' + $id) }
    if (Test-Path -LiteralPath $OutputDirectory) { throw 'Comparison output directory must be new.' }
    $saved = (New-Item -ItemType Directory -Path $OutputDirectory).FullName
    $local = (New-Item -ItemType Directory -Path (Join-Path $env:TEMP ('C3XReplayCompare\' + $id))).FullName
    # Preparation is outside playback. Avoid shared-folder journal I/O in the
    # timed input feeder; keep the original recording and assets unchanged.
    $inputBytes = (Get-ChildItem -LiteralPath $inputs -File | Measure-Object -Property Length -Sum).Sum
    $volume = New-Object System.IO.DriveInfo([IO.Path]::GetPathRoot($local))
    if ($volume.AvailableFreeSpace -lt $inputBytes + 536870912) { throw 'Insufficient local disk space for comparison inputs.' }
    Write-Host 'Preparing one local input copy and freezing both renderer builds...'
    Copy-Item -LiteralPath $inputs -Destination (Join-Path $local 'inputs') -Recurse
    Copy-Item -LiteralPath $player -Destination (Join-Path $local 'replay_inputs.exe')
    Copy-Item -LiteralPath $inspector -Destination (Join-Path $local 'inspect_inputs.exe')
    Copy-Item -LiteralPath $before -Destination (Join-Path $local 'before.dll')
    Copy-Item -LiteralPath $after -Destination (Join-Path $local 'after.dll')
    foreach ($arm in @('before','after')) {
        if ((Get-FileHash -LiteralPath (Join-Path $local ($arm + '.dll')) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $receipt[($arm + '_sha256')]) {
            throw 'Renderer changed while preparing comparison.'
        }
    }
    & (Join-Path $local 'inspect_inputs.exe') (Join-Path $local 'inputs') (Join-Path $local 'inspection') |
        Set-Content -LiteralPath (Join-Path $local 'inspection.log')
    if ($LASTEXITCODE -ne 0) { throw 'Input inspection failed.' }
    $inspection = Get-Content -LiteralPath (Join-Path $local 'inspection\report.json') -Raw | ConvertFrom-Json
    if (-not $inspection.complete -or -not $inspection.verified_prefix) { throw 'Comparison requires a complete verified input recording.' }
    $receipt.input_seconds = $inspection.duration_seconds
    Write-Host 'Both runs use the same real-time mode. Only the renderer DLL changes.'
    Write-Host 'Native consumption points remain recorded; this does not yet reproduce all live game scheduling.'
    if ($receipt.same_binary_control) { Write-Host 'This is an identical-build repeatability control, not a claimed improvement.' }
    foreach ($arm in @('before','after')) {
        $label = $arm.ToUpperInvariant()
        Write-Host ("`n===== " + $label + " =====") -ForegroundColor Cyan
        Write-Host 'Original input timing, independent animation. Escape aborts the comparison.'
        Start-Sleep -Seconds 3
        $playArgs = @('--development',(Join-Path $local ($arm + '.dll')),(Join-Path $local 'inputs'),
            '--compare-candidate','--realtime',(Join-Path $local ($arm + '.jsonl')),'--label',('C3X Replay - ' + $label))
        if ($ReserveMiB) { $playArgs += @('--reserve-mib',[string]$ReserveMiB) }
        & (Join-Path $local 'replay_inputs.exe') @playArgs | Tee-Object -FilePath (Join-Path $local ($arm + '.log'))
        $code = $LASTEXITCODE
        $receipt.runs += @{ arm=$arm; exit_code=$code }
        if ($code -ne 0) { throw "$label playback failed. Its diagnostics are preserved; no successful comparison is claimed." }
    }
    $receipt.status = 'pass'
} catch {
    $receipt.error = $_.Exception.Message
    Write-Host ('Comparison: ' + $receipt.error) -ForegroundColor Yellow
} finally {
    if ($saved -and $local) {
        # The original session already owns the input journal. Save build identity
        # and measurements, not another shared-folder copy of every segment.
        foreach ($file in @('before.dll','after.dll','replay_inputs.exe','before.jsonl','after.jsonl','before.log','after.log','inspection.log')) {
            $path = Join-Path $local $file
            if (Test-Path -LiteralPath $path) { Copy-Item -LiteralPath $path -Destination $saved }
        }
        $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $saved 'comparison.json') -Encoding UTF8
        Write-Host ('Comparison saved: ' + $saved)
        # Keep failed local runs recoverable. Successful local copies are
        # disposable; remove only this invocation's owned temporary directory.
        if ($receipt.status -eq 'pass') { Remove-Item -LiteralPath $local -Recurse -Force }
    }
}
if ($receipt.status -eq 'pass') { exit 0 } else { exit 1 }
