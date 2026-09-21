# Watch regenerated renderer output using the session's frozen DLL and tools.
param([Parameter(Position=0)][string]$SessionDirectory, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
try {
    if (-not $SessionDirectory) {
        if ($CheckOnly) { throw 'CheckOnly requires a session directory.' }
        Add-Type -AssemblyName System.Windows.Forms
        $picker = New-Object System.Windows.Forms.FolderBrowserDialog
        $picker.Description = 'Choose a renderer input capture session (containing inputs and C3XRenderer.dll).'
        $picker.SelectedPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'native\build\live-captures'
        if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
        $SessionDirectory = $picker.SelectedPath
        $picker.Dispose()
    }
    $session = (Resolve-Path -LiteralPath $SessionDirectory).ProviderPath
    $dll = Join-Path $session 'C3XRenderer.dll'
    $replay = Join-Path $session 'replay-tools\replay_inputs.exe'
    $inputs = Join-Path $session 'inputs'
    foreach ($file in @($dll, $replay, (Join-Path $inputs 'started.json'))) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing replay input: $file. Older pixel-only recordings cannot use this player." }
    }
    $receiptPath = Join-Path $session 'capture-build.json'
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) { throw 'The session has no pinned capture-build receipt.' }
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($receipt.dll_sha256 -ne (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash.ToLowerInvariant() -or
        $receipt.replay_sha256 -ne (Get-FileHash -LiteralPath $replay -Algorithm SHA256).Hash.ToLowerInvariant()) {
        throw 'The session DLL or replay tool changed after capture.'
    }
    if ($CheckOnly) { Write-Host 'PASS: pinned playback files found. Nothing launched.'; exit 0 }
    Write-Host 'Rebuilding recorded frames in a visible window. Escape closes playback.'
    Write-Host 'Playback keeps every recorded presentation and reports lag. This is not a live FPS benchmark.'
    # The call operator passes filenames directly without cmd.exe reparsing.
    & $replay --development $dll $inputs --watch
    exit $LASTEXITCODE
} catch {
    Write-Host ('Replay: ' + $_.Exception.Message) -ForegroundColor Yellow
    exit 1
}
