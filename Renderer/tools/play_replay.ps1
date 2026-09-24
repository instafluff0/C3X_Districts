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
    $remote = $receipt.scope -eq 'renderer64-short-diagnostic-capture'
    if ($remote) {
        $helper = Join-Path $session 'renderer64\C3XRendererHelper64.exe'
        $renderer64Dll = Join-Path $session 'renderer64\C3XRenderer_x64.dll'
        foreach ($pair in @(@($helper, $receipt.renderer64_helper_sha256), @($renderer64Dll, $receipt.renderer64_dll_sha256))) {
            if (-not (Test-Path -LiteralPath $pair[0] -PathType Leaf) -or
                $pair[1] -ne (Get-FileHash -LiteralPath $pair[0] -Algorithm SHA256).Hash.ToLowerInvariant()) {
                throw 'A pinned Renderer64 companion is missing or changed.'
            }
        }
    }
    if ($CheckOnly) { Write-Host 'PASS: pinned playback files found. Nothing launched.'; exit 0 }
    Write-Host 'Rebuilding recorded frames in a visible window. Escape closes playback.'
    Write-Host 'Playback keeps every recorded presentation and reports lag. This is not a live FPS benchmark.'
    # The call operator passes filenames directly without cmd.exe reparsing.
    $arguments = @('--development', $dll, $inputs, '--watch')
    if ($remote) { $arguments += @('--x64-primary', $helper, $renderer64Dll, '--direct-surface-trial') }
    if ($remote) {
        $metadata = Get-Content -LiteralPath (Join-Path $session 'session.json') -Raw | ConvertFrom-Json
        if (-not $metadata.game_working_directory -or -not (Test-Path -LiteralPath $metadata.game_working_directory -PathType Container)) {
            throw 'The recorded game working directory is unavailable; relative art paths cannot be replayed safely.'
        }
        Push-Location -LiteralPath $metadata.game_working_directory
        try { & $replay @arguments; $code = $LASTEXITCODE } finally { Pop-Location }
    } else { & $replay @arguments; $code = $LASTEXITCODE }
    exit $code
} catch {
    Write-Host ('Replay: ' + $_.Exception.Message) -ForegroundColor Yellow
    exit 1
}
