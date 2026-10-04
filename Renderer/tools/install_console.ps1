# Explicit installation only: same C3X_INSTALL path, with no dialog to dismiss.
# Run elevated from the shared checkout beneath the installed Conquests folder.
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
# Windows may retain an exited crash record; only live processes own a session.
if (Get-Process Civ3Conquests -ErrorAction SilentlyContinue | Where-Object { -not $_.HasExited }) { throw 'Close Civ III before installing.' }
if ($root -notmatch '\\Conquests\\') { throw 'Use the shared checkout link beneath the installed Conquests directory.' }
$build = Join-Path $root 'Renderer\native\build\console-install'
New-Item -ItemType Directory -Force -Path $build | Out-Null
$exe = Join-Path $build ('installer-' + [guid]::NewGuid().ToString('N') + '.exe')
$previousPath = $env:PATH
Push-Location $root
try {
    & '.\tcc\tcc.exe' -m32 '-Wl,-nostdlib' -g -lmsvcrt -luser32 -lkernel32 -DC3X_INSTALL 'Renderer\tools\install_console.c' -o $exe
    if ($LASTEXITCODE -ne 0) { throw ('Installer compilation failed: ' + $LASTEXITCODE) }
    $env:PATH = (Join-Path $root 'tcc') + ';' + $previousPath
    $installer = Start-Process -FilePath $exe -WorkingDirectory $root -NoNewWindow -Wait -PassThru
    if ($installer.ExitCode -ne 0) { throw ('C3X installation failed: ' + $installer.ExitCode) }
    Write-Output 'CONSOLE_INSTALL_OK'
} finally {
    Pop-Location
    $env:PATH = $previousPath
}
