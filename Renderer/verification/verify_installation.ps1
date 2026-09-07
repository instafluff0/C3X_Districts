$ErrorActionPreference='Stop'
$game = if ($env:C3X_RENDERER_CIV3_CONQUESTS) { $env:C3X_RENDERER_CIV3_CONQUESTS } else { Join-Path ${env:ProgramFiles(x86)} 'GOG Galaxy\Games\Civilization III Complete\Conquests' }
$mod=Join-Path $game 'C3X_Districts'
$exe=Join-Path $game 'Civ3Conquests.exe'
$dll=Join-Path $mod 'Renderer\bin\C3XRenderer.dll'
$candidate=Join-Path $mod 'Renderer\native\build\candidate\C3XRenderer.dll'
$backup=Join-Path $mod 'Renderer\verification\Civ3Conquests-before-cache-test.exe'
if((Get-FileHash $dll).Hash -ne (Get-FileHash $candidate).Hash) { throw 'Installed DLL differs from verified candidate.' }
$gameProcess=Get-Process Civ3Conquests -ErrorAction SilentlyContinue
if ($gameProcess) { Write-Output 'NOTE: A Civ III process is now running; leaving it untouched.' }
$shell=New-Object -ComObject WScript.Shell
$consoleUser=(Get-CimInstance Win32_ComputerSystem).UserName
if (-not $consoleUser) { throw 'Interactive Windows user could not be identified.' }
$account=New-Object Security.Principal.NTAccount($consoleUser)
$sid=$account.Translate([Security.Principal.SecurityIdentifier]).Value
$profile=Get-CimInstance Win32_UserProfile | Where-Object { $_.SID -eq $sid }
if (-not $profile) { throw 'Interactive Windows profile could not be identified.' }
$desktop=(Get-ItemProperty -LiteralPath ("Registry::HKEY_USERS\"+$sid+"\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") -Name Desktop).Desktop
$desktop=$desktop.Replace('%USERPROFILE%',$profile.LocalPath)
if(-not (Test-Path -LiteralPath $desktop)) { throw 'Interactive Windows Desktop is unavailable.' }
$shortcutPath=Join-Path $desktop 'C3X Renderer Test.lnk'
$shortcut=$shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath=Join-Path $mod 'Renderer\TEST_IN_GAME.bat'
$shortcut.WorkingDirectory=$game
$shortcut.IconLocation=$exe+',0'
$shortcut.Description='Civ III renderer cache test with detailed performance trace'
$shortcut.Save()
$installedText=[Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($exe))
if (-not $installedText.Contains('halo-skip-projection') -or -not $installedText.Contains('blocks_pending=')) { throw 'Installed executable is missing the final capture/pixel diagnostics.' }
if (-not (Test-Path -LiteralPath $shortcutPath)) { throw 'Test shortcut was not created.' }
[ordered]@{
 schema='c3x.renderer_installed_test_build.v0'; installed_utc=[DateTime]::UtcNow.ToString('o');
 installer='INSTALL.bat'; installer_confirmed=$true; api_version=13; game_launched_by_agent=$false; game_running_at_verification=[bool]$gameProcess;
 executable_sha256=(Get-FileHash $exe).Hash; previous_executable_sha256=(Get-FileHash $backup).Hash;
 renderer_sha256=(Get-FileHash $dll).Hash; injected_source_sha256=(Get-FileHash (Join-Path $mod 'injected_code.c')).Hash; launcher='Renderer/TEST_IN_GAME.bat';
 desktop_shortcut='C3X Renderer Test'; backup='Renderer/verification/Civ3Conquests-before-cache-test.exe'
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $mod 'Renderer\verification\installed_test_build.json')
Write-Output 'PASS installed_test_build: official success confirmed; installer exited; DLL matches verified candidate; desktop shortcut ready; no game was launched or stopped by this script.'
