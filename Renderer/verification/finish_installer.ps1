$ErrorActionPreference='Stop'
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public class FinishDialog {
 [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr window,StringBuilder text,int maximum);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window,out uint process);
 [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr window,uint message,IntPtr wParam,IntPtr lParam);
}
'@
$window=[IntPtr]458872
[uint32]$owner=0
[void][FinishDialog]::GetWindowThreadProcessId($window,[ref]$owner)
$title=New-Object Text.StringBuilder 256
[void][FinishDialog]::GetWindowText($window,$title,256)
$body=New-Object Text.StringBuilder 256
[void][FinishDialog]::GetWindowText([IntPtr]65668,$body,256)
if($owner -ne 6408 -or $title.ToString() -ne 'Success' -or $body.ToString() -ne 'Mod installed successfully') { throw 'The observed installer dialog changed; nothing was acknowledged.' }
[void][FinishDialog]::PostMessage($window,0x111,[IntPtr]1,[IntPtr]::Zero)
$deadline=[DateTime]::UtcNow.AddSeconds(10)
while((Get-Process -Id 6408 -ErrorAction SilentlyContinue) -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 200 }
if(Get-Process -Id 6408 -ErrorAction SilentlyContinue) { throw 'Installer has not exited.' }
$game = if ($env:C3X_RENDERER_CIV3_CONQUESTS) { $env:C3X_RENDERER_CIV3_CONQUESTS } else { Join-Path ${env:ProgramFiles(x86)} 'GOG Galaxy\Games\Civilization III Complete\Conquests' }
$mod=Join-Path $game 'C3X_Districts'
$exe=Join-Path $game 'Civ3Conquests.exe'
$dll=Join-Path $mod 'Renderer\bin\C3XRenderer.dll'
$candidate=Join-Path $mod 'Renderer\native\build\candidate\C3XRenderer.dll'
$backup=Join-Path $mod 'Renderer\verification\Civ3Conquests-before-cache-test.exe'
if((Get-FileHash $dll).Hash -ne (Get-FileHash $candidate).Hash) { throw 'Installed DLL differs from verified candidate.' }
if(Get-Process Civ3Conquests -ErrorAction SilentlyContinue) { throw 'Civ III is unexpectedly running.' }
$shell=New-Object -ComObject WScript.Shell
$shortcut=$shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'C3X Renderer Test.lnk'))
$shortcut.TargetPath=Join-Path $mod 'Renderer\TEST_IN_GAME.bat'
$shortcut.WorkingDirectory=$game
$shortcut.IconLocation=$exe+',0'
$shortcut.Description='Civ III renderer cache test with detailed performance trace'
$shortcut.Save()
[ordered]@{
 schema='c3x.renderer_installed_test_build.v0'; installed_utc=[DateTime]::UtcNow.ToString('o');
 installer='INSTALL.bat'; installer_confirmed=$true; api_version=13; game_launched=$false;
 executable_sha256=(Get-FileHash $exe).Hash; previous_executable_sha256=(Get-FileHash $backup).Hash;
 renderer_sha256=(Get-FileHash $dll).Hash; launcher='Renderer/TEST_IN_GAME.bat';
 desktop_shortcut='C3X Renderer Test'; backup='Renderer/verification/Civ3Conquests-before-cache-test.exe'
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $mod 'Renderer\verification\installed_test_build.json')
Write-Output 'PASS installed_test_build: official success confirmed; installer exited; DLL matches verified candidate; desktop shortcut ready; Civ III remains closed.'
