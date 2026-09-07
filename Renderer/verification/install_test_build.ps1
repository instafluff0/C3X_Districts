$ErrorActionPreference = 'Stop'
$game = if ($env:C3X_RENDERER_CIV3_CONQUESTS) { $env:C3X_RENDERER_CIV3_CONQUESTS } else { Join-Path ${env:ProgramFiles(x86)} 'GOG Galaxy\Games\Civilization III Complete\Conquests' }
$mod = Join-Path $game 'C3X_Districts'
if (Get-Process Civ3Conquests -ErrorAction SilentlyContinue) { throw 'Civ III is running; the installed executable was not changed.' }
$exe = Join-Path $game 'Civ3Conquests.exe'
$backup = Join-Path $mod 'Renderer\verification\Civ3Conquests-before-cache-test.exe'
Copy-Item -LiteralPath $exe -Destination $backup -Force
$before = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public class InstallDialog {
 public delegate bool Callback(IntPtr window, IntPtr argument);
 [DllImport("user32.dll")] public static extern bool EnumWindows(Callback callback, IntPtr argument);
 [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr window, Callback callback, IntPtr argument);
 [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr window, StringBuilder text, int maximum);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window, out uint process);
 [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr window, uint message, IntPtr wParam, IntPtr lParam);
}
'@
$script:confirmed = $false
$installer = Start-Process -FilePath $env:ComSpec -ArgumentList '/d /c call INSTALL.bat' -WorkingDirectory $mod -PassThru
$deadline = [DateTime]::UtcNow.AddSeconds(90)
while (-not $installer.HasExited -and [DateTime]::UtcNow -lt $deadline) {
 $owned = @(Get-CimInstance Win32_Process -Filter "Name='temp.exe'" | Where-Object { $_.ParentProcessId -eq $installer.Id } | ForEach-Object { [uint32]$_.ProcessId })
 if ($owned.Count) {
  [InstallDialog]::EnumWindows({param($window,$argument)
   [uint32]$owner=0
   [void][InstallDialog]::GetWindowThreadProcessId($window,[ref]$owner)
   if ($owned -contains $owner) {
    $title=New-Object Text.StringBuilder 256
    [void][InstallDialog]::GetWindowText($window,$title,256)
    if ($title.ToString() -eq 'Success') {
     $script:correctText=$false
     [InstallDialog]::EnumChildWindows($window,{param($child,$argument)
      $text=New-Object Text.StringBuilder 512
      [void][InstallDialog]::GetWindowText($child,$text,512)
      if ($text.ToString() -eq 'Mod installed successfully') { $script:correctText=$true }
      return $true
     },[IntPtr]::Zero) | Out-Null
     if ($script:correctText) {
      $script:confirmed=$true
      [void][InstallDialog]::PostMessage($window,0x111,[IntPtr]1,[IntPtr]::Zero)
     }
    }
   }
   return $true
  },[IntPtr]::Zero) | Out-Null
 }
 Start-Sleep -Milliseconds 200
 $installer.Refresh()
}
if (-not $installer.HasExited) { throw 'Installer still active after 90 seconds; inspect its dialog. No game process was terminated.' }
if (-not $script:confirmed -or $installer.ExitCode -ne 0) { throw 'The official installer did not confirm success.' }
$after = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
$dll = Join-Path $mod 'Renderer\bin\C3XRenderer.dll'
$candidate = Join-Path $mod 'Renderer\native\build\candidate\C3XRenderer.dll'
if ((Get-FileHash $dll).Hash -ne (Get-FileHash $candidate).Hash) { throw 'Installed renderer DLL differs from verified candidate.' }
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'C3X Renderer Test.lnk'))
$shortcut.TargetPath = Join-Path $mod 'Renderer\TEST_IN_GAME.bat'
$shortcut.WorkingDirectory = $game
$shortcut.IconLocation = $exe + ',0'
$shortcut.Description = 'Civ III renderer cache test with bounded performance trace'
$shortcut.Save()
[ordered]@{
 schema='c3x.renderer_installed_test_build.v0'; installed_utc=[DateTime]::UtcNow.ToString('o');
 installer='INSTALL.bat'; installer_confirmed=$script:confirmed; api_version=13;
 executable_sha256=$after; previous_executable_sha256=$before;
 renderer_sha256=(Get-FileHash $dll).Hash; launcher='Renderer/TEST_IN_GAME.bat';
 desktop_shortcut='C3X Renderer Test'; backup='Renderer/verification/Civ3Conquests-before-cache-test.exe'
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $mod 'Renderer\verification\installed_test_build.json')
Write-Output 'PASS installed_test_build: official installer confirmed success; verified DLL matches; desktop launcher is ready.'
