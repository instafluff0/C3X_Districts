$ErrorActionPreference='Stop'
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public class InspectDialog {
 public delegate bool Callback(IntPtr window, IntPtr argument);
 [DllImport("user32.dll")] public static extern bool EnumWindows(Callback callback, IntPtr argument);
 [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr window, Callback callback, IntPtr argument);
 [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr window, StringBuilder text, int maximum);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window, out uint process);
}
'@
Get-CimInstance Win32_Process -Filter "Name='temp.exe' OR Name='Civ3Conquests.exe'" | Select-Object Name,ProcessId,ParentProcessId,SessionId,ExecutablePath | ConvertTo-Json -Compress
[InspectDialog]::EnumWindows({param($window,$argument)
 [uint32]$owner=0; [void][InspectDialog]::GetWindowThreadProcessId($window,[ref]$owner)
 $process=Get-Process -Id $owner -ErrorAction SilentlyContinue
 if ($process.ProcessName -eq 'temp') {
  $title=New-Object Text.StringBuilder 512
  [void][InspectDialog]::GetWindowText($window,$title,512)
  Write-Host "INSTALLER WINDOW handle=$window pid=$owner title=$title"
  [InspectDialog]::EnumChildWindows($window,{param($child,$argument)
   $text=New-Object Text.StringBuilder 2048
   [void][InspectDialog]::GetWindowText($child,$text,2048)
   if($text.Length) { Write-Host "CHILD $child $text" }
   return $true
  },[IntPtr]::Zero) | Out-Null
 }
 return $true
},[IntPtr]::Zero) | Out-Null
