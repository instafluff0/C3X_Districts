$ErrorActionPreference='Stop'
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public class AcknowledgeInstaller {
 [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr window,StringBuilder text,int maximum);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window,out uint process);
 [DllImport("user32.dll", SetLastError=true)] public static extern IntPtr SendMessageTimeout(IntPtr window,uint message,IntPtr wParam,IntPtr lParam,uint flags,uint timeout,out IntPtr result);
}
'@
$text=New-Object Text.StringBuilder 256
[void][AcknowledgeInstaller]::GetWindowText([IntPtr]65668,$text,256)
[uint32]$owner=0
[void][AcknowledgeInstaller]::GetWindowThreadProcessId([IntPtr]458872,[ref]$owner)
if($owner -ne 6408 -or $text.ToString() -ne 'Mod installed successfully') { throw 'Success dialog changed.' }
$result=[IntPtr]::Zero
$sent=[AcknowledgeInstaller]::SendMessageTimeout([IntPtr]262270,0xF5,[IntPtr]::Zero,[IntPtr]::Zero,2,1000,[ref]$result)
Write-Output ('Button acknowledgement result='+$sent+' error='+[Runtime.InteropServices.Marshal]::GetLastWin32Error())
Start-Sleep -Milliseconds 500
Get-Process -Id 6408 -ErrorAction SilentlyContinue | Select-Object ProcessName,Id,Responding
