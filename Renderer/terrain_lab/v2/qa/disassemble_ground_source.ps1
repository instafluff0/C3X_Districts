param([string]$Source = '')
$ErrorActionPreference = 'Stop'
# Local licensed-source inspection only; output stays in the ignored audit directory.
Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Runtime.InteropServices;
public class GroundShaderInspection {
    [DllImport("d3dcompiler_47.dll", CallingConvention=CallingConvention.StdCall)]
    private static extern int D3DDisassemble(byte[] data, UIntPtr size, uint flags, IntPtr comment, out IntPtr result);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate IntPtr BufferPointer(IntPtr self);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate UIntPtr BufferSize(IntPtr self);
    public static void Write(string path) {
        byte[] data = File.ReadAllBytes(path);
        IntPtr blob;
        int result = D3DDisassemble(data, (UIntPtr)data.Length, 0, IntPtr.Zero, out blob);
        Marshal.ThrowExceptionForHR(result);
        try {
            IntPtr table = Marshal.ReadIntPtr(blob);
            var pointer = (BufferPointer)Marshal.GetDelegateForFunctionPointer(Marshal.ReadIntPtr(table, 3*IntPtr.Size), typeof(BufferPointer));
            var size = (BufferSize)Marshal.GetDelegateForFunctionPointer(Marshal.ReadIntPtr(table, 4*IntPtr.Size), typeof(BufferSize));
            byte[] text = new byte[checked((int)size(blob).ToUInt64())];
            Marshal.Copy(pointer(blob), text, 0, text.Length);
            File.WriteAllBytes(Path.ChangeExtension(path, ".asm"), text);
        } finally { Marshal.Release(blob); }
    }
}
'@
if (!$Source) { $Source = Join-Path $PSScriptRoot '../audits/beauty/out/ground-shader-source' }
$files = @(Get-ChildItem $source -Filter '*.dxbc')
if ($files.Count -eq 0) { throw 'No extracted source shaders found' }
foreach ($file in $files) { [GroundShaderInspection]::Write($file.FullName) }
Write-Output "PASS disassembled $($files.Count) local shader containers"
