@echo off
setlocal

set "TOOLS_DIR=%~dp0"
set "OUTPUT_DIR=%TOOLS_DIR%..\..\preview\out\animation_tools"
set "CSC=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"

if not exist "%CSC%" (
  echo Missing 64-bit .NET Framework compiler: %CSC% 1>&2
  exit /b 1
)
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%CSC%" /nologo /platform:x64 /r:System.Windows.Forms.dll /out:"%OUTPUT_DIR%\import_cn6_model.exe" "%TOOLS_DIR%import_cn6_model.cs"
if errorlevel 1 exit /b %errorlevel%

echo %OUTPUT_DIR%\import_cn6_model.exe
