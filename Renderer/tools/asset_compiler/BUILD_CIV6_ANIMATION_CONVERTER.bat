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

"%CSC%" /nologo /platform:x64 /out:"%OUTPUT_DIR%\export_civ6_animation.exe" "%TOOLS_DIR%export_civ6_animation.cs"
if errorlevel 1 exit /b %errorlevel%

echo %OUTPUT_DIR%\export_civ6_animation.exe
