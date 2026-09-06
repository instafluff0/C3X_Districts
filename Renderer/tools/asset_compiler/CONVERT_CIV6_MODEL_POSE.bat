@echo off
setlocal

if "%~4"=="" (
  echo usage: CONVERT_CIV6_MODEL_POSE animation model.fgx output.c3pose translation-scale 1>&2
  exit /b 2
)

set "TOOLS_DIR=%~dp0"
set "OUTPUT_DIR=%TOOLS_DIR%..\..\preview\out\animation_tools"
set "CSC=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
set "EXPORTER=%OUTPUT_DIR%\export_civ6_model_pose.exe"
set "CIVNEXUS=%TOOLS_DIR%..\..\third_party\CivNexus6\bin\Release\CivNexus6.exe"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
"%CSC%" /nologo /platform:x64 /out:"%EXPORTER%" "%TOOLS_DIR%export_civ6_model_pose.cs"
if errorlevel 1 exit /b %errorlevel%

"%EXPORTER%" "%CIVNEXUS%" "%~f1" "%~f2" "%~f3" "%~4"
exit /b %errorlevel%
