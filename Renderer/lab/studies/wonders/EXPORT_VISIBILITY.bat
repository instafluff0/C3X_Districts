@echo off
setlocal

if "%~2"=="" (
  echo usage: EXPORT_VISIBILITY raw-granny.fgx output.txt 1>&2
  exit /b 2
)

set "STUDY_DIR=%~dp0"
set "OUTPUT_DIR=%STUDY_DIR%..\..\..\preview\out\animation_tools"
set "CSC=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
set "CIVNEXUS=%STUDY_DIR%..\..\..\third_party\CivNexus6\bin\Release\CivNexus6.exe"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
"%CSC%" /nologo /platform:x64 /unsafe /out:"%OUTPUT_DIR%\export_wonder_visibility.exe" "%STUDY_DIR%export_visibility.cs"
if errorlevel 1 exit /b %errorlevel%

"%OUTPUT_DIR%\export_wonder_visibility.exe" "%CIVNEXUS%" "%~f1" "%~f2"
exit /b %errorlevel%
