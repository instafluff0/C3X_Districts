@echo off
setlocal

if "%~2"=="" (
  echo usage: CONVERT_CIV6_ANIMATION input output.c3anim [translation-scale] 1>&2
  exit /b 2
)

set "TOOLS_DIR=%~dp0"
set "CONVERTER=%TOOLS_DIR%..\..\preview\out\animation_tools\export_civ6_animation.exe"
set "CIVNEXUS=%TOOLS_DIR%..\..\third_party\CivNexus6\bin\Release\CivNexus6.exe"

call "%TOOLS_DIR%BUILD_CIV6_ANIMATION_CONVERTER.bat"
if errorlevel 1 exit /b %errorlevel%

"%CONVERTER%" "%CIVNEXUS%" "%~f1" "%~f2" %3
exit /b %errorlevel%
