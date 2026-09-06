@echo off
setlocal

if "%~2"=="" (
  echo usage: IMPORT_CN6_MODEL input.cn6 output.fgx 1>&2
  exit /b 2
)

set "TOOLS_DIR=%~dp0"
set "IMPORTER=%TOOLS_DIR%..\..\preview\out\animation_tools\import_cn6_model.exe"
set "CIVNEXUS=%TOOLS_DIR%..\..\third_party\CivNexus6\bin\Release\CivNexus6.exe"

call "%TOOLS_DIR%BUILD_CN6_MODEL_IMPORTER.bat"
if errorlevel 1 exit /b %errorlevel%

"%IMPORTER%" "%CIVNEXUS%" "%~f1" "%~f2"
exit /b %errorlevel%
