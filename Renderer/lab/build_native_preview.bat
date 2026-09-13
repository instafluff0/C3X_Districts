@echo off
setlocal
pushd "%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 1
set "C3X_LAB_VS_RECORD=%TEMP%\c3x-renderer-lab-vs-path.txt"
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%C3X_LAB_VS_RECORD%"
set /p C3X_LAB_VS=<"%C3X_LAB_VS_RECORD%"
if not defined C3X_LAB_VS "%VSWHERE%" -all -products * -property installationPath >"%C3X_LAB_VS_RECORD%"
if not defined C3X_LAB_VS set /p C3X_LAB_VS=<"%C3X_LAB_VS_RECORD%"
del "%C3X_LAB_VS_RECORD%" >nul 2>nul
if not defined C3X_LAB_VS exit /b 1
if not exist "%C3X_LAB_VS%\VC\Auxiliary\Build\vcvars32.bat" exit /b 1
call "%C3X_LAB_VS%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 1
if not exist ".cache" mkdir ".cache"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX native_preview.cpp /Fo:.cache\native_preview.obj /Fe:.cache\native_preview.exe /link /LARGEADDRESSAWARE gdi32.lib user32.lib
exit /b %errorlevel%
