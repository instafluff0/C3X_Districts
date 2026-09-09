@echo off
setlocal
pushd "%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 1
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "C3X_LAB_VS=%%I"
if not defined C3X_LAB_VS exit /b 1
call "%C3X_LAB_VS%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 1
if not exist ".cache" mkdir ".cache"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX native_preview.cpp /Fo:.cache\native_preview.obj /Fe:.cache\native_preview.exe /link /LARGEADDRESSAWARE gdi32.lib user32.lib
exit /b %errorlevel%
