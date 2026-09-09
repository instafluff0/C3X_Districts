@echo off
setlocal
pushd "%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "C3X_VS_PATH=%%I"
if not defined C3X_VS_PATH exit /b 2
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 2
if not defined C3X_READBACK_OUT set "C3X_READBACK_OUT=build\readback-floor"
if not exist "%C3X_READBACK_OUT%" mkdir "%C3X_READBACK_OUT%"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX benchmark_readback.cpp /Fo:%C3X_READBACK_OUT%\ /Fe:%C3X_READBACK_OUT%\benchmark_readback.exe /link /LARGEADDRESSAWARE d3d11.lib
if errorlevel 1 exit /b 1
%C3X_READBACK_OUT%\benchmark_readback.exe >%C3X_READBACK_OUT%\benchmark.log 2>&1
set "C3X_READBACK_RESULT=%errorlevel%"
>%C3X_READBACK_OUT%\completion.txt echo %C3X_READBACK_RESULT%
type %C3X_READBACK_OUT%\benchmark.log
exit /b %C3X_READBACK_RESULT%
