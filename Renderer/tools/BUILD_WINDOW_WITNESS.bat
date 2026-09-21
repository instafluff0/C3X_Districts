@echo off
setlocal
pushd "%~dp0"
set "C3X_WITNESS_ARCH=x64"


if defined C3X_VS_PATH goto compiler_ready
set "C3X_WITNESS_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%C3X_WITNESS_VSWHERE%" exit /b 1
for /f "usebackq tokens=*" %%i in (`"%C3X_WITNESS_VSWHERE%" -latest -prerelease -products * -property installationPath`) do set "C3X_VS_PATH=%%i"
:compiler_ready
if not defined C3X_VS_PATH exit /b 1
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat" %C3X_WITNESS_ARCH%
if errorlevel 1 exit /b 1
if not exist "..\native\build\window-witness" mkdir "..\native\build\window-witness"
cl /nologo /std:c++20 /EHsc /O2 /W4 /WX window_witness.cpp /Fo:..\native\build\window-witness\ /Fe:..\native\build\window-witness\window_witness.exe /link d3d11.lib dxgi.lib windowsapp.lib windowscodecs.lib user32.lib gdi32.lib ole32.lib uuid.lib
set "C3X_WITNESS_RESULT=%errorlevel%"
popd
exit /b %C3X_WITNESS_RESULT%
