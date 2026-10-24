@echo off
setlocal
rem Builds sound_service_probe.cpp (x86, to read the game's thread contexts) into native\build\sound-probe.
pushd "%~dp0"
if errorlevel 1 exit /b 1
if defined C3X_VS_PATH goto compiler_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto fail
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "C3X_VS_PATH=%%i"
:compiler_ready
if not defined C3X_VS_PATH goto fail
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 goto fail
if not exist "..\native\build\sound-probe" mkdir "..\native\build\sound-probe"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX sound_service_probe.cpp /Fo:..\native\build\sound-probe\ /Fe:..\native\build\sound-probe\sound_service_probe.exe
if errorlevel 1 goto fail
popd
exit /b 0
:fail
popd
exit /b 1
