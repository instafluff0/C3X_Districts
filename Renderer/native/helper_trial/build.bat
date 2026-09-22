@echo off
setlocal
pushd "%~dp0.."
if errorlevel 1 exit /b 1

if defined C3X_VS_PATH goto compiler_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto fail
set "VS_RECORD=%TEMP%\c3x-helper-trial-vs-path.txt"
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%VS_RECORD%"
set /p C3X_VS_PATH=<"%VS_RECORD%"
del "%VS_RECORD%" >nul 2>nul
:compiler_ready
if not defined C3X_VS_PATH goto fail
if not exist "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" goto fail
if not exist "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" goto fail

if not exist "build\helper_trial" mkdir "build\helper_trial"
if not exist "build\helper_trial\bin" mkdir "build\helper_trial\bin"
if not exist "build\helper_trial\obj64" mkdir "build\helper_trial\obj64"
if not exist "build\helper_trial\obj32" mkdir "build\helper_trial\obj32"

setlocal
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX helper_trial\producer.cpp /Fo:build\helper_trial\obj64\ /Fe:build\helper_trial\bin\producer.exe /link d3d11.lib dxgi.lib psapi.lib
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX helper_trial\ipc_event.cpp /Fo:build\helper_trial\obj64\ipc_event.obj /Fe:build\helper_trial\bin\ipc_x64.exe
if errorlevel 1 goto fail
endlocal

setlocal
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX helper_trial\consumer.cpp /Fo:build\helper_trial\obj32\ /Fe:build\helper_trial\bin\consumer.exe /link /LARGEADDRESSAWARE d3d11.lib d3dcompiler.lib dxgi.lib dcomp.lib user32.lib gdi32.lib psapi.lib
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX helper_trial\ipc_event.cpp /Fo:build\helper_trial\obj32\ipc_event.obj /Fe:build\helper_trial\bin\ipc_x86.exe /link /LARGEADDRESSAWARE
if errorlevel 1 goto fail
endlocal

popd
exit /b 0
:fail
popd
exit /b 1
