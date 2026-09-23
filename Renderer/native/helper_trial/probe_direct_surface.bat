@echo off
setlocal
pushd "%~dp0.."
if errorlevel 1 exit /b 1
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto fail
for /f "usebackq delims=" %%P in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "C3X_VS_PATH=%%P"
if not defined C3X_VS_PATH goto fail
if not exist "build\helper_trial\bin" mkdir "build\helper_trial\bin"
if not exist "build\helper_trial\obj64" mkdir "build\helper_trial\obj64"
if not exist "build\helper_trial\obj32" mkdir "build\helper_trial\obj32"
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX helper_trial\surface_producer.cpp /Fo:build\helper_trial\obj64\surface_producer.obj /Fe:build\helper_trial\bin\surface_producer.exe /link d3d11.lib dxgi.lib
if errorlevel 1 goto fail
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX helper_trial\surface_capability_probe.cpp /Fo:build\helper_trial\obj32\surface_capability_probe.obj /Fe:build\helper_trial\bin\surface_capability_probe_x86.exe /link d3d11.lib dcomp.lib dxgi.lib
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX helper_trial\surface_consumer.cpp /Fo:build\helper_trial\obj32\surface_consumer.obj /Fe:build\helper_trial\bin\surface_consumer.exe /link /LARGEADDRESSAWARE d3d11.lib dcomp.lib dxgi.lib dwmapi.lib user32.lib gdi32.lib
if errorlevel 1 goto fail
build\helper_trial\bin\surface_capability_probe_x86.exe
if errorlevel 1 goto fail
build\helper_trial\bin\surface_consumer.exe %*
if errorlevel 1 goto fail
popd
exit /b 0
:fail
popd
exit /b 1
