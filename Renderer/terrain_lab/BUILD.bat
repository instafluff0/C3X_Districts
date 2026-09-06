@echo off
setlocal
pushd "%~dp0"

set "C3X_LAB_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%C3X_LAB_VSWHERE%" (
  echo Visual Studio Installer's vswhere.exe was not found. 1>&2
  popd
  exit /b 1
)

for /f "usebackq tokens=*" %%I in (`"%C3X_LAB_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "C3X_LAB_VS_PATH=%%I"
if not defined C3X_LAB_VS_PATH (
  echo A Visual Studio installation with the x86 C++ toolchain was not found. 1>&2
  popd
  exit /b 1
)

call "%C3X_LAB_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 (
  popd
  exit /b 1
)

if not exist "build" mkdir "build"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX terrain_lab.cpp ..\native\environment_runtime.cpp /Fo:build\ /Fe:build\terrain_lab.exe /link d3d11.lib d3dcompiler.lib dxgi.lib
set "C3X_LAB_BUILD_RESULT=%errorlevel%"
popd
exit /b %C3X_LAB_BUILD_RESULT%
