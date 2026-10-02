@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
if /i "%~1"=="help" goto usage
if /i "%~1"=="--help" goto usage
if /i "%~1"=="/?" goto usage

rem Build the 32-bit game bridge and both 64-bit companions from this checkout.
set "C3X_RENDERER_ORACLE_FLAGS="
if /i "%~2"=="oracle" set "C3X_RENDERER_ORACLE_FLAGS=/DC3X_RENDERER_BENCHMARK_ORACLE"
call BUILD.bat candidate-compile no-stage
if errorlevel 1 goto fail

if defined C3X_VS_PATH goto compiler_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto fail
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%TEMP%\c3x-renderer64-vs-path.txt"
set /p C3X_VS_PATH=<"%TEMP%\c3x-renderer64-vs-path.txt"
del "%TEMP%\c3x-renderer64-vs-path.txt" >nul 2>nul
:compiler_ready
if not defined C3X_VS_PATH (
  echo Visual Studio compiler not found. 1>&2
  goto fail
)
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX renderer64_startup_probe.cpp /Fo:build\renderer64_startup_probe.obj /Fe:build\renderer64_startup_probe.exe
if errorlevel 1 goto fail
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto fail
if not exist "build\renderer64\obj" mkdir "build\renderer64\obj"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /bigobj /DC3X_HELPER_TRIAL /DC3X_RENDERER64_FRESH %C3X_RENDERER_ORACLE_FLAGS% /LD ..\sandbox\resident_scene.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:build\renderer64\obj\ /Fe:build\renderer64\C3XRenderer_x64.dll /link /DEF:c3x_renderer.def /IMPLIB:build\renderer64\C3XRenderer_x64.lib d3d11.lib d3dcompiler.lib dxgi.lib dcomp.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib ole32.lib windowscodecs.lib
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4191 helper_trial\scene_workload.cpp /Fo:build\renderer64\obj\scene_workload.obj /Fe:build\renderer64\C3XRendererHelper64.exe /link psapi.lib d3d11.lib dxgi.lib
if errorlevel 1 goto fail

if /i "%~1"=="no-stage" goto done
if not exist "..\packs\TerrainNormalized\natural_runtime\natural.bin" (
  echo The default terrain pack's Renderer64 payload is missing; refusing to stage a black-map build. 1>&2
  echo Run python3 Renderer/renderer.py prepare from the checkout first. 1>&2
  goto fail
)
rem Validate the runtime selected for evaluation before changing staged binaries.
rem The same absolute override must be supplied to the evaluation game process.
set "C3X_RENDERER_STAGE_SHADER_ROOT=%C3X_RENDERER_SHADER_SOURCE_ROOT%"
if not defined C3X_RENDERER_STAGE_SHADER_ROOT set "C3X_RENDERER_STAGE_SHADER_ROOT=%~dp0..\packs\Renderer64CutoverControl"
for %%I in ("%C3X_RENDERER_STAGE_SHADER_ROOT%") do set "C3X_RENDERER_STAGE_SHADER_ROOT=%%~fI"
if not exist "%C3X_RENDERER_STAGE_SHADER_ROOT%\Renderer\native\city_fidelity\terrain.hlsl" (
  echo The selected Renderer64 shader sources are missing; refusing to stage. 1>&2
  goto fail
)
call :require_resident_entry "source_fidelity\objects.hlsl" "VSResidentInstance"
if errorlevel 1 goto fail
call :require_resident_entry "source_fidelity\instance_caster.hlsl" "VSResidentInstance"
if errorlevel 1 goto fail
call :require_resident_entry "source_fidelity\instance_caster.hlsl" "VSResidentPlacedInstance"
if errorlevel 1 goto fail
call :require_resident_entry "environment_refresh\objects.hlsl" "VSResidentInstance"
if errorlevel 1 goto fail
call :require_resident_entry "environment_refresh\objects.hlsl" "VSResidentReflectionInstance"
if errorlevel 1 goto fail
call :require_resident_entry "city_fidelity\objects.hlsl" "VSResidentInstance"
if errorlevel 1 goto fail
call :require_resident_entry "city_fidelity\objects.hlsl" "VSResidentReflectionInstance"
if errorlevel 1 goto fail
call :require_resident_entry "city_fidelity\rigid_feature.hlsl" "VSResidentSharedFeature"
if errorlevel 1 goto fail
call :require_resident_entry "city_fidelity\rigid_feature.hlsl" "VSResidentSharedFeatureReflection"
if errorlevel 1 goto fail
call :require_resident_entry "city_fidelity\rigid_caster.hlsl" "VSResidentSharedCaster"
if errorlevel 1 goto fail
call :require_resident_entry "city_fidelity\rigid_caster.hlsl" "VSResidentPlacedCaster"
if errorlevel 1 goto fail
tasklist /fi "imagename eq Civ3Conquests.exe" /nh 2>nul | find /i "Civ3Conquests.exe" >nul
if not errorlevel 1 (
  echo Exit Civ III before staging Renderer64. 1>&2
  goto fail
)
if not exist "..\bin\renderer64" mkdir "..\bin\renderer64"
copy /y "build\candidate\C3XRenderer.dll" "..\bin\renderer64\C3XRenderer.dll" >nul
if errorlevel 1 goto fail
copy /y "build\renderer64\C3XRenderer_x64.dll" "..\bin\renderer64\C3XRenderer_x64.dll" >nul
if errorlevel 1 goto fail
copy /y "build\renderer64\C3XRendererHelper64.exe" "..\bin\renderer64\C3XRendererHelper64.exe" >nul
if errorlevel 1 goto fail
pushd ..\..
"%~dp0build\renderer64_startup_probe.exe" . "Renderer\bin\renderer64\C3XRenderer.dll"
set "C3X_STARTUP_RESULT=%errorlevel%"
popd
if not "%C3X_STARTUP_RESULT%"=="0" goto fail
echo Renderer64 bridge, renderer DLL and helper staged together.
echo Evaluation shader root: %C3X_RENDERER_STAGE_SHADER_ROOT%
echo Keep C3X_RENDERER_SHADER_SOURCE_ROOT set to this absolute path when launching the evaluation game.
:done
popd
exit /b 0
:fail
popd
exit /b 1

:require_resident_entry
if not exist "%C3X_RENDERER_STAGE_SHADER_ROOT%\Renderer\native\%~1" goto resident_entry_missing
findstr /l /c:"%~2(" "%C3X_RENDERER_STAGE_SHADER_ROOT%\Renderer\native\%~1" >nul
if errorlevel 1 goto resident_entry_missing
findstr /l /c:"uint selection:TEXCOORD1;" "%C3X_RENDERER_STAGE_SHADER_ROOT%\Renderer\native\%~1" >nul
if errorlevel 1 goto resident_entry_missing
findstr /l /c:"StructuredBuffer<ResidentPlacement> C3XResidentPlacements:register(t15);" "%C3X_RENDERER_STAGE_SHADER_ROOT%\Renderer\native\%~1" >nul
if errorlevel 1 goto resident_entry_missing
exit /b 0
:resident_entry_missing
echo Resident shader entry or input binding missing: %~1 / %~2. Refusing to stage incompatible binaries. 1>&2
echo Prepare a separate candidate with Renderer/tools/prepare_resident_submission_shaders.py, then set C3X_RENDERER_SHADER_SOURCE_ROOT to its absolute path. 1>&2
echo BUILD_RENDERER64.bat no-stage remains available for compile-only verification. 1>&2
exit /b 1

:usage
echo Usage: BUILD_RENDERER64.bat [no-stage] [oracle]
echo Builds the game bridge, x64 renderer, helper and startup probe from this checkout.
echo no-stage leaves Renderer/bin untouched and skips runtime staging checks.
echo Staging requires Civ III closed and a runtime containing the resident vertex entries.
echo Set C3X_RENDERER_SHADER_SOURCE_ROOT to an absolute prepared candidate root for evaluation.
echo The default pinned pack is checked when the override is unset; it is never modified.
popd
exit /b 0
