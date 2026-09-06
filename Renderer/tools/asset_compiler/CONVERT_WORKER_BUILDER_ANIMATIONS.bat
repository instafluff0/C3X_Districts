@echo off
setlocal

set "TOOLS_DIR=%~dp0"
set "SOURCE=Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets\Base\Platforms\Windows\BLPs\SHARED_DATA"
if not exist "%SOURCE%" set "SOURCE=\\Mac\Home\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets\Base\Platforms\Windows\BLPs\SHARED_DATA"
set "PACK=%TOOLS_DIR%..\..\packs\WorkerBuilderLab"
set "CONVERTER=%TOOLS_DIR%..\..\preview\out\animation_tools\export_civ6_animation.exe"
set "CIVNEXUS=%TOOLS_DIR%..\..\third_party\CivNexus6\bin\Release\CivNexus6.exe"

call "%TOOLS_DIR%BUILD_CIV6_ANIMATION_CONVERTER.bat"
if errorlevel 1 exit /b %errorlevel%
if not exist "%PACK%\animations\unit\worker" mkdir "%PACK%\animations\unit\worker"

call :convert work_ground ANIMATION_Builder_BuildAction01_Shovel
if errorlevel 1 exit /b %errorlevel%
call :convert work_heavy ANIMATION_Builder_BuildAction02_2H
if errorlevel 1 exit /b %errorlevel%
call :convert work_cut ANIMATION_Builder_BuildAction03_Axe
if errorlevel 1 exit /b %errorlevel%
call :convert work_repair_1 ANIMATION_Builder_RepairAction01_1H
if errorlevel 1 exit /b %errorlevel%
call :convert work_repair_2 ANIMATION_Builder_RepairAction02_1H
if errorlevel 1 exit /b %errorlevel%
call :convert work_repair_3 ANIMATION_Builder_RepairAction03_1H
if errorlevel 1 exit /b %errorlevel%
call :convert work_repair_4 ANIMATION_Builder_RepairAction04_1H
if errorlevel 1 exit /b %errorlevel%
call :convert captured_1 ANIMATION_Builder_Captured01
if errorlevel 1 exit /b %errorlevel%
call :convert captured_2 ANIMATION_Builder_Captured02
if errorlevel 1 exit /b %errorlevel%
call :convert captured_3 ANIMATION_Builder_Captured03
if errorlevel 1 exit /b %errorlevel%
call :convert captured_4 ANIMATION_Builder_Captured04
if errorlevel 1 exit /b %errorlevel%

echo Converted 11 Builder specialty animation clips.
exit /b 0

:convert
if not exist "%SOURCE%\%~2" (
  echo Missing Builder animation source: %SOURCE%\%~2 1>&2
  exit /b 1
)
"%CONVERTER%" "%CIVNEXUS%" "%SOURCE%\%~2" "%PACK%\animations\unit\worker\%~1.c3anim" 0.01
if errorlevel 1 exit /b %errorlevel%
exit /b 0
