@echo off
setlocal

set "TOOLS_DIR=%~dp0"
set "SOURCE=Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets\Base\Platforms\Windows\BLPs\SHARED_DATA"
if not exist "%SOURCE%" set "SOURCE=\\Mac\Home\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets\Base\Platforms\Windows\BLPs\SHARED_DATA"
set "PACK=%TOOLS_DIR%..\..\packs\UnitEarlyLab"
set "CONVERTER=%TOOLS_DIR%..\..\preview\out\animation_tools\export_civ6_animation.exe"
set "CIVNEXUS=%TOOLS_DIR%..\..\third_party\CivNexus6\bin\Release\CivNexus6.exe"

call "%TOOLS_DIR%BUILD_CIV6_ANIMATION_CONVERTER.bat"
if errorlevel 1 exit /b %errorlevel%

call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" warrior idle ANIMATION_Warrior_IdleB
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" warrior move ANIMATION_UnitMedium_Run_SwordAndShieldA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" warrior attack ANIMATION_Warrior_AttackMeleeB
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" warrior death ANIMATION_Warrior_DeathMeleeA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" warrior fidget ANIMATION_Warrior_IdleD
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" warrior fortify ANIMATION_Warrior_BraceA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" warrior defend ANIMATION_Warrior_BlockA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" scout idle ANIMATION_Scout_IdleA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" scout move ANIMATION_Scout_Run
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" scout attack ANIMATION_Scout_AttackD
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" scout death ANIMATION_Scout_DeathA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" scout fidget ANIMATION_Scout_IdleB
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" scout fortify ANIMATION_Scout_FortifyA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" scout defend ANIMATION_Scout_DodgeBack
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" settler idle ANIMATION_Scout_BreathingB
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" settler move ANIMATION_Scout_Jog
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" settler fidget ANIMATION_Scout_IdleD
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" settler fortify ANIMATION_Scout_FortifyB
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" settler capture ANIMATION_SettlerLeader_CaptureB
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker idle ANIMATION_Builder_IdleRest01_2H
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker move ANIMATION_Builder_RunFwdA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker fidget ANIMATION_Builder_IdleRest06_2H
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker fortify ANIMATION_Builder_RunStop1
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker capture ANIMATION_Builder_Captured01
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" settler build ANIMATION_SettlerLeader_CITYA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker road ANIMATION_Builder_BuildAction01_Shovel
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker mine ANIMATION_Builder_BuildAction02_2H
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker irrigate ANIMATION_Builder_BuildAction01_Shovel
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker fortress ANIMATION_Builder_BuildAction02_2H
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker forest ANIMATION_Builder_BuildAction03_Axe
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker jungle ANIMATION_Builder_BuildAction03_Axe
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker plant ANIMATION_Builder_BuildAction01_Shovel
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" worker build ANIMATION_Builder_BuildAction02_2H
if errorlevel 1 exit /b %errorlevel%
echo Converted 33 early-unit animation clips.
exit /b 0
