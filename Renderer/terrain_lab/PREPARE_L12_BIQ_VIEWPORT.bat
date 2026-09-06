@echo off
setlocal
pushd "%~dp0"

set "C3X_LAB_BIQ=C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete\Conquests\Conquests\5 Mesoamerica.biq"
set "C3X_LAB_BIQ_VIEW=..\preview\out\terrain_lab\test_biq_l12_volcano_192.csv"

if exist "%C3X_LAB_BIQ%" goto biq_found
echo terrain_lab: authoritative Mesoamerica BIQ not found: %C3X_LAB_BIQ%
popd
exit /b 1

:biq_found
node ..\tools\export_biq_terrain_scene.js "%C3X_LAB_BIQ%" "%C3X_LAB_BIQ_VIEW%" --window-columns 16 --window-rows 12 --window-shape diamond --prefer-real volcano --require-preferred
set "C3X_LAB_BIQ_RESULT=%errorlevel%"
popd
exit /b %C3X_LAB_BIQ_RESULT%
