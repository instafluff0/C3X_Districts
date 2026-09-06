@echo off
setlocal
pushd "%~dp0"

set "C3X_LAB_BIQ=C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete\Conquests\Conquests\Intro1 Ancient Treasures.biq"
set "C3X_LAB_BIQ_VIEW=..\preview\out\terrain_lab\test_biq_l13_rivers_192.csv"

if exist "%C3X_LAB_BIQ%" goto biq_found
echo terrain_lab: authoritative Ancient Treasures BIQ not found: %C3X_LAB_BIQ%
popd
exit /b 1

:biq_found
node ..\tools\export_biq_terrain_scene.js "%C3X_LAB_BIQ%" "%C3X_LAB_BIQ_VIEW%" --window-columns 16 --window-rows 12 --window-shape diamond --prefer-real desert --prefer-real plains --prefer-real grassland --prefer-real floodplain --prefer-real hills --prefer-real mountain --prefer-real forest --prefer-real jungle --prefer-real marsh --prefer-real volcano --prefer-real coast --prefer-real sea --prefer-real ocean --require-all-preferred --prefer-river --require-river --prefer-wrap --require-wrap
set "C3X_LAB_BIQ_RESULT=%errorlevel%"
popd
exit /b %C3X_LAB_BIQ_RESULT%
